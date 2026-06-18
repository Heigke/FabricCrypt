#!/usr/bin/env python3
"""H1 VCEK permutation sweep on Qwen3-0.6B — MMLU-STEM-500 eval.

Pre-registration: research_plan/H1_PREREG_2026-06-09.md
Grid: |P| in {8, 32, 128, 512, 2048} x L in {6, 12, 20} x seed in {0..4}
Per cell: derive permutation P from KDF(TPM_EK || cell_info), inject into
post-MLP residual stream of layer L, evaluate MMLU-STEM 500 prompts 5-shot.

THIS IS A BOOTSTRAP. The TPM_EK extraction itself is wired to use
TPM_EK = SHA256(host_id || "vcek-mock") for development.
Before any production claim, swap _read_tpm_ek_pub() for a real tpm2_quote
read from /dev/tpm0 — and the script will refuse to run if the dev path
returns the mock value on a host that has /dev/tpm0 present.

Usage:
  HSA_OVERRIDE_GFX_VERSION=11.0.0 python h1_qwen_sweep.py \
      --layers 6,12,20 --p-sizes 8,32,128,512,2048 --seeds 0,1,2,3,4 \
      --eval-only      # to just eval, not train
      --mmlu-subset stem500   # selects the 500-prompt STEM subset

The training/eval body is intentionally minimal — the heavy lift is
deferred until Eric OKs the 40 GPU-h zgx commitment.
"""
import argparse
import hashlib
import hmac
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HOST = socket.gethostname()
ROOT = Path(__file__).resolve().parents[3]
OUT_DIR = ROOT / "results/IDENTITY_H1_2026-06-09"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def _read_tpm_ek_identity() -> bytes:
    """Read TPM endorsement-key identity bytes. No mocks.

    Strategy: tpm2_readpublic at the canonical EK persistent handle 0x81010001.
    The 'name:' field is a SHA-256 of the public area and is what we KDF on —
    no need to also extract the raw modulus. If the handle is empty, we try
    tpm2_createek to provision once, then re-read.

    Hard refusal: if no readable EK on this host, raise RuntimeError. We do
    NOT fall back to anything synthetic.
    """
    def _query_handle():
        r = subprocess.run(["tpm2_readpublic", "-c", "0x81010001"],
                           capture_output=True, text=True, timeout=10)
        if r.returncode != 0:
            return None
        for line in r.stdout.splitlines():
            line = line.strip()
            if line.startswith("name:"):
                hex_name = line.split(":", 1)[1].strip()
                return bytes.fromhex(hex_name)
        return None

    name = _query_handle()
    if name is None:
        # Try to provision once.
        subprocess.run(["tpm2_createek", "-c", "0x81010001", "-G", "rsa", "-u", "/tmp/ek.pub"],
                       capture_output=True, text=True, timeout=20)
        name = _query_handle()
    if name is None:
        raise RuntimeError(
            "No readable TPM EK on this host. Either /dev/tpm0 is missing or "
            "tpm2_readpublic at 0x81010001 returned nothing. Refuse to proceed "
            "with H1 — H1 is crypto-binding and requires a real EK."
        )
    return name


def derive_keystream(salt: bytes, ikm: bytes, info: bytes, length: int) -> bytes:
    """SHAKE256-based extract-then-expand (KDF) — unlimited output length.
    Equivalent in spirit to HKDF but lifts the 255*HashLen ceiling that
    would prevent |P|=2048 expansion.
    """
    h = hashlib.shake_256()
    h.update(b"AMD-IDENTITY-H1-2026-06-09::")
    h.update(salt)
    h.update(b"||IKM||")
    h.update(ikm)
    h.update(b"||INFO||")
    h.update(info)
    return h.digest(length)


def derive_permutation(tpm_ek: bytes, layer: int, p_size: int, seed: int, hidden: int = 2048) -> np.ndarray:
    """Deterministic permutation of `hidden` dims with `p_size` block permuted.

    Returns an int64 array of length `hidden` representing dst_index_for_src.
    Only the first `p_size` indices are non-identity (block permutation).
    Choice of which block of `p_size` is permuted is also KDF-derived.
    """
    info = f"H1/vcek/qwen-0.6B/L={layer}/|P|={p_size}/seed={seed}".encode()
    okm = derive_keystream(b"AMD-IDENTITY-H1-2026-06-09", tpm_ek, info, length=4 * p_size + 4)
    # which block (start offset)
    start = int.from_bytes(okm[:4], "big") % (hidden - p_size + 1)
    keys = np.frombuffer(okm[4:], dtype=">u4")
    order = np.argsort(keys)
    perm = np.arange(hidden, dtype=np.int64)
    perm[start:start + p_size] = start + order
    return perm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", default="6,12,20")
    ap.add_argument("--p-sizes", default="8,32,128,512,2048")
    ap.add_argument("--seeds", default="0,1,2,3,4")
    ap.add_argument("--dry-run", action="store_true",
                    help="Just derive permutations + save manifest; do NOT touch the model.")
    ap.add_argument("--gpu-h-budget", type=float, default=40.0)
    args = ap.parse_args()

    tpm_ek = _read_tpm_ek_identity()
    print(f"[info] host={HOST} TPM_EK hash={hashlib.sha256(tpm_ek).hexdigest()[:16]}")
    layers   = [int(x) for x in args.layers.split(",")]
    p_sizes  = [int(x) for x in args.p_sizes.split(",")]
    seeds    = [int(x) for x in args.seeds.split(",")]
    grid = [(L, P, s) for L in layers for P in p_sizes for s in seeds]
    print(f"[info] grid = {len(grid)} cells")

    manifest = {
        "preregistration": "research_plan/H1_PREREG_2026-06-09.md",
        "host": HOST,
        "ek_sha16": hashlib.sha256(tpm_ek).hexdigest()[:16],
        "cells": [],
    }
    for L, P, s in grid:
        perm = derive_permutation(tpm_ek, L, P, s)
        cell_id = f"L{L}_P{P}_s{s}"
        np.save(OUT_DIR / f"perm_{cell_id}_{HOST}.npy", perm)
        manifest["cells"].append({
            "id": cell_id, "L": L, "P": P, "seed": s,
            "perm_sha16": hashlib.sha256(perm.tobytes()).hexdigest()[:16],
        })
    (OUT_DIR / f"manifest_{HOST}.json").write_text(json.dumps(manifest, indent=2))
    print(f"[ok] wrote {len(grid)} permutations + manifest to {OUT_DIR}")

    if args.dry_run:
        print("[dry-run] stopping before model load. To execute the sweep:")
        print("          1) confirm zgx GPU-h budget with Eric (~40h)")
        print("          2) rerun without --dry-run on zgx")
        return

    # --- guarded model load (only on zgx; refuse to launch elsewhere) ---
    if HOST not in {"zgx"}:
        print(f"[refuse] H1 training is zgx-only; current host={HOST}. Use --dry-run.")
        return

    print(f"[info] would launch Qwen3-0.6B training over {len(grid)} cells "
          f"(budget {args.gpu_h_budget} GPU-h). Implementation pending Eric OK.")


if __name__ == "__main__":
    main()
