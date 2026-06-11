#!/usr/bin/env python3
"""01b_puf_signature_thermal.py — Phase 1b PUF signature with thermal clamp.

For each of N regimes (cold/idle/warm), clamps temperature using
_thermal_clamp.clamp_regime, then runs PUF reps, then writes
raw_<regime>.npz + signature_thermal.json (one combined file per device).

Output schema (signature_thermal.json):
{
  "device_name": "...",
  "phase": "1b",
  "timestamp": "...",
  "reps_per_regime": int,
  "dvfs_writable": bool,
  "regimes": {
     "cold": { ..., "achieved_temp_C", "in_band", "temp_during_run_mean",
                "temp_during_run_max", "sig_bits", "process_stat", ...,
                "raw_npz": str },
     "idle": {...},
     "warm": {...}
  }
}
"""
from __future__ import annotations
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent.parent
sys.path.insert(0, str(HERE))

from _thermal_clamp import (clamp_regime, read_temp, dvfs_writable, set_dvfs,
                            NEVER_EXCEED_C)
# reuse the kernel runner + parser + signature compute from 01_puf_signature
import importlib.util
spec = importlib.util.spec_from_file_location(
    "puf01", HERE / "01_puf_signature.py")
puf01 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(puf01)

PUF_BIN = HERE / "puf_kernel"
PAUSE_AT_C = 75.0


def run_regime(regime: str, reps: int, batch: int, results_root: Path,
               device_name: str) -> dict:
    print(f"\n=== regime={regime} ===", flush=True)
    clamp_res = clamp_regime(regime)
    print(f"[1b] clamp -> {clamp_res}", flush=True)

    all_payloads = []
    done = 0
    bin_idx = 0
    temps_during_run = []
    while done < reps:
        t = read_temp()
        temps_during_run.append(t)
        if t >= NEVER_EXCEED_C:
            print(f"[1b] ABORT temp={t:.1f}C >= {NEVER_EXCEED_C}", flush=True)
            break
        if t >= PAUSE_AT_C:
            print(f"[1b] pause at {t:.1f}C, cooling...", flush=True)
            # brief cool to below 60
            t0 = time.time()
            while time.time() - t0 < 60.0 and read_temp() > 60.0:
                time.sleep(2.0)
            # if warm regime, re-clamp briefly
            if regime == "warm":
                clamp_regime("warm")
        this_batch = min(batch, reps - done)
        tmp_bin = results_root / f"_raw_{regime}_b{bin_idx:04d}.bin"
        puf01.run_kernel(this_batch, tmp_bin)
        payload = puf01.parse_bin(tmp_bin)
        all_payloads.append(payload)
        tmp_bin.unlink(missing_ok=True)
        done += this_batch
        bin_idx += 1
        if bin_idx % 5 == 0:
            print(f"  reps={done}/{reps} temp={read_temp():.1f}C", flush=True)

    if not all_payloads:
        return {"regime": regime, "error": "no_payloads",
                "clamp": clamp_res}

    samples = np.concatenate(all_payloads, axis=0)
    sig, raw = puf01.compute_signature(samples, f"{device_name}_{regime}")

    npz_path = results_root / f"raw_{regime}.npz"
    np.savez_compressed(
        npz_path,
        **{k: np.asarray(v) for k, v in raw.items()
           if isinstance(v, np.ndarray)})
    sig["raw_npz"] = str(npz_path)
    sig["regime"] = regime
    sig["clamp"] = clamp_res
    sig["temp_during_run_mean"] = float(np.mean(temps_during_run))
    sig["temp_during_run_max"] = float(np.max(temps_during_run))
    sig["temp_during_run_min"] = float(np.min(temps_during_run))
    sig["temp_post_run"] = float(read_temp())
    print(f"[1b] regime={regime} bit_stab_mean={sig['bit_stability_mean']:.3f} "
          f"temp_mean={sig['temp_during_run_mean']:.1f}C", flush=True)
    return sig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device-name", required=True)
    ap.add_argument("--reps", type=int, default=500)
    ap.add_argument("--batch", type=int, default=10)
    ap.add_argument("--regimes", default="cold,idle,warm")
    ap.add_argument("--results-root", default=None)
    args = ap.parse_args()

    results_root = Path(args.results_root) if args.results_root else (
        REPO_ROOT / "results" / "IDENTITY_BENCHMARK_2026-05-30" / args.device_name)
    results_root.mkdir(parents=True, exist_ok=True)

    regimes = [r.strip() for r in args.regimes.split(",") if r.strip()]
    print(f"[1b] device={args.device_name} reps={args.reps} "
          f"regimes={regimes} dvfs_writable={dvfs_writable()}", flush=True)
    print(f"[1b] starting temp={read_temp():.1f}C", flush=True)

    all_sig = {}
    for r in regimes:
        all_sig[r] = run_regime(r, args.reps, args.batch, results_root,
                                args.device_name)

    if dvfs_writable():
        set_dvfs("auto")

    out = {
        "device_name": args.device_name,
        "phase": "1b",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "reps_per_regime": args.reps,
        "dvfs_writable": dvfs_writable(),
        "regimes": all_sig,
    }
    sig_path = results_root / "signature_thermal.json"
    with open(sig_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n[1b] wrote {sig_path}", flush=True)


if __name__ == "__main__":
    main()
