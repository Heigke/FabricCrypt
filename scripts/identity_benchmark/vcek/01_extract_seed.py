#!/usr/bin/env python3
"""Stage 1: Extract per-die cryptographic seed -> permutation P of size 512.

SEV-SNP is disabled on Strix Halo (sev_snp=N), so we use the TPM 2.0
Endorsement Key (EK) public modulus as the per-die crypto anchor. The EK
is silicon-bound at TPM provisioning and survives reboot/reinstall.

Outputs:
  results/IDENTITY_BENCHMARK_2026-05-30/vcek/seed_<host>.bin     (32-byte SHA256 seed)
  results/IDENTITY_BENCHMARK_2026-05-30/vcek/P_<host>.npy        (perm of 512)
  results/IDENTITY_BENCHMARK_2026-05-30/vcek/ek_<host>.pub       (EK PEM)
  results/IDENTITY_BENCHMARK_2026-05-30/vcek/meta_<host>.json
"""
import hashlib
import json
import os
import socket
import subprocess
import sys
from pathlib import Path

import numpy as np

HOST = socket.gethostname()
OUT = Path("results/IDENTITY_BENCHMARK_2026-05-30/vcek")
OUT.mkdir(parents=True, exist_ok=True)


def extract_ek_pem() -> bytes:
    """Read EK (handle 0x81010001) via tpm2_readpublic. Requires sudo."""
    pem_path = OUT / f"ek_{HOST}.pub"
    # tpm2_readpublic writes the public area; we capture both PEM and raw text
    r = subprocess.run(
        ["sudo", "-n", "tpm2_readpublic",
         "-c", "0x81010001",
         "-T", "device:/dev/tpm0",
         "-o", str(pem_path),
         "-f", "pem"],
        capture_output=True, text=True,
    )
    if r.returncode != 0:
        raise RuntimeError(f"tpm2_readpublic failed: {r.stderr}")
    # Also pull the modulus hex from stdout for seed material (PEM file has
    # restrictive perms; modulus line is canonical).
    modulus_hex = None
    name_hex = None
    for line in r.stdout.splitlines():
        s = line.strip()
        if s.startswith("rsa:"):
            modulus_hex = s.split(":", 1)[1].strip()
        elif s.startswith("name:"):
            name_hex = s.split(":", 1)[1].strip()
    if not modulus_hex:
        raise RuntimeError("modulus not found in tpm2_readpublic output")
    # Use modulus bytes as the seed material (256 bytes for RSA-2048).
    seed_material = bytes.fromhex(modulus_hex)
    # Persist the readable stdout for the report
    (OUT / f"ek_{HOST}.txt").write_text(r.stdout)
    return seed_material, modulus_hex, name_hex


def derive(seed_material: bytes) -> tuple[bytes, np.ndarray]:
    seed32 = hashlib.sha256(seed_material).digest()
    rng = np.random.default_rng(np.frombuffer(seed32, dtype=np.uint32))
    P = rng.permutation(512).astype(np.int64)
    return seed32, P


def main():
    seed_material, modulus_hex, name_hex = extract_ek_pem()
    seed32, P = derive(seed_material)

    (OUT / f"seed_{HOST}.bin").write_bytes(seed32)
    np.save(OUT / f"P_{HOST}.npy", P)

    # Determinism check: re-extract & re-derive twice, confirm identical
    sm2, _, _ = extract_ek_pem()
    s2, P2 = derive(sm2)
    sm3, _, _ = extract_ek_pem()
    s3, P3 = derive(sm3)
    deterministic = (s2 == seed32 == s3) and np.array_equal(P, P2) and np.array_equal(P, P3)

    meta = {
        "host": HOST,
        "crypto_source": "TPM2_EK_RSA2048_modulus@0x81010001",
        "sev_snp_available": False,
        "sev_snp_note": "/sys/module/kvm_amd/parameters/sev_snp == 'N' on Strix Halo",
        "ek_name_sha256": name_hex,
        "ek_modulus_sha256_hex": hashlib.sha256(seed_material).hexdigest(),
        "seed_hex": seed32.hex(),
        "perm_size": int(P.size),
        "perm_first_8": P[:8].tolist(),
        "perm_last_8": P[-8:].tolist(),
        "deterministic_3x": bool(deterministic),
    }
    (OUT / f"meta_{HOST}.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
