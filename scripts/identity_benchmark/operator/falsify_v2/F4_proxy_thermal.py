"""F4 — Cross-state stability (proxy for reboot).

Documents the cross-reboot protocol and uses the existing ikaros 32-rep
capture to measure modal-value stability across thermal/process states.

Because we do not reboot ourselves, we use the existing 32-rep capture
as the BASELINE and emit the protocol for the user to manually re-run
post-reboot.

Output: F4_proxy.json with modal bits per element and stability metrics.
"""
from __future__ import annotations
import json
import struct
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path("/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy")
RES = ROOT / "results/IDENTITY_OPERATOR_2026-05-31"
OUT = RES / "falsify_v2"
OUT.mkdir(parents=True, exist_ok=True)


def load_bin(p: Path):
    raw = p.read_bytes()
    M, R, hl = struct.unpack("iii", raw[:12])
    host = raw[12:12 + hl].decode()
    body = raw[12 + hl:]
    arr = np.frombuffer(body, dtype=np.float32).reshape(R, M).copy()
    return host, M, R, arr


def modal_bits(arr: np.ndarray) -> np.ndarray:
    """Return (M,) of modal uint32 bit-pattern per element."""
    bits = arr.view(np.uint32)
    out = np.zeros(arr.shape[1], dtype=np.uint32)
    for m in range(arr.shape[1]):
        c = Counter(bits[:, m].tolist())
        out[m] = c.most_common(1)[0][0]
    return out


def modal_freq(arr: np.ndarray) -> np.ndarray:
    bits = arr.view(np.uint32)
    out = np.zeros(arr.shape[1], dtype=np.float64)
    for m in range(arr.shape[1]):
        c = Counter(bits[:, m].tolist())
        out[m] = c.most_common(1)[0][1] / arr.shape[0]
    return out


def main():
    ik = RES / "ikaros_div.bin"
    dd = RES / "daedalus_div.bin"
    hi, Mi, Ri, A = load_bin(ik)
    hd, Md, Rd, B = load_bin(dd)

    mode_a = modal_bits(A)
    mode_b = modal_bits(B)
    freq_a = modal_freq(A)
    freq_b = modal_freq(B)

    diff = int((mode_a != mode_b).sum())
    out = {
        "F4_proxy": {
            "ikaros": {"host": hi, "M": Mi, "R": Ri,
                       "modal_stability_mean": float(freq_a.mean()),
                       "modal_stability_min": float(freq_a.min()),
                       "modal_bits_sha256_prefix": int(mode_a.sum() & 0xFFFFFFFF)},
            "daedalus": {"host": hd, "M": Md, "R": Rd,
                         "modal_stability_mean": float(freq_b.mean()),
                         "modal_stability_min": float(freq_b.min()),
                         "modal_bits_sha256_prefix": int(mode_b.sum() & 0xFFFFFFFF)},
            "cross_chip_modal_diff_count": diff,
            "cross_chip_modal_diff_frac": diff / Mi,
        },
        "protocol_documented": {
            "reboot_step_1": "save ikaros_div.bin (DONE; this baseline)",
            "reboot_step_2": "user reboots ikaros manually",
            "reboot_step_3": "rerun divergent_matmul 64 4096 32 ikaros_div_postreboot.bin",
            "reboot_step_4": "compare modal_bits arrays elementwise: if equal -> die-bound; if shifted -> boot-state",
        },
        "note": "True cross-reboot test requires manual reboot. Modal values saved.",
    }
    # also dump modal arrays so post-reboot run can compare bit-exact
    np.save(OUT / "F4_ikaros_modal.npy", mode_a)
    np.save(OUT / "F4_daedalus_modal.npy", mode_b)
    (OUT / "F4_proxy.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out["F4_proxy"], indent=2))


if __name__ == "__main__":
    main()
