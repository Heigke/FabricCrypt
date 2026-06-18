"""F9 — compare two time-separated snapshots for modal-value drift."""
from __future__ import annotations
import json, struct, sys
from collections import Counter
from pathlib import Path
import numpy as np

F9 = Path("/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy/results/IDENTITY_OPERATOR_2026-05-31/falsify_v2/F9")
import socket
HOST = socket.gethostname()

def load(p):
    raw = p.read_bytes()
    M, R, hl = struct.unpack("iii", raw[:12])
    return np.frombuffer(raw[12+hl:], dtype=np.float32).reshape(R, M).copy()

def modal(arr):
    bits = arr.view(np.uint32); R, M = bits.shape
    v = np.zeros(M, dtype=np.uint32); f = np.zeros(M)
    for m in range(M):
        c = Counter(bits[:, m].tolist())
        val, n = c.most_common(1)[0]
        v[m] = val; f[m] = n / R
    return v, f

p0 = F9 / f"{HOST}_t0.bin"
p1 = F9 / f"{HOST}_t1.bin"
out = {"t0_present": p0.exists(), "t1_present": p1.exists()}
if p0.exists() and p1.exists():
    A = load(p0); B = load(p1)
    va, fa = modal(A); vb, fb = modal(B)
    drift = int((va != vb).sum())
    out.update({
        "host": HOST,
        "M": int(A.shape[1]),
        "modal_drift_count": drift,
        "modal_drift_frac": drift / A.shape[1],
        "t0_stability": float(fa.mean()),
        "t1_stability": float(fb.mean()),
        "verdict": ("STABLE (silicon-bound)" if drift / A.shape[1] < 0.05
                    else "DRIFTED (state-bound)"),
    })
(F9 / f"F9_{HOST}_summary.json").write_text(json.dumps(out, indent=2))
print(json.dumps(out, indent=2))
