"""F11 analyse — cross-chip divergence on adversarial inputs."""
from __future__ import annotations
import json, struct
from collections import Counter
from pathlib import Path
import numpy as np

F11 = Path("/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy/results/IDENTITY_OPERATOR_2026-05-31/falsify_v2/F11")

def load(p):
    raw = p.read_bytes()
    M, R, hl = struct.unpack("iii", raw[:12])
    return np.frombuffer(raw[12+hl:], dtype=np.float32).reshape(R, M).copy()

def modal_stab(arr):
    bits = arr.view(np.uint32); R, M = bits.shape
    out = np.zeros(M); vals = np.zeros(M, dtype=np.uint32)
    for m in range(M):
        c = Counter(bits[:, m].tolist())
        v, n = c.most_common(1)[0]
        vals[m] = v; out[m] = n / R
    return vals, out

def bit_diff_rate(a, b):
    ai = a.view(np.uint32); bi = b.view(np.uint32)
    pc = 0; v = ai ^ bi
    for _ in range(32):
        pc += int((v & 1).sum()); v >>= 1
    return pc / (a.size * 32)

pi = F11 / "ikaros_f11.bin"
pd = F11 / "daedalus_f11.bin"
out = {"ikaros_present": pi.exists(), "daedalus_present": pd.exists()}
if pi.exists() and pd.exists():
    A = load(pi); B = load(pd)
    va, fa = modal_stab(A); vb, fb = modal_stab(B)
    div_frac = float((va != vb).sum() / A.shape[1])
    out.update({
        "M": int(A.shape[1]), "R": int(A.shape[0]),
        "ikaros_modal_stability": float(fa.mean()),
        "daedalus_modal_stability": float(fb.mean()),
        "cross_chip_modal_divergence": div_frac,
        "bit_diff_rate_rep0": float(bit_diff_rate(A[0], B[0])),
        "baseline_random_input_divergence": 0.297,
        "interpretation": (
            "If adversarial >> baseline: divergence is precision-bounded; "
            "operator-substrate signal can be amplified by input choice. "
            "If similar: signal is hardware-bounded regardless of input."
        ),
    })
(F11 / "F11_summary.json").write_text(json.dumps(out, indent=2))
print(json.dumps(out, indent=2))
