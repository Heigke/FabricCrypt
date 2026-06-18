"""F5 analyse — for each variant, compute cross-chip modal divergence.
Requires both ikaros and daedalus outputs for each variant.
"""
from __future__ import annotations
import json, struct
from collections import Counter
from pathlib import Path
import numpy as np

ROOT = Path("/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy")
F5 = ROOT / "results/IDENTITY_OPERATOR_2026-05-31/falsify_v2/F5"

VARIANTS = ["F5a_no_atomics", "F5b_no_fma", "F5c_strict_denorm",
            "F5d_no_fast_math", "F5e_all_off"]


def load(p: Path):
    raw = p.read_bytes()
    M, R, hl = struct.unpack("iii", raw[:12])
    body = raw[12 + hl:]
    return np.frombuffer(body, dtype=np.float32).reshape(R, M).copy()


def modal(arr):
    bits = arr.view(np.uint32)
    out = np.zeros(arr.shape[1], dtype=np.uint32)
    freq = np.zeros(arr.shape[1])
    for m in range(arr.shape[1]):
        c = Counter(bits[:, m].tolist())
        v, n = c.most_common(1)[0]
        out[m] = v
        freq[m] = n / arr.shape[0]
    return out, freq


out = {}
for v in VARIANTS:
    pi = F5 / f"{v}_ikaros.bin"
    pd = F5 / f"{v}_daedalus.bin"
    rec = {"ikaros_present": pi.exists(), "daedalus_present": pd.exists()}
    if pi.exists() and pd.exists():
        try:
            A, B = load(pi), load(pd)
            ma, fa = modal(A)
            mb, fb = modal(B)
            diff = int((ma != mb).sum())
            rec.update({
                "cross_chip_modal_diff_frac": diff / A.shape[1],
                "ikaros_modal_stability": float(fa.mean()),
                "daedalus_modal_stability": float(fb.mean()),
                "ikaros_within_var_mean": float(A.var(axis=0).mean()),
                "daedalus_within_var_mean": float(B.var(axis=0).mean()),
                "M": int(A.shape[1]),
                "R": int(A.shape[0]),
            })
        except Exception as e:
            rec["error"] = repr(e)
    out[v] = rec

# Interpretation: the variant where cross_chip_modal_diff_frac DROPS to ~0
# is the variant whose disabled feature was the source of divergence.
baseline = 0.297  # from current measurement
out["_interpretation"] = {
    "baseline_cross_chip_divergence": baseline,
    "rule": "If variant X drops divergence to ~0, then disabling feature X removed the signal.",
}

(F5 / "F5_summary.json").write_text(json.dumps(out, indent=2))
print(json.dumps(out, indent=2))
