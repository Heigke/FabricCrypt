"""F6 — bootstrap 95% CI on stability + cross-chip divergence."""
from __future__ import annotations
import json, struct
from collections import Counter
from pathlib import Path
import numpy as np

ROOT = Path("/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy")
F6 = ROOT / "results/IDENTITY_OPERATOR_2026-05-31/falsify_v2/F6"

def load(p):
    raw = p.read_bytes()
    M, R, hl = struct.unpack("iii", raw[:12])
    return np.frombuffer(raw[12 + hl:], dtype=np.float32).reshape(R, M).copy()

def modal_freq(arr):
    bits = arr.view(np.uint32)
    f = np.zeros(arr.shape[1])
    for m in range(arr.shape[1]):
        c = Counter(bits[:, m].tolist())
        f[m] = c.most_common(1)[0][1] / arr.shape[0]
    return f

def modal_val(arr):
    bits = arr.view(np.uint32)
    v = np.zeros(arr.shape[1], dtype=np.uint32)
    for m in range(arr.shape[1]):
        c = Counter(bits[:, m].tolist())
        v[m] = c.most_common(1)[0][0]
    return v

def bit_diff_rate(a, b):
    ai = a.view(np.uint32); bi = b.view(np.uint32)
    pc = np.zeros_like(ai)
    v = ai ^ bi
    for _ in range(32):
        pc += (v & 1); v >>= 1
    return pc.sum() / (a.size * 32)

def main():
    pi = F6 / "ikaros_500.bin"
    pd = F6 / "daedalus_500.bin"
    if not (pi.exists() and pd.exists()):
        print(json.dumps({"error": "need both ikaros_500 + daedalus_500 bins",
                          "ikaros": pi.exists(), "daedalus": pd.exists()}, indent=2))
        return

    A = load(pi); B = load(pd)
    R, M = A.shape
    assert B.shape == A.shape
    print(f"R={R} M={M}")

    rng = np.random.default_rng(42)
    NBOOT = 2000

    fa = modal_freq(A); fb = modal_freq(B)
    mva = modal_val(A); mvb = modal_val(B)
    div = (mva != mvb).astype(float)

    bd_rate = bit_diff_rate(A[0], B[0])

    def boot_mean(x, nboot=NBOOT):
        n = len(x)
        means = np.empty(nboot)
        for i in range(nboot):
            means[i] = x[rng.integers(0, n, n)].mean()
        return np.quantile(means, [0.025, 0.5, 0.975])

    ci_fa = boot_mean(fa)
    ci_fb = boot_mean(fb)
    ci_div = boot_mean(div)

    # Bit-diff bootstrap: resample columns of A[0],B[0]
    def boot_bd(a0, b0, nboot=NBOOT):
        n = len(a0)
        ai = a0.view(np.uint32); bi = b0.view(np.uint32)
        per_elem_bits = np.zeros(n)
        for j in range(n):
            x = ai[j] ^ bi[j]
            per_elem_bits[j] = bin(x).count("1") / 32.0
        means = np.empty(nboot)
        for i in range(nboot):
            means[i] = per_elem_bits[rng.integers(0, n, n)].mean()
        return np.quantile(means, [0.025, 0.5, 0.975])

    ci_bd = boot_bd(A[0], B[0])

    out = {
        "F6_bootstrap": {
            "R_per_host": R,
            "M": M,
            "ikaros_modal_stability_CI95": ci_fa.tolist(),
            "daedalus_modal_stability_CI95": ci_fb.tolist(),
            "cross_chip_modal_divergence_CI95": ci_div.tolist(),
            "bit_diff_rate_rep0_CI95": ci_bd.tolist(),
            "bit_diff_rate_rep0_point": float(bd_rate),
            "baseline_32rep_claim": {"modal_stab": 0.78, "cross_chip_div": 0.297},
            "verdict": (
                "tight" if (ci_div[2] - ci_div[0]) < 0.10
                else "loose" if (ci_div[2] - ci_div[0]) < 0.25
                else "very_loose"
            ),
        }
    }
    (F6 / "F6_summary.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))

if __name__ == "__main__":
    main()
