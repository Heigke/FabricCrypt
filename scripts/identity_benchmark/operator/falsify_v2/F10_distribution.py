"""F10 — distribution-of-differences analysis.

For the 22% of reps that are non-modal on each chip, examine WHICH bit
positions flip, and whether they flip together (cluster analysis).
"""
from __future__ import annotations
import json, struct
from collections import Counter
from pathlib import Path
import numpy as np

ROOT = Path("/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy")
RES = ROOT / "results/IDENTITY_OPERATOR_2026-05-31"
OUT = RES / "falsify_v2/F10"
OUT.mkdir(parents=True, exist_ok=True)


def load(p):
    raw = p.read_bytes()
    M, R, hl = struct.unpack("iii", raw[:12])
    return np.frombuffer(raw[12 + hl:], dtype=np.float32).reshape(R, M).copy()


def analyze(arr, label):
    bits = arr.view(np.uint32)  # (R, M)
    R, M = bits.shape
    modal = np.zeros(M, dtype=np.uint32)
    for m in range(M):
        c = Counter(bits[:, m].tolist())
        modal[m] = c.most_common(1)[0][0]

    # Per-element, per-rep XOR vs modal
    xor = bits ^ modal[None, :]  # (R, M)
    # Per-bit-position aggregate (over all reps & elements)
    bit_counts = np.zeros(32, dtype=np.int64)
    for b in range(32):
        bit_counts[b] = int(((xor >> b) & 1).sum())

    # Mantissa (bits 0-22) vs exponent (23-30) vs sign (31)
    mantissa_total = int(bit_counts[:23].sum())
    exponent_total = int(bit_counts[23:31].sum())
    sign_total = int(bit_counts[31])

    # Fraction of reps where each element differs from modal
    diff_per_rep_per_elem = (xor != 0).astype(int)
    non_modal_frac_per_elem = diff_per_rep_per_elem.mean(axis=0)
    avg_bits_flipped_when_diff = []
    for m in range(M):
        diff_mask = diff_per_rep_per_elem[:, m].astype(bool)
        if diff_mask.any():
            bf = []
            for r in np.where(diff_mask)[0]:
                bf.append(bin(int(xor[r, m])).count("1"))
            avg_bits_flipped_when_diff.append(float(np.mean(bf)))
        else:
            avg_bits_flipped_when_diff.append(0.0)

    # Cluster: do bits 0-3 (LSBs) co-occur with bits 4-7 etc?
    # Pearson corr of binary flips
    flip_mat = np.zeros((32, R * M), dtype=np.int8)
    for b in range(32):
        flip_mat[b] = ((xor.flatten() >> b) & 1)
    # corr only between bits with nonzero variance
    active = [b for b in range(32) if flip_mat[b].sum() > 5]
    corr_sub = np.corrcoef(flip_mat[active]) if len(active) > 1 else np.zeros((0, 0))

    summary = {
        "label": label,
        "R": R, "M": M,
        "per_bit_flip_counts": bit_counts.tolist(),
        "mantissa_bits_flipped_total": mantissa_total,
        "exponent_bits_flipped_total": exponent_total,
        "sign_bits_flipped_total": sign_total,
        "mantissa_dominance": mantissa_total / max(1, mantissa_total + exponent_total + sign_total),
        "non_modal_frac_per_elem_mean": float(non_modal_frac_per_elem.mean()),
        "avg_bits_flipped_when_diff_mean": float(np.mean(avg_bits_flipped_when_diff)),
        "active_flip_bits": active,
        "active_bit_corr_matrix_offdiag_mean": float(
            (corr_sub.sum() - np.trace(corr_sub)) / max(1, corr_sub.size - corr_sub.shape[0])
        ) if corr_sub.size > 0 else 0.0,
        "interpretation_rule": (
            "mantissa_dominance ~1.0 + LSB concentration = silicon-bound (accumulator order). "
            "exponent flips = catastrophic precision loss = software path matters."
        ),
    }
    return summary


def main():
    A = load(RES / "ikaros_div.bin")
    B = load(RES / "daedalus_div.bin")
    sa = analyze(A, "ikaros")
    sb = analyze(B, "daedalus")
    out = {"F10": {"ikaros": sa, "daedalus": sb}}
    (OUT / "F10_distribution.json").write_text(json.dumps(out, indent=2))
    print(json.dumps({k: {kk: v for kk, v in vv.items()
                          if kk not in ("per_bit_flip_counts", "active_flip_bits")}
                      for k, vv in out["F10"].items()}, indent=2))


if __name__ == "__main__":
    main()
