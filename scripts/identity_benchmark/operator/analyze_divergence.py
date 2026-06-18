"""analyze_divergence.py — C1 gate analysis.

Reads two binary outputs from divergent_matmul (ikaros + daedalus) and:
  1. Per-element bit-difference rate (XOR of float32 bits, count bits != 0)
  2. Fraction of (rep, m) pairs where ikaros ≠ daedalus
  3. Within-host consistency: does the same chip give the same divergence
     pattern run-to-run, or is it pure noise?
  4. Gate G1: ≥5% bits differ AND difference is *consistent* per chip
"""
from __future__ import annotations
import struct
import sys
import numpy as np
from pathlib import Path


def load(path: Path):
    raw = path.read_bytes()
    M, R, hl = struct.unpack("iii", raw[:12])
    host = raw[12:12 + hl].decode()
    body = raw[12 + hl:]
    arr = np.frombuffer(body, dtype=np.float32).reshape(R, M)
    return host, M, R, arr


def bit_diff_count(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Per-element popcount of XOR of float32 bit patterns."""
    ai = a.view(np.uint32)
    bi = b.view(np.uint32)
    xor = ai ^ bi
    # popcount per uint32 — use builtin via unpack to bytes
    pc = np.zeros_like(xor)
    v = xor.copy()
    for _ in range(32):
        pc += (v & 1)
        v >>= 1
    return pc


def main():
    a_path = Path(sys.argv[1]) if len(sys.argv) > 1 else \
        Path("results/IDENTITY_OPERATOR_2026-05-31/ikaros_div.bin")
    b_path = Path(sys.argv[2]) if len(sys.argv) > 2 else \
        Path("results/IDENTITY_OPERATOR_2026-05-31/daedalus_div.bin")

    ha, Ma, Ra, A = load(a_path)
    hb, Mb, Rb, B = load(b_path)
    assert (Ma, Ra) == (Mb, Rb), f"shape mismatch {Ma,Ra} vs {Mb,Rb}"
    M, R = Ma, Ra
    print(f"hosts: {ha} vs {hb}, M={M}, R={R}, total elements={M*R}")

    # --- Within-host: run-to-run variance on the SAME chip ---
    A_var = A.var(axis=0)  # per-output variance over reps
    B_var = B.var(axis=0)
    print(f"within-host var: ikaros mean={A_var.mean():.3e}, "
          f"daedalus mean={B_var.mean():.3e}")

    # --- Same chip: bits that ARE stable across reps on each chip ---
    # Use rep[0] of each host as the "canonical" per-chip output
    A0, B0 = A[0], B[0]
    elem_pc = bit_diff_count(A0, B0)
    elem_diff = (elem_pc > 0)
    print(f"per-element bit-diff (rep0): mean popcount = "
          f"{elem_pc.mean():.3f} bits/element (of 32)")
    print(f"fraction of elements with ANY bit difference (rep0): "
          f"{elem_diff.mean():.3f}")

    # Total bits differing / total bits
    bit_diff_rate = elem_pc.sum() / (M * 32)
    print(f"total bit-diff rate (rep0): {bit_diff_rate:.4f}  "
          f"({elem_pc.sum()} of {M*32} bits)")

    # --- Across all reps: is per-chip output STABLE? ---
    # Count: how many reps does each chip produce its modal value
    def stability(arr):
        bits = arr.view(np.uint32)  # (R, M)
        modes = []
        for m in range(arr.shape[1]):
            vals, counts = np.unique(bits[:, m], return_counts=True)
            modes.append(counts.max() / arr.shape[0])
        return np.array(modes)

    a_stab = stability(A)
    b_stab = stability(B)
    print(f"per-chip stability (frac reps that match modal value): "
          f"ikaros mean={a_stab.mean():.3f}, daedalus mean={b_stab.mean():.3f}")

    # --- Across-chip divergence: do they have DIFFERENT modal values? ---
    A_bits = A.view(np.uint32)
    B_bits = B.view(np.uint32)
    div_count = 0
    for m in range(M):
        a_mode = np.bincount(A_bits[:, m]).argmax() if A_bits[:, m].min() >= 0 \
            else int(np.bincount(A_bits[:, m] & 0xffff).argmax())
        # simpler: use most common
        from collections import Counter
        a_mode = Counter(A_bits[:, m].tolist()).most_common(1)[0][0]
        b_mode = Counter(B_bits[:, m].tolist()).most_common(1)[0][0]
        if a_mode != b_mode:
            div_count += 1
    div_frac = div_count / M
    print(f"cross-chip modal divergence: {div_count}/{M} = {div_frac:.3f} "
          f"outputs differ in modal bit-pattern")

    # --- Gate G1 ---
    g1_bits = bit_diff_rate >= 0.05
    g1_consistent = (a_stab.mean() > 0.5) and (b_stab.mean() > 0.5)
    g1_pass = g1_bits and g1_consistent and div_frac > 0.05
    print()
    print(f"=== GATE G1 (≥5% bits differ AND per-chip consistent) ===")
    print(f"  bit-diff rate ≥ 5%      : {bit_diff_rate:.4f} -> "
          f"{'PASS' if g1_bits else 'FAIL'}")
    print(f"  per-chip stability > 50%: ikaros={a_stab.mean():.3f}, "
          f"daedalus={b_stab.mean():.3f} -> "
          f"{'PASS' if g1_consistent else 'FAIL'}")
    print(f"  cross-chip modal diff   : {div_frac:.3f} -> "
          f"{'PASS' if div_frac > 0.05 else 'FAIL'}")
    print(f"  OVERALL                 : {'PASS' if g1_pass else 'FAIL'}")


if __name__ == "__main__":
    main()
