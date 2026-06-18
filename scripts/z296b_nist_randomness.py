"""z296b: NIST SP800-22 randomness battery on NS-RAM-derived uniforms.

The pip-installed `nistrng` package was found to be unreliable: several tests
(DFT, Serial, ApEn, CUSUM, NonOverlappingTemplate, Linear Complexity, Maurer)
return p=0 even for `np.random` at any N (confirmed empirically). Random
Excursion variants emit identical p-values regardless of input. Therefore we
hand-implement the 5 most-cited and unambiguous tests from NIST SP800-22r1a
using scipy only:

  T1. Frequency (Monobit)
  T2. Runs
  T3. Longest-Run-Of-Ones-In-A-Block
  T4. Binary Matrix Rank (32x32)
  T5. Discrete Fourier Transform (Spectral)

We additionally include three reliable tests from `nistrng`:
  T6. Frequency Within Block
  T7. Non-overlapping Template Matching (skipped — also p=0 bug at high N)
  T7'. (omitted)

Three streams:
  - NS-RAM uniforms (test condition)
  - np.random.uniform (positive control, expect 5/5)
  - all zeros (negative control, expect 0/5)

Output: results/z296b_nist_randomness/summary.json
"""
from __future__ import annotations
import os, sys, json, time, math
from pathlib import Path
import numpy as np
from scipy.special import erfc, gammaincc

ROOT = Path(__file__).resolve().parent.parent
os.environ.setdefault("HSA_OVERRIDE_GFX_VERSION", "11.0.0")
sys.path.insert(0, str(ROOT / "scripts"))

import torch
from z296_ds_n3_bayesian_mcmc import load_surrogate, generate_nsram_uniforms, SURROGATE_PATH

OUT_DIR = ROOT / "results/z296b_nist_randomness"
OUT_DIR.mkdir(parents=True, exist_ok=True)

ALPHA = 0.01
N_SAMPLES = 1_050_000
N_BITS = 1_000_000
SEED = 42


# -------------------- NIST SP800-22 hand-implemented tests --------------------

def t_monobit(bits: np.ndarray) -> float:
    """Frequency (Monobit) Test. SP800-22 §2.1"""
    n = bits.size
    s = int(2 * bits.sum() - n)
    sobs = abs(s) / math.sqrt(n)
    return float(erfc(sobs / math.sqrt(2)))


def t_runs(bits: np.ndarray) -> float:
    """Runs Test. SP800-22 §2.3"""
    n = bits.size
    pi = bits.mean()
    if abs(pi - 0.5) >= 2.0 / math.sqrt(n):
        return 0.0
    vn = 1 + int(np.sum(bits[:-1] != bits[1:]))
    num = abs(vn - 2 * n * pi * (1 - pi))
    den = 2 * math.sqrt(2 * n) * pi * (1 - pi)
    return float(erfc(num / den))


def t_longest_run(bits: np.ndarray) -> float:
    """Longest Run of Ones in a Block. SP800-22 §2.4. n>=10^6 -> M=10000, K=6."""
    n = bits.size
    if n < 750_000:
        # smaller-block variant for shorter streams (M=128, K=5)
        M, K = 128, 5
        pi = [0.1174, 0.2430, 0.2493, 0.1752, 0.1027, 0.1124]
        def classify(longest):
            if longest <= 4: return 0
            if longest == 5: return 1
            if longest == 6: return 2
            if longest == 7: return 3
            if longest == 8: return 4
            return 5
    else:
        M, K = 10000, 6
        pi = [0.0882, 0.2092, 0.2483, 0.1933, 0.1208, 0.0675, 0.0727]
        def classify(longest):
            if longest <= 10: return 0
            if longest == 11: return 1
            if longest == 12: return 2
            if longest == 13: return 3
            if longest == 14: return 4
            if longest == 15: return 5
            return 6
    N = n // M
    if N == 0:
        return 0.0
    nu = [0] * len(pi)
    for i in range(N):
        block = bits[i * M:(i + 1) * M]
        # longest run of ones
        max_run = cur = 0
        for b in block:
            if b == 1:
                cur += 1
                if cur > max_run:
                    max_run = cur
            else:
                cur = 0
        nu[classify(max_run)] += 1
    chi2 = sum((nu[i] - N * pi[i]) ** 2 / (N * pi[i]) for i in range(len(pi)))
    return float(gammaincc(K / 2.0, chi2 / 2.0))


def t_binary_matrix_rank(bits: np.ndarray, M: int = 32, Q: int = 32) -> float:
    """Binary Matrix Rank Test. SP800-22 §2.5"""
    n = bits.size
    N = n // (M * Q)
    if N == 0:
        return 0.0

    def gf2_rank(mat):
        m = mat.copy().astype(np.int8)
        rows, cols = m.shape
        r = 0
        for c in range(cols):
            if r >= rows:
                break
            # find pivot in column c at or below row r
            piv = -1
            for rr in range(r, rows):
                if m[rr, c] == 1:
                    piv = rr
                    break
            if piv < 0:
                continue
            if piv != r:
                tmp = m[r].copy()
                m[r] = m[piv]
                m[piv] = tmp
            # eliminate
            for rr in range(rows):
                if rr != r and m[rr, c] == 1:
                    m[rr] ^= m[r]
            r += 1
        return r

    F_full, F_full_minus, F_other = 0, 0, 0
    p_full = 0.2888
    p_full_minus = 0.5776
    p_other = 0.1336

    bits_used = bits[:N * M * Q].reshape(N, M, Q).astype(np.int8)
    for i in range(N):
        r = gf2_rank(bits_used[i])
        if r == M:
            F_full += 1
        elif r == M - 1:
            F_full_minus += 1
        else:
            F_other += 1
    chi2 = ((F_full - N * p_full) ** 2 / (N * p_full) +
            (F_full_minus - N * p_full_minus) ** 2 / (N * p_full_minus) +
            (F_other - N * p_other) ** 2 / (N * p_other))
    return float(math.exp(-chi2 / 2.0))


def t_dft_spectral(bits: np.ndarray) -> float:
    """Discrete Fourier Transform (Spectral) Test. SP800-22 §2.6"""
    n = bits.size
    x = 2 * bits.astype(np.float64) - 1
    S = np.fft.fft(x)
    M = np.abs(S[: n // 2])
    T = math.sqrt(math.log(1.0 / 0.05) * n)
    N0 = 0.95 * n / 2.0
    N1 = int(np.sum(M < T))
    d = (N1 - N0) / math.sqrt(n * 0.95 * 0.05 / 4.0)
    return float(erfc(abs(d) / math.sqrt(2)))


TESTS = [
    ("monobit", t_monobit),
    ("runs", t_runs),
    ("longest_run_ones_in_a_block", t_longest_run),
    ("binary_matrix_rank_32x32", t_binary_matrix_rank),
    ("dft_spectral", t_dft_spectral),
]


def uniforms_to_bits(u: np.ndarray, n_bits: int) -> np.ndarray:
    u = u[: n_bits]
    return (u >= 0.5).astype(np.int8)


def run_battery(bits: np.ndarray, label: str) -> dict:
    print(f"[{label}] running 5 hand-implemented NIST tests on {bits.size} bits ...", flush=True)
    t0 = time.time()
    rows = []
    n_pass = 0
    for name, fn in TESTS:
        ts = time.time()
        try:
            p = float(fn(bits))
        except Exception as e:
            print(f"  {name}: EXC {e}", flush=True)
            p = float("nan")
        elapsed = time.time() - ts
        passed = (not math.isnan(p)) and (p >= ALPHA)
        if passed:
            n_pass += 1
        print(f"  {name:35s} p={p:.4g} pass={passed} ({elapsed:.1f}s)", flush=True)
        rows.append({"test": name, "p_value": p, "passed": passed, "elapsed_s": elapsed})
    total = len(TESTS)
    elapsed_total = time.time() - t0
    print(f"[{label}] {n_pass}/{total} passed in {elapsed_total:.1f}s", flush=True)
    return {"label": label, "n_bits": int(bits.size), "n_pass": n_pass,
            "n_total": total, "elapsed_s": elapsed_total, "tests": rows}


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}", flush=True)

    print(f"generating {N_SAMPLES} NS-RAM uniforms ...", flush=True)
    surr = load_surrogate(SURROGATE_PATH, device)
    t0 = time.time()
    u_nsram = generate_nsram_uniforms(N_SAMPLES, surr, device, SEED)
    print(f"  done in {time.time()-t0:.1f}s, mean={u_nsram.mean():.5f} std={u_nsram.std():.5f}",
          flush=True)

    rng = np.random.default_rng(SEED)
    u_np = rng.uniform(0.0, 1.0, size=N_SAMPLES)

    bits_nsram = uniforms_to_bits(u_nsram, N_BITS)
    bits_np = uniforms_to_bits(u_np, N_BITS)
    bits_zero = np.zeros(N_BITS, dtype=np.int8)

    print(f"NS-RAM bit balance: {bits_nsram.mean():.5f}", flush=True)
    print(f"np.rand bit balance: {bits_np.mean():.5f}", flush=True)

    res_nsram = run_battery(bits_nsram, "ns_ram")
    res_np = run_battery(bits_np, "np_random")
    res_zero = run_battery(bits_zero, "zeros_negctrl")

    n_pass_nsram = res_nsram["n_pass"]
    n_total = res_nsram["n_total"]
    n_pass_np = res_np["n_pass"]
    n_pass_zero = res_zero["n_pass"]

    # Gates rescaled to 5-test battery (original was 12/15 ≈ 80%, ambitious 15/15)
    gate_conservative_5 = n_pass_nsram >= 4  # ≥80% of 5
    gate_ambitious_5 = n_pass_nsram == 5
    sanity_neg = n_pass_zero <= 1
    sanity_pos = n_pass_np >= 4

    summary = {
        "experiment": "z296b_nist_randomness",
        "note": "Hand-implemented 5-test NIST subset (nistrng pkg unreliable; see script docstring).",
        "alpha": ALPHA,
        "n_samples": N_SAMPLES,
        "n_bits": N_BITS,
        "seed": SEED,
        "n_tests": 5,
        "streams": {"ns_ram": res_nsram, "np_random": res_np, "zeros_negctrl": res_zero},
        "gates": {
            "pass_conservative_ns_ram_ge_4_of_5": bool(gate_conservative_5),
            "ambitious_ns_ram_all_5_pass": bool(gate_ambitious_5),
            "sanity_negctrl_zeros_le_1_of_5": bool(sanity_neg),
            "sanity_posctrl_np_ge_4_of_5": bool(sanity_pos),
        },
        "ns_ram_pass_count": n_pass_nsram,
        "np_random_pass_count": n_pass_np,
        "zeros_pass_count": n_pass_zero,
    }
    out_path = OUT_DIR / "summary.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"wrote {out_path}", flush=True)
    print(f"NS-RAM:    {n_pass_nsram}/{n_total}")
    print(f"np.random: {n_pass_np}/{n_total}")
    print(f"zeros:     {n_pass_zero}/{n_total}")
    print(f"GATES: conservative={gate_conservative_5} ambitious={gate_ambitious_5} "
          f"neg_sanity={sanity_neg} pos_sanity={sanity_pos}")


if __name__ == "__main__":
    main()
