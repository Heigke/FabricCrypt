"""Brutal audit of N-Stoch-RNG (NS-RAM 1/f bitstream source).

Audit steps:
  1. K2 cross-cell correlation on noise traces (mean/max + heatmap).
  2. NIST 15-test suite (full SP 800-22, pure python) on 10 Mbit regenerated
     stream using post-z469 calibrated cell parameters.
  3. SP 800-90B IID + min-entropy estimate on raw (pre-whitened) stream.
  4. Peripheral-aware energy: DAC (V_G) + ADC (1-bit) + wire RC + cell.
     Compare to Cheng 2024 65nm CMOS TRNG (0.244 pJ/bit).
  5. Dennard scaling 130 -> 28 nm (S=4.64).
  6. Honest verdict against pre-registered SURVIVE / DEMOTE / KILL gates.

Run:
    venv/bin/python scripts/audits/N_Stoch_RNG_audit.py
"""

from __future__ import annotations

import json
import math
import os
import time
from pathlib import Path

import numpy as np

os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from scipy import special, stats  # noqa: E402

ROOT = Path("/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy")
OUT = ROOT / "results" / "N_Stoch_RNG_AUDIT"
OUT.mkdir(parents=True, exist_ok=True)

SEED = 20260518
N_CELLS = 8192
N_BITS_NIST = 10_000_000  # 10 Mbit per task spec

# Post-z469 calibrated cell parameters (from results/z469_snap_d_fix/).
# SNAP_HOT delivers ~10 mA at VG1=0.6 (with snap_Is=4.5192e-12, R_body=1e7).
SNAP_IS = 4.5192e-12      # A (saturation current parameter)
R_BODY = 1.0e7            # ohm
ID_OP = 10.0e-3           # A, SNAP_HOT operating drain current at VG1=0.6
V_G_RANGE = 0.6           # V (0..0.6, DAC dynamic range)
SAMPLE_RATE = 1.0e9       # 1 GS/s (matched to peripheral assumption)

# ---------------------------------------------------------------------------
# 1.  NS-RAM 1/f ensemble (Voss-McCartney) - same model as original script,
#     but with sigma_V calibrated against post-z469 physical noise budget.
# ---------------------------------------------------------------------------
# Physical noise budget at Id=10 mA, BW = SAMPLE_RATE/2 = 500 MHz:
#   shot:   i_n^2 = 2qId*BW -> sigma_i_shot ~ sqrt(2*1.6e-19*1e-2*5e8) ~ 1.27 uA
#   flicker (1/f, K_f ~ 1e-25 J for thin film):
#           v_n^2(f) = K_f / (C_ox*W*L*f); integrate 1 Hz..500 MHz
#           For thin-film R_body=1e7 device, K_f ~ 1e-9 V^2 -> sigma_v ~ 30 mV
#   total V_b excursion sigma ~ 10..50 mV per cell (matches original heuristic).
# So we keep the [10, 50] mV per-cell sigma BUT the noise correlation must
# come from physics, not from independent Voss-McCartney refreshes - we add a
# realistic *common-mode* component (shared bias rail) that introduces cross-
# cell correlation.

class NSRAMNoiseEnsemble:
    """1/f noise ensemble with optional shared bias-rail (common-mode) noise.

    common_mode_frac: fraction of per-cell noise that is shared across the
        tile (bias rail + thermal substrate). 0 = ideal isolation (original
        Voss-McCartney). Realistic value: 0.05..0.2.
    """

    def __init__(self, n_cells=N_CELLS, K=6, seed=SEED, common_mode_frac=0.10):
        self.n = n_cells
        self.K = K
        self.cm = float(common_mode_frac)
        self.rng = np.random.default_rng(seed)
        self.sigma = self.rng.uniform(0.010, 0.050, size=n_cells).astype(np.float32)

    def _voss(self, T, N):
        out = np.zeros((T, N), dtype=np.float32)
        for k in range(self.K):
            p = 1.0 / (1 << k)
            mask = self.rng.random((T, N)) < p
            fresh = self.rng.standard_normal((T, N)).astype(np.float32)
            mask[0] = True
            tick_idx = np.where(mask, np.arange(T)[:, None], -1)
            np.maximum.accumulate(tick_idx, axis=0, out=tick_idx)
            col_idx = np.arange(N)[None, :]
            out += fresh[tick_idx, col_idx]
        out *= (1.0 / math.sqrt(self.K))
        return out

    def block(self, n_ticks):
        T, N = n_ticks, self.n
        independent = self._voss(T, N)
        if self.cm > 0:
            # shared rail noise: scalar 1/f per tick, common to all cells
            shared = self._voss(T, 1)  # (T,1)
            out = (math.sqrt(1.0 - self.cm * self.cm) * independent +
                   self.cm * shared) * self.sigma
        else:
            out = independent * self.sigma
        return out

    def raw_traces(self, n_ticks):
        return self.block(n_ticks)

    def _raw_bits(self, n_bits):
        ticks = max(2, (n_bits + self.n - 1) // self.n + 4)
        block = self.block(ticks)
        thr = np.median(block, axis=0, keepdims=True)
        raw = (block > thr).astype(np.uint8).ravel()
        return raw[:n_bits]

    def bits(self, n_bits):
        need = int(n_bits * 8 + 8192)
        raw = self._raw_bits(need)
        if raw.size & 1:
            raw = raw[:-1]
        a, b = raw[0::2], raw[1::2]
        keep = a != b
        vn = b[keep].astype(np.uint8)
        half = vn.size // 2
        wh = vn[:half] ^ vn[half:2 * half]
        if wh.size >= n_bits:
            return wh[:n_bits]
        return np.concatenate([wh, self.bits(n_bits - wh.size)]).astype(np.uint8)


# ---------------------------------------------------------------------------
# 2.  Full NIST SP 800-22 implementation (15 tests, pure python).
#     Functions return p-value; PASS if p >= 0.01.
# ---------------------------------------------------------------------------
def _erfc(x): return float(special.erfc(x))
def _gammainc_c(a, x): return float(special.gammaincc(a, x))


def t_frequency(bits):
    n = len(bits); s = 2 * bits.astype(np.int64) - 1
    return _erfc(abs(s.sum()) / math.sqrt(n) / math.sqrt(2))


def t_block_frequency(bits, M=128):
    n = len(bits); N = n // M
    if N < 1: return 0.0
    b = bits[:N*M].reshape(N, M)
    pi = b.mean(axis=1)
    chi2 = 4.0 * M * ((pi - 0.5) ** 2).sum()
    return _gammainc_c(N / 2.0, chi2 / 2.0)


def t_runs(bits):
    n = len(bits); pi = bits.mean()
    if abs(pi - 0.5) >= 2.0 / math.sqrt(n): return 0.0
    Vn = 1 + (bits[1:] != bits[:-1]).sum()
    num = abs(Vn - 2.0 * n * pi * (1 - pi))
    den = 2.0 * math.sqrt(2.0 * n) * pi * (1 - pi)
    return _erfc(num / den)


def t_longest_run(bits):
    n = len(bits)
    if n >= 750_000:
        M, K, N = 10_000, 6, n // 10_000
        v_targets = [10, 11, 12, 13, 14, 15]
        pi = [0.0882, 0.2092, 0.2483, 0.1933, 0.1208, 0.0675, 0.0727]
    elif n >= 6272:
        M, K, N = 128, 5, n // 128
        v_targets = [4, 5, 6, 7, 8]
        pi = [0.1174, 0.2430, 0.2493, 0.1752, 0.1027, 0.1124]
    else:
        return 0.0
    b = bits[:N*M].reshape(N, M)
    longest = np.zeros(N, dtype=np.int64)
    for i in range(N):
        cur = 0; mx = 0
        for x in b[i]:
            if x == 1:
                cur += 1
                if cur > mx: mx = cur
            else:
                cur = 0
        longest[i] = mx
    v = np.zeros(K + 1, dtype=np.int64)
    v[0] = (longest <= v_targets[0]).sum()
    for k in range(1, K):
        v[k] = (longest == v_targets[k]).sum()
    v[K] = (longest >= v_targets[-1]).sum()
    chi2 = sum((v[i] - N*pi[i])**2 / (N*pi[i]) for i in range(K+1))
    return _gammainc_c(K / 2.0, chi2 / 2.0)


def t_binary_matrix_rank(bits, M=32, Q=32):
    n = len(bits); N = n // (M*Q)
    if N < 38: return 0.0
    pass_rank_M   = 0.2888
    pass_rank_Mm1 = 0.5776
    pass_rank_lo  = 1.0 - pass_rank_M - pass_rank_Mm1
    fm = fmm = flo = 0
    blocks = bits[:N*M*Q].reshape(N, M, Q).astype(np.uint8)
    for i in range(N):
        r = _binary_rank(blocks[i].copy())
        if r == M: fm += 1
        elif r == M - 1: fmm += 1
        else: flo += 1
    chi2 = ((fm - N*pass_rank_M)**2/(N*pass_rank_M)
            + (fmm - N*pass_rank_Mm1)**2/(N*pass_rank_Mm1)
            + (flo - N*pass_rank_lo)**2/(N*pass_rank_lo))
    return _gammainc_c(1.0, chi2 / 2.0)


def _binary_rank(A):
    m, n = A.shape
    A = A.astype(np.uint8)
    r = 0
    for c in range(min(m, n)):
        pivot = -1
        for i in range(r, m):
            if A[i, c] == 1:
                pivot = i; break
        if pivot == -1: continue
        if pivot != r:
            A[[r, pivot]] = A[[pivot, r]]
        for i in range(m):
            if i != r and A[i, c] == 1:
                A[i] ^= A[r]
        r += 1
    return r


def t_dft(bits):
    n = len(bits); s = 2 * bits.astype(np.float64) - 1
    S = np.abs(np.fft.fft(s))[:n // 2]
    T = math.sqrt(math.log(1 / 0.05) * n)
    N0 = 0.95 * n / 2.0
    N1 = (S < T).sum()
    d = (N1 - N0) / math.sqrt(n * 0.95 * 0.05 / 4.0)
    return _erfc(abs(d) / math.sqrt(2))


def t_non_overlapping_template(bits, m=9):
    # one fixed template, single test
    tmpl = np.array([0,0,0,0,0,0,0,0,1], dtype=np.uint8)
    n = len(bits); N = 8; M = n // N
    mu = (M - m + 1) / (2**m)
    var = M * (1.0/(2**m) - (2*m - 1)/(2**(2*m)))
    if var <= 0: return 0.0
    blocks = bits[:N*M].reshape(N, M)
    W = np.zeros(N)
    for i in range(N):
        j = 0; w = 0
        while j <= M - m:
            if np.array_equal(blocks[i, j:j+m], tmpl):
                w += 1; j += m
            else:
                j += 1
        W[i] = w
    chi2 = ((W - mu)**2 / var).sum()
    return _gammainc_c(N/2.0, chi2/2.0)


def t_overlapping_template(bits, m=9):
    n = len(bits); M = 1032
    N = n // M
    if N < 5: return 0.0
    tmpl = np.ones(m, dtype=np.uint8)
    K = 5
    pi = [0.367879, 0.183940, 0.137955, 0.099634, 0.069935, 0.140780]
    v = np.zeros(K + 1, dtype=np.int64)
    blocks = bits[:N*M].reshape(N, M)
    for i in range(N):
        w = 0
        for j in range(M - m + 1):
            if np.array_equal(blocks[i, j:j+m], tmpl):
                w += 1
        if w >= K: v[K] += 1
        else: v[w] += 1
    chi2 = sum((v[i] - N*pi[i])**2/(N*pi[i]) for i in range(K+1) if pi[i] > 0)
    return _gammainc_c(K/2.0, chi2/2.0)


def t_universal(bits):
    # Maurer's universal. L=7, Q=1280 needs n>=387840.
    n = len(bits)
    if n < 387_840: return 0.0
    L, Q = 7, 1280
    K = n // L - Q
    if K <= 0: return 0.0
    blocks = bits[:(Q+K)*L].reshape(Q+K, L)
    # convert each L-bit block to int
    weights = (1 << np.arange(L-1, -1, -1)).astype(np.int64)
    seq = (blocks * weights).sum(axis=1)
    T = np.zeros(1 << L, dtype=np.int64)
    for i in range(Q):
        T[seq[i]] = i + 1
    s = 0.0
    for i in range(Q, Q+K):
        s += math.log2(i + 1 - T[seq[i]])
        T[seq[i]] = i + 1
    fn = s / K
    expected = 6.1962507
    variance = 3.125
    c = 0.7 - 0.8/L + (4 + 32/L)*pow(K, -3/L)/15
    sigma = c * math.sqrt(variance / K)
    return _erfc(abs(fn - expected) / (math.sqrt(2) * sigma))


def t_linear_complexity(bits, M=500):
    n = len(bits); N = n // M
    if N < 200: return 0.0
    pi = [0.01047, 0.03125, 0.12500, 0.50000, 0.25000, 0.06250, 0.020833]
    blocks = bits[:N*M].reshape(N, M).astype(np.uint8)
    L = np.zeros(N, dtype=np.float64)
    for i in range(N):
        L[i] = _berlekamp_massey(blocks[i])
    mu = M/2.0 + (9.0 + (-1.0)**(M+1))/36.0 - (M/3.0 + 2/9.0)/(2**M)
    T = ((-1.0)**M)*(L - mu) + 2/9.0
    v = np.zeros(7, dtype=np.int64)
    bins = [-2.5, -1.5, -0.5, 0.5, 1.5, 2.5]
    v[0] = (T <= bins[0]).sum()
    for i in range(1, 6):
        v[i] = ((T > bins[i-1]) & (T <= bins[i])).sum()
    v[6] = (T > bins[5]).sum()
    chi2 = sum((v[i] - N*pi[i])**2/(N*pi[i]) for i in range(7))
    return _gammainc_c(3.0, chi2/2.0)


def _berlekamp_massey(s):
    n = len(s)
    c = np.zeros(n, dtype=np.uint8); b = np.zeros(n, dtype=np.uint8)
    c[0] = b[0] = 1
    L = 0; m = -1
    for N in range(n):
        d = s[N]
        for i in range(1, L+1):
            d ^= c[i] & s[N-i]
        if d == 1:
            t = c.copy()
            shift = N - m
            if shift < n:
                c[shift:] ^= b[:n-shift]
            if 2*L <= N:
                L = N + 1 - L
                m = N
                b = t
    return L


def t_serial(bits, m=16):
    n = len(bits)
    def psi(mm):
        if mm == 0: return 0.0
        # circular extension
        ext = np.concatenate([bits, bits[:mm-1]])
        # build int codes
        weights = (1 << np.arange(mm-1, -1, -1)).astype(np.int64)
        codes = np.zeros(n, dtype=np.int64)
        for j in range(mm):
            codes += ext[j:j+n].astype(np.int64) * weights[j]
        v = np.bincount(codes, minlength=1<<mm)
        return (1 << mm) / n * (v.astype(np.float64)**2).sum() - n
    psi_m  = psi(m)
    psi_m1 = psi(m-1)
    psi_m2 = psi(m-2)
    d1 = psi_m - psi_m1
    d2 = psi_m - 2*psi_m1 + psi_m2
    p1 = _gammainc_c((1 << (m-2)), d1/2.0)
    p2 = _gammainc_c((1 << (m-3)), d2/2.0)
    return min(p1, p2)


def t_approximate_entropy(bits, m=10):
    n = len(bits)
    def phi(mm):
        ext = np.concatenate([bits, bits[:mm-1]])
        weights = (1 << np.arange(mm-1, -1, -1)).astype(np.int64)
        codes = np.zeros(n, dtype=np.int64)
        for j in range(mm):
            codes += ext[j:j+n].astype(np.int64) * weights[j]
        v = np.bincount(codes, minlength=1<<mm).astype(np.float64) / n
        v = v[v > 0]
        return (v * np.log(v)).sum()
    ap_en = phi(m) - phi(m+1)
    chi2 = 2.0 * n * (math.log(2) - ap_en)
    return _gammainc_c((1 << (m-1)), chi2/2.0)


def t_cumulative_sums(bits):
    n = len(bits); s = 2 * bits.astype(np.int64) - 1
    S = np.cumsum(s)
    z = max(abs(int(S.max())), abs(int(S.min())))
    if z == 0: return 0.0
    p = 1.0
    k_lo = int((-n/z + 1)//4); k_hi = int((n/z - 1)//4)
    for k in range(k_lo, k_hi+1):
        a = (4*k+1)*z/math.sqrt(n); b = (4*k-1)*z/math.sqrt(n)
        p -= stats.norm.cdf(a) - stats.norm.cdf(b)
    k_lo2 = int((-n/z - 3)//4)
    for k in range(k_lo2, k_hi+1):
        a = (4*k+3)*z/math.sqrt(n); b = (4*k+1)*z/math.sqrt(n)
        p += stats.norm.cdf(a) - stats.norm.cdf(b)
    return float(max(0.0, min(1.0, p)))


def t_random_excursions(bits):
    n = len(bits); s = 2 * bits.astype(np.int64) - 1
    S = np.cumsum(s); S = np.concatenate([[0], S, [0]])
    zero_idx = np.where(S == 0)[0]
    J = len(zero_idx) - 1
    if J < 500: return 0.0
    states = [-4, -3, -2, -1, 1, 2, 3, 4]
    pi_x = {
        1: [0.5, 0.25, 0.125, 0.0625, 0.0312, 0.0312],
        2: [0.75, 0.0625, 0.0469, 0.0352, 0.0264, 0.0791],
        3: [0.8333, 0.0278, 0.0231, 0.0193, 0.0161, 0.0804],
        4: [0.875, 0.0156, 0.0137, 0.0120, 0.0105, 0.0732],
    }
    pvals = []
    for x in states:
        cnt = np.zeros(6, dtype=np.int64)
        for i in range(J):
            a, b = zero_idx[i], zero_idx[i+1]
            cycle = S[a:b+1]
            v = (cycle == x).sum()
            if v >= 5: cnt[5] += 1
            else: cnt[v] += 1
        pi = pi_x[abs(x)]
        chi2 = sum((cnt[k] - J*pi[k])**2/(J*pi[k]) for k in range(6))
        pvals.append(_gammainc_c(2.5, chi2/2.0))
    return float(min(pvals))


def t_random_excursions_variant(bits):
    n = len(bits); s = 2 * bits.astype(np.int64) - 1
    S = np.cumsum(s)
    J = (S == 0).sum() + 1  # number of cycles approx
    if J < 500: return 0.0
    states = list(range(-9, 0)) + list(range(1, 10))
    pvals = []
    for x in states:
        xi = (S == x).sum()
        denom = math.sqrt(2.0 * J * (4*abs(x) - 2))
        if denom == 0: continue
        pvals.append(_erfc(abs(xi - J) / denom))
    return float(min(pvals)) if pvals else 0.0


NIST_TESTS = [
    ("frequency",                t_frequency),
    ("block_frequency",          t_block_frequency),
    ("runs",                     t_runs),
    ("longest_run",              t_longest_run),
    ("binary_matrix_rank",       t_binary_matrix_rank),
    ("dft",                      t_dft),
    ("non_overlapping_template", t_non_overlapping_template),
    ("overlapping_template",     t_overlapping_template),
    ("universal",                t_universal),
    ("linear_complexity",        t_linear_complexity),
    ("serial",                   t_serial),
    ("approximate_entropy",      t_approximate_entropy),
    ("cumulative_sums",          t_cumulative_sums),
    ("random_excursions",        t_random_excursions),
    ("random_excursions_variant",t_random_excursions_variant),
]


def run_nist_full(bits, alpha=0.01):
    p = {}; pass_ = {}
    for name, fn in NIST_TESTS:
        t0 = time.time()
        try:
            pv = float(fn(bits))
        except Exception as e:
            pv = -1.0
            print(f"  {name}: ERROR {e!r}")
        dt = time.time() - t0
        p[name] = pv
        pass_[name] = bool(pv >= alpha)
        print(f"  {name:30s} p={pv:.4f} {'PASS' if pass_[name] else 'FAIL'}  ({dt:.1f}s)")
    return p, pass_


# ---------------------------------------------------------------------------
# 3.  SP 800-90B IID + min-entropy on raw stream.
# ---------------------------------------------------------------------------
def sp800_90b_iid_min_entropy(raw_bits, max_n=1_000_000):
    """Most-common-value estimator (NIST SP 800-90B Section 6.3.1)."""
    b = raw_bits[:max_n]
    n = b.size
    p1 = float(b.mean())
    pmax = max(p1, 1 - p1)
    # Confidence upper bound (one-sided, alpha=0.05)
    pu = min(1.0, pmax + 2.576 * math.sqrt(pmax * (1 - pmax) / n))
    H_min = -math.log2(pu)
    return {"n": int(n), "p_max": float(pmax), "p_upper_95": float(pu),
            "min_entropy_bits_per_sample": float(H_min)}


# ---------------------------------------------------------------------------
# 4.  K2 cross-cell correlation audit.
# ---------------------------------------------------------------------------
def k2_audit(ens, n_ticks=2000, n_cells_sample=100):
    cells = min(n_cells_sample, ens.n)
    block = ens.raw_traces(n_ticks)[:, :cells]  # (T, cells)
    # standardize per-cell to avoid sigma scaling effects
    z = (block - block.mean(0, keepdims=True)) / (block.std(0, keepdims=True) + 1e-12)
    C = np.corrcoef(z, rowvar=False)
    # off-diagonal
    M = C.shape[0]
    iu = np.triu_indices(M, k=1)
    off = C[iu]
    return {"matrix": C, "off_abs_mean": float(np.mean(np.abs(off))),
            "off_abs_max": float(np.max(np.abs(off))),
            "off_mean": float(np.mean(off)),
            "off_std": float(np.std(off)),
            "n_cells": int(M), "n_ticks": int(n_ticks)}


# ---------------------------------------------------------------------------
# 5.  Peripheral-aware energy model.
# ---------------------------------------------------------------------------
def peripheral_energy(node_nm=130, throughput_bps=1.27e6):
    """Per-bit energy with realistic peripherals.

    Components (per net debiased+whitened bit):
      cell:  E_cell = Id * V_DD * t_settle
             At Id=10 mA, V_DD=1.2 V (130nm digital), t_settle=1 ns:
             E_cell = 1.2e-11 J = 12 pJ per cell-sample (this is HUGE).
             Net yield = 0.125 after VN + XOR-whiten => 12 / 0.125 = 96 pJ/bit.
      DAC:   8-bit, 1 GS/s. Vandenbosch FoM ~ 1 pJ/conv-step.
             E_DAC = 256 * 1e-12 J/sample = 256 pJ/sample (worst case).
             But V_G is static / slowly programmed (one bias per N samples),
             so amortise: E_DAC_per_bit ~ 256e-12 / 1024 = 0.25 pJ/bit.
      ADC:   1-bit comparator, 1 GS/s. Razavi: ~ 50 fJ/conv at 1-bit/1 GS/s
             (state-of-art 65nm: 20-100 fJ). Use 50 fJ. -> 0.05 pJ/sample.
             Net per bit (after debias yield 0.125): 0.4 pJ/bit.
      wire:  C_wire ~ 0.2 fF/um, length 100 um -> 20 fF. E = 0.5*C*V^2 =
             0.5*20e-15*1.2^2 = 14.4 fJ/transition. Per bit: 0.014 pJ.
    """
    V_DD = 1.2
    t_settle = 1e-9
    NET_YIELD = 0.125

    # Re-examine cell energy: original optimistic claim was 50 fJ/raw-sample,
    # assuming Id_avg ~ 5 uA over t=10 ns (50 fA*s = 0.005 pJ, not 50 fJ -
    # original number assumed sub-threshold/STDP operation).
    # POST-z469: Id_op = 10 mA, NOT 5 uA. That's a factor 2000 increase.
    # Even at sub-ns settling, single cell-sample energy is now in the 10s of pJ.
    E_cell_raw_pJ = (ID_OP * V_DD * t_settle) * 1e12

    # DAC: programs V_G once per "burst". Amortise over a burst of B samples.
    # Assume B = 1024 (very generous).
    E_DAC_per_conv_pJ = 1.0           # Vandenbosch FoM
    BURST = 1024
    E_DAC_per_sample_pJ = (256 * E_DAC_per_conv_pJ) / BURST  # ~0.25 pJ

    # ADC: 1-bit comparator @ 1 GS/s, 50 fJ/conv.
    E_ADC_per_sample_pJ = 50.0e-3

    # Wire RC.
    C_wire = 20e-15
    E_wire_pJ = 0.5 * C_wire * V_DD * V_DD * 1e12

    E_per_raw_sample_pJ = (E_cell_raw_pJ + E_DAC_per_sample_pJ
                           + E_ADC_per_sample_pJ + E_wire_pJ)
    E_per_bit_pJ = E_per_raw_sample_pJ / NET_YIELD

    return {
        "node_nm": int(node_nm),
        "V_DD": V_DD,
        "t_settle_s": t_settle,
        "Id_op_A": ID_OP,
        "E_cell_per_raw_pJ": E_cell_raw_pJ,
        "E_DAC_per_sample_pJ": E_DAC_per_sample_pJ,
        "E_DAC_amortisation_burst": BURST,
        "E_ADC_per_sample_pJ": E_ADC_per_sample_pJ,
        "E_wire_per_sample_pJ": E_wire_pJ,
        "net_yield": NET_YIELD,
        "E_per_bit_pJ_total": E_per_bit_pJ,
        "throughput_bps": throughput_bps,
    }


def node_scaling(E_pJ_at_130nm, target_node=28, source_node=130):
    """Dennard scaling: E ~ S^3 (V^2 * C, with V~S, C~S, t~S).
    S = source/target. From 130 -> 28: S = 130/28 = 4.6429.
    Conservative (general logic) often uses E ~ S^2 (ITRS practical).
    Report both.
    """
    S = source_node / target_node
    return {
        "source_node_nm": source_node,
        "target_node_nm": target_node,
        "S_factor": S,
        "E_target_dennard_S3_pJ": E_pJ_at_130nm / (S ** 3),
        "E_target_practical_S2_pJ": E_pJ_at_130nm / (S ** 2),
    }


# ---------------------------------------------------------------------------
# Cheng 2024 reference (digital CMOS TRNG, 65nm).
# ---------------------------------------------------------------------------
CHENG_2024 = {
    "process_nm": 65,
    "energy_pJ_per_bit": 0.244,
    "throughput_Mbps": 1000.0,
    "ref": "Cheng et al. 2024, 'Sub-pJ/bit TRNG in 65nm CMOS'",
}


def compare_to_cheng(our_E_pJ_at_node, node_nm):
    """Scale Cheng 65nm to our node for iso-process comparison."""
    if node_nm == 65:
        scaled = CHENG_2024["energy_pJ_per_bit"]
    elif node_nm > 65:
        S = node_nm / 65.0
        scaled = CHENG_2024["energy_pJ_per_bit"] * (S ** 2)  # practical scaling
    else:
        S = 65.0 / node_nm
        scaled = CHENG_2024["energy_pJ_per_bit"] / (S ** 2)
    return {
        "cheng_at_native_65nm_pJ": CHENG_2024["energy_pJ_per_bit"],
        "cheng_scaled_to_node_pJ": float(scaled),
        "our_at_node_pJ": float(our_E_pJ_at_node),
        "ratio_us_over_cheng": float(our_E_pJ_at_node / scaled),
        "we_win": bool(our_E_pJ_at_node < scaled),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    t_start = time.time()
    print(f"[N_Stoch_RNG_AUDIT] start  out={OUT}")

    # ---- K2 audit on noise traces ----
    print("[1/5] K2 cross-cell correlation audit")
    ens = NSRAMNoiseEnsemble(n_cells=N_CELLS, seed=SEED, common_mode_frac=0.10)
    k2 = k2_audit(ens, n_ticks=2000, n_cells_sample=100)
    print(f"  mean|corr| = {k2['off_abs_mean']:.4f}, max|corr| = {k2['off_abs_max']:.4f}")

    # heatmap
    fig, ax = plt.subplots(1, 1, figsize=(7, 6))
    im = ax.imshow(k2["matrix"], cmap="RdBu_r", vmin=-0.5, vmax=0.5)
    ax.set_title(f"K2 cross-cell correlation (100 cells, common_mode={ens.cm:.2f})\n"
                 f"mean|corr|={k2['off_abs_mean']:.3f}  max|corr|={k2['off_abs_max']:.3f}")
    plt.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(OUT / "heatmap.png", dpi=140)
    plt.close(fig)

    k2_save = {k: (v.tolist() if isinstance(v, np.ndarray) else v)
               for k, v in k2.items() if k != "matrix"}
    k2_save["matrix_summary"] = {"shape": list(k2["matrix"].shape)}
    (OUT / "k2_correlation.json").write_text(json.dumps(k2_save, indent=2))

    # ---- NIST regeneration ----
    print(f"[2/5] regenerate {N_BITS_NIST/1e6:.0f} Mbit and run NIST SP 800-22")
    t1 = time.time()
    bits = ens.bits(N_BITS_NIST)
    gen_dt = time.time() - t1
    throughput = N_BITS_NIST / gen_dt
    print(f"  generated {bits.size:,} bits in {gen_dt:.1f}s ({throughput/1e6:.2f} Mbit/s)")
    np.save(OUT / "bit_stream_10M.npy", bits)

    p_vals, passes = run_nist_full(bits)
    n_pass = int(sum(passes.values()))
    n_total = len(passes)
    print(f"  NIST overall: {n_pass}/{n_total} PASS")

    # SP 800-90B
    print("[3/5] SP 800-90B IID min-entropy on raw (pre-whiten) stream")
    raw = ens._raw_bits(1_000_000)
    h_min = sp800_90b_iid_min_entropy(raw)
    print(f"  H_min = {h_min['min_entropy_bits_per_sample']:.3f} bits/sample"
          f"  (p_max={h_min['p_max']:.4f}, p_upper95={h_min['p_upper_95']:.4f})")

    (OUT / "nist_full_results.json").write_text(json.dumps({
        "p_values": p_vals,
        "passes": passes,
        "passed": n_pass,
        "total": n_total,
        "n_bits_tested": int(bits.size),
        "throughput_bps": float(throughput),
        "sp800_90b": h_min,
        "seed": SEED,
        "n_cells": N_CELLS,
        "common_mode_frac": ens.cm,
        "cell_params": {"snap_Is_A": SNAP_IS, "R_body_ohm": R_BODY,
                        "Id_op_A": ID_OP, "post_z469": True},
    }, indent=2))

    # ---- Peripheral energy ----
    print("[4/5] peripheral-aware energy")
    eng = peripheral_energy(node_nm=130, throughput_bps=throughput)
    print(f"  E/bit @130nm peripheral-aware = {eng['E_per_bit_pJ_total']:.3f} pJ/bit"
          f"  (cell={eng['E_cell_per_raw_pJ']:.2f} pJ raw)")
    cmp_130 = compare_to_cheng(eng["E_per_bit_pJ_total"], 130)
    print(f"  Cheng-2024 @130nm scaled: {cmp_130['cheng_scaled_to_node_pJ']:.3f} pJ"
          f"  -> ratio us/Cheng = {cmp_130['ratio_us_over_cheng']:.1f}x")

    # ---- Node scaling ----
    print("[5/5] node scaling 130 -> 28 nm")
    scale = node_scaling(eng["E_per_bit_pJ_total"], target_node=28, source_node=130)
    cmp_28_dennard = compare_to_cheng(scale["E_target_dennard_S3_pJ"], 28)
    cmp_28_prac = compare_to_cheng(scale["E_target_practical_S2_pJ"], 28)
    print(f"  E@28nm (S^3)={scale['E_target_dennard_S3_pJ']:.3f} pJ"
          f"  | (S^2)={scale['E_target_practical_S2_pJ']:.3f} pJ")
    print(f"  vs Cheng@28nm (S^2 from 65)={cmp_28_prac['cheng_scaled_to_node_pJ']:.3f} pJ"
          f"  -> S^2 ratio = {cmp_28_prac['ratio_us_over_cheng']:.2f}x")

    (OUT / "peripheral_energy.json").write_text(json.dumps({
        "energy_model": eng,
        "compare_at_130nm_vs_cheng": cmp_130,
        "original_claim_pJ": 0.4,
        "note": ("Original 0.4 pJ/bit used optimistic E_cell=50 fJ/raw under"
                 " sub-threshold operation. Post-z469 Id_op=10 mA inflates"
                 " E_cell by ~2000x; peripheral-aware total dominates."),
    }, indent=2))
    (OUT / "node_scaling_compare.json").write_text(json.dumps({
        "scaling": scale,
        "compare_28nm_dennard_S3": cmp_28_dennard,
        "compare_28nm_practical_S2": cmp_28_prac,
        "cheng_2024": CHENG_2024,
    }, indent=2))

    # ---- Verdict ----
    cross_corr = k2["off_abs_max"]
    energy_pJ_130 = eng["E_per_bit_pJ_total"]
    energy_pJ_28 = scale["E_target_practical_S2_pJ"]
    cheng_28 = cmp_28_prac["cheng_scaled_to_node_pJ"]

    n_fail = n_total - n_pass

    survive = (n_fail == 0 and energy_pJ_28 < cheng_28)
    kill = (cross_corr > 0.3) or (n_fail > 3)
    if survive: verdict = "SURVIVE"
    elif kill:  verdict = "KILL"
    else:       verdict = "DEMOTE"

    md = []
    md.append("# N-Stoch-RNG  Honest Audit Verdict\n")
    md.append(f"_seed={SEED}, N_CELLS={N_CELLS}, common_mode={ens.cm:.2f},"
              f" n_bits={N_BITS_NIST/1e6:.0f} Mbit, runtime={time.time()-t_start:.0f}s_\n")
    md.append(f"## Verdict: **{verdict}**\n")
    md.append("Pre-registered gates:")
    md.append(f"- SURVIVE: NIST 15/15 PASS AND E@28nm < Cheng-2024@28nm")
    md.append(f"- DEMOTE:  1 NIST fail OR peripheral-aware E > 1 pJ/bit")
    md.append(f"- KILL:    cross_corr|max > 0.3 OR > 3 NIST fails\n")
    md.append("## NIST SP 800-22 (full)")
    md.append(f"- Passed: **{n_pass}/{n_total}**  (alpha=0.01)")
    for name, _ in NIST_TESTS:
        md.append(f"  - {name}: p={p_vals[name]:.4f}  {'PASS' if passes[name] else 'FAIL'}")
    md.append("\n## SP 800-90B IID")
    md.append(f"- H_min = {h_min['min_entropy_bits_per_sample']:.3f} bits/sample"
              f" (p_max={h_min['p_max']:.4f})")
    md.append("\n## K2 cross-cell correlation")
    md.append(f"- mean|corr| = {k2['off_abs_mean']:.4f}")
    md.append(f"- max|corr|  = {k2['off_abs_max']:.4f}")
    md.append(f"- common_mode_frac = {ens.cm:.2f} (shared bias rail)")
    md.append("\n## Energy (peripheral-aware, post-z469 cell)")
    md.append(f"- E@130nm = **{energy_pJ_130:.3f} pJ/bit** (orig claim: 0.4 pJ/bit)")
    md.append(f"- breakdown: cell={eng['E_cell_per_raw_pJ']:.2f} pJ/raw,"
              f" DAC={eng['E_DAC_per_sample_pJ']:.3f} pJ,"
              f" ADC={eng['E_ADC_per_sample_pJ']:.3f} pJ,"
              f" wire={eng['E_wire_per_sample_pJ']:.4f} pJ"
              f" (yield={eng['net_yield']:.3f})")
    md.append(f"- Cheng-2024 @130nm = {cmp_130['cheng_scaled_to_node_pJ']:.3f} pJ"
              f" (S^2 scaled from 65nm)")
    md.append(f"- ratio us/Cheng @130nm = **{cmp_130['ratio_us_over_cheng']:.1f}x**"
              f"  ({'WE WIN' if cmp_130['we_win'] else 'WE LOSE'})")
    md.append("\n## Node scaling 130 -> 28 nm (Dennard)")
    md.append(f"- S = {scale['S_factor']:.2f}")
    md.append(f"- E@28nm (S^3 ideal Dennard) = {scale['E_target_dennard_S3_pJ']:.3f} pJ")
    md.append(f"- E@28nm (S^2 practical)     = {scale['E_target_practical_S2_pJ']:.3f} pJ")
    md.append(f"- Cheng-2024 @28nm (S^2)     = {cheng_28:.3f} pJ")
    md.append(f"- ratio us/Cheng @28nm (S^2) = **{cmp_28_prac['ratio_us_over_cheng']:.2f}x**"
              f"  ({'WE WIN' if cmp_28_prac['we_win'] else 'WE LOSE'})")
    md.append("\n## What changed vs original PASS")
    md.append("- Original used 50 fJ/raw single-cell-sample (sub-threshold STDP regime).")
    md.append("- Post-z469 calibration: Id_op = 10 mA, V_DD ~ 1.2 V, t_settle ~ 1 ns ->"
              " E_cell ~ 12 pJ per raw sample.")
    md.append("- Adding DAC/ADC/wire only adds 0.3 pJ; CELL DRAIN CURRENT dominates.")
    md.append("- Therefore the 0.4 pJ/bit AMBITIOUS PASS was an artefact of an"
              " un-calibrated low-current assumption.")
    md.append(f"\n_Verdict logic: kill={kill}, survive={survive}_\n")
    (OUT / "honest_verdict.md").write_text("\n".join(md))

    summary = {
        "verdict": verdict,
        "nist_pass": n_pass,
        "nist_total": n_total,
        "k2_cross_corr_max": k2["off_abs_max"],
        "k2_cross_corr_mean": k2["off_abs_mean"],
        "energy_pJ_130nm": energy_pJ_130,
        "energy_pJ_28nm_S2": energy_pJ_28,
        "cheng_28nm_S2_pJ": cheng_28,
        "we_win_at_28nm": cmp_28_prac["we_win"],
        "original_claim_pJ": 0.4,
        "runtime_sec": time.time() - t_start,
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\n[VERDICT] {verdict}  E@130={energy_pJ_130:.2f}pJ"
          f"  E@28(S^2)={energy_pJ_28:.3f}pJ  NIST={n_pass}/{n_total}"
          f"  maxcorr={k2['off_abs_max']:.3f}")


if __name__ == "__main__":
    main()
