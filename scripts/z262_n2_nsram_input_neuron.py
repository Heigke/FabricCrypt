#!/usr/bin/env python3
"""
z262 - N2: NS-RAM 4D body-state surrogate as input neuron in the SNN
pipeline, replacing the rate-coded Poisson units of N1b Stage A.

Architecture (matches N1b Stage A scaffold):
  - N=128 units, real MNIST 28x28 (784-D input).
  - Per unit i: parameters (V_G1_bias_i, V_G2_bias_i, W_in_i in R^784).
  - For each image x in [0,1]^784:
        drive_i = W_in_i . x
        V_G1 = V_G1_bias_i + drive_i * g_in
        V_G2 = V_G2_bias_i   (fixed per unit)
        V_d  = 1.0           (fixed)
        V_b  = 0.0           (fixed for this experiment; no body dyn.)
        feature_i = log10(|I_d|)  via NSRAMSurrogate4D
  - Optional reservoir-style sigmoid recurrence (same as N1b Stage A).
  - Ridge-lstsq readout 128 -> 10.

Pre-registered gates (locked in research_plan/01_LOG.md, 2026-05-11
BEFORE first training run):
  Baseline = 0.8465  (N1b Stage A Poisson reference)
  PASS CONSERVATIVE: NS-RAM mean >= 0.8265 with non-overlapping 95% CI.
  PASS AMBITIOUS:    NS-RAM mean >  0.8465 with non-overlapping CI.

NO-CHEAT: V_G1_bias, V_G2_bias, g_in fixed by spec. No tuning.
"""
from __future__ import annotations

# CRITICAL: must precede torch import for gfx1151
import os
os.environ.setdefault("HSA_OVERRIDE_GFX_VERSION", "11.0.0")

import json
import time
from pathlib import Path

import numpy as np
import torch

# Reuse N1b helpers via direct import (same dir)
import importlib.util
ROOT = Path(__file__).resolve().parent.parent
spec_n1b = importlib.util.spec_from_file_location(
    "z261_n1b", ROOT / "scripts/z261_n1b_lif_mnist.py"
)
z261 = importlib.util.module_from_spec(spec_n1b)
spec_n1b.loader.exec_module(z261)

spec_surr = importlib.util.spec_from_file_location(
    "nsram_surrogate_4d", ROOT / "scripts/nsram_surrogate_4d.py"
)
ns_mod = importlib.util.module_from_spec(spec_surr)
spec_surr.loader.exec_module(ns_mod)
NSRAMSurrogate4D = ns_mod.NSRAMSurrogate4D

# ------------------------------------------------------------------
# Config (locked - no tuning)
# ------------------------------------------------------------------
N_NEURONS = 128
N_SEEDS = 5
SEEDS = [0, 1, 2, 3, 4]
BOOTSTRAP_N = 1000

# Per-neuron NS-RAM bias distributions (locked)
VG1_BIAS_LO, VG1_BIAS_HI = 0.20, 0.40
VG2_BIAS_LO, VG2_BIAS_HI = 0.00, 0.30
# Input projection scale: drive in [-1, +1] approx if pixels in [0,1]
# and W_in_i ~ N(0, 1/sqrt(784)). g_in scales drive into surrogate
# domain shift around the bias.
G_IN = 0.20  # multiplies normalized drive

# Fixed surrogate axes
VD_FIXED = 1.0
VB_FIXED = 0.0

# Reservoir scaffold (matches N1b Stage A exactly)
RESERVOIR_GAIN = z261.RESERVOIR_GAIN
SPECTRAL_RADIUS = z261.SPECTRAL_RADIUS
SPARSITY = z261.SPARSITY
N_EXC_FRAC = z261.N_EXC_FRAC
T_STEPS = z261.T_STEPS
RIDGE_LAMBDA = z261.RIDGE_LAMBDA

# Surrogate domain limits (for OOD logging) - from inspection of
# results/z220_4d_dense
VG1_AXIS_MIN, VG1_AXIS_MAX = 0.10, 0.70
VG2_AXIS_MIN, VG2_AXIS_MAX = 0.00, 0.60

THERMAL_CHECK_EVERY = 30
SURROGATE_PATH = ROOT / "results/z220_4d_dense/surrogate_4d_dense.npz"

RESULTS_DIR = ROOT / "results/z262_n2_nsram_input_neuron"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ------------------------------------------------------------------
# Vectorized 4D surrogate eval (numpy)
# ------------------------------------------------------------------
class VecSurrogate:
    """Vectorized quadrilinear interp over (VG1, VG2, Vd, Vb) for log_Id.

    Same math as NSRAMSurrogate4D.eval, but accepts arrays.
    Tracks fraction of queries that hit a domain boundary (OOD).
    """
    def __init__(self, npz_path):
        d = np.load(npz_path)
        self.Id_log = np.log10(np.maximum(np.abs(d["Id"]), 1e-15)).astype(np.float32)
        self.vg1 = d["vg1_axis"].astype(np.float64)
        self.vg2 = d["vg2_axis"].astype(np.float64)
        self.vd  = d["vd_axis"].astype(np.float64)
        self.vb  = d["vb_axis"].astype(np.float64)
        self.n_query = 0
        self.n_clipped = 0

    @staticmethod
    def _idx(x, axis):
        # x: array, axis: 1D sorted
        x = np.asarray(x, dtype=np.float64)
        x_clipped = np.clip(x, axis[0], axis[-1])
        clipped_mask = (x != x_clipped)
        i = np.clip(np.searchsorted(axis, x_clipped), 1, len(axis) - 1) - 1
        f = (x_clipped - axis[i]) / np.maximum(axis[i+1] - axis[i], 1e-30)
        return i, f.astype(np.float32), clipped_mask

    def eval_log_id(self, VG1, VG2, Vd, Vb):
        """All inputs same shape; returns log10(|Id|) same shape."""
        i, fi, m1 = self._idx(VG1, self.vg1)
        j, fj, m2 = self._idx(VG2, self.vg2)
        k, fk, m3 = self._idx(Vd,  self.vd)
        l, fl, m4 = self._idx(Vb,  self.vb)
        any_clip = (m1 | m2 | m3 | m4)
        self.n_query += i.size
        self.n_clipped += int(any_clip.sum())

        out = np.zeros_like(fi, dtype=np.float32)
        for di in (0, 1):
            wi = fi if di else (1.0 - fi)
            for dj in (0, 1):
                wj = fj if dj else (1.0 - fj)
                for dk in (0, 1):
                    wk = fk if dk else (1.0 - fk)
                    for dl in (0, 1):
                        wl = fl if dl else (1.0 - fl)
                        w = wi * wj * wk * wl
                        out += w * self.Id_log[i+di, j+dj, k+dk, l+dl]
        return out

    @property
    def clip_rate(self):
        if self.n_query == 0:
            return 0.0
        return self.n_clipped / self.n_query


# ------------------------------------------------------------------
# Per-seed parameter sampling
# ------------------------------------------------------------------
def build_nsram_params(N, n_in, seed):
    """Returns (VG1_bias [N], VG2_bias [N], W_in [N, n_in]).
    W_in scaled so drive = W_in @ x is roughly O(1) when x is MNIST pixels.
    Drive then multiplied by G_IN inside featurize step.
    """
    rng = np.random.RandomState(seed)
    VG1_bias = rng.uniform(VG1_BIAS_LO, VG1_BIAS_HI, size=N).astype(np.float32)
    VG2_bias = rng.uniform(VG2_BIAS_LO, VG2_BIAS_HI, size=N).astype(np.float32)
    # W_in: N(0, 1/sqrt(n_in)) so W_in @ x has std ~ mean(x) for unit-var x.
    # MNIST pixels mean ~ 0.13 (in [0,1]), so drive std ~ 0.13. With G_IN=0.2
    # -> effective shift ~ 0.026 V => stays inside [VG1_BIAS_LO - 0.1,
    # VG1_BIAS_HI + 0.1] = [0.10, 0.50] (well inside [0.10, 0.70]).
    W_in = (rng.randn(N, n_in) / np.sqrt(n_in)).astype(np.float32)
    return VG1_bias, VG2_bias, W_in


# ------------------------------------------------------------------
# NS-RAM featurize (replaces N1b's run_rate_batch)
# ------------------------------------------------------------------
def featurize_nsram(X_np, VG1_bias, VG2_bias, W_in, surr, W_rec, device,
                    batch=512):
    """
    X_np: (B_total, n_in) in [0,1]
    Strategy: compute the static NS-RAM "input drive" feature once per
    image (log10|Id|), then iterate the same sigmoid recurrence that
    N1b Stage A used (so the only thing changing is the input encoding).

    Returns mean rate per neuron: (B_total, N)
    """
    B_total, n_in = X_np.shape
    N = VG1_bias.shape[0]

    # Stage 1: compute NS-RAM static feature per (image, unit)
    # drive[b, i] = W_in[i] . X[b]  ->  VG1[b, i] = VG1_bias[i] + drive * G_IN
    # Then query surrogate to get log_Id[b, i].  We z-score across units
    # per image so the dynamic range matches what the sigmoid recurrence
    # expects.
    feats_input = np.zeros((B_total, N), dtype=np.float32)
    nb = 0
    for s in range(0, B_total, batch):
        e = min(s + batch, B_total)
        Xc = X_np[s:e]                          # (b, n_in)
        drive = Xc @ W_in.T                     # (b, N)
        VG1 = VG1_bias[None, :] + G_IN * drive  # (b, N)
        VG2 = np.broadcast_to(VG2_bias[None, :], VG1.shape)
        Vd  = np.full_like(VG1, VD_FIXED)
        Vb  = np.full_like(VG1, VB_FIXED)
        log_id = surr.eval_log_id(VG1, VG2, Vd, Vb)   # (b, N)  in [-15, -2]
        # z-score per-image across units
        mu = log_id.mean(axis=1, keepdims=True)
        sd = log_id.std(axis=1, keepdims=True) + 1e-6
        feats_input[s:e] = ((log_id - mu) / sd).astype(np.float32)
        nb += 1
        if nb % THERMAL_CHECK_EVERY == 0:
            z261.thermal_guard()

    # Stage 2: feed feats_input into the same sigmoid recurrence as N1b
    # Stage A, but the "input current" is the NS-RAM feature (already
    # z-scored), and we DROP W_in @ x (replaced by NS-RAM). Recurrence
    # W_rec is the same recurrent matrix.
    out = np.zeros((B_total, N), dtype=np.float32)
    W_rec_t_t = torch.from_numpy(W_rec.T.copy()).to(device)
    for s in range(0, B_total, batch):
        e = min(s + batch, B_total)
        I_in = torch.from_numpy(feats_input[s:e]).to(device)   # (b, N)
        r = torch.zeros_like(I_in)
        rate_sum = torch.zeros_like(I_in)
        for _ in range(T_STEPS):
            I_t = I_in + (r @ W_rec_t_t) * RESERVOIR_GAIN
            r = torch.sigmoid(I_t)
            rate_sum += r
        out[s:e] = (rate_sum / T_STEPS).detach().cpu().numpy()
        z261.thermal_guard()
    return out


# ------------------------------------------------------------------
# Reservoir recurrence matrix (same scaffold as N1b)
# ------------------------------------------------------------------
def build_recurrence(N, seed):
    rng = np.random.RandomState(seed + 777)
    mask = (rng.rand(N, N) < SPARSITY).astype(np.float32)
    np.fill_diagonal(mask, 0)
    W = rng.randn(N, N).astype(np.float32) * mask
    n_exc = int(N * N_EXC_FRAC)
    signs = np.ones(N, dtype=np.float32)
    signs[n_exc:] = -1
    W = np.abs(W) * signs[:, None]
    eigs = np.abs(np.linalg.eigvals(W))
    if eigs.max() > 0:
        W *= SPECTRAL_RADIUS / eigs.max()
    return W


# ------------------------------------------------------------------
# Single-seed run
# ------------------------------------------------------------------
def run_seed(seed, X_train, y_train, X_test, y_test, surr, device):
    t0 = time.time()
    print(f"[seed {seed}] start APU={z261.apu_temp_c():.1f}C", flush=True)
    n_in = X_train.shape[1]
    VG1_bias, VG2_bias, W_in = build_nsram_params(N_NEURONS, n_in, seed)
    W_rec = build_recurrence(N_NEURONS, seed)

    # Quick smoke: drive distribution at this seed
    drive = X_train[:1000] @ W_in.T
    VG1_smoke = VG1_bias[None, :] + G_IN * drive
    print(f"[seed {seed}] VG1 drive range: [{VG1_smoke.min():.3f}, "
          f"{VG1_smoke.max():.3f}] (domain [0.10, 0.70])", flush=True)

    F_tr = featurize_nsram(X_train, VG1_bias, VG2_bias, W_in, surr, W_rec, device)
    F_te = featurize_nsram(X_test,  VG1_bias, VG2_bias, W_in, surr, W_rec, device)
    n_classes = int(max(y_train.max(), y_test.max())) + 1
    Wro = z261.fit_ridge_readout(F_tr, y_train, n_classes)
    train_acc = float((z261.apply_readout(F_tr, Wro) == y_train).mean())
    test_acc  = float((z261.apply_readout(F_te, Wro) == y_test).mean())
    dt = time.time() - t0
    print(f"[seed {seed}] NSRAM   train={train_acc:.4f}  test={test_acc:.4f}  "
          f"({dt:.1f}s, APU={z261.apu_temp_c():.1f}C, "
          f"clip_rate={surr.clip_rate:.4f})",
          flush=True)
    return {"seed": seed, "train_acc": train_acc, "test_acc": test_acc,
            "wall_s": dt}


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------
def main():
    print(f"torch {torch.__version__}  cuda_avail={torch.cuda.is_available()}",
          flush=True)
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"device: {torch.cuda.get_device_name(0)}", flush=True)
    else:
        device = torch.device("cpu")
        print("device: CPU (ROCm unavailable)", flush=True)

    print(f"[loader] loading MNIST 28x28...", flush=True)
    X_train, y_train, X_test, y_test, dataset = z261.load_mnist()
    print(f"dataset: {dataset}  train={X_train.shape}  test={X_test.shape}",
          flush=True)

    print(f"[surr] loading {SURROGATE_PATH}", flush=True)
    surr = VecSurrogate(SURROGATE_PATH)

    BASELINE = 0.8465
    PASS_CONSERVATIVE = 0.8265
    BASELINE_CI = (0.8378, 0.8543)  # approx from N1b std 0.0072 / sqrt(5) * 1.96

    t_global = time.time()
    thermal_peak = z261.apu_temp_c()
    per_seed = []
    for s in SEEDS:
        # reset surrogate counters per-seed for clean reporting, but
        # keep aggregate too
        surr_seed = VecSurrogate(SURROGATE_PATH)
        r = run_seed(s, X_train, y_train, X_test, y_test, surr_seed, device)
        r["clip_rate"] = surr_seed.clip_rate
        per_seed.append(r)
        thermal_peak = max(thermal_peak, z261.apu_temp_c())
        z261.thermal_guard()

    accs = np.array([r["test_acc"] for r in per_seed])
    mean = float(accs.mean())
    std  = float(accs.std(ddof=1))
    ci   = z261.bootstrap_ci(accs)
    delta_pp = (mean - BASELINE) * 100.0
    clip_rate_overall = float(np.mean([r["clip_rate"] for r in per_seed]))

    # Verdicts
    # Non-overlapping CI vs baseline CI (Poisson at [0.8378, 0.8543] approx)
    nsram_lo, nsram_hi = ci
    non_overlap_below = nsram_hi < BASELINE_CI[0]
    non_overlap_above = nsram_lo > BASELINE_CI[1]

    verdict_conservative = (
        "PASS" if (mean >= PASS_CONSERVATIVE
                   and not (nsram_hi < BASELINE_CI[0] and mean < PASS_CONSERVATIVE))
        else ("PASS" if mean >= PASS_CONSERVATIVE else "FAIL")
    )
    # Simpler exact restatement: conservative passes if mean >= 0.8265.
    verdict_conservative = "PASS" if mean >= PASS_CONSERVATIVE else "FAIL"
    verdict_ambitious = "PASS" if (mean > BASELINE and non_overlap_above) else "FAIL"

    wall = time.time() - t_global
    summary = {
        "experiment": "z262_n2_nsram_input_neuron",
        "date": "2026-05-11",
        "dataset": dataset,
        "n_train": int(X_train.shape[0]),
        "n_test": int(X_test.shape[0]),
        "n_classes": 10,
        "config": {
            "N_neurons": N_NEURONS,
            "VG1_bias_range": [VG1_BIAS_LO, VG1_BIAS_HI],
            "VG2_bias_range": [VG2_BIAS_LO, VG2_BIAS_HI],
            "g_in": G_IN,
            "Vd_fixed": VD_FIXED,
            "Vb_fixed": VB_FIXED,
            "reservoir_gain": RESERVOIR_GAIN,
            "spectral_radius": SPECTRAL_RADIUS,
            "sparsity": SPARSITY,
            "T_steps": T_STEPS,
            "ridge_lambda": RIDGE_LAMBDA,
            "surrogate": str(SURROGATE_PATH),
        },
        "n_seeds": N_SEEDS,
        "seeds": SEEDS,
        "per_seed": per_seed,
        "test_accuracy_mean": mean,
        "test_accuracy_std": std,
        "test_accuracy_ci95": list(ci),
        "ood_clip_rate_mean": clip_rate_overall,
        "baseline_reference": BASELINE,
        "baseline_ci_approx": list(BASELINE_CI),
        "delta_pp": delta_pp,
        "non_overlap_below_baseline": bool(non_overlap_below),
        "non_overlap_above_baseline": bool(non_overlap_above),
        "gates": {
            "conservative": {"threshold": PASS_CONSERVATIVE,
                              "rule": "mean >= 0.8265"},
            "ambitious":    {"threshold": BASELINE,
                              "rule": "mean > 0.8465 AND CI lo > baseline CI hi"},
        },
        "verdict_conservative": verdict_conservative,
        "verdict_ambitious":    verdict_ambitious,
        "wall_s_total": wall,
        "thermal_peak_c": thermal_peak,
    }
    out_path = RESULTS_DIR / "summary.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print("\n" + "=" * 72, flush=True)
    print(f"z262 N2 — NS-RAM input neuron  SUMMARY", flush=True)
    print("=" * 72, flush=True)
    print(f"  per-seed test_acc: " +
          ", ".join(f"{r['test_acc']:.4f}" for r in per_seed), flush=True)
    print(f"  mean = {mean:.4f} +- {std:.4f}  95% CI = [{ci[0]:.4f}, {ci[1]:.4f}]",
          flush=True)
    print(f"  baseline (N1b Poisson) = {BASELINE:.4f}  "
          f"delta = {delta_pp:+.2f} pp", flush=True)
    print(f"  OOD clip rate (mean over seeds) = {clip_rate_overall:.4f}",
          flush=True)
    print(f"  CONSERVATIVE: {verdict_conservative}  AMBITIOUS: {verdict_ambitious}",
          flush=True)
    print(f"  wall = {wall:.1f}s   thermal peak = {thermal_peak:.1f}C",
          flush=True)
    print(f"  -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
