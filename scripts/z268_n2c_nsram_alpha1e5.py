#!/usr/bin/env python3
"""
z268 - N2c: NS-RAM 4D body-state TRANSIENT surrogate at CORRECTED
timescale rescale alpha = 1e5 (was 1e3 in z263 N2b).

Oracle O42 (gemini, grok, openai) unanimously read Sebas slide 19
timescale label as "1e5" / "10^5", not 10^3.

Only delta vs N2b:
    dt = 1e-5 s  (was 1e-7 in N2b)   -- T*dt = 1 ms (was 10 us)
    T  = 100 (unchanged)
ALL other params LOCKED, identical to N2b.

Hypothesis: at correct alpha=1e5 the integration window is 100x wider
than N2b, V_b should drift ~100x more, exercise surrogate V_b axis
[0, 0.7] more fully. May unlock transient mode that was structurally
invisible at alpha=1e3. Watch vb_rail_frac: if > 50% V_b is railing,
the corrected rescale has overshot in the opposite direction.

PRE-REGISTERED GATES (logged 2026-05-11 in research_plan/01_LOG.md
BEFORE running):
    conservative_vs_poisson : mean >= 82.65 %
    ambitious_vs_poisson    : mean > 84.65 % AND CI non-overlap
    unlock_vs_n2b           : mean > 78.37 % AND CI non-overlap (N2b
                              alpha=1e3)
    unlock_vs_static        : mean > 79.27 % AND CI non-overlap (N2)

Baselines:
    N1b Poisson      84.65 %
    N2 static NSRAM  79.27 %
    N2b alpha=1e3    78.37 %

5 seeds, 10k train / 2k test subsample (same as N2b).
"""
from __future__ import annotations

import os
os.environ.setdefault("HSA_OVERRIDE_GFX_VERSION", "11.0.0")

import importlib.util
import json
import time
from pathlib import Path

import numpy as np
import torch  # noqa: F401

ROOT = Path(__file__).resolve().parent.parent

# Reuse N1b's loader / thermal / ridge / bootstrap
spec_z261 = importlib.util.spec_from_file_location(
    "z261_n1b_lif_mnist", ROOT / "scripts/z261_n1b_lif_mnist.py"
)
z261 = importlib.util.module_from_spec(spec_z261)
spec_z261.loader.exec_module(z261)

# Reuse N2b's surrogate + featurizer; we only need to override DT.
spec_z263 = importlib.util.spec_from_file_location(
    "z263_n2b_nsram_transient_snn", ROOT / "scripts/z263_n2b_nsram_transient_snn.py"
)
z263 = importlib.util.module_from_spec(spec_z263)
spec_z263.loader.exec_module(z263)

# ------------------------------------------------------------------
# CONFIG - locked. Only DT changes vs N2b.
# ------------------------------------------------------------------
ALPHA_CORRECTED = 1e5     # oracle O42 slide-19 re-read
DT_TRANS_NEW = 1e-5       # 10 us per Poisson step (was 1e-7 = 0.1 us)
T_STEPS = 100             # unchanged

# Patch the N2b module's DT in place. T_STEPS unchanged.
z263.DT_TRANS = DT_TRANS_NEW

N_NEURONS = z263.N_NEURONS
N_SEEDS = z263.N_SEEDS
SEEDS = z263.SEEDS

BASELINE_POISSON = 0.8465
BASELINE_STATIC  = 0.7927
BASELINE_N2B     = 0.7837
PASS_CONSERVATIVE = 0.8265

# CI approximations (from prior summaries)
POISSON_CI = (0.8378, 0.8543)
STATIC_CI  = (0.7851, 0.7998)
# N2b CI approx (std~=0.0050 over 5 seeds, mean 0.7837): +-1.96*sd/sqrt(5)
N2B_CI     = (0.7793, 0.7881)

RESULTS_DIR = ROOT / "results/z268_n2c_nsram_alpha1e5"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def main():
    print(f"torch {torch.__version__}  cuda_avail={torch.cuda.is_available()}",
          flush=True)
    print(f"[N2c] CORRECTED alpha={ALPHA_CORRECTED:.0e}  dt={DT_TRANS_NEW:.0e}s  "
          f"T={T_STEPS}  -> T*dt={T_STEPS*DT_TRANS_NEW:.2e}s", flush=True)
    print("[loader] loading MNIST 28x28...", flush=True)
    X_train, y_train, X_test, y_test, dataset = z261.load_mnist()
    print(f"dataset: {dataset}  train={X_train.shape}  test={X_test.shape}",
          flush=True)

    # Same deterministic subsample as N2b (same seed 424242)
    N_TRAIN_SUB = z263.N_TRAIN_SUB
    N_TEST_SUB  = z263.N_TEST_SUB
    rng_sub = np.random.RandomState(424242)
    tr_idx = rng_sub.choice(X_train.shape[0], size=N_TRAIN_SUB, replace=False)
    te_idx = rng_sub.choice(X_test.shape[0],  size=N_TEST_SUB,  replace=False)
    X_train_use = X_train[tr_idx]; y_train_use = y_train[tr_idx]
    X_test_use  = X_test[te_idx];  y_test_use  = y_test[te_idx]
    sub_info = {
        "subsampled": True,
        "n_train_used": N_TRAIN_SUB,
        "n_test_used":  N_TEST_SUB,
        "subsample_seed": 424242,
    }
    print(f"[subsample] {sub_info}", flush=True)

    t_global = time.time()
    thermal_peak = z261.apu_temp_c()
    per_seed = []
    for s in SEEDS:
        r = z263.run_seed(s, X_train_use, y_train_use, X_test_use, y_test_use)
        per_seed.append(r)
        thermal_peak = max(thermal_peak, z261.apu_temp_c())
        z261.thermal_guard()

    accs = np.array([r["test_acc"] for r in per_seed])
    mean = float(accs.mean())
    std  = float(accs.std(ddof=1))
    ci   = z261.bootstrap_ci(accs)
    lo, hi = ci

    delta_vs_poisson_pp = (mean - BASELINE_POISSON) * 100.0
    delta_vs_static_pp  = (mean - BASELINE_STATIC)  * 100.0
    delta_vs_n2b_pp     = (mean - BASELINE_N2B)     * 100.0

    non_overlap_above_poisson = lo > POISSON_CI[1]
    non_overlap_above_static  = lo > STATIC_CI[1]
    non_overlap_above_n2b     = lo > N2B_CI[1]

    verdict_conservative_vs_poisson = "PASS" if mean >= PASS_CONSERVATIVE else "FAIL"
    verdict_ambitious_vs_poisson    = (
        "PASS" if (mean > BASELINE_POISSON and non_overlap_above_poisson)
        else "FAIL"
    )
    verdict_unlock_vs_n2b = (
        "PASS" if (mean > BASELINE_N2B and non_overlap_above_n2b) else "FAIL"
    )
    verdict_unlock_vs_static = (
        "PASS" if (mean > BASELINE_STATIC and non_overlap_above_static) else "FAIL"
    )

    clip_rate_overall = float(np.mean([r["clip_rate"] for r in per_seed]))
    vb_rail_overall   = float(np.mean([0.5*(r["vb_rail_frac_train"]
                                            + r["vb_rail_frac_test"])
                                       for r in per_seed]))

    wall = time.time() - t_global

    summary = {
        "experiment": "z268_n2c_nsram_alpha1e5",
        "date": "2026-05-11",
        "dataset": dataset,
        "n_train_full": int(X_train.shape[0]),
        "n_test_full":  int(X_test.shape[0]),
        "subsample": sub_info,
        "config": {
            "N_neurons": N_NEURONS,
            "VG1_bias_range": [z263.VG1_BIAS_LO, z263.VG1_BIAS_HI],
            "VG2_bias_range": [z263.VG2_BIAS_LO, z263.VG2_BIAS_HI],
            "g_in": z263.G_IN,
            "C_b_F": z263.C_B,
            "dt_s":  DT_TRANS_NEW,
            "alpha_rescale": ALPHA_CORRECTED,
            "alpha_note": "oracle O42 slide-19 unanimous re-read: 1e5 (was 1e3 in N2b)",
            "T_steps": T_STEPS,
            "window_s": T_STEPS * DT_TRANS_NEW,
            "Vd_fixed": z263.VD_FIXED,
            "Vb_init":  z263.VB_INIT,
            "Vb_axis_range": [z263.VB_AXIS_MIN, z263.VB_AXIS_MAX],
            "ridge_lambda": z261.RIDGE_LAMBDA,
            "feature": "log10(mean |I_d| over T steps), z-scored per-unit (train stats)",
            "surrogate": str(z263.SURROGATE_PATH),
        },
        "n_seeds": N_SEEDS,
        "seeds": SEEDS,
        "per_seed": per_seed,
        "test_accuracy_mean": mean,
        "test_accuracy_std":  std,
        "test_accuracy_ci95": list(ci),
        "baseline_n1b_poisson": BASELINE_POISSON,
        "baseline_n2_static":   BASELINE_STATIC,
        "baseline_n2b_alpha1e3": BASELINE_N2B,
        "baseline_poisson_ci_approx": list(POISSON_CI),
        "baseline_static_ci_approx":  list(STATIC_CI),
        "baseline_n2b_ci_approx":     list(N2B_CI),
        "delta_vs_poisson_pp": delta_vs_poisson_pp,
        "delta_vs_static_pp":  delta_vs_static_pp,
        "delta_vs_n2b_pp":     delta_vs_n2b_pp,
        "non_overlap_above_poisson": bool(non_overlap_above_poisson),
        "non_overlap_above_static":  bool(non_overlap_above_static),
        "non_overlap_above_n2b":     bool(non_overlap_above_n2b),
        "ood_clip_rate_mean": clip_rate_overall,
        "vb_rail_frac_mean":  vb_rail_overall,
        "gates": {
            "conservative_vs_poisson": {"threshold": PASS_CONSERVATIVE,
                                        "rule": "mean >= 0.8265"},
            "ambitious_vs_poisson": {"threshold": BASELINE_POISSON,
                                     "rule": "mean > 0.8465 AND CI lo > Poisson CI hi"},
            "unlock_vs_n2b": {"threshold": BASELINE_N2B,
                              "rule": "mean > 0.7837 AND CI lo > N2b CI hi"},
            "unlock_vs_static": {"threshold": BASELINE_STATIC,
                                 "rule": "mean > 0.7927 AND CI lo > Static CI hi"},
        },
        "verdict_conservative_vs_poisson": verdict_conservative_vs_poisson,
        "verdict_ambitious_vs_poisson":    verdict_ambitious_vs_poisson,
        "verdict_unlock_vs_n2b":           verdict_unlock_vs_n2b,
        "verdict_unlock_vs_static":        verdict_unlock_vs_static,
        "wall_s_total": wall,
        "thermal_peak_c": thermal_peak,
    }
    out_path = RESULTS_DIR / "summary.json"
    out_path.write_text(json.dumps(summary, indent=2))

    print("\n" + "=" * 72, flush=True)
    print("z268 N2c - NS-RAM TRANSIENT alpha=1e5  SUMMARY", flush=True)
    print("=" * 72, flush=True)
    print(f"  subsample: {sub_info}", flush=True)
    print(f"  per-seed test_acc: " +
          ", ".join(f"{r['test_acc']:.4f}" for r in per_seed), flush=True)
    print(f"  mean = {mean:.4f} +- {std:.4f}  95% CI = [{lo:.4f}, {hi:.4f}]",
          flush=True)
    print(f"  vs N1b Poisson  ({BASELINE_POISSON:.4f}): "
          f"delta = {delta_vs_poisson_pp:+.2f} pp", flush=True)
    print(f"  vs N2 static    ({BASELINE_STATIC:.4f}): "
          f"delta = {delta_vs_static_pp:+.2f} pp", flush=True)
    print(f"  vs N2b a=1e3    ({BASELINE_N2B:.4f}): "
          f"delta = {delta_vs_n2b_pp:+.2f} pp", flush=True)
    print(f"  OOD clip rate (mean) = {clip_rate_overall:.4f}", flush=True)
    print(f"  Vb rail frac  (mean) = {vb_rail_overall:.4f}", flush=True)
    print(f"  CONSERVATIVE vs Poisson : {verdict_conservative_vs_poisson}", flush=True)
    print(f"  AMBITIOUS    vs Poisson : {verdict_ambitious_vs_poisson}", flush=True)
    print(f"  UNLOCK       vs N2b     : {verdict_unlock_vs_n2b}", flush=True)
    print(f"  UNLOCK       vs Static  : {verdict_unlock_vs_static}", flush=True)
    print(f"  wall = {wall:.1f}s   thermal peak = {thermal_peak:.1f}C", flush=True)
    print(f"  -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
