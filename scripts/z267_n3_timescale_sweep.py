#!/usr/bin/env python3
"""
z267 - N3: Timescale rescaling consistency check on the N2b NS-RAM transient SNN.

Pre-registered gate (reaffirmed at top of run):
    Sweep alpha in {100, 1000, 10000}. Keep total integration window
    T*dt = 1e-5 s constant. RESCALED dt = dt_base * (alpha / 1000);
    T = T_base * (1000 / alpha).
        alpha = 100   -> dt = 1e-8 s, T = 1000 steps  (fine,  long)
        alpha = 1000  -> dt = 1e-7 s, T = 100  steps  (N2b baseline)
        alpha = 10000 -> dt = 1e-6 s, T = 10   steps  (coarse, short)

    PASS  if max pairwise |delta(test_acc)| <= 2 pp across the three
          alpha conditions (rescaling is BENIGN).
    FAIL  otherwise (timescale is a hidden lever; N2b's FAIL at
          alpha=1000 might be a rescaling artifact).

NO-CHEAT:
    - Do NOT tune C_b, V_G1_bias, V_G2_bias, g_in.
    - Do NOT change the gate.
    - 5 seeds per condition, non-negotiable.
    - Divergence (V_b explode) at extreme alpha is logged honestly and
      excluded from the gate, but still reported.

Reuses N2b infrastructure (z263) entirely; only T and dt vary.
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

# Load N2b module (which itself loads N1b)
spec_z263 = importlib.util.spec_from_file_location(
    "z263_n2b_nsram_transient_snn",
    ROOT / "scripts/z263_n2b_nsram_transient_snn.py",
)
z263 = importlib.util.module_from_spec(spec_z263)
spec_z263.loader.exec_module(z263)
z261 = z263.z261  # N1b loader/ridge/thermal

# ------------------------------------------------------------------
# Config
# ------------------------------------------------------------------
DT_BASE = 1e-7
T_BASE  = 100
ALPHAS = [100, 1000, 10000]   # 10^2, 10^3 (baseline), 10^4
SEEDS = [0, 1, 2, 3, 4]

RESULTS_DIR = ROOT / "results/z267_n3_timescale_sweep"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

GATE_DELTA_PP = 2.0  # PASS if max pairwise delta <= 2 pp
BUDGET_S = 3 * 3600  # 3 h


def alpha_to_dt_T(alpha: int):
    # dt * T must stay at DT_BASE * T_BASE = 1e-5 s
    dt = DT_BASE * (alpha / 1000.0)
    T  = int(round(T_BASE * (1000.0 / alpha)))
    return dt, T


def run_seed_with_timescale(seed, alpha, X_train, y_train, X_test, y_test):
    """Mirror of z263.run_seed, but with dt/T overridden per alpha.

    We monkey-patch z263's module-level DT_TRANS and T_STEPS for the
    duration of this call. featurize_transient reads them as globals.
    """
    dt, T = alpha_to_dt_T(alpha)
    old_dt = z263.DT_TRANS
    old_T  = z263.T_STEPS
    z263.DT_TRANS = dt
    z263.T_STEPS  = T
    try:
        r = z263.run_seed(seed, X_train, y_train, X_test, y_test)
    finally:
        z263.DT_TRANS = old_dt
        z263.T_STEPS  = old_T
    r["alpha"] = alpha
    r["dt_s"] = dt
    r["T_steps"] = T
    return r


def summarize(accs_arr):
    accs = np.asarray(accs_arr, dtype=np.float64)
    mean = float(accs.mean())
    std  = float(accs.std(ddof=1)) if len(accs) > 1 else 0.0
    if len(accs) >= 2:
        ci = z261.bootstrap_ci(accs)
    else:
        ci = (mean, mean)
    return mean, std, (float(ci[0]), float(ci[1]))


def main():
    print(f"torch {torch.__version__}  cuda_avail={torch.cuda.is_available()}",
          flush=True)
    print("=" * 72, flush=True)
    print("z267 N3 - timescale rescaling consistency check", flush=True)
    print("=" * 72, flush=True)
    print("PRE-REGISTERED GATE (reaffirmed):", flush=True)
    print("  Sweep alpha in {100, 1000, 10000}; T*dt = 1e-5 s constant.", flush=True)
    print(f"  PASS if max pairwise |delta(test_acc)| <= {GATE_DELTA_PP:.1f} pp.", flush=True)
    print("  FAIL otherwise.", flush=True)
    print("  NO-CHEAT: 5 seeds each; no parameter tuning; alpha-divergent", flush=True)
    print("  conditions excluded from gate but still reported.", flush=True)
    for a in ALPHAS:
        dt, T = alpha_to_dt_T(a)
        print(f"    alpha={a:>6}  ->  dt={dt:.2e} s, T={T} steps", flush=True)
    print("=" * 72, flush=True)

    # Load MNIST + subsample identically to N2b (seed 424242)
    print("[loader] loading MNIST 28x28...", flush=True)
    X_train, y_train, X_test, y_test, dataset = z261.load_mnist()
    print(f"dataset: {dataset}  train={X_train.shape}  test={X_test.shape}",
          flush=True)
    rng_sub = np.random.RandomState(424242)
    tr_idx = rng_sub.choice(X_train.shape[0], size=z263.N_TRAIN_SUB, replace=False)
    te_idx = rng_sub.choice(X_test.shape[0],  size=z263.N_TEST_SUB,  replace=False)
    X_train_use = X_train[tr_idx]; y_train_use = y_train[tr_idx]
    X_test_use  = X_test[te_idx];  y_test_use  = y_test[te_idx]
    sub_info = {
        "subsampled": True,
        "n_train_used": z263.N_TRAIN_SUB,
        "n_test_used":  z263.N_TEST_SUB,
        "subsample_seed": 424242,
    }
    print(f"[subsample] {sub_info}", flush=True)

    t_global = time.time()
    thermal_peak = z261.apu_temp_c()

    # condition order: cheap (alpha=10000, T=10) -> baseline (alpha=1000, T=100)
    # -> expensive (alpha=100, T=1000). This lets us catch budget overrun
    # BEFORE the most expensive condition.
    cond_order = [10000, 1000, 100]
    per_cond = {a: {"alpha": a, "dt_s": None, "T_steps": None,
                    "per_seed": [], "diverged": False,
                    "diverged_reason": None,
                    "mean": None, "std": None, "ci95": None} for a in ALPHAS}
    for a in ALPHAS:
        dt, T = alpha_to_dt_T(a)
        per_cond[a]["dt_s"] = dt
        per_cond[a]["T_steps"] = T

    aborted_early = False
    abort_reason = None
    skipped_alphas = []

    for cond_idx, alpha in enumerate(cond_order):
        dt, T = alpha_to_dt_T(alpha)
        print("\n" + "-" * 72, flush=True)
        print(f"CONDITION {cond_idx+1}/3: alpha={alpha}  dt={dt:.2e} s  T={T}",
              flush=True)
        print("-" * 72, flush=True)
        # Pre-condition budget check: estimate cost of this condition
        # from any prior seed's wall (cost scales ~T).
        elapsed_now = time.time() - t_global
        prior_wall_per_step = None
        for a_done in cond_order[:cond_idx]:
            if per_cond[a_done]["per_seed"]:
                _, T_done = alpha_to_dt_T(a_done)
                wmean = float(np.mean([s["wall_s"]
                                       for s in per_cond[a_done]["per_seed"]]))
                prior_wall_per_step = wmean / T_done
                break
        if prior_wall_per_step is not None:
            est_cond_wall = prior_wall_per_step * T * len(SEEDS)
            est_total = elapsed_now + est_cond_wall
            print(f"[budget] cond alpha={alpha}: est_cond_wall={est_cond_wall:.0f}s "
                  f"est_total={est_total:.0f}s (budget={BUDGET_S}s)", flush=True)
            if est_total > BUDGET_S:
                skipped_alphas.append(alpha)
                print(f"[budget] SKIPPING alpha={alpha}: would exceed 3h budget.",
                      flush=True)
                continue

        seed_results = []
        cond_t0 = time.time()
        for si, seed in enumerate(SEEDS):
            r = run_seed_with_timescale(seed, alpha,
                                        X_train_use, y_train_use,
                                        X_test_use,  y_test_use)
            seed_results.append(r)
            thermal_peak = max(thermal_peak, z261.apu_temp_c())
            z261.thermal_guard()
            # Save incremental progress
            per_cond[alpha]["per_seed"] = seed_results
            (RESULTS_DIR / "summary.json.partial").write_text(
                json.dumps({"per_cond": per_cond,
                            "thermal_peak_c": thermal_peak,
                            "elapsed_s": time.time() - t_global}, indent=2))

            # Detect divergence: V_b rail frac > 0.8 means body voltage
            # explodes / sticks to the rail; surrogate clip > 0.5 means
            # VG1 is out-of-domain dominant. Either flags this condition.
            mean_rail = 0.5 * (r["vb_rail_frac_train"] + r["vb_rail_frac_test"])
            if mean_rail > 0.80 or r["clip_rate"] > 0.5:
                per_cond[alpha]["diverged"] = True
                per_cond[alpha]["diverged_reason"] = (
                    f"vb_rail_frac={mean_rail:.3f}, clip_rate={r['clip_rate']:.3f}"
                )

        accs = np.array([r["test_acc"] for r in seed_results])
        if len(accs) > 0:
            mean, std, ci = summarize(accs)
            per_cond[alpha]["mean"]  = mean
            per_cond[alpha]["std"]   = std
            per_cond[alpha]["ci95"]  = list(ci)
            cond_wall = time.time() - cond_t0
            print(f"[cond alpha={alpha}] mean={mean:.4f} +/- {std:.4f}  "
                  f"CI95=[{ci[0]:.4f}, {ci[1]:.4f}]  wall={cond_wall:.1f}s",
                  flush=True)

        if aborted_early:
            break

    if skipped_alphas:
        aborted_early = True
        abort_reason = f"skipped alphas due to budget: {skipped_alphas}"

    # Compute pairwise deltas vs alpha=1000 baseline
    base_alpha = 1000
    base_mean = per_cond[base_alpha]["mean"]
    deltas_pp = {}
    valid_means = []
    for a in ALPHAS:
        m = per_cond[a]["mean"]
        if m is None:
            deltas_pp[a] = None
            continue
        if base_mean is not None and a != base_alpha:
            deltas_pp[a] = (m - base_mean) * 100.0
        if not per_cond[a]["diverged"]:
            valid_means.append((a, m))

    # max pairwise delta across non-divergent conditions
    if len(valid_means) >= 2:
        vals = [m for _, m in valid_means]
        max_pair_pp = float((max(vals) - min(vals)) * 100.0)
        gate_eligible = True
    else:
        max_pair_pp = None
        gate_eligible = False

    n_completed_alphas = sum(1 for a in ALPHAS
                             if per_cond[a]["mean"] is not None)
    if not gate_eligible:
        verdict = "INCONCLUSIVE"
    elif n_completed_alphas < len(ALPHAS):
        # Partial result: we report max pairwise across what we got but
        # mark verdict as PARTIAL (cannot fully pre-register gate without
        # all three points). This is honest given the budget constraint.
        verdict = ("PARTIAL_PASS" if max_pair_pp <= GATE_DELTA_PP
                   else "PARTIAL_FAIL")
    else:
        verdict = "PASS" if max_pair_pp <= GATE_DELTA_PP else "FAIL"

    delta_100_vs_1000  = deltas_pp.get(100)
    delta_10000_vs_1000 = deltas_pp.get(10000)

    wall = time.time() - t_global

    summary = {
        "experiment": "z267_n3_timescale_sweep",
        "date": "2026-05-11",
        "dataset": dataset,
        "subsample": sub_info,
        "config": {
            "dt_base_s": DT_BASE,
            "T_base_steps": T_BASE,
            "alphas": ALPHAS,
            "cond_order_executed": cond_order,
            "N_neurons": z263.N_NEURONS,
            "VG1_bias_range": [z263.VG1_BIAS_LO, z263.VG1_BIAS_HI],
            "VG2_bias_range": [z263.VG2_BIAS_LO, z263.VG2_BIAS_HI],
            "g_in": z263.G_IN,
            "C_b_F": z263.C_B,
            "ridge_lambda": z261.RIDGE_LAMBDA,
        },
        "n_seeds": len(SEEDS),
        "seeds": SEEDS,
        "per_condition": per_cond,
        "delta_alpha100_vs_alpha1000_pp":   delta_100_vs_1000,
        "delta_alpha10000_vs_alpha1000_pp": delta_10000_vs_1000,
        "max_pairwise_delta_pp": max_pair_pp,
        "gate_threshold_pp": GATE_DELTA_PP,
        "gate_rule": "PASS if max pairwise |delta(test_acc)| <= 2 pp across non-divergent alphas",
        "verdict": verdict,
        "aborted_early": aborted_early,
        "abort_reason": abort_reason,
        "wall_s_total": wall,
        "thermal_peak_c": thermal_peak,
        "budget_s": BUDGET_S,
    }
    out_path = RESULTS_DIR / "summary.json"
    out_path.write_text(json.dumps(summary, indent=2))
    # Remove partial
    p = RESULTS_DIR / "summary.json.partial"
    if p.exists():
        p.unlink()

    print("\n" + "=" * 72, flush=True)
    print("z267 N3 - timescale rescaling consistency SUMMARY", flush=True)
    print("=" * 72, flush=True)
    for a in ALPHAS:
        c = per_cond[a]
        if c["mean"] is None:
            print(f"  alpha={a:>6}  dt={c['dt_s']:.2e}  T={c['T_steps']}  "
                  f"-> SKIPPED (budget/abort)", flush=True)
        else:
            ci = c["ci95"]
            div = "  [DIVERGED]" if c["diverged"] else ""
            print(f"  alpha={a:>6}  dt={c['dt_s']:.2e}  T={c['T_steps']}  "
                  f"mean={c['mean']:.4f}+-{c['std']:.4f}  "
                  f"CI95=[{ci[0]:.4f},{ci[1]:.4f}]{div}", flush=True)
    print(f"  delta(alpha=100   vs 1000) = "
          f"{('%+.2f pp' % delta_100_vs_1000)  if delta_100_vs_1000  is not None else 'n/a'}",
          flush=True)
    print(f"  delta(alpha=10000 vs 1000) = "
          f"{('%+.2f pp' % delta_10000_vs_1000) if delta_10000_vs_1000 is not None else 'n/a'}",
          flush=True)
    print(f"  max pairwise delta = "
          f"{('%.2f pp' % max_pair_pp) if max_pair_pp is not None else 'n/a'}",
          flush=True)
    print(f"  GATE (<= {GATE_DELTA_PP:.1f} pp): {verdict}", flush=True)
    print(f"  wall = {wall:.1f}s   thermal peak = {thermal_peak:.1f}C", flush=True)
    if aborted_early:
        print(f"  ABORTED EARLY: {abort_reason}", flush=True)
    print(f"  -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
