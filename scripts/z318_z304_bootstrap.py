#!/usr/bin/env python3
"""
z318 — Row-bootstrap 95% CI for z304 (DA3) and z313 baselines.

Addresses O55 Q2 cherry-pick concern: the reported "0.99 dec" baseline for
z304 (DA3, Bf=3000) lacked a CI, asymmetric with z313's multi-variant report.

Method:
  - Pull per-row log_rmse from per_curve arrays
  - 1000 bootstrap resamples with replacement of IV rows
  - Report median + 95% percentile CI
  - Δ(z313 - z304) with CI overlap analysis

Inputs:
  results/z303_mario_bjt/summary.json   (configs[label=='da3'].per_curve)
  results/z313_pyport_v4/summary.json   (per_branch_full_A[branch].per_curve)

Output:
  results/z318_baseline_ci/summary.json
"""

import json
import os
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SEED = 20260513
N_BOOT = 1000
ALPHA = 0.05  # 95% CI


def load_z304_da3_rows():
    """Return per-row log_rmse for DA3 (Bf=3000) — the 0.99-dec baseline."""
    p = os.path.join(ROOT, "results/z303_mario_bjt/summary.json")
    s = json.load(open(p))
    da3 = [c for c in s["configs"] if c["label"] == "da3"][0]
    pc = da3["per_curve"]
    rows = []
    for c in pc:
        v = c.get("forward_log_rmse")
        if v is None or not np.isfinite(v):
            continue
        rows.append({
            "vg1": c.get("vg1"),
            "vg2": c.get("vg2"),
            "log_rmse": float(v),
        })
    return rows


def load_z313_rows(variant="A"):
    """Return per-row log_rmse for z313 run A (TAT off) — cell-wide rows."""
    p = os.path.join(ROOT, "results/z313_pyport_v4/summary.json")
    s = json.load(open(p))
    key = f"per_branch_full_{variant}"
    pbf = s[key]
    rows = []
    for branch_key, bd in pbf.items():
        vg1 = float(branch_key)
        for c in bd["per_curve"]:
            v = c.get("log_rmse")
            if v is None or not np.isfinite(v):
                continue
            rows.append({
                "vg1": vg1,
                "vg2": c.get("VG2"),
                "log_rmse": float(v),
            })
    return rows


def bootstrap_median(values, n_boot=N_BOOT, alpha=ALPHA, seed=SEED):
    """Row-bootstrap with replacement; return point median and 95% percentile CI."""
    rng = np.random.default_rng(seed)
    vals = np.asarray(values, dtype=float)
    n = len(vals)
    boot_meds = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boot_meds[i] = np.median(vals[idx])
    point = float(np.median(vals))
    lo = float(np.quantile(boot_meds, alpha / 2))
    hi = float(np.quantile(boot_meds, 1 - alpha / 2))
    return {
        "n_rows": int(n),
        "point_median": point,
        "ci95_lo": lo,
        "ci95_hi": hi,
        "boot_mean_of_medians": float(np.mean(boot_meds)),
        "boot_std_of_medians": float(np.std(boot_meds, ddof=1)),
        "boot_min": float(np.min(boot_meds)),
        "boot_max": float(np.max(boot_meds)),
    }


def bootstrap_delta(a_vals, b_vals, n_boot=N_BOOT, alpha=ALPHA, seed=SEED):
    """Bootstrap Δ = median(b) - median(a) by resampling each set independently."""
    rng = np.random.default_rng(seed + 1)
    a = np.asarray(a_vals, dtype=float)
    b = np.asarray(b_vals, dtype=float)
    na, nb = len(a), len(b)
    deltas = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        ai = rng.integers(0, na, size=na)
        bi = rng.integers(0, nb, size=nb)
        deltas[i] = np.median(b[bi]) - np.median(a[ai])
    point = float(np.median(b) - np.median(a))
    lo = float(np.quantile(deltas, alpha / 2))
    hi = float(np.quantile(deltas, 1 - alpha / 2))
    # one-sided p (b worse than a, i.e. delta>0)
    p_b_worse = float(np.mean(deltas <= 0))
    return {
        "delta_point": point,
        "ci95_lo": lo,
        "ci95_hi": hi,
        "boot_mean": float(np.mean(deltas)),
        "boot_std": float(np.std(deltas, ddof=1)),
        "fraction_delta_le_0": p_b_worse,
        "significant_at_0.05": (lo > 0.0) or (hi < 0.0),
    }


def main():
    z304_rows = load_z304_da3_rows()
    z313_rows = load_z313_rows("A")
    z313B_rows = load_z313_rows("B")

    z304_vals = [r["log_rmse"] for r in z304_rows]
    z313_vals = [r["log_rmse"] for r in z313_rows]
    z313B_vals = [r["log_rmse"] for r in z313B_rows]

    z304_ci = bootstrap_median(z304_vals)
    z313_ci = bootstrap_median(z313_vals)
    z313B_ci = bootstrap_median(z313B_vals)

    delta_A = bootstrap_delta(z304_vals, z313_vals)
    delta_B = bootstrap_delta(z304_vals, z313B_vals)

    # CI overlap
    overlap_A = not (z313_ci["ci95_lo"] > z304_ci["ci95_hi"] or z313_ci["ci95_hi"] < z304_ci["ci95_lo"])
    overlap_B = not (z313B_ci["ci95_lo"] > z304_ci["ci95_hi"] or z313B_ci["ci95_hi"] < z304_ci["ci95_lo"])

    # Verdict: z313 is a "regression" (worse) iff delta_A CI strictly > 0
    verdict_A = (
        "REGRESSION_CONFIRMED" if delta_A["ci95_lo"] > 0 else
        "NO_SIGNIFICANT_DIFFERENCE" if delta_A["ci95_lo"] <= 0 <= delta_A["ci95_hi"] else
        "IMPROVEMENT_CONFIRMED"
    )
    verdict_B = (
        "REGRESSION_CONFIRMED" if delta_B["ci95_lo"] > 0 else
        "NO_SIGNIFICANT_DIFFERENCE" if delta_B["ci95_lo"] <= 0 <= delta_B["ci95_hi"] else
        "IMPROVEMENT_CONFIRMED"
    )

    out = {
        "script": "z318_z304_bootstrap",
        "method": "row-bootstrap, n_boot=1000, percentile 95% CI",
        "seed": SEED,
        "n_boot": N_BOOT,
        "z304_da3_baseline": {
            "source": "results/z303_mario_bjt/summary.json :: configs[label=da3].per_curve",
            "config": "Bf=3000, Va=0.55, no avalanche (DA3 reference)",
            "rows": z304_rows,
            "ci": z304_ci,
        },
        "z313_runA_TAT_off": {
            "source": "results/z313_pyport_v4/summary.json :: per_branch_full_A",
            "rows": z313_rows,
            "ci": z313_ci,
        },
        "z313_runB_TAT_on": {
            "source": "results/z313_pyport_v4/summary.json :: per_branch_full_B",
            "rows": z313B_rows,
            "ci": z313B_ci,
        },
        "delta_z313A_minus_z304": {
            **delta_A,
            "ci_overlap_with_z304": overlap_A,
            "verdict": verdict_A,
        },
        "delta_z313B_minus_z304": {
            **delta_B,
            "ci_overlap_with_z304": overlap_B,
            "verdict": verdict_B,
        },
        "summary_text": {
            "z304_da3": f"median={z304_ci['point_median']:.3f} dec, 95% CI=[{z304_ci['ci95_lo']:.3f}, {z304_ci['ci95_hi']:.3f}], n={z304_ci['n_rows']}",
            "z313_A":  f"median={z313_ci['point_median']:.3f} dec, 95% CI=[{z313_ci['ci95_lo']:.3f}, {z313_ci['ci95_hi']:.3f}], n={z313_ci['n_rows']}",
            "z313_B":  f"median={z313B_ci['point_median']:.3f} dec, 95% CI=[{z313B_ci['ci95_lo']:.3f}, {z313B_ci['ci95_hi']:.3f}], n={z313B_ci['n_rows']}",
            "delta_A_text": f"Δ(A-DA3) = {delta_A['delta_point']:+.3f} dec, 95% CI=[{delta_A['ci95_lo']:+.3f}, {delta_A['ci95_hi']:+.3f}], verdict={verdict_A}",
            "delta_B_text": f"Δ(B-DA3) = {delta_B['delta_point']:+.3f} dec, 95% CI=[{delta_B['ci95_lo']:+.3f}, {delta_B['ci95_hi']:+.3f}], verdict={verdict_B}",
        },
        "gate_LOCKED": {
            "z304_baseline_ci_reported": True,
            "delta_to_z313_retains_significance_or_honest_null": True,
        },
    }

    out_dir = os.path.join(ROOT, "results/z318_baseline_ci")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "summary.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)

    print(f"[z318] wrote {out_path}")
    for k, v in out["summary_text"].items():
        print(f"  {k}: {v}")
    print(f"  verdict_A: {verdict_A}")
    print(f"  verdict_B: {verdict_B}")


if __name__ == "__main__":
    main()
