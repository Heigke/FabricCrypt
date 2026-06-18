"""M2: per-V_G1-branch residual decomposition.

Reads existing pyport DC fit residuals (z91f_validate_sebas) and decomposes
them by V_G1 ∈ {0.20, 0.40, 0.60} branch. Reports per-branch mean |residual|
with 95% bootstrap CI and cross-branch verdict.

GATE: PASS if max_excess (= max branch_mean − cross_branch_mean) ≥ 0.30 dec.
"""

import json
import math
import os
import random
from collections import defaultdict

random.seed(20260511)

PRED_PATH = "results/z91f_validate_sebas/predictions.json"
OUT_DIR = "results/z256_m2_branch_residual"
OUT_PATH = os.path.join(OUT_DIR, "summary.json")
N_BOOTSTRAP = 1000
GATE_DEC = 0.30


def per_row_residuals(curve):
    """Return list of |log10(|I_pred|) - log10(|I_meas|)| for converged rows."""
    out = []
    for ip, im in zip(curve["Id_pred"], curve["Id_meas"]):
        if ip is None or im is None:
            continue
        try:
            ip_f = float(ip)
            im_f = float(im)
        except (TypeError, ValueError):
            continue
        if not math.isfinite(ip_f) or not math.isfinite(im_f):
            continue
        if ip_f == 0.0 or im_f == 0.0:
            continue
        out.append(abs(math.log10(abs(ip_f)) - math.log10(abs(im_f))))
    return out


def bootstrap_ci(values, n=N_BOOTSTRAP, alpha=0.05):
    if not values:
        return [float("nan"), float("nan")]
    means = []
    k = len(values)
    for _ in range(n):
        sample = [values[random.randrange(k)] for _ in range(k)]
        means.append(sum(sample) / k)
    means.sort()
    lo = means[int(alpha / 2 * n)]
    hi = means[int((1 - alpha / 2) * n) - 1]
    return [lo, hi]


def main():
    with open(PRED_PATH) as f:
        data = json.load(f)

    branch_rows = defaultdict(list)
    branch_curve_count = defaultdict(int)
    skipped = 0
    for curve in data:
        if curve.get("skipped"):
            skipped += 1
            continue
        vg1 = round(float(curve["VG1"]), 2)
        if vg1 not in (0.20, 0.40, 0.60):
            continue
        rows = per_row_residuals(curve)
        if rows:
            branch_rows[vg1].extend(rows)
            branch_curve_count[vg1] += 1

    branch_summary = {}
    means = {}
    for vg1 in (0.20, 0.40, 0.60):
        vals = branch_rows[vg1]
        if not vals:
            branch_summary[f"branch_VG1_{vg1:.2f}"] = {
                "n_curves": branch_curve_count[vg1],
                "n_rows": 0,
                "mean_resid_dec": None,
                "ci95": [None, None],
            }
            continue
        m = sum(vals) / len(vals)
        ci = bootstrap_ci(vals)
        means[vg1] = m
        branch_summary[f"branch_VG1_{vg1:.2f}"] = {
            "n_curves": branch_curve_count[vg1],
            "n_rows": len(vals),
            "mean_resid_dec": m,
            "ci95": ci,
        }

    cross = sum(means.values()) / len(means)
    max_excess = max(means.values()) - cross
    verdict = "PASS" if max_excess >= GATE_DEC else "FAIL"

    summary = {
        "source": PRED_PATH,
        "n_curves_total": len(data),
        "n_curves_skipped_upstream": skipped,
        "n_bootstrap": N_BOOTSTRAP,
        "gate_threshold_dec": GATE_DEC,
        **branch_summary,
        "cross_branch_mean": cross,
        "max_excess": max_excess,
        "verdict": verdict,
    }

    os.makedirs(OUT_DIR, exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
