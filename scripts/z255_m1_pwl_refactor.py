"""z255_m1_pwl_refactor.py — M1 PWL(V_G2) bulk-current refactor.

Implements M1 from research_plan/POST_AUDIT_FIX_PLAN_2026-05-11.md.

Compares two functional forms for predicting Sebas's 33 measured
I-V curves (drain current vs V_d, parameterised by (V_G1, V_G2)):

  POLY baseline: polynomial regression in (V_G1, V_G2, V_d) of
                 log10(I_d). Degree 2 in V_G1, V_G2; degree 3 in V_d.
                 Closed-form least-squares.
  PWL form:      Sebas slide 12.26 form
                 I_pwl = a·V_d^c + b   for V_d ≥ −j
                 I_exp = 10^(d·V_d)
                 with a,b,c,d as PWL(V_G2), j scalar, fit per V_G1
                 branch (three branches: V_G1 ∈ {0.2, 0.4, 0.6}).
                 Optimiser: scipy least_squares.

Pre-registered gate: PASS if PWL residual ≤ POLY residual − 0.05 dec.

Outputs: results/z255_m1_pwl_refactor/summary.json
"""
from __future__ import annotations
import csv, json, re, sys, time
from pathlib import Path

import numpy as np

# Ensure local nsram package on sys.path
ROOT = Path("/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy")
sys.path.insert(0, str(ROOT / "nsram"))

from nsram.bsim4_port.pwl_bulk import (
    PWLBulkCurrentParams,
    DEFAULT_VG2_KNOTS,
    evaluate_pwl_bulk_current,
    fit_pwl_to_iv_family,
)


DATA = ROOT / "data/sebas_2026_04_22"
OUT = ROOT / "results/z255_m1_pwl_refactor"
OUT.mkdir(parents=True, exist_ok=True)

VG_RE = re.compile(r"VG2=(-?\d+\.?\d*)_VG=(\d+\.\d+)")
ID_FLOOR = 1e-13


def load_iv_curves():
    """Load all I-V curves from the three V_G1 subdirectories.

    Returns list of dicts with keys: vg1, vg2, Vd, Id.
    Filters to Id > ID_FLOOR and Vd ≥ 0 with unique Vd samples.
    """
    curves = []
    for sub in sorted(DATA.iterdir()):
        if not sub.is_dir():
            continue
        for fn in sorted(sub.glob("*.csv")):
            m = VG_RE.search(fn.name)
            if not m:
                continue
            vg2 = float(m.group(1))
            vg1 = float(m.group(2))
            rows = []
            with open(fn) as f:
                rdr = csv.reader(f)
                next(rdr)
                for r in rdr:
                    try:
                        # columns: vdata, idata, tdata, Var4, vfixgdata, ifixdata
                        rows.append((float(r[0]), float(r[1])))
                    except (ValueError, IndexError):
                        continue
            if not rows:
                continue
            Vd = np.array([r[0] for r in rows])
            Id = np.array([r[1] for r in rows])
            # keep ascending V_d up to peak (ignore retrace)
            peak = int(np.argmax(Vd))
            Vd = Vd[:peak + 1]
            Id = Id[:peak + 1]
            m_keep = (Id > ID_FLOOR) & (Vd >= 0.0)
            if m_keep.sum() < 5:
                continue
            Vd = Vd[m_keep]
            Id = Id[m_keep]
            uVd, idx = np.unique(Vd, return_index=True)
            Id = Id[idx]
            Vd = uVd
            curves.append({"vg1": vg1, "vg2": vg2, "Vd": Vd, "Id": Id})
    return curves


# ---- Polynomial baseline: regression in (VG1, VG2, V_d) of log10(Id) ----

def _poly_design(vg1, vg2, vd):
    """Design matrix: deg-2 in (vg1, vg2) ⊗ deg-3 in vd. Returns (N, K)."""
    v1, v2, vx = vg1, vg2, vd
    cols = []
    # terms in (vg1, vg2): 1, v1, v2, v1·v2, v1², v2²
    poly12 = [np.ones_like(v1), v1, v2, v1 * v2, v1 * v1, v2 * v2]
    # terms in vd: 1, vx, vx², vx³
    polyd = [np.ones_like(vx), vx, vx * vx, vx ** 3]
    for p12 in poly12:
        for pd in polyd:
            cols.append(p12 * pd)
    return np.column_stack(cols)


def fit_poly_baseline(curves):
    """Closed-form least-squares fit on log10(Id)."""
    vg1, vg2, vd, logid = [], [], [], []
    for c in curves:
        n = len(c["Vd"])
        vg1.append(np.full(n, c["vg1"]))
        vg2.append(np.full(n, c["vg2"]))
        vd.append(c["Vd"])
        logid.append(np.log10(c["Id"]))
    vg1 = np.concatenate(vg1)
    vg2 = np.concatenate(vg2)
    vd = np.concatenate(vd)
    logid = np.concatenate(logid)

    A = _poly_design(vg1, vg2, vd)
    coef, *_ = np.linalg.lstsq(A, logid, rcond=None)
    pred = A @ coef
    rmse = float(np.sqrt(np.mean((pred - logid) ** 2)))

    # Per-row residuals
    per_row = []
    for c in curves:
        v1 = np.full(len(c["Vd"]), c["vg1"])
        v2 = np.full(len(c["Vd"]), c["vg2"])
        Ap = _poly_design(v1, v2, c["Vd"])
        pp = Ap @ coef
        rr = float(np.sqrt(np.mean((pp - np.log10(c["Id"])) ** 2)))
        per_row.append({"vg1": c["vg1"], "vg2": c["vg2"], "n_pts": int(len(c["Vd"])), "rmse_dec": rr})

    return {
        "coef": coef.tolist(),
        "rmse_log10_dec": rmse,
        "n_points": int(len(logid)),
        "n_terms": int(A.shape[1]),
        "per_row": per_row,
    }, coef


def predict_poly(coef, vg1, vg2, vd):
    """Predicted log10(Id) from poly coefficients."""
    v1 = np.atleast_1d(vg1).astype(float)
    v2 = np.atleast_1d(vg2).astype(float)
    vx = np.atleast_1d(vd).astype(float)
    return _poly_design(v1, v2, vx) @ coef


# ---- PWL fit per V_G1 branch ----

def fit_pwl_per_branch(curves, poly_coef):
    """Fit PWL form separately on each V_G1 branch.

    For each branch, build the (V_G2, V_d, I_d) point set and fit the
    PWL form. Initial a,b,c,d are taken from the polynomial baseline
    evaluated at (vg1_branch, knot_vg2, V_d_anchor=1.0) where possible.
    """
    branches = sorted({c["vg1"] for c in curves})
    per_branch_results = {}
    all_residuals_log = []   # for total RMS over all 33 curves

    for vg1_b in branches:
        sub = [c for c in curves if c["vg1"] == vg1_b]

        # Init from poly: at each knot vg2, evaluate poly at vd_probe values
        # Use a rough init pattern: a~1e-7, c~2, b~0, d~-2.
        n_knots = len(DEFAULT_VG2_KNOTS)
        init_a = np.full(n_knots, 1e-7)
        init_b = np.full(n_knots, 0.0)
        init_c = np.full(n_knots, 2.0)
        init_d = np.full(n_knots, -2.0)

        # Refine init by per-knot 1D log-log fit on nearest measured curve.
        # For each knot V_G2, find the closest measured curve in this branch
        # and fit log10(Id) = log10(a) + c·log10(V_d) on Vd > 0.05 V points,
        # then init_a, init_c from that.  init_d set so that I_exp at low V_d
        # matches the measured minimum (rough but bounded).
        meas_vg2s = np.array([c["vg2"] for c in sub])
        for ki, vg2_k in enumerate(DEFAULT_VG2_KNOTS):
            j = int(np.argmin(np.abs(meas_vg2s - vg2_k)))
            cc = sub[j]
            vd_pos = cc["Vd"] > 0.1
            if vd_pos.sum() >= 3:
                lv = np.log10(cc["Vd"][vd_pos])
                li = np.log10(cc["Id"][vd_pos])
                # Solve [1, lv] · [log_a, c]^T = li by lstsq
                A1 = np.column_stack([np.ones_like(lv), lv])
                sol, *_ = np.linalg.lstsq(A1, li, rcond=None)
                log_a_init, c_init = sol
                init_a[ki] = float(np.clip(10.0 ** log_a_init, 1e-12, 1e-2))
                init_c[ki] = float(np.clip(c_init, 0.2, 5.0))
            # init_d: small negative (exponential floor below measurement)
            init_d[ki] = -3.0
            init_b[ki] = 0.0

        V_G1s = [c["vg1"] for c in sub]
        V_G2s = [c["vg2"] for c in sub]
        Vd_list = [c["Vd"] for c in sub]
        Id_list = [c["Id"] for c in sub]

        params, diag = fit_pwl_to_iv_family(
            V_G1s, V_G2s, Vd_list, Id_list,
            vg2_knots=DEFAULT_VG2_KNOTS,
            init_a=init_a, init_b=init_b, init_c=init_c, init_d=init_d,
            init_j=0.5, max_nfev=4000, verbose=0,
        )

        # Per-row residuals on this branch
        per_row = []
        for c in sub:
            vg2_arr = np.full(len(c["Vd"]), c["vg2"])
            I_pred = evaluate_pwl_bulk_current(vg2_arr, c["Vd"], params)
            log_pred = np.log10(np.maximum(I_pred, 1e-30))
            log_meas = np.log10(c["Id"])
            res = log_pred - log_meas
            rr = float(np.sqrt(np.mean(res ** 2)))
            per_row.append({"vg1": c["vg1"], "vg2": c["vg2"], "n_pts": int(len(c["Vd"])), "rmse_dec": rr})
            all_residuals_log.append(res)

        per_branch_results[f"VG1={vg1_b}"] = {
            "params": params.to_dict(),
            "diag": diag,
            "per_row": per_row,
        }

    all_res = np.concatenate(all_residuals_log)
    overall_rmse = float(np.sqrt(np.mean(all_res ** 2)))
    return {
        "rmse_log10_dec": overall_rmse,
        "n_points": int(all_res.size),
        "per_branch": per_branch_results,
    }


def main():
    t0 = time.time()
    print("[z255] loading I-V curves...")
    curves = load_iv_curves()
    print(f"[z255] loaded {len(curves)} curves")
    branches = sorted({c["vg1"] for c in curves})
    print(f"[z255] V_G1 branches: {branches}")
    for vg1_b in branches:
        n_b = sum(1 for c in curves if c["vg1"] == vg1_b)
        print(f"  V_G1={vg1_b}: {n_b} curves")

    print("[z255] fitting polynomial baseline...")
    poly_summary, poly_coef = fit_poly_baseline(curves)
    print(f"[z255] POLY RMSE = {poly_summary['rmse_log10_dec']:.4f} dec "
          f"({poly_summary['n_points']} pts, {poly_summary['n_terms']} terms)")

    print("[z255] fitting PWL form per V_G1 branch...")
    pwl_summary = fit_pwl_per_branch(curves, poly_coef)
    print(f"[z255] PWL  RMSE = {pwl_summary['rmse_log10_dec']:.4f} dec "
          f"({pwl_summary['n_points']} pts)")

    delta = pwl_summary["rmse_log10_dec"] - poly_summary["rmse_log10_dec"]
    gate_margin = -0.05  # PWL must be at least 0.05 dec BETTER
    passed = (delta <= gate_margin)
    verdict = "PASS" if passed else "FAIL"

    summary = {
        "script": "z255_m1_pwl_refactor.py",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "wall_seconds": round(time.time() - t0, 2),
        "regression_target": "33 measured I-V curves from data/sebas_2026_04_22",
        "n_curves": len(curves),
        "loss_metric": "RMS(log10(Id_pred) - log10(Id_meas)) in decades",
        "gate": {
            "rule": "PASS if PWL residual <= POLY residual - 0.05 dec",
            "poly_rmse_dec": poly_summary["rmse_log10_dec"],
            "pwl_rmse_dec": pwl_summary["rmse_log10_dec"],
            "delta_dec": delta,
            "required_margin_dec": gate_margin,
            "verdict": verdict,
        },
        "poly_baseline": poly_summary,
        "pwl_fit": pwl_summary,
        "vg2_knots": list(DEFAULT_VG2_KNOTS),
    }

    out_path = OUT / "summary.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[z255] summary written → {out_path}")
    print(f"[z255] VERDICT: {verdict} (poly={poly_summary['rmse_log10_dec']:.4f} dec, "
          f"pwl={pwl_summary['rmse_log10_dec']:.4f} dec, Δ={delta:+.4f} dec)")
    return summary


if __name__ == "__main__":
    main()
