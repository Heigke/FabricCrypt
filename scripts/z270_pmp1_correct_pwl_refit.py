"""z270_pmp1_correct_pwl_refit.py — PMP-1 correct bulk-current PWL refit.

Reimplementation of the M1 (z255) PWL refit with the CORRECTED Sebas
slide-13 functional form that O42 oracles unanimously confirmed:

    I_exp = a · exp[b · (V_D + c)]                    # natural exp
    I_pow = d · (V_D + f)^e   for V_D > -f, else 0
    I_bulk = max(0, I_pow) + I_exp

where a, b, d, e, f are PWL functions of the gate voltage and c is a
scalar constant. The slide does NOT disambiguate V_G1 vs V_G2, so we
fit TWO variants:

    Variant A: PWL(V_G1)  (parameter sweep over V_G1, V_G2 ignored)
    Variant B: PWL(V_G2)  (parameter sweep over V_G2, V_G1 ignored)

Knots (FIXED, pre-registered in 01_LOG.md):
    V_G ∈ {0.0, 0.1, 0.2, 0.3, 0.4, 0.5} V.

Target: total measured drain current across all 33 curves; the slide
equation describes bulk current. Because we don't have bulk-only
measurements, we fit against the measured drain current directly
(same target as M1; this isolates the EQUATION FORM change as the
sole experimental variable vs M1).

Loss: RMS(log10(I_pred) - log10(I_meas)) in decades.
Optimiser: scipy.optimize.least_squares (TRF), no auto-restart.

Pre-registered gate:
  PASS              if best PWL variant <= poly - 0.10 dec
  INFORMATIVE-PASS  if |best PWL - poly| <= 0.10 dec
  FAIL              otherwise (PWL > poly + 0.10 dec)

Output: results/z270_pmp1_correct_pwl/summary.json
"""
from __future__ import annotations
import csv
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares


ROOT = Path("/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy")
DATA = ROOT / "data/sebas_2026_04_22"
OUT = ROOT / "results/z270_pmp1_correct_pwl"
OUT.mkdir(parents=True, exist_ok=True)

VG_RE = re.compile(r"VG2=(-?\d+\.?\d*)_VG=(\d+\.\d+)")
ID_FLOOR = 1e-13

# Pre-registered knots (FIXED)
VG_KNOTS = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5], dtype=float)
N_KNOTS = len(VG_KNOTS)


# ----- Data loader (same as M1) -----

def load_iv_curves():
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
                        rows.append((float(r[0]), float(r[1])))
                    except (ValueError, IndexError):
                        continue
            if not rows:
                continue
            Vd = np.array([r[0] for r in rows])
            Id = np.array([r[1] for r in rows])
            peak = int(np.argmax(Vd))
            Vd = Vd[:peak + 1]
            Id = Id[:peak + 1]
            keep = (Id > ID_FLOOR) & (Vd >= 0.0)
            if keep.sum() < 5:
                continue
            Vd = Vd[keep]
            Id = Id[keep]
            uVd, idx = np.unique(Vd, return_index=True)
            curves.append({"vg1": vg1, "vg2": vg2, "Vd": uVd, "Id": Id[idx]})
    return curves


# ----- Poly baseline (identical to M1) -----

def _poly_design(vg1, vg2, vd):
    v1, v2, vx = vg1, vg2, vd
    poly12 = [np.ones_like(v1), v1, v2, v1 * v2, v1 * v1, v2 * v2]
    polyd = [np.ones_like(vx), vx, vx * vx, vx ** 3]
    cols = []
    for p12 in poly12:
        for pd in polyd:
            cols.append(p12 * pd)
    return np.column_stack(cols)


def fit_poly_baseline(curves):
    vg1, vg2, vd, logid = [], [], [], []
    for c in curves:
        n = len(c["Vd"])
        vg1.append(np.full(n, c["vg1"]))
        vg2.append(np.full(n, c["vg2"]))
        vd.append(c["Vd"])
        logid.append(np.log10(c["Id"]))
    vg1 = np.concatenate(vg1); vg2 = np.concatenate(vg2)
    vd = np.concatenate(vd);   logid = np.concatenate(logid)
    A = _poly_design(vg1, vg2, vd)
    coef, *_ = np.linalg.lstsq(A, logid, rcond=None)
    pred = A @ coef
    rmse = float(np.sqrt(np.mean((pred - logid) ** 2)))
    return {
        "rmse_log10_dec": rmse,
        "n_points": int(len(logid)),
        "n_terms": int(A.shape[1]),
        "coef": coef.tolist(),
    }


# ----- CORRECT PWL form (O42-confirmed slide-13 equation) -----
#
# I_exp = a · exp[b · (V_D + c)]
# I_pow = d · (V_D + f)^e   for V_D > -f, else 0
# I_bulk = max(0, I_pow) + I_exp
#
# Flat parameter vector layout (length 5*N + 1):
#   [a(knots), b(knots), d(knots), e(knots), f(knots), c]
# with a, b, d, e, f PWL over VG_KNOTS and c scalar.


def _unpack(x):
    n = N_KNOTS
    a = x[0:n]
    b = x[n:2 * n]
    d = x[2 * n:3 * n]
    e = x[3 * n:4 * n]
    f = x[4 * n:5 * n]
    c = float(x[5 * n])
    return a, b, d, e, f, c


def evaluate_pwl_correct(vg_param, vd, x):
    """Evaluate corrected slide-13 bulk current.

    vg_param: gate voltage used to index PWL (V_G1 OR V_G2 depending on
              variant; selected by caller).
    vd: drain voltage.
    """
    a_k, b_k, d_k, e_k, f_k, c = _unpack(x)
    a = np.interp(vg_param, VG_KNOTS, a_k)
    b = np.interp(vg_param, VG_KNOTS, b_k)
    d = np.interp(vg_param, VG_KNOTS, d_k)
    e = np.interp(vg_param, VG_KNOTS, e_k)
    f = np.interp(vg_param, VG_KNOTS, f_k)

    # I_exp = a · exp[b · (V_D + c)]; clip exponent for safety.
    arg = np.clip(b * (vd + c), -60.0, 60.0)
    I_exp = a * np.exp(arg)

    # I_pow = d · (V_D + f)^e for V_D > -f, else 0.
    base = vd + f
    # Power needs non-negative base; mask first, then floor.
    mask = base > 0.0
    base_safe = np.where(mask, np.maximum(base, 1e-12), 1e-12)
    I_pow = np.where(mask, d * np.power(base_safe, e), 0.0)

    I_total = np.maximum(I_pow, 0.0) + I_exp
    return np.maximum(I_total, 1e-30)


def fit_pwl_variant(curves, which_vg: str):
    """Fit corrected PWL form across all 33 curves.

    which_vg: "VG1" or "VG2" — controls which gate voltage parameterises
    the PWL.
    """
    assert which_vg in ("VG1", "VG2")
    vg_param_list = []
    vd_list = []
    logid_list = []
    for c in curves:
        n = len(c["Vd"])
        vg_param_list.append(np.full(n, c["vg1"] if which_vg == "VG1" else c["vg2"]))
        vd_list.append(c["Vd"])
        logid_list.append(np.log10(c["Id"]))
    vg_arr = np.concatenate(vg_param_list)
    vd_arr = np.concatenate(vd_list)
    log_meas = np.concatenate(logid_list)

    # Initial guess. From M1 behavior, expect:
    #   a ~ 1e-9..1e-6 (exp prefactor)
    #   b ~ 5..15 (exp slope in 1/V; nat-log slope of ~3-7 dec/V)
    #   d ~ 1e-7..1e-4 (power-law amplitude)
    #   e ~ 1..4 (power exponent)
    #   f ~ 0.05..0.5 (shift)
    #   c ~ 0..1 (shift inside exp)
    init_a = np.full(N_KNOTS, 1e-8)
    init_b = np.full(N_KNOTS, 8.0)
    init_d = np.full(N_KNOTS, 1e-6)
    init_e = np.full(N_KNOTS, 2.0)
    init_f = np.full(N_KNOTS, 0.2)
    init_c = 0.0
    x0 = np.concatenate([init_a, init_b, init_d, init_e, init_f, [init_c]])

    # Bounds: physically reasonable, wide enough not to constrain fit.
    lo = np.concatenate([
        np.full(N_KNOTS, 0.0),        # a >= 0
        np.full(N_KNOTS, 0.0),        # b >= 0 (exp must rise with V_D)
        np.full(N_KNOTS, 0.0),        # d >= 0
        np.full(N_KNOTS, 0.1),        # e
        np.full(N_KNOTS, 0.0),        # f >= 0 (shift)
        [-2.0],                       # c
    ])
    hi = np.concatenate([
        np.full(N_KNOTS, 1e-1),       # a
        np.full(N_KNOTS, 40.0),       # b
        np.full(N_KNOTS, 1e-1),       # d
        np.full(N_KNOTS, 8.0),        # e
        np.full(N_KNOTS, 5.0),        # f
        [2.0],                        # c
    ])

    def residuals(x):
        I_pred = evaluate_pwl_correct(vg_arr, vd_arr, x)
        return np.log10(np.maximum(I_pred, 1e-30)) - log_meas

    res = least_squares(
        residuals, x0, bounds=(lo, hi),
        method="trf", max_nfev=8000, verbose=0,
        xtol=1e-10, ftol=1e-10, gtol=1e-10,
    )

    rmse = float(np.sqrt(np.mean(res.fun ** 2)))
    a_k, b_k, d_k, e_k, f_k, c = _unpack(res.x)

    # Per-row residuals
    per_row = []
    for c_curve in curves:
        vg_v = c_curve["vg1"] if which_vg == "VG1" else c_curve["vg2"]
        vg_arr_c = np.full(len(c_curve["Vd"]), vg_v)
        I_pred = evaluate_pwl_correct(vg_arr_c, c_curve["Vd"], res.x)
        diff = np.log10(np.maximum(I_pred, 1e-30)) - np.log10(c_curve["Id"])
        rr = float(np.sqrt(np.mean(diff ** 2)))
        per_row.append({
            "vg1": c_curve["vg1"], "vg2": c_curve["vg2"],
            "n_pts": int(len(c_curve["Vd"])), "rmse_dec": rr,
        })

    return {
        "variant": which_vg,
        "rmse_log10_dec": rmse,
        "n_points": int(log_meas.size),
        "nfev": int(res.nfev),
        "status": int(res.status),
        "message": str(res.message),
        "success": bool(res.success),
        "cost": float(res.cost),
        "coefficients": {
            "vg_knots": VG_KNOTS.tolist(),
            "a": a_k.tolist(),
            "b": b_k.tolist(),
            "d": d_k.tolist(),
            "e": e_k.tolist(),
            "f": f_k.tolist(),
            "c_scalar": float(c),
        },
        "per_row": per_row,
    }


def main():
    t0 = time.time()
    print("[z270] loading I-V curves...")
    curves = load_iv_curves()
    print(f"[z270] loaded {len(curves)} curves")
    branches = sorted({c["vg1"] for c in curves})
    for b in branches:
        n_b = sum(1 for c in curves if c["vg1"] == b)
        print(f"  V_G1={b}: {n_b} curves")

    print("[z270] fitting polynomial baseline (sanity)...")
    poly = fit_poly_baseline(curves)
    print(f"[z270] POLY RMSE = {poly['rmse_log10_dec']:.4f} dec")

    print("[z270] fitting CORRECT PWL form, variant A (PWL on V_G1)...")
    pwl_a = fit_pwl_variant(curves, "VG1")
    print(f"[z270] PWL(V_G1) RMSE = {pwl_a['rmse_log10_dec']:.4f} dec  (nfev={pwl_a['nfev']}, ok={pwl_a['success']})")

    print("[z270] fitting CORRECT PWL form, variant B (PWL on V_G2)...")
    pwl_b = fit_pwl_variant(curves, "VG2")
    print(f"[z270] PWL(V_G2) RMSE = {pwl_b['rmse_log10_dec']:.4f} dec  (nfev={pwl_b['nfev']}, ok={pwl_b['success']})")

    poly_r = poly["rmse_log10_dec"]
    a_r = pwl_a["rmse_log10_dec"]
    b_r = pwl_b["rmse_log10_dec"]
    best_variant = "VG1" if a_r <= b_r else "VG2"
    best_r = min(a_r, b_r)
    delta = best_r - poly_r

    if delta <= -0.10:
        verdict = "PASS"
    elif abs(delta) <= 0.10:
        verdict = "INFORMATIVE-PASS"
    else:
        verdict = "FAIL"

    summary = {
        "script": "z270_pmp1_correct_pwl_refit.py",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "wall_seconds": round(time.time() - t0, 2),
        "n_curves": len(curves),
        "regression_target": "33 measured I-V curves (drain current) from data/sebas_2026_04_22",
        "equation_form": "I_exp = a*exp(b*(V_D+c)); I_pow = d*(V_D+f)^e for V_D>-f else 0; I = max(0,I_pow)+I_exp",
        "knots_VG": VG_KNOTS.tolist(),
        "loss_metric": "RMS(log10(I_pred) - log10(I_meas)) in decades",
        "gate": {
            "rule": "PASS if best PWL <= poly - 0.10 dec; INFORMATIVE-PASS if |delta|<=0.10; FAIL otherwise",
            "poly_residual_dec": poly_r,
            "pwl_VG1_residual_dec": a_r,
            "pwl_VG2_residual_dec": b_r,
            "best_variant": best_variant,
            "best_residual_dec": best_r,
            "delta_dec_vs_poly": delta,
            "verdict": verdict,
        },
        "poly_baseline": poly,
        "pwl_VG1": pwl_a,
        "pwl_VG2": pwl_b,
    }
    out_path = OUT / "summary.json"
    with open(out_path, "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"[z270] summary written -> {out_path}")
    print(f"[z270] VERDICT: {verdict}")
    print(f"[z270]   poly      = {poly_r:.4f} dec")
    print(f"[z270]   PWL(V_G1) = {a_r:.4f} dec")
    print(f"[z270]   PWL(V_G2) = {b_r:.4f} dec")
    print(f"[z270]   best={best_variant}, delta={delta:+.4f} dec")
    return summary


if __name__ == "__main__":
    main()
