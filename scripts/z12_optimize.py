"""z12_optimize.py — proper global optimization of full-physics BSIM4 model.

After adding §2.2 SCE+DIBL+NLX-like terms, §5.4 mobility degradation,
§6.1 non-local Iii and explicit subthreshold current, we have ~8
unknowns to calibrate jointly. Grid sweeps are inefficient at this
dimensionality. Use scipy.optimize.differential_evolution — a global
gradient-free optimizer well-suited to noisy objective surfaces.

Parameters optimized (with physical bounds):
  VTH0    ∈ [0.40, 1.20] V     threshold voltage
  LTW     ∈ [15e-9, 100e-9] m  SCE characteristic length
  L_NONLOCAL ∈ [0, 300e-9] m   non-local Iii length
  log10(BJT_AREA)  ∈ [-8, 0]   SPICE area multiplier
  log10(Rb)        ∈ [6, 10]   body leakage resistance
  log10(ALPHA0_mult) ∈ [-2, 2] multiplier on PTM130 ALPHA0

Objective: median log-RMSE across all 33 curves (using the
branch-following continuation solver for correct firing bifurcation).

Runs CPU-parallel across workers. GPU isn't useful for the serial
branch-following loop — parallelism instead comes from evaluating many
(population × curves) in parallel workers.
"""
from __future__ import annotations

import csv, json, re, time
from dataclasses import replace
from pathlib import Path
from multiprocessing import Pool

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import brentq, differential_evolution

from nsram.bsim4 import (BSIM4_PRESETS, bipolar_collector_current_ss,
                          drain_current_bsim, gidl_current,
                          impact_ionization_bsim4)
from nsram.physics import thermal_voltage


DATA_DIR = Path("/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy/"
                "data/sebas_2026_04_22")
OUT_DIR = Path("/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy/"
               "results/z12_optimize")
OUT_DIR.mkdir(parents=True, exist_ok=True)

BASE = BSIM4_PRESETS["ns_ram_130nm_pazos"]
Vt_th = thermal_voltage(300.0)
VG_RE = re.compile(r"VG2=(-?\d+\.?\d*)_VG=(\d+\.\d+)")


def load_curves(n_ds: int = 20):
    curves = []
    for sub in sorted(DATA_DIR.iterdir()):
        if not sub.is_dir():
            continue
        for fn in sorted(sub.glob("*.csv")):
            m = VG_RE.search(fn.name)
            if not m:
                continue
            rows = []
            with open(fn) as f:
                rdr = csv.reader(f); next(rdr)
                for r in rdr:
                    try:
                        rows.append((float(r[2]), float(r[0]), float(r[1])))
                    except ValueError:
                        continue
            rows.sort()
            Vd = np.array([r[1] for r in rows])
            Id = np.array([r[2] for r in rows])
            peak = int(np.argmax(Vd))
            Vd = Vd[:peak + 1]; Id = Id[:peak + 1]
            nvd = np.linspace(0.1, float(Vd.max()), n_ds)
            nid = np.interp(nvd, Vd, Id)
            curves.append((float(m.group(2)), float(m.group(1)), nvd, nid))
    return curves


CURVES = load_curves(20)
print(f"Loaded {len(CURVES)} curves")


def kcl(Vb, Vg1, Vd, p):
    Iii = float(impact_ionization_bsim4(Vg1, Vd, Vb, p))
    Ig = float(gidl_current(Vd, Vg1, Vb, p)) if p.AGIDL > 0 else 0.0
    IS = p.BJT_IS * max(p.BJT_AREA, 1e-30)
    Vc = min(Vb, p.BJT_VJE * 1.1)
    e = np.clip(Vc / (p.BJT_NE * Vt_th), -60, 60)
    Io = (IS / p.BJT_BF) * (np.exp(e) - 1.0) + Vb / p.Rb
    return (Iii + Ig) - Io


def find_vb(Vg1, Vd, p, Vb0, vmax=0.85):
    lo = max(0.0, Vb0 - 0.02)
    hi = min(vmax, max(Vb0 + 0.02, 0.02))
    try:
        if kcl(lo, Vg1, Vd, p) * kcl(hi, Vg1, Vd, p) <= 0:
            return float(brentq(kcl, lo, hi, args=(Vg1, Vd, p),
                                  xtol=1e-5, rtol=1e-5))
    except ValueError:
        pass
    g = np.linspace(0.0, vmax, 61)
    f = np.array([kcl(v, Vg1, Vd, p) for v in g])
    sc = np.where(np.sign(f[:-1]) != np.sign(f[1:]))[0]
    if len(sc) == 0:
        return float(vmax)
    roots = []
    for i in sc:
        try:
            roots.append(brentq(kcl, g[i], g[i+1], args=(Vg1, Vd, p),
                                  xtol=1e-5, rtol=1e-5))
        except ValueError:
            continue
    if not roots:
        return float(vmax)
    return float(min(roots) if Vb0 < 0.1
                 else min(roots, key=lambda r: abs(r - Vb0)))


def trace(Vg1, Vds, p):
    Vb = 0.0
    Vbs = np.zeros_like(Vds)
    for k, vd in enumerate(Vds):
        Vb = find_vb(Vg1, float(vd), p, Vb0=Vb)
        Vbs[k] = Vb
    Id, _ = drain_current_bsim(Vg1, Vds, Vbs, p)
    Ic = bipolar_collector_current_ss(Vg1, Vds, Vbs, p)
    return np.asarray(Id) + np.asarray(Ic)


def build_params(x):
    """x = [VTH0, LTW, L_NONLOCAL, log10_area, log10_rb, log10_alpha_mult]"""
    vth0, ltw, lnl, lga, lgrb, lgam = x
    return replace(BASE,
                    VTH0=float(vth0),
                    LTW=float(ltw),
                    L_NONLOCAL=float(lnl),
                    BJT_AREA=float(10 ** lga),
                    Rb=float(10 ** lgrb),
                    ALPHA0=float(BASE.ALPHA0 * 10 ** lgam))


def objective_single(x, metric="median"):
    p = build_params(x)
    rmses = []
    for vg1, vg2, vd, idd in CURVES:
        try:
            pred = trace(vg1, vd, p)
            m = (idd > 1e-13) & (pred > 0)
            if m.sum() < 5:
                continue
            rmses.append(float(np.sqrt(np.mean((np.log10(idd[m]) -
                                                  np.log10(pred[m])) ** 2))))
        except Exception:
            continue
    if not rmses:
        return 10.0
    rs = np.array(rmses)
    # Minimize 0.5·median + 0.5·p90 — balances typical vs worst curve
    return 0.5 * float(np.median(rs)) + 0.5 * float(np.percentile(rs, 90))


BOUNDS = [
    (0.50, 1.20),       # VTH0
    (15e-9, 100e-9),    # LTW
    (0.0,   300e-9),    # L_NONLOCAL
    (-8.0, 0.0),        # log10 BJT_AREA
    (6.0,  10.0),       # log10 Rb
    (-2.0, 2.0),        # log10 ALPHA0_mult
]


def main():
    print(f"\nOptimizing {len(BOUNDS)} parameters with differential evolution")
    print(f"  ({len(CURVES)} curves, 20 Vd pts each)")

    t0 = time.time()
    call_count = [0]

    def callback(xk, convergence):
        call_count[0] += 1
        obj = objective_single(xk)
        elapsed = time.time() - t0
        print(f"  iter {call_count[0]:3d}  obj={obj:.3f}  conv={convergence:.3f}  "
              f"VTH0={xk[0]:.3f} LTW={xk[1]*1e9:.0f}n L_nl={xk[2]*1e9:.0f}n "
              f"AREA=1e{xk[3]:.1f} Rb=1e{xk[4]:.1f} A0×10^{xk[5]:.2f}  "
              f"({elapsed:.0f}s)")
        return False

    result = differential_evolution(
        objective_single, BOUNDS,
        maxiter=40, popsize=10,
        mutation=(0.5, 1.2), recombination=0.7,
        seed=42, tol=1e-3, workers=-1, updating="deferred",
        callback=callback, polish=True, disp=False,
    )

    x_opt = result.x
    print(f"\n═══ OPTIMIZATION DONE ═══")
    print(f"  iterations    : {result.nit}")
    print(f"  func evals    : {result.nfev}")
    print(f"  success       : {result.success}")
    print(f"  final objective : {result.fun:.3f}")
    print(f"\n  VTH0          = {x_opt[0]:.4f} V")
    print(f"  LTW           = {x_opt[1]*1e9:.1f} nm")
    print(f"  L_NONLOCAL    = {x_opt[2]*1e9:.1f} nm")
    print(f"  BJT_AREA      = 10^{x_opt[3]:.3f}")
    print(f"  Rb            = 10^{x_opt[4]:.3f} Ω")
    print(f"  ALPHA0 × 10^{x_opt[5]:.3f}")

    # Re-evaluate final residuals
    p = build_params(x_opt)
    per_curve = []
    for vg1, vg2, vd, idd in CURVES:
        try:
            pred = trace(vg1, vd, p)
            m = (idd > 1e-13) & (pred > 0)
            if m.sum() < 5:
                continue
            r = float(np.sqrt(np.mean((np.log10(idd[m]) - np.log10(pred[m])) ** 2)))
            per_curve.append({"vg1": vg1, "vg2": vg2, "log_rmse": r})
        except Exception:
            continue
    rs = np.array([r["log_rmse"] for r in per_curve])
    print(f"\n═══ Final fit quality ═══")
    print(f"  median log-RMSE : {np.median(rs):.2f} dec")
    print(f"  p90    log-RMSE : {np.percentile(rs, 90):.2f} dec")
    print(f"  worst           : {np.max(rs):.2f} dec")

    with open(OUT_DIR / "summary.json", "w") as f:
        json.dump({
            "x_opt": x_opt.tolist(),
            "VTH0": float(x_opt[0]),
            "LTW_nm": float(x_opt[1] * 1e9),
            "L_NONLOCAL_nm": float(x_opt[2] * 1e9),
            "BJT_AREA": float(10 ** x_opt[3]),
            "Rb": float(10 ** x_opt[4]),
            "ALPHA0": float(BASE.ALPHA0 * 10 ** x_opt[5]),
            "median_log_rmse": float(np.median(rs)),
            "p90_log_rmse": float(np.percentile(rs, 90)),
            "worst_log_rmse": float(np.max(rs)),
            "per_curve": per_curve,
        }, f, indent=2)

    # Overlay
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.3), sharey=True)
    shown = {}
    for vg1, vg2, vd, idd in CURVES:
        if vg1 not in shown and abs(vg2) < 0.06:
            shown[vg1] = (vg2, vd, idd, trace(vg1, vd, p))
    for ax, vg1 in zip(axes, sorted(shown)):
        vg2, vd, idd, pred = shown[vg1]
        m = (idd > 1e-13) & (pred > 0)
        r = float(np.sqrt(np.mean((np.log10(idd[m]) - np.log10(pred[m])) ** 2))) if m.any() else float("nan")
        ax.semilogy(vd, np.clip(idd, 1e-15, None), "k-", lw=2, label="meas")
        ax.semilogy(vd, np.clip(pred, 1e-22, None), "g-", lw=1.7,
                    label=f"fit ({r:.2f})")
        ax.set_title(f"VG1={vg1}")
        ax.set_xlabel("Vd [V]")
        ax.grid(alpha=0.3, which="both"); ax.legend(fontsize=8)
    axes[0].set_ylabel("|Id| [A]")
    fig.suptitle(f"Full BSIM4 + NPN + branch-follow — DE optimized "
                 f"(median {np.median(rs):.2f}, worst {np.max(rs):.2f})")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "overlay.png", dpi=140)
    plt.close(fig)
    print(f"\nWrote {OUT_DIR / 'overlay.png'}, summary.json")


if __name__ == "__main__":
    main()
