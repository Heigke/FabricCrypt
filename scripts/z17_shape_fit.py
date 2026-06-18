"""z17_shape_fit.py — DE fit with wider Rb bounds + shape-aware objective.

Diagnosis (z14/z15/z16) showed the z13 best has Rb=754kΩ which is too low
for the parasitic NPN to ever latch — kcl(Vb) is monostable at Vb≈0 for
every Vd. To latch, Rb must be ≳10⁸ Ω (floating body regime).

This script:
  • widens log10(Rb) bound to [7.0, 12.0]
  • augments objective: log-RMSE + λ·|peak-log-slope mismatch|
    (so smoothed fits can no longer hide inside the RMSE metric)
  • reuses the branch-following trace() from z12
"""
from __future__ import annotations
import json, time
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import differential_evolution

from scripts.z12_optimize import build_params, trace, CURVES, BASE

OUT = Path("/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy/"
           "results/z17_shape_fit")
OUT.mkdir(parents=True, exist_ok=True)


def peak_log_slope(vd, I, floor=1e-13):
    """Max of dlog10(I)/dVd, clipped to sensible values."""
    m = I > floor
    if m.sum() < 3:
        return 0.0
    logI = np.log10(np.clip(I[m], floor, None))
    slope = np.gradient(logI, vd[m])
    return float(np.max(slope))


LAMBDA_SHAPE = 0.30  # weight on shape term (decades per volt mismatch)


def objective_shape(x):
    p = build_params(x)
    scores = []
    for vg1, vg2, vd, idd in CURVES:
        try:
            pred = trace(vg1, vd, p)
            m = (idd > 1e-13) & (pred > 0)
            if m.sum() < 5:
                continue
            rmse = float(np.sqrt(np.mean(
                (np.log10(idd[m]) - np.log10(pred[m])) ** 2)))
            s_meas = peak_log_slope(vd, idd)
            s_fit = peak_log_slope(vd, pred)
            slope_mismatch = abs(s_meas - s_fit) / max(s_meas, 1.0)
            scores.append(rmse + LAMBDA_SHAPE * slope_mismatch)
        except Exception:
            continue
    if not scores:
        return 10.0
    arr = np.array(scores)
    return 0.5 * float(np.median(arr)) + 0.5 * float(np.percentile(arr, 90))


BOUNDS = [
    (0.50, 1.20),       # VTH0
    (15e-9, 100e-9),    # LTW
    (0.0,   300e-9),    # L_NONLOCAL
    (-8.0, 0.0),        # log10 BJT_AREA
    ( 7.0, 12.0),       # log10 Rb   ← WIDENED to allow floating body
    (-2.0, 2.0),        # log10 ALPHA0_mult
]


def main():
    print(f"Shape-aware DE, {len(CURVES)} curves, λ_shape={LAMBDA_SHAPE}")
    t0 = time.time()
    count = [0]
    def cb(xk, conv):
        count[0] += 1
        obj = objective_shape(xk)
        p_ = build_params(xk)
        print(f"  iter {count[0]:3d} obj={obj:.3f} conv={conv:.3f}  "
              f"VTH0={xk[0]:.3f} LTW={xk[1]*1e9:.0f}n L_nl={xk[2]*1e9:.0f}n "
              f"A=1e{xk[3]:.1f} Rb=1e{xk[4]:.1f} A0×10^{xk[5]:.2f} "
              f"({time.time()-t0:.0f}s)", flush=True)
        return False
    res = differential_evolution(
        objective_shape, BOUNDS,
        maxiter=35, popsize=12,
        mutation=(0.5, 1.2), recombination=0.7,
        seed=7, tol=1e-3, workers=-1, updating="deferred",
        callback=cb, polish=True, disp=False,
    )
    xo = res.x
    print(f"\nBest: obj={res.fun:.3f}")
    p = build_params(xo)
    per_curve = []
    for vg1, vg2, vd, idd in CURVES:
        try:
            pred = trace(vg1, vd, p)
            m = (idd > 1e-13) & (pred > 0)
            if m.sum() < 5: continue
            r = float(np.sqrt(np.mean((np.log10(idd[m]) - np.log10(pred[m]))**2)))
            per_curve.append({"vg1": vg1, "vg2": vg2, "log_rmse": r,
                                "s_meas": peak_log_slope(vd, idd),
                                "s_fit":  peak_log_slope(vd, pred)})
        except Exception:
            continue
    rs = np.array([c["log_rmse"] for c in per_curve])
    print(f"median={np.median(rs):.2f}  p90={np.percentile(rs,90):.2f}  "
          f"worst={np.max(rs):.2f}")

    with open(OUT / "summary.json", "w") as f:
        json.dump({
            "VTH0": float(xo[0]), "LTW_nm": float(xo[1]*1e9),
            "L_NONLOCAL_nm": float(xo[2]*1e9),
            "BJT_AREA": float(10**xo[3]), "Rb": float(10**xo[4]),
            "ALPHA0": float(BASE.ALPHA0*10**xo[5]),
            "median_log_rmse": float(np.median(rs)),
            "p90_log_rmse": float(np.percentile(rs, 90)),
            "worst_log_rmse": float(np.max(rs)),
            "lambda_shape": LAMBDA_SHAPE,
            "per_curve": per_curve,
        }, f, indent=2)

    # Overlay — same three VG1 panels
    shown = {}
    for vg1, vg2, vd, idd in CURVES:
        if vg1 not in shown and abs(vg2) < 0.06:
            shown[vg1] = (vg2, vd, idd, trace(vg1, vd, p))
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.3), sharey=True)
    for ax, vg1 in zip(axes, sorted(shown)):
        vg2, vd, idd, pred = shown[vg1]
        m = (idd > 1e-13) & (pred > 0)
        r = float(np.sqrt(np.mean((np.log10(idd[m]) - np.log10(pred[m]))**2))) \
             if m.any() else float("nan")
        ax.semilogy(vd, np.clip(idd, 1e-15, None), "k-", lw=2, label="meas")
        ax.semilogy(vd, np.clip(pred, 1e-22, None), "g-", lw=1.7,
                     label=f"fit ({r:.2f})")
        ax.set_title(f"VG1={vg1}"); ax.set_xlabel("Vd [V]")
        ax.grid(alpha=0.3, which="both"); ax.legend(fontsize=8)
    axes[0].set_ylabel("|Id| [A]")
    fig.suptitle(f"Shape-aware fit (λ={LAMBDA_SHAPE}) — "
                  f"median {np.median(rs):.2f} worst {np.max(rs):.2f} "
                  f"| Rb=10^{xo[4]:.2f}")
    fig.tight_layout(); fig.savefig(OUT / "overlay.png", dpi=140); plt.close(fig)
    print(f"Wrote {OUT/'overlay.png'}")


if __name__ == "__main__":
    main()
