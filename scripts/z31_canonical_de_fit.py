"""z31_canonical_de_fit.py — DE fit on Sebas data using canonical NSRAM model.

This is the test of: does using the validated canonical BSIM4 (instead of
the simplified one) give a better fit than z25 (median 0.74)?
"""
from __future__ import annotations
import csv, json, re, time
from pathlib import Path
from dataclasses import replace

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import differential_evolution

from nsram.nsram_canonical import trace_nsram, make_pazos_130nm

DATA = Path("/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy/"
             "data/sebas_2026_04_22")
OUT = Path("/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy/"
           "results/z31_canonical_de_fit")
OUT.mkdir(parents=True, exist_ok=True)
VG_RE = re.compile(r"VG2=(-?\d+\.?\d*)_VG=(\d+\.\d+)")
ID_FLOOR = 1e-13
N_DS = 20


def load_curves():
    curves = []
    for sub in sorted(DATA.iterdir()):
        if not sub.is_dir(): continue
        for fn in sorted(sub.glob("*.csv")):
            m = VG_RE.search(fn.name)
            if not m: continue
            vg2 = float(m.group(1)); vg1 = float(m.group(2))
            rows = []
            with open(fn) as f:
                rdr = csv.reader(f); next(rdr)
                for r in rdr:
                    try: rows.append((float(r[2]), float(r[0]), float(r[1])))
                    except ValueError: continue
            rows.sort()
            Vd = np.array([r[1] for r in rows]); Id = np.array([r[2] for r in rows])
            peak = int(np.argmax(Vd))
            Vd = Vd[:peak+1]; Id = Id[:peak+1]
            mask = Id > ID_FLOOR
            if mask.sum() < 10: continue
            Vd, Id = Vd[mask], Id[mask]
            uVd, idx = np.unique(Vd, return_index=True)
            Id = Id[idx]; Vd = uVd
            nvd = np.linspace(max(0.1, Vd.min()), Vd.max(), N_DS)
            nid = np.power(10.0, np.interp(nvd, Vd, np.log10(Id)))
            curves.append((vg1, vg2, nvd, nid))
    return curves


CURVES = load_curves()
print(f"Loaded {len(CURVES)} curves × {N_DS} pts")


def build(x):
    """x = [VTH0, log10_AREA, log10_alpha, log10_Rb, U0]"""
    vth0, lga, lgam, lgrb, u0 = x
    return replace(make_pazos_130nm(),
                    VTH0=float(vth0),
                    BJT_AREA=float(10**lga),
                    ALPHA0_mult=float(10**lgam),
                    Rb=float(10**lgrb),
                    U0=float(u0))


def objective(x):
    p = build(x)
    rmses = []
    for vg1, vg2, vd, idd in CURVES:
        try:
            pred, _ = trace_nsram(vg1, vg2, vd, p)
            m = (idd > ID_FLOOR) & (pred > 0)
            if m.sum() < 5: continue
            rmses.append(float(np.sqrt(np.mean(
                (np.log10(idd[m]) - np.log10(pred[m]))**2))))
        except Exception:
            continue
    if not rmses: return 10.0
    rs = np.array(rmses)
    return 0.5 * float(np.median(rs)) + 0.5 * float(np.percentile(rs, 90))


BOUNDS = [
    (0.40, 1.00),     # VTH0
    (-7.0, -3.0),     # log10 BJT_AREA
    (-3.0, 1.0),      # log10 ALPHA0_mult
    ( 7.0, 12.0),     # log10 Rb
    (0.02, 0.10),     # U0
]


def main():
    print("DE on canonical NSRAM (5 params)")
    t0 = time.time(); ctr = [0]
    def cb(xk, conv):
        ctr[0] += 1
        o = objective(xk)
        print(f"  iter {ctr[0]:3d} obj={o:.3f}  VTH0={xk[0]:.3f} "
              f"AREA=1e{xk[1]:.1f} A0×10^{xk[2]:.2f} Rb=1e{xk[3]:.1f} "
              f"U0={xk[4]:.3f}  ({time.time()-t0:.0f}s)", flush=True)
        return False
    res = differential_evolution(
        objective, BOUNDS,
        maxiter=30, popsize=10,
        mutation=(0.5, 1.2), recombination=0.7,
        seed=42, tol=1e-3, workers=-1, updating="deferred",
        callback=cb, polish=False, disp=False,
    )
    xo = res.x
    p = build(xo)
    print(f"\nBest obj = {res.fun:.3f}")

    per_curve = []
    for vg1, vg2, vd, idd in CURVES:
        try:
            pred, _ = trace_nsram(vg1, vg2, vd, p)
            m = (idd > ID_FLOOR) & (pred > 0)
            if m.sum() < 5: continue
            r = float(np.sqrt(np.mean((np.log10(idd[m]) - np.log10(pred[m]))**2)))
            per_curve.append({"vg1": vg1, "vg2": vg2, "log_rmse": r})
        except Exception: continue
    rs = np.array([c["log_rmse"] for c in per_curve])
    print(f"median={np.median(rs):.2f}  p90={np.percentile(rs,90):.2f}  "
           f"worst={np.max(rs):.2f}")

    with open(OUT / "summary.json", "w") as f:
        json.dump({
            "VTH0": float(xo[0]), "BJT_AREA": float(10**xo[1]),
            "ALPHA0_mult": float(10**xo[2]), "Rb": float(10**xo[3]),
            "U0": float(xo[4]),
            "median_log_rmse": float(np.median(rs)),
            "worst_log_rmse": float(np.max(rs)),
            "n_curves": len(rs),
            "per_curve": per_curve,
        }, f, indent=2)

    target_vg2 = [-0.15, -0.05, 0.05, 0.15, 0.25]
    fig, axes = plt.subplots(3, 5, figsize=(17, 9), sharey="row")
    for row, vg1 in enumerate([0.2, 0.4, 0.6]):
        cands = [c for c in CURVES if abs(c[0]-vg1) < 0.01]
        for col, vg2_t in enumerate(target_vg2):
            hit = min(cands, key=lambda c: abs(c[1]-vg2_t))
            _, vg2, vd, idd = hit
            pred, _ = trace_nsram(vg1, vg2, vd, p)
            m = (idd > ID_FLOOR) & (pred > 0)
            r = float(np.sqrt(np.mean(
                (np.log10(idd[m]) - np.log10(pred[m]))**2))) if m.any() else float("nan")
            ax = axes[row, col]
            ax.semilogy(vd, np.clip(idd, 1e-14, None), "k-", lw=1.6, label="meas")
            ax.semilogy(vd, np.clip(pred, 1e-22, None), "g-", lw=1.2,
                         label=f"fit ({r:.2f})")
            ax.set_title(f"VG1={vg1}  VG2={vg2:+.2f}", fontsize=8)
            if row == 2: ax.set_xlabel("Vd [V]")
            if col == 0: ax.set_ylabel("|Id| [A]")
            ax.grid(alpha=0.3, which="both"); ax.legend(fontsize=6)
    fig.suptitle(f"Canonical NSRAM DE fit — median {np.median(rs):.2f}, "
                  f"worst {np.max(rs):.2f}\n"
                  f"VTH0={xo[0]:.3f} AREA=1e{xo[1]:.1f} A0=1e{xo[2]:.1f} "
                  f"Rb=1e{xo[3]:.1f} U0={xo[4]:.3f}")
    fig.tight_layout(); fig.savefig(OUT / "overlay.png", dpi=130); plt.close(fig)
    print(f"Wrote {OUT/'overlay.png'}")


if __name__ == "__main__":
    main()
