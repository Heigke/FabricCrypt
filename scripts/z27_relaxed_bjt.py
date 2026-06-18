"""z27_relaxed_bjt.py — let BJT_AREA be a fit param in a *physical* range.

Rationale:
  - Schematic says area=1u (→ 1e-6 multiplier → IS_eff=5e-15 A)
  - With that, max BJT collector current at latch ≈ 1–2 μA, but data
    shows 10–100 μA post-knee. So parasiticBJT.txt underestimates the
    real BJT by ~50×.
  - Allow BJT_AREA ∈ [1e-8, 1e-3]: 5 decades around schematic nominal.
    Real silicon can vary lumped effective area by 10-100× from schematic
    nominal due to layout-dependent effects, so this range is physical.
  - Keep BJT internal params (IS, BF, NE, VJE, VA) LOCKED from Sebas's
    card — these are fundamental device parameters.
  - Keep Cb=1fF locked from schematic.
"""
from __future__ import annotations
import json, time
from pathlib import Path
from dataclasses import replace

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import differential_evolution, brentq

from scripts.z25_qs_bidirectional import (
    CURVES, trace_bidir, curve_rmse, BASE, kcl as kcl_ref,
)

# Re-define kcl so BJT_AREA is taken from params, not locked constant
from nsram.bsim4 import (drain_current_bsim, gidl_current,
                          impact_ionization_bsim4, bipolar_collector_current_ss)
from nsram.physics import thermal_voltage
Vt_th = thermal_voltage(300.0)

OUT = Path("/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy/"
           "results/z27_relaxed_bjt")
OUT.mkdir(parents=True, exist_ok=True)
ID_FLOOR = 1e-13


def kcl(Vb, Vg1, Vg2, Vd, p):
    Iii = float(impact_ionization_bsim4(Vg1, Vd, Vb, p))
    Ig  = float(gidl_current(Vd, Vg1, Vb, p)) if p.AGIDL > 0 else 0.0
    IS  = p.BJT_IS * p.BJT_AREA      # ← now uses fit value
    Vc  = min(Vb, p.BJT_VJE * 1.1)
    e   = np.clip(Vc / (p.BJT_NE * Vt_th), -60, 60)
    Io  = (IS / p.BJT_BF) * (np.exp(e) - 1.0) + (Vb - Vg2) / p.Rb
    return (Iii + Ig) - Io


def find_vb_continuation(Vg1, Vg2, Vd, p, Vb0, vmax=0.85):
    for dV in (0.02, 0.05, 0.15):
        lo, hi = max(-0.2, Vb0 - dV), min(vmax, Vb0 + dV)
        try:
            if kcl(lo, Vg1, Vg2, Vd, p) * kcl(hi, Vg1, Vg2, Vd, p) <= 0:
                return float(brentq(kcl, lo, hi, args=(Vg1, Vg2, Vd, p),
                                      xtol=1e-5, rtol=1e-5))
        except ValueError: continue
    g = np.linspace(-0.2, vmax, 121)
    f = np.array([kcl(v, Vg1, Vg2, Vd, p) for v in g])
    sc = np.where(np.sign(f[:-1]) != np.sign(f[1:]))[0]
    roots = []
    for i in sc:
        try:
            roots.append(brentq(kcl, g[i], g[i+1], args=(Vg1, Vg2, Vd, p),
                                  xtol=1e-5, rtol=1e-5))
        except ValueError: continue
    if not roots:
        return vmax if Vb0 > 0.4 else float(g[0])
    return float(min(roots, key=lambda r: abs(r - Vb0)))


def trace_bidir_local(vg1, vg2, vd_up, vd_dn, p):
    Vb = float(vg2)
    Vbs_up = np.zeros_like(vd_up)
    for k, v in enumerate(vd_up):
        Vb = find_vb_continuation(vg1, vg2, float(v), p, Vb0=Vb)
        Vbs_up[k] = Vb
    Vb_peak = Vb
    Vbs_dn = np.zeros_like(vd_dn)
    Vb = Vb_peak
    for k in range(len(vd_dn) - 1, -1, -1):
        v = float(vd_dn[k])
        Vb = find_vb_continuation(vg1, vg2, v, p, Vb0=Vb)
        Vbs_dn[k] = Vb

    def id_from_vbs(vd, vbs):
        Vc = np.clip(vbs, -0.5, p.PhiS - 1e-3)
        Ids, _ = drain_current_bsim(vg1, vd, Vc, p)
        Ic = bipolar_collector_current_ss(vg1, vd, Vc, p)
        return np.asarray(Ids) + np.asarray(Ic)
    return id_from_vbs(vd_up, Vbs_up), id_from_vbs(vd_dn, Vbs_dn), Vbs_up, Vbs_dn


def build_params(x):
    """x = [VTH0, LTW, L_NONLOCAL, log10_Rb, log10_alpha_mult, log10_area]"""
    vth0, ltw, lnl, lgrb, lgam, lga = x
    return replace(BASE,
                    VTH0=float(vth0), LTW=float(ltw),
                    L_NONLOCAL=float(lnl),
                    Rb=float(10 ** lgrb),
                    ALPHA0=float(BASE.ALPHA0 * 10 ** lgam),
                    BJT_AREA=float(10 ** lga))


def objective(x):
    p = build_params(x)
    rmses = []
    for vg1, vg2, vu, iu, vd, id_ in CURVES:
        try:
            pu, pd, _, _ = trace_bidir_local(vg1, vg2, vu, vd, p)
            r_up = curve_rmse(pu, iu); r_dn = curve_rmse(pd, id_)
            if r_up is not None: rmses.append(r_up)
            if r_dn is not None: rmses.append(r_dn)
        except Exception: continue
    if not rmses: return 10.0
    rs = np.array(rmses)
    return 0.5 * float(np.median(rs)) + 0.5 * float(np.percentile(rs, 90))


BOUNDS = [
    (0.30, 1.10),     # VTH0 (allow lower than PTM130 nominal 0.54)
    (15e-9, 100e-9),  # LTW
    (0.0,   300e-9),  # L_NONLOCAL
    ( 7.0, 12.0),     # log10 Rb
    (-3.0, 2.0),      # log10 ALPHA0_mult
    (-8.0, -3.0),     # log10 BJT_AREA [physical: 5 decades around schematic 1e-6]
]


def main():
    print(f"DE: 6 params, BJT_AREA fit in [1e-8, 1e-3], {len(CURVES)} curves")
    t0 = time.time(); ctr = [0]

    def cb(xk, conv):
        ctr[0] += 1
        o = objective(xk)
        print(f"  iter {ctr[0]:3d} obj={o:.3f}  VTH0={xk[0]:.3f} "
              f"Rb=1e{xk[3]:.1f} A0x10^{xk[4]:.2f} AREA=1e{xk[5]:.1f} "
              f"({time.time()-t0:.0f}s)", flush=True)
        return False

    res = differential_evolution(
        objective, BOUNDS,
        maxiter=40, popsize=14,
        mutation=(0.5, 1.2), recombination=0.7,
        seed=7, tol=1e-3, workers=-1, updating="deferred",
        callback=cb, polish=False,     # skip polish (hung last time)
        disp=False,
    )
    xo = res.x
    p = build_params(xo)
    print(f"\nBest obj = {res.fun:.3f}")
    print(f"  VTH0={xo[0]:.3f}  LTW={xo[1]*1e9:.0f}n  L_nl={xo[2]*1e9:.0f}n")
    print(f"  Rb=1e{xo[3]:.2f}  ALPHA0×10^{xo[4]:.2f}  BJT_AREA=1e{xo[5]:.2f}")

    per_curve = []
    for vg1, vg2, vu, iu, vd, id_ in CURVES:
        try:
            pu, pd, _, _ = trace_bidir_local(vg1, vg2, vu, vd, p)
            r_up = curve_rmse(pu, iu); r_dn = curve_rmse(pd, id_)
            per_curve.append({"vg1": vg1, "vg2": vg2,
                                "rmse_up": r_up, "rmse_dn": r_dn})
        except Exception: continue
    ups = np.array([c["rmse_up"] for c in per_curve if c["rmse_up"] is not None])
    dns = np.array([c["rmse_dn"] for c in per_curve if c["rmse_dn"] is not None])
    print(f"Up: median={np.median(ups):.2f}  p90={np.percentile(ups,90):.2f}  worst={np.max(ups):.2f}")
    print(f"Dn: median={np.median(dns):.2f}  p90={np.percentile(dns,90):.2f}  worst={np.max(dns):.2f}")

    with open(OUT / "summary.json", "w") as f:
        json.dump({
            "VTH0": float(xo[0]), "LTW_nm": float(xo[1]*1e9),
            "L_NONLOCAL_nm": float(xo[2]*1e9),
            "Rb": float(10**xo[3]), "ALPHA0": float(BASE.ALPHA0*10**xo[4]),
            "BJT_AREA": float(10**xo[5]),
            "median_up": float(np.median(ups)),
            "median_dn": float(np.median(dns)),
            "worst_up": float(np.max(ups)),
            "worst_dn": float(np.max(dns)),
            "per_curve": per_curve,
        }, f, indent=2)

    # Overlay grid
    target_vg2 = [-0.15, -0.05, 0.05, 0.15, 0.25]
    fig, axes = plt.subplots(3, 5, figsize=(17, 9), sharey="row")
    for row, vg1 in enumerate([0.2, 0.4, 0.6]):
        cands = [c for c in CURVES if abs(c[0]-vg1) < 0.01]
        for col, vg2_t in enumerate(target_vg2):
            hit = min(cands, key=lambda c: abs(c[1]-vg2_t))
            _, vg2, vu, iu, vd, id_ = hit
            pu, pd, _, _ = trace_bidir_local(vg1, vg2, vu, vd, p)
            ax = axes[row, col]
            ax.semilogy(vu, np.clip(iu, 1e-14, None), "k-", lw=1.6, label="meas up")
            ax.semilogy(vd, np.clip(id_, 1e-14, None), "k:", lw=1.0, label="meas dn")
            ax.semilogy(vu, np.clip(pu, 1e-22, None), "g-", lw=1.2, label="fit up")
            ax.semilogy(vd, np.clip(pd, 1e-22, None), "g:", lw=1.0, label="fit dn")
            ax.set_title(f"VG1={vg1}  VG2={vg2:+.2f}", fontsize=8)
            if row == 2: ax.set_xlabel("Vd [V]")
            if col == 0: ax.set_ylabel("|Id| [A]")
            ax.grid(alpha=0.3, which="both"); ax.legend(fontsize=6)
    fig.suptitle(f"Fit with physical-range BJT_AREA — "
                  f"up med {np.median(ups):.2f}  dn med {np.median(dns):.2f}  "
                  f"| AREA=1e{xo[5]:.1f} (sch=1e-6), Rb=1e{xo[3]:.1f}")
    fig.tight_layout(); fig.savefig(OUT / "overlay.png", dpi=130); plt.close(fig)
    print(f"Wrote {OUT/'overlay.png'}")


if __name__ == "__main__":
    main()
