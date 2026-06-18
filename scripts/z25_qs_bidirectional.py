"""z25_qs_bidirectional.py — quasi-static with locked BJT area + bidirectional
sweep (up+down) to reproduce hysteresis.

Key insight from z24: with Cb=1fF, the body RC time-constant is μs — much
faster than Sebas's 0.2 V/s sweep. Dynamic ODE adds no information;
hysteresis comes from *bifurcation continuation*, not from transient dynamics.

Approach:
  - Lock BJT_AREA = 1e-6 (schematic area=1u → IS_eff = 5e-15)
  - Lock BJT parameters from parasiticBJT.txt
  - Fit only {VTH0, LTW, L_NONLOCAL, Rb, ALPHA0_mult}
  - Run BOTH up-sweep (Vb starts at VG2) and down-sweep (Vb starts at latched
    state reached at peak Vd). Score against BOTH branches of measured data.
"""
from __future__ import annotations
import csv, json, re, time
from dataclasses import replace
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import brentq, differential_evolution

from nsram.bsim4 import (BSIM4_PRESETS, drain_current_bsim,
                          gidl_current, impact_ionization_bsim4,
                          bipolar_collector_current_ss)
from nsram.physics import thermal_voltage

DATA = Path("/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy/"
             "data/sebas_2026_04_22")
OUT = Path("/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy/"
           "results/z25_qs_bidirectional")
OUT.mkdir(parents=True, exist_ok=True)

BASE = BSIM4_PRESETS["ns_ram_130nm_pazos"]
Vt_th = thermal_voltage(300.0)
VG_RE = re.compile(r"VG2=(-?\d+\.?\d*)_VG=(\d+\.\d+)")
BJT_AREA_SCH = 1e-6       # area=1u from schematic
ID_FLOOR = 1e-13
N_DS = 30                 # points per sweep direction


def load_curves_bidir():
    """Return (vg1, vg2, vd_up, id_up, vd_dn, id_dn) per curve."""
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
            Vd = np.array([r[1] for r in rows])
            Id = np.array([r[2] for r in rows])
            peak = int(np.argmax(Vd))
            Vd_up, Id_up = Vd[:peak+1], Id[:peak+1]
            Vd_dn, Id_dn = Vd[peak:][::-1], Id[peak:][::-1]  # reverse: 0→2V for dn too
            def resample(V, I):
                m = I > ID_FLOOR
                if m.sum() < 10: return None, None
                V, I = V[m], I[m]
                uV, idx = np.unique(V, return_index=True)
                I = I[idx]
                nv = np.linspace(max(0.1, uV.min()), uV.max(), N_DS)
                ni = 10 ** np.interp(nv, uV, np.log10(I))
                return nv, ni
            vu, iu = resample(Vd_up, Id_up)
            vd, id_ = resample(Vd_dn, Id_dn)
            if vu is None or vd is None: continue
            curves.append((vg1, vg2, vu, iu, vd, id_))
    return curves


CURVES = load_curves_bidir()
print(f"Loaded {len(CURVES)} curves (up + down sweep resampled)")


def kcl(Vb, Vg1, Vg2, Vd, p):
    Iii = float(impact_ionization_bsim4(Vg1, Vd, Vb, p))
    Ig  = float(gidl_current(Vd, Vg1, Vb, p)) if p.AGIDL > 0 else 0.0
    IS  = p.BJT_IS * BJT_AREA_SCH
    Vc  = min(Vb, p.BJT_VJE * 1.1)
    e   = np.clip(Vc / (p.BJT_NE * Vt_th), -60, 60)
    Io  = (IS / p.BJT_BF) * (np.exp(e) - 1.0) + (Vb - Vg2) / p.Rb
    return (Iii + Ig) - Io


def find_vb_continuation(Vg1, Vg2, Vd, p, Vb0, vmax=0.85):
    """Find root nearest Vb0 (branch continuation)."""
    # Local first
    for dV in (0.02, 0.05, 0.15):
        lo, hi = max(-0.2, Vb0 - dV), min(vmax, Vb0 + dV)
        try:
            if kcl(lo, Vg1, Vg2, Vd, p) * kcl(hi, Vg1, Vg2, Vd, p) <= 0:
                return float(brentq(kcl, lo, hi, args=(Vg1, Vg2, Vd, p),
                                      xtol=1e-5, rtol=1e-5))
        except ValueError:
            continue
    # Global enumeration
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
        # Saddle-node collapsed; pick edge closer to previous branch
        return vmax if Vb0 > 0.4 else float(g[0])
    return float(min(roots, key=lambda r: abs(r - Vb0)))


def trace_bidir(vg1, vg2, vd_up, vd_dn, p):
    """Both sweeps. Up starts at Vb=VG2; down starts at whatever Vb we
    reached at peak Vd on the up sweep."""
    # Up
    Vb = float(vg2)
    Vbs_up = np.zeros_like(vd_up)
    for k, v in enumerate(vd_up):
        Vb = find_vb_continuation(vg1, vg2, float(v), p, Vb0=Vb)
        Vbs_up[k] = Vb
    Vb_peak = Vb
    # Down: sweep from peak Vd back to 0.1V, starting at latched Vb
    Vbs_dn = np.zeros_like(vd_dn)
    Vb = Vb_peak
    # vd_dn was resampled 0→2V; iterate REVERSED so physically going 2→0
    for k in range(len(vd_dn) - 1, -1, -1):
        v = float(vd_dn[k])
        Vb = find_vb_continuation(vg1, vg2, v, p, Vb0=Vb)
        Vbs_dn[k] = Vb
    # Compute Id on both
    def id_from_vbs(vd, vbs):
        Vc = np.clip(vbs, -0.5, p.PhiS - 1e-3)
        Ids, _ = drain_current_bsim(vg1, vd, Vc, p)
        Ic = bipolar_collector_current_ss(vg1, vd, Vc, p)
        return np.asarray(Ids) + np.asarray(Ic)
    return id_from_vbs(vd_up, Vbs_up), id_from_vbs(vd_dn, Vbs_dn), Vbs_up, Vbs_dn


def build_params_locked(x):
    """x = [VTH0, LTW, L_NONLOCAL, log10_Rb, log10_alpha_mult]"""
    vth0, ltw, lnl, lgrb, lgam = x
    return replace(BASE,
                    VTH0=float(vth0), LTW=float(ltw),
                    L_NONLOCAL=float(lnl),
                    BJT_AREA=BJT_AREA_SCH,
                    Rb=float(10 ** lgrb),
                    ALPHA0=float(BASE.ALPHA0 * 10 ** lgam))


def curve_rmse(pred, meas):
    m = (meas > ID_FLOOR) & (pred > 0)
    if m.sum() < 5: return None
    return float(np.sqrt(np.mean((np.log10(meas[m]) - np.log10(pred[m]))**2)))


def objective(x):
    p = build_params_locked(x)
    rmses = []
    for vg1, vg2, vu, iu, vd, id_ in CURVES:
        try:
            pu, pd, _, _ = trace_bidir(vg1, vg2, vu, vd, p)
            r_up = curve_rmse(pu, iu)
            r_dn = curve_rmse(pd, id_)
            if r_up is not None: rmses.append(r_up)
            if r_dn is not None: rmses.append(r_dn)
        except Exception:
            continue
    if not rmses: return 10.0
    rs = np.array(rmses)
    return 0.5 * float(np.median(rs)) + 0.5 * float(np.percentile(rs, 90))


BOUNDS = [
    (0.40, 1.10),     # VTH0
    (15e-9, 100e-9),  # LTW
    (0.0,   300e-9),  # L_NONLOCAL
    ( 7.0, 12.0),     # log10 Rb
    (-3.0, 2.0),      # log10 ALPHA0_mult
]


def main():
    print("QS bidirectional DE with locked schematic BJT")
    t0 = time.time(); ctr = [0]

    def cb(xk, conv):
        ctr[0] += 1
        o = objective(xk)
        print(f"  iter {ctr[0]:3d} obj={o:.3f}  VTH0={xk[0]:.3f} "
              f"LTW={xk[1]*1e9:.0f}n L_nl={xk[2]*1e9:.0f}n Rb=1e{xk[3]:.1f} "
              f"A0x10^{xk[4]:.2f}  ({time.time()-t0:.0f}s)", flush=True)
        return False

    res = differential_evolution(
        objective, BOUNDS,
        maxiter=50, popsize=12,
        mutation=(0.5, 1.2), recombination=0.7,
        seed=3, tol=1e-3, workers=-1, updating="deferred",
        callback=cb, polish=True, disp=False,
    )
    xo = res.x
    p = build_params_locked(xo)
    print(f"\nBest obj = {res.fun:.3f}")

    per_curve = []
    for vg1, vg2, vu, iu, vd, id_ in CURVES:
        try:
            pu, pd, _, _ = trace_bidir(vg1, vg2, vu, vd, p)
            r_up = curve_rmse(pu, iu); r_dn = curve_rmse(pd, id_)
            per_curve.append({"vg1": vg1, "vg2": vg2,
                                "rmse_up": r_up, "rmse_dn": r_dn})
        except Exception: continue
    ups = np.array([c["rmse_up"] for c in per_curve if c["rmse_up"] is not None])
    dns = np.array([c["rmse_dn"] for c in per_curve if c["rmse_dn"] is not None])
    print(f"Up: median={np.median(ups):.2f} worst={np.max(ups):.2f}")
    print(f"Dn: median={np.median(dns):.2f} worst={np.max(dns):.2f}")

    with open(OUT / "summary.json", "w") as f:
        json.dump({
            "VTH0": float(xo[0]), "LTW_nm": float(xo[1]*1e9),
            "L_NONLOCAL_nm": float(xo[2]*1e9),
            "Rb": float(10**xo[3]), "ALPHA0": float(BASE.ALPHA0*10**xo[4]),
            "BJT_AREA_LOCKED": BJT_AREA_SCH,
            "median_up": float(np.median(ups)),
            "median_dn": float(np.median(dns)),
            "worst_up": float(np.max(ups)),
            "worst_dn": float(np.max(dns)),
            "per_curve": per_curve,
        }, f, indent=2)

    # Overlay 3 × 5 with both branches
    target_vg2 = [-0.15, -0.05, 0.05, 0.15, 0.25]
    fig, axes = plt.subplots(3, 5, figsize=(17, 9), sharey="row")
    for row, vg1 in enumerate([0.2, 0.4, 0.6]):
        cands = [c for c in CURVES if abs(c[0]-vg1) < 0.01]
        for col, vg2_t in enumerate(target_vg2):
            hit = min(cands, key=lambda c: abs(c[1]-vg2_t))
            _, vg2, vu, iu, vd, id_ = hit
            pu, pd, _, _ = trace_bidir(vg1, vg2, vu, vd, p)
            ax = axes[row, col]
            ax.semilogy(vu, np.clip(iu, 1e-14, None), "k-", lw=1.5, label="meas up")
            ax.semilogy(vd, np.clip(id_, 1e-14, None), "k:", lw=1.0, label="meas dn")
            ax.semilogy(vu, np.clip(pu, 1e-22, None), "g-", lw=1.2, label="fit up")
            ax.semilogy(vd, np.clip(pd, 1e-22, None), "g:", lw=1.0, label="fit dn")
            ax.set_title(f"VG1={vg1}  VG2={vg2:+.2f}", fontsize=8)
            if row == 2: ax.set_xlabel("Vd [V]")
            if col == 0: ax.set_ylabel("|Id| [A]")
            ax.grid(alpha=0.3, which="both"); ax.legend(fontsize=6)
    fig.suptitle(f"QS bidirectional fit — up med {np.median(ups):.2f}  dn med {np.median(dns):.2f}  "
                  f"(BJT_AREA=1u locked, Rb=1e{xo[3]:.1f})")
    fig.tight_layout(); fig.savefig(OUT / "overlay.png", dpi=130); plt.close(fig)
    print(f"Wrote {OUT/'overlay.png'}")


if __name__ == "__main__":
    main()
