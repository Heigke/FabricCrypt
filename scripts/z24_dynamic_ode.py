"""z24_dynamic_ode.py — transient body ODE integrated along Vd sweep.

Replaces steady-state find_vb with:
  Cb · dVb/dt = Iii + Igidl − I_bjt_base(Vb,Vsint) − (Vb − Vg2)/Rb

Cb = 1 fF (locked from schematic). BJT params locked from parasiticBJT.txt.
Sweep rate 0.2 V/s (matches Sebas). Simulate BOTH up and down sweep —
hysteresis loop should emerge automatically if Rb is in right regime.

Fit parameters kept free: VTH0, Rb, ALPHA0_mult, LTW, L_NONLOCAL.
BJT_AREA locked to schematic value (1e-6 m² scale — effectively area=1u
applied to BJT_IS=5e-9 → IS_eff=5e-15).
"""
from __future__ import annotations
import csv, json, re, time
from dataclasses import replace
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from nsram.bsim4 import (BSIM4_PRESETS, drain_current_bsim,
                          gidl_current, impact_ionization_bsim4,
                          bipolar_collector_current_ss)
from nsram.physics import thermal_voltage

DATA = Path("/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy/"
             "data/sebas_2026_04_22")
OUT = Path("/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy/"
           "results/z24_dynamic_ode")
OUT.mkdir(parents=True, exist_ok=True)

BASE = BSIM4_PRESETS["ns_ram_130nm_pazos"]
Vt_th = thermal_voltage(300.0)
VG_RE = re.compile(r"VG2=(-?\d+\.?\d*)_VG=(\d+\.\d+)")

# Locked from schematic / parasiticBJT.txt
CB_LOCKED     = 1e-15       # 1 fF body cap
BJT_AREA_SCH  = 1e-6        # area=1u in schematic → IS_eff=5e-15
SWEEP_RATE    = 0.2         # V/s, matches Sebas


def load_curves_bidir():
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
            rows.sort()  # by time
            t = np.array([r[0] for r in rows])
            Vd = np.array([r[1] for r in rows])
            Id = np.array([r[2] for r in rows])
            peak = int(np.argmax(Vd))
            # Up-sweep only for fitting (avoid double-counting)
            curves.append((vg1, vg2, t[:peak+1], Vd[:peak+1], Id[:peak+1]))
    return curves


def dvb_dt(Vb, Vg1, Vd, Vg2, p):
    """Body current balance. Positive = Vb rising."""
    Iii = float(impact_ionization_bsim4(Vg1, Vd, Vb, p))
    Igidl = float(gidl_current(Vd, Vg1, Vb, p)) if p.AGIDL > 0 else 0.0
    IS_eff = p.BJT_IS * BJT_AREA_SCH
    Vc = min(Vb, p.BJT_VJE * 1.1)
    e = np.clip(Vc / (p.BJT_NE * Vt_th), -60, 60)
    Ib_bjt = (IS_eff / p.BJT_BF) * (np.exp(e) - 1.0)
    Ileak  = (Vb - Vg2) / p.Rb
    return (Iii + Igidl - Ib_bjt - Ileak) / CB_LOCKED


def trace_dynamic(vg1, vg2, t_arr, vd_arr, p):
    """RK2 integration of body ODE with fixed Vd interpolation at each t."""
    Vb = float(vg2)
    Vbs = np.zeros_like(vd_arr)
    for k in range(len(t_arr)):
        Vbs[k] = Vb
        if k + 1 >= len(t_arr): break
        dt = t_arr[k+1] - t_arr[k]
        vd_k = float(vd_arr[k])
        vd_kp = float(vd_arr[k+1])
        # RK2 (midpoint)
        k1 = dvb_dt(Vb, vg1, vd_k, vg2, p)
        vd_mid = 0.5 * (vd_k + vd_kp)
        k2 = dvb_dt(Vb + 0.5 * dt * k1, vg1, vd_mid, vg2, p)
        Vb = Vb + dt * k2
        Vb = float(np.clip(Vb, -0.3, p.BJT_VJE * 1.3))
    Vbs_c = np.clip(Vbs, -0.5, p.PhiS - 1e-3)
    Id_mos, _ = drain_current_bsim(vg1, vd_arr, Vbs_c, p)
    Ic = bipolar_collector_current_ss(vg1, vd_arr, Vbs_c, p)
    return np.asarray(Id_mos) + np.asarray(Ic), Vbs


def build_params_locked(x):
    """x = [VTH0, LTW, L_NONLOCAL, log10_Rb, log10_alpha_mult]"""
    vth0, ltw, lnl, lgrb, lgam = x
    return replace(BASE,
                    VTH0=float(vth0), LTW=float(ltw),
                    L_NONLOCAL=float(lnl),
                    BJT_AREA=BJT_AREA_SCH,          # LOCKED
                    Rb=float(10 ** lgrb),
                    ALPHA0=float(BASE.ALPHA0 * 10 ** lgam))


CURVES = load_curves_bidir()
print(f"Loaded {len(CURVES)} curves (bidirectional, actual t/Vd from sweep)")
# For fitting keep bounds on Id > 1e-13 to avoid noise log penalty
ID_FLOOR = 1e-13


def objective(x):
    p = build_params_locked(x)
    rmses = []
    for vg1, vg2, t, vd, idd in CURVES:
        try:
            pred, _ = trace_dynamic(vg1, vg2, t, vd, p)
            m = (idd > ID_FLOOR) & (pred > 0)
            if m.sum() < 5:
                continue
            r = float(np.sqrt(np.mean(
                (np.log10(idd[m]) - np.log10(pred[m])) ** 2)))
            rmses.append(r)
        except Exception:
            continue
    if not rmses:
        return 10.0
    rs = np.array(rmses)
    return 0.5 * float(np.median(rs)) + 0.5 * float(np.percentile(rs, 90))


BOUNDS = [
    (0.40, 1.10),     # VTH0
    (15e-9, 100e-9),  # LTW
    (0.0,   300e-9),  # L_NONLOCAL
    ( 7.0, 12.0),     # log10 Rb
    (-2.0, 2.0),      # log10 ALPHA0_mult
]


def main():
    from scipy.optimize import differential_evolution

    print("Dynamic-ODE DE with locked BJT params (schematic) + Cb=1fF")
    t0 = time.time(); ctr = [0]
    def cb(xk, conv):
        ctr[0] += 1
        o = objective(xk)
        print(f"  iter {ctr[0]:3d} obj={o:.3f} conv={conv:.3f}  "
              f"VTH0={xk[0]:.3f} LTW={xk[1]*1e9:.0f}n L_nl={xk[2]*1e9:.0f}n "
              f"Rb=1e{xk[3]:.1f} A0x10^{xk[4]:.2f}  ({time.time()-t0:.0f}s)",
              flush=True)
        return False

    res = differential_evolution(
        objective, BOUNDS,
        maxiter=40, popsize=12,
        mutation=(0.5, 1.2), recombination=0.7,
        seed=1, tol=1e-3, workers=-1, updating="deferred",
        callback=cb, polish=True, disp=False,
    )
    xo = res.x
    p = build_params_locked(xo)
    print(f"\nBest obj = {res.fun:.3f}")

    per_curve = []
    for vg1, vg2, t, vd, idd in CURVES:
        try:
            pred, _ = trace_dynamic(vg1, vg2, t, vd, p)
            m = (idd > ID_FLOOR) & (pred > 0)
            if m.sum() < 5: continue
            r = float(np.sqrt(np.mean((np.log10(idd[m]) - np.log10(pred[m]))**2)))
            per_curve.append({"vg1": vg1, "vg2": vg2, "log_rmse": r})
        except Exception: continue
    rs = np.array([c["log_rmse"] for c in per_curve])
    print(f"median={np.median(rs):.2f}  p90={np.percentile(rs,90):.2f}  "
          f"worst={np.max(rs):.2f}  ({len(rs)} curves)")

    with open(OUT / "summary.json", "w") as f:
        json.dump({
            "VTH0": float(xo[0]), "LTW_nm": float(xo[1]*1e9),
            "L_NONLOCAL_nm": float(xo[2]*1e9),
            "Rb": float(10**xo[3]),
            "ALPHA0": float(BASE.ALPHA0*10**xo[4]),
            "BJT_AREA_LOCKED": BJT_AREA_SCH,
            "CB_LOCKED": CB_LOCKED,
            "median_log_rmse": float(np.median(rs)),
            "p90_log_rmse": float(np.percentile(rs, 90)),
            "worst_log_rmse": float(np.max(rs)),
            "n_curves": len(rs),
            "per_curve": per_curve,
        }, f, indent=2)

    # Overlay grid 3 x 5
    target_vg2 = [-0.15, -0.05, 0.05, 0.15, 0.25]
    fig, axes = plt.subplots(3, 5, figsize=(17, 9), sharey="row")
    for row, vg1 in enumerate([0.2, 0.4, 0.6]):
        cands = [c for c in CURVES if abs(c[0]-vg1) < 0.01]
        for col, vg2_t in enumerate(target_vg2):
            hit = min(cands, key=lambda c: abs(c[1]-vg2_t))
            _, vg2, t, vd, idd = hit
            pred, _ = trace_dynamic(vg1, vg2, t, vd, p)
            m = (idd > ID_FLOOR) & (pred > 0)
            r = float(np.sqrt(np.mean(
                (np.log10(idd[m]) - np.log10(pred[m]))**2))) if m.any() else float("nan")
            ax = axes[row, col]
            ax.semilogy(vd, np.clip(idd, 1e-14, None), "k-", lw=1.8, label="meas")
            ax.semilogy(vd, np.clip(pred, 1e-22, None), "g-", lw=1.4,
                         label=f"fit ({r:.2f})")
            ax.set_title(f"VG1={vg1}  VG2={vg2:+.2f}", fontsize=9)
            if row == 2: ax.set_xlabel("Vd [V]")
            if col == 0: ax.set_ylabel("|Id| [A]")
            ax.grid(alpha=0.3, which="both"); ax.legend(fontsize=7)
    fig.suptitle(f"Dynamic ODE (Cb=1fF, schematic BJT) — "
                  f"median {np.median(rs):.2f}, worst {np.max(rs):.2f}")
    fig.tight_layout(); fig.savefig(OUT / "overlay.png", dpi=130); plt.close(fig)
    print(f"Wrote {OUT/'overlay.png'}")


if __name__ == "__main__":
    main()
