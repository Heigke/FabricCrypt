"""z20_vg2_fit.py — VG2-aware fit.

Fixes identified from z14–z19 diagnostics:
  (1) Back-gate VG2 directly biases the body via Rb: kcl uses (Vb-VG2)/Rb,
      and Vb is initialised to VG2 at Vd=0.  This is the missing physics —
      VG2 shifts the latch threshold ~2V/V in data.
  (2) Sampling densified to 40 Vd points + log-interp to preserve knee.
  (3) Id <= 1e-13 masked (measurement noise / negative values).
  (4) All 33 curves (VG1 × VG2) fit jointly.

Run:
    HSA_OVERRIDE_GFX_VERSION=11.0.0 python -m scripts.z20_vg2_fit
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
                          gidl_current, impact_ionization_bsim4)
from nsram.physics import thermal_voltage


DATA_DIR = Path("/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy/"
                "data/sebas_2026_04_22")
OUT = Path("/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy/"
           "results/z20_vg2_fit")
OUT.mkdir(parents=True, exist_ok=True)

BASE = BSIM4_PRESETS["ns_ram_130nm_pazos"]
Vt_th = thermal_voltage(300.0)
VG_RE = re.compile(r"VG2=(-?\d+\.?\d*)_VG=(\d+\.\d+)")

ID_FLOOR = 1e-13   # below this = noise / negative readings
N_DS = 40          # sample density per curve


def load_curves(n_ds: int = N_DS):
    """Return list of (vg1, vg2, vd, id) with Id>ID_FLOOR only, log-interp."""
    curves = []
    for sub in sorted(DATA_DIR.iterdir()):
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
                rdr = csv.reader(f); next(rdr)
                for r in rdr:
                    try:
                        rows.append((float(r[2]), float(r[0]), float(r[1])))
                    except ValueError:
                        continue
            rows.sort()
            Vd = np.array([r[1] for r in rows])
            Id = np.array([r[2] for r in rows])
            # Keep only upsweep (Vd monotonic to peak)
            peak = int(np.argmax(Vd))
            Vd = Vd[:peak + 1]; Id = Id[:peak + 1]
            # Drop noise / negative points
            mask = Id > ID_FLOOR
            if mask.sum() < 10:
                continue
            Vd = Vd[mask]; Id = Id[mask]
            # Collapse duplicate Vd (sweep tool sometimes double-samples)
            uniq_Vd, idx = np.unique(Vd, return_index=True)
            Id = Id[idx]; Vd = uniq_Vd
            # Log-interpolate onto dense linear Vd grid
            nvd = np.linspace(max(0.1, Vd.min()), Vd.max(), n_ds)
            nid = np.power(10.0, np.interp(nvd, Vd, np.log10(Id)))
            curves.append((vg1, vg2, nvd, nid))
    return curves


CURVES = load_curves()
print(f"Loaded {len(CURVES)} curves (>= {ID_FLOOR} A, {N_DS} pts each)")


# ──────────────────────────────────────────────────────────────
# Physics with VG2
# ──────────────────────────────────────────────────────────────
def kcl(Vb, Vg1, Vg2, Vd, p):
    """Body KCL — note Rb now terminates at VG2 (back-gate contact)."""
    Iii = float(impact_ionization_bsim4(Vg1, Vd, Vb, p))
    Ig  = float(gidl_current(Vd, Vg1, Vb, p)) if p.AGIDL > 0 else 0.0
    IS = p.BJT_IS * max(p.BJT_AREA, 1e-30)
    Vc = min(Vb, p.BJT_VJE * 1.1)
    e  = np.clip(Vc / (p.BJT_NE * Vt_th), -60, 60)
    Io = (IS / p.BJT_BF) * (np.exp(e) - 1.0) + (Vb - Vg2) / p.Rb
    return (Iii + Ig) - Io


def find_vb(Vg1, Vg2, Vd, p, Vb0, vmax=0.85):
    """Branch-following root finder: prefer root near Vb0, snap when needed."""
    lo = max(-0.2, Vb0 - 0.03)
    hi = min(vmax, max(Vb0 + 0.03, Vb0 + 0.02))
    try:
        if kcl(lo, Vg1, Vg2, Vd, p) * kcl(hi, Vg1, Vg2, Vd, p) <= 0:
            return float(brentq(kcl, lo, hi, args=(Vg1, Vg2, Vd, p),
                                  xtol=1e-5, rtol=1e-5))
    except ValueError:
        pass
    # Global enumeration — denser grid for robustness
    g = np.linspace(-0.2, vmax, 121)
    f = np.array([kcl(v, Vg1, Vg2, Vd, p) for v in g])
    sc = np.where(np.sign(f[:-1]) != np.sign(f[1:]))[0]
    roots = []
    for i in sc:
        try:
            roots.append(brentq(kcl, g[i], g[i+1], args=(Vg1, Vg2, Vd, p),
                                  xtol=1e-5, rtol=1e-5))
        except ValueError:
            continue
    if not roots:
        return float(vmax)
    # Pick root nearest previous Vb (branch continuation)
    return float(min(roots, key=lambda r: abs(r - Vb0)))


def trace(Vg1, Vg2, Vds, p):
    """Full trace — start from body equilibrium at Vd=0 (≈ Vg2)."""
    Vb = float(Vg2)  # body follows back-gate at Vd=0 (no Iii, only leak)
    Vbs = np.zeros_like(Vds)
    for k, vd in enumerate(Vds):
        Vb = find_vb(Vg1, Vg2, float(vd), p, Vb0=Vb)
        Vbs[k] = Vb
    # BSIM4 expects Vbs relative to source; clamp for numerical safety.
    Vbs_c = np.clip(Vbs, -0.5, p.PhiS - 1e-3)
    Id, _ = drain_current_bsim(Vg1, Vds, Vbs_c, p)
    # Parasitic BJT collector current
    from nsram.bsim4 import bipolar_collector_current_ss
    Ic = bipolar_collector_current_ss(Vg1, Vds, Vbs_c, p)
    return np.asarray(Id) + np.asarray(Ic)


def build_params(x):
    vth0, ltw, lnl, lga, lgrb, lgam = x
    return replace(BASE,
                    VTH0=float(vth0), LTW=float(ltw),
                    L_NONLOCAL=float(lnl),
                    BJT_AREA=float(10 ** lga),
                    Rb=float(10 ** lgrb),
                    ALPHA0=float(BASE.ALPHA0 * 10 ** lgam))


def objective(x):
    p = build_params(x)
    rmses = []
    for vg1, vg2, vd, idd in CURVES:
        try:
            pred = trace(vg1, vg2, vd, p)
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
    (0.50, 1.20),     # VTH0
    (15e-9, 100e-9),  # LTW
    (0.0,   300e-9),  # L_NONLOCAL
    (-8.0, 0.0),      # log10 BJT_AREA
    ( 7.0, 12.0),     # log10 Rb   (floating-body regime)
    (-2.0, 2.0),      # log10 ALPHA0_mult
]


def main():
    print(f"VG2-aware DE fit, {len(CURVES)} curves, {N_DS} pts each")
    t0 = time.time(); ctr = [0]

    def cb(xk, conv):
        ctr[0] += 1
        o = objective(xk)
        print(f"  iter {ctr[0]:3d} obj={o:.3f} conv={conv:.3f}  "
              f"VTH0={xk[0]:.3f} LTW={xk[1]*1e9:.0f}n L_nl={xk[2]*1e9:.0f}n "
              f"A=1e{xk[3]:.1f} Rb=1e{xk[4]:.1f} A0×10^{xk[5]:.2f} "
              f"({time.time()-t0:.0f}s)", flush=True)
        return False

    res = differential_evolution(
        objective, BOUNDS,
        maxiter=60, popsize=15,
        mutation=(0.5, 1.3), recombination=0.7,
        seed=42, tol=1e-4, workers=-1, updating="deferred",
        callback=cb, polish=True, disp=False,
    )
    xo = res.x
    p = build_params(xo)
    print(f"\nBest objective = {res.fun:.3f}")

    per_curve = []
    for vg1, vg2, vd, idd in CURVES:
        try:
            pred = trace(vg1, vg2, vd, p)
            m = (idd > ID_FLOOR) & (pred > 0)
            if m.sum() < 5: continue
            r = float(np.sqrt(np.mean((np.log10(idd[m]) - np.log10(pred[m]))**2)))
            per_curve.append({"vg1": vg1, "vg2": vg2, "log_rmse": r})
        except Exception:
            continue
    rs = np.array([c["log_rmse"] for c in per_curve])
    print(f"median={np.median(rs):.2f}  p90={np.percentile(rs,90):.2f}  "
          f"worst={np.max(rs):.2f}  ({len(rs)} curves scored)")

    with open(OUT / "summary.json", "w") as f:
        json.dump({
            "VTH0": float(xo[0]), "LTW_nm": float(xo[1]*1e9),
            "L_NONLOCAL_nm": float(xo[2]*1e9),
            "BJT_AREA": float(10**xo[3]), "Rb": float(10**xo[4]),
            "ALPHA0": float(BASE.ALPHA0*10**xo[5]),
            "median_log_rmse": float(np.median(rs)),
            "p90_log_rmse": float(np.percentile(rs, 90)),
            "worst_log_rmse": float(np.max(rs)),
            "n_curves": len(rs),
            "per_curve": per_curve,
        }, f, indent=2)

    # Overlay grid: 3 VG1 × 5 representative VG2 values
    target_vg2 = [-0.15, -0.05, 0.05, 0.15, 0.25]
    fig, axes = plt.subplots(3, 5, figsize=(17, 9), sharey="row")
    for row, vg1 in enumerate([0.2, 0.4, 0.6]):
        cands = [c for c in CURVES if abs(c[0]-vg1) < 0.01]
        for col, vg2_t in enumerate(target_vg2):
            hit = min(cands, key=lambda c: abs(c[1]-vg2_t))
            _, vg2, vd, idd = hit
            pred = trace(vg1, vg2, vd, p)
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
    fig.suptitle(f"VG2-aware fit — median {np.median(rs):.2f}, "
                  f"worst {np.max(rs):.2f} | Rb=10^{xo[4]:.2f}")
    fig.tight_layout(); fig.savefig(OUT / "overlay.png", dpi=130); plt.close(fig)
    print(f"Wrote {OUT/'overlay.png'}")


if __name__ == "__main__":
    main()
