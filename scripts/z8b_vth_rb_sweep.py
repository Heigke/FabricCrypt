"""z8b_vth_rb_sweep.py — fine-tune Vth_M1 and Rb with the branch-following solver.

z8 revealed two remaining calibration gaps:
  1. Pre-fire plateau ~4 decades too high  →  real Vth_M1 > PTM130's 0.54
  2. Fire timing too late                  →  real Rb (body leakage) different

Sweep both against the actual I-V data.
"""
from __future__ import annotations

import csv, json, re, time
from dataclasses import replace
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import brentq

from nsram.bsim4 import (BSIM4_PRESETS, bipolar_collector_current_ss,
                         drain_current_bsim, gidl_current,
                         impact_ionization_bsim4)
from nsram.physics import thermal_voltage


DATA_DIR = Path("/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy/"
                "data/sebas_2026_04_22")
OUT_DIR = Path("/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy/"
               "results/z8b_vth_rb_sweep")
OUT_DIR.mkdir(parents=True, exist_ok=True)

PRESET_BASE = BSIM4_PRESETS["ns_ram_130nm_pazos"]
Vt_th = thermal_voltage(300.0)
VG_RE = re.compile(r"VG2=(-?\d+\.?\d*)_VG=(\d+\.\d+)")


def kcl_net(Vb, Vg1, Vd, p):
    Iii = float(impact_ionization_bsim4(Vg1, Vd, Vb, p))
    Igidl = float(gidl_current(Vd, Vg1, Vb, p)) if p.AGIDL > 0 else 0.0
    IS_eff = p.BJT_IS * max(p.BJT_AREA, 1e-30)
    Vb_c = min(Vb, p.BJT_VJE * 1.1)
    exp_arg = np.clip(Vb_c / (p.BJT_NE * Vt_th), -60, 60)
    Ib_out = (IS_eff / p.BJT_BF) * (np.exp(exp_arg) - 1.0) + Vb / p.Rb
    return (Iii + Igidl) - Ib_out


def find_vb(Vg1, Vd, p, Vb_init, vb_max=0.85):
    lo = max(0.0, Vb_init - 0.02)
    hi = min(vb_max, max(Vb_init + 0.02, 0.02))
    try:
        if kcl_net(lo, Vg1, Vd, p) * kcl_net(hi, Vg1, Vd, p) <= 0:
            return float(brentq(kcl_net, lo, hi, args=(Vg1, Vd, p),
                                  xtol=1e-6, rtol=1e-6))
    except ValueError:
        pass
    grid = np.linspace(0.0, vb_max, 81)
    fs = np.array([kcl_net(v, Vg1, Vd, p) for v in grid])
    sc = np.where(np.sign(fs[:-1]) != np.sign(fs[1:]))[0]
    if len(sc) == 0:
        return float(vb_max)
    roots = []
    for i in sc:
        try:
            roots.append(brentq(kcl_net, grid[i], grid[i+1],
                                  args=(Vg1, Vd, p), xtol=1e-6, rtol=1e-6))
        except ValueError:
            continue
    if not roots:
        return float(vb_max)
    return float(min(roots) if Vb_init < 0.1
                 else min(roots, key=lambda r: abs(r - Vb_init)))


def trace_id(Vg1, Vd_sweep, p):
    Vb = 0.0
    Vbs = np.zeros_like(Vd_sweep)
    for k, vd in enumerate(Vd_sweep):
        Vb = find_vb(Vg1, float(vd), p, Vb_init=Vb)
        Vbs[k] = Vb
    Ids, _ = drain_current_bsim(Vg1, Vd_sweep, Vbs, p)
    Ic = bipolar_collector_current_ss(Vg1, Vd_sweep, Vbs, p)
    return np.asarray(Ids) + np.asarray(Ic)


def load_curves(n_ds=25):
    curves = []
    for sub in sorted(DATA_DIR.iterdir()):
        if not sub.is_dir(): continue
        for fn in sorted(sub.glob("*.csv")):
            m = VG_RE.search(fn.name)
            if not m: continue
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
            Vd = Vd[:peak+1]; Id = Id[:peak+1]
            new_vd = np.linspace(0.1, float(Vd.max()), n_ds)
            new_id = np.interp(new_vd, Vd, Id)
            curves.append((float(m.group(2)), float(m.group(1)), new_vd, new_id))
    return curves


def eval_rmse(curves, p):
    rs = []
    for vg1, vg2, vd, idd in curves:
        try:
            pred = trace_id(vg1, vd, p)
            m = (idd > 1e-13) & (pred > 0)
            if m.sum() < 5:
                continue
            lm = np.log10(idd[m]); lp = np.log10(pred[m])
            rs.append(float(np.sqrt(np.mean((lm - lp) ** 2))))
        except Exception:
            continue
    return np.array(rs) if rs else np.array([np.inf])


def main():
    print("Loading curves (25 Vd pts each)...")
    curves = load_curves(25)
    print(f"  {len(curves)} curves\n")

    vth_grid = np.round(np.arange(0.50, 1.01, 0.05), 3)   # 11 values
    rb_grid = np.logspace(6, 10, 5)                        # 1e6 .. 1e10 Ω
    print(f"Sweep: {len(vth_grid)} × {len(rb_grid)} = "
          f"{len(vth_grid)*len(rb_grid)} combos")

    med = np.full((len(vth_grid), len(rb_grid)), np.inf)
    t0 = time.time()
    for i, vth in enumerate(vth_grid):
        row = []
        for j, rb in enumerate(rb_grid):
            p = replace(PRESET_BASE, VTH0=float(vth), Rb=float(rb))
            rs = eval_rmse(curves, p)
            med[i, j] = float(np.median(rs))
            row.append(f"{med[i,j]:4.2f}")
        print(f"  VTH0={vth:.2f}: " + " ".join(row) +
              f"  ({time.time()-t0:.0f}s)")

    i_best, j_best = np.unravel_index(np.nanargmin(med), med.shape)
    vth_star = float(vth_grid[i_best])
    rb_star  = float(rb_grid[j_best])
    best_med = float(med[i_best, j_best])
    print(f"\n═══ BEST ═══")
    print(f"  VTH0_M1 = {vth_star:.3f} V   (PTM130 card says 0.540)")
    print(f"  Rb      = {rb_star:.0e} Ω   (schematic implicit)")
    print(f"  median log-RMSE = {best_med:.2f} dec")

    # Heatmap
    fig, ax = plt.subplots(figsize=(7, 4.5))
    im = ax.imshow(med, origin="lower", aspect="auto",
                    extent=[np.log10(rb_grid[0])-0.25,
                            np.log10(rb_grid[-1])+0.25,
                            vth_grid[0]-0.025, vth_grid[-1]+0.025],
                    cmap="viridis_r",
                    vmin=max(0.5, np.nanmin(med)-0.1),
                    vmax=min(3.5, np.nanmax(med)))
    ax.plot(np.log10(rb_star), vth_star, "rx", ms=15, mew=2.5)
    ax.set_xlabel("log10(Rb) [Ω]"); ax.set_ylabel("VTH0_M1 [V]")
    ax.set_title(f"Median log-RMSE — best {best_med:.2f} at "
                 f"VTH0={vth_star:.2f}, Rb={rb_star:.0e}")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "heatmap.png", dpi=140)
    plt.close(fig)

    # Overlay at best
    p_best = replace(PRESET_BASE, VTH0=vth_star, Rb=rb_star)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.3), sharey=True)
    shown = {}
    for vg1, vg2, vd, idd in curves:
        if vg1 not in shown and abs(vg2) < 0.06:
            pred = trace_id(vg1, vd, p_best)
            shown[vg1] = (vg2, vd, idd, pred)
    for ax, vg1 in zip(axes, sorted(shown)):
        vg2, vd, idd, pred = shown[vg1]
        m = (idd > 1e-13) & (pred > 0)
        r = float(np.sqrt(np.mean((np.log10(idd[m]) - np.log10(pred[m]))**2))) if m.any() else float("nan")
        ax.semilogy(vd, np.clip(idd, 1e-15, None), "k-", lw=2, label=f"meas")
        ax.semilogy(vd, np.clip(pred, 1e-22, None), "g-", lw=1.6,
                     label=f"fit ({r:.2f} dec)")
        ax.set_title(f"VG1={vg1}"); ax.set_xlabel("Vd [V]")
        ax.grid(alpha=0.3, which="both"); ax.legend(fontsize=8)
    axes[0].set_ylabel("|Id| [A]")
    fig.suptitle(f"Best: VTH0={vth_star:.2f}, Rb={rb_star:.0e}")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "overlay_best.png", dpi=140)
    plt.close(fig)

    with open(OUT_DIR / "summary.json", "w") as f:
        json.dump({
            "best_VTH0_M1": vth_star,
            "best_Rb": rb_star,
            "best_median_log_rmse": best_med,
            "vth_grid": list(map(float, vth_grid)),
            "rb_grid": list(map(float, rb_grid)),
        }, f, indent=2)
    np.savez(OUT_DIR / "grid.npz", vth=vth_grid, rb=rb_grid, median=med)
    print(f"\nWrote heatmap.png, overlay_best.png, summary.json")


if __name__ == "__main__":
    main()
