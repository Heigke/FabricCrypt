"""z8c_3d_sweep.py — 3D physical calibration sweep with branch-following solver.

z8b fixed VTH0 at 0.80 + Rb at 1e8 and got 0.68 dec median, but the
overlay shows VG1=0.6 fires 0.7V too late — a local failure the
median hides. The fire-timing depends strongly on how much body
charging current (Iii + IGIDL) the M1 cell delivers BEFORE the
bifurcation. Three unknowns:

  1. VTH0_M1     — sets Ids and thence Iii
  2. ALPHA0_M1   — sets Iii directly (§6.1)
  3. BJT_AREA    — sets the BJT's IS_eff, setting the firing
                    threshold voltage (where body KCL bifurcates)

Sweep all three with the branch-following continuation solver.
Report the median, the SPREAD across curves, and the worst case
so we catch local failures that z8b's median hid.
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
               "results/z8c_3d_sweep")
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
            curves.append((float(m.group(2)), float(m.group(1)),
                            new_vd, new_id))
    return curves


def eval_params(curves, vth, am, area, rb=1e8):
    p = replace(PRESET_BASE, VTH0=vth,
                  ALPHA0=PRESET_BASE.ALPHA0 * am,
                  BJT_AREA=area, Rb=rb)
    rmses = []
    for vg1, vg2, vd, idd in curves:
        try:
            pred = trace_id(vg1, vd, p)
            m = (idd > 1e-13) & (pred > 0)
            if m.sum() < 5: continue
            lm = np.log10(idd[m]); lp = np.log10(pred[m])
            rmses.append(float(np.sqrt(np.mean((lm - lp)**2))))
        except Exception:
            continue
    if not rmses:
        return {"median": np.inf, "p90": np.inf, "worst": np.inf, "n": 0}
    rs = np.array(rmses)
    return {"median": float(np.median(rs)),
            "p90":    float(np.percentile(rs, 90)),
            "worst":  float(rs.max()),
            "n":      int(rs.size)}


def main():
    print("Loading curves...")
    curves = load_curves(25)
    print(f"  {len(curves)} curves\n")

    vth_grid   = np.round(np.arange(0.70, 0.91, 0.025), 3)     # 9
    alpha_grid = np.array([0.3, 1.0, 3.0, 10.0, 30.0, 100.0])  # 6
    area_grid  = np.array([1e-8, 1e-6, 1e-4, 1e-2, 1.0])       # 5
    total = len(vth_grid) * len(alpha_grid) * len(area_grid)
    print(f"3D sweep: {len(vth_grid)}×{len(alpha_grid)}×{len(area_grid)} "
          f"= {total} combos")
    print(f"  VTH0    ∈ [{vth_grid[0]}, {vth_grid[-1]}]")
    print(f"  ALPHA0× ∈ {list(alpha_grid)}")
    print(f"  AREA    ∈ {list(area_grid)}\n")

    # For each (vth, am, area) store (median, p90, worst)
    med = np.full((len(vth_grid), len(alpha_grid), len(area_grid)), np.inf)
    p90 = np.full_like(med, np.inf)
    wst = np.full_like(med, np.inf)

    t0 = time.time()
    done = 0
    for i, vth in enumerate(vth_grid):
        for j, am in enumerate(alpha_grid):
            for k, area in enumerate(area_grid):
                r = eval_params(curves, float(vth), float(am), float(area))
                med[i, j, k] = r["median"]
                p90[i, j, k] = r["p90"]
                wst[i, j, k] = r["worst"]
                done += 1
        print(f"  VTH0={vth:.3f}  done {done}/{total}  "
              f"({time.time()-t0:.0f}s)  best so far: "
              f"{np.nanmin(med[:i+1]):.2f} dec")

    # Best by median
    i_b, j_b, k_b = np.unravel_index(np.nanargmin(med), med.shape)
    vth_s  = float(vth_grid[i_b])
    am_s   = float(alpha_grid[j_b])
    area_s = float(area_grid[k_b])
    print(f"\n═══ BEST by MEDIAN ═══")
    print(f"  VTH0      = {vth_s:.3f} V")
    print(f"  ALPHA0×   = {am_s:.1f}  → ALPHA0 = {PRESET_BASE.ALPHA0*am_s:.2e}")
    print(f"  BJT_AREA  = {area_s:.0e}")
    print(f"  median / p90 / worst = {med[i_b,j_b,k_b]:.2f} / "
          f"{p90[i_b,j_b,k_b]:.2f} / {wst[i_b,j_b,k_b]:.2f} dec")

    # Best by WORST (more honest — catches local failures)
    i_w, j_w, k_w = np.unravel_index(np.nanargmin(wst), wst.shape)
    vth_w  = float(vth_grid[i_w])
    am_w   = float(alpha_grid[j_w])
    area_w = float(area_grid[k_w])
    print(f"\n═══ BEST by WORST-CASE ═══")
    print(f"  VTH0      = {vth_w:.3f} V")
    print(f"  ALPHA0×   = {am_w:.1f}")
    print(f"  BJT_AREA  = {area_w:.0e}")
    print(f"  median / p90 / worst = {med[i_w,j_w,k_w]:.2f} / "
          f"{p90[i_w,j_w,k_w]:.2f} / {wst[i_w,j_w,k_w]:.2f} dec")

    with open(OUT_DIR / "summary.json", "w") as f:
        json.dump({
            "best_by_median": {"VTH0": vth_s, "ALPHA0_mult": am_s,
                                 "BJT_AREA": area_s,
                                 "median": float(med[i_b,j_b,k_b]),
                                 "p90":    float(p90[i_b,j_b,k_b]),
                                 "worst":  float(wst[i_b,j_b,k_b])},
            "best_by_worst":  {"VTH0": vth_w, "ALPHA0_mult": am_w,
                                 "BJT_AREA": area_w,
                                 "median": float(med[i_w,j_w,k_w]),
                                 "p90":    float(p90[i_w,j_w,k_w]),
                                 "worst":  float(wst[i_w,j_w,k_w])},
            "vth_grid": list(map(float, vth_grid)),
            "alpha_mult_grid": list(map(float, alpha_grid)),
            "area_grid": list(map(float, area_grid)),
        }, f, indent=2)
    np.savez(OUT_DIR / "grid.npz",
              vth=vth_grid, alpha_mult=alpha_grid, area=area_grid,
              median=med, p90=p90, worst=wst)

    # Overlay at BEST-by-WORST
    p_best = replace(PRESET_BASE, VTH0=vth_w,
                      ALPHA0=PRESET_BASE.ALPHA0 * am_w,
                      BJT_AREA=area_w, Rb=1e8)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.3), sharey=True)
    shown = {}
    for vg1, vg2, vd, idd in curves:
        if vg1 not in shown and abs(vg2) < 0.06:
            shown[vg1] = (vg2, vd, idd, trace_id(vg1, vd, p_best))
    for ax, vg1 in zip(axes, sorted(shown)):
        vg2, vd, idd, pred = shown[vg1]
        m = (idd > 1e-13) & (pred > 0)
        r = float(np.sqrt(np.mean((np.log10(idd[m]) - np.log10(pred[m]))**2))) if m.any() else float("nan")
        ax.semilogy(vd, np.clip(idd, 1e-15, None), "k-", lw=2,
                     label="meas")
        ax.semilogy(vd, np.clip(pred, 1e-22, None), "g-", lw=1.7,
                     label=f"fit ({r:.2f} dec)")
        ax.set_title(f"VG1={vg1}"); ax.set_xlabel("Vd [V]")
        ax.grid(alpha=0.3, which="both"); ax.legend(fontsize=8)
    axes[0].set_ylabel("|Id| [A]")
    fig.suptitle(f"Best-by-worst: VTH0={vth_w:.3f}, ALPHA0×{am_w:.1f}, "
                 f"AREA={area_w:.0e}  (worst={wst[i_w,j_w,k_w]:.2f} dec)")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "overlay_best_by_worst.png", dpi=140)
    plt.close(fig)

    # 2D slice at best AREA
    fig, ax = plt.subplots(figsize=(7, 4.5))
    slice_med = med[:, :, k_w]
    im = ax.imshow(slice_med, origin="lower", aspect="auto",
                    extent=[np.log10(alpha_grid[0])-0.25,
                            np.log10(alpha_grid[-1])+0.25,
                            vth_grid[0]-0.0125, vth_grid[-1]+0.0125],
                    cmap="viridis_r",
                    vmin=max(0.3, np.nanmin(slice_med)-0.1),
                    vmax=min(3.0, np.nanmax(slice_med)))
    ax.plot(np.log10(am_w), vth_w, "rx", ms=15, mew=2.5)
    ax.set_xlabel("log10(ALPHA0 multiplier)"); ax.set_ylabel("VTH0 [V]")
    ax.set_title(f"Median log-RMSE at AREA={area_w:.0e}")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "heatmap_vth_alpha.png", dpi=140)
    plt.close(fig)

    print(f"\nWrote overlay_best_by_worst.png, heatmap_vth_alpha.png")


if __name__ == "__main__":
    main()
