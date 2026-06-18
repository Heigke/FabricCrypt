"""z7b_bjt_alpha_sweep.py — the REAL missing calibration knobs.

z7a (M2 sweep) showed M2 is not the dominant lever: its best
configuration (Vth=-0.2, M2 ~always on) just recovers M1+NPN-only
behaviour at 2.1 dec — the 2T solver can't fix a 2-decade overshoot
that is already baked into M1+NPN.

The overshoot is in the *M1 + parasitic-NPN* path itself. Two knobs
that move the amplitude by orders of magnitude:

  1. BJT_AREA   — SPICE area multiplier on IS/IKF. We set it to 1e-6
                  ("area=1u" in schematic), but the convention matters:
                  if Sebas meant 1 unit (unitless) the effective IS is
                  10^6× higher than we assume; if micrometer² it's what
                  we have. Sweep across 4 decades to see which works.
  2. ALPHA0_eff — M1's impact-ionization prefactor. PTM130 card says
                  7.84e-5; maybe Sebas's real device has different
                  value, or layout stress shifts it.

We hold Vth_M2 = -0.2 (effectively M2 always conducting, which was
z7a's best) and sweep the 2D (BJT_AREA, ALPHA0_mult) grid.
"""
from __future__ import annotations

import csv, json, re, time
from dataclasses import replace
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from nsram.bsim4 import BSIM4_PRESETS, two_transistor_cell_ss


DATA_DIR = Path("/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy/"
                "data/sebas_2026_04_22")
OUT_DIR = Path("/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy/"
               "results/z7b_bjt_alpha_sweep")
OUT_DIR.mkdir(parents=True, exist_ok=True)

PRESET_BASE = BSIM4_PRESETS["ns_ram_130nm_pazos"]
VTH_M2_FIXED = -0.2   # from z7a best-fit

VG_RE = re.compile(r"VG2=(-?\d+\.?\d*)_VG=(\d+\.\d+)")


def load_curves(downsample_n: int = 20):
    curves = []
    for sub in sorted(DATA_DIR.iterdir()):
        if not sub.is_dir():
            continue
        for fn in sorted(sub.glob("*.csv")):
            m = VG_RE.search(fn.name)
            if not m:
                continue
            vg1 = float(m.group(2))
            vg2 = float(m.group(1))
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
            new_vd = np.linspace(0.3, float(Vd.max()), downsample_n)
            new_id = np.interp(new_vd, Vd, Id)
            curves.append((vg1, vg2, new_vd, new_id))
    return curves


def eval_combo(curves, bjt_area: float, alpha0_mult: float,
                n_iter: int = 15) -> float:
    p_m1 = replace(
        PRESET_BASE,
        ALPHA0=PRESET_BASE.ALPHA0 * alpha0_mult,
        BJT_AREA=bjt_area,
    )
    p_m2 = replace(
        PRESET_BASE,
        Leff=PRESET_BASE.Leff * 10.0,
        VTH0=VTH_M2_FIXED,
        BJT_IS=0.0, BJT_BF=0.0,
    )
    rmses = []
    for vg1, vg2, vd, idd in curves:
        try:
            I_pred, _, _ = two_transistor_cell_ss(
                vg1, vg2, vd, p_m1, p_m2=p_m2,
                n_iter=n_iter, damping=0.4,
            )
            m = (idd > 1e-12) & (np.asarray(I_pred) > 0)
            if m.sum() < 4:
                continue
            lm = np.log10(idd[m]); lp = np.log10(np.asarray(I_pred)[m])
            rmses.append(float(np.sqrt(np.mean((lm - lp) ** 2))))
        except Exception:
            continue
    return (float(np.median(rmses)) if rmses else float("inf"),
            float(np.percentile(rmses, 90)) if rmses else float("inf"),
            len(rmses))


def main():
    print("Loading curves...")
    curves = load_curves(20)
    print(f"  {len(curves)} curves\n")

    bjt_area_grid   = np.logspace(-8, 0, 9)           # 1e-8 .. 1.0
    alpha0_mult_grid = np.logspace(-3, 2, 11)          # 0.001 .. 100
    print(f"Sweep: {len(bjt_area_grid)} × {len(alpha0_mult_grid)} = "
          f"{len(bjt_area_grid) * len(alpha0_mult_grid)} combos")
    print(f"  BJT_AREA     : {bjt_area_grid[0]:.0e} .. {bjt_area_grid[-1]:.0e}")
    print(f"  ALPHA0_mult  : {alpha0_mult_grid[0]:.3f} .. {alpha0_mult_grid[-1]:.0f}\n")

    med = np.full((len(bjt_area_grid), len(alpha0_mult_grid)), np.inf)
    p90 = np.full_like(med, np.inf)
    t0 = time.time()
    for i, area in enumerate(bjt_area_grid):
        row_txt = []
        for j, am in enumerate(alpha0_mult_grid):
            m, p, n = eval_combo(curves, area, am)
            med[i, j] = m; p90[i, j] = p
            row_txt.append(f"{m:4.2f}")
        print(f"  BJT_AREA={area:.0e}: " + " ".join(row_txt) +
              f"  ({time.time()-t0:.0f}s)")

    # Best
    i_best, j_best = np.unravel_index(np.nanargmin(med), med.shape)
    area_star = float(bjt_area_grid[i_best])
    am_star = float(alpha0_mult_grid[j_best])
    best = med[i_best, j_best]
    print(f"\n═══ BEST ═══")
    print(f"  BJT_AREA        = {area_star:.0e}")
    print(f"  ALPHA0_mult     = {am_star:.3f}  → ALPHA0 = {PRESET_BASE.ALPHA0*am_star:.2e}")
    print(f"  median log-RMSE = {best:.2f} dec")
    print(f"  p90    log-RMSE = {p90[i_best, j_best]:.2f} dec")

    summary = {
        "Vth_M2_fixed": VTH_M2_FIXED,
        "best_BJT_AREA": area_star,
        "best_ALPHA0_mult": am_star,
        "best_ALPHA0_effective": float(PRESET_BASE.ALPHA0 * am_star),
        "best_median_log_rmse": float(best),
        "best_p90_log_rmse": float(p90[i_best, j_best]),
        "bjt_area_grid": list(map(float, bjt_area_grid)),
        "alpha0_mult_grid": list(map(float, alpha0_mult_grid)),
    }
    with open(OUT_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    np.savez(OUT_DIR / "grid.npz",
              bjt_area=bjt_area_grid, alpha0_mult=alpha0_mult_grid,
              median=med, p90=p90)

    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(med, origin="lower", aspect="auto",
                    extent=[np.log10(alpha0_mult_grid[0])-0.25,
                            np.log10(alpha0_mult_grid[-1])+0.25,
                            np.log10(bjt_area_grid[0])-0.5,
                            np.log10(bjt_area_grid[-1])+0.5],
                    cmap="viridis_r", vmin=max(0.5, np.min(med)-0.2),
                    vmax=5.0)
    ax.set_xlabel("log10(ALPHA0 multiplier)")
    ax.set_ylabel("log10(BJT_AREA)")
    ax.plot(np.log10(am_star), np.log10(area_star), "rx", ms=14, mew=2.5,
             label=f"best: {best:.2f} dec")
    ax.set_title("Median log-RMSE — BSIM4 + NPN (Vth_M2 fixed = -0.2V)")
    fig.colorbar(im, ax=ax, label="log-RMSE [decades]")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "heatmap.png", dpi=140)
    plt.close(fig)

    # Overlay with best
    p_m1 = replace(PRESET_BASE, ALPHA0=PRESET_BASE.ALPHA0 * am_star,
                     BJT_AREA=area_star)
    p_m2 = replace(PRESET_BASE, Leff=PRESET_BASE.Leff * 10.0,
                     VTH0=VTH_M2_FIXED, BJT_IS=0.0, BJT_BF=0.0)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.3), sharey=True)
    by_vg1 = {}
    for vg1, vg2, vd, idd in curves:
        if vg1 not in by_vg1 and abs(vg2) < 0.06:
            try:
                I_pred, _, _ = two_transistor_cell_ss(
                    vg1, vg2, vd, p_m1, p_m2=p_m2, n_iter=25)
                by_vg1[vg1] = (vg2, vd, idd, np.asarray(I_pred))
            except Exception:
                pass
    for ax, vg1 in zip(axes, sorted(by_vg1)):
        vg2, vd, idd, pred = by_vg1[vg1]
        m = (idd > 1e-12) & (pred > 0)
        rmse = float(np.sqrt(np.mean((np.log10(idd[m]) - np.log10(pred[m]))**2))) if m.any() else float("nan")
        ax.semilogy(vd, np.clip(idd, 1e-14, None), "k-", lw=2, label=f"meas (VG2={vg2:+.2f})")
        ax.semilogy(vd, np.clip(pred, 1e-20, None), "g-", lw=1.6,
                     label=f"fit ({rmse:.2f} dec)")
        ax.set_title(f"VG1={vg1} V"); ax.set_xlabel("Vd [V]")
        ax.grid(alpha=0.3, which="both"); ax.legend(fontsize=7)
    axes[0].set_ylabel("|Id| [A]")
    fig.suptitle(f"BJT_AREA={area_star:.0e}, ALPHA0×{am_star:.3f} (Vth_M2=-0.2)")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "overlay_best.png", dpi=140)
    plt.close(fig)
    print(f"\nWrote heatmap.png, overlay_best.png, summary.json")


if __name__ == "__main__":
    main()
