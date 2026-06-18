"""z7_sebas_m2_sweep.py — Infer Sebastian's undisclosed M2 parameters from data.

The 2T NS-RAM cell has M1 (firing) + Q1 (parasitic NPN) + M2 (access).
We have Sebas's PTM130 card for M1 (Vth0=0.54, normal NMOS), but his
real M2 is almost certainly a low-Vt or native flavor — that info is
only in the NDA-protected foundry model we don't have. Without it,
the first-principles 2T solver overshoots/undershoots by 2-4 decades.

Rather than wait, INFER it from the 33 I-V measurements:

    For each candidate (Vth_M2, mu0_M2_mult):
      build p_m2 = PTM130 with that Vth / mu
      run two_transistor_cell_ss on all 33 curves
      report median log-RMSE

    Pick the (Vth_M2, mu0_M2_mult) that minimises the residual.

That's Sebastian's M2 Vth, read off the data. Options A and B from
the last chat become data-driven.

Usage: python scripts/z7_sebas_m2_sweep.py
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
               "results/z7_sebas_m2_sweep")
OUT_DIR.mkdir(parents=True, exist_ok=True)

PRESET_BASE = BSIM4_PRESETS["ns_ram_130nm_pazos"]


VG_RE = re.compile(r"VG2=(-?\d+\.?\d*)_VG=(\d+\.\d+)")


def load_curves(downsample_n: int = 20):
    """Return list of (vg1, vg2, vd, id) with Vd downsampled for speed."""
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
            # Downsample to `downsample_n` uniformly in Vd ∈ [0.3, 2.0]
            # (only the fit region that matters).
            new_vd = np.linspace(0.3, float(Vd.max()), downsample_n)
            new_id = np.interp(new_vd, Vd, Id)
            curves.append((vg1, vg2, new_vd, new_id))
    return curves


def eval_params(curves, vth_m2: float, mu0_mult: float,
                 n_iter: int = 15) -> dict:
    """Run 2T solver with modified M2 params, return aggregate residuals."""
    p_m2 = replace(
        PRESET_BASE,
        Leff=PRESET_BASE.Leff * 10.0,
        VTH0=vth_m2,
        mu0=PRESET_BASE.mu0 * mu0_mult,
        BJT_IS=0.0, BJT_BF=0.0,       # no parasitic NPN in M2
    )
    rmses, fails = [], 0
    for vg1, vg2, vd, idd in curves:
        try:
            I_pred, _, _ = two_transistor_cell_ss(
                vg1, vg2, vd, PRESET_BASE,
                p_m2=p_m2, n_iter=n_iter, damping=0.4,
            )
            mask = (idd > 1e-12) & (np.asarray(I_pred) > 0)
            if mask.sum() < 4:
                fails += 1; continue
            lm = np.log10(idd[mask])
            lp = np.log10(np.asarray(I_pred)[mask])
            rmses.append(float(np.sqrt(np.mean((lm - lp) ** 2))))
        except Exception:
            fails += 1
    if not rmses:
        return {"median": np.inf, "p90": np.inf, "mean": np.inf,
                "n_ok": 0, "n_fail": fails}
    rmses = np.array(rmses)
    return {
        "median": float(np.median(rmses)),
        "p90":    float(np.percentile(rmses, 90)),
        "mean":   float(np.mean(rmses)),
        "n_ok":   int(rmses.size),
        "n_fail": int(fails),
    }


def main():
    print("Loading curves...")
    curves = load_curves(downsample_n=20)
    print(f"  {len(curves)} curves, 20 Vd points each\n")

    # ── Sweep grid ──
    vth_grid = np.round(np.arange(-0.2, 0.66, 0.05), 3)         # 18 values
    mu_grid  = np.array([0.25, 0.5, 1.0, 2.0, 4.0])             # 5 values
    print(f"Sweep: {len(vth_grid)} × {len(mu_grid)} = "
          f"{len(vth_grid) * len(mu_grid)} param combos")
    print(f"  Vth_M2  ∈ [{vth_grid[0]:+.2f}, {vth_grid[-1]:+.2f}]")
    print(f"  mu0_M2 ∈ {list(mu_grid)} × base ({PRESET_BASE.mu0})\n")

    med_grid = np.full((len(vth_grid), len(mu_grid)), np.nan)
    p90_grid = np.full_like(med_grid, np.nan)
    t0 = time.time()
    for i, vth in enumerate(vth_grid):
        row = []
        for j, mu_mult in enumerate(mu_grid):
            r = eval_params(curves, float(vth), float(mu_mult))
            med_grid[i, j] = r["median"]
            p90_grid[i, j] = r["p90"]
            row.append(f"{r['median']:>5.2f}")
        print(f"  Vth={vth:+.2f} V : " + "  ".join(row)
              + f"   ({time.time()-t0:.0f}s)")

    # Best point
    i_best, j_best = np.unravel_index(np.nanargmin(med_grid), med_grid.shape)
    vth_star = float(vth_grid[i_best])
    mu_star  = float(mu_grid[j_best])
    r_star   = eval_params(curves, vth_star, mu_star, n_iter=25)
    print(f"\n═══ BEST M2 PARAMETERS (inferred from data) ═══")
    print(f"  Vth_M2     = {vth_star:+.3f} V   "
          f"(vs PTM130 normal: {PRESET_BASE.VTH0:+.3f} V)")
    print(f"  mu0_M2     = {PRESET_BASE.mu0*mu_star:.4f}  "
          f"(base × {mu_star})")
    print(f"  median log-RMSE : {r_star['median']:.2f} decades")
    print(f"  p90    log-RMSE : {r_star['p90']:.2f} decades")
    print(f"  mean   log-RMSE : {r_star['mean']:.2f} decades")

    summary = {
        "sweep_vth_range": [float(vth_grid[0]), float(vth_grid[-1])],
        "sweep_mu_mult":   list(map(float, mu_grid)),
        "best_Vth_M2":     vth_star,
        "best_mu0_mult":   mu_star,
        "best_mu0_M2":     float(PRESET_BASE.mu0 * mu_star),
        "best_median_log_rmse":  r_star["median"],
        "best_p90_log_rmse":     r_star["p90"],
        "best_mean_log_rmse":    r_star["mean"],
        "M1_PTM130_VTH0_for_ref": PRESET_BASE.VTH0,
    }
    with open(OUT_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    np.savez(OUT_DIR / "sweep_grid.npz",
              vth_grid=vth_grid, mu_grid=mu_grid,
              median=med_grid, p90=p90_grid)

    # ── Heatmap ──
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.3))
    for ax, grid, title in [
        (axes[0], med_grid, "Median log-RMSE [decades]"),
        (axes[1], p90_grid, "p90 log-RMSE [decades]"),
    ]:
        im = ax.imshow(grid.T, origin="lower", aspect="auto",
                        extent=[vth_grid[0]-0.025, vth_grid[-1]+0.025,
                                -0.5, len(mu_grid)-0.5],
                        cmap="viridis_r")
        ax.set_yticks(range(len(mu_grid)))
        ax.set_yticklabels([f"×{m:.2f}" for m in mu_grid])
        ax.set_xlabel("Vth_M2 [V]")
        ax.set_ylabel("mu0_M2 multiplier")
        ax.set_title(title)
        fig.colorbar(im, ax=ax, shrink=0.85)
        # Mark best
        ax.plot(vth_star, list(mu_grid).index(mu_star), "rx", ms=12, mew=2)
    fig.suptitle(f"Inferred M2 params — best: Vth={vth_star:+.2f}V, "
                 f"mu0×{mu_star:.2f} → {r_star['median']:.2f} dec")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "sweep_heatmap.png", dpi=140)
    plt.close(fig)
    print(f"\nWrote {OUT_DIR / 'sweep_heatmap.png'}")
    print(f"Wrote {OUT_DIR / 'summary.json'}")

    # ── Overlay with inferred M2 ──
    p_m2_star = replace(
        PRESET_BASE, Leff=PRESET_BASE.Leff * 10.0,
        VTH0=vth_star, mu0=PRESET_BASE.mu0 * mu_star,
        BJT_IS=0.0, BJT_BF=0.0,
    )
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.3), sharey=True)
    by_vg1 = {}
    for vg1, vg2, vd, idd in curves:
        if vg1 not in by_vg1 and abs(vg2) < 0.06:
            I_pred, Sint, Vbs = two_transistor_cell_ss(
                vg1, vg2, vd, PRESET_BASE, p_m2=p_m2_star, n_iter=25,
            )
            by_vg1[vg1] = (vg2, vd, idd, np.asarray(I_pred), Sint, Vbs)
    for ax, vg1 in zip(axes, sorted(by_vg1)):
        vg2, vd, idd, pred, Sint, Vbs = by_vg1[vg1]
        m = (idd > 1e-12) & (pred > 0)
        rmse = float(np.sqrt(np.mean((np.log10(idd[m]) - np.log10(pred[m]))**2)))
        ax.semilogy(vd, np.clip(idd, 1e-14, None), "k-", lw=2,
                     label=f"meas (VG2={vg2:+.2f})")
        ax.semilogy(vd, np.clip(pred, 1e-20, None), "g-", lw=1.6,
                     label=f"2T w/ inferred M2 ({rmse:.2f} dec)")
        ax.set_title(f"VG1={vg1} V")
        ax.set_xlabel("Vd [V]")
        ax.grid(alpha=0.3, which="both"); ax.legend(fontsize=7)
    axes[0].set_ylabel("|Id| [A]")
    fig.suptitle(f"Inferred M2: Vth={vth_star:+.2f} V, "
                 f"mu0_M2={PRESET_BASE.mu0*mu_star:.4f}")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "overlay_best.png", dpi=140)
    plt.close(fig)
    print(f"Wrote {OUT_DIR / 'overlay_best.png'}")


if __name__ == "__main__":
    main()
