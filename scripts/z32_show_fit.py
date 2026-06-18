"""z32_show_fit.py — clean, high-res visualization of canonical NSRAM fit
on all 33 curves. Three-panel layout, large fonts, per-curve RMSE.
"""
from __future__ import annotations
import csv, json, re
from pathlib import Path
from dataclasses import replace

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from nsram.nsram_canonical import trace_nsram, make_pazos_130nm

DATA = Path("/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy/"
             "data/sebas_2026_04_22")
OUT = Path("/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy/"
           "results/z32_show_fit")
OUT.mkdir(parents=True, exist_ok=True)
VG_RE = re.compile(r"VG2=(-?\d+\.?\d*)_VG=(\d+\.\d+)")
ID_FLOOR = 1e-13


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
            curves.append((vg1, vg2, Vd, Id))
    return curves


def main():
    # Load z31 best params
    summary = json.loads(Path("/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy/"
                                "results/z31_canonical_de_fit/summary.json").read_text())
    p = replace(make_pazos_130nm(),
                 VTH0=summary["VTH0"],
                 BJT_AREA=summary["BJT_AREA"],
                 ALPHA0_mult=summary["ALPHA0_mult"],
                 Rb=summary["Rb"],
                 U0=summary["U0"])

    curves = load_curves()
    print(f"Loaded {len(curves)} curves")

    # Plot ALL curves in 3 large panels (one per VG1)
    fig, axes = plt.subplots(1, 3, figsize=(20, 7), sharey=True)
    rmses = []
    for ax, vg1 in zip(axes, [0.2, 0.4, 0.6]):
        cands = sorted([c for c in curves if abs(c[0]-vg1) < 0.01],
                        key=lambda c: c[1])
        cmap = plt.cm.coolwarm(np.linspace(0, 1, len(cands)))
        for (_, vg2, vd_meas, id_meas), color in zip(cands, cmap):
            # Resample model on common grid
            vd_grid = np.linspace(0.1, vd_meas.max(), 25)
            try:
                pred, _ = trace_nsram(vg1, vg2, vd_grid, p)
            except Exception:
                continue
            id_at_grid = np.power(10.0, np.interp(
                vd_grid, vd_meas, np.log10(id_meas)))
            m = (id_at_grid > ID_FLOOR) & (pred > 0)
            if m.sum() > 5:
                r = float(np.sqrt(np.mean(
                    (np.log10(id_at_grid[m]) - np.log10(pred[m]))**2)))
                rmses.append(r)
            # Plot data (solid) + fit (dashed) in same color per VG2
            ax.semilogy(vd_meas, np.clip(id_meas, 1e-14, None),
                          color=color, lw=1.6, alpha=0.85,
                          label=f"VG2={vg2:+.2f} ({r:.2f})")
            ax.semilogy(vd_grid, np.clip(pred, 1e-22, None),
                          color=color, lw=1.0, ls="--", alpha=0.85)
        ax.set_title(f"VG1={vg1} V — {len(cands)} VG2 values\n"
                      f"solid = measured, dashed = canonical NSRAM fit",
                      fontsize=11)
        ax.set_xlabel("Vd [V]", fontsize=11)
        ax.set_ylim(1e-13, 1e-3)
        ax.grid(alpha=0.3, which="both")
        ax.legend(fontsize=7, ncol=2, loc="lower right")
    axes[0].set_ylabel("|Id| [A]", fontsize=11)
    rs = np.array(rmses)
    fig.suptitle(f"Canonical NSRAM fit on Sebas 2026-04-22 data — "
                  f"all {len(curves)} curves\n"
                  f"median log-RMSE = {np.median(rs):.2f} dec  ·  "
                  f"p90 = {np.percentile(rs, 90):.2f} dec  ·  "
                  f"worst = {np.max(rs):.2f} dec", fontsize=13)
    fig.tight_layout(); fig.savefig(OUT / "all_curves.png", dpi=150)
    plt.close(fig)
    print(f"Wrote {OUT/'all_curves.png'}")

    # Histogram of residuals
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.hist(rs, bins=20, color="#27ae60", alpha=0.7, edgecolor="black")
    ax.axvline(np.median(rs), color="red", lw=1.5, label=f"median={np.median(rs):.2f}")
    ax.axvline(np.percentile(rs, 90), color="orange", lw=1.5,
                ls="--", label=f"p90={np.percentile(rs, 90):.2f}")
    ax.set_xlabel("log-RMSE per curve [decades]", fontsize=11)
    ax.set_ylabel("# curves", fontsize=11)
    ax.set_title(f"Per-curve fit residual distribution ({len(rs)} curves)")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(OUT / "residual_hist.png", dpi=150)
    plt.close(fig)
    print(f"Wrote {OUT/'residual_hist.png'}")
    print(f"\nSummary: median {np.median(rs):.2f}  p90 {np.percentile(rs, 90):.2f}  "
           f"worst {np.max(rs):.2f}  ({len(rs)} curves)")


if __name__ == "__main__":
    main()
