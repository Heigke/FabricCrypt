"""z46_canonical_check.py — sanity check: does our canonical model produce
the right dVd_up/dVG2 trend that Sebas's data shows?

If yes: proceed to use canonical as ground truth for cell_fast calibration.
If no: we have a fundamental physics gap that needs to be filled first.
"""
from __future__ import annotations
import time
from pathlib import Path
from dataclasses import replace

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from nsram.nsram_canonical import trace_nsram_series, make_pazos_130nm

OUT = Path("/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy/"
           "results/z46_canonical_check")
OUT.mkdir(parents=True, exist_ok=True)


def detect_snapback(Vd, Id, factor=5.0, vd_min=0.20):
    if len(Vd) < 5: return None
    log_Id = np.log10(np.clip(Id, 1e-13, None))
    dlog = np.diff(log_Id)
    valid = Vd[1:] >= vd_min
    if not valid.any(): return None
    masked = np.where(valid, dlog, -np.inf)
    if masked.max() < np.log10(factor): return None
    return float(Vd[int(np.argmax(masked)) + 1])


def main():
    # Use z37-style params (lower VTH0, higher BJT to encourage latch)
    p = replace(make_pazos_130nm(),
                 VTH0=0.30, U0=0.08,
                 BJT_AREA=1e-4, ALPHA0_mult=0.5,
                 Rb=1e10, AGIDL=1e-7, BGIDL=2.4e9 * 1.5)

    Vd_grid = np.linspace(0.05, 2.5, 25)
    print("Canonical model snapback Vd_up vs VG2 at three VG1 levels:")
    print()
    print(f"{'VG1':>5}  " + "  ".join(f"{vg2:+.2f}" for vg2 in [-0.20, -0.10, 0.0, 0.10, 0.20, 0.30]))

    fig, ax = plt.subplots(figsize=(9, 6))
    cmap = {0.2: "blue", 0.4: "green", 0.6: "red"}

    for vg1 in [0.2, 0.4, 0.6]:
        vd_ups = []
        line = f"{vg1:>5}  "
        for vg2 in [-0.20, -0.10, 0.0, 0.10, 0.20, 0.30]:
            try:
                Id, Vbs = trace_nsram_series(vg1, vg2, Vd_grid, p)
                vu = detect_snapback(Vd_grid, Id, factor=3.0)
                vd_ups.append(vu)
                line += f"{vu:.2f} " if vu is not None else "  --   "
            except Exception as e:
                vd_ups.append(None)
                line += "  err  "
        print(line)
        # plot
        vg2s = np.array([-0.20, -0.10, 0.0, 0.10, 0.20, 0.30])
        ys = np.array([v if v is not None else np.nan for v in vd_ups])
        ax.plot(vg2s, ys, "o-", color=cmap[vg1], lw=2, label=f"VG1={vg1}")

    # Add data trend lines from z45 measurements (manual)
    data_anchors = {
        0.2: [(-0.20, 1.20), (-0.15, 1.20), (-0.10, 1.25), (-0.05, 1.35),
               (0.00, 1.55), (0.05, 1.75), (0.10, 2.00)],
        0.4: [(-0.20, 0.90), (-0.15, 0.95), (-0.10, 0.95), (-0.05, 1.00),
               (0.00, 1.05), (0.05, 1.15), (0.10, 1.25), (0.15, 1.35),
               (0.20, 1.55), (0.25, 1.75), (0.30, 1.95)],
        0.6: [(0.00, 0.90), (0.05, 0.95), (0.10, 1.05), (0.15, 1.10),
               (0.20, 1.15), (0.25, 1.25), (0.30, 1.35)],
    }
    for vg1, pts in data_anchors.items():
        xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
        ax.plot(xs, ys, "x--", color=cmap[vg1], lw=1.5, alpha=0.7,
                  label=f"DATA VG1={vg1}")

    ax.set_xlabel("VG2 [V]"); ax.set_ylabel("Vd_up (snapback) [V]")
    ax.set_title("Canonical model vs Sebas's data — does VG2 trend match?")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "canonical_vs_data.png", dpi=140)
    plt.close(fig)
    print(f"\nWrote {OUT/'canonical_vs_data.png'}")


if __name__ == "__main__":
    main()
