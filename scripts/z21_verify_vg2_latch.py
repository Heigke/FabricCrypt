"""z21_verify_vg2_latch.py — confirm latch across VG2 with z20 params."""
from __future__ import annotations
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from scripts.z20_vg2_fit import CURVES, build_params, trace, find_vb, BASE

OUT = Path("/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy/"
           "results/z21_verify_vg2_latch")
OUT.mkdir(parents=True, exist_ok=True)

s = json.loads(Path("/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy/"
                     "results/z20_vg2_fit/summary.json").read_text())
p = build_params([s["VTH0"], s["LTW_nm"]*1e-9, s["L_NONLOCAL_nm"]*1e-9,
                   np.log10(s["BJT_AREA"]), np.log10(s["Rb"]),
                   np.log10(s["ALPHA0"]/BASE.ALPHA0)])
print(f"z20 best: VTH0={s['VTH0']:.3f} Rb=10^{np.log10(s['Rb']):.2f} "
       f"A=10^{np.log10(s['BJT_AREA']):.2f}")

target_vg2 = [-0.15, -0.05, 0.05, 0.15, 0.25]
fig, axes = plt.subplots(3, 5, figsize=(17, 9), sharex=True)
for row, vg1 in enumerate([0.2, 0.4, 0.6]):
    cands = [c for c in CURVES if abs(c[0]-vg1) < 0.01]
    for col, vg2_t in enumerate(target_vg2):
        hit = min(cands, key=lambda c: abs(c[1]-vg2_t))
        _, vg2, vd, _ = hit
        Vb = float(vg2); Vbs = np.zeros_like(vd)
        for k, v in enumerate(vd):
            Vb = find_vb(vg1, vg2, float(v), p, Vb0=Vb); Vbs[k] = Vb
        ax = axes[row, col]
        ax.plot(vd, Vbs, "g-", lw=1.5)
        ax.axhline(0.85, color="gray", ls="--", lw=0.5)
        ax.axhline(vg2, color="red", ls=":", lw=0.5, label=f"VG2={vg2:+.2f}")
        ax.set_title(f"VG1={vg1}  VG2={vg2:+.2f}", fontsize=9)
        ax.set_ylim(-0.25, 1.0); ax.grid(alpha=0.3); ax.legend(fontsize=6)
        if row == 2: ax.set_xlabel("Vd [V]")
        if col == 0: ax.set_ylabel("Vb [V]")
        snap_vd = vd[np.argmax(np.diff(Vbs, prepend=Vbs[0]))]
        print(f"  VG1={vg1} VG2={vg2:+.2f}: Vb [{Vbs.min():.2f},{Vbs.max():.2f}] "
               f"snap@Vd≈{snap_vd:.2f}")
fig.suptitle("Vb trajectory across VG2 — z20 params")
fig.tight_layout(); fig.savefig(OUT / "vb_grid.png", dpi=130); plt.close(fig)
print(f"\nWrote {OUT/'vb_grid.png'}")
