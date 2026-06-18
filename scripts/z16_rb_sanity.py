"""z16_rb_sanity.py — does bumping Rb alone produce the latch?"""
from __future__ import annotations
import json
from pathlib import Path
from dataclasses import replace
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from scripts.z12_optimize import (build_params, find_vb, trace, CURVES, BASE)
from nsram.bsim4 import (drain_current_bsim, bipolar_collector_current_ss)

OUT = Path("/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy/"
           "results/z16_rb_sanity")
OUT.mkdir(parents=True, exist_ok=True)

z13 = json.loads(Path("/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy/"
                       "results/z13_gpu_solver/summary.json").read_text())
pz = z13["params"]
p0 = build_params([pz["VTH0"], pz["LTW_nm"]*1e-9, pz["L_NONLOCAL_nm"]*1e-9,
                    np.log10(pz["BJT_AREA"]), np.log10(pz["Rb"]),
                    np.log10(pz["ALPHA0"]/BASE.ALPHA0)])

shown = {}
for vg1, vg2, vd, idd in CURVES:
    if vg1 not in shown and abs(vg2) < 0.06:
        shown[vg1] = (vg2, vd, idd)

rb_values = [7.54e5, 1e7, 1e9, 1e11]  # original then sweep up
colors = ["red", "orange", "green", "blue"]
fig, axes = plt.subplots(2, 3, figsize=(15, 7.5), sharex=True)
for col, vg1 in enumerate(sorted(shown)):
    vg2, vd, idd = shown[vg1]
    ax0 = axes[0, col]
    ax0.semilogy(vd, np.clip(idd, 1e-15, None), "k-", lw=2, label="meas")
    ax1 = axes[1, col]
    for rb, c in zip(rb_values, colors):
        p = replace(p0, Rb=rb)
        Vb = 0.0; Vbs = np.zeros_like(vd)
        for k, v in enumerate(vd):
            Vb = find_vb(vg1, float(v), p, Vb0=Vb); Vbs[k] = Vb
        Ids_np, _ = drain_current_bsim(vg1, vd, Vbs, p)
        Ic_np = bipolar_collector_current_ss(vg1, vd, Vbs, p)
        Id_pred = np.asarray(Ids_np) + np.asarray(Ic_np)
        ax0.semilogy(vd, np.clip(Id_pred, 1e-22, None), c, lw=1.3,
                      label=f"Rb=10^{np.log10(rb):.1f}")
        ax1.plot(vd, Vbs, c, lw=1.3, label=f"Rb=10^{np.log10(rb):.1f}")
    ax0.set_title(f"VG1={vg1}, VG2={vg2:+.2f}")
    ax0.set_ylabel("|I| [A]"); ax0.set_ylim(1e-14, 1e-3)
    ax0.grid(alpha=0.3, which="both"); ax0.legend(fontsize=7, loc="lower right")
    ax1.set_xlabel("Vd [V]"); ax1.set_ylabel("Vb [V]"); ax1.set_ylim(-0.05, 0.95)
    ax1.grid(alpha=0.3); ax1.legend(fontsize=7, loc="best")
fig.suptitle("Rb sweep (all other params = z13 best)")
fig.tight_layout(); fig.savefig(OUT / "rb_sweep.png", dpi=140); plt.close(fig)
print(f"Wrote {OUT/'rb_sweep.png'}")
