"""z14_diag_latch.py — diagnostic: does the model actually LATCH?

For each of the three overlay VG1 curves (0.2/0.4/0.6) at VG2~0, plot the
internal trajectory Vb(Vd), and the contributions Ids, Iii, Ic(Iii) across
the sweep using z13's best parameters.

Goal: see whether the model reaches a saddle-node where Vb snaps up, or
whether it just slides smoothly through intermediate Vb — which would
explain the user's "alla kurvor ser slöa ut utan något hopp" observation.
"""
from __future__ import annotations
import csv, json, re
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from scripts.z12_optimize import (build_params, find_vb, trace, CURVES, BASE)
from nsram.bsim4 import (drain_current_bsim, impact_ionization_bsim4,
                          bipolar_collector_current_ss, gidl_current)

OUT = Path("/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy/"
           "results/z14_diag_latch")
OUT.mkdir(parents=True, exist_ok=True)

# z13's best
z13 = json.loads(Path("/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy/"
                       "results/z13_gpu_solver/summary.json").read_text())
pz = z13["params"]
p = build_params([pz["VTH0"], pz["LTW_nm"]*1e-9, pz["L_NONLOCAL_nm"]*1e-9,
                   np.log10(pz["BJT_AREA"]), np.log10(pz["Rb"]),
                   np.log10(pz["ALPHA0"]/BASE.ALPHA0)])
print(f"[params] {pz}")

# Pick one curve per VG1 around VG2=0
shown = {}
for vg1, vg2, vd, idd in CURVES:
    if vg1 not in shown and abs(vg2) < 0.06:
        shown[vg1] = (vg2, vd, idd)

fig, axes = plt.subplots(2, 3, figsize=(15, 7.5), sharex=True)
for col, vg1 in enumerate(sorted(shown)):
    vg2, vd, idd = shown[vg1]
    # Re-run trace and capture Vb trajectory
    Vb = 0.0
    Vbs = np.zeros_like(vd)
    for k, v in enumerate(vd):
        Vb = find_vb(vg1, float(v), p, Vb0=Vb)
        Vbs[k] = Vb
    Ids_np, _ = drain_current_bsim(vg1, vd, Vbs, p)
    Iii_np = np.array([float(impact_ionization_bsim4(vg1, float(v), float(b), p))
                        for v, b in zip(vd, Vbs)])
    Ic_np = bipolar_collector_current_ss(vg1, vd, Vbs, p)
    Id_pred = np.asarray(Ids_np) + np.asarray(Ic_np)

    ax0 = axes[0, col]
    ax0.semilogy(vd, np.clip(idd, 1e-15, None), "k-", lw=2, label="meas")
    ax0.semilogy(vd, np.clip(Id_pred, 1e-22, None), "g-", lw=1.7, label="fit total")
    ax0.semilogy(vd, np.clip(Ids_np, 1e-22, None), "b--", lw=1, label="Ids (MOS)")
    ax0.semilogy(vd, np.clip(Ic_np, 1e-22, None), "r--", lw=1, label="Ic (BJT)")
    ax0.semilogy(vd, np.clip(Iii_np, 1e-22, None), "m:", lw=1, label="Iii")
    ax0.set_title(f"VG1={vg1}, VG2={vg2:+.2f}")
    ax0.set_ylabel("|I| [A]")
    ax0.set_ylim(1e-14, 1e-3)
    ax0.grid(alpha=0.3, which="both"); ax0.legend(fontsize=7, loc="lower right")

    ax1 = axes[1, col]
    ax1.plot(vd, Vbs, "g-", lw=1.7, label="Vb (body)")
    ax1.axhline(0.85, color="gray", ls="--", lw=0.7, label="vmax=0.85")
    ax1.set_xlabel("Vd [V]"); ax1.set_ylabel("Vb [V]")
    ax1.set_ylim(-0.05, 0.95)
    ax1.grid(alpha=0.3); ax1.legend(fontsize=7, loc="lower right")

    # Log slope of data and fit (finite difference on log)
    logI_meas = np.log10(np.clip(idd, 1e-22, None))
    logI_fit = np.log10(np.clip(Id_pred, 1e-22, None))
    slope_meas = np.gradient(logI_meas, vd)
    slope_fit = np.gradient(logI_fit, vd)
    print(f"VG1={vg1}: max dlogI/dV  meas={slope_meas.max():.2f}  "
          f"fit={slope_fit.max():.2f}  "
          f"Vb range=[{Vbs.min():.3f},{Vbs.max():.3f}]")

fig.suptitle(f"Diagnostic: Vb trajectory + I-components  (z13 best params)")
fig.tight_layout()
fig.savefig(OUT / "latch_diag.png", dpi=140)
plt.close(fig)
print(f"\nWrote {OUT/'latch_diag.png'}")
