"""z18_verify_latch.py — verify z17 params actually latch."""
from __future__ import annotations
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from scripts.z12_optimize import build_params, find_vb, trace, CURVES, BASE
from nsram.bsim4 import (drain_current_bsim, bipolar_collector_current_ss,
                          impact_ionization_bsim4)

OUT = Path("/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy/"
           "results/z18_verify_latch")
OUT.mkdir(parents=True, exist_ok=True)

z17 = json.loads(Path("/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy/"
                       "results/z17_shape_fit/summary.json").read_text())
p = build_params([z17["VTH0"], z17["LTW_nm"]*1e-9, z17["L_NONLOCAL_nm"]*1e-9,
                   np.log10(z17["BJT_AREA"]), np.log10(z17["Rb"]),
                   np.log10(z17["ALPHA0"]/BASE.ALPHA0)])
print(f"[z17] Rb=10^{np.log10(z17['Rb']):.2f}  AREA=10^{np.log10(z17['BJT_AREA']):.2f}")

shown = {}
for vg1, vg2, vd, idd in CURVES:
    if vg1 not in shown and abs(vg2) < 0.06:
        shown[vg1] = (vg2, vd, idd)

fig, axes = plt.subplots(2, 3, figsize=(15, 7.5), sharex=True)
for col, vg1 in enumerate(sorted(shown)):
    vg2, vd, idd = shown[vg1]
    Vb = 0.0; Vbs = np.zeros_like(vd)
    for k, v in enumerate(vd):
        Vb = find_vb(vg1, float(v), p, Vb0=Vb); Vbs[k] = Vb
    Ids_np, _ = drain_current_bsim(vg1, vd, Vbs, p)
    Iii = np.array([float(impact_ionization_bsim4(vg1, float(v), float(b), p))
                     for v, b in zip(vd, Vbs)])
    Ic = bipolar_collector_current_ss(vg1, vd, Vbs, p)
    Id_pred = np.asarray(Ids_np) + np.asarray(Ic)
    a0 = axes[0, col]
    a0.semilogy(vd, np.clip(idd, 1e-15, None), "k-", lw=2, label="meas")
    a0.semilogy(vd, np.clip(Id_pred, 1e-22, None), "g-", lw=1.7, label="total fit")
    a0.semilogy(vd, np.clip(Ids_np, 1e-22, None), "b--", lw=1, label="Ids (MOS)")
    a0.semilogy(vd, np.clip(Ic, 1e-22, None), "r--", lw=1, label="Ic (BJT)")
    a0.semilogy(vd, np.clip(Iii, 1e-22, None), "m:", lw=1, label="Iii")
    a0.set_title(f"VG1={vg1}"); a0.set_ylabel("|I| [A]"); a0.set_ylim(1e-14, 1e-3)
    a0.grid(alpha=0.3, which="both"); a0.legend(fontsize=7, loc="lower right")
    a1 = axes[1, col]
    a1.plot(vd, Vbs, "g-", lw=1.7); a1.axhline(0.85, color="gray", ls="--", lw=0.7)
    a1.set_xlabel("Vd [V]"); a1.set_ylabel("Vb [V]"); a1.set_ylim(-0.05, 0.95)
    a1.grid(alpha=0.3)
    print(f"VG1={vg1}: Vb range [{Vbs.min():.3f}, {Vbs.max():.3f}]  "
           f"latch Vd ≈ {vd[Vbs.argmax()]:.2f}V")
fig.suptitle(f"z17 params — Vb trajectory + components  "
              f"(Rb=10^{np.log10(z17['Rb']):.2f})")
fig.tight_layout(); fig.savefig(OUT / "latch_verify.png", dpi=140); plt.close(fig)
print(f"Wrote {OUT/'latch_verify.png'}")
