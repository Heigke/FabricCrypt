"""z15_kcl_scan.py — scan kcl(Vb) at several Vd to find saddle-node.

If the low-Vb root persists at all measured Vd values, the model CANNOT
latch with these parameters — the parasitic BJT is too weak or Rb too
low. In that case we need wider search bounds, not a better solver.
"""
from __future__ import annotations
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from scripts.z12_optimize import build_params, kcl, CURVES, BASE

OUT = Path("/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy/"
           "results/z15_kcl_scan")
OUT.mkdir(parents=True, exist_ok=True)

z13 = json.loads(Path("/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy/"
                       "results/z13_gpu_solver/summary.json").read_text())
pz = z13["params"]
p = build_params([pz["VTH0"], pz["LTW_nm"]*1e-9, pz["L_NONLOCAL_nm"]*1e-9,
                   np.log10(pz["BJT_AREA"]), np.log10(pz["Rb"]),
                   np.log10(pz["ALPHA0"]/BASE.ALPHA0)])

# Pick VG1=0.4 ~ VG2=0 curve
tgt = None
for vg1, vg2, vd, idd in CURVES:
    if abs(vg1 - 0.4) < 0.01 and abs(vg2) < 0.06:
        tgt = (vg1, vg2, vd, idd); break
assert tgt, "no VG1=0.4 curve found"
vg1, vg2, vd_arr, id_meas = tgt
print(f"Scanning kcl(Vb) for VG1={vg1} VG2={vg2}")

vb_grid = np.linspace(0.0, 0.90, 401)

# Pick 6 Vd values spanning the sweep
vd_probes = np.array([0.3, 0.7, 1.0, 1.25, 1.5, 1.8])
fig, axes = plt.subplots(2, 3, figsize=(15, 7), sharey=False)
for ax, vd in zip(axes.flat, vd_probes):
    f = np.array([kcl(vb, vg1, float(vd), p) for vb in vb_grid])
    # Find sign changes (roots)
    signs = np.sign(f)
    crossings = np.where(signs[:-1] != signs[1:])[0]
    roots = []
    for i in crossings:
        # Linear interp
        r = vb_grid[i] - f[i] * (vb_grid[i+1] - vb_grid[i]) / (f[i+1] - f[i])
        roots.append(r)

    ax.plot(vb_grid, f, "b-", lw=1)
    ax.axhline(0, color="k", lw=0.5)
    for r in roots:
        ax.axvline(r, color="r", ls="--", lw=0.7)
    ax.set_xlabel("Vb [V]")
    ax.set_ylabel("kcl net [A]")
    ax.set_title(f"Vd={vd:.2f}V  roots={[f'{r:.3f}' for r in roots]}")
    ax.set_yscale("symlog", linthresh=1e-11)
    ax.grid(alpha=0.3)

fig.suptitle(f"kcl(Vb) scan — VG1={vg1}, VG2={vg2}  (z13 params)")
fig.tight_layout()
fig.savefig(OUT / "kcl_scan_vg1_04.png", dpi=140)
plt.close(fig)

# Also scan the Ic contribution magnitude: at Vb=0.6V what's Ic vs Iii?
from nsram.bsim4 import (impact_ionization_bsim4, bipolar_collector_current_ss)
Vb_hi = 0.6
print(f"\nAt Vb={Vb_hi}V, hypothetical high-branch:")
for vd in vd_probes:
    Iii = float(impact_ionization_bsim4(vg1, float(vd), Vb_hi, p))
    Ic = float(bipolar_collector_current_ss(vg1, np.array([vd]),
                                              np.array([Vb_hi]), p)[0])
    # BJT hole current into body (emitter side)
    Is_eff = p.BJT_IS * p.BJT_AREA
    Vt = 0.02585
    Ib_hole = (Is_eff / p.BJT_BF) * (np.exp(min(Vb_hi/(p.BJT_NE*Vt), 60)) - 1)
    Ileak = Vb_hi / p.Rb
    print(f"  Vd={vd:.2f}: Iii={Iii:.2e}  Ic(latch)={Ic:.2e}  "
          f"Ib_hole={Ib_hole:.2e}  Ileak={Ileak:.2e}")
print(f"\nWrote {OUT/'kcl_scan_vg1_04.png'}")
