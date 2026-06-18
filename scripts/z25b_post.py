"""z25b_post.py — produce overlay + summary with DE-best params from z25
(polish hung, so we finalize manually)."""
from __future__ import annotations
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from scripts.z25_qs_bidirectional import (
    CURVES, build_params_locked, trace_bidir, curve_rmse, BJT_AREA_SCH, BASE
)

OUT = Path("/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy/"
           "results/z25_qs_bidirectional")
OUT.mkdir(parents=True, exist_ok=True)

# DE-best from iter 50 log
xo = [1.070, 36e-9, 23e-9, 7.1, 0.01]
p = build_params_locked(xo)
print(f"z25 DE-best: VTH0=1.070 LTW=36n L_nl=23n Rb=1e7.1 A0×10^0.01")
print(f"  BJT_AREA locked = {BJT_AREA_SCH}")

per_curve = []
for vg1, vg2, vu, iu, vd, id_ in CURVES:
    try:
        pu, pd, _, _ = trace_bidir(vg1, vg2, vu, vd, p)
        r_up = curve_rmse(pu, iu); r_dn = curve_rmse(pd, id_)
        per_curve.append({"vg1": vg1, "vg2": vg2,
                            "rmse_up": r_up, "rmse_dn": r_dn})
    except Exception: continue
ups = np.array([c["rmse_up"] for c in per_curve if c["rmse_up"] is not None])
dns = np.array([c["rmse_dn"] for c in per_curve if c["rmse_dn"] is not None])
print(f"Up: median={np.median(ups):.2f}  p90={np.percentile(ups,90):.2f}  worst={np.max(ups):.2f}")
print(f"Dn: median={np.median(dns):.2f}  p90={np.percentile(dns,90):.2f}  worst={np.max(dns):.2f}")

with open(OUT / "summary.json", "w") as f:
    json.dump({
        "VTH0": xo[0], "LTW_nm": xo[1]*1e9, "L_NONLOCAL_nm": xo[2]*1e9,
        "Rb": 10**xo[3], "ALPHA0": BASE.ALPHA0*10**xo[4],
        "BJT_AREA_LOCKED": BJT_AREA_SCH,
        "median_up": float(np.median(ups)),
        "median_dn": float(np.median(dns)),
        "worst_up": float(np.max(ups)),
        "worst_dn": float(np.max(dns)),
        "per_curve": per_curve,
    }, f, indent=2)

# Overlay 3 × 5 with both branches
target_vg2 = [-0.15, -0.05, 0.05, 0.15, 0.25]
fig, axes = plt.subplots(3, 5, figsize=(17, 9), sharey="row")
for row, vg1 in enumerate([0.2, 0.4, 0.6]):
    cands = [c for c in CURVES if abs(c[0]-vg1) < 0.01]
    for col, vg2_t in enumerate(target_vg2):
        hit = min(cands, key=lambda c: abs(c[1]-vg2_t))
        _, vg2, vu, iu, vd, id_ = hit
        pu, pd, Vbu, Vbd = trace_bidir(vg1, vg2, vu, vd, p)
        ax = axes[row, col]
        ax.semilogy(vu, np.clip(iu, 1e-14, None), "k-", lw=1.6, label="meas up")
        ax.semilogy(vd, np.clip(id_, 1e-14, None), "k:", lw=1.0, label="meas dn")
        ax.semilogy(vu, np.clip(pu, 1e-22, None), "g-", lw=1.2, label="fit up")
        ax.semilogy(vd, np.clip(pd, 1e-22, None), "g:", lw=1.0, label="fit dn")
        ax.set_title(f"VG1={vg1}  VG2={vg2:+.2f}", fontsize=8)
        if row == 2: ax.set_xlabel("Vd [V]")
        if col == 0: ax.set_ylabel("|Id| [A]")
        ax.grid(alpha=0.3, which="both"); ax.legend(fontsize=6)
fig.suptitle(f"QS bidirectional + schematic-locked BJT — "
              f"up med {np.median(ups):.2f}  dn med {np.median(dns):.2f}  "
              f"(BJT_AREA=1u locked, Cb=1fF, Rb=1e{xo[3]:.1f})")
fig.tight_layout(); fig.savefig(OUT / "overlay.png", dpi=130); plt.close(fig)
print(f"Wrote {OUT/'overlay.png'}")
