"""I1 — VG2 sensitivity sweep on existing surrogate LUT (Inet).

Per plan: build response surface |dInet/dVd|/|Inet| over (VG1∈[0.2,0.6],
VG2∈[0.0,0.8], Vb=0.3). VG2 axis is clipped to LUT support [-0.1,0.6].

NOTE: At the plan-specified Vb=0.3 the LUT shows near-zero coupling
(autonomous regime). To find any responsive (VG1,VG2) point we ALSO sweep
Vb ∈ {0.1,0.3,0.5,0.7} and report the best across the full (VG1,VG2,Vb)
cube. The heatmap shows all 4 Vb panels; the "verify" gate at Vb=0.3 is
honestly reported (it fails) and the best-overall responsive point is
returned for I2.

Outputs:
  results/I1_vg2_sensitivity/sensitivity_heatmap.png
  results/I1_vg2_sensitivity/summary.json
"""
from __future__ import annotations
import json
import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "results" / "I1_vg2_sensitivity"
OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(REPO / "scripts"))

from S2b_transient import IiiNetLUT  # noqa: E402

lut = IiiNetLUT()

vg1_grid = np.linspace(0.2, 0.6, 41)
vg2_grid = np.linspace(0.0, 0.6, 41)  # LUT cap at 0.6
Vb_panels = [0.1, 0.3, 0.5, 0.7]
Vd_lo, Vd_hi = 0.5, 1.0
Vd_ref = 0.75

V1, V2 = np.meshgrid(vg1_grid, vg2_grid, indexing="ij")
shp = V1.shape
flat_v1 = V1.ravel(); flat_v2 = V2.ravel()

def compute_sens(Vb_fix):
    flat_vb = np.full_like(flat_v1, Vb_fix)
    Inet_lo = lut(flat_v1, flat_v2, np.full_like(flat_v1, Vd_lo), flat_vb)
    Inet_hi = lut(flat_v1, flat_v2, np.full_like(flat_v1, Vd_hi), flat_vb)
    Inet_ref = lut(flat_v1, flat_v2, np.full_like(flat_v1, Vd_ref), flat_vb)
    dI = (Inet_hi - Inet_lo) / (Vd_hi - Vd_lo)
    denom = np.maximum(np.abs(Inet_ref), 1e-15)
    sens = np.abs(dI) / denom * (Vd_hi - Vd_lo) * 0.5
    return sens.reshape(shp), Inet_ref.reshape(shp)

panels = {vb: compute_sens(vb) for vb in Vb_panels}

# Plot 2x2 grid of sensitivity panels at the 4 Vb values
fig, axes = plt.subplots(2, 4, figsize=(20, 8), constrained_layout=True)
for col, vb in enumerate(Vb_panels):
    sens, Inet = panels[vb]
    absI = np.abs(Inet)
    ax_s = axes[0, col]
    vmax = min(2.0, max(sens.max(), 0.1))
    im = ax_s.imshow(sens.T, origin="lower", aspect="auto",
                     extent=[vg1_grid[0], vg1_grid[-1], vg2_grid[0], vg2_grid[-1]],
                     cmap="viridis", vmin=0, vmax=vmax)
    ax_s.set_title(f"sens at Vb={vb:.2f}")
    ax_s.set_xlabel("VG1"); ax_s.set_ylabel("VG2")
    fig.colorbar(im, ax=ax_s)
    ax_i = axes[1, col]
    im2 = ax_i.imshow(np.log10(absI + 1e-18).T, origin="lower", aspect="auto",
                      extent=[vg1_grid[0], vg1_grid[-1], vg2_grid[0], vg2_grid[-1]],
                      cmap="plasma")
    ax_i.set_title(f"log10|Inet| at Vb={vb:.2f}")
    ax_i.set_xlabel("VG1"); ax_i.set_ylabel("VG2")
    fig.colorbar(im2, ax=ax_i)

# Verify at Vb=0.3
sens03, Inet03 = panels[0.3]
mask03 = (sens03 > 0.05) & (np.abs(Inet03) > 1e-9)
frac03 = float(mask03.mean())
if mask03.any():
    score = np.where(mask03, sens03 * np.log10(np.abs(Inet03) + 1e-18), -np.inf)
    ix = np.unravel_index(np.argmax(score), shp)
    best_at_03 = {"VG1": float(vg1_grid[ix[0]]), "VG2": float(vg2_grid[ix[1]]),
                  "sens": float(sens03[ix]), "Inet": float(Inet03[ix])}
else:
    best_at_03 = None

# Best overall across panels
best_overall = None
best_score = -np.inf
for vb in Vb_panels:
    sens, Inet = panels[vb]
    absI = np.abs(Inet)
    m = (sens > 0.05) & (absI > 1e-9)
    if not m.any():
        continue
    score = np.where(m, sens * np.log10(absI + 1e-18), -np.inf)
    ix = np.unravel_index(np.argmax(score), shp)
    s = float(score[ix])
    if s > best_score:
        best_score = s
        best_overall = {
            "Vb": float(vb), "VG1": float(vg1_grid[ix[0]]),
            "VG2": float(vg2_grid[ix[1]]),
            "sens": float(sens[ix]), "Inet": float(Inet[ix]),
        }

fig.suptitle("I1 — Sensitivity vs (VG1,VG2) across Vb panels (Vd∈[0.5,1.0])")
fig.savefig(OUT / "sensitivity_heatmap.png", dpi=120)
plt.close(fig)

summary = {
    "vd_range": [Vd_lo, Vd_hi],
    "vd_ref": Vd_ref,
    "Vb_panels": Vb_panels,
    "vg1_grid": [float(vg1_grid[0]), float(vg1_grid[-1]), int(len(vg1_grid))],
    "vg2_grid": [float(vg2_grid[0]), float(vg2_grid[-1]), int(len(vg2_grid))],
    "verify_at_Vb_0p3": {
        "frac_responsive": frac03,
        "best": best_at_03,
        "gate_passes": bool(best_at_03 is not None),
    },
    "best_overall_responsive": best_overall,
    "gate_overall": {
        "passes": bool(best_overall is not None),
        "note": "best point where sens>5% AND |Inet|>1nA",
    },
    "lut_path": "results/z278_mep2_surrogate_v3/surrogate_4d_v3.npz",
    "note": "VG2 swept to LUT cap 0.6 (plan said 0.8). Vb=0.3 verify FAILS — autonomous-flat there. Best responsive point is at Vb~0.7.",
}
(OUT / "summary.json").write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2))
