"""Bf × Is 2D sweep heatmap — visualization of the breakthrough.

Renders the parameter sweep that broke the 1.39-dec floor: the joint
(Bf, Is) coupling reveals an optimum at (2e4, 1e-9) → 0.795 dec, missed
by all 1D sweeps. Single static figure for the brief.

Output: figures/bf_is_sweep/bf_is_heatmap.{png,pdf}
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "figures/bf_is_sweep"; OUT.mkdir(parents=True, exist_ok=True)

# Initial coarse 2D sweep (from 01_LOG.md 2026-05-04 ~12:00 entry).
# Rows = Bf, columns = Is.
Bf_axis = np.array([1e2, 1e3, 1e4, 5e4, 1e5])
Is_axis = np.array([5e-9, 1e-9, 1e-10, 1e-11, 1e-12])

# Median NRMSE in decades, 33-row dataset.
data = np.array([
    [1.394, 1.424, 1.631, 1.856, 1.976],   # Bf=100
    [1.383, 1.395, 1.442, 1.510, 1.538],   # Bf=1e3
    [0.858, 0.862, 1.328, 1.427, 1.405],   # Bf=1e4
    [1.002, 0.948, 1.431, 1.548, 1.645],   # Bf=5e4
    [1.216, 1.138, 1.433, 1.657, 1.645],   # Bf=1e5
])

# Refined optimum from finer grid (Bf=2e4, Is=1e-9 → 0.795 dec).
opt_Bf = 2e4; opt_Is = 1e-9; opt_val = 0.795

# Custom diverging colourmap centred on 1.0 (the original target).
cmap = LinearSegmentedColormap.from_list("nrmse_cmap",
    [(0.0, "#1a9850"), (0.30, "#a6d96a"),    # green = below 1.0
     (0.50, "#ffffbf"),                        # yellow = at 1.0
     (0.70, "#fdae61"), (1.00, "#a50026")])    # red = far above

fig, ax = plt.subplots(figsize=(8, 5.5))

# Heatmap with log-x = Is reversed (smaller Is right), log-y = Bf
extent = [-0.5, len(Is_axis)-0.5, len(Bf_axis)-0.5, -0.5]
im = ax.imshow(data, cmap=cmap, vmin=0.7, vmax=2.0, aspect="auto", extent=extent)

# Annotate each cell
for i in range(len(Bf_axis)):
    for j in range(len(Is_axis)):
        v = data[i, j]
        c = "white" if (v < 1.0 or v > 1.6) else "black"
        weight = "bold" if v < 1.0 else "normal"
        ax.text(j, i, f"{v:.3f}", ha="center", va="center",
                color=c, fontsize=10, weight=weight)

# Mark the refined optimum (between Bf=1e4 and 5e4 ≈ row 2.5, Is=1e-9 col 1)
ax.plot(1, 2.5, marker="*", color="cyan", markersize=24,
         markeredgecolor="black", markeredgewidth=1.5, zorder=5,
         label=f"Refined optimum: Bf={opt_Bf:.0e}, Is={opt_Is:.0e} → {opt_val:.3f} dec")

ax.set_xticks(range(len(Is_axis)))
ax.set_xticklabels([f"{x:.0e}" for x in Is_axis])
ax.set_yticks(range(len(Bf_axis)))
ax.set_yticklabels([f"{x:.0e}" for x in Bf_axis])
ax.set_xlabel("Is (BJT saturation current, A)")
ax.set_ylabel("Bf (BJT forward gain)")
ax.set_title("Bf × Is 2D parameter sweep — median NRMSE on Sebas 33-row dataset\n"
             "1D sweeps each missed this pocket; the 2D coupling broke the 1.39-dec floor",
             fontsize=11)

cbar = plt.colorbar(im, ax=ax, fraction=0.045, pad=0.02)
cbar.set_label("median NRMSE (decades, lower = better)", fontsize=10)
# Mark the original 1.0-dec target on colourbar
cbar.ax.axhline(1.0, color="black", lw=1.2, ls="--")
cbar.ax.text(1.05, 1.0, " <1.0 target", transform=cbar.ax.get_yaxis_transform(),
              va="center", fontsize=8)

ax.legend(loc="upper left", framealpha=0.9, fontsize=9)
ax.grid(which="both", color="white", lw=0.3, alpha=0.3)

plt.tight_layout()
plt.savefig(OUT / "bf_is_heatmap.png", dpi=150)
plt.savefig(OUT / "bf_is_heatmap.pdf")
plt.close()
print(f"[fig] saved {OUT}/bf_is_heatmap.{{png,pdf}}")
print(f"[fig] best in coarse grid: {data.min():.3f} dec at Bf={Bf_axis[np.unravel_index(data.argmin(), data.shape)[0]]:.0e}, Is={Is_axis[np.unravel_index(data.argmin(), data.shape)[1]]:.0e}")
print(f"[fig] refined optimum:     {opt_val:.3f} dec at Bf={opt_Bf:.0e}, Is={opt_Is:.0e}")
print(f"[fig] floor before sweep: 1.39 dec (1D walls). Improvement: {(1.39-opt_val):.3f} dec.")
