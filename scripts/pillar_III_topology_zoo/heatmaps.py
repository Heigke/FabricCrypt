"""Render heatmaps from phase_diagram_3d.json: 5 stimuli x 6 topologies x 4 het.

Output: results/Pillar_III_topology_zoo/heatmaps.png
For each metric (Lyapunov, NARMA, edge-of-chaos flag, Kuramoto R) we make a
grid of (stim x topo) heatmaps with heterogeneity on x-axis of each cell.

To keep things readable we make 4 panels, one per metric.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path(__file__).resolve().parents[2] / "results" / "Pillar_III_topology_zoo"
data = json.loads((OUT / "phase_diagram_3d.json").read_text())
TOPS = data["topologies"]; STIMS = data["stimuli"]; HETS = data["heterogeneities"]
res = data["results"]


def grid(metric_key: str, default=np.nan) -> np.ndarray:
    arr = np.full((len(STIMS), len(TOPS), len(HETS)), default, dtype=float)
    for r in res:
        i = STIMS.index(r["stimulus"]); j = TOPS.index(r["topology"]); k = HETS.index(r["heterogeneity"])
        v = r[metric_key]
        arr[i, j, k] = float(v) if v is not None else default
    return arr


def render_panel(ax, mat: np.ndarray, title: str, cmap: str, vmin=None, vmax=None) -> None:
    # mat: (stims, topos, hets) -> render as (stims*hets) rows x topos cols
    nS, nT, nH = mat.shape
    big = mat.transpose(0, 2, 1).reshape(nS * nH, nT)
    im = ax.imshow(big, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax, origin="upper")
    ax.set_xticks(np.arange(nT)); ax.set_xticklabels(TOPS, rotation=35, ha="right", fontsize=8)
    yt = []
    for s in STIMS:
        for h in HETS:
            yt.append(f"{s[:8]}|h={h:.2f}")
    ax.set_yticks(np.arange(nS * nH)); ax.set_yticklabels(yt, fontsize=6)
    ax.set_title(title)
    plt.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    # Light separators between stimuli.
    for s in range(1, nS):
        ax.axhline(s * nH - 0.5, color="white", lw=1.0)


fig, axes = plt.subplots(2, 2, figsize=(20, 22))
render_panel(axes[0, 0], grid("lyapunov"), "Largest Lyapunov", "RdBu_r", vmin=-0.3, vmax=0.3)
render_panel(axes[0, 1], grid("narma30_r2"), "NARMA-30 r^2", "viridis", vmin=-0.2, vmax=1.0)
render_panel(axes[1, 0], grid("edge_of_chaos").astype(float), "Edge-of-Chaos flag",
             "Greens", vmin=0, vmax=1)
render_panel(axes[1, 1], grid("kuramoto_R"), "Kuramoto R", "magma", vmin=0, vmax=1)

fig.suptitle("Pillar III Topology Zoo: 6 topologies x 5 stimuli x 4 heterogeneity levels",
             fontsize=14)
fig.tight_layout(rect=[0, 0, 1, 0.97])
fig.savefig(OUT / "heatmaps.png", dpi=120)
print(f"saved {OUT / 'heatmaps.png'}")
