"""Generate a combined multi-panel "network in action" figure for the brief:
- (left col, top) Mackey-Glass true vs predicted time series
- (left col, bottom) reservoir activity heatmap (64 cells × T)
- (right col) ER_SPARSE topology graph drawn on a circle layout
- caption-ready PDF + PNG
"""
from __future__ import annotations
import importlib.util
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import torch
torch.set_default_dtype(torch.float64)

ROOT = Path(__file__).resolve().parent.parent
sp = importlib.util.spec_from_file_location("mg", ROOT / "scripts/demo_mackey_glass.py")
mg_mod = importlib.util.module_from_spec(sp); sp.loader.exec_module(mg_mod)
sp2 = importlib.util.spec_from_file_location("z119", ROOT / "scripts/z119_topology_sweep.py")
z119 = importlib.util.module_from_spec(sp2); sp2.loader.exec_module(z119)

OUT = ROOT / "figures" / "network_in_action"
OUT.mkdir(parents=True, exist_ok=True)

# Generate MG signal + run reservoir
N, T, kappa = 64, 600, 0.30
mg = mg_mod.gen_mackey_glass(T)
log_Id = mg_mod.run_reservoir(N, T, kappa, mg, seed=42, Bf=100.0)
fit = mg_mod.fit_forecast(log_Id, mg, horizon=6, warmup=80, train_frac=0.6)

# Build the same ER_SPARSE topology
rng = np.random.default_rng(42)
W = z119.build_W("ER_SPARSE", N, rho=0.9, rng=rng)

# ------------------- figure -------------------
fig = plt.figure(figsize=(12, 5.5))
gs = GridSpec(2, 3, figure=fig, width_ratios=[1.6, 1.6, 1.0],
              hspace=0.45, wspace=0.35,
              left=0.06, right=0.98, top=0.92, bottom=0.10)

# Top-left: MG true vs predicted (test segment)
ax_top = fig.add_subplot(gs[0, :2])
warmup, n_tr = fit["warmup"], fit["n_tr"]
truth = fit["truth"]; pred = fit["pred"]
t_test = np.arange(len(truth))
ax_top.plot(t_test, truth, color="#222", lw=1.4, label="Mackey–Glass (true)")
ax_top.plot(t_test, pred, color="#c0392b", lw=1.2, ls="--",
             label=f"Reservoir forecast (h=6, NRMSE={fit['nrmse']:.3f})")
ax_top.set_xlim(0, len(truth) - 1)
ax_top.set_ylim(-0.05, 1.05)
ax_top.set_ylabel("MG amplitude (a.u.)")
ax_top.set_title("(a) Mackey–Glass forecast at +6 steps", loc="left",
                  fontsize=11, weight="bold")
ax_top.grid(alpha=0.25)
ax_top.legend(loc="upper right", framealpha=0.9, fontsize=9)

# Bottom-left: reservoir activity heatmap, test segment
ax_mid = fig.add_subplot(gs[1, :2])
test_start = warmup + n_tr
hm = log_Id[:, test_start: test_start + len(truth)]
# normalise per-row for visual clarity
hm = (hm - hm.mean(axis=1, keepdims=True)) / (hm.std(axis=1, keepdims=True) + 1e-9)
im = ax_mid.imshow(hm, aspect="auto", cmap="RdBu_r", vmin=-2.5, vmax=2.5,
                    interpolation="nearest")
ax_mid.set_xlabel("time step")
ax_mid.set_ylabel("cell index (64 NS-RAM cells)")
ax_mid.set_title("(b) Reservoir activity log|Id| (z-scored per cell)", loc="left",
                  fontsize=11, weight="bold")
cbar = plt.colorbar(im, ax=ax_mid, fraction=0.04, pad=0.02)
cbar.set_label("z-score", fontsize=9)

# Right column: ER_SPARSE network graph on circle layout
ax_net = fig.add_subplot(gs[:, 2])
theta = np.linspace(0, 2 * np.pi, N, endpoint=False)
pos = np.column_stack([np.cos(theta), np.sin(theta)])
# Draw edges (subsample if too many)
i_idx, j_idx = np.where(np.abs(W) > 1e-6)
mask = i_idx != j_idx
i_idx, j_idx = i_idx[mask], j_idx[mask]
n_edges_drawn = 0
for ii, jj in zip(i_idx, j_idx):
    w = W[ii, jj]
    color = "#3498db" if w > 0 else "#e67e22"
    alpha = min(0.8, 0.15 + 1.6 * abs(w))
    ax_net.plot([pos[ii, 0], pos[jj, 0]],
                 [pos[ii, 1], pos[jj, 1]],
                 color=color, alpha=alpha, lw=0.5, zorder=1)
    n_edges_drawn += 1
ax_net.scatter(pos[:, 0], pos[:, 1], s=70, c="#222", zorder=3,
                edgecolors="white", linewidth=0.8)
# Highlight a few example cells with their bias colour
ax_net.set_xlim(-1.25, 1.25)
ax_net.set_ylim(-1.25, 1.25)
ax_net.set_aspect("equal")
ax_net.axis("off")
ax_net.set_title(f"(c) ER_SPARSE topology\nN={N}, density~10%, "
                  f"{n_edges_drawn} edges",
                  loc="center", fontsize=11, weight="bold")

# Title bar
fig.suptitle(
    f"NS-RAM reservoir at honest physical params (Bf=100, η-bounded): "
    f"forecast NRMSE={fit['nrmse']:.3f} on test split",
    fontsize=12, weight="bold", y=0.985,
)

OUT_PNG = OUT / "network_in_action.png"
OUT_PDF = OUT / "network_in_action.pdf"
plt.savefig(OUT_PNG, dpi=150, bbox_inches="tight")
plt.savefig(OUT_PDF, bbox_inches="tight")
plt.close()
print(f"[fig] saved {OUT_PNG} and {OUT_PDF}")
print(f"[fig] NRMSE={fit['nrmse']:.4f}, edges={n_edges_drawn}")
