"""Animated 3-panel mp4 of the NS-RAM Mackey-Glass reservoir:
- (a) MG signal + forecast revealed step-by-step
- (b) Reservoir activity heatmap revealed column-by-column
- (c) ER_SPARSE network with nodes pulsing by current activity
Written to figures/network_in_action/network_in_action.mp4
"""
from __future__ import annotations
import importlib.util
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
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

N, T, kappa = 64, 600, 0.30
mg = mg_mod.gen_mackey_glass(T)
log_Id = mg_mod.run_reservoir(N, T, kappa, mg, seed=42, Bf=100.0)
fit = mg_mod.fit_forecast(log_Id, mg, horizon=6, warmup=80, train_frac=0.6)
warmup, n_tr = fit["warmup"], fit["n_tr"]
truth = fit["truth"]; pred = fit["pred"]
n_test = len(truth)
test_start = warmup + n_tr
hm = log_Id[:, test_start: test_start + n_test]
hm = (hm - hm.mean(axis=1, keepdims=True)) / (hm.std(axis=1, keepdims=True) + 1e-9)
hm = np.clip(hm, -2.5, 2.5)

# Activity for the network panel: per-cell normalized log|Id| at each time
node_act = (log_Id[:, test_start: test_start + n_test]
             - log_Id.mean(axis=1, keepdims=True)) / (
             log_Id.std(axis=1, keepdims=True) + 1e-9)
node_act = np.clip(node_act, -2.5, 2.5)

# Build ER_SPARSE same as the still
rng = np.random.default_rng(42)
W = z119.build_W("ER_SPARSE", N, rho=0.9, rng=rng)
theta = np.linspace(0, 2 * np.pi, N, endpoint=False)
pos = np.column_stack([np.cos(theta), np.sin(theta)])

# ---------------- figure ----------------
fig = plt.figure(figsize=(12, 5.5))
gs = GridSpec(2, 3, figure=fig, width_ratios=[1.6, 1.6, 1.0],
              hspace=0.45, wspace=0.35,
              left=0.06, right=0.98, top=0.92, bottom=0.10)

ax_top = fig.add_subplot(gs[0, :2])
line_truth, = ax_top.plot([], [], color="#222", lw=1.4, label="MG (true)")
line_pred, = ax_top.plot([], [], color="#c0392b", lw=1.2, ls="--",
                          label=f"Forecast (h=6, NRMSE={fit['nrmse']:.3f})")
ax_top.set_xlim(0, n_test - 1); ax_top.set_ylim(-0.05, 1.05)
ax_top.set_ylabel("MG amplitude")
ax_top.grid(alpha=0.25); ax_top.legend(loc="upper right", fontsize=9)
ax_top.set_title("(a) Mackey–Glass forecast", loc="left", fontsize=11, weight="bold")

ax_mid = fig.add_subplot(gs[1, :2])
hm_canvas = np.full_like(hm, np.nan, dtype=float)
im = ax_mid.imshow(hm_canvas, aspect="auto", cmap="RdBu_r", vmin=-2.5, vmax=2.5,
                    interpolation="nearest")
ax_mid.set_xlim(-0.5, n_test - 0.5); ax_mid.set_ylim(N - 0.5, -0.5)
ax_mid.set_xlabel("time step"); ax_mid.set_ylabel("cell index")
ax_mid.set_title("(b) Reservoir activity log|Id| (z-scored)",
                  loc="left", fontsize=11, weight="bold")
cbar = plt.colorbar(im, ax=ax_mid, fraction=0.04, pad=0.02); cbar.set_label("z", fontsize=9)

ax_net = fig.add_subplot(gs[:, 2])
i_idx, j_idx = np.where(np.abs(W) > 1e-6)
mask = i_idx != j_idx
i_idx, j_idx = i_idx[mask], j_idx[mask]
edge_segs = []
for ii, jj in zip(i_idx, j_idx):
    seg, = ax_net.plot([pos[ii, 0], pos[jj, 0]], [pos[ii, 1], pos[jj, 1]],
                        color="#bdc3c7", alpha=0.18, lw=0.4, zorder=1)
    edge_segs.append(seg)
node_scatter = ax_net.scatter(pos[:, 0], pos[:, 1], s=80, c=np.zeros(N),
                                cmap="RdBu_r", vmin=-2.5, vmax=2.5,
                                edgecolors="white", linewidth=0.8, zorder=3)
ax_net.set_xlim(-1.25, 1.25); ax_net.set_ylim(-1.25, 1.25)
ax_net.set_aspect("equal"); ax_net.axis("off")
ax_net.set_title(f"(c) ER_SPARSE\nN={N}, ~10% density",
                  loc="center", fontsize=11, weight="bold")

title_text = fig.suptitle("", fontsize=12, weight="bold", y=0.985)


def init():
    line_truth.set_data([], [])
    line_pred.set_data([], [])
    im.set_data(np.full_like(hm, np.nan, dtype=float))
    node_scatter.set_array(np.zeros(N))
    return [line_truth, line_pred, im, node_scatter, title_text]


def update(frame):
    t = frame
    x_axis = np.arange(t + 1)
    line_truth.set_data(x_axis, truth[: t + 1])
    line_pred.set_data(x_axis, pred[: t + 1])
    canvas = np.full_like(hm, np.nan, dtype=float)
    canvas[:, : t + 1] = hm[:, : t + 1]
    im.set_data(canvas)
    node_scatter.set_array(node_act[:, t])
    title_text.set_text(
        f"NS-RAM reservoir (Bf=100 honest physical) — t={t+1}/{n_test}"
    )
    return [line_truth, line_pred, im, node_scatter, title_text]


anim = animation.FuncAnimation(fig, update, init_func=init,
                                frames=n_test, interval=40, blit=False)
out_mp4 = OUT / "network_in_action.mp4"
print(f"[anim] rendering {n_test} frames to {out_mp4} ...")
anim.save(out_mp4, dpi=110,
            writer=animation.FFMpegWriter(fps=25, bitrate=2400))
print(f"[anim] saved {out_mp4}")
plt.close()
