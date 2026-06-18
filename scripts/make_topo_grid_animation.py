"""Side-by-side 2×2 animation of the four BEST z200 topologies at N=1024.

z200 winners (FF rule):
  - WS_SMALLWORLD: 1.000 best, 0.833 final  ⭐ best stable
  - HUB_SPOKE:     0.958 best, 0.833 final
  - MODULAR:       0.958 best, 0.833 final
  - RAND_GAUSS:    0.958 best, 0.833 final

Single mp4 with 4 panels evolving in sync — demonstrates how
different physical chip layouts converge to similar final accuracy
via the same Forward-Forward rule.

Output: figures/z200_animations/topo_grid_2x2.mp4
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.gridspec import GridSpec

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results/z200_topo_rule_sweep"
OUT = ROOT / "figures/z200_animations"; OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ROOT))
from scripts.make_z200_animations import topology_layout

PICKS = [
    ("WS_SMALLWORLD", "ff", 1024, "(a) WS Small-World"),
    ("HUB_SPOKE",     "ff", 1024, "(b) Hub-Spoke"),
    ("MODULAR",       "ff", 1024, "(c) Modular (4 clusters)"),
    ("RAND_GAUSS",    "ff", 1024, "(d) Random Gaussian"),
]
N_FRAMES = 12
SUB_NODES = 256


def load_pick(topo, rule, N):
    npz = RES / f"{topo}_{rule}_N{N}_s0_W.npz"
    js  = RES / f"{topo}_{rule}_N{N}_s0.json"
    d = np.load(npz)
    meta = json.loads(js.read_text())
    return d["W_history"], meta


# Pre-load all
loaded = {}
for topo, rule, N, label in PICKS:
    Wh, meta = load_pick(topo, rule, N)
    pos = topology_layout(topo, N)
    rng = np.random.default_rng(0)
    sub = rng.choice(N, size=SUB_NODES, replace=False) if N > SUB_NODES else np.arange(N)
    loaded[topo] = {
        "Wh": Wh, "meta": meta, "pos": pos[sub], "sub": sub,
        "accs": [0.0] + [h["acc"] for h in meta["history"]],
        "max_ep": Wh.shape[0] - 1,
    }
    print(f"[anim] {topo}: history len {Wh.shape[0]}, "
          f"final acc {meta['final_acc']:.3f}", flush=True)

fig = plt.figure(figsize=(13, 10))
gs = GridSpec(2, 2, figure=fig, hspace=0.20, wspace=0.10,
              left=0.04, right=0.97, top=0.93, bottom=0.05)
axes = [fig.add_subplot(gs[i//2, i%2]) for i in range(4)]
for ax in axes:
    ax.set_xlim(-1.3, 1.3); ax.set_ylim(-1.3, 1.3)
    ax.set_aspect("equal"); ax.axis("off")

scatters = []
for i, (topo, _, _, label) in enumerate(PICKS):
    d = loaded[topo]
    s = axes[i].scatter(d["pos"][:,0], d["pos"][:,1], s=14,
                          color="#2c3e50", zorder=3,
                          edgecolors="white", linewidth=0.4)
    scatters.append(s)

edge_lines = [[] for _ in PICKS]
sub_titles = [None]*4
for i, (topo, rule, N, label) in enumerate(PICKS):
    d = loaded[topo]
    sub_titles[i] = axes[i].text(
        0, 1.18, "", fontsize=10, ha="center",
        weight="bold", transform=axes[i].transData)

title = fig.suptitle("", fontsize=13, weight="bold", y=0.985)


def render_frame(fr):
    ep_idx_global = int(round(fr * 11 / max(1, N_FRAMES - 1)))
    artists = []
    for i, (topo, rule, N, label) in enumerate(PICKS):
        d = loaded[topo]
        ax = axes[i]
        # Remove old edges
        for ln in edge_lines[i]:
            ln.remove()
        edge_lines[i].clear()
        # Map epoch
        ep = min(ep_idx_global, d["max_ep"])
        W = d["Wh"][ep][np.ix_(d["sub"], d["sub"])]
        # top 1.5%
        Wabs = np.abs(W)
        if (Wabs > 0).any():
            thresh = np.quantile(Wabs[Wabs > 0], 0.985)
            ii, jj = np.where(Wabs >= thresh)
            if len(ii) > 150:
                order = np.argsort(-Wabs[ii, jj])[:150]
                ii = ii[order]; jj = jj[order]
            for ia, ib in zip(ii, jj):
                w = W[ia, ib]
                color = "#3498db" if w > 0 else "#e67e22"
                alpha = float(min(0.7, 0.15 + 1.2 * abs(w) / Wabs.max()))
                ln, = ax.plot([d["pos"][ia,0], d["pos"][ib,0]],
                                [d["pos"][ia,1], d["pos"][ib,1]],
                                color=color, alpha=alpha, lw=0.7, zorder=1)
                edge_lines[i].append(ln)
        artists.extend(edge_lines[i])
        artists.append(scatters[i])
        sub_titles[i].set_text(f"{label}  ep {ep}/{d['max_ep']}  "
                               f"acc={d['accs'][ep]:.2f}")
        artists.append(sub_titles[i])
    title.set_text(f"NS-RAM self-learning across 4 topologies — "
                   f"Forward-Forward, N=1024, no readout — frame {fr+1}/{N_FRAMES}")
    artists.append(title)
    return artists


anim = animation.FuncAnimation(fig, render_frame, frames=N_FRAMES,
                                 interval=600, blit=False)
out_mp4 = OUT / "topo_grid_2x2.mp4"
print(f"[anim] rendering {N_FRAMES} frames → {out_mp4} ...", flush=True)
anim.save(out_mp4, dpi=120, writer=animation.FFMpegWriter(fps=2, bitrate=2400))
plt.close()
print(f"[anim] saved {out_mp4}")
