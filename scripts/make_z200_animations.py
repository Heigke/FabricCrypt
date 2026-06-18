"""z200 weight-evolution animations: top topologies learning over epochs.

For each (topo, rule, N) we have W_history shape (EPOCHS+1, N, N). Render
a side-by-side mp4 per config showing:
  - LEFT: network graph (nodes positioned by topology-specific layout,
          edges colored ±W, thickness ∝ |W|), top 1% edges by |W|
  - RIGHT: accuracy curve filling in epoch-by-epoch + |W| norm

Saves to figures/z200_animations/{topo}_{rule}_N{N}.mp4

Top picks (for one or two big "winners" demo):
  - WS_SMALLWORLD/ff/N=1024 — best=1.000, final=0.833 (only stable peak)
  - HUB_SPOKE/ff/N=1024 — best=0.958, final=0.833
  - SCALE_FREE/ff/N=1024 — best=1.000, final=0.500 (lottery)
"""
from __future__ import annotations
import json
import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.gridspec import GridSpec

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results/z200_topo_rule_sweep"
OUT = ROOT / "figures/z200_animations"
OUT.mkdir(parents=True, exist_ok=True)


def topology_layout(name, N):
    """Return (N, 2) positions per topology — domain-meaningful where possible."""
    if name == "RING" or name == "WS_SMALLWORLD":
        theta = np.linspace(0, 2*np.pi, N, endpoint=False)
        return np.column_stack([np.cos(theta), np.sin(theta)])
    elif name == "GRID_2D":
        side = int(np.sqrt(N))
        xs, ys = np.meshgrid(np.linspace(-1, 1, side), np.linspace(-1, 1, side))
        return np.column_stack([xs.flatten()[:N], ys.flatten()[:N]])
    elif name == "HUB_SPOKE":
        # Hubs in inner ring, spokes outer
        n_hub = max(2, N//20)
        rng = np.random.default_rng(0)
        is_hub = np.zeros(N, dtype=bool)
        is_hub[rng.choice(N, size=n_hub, replace=False)] = True
        pos = np.zeros((N, 2))
        # Hubs: small inner circle
        hub_idx = np.where(is_hub)[0]
        spoke_idx = np.where(~is_hub)[0]
        th_h = np.linspace(0, 2*np.pi, len(hub_idx), endpoint=False)
        pos[hub_idx] = 0.3*np.column_stack([np.cos(th_h), np.sin(th_h)])
        th_s = np.linspace(0, 2*np.pi, len(spoke_idx), endpoint=False)
        pos[spoke_idx] = np.column_stack([np.cos(th_s), np.sin(th_s)])
        return pos
    elif name == "MODULAR":
        n_mod = 4
        sz = N // n_mod
        # 4 cluster centres
        centres = [(np.cos(np.pi/2 + i*np.pi/2)*0.7, np.sin(np.pi/2 + i*np.pi/2)*0.7)
                   for i in range(n_mod)]
        pos = np.zeros((N, 2))
        rng = np.random.default_rng(1)
        for m in range(n_mod):
            s = m*sz; e = s+sz
            ang = np.linspace(0, 2*np.pi, e-s, endpoint=False)
            r = 0.25 + rng.uniform(-0.05, 0.05, e-s)
            pos[s:e] = np.column_stack([centres[m][0] + r*np.cos(ang),
                                          centres[m][1] + r*np.sin(ang)])
        return pos
    elif name == "SCALE_FREE":
        # Random but degree-aware: high-degree nodes near centre
        rng = np.random.default_rng(2)
        r = rng.uniform(0.1, 1.0, N)
        th = rng.uniform(0, 2*np.pi, N)
        return np.column_stack([r*np.cos(th), r*np.sin(th)])
    else:  # ER_SPARSE, RAND_GAUSS — random
        rng = np.random.default_rng(3)
        return rng.uniform(-1, 1, (N, 2))


def make_animation(topo, rule, N, max_frames=12):
    npz_path = RES / f"{topo}_{rule}_N{N}_s0_W.npz"
    json_path = RES / f"{topo}_{rule}_N{N}_s0.json"
    if not npz_path.exists():
        print(f"[anim] skip {topo}/{rule}/N={N}: no W history"); return
    d = np.load(npz_path)
    W_hist = d["W_history"]      # (EPOCHS+1, N, N)
    meta = json.loads(json_path.read_text())
    history = meta["history"]
    accs = [0.0] + [h["acc"] for h in history]
    Wnorms = [float(np.linalg.norm(W_hist[0]))] + [h["Wnorm"] for h in history]
    n_frames = min(max_frames, W_hist.shape[0])

    pos = topology_layout(topo, N)
    # For visual sanity, sub-sample to max 256 nodes for plotting
    if N > 256:
        rng = np.random.default_rng(0)
        sub = rng.choice(N, size=256, replace=False)
    else:
        sub = np.arange(N)
    pos_s = pos[sub]

    fig = plt.figure(figsize=(12, 5.5))
    gs = GridSpec(2, 2, figure=fig, width_ratios=[1.4, 1.0],
                  height_ratios=[1, 1], hspace=0.35, wspace=0.3,
                  left=0.04, right=0.97, top=0.92, bottom=0.10)
    ax_net = fig.add_subplot(gs[:, 0])
    ax_acc = fig.add_subplot(gs[0, 1])
    ax_wn  = fig.add_subplot(gs[1, 1])

    # Accuracy axis prep
    ax_acc.set_xlim(0, len(accs)-1); ax_acc.set_ylim(-0.02, 1.05)
    ax_acc.axhline(0.5, ls=":", color="grey", lw=0.7)
    ax_acc.set_ylabel("test accuracy"); ax_acc.set_xlabel("epoch")
    ax_acc.grid(alpha=0.25)
    line_acc, = ax_acc.plot([], [], "o-", color="#c0392b", lw=1.5)
    ax_wn.set_xlim(0, len(accs)-1)
    ax_wn.set_ylabel("|W|"); ax_wn.set_xlabel("epoch")
    ax_wn.grid(alpha=0.25)
    line_wn, = ax_wn.plot([], [], "o-", color="#2980b9", lw=1.5)
    ax_wn.set_ylim(0, max(Wnorms)*1.15 + 1)

    ax_net.set_xlim(-1.3, 1.3); ax_net.set_ylim(-1.3, 1.3)
    ax_net.set_aspect("equal"); ax_net.axis("off")

    title = fig.suptitle("", fontsize=12, weight="bold", y=0.98)

    # Permanent objects
    edge_lines = []  # rebuilt each frame
    node_scatter = ax_net.scatter(pos_s[:,0], pos_s[:,1], s=18,
                                    color="#2c3e50", zorder=3,
                                    edgecolors="white", linewidth=0.5)

    def render_frame(fr):
        # Remove old edges
        for ln in edge_lines:
            ln.remove()
        edge_lines.clear()
        # Map frame index in animation → frame in W_hist
        ep_idx = int(round(fr * (W_hist.shape[0] - 1) / max(1, n_frames - 1)))
        W = W_hist[ep_idx][np.ix_(sub, sub)]
        # top 1% by |W|
        Wabs = np.abs(W)
        thresh = np.quantile(Wabs[Wabs > 0], 0.99) if (Wabs>0).any() else 1
        ii, jj = np.where(Wabs >= thresh)
        # cap at 200 edges to keep render fast
        if len(ii) > 200:
            order = np.argsort(-Wabs[ii, jj])[:200]
            ii = ii[order]; jj = jj[order]
        for i, j in zip(ii, jj):
            w = W[i, j]
            color = "#3498db" if w > 0 else "#e67e22"
            alpha = float(min(0.7, 0.15 + 1.2 * abs(w)/Wabs.max()))
            ln, = ax_net.plot([pos_s[i,0], pos_s[j,0]],
                                [pos_s[i,1], pos_s[j,1]],
                                color=color, alpha=alpha, lw=0.8, zorder=1)
            edge_lines.append(ln)
        ax_net.set_title(f"{topo} (N={N}) — epoch {ep_idx}/{W_hist.shape[0]-1}",
                          fontsize=11, weight="bold")
        line_acc.set_data(range(ep_idx+1), accs[:ep_idx+1])
        line_wn.set_data(range(ep_idx+1), Wnorms[:ep_idx+1])
        title.set_text(f"{topo} + {rule}  →  best={meta['best_acc']:.3f}  final={meta['final_acc']:.3f}")
        return [node_scatter, line_acc, line_wn, title] + edge_lines

    anim = animation.FuncAnimation(fig, render_frame, frames=n_frames,
                                     interval=600, blit=False)
    out_mp4 = OUT / f"{topo}_{rule}_N{N}.mp4"
    print(f"[anim] rendering {topo}/{rule}/N={N} → {out_mp4} ...", flush=True)
    anim.save(out_mp4, dpi=110, writer=animation.FFMpegWriter(fps=2, bitrate=2400))
    plt.close()
    print(f"[anim] saved {out_mp4}")


if __name__ == "__main__":
    # Top picks: best-stable + best-peak, + a few interesting layouts
    targets = [
        ("WS_SMALLWORLD", "ff",   1024),  # best stable: 1.000 best, 0.833 final
        ("HUB_SPOKE",     "ff",   1024),  # 0.958 best, 0.833 final
        ("MODULAR",       "ff",   1024),  # 0.958 best, 0.833 final
        ("RAND_GAUSS",    "ff",   1024),  # 0.958 best, 0.833 final
        ("SCALE_FREE",    "ff",   1024),  # 1.000 peak, lottery
        ("GRID_2D",       "ff",   1024),  # spatial layout, fun viz
    ]
    if len(sys.argv) > 1:
        # CLI: comma-separated topo names to render (or "all")
        if sys.argv[1] != "all":
            keep = sys.argv[1].split(",")
            targets = [t for t in targets if t[0] in keep]
    for t in targets:
        try:
            make_animation(*t)
        except Exception as e:
            print(f"[anim] FAIL {t}: {e}")
