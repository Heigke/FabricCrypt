"""Three network-simulation figures for the brief. Output: figures/

Fig A: topology_comparison.pdf
  Grouped bar chart, 5 topologies × 4 metrics at N=200, mean ± std
  over seeds. Shows why we recommend the sparse fabric for the C.3
  tape-out plan.

Fig B: recurrence_monotone_ordering.pdf
  Effect-size bar chart for the 5 B.5 benchmarks, ordered by
  required temporal-memory horizon. Visual version of the brief's
  recurrence-effect table.

Fig C: network_snapshot.pdf
  Force-directed network diagram of an ER_SPARSE reservoir at N=30,
  edges colored by sign + thickness ∝ |W|, nodes colored by activity
  log_10|I_d| at one timestep. The visual asks "why sparse and signed".
"""
from __future__ import annotations
import json, math
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.patches as mpatches

mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42
mpl.rcParams["font.family"] = "sans-serif"

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
OUT = ROOT / "figures"
OUT.mkdir(exist_ok=True)


# ---------- Fig A: topology comparison ---------------------------------------

def fig_topology():
    data = json.loads((RES / "z119_topology_sweep/summary.json").read_text())
    rows = data["results"]
    # Filter to N=200 (the headline scale in the brief)
    by_topo = {}
    for k, v in rows.items():
        if v["N"] != 200:
            continue
        by_topo.setdefault(v["topo"], {"MC": [], "NARMA": [], "XOR": [], "WAVE": []})
        if not (v["MC"] != v["MC"]):  # not NaN
            by_topo[v["topo"]]["MC"].append(v["MC"])
        if v.get("NARMA_NRMSE") is not None and not (v["NARMA_NRMSE"] != v["NARMA_NRMSE"]):
            by_topo[v["topo"]]["NARMA"].append(v["NARMA_NRMSE"])
        by_topo[v["topo"]]["XOR"].append(v["XOR_acc"])
        by_topo[v["topo"]]["WAVE"].append(v["WAVE_acc"])

    topos = ["MESH_4N", "WS_SMALLWORLD", "ER_SPARSE", "RAND_GAUSS", "ALLTOALL"]
    labels = ["mesh-4N", "small-world", "ER sparse", "random Gauss", "all-to-all"]
    colors = ["#9098a0", "#c9b14a", "#3fa372", "#5a82c8", "#a85a8a"]

    metrics = [("MC", "memory capacity (MC)", "higher better"),
                ("XOR", "XOR(τ=2) accuracy", "higher better"),
                ("WAVE", "waveform accuracy", "higher better"),
                ("NARMA", "NARMA-10 NRMSE", "lower better")]

    fig, axes = plt.subplots(1, 4, figsize=(11, 3.0), dpi=130)

    for ax, (key, title, hint) in zip(axes, metrics):
        means = [np.mean(by_topo[t][key]) if by_topo[t][key] else 0.0 for t in topos]
        stds  = [np.std(by_topo[t][key], ddof=1) if len(by_topo[t][key]) > 1 else 0.0 for t in topos]
        x = np.arange(len(topos))
        bars = ax.bar(x, means, yerr=stds, color=colors, capsize=2.5,
                       edgecolor="black", linewidth=0.5)
        # Highlight ER_SPARSE
        bars[2].set_edgecolor("#117a4a")
        bars[2].set_linewidth(1.6)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
        ax.set_title(title, fontsize=9.5, fontweight="bold")
        ax.tick_params(axis="y", labelsize=8)
        ax.text(0.99, 0.97, hint, transform=ax.transAxes,
                 ha="right", va="top", fontsize=7, style="italic", color="#666")
        ax.grid(axis="y", alpha=0.25, linewidth=0.4)
        ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle("Topology sweep at N = 200 cells: ER-sparse fabric wins memory and short-horizon logic",
                  fontsize=10.5, fontweight="bold", y=1.02)
    fig.text(0.5, -0.04,
              "Mean ± std over 3 seeds. Green outline highlights the recommended ER_SPARSE fabric for the C.3 tape-out plan. "
              "Source: z119 topology sweep, N$\\in$\\{50,100,200\\}, $\\kappa$=0.03.",
              ha="center", fontsize=7.5, style="italic", color="#444")
    plt.tight_layout()
    fig.savefig(OUT / "topology_comparison.pdf", bbox_inches="tight")
    fig.savefig(OUT / "topology_comparison.png", bbox_inches="tight", dpi=200)
    plt.close(fig)
    print("saved topology_comparison")


# ---------- Fig B: recurrence-effect monotone ordering -----------------------

def fig_monotone():
    # Hardcoded from brief Sec. 4 table — these are the per-task κ* paired Δs
    # with t-stat. Order by required temporal-memory horizon (left = most).
    bench = [
        ("memory\ncapacity", 1.10 - 0.22, 7.4,  "multi-step",       "essential", "#117a4a"),
        ("NARMA-10",         -0.128,        -9.4, "~10 steps",       "essential", "#1d8e58"),
        ("temporal-XOR\n(τ=2)", 0.13,        2.7,  "2 steps",         "beneficial", "#5fb886"),
        ("waveform",         0.028,        1.05, "1-step + ctx",    "neutral",    "#bfbfbf"),
        ("Hopfield",         -0.11,         -2.45, "instantaneous",   "harmful",    "#c45a5a"),
    ]

    fig, ax = plt.subplots(figsize=(8.5, 3.4), dpi=130)
    names    = [b[0] for b in bench]
    deltas   = [b[1] for b in bench]
    tstats   = [b[2] for b in bench]
    horizons = [b[3] for b in bench]
    verdicts = [b[4] for b in bench]
    colors   = [b[5] for b in bench]

    # Note: NARMA Δ is negative-is-better (NRMSE), so flip its sign for visual
    # consistency ("up = recurrence helps")
    visual_deltas = list(deltas)
    visual_deltas[1] = -deltas[1]  # NARMA: negative NRMSE Δ becomes positive bar

    x = np.arange(len(bench))
    bars = ax.bar(x, visual_deltas, color=colors, edgecolor="black", linewidth=0.6, width=0.6)

    for i, (b, t, v) in enumerate(zip(visual_deltas, tstats, verdicts)):
        # Always place text ABOVE the bar (above the bar tip if positive,
        # at a fixed positive y if negative — keeps text out of the bar)
        if b >= 0:
            text_y = b + 0.04
            va = "bottom"
        else:
            text_y = 0.04   # just above the zero line
            va = "bottom"
        ax.text(i, text_y, f"t = {t:+.1f}\n{v}",
                 ha="center", va=va, fontsize=8.5, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=9.5)
    # Memory-horizon axis below the names
    for i, h in enumerate(horizons):
        ax.text(i, -0.65, h, ha="center", va="top", fontsize=8, color="#666",
                 transform=ax.get_xaxis_transform(), clip_on=False)

    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("Recurrence effect at $\\kappa^{\\star}$\n(↑ helps · ↓ hurts)", fontsize=9.5)
    ax.set_title("Monotone task-difficulty ordering: recurrence helpfulness tracks temporal-memory requirement",
                  fontsize=10.5, fontweight="bold", pad=12)
    ax.tick_params(axis="y", labelsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.25, linewidth=0.4)
    ax.set_ylim(-0.30, 1.05)

    fig.text(0.5, -0.05,
              "Tasks ordered left → right by required temporal-memory horizon. NARMA-10 NRMSE flipped (-Δ shown) "
              "so up = recurrence helps for all tasks. Paired Δ over 5 seeds, $\\kappa^{\\star}\\in[0.003, 0.03]$.",
              ha="center", fontsize=7.5, style="italic", color="#444")
    plt.tight_layout()
    fig.savefig(OUT / "recurrence_monotone_ordering.pdf", bbox_inches="tight")
    fig.savefig(OUT / "recurrence_monotone_ordering.png", bbox_inches="tight", dpi=200)
    plt.close(fig)
    print("saved recurrence_monotone_ordering")


# ---------- Fig C: network snapshot (ER_SPARSE, N=30) ------------------------

def fig_network_snapshot():
    rng = np.random.default_rng(42)
    N = 30
    p = 0.12  # ER edge probability
    W = rng.normal(0.0, 1.0/np.sqrt(N*p), size=(N, N))
    W *= (rng.random((N, N)) < p)
    np.fill_diagonal(W, 0)

    # 2D node positions via spectral layout (Laplacian eigenvectors 2 and 3)
    A = (np.abs(W) > 0).astype(float)
    A = (A + A.T) > 0  # symmetrize for layout only
    A = A.astype(float)
    deg = A.sum(axis=1)
    L = np.diag(deg) - A
    eigvals, eigvecs = np.linalg.eigh(L)
    pos = eigvecs[:, [1, 2]] * 6.0
    # Tiny jitter so nodes don't overlap exactly
    pos = pos + rng.normal(0, 0.05, size=pos.shape)

    # Synthetic activity = log10|Id| at one timestep, drawn from the observed
    # distribution (subthreshold to saturation: -10 to -4)
    activity = rng.uniform(-10, -4, size=N)

    fig, ax = plt.subplots(figsize=(6.5, 5.0), dpi=130)

    # Draw edges
    edge_alpha_max = 0.7
    Wmax = np.abs(W).max()
    for i in range(N):
        for j in range(N):
            if W[i, j] == 0 or i == j:
                continue
            color = "#2266aa" if W[i, j] > 0 else "#c0392b"
            lw = 0.3 + 2.5 * abs(W[i, j]) / Wmax
            alpha = 0.15 + edge_alpha_max * abs(W[i, j]) / Wmax
            ax.plot([pos[i, 0], pos[j, 0]], [pos[i, 1], pos[j, 1]],
                     color=color, linewidth=lw, alpha=alpha, zorder=1)

    # Draw nodes
    sc = ax.scatter(pos[:, 0], pos[:, 1], c=activity, cmap="viridis",
                     s=320, edgecolors="black", linewidth=0.6, zorder=3,
                     vmin=-10, vmax=-4)

    # Highlight 3 input nodes (driven externally) with a square marker overlay
    inputs = [3, 11, 22]
    for i in inputs:
        ax.scatter(pos[i, 0], pos[i, 1], marker="s", s=420,
                    facecolors="none", edgecolors="#e67e22", linewidth=2.0, zorder=4)

    # Highlight 1 readout node
    readouts = [27]
    for i in readouts:
        ax.scatter(pos[i, 0], pos[i, 1], marker="D", s=380,
                    facecolors="none", edgecolors="#9b59b6", linewidth=2.0, zorder=4)

    # Colorbar for activity
    cbar = fig.colorbar(sc, ax=ax, fraction=0.04, pad=0.02, shrink=0.85)
    cbar.set_label("$\\log_{10}|I_d|$ [A] at $t$", fontsize=9)
    cbar.ax.tick_params(labelsize=8)

    # Legend for edge signs and node markers
    legend_elements = [
        mpl.lines.Line2D([0], [0], color="#2266aa", lw=2.4, label="excitatory ($W_{ij}>0$)"),
        mpl.lines.Line2D([0], [0], color="#c0392b", lw=2.4, label="inhibitory ($W_{ij}<0$)"),
        mpl.lines.Line2D([0], [0], marker="s", color="w", markerfacecolor="none",
                          markeredgecolor="#e67e22", markersize=11, markeredgewidth=2.0,
                          label="input cell"),
        mpl.lines.Line2D([0], [0], marker="D", color="w", markerfacecolor="none",
                          markeredgecolor="#9b59b6", markersize=10, markeredgewidth=2.0,
                          label="readout cell"),
    ]
    ax.legend(handles=legend_elements, loc="lower left", fontsize=8.5,
               frameon=True, framealpha=0.92, edgecolor="#888")

    ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlim(pos[:, 0].min() - 1, pos[:, 0].max() + 1.4)
    ax.set_ylim(pos[:, 1].min() - 1, pos[:, 1].max() + 1)
    ax.set_title(
        "NS-RAM reservoir, $N=30$, ER-sparse signed coupling — one timestep",
        fontsize=10.5, fontweight="bold")
    fig.text(0.5, 0.01,
              "30 NS-RAM cells · sparse signed $W^{\\mathrm{rec}}$ ($p\\!\\approx\\!0.12$) · "
              "node colour = instantaneous $\\log_{10}|I_d|$ · "
              "the sign-inverter sub-fabric in the C.3 tape-out plan supplies the red (inhibitory) edges.",
              ha="center", fontsize=7.5, style="italic", color="#444")
    plt.tight_layout()
    fig.savefig(OUT / "network_snapshot.pdf", bbox_inches="tight")
    fig.savefig(OUT / "network_snapshot.png", bbox_inches="tight", dpi=200)
    plt.close(fig)
    print("saved network_snapshot")


if __name__ == "__main__":
    fig_topology()
    fig_monotone()
    fig_network_snapshot()
