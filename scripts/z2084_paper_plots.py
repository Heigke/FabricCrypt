#!/usr/bin/env python3
"""z2084 Neuromorphic Reservoir Transformer — Publication-Quality Figures.

Generates 4 figures for the paper:
  1. fig_z2084_scorecard.png    — 18-test pass/fail scorecard
  2. fig_z2084_architecture.png — Architecture diagram with substrate tokens
  3. fig_z2084_evolution.png    — z2076→z2084 evolution chart
  4. fig_z2084_ablation.png     — Ablation analysis
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

OUT_DIR = "/tmp/z2038_paper/figures"
os.makedirs(OUT_DIR, exist_ok=True)

# ---------- Style ----------
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial"],
    "font.size": 10,
    "axes.linewidth": 0.8,
    "axes.edgecolor": "#333333",
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "figure.dpi": 200,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.15,
})

GREEN = "#2ecc71"
RED = "#e74c3c"
BLUE = "#2980b9"
ORANGE = "#e67e22"
THERMALRED = "#c0392b"
GRAY = "#7f8c8d"
PURPLE = "#8e44ad"
GOLD = "#f39c12"
LIGHTGRAY = "#ecf0f1"

# ============================================================
# FIGURE 1: Scorecard
# ============================================================
def fig_scorecard():
    tests = [
        ("T1  Classification accuracy", True, "avg=99.3%"),
        ("T2  Gate separation", True, "sep=0.876"),
        ("T3  Embodiment gap", True, "gap=49.7pp"),
        ("T4  Statistical significance", True, "t=14.36"),
        ("T5  Cross-arch AUROC", True, "AUROC=0.855"),
        ("T6  Gate–HW correlation", True, "r=0.712"),
        ("T7  Gaslighting consistency", False, "sep=0.0006"),
        ("T8  Gaslighting AUROC", False, "AUROC=0.463"),
        ("T9  Thermal self-model", True, "MAE=1.35°C"),
        ("T10 Energy efficiency", True, "16.96 acc/W"),
        ("T11 Delta attention weight", False, "attn=0.185"),
        ("T12 Analog ablation", False, "drop=0.01pp"),
        ("T13 Delta scramble", False, "drop=0.01pp"),
        ("T14 Action variability", True, "var=0.500"),
        ("T15 Signal hierarchy", False, "FAIL"),
        ("T16 Capacity cost", True, "464K vs 439K"),
        ("T17 Reservoir dynamics", True, "spec=0.0024"),
        ("T18 SMN responsive", True, "thm changed"),
    ]

    fig, ax = plt.subplots(figsize=(8, 6.5))
    n = len(tests)
    y_pos = np.arange(n)

    for i, (name, passed, detail) in enumerate(tests):
        color = GREEN if passed else RED
        ax.barh(i, 1.0, color=color, alpha=0.85, height=0.7, edgecolor="white", linewidth=0.5)
        ax.text(0.02, i, name, va="center", ha="left", fontsize=8.5,
                fontweight="bold", color="white",
                path_effects=[pe.withStroke(linewidth=2, foreground="#333")])
        ax.text(0.98, i, detail, va="center", ha="right", fontsize=7.5,
                color="white", fontstyle="italic",
                path_effects=[pe.withStroke(linewidth=1.5, foreground="#333")])

    ax.set_yticks([])
    ax.set_xticks([])
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.5, n - 0.5)
    ax.invert_yaxis()
    ax.set_title("z2084 Neuromorphic Reservoir Transformer — Test Scorecard (12/18 PASS)",
                 fontsize=11, fontweight="bold", pad=12)

    # Legend
    p_pass = mpatches.Patch(color=GREEN, label="PASS (12)")
    p_fail = mpatches.Patch(color=RED, label="FAIL (6)")
    ax.legend(handles=[p_pass, p_fail], loc="lower right", fontsize=9,
              frameon=True, fancybox=True, shadow=False, edgecolor="#ccc")

    for spine in ax.spines.values():
        spine.set_visible(False)

    fig.savefig(os.path.join(OUT_DIR, "fig_z2084_scorecard.png"))
    plt.close(fig)
    print("  [1/4] fig_z2084_scorecard.png")


# ============================================================
# FIGURE 2: Architecture Diagram
# ============================================================
def fig_architecture():
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 8)
    ax.set_aspect("equal")
    ax.axis("off")

    def box(x, y, w, h, color, label, fontsize=8, alpha=0.9, textcolor="white"):
        rect = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.12",
                              facecolor=color, edgecolor="white", linewidth=1.5, alpha=alpha)
        ax.add_patch(rect)
        ax.text(x + w/2, y + h/2, label, ha="center", va="center",
                fontsize=fontsize, fontweight="bold", color=textcolor,
                path_effects=[pe.withStroke(linewidth=1.5, foreground="#222")])

    def arrow(x1, y1, x2, y2, color="#555", style="->", lw=1.5):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle=style, color=color, lw=lw,
                                    connectionstyle="arc3,rad=0.0"))

    def curved_arrow(x1, y1, x2, y2, color="#555", rad=0.3, lw=1.5):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="->", color=color, lw=lw,
                                    connectionstyle=f"arc3,rad={rad}"))

    # Title
    ax.text(5, 7.6, "z2084 Neuromorphic Reservoir Transformer Architecture",
            ha="center", va="center", fontsize=13, fontweight="bold")
    ax.text(5, 7.25, "5 substrate tokens × 4-head self-attention × 464K params",
            ha="center", va="center", fontsize=9, color="#555")

    # --- Bottom: ISA Execution (GPU) ---
    box(1.5, 0.15, 7, 0.7, "#34495e", "GPU ISA Execution  (s_setreg MODE[7:0], s_getreg SHADER_CYCLES)", fontsize=8)

    # --- Substrate Tokens (row 1) ---
    tokens = [
        ("δ delta\n(5 dims)", BLUE, 0.8),
        ("CLK\n(4 dims)", ORANGE, 2.6),
        ("Thermal\n(15 dims)", THERMALRED, 4.4),
        ("Status\n(3 dims)", GRAY, 6.2),
        ("Action\n(2 dims)", PURPLE, 8.0),
    ]
    tok_y = 1.3
    tok_h = 0.9
    tok_w = 1.4
    for label, color, x in tokens:
        box(x, tok_y, tok_w, tok_h, color, label, fontsize=7.5)

    # Arrows from ISA to tokens
    for label, color, x in tokens:
        arrow(x + tok_w/2, 0.85, x + tok_w/2, tok_y, color="#666", lw=1.0)

    # --- Linear Projection ---
    box(1.5, 2.6, 7, 0.55, "#2c3e50", "Linear Projection → TOKEN_DIM=32 each", fontsize=9)
    for _, _, x in tokens:
        arrow(x + tok_w/2, tok_y + tok_h, x + tok_w/2, 2.6, color="#666", lw=1.0)

    # --- Multi-Head Self-Attention ---
    attn_y = 3.5
    box(1.5, attn_y, 7, 0.7, "#1a5276", "4-Head Multi-Head Self-Attention  (5×32 → 5×32)", fontsize=9.5)
    arrow(5, 3.15, 5, attn_y, color="#444", lw=1.5)

    # --- Reservoir Dynamics ---
    box(1.5, 4.55, 7, 0.5, "#1b4f72", "Reservoir Dynamics  (spectral radius ≈ 0.0024)", fontsize=9)
    arrow(5, attn_y + 0.7, 5, 4.55, color="#444", lw=1.5)

    # --- Output Heads ---
    heads = [
        ("Gate\nHead", GREEN, 1.0, "g∈[0,1]"),
        ("Task\nHead", BLUE, 3.0, "ŷ (class)"),
        ("Action\nHead", PURPLE, 5.0, "ISA config"),
        ("Self-Model\nHead", THERMALRED, 7.0, "T̂ (°C)"),
    ]
    head_y = 5.5
    head_h = 0.8
    head_w = 1.8
    for label, color, x, sub in heads:
        box(x, head_y, head_w, head_h, color, label, fontsize=8)
        ax.text(x + head_w/2, head_y - 0.18, sub, ha="center", va="top",
                fontsize=6.5, color="#555", fontstyle="italic")

    for _, _, x, _ in heads:
        arrow(x + head_w/2, 5.05, x + head_w/2, head_y, color="#444", lw=1.2)

    # --- Consistency Head (small) ---
    box(3.5, 6.6, 3, 0.5, GOLD, "Consistency Head (gaslighting detector)", fontsize=8)
    arrow(5, head_y + head_h, 5, 6.6, color="#444", lw=1.0)

    # --- Closed-loop arrow from Action Head back to ISA ---
    curved_arrow(5.9, 5.5, 8.8, 0.85, color=PURPLE, rad=-0.4, lw=2.0)
    ax.text(9.3, 3.1, "Closed\nLoop", ha="center", va="center",
            fontsize=7, color=PURPLE, fontweight="bold", rotation=90)

    fig.savefig(os.path.join(OUT_DIR, "fig_z2084_architecture.png"))
    plt.close(fig)
    print("  [2/4] fig_z2084_architecture.png")


# ============================================================
# FIGURE 3: Evolution Chart
# ============================================================
def fig_evolution():
    exps = ["z2076", "z2077", "z2078", "z2079", "z2080", "z2081", "z2084"]
    acc =  [98.9,    98.9,    98.6,    99.2,    87.5,    91.3,    99.3]
    gap =  [40.5,    40.5,    50.2,    49.2,    38.0,    41.8,    49.7]
    passes = ["12/12", "13/15", "13/14", "11/16", "12/14", "15/16", "12/18"]
    descs = [
        "Pure Math\nActuation",
        "Tri-Sensor\nSelf-Model",
        "Closed-Loop\nISA",
        "Analog\nFalsification",
        "Deep Analog\nEmbodiment",
        "Per-Core\nPhysiology",
        "Neuromorphic\nReservoir Tx",
    ]

    x = np.arange(len(exps))
    width = 0.55

    fig, ax1 = plt.subplots(figsize=(10, 5))

    # Bars: embodiment gap
    bars = ax1.bar(x, gap, width, color=[BLUE]*6 + [ORANGE], alpha=0.75,
                   edgecolor="white", linewidth=1, zorder=2, label="Embodiment gap (pp)")
    ax1.set_ylabel("Embodiment Gap (pp)", fontsize=10, color=BLUE)
    ax1.set_ylim(0, 65)
    ax1.tick_params(axis="y", labelcolor=BLUE)

    # Pass-rate annotation
    for i, (b, p) in enumerate(zip(bars, passes)):
        ax1.text(b.get_x() + b.get_width()/2, b.get_height() + 1.2, p,
                 ha="center", va="bottom", fontsize=8, fontweight="bold", color="#333")

    # Line: accuracy
    ax2 = ax1.twinx()
    ax2.plot(x, acc, "o-", color=THERMALRED, linewidth=2.2, markersize=7,
             zorder=3, label="Accuracy (%)")
    ax2.set_ylabel("Classification Accuracy (%)", fontsize=10, color=THERMALRED)
    ax2.set_ylim(80, 105)
    ax2.tick_params(axis="y", labelcolor=THERMALRED)

    # X-axis
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"{e}\n{d}" for e, d in zip(exps, descs)],
                        fontsize=7.5, ha="center")
    ax1.set_xlabel("")

    # Grid
    ax1.set_axisbelow(True)
    ax1.yaxis.grid(True, alpha=0.3, linestyle="--")
    ax1.spines["top"].set_visible(False)
    ax2.spines["top"].set_visible(False)

    # Legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left",
               fontsize=9, frameon=True, fancybox=True, edgecolor="#ccc")

    ax1.set_title("FEEL Experiment Evolution: z2076 → z2084",
                  fontsize=12, fontweight="bold", pad=12)

    fig.savefig(os.path.join(OUT_DIR, "fig_z2084_evolution.png"))
    plt.close(fig)
    print("  [3/4] fig_z2084_evolution.png")


# ============================================================
# FIGURE 4: Ablation Analysis
# ============================================================
def fig_ablation():
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.5), gridspec_kw={"width_ratios": [3, 2, 2]})

    # --- Panel A: Classification accuracy ablation ---
    ax = axes[0]
    conditions = ["Full\nModel", "Flat\nBaseline", "No Analog\n(δ only)", "Scrambled\nδ"]
    accs = [99.3, 49.6, 99.3, 99.3]
    colors = [GREEN, RED, BLUE, ORANGE]
    bars = ax.bar(range(4), accs, color=colors, alpha=0.85, edgecolor="white", width=0.65)
    for b, v in zip(bars, accs):
        ax.text(b.get_x() + b.get_width()/2, v + 1.5, f"{v}%",
                ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax.set_xticks(range(4))
    ax.set_xticklabels(conditions, fontsize=8)
    ax.set_ylabel("Accuracy (%)", fontsize=10)
    ax.set_ylim(0, 115)
    ax.set_title("A. Classification Ablation", fontsize=10, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.yaxis.grid(True, alpha=0.25, linestyle="--")
    ax.set_axisbelow(True)

    # Annotation: key finding
    ax.annotate("Analog channels carry\n0 signal under ISA-only\nactuation (same acc!)",
                xy=(2, 99.3), xytext=(2.7, 75),
                fontsize=7.5, ha="center", color=RED, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=RED, lw=1.2),
                bbox=dict(boxstyle="round,pad=0.3", fc="#fff3e0", ec=RED, alpha=0.9))

    # --- Panel B: Gate separation ---
    ax = axes[1]
    conds_g = ["Full", "Flat", "No\nAnalog", "Scrambled"]
    gate_sep = [0.876, 0.0, 0.876, 0.876]
    bars = ax.bar(range(4), gate_sep, color=[GREEN, RED, BLUE, ORANGE],
                  alpha=0.85, edgecolor="white", width=0.6)
    for b, v in zip(bars, gate_sep):
        ax.text(b.get_x() + b.get_width()/2, v + 0.02, f"{v:.3f}",
                ha="center", va="bottom", fontsize=8, fontweight="bold")
    ax.set_xticks(range(4))
    ax.set_xticklabels(conds_g, fontsize=8)
    ax.set_ylabel("Gate Separation", fontsize=10)
    ax.set_ylim(0, 1.1)
    ax.set_title("B. Gate Separation", fontsize=10, fontweight="bold")
    ax.axhline(0.5, color="#999", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.yaxis.grid(True, alpha=0.25, linestyle="--")
    ax.set_axisbelow(True)

    # --- Panel C: Thermal self-model + attention ---
    ax = axes[2]
    metrics = ["Thermal\nMAE (°C)", "δ Attention\nWeight", "Analog\nAblation (pp)"]
    values = [1.35, 0.185, 0.01]
    thresholds = [5.0, 0.25, 5.0]
    colors_c = [GREEN, RED, RED]
    bars = ax.bar(range(3), values, color=colors_c, alpha=0.85, edgecolor="white", width=0.55)

    # Threshold markers
    for i, thr in enumerate(thresholds):
        if thr <= 6:
            ax.plot([i - 0.25, i + 0.25], [thr, thr], "k--", linewidth=1.0, alpha=0.6)
            ax.text(i + 0.28, thr, f"thr={thr}", fontsize=6, va="center", color="#666")

    for b, v in zip(bars, values):
        ax.text(b.get_x() + b.get_width()/2, v + 0.1, f"{v}",
                ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax.set_xticks(range(3))
    ax.set_xticklabels(metrics, fontsize=8)
    ax.set_ylabel("Value", fontsize=10)
    ax.set_ylim(0, 6.5)
    ax.set_title("C. Secondary Metrics", fontsize=10, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.yaxis.grid(True, alpha=0.25, linestyle="--")
    ax.set_axisbelow(True)

    fig.suptitle("z2084 Ablation Analysis — Analog Channels Carry No ISA Personality Signal",
                 fontsize=12, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "fig_z2084_ablation.png"))
    plt.close(fig)
    print("  [4/4] fig_z2084_ablation.png")


# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    print(f"Saving figures to {OUT_DIR}/")
    fig_scorecard()
    fig_architecture()
    fig_evolution()
    fig_ablation()
    print(f"\nDone. 4 figures saved to {OUT_DIR}/")
