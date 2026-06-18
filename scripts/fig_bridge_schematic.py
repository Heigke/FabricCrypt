#!/usr/bin/env python3
"""
fig_bridge_schematic.py — Publication-quality NS-RAM <-> GPU bridge block diagram.

Generates: /tmp/feel_paper/FEEL__Functionally_Embodied_Emergent_Learning__13_/figures/fig_bridge_schematic.png
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

OUT = "/tmp/feel_paper/FEEL__Functionally_Embodied_Emergent_Learning__13_/figures/fig_bridge_schematic.png"

# ── Style ────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 8,
    "axes.linewidth": 0.5,
    "text.usetex": False,
})

fig, ax = plt.subplots(figsize=(12, 6))
ax.set_xlim(0, 12)
ax.set_ylim(0, 6)
ax.set_aspect("equal")
ax.axis("off")

# ── Colour palette ───────────────────────────────────────────────────────
BLUE   = "#2563EB"
GREEN  = "#16A34A"
RED    = "#DC2626"
BLUE_L = "#DBEAFE"
GRN_L  = "#DCFCE7"
RED_L  = "#FEE2E2"
BG     = "#FFFFFF"

# ── Helper: rounded box with title ──────────────────────────────────────
def draw_block(x, y, w, h, title, items, border_col, fill_col,
               title_fs=9, item_fs=7, bold_title=True):
    box = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.08",
        linewidth=1.8, edgecolor=border_col, facecolor=fill_col,
        zorder=2,
    )
    ax.add_patch(box)
    weight = "bold" if bold_title else "normal"
    ax.text(x + w / 2, y + h - 0.22, title,
            ha="center", va="top", fontsize=title_fs, fontweight=weight,
            color=border_col, zorder=3)
    # horizontal rule under title
    ax.plot([x + 0.15, x + w - 0.15], [y + h - 0.38, y + h - 0.38],
            lw=0.6, color=border_col, alpha=0.5, zorder=3)
    for i, txt in enumerate(items):
        ax.text(x + 0.18, y + h - 0.55 - i * 0.28, txt,
                ha="left", va="top", fontsize=item_fs, color="#1F2937", zorder=3)

# ── Helper: signal node label (small circle + text) ─────────────────────
def node_label(x, y, label, col="#374151", fs=7, ha="center"):
    ax.plot(x, y, "o", ms=3.5, mfc="white", mec=col, mew=1.0, zorder=4)
    ax.text(x, y + 0.16, label, ha=ha, va="bottom", fontsize=fs,
            fontstyle="italic", color=col, zorder=4)

# ═══════════════════════════════════════════════════════════════════════
#  BLOCK 1 — NS-RAM Lanza Device  (left)
# ═══════════════════════════════════════════════════════════════════════
nsram_x, nsram_y, nsram_w, nsram_h = 0.3, 1.3, 3.4, 3.6
nsram_items = [
    "M2: NMOS PTM130 (W=500nm, L=250nm)",
    "Q1: Parasitic NPN BJT",
    "D1, D2: Zener (BV=2.7 V)",
    r"Avalanche: I = I$_0$·exp((V$_{cb}$−BV$_{par}$)/V$_t$)",
    r"BV$_{par}$ = 3.5 − 1.5·V$_{gate}$",
]
draw_block(nsram_x, nsram_y, nsram_w, nsram_h,
           "NS-RAM Neuron (Lanza et al. 2024)", nsram_items, BLUE, BLUE_L)

# Signal nodes
node_label(nsram_x + 0.4, nsram_y + 0.22, r"V$_{gate}$", BLUE, ha="left")
node_label(nsram_x + nsram_w - 0.25, nsram_y + nsram_h / 2 + 0.15, r"s$_{node}$", BLUE)
node_label(nsram_x + nsram_w / 2, nsram_y + 0.22, r"b$_{node}$", BLUE)

# ═══════════════════════════════════════════════════════════════════════
#  BLOCK 2 — Pazos LIF Membrane  (centre)
# ═══════════════════════════════════════════════════════════════════════
lif_x, lif_y, lif_w, lif_h = 4.3, 1.3, 3.4, 3.6
lif_items = [
    r"C$_{int}$ = 102 fF",
    "Spike detector (adaptive threshold)",
    "Double-inverter threshold",
    "Auto-reset circuit",
    "4 synaptic inputs (gated via spike_det)",
]
draw_block(lif_x, lif_y, lif_w, lif_h,
           "LIF Neuron (Pazos et al.)", lif_items, GREEN, GRN_L)

# I/O labels
node_label(lif_x + 0.15, lif_y + lif_h / 2 + 0.15, "spike_det", GREEN, ha="left")
node_label(lif_x + lif_w - 0.15, lif_y + lif_h / 2 + 0.15, r"V$_{spike}$", GREEN, ha="right")

# ═══════════════════════════════════════════════════════════════════════
#  BLOCK 3 — GPU FP16 Feedback  (right)
# ═══════════════════════════════════════════════════════════════════════
gpu_x, gpu_y, gpu_w, gpu_h = 8.3, 1.3, 3.4, 3.6
gpu_items = [
    "FP16 GEMM with rounding mode",
    "MAC accumulator",
    r"Rounding mode: V$_{mac}$ mod. by V$_{spike}$",
    r"Output: V$_{mac\_out}$",
]
draw_block(gpu_x, gpu_y, gpu_w, gpu_h,
           "GPU MAC Unit (FEEL)", gpu_items, RED, RED_L, item_fs=7)

# I/O labels
node_label(gpu_x + 0.15, gpu_y + gpu_h / 2 + 0.15, r"V$_{spike}$", RED, ha="left")
node_label(gpu_x + gpu_w - 0.25, gpu_y + gpu_h / 2 + 0.15, r"MAC$_{norm}$", RED, ha="right")

# ═══════════════════════════════════════════════════════════════════════
#  ARROWS between blocks
# ═══════════════════════════════════════════════════════════════════════

arrow_kw = dict(arrowstyle="->,head_length=0.25,head_width=0.15",
                lw=2.0, zorder=5)

# NS-RAM → LIF  (green, forward)
fwd1 = FancyArrowPatch(
    (nsram_x + nsram_w, nsram_y + nsram_h / 2 + 0.15),
    (lif_x, lif_y + lif_h / 2 + 0.15),
    connectionstyle="arc3,rad=0.0",
    color=GREEN, **arrow_kw,
)
ax.add_patch(fwd1)
ax.text((nsram_x + nsram_w + lif_x) / 2, nsram_y + nsram_h / 2 + 0.55,
        "Avalanche drain\nspikes → spike_det",
        ha="center", va="bottom", fontsize=6.5, color=GREEN, fontstyle="italic")

# LIF → GPU  (green, forward)
fwd2 = FancyArrowPatch(
    (lif_x + lif_w, lif_y + lif_h / 2 + 0.15),
    (gpu_x, gpu_y + gpu_h / 2 + 0.15),
    connectionstyle="arc3,rad=0.0",
    color=GREEN, **arrow_kw,
)
ax.add_patch(fwd2)
ax.text((lif_x + lif_w + gpu_x) / 2, lif_y + lif_h / 2 + 0.55,
        "LIF spikes →\nrounding mode",
        ha="center", va="bottom", fontsize=6.5, color=GREEN, fontstyle="italic")

# GPU → NS-RAM  (red, feedback — curves below)
fb = FancyArrowPatch(
    (gpu_x + gpu_w / 2, gpu_y),
    (nsram_x + nsram_w / 2, nsram_y),
    connectionstyle="arc3,rad=-0.35",
    color=RED, linestyle="--", **arrow_kw,
)
ax.add_patch(fb)
ax.text(6.0, 0.45,
        r"V$_{gate}$ = V$_{bias}$ + 0.2·MAC$_{norm}$ → BV$_{par}$ modulation",
        ha="center", va="center", fontsize=6.5, color=RED, fontstyle="italic",
        bbox=dict(boxstyle="round,pad=0.15", fc="white", ec=RED, lw=0.5, alpha=0.9))

# ═══════════════════════════════════════════════════════════════════════
#  KILL-SHOT ANNOTATIONS  (red X marks)
# ═══════════════════════════════════════════════════════════════════════

def kill_x(x, y, label, label_offset=(0, 0.22)):
    ax.plot(x, y, "x", ms=11, mew=2.5, color=RED, zorder=6)
    ax.text(x + label_offset[0], y + label_offset[1], label,
            ha="center", va="bottom", fontsize=6.5, fontweight="bold",
            color=RED, zorder=6,
            bbox=dict(boxstyle="round,pad=0.1", fc="white", ec=RED, lw=0.6, alpha=0.92))

# Kill-shot B: open loop — on feedback arrow (bottom centre)
kill_x(7.6, 0.62, "B: open loop", label_offset=(0.0, 0.18))

# Kill-shot C: reversed — on feedback arrow (near NS-RAM end)
kill_x(4.4, 0.72, "C: reversed", label_offset=(0.0, 0.18))

# Kill-shot D: no avalanche — inside NS-RAM
kill_x(1.8, 2.15, "D: no avalanche", label_offset=(0.0, 0.20))

# ═══════════════════════════════════════════════════════════════════════
#  BOTTOM ANNOTATION BAR
# ═══════════════════════════════════════════════════════════════════════

bar_y = 0.08
bar_h = 0.18
# Analog physics (blue) under NS-RAM
ax.add_patch(FancyBboxPatch((nsram_x, bar_y), nsram_w, bar_h,
             boxstyle="round,pad=0.04", fc=BLUE, ec="none", alpha=0.15, zorder=1))
ax.text(nsram_x + nsram_w / 2, bar_y + bar_h / 2, "Analog physics",
        ha="center", va="center", fontsize=7, color=BLUE, fontweight="bold")

# Mixed-signal (green) under LIF
ax.add_patch(FancyBboxPatch((lif_x, bar_y), lif_w, bar_h,
             boxstyle="round,pad=0.04", fc=GREEN, ec="none", alpha=0.15, zorder=1))
ax.text(lif_x + lif_w / 2, bar_y + bar_h / 2, "Mixed-signal",
        ha="center", va="center", fontsize=7, color=GREEN, fontweight="bold")

# Digital computation (red) under GPU
ax.add_patch(FancyBboxPatch((gpu_x, bar_y), gpu_w, bar_h,
             boxstyle="round,pad=0.04", fc=RED, ec="none", alpha=0.15, zorder=1))
ax.text(gpu_x + gpu_w / 2, bar_y + bar_h / 2, "Digital computation",
        ha="center", va="center", fontsize=7, color=RED, fontweight="bold")

# Spectrum arrow
ax.annotate("", xy=(11.7, bar_y - 0.08), xytext=(0.3, bar_y - 0.08),
            arrowprops=dict(arrowstyle="<->", color="#6B7280", lw=1.0))
ax.text(6.0, bar_y - 0.22,
        r"$\leftarrow$ More constitutive  |  More abstract $\rightarrow$",
        ha="center", va="top", fontsize=6.5, color="#6B7280")

# ═══════════════════════════════════════════════════════════════════════
#  TITLE
# ═══════════════════════════════════════════════════════════════════════
ax.text(6.0, 5.65,
        r"Bidirectional NS-RAM $\leftrightarrow$ GPU Bridge: Circuit Architecture and Kill-Shot Points",
        ha="center", va="center", fontsize=11, fontweight="bold", color="#111827")

# ═══════════════════════════════════════════════════════════════════════
#  SAVE
# ═══════════════════════════════════════════════════════════════════════
fig.savefig(OUT, dpi=300, bbox_inches="tight", facecolor="white", pad_inches=0.15)
plt.close(fig)
print(f"Saved → {OUT}")
