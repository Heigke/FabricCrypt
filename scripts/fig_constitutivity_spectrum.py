#!/usr/bin/env python3
"""
fig_constitutivity_spectrum.py

Publication-quality conceptual figure: The Constitutivity Spectrum.
Central thesis of FEEL paper bridge section — from software simulation
to substrate-is-computation.

Output: /tmp/feel_paper/FEEL__Functionally_Embodied_Emergent_Learning__13_/figures/fig_constitutivity_spectrum.png
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import FancyArrowPatch
import matplotlib.patheffects as pe

OUT = "/tmp/feel_paper/FEEL__Functionally_Embodied_Emergent_Learning__13_/figures/fig_constitutivity_spectrum.png"

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
positions = [0, 1, 2, 3, 4, 5]
n = len(positions)
xs = np.linspace(0.08, 0.92, n)  # normalised x positions

labels_top = [
    "Software\nSimulation",
    "Sysfs\nTelemetry",
    "ISA\nRegisters",
    "MODE writes +\nThermalSoftmax",
    "NS-RAM\nAvalanche",
    "Biology",
]

labels_mid = [
    "z907",
    "z2000s",
    "z2050",
    "z2103",
    "Lanza 2024",
    "Unknown",
]

killshots = [
    r"Kill-shot: $p$=1.0 FAIL",
    "Kill-shot: <1 pp",
    "Kill-shot: 98.6 pp PASS",
    "Kill-shot: 99.0 pp PASS",
    "Kill-shot: By construction",
    "Kill-shot: Unknown",
]

labels_bottom = [
    "Hardware\nfungible",
    "Telemetry\nreads",
    "Register\ncoupling",
    "Transfer fn.\nmodulation",
    "Device physics\n= algorithm",
    "Ion channel\nkinetics",
]

# colours for kill-shot text: fail=grey, marginal=grey, pass=dark green, neutral=grey
ks_colors = ["#a50026", "#666666", "#1a7a2e", "#1a7a2e", "#555555", "#555555"]

# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "mathtext.fontset": "cm",
})

fig, ax = plt.subplots(figsize=(12, 5))
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis("off")

# -- gradient bar ----------------------------------------------------------
bar_y = 0.52
bar_h = 0.06
cmap = LinearSegmentedColormap.from_list(
    "spectrum", ["#2166ac", "#f7f7f7", "#b2182b"], N=512
)
gradient = np.linspace(0, 1, 512).reshape(1, -1)
ax.imshow(
    gradient,
    aspect="auto",
    cmap=cmap,
    extent=[0.04, 0.96, bar_y - bar_h / 2, bar_y + bar_h / 2],
    zorder=2,
)
# thin border around bar
rect = mpatches.FancyBboxPatch(
    (0.04, bar_y - bar_h / 2),
    0.92,
    bar_h,
    boxstyle="round,pad=0.003",
    linewidth=0.8,
    edgecolor="#333333",
    facecolor="none",
    zorder=3,
)
ax.add_patch(rect)

# endpoint labels on bar
ax.text(0.02, bar_y, "Spectatorial", ha="right", va="center", fontsize=9,
        fontstyle="italic", color="#2166ac", fontweight="bold",
        transform=ax.transAxes)
ax.text(0.98, bar_y, "Constitutive", ha="left", va="center", fontsize=9,
        fontstyle="italic", color="#b2182b", fontweight="bold",
        transform=ax.transAxes)

# -- markers and labels ----------------------------------------------------
for i, x in enumerate(xs):
    # vertical tick through bar
    ax.plot([x, x], [bar_y - bar_h / 2 - 0.01, bar_y + bar_h / 2 + 0.01],
            color="#333333", lw=1.0, zorder=4)
    # dot on bar
    frac = (x - 0.04) / 0.92  # 0..1 position in bar
    dot_color = cmap(frac)
    ax.plot(x, bar_y, "o", color=dot_color, markersize=9,
            markeredgecolor="#333333", markeredgewidth=0.8, zorder=5)

    # top label (name)
    ax.text(x, bar_y + 0.14, labels_top[i], ha="center", va="bottom",
            fontsize=9.5, fontweight="bold", linespacing=1.2)

    # experiment id
    ax.text(x, bar_y + 0.08, labels_mid[i], ha="center", va="bottom",
            fontsize=8, fontstyle="italic", color="#555555")

    # kill-shot below bar
    ax.text(x, bar_y - 0.09, killshots[i], ha="center", va="top",
            fontsize=7.5, color=ks_colors[i], fontweight="medium")

    # bottom mechanistic label
    ax.text(x, bar_y - 0.22, labels_bottom[i], ha="center", va="top",
            fontsize=8, color="#444444", linespacing=1.15,
            bbox=dict(boxstyle="round,pad=0.25", facecolor="#f0f0f0",
                      edgecolor="#cccccc", linewidth=0.5))

# -- curved arrow: FEEL trajectory (positions 0 -> 3) ----------------------
arrow_y = bar_y - 0.34
mid_x = (xs[0] + xs[3]) / 2
# draw a curved path with annotation
ax.annotate(
    "",
    xy=(xs[3], arrow_y + 0.02),
    xytext=(xs[0], arrow_y + 0.02),
    arrowprops=dict(
        arrowstyle="->,head_width=0.3,head_length=0.15",
        connectionstyle="arc3,rad=-0.25",
        color="#2166ac",
        lw=1.8,
    ),
    zorder=4,
)
ax.text(mid_x, arrow_y - 0.06, "FEEL 14-month trajectory",
        ha="center", va="top", fontsize=8.5, fontstyle="italic",
        color="#2166ac", fontweight="bold")

# -- dashed connector: positions 3 -> 4 ("This work: SPICE bridge") --------
mid34 = (xs[3] + xs[4]) / 2
ax.annotate(
    "",
    xy=(xs[4] - 0.005, bar_y - 0.125),
    xytext=(xs[3] + 0.005, bar_y - 0.125),
    arrowprops=dict(
        arrowstyle="<->",
        linestyle="dashed",
        color="#8856a7",
        lw=1.5,
    ),
    zorder=4,
)
ax.text(mid34, bar_y - 0.17, "This work:\nSPICE bridge",
        ha="center", va="top", fontsize=7.5, color="#8856a7",
        fontweight="bold", linespacing=1.1)

# -- title -----------------------------------------------------------------
ax.text(
    0.50, 0.97,
    "The Constitutivity Spectrum: From Software Simulation to Substrate-Is-Computation",
    ha="center", va="top", fontsize=13, fontweight="bold",
    transform=ax.transAxes,
)

# ---------------------------------------------------------------------------
plt.tight_layout(pad=0.5)
fig.savefig(OUT, dpi=300, bbox_inches="tight", facecolor="white")
plt.close()
print(f"Saved -> {OUT}")
