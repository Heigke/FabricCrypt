#!/usr/bin/env python3
"""
plot_z2137_bridge.py — Publication figures for FEEL paper Section 11
  Figure 1: ThermalSoftmax SPICE validation (v6 circuit)
  Figure 2: Dual LoRA Gate SPICE validation (v7 circuit)
"""

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

# ---------- paths ----------
BASE = Path("/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy")
JSON_PATH = BASE / "results" / "z2137_feel_reaches_down.json"
OUT_DIR = BASE / "results" / "_paper_work" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

with open(JSON_PATH) as f:
    data = json.load(f)

# ---------- style ----------
try:
    plt.style.use("seaborn-v0_8-paper")
except OSError:
    plt.style.use("seaborn-paper")
plt.rcParams.update({
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "figure.dpi": 200,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
})

# ====================================================================
# FIGURE 1 — ThermalSoftmax SPICE validation (v6)
# ====================================================================
v6 = data["v6_thermalsoftmax"]
temps_plot = ["250", "300", "350", "400"]
temp_labels = [f"{t}K" for t in temps_plot]
att_matrix = np.array([v6["temp_sweep"][t]["att_mean"] for t in temps_plot])  # (4,4)

fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

# --- Left panel: grouped bar chart ---
n_temps = len(temps_plot)
n_att = 4
x = np.arange(n_temps)
width = 0.18
colors_att = ["#4C72B0", "#55A868", "#C44E52", "#8172B2"]
att_labels = [f"att{i+1}" for i in range(n_att)]

for i in range(n_att):
    offset = (i - 1.5) * width
    bars = ax1.bar(x + offset, att_matrix[:, i], width, label=att_labels[i],
                   color=colors_att[i], edgecolor="white", linewidth=0.5)

ax1.set_xticks(x)
ax1.set_xticklabels(temp_labels)
ax1.set_xlabel("Temperature")
ax1.set_ylabel("Attention Weight")
ax1.set_title("ThermalSoftmax: Attention Shift with Temperature")
ax1.legend(fontsize=8, ncol=2, loc="upper right")
ax1.set_ylim(0, 0.72)
ax1.grid(False)

# --- Right panel: monotonic decrease of dominant weight (att3) ---
all_temps = sorted(v6["temp_sweep"].keys(), key=int)
all_T = [int(t) for t in all_temps]
att3_vals = [v6["temp_sweep"][t]["att_mean"][2] for t in all_temps]

ax2.plot(all_T, att3_vals, "o-", color="#C44E52", linewidth=2, markersize=6, zorder=5)
ax2.fill_between(all_T, att3_vals, alpha=0.15, color="#C44E52")
ax2.set_xlabel("Temperature (K)")
ax2.set_ylabel("att3 (Dominant Weight)")
ax2.set_title("Dominant Weight Monotonic Decrease")
ax2.grid(True, alpha=0.3)
ax2.text(0.05, 0.05, "Spearman \u03c1 = \u22121.000",
         transform=ax2.transAxes, fontsize=10, fontweight="bold",
         bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#C44E52", alpha=0.9))

fig1.tight_layout()
out1 = OUT_DIR / "fig_feel_reaches_down_v6.png"
fig1.savefig(out1)
plt.close(fig1)
print(f"Saved: {out1}")

# ====================================================================
# FIGURE 2 — Dual LoRA Gate SPICE validation (v7)
# ====================================================================
v7 = data["v7_dual_lora_gate"]
conds_order = ["cold_250K", "normal_350K", "killshot_350K", "hot_400K"]
cond_labels = ["Cold\n250K", "Normal\n350K", "Kill-shot\n350K", "Hot\n400K"]

gates = [v7["conditions"][c]["gate"] for c in conds_order]
outputs = [v7["conditions"][c]["output"] for c in conds_order]

# Colors: blue gradient, killshot red
bar_colors = ["#6BAED6", "#2171B5", "#E31A1C", "#08519C"]

fig2, axes = plt.subplots(2, 2, figsize=(10, 8))

# --- Top-left: Sigmoid gate response ---
ax = axes[0, 0]
bars = ax.bar(range(4), gates, color=bar_colors, edgecolor="white", linewidth=0.8)
ax.set_xticks(range(4))
ax.set_xticklabels(cond_labels, fontsize=8)
ax.set_ylabel("Gate Value")
ax.set_title("Sigmoid Gate Response")
ax.set_ylim(0, 1.15)
ax.grid(False)
for i, v in enumerate(gates):
    ax.text(i, v + 0.02, f"{v:.3f}", ha="center", fontsize=8, fontweight="bold")

# --- Top-right: Crossbar output ---
ax = axes[0, 1]
bars = ax.bar(range(4), outputs, color=bar_colors, edgecolor="white", linewidth=0.8)
ax.set_xticks(range(4))
ax.set_xticklabels(cond_labels, fontsize=8)
ax.set_ylabel("Output (V)")
ax.set_title("Crossbar Output")
ax.set_ylim(0, 2.8)
ax.grid(False)
for i, v in enumerate(outputs):
    ax.text(i, v + 0.05, f"{v:.3f}", ha="center", fontsize=8, fontweight="bold")
# Annotations
kill_ratio = v7["metrics"]["kill_ratio"]
regime_ratio = v7["metrics"]["regime_ratio"]
ax.annotate(f"Kill ratio: {kill_ratio:.3f}x",
            xy=(2, outputs[2]), xytext=(2.6, 1.8),
            arrowprops=dict(arrowstyle="->", color="#E31A1C", lw=1.5),
            fontsize=9, color="#E31A1C", fontweight="bold")
ax.annotate(f"Regime: {regime_ratio:.3f}x",
            xy=(3, outputs[3]), xytext=(3.3, 1.0),
            arrowprops=dict(arrowstyle="->", color="#08519C", lw=1.5),
            fontsize=9, color="#08519C", fontweight="bold")

# --- Bottom-left: Bridge score ---
ax = axes[1, 0]
bridge = data["bridge_analysis"]["bridge_comparisons"]
test_names = [b["test"] for b in bridge]
matches = [b["direction_match"] for b in bridge]
colors_match = ["#2CA02C" if m == "MATCH" else "#D62728" for m in matches]

y_pos = np.arange(len(test_names))
ax.barh(y_pos, [1]*len(test_names), color=colors_match, edgecolor="white", linewidth=0.8, height=0.6)
ax.set_yticks(y_pos)
ax.set_yticklabels(test_names, fontsize=8)
ax.set_xlim(0, 1.3)
ax.set_xticks([])
for i, m in enumerate(matches):
    ax.text(1.05, i, m, va="center", fontsize=9, fontweight="bold", color=colors_match[i])
ax.set_title(f"Bridge Score: {data['bridge_analysis']['summary']['matches']}/{data['bridge_analysis']['summary']['total_comparisons']}")
ax.invert_yaxis()
ax.grid(False)

# --- Bottom-right: Mechanism map ---
ax = axes[1, 1]
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.axis("off")
ax.set_title("FEEL \u2192 SPICE Mechanism Map")

mappings = [
    ("ThermalSoftmax", "BJT softmax"),
    ("BodyGatedLoRA", "Dual NS-RAM + crossbar"),
    ("MetabolicController", "RC filters (3 timescale)"),
    ("Pulse field", "RC integrator"),
    ("Kill-shot (T7)", "Gate clamp \u2192 0.5"),
]

y_start = 8.5
for i, (feel, spice) in enumerate(mappings):
    y = y_start - i * 1.6
    # FEEL box
    ax.text(1.5, y, feel, ha="center", va="center", fontsize=9, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#D4E6F1", edgecolor="#2C3E50"))
    # Arrow
    ax.annotate("", xy=(4.2, y), xytext=(3.0, y),
                arrowprops=dict(arrowstyle="->", color="#2C3E50", lw=1.5))
    # SPICE box
    ax.text(6.5, y, spice, ha="center", va="center", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#FADBD8", edgecolor="#922B21"))

# Column headers
ax.text(1.5, 9.5, "FEEL (Python)", ha="center", va="center", fontsize=10,
        fontweight="bold", color="#2C3E50")
ax.text(6.5, 9.5, "SPICE (Circuit)", ha="center", va="center", fontsize=10,
        fontweight="bold", color="#922B21")

fig2.tight_layout()
out2 = OUT_DIR / "fig_feel_reaches_down_v7.png"
fig2.savefig(out2)
plt.close(fig2)
print(f"Saved: {out2}")

print("Done.")
