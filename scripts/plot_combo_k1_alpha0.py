#!/usr/bin/env python3
"""Plot K1+ALPHA0 combo ablation results."""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path("/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy")
OUT = ROOT / "results/track_combo_k1_alpha0"
d = json.load(open(OUT / "ablation.json"))

labels, med, vg2, vg4, vg6, ratios = [], [], [], [], [], []
for k, s in d.items():
    if "error" in s:
        continue
    k1 = float(k.split("K1=")[1].split("__")[0])
    a0 = float(k.split("ALPHA0=")[1])
    k1_tag = {0.41825: "K1=baseline (0.418)",
              0.53825: "K1=BSIM card (0.538)",
              0.6459:  "K1=card×1.2 (0.646)"}[k1]
    a0_tag = "ALPHA0=CSV (7.8e-5)" if a0 < 1e-4 else "ALPHA0=card (7.8e-4)"
    labels.append(f"{k1_tag}\n+ {a0_tag}")
    med.append(s["median_dec_all"]["median"])
    vg2.append(s["median_dec_VG1=0.2"]["median"])
    vg4.append(s["median_dec_VG1=0.4"]["median"])
    vg6.append(s["median_dec_VG1=0.6"]["median"])
    ratios.append(s["worst_VG1=0.6"]["Imeas_over_Ipred_med"])

x = np.arange(len(labels))
w = 0.20
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Left: median dec per VG1 branch
ax1.bar(x - 1.5*w, med, w, label="all 33 biases", color="#222")
ax1.bar(x - 0.5*w, vg2, w, label="VG1=0.2", color="#4caf50")
ax1.bar(x + 0.5*w, vg4, w, label="VG1=0.4", color="#ff9800")
ax1.bar(x + 1.5*w, vg6, w, label="VG1=0.6", color="#e53935")
ax1.axhline(0.5, color="blue", ls="--", lw=1.2, label="0.5 dec target")
ax1.axhline(1.163, color="grey", ls=":", lw=1, label="baseline (1.163)")
ax1.set_xticks(x)
ax1.set_xticklabels(labels, fontsize=8, rotation=0)
ax1.set_ylabel("median log10-RMSE (dec)")
ax1.set_title("K1 + ALPHA0 combined fix — DC fit quality (lower = better)")
ax1.legend(loc="upper right", fontsize=9)
ax1.grid(alpha=0.3)
for i, m in enumerate(med):
    ax1.text(i - 1.5*w, m + 0.02, f"{m:.3f}", ha="center", fontsize=8, fontweight="bold")

# Right: Imeas/Ipred at VG1=0.6
colors = ["#888", "#666", "#ff9800", "#4caf50", "#1e88e5", "#3949ab"][:len(ratios)]
bars = ax2.bar(x, ratios, color=colors)
ax2.axhline(1.0, color="green", ls="--", lw=1.2, label="perfect (1×)")
ax2.set_yscale("log")
ax2.set_xticks(x)
ax2.set_xticklabels(labels, fontsize=8)
ax2.set_ylabel("Imeas / Ipred (median at VG1=0.6)")
ax2.set_title("Triode-regime current shortfall (lower = closer to data)")
ax2.legend()
ax2.grid(alpha=0.3, which="both")
for i, r in enumerate(ratios):
    ax2.text(i, r * 1.15, f"{r:.1f}×", ha="center", fontsize=9, fontweight="bold")

plt.suptitle(f"NS-RAM 2T cell — K1 + ALPHA0 ablation on Sebas 33-bias (n=66 fwd+bwd)\n"
             f"Both fixes are BSIM card values, not parameter tuning",
             fontsize=11)
plt.tight_layout()
out = OUT / "plot_combo.png"
plt.savefig(out, dpi=120, bbox_inches="tight")
print(out)
