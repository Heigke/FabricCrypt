#!/usr/bin/env python3
"""Generate publication-quality figures for FEEL paper from bridge experiment results."""

import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

RESULTS = "/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy/results"
OUTDIR = "/tmp/feel_paper/FEEL__Functionally_Embodied_Emergent_Learning__13_-3/figures"

# ── Figure 1: z2139 Kill-Shots ──────────────────────────────────────────────

with open(f"{RESULTS}/z2139_closed_loop_bridge.json") as f:
    d139 = json.load(f)

normal_ppl = d139["normal_ppl"]
ks = d139["kill_shots"]

labels = ["Normal", "K1\nopen-loop", "K2\nreversed", "K3\nno-avalanche", "K4\nno-MODE"]
ppls = [
    normal_ppl,
    ks["K1_open_loop"]["kill_ppl"],
    ks["K2_reversed"]["kill_ppl"],
    ks["K3_no_avalanche"]["kill_ppl"],
    ks["K4_no_mode"]["kill_ppl"],
]
colors = ["#4C72B0", "#4C72B0", "#4C72B0", "#C44E52", "#4C72B0"]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

# Panel (a) — Kill-shot PPL bar chart
bars = ax1.bar(labels, ppls, color=colors, edgecolor="black", linewidth=0.6)
ax1.axhline(normal_ppl, ls="--", color="grey", lw=0.8, label=f"normal PPL = {normal_ppl:.3f}")
for b, v in zip(bars, ppls):
    ax1.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.3f}",
             ha="center", va="bottom", fontsize=8)
ax1.set_ylabel("Perplexity", fontsize=10)
ax1.set_title("(a) Kill-Shot PPL Ratios", fontsize=10)
ax1.legend(fontsize=8, loc="upper left")
ax1.tick_params(labelsize=9)

# Panel (b) — Training convergence
steps = [20, 40, 60, 80, 100, 120, 140, 160, 180, 200]
train_ppls = [110.69, 99.37, 90.73, 78.95, 68.18, 61.80, 47.37, 34.45, 23.69, 15.64]
baseline_ppl = d139["baseline_ppl"]

ax2.plot(steps, train_ppls, "o-", color="#4C72B0", markersize=4, linewidth=1.5, label="train PPL")
ax2.axhline(baseline_ppl, ls="--", color="#C44E52", lw=0.8, label=f"baseline = {baseline_ppl:.1f}")
ax2.axhline(normal_ppl, ls="--", color="#55A868", lw=0.8, label=f"final eval = {normal_ppl:.3f}")
ax2.set_xlabel("Training Step", fontsize=10)
ax2.set_ylabel("Perplexity", fontsize=10)
ax2.set_title("(b) Closed-Loop Training Convergence", fontsize=10)
ax2.legend(fontsize=8)
ax2.tick_params(labelsize=9)

plt.tight_layout()
fig.savefig(f"{OUTDIR}/fig_z2139_killshots.png", dpi=300, bbox_inches="tight")
plt.close(fig)
print(f"Saved {OUTDIR}/fig_z2139_killshots.png")

# ── Figure 2: z2143 Test Battery ────────────────────────────────────────────

with open(f"{RESULTS}/z2143_bridge_test_battery.json") as f:
    d143 = json.load(f)

tests = d143["tests"]
test_labels = []
metric_texts = []
statuses = []

for t in tests:
    test_labels.append(f"{t['test']}: {t['name']}")
    if t["test"] == "T41":
        metric_texts.append(f"ratio = {t['ratio']:.3f}")
    elif t["test"] == "T42":
        metric_texts.append(f"diff_ratio = {t['diff_ratio']:.1f}")
    elif t["test"] == "T43":
        metric_texts.append(f"LZc ratio = {t['ratio']:.3f}")
    elif t["test"] == "T44":
        metric_texts.append(f"corr = {t['correlation']:.3f}")
    elif t["test"] == "T45":
        metric_texts.append(f"\u03c1 = {t['rho']:.3f}")
    elif t["test"] == "T46":
        metric_texts.append("TE ns")
    statuses.append(t["status"])

bar_colors = ["#55A868" if s == "PASS" else "#C44E52" for s in statuses]
y_pos = np.arange(len(test_labels))

fig, ax = plt.subplots(figsize=(8, 4))
bars = ax.barh(y_pos, [1] * len(test_labels), color=bar_colors, edgecolor="black", linewidth=0.6)
for i, (bar, txt, status) in enumerate(zip(bars, metric_texts, statuses)):
    ax.text(0.5, bar.get_y() + bar.get_height() / 2,
            f"{txt}  [{status}]", ha="center", va="center", fontsize=9,
            fontweight="bold", color="white")
ax.set_yticks(y_pos)
ax.set_yticklabels(test_labels, fontsize=9)
ax.set_xlim(0, 1)
ax.set_xticks([])
ax.set_title("z2143 Bridge Test Battery (4/6 PASS)", fontsize=10)
ax.invert_yaxis()

plt.tight_layout()
fig.savefig(f"{OUTDIR}/fig_z2143_battery.png", dpi=300, bbox_inches="tight")
plt.close(fig)
print(f"Saved {OUTDIR}/fig_z2143_battery.png")

# ── Figure 3: z2141 Criticality Sweep ───────────────────────────────────────

with open(f"{RESULTS}/z2141_criticality_sweep.json") as f:
    d141 = json.load(f)

sweep = d141["sweep"]
vgs = [e["vg"] for e in sweep]
spike_rates = [e["mean_spike_rate"] for e in sweep]
branching = [e["branching_ratio"] for e in sweep]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

# Panel (a) — Spike rate (log scale)
ax1.semilogy(vgs, spike_rates, "o-", color="#4C72B0", markersize=4, linewidth=1.5)
ax1.set_xlabel("$V_g$", fontsize=10)
ax1.set_ylabel("Mean Spike Rate (log)", fontsize=10)
ax1.set_title(r"(a) Spike Rate vs $V_g$", fontsize=10)
ax1.tick_params(labelsize=9)
ax1.grid(True, alpha=0.3)

# Panel (b) — Branching ratio
ax2.plot(vgs, branching, "o-", color="#DD8452", markersize=4, linewidth=1.5)
ax2.axhline(1.0, ls="--", color="grey", lw=0.8, label=r"$\sigma = 1.0$")
ax2.set_xlabel("$V_g$", fontsize=10)
ax2.set_ylabel(r"Branching Ratio $\sigma$", fontsize=10)
ax2.set_title(r"(b) Branching Ratio $\sigma$", fontsize=10)
ax2.legend(fontsize=8)
ax2.tick_params(labelsize=9)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
fig.savefig(f"{OUTDIR}/fig_z2141_criticality.png", dpi=300, bbox_inches="tight")
plt.close(fig)
print(f"Saved {OUTDIR}/fig_z2141_criticality.png")

print("\nAll 3 figures generated successfully.")
