"""Visualization of z233 seq-MNIST 28x28 result.

Two-panel: (A) per-seed paired accuracy reservoir vs projection,
(B) boxplot of Δ with CI annotation.

Output: figures/z233_seq_mnist28/{cross_task_negative.pdf,.png}
"""
from __future__ import annotations
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
SEED_DIR = ROOT / "results/z233_seq_mnist28"
SUMMARY = SEED_DIR / "summary.json"
OUT = ROOT / "figures/z233_seq_mnist28"; OUT.mkdir(parents=True, exist_ok=True)

# Load
seeds = []
for p in sorted(SEED_DIR.glob("seed*.json")):
    if "CRASH" in p.name: continue
    seeds.append(json.loads(p.read_text()))
seeds.sort(key=lambda r: r["seed"])
ids = np.array([r["seed"] for r in seeds])
res = np.array([r["test_acc"] for r in seeds])
proj = np.array([r["proj_acc"] for r in seeds])
delta_pp = (res - proj) * 100
S = json.loads(SUMMARY.read_text())

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.2),
                                 gridspec_kw={"width_ratios": [1.7, 1]})

x = np.arange(len(seeds))
w = 0.4
ax1.bar(x - w/2, res*100, w, label="Reservoir (frozen NARMA-10 config)",
         color="#d62728", edgecolor="black", lw=0.6)
ax1.bar(x + w/2, proj*100, w, label="Pure projection baseline",
         color="#1f77b4", edgecolor="black", lw=0.6)
ax1.set_xticks(x); ax1.set_xticklabels([str(i) for i in ids], fontsize=7)
ax1.set_xlabel("Seed (paired across both conditions)")
ax1.set_ylabel("Test accuracy (%)")
ax1.set_ylim(30, 50)
ax1.axhline(10, color="gray", ls=":", lw=1, label="Chance (10 classes)")
ax1.set_title(f"Panel A — Per-seed paired accuracy (n={len(seeds)} seeds)",
                fontsize=11, weight="bold")
ax1.legend(loc="upper right", fontsize=9, framealpha=0.9)
ax1.grid(alpha=0.25, axis="y")

# Panel B: boxplot of delta
bp = ax2.boxplot([delta_pp], widths=0.5, patch_artist=True, showmeans=True,
                   meanprops=dict(marker="D", markerfacecolor="white",
                                    markeredgecolor="black", markersize=8))
bp["boxes"][0].set(facecolor="#fee", edgecolor="#b22", linewidth=1.5)
bp["medians"][0].set(color="#b22", linewidth=2)

# CI bracket
ci_lo, ci_hi = S["ci95_pp"]
ax2.errorbar([1.3], [S["delta_median_pp"]],
              yerr=[[S["delta_median_pp"] - ci_lo], [ci_hi - S["delta_median_pp"]]],
              fmt="none", ecolor="#b22", lw=2.5, capsize=14, capthick=2)
ax2.text(1.45, S["delta_median_pp"], f"95% CI\n[{ci_lo:.2f}, {ci_hi:.2f}] pp",
          fontsize=9, color="#b22", va="center")

ax2.axhline(0, color="black", lw=1)
ax2.axhline(3, color="green", lw=1, ls="--", alpha=0.6,
             label="O35 acceptance (≥+3 pp, CI excl. 0)")
ax2.axhspan(0, 3, color="green", alpha=0.07)
ax2.scatter(np.full(len(delta_pp), 1) + np.random.uniform(-0.05, 0.05, len(delta_pp)),
             delta_pp, s=20, c="#b22", alpha=0.6, edgecolor="black", lw=0.4, zorder=3)

ax2.set_xticks([1, 1.3]); ax2.set_xticklabels(["Δ per seed", ""])
ax2.set_ylabel("Δ accuracy  (reservoir − projection)  [pp]")
ax2.set_ylim(-8, 5)
ax2.set_title(f"Panel B — Cross-task negative\n"
              f"Δ = {S['delta_mean_pp']:+.2f} pp  (paired t = {S['paired_t']:+.2f}, "
              f"p = {S['p_value']:.1e})",
              fontsize=11, weight="bold")
ax2.legend(loc="lower right", fontsize=9, framealpha=0.9)
ax2.grid(alpha=0.25, axis="y")

plt.suptitle(
    "z233 — 28×28 sequential MNIST at FROZEN NARMA-10 hyperparameters\n"
    "Reservoir is significantly worse than pure projection ⇒ NARMA config does NOT generalize",
    fontsize=11.5, weight="bold")
plt.tight_layout(rect=[0, 0, 1, 0.93])

png = OUT / "cross_task_negative.png"
pdf = OUT / "cross_task_negative.pdf"
plt.savefig(pdf, bbox_inches="tight")
plt.savefig(png, bbox_inches="tight", dpi=150)
plt.close()
print(f"saved {png} ({png.stat().st_size // 1024} KB)")
print(f"saved {pdf} ({pdf.stat().st_size // 1024} KB)")
