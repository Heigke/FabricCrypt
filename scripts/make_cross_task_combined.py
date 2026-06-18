"""Combined cross-task figure: z233 frozen NEGATIVE → z235 retuned POSITIVE.

Tells the full Mario v2 narrative in one figure:
  Panel A: z233 frozen config (n=27) Δ=-4.7pp negative
  Panel B: z235 retuned config (n=25) Δ=+5.1pp positive
  Shared y-axis for direct visual comparison.

Output: figures/cross_task_combined/cross_task_before_after.{pdf,png}
"""
from __future__ import annotations
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "figures/cross_task_combined"; OUT.mkdir(parents=True, exist_ok=True)


def load_seeds(d):
    seeds = []
    for p in sorted(Path(d).glob("seed*.json")):
        if "CRASH" in p.name: continue
        seeds.append(json.loads(p.read_text()))
    seeds.sort(key=lambda r: r["seed"])
    return seeds


z233_seeds = load_seeds(ROOT / "results/z233_seq_mnist28")
z235_seeds = load_seeds(ROOT / "results/z235_strong_input_30seed")
z233_S = json.loads((ROOT / "results/z233_seq_mnist28/summary.json").read_text())
z235_S = json.loads((ROOT / "results/z235_strong_input_30seed/summary.json").read_text())

z233_d = np.array([r["delta_pp"] for r in z233_seeds])
z235_d = np.array([r["delta_pp"] for r in z235_seeds])

fig, (axL, axR) = plt.subplots(1, 2, figsize=(13, 5.6), sharey=True)

# Panel A: frozen NEGATIVE
bp1 = axL.boxplot([z233_d], widths=0.5, patch_artist=True, showmeans=True,
                    meanprops=dict(marker="D", markerfacecolor="white",
                                     markeredgecolor="black", markersize=8))
bp1["boxes"][0].set(facecolor="#fee", edgecolor="#b22", linewidth=1.5)
bp1["medians"][0].set(color="#b22", linewidth=2)
ci_lo, ci_hi = z233_S["ci95_pp"]
axL.errorbar([1.35], [z233_S["delta_median_pp"]],
                yerr=[[z233_S["delta_median_pp"] - ci_lo],
                       [ci_hi - z233_S["delta_median_pp"]]],
                fmt="none", ecolor="#b22", lw=2.5, capsize=14, capthick=2)
axL.text(1.5, z233_S["delta_median_pp"],
            f"95% CI\n[{ci_lo:.2f}, {ci_hi:.2f}] pp",
            fontsize=9, color="#b22", va="center")
rng = np.random.default_rng(0)
axL.scatter(np.full(len(z233_d), 1) + rng.uniform(-0.05, 0.05, len(z233_d)),
              z233_d, s=20, c="#b22", alpha=0.6, edgecolor="black", lw=0.4, zorder=3)

# Panel B: retuned POSITIVE
bp2 = axR.boxplot([z235_d], widths=0.5, patch_artist=True, showmeans=True,
                    meanprops=dict(marker="D", markerfacecolor="white",
                                     markeredgecolor="black", markersize=8))
bp2["boxes"][0].set(facecolor="#efe", edgecolor="#2ca02c", linewidth=1.5)
bp2["medians"][0].set(color="#2ca02c", linewidth=2)
ci_lo2, ci_hi2 = z235_S["ci95_pp_median"]
axR.errorbar([1.35], [z235_S["delta_median_pp"]],
                yerr=[[z235_S["delta_median_pp"] - ci_lo2],
                       [ci_hi2 - z235_S["delta_median_pp"]]],
                fmt="none", ecolor="#2ca02c", lw=2.5, capsize=14, capthick=2)
axR.text(1.5, z235_S["delta_median_pp"],
            f"95% CI\n[+{ci_lo2:.2f}, +{ci_hi2:.2f}] pp",
            fontsize=9, color="#2ca02c", va="center")
axR.scatter(np.full(len(z235_d), 1) + rng.uniform(-0.05, 0.05, len(z235_d)),
              z235_d, s=20, c="#2ca02c", alpha=0.6, edgecolor="black", lw=0.4, zorder=3)

# Shared zero line + acceptance bands
for ax, side in [(axL, "L"), (axR, "R")]:
    ax.axhline(0, color="black", lw=1)
    ax.axhspan(0, 5.99, color="#1f77b4", alpha=0.06, zorder=0)
    ax.axhspan(6, 100, color="#2ca02c", alpha=0.08, zorder=0)
    ax.set_xticks([1, 1.35]); ax.set_xticklabels(["Δ per seed", ""])
    ax.set_xlim(0.5, 2.0)
    ax.set_ylim(-8, 13)
    ax.grid(alpha=0.25, axis="y")

axL.set_ylabel("Δ accuracy  (reservoir − pure-projection)  [pp]")

axL.set_title(f"z233 — FROZEN NARMA-10 config\n"
                f"Δ = {z233_S['delta_mean_pp']:+.2f} pp  (n={z233_S['n_seeds']}, "
                f"p = {z233_S['p_value']:.1e})\n"
                f"Cross-task FAILS at frozen hyperparameters",
                fontsize=11, weight="bold", color="#b22")
axR.set_title(f"z235 — RETUNED single knob (g_VG2: 0.05 → 0.20)\n"
                f"Δ = {z235_S['delta_mean_pp']:+.2f} pp  (n={z235_S['n_seeds']}, "
                f"p = {z235_S['p_value']:.1e}, "
                f"{z235_S['n_positive']}/{z235_S['n_seeds']} pos)\n"
                f"Small but extremely robust positive",
                fontsize=11, weight="bold", color="#2ca02c")

# Annotation arrow
fig.text(0.5, 0.5, "→",
            ha="center", va="center", fontsize=42, color="#888")
fig.text(0.5, 0.43, "single hyperparameter\nretune (input gain)",
            ha="center", va="center", fontsize=9, color="#444",
            style="italic")

plt.suptitle(
    "Cross-task generalization on 28×28 sequential MNIST\n"
    "From −4.7 pp (frozen) to +5.1 pp (single-knob retune): direction-robust, magnitude small",
    fontsize=12, weight="bold")
plt.tight_layout(rect=[0, 0, 1, 0.91])

png = OUT / "cross_task_before_after.png"
pdf = OUT / "cross_task_before_after.pdf"
plt.savefig(pdf, bbox_inches="tight")
plt.savefig(png, bbox_inches="tight", dpi=150)
plt.close()
print(f"saved {png} ({png.stat().st_size // 1024} KB)")
print(f"saved {pdf} ({pdf.stat().st_size // 1024} KB)")
