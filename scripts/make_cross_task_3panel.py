"""3-panel cross-task figure for Mario v2 (post-z236).

Replaces 2-panel before-after with 3-experiment honest narrative:
  Panel A: z233 frozen on seq-MNIST       (RED, FAILS)
  Panel B: z235 retuned on seq-MNIST      (GREEN, RECOVERS)
  Panel C: z236 SAME retune on FashionMNIST (RED, FAILS)
Shared y-axis. Annotation: "Same hyperparameters" between B and C.

Output: figures/cross_task_3panel/cross_task_3panel.{pdf,png}
"""
from __future__ import annotations
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "figures/cross_task_3panel"; OUT.mkdir(parents=True, exist_ok=True)


def load_seeds(d):
    return [json.loads(p.read_text()) for p in sorted(Path(d).glob("seed*.json"))
              if "CRASH" not in p.name]


z233 = load_seeds(ROOT / "results/z233_seq_mnist28")
z235 = load_seeds(ROOT / "results/z235_strong_input_30seed")
z236 = load_seeds(ROOT / "results/z236_fashion_mnist")
S233 = json.loads((ROOT / "results/z233_seq_mnist28/summary.json").read_text())
S235 = json.loads((ROOT / "results/z235_strong_input_30seed/summary.json").read_text())
S236 = json.loads((ROOT / "results/z236_fashion_mnist/summary.json").read_text())

d233 = np.array([r["delta_pp"] for r in z233])
d235 = np.array([r["delta_pp"] for r in z235])
d236 = np.array([r["delta_pp"] for r in z236])

fig, axes = plt.subplots(1, 3, figsize=(16, 5.6), sharey=True)

panels = [
    (axes[0], d233, S233, "z233", "seq-MNIST FROZEN config\n(g_VG2=0.05)",
     "#b22", "FAILS — frozen NARMA params don't transfer"),
    (axes[1], d235, S235, "z235", "seq-MNIST RETUNED\n(g_VG2: 0.05 → 0.20)",
     "#2ca02c", "RECOVERS — small but robust positive"),
    (axes[2], d236, S236, "z236", "FashionMNIST SAME retune\n(g_VG2 = 0.20)",
     "#b22", "FAILS — retune doesn't generalize across tasks"),
]

rng = np.random.default_rng(0)
for ax, deltas, S, lab, cfg_text, color, verdict in panels:
    facecol = "#fee" if color == "#b22" else "#efe"
    bp = ax.boxplot([deltas], widths=0.5, patch_artist=True, showmeans=True,
                      meanprops=dict(marker="D", markerfacecolor="white",
                                       markeredgecolor="black", markersize=8))
    bp["boxes"][0].set(facecolor=facecol, edgecolor=color, linewidth=1.5)
    bp["medians"][0].set(color=color, linewidth=2)

    # CI bracket — handle both summary key formats
    ci_key = "ci95_pp" if "ci95_pp" in S else "ci95_pp_median"
    ci_lo, ci_hi = S[ci_key]
    median_pp = S.get("delta_median_pp", S.get("delta_mean_pp"))
    ax.errorbar([1.35], [median_pp],
                  yerr=[[median_pp - ci_lo], [ci_hi - median_pp]],
                  fmt="none", ecolor=color, lw=2.5, capsize=14, capthick=2)
    sign = "+" if ci_lo >= 0 else ""
    ax.text(1.5, median_pp,
              f"95% CI\n[{ci_lo:+.2f}, {ci_hi:+.2f}] pp",
              fontsize=9, color=color, va="center")

    ax.scatter(np.full(len(deltas), 1) + rng.uniform(-0.05, 0.05, len(deltas)),
                 deltas, s=20, c=color, alpha=0.6, edgecolor="black", lw=0.4, zorder=3)
    ax.axhline(0, color="black", lw=1)
    ax.axhspan(0, 8, color="#1f77b4", alpha=0.05, zorder=0)
    ax.set_xticks([1, 1.35]); ax.set_xticklabels(["Δ", ""])
    ax.set_xlim(0.5, 2.0)
    ax.set_ylim(-15, 13)
    ax.grid(alpha=0.25, axis="y")

    pval = S.get("p_value", 1.0)
    n = S.get("n_seeds", S.get("N", len(deltas)))
    pos_str = ""
    if "n_positive" in S:
        pos_str = f", {S['n_positive']}/{n} pos"
    ax.set_title(f"{lab} — {cfg_text}\n"
                 f"Δ = {S['delta_mean_pp']:+.2f} pp  (n={len(deltas)}, "
                 f"p = {pval:.1e}{pos_str})\n{verdict}",
                 fontsize=10, weight="bold", color=color)

axes[0].set_ylabel("Δ accuracy  (reservoir − pure-projection)  [pp]")

# Annotation arrows between panels
fig.text(0.355, 0.50, "→", ha="center", va="center", fontsize=42, color="#888")
fig.text(0.355, 0.43, "single hyperparameter\nretune", ha="center", va="center",
            fontsize=9, color="#444", style="italic")
fig.text(0.665, 0.50, "→", ha="center", va="center", fontsize=42, color="#b22",
            weight="bold")
fig.text(0.665, 0.43, "SAME hyperparameters,\ndifferent task",
            ha="center", va="center", fontsize=9, color="#b22",
            style="italic", weight="bold")

plt.suptitle(
    "Cross-task generalization on 28×28 sequential image classification\n"
    "Reservoir helps where linear baseline is weak (MNIST proj=43%); hurts where strong (FMNIST proj=72%)",
    fontsize=12, weight="bold")
plt.tight_layout(rect=[0, 0, 1, 0.91])

png = OUT / "cross_task_3panel.png"
pdf = OUT / "cross_task_3panel.pdf"
plt.savefig(pdf, bbox_inches="tight")
plt.savefig(png, bbox_inches="tight", dpi=150)
plt.close()
print(f"saved {png} ({png.stat().st_size // 1024} KB)")
print(f"saved {pdf} ({pdf.stat().st_size // 1024} KB)")
