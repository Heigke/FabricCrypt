"""Δ-vs-projection-baseline scatter — visualizes the post-z237 claim:
reservoir contribution decreases monotonically with linear baseline strength.

3 datapoints (MNIST z235, KMNIST z237, FashionMNIST z236) + linear fit
with extrapolation band + crossover-zero annotation.

Output: figures/monotonic_baseline/delta_vs_baseline.{pdf,png}
"""
from __future__ import annotations
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "figures/monotonic_baseline"; OUT.mkdir(parents=True, exist_ok=True)

S235 = json.loads((ROOT / "results/z235_strong_input_30seed/summary.json").read_text())
S236 = json.loads((ROOT / "results/z236_fashion_mnist/summary.json").read_text())
S237 = json.loads((ROOT / "results/z237_kmnist/summary.json").read_text())
S238 = json.loads((ROOT / "results/z238_fmnist_smalltrain/summary.json").read_text())
S240 = json.loads((ROOT / "results/z240_cifar_local/summary.json").read_text())

points = [
    ("CIFAR-10\n(grayscale)", S240["proj_mean"]*100, S240["delta_mean_pp"],  S240["ci95_pp_median"], S240["n_seeds"]),
    ("MNIST",            S235["proj_mean"]*100, S235["delta_mean_pp"],  S235["ci95_pp_median"], S235["n_seeds"]),
    ("KMNIST",           S237["proj_mean"]*100, S237["delta_mean_pp"],  S237["ci95_pp_median"], S237["n_seeds"]),
    ("FMNIST\n(train=200)", S238["proj_mean"]*100, S238["delta_mean_pp"], S238["ci95_pp_median"], S238["n_seeds"]),
    ("FashionMNIST",     S236["proj_mean"]*100, S236["delta_mean_pp"],  S236["ci95_pp_median"], S236["n_seeds"]),
]

x = np.array([p[1] for p in points])
y = np.array([p[2] for p in points])
ci_los = np.array([p[3][0] for p in points])
ci_his = np.array([p[3][1] for p in points])

# Linear fit on MNIST-family ONLY (proj 40-75%); CIFAR is out-of-sample
mask_in_band = (x >= 40) & (x <= 75)
slope, intercept = np.polyfit(x[mask_in_band], y[mask_in_band], 1)
crossover = -intercept / slope  # x where y=0

xfit = np.linspace(10, 80, 300)
yfit = slope * xfit + intercept

fig, ax = plt.subplots(figsize=(9, 6))

# Background band: positive zone (above zero, blue) and negative zone (red)
ax.axhspan(0, 12, color="#2ca02c", alpha=0.06, zorder=0,
             label="Reservoir helps")
ax.axhspan(-15, 0, color="#d62728", alpha=0.06, zorder=0,
             label="Reservoir hurts")
ax.axhline(0, color="black", lw=1, zorder=1)

# Linear fit + CI band (rough — propagate point CIs as fit envelope)
ax.plot(xfit, yfit, color="#1f77b4", lw=2.5, alpha=0.85,
          label=f"Linear fit:  Δ ≈ {intercept:+.1f} {slope:+.2f}·proj%",
          zorder=2)
# Approximate envelope from point CIs
yhi_fit = np.interp(xfit, x, ci_his) if len(x) > 1 else yfit
ylo_fit = np.interp(xfit, x, ci_los) if len(x) > 1 else yfit
ax.fill_between(xfit, ylo_fit, yhi_fit, color="#1f77b4", alpha=0.15, zorder=1)

# Crossover annotation
ax.axvline(crossover, color="black", lw=1, ls=":", alpha=0.7)
ax.text(crossover + 0.5, -13.5, f"zero-crossing\nproj ≈ {crossover:.0f}%",
          fontsize=10, color="black", style="italic")

# Data points with CI bars (CI now explicit per O38 oracle rec)
colors = ["#ff7f0e", "#2ca02c", "#1f77b4", "#9467bd", "#d62728"]
for i, (lab, px, py, ci, n) in enumerate(points):
    ax.errorbar([px], [py], yerr=[[py - ci[0]], [ci[1] - py]],
                  fmt="o", ms=14, color=colors[i],
                  ecolor=colors[i], lw=2.5, capsize=10, capthick=2,
                  markeredgecolor="black", markeredgewidth=1.5, zorder=4)
    # CI numeric annotation next to each point
    ci_text = f"[{ci[0]:+.1f}, {ci[1]:+.1f}]"
    ax.text(px, py - 1.3, ci_text, fontsize=7, color=colors[i],
              ha="center", va="top", style="italic", alpha=0.85)
    # Label offset to avoid overlap
    dx = -2 if px > 65 else 2
    ha = "right" if px > 65 else "left"
    ax.text(px + dx, py, f"  {lab}  \n(n={n})",
              fontsize=11, fontweight="bold", color=colors[i],
              ha=ha, va="center")

ax.set_xlabel("Pure-projection baseline accuracy  [%]", fontsize=12)
ax.set_ylabel("Reservoir contribution  Δ  [pp]", fontsize=12)
ax.set_xlim(10, 80)
# Highlight MNIST-family validity band
ax.axvspan(40, 75, color="#1f77b4", alpha=0.05, zorder=0)
ax.text(57.5, 11.5, "MNIST-family band\n(linear fit valid here)",
          ha="center", va="top", fontsize=9, color="#1f77b4", style="italic")
ax.text(15, 4, "CIFAR-10 lands here\n(linear extrapolation\nfails by ~10×)",
          ha="left", va="center", fontsize=9, color="#ff7f0e", style="italic")
ax.set_ylim(-15, 12)
ax.grid(alpha=0.25)
ax.legend(loc="upper right", fontsize=10, framealpha=0.92)

ax.set_title(
    "Reservoir contribution: linear within MNIST-family band; saturates outside\n"
    "Same NS-RAM hyperparameters (leak=0.30, g_VG2=0.20, N=1000) across 5 image-classification tasks\n"
    "Linear fit valid in 40–75% baseline band; CIFAR-10 (proj=15%) shows direction-match but magnitude saturates",
    fontsize=10.5, weight="bold")

plt.tight_layout()

png = OUT / "delta_vs_baseline.png"
pdf = OUT / "delta_vs_baseline.pdf"
plt.savefig(pdf, bbox_inches="tight")
plt.savefig(png, bbox_inches="tight", dpi=150)
plt.close()
print(f"saved {png} ({png.stat().st_size // 1024} KB)")
print(f"saved {pdf} ({pdf.stat().st_size // 1024} KB)")
print(f"linear fit:  Δ = {intercept:+.2f} {slope:+.3f}·proj%")
print(f"zero-crossing at projection ≈ {crossover:.1f}%")
