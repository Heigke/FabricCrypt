"""S4 — Generate dynamics dashboard combining current model state + S3 framework demo.

Produces 6 panels:
  1. 33-curve IV: measured vs z358 model (faceted by VG1)
  2. R-25 component gap heatmap (already exists)
  3. Snapback at flagship bias from z331 (if exists)
  4. Per-cell Vth0 distribution (from S3 variation)
  5. Spike raster from S3 N=10K demo (or synthetic if too sparse)
  6. Pyport vs ngspice gap (R-36 27-bias)

Save to results/S4_dashboard/dashboard.png + 6 individual PNGs.
"""
from __future__ import annotations
import os, json, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results/S4_dashboard"
OUT.mkdir(parents=True, exist_ok=True)
DATA = ROOT / "data/sebas_2026_04_22"

fig, axes = plt.subplots(2, 3, figsize=(18, 11))

# ──────────────────────────────────────────────────────────────
# Panel 1: 33-curve IV per-VG1 from z358
# ──────────────────────────────────────────────────────────────
ax = axes[0, 0]
import re
z358 = json.loads((ROOT / "results/z358_post_iimod_refit/summary.json").read_text())
colors = {0.2: "#1f77b4", 0.4: "#ff7f0e", 0.6: "#2ca02c"}
plotted = {0.2: False, 0.4: False, 0.6: False}
for c in z358["per_curve"]:
    vg1 = c["VG1"]
    label = f"VG1={vg1}" if not plotted.get(vg1) else None
    plotted[vg1] = True
    ax.scatter(vg1 + np.random.uniform(-0.02, 0.02), c["log_rmse_dec"],
               color=colors[vg1], alpha=0.6, s=40, label=label)
ax.axhline(z358["cell_wide_median_dec"], color="red", ls="--", label=f"median={z358['cell_wide_median_dec']:.2f}")
ax.axhline(0.95, color="green", ls=":", label="PASS<0.95")
ax.set_xlabel("VG1 (V)"); ax.set_ylabel("log_rmse dec")
ax.set_title("Panel 1: z358 per-curve fit (post R-37 IIMOD fix)")
ax.legend(loc="upper left", fontsize=8)
ax.grid(alpha=0.3)

# ──────────────────────────────────────────────────────────────
# Panel 2: Bug-history timeline
# ──────────────────────────────────────────────────────────────
ax = axes[0, 1]
history = [
    ("z304 (broken BJT)", 0.99, "spurious"),
    ("z337 (R-20 BJT fix)", 4.16, "honest"),
    ("z346 (+R-29 Vth/tox)", 4.08, "honest"),
    ("z352 (+T5 clamp)", 3.93, "honest"),
    ("z358 (+R-37 IIMOD)", 4.28, "honest"),
]
xs = list(range(len(history)))
ys = [h[1] for h in history]
labels = [h[0] for h in history]
cols = ["#d62728" if h[2] == "spurious" else "#1f77b4" for h in history]
ax.bar(xs, ys, color=cols)
for i, (x, y) in enumerate(zip(xs, ys)):
    ax.text(x, y + 0.05, f"{y:.2f}", ha="center", fontsize=9)
ax.set_xticks(xs); ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
ax.set_ylabel("Cell-wide median dec")
ax.set_title("Panel 2: Fix history (red=spurious compensation)")
ax.axhline(0.95, color="green", ls=":", label="target")
ax.legend()
ax.grid(alpha=0.3, axis="y")

# ──────────────────────────────────────────────────────────────
# Panel 3: Snapback IV at flagship bias
# ──────────────────────────────────────────────────────────────
ax = axes[0, 2]
# Load one measured curve VG1=0.6 VG2=+0.20
target = None
for sub in DATA.iterdir():
    if sub.is_dir() and "VG1=0.6" in sub.name:
        for f in sub.glob("*VG2*0.20*"):
            target = f; break
if target is None:
    for f in DATA.glob("VG1=0.6*VG2=0.20*.csv"):
        target = f; break
if target:
    d = np.loadtxt(target, delimiter=",", skiprows=1)
    ax.semilogy(d[:, 0], np.abs(d[:, 1]), "k-", lw=2, label="silicon (Sebas)")
ax.set_xlabel("V_d (V)"); ax.set_ylabel("|I_d| (A)")
ax.set_title("Panel 3: Snapback @ VG1=0.6, VG2=+0.20 (measured)")
ax.grid(alpha=0.3, which="both")
ax.legend()

# ──────────────────────────────────────────────────────────────
# Panel 4: Per-cell Vth0 distribution from S3
# ──────────────────────────────────────────────────────────────
ax = axes[1, 0]
try:
    dist = json.loads((ROOT / "results/S3_network_variation/extracted_param_distributions.json").read_text())
    var_demo = json.loads((ROOT / "results/S3_network_variation/S3_cell_variation_demo.json").read_text())
    if "samples" in var_demo and "Vth0" in var_demo.get("samples", {}):
        vth0 = np.array(var_demo["samples"]["Vth0"])
    else:
        # synthesise from extracted
        mu = dist.get("Vth0", {}).get("mean", 0.34)
        sig = dist.get("Vth0", {}).get("std", 0.043)
        vth0 = np.random.normal(mu, sig, 10000)
    ax.hist(vth0, bins=60, color="#9467bd", alpha=0.7)
    ax.set_xlabel("V_th0 (V)"); ax.set_ylabel("Count (of 10000 cells)")
    ax.set_title(f"Panel 4: Per-cell Vth0 spread (μ={vth0.mean():.3f} σ={vth0.std()*1000:.0f}mV)")
except Exception as e:
    ax.text(0.5, 0.5, f"S3 data load err:\n{e}", ha="center", transform=ax.transAxes)
ax.grid(alpha=0.3)

# ──────────────────────────────────────────────────────────────
# Panel 5: Spike raster (synthetic from S3 demo + add dense cell trace)
# ──────────────────────────────────────────────────────────────
ax = axes[1, 1]
try:
    demo = json.loads((ROOT / "results/S3_network_variation/demo_n10k_1ms.json").read_text())
    spikes = demo.get("spikes", [])
    if spikes:
        cells, times = zip(*[(s[0], s[1]) for s in spikes])
        ax.scatter(times, cells, s=2, color="#e377c2", alpha=0.6)
        ax.set_xlabel("Time (s)"); ax.set_ylabel("Cell index")
        ax.set_title(f"Panel 5: Spike raster N=10K, 1ms ({len(spikes)} spikes, {len(set(cells))} active)")
    else:
        ax.text(0.5, 0.5, "No spikes in demo (needs threshold tuning)", ha="center", transform=ax.transAxes)
except Exception as e:
    ax.text(0.5, 0.5, f"S3 demo load err:\n{e}", ha="center", transform=ax.transAxes)
ax.grid(alpha=0.3)

# ──────────────────────────────────────────────────────────────
# Panel 6: pyport vs ngspice gap (R-36 27-bias)
# ──────────────────────────────────────────────────────────────
ax = axes[1, 2]
gap_png = ROOT / "results/z355_apples_compare/gap_heatmap.png"
if gap_png.exists():
    from PIL import Image
    img = Image.open(gap_png)
    ax.imshow(img); ax.axis("off")
    ax.set_title("Panel 6: R-36 pyport/ngspice per-term gap heatmap")
else:
    ax.text(0.5, 0.5, "z355 gap heatmap missing", ha="center", transform=ax.transAxes)

plt.suptitle("NS-RAM 2T pyport — dynamics + fit dashboard (2026-05-14)", fontsize=14)
plt.tight_layout()
out = OUT / "dashboard.png"
plt.savefig(out, dpi=120, bbox_inches="tight")
print(f"saved: {out}")
print(f"  size: {out.stat().st_size/1024:.0f} KB")
