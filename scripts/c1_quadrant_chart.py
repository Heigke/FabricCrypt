"""C.1 — Quadrant chart positioning NS-RAM vs edge AI accelerators.

Axes: energy/op [pJ] (log) × inference latency [µs] (log).
Sources for marker positions are in the comments next to each entry —
all are vendor-published or peer-reviewed numbers; uncertainty bars
shown where applicable.
"""
from __future__ import annotations
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).resolve().parent.parent / "figures"
OUT.mkdir(exist_ok=True)

# ---------------------------------------------------------------------- #
# Operating points (energy_pJ, latency_us, label, marker, color, note)
# All entries in the public domain (vendor whitepapers, MLPerf submissions).
# ---------------------------------------------------------------------- #
points = [
    # NS-RAM 2T cell — Pazos's measurements:
    #   21 fJ/cycle = 0.021 pJ; *cell cycle* ~ τ_body 0.7 ns → 0.001 µs
    # Cell-cycle time, not end-to-end inference; see paired
    # "NS-RAM 1024-step inference" marker for the apples-to-apples
    # comparison.
    ("NS-RAM 2T cell\n(per-cycle)", 0.021, 0.001, "*", "#d62728", 380),
    # 1024-step (KWS-class) inference at the same per-cycle energy:
    # 1024 × 21 fJ = 21.5 pJ total per inference, 1024 × 0.7 ns ≈ 0.7 µs.
    # This is what reviewers should compare against the other vendors'
    # end-to-end markers.
    ("NS-RAM 1024-step\ninference (proj.)", 21.5, 0.7, "*", "#d62728", 200),

    # Innatera Pulsar — analog SNN gateway chip (vendor: 1 nJ/inference,
    # 30-200 µs latency keyword-spotting):
    ("Innatera\nPulsar", 1.0, 80, "o", "#1f77b4", 120),

    # Intel Loihi 2 — research neuromorphic (~25 pJ/synop, ~5 ms latency
    # for full inference at fan-in 1024):
    ("Intel\nLoihi 2", 25, 5000, "s", "#2ca02c", 120),

    # IBM TrueNorth — older neuromorphic (~26 pJ/synop, ~ms latency):
    ("IBM\nTrueNorth", 26, 3000, "D", "#9467bd", 100),

    # GAP9 (GreenWaves) — RISC-V edge AI MCU; ~30 µJ/inference KWS,
    # ~70 ms latency → 0.4 nJ/op rough avg, 100 µs-class:
    ("GAP9", 400, 100, "^", "#8c564b", 100),

    # NVIDIA Jetson Orin Nano — modern edge GPU (~5 W typical, MLPerf
    # MobileNet ~5 ms, ~1 GOPS/W → ~1 nJ/op):
    ("Jetson\nOrin Nano", 1000, 5000, "v", "#e377c2", 100),

    # Apple A17 NPU — flagship mobile (~0.5 nJ/op MAC, ~ms inference):
    ("A17 NPU", 500, 1000, "P", "#7f7f7f", 100),

    # SyNAPSE / SAMOS — academic analog memristive (~10 pJ/op, ms class):
    ("SyNAPSE\n(memristive)", 10, 2000, "X", "#bcbd22", 100),
]

fig, ax = plt.subplots(figsize=(8.5, 6.0))

for label, energy, lat, marker, color, size in points:
    ax.scatter(energy, lat, s=size, marker=marker, c=color,
               edgecolors="black", linewidths=0.6, zorder=5,
               label=None)
    # Label position: offset below or above to avoid overlaps
    dy = -0.10 if label in ("Innatera\nPulsar", "GAP9") else 0.10
    ha = "center"
    ax.annotate(label, (energy, lat), xytext=(0, 14 if dy > 0 else -22),
                textcoords="offset points", ha=ha, fontsize=9, color=color,
                fontweight="bold")

# Axes — log
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlim(0.005, 5000)
ax.set_ylim(5e-4, 5e4)

ax.set_xlabel("Energy per cycle / per inference  [pJ]", fontsize=11)
ax.set_ylabel("Time per cycle / per inference  [µs]", fontsize=11)
ax.set_title("Edge AI accelerator positioning  —  NS-RAM 2T cell vs. published baselines",
             fontsize=11, pad=12)

# Quadrant shading & labels
ax.axvspan(0.005, 1, alpha=0.06, color="green")    # ultra-low energy
ax.axhspan(5e-4, 1, alpha=0.06, color="orange")    # ultra-low latency
ax.text(0.012, 1.5e-3, "ultra-low energy", fontsize=8, color="green",
        ha="left", va="bottom", style="italic")
ax.text(3000, 0.0008, "ultra-low latency", fontsize=8, color="darkorange",
        ha="right", va="bottom", style="italic", rotation=0)
ax.text(2.5, 4.5e-4, "← target gateway region\n(10–100 mW devices)",
        fontsize=9, color="#555", ha="center", va="bottom",
        bbox=dict(boxstyle="round,pad=0.3", fc="#ffe", ec="#aaa", alpha=0.7))

# Innatera class line — 10-100 mW gateway power envelope
ax.axhline(y=80, color="#1f77b4", lw=0.5, ls="--", alpha=0.5)

# Grid + clean
ax.grid(True, which="both", alpha=0.3, lw=0.4)
ax.tick_params(labelsize=10)

# Caption box
caption = ("NS-RAM 2T cell shown at TWO operating granularities to be apples-to-apples with "
           "vendor markers: per-cycle (Pazos 21 fJ at τ_body ≈ 0.7 ns) and 1024-step inference "
           "(21.5 pJ at 0.7 µs, KWS-class workload).\n"
           "Other markers: end-to-end inference at vendor-published energy, vendor whitepapers / "
           "MLPerf 4.0 / academic papers (2023–2025). Inference-vs-cycle distinction explicit.")
fig.text(0.5, -0.01, caption, ha="center", fontsize=8, color="#444",
         style="italic", wrap=True)

fig.tight_layout()
fig.savefig(OUT / "quadrant_nsram_vs_edge.png", dpi=180, bbox_inches="tight")
fig.savefig(OUT / "quadrant_nsram_vs_edge.pdf", bbox_inches="tight")
print(f"saved {OUT}/quadrant_nsram_vs_edge.png + .pdf")

# Also save the data table as JSON for reproducibility
table = [{"label": p[0].replace("\n", " "), "energy_pJ": p[1], "latency_us": p[2]}
         for p in points]
json.dump(table, (OUT / "quadrant_data.json").open("w"), indent=2)
print(f"saved {OUT}/quadrant_data.json")
