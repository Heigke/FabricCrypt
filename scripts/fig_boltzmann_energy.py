#!/usr/bin/env python3
"""
Publication figure: NS-RAM Avalanche Threshold Characterisation
Three-panel figure: (a) Boltzmann step, (b) Energy vs Vg, (c) Phase diagram.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from pathlib import Path

# ── Colours ──────────────────────────────────────────────────────────────────
BLUE   = "#2166ac"
RED    = "#b2182b"
GREEN  = "#1b7837"

# ── Load data ────────────────────────────────────────────────────────────────
base = Path(__file__).resolve().parent.parent / "results"

with open(base / "nsram_boltzmann_sweep.json") as f:
    boltz = json.load(f)

with open(base / "nsram_energy_sweep.json") as f:
    energy = json.load(f)

boltz_data = boltz["data"]
energy_data = energy["sweep"]

# Extract arrays
b_T = np.array([d["T_K"] for d in boltz_data])
b_spikes = np.array([d["spikes"] for d in boltz_data])
b_Vg = np.array([d["Vg_eff"] for d in boltz_data])

e_Vg = np.array([d["Vg_V"] for d in energy_data])
e_spikes = np.array([d["n_LIF_spikes"] for d in energy_data])
e_energy = np.array([d["E_per_burst_pazos_scaled_fJ"] if d["E_per_burst_pazos_scaled_fJ"] is not None else np.nan for d in energy_data])

# ── Figure setup ─────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 8.5,
})

fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
fig.suptitle("NS-RAM Avalanche Threshold Characterisation",
             fontsize=14, fontweight="bold", y=0.98)

# ══════════════════════════════════════════════════════════════════════════════
# (a) Boltzmann Step Function
# ══════════════════════════════════════════════════════════════════════════════
ax = axes[0]

# Shaded bands
ax.axvspan(310, 318, alpha=0.12, color=BLUE, label="Sub-threshold")
ax.axvspan(318, 335, alpha=0.10, color=RED, label="Supra-threshold")

# Data
ax.plot(b_T, b_spikes, "o-", color=BLUE, markersize=5, linewidth=1.5,
        markerfacecolor="white", markeredgewidth=1.3, markeredgecolor=BLUE,
        zorder=5)

# Critical temperature line
ax.axvline(318, color=RED, linestyle="--", linewidth=1.2, zorder=4)
ax.text(318.4, 7.0, r"$T_{\mathrm{crit}}$ = 318 K", color=RED, fontsize=9,
        fontweight="bold", va="center")

# Annotation for critical Vg
ax.annotate(r"Critical $V_{g,\mathrm{eff}}$ = 0.486 V",
            xy=(318, 13), xytext=(312, 11),
            fontsize=8, color="0.3",
            arrowprops=dict(arrowstyle="->", color="0.4", lw=0.8))

ax.set_xlim(310, 335)
ax.set_ylim(-0.8, 16)
ax.set_xlabel("Temperature (K)")
ax.set_ylabel("LIF Spike Count")
ax.set_title("Thermal Threshold: Step Function at $T$ = 318 K")
ax.legend(loc="center right", framealpha=0.9, edgecolor="0.8")
ax.text(-0.12, 1.05, "(a)", transform=ax.transAxes,
        fontsize=13, fontweight="bold")

# ══════════════════════════════════════════════════════════════════════════════
# (b) Energy per Spike vs Gate Voltage
# ══════════════════════════════════════════════════════════════════════════════
ax = axes[1]

# Pazos band
ax.axhspan(0.2, 21.0, alpha=0.12, color=GREEN, zorder=0)
ax.axhline(0.2, color=GREEN, linestyle="--", linewidth=0.8, alpha=0.6)
ax.axhline(21.0, color=GREEN, linestyle="--", linewidth=0.8, alpha=0.6)
ax.text(0.405, 11.0, "Pazos et al.\nrange", fontsize=8, color=GREEN,
        fontstyle="italic", va="center")

# Right axis: spike count bars
ax2 = ax.twinx()
ax2.bar(e_Vg, e_spikes, width=0.018, alpha=0.22, color=BLUE, zorder=1,
        label="Spike count")
ax2.set_ylabel("LIF Spike Count", color=BLUE)
ax2.tick_params(axis="y", labelcolor=BLUE)
ax2.set_ylim(0, 20)

# Energy (only where not NaN)
mask = ~np.isnan(e_energy)
ax.plot(e_Vg[mask], e_energy[mask], "D-", color=RED, markersize=6,
        linewidth=1.5, markerfacecolor="white", markeredgewidth=1.3,
        markeredgecolor=RED, zorder=6, label="Energy/spike")

ax.set_xlim(0.39, 0.61)
ax.set_ylim(0, 22)
ax.set_xlabel(r"Gate Voltage $V_g$ (V)")
ax.set_ylabel("Energy per spike (fJ)", color=RED)
ax.tick_params(axis="y", labelcolor=RED)
ax.xaxis.set_major_locator(ticker.MultipleLocator(0.05))
ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.025))
ax.set_title(r"Energy Characteristic: $V_g$ Sweep")

# Combined legend
lines_a, labels_a = ax.get_legend_handles_labels()
lines_b, labels_b = ax2.get_legend_handles_labels()
ax.legend(lines_a + lines_b, labels_a + labels_b,
          loc="upper left", framealpha=0.9, edgecolor="0.8")

ax.text(-0.15, 1.05, "(b)", transform=ax.transAxes,
        fontsize=13, fontweight="bold")

# ══════════════════════════════════════════════════════════════════════════════
# (c) Phase Diagram — Temperature x Gate Voltage
# ══════════════════════════════════════════════════════════════════════════════
ax = axes[2]

# Phase boundary: Vg_crit(T) = 0.45 + 0.002*(T - 300)
T_line = np.linspace(290, 370, 200)
Vg_crit = 0.45 + 0.002 * (T_line - 300)

# Shaded regions
ax.fill_between(T_line, 0.35, Vg_crit, alpha=0.12, color=BLUE, label="Silent")
ax.fill_between(T_line, Vg_crit, 0.65, alpha=0.10, color=RED, label="Spiking")
ax.plot(T_line, Vg_crit, "k-", linewidth=1.3, zorder=4)

# Region labels
ax.text(300, 0.42, "Silent", fontsize=10, color=BLUE, fontstyle="italic",
        fontweight="bold", ha="center")
ax.text(345, 0.57, "Spiking", fontsize=10, color=RED, fontstyle="italic",
        fontweight="bold", ha="center")

# Original 5 thermal data points (300, 318, 327, 345, 358 K approx)
T_orig = np.array([300, 318, 327, 345, 358])
Vg_orig = 0.45 + 0.002 * (T_orig - 300)
spk_orig = np.array([0, 13, 14, 14, 15])
ax.scatter(T_orig, Vg_orig, marker="*", s=120, c=[BLUE if s == 0 else RED for s in spk_orig],
           edgecolors="k", linewidths=0.6, zorder=7, label="Thermal (5 pts)")

# Energy sweep points — all at effective T=300K, varying Vg
ax.scatter(np.full(len(e_Vg), 300), e_Vg, marker="D", s=50,
           c=[BLUE if s == 0 else RED for s in e_spikes],
           edgecolors="k", linewidths=0.6, zorder=7, label=r"$V_g$ sweep")

# Boltzmann sweep — 26 points, color by spike count
sc = ax.scatter(b_T, b_Vg, c=b_spikes, cmap="RdYlBu_r", marker="o", s=40,
                edgecolors="k", linewidths=0.5, zorder=6, vmin=0, vmax=15,
                label="Boltzmann sweep")
cbar = fig.colorbar(sc, ax=ax, shrink=0.75, pad=0.02)
cbar.set_label("Spike count", fontsize=9)

ax.set_xlim(290, 370)
ax.set_ylim(0.38, 0.62)
ax.set_xlabel("Temperature (K)")
ax.set_ylabel(r"Gate Voltage $V_g$ (V)")
ax.set_title("Phase Diagram: Spiking Boundary")
ax.legend(loc="lower right", fontsize=7.5, framealpha=0.9, edgecolor="0.8")
ax.text(-0.12, 1.05, "(c)", transform=ax.transAxes,
        fontsize=13, fontweight="bold")

# ── Save ─────────────────────────────────────────────────────────────────────
plt.tight_layout(rect=[0, 0, 1, 0.94])

out = Path("/tmp/feel_paper/FEEL__Functionally_Embodied_Emergent_Learning__13_/figures/fig_boltzmann_energy.png")
out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
plt.close()
print(f"Saved: {out}  ({out.stat().st_size / 1024:.1f} KB)")
