#!/usr/bin/env python3
"""
Plot validation results from three-toggle demo and benefit-collapse test.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Find the latest results
three_toggle_dir = Path("results/three_toggle_demo")
benefit_collapse_dir = Path("results/benefit_collapse")

# Get latest files
three_toggle_files = sorted(three_toggle_dir.glob("three_toggle_*.json"))
benefit_collapse_files = sorted(benefit_collapse_dir.glob("benefit_collapse_*.json"))

if not three_toggle_files:
    print("No three-toggle results found")
    exit(1)

three_toggle_file = three_toggle_files[-1]
print(f"Using three-toggle: {three_toggle_file}")

# Load data
with open(three_toggle_file) as f:
    three_toggle = json.load(f)

# Create output directory
output_dir = Path("reports/validation")
output_dir.mkdir(parents=True, exist_ok=True)

# ============================================================
# Plot 1: Three-toggle temperature comparison
# ============================================================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Temperature time series
ax = axes[0, 0]
for mode_name, color in [("live", "green"), ("disabled", "red"), ("cross_swap", "blue")]:
    if mode_name in three_toggle["results"]:
        data = three_toggle["results"][mode_name]
        ax.plot(data["timestamps"], data["temps"], color=color, label=mode_name.upper(), alpha=0.8)
ax.axhline(60, color="orange", linestyle="--", label="T_high=60°C")
ax.set_xlabel("Time (s)")
ax.set_ylabel("Temperature (°C)")
ax.set_title("Temperature Over Time (Three-Toggle Demo)")
ax.legend()
ax.grid(True, alpha=0.3)

# Stress scores over time
ax = axes[0, 1]
for mode_name, color in [("live", "green"), ("disabled", "red"), ("cross_swap", "blue")]:
    if mode_name in three_toggle["results"]:
        data = three_toggle["results"][mode_name]
        ax.plot(data["timestamps"], data["stress_scores"], color=color, label=mode_name.upper(), alpha=0.8)
ax.axhline(0.6, color="orange", linestyle="--", label="Stress threshold")
ax.set_xlabel("Time (s)")
ax.set_ylabel("Stress Score")
ax.set_title("Thermal Stress (from z_feel)")
ax.legend()
ax.grid(True, alpha=0.3)

# Bar chart: mean temperatures
ax = axes[1, 0]
modes = ["live", "disabled", "cross_swap"]
temps = []
colors = ["green", "red", "blue"]
for mode in modes:
    if mode in three_toggle["results"]:
        temps.append(np.mean(three_toggle["results"][mode]["temps"]))
    else:
        temps.append(0)

bars = ax.bar(modes, temps, color=colors, alpha=0.7, edgecolor="black")
ax.set_ylabel("Mean Temperature (°C)")
ax.set_title("Mean Temperature by Mode")
ax.axhline(60, color="orange", linestyle="--", label="T_high")

# Add value labels
for bar, temp in zip(bars, temps):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
            f"{temp:.1f}°C", ha="center", va="bottom", fontweight="bold")

# Improvement annotation
if "live" in three_toggle["results"] and "disabled" in three_toggle["results"]:
    live_temp = np.mean(three_toggle["results"]["live"]["temps"])
    disabled_temp = np.mean(three_toggle["results"]["disabled"]["temps"])
    diff = disabled_temp - live_temp
    ax.annotate(f"Controller saves\n{diff:.1f}°C", xy=(0.5, 0.85), xycoords="axes fraction",
                fontsize=12, fontweight="bold", ha="center",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.8))

ax.grid(True, alpha=0.3, axis="y")

# DVFS action distribution
ax = axes[1, 1]
if "live" in three_toggle["results"]:
    dvfs_actions = three_toggle["results"]["live"]["dvfs_actions"]
    from collections import Counter
    counts = Counter(dvfs_actions)
    labels = list(counts.keys())
    values = list(counts.values())
    ax.bar(labels, values, color=["skyblue", "salmon", "lightgreen"][:len(labels)], edgecolor="black")
    ax.set_ylabel("Count")
    ax.set_title("DVFS Actions (LIVE mode)")
    ax.grid(True, alpha=0.3, axis="y")

plt.tight_layout()
plt.savefig(output_dir / "three_toggle_results.png", dpi=150, bbox_inches="tight")
print(f"Saved: {output_dir / 'three_toggle_results.png'}")

# ============================================================
# Plot 2: Benefit collapse test (if available)
# ============================================================
if benefit_collapse_files:
    benefit_file = benefit_collapse_files[-1]
    print(f"Using benefit-collapse: {benefit_file}")

    with open(benefit_file) as f:
        benefit_data = json.load(f)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Temperature comparison
    ax = axes[0]
    modes = ["baseline", "live", "cross_swap", "z_feel_shuffle"]
    temps = []
    colors = ["gray", "green", "blue", "orange"]

    for mode in modes:
        if mode in benefit_data["results"]:
            temps.append(np.mean(benefit_data["results"][mode]["traces"]["temps"]))
        else:
            temps.append(0)

    bars = ax.bar(modes, temps, color=colors, alpha=0.7, edgecolor="black")
    ax.set_ylabel("Mean Temperature (°C)")
    ax.set_title("Temperature by Falsification Mode")
    ax.set_xticklabels([m.replace("_", "\n") for m in modes], rotation=0)

    for bar, temp in zip(bars, temps):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f"{temp:.1f}°C", ha="center", va="bottom", fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")

    # Time above threshold
    ax = axes[1]
    time_above = []
    for mode in modes:
        if mode in benefit_data["results"]:
            time_above.append(benefit_data["results"][mode]["metrics"]["time_above_threshold"] * 100)
        else:
            time_above.append(0)

    bars = ax.bar(modes, time_above, color=colors, alpha=0.7, edgecolor="black")
    ax.set_ylabel("Time Above 60°C (%)")
    ax.set_title("Time Above Threshold")
    ax.set_xticklabels([m.replace("_", "\n") for m in modes], rotation=0)

    for bar, val in zip(bars, time_above):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
                f"{val:.1f}%", ha="center", va="bottom", fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")

    # Overshoot area
    ax = axes[2]
    overshoot = []
    for mode in modes:
        if mode in benefit_data["results"]:
            overshoot.append(benefit_data["results"][mode]["metrics"]["overshoot_area"])
        else:
            overshoot.append(0)

    bars = ax.bar(modes, overshoot, color=colors, alpha=0.7, edgecolor="black")
    ax.set_ylabel("Overshoot Area (°C·s)")
    ax.set_title("Thermal Overshoot")
    ax.set_xticklabels([m.replace("_", "\n") for m in modes], rotation=0)

    for bar, val in zip(bars, overshoot):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
                f"{val:.1f}", ha="center", va="bottom", fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(output_dir / "benefit_collapse_results.png", dpi=150, bbox_inches="tight")
    print(f"Saved: {output_dir / 'benefit_collapse_results.png'}")

# ============================================================
# Plot 3: Summary comparison chart
# ============================================================
fig, ax = plt.subplots(figsize=(10, 6))

# Data from three-toggle
if "live" in three_toggle["results"] and "disabled" in three_toggle["results"]:
    live = three_toggle["results"]["live"]
    disabled = three_toggle["results"]["disabled"]

    categories = ["Mean Temp\n(°C)", "Max Temp\n(°C)", "Temp Std\n(°C)", "Time >60°C\n(%)"]
    live_vals = [
        np.mean(live["temps"]),
        np.max(live["temps"]),
        np.std(live["temps"]),
        np.mean(np.array(live["temps"]) > 60) * 100
    ]
    disabled_vals = [
        np.mean(disabled["temps"]),
        np.max(disabled["temps"]),
        np.std(disabled["temps"]),
        np.mean(np.array(disabled["temps"]) > 60) * 100
    ]

    x = np.arange(len(categories))
    width = 0.35

    bars1 = ax.bar(x - width/2, live_vals, width, label="LIVE (Controller)", color="green", alpha=0.7)
    bars2 = ax.bar(x + width/2, disabled_vals, width, label="DISABLED (No Controller)", color="red", alpha=0.7)

    ax.set_ylabel("Value")
    ax.set_title("FEEL Controller Performance: LIVE vs DISABLED")
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f"{height:.1f}",
                       xy=(bar.get_x() + bar.get_width()/2, height),
                       xytext=(0, 3), textcoords="offset points",
                       ha="center", va="bottom", fontsize=9)

plt.tight_layout()
plt.savefig(output_dir / "controller_comparison.png", dpi=150, bbox_inches="tight")
print(f"Saved: {output_dir / 'controller_comparison.png'}")

print("\nAll plots generated successfully!")
