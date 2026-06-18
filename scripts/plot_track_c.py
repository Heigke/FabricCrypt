#!/usr/bin/env python3
"""
Plot Track C Loop Proof Results
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

results_dir = Path(__file__).parent.parent / "results"

# Load loop proof single run
single_path = results_dir / "loop_proof" / "loop_proof_single_20260106_202207.json"
with open(single_path) as f:
    single_data = json.load(f)

# Load comparison
compare_path = results_dir / "loop_proof" / "loop_proof_compare_20260106_202814.json"
with open(compare_path) as f:
    compare_data = json.load(f)

# Create figure with 3 subplots
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

# Plot 1: Loop Proof - Temperature and Effort over time
ax1 = axes[0]
traces = single_data["traces"]
timestamps = np.array(traces["timestamps"])
temps = np.array(traces["temps"])
effort = np.array(traces["effort_ema"])
stress = np.array(traces["stressor_active"])

# Find stress regions
stress_start = None
stress_end = None
for i, s in enumerate(stress):
    if s and stress_start is None:
        stress_start = timestamps[i]
    if not s and stress_start is not None and stress_end is None:
        stress_end = timestamps[i]

ax1_twin = ax1.twinx()

# Plot temperature
line1, = ax1.plot(timestamps, temps, 'r-', linewidth=2, label='Temperature (°C)')
ax1.set_ylabel('Temperature (°C)', color='red', fontsize=11)
ax1.tick_params(axis='y', labelcolor='red')
ax1.set_ylim(45, 70)

# Plot effort
line2, = ax1_twin.plot(timestamps, effort, 'b-', linewidth=2, label='Effort (EMA)')
ax1_twin.set_ylabel('Effort', color='blue', fontsize=11)
ax1_twin.tick_params(axis='y', labelcolor='blue')
ax1_twin.set_ylim(0.3, 0.8)

# Shade stress region
if stress_start and stress_end:
    ax1.axvspan(stress_start, stress_end, alpha=0.2, color='orange', label='GPU Stress')

# Add K threshold line
ax1_twin.axhline(y=0.6, color='green', linestyle='--', alpha=0.7, label='K threshold')

ax1.set_xlabel('Time (s)', fontsize=11)
ax1.set_title('Loop Proof: Temp → Effort → K\n(Stress ON: 30-60s)', fontsize=12, fontweight='bold')
ax1.legend([line1, line2], ['Temperature', 'Effort'], loc='upper left')

# Plot 2: K values over time
ax2 = axes[1]
K_values = np.array(traces["K_values"])

ax2.step(timestamps, K_values, 'g-', linewidth=2.5, where='post')
ax2.fill_between(timestamps, K_values, step='post', alpha=0.3, color='green')

if stress_start and stress_end:
    ax2.axvspan(stress_start, stress_end, alpha=0.2, color='orange')

ax2.set_xlabel('Time (s)', fontsize=11)
ax2.set_ylabel('K (self-consistency samples)', fontsize=11)
ax2.set_title('K Flips from 1→2 During Stress', fontsize=12, fontweight='bold')
ax2.set_ylim(0.5, 2.5)
ax2.set_yticks([1, 2])
ax2.axhline(y=1.5, color='gray', linestyle=':', alpha=0.5)

# Annotate
ax2.annotate('K=1\n(low effort)', xy=(15, 1), fontsize=10, ha='center', color='#27ae60')
ax2.annotate('K=2\n(high effort)', xy=(50, 2), fontsize=10, ha='center', color='#27ae60')

# Plot 3: Falsification Comparison
ax3 = axes[2]

comparison = compare_data["comparison"]
modes = ['Live', 'Shuffle\n(Falsified)']
correlations = [
    comparison["live_temp_effort_correlation"],
    comparison["cross_temp_effort_correlation"]
]
colors = ['#2ecc71', '#e74c3c']

bars = ax3.bar(modes, correlations, color=colors, edgecolor='black', linewidth=1.5)
ax3.set_ylabel('Temp↔Effort Correlation', fontsize=11)
ax3.set_title('Falsification Test\n(Shuffle Breaks Correlation)', fontsize=12, fontweight='bold')
ax3.set_ylim(0, 0.8)

# Add value labels
for bar, corr in zip(bars, correlations):
    ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
             f'{corr:.3f}', ha='center', va='bottom', fontsize=12, fontweight='bold')

# Add collapse annotation
collapse = comparison["correlation_collapse"]
ax3.annotate('', xy=(1, correlations[1]), xytext=(0, correlations[0]),
            arrowprops=dict(arrowstyle='->', color='red', lw=2))
ax3.text(0.5, 0.48, f'−46%\nCollapse', ha='center', fontsize=11, color='red', fontweight='bold')

plt.tight_layout()

# Save
output_path = results_dir / "loop_proof" / "track_c_plots.png"
plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
print(f"Saved: {output_path}")

# Also save to reports for README
reports_path = Path(__file__).parent.parent / "reports" / "track_c_loop_proof.png"
reports_path.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(reports_path, dpi=150, bbox_inches='tight', facecolor='white')
print(f"Saved: {reports_path}")

plt.close()
