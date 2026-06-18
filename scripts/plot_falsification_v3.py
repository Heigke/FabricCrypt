#!/usr/bin/env python3
"""
Plot FEEL v3 Falsification Results
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Load results
results_dir = Path(__file__).parent.parent / "results"
falsification_path = results_dir / "falsification_v2" / "falsification_results.json"

with open(falsification_path) as f:
    data = json.load(f)

# Create figure with 2 subplots
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Plot 1: Cross-Prompt Swap
ax1 = axes[0]
conditions = ['Live', 'Cross-Prompt\nSwap']
heat_accs = [
    data['results']['cross_prompt']['live']['heat']['acc'] * 100,
    data['results']['cross_prompt']['cross']['heat']['acc'] * 100
]
colors = ['#2ecc71', '#e74c3c']
bars = ax1.bar(conditions, heat_accs, color=colors, edgecolor='black', linewidth=1.5)
ax1.axhline(y=25, color='gray', linestyle='--', linewidth=2, label='Chance (25%)')
ax1.set_ylabel('Heat Accuracy (%)', fontsize=12)
ax1.set_title('Cross-Prompt Swap Test\n(Information-Destructive Falsification)', fontsize=12, fontweight='bold')
ax1.set_ylim(0, 110)
ax1.legend(loc='upper right')

# Add value labels
for bar, acc in zip(bars, heat_accs):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
             f'{acc:.1f}%', ha='center', va='bottom', fontsize=14, fontweight='bold')

# Add annotation for collapse
ax1.annotate('', xy=(1, heat_accs[1]), xytext=(0, heat_accs[0]),
            arrowprops=dict(arrowstyle='->', color='red', lw=2))
ax1.text(0.5, 60, '−70%\nCollapse!', ha='center', fontsize=11, color='red', fontweight='bold')

# Plot 2: Lag Sweep
ax2 = axes[1]
lags = [0, 8, 32, 128]
lag_accs = [
    data['results']['lag_sweep']['0']['heat']['acc'] * 100,
    data['results']['lag_sweep']['8']['heat']['acc'] * 100,
    data['results']['lag_sweep']['32']['heat']['acc'] * 100,
    data['results']['lag_sweep']['128']['heat']['acc'] * 100
]

ax2.plot(lags, lag_accs, 'o-', color='#3498db', linewidth=2.5, markersize=10, markerfacecolor='white', markeredgewidth=2)
ax2.axhline(y=25, color='gray', linestyle='--', linewidth=2, label='Chance (25%)')
ax2.fill_between(lags, lag_accs, alpha=0.3, color='#3498db')

ax2.set_xlabel('Lag (tokens)', fontsize=12)
ax2.set_ylabel('Heat Accuracy (%)', fontsize=12)
ax2.set_title('Lag Sweep Test\n(Monotonic Temporal Degradation)', fontsize=12, fontweight='bold')
ax2.set_ylim(0, 110)
ax2.set_xticks(lags)
ax2.legend(loc='upper right')

# Add value labels
for lag, acc in zip(lags, lag_accs):
    ax2.annotate(f'{acc:.0f}%', (lag, acc), textcoords="offset points",
                xytext=(0, 10), ha='center', fontsize=11, fontweight='bold')

# Add monotonic arrow
ax2.annotate('Monotonic\nDegradation', xy=(80, 30), fontsize=10, color='#2c3e50',
            ha='center', style='italic')

plt.tight_layout()

# Save
output_path = results_dir / "falsification_v2" / "falsification_plots.png"
plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
print(f"Saved: {output_path}")

# Also save to reports for README
reports_path = Path(__file__).parent.parent / "reports" / "feel_v3_falsification.png"
reports_path.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(reports_path, dpi=150, bbox_inches='tight', facecolor='white')
print(f"Saved: {reports_path}")

plt.close()
