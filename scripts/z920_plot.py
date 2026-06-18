#!/usr/bin/env python3
"""Generate z920 comparison plot."""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Load data
results_path = Path(__file__).parent.parent / "results" / "z920_comprehensive.json"
with open(results_path) as f:
    data = json.load(f)

# Extract data
controllers = ['Fixed-eco', 'Fixed-balanced', 'Adaptive', 'Memory']
steady = [data['results'][f'steady_{c}']['j_per_token'] * 1000 for c in controllers]
thermal = [
    None,  # eco not tested
    data['results']['thermal_Fixed-balanced']['j_per_token'] * 1000,
    data['results']['thermal_Adaptive']['j_per_token'] * 1000,
    data['results']['thermal_Memory']['j_per_token'] * 1000
]

# Create figure
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Steady-state plot
colors = ['#2196F3', '#4CAF50', '#FF9800', '#9C27B0']
bars1 = ax1.bar(controllers, steady, color=colors, edgecolor='black')
ax1.set_ylabel('mJ/token (lower is better)', fontsize=12)
ax1.set_title('z920: Steady-State Efficiency', fontsize=14, fontweight='bold')
ax1.axhline(y=min(steady), color='green', linestyle='--', alpha=0.5, label=f'Best: {min(steady):.3f}')
ax1.legend()
for bar, val in zip(bars1, steady):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
             f'{val:.3f}', ha='center', va='bottom', fontsize=10)

# Thermal stress plot
thermal_controllers = ['Fixed-balanced', 'Adaptive', 'Memory']
thermal_vals = [v for v in thermal if v is not None]
thermal_colors = ['#4CAF50', '#FF9800', '#9C27B0']
bars2 = ax2.bar(thermal_controllers, thermal_vals, color=thermal_colors, edgecolor='black')
ax2.set_ylabel('mJ/token (lower is better)', fontsize=12)
ax2.set_title('z920: Thermal Stress Test', fontsize=14, fontweight='bold')
ax2.axhline(y=min(thermal_vals), color='green', linestyle='--', alpha=0.5, label=f'Best: {min(thermal_vals):.3f}')
ax2.legend()
for bar, val in zip(bars2, thermal_vals):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
             f'{val:.3f}', ha='center', va='bottom', fontsize=10)

plt.suptitle('z920 Comprehensive FEEL Validation - AMD Radeon 8060S', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig(Path(__file__).parent.parent / 'reports' / 'z920_comparison.png', dpi=150, bbox_inches='tight')
print("Saved to reports/z920_comparison.png")
