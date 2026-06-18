#!/usr/bin/env python3
"""
z1013: Plot validation summary for z1006-z1012 series.
Creates visualization of what works vs doesn't work.
"""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Data from experiments
experiments = {
    'z1006': {'name': 'Simplified Arch', 'baseline_ppl': 1.49, 'energy_ppl': 13.11, 'verdict': 'FAIL'},
    'z1007': {'name': 'FiLM Arch', 'baseline_ppl': 1.01, 'energy_ppl': 11.84, 'verdict': 'FAIL'},
    'z1008': {'name': 'Stable MSE', 'baseline_ppl': 1.00, 'energy_ppl': 1.00, 'verdict': 'OK'},
    'z1009': {'name': 'Generalization', 'baseline_ppl': 11.77, 'energy_ppl': 11.89, 'verdict': 'NEUTRAL'},
    'z1010': {'name': 'Kernel Active', 'baseline_ppl': 1.29, 'energy_ppl': 1.76, 'verdict': 'TRADEOFF'},
    'z1011': {'name': 'Gates+Energy', 'baseline_ppl': 10.00, 'energy_ppl': 37.78, 'verdict': 'FAIL'},
    'z1012': {'name': 'Decoupled', 'baseline_ppl': 10.00, 'energy_ppl': 10.00, 'verdict': 'NEUTRAL'},
}

# Create figure with multiple subplots
fig = plt.figure(figsize=(16, 12))
fig.suptitle('z1000 Series: Energy-Aware LM Validation Summary\n(FEEL Hypothesis Testing)',
             fontsize=14, fontweight='bold')

# 1. PPL Comparison Bar Chart
ax1 = fig.add_subplot(2, 2, 1)
x = np.arange(len(experiments))
width = 0.35

baselines = [v['baseline_ppl'] for v in experiments.values()]
energies = [v['energy_ppl'] for v in experiments.values()]
names = [v['name'] for v in experiments.values()]

bars1 = ax1.bar(x - width/2, baselines, width, label='Baseline', color='steelblue', alpha=0.8)
bars2 = ax1.bar(x + width/2, energies, width, label='With Energy', color='coral', alpha=0.8)

ax1.set_xlabel('Experiment')
ax1.set_ylabel('Perplexity (lower is better)')
ax1.set_title('PPL: Baseline vs Energy-Aware')
ax1.set_xticks(x)
ax1.set_xticklabels(names, rotation=45, ha='right')
ax1.legend()
ax1.set_yscale('log')
ax1.grid(axis='y', alpha=0.3)

# Add verdict colors
verdicts = [v['verdict'] for v in experiments.values()]
colors = {'FAIL': 'red', 'OK': 'green', 'NEUTRAL': 'gray', 'TRADEOFF': 'orange'}
for i, (bar, verdict) in enumerate(zip(bars2, verdicts)):
    bar.set_edgecolor(colors[verdict])
    bar.set_linewidth(2)

# 2. Verdict Summary Pie Chart
ax2 = fig.add_subplot(2, 2, 2)
verdict_counts = {'FAIL': 0, 'OK': 0, 'NEUTRAL': 0, 'TRADEOFF': 0}
for v in experiments.values():
    verdict_counts[v['verdict']] += 1

labels = [f"{k}\n({v})" for k, v in verdict_counts.items() if v > 0]
sizes = [v for v in verdict_counts.values() if v > 0]
colors_pie = [colors[k] for k, v in verdict_counts.items() if v > 0]

ax2.pie(sizes, labels=labels, colors=colors_pie, autopct='%1.0f%%',
        startangle=90, explode=[0.05]*len(sizes))
ax2.set_title('Experiment Outcomes')

# 3. Key Metrics Summary
ax3 = fig.add_subplot(2, 2, 3)
ax3.axis('off')

summary_text = """
KEY FINDINGS FROM z1006-z1012:

✅ WHAT WORKS:
• Energy prediction IS learnable (1.5% MAPE)
• Self-modeling works (93% better hidden prediction)
• Active inference helps coherence (+65%)
• Gates can reduce computation (~2%)

❌ WHAT DOESN'T WORK:
• Energy awareness improving LM quality (p=0.55)
• Joint training with energy loss (destabilizes)
• Telemetry-only gates (no signal)
• Energy modulation (hurts PPL +36%)

🔑 CRITICAL INSIGHT (z1012):
Hidden states do NOT carry per-sample energy info.
Energy is batch-level constant, not content-dependent.
Correlation = 0.021 (near zero!)

VERDICT: FEEL hypothesis FALSIFIED at this level.
Need to go LOWER into HW/SW boundary.
"""

ax3.text(0.05, 0.95, summary_text, transform=ax3.transAxes,
         fontsize=10, verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# 4. PPL Change Waterfall
ax4 = fig.add_subplot(2, 2, 4)

# Calculate PPL changes
changes = [(v['energy_ppl'] - v['baseline_ppl']) / v['baseline_ppl'] * 100
           for v in experiments.values()]

colors_bar = ['red' if c > 10 else 'orange' if c > 0 else 'green' for c in changes]
bars = ax4.barh(names, changes, color=colors_bar, alpha=0.8)
ax4.axvline(x=0, color='black', linewidth=1)
ax4.axvline(x=10, color='red', linewidth=1, linestyle='--', alpha=0.5, label='10% threshold')
ax4.axvline(x=-10, color='green', linewidth=1, linestyle='--', alpha=0.5)

ax4.set_xlabel('PPL Change (%)')
ax4.set_title('Impact of Energy Awareness on PPL')
ax4.set_xlim(-20, 300)

# Add value labels
for bar, change in zip(bars, changes):
    width = bar.get_width()
    label = f'+{change:.0f}%' if change > 0 else f'{change:.0f}%'
    x_pos = min(width + 5, 280) if width > 0 else width - 15
    ax4.text(x_pos, bar.get_y() + bar.get_height()/2, label,
             va='center', fontsize=9)

plt.tight_layout()

# Save
out_path = Path(__file__).parent.parent / "reports" / "z1000_validation_summary.png"
plt.savefig(out_path, dpi=150, bbox_inches='tight')
print(f"Saved to {out_path}")

# Also save a simpler version
fig2, ax = plt.subplots(figsize=(10, 6))

# Simple bar chart showing the key message
categories = ['z1008\n(Learnable)', 'z1009\n(Quality)', 'z1010\n(Tradeoff)', 'z1011\n(Combined)', 'z1012\n(Decoupled)']
ppl_ratios = [1.00, 1.01, 1.36, 3.78, 1.00]
energy_learnable = [True, True, True, False, True]

colors_simple = ['green' if r < 1.1 else 'orange' if r < 1.5 else 'red' for r in ppl_ratios]
bars = ax.bar(categories, ppl_ratios, color=colors_simple, alpha=0.8, edgecolor='black')

ax.axhline(y=1.0, color='blue', linestyle='--', linewidth=2, label='Baseline (1.0)')
ax.axhline(y=1.1, color='green', linestyle=':', label='Acceptable (10%)')
ax.axhline(y=1.5, color='orange', linestyle=':', label='Tradeoff (50%)')

ax.set_ylabel('PPL Ratio (Energy / Baseline)')
ax.set_title('Energy Awareness Impact on Language Modeling\n(Lower is better, 1.0 = no change)', fontsize=12)
ax.legend(loc='upper right')
ax.set_ylim(0, 4.5)

# Add labels
for bar, ratio in zip(bars, ppl_ratios):
    height = bar.get_height()
    label = f'{ratio:.2f}x'
    ax.text(bar.get_x() + bar.get_width()/2, height + 0.1, label,
            ha='center', va='bottom', fontsize=10, fontweight='bold')

# Add verdict text
verdicts_simple = ['✅ Learnable', '➖ No benefit', '⚠️ Tradeoff', '❌ Destroys', '➖ No signal']
for bar, verdict in zip(bars, verdicts_simple):
    ax.text(bar.get_x() + bar.get_width()/2, 0.1, verdict,
            ha='center', va='bottom', fontsize=8, rotation=90)

plt.tight_layout()
out_path2 = Path(__file__).parent.parent / "reports" / "z1000_simple_summary.png"
plt.savefig(out_path2, dpi=150, bbox_inches='tight')
print(f"Saved to {out_path2}")

plt.show()
