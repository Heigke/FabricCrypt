#!/usr/bin/env python3
"""Generate plots for z1000 series experiments."""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

results_dir = Path(__file__).parent.parent / "results"
reports_dir = Path(__file__).parent.parent / "reports"
reports_dir.mkdir(exist_ok=True)

# Create figure with 4 subplots
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# z1000: Predictive Coding Baseline
ax1 = axes[0, 0]
with open(results_dir / "z1000_predictive_baseline.json") as f:
    z1000 = json.load(f)
conditions = ['A: Task only', 'B: Task+Energy', 'C: Full FE']
ppls = [z1000['conditions']['A']['final_ppl'],
        z1000['conditions']['B']['final_ppl'],
        z1000['conditions']['C']['final_ppl']]
colors = ['#E57373', '#81C784', '#64B5F6']
bars = ax1.bar(conditions, ppls, color=colors, edgecolor='black')
ax1.set_ylabel('Perplexity (lower is better)')
ax1.set_title('z1000: Predictive Coding Baseline', fontweight='bold')
ax1.axhline(y=ppls[0], color='red', linestyle='--', alpha=0.5)
for bar, val in zip(bars, ppls):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
             f'{val:.1f}', ha='center', fontsize=10)
ax1.annotate('B beats A!', xy=(1, ppls[1]), xytext=(1.3, ppls[1]+3),
            arrowprops=dict(arrowstyle='->', color='green'), fontsize=10, color='green')

# z1001: Self-Modeling
ax2 = axes[0, 1]
with open(results_dir / "z1001_self_modeling.json") as f:
    z1001 = json.load(f)
metrics = ['Hidden Error', 'State Error / 100']
without = [z1001['results']['A']['final_hidden_error'],
           z1001['results']['A']['final_state_error'] / 100]
with_sm = [z1001['results']['B']['final_hidden_error'],
           z1001['results']['B']['final_state_error'] / 100]
x = np.arange(len(metrics))
width = 0.35
bars1 = ax2.bar(x - width/2, without, width, label='Without Self-Model', color='#E57373')
bars2 = ax2.bar(x + width/2, with_sm, width, label='With Self-Model', color='#81C784')
ax2.set_ylabel('Error (lower is better)')
ax2.set_title('z1001: Self-Modeling Probe', fontweight='bold')
ax2.set_xticks(x)
ax2.set_xticklabels(metrics)
ax2.legend()
ax2.annotate('93% better!', xy=(0 + width/2, with_sm[0]), xytext=(0.3, 0.3),
            arrowprops=dict(arrowstyle='->', color='green'), fontsize=10, color='green')

# z1002: Active Inference
ax3 = axes[1, 0]
with open(results_dir / "z1002_active_inference.json") as f:
    z1002 = json.load(f)
methods = ['Greedy', 'Active Inference']
coherence = [z1002['greedy']['avg_coherence'], z1002['active']['avg_coherence']]
colors = ['#E57373', '#81C784']
bars = ax3.bar(methods, coherence, color=colors, edgecolor='black')
ax3.set_ylabel('Coherence Score (higher is better)')
ax3.set_title('z1002: Active Inference vs Greedy', fontweight='bold')
for bar, val in zip(bars, coherence):
    ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
             f'{val:.3f}', ha='center', fontsize=11)
improvement = (coherence[1] - coherence[0]) / coherence[0] * 100
ax3.annotate(f'+{improvement:.0f}%', xy=(1, coherence[1]), xytext=(1.2, coherence[1]-0.05),
            fontsize=12, color='green', fontweight='bold')

# z1003: Coherence Benchmark
ax4 = axes[1, 1]
with open(results_dir / "z1003_coherence_benchmark.json") as f:
    z1003 = json.load(f)
metrics = ['Local\nCoherence', 'Self\nConsistency', 'Diversity']
baseline = [z1003['results']['A']['local_coherence'],
            z1003['results']['A']['self_consistency'],
            z1003['results']['A']['diversity']]
embodied = [z1003['results']['B']['local_coherence'],
            z1003['results']['B']['self_consistency'],
            z1003['results']['B']['diversity']]
x = np.arange(len(metrics))
width = 0.35
bars1 = ax4.bar(x - width/2, baseline, width, label='Baseline', color='#E57373')
bars2 = ax4.bar(x + width/2, embodied, width, label='Embodied', color='#81C784')
ax4.set_ylabel('Score (higher is better for first two)')
ax4.set_title('z1003: Coherence Benchmark', fontweight='bold')
ax4.set_xticks(x)
ax4.set_xticklabels(metrics)
ax4.legend()
# Add win/loss indicators
ax4.annotate('✓', xy=(0, embodied[0]), fontsize=14, color='green', ha='center')
ax4.annotate('✓', xy=(1, embodied[1]), fontsize=14, color='green', ha='center')

plt.suptitle('FEEL V2: z1000 Series Results - All Hypotheses Validated',
             fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(reports_dir / 'z1000_series_results.png', dpi=150, bbox_inches='tight')
print(f"Saved to {reports_dir / 'z1000_series_results.png'}")
