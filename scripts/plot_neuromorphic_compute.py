#!/usr/bin/env python3
"""Generate figure for neuromorphic computation section of FEEL paper."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

fig, axes = plt.subplots(2, 2, figsize=(13, 9))

# ─── Panel (a): 4 computation modes (from neuromorphic_compute_step5.txt) ───
modes = ['Cross-WF\nCoupling', 'Analog\nTiming', 'PLL\nDrift', 'Instr.\nMix']
accs = [96.7, 90.7, 72.0, 28.7]
repros = [0.994, 0.313, 0.916, -0.073]
colors = ['#2ecc71', '#3498db', '#9b59b6', '#e74c3c']

bars = axes[0,0].bar(modes, accs, color=colors, edgecolor='black', linewidth=0.5, width=0.65)
axes[0,0].axhline(y=25, color='gray', linestyle='--', alpha=0.7, label='Chance (25%)')
axes[0,0].set_ylabel('Classification accuracy (%)', fontsize=10)
axes[0,0].set_title('(a) Four computation modes', fontsize=11, fontweight='bold')
axes[0,0].set_ylim(0, 105)
axes[0,0].legend(fontsize=8)

for bar, r in zip(bars, repros):
    axes[0,0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.5,
                f'r={r:.2f}', ha='center', va='bottom', fontsize=7.5, style='italic')

# ─── Panel (b): Recurrent vs non-recurrent vs baselines ───
methods = ['Recurrent\nSilicon', 'Non-recurrent\nSilicon', 'LIF+PRNG', 'Tanh ESN']
acc_4c = [100.0, 94.4, 92.2, 90.0]
acc_8c = [95.6, 95.0, 93.1, 78.3]
x = np.arange(len(methods))
w = 0.32

b1 = axes[0,1].bar(x - w/2, acc_4c, w, label='4-class', color='#2ecc71', edgecolor='black', linewidth=0.5)
b2 = axes[0,1].bar(x + w/2, acc_8c, w, label='8-class', color='#3498db', edgecolor='black', linewidth=0.5)
axes[0,1].set_ylabel('Classification accuracy (%)', fontsize=10)
axes[0,1].set_title('(b) Recurrent silicon vs baselines', fontsize=11, fontweight='bold')
axes[0,1].set_xticks(x)
axes[0,1].set_xticklabels(methods, fontsize=9)
axes[0,1].set_ylim(70, 105)
axes[0,1].legend(fontsize=8, loc='lower left')

for bars_group in [b1, b2]:
    for bar in bars_group:
        axes[0,1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                    f'{bar.get_height():.1f}', ha='center', va='bottom', fontsize=7.5)

# ─── Panel (c): Feature ablation ───
feat_names = ['All 12\nfeatures', 'Temporal\nonly (8)', 'Basic\nonly (3)']
feat_accs = [95.0, 93.3, 81.7]
feat_colors = ['#2ecc71', '#3498db', '#e67e22']

bars_f = axes[1,0].bar(feat_names, feat_accs, color=feat_colors, edgecolor='black', linewidth=0.5, width=0.55)
axes[1,0].axhline(y=12.5, color='gray', linestyle='--', alpha=0.7, label='Chance (12.5%)')
axes[1,0].set_ylabel('8-class accuracy (%)', fontsize=10)
axes[1,0].set_title('(c) Feature ablation', fontsize=11, fontweight='bold')
axes[1,0].set_ylim(0, 105)
axes[1,0].legend(fontsize=8)

for bar in bars_f:
    axes[1,0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                  f'{bar.get_height():.1f}%', ha='center', va='bottom', fontsize=9)

# ─── Panel (d): Full 9-task comparison ───
tasks = ['4-cl', '8-cl', 'XOR\nτ=1', 'XOR\nτ=2', 'XOR\nτ=3', 'XOR\nτ=5', 'XOR\nτ=10', 'NAR\n3', 'NAR\n5']
neuro = [100.0, 95.6, 52.2, 52.2, 55.6, 58.3, 45.6, 50.0, 43.3]
lif   = [91.7, 93.3, 53.3, 46.7, 51.7, 53.9, 55.0, 50.6, 37.8]
esn   = [91.7, 90.0, 48.3, 48.3, 48.3, 48.3, 50.0, 50.0, 45.0]
x9 = np.arange(len(tasks))
w9 = 0.25

axes[1,1].bar(x9 - w9, neuro, w9, label='Recurrent Silicon', color='#2ecc71', edgecolor='black', linewidth=0.5)
axes[1,1].bar(x9, lif, w9, label='LIF+PRNG', color='#3498db', edgecolor='black', linewidth=0.5)
axes[1,1].bar(x9 + w9, esn, w9, label='Tanh ESN', color='#9b59b6', edgecolor='black', linewidth=0.5)
axes[1,1].axhline(y=50, color='gray', linestyle='--', alpha=0.5, label='Chance (binary)')
axes[1,1].set_ylabel('Accuracy (%)', fontsize=10)
axes[1,1].set_title('(d) Full 9-task comparison', fontsize=11, fontweight='bold')
axes[1,1].set_xticks(x9)
axes[1,1].set_xticklabels(tasks, fontsize=8)
axes[1,1].set_ylim(30, 105)
axes[1,1].legend(fontsize=7, loc='upper right', ncol=2)

plt.tight_layout()
outdir = 'results/FEEL_paper_update/FEEL__Functionally_Embodied_Emergent_Learning__13_-5/figures'
plt.savefig(f'{outdir}/fig_neuromorphic_compute.pdf', bbox_inches='tight', dpi=300)
plt.savefig(f'{outdir}/fig_neuromorphic_compute.png', bbox_inches='tight', dpi=200)
print(f'Saved to {outdir}/fig_neuromorphic_compute.pdf')
