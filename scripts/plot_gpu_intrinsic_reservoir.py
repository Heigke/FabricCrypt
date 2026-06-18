#!/usr/bin/env python3
"""Generate figures for GPU-intrinsic reservoir section of FEEL paper."""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

fig_dir = '/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy/results/FEEL_paper_update/FEEL__Functionally_Embodied_Emergent_Learning__13_-5/figures'

# ============================================================================
# Figure 1: Noise source hierarchy + reservoir results
# ============================================================================
fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), gridspec_kw={'width_ratios': [1, 1.2, 1]})

# Panel (a): Noise source hierarchy - what's deterministic vs stochastic
ax = axes[0]
sources = [
    'ALU/FPU',
    'LDS',
    'Register file',
    'Transcendentals',
    'DRAM refresh',
    'MC bank conflicts',
    'L1 cache (solo)',
    'L1 cache (thermal)',
    'Memory stride',
    'Cross-PLL drift',
    'wall_clock64'
]
nondet = [0, 0, 0, 0, 0, 0, 0.3, 0.01, 1.2, 100, 47.1]  # % non-deterministic between runs
colors = ['#cccccc']*6 + ['#ffcccc', '#ffcccc', '#ffe0b2', '#ff9800', '#e53935']
barh = ax.barh(range(len(sources)), nondet, color=colors, edgecolor='#333', linewidth=0.5)
ax.set_yticks(range(len(sources)))
ax.set_yticklabels(sources, fontsize=8)
ax.set_xlabel('Non-deterministic (%)', fontsize=9)
ax.set_title('(a) Noise sources in gfx1151', fontsize=10, fontweight='bold')
ax.axvline(x=1, color='red', linestyle='--', alpha=0.5, linewidth=0.8)
ax.text(1.5, 3.5, 'deterministic\nthreshold', fontsize=7, color='red', alpha=0.7)
ax.set_xscale('symlog', linthresh=1)
ax.set_xlim(-0.5, 120)

# Panel (b): Reservoir classification comparison — all experiments
ax = axes[1]
experiments = ['Wallclock\n4-class\n(64N)', 'Deep PLL\n6-class\n(768N)', 'Placement\n4-class\n(768N)', 'Deep\nControl', 'Place.\nControl']
accs = [100.0, 89.2, 90.4, 47.1, 87.5]
chances = [25.0, 16.7, 25.0, 16.7, 25.0]
bar_colors = ['#e53935', '#d32f2f', '#ff9800', '#bdbdbd', '#bdbdbd']
bars = ax.bar(range(len(experiments)), accs, color=bar_colors, edgecolor='#333', linewidth=0.5)
ax.axhline(y=16.7, color='black', linestyle='--', alpha=0.3, linewidth=0.8)
ax.text(4.4, 18, '6-class chance', fontsize=6, ha='right', alpha=0.5)
ax.axhline(y=25.0, color='black', linestyle=':', alpha=0.3, linewidth=0.8)
ax.text(4.4, 26.5, '4-class chance', fontsize=6, ha='right', alpha=0.5)
ax.set_xticks(range(len(experiments)))
ax.set_xticklabels(experiments, fontsize=7)
ax.set_ylabel('Classification accuracy (%)', fontsize=9)
ax.set_title('(b) GPU-intrinsic reservoir', fontsize=10, fontweight='bold')
ax.set_ylim(0, 110)
for bar, acc in zip(bars, accs):
    ax.text(bar.get_x() + bar.get_width()/2, acc + 2, f'{acc:.0f}%',
            ha='center', fontsize=8, fontweight='bold')

# Panel (c): Deep PLL memory capacity + depth discrimination
ax = axes[2]
delays = [0, 1, 2, 3, 4, 5, 6, 7]
mc_vals = [0.460, 0.315, 0.365, 0.380, 0.335, 0.400, 0.400, 0.330]
bars = ax.bar(delays, mc_vals, color='#1976d2', edgecolor='#333', linewidth=0.5, alpha=0.8)
ax.set_xlabel('Delay (steps)', fontsize=9)
ax.set_ylabel('MC', fontsize=9)
ax.set_title('(c) Deep PLL memory capacity', fontsize=10, fontweight='bold')
ax.set_ylim(0, 0.7)
ax.axhline(y=1.0/6, color='gray', linestyle=':', alpha=0.5)
ax.text(7.5, 1.0/6+0.01, 'chance', fontsize=7, ha='right', alpha=0.5)
ax.text(0.5, 0.62, f'Total MC = {sum(mc_vals):.2f}', fontsize=8, fontweight='bold', color='#1976d2')
ax.annotate('Depth discrimination:\n100% (d=8 vs d=128)',
            xy=(5.5, 0.55), fontsize=7, fontweight='bold',
            ha='center', va='top',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#c8e6c9', edgecolor='#388e3c'))

plt.tight_layout()
plt.savefig(f'{fig_dir}/fig_gpu_intrinsic_reservoir.pdf', dpi=300, bbox_inches='tight')
plt.savefig(f'{fig_dir}/fig_gpu_intrinsic_reservoir.png', dpi=300, bbox_inches='tight')
print(f"Saved figure to {fig_dir}/fig_gpu_intrinsic_reservoir.pdf")

# ============================================================================
# Figure 2: Per-CU timing fingerprint
# ============================================================================
fig2, axes2 = plt.subplots(1, 2, figsize=(10, 4))

# Panel (a): Per-CU timing (from probe results)
ax = axes2[0]
cu_ids = list(range(12))
run1 = [231168, 231154, 231162, 231147, 231159, 231144, 231155, 231140, 231153, 231138, 231167, 231159]
run2 = [230330, 230339, 230327, 230336, 230323, 230332, 230321, 230624, 230608, 230622, 230606, 230615]
x = np.arange(12)
w = 0.35
ax.bar(x - w/2, [r - 230000 for r in run1], w, label='Run 1', color='#1976d2', edgecolor='#333', linewidth=0.3)
ax.bar(x + w/2, [r - 230000 for r in run2], w, label='Run 2', color='#e53935', edgecolor='#333', linewidth=0.3)
ax.set_xlabel('Compute Unit ID', fontsize=9)
ax.set_ylabel('Cycles (offset by 230000)', fontsize=9)
ax.set_title('(a) Per-CU FMA timing (10k iters)', fontsize=10, fontweight='bold')
ax.set_xticks(x)
ax.legend(fontsize=8)
ax.text(0.5, 0.95, 'std = 9.5 cycles\n0% identical between runs',
        transform=ax.transAxes, fontsize=8, va='top', ha='center',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

# Panel (b): Cross-CU communication delays
ax = axes2[1]
delays = [828, 838, 828, 837, 834, 837, 834, 836, 832, 835, 831, 834]
colors_delay = plt.cm.RdYlBu_r(np.linspace(0.2, 0.8, 12))
ax.bar(range(12), delays, color=colors_delay, edgecolor='#333', linewidth=0.5)
ax.set_xlabel('Block ID', fontsize=9)
ax.set_ylabel('Cycles to partner', fontsize=9)
ax.set_title('(b) Cross-CU communication delay', fontsize=10, fontweight='bold')
ax.set_ylim(820, 845)
ax.axhline(y=np.mean(delays), color='black', linestyle='--', alpha=0.5)
ax.text(11, np.mean(delays)+0.5, f'mean={np.mean(delays):.0f}', fontsize=8, ha='right')

plt.tight_layout()
plt.savefig(f'{fig_dir}/fig_cu_fingerprint.pdf', dpi=300, bbox_inches='tight')
plt.savefig(f'{fig_dir}/fig_cu_fingerprint.png', dpi=300, bbox_inches='tight')
print(f"Saved figure to {fig_dir}/fig_cu_fingerprint.pdf")
