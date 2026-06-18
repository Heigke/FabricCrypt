#!/usr/bin/env python3
"""Generate paper-quality plots for z2090 Deep Self-Optimizing Embodied Intelligence."""

import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# Load results
with open('/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy/results/z2090_deep_self_optimizer.json') as f:
    res = json.load(f)

R = res['results']

OUT = '/tmp/paper_work/figures'
import os
os.makedirs(OUT, exist_ok=True)

# ─── Figure 1: z2090 scorecard (bar chart of all 18 tests) ───
fig, ax = plt.subplots(figsize=(12, 4))
tests = [
    ('T1\nAccuracy', R['T1_accuracy']['pass'] == 'True' or R['T1_accuracy']['pass'] is True),
    ('T2\nAUROC', R['T2_self_awareness']['pass'] is True or R['T2_self_awareness']['pass'] == 'True'),
    ('T3\nGate Sep', R['T3_gate_sep']['pass'] == 'True'),
    ('T4\nEmbodiment', R['T4_embodiment_gap']['pass'] == 'True'),
    ('T5\nAnalog Sig', R['T5_analog_signal']['pass'] == 'True'),
    ('T6\nDelta Sig', R['T6_delta_signal']['pass'] == 'True'),
    ('T7\nPM Deep', R['T7_pm_deep_signal']['pass'] == 'True'),
    ('T8\nSMN Raw', R['T8_smn_raw_signal']['pass'] == 'True'),
    ('T9\nRegime Det', R['T9_regime_detection']['pass'] == 'True'),
    ('T10\nGaslight', R['T10_gaslighting']['pass'] == 'True' or R['T10_gaslighting']['pass'] is True),
    ('T11\nThermal', R['T11_thermal']['pass'] == 'True' or R['T11_thermal']['pass'] is True),
    ('T12\nAttention', R['T12_attention']['pass'] is True or R['T12_attention']['pass'] == 'True'),
    ('T13\nDVFS Kill', R['T13_deep_scramble']['pass'] == 'True'),
    ('T14\nEnergy', R['T14_energy_efficiency']['pass'] is True or R['T14_energy_efficiency']['pass'] == 'True'),
    ('T15\nReg Ablate', R['T15_regime_ablation']['pass'] == 'True'),
    ('T16\nCross-Act', R['T16_cross_actuation']['pass'] is True or R['T16_cross_actuation']['pass'] == 'True'),
    ('T17\nIndepend', R['T17_independence']['pass'] == 'True'),
    ('T18\nReg Gate', R['T18_regime_gate']['pass'] == 'True'),
]
names = [t[0] for t in tests]
passes = [t[1] for t in tests]
colors = ['#2ecc71' if p else '#e74c3c' for p in passes]
bars = ax.bar(range(len(tests)), [1]*len(tests), color=colors, edgecolor='white', linewidth=0.5)
ax.set_xticks(range(len(tests)))
ax.set_xticklabels(names, fontsize=7, ha='center')
ax.set_yticks([])
ax.set_title(f'z2090 Deep Self-Optimizing Embodied Intelligence: {sum(passes)}/{len(passes)} PASS', fontsize=13, fontweight='bold')
ax.set_xlim(-0.6, len(tests)-0.4)
for i, p in enumerate(passes):
    ax.text(i, 0.5, 'PASS' if p else 'FAIL', ha='center', va='center',
            fontsize=7, fontweight='bold', color='white')
plt.tight_layout()
plt.savefig(f'{OUT}/fig_z2090_scorecard.png', dpi=200, bbox_inches='tight')
plt.close()

# ─── Figure 2: z2090 key results (3-panel) ───
fig = plt.figure(figsize=(14, 4.5))
gs = GridSpec(1, 3, width_ratios=[1, 1, 1], wspace=0.35)

# Panel 1: DVFS Kill-Shot (the definitive proof)
ax1 = fig.add_subplot(gs[0])
categories = ['Normal\n(correct DVFS)', 'Scrambled\n(wrong DVFS)']
values = [R['T13_deep_scramble']['normal'], R['T13_deep_scramble']['scrambled']]
bars = ax1.bar(categories, values, color=['#2ecc71', '#e74c3c'], width=0.5, edgecolor='white')
ax1.set_ylabel('Accuracy (%)', fontsize=11)
ax1.set_title('T13: DVFS Kill-Shot', fontsize=12, fontweight='bold')
ax1.set_ylim(0, 110)
ax1.axhline(50, color='gray', linestyle='--', alpha=0.5, label='Chance')
ax1.axhline(10, color='gray', linestyle=':', alpha=0.3, label='Random (1/10)')
for bar, val in zip(bars, values):
    ax1.text(bar.get_x() + bar.get_width()/2, val + 2, f'{val:.1f}%',
             ha='center', va='bottom', fontsize=11, fontweight='bold')
# Add the drop annotation
ax1.annotate('', xy=(1, values[1]+5), xytext=(0, values[0]-5),
             arrowprops=dict(arrowstyle='->', color='red', lw=2))
ax1.text(0.5, 55, f'↓ {R["T13_deep_scramble"]["drop_pp"]:.1f}pp', ha='center',
         fontsize=14, fontweight='bold', color='red')

# Panel 2: Dual gate separation
ax2 = fig.add_subplot(gs[1])
gate_data = {
    'Personality\nGate': [R['T3_gate_sep']['mean_A'], R['T3_gate_sep']['mean_B']],
    'Regime\nGate': [R['T18_regime_gate']['mean_low'], R['T18_regime_gate']['mean_high']],
}
x = np.arange(2)
width = 0.3
bars1 = ax2.bar(x - width/2, [gate_data['Personality\nGate'][0], gate_data['Regime\nGate'][0]],
                width, label='State A / Low', color='#3498db', edgecolor='white')
bars2 = ax2.bar(x + width/2, [gate_data['Personality\nGate'][1], gate_data['Regime\nGate'][1]],
                width, label='State B / High', color='#e67e22', edgecolor='white')
ax2.set_xticks(x)
ax2.set_xticklabels(['Personality Gate\n(ISA identity)', 'Regime Gate\n(DVFS state)'], fontsize=9)
ax2.set_ylabel('Gate Value', fontsize=11)
ax2.set_title('Dual-Channel Gate Separation', fontsize=12, fontweight='bold')
ax2.set_ylim(0, 1.15)
ax2.legend(fontsize=8, loc='upper right')
# Add separation annotations
ax2.text(0, 0.55, f'sep={R["T3_gate_sep"]["sep"]:.3f}', ha='center', fontsize=9, fontweight='bold', color='#2c3e50')
ax2.text(1, 0.55, f'sep={R["T18_regime_gate"]["sep"]:.3f}', ha='center', fontsize=9, fontweight='bold', color='#2c3e50')

# Panel 3: Energy efficiency comparison
ax3 = fig.add_subplot(gs[2])
energy_labels = ['Fixed Low\n(600 MHz)', 'Fixed High\n(~2400 MHz)', 'Self-Optimized\n(model DVFS)']
energy_vals = [R['T14_energy_efficiency']['j_per_correct_low'] * 1000,
               R['T14_energy_efficiency']['j_per_correct_high'] * 1000,
               R['T14_energy_efficiency']['j_per_correct_model'] * 1000]
colors_e = ['#95a5a6', '#95a5a6', '#2ecc71']
bars = ax3.bar(energy_labels, energy_vals, color=colors_e, width=0.5, edgecolor='white')
ax3.set_ylabel('mJ / correct prediction', fontsize=11)
ax3.set_title('T14: Energy Efficiency', fontsize=12, fontweight='bold')
for bar, val in zip(bars, energy_vals):
    ax3.text(bar.get_x() + bar.get_width()/2, val + 0.02, f'{val:.2f}',
             ha='center', va='bottom', fontsize=10, fontweight='bold')
ax3.text(2, energy_vals[2] + 0.15, f'+{R["T14_energy_efficiency"]["efficiency_gain_pct"]:.1f}%\nvs best fixed',
         ha='center', fontsize=9, color='#27ae60', fontweight='bold')

plt.savefig(f'{OUT}/fig_z2090_results.png', dpi=200, bbox_inches='tight')
plt.close()

# ─── Figure 3: Channel signal hierarchy (analog t-values) ───
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

# Panel 1: Analog channel t-values
channels = ['freq_est', 'gpu_power', 'gpu_temp', 'df_dram_r', 'df_coherent', 'df_dram_w']
t_vals = [
    R['T5_analog_signal']['per_channel']['freq_est']['t'],
    R['T5_analog_signal']['per_channel']['gpu_power']['t'],
    R['T5_analog_signal']['per_channel']['gpu_temp']['t'],
    R['T5_analog_signal']['per_channel']['df_dram_r']['t'],
    R['T5_analog_signal']['per_channel']['df_coherent']['t'],
    R['T5_analog_signal']['per_channel']['df_dram_w']['t'],
]
colors_ch = ['#2ecc71' if abs(t) > 3 else '#e74c3c' for t in t_vals]
ax1.barh(range(len(channels)), t_vals, color=colors_ch, edgecolor='white')
ax1.set_yticks(range(len(channels)))
ax1.set_yticklabels(channels, fontsize=10)
ax1.axvline(3.0, color='red', linestyle='--', alpha=0.5, label='$t=3$ threshold')
ax1.set_xlabel('t-statistic (regime low vs high)', fontsize=10)
ax1.set_title('T5: Analog Channel DVFS Signal', fontsize=12, fontweight='bold')
ax1.legend(fontsize=8)
# Add t-value labels
for i, t in enumerate(t_vals):
    ax1.text(t + 1, i, f't={t:.1f}', va='center', fontsize=8)

# Panel 2: Deep sensor t-values (PM table + SMN)
pm_channels = list(R['T7_pm_deep_signal']['per_channel'].keys())
pm_t = list(R['T7_pm_deep_signal']['per_channel'].values())
smn_channels = list(R['T8_smn_raw_signal']['per_channel'].keys())
smn_t = list(R['T8_smn_raw_signal']['per_channel'].values())

all_ch = pm_channels + smn_channels
all_t = pm_t + smn_t
# Replace NaN with 0
all_t = [0 if (t != t) else t for t in all_t]  # NaN != NaN
colors_deep = []
for i, t in enumerate(all_t):
    if i < len(pm_channels):
        colors_deep.append('#3498db' if abs(t) > 3 else '#bdc3c7')
    else:
        colors_deep.append('#e67e22' if abs(t) > 3 else '#bdc3c7')

ax2.barh(range(len(all_ch)), all_t, color=colors_deep, edgecolor='white')
ax2.set_yticks(range(len(all_ch)))
ax2.set_yticklabels(all_ch, fontsize=8)
ax2.axvline(3.0, color='red', linestyle='--', alpha=0.5, label='$t=3$ threshold')
ax2.set_xlabel('t-statistic (regime low vs high)', fontsize=10)
ax2.set_title('T7/T8: Deep Below-Firmware Signal', fontsize=12, fontweight='bold')
# Label PM vs SMN
ax2.axhline(len(pm_channels)-0.5, color='gray', linestyle='-', alpha=0.3)
ax2.text(max(all_t)*0.8, len(pm_channels)/2-0.5, 'PM table', fontsize=8, color='#3498db', fontweight='bold')
ax2.text(max(all_t)*0.8, len(pm_channels)+len(smn_channels)/2-0.5, 'SMN raw', fontsize=8, color='#e67e22', fontweight='bold')
ax2.legend(fontsize=8)

plt.tight_layout()
plt.savefig(f'{OUT}/fig_z2090_channels.png', dpi=200, bbox_inches='tight')
plt.close()

# ─── Figure 4: Attention distribution ───
fig, ax = plt.subplots(figsize=(8, 4))
attn = R['T12_attention']['token_attention']
tokens = list(attn.keys())
values_a = [attn[t] * 100 for t in tokens]  # as percentages
# Color: warm for body-sense, cool for identity/control
token_colors = {
    'delta': '#e74c3c',     # identity
    'analog': '#3498db',    # body-sense
    'energy': '#2ecc71',    # body-sense
    'freq': '#27ae60',      # body-sense
    'intrinsic': '#9b59b6', # body-sense
    'thermal': '#e67e22',   # body-sense
    'pm_deep': '#1abc9c',   # deep body-sense
    'smn_raw': '#16a085',   # deep body-sense
    'status': '#f39c12',    # identity
    'action': '#95a5a6',    # control
}
colors_a = [token_colors.get(t, '#95a5a6') for t in tokens]
bars = ax.bar(tokens, values_a, color=colors_a, edgecolor='white', width=0.7)
ax.set_ylabel('Attention Weight (%)', fontsize=11)
ax.set_title('T12: Transformer Attention Distribution (10 Tokens)', fontsize=12, fontweight='bold')
ax.set_xticklabels(tokens, rotation=30, ha='right', fontsize=9)
for bar, val in zip(bars, values_a):
    ax.text(bar.get_x() + bar.get_width()/2, val + 0.3, f'{val:.1f}%',
            ha='center', va='bottom', fontsize=8)
# Add category labels
ax.axhline(100/10, color='gray', linestyle='--', alpha=0.3, label='Uniform (10%)')
ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig(f'{OUT}/fig_z2090_attention.png', dpi=200, bbox_inches='tight')
plt.close()

# ─── Figure 5: Evolution plot z2060→z2090 ───
fig, ax = plt.subplots(figsize=(10, 4.5))
exps = ['z2060', 'z2061', 'z2067', 'z2068', 'z2069', 'z2076', 'z2080', 'z2081', 'z2084', 'z2085', 'z2086', 'z2087', 'z2088', 'z2089', 'z2090']
pass_n = [8, 12, 17, 21, 21, 12, 12, 15, 12, 11, 11, 12, 12, 12, 17]
total_n = [8, 12, 18, 22, 22, 12, 14, 16, 18, 20, 12, 14, 16, 16, 18]
pct = [p/t*100 for p, t in zip(pass_n, total_n)]

# Highlight z2090
colors_ev = ['#3498db'] * len(exps)
colors_ev[-1] = '#e74c3c'  # z2090 highlighted

ax.bar(range(len(exps)), pct, color=colors_ev, edgecolor='white', width=0.7)
ax.set_xticks(range(len(exps)))
ax.set_xticklabels(exps, rotation=45, ha='right', fontsize=8)
ax.set_ylabel('Pass Rate (%)', fontsize=11)
ax.set_title('FEEL System Evolution: z2060 → z2090', fontsize=12, fontweight='bold')
ax.set_ylim(0, 110)
ax.axhline(90, color='green', linestyle='--', alpha=0.3, label='90% threshold')
for i, (p, t, pctv) in enumerate(zip(pass_n, total_n, pct)):
    ax.text(i, pctv + 2, f'{p}/{t}', ha='center', va='bottom', fontsize=7, fontweight='bold')
ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig(f'{OUT}/fig_z2090_evolution.png', dpi=200, bbox_inches='tight')
plt.close()

print(f"All plots saved to {OUT}/")
print("  fig_z2090_scorecard.png")
print("  fig_z2090_results.png")
print("  fig_z2090_channels.png")
print("  fig_z2090_attention.png")
print("  fig_z2090_evolution.png")
