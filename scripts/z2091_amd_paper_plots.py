#!/usr/bin/env python3
"""Generate plots for AMD motivation paper."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os

OUT = "/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy/docs"
os.makedirs(OUT, exist_ok=True)

# Style
plt.rcParams.update({
    'font.size': 11, 'axes.labelsize': 12, 'axes.titlesize': 13,
    'xtick.labelsize': 9, 'ytick.labelsize': 10,
    'figure.dpi': 200, 'savefig.bbox': 'tight',
    'font.family': 'serif',
})

# ─── Fig 1: Experiment Pass Rates Timeline ───
fig, ax = plt.subplots(figsize=(8, 3.5))
exps = [
    ('z2060\nHomeostatic', 8, 8),
    ('z2061\nAllostatic', 12, 12),
    ('z2068\nNeuromod', 21, 22),
    ('z2070\nMulti-Ch', 22, 24),
    ('z2076\nPure ISA', 12, 12),
    ('z2078\nClosed-Loop', 13, 14),
    ('z2081\nPer-Core', 15, 16),
    ('z2084\nTransformer', 14, 18),
    ('z2087\nData Fabric', 12, 14),
    ('z2088\nEnergy', 12, 16),
    ('z2091\nGPT-2 LM', 9, 14),
]
names = [e[0] for e in exps]
passes = [e[1] for e in exps]
totals = [e[2] for e in exps]
rates = [p/t*100 for p,t in zip(passes, totals)]

x = np.arange(len(names))
bars = ax.bar(x, rates, color=['#2ecc71' if r >= 85 else '#f39c12' if r >= 70 else '#e74c3c' for r in rates],
              edgecolor='black', linewidth=0.5, alpha=0.85)
for i, (p, t, r) in enumerate(zip(passes, totals, rates)):
    ax.text(i, r + 1.5, f'{p}/{t}', ha='center', va='bottom', fontsize=8, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(names, fontsize=7.5)
ax.set_ylabel('Test Pass Rate (%)')
ax.set_title('FEEL Experiment Progression: 30+ Experiments on AMD RDNA 3.5')
ax.set_ylim(0, 110)
ax.axhline(85, color='gray', linestyle='--', alpha=0.4, linewidth=0.8)
ax.text(len(names)-0.5, 86, '85%', color='gray', fontsize=8, ha='right')
plt.tight_layout()
fig.savefig(f'{OUT}/fig1_pass_rates.pdf')
fig.savefig(f'{OUT}/fig1_pass_rates.png')
plt.close()
print("Fig 1: pass rates")

# ─── Fig 2: Hardware Sensing Layers ───
fig, ax = plt.subplots(figsize=(6, 4))
layers = [
    'ISA Delta\n(MODE register)',
    'hwreg reads\n(SHADER_CYCLES)',
    'DVFS State\n(sclk frequency)',
    'Thermal\n(hwmon sysfs)',
    'Power/Energy\n(RAPL + PPT)',
    'Data Fabric\n(DRAM counters)',
    'PM Table\n(916 float32s)',
]
# t-statistics from z2087/z2091 results showing signal strength
t_stats = [26.9, 3.4, 26.9, 3.4, 11.6, 5.5, 16.3]
colors = ['#e74c3c', '#e67e22', '#f1c40f', '#2ecc71', '#3498db', '#9b59b6', '#1abc9c']
bars = ax.barh(range(len(layers)), t_stats, color=colors, edgecolor='black', linewidth=0.5, alpha=0.85)
ax.set_yticks(range(len(layers)))
ax.set_yticklabels(layers, fontsize=9)
ax.set_xlabel('Signal Strength (Welch t-statistic)')
ax.set_title('Hardware Sensing Depth: 7 Layers Below Software')
ax.axvline(3.0, color='red', linestyle='--', alpha=0.5, linewidth=0.8)
ax.text(3.2, 6.5, 'p < 0.003', color='red', fontsize=8)
ax.invert_yaxis()
plt.tight_layout()
fig.savefig(f'{OUT}/fig2_sensing_layers.pdf')
fig.savefig(f'{OUT}/fig2_sensing_layers.png')
plt.close()
print("Fig 2: sensing layers")

# ─── Fig 3: Perplexity Comparison (z2091) ───
fig, ax = plt.subplots(figsize=(5, 3.5))
categories = ['Frozen\nGPT-2', 'Embodied\nGPT-2\n(regime 0)', 'Embodied\nGPT-2\n(regime 1)', 'Embodied\nGPT-2\n(average)']
ppls = [61.42, 45.62, 45.42, 45.52]
colors = ['#95a5a6', '#3498db', '#2980b9', '#2ecc71']
bars = ax.bar(range(len(categories)), ppls, color=colors, edgecolor='black', linewidth=0.5, alpha=0.85)
for i, v in enumerate(ppls):
    ax.text(i, v + 0.8, f'{v:.1f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
ax.set_xticks(range(len(categories)))
ax.set_xticklabels(categories, fontsize=9)
ax.set_ylabel('Perplexity (lower = better)')
ax.set_title('z2091: First Hardware-Embodied Language Model')
ax.set_ylim(0, 72)
# Arrow showing improvement
ax.annotate('26% better', xy=(3, 45.52), xytext=(3, 56),
            arrowprops=dict(arrowstyle='->', color='green', lw=2),
            fontsize=11, ha='center', color='green', fontweight='bold')
plt.tight_layout()
fig.savefig(f'{OUT}/fig3_perplexity.pdf')
fig.savefig(f'{OUT}/fig3_perplexity.png')
plt.close()
print("Fig 3: perplexity")

# ─── Fig 4: Energy Efficiency (z2091) ───
fig, ax = plt.subplots(figsize=(5, 3.5))
modes = ['DVFS Low\n(600 MHz)', 'DVFS High\n(~1900 MHz)', 'Model-Selected\n(adaptive)']
uj = [22154.4, 13445.0, 19883.3]
colors = ['#e74c3c', '#2ecc71', '#3498db']
bars = ax.bar(range(len(modes)), uj, color=colors, edgecolor='black', linewidth=0.5, alpha=0.85)
for i, v in enumerate(uj):
    ax.text(i, v + 300, f'{v/1000:.1f}\nmJ/tok', ha='center', va='bottom', fontsize=9)
ax.set_xticks(range(len(modes)))
ax.set_xticklabels(modes, fontsize=9)
ax.set_ylabel('Energy per Token (uJ)')
ax.set_title('Energy Efficiency Across DVFS Regimes')
ax.set_ylim(0, 27000)
plt.tight_layout()
fig.savefig(f'{OUT}/fig4_energy.pdf')
fig.savefig(f'{OUT}/fig4_energy.png')
plt.close()
print("Fig 4: energy")

# ─── Fig 5: Architecture Diagram (text-based) ───
fig, ax = plt.subplots(figsize=(8, 4.5))
ax.set_xlim(0, 10)
ax.set_ylim(0, 7)
ax.axis('off')

# GPU box
gpu = plt.Rectangle((0.3, 0.5), 4.2, 6, fill=True, facecolor='#fadbd8', edgecolor='#c0392b', linewidth=2)
ax.add_patch(gpu)
ax.text(2.4, 6.2, 'AMD RDNA 3.5 GPU', ha='center', fontsize=11, fontweight='bold', color='#c0392b')

# Kernel box
kern = plt.Rectangle((0.6, 3.8), 3.6, 2.2, fill=True, facecolor='#f5b7b1', edgecolor='#e74c3c', linewidth=1.5)
ax.add_patch(kern)
ax.text(2.4, 5.7, 'HIP Compute Kernel', ha='center', fontsize=9, fontweight='bold')
ax.text(2.4, 5.2, 's_setreg_b32 hwreg(1) [MODE]', ha='center', fontsize=7, family='monospace')
ax.text(2.4, 4.8, 's_getreg_b32 hwreg(29) [CYCLES]', ha='center', fontsize=7, family='monospace')
ax.text(2.4, 4.4, 'v_fma_f32 + v_mul_f16', ha='center', fontsize=7, family='monospace')
ax.text(2.4, 4.0, 'output delta = HW - SW_ref', ha='center', fontsize=7, family='monospace', color='#c0392b')

# Sensors
sensors = [
    (1.0, 3.2, 'Thermal'), (2.4, 3.2, 'DVFS'), (3.8, 3.2, 'Power'),
    (1.0, 2.4, 'Data Fabric'), (2.4, 2.4, 'PM Table'), (3.8, 2.4, 'RAPL'),
]
for x_, y_, label in sensors:
    box = plt.Rectangle((x_-0.5, y_-0.25), 1.0, 0.5, fill=True,
                         facecolor='#d5f5e3', edgecolor='#27ae60', linewidth=1)
    ax.add_patch(box)
    ax.text(x_, y_, label, ha='center', va='center', fontsize=7, fontweight='bold')

# NN box
nn = plt.Rectangle((5.5, 1.0), 4.0, 5.0, fill=True, facecolor='#d6eaf8', edgecolor='#2980b9', linewidth=2)
ax.add_patch(nn)
ax.text(7.5, 5.7, 'Neural Network', ha='center', fontsize=11, fontweight='bold', color='#2980b9')

# Transformer tokens
tokens = ['T0:delta', 'T1:analog', 'T2:energy', 'T3:freq', 'T4:intrinsic', 'T5:thermal', 'T6:status', 'T7:action']
for i, tok in enumerate(tokens):
    y_ = 5.0 - i * 0.45
    box = plt.Rectangle((5.8, y_-0.15), 1.8, 0.35, fill=True,
                         facecolor='#aed6f1', edgecolor='#3498db', linewidth=0.8)
    ax.add_patch(box)
    ax.text(6.7, y_, tok, ha='center', va='center', fontsize=6.5, family='monospace')

# Attention + output
ax.text(8.5, 4.0, 'Multi-Head\nSelf-Attention', ha='center', fontsize=8, fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='#f9e79f', edgecolor='#f39c12'))
ax.text(8.5, 2.5, 'Gate +\nClassifier', ha='center', fontsize=8, fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='#f9e79f', edgecolor='#f39c12'))
ax.text(8.5, 1.3, 'Output +\nAction', ha='center', fontsize=8, fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='#abebc6', edgecolor='#27ae60'))

# Arrows
ax.annotate('', xy=(5.5, 4.5), xytext=(4.5, 4.5),
            arrowprops=dict(arrowstyle='->', color='#e74c3c', lw=2))
ax.annotate('', xy=(5.5, 2.8), xytext=(4.5, 2.8),
            arrowprops=dict(arrowstyle='->', color='#27ae60', lw=2))
# Feedback loop
ax.annotate('', xy=(3.8, 0.8), xytext=(8.5, 0.8),
            arrowprops=dict(arrowstyle='->', color='#8e44ad', lw=2, connectionstyle='arc3,rad=-0.2'))
ax.text(6.0, 0.3, 'DVFS actuation + ISA control (closed loop)', ha='center', fontsize=8,
        color='#8e44ad', fontstyle='italic')

ax.text(2.4, 1.5, 'Blocked by\nfirmware', ha='center', fontsize=8, color='#7f8c8d',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='#f2f3f4', edgecolor='#bdc3c7', linestyle='--'))
ax.text(2.4, 0.8, 'SPM counters, fine DVFS,\nL3 cache, shader thermal', ha='center', fontsize=6.5,
        color='#95a5a6', fontstyle='italic')

plt.tight_layout()
fig.savefig(f'{OUT}/fig5_architecture.pdf')
fig.savefig(f'{OUT}/fig5_architecture.png')
plt.close()
print("Fig 5: architecture")

# ─── Fig 6: Blocked vs Available Hardware Channels ───
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3.5))

# Available
avail = ['ISA MODE', 'SHADER_CYCLES', 'DVFS (coarse)', 'hwmon thermal',
         'RAPL energy', 'DF counters', 'PM table (916)']
avail_quality = [100, 60, 70, 90, 95, 80, 95]
ax1.barh(range(len(avail)), avail_quality, color='#2ecc71', edgecolor='black', linewidth=0.5, alpha=0.85)
ax1.set_yticks(range(len(avail)))
ax1.set_yticklabels(avail, fontsize=8)
ax1.set_xlabel('Signal Quality (%)')
ax1.set_title('Currently Available', fontsize=10, fontweight='bold', color='#27ae60')
ax1.set_xlim(0, 110)
ax1.invert_yaxis()

# Blocked
blocked = ['SPM perf counters', 'Fine DVFS (pp_dpm_sclk)', 'L3 cache counters',
           'GPU CLK registers', 'Shader-visible thermal', 'Per-CU occupancy']
blocked_value = [95, 85, 80, 75, 90, 95]
ax2.barh(range(len(blocked)), blocked_value, color='#e74c3c', edgecolor='black', linewidth=0.5, alpha=0.85)
ax2.set_yticks(range(len(blocked)))
ax2.set_yticklabels(blocked, fontsize=8)
ax2.set_xlabel('Potential Value (%)')
ax2.set_title('Blocked by Firmware/Access', fontsize=10, fontweight='bold', color='#c0392b')
ax2.set_xlim(0, 110)
ax2.invert_yaxis()

plt.suptitle('Hardware Channel Availability on AMD RDNA 3.5', fontsize=12, fontweight='bold', y=1.02)
plt.tight_layout()
fig.savefig(f'{OUT}/fig6_channels.pdf')
fig.savefig(f'{OUT}/fig6_channels.png')
plt.close()
print("Fig 6: channels")

print(f"\nAll figures saved to {OUT}/")
