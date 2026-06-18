#!/usr/bin/env python3
"""z2038_new_plots.py — Publication-quality plots for the FEEL paper.

Generates 6 figures covering z2061 ablation results, self-regulation metrics,
energy savings, SOTA comparison, causal chain diagram, and hardware progression.
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np
import os

OUT_DIR = '/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy/results/figures'
os.makedirs(OUT_DIR, exist_ok=True)

# Use a clean academic style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'figure.dpi': 150,
    'savefig.dpi': 150,
    'savefig.bbox': 'tight',
    'font.family': 'sans-serif',
})


# ──────────────────────────────────────────────────────────────────────────────
# 1. fig_z2061_ablation.png — Bar chart of ablation results
# ──────────────────────────────────────────────────────────────────────────────
def plot_ablation():
    fig, ax = plt.subplots(figsize=(10, 6))

    labels = [
        'A\nAllostatic',
        'B\nBlind',
        'E\nScrambled',
        'F\nNo self-model',
        'G\nNo effort',
        'H\nAlways high',
    ]
    values = [99.2, 60.0, 20.1, 59.0, 64.9, 62.9]
    colors = ['#2ca02c', '#888888', '#d62728', '#ff7f0e', '#9467bd', '#1f77b4']

    bars = ax.bar(labels, values, color=colors, edgecolor='black', linewidth=0.8,
                  width=0.65, zorder=3)

    # Chance line
    ax.axhline(y=10, color='black', linestyle='--', linewidth=1.0, alpha=0.6, zorder=2)
    ax.text(5.45, 11.5, 'chance (10%)', fontsize=10, ha='right', style='italic',
            color='#444444')

    # Value labels on bars
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5,
                f'{val:.1f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')

    # Gap annotations
    # A-F gap
    ax.annotate('', xy=(3, 99.2), xytext=(3, 59.0),
                arrowprops=dict(arrowstyle='<->', color='#cc0000', lw=2))
    ax.text(3.55, 79, 'A\u2212F = 40.2 pp', fontsize=10, color='#cc0000', fontweight='bold',
            ha='left', va='center')

    # A-G gap
    ax.annotate('', xy=(4, 99.2), xytext=(4, 64.9),
                arrowprops=dict(arrowstyle='<->', color='#7b2d8e', lw=2))
    ax.text(4.55, 82, 'A\u2212G = 34.3 pp', fontsize=10, color='#7b2d8e', fontweight='bold',
            ha='left', va='center')

    ax.set_ylabel('Accuracy (%)')
    ax.set_ylim(0, 115)
    ax.set_title('z2061: Closed-Loop Allostatic Ablation', fontsize=14, fontweight='bold')
    ax.set_axisbelow(True)

    fig.tight_layout()
    path = os.path.join(OUT_DIR, 'fig_z2061_ablation.png')
    fig.savefig(path)
    plt.close(fig)
    print(f'Saved {path}')


# ──────────────────────────────────────────────────────────────────────────────
# 2. fig_z2061_gate_effort.png — Two-panel: gate adaptation + closed-loop control
# ──────────────────────────────────────────────────────────────────────────────
def plot_gate_effort():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Left: Gate values
    gate_labels = ['High state', 'Low state']
    gate_values = [0.895, 0.293]
    gate_colors = ['#2ca02c', '#d62728']

    bars1 = ax1.bar(gate_labels, gate_values, color=gate_colors, edgecolor='black',
                    linewidth=0.8, width=0.5, zorder=3)
    ax1.axhline(y=0.5, color='black', linestyle='--', linewidth=1.0, alpha=0.6, zorder=2)
    ax1.text(1.35, 0.52, 'threshold (0.5)', fontsize=10, ha='right', style='italic',
             color='#444444')

    for bar, val in zip(bars1, gate_values):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                 f'{val:.3f}', ha='center', va='bottom', fontsize=12, fontweight='bold')

    ax1.set_ylabel('Gate Value')
    ax1.set_ylim(0, 1.1)
    ax1.set_title('Gate Adaptation', fontsize=14, fontweight='bold')

    # Right: Effort accuracy and temporal correlation
    ctrl_labels = ['Effort Accuracy', 'Temporal Correlation']
    ctrl_values = [100.0, 100.0]
    ctrl_colors = ['#1f77b4', '#ff7f0e']

    bars2 = ax2.bar(ctrl_labels, ctrl_values, color=ctrl_colors, edgecolor='black',
                    linewidth=0.8, width=0.5, zorder=3)

    for bar, val in zip(bars2, ctrl_values):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5,
                 f'{val:.0f}%', ha='center', va='bottom', fontsize=12, fontweight='bold')

    ax2.set_ylabel('Score (%)')
    ax2.set_ylim(0, 115)
    ax2.set_title('Closed-Loop Control', fontsize=14, fontweight='bold')

    fig.suptitle('z2061: Self-Regulation Metrics', fontsize=15, fontweight='bold', y=1.02)
    fig.tight_layout()
    path = os.path.join(OUT_DIR, 'fig_z2061_gate_effort.png')
    fig.savefig(path)
    plt.close(fig)
    print(f'Saved {path}')


# ──────────────────────────────────────────────────────────────────────────────
# 3. fig_z2061_energy.png — Stacked bar showing DVFS distribution + savings
# ──────────────────────────────────────────────────────────────────────────────
def plot_energy():
    fig, ax = plt.subplots(figsize=(10, 6))

    categories = ['A: Model-Controlled\n(Allostatic)', 'H: Always High\n(Baseline)']
    high_pct = [43.6, 100.0]
    low_pct = [56.4, 0.0]

    x = np.arange(len(categories))
    width = 0.45

    bars_high = ax.bar(x, high_pct, width, label='High SCLK', color='#d62728',
                       edgecolor='black', linewidth=0.8, zorder=3)
    bars_low = ax.bar(x, low_pct, width, bottom=high_pct, label='Low SCLK',
                      color='#2ca02c', edgecolor='black', linewidth=0.8, zorder=3)

    # Annotate segments
    ax.text(0, 43.6 / 2, '43.6%\nhigh', ha='center', va='center', fontsize=11,
            fontweight='bold', color='white')
    ax.text(0, 43.6 + 56.4 / 2, '56.4%\nlow', ha='center', va='center', fontsize=11,
            fontweight='bold', color='white')
    ax.text(1, 50, '100%\nhigh', ha='center', va='center', fontsize=11,
            fontweight='bold', color='white')

    # Energy savings annotation
    ax.annotate('56% energy savings',
                xy=(0.5, 85), fontsize=14, fontweight='bold', color='#2ca02c',
                ha='center', va='center',
                bbox=dict(boxstyle='round,pad=0.4', facecolor='#e6ffe6',
                          edgecolor='#2ca02c', linewidth=2))

    # Arrow from annotation to the green segment
    ax.annotate('', xy=(0, 72), xytext=(0.35, 82),
                arrowprops=dict(arrowstyle='->', color='#2ca02c', lw=2))

    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.set_ylabel('DVFS Time Distribution (%)')
    ax.set_ylim(0, 110)
    ax.set_title('z2061: Energy-Aware DVFS Control', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', framealpha=0.9)
    ax.set_axisbelow(True)

    fig.tight_layout()
    path = os.path.join(OUT_DIR, 'fig_z2061_energy.png')
    fig.savefig(path)
    plt.close(fig)
    print(f'Saved {path}')


# ──────────────────────────────────────────────────────────────────────────────
# 4. fig_sota_comparison.png — Grouped bar comparing energy savings to SOTA
# ──────────────────────────────────────────────────────────────────────────────
def plot_sota_comparison():
    fig, ax = plt.subplots(figsize=(10, 6))

    methods = ['Zeus\n(Google)', 'LithOS\n(MIT)', 'mu-Serve\n(UMich)', 'FEEL z2061\n(Ours)']
    savings = [45, 26, 61, 56]
    errors_low = [45 - 15, 0, 0, 0]   # lower error
    errors_high = [75 - 45, 0, 0, 0]  # upper error
    yerr = [errors_low, errors_high]

    colors = ['#aaaaaa', '#aaaaaa', '#aaaaaa', '#2ca02c']
    edgecolors = ['#666666', '#666666', '#666666', '#1a7a1a']

    bars = ax.bar(methods, savings, color=colors, edgecolor=edgecolors, linewidth=1.2,
                  width=0.55, zorder=3, yerr=yerr, capsize=6,
                  error_kw=dict(lw=1.5, capthick=1.5, color='#333333'))

    # Value labels
    for bar, val, el, eh in zip(bars, savings, errors_low, errors_high):
        top = val + eh if eh > 0 else val
        label = f'{val}%' if el == 0 and eh == 0 else f'{val}%\n(15\u201375%)'
        ax.text(bar.get_x() + bar.get_width() / 2, top + 2,
                label, ha='center', va='bottom', fontsize=11, fontweight='bold')

    # Annotation for FEEL's unique property
    ax.annotate('Model controls\nits own hardware',
                xy=(3, 56), xytext=(2.1, 80),
                fontsize=11, fontweight='bold', color='#2ca02c',
                ha='center',
                arrowprops=dict(arrowstyle='->', color='#2ca02c', lw=2),
                bbox=dict(boxstyle='round,pad=0.3', facecolor='#e6ffe6',
                          edgecolor='#2ca02c', linewidth=1.5))

    # Accuracy note
    ax.text(0.5, -0.12,
            'FEEL is the only approach where the neural network controls its own hardware \u2014 '
            'with zero accuracy loss.',
            transform=ax.transAxes, fontsize=10, ha='center', style='italic',
            color='#555555')

    ax.set_ylabel('Energy Savings (%)')
    ax.set_ylim(0, 100)
    ax.set_title('Energy Savings vs. SOTA GPU Power Management', fontsize=14,
                 fontweight='bold')
    ax.set_axisbelow(True)

    fig.tight_layout()
    path = os.path.join(OUT_DIR, 'fig_sota_comparison.png')
    fig.savefig(path)
    plt.close(fig)
    print(f'Saved {path}')


# ──────────────────────────────────────────────────────────────────────────────
# 5. fig_z2061_causal_chain.png — Diagram of causal chain
# ──────────────────────────────────────────────────────────────────────────────
def plot_causal_chain():
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.set_xlim(-0.5, 8.5)
    ax.set_ylim(-1.5, 2.0)
    ax.axis('off')

    nodes = [
        ('Demand\nCue', 0),
        ('Effort\nHead', 1),
        ('DVFS', 2),
        ('SCLK', 3),
        ('Timing', 4),
        ('Self\nModel', 5),
        ('Gate', 6),
        ('Accuracy', 7),
    ]

    # Node categories for coloring
    node_colors = [
        '#b3cde3',  # demand cue — light blue (input)
        '#fbb4ae',  # effort head — light red (neural)
        '#ccebc5',  # DVFS — light green (hardware)
        '#ccebc5',  # SCLK — light green (hardware)
        '#decbe4',  # timing — light purple (measurement)
        '#fbb4ae',  # self model — light red (neural)
        '#fed9a6',  # gate — light orange (neural)
        '#ffffcc',  # accuracy — light yellow (outcome)
    ]

    edge_colors = [
        '#666666',  # demand->effort
        '#2ca02c',  # effort->DVFS (causal action)
        '#2ca02c',  # DVFS->SCLK
        '#666666',  # SCLK->timing
        '#666666',  # timing->self-model
        '#666666',  # self-model->gate
        '#666666',  # gate->accuracy
    ]

    box_width = 0.82
    box_height = 0.9

    # Draw boxes
    for (label, x), color in zip(nodes, node_colors):
        rect = FancyBboxPatch(
            (x - box_width / 2, -box_height / 2),
            box_width, box_height,
            boxstyle="round,pad=0.08",
            facecolor=color,
            edgecolor='#333333',
            linewidth=1.5,
            zorder=3,
        )
        ax.add_patch(rect)
        ax.text(x, 0, label, ha='center', va='center', fontsize=10,
                fontweight='bold', zorder=4)

    # Draw arrows between boxes
    for i in range(len(nodes) - 1):
        x_start = nodes[i][1] + box_width / 2 + 0.02
        x_end = nodes[i + 1][1] - box_width / 2 - 0.02
        arrow = FancyArrowPatch(
            (x_start, 0), (x_end, 0),
            arrowstyle='->', mutation_scale=18,
            color=edge_colors[i], linewidth=2.5,
            zorder=2,
        )
        ax.add_patch(arrow)

    # Category legend
    legend_items = [
        ('Input', '#b3cde3'),
        ('Neural', '#fbb4ae'),
        ('Hardware', '#ccebc5'),
        ('Measurement', '#decbe4'),
        ('Control', '#fed9a6'),
        ('Outcome', '#ffffcc'),
    ]
    for i, (label, color) in enumerate(legend_items):
        ax.add_patch(FancyBboxPatch(
            (i * 1.4 + 0.2, -1.3), 0.3, 0.3,
            boxstyle="round,pad=0.03", facecolor=color,
            edgecolor='#333333', linewidth=0.8, zorder=3))
        ax.text(i * 1.4 + 0.7, -1.15, label, fontsize=9, va='center', ha='left')

    # Feedback arc (gate -> effort, conceptual)
    ax.annotate('', xy=(1, -box_height / 2 - 0.15), xytext=(6, -box_height / 2 - 0.15),
                arrowprops=dict(arrowstyle='->', color='#d62728', lw=1.5,
                                connectionstyle='arc3,rad=0.35', linestyle='--'))
    ax.text(3.5, -1.0, 'feedback (next step)', fontsize=9, ha='center',
            color='#d62728', style='italic')

    ax.set_title('z2061: Closed-Loop Causal Chain', fontsize=14, fontweight='bold',
                 pad=15)

    fig.tight_layout()
    path = os.path.join(OUT_DIR, 'fig_z2061_causal_chain.png')
    fig.savefig(path)
    plt.close(fig)
    print(f'Saved {path}')


# ──────────────────────────────────────────────────────────────────────────────
# 6. fig_hw_progression.png — Line plot of pass rates z2050-z2061
# ──────────────────────────────────────────────────────────────────────────────
def plot_hw_progression():
    fig, ax = plt.subplots(figsize=(10, 6))

    experiments = [
        'z2050', 'z2051', 'z2052', 'z2053', 'z2054',
        'z2055', 'z2056', 'z2057', 'z2058', 'z2059',
        'z2060', 'z2061',
    ]
    passed = [5, 7, 7, 4, 6, 5, 4, 8, 7, 7, 8, 12]
    total  = [5, 7, 8, 7, 7, 8, 8, 8, 8, 8, 8, 12]
    rates = [p / t * 100 for p, t in zip(passed, total)]

    x = np.arange(len(experiments))

    # Line
    ax.plot(x, rates, color='#333333', linewidth=2, zorder=2, marker='o',
            markersize=0)

    # Scatter with conditional coloring
    for i, (exp, rate, p, t) in enumerate(zip(experiments, rates, passed, total)):
        is_perfect = (p == t)
        color = '#2ca02c' if is_perfect else '#ff7f0e'
        edge = '#1a7a1a' if is_perfect else '#cc6600'
        ax.scatter(i, rate, color=color, edgecolors=edge, s=120, linewidths=1.5,
                   zorder=4)

    # Annotate z2061 and z2060
    ax.annotate('12/12 PASS', xy=(11, 100), xytext=(10.2, 90),
                fontsize=10, fontweight='bold', color='#2ca02c',
                arrowprops=dict(arrowstyle='->', color='#2ca02c', lw=1.5),
                ha='center')
    ax.annotate('8/8 PASS', xy=(10, 100), xytext=(9.2, 82),
                fontsize=10, fontweight='bold', color='#2ca02c',
                arrowprops=dict(arrowstyle='->', color='#2ca02c', lw=1.5),
                ha='center')

    # Annotate z2056 (worst)
    ax.annotate('4/8', xy=(6, 50), xytext=(6, 40),
                fontsize=10, fontweight='bold', color='#ff7f0e',
                ha='center')

    ax.set_xticks(x)
    ax.set_xticklabels(experiments, rotation=45, ha='right')
    ax.set_ylabel('Pass Rate (%)')
    ax.set_ylim(30, 110)
    ax.set_title('Hardware Integration Pass Rate Progression', fontsize=14,
                 fontweight='bold')

    # Legend
    perfect_patch = mpatches.Patch(color='#2ca02c', label='Perfect score')
    partial_patch = mpatches.Patch(color='#ff7f0e', label='Partial pass')
    ax.legend(handles=[perfect_patch, partial_patch], loc='lower right',
              framealpha=0.9)

    ax.set_axisbelow(True)
    fig.tight_layout()
    path = os.path.join(OUT_DIR, 'fig_hw_progression.png')
    fig.savefig(path)
    plt.close(fig)
    print(f'Saved {path}')


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print('Generating publication-quality plots...')
    plot_ablation()
    plot_gate_effort()
    plot_energy()
    plot_sota_comparison()
    plot_causal_chain()
    plot_hw_progression()
    print('All figures saved.')
