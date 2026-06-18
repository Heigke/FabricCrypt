#!/usr/bin/env python3
"""Generate publication-quality plots for paper sections 6.3 onward."""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os

OUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'results', 'figures')
os.makedirs(OUT_DIR, exist_ok=True)

# Publication style
plt.rcParams.update({
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.dpi': 200,
    'savefig.dpi': 200,
    'savefig.bbox': 'tight',
    'axes.spines.top': False,
    'axes.spines.right': False,
})

PASS_GREEN = '#2ecc71'
FAIL_RED = '#e74c3c'
BLUE = '#3498db'
ORANGE = '#e67e22'
PURPLE = '#9b59b6'
GRAY = '#95a5a6'
DARK = '#2c3e50'

# ============================================================
# 1. BLINDSIGHT DISSOCIATION (z2021, z2028, z2030)
# ============================================================
def plot_blindsight():
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Data from JSONs
    architectures = ['CNN\n(z2021)', 'ResNet\n(z2028)', 'ViT\n(z2030)']

    # Task accuracy: full vs ablated
    task_full = [0.7065, 0.8946, 0.7097]
    task_ablated = [0.7106, 0.8946, 0.7094]

    # AUROC: full vs ablated
    auroc_full = [0.966, 0.903, 0.913]
    auroc_ablated = [0.500, 0.500, 0.500]
    auroc_scrambled = [0.491, 0.514, 0.497]
    auroc_encoder_abl = [0.478, 0.502, 0.470]

    x = np.arange(len(architectures))
    w = 0.18

    # Left panel: AUROC under different conditions
    ax = axes[0]
    bars1 = ax.bar(x - 1.5*w, auroc_full, w, label='Full model', color=BLUE, zorder=3)
    bars2 = ax.bar(x - 0.5*w, auroc_ablated, w, label='Self-model ablated', color=FAIL_RED, zorder=3)
    bars3 = ax.bar(x + 0.5*w, auroc_scrambled, w, label='Scrambled', color=ORANGE, zorder=3)
    bars4 = ax.bar(x + 1.5*w, auroc_encoder_abl, w, label='Encoder ablated', color=GRAY, zorder=3)

    ax.axhline(y=0.5, color='black', linestyle='--', alpha=0.3, label='Chance')
    ax.set_ylabel('Metacognitive AUROC')
    ax.set_title('Blindsight: AUROC Collapses Under Self-Model Ablation')
    ax.set_xticks(x)
    ax.set_xticklabels(architectures)
    ax.set_ylim(0, 1.05)
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(axis='y', alpha=0.3)

    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., h + 0.01,
                    f'{h:.2f}', ha='center', va='bottom', fontsize=8)

    # Right panel: Task accuracy preserved
    ax = axes[1]
    bars1 = ax.bar(x - 0.2, task_full, 0.35, label='Full model', color=BLUE, zorder=3)
    bars2 = ax.bar(x + 0.2, task_ablated, 0.35, label='Self-model ablated', color=FAIL_RED, zorder=3)

    ax.set_ylabel('Task Accuracy')
    ax.set_title('Task Performance Preserved Under Ablation')
    ax.set_xticks(x)
    ax.set_xticklabels(architectures)
    ax.set_ylim(0.5, 1.0)
    ax.legend(loc='lower right')
    ax.grid(axis='y', alpha=0.3)

    for bars in [bars1, bars2]:
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., h + 0.005,
                    f'{h:.1%}', ha='center', va='bottom', fontsize=9)

    plt.suptitle('Synthetic Blindsight: Cross-Architecture Replication (z2021/z2028/z2030)',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'fig_blindsight_dissociation.png'))
    plt.close()
    print('  [+] fig_blindsight_dissociation.png')


# ============================================================
# 2. PHENOMENAL OVERFLOW (z2026)
# ============================================================
def plot_overflow():
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))

    conditions = ['A: Narrow WS\n(16d, 8 items)', 'B: Wide WS\n(128d, 8 items)',
                  'C: No WS\n(8 items)', 'D: Narrow WS\n(16d, 16 items)']
    report_acc = [0.2956, 0.2930, 0.9850, 0.2274]
    probe_acc = [0.9772, 0.9784, 0.9852, 0.9782]

    x = np.arange(len(conditions))
    w = 0.35

    # Left: encoder vs workspace accuracy
    ax = axes[0]
    bars1 = ax.bar(x - w/2, probe_acc, w, label='Encoder (probe)', color=BLUE, zorder=3)
    bars2 = ax.bar(x + w/2, report_acc, w, label='Workspace (report)', color=ORANGE, zorder=3)

    ax.set_ylabel('Accuracy')
    ax.set_title('Phenomenal Overflow: Encoder vs Workspace')
    ax.set_xticks(x)
    ax.set_xticklabels(conditions, fontsize=9)
    ax.set_ylim(0, 1.15)
    ax.legend(loc='upper right')
    ax.grid(axis='y', alpha=0.3)

    for bars in [bars1, bars2]:
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., h + 0.01,
                    f'{h:.1%}', ha='center', va='bottom', fontsize=9)

    # Annotate the 68% gap
    ax.annotate('68% gap', xy=(0, 0.64), fontsize=12, fontweight='bold',
                color=FAIL_RED, ha='center',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))

    # Right: overflow gap per condition
    ax = axes[1]
    gaps = [p - r for p, r in zip(probe_acc, report_acc)]
    colors = [FAIL_RED if g > 0.05 else PASS_GREEN for g in gaps]
    bars = ax.bar(x, gaps, 0.6, color=colors, zorder=3)
    ax.axhline(y=0, color='black', linewidth=0.5)
    ax.set_ylabel('Overflow Gap (Probe - Report)')
    ax.set_title('Capacity Limitation: Gap Size')
    ax.set_xticks(x)
    ax.set_xticklabels(conditions, fontsize=9)
    ax.grid(axis='y', alpha=0.3)

    for bar, gap in zip(bars, gaps):
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., h + 0.01 if h > 0 else h - 0.04,
                f'{gap:.1%}', ha='center', va='bottom' if h > 0 else 'top',
                fontsize=10, fontweight='bold')

    plt.suptitle('z2026: Phenomenal Overflow (Block 2011) — 4/4 PASS',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'fig_overflow_gap.png'))
    plt.close()
    print('  [+] fig_overflow_gap.png')


# ============================================================
# 3. INFORMATION SYNERGY PID (z2027)
# ============================================================
def plot_synergy():
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))

    conditions = ['A: WS (32d)', 'B: No WS', 'C: Wide WS\n(128d)', 'D: Random']
    redundancy = [0.1665, 0.0710, 0.0217, 0.0267]
    unique_b = [0.0486, 0.0736, 0.1369, 0.0112]
    synergy = [0.1016, 0.0687, 0.0505, 0.0436]
    total_mi = [0.3168, 0.2133, 0.2092, 0.0814]
    syn_ratio = [0.321, 0.322, 0.241, 0.535]

    x = np.arange(len(conditions))
    w = 0.6

    # Left: Stacked bar PID decomposition
    ax = axes[0]
    b1 = ax.bar(x, redundancy, w, label='Redundancy', color=GRAY, zorder=3)
    b2 = ax.bar(x, unique_b, w, bottom=redundancy, label='Unique', color=BLUE, zorder=3)
    bottoms = [r + u for r, u in zip(redundancy, unique_b)]
    b3 = ax.bar(x, synergy, w, bottom=bottoms, label='Synergy', color=PASS_GREEN, zorder=3)

    ax.set_ylabel('Mutual Information (bits)')
    ax.set_title('PID Decomposition of Workspace')
    ax.set_xticks(x)
    ax.set_xticklabels(conditions)
    ax.legend(loc='upper right')
    ax.grid(axis='y', alpha=0.3)

    # Add total MI labels
    for i, mi in enumerate(total_mi):
        ax.text(i, mi + 0.005, f'{mi:.3f}', ha='center', va='bottom',
                fontsize=9, fontweight='bold')

    # Right: Synergy absolute values
    ax = axes[1]
    colors = [PASS_GREEN, BLUE, ORANGE, GRAY]
    bars = ax.bar(x, synergy, 0.6, color=colors, zorder=3)
    ax.set_ylabel('Synergy (bits)')
    ax.set_title('Synergistic Information')
    ax.set_xticks(x)
    ax.set_xticklabels(conditions)
    ax.grid(axis='y', alpha=0.3)

    for bar, s, sr in zip(bars, synergy, syn_ratio):
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., h + 0.002,
                f'{s:.3f}\n({sr:.0%})', ha='center', va='bottom', fontsize=9)

    # Annotation
    ax.annotate('Workspace forces\nsynergistic integration',
                xy=(0, synergy[0]), xytext=(1.5, 0.11),
                fontsize=9, ha='center',
                arrowprops=dict(arrowstyle='->', color=DARK),
                bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow'))

    plt.suptitle('z2027: Information Synergy (Luppi et al. 2024) — 4/4 PASS',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'fig_synergy_pid.png'))
    plt.close()
    print('  [+] fig_synergy_pid.png')


# ============================================================
# 4. PCI INVERSION (z2023)
# ============================================================
def plot_pci_inversion():
    fig, ax = plt.subplots(figsize=(8, 5))

    conditions = ['A: Trained\n+ WS', 'B: Trained\nno WS', 'C: Random\n+ WS', 'D: Random\nno WS']
    pci = [0.871, 0.719, 1.026, 1.042]
    pci_shuffled = [1.048, 1.044, 1.045, 1.044]

    x = np.arange(len(conditions))
    w = 0.35

    bars1 = ax.bar(x - w/2, pci, w, label='Actual PCI', color=[BLUE, BLUE, GRAY, GRAY], zorder=3)
    bars2 = ax.bar(x + w/2, pci_shuffled, w, label='Shuffled PCI', color=[ORANGE]*4, alpha=0.7, zorder=3)

    # Clinical threshold
    ax.axhline(y=0.31, color=FAIL_RED, linestyle='--', linewidth=2,
               label='Clinical threshold (PCI* = 0.31)')
    ax.axhline(y=1.0, color=GRAY, linestyle=':', linewidth=1, alpha=0.5)

    ax.set_ylabel('PCI (Lempel-Ziv complexity)')
    ax.set_title('z2023: PCI INVERTS — Training Reduces Complexity', fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(conditions)
    ax.set_ylim(0, 1.15)
    ax.legend(loc='upper left')
    ax.grid(axis='y', alpha=0.3)

    for bar in bars1:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., h + 0.01,
                f'{h:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

    # Arrow showing inversion
    ax.annotate('TRAINING\nREDUCES PCI',
                xy=(0.5, 0.79), xytext=(2.2, 0.6),
                fontsize=11, fontweight='bold', color=FAIL_RED,
                arrowprops=dict(arrowstyle='->', color=FAIL_RED, lw=2),
                bbox=dict(boxstyle='round,pad=0.4', facecolor='#ffdddd'))

    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'fig_pci_inversion.png'))
    plt.close()
    print('  [+] fig_pci_inversion.png')


# ============================================================
# 5. WORKSPACE NECESSITY (z2037)
# ============================================================
def plot_workspace_necessity():
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    tasks = ['Simple', 'Composite', 'Triple']
    normal = [0.9864, 0.9851, 0.9901]
    zero = [0.4926, 0.4077, 0.6977]
    random = [0.500, 0.4962, 0.5153]
    frozen = [0.5074, 0.5923, 0.6977]
    noisy = [0.9823, 0.9749, 0.9866]

    x = np.arange(len(tasks))
    w = 0.15

    # Left: accuracy under ablation
    ax = axes[0]
    ax.bar(x - 2*w, normal, w, label='Normal', color=PASS_GREEN, zorder=3)
    ax.bar(x - w, zero, w, label='Zero', color=FAIL_RED, zorder=3)
    ax.bar(x, random, w, label='Random', color=ORANGE, zorder=3)
    ax.bar(x + w, frozen, w, label='Frozen', color=PURPLE, zorder=3)
    ax.bar(x + 2*w, noisy, w, label='Noisy', color=BLUE, zorder=3)

    ax.axhline(y=0.5, color='black', linestyle='--', alpha=0.3, label='Chance')
    ax.set_ylabel('Accuracy')
    ax.set_title('Accuracy Under Workspace Ablation')
    ax.set_xticks(x)
    ax.set_xticklabels(tasks)
    ax.set_ylim(0, 1.1)
    ax.legend(loc='upper right', fontsize=9, ncol=2)
    ax.grid(axis='y', alpha=0.3)

    # Right: necessity scores
    ax = axes[1]
    necessity = [0.490, 0.533, 0.384]
    zero_drop = [0.494, 0.577, 0.292]
    random_drop = [0.486, 0.489, 0.475]
    frozen_drop = [0.479, 0.393, 0.292]

    w2 = 0.2
    ax.bar(x - 1.5*w2, zero_drop, w2, label='Zero drop', color=FAIL_RED, zorder=3)
    ax.bar(x - 0.5*w2, random_drop, w2, label='Random drop', color=ORANGE, zorder=3)
    ax.bar(x + 0.5*w2, frozen_drop, w2, label='Frozen drop', color=PURPLE, zorder=3)
    ax.bar(x + 1.5*w2, necessity, w2, label='Necessity (mean)', color=DARK, zorder=3)

    ax.axhline(y=0.05, color=PASS_GREEN, linestyle='--', alpha=0.5, label='Threshold (0.05)')
    ax.set_ylabel('Accuracy Drop')
    ax.set_title('Workspace Necessity Score by Task')
    ax.set_xticks(x)
    ax.set_xticklabels(tasks)
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(axis='y', alpha=0.3)

    # Annotate composite strongest
    ax.annotate('Strongest for\nintegration tasks',
                xy=(1, 0.533), xytext=(2.2, 0.55),
                fontsize=9, ha='center',
                arrowprops=dict(arrowstyle='->', color=DARK),
                bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow'))

    plt.suptitle('z2037: Workspace Causally Necessary — 4/4 PASS',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'fig_workspace_necessity.png'))
    plt.close()
    print('  [+] fig_workspace_necessity.png')


# ============================================================
# 6. UNIFIED SCORECARD (z2035)
# ============================================================
def plot_scorecard():
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))

    # Left: Per-experiment scores
    ax = axes[0]
    experiments = [
        ('z2020 Capacity', 3, 4),
        ('z2021 Blindsight (CNN)', 4, 4),
        ('z2022 Att. Blink', 2, 4),
        ('z2023 PCI', 1, 4),
        ('z2024 Ignition', 2, 4),
        ('z2025 Rec. Depth', 2, 4),
        ('z2026 Overflow', 4, 4),
        ('z2027 Synergy', 4, 4),
        ('z2028 Blindsight (ResNet)', 4, 4),
        ('z2029 Inatt. Blindness', 2, 4),
        ('z2030 Blindsight (ViT)', 4, 4),
        ('z2031 Pred. Error', 2, 4),
        ('z2032 Rivalry', 1, 4),
        ('z2033 Masking', 0, 4),
        ('z2034 Cost Scale', 3, 4),
        ('z2036 Contrastive', 4, 4),
        ('z2037 Necessity', 4, 4),
    ]

    names = [e[0] for e in experiments]
    scores = [e[1] for e in experiments]
    totals = [e[2] for e in experiments]
    pcts = [s/t for s, t in zip(scores, totals)]

    y = np.arange(len(names))
    colors = [PASS_GREEN if p >= 1.0 else BLUE if p >= 0.75 else ORANGE if p >= 0.5 else FAIL_RED for p in pcts]

    bars = ax.barh(y, scores, color=colors, zorder=3)
    ax.set_xlim(0, 4.5)
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=8.5)
    ax.set_xlabel('Tests Passed (out of 4)')
    ax.set_title('Per-Experiment Scores')
    ax.axvline(x=4, color=PASS_GREEN, linestyle='--', alpha=0.3)
    ax.grid(axis='x', alpha=0.3)
    ax.invert_yaxis()

    for bar, s, t in zip(bars, scores, totals):
        ax.text(bar.get_width() + 0.05, bar.get_y() + bar.get_height()/2,
                f'{s}/{t}', va='center', fontsize=9, fontweight='bold')

    # Right: Theory breakdown
    ax = axes[1]
    theories = ['HOT', 'GWT', 'IIT', 'PP', 'RPT']
    passes = [16, 37, 5, 2, 2]
    totals_t = [17, 53, 8, 4, 8]
    pcts_t = [p/t*100 for p, t in zip(passes, totals_t)]

    colors_t = [PASS_GREEN if p >= 80 else BLUE if p >= 60 else ORANGE if p >= 40 else FAIL_RED for p in pcts_t]

    y = np.arange(len(theories))
    bars = ax.barh(y, pcts_t, color=colors_t, zorder=3)
    ax.set_xlim(0, 110)
    ax.set_yticks(y)
    ax.set_yticklabels(theories, fontsize=11, fontweight='bold')
    ax.set_xlabel('Pass Rate (%)')
    ax.set_title('Score by Consciousness Theory')
    ax.axvline(x=50, color=GRAY, linestyle='--', alpha=0.3)
    ax.grid(axis='x', alpha=0.3)
    ax.invert_yaxis()

    for bar, p, pa, t in zip(bars, pcts_t, passes, totals_t):
        ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,
                f'{p:.0f}% ({pa}/{t})', va='center', fontsize=10, fontweight='bold')

    # Add tier scores as text box
    textstr = 'Tier 1 (Unforgeable): 32/37 = 86%\nTier 2 (Suggestive): 14/32 = 44%\nOverall: 46/69 = 67%'
    props = dict(boxstyle='round', facecolor='lightyellow', alpha=0.8)
    ax.text(55, 4.2, textstr, fontsize=10, verticalalignment='top', bbox=props)

    plt.suptitle('z2035: Unified Consciousness Scorecard (z2020–z2037)',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'fig_scorecard_summary.png'))
    plt.close()
    print('  [+] fig_scorecard_summary.png')


# ============================================================
# 7. DESIGN PATTERN (Discussion)
# ============================================================
def plot_design_pattern():
    fig, ax = plt.subplots(figsize=(9, 5))

    categories = [
        'Ablation-\ndissociation',
        'Cost-based',
        'Information\ndecomposition',
        'Positive-\nproperty',
        'High-level\nconditioning\n(z900)'
    ]
    mean_scores = [3.43, 3.67, 4.0, 1.57, 0.3]
    max_scores = [4, 4, 4, 4, 1]
    n_exps = [7, 3, 1, 7, 5]
    perfect_count = [5, 1, 1, 0, 0]

    pcts = [m/mx*100 for m, mx in zip(mean_scores, max_scores)]
    colors = [PASS_GREEN if p >= 80 else BLUE if p >= 60 else ORANGE if p >= 40 else FAIL_RED for p in pcts]

    x = np.arange(len(categories))
    bars = ax.bar(x, mean_scores, 0.65, color=colors, edgecolor=DARK, linewidth=1, zorder=3)

    # Max possible line
    ax.bar(x, max_scores, 0.65, fill=False, edgecolor=GRAY, linewidth=1.5,
           linestyle='--', zorder=2)

    ax.set_ylabel('Mean Score (out of max)')
    ax.set_title('Design Pattern: What PASSES and What FAILS', fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=10)
    ax.set_ylim(0, 4.8)
    ax.grid(axis='y', alpha=0.3)

    for bar, m, mx, n, pc in zip(bars, mean_scores, max_scores, n_exps, perfect_count):
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., h + 0.08,
                f'{m:.1f}/{mx}\n(n={n}, {pc} perfect)',
                ha='center', va='bottom', fontsize=9)

    # Legend patches
    legend_elements = [
        mpatches.Patch(facecolor=PASS_GREEN, label='Strong PASS (>80%)'),
        mpatches.Patch(facecolor=BLUE, label='Moderate (60-80%)'),
        mpatches.Patch(facecolor=ORANGE, label='Weak (40-60%)'),
        mpatches.Patch(facecolor=FAIL_RED, label='FAIL (<40%)'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=9)

    # Key insight annotation
    ax.annotate('Tests measuring COSTS\nand BREAKAGE pass;\npositive metrics fail',
                xy=(1, 3.67), xytext=(3.5, 3.8),
                fontsize=10, ha='center',
                arrowprops=dict(arrowstyle='->', color=DARK, lw=1.5),
                bbox=dict(boxstyle='round,pad=0.4', facecolor='lightyellow', alpha=0.9))

    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'fig_design_pattern.png'))
    plt.close()
    print('  [+] fig_design_pattern.png')


# ============================================================
# 8. z1901 FALSIFICATION BATTERY (Section 6.3)
# ============================================================
def plot_z1901_falsification():
    fig, ax = plt.subplots(figsize=(10, 5.5))

    tests = [
        'T1: Zero\ntelemetry',
        'T2: Random\ntelemetry',
        'T3: Historical\ntelemetry',
        'T4: Constant\ntelemetry',
        'T5: Inverted\ntelemetry',
        'T6: Perturbation\ndetection'
    ]
    distances = [0.00207, 0.00111, 0.00139, 0.000247, 0.00123, 0.000011]
    thresholds = [0.05, 0.05, 0.03, 0.03, 0.05, 0.0]  # T6 different metric

    x = np.arange(len(tests))

    # Bar chart of distances
    bars = ax.bar(x, distances, 0.6, color=FAIL_RED, edgecolor=DARK, linewidth=1, zorder=3)

    # Threshold lines for each
    for i, th in enumerate(thresholds[:5]):
        ax.plot([i - 0.35, i + 0.35], [th, th], 'k--', linewidth=2, zorder=4)

    # Add general threshold line
    ax.axhline(y=0.05, color=DARK, linestyle='--', linewidth=1, alpha=0.3,
               label='Threshold (0.03-0.05)')

    ax.set_ylabel('Output Distance (lower = more similar)')
    ax.set_title('z1901: ALL 6 Embodiment Indicators FALSIFIED', fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(tests, fontsize=9)
    ax.set_ylim(0, 0.065)
    ax.grid(axis='y', alpha=0.3)

    # Labels
    for bar, d, th in zip(bars, distances, thresholds[:5]):
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., h + 0.001,
                f'{d:.4f}\n(need >{th})',
                ha='center', va='bottom', fontsize=8, color=FAIL_RED)

    # Big FALSIFIED stamp
    ax.text(0.5, 0.85, 'ALL FALSIFIED', transform=ax.transAxes,
            fontsize=20, fontweight='bold', color=FAIL_RED, alpha=0.3,
            ha='center', va='center', rotation=15,
            bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor=FAIL_RED, alpha=0.3))

    ax.text(0.5, -0.18, 'All distances far below thresholds — model ignores telemetry completely',
            transform=ax.transAxes, fontsize=10, ha='center', style='italic')

    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'fig_z1901_falsification.png'))
    plt.close()
    print('  [+] fig_z1901_falsification.png')


# ============================================================
# 9. z1990 UNIFIED CONSCIOUSNESS PROOF (Section 6.3)
# ============================================================
def plot_z1990_unified():
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: Falsification radar
    ax = axes[0]
    tests = ['F1: Embod.\nnecessity', 'F2: Substrate\ndependence',
             'F3: Integration', 'F4: Causal\nefficacy',
             'F5: Temporal\ncoherence', 'F6: GWT\nbroadcast',
             'F7: HOT\ncalibration', 'F8: Continual\nlearning']
    values = [5.184, 0.784, 0.0, 1.0, 0.0, 0.006, -0.098, 0.0]
    thresholds = [1.5, 0.05, 0.1, 0.05, 0.3, 0.5, 0.0, 0.5]
    passed = [True, True, False, False, False, False, False, False]

    colors = [PASS_GREEN if p else FAIL_RED for p in passed]
    y = np.arange(len(tests))

    bars = ax.barh(y, [1 if p else 0 for p in passed], color=colors, alpha=0.7, zorder=3)

    ax.set_yticks(y)
    ax.set_yticklabels(tests, fontsize=9)
    ax.set_xlim(-0.1, 1.3)
    ax.set_xlabel('Pass (1) / Fail (0)')
    ax.set_title('z1990: 2/8 Tests Pass')
    ax.invert_yaxis()
    ax.grid(axis='x', alpha=0.3)

    for i, (v, t, p) in enumerate(zip(values, thresholds, passed)):
        label = f'{v:.3f}' if abs(v) < 10 else f'{v:.1f}'
        color = PASS_GREEN if p else FAIL_RED
        ax.text(1.05, i, f'{label} (need {">" if t >= 0 else "<"}{t})',
                va='center', fontsize=8, color=color)

    # Right: Training loss over epochs
    ax = axes[1]
    epochs = list(range(10, 20))
    losses = [2.094, 2.075, 2.060, 2.047, 2.039, 2.031, 2.026, 2.023, 2.021, 2.021]
    accs = [0.344, 0.347, 0.349, 0.351, 0.352, 0.354, 0.354, 0.354, 0.355, 0.355]

    ax2 = ax.twinx()
    l1 = ax.plot(epochs, losses, 'o-', color=BLUE, linewidth=2, label='Loss')
    l2 = ax2.plot(epochs, accs, 's-', color=ORANGE, linewidth=2, label='Accuracy')

    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss', color=BLUE)
    ax2.set_ylabel('Accuracy', color=ORANGE)
    ax.set_title('z1990: Training Plateau (20 epochs)')
    ax.grid(alpha=0.3)

    lines = l1 + l2
    labels = [l.get_label() for l in lines]
    ax.legend(lines, labels, loc='center right')

    ax.text(0.5, 0.15, 'Loss plateau → 2.02\nAccuracy plateau → 35.5%',
            transform=ax.transAxes, fontsize=10, ha='center',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.8))

    plt.suptitle('z1990: Unified Consciousness Proof — FALSIFIED (2/8)',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'fig_z1990_unified.png'))
    plt.close()
    print('  [+] fig_z1990_unified.png')


# ============================================================
# 10. z2001 GRANGER CAUSALITY (Section 7)
# ============================================================
def plot_granger_causality():
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: FiLM model - p-values for telemetry->output channels
    ax = axes[0]
    channels = ['temp→hidden', 'temp→entropy', 'temp→action',
                'power→hidden', 'power→entropy', 'power→action',
                'util→hidden', 'util→entropy', 'util→action']
    p_values_film = [0.333, 0.058, 0.014, 0.275, 0.303, 6.9e-10, 0.039, 0.302, 3.4e-12]
    significant_film = [p < 0.05 for p in p_values_film]

    # Log scale p-values
    log_p = [-np.log10(max(p, 1e-15)) for p in p_values_film]
    colors = [PASS_GREEN if s else GRAY for s in significant_film]

    y = np.arange(len(channels))
    bars = ax.barh(y, log_p, color=colors, zorder=3)

    # Significance line
    ax.axvline(x=-np.log10(0.05), color=FAIL_RED, linestyle='--', linewidth=2,
               label='p = 0.05')
    ax.axvline(x=-np.log10(0.001), color=ORANGE, linestyle=':', linewidth=1.5,
               label='p = 0.001')

    ax.set_yticks(y)
    ax.set_yticklabels(channels, fontsize=9)
    ax.set_xlabel('$-\\log_{10}(p)$  (higher = more significant)')
    ax.set_title('FiLM Model: 4/12 Significant')
    ax.legend(loc='lower right', fontsize=9)
    ax.invert_yaxis()
    ax.grid(axis='x', alpha=0.3)

    for bar, p, s in zip(bars, p_values_film, significant_film):
        w = bar.get_width()
        label = f'p={p:.1e}' if p < 0.01 else f'p={p:.3f}'
        ax.text(w + 0.1, bar.get_y() + bar.get_height()/2,
                label, va='center', fontsize=8,
                fontweight='bold' if s else 'normal',
                color=PASS_GREEN if s else GRAY)

    # Right: Baseline model - p-values
    ax = axes[1]
    p_values_base = [0.194, 0.366, 0.078, 0.076, 0.097, 0.093, 0.573, 0.089, 0.278]
    significant_base = [p < 0.05 for p in p_values_base]

    log_p_base = [-np.log10(max(p, 1e-15)) for p in p_values_base]
    colors_base = [PASS_GREEN if s else GRAY for s in significant_base]

    bars = ax.barh(y, log_p_base, color=colors_base, zorder=3)
    ax.axvline(x=-np.log10(0.05), color=FAIL_RED, linestyle='--', linewidth=2,
               label='p = 0.05')

    ax.set_yticks(y)
    ax.set_yticklabels(channels, fontsize=9)
    ax.set_xlabel('$-\\log_{10}(p)$')
    ax.set_title('Baseline: 0/12 Significant')
    ax.legend(loc='lower right', fontsize=9)
    ax.invert_yaxis()
    ax.grid(axis='x', alpha=0.3)
    ax.set_xlim(ax.get_xlim()[0], max(log_p) * 0.3)

    plt.suptitle('z2001: Granger Causality — FiLM Creates Detectable Causal Influence',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'fig_granger_causality.png'))
    plt.close()
    print('  [+] fig_granger_causality.png')


# ============================================================
# 11. PCI PROGRESSION (Section 7 — z2008-z2014)
# ============================================================
def plot_pci_progression():
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: PCI over experiments
    ax = axes[0]
    exps = ['z2008', 'z2009', 'z2010', 'z2012', 'z2013', 'z2014']
    pci = [0.014, 0.023, 0.042, 0.089, 0.122, 0.157]
    integration = [0.010, 0.015, 0.028, 0.095, 0.210, 0.310]
    diff = [0.002, 0.003, 0.008, 0.020, 0.035, 0.065]

    x = np.arange(len(exps))

    ax.plot(x, pci, 'o-', color=BLUE, linewidth=2.5, markersize=8, label='PCI', zorder=3)
    ax.plot(x, integration, 's--', color=PASS_GREEN, linewidth=1.5, markersize=6,
            label='Integration', zorder=3)
    ax.plot(x, diff, '^--', color=ORANGE, linewidth=1.5, markersize=6,
            label='Differentiation', zorder=3)

    # Clinical threshold
    ax.axhline(y=0.31, color=FAIL_RED, linestyle='--', linewidth=2, alpha=0.6,
               label='Clinical PCI* = 0.31')

    ax.set_xticks(x)
    ax.set_xticklabels(exps, rotation=30, fontsize=9)
    ax.set_ylabel('Score')
    ax.set_title('PCI Progression (0.014 → 0.157)')
    ax.legend(loc='upper left', fontsize=9)
    ax.grid(alpha=0.3)

    # Annotate "training on the test"
    ax.annotate('Integration loss\nadded here →',
                xy=(4, 0.122), xytext=(2, 0.18),
                fontsize=9, ha='center',
                arrowprops=dict(arrowstyle='->', color=FAIL_RED),
                bbox=dict(boxstyle='round,pad=0.3', facecolor='#ffdddd'))

    for i, p in enumerate(pci):
        ax.text(i, p + 0.008, f'{p:.3f}', ha='center', fontsize=8, fontweight='bold')

    # Right: Key changes per experiment
    ax = axes[1]
    changes = ['Baseline', 'Unified\narchitecture', 'Hardware\ncontingent',
               'Full\nintegration', 'Forced\nintegration\nloss', 'Global WS\nbottleneck']

    bars = ax.barh(np.arange(len(exps)), pci, color=[GRAY, GRAY, BLUE, BLUE, ORANGE, PASS_GREEN],
                   zorder=3)
    ax.axvline(x=0.31, color=FAIL_RED, linestyle='--', linewidth=2, alpha=0.6)
    ax.set_yticks(np.arange(len(exps)))
    ax.set_yticklabels([f'{e}\n{c}' for e, c in zip(exps, changes)], fontsize=8)
    ax.set_xlabel('PCI')
    ax.set_title('Key Architectural Change per Step')
    ax.invert_yaxis()
    ax.grid(axis='x', alpha=0.3)

    ax.text(0.31 + 0.005, 0, 'Clinical\nthreshold\n(0.31)', fontsize=8,
            color=FAIL_RED, va='center')

    # WARNING box
    ax.text(0.95, 0.95, 'WARNING:\nTraining on\nthe test!',
            transform=ax.transAxes, fontsize=10, fontweight='bold',
            color=FAIL_RED, ha='right', va='top',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='#ffdddd', edgecolor=FAIL_RED))

    plt.suptitle('z2008–z2014: PCI Progression — Achieved by Optimizing the Metric Directly',
                 fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'fig_pci_progression.png'))
    plt.close()
    print('  [+] fig_pci_progression.png')


# ============================================================
# 12. z2040 EMBODIED WORKSPACE (Section 9)
# ============================================================
def plot_z2040_embodied():
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    conditions = ['A: Embodied\n(GPU gates)', 'B: Fixed\nworkspace', 'C: Random\ngated']

    # Left: Task accuracy
    ax = axes[0]
    accs = [0.9888, 0.9887, 0.9882]
    colors = [BLUE, GRAY, ORANGE]
    bars = ax.bar(range(3), accs, color=colors, zorder=3)
    ax.set_ylabel('Accuracy')
    ax.set_title('T4: Task Performance (PASS)')
    ax.set_xticks(range(3))
    ax.set_xticklabels(conditions, fontsize=9)
    ax.set_ylim(0.98, 1.0)
    ax.grid(axis='y', alpha=0.3)
    for bar, a in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width()/2., a + 0.0002,
                f'{a:.2%}', ha='center', fontsize=10, fontweight='bold')

    # Center: Workspace necessity
    ax = axes[1]
    necessity = [0.535, 0.540, 0.442]
    bars = ax.bar(range(3), necessity, color=colors, zorder=3)
    ax.set_ylabel('Necessity Score')
    ax.set_title('T1: Workspace Necessity (FAIL)')
    ax.set_xticks(range(3))
    ax.set_xticklabels(conditions, fontsize=9)
    ax.set_ylim(0, 0.7)
    ax.grid(axis='y', alpha=0.3)
    ax.axhline(y=0.05, color=PASS_GREEN, linestyle='--', alpha=0.5)

    diff = abs(necessity[0] - necessity[1])
    ax.annotate(f'A-B diff = {diff:.3f}\n(need >0.01)',
                xy=(0.5, 0.537), xytext=(1.5, 0.62),
                fontsize=9, ha='center', color=FAIL_RED,
                arrowprops=dict(arrowstyle='->', color=FAIL_RED),
                bbox=dict(boxstyle='round,pad=0.3', facecolor='#ffdddd'))

    for bar, n in zip(bars, necessity):
        ax.text(bar.get_x() + bar.get_width()/2., n + 0.01,
                f'{n:.3f}', ha='center', fontsize=10)

    # Right: CKA (representation divergence)
    ax = axes[2]
    cka = [0.999, 1.000, 0.997]
    bars = ax.bar(range(3), cka, color=colors, zorder=3)
    ax.set_ylabel('Mean CKA')
    ax.set_title('T3: Representation Divergence (FAIL)')
    ax.set_xticks(range(3))
    ax.set_xticklabels(conditions, fontsize=9)
    ax.set_ylim(0.99, 1.005)
    ax.grid(axis='y', alpha=0.3)
    ax.axhline(y=0.90, color=PASS_GREEN, linestyle='--', alpha=0.3)

    for bar, c in zip(bars, cka):
        ax.text(bar.get_x() + bar.get_width()/2., c + 0.0003,
                f'{c:.3f}', ha='center', fontsize=10)

    ax.text(0.5, 0.15, 'Want CKA < 0.90\nGot 0.999',
            transform=ax.transAxes, fontsize=9, ha='center', color=FAIL_RED,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#ffdddd'))

    plt.suptitle('z2040: Embodied Workspace Revisit — PARTIAL (2/4): Gates respond but MNIST too easy',
                 fontsize=13, fontweight='bold', y=1.03)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'fig_z2040_embodied.png'))
    plt.close()
    print('  [+] fig_z2040_embodied.png')


# ============================================================
# 13. ANALOG GATE TRAJECTORY (z2055 vs z2056)
# ============================================================
def plot_analog_gate_trajectory():
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # z2055: gate suppressed (physics optional)
    ax = axes[0]
    gate_z2055 = [0.366, 0.286, 0.238, 0.221, 0.207, 0.174, 0.148, 0.127,
                  0.139, 0.126, 0.120, 0.093]
    gate_z2056 = [0.682, 0.968, 0.988, 0.986, 0.993, 0.989, 0.996, 0.990,
                  0.993, 0.991, 0.998, 0.993, 0.996, 0.992, 0.995]
    epochs_55 = list(range(1, len(gate_z2055) + 1))
    epochs_56 = list(range(1, len(gate_z2056) + 1))

    ax.plot(epochs_55, gate_z2055, 'o-', color=FAIL_RED, linewidth=2.5,
            markersize=6, label='z2055: Physics optional', zorder=3)
    ax.plot(epochs_56, gate_z2056, 's-', color=PASS_GREEN, linewidth=2.5,
            markersize=6, label='z2056: Physics required', zorder=3)

    ax.axhline(y=0.5, color=GRAY, linestyle='--', alpha=0.3)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Learned Gate Value (sigmoid)')
    ax.set_title('Analog Gate Trajectory: Optimizer Controls Physics Coupling')
    ax.set_ylim(0, 1.05)
    ax.legend(loc='center right', fontsize=10)
    ax.grid(alpha=0.3)

    # Annotations
    ax.annotate('GD suppresses\nnoisy physics\n(gate → 0.075)',
                xy=(11, 0.093), xytext=(7, 0.35),
                fontsize=9, ha='center', color=FAIL_RED,
                arrowprops=dict(arrowstyle='->', color=FAIL_RED),
                bbox=dict(boxstyle='round,pad=0.3', facecolor='#ffdddd'))
    ax.annotate('Task requires physics\n→ gate stays open\n(gate → 0.999)',
                xy=(14, 0.995), xytext=(8, 0.75),
                fontsize=9, ha='center', color=PASS_GREEN,
                arrowprops=dict(arrowstyle='->', color=PASS_GREEN),
                bbox=dict(boxstyle='round,pad=0.3', facecolor='#ddffdd'))

    # Right panel: condition comparison bar chart
    ax = axes[1]
    conditions = ['A: Analog\n(z2055)', 'B: Digital\n(z2055)', 'C: Blind\n(z2055)',
                  'D: Scrambled\n(z2055)', 'A: Physics\n(z2056)', 'D: Scrambled\n(z2056)']
    accs = [0.990, 0.990, 0.987, 0.263, 0.989, 0.506]
    colors_bar = [BLUE, GRAY, GRAY, FAIL_RED, PASS_GREEN, FAIL_RED]

    bars = ax.bar(range(len(conditions)), accs, color=colors_bar, zorder=3)
    ax.set_ylabel('Accuracy')
    ax.set_title('Kill Shot: Scrambled Routing Destroys Performance')
    ax.set_xticks(range(len(conditions)))
    ax.set_xticklabels(conditions, fontsize=8)
    ax.set_ylim(0, 1.15)
    ax.grid(axis='y', alpha=0.3)

    for bar, a in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width()/2., a + 0.01,
                f'{a:.1%}', ha='center', fontsize=9, fontweight='bold')

    # Annotate kill shots
    ax.annotate('72.7% gap', xy=(3, 0.263), xytext=(3, 0.55),
                fontsize=10, fontweight='bold', color=FAIL_RED, ha='center',
                arrowprops=dict(arrowstyle='->', color=FAIL_RED))
    ax.annotate('48.3% gap', xy=(5, 0.506), xytext=(5, 0.75),
                fontsize=10, fontweight='bold', color=FAIL_RED, ha='center',
                arrowprops=dict(arrowstyle='->', color=FAIL_RED))

    plt.suptitle('z2055–z2056: Analog Physics Gate — Optimizer Controls Digital-to-Analog Spectrum',
                 fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'fig_analog_gate_trajectory.png'))
    plt.close()
    print('  [+] fig_analog_gate_trajectory.png')


# ============================================================
# 14. DUAL-CHANNEL HIERARCHY (z2057) + CROSS-MACHINE
# ============================================================
def plot_dual_channel_hierarchy():
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    # Left: ikaros accuracy bars with hierarchy
    ax = axes[0]
    conditions = ['A: Full\n(bank+timing)', 'B: Blind', 'C: Bank\nonly',
                  'D: No\ntiming', 'E: Scr.\ntiming', 'F: Scr.\nbank',
                  'G: Scr.\nboth']
    accs_ik = [0.9772, 0.2562, 0.4961, 0.4949, 0.0010, 0.0021, 0.0008]
    colors_bar = [PASS_GREEN, GRAY, BLUE, BLUE, FAIL_RED, FAIL_RED, FAIL_RED]

    bars = ax.bar(range(len(conditions)), accs_ik, color=colors_bar,
                  edgecolor=DARK, linewidth=0.5, zorder=3)
    ax.set_ylabel('Accuracy')
    ax.set_title('ikaros: 8/8 PASS — Information-Theoretic Hierarchy')
    ax.set_xticks(range(len(conditions)))
    ax.set_xticklabels(conditions, fontsize=8)
    ax.set_ylim(0, 1.15)
    ax.grid(axis='y', alpha=0.3)

    for bar, a in zip(bars, accs_ik):
        ax.text(bar.get_x() + bar.get_width()/2., a + 0.015,
                f'{a:.1%}', ha='center', fontsize=8, fontweight='bold')

    # Hierarchy annotation
    ax.annotate('', xy=(0, 0.977), xytext=(2, 0.496),
                arrowprops=dict(arrowstyle='->', color=DARK, lw=1.5))
    ax.annotate('', xy=(2, 0.496), xytext=(1, 0.256),
                arrowprops=dict(arrowstyle='->', color=DARK, lw=1.5))
    ax.text(3.5, 1.05, '0 ch = 25%  →  1 ch = 50%  →  2 ch = 98%',
            fontsize=9, fontweight='bold', ha='center',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow'))

    # Right: cross-machine comparison
    ax = axes[1]
    conds_cross = ['A: Full', 'B: Blind', 'C: Bank\nonly', 'E: Scr.\ntiming']
    accs_ik_sub = [0.9772, 0.2562, 0.4961, 0.0010]
    accs_dae = [0.4953, 0.2558, 0.4954, 0.4946]

    x = np.arange(len(conds_cross))
    w = 0.35
    bars1 = ax.bar(x - w/2, accs_ik_sub, w, label='ikaros (4.43× ratio)',
                   color=BLUE, zorder=3)
    bars2 = ax.bar(x + w/2, accs_dae, w, label='daedalus (1.05× ratio)',
                   color=ORANGE, zorder=3)

    ax.set_ylabel('Accuracy')
    ax.set_title('Cross-Machine: DVFS Flat → Timing Channel Dead')
    ax.set_xticks(x)
    ax.set_xticklabels(conds_cross, fontsize=9)
    ax.set_ylim(0, 1.15)
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(axis='y', alpha=0.3)

    for bars in [bars1, bars2]:
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., h + 0.015,
                    f'{h:.1%}', ha='center', fontsize=8)

    # Annotate key difference
    ax.annotate('Timing dead on daedalus:\nDVFS gives only 1.05× ratio\n→ A ≈ C (bank only)',
                xy=(0.3, 0.50), xytext=(1.8, 0.80),
                fontsize=9, ha='center',
                arrowprops=dict(arrowstyle='->', color=ORANGE, lw=1.5),
                bbox=dict(boxstyle='round,pad=0.3', facecolor='#fff3dd'))

    plt.suptitle('z2057: SCLK Wall-Clock Dual-Channel Embodiment (8/8 PASS ikaros, 3/8 daedalus)',
                 fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'fig_dual_channel_hierarchy.png'))
    plt.close()
    print('  [+] fig_dual_channel_hierarchy.png')


# ============================================================
# 15. WALL-CLOCK TIMING CHARACTERIZATION (z2057)
# ============================================================
def plot_wall_clock_timing():
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: timing distributions
    ax = axes[0]
    # Simulated distributions based on mean/std from JSON
    np.random.seed(42)
    low_times = np.random.normal(0.8505, 0.0115, 500)
    high_times = np.random.normal(0.1919, 0.0255, 500)

    ax.hist(low_times, bins=30, alpha=0.7, color=FAIL_RED, label='Low SCLK (600 MHz)',
            density=True, zorder=3)
    ax.hist(high_times, bins=30, alpha=0.7, color=BLUE, label='High SCLK (1902 MHz)',
            density=True, zorder=3)

    ax.axvline(x=0.5, color=DARK, linestyle='--', linewidth=1.5, alpha=0.5,
               label='Threshold (0.5)')
    ax.set_xlabel('Kernel Wall-Clock Time (ms)')
    ax.set_ylabel('Density')
    ax.set_title('Bimodal Timing: 4.43× SCLK Ratio')
    ax.legend(loc='upper center', fontsize=9)
    ax.grid(alpha=0.3)

    ax.annotate(f'4.43× ratio\n0.85ms vs 0.19ms',
                xy=(0.5, 0.5), xytext=(0.5, 0.7),
                fontsize=11, fontweight='bold', ha='center',
                transform=ax.transAxes,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow'))

    # Right: gate values at different SCLK
    ax = axes[1]
    sclk_labels = ['Low SCLK\n(600 MHz)', 'High SCLK\n(1902 MHz)', 'Mean']
    gate_vals = [0.962, 0.834, 0.898]
    colors_g = [FAIL_RED, BLUE, DARK]

    bars = ax.bar(range(3), gate_vals, color=colors_g, edgecolor=DARK,
                  linewidth=0.5, zorder=3)
    ax.set_ylabel('Learned Gate Value')
    ax.set_title('Gate Responds to SCLK State')
    ax.set_xticks(range(3))
    ax.set_xticklabels(sclk_labels, fontsize=10)
    ax.set_ylim(0, 1.1)
    ax.grid(axis='y', alpha=0.3)
    ax.axhline(y=0.3, color=PASS_GREEN, linestyle='--', alpha=0.3, label='Threshold (0.3)')
    ax.legend(fontsize=9)

    for bar, g in zip(bars, gate_vals):
        ax.text(bar.get_x() + bar.get_width()/2., g + 0.02,
                f'{g:.3f}', ha='center', fontsize=11, fontweight='bold')

    ax.annotate('Gate higher at low SCLK:\nmodel relies MORE on timing\nwhen signal is stronger',
                xy=(0, 0.962), xytext=(1.5, 0.55),
                fontsize=9, ha='center',
                arrowprops=dict(arrowstyle='->', color=DARK),
                bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow'))

    plt.suptitle('z2057: Wall-Clock SCLK Timing — Genuine Physics Measurement',
                 fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'fig_wall_clock_timing.png'))
    plt.close()
    print('  [+] fig_wall_clock_timing.png')


# ============================================================
# 16. THREE-CHANNEL UNIFIED (z2058)
# ============================================================
def plot_three_channel():
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    # Left: accuracy bars for all conditions
    ax = axes[0]
    conditions = ['A: Full\n(3 ch)', 'B: Blind', 'C: Bank\nonly',
                  'D: Bank\n+fp16', 'E: Scr.\ntiming', 'F: Scr.\nbank',
                  'G: Scr.\nfp16']
    accs = [0.9773, 0.260, 0.495, 0.496, 0.0012, 0.0026, 0.9776]
    colors_bar = [PASS_GREEN, GRAY, BLUE, PURPLE, FAIL_RED, FAIL_RED, ORANGE]

    bars = ax.bar(range(len(conditions)), accs, color=colors_bar,
                  edgecolor=DARK, linewidth=0.5, zorder=3)
    ax.set_ylabel('Accuracy')
    ax.set_title('z2058: Three-Channel Unified (7/8 PASS)')
    ax.set_xticks(range(len(conditions)))
    ax.set_xticklabels(conditions, fontsize=8)
    ax.set_ylim(0, 1.15)
    ax.grid(axis='y', alpha=0.3)

    for bar, a in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width()/2., a + 0.015,
                f'{a:.1%}', ha='center', fontsize=8, fontweight='bold')

    # Annotate fp16 non-functional
    ax.annotate('G ≈ A: scrambling fp16\nhas NO effect →\nfp16 channel\nnon-functional',
                xy=(6, 0.978), xytext=(4.5, 0.75),
                fontsize=9, ha='center', color=ORANGE,
                arrowprops=dict(arrowstyle='->', color=ORANGE, lw=1.5),
                bbox=dict(boxstyle='round,pad=0.3', facecolor='#fff3dd'))

    # D ≈ C annotation
    ax.annotate('D ≈ C: adding fp16\nto bank adds nothing',
                xy=(3, 0.496), xytext=(1.5, 0.70),
                fontsize=9, ha='center', color=PURPLE,
                arrowprops=dict(arrowstyle='->', color=PURPLE),
                bbox=dict(boxstyle='round,pad=0.3', facecolor='#f0ddff'))

    # Right: channel contribution breakdown
    ax = axes[1]
    channels = ['WGP Bank\n(digital)', 'SCLK Timing\n(analog)', 'FP16 Rounding\n(compute)']
    contributions = [49.6 - 26.0, 97.7 - 49.6, 49.6 - 49.5]
    # Marginal contribution: C-B for bank, A-C for timing, D-C for fp16
    colors_ch = [BLUE, PASS_GREEN, FAIL_RED]

    bars = ax.bar(range(3), contributions, color=colors_ch,
                  edgecolor=DARK, linewidth=0.5, zorder=3)
    ax.set_ylabel('Marginal Accuracy Contribution (pp)')
    ax.set_title('Channel Information Content')
    ax.set_xticks(range(3))
    ax.set_xticklabels(channels, fontsize=10)
    ax.grid(axis='y', alpha=0.3)
    ax.axhline(y=0, color='black', linewidth=0.5)

    for bar, c in zip(bars, contributions):
        ax.text(bar.get_x() + bar.get_width()/2., c + 0.5,
                f'+{c:.1f}pp', ha='center', fontsize=11, fontweight='bold')

    ax.annotate('FP16 operands (3.14+1.72)\ngive same result regardless\nof rounding mode → 0 info',
                xy=(2, 0.1), xytext=(1.2, 15),
                fontsize=9, ha='center', color=FAIL_RED,
                arrowprops=dict(arrowstyle='->', color=FAIL_RED),
                bbox=dict(boxstyle='round,pad=0.3', facecolor='#ffdddd'))

    plt.suptitle('z2058: Three-Channel Unified — FP16 Rounding Non-Functional (7/8 PASS)',
                 fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'fig_three_channel.png'))
    plt.close()
    print('  [+] fig_three_channel.png')


# ============================================================
# 17. HARDWARE INTEGRATION SUMMARY (z2050-z2060)
# ============================================================
def plot_hw_integration_summary():
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))

    # Left: pass rates across all experiments
    ax = axes[0]
    experiments = [
        ('z2050\nFPGA-like', 5, 5),
        ('z2051\nSi PUF', 7, 7),
        ('z2052\nCU-mask', 7, 8),
        ('z2053\nBidir', 4, 7),
        ('z2054\nFP16', 6, 7),
        ('z2055\nAnalog', 5, 8),
        ('z2056\nPhysics', 4, 8),
        ('z2057\nDual-ch', 8, 8),
        ('z2058\n3-ch', 7, 8),
        ('z2059\nHomeo', 7, 8),
        ('z2060\nExcl.', 8, 8),
        ('z2061\nAllost.', 12, 12),
    ]

    names = [e[0] for e in experiments]
    passes = [e[1] for e in experiments]
    totals = [e[2] for e in experiments]
    pcts = [p/t for p, t in zip(passes, totals)]

    x = np.arange(len(names))
    colors_hw = [PASS_GREEN if p >= 0.875 else BLUE if p >= 0.625 else ORANGE if p >= 0.5 else FAIL_RED
                 for p in pcts]

    bars = ax.bar(x, passes, color=colors_hw, edgecolor=DARK, linewidth=0.5, zorder=3)
    for i, t in enumerate(totals):
        ax.plot([i - 0.35, i + 0.35], [t, t], 'k--', linewidth=1, alpha=0.3)

    ax.set_ylabel('Tests Passed')
    ax.set_title('Hardware Integration: Pass Rates (z2050-z2061)')
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=7)
    ax.set_ylim(0, 14)
    ax.grid(axis='y', alpha=0.3)

    for bar, p, t in zip(bars, passes, totals):
        ax.text(bar.get_x() + bar.get_width()/2., p + 0.15,
                f'{p}/{t}', ha='center', fontsize=8, fontweight='bold')

    mean_pct = np.mean(pcts)
    ax.axhline(y=mean_pct * 8, color=DARK, linestyle=':', alpha=0.5)
    ax.text(len(names)-0.5, mean_pct * 8 + 0.2, f'Mean: {mean_pct:.0%}', fontsize=9, va='bottom')

    # Right: key metrics comparison (including z2059/z2060)
    ax = axes[1]
    exp_labels = ['z2050', 'z2051', 'z2055', 'z2056', 'z2057', 'z2058', 'z2059', 'z2060', 'z2061']
    kill_shots = [98.6, 48.8, 72.7, 48.3, 97.6, 97.6, 93.8, 75.0, 79.1]
    gates = [None, None, 0.075, 0.999, 0.898, 0.838, 0.566, 0.494, 0.594]

    x2 = np.arange(len(exp_labels))
    bars = ax.bar(x2, kill_shots, color=[FAIL_RED]*len(exp_labels),
                  edgecolor=DARK, linewidth=0.5, zorder=3, alpha=0.8)
    ax.set_ylabel('Kill Shot Gap (pp)', color=FAIL_RED)
    ax.set_title('Kill Shot Strength & Gate Values')
    ax.set_xticks(x2)
    ax.set_xticklabels(exp_labels, fontsize=9)
    ax.set_ylim(0, 115)
    ax.grid(axis='y', alpha=0.3)

    for bar, k in zip(bars, kill_shots):
        ax.text(bar.get_x() + bar.get_width()/2., k + 1,
                f'{k:.0f}%', ha='center', fontsize=7, fontweight='bold', color=FAIL_RED)

    ax2 = ax.twinx()
    gate_x = [i for i, g in enumerate(gates) if g is not None]
    gate_y = [g for g in gates if g is not None]
    ax2.plot(gate_x, gate_y, 'D-', color=PASS_GREEN, linewidth=2.5, markersize=8,
             label='Learned gate', zorder=4)
    ax2.set_ylabel('Learned Gate Value', color=PASS_GREEN)
    ax2.set_ylim(0, 1.1)
    ax2.legend(loc='upper left', fontsize=9)

    plt.suptitle('z2050-z2061: Hardware-IS-Computation — 12 Experiments',
                 fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'fig_hw_integration_summary.png'))
    plt.close()
    print('  [+] fig_hw_integration_summary.png')


# ============================================================
# 18. z2060 ABLATION DISSOCIATION
# ============================================================
def plot_z2060_ablation():
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Left: condition comparison
    ax = axes[0]
    conditions = ['A\nHomeo', 'B\nBlind', 'C\nNo SM', 'D\nLight', 'E\nScram', 'F\nAblat']
    accs = [0.9707, 0.5996, 0.5943, 0.5940, 0.2207, 0.5872]
    colors_bar = [PASS_GREEN, GRAY, BLUE, BLUE, FAIL_RED, ORANGE]

    bars = ax.bar(range(len(conditions)), accs, color=colors_bar, edgecolor=DARK,
                  linewidth=0.5, zorder=3)
    ax.set_ylabel('Accuracy')
    ax.set_title('Accuracy Across Conditions')
    ax.set_xticks(range(len(conditions)))
    ax.set_xticklabels(conditions, fontsize=10)
    ax.set_ylim(0, 1.15)
    ax.grid(axis='y', alpha=0.3)

    for bar, a in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width()/2., a + 0.015,
                f'{a:.1%}', ha='center', fontsize=9, fontweight='bold')

    # Annotate A-F gap
    ax.annotate('38.3pp\ndrop', xy=(5, 0.587), xytext=(4.2, 0.82),
                fontsize=11, fontweight='bold', color=FAIL_RED, ha='center',
                arrowprops=dict(arrowstyle='->', color=FAIL_RED, lw=2),
                bbox=dict(boxstyle='round,pad=0.3', facecolor='#ffdddd'))

    # Middle: per-state breakdown for F
    ax = axes[1]
    states = ['F\nhigh SCLK', 'F\nlow SCLK']
    f_accs = [0.9944, 0.2004]
    colors_state = [PASS_GREEN, FAIL_RED]

    bars = ax.bar(range(2), f_accs, 0.6, color=colors_state, edgecolor=DARK,
                  linewidth=0.5, zorder=3)
    ax.set_ylabel('Accuracy')
    ax.set_title('Ablated Model: Per-State Breakdown')
    ax.set_xticks(range(2))
    ax.set_xticklabels(states, fontsize=11)
    ax.set_ylim(0, 1.15)
    ax.grid(axis='y', alpha=0.3)
    ax.axhline(y=0.5, color=DARK, linestyle='--', alpha=0.3, label='Chance')

    for bar, a in zip(bars, f_accs):
        ax.text(bar.get_x() + bar.get_width()/2., a + 0.02,
                f'{a:.1%}', ha='center', fontsize=12, fontweight='bold')

    ax.annotate('Full path works\n(bank-shifted correct)',
                xy=(0, 0.994), xytext=(0.5, 0.75),
                fontsize=9, ha='center', color=PASS_GREEN,
                arrowprops=dict(arrowstyle='->', color=PASS_GREEN),
                bbox=dict(boxstyle='round,pad=0.2', facecolor='#ddffdd'))
    ax.annotate('Full path WRONG\nfor reversed labels',
                xy=(1, 0.200), xytext=(1.0, 0.45),
                fontsize=9, ha='center', color=FAIL_RED,
                arrowprops=dict(arrowstyle='->', color=FAIL_RED),
                bbox=dict(boxstyle='round,pad=0.2', facecolor='#ffdddd'))

    # Right: gate values comparison
    ax = axes[2]
    models = ['A\nhigh', 'A\nlow', 'F\nhigh', 'F\nlow']
    gate_vals = [0.823, 0.165, 0.516, 0.516]
    colors_g = [PASS_GREEN, BLUE, ORANGE, ORANGE]

    bars = ax.bar(range(4), gate_vals, 0.6, color=colors_g, edgecolor=DARK,
                  linewidth=0.5, zorder=3)
    ax.set_ylabel('Gate Value')
    ax.set_title('Gate: Adaptive vs Constant')
    ax.set_xticks(range(4))
    ax.set_xticklabels(models, fontsize=10)
    ax.set_ylim(0, 1.05)
    ax.grid(axis='y', alpha=0.3)
    ax.axhline(y=0.5, color=DARK, linestyle='--', alpha=0.3)

    for bar, g in zip(bars, gate_vals):
        ax.text(bar.get_x() + bar.get_width()/2., g + 0.02,
                f'{g:.3f}', ha='center', fontsize=10, fontweight='bold')

    ax.annotate('Self-model enables\nadaptive gating', xy=(0.5, 0.5),
                xytext=(0.5, 0.92), fontsize=9, ha='center',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='lightyellow'))

    plt.suptitle('z2060: Self-Model Ablation — Causal Necessity Demonstrated (T4 PASS: 38.3pp)',
                 fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'fig_z2060_ablation.png'))
    plt.close()
    print('  [+] fig_z2060_ablation.png')


# ============================================================
# 19. z2059 vs z2060 COMPARISON
# ============================================================
def plot_z2060_comparison():
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    # Left: z2059 vs z2060 key metrics
    ax = axes[0]
    metrics = ['Accuracy\n(A)', 'T4 gap\n(A-F)', 'Gate\nhigh', 'Gate\nlow', 'Gate-SCLK\n|r|', 'AUROC']
    z2059_vals = [96.6, -0.6, 0.696, 0.436, 0.676, 1.000]
    z2060_vals = [97.1, 38.3, 0.823, 0.165, 0.816, 0.974]

    x = np.arange(len(metrics))
    w = 0.35
    bars1 = ax.bar(x - w/2, z2059_vals, w, label='z2059 (7/8)', color=BLUE,
                   edgecolor=DARK, linewidth=0.5, zorder=3)
    bars2 = ax.bar(x + w/2, z2060_vals, w, label='z2060 (8/8)', color=PASS_GREEN,
                   edgecolor=DARK, linewidth=0.5, zorder=3)

    ax.set_ylabel('Value (% or ratio)')
    ax.set_title('z2059 → z2060: Key Metric Improvements')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=9)
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(axis='y', alpha=0.3)
    ax.axhline(y=0, color='black', linewidth=0.5)

    for bars in [bars1, bars2]:
        for bar in bars:
            h = bar.get_height()
            if abs(h) > 1:
                ax.text(bar.get_x() + bar.get_width()/2., h + (1 if h > 0 else -3),
                        f'{h:.1f}', ha='center', fontsize=8, fontweight='bold')
            else:
                ax.text(bar.get_x() + bar.get_width()/2., h + 0.02,
                        f'{h:.3f}', ha='center', fontsize=8)

    # Highlight T4 fix
    ax.annotate('T4 FIXED!\n-0.6 → +38.3',
                xy=(1.2, 38.3), xytext=(3, 60),
                fontsize=10, fontweight='bold', color=PASS_GREEN, ha='center',
                arrowprops=dict(arrowstyle='->', color=PASS_GREEN, lw=2),
                bbox=dict(boxstyle='round,pad=0.3', facecolor='#ddffdd'))

    # Right: architectural diagram comparison
    ax = axes[1]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('Architectural Fix: Exclusive Path Specialization', fontsize=12, fontweight='bold')

    # z2059 (top)
    ax.text(5, 9.5, 'z2059 (FAIL T4)', fontsize=11, fontweight='bold', ha='center', color=FAIL_RED)
    ax.add_patch(mpatches.FancyBboxPatch((1, 7.5), 3, 1.2, boxstyle='round,pad=0.1',
                 facecolor='#ddddff', edgecolor=DARK))
    ax.text(2.5, 8.1, 'bank_w × [h_img, h_hw]\n(timing leaks!)', fontsize=8, ha='center', color=FAIL_RED)

    ax.add_patch(mpatches.FancyBboxPatch((5.5, 7.5), 3, 1.2, boxstyle='round,pad=0.1',
                 facecolor='#dddddd', edgecolor=DARK))
    ax.text(7, 8.1, 'head_light(h_img)', fontsize=8, ha='center')

    ax.annotate('', xy=(4.5, 7), xytext=(2.5, 7.5),
                arrowprops=dict(arrowstyle='->', color=DARK))
    ax.annotate('', xy=(4.5, 7), xytext=(7, 7.5),
                arrowprops=dict(arrowstyle='->', color=DARK))
    ax.text(4.5, 6.6, 'gate (const OK)', fontsize=8, ha='center', color=FAIL_RED)

    # z2060 (bottom)
    ax.text(5, 5.5, 'z2060 (PASS T4)', fontsize=11, fontweight='bold', ha='center', color=PASS_GREEN)
    ax.add_patch(mpatches.FancyBboxPatch((1, 3.5), 3, 1.2, boxstyle='round,pad=0.1',
                 facecolor='#ddffdd', edgecolor=DARK))
    ax.text(2.5, 4.1, 'bank_w × h_img\n(NO timing)', fontsize=8, ha='center', color=PASS_GREEN)

    ax.add_patch(mpatches.FancyBboxPatch((5.5, 3.5), 3, 1.2, boxstyle='round,pad=0.1',
                 facecolor='#ddffdd', edgecolor=DARK))
    ax.text(7, 4.1, 'head_light(h_img)\n(NO banks)', fontsize=8, ha='center', color=PASS_GREEN)

    ax.annotate('', xy=(4.5, 3), xytext=(2.5, 3.5),
                arrowprops=dict(arrowstyle='->', color=DARK))
    ax.annotate('', xy=(4.5, 3), xytext=(7, 3.5),
                arrowprops=dict(arrowstyle='->', color=DARK))
    ax.text(4.5, 2.6, 'gate (MUST be correct)', fontsize=8, ha='center',
            color=PASS_GREEN, fontweight='bold')

    # Self-model arrow
    ax.add_patch(mpatches.FancyBboxPatch((3, 1), 4, 1, boxstyle='round,pad=0.1',
                 facecolor='lightyellow', edgecolor=DARK))
    ax.text(5, 1.5, 'Self-model → Gate\n(ONLY timing-aware)', fontsize=8,
            ha='center', fontweight='bold')
    ax.annotate('', xy=(4.5, 2.6), xytext=(5, 2),
                arrowprops=dict(arrowstyle='->', color=DARK, lw=1.5))

    plt.suptitle('z2059 → z2060: Removing Timing Leakage Makes Self-Model Causally Necessary',
                 fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'fig_z2060_comparison.png'))
    plt.close()
    print('  [+] fig_z2060_comparison.png')


# ============================================================
# 20. z2061 CLOSED-LOOP ALLOSTATIC EMBODIMENT
# ============================================================
def plot_z2061_allostatic():
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))

    # Left: condition comparison with effort ablation
    ax = axes[0]
    conditions = ['A\nAllostatic', 'B\nBlind', 'E\nScrambled', 'F\nNo Self-M', 'G\nNo Effort', 'H\nAlways-Hi']
    accs = [0.9919, 0.6001, 0.2008, 0.5896, 0.6485, 0.6285]
    colors_bar = [PASS_GREEN, GRAY, FAIL_RED, ORANGE, PURPLE, BLUE]

    bars = ax.bar(range(len(conditions)), accs, color=colors_bar, edgecolor=DARK,
                  linewidth=0.5, zorder=3)
    ax.set_ylabel('Accuracy')
    ax.set_title('Accuracy: 6 Conditions')
    ax.set_xticks(range(len(conditions)))
    ax.set_xticklabels(conditions, fontsize=9)
    ax.set_ylim(0, 1.2)
    ax.grid(axis='y', alpha=0.3)
    ax.axhline(y=0.5, color=DARK, linestyle='--', alpha=0.3)

    for bar, a in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width()/2., a + 0.015,
                f'{a:.1%}', ha='center', fontsize=8, fontweight='bold')

    # Annotate key gaps
    ax.annotate('Perception\n-40.2pp', xy=(3, 0.59), xytext=(3, 0.85),
                fontsize=9, fontweight='bold', color=ORANGE, ha='center',
                arrowprops=dict(arrowstyle='->', color=ORANGE, lw=1.5),
                bbox=dict(boxstyle='round,pad=0.2', facecolor='#fff3dd'))
    ax.annotate('Action\n-34.3pp', xy=(4, 0.65), xytext=(4.5, 1.05),
                fontsize=9, fontweight='bold', color=PURPLE, ha='center',
                arrowprops=dict(arrowstyle='->', color=PURPLE, lw=1.5),
                bbox=dict(boxstyle='round,pad=0.2', facecolor='#f0ddff'))

    # Middle: sensorimotor loop diagram
    ax = axes[1]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('Closed-Loop Causal Chain', fontsize=12, fontweight='bold')

    # Draw loop: demand → effort → DVFS → SCLK → self-model → gate → accuracy
    loop_labels = ['Demand\nCue', 'Effort\nHead', 'DVFS\nActuation', 'SCLK\nChange',
                   'Wall-Clock\nTiming', 'Self-Model\nAUROC=1.0', 'Gate\nh=0.90 l=0.29',
                   'Label\nRouting']
    angles = np.linspace(0, 2*np.pi, len(loop_labels), endpoint=False) - np.pi/2
    cx, cy, r = 5, 5, 3.5
    for i, (label, angle) in enumerate(zip(loop_labels, angles)):
        x_pos = cx + r * np.cos(angle)
        y_pos = cy + r * np.sin(angle)
        color = PASS_GREEN if i in [1, 5, 6] else BLUE
        ax.add_patch(mpatches.FancyBboxPatch(
            (x_pos-1.1, y_pos-0.45), 2.2, 0.9,
            boxstyle='round,pad=0.1', facecolor='#f0f8ff', edgecolor=color, linewidth=1.5))
        ax.text(x_pos, y_pos, label, fontsize=7, ha='center', va='center',
                fontweight='bold', color=DARK)
        # Arrow to next
        next_i = (i + 1) % len(loop_labels)
        next_angle = angles[next_i]
        x2 = cx + r * np.cos(next_angle)
        y2 = cy + r * np.sin(next_angle)
        mid_angle = (angle + next_angle) / 2
        if abs(next_angle - angle) > np.pi:
            mid_angle += np.pi
        ax.annotate('', xy=(x2, y2), xytext=(x_pos, y_pos),
                    arrowprops=dict(arrowstyle='->', color=DARK, lw=1, connectionstyle='arc3,rad=0.2'))

    ax.text(5, 5, 'CLOSED\nLOOP', fontsize=10, ha='center', va='center',
            fontweight='bold', color=PASS_GREEN, alpha=0.5)

    # Right: energy efficiency + temporal correlation
    ax = axes[2]
    metrics = ['Energy\nRatio', 'Temporal\nCorr', 'Effort\nAcc', 'SCLK\nBalance']
    values = [0.436, 1.000, 1.000, 0.564]
    thresholds = [0.90, 0.50, 0.80, 0.75]
    colors_m = [PASS_GREEN if v < t or (m == 'Temporal\nCorr' and v > t) or
                (m == 'Effort\nAcc' and v > t) else FAIL_RED
                for v, t, m in zip(values, thresholds, metrics)]
    # Override with correct pass/fail
    colors_m = [PASS_GREEN, PASS_GREEN, PASS_GREEN, PASS_GREEN]

    bars = ax.bar(range(len(metrics)), values, color=colors_m, edgecolor=DARK,
                  linewidth=0.5, zorder=3, alpha=0.85)
    ax.set_ylabel('Value')
    ax.set_title('Sensorimotor Loop Metrics')
    ax.set_xticks(range(len(metrics)))
    ax.set_xticklabels(metrics, fontsize=9)
    ax.set_ylim(0, 1.2)
    ax.grid(axis='y', alpha=0.3)

    # Threshold lines
    for i, (v, t, m) in enumerate(zip(values, thresholds, metrics)):
        ax.plot([i-0.4, i+0.4], [t, t], '--', color=FAIL_RED, linewidth=1, alpha=0.5)

    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2., v + 0.02,
                f'{v:.3f}', ha='center', fontsize=10, fontweight='bold')

    ax.text(0.5, 0.95, '56% energy saved\nvs always-high', fontsize=9, ha='center',
            transform=ax.transAxes, color=PASS_GREEN, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#ddffdd'))

    plt.suptitle('z2061: Closed-Loop Allostatic Embodiment — 12/12 PASS (PERFECT)',
                 fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'fig_z2061_allostatic.png'))
    plt.close()
    print('  [+] fig_z2061_allostatic.png')


# ============================================================
# 21. z2059 → z2060 → z2061 EVOLUTION
# ============================================================
def plot_z2061_evolution():
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    # Left: score evolution z2059 → z2060 → z2061
    ax = axes[0]
    exps = ['z2059\nHomeostatic', 'z2060\nExclusive', 'z2061\nAllostatic']
    scores = [7/8*100, 8/8*100, 12/12*100]
    acc = [96.6, 97.1, 99.2]
    t4_gap = [-0.6, 38.3, 40.2]
    gate_r = [0.676, 0.816, 0.780]

    x = np.arange(len(exps))
    w = 0.2

    bars1 = ax.bar(x - 1.5*w, scores, w, label='Pass Rate %', color=PASS_GREEN,
                   edgecolor=DARK, linewidth=0.5, zorder=3)
    bars2 = ax.bar(x - 0.5*w, acc, w, label='Accuracy %', color=BLUE,
                   edgecolor=DARK, linewidth=0.5, zorder=3)
    bars3 = ax.bar(x + 0.5*w, t4_gap, w, label='Self-Model Gap (pp)', color=ORANGE,
                   edgecolor=DARK, linewidth=0.5, zorder=3)
    bars4 = ax.bar(x + 1.5*w, [r*100 for r in gate_r], w, label='Gate-SCLK |r| ×100',
                   color=PURPLE, edgecolor=DARK, linewidth=0.5, zorder=3)

    ax.set_ylabel('Value')
    ax.set_title('Evolution: Perception → Routing → Action')
    ax.set_xticks(x)
    ax.set_xticklabels(exps, fontsize=10)
    ax.legend(loc='upper left', fontsize=8, ncol=2)
    ax.grid(axis='y', alpha=0.3)
    ax.axhline(y=0, color='black', linewidth=0.5)

    for bars in [bars1, bars2, bars3, bars4]:
        for bar in bars:
            h = bar.get_height()
            if abs(h) > 1:
                ax.text(bar.get_x() + bar.get_width()/2., h + (1 if h > 0 else -4),
                        f'{h:.0f}', ha='center', fontsize=7, fontweight='bold')

    # Right: what each experiment added
    ax = axes[1]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    ax.set_title('What Each Experiment Added', fontsize=12, fontweight='bold')

    y_pos = [8.5, 5.5, 2.0]
    titles = ['z2059: Homeostatic', 'z2060: Exclusive Paths', 'z2061: Closed Loop']
    descriptions = [
        'Self-model + gate + timing\nPerceives SCLK, adapts routing\nBUT: timing leaks → T4 FAIL',
        'Exclusive path specialization\nConflicting label schemes\nSelf-model causally necessary',
        'Effort head controls DVFS\nModel changes own clock speed\nDemand→effort→SCLK→perception→action'
    ]
    colors_t = [BLUE, PASS_GREEN, '#ff8c00']
    pass_rates = ['7/8', '8/8', '12/12']

    for y, title, desc, color, pr in zip(y_pos, titles, descriptions, colors_t, pass_rates):
        ax.add_patch(mpatches.FancyBboxPatch(
            (0.5, y-0.8), 9, 2.2, boxstyle='round,pad=0.2',
            facecolor='#f8f8ff', edgecolor=color, linewidth=2))
        ax.text(1, y+1, f'{title}  ({pr} PASS)', fontsize=11, fontweight='bold',
                color=color, va='center')
        ax.text(1, y-0.1, desc, fontsize=9, va='center', color=DARK, family='monospace')
        # Arrow between
        if y > 2.5:
            ax.annotate('', xy=(5, y-0.9), xytext=(5, y-1.4),
                        arrowprops=dict(arrowstyle='->', color=DARK, lw=2))

    plt.suptitle('z2059 → z2060 → z2061: From Perception to Action — Full Sensorimotor Loop',
                 fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'fig_z2061_evolution.png'))
    plt.close()
    print('  [+] fig_z2061_evolution.png')


# ============================================================
# 22. z2066/z2067: DEEP ANALOG EMBODIMENT
# ============================================================
def plot_z2067_deep_analog():
    """z2066→z2067 deep analog comparison: accuracy across conditions."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    conditions = ['A\nDeep', 'B\nBlind', 'E\nScrambled', 'F\nNo Self-M', 'G\nNo Effort', 'H\nAlways-Hi']
    colors_bar = [PASS_GREEN, GRAY, FAIL_RED, ORANGE, PURPLE, BLUE]

    # z2066 data (14/18 PASS)
    accs_66 = [0.9903, 0.2827, 0.1953, 0.5481, 0.0481, 0.0818]
    # z2067 data (17/18 PASS)
    accs_67 = [0.9695, 0.4929, 0.2321, 0.6166, 0.1116, 0.1133]

    # Left: z2066
    ax = axes[0]
    bars = ax.bar(range(len(conditions)), accs_66, color=colors_bar, edgecolor=DARK,
                  linewidth=0.5, zorder=3)
    ax.set_ylabel('Accuracy')
    ax.set_title('z2066: Deep Analog v1 (26-dim HW)\n14/18 PASS', fontweight='bold')
    ax.set_xticks(range(len(conditions)))
    ax.set_xticklabels(conditions, fontsize=9)
    ax.set_ylim(0, 1.2)
    ax.grid(axis='y', alpha=0.3)
    ax.axhline(y=0.5, color=DARK, linestyle='--', alpha=0.3)
    for bar, a in zip(bars, accs_66):
        ax.text(bar.get_x() + bar.get_width()/2., a + 0.015,
                f'{a:.1%}', ha='center', fontsize=8, fontweight='bold')
    # Annotate key gaps
    ax.annotate(f'A-F={100*(accs_66[0]-accs_66[3]):.0f}pp', xy=(3, accs_66[3]+0.01),
                fontsize=9, fontweight='bold', color=ORANGE, ha='center')
    ax.annotate(f'A-G={100*(accs_66[0]-accs_66[4]):.0f}pp', xy=(4, accs_66[4]+0.06),
                fontsize=9, fontweight='bold', color=PURPLE, ha='center')

    # Right: z2067
    ax = axes[1]
    bars = ax.bar(range(len(conditions)), accs_67, color=colors_bar, edgecolor=DARK,
                  linewidth=0.5, zorder=3)
    ax.set_ylabel('Accuracy')
    ax.set_title('z2067: Deep Analog v2 (27-dim HW, freq_est)\n17/18 PASS', fontweight='bold')
    ax.set_xticks(range(len(conditions)))
    ax.set_xticklabels(conditions, fontsize=9)
    ax.set_ylim(0, 1.2)
    ax.grid(axis='y', alpha=0.3)
    ax.axhline(y=0.5, color=DARK, linestyle='--', alpha=0.3)
    for bar, a in zip(bars, accs_67):
        ax.text(bar.get_x() + bar.get_width()/2., a + 0.015,
                f'{a:.1%}', ha='center', fontsize=8, fontweight='bold')
    ax.annotate(f'A-F={100*(accs_67[0]-accs_67[3]):.0f}pp', xy=(3, accs_67[3]+0.01),
                fontsize=9, fontweight='bold', color=ORANGE, ha='center')
    ax.annotate(f'A-G={100*(accs_67[0]-accs_67[4]):.0f}pp', xy=(4, accs_67[4]+0.06),
                fontsize=9, fontweight='bold', color=PURPLE, ha='center')

    plt.suptitle('Deep Analog Embodiment: 27-Dim Hardware Vector, DRM ioctl MMIO',
                 fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'fig_z2067_deep_analog.png'))
    plt.close()
    print('  [+] fig_z2067_deep_analog.png')


def plot_z2067_self_model():
    """z2067 self-model R² radar + gate/freq_est metrics."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Self-model R² values from z2067
    targets = ['sclk', 'timing', 'se0_busy', 'power', 'temp', 'grbm_load', 'freq_est']
    r2_vals = [0.5642, 0.9433, 0.6647, 0.0808, 0.7808, 0.7355, 0.8825]

    # Left: self-model bar chart
    ax = axes[0]
    colors_sm = [BLUE if v > 0.3 else GRAY if v > 0 else FAIL_RED for v in r2_vals]
    bars = ax.barh(range(len(targets)), r2_vals, color=colors_sm, edgecolor=DARK,
                   linewidth=0.5, zorder=3)
    ax.set_yticks(range(len(targets)))
    ax.set_yticklabels(targets, fontsize=10)
    ax.set_xlabel('R² (Self-Model Accuracy)')
    ax.set_title('z2067: Self-Model Predicts 13 Hardware Targets', fontweight='bold')
    ax.set_xlim(-0.1, 1.1)
    ax.axvline(x=0.15, color=FAIL_RED, linestyle='--', linewidth=1, alpha=0.5, label='Threshold (0.15)')
    ax.legend(loc='lower right', fontsize=9)
    ax.grid(axis='x', alpha=0.3)
    for bar, v in zip(bars, r2_vals):
        ax.text(max(v, 0) + 0.02, bar.get_y() + bar.get_height()/2.,
                f'{v:.3f}', ha='left', va='center', fontsize=9, fontweight='bold')

    # Right: gate/energy metrics comparison z2066 vs z2067
    ax = axes[1]
    metrics = ['Gate r', 'Gate\nhigh', 'Gate\nlow', 'Energy\nratio', 'freq_est\nR²']
    z66 = [0.9881, 0.5238, 0.4009, 0.5716, 0]  # z2066 has no freq_est
    z67 = [0.9375, 0.6788, 0.2732, 0.7045, 0.8825]

    x = np.arange(len(metrics))
    w = 0.35
    bars1 = ax.bar(x - w/2, z66, w, label='z2066 (14/18)', color=GRAY, edgecolor=DARK,
                   linewidth=0.5, zorder=3)
    bars2 = ax.bar(x + w/2, z67, w, label='z2067 (17/18)', color=PASS_GREEN, edgecolor=DARK,
                   linewidth=0.5, zorder=3)
    ax.set_ylabel('Value')
    ax.set_title('z2066 → z2067: Key Metric Improvements', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=9)
    ax.set_ylim(0, 1.15)
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.3)

    for bars in [bars1, bars2]:
        for bar in bars:
            h = bar.get_height()
            if h > 0:
                ax.text(bar.get_x() + bar.get_width()/2., h + 0.015,
                        f'{h:.3f}', ha='center', fontsize=8, fontweight='bold')

    # Annotate freq_est as NEW
    ax.annotate('NEW\nchannel', xy=(4 + w/2, 0.89), fontsize=8, fontweight='bold',
                color=PASS_GREEN, ha='center',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='#ddffdd', edgecolor=PASS_GREEN))

    plt.suptitle('z2067: Deep Analog v2 — Self-Model & Gate Metrics',
                 fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'fig_z2067_self_model.png'))
    plt.close()
    print('  [+] fig_z2067_self_model.png')


def plot_z2067_evolution():
    """z2059 → z2060 → z2061 → z2066 → z2067: Full evolution timeline."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Left: pass rate + accuracy evolution
    ax = axes[0]
    exps = ['z2059\n7/8', 'z2060\n8/8', 'z2061\n12/12', 'z2066\n14/18', 'z2067\n17/18']
    total_tests = [8, 8, 12, 18, 18]
    pass_counts = [7, 8, 12, 14, 17]
    pass_pct = [p/t*100 for p, t in zip(pass_counts, total_tests)]
    acc = [96.6, 97.1, 99.2, 99.0, 96.9]
    t4_gap = [-0.6, 38.3, 40.2, 44.2, 35.3]
    gate_r = [0.676, 0.816, 0.780, 0.988, 0.938]
    hw_dim = [7, 7, 7, 26, 27]

    x = np.arange(len(exps))
    w = 0.15

    bars1 = ax.bar(x - 2*w, pass_pct, w, label='Pass Rate %', color=PASS_GREEN,
                   edgecolor=DARK, linewidth=0.5, zorder=3)
    bars2 = ax.bar(x - w, acc, w, label='Accuracy %', color=BLUE,
                   edgecolor=DARK, linewidth=0.5, zorder=3)
    bars3 = ax.bar(x, t4_gap, w, label='Self-Model Gap (pp)', color=ORANGE,
                   edgecolor=DARK, linewidth=0.5, zorder=3)
    bars4 = ax.bar(x + w, [r*100 for r in gate_r], w, label='Gate |r| ×100',
                   color=PURPLE, edgecolor=DARK, linewidth=0.5, zorder=3)
    bars5 = ax.bar(x + 2*w, hw_dim, w, label='HW Dim', color='#1abc9c',
                   edgecolor=DARK, linewidth=0.5, zorder=3)

    ax.set_ylabel('Value')
    ax.set_title('Hardware Integration Evolution')
    ax.set_xticks(x)
    ax.set_xticklabels(exps, fontsize=9)
    ax.legend(loc='upper left', fontsize=7, ncol=3)
    ax.grid(axis='y', alpha=0.3)
    ax.axhline(y=0, color='black', linewidth=0.5)

    for bars in [bars1, bars2, bars3, bars4, bars5]:
        for bar in bars:
            h = bar.get_height()
            if abs(h) > 1:
                ax.text(bar.get_x() + bar.get_width()/2., h + (1 if h > 0 else -5),
                        f'{h:.0f}', ha='center', fontsize=6, fontweight='bold')

    # Right: what each added
    ax = axes[1]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 12)
    ax.axis('off')
    ax.set_title('What Each Experiment Added', fontsize=12, fontweight='bold')

    entries = [
        ('z2059: Homeostatic', '7/8', 'Self-model + gate + timing\nFirst HW-bound consciousness', BLUE, 10.5),
        ('z2060: Exclusive Paths', '8/8', 'Conflicting label schemes\nSelf-model causally necessary', PASS_GREEN, 8.2),
        ('z2061: Closed Loop', '12/12', 'Effort head controls DVFS\nFirst NN controlling own clock', '#ff8c00', 5.9),
        ('z2066: Deep Analog', '14/18', '26-dim HW: ISA regs + MMIO\n6 new channels (STATUS, CYCLES...)', PURPLE, 3.6),
        ('z2067: Deep Analog v2', '17/18', 'DRM ioctl + freq_est channel\n27-dim, 13-target self-model', '#e74c3c', 1.3),
    ]

    for title, pr, desc, color, y in entries:
        ax.add_patch(mpatches.FancyBboxPatch(
            (0.3, y-0.5), 9.4, 1.8, boxstyle='round,pad=0.2',
            facecolor='#f8f8ff', edgecolor=color, linewidth=2))
        ax.text(0.7, y+0.9, f'{title}  ({pr} PASS)', fontsize=10, fontweight='bold',
                color=color, va='center')
        ax.text(0.7, y+0.0, desc, fontsize=8, va='center', color=DARK, family='monospace')
        if y < 10:
            ax.annotate('', xy=(5, y+1.3), xytext=(5, y+1.7),
                        arrowprops=dict(arrowstyle='->', color=DARK, lw=2))

    plt.suptitle('z2059 → z2067: From Perception to 27-Dimensional Hardware Embodiment',
                 fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'fig_z2067_evolution.png'))
    plt.close()
    print('  [+] fig_z2067_evolution.png')


def plot_z2067_freq_est():
    """z2067 freq_est: the key new channel that fixed z2066's failures."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: freq_est differentiation (high vs low SCLK)
    ax = axes[0]
    states = ['Low SCLK\n(~600 MHz)', 'High SCLK\n(~1900 MHz)']
    freq_est = [463485.4, 1002090.3]
    colors_fe = [BLUE, PASS_GREEN]

    bars = ax.bar(states, freq_est, color=colors_fe, edgecolor=DARK,
                  linewidth=0.5, zorder=3, width=0.5)
    for bar, v in zip(bars, freq_est):
        ax.text(bar.get_x() + bar.get_width()/2., v + 15000,
                f'{v:,.0f}\ncycles/ms', ha='center', fontsize=10, fontweight='bold')

    ax.set_ylabel('freq_est (cycles / ms)')
    ax.set_title('freq_est Differentiates SCLK States\n(p = 0.003, 2.2x ratio)', fontweight='bold')
    ax.set_ylim(0, 1200000)
    ax.grid(axis='y', alpha=0.3)

    # Annotate key insight
    ax.text(0.5, 0.85, 'SHADER_CYCLES constant (~484k)\nWall time varies with SCLK\nfreq_est = cycles / wall_ms',
            transform=ax.transAxes, fontsize=9, ha='center', va='center',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='lightyellow', edgecolor=DARK),
            family='monospace')

    # Right: z2066 failures vs z2067 fixes
    ax = axes[1]
    tests = ['T2\nsclk\nself-model', 'T12\nGRBM\nvaries', 'T13\ncycle\nR²', 'T15\ncycle\np-val']
    z66_vals = [0.145, 1, -0.569, 0.868]  # z2066 values (all FAIL)
    z67_vals = [0.883, 3, 0.883, 0.003]   # z2067 values (all PASS)
    thresholds = [0.15, 2, 0.1, 0.05]

    x = np.arange(len(tests))
    w = 0.35

    bars1 = ax.bar(x - w/2, z66_vals, w, label='z2066 (FAIL)', color=FAIL_RED,
                   edgecolor=DARK, linewidth=0.5, zorder=3, alpha=0.7)
    bars2 = ax.bar(x + w/2, z67_vals, w, label='z2067 (PASS)', color=PASS_GREEN,
                   edgecolor=DARK, linewidth=0.5, zorder=3)
    ax.set_ylabel('Test Value')
    ax.set_title('4 Failures Fixed: z2066 → z2067', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(tests, fontsize=9)
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.3)
    ax.axhline(y=0, color='black', linewidth=0.5)

    for bars in [bars1, bars2]:
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., h + (0.03 if h >= 0 else -0.08),
                    f'{h:.3f}', ha='center', fontsize=7, fontweight='bold')

    plt.suptitle('z2067: freq_est Channel — Key Innovation for Deep Analog Embodiment',
                 fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'fig_z2067_freq_est.png'))
    plt.close()
    print('  [+] fig_z2067_freq_est.png')


# ============================================================
# MAIN
# ============================================================
if __name__ == '__main__':
    print('Generating paper plots for sections 6.3+...')
    # Section 6.3: z1900 falsification
    plot_z1901_falsification()
    plot_z1990_unified()
    # Section 7: z2000-z2018
    plot_granger_causality()
    plot_pci_progression()
    # Section 8: z2020-z2037
    plot_blindsight()
    plot_overflow()
    plot_synergy()
    plot_pci_inversion()
    plot_workspace_necessity()
    plot_scorecard()
    # Section 9: z2039-z2040
    plot_z2040_embodied()
    # Section 10: z2055-z2058
    plot_analog_gate_trajectory()
    plot_dual_channel_hierarchy()
    plot_wall_clock_timing()
    plot_three_channel()
    plot_hw_integration_summary()
    # Section 10.6-10.7: z2059-z2060
    plot_z2060_ablation()
    plot_z2060_comparison()
    # Section 10.8: z2061
    plot_z2061_allostatic()
    plot_z2061_evolution()
    # Section 10.9: z2066-z2067
    plot_z2067_deep_analog()
    plot_z2067_self_model()
    plot_z2067_freq_est()
    plot_z2067_evolution()
    # Section 11: Discussion
    plot_design_pattern()
    print(f'\nAll 25 plots saved to {OUT_DIR}')
