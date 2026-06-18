#!/usr/bin/env python3
"""Generate plots for z2050-z2052 hardware integration results for the paper."""

import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

ROOT = Path(__file__).parent.parent
RESULTS = ROOT / 'results'
FIGURES = RESULTS / 'figures'
FIGURES.mkdir(exist_ok=True)

# Color scheme
C_PASS = '#2ecc71'
C_FAIL = '#e74c3c'
C_EMBOD = '#3498db'
C_BLIND = '#95a5a6'
C_SCRAMBLE = '#e74c3c'
C_RANDOM = '#f39c12'
C_DIGITAL = '#9b59b6'
C_FIXED = '#1abc9c'

# =============================================================================
# Figure 1: z2050-z2052 Accuracy Comparison (Bar Chart)
# =============================================================================
def fig_hw_accuracy():
    fig, axes = plt.subplots(1, 3, figsize=(14, 5), sharey=True)

    # z2050
    z50 = json.load(open(RESULTS / 'z2050_fpga_like_gpu.json'))
    conds_50 = ['A_embodied', 'B_blind', 'C_random', 'D_fixed', 'E_scrambled']
    labels_50 = ['Embodied\n(WGP bank)', 'Blind\n(no HW)', 'Random\n(rand bank)', 'Fixed\n(bank 0)', 'Scrambled\n(wrong map)']
    colors_50 = [C_EMBOD, C_BLIND, C_RANDOM, C_FIXED, C_SCRAMBLE]
    accs_50 = [z50['conditions'][c]['eval']['accuracy'] * 100 for c in conds_50]

    bars = axes[0].bar(range(len(conds_50)), accs_50, color=colors_50, edgecolor='black', linewidth=0.5)
    axes[0].set_xticks(range(len(conds_50)))
    axes[0].set_xticklabels(labels_50, fontsize=8)
    axes[0].set_ylabel('Accuracy (%)', fontsize=12)
    axes[0].set_title('z2050: FPGA-Like GPU\n(5/5 PASS)', fontsize=11, fontweight='bold')
    axes[0].axhline(y=50, color='gray', linestyle='--', alpha=0.5, label='Chance')
    axes[0].set_ylim(0, 105)
    for bar, acc in zip(bars, accs_50):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                     f'{acc:.1f}%', ha='center', va='bottom', fontsize=8, fontweight='bold')

    # z2051
    z51 = json.load(open(RESULTS / 'z2051_silicon_puf_neural_net.json'))
    conds_51 = ['A_embodied', 'B_blind', 'C_digital', 'D_analog', 'E_scrambled', 'F_random']
    labels_51 = ['Embodied\n(dig+ana)', 'Blind\n(no HW)', 'Digital\n(WGP only)', 'Analog\n(timing)', 'Scrambled\n(wrong map)', 'Random\n(rand bank)']
    colors_51 = [C_EMBOD, C_BLIND, C_DIGITAL, '#e67e22', C_SCRAMBLE, C_RANDOM]
    accs_51 = [z51['conditions'][c]['eval']['accuracy'] * 100 for c in conds_51]

    bars = axes[1].bar(range(len(conds_51)), accs_51, color=colors_51, edgecolor='black', linewidth=0.5)
    axes[1].set_xticks(range(len(conds_51)))
    axes[1].set_xticklabels(labels_51, fontsize=7)
    axes[1].set_title('z2051: Silicon PUF\n(7/7 PASS)', fontsize=11, fontweight='bold')
    axes[1].axhline(y=50, color='gray', linestyle='--', alpha=0.5)
    for bar, acc in zip(bars, accs_51):
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                     f'{acc:.1f}%', ha='center', va='bottom', fontsize=7, fontweight='bold')

    # z2052 — different structure: eval_scheduler / eval_cu_masked
    z52 = json.load(open(RESULTS / 'z2052_cu_placed_lds_persistent.json'))
    conds_52 = ['A_placed', 'B_scheduler', 'C_blind', 'D_wrong_cu']
    labels_52 = ['CU-Placed\n(mask)', 'Scheduler\n(natural)', 'Blind\n(no HW)', 'Wrong CU\n(scrambled)']
    colors_52 = [C_EMBOD, C_DIGITAL, C_BLIND, C_SCRAMBLE]
    def get_z52_acc(cond_data):
        for key in ['eval', 'eval_scheduler', 'eval_cu_masked']:
            if key in cond_data and 'accuracy' in cond_data[key]:
                return cond_data[key]['accuracy'] * 100
        return 0
    accs_52 = [get_z52_acc(z52['conditions'][c]) for c in conds_52]

    bars = axes[2].bar(range(len(conds_52)), accs_52, color=colors_52, edgecolor='black', linewidth=0.5)
    axes[2].set_xticks(range(len(conds_52)))
    axes[2].set_xticklabels(labels_52, fontsize=8)
    axes[2].set_title('z2052: CU-Placed\n(7/8 PASS)', fontsize=11, fontweight='bold')
    axes[2].axhline(y=50, color='gray', linestyle='--', alpha=0.5)
    for bar, acc in zip(bars, accs_52):
        axes[2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                     f'{acc:.1f}%', ha='center', va='bottom', fontsize=8, fontweight='bold')

    plt.tight_layout()
    plt.savefig(FIGURES / 'fig_hw_integration_accuracy.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved fig_hw_integration_accuracy.png")


# =============================================================================
# Figure 2: Kill Shot Comparison (Embodied vs Scrambled)
# =============================================================================
def fig_kill_shots():
    fig, ax = plt.subplots(figsize=(10, 5))

    experiments = ['z2050\nFPGA-Like', 'z2051\nSilicon PUF\n(ikaros)', 'z2051\nSilicon PUF\n(daedalus)',
                   'z2052\nCU-Placed\n(ikaros)', 'z2052\nCU-Placed\n(daedalus)']

    z50 = json.load(open(RESULTS / 'z2050_fpga_like_gpu.json'))
    z51_ik = json.load(open(RESULTS / 'z2051_silicon_puf_neural_net.json'))
    z51_da = json.load(open(RESULTS / 'z2051_daedalus_silicon_puf.json'))
    z52_ik = json.load(open(RESULTS / 'z2052_cu_placed_lds_persistent.json'))

    def safe_acc(cond, pct=True):
        for key in ['eval', 'eval_scheduler', 'eval_cu_masked']:
            if key in cond and 'accuracy' in cond[key]:
                return cond[key]['accuracy'] * (100 if pct else 1)
        return 0

    embodied = [
        safe_acc(z50['conditions']['A_embodied']),
        safe_acc(z51_ik['conditions']['A_embodied']),
        safe_acc(z51_da['conditions']['A_embodied']),
        safe_acc(z52_ik['conditions']['A_placed']),
        98.88,  # daedalus z2052
    ]
    scrambled = [
        safe_acc(z50['conditions']['E_scrambled']),
        safe_acc(z51_ik['conditions']['E_scrambled']),
        safe_acc(z51_da['conditions']['E_scrambled']),
        safe_acc(z52_ik['conditions']['D_wrong_cu']),
        0.29,  # daedalus z2052
    ]

    x = np.arange(len(experiments))
    width = 0.35

    bars1 = ax.bar(x - width/2, embodied, width, label='Embodied (correct routing)',
                   color=C_EMBOD, edgecolor='black', linewidth=0.5)
    bars2 = ax.bar(x + width/2, scrambled, width, label='Scrambled (wrong routing)',
                   color=C_SCRAMBLE, edgecolor='black', linewidth=0.5)

    for bar, val in zip(bars1, embodied):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{val:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')
    for bar, val in zip(bars2, scrambled):
        ax.text(bar.get_x() + bar.get_width()/2, max(bar.get_height(), 2) + 1,
                f'{val:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold',
                color=C_SCRAMBLE)

    # Draw gap arrows
    for i in range(len(experiments)):
        gap = embodied[i] - scrambled[i]
        ax.annotate(f'{gap:.1f}pp\ngap',
                    xy=(x[i], (embodied[i] + scrambled[i])/2),
                    fontsize=8, ha='center', color='#2c3e50', fontweight='bold')

    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_title('Kill Shot: Correct vs Wrong Hardware Routing\nAcross z2050-z2052 (5 experiments, 2 machines)', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(experiments, fontsize=9)
    ax.legend(fontsize=10)
    ax.axhline(y=50, color='gray', linestyle='--', alpha=0.3, label='Chance')
    ax.axhline(y=10, color='red', linestyle=':', alpha=0.3, label='Random guessing')
    ax.set_ylim(0, 110)

    plt.tight_layout()
    plt.savefig(FIGURES / 'fig_hw_kill_shots.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved fig_hw_kill_shots.png")


# =============================================================================
# Figure 3: Energy Efficiency (z2051)
# =============================================================================
def fig_energy():
    fig, ax = plt.subplots(figsize=(8, 5))

    z51_ik = json.load(open(RESULTS / 'z2051_silicon_puf_neural_net.json'))

    conditions = ['A_embodied', 'B_blind', 'C_digital', 'D_analog', 'F_random']
    labels = ['Embodied\n(dig+ana)', 'Blind\n(no HW)', 'Digital\n(WGP only)', 'Analog\n(timing)', 'Random\n(rand bank)']
    colors = [C_EMBOD, C_BLIND, C_DIGITAL, '#e67e22', C_RANDOM]

    energies = []
    accs = []
    for c in conditions:
        e = z51_ik['conditions'][c]['eval'].get('joules_per_correct', 0) * 1000  # mJ
        a = z51_ik['conditions'][c]['eval']['accuracy'] * 100
        energies.append(e)
        accs.append(a)

    bars = ax.bar(range(len(conditions)), energies, color=colors, edgecolor='black', linewidth=0.5)

    for bar, e, a in zip(bars, energies, accs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{e:.2f} mJ\n({a:.0f}%)', ha='center', va='bottom', fontsize=9, fontweight='bold')

    ax.set_xticks(range(len(conditions)))
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel('Energy per Correct Answer (mJ)', fontsize=11)
    ax.set_title('z2051: Energy Efficiency — Embodied is 37% More Efficient\nThan Blind Despite Both Achieving High Accuracy', fontsize=11, fontweight='bold')

    # Highlight efficiency gain
    if len(energies) >= 2:
        ratio = (energies[1] - energies[0]) / energies[1] * 100
        ax.annotate(f'{ratio:.0f}% less energy\n(embodied vs blind)',
                    xy=(0.5, (energies[0] + energies[1])/2),
                    fontsize=10, ha='center', color='green', fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', edgecolor='green'))

    plt.tight_layout()
    plt.savefig(FIGURES / 'fig_hw_energy.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved fig_hw_energy.png")


# =============================================================================
# Figure 4: Cross-Machine Replication (ikaros vs daedalus)
# =============================================================================
def fig_cross_machine():
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # z2051 cross-machine
    z51_ik = json.load(open(RESULTS / 'z2051_silicon_puf_neural_net.json'))
    z51_da = json.load(open(RESULTS / 'z2051_daedalus_silicon_puf.json'))

    conds = ['A_embodied', 'B_blind', 'E_scrambled']
    labels = ['Embodied', 'Blind', 'Scrambled']
    ik_51 = [z51_ik['conditions'][c]['eval']['accuracy'] * 100 for c in conds]
    da_51 = [z51_da['conditions'][c]['eval']['accuracy'] * 100 for c in conds]

    x = np.arange(len(conds))
    w = 0.35
    axes[0].bar(x - w/2, ik_51, w, label='ikaros', color='#3498db', edgecolor='black', linewidth=0.5)
    axes[0].bar(x + w/2, da_51, w, label='daedalus', color='#e67e22', edgecolor='black', linewidth=0.5)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, fontsize=10)
    axes[0].set_ylabel('Accuracy (%)', fontsize=11)
    axes[0].set_title('z2051: Silicon PUF\n(ikaros vs daedalus)', fontsize=11, fontweight='bold')
    axes[0].legend(fontsize=10)
    axes[0].set_ylim(0, 110)
    axes[0].axhline(y=50, color='gray', linestyle='--', alpha=0.3)

    for i, (ik, da) in enumerate(zip(ik_51, da_51)):
        axes[0].text(x[i] - w/2, ik + 1, f'{ik:.1f}', ha='center', fontsize=8, fontweight='bold')
        axes[0].text(x[i] + w/2, da + 1, f'{da:.1f}', ha='center', fontsize=8, fontweight='bold')

    # z2052 cross-machine
    z52_ik = json.load(open(RESULTS / 'z2052_cu_placed_lds_persistent.json'))
    conds_52 = ['A_placed', 'C_blind', 'D_wrong_cu']
    labels_52 = ['CU-Placed', 'Blind', 'Wrong CU']
    def get_z52_acc(cond_data):
        for key in ['eval', 'eval_scheduler', 'eval_cu_masked']:
            if key in cond_data and 'accuracy' in cond_data[key]:
                return cond_data[key]['accuracy'] * 100
        return 0
    ik_52 = [get_z52_acc(z52_ik['conditions'][c]) for c in conds_52]
    da_52 = [98.88, 50.62, 0.29]  # daedalus results from output

    x2 = np.arange(len(conds_52))
    axes[1].bar(x2 - w/2, ik_52, w, label='ikaros', color='#3498db', edgecolor='black', linewidth=0.5)
    axes[1].bar(x2 + w/2, da_52, w, label='daedalus', color='#e67e22', edgecolor='black', linewidth=0.5)
    axes[1].set_xticks(x2)
    axes[1].set_xticklabels(labels_52, fontsize=10)
    axes[1].set_title('z2052: CU-Placed\n(ikaros vs daedalus)', fontsize=11, fontweight='bold')
    axes[1].legend(fontsize=10)
    axes[1].set_ylim(0, 110)
    axes[1].axhline(y=50, color='gray', linestyle='--', alpha=0.3)

    for i, (ik, da) in enumerate(zip(ik_52, da_52)):
        axes[1].text(x2[i] - w/2, max(ik, 3) + 1, f'{ik:.1f}', ha='center', fontsize=8, fontweight='bold')
        axes[1].text(x2[i] + w/2, max(da, 3) + 1, f'{da:.1f}', ha='center', fontsize=8, fontweight='bold')

    plt.suptitle('Cross-Machine Replication: Two AMD gfx1151 GPUs', fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(FIGURES / 'fig_hw_cross_machine.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved fig_hw_cross_machine.png")


# =============================================================================
# Figure 5: FPGA Isomorphism Diagram
# =============================================================================
def fig_fpga_isomorphism():
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 8)
    ax.axis('off')

    # FPGA side
    fpga_box = mpatches.FancyBboxPatch((0.5, 1), 4, 6, boxstyle="round,pad=0.2",
                                        facecolor='#ebf5fb', edgecolor='#2980b9', linewidth=2)
    ax.add_patch(fpga_box)
    ax.text(2.5, 6.7, 'FPGA', fontsize=14, fontweight='bold', ha='center', color='#2980b9')

    # GPU side
    gpu_box = mpatches.FancyBboxPatch((5.5, 1), 4, 6, boxstyle="round,pad=0.2",
                                       facecolor='#fef9e7', edgecolor='#f39c12', linewidth=2)
    ax.add_patch(gpu_box)
    ax.text(7.5, 6.7, 'GPU (z2050)', fontsize=14, fontweight='bold', ha='center', color='#f39c12')

    # Rows
    rows = [
        ('CLB Location', 'WGP_id\n(s_getreg_b32 hwreg(23))', 5.8),
        ('LUT Content', 'Weight Bank\n(learned params)', 4.3),
        ('Place & Route', 'GPU Scheduler\n(hardware assigns)', 2.8),
    ]

    for fpga_label, gpu_label, y in rows:
        # FPGA item
        ax.text(2.5, y, fpga_label, fontsize=11, ha='center', va='center',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='#3498db'))
        # GPU item
        ax.text(7.5, y, gpu_label, fontsize=10, ha='center', va='center',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='#e67e22'))
        # Arrow
        ax.annotate('', xy=(5.6, y), xytext=(4.4, y),
                    arrowprops=dict(arrowstyle='->', color='#2c3e50', lw=2))
        ax.text(5.0, y + 0.3, '≡', fontsize=16, ha='center', fontweight='bold', color='#2c3e50')

    ax.text(5.0, 0.5, 'Physical location determines computation — hardware IS software',
            fontsize=11, ha='center', fontstyle='italic', color='#2c3e50')

    plt.tight_layout()
    plt.savefig(FIGURES / 'fig_fpga_isomorphism.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved fig_fpga_isomorphism.png")


# =============================================================================
# Figure 6: Depth Hierarchy
# =============================================================================
def fig_depth_hierarchy():
    fig, ax = plt.subplots(figsize=(10, 6))

    levels = [
        ('FPGA gates\n(gate delay IS output)', 100, '#27ae60', 'z1100+\n(Artix-7)'),
        ('s_getreg_b32 hwreg(23)\n(physical CU identity)', 90, '#2ecc71', 'z2050-z2052'),
        ('hipExtStreamCreateWithCUMask\n(CU placement control)', 80, '#3498db', 'z2052'),
        ('HIP __clock64()\n(timing inside kernel)', 65, '#9b59b6', 'z2047-z2048'),
        ('DRM ioctl / debugfs MMIO\n(register reads via driver)', 50, '#e67e22', 'z2043/z2049'),
        ('sysfs SCLK/power\n(filesystem interface)', 35, '#f39c12', 'z2046'),
        ('Python telemetry\n(temperature via reads)', 20, '#e74c3c', 'z900-z907'),
    ]

    for i, (label, depth, color, exps) in enumerate(levels):
        bar = ax.barh(i, depth, color=color, edgecolor='black', linewidth=0.5, height=0.7)
        ax.text(depth + 1, i, f'{label}  [{exps}]', va='center', fontsize=9)

    ax.set_yticks(range(len(levels)))
    ax.set_yticklabels(['' for _ in levels])
    ax.set_xlabel('Depth (closer to silicon)', fontsize=11)
    ax.set_title('Hardware Integration Depth Hierarchy\n(deepest at top, shallowest at bottom)', fontsize=12, fontweight='bold')
    ax.invert_yaxis()
    ax.set_xlim(0, 110)

    # Add annotation
    ax.axvline(x=80, color='green', linestyle=':', alpha=0.4)
    ax.text(81, len(levels) - 0.5, 'FPGA-equivalent\ndepth', fontsize=8, color='green', fontstyle='italic')

    plt.tight_layout()
    plt.savefig(FIGURES / 'fig_hw_depth_hierarchy.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved fig_hw_depth_hierarchy.png")


# =============================================================================
# Figure 7: WGP Distribution (z2050)
# =============================================================================
def fig_wgp_distribution():
    fig, ax = plt.subplots(figsize=(8, 5))

    z50 = json.load(open(RESULTS / 'z2050_fpga_like_gpu.json'))
    banks = z50['conditions']['A_embodied']['eval']['bank_counts']
    wgps = z50['config']['wgp_values']

    colors = plt.cm.Set3(np.linspace(0, 1, len(wgps)))
    bars = ax.bar(range(len(wgps)), banks, color=colors, edgecolor='black', linewidth=0.5)

    for bar, count, wgp in zip(bars, banks, wgps):
        parity = 'even' if (wgps.index(wgp) % 2 == 0) else 'odd'
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 20,
                f'WGP {wgp}\n({count})\n[{parity}]', ha='center', va='bottom', fontsize=8)

    ax.set_xlabel('Bank Index', fontsize=11)
    ax.set_ylabel('Samples Assigned', fontsize=11)
    ax.set_title('z2050: WGP Distribution Across 8 Banks\n(GPU scheduler assigns blocks to physical WGPs)', fontsize=11, fontweight='bold')
    ax.set_xticks(range(len(wgps)))
    ax.set_xticklabels([f'Bank {i}' for i in range(len(wgps))], fontsize=9)

    # Show even/odd grouping
    for i in range(len(wgps)):
        if i % 2 == 0:
            ax.axvspan(i - 0.4, i + 0.4, alpha=0.08, color='blue')
        else:
            ax.axvspan(i - 0.4, i + 0.4, alpha=0.08, color='red')

    ax.text(0.02, 0.95, 'Blue: identity labels (even banks)\nRed: shifted labels (odd banks)',
            transform=ax.transAxes, fontsize=9, va='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(FIGURES / 'fig_hw_wgp_distribution.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved fig_hw_wgp_distribution.png")


if __name__ == '__main__':
    print("Generating hardware integration figures...")
    fig_hw_accuracy()
    fig_kill_shots()
    fig_energy()
    fig_cross_machine()
    fig_fpga_isomorphism()
    fig_depth_hierarchy()
    fig_wgp_distribution()
    print(f"\nAll figures saved to {FIGURES}/")
