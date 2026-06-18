#!/usr/bin/env python3
"""Generate dark-background bridge diagrams for FEEL paper Section 11."""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

# ============================================================
# Style setup — dark background, clean modern look
# ============================================================
plt.rcParams.update({
    'figure.facecolor': '#0d1117',
    'axes.facecolor': '#0d1117',
    'text.color': '#e6edf3',
    'axes.labelcolor': '#e6edf3',
    'xtick.color': '#8b949e',
    'ytick.color': '#8b949e',
    'font.family': 'sans-serif',
    'font.size': 11,
})

# Color palette
C = {
    'bg': '#0d1117',
    'card': '#161b22',
    'border': '#30363d',
    'accent1': '#58a6ff',  # blue
    'accent2': '#3fb950',  # green
    'accent3': '#d29922',  # amber
    'accent4': '#f85149',  # red
    'accent5': '#bc8cff',  # purple
    'accent6': '#79c0ff',  # light blue
    'text': '#e6edf3',
    'dim': '#8b949e',
    'glow_blue': '#1f6feb',
    'glow_green': '#238636',
    'glow_amber': '#9e6a03',
}

OUTDIR = 'scripts/img/diagrams'


def rounded_box(ax, xy, w, h, color, label='', sublabel='',
                border_color=None, alpha=0.85, fontsize=11, lw=1.5):
    """Draw a rounded rectangle with optional label."""
    if border_color is None:
        border_color = color
    box = FancyBboxPatch(xy, w, h, boxstyle="round,pad=0.12",
                         facecolor=color, edgecolor=border_color,
                         linewidth=lw, alpha=alpha, zorder=2)
    ax.add_patch(box)
    cx, cy = xy[0] + w/2, xy[1] + h/2
    if label:
        offset = 0.08 if sublabel else 0
        ax.text(cx, cy + offset, label, ha='center', va='center',
                fontsize=fontsize, fontweight='bold', color=C['text'], zorder=3)
    if sublabel:
        ax.text(cx, cy - 0.15, sublabel, ha='center', va='center',
                fontsize=fontsize - 2, color=C['dim'], zorder=3,
                style='italic')


def arrow(ax, start, end, color=C['accent1'], lw=1.8, style='->', connectionstyle='arc3,rad=0'):
    """Draw an arrow between two points."""
    ax.annotate('', xy=end, xytext=start,
                arrowprops=dict(arrowstyle=style, color=color, lw=lw,
                                connectionstyle=connectionstyle),
                zorder=4)


# ============================================================
# DIAGRAM 1: The Constitutivity Spectrum
# ============================================================
def diagram_constitutivity_spectrum():
    fig, ax = plt.subplots(figsize=(16, 7))
    ax.set_xlim(-0.5, 16)
    ax.set_ylim(-1.5, 6.5)
    ax.axis('off')

    # Title
    ax.text(8, 6.0, 'THE CONSTITUTIVITY SPECTRUM',
            ha='center', va='center', fontsize=20, fontweight='bold',
            color=C['accent1'])
    ax.text(8, 5.5, 'From software abstraction to biological substrate',
            ha='center', va='center', fontsize=12, color=C['dim'])

    # Gradient bar at bottom
    gradient = np.linspace(0, 1, 256).reshape(1, -1)
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list('spec',
        [C['accent4'], C['accent3'], C['accent2'], C['accent1'], C['accent5']])
    ax.imshow(gradient, aspect='auto', cmap=cmap,
              extent=[0.5, 15.5, -1.2, -0.6], zorder=1, alpha=0.7)
    ax.text(0.5, -1.5, 'Substrate-independent', fontsize=9, color=C['accent4'],
            ha='left', va='center')
    ax.text(15.5, -1.5, 'Substrate IS computation', fontsize=9, color=C['accent5'],
            ha='right', va='center')

    # The stages
    stages = [
        (1.0, 'Software\nConditioning', 'z907: p=1.0\n(FALSIFIED)', C['accent4'], ''),
        (3.8, 'Sysfs\nTelemetry', 'z2042-z2070\nWeak coupling', C['accent3'], ''),
        (6.6, 'ISA Register\nCoupling', 'z2076-z2115\n99.0pp kill-shot', C['accent3'], ''),
        (9.4, 'MODE Writes\n+ Pulse Field', 'z2115-z2138\n10/10 bridge', C['accent2'], ''),
        (12.2, 'NS-RAM\n(SPICE + FPGA)', '21/21 metrics\nVt = kT/q', C['accent1'], ''),
    ]

    for x, title, detail, color, _ in stages:
        # Vertical line from bar to box
        ax.plot([x + 1.0, x + 1.0], [-0.6, 0.5], color=color, lw=1.5,
                alpha=0.6, zorder=2)
        ax.plot(x + 1.0, -0.6, 'o', color=color, markersize=8, zorder=3)
        rounded_box(ax, (x, 0.5), 2.0, 1.6, C['card'],
                    label=title, border_color=color, fontsize=10)
        ax.text(x + 1.0, 0.2, detail, ha='center', va='top',
                fontsize=8, color=C['dim'], zorder=3)
        # Move detail below box? No, put inside lower half
        # Actually put detail inside box below title
        ax.text(x + 1.0, 0.75, detail, ha='center', va='center',
                fontsize=8, color=C['dim'], zorder=3)

    # Arrows between stages
    for i in range(len(stages) - 1):
        x1 = stages[i][0] + 2.0
        x2 = stages[i+1][0]
        y = 1.3
        arrow(ax, (x1, y), (x2, y), color=C['dim'], lw=1.2)

    # Top annotation: three validation layers
    for x, w, label, color, score in [
        (1.0, 4.6, 'High-Level SPICE (v6/v7)', C['accent2'], '5/5'),
        (6.6, 4.6, 'ISA Kernel Bridge (v8-v10)', C['accent1'], '10/10'),
        (12.2, 2.0, 'FPGA HW (E1-E6)', C['accent5'], '6/6'),
    ]:
        y_top = 3.5
        rounded_box(ax, (x, y_top), w, 0.9, C['card'],
                    label=f'{label}', sublabel=f'Score: {score}',
                    border_color=color, fontsize=10)

    # Total badge
    rounded_box(ax, (6.0, 4.8), 4.0, 0.7, C['glow_green'],
                label='TOTAL: 21/21 BRIDGE METRICS PASS',
                border_color=C['accent2'], fontsize=12, alpha=0.9)

    fig.tight_layout()
    fig.savefig(f'{OUTDIR}/constitutivity_spectrum.png', dpi=200,
                bbox_inches='tight', facecolor=C['bg'])
    plt.close()
    print(f'  Saved {OUTDIR}/constitutivity_spectrum.png')


# ============================================================
# DIAGRAM 2: ISA-to-Circuit Mapping (v8, v9, v10)
# ============================================================
def diagram_isa_circuit_mapping():
    fig, axes = plt.subplots(1, 3, figsize=(18, 8))

    for ax in axes:
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 12)
        ax.axis('off')
        ax.set_facecolor(C['bg'])

    # ---- v8: MODE Register ↔ Vg Switch ----
    ax = axes[0]
    ax.text(5, 11.5, 'v8: MODE Register', ha='center', fontsize=14,
            fontweight='bold', color=C['accent1'])
    ax.text(5, 11.0, 'ISA rounding mode → Vg switch', ha='center',
            fontsize=10, color=C['dim'])

    # FEEL side (left column)
    ax.text(2.5, 10.0, 'FEEL (GPU)', ha='center', fontsize=11,
            fontweight='bold', color=C['accent3'])
    rounded_box(ax, (0.3, 8.5), 4.4, 1.2, C['card'],
                label='s_setreg hwreg(MODE,2,1)',
                border_color=C['accent3'], fontsize=9)
    ax.text(2.5, 8.3, 'Changes FP16 rounding mid-GEMM', ha='center',
            fontsize=8, color=C['dim'])
    rounded_box(ax, (0.3, 6.5), 4.4, 1.2, C['card'],
                label='12/12 math fingerprints',
                border_color=C['accent3'], fontsize=9)
    ax.text(2.5, 6.3, 'σ=0.0106 PPL across modes', ha='center',
            fontsize=8, color=C['dim'])

    # Arrow down to circuit
    ax.annotate('', xy=(5, 5.5), xytext=(5, 8.0),
                arrowprops=dict(arrowstyle='->', color=C['accent1'],
                                lw=2.5, connectionstyle='arc3,rad=0'))
    ax.text(5.1, 6.7, '≡', ha='center', va='center', fontsize=22,
            color=C['accent1'], fontweight='bold')

    # Circuit side (bottom)
    ax.text(7.5, 10.0, 'NS-RAM (SPICE)', ha='center', fontsize=11,
            fontweight='bold', color=C['accent1'])
    rounded_box(ax, (5.3, 8.5), 4.4, 1.2, C['card'],
                label='Vg change → BVpar shift',
                border_color=C['accent1'], fontsize=9)
    ax.text(7.5, 8.3, 'Avalanche threshold changes', ha='center',
            fontsize=8, color=C['dim'])
    rounded_box(ax, (5.3, 6.5), 4.4, 1.2, C['card'],
                label='5 modes tested',
                border_color=C['accent1'], fontsize=9)
    ax.text(7.5, 6.3, 'Vg ∈ {0.15, 0.30, 0.35, 0.40, 0.45}V', ha='center',
            fontsize=8, color=C['dim'])

    # Metrics
    metrics_v8 = [
        ('CV = 0.495', True),
        ('ρ(Vg, output) = 0.70', True),
        ('Mode ratio = 123×', True),
    ]
    for j, (m, passed) in enumerate(metrics_v8):
        y = 4.8 - j * 1.1
        color = C['accent2'] if passed else C['accent4']
        rounded_box(ax, (1.5, y - 0.35), 7.0, 0.7, C['card'],
                    label=f'{"✓" if passed else "✗"}  {m}',
                    border_color=color, fontsize=10)

    # Score badge
    rounded_box(ax, (2.5, 0.8), 5.0, 0.8, C['glow_green'],
                label='SCORE: 3/3 PASS', border_color=C['accent2'],
                fontsize=12, alpha=0.9)

    # ---- v9: WGP Routing ↔ Neuron Ensemble ----
    ax = axes[1]
    ax.text(5, 11.5, 'v9: WGP Ensemble', ha='center', fontsize=14,
            fontweight='bold', color=C['accent1'])
    ax.text(5, 11.0, 'WGP-routed experts → Vg-gradient neurons', ha='center',
            fontsize=10, color=C['dim'])

    ax.text(2.5, 10.0, 'FEEL (GPU)', ha='center', fontsize=11,
            fontweight='bold', color=C['accent3'])
    rounded_box(ax, (0.3, 8.5), 4.4, 1.2, C['card'],
                label='8 WGP-specific LoRA experts',
                border_color=C['accent3'], fontsize=9)
    ax.text(2.5, 8.3, 'Routed by physical WGP ID', ha='center',
            fontsize=8, color=C['dim'])
    rounded_box(ax, (0.3, 6.5), 4.4, 1.2, C['card'],
                label='T26d: 0.47% PPL on kill',
                border_color=C['accent3'], fontsize=9)
    ax.text(2.5, 6.3, 'FPGA E6: ρ=0.994, 96.6% kill', ha='center',
            fontsize=8, color=C['dim'])

    ax.annotate('', xy=(5, 5.5), xytext=(5, 8.0),
                arrowprops=dict(arrowstyle='->', color=C['accent1'],
                                lw=2.5))
    ax.text(5.1, 6.7, '≡', ha='center', va='center', fontsize=22,
            color=C['accent1'], fontweight='bold')

    ax.text(7.5, 10.0, 'NS-RAM (SPICE)', ha='center', fontsize=11,
            fontweight='bold', color=C['accent1'])
    rounded_box(ax, (5.3, 8.5), 4.4, 1.2, C['card'],
                label='4-neuron Vg-gradient bank',
                border_color=C['accent1'], fontsize=9)
    ax.text(7.5, 8.3, 'Vg ∈ {0.20, 0.30, 0.40, 0.50}V', ha='center',
            fontsize=8, color=C['dim'])
    rounded_box(ax, (5.3, 6.5), 4.4, 1.2, C['card'],
                label='Weighted ensemble readout',
                border_color=C['accent1'], fontsize=9)
    ax.text(7.5, 6.3, 'Kill / scramble / homogeneous tests', ha='center',
            fontsize=8, color=C['dim'])

    metrics_v9 = [
        ('Kill-shot shift = 32.2%', True),
        ('Contribution CV = 0.546', True),
        ('Diversity ratio = ∞', True),
    ]
    for j, (m, passed) in enumerate(metrics_v9):
        y = 4.8 - j * 1.1
        color = C['accent2'] if passed else C['accent4']
        rounded_box(ax, (1.5, y - 0.35), 7.0, 0.7, C['card'],
                    label=f'{"✓" if passed else "✗"}  {m}',
                    border_color=color, fontsize=10)

    rounded_box(ax, (2.5, 0.8), 5.0, 0.8, C['glow_green'],
                label='SCORE: 3/3 PASS', border_color=C['accent2'],
                fontsize=12, alpha=0.9)

    # ---- v10: XOR Feedback ↔ Noise + Pulse ----
    ax = axes[2]
    ax.text(5, 11.5, 'v10: Stochastic Feedback', ha='center', fontsize=14,
            fontweight='bold', color=C['accent1'])
    ax.text(5, 11.0, 'XOR feedback rounding → noise + RC pulse', ha='center',
            fontsize=10, color=C['dim'])

    ax.text(2.5, 10.0, 'FEEL (GPU)', ha='center', fontsize=11,
            fontweight='bold', color=C['accent3'])
    rounded_box(ax, (0.3, 8.5), 4.4, 1.2, C['card'],
                label='XOR feedback rounding',
                border_color=C['accent3'], fontsize=9)
    ax.text(2.5, 8.3, 'acc bits ⊕ SHADER_CYCLES ⊕ WGP_ID', ha='center',
            fontsize=8, color=C['dim'])
    rounded_box(ax, (0.3, 6.5), 4.4, 1.2, C['card'],
                label='Pulse field (PULSE_EPS=0.30)',
                border_color=C['accent3'], fontsize=9)
    ax.text(2.5, 6.3, '±20% gain modulation', ha='center',
            fontsize=8, color=C['dim'])

    ax.annotate('', xy=(5, 5.5), xytext=(5, 8.0),
                arrowprops=dict(arrowstyle='->', color=C['accent1'],
                                lw=2.5))
    ax.text(5.1, 6.7, '≡', ha='center', va='center', fontsize=22,
            color=C['accent1'], fontweight='bold')

    ax.text(7.5, 10.0, 'NS-RAM (SPICE)', ha='center', fontsize=11,
            fontweight='bold', color=C['accent1'])
    rounded_box(ax, (5.3, 8.5), 4.4, 1.2, C['card'],
                label='Avalanche noise on Vt',
                border_color=C['accent1'], fontsize=9)
    ax.text(7.5, 8.3, '20% amplitude, 3 frequencies', ha='center',
            fontsize=8, color=C['dim'])
    rounded_box(ax, (5.3, 6.5), 4.4, 1.2, C['card'],
                label='RC pulse feedback on BVpar',
                border_color=C['accent1'], fontsize=9)
    ax.text(7.5, 6.3, 'τ=10μs, PULSE_EPS=0.30', ha='center',
            fontsize=8, color=C['dim'])

    metrics_v10 = [
        ('Noise amplification = 1.08%', True),
        ('BVpar feedback shift = 4.06%', True),
        ('Temp modulation = 1.15%', True),
        ('Combined causality = 1.63%', True),
    ]
    for j, (m, passed) in enumerate(metrics_v10):
        y = 4.8 - j * 0.95
        color = C['accent2'] if passed else C['accent4']
        rounded_box(ax, (1.5, y - 0.3), 7.0, 0.6, C['card'],
                    label=f'{"✓" if passed else "✗"}  {m}',
                    border_color=color, fontsize=9)

    rounded_box(ax, (2.5, 0.8), 5.0, 0.8, C['glow_green'],
                label='SCORE: 4/4 PASS', border_color=C['accent2'],
                fontsize=12, alpha=0.9)

    fig.suptitle('ISA-Level Kernel Mechanisms → NS-RAM Circuit Primitives',
                 fontsize=18, fontweight='bold', color=C['text'], y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(f'{OUTDIR}/isa_circuit_mapping.png', dpi=200,
                bbox_inches='tight', facecolor=C['bg'])
    plt.close()
    print(f'  Saved {OUTDIR}/isa_circuit_mapping.png')


# ============================================================
# DIAGRAM 3: Three-Layer Bridge Overview
# ============================================================
def diagram_three_layer_bridge():
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 13)
    ax.axis('off')

    ax.text(7, 12.5, 'THREE-LAYER BRIDGE VALIDATION',
            ha='center', fontsize=20, fontweight='bold', color=C['accent1'])
    ax.text(7, 12.0, '21 passing metrics connecting GPU registers to transistor physics',
            ha='center', fontsize=11, color=C['dim'])

    # Layer 1: High-Level SPICE (top)
    rounded_box(ax, (1, 9.5), 12, 2.0, C['card'],
                border_color=C['accent5'], lw=2)
    ax.text(7, 11.1, 'LAYER 1: High-Level SPICE (v6/v7)', ha='center',
            fontsize=14, fontweight='bold', color=C['accent5'])
    items_l1 = [
        'ThermalSoftmax: 25.2% rate Δ ↔ 21.3% weight shift',
        'Kill-shot: ∞ (100% drop) ↔ 1.91× gate clamp',
        'Regime separation: 29.3× ↔ 8.62×',
        'Monotonicity: ρ = −1.0',
        'Pulse field: ±20% gain ↔ ΔV = 0.35V',
    ]
    for j, item in enumerate(items_l1):
        ax.text(2.0, 10.6 - j * 0.3, f'✓  {item}', fontsize=8,
                color=C['accent2'], family='monospace')

    # Score badge L1
    rounded_box(ax, (10.5, 9.7), 2.2, 0.6, '#1a2e1a',
                label='5/5', border_color=C['accent2'], fontsize=14)

    # Arrow down
    ax.annotate('', xy=(7, 9.3), xytext=(7, 9.5),
                arrowprops=dict(arrowstyle='->', color=C['accent1'], lw=2))

    # Layer 2: ISA Kernel Bridge (middle)
    rounded_box(ax, (1, 5.3), 12, 3.8, C['card'],
                border_color=C['accent1'], lw=2)
    ax.text(7, 8.7, 'LAYER 2: ISA Kernel Bridge (v8/v9/v10)', ha='center',
            fontsize=14, fontweight='bold', color=C['accent1'])

    # v8 subgroup
    ax.text(2.0, 8.2, 'v8  MODE Register → Vg Switch', fontsize=10,
            fontweight='bold', color=C['accent6'])
    for j, item in enumerate([
        'CV = 0.495 (>> FEEL σ=0.0106)',
        'ρ(Vg, output) = 0.70',
        'Mode ratio = 123×',
    ]):
        ax.text(2.5, 7.8 - j * 0.3, f'✓  {item}', fontsize=8,
                color=C['accent2'], family='monospace')

    # v9 subgroup
    ax.text(2.0, 6.9, 'v9  WGP Routing → Neuron Ensemble', fontsize=10,
            fontweight='bold', color=C['accent6'])
    for j, item in enumerate([
        'Kill-shot shift = 32.2%',
        'Contribution CV = 0.546',
        'Diversity ratio = ∞ (grad vs homo)',
    ]):
        ax.text(2.5, 6.5 - j * 0.3, f'✓  {item}', fontsize=8,
                color=C['accent2'], family='monospace')

    # v10 subgroup
    ax.text(2.0, 5.6, 'v10  XOR Feedback → Noise + Pulse', fontsize=10,
            fontweight='bold', color=C['accent6'])
    for j, item in enumerate([
        'Noise amplification = 1.08%   |  BVpar shift = 4.06%',
        'Temp modulation = 1.15%       |  Combined = 1.63%',
    ]):
        ax.text(2.5, 5.2 - j * 0.3, f'✓  {item}', fontsize=8,
                color=C['accent2'], family='monospace')

    # Score badge L2
    rounded_box(ax, (10.5, 6.5), 2.2, 0.6, '#0d2137',
                label='10/10', border_color=C['accent1'], fontsize=14)

    # Arrow down
    ax.annotate('', xy=(7, 5.1), xytext=(7, 5.3),
                arrowprops=dict(arrowstyle='->', color=C['accent1'], lw=2))

    # Layer 3: FPGA Hardware (bottom)
    rounded_box(ax, (1, 2.5), 12, 2.4, C['card'],
                border_color=C['accent3'], lw=2)
    ax.text(7, 4.5, 'LAYER 3: FPGA Hardware (E1–E6)', ha='center',
            fontsize=14, fontweight='bold', color=C['accent3'])
    items_l3 = [
        'E1  ThermalSoftmax: 25.2% rate shift across regimes',
        'E2  Kill-shot: 100% drop, full recovery (infinite ratio)',
        'E3  Dual-regime: 29.3× dynamic range, smooth interpolation',
        'E4  Cross-substrate: GPT-2 PPL 2.65× shift from FPGA spikes',
        'E5  Causal transfer: spike timing → LM quality (causal, not corr.)',
        'E6  Selective ensemble: ρ=0.994, 96.6% selective kill',
    ]
    for j, item in enumerate(items_l3):
        ax.text(2.0, 4.0 - j * 0.3, f'✓  {item}', fontsize=8,
                color=C['accent2'], family='monospace')

    # Score badge L3
    rounded_box(ax, (10.5, 2.8), 2.2, 0.6, '#1a2e1a',
                label='6/6', border_color=C['accent3'], fontsize=14)

    # Total badge at bottom
    rounded_box(ax, (3.5, 0.5), 7.0, 1.2, C['glow_green'],
                border_color=C['accent2'], lw=2, alpha=0.9)
    ax.text(7, 1.2, 'TOTAL: 21/21 BRIDGE METRICS',
            ha='center', fontsize=16, fontweight='bold', color=C['text'])
    ax.text(7, 0.75, 'Shared physical quantity: Vt = kT/q = 26 mV @ 300K',
            ha='center', fontsize=10, color=C['accent3'])

    fig.tight_layout()
    fig.savefig(f'{OUTDIR}/three_layer_bridge.png', dpi=200,
                bbox_inches='tight', facecolor=C['bg'])
    plt.close()
    print(f'  Saved {OUTDIR}/three_layer_bridge.png')


# ============================================================
# DIAGRAM 4: Mechanism Bridge — circuit schematic style
# ============================================================
def diagram_mechanism_bridge():
    fig, ax = plt.subplots(figsize=(16, 9))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 11)
    ax.axis('off')

    ax.text(8, 10.5, 'FEEL ↔ NS-RAM: Constitutive Mechanism Map',
            ha='center', fontsize=18, fontweight='bold', color=C['accent1'])

    # Left column: FEEL mechanisms
    ax.text(3.5, 9.6, 'FEEL  (AMD RDNA 3.5 GPU)', ha='center',
            fontsize=13, fontweight='bold', color=C['accent3'])

    feel_items = [
        ('s_setreg hwreg(MODE,2,1)', 'FP16 rounding mode switch', 8.4),
        ('WGP-routed LoRA rank-2', 'Per-WGP correction experts', 6.6),
        ('XOR feedback rounding', 'acc ⊕ SHADER_CYCLES ⊕ WGP_ID', 4.8),
        ('Pulse field (PULSE_EPS)', '±20% gain modulation', 3.0),
        ('Thermal voltage Vt = kT/q', 'Shared physical bridge', 1.2),
    ]

    for label, sub, y in feel_items:
        color = C['accent5'] if 'Vt' in label else C['accent3']
        rounded_box(ax, (0.5, y), 6.0, 1.2, C['card'],
                    label=label, sublabel=sub,
                    border_color=color, fontsize=10)

    # Right column: NS-RAM circuit
    ax.text(12.5, 9.6, 'NS-RAM  (SPICE Circuit)', ha='center',
            fontsize=13, fontweight='bold', color=C['accent1'])

    nsram_items = [
        ('Vg change → BVpar shift', 'Avalanche threshold modulation', 8.4),
        ('4-neuron Vg-gradient ensemble', 'Weighted readout, selective kill', 6.6),
        ('Avalanche noise on Vt', '20% shot noise, 3 frequencies', 4.8),
        ('RC pulse feedback on BVpar', 'τ=10μs leaky integrator', 3.0),
        ('Vt = kT/q = 26 mV @ 300K', 'Shared physical bridge', 1.2),
    ]

    for label, sub, y in nsram_items:
        color = C['accent5'] if 'Vt' in label else C['accent1']
        rounded_box(ax, (9.5, y), 6.0, 1.2, C['card'],
                    label=label, sublabel=sub,
                    border_color=color, fontsize=10)

    # Bidirectional arrows between pairs
    for _, _, y in feel_items:
        ymid = y + 0.6
        ax.annotate('', xy=(9.3, ymid), xytext=(6.7, ymid),
                    arrowprops=dict(arrowstyle='<->', color=C['accent1'],
                                    lw=2.0))

    # Bridge labels on arrows
    bridge_labels = ['≡', '≡', '≡', '≡', '=']
    for i, (_, _, y) in enumerate(feel_items):
        ymid = y + 0.6
        ax.text(8.0, ymid, bridge_labels[i], ha='center', va='center',
                fontsize=18, fontweight='bold', color=C['accent1'])

    fig.tight_layout()
    fig.savefig(f'{OUTDIR}/mechanism_bridge.png', dpi=200,
                bbox_inches='tight', facecolor=C['bg'])
    plt.close()
    print(f'  Saved {OUTDIR}/mechanism_bridge.png')


# ============================================================
# Main
# ============================================================
if __name__ == '__main__':
    print('Generating bridge diagrams...')
    diagram_constitutivity_spectrum()
    diagram_isa_circuit_mapping()
    diagram_three_layer_bridge()
    diagram_mechanism_bridge()
    print('Done!')
