#!/usr/bin/env python3
"""
Generate dark-background EDA-style circuit schematics for NS-RAM bridge circuits.
Draws actual transistor symbols, resistors, capacitors, diodes, wires.
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, Arc, Circle
from matplotlib.lines import Line2D
import numpy as np

OUTDIR = 'scripts/img/diagrams'

# ============================================================
# Color palette — dark EDA style
# ============================================================
BG      = '#0a0e14'
WIRE    = '#4fc1e9'      # cyan wires
COMP    = '#48cfad'      # green components
LABEL   = '#ffce54'      # yellow labels
ANNOT   = '#fc6e51'      # red annotations
DIM     = '#656d78'      # dim grey
WHITE   = '#e8e8e8'
ACCENT  = '#a0d468'      # light green for pass markers
PURPLE  = '#ac92ec'
ORANGE  = '#f6bb42'
RED_PIN = '#ed5565'
BLUE_L  = '#5db8fe'

def setup_ax(ax, xlim, ylim):
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_facecolor(BG)

# ============================================================
# Circuit drawing primitives
# ============================================================

def wire(ax, points, color=WIRE, lw=1.2):
    """Draw a wire through a list of (x,y) points."""
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    ax.plot(xs, ys, color=color, lw=lw, solid_capstyle='round', zorder=2)

def dot(ax, x, y, color=WIRE, size=4):
    """Draw a junction dot."""
    ax.plot(x, y, 'o', color=color, markersize=size, zorder=3)

def gnd_symbol(ax, x, y, size=0.3, color=WIRE):
    """Draw ground symbol."""
    wire(ax, [(x, y), (x, y - size*0.4)])
    for i, w in enumerate([1.0, 0.65, 0.3]):
        yy = y - size*0.4 - i * size * 0.25
        wire(ax, [(x - size*w/2, yy), (x + size*w/2, yy)], color=color, lw=1.5)

def vdd_symbol(ax, x, y, label='VDD', size=0.3, color=WIRE):
    """Draw VDD rail symbol."""
    wire(ax, [(x, y), (x, y + size*0.5)])
    wire(ax, [(x - size*0.6, y + size*0.5), (x + size*0.6, y + size*0.5)], color=color, lw=2)
    ax.text(x, y + size*0.8, label, ha='center', va='bottom', fontsize=7,
            color=LABEL, fontweight='bold')

def resistor(ax, x1, y1, x2, y2, label='', color=COMP, lw=1.3):
    """Draw a resistor between two points (vertical or horizontal)."""
    dx, dy = x2 - x1, y2 - y1
    length = np.sqrt(dx**2 + dy**2)
    ux, uy = dx/length, dy/length  # unit vector
    nx, ny = -uy, ux  # normal

    # Lead-in
    lead = length * 0.25
    zigzag_len = length - 2*lead
    n_zigs = 5
    seg = zigzag_len / (2 * n_zigs)
    amp = 0.12

    pts = [(x1, y1)]
    # lead in
    pts.append((x1 + ux*lead, y1 + uy*lead))
    # zigzag
    cx, cy = x1 + ux*lead, y1 + uy*lead
    for i in range(n_zigs):
        sign = 1 if i % 2 == 0 else -1
        cx += ux * seg
        cy += uy * seg
        pts.append((cx + nx*amp*sign, cy + ny*amp*sign))
        cx += ux * seg
        cy += uy * seg
        pts.append((cx - nx*amp*sign*0.3, cy - ny*amp*sign*0.3))
    # Fix last point
    pts.append((x2 - ux*lead, y2 - uy*lead))
    pts.append((x2, y2))

    wire(ax, pts, color=color, lw=lw)
    if label:
        mx, my = (x1+x2)/2, (y1+y2)/2
        offset = 0.25
        ax.text(mx + nx*offset, my + ny*offset, label, ha='center', va='center',
                fontsize=6, color=LABEL, rotation=np.degrees(np.arctan2(dy, dx)))

def capacitor(ax, x1, y1, x2, y2, label='', color=COMP, lw=1.3):
    """Draw a capacitor between two points."""
    dx, dy = x2 - x1, y2 - y1
    length = np.sqrt(dx**2 + dy**2)
    ux, uy = dx/length, dy/length
    nx, ny = -uy, ux

    mx, my = (x1+x2)/2, (y1+y2)/2
    gap = 0.08
    plate_w = 0.2

    # Leads
    wire(ax, [(x1, y1), (mx - ux*gap, my - uy*gap)], color=color, lw=lw)
    wire(ax, [(mx + ux*gap, my + uy*gap), (x2, y2)], color=color, lw=lw)
    # Plates
    wire(ax, [(mx - ux*gap + nx*plate_w, my - uy*gap + ny*plate_w),
              (mx - ux*gap - nx*plate_w, my - uy*gap - ny*plate_w)], color=color, lw=2)
    wire(ax, [(mx + ux*gap + nx*plate_w, my + uy*gap + ny*plate_w),
              (mx + ux*gap - nx*plate_w, my + uy*gap - ny*plate_w)], color=color, lw=2)

    if label:
        offset = 0.3
        ax.text(mx + nx*offset, my + ny*offset, label, ha='center', va='center',
                fontsize=6, color=LABEL)

def nmos(ax, x, y, label='', gate_label='', size=0.4, color=COMP):
    """Draw NMOS transistor symbol. x,y = center. Gate on left, D top, S bottom."""
    s = size
    # Channel (vertical bar)
    wire(ax, [(x, y-s), (x, y+s)], color=color, lw=2)
    # Gate (vertical bar left of channel)
    wire(ax, [(x-s*0.3, y-s*0.7), (x-s*0.3, y+s*0.7)], color=color, lw=2)
    # Gate lead
    wire(ax, [(x-s*0.8, y), (x-s*0.3, y)], color=color, lw=1.2)
    # Drain lead (top)
    wire(ax, [(x, y+s), (x+s*0.5, y+s)], color=color, lw=1.2)
    # Source lead (bottom)
    wire(ax, [(x, y-s), (x+s*0.5, y-s)], color=color, lw=1.2)
    # Arrow on source (pointing in for NMOS)
    arr_s = s*0.2
    ax.annotate('', xy=(x+s*0.1, y-s*0.0), xytext=(x-s*0.15, y),
                arrowprops=dict(arrowstyle='->', color=color, lw=1))

    if label:
        ax.text(x+s*0.6, y, label, fontsize=7, color=LABEL, va='center')
    if gate_label:
        ax.text(x-s*0.9, y+s*0.2, gate_label, fontsize=6, color=DIM, ha='right')

    # Return pin positions: gate, drain, source
    return (x-s*0.8, y), (x+s*0.5, y+s), (x+s*0.5, y-s)

def npn(ax, x, y, label='', size=0.4, color=COMP):
    """Draw NPN BJT symbol. x,y = center. Base left, C top, E bottom."""
    s = size
    # Vertical bar (base region)
    wire(ax, [(x, y-s*0.6), (x, y+s*0.6)], color=color, lw=2.5)
    # Base lead
    wire(ax, [(x-s*0.7, y), (x, y)], color=color, lw=1.2)
    # Collector (top right)
    wire(ax, [(x, y+s*0.35), (x+s*0.6, y+s)], color=color, lw=1.2)
    # Emitter (bottom right, with arrow)
    wire(ax, [(x, y-s*0.35), (x+s*0.6, y-s)], color=color, lw=1.2)
    # Arrow on emitter
    ax.annotate('', xy=(x+s*0.55, y-s*0.9), xytext=(x+s*0.1, y-s*0.45),
                arrowprops=dict(arrowstyle='->', color=color, lw=1.2))

    if label:
        ax.text(x+s*0.7, y, label, fontsize=7, color=LABEL, va='center')

    return (x-s*0.7, y), (x+s*0.6, y+s), (x+s*0.6, y-s)

def diode(ax, x1, y1, x2, y2, label='', color=COMP, lw=1.2, zener=False):
    """Draw diode from (x1,y1) anode to (x2,y2) cathode."""
    dx, dy = x2-x1, y2-y1
    length = np.sqrt(dx**2 + dy**2)
    ux, uy = dx/length, dy/length
    nx, ny = -uy, ux

    mx, my = (x1+x2)/2, (y1+y2)/2
    tri_h = 0.15
    tri_w = 0.12

    # Leads
    wire(ax, [(x1, y1), (mx - ux*tri_h, my - uy*tri_h)], color=color, lw=lw)
    wire(ax, [(mx + ux*tri_h, my + uy*tri_h), (x2, y2)], color=color, lw=lw)
    # Triangle
    p1 = (mx - ux*tri_h + nx*tri_w, my - uy*tri_h + ny*tri_w)
    p2 = (mx - ux*tri_h - nx*tri_w, my - uy*tri_h - ny*tri_w)
    p3 = (mx + ux*tri_h, my + uy*tri_h)
    triangle = plt.Polygon([p1, p2, p3], closed=True, fill=True,
                           facecolor=color, edgecolor=color, lw=lw, zorder=3)
    ax.add_patch(triangle)
    # Bar at cathode
    wire(ax, [(mx + ux*tri_h + nx*tri_w, my + uy*tri_h + ny*tri_w),
              (mx + ux*tri_h - nx*tri_w, my + uy*tri_h - ny*tri_w)],
         color=color, lw=2)
    if zener:
        # Small hooks on bar
        wire(ax, [(mx + ux*tri_h + nx*tri_w, my + uy*tri_h + ny*tri_w),
                  (mx + ux*(tri_h+0.05) + nx*tri_w, my + uy*(tri_h+0.05) + ny*tri_w)],
             color=color, lw=1.5)

    if label:
        offset = 0.25
        ax.text(mx + nx*offset, my + ny*offset, label, fontsize=6, color=LABEL,
                ha='center', va='center')

def voltage_src(ax, x, y, label='', value='', size=0.3, color=COMP):
    """Draw voltage source circle with +/- labels."""
    circle = plt.Circle((x, y), size, fill=False, edgecolor=color, lw=1.5, zorder=3)
    ax.add_patch(circle)
    ax.text(x, y+size*0.35, '+', ha='center', va='center', fontsize=8, color=color)
    ax.text(x, y-size*0.35, '−', ha='center', va='center', fontsize=10, color=color)
    if label:
        ax.text(x-size*1.4, y, label, fontsize=7, color=LABEL, ha='right', va='center')
    if value:
        ax.text(x+size*1.4, y, value, fontsize=6, color=DIM, ha='left', va='center')
    return (x, y+size), (x, y-size)  # top, bottom pins

def current_src(ax, x, y, label='', size=0.3, color=COMP):
    """Draw current source circle with arrow."""
    circle = plt.Circle((x, y), size, fill=False, edgecolor=color, lw=1.5, zorder=3)
    ax.add_patch(circle)
    ax.annotate('', xy=(x, y+size*0.4), xytext=(x, y-size*0.4),
                arrowprops=dict(arrowstyle='->', color=color, lw=1.5))
    if label:
        ax.text(x+size*1.3, y, label, fontsize=6, color=LABEL, ha='left', va='center')

def label_node(ax, x, y, text, color=WHITE, fontsize=7, ha='left', va='center', offset=(0.1, 0)):
    """Label a circuit node."""
    ax.text(x+offset[0], y+offset[1], text, fontsize=fontsize, color=color,
            ha=ha, va=va, zorder=5)

def annotation_box(ax, x, y, text, color=ANNOT, fontsize=8, w=2.5, h=0.5):
    """Draw annotation callout box."""
    box = FancyBboxPatch((x-w/2, y-h/2), w, h, boxstyle="round,pad=0.1",
                         facecolor=BG, edgecolor=color, linewidth=1.5,
                         alpha=0.95, zorder=6)
    ax.add_patch(box)
    ax.text(x, y, text, ha='center', va='center', fontsize=fontsize,
            color=color, fontweight='bold', zorder=7)

def annotation_arrow(ax, x1, y1, x2, y2, text='', color=ANNOT):
    """Arrow with optional label for callouts."""
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color=color, lw=1.5,
                                connectionstyle='arc3,rad=0.2'))
    if text:
        mx, my = (x1+x2)/2, (y1+y2)/2
        ax.text(mx, my+0.15, text, fontsize=6, color=color, ha='center')


# ============================================================
# SCHEMATIC 1: v8 — Single NS-RAM neuron with Vg MODE control
# ============================================================
def schematic_v8_mode_register():
    fig, ax = plt.subplots(figsize=(16, 10))
    setup_ax(ax, (-1, 15), (-2, 10))

    # Title
    ax.text(7, 9.5, 'NS-RAM Bridge v8 — MODE Register as Mid-Computation Vg Switch',
            ha='center', fontsize=14, fontweight='bold', color=WHITE)
    ax.text(7, 9.0, 'FEEL: s_setreg hwreg(MODE,2,1) mid-GEMM  ↔  SPICE: Vg changes BVpar → spike rate shift',
            ha='center', fontsize=9, color=DIM)

    # === The NS-RAM neuron core ===
    # Vgate source
    vg_top, vg_bot = voltage_src(ax, 1.5, 4.5, label='Vgate', value='MODE\nregister', size=0.35)
    wire(ax, [(1.5, vg_top[1]), (1.5, 6.0)])
    label_node(ax, 1.5, 6.1, 'gate_node', color=BLUE_L, fontsize=7, ha='center', offset=(0, 0.1))
    gnd_symbol(ax, 1.5, vg_bot[1] - 0.1)

    # NMOS M2
    g, d, s = nmos(ax, 3.5, 5.5, label='M2', gate_label='Vg', size=0.45)
    wire(ax, [(1.5, 6.0), (g[0], g[1])])  # gate connection
    wire(ax, [(g[0], g[1]), (3.5-0.45*0.8, 5.5)])  # to gate pin

    # NPN Q1 (avalanche BJT)
    b, c, e = npn(ax, 5.5, 5.5, label='Q1\n(avalanche)', size=0.5)

    # Connect M2 drain to Q1 collector area
    wire(ax, [(d[0], d[1]), (d[0], 7.0), (5.5, 7.0)])
    label_node(ax, 4.5, 7.1, 'di', color=BLUE_L, fontsize=7)

    # Connect M2 source to Q1 base
    wire(ax, [(s[0], s[1]), (s[0], 4.0), (b[0]-0.3, 4.0), (b[0]-0.3, b[1]), (b[0], b[1])])
    label_node(ax, 4.0, 3.9, 'b_node', color=BLUE_L, fontsize=7, ha='center', offset=(0, -0.2))

    # Q1 collector to top rail
    wire(ax, [(c[0], c[1]), (c[0], 7.0)])

    # Q1 emitter to ground through R3
    wire(ax, [(e[0], e[1]), (e[0], 3.5)])
    resistor(ax, e[0], 3.5, e[0], 2.5, label='R3\n1Ω')
    label_node(ax, e[0]+0.2, 3.6, 's_node', color=BLUE_L, fontsize=7)
    wire(ax, [(e[0], 2.5), (e[0], 2.0)])
    gnd_symbol(ax, e[0], 2.0)

    # Zener diodes D1, D2
    diode(ax, 4.0, 6.5, 3.0, 6.5, label='D1 (zener)', zener=True)
    wire(ax, [(4.0, 6.5), (4.0, 7.0)])  # to di node
    wire(ax, [(3.0, 6.5), (1.5, 6.5), (1.5, 6.0)])  # to gate

    diode(ax, 4.0, 4.5, 5.0, 4.5, label='D2 (zener)', zener=True)
    wire(ax, [(4.0, 4.5), (4.0, 4.0)])  # to b via short route
    wire(ax, [(5.0, 4.5), (e[0], 4.5), (e[0], 3.5)])  # to s_node

    # R4 from di to d_node
    wire(ax, [(5.5, 7.0), (7.0, 7.0)])
    resistor(ax, 7.0, 7.0, 8.0, 7.0, label='R4 1Ω')
    label_node(ax, 8.1, 7.1, 'd_node', color=BLUE_L, fontsize=7)

    # Vpulse source
    vp_top, vp_bot = voltage_src(ax, 8.5, 5.5, label='Vpulse', value='PULSE\n0→3.4V', size=0.35)
    wire(ax, [(8.5, vp_top[1]), (8.5, 7.0)])
    gnd_symbol(ax, 8.5, vp_bot[1] - 0.1)

    # Cbulk on b_node
    wire(ax, [(4.0, 4.0), (3.0, 4.0), (3.0, 3.0)])
    capacitor(ax, 3.0, 3.0, 3.0, 2.2, label='Cbulk\n500fF')
    gnd_symbol(ax, 3.0, 2.0)

    # === Avalanche current model ===
    # Baval (behavioral current source)
    ax.text(7.0, 8.2, 'Baval: I = 10p·min(exp((Vcb−BVpar)/Vt), 2000)',
            fontsize=7, color=PURPLE, family='monospace', ha='center')

    # BVpar computation
    annotation_box(ax, 7.0, 8.7, 'BVpar = 3.5 − 1.5·Vg + Tbv·(T−300)',
                   color=ORANGE, fontsize=7, w=4.0, h=0.4)

    # === LIF Membrane (right side) ===
    wire(ax, [(e[0], 3.6), (9.5, 3.6)])
    label_node(ax, 9.5, 3.8, 'spike detect', color=DIM, fontsize=6, ha='center', offset=(0.5,0))

    # Cint (membrane capacitor)
    wire(ax, [(10.5, 3.6), (10.5, 3.0)])
    capacitor(ax, 10.5, 3.0, 10.5, 2.2, label='Cint\n102fF')
    label_node(ax, 10.7, 3.5, 'vmem', color=BLUE_L, fontsize=7)
    gnd_symbol(ax, 10.5, 2.0)

    # Inverter chain
    # Simple box for inverter
    for ix, lbl in [(11.5, 'INV1'), (12.5, 'INV2')]:
        box = FancyBboxPatch((ix-0.3, 3.1), 0.6, 0.9, boxstyle="round,pad=0.05",
                             facecolor=BG, edgecolor=COMP, lw=1.2, zorder=3)
        ax.add_patch(box)
        ax.text(ix, 3.55, lbl, fontsize=5, color=COMP, ha='center', va='center')

    wire(ax, [(10.5, 3.6), (11.2, 3.6)])
    wire(ax, [(11.8, 3.6), (12.2, 3.6)])
    wire(ax, [(12.8, 3.6), (13.2, 3.6)])
    label_node(ax, 13.2, 3.8, 'vspk', color=BLUE_L, fontsize=7)

    # Reset MOSFET
    g_r, d_r, s_r = nmos(ax, 11.0, 2.0, label='M_rst', size=0.3)
    wire(ax, [(d_r[0], d_r[1]), (10.5, d_r[1]), (10.5, 2.5)])  # drain to vmem
    gnd_symbol(ax, s_r[0], s_r[1] - 0.2)

    # Analog output (low-pass filtered avalanche)
    wire(ax, [(8.5, 7.0), (10.0, 7.0)])
    resistor(ax, 10.0, 7.0, 11.5, 7.0, label='Rlpf 10k')
    wire(ax, [(11.5, 7.0), (13.0, 7.0)])
    capacitor(ax, 13.0, 7.0, 13.0, 6.2, label='Clpf 10pF')
    gnd_symbol(ax, 13.0, 6.0)
    label_node(ax, 11.8, 7.2, 'aval_filt', color=BLUE_L, fontsize=7)

    # Output label
    annotation_box(ax, 13.3, 7.6, 'ANALOG\nOUTPUT', color=ACCENT, fontsize=8, w=1.6, h=0.7)

    # === Annotation callouts ===
    # MODE register annotation
    annotation_box(ax, 1.5, 8.0, 'FEEL: s_setreg hwreg(MODE,2,1)\n'
                   'Changes FP16 rounding mid-GEMM\n'
                   '→ 12/12 unique math fingerprints',
                   color=ANNOT, fontsize=7, w=3.8, h=0.9)
    annotation_arrow(ax, 1.5, 7.5, 1.5, 6.2, color=ANNOT)

    # Vg→BVpar annotation
    annotation_box(ax, 11.0, 1.0, 'Vg={0.15, 0.30, 0.35, 0.40, 0.45}V\n'
                   'CV=0.495  ρ=0.70  Ratio=123×\n'
                   'SCORE: 3/3 PASS ✓',
                   color=ACCENT, fontsize=7, w=3.6, h=0.8)

    # Thermal voltage
    ax.text(7.0, 0.5, 'Shared bridge quantity:  Vt = kT/q = 25.85 mV @ 300K',
            ha='center', fontsize=9, color=PURPLE, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor=BG, edgecolor=PURPLE, lw=1.5))

    # VDD rail
    wire(ax, [(5.5, 7.0), (5.5, 7.8)])
    vdd_symbol(ax, 5.5, 7.8, label='VD_supply\n3.5V')

    fig.tight_layout()
    fig.savefig(f'{OUTDIR}/v8_mode_register_schematic.png', dpi=200,
                bbox_inches='tight', facecolor=BG)
    plt.close()
    print(f'  Saved v8_mode_register_schematic.png')


# ============================================================
# SCHEMATIC 2: v9 — 4-neuron WGP ensemble
# ============================================================
def schematic_v9_wgp_ensemble():
    fig, ax = plt.subplots(figsize=(18, 11))
    setup_ax(ax, (-1, 17), (-2.5, 10.5))

    ax.text(8, 10.0, 'NS-RAM Bridge v9 — WGP-Routed Ensemble (4-Neuron Vg-Gradient Bank)',
            ha='center', fontsize=14, fontweight='bold', color=WHITE)
    ax.text(8, 9.5, 'FEEL: 8 WGP-specific rank-2 LoRA experts routed by physical WGP ID  ↔  SPICE: per-neuron Vg gradient',
            ha='center', fontsize=9, color=DIM)

    # Draw 4 neuron columns
    neuron_x = [1.0, 5.0, 9.0, 13.0]
    vg_vals = [0.20, 0.30, 0.40, 0.50]
    wgp_ids = ['WGP 0\n(cold)', 'WGP 2\n(mod)', 'WGP 4\n(warm)', 'WGP 6\n(hot)']
    colors_n = ['#5db8fe', '#48cfad', '#f6bb42', '#ed5565']

    for ni, (nx, vg, wgp, ncol) in enumerate(zip(neuron_x, vg_vals, wgp_ids, colors_n)):
        # Neuron label
        ax.text(nx+1.0, 8.8, f'NEURON {ni}', ha='center', fontsize=10,
                fontweight='bold', color=ncol)
        ax.text(nx+1.0, 8.4, wgp, ha='center', fontsize=7, color=DIM)

        # Vgate
        vt, vb = voltage_src(ax, nx+0.3, 6.0, label=f'Vg{ni}', value=f'{vg}V', size=0.3, color=ncol)
        gnd_symbol(ax, nx+0.3, vb[1]-0.1, color=ncol)

        # NMOS
        g, d, s = nmos(ax, nx+1.2, 6.5, label=f'M2_{ni}', size=0.35, color=ncol)
        wire(ax, [(nx+0.3, vt[1]), (nx+0.3, 7.0), (g[0], 7.0), (g[0], g[1])], color=ncol)

        # NPN
        b, c, e = npn(ax, nx+1.8, 5.5, label=f'Q1_{ni}', size=0.35, color=ncol)
        wire(ax, [(d[0], d[1]), (d[0], 7.5), (c[0], 7.5), (c[0], c[1])], color=ncol)
        wire(ax, [(s[0], s[1]), (s[0], 5.0), (b[0]-0.2, 5.0), (b[0]-0.2, b[1])], color=ncol)

        # Vpulse
        wire(ax, [(d[0], 7.5), (nx+2.5, 7.5)], color=ncol)
        vpt, vpb = voltage_src(ax, nx+2.5, 7.5, value='3.4V\npulse', size=0.2, color=ncol)

        # R3 to ground
        wire(ax, [(e[0], e[1]), (e[0], 4.2)], color=ncol)
        resistor(ax, e[0], 4.2, e[0], 3.5, label='R3', color=ncol)
        gnd_symbol(ax, e[0], 3.3, color=ncol)

        # Zener diodes (simplified — just labels)
        ax.text(nx+0.5, 7.3, 'D1,D2\n(zener)', fontsize=5, color=DIM, ha='center')

        # LPF output
        wire(ax, [(nx+2.5, 7.5), (nx+2.5, 8.0)], color=ncol)
        label_node(ax, nx+2.5, 8.1, f'filt{ni}', color=ncol, fontsize=6, ha='center', offset=(0, 0.1))

        # Avalanche equation
        ax.text(nx+1.0, 3.0, f'BVpar = 3.5−1.5·{vg}\n= {3.5-1.5*vg:.2f}V',
                fontsize=6, color=ncol, ha='center',
                bbox=dict(boxstyle='round,pad=0.15', facecolor=BG, edgecolor=ncol, lw=0.8))

    # Weighted readout summing node
    wire(ax, [(3.5, 8.2), (3.5, 2.0), (8.0, 1.5)], color=WIRE)
    wire(ax, [(7.5, 8.2), (7.5, 2.2), (8.0, 1.5)], color=WIRE)
    wire(ax, [(11.5, 8.2), (11.5, 2.2), (8.0, 1.5)], color=WIRE)
    wire(ax, [(15.5, 8.2), (15.5, 2.0), (8.0, 1.5)], color=WIRE)

    # Summing amplifier symbol
    # Triangle
    tri = plt.Polygon([(8.0, 1.5), (8.0, 2.5), (9.5, 2.0)], closed=True,
                       fill=False, edgecolor=WIRE, lw=2, zorder=3)
    ax.add_patch(tri)
    ax.text(8.5, 2.0, 'Σ', fontsize=14, color=WHITE, ha='center', va='center',
            fontweight='bold', zorder=4)
    ax.text(8.7, 1.5, 'W₀=W₁=W₂=W₃=0.25', fontsize=6, color=DIM, ha='center')

    # Output
    wire(ax, [(9.5, 2.0), (11.0, 2.0)], color=WIRE, lw=2)
    annotation_box(ax, 12.5, 2.0, 'ENSEMBLE\nOUTPUT', color=ACCENT, fontsize=9, w=2.0, h=0.7)

    # Weight labels on wires
    for i, nx in enumerate(neuron_x):
        filt_x = nx + 2.5
        ax.text(filt_x + 0.2, 2.5, f'W{i}', fontsize=6, color=ORANGE)

    # Results annotation
    annotation_box(ax, 8.0, -0.5,
                   'Kill N3 (hot): 32.2% shift  |  Kill N0 (cold): 1.2% shift  |  '
                   'Scramble: 0% shift  |  Homogeneous: 0 diversity\n'
                   'Contribution CV = 0.546  |  Diversity ratio = ∞  |  SCORE: 3/3 PASS ✓',
                   color=ACCENT, fontsize=8, w=14.0, h=0.8)

    # FEEL mapping
    annotation_box(ax, 8.0, -1.7,
                   'FEEL mapping: Each neuron = one WGP LoRA expert  |  Vg gradient = WGP-specific correction magnitude  |  '
                   'Ensemble readout = combined WGP output',
                   color=ORANGE, fontsize=7, w=14.0, h=0.5)

    fig.tight_layout()
    fig.savefig(f'{OUTDIR}/v9_wgp_ensemble_schematic.png', dpi=200,
                bbox_inches='tight', facecolor=BG)
    plt.close()
    print(f'  Saved v9_wgp_ensemble_schematic.png')


# ============================================================
# SCHEMATIC 3: v10 — Stochastic feedback (noise + pulse)
# ============================================================
def schematic_v10_stochastic_feedback():
    fig, ax = plt.subplots(figsize=(16, 11))
    setup_ax(ax, (-1, 15), (-2.5, 11))

    ax.text(7, 10.5, 'NS-RAM Bridge v10 — Stochastic Feedback Rounding',
            ha='center', fontsize=14, fontweight='bold', color=WHITE)
    ax.text(7, 10.0, 'FEEL: XOR feedback rounding + pulse field  ↔  SPICE: avalanche noise + RC pulse feedback on BVpar',
            ha='center', fontsize=9, color=DIM)

    # === Core neuron (center) ===
    # NMOS
    g, d, s = nmos(ax, 5.0, 6.0, label='M2m', size=0.45)
    # NPN
    b, c, e = npn(ax, 6.5, 5.5, label='Q1m\n(avalanche)', size=0.5)

    # Wiring
    wire(ax, [(d[0], d[1]), (d[0], 7.5), (6.5, 7.5)])
    wire(ax, [(c[0], c[1]), (c[0], 7.5)])
    wire(ax, [(s[0], s[1]), (s[0], 4.5), (b[0]-0.3, 4.5), (b[0]-0.3, b[1])])
    label_node(ax, 5.8, 7.6, 'di', color=BLUE_L)
    label_node(ax, 5.3, 4.4, 'b_node', color=BLUE_L, fontsize=6)

    # Vgate
    vt, vb = voltage_src(ax, 2.5, 5.5, label='Vgate', value='0.35V', size=0.35)
    wire(ax, [(2.5, vt[1]), (2.5, 6.5), (g[0]-0.5, 6.5), (g[0]-0.5, g[1]), (g[0], g[1])])
    gnd_symbol(ax, 2.5, vb[1]-0.1)

    # R3 + gnd
    wire(ax, [(e[0], e[1]), (e[0], 3.5)])
    resistor(ax, e[0], 3.5, e[0], 2.5, label='R3 1Ω')
    gnd_symbol(ax, e[0], 2.3)

    # Vpulse
    wire(ax, [(6.5, 7.5), (8.0, 7.5)])
    resistor(ax, 8.0, 7.5, 9.0, 7.5, label='R4 1Ω')
    vpt, vpb = voltage_src(ax, 9.5, 6.5, value='PULSE\n0→3.4V', size=0.3)
    wire(ax, [(9.0, 7.5), (9.5, 7.5), (9.5, vpt[1])])
    gnd_symbol(ax, 9.5, vpb[1]-0.1)
    label_node(ax, 9.0, 7.7, 'd_node', color=BLUE_L, fontsize=6)

    # Cbulk
    wire(ax, [(b[0]-0.3, 4.5), (4.0, 4.5), (4.0, 3.5)])
    capacitor(ax, 4.0, 3.5, 4.0, 2.7, label='Cbulk\n500fF')
    gnd_symbol(ax, 4.0, 2.5)

    # Zener diodes
    ax.text(4.5, 7.0, 'D1,D2 (zener)', fontsize=6, color=DIM, ha='center')

    # === NOISE SOURCE (left side) ===
    noise_x, noise_y = 0.5, 3.0
    box_n = FancyBboxPatch((noise_x-0.8, noise_y-0.6), 2.8, 1.8, boxstyle="round,pad=0.1",
                            facecolor=BG, edgecolor=ANNOT, lw=1.5, zorder=3, alpha=0.9)
    ax.add_patch(box_n)
    ax.text(noise_x+0.6, noise_y+0.8, 'NOISE SOURCE', fontsize=8, fontweight='bold',
            color=ANNOT, ha='center')
    ax.text(noise_x+0.6, noise_y+0.3, '20% of Vt amplitude', fontsize=6, color=DIM, ha='center')
    ax.text(noise_x+0.6, noise_y-0.0, '3 incommensurate freq.', fontsize=6, color=DIM, ha='center')
    ax.text(noise_x+0.6, noise_y-0.3, '→ pseudo-random beating', fontsize=6, color=DIM, ha='center')

    # noise_en switch
    ax.text(noise_x+0.6, noise_y-0.7, 'noise_en: 0/1', fontsize=6, color=ANNOT, ha='center')

    # Wire noise to Vt_eff
    wire(ax, [(noise_x+2.0, noise_y+0.5), (3.5, noise_y+0.5), (3.5, 1.5)], color=ANNOT)

    # === Vt COMPUTATION ===
    vt_box = FancyBboxPatch((2.5, 0.5), 5.0, 1.2, boxstyle="round,pad=0.1",
                             facecolor=BG, edgecolor=PURPLE, lw=1.5, zorder=3)
    ax.add_patch(vt_box)
    ax.text(5.0, 1.3, 'Vt_eff = Vt + noise_en · noise_sig', fontsize=7,
            color=PURPLE, ha='center', family='monospace')
    ax.text(5.0, 0.8, 'Vt = kT/q = 25.85 mV @ 300K', fontsize=7,
            color=PURPLE, ha='center')

    # === BVpar COMPUTATION ===
    bv_box = FancyBboxPatch((2.5, -0.8), 5.0, 1.0, boxstyle="round,pad=0.1",
                             facecolor=BG, edgecolor=ORANGE, lw=1.5, zorder=3)
    ax.add_patch(bv_box)
    ax.text(5.0, -0.1, 'BVpar_eff = BVpar_base − fb_en · 0.30 · fb_norm',
            fontsize=7, color=ORANGE, ha='center', family='monospace')
    ax.text(5.0, -0.55, 'BVpar_base = 3.5 − 1.5·Vg + Tbv·ΔT', fontsize=6,
            color=DIM, ha='center')

    # === AVALANCHE MODEL (center) ===
    aval_box = FancyBboxPatch((8.5, 0.2), 5.5, 1.5, boxstyle="round,pad=0.1",
                               facecolor=BG, edgecolor=WIRE, lw=2, zorder=3)
    ax.add_patch(aval_box)
    ax.text(11.25, 1.3, 'AVALANCHE MODEL', fontsize=9, fontweight='bold',
            color=WIRE, ha='center')
    ax.text(11.25, 0.8, 'I = 10p · min(exp((Vcb−BVpar_eff)/Vt_eff), 2000)',
            fontsize=7, color=WIRE, ha='center', family='monospace')
    ax.text(11.25, 0.4, '→ analog output × 1e9 → LPF', fontsize=6,
            color=DIM, ha='center')

    # Arrows from Vt and BVpar into avalanche
    wire(ax, [(7.5, 1.0), (8.5, 1.0)], color=PURPLE, lw=1.5)
    wire(ax, [(7.5, -0.3), (8.2, -0.3), (8.2, 0.6), (8.5, 0.6)], color=ORANGE, lw=1.5)

    # === LPF OUTPUT ===
    wire(ax, [(14.0, 1.0), (14.5, 1.0)])
    wire(ax, [(14.5, 1.0), (14.5, 3.0)])
    resistor(ax, 14.5, 3.0, 14.5, 4.5, label='Rlpf\n10k')
    wire(ax, [(14.5, 4.5), (14.5, 5.5)])
    capacitor(ax, 14.5, 5.5, 14.5, 6.5, label='Clpf\n10pF')
    gnd_symbol(ax, 14.5, 6.7)
    label_node(ax, 14.5, 4.8, 'aval_raw', color=BLUE_L, fontsize=6, ha='center', offset=(0.5, 0))
    label_node(ax, 14.5, 5.8, 'filt_out', color=BLUE_L, fontsize=6, ha='center', offset=(0.5, 0))

    # Output label
    annotation_box(ax, 14.5, 3.5, 'ANALOG\nOUTPUT', color=ACCENT, fontsize=8, w=1.6, h=0.7)

    # === RC FEEDBACK LOOP (right side, going back down) ===
    # Feedback path from aval_raw back to BVpar
    fb_x = 12.0
    wire(ax, [(14.5, 4.5), (14.5, 8.0), (fb_x+1.5, 8.0)], color=ANNOT, lw=1.5)

    fb_box = FancyBboxPatch((fb_x-1.2, 7.2), 4.0, 2.0, boxstyle="round,pad=0.1",
                             facecolor=BG, edgecolor=ANNOT, lw=1.5, zorder=3, alpha=0.9)
    ax.add_patch(fb_box)
    ax.text(fb_x+0.8, 8.8, 'RC PULSE FEEDBACK', fontsize=8, fontweight='bold',
            color=ANNOT, ha='center')

    # RC components inside
    ax.text(fb_x+0.8, 8.3, 'Rfb = 100kΩ', fontsize=7, color=COMP, ha='center')
    ax.text(fb_x+0.8, 7.9, 'Cfb = 100pF', fontsize=7, color=COMP, ha='center')
    ax.text(fb_x+0.8, 7.5, 'τ = 10μs', fontsize=7, color=ORANGE, ha='center', fontweight='bold')

    # fb_norm
    wire(ax, [(fb_x-1.2, 8.0), (fb_x-2.0, 8.0), (fb_x-2.0, -0.3), (7.5, -0.3)], color=ANNOT, lw=1.5)
    label_node(ax, fb_x-2.2, 5.0, 'fb_norm\n= min(fb/5, 1)',
               color=ANNOT, fontsize=6, ha='center', offset=(0, 0))

    # fb_en switch label
    ax.text(fb_x-2.0, 6.5, 'fb_en: 0/1', fontsize=6, color=ANNOT, ha='center')

    # === FEEL MAPPING (top annotation) ===
    feel_box = FancyBboxPatch((-0.5, 9.0), 8.0, 0.8, boxstyle="round,pad=0.1",
                               facecolor=BG, edgecolor=ORANGE, lw=1.5, zorder=3)
    ax.add_patch(feel_box)
    ax.text(3.5, 9.5, 'FEEL: acc_bits ⊕ SHADER_CYCLES ⊕ WGP_ID → rounding mode',
            fontsize=7, color=ORANGE, ha='center')
    ax.text(3.5, 9.15, 'Pulse field: PULSE_EPS=0.30 → ±20% gain modulation',
            fontsize=7, color=ORANGE, ha='center')

    # Results
    annotation_box(ax, 7.0, -1.8,
                   'Noise amplification: 1.08%  |  BVpar shift: 4.06%  |  '
                   'Temp modulation: 1.15%  |  Combined: 1.63%  |  '
                   'SCORE: 4/4 PASS ✓',
                   color=ACCENT, fontsize=8, w=13.0, h=0.5)

    # Key insight
    ax.text(7.0, -2.3, 'KEY INSIGHT: Noise alone is absorbed by exp() saturation. '
            'Noise THROUGH the feedback loop creates 1.08% amplification — '
            'mirroring FEEL\'s XOR mechanism.',
            fontsize=7, color=PURPLE, ha='center', style='italic')

    fig.tight_layout()
    fig.savefig(f'{OUTDIR}/v10_stochastic_feedback_schematic.png', dpi=200,
                bbox_inches='tight', facecolor=BG)
    plt.close()
    print(f'  Saved v10_stochastic_feedback_schematic.png')


# ============================================================
# SCHEMATIC 4: Overview — all three circuits + bridge
# ============================================================
def schematic_overview():
    fig = plt.figure(figsize=(18, 12))
    fig.patch.set_facecolor(BG)

    # Title
    fig.text(0.5, 0.97, 'NS-RAM Bridge Circuits — ISA Kernel Mechanisms as Circuit Primitives',
             ha='center', fontsize=16, fontweight='bold', color=WHITE)
    fig.text(0.5, 0.945, 'Three SPICE circuits mapping FEEL\'s lowest-level GPU mechanisms to neuromorphic avalanche physics  |  10/10 metrics pass',
             ha='center', fontsize=9, color=DIM)

    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.25,
                          left=0.03, right=0.97, top=0.92, bottom=0.08)

    # --- v8 simplified ---
    ax1 = fig.add_subplot(gs[0, 0])
    setup_ax(ax1, (0, 10), (0, 10))
    ax1.text(5, 9.5, 'v8: MODE Register → Vg Switch', ha='center',
             fontsize=11, fontweight='bold', color='#58a6ff')

    # Simplified single neuron
    g, d, s = nmos(ax1, 3.0, 5.5, label='M2', size=0.5)
    b, c, e = npn(ax1, 5.0, 5.0, label='Q1', size=0.5)
    wire(ax1, [(d[0], d[1]), (d[0], 7.0), (c[0], 7.0), (c[0], c[1])])
    wire(ax1, [(s[0], s[1]), (s[0], 4.0), (b[0]-0.3, 4.0), (b[0]-0.3, b[1])])
    wire(ax1, [(e[0], e[1]), (e[0], 3.0)])
    gnd_symbol(ax1, e[0], 2.8)

    # Vgate with multiple values
    vt, vb = voltage_src(ax1, 1.0, 5.0, label='Vg', size=0.35)
    wire(ax1, [(1.0, vt[1]), (1.0, 6.0), (g[0]-0.5, 6.0), (g[0]-0.5, g[1])])
    gnd_symbol(ax1, 1.0, vb[1]-0.1)

    # Mode labels
    for i, (vg, yy) in enumerate([(0.15, 3.8), (0.30, 3.4), (0.35, 3.0), (0.40, 2.6), (0.45, 2.2)]):
        col = ['#ed5565', '#f6bb42', '#48cfad', '#5db8fe', '#ac92ec'][i]
        ax1.text(1.5, yy, f'Mode {chr(65+i)}: Vg={vg}V', fontsize=6, color=col)

    annotation_box(ax1, 5.0, 1.2, 'BVpar = 3.5 − 1.5·Vg', color=ORANGE, fontsize=7, w=3.0, h=0.4)

    # Output
    wire(ax1, [(d[0], 7.0), (7.5, 7.0)])
    resistor(ax1, 7.5, 7.0, 8.5, 7.0, label='Rlpf')
    label_node(ax1, 8.6, 7.2, 'out', color=ACCENT, fontsize=7)

    # Score
    ax1.text(5, 0.3, '✓ CV=0.495  ✓ ρ=0.70  ✓ 123× ratio',
             ha='center', fontsize=7, color=ACCENT, fontweight='bold',
             bbox=dict(boxstyle='round,pad=0.2', facecolor=BG, edgecolor=ACCENT, lw=1))

    # --- v9 simplified ---
    ax2 = fig.add_subplot(gs[0, 1:])
    setup_ax(ax2, (0, 16), (0, 10))
    ax2.text(8, 9.5, 'v9: WGP Routing → 4-Neuron Vg-Gradient Ensemble', ha='center',
             fontsize=11, fontweight='bold', color='#58a6ff')

    # 4 neurons side by side
    for ni, (nx, vg, ncol) in enumerate(zip([1, 4.5, 8, 11.5],
                                             [0.20, 0.30, 0.40, 0.50],
                                             ['#5db8fe', '#48cfad', '#f6bb42', '#ed5565'])):
        # Simplified neuron: just NMOS + NPN + label
        g, d, s = nmos(ax2, nx+0.5, 6.0, size=0.35, color=ncol)
        b, c, e = npn(ax2, nx+1.5, 5.5, size=0.35, color=ncol)
        wire(ax2, [(d[0], d[1]), (d[0], 7.2), (c[0], 7.2), (c[0], c[1])], color=ncol)
        wire(ax2, [(s[0], s[1]), (s[0], 4.5), (b[0]-0.2, 4.5), (b[0]-0.2, b[1])], color=ncol)
        wire(ax2, [(e[0], e[1]), (e[0], 3.8)], color=ncol)
        gnd_symbol(ax2, e[0], 3.6, color=ncol)

        ax2.text(nx+1.0, 8.5, f'N{ni}: Vg={vg}V', ha='center', fontsize=7,
                 color=ncol, fontweight='bold')
        ax2.text(nx+1.0, 8.0, f'BVpar={3.5-1.5*vg:.2f}V', ha='center', fontsize=6, color=DIM)

        # Output wire to summing
        wire(ax2, [(d[0], 7.2), (d[0], 7.8)], color=ncol)
        label_node(ax2, d[0]-0.3, 7.9, f'filt{ni}', color=ncol, fontsize=5)

        # Wire down to sum
        wire(ax2, [(nx+1.0, 7.8), (nx+1.0, 2.5), (7.5, 1.8)], color=ncol, lw=0.8)
        ax2.text(nx+1.2, 2.8, f'W{ni}=0.25', fontsize=5, color=DIM)

    # Summing node
    tri = plt.Polygon([(7.5, 1.5), (7.5, 2.3), (8.8, 1.9)], closed=True,
                       fill=False, edgecolor=WIRE, lw=1.5, zorder=3)
    ax2.add_patch(tri)
    ax2.text(7.9, 1.9, 'Σ', fontsize=12, color=WHITE, ha='center', va='center', fontweight='bold')
    wire(ax2, [(8.8, 1.9), (10.0, 1.9)], color=WIRE, lw=2)
    annotation_box(ax2, 11.5, 1.9, 'ENSEMBLE\nOUT', color=ACCENT, fontsize=7, w=1.8, h=0.6)

    # Kill test illustration
    # X over neuron 3
    ax2.plot([12.3, 13.3], [4.5, 7.0], color=ANNOT, lw=2.5, zorder=5)
    ax2.plot([12.3, 13.3], [7.0, 4.5], color=ANNOT, lw=2.5, zorder=5)
    ax2.text(12.8, 4.0, 'KILL\n−32.2%', fontsize=7, color=ANNOT, ha='center', fontweight='bold')

    ax2.text(8, 0.5, '✓ Kill=32.2%  ✓ CV=0.546  ✓ Diversity=∞ (grad vs homo)',
             ha='center', fontsize=7, color=ACCENT, fontweight='bold',
             bbox=dict(boxstyle='round,pad=0.2', facecolor=BG, edgecolor=ACCENT, lw=1))

    # --- v10 simplified ---
    ax3 = fig.add_subplot(gs[1, :])
    setup_ax(ax3, (0, 20), (0, 8))
    ax3.text(10, 7.5, 'v10: XOR Feedback Rounding → Avalanche Noise + RC Pulse Feedback on BVpar', ha='center',
             fontsize=11, fontweight='bold', color='#58a6ff')

    # Core neuron (center)
    g, d, s = nmos(ax3, 7.0, 4.5, label='M2m', size=0.45)
    b, c, e = npn(ax3, 8.5, 4.0, label='Q1m', size=0.45)
    wire(ax3, [(d[0], d[1]), (d[0], 6.0), (c[0], 6.0), (c[0], c[1])])
    wire(ax3, [(s[0], s[1]), (s[0], 3.0), (b[0]-0.3, 3.0), (b[0]-0.3, b[1])])
    wire(ax3, [(e[0], e[1]), (e[0], 2.0)])
    gnd_symbol(ax3, e[0], 1.8)

    # Vgate
    vt_g, vb_g = voltage_src(ax3, 5.0, 4.0, label='Vg', value='0.35V', size=0.3)
    wire(ax3, [(5.0, vt_g[1]), (5.0, 5.0), (g[0]-0.5, 5.0), (g[0]-0.5, g[1])])
    gnd_symbol(ax3, 5.0, vb_g[1]-0.1)

    # Vpulse
    wire(ax3, [(d[0], 6.0), (10.0, 6.0)])
    vpt3, vpb3 = voltage_src(ax3, 10.5, 5.0, value='PULSE', size=0.25)
    wire(ax3, [(10.0, 6.0), (10.5, 6.0), (10.5, vpt3[1])])
    gnd_symbol(ax3, 10.5, vpb3[1]-0.1)

    # === NOISE SOURCE (left) ===
    noise_box = FancyBboxPatch((0.5, 2.0), 3.0, 2.5, boxstyle="round,pad=0.1",
                                facecolor=BG, edgecolor=ANNOT, lw=2, zorder=3)
    ax3.add_patch(noise_box)
    ax3.text(2.0, 4.1, 'NOISE SOURCE', fontsize=9, fontweight='bold', color=ANNOT, ha='center')
    # Draw a noisy waveform
    t = np.linspace(0, 2*np.pi*3, 100)
    noise_wave = 0.3 * (np.sin(t*1.73) * np.sin(t*7.91) + 0.6*np.sin(t*3.37))
    ax3.plot(np.linspace(1.0, 3.0, 100), noise_wave + 3.3, color=ANNOT, lw=1, zorder=4)
    ax3.text(2.0, 2.6, '20% Vt · sin(ω₁t)·sin(ω₂t)\n+ harmonics', fontsize=6,
             color=DIM, ha='center')
    ax3.text(2.0, 2.2, 'noise_en: ON/OFF', fontsize=6, color=ANNOT, ha='center')

    # Wire from noise to Vt_eff
    wire(ax3, [(3.5, 3.3), (4.5, 3.3), (4.5, 1.5)], color=ANNOT, lw=1.5)

    # Vt_eff box
    vt_eff_box = FancyBboxPatch((4.0, 0.5), 4.0, 0.8, boxstyle="round,pad=0.1",
                                 facecolor=BG, edgecolor=PURPLE, lw=1.5, zorder=3)
    ax3.add_patch(vt_eff_box)
    ax3.text(6.0, 1.0, 'Vt_eff = Vt + noise_en·noise', fontsize=7,
             color=PURPLE, ha='center', family='monospace')
    ax3.text(6.0, 0.65, 'Vt = kT/q = 25.85 mV', fontsize=6, color=DIM, ha='center')

    # === AVALANCHE (center-right) ===
    aval_box = FancyBboxPatch((9.0, 0.5), 5.0, 1.2, boxstyle="round,pad=0.1",
                               facecolor=BG, edgecolor=WIRE, lw=2, zorder=3)
    ax3.add_patch(aval_box)
    ax3.text(11.5, 1.3, 'I = 10p·min(exp((Vcb−BVpar_eff)/Vt_eff), 2000)',
             fontsize=7, color=WIRE, ha='center', family='monospace')
    ax3.text(11.5, 0.7, 'Avalanche current → analog output', fontsize=6,
             color=DIM, ha='center')

    wire(ax3, [(8.0, 1.0), (9.0, 1.0)], color=PURPLE, lw=1.5)

    # BVpar_eff arrow in
    wire(ax3, [(9.0, 0.5), (8.5, 0.0), (8.5, -0.3)], color=ORANGE, lw=1.5)
    ax3.text(8.5, -0.6, 'BVpar_eff = BVpar − fb_en·0.30·fb_norm',
             fontsize=6, color=ORANGE, ha='center', family='monospace')

    # LPF output
    wire(ax3, [(14.0, 1.0), (14.5, 1.0), (14.5, 3.0)])
    resistor(ax3, 14.5, 3.0, 14.5, 4.0, label='Rlpf')
    wire(ax3, [(14.5, 4.0), (14.5, 5.0)])
    capacitor(ax3, 14.5, 5.0, 14.5, 5.8, label='Clpf')
    gnd_symbol(ax3, 14.5, 6.0)
    label_node(ax3, 14.7, 3.5, 'aval_raw', color=BLUE_L, fontsize=6)
    label_node(ax3, 14.7, 5.2, 'filt_out', color=BLUE_L, fontsize=6)

    annotation_box(ax3, 15.5, 4.0, 'ANALOG\nOUT', color=ACCENT, fontsize=8, w=1.4, h=0.6)

    # === RC FEEDBACK LOOP (right side) ===
    fb_box = FancyBboxPatch((15.5, 5.5), 3.5, 2.0, boxstyle="round,pad=0.1",
                             facecolor=BG, edgecolor=ORANGE, lw=2, zorder=3)
    ax3.add_patch(fb_box)
    ax3.text(17.25, 7.1, 'RC PULSE FEEDBACK', fontsize=8, fontweight='bold',
             color=ORANGE, ha='center')
    ax3.text(17.25, 6.5, 'Rfb=100kΩ  Cfb=100pF', fontsize=7, color=COMP, ha='center')
    ax3.text(17.25, 6.1, 'τ = RfbCfb = 10μs', fontsize=7, color=ORANGE, ha='center',
             fontweight='bold')
    ax3.text(17.25, 5.7, 'fb_en: ON/OFF', fontsize=6, color=ORANGE, ha='center')

    # Feedback wire: out → RC → back to BVpar
    wire(ax3, [(14.5, 3.5), (16.0, 3.5), (16.0, 5.5)], color=ORANGE, lw=1.8)
    wire(ax3, [(17.25, 5.5), (17.25, 0.0), (9.0, 0.0), (9.0, 0.5)], color=ORANGE, lw=1.8)
    ax3.text(17.5, 2.5, 'fb_norm\n→ modulates\nBVpar', fontsize=6, color=ORANGE, ha='center')

    # Feedback loop arrow label
    ax3.annotate('FEEDBACK\nLOOP', xy=(17.25, 3.0), fontsize=7, color=ORANGE,
                 ha='center', fontweight='bold')

    # Score
    ax3.text(10, -1.0, '✓ Noise amp.=1.08%   ✓ BVpar shift=4.06%   '
             '✓ Temp mod.=1.15%   ✓ Combined=1.63%',
             ha='center', fontsize=8, color=ACCENT, fontweight='bold',
             bbox=dict(boxstyle='round,pad=0.2', facecolor=BG, edgecolor=ACCENT, lw=1.5))

    # Key insight at bottom
    ax3.text(10, -1.8, 'Key: noise alone absorbed by saturation; noise THROUGH feedback → '
             '1.08% amplification (mirrors FEEL XOR mechanism)',
             ha='center', fontsize=7, color=PURPLE, style='italic')

    fig.savefig(f'{OUTDIR}/bridge_circuits_overview.png', dpi=200,
                bbox_inches='tight', facecolor=BG)
    plt.close()
    print(f'  Saved bridge_circuits_overview.png')


# ============================================================
if __name__ == '__main__':
    print('Generating circuit schematics...')
    schematic_v8_mode_register()
    schematic_v9_wgp_ensemble()
    schematic_v10_stochastic_feedback()
    schematic_overview()
    print('Done!')
