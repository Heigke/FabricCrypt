#!/usr/bin/env python3
"""
Professional EDA-style schematic for NS-RAM ↔ GPU Bridge v2
Dark background, proper circuit symbols, Pazos paper aesthetic
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, Arc
import numpy as np

# ─── Color palette (LTspice dark theme) ───
BG      = '#1a1a2e'
WIRE    = '#00cc66'
WIRE2   = '#00aaff'
WIRE3   = '#ff6644'
NODE    = '#ff3333'
TEXT    = '#ffdd44'
ANNOT   = '#ffdd44'
LABEL   = '#cccccc'
DIM     = '#666666'
BOX_LIF = '#00ccaa'
BOX_AVL = '#ff8844'
BOX_GPU = '#6688ff'
BOX_SYN = '#cccc00'
COMP    = '#00ee77'
VDD_C   = '#ff4444'
GND_C   = '#888888'

def draw_resistor(ax, x0, y0, x1, y1, label=None, color=COMP, lw=1.5):
    """Draw zigzag resistor between two points (vertical or horizontal)"""
    dx, dy = x1 - x0, y1 - y0
    length = np.sqrt(dx**2 + dy**2)
    n_zag = 6
    # Unit vectors
    ux, uy = dx/length, dy/length
    # Perpendicular
    px, py = -uy, ux

    lead = 0.15 * length
    zag_len = (length - 2*lead) / n_zag
    amp = 0.08 * length

    pts_x, pts_y = [x0], [y0]
    # Lead in
    cx, cy = x0 + lead*ux, y0 + lead*uy
    pts_x.append(cx); pts_y.append(cy)
    # Zigzag
    for i in range(n_zag):
        mid = cx + (i+0.5)*zag_len*ux, cy + (i+0.5)*zag_len*uy
        sign = 1 if i % 2 == 0 else -1
        zx = cx + (i+0.5)*zag_len*ux + sign*amp*px
        zy = cy + (i+0.5)*zag_len*uy + sign*amp*py
        pts_x.append(zx); pts_y.append(zy)
    # Lead out
    pts_x.append(x1 - lead*ux + lead*ux)
    pts_y.append(y1 - lead*uy + lead*uy)
    pts_x.append(x1); pts_y.append(y1)

    ax.plot(pts_x, pts_y, color=color, lw=lw, solid_capstyle='round')
    # Red dots at terminals
    ax.plot(x0, y0, 'o', color=NODE, ms=4, zorder=10)
    ax.plot(x1, y1, 'o', color=NODE, ms=4, zorder=10)
    if label:
        mx, my = (x0+x1)/2 + 0.15*px, (y0+y1)/2 + 0.15*py
        ax.text(mx, my, label, color=LABEL, fontsize=7, ha='center', va='center',
                fontfamily='monospace')

def draw_capacitor(ax, x0, y0, x1, y1, label=None, color=COMP, lw=1.5):
    """Draw capacitor between two points"""
    dx, dy = x1 - x0, y1 - y0
    length = np.sqrt(dx**2 + dy**2)
    ux, uy = dx/length, dy/length
    px, py = -uy, ux

    mx, my = (x0+x1)/2, (y0+y1)/2
    gap = 0.04 * length
    plate = 0.12 * length

    # Lead to plate 1
    p1x, p1y = mx - gap*ux, my - gap*uy
    ax.plot([x0, p1x], [y0, p1y], color=color, lw=lw)
    # Plate 1
    ax.plot([p1x - plate*px, p1x + plate*px],
            [p1y - plate*py, p1y + plate*py], color=color, lw=2)
    # Lead from plate 2
    p2x, p2y = mx + gap*ux, my + gap*uy
    ax.plot([p2x, x1], [p2y, y1], color=color, lw=lw)
    # Plate 2
    ax.plot([p2x - plate*px, p2x + plate*px],
            [p2y - plate*py, p2y + plate*py], color=color, lw=2)

    ax.plot(x0, y0, 'o', color=NODE, ms=4, zorder=10)
    ax.plot(x1, y1, 'o', color=NODE, ms=4, zorder=10)
    if label:
        lx, ly = mx + 0.18*px + 0.02*ux, my + 0.18*py + 0.02*uy
        ax.text(lx, ly, label, color=LABEL, fontsize=7, ha='center', va='center',
                fontfamily='monospace')

def draw_nmos(ax, x, y, label=None, w=None, l=None, color=COMP, lw=1.5, flip=False):
    """Draw NMOS transistor symbol at (x,y) = gate connection point
    Gate on left, Drain top-right, Source bottom-right
    Returns (gate_xy, drain_xy, source_xy)
    """
    s = 0.3  # symbol size
    sign = -1 if flip else 1

    # Gate line (horizontal stub from gate point to channel)
    gx = x + sign*0.15
    ax.plot([x, gx], [y, y], color=color, lw=lw)
    # Gate plate (vertical)
    ax.plot([gx, gx], [y - s*0.4, y + s*0.4], color=color, lw=2)

    # Channel (vertical, slightly offset from gate)
    cx = gx + sign*0.06
    ax.plot([cx, cx], [y - s*0.4, y + s*0.4], color=color, lw=1.5)

    # Drain (top) - horizontal then up
    dx_pt = cx + sign*0.15
    dy_pt = y + s*0.4
    ax.plot([cx, dx_pt], [dy_pt, dy_pt], color=color, lw=lw)
    drain = (dx_pt, dy_pt + 0.15)
    ax.plot([dx_pt, drain[0]], [dy_pt, drain[1]], color=color, lw=lw)

    # Source (bottom) - horizontal then down
    sy_pt = y - s*0.4
    ax.plot([cx, dx_pt], [sy_pt, sy_pt], color=color, lw=lw)
    source = (dx_pt, sy_pt - 0.15)
    ax.plot([dx_pt, source[0]], [sy_pt, source[1]], color=color, lw=lw)

    # Arrow on source (pointing inward for NMOS)
    arr_y = sy_pt
    arr_x = cx + sign*0.03
    ax.annotate('', xy=(cx + sign*0.12, arr_y), xytext=(cx + sign*0.02, arr_y),
                arrowprops=dict(arrowstyle='->', color=color, lw=1.2))

    # Body connection (middle of channel)
    body = (dx_pt, y)

    ax.plot(x, y, 'o', color=NODE, ms=4, zorder=10)
    ax.plot(*drain, 'o', color=NODE, ms=4, zorder=10)
    ax.plot(*source, 'o', color=NODE, ms=4, zorder=10)

    if label:
        ax.text(dx_pt + sign*0.15, y, label, color=TEXT, fontsize=7,
                ha='left' if not flip else 'right', va='center', fontfamily='monospace',
                fontweight='bold')
    if w and l:
        ax.text(dx_pt + sign*0.15, y - 0.12, f'W={w}\nL={l}', color=DIM, fontsize=5.5,
                ha='left' if not flip else 'right', va='top', fontfamily='monospace')

    return (x, y), drain, source

def draw_pmos(ax, x, y, label=None, w=None, l=None, color=COMP, lw=1.5):
    """Draw PMOS transistor symbol. Same layout as NMOS but with bubble on gate."""
    s = 0.3
    # Gate line with bubble
    gx = x + 0.10
    ax.plot([x, gx], [y, y], color=color, lw=lw)
    circle = plt.Circle((gx + 0.03, y), 0.03, fill=False, ec=color, lw=1.2)
    ax.add_patch(circle)

    # Gate plate
    gpx = gx + 0.08
    ax.plot([gpx, gpx], [y - s*0.4, y + s*0.4], color=color, lw=2)

    # Channel
    cx = gpx + 0.06
    ax.plot([cx, cx], [y - s*0.4, y + s*0.4], color=color, lw=1.5)

    # Source (top for PMOS)
    dx_pt = cx + 0.15
    sy_pt = y + s*0.4
    ax.plot([cx, dx_pt], [sy_pt, sy_pt], color=color, lw=lw)
    source = (dx_pt, sy_pt + 0.15)
    ax.plot([dx_pt, source[0]], [sy_pt, source[1]], color=color, lw=lw)

    # Drain (bottom for PMOS)
    dy_pt = y - s*0.4
    ax.plot([cx, dx_pt], [dy_pt, dy_pt], color=color, lw=lw)
    drain = (dx_pt, dy_pt - 0.15)
    ax.plot([dx_pt, drain[0]], [dy_pt, drain[1]], color=color, lw=lw)

    # Arrow on source pointing outward for PMOS
    ax.annotate('', xy=(cx + 0.02, sy_pt), xytext=(cx + 0.12, sy_pt),
                arrowprops=dict(arrowstyle='->', color=color, lw=1.2))

    ax.plot(x, y, 'o', color=NODE, ms=4, zorder=10)
    ax.plot(*drain, 'o', color=NODE, ms=4, zorder=10)
    ax.plot(*source, 'o', color=NODE, ms=4, zorder=10)

    if label:
        ax.text(dx_pt + 0.15, y, label, color=TEXT, fontsize=7, ha='left', va='center',
                fontfamily='monospace', fontweight='bold')
    if w and l:
        ax.text(dx_pt + 0.15, y - 0.12, f'W={w}\nL={l}', color=DIM, fontsize=5.5,
                ha='left', va='top', fontfamily='monospace')

    return (x, y), drain, source

def draw_npn(ax, x, y, label=None, color=COMP, lw=1.5):
    """Draw NPN BJT symbol at (x,y) = base point
    Returns (base, collector, emitter)
    """
    s = 0.25
    # Base line
    bx = x + 0.15
    ax.plot([x, bx], [y, y], color=color, lw=lw)
    # Vertical bar (base region)
    ax.plot([bx, bx], [y - s, y + s], color=color, lw=2.5)

    # Collector (upper right)
    cx_pt, cy_pt = bx + 0.25, y + s + 0.1
    ax.plot([bx, cx_pt], [y + s*0.4, cy_pt], color=color, lw=lw)
    collector = (cx_pt, cy_pt + 0.1)
    ax.plot([cx_pt, collector[0]], [cy_pt, collector[1]], color=color, lw=lw)

    # Emitter (lower right) with arrow
    ex_pt, ey_pt = bx + 0.25, y - s - 0.1
    ax.plot([bx, ex_pt], [y - s*0.4, ey_pt], color=color, lw=lw)
    emitter = (ex_pt, ey_pt - 0.1)
    ax.plot([ex_pt, emitter[0]], [ey_pt, emitter[1]], color=color, lw=lw)

    # Arrow on emitter
    ax.annotate('', xy=(ex_pt, ey_pt), xytext=(bx + 0.05, y - s*0.5),
                arrowprops=dict(arrowstyle='->', color=color, lw=1.5))

    ax.plot(x, y, 'o', color=NODE, ms=4, zorder=10)
    ax.plot(*collector, 'o', color=NODE, ms=4, zorder=10)
    ax.plot(*emitter, 'o', color=NODE, ms=4, zorder=10)

    if label:
        ax.text(bx + 0.35, y, label, color=TEXT, fontsize=7, ha='left', va='center',
                fontfamily='monospace', fontweight='bold')

    return (x, y), collector, emitter

def draw_diode(ax, x0, y0, x1, y1, label=None, zener=False, color=COMP, lw=1.5):
    """Draw diode between two points (anode=start, cathode=end)"""
    dx, dy = x1 - x0, y1 - y0
    length = np.sqrt(dx**2 + dy**2)
    ux, uy = dx/length, dy/length
    px, py = -uy, ux

    mx, my = (x0+x1)/2, (y0+y1)/2
    tri_h = 0.08 * length
    tri_w = 0.06 * length

    # Lead in
    ax.plot([x0, mx - tri_h*ux], [y0, my - tri_h*uy], color=color, lw=lw)
    # Triangle
    t_base_x = mx - tri_h*ux
    t_base_y = my - tri_h*uy
    t_tip_x = mx + tri_h*ux
    t_tip_y = my + tri_h*uy
    triangle = plt.Polygon([
        [t_base_x - tri_w*px, t_base_y - tri_w*py],
        [t_base_x + tri_w*px, t_base_y + tri_w*py],
        [t_tip_x, t_tip_y]
    ], closed=True, fc=color, ec=color, lw=lw, alpha=0.6)
    ax.add_patch(triangle)
    # Cathode bar
    ax.plot([t_tip_x - tri_w*px, t_tip_x + tri_w*px],
            [t_tip_y - tri_w*py, t_tip_y + tri_w*py], color=color, lw=2)
    if zener:
        # Zener bends
        ax.plot([t_tip_x - tri_w*px, t_tip_x - tri_w*px - 0.02*ux],
                [t_tip_y - tri_w*py, t_tip_y - tri_w*py - 0.02*uy], color=color, lw=2)
        ax.plot([t_tip_x + tri_w*px, t_tip_x + tri_w*px + 0.02*ux],
                [t_tip_y + tri_w*py, t_tip_y + tri_w*py + 0.02*uy], color=color, lw=2)
    # Lead out
    ax.plot([t_tip_x, x1], [t_tip_y, y1], color=color, lw=lw)

    ax.plot(x0, y0, 'o', color=NODE, ms=4, zorder=10)
    ax.plot(x1, y1, 'o', color=NODE, ms=4, zorder=10)
    if label:
        lx = mx + 0.15*px
        ly = my + 0.15*py
        ax.text(lx, ly, label, color=LABEL, fontsize=7, ha='center', va='center',
                fontfamily='monospace')

def draw_gnd(ax, x, y, color=GND_C, size=0.12):
    """Draw ground symbol"""
    ax.plot([x, x], [y, y - size*0.4], color=color, lw=1.5)
    for i, w in enumerate([1.0, 0.6, 0.3]):
        yy = y - size*0.4 - i*size*0.2
        ax.plot([x - size*w*0.5, x + size*w*0.5], [yy, yy], color=color, lw=1.5)

def draw_vdd(ax, x, y, label='VDD', color=VDD_C, size=0.1):
    """Draw VDD power symbol"""
    ax.plot([x, x], [y, y + size], color=color, lw=1.5)
    ax.plot([x - size, x + size], [y + size, y + size], color=color, lw=2)
    ax.text(x, y + size + 0.05, label, color=color, fontsize=7, ha='center', va='bottom',
            fontfamily='monospace', fontweight='bold')

def draw_wire(ax, points, color=WIRE, lw=1.2):
    """Draw wire through list of (x,y) points"""
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    ax.plot(xs, ys, color=color, lw=lw, solid_capstyle='round')

def draw_node(ax, x, y):
    ax.plot(x, y, 'o', color=NODE, ms=5, zorder=10)

def draw_section_box(ax, x0, y0, x1, y1, label, color, lw=1.5):
    """Draw dashed section box with label"""
    rect = mpatches.FancyBboxPatch((x0, y0), x1-x0, y1-y0,
                                     boxstyle='round,pad=0.05',
                                     fill=False, ec=color, lw=lw,
                                     ls='--', alpha=0.8)
    ax.add_patch(rect)
    ax.text(x0 + 0.1, y1 - 0.1, label, color=color, fontsize=9,
            fontweight='bold', fontfamily='monospace', va='top')

def draw_annotation(ax, x, y, text, arrow_to=None, color=ANNOT):
    """Draw yellow annotation with optional arrow"""
    if arrow_to:
        ax.annotate(text, xy=arrow_to, xytext=(x, y),
                    color=color, fontsize=8, fontweight='bold', fontfamily='monospace',
                    arrowprops=dict(arrowstyle='->', color=color, lw=1.5),
                    ha='center', va='center')
    else:
        ax.text(x, y, text, color=color, fontsize=8, fontweight='bold',
                fontfamily='monospace', ha='center', va='center')

def draw_vsource(ax, x, y_bot, y_top, label=None, color=COMP, lw=1.5):
    """Draw voltage source (circle with +/-)"""
    mx, my = x, (y_bot + y_top) / 2
    r = abs(y_top - y_bot) * 0.25
    circle = plt.Circle((mx, my), r, fill=False, ec=color, lw=lw)
    ax.add_patch(circle)
    ax.plot([x, x], [y_bot, my - r], color=color, lw=lw)
    ax.plot([x, x], [my + r, y_top], color=color, lw=lw)
    ax.text(mx, my + r*0.4, '+', color=color, fontsize=8, ha='center', va='center')
    ax.text(mx, my - r*0.4, '−', color=color, fontsize=8, ha='center', va='center')
    ax.plot(x, y_bot, 'o', color=NODE, ms=4, zorder=10)
    ax.plot(x, y_top, 'o', color=NODE, ms=4, zorder=10)
    if label:
        ax.text(mx + 0.2, my, label, color=LABEL, fontsize=7, ha='left', va='center',
                fontfamily='monospace')

def draw_bsource(ax, x, y, label, color='#ff9944', size=0.2):
    """Draw behavioral source (diamond shape)"""
    diamond = plt.Polygon([
        [x, y + size], [x + size*0.6, y], [x, y - size], [x - size*0.6, y]
    ], closed=True, fill=False, ec=color, lw=1.5)
    ax.add_patch(diamond)
    ax.text(x, y, 'B', color=color, fontsize=8, ha='center', va='center',
            fontweight='bold', fontfamily='monospace')
    ax.text(x + size*0.7, y, label, color=color, fontsize=6, ha='left', va='center',
            fontfamily='monospace')


# ═══════════════════════════════════════════════════════════════
# MAIN FIGURE
# ═══════════════════════════════════════════════════════════════
fig, ax = plt.subplots(1, 1, figsize=(22, 14), facecolor=BG)
ax.set_facecolor(BG)
ax.set_xlim(-0.5, 16)
ax.set_ylim(-1.5, 10.5)
ax.set_aspect('equal')
ax.axis('off')

# Title
ax.text(8, 10.2, 'NS-RAM ↔ GPU Analog Bridge v2 — Transistor-Level Schematic',
        color='#ffffff', fontsize=14, ha='center', va='center',
        fontfamily='monospace', fontweight='bold')
ax.text(8, 9.85, 'Behavioral Avalanche Model  •  Pazos LIF Neuron  •  GPU FP16 Rounding Feedback',
        color=DIM, fontsize=9, ha='center', va='center', fontfamily='monospace')

# ═══════════════════════════════════════════════════════════════
# SECTION 1: LANZA NS-RAM DEVICE (left)
# ═══════════════════════════════════════════════════════════════
draw_section_box(ax, 0, 0.5, 5.8, 9.2, 'Lanza NS-RAM Device', BOX_AVL)
ax.text(0.15, 9.0, '(Punch-Through Avalanche)', color=BOX_AVL, fontsize=7,
        fontfamily='monospace', alpha=0.7)

# VDD rail at top
draw_vdd(ax, 2.8, 8.5, 'VDD = 1.8V', color=VDD_C)

# R_drain (R3) from VDD to drain node
draw_resistor(ax, 2.8, 8.5, 2.8, 7.5, 'R3\n1kΩ', color=COMP)

# M2 - PTM130 NMOS (main NS-RAM transistor)
# Gate at left, drain/source on right side
g2, d2, s2 = draw_nmos(ax, 1.0, 6.5, 'M2', w='0.5μ', l='250n', color=COMP)

# Wire from R3 bottom to M2 drain
draw_wire(ax, [(2.8, 7.5), (2.8, 6.65), (d2[0], d2[1])], color=WIRE)
draw_node(ax, 2.8, 7.5)

# Label drain node
ax.text(2.8, 7.6, 'd_node', color=WIRE, fontsize=6, ha='center', va='bottom',
        fontfamily='monospace')

# Q1 - NPN BJT (avalanche transistor)
# Place to the right of M2
b1, c1, e1 = draw_npn(ax, 3.5, 6.5, 'Q1', color='#ff8866')
ax.text(3.5 + 0.6, 6.3, 'avalBJT\n(NPN)', color='#ff8866', fontsize=6,
        fontfamily='monospace')

# Wire collector to drain node
draw_wire(ax, [(c1[0], c1[1]), (c1[0], 7.5), (2.8, 7.5)], color=WIRE)

# Wire emitter to source area
draw_wire(ax, [(e1[0], e1[1]), (e1[0], 5.5)], color=WIRE)

# M2 source to ground area
draw_wire(ax, [(s2[0], s2[1]), (s2[0], 5.2)], color=WIRE)
ax.text(s2[0], 5.3, 's_node', color=WIRE, fontsize=6, ha='center', va='top',
        fontfamily='monospace')

# R_source (R4) to ground
draw_resistor(ax, s2[0], 5.2, s2[0], 4.2, 'R4\n100Ω', color=COMP)
draw_gnd(ax, s2[0], 4.2)

# D1 - Zener diode (drain to source)
draw_diode(ax, 4.8, 7.5, 4.8, 5.5, 'D1\nzener', zener=True, color='#44aaff')
# Connect D1 to drain rail
draw_wire(ax, [(2.8, 7.5), (4.8, 7.5)], color=WIRE)
draw_node(ax, 4.8, 7.5)
# Connect D1 cathode to source rail
draw_wire(ax, [(4.8, 5.5), (e1[0], 5.5)], color=WIRE)
draw_node(ax, e1[0], 5.5)

# D2 - Feedback diode
draw_diode(ax, 5.3, 7.5, 5.3, 5.5, 'D2', color='#44aaff')
draw_wire(ax, [(4.8, 7.5), (5.3, 7.5)], color=WIRE)
draw_wire(ax, [(4.8, 5.5), (5.3, 5.5)], color=WIRE)

# Behavioral avalanche source (key innovation)
draw_bsource(ax, 1.5, 3.5, '', color=BOX_AVL)
ax.text(1.5, 2.9, 'Baval', color=BOX_AVL, fontsize=8, ha='center', fontweight='bold',
        fontfamily='monospace')
ax.text(1.5, 2.5, 'I = 10p·min(exp((Vcb−BVpar)/50mV), 200)',
        color=BOX_AVL, fontsize=6, ha='center', fontfamily='monospace')

# Wire from Baval up to d_node / s_node
draw_wire(ax, [(1.5, 3.7), (1.5, 5.5), (s2[0], 5.5)], color='#ff8844', lw=1.5)
draw_wire(ax, [(1.5, 3.3), (1.5, 2.0), (0.5, 2.0), (0.5, 7.5), (2.8, 7.5)],
          color='#ff8844', lw=1.5)

# BVpar behavioral source
draw_bsource(ax, 4.0, 3.5, '', color='#ffaa00')
ax.text(4.0, 2.9, 'BVpar', color='#ffaa00', fontsize=8, ha='center', fontweight='bold',
        fontfamily='monospace')
ax.text(4.0, 2.5, 'V = 3.5 − 1.5·V(gate)',
        color='#ffaa00', fontsize=6, ha='center', fontfamily='monospace')

# Gate bias source
draw_vsource(ax, 0.8, 4.5, 5.5, 'Vg_bias\n0.4V', color=COMP)
# Wire gate bias to M2 gate
draw_wire(ax, [(0.8, 5.5), (0.8, 6.5), (g2[0], g2[1])], color=WIRE2)
draw_gnd(ax, 0.8, 4.5)

# Gate label
ax.text(0.5, 6.7, 'nsram_gate', color=WIRE2, fontsize=6, ha='center',
        fontfamily='monospace')

# M1 - BSS145 (external MOSFET for gate control)
g1, d1, s1_m1 = draw_nmos(ax, 3.8, 1.5, 'M1', w='BSS145', l='', color='#88cc44')
draw_gnd(ax, s1_m1[0], s1_m1[1])

# ═══════════════════════════════════════════════════════════════
# SECTION 2: PAZOS LIF NEURON (center)
# ═══════════════════════════════════════════════════════════════
draw_section_box(ax, 6, 0.5, 10.5, 9.2, 'Pazos LIF Neuron', BOX_LIF)
ax.text(6.15, 9.0, '(Integrate-and-Fire + Reset)', color=BOX_LIF, fontsize=7,
        fontfamily='monospace', alpha=0.7)

# VDD for neuron
draw_vdd(ax, 8.2, 8.5, 'VDD', color=VDD_C)

# Integration capacitor Cint
draw_capacitor(ax, 7.5, 5.0, 7.5, 3.5, 'Cint\n102fF', color='#00ddaa')
draw_gnd(ax, 7.5, 3.5)

# Membrane node
ax.text(7.5, 5.2, 'V_membrane', color=BOX_LIF, fontsize=7, ha='center', va='bottom',
        fontfamily='monospace', fontweight='bold')
draw_node(ax, 7.5, 5.0)

# Leak transistor M_leak
g_lk, d_lk, s_lk = draw_nmos(ax, 6.5, 5.0, 'M_leak', w='130n', l='130n', color='#00ddaa')
# Wire drain to membrane
draw_wire(ax, [(d_lk[0], d_lk[1]), (7.5, d_lk[1]), (7.5, 5.0)], color=BOX_LIF)
# Leak gate bias
ax.text(6.2, 5.0, 'Vlk\n0.2V', color=DIM, fontsize=6, ha='right', va='center',
        fontfamily='monospace')
draw_gnd(ax, s_lk[0], s_lk[1])

# Double inverter (threshold detector)
# Inv1: PMOS + NMOS
g_p1, d_p1, s_p1 = draw_pmos(ax, 7.5, 7.5, 'Mp1', w='460n', l='130n', color='#00ddaa')
g_n1, d_n1, s_n1 = draw_nmos(ax, 7.5, 6.3, 'Mn1', w='230n', l='130n', color='#00ddaa')
# VDD to PMOS source
draw_wire(ax, [(s_p1[0], s_p1[1]), (s_p1[0], 8.5), (8.2, 8.5)], color=VDD_C)
# PMOS drain to NMOS drain
draw_wire(ax, [(d_p1[0], d_p1[1]), (d_n1[0], d_n1[1])], color=BOX_LIF)
# NMOS source to ground
draw_gnd(ax, s_n1[0], s_n1[1])

# Inv1 output node
inv1_out = (d_p1[0] + 0.3, (d_p1[1] + d_n1[1])/2)
draw_wire(ax, [(d_p1[0], (d_p1[1] + d_n1[1])/2), inv1_out], color=BOX_LIF)
draw_node(ax, *inv1_out)

# Wire membrane to inverter gates
draw_wire(ax, [(7.5, 5.0), (7.5, 6.3)], color=BOX_LIF)
draw_wire(ax, [(7.5, 6.3), (7.5, 7.5)], color=BOX_LIF)

# Inv2 (simplified as box)
inv2_x, inv2_y = 9.2, 7.0
ax.add_patch(plt.Polygon([[inv2_x, inv2_y+0.3], [inv2_x, inv2_y-0.3],
                           [inv2_x+0.4, inv2_y]], closed=True,
                          fill=False, ec=BOX_LIF, lw=1.5))
# Bubble
circle2 = plt.Circle((inv2_x + 0.45, inv2_y), 0.05, fill=False, ec=BOX_LIF, lw=1.2)
ax.add_patch(circle2)
ax.text(inv2_x + 0.2, inv2_y + 0.45, 'Inv2', color=BOX_LIF, fontsize=6,
        fontfamily='monospace')
# Wire inv1 output to inv2 input
draw_wire(ax, [inv1_out, (inv2_x, inv2_y)], color=BOX_LIF)

# Spike output
spike_out = (inv2_x + 0.7, inv2_y)
draw_wire(ax, [(inv2_x + 0.5, inv2_y), spike_out], color=WIRE3)
draw_node(ax, *spike_out)
ax.text(spike_out[0] + 0.1, spike_out[1] + 0.15, 'V_spike', color=WIRE3, fontsize=8,
        ha='left', fontweight='bold', fontfamily='monospace')

# Reset transistor (latching)
g_rst, d_rst, s_rst = draw_nmos(ax, 9.5, 5.0, 'M_rst', w='460n', l='130n', color='#ff6666')
# Reset gate from spike
draw_wire(ax, [spike_out, (spike_out[0], 5.0), (9.5, 5.0)], color=WIRE3)
# Reset drain to membrane
draw_wire(ax, [(d_rst[0], d_rst[1]), (d_rst[0], 5.0+0.55), (7.5, 5.0+0.55), (7.5, 5.0)],
          color='#ff6666')
draw_gnd(ax, s_rst[0], s_rst[1])
ax.text(9.8, 4.5, 'Self-Reset\n(Latch)', color='#ff6666', fontsize=6, ha='center',
        fontfamily='monospace')

# Synaptic input from NS-RAM to LIF
draw_wire(ax, [(5.3, 7.5), (5.8, 7.5), (5.8, 5.0), (6.3, 5.0), (6.5, 5.0)],
          color=WIRE, lw=2)
draw_annotation(ax, 5.8, 8.2, 'Synaptic\nCoupling', arrow_to=(5.8, 7.7))

# ═══════════════════════════════════════════════════════════════
# SECTION 3: GPU FP16 ROUNDING FEEDBACK (right)
# ═══════════════════════════════════════════════════════════════
draw_section_box(ax, 11, 0.5, 15.5, 9.2, 'GPU FP16 Rounding Feedback', BOX_GPU)
ax.text(11.15, 9.0, '(AMD gfx1151 s_setreg)', color=BOX_GPU, fontsize=7,
        fontfamily='monospace', alpha=0.7)

# Spike detector block
det_x, det_y = 11.8, 7.0
ax.add_patch(mpatches.FancyBboxPatch((det_x, det_y-0.4), 1.8, 0.8,
             boxstyle='round,pad=0.05', fill=True, fc='#222244', ec=BOX_GPU, lw=1.5))
ax.text(det_x + 0.9, det_y, 'Spike Detector', color=BOX_GPU, fontsize=7,
        ha='center', va='center', fontfamily='monospace', fontweight='bold')
ax.text(det_x + 0.9, det_y - 0.25, 'V(s) > max(2·avg, 1μA)', color=DIM, fontsize=5.5,
        ha='center', va='center', fontfamily='monospace')

# Wire spike to detector
draw_wire(ax, [spike_out, (11.0, spike_out[1]), (11.0, det_y), (det_x, det_y)],
          color=WIRE3, lw=2)

# MAC compute block
mac_x, mac_y = 11.8, 5.5
ax.add_patch(mpatches.FancyBboxPatch((mac_x, mac_y-0.4), 1.8, 0.8,
             boxstyle='round,pad=0.05', fill=True, fc='#222244', ec=BOX_GPU, lw=1.5))
ax.text(mac_x + 0.9, mac_y, 'MAC Unit', color=BOX_GPU, fontsize=7,
        ha='center', va='center', fontfamily='monospace', fontweight='bold')
ax.text(mac_x + 0.9, mac_y - 0.25, 'out = in·(1−0.002·spike)', color=DIM, fontsize=5.5,
        ha='center', va='center', fontfamily='monospace')

# Arrow detector → MAC
ax.annotate('', xy=(mac_x + 0.9, mac_y + 0.4), xytext=(det_x + 0.9, det_y - 0.4),
            arrowprops=dict(arrowstyle='->', color=BOX_GPU, lw=2))

# Rounding mode block
rnd_x, rnd_y = 11.8, 4.0
ax.add_patch(mpatches.FancyBboxPatch((rnd_x, rnd_y-0.4), 1.8, 0.8,
             boxstyle='round,pad=0.05', fill=True, fc='#222244', ec='#8888ff', lw=1.5))
ax.text(rnd_x + 0.9, rnd_y, 's_setreg', color='#aabbff', fontsize=7,
        ha='center', va='center', fontfamily='monospace', fontweight='bold')
ax.text(rnd_x + 0.9, rnd_y - 0.25, 'hwreg(MODE,0,8)', color=DIM, fontsize=5.5,
        ha='center', va='center', fontfamily='monospace')

# Arrow MAC → rounding
ax.annotate('', xy=(rnd_x + 0.9, rnd_y + 0.4), xytext=(mac_x + 0.9, mac_y - 0.4),
            arrowprops=dict(arrowstyle='->', color=BOX_GPU, lw=2))

# Gate feedback (right to left)
# Bgate behavioral source
draw_bsource(ax, 14.2, 5.5, '', color='#ff66aa')
ax.text(14.2, 4.9, 'Bgate', color='#ff66aa', fontsize=7, ha='center', fontweight='bold',
        fontfamily='monospace')
ax.text(14.2, 4.5, 'V = Vbias + 0.1·V(mac)', color='#ff66aa', fontsize=6, ha='center',
        fontfamily='monospace')

# Wire MAC output to Bgate
draw_wire(ax, [(mac_x + 1.8, mac_y), (14.2, mac_y), (14.2, 5.7)], color='#ff66aa')

# Feedback arrow: Bgate → gate (going all the way left)
fb_y = 1.2
draw_wire(ax, [(14.2, 5.3), (14.2, fb_y), (0.8, fb_y), (0.8, 4.5)],
          color='#ff66aa', lw=2)

# Feedback annotation
draw_annotation(ax, 7.5, 0.7, 'Bidirectional Feedback Loop: spike → truncation → gate modulation → BVpar shift',
                color='#ff66aa')

# Synaptic weight resistors
draw_section_box(ax, 11.2, 1.5, 15.2, 3.2, 'Synaptic Weights', BOX_SYN)

syn_labels = ['Rsyn1\n50kΩ', 'Rsyn2\n75kΩ', 'Rsyn3\n60kΩ', 'Rsyn4\n80kΩ']
for i, sl in enumerate(syn_labels):
    sx = 11.8 + i * 0.9
    draw_resistor(ax, sx, 2.8, sx, 1.8, sl, color='#cccc44')

# ═══════════════════════════════════════════════════════════════
# KEY ANNOTATIONS
# ═══════════════════════════════════════════════════════════════

# Energy annotation
ax.text(8.2, 0.0, '46.2 fJ/spike  •  402 kHz  •  79 spikes/200μs  •  Gate swing: 75mV  •  BVpar swing: 112.5mV',
        color='#88ff88', fontsize=8, ha='center', va='center', fontfamily='monospace',
        fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.3', fc='#0a0a1a', ec='#88ff88', lw=1, alpha=0.9))

# Avalanche annotation
draw_annotation(ax, 2.0, 9.5, 'BVcbo replaced by explicit\nbehavioral B-source',
                arrow_to=(1.5, 3.8), color='#ff8844')

# Pazos annotation
draw_annotation(ax, 8.2, 9.5, 'Pazos (2024): 0.2−21 fJ/spike\nOur membrane: 18.4 fJ ✓',
                color='#00ffaa')

# GPU annotation
draw_annotation(ax, 14.0, 9.5, 'FP16 rounding mode\ncontrols avalanche threshold',
                arrow_to=(14.2, 5.8), color='#6688ff')

# Reference
ax.text(8, -1.2, 'Lanza et al. (Zenodo 10.5281/zenodo.13843362)  •  Pazos et al. IEEE JSSC 2024  •  FEEL Project gfx1151',
        color=DIM, fontsize=7, ha='center', va='center', fontfamily='monospace')

plt.tight_layout(pad=0.5)
plt.savefig('results/nsram_bridge_v2_eda.png', dpi=200, facecolor=BG,
            bbox_inches='tight', pad_inches=0.3)
print("Saved: results/nsram_bridge_v2_eda.png")
plt.close()
