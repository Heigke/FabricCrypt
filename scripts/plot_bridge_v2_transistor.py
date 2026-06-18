#!/usr/bin/env python3
"""Transistor-level schematic for NS-RAM <-> GPU Bridge v2."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch

fig, ax = plt.subplots(1, 1, figsize=(24, 16))
ax.set_xlim(0, 24)
ax.set_ylim(0, 16)
ax.set_aspect('equal')
ax.axis('off')

ax.text(12, 15.5, 'NS-RAM ↔ GPU Bridge v2 — Transistor-Level Netlist',
        ha='center', fontsize=16, fontweight='bold')

# ── helpers ──
def mosfet(x, y, label, info, color='#1565C0'):
    """Draw MOSFET symbol (simplified box)."""
    r = FancyBboxPatch((x-0.6, y-0.35), 1.2, 0.7, boxstyle="round,pad=0.05",
                        fc='#E3F2FD', ec=color, lw=1.5)
    ax.add_patch(r)
    ax.text(x, y+0.1, label, ha='center', fontsize=8, fontweight='bold', color=color)
    ax.text(x, y-0.15, info, ha='center', fontsize=6, color='#555')

def bjt(x, y, label, info):
    r = FancyBboxPatch((x-0.6, y-0.35), 1.2, 0.7, boxstyle="round,pad=0.05",
                        fc='#FCE4EC', ec='#C62828', lw=1.5)
    ax.add_patch(r)
    ax.text(x, y+0.1, label, ha='center', fontsize=8, fontweight='bold', color='#C62828')
    ax.text(x, y-0.15, info, ha='center', fontsize=6, color='#555')

def diode(x, y, label, info):
    r = FancyBboxPatch((x-0.55, y-0.3), 1.1, 0.6, boxstyle="round,pad=0.05",
                        fc='#F3E5F5', ec='#6A1B9A', lw=1.2)
    ax.add_patch(r)
    ax.text(x, y+0.07, label, ha='center', fontsize=7, fontweight='bold', color='#6A1B9A')
    ax.text(x, y-0.15, info, ha='center', fontsize=5.5, color='#555')

def cap(x, y, label, val):
    ax.plot([x-0.15, x-0.15], [y-0.2, y+0.2], 'k-', lw=2)
    ax.plot([x+0.15, x+0.15], [y-0.2, y+0.2], 'k-', lw=2)
    ax.text(x, y+0.35, label, ha='center', fontsize=7, fontweight='bold')
    ax.text(x, y-0.35, val, ha='center', fontsize=6, color='#555')

def resistor(x, y, label, val):
    ax.plot([x, x], [y-0.3, y+0.3], color='#795548', lw=2, ls='-')
    xs = [x-0.1, x+0.1, x-0.1, x+0.1, x-0.1, x+0.1]
    ys = [y-0.2, y-0.1, y, y+0.1, y+0.2, y+0.3]
    ax.plot(xs[:4], ys[:4], color='#795548', lw=1.5)
    ax.text(x+0.3, y+0.1, label, ha='left', fontsize=7, fontweight='bold', color='#795548')
    ax.text(x+0.3, y-0.1, val, ha='left', fontsize=6, color='#555')

def bsource(x, y, label, eq, color='#B71C1C'):
    r = FancyBboxPatch((x-1.1, y-0.4), 2.2, 0.8, boxstyle="round,pad=0.05",
                        fc='#FFCDD2', ec=color, lw=2)
    ax.add_patch(r)
    ax.text(x, y+0.15, label, ha='center', fontsize=8, fontweight='bold', color=color)
    ax.text(x, y-0.15, eq, ha='center', fontsize=6, color=color)

def wire(x1, y1, x2, y2, color='k', lw=1):
    ax.plot([x1, x2], [y1, y2], color=color, lw=lw, solid_capstyle='round')

def node(x, y, label, color='k'):
    ax.plot(x, y, 'o', color=color, ms=4, zorder=5)
    ax.text(x, y+0.25, label, ha='center', fontsize=7, fontweight='bold', color=color)

# ================================================================
# SECTION LABELS
# ================================================================
s1 = FancyBboxPatch((0.3, 8.5), 7.4, 6.5, boxstyle="round,pad=0.15",
                     fc='none', ec='#E65100', lw=2, ls='--')
ax.add_patch(s1)
ax.text(4, 14.7, 'LANZA NS-RAM DEVICE', fontsize=11, fontweight='bold', color='#E65100')

s2 = FancyBboxPatch((8.2, 8.5), 6.6, 6.5, boxstyle="round,pad=0.15",
                     fc='none', ec='#283593', lw=2, ls='--')
ax.add_patch(s2)
ax.text(11.5, 14.7, 'PAZOS LIF NEURON', fontsize=11, fontweight='bold', color='#283593')

s3 = FancyBboxPatch((15.3, 8.5), 8.2, 6.5, boxstyle="round,pad=0.15",
                     fc='none', ec='#00695C', lw=2, ls='--')
ax.add_patch(s3)
ax.text(19.4, 14.7, 'GPU FP16 MAC', fontsize=11, fontweight='bold', color='#00695C')

# ================================================================
# LANZA DEVICE (left)
# ================================================================
# VDD rail
wire(1, 14, 7, 14, '#C62828', 2)
ax.text(0.7, 14, 'Vd', fontsize=8, fontweight='bold', color='#C62828', ha='right')

# Vpulse
ax.text(1, 14.3, 'Vpulse 0→3.5V', fontsize=7, color='#C62828')
ax.text(1, 13.7, '@ 100kHz', fontsize=6, color='#888')

# R4 (drain sense)
resistor(2, 13.3, 'R4', '1Ω')
wire(2, 14, 2, 13.6)
wire(2, 13, 2, 12.7)
node(2, 12.7, 'd_node', '#C62828')

# di node
wire(2, 12.7, 3, 12.7)
wire(3, 12.7, 3, 12.3)
wire(5, 12.7, 5, 12.3)
wire(3, 12.7, 5, 12.7)
node(4, 12.7, 'di', '#C62828')

# M2
mosfet(3, 12, 'M2', 'PTM130 W=0.5μ')
ax.text(3, 11.45, 'L=250n', fontsize=6, color='#555', ha='center')

# Q1
bjt(5, 12, 'Q1', 'avalBJT')
ax.text(5, 11.45, 'area=0.1', fontsize=6, color='#555', ha='center')

# b_node
wire(3.6, 12, 4.4, 12)  # M2 bulk to Q1 base
node(4, 12, 'b_node', '#6A1B9A')

# s_node
wire(3, 11.65, 3, 11)
wire(5, 11.65, 5, 11)
wire(3, 11, 5, 11)
node(4, 11, 's_node', '#2E7D32')

# R3
resistor(4, 10.3, 'R3', '1Ω')
wire(4, 11, 4, 10.6)
wire(4, 10, 4, 9.5)
ax.text(4, 9.3, 'GND', fontsize=7, ha='center', color='#555')

# D1 (gate-bulk zener)
diode(1.5, 12, 'D1', 'zener')
wire(1.5, 11.7, 1.5, 11)
wire(1.5, 11, 3, 11)  # to s_node area
wire(1.5, 12.3, 1.5, 12.7)

# Gate connection
wire(1.5, 12.7, 1.5, 13.5)
node(1.5, 13.5, 'nsram_gate', '#E65100')

# M2 gate connection
wire(1.5, 13.5, 2.4, 13.5)
wire(2.4, 13.5, 2.4, 12.35)

# D2
diode(6.5, 11.5, 'D2', 'zener')
wire(5, 11, 6.5, 11)
wire(6.5, 11.2, 6.5, 11)
wire(6.5, 11.8, 6.5, 12)
wire(6.5, 12, 5, 12)  # back to b_node

# M1 (bulk access)
mosfet(6.5, 10, 'M1', 'BSS145', '#2E7D32')
wire(6.5, 10.35, 6.5, 11)
wire(6.5, 9.65, 6.5, 9.3)
ax.text(6.5, 9.1, 'GND', fontsize=7, ha='center', color='#555')
ax.text(7.3, 10, 'Vgb=2.5V→', fontsize=6, color='#2E7D32', ha='left')

# Baval (THE KEY FIX)
bsource(4, 9, 'Baval', 'I₀·exp((Vcb−BVpar)/50mV)', '#B71C1C')
wire(4, 9.4, 4, 9.5)
ax.annotate('', xy=(4, 9.45), xytext=(4, 10),
            arrowprops=dict(arrowstyle='->', color='#B71C1C', lw=1.5))
ax.text(5.5, 8.5, 'BVpar = 3.5 − 1.5·Vg', fontsize=7, fontweight='bold', color='#B71C1C')

# ================================================================
# PAZOS LIF NEURON (center)
# ================================================================
# VDD
wire(9, 14, 14, 14, '#283593', 2)
ax.text(8.7, 14, 'VDD=1.5V', fontsize=7, color='#283593', ha='right')

# Cint
cap(9.5, 12.5, 'Cint', '102fF')
node(9.5, 13, 'vmem', '#283593')
wire(9.5, 12.8, 9.5, 13)
wire(9.5, 12.2, 9.5, 11.8)
ax.text(9.5, 11.6, 'GND', fontsize=7, ha='center', color='#555')

# M_leak
mosfet(11, 12, 'M_leak', 'W=180n L=1μ', '#283593')
ax.text(11, 11.45, 'Vlk=0.2V', fontsize=6, color='#555', ha='center')
wire(9.5, 13, 11, 13)
wire(11, 13, 11, 12.35)
wire(11, 11.65, 11, 11.3)
ax.text(11, 11.1, 'GND', fontsize=7, ha='center', color='#555')

# Inverters
mosfet(12.5, 13, 'INV1p', 'PMOS 460n', '#283593')
mosfet(12.5, 12, 'INV1n', 'NMOS 920n', '#283593')
wire(12.5, 14, 12.5, 13.35)
wire(12.5, 12.65, 12.5, 12.35)
wire(12.5, 11.65, 12.5, 11.3)
ax.text(12.5, 11.1, 'GND', fontsize=7, ha='center', color='#555')
wire(11, 13, 12.5, 13)  # vmem to inv gate

node(13.2, 12.5, 'inv1', '#283593')
wire(12.5, 12.65, 13.2, 12.5)

mosfet(14, 13, 'INV2p', 'PMOS 460n', '#283593')
mosfet(14, 12, 'INV2n', 'NMOS 920n', '#283593')
wire(14, 14, 14, 13.35)
wire(14, 12.65, 14, 12.35)
wire(14, 11.65, 14, 11.3)
wire(13.2, 12.5, 14, 12.5)  # inv1 to inv2 gate

node(14.5, 12.5, 'vspike', 'red')

# M_reset
mosfet(11, 10.5, 'M_reset', 'W=920n', '#283593')
wire(9.5, 13, 9.5, 10.5)  # vmem down
wire(9.5, 10.5, 10.4, 10.5)
wire(11, 10.15, 11, 9.8)
ax.text(11, 9.6, 'GND', fontsize=7, ha='center', color='#555')
ax.text(11, 10.95, 'rst_del→', fontsize=6, color='#555', ha='center')

# Bexc (avalanche excitation)
bsource(10.5, 9, 'Bexc', '50nA × spike_det', '#E65100')
wire(10.5, 9.4, 10.5, 9.8)
ax.text(10.5, 8.5, '← from Lanza', fontsize=7, color='#E65100', fontweight='bold')

# ================================================================
# GPU FP16 MAC (right)
# ================================================================
# Vmac
r = FancyBboxPatch((16, 13), 2, 0.8, boxstyle="round,pad=0.05",
                    fc='#B2DFDB', ec='#00695C', lw=1.5)
ax.add_patch(r)
ax.text(17, 13.55, 'Vmac (GEMM)', fontsize=8, fontweight='bold', color='#00695C')
ax.text(17, 13.2, '0.3→1.2V triangle', fontsize=7, color='#555')
node(17, 12.7, 'mac_in', '#00695C')

# Bmac
bsource(19.5, 12.5, 'Bmac', 'V_in·(1−0.002·spike)', '#00695C')
wire(18, 12.5, 18.4, 12.5)
node(20.8, 12.5, 'mac_out', '#00695C')

# Rounding mode box
r2 = FancyBboxPatch((16, 11), 3.5, 1.2, boxstyle="round,pad=0.1",
                     fc='#FFECB3', ec='#F57F17', lw=1.5)
ax.add_patch(r2)
ax.text(17.75, 11.85, 's_setreg hwreg(MODE,0,8)', fontsize=8,
        fontweight='bold', color='#F57F17')
ax.text(17.75, 11.4, 'vspike>0.5 → truncation mode', fontsize=7, color='#555')

# vspike arrow to rounding mode
ax.annotate('', xy=(16, 11.5), xytext=(14.5, 12.5),
            arrowprops=dict(arrowstyle='->', color='red', lw=2.5))
ax.text(15, 12.2, 'spike!', fontsize=8, color='red', fontweight='bold')

# Synaptic channels
r3 = FancyBboxPatch((16, 9.5), 5.5, 1.2, boxstyle="round,pad=0.1",
                     fc='#DCEDC8', ec='#33691E', lw=1.2)
ax.add_patch(r3)
ax.text(18.75, 10.35, 'Bsyn1-4 + Bbias (spike-gated)', fontsize=8,
        fontweight='bold', color='#33691E')
ax.text(18.75, 9.85, '8-20nA each, gated off during vspike>0.3V', fontsize=7, color='#555')

# GPU -> gate feedback (big arrow)
ax.annotate('', xy=(1.5, 13.5), xytext=(20.8, 12.5),
            arrowprops=dict(arrowstyle='->', color='#E65100', lw=3,
                          connectionstyle='arc3,rad=0.35'))

# Feedback equation box
fb = FancyBboxPatch((3, 7.2), 18, 1.2, boxstyle="round,pad=0.15",
                     fc='#FFF9C4', ec='#F57F17', lw=2)
ax.add_patch(fb)
ax.text(12, 8.0, 'CONSTITUTIVE FEEDBACK:  Vgate = 0.35 + 0.1·mac_norm  →  '
        'BVpar = 3.5 − 1.5·Vgate  →  avalanche threshold shifts',
        ha='center', fontsize=9, fontweight='bold', color='#E65100')
ax.text(12, 7.5, 'GPU MAC arithmetic output directly controls NS-RAM avalanche breakdown voltage — no software intermediary',
        ha='center', fontsize=8, color='#555', style='italic')

plt.savefig('results/nsram_bridge_v2_transistor.png', dpi=150, bbox_inches='tight')
print("Saved: results/nsram_bridge_v2_transistor.png")
