#!/usr/bin/env python3
"""Generate circuit schematic for NS-RAM <-> GPU MAC Bridge v2."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

fig, ax = plt.subplots(1, 1, figsize=(22, 14))
ax.set_xlim(0, 22)
ax.set_ylim(0, 14)
ax.set_aspect('equal')
ax.axis('off')

# Title
ax.text(11, 13.5, 'NS-RAM ↔ GPU MAC Bridge v2 — Circuit Schematic',
        ha='center', va='center', fontsize=16, fontweight='bold')
ax.text(11, 13.0, 'Lanza/Pazos Avalanche Device (PTM130) + AMD GPU FP16 Rounding Mode',
        ha='center', va='center', fontsize=11, color='gray')

# ================================================================
# LEFT SIDE: LANZA NS-RAM DEVICE
# ================================================================
# Box for Lanza device
lanza_box = FancyBboxPatch((0.5, 5.5), 7.5, 6.5, boxstyle="round,pad=0.2",
                            facecolor='#FFF3E0', edgecolor='#E65100', linewidth=2.5)
ax.add_patch(lanza_box)
ax.text(4.25, 11.6, 'LANZA NS-RAM DEVICE', ha='center', fontsize=12,
        fontweight='bold', color='#E65100')
ax.text(4.25, 11.2, '(Zenodo 10.5281/zenodo.13843362)', ha='center',
        fontsize=8, color='#E65100', style='italic')

# M2: Main MOSFET
m2_box = FancyBboxPatch((1.2, 8.8), 2.4, 1.5, boxstyle="round,pad=0.1",
                          facecolor='#E3F2FD', edgecolor='#1565C0', linewidth=1.5)
ax.add_patch(m2_box)
ax.text(2.4, 9.85, 'M2 (PTM130)', ha='center', fontsize=9, fontweight='bold', color='#1565C0')
ax.text(2.4, 9.45, 'NMOS W=0.5μ L=250n', ha='center', fontsize=7, color='#1565C0')
ax.text(2.4, 9.1, 'G=nsram_gate  D=di  S=s_node', ha='center', fontsize=6.5, color='#555')

# Q1: Avalanche BJT
q1_box = FancyBboxPatch((4.2, 8.8), 2.4, 1.5, boxstyle="round,pad=0.1",
                          facecolor='#FCE4EC', edgecolor='#C62828', linewidth=1.5)
ax.add_patch(q1_box)
ax.text(5.4, 9.85, 'Q1 (avalBJT)', ha='center', fontsize=9, fontweight='bold', color='#C62828')
ax.text(5.4, 9.45, 'NPN parasitic', ha='center', fontsize=7, color='#C62828')
ax.text(5.4, 9.1, 'C=di  B=b_node  E=s_node', ha='center', fontsize=6.5, color='#555')

# D1, D2: Zener diodes
d_box = FancyBboxPatch((1.2, 7.2), 2.4, 1.2, boxstyle="round,pad=0.1",
                         facecolor='#F3E5F5', edgecolor='#6A1B9A', linewidth=1.2)
ax.add_patch(d_box)
ax.text(2.4, 7.95, 'D1 (Zener)', ha='center', fontsize=8, fontweight='bold', color='#6A1B9A')
ax.text(2.4, 7.5, 'gate↔bulk breakdown', ha='center', fontsize=7, color='#6A1B9A')

d2_box = FancyBboxPatch((4.2, 7.2), 2.4, 1.2, boxstyle="round,pad=0.1",
                          facecolor='#F3E5F5', edgecolor='#6A1B9A', linewidth=1.2)
ax.add_patch(d2_box)
ax.text(5.4, 7.95, 'D2 (Zener)', ha='center', fontsize=8, fontweight='bold', color='#6A1B9A')
ax.text(5.4, 7.5, 'base-collector', ha='center', fontsize=7, color='#6A1B9A')

# M1: Bulk access
m1_box = FancyBboxPatch((1.2, 6.0), 2.4, 0.9, boxstyle="round,pad=0.1",
                          facecolor='#E8F5E9', edgecolor='#2E7D32', linewidth=1.2)
ax.add_patch(m1_box)
ax.text(2.4, 6.6, 'M1 (BSS145)', ha='center', fontsize=8, fontweight='bold', color='#2E7D32')
ax.text(2.4, 6.25, 'Bulk access  Vgb=2.5V', ha='center', fontsize=7, color='#2E7D32')

# Avalanche B-source (THE FIX)
aval_box = FancyBboxPatch((4.2, 5.8), 3.5, 1.2, boxstyle="round,pad=0.1",
                            facecolor='#FFCDD2', edgecolor='#B71C1C', linewidth=2)
ax.add_patch(aval_box)
ax.text(5.95, 6.65, 'Baval (AVALANCHE)', ha='center', fontsize=9, fontweight='bold', color='#B71C1C')
ax.text(5.95, 6.25, 'I = I₀·exp((Vcb−BVpar)/50mV)', ha='center', fontsize=7.5, color='#B71C1C')
ax.text(5.95, 5.95, 'BVpar = 3.5 − 1.5·V(gate)  ← GPU!', ha='center',
        fontsize=7.5, fontweight='bold', color='#B71C1C')

# Drive
ax.text(4.25, 10.85, 'Vpulse: 0→3.5V @ 100kHz', ha='center', fontsize=8, color='#555')

# ================================================================
# MIDDLE: PAZOS LIF NEURON
# ================================================================
lif_box = FancyBboxPatch((8.5, 5.5), 5, 6.5, boxstyle="round,pad=0.2",
                           facecolor='#E8EAF6', edgecolor='#283593', linewidth=2.5)
ax.add_patch(lif_box)
ax.text(11, 11.6, 'PAZOS LIF NEURON', ha='center', fontsize=12,
        fontweight='bold', color='#283593')

# Cint
cint_box = FancyBboxPatch((9.0, 9.8), 2.0, 1.0, boxstyle="round,pad=0.1",
                            facecolor='#C5CAE9', edgecolor='#283593', linewidth=1.2)
ax.add_patch(cint_box)
ax.text(10.0, 10.45, 'Cint = 102fF', ha='center', fontsize=9, fontweight='bold', color='#283593')
ax.text(10.0, 10.1, 'vmem (membrane)', ha='center', fontsize=7, color='#283593')

# Inverters
inv_box = FancyBboxPatch((11.5, 9.8), 1.7, 1.0, boxstyle="round,pad=0.1",
                           facecolor='#C5CAE9', edgecolor='#283593', linewidth=1.2)
ax.add_patch(inv_box)
ax.text(12.35, 10.45, '2× Inverter', ha='center', fontsize=8, fontweight='bold', color='#283593')
ax.text(12.35, 10.1, 'W=460n/920n', ha='center', fontsize=7, color='#283593')

# Leak
leak_box = FancyBboxPatch((9.0, 8.6), 2.0, 0.8, boxstyle="round,pad=0.1",
                            facecolor='#C5CAE9', edgecolor='#283593', linewidth=1)
ax.add_patch(leak_box)
ax.text(10.0, 9.1, 'M_leak (Vlk=0.2V)', ha='center', fontsize=7.5, color='#283593')

# Reset
rst_box = FancyBboxPatch((11.5, 8.6), 1.7, 0.8, boxstyle="round,pad=0.1",
                           facecolor='#C5CAE9', edgecolor='#283593', linewidth=1)
ax.add_patch(rst_box)
ax.text(12.35, 9.1, 'M_reset (latch)', ha='center', fontsize=7.5, color='#283593')

# Synaptic channels
syn_box = FancyBboxPatch((9.0, 7.2), 4.2, 1.1, boxstyle="round,pad=0.1",
                           facecolor='#DCEDC8', edgecolor='#33691E', linewidth=1.2)
ax.add_patch(syn_box)
ax.text(11.1, 7.95, '4 Synaptic Channels (spike-gated)', ha='center',
        fontsize=8, fontweight='bold', color='#33691E')
ax.text(11.1, 7.55, 'Bsyn1-4: 8-20nA each, GPU wavefront model', ha='center',
        fontsize=7, color='#33691E')

# Excitation from avalanche
exc_box = FancyBboxPatch((9.0, 6.0), 4.2, 0.9, boxstyle="round,pad=0.1",
                           facecolor='#FFE0B2', edgecolor='#E65100', linewidth=1.2)
ax.add_patch(exc_box)
ax.text(11.1, 6.6, 'Bexc: 50nA × spike_det', ha='center', fontsize=8,
        fontweight='bold', color='#E65100')
ax.text(11.1, 6.25, 'Avalanche events → membrane current', ha='center',
        fontsize=7, color='#E65100')

# ================================================================
# RIGHT SIDE: GPU FP16 MAC
# ================================================================
gpu_box = FancyBboxPatch((14, 5.5), 7.5, 6.5, boxstyle="round,pad=0.2",
                           facecolor='#E0F7FA', edgecolor='#00695C', linewidth=2.5)
ax.add_patch(gpu_box)
ax.text(17.75, 11.6, 'AMD GPU FP16 MAC', ha='center', fontsize=12,
        fontweight='bold', color='#00695C')

# MAC block
mac_box = FancyBboxPatch((14.8, 9.5), 3.2, 1.5, boxstyle="round,pad=0.1",
                           facecolor='#B2DFDB', edgecolor='#00695C', linewidth=1.5)
ax.add_patch(mac_box)
ax.text(16.4, 10.6, 'FP16 GEMM MAC', ha='center', fontsize=10, fontweight='bold', color='#00695C')
ax.text(16.4, 10.2, 'Vmac: 0.3→1.2V triangle', ha='center', fontsize=7.5, color='#00695C')
ax.text(16.4, 9.85, '(accumulator model)', ha='center', fontsize=7, color='#555')

# Rounding mode
rnd_box = FancyBboxPatch((14.8, 7.8), 3.2, 1.4, boxstyle="round,pad=0.1",
                           facecolor='#FFECB3', edgecolor='#F57F17', linewidth=1.5)
ax.add_patch(rnd_box)
ax.text(16.4, 8.8, 's_setreg hwreg(MODE)', ha='center', fontsize=9,
        fontweight='bold', color='#F57F17')
ax.text(16.4, 8.4, 'Mode 0: nearest-even', ha='center', fontsize=7.5, color='#555')
ax.text(16.4, 8.05, 'Mode 3: toward-zero (trunc)', ha='center', fontsize=7.5, color='#555')

# MAC output
out_box = FancyBboxPatch((18.5, 8.5), 2.5, 1.5, boxstyle="round,pad=0.1",
                           facecolor='#B2DFDB', edgecolor='#00695C', linewidth=1.2)
ax.add_patch(out_box)
ax.text(19.75, 9.5, 'MAC Output', ha='center', fontsize=9, fontweight='bold', color='#00695C')
ax.text(19.75, 9.1, 'V_out = V_in ×', ha='center', fontsize=7.5, color='#00695C')
ax.text(19.75, 8.75, '(1 − 0.002·spike)', ha='center', fontsize=7.5, color='#00695C')

# Bias coupling
bias_box = FancyBboxPatch((18.5, 6.8), 2.5, 1.3, boxstyle="round,pad=0.1",
                            facecolor='#B2DFDB', edgecolor='#00695C', linewidth=1.2)
ax.add_patch(bias_box)
ax.text(19.75, 7.65, 'Bbias coupling', ha='center', fontsize=8, fontweight='bold', color='#00695C')
ax.text(19.75, 7.2, '(20n+30n·mac_out)', ha='center', fontsize=7, color='#00695C')

# ================================================================
# ARROWS — BIDIRECTIONAL COUPLING
# ================================================================

# Arrow 1: Lanza avalanche -> spike_det -> LIF excitation
ax.annotate('', xy=(9.0, 6.4), xytext=(7.7, 6.4),
            arrowprops=dict(arrowstyle='->', color='#E65100', lw=2.5))
ax.text(8.35, 6.7, 'avalanche\nspikes', ha='center', fontsize=7, color='#E65100', fontweight='bold')

# Arrow 2: LIF vspike -> GPU rounding mode
ax.annotate('', xy=(14.8, 8.5), xytext=(13.2, 10.3),
            arrowprops=dict(arrowstyle='->', color='red', lw=3,
                          connectionstyle='arc3,rad=-0.2'))
ax.text(13.6, 9.0, 'vspike →\nmode_sel', ha='center', fontsize=8,
        color='red', fontweight='bold')

# Arrow 3: GPU MAC output -> gate feedback (big curved arrow, the key path)
ax.annotate('', xy=(4.25, 5.3), xytext=(19.75, 6.8),
            arrowprops=dict(arrowstyle='->', color='#00695C', lw=3,
                          connectionstyle='arc3,rad=0.4'))
ax.text(12, 4.2, 'GPU → NS-RAM GATE FEEDBACK', ha='center', fontsize=10,
        fontweight='bold', color='#00695C')
ax.text(12, 3.7, 'V(gate) = 0.35 + 0.1·mac_norm  →  BVpar = 3.5 − 1.5·V(gate)',
        ha='center', fontsize=9, color='#00695C')
ax.text(12, 3.2, 'Higher MAC output → higher Vg → lower BVpar → easier avalanche → more spikes',
        ha='center', fontsize=8, color='#555')

# Arrow 4: Synaptic channels from GPU
ax.annotate('', xy=(13.2, 7.7), xytext=(14.8, 7.5),
            arrowprops=dict(arrowstyle='<-', color='#33691E', lw=2))

# Arrow 5: vspike back to Lanza (already through LIF)
ax.annotate('', xy=(8.5, 10.3), xytext=(7.7, 9.5),
            arrowprops=dict(arrowstyle='->', color='#283593', lw=2))
ax.text(7.6, 10.2, 'spike_det', ha='center', fontsize=7, color='#283593')

# ================================================================
# FEEDBACK LOOP LABEL
# ================================================================
loop_box = FancyBboxPatch((3.5, 0.8), 15, 2.0, boxstyle="round,pad=0.2",
                           facecolor='#FFF9C4', edgecolor='#F57F17', linewidth=2)
ax.add_patch(loop_box)
ax.text(11, 2.3, 'CONSTITUTIVE BIDIRECTIONAL COUPLING', ha='center',
        fontsize=12, fontweight='bold', color='#E65100')
ax.text(11, 1.7, 'NS-RAM spike → GPU truncation → less MAC output → lower gate → higher BVpar → fewer spikes (NEGATIVE)',
        ha='center', fontsize=8, color='#333')
ax.text(11, 1.2, 'No firmware, no software — direct analog physics coupling between avalanche breakdown and FP16 arithmetic',
        ha='center', fontsize=8, color='#555', style='italic')

# ================================================================
# ENERGY LABELS
# ================================================================
ax.text(4.25, 5.15, 'E_avalanche ≈ 23 fJ/spike', ha='center', fontsize=8,
        color='#B71C1C', fontweight='bold')
ax.text(11, 5.15, 'E_membrane = 18.4 fJ/spike', ha='center', fontsize=8,
        color='#283593', fontweight='bold')
ax.text(17.75, 5.15, 'E_MAC ≈ 25 pJ/op (1000:1 ratio)', ha='center', fontsize=8,
        color='#00695C', fontweight='bold')

plt.tight_layout()
plt.savefig('results/nsram_bridge_v2_schematic.png', dpi=150, bbox_inches='tight')
print("Saved: results/nsram_bridge_v2_schematic.png")
