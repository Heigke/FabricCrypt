#!/usr/bin/env python3
"""Plot NS-RAM <-> GPU MAC Bridge v1 simulation results."""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

data = np.loadtxt('results/nsram_bridge_v1.csv')

# 8 signals x 2 columns (time, value) = 16 columns
t_vmem = data[:, 0]; vmem = data[:, 1]
t_spike = data[:, 2]; vspike = data[:, 3]
t_snode = data[:, 4]; s_node = data[:, 5]
t_sdet = data[:, 6]; spike_det = data[:, 7]
t_macin = data[:, 8]; mac_in = data[:, 9]
t_macout = data[:, 10]; mac_out = data[:, 11]
t_gate = data[:, 12]; nsram_gate = data[:, 13]
t_bulk = data[:, 14]; b_node = data[:, 15]

t_us = t_vmem * 1e6
ts_us = t_spike * 1e6
tsn_us = t_snode * 1e6
tsd_us = t_sdet * 1e6
tmi_us = t_macin * 1e6
tmo_us = t_macout * 1e6
tg_us = t_gate * 1e6
tb_us = t_bulk * 1e6

# ============================================================
# Full simulation: 5 panels
# ============================================================
fig, axes = plt.subplots(5, 1, figsize=(16, 16), sharex=True)
fig.suptitle('NS-RAM $\\leftrightarrow$ GPU MAC Bridge v1\n'
             'Lanza Avalanche Device (PTM130) + AMD GPU FP16 Rounding Mode',
             fontsize=14, fontweight='bold')

# Panel 1: Lanza avalanche device — source current
ax = axes[0]
ax.plot(tsn_us, s_node * 1000, 'darkred', linewidth=0.4, label='I$_{source}$ (source node voltage ≈ current)')
ax.plot(tsd_us, spike_det, 'orange', linewidth=0.5, alpha=0.7, label='Spike detector output')
ax.set_ylabel('V / mA', fontsize=10)
ax.legend(loc='upper right', fontsize=8)
ax.set_title('Lanza NS-RAM: Avalanche Current Spikes (punch-through in floating bulk)', fontsize=11)
ax.grid(True, alpha=0.3)

# Panel 2: NS-RAM gate (GPU feedback path)
ax = axes[1]
ax.plot(tg_us, nsram_gate, 'purple', linewidth=0.5, label='V$_{gate}$ (GPU-modulated)')
ax.axhline(0.35, color='gray', linestyle='--', alpha=0.4, label='Bias = 0.35V')
ax.set_ylabel('V$_{gate}$ (V)', fontsize=10)
ax.legend(loc='upper right', fontsize=8)
ax.set_title('GPU → NS-RAM: MAC output modulates gate → controls BV$_{par}$ = 3.5 - 1.5·V$_g$', fontsize=11)
ax.grid(True, alpha=0.3)

# Panel 3: Pazos LIF membrane
ax = axes[2]
ax.plot(t_us, vmem, 'b-', linewidth=0.4, label='V$_{mem}$ (membrane)')
ax.axhline(0.55, color='r', linestyle='--', alpha=0.4, linewidth=0.8, label='Inverter trip')
ax.set_ylabel('V$_{mem}$ (V)', fontsize=10)
ax.set_ylim(-0.05, 0.75)
ax.legend(loc='upper right', fontsize=8)
ax.set_title('Pazos LIF Membrane: Integrate (avalanche + synaptic) → Threshold → Spike → Reset', fontsize=11)
ax.grid(True, alpha=0.3)

# Panel 4: Spike output → GPU mode select
ax = axes[3]
ax.plot(ts_us, vspike, 'r-', linewidth=0.4, label='V$_{spike}$ → GPU mode_sel')
ax.axhline(0.5, color='gray', linestyle='--', alpha=0.4, linewidth=0.8)
ax.set_ylabel('V$_{spike}$ (V)', fontsize=10)
ax.set_ylim(-0.15, 1.6)
ax.legend(loc='upper right', fontsize=8)
ax.set_title('NS-RAM → GPU: Spike directly drives s_setreg hwreg(MODE) rounding select', fontsize=11)
ax.grid(True, alpha=0.3)

# Panel 5: GPU MAC
ax = axes[4]
ax.plot(tmi_us, mac_in, 'g-', linewidth=0.8, alpha=0.6, label='MAC input (GEMM accumulator)')
ax.plot(tmo_us, mac_out, 'm-', linewidth=0.8, label='MAC output (rounding-mode modulated)')
ax.set_ylabel('Voltage (V)', fontsize=10)
ax.set_xlabel('Time (μs)', fontsize=11)
ax.set_ylim(0, 1.5)
ax.legend(loc='upper right', fontsize=8)
ax.set_title('GPU FP16 MAC: truncation during spike → output feeds back to NS-RAM gate', fontsize=11)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('results/nsram_bridge_v1_full.png', dpi=150, bbox_inches='tight')
print(f"Saved: results/nsram_bridge_v1_full.png")

# ============================================================
# Zoomed: first 30us
# ============================================================
fig2, axes2 = plt.subplots(5, 1, figsize=(16, 16), sharex=True)
fig2.suptitle('NS-RAM $\\leftrightarrow$ GPU Bridge v1 — Zoomed (0-30 μs)\n'
              'Bidirectional Constitutive Coupling',
              fontsize=14, fontweight='bold')

t_zoom = 30

ax = axes2[0]
m = tsn_us <= t_zoom
ax.plot(tsn_us[m], s_node[m] * 1000, 'darkred', linewidth=1.0)
m2 = tsd_us <= t_zoom
ax.plot(tsd_us[m2], spike_det[m2], 'orange', linewidth=0.8, alpha=0.7)
ax.set_ylabel('V / mA', fontsize=10)
ax.set_title('Lanza Avalanche Current Bursts', fontsize=11)
ax.grid(True, alpha=0.3)

ax = axes2[1]
m = tg_us <= t_zoom
ax.plot(tg_us[m], nsram_gate[m], 'purple', linewidth=1.0)
ax.axhline(0.35, color='gray', linestyle='--', alpha=0.4)
ax.set_ylabel('V$_{gate}$ (V)', fontsize=10)
ax.set_title('GPU Feedback → NS-RAM Gate', fontsize=11)
ax.grid(True, alpha=0.3)

ax = axes2[2]
m = t_us <= t_zoom
ax.plot(t_us[m], vmem[m], 'b-', linewidth=1.2)
ax.axhline(0.55, color='r', linestyle='--', alpha=0.5)
ax.fill_between(t_us[m], 0, vmem[m], alpha=0.15, color='blue')
ax.set_ylabel('V$_{mem}$ (V)', fontsize=10)
ax.set_title('LIF Membrane Integration', fontsize=11)
ax.grid(True, alpha=0.3)

ax = axes2[3]
m = ts_us <= t_zoom
ax.plot(ts_us[m], vspike[m], 'r-', linewidth=1.2)
ax.set_ylabel('V$_{spike}$ (V)', fontsize=10)
ax.set_title('Spike → GPU Mode Select', fontsize=11)
ax.grid(True, alpha=0.3)

ax = axes2[4]
mi = tmi_us <= t_zoom
mo = tmo_us <= t_zoom
ax.plot(tmi_us[mi], mac_in[mi], 'g-', linewidth=1.0, alpha=0.6, label='MAC in')
ax.plot(tmo_us[mo], mac_out[mo], 'm-', linewidth=1.0, label='MAC out')
ax.set_ylabel('V (V)', fontsize=10)
ax.set_xlabel('Time (μs)', fontsize=11)
ax.legend(loc='upper right', fontsize=8)
ax.set_title('GPU MAC Rounding Mode Modulation', fontsize=11)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('results/nsram_bridge_v1_zoomed.png', dpi=150, bbox_inches='tight')
print(f"Saved: results/nsram_bridge_v1_zoomed.png")

# ============================================================
# Statistics
# ============================================================
spike_crossings = np.where(np.diff(np.sign(vspike - 0.75)) > 0)[0]
n_spikes = len(spike_crossings)
print(f"\n{'='*60}")
print(f"NS-RAM <-> GPU Bridge v1 — Statistics")
print(f"{'='*60}")
print(f"Total spikes (0.75V crossings): {n_spikes}")
if n_spikes >= 2:
    spike_times = ts_us[spike_crossings]
    isis = np.diff(spike_times)
    print(f"Mean ISI: {np.mean(isis):.3f} μs")
    print(f"Std ISI:  {np.std(isis):.3f} μs")
    freq_khz = 1000.0 / np.mean(isis)
    print(f"Mean frequency: {freq_khz:.0f} kHz")
    if 60 <= freq_khz <= 2000:
        print(f"  FREQUENCY IN RANGE (Pazos: 60-360 kHz)")

# Avalanche stats
print(f"\nAvalanche device:")
print(f"  Source node max: {s_node.max()*1000:.3f} mA")
print(f"  Source node mean: {s_node.mean()*1000:.3f} mA")
print(f"  Bulk node max: {b_node.max():.4f} V")

# Gate modulation
print(f"\nGate modulation (GPU feedback):")
print(f"  Gate min: {nsram_gate.min():.4f} V")
print(f"  Gate max: {nsram_gate.max():.4f} V")
print(f"  Gate range: {(nsram_gate.max()-nsram_gate.min())*1000:.1f} mV")
print(f"  BVPar range: {3.5-1.5*nsram_gate.max():.3f} - {3.5-1.5*nsram_gate.min():.3f} V")

# Feedback check
spike_active = vspike > 0.5
if any(spike_active) and any(~spike_active):
    mac_during = mac_out[spike_active].mean()
    mac_without = mac_out[~spike_active].mean()
    print(f"\nFeedback check:")
    print(f"  MAC during spike: {mac_during:.4f} V (truncation mode)")
    print(f"  MAC without spike: {mac_without:.4f} V (nearest-even)")
    print(f"  Difference: {abs(mac_without - mac_during)*1000:.2f} mV")
print(f"{'='*60}")
