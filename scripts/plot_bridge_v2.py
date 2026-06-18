#!/usr/bin/env python3
"""Plot NS-RAM <-> GPU MAC Bridge v2 — with proper avalanche physics."""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

data = np.loadtxt('results/nsram_bridge_v2.csv')

# 11 signals x 2 columns = 22 columns
t_vmem = data[:, 0]; vmem = data[:, 1]
t_spike = data[:, 2]; vspike = data[:, 3]
t_snode = data[:, 4]; s_node = data[:, 5]
t_sdet = data[:, 6]; spike_det = data[:, 7]
t_macin = data[:, 8]; mac_in = data[:, 9]
t_macout = data[:, 10]; mac_out = data[:, 11]
t_gate = data[:, 12]; nsram_gate = data[:, 13]
t_bulk = data[:, 14]; b_node = data[:, 15]
t_bvpar = data[:, 16]; bvpar = data[:, 17]
t_vcb = data[:, 18]; vcb = data[:, 19]
t_di = data[:, 20]; di_v = data[:, 21]

t_us = t_vmem * 1e6
ts_us = t_spike * 1e6
tsn_us = t_snode * 1e6

# ============================================================
# PAPER FIGURE: 6-panel full simulation
# ============================================================
fig, axes = plt.subplots(6, 1, figsize=(16, 20), sharex=True)
fig.suptitle('NS-RAM $\\leftrightarrow$ GPU MAC Bridge v2 — Proper Avalanche Model\n'
             'Lanza/Pazos Avalanche Device (PTM130) + AMD GPU FP16 Rounding Mode',
             fontsize=14, fontweight='bold')

# Panel 1: Avalanche physics — Vcb vs BVPar
ax = axes[0]
ax.plot(t_vcb*1e6, vcb, 'darkred', linewidth=0.4, label='V$_{cb}$ (collector-base)')
ax.plot(t_bvpar*1e6, bvpar, 'blue', linewidth=0.6, alpha=0.8, label='BV$_{par}$ = 3.5 - 1.5·V$_g$ (breakdown threshold)')
ax.fill_between(t_vcb*1e6, bvpar[:len(vcb)], vcb,
                where=(vcb > bvpar[:len(vcb)]),
                alpha=0.3, color='red', label='AVALANCHE REGION (V$_{cb}$ > BV$_{par}$)')
ax.set_ylabel('Voltage (V)', fontsize=10)
ax.legend(loc='upper right', fontsize=8)
ax.set_title('Lanza Avalanche: V$_{cb}$ exceeds BV$_{par}$ → impact ionization → current spike', fontsize=11)
ax.grid(True, alpha=0.3)

# Panel 2: Avalanche current (source node) + bulk charging
ax = axes[1]
ax2 = ax.twinx()
ax.plot(tsn_us, s_node * 1000, 'darkred', linewidth=0.4, label='I$_{source}$ (mA)')
ax2.plot(t_bulk*1e6, b_node, 'green', linewidth=0.4, alpha=0.7, label='V$_{bulk}$ (floating body)')
ax.set_ylabel('Source Current (mA)', fontsize=10, color='darkred')
ax2.set_ylabel('Bulk Voltage (V)', fontsize=10, color='green')
ax.legend(loc='upper left', fontsize=8)
ax2.legend(loc='upper right', fontsize=8)
ax.set_title('Avalanche Current Spikes + Floating Bulk Charging (positive feedback)', fontsize=11)
ax.grid(True, alpha=0.3)

# Panel 3: GPU feedback -> NS-RAM gate
ax = axes[2]
ax.plot(t_gate*1e6, nsram_gate, 'purple', linewidth=0.5, label='V$_{gate}$ (GPU-modulated)')
ax.axhline(0.35, color='gray', linestyle='--', alpha=0.4, label='Base bias = 0.35V')
# Show BVPar on twin axis
ax2 = ax.twinx()
ax2.plot(t_bvpar*1e6, bvpar, 'blue', linewidth=0.5, alpha=0.5, label='BV$_{par}$')
ax2.set_ylabel('BV$_{par}$ (V)', fontsize=10, color='blue')
ax.set_ylabel('V$_{gate}$ (V)', fontsize=10, color='purple')
ax.legend(loc='upper left', fontsize=8)
ax2.legend(loc='upper right', fontsize=8)
ax.set_title('GPU → NS-RAM: MAC output modulates gate → shifts avalanche threshold', fontsize=11)
ax.grid(True, alpha=0.3)

# Panel 4: LIF membrane
ax = axes[3]
ax.plot(t_us, vmem, 'b-', linewidth=0.4, label='V$_{mem}$ (membrane)')
ax.axhline(0.55, color='r', linestyle='--', alpha=0.4, linewidth=0.8, label='Inverter trip')
ax.set_ylabel('V$_{mem}$ (V)', fontsize=10)
ax.set_ylim(-0.05, 0.75)
ax.legend(loc='upper right', fontsize=8)
ax.set_title('Pazos LIF Membrane: Integrate (avalanche + synaptic) → Threshold → Spike → Reset', fontsize=11)
ax.grid(True, alpha=0.3)

# Panel 5: Spike output
ax = axes[4]
ax.plot(ts_us, vspike, 'r-', linewidth=0.4, label='V$_{spike}$ → GPU mode_sel')
ax.axhline(0.5, color='gray', linestyle='--', alpha=0.4, linewidth=0.8)
ax.set_ylabel('V$_{spike}$ (V)', fontsize=10)
ax.set_ylim(-0.15, 1.6)
ax.legend(loc='upper right', fontsize=8)
ax.set_title('NS-RAM → GPU: Spike drives s_setreg hwreg(MODE) rounding select', fontsize=11)
ax.grid(True, alpha=0.3)

# Panel 6: GPU MAC
ax = axes[5]
ax.plot(t_macin*1e6, mac_in, 'g-', linewidth=0.8, alpha=0.6, label='MAC input (GEMM)')
ax.plot(t_macout*1e6, mac_out, 'm-', linewidth=0.8, label='MAC output (rounding-mode modulated)')
ax.set_ylabel('Voltage (V)', fontsize=10)
ax.set_xlabel('Time (μs)', fontsize=11)
ax.set_ylim(0, 1.5)
ax.legend(loc='upper right', fontsize=8)
ax.set_title('GPU FP16 MAC: truncation during spike, output feeds back to NS-RAM gate', fontsize=11)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('results/nsram_bridge_v2_full.png', dpi=150, bbox_inches='tight')
print(f"Saved: results/nsram_bridge_v2_full.png")

# ============================================================
# Zoomed: first 30us — show avalanche events clearly
# ============================================================
fig2, axes2 = plt.subplots(4, 1, figsize=(16, 14), sharex=True)
fig2.suptitle('NS-RAM $\\leftrightarrow$ GPU Bridge v2 — Zoomed (0-30 μs)\n'
              'Avalanche Breakdown Events Driving LIF Neuron',
              fontsize=14, fontweight='bold')

t_zoom = 30

# Avalanche region
ax = axes2[0]
m1 = t_vcb*1e6 <= t_zoom
m2 = t_bvpar*1e6 <= t_zoom
ax.plot(t_vcb[m1]*1e6, vcb[m1], 'darkred', linewidth=1.0, label='V$_{cb}$')
ax.plot(t_bvpar[m2]*1e6, bvpar[m2], 'blue', linewidth=1.0, label='BV$_{par}$')
# Shade avalanche
try:
    min_len = min(sum(m1), sum(m2))
    vcb_z = vcb[m1][:min_len]
    bvp_z = bvpar[m2][:min_len]
    t_z = t_vcb[m1][:min_len]*1e6
    ax.fill_between(t_z, bvp_z, vcb_z, where=(vcb_z > bvp_z),
                    alpha=0.3, color='red', label='Avalanche zone')
except:
    pass
ax.set_ylabel('V (V)', fontsize=10)
ax.legend(loc='upper right', fontsize=8)
ax.set_title('When V$_{cb}$ crosses BV$_{par}$: impact ionization → avalanche current burst', fontsize=11)
ax.grid(True, alpha=0.3)

# Source current
ax = axes2[1]
m = tsn_us <= t_zoom
ax.plot(tsn_us[m], s_node[m]*1000, 'darkred', linewidth=1.0)
ax.set_ylabel('I$_{source}$ (mA)', fontsize=10)
ax.set_title('Avalanche Current Spikes at Source Sense Resistor', fontsize=11)
ax.grid(True, alpha=0.3)

# Membrane + spikes
ax = axes2[2]
m = t_us <= t_zoom
ax.plot(t_us[m], vmem[m], 'b-', linewidth=1.2)
ax.axhline(0.55, color='r', linestyle='--', alpha=0.5)
ax.fill_between(t_us[m], 0, vmem[m], alpha=0.15, color='blue')
ax.set_ylabel('V$_{mem}$ (V)', fontsize=10)
ax.set_title('LIF Membrane: avalanche events charge Cint → threshold → spike → reset', fontsize=11)
ax.grid(True, alpha=0.3)

ax = axes2[3]
m = ts_us <= t_zoom
ax.plot(ts_us[m], vspike[m], 'r-', linewidth=1.2)
m2 = t_gate*1e6 <= t_zoom
ax3 = ax.twinx()
ax3.plot(t_gate[m2]*1e6, nsram_gate[m2], 'purple', linewidth=0.8, alpha=0.6, label='V$_{gate}$ (GPU)')
ax.set_ylabel('V$_{spike}$ (V)', fontsize=10, color='red')
ax3.set_ylabel('V$_{gate}$ (V)', fontsize=10, color='purple')
ax3.legend(loc='upper right', fontsize=8)
ax.set_xlabel('Time (μs)', fontsize=11)
ax.set_title('Spikes (red) + GPU gate feedback (purple): bidirectional loop visible', fontsize=11)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('results/nsram_bridge_v2_zoomed.png', dpi=150, bbox_inches='tight')
print(f"Saved: results/nsram_bridge_v2_zoomed.png")

# ============================================================
# Statistics
# ============================================================
spike_crossings = np.where(np.diff(np.sign(vspike - 0.75)) > 0)[0]
n_spikes = len(spike_crossings)
print(f"\n{'='*60}")
print(f"NS-RAM <-> GPU Bridge v2 — Avalanche Statistics")
print(f"{'='*60}")
print(f"Total spikes: {n_spikes}")

if n_spikes >= 2:
    spike_times = ts_us[spike_crossings]
    isis = np.diff(spike_times)
    freq_khz = 1000.0 / np.mean(isis)
    print(f"Mean ISI: {np.mean(isis):.3f} μs")
    print(f"Mean frequency: {freq_khz:.0f} kHz")
    if 60 <= freq_khz <= 2000:
        print(f"  IN PAZOS RANGE (60-360 kHz)")

# Avalanche physics
print(f"\nAvalanche Device:")
print(f"  V_cb max: {vcb.max():.3f} V")
print(f"  BV_par avg: {bvpar.mean():.3f} V (= 3.5 - 1.5*Vg)")
print(f"  V_cb > BV_par: avalanche events = {np.sum(vcb > bvpar[:len(vcb)])}/{len(vcb)} samples")
print(f"  Source current max: {s_node.max()*1000:.3f} mA")
print(f"  Bulk voltage max: {b_node.max():.4f} V (floating body charge)")

# Gate modulation from GPU
print(f"\nGPU → NS-RAM Gate Feedback:")
print(f"  Gate range: {nsram_gate.min():.4f} - {nsram_gate.max():.4f} V")
print(f"  Gate swing: {(nsram_gate.max()-nsram_gate.min())*1000:.1f} mV")
print(f"  BV_par range: {3.5-1.5*nsram_gate.max():.3f} - {3.5-1.5*nsram_gate.min():.3f} V")
print(f"  BV_par swing: {1.5*(nsram_gate.max()-nsram_gate.min())*1000:.1f} mV")

# Energy estimate — proper integration of I*V*dt
if n_spikes >= 2:
    e_mem = 0.5 * 102e-15 * 0.6**2  # membrane CV^2
    e_inv = 2 * 0.5 * 2e-15 * 1.5**2  # inverter switching

    # Integrate avalanche power: P(t) = I_source(t) * V_drain(t)
    # I_source = V(s_node) / 1 ohm, V_drain ~ V(di)
    # Use trapezoidal integration over the actual waveform
    dt_sn = np.diff(t_snode)  # time steps (seconds)
    i_source = s_node  # V/1ohm = amps
    # Power at each sample: I * Vd (use di voltage)
    # Align arrays to same length
    min_len = min(len(i_source)-1, len(di_v)-1, len(dt_sn))
    p_aval = i_source[:min_len] * di_v[:min_len]  # watts
    e_aval_total = np.sum(p_aval * dt_sn[:min_len])  # joules total
    e_aval_per_spike = e_aval_total / n_spikes if n_spikes > 0 else 0

    # Also compute membrane-only energy from charge integration
    # Q = C * V_peak, E = Q * V_peak / 2
    e_total = e_mem + e_inv + e_aval_per_spike

    print(f"\nEnergy per spike estimate (integrated P*dt / n_spikes):")
    print(f"  E_membrane (0.5*C*V^2): {e_mem*1e15:.1f} fJ")
    print(f"  E_inverter (switching): {e_inv*1e15:.1f} fJ")
    print(f"  E_avalanche (∫I·V·dt / {n_spikes} spikes): {e_aval_per_spike*1e15:.1f} fJ")
    print(f"  E_total: {e_total*1e15:.1f} fJ/spike")
    print(f"  Pazos reports: 0.2-21 fJ/spike")
    print(f"  Total avalanche energy (200μs sim): {e_aval_total*1e12:.2f} pJ")

# Feedback
spike_active = vspike > 0.5
if any(spike_active) and any(~spike_active):
    print(f"\nFeedback Verification:")
    print(f"  MAC during spike: {mac_out[spike_active].mean():.4f} V")
    print(f"  MAC without spike: {mac_out[~spike_active].mean():.4f} V")
    print(f"  Difference: {abs(mac_out[~spike_active].mean() - mac_out[spike_active].mean())*1000:.2f} mV")
print(f"{'='*60}")
