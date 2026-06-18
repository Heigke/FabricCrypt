#!/usr/bin/env python3
"""Plot NS-RAM + GPU MAC hybrid v5 SPICE simulation results."""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Load CSV: wrdata format is pairs of (time, value) columns
# Columns: vmem, vspike, mac_in, mac_out, inv1, rst_del
data = np.loadtxt('results/nsram_gpu_v5.csv')

# Each signal has 2 columns (time, value) — 6 signals = 12 columns
t_vmem = data[:, 0]
vmem = data[:, 1]
t_spike = data[:, 2]
vspike = data[:, 3]
t_macin = data[:, 4]
mac_in = data[:, 5]
t_macout = data[:, 6]
mac_out = data[:, 7]
t_inv1 = data[:, 8]
inv1 = data[:, 9]
t_rst = data[:, 10]
rst_del = data[:, 11]

# Convert to microseconds
t_vmem_us = t_vmem * 1e6
t_spike_us = t_spike * 1e6
t_macin_us = t_macin * 1e6
t_macout_us = t_macout * 1e6

fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)
fig.suptitle('NS-RAM + GPU MAC Closed-Loop Hybrid (v5)\n'
             'Pazos/Lanza NS-RAM LIF + AMD GPU FP16 Rounding Mode',
             fontsize=14, fontweight='bold')

# Panel 1: Membrane voltage
ax = axes[0]
ax.plot(t_vmem_us, vmem, 'b-', linewidth=0.5, label='V_mem')
ax.axhline(0.55, color='r', linestyle='--', alpha=0.5, label='Inverter trip (~0.55V)')
ax.set_ylabel('V_mem (V)')
ax.set_ylim(-0.05, 0.8)
ax.legend(loc='upper right', fontsize=8)
ax.set_title('NS-RAM Membrane Voltage (Cint=102fF)', fontsize=10)
ax.grid(True, alpha=0.3)

# Panel 2: Spike output
ax = axes[1]
ax.plot(t_spike_us, vspike, 'r-', linewidth=0.5, label='V_spike')
ax.axhline(0.75, color='gray', linestyle='--', alpha=0.5, label='Mode switch threshold')
ax.set_ylabel('V_spike (V)')
ax.set_ylim(-0.1, 1.6)
ax.legend(loc='upper right', fontsize=8)
ax.set_title('NS-RAM Spike Output (feeds GPU mode_sel)', fontsize=10)
ax.grid(True, alpha=0.3)

# Panel 3: MAC input/output
ax = axes[2]
ax.plot(t_macin_us, mac_in, 'g-', linewidth=0.8, alpha=0.7, label='MAC input (GEMM)')
ax.plot(t_macout_us, mac_out, 'm-', linewidth=0.8, label='MAC output (rounding)')
ax.set_ylabel('Voltage (V)')
ax.set_ylim(0, 1.5)
ax.legend(loc='upper right', fontsize=8)
ax.set_title('GPU FP16 MAC (rounding mode modulated by NS-RAM spike)', fontsize=10)
ax.grid(True, alpha=0.3)

# Panel 4: Reset delay and internal
ax = axes[3]
ax.plot(t_rst * 1e6, rst_del, 'orange', linewidth=0.5, label='rst_delayed (RC)')
ax.plot(t_inv1 * 1e6, inv1, 'purple', linewidth=0.5, alpha=0.6, label='inv1 (internal)')
ax.set_ylabel('Voltage (V)')
ax.set_xlabel('Time (us)')
ax.legend(loc='upper right', fontsize=8)
ax.set_title('Reset Delay & Internal Inverter', fontsize=10)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('results/nsram_gpu_v5_waveforms.png', dpi=150, bbox_inches='tight')
print(f"Saved: results/nsram_gpu_v5_waveforms.png")

# Also make a zoomed version showing a few spike cycles
fig2, axes2 = plt.subplots(3, 1, figsize=(14, 8), sharex=True)
fig2.suptitle('NS-RAM + GPU Hybrid v5 — Zoomed (0-10 us)\n'
              'Showing integrate-and-fire cycles with GPU feedback',
              fontsize=13, fontweight='bold')

t_zoom = 10  # microseconds

ax = axes2[0]
mask = t_vmem_us <= t_zoom
ax.plot(t_vmem_us[mask], vmem[mask], 'b-', linewidth=1.0)
ax.axhline(0.55, color='r', linestyle='--', alpha=0.5, label='Trip point')
ax.set_ylabel('V_mem (V)')
ax.legend(loc='upper right', fontsize=8)
ax.set_title('Membrane integration → threshold → spike → reset', fontsize=10)
ax.grid(True, alpha=0.3)

ax = axes2[1]
mask = t_spike_us <= t_zoom
ax.plot(t_spike_us[mask], vspike[mask], 'r-', linewidth=1.0)
ax.axhline(0.75, color='gray', linestyle='--', alpha=0.5)
ax.set_ylabel('V_spike (V)')
ax.set_title('Spike pulses → each one switches GPU rounding mode', fontsize=10)
ax.grid(True, alpha=0.3)

ax = axes2[2]
mask_in = t_macin_us <= t_zoom
mask_out = t_macout_us <= t_zoom
ax.plot(t_macin_us[mask_in], mac_in[mask_in], 'g-', linewidth=1.0, alpha=0.7, label='MAC in')
ax.plot(t_macout_us[mask_out], mac_out[mask_out], 'm-', linewidth=1.0, label='MAC out')
ax.set_ylabel('V (V)')
ax.set_xlabel('Time (us)')
ax.legend(loc='upper right', fontsize=8)
ax.set_title('GPU MAC: rounding mode changes propagate back as synaptic current', fontsize=10)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('results/nsram_gpu_v5_zoomed.png', dpi=150, bbox_inches='tight')
print(f"Saved: results/nsram_gpu_v5_zoomed.png")

# Print stats
spike_crossings = np.where(np.diff(np.sign(vspike - 0.75)) > 0)[0]
n_spikes = len(spike_crossings)
print(f"\nSpike Statistics:")
print(f"  Total spikes (0.75V crossings): {n_spikes}")
if n_spikes >= 2:
    spike_times = t_spike_us[spike_crossings]
    isis = np.diff(spike_times)
    print(f"  Mean ISI: {np.mean(isis):.3f} us")
    print(f"  Min ISI: {np.min(isis):.3f} us")
    print(f"  Max ISI: {np.max(isis):.3f} us")
    print(f"  Mean frequency: {1.0/np.mean(isis):.1f} MHz = {1000.0/np.mean(isis):.0f} kHz")
    print(f"  First spike at: {spike_times[0]:.3f} us")
    print(f"  Last spike at: {spike_times[-1]:.3f} us")
print(f"\nVmem range: {vmem.min():.4f} - {vmem.max():.4f} V")
print(f"Spike range: {vspike.min():.4f} - {vspike.max():.4f} V")
print(f"MAC out avg: {mac_out.mean():.4f} V")
