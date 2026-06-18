#!/usr/bin/env python3
"""
z2233_partial_reset_sweep.py — Find where partial reset matters
================================================================
Quick sweep: THRESHOLD × LEAK × Vg, all set at runtime (no rebuild needed).
Measure spike rate + vmem residual after spike.
Goal: find regime where neurons spike at moderate rates (10-100 spk/s)
so the partial reset (vmem = vmem - threshold) creates temporal memory.

NOTE: spike counters RESET after each telemetry read (nb_count_reset in RTL).
So each packet's spike_counts = spikes since LAST read. Must SUM across packets.
"""
import sys, time, json
import numpy as np
sys.path.insert(0, "scripts")
from fpga_host_eth import FPGAEthBridge

N_NEURONS = 128
MEASURE_TIME = 0.5  # seconds per point
SETTLE_TIME  = 0.15 # seconds to let neurons settle after param change

THRESH_VALUES = [0.20, 0.30, 0.40, 0.50, 0.60, 0.75]
LEAK_VALUES   = [0x0002, 0x0004, 0x0008, 0x0011, 0x0040, 0x0100]
LEAK_NAMES    = ["0x0002", "0x0004", "0x0008", "0x0011", "0x0040", "0x0100"]
VG_VALUES     = [0.50, 0.55, 0.58, 0.62, 0.68, 0.78]

def drain(fpga, n=50):
    for _ in range(n):
        try: fpga.recv_auto_telemetry(timeout=0.003)
        except: break

def main():
    fpga = FPGAEthBridge()
    if not fpga.connect():
        print("FAIL: Cannot connect to FPGA")
        return

    fpga.set_kill(False)
    fpga.enable_auto_telemetry(2000)  # 2kHz = 0.5ms interval
    time.sleep(0.3)
    drain(fpga, 200)

    results = {}
    total = len(THRESH_VALUES) * len(LEAK_VALUES) * len(VG_VALUES)
    done = 0

    print(f"{'THRESH':>6} {'LEAK':>8} {'Vg':>6} | {'Rate/n':>8} {'RateBank':>8} {'vmem_m':>8} {'vmem_sd':>8} {'residual':>8} {'spk_n':>6} {'pkts':>5}")
    print("-" * 96)

    for thresh in THRESH_VALUES:
        fpga.set_threshold(thresh)
        time.sleep(0.02)

        for leak_val, leak_name in zip(LEAK_VALUES, LEAK_NAMES):
            fpga.set_leak_cond(leak_val)
            time.sleep(0.02)

            for vg in VG_VALUES:
                done += 1

                # Set Vg (no kill/reset — let neurons keep state for faster sweeps)
                fpga.set_mac_signal(0.0)
                fpga.set_vg_batch(0, [vg] * 64)
                fpga.set_vg_batch(64, [vg] * 64)

                # Settle and drain stale packets
                time.sleep(SETTLE_TIME)
                drain(fpga, 100)

                # Accumulate spike counts and vmem across all packets in window
                total_spikes = np.zeros(N_NEURONS, dtype=np.int64)
                vmem_samples = []
                n_pkts = 0

                t_start = time.perf_counter()
                while time.perf_counter() - t_start < MEASURE_TIME:
                    try:
                        pkt = fpga.recv_auto_telemetry(timeout=0.01)
                        if pkt is not None:
                            # Each packet's spike_counts = spikes since LAST read
                            total_spikes += pkt['spike_counts'].astype(np.int64)
                            vmem_samples.append(pkt['vmem'].astype(np.float32))
                            n_pkts += 1
                    except:
                        break

                if n_pkts < 3:
                    print(f"{thresh:>6.2f} {leak_name:>8} {vg:>6.2f} | {'TIMEOUT':>8} (pkts={n_pkts}) [{done}/{total}]")
                    continue

                elapsed = time.perf_counter() - t_start

                # Rate per neuron (spk/s)
                rate_per_neuron = (total_spikes / elapsed).mean()
                rate_bank = total_spikes.sum() / elapsed

                vmem_all = np.array(vmem_samples)  # shape: (n_pkts, 128)
                vmem_mean = vmem_all.mean()
                vmem_std = vmem_all.std()

                # Residual: mean vmem of neurons that spiked (should be > 0 with partial reset)
                spiking_mask = total_spikes > 0
                n_spiking = spiking_mask.sum()
                if n_spiking > 0:
                    # Average vmem of spiking neurons across all snapshots
                    residual = vmem_all[:, spiking_mask].mean()
                else:
                    residual = vmem_mean

                key = f"th={thresh:.2f}_leak={leak_name}_vg={vg:.2f}"
                results[key] = {
                    'threshold': thresh, 'leak': leak_name, 'vg': vg,
                    'rate_per_neuron': float(rate_per_neuron),
                    'rate_bank': float(rate_bank),
                    'vmem_mean': float(vmem_mean),
                    'vmem_std': float(vmem_std),
                    'residual': float(residual),
                    'n_spiking': int(n_spiking),
                    'n_pkts': n_pkts,
                    'total_spikes_sum': int(total_spikes.sum()),
                }

                tag = " ***" if 5 < rate_per_neuron < 200 else ""
                print(f"{thresh:>6.2f} {leak_name:>8} {vg:>6.2f} | {rate_per_neuron:>8.1f} {rate_bank:>8.0f} {vmem_mean:>8.4f} {vmem_std:>8.4f} {residual:>8.4f} {n_spiking:>6d} {n_pkts:>5d}{tag} [{done}/{total}]")

    # Best combos for memory
    print(f"\n{'='*96}")
    print("TOP 15 — moderate spike rate (5-200 spk/s/neuron) with highest vmem variance:")
    moderate = [(k, v) for k, v in results.items() if 5 < v['rate_per_neuron'] < 200]
    moderate.sort(key=lambda x: x[1]['vmem_std'], reverse=True)
    for i, (k, v) in enumerate(moderate[:15]):
        print(f"  {i+1}. {k}: rate/n={v['rate_per_neuron']:.1f}, vmem={v['vmem_mean']:.4f}±{v['vmem_std']:.4f}, residual={v['residual']:.4f}, spiking={v['n_spiking']}")

    if not moderate:
        print("  (none in 5-200 range — showing top 15 by rate)")
        all_sorted = [(k, v) for k, v in results.items() if v.get('rate_per_neuron', 0) > 0]
        all_sorted.sort(key=lambda x: x[1]['rate_per_neuron'], reverse=True)
        for i, (k, v) in enumerate(all_sorted[:15]):
            print(f"  {i+1}. {k}: rate/n={v['rate_per_neuron']:.1f}, vmem={v['vmem_mean']:.4f}±{v['vmem_std']:.4f}, residual={v['residual']:.4f}")

    # Summary stats
    rates = [v['rate_per_neuron'] for v in results.values()]
    if rates:
        print(f"\nRate stats: min={min(rates):.1f}, max={max(rates):.1f}, median={np.median(rates):.1f} spk/s/neuron")

    with open("results/z2233_partial_reset_sweep.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to results/z2233_partial_reset_sweep.json ({len(results)} points)")
    fpga.close()

if __name__ == "__main__":
    main()
