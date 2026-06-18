#!/usr/bin/env python3
"""
z2233b_extended_sweep.py — Extended sweep into lower-rate regime
================================================================
From z2233: all rates >148 spk/s. Need to find regime with 10-100 spk/s.
Strategy: higher threshold + slower integration (smaller DT_OVER_C).
Also sweep BASE_EXC lower to reduce excitation.

Spike counters reset on each telemetry read — accumulate across packets.
"""
import sys, time, json
import numpy as np
sys.path.insert(0, "scripts")
from fpga_host_eth import FPGAEthBridge

N_NEURONS = 128
MEASURE_TIME = 0.5

# Higher thresholds to slow down spiking
THRESH_VALUES = [0.75, 1.00, 1.50, 2.00, 3.00]
# Slower integration steps
DTC_VALUES    = [0x0200, 0x0100, 0x0080, 0x0040, 0x0020]
DTC_NAMES     = ["0x0200", "0x0100", "0x0080", "0x0040", "0x0020"]
# Lower excitation gains
BEXC_VALUES   = [0x0333, 0x0200, 0x0100, 0x0080, 0x0040]
BEXC_NAMES    = ["0x0333", "0x0200", "0x0100", "0x0080", "0x0040"]

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
    fpga.enable_auto_telemetry(2000)
    time.sleep(0.3)
    drain(fpga, 200)

    # Fixed params
    VG = 0.62
    LEAK = 0x0004  # slow leak (τ≈210ms)
    fpga.set_leak_cond(LEAK)
    fpga.set_mac_signal(0.0)
    fpga.set_vg_batch(0, [VG] * 64)
    fpga.set_vg_batch(64, [VG] * 64)
    time.sleep(0.1)

    results = {}
    total = len(THRESH_VALUES) * len(DTC_VALUES) * len(BEXC_VALUES)
    done = 0

    print(f"Vg={VG}, LEAK=0x{LEAK:04X}")
    print(f"{'THRESH':>6} {'DT/C':>8} {'B_EXC':>8} | {'Rate/n':>8} {'RateBank':>8} {'vmem_m':>8} {'vmem_sd':>8} {'residual':>8} {'spk_n':>6}")
    print("-" * 96)

    for thresh in THRESH_VALUES:
        fpga.set_threshold(thresh)
        time.sleep(0.02)

        for dtc_val, dtc_name in zip(DTC_VALUES, DTC_NAMES):
            fpga.set_dt_over_c_raw(dtc_val)
            time.sleep(0.02)

            for bexc_val, bexc_name in zip(BEXC_VALUES, BEXC_NAMES):
                done += 1
                fpga.set_base_exc_raw(bexc_val)
                time.sleep(0.15)
                drain(fpga, 100)

                total_spikes = np.zeros(N_NEURONS, dtype=np.int64)
                vmem_samples = []
                n_pkts = 0

                t_start = time.perf_counter()
                while time.perf_counter() - t_start < MEASURE_TIME:
                    try:
                        pkt = fpga.recv_auto_telemetry(timeout=0.01)
                        if pkt is not None:
                            total_spikes += pkt['spike_counts'].astype(np.int64)
                            vmem_samples.append(pkt['vmem'].astype(np.float32))
                            n_pkts += 1
                    except:
                        break

                if n_pkts < 3:
                    print(f"{thresh:>6.2f} {dtc_name:>8} {bexc_name:>8} | {'TIMEOUT':>8} [{done}/{total}]")
                    continue

                elapsed = time.perf_counter() - t_start
                rate_per_neuron = (total_spikes / elapsed).mean()
                rate_bank = total_spikes.sum() / elapsed

                vmem_all = np.array(vmem_samples)
                vmem_mean = vmem_all.mean()
                vmem_std = vmem_all.std()

                spiking_mask = total_spikes > 0
                n_spiking = spiking_mask.sum()
                residual = vmem_all[:, spiking_mask].mean() if n_spiking > 0 else vmem_mean

                key = f"th={thresh:.2f}_dtc={dtc_name}_bexc={bexc_name}"
                results[key] = {
                    'threshold': thresh, 'dt_over_c': dtc_name, 'base_exc': bexc_name,
                    'rate_per_neuron': float(rate_per_neuron),
                    'rate_bank': float(rate_bank),
                    'vmem_mean': float(vmem_mean),
                    'vmem_std': float(vmem_std),
                    'residual': float(residual),
                    'n_spiking': int(n_spiking),
                    'n_pkts': n_pkts,
                }

                tag = " ***" if 5 < rate_per_neuron < 100 else (" **" if 100 < rate_per_neuron < 200 else "")
                print(f"{thresh:>6.2f} {dtc_name:>8} {bexc_name:>8} | {rate_per_neuron:>8.1f} {rate_bank:>8.0f} {vmem_mean:>8.4f} {vmem_std:>8.4f} {residual:>8.4f} {n_spiking:>6d}{tag} [{done}/{total}]")

    # Best combos
    print(f"\n{'='*96}")
    print("TOP 15 — moderate spike rate (5-100 spk/s/neuron) with highest vmem variance:")
    moderate = [(k, v) for k, v in results.items() if 5 < v['rate_per_neuron'] < 100]
    moderate.sort(key=lambda x: x[1]['vmem_std'], reverse=True)
    for i, (k, v) in enumerate(moderate[:15]):
        print(f"  {i+1}. {k}: rate/n={v['rate_per_neuron']:.1f}, vmem={v['vmem_mean']:.4f}±{v['vmem_std']:.4f}, residual={v['residual']:.4f}, spiking={v['n_spiking']}")

    if not moderate:
        print("  (none in 5-100 range)")
        all_sorted = [(k, v) for k, v in results.items()]
        all_sorted.sort(key=lambda x: x[1]['rate_per_neuron'])
        print("  Lowest rates:")
        for i, (k, v) in enumerate(all_sorted[:10]):
            print(f"  {i+1}. {k}: rate/n={v['rate_per_neuron']:.1f}, vmem={v['vmem_mean']:.4f}±{v['vmem_std']:.4f}")

    rates = [v['rate_per_neuron'] for v in results.values()]
    if rates:
        print(f"\nRate stats: min={min(rates):.1f}, max={max(rates):.1f}, median={np.median(rates):.1f}")

    with open("results/z2233b_extended_sweep.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved ({len(results)} points)")
    fpga.close()

if __name__ == "__main__":
    main()
