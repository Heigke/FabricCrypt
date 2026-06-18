#!/usr/bin/env python3
"""
z2232_leak_sweep.py — Sweep LEAK_COND values to find optimal dynamic range
==========================================================================
For each LEAK_COND value, measure spike rates across a range of Vg values.
Goal: find LEAK_COND that gives good dynamic range (low Vg = few spikes,
high Vg = many spikes, NOT saturated at 65535).

Uses CMD_SET_LEAK (0x0A) runtime-configurable register.
"""

import sys, time, json
import numpy as np
sys.path.insert(0, "scripts")
from fpga_host_eth import FPGAEthBridge

N_NEURONS = 128
MEASURE_TIME = 0.5  # seconds per measurement point

# LEAK_COND values to test (Q16.16 integer values)
# 0x0004 = τ≈210ms (current, too slow)
# 0x0008 = τ≈105ms
# 0x0011 = τ≈49ms (original)
# 0x0020 = τ≈26ms
# 0x0040 = τ≈13ms
LEAK_VALUES = [0x0004, 0x0008, 0x0010, 0x0020, 0x0040, 0x0080]
LEAK_NAMES  = ["0x0004(τ≈210ms)", "0x0008(τ≈105ms)", "0x0010(τ≈52ms)",
               "0x0020(τ≈26ms)", "0x0040(τ≈13ms)", "0x0080(τ≈6.5ms)"]

# Vg values to test
VG_VALUES = [0.40, 0.45, 0.50, 0.55, 0.58, 0.60, 0.62, 0.65, 0.70, 0.80]

def measure_spike_rate(fpga, vg, leak_val, measure_time=MEASURE_TIME):
    """Set all neurons to vg, set leak_cond, wait, measure spike rate."""
    # Set leak conductance
    fpga.set_leak_cond(leak_val)
    time.sleep(0.05)

    # Set all neurons to same Vg
    fpga.set_vg_batch(0, [vg] * min(N_NEURONS, 64))
    fpga.set_vg_batch(64, [vg] * (N_NEURONS - 64))
    time.sleep(0.1)

    # Read baseline spike counts
    t0 = fpga.read_telemetry()
    if t0 is None:
        return None, None
    sc0 = t0["spike_counts"].copy()

    time.sleep(measure_time)

    # Read final spike counts
    t1 = fpga.read_telemetry()
    if t1 is None:
        return None, None
    sc1 = t1["spike_counts"].copy()

    # Delta (handle saturation: if sc1 == sc0 == 65535, delta is unknown)
    delta = sc1.astype(np.int32) - sc0.astype(np.int32)
    delta[delta < 0] = 0  # wrap protection

    mean_rate = delta.mean() / measure_time
    std_rate = delta.std() / measure_time
    saturated = np.sum(sc1 == 65535)

    return mean_rate, std_rate, saturated


def main():
    fpga = FPGAEthBridge()
    if not fpga.connect():
        print("FAIL: Cannot connect to FPGA")
        return

    fpga.set_kill(False)
    time.sleep(0.2)

    # Zero MAC to isolate Vg-dependent avalanche dynamics
    fpga.set_mac_signal(0.0)
    time.sleep(0.1)

    results = {}

    for li, (leak_val, leak_name) in enumerate(zip(LEAK_VALUES, LEAK_NAMES)):
        print(f"\n{'='*60}")
        print(f"LEAK_COND = {leak_name}")
        print(f"{'='*60}")

        # First: kill and reset to clear counters
        fpga.set_kill(True)
        time.sleep(0.3)
        fpga.set_kill(False)
        time.sleep(0.3)

        rates = []
        stds = []
        sats = []

        for vg in VG_VALUES:
            result = measure_spike_rate(fpga, vg, leak_val)
            if result[0] is None:
                print(f"  Vg={vg:.2f}: TIMEOUT")
                rates.append(0)
                stds.append(0)
                sats.append(0)
            else:
                mean_r, std_r, sat = result
                print(f"  Vg={vg:.2f}: rate={mean_r:8.1f} spk/s (std={std_r:6.1f}), saturated={sat}/{N_NEURONS}")
                rates.append(float(mean_r))
                stds.append(float(std_r))
                sats.append(int(sat))

        # Dynamic range: ratio of max to min non-zero rate
        rates_arr = np.array(rates)
        nonzero = rates_arr[rates_arr > 10]  # ignore near-zero
        if len(nonzero) >= 2:
            dynamic_range = nonzero.max() / nonzero.min()
        else:
            dynamic_range = 1.0

        any_saturated = any(s > 0 for s in sats)

        print(f"  Dynamic range: {dynamic_range:.1f}x, Saturated: {any_saturated}")

        results[leak_name] = {
            "leak_val_hex": hex(leak_val),
            "vg_values": VG_VALUES,
            "rates": rates,
            "stds": stds,
            "saturated_counts": sats,
            "dynamic_range": float(dynamic_range),
            "any_saturated": any_saturated
        }

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"{'LEAK_COND':<22} {'DynRange':>10} {'Saturated':>10} {'Rate@0.58':>12} {'Rate@0.40':>12}")
    for leak_name in LEAK_NAMES:
        r = results[leak_name]
        idx_058 = VG_VALUES.index(0.58) if 0.58 in VG_VALUES else -1
        idx_040 = VG_VALUES.index(0.40) if 0.40 in VG_VALUES else -1
        rate_058 = r["rates"][idx_058] if idx_058 >= 0 else 0
        rate_040 = r["rates"][idx_040] if idx_040 >= 0 else 0
        print(f"{leak_name:<22} {r['dynamic_range']:>10.1f}x {'YES' if r['any_saturated'] else 'no':>10} {rate_058:>12.1f} {rate_040:>12.1f}")

    # Best: highest dynamic range without saturation
    best = None
    best_dr = 0
    for leak_name in LEAK_NAMES:
        r = results[leak_name]
        if not r["any_saturated"] and r["dynamic_range"] > best_dr:
            best_dr = r["dynamic_range"]
            best = leak_name
    if best is None:
        # If all saturate, pick highest dynamic range anyway
        for leak_name in LEAK_NAMES:
            r = results[leak_name]
            if r["dynamic_range"] > best_dr:
                best_dr = r["dynamic_range"]
                best = leak_name

    print(f"\nBEST: {best} (dynamic range {best_dr:.1f}x)")

    with open("results/z2232_leak_sweep.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to results/z2232_leak_sweep.json")

    fpga.close()


if __name__ == "__main__":
    main()
