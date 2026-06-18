#!/usr/bin/env python3
"""
z1227: Test v4 Sub-Cycle Timing (2.5ns resolution)

Config encoding for v4:
  [row:14][col:10][bank:3][phase_offset:2][tras:5]
  bits: [33:20]=row, [19:10]=col, [9:7]=bank, [6:5]=ph_off, [4:0]=tras

Total timing = tras * 10ns + phase_offset * 2.5ns

Test matrix:
  tras=0, ph=0 -> 0ns
  tras=0, ph=1 -> 2.5ns
  tras=0, ph=2 -> 5ns
  tras=0, ph=3 -> 7.5ns
  tras=1, ph=0 -> 10ns
  tras=1, ph=1 -> 12.5ns
"""

import time
import json
import subprocess
from datetime import datetime
from litex.tools.litex_client import RemoteClient

# CSRs
DFII_CONTROL = 0x3000
PHASE_SIZE = 0x18
def pi_base(phase): return 0x3004 + phase * PHASE_SIZE

# Partial write v4 CSRs
PW_CONFIG = 0x2800
PW_WRITE_DATA = 0x2804
PW_CONTROL = 0x280c
PW_STATUS = 0x2810
PW_RESULT = 0x2814


def make_config_v4(row, col, bank, phase_offset, tras):
    """
    v4 config: [row:14][col:10][bank:3][phase_offset:2][tras:5]
    """
    return ((row & 0x3FFF) << 20) | ((col & 0x3FF) << 10) | ((bank & 0x7) << 7) | ((phase_offset & 0x3) << 5) | (tras & 0x1F)


def sw_write(wb, row, col, bank, data):
    """Standard software write."""
    WRPHASE = 3
    wb.write(DFII_CONTROL, 0x0E)
    time.sleep(0.001)
    wb.write(pi_base(0) + 0x08, 0x400)
    wb.write(pi_base(0) + 0x0c, 0)
    wb.write(pi_base(0) + 0x00, 0x0B)
    wb.write(pi_base(0) + 0x04, 1)
    time.sleep(0.0001)
    wb.write(pi_base(0) + 0x08, row)
    wb.write(pi_base(0) + 0x0c, bank)
    wb.write(pi_base(0) + 0x00, 0x09)
    wb.write(pi_base(0) + 0x04, 1)
    time.sleep(0.0001)
    wb.write(pi_base(WRPHASE) + 0x10, data)
    wb.write(pi_base(WRPHASE) + 0x08, col)
    wb.write(pi_base(WRPHASE) + 0x0c, bank)
    wb.write(pi_base(WRPHASE) + 0x00, 0x17)
    wb.write(pi_base(WRPHASE) + 0x04, 1)
    time.sleep(0.001)
    wb.write(pi_base(0) + 0x08, 0x400)
    wb.write(pi_base(0) + 0x0c, 0)
    wb.write(pi_base(0) + 0x00, 0x0B)
    wb.write(pi_base(0) + 0x04, 1)
    time.sleep(0.0001)
    wb.write(DFII_CONTROL, 0x0F)


def sw_read(wb, row, col, bank):
    """Standard software read."""
    RDPHASE = 2
    wb.write(DFII_CONTROL, 0x0E)
    time.sleep(0.0005)
    wb.write(pi_base(0) + 0x08, 0x400)
    wb.write(pi_base(0) + 0x0c, 0)
    wb.write(pi_base(0) + 0x00, 0x0B)
    wb.write(pi_base(0) + 0x04, 1)
    time.sleep(0.0001)
    wb.write(pi_base(0) + 0x08, row)
    wb.write(pi_base(0) + 0x0c, bank)
    wb.write(pi_base(0) + 0x00, 0x09)
    wb.write(pi_base(0) + 0x04, 1)
    time.sleep(0.0001)
    wb.write(pi_base(RDPHASE) + 0x08, col)
    wb.write(pi_base(RDPHASE) + 0x0c, bank)
    wb.write(pi_base(RDPHASE) + 0x00, 0x25)
    wb.write(pi_base(RDPHASE) + 0x04, 1)
    time.sleep(0.0005)
    result = wb.read(pi_base(0) + 0x14)
    wb.write(pi_base(0) + 0x08, 0x400)
    wb.write(pi_base(0) + 0x0c, 0)
    wb.write(pi_base(0) + 0x00, 0x0B)
    wb.write(pi_base(0) + 0x04, 1)
    time.sleep(0.0001)
    wb.write(DFII_CONTROL, 0x0F)
    return result


def run_fsm_v4(wb, row, col, bank, tras, phase_offset, write_data):
    """Run v4 FSM with sub-cycle timing."""
    config = make_config_v4(row, col, bank, phase_offset, tras)
    wb.write(PW_CONFIG, config)
    wb.write(PW_WRITE_DATA, write_data)
    wb.write(PW_CONTROL, 0)
    time.sleep(0.001)
    wb.write(PW_CONTROL, 1)
    for _ in range(100):
        status = wb.read(PW_STATUS)
        if (status >> 1) & 1:  # done bit
            break
        time.sleep(0.001)
    result = wb.read(PW_RESULT)
    wb.write(PW_CONTROL, 0)
    return result


def count_ones(val):
    return bin(val).count('1')


def main():
    print("=" * 70)
    print("z1227: Sub-Cycle Timing Test (2.5ns resolution)")
    print("=" * 70)
    print()

    ROW = 800
    BANK = 0
    COL = 0
    PATTERN = 0xFFFFFFFF

    results = {
        "experiment": "z1227_subcycle_test",
        "timestamp": datetime.now().isoformat(),
        "timing_sweep": []
    }

    print("Connecting to FPGA...")
    wb = RemoteClient(host='localhost', port=1234)
    wb.open()

    print("\nInitializing DDR3...")
    subprocess.run(["python3", "scripts/z1210_ddr3_litex_init.py"],
                   capture_output=True, text=True)
    print("  DDR3 init complete")

    # Test sub-cycle timing
    print("\n" + "=" * 70)
    print("Sub-Cycle Timing Sweep")
    print("=" * 70)
    print(f"{'tRAS':>5} | {'Ph':>2} | {'Total ns':>8} | {'Result':>12} | {'Bits':>4}")
    print("-" * 50)

    # Sweep tras from 0-3 and phase_offset from 0-3
    for tras in range(4):
        for ph_off in range(4):
            total_ns = tras * 10.0 + ph_off * 2.5

            # Clear row
            sw_write(wb, ROW, COL, BANK, 0x00000000)

            # Run FSM with sub-cycle timing
            run_fsm_v4(wb, ROW, COL, BANK, tras, ph_off, PATTERN)

            # Read back via software
            val = sw_read(wb, ROW, COL, BANK)
            bits = count_ones(val)

            print(f"{tras:5d} | {ph_off:2d} | {total_ns:7.1f} | 0x{val:08x} | {bits:4d}")

            results["timing_sweep"].append({
                "tras": tras,
                "phase_offset": ph_off,
                "total_ns": total_ns,
                "result": hex(val),
                "bits": bits
            })

    # Analysis
    print("\n" + "=" * 70)
    print("ANALYSIS")
    print("=" * 70)

    unique_bits = sorted(set(d["bits"] for d in results["timing_sweep"]))
    print(f"Unique bit counts: {unique_bits}")

    # Group by total timing
    by_timing = {}
    for d in results["timing_sweep"]:
        t = d["total_ns"]
        if t not in by_timing:
            by_timing[t] = []
        by_timing[t].append(d["bits"])

    print("\nBits by timing:")
    for t in sorted(by_timing.keys()):
        bits = by_timing[t]
        avg = sum(bits) / len(bits)
        print(f"  {t:6.1f}ns: {bits[0]:4d} bits")

    # Check for gradual increase
    timings = sorted(by_timing.keys())
    prev_bits = by_timing[timings[0]][0]
    transitions = []
    for t in timings[1:]:
        curr_bits = by_timing[t][0]
        if curr_bits != prev_bits:
            transitions.append((t, prev_bits, curr_bits))
        prev_bits = curr_bits

    if transitions:
        print("\nTransitions detected:")
        for t, b1, b2 in transitions:
            print(f"  At {t:.1f}ns: {b1} -> {b2} bits")
        results["verdict"] = "TRANSITIONS_FOUND"
    else:
        print("\nNo transitions in bit count")
        results["verdict"] = "NO_TRANSITIONS"

    # Save
    output_path = "/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy/results/z1227_subcycle_test.json"
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_path}")

    wb.close()
    print("\nDone!")


if __name__ == "__main__":
    main()
