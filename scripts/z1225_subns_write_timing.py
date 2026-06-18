#!/usr/bin/env python3
"""
z1225: Sub-nanosecond Write Timing via ODELAY

Current state:
- tRAS=0: 0 bits (too fast)
- tRAS=1: 16 bits (partial)
- tRAS=2: 32 bits (full)

Goal: Find intermediate states between tRAS=0 and tRAS=1 by using
ODELAY to shift write data timing by ~78ps per tap.

If we can get 4, 8, 12 bits etc, that suggests we're approaching
the actual charge transfer threshold, not just DQ lane selection.
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

# DDRPHY CSRs
PHY_RST = 0x800
PHY_DLY_SEL = 0x804
PHY_HALF_SYS8X_TAPS = 0x808
PHY_WLEVEL_EN = 0x80c
PHY_WLEVEL_STROBE = 0x810
PHY_RDLY_DQ_RST = 0x814
PHY_RDLY_DQ_INC = 0x818
PHY_RDLY_DQ_BITSLIP_RST = 0x81c
PHY_RDLY_DQ_BITSLIP = 0x820
PHY_WDLY_DQ_RST = 0x824
PHY_WDLY_DQ_INC = 0x828
PHY_WDLY_DQS_RST = 0x82c
PHY_WDLY_DQS_INC = 0x830

# Partial write FSM
PW_CONFIG = 0x2800
PW_WRITE_DATA = 0x2804
PW_CONTROL = 0x280c
PW_STATUS = 0x2810
PW_RESULT = 0x2814


def make_config(row, col, bank, tras):
    return (row << 18) | (col << 8) | (bank << 5) | tras


def sw_write(wb, row, col, bank, data):
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


def run_fsm(wb, row, col, bank, tras, write_data):
    config = make_config(row, col, bank, tras)
    wb.write(PW_CONFIG, config)
    wb.write(PW_WRITE_DATA, write_data)
    wb.write(PW_CONTROL, 0)
    time.sleep(0.001)
    wb.write(PW_CONTROL, 1)
    for _ in range(100):
        status = wb.read(PW_STATUS)
        if (status >> 1) & 1:
            break
        time.sleep(0.001)
    result = wb.read(PW_RESULT)
    wb.write(PW_CONTROL, 0)
    return result


def set_wdly_dq(wb, byte_lane, taps):
    """Set write DQ delay to specific tap value."""
    wb.write(PHY_DLY_SEL, 1 << byte_lane)
    time.sleep(0.0001)
    wb.write(PHY_WDLY_DQ_RST, 1)
    time.sleep(0.0001)
    for _ in range(taps):
        wb.write(PHY_WDLY_DQ_INC, 1)
        time.sleep(0.0001)


def set_wdly_dqs(wb, byte_lane, taps):
    """Set write DQS delay to specific tap value."""
    wb.write(PHY_DLY_SEL, 1 << byte_lane)
    time.sleep(0.0001)
    wb.write(PHY_WDLY_DQS_RST, 1)
    time.sleep(0.0001)
    for _ in range(taps):
        wb.write(PHY_WDLY_DQS_INC, 1)
        time.sleep(0.0001)


def count_ones(val):
    return bin(val).count('1')


def main():
    print("=" * 70)
    print("z1225: Sub-nanosecond Write Timing via ODELAY")
    print("=" * 70)
    print()
    print("Goal: Find bit counts between 0 (tRAS=0) and 16 (tRAS=1)")
    print("      by shifting write data timing with ODELAY")
    print()

    ROW = 600
    BANK = 0
    COL = 0
    PATTERN = 0xFFFFFFFF

    results = {
        "experiment": "z1225_subns_write_timing",
        "timestamp": datetime.now().isoformat(),
        "goal": "Find intermediate bit counts via ODELAY tuning",
        "sweeps": []
    }

    print("Connecting to FPGA...")
    wb = RemoteClient(host='localhost', port=1234)
    wb.open()

    print("\nInitializing DDR3...")
    subprocess.run(["python3", "scripts/z1210_ddr3_litex_init.py"],
                   capture_output=True, text=True)
    print("  DDR3 init complete")

    # Baseline measurements
    print("\n=== Baseline (no ODELAY adjustment) ===")
    for tras in [0, 1, 2]:
        sw_write(wb, ROW, COL, BANK, 0x00000000)
        run_fsm(wb, ROW, COL, BANK, tras=tras, write_data=PATTERN)
        val = sw_read(wb, ROW, COL, BANK)
        print(f"  tRAS={tras}: 0x{val:08x} ({count_ones(val)} bits)")

    # Sweep 1: Adjust write DQ delay while using tRAS=1
    print("\n" + "=" * 70)
    print("Sweep 1: WDLY_DQ (both lanes) + tRAS=1")
    print("=" * 70)
    print(f"{'Tap':>4} | {'~ps':>6} | {'Result':>12} | {'Bits':>4}")
    print("-" * 40)

    sweep1 = []
    for tap in range(0, 32, 2):  # Step by 2 for speed
        # Set both byte lanes
        set_wdly_dq(wb, 0, tap)
        set_wdly_dq(wb, 1, tap)

        # Clear and write
        sw_write(wb, ROW, COL, BANK, 0x00000000)
        run_fsm(wb, ROW, COL, BANK, tras=1, write_data=PATTERN)
        val = sw_read(wb, ROW, COL, BANK)
        bits = count_ones(val)

        print(f"{tap:4d} | {tap*78:6d} | 0x{val:08x} | {bits:4d}")
        sweep1.append({"tap": tap, "ps": tap*78, "result": hex(val), "bits": bits})

    results["sweeps"].append({"name": "wdly_dq_tras1", "data": sweep1})

    # Reset delays
    set_wdly_dq(wb, 0, 0)
    set_wdly_dq(wb, 1, 0)

    # Sweep 2: Try tRAS=0 with increased write delay (maybe enough delay = some bits)
    print("\n" + "=" * 70)
    print("Sweep 2: WDLY_DQ + tRAS=0 (can we get ANY bits?)")
    print("=" * 70)
    print(f"{'Tap':>4} | {'~ps':>6} | {'Result':>12} | {'Bits':>4}")
    print("-" * 40)

    sweep2 = []
    for tap in range(0, 32, 2):
        set_wdly_dq(wb, 0, tap)
        set_wdly_dq(wb, 1, tap)

        sw_write(wb, ROW, COL, BANK, 0x00000000)
        run_fsm(wb, ROW, COL, BANK, tras=0, write_data=PATTERN)
        val = sw_read(wb, ROW, COL, BANK)
        bits = count_ones(val)

        print(f"{tap:4d} | {tap*78:6d} | 0x{val:08x} | {bits:4d}")
        sweep2.append({"tap": tap, "ps": tap*78, "result": hex(val), "bits": bits})

    results["sweeps"].append({"name": "wdly_dq_tras0", "data": sweep2})

    # Reset delays
    set_wdly_dq(wb, 0, 0)
    set_wdly_dq(wb, 1, 0)

    # Sweep 3: Try negative direction - reduce DQS delay to make data arrive earlier
    print("\n" + "=" * 70)
    print("Sweep 3: WDLY_DQS adjustment + tRAS=1")
    print("=" * 70)
    print(f"{'Tap':>4} | {'~ps':>6} | {'Result':>12} | {'Bits':>4}")
    print("-" * 40)

    sweep3 = []
    for tap in range(0, 32, 2):
        set_wdly_dqs(wb, 0, tap)
        set_wdly_dqs(wb, 1, tap)

        sw_write(wb, ROW, COL, BANK, 0x00000000)
        run_fsm(wb, ROW, COL, BANK, tras=1, write_data=PATTERN)
        val = sw_read(wb, ROW, COL, BANK)
        bits = count_ones(val)

        print(f"{tap:4d} | {tap*78:6d} | 0x{val:08x} | {bits:4d}")
        sweep3.append({"tap": tap, "ps": tap*78, "result": hex(val), "bits": bits})

    results["sweeps"].append({"name": "wdly_dqs_tras1", "data": sweep3})

    # Reset all delays
    set_wdly_dq(wb, 0, 0)
    set_wdly_dq(wb, 1, 0)
    set_wdly_dqs(wb, 0, 0)
    set_wdly_dqs(wb, 1, 0)

    # Analysis
    print("\n" + "=" * 70)
    print("ANALYSIS")
    print("=" * 70)

    # Check for intermediate values
    all_bits = set()
    for sweep in results["sweeps"]:
        for d in sweep["data"]:
            all_bits.add(d["bits"])

    print(f"Unique bit counts observed: {sorted(all_bits)}")

    if len(all_bits) > 3:  # More than just 0, 16, 32
        print("\n*** FOUND INTERMEDIATE VALUES! ***")
        results["verdict"] = "INTERMEDIATE_VALUES_FOUND"
    else:
        print("\nNo intermediate values found - timing is quantized")
        results["verdict"] = "QUANTIZED"

    # Save
    output_path = "/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy/results/z1225_subns_write_timing.json"
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_path}")

    wb.close()
    print("\nDone!")


if __name__ == "__main__":
    main()
