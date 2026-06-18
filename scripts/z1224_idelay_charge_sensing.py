#!/usr/bin/env python3
"""
z1224: IDELAY-based Charge Threshold Sensing

Theory: If cells have partial charge, the read voltage will be intermediate.
By sweeping the IDELAY tap setting, we sample at different points in the
data eye. Partially charged cells should show:
- More sensitivity to delay position
- Transitions at different delay values
- Probabilistic reads (sometimes 0, sometimes 1)

Approach:
1. Write full charge (tRAS=5) to one row
2. Write partial (tRAS=1) to another row
3. Sweep IDELAY from 0-31 taps
4. At each tap, read both rows multiple times
5. Compare stability/variability

IDELAY tap = ~78ps on Artix-7 (varies with PVT)
32 taps = ~2.5ns sweep range
"""

import time
import json
import subprocess
from datetime import datetime
from collections import Counter
from litex.tools.litex_client import RemoteClient

# CSRs
DFII_CONTROL = 0x3000
PHASE_SIZE = 0x18
def pi_base(phase): return 0x3004 + phase * PHASE_SIZE

# DDRPHY CSRs (based on LiteDRAM s7ddrphy.py order)
PHY_RST = 0x800
PHY_DLY_SEL = 0x804
PHY_HALF_SYS8X_TAPS = 0x808
PHY_WLEVEL_EN = 0x80c
PHY_WLEVEL_STROBE = 0x810
# For builds without cdly: rdly starts at 0x814
PHY_RDLY_DQ_RST = 0x814
PHY_RDLY_DQ_INC = 0x818
PHY_RDLY_DQ_BITSLIP_RST = 0x81c
PHY_RDLY_DQ_BITSLIP = 0x820
PHY_WDLY_DQ_RST = 0x824
PHY_WDLY_DQ_INC = 0x828
PHY_WDLY_DQS_RST = 0x82c
PHY_WDLY_DQS_INC = 0x830
PHY_RDPHASE = 0x834
PHY_WRPHASE = 0x838

# Partial write FSM CSRs
PW_CONFIG = 0x2800
PW_WRITE_DATA = 0x2804
PW_CONTROL = 0x280c
PW_STATUS = 0x2810
PW_RESULT = 0x2814


def make_config(row, col, bank, tras):
    return (row << 18) | (col << 8) | (bank << 5) | tras


def sw_write(wb, row, col, bank, data):
    """Software mode write via DFII phase registers."""
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
    """Software mode read via DFII phase registers."""
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
    """Run partial write FSM with configurable tRAS."""
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


def set_rdly(wb, byte_lane, taps):
    """Set read delay to specific tap value."""
    # Select byte lane
    wb.write(PHY_DLY_SEL, 1 << byte_lane)
    time.sleep(0.0001)

    # Reset delay to 0
    wb.write(PHY_RDLY_DQ_RST, 1)
    time.sleep(0.0001)

    # Increment to desired tap
    for _ in range(taps):
        wb.write(PHY_RDLY_DQ_INC, 1)
        time.sleep(0.0001)


def count_ones(val):
    return bin(val).count('1')


def main():
    print("=" * 70)
    print("z1224: IDELAY-based Charge Threshold Sensing")
    print("=" * 70)
    print()

    # Test parameters
    FULL_ROW = 500
    PARTIAL_ROW = 501
    BANK = 0
    COL = 0
    PATTERN = 0xFFFFFFFF
    NUM_READS = 10  # Multiple reads per setting to check stability
    MAX_TAPS = 32

    results = {
        "experiment": "z1224_idelay_charge_sensing",
        "timestamp": datetime.now().isoformat(),
        "theory": "Partial charge cells should show more IDELAY sensitivity",
        "full_row": FULL_ROW,
        "partial_row": PARTIAL_ROW,
        "num_reads_per_tap": NUM_READS,
        "idelay_sweep": []
    }

    print("Connecting to FPGA...")
    wb = RemoteClient(host='localhost', port=1234)
    wb.open()

    # Initialize DDR3
    print("\nInitializing DDR3...")
    subprocess.run(["python3", "scripts/z1210_ddr3_litex_init.py"],
                   capture_output=True, text=True)
    print("  DDR3 init complete")

    # Check current PHY state
    print("\nPHY register state:")
    print(f"  DLY_SEL: 0x{wb.read(PHY_DLY_SEL):02x}")
    print(f"  HALF_SYS8X_TAPS: {wb.read(PHY_HALF_SYS8X_TAPS)}")

    # Write test data
    print("\nPreparing test rows...")

    # Full charge row (tRAS=5)
    sw_write(wb, FULL_ROW, COL, BANK, 0x00000000)
    run_fsm(wb, FULL_ROW, COL, BANK, tras=5, write_data=PATTERN)
    full_initial = sw_read(wb, FULL_ROW, COL, BANK)
    print(f"  Full charge row {FULL_ROW}: 0x{full_initial:08x} ({count_ones(full_initial)} bits)")

    # Partial charge row (tRAS=1)
    sw_write(wb, PARTIAL_ROW, COL, BANK, 0x00000000)
    run_fsm(wb, PARTIAL_ROW, COL, BANK, tras=1, write_data=PATTERN)
    partial_initial = sw_read(wb, PARTIAL_ROW, COL, BANK)
    print(f"  Partial charge row {PARTIAL_ROW}: 0x{partial_initial:08x} ({count_ones(partial_initial)} bits)")

    # Sweep IDELAY
    print()
    print("=" * 70)
    print("IDELAY Sweep (byte lane 0)")
    print("=" * 70)
    print(f"{'Tap':>4} | {'Full bits':>10} | {'Full var':>8} | {'Part bits':>10} | {'Part var':>8}")
    print("-" * 70)

    for tap in range(MAX_TAPS):
        set_rdly(wb, byte_lane=0, taps=tap)

        # Read full row multiple times
        full_reads = []
        for _ in range(NUM_READS):
            val = sw_read(wb, FULL_ROW, COL, BANK)
            full_reads.append(val)

        # Read partial row multiple times
        partial_reads = []
        for _ in range(NUM_READS):
            val = sw_read(wb, PARTIAL_ROW, COL, BANK)
            partial_reads.append(val)

        # Calculate statistics
        full_bits = [count_ones(v) for v in full_reads]
        partial_bits = [count_ones(v) for v in partial_reads]

        full_avg = sum(full_bits) / len(full_bits)
        partial_avg = sum(partial_bits) / len(partial_bits)

        full_unique = len(set(full_reads))
        partial_unique = len(set(partial_reads))

        print(f"{tap:4d} | {full_avg:10.1f} | {full_unique:8d} | {partial_avg:10.1f} | {partial_unique:8d}")

        results["idelay_sweep"].append({
            "tap": tap,
            "full_reads": [hex(v) for v in full_reads],
            "full_bits_avg": round(full_avg, 2),
            "full_unique_values": full_unique,
            "partial_reads": [hex(v) for v in partial_reads],
            "partial_bits_avg": round(partial_avg, 2),
            "partial_unique_values": partial_unique
        })

    # Also sweep byte lane 1
    print()
    print("=" * 70)
    print("IDELAY Sweep (byte lane 1)")
    print("=" * 70)
    print(f"{'Tap':>4} | {'Full bits':>10} | {'Full var':>8} | {'Part bits':>10} | {'Part var':>8}")
    print("-" * 70)

    results["idelay_sweep_lane1"] = []

    for tap in range(MAX_TAPS):
        # Reset lane 0 to good value first
        set_rdly(wb, byte_lane=0, taps=8)
        # Now sweep lane 1
        set_rdly(wb, byte_lane=1, taps=tap)

        full_reads = []
        for _ in range(NUM_READS):
            val = sw_read(wb, FULL_ROW, COL, BANK)
            full_reads.append(val)

        partial_reads = []
        for _ in range(NUM_READS):
            val = sw_read(wb, PARTIAL_ROW, COL, BANK)
            partial_reads.append(val)

        full_bits = [count_ones(v) for v in full_reads]
        partial_bits = [count_ones(v) for v in partial_reads]

        full_avg = sum(full_bits) / len(full_bits)
        partial_avg = sum(partial_bits) / len(partial_bits)

        full_unique = len(set(full_reads))
        partial_unique = len(set(partial_reads))

        print(f"{tap:4d} | {full_avg:10.1f} | {full_unique:8d} | {partial_avg:10.1f} | {partial_unique:8d}")

        results["idelay_sweep_lane1"].append({
            "tap": tap,
            "full_bits_avg": round(full_avg, 2),
            "full_unique_values": full_unique,
            "partial_bits_avg": round(partial_avg, 2),
            "partial_unique_values": partial_unique
        })

    # Reset delays to reasonable values
    print("\nResetting IDELAY to default...")
    set_rdly(wb, byte_lane=0, taps=8)
    set_rdly(wb, byte_lane=1, taps=8)

    # Analysis
    print()
    print("=" * 70)
    print("ANALYSIS")
    print("=" * 70)

    # Check if partial shows more variability
    lane0_data = results["idelay_sweep"]
    full_total_unique = sum(d["full_unique_values"] for d in lane0_data)
    partial_total_unique = sum(d["partial_unique_values"] for d in lane0_data)

    print(f"Total unique values across all taps (lane 0):")
    print(f"  Full charge: {full_total_unique}")
    print(f"  Partial:     {partial_total_unique}")

    if partial_total_unique > full_total_unique * 1.5:
        verdict = "PARTIAL CHARGE INDICATED - more read variability"
        results["verdict"] = "INDICATED"
    elif partial_total_unique == full_total_unique:
        verdict = "NO DIFFERENCE - both equally stable"
        results["verdict"] = "NO_DIFFERENCE"
    else:
        verdict = "INCONCLUSIVE - need more analysis"
        results["verdict"] = "INCONCLUSIVE"

    print(f"\nVERDICT: {verdict}")
    results["verdict_detail"] = verdict

    # Save results
    output_path = "/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy/results/z1224_idelay_charge_sensing.json"
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_path}")

    wb.close()
    print("\nDone!")


if __name__ == "__main__":
    main()
