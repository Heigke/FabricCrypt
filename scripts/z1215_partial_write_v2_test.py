#!/usr/bin/env python3
"""
z1215: Test Partial Write FSM v2 (with fixed rddata capture)

The v2 build captures rddata from dfii.master (PHY output) instead of
ext_dfi (command input). This should fix the zero results issue.

Test sequence:
1. Initialize DDR3 (if needed)
2. Run partial write sweep with tras 0-31 cycles
3. Look for partial write effects (bits between 0% and 100%)
"""

import time
import json
from datetime import datetime
from litex.tools.litex_client import RemoteClient

# CSR addresses (will be updated from csr.csv after build)
DFII_CONTROL = 0x3000
DFII_PI0_COMMAND = 0x3004
DFII_PI0_COMMAND_ISSUE = 0x3008
DFII_PI0_ADDRESS = 0x300c
DFII_PI0_BADDRESS = 0x3010
DFII_PI0_WRDATA = 0x3014
DFII_PI0_RDDATA = 0x3018

# Partial Write CSRs (base address may change)
PW_BASE = 0x2800
PW_CONFIG = PW_BASE + 0x00
PW_WRITE_DATA = PW_BASE + 0x04
PW_REF_DATA = PW_BASE + 0x08
PW_CONTROL = PW_BASE + 0x0c
PW_STATUS = PW_BASE + 0x10
PW_RESULT = PW_BASE + 0x14
PW_RESULT_P1 = PW_BASE + 0x18
PW_RESULT_P2 = PW_BASE + 0x1c
PW_RESULT_P3 = PW_BASE + 0x20
PW_DEBUG = PW_BASE + 0x24
PW_EDGE_COUNT = PW_BASE + 0x28


def count_ones(val):
    """Count number of 1 bits in a 32-bit value."""
    return bin(val).count('1')


def init_ddr3(wb):
    """Initialize DDR3 using DFII software mode (from z1210)."""
    print("Initializing DDR3...")

    # Enter software mode
    wb.write(DFII_CONTROL, 0x0E)  # CKE|ODT|RESET_N, SEL=0
    time.sleep(0.01)

    # Precharge all
    wb.write(DFII_PI0_ADDRESS, 0x400)
    wb.write(DFII_PI0_BADDRESS, 0)
    wb.write(DFII_PI0_COMMAND, 0x0B)  # PRE
    wb.write(DFII_PI0_COMMAND_ISSUE, 1)
    time.sleep(0.001)

    # Return to hardware mode
    wb.write(DFII_CONTROL, 0x0F)
    time.sleep(0.01)

    print("DDR3 initialized")


def run_partial_write_test(wb, row, col, bank, tras, ref_data, write_data):
    """
    Run a single partial write test with specified tras.

    Returns dict with result and bit counts.
    """
    # Configure FSM
    config = (row << 18) | (col << 8) | (bank << 5) | tras
    wb.write(PW_CONFIG, config)
    wb.write(PW_REF_DATA, ref_data)
    wb.write(PW_WRITE_DATA, write_data)

    # Clear any previous state
    wb.write(PW_CONTROL, 0)
    time.sleep(0.001)

    # Trigger FSM
    wb.write(PW_CONTROL, 1)

    # Wait for completion (with timeout)
    for _ in range(100):
        status = wb.read(PW_STATUS)
        done = (status >> 1) & 0x01
        if done:
            break
        time.sleep(0.001)
    else:
        print(f"  WARNING: FSM timeout at tras={tras}")

    # Clear control
    wb.write(PW_CONTROL, 0)

    # Read results from all phases
    results = {
        'p0': wb.read(PW_RESULT),
        'p1': wb.read(PW_RESULT_P1),
        'p2': wb.read(PW_RESULT_P2),
        'p3': wb.read(PW_RESULT_P3),
    }

    # Primary result is from phase 0 (where read data appears)
    result = results['p0']
    ones = count_ones(result)
    pct = ones / 32.0 * 100

    return {
        'tras': tras,
        'result': hex(result),
        'ones': ones,
        'percent': pct,
        'all_phases': {k: hex(v) for k, v in results.items()},
    }


def main():
    print("=" * 60)
    print("z1215: Partial Write FSM v2 Test")
    print("=" * 60)

    wb = RemoteClient(host='localhost', port=1234)
    wb.open()

    # Check identifier
    ident = wb.read(0x1800)
    print(f"\nConnected! Identifier: 0x{ident:08x}")

    # Check FSM state
    status = wb.read(PW_STATUS)
    debug = wb.read(PW_DEBUG)
    edge_count = wb.read(PW_EDGE_COUNT)
    print(f"FSM Status: 0x{status:02x}, State: {debug}, Edge count: {edge_count}")

    # Test parameters
    row = 0
    col = 0
    bank = 0
    ref_data = 0x00000000  # Reference: all zeros
    write_data = 0xFFFFFFFF  # Write: all ones

    # First, do a baseline test with full tRAS to verify FSM works
    print("\n=== Baseline Test (full tRAS=31) ===")
    result = run_partial_write_test(wb, row, col, bank, 31, ref_data, write_data)
    print(f"  Result: {result['result']} ({result['ones']}/32 bits = {result['percent']:.1f}%)")
    print(f"  All phases: {result['all_phases']}")

    if result['percent'] == 0:
        print("\nWARNING: Baseline test returned zeros!")
        print("Commands may not be reaching DRAM. Checking DFII state...")

        dfii_ctrl = wb.read(DFII_CONTROL)
        print(f"  DFII Control: 0x{dfii_ctrl:02x}")

        # Try direct SW mode read
        print("\nTrying direct SW mode read...")
        init_ddr3(wb)

        # Now retry baseline
        print("\n=== Retry Baseline Test ===")
        result = run_partial_write_test(wb, row, col, bank, 31, ref_data, write_data)
        print(f"  Result: {result['result']} ({result['ones']}/32 bits = {result['percent']:.1f}%)")

    # Run partial write sweep
    print("\n=== Partial Write Sweep ===")
    print("ref_data=0x00000000, write_data=0xFFFFFFFF")
    print("Sweeping tras from 0 to 31 cycles (10ns each at 100MHz)")
    print()

    results = []
    for tras in range(0, 32, 1):  # 0-31 cycles
        result = run_partial_write_test(wb, row, col, bank, tras, ref_data, write_data)
        print(f"  tras={tras:2d} ({tras*10:3d}ns): {result['result']} ({result['ones']:2d}/32 = {result['percent']:5.1f}%)")
        results.append(result)

    # Analyze results
    print("\n=== Analysis ===")
    partial_effects = [r for r in results if 0 < r['percent'] < 100]
    all_zeros = [r for r in results if r['percent'] == 0]
    all_ones = [r for r in results if r['percent'] == 100]

    print(f"Results at 0%: {len(all_zeros)}")
    print(f"Results at 100%: {len(all_ones)}")
    print(f"Partial results: {len(partial_effects)}")

    if partial_effects:
        print("\n*** PARTIAL WRITE EFFECTS OBSERVED! ***")
        for r in partial_effects:
            print(f"  tras={r['tras']}: {r['percent']:.1f}%")
    elif len(all_zeros) > 0 and len(all_ones) > 0:
        # Transition from 0 to 100% at some tras threshold
        first_nonzero = next((r['tras'] for r in results if r['percent'] > 0), None)
        print(f"\n*** Transition observed at tras={first_nonzero} cycles ***")
        print("(This suggests discrete threshold, not analog decay)")
    elif all(r['percent'] == 100 for r in results):
        print("\n*** All writes complete (100%) - no decay observed ***")
        print("tRAS timing may not be tight enough, or refresh is interfering")
    elif all(r['percent'] == 0 for r in results):
        print("\n*** All results zero - commands may not be reaching DRAM ***")

    # Save results
    output = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "experiment": "z1215 Partial Write v2",
        "ref_data": hex(ref_data),
        "write_data": hex(write_data),
        "row": row,
        "col": col,
        "bank": bank,
        "results": results,
    }

    with open("results/z1215_partial_write_v2.json", "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to results/z1215_partial_write_v2.json")

    wb.close()


if __name__ == "__main__":
    main()
