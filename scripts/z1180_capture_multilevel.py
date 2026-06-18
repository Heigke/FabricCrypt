#!/usr/bin/env python3
"""
z1180: Multi-Level Capture via Hardware FSM + BRAM Buffer

Tests the capture FSM that:
1. Writes pattern to memory
2. Executes partial write (ACT→delay→PRE)
3. IMMEDIATELY captures data to BRAM (before decay!)
4. Returns captured data over Ethernet

This bypasses Ethernet latency for TRUE multi-level observation!
"""

import sys
import time
import json
from datetime import datetime

sys.path.insert(0, '/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy/venv/lib/python3.12/site-packages')

from litex.tools.litex_client import RemoteClient

# CSR addresses from generated header
# CAPTURE (buffer) at base 0x0000
CAP_READ_ADDR = 0x0000
CAP_READ_DATA = 0x0004
CAP_WRITE_COUNT = 0x0008
CAP_CLEAR = 0x000c

# PWCAP (FSM) at base 0x3000
PWCAP_ROW_ADDR = 0x3000
PWCAP_BANK_ADDR = 0x3004
PWCAP_COL_ADDR = 0x3008
PWCAP_TRAS_CYCLES = 0x300c
PWCAP_NUM_CAPTURES = 0x3010
PWCAP_WRITE_PATTERN = 0x3014
PWCAP_TRIGGER = 0x3018
PWCAP_STATUS = 0x301c
PWCAP_RESULT = 0x3020
PWCAP_OPS_COUNT = 0x3024

# SDRAM at base 0x3800
SDRAM_DFII_CONTROL = 0x3800

HW_MODE = 0x0B


def count_ones(values):
    """Count total ones in a list of 32-bit values"""
    return sum(bin(v).count('1') for v in values)


def main():
    print("=" * 60)
    print("z1180: Multi-Level Capture via Hardware FSM")
    print("=" * 60)

    wb = RemoteClient()
    wb.open()
    time.sleep(0.2)

    # Verify connection
    ident_base = 0x800  # identifier_mem
    ident_chars = []
    for i in range(40):
        val = wb.read(ident_base + i * 4)
        if val == 0:
            break
        ident_chars.append(chr(val & 0x7F))
    print(f"Connected: {''.join(ident_chars)}")

    # Ensure HW mode for normal SDRAM operation
    wb.write(SDRAM_DFII_CONTROL, HW_MODE)
    time.sleep(0.01)

    results = {
        "experiment": "z1180_capture_multilevel",
        "timestamp": datetime.now().isoformat(),
        "tests": []
    }

    # Test parameters
    pattern = 0xFFFFFFFF  # All ones
    num_captures = 16     # Capture 16 words
    test_row = 100
    test_bank = 0
    test_col = 0

    # Test 1: Basic capture functionality
    print("\n" + "=" * 50)
    print("Test 1: Basic Capture Functionality")
    print("=" * 50)

    # Clear capture buffer
    wb.write(CAP_CLEAR, 1)
    time.sleep(0.001)
    wb.write(CAP_CLEAR, 0)

    # Configure FSM
    wb.write(PWCAP_ROW_ADDR, test_row)
    wb.write(PWCAP_BANK_ADDR, test_bank)
    wb.write(PWCAP_COL_ADDR, test_col)
    wb.write(PWCAP_TRAS_CYCLES, 4)  # Normal tRAS
    wb.write(PWCAP_NUM_CAPTURES, num_captures)
    wb.write(PWCAP_WRITE_PATTERN, pattern)

    print(f"Config: row={test_row}, bank={test_bank}, col={test_col}")
    print(f"        tRAS=4 cycles, captures={num_captures}, pattern=0x{pattern:08x}")

    # Check initial status
    status = wb.read(PWCAP_STATUS)
    print(f"Initial status: {status}")

    # Trigger operation
    print("Triggering capture FSM...")
    t_start = time.time()
    wb.write(PWCAP_TRIGGER, 1)

    # Wait for completion
    timeout = time.time() + 5.0
    while time.time() < timeout:
        status = wb.read(PWCAP_STATUS)
        if status == 7:  # DONE state
            break
        time.sleep(0.001)

    t_elapsed = (time.time() - t_start) * 1000
    result = wb.read(PWCAP_RESULT)
    ops = wb.read(PWCAP_OPS_COUNT)
    write_count = wb.read(CAP_WRITE_COUNT)

    print(f"Status: {status}, Result: {result}, Ops: {ops}")
    print(f"Capture buffer has {write_count} words")
    print(f"Elapsed: {t_elapsed:.2f}ms")

    # Clear trigger
    wb.write(PWCAP_TRIGGER, 0)
    time.sleep(0.001)

    # Read captured data
    captured = []
    for i in range(num_captures):
        wb.write(CAP_READ_ADDR, i)
        time.sleep(0.0001)
        val = wb.read(CAP_READ_DATA)
        captured.append(val)

    ones = count_ones(captured)
    print(f"Captured data: {ones}/{num_captures*32} ones")
    print(f"Sample: {[f'0x{v:08x}' for v in captured[:4]]}")

    results["tests"].append({
        "name": "basic_capture",
        "tras_cycles": 4,
        "captured_words": write_count,
        "ones_count": ones,
        "expected_ones": num_captures * 32,
        "sample_values": [f"0x{v:08x}" for v in captured[:8]]
    })

    # Test 2: Sweep tRAS values
    print("\n" + "=" * 50)
    print("Test 2: tRAS Sweep with Immediate Capture")
    print("=" * 50)

    tras_results = []
    tras_values = [1, 2, 3, 4, 6, 8, 12, 16, 24, 32]

    for tras in tras_values:
        # Clear buffer
        wb.write(CAP_CLEAR, 1)
        time.sleep(0.001)
        wb.write(CAP_CLEAR, 0)

        # Configure
        wb.write(PWCAP_ROW_ADDR, test_row + tras)  # Different row for each
        wb.write(PWCAP_TRAS_CYCLES, tras)
        wb.write(PWCAP_NUM_CAPTURES, num_captures)
        wb.write(PWCAP_WRITE_PATTERN, pattern)

        # Trigger
        wb.write(PWCAP_TRIGGER, 1)

        # Wait
        timeout = time.time() + 2.0
        while time.time() < timeout:
            if wb.read(PWCAP_STATUS) == 7:
                break
            time.sleep(0.001)

        # Clear trigger
        wb.write(PWCAP_TRIGGER, 0)
        time.sleep(0.001)

        # Read captured data
        captured = []
        for i in range(num_captures):
            wb.write(CAP_READ_ADDR, i)
            time.sleep(0.0001)
            val = wb.read(CAP_READ_DATA)
            captured.append(val)

        ones = count_ones(captured)
        retention_pct = ones / (num_captures * 32) * 100

        print(f"  tRAS={tras:2d} ({tras*10:3d}ns): {ones:4d}/{num_captures*32} ones ({retention_pct:5.1f}% retention)")

        tras_results.append({
            "tras_cycles": tras,
            "tras_ns": tras * 10,
            "ones_count": ones,
            "retention_percent": retention_pct,
            "sample_values": [f"0x{v:08x}" for v in captured[:4]]
        })

    results["tests"].append({
        "name": "tras_sweep",
        "pattern": f"0x{pattern:08x}",
        "num_captures": num_captures,
        "results": tras_results
    })

    # Test 3: Multiple captures at same tRAS (statistical)
    print("\n" + "=" * 50)
    print("Test 3: Statistical Sampling (10 trials per tRAS)")
    print("=" * 50)

    stat_results = []
    for tras in [2, 4, 8, 16]:
        ones_samples = []

        for trial in range(10):
            # Clear buffer
            wb.write(CAP_CLEAR, 1)
            time.sleep(0.001)
            wb.write(CAP_CLEAR, 0)

            # Configure (use different row each trial for independence)
            wb.write(PWCAP_ROW_ADDR, 500 + tras * 10 + trial)
            wb.write(PWCAP_TRAS_CYCLES, tras)
            wb.write(PWCAP_NUM_CAPTURES, num_captures)
            wb.write(PWCAP_WRITE_PATTERN, pattern)

            # Trigger
            wb.write(PWCAP_TRIGGER, 1)
            while wb.read(PWCAP_STATUS) != 7:
                time.sleep(0.001)
            wb.write(PWCAP_TRIGGER, 0)
            time.sleep(0.001)

            # Read
            captured = []
            for i in range(num_captures):
                wb.write(CAP_READ_ADDR, i)
                time.sleep(0.0001)
                captured.append(wb.read(CAP_READ_DATA))

            ones_samples.append(count_ones(captured))

        avg_ones = sum(ones_samples) / len(ones_samples)
        min_ones = min(ones_samples)
        max_ones = max(ones_samples)
        retention_pct = avg_ones / (num_captures * 32) * 100

        print(f"  tRAS={tras:2d}: avg={avg_ones:.1f}, min={min_ones}, max={max_ones} ({retention_pct:.1f}%)")

        stat_results.append({
            "tras_cycles": tras,
            "avg_ones": avg_ones,
            "min_ones": min_ones,
            "max_ones": max_ones,
            "retention_percent": retention_pct,
            "samples": ones_samples
        })

    results["tests"].append({
        "name": "statistical_sampling",
        "trials_per_tras": 10,
        "results": stat_results
    })

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    # Check for multi-level
    ones_counts = [r["ones_count"] for r in tras_results]
    unique_levels = len(set(ones_counts))

    if unique_levels > 1:
        print(f"MULTI-LEVEL ACHIEVED: {unique_levels} distinct levels!")
        for r in tras_results:
            print(f"  tRAS {r['tras_cycles']:2d}: {r['ones_count']:4d} ones ({r['retention_percent']:.1f}%)")
        results["multilevel_achieved"] = True
        results["num_levels"] = unique_levels
    else:
        if ones_counts[0] == num_captures * 32:
            print("All captures show full retention - partial write may not be affecting cells")
            print("This could indicate:")
            print("  - DFI commands not reaching DRAM")
            print("  - Cache still serving reads")
            print("  - Timing issue in FSM")
        elif ones_counts[0] == 0:
            print("All captures show complete decay")
            print("This indicates capture is happening after decay (timing issue)")
        else:
            print(f"Uniform partial retention: {ones_counts[0]}/{num_captures*32} ones")

        results["multilevel_achieved"] = False

    # Save results
    results_path = '/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy/results/z1180_capture_multilevel.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {results_path}")

    wb.close()
    print("Done!")


if __name__ == "__main__":
    main()
