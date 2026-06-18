#!/usr/bin/env python3
"""
z1179: Multi-Level Partial Write via Hardware FSM

Tests whether different tRAS values produce different charge levels,
enabling analog/multi-level storage in DDR3 cells.

Protocol:
1. Write known pattern to memory
2. Execute hardware FSM partial write with varying tRAS
3. Immediately read back (before natural decay)
4. Compare charge levels across different tRAS values
"""

import sys
import time
import json
from datetime import datetime

sys.path.insert(0, '/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy/venv/lib/python3.12/site-packages')

from litex.tools.litex_client import RemoteClient

# CSR addresses
PARTIAL_WRITE_BASE = 0x2800
PW_ROW_ADDR = PARTIAL_WRITE_BASE + 0x00
PW_BANK_ADDR = PARTIAL_WRITE_BASE + 0x04
PW_TRAS_CYCLES = PARTIAL_WRITE_BASE + 0x08
PW_TRIGGER = PARTIAL_WRITE_BASE + 0x0c
PW_STATUS = PARTIAL_WRITE_BASE + 0x10
PW_RESULT = PARTIAL_WRITE_BASE + 0x14
PW_OPS_COUNT = PARTIAL_WRITE_BASE + 0x18

SDRAM_BASE = 0x3000
DFII_CONTROL = SDRAM_BASE + 0x00

DDR3_BASE = 0x40000000

HW_MODE = 0x0B
SW_MODE = 0x0A

# DFII software mode commands
CMD_PRECHARGE = 0x0B  # RAS | WE | CS
CMD_REFRESH = 0x0D    # RAS | CAS | CS


def init_connection():
    """Initialize connection to FPGA"""
    print("Connecting to FPGA...")
    wb = RemoteClient()
    wb.open()
    time.sleep(0.2)
    return wb


def manual_refresh(wb):
    """Issue manual refresh command via DFII"""
    # Switch to SW mode
    wb.write(DFII_CONTROL, SW_MODE)
    time.sleep(0.0001)

    # Precharge all
    wb.write(SDRAM_BASE + 0x0c, 0x400)  # Address with A10=1
    wb.write(SDRAM_BASE + 0x10, 0)       # Bank
    wb.write(SDRAM_BASE + 0x04, CMD_PRECHARGE)
    wb.write(SDRAM_BASE + 0x08, 1)
    time.sleep(0.0001)

    # Refresh
    wb.write(SDRAM_BASE + 0x04, CMD_REFRESH)
    wb.write(SDRAM_BASE + 0x08, 1)
    time.sleep(0.0001)

    # Back to HW mode
    wb.write(DFII_CONTROL, HW_MODE)


def evict_cache(wb):
    """Evict L2 cache by writing to distant memory"""
    evict_base = DDR3_BASE + 0x4000000
    for i in range(1024):
        wb.write(evict_base + i * 4, 0xCAFEBABE)


def execute_partial_write(wb, row, bank, tras_cycles):
    """Execute hardware partial write FSM"""
    wb.write(PW_ROW_ADDR, row)
    wb.write(PW_BANK_ADDR, bank)
    wb.write(PW_TRAS_CYCLES, tras_cycles)

    # Trigger
    wb.write(PW_TRIGGER, 1)

    # Wait for completion
    timeout = time.time() + 0.1
    while time.time() < timeout:
        if wb.read(PW_STATUS) == 2:
            break
        time.sleep(0.0001)

    # Clear trigger
    wb.write(PW_TRIGGER, 0)
    while wb.read(PW_STATUS) != 0:
        time.sleep(0.0001)


def count_ones(values):
    """Count total ones in a list of 32-bit values"""
    return sum(bin(v).count('1') for v in values)


def main():
    print("=" * 60)
    print("z1179: Multi-Level Partial Write Experiment")
    print("=" * 60)

    wb = init_connection()

    results = {
        "experiment": "z1179_multilevel_partial_write",
        "timestamp": datetime.now().isoformat(),
        "tests": []
    }

    # Test parameters
    test_rows = [100, 200, 300]  # Different rows for isolation
    test_bank = 0
    num_words = 32
    pattern = 0xFFFFFFFF  # All ones - decay shows as bit flips

    # tRAS values to test (in clock cycles, each cycle = 10ns at 100MHz)
    # Normal tRAS minimum is ~4 cycles (40ns)
    # Shorter values = truncated charge = partial write
    tras_values = [1, 2, 3, 4, 6, 8, 12, 16, 24, 32]

    print(f"\nTesting tRAS values: {tras_values}")
    print(f"Each cycle = 10ns at 100MHz")
    print(f"Testing {len(test_rows)} rows with {num_words} words each\n")

    # Test 1: Fresh write + immediate read (baseline)
    print("=" * 50)
    print("Test 1: Baseline - Fresh write without partial write")
    print("=" * 50)

    manual_refresh(wb)
    time.sleep(0.01)

    test_addr = DDR3_BASE + (test_rows[0] << 3)
    for i in range(num_words):
        wb.write(test_addr + i * 4, pattern)

    # Read back immediately
    baseline_vals = [wb.read(test_addr + i * 4) for i in range(num_words)]
    baseline_ones = count_ones(baseline_vals)
    print(f"Baseline: {baseline_ones}/{num_words*32} ones (should be {num_words*32})")

    results["baseline"] = {
        "ones_count": baseline_ones,
        "expected": num_words * 32
    }

    # Test 2: Partial write with varying tRAS
    print("\n" + "=" * 50)
    print("Test 2: Partial write with varying tRAS")
    print("=" * 50)

    tras_results = []

    for tras in tras_values:
        # Use a fresh row for each tRAS value
        row_offset = tras % len(test_rows)
        test_row = test_rows[0] + tras * 10  # Spread out rows
        test_addr = DDR3_BASE + (test_row << 3)

        # Refresh to restore full charge
        manual_refresh(wb)
        time.sleep(0.01)

        # Write all-ones pattern
        for i in range(num_words):
            wb.write(test_addr + i * 4, pattern)

        # Execute partial write
        execute_partial_write(wb, test_row, test_bank, tras)

        # Small delay before read
        time.sleep(0.001)

        # Evict cache
        evict_cache(wb)

        # Read back
        after_vals = [wb.read(test_addr + i * 4) for i in range(num_words)]
        after_ones = count_ones(after_vals)

        decay_pct = (num_words * 32 - after_ones) / (num_words * 32) * 100

        print(f"  tRAS={tras:2d} ({tras*10:3d}ns): {after_ones:4d}/{num_words*32} ones ({decay_pct:5.1f}% decay)")

        tras_results.append({
            "tras_cycles": tras,
            "tras_ns": tras * 10,
            "ones_after": after_ones,
            "decay_percent": decay_pct,
            "sample_values": [f"0x{v:08x}" for v in after_vals[:4]]
        })

    results["tests"].append({
        "name": "tras_sweep",
        "pattern": f"0x{pattern:08x}",
        "num_words": num_words,
        "results": tras_results
    })

    # Test 3: Multiple partial writes to same row (cumulative effect)
    print("\n" + "=" * 50)
    print("Test 3: Cumulative partial writes")
    print("=" * 50)

    test_row = test_rows[1]
    test_addr = DDR3_BASE + (test_row << 3)
    tras = 2  # Short tRAS for maximum effect

    manual_refresh(wb)
    time.sleep(0.01)

    for i in range(num_words):
        wb.write(test_addr + i * 4, pattern)

    cumulative_results = []
    for num_ops in [0, 1, 5, 10, 20, 50, 100]:
        if num_ops > 0:
            # Execute partial writes
            for _ in range(num_ops - (len(cumulative_results) - 1 if cumulative_results else 0)):
                execute_partial_write(wb, test_row, test_bank, tras)

        evict_cache(wb)
        vals = [wb.read(test_addr + i * 4) for i in range(num_words)]
        ones = count_ones(vals)

        print(f"  After {num_ops:3d} partial writes: {ones:4d}/{num_words*32} ones")

        cumulative_results.append({
            "num_partial_writes": num_ops,
            "ones_count": ones
        })

    results["tests"].append({
        "name": "cumulative_partial_writes",
        "tras_cycles": tras,
        "results": cumulative_results
    })

    # Test 4: Check different bit patterns
    print("\n" + "=" * 50)
    print("Test 4: Different bit patterns")
    print("=" * 50)

    patterns = [
        (0xFFFFFFFF, "all_ones"),
        (0xAAAAAAAA, "alternating_10"),
        (0x55555555, "alternating_01"),
        (0xFF00FF00, "bytes_alt"),
        (0x0F0F0F0F, "nibbles_alt"),
    ]

    test_row = test_rows[2]
    test_addr = DDR3_BASE + (test_row << 3)
    tras = 2

    pattern_results = []
    for pat, name in patterns:
        manual_refresh(wb)
        time.sleep(0.01)

        expected_ones = num_words * bin(pat).count('1')

        for i in range(num_words):
            wb.write(test_addr + i * 4, pat)

        before_vals = [wb.read(test_addr + i * 4) for i in range(num_words)]
        before_ones = count_ones(before_vals)

        execute_partial_write(wb, test_row, test_bank, tras)

        evict_cache(wb)

        after_vals = [wb.read(test_addr + i * 4) for i in range(num_words)]
        after_ones = count_ones(after_vals)

        print(f"  {name:15s}: before={before_ones:4d}, after={after_ones:4d}, diff={before_ones-after_ones:4d}")

        pattern_results.append({
            "pattern": f"0x{pat:08x}",
            "name": name,
            "expected_ones": expected_ones,
            "before_ones": before_ones,
            "after_ones": after_ones,
            "difference": before_ones - after_ones
        })

    results["tests"].append({
        "name": "pattern_comparison",
        "tras_cycles": tras,
        "results": pattern_results
    })

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    # Check if we got different levels
    ones_counts = [r["ones_after"] for r in tras_results]
    unique_levels = len(set(ones_counts))

    if unique_levels > 1:
        print(f"MULTI-LEVEL ACHIEVED: {unique_levels} distinct levels observed")
        for r in tras_results:
            print(f"  tRAS {r['tras_cycles']:2d}: {r['ones_after']:4d} ones")
        results["multilevel_achieved"] = True
        results["num_levels"] = unique_levels
    else:
        print("Single level only - all tRAS values produced same result")
        print("This may indicate:")
        print("  - Complete decay regardless of tRAS (too fast)")
        print("  - No decay (cells retaining charge)")
        print("  - Cache masking the effect")
        results["multilevel_achieved"] = False

    # Save results
    results_path = '/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy/results/z1179_multilevel_partial_write.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {results_path}")

    wb.close()
    print("Done!")


if __name__ == "__main__":
    main()
