#!/usr/bin/env python3
"""
z1177: Stress-Based Multi-Level Decay

Approach: Use rapid ACT/PRE cycles to stress cells, then measure
differential decay rates to achieve intermediate charge levels.

Key insight: Cells with more stress/wear decay faster.
By varying stress level (number of cycles), we can create
different decay rates, which translates to multi-level storage.

Protocol:
1. Write pattern
2. Apply N ACT/PRE stress cycles
3. Wait fixed decay time
4. Read back - stressed cells should show more decay
"""

import sys
import time
import json
from datetime import datetime

sys.path.insert(0, '/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy/venv/lib/python3.12/site-packages')

from litex.tools.litex_client import RemoteClient

DDR3_BASE = 0x40000000
SDRAM_BASE = 0x3000
DFII_CONTROL = SDRAM_BASE + 0x00
DFII_PI0_COMMAND = SDRAM_BASE + 0x04
DFII_PI0_COMMAND_ISSUE = SDRAM_BASE + 0x08
DFII_PI0_ADDRESS = SDRAM_BASE + 0x0c
DFII_PI0_BADDRESS = SDRAM_BASE + 0x10

HW_MODE = 0x0B
SW_MODE = 0x0A
CMD_ACTIVATE = 0x09
CMD_PRECHARGE = 0x0B
CMD_REFRESH = 0x0D


def init_ddr3(wb):
    wb.write(DFII_CONTROL, HW_MODE)
    time.sleep(0.1)
    for _ in range(10):
        issue_refresh(wb)


def issue_refresh(wb):
    wb.write(DFII_CONTROL, SW_MODE)
    time.sleep(0.0001)
    wb.write(DFII_PI0_ADDRESS, 0x400)
    wb.write(DFII_PI0_BADDRESS, 0)
    wb.write(DFII_PI0_COMMAND, CMD_PRECHARGE)
    wb.write(DFII_PI0_COMMAND_ISSUE, 1)
    time.sleep(0.0001)
    wb.write(DFII_PI0_COMMAND, CMD_REFRESH)
    wb.write(DFII_PI0_COMMAND_ISSUE, 1)
    time.sleep(0.0001)
    wb.write(DFII_CONTROL, HW_MODE)


def stress_row(wb, row, bank, num_cycles):
    """Apply ACT/PRE stress cycles to a row"""
    wb.write(DFII_CONTROL, SW_MODE)
    for _ in range(num_cycles):
        wb.write(DFII_PI0_ADDRESS, row)
        wb.write(DFII_PI0_BADDRESS, bank)
        wb.write(DFII_PI0_COMMAND, CMD_ACTIVATE)
        wb.write(DFII_PI0_COMMAND_ISSUE, 1)
        wb.write(DFII_PI0_ADDRESS, 0x400)
        wb.write(DFII_PI0_COMMAND, CMD_PRECHARGE)
        wb.write(DFII_PI0_COMMAND_ISSUE, 1)
    wb.write(DFII_CONTROL, HW_MODE)


def evict_cache(wb):
    evict_base = DDR3_BASE + 0x4000000
    for i in range(4096):
        wb.write(evict_base + i * 4, 0xDEADC0DE)


def main():
    print("=" * 60)
    print("z1177: Stress-Based Multi-Level Decay")
    print("=" * 60)

    wb = RemoteClient()
    wb.open()
    print("Connected to Etherbone")

    results = {
        "experiment": "z1177_stress_decay_levels",
        "timestamp": datetime.now().isoformat(),
        "tests": []
    }

    init_ddr3(wb)
    print(f"DFII: 0x{wb.read(DFII_CONTROL):02x}")

    # Test different stress levels
    print("\n=== Stress Level vs Decay Rate ===")
    print("Testing if stress cycles affect decay rate...")

    pattern = 0xFFFFFFFF  # All ones - will decay to zeros
    decay_time_ms = 5  # Short decay window
    test_row = 100
    test_bank = 0
    test_addr = DDR3_BASE + (test_row << 3)  # Approximate address mapping

    stress_results = []

    for stress_cycles in [0, 10, 50, 100, 200, 500]:
        # Refresh to restore full charge
        issue_refresh(wb)
        time.sleep(0.01)

        # Write all-ones pattern
        wb.write(test_addr, pattern)
        for i in range(16):
            wb.write(test_addr + i * 4, pattern)

        # Apply stress cycles
        if stress_cycles > 0:
            stress_row(wb, test_row, test_bank, stress_cycles)

        # Evict cache
        evict_cache(wb)

        # Wait for decay
        time.sleep(decay_time_ms / 1000.0)

        # Read back multiple words
        ones_counts = []
        for i in range(16):
            val = wb.read(test_addr + i * 4)
            ones_counts.append(bin(val).count('1'))

        avg_ones = sum(ones_counts) / len(ones_counts)
        min_ones = min(ones_counts)
        max_ones = max(ones_counts)

        print(f"  {stress_cycles:4d} cycles: avg={avg_ones:.1f}/32 ones (min={min_ones}, max={max_ones})")

        stress_results.append({
            "stress_cycles": stress_cycles,
            "avg_ones": avg_ones,
            "min_ones": min_ones,
            "max_ones": max_ones,
            "decay_percent": (32 - avg_ones) / 32 * 100
        })

    results["tests"].append({
        "name": "stress_vs_decay",
        "decay_time_ms": decay_time_ms,
        "results": stress_results
    })

    # Test different decay times at fixed stress
    print("\n=== Decay Time Curve (with 100 stress cycles) ===")

    decay_curve = []
    stress_cycles = 100

    for decay_ms in [1, 2, 5, 10, 20, 50]:
        issue_refresh(wb)
        time.sleep(0.01)

        # Write pattern
        for i in range(16):
            wb.write(test_addr + i * 4, pattern)

        # Apply stress
        stress_row(wb, test_row, test_bank, stress_cycles)

        # Evict and wait
        evict_cache(wb)
        time.sleep(decay_ms / 1000.0)

        # Read
        ones_counts = []
        for i in range(16):
            val = wb.read(test_addr + i * 4)
            ones_counts.append(bin(val).count('1'))

        avg_ones = sum(ones_counts) / len(ones_counts)
        print(f"  {decay_ms:3d}ms: {avg_ones:.1f}/32 ones ({(32-avg_ones)/32*100:.0f}% decayed)")

        decay_curve.append({
            "decay_ms": decay_ms,
            "avg_ones": avg_ones,
            "decay_percent": (32 - avg_ones) / 32 * 100
        })

    results["tests"].append({
        "name": "decay_curve",
        "stress_cycles": stress_cycles,
        "results": decay_curve
    })

    # Test multi-level by combining stress + decay time
    print("\n=== Multi-Level Attempt ===")
    print("Combining different stress + decay for analog levels...")

    multilevel = []
    levels = [
        (0, 1),     # Level 0: no stress, minimal decay
        (50, 2),    # Level 1: medium stress, short decay
        (100, 5),   # Level 2: high stress, medium decay
        (200, 10),  # Level 3: very high stress, longer decay
    ]

    for stress, decay_ms in levels:
        issue_refresh(wb)
        time.sleep(0.01)

        for i in range(16):
            wb.write(test_addr + i * 4, pattern)

        if stress > 0:
            stress_row(wb, test_row, test_bank, stress)

        evict_cache(wb)
        time.sleep(decay_ms / 1000.0)

        ones_counts = []
        for i in range(16):
            val = wb.read(test_addr + i * 4)
            ones_counts.append(bin(val).count('1'))

        avg_ones = sum(ones_counts) / len(ones_counts)
        level_pct = avg_ones / 32 * 100
        print(f"  Stress={stress:3d}, Decay={decay_ms:2d}ms: {avg_ones:.1f}/32 ({level_pct:.0f}%)")

        multilevel.append({
            "stress_cycles": stress,
            "decay_ms": decay_ms,
            "avg_ones": avg_ones,
            "level_percent": level_pct
        })

    results["tests"].append({
        "name": "multilevel_attempt",
        "results": multilevel
    })

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    # Check if we got different levels
    levels_achieved = [r["avg_ones"] for r in multilevel]
    unique_levels = len(set([round(l) for l in levels_achieved]))

    if unique_levels > 1:
        print(f"MULTI-LEVEL ACHIEVED: {unique_levels} distinct levels")
        for i, r in enumerate(multilevel):
            print(f"  Level {i}: {r['level_percent']:.0f}% charge")
        results["multilevel_achieved"] = True
        results["num_levels"] = unique_levels
    else:
        print("Single level only - stress doesn't significantly affect decay rate")
        print("Decay is too fast even with minimal stress")
        results["multilevel_achieved"] = False

    results["conclusion"] = "stress_decay_test_complete"

    # Save
    results_path = '/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy/results/z1177_stress_decay_levels.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {results_path}")

    wb.close()
    print("Done!")

    return results


if __name__ == "__main__":
    main()
