#!/usr/bin/env python3
"""
z1162: DDR3 Charge Decay Test

Test actual charge decay by controlling refresh timing.
DDR3 cells need refresh every 64ms - we'll test decay behavior.
"""

import sys
import time
import json
from datetime import datetime

sys.path.insert(0, '/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy/venv/lib/python3.12/site-packages')

from litex.tools.litex_client import RemoteClient

# CSR addresses
DDRPHY_BASE = 0x800
DDRPHY_DLY_SEL = DDRPHY_BASE + 0x04
DDRPHY_RDLY_RST = DDRPHY_BASE + 0x14
DDRPHY_RDLY_INC = DDRPHY_BASE + 0x18

SDRAM_BASE = 0x2800
SDRAM_DFII_CONTROL = SDRAM_BASE + 0x00
SDRAM_PI0_COMMAND = SDRAM_BASE + 0x04
SDRAM_PI0_COMMAND_ISSUE = SDRAM_BASE + 0x08
SDRAM_PI0_ADDRESS = SDRAM_BASE + 0x0c
SDRAM_PI0_BADDRESS = SDRAM_BASE + 0x10

DDR3_BASE = 0x40000000

# DFII control bits
DFII_SEL = 0x01
DFII_CKE = 0x02
DFII_ODT = 0x04
DFII_RESET_N = 0x08

# DDR3 commands
CMD_NOP = 0x00
CMD_PRECHARGE = 0x02  # WE
CMD_REFRESH = 0x04    # CAS
CMD_ACTIVATE = 0x08   # RAS
CMD_READ = 0x04 | 0x01  # CAS + CS
CMD_WRITE = 0x04 | 0x02 | 0x01  # CAS + WE + CS


def set_idelay(wb, taps):
    """Set IDELAY to specified tap count (0-31)"""
    for dqs in range(2):
        wb.write(DDRPHY_DLY_SEL, 1 << dqs)
        wb.write(DDRPHY_RDLY_RST, 1)
        time.sleep(0.001)
        for _ in range(taps):
            wb.write(DDRPHY_RDLY_INC, 1)
            time.sleep(0.0001)


def count_bit_errors(expected, actual):
    """Count differing bits"""
    return bin(expected ^ actual).count('1')


def issue_refresh(wb):
    """Issue manual refresh command"""
    # This is the low-level way to issue refresh
    # The LiteDRAM controller normally handles this automatically
    pass  # Controller handles refresh automatically


def main():
    print("=" * 60)
    print("z1162: DDR3 Charge Decay Test")
    print("=" * 60)

    wb = RemoteClient()
    wb.open()
    print("Connected to Etherbone at 192.168.0.50:1234")

    results = {
        "experiment": "z1162_ddr3_decay",
        "timestamp": datetime.now().isoformat(),
        "tests": []
    }

    # Check current DFII state
    dfii_ctrl = wb.read(SDRAM_DFII_CONTROL)
    print(f"\nCurrent DFII control: 0x{dfii_ctrl:02x}")

    # ===========================================
    # Test 1: Natural decay observation
    # ===========================================
    print("\n--- Test 1: Natural Decay Observation ---")
    print("Writing patterns and observing stability over time...")
    print("(Note: LiteDRAM controller handles refresh automatically)")

    test1_results = []
    test_addr = DDR3_BASE + 0x10000

    # Test patterns
    patterns = [
        (0xFFFFFFFF, "all_ones"),
        (0x00000000, "all_zeros"),
        (0xAAAAAAAA, "checkerboard"),
    ]

    # Test over time intervals
    time_intervals = [0, 0.001, 0.01, 0.1, 1.0, 5.0]  # seconds

    for pattern, name in patterns:
        print(f"\n  Pattern: {name}")
        for wait_time in time_intervals:
            # Write pattern
            wb.write(test_addr, pattern)

            # Wait
            if wait_time > 0:
                time.sleep(wait_time)

            # Read back
            actual = wb.read(test_addr)
            errors = count_bit_errors(pattern, actual)

            result = {
                "pattern": name,
                "wait_seconds": wait_time,
                "expected": f"0x{pattern:08x}",
                "actual": f"0x{actual:08x}",
                "bit_errors": errors
            }
            test1_results.append(result)

            status = "OK" if errors == 0 else f"ERROR ({errors} bits)"
            print(f"    After {wait_time:5.3f}s: {status}")

    results["tests"].append({"name": "natural_decay", "results": test1_results})

    # ===========================================
    # Test 2: Read stability window measurement
    # ===========================================
    print("\n--- Test 2: Read Stability Window ---")
    print("Measuring IDELAY window where reads are stable...")

    test2_results = []

    for pattern, name in patterns:
        wb.write(test_addr, pattern)
        time.sleep(0.01)

        stable_start = -1
        stable_end = -1
        stability_map = []

        for tap in range(32):
            set_idelay(wb, tap)

            # Multiple reads to check stability
            stable = True
            for _ in range(5):
                if wb.read(test_addr) != pattern:
                    stable = False
                    break

            stability_map.append(stable)
            if stable and stable_start < 0:
                stable_start = tap
            if stable:
                stable_end = tap

        window_size = stable_end - stable_start + 1 if stable_start >= 0 else 0
        window_ps = window_size * 78

        print(f"  {name}: taps {stable_start}-{stable_end} ({window_size} taps = {window_ps}ps)")

        test2_results.append({
            "pattern": name,
            "stable_start": stable_start,
            "stable_end": stable_end,
            "window_taps": window_size,
            "window_ps": window_ps,
            "stability_map": stability_map
        })

    results["tests"].append({"name": "stability_window", "results": test2_results})

    # Reset IDELAY
    set_idelay(wb, 0)

    # ===========================================
    # Test 3: Write-Read timing sensitivity
    # ===========================================
    print("\n--- Test 3: Write-Read Timing Sensitivity ---")
    print("Testing reads immediately after writes vs delayed...")

    test3_results = []
    timing_addr = DDR3_BASE + 0x20000

    for pattern, name in patterns:
        # Immediate read after write
        wb.write(timing_addr, pattern)
        immediate = wb.read(timing_addr)
        immediate_err = count_bit_errors(pattern, immediate)

        # Delayed reads
        wb.write(timing_addr, pattern)
        time.sleep(0.1)
        delayed = wb.read(timing_addr)
        delayed_err = count_bit_errors(pattern, delayed)

        print(f"  {name}: immediate={immediate_err} errors, 100ms delayed={delayed_err} errors")

        test3_results.append({
            "pattern": name,
            "immediate_errors": immediate_err,
            "delayed_errors": delayed_err
        })

    results["tests"].append({"name": "timing_sensitivity", "results": test3_results})

    # ===========================================
    # Test 4: Pattern interference test
    # ===========================================
    print("\n--- Test 4: Pattern Interference Test ---")
    print("Testing if adjacent writes affect reads (analog crosstalk)...")

    test4_results = []
    base_addr = DDR3_BASE + 0x30000

    # Write alternating patterns to adjacent addresses
    for offset in range(0, 64, 4):
        pattern = 0xFFFFFFFF if (offset // 4) % 2 == 0 else 0x00000000
        wb.write(base_addr + offset, pattern)

    time.sleep(0.01)

    # Read back and check
    errors = 0
    for offset in range(0, 64, 4):
        expected = 0xFFFFFFFF if (offset // 4) % 2 == 0 else 0x00000000
        actual = wb.read(base_addr + offset)
        if actual != expected:
            errors += 1
            print(f"  Offset {offset}: expected 0x{expected:08x}, got 0x{actual:08x}")

    print(f"  Pattern interference: {errors}/16 addresses affected")
    test4_results.append({"errors": errors, "total": 16})
    results["tests"].append({"name": "interference", "results": test4_results})

    # ===========================================
    # Test 5: Partial charge simulation via early readback
    # ===========================================
    print("\n--- Test 5: Partial Charge via Early Readback ---")
    print("Reading during write (not possible via memory interface)...")
    print("This would require direct DFI control - skipping for now.")

    # ===========================================
    # Test 6: Row hammer effect test (might show analog effects)
    # ===========================================
    print("\n--- Test 6: Row Hammer Sensitivity Test ---")
    print("Testing if repeated access to one row affects adjacent rows...")

    test6_results = []

    # Use different rows (rows are 8KB apart in this configuration)
    victim_addr = DDR3_BASE + 0x40000
    aggressor_addr = DDR3_BASE + 0x42000  # Adjacent row

    # Write victim pattern
    victim_pattern = 0xAAAAAAAA
    for offset in range(0, 256, 4):
        wb.write(victim_addr + offset, victim_pattern)

    # Hammer the aggressor row
    hammer_count = 10000
    print(f"  Hammering aggressor row {hammer_count} times...")
    for i in range(hammer_count):
        wb.read(aggressor_addr)
        if i % 1000 == 0:
            # Check victim occasionally
            victim_check = wb.read(victim_addr)
            if victim_check != victim_pattern:
                print(f"    Victim corrupted after {i} hammers!")
                break

    # Check victim after hammering
    victim_errors = 0
    for offset in range(0, 256, 4):
        actual = wb.read(victim_addr + offset)
        if actual != victim_pattern:
            victim_errors += 1

    print(f"  Victim errors after row hammer: {victim_errors}/64 words")
    test6_results.append({
        "hammer_count": hammer_count,
        "victim_errors": victim_errors
    })
    results["tests"].append({"name": "row_hammer", "results": test6_results})

    # ===========================================
    # Summary
    # ===========================================
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    total_decay_errors = sum(r["bit_errors"] for r in test1_results)
    avg_window = sum(r["window_taps"] for r in test2_results) / len(test2_results)

    print(f"Natural decay errors: {total_decay_errors}")
    print(f"Average stability window: {avg_window:.1f} taps ({avg_window*78:.0f}ps)")

    if total_decay_errors == 0:
        print("\nNo decay observed - refresh is working correctly.")
        print("\nFor TRUE analog multi-level sensing, options are:")
        print("1. Modify LiteDRAM to expose refresh control via CSR")
        print("2. Use DFI direct control mode to manually control refresh")
        print("3. Build custom PHY with finer timing control")
        results["conclusion"] = "no_decay_refresh_active"
    else:
        print(f"\nDecay detected! {total_decay_errors} bit errors observed.")
        results["conclusion"] = "decay_detected"

    # Save results
    results_path = '/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy/results/z1162_ddr3_decay.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {results_path}")

    wb.close()
    print("\nDone!")

    return results


if __name__ == "__main__":
    main()
