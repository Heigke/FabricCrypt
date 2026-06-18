#!/usr/bin/env python3
"""
z1161: DDR3 Analog Multi-Level Sensing Test

Test partial charge decay and IDELAY-based analog sensing on Arty A7 DDR3.

Key concepts:
1. DDR3 cells store charge that decays over time (64ms refresh interval)
2. IDELAY2 allows sampling at different points (~78ps resolution)
3. By controlling refresh and sampling timing, we can detect analog charge levels
"""

import sys
import time
import json
from datetime import datetime

sys.path.insert(0, '/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy/venv/lib/python3.12/site-packages')

from litex.tools.litex_client import RemoteClient

# CSR addresses
DDRPHY_BASE = 0x800
DDRPHY_RST = DDRPHY_BASE + 0x00
DDRPHY_DLY_SEL = DDRPHY_BASE + 0x04
DDRPHY_RDLY_RST = DDRPHY_BASE + 0x14
DDRPHY_RDLY_INC = DDRPHY_BASE + 0x18

SDRAM_BASE = 0x2800
SDRAM_DFII_CONTROL = SDRAM_BASE + 0x00

DDR3_BASE = 0x40000000

# DFII control bits
DFII_SEL = 0x01
DFII_CKE = 0x02
DFII_ODT = 0x04
DFII_RESET_N = 0x08


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
    """Count differing bits between two 32-bit values"""
    xor = expected ^ actual
    return bin(xor).count('1')


def test_decay_with_idelay(wb, test_addr, pattern, delay_ms, idelay_taps):
    """Write pattern, wait, read with specific IDELAY setting"""
    # Write pattern
    wb.write(test_addr, pattern)

    # Wait for decay
    time.sleep(delay_ms / 1000.0)

    # Set IDELAY
    set_idelay(wb, idelay_taps)

    # Read back
    actual = wb.read(test_addr)
    errors = count_bit_errors(pattern, actual)

    return actual, errors


def main():
    print("=" * 60)
    print("z1161: DDR3 Analog Multi-Level Sensing Test")
    print("=" * 60)

    wb = RemoteClient()
    wb.open()
    print("Connected to Etherbone at 192.168.0.50:1234")

    results = {
        "experiment": "z1161_ddr3_analog",
        "timestamp": datetime.now().isoformat(),
        "tests": []
    }

    # Test addresses (spread across different banks/rows)
    test_addrs = [
        DDR3_BASE + 0x0000,
        DDR3_BASE + 0x1000,
        DDR3_BASE + 0x10000,
        DDR3_BASE + 0x100000,
    ]

    # Test patterns
    patterns = [
        (0xFFFFFFFF, "all_ones"),
        (0x00000000, "all_zeros"),
        (0xAAAAAAAA, "alternating_10"),
        (0x55555555, "alternating_01"),
        (0xFF00FF00, "byte_alt"),
    ]

    # ===========================================
    # Test 1: IDELAY sweep at normal refresh
    # ===========================================
    print("\n--- Test 1: IDELAY Sweep (normal refresh) ---")
    print("Testing sensitivity of reads to IDELAY timing...")

    test1_results = []
    for addr in test_addrs[:1]:  # Just first address for quick test
        for pattern, name in patterns:
            wb.write(addr, pattern)
            time.sleep(0.01)

            for taps in range(0, 32, 2):
                set_idelay(wb, taps)
                actual = wb.read(addr)
                errors = count_bit_errors(pattern, actual)
                delay_ps = taps * 78

                test1_results.append({
                    "pattern": name,
                    "idelay_taps": taps,
                    "delay_ps": delay_ps,
                    "expected": f"0x{pattern:08x}",
                    "actual": f"0x{actual:08x}",
                    "bit_errors": errors
                })

                if errors > 0:
                    print(f"  {name} @ tap {taps:2d} ({delay_ps:4d}ps): {errors} bit errors!")

    # Reset IDELAY
    set_idelay(wb, 0)

    error_count = sum(1 for r in test1_results if r["bit_errors"] > 0)
    print(f"Test 1 complete: {error_count}/{len(test1_results)} reads had errors")
    results["tests"].append({"name": "idelay_sweep", "results": test1_results})

    # ===========================================
    # Test 2: Rapid read/write stress test
    # ===========================================
    print("\n--- Test 2: Rapid Access Stress Test ---")
    print("Testing thermal/timing stability under stress...")

    test2_results = []
    stress_addr = DDR3_BASE + 0x2000
    stress_pattern = 0xDEADBEEF

    # Stress test - rapid writes and reads
    errors_at_tap = {t: 0 for t in range(0, 32, 4)}
    iterations = 100

    for iteration in range(iterations):
        # Write
        wb.write(stress_addr, stress_pattern)

        # Read at different IDELAY settings
        for taps in errors_at_tap.keys():
            set_idelay(wb, taps)
            actual = wb.read(stress_addr)
            if actual != stress_pattern:
                errors_at_tap[taps] += 1

    print("Errors per IDELAY tap after stress test:")
    for taps, err_count in errors_at_tap.items():
        print(f"  Tap {taps:2d}: {err_count}/{iterations} errors ({100*err_count/iterations:.1f}%)")
        test2_results.append({
            "idelay_taps": taps,
            "errors": err_count,
            "iterations": iterations
        })

    results["tests"].append({"name": "stress_test", "results": test2_results})

    # ===========================================
    # Test 3: Check DFII control for refresh manipulation
    # ===========================================
    print("\n--- Test 3: DFII Control Analysis ---")

    dfii_ctrl = wb.read(SDRAM_DFII_CONTROL)
    print(f"Current DFII control: 0x{dfii_ctrl:02x}")
    print(f"  SEL (software control): {bool(dfii_ctrl & DFII_SEL)}")
    print(f"  CKE (clock enable): {bool(dfii_ctrl & DFII_CKE)}")
    print(f"  ODT (on-die termination): {bool(dfii_ctrl & DFII_ODT)}")
    print(f"  RESET_N: {bool(dfii_ctrl & DFII_RESET_N)}")

    results["dfii_control"] = {
        "value": f"0x{dfii_ctrl:02x}",
        "sel": bool(dfii_ctrl & DFII_SEL),
        "cke": bool(dfii_ctrl & DFII_CKE),
        "odt": bool(dfii_ctrl & DFII_ODT),
        "reset_n": bool(dfii_ctrl & DFII_RESET_N)
    }

    # ===========================================
    # Test 4: Edge timing detection
    # ===========================================
    print("\n--- Test 4: Edge Timing Detection ---")
    print("Looking for sampling edges where data becomes unstable...")

    test4_results = []
    edge_addr = DDR3_BASE + 0x4000

    # Use high-frequency pattern most sensitive to timing
    edge_patterns = [
        (0xAAAAAAAA, "checkerboard_A"),
        (0x55555555, "checkerboard_5"),
    ]

    for pattern, name in edge_patterns:
        print(f"\n  Pattern: {name} (0x{pattern:08x})")
        wb.write(edge_addr, pattern)
        time.sleep(0.01)

        # Fine sweep through all 32 taps
        tap_errors = []
        for taps in range(32):
            set_idelay(wb, taps)

            # Read multiple times to detect marginal timing
            read_errors = 0
            reads = 10
            for _ in range(reads):
                actual = wb.read(edge_addr)
                if actual != pattern:
                    read_errors += 1

            error_rate = read_errors / reads
            tap_errors.append((taps, error_rate))

            if error_rate > 0:
                print(f"    Tap {taps:2d} ({taps*78:4d}ps): {error_rate*100:.0f}% error rate")

        test4_results.append({
            "pattern": name,
            "tap_errors": tap_errors
        })

    results["tests"].append({"name": "edge_timing", "results": test4_results})

    # Reset to safe default
    set_idelay(wb, 0)

    # ===========================================
    # Test 5: Analog value extraction attempt
    # ===========================================
    print("\n--- Test 5: Analog Value Extraction ---")
    print("Attempting to extract intermediate charge levels...")

    # The idea: sweep IDELAY during read window to detect voltage level
    # Higher charge = more stable across timing window
    # Lower charge = only stable at optimal tap

    analog_addr = DDR3_BASE + 0x8000

    # Write all 1s (high charge)
    wb.write(analog_addr, 0xFFFFFFFF)
    time.sleep(0.001)

    # Measure stability window (number of taps where read is correct)
    ones_stable_taps = 0
    for taps in range(32):
        set_idelay(wb, taps)
        if wb.read(analog_addr) == 0xFFFFFFFF:
            ones_stable_taps += 1

    # Write all 0s (no charge)
    wb.write(analog_addr, 0x00000000)
    time.sleep(0.001)

    zeros_stable_taps = 0
    for taps in range(32):
        set_idelay(wb, taps)
        if wb.read(analog_addr) == 0x00000000:
            zeros_stable_taps += 1

    print(f"  All 1s stable across {ones_stable_taps}/32 taps")
    print(f"  All 0s stable across {zeros_stable_taps}/32 taps")

    results["analog_test"] = {
        "ones_stable_taps": ones_stable_taps,
        "zeros_stable_taps": zeros_stable_taps,
        "notes": "Higher stability window indicates stronger signal"
    }

    # Reset
    set_idelay(wb, 0)

    # ===========================================
    # Summary
    # ===========================================
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    total_errors = sum(r["bit_errors"] for r in test1_results)
    stress_errors = sum(errors_at_tap.values())

    if total_errors == 0 and stress_errors == 0:
        print("All reads returned correct data across all IDELAY settings.")
        print("This indicates the DDR3 has a wide timing margin.")
        print("\nTo achieve TRUE analog sensing, we need to:")
        print("1. Disable auto-refresh (requires taking DFII into software control)")
        print("2. Wait for cell charge to decay (tens of milliseconds)")
        print("3. Then sample with IDELAY sweep to detect charge level")
        results["conclusion"] = "wide_timing_margin_need_refresh_control"
    else:
        print(f"Detected {total_errors + stress_errors} total errors!")
        print("Some IDELAY settings produce read errors - this is promising for analog sensing.")
        results["conclusion"] = "timing_edges_detected"

    # Save results
    results_path = '/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy/results/z1161_ddr3_analog.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {results_path}")

    wb.close()
    print("\nDone!")

    return results


if __name__ == "__main__":
    main()
