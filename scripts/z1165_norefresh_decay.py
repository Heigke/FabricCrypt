#!/usr/bin/env python3
"""
z1165: No-Refresh DDR3 Decay Test

TRUE decay observation with LiteDRAM's Refresher FSM disabled at design time.
This bitstream has with_refresh=False, so memory WILL decay without manual refresh.
"""

import sys
import time
import json
from datetime import datetime

sys.path.insert(0, '/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy/venv/lib/python3.12/site-packages')

from litex.tools.litex_client import RemoteClient

# CSR addresses (from csr.h)
MANUAL_REFRESH_BASE = 0x2800
MANUAL_REFRESH_TRIGGER = MANUAL_REFRESH_BASE + 0x00    # csrstorage_36
MANUAL_REFRESH_COUNT = MANUAL_REFRESH_BASE + 0x04      # csrstatus_37
MANUAL_REFRESH_DECAY_TIMER = MANUAL_REFRESH_BASE + 0x08  # csrstatus_38

SDRAM_BASE = 0x3000
DFII_CONTROL = SDRAM_BASE + 0x00           # csrstorage_57
DFII_PI0_COMMAND = SDRAM_BASE + 0x04       # csrstorage_58
DFII_PI0_COMMAND_ISSUE = SDRAM_BASE + 0x08 # csr_59
DFII_PI0_ADDRESS = SDRAM_BASE + 0x0c       # csrstorage_60
DFII_PI0_BADDRESS = SDRAM_BASE + 0x10      # csrstorage_61

DDRPHY_BASE = 0x800
DDRPHY_DLY_SEL = DDRPHY_BASE + 0x04        # csrstorage_40
DDRPHY_RDLY_RST = DDRPHY_BASE + 0x14       # csr_44
DDRPHY_RDLY_INC = DDRPHY_BASE + 0x18       # csr_45

DDR3_BASE = 0x40000000

# DFII control bits
DFII_SEL = 0x01       # Hardware control when set
DFII_CKE = 0x02       # Clock enable
DFII_ODT = 0x04       # On-die termination
DFII_RESET_N = 0x08   # Reset (active low)

# DFII command bits
CMD_CS = 0x01
CMD_WE = 0x02
CMD_CAS = 0x04
CMD_RAS = 0x08

# Derived commands
CMD_REFRESH = CMD_RAS | CMD_CAS | CMD_CS   # Auto refresh
CMD_PRECHARGE = CMD_RAS | CMD_WE | CMD_CS  # Precharge


def count_bit_errors(expected, actual):
    """Count differing bits"""
    return bin(expected ^ actual).count('1')


def issue_manual_refresh(wb):
    """Issue manual refresh via DFII in software mode"""
    # Store current mode
    current = wb.read(DFII_CONTROL)

    # Switch to software mode
    sw_mode = DFII_CKE | DFII_RESET_N  # 0x0A
    wb.write(DFII_CONTROL, sw_mode)
    time.sleep(0.001)

    # Precharge all banks
    wb.write(DFII_PI0_ADDRESS, 0x400)  # A10=1 for all banks
    wb.write(DFII_PI0_BADDRESS, 0)
    wb.write(DFII_PI0_COMMAND, CMD_PRECHARGE)
    wb.write(DFII_PI0_COMMAND_ISSUE, 1)
    time.sleep(0.0001)

    # Auto refresh
    wb.write(DFII_PI0_COMMAND, CMD_REFRESH)
    wb.write(DFII_PI0_COMMAND_ISSUE, 1)
    time.sleep(0.0001)

    # Restore mode
    wb.write(DFII_CONTROL, current)


def main():
    print("=" * 60)
    print("z1165: No-Refresh DDR3 Decay Test")
    print("=" * 60)
    print("This bitstream has with_refresh=False - memory WILL decay!")

    wb = RemoteClient()
    wb.open()
    print("Connected to Etherbone at 192.168.0.50:1234")

    results = {
        "experiment": "z1165_norefresh_decay",
        "timestamp": datetime.now().isoformat(),
        "bitstream": "build_norefresh_ddr3 (with_refresh=False)",
        "tests": []
    }

    # Read current state
    print("\n=== Initial State ===")
    dfii_ctrl = wb.read(DFII_CONTROL)
    decay_timer = wb.read(MANUAL_REFRESH_DECAY_TIMER)
    refresh_count = wb.read(MANUAL_REFRESH_COUNT)

    print(f"DFII Control: 0x{dfii_ctrl:02x}")
    print(f"  SEL (hw ctrl): {bool(dfii_ctrl & DFII_SEL)}")
    print(f"  CKE: {bool(dfii_ctrl & DFII_CKE)}")
    print(f"  RESET_N: {bool(dfii_ctrl & DFII_RESET_N)}")
    print(f"Decay Timer: {decay_timer} cycles")
    print(f"Manual Refresh Count: {refresh_count}")

    # Quick memory test
    test_addr = DDR3_BASE + 0x100000
    test_pattern = 0xDEADBEEF
    wb.write(test_addr, test_pattern)
    result = wb.read(test_addr)
    if result != test_pattern:
        print(f"ERROR: Memory not working! Wrote 0x{test_pattern:08x}, got 0x{result:08x}")
        wb.close()
        return

    print("Memory verified working.")

    # Issue some refreshes to stabilize memory
    print("\n=== Stabilizing Memory ===")
    for _ in range(10):
        issue_manual_refresh(wb)
    print("Issued 10 manual refreshes")

    # ===========================================
    # Test 1: Short-term decay (ms to seconds)
    # ===========================================
    print("\n=== Test 1: Short-Term Decay ===")
    print("Writing patterns and observing decay WITHOUT refresh...")

    decay_results = []
    decay_addr = DDR3_BASE + 0x200000

    # Test patterns - most susceptible to decay
    patterns = [
        (0xFFFFFFFF, "all_ones"),
        (0x00000000, "all_zeros"),
        (0xAAAAAAAA, "checkerboard"),
        (0x55555555, "inv_checkerboard"),
    ]

    # Short wait times first
    wait_times = [0.001, 0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]

    for pattern, name in patterns:
        print(f"\n  Pattern: {name} (0x{pattern:08x})")

        # Issue refresh before starting
        issue_manual_refresh(wb)

        # Write pattern to 64 words
        for offset in range(0, 256, 4):
            wb.write(decay_addr + offset, pattern)

        # Verify write succeeded
        errors = sum(1 for offset in range(0, 256, 4)
                    if wb.read(decay_addr + offset) != pattern)
        if errors > 0:
            print(f"    ERROR: Initial write has {errors}/64 errors!")
            continue

        print(f"    Initial write verified (64 words)")

        for wait in wait_times:
            # Re-write pattern
            issue_manual_refresh(wb)
            for offset in range(0, 256, 4):
                wb.write(decay_addr + offset, pattern)

            # Wait WITHOUT refresh (this is the decay period)
            print(f"    Wait {wait:6.3f}s...", end="", flush=True)
            time.sleep(wait)

            # Count errors
            bit_errors = 0
            word_errors = 0
            first_error_offset = None

            for offset in range(0, 256, 4):
                actual = wb.read(decay_addr + offset)
                if actual != pattern:
                    word_errors += 1
                    bit_errors += count_bit_errors(pattern, actual)
                    if first_error_offset is None:
                        first_error_offset = offset
                        print(f" DECAY @ +{offset}: 0x{pattern:08x}->0x{actual:08x}", end="")

            status = f" {word_errors}/64 words, {bit_errors} bits" if word_errors > 0 else " OK"
            print(status)

            decay_results.append({
                "pattern": name,
                "wait_seconds": wait,
                "word_errors": word_errors,
                "bit_errors": bit_errors
            })

            if bit_errors > 0:
                print(f"    *** DECAY DETECTED! ***")
                break

    results["tests"].append({
        "name": "short_term_decay",
        "results": decay_results
    })

    # ===========================================
    # Test 2: Longer decay at multiple addresses
    # ===========================================
    print("\n=== Test 2: Extended Decay (30s) ===")
    print("Testing extended decay at multiple memory regions...")

    extended_results = []
    test_regions = [
        DDR3_BASE + 0x000000,
        DDR3_BASE + 0x100000,
        DDR3_BASE + 0x200000,
        DDR3_BASE + 0x400000,
    ]

    pattern = 0xAAAAAAAA  # Checkerboard

    # Issue refresh, write pattern to all regions
    issue_manual_refresh(wb)
    for region in test_regions:
        for offset in range(0, 256, 4):
            wb.write(region + offset, pattern)

    print("  Written 0xAAAAAAAA to 4 regions (256B each)")

    # Wait 30 seconds without any refresh
    wait_time = 30.0
    print(f"  Waiting {wait_time}s without refresh...")
    start_timer = wb.read(MANUAL_REFRESH_DECAY_TIMER)
    time.sleep(wait_time)
    end_timer = wb.read(MANUAL_REFRESH_DECAY_TIMER)

    print(f"  Decay timer: {start_timer} -> {end_timer} (+{end_timer - start_timer} cycles)")

    # Check all regions
    total_word_errors = 0
    total_bit_errors = 0

    for i, region in enumerate(test_regions):
        word_errors = 0
        bit_errors = 0

        for offset in range(0, 256, 4):
            actual = wb.read(region + offset)
            if actual != pattern:
                word_errors += 1
                bit_errors += count_bit_errors(pattern, actual)

        status = f"{word_errors}/64 words, {bit_errors} bits"
        if word_errors > 0:
            status += " *** DECAY ***"
        print(f"  Region {i} (0x{region:08x}): {status}")

        total_word_errors += word_errors
        total_bit_errors += bit_errors

        extended_results.append({
            "region": f"0x{region:08x}",
            "word_errors": word_errors,
            "bit_errors": bit_errors
        })

    results["tests"].append({
        "name": "extended_decay_30s",
        "wait_seconds": wait_time,
        "decay_cycles": end_timer - start_timer,
        "results": extended_results
    })

    # ===========================================
    # Summary
    # ===========================================
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    total_decay = sum(r["bit_errors"] for r in decay_results)
    total_decay += total_bit_errors

    if total_decay > 0:
        print(f"DECAY DETECTED: {total_decay} total bit errors!")
        print("TRUE analog sensing is NOW possible!")
        print("We can use refresh timing for multi-level storage.")
        results["conclusion"] = "decay_detected_success"
        results["total_bit_errors"] = total_decay
    else:
        print("No decay observed even with refresh disabled.")
        print("Possible causes:")
        print("  1. DDR3 retention is very good (>30s)")
        print("  2. Temperature affects retention time")
        print("  3. Some cells are stronger than others")
        results["conclusion"] = "no_decay_strong_retention"

    # Save results
    results_path = '/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy/results/z1165_norefresh_decay.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {results_path}")

    wb.close()
    print("\nDone!")

    return results


if __name__ == "__main__":
    main()
