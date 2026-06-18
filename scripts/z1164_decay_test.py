#!/usr/bin/env python3
"""
z1164: DDR3 Decay Test with Refresh Control

Test TRUE analog decay by controlling refresh via DFII software mode.
"""

import sys
import time
import json
from datetime import datetime

sys.path.insert(0, '/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy/venv/lib/python3.12/site-packages')

from litex.tools.litex_client import RemoteClient

# CSR addresses (from generated csr.h)
REFRESH_CTRL_BASE = 0x2800
REFRESH_ENABLE = REFRESH_CTRL_BASE + 0x00  # csrstorage_36
REFRESH_COUNT = REFRESH_CTRL_BASE + 0x04   # csrstatus_37
DECAY_TIMER = REFRESH_CTRL_BASE + 0x08     # csrstatus_38

SDRAM_BASE = 0x3000
DFII_CONTROL = SDRAM_BASE + 0x00           # csrstorage_57
DFII_PI0_COMMAND = SDRAM_BASE + 0x04       # csrstorage_58
DFII_PI0_COMMAND_ISSUE = SDRAM_BASE + 0x08 # csr_59
DFII_PI0_ADDRESS = SDRAM_BASE + 0x0c       # csrstorage_60
DFII_PI0_BADDRESS = SDRAM_BASE + 0x10      # csrstorage_61

DDRPHY_BASE = 0x800
DDRPHY_DLY_SEL = DDRPHY_BASE + 0x04
DDRPHY_RDLY_RST = DDRPHY_BASE + 0x14
DDRPHY_RDLY_INC = DDRPHY_BASE + 0x18

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


def set_idelay(wb, taps):
    """Set IDELAY to specified tap count (0-31)"""
    for dqs in range(2):
        wb.write(DDRPHY_DLY_SEL, 1 << dqs)
        wb.write(DDRPHY_RDLY_RST, 1)
        time.sleep(0.001)
        for _ in range(taps):
            wb.write(DDRPHY_RDLY_INC, 1)


def issue_refresh(wb):
    """Manually issue refresh command via DFII"""
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


def main():
    print("=" * 60)
    print("z1164: DDR3 Decay Test with Refresh Control")
    print("=" * 60)

    wb = RemoteClient()
    wb.open()
    print("Connected to Etherbone at 192.168.0.50:1234")

    results = {
        "experiment": "z1164_decay_test",
        "timestamp": datetime.now().isoformat(),
        "tests": []
    }

    # Read current state
    print("\n=== Initial State ===")
    dfii_ctrl = wb.read(DFII_CONTROL)
    refresh_en = wb.read(REFRESH_ENABLE)
    print(f"DFII Control: 0x{dfii_ctrl:02x}")
    print(f"  SEL (hw ctrl): {bool(dfii_ctrl & DFII_SEL)}")
    print(f"  CKE: {bool(dfii_ctrl & DFII_CKE)}")
    print(f"  RESET_N: {bool(dfii_ctrl & DFII_RESET_N)}")
    print(f"Refresh Enable: {refresh_en}")

    # Test memory works
    test_addr = DDR3_BASE + 0x100000
    test_pattern = 0xDEADBEEF
    wb.write(test_addr, test_pattern)
    if wb.read(test_addr) != test_pattern:
        print("ERROR: Memory not working!")
        wb.close()
        return

    print("Memory verified working.")

    # ===========================================
    # Test 1: Software mode control
    # ===========================================
    print("\n=== Test 1: DFII Software Mode ===")

    # Switch to software mode (clear SEL, keep CKE and RESET_N)
    sw_mode = DFII_CKE | DFII_RESET_N  # 0x0A
    hw_mode = DFII_SEL | DFII_CKE | DFII_RESET_N  # 0x0B

    print(f"Switching to software mode (0x{sw_mode:02x})...")
    wb.write(DFII_CONTROL, sw_mode)
    time.sleep(0.01)

    check = wb.read(DFII_CONTROL)
    print(f"DFII Control now: 0x{check:02x}")

    # Test memory still works
    wb.write(test_addr, 0x12345678)
    result = wb.read(test_addr)
    print(f"Memory test in SW mode: {'OK' if result == 0x12345678 else 'FAIL'}")

    # Issue manual refresh
    print("Issuing manual refresh...")
    issue_refresh(wb)
    print("Manual refresh issued.")

    # Switch back to hardware mode
    print(f"Switching back to hardware mode (0x{hw_mode:02x})...")
    wb.write(DFII_CONTROL, hw_mode)
    time.sleep(0.01)

    results["tests"].append({
        "name": "dfii_software_mode",
        "sw_mode_value": f"0x{sw_mode:02x}",
        "manual_refresh": True,
        "memory_ok": result == 0x12345678
    })

    # ===========================================
    # Test 2: Decay observation
    # ===========================================
    print("\n=== Test 2: Decay Observation ===")
    print("Writing patterns and observing decay in software mode...")

    decay_results = []
    decay_addr = DDR3_BASE + 0x200000

    # Test patterns
    patterns = [
        (0xFFFFFFFF, "all_ones"),
        (0x00000000, "all_zeros"),
        (0xAAAAAAAA, "checkerboard"),
    ]

    # Switch to software mode
    wb.write(DFII_CONTROL, sw_mode)
    time.sleep(0.01)

    for pattern, name in patterns:
        print(f"\n  Pattern: {name} (0x{pattern:08x})")

        # Write pattern to multiple addresses
        for offset in range(0, 64, 4):
            wb.write(decay_addr + offset, pattern)

        # Verify write
        errors = sum(1 for offset in range(0, 64, 4)
                    if wb.read(decay_addr + offset) != pattern)
        print(f"    Initial write: {errors}/16 errors")

        # Now wait WITHOUT issuing refresh
        wait_times = [0.01, 0.1, 1.0, 5.0, 10.0, 30.0]

        for wait in wait_times:
            # Re-write pattern
            for offset in range(0, 64, 4):
                wb.write(decay_addr + offset, pattern)

            # Wait (no refresh)
            print(f"    Waiting {wait}s without refresh...", end="", flush=True)
            time.sleep(wait)

            # Count errors
            bit_errors = 0
            word_errors = 0
            for offset in range(0, 64, 4):
                actual = wb.read(decay_addr + offset)
                if actual != pattern:
                    word_errors += 1
                    bit_errors += count_bit_errors(pattern, actual)

            print(f" {word_errors}/16 words, {bit_errors} bits corrupted")

            decay_results.append({
                "pattern": name,
                "wait_seconds": wait,
                "word_errors": word_errors,
                "bit_errors": bit_errors
            })

            if bit_errors > 0:
                print(f"    DECAY DETECTED!")
                break

        # Issue refresh to restore
        issue_refresh(wb)

    # Switch back to hardware mode
    wb.write(DFII_CONTROL, hw_mode)

    results["tests"].append({
        "name": "decay_observation",
        "results": decay_results
    })

    # ===========================================
    # Test 3: Timing-based analog sensing
    # ===========================================
    print("\n=== Test 3: IDELAY + Decay ===")
    print("Testing if IDELAY affects decay visibility...")

    # Switch to software mode
    wb.write(DFII_CONTROL, sw_mode)

    timing_results = []
    timing_addr = DDR3_BASE + 0x300000
    timing_pattern = 0xAAAAAAAA

    # Write pattern
    wb.write(timing_addr, timing_pattern)

    # Wait for some potential decay
    wait_time = 5.0
    print(f"  Waiting {wait_time}s without refresh...")
    time.sleep(wait_time)

    # Try reading at different IDELAY settings
    print("  Reading at different IDELAY taps:")
    for taps in range(0, 32, 4):
        set_idelay(wb, taps)
        actual = wb.read(timing_addr)
        errors = count_bit_errors(timing_pattern, actual)
        status = "OK" if errors == 0 else f"{errors} bit errors"
        print(f"    Tap {taps:2d} (~{taps*78}ps): 0x{actual:08x} - {status}")
        timing_results.append({
            "tap": taps,
            "delay_ps": taps * 78,
            "value": f"0x{actual:08x}",
            "bit_errors": errors
        })

    # Reset IDELAY
    set_idelay(wb, 0)

    # Switch back to hardware mode
    wb.write(DFII_CONTROL, hw_mode)

    results["tests"].append({
        "name": "idelay_decay",
        "wait_seconds": wait_time,
        "results": timing_results
    })

    # ===========================================
    # Summary
    # ===========================================
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    total_decay_errors = sum(r["bit_errors"] for r in decay_results)

    if total_decay_errors > 0:
        print(f"DECAY DETECTED: {total_decay_errors} total bit errors!")
        print("TRUE analog sensing is possible via software-controlled refresh.")
        results["conclusion"] = "decay_detected"
    else:
        print("No decay observed even in software mode.")
        print("The LiteDRAM controller may still be issuing refresh commands.")
        print("Need to further investigate the refresh path.")
        results["conclusion"] = "no_decay_controller_refresh"

    # Save results
    results_path = '/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy/results/z1164_decay_test.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {results_path}")

    wb.close()
    print("\nDone!")

    return results


if __name__ == "__main__":
    main()
