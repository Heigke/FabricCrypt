#!/usr/bin/env python3
"""
z1169: Proper DDR3 Decay Test with Correct Initialization

Key fix: Initialize DFII to 0x0B (SEL=1, CKE=1, RESET_N=1) before testing.
The no-refresh bitstream starts with CKE=0 which disables DDR3 clock.
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

DFII_SEL = 0x01
DFII_CKE = 0x02
DFII_ODT = 0x04
DFII_RESET_N = 0x08

# Correct modes
HW_MODE = DFII_SEL | DFII_CKE | DFII_RESET_N  # 0x0B
SW_MODE = DFII_CKE | DFII_RESET_N             # 0x0A

CMD_CS = 0x01
CMD_WE = 0x02
CMD_CAS = 0x04
CMD_RAS = 0x08
CMD_REFRESH = CMD_RAS | CMD_CAS | CMD_CS
CMD_PRECHARGE = CMD_RAS | CMD_WE | CMD_CS


def count_bit_errors(expected, actual):
    return bin(expected ^ actual).count('1')


def init_ddr3(wb):
    """Properly initialize DDR3 with CKE and RESET_N"""
    print("Initializing DDR3...")

    # Set hardware mode with CKE and RESET_N
    wb.write(DFII_CONTROL, HW_MODE)
    time.sleep(0.1)

    # Issue several refreshes to stabilize
    for _ in range(10):
        wb.write(DFII_CONTROL, SW_MODE)
        time.sleep(0.001)
        wb.write(DFII_PI0_ADDRESS, 0x400)
        wb.write(DFII_PI0_BADDRESS, 0)
        wb.write(DFII_PI0_COMMAND, CMD_PRECHARGE)
        wb.write(DFII_PI0_COMMAND_ISSUE, 1)
        time.sleep(0.0001)
        wb.write(DFII_PI0_COMMAND, CMD_REFRESH)
        wb.write(DFII_PI0_COMMAND_ISSUE, 1)
        time.sleep(0.0001)
        wb.write(DFII_CONTROL, HW_MODE)
        time.sleep(0.01)

    print("DDR3 initialized")


def issue_refresh(wb):
    """Issue manual refresh"""
    wb.write(DFII_CONTROL, SW_MODE)
    time.sleep(0.001)
    wb.write(DFII_PI0_ADDRESS, 0x400)
    wb.write(DFII_PI0_BADDRESS, 0)
    wb.write(DFII_PI0_COMMAND, CMD_PRECHARGE)
    wb.write(DFII_PI0_COMMAND_ISSUE, 1)
    time.sleep(0.0001)
    wb.write(DFII_PI0_COMMAND, CMD_REFRESH)
    wb.write(DFII_PI0_COMMAND_ISSUE, 1)
    time.sleep(0.0001)
    wb.write(DFII_CONTROL, HW_MODE)


def main():
    print("=" * 60)
    print("z1169: Proper DDR3 Decay Test")
    print("=" * 60)

    wb = RemoteClient()
    wb.open()
    print("Connected to Etherbone at 192.168.0.50:1234")

    results = {
        "experiment": "z1169_proper_decay_test",
        "timestamp": datetime.now().isoformat(),
        "bitstream": "no_refresh (with_refresh=False)",
        "tests": []
    }

    # Initialize DDR3 properly
    init_ddr3(wb)

    # Verify DFII state
    dfii = wb.read(DFII_CONTROL)
    print(f"DFII Control: 0x{dfii:02x} (should be 0x0B)")

    # Quick memory test
    print("\n=== Quick Memory Test ===")
    test_patterns = [0xDEADBEEF, 0xCAFEBABE, 0x12345678, 0xFFFFFFFF, 0x00000000, 0xAAAAAAAA]
    for pat in test_patterns:
        wb.write(DDR3_BASE, pat)
        result = wb.read(DDR3_BASE)
        status = "OK" if result == pat else f"FAIL (0x{result:08x})"
        print(f"  0x{pat:08x}: {status}")

    # Use spread addresses (1MB apart) to avoid L2 cache
    num_words = 16
    test_addrs = [DDR3_BASE + i * 0x100000 for i in range(num_words)]
    pattern = 0xAAAAAAAA

    print(f"\nUsing {num_words} addresses, 1MB apart")

    # Decay test
    print("\n=== Decay Test ===")
    print("Testing at wait times from 10s to 120s...")

    wait_times = [10, 15, 20, 25, 30, 40, 50, 60, 90, 120]
    decay_results = []

    for wait in wait_times:
        # Refresh and write
        issue_refresh(wb)
        for addr in test_addrs:
            wb.write(addr, pattern)

        # Verify write (quick check)
        sample = wb.read(test_addrs[0])
        if sample != pattern:
            print(f"  WRITE ERROR: Got 0x{sample:08x}")
            continue

        # Wait WITHOUT any refresh (this is the decay period)
        print(f"  {wait:3d}s: ", end="", flush=True)
        time.sleep(wait)

        # Read back
        word_errors = 0
        bit_errors = 0
        first_error = None

        for addr in test_addrs:
            actual = wb.read(addr)
            if actual != pattern:
                word_errors += 1
                errors = count_bit_errors(pattern, actual)
                bit_errors += errors
                if first_error is None:
                    first_error = (addr, actual, errors)

        pct = word_errors / num_words * 100

        if word_errors > 0:
            addr, actual, errs = first_error
            print(f"{word_errors}/{num_words} ({pct:.0f}%), {bit_errors} bits")
            print(f"       First error: 0x{addr:08x}: 0x{pattern:08x}->0x{actual:08x}")
        else:
            print("OK")

        decay_results.append({
            "wait_seconds": wait,
            "word_errors": word_errors,
            "bit_errors": bit_errors,
            "percent_decayed": pct
        })

        # If we see decay, continue to characterize it
        if word_errors > 0 and pct < 100:
            # Keep testing longer to see progression
            pass

    results["tests"].append({
        "name": "decay_characterization",
        "pattern": "0xAAAAAAAA",
        "num_addresses": num_words,
        "results": decay_results
    })

    # Find threshold
    threshold = None
    for r in decay_results:
        if r["bit_errors"] > 0:
            threshold = r["wait_seconds"]
            break

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    if threshold:
        print(f"Retention threshold: ~{threshold}s")
        print("Decay confirmed!")

        # Calculate decay rate (bits/second after threshold)
        decay_data = [r for r in decay_results if r["bit_errors"] > 0]
        if len(decay_data) >= 2:
            dt = decay_data[-1]["wait_seconds"] - decay_data[0]["wait_seconds"]
            db = decay_data[-1]["bit_errors"] - decay_data[0]["bit_errors"]
            if dt > 0:
                rate = db / dt
                print(f"Decay rate after threshold: ~{rate:.1f} bits/second")

        results["retention_threshold_s"] = threshold
        results["conclusion"] = "decay_confirmed"
    else:
        print("No decay observed up to 120s")
        print("DDR3 has excellent retention at room temperature")
        results["conclusion"] = "excellent_retention"

    # Save
    results_path = '/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy/results/z1169_proper_decay_test.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {results_path}")

    wb.close()
    print("Done!")

    return results


if __name__ == "__main__":
    main()
