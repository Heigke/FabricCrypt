#!/usr/bin/env python3
"""
z1170: Simple DDR3 Decay Test

Use contiguous addresses and force cache eviction by writing to
a second, distant region before reading back.
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
DFII_RESET_N = 0x08

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
    """Initialize DDR3 with CKE and RESET_N"""
    wb.write(DFII_CONTROL, HW_MODE)
    time.sleep(0.1)
    for _ in range(10):
        issue_refresh(wb)
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


def evict_cache(wb, test_region):
    """Evict cache by accessing distant memory region"""
    # L2 cache is 8KB. Write 16KB to a distant region to ensure eviction.
    evict_base = DDR3_BASE + 0x4000000  # 64MB away
    if test_region == evict_base:
        evict_base = DDR3_BASE + 0x5000000

    for i in range(4096):  # 16KB
        wb.write(evict_base + i * 4, 0xDEADC0DE)


def main():
    print("=" * 60)
    print("z1170: Simple DDR3 Decay Test")
    print("=" * 60)

    wb = RemoteClient()
    wb.open()
    print("Connected to Etherbone")

    results = {
        "experiment": "z1170_simple_decay_test",
        "timestamp": datetime.now().isoformat(),
        "tests": []
    }

    init_ddr3(wb)

    # Verify DFII
    dfii = wb.read(DFII_CONTROL)
    print(f"DFII Control: 0x{dfii:02x}")

    # Test region - contiguous 256 bytes
    test_base = DDR3_BASE + 0x1000000  # 16MB offset
    num_words = 64
    pattern = 0xAAAAAAAA

    # Quick verification that memory works
    print("\n=== Quick Memory Test ===")
    issue_refresh(wb)
    for i in range(8):
        addr = test_base + i * 4
        wb.write(addr, pattern)
        result = wb.read(addr)
        status = "OK" if result == pattern else f"FAIL (0x{result:08x})"
        print(f"  0x{addr:08x}: {status}")

    # Decay test approach:
    # 1. Write pattern to test region
    # 2. Verify write succeeded
    # 3. Evict cache (write to distant region)
    # 4. Wait for decay
    # 5. Read back test region

    print("\n=== Decay Test ===")
    print("Write → Evict cache → Wait → Read")

    wait_times = [10, 20, 30, 45, 60, 90, 120]
    decay_results = []

    for wait in wait_times:
        # 1. Issue refresh
        issue_refresh(wb)

        # 2. Write pattern
        for i in range(num_words):
            wb.write(test_base + i * 4, pattern)

        # 3. Verify write (sample a few)
        errors_pre = 0
        for i in [0, num_words//2, num_words-1]:
            if wb.read(test_base + i * 4) != pattern:
                errors_pre += 1

        if errors_pre > 0:
            print(f"  {wait:3d}s: WRITE VERIFY FAILED ({errors_pre}/3 samples)")
            continue

        # 4. Evict cache
        evict_cache(wb, test_base)

        # 5. Wait for decay (no refresh!)
        print(f"  {wait:3d}s: waiting...", end="", flush=True)
        time.sleep(wait)

        # 6. Read back
        word_errors = 0
        bit_errors = 0
        sample_errors = []

        for i in range(num_words):
            addr = test_base + i * 4
            actual = wb.read(addr)
            if actual != pattern:
                word_errors += 1
                bits = count_bit_errors(pattern, actual)
                bit_errors += bits
                if len(sample_errors) < 3:
                    sample_errors.append((i, actual, bits))

        pct = word_errors / num_words * 100

        if word_errors > 0:
            print(f" {word_errors}/{num_words} ({pct:.0f}%), {bit_errors} bits")
            for i, val, bits in sample_errors:
                print(f"       Word {i}: 0x{val:08x} ({bits} bits)")
        else:
            print(" OK")

        decay_results.append({
            "wait_seconds": wait,
            "word_errors": word_errors,
            "bit_errors": bit_errors,
            "percent_decayed": pct
        })

    results["tests"].append({
        "name": "decay_with_cache_eviction",
        "pattern": "0xAAAAAAAA",
        "num_words": num_words,
        "results": decay_results
    })

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    threshold = None
    for r in decay_results:
        if r["bit_errors"] > 0:
            threshold = r["wait_seconds"]
            break

    if threshold:
        print(f"Retention threshold: ~{threshold}s")
        print("TRUE analog decay observed!")
        results["retention_threshold_s"] = threshold
        results["conclusion"] = "decay_confirmed"
    else:
        print("No decay observed up to 120s")
        print("DDR3 has excellent retention at room temperature")
        print("(Or decay happens at longer timescales)")
        results["conclusion"] = "excellent_retention"

    # Save
    results_path = '/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy/results/z1170_simple_decay_test.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {results_path}")

    wb.close()
    print("Done!")

    return results


if __name__ == "__main__":
    main()
