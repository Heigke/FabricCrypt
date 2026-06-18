#!/usr/bin/env python3
"""
z1171: Decay Verification - Confirm ones→zeros and find exact threshold
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

CMD_CS = 0x01
CMD_WE = 0x02
CMD_CAS = 0x04
CMD_RAS = 0x08
CMD_REFRESH = CMD_RAS | CMD_CAS | CMD_CS
CMD_PRECHARGE = CMD_RAS | CMD_WE | CMD_CS


def init_ddr3(wb):
    wb.write(DFII_CONTROL, HW_MODE)
    time.sleep(0.1)
    for _ in range(10):
        issue_refresh(wb)


def issue_refresh(wb):
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


def evict_cache(wb, avoid_region):
    evict_base = DDR3_BASE + 0x4000000
    if avoid_region >= evict_base and avoid_region < evict_base + 0x100000:
        evict_base = DDR3_BASE + 0x5000000
    for i in range(4096):
        wb.write(evict_base + i * 4, 0xDEADC0DE)


def count_bit_errors(expected, actual):
    return bin(expected ^ actual).count('1')


def test_decay(wb, test_base, pattern, wait_seconds, num_words=64):
    """Test decay for a specific pattern and wait time"""
    # Refresh and write
    issue_refresh(wb)
    for i in range(num_words):
        wb.write(test_base + i * 4, pattern)

    # Verify write
    if wb.read(test_base) != pattern:
        return None, None  # Write failed

    # Evict cache
    evict_cache(wb, test_base)

    # Wait
    time.sleep(wait_seconds)

    # Read back
    word_errors = 0
    bit_errors = 0
    read_values = []

    for i in range(num_words):
        actual = wb.read(test_base + i * 4)
        if actual != pattern:
            word_errors += 1
            bit_errors += count_bit_errors(pattern, actual)
        if i < 4:
            read_values.append(actual)

    return word_errors, bit_errors, read_values


def main():
    print("=" * 60)
    print("z1171: Decay Verification")
    print("=" * 60)

    wb = RemoteClient()
    wb.open()
    print("Connected to Etherbone")

    results = {
        "experiment": "z1171_decay_verification",
        "timestamp": datetime.now().isoformat(),
        "tests": []
    }

    init_ddr3(wb)
    print(f"DFII: 0x{wb.read(DFII_CONTROL):02x}")

    test_base = DDR3_BASE + 0x2000000

    # Test 1: Verify ones→zeros hypothesis
    print("\n=== Test 1: Pattern Comparison at 15s ===")
    print("Testing which bits decay...")

    patterns = [
        (0x00000000, "all_zeros"),
        (0xFFFFFFFF, "all_ones"),
        (0xAAAAAAAA, "checkerboard"),
        (0x55555555, "inv_checker"),
    ]

    pattern_results = []
    for pat, name in patterns:
        result = test_decay(wb, test_base, pat, 15.0)
        if result[0] is None:
            print(f"  {name:15s}: WRITE FAILED")
            continue

        word_errors, bit_errors, read_vals = result
        pct = word_errors / 64 * 100

        # Analyze what happened
        sample_val = read_vals[0]
        ones_in_pattern = bin(pat).count('1')
        ones_in_result = bin(sample_val).count('1')

        print(f"  {name:15s}: {word_errors}/64 words, {bit_errors} bits")
        print(f"       Pattern: 0x{pat:08x} ({ones_in_pattern} ones)")
        print(f"       Result:  0x{sample_val:08x} ({ones_in_result} ones)")

        pattern_results.append({
            "pattern": name,
            "pattern_value": f"0x{pat:08x}",
            "ones_in_pattern": ones_in_pattern,
            "sample_result": f"0x{sample_val:08x}",
            "ones_in_result": ones_in_result,
            "word_errors": word_errors,
            "bit_errors": bit_errors
        })

    results["tests"].append({
        "name": "pattern_comparison",
        "wait_seconds": 15,
        "results": pattern_results
    })

    # Test 2: Find exact threshold with binary search
    print("\n=== Test 2: Find Decay Threshold ===")
    print("Binary search for first decay...")

    pattern = 0xAAAAAAAA
    low, high = 0.5, 15.0

    while high - low > 0.5:
        mid = (low + high) / 2
        result = test_decay(wb, test_base, pattern, mid)
        if result[0] is None:
            print(f"  {mid:.1f}s: WRITE FAILED")
            break

        word_errors, bit_errors, _ = result
        if bit_errors > 0:
            high = mid
            print(f"  {mid:.1f}s: DECAY ({bit_errors} bits)")
        else:
            low = mid
            print(f"  {mid:.1f}s: OK")

    print(f"\nThreshold: between {low:.1f}s and {high:.1f}s")

    results["threshold_range"] = {"low_s": low, "high_s": high}

    # Test 3: Fine characterization around threshold
    print("\n=== Test 3: Fine Characterization ===")

    fine_results = []
    for wait in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
        result = test_decay(wb, test_base, pattern, float(wait))
        if result[0] is None:
            continue

        word_errors, bit_errors, _ = result
        pct = word_errors / 64 * 100
        print(f"  {wait:2d}s: {word_errors}/64 ({pct:.0f}%), {bit_errors} bits")

        fine_results.append({
            "wait_seconds": wait,
            "word_errors": word_errors,
            "bit_errors": bit_errors
        })

    results["tests"].append({
        "name": "fine_characterization",
        "results": fine_results
    })

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    # Analyze ones→zeros
    zeros_result = None
    ones_result = None
    for r in pattern_results:
        if r["pattern"] == "all_zeros":
            zeros_result = r
        elif r["pattern"] == "all_ones":
            ones_result = r

    if zeros_result and ones_result:
        if zeros_result["bit_errors"] == 0 and ones_result["bit_errors"] > 0:
            print("CONFIRMED: Ones decay to zeros (DRAM capacitor discharge)")
            results["decay_direction"] = "ones_to_zeros"
        elif zeros_result["bit_errors"] > 0 and ones_result["bit_errors"] == 0:
            print("CONFIRMED: Zeros decay to ones (unexpected)")
            results["decay_direction"] = "zeros_to_ones"
        else:
            print("Both patterns decay (or neither)")

    print(f"Retention threshold: {low:.1f}s to {high:.1f}s")
    results["conclusion"] = "decay_characterized"

    # Save
    results_path = '/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy/results/z1171_decay_verification.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {results_path}")

    wb.close()
    print("Done!")

    return results


if __name__ == "__main__":
    main()
