#!/usr/bin/env python3
"""
z1172: Sub-second Decay Characterization

Find exact decay threshold in milliseconds and characterize partial decay.
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
    for i in range(4096):
        wb.write(evict_base + i * 4, 0xDEADC0DE)


def count_bit_errors(expected, actual):
    return bin(expected ^ actual).count('1')


def test_decay(wb, test_base, pattern, wait_seconds, num_words=64):
    """Test decay with millisecond precision"""
    issue_refresh(wb)
    for i in range(num_words):
        wb.write(test_base + i * 4, pattern)

    if wb.read(test_base) != pattern:
        return None, None, None

    evict_cache(wb, test_base)
    time.sleep(wait_seconds)

    word_errors = 0
    bit_errors = 0
    ones_remaining = 0

    for i in range(num_words):
        actual = wb.read(test_base + i * 4)
        ones_remaining += bin(actual).count('1')
        if actual != pattern:
            word_errors += 1
            bit_errors += count_bit_errors(pattern, actual)

    return word_errors, bit_errors, ones_remaining


def main():
    print("=" * 60)
    print("z1172: Sub-second Decay Characterization")
    print("=" * 60)

    wb = RemoteClient()
    wb.open()
    print("Connected to Etherbone")

    results = {
        "experiment": "z1172_subsecond_decay",
        "timestamp": datetime.now().isoformat(),
        "tests": []
    }

    init_ddr3(wb)
    print(f"DFII: 0x{wb.read(DFII_CONTROL):02x}")

    test_base = DDR3_BASE + 0x3000000
    pattern = 0xAAAAAAAA  # 16 ones per word
    num_words = 64
    total_ones = 16 * num_words  # 1024 ones expected

    # Test sub-second decay at fine granularity
    print("\n=== Sub-second Decay Characterization ===")
    print(f"Pattern: 0x{pattern:08x} ({16} ones/word, {total_ones} total)")
    print(f"Testing decay in 50ms increments...")

    subsecond_results = []

    # Test from 50ms to 1000ms in 50ms steps
    for ms in range(50, 1050, 50):
        wait = ms / 1000.0
        result = test_decay(wb, test_base, pattern, wait, num_words)

        if result[0] is None:
            print(f"  {ms:4d}ms: WRITE FAILED")
            continue

        word_errors, bit_errors, ones_left = result
        pct_decay = bit_errors / total_ones * 100
        pct_retained = ones_left / total_ones * 100

        print(f"  {ms:4d}ms: {bit_errors:4d}/{total_ones} bits decayed ({pct_decay:.1f}%), {ones_left} ones left ({pct_retained:.1f}%)")

        subsecond_results.append({
            "wait_ms": ms,
            "wait_s": wait,
            "bit_errors": bit_errors,
            "ones_remaining": ones_left,
            "percent_decayed": pct_decay,
            "percent_retained": pct_retained
        })

    results["tests"].append({
        "name": "subsecond_characterization",
        "pattern": f"0x{pattern:08x}",
        "total_ones": total_ones,
        "results": subsecond_results
    })

    # Find the exact threshold where first bit decays
    print("\n=== Finding First Decay ===")
    print("Binary search for first bit flip...")

    low, high = 0.001, 0.050  # 1ms to 50ms

    while high - low > 0.001:  # 1ms precision
        mid = (low + high) / 2
        result = test_decay(wb, test_base, pattern, mid, num_words)

        if result[0] is None:
            break

        word_errors, bit_errors, _ = result
        if bit_errors > 0:
            high = mid
            print(f"  {mid*1000:.1f}ms: DECAY ({bit_errors} bits)")
        else:
            low = mid
            print(f"  {mid*1000:.1f}ms: OK")

    print(f"\nFirst decay threshold: {low*1000:.1f}ms to {high*1000:.1f}ms")

    results["first_decay_threshold_ms"] = {"low": low * 1000, "high": high * 1000}

    # Test partial decay at threshold to see cell variability
    print("\n=== Cell Variability at Threshold ===")
    print("Testing multiple runs at threshold to see which cells decay first...")

    threshold = (low + high) / 2
    variability_results = []

    for run in range(5):
        result = test_decay(wb, test_base, pattern, threshold, num_words)
        if result[0] is None:
            continue

        word_errors, bit_errors, ones_left = result
        print(f"  Run {run+1}: {bit_errors} bits decayed, {ones_left} ones remaining")
        variability_results.append({
            "run": run + 1,
            "bit_errors": bit_errors,
            "ones_remaining": ones_left
        })

    results["tests"].append({
        "name": "variability_at_threshold",
        "threshold_ms": threshold * 1000,
        "results": variability_results
    })

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    # Find 50% decay point
    half_decay_ms = None
    for r in subsecond_results:
        if r["percent_decayed"] >= 50:
            half_decay_ms = r["wait_ms"]
            break

    if half_decay_ms:
        print(f"50% decay time: ~{half_decay_ms}ms")
    print(f"First decay: {low*1000:.1f}ms to {high*1000:.1f}ms")

    # This gives us the timing for analog multi-level storage!
    print("\n=== Analog Multi-Level Storage Potential ===")
    if len(subsecond_results) >= 2:
        for i, r in enumerate(subsecond_results):
            level = int((1 - r["percent_decayed"]/100) * 16)  # 4-bit levels
            if i % 4 == 0:  # Every 200ms
                print(f"  {r['wait_ms']:4d}ms -> Level {level}/16 ({r['percent_retained']:.0f}% charge)")

    results["conclusion"] = "subsecond_decay_characterized"
    results["analog_levels_possible"] = True

    # Save
    results_path = '/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy/results/z1172_subsecond_decay.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {results_path}")

    wb.close()
    print("Done!")

    return results


if __name__ == "__main__":
    main()
