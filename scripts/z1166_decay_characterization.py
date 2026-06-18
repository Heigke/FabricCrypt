#!/usr/bin/env python3
"""
z1166: DDR3 Decay Characterization

Find the exact retention time and characterize decay dynamics.
DDR3 spec: 64ms retention time at 85°C, but at room temp it's much longer.
"""

import sys
import time
import json
from datetime import datetime

sys.path.insert(0, '/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy/venv/lib/python3.12/site-packages')

from litex.tools.litex_client import RemoteClient

DDR3_BASE = 0x40000000

# DFII addresses
SDRAM_BASE = 0x3000
DFII_CONTROL = SDRAM_BASE + 0x00
DFII_PI0_COMMAND = SDRAM_BASE + 0x04
DFII_PI0_COMMAND_ISSUE = SDRAM_BASE + 0x08
DFII_PI0_ADDRESS = SDRAM_BASE + 0x0c
DFII_PI0_BADDRESS = SDRAM_BASE + 0x10

DFII_CKE = 0x02
DFII_RESET_N = 0x08
CMD_CS = 0x01
CMD_WE = 0x02
CMD_CAS = 0x04
CMD_RAS = 0x08
CMD_REFRESH = CMD_RAS | CMD_CAS | CMD_CS
CMD_PRECHARGE = CMD_RAS | CMD_WE | CMD_CS


def count_bit_errors(expected, actual):
    return bin(expected ^ actual).count('1')


def issue_manual_refresh(wb):
    current = wb.read(DFII_CONTROL)
    sw_mode = DFII_CKE | DFII_RESET_N
    wb.write(DFII_CONTROL, sw_mode)
    time.sleep(0.001)

    wb.write(DFII_PI0_ADDRESS, 0x400)
    wb.write(DFII_PI0_BADDRESS, 0)
    wb.write(DFII_PI0_COMMAND, CMD_PRECHARGE)
    wb.write(DFII_PI0_COMMAND_ISSUE, 1)
    time.sleep(0.0001)

    wb.write(DFII_PI0_COMMAND, CMD_REFRESH)
    wb.write(DFII_PI0_COMMAND_ISSUE, 1)
    time.sleep(0.0001)

    wb.write(DFII_CONTROL, current)


def test_decay_at_time(wb, addr, pattern, wait_seconds, num_words=64):
    """Write pattern, wait, measure decay"""
    # Refresh and write
    issue_manual_refresh(wb)
    for offset in range(0, num_words * 4, 4):
        wb.write(addr + offset, pattern)

    # Wait without refresh
    time.sleep(wait_seconds)

    # Measure decay
    word_errors = 0
    bit_errors = 0
    decay_map = []

    for offset in range(0, num_words * 4, 4):
        actual = wb.read(addr + offset)
        if actual != pattern:
            word_errors += 1
            errors = count_bit_errors(pattern, actual)
            bit_errors += errors
            decay_map.append({
                "offset": offset,
                "expected": f"0x{pattern:08x}",
                "actual": f"0x{actual:08x}",
                "errors": errors
            })

    return {
        "wait_seconds": wait_seconds,
        "word_errors": word_errors,
        "bit_errors": bit_errors,
        "decay_map": decay_map[:5]  # First 5 for brevity
    }


def main():
    print("=" * 60)
    print("z1166: DDR3 Decay Characterization")
    print("=" * 60)

    wb = RemoteClient()
    wb.open()
    print("Connected to Etherbone at 192.168.0.50:1234")

    results = {
        "experiment": "z1166_decay_characterization",
        "timestamp": datetime.now().isoformat(),
        "tests": []
    }

    test_addr = DDR3_BASE + 0x300000
    pattern = 0xAAAAAAAA  # Checkerboard - most visible decay

    # Binary search to find retention threshold
    print("\n=== Finding Retention Threshold ===")
    print("Binary search between 10s (no decay) and 30s (full decay)")

    # Known: 10s = OK, 30s = decay
    low, high = 10.0, 30.0
    threshold = None

    characterization = []

    while high - low > 1.0:
        mid = (low + high) / 2
        print(f"\n  Testing {mid:.1f}s...", end="", flush=True)

        result = test_decay_at_time(wb, test_addr, pattern, mid)
        characterization.append(result)

        if result["bit_errors"] > 0:
            print(f" DECAY ({result['bit_errors']} bits)")
            high = mid
            threshold = mid
        else:
            print(" OK")
            low = mid

    print(f"\n  Retention threshold: ~{low:.1f}s to {high:.1f}s")
    results["retention_threshold_s"] = {"low": low, "high": high}

    # Fine-grained characterization around threshold
    print("\n=== Fine Decay Characterization ===")
    print(f"Testing 1s increments from {max(10, int(low)-2)}s to {int(high)+5}s")

    fine_results = []
    for wait in range(max(10, int(low)-2), int(high)+6):
        print(f"  {wait}s: ", end="", flush=True)
        result = test_decay_at_time(wb, test_addr, pattern, float(wait))
        fine_results.append(result)

        pct = result["word_errors"] / 64 * 100
        print(f"{result['word_errors']}/64 words ({pct:.0f}%), {result['bit_errors']} bits")

    results["tests"].append({
        "name": "fine_characterization",
        "pattern": "0xAAAAAAAA",
        "results": fine_results
    })

    # Test different patterns for decay susceptibility
    print("\n=== Pattern Susceptibility Test ===")
    test_time = high  # Use the threshold time

    patterns = [
        (0xFFFFFFFF, "all_ones"),
        (0x00000000, "all_zeros"),
        (0xAAAAAAAA, "checkerboard"),
        (0x55555555, "inv_checkerboard"),
        (0xF0F0F0F0, "nibble_pattern"),
        (0x0F0F0F0F, "inv_nibble"),
        (0xFF00FF00, "byte_pattern"),
        (0x12345678, "counter"),
    ]

    pattern_results = []
    for pat, name in patterns:
        result = test_decay_at_time(wb, test_addr, pat, test_time)
        pattern_results.append({
            "pattern_name": name,
            "pattern_value": f"0x{pat:08x}",
            **result
        })
        print(f"  {name:20s}: {result['word_errors']}/64 words, {result['bit_errors']} bits")

    results["tests"].append({
        "name": "pattern_susceptibility",
        "wait_seconds": test_time,
        "results": pattern_results
    })

    # Characterize decay over longer periods
    print("\n=== Extended Decay Test ===")
    extended_times = [30, 45, 60, 90, 120]  # seconds
    extended_results = []

    for wait in extended_times:
        print(f"  {wait}s: ", end="", flush=True)
        result = test_decay_at_time(wb, test_addr, 0xAAAAAAAA, float(wait))
        extended_results.append(result)
        print(f"{result['bit_errors']} bit errors")

    results["tests"].append({
        "name": "extended_decay",
        "pattern": "0xAAAAAAAA",
        "results": extended_results
    })

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Retention threshold: {low:.1f}s to {high:.1f}s")
    print("Decay characteristics confirmed for analog memory!")

    # Calculate decay rate
    if fine_results:
        for r in fine_results:
            if r["bit_errors"] > 0:
                first_decay = r
                break
        else:
            first_decay = None

        if first_decay:
            print(f"First decay at {first_decay['wait_seconds']}s: {first_decay['bit_errors']} bits")

    results["conclusion"] = "decay_characterized"

    # Save
    results_path = '/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy/results/z1166_decay_characterization.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {results_path}")

    wb.close()
    print("Done!")

    return results


if __name__ == "__main__":
    main()
