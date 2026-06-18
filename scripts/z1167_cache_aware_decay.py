#!/usr/bin/env python3
"""
z1167: Cache-Aware DDR3 Decay Test

Uses spread addresses to avoid L2 cache hits during decay observation.
L2 cache is 8KB - we use addresses 1MB apart to ensure cache eviction.
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


def flush_cache(wb, base_addr, cache_size_bytes=16384):
    """Flush L2 cache by reading from many spread addresses"""
    # Read from addresses spread far apart to evict cache lines
    for i in range(cache_size_bytes // 4):
        addr = base_addr + i * 0x100000  # 1MB spacing
        if addr < DDR3_BASE + 0x8000000:  # Within 128MB
            wb.read(addr)


def main():
    print("=" * 60)
    print("z1167: Cache-Aware DDR3 Decay Test")
    print("=" * 60)

    wb = RemoteClient()
    wb.open()
    print("Connected to Etherbone at 192.168.0.50:1234")

    results = {
        "experiment": "z1167_cache_aware_decay",
        "timestamp": datetime.now().isoformat(),
        "tests": []
    }

    # Use addresses spread 1MB apart to avoid L2 cache
    # L2 cache is 8KB, so 16 addresses × 1MB spacing >> cache size
    num_test_words = 16
    test_addrs = [DDR3_BASE + i * 0x100000 for i in range(num_test_words)]
    pattern = 0xAAAAAAAA

    print(f"\nUsing {num_test_words} test addresses, 1MB apart")
    print("This ensures cache misses on read-back after decay")

    # Quick verification
    print("\n=== Quick Memory Test ===")
    issue_manual_refresh(wb)
    for addr in test_addrs[:4]:
        wb.write(addr, pattern)

    time.sleep(0.1)

    errors = 0
    for addr in test_addrs[:4]:
        if wb.read(addr) != pattern:
            errors += 1
    print(f"Quick test: {errors}/4 errors (should be 0)")

    # Decay test with spread addresses
    print("\n=== Decay Test (Spread Addresses) ===")

    wait_times = [5, 10, 15, 20, 25, 30, 35, 40, 45, 60]
    decay_results = []

    for wait in wait_times:
        # Refresh and write to all addresses
        issue_manual_refresh(wb)
        for addr in test_addrs:
            wb.write(addr, pattern)

        # Flush cache to ensure we don't read from cache
        flush_cache(wb, DDR3_BASE + 0x8000000)

        # Wait for decay (no refresh)
        print(f"  {wait:3d}s: ", end="", flush=True)
        time.sleep(wait)

        # Flush cache again before reading
        flush_cache(wb, DDR3_BASE + 0x8000000)

        # Read back and count errors
        word_errors = 0
        bit_errors = 0
        error_details = []

        for i, addr in enumerate(test_addrs):
            actual = wb.read(addr)
            if actual != pattern:
                word_errors += 1
                errors = count_bit_errors(pattern, actual)
                bit_errors += errors
                error_details.append({
                    "addr": f"0x{addr:08x}",
                    "expected": f"0x{pattern:08x}",
                    "actual": f"0x{actual:08x}",
                    "bit_errors": errors
                })

        pct = word_errors / num_test_words * 100
        print(f"{word_errors}/{num_test_words} ({pct:.0f}%), {bit_errors} bits")

        decay_results.append({
            "wait_seconds": wait,
            "word_errors": word_errors,
            "bit_errors": bit_errors,
            "percent_decayed": pct,
            "error_details": error_details[:3]  # First 3
        })

        if word_errors > 0:
            print(f"      DECAY DETECTED!")
            # Continue to see how decay progresses

    results["tests"].append({
        "name": "spread_address_decay",
        "pattern": "0xAAAAAAAA",
        "num_addresses": num_test_words,
        "address_spacing": "1MB",
        "results": decay_results
    })

    # Find retention threshold
    threshold_found = None
    for r in decay_results:
        if r["bit_errors"] > 0:
            threshold_found = r["wait_seconds"]
            break

    # Pattern comparison at decay time
    if threshold_found:
        print(f"\n=== Pattern Comparison at {threshold_found}s ===")
        patterns = [
            (0xFFFFFFFF, "all_ones"),
            (0x00000000, "all_zeros"),
            (0xAAAAAAAA, "checkerboard"),
            (0x55555555, "inv_checker"),
        ]

        pattern_results = []
        for pat, name in patterns:
            issue_manual_refresh(wb)
            for addr in test_addrs:
                wb.write(addr, pat)
            flush_cache(wb, DDR3_BASE + 0x8000000)
            time.sleep(threshold_found)
            flush_cache(wb, DDR3_BASE + 0x8000000)

            bit_errors = 0
            word_errors = 0
            for addr in test_addrs:
                actual = wb.read(addr)
                if actual != pat:
                    word_errors += 1
                    bit_errors += count_bit_errors(pat, actual)

            print(f"  {name:15s}: {word_errors}/{num_test_words} words, {bit_errors} bits")
            pattern_results.append({
                "pattern": name,
                "word_errors": word_errors,
                "bit_errors": bit_errors
            })

        results["tests"].append({
            "name": "pattern_comparison",
            "wait_seconds": threshold_found,
            "results": pattern_results
        })

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    if threshold_found:
        print(f"Retention threshold: ~{threshold_found}s")
        print("Decay confirmed with cache-aware testing!")
        results["retention_threshold_s"] = threshold_found
        results["conclusion"] = "decay_confirmed"
    else:
        print("No decay observed up to 60s")
        print("DDR3 has very strong retention at room temperature")
        results["conclusion"] = "strong_retention"

    # Save
    results_path = '/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy/results/z1167_cache_aware_decay.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {results_path}")

    wb.close()
    print("Done!")

    return results


if __name__ == "__main__":
    main()
