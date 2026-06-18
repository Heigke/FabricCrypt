#!/usr/bin/env python3
"""
z1175: IDELAY Analog Sensing

Use IDELAY2 to sample DDR3 data at different points in time.
By sampling during signal transitions, we may detect analog voltage levels.

IDELAY2 specs:
- 32 taps (0-31)
- ~78ps per tap at 200MHz reference
- Total range: ~2.5ns

Strategy:
1. Write pattern to memory
2. Sweep IDELAY taps while reading
3. Look for intermediate values (not just 0s and 1s)
4. Combine with decay to see voltage-dependent effects
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

DDRPHY_BASE = 0x800
DDRPHY_DLY_SEL = DDRPHY_BASE + 0x04
DDRPHY_RDLY_RST = DDRPHY_BASE + 0x14
DDRPHY_RDLY_INC = DDRPHY_BASE + 0x18

HW_MODE = 0x0B
SW_MODE = 0x0A
CMD_REFRESH = 0x0D
CMD_PRECHARGE = 0x0B


def init_ddr3(wb):
    wb.write(DFII_CONTROL, HW_MODE)
    time.sleep(0.1)
    for _ in range(10):
        issue_refresh(wb)


def issue_refresh(wb):
    wb.write(DFII_CONTROL, SW_MODE)
    time.sleep(0.0001)
    wb.write(DFII_PI0_ADDRESS, 0x400)
    wb.write(DFII_PI0_BADDRESS, 0)
    wb.write(DFII_PI0_COMMAND, CMD_PRECHARGE)
    wb.write(DFII_PI0_COMMAND_ISSUE, 1)
    time.sleep(0.0001)
    wb.write(DFII_PI0_COMMAND, CMD_REFRESH)
    wb.write(DFII_PI0_COMMAND_ISSUE, 1)
    time.sleep(0.0001)
    wb.write(DFII_CONTROL, HW_MODE)


def set_idelay(wb, taps):
    """Set IDELAY to specified tap count (0-31)"""
    for dqs in range(2):  # 2 DQS groups for 16-bit interface
        wb.write(DDRPHY_DLY_SEL, 1 << dqs)
        wb.write(DDRPHY_RDLY_RST, 1)
        time.sleep(0.0001)
        for _ in range(taps):
            wb.write(DDRPHY_RDLY_INC, 1)
    time.sleep(0.001)


def evict_cache(wb):
    """Evict L2 cache"""
    evict_base = DDR3_BASE + 0x4000000
    for i in range(4096):
        wb.write(evict_base + i * 4, 0xDEADC0DE)


def main():
    print("=" * 60)
    print("z1175: IDELAY Analog Sensing")
    print("=" * 60)

    wb = RemoteClient()
    wb.open()
    print("Connected to Etherbone")

    results = {
        "experiment": "z1175_idelay_analog_sensing",
        "timestamp": datetime.now().isoformat(),
        "tests": []
    }

    init_ddr3(wb)
    print(f"DFII: 0x{wb.read(DFII_CONTROL):02x}")

    test_addr = DDR3_BASE + 0x1000000
    patterns = [
        (0xFFFFFFFF, "all_ones"),
        (0x00000000, "all_zeros"),
        (0xAAAAAAAA, "checkerboard"),
        (0x55555555, "inv_checker"),
    ]

    # Test 1: IDELAY sweep on fresh data
    print("\n=== Test 1: IDELAY Sweep (Fresh Data) ===")
    print("Sweeping IDELAY taps to find sampling window...")

    for pattern, name in patterns:
        print(f"\n  Pattern: {name} (0x{pattern:08x})")

        idelay_results = []

        for tap in range(0, 32, 2):  # Every 2 taps
            issue_refresh(wb)
            wb.write(test_addr, pattern)

            set_idelay(wb, tap)
            result = wb.read(test_addr)
            ones = bin(result).count('1')
            match = "OK" if result == pattern else f"ERROR 0x{result:08x}"

            print(f"    Tap {tap:2d} ({tap*78:4d}ps): {match} ({ones} ones)")

            idelay_results.append({
                "tap": tap,
                "delay_ps": tap * 78,
                "result": f"0x{result:08x}",
                "ones": ones,
                "match": result == pattern
            })

        results["tests"].append({
            "name": f"idelay_sweep_{name}",
            "pattern": f"0x{pattern:08x}",
            "results": idelay_results
        })

    # Reset IDELAY to optimal
    set_idelay(wb, 0)

    # Test 2: IDELAY + Decay combined
    print("\n=== Test 2: IDELAY + Decay Combined ===")
    print("Testing if decayed cells show different IDELAY response...")

    pattern = 0xAAAAAAAA
    print(f"Pattern: 0x{pattern:08x}")

    decay_idelay_results = []

    # Write pattern
    issue_refresh(wb)
    wb.write(test_addr, pattern)

    # Evict cache to force decay
    evict_cache(wb)

    # Wait for partial decay (try different times)
    for decay_ms in [0, 1, 5, 10]:
        print(f"\n  After {decay_ms}ms decay:")

        if decay_ms > 0:
            # Re-write and wait
            issue_refresh(wb)
            wb.write(test_addr, pattern)
            evict_cache(wb)
            time.sleep(decay_ms / 1000.0)

        # Sample at different IDELAY taps
        tap_results = []
        for tap in [0, 8, 16, 24, 31]:
            set_idelay(wb, tap)
            result = wb.read(test_addr)
            ones = bin(result).count('1')
            print(f"    Tap {tap:2d}: 0x{result:08x} ({ones} ones)")
            tap_results.append({"tap": tap, "result": f"0x{result:08x}", "ones": ones})

        decay_idelay_results.append({
            "decay_ms": decay_ms,
            "tap_results": tap_results
        })

    results["tests"].append({
        "name": "decay_idelay_combined",
        "pattern": f"0x{pattern:08x}",
        "results": decay_idelay_results
    })

    # Reset IDELAY
    set_idelay(wb, 0)

    # Test 3: Find edges in IDELAY response
    print("\n=== Test 3: Fine IDELAY Edge Detection ===")
    print("Looking for transition regions in IDELAY response...")

    issue_refresh(wb)
    pattern = 0xAAAAAAAA
    wb.write(test_addr, pattern)

    edge_results = []
    prev_ones = None

    for tap in range(32):
        set_idelay(wb, tap)
        result = wb.read(test_addr)
        ones = bin(result).count('1')

        # Detect edge (change in ones count)
        edge = ""
        if prev_ones is not None and ones != prev_ones:
            edge = f" <-- EDGE ({prev_ones}->{ones})"

        print(f"  Tap {tap:2d}: {ones:2d} ones{edge}")

        edge_results.append({
            "tap": tap,
            "ones": ones,
            "is_edge": prev_ones is not None and ones != prev_ones
        })
        prev_ones = ones

    results["tests"].append({
        "name": "idelay_edge_detection",
        "results": edge_results
    })

    # Reset IDELAY to safe value
    set_idelay(wb, 0)

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    # Check for edges
    edges = [r for r in edge_results if r["is_edge"]]
    if edges:
        print(f"Found {len(edges)} IDELAY edge(s) - sampling window boundaries")
        for e in edges:
            print(f"  Tap {e['tap']}: transition point")
        results["edges_found"] = len(edges)
    else:
        print("No IDELAY edges found - full valid sampling window")
        results["edges_found"] = 0

    # Check decay effect
    decay_effect = False
    for r in decay_idelay_results:
        if r["decay_ms"] > 0:
            for t in r["tap_results"]:
                if t["ones"] != 16:  # Pattern has 16 ones
                    decay_effect = True
                    break

    if decay_effect:
        print("Decay affects IDELAY response!")
        results["decay_affects_idelay"] = True
    else:
        print("Decay doesn't significantly affect IDELAY response")
        results["decay_affects_idelay"] = False

    results["conclusion"] = "idelay_sensing_complete"

    # Save
    results_path = '/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy/results/z1175_idelay_analog_sensing.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {results_path}")

    wb.close()
    print("Done!")

    return results


if __name__ == "__main__":
    main()
