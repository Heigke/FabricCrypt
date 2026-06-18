#!/usr/bin/env python3
"""
z1188: Hybrid Test - SW mode partial write, L2 read

Strategy:
1. Write via L2 cache (known working)
2. Trigger FSM for partial write only (ACT → tRAS → PRE)
3. Read via L2 cache immediately

This tests if the partial write affects DRAM cells, using
L2 cache for read instead of broken DFII rddata.
"""

import json
import time
import datetime
from litex import RemoteClient

DRAM_BASE = 0x40000000

# SWCAP CSRs
SWCAP_BASE = 0x3000
SWCAP_ROW_ADDR = SWCAP_BASE + 0x00
SWCAP_BANK_ADDR = SWCAP_BASE + 0x04
SWCAP_COL_ADDR = SWCAP_BASE + 0x08
SWCAP_TRAS_CYCLES = SWCAP_BASE + 0x0c
SWCAP_PATTERN = SWCAP_BASE + 0x10
SWCAP_TRIGGER = SWCAP_BASE + 0x14
SWCAP_STATUS = SWCAP_BASE + 0x18
SWCAP_OPS_COUNT = SWCAP_BASE + 0x20


def dram_addr(row, bank, col):
    """Calculate DRAM byte address"""
    return DRAM_BASE + (row << 14) + (bank << 11) + col


def main():
    print("z1188: Hybrid Test - SW partial write + L2 read")
    print("="*60)

    wb = RemoteClient()
    wb.open()

    # Ensure HW mode
    wb.write(0x2800, 0x0b)
    time.sleep(0.01)

    # Test parameters - use very small addresses to minimize timing
    test_row = 0x010
    test_bank = 0
    test_col = 0
    pattern = 0xFFFFFFFF

    base_addr = dram_addr(test_row, test_bank, test_col)
    print(f"Test address: 0x{base_addr:08x}")

    results = {
        "timestamp": datetime.datetime.now().isoformat(),
        "mode": "hybrid_sw_partial_l2_read",
        "pattern": f"0x{pattern:08x}",
        "tests": []
    }

    tras_values = [1, 2, 4, 8, 16, 32, 64, 128, 256]

    for tras in tras_values:
        print(f"\n--- tRAS={tras} cycles ---")

        # Step 1: Write pattern via L2 cache
        for i in range(4):  # 16 bytes
            wb.write(base_addr + i*4, pattern)
        time.sleep(0.001)

        # Verify write (cache read)
        verify = wb.read(base_addr)
        print(f"  After L2 write: 0x{verify:08x}")

        # Step 2: Configure FSM for partial write ONLY
        # The FSM will: SW_MODE → PRE_ALL → ACT → tRAS → PRE → (skip read) → HW_MODE
        # But our current FSM also does a write. We need to just do ACT → tRAS → PRE
        # Actually, the FSM does its own write which might overwrite our L2 data
        # Let's set the FSM pattern to the same value
        wb.write(SWCAP_ROW_ADDR, test_row)
        wb.write(SWCAP_BANK_ADDR, test_bank)
        wb.write(SWCAP_COL_ADDR, test_col)
        wb.write(SWCAP_TRAS_CYCLES, tras)
        wb.write(SWCAP_PATTERN, pattern)

        # Trigger FSM
        wb.write(SWCAP_TRIGGER, 0)
        time.sleep(0.002)
        wb.write(SWCAP_TRIGGER, 1)

        # Wait for FSM completion
        start = time.time()
        while True:
            status = wb.read(SWCAP_STATUS)
            if status == 32:
                break
            if time.time() - start > 2.0:
                print(f"  TIMEOUT! status={status}")
                break
            time.sleep(0.001)

        elapsed_ms = (time.time() - start) * 1000

        # Clear trigger
        wb.write(SWCAP_TRIGGER, 0)

        # Step 3: Read via L2 cache IMMEDIATELY (no flush!)
        # This might hit cache, but we need to evict our specific line
        # Access a different address in same cache set to force eviction

        # Read from offset to evict this cache line
        # L2 cache is 8KB = 8192 bytes, direct-mapped or 2-way
        # Cache line size is typically 32 bytes for LiteX
        # To evict line at base_addr, read from base_addr + 8192
        evict_addr = base_addr + 8192
        for i in range(4):
            _ = wb.read(evict_addr + i*4)

        # Now read original address - should come from DRAM
        read_data = []
        ones_total = 0
        for i in range(4):
            data = wb.read(base_addr + i*4)
            read_data.append(f"0x{data:08x}")
            ones_total += bin(data).count('1')

        print(f"  FSM elapsed: {elapsed_ms:.1f}ms")
        print(f"  Read back: {read_data}")
        print(f"  Ones: {ones_total}/128")

        results["tests"].append({
            "tras": tras,
            "elapsed_ms": elapsed_ms,
            "read_data": read_data,
            "ones": ones_total
        })

        # Use different row for next test
        test_row += 1
        base_addr = dram_addr(test_row, test_bank, test_col)

    # Summary
    ones_list = [t["ones"] for t in results["tests"]]
    unique = len(set(ones_list))
    results["unique_levels"] = unique
    results["ones_range"] = [min(ones_list), max(ones_list)]

    print(f"\n{'='*60}")
    print(f"Summary: {unique} unique levels")
    print(f"Ones range: {min(ones_list)} to {max(ones_list)}")

    if unique > 1:
        print("SUCCESS: Multi-level detected!")
    else:
        print("NOTICE: Single level (partial write not affecting cells)")

    wb.close()

    # Save
    result_path = "/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy/results/z1188_hybrid_test.json"
    with open(result_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {result_path}")


if __name__ == "__main__":
    main()
