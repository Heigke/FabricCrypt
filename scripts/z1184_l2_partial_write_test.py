#!/usr/bin/env python3
"""
z1184: L2 Cache + DFI Partial Write Test

Strategy:
1. Write via L2 cache (goes through controller, known to work)
2. Trigger partial write via ext_dfi (ACT -> short tRAS -> PRE)
3. Read via L2 cache (may hit cache, but at least verifies data path)
4. Flush cache by accessing different region, then read again

This approach uses the proven L2 cache path for writes and tests if
the DFI partial write actually affects the DRAM cells.
"""

import json
import time
import datetime
from litex import RemoteClient

# DRAM base
DRAM_BASE = 0x40000000

# SWCAP CSR addresses (FSM controls partial write)
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
    """Calculate DRAM byte address from row/bank/col"""
    # DDR3: row (14 bits), bank (3 bits), col (10 bits)
    # Address mapping (for MT41K128M16):
    # byte_addr = base + (row << 14) + (bank << 11) + col
    return DRAM_BASE + (row << 14) + (bank << 11) + col


def main():
    print("z1184: L2 Cache + DFI Partial Write Test")
    print("="*60)

    wb = RemoteClient()
    wb.open()

    # Check identity
    ident = ""
    for i in range(256):
        c = wb.read(0x1800 + i*4)
        if c == 0:
            break
        ident += chr(c)
    print(f"SoC Identity: {ident}")

    # Test parameters
    test_row = 0x200    # Different row to avoid prior tests
    test_bank = 1       # Different bank
    test_col = 0
    pattern = 0xFFFFFFFF  # All 1s

    # Calculate addresses
    addr = dram_addr(test_row, test_bank, test_col)
    # Flush address - same row but different cache line
    flush_addr = dram_addr(test_row + 0x100, test_bank, 0)

    print(f"Test address: 0x{addr:08x} (row=0x{test_row:x}, bank={test_bank}, col={test_col})")
    print(f"Flush address: 0x{flush_addr:08x}")

    results = {
        "timestamp": datetime.datetime.now().isoformat(),
        "mode": "l2_partial_write",
        "pattern": f"0x{pattern:08x}",
        "row": f"0x{test_row:04x}",
        "bank": test_bank,
        "col": test_col,
        "tests": []
    }

    tras_values = [1, 2, 4, 8, 16, 32, 64]

    for tras in tras_values:
        print(f"\n--- Testing tRAS={tras} cycles ---")

        # Step 1: Write pattern via L2 cache
        print(f"  Writing 0x{pattern:08x} via L2 cache...")
        for i in range(16):  # 64 bytes = 16 x 32-bit words
            wb.write(addr + i*4, pattern)
        time.sleep(0.01)

        # Verify write
        verify = wb.read(addr)
        print(f"  Verify write: 0x{verify:08x}")

        # Step 2: Flush cache by reading from different region
        print("  Flushing cache...")
        for i in range(256):  # Read enough to flush L2 cache
            _ = wb.read(flush_addr + i*4)
        time.sleep(0.01)

        # Step 3: Configure FSM for partial write
        wb.write(SWCAP_ROW_ADDR, test_row)
        wb.write(SWCAP_BANK_ADDR, test_bank)
        wb.write(SWCAP_COL_ADDR, test_col)
        wb.write(SWCAP_TRAS_CYCLES, tras)
        wb.write(SWCAP_PATTERN, pattern)  # FSM will also try to write this

        # Step 4: Trigger partial write FSM
        # The FSM does: SW_MODE -> PRE_ALL -> ACT -> WAIT_TRAS -> PRE -> READ -> HW_MODE
        # But we already wrote via L2, so the partial write (ACT->tRAS->PRE)
        # should partially discharge the cells

        wb.write(SWCAP_TRIGGER, 0)
        time.sleep(0.01)
        wb.write(SWCAP_TRIGGER, 1)

        # Wait for FSM completion
        start_time = time.time()
        while True:
            status = wb.read(SWCAP_STATUS)
            if status == 32:
                break
            if time.time() - start_time > 2.0:
                print(f"  TIMEOUT! status={status}")
                break
            time.sleep(0.001)

        # Clear trigger
        wb.write(SWCAP_TRIGGER, 0)
        time.sleep(0.01)

        # Step 5: Flush cache again
        print("  Flushing cache before read...")
        for i in range(256):
            _ = wb.read(flush_addr + i*4)
        time.sleep(0.01)

        # Step 6: Read back via L2 cache (should come from DRAM now)
        read_data = []
        ones_total = 0
        for i in range(16):
            data = wb.read(addr + i*4)
            read_data.append(f"0x{data:08x}")
            ones_total += bin(data).count('1')

        print(f"  Read back: {read_data[:4]}...")
        print(f"  Total ones: {ones_total}/512")

        test_result = {
            "tras": tras,
            "read_data": read_data,
            "ones": ones_total,
            "status": status
        }
        results["tests"].append(test_result)

    # Summary
    ones_counts = [t["ones"] for t in results["tests"]]
    unique_levels = len(set(ones_counts))
    results["unique_levels"] = unique_levels
    results["ones_range"] = [min(ones_counts), max(ones_counts)]

    print(f"\n{'='*60}")
    print(f"Summary: {unique_levels} unique levels")
    print(f"Ones range: {min(ones_counts)} to {max(ones_counts)}")

    if unique_levels > 1:
        print("SUCCESS: Multi-level charge states detected!")
    else:
        if max(ones_counts) == 512:
            print("NOTICE: All reads returned full 1s (partial write not affecting DRAM)")
        elif max(ones_counts) == 0:
            print("WARNING: All reads returned 0 (data not reaching DRAM)")
        else:
            print(f"NOTICE: Constant value returned ({max(ones_counts)}/512 ones)")

    wb.close()

    # Save results
    result_path = "/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy/results/z1184_l2_partial_write.json"
    with open(result_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {result_path}")


if __name__ == "__main__":
    main()
