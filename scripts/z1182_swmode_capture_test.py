#!/usr/bin/env python3
"""
z1182: SW Mode Capture Test

Tests the SW mode capture approach for partial writes via DFII.
The FSM directly controls DFII CSRs for complete command authority.
"""

import json
import time
import datetime
from litex import RemoteClient

# CSR addresses for SWCAP module (from build_swmode_capture csr.h)
SWCAP_BASE = 0x3000
SWCAP_ROW_ADDR = SWCAP_BASE + 0x00      # _row_addr (storage)
SWCAP_BANK_ADDR = SWCAP_BASE + 0x04     # _bank_addr (storage)
SWCAP_COL_ADDR = SWCAP_BASE + 0x08      # _col_addr (storage)
SWCAP_TRAS_CYCLES = SWCAP_BASE + 0x0c   # _tras_cycles (storage)
SWCAP_PATTERN = SWCAP_BASE + 0x10       # _pattern (storage)
SWCAP_TRIGGER = SWCAP_BASE + 0x14       # _trigger (storage)
SWCAP_STATUS = SWCAP_BASE + 0x18        # _status (status)
SWCAP_ERROR = SWCAP_BASE + 0x1c         # _error (status)
SWCAP_OPS_COUNT = SWCAP_BASE + 0x20     # _ops_count (status)
SWCAP_CAP_ADDR = SWCAP_BASE + 0x24      # _cap_addr (storage)
SWCAP_CAP_DATA = SWCAP_BASE + 0x28      # _cap_data (status)
SWCAP_CAP_COUNT = SWCAP_BASE + 0x2c     # _cap_count (status)
SWCAP_DBG_RDDATA = SWCAP_BASE + 0x30    # _dbg_rddata (status)

# DFII CSR addresses
DFII_CONTROL = 0x2800


def main():
    print("z1182: SW Mode Capture Test")
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

    # Read initial state
    status = wb.read(SWCAP_STATUS)
    ops = wb.read(SWCAP_OPS_COUNT)
    dfii_ctrl = wb.read(DFII_CONTROL)
    print(f"Initial: status={status}, ops_count={ops}, dfii_control=0x{dfii_ctrl:02x}")

    # Test pattern
    pattern = 0xCAFEBABE

    # Configure test parameters
    test_row = 0x100
    test_bank = 0
    test_col = 0

    results = {
        "timestamp": datetime.datetime.now().isoformat(),
        "mode": "swmode_capture",
        "pattern": f"0x{pattern:08x}",
        "row": f"0x{test_row:04x}",
        "bank": test_bank,
        "col": test_col,
        "tests": []
    }

    # tRAS sweep
    tras_values = [1, 2, 4, 8, 16, 32, 64, 128]

    for tras in tras_values:
        print(f"\n--- Testing tRAS={tras} cycles ---")

        # Configure FSM
        wb.write(SWCAP_ROW_ADDR, test_row)
        wb.write(SWCAP_BANK_ADDR, test_bank)
        wb.write(SWCAP_COL_ADDR, test_col)
        wb.write(SWCAP_TRAS_CYCLES, tras)
        wb.write(SWCAP_PATTERN, pattern)

        # Clear trigger
        wb.write(SWCAP_TRIGGER, 0)
        time.sleep(0.01)

        # Trigger FSM
        wb.write(SWCAP_TRIGGER, 1)

        # Wait for completion
        start_time = time.time()
        timeout = 2.0

        while True:
            status = wb.read(SWCAP_STATUS)
            if status == 32:  # DONE state
                break
            if time.time() - start_time > timeout:
                print(f"  TIMEOUT! status={status}")
                break
            time.sleep(0.001)

        elapsed_ms = (time.time() - start_time) * 1000

        # Clear trigger
        wb.write(SWCAP_TRIGGER, 0)
        time.sleep(0.01)

        # Read results
        ops_count = wb.read(SWCAP_OPS_COUNT)
        cap_count = wb.read(SWCAP_CAP_COUNT)
        dbg_rddata = wb.read(SWCAP_DBG_RDDATA)
        error = wb.read(SWCAP_ERROR)
        dfii_ctrl = wb.read(DFII_CONTROL)

        # Read captured data
        captured = []
        for i in range(min(cap_count, 8)):
            wb.write(SWCAP_CAP_ADDR, i)
            time.sleep(0.001)
            data = wb.read(SWCAP_CAP_DATA)
            captured.append(f"0x{data:08x}")

        # Count ones in captured data
        ones = 0
        for cap_hex in captured:
            val = int(cap_hex, 16)
            ones += bin(val).count('1')

        print(f"  status={status}, ops={ops_count}, cap_count={cap_count}")
        print(f"  dbg_rddata=0x{dbg_rddata:08x}, error={error}")
        print(f"  dfii_control=0x{dfii_ctrl:02x}")
        print(f"  elapsed={elapsed_ms:.1f}ms")
        print(f"  captured: {captured[:4]}")
        print(f"  ones in captured: {ones}")

        test_result = {
            "tras": tras,
            "status": status,
            "ops_count": ops_count,
            "cap_count": cap_count,
            "dbg_rddata": f"0x{dbg_rddata:08x}",
            "error": error,
            "dfii_control": f"0x{dfii_ctrl:02x}",
            "elapsed_ms": elapsed_ms,
            "captured": captured,
            "ones": ones
        }
        results["tests"].append(test_result)

    # Check for variation in ones count
    ones_counts = [t["ones"] for t in results["tests"]]
    unique_levels = len(set(ones_counts))
    results["unique_levels"] = unique_levels
    results["ones_range"] = [min(ones_counts), max(ones_counts)]

    print(f"\n{'='*60}")
    print(f"Summary: {unique_levels} unique levels")
    print(f"Ones range: {min(ones_counts)} to {max(ones_counts)}")

    # Check if partial write is working
    if unique_levels > 1:
        print("SUCCESS: Multi-level charge states detected!")
    else:
        if max(ones_counts) > 0:
            print("NOTICE: All reads returned same value (no partial write effect)")
        else:
            print("WARNING: All reads returned 0 (check DFII rddata capture)")

    wb.close()

    # Save results
    result_path = "/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy/results/z1182_swmode_capture.json"
    with open(result_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {result_path}")


if __name__ == "__main__":
    main()
