#!/usr/bin/env python3
"""
z1187: Quick verification of FSM SW mode operation

The FSM takes ~30ms to complete. If it properly writes data via DFII SW mode,
we should see that data when reading back immediately (before decay).

Test sequence:
1. Trigger FSM (writes pattern, does partial write, reads via SW mode)
2. Immediately read FSM captured data
3. Compare with pattern
"""

import time
from litex import RemoteClient

SWCAP_BASE = 0x3000
SWCAP_ROW_ADDR = SWCAP_BASE + 0x00
SWCAP_BANK_ADDR = SWCAP_BASE + 0x04
SWCAP_COL_ADDR = SWCAP_BASE + 0x08
SWCAP_TRAS_CYCLES = SWCAP_BASE + 0x0c
SWCAP_PATTERN = SWCAP_BASE + 0x10
SWCAP_TRIGGER = SWCAP_BASE + 0x14
SWCAP_STATUS = SWCAP_BASE + 0x18
SWCAP_OPS_COUNT = SWCAP_BASE + 0x20
SWCAP_CAP_ADDR = SWCAP_BASE + 0x24
SWCAP_CAP_DATA = SWCAP_BASE + 0x28
SWCAP_CAP_COUNT = SWCAP_BASE + 0x2c
SWCAP_DBG_RDDATA = SWCAP_BASE + 0x30


def main():
    print("z1187: Quick FSM Verification")
    print("="*60)

    wb = RemoteClient()
    wb.open()

    # Ensure HW mode initially
    wb.write(0x2800, 0x0b)
    time.sleep(0.01)

    # Test with different patterns
    patterns = [0xFFFFFFFF, 0x00000000, 0xAAAAAAAA, 0x55555555, 0xDEADBEEF]

    # Use high tRAS (full write) first to verify write path
    test_row = 0x300
    test_bank = 2
    test_col = 0

    for pattern in patterns:
        print(f"\n--- Pattern: 0x{pattern:08x} ---")

        # Configure FSM with HIGH tRAS (256 cycles = full write)
        wb.write(SWCAP_ROW_ADDR, test_row)
        wb.write(SWCAP_BANK_ADDR, test_bank)
        wb.write(SWCAP_COL_ADDR, test_col)
        wb.write(SWCAP_TRAS_CYCLES, 256)  # Long tRAS for full write
        wb.write(SWCAP_PATTERN, pattern)

        # Trigger FSM
        wb.write(SWCAP_TRIGGER, 0)
        time.sleep(0.005)
        wb.write(SWCAP_TRIGGER, 1)

        # Wait for completion
        start = time.time()
        while True:
            status = wb.read(SWCAP_STATUS)
            if status == 32:  # DONE
                break
            if time.time() - start > 1.0:
                print(f"  TIMEOUT! status={status}")
                break
            time.sleep(0.001)

        elapsed_ms = (time.time() - start) * 1000

        # Clear trigger
        wb.write(SWCAP_TRIGGER, 0)

        # Read captured data
        wb.write(SWCAP_CAP_ADDR, 0)
        time.sleep(0.001)
        cap_data = wb.read(SWCAP_CAP_DATA)
        dbg_rddata = wb.read(SWCAP_DBG_RDDATA)
        cap_count = wb.read(SWCAP_CAP_COUNT)

        print(f"  FSM elapsed: {elapsed_ms:.1f}ms")
        print(f"  Captured: 0x{cap_data:08x}")
        print(f"  Debug rddata: 0x{dbg_rddata:08x}")
        print(f"  cap_count: {cap_count}")

        if cap_data == pattern:
            print("  MATCH - FSM write/read working!")
        else:
            print(f"  MISMATCH - expected 0x{pattern:08x}")

    # Now test with short tRAS to see if we can affect the charge
    print("\n" + "="*60)
    print("Now testing with varying tRAS to see partial write effect:")

    pattern = 0xFFFFFFFF
    wb.write(SWCAP_PATTERN, pattern)
    wb.write(SWCAP_ROW_ADDR, test_row + 1)  # Different row

    for tras in [1, 2, 4, 8, 16, 32, 64, 128, 256]:
        wb.write(SWCAP_TRAS_CYCLES, tras)

        # Trigger
        wb.write(SWCAP_TRIGGER, 0)
        time.sleep(0.005)
        wb.write(SWCAP_TRIGGER, 1)

        # Wait for completion
        while wb.read(SWCAP_STATUS) != 32:
            time.sleep(0.001)

        wb.write(SWCAP_TRIGGER, 0)

        # Read result
        wb.write(SWCAP_CAP_ADDR, 0)
        time.sleep(0.001)
        cap_data = wb.read(SWCAP_CAP_DATA)
        ones = bin(cap_data).count('1')

        print(f"  tRAS={tras:3d}: 0x{cap_data:08x} ({ones:2d}/32 ones)")

    wb.close()


if __name__ == "__main__":
    main()
