#!/usr/bin/env python3
"""
z1217: Verify ext_dfi path by checking if FSM actually writes to DRAM.

Test sequence:
1. Init DDR3
2. Write known pattern via SW mode to location
3. Trigger FSM with DIFFERENT pattern
4. Read location via SW mode
5. If value changed, ext_dfi write path works
6. If value unchanged, ext_dfi commands not reaching DRAM
"""

import time
from litex.tools.litex_client import RemoteClient

# CSR addresses
DFII_CONTROL = 0x3000
PHASE_SIZE = 0x18

def pi_base(phase):
    return 0x3004 + phase * PHASE_SIZE

# Partial Write CSRs
PW_BASE = 0x2800
PW_CONFIG = PW_BASE + 0x00
PW_WRITE_DATA = PW_BASE + 0x04
PW_REF_DATA = PW_BASE + 0x08
PW_CONTROL = PW_BASE + 0x0c
PW_STATUS = PW_BASE + 0x10
PW_RESULT = PW_BASE + 0x14
PW_DEBUG = 0x2824


def sw_write(wb, row, col, bank, data):
    """Write data using SW mode."""
    WRPHASE = 3

    wb.write(DFII_CONTROL, 0x0E)  # SW mode
    time.sleep(0.001)

    # Precharge all
    wb.write(pi_base(0) + 0x08, 0x400)
    wb.write(pi_base(0) + 0x0c, 0)
    wb.write(pi_base(0) + 0x00, 0x0B)
    wb.write(pi_base(0) + 0x04, 1)
    time.sleep(0.0001)

    # Activate
    wb.write(pi_base(0) + 0x08, row)
    wb.write(pi_base(0) + 0x0c, bank)
    wb.write(pi_base(0) + 0x00, 0x09)
    wb.write(pi_base(0) + 0x04, 1)
    time.sleep(0.0001)

    # Write
    wb.write(pi_base(WRPHASE) + 0x10, data)
    wb.write(pi_base(WRPHASE) + 0x08, col)
    wb.write(pi_base(WRPHASE) + 0x0c, bank)
    wb.write(pi_base(WRPHASE) + 0x00, 0x17)
    wb.write(pi_base(WRPHASE) + 0x04, 1)
    time.sleep(0.001)

    # Precharge
    wb.write(pi_base(0) + 0x08, 0x400)
    wb.write(pi_base(0) + 0x0c, 0)
    wb.write(pi_base(0) + 0x00, 0x0B)
    wb.write(pi_base(0) + 0x04, 1)
    time.sleep(0.0001)

    wb.write(DFII_CONTROL, 0x0F)  # HW mode


def sw_read(wb, row, col, bank):
    """Read data using SW mode."""
    RDPHASE = 2

    wb.write(DFII_CONTROL, 0x0E)  # SW mode
    time.sleep(0.001)

    # Precharge all
    wb.write(pi_base(0) + 0x08, 0x400)
    wb.write(pi_base(0) + 0x0c, 0)
    wb.write(pi_base(0) + 0x00, 0x0B)
    wb.write(pi_base(0) + 0x04, 1)
    time.sleep(0.0001)

    # Activate
    wb.write(pi_base(0) + 0x08, row)
    wb.write(pi_base(0) + 0x0c, bank)
    wb.write(pi_base(0) + 0x00, 0x09)
    wb.write(pi_base(0) + 0x04, 1)
    time.sleep(0.0001)

    # Read
    wb.write(pi_base(RDPHASE) + 0x08, col)
    wb.write(pi_base(RDPHASE) + 0x0c, bank)
    wb.write(pi_base(RDPHASE) + 0x00, 0x25)
    wb.write(pi_base(RDPHASE) + 0x04, 1)
    time.sleep(0.001)

    result = wb.read(pi_base(0) + 0x14)

    # Precharge
    wb.write(pi_base(0) + 0x08, 0x400)
    wb.write(pi_base(0) + 0x0c, 0)
    wb.write(pi_base(0) + 0x00, 0x0B)
    wb.write(pi_base(0) + 0x04, 1)
    time.sleep(0.0001)

    wb.write(DFII_CONTROL, 0x0F)
    return result


def trigger_fsm(wb, row, col, bank, tras, ref_data, write_data):
    """Trigger FSM and wait for completion."""
    config = (row << 18) | (col << 8) | (bank << 5) | tras
    wb.write(PW_CONFIG, config)
    wb.write(PW_REF_DATA, ref_data)
    wb.write(PW_WRITE_DATA, write_data)

    wb.write(PW_CONTROL, 0)
    time.sleep(0.001)
    wb.write(PW_CONTROL, 1)

    # Wait for done
    for _ in range(100):
        status = wb.read(PW_STATUS)
        if (status >> 1) & 1:
            break
        time.sleep(0.001)

    wb.write(PW_CONTROL, 0)

    result = wb.read(PW_RESULT)
    debug_state = wb.read(PW_DEBUG)
    return result, status, debug_state


def main():
    print("=" * 60)
    print("z1217: Verify ext_dfi Write Path")
    print("=" * 60)

    wb = RemoteClient(host='localhost', port=1234)
    wb.open()
    print(f"\nConnected! Identifier: 0x{wb.read(0x1800):08x}")

    row, col, bank = 100, 0, 0  # Use row 100 to avoid conflicts

    # Test 1: Verify SW mode works
    print("\n=== Test 1: SW Mode Verification ===")
    sw_write(wb, row, col, bank, 0xAAAAAAAA)
    val = sw_read(wb, row, col, bank)
    print(f"  Write 0xAAAAAAAA, Read 0x{val:08x}")
    if val != 0xAAAAAAAA:
        print("  ERROR: SW mode not working!")
        wb.close()
        return

    print("  SW mode OK!")

    # Test 2: Check FSM write affects DRAM
    print("\n=== Test 2: FSM Write Path ===")
    # Write initial pattern via SW
    sw_write(wb, row, col, bank, 0x11111111)
    val_before = sw_read(wb, row, col, bank)
    print(f"  Initial value (SW): 0x{val_before:08x}")

    # Trigger FSM to write 0xFFFFFFFF with ref_data=0x00000000
    # Full tRAS (tras=31) - should complete the write fully
    print("  Triggering FSM: ref=0x00000000, write=0xFFFFFFFF, tras=31...")
    result, status, state = trigger_fsm(wb, row, col, bank, 31, 0x00000000, 0xFFFFFFFF)
    print(f"  FSM result: 0x{result:08x}, status: 0x{status:02x}, state: {state}")

    # Read back via SW mode
    val_after = sw_read(wb, row, col, bank)
    print(f"  Value after FSM (SW): 0x{val_after:08x}")

    print("\n=== Analysis ===")
    if val_after == 0xFFFFFFFF:
        print("  *** ext_dfi WRITE PATH WORKS! ***")
        print("  FSM successfully wrote 0xFFFFFFFF to DRAM.")
        print("  Issue is only with rddata capture.")
    elif val_after == 0x00000000:
        print("  Value is ref_data (0x00000000).")
        print("  FSM ref write works, partial write may have been interrupted.")
    elif val_after == 0x11111111:
        print("  *** ext_dfi WRITE NOT WORKING ***")
        print("  Value unchanged from initial (0x11111111).")
        print("  ext_dfi commands not reaching DRAM.")
    else:
        print(f"  Unexpected value: 0x{val_after:08x}")

    # Test 3: Check with different tras values
    print("\n=== Test 3: tRAS Sweep ===")
    for tras in [0, 1, 5, 10, 31]:
        sw_write(wb, row, col, bank, 0x11111111)
        result, _, _ = trigger_fsm(wb, row, col, bank, tras, 0x00000000, 0xFFFFFFFF)
        val = sw_read(wb, row, col, bank)
        ones = bin(val).count('1')
        print(f"  tras={tras:2d}: FSM result=0x{result:08x}, DRAM value=0x{val:08x} ({ones}/32 bits)")

    wb.close()
    print("\nDone!")


if __name__ == "__main__":
    main()
