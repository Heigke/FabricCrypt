#!/usr/bin/env python3
"""
z1216: Debug the DFI mux to understand why ext_dfi commands aren't reaching DRAM.

The issue might be:
1. ext_dfi_sel not being set/respected
2. DFI commands not appearing on master
3. PHY not processing commands
4. DFII internal structure different than expected

Let's investigate by:
1. Checking DFII control and phase injector state
2. Manually writing via SW mode to confirm path works
3. Triggering FSM and monitoring what happens
"""

import time
from litex.tools.litex_client import RemoteClient

# CSR addresses
DFII_CONTROL = 0x3000
PHASE_SIZE = 0x18

# Phase injector registers per phase
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


def read_phase_state(wb, phase):
    """Read all phase injector state."""
    base = pi_base(phase)
    return {
        'command': wb.read(base + 0x00),
        'command_issue': wb.read(base + 0x04),
        'address': wb.read(base + 0x08),
        'baddress': wb.read(base + 0x0c),
        'wrdata': wb.read(base + 0x10),
        'rddata': wb.read(base + 0x14),
    }


def sw_mode_write_read(wb, row, col, bank, data):
    """
    Write and read using SW mode (DFII phase injectors).
    This is the known-working path from z1213.
    """
    WRPHASE = 3
    RDPHASE = 2

    # Enter SW mode
    wb.write(DFII_CONTROL, 0x0E)  # SEL=0, CKE|ODT|RESET_N
    time.sleep(0.01)

    # Precharge all
    wb.write(pi_base(0) + 0x08, 0x400)  # address = A10
    wb.write(pi_base(0) + 0x0c, 0)  # bank
    wb.write(pi_base(0) + 0x00, 0x0B)  # PRE: RAS|WE|CS
    wb.write(pi_base(0) + 0x04, 1)  # issue
    time.sleep(0.001)

    # Activate
    wb.write(pi_base(0) + 0x08, row)
    wb.write(pi_base(0) + 0x0c, bank)
    wb.write(pi_base(0) + 0x00, 0x09)  # ACT: RAS|CS
    wb.write(pi_base(0) + 0x04, 1)
    time.sleep(0.001)

    # Write on WRPHASE
    wb.write(pi_base(WRPHASE) + 0x10, data)  # wrdata
    wb.write(pi_base(WRPHASE) + 0x08, col)  # address
    wb.write(pi_base(WRPHASE) + 0x0c, bank)  # bank
    wb.write(pi_base(WRPHASE) + 0x00, 0x17)  # WRITE: CAS|WE|CS|WRDATA
    wb.write(pi_base(WRPHASE) + 0x04, 1)  # issue
    time.sleep(0.01)

    # Precharge
    wb.write(pi_base(0) + 0x08, 0x400)
    wb.write(pi_base(0) + 0x0c, bank)
    wb.write(pi_base(0) + 0x00, 0x0B)
    wb.write(pi_base(0) + 0x04, 1)
    time.sleep(0.001)

    # Activate again for read
    wb.write(pi_base(0) + 0x08, row)
    wb.write(pi_base(0) + 0x0c, bank)
    wb.write(pi_base(0) + 0x00, 0x09)
    wb.write(pi_base(0) + 0x04, 1)
    time.sleep(0.001)

    # Read on RDPHASE
    wb.write(pi_base(RDPHASE) + 0x08, col)
    wb.write(pi_base(RDPHASE) + 0x0c, bank)
    wb.write(pi_base(RDPHASE) + 0x00, 0x25)  # READ: CAS|CS|RDDATA
    wb.write(pi_base(RDPHASE) + 0x04, 1)
    time.sleep(0.01)

    # Read result from phase 0 (where data appears)
    result = wb.read(pi_base(0) + 0x14)

    # Precharge
    wb.write(pi_base(0) + 0x08, 0x400)
    wb.write(pi_base(0) + 0x0c, 0)
    wb.write(pi_base(0) + 0x00, 0x0B)
    wb.write(pi_base(0) + 0x04, 1)
    time.sleep(0.001)

    # Return to HW mode
    wb.write(DFII_CONTROL, 0x0F)

    return result


def main():
    print("=" * 60)
    print("z1216: DFI Mux Debug")
    print("=" * 60)

    wb = RemoteClient(host='localhost', port=1234)
    wb.open()
    print(f"\nConnected! Identifier: 0x{wb.read(0x1800):08x}")

    # Check DFII state
    print("\n=== DFII State ===")
    dfii_ctrl = wb.read(DFII_CONTROL)
    print(f"DFII Control: 0x{dfii_ctrl:02x}")
    print(f"  SEL={(dfii_ctrl >> 0) & 1}, CKE={(dfii_ctrl >> 1) & 1}, ODT={(dfii_ctrl >> 2) & 1}, RESET_N={(dfii_ctrl >> 3) & 1}")

    # Read all phase injector states
    print("\n=== Phase Injector States ===")
    for phase in range(4):
        state = read_phase_state(wb, phase)
        print(f"  Phase {phase}: cmd=0x{state['command']:02x}, addr=0x{state['address']:04x}, rddata=0x{state['rddata']:08x}")

    # Test SW mode write/read (known working)
    print("\n=== Test SW Mode Write/Read ===")
    test_data = 0xDEADBEEF
    result = sw_mode_write_read(wb, row=0, col=0, bank=0, data=test_data)
    print(f"  Write 0x{test_data:08x}, Read 0x{result:08x}")
    if result == test_data:
        print("  SW mode WORKING!")
    else:
        print("  SW mode FAILED!")

    # Now try FSM
    print("\n=== FSM Test ===")
    # Configure FSM
    row, col, bank, tras = 0, 0, 0, 10
    config = (row << 18) | (col << 8) | (bank << 5) | tras
    wb.write(PW_CONFIG, config)
    wb.write(PW_REF_DATA, 0x00000000)
    wb.write(PW_WRITE_DATA, 0xFFFFFFFF)

    # Get state before trigger
    print("  Before trigger:")
    dfii_ctrl = wb.read(DFII_CONTROL)
    status = wb.read(PW_STATUS)
    print(f"    DFII Control: 0x{dfii_ctrl:02x}")
    print(f"    FSM Status: 0x{status:02x}")

    # Trigger
    wb.write(PW_CONTROL, 1)
    time.sleep(0.001)

    # Check state during FSM execution
    print("  During execution (1ms after trigger):")
    dfii_ctrl = wb.read(DFII_CONTROL)
    status = wb.read(PW_STATUS)
    print(f"    DFII Control: 0x{dfii_ctrl:02x}")
    print(f"    FSM Status: 0x{status:02x} (busy={(status>>0)&1}, done={(status>>1)&1})")

    # Wait for completion
    time.sleep(0.1)
    wb.write(PW_CONTROL, 0)

    print("  After completion:")
    dfii_ctrl = wb.read(DFII_CONTROL)
    status = wb.read(PW_STATUS)
    result = wb.read(PW_RESULT)
    print(f"    DFII Control: 0x{dfii_ctrl:02x}")
    print(f"    FSM Status: 0x{status:02x}")
    print(f"    Result: 0x{result:08x}")

    # Check phase rddata after FSM
    print("\n  Phase rddata after FSM:")
    for phase in range(4):
        rddata = wb.read(pi_base(phase) + 0x14)
        print(f"    Phase {phase}: 0x{rddata:08x}")

    # Read memory via SW mode to see what's actually in DRAM
    print("\n=== Read Same Location via SW Mode ===")
    result_sw = sw_mode_write_read(wb, row=0, col=0, bank=0, data=0x12345678)
    # Write a marker, then read
    wb.write(DFII_CONTROL, 0x0E)  # SW mode
    time.sleep(0.01)

    # Read without writing first
    wb.write(pi_base(0) + 0x08, 0x400)
    wb.write(pi_base(0) + 0x0c, 0)
    wb.write(pi_base(0) + 0x00, 0x0B)
    wb.write(pi_base(0) + 0x04, 1)
    time.sleep(0.001)

    wb.write(pi_base(0) + 0x08, 0)  # row 0
    wb.write(pi_base(0) + 0x0c, 0)  # bank 0
    wb.write(pi_base(0) + 0x00, 0x09)  # ACT
    wb.write(pi_base(0) + 0x04, 1)
    time.sleep(0.001)

    wb.write(pi_base(2) + 0x08, 0)  # col 0
    wb.write(pi_base(2) + 0x0c, 0)  # bank 0
    wb.write(pi_base(2) + 0x00, 0x25)  # READ
    wb.write(pi_base(2) + 0x04, 1)
    time.sleep(0.01)

    result_now = wb.read(pi_base(0) + 0x14)
    print(f"  Current value at row=0, col=0, bank=0: 0x{result_now:08x}")

    wb.write(DFII_CONTROL, 0x0F)  # Back to HW mode

    # Analysis
    print("\n=== Analysis ===")
    if result_now == 0xFFFFFFFF:
        print("  Data shows 0xFFFFFFFF - FSM write DID reach DRAM!")
        print("  Issue is read capture, not write path.")
    elif result_now == 0x00000000:
        print("  Data shows 0x00000000 - FSM write did NOT reach DRAM.")
        print("  Issue is in ext_dfi command routing.")
    else:
        print(f"  Data shows 0x{result_now:08x} - unexpected value.")
        print("  Something else wrote to this location.")

    wb.close()
    print("\nDone!")


if __name__ == "__main__":
    main()
