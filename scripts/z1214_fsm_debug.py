#!/usr/bin/env python3
"""
z1214: Debug the ext_dfi FSM to understand why commands aren't routing.

The FSM is stuck at state 10 (ACTIVATE for partial write).
Let's investigate:
1. Check all FSM debug registers
2. Try manually triggering and observing state transitions
3. Check if DFII is properly set for ext_dfi control
"""

import time
from litex.tools.litex_client import RemoteClient

# CSR addresses from csr.csv
DFII_CONTROL = 0x3000
DFII_PI0_COMMAND = 0x3004
DFII_PI0_COMMAND_ISSUE = 0x3008
DFII_PI0_ADDRESS = 0x300c
DFII_PI0_BADDRESS = 0x3010
DFII_PI0_WRDATA = 0x3014
DFII_PI0_RDDATA = 0x3018

# Partial Write FSM CSRs (offset from 0x2800)
PW_BASE = 0x2800
PW_CONFIG = PW_BASE + 0x00
PW_WRITE_DATA = PW_BASE + 0x04
PW_REF_DATA = PW_BASE + 0x08
PW_CONTROL = PW_BASE + 0x0c
PW_STATUS = PW_BASE + 0x10
PW_RESULT = PW_BASE + 0x14
PW_RESULT_P1 = PW_BASE + 0x18
PW_RESULT_P2 = PW_BASE + 0x1c
PW_RESULT_P3 = PW_BASE + 0x20
PW_DEBUG = PW_BASE + 0x24
PW_EDGE_COUNT = PW_BASE + 0x28


def main():
    print("=" * 60)
    print("z1214: ext_dfi FSM Debug")
    print("=" * 60)

    wb = RemoteClient(host='localhost', port=1234)
    wb.open()
    print(f"\nConnected! Identifier: 0x{wb.read(0x1800):08x}")

    # Read current FSM state
    print("\n=== Current FSM State ===")
    status = wb.read(PW_STATUS)
    debug = wb.read(PW_DEBUG)
    edge_count = wb.read(PW_EDGE_COUNT)
    config = wb.read(PW_CONFIG)

    busy = status & 0x01
    done = (status >> 1) & 0x01
    state = debug & 0xFF

    print(f"  Status: 0x{status:08x}")
    print(f"    busy={busy}, done={done}")
    print(f"  Debug (state): {state}")
    print(f"  Edge count: {edge_count}")
    print(f"  Config: 0x{config:08x}")

    # Decode config
    tras = config & 0x1F
    bank = (config >> 5) & 0x7
    col = (config >> 8) & 0x3FF
    row = (config >> 18) & 0x3FFF
    print(f"    tras={tras}, bank={bank}, col={col}, row={row}")

    # Check DFII control
    dfii_ctrl = wb.read(DFII_CONTROL)
    print(f"\n  DFII Control: 0x{dfii_ctrl:02x}")
    print(f"    SEL={(dfii_ctrl >> 0) & 1} (1=HW, 0=SW)")
    print(f"    CKE={(dfii_ctrl >> 1) & 1}")
    print(f"    ODT={(dfii_ctrl >> 2) & 1}")
    print(f"    RESET_N={(dfii_ctrl >> 3) & 1}")

    # Read result registers
    print("\n=== Result Registers ===")
    result_p0 = wb.read(PW_RESULT)
    result_p1 = wb.read(PW_RESULT_P1)
    result_p2 = wb.read(PW_RESULT_P2)
    result_p3 = wb.read(PW_RESULT_P3)
    print(f"  Phase 0: 0x{result_p0:08x}")
    print(f"  Phase 1: 0x{result_p1:08x}")
    print(f"  Phase 2: 0x{result_p2:08x}")
    print(f"  Phase 3: 0x{result_p3:08x}")

    # Try to reset the FSM by clearing control
    print("\n=== Reset FSM ===")
    wb.write(PW_CONTROL, 0)
    time.sleep(0.01)
    state_after = wb.read(PW_DEBUG) & 0xFF
    print(f"  After control=0: state={state_after}")

    # Configure test parameters
    print("\n=== Configure Test ===")
    # config = [row:14][col:10][bank:3][tras:5]
    row = 0
    col = 0
    bank = 0
    tras = 5  # 5 cycles truncated tRAS
    config = (row << 18) | (col << 8) | (bank << 5) | tras
    wb.write(PW_CONFIG, config)
    wb.write(PW_REF_DATA, 0x00000000)
    wb.write(PW_WRITE_DATA, 0xFFFFFFFF)

    print(f"  Row={row}, Col={col}, Bank={bank}, tRAS={tras}")
    print(f"  Config written: 0x{config:08x}")

    # First ensure DFII is in HW mode
    print("\n=== Pre-trigger Check ===")
    wb.write(DFII_CONTROL, 0x0F)  # Ensure HW mode
    time.sleep(0.01)
    dfii_ctrl = wb.read(DFII_CONTROL)
    print(f"  DFII Control: 0x{dfii_ctrl:02x} (should be 0x0F)")

    # Trigger FSM
    print("\n=== Trigger FSM ===")
    edge_before = wb.read(PW_EDGE_COUNT)
    wb.write(PW_CONTROL, 1)  # Set start bit
    time.sleep(0.001)  # Small delay

    # Check if edge was detected
    edge_after = wb.read(PW_EDGE_COUNT)
    print(f"  Edge count: {edge_before} -> {edge_after}")

    # Monitor state progression
    print("\n=== State Progression ===")
    prev_state = -1
    for i in range(50):
        status = wb.read(PW_STATUS)
        state = wb.read(PW_DEBUG) & 0xFF
        dfii_ctrl = wb.read(DFII_CONTROL)

        if state != prev_state:
            print(f"  [{i*10:4d}ms] State {state}, Status 0x{status:02x}, DFII 0x{dfii_ctrl:02x}")
            prev_state = state

        done = (status >> 1) & 0x01
        if done:
            print("  FSM completed!")
            break

        time.sleep(0.01)

    # Clear start bit
    wb.write(PW_CONTROL, 0)

    # Read final results
    print("\n=== Final Results ===")
    status = wb.read(PW_STATUS)
    state = wb.read(PW_DEBUG) & 0xFF
    result_p0 = wb.read(PW_RESULT)
    result_p1 = wb.read(PW_RESULT_P1)
    result_p2 = wb.read(PW_RESULT_P2)
    result_p3 = wb.read(PW_RESULT_P3)

    print(f"  Final state: {state}")
    print(f"  Final status: 0x{status:02x}")
    print(f"  Results:")
    print(f"    Phase 0: 0x{result_p0:08x}")
    print(f"    Phase 1: 0x{result_p1:08x}")
    print(f"    Phase 2: 0x{result_p2:08x}")
    print(f"    Phase 3: 0x{result_p3:08x}")

    # Also check DFII phase rddata directly
    print("\n=== DFII Phase RdData ===")
    for phase in range(4):
        rddata_addr = DFII_PI0_RDDATA + phase * 0x18
        rddata = wb.read(rddata_addr)
        print(f"  Phase {phase} rddata: 0x{rddata:08x}")

    # Try direct SW mode read to verify memory works
    print("\n=== Verify Memory via SW Mode ===")
    # Enter SW mode
    wb.write(DFII_CONTROL, 0x0E)  # SEL=0
    time.sleep(0.01)

    # Precharge all
    wb.write(DFII_PI0_ADDRESS, 0x400)  # A10=1
    wb.write(DFII_PI0_BADDRESS, 0)
    wb.write(DFII_PI0_COMMAND, 0x0B)  # PRE: RAS|WE|CS
    wb.write(DFII_PI0_COMMAND_ISSUE, 1)
    time.sleep(0.001)

    # Activate row 0, bank 0
    wb.write(DFII_PI0_ADDRESS, 0)
    wb.write(DFII_PI0_BADDRESS, 0)
    wb.write(DFII_PI0_COMMAND, 0x09)  # ACT: RAS|CS
    wb.write(DFII_PI0_COMMAND_ISSUE, 1)
    time.sleep(0.001)

    # Read col 0 (use phase 2 for read)
    rdphase = 2
    pi_cmd = 0x3004 + rdphase * 0x18
    pi_issue = 0x3008 + rdphase * 0x18
    pi_addr = 0x300c + rdphase * 0x18
    pi_bank = 0x3010 + rdphase * 0x18

    wb.write(pi_addr, 0)
    wb.write(pi_bank, 0)
    wb.write(pi_cmd, 0x25)  # READ: CAS|CS|RDDATA
    wb.write(pi_issue, 1)
    time.sleep(0.01)

    # Read data from all phases
    print("  After SW mode read command:")
    for phase in range(4):
        rddata_addr = DFII_PI0_RDDATA + phase * 0x18
        rddata = wb.read(rddata_addr)
        print(f"    Phase {phase} rddata: 0x{rddata:08x}")

    # Return to HW mode
    wb.write(DFII_CONTROL, 0x0F)

    wb.close()
    print("\nDone!")


if __name__ == "__main__":
    main()
