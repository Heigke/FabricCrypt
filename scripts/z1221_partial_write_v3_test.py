#!/usr/bin/env python3
"""
z1221: Test Partial Write v3 with Hardware FSM

This tests the cycle-accurate partial write functionality.
Key parameter: tRAS (row active time) controls how much of the write completes.

CSRs (from csr.h):
- 0x2800: config  [row:14][col:10][bank:3][tras:5]
- 0x2804: write_data (32-bit)
- 0x2808: ref_data (unused)
- 0x280c: control [start:1]
- 0x2810: status [busy:1][done:1][state:4]
- 0x2814: result (32-bit read result)
- 0x2818: debug (state)
"""

import time
from litex.tools.litex_client import RemoteClient

# DFII CSRs
DFII_CONTROL = 0x3000
PHASE_SIZE = 0x18

def pi_base(phase):
    return 0x3004 + phase * PHASE_SIZE

# Partial Write v3 CSRs
PW_CONFIG = 0x2800
PW_WRITE_DATA = 0x2804
PW_REF_DATA = 0x2808
PW_CONTROL = 0x280c
PW_STATUS = 0x2810
PW_RESULT = 0x2814
PW_DEBUG = 0x2818


def make_config(row, col, bank, tras):
    """Pack config: [row:14][col:10][bank:3][tras:5]"""
    return (row << 18) | (col << 8) | (bank << 5) | tras


def sw_write(wb, row, col, bank, data):
    """Write via SW mode for reference."""
    WRPHASE = 3
    wb.write(DFII_CONTROL, 0x0E)
    time.sleep(0.001)

    # PRE ALL
    wb.write(pi_base(0) + 0x08, 0x400)
    wb.write(pi_base(0) + 0x0c, 0)
    wb.write(pi_base(0) + 0x00, 0x0B)
    wb.write(pi_base(0) + 0x04, 1)
    time.sleep(0.0001)

    # ACT
    wb.write(pi_base(0) + 0x08, row)
    wb.write(pi_base(0) + 0x0c, bank)
    wb.write(pi_base(0) + 0x00, 0x09)
    wb.write(pi_base(0) + 0x04, 1)
    time.sleep(0.0001)

    # WRITE
    wb.write(pi_base(WRPHASE) + 0x10, data)
    wb.write(pi_base(WRPHASE) + 0x08, col)
    wb.write(pi_base(WRPHASE) + 0x0c, bank)
    wb.write(pi_base(WRPHASE) + 0x00, 0x17)
    wb.write(pi_base(WRPHASE) + 0x04, 1)
    time.sleep(0.001)

    # PRE
    wb.write(pi_base(0) + 0x08, 0x400)
    wb.write(pi_base(0) + 0x0c, 0)
    wb.write(pi_base(0) + 0x00, 0x0B)
    wb.write(pi_base(0) + 0x04, 1)
    time.sleep(0.0001)

    wb.write(DFII_CONTROL, 0x0F)


def sw_read(wb, row, col, bank):
    """Read via SW mode for verification."""
    RDPHASE = 2
    wb.write(DFII_CONTROL, 0x0E)
    time.sleep(0.001)

    # PRE ALL
    wb.write(pi_base(0) + 0x08, 0x400)
    wb.write(pi_base(0) + 0x0c, 0)
    wb.write(pi_base(0) + 0x00, 0x0B)
    wb.write(pi_base(0) + 0x04, 1)
    time.sleep(0.0001)

    # ACT
    wb.write(pi_base(0) + 0x08, row)
    wb.write(pi_base(0) + 0x0c, bank)
    wb.write(pi_base(0) + 0x00, 0x09)
    wb.write(pi_base(0) + 0x04, 1)
    time.sleep(0.0001)

    # READ
    wb.write(pi_base(RDPHASE) + 0x08, col)
    wb.write(pi_base(RDPHASE) + 0x0c, bank)
    wb.write(pi_base(RDPHASE) + 0x00, 0x25)
    wb.write(pi_base(RDPHASE) + 0x04, 1)
    time.sleep(0.001)

    result = wb.read(pi_base(0) + 0x14)

    # PRE
    wb.write(pi_base(0) + 0x08, 0x400)
    wb.write(pi_base(0) + 0x0c, 0)
    wb.write(pi_base(0) + 0x00, 0x0B)
    wb.write(pi_base(0) + 0x04, 1)
    time.sleep(0.0001)

    wb.write(DFII_CONTROL, 0x0F)
    return result


def run_fsm(wb, row, col, bank, tras, write_data):
    """Run the partial write FSM."""
    config = make_config(row, col, bank, tras)
    wb.write(PW_CONFIG, config)
    wb.write(PW_WRITE_DATA, write_data)
    wb.write(PW_CONTROL, 0)
    time.sleep(0.001)
    wb.write(PW_CONTROL, 1)

    # Wait for done
    for _ in range(100):
        status = wb.read(PW_STATUS)
        if (status >> 1) & 1:  # done bit
            break
        time.sleep(0.001)

    result = wb.read(PW_RESULT)
    debug = wb.read(PW_DEBUG)

    wb.write(PW_CONTROL, 0)
    return result, status, debug


def main():
    print("=" * 60)
    print("z1221: Partial Write v3 Test")
    print("=" * 60)

    wb = RemoteClient(host='localhost', port=1234)
    wb.open()

    ident = wb.read(0x1800)
    print(f"\nConnected! Identifier: 0x{ident:08x}")

    # Init DDR3
    print("\n=== Initialize DDR3 ===")
    import subprocess
    result = subprocess.run(
        ["python3", "scripts/z1210_ddr3_litex_init.py"],
        capture_output=True,
        text=True
    )
    if "SUCCESSFUL" in result.stdout:
        print("  DDR3 init successful")
    else:
        print("  DDR3 init may have failed")

    # Test parameters
    row, col, bank = 300, 0, 0

    # Test 1: Verify SW mode still works
    print("\n=== Test 1: SW Mode Verification ===")
    sw_write(wb, row, col, bank, 0x12345678)
    val = sw_read(wb, row, col, bank)
    print(f"  SW Write 0x12345678, Read 0x{val:08x}")
    if val != 0x12345678:
        print("  ERROR: SW mode broken!")
        wb.close()
        return
    print("  SW mode OK")

    # Test 2: Basic FSM operation with full tRAS
    print("\n=== Test 2: FSM with Full tRAS (31 cycles) ===")
    # First write reference via SW
    sw_write(wb, row, col, bank, 0x00000000)

    # Run FSM with maximum tRAS
    result, status, debug = run_fsm(wb, row, col, bank, tras=31, write_data=0xFFFFFFFF)
    print(f"  FSM result: 0x{result:08x}")
    print(f"  FSM status: 0x{status:02x} (done={(status>>1)&1}, busy={(status>>0)&1})")
    print(f"  FSM debug: {debug}")

    # Verify with SW read
    sw_val = sw_read(wb, row, col, bank)
    print(f"  SW verify: 0x{sw_val:08x}")

    if sw_val == 0xFFFFFFFF:
        print("  *** FSM WRITE WORKS! ***")
    else:
        print("  FSM write may have issues")

    # Test 3: tRAS sweep to observe partial writes
    print("\n=== Test 3: tRAS Sweep (Decay Observation) ===")
    print("  tras | FSM result   | SW verify    | bits set")
    print("  -----|--------------|--------------|----------")

    for tras in [31, 20, 15, 10, 5, 3, 2, 1, 0]:
        # Write 0x00000000 first
        sw_write(wb, row, col, bank, 0x00000000)

        # Run FSM to write 0xFFFFFFFF with limited tRAS
        result, _, _ = run_fsm(wb, row, col, bank, tras=tras, write_data=0xFFFFFFFF)

        # Verify
        sw_val = sw_read(wb, row, col, bank)
        bits = bin(sw_val).count('1')

        print(f"  {tras:4d} | 0x{result:08x} | 0x{sw_val:08x} | {bits:2d}/32")

    # Test 4: Decay pattern
    print("\n=== Test 4: Bit Pattern Analysis ===")
    # Write known pattern
    sw_write(wb, row, col, bank, 0xAAAAAAAA)

    # Run FSM with different tRAS values
    for tras in [5, 3, 1]:
        # Don't clear first - overwrite the pattern
        result, _, _ = run_fsm(wb, row, col, bank, tras=tras, write_data=0x55555555)
        sw_val = sw_read(wb, row, col, bank)
        print(f"  tras={tras}: FSM=0x{result:08x}, SW=0x{sw_val:08x}")

    wb.close()
    print("\nDone!")


if __name__ == "__main__":
    main()
