#!/usr/bin/env python3
"""
z1168: DDR3 Diagnostic - Check memory state and initialization
"""

import sys
import time
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
DDRPHY_RST = DDRPHY_BASE + 0x00  # csrstorage_39
DDRPHY_DLY_SEL = DDRPHY_BASE + 0x04
DDRPHY_RDLY_RST = DDRPHY_BASE + 0x14
DDRPHY_RDLY_INC = DDRPHY_BASE + 0x18

DFII_SEL = 0x01
DFII_CKE = 0x02
DFII_ODT = 0x04
DFII_RESET_N = 0x08

CMD_CS = 0x01
CMD_WE = 0x02
CMD_CAS = 0x04
CMD_RAS = 0x08
CMD_REFRESH = CMD_RAS | CMD_CAS | CMD_CS
CMD_PRECHARGE = CMD_RAS | CMD_WE | CMD_CS


def set_idelay(wb, taps):
    """Set IDELAY to specified tap count (0-31)"""
    for dqs in range(2):
        wb.write(DDRPHY_DLY_SEL, 1 << dqs)
        wb.write(DDRPHY_RDLY_RST, 1)
        time.sleep(0.001)
        for _ in range(taps):
            wb.write(DDRPHY_RDLY_INC, 1)


def main():
    print("=" * 60)
    print("z1168: DDR3 Diagnostic")
    print("=" * 60)

    wb = RemoteClient()
    wb.open()
    print("Connected to Etherbone")

    # Read DFII state
    print("\n=== DFII State ===")
    dfii = wb.read(DFII_CONTROL)
    print(f"DFII Control: 0x{dfii:02x}")
    print(f"  SEL (HW mode): {bool(dfii & DFII_SEL)}")
    print(f"  CKE (Clock En): {bool(dfii & DFII_CKE)}")
    print(f"  ODT: {bool(dfii & DFII_ODT)}")
    print(f"  RESET_N: {bool(dfii & DFII_RESET_N)}")

    # Fix DFII if needed
    if not (dfii & DFII_CKE) or not (dfii & DFII_RESET_N):
        print("\n*** FIXING DFII: Enabling CKE and RESET_N ***")
        correct_dfii = DFII_SEL | DFII_CKE | DFII_RESET_N  # 0x0B
        wb.write(DFII_CONTROL, correct_dfii)
        time.sleep(0.1)
        dfii = wb.read(DFII_CONTROL)
        print(f"DFII Control now: 0x{dfii:02x}")

    # Issue refresh to ensure DDR3 is stable
    print("\n=== Issuing Refreshes ===")
    sw_mode = DFII_CKE | DFII_RESET_N
    hw_mode = DFII_SEL | DFII_CKE | DFII_RESET_N

    for i in range(5):
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
        wb.write(DFII_CONTROL, hw_mode)
        time.sleep(0.01)

    print("Issued 5 refreshes")

    # Test basic memory at different addresses
    print("\n=== Basic Memory Test ===")
    test_addrs = [
        DDR3_BASE + 0x000000,
        DDR3_BASE + 0x100000,
        DDR3_BASE + 0x200000,
        DDR3_BASE + 0x400000,
    ]

    for addr in test_addrs:
        wb.write(addr, 0xDEADBEEF)
        result = wb.read(addr)
        status = "OK" if result == 0xDEADBEEF else f"FAIL (got 0x{result:08x})"
        print(f"  0x{addr:08x}: {status}")

    # Test various patterns
    print("\n=== Pattern Test at 0x{:08x} ===".format(DDR3_BASE))
    patterns = [
        0xFFFFFFFF,
        0x00000000,
        0xAAAAAAAA,
        0x55555555,
        0xDEADBEEF,
        0x12345678,
    ]

    for pat in patterns:
        wb.write(DDR3_BASE, pat)
        result = wb.read(DDR3_BASE)
        status = "OK" if result == pat else f"FAIL (got 0x{result:08x})"
        print(f"  0x{pat:08x}: {status}")

    # Test IDELAY settings
    print("\n=== IDELAY Sweep ===")
    test_addr = DDR3_BASE + 0x1000
    test_pat = 0xAAAAAAAA

    for taps in range(0, 32, 4):
        set_idelay(wb, taps)
        wb.write(test_addr, test_pat)
        result = wb.read(test_addr)
        errors = bin(test_pat ^ result).count('1') if result != test_pat else 0
        status = "OK" if errors == 0 else f"{errors} bit errors (0x{result:08x})"
        print(f"  Tap {taps:2d} (~{taps*78}ps): {status}")

    # Reset IDELAY to optimal
    set_idelay(wb, 0)

    # Final test after diagnostics
    print("\n=== Final Memory Test ===")
    for addr in test_addrs[:2]:
        wb.write(addr, 0xCAFEBABE)
        result = wb.read(addr)
        status = "OK" if result == 0xCAFEBABE else f"FAIL (got 0x{result:08x})"
        print(f"  0x{addr:08x}: {status}")

    wb.close()
    print("\nDone!")


if __name__ == "__main__":
    main()
