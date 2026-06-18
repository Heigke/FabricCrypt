#!/usr/bin/env python3
"""
z1210: DDR3 Initialization via litex_server/client

Uses the proper LiteX infrastructure for Etherbone communication.
Requires litex_server running:
  litex_server --udp --udp-ip 192.168.0.50 --udp-port 1234

This implements the init_sequence() from sdram_phy.h for proper DDR3 initialization.
"""

import time
import json
from datetime import datetime
from litex.tools.litex_client import RemoteClient

# CSR addresses from build_partial_write_fsm/csr.h
CSR_BASE = 0x0
DDRPHY_BASE = CSR_BASE + 0x800
SDRAM_BASE = CSR_BASE + 0x3000
LEDS_BASE = CSR_BASE + 0x2000
IDENTIFIER = CSR_BASE + 0x1800

# DDRPHY CSRs (with anonymized names mapped)
# From sdram_phy.h: ddrphy_rst, ddrphy_dly_sel, etc.
PHY_RST = DDRPHY_BASE + 0x00  # csrstorage_4
PHY_DLY_SEL = DDRPHY_BASE + 0x04  # csrstorage_5
PHY_RDLY_DQ_RST = DDRPHY_BASE + 0x10  # csr_8
PHY_RDLY_DQ_INC = DDRPHY_BASE + 0x14  # csr_9
PHY_RDLY_BITSLIP_RST = DDRPHY_BASE + 0x18  # csr_10
PHY_RDLY_BITSLIP = DDRPHY_BASE + 0x1c  # csr_11
PHY_WDLY_BITSLIP_RST = DDRPHY_BASE + 0x20  # csr_12
PHY_WDLY_BITSLIP = DDRPHY_BASE + 0x24  # csr_13
PHY_RDPHASE = DDRPHY_BASE + 0x2c  # csrstorage_15
PHY_WRPHASE = DDRPHY_BASE + 0x30  # csrstorage_16

# SDRAM DFII CSRs
DFII_CONTROL = SDRAM_BASE + 0x00  # csrstorage_22

# Phase injector addresses - each phase has: command, command_issue, address, baddress, wrdata, rddata
DFII_PI0_COMMAND = SDRAM_BASE + 0x04
DFII_PI0_COMMAND_ISSUE = SDRAM_BASE + 0x08
DFII_PI0_ADDRESS = SDRAM_BASE + 0x0c
DFII_PI0_BADDRESS = SDRAM_BASE + 0x10
DFII_PI0_WRDATA = SDRAM_BASE + 0x14
DFII_PI0_RDDATA = SDRAM_BASE + 0x18

# Phase spacing is 0x18 (24 bytes per phase)
PHASE_STRIDE = 0x18

# DFII Control bits
DFII_CONTROL_SEL = 0x01
DFII_CONTROL_CKE = 0x02
DFII_CONTROL_ODT = 0x04
DFII_CONTROL_RESET_N = 0x08

# DFII Command bits
DFII_COMMAND_CS = 0x01
DFII_COMMAND_WE = 0x02
DFII_COMMAND_CAS = 0x04
DFII_COMMAND_RAS = 0x08
DFII_COMMAND_WRDATA = 0x10
DFII_COMMAND_RDDATA = 0x20


def cdelay(n):
    """Delay for approximately n cycles at 100MHz."""
    time.sleep(n * 0.00001)


def command_p0(wb, cmd):
    """Issue command on phase 0."""
    wb.write(DFII_PI0_COMMAND, cmd)
    wb.write(DFII_PI0_COMMAND_ISSUE, 1)


def init_sequence(wb):
    """
    DDR3 initialization sequence from sdram_phy.h init_sequence().
    This is the exact sequence the BIOS runs.
    """
    print("\n=== DDR3 Initialization Sequence ===")

    # Step 1: Release reset, keep CKE low
    print("  1. Release RESET_N, CKE low...")
    wb.write(DFII_CONTROL, DFII_CONTROL_ODT | DFII_CONTROL_RESET_N)
    cdelay(50000)

    # Step 2: Bring CKE high
    print("  2. CKE high...")
    wb.write(DFII_CONTROL, DFII_CONTROL_CKE | DFII_CONTROL_ODT | DFII_CONTROL_RESET_N)
    cdelay(10000)

    # Step 3: Load Mode Register 2 (MR2)
    # From sdram_phy.h: address=0x200, bank=2 (CWL=5)
    print("  3. Load MR2 (CWL=5)...")
    wb.write(DFII_PI0_ADDRESS, 0x200)
    wb.write(DFII_PI0_BADDRESS, 2)
    command_p0(wb, DFII_COMMAND_RAS | DFII_COMMAND_CAS | DFII_COMMAND_WE | DFII_COMMAND_CS)
    cdelay(200)

    # Step 4: Load Mode Register 3 (MR3)
    # From sdram_phy.h: address=0x0, bank=3 (MPR disabled)
    print("  4. Load MR3 (MPR disabled)...")
    wb.write(DFII_PI0_ADDRESS, 0x0)
    wb.write(DFII_PI0_BADDRESS, 3)
    command_p0(wb, DFII_COMMAND_RAS | DFII_COMMAND_CAS | DFII_COMMAND_WE | DFII_COMMAND_CS)
    cdelay(200)

    # Step 5: Load Mode Register 1 (MR1)
    # From sdram_phy.h: address=0x6, bank=1 (DLL on, Rtt=60ohm, ODS=RZQ/7)
    print("  5. Load MR1 (DLL on, Rtt=60ohm)...")
    wb.write(DFII_PI0_ADDRESS, 0x6)
    wb.write(DFII_PI0_BADDRESS, 1)
    command_p0(wb, DFII_COMMAND_RAS | DFII_COMMAND_CAS | DFII_COMMAND_WE | DFII_COMMAND_CS)
    cdelay(200)

    # Step 6: Load Mode Register 0 (MR0)
    # From sdram_phy.h: address=0x930, bank=0 (CL=7, BL=8, DLL reset)
    print("  6. Load MR0 (CL=7, BL=8, DLL reset)...")
    wb.write(DFII_PI0_ADDRESS, 0x930)
    wb.write(DFII_PI0_BADDRESS, 0)
    command_p0(wb, DFII_COMMAND_RAS | DFII_COMMAND_CAS | DFII_COMMAND_WE | DFII_COMMAND_CS)
    cdelay(200)

    # Step 7: ZQ Calibration Long
    # From sdram_phy.h: address=0x400 (A10=1 for ZQ long)
    print("  7. ZQ Calibration Long...")
    wb.write(DFII_PI0_ADDRESS, 0x400)
    wb.write(DFII_PI0_BADDRESS, 0)
    command_p0(wb, DFII_COMMAND_WE | DFII_COMMAND_CS)
    cdelay(600)  # tZQINIT = 512 CK = ~5.12us at 100MHz

    print("  Initialization complete!")


def read_leveling(wb):
    """
    Simplified read leveling: sweep DQ delay and bitslip to find valid settings.
    """
    print("\n=== Read Leveling ===")

    results = []
    modules = 2  # Two byte lanes (16-bit DDR3)

    for module in range(modules):
        print(f"\n  Module {module}:")

        # Select this module
        wb.write(PHY_DLY_SEL, 1 << module)

        best_delay = 0
        best_bitslip = 0
        best_score = -1

        for bitslip in range(8):
            # Reset and set bitslip
            wb.write(PHY_RDLY_BITSLIP_RST, 1)
            for _ in range(bitslip):
                wb.write(PHY_RDLY_BITSLIP, 1)

            for delay in range(32):
                # Reset and set delay
                wb.write(PHY_RDLY_DQ_RST, 1)
                for _ in range(delay):
                    wb.write(PHY_RDLY_DQ_INC, 1)

                # Do a test read via DFII
                # First, precharge all
                wb.write(DFII_PI0_ADDRESS, 0x400)  # A10=1 for all banks
                wb.write(DFII_PI0_BADDRESS, 0)
                command_p0(wb, DFII_COMMAND_RAS | DFII_COMMAND_WE | DFII_COMMAND_CS)
                cdelay(20)

                # Activate row 0, bank 0
                wb.write(DFII_PI0_ADDRESS, 0)
                wb.write(DFII_PI0_BADDRESS, 0)
                command_p0(wb, DFII_COMMAND_RAS | DFII_COMMAND_CS)
                cdelay(20)

                # Read column 0
                wb.write(DFII_PI0_ADDRESS, 0)
                command_p0(wb, DFII_COMMAND_CAS | DFII_COMMAND_CS | DFII_COMMAND_RDDATA)
                cdelay(20)

                # Check read data
                data = wb.read(DFII_PI0_RDDATA)

                # Precharge
                wb.write(DFII_PI0_ADDRESS, 0x400)
                command_p0(wb, DFII_COMMAND_RAS | DFII_COMMAND_WE | DFII_COMMAND_CS)
                cdelay(20)

                # Score based on non-zero data and pattern quality
                score = 0
                if data != 0x00000000:
                    score = 1
                    # Bonus for alternating patterns
                    if data == 0xAAAAAAAA or data == 0x55555555:
                        score = 2

                if score > best_score:
                    best_score = score
                    best_delay = delay
                    best_bitslip = bitslip

        # Apply best settings
        wb.write(PHY_RDLY_BITSLIP_RST, 1)
        for _ in range(best_bitslip):
            wb.write(PHY_RDLY_BITSLIP, 1)
        wb.write(PHY_RDLY_DQ_RST, 1)
        for _ in range(best_delay):
            wb.write(PHY_RDLY_DQ_INC, 1)

        results.append({
            "module": module,
            "delay": best_delay,
            "bitslip": best_bitslip,
            "score": best_score
        })
        print(f"    Best: delay={best_delay}, bitslip={best_bitslip}, score={best_score}")

    return results


def memtest(wb, base_addr=0x40000000, count=16):
    """Test memory access."""
    print(f"\n=== Memory Test (base=0x{base_addr:08x}) ===")

    # Switch to hardware control
    wb.write(DFII_CONTROL, DFII_CONTROL_SEL | DFII_CONTROL_CKE | DFII_CONTROL_ODT | DFII_CONTROL_RESET_N)
    time.sleep(0.01)

    passed = 0
    failed = 0

    patterns = [0xCAFEBABE, 0x12345678, 0xAAAA5555, 0x55AAAA55,
                0xDEADBEEF, 0xF0F0F0F0, 0x0F0F0F0F, 0xFFFFFFFF,
                0x00000000, 0x11111111, 0x22222222, 0x33333333,
                0x44444444, 0x55555555, 0x66666666, 0x77777777]

    for i, pattern in enumerate(patterns[:count]):
        addr = base_addr + i * 4

        # Write
        wb.write(addr, pattern)
        time.sleep(0.005)

        # Read back
        readback = wb.read(addr)

        if readback == pattern:
            passed += 1
            status = "PASS"
        else:
            failed += 1
            status = "FAIL"

        print(f"  [{status}] 0x{addr:08x}: write=0x{pattern:08x} read=0x{readback:08x}")

    print(f"\n  Result: {passed}/{count} passed")
    return passed == count


def main():
    print("=" * 60)
    print("z1210: DDR3 Initialization via litex_server/client")
    print("=" * 60)
    print("\nNOTE: Requires litex_server running:")
    print("  litex_server --udp --udp-ip 192.168.0.50 --udp-port 1234")

    try:
        wb = RemoteClient(host='localhost', port=1234)
        wb.open()
        print("\nConnected to litex_server!")

        # Verify connection
        ident = wb.read(IDENTIFIER)
        print(f"Identifier: 0x{ident:08x}")

        # LED test - visual confirmation
        wb.write(LEDS_BASE, 0x01)

        # Enter software control mode (disable hardware controller)
        print("\nEntering software control mode...")
        wb.write(DFII_CONTROL, 0)  # Disable all, software control
        time.sleep(0.01)

        # Run initialization sequence
        init_sequence(wb)

        # Run read leveling
        leveling = read_leveling(wb)

        # Test memory
        memtest_ok = memtest(wb)

        # LED indication
        if memtest_ok:
            wb.write(LEDS_BASE, 0x0F)  # All LEDs on for success
        else:
            wb.write(LEDS_BASE, 0x05)  # Alternating for failure

        # Save results
        output = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "experiment": "z1210 DDR3 Init via LiteX Server",
            "leveling": leveling,
            "memtest_passed": memtest_ok
        }

        with open("results/z1210_ddr3_litex_init.json", "w") as f:
            json.dump(output, f, indent=2)

        print(f"\nResults saved to results/z1210_ddr3_litex_init.json")

        if memtest_ok:
            print("\n*** DDR3 INITIALIZATION SUCCESSFUL! ***")
        else:
            print("\n*** Memtest failed - check leveling ***")

        wb.close()

    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        return


if __name__ == "__main__":
    main()
