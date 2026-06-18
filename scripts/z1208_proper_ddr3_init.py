#!/usr/bin/env python3
"""
z1208: Proper DDR3 Initialization - Ported from Generated sdram_phy.h

This script implements the EXACT init_sequence() from the generated sdram_phy.h file,
plus read leveling from liblitedram/sdram.c.

Key parameters from sdram_phy.h:
- CL=7 (CAS Latency)
- CWL=5 (CAS Write Latency)
- BL=8 (Burst Length)
- RDPHASE=2, WRPHASE=3
- DELAYS=32, BITSLIPS=8

Reference: https://github.com/enjoy-digital/litex/blob/master/litex/soc/software/liblitedram/sdram.c
"""

import socket
import struct
import time
import json
from datetime import datetime

# Etherbone connection
FPGA_IP = "192.168.0.50"
FPGA_PORT = 1234

# CSR addresses from csr.h
CSR_BASE = 0x0

# DDRPHY CSRs
DDRPHY_BASE = CSR_BASE + 0x800
DDRPHY_RST = DDRPHY_BASE + 0x00
DDRPHY_DLY_SEL = DDRPHY_BASE + 0x04
DDRPHY_RDLY_DQ_RST = DDRPHY_BASE + 0x10
DDRPHY_RDLY_DQ_INC = DDRPHY_BASE + 0x14
DDRPHY_RDLY_DQ_BITSLIP_RST = DDRPHY_BASE + 0x18
DDRPHY_RDLY_DQ_BITSLIP = DDRPHY_BASE + 0x1c
DDRPHY_RDPHASE = DDRPHY_BASE + 0x2c
DDRPHY_WRPHASE = DDRPHY_BASE + 0x30

# SDRAM DFII CSRs - corrected offsets
SDRAM_BASE = CSR_BASE + 0x3000
DFII_CONTROL = SDRAM_BASE + 0x00

# Phase 0: command=0x04, issue=0x08, address=0x0c, baddress=0x10, wrdata=0x14, rddata=0x18
DFII_PI0_COMMAND = SDRAM_BASE + 0x04
DFII_PI0_COMMAND_ISSUE = SDRAM_BASE + 0x08
DFII_PI0_ADDRESS = SDRAM_BASE + 0x0c
DFII_PI0_BADDRESS = SDRAM_BASE + 0x10
DFII_PI0_WRDATA = SDRAM_BASE + 0x14
DFII_PI0_RDDATA = SDRAM_BASE + 0x18

# Phase 1-3 (each 0x18 bytes apart from phase 0 start)
DFII_PI1_COMMAND = SDRAM_BASE + 0x1c
DFII_PI1_RDDATA = SDRAM_BASE + 0x30
DFII_PI2_RDDATA = SDRAM_BASE + 0x48
DFII_PI3_RDDATA = SDRAM_BASE + 0x60

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

# PHY parameters from sdram_phy.h
SDRAM_PHY_RDPHASE = 2
SDRAM_PHY_WRPHASE = 3
SDRAM_PHY_DELAYS = 32
SDRAM_PHY_BITSLIPS = 8
SDRAM_PHY_MODULES = 2

# Test address
DRAM_BASE = 0x40000000


class EtherboneClient:
    """Etherbone client with proper port binding."""

    def __init__(self, ip, port):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind(('', port))
        self.sock.settimeout(0.5)
        self.addr = (ip, port)

    def read(self, addr):
        pkt = bytes([
            0x4e, 0x6f, 0x10, 0x44,
            0x00, 0x00, 0x00, 0x00,
            0x00, 0x00, 0x00, 0x01,
            0x00, 0x00, 0x00, 0x00,
            (addr >> 24) & 0xff, (addr >> 16) & 0xff,
            (addr >> 8) & 0xff, addr & 0xff,
        ])
        self.sock.sendto(pkt, self.addr)
        try:
            resp, _ = self.sock.recvfrom(256)
            if len(resp) >= 20:
                return struct.unpack(">I", resp[16:20])[0]
        except socket.timeout:
            pass
        return None

    def write(self, addr, value):
        pkt = bytes([
            0x4e, 0x6f, 0x10, 0x44,
            0x00, 0x00, 0x00, 0x00,
            0x00, 0x00, 0x00, 0x00,
            0x00, 0x00, 0x00, 0x01,
            (addr >> 24) & 0xff, (addr >> 16) & 0xff,
            (addr >> 8) & 0xff, addr & 0xff,
            (value >> 24) & 0xff, (value >> 16) & 0xff,
            (value >> 8) & 0xff, value & 0xff,
        ])
        self.sock.sendto(pkt, self.addr)
        time.sleep(0.001)

    def close(self):
        self.sock.close()


def cdelay(n):
    """Approximate cdelay from C code - each unit ~10ns at 100MHz."""
    time.sleep(n * 0.00001)  # Scale for Etherbone latency


def command_p0(eb, cmd):
    """Issue command on phase 0."""
    eb.write(DFII_PI0_COMMAND, cmd)
    eb.write(DFII_PI0_COMMAND_ISSUE, 1)


def init_sequence(eb):
    """
    Exact port of init_sequence() from sdram_phy.h.
    Programs DDR3 mode registers for CL=7, CWL=5, BL=8.
    """
    print("  Running init_sequence (from sdram_phy.h)...")

    # Release reset
    eb.write(DFII_PI0_ADDRESS, 0x0)
    eb.write(DFII_PI0_BADDRESS, 0)
    eb.write(DFII_CONTROL, DFII_CONTROL_ODT | DFII_CONTROL_RESET_N)
    cdelay(50000)
    print("    Reset released")

    # Bring CKE high
    eb.write(DFII_PI0_ADDRESS, 0x0)
    eb.write(DFII_PI0_BADDRESS, 0)
    eb.write(DFII_CONTROL, DFII_CONTROL_CKE | DFII_CONTROL_ODT | DFII_CONTROL_RESET_N)
    cdelay(10000)
    print("    CKE high")

    # Load Mode Register 2, CWL=5
    eb.write(DFII_PI0_ADDRESS, 0x200)
    eb.write(DFII_PI0_BADDRESS, 2)
    command_p0(eb, DFII_COMMAND_RAS | DFII_COMMAND_CAS | DFII_COMMAND_WE | DFII_COMMAND_CS)
    cdelay(100)
    print("    MR2 loaded (CWL=5)")

    # Load Mode Register 3
    eb.write(DFII_PI0_ADDRESS, 0x0)
    eb.write(DFII_PI0_BADDRESS, 3)
    command_p0(eb, DFII_COMMAND_RAS | DFII_COMMAND_CAS | DFII_COMMAND_WE | DFII_COMMAND_CS)
    cdelay(100)
    print("    MR3 loaded")

    # Load Mode Register 1
    eb.write(DFII_PI0_ADDRESS, 0x6)
    eb.write(DFII_PI0_BADDRESS, 1)
    command_p0(eb, DFII_COMMAND_RAS | DFII_COMMAND_CAS | DFII_COMMAND_WE | DFII_COMMAND_CS)
    cdelay(100)
    print("    MR1 loaded")

    # Load Mode Register 0, CL=7, BL=8
    eb.write(DFII_PI0_ADDRESS, 0x930)
    eb.write(DFII_PI0_BADDRESS, 0)
    command_p0(eb, DFII_COMMAND_RAS | DFII_COMMAND_CAS | DFII_COMMAND_WE | DFII_COMMAND_CS)
    cdelay(200)
    print("    MR0 loaded (CL=7, BL=8)")

    # ZQ Calibration
    eb.write(DFII_PI0_ADDRESS, 0x400)
    eb.write(DFII_PI0_BADDRESS, 0)
    command_p0(eb, DFII_COMMAND_WE | DFII_COMMAND_CS)
    cdelay(200)
    print("    ZQ calibration complete")


def sdram_software_control(eb):
    """Switch to software control mode."""
    ctrl = eb.read(DFII_CONTROL)
    if ctrl is None:
        ctrl = 0
    eb.write(DFII_CONTROL, ctrl & ~DFII_CONTROL_SEL)


def sdram_hardware_control(eb):
    """Switch to hardware control mode."""
    ctrl = eb.read(DFII_CONTROL)
    if ctrl is None:
        ctrl = DFII_CONTROL_CKE | DFII_CONTROL_ODT | DFII_CONTROL_RESET_N
    eb.write(DFII_CONTROL, ctrl | DFII_CONTROL_SEL)


def sdram_read_leveling_rst_delay(eb, module):
    """Reset read delay for a module."""
    eb.write(DDRPHY_DLY_SEL, 1 << module)
    eb.write(DDRPHY_RDLY_DQ_RST, 1)


def sdram_read_leveling_inc_delay(eb, module):
    """Increment read delay for a module."""
    eb.write(DDRPHY_DLY_SEL, 1 << module)
    eb.write(DDRPHY_RDLY_DQ_INC, 1)


def sdram_read_leveling_rst_bitslip(eb, module):
    """Reset bitslip for a module."""
    eb.write(DDRPHY_DLY_SEL, 1 << module)
    eb.write(DDRPHY_RDLY_DQ_BITSLIP_RST, 1)


def sdram_read_leveling_inc_bitslip(eb, module):
    """Increment bitslip for a module."""
    eb.write(DDRPHY_DLY_SEL, 1 << module)
    eb.write(DDRPHY_RDLY_DQ_BITSLIP, 1)


def sdram_activate_row(eb, row, bank):
    """Activate a row."""
    eb.write(DFII_PI0_ADDRESS, row)
    eb.write(DFII_PI0_BADDRESS, bank)
    command_p0(eb, DFII_COMMAND_RAS | DFII_COMMAND_CS)
    cdelay(15)


def sdram_precharge_all(eb):
    """Precharge all banks."""
    eb.write(DFII_PI0_ADDRESS, 0x400)
    eb.write(DFII_PI0_BADDRESS, 0)
    command_p0(eb, DFII_COMMAND_RAS | DFII_COMMAND_WE | DFII_COMMAND_CS)
    cdelay(15)


def sdram_write(eb, col, bank, pattern):
    """Write data pattern."""
    eb.write(DFII_PI0_WRDATA, pattern)
    eb.write(DFII_PI0_ADDRESS, col)
    eb.write(DFII_PI0_BADDRESS, bank)
    command_p0(eb, DFII_COMMAND_CAS | DFII_COMMAND_WE | DFII_COMMAND_CS | DFII_COMMAND_WRDATA)
    cdelay(15)


def sdram_read(eb, col, bank):
    """Read data from DRAM."""
    eb.write(DFII_PI0_ADDRESS, col)
    eb.write(DFII_PI0_BADDRESS, bank)
    command_p0(eb, DFII_COMMAND_CAS | DFII_COMMAND_CS | DFII_COMMAND_RDDATA)
    cdelay(30)

    data = []
    for rddata_addr in [DFII_PI0_RDDATA, DFII_PI1_RDDATA, DFII_PI2_RDDATA, DFII_PI3_RDDATA]:
        val = eb.read(rddata_addr)
        data.append(val if val is not None else 0)
    return data


def sdram_read_leveling_scan_module(eb, module, pattern=0xA5A5A5A5):
    """
    Scan delay and bitslip values for a module.
    Based on sdram_read_leveling_scan_module() from sdram.c.
    """
    print(f"    Module {module}...")

    best_score = -1
    best_delay = 0
    best_bitslip = 0

    for bitslip in range(SDRAM_PHY_BITSLIPS):
        sdram_read_leveling_rst_bitslip(eb, module)
        for _ in range(bitslip):
            sdram_read_leveling_inc_bitslip(eb, module)

        delays_working = []

        for delay in range(SDRAM_PHY_DELAYS):
            sdram_read_leveling_rst_delay(eb, module)
            for _ in range(delay):
                sdram_read_leveling_inc_delay(eb, module)

            # Write pattern
            sdram_precharge_all(eb)
            sdram_activate_row(eb, 0, 0)
            sdram_write(eb, 0, 0, pattern)
            sdram_precharge_all(eb)

            # Read back
            sdram_activate_row(eb, 0, 0)
            data = sdram_read(eb, 0, 0)
            sdram_precharge_all(eb)

            # Check match for this module
            if module == 0:
                match = all((d & 0xFFFF) == (pattern & 0xFFFF) for d in data)
            else:
                match = all((d >> 16) == (pattern >> 16) for d in data)

            if match:
                delays_working.append(delay)

        # Find best window
        if delays_working:
            score = len(delays_working)
            if score > best_score:
                best_score = score
                best_delay = delays_working[len(delays_working) // 2]  # Center of window
                best_bitslip = bitslip

            if len(delays_working) >= 3:
                print(f"      bitslip={bitslip}: working delays {delays_working[0]}-{delays_working[-1]} (width={score})")

    # Apply best settings
    sdram_read_leveling_rst_bitslip(eb, module)
    for _ in range(best_bitslip):
        sdram_read_leveling_inc_bitslip(eb, module)

    sdram_read_leveling_rst_delay(eb, module)
    for _ in range(best_delay):
        sdram_read_leveling_inc_delay(eb, module)

    print(f"      Best: bitslip={best_bitslip} delay={best_delay} score={best_score}")
    return best_delay, best_bitslip, best_score


def sdram_read_leveling(eb):
    """Perform read leveling for all modules."""
    print("\n  Read leveling...")

    results = []
    for module in range(SDRAM_PHY_MODULES):
        delay, bitslip, score = sdram_read_leveling_scan_module(eb, module)
        results.append((delay, bitslip, score))

    return results


def sdram_init(eb):
    """
    Full SDRAM initialization - port of sdram_init() from sdram.c.
    """
    print("\n=== SDRAM Initialization ===")

    # Set PHY phases
    print("  Setting PHY phases...")
    eb.write(DDRPHY_RDPHASE, SDRAM_PHY_RDPHASE)
    eb.write(DDRPHY_WRPHASE, SDRAM_PHY_WRPHASE)

    # Reset all delays
    print("  Resetting delays...")
    for module in range(SDRAM_PHY_MODULES):
        sdram_read_leveling_rst_delay(eb, module)
        sdram_read_leveling_rst_bitslip(eb, module)

    # Switch to software control
    print("  Switching to software control...")
    sdram_software_control(eb)
    time.sleep(0.01)

    # Run init sequence (mode register programming)
    init_sequence(eb)

    # Perform read leveling
    leveling = sdram_read_leveling(eb)

    # Switch back to hardware control
    print("\n  Switching to hardware control...")
    sdram_hardware_control(eb)
    time.sleep(0.01)

    return leveling


def memtest(eb, addr, size=256):
    """Quick memory test."""
    print("\n=== Memory Test ===")

    errors = 0
    patterns = [0xCAFEBABE, 0x12345678, 0xAAAA5555, 0x55AA55AA, 0xFFFFFFFF]

    for i, pattern in enumerate(patterns):
        test_addr = addr + i * 4
        eb.write(test_addr, pattern)
        time.sleep(0.01)
        readback = eb.read(test_addr)
        rb = readback if readback is not None else 0

        if rb == pattern:
            status = "OK"
        else:
            status = "FAIL"
            errors += 1

        print(f"  0x{test_addr:08x}: write 0x{pattern:08x} read 0x{rb:08x} [{status}]")

    print(f"\nResult: {len(patterns) - errors}/{len(patterns)} passed")
    return errors == 0


def main():
    print("=" * 60)
    print("z1208: Proper DDR3 Init (from generated sdram_phy.h)")
    print("=" * 60)

    eb = EtherboneClient(FPGA_IP, FPGA_PORT)

    # Check connection
    print(f"\nConnecting to {FPGA_IP}:{FPGA_PORT}...")
    test_val = eb.read(0x1800)
    if test_val is None:
        print("ERROR: Cannot connect to FPGA")
        return
    print(f"Connected! Identifier: 0x{test_val:08x}")

    # Initialize SDRAM
    leveling = sdram_init(eb)

    # Test memory
    success = memtest(eb, DRAM_BASE + 0x100000)

    # Save results
    output = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "experiment": "z1208 Proper DDR3 Init from sdram_phy.h",
        "leveling": [{"module": i, "delay": d, "bitslip": b, "score": s}
                     for i, (d, b, s) in enumerate(leveling)],
        "memtest_passed": success
    }

    with open("results/z1208_proper_init.json", "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to results/z1208_proper_init.json")

    eb.close()

    if success:
        print("\n*** DDR3 INITIALIZATION SUCCESSFUL! ***")
        print("Partial writes are now available!")
    else:
        print("\n*** DDR3 INITIALIZATION FAILED ***")


if __name__ == "__main__":
    main()
