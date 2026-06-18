#!/usr/bin/env python3
"""
z1206: Option 1 - DDR3 Read Leveling via Etherbone

Implements LiteDRAM's read leveling calibration algorithm via Etherbone.
Port of sdram_leveling.c read leveling to Python.

Read leveling finds the correct IDELAY and bitslip settings by:
1. Issue READ commands via DFII software mode
2. Sweep IDELAY taps looking for consistent "01" pattern on DQS
3. Find center of valid window
4. Set bitslip to align data correctly
"""

import socket
import struct
import time
import json
from datetime import datetime

# Etherbone connection
FPGA_IP = "192.168.0.50"
FPGA_PORT = 1234

# CSR addresses
CSR_BASE = 0x0
DDRPHY_BASE = CSR_BASE + 0x800
PHY_RST = DDRPHY_BASE + 0x00
PHY_DLY_SEL = DDRPHY_BASE + 0x04
PHY_RDLY_DQ_RST = DDRPHY_BASE + 0x10
PHY_RDLY_DQ_INC = DDRPHY_BASE + 0x14
PHY_RDLY_BITSLIP_RST = DDRPHY_BASE + 0x18
PHY_RDLY_BITSLIP = DDRPHY_BASE + 0x1c
PHY_WDLY_BITSLIP_RST = DDRPHY_BASE + 0x20
PHY_WDLY_BITSLIP = DDRPHY_BASE + 0x24
PHY_RDPHASE = DDRPHY_BASE + 0x2c
PHY_WRPHASE = DDRPHY_BASE + 0x30

# SDRAM DFII CSRs
SDRAM_BASE = CSR_BASE + 0x3000
DFII_CONTROL = SDRAM_BASE + 0x00

# Phase injector bases (4 phases)
DFII_PI_BASES = [
    SDRAM_BASE + 0x04,   # Phase 0
    SDRAM_BASE + 0x1c,   # Phase 1
    SDRAM_BASE + 0x34,   # Phase 2
    SDRAM_BASE + 0x4c,   # Phase 3
]

# Offsets within each phase injector
PI_COMMAND = 0x00
PI_COMMAND_ISSUE = 0x04
PI_ADDRESS = 0x08
PI_BADDRESS = 0x0c
PI_WRDATA = 0x10
PI_RDDATA = 0x14

# DDR3 Commands (active-low encoding)
DFII_CONTROL_SEL = 0x01
DFII_CONTROL_CKE = 0x02
DFII_CONTROL_ODT = 0x04
DFII_CONTROL_RESET_N = 0x08

DFII_COMMAND_CS = 0x01
DFII_COMMAND_WE = 0x02
DFII_COMMAND_CAS = 0x04
DFII_COMMAND_RAS = 0x08
DFII_COMMAND_WRDATA = 0x10
DFII_COMMAND_RDDATA = 0x20

# Common commands
CMD_NOP = 0x00
CMD_ACTIVATE = DFII_COMMAND_CS | DFII_COMMAND_RAS
CMD_READ = DFII_COMMAND_CS | DFII_COMMAND_CAS | DFII_COMMAND_RDDATA
CMD_WRITE = DFII_COMMAND_CS | DFII_COMMAND_CAS | DFII_COMMAND_WE | DFII_COMMAND_WRDATA
CMD_PRECHARGE = DFII_COMMAND_CS | DFII_COMMAND_RAS | DFII_COMMAND_WE

# DDR3 configuration
SDRAM_PHY_DELAYS = 32  # IDELAY taps
SDRAM_PHY_BITSLIPS = 8  # Bitslip positions


class EtherboneClient:
    """Minimal Etherbone client for CSR access."""

    def __init__(self, ip, port):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        # LiteX Etherbone reflects dest port as source - must bind to same port
        self.sock.bind(('', port))
        self.sock.settimeout(0.5)
        self.addr = (ip, port)

    def read(self, addr):
        """Read 32-bit value from address."""
        pkt = bytes([
            0x4e, 0x6f, 0x10, 0x44,
            0x00, 0x00, 0x00, 0x00,
            0x00, 0x00, 0x00, 0x01,
            0x00, 0x00, 0x00, 0x00,
            (addr >> 24) & 0xff,
            (addr >> 16) & 0xff,
            (addr >> 8) & 0xff,
            addr & 0xff,
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
        """Write 32-bit value to address."""
        pkt = bytes([
            0x4e, 0x6f, 0x10, 0x44,
            0x00, 0x00, 0x00, 0x00,
            0x00, 0x00, 0x00, 0x00,
            0x00, 0x00, 0x00, 0x01,
            (addr >> 24) & 0xff,
            (addr >> 16) & 0xff,
            (addr >> 8) & 0xff,
            addr & 0xff,
            (value >> 24) & 0xff,
            (value >> 16) & 0xff,
            (value >> 8) & 0xff,
            value & 0xff,
        ])
        self.sock.sendto(pkt, self.addr)
        time.sleep(0.001)


def dfii_set_control(eb, value):
    """Set DFII control register."""
    eb.write(DFII_CONTROL, value)


def dfii_command(eb, phase, command, address=0, bank=0):
    """Issue a DFII command on specified phase."""
    base = DFII_PI_BASES[phase]
    eb.write(base + PI_ADDRESS, address)
    eb.write(base + PI_BADDRESS, bank)
    eb.write(base + PI_COMMAND, command)
    eb.write(base + PI_COMMAND_ISSUE, 1)
    time.sleep(0.001)


def dfii_read(eb, phase):
    """Read data from DFII phase."""
    base = DFII_PI_BASES[phase]
    return eb.read(base + PI_RDDATA)


def dfii_write(eb, phase, data):
    """Write data to DFII phase."""
    base = DFII_PI_BASES[phase]
    eb.write(base + PI_WRDATA, data)


def set_rdly(eb, module, delay):
    """Set read delay for a module (byte lane)."""
    eb.write(PHY_DLY_SEL, 1 << module)
    eb.write(PHY_RDLY_DQ_RST, 1)
    time.sleep(0.001)
    for _ in range(delay):
        eb.write(PHY_RDLY_DQ_INC, 1)
        time.sleep(0.0001)


def set_bitslip(eb, module, bitslip):
    """Set read bitslip for a module."""
    eb.write(PHY_DLY_SEL, 1 << module)
    eb.write(PHY_RDLY_BITSLIP_RST, 1)
    time.sleep(0.001)
    for _ in range(bitslip):
        eb.write(PHY_RDLY_BITSLIP, 1)
        time.sleep(0.0001)


def enter_software_mode(eb):
    """Enter DFII software control mode."""
    # Set control: SEL=1 (software), CKE=1, ODT=1, RESET_N=1
    control = DFII_CONTROL_CKE | DFII_CONTROL_ODT | DFII_CONTROL_RESET_N
    dfii_set_control(eb, control)
    time.sleep(0.01)


def exit_software_mode(eb):
    """Exit DFII software control mode (back to controller)."""
    # Set SEL=1 for hardware control
    control = DFII_CONTROL_SEL | DFII_CONTROL_CKE | DFII_CONTROL_ODT | DFII_CONTROL_RESET_N
    dfii_set_control(eb, control)
    time.sleep(0.01)


def sdram_activate_row(eb, row, bank):
    """Activate a DRAM row."""
    dfii_command(eb, 0, CMD_ACTIVATE, row, bank)
    time.sleep(0.00001)  # tRCD


def sdram_precharge_all(eb):
    """Precharge all banks."""
    dfii_command(eb, 0, CMD_PRECHARGE, 0x400, 0)  # A10=1 for all banks
    time.sleep(0.00001)  # tRP


def sdram_write_pattern(eb, col, bank, pattern):
    """Write a pattern to DRAM (column)."""
    for phase in range(4):
        dfii_write(eb, phase, pattern)
    dfii_command(eb, 0, CMD_WRITE, col, bank)
    time.sleep(0.00001)  # tWR


def sdram_read_data(eb, col, bank):
    """Read data from DRAM (column)."""
    dfii_command(eb, 0, CMD_READ, col, bank)
    time.sleep(0.00002)  # Read latency
    data = []
    for phase in range(4):
        d = dfii_read(eb, phase)
        data.append(d if d is not None else 0)
    return data


def read_leveling_module(eb, module, row=0, bank=0, col=0):
    """
    Perform read leveling for a single module (byte lane).

    Returns (best_delay, best_bitslip, window_size)
    """
    print(f"\n  Read leveling module {module}...")

    # Write known pattern to DRAM
    enter_software_mode(eb)
    sdram_precharge_all(eb)
    time.sleep(0.001)
    sdram_activate_row(eb, row, bank)
    time.sleep(0.001)

    # Write pattern
    pattern = 0xA5A5A5A5  # Alternating bits for clock recovery
    sdram_write_pattern(eb, col, bank, pattern)
    time.sleep(0.001)
    sdram_precharge_all(eb)
    time.sleep(0.001)

    best_delay = 0
    best_bitslip = 0
    best_score = 0

    for bitslip in range(SDRAM_PHY_BITSLIPS):
        set_bitslip(eb, module, bitslip)

        working_delays = []

        for delay in range(SDRAM_PHY_DELAYS):
            set_rdly(eb, module, delay)
            time.sleep(0.001)

            # Re-activate and read
            sdram_activate_row(eb, row, bank)
            time.sleep(0.001)
            data = sdram_read_data(eb, col, bank)
            sdram_precharge_all(eb)
            time.sleep(0.001)

            # Check if read matches pattern
            # For 16-bit DDR3, check relevant bits based on module
            if module == 0:
                mask = 0x0000FFFF
            else:
                mask = 0xFFFF0000

            matches = sum(1 for d in data if (d & mask) == (pattern & mask))

            if matches >= 2:  # At least 2 of 4 phases match
                working_delays.append(delay)

        # Find center of working window
        if working_delays:
            window_size = len(working_delays)
            center_idx = window_size // 2
            center_delay = working_delays[center_idx]

            if window_size > best_score:
                best_score = window_size
                best_delay = center_delay
                best_bitslip = bitslip

            print(f"    bitslip={bitslip}: window=[{working_delays[0]}-{working_delays[-1]}] size={window_size} center={center_delay}")

    exit_software_mode(eb)
    return best_delay, best_bitslip, best_score


def sdram_read_leveling(eb, modules=2):
    """Perform read leveling for all modules."""
    print("\n=== DDR3 Read Leveling ===")

    results = {}

    for module in range(modules):
        delay, bitslip, window = read_leveling_module(eb, module)

        results[f"module_{module}"] = {
            "delay": delay,
            "bitslip": bitslip,
            "window_size": window
        }

        # Apply best settings
        set_rdly(eb, module, delay)
        set_bitslip(eb, module, bitslip)

        print(f"  Module {module}: delay={delay} bitslip={bitslip} window={window}")

    return results


def verify_dram(eb, num_tests=10):
    """Verify DRAM access works after leveling."""
    print("\n=== Verifying DRAM Access ===")

    DRAM_BASE = 0x40000000
    successes = 0

    patterns = [0xCAFEBABE, 0x12345678, 0xAAAA5555, 0x55AA55AA,
                0x00000000, 0xFFFFFFFF, 0x0F0F0F0F, 0xF0F0F0F0,
                0xDEADBEEF, 0xBAADF00D]

    for i, pattern in enumerate(patterns[:num_tests]):
        addr = DRAM_BASE + 0x100000 + (i * 0x1000)
        eb.write(addr, pattern)
        time.sleep(0.01)
        readback = eb.read(addr)

        if readback == pattern:
            successes += 1
            status = "PASS"
        else:
            status = "FAIL"

        rb_val = readback if readback is not None else 0
        print(f"  [{status}] Addr=0x{addr:08x} Write=0x{pattern:08x} Read=0x{rb_val:08x}")

    print(f"\nResult: {successes}/{num_tests} tests passed")
    return successes == num_tests


def main():
    print("=" * 60)
    print("z1206: DDR3 Read Leveling via Etherbone")
    print("=" * 60)

    eb = EtherboneClient(FPGA_IP, FPGA_PORT)

    # Check connection
    print(f"\nConnecting to {FPGA_IP}:{FPGA_PORT}...")
    test_val = eb.read(0x1800)
    if test_val is None:
        print("ERROR: Cannot connect to FPGA")
        return
    print(f"Connected! Identifier: 0x{test_val:08x}")

    # Read current settings
    print("\nCurrent PHY settings:")
    rdphase = eb.read(PHY_RDPHASE)
    wrphase = eb.read(PHY_WRPHASE)
    print(f"  rdphase: {rdphase}")
    print(f"  wrphase: {wrphase}")

    # Perform read leveling
    leveling_results = sdram_read_leveling(eb, modules=2)

    # Verify
    success = verify_dram(eb)

    # Save results
    output = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "experiment": "z1206 Read Leveling via Etherbone",
        "leveling": leveling_results,
        "verification_passed": success
    }

    with open("results/z1206_read_leveling.json", "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to results/z1206_read_leveling.json")

    if success:
        print("\n*** READ LEVELING SUCCESSFUL! ***")
    else:
        print("\n*** READ LEVELING DID NOT FIX ISSUE ***")
        print("Next step: Try Option 2 (build with CPU + BIOS)")


if __name__ == "__main__":
    main()
