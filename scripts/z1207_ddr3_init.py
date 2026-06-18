#!/usr/bin/env python3
"""
z1207: DDR3 Full Initialization via Etherbone

Manually performs the complete DDR3 initialization sequence that BIOS normally does:
1. DDR3 reset sequence
2. Mode register programming (MR0-MR3)
3. ZQ calibration
4. Read leveling / Write leveling
5. Memory test

This bypasses the need for a CPU/BIOS by doing all init via Etherbone CSR access.
"""

import socket
import struct
import time
import json
from datetime import datetime

# Etherbone connection
FPGA_IP = "192.168.0.50"
FPGA_PORT = 1234

# CSR addresses (from current build's csr.h)
CSR_BASE = 0x0
DDRPHY_BASE = CSR_BASE + 0x800
SDRAM_BASE = CSR_BASE + 0x3000

# DDRPHY CSRs
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
DFII_CONTROL = SDRAM_BASE + 0x00

# Phase injector offsets
DFII_PI_BASES = [
    SDRAM_BASE + 0x04,   # Phase 0
    SDRAM_BASE + 0x1c,   # Phase 1
    SDRAM_BASE + 0x34,   # Phase 2
    SDRAM_BASE + 0x4c,   # Phase 3
]
PI_COMMAND = 0x00
PI_COMMAND_ISSUE = 0x04
PI_ADDRESS = 0x08
PI_BADDRESS = 0x0c
PI_WRDATA = 0x10
PI_RDDATA = 0x14

# DFII Control bits
DFII_SEL = 0x01
DFII_CKE = 0x02
DFII_ODT = 0x04
DFII_RESET_N = 0x08

# DFII Command bits
DFII_CS = 0x01
DFII_WE = 0x02
DFII_CAS = 0x04
DFII_RAS = 0x08
DFII_WRDATA = 0x10
DFII_RDDATA = 0x20

# DDR3 commands
CMD_NOP = 0
CMD_ACT = DFII_CS | DFII_RAS
CMD_READ = DFII_CS | DFII_CAS | DFII_RDDATA
CMD_WRITE = DFII_CS | DFII_CAS | DFII_WE | DFII_WRDATA
CMD_PRE = DFII_CS | DFII_RAS | DFII_WE
CMD_REF = DFII_CS | DFII_RAS | DFII_CAS
CMD_MRS = DFII_CS | DFII_RAS | DFII_CAS | DFII_WE
CMD_ZQCL = DFII_CS | DFII_WE

# DDR3 timing (conservative, in ~10ns units at 100MHz)
tCK = 1        # Clock period
tRFC = 16      # Refresh to Active (160ns)
tXPR = 17      # Exit reset to any command (tRFC + 10ns)
tMOD = 2       # Mode register set to any command
tZQINIT = 52   # ZQ calibration long (512 CK)
tRCD = 2       # RAS to CAS delay
tRP = 2        # Precharge period
tWR = 2        # Write recovery
tWL = 1        # Write latency
tRL = 2        # Read latency (CL - AL)


class EtherboneClient:
    """Minimal Etherbone client for CSR access."""

    def __init__(self, ip, port):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
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


def dfii_control(eb, ctrl):
    """Set DFII control register."""
    eb.write(DFII_CONTROL, ctrl)


def dfii_command(eb, phase, cmd, addr=0, bank=0):
    """Issue a command on a phase."""
    base = DFII_PI_BASES[phase]
    eb.write(base + PI_ADDRESS, addr)
    eb.write(base + PI_BADDRESS, bank)
    eb.write(base + PI_COMMAND, cmd)
    eb.write(base + PI_COMMAND_ISSUE, 1)
    time.sleep(0.001)


def dfii_read(eb, phase):
    """Read data from phase."""
    base = DFII_PI_BASES[phase]
    return eb.read(base + PI_RDDATA)


def dfii_write_data(eb, phase, data):
    """Write data to phase."""
    base = DFII_PI_BASES[phase]
    eb.write(base + PI_WRDATA, data)


def wait_ck(n):
    """Wait n clock cycles (approx)."""
    time.sleep(n * 0.00001)  # 10ns per cycle at 100MHz


def ddr3_reset(eb):
    """DDR3 reset sequence per JEDEC."""
    print("  DDR3 reset sequence...")

    # Assert RESET_N low, CKE low
    dfii_control(eb, 0)
    wait_ck(20)

    # De-assert RESET_N (high), keep CKE low
    dfii_control(eb, DFII_RESET_N)
    wait_ck(50)  # Wait 500us minimum (we wait longer via sleep)

    # Assert CKE
    dfii_control(eb, DFII_RESET_N | DFII_CKE)
    wait_ck(tXPR)


def ddr3_mode_registers(eb):
    """Program DDR3 mode registers."""
    print("  Programming mode registers...")

    # MR2: CAS Write Latency = 5, Auto Self-Refresh
    # Bits: [10:9]=CWL, [7]=SRT, [6]=ASR
    mr2 = 0x000  # CWL=5 (000), no temp extensions
    dfii_command(eb, 0, CMD_MRS, mr2, 2)
    wait_ck(tMOD)

    # MR3: MPR disable
    mr3 = 0x000
    dfii_command(eb, 0, CMD_MRS, mr3, 3)
    wait_ck(tMOD)

    # MR1: DLL enable, output drive strength, Rtt_Nom, AL, write leveling
    # Bits: [0]=DLL_EN, [1]=ODS0, [5]=ODS1, [2]=Rtt0, [6]=Rtt1, [9]=Rtt2
    # ODS = 01 (RZQ/7), Rtt_Nom = 001 (RZQ/4 = 60 ohm)
    mr1 = (1 << 1) | (1 << 2)  # ODS=01, Rtt_Nom=01
    dfii_command(eb, 0, CMD_MRS, mr1, 1)
    wait_ck(tMOD)

    # MR0: Burst length, CAS latency, DLL reset, write recovery
    # Bits: [1:0]=BL, [3]=BT, [6:4][2]=CL, [8]=DLL_RST, [11:9]=WR
    # BL=8 (00), BT=seq (0), CL=6 (0010), DLL_RST=1, WR=6 (010)
    # CL bits: [6:4] = 010, [2] = 0 => CL=6
    # WR bits: [11:9] = 010 => WR=6
    mr0 = (0b010 << 9) | (1 << 8) | (0b010 << 4) | 0b00  # WR=6, DLL_RST, CL=6, BL=8
    dfii_command(eb, 0, CMD_MRS, mr0, 0)
    wait_ck(12)  # tMOD + DLL lock time


def ddr3_zq_calibration(eb):
    """ZQ calibration long."""
    print("  ZQ calibration...")
    dfii_command(eb, 0, CMD_ZQCL, 0x400, 0)  # A10=1 for ZQ long
    wait_ck(tZQINIT)


def ddr3_precharge_all(eb):
    """Precharge all banks."""
    dfii_command(eb, 0, CMD_PRE, 0x400, 0)  # A10=1 for all banks
    wait_ck(tRP)


def ddr3_refresh(eb):
    """Issue refresh command."""
    dfii_command(eb, 0, CMD_REF, 0, 0)
    wait_ck(tRFC)


def read_leveling(eb, modules=2):
    """Simple read leveling - sweep delays for each byte lane."""
    print("  Read leveling...")

    # Enter software mode
    dfii_control(eb, DFII_CKE | DFII_RESET_N)
    time.sleep(0.01)

    # Write test pattern
    ddr3_precharge_all(eb)
    time.sleep(0.001)

    # Activate row 0
    dfii_command(eb, 0, CMD_ACT, 0, 0)
    wait_ck(tRCD)

    # Write pattern
    for phase in range(4):
        dfii_write_data(eb, phase, 0xA5A5A5A5)
    dfii_command(eb, 0, CMD_WRITE, 0, 0)
    wait_ck(tWR + 2)
    ddr3_precharge_all(eb)

    best_settings = []

    for module in range(modules):
        print(f"    Module {module}...")

        eb.write(PHY_DLY_SEL, 1 << module)

        best_delay = 0
        best_bitslip = 0
        best_matches = 0

        for bitslip in range(8):
            # Set bitslip
            eb.write(PHY_RDLY_BITSLIP_RST, 1)
            for _ in range(bitslip):
                eb.write(PHY_RDLY_BITSLIP, 1)

            for delay in range(32):
                # Set delay
                eb.write(PHY_RDLY_DQ_RST, 1)
                for _ in range(delay):
                    eb.write(PHY_RDLY_DQ_INC, 1)

                # Activate and read
                dfii_command(eb, 0, CMD_ACT, 0, 0)
                wait_ck(tRCD)
                dfii_command(eb, 0, CMD_READ, 0, 0)
                wait_ck(tRL + 2)

                # Check data
                matches = 0
                for phase in range(4):
                    data = dfii_read(eb, phase)
                    if data is not None:
                        if module == 0:
                            if (data & 0xFFFF) == 0xA5A5:
                                matches += 1
                        else:
                            if (data >> 16) == 0xA5A5:
                                matches += 1

                ddr3_precharge_all(eb)

                if matches > best_matches:
                    best_matches = matches
                    best_delay = delay
                    best_bitslip = bitslip

        # Apply best settings
        eb.write(PHY_RDLY_BITSLIP_RST, 1)
        for _ in range(best_bitslip):
            eb.write(PHY_RDLY_BITSLIP, 1)
        eb.write(PHY_RDLY_DQ_RST, 1)
        for _ in range(best_delay):
            eb.write(PHY_RDLY_DQ_INC, 1)

        best_settings.append((best_delay, best_bitslip, best_matches))
        print(f"      Best: delay={best_delay} bitslip={best_bitslip} matches={best_matches}")

    return best_settings


def init_ddr3(eb):
    """Full DDR3 initialization sequence."""
    print("\n=== DDR3 Initialization ===")

    # Reset sequence
    ddr3_reset(eb)

    # Program mode registers
    ddr3_mode_registers(eb)

    # ZQ calibration
    ddr3_zq_calibration(eb)

    # Issue a few refreshes
    for _ in range(8):
        ddr3_refresh(eb)

    # Read leveling
    leveling = read_leveling(eb)

    # Switch back to hardware control
    dfii_control(eb, DFII_SEL | DFII_CKE | DFII_ODT | DFII_RESET_N)
    time.sleep(0.01)

    print("\n  Initialization complete!")
    return leveling


def test_dram(eb, num_tests=10):
    """Test DRAM read/write."""
    print("\n=== DRAM Test ===")

    DRAM_BASE = 0x40000000
    patterns = [0xCAFEBABE, 0x12345678, 0xAAAA5555, 0x55AA55AA,
                0x00000000, 0xFFFFFFFF, 0x0F0F0F0F, 0xF0F0F0F0,
                0xDEADBEEF, 0xBAADF00D]

    successes = 0
    for i, pattern in enumerate(patterns[:num_tests]):
        addr = DRAM_BASE + 0x100000 + (i * 0x1000)
        eb.write(addr, pattern)
        time.sleep(0.01)
        readback = eb.read(addr)
        rb_val = readback if readback is not None else 0

        if rb_val == pattern:
            successes += 1
            status = "PASS"
        else:
            status = "FAIL"

        print(f"  [{status}] Addr=0x{addr:08x} Write=0x{pattern:08x} Read=0x{rb_val:08x}")

    print(f"\nResult: {successes}/{num_tests} tests passed")
    return successes == num_tests


def main():
    print("=" * 60)
    print("z1207: DDR3 Full Initialization via Etherbone")
    print("=" * 60)

    eb = EtherboneClient(FPGA_IP, FPGA_PORT)

    # Check connection
    print(f"\nConnecting to {FPGA_IP}:{FPGA_PORT}...")
    test_val = eb.read(0x1800)
    if test_val is None:
        print("ERROR: Cannot connect to FPGA")
        return
    print(f"Connected! Identifier: 0x{test_val:08x}")

    # Initialize DDR3
    leveling = init_ddr3(eb)

    # Test DRAM
    success = test_dram(eb)

    # Save results
    output = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "experiment": "z1207 DDR3 Full Init via Etherbone",
        "leveling": [{"delay": d, "bitslip": b, "matches": m} for d, b, m in leveling],
        "test_passed": success
    }

    with open("results/z1207_ddr3_init.json", "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to results/z1207_ddr3_init.json")

    if success:
        print("\n*** DDR3 INITIALIZATION SUCCESSFUL! ***")
        print("You can now run partial write experiments.")
    else:
        print("\n*** DDR3 INITIALIZATION FAILED ***")


if __name__ == "__main__":
    main()
