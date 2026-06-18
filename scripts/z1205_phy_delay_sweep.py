#!/usr/bin/env python3
"""
z1205: Option 3 - Manual PHY Delay Sweep

Sweeps read delay (IDELAY) values to find working DDR3 read timing.
The S7DDRPHY uses Xilinx IDELAY primitives for read timing calibration.

CSR Layout (from s7ddrphy.py):
  DDRPHY_BASE = 0x800
  0x800: _rst          (CSRStorage) - PHY reset
  0x804: _dly_sel      (CSRStorage) - Byte lane select (0-1 for 16-bit)
  0x808-0x80c: unknown
  0x810: _rdly_dq_rst  (CSR) - Reset read delay to 0
  0x814: _rdly_dq_inc  (CSR) - Increment read delay by 1 tap
  0x818: _rdly_dq_bitslip_rst (CSR) - Reset read bitslip
  0x81c: _rdly_dq_bitslip     (CSR) - Increment read bitslip
  0x820: _wdly_dq_bitslip_rst (CSR) - Reset write bitslip
  0x824: _wdly_dq_bitslip     (CSR) - Increment write bitslip
  0x828: unknown
  0x82c: _rdphase     (CSRStorage) - Read phase select (0-3)
  0x830: _wrphase     (CSRStorage) - Write phase select (0-3)
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
PHY_RDPHASE = DDRPHY_BASE + 0x2c
PHY_WRPHASE = DDRPHY_BASE + 0x30

# SDRAM CSRs
SDRAM_BASE = CSR_BASE + 0x3000
DFII_CONTROL = SDRAM_BASE + 0x00
DFII_PI0_COMMAND = SDRAM_BASE + 0x04
DFII_PI0_COMMAND_ISSUE = SDRAM_BASE + 0x08
DFII_PI0_ADDRESS = SDRAM_BASE + 0x0c
DFII_PI0_BADDRESS = SDRAM_BASE + 0x10
DFII_PI0_WRDATA = SDRAM_BASE + 0x14
DFII_PI0_RDDATA = SDRAM_BASE + 0x18

# DRAM address for testing
DRAM_BASE = 0x40000000
TEST_ADDR = DRAM_BASE + 0x100000  # 1MB offset


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
        # Etherbone read packet
        pkt = bytes([
            0x4e, 0x6f,  # Magic
            0x10, 0x44,  # Version, flags
            0x00, 0x00, 0x00, 0x00,  # Reserved
            0x00, 0x00, 0x00, 0x01,  # Read count = 1
            0x00, 0x00, 0x00, 0x00,  # Write count = 0
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
            0x4e, 0x6f,  # Magic
            0x10, 0x44,  # Version, flags
            0x00, 0x00, 0x00, 0x00,  # Reserved
            0x00, 0x00, 0x00, 0x00,  # Read count = 0
            0x00, 0x00, 0x00, 0x01,  # Write count = 1
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


def test_dram_access(eb, test_pattern=0xCAFEBABE):
    """Test DRAM read/write at TEST_ADDR."""
    eb.write(TEST_ADDR, test_pattern)
    time.sleep(0.01)
    readback = eb.read(TEST_ADDR)
    return readback == test_pattern, readback


def set_rdly(eb, byte_lane, delay_taps):
    """Set read delay for a byte lane."""
    # Select byte lane
    eb.write(PHY_DLY_SEL, 1 << byte_lane)
    time.sleep(0.001)

    # Reset delay
    eb.write(PHY_RDLY_DQ_RST, 1)
    time.sleep(0.001)

    # Increment to target delay
    for _ in range(delay_taps):
        eb.write(PHY_RDLY_DQ_INC, 1)
        time.sleep(0.0001)


def set_rdly_bitslip(eb, byte_lane, bitslip):
    """Set read bitslip for a byte lane."""
    eb.write(PHY_DLY_SEL, 1 << byte_lane)
    time.sleep(0.001)

    # Reset bitslip
    eb.write(PHY_RDLY_BITSLIP_RST, 1)
    time.sleep(0.001)

    # Increment to target
    for _ in range(bitslip):
        eb.write(PHY_RDLY_BITSLIP, 1)
        time.sleep(0.0001)


def sweep_delays(eb):
    """Sweep all delay combinations to find working settings."""
    results = []

    print("\n=== PHY Read Delay Sweep ===")
    print("Testing delay tap values 0-31 and bitslip values 0-7...")

    # IDELAY has 32 taps (~78ps each = 2.5ns total range)
    # Bitslip has 8 positions (bit alignment)

    for rdphase in range(4):
        print(f"\n--- Testing rdphase={rdphase} ---")
        eb.write(PHY_RDPHASE, rdphase)
        time.sleep(0.01)

        for bitslip in range(8):
            for delay in range(0, 32, 4):  # Step by 4 for speed
                # Set both byte lanes
                for lane in range(2):
                    set_rdly(eb, lane, delay)
                    set_rdly_bitslip(eb, lane, bitslip)

                time.sleep(0.01)

                # Test multiple patterns
                patterns = [0xCAFEBABE, 0x12345678, 0xAAAA5555, 0x00000000, 0xFFFFFFFF]
                matches = 0
                for pat in patterns:
                    success, readback = test_dram_access(eb, pat)
                    if success:
                        matches += 1

                if matches > 0:
                    result = {
                        "rdphase": rdphase,
                        "bitslip": bitslip,
                        "delay": delay,
                        "matches": matches,
                        "total": len(patterns)
                    }
                    results.append(result)
                    print(f"  rdphase={rdphase} bitslip={bitslip} delay={delay}: {matches}/{len(patterns)} patterns OK")

    return results


def fine_tune_best(eb, best_config):
    """Fine-tune around best found configuration."""
    print(f"\n=== Fine-tuning around best config ===")
    print(f"Best so far: rdphase={best_config['rdphase']} bitslip={best_config['bitslip']} delay={best_config['delay']}")

    eb.write(PHY_RDPHASE, best_config['rdphase'])

    results = []
    for bitslip in range(max(0, best_config['bitslip']-1), min(8, best_config['bitslip']+2)):
        for delay in range(max(0, best_config['delay']-4), min(32, best_config['delay']+5)):
            for lane in range(2):
                set_rdly(eb, lane, delay)
                set_rdly_bitslip(eb, lane, bitslip)

            time.sleep(0.01)

            # Test more patterns
            patterns = [0xCAFEBABE, 0x12345678, 0xAAAA5555, 0x55AAAA55,
                       0x00000000, 0xFFFFFFFF, 0x0F0F0F0F, 0xF0F0F0F0]
            matches = 0
            for pat in patterns:
                success, _ = test_dram_access(eb, pat)
                if success:
                    matches += 1

            if matches >= best_config.get('matches', 0):
                result = {
                    "rdphase": best_config['rdphase'],
                    "bitslip": bitslip,
                    "delay": delay,
                    "matches": matches,
                    "total": len(patterns)
                }
                results.append(result)
                if matches == len(patterns):
                    print(f"  PERFECT: bitslip={bitslip} delay={delay}: {matches}/{len(patterns)}")

    return results


def main():
    print("=" * 60)
    print("z1205: PHY Delay Sweep for DDR3 Read Calibration")
    print("=" * 60)

    eb = EtherboneClient(FPGA_IP, FPGA_PORT)

    # Check connection
    print(f"\nConnecting to {FPGA_IP}:{FPGA_PORT}...")
    test_val = eb.read(0x1800)  # Read identifier
    if test_val is None:
        print("ERROR: Cannot connect to FPGA")
        return
    print(f"Connected! Identifier memory: 0x{test_val:08x}")

    # Read current PHY settings
    print("\nCurrent PHY settings:")
    rdphase = eb.read(PHY_RDPHASE)
    wrphase = eb.read(PHY_WRPHASE)
    print(f"  rdphase: {rdphase}")
    print(f"  wrphase: {wrphase}")

    # Quick test first
    print("\nInitial DRAM test...")
    success, readback = test_dram_access(eb)
    rb_val = readback if readback is not None else 0
    print(f"  Write 0xCAFEBABE, Read 0x{rb_val:08x}: {'PASS' if success else 'FAIL'}")

    # Sweep delays
    results = sweep_delays(eb)

    if results:
        # Find best result
        best = max(results, key=lambda x: x['matches'])
        print(f"\n=== Best Configuration Found ===")
        print(f"  rdphase: {best['rdphase']}")
        print(f"  bitslip: {best['bitslip']}")
        print(f"  delay: {best['delay']}")
        print(f"  matches: {best['matches']}/{best['total']}")

        # Fine tune
        fine_results = fine_tune_best(eb, best)
        if fine_results:
            final_best = max(fine_results, key=lambda x: x['matches'])
            print(f"\n=== Final Best Configuration ===")
            print(f"  rdphase: {final_best['rdphase']}")
            print(f"  bitslip: {final_best['bitslip']}")
            print(f"  delay: {final_best['delay']}")
            print(f"  matches: {final_best['matches']}/{final_best['total']}")

            # Apply final settings
            eb.write(PHY_RDPHASE, final_best['rdphase'])
            for lane in range(2):
                set_rdly(eb, lane, final_best['delay'])
                set_rdly_bitslip(eb, lane, final_best['bitslip'])

            results = fine_results
    else:
        print("\nNo working configuration found!")
        final_best = None

    # Save results
    output = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "experiment": "z1205 PHY Delay Sweep",
        "all_results": results,
        "best_config": final_best if 'final_best' in dir() else None
    }

    with open("results/z1205_phy_sweep.json", "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to results/z1205_phy_sweep.json")


if __name__ == "__main__":
    main()
