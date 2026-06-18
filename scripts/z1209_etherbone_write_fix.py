#!/usr/bin/env python3
"""
z1209: Fix Etherbone Write Packet Format

The previous scripts had wcount=0 for write operations, causing writes to be ignored.
This script uses the correct Etherbone record header format:
  - Byte 0: flags (0x00 for normal op)
  - Byte 1: byte_enable (0x0f for all 4 bytes)
  - Byte 2: wcount (number of writes - MUST BE 1 for single write!)
  - Byte 3: rcount (0 for write-only)

Reference: https://github.com/enjoy-digital/litex/blob/master/litex/tools/remote/etherbone.py
"""

import socket
import struct
import time
import json
from datetime import datetime

FPGA_IP = "192.168.0.50"
FPGA_PORT = 1234

# CSR addresses from csr.h (build_partial_write_fsm)
CSR_BASE = 0x0
LEDS_OUT = CSR_BASE + 0x2000  # LED output register (csrstorage_363)
DDRPHY_BASE = CSR_BASE + 0x800
SDRAM_BASE = CSR_BASE + 0x3000
IDENTIFIER = CSR_BASE + 0x1800


class EtherboneClient:
    """Fixed Etherbone client with correct write packet format."""

    def __init__(self, ip, port):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind(('', port))
        self.sock.settimeout(0.5)
        self.addr = (ip, port)

    def read(self, addr):
        """Read 32-bit value from address."""
        # Etherbone packet header (8 bytes)
        # Magic: 0x4e6f, Version: 0x10, Flags: 0x44 (no probe reply, 32-bit addr/port)
        pkt_header = bytes([0x4e, 0x6f, 0x10, 0x44, 0x00, 0x00, 0x00, 0x00])

        # Record header (4 bytes): flags=0, byte_en=0x0f, wcount=0, rcount=1
        rec_header = bytes([0x00, 0x0f, 0x00, 0x01])

        # Read base address (where reply goes - we ignore this, use 0)
        read_base = bytes([0x00, 0x00, 0x00, 0x00])

        # Read address
        read_addr = struct.pack(">I", addr)

        pkt = pkt_header + rec_header + read_base + read_addr

        self.sock.sendto(pkt, self.addr)
        try:
            resp, _ = self.sock.recvfrom(256)
            if len(resp) >= 20:
                # Response format: 8 byte header + 4 byte record header + 4 byte base + 4 byte data
                return struct.unpack(">I", resp[16:20])[0]
        except socket.timeout:
            pass
        return None

    def write(self, addr, value):
        """Write 32-bit value to address - FIXED FORMAT."""
        # Etherbone packet header (8 bytes)
        pkt_header = bytes([0x4e, 0x6f, 0x10, 0x44, 0x00, 0x00, 0x00, 0x00])

        # Record header (4 bytes): flags=0, byte_en=0x0f, wcount=1, rcount=0
        # THIS IS THE KEY FIX: wcount=1 instead of 0!
        rec_header = bytes([0x00, 0x0f, 0x01, 0x00])

        # Write base address
        write_base = struct.pack(">I", addr)

        # Write value
        write_data = struct.pack(">I", value)

        pkt = pkt_header + rec_header + write_base + write_data

        self.sock.sendto(pkt, self.addr)
        time.sleep(0.001)  # Small delay for write to complete

    def close(self):
        self.sock.close()


def test_write_readback():
    """Test that writes actually take effect."""
    print("=" * 60)
    print("z1209: Testing Fixed Etherbone Write Format")
    print("=" * 60)

    eb = EtherboneClient(FPGA_IP, FPGA_PORT)

    # Test connection
    print(f"\nConnecting to {FPGA_IP}:{FPGA_PORT}...")
    ident = eb.read(IDENTIFIER)
    if ident is None:
        print("ERROR: Cannot connect to FPGA")
        return None
    print(f"Connected! Identifier: 0x{ident:08x}")

    results = {"tests": [], "success": False}

    # Test 1: LED register
    print("\n--- Test 1: LED Register ---")
    led_before = eb.read(LEDS_OUT)
    led_b = led_before if led_before is not None else 0
    print(f"  LED before: 0x{led_b:02x}")

    # Write different patterns
    for pattern in [0x05, 0x0A, 0x0F, 0x00]:
        eb.write(LEDS_OUT, pattern)
        time.sleep(0.01)
        led_after = eb.read(LEDS_OUT)
        led_a = led_after if led_after is not None else 0
        match = led_a == pattern
        status = "PASS" if match else "FAIL"
        print(f"  Write 0x{pattern:02x} -> Read 0x{led_a:02x} [{status}]")
        results["tests"].append({
            "register": "LEDS_OUT",
            "write": pattern,
            "read": led_a,
            "match": match
        })

    # Test 2: DFII_CONTROL register
    print("\n--- Test 2: DFII_CONTROL Register ---")
    DFII_CONTROL = SDRAM_BASE + 0x00

    ctrl_before = eb.read(DFII_CONTROL)
    ctrl_b = ctrl_before if ctrl_before is not None else 0
    print(f"  DFII_CONTROL before: 0x{ctrl_b:02x}")

    # DFII_CONTROL bits: SEL=0x01, CKE=0x02, ODT=0x04, RESET_N=0x08
    for pattern in [0x0E, 0x0F, 0x08, 0x00]:
        eb.write(DFII_CONTROL, pattern)
        time.sleep(0.01)
        ctrl_after = eb.read(DFII_CONTROL)
        ctrl_a = ctrl_after if ctrl_after is not None else 0
        match = ctrl_a == pattern
        status = "PASS" if match else "FAIL"
        print(f"  Write 0x{pattern:02x} -> Read 0x{ctrl_a:02x} [{status}]")
        results["tests"].append({
            "register": "DFII_CONTROL",
            "write": pattern,
            "read": ctrl_a,
            "match": match
        })

    # Test 3: Phase 0 Address register (csrstorage_25 at 0x300c)
    print("\n--- Test 3: Phase 0 Address Register ---")
    PI0_ADDRESS = SDRAM_BASE + 0x0c  # csrstorage_25

    for pattern in [0x1234, 0xABCD, 0x0000]:
        eb.write(PI0_ADDRESS, pattern)
        time.sleep(0.01)
        addr_after = eb.read(PI0_ADDRESS)
        addr_a = addr_after if addr_after is not None else 0
        match = addr_a == pattern
        status = "PASS" if match else "FAIL"
        print(f"  Write 0x{pattern:04x} -> Read 0x{addr_a:04x} [{status}]")
        results["tests"].append({
            "register": "PI0_ADDRESS",
            "write": pattern,
            "read": addr_a,
            "match": match
        })

    # Summary
    passes = sum(1 for t in results["tests"] if t["match"])
    total = len(results["tests"])
    results["success"] = passes == total
    results["summary"] = f"{passes}/{total} tests passed"

    print(f"\n{'='*60}")
    print(f"Result: {results['summary']}")
    print(f"{'='*60}")

    if results["success"]:
        print("\n*** ETHERBONE WRITES NOW WORKING! ***")
        print("DDR3 initialization should now be possible.")
    else:
        print("\n*** WRITES STILL FAILING ***")
        print("Need to investigate further - check Wishbone routing in gateware.")

    eb.close()
    return results


def main():
    results = test_write_readback()

    if results:
        output = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "experiment": "z1209 Etherbone Write Fix",
            "results": results
        }

        with open("results/z1209_etherbone_write_fix.json", "w") as f:
            json.dump(output, f, indent=2)

        print(f"\nResults saved to results/z1209_etherbone_write_fix.json")


if __name__ == "__main__":
    main()
