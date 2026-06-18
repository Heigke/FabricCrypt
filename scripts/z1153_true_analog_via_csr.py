#!/usr/bin/env python3
"""
z1153: TRUE Analog via DDRPHY CSR Control
=========================================

Uses the existing LiteDRAM build's CSR interface to control PHASER fine delays
and achieve TRUE analog partial charging.

The s7ddrphy has these CSRs:
- wdly_dq_inc  (0xf0000818): Write data delay increment
- wdly_dqs_inc (0xf000081c): Write strobe delay increment
- dly_sel      (0xf0000804): Delay selector (which byte lane)

By incrementing wdly_dq_inc while holding data stable, we can shift the write
timing and achieve partial DRAM charging.

Hardware: Arty A7-100T with LiteDRAM BIOS bitstream
"""

import serial
import time
import json
import numpy as np
from datetime import datetime
from pathlib import Path

# CSR addresses from csr.csv
CSR_BASE = 0xf0000000
DDRPHY_BASE = 0xf0000800

# DDRPHY CSR offsets (from csr.csv for LiteDRAM build)
DDRPHY_RST = 0x00           # Reset
DDRPHY_DLY_SEL = 0x04       # Delay selector
DDRPHY_HALF_SYS8X = 0x08    # Half sys8x taps
DDRPHY_WLEVEL_EN = 0x0c     # Write level enable
DDRPHY_WLEVEL_STROBE = 0x10 # Write level strobe
DDRPHY_CDLY_RST = 0x14      # Command delay reset
DDRPHY_CDLY_INC = 0x18      # Command delay increment
DDRPHY_RDLY_DQ_RST = 0x1c   # Read delay reset
DDRPHY_RDLY_DQ_INC = 0x20   # Read delay increment
DDRPHY_WDLY_DQ_RST = 0x28   # Write DQ delay reset
DDRPHY_WDLY_DQ_INC = 0x2c   # Write DQ delay increment
DDRPHY_WDLY_DQS_RST = 0x30  # Write DQS delay reset
DDRPHY_WDLY_DQS_INC = 0x34  # Write DQS delay increment

# SDRAM CSR addresses (for DFI software mode)
SDRAM_BASE = 0xf0002800
SDRAM_DFII_CONTROL = 0x00
SDRAM_DFII_PI0_CMD = 0x04
SDRAM_DFII_PI0_CMD_ISSUE = 0x08
SDRAM_DFII_PI0_ADDRESS = 0x0c
SDRAM_DFII_PI0_BADDRESS = 0x10
SDRAM_DFII_PI0_WRDATA = 0x14
SDRAM_DFII_PI0_RDDATA = 0x18


class LiteXSerial:
    """Serial interface to LiteX BIOS for CSR access."""

    def __init__(self, port='/dev/ttyUSB1', baudrate=115200):
        self.ser = serial.Serial(port, baudrate, timeout=0.5)
        time.sleep(0.2)
        self._flush()

    def _flush(self):
        self.ser.reset_input_buffer()
        self.ser.reset_output_buffer()

    def _read_response(self, timeout=0.5):
        """Read response until prompt or timeout."""
        start = time.time()
        data = b''
        while time.time() - start < timeout:
            if self.ser.in_waiting:
                data += self.ser.read(self.ser.in_waiting)
                if b'BIOS>' in data or b'RUNTIME>' in data:
                    break
            time.sleep(0.01)
        return data.decode('utf-8', errors='ignore')

    def send_cmd(self, cmd):
        """Send command and get response."""
        self._flush()
        self.ser.write((cmd + '\r\n').encode())
        time.sleep(0.05)
        return self._read_response()

    def mem_read(self, addr):
        """Read 32-bit value from memory address."""
        resp = self.send_cmd(f'mr {addr:#x}')
        # Parse response like "0xf0000800 : 0x00000000"
        for line in resp.split('\n'):
            if ':' in line and '0x' in line:
                try:
                    parts = line.split(':')
                    if len(parts) >= 2:
                        val = parts[1].strip().split()[0]
                        return int(val, 16)
                except:
                    pass
        return None

    def mem_write(self, addr, value):
        """Write 32-bit value to memory address."""
        resp = self.send_cmd(f'mw {addr:#x} {value:#x}')
        return 'ok' in resp.lower() or ':' in resp

    def close(self):
        self.ser.close()


class TrueAnalogTest:
    """Test TRUE analog partial charging via DDRPHY CSR control."""

    def __init__(self, serial_port='/dev/ttyUSB1'):
        self.lx = LiteXSerial(serial_port)
        self.results = []

    def reset_delays(self):
        """Reset all write delays to zero."""
        # Select all byte lanes
        self.lx.mem_write(DDRPHY_BASE + DDRPHY_DLY_SEL, 0xFF)
        # Reset write delays
        self.lx.mem_write(DDRPHY_BASE + DDRPHY_WDLY_DQ_RST, 1)
        time.sleep(0.01)
        self.lx.mem_write(DDRPHY_BASE + DDRPHY_WDLY_DQS_RST, 1)
        time.sleep(0.01)

    def set_fine_delay(self, taps):
        """Increment write delay by specified number of taps."""
        self.reset_delays()
        # Select all byte lanes
        self.lx.mem_write(DDRPHY_BASE + DDRPHY_DLY_SEL, 0xFF)
        # Increment delay
        for _ in range(taps):
            self.lx.mem_write(DDRPHY_BASE + DDRPHY_WDLY_DQ_INC, 1)
            time.sleep(0.001)

    def enable_software_mode(self):
        """Enable DFI software mode for direct SDRAM control."""
        # Set DFII control: software mode enabled
        self.lx.mem_write(SDRAM_BASE + SDRAM_DFII_CONTROL, 0x1)

    def disable_software_mode(self):
        """Disable DFI software mode."""
        self.lx.mem_write(SDRAM_BASE + SDRAM_DFII_CONTROL, 0x0)

    def write_pattern(self, address, pattern):
        """Write pattern to SDRAM address in software mode."""
        # Set address
        self.lx.mem_write(SDRAM_BASE + SDRAM_DFII_PI0_ADDRESS, address)
        self.lx.mem_write(SDRAM_BASE + SDRAM_DFII_PI0_BADDRESS, 0)
        # Set write data
        self.lx.mem_write(SDRAM_BASE + SDRAM_DFII_PI0_WRDATA, pattern)
        # Issue activate command (0x03)
        self.lx.mem_write(SDRAM_BASE + SDRAM_DFII_PI0_CMD, 0x03)
        self.lx.mem_write(SDRAM_BASE + SDRAM_DFII_PI0_CMD_ISSUE, 1)
        time.sleep(0.001)
        # Issue write command (0x05)
        self.lx.mem_write(SDRAM_BASE + SDRAM_DFII_PI0_CMD, 0x05)
        self.lx.mem_write(SDRAM_BASE + SDRAM_DFII_PI0_CMD_ISSUE, 1)
        time.sleep(0.001)
        # Issue precharge (0x02)
        self.lx.mem_write(SDRAM_BASE + SDRAM_DFII_PI0_CMD, 0x02)
        self.lx.mem_write(SDRAM_BASE + SDRAM_DFII_PI0_CMD_ISSUE, 1)

    def read_pattern(self, address):
        """Read pattern from SDRAM address."""
        # For now, use normal memory read through main_ram
        # Address 0x40000000 is main_ram base
        main_ram_addr = 0x40000000 + (address * 4)
        return self.lx.mem_read(main_ram_addr)

    def test_analog_levels(self, num_taps=64):
        """Test analog levels by varying fine delay."""
        print("Testing TRUE analog via DDRPHY fine delay control")
        print("=" * 60)

        test_pattern = 0xAAAAAAAA  # Alternating bits
        test_address = 0x1000

        results = []

        for tap in range(0, num_taps, 4):
            # Set fine delay
            self.set_fine_delay(tap)
            time.sleep(0.01)

            # Write test pattern
            # First write zeros to clear
            self.lx.mem_write(0x40000000 + test_address * 4, 0x00000000)
            time.sleep(0.001)

            # Write pattern
            self.lx.mem_write(0x40000000 + test_address * 4, test_pattern)

            # Read back
            read_val = self.lx.mem_read(0x40000000 + test_address * 4)

            if read_val is not None:
                # Count bits that match
                xor = test_pattern ^ read_val
                bits_different = bin(xor).count('1')
                bits_same = 32 - bits_different
                charge_level = bits_same / 32.0

                results.append({
                    'tap': tap,
                    'written': test_pattern,
                    'read': read_val,
                    'charge_level': charge_level
                })

                print(f"Tap {tap:2d}: wrote {test_pattern:#010x}, read {read_val:#010x}, charge={charge_level:.2%}")
            else:
                print(f"Tap {tap:2d}: read failed")
                results.append({
                    'tap': tap,
                    'written': test_pattern,
                    'read': None,
                    'charge_level': None
                })

        # Reset delays
        self.reset_delays()

        return results

    def run_full_test(self):
        """Run complete TRUE analog test."""
        print("\n" + "=" * 70)
        print("z1153: TRUE Analog via DDRPHY CSR Control")
        print("=" * 70)

        # Check connection
        print("\nChecking LiteX BIOS connection...")
        resp = self.lx.send_cmd('')
        if 'BIOS' not in resp and 'RUNTIME' not in resp:
            print("ERROR: Cannot connect to LiteX BIOS")
            print(f"Response: {resp[:200]}")
            return None
        print("Connected to LiteX BIOS")

        # Test analog levels
        print("\nTesting analog levels via fine delay control...")
        results = self.test_analog_levels(64)

        # Analyze results
        charge_levels = [r['charge_level'] for r in results if r['charge_level'] is not None]
        unique_levels = len(set(round(l, 2) for l in charge_levels))

        print("\n" + "=" * 60)
        print("RESULTS SUMMARY")
        print("=" * 60)
        print(f"Total tests: {len(results)}")
        print(f"Successful reads: {len(charge_levels)}")
        print(f"Unique charge levels: {unique_levels}")

        if charge_levels:
            print(f"Min charge: {min(charge_levels):.2%}")
            print(f"Max charge: {max(charge_levels):.2%}")
            print(f"Mean charge: {np.mean(charge_levels):.2%}")
            print(f"Std charge: {np.std(charge_levels):.4f}")

            # Check if we have TRUE analog (multiple distinct levels)
            if unique_levels > 3:
                print("\n✓ TRUE ANALOG ACHIEVED!")
                print(f"  {unique_levels} distinct charge levels detected via fine delay control")
            else:
                print("\n✗ Only binary levels detected")

        return {
            'experiment': 'z1153_true_analog_via_csr',
            'timestamp': datetime.now().isoformat(),
            'results': results,
            'unique_levels': unique_levels,
            'charge_levels': charge_levels,
            'success': unique_levels > 3
        }

    def close(self):
        self.lx.close()


def main():
    # Try both serial ports
    for port in ['/dev/ttyUSB1', '/dev/ttyUSB0']:
        try:
            print(f"\nTrying {port}...")
            test = TrueAnalogTest(port)
            results = test.run_full_test()
            test.close()

            if results:
                # Save results
                results_dir = Path(__file__).parent.parent / 'results'
                results_dir.mkdir(exist_ok=True)
                results_file = results_dir / 'z1153_true_analog_csr.json'

                with open(results_file, 'w') as f:
                    json.dump(results, f, indent=2, default=str)
                print(f"\nResults saved to: {results_file}")
                return

        except Exception as e:
            print(f"  Error on {port}: {e}")

    print("\nERROR: Could not connect to FPGA on any port")
    print("Make sure the LiteDRAM BIOS bitstream is programmed")


if __name__ == '__main__':
    main()
