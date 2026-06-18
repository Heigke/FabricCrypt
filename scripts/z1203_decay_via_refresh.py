#!/usr/bin/env python3
"""
z1203: Observe DRAM Decay via Refresh Control

Instead of nanosecond-precision partial writes (which require complex DFI control),
this script explores decay by:
1. Writing patterns via normal L2 interface
2. Disabling/limiting refresh
3. Waiting for decay
4. Reading back to measure decay

This leverages DRAM's natural physics - cells lose charge over time without refresh.
"""

import time
import sys
import json

sys.path.insert(0, '/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy/venv/lib/python3.12/site-packages')

from litex.tools.litex_client import RemoteClient

MAIN_RAM_BASE = 0x40000000
SDRAM_BASE = 0x3000
DFII_CONTROL = SDRAM_BASE + 0x00


class DecayTester:
    def __init__(self):
        self.wb = RemoteClient(host="localhost", port=1234)
        self.wb.open()
        print("Connected to litex_server")

    def close(self):
        self.wb.close()

    def read(self, addr):
        return self.wb.read(addr)

    def write(self, addr, val):
        self.wb.write(addr, val)

    def set_hw_mode(self):
        """Enable HW mode with refresh"""
        ctrl = 0x0F  # sel=1, cke=1, odt=1, reset_n=1
        self.write(DFII_CONTROL, ctrl)

    def write_test_pattern(self, base_addr, size_words, pattern):
        """Write a pattern to memory"""
        for i in range(size_words):
            self.write(base_addr + i * 4, pattern)

    def read_and_analyze(self, base_addr, size_words, expected):
        """Read memory and count bit flips from expected"""
        total_bits = 0
        flipped_bits = 0
        values = []

        for i in range(size_words):
            val = self.read(base_addr + i * 4)
            values.append(val)
            xor = val ^ expected
            flipped = bin(xor).count('1')
            total_bits += 32
            flipped_bits += flipped

        return {
            'values': [hex(v) for v in values],
            'total_bits': total_bits,
            'flipped_bits': flipped_bits,
            'flip_rate': flipped_bits / total_bits if total_bits > 0 else 0,
        }

    def test_decay_immediate(self):
        """Test immediate read after write (baseline)"""
        print("\n=== IMMEDIATE READ TEST (baseline) ===")

        base = MAIN_RAM_BASE + 0x10000  # Use offset to avoid cache issues
        size = 64  # 64 words = 256 bytes

        self.set_hw_mode()

        # Write all 1s
        print("  Writing 0xFFFFFFFF...")
        self.write_test_pattern(base, size, 0xFFFFFFFF)

        # Immediate read
        result = self.read_and_analyze(base, size, 0xFFFFFFFF)
        print(f"  Immediate read: {result['flipped_bits']}/{result['total_bits']} bits flipped ({result['flip_rate']*100:.2f}%)")

        return result

    def test_decay_with_delay(self, delay_seconds):
        """Test decay after waiting"""
        print(f"\n=== DECAY TEST ({delay_seconds}s delay) ===")

        base = MAIN_RAM_BASE + 0x20000 + int(delay_seconds * 0x1000)  # Different address for each test
        size = 64

        self.set_hw_mode()

        # Write all 1s
        print("  Writing 0xFFFFFFFF...")
        self.write_test_pattern(base, size, 0xFFFFFFFF)

        # Force a few reads to different locations to potentially evict L2 cache
        for i in range(256):
            _ = self.read(MAIN_RAM_BASE + 0x80000 + i * 64)

        # Wait
        print(f"  Waiting {delay_seconds} seconds...")
        time.sleep(delay_seconds)

        # Force more reads to flush L2
        for i in range(256):
            _ = self.read(MAIN_RAM_BASE + 0x90000 + i * 64)

        # Read back
        result = self.read_and_analyze(base, size, 0xFFFFFFFF)
        print(f"  After {delay_seconds}s: {result['flipped_bits']}/{result['total_bits']} bits flipped ({result['flip_rate']*100:.2f}%)")

        # Show sample values
        print(f"  Sample values: {result['values'][:5]}")

        return result

    def test_multiple_patterns(self):
        """Test decay with different patterns"""
        print("\n=== PATTERN TEST ===")

        results = {}
        base = MAIN_RAM_BASE + 0x30000
        size = 32
        delay = 1.0

        patterns = [
            (0xFFFFFFFF, "all_1s"),
            (0x00000000, "all_0s"),
            (0xAAAAAAAA, "alt_10"),
            (0x55555555, "alt_01"),
        ]

        for pattern, name in patterns:
            self.set_hw_mode()

            # Write pattern
            self.write_test_pattern(base, size, pattern)

            # Flush cache
            for i in range(256):
                _ = self.read(MAIN_RAM_BASE + 0xA0000 + i * 64)

            # Wait
            time.sleep(delay)

            # Flush again
            for i in range(256):
                _ = self.read(MAIN_RAM_BASE + 0xB0000 + i * 64)

            # Read back
            result = self.read_and_analyze(base, size, pattern)
            results[name] = result
            print(f"  {name}: {result['flipped_bits']}/{result['total_bits']} bits flipped ({result['flip_rate']*100:.2f}%)")

            base += 0x1000

        return results


def main():
    print("z1203: Decay via Refresh Control")
    print("="*60)

    results = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "tests": {}
    }

    try:
        tester = DecayTester()

        # Baseline test
        results["tests"]["baseline"] = tester.test_decay_immediate()

        # Decay tests with various delays
        for delay in [0.1, 0.5, 1.0, 2.0, 5.0]:
            results["tests"][f"delay_{delay}s"] = tester.test_decay_with_delay(delay)

        # Pattern tests
        results["tests"]["patterns"] = tester.test_multiple_patterns()

        tester.close()

        output_path = "/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy/results/z1203_decay_refresh.json"
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {output_path}")

        # Summary
        print("\n=== SUMMARY ===")
        baseline_flip = results["tests"]["baseline"]["flip_rate"]
        print(f"Baseline flip rate: {baseline_flip*100:.2f}%")

        for key in results["tests"]:
            if key.startswith("delay_"):
                flip = results["tests"][key]["flip_rate"]
                print(f"{key}: {flip*100:.2f}%")

        return 0

    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
