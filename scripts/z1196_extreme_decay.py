#!/usr/bin/env python3
"""
z1196: Extreme Decay Test - Very Long Delays

DDR3 cells at room temperature may retain data for 10+ seconds.
This test uses extreme delays to try to observe decay.
"""

import time
import sys
import json

sys.path.insert(0, '/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy/venv/lib/python3.12/site-packages')

from litex.tools.litex_client import RemoteClient

CSR_BASE = 0xf0000000
SDRAM_BASE = CSR_BASE + 0x2800
DFII_CONTROL = SDRAM_BASE + 0x00

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

CMD_PRECHARGE_ALL = DFII_COMMAND_RAS | DFII_COMMAND_WE | DFII_COMMAND_CS
CMD_ACTIVATE = DFII_COMMAND_RAS | DFII_COMMAND_CS
CMD_READ = DFII_COMMAND_CAS | DFII_COMMAND_CS | DFII_COMMAND_RDDATA
CMD_WRITE = DFII_COMMAND_CAS | DFII_COMMAND_WE | DFII_COMMAND_CS | DFII_COMMAND_WRDATA


def get_phase_base(phase):
    return SDRAM_BASE + 0x04 + (phase * 0x18)


class ExtremeTester:
    def __init__(self):
        self.wb = RemoteClient(host="localhost", port=1234)
        self.wb.open()

    def close(self):
        self.wb.close()

    def read(self, addr):
        return self.wb.read(addr)

    def write(self, addr, val):
        self.wb.write(addr, val)

    def command_phase(self, phase, cmd):
        base = get_phase_base(phase)
        self.write(base, cmd)
        self.write(base + 4, 1)

    def set_sw_mode(self):
        ctrl = DFII_CONTROL_CKE | DFII_CONTROL_ODT | DFII_CONTROL_RESET_N
        self.write(DFII_CONTROL, ctrl)

    def set_hw_mode(self):
        ctrl = DFII_CONTROL_SEL | DFII_CONTROL_CKE | DFII_CONTROL_ODT | DFII_CONTROL_RESET_N
        self.write(DFII_CONTROL, ctrl)

    def precharge_all(self):
        base = get_phase_base(0)
        self.write(base + 8, 0x400)
        self.write(base + 12, 0)
        self.command_phase(0, CMD_PRECHARGE_ALL)
        time.sleep(0.00002)

    def activate(self, bank, row):
        base = get_phase_base(0)
        self.write(base + 8, row)
        self.write(base + 12, bank)
        self.command_phase(0, CMD_ACTIVATE)
        time.sleep(0.00002)

    def write_dram(self, bank, col, data):
        for phase in range(4):
            base = get_phase_base(phase)
            self.write(base + 16, data)
        base = get_phase_base(3)
        self.write(base + 8, col)
        self.write(base + 12, bank)
        self.command_phase(3, CMD_WRITE)
        time.sleep(0.00001)

    def read_dram(self, bank, col):
        base = get_phase_base(2)
        self.write(base + 8, col)
        self.write(base + 12, bank)
        self.command_phase(2, CMD_READ)
        time.sleep(0.00001)
        return self.read(get_phase_base(0) + 20)

    def test_extreme_decay(self, delay_seconds):
        """Test with very long delay"""
        test_row = 30000

        self.set_sw_mode()
        self.precharge_all()

        # Write all 0s (should decay to 1s)
        pattern = 0x00000000
        num_locs = 32

        self.activate(0, test_row)
        for col in range(0, num_locs * 8, 8):
            self.write_dram(0, col, pattern)

        # Keep row CLOSED (precharge) - cells are on their own
        self.precharge_all()

        # Wait
        print(f"    Waiting {delay_seconds}s in SW mode...", end="", flush=True)
        time.sleep(delay_seconds)
        print(" done")

        # Read back
        self.activate(0, test_row)
        errors_0to1 = 0
        for col in range(0, num_locs * 8, 8):
            val = self.read_dram(0, col)
            errors_0to1 += bin(val).count('1')  # Bits that flipped to 1
        self.precharge_all()

        total_bits = num_locs * 32

        # Also test 1s->0s
        test_row2 = test_row + 100
        pattern2 = 0xFFFFFFFF

        self.activate(0, test_row2)
        for col in range(0, num_locs * 8, 8):
            self.write_dram(0, col, pattern2)
        self.precharge_all()

        print(f"    Waiting {delay_seconds}s again...", end="", flush=True)
        time.sleep(delay_seconds)
        print(" done")

        self.activate(0, test_row2)
        errors_1to0 = 0
        for col in range(0, num_locs * 8, 8):
            val = self.read_dram(0, col)
            errors_1to0 += bin(0xFFFFFFFF ^ val).count('1')  # Bits that flipped to 0
        self.precharge_all()

        self.set_hw_mode()

        return {
            "delay_seconds": delay_seconds,
            "total_bits": total_bits,
            "errors_0to1": errors_0to1,
            "errors_1to0": errors_1to0,
            "pct_0to1": (errors_0to1 / total_bits) * 100,
            "pct_1to0": (errors_1to0 / total_bits) * 100
        }


def main():
    print("z1196: Extreme Decay Test")
    print("="*60)

    results = {"tests": []}

    try:
        tester = ExtremeTester()

        # Test with progressively longer delays
        delays = [10, 20, 30, 60]  # seconds

        for delay_s in delays:
            print(f"\n=== Testing {delay_s} second delay ===")
            r = tester.test_extreme_decay(delay_s)
            results["tests"].append(r)

            print(f"  0->1 transitions: {r['errors_0to1']}/{r['total_bits']} ({r['pct_0to1']:.1f}%)")
            print(f"  1->0 transitions: {r['errors_1to0']}/{r['total_bits']} ({r['pct_1to0']:.1f}%)")

            # If we detect decay, we're done
            if r['errors_0to1'] > 0 or r['errors_1to0'] > 0:
                print("\n*** DECAY DETECTED! ***")
                break

        tester.close()

        # Save
        output_path = "/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy/results/z1196_extreme_decay.json"
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {output_path}")

        # Summary
        if all(r['errors_0to1'] == 0 and r['errors_1to0'] == 0 for r in results["tests"]):
            print("\n*** NO DECAY DETECTED at any delay! ***")
            print("Possible causes:")
            print("1. PHY may be auto-refreshing at hardware level")
            print("2. DRAM cells very robust at room temperature")
            print("3. Self-refresh being triggered automatically")
            print("4. Some refresh mechanism we haven't disabled")

        return 0

    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
