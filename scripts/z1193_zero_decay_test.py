#!/usr/bin/env python3
"""
z1193: Zero-to-One Decay Test

DDR3 cells decay to their natural state which is typically "1" (charged).
Writing 0s and waiting should show bits flipping to 1s if decay occurs.
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
CMD_AUTO_REFRESH = DFII_COMMAND_RAS | DFII_COMMAND_CAS | DFII_COMMAND_CS


def get_phase_base(phase):
    return SDRAM_BASE + 0x04 + (phase * 0x18)


class DecayTester:
    def __init__(self):
        self.wb = RemoteClient(host="localhost", port=1234)
        self.wb.open()
        print("Connected")

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
        rddata = []
        for phase in range(4):
            base = get_phase_base(phase)
            rddata.append(self.read(base + 20))
        return rddata

    def issue_refresh(self):
        """Manually issue auto-refresh"""
        base = get_phase_base(0)
        self.write(base + 8, 0)
        self.write(base + 12, 0)
        self.command_phase(0, CMD_AUTO_REFRESH)
        time.sleep(0.0001)

    def test_zero_decay(self, delay_ms, num_locations=8):
        """Write 0s, wait, check if any bits flipped to 1s"""
        self.set_sw_mode()
        self.precharge_all()

        # Write all 0s to multiple locations
        test_row = 6000 + (delay_ms // 10)  # Different row per test
        self.activate(0, test_row)
        for col in range(0, num_locations * 8, 8):
            self.write_dram(0, col, 0x00000000)
        self.precharge_all()

        # Wait
        time.sleep(delay_ms / 1000.0)

        # Read back
        self.activate(0, test_row)
        flipped = 0
        total = 0
        for col in range(0, num_locations * 8, 8):
            rddata = self.read_dram(0, col)
            val = rddata[0]
            flipped += bin(val).count('1')  # 0->1 transitions
            total += 32
        self.precharge_all()

        self.set_hw_mode()

        return flipped, total

    def test_one_decay(self, delay_ms, num_locations=8):
        """Write 1s, wait, check if any bits flipped to 0s"""
        self.set_sw_mode()
        self.precharge_all()

        test_row = 7000 + (delay_ms // 10)
        self.activate(0, test_row)
        for col in range(0, num_locations * 8, 8):
            self.write_dram(0, col, 0xFFFFFFFF)
        self.precharge_all()

        time.sleep(delay_ms / 1000.0)

        self.activate(0, test_row)
        flipped = 0
        total = 0
        for col in range(0, num_locations * 8, 8):
            rddata = self.read_dram(0, col)
            val = rddata[0]
            flipped += bin(0xFFFFFFFF ^ val).count('1')  # 1->0 transitions
            total += 32
        self.precharge_all()

        self.set_hw_mode()

        return flipped, total

    def test_very_long_decay(self, delay_seconds=2):
        """Very long delay test"""
        print(f"\n=== VERY LONG DECAY TEST ({delay_seconds}s) ===")

        self.set_sw_mode()
        ctrl = self.read(DFII_CONTROL)
        print(f"Control: 0x{ctrl:02X} (SW mode)")

        self.precharge_all()

        # Write to 64 locations (2KB data)
        test_row = 8000
        num_locs = 64

        print(f"Writing {num_locs} x 32-bit = {num_locs*4} bytes of 0x00000000...")
        self.activate(0, test_row)
        for col in range(0, num_locs * 8, 8):
            self.write_dram(0, col, 0x00000000)
        self.precharge_all()

        # Long wait
        print(f"Waiting {delay_seconds} seconds in SW mode (no refresh)...")
        time.sleep(delay_seconds)

        # Read back
        print("Reading back...")
        self.activate(0, test_row)
        flipped_0to1 = 0
        total = num_locs * 32

        for col in range(0, num_locs * 8, 8):
            rddata = self.read_dram(0, col)
            val = rddata[0]
            flipped_0to1 += bin(val).count('1')
        self.precharge_all()

        # Also test 1s->0s
        test_row2 = 8100
        print(f"\nWriting {num_locs} x 32-bit of 0xFFFFFFFF...")
        self.activate(0, test_row2)
        for col in range(0, num_locs * 8, 8):
            self.write_dram(0, col, 0xFFFFFFFF)
        self.precharge_all()

        print(f"Waiting {delay_seconds} seconds...")
        time.sleep(delay_seconds)

        self.activate(0, test_row2)
        flipped_1to0 = 0
        for col in range(0, num_locs * 8, 8):
            rddata = self.read_dram(0, col)
            val = rddata[0]
            flipped_1to0 += bin(0xFFFFFFFF ^ val).count('1')
        self.precharge_all()

        self.set_hw_mode()

        pct_0to1 = (flipped_0to1 / total) * 100
        pct_1to0 = (flipped_1to0 / total) * 100

        print(f"\nResults after {delay_seconds}s:")
        print(f"  0->1 transitions: {flipped_0to1}/{total} ({pct_0to1:.2f}%)")
        print(f"  1->0 transitions: {flipped_1to0}/{total} ({pct_1to0:.2f}%)")

        return {
            "delay_seconds": delay_seconds,
            "total_bits": total,
            "flipped_0to1": flipped_0to1,
            "flipped_1to0": flipped_1to0,
            "pct_0to1": pct_0to1,
            "pct_1to0": pct_1to0
        }


def main():
    print("z1193: Zero-to-One Decay Test")
    print("="*60)

    results = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "tests": {}
    }

    try:
        tester = DecayTester()

        # Quick sweep
        print("\n=== DECAY SWEEP (0->1 and 1->0) ===")
        delays = [100, 200, 500, 1000]  # ms

        for delay_ms in delays:
            flipped_0to1, total = tester.test_zero_decay(delay_ms)
            flipped_1to0, _ = tester.test_one_decay(delay_ms)

            print(f"{delay_ms:4d}ms: 0->1: {flipped_0to1:3d}/{total} ({100*flipped_0to1/total:.1f}%), "
                  f"1->0: {flipped_1to0:3d}/{total} ({100*flipped_1to0/total:.1f}%)")

        # Very long test
        long_result = tester.test_very_long_decay(delay_seconds=5)
        results["tests"]["long_decay_5s"] = long_result

        tester.close()

        # Save results
        output_path = "/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy/results/z1193_zero_decay.json"
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {output_path}")

        return 0

    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
