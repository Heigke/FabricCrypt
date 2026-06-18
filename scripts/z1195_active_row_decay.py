#!/usr/bin/env python3
"""
z1195: Active Row Decay Test

Keep a row ACTIVE (open) during the wait period to prevent self-refresh.
When a row is activated, the sense amplifiers hold the data but no refresh
occurs to the row. This should show decay.

DDR3 cells need refresh every 64ms max. With row open and no refresh,
we should see decay within 100-200ms.
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


class ActiveRowTester:
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

    def test_active_row_decay(self, delay_ms, pattern=0xFFFFFFFF):
        """
        1. Activate row
        2. Write pattern
        3. Keep row ACTIVE and wait (no precharge)
        4. Read back

        With row active, sense amplifiers hold data but cells get no refresh.
        Decay should occur to cells in the open row.
        """
        test_row = 20000 + (delay_ms // 10)

        self.set_sw_mode()
        self.precharge_all()

        # Activate row and write
        self.activate(0, test_row)
        for col in range(0, 64, 8):
            self.write_dram(0, col, pattern)

        # DO NOT precharge - keep row active!
        # Wait with row open
        time.sleep(delay_ms / 1000.0)

        # Now read back (row still active)
        results = []
        for col in range(0, 64, 8):
            rddata = self.read_dram(0, col)
            results.append(rddata[0])

        # Now precharge
        self.precharge_all()
        self.set_hw_mode()

        # Analyze
        total_bits = len(results) * 32
        errors = 0
        for val in results:
            errors += bin(pattern ^ val).count('1')

        return {
            "delay_ms": delay_ms,
            "pattern": hex(pattern),
            "total_bits": total_bits,
            "errors": errors,
            "decay_percent": (errors / total_bits) * 100,
            "samples": [hex(v) for v in results[:4]]
        }

    def test_activated_different_row_decay(self, delay_ms):
        """
        Test if decay occurs on a DIFFERENT row than the one activated.

        1. Write to row A
        2. Precharge
        3. Activate row B (different row)
        4. Wait with row B active
        5. Precharge
        6. Read row A

        Row A should experience decay since it's not being refreshed
        while row B is active.
        """
        row_a = 25000
        row_b = 26000
        pattern = 0xAAAA5555

        self.set_sw_mode()
        self.precharge_all()

        # Write to row A
        self.activate(0, row_a)
        for col in range(0, 64, 8):
            self.write_dram(0, col, pattern)
        self.precharge_all()

        # Activate row B (different row, same bank)
        self.activate(0, row_b)

        # Wait with row B active
        time.sleep(delay_ms / 1000.0)

        # Precharge
        self.precharge_all()

        # Read row A
        self.activate(0, row_a)
        errors = 0
        total_bits = 0
        for col in range(0, 64, 8):
            rddata = self.read_dram(0, col)
            val = rddata[0]
            errors += bin(pattern ^ val).count('1')
            total_bits += 32
        self.precharge_all()

        self.set_hw_mode()

        return {
            "delay_ms": delay_ms,
            "errors": errors,
            "total_bits": total_bits,
            "decay_percent": (errors / total_bits) * 100
        }


def main():
    print("z1195: Active Row Decay Test")
    print("="*60)

    results = {"tests": {}}

    try:
        tester = ActiveRowTester()

        # Test 1: Same row active during wait
        print("\n=== SAME ROW ACTIVE DECAY ===")
        print("(Row stays open during wait - tests sense amp holding)")

        active_row_results = []
        for delay_ms in [0, 100, 200, 500, 1000, 2000]:
            r = tester.test_active_row_decay(delay_ms, pattern=0x00000000)
            active_row_results.append(r)
            print(f"  {delay_ms:4d}ms: {r['errors']:3d}/{r['total_bits']} errors ({r['decay_percent']:.1f}%)")

        results["tests"]["active_row"] = active_row_results

        # Test 2: Different row active (target row closed but not refreshed)
        print("\n=== DIFFERENT ROW ACTIVE (target closed) ===")
        print("(Another row is open, target row gets no refresh)")

        diff_row_results = []
        for delay_ms in [100, 200, 500, 1000, 2000, 5000]:
            r = tester.test_activated_different_row_decay(delay_ms)
            diff_row_results.append(r)
            print(f"  {delay_ms:4d}ms: {r['errors']:3d}/{r['total_bits']} errors ({r['decay_percent']:.1f}%)")

        results["tests"]["different_row"] = diff_row_results

        tester.close()

        # Save
        output_path = "/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy/results/z1195_active_row_decay.json"
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
