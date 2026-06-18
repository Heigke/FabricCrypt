#!/usr/bin/env python3
"""
z1197: Multi-Level Partial Write via Truncated tRAS

Now that we've confirmed:
1. DDR3 init works via Etherbone
2. DFII SW mode works correctly
3. No natural decay at room temp (60s+)

We can focus on partial writes with truncated tRAS.

Approach:
1. Write 0x00000000 (all 0s) - reference state
2. Activate row
3. Write 0xFFFFFFFF to DQ
4. Precharge IMMEDIATELY (truncate tRAS)
5. Read back and count bits that flipped to 1

Varying the delay between write and precharge should give different
charge levels on cells.
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
CMD_PRECHARGE = DFII_COMMAND_RAS | DFII_COMMAND_WE | DFII_COMMAND_CS
CMD_ACTIVATE = DFII_COMMAND_RAS | DFII_COMMAND_CS
CMD_READ = DFII_COMMAND_CAS | DFII_COMMAND_CS | DFII_COMMAND_RDDATA
CMD_WRITE = DFII_COMMAND_CAS | DFII_COMMAND_WE | DFII_COMMAND_CS | DFII_COMMAND_WRDATA


def get_phase_base(phase):
    return SDRAM_BASE + 0x04 + (phase * 0x18)


class PartialWriteTester:
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

    def precharge_bank(self, bank):
        base = get_phase_base(0)
        self.write(base + 8, 0)  # A10=0 for single bank
        self.write(base + 12, bank)
        self.command_phase(0, CMD_PRECHARGE)

    def activate(self, bank, row):
        base = get_phase_base(0)
        self.write(base + 8, row)
        self.write(base + 12, bank)
        self.command_phase(0, CMD_ACTIVATE)

    def write_dram_raw(self, bank, col, data):
        """Write without timing delays"""
        for phase in range(4):
            base = get_phase_base(phase)
            self.write(base + 16, data)
        base = get_phase_base(3)
        self.write(base + 8, col)
        self.write(base + 12, bank)
        self.command_phase(3, CMD_WRITE)

    def read_dram(self, bank, col):
        base = get_phase_base(2)
        self.write(base + 8, col)
        self.write(base + 12, bank)
        self.command_phase(2, CMD_READ)
        time.sleep(0.00001)
        return self.read(get_phase_base(0) + 20)

    def partial_write_sequence(self, row, col, delay_us):
        """
        1. Ensure cells are at 0 (reference)
        2. Activate row
        3. Issue write command with 0xFFFFFFFF
        4. Wait delay_us microseconds
        5. Precharge (interrupt the write/restore)
        6. Re-activate and read back
        """
        bank = 0

        # Step 1: Write reference (all 0s)
        self.precharge_all()
        time.sleep(0.00002)
        self.activate(bank, row)
        time.sleep(0.00002)
        self.write_dram_raw(bank, col, 0x00000000)
        time.sleep(0.00005)  # Full tRAS for reference write
        self.precharge_all()
        time.sleep(0.00002)

        # Step 2: Partial write (all 1s with truncated tRAS)
        self.activate(bank, row)
        time.sleep(0.00002)  # tRCD

        # Issue write command
        self.write_dram_raw(bank, col, 0xFFFFFFFF)

        # Wait the specified delay (this is our truncated tRAS)
        if delay_us > 0:
            time.sleep(delay_us * 1e-6)

        # Immediately precharge (interrupt restore)
        self.precharge_bank(bank)
        time.sleep(0.00002)

        # Step 3: Read back
        self.activate(bank, row)
        time.sleep(0.00002)
        result = self.read_dram(bank, col)
        self.precharge_all()

        return result

    def test_partial_write_sweep(self):
        """Sweep different tRAS truncation values"""
        print("\n=== PARTIAL WRITE tRAS SWEEP ===")

        # Test various delay values
        # DDR3 tRAS min is typically 35ns = 0.035us
        # We'll test from 0 to 100us
        delays_us = [0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0]

        results = []
        base_row = 40000

        for delay_us in delays_us:
            row = base_row + int(delay_us * 10)  # Different row per test

            val = self.partial_write_sequence(row, 0, delay_us)
            bits_set = bin(val).count('1')

            result = {
                "delay_us": delay_us,
                "value": hex(val),
                "bits_set": bits_set,
                "percent": (bits_set / 32) * 100
            }
            results.append(result)

            print(f"  tRAS={delay_us:6.2f}us: 0x{val:08X} ({bits_set:2d}/32 bits = {result['percent']:.0f}%)")

        return results

    def test_consistent_levels(self):
        """Test if partial writes produce consistent levels"""
        print("\n=== CONSISTENCY TEST ===")

        delays_to_test = [0.0, 0.1, 1.0, 10.0]
        base_row = 50000

        for delay_us in delays_to_test:
            print(f"\n  Testing delay={delay_us}us ({5} trials):")
            bits_list = []

            for trial in range(5):
                row = base_row + int(delay_us * 100) + trial
                val = self.partial_write_sequence(row, 0, delay_us)
                bits = bin(val).count('1')
                bits_list.append(bits)
                print(f"    Trial {trial+1}: {bits}/32 bits")

            avg = sum(bits_list) / len(bits_list)
            print(f"    Average: {avg:.1f}/32 bits")


def main():
    print("z1197: Multi-Level Partial Write Test")
    print("="*60)

    results = {}

    try:
        tester = PartialWriteTester()
        tester.set_sw_mode()

        # Run partial write sweep
        sweep_results = tester.test_partial_write_sweep()
        results["sweep"] = sweep_results

        # Test consistency
        tester.test_consistent_levels()

        tester.set_hw_mode()
        tester.close()

        # Save results
        output_path = "/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy/results/z1197_partial_write.json"
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
