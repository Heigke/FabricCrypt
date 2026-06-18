#!/usr/bin/env python3
"""
z1192: True Decay Test with L2 Bypass

Key improvements:
1. Write to multiple rows to defeat L2 caching
2. Longer delays (100ms+)
3. Verify SW mode is truly active
4. Multiple read attempts at different phases
"""

import time
import sys
import json

sys.path.insert(0, '/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy/venv/lib/python3.12/site-packages')

from litex.tools.litex_client import RemoteClient

CSR_BASE = 0xf0000000
SDRAM_BASE = CSR_BASE + 0x2800
DFII_CONTROL = SDRAM_BASE + 0x00

# L2 cache is 8KB = 2048 x 32-bit words
# To bypass L2, we need to write more than 8KB of data

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


class DecayTester:
    def __init__(self):
        self.wb = RemoteClient(host="localhost", port=1234)
        self.wb.open()

    def close(self):
        self.wb.close()

    def read(self, addr):
        return self.wb.read(addr)

    def write(self, addr, val):
        self.wb.write(addr, val)

    def command_p0(self, cmd):
        base = get_phase_base(0)
        self.write(base, cmd)
        self.write(base + 4, 1)

    def command_phase(self, phase, cmd):
        base = get_phase_base(phase)
        self.write(base, cmd)
        self.write(base + 4, 1)

    def set_sw_mode(self):
        """SW mode = no SEL bit, no refresh"""
        ctrl = DFII_CONTROL_CKE | DFII_CONTROL_ODT | DFII_CONTROL_RESET_N
        self.write(DFII_CONTROL, ctrl)

    def set_hw_mode(self):
        """HW mode = SEL bit set, controller handles refresh"""
        ctrl = DFII_CONTROL_SEL | DFII_CONTROL_CKE | DFII_CONTROL_ODT | DFII_CONTROL_RESET_N
        self.write(DFII_CONTROL, ctrl)

    def verify_mode(self):
        """Read back and verify control register"""
        ctrl = self.read(DFII_CONTROL)
        sel = (ctrl & DFII_CONTROL_SEL) != 0
        return "HW" if sel else "SW", ctrl

    def precharge_all(self):
        base = get_phase_base(0)
        self.write(base + 8, 0x400)
        self.write(base + 12, 0)
        self.command_p0(CMD_PRECHARGE_ALL)
        time.sleep(0.00002)  # tRP

    def activate(self, bank, row):
        base = get_phase_base(0)
        self.write(base + 8, row)
        self.write(base + 12, bank)
        self.command_p0(CMD_ACTIVATE)
        time.sleep(0.00002)  # tRCD

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

    def fill_row_with_pattern(self, bank, row, pattern, num_cols=64):
        """Fill a row with data pattern"""
        self.activate(bank, row)
        for col in range(0, num_cols * 8, 8):  # Step by 8 (BL8)
            self.write_dram(bank, col, pattern)
        self.precharge_all()

    def read_row(self, bank, row, num_cols=64):
        """Read a row and return all data"""
        self.activate(bank, row)
        data = []
        for col in range(0, num_cols * 8, 8):
            rddata = self.read_dram(bank, col)
            data.append(rddata[0])  # Use phase 0
        self.precharge_all()
        return data

    def test_decay_sweep(self, max_delay_ms=200, step_ms=25):
        """Test decay at various delay intervals"""
        print("\n" + "="*60)
        print("DECAY SWEEP TEST")
        print("="*60)

        results = []

        # Use different rows for each delay test to avoid L2 effects
        test_row = 1000

        for delay_ms in range(0, max_delay_ms + 1, step_ms):
            # Switch to SW mode (disable refresh)
            self.set_sw_mode()
            mode, ctrl = self.verify_mode()

            if delay_ms == 0:
                print(f"\nControl register: 0x{ctrl:02X} (mode: {mode})")

            self.precharge_all()

            # Write all 1s to test row
            test_pattern = 0xFFFFFFFF
            self.fill_row_with_pattern(0, test_row, test_pattern, num_cols=8)

            # Wait without refresh
            if delay_ms > 0:
                time.sleep(delay_ms / 1000.0)

            # Read back
            data = self.read_row(0, test_row, num_cols=8)

            # Switch back to HW mode
            self.set_hw_mode()

            # Analyze decay
            total_bits = len(data) * 32
            decayed_bits = 0
            for d in data:
                decayed_bits += bin(test_pattern ^ d).count('1')

            decay_pct = (decayed_bits / total_bits) * 100

            result = {
                "delay_ms": delay_ms,
                "total_bits": total_bits,
                "decayed_bits": decayed_bits,
                "decay_percent": decay_pct,
                "sample_values": [hex(d) for d in data[:4]]
            }
            results.append(result)

            print(f"  {delay_ms:3d}ms: {decayed_bits:3d}/{total_bits} bits decayed ({decay_pct:.1f}%)")

            # Use different row next time
            test_row += 10

        return results

    def test_multi_row_decay(self, delay_ms=100):
        """Write to many rows, wait, read back - bypasses L2"""
        print(f"\n" + "="*60)
        print(f"MULTI-ROW DECAY TEST ({delay_ms}ms)")
        print("="*60)

        num_rows = 32  # Enough to overflow 8KB L2
        test_pattern = 0xFFFFFFFF

        # Switch to SW mode
        self.set_sw_mode()
        mode, ctrl = self.verify_mode()
        print(f"Control: 0x{ctrl:02X}, Mode: {mode}")

        # Write to all rows
        print(f"Writing {num_rows} rows...")
        for row in range(num_rows):
            self.activate(0, row)
            for col in range(0, 64, 8):  # 8 words per row
                self.write_dram(0, col, test_pattern)
            self.precharge_all()

        # Wait without refresh
        print(f"Waiting {delay_ms}ms without refresh...")
        time.sleep(delay_ms / 1000.0)

        # Read back all rows
        print("Reading back...")
        total_errors = 0
        total_bits = 0
        decay_by_row = []

        for row in range(num_rows):
            self.activate(0, row)
            row_errors = 0
            for col in range(0, 64, 8):
                rddata = self.read_dram(0, col)
                actual = rddata[0]
                row_errors += bin(test_pattern ^ actual).count('1')
            self.precharge_all()

            total_errors += row_errors
            total_bits += 8 * 32
            decay_by_row.append(row_errors)

        self.set_hw_mode()

        decay_pct = (total_errors / total_bits) * 100
        print(f"\nTotal: {total_errors}/{total_bits} bits decayed ({decay_pct:.1f}%)")
        print(f"Per-row decay: min={min(decay_by_row)}, max={max(decay_by_row)}, avg={sum(decay_by_row)/len(decay_by_row):.1f}")

        return {
            "delay_ms": delay_ms,
            "num_rows": num_rows,
            "total_bits": total_bits,
            "total_errors": total_errors,
            "decay_percent": decay_pct,
            "decay_by_row": decay_by_row
        }

    def test_refresh_disable_verify(self):
        """Verify that SW mode truly disables refresh"""
        print("\n" + "="*60)
        print("REFRESH DISABLE VERIFICATION")
        print("="*60)

        # Test: write data, wait long time in SW mode, check if it survives
        # DDR3 cells need refresh every 64ms max per spec

        test_row = 5000
        test_pattern = 0xAAAA5555

        # Write in SW mode
        self.set_sw_mode()
        mode, ctrl = self.verify_mode()
        print(f"Mode: {mode}, Control: 0x{ctrl:02X}")

        self.precharge_all()
        self.activate(0, test_row)
        self.write_dram(0, 0, test_pattern)
        self.precharge_all()

        # Wait 150ms (more than 2x refresh period)
        delay_ms = 150
        print(f"Waiting {delay_ms}ms in SW mode...")
        time.sleep(delay_ms / 1000.0)

        # Read back still in SW mode
        self.activate(0, test_row)
        rddata = self.read_dram(0, 0)
        self.precharge_all()
        self.set_hw_mode()

        actual = rddata[0]
        match = actual == test_pattern
        errors = bin(test_pattern ^ actual).count('1')

        print(f"Expected: 0x{test_pattern:08X}")
        print(f"Actual:   0x{actual:08X}")
        print(f"Bit errors: {errors}/32")
        print(f"Result: {'PASS (no decay)' if match else 'DECAY DETECTED!'}")

        return {
            "delay_ms": delay_ms,
            "expected": hex(test_pattern),
            "actual": hex(actual),
            "errors": errors,
            "decayed": not match
        }


def main():
    print("z1192: True Decay Test with L2 Bypass")
    print("="*60)

    results = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "tests": {}
    }

    try:
        tester = DecayTester()

        # Test 1: Verify refresh is truly disabled
        verify_result = tester.test_refresh_disable_verify()
        results["tests"]["refresh_disable"] = verify_result

        # Test 2: Multi-row decay at 100ms
        multi_row = tester.test_multi_row_decay(delay_ms=100)
        results["tests"]["multi_row_100ms"] = multi_row

        # Test 3: Decay sweep
        sweep_results = tester.test_decay_sweep(max_delay_ms=200, step_ms=50)
        results["tests"]["decay_sweep"] = sweep_results

        tester.close()

        # Save results
        output_path = "/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy/results/z1192_true_decay.json"
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
