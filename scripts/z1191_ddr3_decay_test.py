#!/usr/bin/env python3
"""
z1191: DDR3 Decay Test - Now that init works!

Tests:
1. Basic memory operations across multiple addresses
2. Data retention without refresh (decay test)
3. Partial write attempts with truncated tRAS
"""

import time
import sys
import json

sys.path.insert(0, '/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy/venv/lib/python3.12/site-packages')

from litex.tools.litex_client import RemoteClient

# CSR Addresses (from build_with_cpu csr.h)
CSR_BASE = 0xf0000000

# SDRAM DFII CSRs
SDRAM_BASE = CSR_BASE + 0x2800
DFII_CONTROL = SDRAM_BASE + 0x00

# DFII Control bits
DFII_CONTROL_SEL = 0x01
DFII_CONTROL_CKE = 0x02
DFII_CONTROL_ODT = 0x04
DFII_CONTROL_RESET_N = 0x08

# DFII Command bits
DFII_COMMAND_CS = 0x01
DFII_COMMAND_WE = 0x02
DFII_COMMAND_CAS = 0x04
DFII_COMMAND_RAS = 0x08
DFII_COMMAND_WRDATA = 0x10
DFII_COMMAND_RDDATA = 0x20

# Commands
CMD_MODE_REGISTER = DFII_COMMAND_RAS | DFII_COMMAND_CAS | DFII_COMMAND_WE | DFII_COMMAND_CS
CMD_PRECHARGE_ALL = DFII_COMMAND_RAS | DFII_COMMAND_WE | DFII_COMMAND_CS
CMD_ACTIVATE = DFII_COMMAND_RAS | DFII_COMMAND_CS
CMD_PRECHARGE = DFII_COMMAND_RAS | DFII_COMMAND_WE | DFII_COMMAND_CS
CMD_READ = DFII_COMMAND_CAS | DFII_COMMAND_CS | DFII_COMMAND_RDDATA
CMD_WRITE = DFII_COMMAND_CAS | DFII_COMMAND_WE | DFII_COMMAND_CS | DFII_COMMAND_WRDATA


def get_phase_base(phase):
    return SDRAM_BASE + 0x04 + (phase * 0x18)


class DDR3Tester:
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

    def command_p0(self, cmd):
        base = get_phase_base(0)
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
        self.write(base + 8, 0x400)  # address (A10=1 for all banks)
        self.write(base + 12, 0)     # bank
        self.command_p0(CMD_PRECHARGE_ALL)
        time.sleep(0.0001)

    def activate(self, bank, row):
        base = get_phase_base(0)
        self.write(base + 8, row)
        self.write(base + 12, bank)
        self.command_p0(CMD_ACTIVATE)
        time.sleep(0.0001)

    def precharge_bank(self, bank):
        """Precharge single bank (A10=0)"""
        base = get_phase_base(0)
        self.write(base + 8, 0)  # A10=0 for single bank
        self.write(base + 12, bank)
        self.command_p0(CMD_PRECHARGE)
        time.sleep(0.0001)

    def write_dram(self, bank, col, data):
        # Write data on all phases
        for phase in range(4):
            base = get_phase_base(phase)
            self.write(base + 16, data)

        # Issue write on phase 3
        base = get_phase_base(3)
        self.write(base + 8, col)
        self.write(base + 12, bank)
        self.write(base, CMD_WRITE)
        self.write(base + 4, 1)
        time.sleep(0.0001)

    def read_dram(self, bank, col):
        # Issue read on phase 2
        base = get_phase_base(2)
        self.write(base + 8, col)
        self.write(base + 12, bank)
        self.write(base, CMD_READ)
        self.write(base + 4, 1)
        time.sleep(0.0001)

        # Read from all phases
        rddata = []
        for phase in range(4):
            base = get_phase_base(phase)
            rddata.append(self.read(base + 20))
        return rddata

    def partial_write(self, bank, col, data, tras_us=0.035):
        """
        Partial write with truncated tRAS.
        Normal tRAS is ~35ns minimum. We try shorter.
        """
        # Write data on all phases
        for phase in range(4):
            base = get_phase_base(phase)
            self.write(base + 16, data)

        # Issue write on phase 3
        base = get_phase_base(3)
        self.write(base + 8, col)
        self.write(base + 12, bank)
        self.write(base, CMD_WRITE)
        self.write(base + 4, 1)

        # Wait truncated tRAS
        time.sleep(tras_us * 1e-6)

        # Immediately precharge (truncate restore)
        self.precharge_bank(bank)

    def test_multi_address(self):
        """Test writing to multiple addresses"""
        print("\n" + "="*60)
        print("TEST 1: Multiple Addresses")
        print("="*60)

        self.set_sw_mode()
        self.precharge_all()

        test_data = [
            (0, 0, 0xAAAA5555),
            (0, 8, 0x12345678),
            (0, 16, 0xFEDCBA98),
            (0, 24, 0xCAFEBABE),
        ]

        # Write all
        self.activate(0, 0)
        for bank, col, data in test_data:
            self.write_dram(bank, col, data)
        self.precharge_all()

        # Read back
        self.activate(0, 0)
        results = []
        for bank, col, expected in test_data:
            rddata = self.read_dram(bank, col)
            match = any(d == expected for d in rddata)
            actual = rddata[0]  # Use phase 0
            results.append({
                "col": col,
                "expected": hex(expected),
                "actual": hex(actual),
                "match": match
            })
            status = "PASS" if match else "FAIL"
            print(f"  Col {col:3d}: expected 0x{expected:08X}, got 0x{actual:08X} [{status}]")

        self.precharge_all()
        self.set_hw_mode()

        success = all(r["match"] for r in results)
        return success, results

    def test_decay(self, delay_ms=100):
        """Test data retention after delay without refresh"""
        print(f"\n" + "="*60)
        print(f"TEST 2: Decay Test ({delay_ms}ms delay)")
        print("="*60)

        self.set_sw_mode()
        self.precharge_all()

        # Write pattern
        test_data = 0xFFFFFFFF
        self.activate(0, 100)
        self.write_dram(0, 0, test_data)
        self.precharge_all()

        # Switch to HW mode (refresh enabled)
        self.set_hw_mode()

        # Wait
        print(f"  Waiting {delay_ms}ms with refresh enabled...")
        time.sleep(delay_ms / 1000.0)

        # Read back
        self.set_sw_mode()
        self.activate(0, 100)
        rddata = self.read_dram(0, 0)
        self.precharge_all()
        self.set_hw_mode()

        actual = rddata[0]
        match = actual == test_data

        print(f"  Expected: 0x{test_data:08X}")
        print(f"  Actual:   0x{actual:08X}")
        print(f"  Result:   {'PASS' if match else 'FAIL'}")

        # Count bit errors
        if not match:
            xor = test_data ^ actual
            bit_errors = bin(xor).count('1')
            print(f"  Bit errors: {bit_errors}")

        return match, {"expected": hex(test_data), "actual": hex(actual), "delay_ms": delay_ms}

    def test_decay_no_refresh(self, delay_ms=10):
        """Test decay with refresh disabled (SW mode)"""
        print(f"\n" + "="*60)
        print(f"TEST 3: Decay Without Refresh ({delay_ms}ms)")
        print("="*60)

        self.set_sw_mode()  # No refresh in SW mode
        self.precharge_all()

        # Write all 1s
        test_data = 0xFFFFFFFF
        self.activate(0, 200)
        self.write_dram(0, 0, test_data)
        self.precharge_all()

        # Stay in SW mode (no refresh) and wait
        print(f"  Waiting {delay_ms}ms without refresh...")
        time.sleep(delay_ms / 1000.0)

        # Read back
        self.activate(0, 200)
        rddata = self.read_dram(0, 0)
        self.precharge_all()
        self.set_hw_mode()

        actual = rddata[0]

        print(f"  Expected: 0x{test_data:08X}")
        print(f"  Actual:   0x{actual:08X}")

        # Analyze decay
        if actual != test_data:
            xor = test_data ^ actual
            bit_errors = bin(xor).count('1')
            decay_pct = (bit_errors / 32) * 100
            print(f"  Bit errors: {bit_errors}/32 ({decay_pct:.1f}% decay)")
        else:
            print("  No decay detected")
            decay_pct = 0

        return {
            "delay_ms": delay_ms,
            "expected": hex(test_data),
            "actual": hex(actual),
            "bit_errors": bin(test_data ^ actual).count('1'),
            "decay_percent": decay_pct
        }

    def test_partial_write_levels(self):
        """Test partial writes with different tRAS values"""
        print("\n" + "="*60)
        print("TEST 4: Partial Write Levels")
        print("="*60)

        results = []
        tras_values = [0.035, 0.025, 0.020, 0.015, 0.010, 0.005]

        self.set_sw_mode()

        for tras in tras_values:
            self.precharge_all()

            # First write 0x00000000 (reference)
            self.activate(0, 300)
            self.write_dram(0, 0, 0x00000000)
            self.precharge_all()

            # Partial write 0xFFFFFFFF
            self.activate(0, 300)
            self.partial_write(0, 0, 0xFFFFFFFF, tras_us=tras)

            # Wait a bit
            time.sleep(0.001)

            # Read back
            self.activate(0, 300)
            rddata = self.read_dram(0, 0)
            self.precharge_all()

            actual = rddata[0]
            bits_set = bin(actual).count('1')

            result = {
                "tras_us": tras,
                "value": hex(actual),
                "bits_set": bits_set
            }
            results.append(result)
            print(f"  tRAS={tras:.3f}us: 0x{actual:08X} ({bits_set}/32 bits set)")

        self.set_hw_mode()
        return results


def main():
    print("z1191: DDR3 Decay and Partial Write Test")
    print("="*60)

    results = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "tests": {}
    }

    try:
        tester = DDR3Tester()

        # Test 1: Multiple addresses
        success, test1_results = tester.test_multi_address()
        results["tests"]["multi_address"] = {
            "success": success,
            "data": test1_results
        }

        # Test 2: Decay with refresh (should pass)
        success, test2_results = tester.test_decay(delay_ms=100)
        results["tests"]["decay_with_refresh"] = test2_results

        # Test 3: Decay without refresh at various delays
        decay_results = []
        for delay_ms in [1, 5, 10, 20, 50]:
            r = tester.test_decay_no_refresh(delay_ms=delay_ms)
            decay_results.append(r)
        results["tests"]["decay_no_refresh"] = decay_results

        # Test 4: Partial write levels
        test4_results = tester.test_partial_write_levels()
        results["tests"]["partial_write"] = test4_results

        tester.close()

        # Save results
        output_path = "/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy/results/z1191_ddr3_decay.json"
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
