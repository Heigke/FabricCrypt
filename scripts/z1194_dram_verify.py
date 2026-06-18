#!/usr/bin/env python3
"""
z1194: DRAM Access Verification

Verify that our DFII commands are truly accessing DRAM:
1. Write unique patterns to many addresses
2. Read back in random order
3. Check data integrity

Also test CKE toggling for forced decay.
"""

import time
import sys
import random

sys.path.insert(0, '/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy/venv/lib/python3.12/site-packages')

from litex.tools.litex_client import RemoteClient

CSR_BASE = 0xf0000000
SDRAM_BASE = CSR_BASE + 0x2800
DFII_CONTROL = SDRAM_BASE + 0x00

# Direct memory base (through L2)
MAIN_RAM_BASE = 0x40000000

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


class DRAMVerifier:
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
        return self.read(get_phase_base(0) + 20)

    def test_unique_pattern_access(self):
        """Write unique patterns to many addresses, read back in random order"""
        print("\n=== UNIQUE PATTERN ACCESS TEST ===")

        self.set_sw_mode()
        self.precharge_all()

        # Write unique patterns to 32 locations
        num_locs = 32
        test_row = 10000
        patterns = {}

        print(f"Writing {num_locs} unique patterns...")
        self.activate(0, test_row)
        for i in range(num_locs):
            col = i * 8
            pattern = 0xDEAD0000 | (i * 0x1111)
            self.write_dram(0, col, pattern)
            patterns[col] = pattern
        self.precharge_all()

        # Read back in random order
        print("Reading back in random order...")
        cols = list(patterns.keys())
        random.shuffle(cols)

        self.activate(0, test_row)
        errors = 0
        for col in cols:
            expected = patterns[col]
            actual = self.read_dram(0, col)
            if actual != expected:
                print(f"  ERROR at col {col}: expected 0x{expected:08X}, got 0x{actual:08X}")
                errors += 1
        self.precharge_all()

        self.set_hw_mode()

        print(f"\nResult: {num_locs - errors}/{num_locs} correct")
        return errors == 0

    def test_l2_bypass(self):
        """Test if L2 cache is involved by checking memory access patterns"""
        print("\n=== L2 BYPASS TEST ===")

        # Write via L2 (HW mode)
        self.set_hw_mode()
        test_addr = MAIN_RAM_BASE + 0x100000  # 1MB offset

        print("Writing 0xCAFEBABE via L2...")
        self.write(test_addr, 0xCAFEBABE)

        # Read back
        val_l2 = self.read(test_addr)
        print(f"Read via L2: 0x{val_l2:08X}")

        # Now read same location via DFII
        # Address decoding:
        # Row = bits 10-23, Col = bits 2-9, Bank = bits 24-26
        # For 0x100000: col=0, row=0x100 (256), bank=0
        col = 0
        row = 256
        bank = 0

        self.set_sw_mode()
        self.precharge_all()
        self.activate(bank, row)
        val_dfii = self.read_dram(bank, col)
        self.precharge_all()
        self.set_hw_mode()

        print(f"Read via DFII: 0x{val_dfii:08X}")

        if val_l2 == val_dfii == 0xCAFEBABE:
            print("L2 and DFII see same data!")
        else:
            print("L2 and DFII see different data - possible L2 caching")

        return val_l2, val_dfii

    def test_cke_toggle_decay(self, powerdown_ms=500):
        """Test decay by briefly disabling CKE (self-refresh/power-down)"""
        print(f"\n=== CKE TOGGLE DECAY TEST ({powerdown_ms}ms) ===")

        self.set_sw_mode()
        self.precharge_all()

        # Write test pattern
        test_row = 12000
        test_pattern = 0x12345678

        self.activate(0, test_row)
        self.write_dram(0, 0, test_pattern)
        self.precharge_all()

        # Verify write
        self.activate(0, test_row)
        val_before = self.read_dram(0, 0)
        self.precharge_all()
        print(f"Before CKE toggle: 0x{val_before:08X}")

        # Disable CKE (DRAM enters power-down or self-refresh)
        print(f"Disabling CKE for {powerdown_ms}ms...")
        ctrl_no_cke = DFII_CONTROL_ODT | DFII_CONTROL_RESET_N  # No CKE
        self.write(DFII_CONTROL, ctrl_no_cke)

        time.sleep(powerdown_ms / 1000.0)

        # Re-enable CKE
        print("Re-enabling CKE...")
        self.write(DFII_CONTROL, DFII_CONTROL_CKE | DFII_CONTROL_ODT | DFII_CONTROL_RESET_N)
        time.sleep(0.001)

        # Read back
        self.precharge_all()
        self.activate(0, test_row)
        val_after = self.read_dram(0, 0)
        self.precharge_all()

        self.set_hw_mode()

        print(f"After CKE toggle: 0x{val_after:08X}")

        if val_before == val_after:
            print("Data survived CKE toggle! (DRAM entered self-refresh)")
            # DDR3 may enter self-refresh automatically when CKE goes low
        else:
            diff_bits = bin(val_before ^ val_after).count('1')
            print(f"Data changed! {diff_bits} bits different")

        return val_before, val_after

    def test_controlled_decay(self):
        """Test decay by writing, precharging, and waiting without any access"""
        print("\n=== CONTROLLED DECAY TEST ===")

        self.set_sw_mode()
        self.precharge_all()

        # Write to multiple banks (8 banks in DDR3)
        test_pattern = 0xFFFF0000
        test_row = 15000

        print("Writing to all 8 banks...")
        for bank in range(8):
            self.activate(bank, test_row)
            self.write_dram(bank, 0, test_pattern)
            self.precharge_all()

        # Now wait with all banks precharged (closed)
        # In this state, DRAM cells are holding data without any access
        delay_s = 10
        print(f"All banks precharged. Waiting {delay_s}s without any DRAM access...")
        time.sleep(delay_s)

        # Read back
        print("Reading back...")
        results = []
        for bank in range(8):
            self.activate(bank, test_row)
            val = self.read_dram(bank, 0)
            self.precharge_all()

            errors = bin(test_pattern ^ val).count('1')
            results.append({"bank": bank, "expected": test_pattern, "actual": val, "errors": errors})

            if val != test_pattern:
                print(f"  Bank {bank}: 0x{val:08X} ({errors} bit errors)")
            else:
                print(f"  Bank {bank}: OK")

        self.set_hw_mode()

        total_errors = sum(r["errors"] for r in results)
        print(f"\nTotal bit errors: {total_errors}/256")

        return results


def main():
    print("z1194: DRAM Access Verification")
    print("="*60)

    try:
        verifier = DRAMVerifier()

        # Test 1: Verify DFII actually writes to DRAM
        unique_ok = verifier.test_unique_pattern_access()

        # Test 2: Check L2 involvement
        verifier.test_l2_bypass()

        # Test 3: Test with CKE toggle (may trigger self-refresh)
        verifier.test_cke_toggle_decay(powerdown_ms=1000)

        # Test 4: Controlled decay test
        verifier.test_controlled_decay()

        verifier.close()
        return 0

    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
