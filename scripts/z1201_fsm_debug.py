#!/usr/bin/env python3
"""
z1201: FSM Debug - Check ext_dfi_sel and timing

Verify the FSM state machine by:
1. Checking if ext_dfi_sel is being set
2. Testing longer wait times
3. Checking all phase results
"""

import time
import sys
import json

sys.path.insert(0, '/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy/venv/lib/python3.12/site-packages')

from litex.tools.litex_client import RemoteClient

# CSR addresses
CSR_BASE = 0x00000000
PARTIAL_WRITE_BASE = CSR_BASE + (5 * 0x800)
SDRAM_BASE = CSR_BASE + (6 * 0x800)

PW_CONFIG = PARTIAL_WRITE_BASE + 0x00
PW_WRITE_DATA = PARTIAL_WRITE_BASE + 0x04
PW_REF_DATA = PARTIAL_WRITE_BASE + 0x08
PW_CONTROL = PARTIAL_WRITE_BASE + 0x0C
PW_STATUS = PARTIAL_WRITE_BASE + 0x10
PW_RESULT = PARTIAL_WRITE_BASE + 0x14
PW_RESULT_P1 = PARTIAL_WRITE_BASE + 0x18
PW_RESULT_P2 = PARTIAL_WRITE_BASE + 0x1C
PW_RESULT_P3 = PARTIAL_WRITE_BASE + 0x20
PW_DEBUG = PARTIAL_WRITE_BASE + 0x24

DFII_CONTROL = SDRAM_BASE + 0x00

# Known DFII offsets - let's check the actual register layout
# From litedram CSR output typically:
# sdram_dfii_control = base + 0x00
# sdram_dfii_pi0_command = base + 0x04
# sdram_dfii_pi0_command_issue = base + 0x08
# etc.


def make_config(row, col, bank, tras_cycles):
    config = 0
    config |= (row & 0x3FFF) << 18
    config |= (col & 0x3FF) << 8
    config |= (bank & 0x7) << 5
    config |= (tras_cycles & 0x1F)
    return config


class FSMDebugger:
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

    def dump_csr_region(self, name, base, count=16):
        """Dump CSR values around a base address"""
        print(f"\n{name} CSR region ({hex(base)}):")
        for i in range(count):
            addr = base + (i * 4)
            try:
                val = self.read(addr)
                print(f"  {hex(addr)}: {hex(val)}")
            except Exception as e:
                print(f"  {hex(addr)}: ERROR - {e}")

    def check_dfii_state(self):
        """Check DFII control and status registers"""
        print("\n=== DFII STATE ===")

        # Read control register
        ctrl = self.read(DFII_CONTROL)
        print(f"DFII_CONTROL: {hex(ctrl)}")
        print(f"  sel (bit 0): {(ctrl >> 0) & 1} (1=HW, 0=SW)")
        print(f"  cke (bit 1): {(ctrl >> 1) & 1}")
        print(f"  odt (bit 2): {(ctrl >> 2) & 1}")
        print(f"  reset_n (bit 3): {(ctrl >> 3) & 1}")

        # Dump nearby SDRAM CSRs
        self.dump_csr_region("SDRAM/DFII", SDRAM_BASE, 8)

    def check_partial_write_csrs(self):
        """Check partial write module CSRs"""
        print("\n=== PARTIAL_WRITE CSRs ===")
        self.dump_csr_region("PARTIAL_WRITE", PARTIAL_WRITE_BASE, 12)

    def test_fsm_trigger(self):
        """Test FSM trigger and state transitions"""
        print("\n=== FSM TRIGGER TEST ===")

        # Set HW mode
        DFII_CONTROL_SEL = 0x01
        DFII_CONTROL_CKE = 0x02
        DFII_CONTROL_ODT = 0x04
        DFII_CONTROL_RESET_N = 0x08
        ctrl = DFII_CONTROL_SEL | DFII_CONTROL_CKE | DFII_CONTROL_ODT | DFII_CONTROL_RESET_N
        self.write(DFII_CONTROL, ctrl)
        time.sleep(0.01)

        print(f"Set DFII_CONTROL to {hex(ctrl)}")

        # Read initial state
        status = self.read(PW_STATUS)
        debug = self.read(PW_DEBUG)
        print(f"Initial: status={hex(status)}, debug_state={debug}")

        # Set up minimal config
        config = make_config(row=100, col=0, bank=0, tras_cycles=10)
        self.write(PW_CONFIG, config)
        self.write(PW_REF_DATA, 0x00000000)
        self.write(PW_WRITE_DATA, 0xFFFFFFFF)

        print(f"Config set: {hex(config)}")

        # Clear control
        self.write(PW_CONTROL, 0)
        time.sleep(0.005)

        # Read state before trigger
        status = self.read(PW_STATUS)
        debug = self.read(PW_DEBUG)
        print(f"Before trigger: status={hex(status)}, debug_state={debug}")

        # Trigger FSM
        print("Triggering FSM...")
        self.write(PW_CONTROL, 0x01)

        # Monitor state transitions
        for i in range(20):
            time.sleep(0.001)
            status = self.read(PW_STATUS)
            debug = self.read(PW_DEBUG)
            busy = (status >> 0) & 1
            done = (status >> 1) & 1
            print(f"  t={i}ms: status={hex(status)}, state={debug}, busy={busy}, done={done}")
            if done:
                break

        # Clear control
        self.write(PW_CONTROL, 0)

        # Read results
        print("\nResults:")
        print(f"  Result P0: {hex(self.read(PW_RESULT))}")
        print(f"  Result P1: {hex(self.read(PW_RESULT_P1))}")
        print(f"  Result P2: {hex(self.read(PW_RESULT_P2))}")
        print(f"  Result P3: {hex(self.read(PW_RESULT_P3))}")

    def test_simple_read(self):
        """Test simple memory read to verify data path"""
        print("\n=== SIMPLE READ TEST ===")
        print("First write known pattern via L2, then verify FSM can't see it...")

        # Write via L2
        MAIN_RAM_BASE = 0x40000000
        test_pattern = 0xDEADBEEF
        self.write(MAIN_RAM_BASE, test_pattern)

        # Verify via L2
        readback = self.read(MAIN_RAM_BASE)
        print(f"  L2 write/read: wrote {hex(test_pattern)}, read {hex(readback)}")

        # Check if FSM sees different data
        print("\n  Note: FSM uses row/col/bank addressing,")
        print("  which may not map directly to linear L2 addresses.")


def main():
    print("z1201: FSM Debug")
    print("="*60)

    try:
        debugger = FSMDebugger()

        debugger.check_dfii_state()
        debugger.check_partial_write_csrs()
        debugger.test_fsm_trigger()
        debugger.test_simple_read()

        debugger.close()
        return 0

    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
