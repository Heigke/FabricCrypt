#!/usr/bin/env python3
"""
z1204: Test Partial Write FSM with Correct CSR Addresses

CSR Layout from new build:
- PARTIAL_WRITE_BASE = 0x2800
- SDRAM_BASE = 0x3000
"""

import time
import sys
import json

sys.path.insert(0, '/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy/venv/lib/python3.12/site-packages')

from litex.tools.litex_client import RemoteClient

# CSR addresses from csr.h
CSR_BASE = 0x0

# LEDS
LEDS_BASE = CSR_BASE + 0x2000
LEDS_OUT = LEDS_BASE + 0x00

# DDRPHY
DDRPHY_BASE = CSR_BASE + 0x800
DDRPHY_RST = DDRPHY_BASE + 0x00
DDRPHY_DLY_SEL = DDRPHY_BASE + 0x04
DDRPHY_RDLY_DQ_RST = DDRPHY_BASE + 0x10
DDRPHY_RDLY_DQ_INC = DDRPHY_BASE + 0x14

# Partial Write FSM
PW_BASE = CSR_BASE + 0x2800
PW_CONFIG = PW_BASE + 0x00      # [row:14][col:10][bank:3][tras:5]
PW_WRITE_DATA = PW_BASE + 0x04  # Data to write
PW_REF_DATA = PW_BASE + 0x08    # Reference data
PW_CONTROL = PW_BASE + 0x0c     # bit0=start, bit1=use_ext_dfi
PW_STATUS = PW_BASE + 0x10      # bit0=busy, bit1=done
PW_RESULT = PW_BASE + 0x14      # Result phase 0
PW_RESULT_P1 = PW_BASE + 0x18   # Result phase 1
PW_RESULT_P2 = PW_BASE + 0x1c   # Result phase 2
PW_RESULT_P3 = PW_BASE + 0x20   # Result phase 3
PW_DEBUG = PW_BASE + 0x24       # FSM state
PW_EDGE_COUNT = PW_BASE + 0x28  # Edge detection counter

# SDRAM DFII
SDRAM_BASE = CSR_BASE + 0x3000
DFII_CONTROL = SDRAM_BASE + 0x00
DFII_PI0_COMMAND = SDRAM_BASE + 0x04
DFII_PI0_COMMAND_ISSUE = SDRAM_BASE + 0x08
DFII_PI0_ADDRESS = SDRAM_BASE + 0x0c
DFII_PI0_BADDRESS = SDRAM_BASE + 0x10
DFII_PI0_WRDATA = SDRAM_BASE + 0x14
DFII_PI0_RDDATA = SDRAM_BASE + 0x18

# Phase offsets (7 CSRs per phase)
PHASE_OFFSET = 0x1c

# Control bits
DFII_CONTROL_SEL = 0x01
DFII_CONTROL_CKE = 0x02
DFII_CONTROL_ODT = 0x04
DFII_CONTROL_RESET_N = 0x08

# Command bits
DFII_COMMAND_CS = 0x01
DFII_COMMAND_WE = 0x02
DFII_COMMAND_CAS = 0x04
DFII_COMMAND_RAS = 0x08
DFII_COMMAND_WRDATA = 0x10
DFII_COMMAND_RDDATA = 0x20

CMD_MODE_REGISTER = DFII_COMMAND_RAS | DFII_COMMAND_CAS | DFII_COMMAND_WE | DFII_COMMAND_CS
CMD_PRECHARGE_ALL = DFII_COMMAND_RAS | DFII_COMMAND_WE | DFII_COMMAND_CS
CMD_ZQ_CAL = DFII_COMMAND_WE | DFII_COMMAND_CS

# Main RAM
MAIN_RAM_BASE = 0x40000000


class FSMTester:
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
        """Issue command on phase 0"""
        self.write(DFII_PI0_COMMAND, cmd)
        self.write(DFII_PI0_COMMAND_ISSUE, 1)

    def init_ddr3(self):
        """Run DDR3 initialization sequence"""
        print("\n=== DDR3 INITIALIZATION ===")

        # Step 1: Release reset
        print("1. Release reset...")
        self.write(DFII_PI0_ADDRESS, 0x0)
        self.write(DFII_PI0_BADDRESS, 0)
        self.write(DFII_CONTROL, DFII_CONTROL_ODT | DFII_CONTROL_RESET_N)
        time.sleep(0.001)

        # Step 2: CKE high
        print("2. CKE high...")
        self.write(DFII_CONTROL, DFII_CONTROL_CKE | DFII_CONTROL_ODT | DFII_CONTROL_RESET_N)
        time.sleep(0.001)

        # Step 3-6: Mode registers
        print("3. Load MR2 (CWL=5)...")
        self.write(DFII_PI0_ADDRESS, 0x200)
        self.write(DFII_PI0_BADDRESS, 2)
        self.command_p0(CMD_MODE_REGISTER)
        time.sleep(0.0001)

        print("4. Load MR3...")
        self.write(DFII_PI0_ADDRESS, 0x0)
        self.write(DFII_PI0_BADDRESS, 3)
        self.command_p0(CMD_MODE_REGISTER)
        time.sleep(0.0001)

        print("5. Load MR1...")
        self.write(DFII_PI0_ADDRESS, 0x6)
        self.write(DFII_PI0_BADDRESS, 1)
        self.command_p0(CMD_MODE_REGISTER)
        time.sleep(0.0001)

        print("6. Load MR0 (CL=7)...")
        self.write(DFII_PI0_ADDRESS, 0x930)
        self.write(DFII_PI0_BADDRESS, 0)
        self.command_p0(CMD_MODE_REGISTER)
        time.sleep(0.001)

        # Step 7: ZQ calibration
        print("7. ZQ Calibration...")
        self.write(DFII_PI0_ADDRESS, 0x400)
        self.write(DFII_PI0_BADDRESS, 0)
        self.command_p0(CMD_ZQ_CAL)
        time.sleep(0.001)

        # Switch to HW mode
        print("8. Switch to HW mode...")
        self.write(DFII_CONTROL, DFII_CONTROL_SEL | DFII_CONTROL_CKE | DFII_CONTROL_ODT | DFII_CONTROL_RESET_N)
        time.sleep(0.001)

        print("Init complete!")

    def test_l2_cache(self):
        """Test L2 cache read/write"""
        print("\n=== L2 CACHE TEST ===")

        base = MAIN_RAM_BASE + 0x10000
        test_values = [0xDEADBEEF, 0x12345678, 0xAAAA5555, 0x00000000, 0xFFFFFFFF]

        for i, val in enumerate(test_values):
            addr = base + i * 4
            self.write(addr, val)
            readback = self.read(addr)
            match = "PASS" if readback == val else "FAIL"
            print(f"  Write 0x{val:08X} -> Read 0x{readback:08X} [{match}]")

    def test_fsm_registers(self):
        """Test FSM CSR registers"""
        print("\n=== FSM REGISTER TEST ===")

        # Read current status
        status = self.read(PW_STATUS)
        debug = self.read(PW_DEBUG)
        print(f"  Status: 0x{status:02X} (busy={(status>>0)&1}, done={(status>>1)&1})")
        print(f"  Debug (state): {debug}")

        # Read result registers
        r0 = self.read(PW_RESULT)
        r1 = self.read(PW_RESULT_P1)
        r2 = self.read(PW_RESULT_P2)
        r3 = self.read(PW_RESULT_P3)
        print(f"  Results: P0=0x{r0:08X}, P1=0x{r1:08X}, P2=0x{r2:08X}, P3=0x{r3:08X}")

        # Write test values to config registers
        print("  Writing test config...")
        test_config = (0x100 << 18) | (0x20 << 8) | (0x1 << 5) | 10  # row=256, col=32, bank=1, tras=10
        self.write(PW_CONFIG, test_config)
        self.write(PW_WRITE_DATA, 0xFFFFFFFF)
        self.write(PW_REF_DATA, 0x00000000)

        # Read back
        config_rb = self.read(PW_CONFIG)
        wd_rb = self.read(PW_WRITE_DATA)
        rd_rb = self.read(PW_REF_DATA)
        print(f"  Config readback: 0x{config_rb:08X} (expected 0x{test_config:08X})")
        print(f"  Write data: 0x{wd_rb:08X}")
        print(f"  Ref data: 0x{rd_rb:08X}")

    def run_fsm(self, row, col, bank, tras, verbose=False):
        """Run FSM with given parameters"""
        config = (row << 18) | (col << 8) | (bank << 5) | tras
        self.write(PW_CONFIG, config)
        self.write(PW_WRITE_DATA, 0xFFFFFFFF)
        self.write(PW_REF_DATA, 0x00000000)

        # Ensure control is clear first (to get a proper rising edge)
        self.write(PW_CONTROL, 0x00)
        # Need multiple CSR bus cycles for the edge detector
        _ = self.read(PW_CONTROL)  # Force bus cycle
        _ = self.read(PW_CONTROL)  # Force another
        time.sleep(0.01)  # Longer delay

        # Check initial status
        ctrl0 = self.read(PW_CONTROL)
        status0 = self.read(PW_STATUS)
        state0 = self.read(PW_DEBUG)
        if verbose:
            print(f"    Before start: ctrl=0x{ctrl0:02X} status=0x{status0:02X} state={state0}")

        # Start FSM (rising edge on bit 0)
        self.write(PW_CONTROL, 0x01)  # start=1

        # Check immediately after start
        ctrl1 = self.read(PW_CONTROL)
        status1 = self.read(PW_STATUS)
        state1 = self.read(PW_DEBUG)
        if verbose:
            print(f"    After start:  ctrl=0x{ctrl1:02X} status=0x{status1:02X} state={state1}")

        # Track state progression
        states_seen = [state1]

        # Wait for completion
        timeout = 100
        for i in range(timeout):
            status = self.read(PW_STATUS)
            state = self.read(PW_DEBUG)
            states_seen.append(state)
            if verbose:
                print(f"    tick {i}: status=0x{status:02X} state={state}")
            if status & 0x02:  # done
                break
            time.sleep(0.001)
        else:
            print(f"  TIMEOUT! state={self.read(PW_DEBUG)}")
            return None

        # Clear start
        self.write(PW_CONTROL, 0x00)

        # Get results
        result = {
            'p0': self.read(PW_RESULT),
            'p1': self.read(PW_RESULT_P1),
            'p2': self.read(PW_RESULT_P2),
            'p3': self.read(PW_RESULT_P3),
            'state': self.read(PW_DEBUG),
            'states_seen': list(set(states_seen)),
        }

        if verbose:
            print(f"    States seen: {result['states_seen']}")

        return result

    def test_leds(self):
        """Test LED CSR to verify Etherbone is working"""
        print("\n=== LED TEST ===")

        led_val = self.read(LEDS_OUT)
        print(f"  Current LED value: 0x{led_val:02X}")

        # Toggle LEDs
        for i in range(4):
            self.write(LEDS_OUT, 1 << i)
            time.sleep(0.1)
            led_rb = self.read(LEDS_OUT)
            print(f"  Write 0x{1<<i:02X}, Read 0x{led_rb:02X}")

        # Restore
        self.write(LEDS_OUT, 0x00)

    def test_edge_detection(self):
        """Test if edge detection is working"""
        print("\n=== EDGE DETECTION TEST ===")

        # Read initial edge count
        edge0 = self.read(PW_EDGE_COUNT)
        print(f"  Initial edge count: {edge0}")

        # Clear control and wait
        self.write(PW_CONTROL, 0x00)
        time.sleep(0.1)

        # Read multiple times to confirm cleared state
        for i in range(3):
            ctrl = self.read(PW_CONTROL)
            status = self.read(PW_STATUS)
            state = self.read(PW_DEBUG)
            edge = self.read(PW_EDGE_COUNT)
            print(f"  Read {i}: ctrl=0x{ctrl:02X} status=0x{status:02X} state={state} edges={edge}")
            time.sleep(0.01)

        # Now write control=1 and check
        print("  Writing control=1...")
        self.write(PW_CONTROL, 0x01)

        # Read immediately
        edge1 = self.read(PW_EDGE_COUNT)
        print(f"  Edge count immediately after: {edge1}")

        # Read multiple times to see if state changed
        for i in range(5):
            ctrl = self.read(PW_CONTROL)
            status = self.read(PW_STATUS)
            state = self.read(PW_DEBUG)
            edge = self.read(PW_EDGE_COUNT)
            print(f"  After write {i}: ctrl=0x{ctrl:02X} status=0x{status:02X} state={state} edges={edge}")
            time.sleep(0.01)

        # Toggle again
        print("  Toggling control 0->1 again...")
        self.write(PW_CONTROL, 0x00)
        time.sleep(0.01)
        self.write(PW_CONTROL, 0x01)
        edge2 = self.read(PW_EDGE_COUNT)
        print(f"  Edge count after second toggle: {edge2}")

    def test_fsm_execution(self):
        """Test FSM execution"""
        print("\n=== FSM EXECUTION TEST ===")

        # First test edge detection
        self.test_edge_detection()

        # Then run with verbose to see state progression
        print("\n  Running with verbose=True for tRAS=15:")
        result = self.run_fsm(row=0, col=0, bank=0, tras=15, verbose=True)
        if result:
            print(f"  Result: P0=0x{result['p0']:08X} States: {result['states_seen']}")

        # Test DRAM via normal controller (L2 cache bypass area)
        print("\n  Testing DRAM via L2 cache bypass:")
        test_addr = MAIN_RAM_BASE + 0x100000  # 1MB offset
        test_val = 0xCAFEBABE

        # Write then read
        self.write(test_addr, test_val)
        # Flush cache by reading other addresses
        for i in range(256):
            _ = self.read(MAIN_RAM_BASE + 0x200000 + i * 64)
        # Read back original
        read_val = self.read(test_addr)
        print(f"  Write 0x{test_val:08X} to 0x{test_addr:08X}")
        print(f"  Read back: 0x{read_val:08X}")
        print(f"  Match: {'PASS' if read_val == test_val else 'FAIL'}")

        # Test with different tRAS values
        tras_values = [5, 10, 15, 20, 25, 31]
        print("\n  Summary:")
        for tras in tras_values:
            result = self.run_fsm(row=0, col=0, bank=0, tras=tras)
            if result:
                print(f"  tRAS={tras:2d}: P0=0x{result['p0']:08X} P1=0x{result['p1']:08X} States={result['states_seen']}")
            else:
                print(f"  tRAS={tras:2d}: FAILED")


def main():
    print("z1204: FSM Partial Write Test")
    print("="*60)

    results = {"timestamp": time.strftime("%Y-%m-%d %H:%M:%S")}

    try:
        tester = FSMTester()

        # Check current DFII control
        ctrl = tester.read(DFII_CONTROL)
        print(f"\nCurrent DFII control: 0x{ctrl:02X}")

        # Initialize DDR3
        tester.init_ddr3()

        # Test LEDs first
        tester.test_leds()

        # Test L2 cache
        tester.test_l2_cache()

        # Test FSM registers
        tester.test_fsm_registers()

        # Test FSM execution
        tester.test_fsm_execution()

        tester.close()

        output_path = "/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy/results/z1204_fsm_test.json"
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
