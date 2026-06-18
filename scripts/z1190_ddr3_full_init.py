#!/usr/bin/env python3
"""
z1190: Full DDR3 Initialization via Etherbone

This implements the complete DDR3 init sequence from LiteDRAM's sdram_phy.h:
1. Release reset
2. Bring CKE high
3. Load Mode Registers (MR0-MR3)
4. ZQ Calibration
5. Read leveling (delay sweep)

Based on:
- LiteDRAM init.py init sequence
- sdram_phy.h generated header
- core_ddr3_controller DLL-off approach insights
"""

import time
import sys

# Add venv to path
sys.path.insert(0, '/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy/venv/lib/python3.12/site-packages')

from litex.tools.litex_client import RemoteClient

# CSR Addresses (from build_with_cpu csr.h)
CSR_BASE = 0xf0000000

# DDRPHY CSRs (0x800-0x830)
DDRPHY_BASE = CSR_BASE + 0x800
DDRPHY_RST = DDRPHY_BASE + 0x00           # CSR 39 - _rst
DDRPHY_DLY_SEL = DDRPHY_BASE + 0x04       # CSR 40 - _dly_sel (byte lane select)
DDRPHY_HALF_SYS8X = DDRPHY_BASE + 0x08    # CSR 41 - _half_sys8x_taps
DDRPHY_WDLY_BITSLIP_RST = DDRPHY_BASE + 0x0c  # CSR 42
DDRPHY_RDLY_DQ_RST = DDRPHY_BASE + 0x10   # CSR 43 - _rdly_dq_rst
DDRPHY_RDLY_DQ_INC = DDRPHY_BASE + 0x14   # CSR 44 - _rdly_dq_inc
DDRPHY_RDLY_BITSLIP_RST = DDRPHY_BASE + 0x18  # CSR 45
DDRPHY_RDLY_BITSLIP = DDRPHY_BASE + 0x1c  # CSR 46
DDRPHY_WDLY_BITSLIP_RST2 = DDRPHY_BASE + 0x20  # CSR 47
DDRPHY_WDLY_BITSLIP = DDRPHY_BASE + 0x24  # CSR 48
DDRPHY_CSR_49 = DDRPHY_BASE + 0x28        # CSR 49
DDRPHY_RDPHASE = DDRPHY_BASE + 0x2c       # CSR 50 - _rdphase
DDRPHY_WRPHASE = DDRPHY_BASE + 0x30       # CSR 51 - _wrphase

# SDRAM DFII CSRs (0x2800-0x2860)
SDRAM_BASE = CSR_BASE + 0x2800
DFII_CONTROL = SDRAM_BASE + 0x00          # CSR 57 - control
DFII_PI0_COMMAND = SDRAM_BASE + 0x04      # CSR 58 - pi0 command
DFII_PI0_COMMAND_ISSUE = SDRAM_BASE + 0x08  # CSR 59 - pi0 command issue
DFII_PI0_ADDRESS = SDRAM_BASE + 0x0c      # CSR 60 - pi0 address
DFII_PI0_BADDRESS = SDRAM_BASE + 0x10     # CSR 61 - pi0 baddress (bank)
DFII_PI0_WRDATA = SDRAM_BASE + 0x14       # CSR 62 - pi0 wrdata
DFII_PI0_RDDATA = SDRAM_BASE + 0x18       # CSR 63 - pi0 rddata

# Phase offsets (each phase is 6 CSRs apart = 0x18 bytes)
PHASE_OFFSET = 0x18
def get_phase_base(phase):
    return SDRAM_BASE + 0x04 + (phase * PHASE_OFFSET)

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

# Compound commands
CMD_MODE_REGISTER = DFII_COMMAND_RAS | DFII_COMMAND_CAS | DFII_COMMAND_WE | DFII_COMMAND_CS
CMD_PRECHARGE_ALL = DFII_COMMAND_RAS | DFII_COMMAND_WE | DFII_COMMAND_CS
CMD_ACTIVATE = DFII_COMMAND_RAS | DFII_COMMAND_CS
CMD_READ = DFII_COMMAND_CAS | DFII_COMMAND_CS | DFII_COMMAND_RDDATA
CMD_WRITE = DFII_COMMAND_CAS | DFII_COMMAND_WE | DFII_COMMAND_CS | DFII_COMMAND_WRDATA
CMD_ZQ_CAL = DFII_COMMAND_WE | DFII_COMMAND_CS


class DDR3Controller:
    def __init__(self, host="localhost", port=1234):
        # Connect to litex_server on localhost (it relays to FPGA via UDP)
        self.wb = RemoteClient(host=host, port=port, csr_csv=None)
        self.wb.open()
        print(f"Connected to {host}:{port}")

    def close(self):
        self.wb.close()

    def read(self, addr):
        return self.wb.read(addr)

    def write(self, addr, val):
        self.wb.write(addr, val)

    def cdelay(self, cycles):
        """Delay in cycles (approximate via sleep)"""
        # At 100MHz, 1 cycle = 10ns
        time.sleep(cycles * 10e-9)

    def command_p0(self, cmd):
        """Issue command on phase 0"""
        self.write(DFII_PI0_COMMAND, cmd)
        self.write(DFII_PI0_COMMAND_ISSUE, 1)

    def command_phase(self, phase, cmd):
        """Issue command on specified phase"""
        base = get_phase_base(phase)
        self.write(base, cmd)
        self.write(base + 4, 1)  # command_issue

    def init_sequence(self):
        """Run full DDR3 initialization sequence from sdram_phy.h"""
        print("\n" + "="*60)
        print("DDR3 INITIALIZATION SEQUENCE")
        print("="*60)

        # Step 1: Release reset (UNRESET = ODT | RESET_N)
        print("\n1. Release reset...")
        self.write(DFII_PI0_ADDRESS, 0x0)
        self.write(DFII_PI0_BADDRESS, 0)
        self.write(DFII_CONTROL, DFII_CONTROL_ODT | DFII_CONTROL_RESET_N)
        self.cdelay(50000)
        time.sleep(0.001)  # Extra safety margin

        # Step 2: Bring CKE high
        print("2. Bring CKE high...")
        self.write(DFII_PI0_ADDRESS, 0x0)
        self.write(DFII_PI0_BADDRESS, 0)
        self.write(DFII_CONTROL, DFII_CONTROL_CKE | DFII_CONTROL_ODT | DFII_CONTROL_RESET_N)
        self.cdelay(10000)
        time.sleep(0.001)

        # Step 3: Load Mode Register 2 (CWL=5)
        # MR2 = 0x200 for our config
        print("3. Load MR2 (CWL=5)...")
        self.write(DFII_PI0_ADDRESS, 0x200)
        self.write(DFII_PI0_BADDRESS, 2)
        self.command_p0(CMD_MODE_REGISTER)
        time.sleep(0.0001)

        # Step 4: Load Mode Register 3
        print("4. Load MR3...")
        self.write(DFII_PI0_ADDRESS, 0x0)
        self.write(DFII_PI0_BADDRESS, 3)
        self.command_p0(CMD_MODE_REGISTER)
        time.sleep(0.0001)

        # Step 5: Load Mode Register 1 (RON, RTT_NOM)
        # MR1 = 0x6 (60ohm RTT_NOM, 34ohm RON)
        print("5. Load MR1 (ODT settings)...")
        self.write(DFII_PI0_ADDRESS, 0x6)
        self.write(DFII_PI0_BADDRESS, 1)
        self.command_p0(CMD_MODE_REGISTER)
        time.sleep(0.0001)

        # Step 6: Load Mode Register 0 (CL=7, BL=8, DLL reset)
        # MR0 = 0x930
        print("6. Load MR0 (CL=7, BL=8, DLL reset)...")
        self.write(DFII_PI0_ADDRESS, 0x930)
        self.write(DFII_PI0_BADDRESS, 0)
        self.command_p0(CMD_MODE_REGISTER)
        self.cdelay(200)
        time.sleep(0.001)

        # Step 7: ZQ Calibration
        print("7. ZQ Calibration...")
        self.write(DFII_PI0_ADDRESS, 0x400)
        self.write(DFII_PI0_BADDRESS, 0)
        self.command_p0(CMD_ZQ_CAL)
        self.cdelay(200)
        time.sleep(0.001)

        print("\nInit sequence complete!")

    def set_sw_mode(self):
        """Switch to software control mode"""
        ctrl = DFII_CONTROL_CKE | DFII_CONTROL_ODT | DFII_CONTROL_RESET_N
        # SEL=0 for SW mode (we control), SEL=1 for HW mode (controller controls)
        self.write(DFII_CONTROL, ctrl)  # No SEL bit = SW mode

    def set_hw_mode(self):
        """Switch to hardware control mode"""
        ctrl = DFII_CONTROL_SEL | DFII_CONTROL_CKE | DFII_CONTROL_ODT | DFII_CONTROL_RESET_N
        self.write(DFII_CONTROL, ctrl)

    def precharge_all(self):
        """Issue precharge-all command"""
        self.write(DFII_PI0_ADDRESS, 0x400)  # A10=1 for all banks
        self.write(DFII_PI0_BADDRESS, 0)
        self.command_p0(CMD_PRECHARGE_ALL)
        time.sleep(0.0001)

    def activate(self, bank, row):
        """Activate a row in a bank"""
        self.write(DFII_PI0_ADDRESS, row)
        self.write(DFII_PI0_BADDRESS, bank)
        self.command_p0(CMD_ACTIVATE)
        time.sleep(0.0001)

    def write_dram(self, bank, col, data):
        """Write data to DRAM (row must be activated)"""
        # Set write data on all phases
        for phase in range(4):
            base = get_phase_base(phase)
            self.write(base + 0x10, data)  # wrdata offset within phase

        # Issue write command on write phase (3)
        base = get_phase_base(3)  # WRPHASE=3
        self.write(DFII_PI0_ADDRESS, col)
        self.write(DFII_PI0_BADDRESS, bank)
        self.write(base, CMD_WRITE)
        self.write(base + 4, 1)
        time.sleep(0.0001)

    def read_dram(self, bank, col):
        """Read data from DRAM (row must be activated), returns list of phase data"""
        # Issue read command on read phase (2)
        self.write(DFII_PI0_ADDRESS, col)
        self.write(DFII_PI0_BADDRESS, bank)
        base = get_phase_base(2)  # RDPHASE=2
        self.write(base, CMD_READ)
        self.write(base + 4, 1)
        time.sleep(0.0001)

        # Read data from all phases
        rddata = []
        for phase in range(4):
            base = get_phase_base(phase)
            rddata.append(self.read(base + 0x14))  # rddata offset
        return rddata

    def reset_read_delays(self, module):
        """Reset read delays for a byte lane"""
        # Select module
        self.write(DDRPHY_DLY_SEL, 1 << module)
        # Reset delay
        self.write(DDRPHY_RDLY_DQ_RST, 1)
        time.sleep(0.0001)
        # Deselect
        self.write(DDRPHY_DLY_SEL, 0)

    def inc_read_delay(self, module):
        """Increment read delay for a byte lane"""
        self.write(DDRPHY_DLY_SEL, 1 << module)
        self.write(DDRPHY_RDLY_DQ_INC, 1)
        time.sleep(0.0001)
        self.write(DDRPHY_DLY_SEL, 0)

    def test_memory_basic(self):
        """Basic memory test - write and read back"""
        print("\n" + "="*60)
        print("BASIC MEMORY TEST")
        print("="*60)

        # Switch to SW mode
        self.set_sw_mode()
        time.sleep(0.001)

        # Precharge all banks
        print("\nPrecharge all banks...")
        self.precharge_all()

        # Activate row 0 in bank 0
        print("Activate bank 0, row 0...")
        self.activate(0, 0)

        # Write test pattern
        test_data = 0xDEADBEEF
        print(f"Write 0x{test_data:08X} to col 0...")
        self.write_dram(0, 0, test_data)

        # Read back
        print("Read back...")
        rddata = self.read_dram(0, 0)

        print(f"\nRead data from all phases:")
        for i, d in enumerate(rddata):
            print(f"  Phase {i}: 0x{d:08X}")

        # Check if any phase has our data
        success = any(d == test_data for d in rddata)
        if success:
            print("\n*** SUCCESS! Data matches! ***")
        else:
            print("\n*** FAIL: No phase has matching data ***")

        # Precharge
        self.precharge_all()

        # Back to HW mode
        self.set_hw_mode()

        return success

    def read_delay_sweep(self, module, max_delay=32):
        """Sweep read delay to find working value"""
        print(f"\n--- Read delay sweep for module {module} ---")

        self.set_sw_mode()
        self.reset_read_delays(module)

        working_delays = []

        for delay in range(max_delay):
            # Precharge
            self.precharge_all()

            # Activate
            self.activate(0, 0)

            # Write pattern
            test_data = 0xAAAA5555
            self.write_dram(0, 0, test_data)

            # Read back
            rddata = self.read_dram(0, 0)

            # Check
            match = any(d == test_data for d in rddata)
            if match:
                working_delays.append(delay)
                print(f"  Delay {delay:2d}: PASS")
            else:
                print(f"  Delay {delay:2d}: FAIL (got {[hex(d) for d in rddata]})")

            # Increment delay for next iteration
            self.inc_read_delay(module)

        self.precharge_all()
        self.set_hw_mode()

        return working_delays

    def calibrate_read_leveling(self):
        """Attempt automatic read leveling calibration"""
        print("\n" + "="*60)
        print("READ LEVELING CALIBRATION")
        print("="*60)

        results = {}

        for module in range(2):  # 2 byte lanes
            working = self.read_delay_sweep(module)
            results[module] = working

            if working:
                # Use middle of working range
                best = working[len(working)//2]
                print(f"\nModule {module}: Best delay = {best}")

                # Set to best value
                self.reset_read_delays(module)
                for _ in range(best):
                    self.inc_read_delay(module)
            else:
                print(f"\nModule {module}: NO WORKING DELAYS FOUND!")

        return results


def main():
    print("z1190: DDR3 Full Initialization Test")
    print("="*60)

    try:
        ddr = DDR3Controller()

        # Read current control register
        ctrl = ddr.read(DFII_CONTROL)
        print(f"\nCurrent DFII control: 0x{ctrl:02X}")
        print(f"  SEL={ctrl&1}, CKE={(ctrl>>1)&1}, ODT={(ctrl>>2)&1}, RESET_N={(ctrl>>3)&1}")

        # Check if already in HW mode with proper settings
        if ctrl == 0x0F:  # SEL|CKE|ODT|RESET_N
            print("\nDRAM appears initialized (HW mode active)")
        else:
            print("\nRunning init sequence...")
            ddr.init_sequence()

        # Test basic memory operation
        success = ddr.test_memory_basic()

        if not success:
            print("\n" + "="*60)
            print("Basic test failed - attempting read leveling calibration...")
            print("="*60)

            # Re-run init
            ddr.init_sequence()

            # Try calibration
            results = ddr.calibrate_read_leveling()

            # Re-test
            print("\n" + "="*60)
            print("Re-testing after calibration...")
            print("="*60)
            success = ddr.test_memory_basic()

        ddr.close()

        return 0 if success else 1

    except ConnectionRefusedError:
        print("ERROR: Cannot connect to litex_server")
        print("Make sure to run: litex_server --udp --udp-ip 192.168.0.50")
        return 1
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
