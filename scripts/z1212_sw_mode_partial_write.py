#!/usr/bin/env python3
"""
z1212: Partial Write via DFII Software Mode

Since ext_dfi doesn't seem to be routing commands correctly,
this script uses DFII software mode (phase injectors) to perform
the partial write experiment.

Sequence:
1. Enter software mode (SEL=0)
2. Write reference pattern to DRAM cell
3. Write test pattern with shortened tRAS
4. Read back and observe decay effects
5. Return to hardware mode

This is slower than ext_dfi but uses proven working infrastructure.
"""

import time
import json
from datetime import datetime
from litex.tools.litex_client import RemoteClient

# CSR addresses
CSR_BASE = 0x0
DRAM_BASE = 0x40000000

# DFII registers
DFII_CONTROL = 0x3000
DFII_PI0_COMMAND = 0x3004
DFII_PI0_COMMAND_ISSUE = 0x3008
DFII_PI0_ADDRESS = 0x300c
DFII_PI0_BADDRESS = 0x3010
DFII_PI0_WRDATA = 0x3014
DFII_PI0_RDDATA = 0x3018

# DFII Control bits
DFII_SEL = 0x01
DFII_CKE = 0x02
DFII_ODT = 0x04
DFII_RESET_N = 0x08

# DFII Command bits
DFII_CS = 0x01
DFII_WE = 0x02
DFII_CAS = 0x04
DFII_RAS = 0x08
DFII_WRDATA = 0x10
DFII_RDDATA = 0x20

# DDR3 Commands (active low, so we use the asserted form)
CMD_NOP = 0
CMD_ACT = DFII_CS | DFII_RAS
CMD_READ = DFII_CS | DFII_CAS | DFII_RDDATA
CMD_WRITE = DFII_CS | DFII_CAS | DFII_WE | DFII_WRDATA
CMD_PRE = DFII_CS | DFII_RAS | DFII_WE  # Precharge
CMD_PRE_ALL = DFII_CS | DFII_RAS | DFII_WE  # With A10=1


def cdelay(n):
    """Delay for approximately n cycles."""
    time.sleep(n * 0.00001)


class DDR3Controller:
    def __init__(self, wb):
        self.wb = wb

    def enter_sw_mode(self):
        """Enter software control mode."""
        self.wb.write(DFII_CONTROL, DFII_CKE | DFII_ODT | DFII_RESET_N)  # SEL=0
        cdelay(100)

    def enter_hw_mode(self):
        """Return to hardware control mode."""
        self.wb.write(DFII_CONTROL, DFII_SEL | DFII_CKE | DFII_ODT | DFII_RESET_N)
        cdelay(100)

    def command(self, cmd, addr=0, bank=0):
        """Issue a command on phase 0."""
        self.wb.write(DFII_PI0_ADDRESS, addr)
        self.wb.write(DFII_PI0_BADDRESS, bank)
        self.wb.write(DFII_PI0_COMMAND, cmd)
        self.wb.write(DFII_PI0_COMMAND_ISSUE, 1)
        cdelay(10)

    def precharge_all(self):
        """Precharge all banks."""
        self.command(CMD_PRE_ALL, addr=0x400)  # A10=1

    def activate(self, row, bank):
        """Activate a row."""
        self.command(CMD_ACT, addr=row, bank=bank)

    def write_data(self, col, bank, data):
        """Write data at column in active row."""
        self.wb.write(DFII_PI0_WRDATA, data)
        self.command(CMD_WRITE, addr=col, bank=bank)

    def read_data(self, col, bank):
        """Read data at column in active row."""
        self.command(CMD_READ, addr=col, bank=bank)
        cdelay(50)  # Wait for read data
        return self.wb.read(DFII_PI0_RDDATA)


def partial_write_experiment(wb, row, bank, col, ref_data, write_data, tras_cycles):
    """
    Perform partial write experiment:
    1. Write ref_data with full tRAS
    2. Write write_data with shortened tRAS (partial write)
    3. Read back result
    """
    ddr = DDR3Controller(wb)
    ddr.enter_sw_mode()

    # Step 1: Write reference data (full timing)
    ddr.precharge_all()
    cdelay(20)
    ddr.activate(row, bank)
    cdelay(20)
    ddr.write_data(col, bank, ref_data)
    cdelay(100)  # Full tRAS
    ddr.precharge_all()
    cdelay(20)

    # Step 2: Write test data (shortened timing)
    ddr.activate(row, bank)
    cdelay(20)
    ddr.write_data(col, bank, write_data)
    cdelay(tras_cycles)  # Shortened tRAS!
    ddr.precharge_all()  # Interrupt before fully written
    cdelay(20)

    # Step 3: Read back
    ddr.activate(row, bank)
    cdelay(20)
    result = ddr.read_data(col, bank)
    ddr.precharge_all()

    ddr.enter_hw_mode()
    return result


def main():
    print("=" * 60)
    print("z1212: Partial Write via DFII Software Mode")
    print("=" * 60)

    wb = RemoteClient(host='localhost', port=1234)
    wb.open()
    print(f"\nConnected! Identifier: 0x{wb.read(0x1800):08x}")

    # Test parameters
    row = 0x10
    bank = 0
    col = 0
    ref_data = 0x00000000
    write_data = 0xFFFFFFFF

    results = []

    print("\n=== Partial Write Sweep ===")
    print("Varying tRAS from 100 (full) down to 1 cycle")
    print("ref_data=0x00000000, write_data=0xFFFFFFFF")
    print()

    for tras in [100, 50, 30, 20, 15, 10, 8, 5, 3, 1]:
        result = partial_write_experiment(wb, row, bank, col, ref_data, write_data, tras)

        # Count 1 bits
        ones = bin(result).count('1')
        pct = ones / 32.0 * 100

        print(f"  tRAS={tras:3d} cycles: result=0x{result:08x} ({ones:2d}/32 bits = {pct:.1f}%)")

        results.append({
            "tras_cycles": tras,
            "result": result,
            "ones": ones,
            "percent": pct
        })

    # Verify normal memory still works
    print("\n=== Verify Hardware Mode Memory ===")
    test_addr = DRAM_BASE + 0x1000
    wb.write(test_addr, 0xCAFEBABE)
    time.sleep(0.01)
    readback = wb.read(test_addr)
    print(f"  Write 0xCAFEBABE, read 0x{readback:08x}")

    # Save results
    output = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "experiment": "z1212 Partial Write via SW Mode",
        "ref_data": hex(ref_data),
        "write_data": hex(write_data),
        "row": row,
        "bank": bank,
        "col": col,
        "results": results
    }

    with open("results/z1212_sw_mode_partial_write.json", "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to results/z1212_sw_mode_partial_write.json")

    if any(r["percent"] < 100 and r["percent"] > 0 for r in results):
        print("\n*** PARTIAL WRITE EFFECTS OBSERVED! ***")
    else:
        print("\n*** No partial write effects observed ***")

    wb.close()


if __name__ == "__main__":
    main()
