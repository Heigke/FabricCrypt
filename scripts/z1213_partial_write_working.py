#!/usr/bin/env python3
"""
z1213: Partial Write via DFII Software Mode - WORKING VERSION

Key findings:
- Write on WRPHASE=3
- Read on any phase, but data appears on phase 0 due to latency
- Software mode writes and reads work!

This script performs partial write experiments by varying tRAS timing.
"""

import time
import json
from datetime import datetime
from litex.tools.litex_client import RemoteClient

# DFII registers
DFII_CONTROL = 0x3000
PHASE_SIZE = 0x18

# Command bits
CMD_PRE = 0x0B      # PRE: RAS|WE|CS
CMD_ACT = 0x09      # ACT: RAS|CS
CMD_WRITE = 0x17    # WRITE: CAS|WE|CS|WRDATA
CMD_READ = 0x25     # READ: CAS|CS|RDDATA


class DDR3SWMode:
    def __init__(self, wb):
        self.wb = wb
        self.WRPHASE = 3
        self.RDPHASE = 2  # But actual data appears on phase 0

    def pi_base(self, phase):
        return 0x3004 + phase * PHASE_SIZE

    def command(self, phase, cmd, addr=0, bank=0):
        base = self.pi_base(phase)
        self.wb.write(base + 0x08, addr)   # address
        self.wb.write(base + 0x0c, bank)   # bank
        self.wb.write(base + 0x00, cmd)    # command
        self.wb.write(base + 0x04, 1)      # issue

    def set_wrdata(self, phase, data):
        base = self.pi_base(phase)
        self.wb.write(base + 0x10, data)

    def get_rddata(self, phase):
        base = self.pi_base(phase)
        return self.wb.read(base + 0x14)

    def enter(self):
        self.wb.write(DFII_CONTROL, 0x0E)
        time.sleep(0.01)

    def exit(self):
        self.wb.write(DFII_CONTROL, 0x0F)
        time.sleep(0.01)

    def precharge_all(self):
        self.command(0, CMD_PRE, addr=0x400)
        time.sleep(0.0001)

    def activate(self, row, bank):
        self.command(0, CMD_ACT, addr=row, bank=bank)
        time.sleep(0.0001)

    def write(self, col, bank, data, cycles_before_precharge=50):
        """Write data with controllable timing before precharge."""
        self.set_wrdata(self.WRPHASE, data)
        self.command(self.WRPHASE, CMD_WRITE, addr=col, bank=bank)
        # Wait specified cycles (in ~10us units for now)
        time.sleep(cycles_before_precharge * 0.00001)

    def read(self, col, bank):
        """Read and return data from phase 0 (where it appears)."""
        self.command(self.RDPHASE, CMD_READ, addr=col, bank=bank)
        time.sleep(0.001)  # Wait for read latency
        # Data appears on phase 0 due to timing
        return self.get_rddata(0)


def partial_write_experiment(ddr, row, bank, col, ref_data, write_data, short_timing):
    """
    Perform partial write experiment:
    1. Write ref_data with normal timing
    2. Write write_data with shortened timing (partial write)
    3. Read back result
    """
    ddr.enter()

    # Step 1: Write reference data with full timing
    ddr.precharge_all()
    ddr.activate(row, bank)
    ddr.write(col, bank, ref_data, cycles_before_precharge=100)  # Full timing
    ddr.precharge_all()

    # Step 2: Write test data with shortened timing
    ddr.activate(row, bank)
    ddr.write(col, bank, write_data, cycles_before_precharge=short_timing)
    ddr.precharge_all()  # Interrupt!

    # Step 3: Read back
    ddr.activate(row, bank)
    result = ddr.read(col, bank)
    ddr.precharge_all()

    ddr.exit()
    return result


def main():
    print("=" * 60)
    print("z1213: Partial Write via DFII Software Mode - WORKING")
    print("=" * 60)

    wb = RemoteClient(host='localhost', port=1234)
    wb.open()
    print(f"\nConnected! Identifier: 0x{wb.read(0x1800):08x}")

    ddr = DDR3SWMode(wb)

    # Test basic write/read first
    print("\n=== Basic Write/Read Test ===")
    ddr.enter()
    ddr.precharge_all()
    ddr.activate(row=0, bank=0)
    ddr.write(col=0, bank=0, data=0xCAFEBABE, cycles_before_precharge=100)
    ddr.precharge_all()
    ddr.activate(row=0, bank=0)
    result = ddr.read(col=0, bank=0)
    ddr.precharge_all()
    ddr.exit()
    print(f"Write 0xCAFEBABE, read back 0x{result:08x}")

    if result != 0xCAFEBABE:
        print("ERROR: Basic write/read failed!")
        wb.close()
        return

    print("Basic write/read OK!")

    # Partial write experiments
    print("\n=== Partial Write Sweep ===")
    print("ref_data=0x00000000, write_data=0xFFFFFFFF")
    print("Varying timing from 100 (full) down to 1")
    print()

    results = []
    row, bank, col = 0, 0, 0
    ref_data = 0x00000000
    write_data = 0xFFFFFFFF

    for timing in [100, 50, 30, 20, 15, 10, 8, 5, 3, 2, 1]:
        result = partial_write_experiment(ddr, row, bank, col, ref_data, write_data, timing)

        # Count 1s
        ones = bin(result).count('1')
        pct = ones / 32.0 * 100

        print(f"  timing={timing:3d}: result=0x{result:08x} ({ones:2d}/32 bits = {pct:.1f}%)")

        results.append({
            "timing": timing,
            "result": hex(result),
            "ones": ones,
            "percent": pct
        })

    # Save
    output = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "experiment": "z1213 Partial Write Working",
        "ref_data": hex(ref_data),
        "write_data": hex(write_data),
        "results": results
    }

    with open("results/z1213_partial_write_working.json", "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to results/z1213_partial_write_working.json")

    # Analysis
    partial_effects = [r for r in results if 0 < r["percent"] < 100]
    if partial_effects:
        print(f"\n*** PARTIAL WRITE EFFECTS OBSERVED! ***")
        print(f"    {len(partial_effects)} results show partial data")
    elif all(r["percent"] == 100 for r in results):
        print("\n*** All writes complete (100%) - no decay observed ***")
    elif all(r["percent"] == 0 for r in results):
        print("\n*** All results zero - short timing interrupts writes ***")

    wb.close()


if __name__ == "__main__":
    main()
