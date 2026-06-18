#!/usr/bin/env python3
"""
z1202: Address Mapping Discovery

Write distinctive patterns via L2, then use FSM to read various row/col/bank
combinations to discover the address mapping.
"""

import time
import sys
import json

sys.path.insert(0, '/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy/venv/lib/python3.12/site-packages')

from litex.tools.litex_client import RemoteClient

# Memory map
MAIN_RAM_BASE = 0x40000000

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

DFII_CONTROL = SDRAM_BASE + 0x00


def make_config(row, col, bank, tras_cycles):
    config = 0
    config |= (row & 0x3FFF) << 18
    config |= (col & 0x3FF) << 8
    config |= (bank & 0x7) << 5
    config |= (tras_cycles & 0x1F)
    return config


class AddressMapper:
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

    def init_hw_mode(self):
        """Set DFII to HW mode"""
        ctrl = 0x0F  # sel=1, cke=1, odt=1, reset_n=1
        self.write(DFII_CONTROL, ctrl)
        time.sleep(0.01)

    def fsm_read(self, row, col, bank):
        """Read using FSM and return result"""
        config = make_config(row, col, bank, tras_cycles=10)
        self.write(PW_CONFIG, config)
        self.write(PW_REF_DATA, 0)
        self.write(PW_WRITE_DATA, 0)
        self.write(PW_CONTROL, 0)
        time.sleep(0.001)
        self.write(PW_CONTROL, 0x01)

        timeout = 0.5
        start_time = time.time()
        while time.time() - start_time < timeout:
            status = self.read(PW_STATUS)
            if (status >> 1) & 1:
                break
            time.sleep(0.001)

        self.write(PW_CONTROL, 0)
        return self.read(PW_RESULT)

    def write_patterns(self):
        """Write distinctive patterns via L2"""
        print("\n=== WRITING PATTERNS VIA L2 ===")

        # Write patterns at various offsets
        patterns = {}

        # Pattern at base: marker for row=0, col=0, bank=0
        for offset in range(0, 0x10000, 0x100):
            pattern = 0x10000000 | (offset & 0xFFFF)
            addr = MAIN_RAM_BASE + offset
            self.write(addr, pattern)
            patterns[offset] = pattern

        # Write some distinctive patterns
        test_addrs = [
            (0x00000000, 0xAAAA0000),  # Base
            (0x00000004, 0xAAAA0004),  # +4
            (0x00000008, 0xAAAA0008),  # +8
            (0x00000010, 0xAAAA0010),  # +16
            (0x00000040, 0xAAAA0040),  # +64 (column?)
            (0x00000100, 0xAAAA0100),  # +256
            (0x00001000, 0xAAAA1000),  # +4K
            (0x00002000, 0xAAAA2000),  # +8K (possible row boundary?)
            (0x00004000, 0xAAAA4000),  # +16K
            (0x00008000, 0xAAAA8000),  # +32K
            (0x00010000, 0xAAAAAAAA),  # +64K (row boundary?)
            (0x00020000, 0xBBBBBBBB),  # +128K
            (0x00040000, 0xCCCCCCCC),  # +256K
            (0x00100000, 0xDDDDDDDD),  # +1M (different row region?)
        ]

        for offset, pattern in test_addrs:
            addr = MAIN_RAM_BASE + offset
            self.write(addr, pattern)
            readback = self.read(addr)
            print(f"  {hex(addr)} ({hex(offset)}): wrote {hex(pattern)}, read {hex(readback)}")

        return test_addrs

    def scan_fsm_addresses(self):
        """Scan various row/col/bank via FSM and look for patterns"""
        print("\n=== FSM ADDRESS SCAN ===")

        results = []

        # Scan rows at col=0, bank=0
        print("\nRow scan (col=0, bank=0):")
        for row in [0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]:
            result = self.fsm_read(row, 0, 0)
            results.append(("row", row, 0, 0, result))
            if result != 0 and result != 0xDEADDEAD:
                print(f"  row={row:5d}: {hex(result)} ***")
            else:
                print(f"  row={row:5d}: {hex(result)}")

        # Scan columns at row=0, bank=0
        print("\nColumn scan (row=0, bank=0):")
        for col in [0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512]:
            result = self.fsm_read(0, col, 0)
            results.append(("col", 0, col, 0, result))
            if result != 0 and result != 0xDEADDEAD:
                print(f"  col={col:4d}: {hex(result)} ***")
            else:
                print(f"  col={col:4d}: {hex(result)}")

        # Scan banks at row=0, col=0
        print("\nBank scan (row=0, col=0):")
        for bank in range(8):
            result = self.fsm_read(0, 0, bank)
            results.append(("bank", 0, 0, bank, result))
            if result != 0 and result != 0xDEADDEAD:
                print(f"  bank={bank}: {hex(result)} ***")
            else:
                print(f"  bank={bank}: {hex(result)}")

        return results

    def try_find_pattern(self):
        """Try to find our written patterns"""
        print("\n=== SEARCHING FOR WRITTEN PATTERNS ===")

        found = []

        # Look for 0xAAAAxxxx pattern
        for row in range(0, 100):
            for bank in range(8):
                result = self.fsm_read(row, 0, bank)
                if (result & 0xFFFF0000) == 0xAAAA0000:
                    print(f"  FOUND 0xAAAA at row={row}, bank={bank}: {hex(result)}")
                    found.append((row, 0, bank, result))

        # Also check if results match any expected pattern
        expected = [0xAAAA0000, 0xAAAA0004, 0xAAAA0008, 0xAAAA0010, 0xAAAA0040]
        for row in range(0, 50):
            for col in range(0, 16):
                for bank in range(8):
                    result = self.fsm_read(row, col, bank)
                    if result in expected:
                        print(f"  FOUND {hex(result)} at row={row}, col={col}, bank={bank}")
                        found.append((row, col, bank, result))

        return found


def main():
    print("z1202: Address Mapping Discovery")
    print("="*60)

    results = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    try:
        mapper = AddressMapper()
        mapper.init_hw_mode()

        mapper.write_patterns()
        scan_results = mapper.scan_fsm_addresses()
        found = mapper.try_find_pattern()

        results["scan"] = scan_results
        results["found"] = found

        mapper.close()

        output_path = "/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy/results/z1202_address_mapping.json"
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\nResults saved to {output_path}")

        return 0

    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
