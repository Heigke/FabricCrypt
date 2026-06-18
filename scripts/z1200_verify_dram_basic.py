#!/usr/bin/env python3
"""
z1200: Verify Basic DRAM Read/Write

Test that we can write and read DRAM correctly using:
1. Standard L2 cache memory interface
2. Direct FSM read-only (to verify FSM read path)
"""

import time
import sys
import json

sys.path.insert(0, '/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy/venv/lib/python3.12/site-packages')

from litex.tools.litex_client import RemoteClient

# Memory map
MAIN_RAM_BASE = 0x40000000  # main_ram at Origin: 0x40000000

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


def make_config(row, col, bank, tras_cycles):
    config = 0
    config |= (row & 0x3FFF) << 18
    config |= (col & 0x3FF) << 8
    config |= (bank & 0x7) << 5
    config |= (tras_cycles & 0x1F)
    return config


class DRAMVerifier:
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

    def test_l2_cache(self):
        """Test basic memory read/write through L2 cache"""
        print("\n=== L2 CACHE MEMORY TEST ===")
        print("Testing standard memory interface...")

        test_patterns = [
            0xFFFFFFFF,
            0x00000000,
            0xAAAAAAAA,
            0x55555555,
            0x12345678,
            0xDEADBEEF,
        ]

        results = []
        base_addr = MAIN_RAM_BASE

        for i, pattern in enumerate(test_patterns):
            addr = base_addr + (i * 4)

            # Write
            self.write(addr, pattern)

            # Read back
            readback = self.read(addr)

            match = readback == pattern
            results.append({
                "addr": hex(addr),
                "written": hex(pattern),
                "readback": hex(readback),
                "match": match
            })

            status = "PASS" if match else "FAIL"
            print(f"  {hex(addr)}: wrote {hex(pattern)}, read {hex(readback)} [{status}]")

        all_pass = all(r["match"] for r in results)
        print(f"\nL2 cache test: {'ALL PASS' if all_pass else 'FAIL'}")
        return results, all_pass

    def test_sequential_write_read(self):
        """Write a block of data, then read it all back"""
        print("\n=== SEQUENTIAL BLOCK TEST ===")
        print("Writing 64 words, then reading back...")

        base_addr = MAIN_RAM_BASE + 0x1000
        num_words = 64

        # Write phase
        for i in range(num_words):
            addr = base_addr + (i * 4)
            pattern = (i * 0x01010101) & 0xFFFFFFFF
            self.write(addr, pattern)

        # Read phase
        errors = 0
        for i in range(num_words):
            addr = base_addr + (i * 4)
            expected = (i * 0x01010101) & 0xFFFFFFFF
            actual = self.read(addr)
            if actual != expected:
                errors += 1
                if errors <= 5:
                    print(f"  ERROR at {hex(addr)}: expected {hex(expected)}, got {hex(actual)}")

        print(f"Sequential test: {num_words - errors}/{num_words} correct")
        return errors == 0

    def test_fsm_read_after_l2_write(self):
        """Write via L2, read via FSM to verify FSM read path"""
        print("\n=== FSM READ PATH TEST ===")
        print("Writing via L2 cache, reading via FSM...")

        # The FSM addresses memory by row/col/bank
        # DDR3 MT41K128M16: 14-bit row, 10-bit col, 3-bit bank
        # With 4 phases and 16-bit datapath:
        # Row 0, Col 0, Bank 0 maps to physical address 0

        # First write known patterns via L2
        test_addrs = [
            MAIN_RAM_BASE + 0x0000,
            MAIN_RAM_BASE + 0x0040,  # Different column
            MAIN_RAM_BASE + 0x2000,  # Different row
        ]

        patterns = [0xCAFEBABE, 0xDEADC0DE, 0x12345678]

        print("  Writing via L2...")
        for addr, pattern in zip(test_addrs, patterns):
            self.write(addr, pattern)
            readback = self.read(addr)
            print(f"    {hex(addr)}: wrote {hex(pattern)}, readback {hex(readback)}")

        print("\n  Reading via FSM (testing FSM read-only)...")
        # Now test if FSM can read these locations
        # Note: Row/Col/Bank addressing may differ from linear addresses

        # For FSM, we use different row numbers to avoid conflicts
        # FSM test at row=0 should read from physical address corresponding to row 0

        # Enable HW mode first
        DFII_CONTROL = SDRAM_BASE + 0x00
        DFII_CONTROL_SEL = 0x01
        DFII_CONTROL_CKE = 0x02
        DFII_CONTROL_ODT = 0x04
        DFII_CONTROL_RESET_N = 0x08
        ctrl = DFII_CONTROL_SEL | DFII_CONTROL_CKE | DFII_CONTROL_ODT | DFII_CONTROL_RESET_N
        self.write(DFII_CONTROL, ctrl)
        time.sleep(0.01)

        # Read from various rows via FSM
        for row in [0, 1, 100, 1000]:
            config = make_config(row=row, col=0, bank=0, tras_cycles=10)
            self.write(PW_CONFIG, config)
            self.write(PW_REF_DATA, 0)
            self.write(PW_WRITE_DATA, 0)
            self.write(PW_CONTROL, 0)
            time.sleep(0.001)
            self.write(PW_CONTROL, 0x01)

            timeout = 1.0
            start_time = time.time()
            while time.time() - start_time < timeout:
                status = self.read(PW_STATUS)
                if (status >> 1) & 1:
                    break
                time.sleep(0.001)

            self.write(PW_CONTROL, 0)

            result = self.read(PW_RESULT)
            print(f"    FSM row={row:4d}: result = {hex(result)}")

    def test_fsm_write_verify_l2(self):
        """Write via FSM, verify via L2 to test FSM write path"""
        print("\n=== FSM WRITE PATH TEST ===")
        print("Writing via FSM, reading via L2 to verify...")

        # Enable HW mode
        DFII_CONTROL = SDRAM_BASE + 0x00
        DFII_CONTROL_SEL = 0x01
        DFII_CONTROL_CKE = 0x02
        DFII_CONTROL_ODT = 0x04
        DFII_CONTROL_RESET_N = 0x08
        ctrl = DFII_CONTROL_SEL | DFII_CONTROL_CKE | DFII_CONTROL_ODT | DFII_CONTROL_RESET_N
        self.write(DFII_CONTROL, ctrl)
        time.sleep(0.01)

        # FSM write at row=5000 (should be far from our L2 test area)
        row = 5000
        test_pattern = 0xBEEFCAFE

        config = make_config(row=row, col=0, bank=0, tras_cycles=10)
        self.write(PW_CONFIG, config)
        self.write(PW_REF_DATA, 0x00000000)
        self.write(PW_WRITE_DATA, test_pattern)
        self.write(PW_CONTROL, 0)
        time.sleep(0.001)
        self.write(PW_CONTROL, 0x01)

        timeout = 1.0
        start_time = time.time()
        while time.time() - start_time < timeout:
            status = self.read(PW_STATUS)
            if (status >> 1) & 1:
                break
            time.sleep(0.001)

        self.write(PW_CONTROL, 0)
        fsm_result = self.read(PW_RESULT)
        print(f"  FSM wrote {hex(test_pattern)}, FSM read: {hex(fsm_result)}")

        # Now try to read the same location via L2
        # Row 5000, col 0, bank 0
        # Address calculation: (row * row_size + col) * burst_bytes
        # For MT41K128M16: 16-bit (2 bytes) * 8 burst = 16 bytes per access
        # row_size = 1024 cols * 8 banks * 16 bytes = 128KB per row?
        # Simplified: row * 0x20000 for 14-bit row

        # This is approximate - actual mapping depends on controller config
        l2_addr = MAIN_RAM_BASE + (row * 0x4000)  # Try different multipliers

        # Flush any cached data by reading different addresses first
        for flush_addr in range(MAIN_RAM_BASE, MAIN_RAM_BASE + 0x2000, 0x40):
            _ = self.read(flush_addr)

        l2_readback = self.read(l2_addr)
        print(f"  L2 read at {hex(l2_addr)}: {hex(l2_readback)}")

        # Try a range of addresses that might correspond to row 5000
        print("\n  Searching for FSM-written data in L2 address space...")
        found = False
        for offset in [0x0, 0x4, 0x8, 0xC]:
            for mult in [0x4000, 0x8000, 0x10000, 0x20000]:
                test_addr = MAIN_RAM_BASE + (row * mult) + offset
                if test_addr >= MAIN_RAM_BASE + 0x10000000:
                    continue
                val = self.read(test_addr)
                if val == test_pattern or val == fsm_result:
                    print(f"    FOUND at {hex(test_addr)}: {hex(val)}")
                    found = True

        if not found:
            print("    Pattern not found in L2 address space (may be address mapping issue)")


def main():
    print("z1200: Verify Basic DRAM Read/Write")
    print("="*60)

    results = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "tests": {}
    }

    try:
        verifier = DRAMVerifier()

        l2_results, l2_pass = verifier.test_l2_cache()
        results["tests"]["l2_cache"] = {"results": l2_results, "pass": l2_pass}

        seq_pass = verifier.test_sequential_write_read()
        results["tests"]["sequential"] = {"pass": seq_pass}

        verifier.test_fsm_read_after_l2_write()
        verifier.test_fsm_write_verify_l2()

        verifier.close()

        output_path = "/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy/results/z1200_dram_basic.json"
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
