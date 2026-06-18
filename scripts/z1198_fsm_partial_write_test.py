#!/usr/bin/env python3
"""
z1198: FSM Partial Write Test

Test the hardware FSM for nanosecond-precision partial writes.
The FSM executes:
1. Write reference pattern (full tRAS)
2. Write partial data with truncated tRAS
3. Read back and capture result

At 100MHz, 1 cycle = 10ns
tras_cycles: 0-31 gives 0-310ns control
"""

import time
import sys
import json

sys.path.insert(0, '/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy/venv/lib/python3.12/site-packages')

from litex.tools.litex_client import RemoteClient

# CSR addresses (from build output)
# partial_write is at Location 5, each location = 0x800 bytes
CSR_BASE = 0x00000000
PARTIAL_WRITE_BASE = CSR_BASE + (5 * 0x800)  # Location 5

# CSR offsets within partial_write module
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

# Also need DFII for init
SDRAM_BASE = CSR_BASE + (6 * 0x800)  # Location 6


def make_config(row, col, bank, tras_cycles):
    """Build config register value: [row:14][col:10][bank:3][tras:5]"""
    config = 0
    config |= (row & 0x3FFF) << 18
    config |= (col & 0x3FF) << 8
    config |= (bank & 0x7) << 5
    config |= (tras_cycles & 0x1F)
    return config


class PartialWriteTester:
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

    def run_ddr3_init(self):
        """Run DDR3 init sequence via DFII (same as z1190)"""
        print("\nRunning DDR3 init sequence...")

        # DFII CSRs
        DFII_CONTROL = SDRAM_BASE + 0x00

        DFII_CONTROL_SEL = 0x01
        DFII_CONTROL_CKE = 0x02
        DFII_CONTROL_ODT = 0x04
        DFII_CONTROL_RESET_N = 0x08

        # Set HW mode with all control signals
        ctrl = DFII_CONTROL_SEL | DFII_CONTROL_CKE | DFII_CONTROL_ODT | DFII_CONTROL_RESET_N
        self.write(DFII_CONTROL, ctrl)

        # Wait for controller to stabilize
        time.sleep(0.1)

        print("DDR3 init: HW mode enabled")
        return True

    def run_fsm_test(self, row, col, bank, tras_cycles, ref_data, write_data):
        """Run a single FSM partial write test"""
        # Set config
        config = make_config(row, col, bank, tras_cycles)
        self.write(PW_CONFIG, config)
        self.write(PW_REF_DATA, ref_data)
        self.write(PW_WRITE_DATA, write_data)

        # Clear control (make sure start is 0)
        self.write(PW_CONTROL, 0)
        time.sleep(0.001)

        # Trigger FSM (set start bit)
        self.write(PW_CONTROL, 0x01)

        # Wait for done
        timeout = 1.0
        start_time = time.time()
        while time.time() - start_time < timeout:
            status = self.read(PW_STATUS)
            done = (status >> 1) & 1
            if done:
                break
            time.sleep(0.001)

        # Clear start bit
        self.write(PW_CONTROL, 0)

        # Read results
        result_p0 = self.read(PW_RESULT)
        result_p1 = self.read(PW_RESULT_P1)
        result_p2 = self.read(PW_RESULT_P2)
        result_p3 = self.read(PW_RESULT_P3)
        debug_state = self.read(PW_DEBUG)

        return {
            "status": status,
            "done": done,
            "debug_state": debug_state,
            "result_p0": result_p0,
            "result_p1": result_p1,
            "result_p2": result_p2,
            "result_p3": result_p3,
        }

    def test_basic_operation(self):
        """Test basic FSM operation"""
        print("\n=== BASIC FSM TEST ===")

        # Test with full tRAS (5 cycles = 50ns)
        result = self.run_fsm_test(
            row=1000, col=0, bank=0,
            tras_cycles=5,
            ref_data=0x00000000,
            write_data=0xFFFFFFFF
        )

        print(f"Status: 0x{result['status']:02X}")
        print(f"Done: {result['done']}")
        print(f"Debug state: {result['debug_state']}")
        print(f"Result P0: 0x{result['result_p0']:08X}")
        print(f"Result P1: 0x{result['result_p1']:08X}")
        print(f"Result P2: 0x{result['result_p2']:08X}")
        print(f"Result P3: 0x{result['result_p3']:08X}")

        return result

    def test_tras_sweep(self):
        """Sweep tRAS values and measure bit transitions"""
        print("\n=== tRAS SWEEP TEST ===")
        print("Testing partial writes with varying tRAS cycles")
        print("1 cycle = 10ns at 100MHz\n")

        results = []
        base_row = 2000

        for tras in range(0, 32, 2):  # 0 to 31 cycles, step 2
            # Use different row for each test
            row = base_row + tras

            result = self.run_fsm_test(
                row=row, col=0, bank=0,
                tras_cycles=tras,
                ref_data=0x00000000,
                write_data=0xFFFFFFFF
            )

            # Count bits set (0->1 transitions)
            bits_set = bin(result['result_p0']).count('1')
            tras_ns = tras * 10

            results.append({
                "tras_cycles": tras,
                "tras_ns": tras_ns,
                "value": hex(result['result_p0']),
                "bits_set": bits_set,
                "done": result['done'],
                "debug_state": result['debug_state'],
            })

            print(f"  tRAS={tras:2d} ({tras_ns:3d}ns): 0x{result['result_p0']:08X} ({bits_set:2d}/32 bits) state={result['debug_state']}")

        return results


def main():
    print("z1198: FSM Partial Write Test")
    print("="*60)

    results = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "tests": {}
    }

    try:
        tester = PartialWriteTester()

        # Run DDR3 init
        tester.run_ddr3_init()

        # Basic test
        basic = tester.test_basic_operation()
        results["tests"]["basic"] = basic

        # tRAS sweep
        sweep = tester.test_tras_sweep()
        results["tests"]["tras_sweep"] = sweep

        tester.close()

        # Save results
        output_path = "/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy/results/z1198_fsm_partial_write.json"
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
