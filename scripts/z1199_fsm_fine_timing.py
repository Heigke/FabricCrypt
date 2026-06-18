#!/usr/bin/env python3
"""
z1199: Fine-Grained FSM Timing Test

Test the partial write FSM with:
1. Very fine tRAS granularity (0-10 cycles)
2. Multiple patterns (0->1 and 1->0 transitions)
3. Multiple trials per setting
4. Statistical analysis of bit transition rates
"""

import time
import sys
import json

sys.path.insert(0, '/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy/venv/lib/python3.12/site-packages')

from litex.tools.litex_client import RemoteClient

# CSR addresses
CSR_BASE = 0x00000000
PARTIAL_WRITE_BASE = CSR_BASE + (5 * 0x800)  # Location 5

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

SDRAM_BASE = CSR_BASE + (6 * 0x800)
DFII_CONTROL = SDRAM_BASE + 0x00


def make_config(row, col, bank, tras_cycles):
    """Build config register value: [row:14][col:10][bank:3][tras:5]"""
    config = 0
    config |= (row & 0x3FFF) << 18
    config |= (col & 0x3FF) << 8
    config |= (bank & 0x7) << 5
    config |= (tras_cycles & 0x1F)
    return config


class FineTimingTester:
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

    def init_ddr3(self):
        """Enable HW mode"""
        DFII_CONTROL_SEL = 0x01
        DFII_CONTROL_CKE = 0x02
        DFII_CONTROL_ODT = 0x04
        DFII_CONTROL_RESET_N = 0x08
        ctrl = DFII_CONTROL_SEL | DFII_CONTROL_CKE | DFII_CONTROL_ODT | DFII_CONTROL_RESET_N
        self.write(DFII_CONTROL, ctrl)
        time.sleep(0.1)
        print("DDR3 init: HW mode enabled")

    def run_fsm(self, row, col, bank, tras_cycles, ref_data, write_data):
        """Run a single FSM partial write test"""
        config = make_config(row, col, bank, tras_cycles)
        self.write(PW_CONFIG, config)
        self.write(PW_REF_DATA, ref_data)
        self.write(PW_WRITE_DATA, write_data)
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

        return {
            "result": self.read(PW_RESULT),
            "p1": self.read(PW_RESULT_P1),
            "p2": self.read(PW_RESULT_P2),
            "p3": self.read(PW_RESULT_P3),
            "status": status,
        }

    def count_transitions(self, ref, result):
        """Count bit transitions from ref to result"""
        xor = ref ^ result
        return bin(xor).count('1')

    def test_fine_timing(self):
        """Test with fine tRAS granularity"""
        print("\n=== FINE tRAS TIMING TEST ===")
        print("Testing each cycle from 0-15 (0-150ns)")

        results = []
        base_row = 3000

        for tras in range(16):
            row = base_row + tras

            # Test 0->1 transition (ref=0x00000000, write=0xFFFFFFFF)
            r0to1 = self.run_fsm(row, 0, 0, tras, 0x00000000, 0xFFFFFFFF)
            bits_set = bin(r0to1["result"]).count('1')

            # Test 1->0 transition (ref=0xFFFFFFFF, write=0x00000000)
            r1to0 = self.run_fsm(row, 0, 1, tras, 0xFFFFFFFF, 0x00000000)
            bits_clear = 32 - bin(r1to0["result"]).count('1')

            results.append({
                "tras_cycles": tras,
                "tras_ns": tras * 10,
                "0to1_result": hex(r0to1["result"]),
                "0to1_bits": bits_set,
                "1to0_result": hex(r1to0["result"]),
                "1to0_bits": bits_clear,
            })

            print(f"  tRAS={tras:2d} ({tras*10:3d}ns): 0->1: {bits_set:2d}/32  1->0: {bits_clear:2d}/32  [{hex(r0to1['result'])}] [{hex(r1to0['result'])}]")

        return results

    def test_multiple_trials(self):
        """Run multiple trials at key tRAS values"""
        print("\n=== MULTI-TRIAL TEST ===")
        print("Testing consistency at key tRAS values")

        test_tras_values = [0, 2, 4, 6, 8, 10]
        trials_per_value = 5
        results = {}

        for tras in test_tras_values:
            trial_results = []
            base_row = 4000 + tras * 10

            for trial in range(trials_per_value):
                row = base_row + trial
                r = self.run_fsm(row, 0, 0, tras, 0x00000000, 0xFFFFFFFF)
                trial_results.append(r["result"])

            # Analyze consistency
            unique_values = list(set(trial_results))
            bits_list = [bin(v).count('1') for v in trial_results]

            results[tras] = {
                "trials": [hex(v) for v in trial_results],
                "unique_count": len(unique_values),
                "bits_range": [min(bits_list), max(bits_list)],
            }

            print(f"  tRAS={tras:2d}: {len(unique_values)} unique values, bits: {min(bits_list)}-{max(bits_list)}")
            for v in unique_values[:3]:
                print(f"           {hex(v)} ({bin(v).count('1')}/32 bits)")

        return results

    def test_alternating_pattern(self):
        """Test with alternating bit patterns to see which bits are sensitive"""
        print("\n=== ALTERNATING PATTERN TEST ===")
        print("Testing 0xAAAAAAAA vs 0x55555555")

        results = []
        base_row = 5000

        for tras in [0, 5, 10, 15]:
            row = base_row + tras

            # Write 0xAAAAAAAA over 0x55555555
            r1 = self.run_fsm(row, 0, 0, tras, 0x55555555, 0xAAAAAAAA)

            # Write 0x55555555 over 0xAAAAAAAA
            r2 = self.run_fsm(row, 0, 1, tras, 0xAAAAAAAA, 0x55555555)

            results.append({
                "tras": tras,
                "55_to_AA": hex(r1["result"]),
                "AA_to_55": hex(r2["result"]),
            })

            print(f"  tRAS={tras:2d}: 0x55->0xAA: {hex(r1['result'])}  0xAA->0x55: {hex(r2['result'])}")

        return results


def main():
    print("z1199: Fine-Grained FSM Timing Test")
    print("="*60)

    results = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "tests": {}
    }

    try:
        tester = FineTimingTester()
        tester.init_ddr3()

        results["tests"]["fine_timing"] = tester.test_fine_timing()
        results["tests"]["multi_trial"] = tester.test_multiple_trials()
        results["tests"]["alternating"] = tester.test_alternating_pattern()

        tester.close()

        output_path = "/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy/results/z1199_fsm_fine_timing.json"
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {output_path}")

        # Analysis
        print("\n=== ANALYSIS ===")
        fine = results["tests"]["fine_timing"]
        bits_0to1 = [r["0to1_bits"] for r in fine]
        bits_1to0 = [r["1to0_bits"] for r in fine]
        print(f"0->1 transitions: {min(bits_0to1)}-{max(bits_0to1)} bits (mean: {sum(bits_0to1)/len(bits_0to1):.1f})")
        print(f"1->0 transitions: {min(bits_1to0)}-{max(bits_1to0)} bits (mean: {sum(bits_1to0)/len(bits_1to0):.1f})")

        if max(bits_0to1) - min(bits_0to1) > 5:
            print("SIGNIFICANT tRAS dependence detected for 0->1 transitions!")
        if max(bits_1to0) - min(bits_1to0) > 5:
            print("SIGNIFICANT tRAS dependence detected for 1->0 transitions!")

        return 0

    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
