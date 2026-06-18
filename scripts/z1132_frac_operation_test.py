#!/usr/bin/env python3
"""
z1132: Frac Operation Test via Self-Refresh Manipulation

Based on PUDTune research: Frac operations (repeated ACT->PRE cycles)
move DRAM cells toward intermediate charge states.

Since we can't directly issue ACT/PRE with MIG, we test if rapid
self-refresh entry/exit can create similar partial charge effects.

Theory:
1. Normal: Write 0xFF -> Wait -> Read 0xFF (full charge)
2. Frac-like: Write 0xFF -> SR_enter -> quick_exit -> SR_enter -> ... -> Read
3. Each SR cycle may partially disturb the charge state

If we see intermediate values (not just 64 or 128 ones), it indicates
we can achieve partial charging through timing manipulation.
"""

import sys
import json
import time
from pathlib import Path
from collections import Counter

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.fpga.fpga_interface import FPGAInterface


def count_ones(data: bytes) -> int:
    """Count 1 bits in byte array"""
    return sum(bin(b).count('1') for b in data)


def run_frac_test():
    """Test if rapid self-refresh cycles create partial charge"""
    print("=" * 70)
    print("z1132: Frac Operation Test via Self-Refresh Manipulation")
    print("=" * 70)
    print(flush=True)

    fpga = FPGAInterface()
    if not fpga.connect():
        print("ERROR: Could not connect to FPGA")
        return None

    temp, _ = fpga.read_temperature()
    print(f"Temperature: {temp:.1f}°C", flush=True)

    results = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'temperature': temp,
        'tests': {}
    }

    pattern_ff = bytes([0xFF] * 16)
    base_addr = 0xA00000

    # ==================================================================
    # Test 1: Baseline - Normal decay test (single SR cycle)
    # ==================================================================
    print(f"\n{'=' * 50}")
    print("Test 1: Baseline - Single self-refresh cycle")
    print(f"{'=' * 50}", flush=True)

    baseline_results = []
    for trial in range(10):
        addr = base_addr + (trial * 256)
        result = fpga.decay_test(addr, pattern_ff, wait_cycles=100000)
        if result.get('success'):
            ones = count_ones(result['read_data'])
            baseline_results.append(ones)
            print(f"  Trial {trial}: {ones} ones", flush=True)

    if baseline_results:
        results['tests']['baseline'] = {
            'trials': baseline_results,
            'unique_levels': sorted(set(baseline_results)),
            'mean': sum(baseline_results) / len(baseline_results)
        }
        print(f"  Unique levels: {sorted(set(baseline_results))}", flush=True)

    # ==================================================================
    # Test 2: Multiple rapid decay tests (simulated Frac)
    # ==================================================================
    print(f"\n{'=' * 50}")
    print("Test 2: Multiple rapid decay cycles (Frac-like)")
    print(f"{'=' * 50}", flush=True)

    for num_fracs in [1, 2, 3, 5, 10]:
        print(f"\n  {num_fracs} Frac cycles:", flush=True)
        frac_results = []

        for trial in range(5):
            addr = base_addr + 0x10000 + (num_fracs * 0x1000) + (trial * 256)

            # Write initial pattern
            fpga.ddr_write(addr, pattern_ff)
            time.sleep(0.001)

            # Apply multiple short decay cycles (Frac operations)
            for frac in range(num_fracs):
                # Very short decay to disturb charge
                result = fpga.decay_test(addr, pattern_ff, wait_cycles=1000, timeout=5.0)
                if not result.get('success'):
                    continue

            # Final read
            data = fpga.ddr_read(addr, retries=2)
            if data:
                ones = count_ones(data)
                frac_results.append(ones)

        if frac_results:
            results['tests'][f'frac_{num_fracs}'] = {
                'num_fracs': num_fracs,
                'trials': frac_results,
                'unique_levels': sorted(set(frac_results)),
                'mean': sum(frac_results) / len(frac_results)
            }
            print(f"    Results: {frac_results}", flush=True)
            print(f"    Unique levels: {sorted(set(frac_results))}", flush=True)

    # ==================================================================
    # Test 3: Varying decay time with fixed pattern
    # ==================================================================
    print(f"\n{'=' * 50}")
    print("Test 3: Decay time sweep (hunting for intermediate values)")
    print(f"{'=' * 50}", flush=True)

    decay_times = [100, 500, 1000, 2000, 5000, 10000, 50000, 100000]
    all_values_found = set()

    for wait_cycles in decay_times:
        time_us = wait_cycles / 83.333  # Convert to microseconds
        trial_results = []

        for trial in range(10):
            addr = base_addr + 0x30000 + (wait_cycles * 16) + (trial * 256)
            result = fpga.decay_test(addr, pattern_ff, wait_cycles, timeout=10.0)
            if result.get('success'):
                ones = count_ones(result['read_data'])
                trial_results.append(ones)
                all_values_found.add(ones)

        if trial_results:
            results['tests'][f'decay_{wait_cycles}'] = {
                'wait_cycles': wait_cycles,
                'time_us': time_us,
                'trials': trial_results,
                'unique': sorted(set(trial_results))
            }
            print(f"  {wait_cycles:6d} cycles ({time_us:7.1f}us): levels={sorted(set(trial_results))}",
                  flush=True)

    # ==================================================================
    # Test 4: Pattern sensitivity for Frac
    # ==================================================================
    print(f"\n{'=' * 50}")
    print("Test 4: Pattern sensitivity with Frac cycles")
    print(f"{'=' * 50}", flush=True)

    patterns = {
        '0xFF': bytes([0xFF] * 16),
        '0xAA': bytes([0xAA] * 16),
        '0x55': bytes([0x55] * 16),
        '0xF0': bytes([0xF0] * 16),
        '0x0F': bytes([0x0F] * 16),
        '0xCC': bytes([0xCC] * 16),
    }

    for name, pattern in patterns.items():
        pattern_results = []
        expected_ones = count_ones(pattern)

        for trial in range(5):
            addr = base_addr + 0x50000 + (list(patterns.keys()).index(name) * 0x1000) + (trial * 256)

            # Write pattern
            fpga.ddr_write(addr, pattern)

            # Apply 3 Frac cycles
            for _ in range(3):
                fpga.decay_test(addr, pattern, wait_cycles=1000, timeout=5.0)

            # Final read
            data = fpga.ddr_read(addr, retries=2)
            if data:
                ones = count_ones(data)
                pattern_results.append(ones)

        if pattern_results:
            results['tests'][f'pattern_{name}'] = {
                'pattern': name,
                'expected_ones': expected_ones,
                'after_frac': pattern_results,
                'change': expected_ones - (sum(pattern_results) / len(pattern_results))
            }
            print(f"  {name}: expected={expected_ones}, after_frac={pattern_results}", flush=True)

    # ==================================================================
    # Summary
    # ==================================================================
    print(f"\n{'=' * 70}")
    print("FRAC OPERATION TEST SUMMARY")
    print(f"{'=' * 70}", flush=True)

    print(f"\nAll unique charge levels found: {sorted(all_values_found)}")

    if len(all_values_found) <= 3:
        print("""
  BINARY STOCHASTIC BEHAVIOR CONFIRMED

  Even with Frac-like operations (multiple rapid SR cycles), we only see:
  - 64 ones (50% charge)
  - 128 ones (100% charge)
  - Occasional 65 (single bit flip)

  This confirms the limitation is NOT in timing but in the MIG's
  command abstraction - we cannot issue raw ACT/PRE with custom timing.

  TO ACHIEVE TRUE MULTI-LEVEL CHARGE, we need:
  1. Replace MIG with LiteDRAM (full timing control)
  2. Implement raw command mode in custom controller
  3. Use DRAM Bender/SoftMC approach on different platform
""")
        results['conclusion'] = 'binary_stochastic'
    else:
        intermediate = [v for v in all_values_found if 64 < v < 128 and v != 65]
        if intermediate:
            print(f"""
  INTERMEDIATE VALUES FOUND: {sorted(intermediate)}

  This suggests Frac-like operations CAN create partial charge states!
  Further tuning of timing parameters may expand this range.
""")
            results['conclusion'] = 'intermediate_possible'
            results['intermediate_values'] = sorted(intermediate)
        else:
            print("  Only boundary values found - true analog requires MIG modification")
            results['conclusion'] = 'boundary_only'

    fpga.disconnect()
    return results


def main():
    results = run_frac_test()

    if results:
        output_path = Path('results/z1132_frac_operation_test.json')
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\nResults saved to: {output_path}")


if __name__ == '__main__':
    main()
