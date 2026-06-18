#!/usr/bin/env python3
"""
z1122: Deep Partial Write Analysis

Investigates the offset=16 finding from z1121 where 79 bit errors occurred.
This suggests a specific timing window where partial charging happens.

Goals:
1. Find the optimal timing offset range for analog charge levels
2. Test longer decay times to observe temperature-dependent decay
3. Validate reproducibility of partial write effects
"""

import sys
import os
import json
import time
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.fpga.fpga_interface import FPGAInterface


def get_gpu_telemetry():
    """Get GPU temperature and power"""
    try:
        hwmon_path = "/sys/class/drm/card1/device/hwmon/hwmon7"
        with open(f"{hwmon_path}/temp1_input", 'r') as f:
            temp = float(f.read().strip()) / 1000.0
        power = 0.0
        power_file = f"{hwmon_path}/power1_average"
        if os.path.exists(power_file):
            with open(power_file, 'r') as f:
                power = float(f.read().strip()) / 1e6
        return {'temp': temp, 'power': power}
    except:
        return {'temp': 0.0, 'power': 0.0}


def count_bit_errors(original: bytes, readback: bytes) -> int:
    """Count bit differences"""
    return sum(bin(a ^ b).count('1') for a, b in zip(original, readback))


def analyze_error_pattern(original: bytes, readback: bytes) -> dict:
    """Analyze which bits flipped and how"""
    flipped_1_to_0 = 0
    flipped_0_to_1 = 0
    positions = []

    for byte_idx in range(min(len(original), len(readback))):
        diff = original[byte_idx] ^ readback[byte_idx]
        for bit in range(8):
            if diff & (1 << bit):
                pos = byte_idx * 8 + bit
                if original[byte_idx] & (1 << bit):
                    flipped_1_to_0 += 1
                    positions.append(('1->0', pos))
                else:
                    flipped_0_to_1 += 1
                    positions.append(('0->1', pos))

    return {
        'flipped_1_to_0': flipped_1_to_0,
        'flipped_0_to_1': flipped_0_to_1,
        'total_errors': flipped_1_to_0 + flipped_0_to_1,
        'positions': positions[:10]  # First 10 positions
    }


def main():
    print("=" * 60)
    print("z1122: Deep Partial Write Analysis")
    print("=" * 60)

    fpga = FPGAInterface()

    print("\nConnecting to FPGA...")
    if not fpga.connect():
        print("ERROR: Could not connect")
        return

    status = fpga.ping()
    print(f"FPGA Status: DDR3={'ready' if status.get('ddr3_ready') else 'NOT READY'}")

    temp, _ = fpga.read_temperature()
    print(f"FPGA Temperature: {temp:.2f}C")

    gpu = get_gpu_telemetry()
    print(f"GPU Temperature: {gpu.get('temp', 0):.1f}C")

    results = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'initial_temp': temp,
    }

    pattern = bytes([0xFF] * 16)  # All ones
    base_addr = 0x100000

    # ============================================================
    # Test 1: Fine-grained offset sweep around offset=16
    # ============================================================
    print("\n" + "=" * 50)
    print("Test 1: Fine-grained offset sweep (12-20)")
    print("=" * 50)

    fine_sweep = []
    for offset in range(12, 21):  # Explore around 16
        addr = base_addr + (offset * 16)

        # Multiple trials at each offset
        trials = []
        for trial in range(3):
            write_result = fpga.partial_timing_write(addr + (trial * 256), pattern, offset)

            if write_result.get('success'):
                read_data = fpga.ddr_read(addr + (trial * 256))
                if read_data:
                    errors = count_bit_errors(pattern, read_data)
                    analysis = analyze_error_pattern(pattern, read_data)
                    trials.append({
                        'errors': errors,
                        '1_to_0': analysis['flipped_1_to_0'],
                        '0_to_1': analysis['flipped_0_to_1']
                    })
            time.sleep(0.03)

        if trials:
            avg_errors = np.mean([t['errors'] for t in trials])
            print(f"  Offset {offset:2d}: avg={avg_errors:.1f} errors "
                  f"(trials: {[t['errors'] for t in trials]})")
            fine_sweep.append({
                'offset': offset,
                'avg_errors': avg_errors,
                'trials': trials
            })

    results['fine_sweep'] = fine_sweep

    # ============================================================
    # Test 2: Repeated writes at offset=16 to verify reproducibility
    # ============================================================
    print("\n" + "=" * 50)
    print("Test 2: Reproducibility at offset=16 (10 trials)")
    print("=" * 50)

    offset_16_trials = []
    for trial in range(10):
        addr = base_addr + 0x10000 + (trial * 256)

        write_result = fpga.partial_timing_write(addr, pattern, 16)

        if write_result.get('success'):
            read_data = fpga.ddr_read(addr)
            if read_data:
                errors = count_bit_errors(pattern, read_data)
                analysis = analyze_error_pattern(pattern, read_data)

                offset_16_trials.append({
                    'trial': trial,
                    'errors': errors,
                    '1_to_0': analysis['flipped_1_to_0'],
                    '0_to_1': analysis['flipped_0_to_1'],
                    'temp': write_result.get('temperature', 0)
                })
                print(f"  Trial {trial:2d}: {errors:3d} errors "
                      f"(1->0: {analysis['flipped_1_to_0']:2d}, 0->1: {analysis['flipped_0_to_1']:2d})")

        time.sleep(0.05)

    if offset_16_trials:
        errors_list = [t['errors'] for t in offset_16_trials]
        print(f"\n  Mean: {np.mean(errors_list):.1f} errors")
        print(f"  Std:  {np.std(errors_list):.1f}")
        print(f"  Min:  {min(errors_list)}, Max: {max(errors_list)}")

    results['offset_16_trials'] = offset_16_trials

    # ============================================================
    # Test 3: Decay test with longer wait times using decay_test command
    # ============================================================
    print("\n" + "=" * 50)
    print("Test 3: Long decay tests (using FPGA decay command)")
    print("=" * 50)

    decay_tests = []
    # Wait times in cycles (83333 cycles = 1ms at 83MHz)
    wait_times_cycles = [
        833333,    # 10ms
        4166666,   # 50ms
        8333333,   # 100ms
    ]

    for wait_cycles in wait_times_cycles:
        wait_ms = wait_cycles / 83333.0
        print(f"\n  Testing wait={wait_ms:.0f}ms ({wait_cycles} cycles)...")

        # Use the FPGA's built-in decay test (disables refresh)
        result = fpga.decay_test(base_addr + 0x20000, pattern, wait_cycles, timeout=30.0)

        if result.get('success'):
            analysis = analyze_error_pattern(
                result['original_data'],
                result['read_data']
            )
            print(f"    Errors: {result['bit_errors']} (1->0: {analysis['flipped_1_to_0']}, "
                  f"0->1: {analysis['flipped_0_to_1']})")
            print(f"    Temp: {result['temperature']:.1f}C")

            decay_tests.append({
                'wait_ms': wait_ms,
                'wait_cycles': wait_cycles,
                'bit_errors': result['bit_errors'],
                'flipped_1_to_0': analysis['flipped_1_to_0'],
                'flipped_0_to_1': analysis['flipped_0_to_1'],
                'temperature': result['temperature'],
                'original_hex': result['original_data'].hex(),
                'readback_hex': result['read_data'].hex()
            })
        else:
            print(f"    FAILED (timeout or error)")
            decay_tests.append({
                'wait_ms': wait_ms,
                'success': False
            })

        time.sleep(0.5)  # Cool down between tests

    results['decay_tests'] = decay_tests

    # ============================================================
    # Test 4: Combined partial write + decay
    # ============================================================
    print("\n" + "=" * 50)
    print("Test 4: Partial write (offset=16) + decay")
    print("=" * 50)

    combined_tests = []

    # Write with partial timing, then test decay
    addr = base_addr + 0x30000

    # First: write with offset=16 (partial charge)
    write_result = fpga.partial_timing_write(addr, pattern, 16)

    if write_result.get('success'):
        # Immediate read
        immediate = fpga.ddr_read(addr)
        immediate_errors = count_bit_errors(pattern, immediate) if immediate else -1
        print(f"  After partial write (offset=16): {immediate_errors} errors")

        # Now wait and read again (simulating software-controlled decay)
        for wait_ms in [10, 50, 100]:
            print(f"  Waiting {wait_ms}ms...")
            time.sleep(wait_ms / 1000.0)

            delayed = fpga.ddr_read(addr)
            delayed_errors = count_bit_errors(pattern, delayed) if delayed else -1
            fpga_temp, _ = fpga.read_temperature()

            combined_tests.append({
                'wait_ms': wait_ms,
                'errors_after_wait': delayed_errors,
                'temperature': fpga_temp
            })
            print(f"    Errors after {wait_ms}ms: {delayed_errors}, temp={fpga_temp:.1f}C")

    results['combined_tests'] = combined_tests

    # ============================================================
    # Summary
    # ============================================================
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    # Find optimal offset
    if fine_sweep:
        best_offset = max(fine_sweep, key=lambda x: x['avg_errors'])
        print(f"Best offset for analog effect: {best_offset['offset']} "
              f"(avg {best_offset['avg_errors']:.1f} errors)")

    # Decay analysis
    if decay_tests:
        successful_decays = [d for d in decay_tests if d.get('bit_errors', 0) > 0]
        if successful_decays:
            print(f"Decay observed in {len(successful_decays)}/{len(decay_tests)} tests")
            for d in successful_decays:
                print(f"  {d['wait_ms']:.0f}ms: {d['bit_errors']} errors at {d['temperature']:.1f}C")
        else:
            print("No significant decay observed in tests")

    # Offset 16 analysis
    if offset_16_trials:
        nonzero = [t for t in offset_16_trials if t['errors'] > 0]
        print(f"Offset=16 reproducibility: {len(nonzero)}/{len(offset_16_trials)} trials showed errors")

    # Compute correlation if we have temperature variance
    if decay_tests:
        temps = [d['temperature'] for d in decay_tests if 'temperature' in d]
        errors = [d.get('bit_errors', 0) for d in decay_tests if 'temperature' in d]
        if len(temps) >= 2 and np.std(temps) > 0:
            corr = np.corrcoef(temps, errors)[0, 1]
            print(f"Temperature-error correlation: {corr:.3f}")
            results['temp_error_correlation'] = corr

    fpga.disconnect()

    # Save results
    output_path = Path('results/z1122_partial_write_deep.json')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to: {output_path}")


if __name__ == '__main__':
    main()
