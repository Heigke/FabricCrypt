#!/usr/bin/env python3
"""
z1112: Analog Charge Level Inference via Decay Curve

Key insight: We can't directly write partial charge to DDR3, but we CAN
statistically infer analog charge levels by measuring decay rates:

1. Fully charged cells (1s written) retain longer
2. Partially charged cells (weak writes, cell defects) decay faster
3. By measuring error rate vs time, we get an effective "analog readout"

This is the embodiment signal: the physical state of DRAM cells
manifests as a statistical distribution of decay times.
"""

import sys
import time
import json
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.fpga.fpga_interface import FPGAInterface


def measure_decay_curve(fpga: FPGAInterface, address: int,
                        durations: list, repetitions: int = 3) -> dict:
    """
    Measure decay curve: error rate vs time

    Multiple measurements at each duration give statistical confidence.
    The shape of this curve encodes analog charge information.
    """
    results = {
        'address': address,
        'durations': [],
        'error_rates': [],
        'raw_data': []
    }

    pattern = bytes([0xFF] * 16)  # All 1s - fully charged

    for duration in durations:
        wait_cycles = int(duration * 83333 * 1000)
        errors_at_duration = []

        for rep in range(repetitions):
            # Write fresh pattern
            fpga.ddr_write(address, pattern)
            time.sleep(0.05)  # Small delay for write to complete

            # Run decay test
            result = fpga.decay_test(address, pattern, wait_cycles,
                                     timeout=duration + 10)

            if result.get('success'):
                errors_at_duration.append(result['bit_errors'])
            else:
                errors_at_duration.append(-1)

            time.sleep(0.1)

        # Average error rate for this duration
        valid_errors = [e for e in errors_at_duration if e >= 0]
        if valid_errors:
            avg_errors = np.mean(valid_errors)
            std_errors = np.std(valid_errors)
            error_rate = avg_errors / 64.0  # 64 bits in first 8 bytes
        else:
            avg_errors = -1
            std_errors = 0
            error_rate = 0

        results['durations'].append(duration)
        results['error_rates'].append(error_rate)
        results['raw_data'].append({
            'duration_s': duration,
            'errors': errors_at_duration,
            'mean': avg_errors,
            'std': std_errors,
            'error_rate': error_rate
        })

        print(f"  {duration:6.1f}s: {avg_errors:5.1f} +/- {std_errors:4.1f} errors "
              f"({error_rate*100:5.1f}%)")

    return results


def find_weak_cells(fpga: FPGAInterface, start_addr: int = 0x100000,
                    num_addresses: int = 16, wait_seconds: float = 30.0) -> dict:
    """
    Find cells with partial charge (weak cells) by comparing decay rates
    across different addresses.

    Cells that decay faster have lower effective charge = analog signal!
    """
    results = {
        'wait_seconds': wait_seconds,
        'addresses': [],
        'error_counts': [],
        'weak_cells': []
    }

    pattern = bytes([0xFF] * 16)
    wait_cycles = int(wait_seconds * 83333 * 1000)

    # Write pattern to all addresses first
    print(f"Writing pattern to {num_addresses} addresses...")
    addresses = [start_addr + i * 0x1000 for i in range(num_addresses)]
    for addr in addresses:
        fpga.ddr_write(addr, pattern)
        time.sleep(0.02)

    # Wait for decay (all addresses aging together)
    print(f"Waiting {wait_seconds}s for decay...")
    time.sleep(wait_seconds)

    # Read back and count errors
    print("Reading back and counting errors...")
    for addr in addresses:
        read_data = fpga.ddr_read(addr)
        if read_data:
            errors = sum(bin(a ^ b).count('1') for a, b in zip(pattern, read_data))
            results['addresses'].append(addr)
            results['error_counts'].append(errors)

            if errors > 0:
                results['weak_cells'].append({
                    'address': addr,
                    'errors': errors,
                    'data': read_data.hex()
                })
                print(f"  0x{addr:06X}: {errors} errors - WEAK CELLS!")
            else:
                print(f"  0x{addr:06X}: 0 errors")
        time.sleep(0.02)

    return results


def infer_charge_distribution(decay_curve: dict) -> dict:
    """
    Infer the charge distribution from the decay curve.

    This is the key embodiment insight: the shape of the decay curve
    tells us about the distribution of charge levels in the cells.

    - Steep curve = many partially charged cells
    - Flat curve = uniformly charged cells
    - S-curve = bimodal distribution (some weak, some strong)
    """
    durations = np.array(decay_curve['durations'])
    error_rates = np.array(decay_curve['error_rates'])

    # Simple analysis
    if np.all(error_rates == 0):
        return {
            'interpretation': 'NO_DECAY',
            'message': 'No decay observed - cells fully charged or test too short',
            'suggested_action': 'Increase temperature or wait longer'
        }

    # Compute decay rate (derivative of error curve)
    if len(durations) > 1:
        decay_rate = np.diff(error_rates) / np.diff(durations)
        peak_decay_idx = np.argmax(decay_rate) if len(decay_rate) > 0 else 0

        # The time at which decay accelerates tells us about charge distribution
        # Early peak = many weak cells
        # Late peak = cells well charged
        return {
            'interpretation': 'DECAY_DETECTED',
            'total_error_rate': float(error_rates[-1]) if len(error_rates) > 0 else 0,
            'peak_decay_time': float(durations[peak_decay_idx]) if len(durations) > peak_decay_idx else 0,
            'decay_rate': decay_rate.tolist() if len(decay_rate) > 0 else [],
            'charge_estimate': 'HIGH' if error_rates[-1] < 0.1 else 'MODERATE' if error_rates[-1] < 0.5 else 'LOW',
            'message': f'Decay curve shows {error_rates[-1]*100:.1f}% error rate'
        }

    return {'interpretation': 'INSUFFICIENT_DATA'}


def main():
    print("="*60)
    print("z1112: Analog Charge Level Inference via Decay")
    print("="*60)
    print()
    print("Key concept: We infer analog charge levels from decay statistics")
    print("             rather than trying to write partial charge directly.")
    print()

    fpga = FPGAInterface()

    if not fpga.connect():
        print("ERROR: Could not connect to FPGA")
        return 1

    print("Connected to FPGA")
    temp, _ = fpga.read_temperature()
    print(f"Temperature: {temp:.1f}C")

    all_results = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'initial_temp': temp
    }

    try:
        # Test 1: Measure decay curve at multiple addresses
        print("\n" + "="*60)
        print("1. DECAY CURVE MEASUREMENT")
        print("="*60)

        durations = [1.0, 5.0, 10.0, 30.0, 60.0]

        print(f"\nMeasuring decay at address 0x100000:")
        print(f"Durations: {durations}")
        print("-" * 50)

        decay_curve = measure_decay_curve(fpga, 0x100000, durations, repetitions=2)
        all_results['decay_curve'] = decay_curve

        # Analyze the curve
        print("\nAnalyzing decay curve...")
        analysis = infer_charge_distribution(decay_curve)
        print(f"  Interpretation: {analysis.get('interpretation', 'UNKNOWN')}")
        print(f"  Message: {analysis.get('message', 'N/A')}")
        if 'charge_estimate' in analysis:
            print(f"  Charge estimate: {analysis['charge_estimate']}")
        all_results['charge_analysis'] = analysis

        # Test 2: Find weak cells
        print("\n" + "="*60)
        print("2. WEAK CELL DETECTION")
        print("="*60)

        weak_cell_results = find_weak_cells(fpga, start_addr=0x200000,
                                            num_addresses=8, wait_seconds=30.0)
        all_results['weak_cells'] = weak_cell_results

        if weak_cell_results['weak_cells']:
            print(f"\nFound {len(weak_cell_results['weak_cells'])} addresses with weak cells!")
            print("These represent ANALOG charge levels manifesting as statistical decay.")
        else:
            print("\nNo weak cells detected in this test.")
            print("Try: higher temperature, longer wait, or different addresses.")

        # Save results
        results_file = Path(__file__).parent.parent / 'results' / 'z1112_analog_inference.json'
        results_file.parent.mkdir(exist_ok=True)
        with open(results_file, 'w') as f:
            json.dump(all_results, f, indent=2)
        print(f"\nResults saved to: {results_file}")

    finally:
        fpga.disconnect()

    return 0


if __name__ == '__main__':
    sys.exit(main())
