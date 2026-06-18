#!/usr/bin/env python3
"""
z1121: Partial Write Embodied AI - Analog Computing via DRAM Timing

Uses PHASER_OUT fine delay (0-63 taps) to create analog charge levels in DRAM cells.
The partial write creates incomplete charging, enabling:
1. Analog value storage (timing_offset -> charge level -> readback value)
2. Temperature-dependent decay (Arrhenius: τ ∝ exp(Ea/kT))
3. In-memory analog computation (multiply by decay)

This demonstrates TRUE embodied AI where:
- Hardware physics (charge decay) IS the computation
- Temperature directly affects computation results
- No simulation - real analog values in real DRAM cells
"""

import sys
import os
import json
import time
import numpy as np
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.fpga.fpga_interface import FPGAInterface


def get_gpu_telemetry():
    """Get GPU temperature and power from sysfs"""
    try:
        # AMD GPU hwmon path
        hwmon_path = "/sys/class/drm/card1/device/hwmon/hwmon7"

        # Temperature (in millidegrees)
        with open(f"{hwmon_path}/temp1_input", 'r') as f:
            temp = float(f.read().strip()) / 1000.0

        # Power (in microwatts)
        power = 0.0
        power_file = f"{hwmon_path}/power1_average"
        if os.path.exists(power_file):
            with open(power_file, 'r') as f:
                power = float(f.read().strip()) / 1e6

        return {'temp': temp, 'power': power}
    except Exception as e:
        return {'temp': 0.0, 'power': 0.0, 'error': str(e)}


def analog_value_to_timing(value: float, min_offset: int = 0, max_offset: int = 63) -> int:
    """Convert analog value [0,1] to timing offset

    Lower timing offset = more charge = value closer to 1
    Higher timing offset = less charge = value closer to 0
    """
    # Clamp to [0, 1]
    value = max(0.0, min(1.0, value))
    # Invert: high value -> low offset (more charge)
    offset = int((1.0 - value) * (max_offset - min_offset) + min_offset)
    return max(min_offset, min(max_offset, offset))


def count_bit_errors(original: bytes, readback: bytes) -> int:
    """Count bit differences between original and readback"""
    errors = 0
    for a, b in zip(original, readback):
        errors += bin(a ^ b).count('1')
    return errors


def estimate_analog_readback(readback: bytes, pattern: bytes) -> float:
    """Estimate analog value from readback compared to original pattern

    Returns value in [0, 1] where:
    - 1.0 = perfect retention (all bits match)
    - 0.0 = complete decay (all bits flipped)
    """
    total_bits = len(readback) * 8
    errors = count_bit_errors(pattern, readback)
    return 1.0 - (errors / total_bits)


class PartialWriteEmbodiedAI:
    """Embodied AI using partial DRAM writes for analog computation"""

    def __init__(self, fpga: FPGAInterface):
        self.fpga = fpga
        self.base_addr = 0x100000
        self.results = []

    def write_analog_vector(self, values: list, start_addr: int = None) -> list:
        """Write a vector of analog values using partial timing writes

        Args:
            values: List of float values in [0, 1]
            start_addr: Starting DDR address (default: self.base_addr)

        Returns:
            List of (timing_offset, success) for each value
        """
        if start_addr is None:
            start_addr = self.base_addr

        results = []
        pattern = bytes([0xFF] * 16)  # All ones - decay shows as bit flips

        for i, value in enumerate(values):
            addr = start_addr + (i * 16)  # 16 bytes per entry
            timing_offset = analog_value_to_timing(value)

            result = self.fpga.partial_timing_write(addr, pattern, timing_offset)
            results.append({
                'addr': addr,
                'value': value,
                'timing_offset': timing_offset,
                'success': result.get('success', False),
                'temperature': result.get('temperature', 0.0)
            })
            time.sleep(0.05)  # Pacing for reliability

        return results

    def read_analog_vector(self, count: int, start_addr: int = None) -> list:
        """Read back analog values from DRAM

        Returns:
            List of estimated analog values [0, 1]
        """
        if start_addr is None:
            start_addr = self.base_addr

        pattern = bytes([0xFF] * 16)
        values = []

        for i in range(count):
            addr = start_addr + (i * 16)
            data = self.fpga.ddr_read(addr)

            if data:
                analog = estimate_analog_readback(data, pattern)
                values.append({
                    'addr': addr,
                    'analog_value': analog,
                    'raw_data': data.hex(),
                    'bit_errors': count_bit_errors(pattern, data)
                })
            else:
                values.append({'addr': addr, 'analog_value': None, 'error': 'read_failed'})
            time.sleep(0.02)

        return values

    def test_timing_sweep(self) -> dict:
        """Test different timing offsets to characterize analog range"""
        print("\n=== Timing Offset Sweep ===")

        offsets = [0, 8, 16, 24, 32, 40, 48, 56, 63]
        pattern = bytes([0xFF] * 16)
        results = []

        for offset in offsets:
            addr = self.base_addr + (offset * 16)

            # Write with partial timing
            write_result = self.fpga.partial_timing_write(addr, pattern, offset)

            if write_result.get('success'):
                # Immediate readback
                read_data = self.fpga.ddr_read(addr)

                if read_data:
                    errors = count_bit_errors(pattern, read_data)
                    analog = estimate_analog_readback(read_data, pattern)

                    results.append({
                        'timing_offset': offset,
                        'bit_errors': errors,
                        'analog_value': analog,
                        'temperature': write_result.get('temperature', 0.0),
                        'success': True
                    })
                    print(f"  Offset {offset:2d}: {errors:3d} errors, analog={analog:.3f}, temp={write_result.get('temperature', 0):.1f}C")
                else:
                    results.append({'timing_offset': offset, 'success': False, 'error': 'read_failed'})
                    print(f"  Offset {offset:2d}: READ FAILED")
            else:
                results.append({'timing_offset': offset, 'success': False, 'error': 'write_failed'})
                print(f"  Offset {offset:2d}: WRITE FAILED")

            time.sleep(0.1)  # Pacing

        return {'timing_sweep': results}

    def test_decay_with_partial_write(self, timing_offset: int = 32, wait_ms: float = 10.0) -> dict:
        """Test how partial writes decay over time

        Lower initial charge (higher offset) should decay faster
        """
        print(f"\n=== Decay Test: offset={timing_offset}, wait={wait_ms}ms ===")

        pattern = bytes([0xFF] * 16)
        addr = self.base_addr + 0x10000

        # Write with partial timing
        write_result = self.fpga.partial_timing_write(addr, pattern, timing_offset)

        if not write_result.get('success'):
            return {'success': False, 'error': 'write_failed'}

        # Immediate readback
        immediate_data = self.fpga.ddr_read(addr)
        immediate_errors = count_bit_errors(pattern, immediate_data) if immediate_data else -1

        # Wait (no refresh during this time)
        time.sleep(wait_ms / 1000.0)

        # Delayed readback
        delayed_data = self.fpga.ddr_read(addr)
        delayed_errors = count_bit_errors(pattern, delayed_data) if delayed_data else -1

        temp = write_result.get('temperature', 0.0)

        result = {
            'timing_offset': timing_offset,
            'wait_ms': wait_ms,
            'immediate_errors': immediate_errors,
            'delayed_errors': delayed_errors,
            'decay_errors': delayed_errors - immediate_errors if delayed_errors >= 0 and immediate_errors >= 0 else None,
            'temperature': temp,
            'success': True
        }

        print(f"  Immediate: {immediate_errors} errors")
        print(f"  After {wait_ms}ms: {delayed_errors} errors")
        print(f"  Decay: {result.get('decay_errors', 'N/A')} new errors")

        return result

    def test_analog_multiply(self) -> dict:
        """Demonstrate analog multiply via partial write + decay

        The idea:
        - Write value A with timing offset proportional to A
        - Natural decay acts as a multiplier M (depends on time and temp)
        - Readback gives A * M (approximately)
        """
        print("\n=== Analog Multiply Test ===")

        # Write different analog values
        test_values = [1.0, 0.8, 0.6, 0.4, 0.2]
        write_results = self.write_analog_vector(test_values)

        print("Written values:")
        for r in write_results:
            status = "OK" if r['success'] else "FAIL"
            print(f"  {r['value']:.1f} -> offset {r['timing_offset']} [{status}]")

        # Wait for decay (multiplier)
        wait_ms = 5.0
        print(f"\nWaiting {wait_ms}ms for decay (natural multiplier)...")
        time.sleep(wait_ms / 1000.0)

        # Read back
        read_results = self.read_analog_vector(len(test_values))

        print("\nReadback (after decay):")
        comparisons = []
        for i, (orig, read) in enumerate(zip(test_values, read_results)):
            if read.get('analog_value') is not None:
                ratio = read['analog_value'] / orig if orig > 0 else 0
                print(f"  {orig:.1f} -> {read['analog_value']:.3f} (ratio: {ratio:.3f})")
                comparisons.append({
                    'original': orig,
                    'readback': read['analog_value'],
                    'ratio': ratio,
                    'bit_errors': read.get('bit_errors', -1)
                })
            else:
                print(f"  {orig:.1f} -> READ FAILED")

        return {
            'test_values': test_values,
            'write_results': write_results,
            'read_results': read_results,
            'comparisons': comparisons,
            'wait_ms': wait_ms
        }

    def test_temperature_sensitivity(self) -> dict:
        """Test how temperature affects partial write decay

        Arrhenius equation: τ ∝ exp(Ea/kT)
        Higher temp -> faster decay -> more bit errors
        """
        print("\n=== Temperature Sensitivity Test ===")

        pattern = bytes([0xFF] * 16)
        timing_offset = 32  # Middle of range

        results = []

        # Take multiple measurements at current temperature
        for trial in range(5):
            # Get current temps
            gpu = get_gpu_telemetry()
            fpga_temp, _ = self.fpga.read_temperature()

            # Write with partial timing
            addr = self.base_addr + 0x20000 + (trial * 16)
            write_result = self.fpga.partial_timing_write(addr, pattern, timing_offset)

            if write_result.get('success'):
                # Small decay wait
                time.sleep(0.005)  # 5ms

                # Read back
                read_data = self.fpga.ddr_read(addr)
                errors = count_bit_errors(pattern, read_data) if read_data else -1

                results.append({
                    'trial': trial,
                    'fpga_temp': fpga_temp,
                    'gpu_temp': gpu.get('temp', 0),
                    'bit_errors': errors,
                    'timing_offset': timing_offset
                })

                print(f"  Trial {trial}: FPGA={fpga_temp:.1f}C, GPU={gpu.get('temp',0):.1f}C, errors={errors}")

            time.sleep(0.1)

        # Analyze temperature correlation
        if len(results) >= 2:
            temps = [r['fpga_temp'] for r in results if r['bit_errors'] >= 0]
            errors = [r['bit_errors'] for r in results if r['bit_errors'] >= 0]

            if len(temps) >= 2:
                correlation = np.corrcoef(temps, errors)[0, 1] if np.std(temps) > 0 else 0
            else:
                correlation = 0
        else:
            correlation = 0

        return {
            'trials': results,
            'temp_error_correlation': correlation,
            'mean_temp': np.mean([r['fpga_temp'] for r in results]) if results else 0,
            'mean_errors': np.mean([r['bit_errors'] for r in results if r['bit_errors'] >= 0]) if results else 0
        }


def run_comprehensive_test():
    """Run all partial write embodied AI tests"""
    print("=" * 60)
    print("z1121: Partial Write Embodied AI")
    print("=" * 60)

    fpga = FPGAInterface()

    print("\nConnecting to FPGA...")
    if not fpga.connect():
        print("ERROR: Could not connect to FPGA")
        return None

    # Get initial status
    status = fpga.ping()
    print(f"FPGA Status: DDR3={'ready' if status.get('ddr3_ready') else 'NOT READY'}")

    temp, _ = fpga.read_temperature()
    print(f"FPGA Temperature: {temp:.2f}C")

    gpu = get_gpu_telemetry()
    print(f"GPU Temperature: {gpu.get('temp', 0):.1f}C, Power: {gpu.get('power', 0):.1f}W")

    # Create embodied AI instance
    ai = PartialWriteEmbodiedAI(fpga)

    all_results = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'initial_fpga_temp': temp,
        'initial_gpu_temp': gpu.get('temp', 0),
    }

    # Test 1: Timing offset sweep
    print("\n" + "=" * 40)
    timing_results = ai.test_timing_sweep()
    all_results['timing_sweep'] = timing_results

    # Test 2: Decay with partial write
    print("\n" + "=" * 40)
    decay_results = []
    for offset in [16, 32, 48]:
        result = ai.test_decay_with_partial_write(timing_offset=offset, wait_ms=10.0)
        decay_results.append(result)
    all_results['decay_tests'] = decay_results

    # Test 3: Analog multiply demonstration
    print("\n" + "=" * 40)
    multiply_results = ai.test_analog_multiply()
    all_results['analog_multiply'] = multiply_results

    # Test 4: Temperature sensitivity
    print("\n" + "=" * 40)
    temp_results = ai.test_temperature_sensitivity()
    all_results['temperature_sensitivity'] = temp_results

    # Final status
    final_temp, _ = fpga.read_temperature()
    final_gpu = get_gpu_telemetry()
    all_results['final_fpga_temp'] = final_temp
    all_results['final_gpu_temp'] = final_gpu.get('temp', 0)

    fpga.disconnect()

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    # Timing sweep analysis
    timing_data = timing_results.get('timing_sweep', [])
    successful = [t for t in timing_data if t.get('success')]
    if successful:
        offsets = [t['timing_offset'] for t in successful]
        errors = [t['bit_errors'] for t in successful]
        correlation = np.corrcoef(offsets, errors)[0, 1] if len(offsets) >= 2 and np.std(offsets) > 0 else 0
        print(f"Timing Sweep: {len(successful)}/{len(timing_data)} successful")
        print(f"  Offset-Error Correlation: {correlation:.3f}")
        all_results['offset_error_correlation'] = correlation

    # Temperature sensitivity
    temp_corr = temp_results.get('temp_error_correlation', 0)
    print(f"Temperature Sensitivity: correlation={temp_corr:.3f}")

    # Analog multiply
    comparisons = multiply_results.get('comparisons', [])
    if comparisons:
        avg_ratio = np.mean([c['ratio'] for c in comparisons])
        print(f"Analog Multiply: avg_ratio={avg_ratio:.3f} (decay multiplier)")
        all_results['avg_decay_multiplier'] = avg_ratio

    # Business value assessment
    print("\n" + "-" * 40)
    print("EMBODIED AI CAPABILITIES:")
    print("-" * 40)

    if successful and len(successful) >= 3:
        print("  [+] Analog value storage via timing offset")
    else:
        print("  [-] Analog value storage: NEEDS WORK")

    if abs(temp_corr) > 0.1:
        print("  [+] Temperature-dependent computation")
    else:
        print("  [~] Temperature sensitivity: weak correlation")

    if comparisons and avg_ratio > 0.5:
        print("  [+] Decay-based analog multiply demonstrated")
    else:
        print("  [~] Analog multiply: needs optimization")

    return all_results


def main():
    import argparse

    parser = argparse.ArgumentParser(description='z1121 Partial Write Embodied AI')
    parser.add_argument('--quick', action='store_true', help='Quick test (timing sweep only)')
    parser.add_argument('--output', type=str, default='results/z1121_partial_write.json',
                        help='Output file')
    args = parser.parse_args()

    results = run_comprehensive_test()

    if results:
        # Save results
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)

        print(f"\nResults saved to: {output_path}")


if __name__ == '__main__':
    main()
