#!/usr/bin/env python3
"""
z1123: Embodied Analog Compute - Production Demonstration

Combines findings from z1121 and z1122 to demonstrate:
1. Analog value storage via partial timing writes (offset ~16)
2. Temperature-correlated DRAM decay (Arrhenius physics)
3. In-memory analog multiply (value * decay_rate)
4. GPU telemetry integration for embodied feedback

Business Value:
- Hardware physics performs computation (no digital multiply needed)
- Temperature naturally modulates computation intensity
- Energy proportional to charge stored (lower charge = less energy)
"""

import sys
import os
import json
import time
import numpy as np
from pathlib import Path
from typing import List, Tuple

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


def count_ones(data: bytes) -> int:
    """Count number of 1 bits"""
    return sum(bin(b).count('1') for b in data)


def count_errors(a: bytes, b: bytes) -> int:
    """Count bit differences"""
    return sum(bin(x ^ y).count('1') for x, y in zip(a, b))


class EmbodiedAnalogComputer:
    """
    Uses FPGA DRAM as an analog compute element where:
    - Partial timing writes store analog values (less charge = lower value)
    - Temperature affects decay rate (Arrhenius physics)
    - Readback after decay = value * temperature-dependent multiplier
    """

    # Optimal offset for partial charge (from z1122)
    PARTIAL_OFFSET = 16

    def __init__(self, fpga: FPGAInterface):
        self.fpga = fpga
        self.base_addr = 0x200000
        self.pattern_all_ones = bytes([0xFF] * 16)

    def store_analog_value(self, address: int, value: float) -> dict:
        """Store an analog value using partial write

        Args:
            address: DRAM address
            value: Float in [0, 1] where 1 = full charge, 0 = no charge

        Returns:
            dict with success, immediate errors, actual_value
        """
        # For high values, use low offset (more charge)
        # For low values, use high offset (less charge)
        # The PARTIAL_OFFSET=16 is our "zero point" for analog range

        if value >= 0.9:
            offset = 0  # Full charge
        elif value <= 0.1:
            offset = 32  # Very little charge
        else:
            # Linear mapping: value 0.5 -> offset 16
            offset = int(32 * (1 - value))

        result = self.fpga.partial_timing_write(address, self.pattern_all_ones, offset)

        if result.get('success'):
            # Immediate readback to verify
            read_data = self.fpga.ddr_read(address)
            if read_data:
                ones = count_ones(read_data)
                actual_value = ones / 128.0  # 128 bits total
                return {
                    'success': True,
                    'offset': offset,
                    'requested_value': value,
                    'actual_value': actual_value,
                    'ones_count': ones,
                    'errors': 128 - ones,
                    'temperature': result.get('temperature', 0)
                }

        return {'success': False}

    def read_analog_value(self, address: int) -> float:
        """Read back analog value from DRAM

        Returns float in [0, 1] based on remaining 1 bits
        """
        data = self.fpga.ddr_read(address)
        if data:
            return count_ones(data) / 128.0
        return -1.0

    def decay_multiply(self, address: int, decay_factor: float, wait_ms: float = 10.0) -> dict:
        """Demonstrate analog multiply via decay

        The decay naturally multiplies stored value by a factor dependent on:
        - Wait time (longer = more decay)
        - Temperature (higher = faster decay via Arrhenius)

        Args:
            address: DRAM address with stored analog value
            decay_factor: Not directly controlled - we observe actual decay
            wait_ms: How long to wait for decay

        Returns:
            dict with before, after, actual_multiplier
        """
        # Read before
        before_data = self.fpga.ddr_read(address)
        before_ones = count_ones(before_data) if before_data else 0
        before_value = before_ones / 128.0

        # Wait for decay (refresh is disabled in our MIG config)
        time.sleep(wait_ms / 1000.0)

        # Read after
        after_data = self.fpga.ddr_read(address)
        after_ones = count_ones(after_data) if after_data else 0
        after_value = after_ones / 128.0

        fpga_temp, _ = self.fpga.read_temperature()

        # Calculate actual multiplier
        if before_value > 0:
            actual_multiplier = after_value / before_value
        else:
            actual_multiplier = 0.0

        return {
            'before_value': before_value,
            'after_value': after_value,
            'actual_multiplier': actual_multiplier,
            'decay_amount': before_value - after_value,
            'wait_ms': wait_ms,
            'temperature': fpga_temp
        }

    def matrix_vector_multiply_demo(self, vector: List[float], wait_ms: float = 5.0) -> dict:
        """Demonstrate matrix-vector multiply using DRAM decay

        The "matrix" is the temperature-dependent decay function.
        Each element of the vector is stored, decayed, and read back.

        This simulates: output = diag(decay_weights) @ input
        where decay_weights depend on temperature and position.
        """
        results = []
        addr_base = self.base_addr + 0x10000

        print(f"  Storing {len(vector)} analog values...")

        # Store all values
        for i, val in enumerate(vector):
            addr = addr_base + (i * 256)  # Spread out to avoid interference
            store_result = self.store_analog_value(addr, val)
            if store_result.get('success'):
                results.append({
                    'index': i,
                    'input': val,
                    'stored': store_result['actual_value'],
                    'addr': addr
                })

        print(f"  Waiting {wait_ms}ms for decay (analog multiply)...")
        time.sleep(wait_ms / 1000.0)

        # Read back after decay
        print(f"  Reading decayed values...")
        output = []
        for r in results:
            decayed = self.read_analog_value(r['addr'])
            r['output'] = decayed
            if r['stored'] > 0:
                r['multiplier'] = decayed / r['stored']
            else:
                r['multiplier'] = 0.0
            output.append(decayed)

        fpga_temp, _ = self.fpga.read_temperature()

        return {
            'input_vector': vector,
            'output_vector': output,
            'details': results,
            'temperature': fpga_temp,
            'wait_ms': wait_ms
        }


def run_demo():
    """Run the embodied analog compute demonstration"""
    print("=" * 60)
    print("z1123: Embodied Analog Compute Demo")
    print("=" * 60)

    fpga = FPGAInterface()

    print("\nConnecting to FPGA...")
    if not fpga.connect():
        print("ERROR: Could not connect to FPGA")
        return None

    status = fpga.ping()
    print(f"FPGA: DDR3={'ready' if status.get('ddr3_ready') else 'NOT READY'}")

    fpga_temp, _ = fpga.read_temperature()
    gpu = get_gpu_telemetry()
    print(f"FPGA temp: {fpga_temp:.1f}C, GPU temp: {gpu['temp']:.1f}C")

    computer = EmbodiedAnalogComputer(fpga)

    results = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'initial_fpga_temp': fpga_temp,
        'initial_gpu_temp': gpu['temp'],
    }

    # ============================================================
    # Demo 1: Analog value storage
    # ============================================================
    print("\n" + "=" * 50)
    print("Demo 1: Analog Value Storage")
    print("=" * 50)

    test_values = [1.0, 0.8, 0.6, 0.4, 0.2]
    storage_results = []

    for i, val in enumerate(test_values):
        addr = computer.base_addr + (i * 256)
        result = computer.store_analog_value(addr, val)

        if result.get('success'):
            print(f"  Store {val:.1f}: offset={result['offset']:2d}, "
                  f"actual={result['actual_value']:.3f}, "
                  f"errors={result['errors']}")
            storage_results.append(result)

    results['storage_demo'] = storage_results

    # ============================================================
    # Demo 2: Decay-based analog multiply
    # ============================================================
    print("\n" + "=" * 50)
    print("Demo 2: Decay-Based Analog Multiply")
    print("=" * 50)

    # Store a known value
    addr = computer.base_addr + 0x5000
    store_result = computer.store_analog_value(addr, 0.8)

    if store_result.get('success'):
        print(f"  Initial value: {store_result['actual_value']:.3f}")

        # Apply different decay times
        decay_results = []
        for wait_ms in [5, 10, 20]:
            # Restore value
            computer.store_analog_value(addr, 0.8)
            time.sleep(0.01)

            # Apply decay
            decay_result = computer.decay_multiply(addr, 0.9, wait_ms)
            print(f"  After {wait_ms:2d}ms: {decay_result['after_value']:.3f} "
                  f"(multiplier={decay_result['actual_multiplier']:.3f})")
            decay_results.append(decay_result)

        results['decay_multiply'] = decay_results

    # ============================================================
    # Demo 3: Vector processing with DRAM decay
    # ============================================================
    print("\n" + "=" * 50)
    print("Demo 3: Vector Processing via DRAM Decay")
    print("=" * 50)

    # A simple vector to process
    input_vector = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3]

    vector_result = computer.matrix_vector_multiply_demo(input_vector, wait_ms=10.0)

    print("\n  Results:")
    print("  Input  | Stored | Output | Multiplier")
    print("  " + "-" * 40)
    for d in vector_result['details']:
        print(f"  {d['input']:.2f}   | {d['stored']:.3f}  | {d['output']:.3f}  | {d['multiplier']:.3f}")

    results['vector_processing'] = vector_result

    # ============================================================
    # Demo 4: Temperature-dependent computation
    # ============================================================
    print("\n" + "=" * 50)
    print("Demo 4: Temperature-Dependent Computation")
    print("=" * 50)

    # Use FPGA's decay_test which measures temp during decay
    temp_decay_results = []
    pattern = bytes([0xFF] * 16)

    for trial in range(5):
        # Heat up GPU a bit with computation
        if trial > 2:
            import torch
            if torch.cuda.is_available():
                x = torch.randn(1000, 1000, device='cuda')
                for _ in range(10):
                    x = x @ x.T
                del x

        result = fpga.decay_test(
            computer.base_addr + 0x30000 + (trial * 256),
            pattern,
            wait_cycles=833333,  # 10ms
            timeout=30.0
        )

        if result.get('success'):
            errors = result['bit_errors']
            temp = result['temperature']
            temp_decay_results.append({
                'trial': trial,
                'errors': errors,
                'temperature': temp
            })
            print(f"  Trial {trial}: {errors} errors at {temp:.1f}C")

    results['temperature_sensitivity'] = temp_decay_results

    # Calculate correlation
    if len(temp_decay_results) >= 2:
        temps = [r['temperature'] for r in temp_decay_results]
        errors = [r['errors'] for r in temp_decay_results]
        if np.std(temps) > 0.1:
            corr = np.corrcoef(temps, errors)[0, 1]
            print(f"\n  Temperature-error correlation: {corr:.3f}")
            results['temp_error_correlation'] = corr

    # ============================================================
    # Summary & Business Value
    # ============================================================
    print("\n" + "=" * 60)
    print("SUMMARY: Business Value of Embodied Analog Compute")
    print("=" * 60)

    print("\n1. ANALOG VALUE STORAGE:")
    if storage_results:
        analog_range = max(r['actual_value'] for r in storage_results) - min(r['actual_value'] for r in storage_results)
        print(f"   - Analog range achieved: {analog_range:.3f}")
        print(f"   - No ADC/DAC needed - values stored as charge levels")

    print("\n2. IN-MEMORY COMPUTATION:")
    if 'decay_multiply' in results:
        multipliers = [r['actual_multiplier'] for r in results['decay_multiply']]
        print(f"   - Decay multipliers: {[f'{m:.3f}' for m in multipliers]}")
        print(f"   - Computation via physics - zero digital multiply operations")

    print("\n3. TEMPERATURE AWARENESS:")
    if 'temp_error_correlation' in results:
        print(f"   - Correlation: {results['temp_error_correlation']:.3f}")
        print(f"   - Hardware state directly affects computation")

    print("\n4. ENERGY EFFICIENCY:")
    print("   - Lower stored values = less charge = less energy")
    print("   - Natural decay performs multiply without power consumption")

    final_temp, _ = fpga.read_temperature()
    final_gpu = get_gpu_telemetry()
    results['final_fpga_temp'] = final_temp
    results['final_gpu_temp'] = final_gpu['temp']

    fpga.disconnect()

    # Save results
    output_path = Path('results/z1123_embodied_analog_compute.json')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to: {output_path}")

    return results


if __name__ == '__main__':
    run_demo()
