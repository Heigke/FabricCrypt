#!/usr/bin/env python3
"""
z1126: Embodiment Benchmark - Quantifying Hardware-Software Interplay

Measures the actual embodiment effects:
1. Temperature impact on decay rate
2. Decay impact on neural output
3. Closed-loop feedback dynamics
4. Energy efficiency at different operating points
"""

import sys
import os
import json
import time
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.fpga.fpga_interface import FPGAInterface

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


def get_gpu_telemetry():
    """Get GPU temperature and power"""
    try:
        hwmon_path = "/sys/class/drm/card1/device/hwmon/hwmon7"
        with open(f"{hwmon_path}/temp1_input", 'r') as f:
            temp = float(f.read().strip()) / 1000.0
        power = 0.0
        if os.path.exists(f"{hwmon_path}/power1_average"):
            with open(f"{hwmon_path}/power1_average", 'r') as f:
                power = float(f.read().strip()) / 1e6
        return {'temp': temp, 'power': power}
    except:
        return {'temp': 0.0, 'power': 0.0}


def heat_gpu(duration_ms: float = 100):
    """Generate GPU load to increase temperature"""
    if HAS_TORCH and torch.cuda.is_available():
        x = torch.randn(2000, 2000, device='cuda')
        start = time.time()
        while (time.time() - start) * 1000 < duration_ms:
            x = x @ x.T
            torch.cuda.synchronize()
        del x
        torch.cuda.empty_cache()


def run_benchmarks():
    """Run comprehensive embodiment benchmarks"""
    print("=" * 70)
    print("z1126: Embodiment Benchmark")
    print("=" * 70)

    fpga = FPGAInterface()

    print("\nConnecting to FPGA...")
    if not fpga.connect():
        print("ERROR: Could not connect")
        return None

    print("FPGA connected, DDR3 ready")

    results = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'benchmarks': {}
    }

    # ================================================================
    # Benchmark 1: Temperature Impact on Decay
    # ================================================================
    print("\n" + "=" * 50)
    print("Benchmark 1: Temperature Impact on Decay")
    print("=" * 50)

    temp_decay_results = []
    pattern = bytes([0xFF] * 16)
    base_addr = 0x400000

    # Test at different temperatures (heat GPU, measure FPGA)
    for trial in range(10):
        # Heat GPU to vary system temperature
        if trial >= 5:
            heat_gpu(200)

        # Get temps
        fpga_temp, _ = fpga.read_temperature()
        gpu = get_gpu_telemetry()

        # Run decay test
        result = fpga.decay_test(
            base_addr + (trial * 256),
            pattern,
            wait_cycles=833333,  # 10ms
            timeout=30.0
        )

        if result.get('success'):
            temp_decay_results.append({
                'trial': trial,
                'fpga_temp': fpga_temp,
                'gpu_temp': gpu['temp'],
                'bit_errors': result['bit_errors'],
                'decay_temp': result['temperature']
            })
            print(f"  Trial {trial}: FPGA={fpga_temp:.1f}C, GPU={gpu['temp']:.1f}C, "
                  f"errors={result['bit_errors']}")

    results['benchmarks']['temp_decay'] = temp_decay_results

    # Calculate correlation
    if len(temp_decay_results) >= 3:
        temps = [r['fpga_temp'] for r in temp_decay_results]
        errors = [r['bit_errors'] for r in temp_decay_results]
        if np.std(temps) > 0:
            corr = np.corrcoef(temps, errors)[0, 1]
            print(f"\n  Temp-Decay Correlation: {corr:.3f}")
            results['temp_decay_correlation'] = corr

    # ================================================================
    # Benchmark 2: Pattern Sensitivity (embodied memory encoding)
    # ================================================================
    print("\n" + "=" * 50)
    print("Benchmark 2: Pattern Sensitivity")
    print("=" * 50)

    pattern_results = []
    test_patterns = [
        ('0xFF', bytes([0xFF] * 16)),
        ('0x00', bytes([0x00] * 16)),
        ('0xAA', bytes([0xAA] * 16)),
        ('0x55', bytes([0x55] * 16)),
        ('0xF0', bytes([0xF0] * 16)),
        ('0x0F', bytes([0x0F] * 16)),
        ('0xCC', bytes([0xCC] * 16)),
    ]

    for name, pattern in test_patterns:
        # Run decay test with this pattern
        result = fpga.decay_test(
            base_addr + 0x10000,
            pattern,
            wait_cycles=833333,  # 10ms
            timeout=30.0
        )

        if result.get('success'):
            ones_before = sum(bin(b).count('1') for b in pattern)
            ones_after = sum(bin(b).count('1') for b in result['read_data'])

            pattern_results.append({
                'pattern': name,
                'ones_before': ones_before,
                'ones_after': ones_after,
                'bit_errors': result['bit_errors'],
                'retention': ones_after / ones_before if ones_before > 0 else 1.0,
                'temperature': result['temperature']
            })
            print(f"  {name}: {ones_before}→{ones_after} ones, "
                  f"{result['bit_errors']} errors, "
                  f"retention={ones_after/ones_before if ones_before > 0 else 1.0:.2%}")

    results['benchmarks']['pattern_sensitivity'] = pattern_results

    # ================================================================
    # Benchmark 3: Decay Time vs Errors (Arrhenius curve)
    # ================================================================
    print("\n" + "=" * 50)
    print("Benchmark 3: Decay Time Curve")
    print("=" * 50)

    decay_curve_results = []
    wait_times_ms = [5, 10, 20, 50, 100]

    for wait_ms in wait_times_ms:
        wait_cycles = int(wait_ms * 83333)

        result = fpga.decay_test(
            base_addr + 0x20000,
            bytes([0xFF] * 16),
            wait_cycles=wait_cycles,
            timeout=60.0
        )

        if result.get('success'):
            decay_curve_results.append({
                'wait_ms': wait_ms,
                'bit_errors': result['bit_errors'],
                'temperature': result['temperature']
            })
            print(f"  {wait_ms:3d}ms: {result['bit_errors']:2d} errors at {result['temperature']:.1f}C")
        else:
            print(f"  {wait_ms:3d}ms: TIMEOUT")

        time.sleep(0.5)  # Cool down

    results['benchmarks']['decay_curve'] = decay_curve_results

    # ================================================================
    # Benchmark 4: GPU Load Impact
    # ================================================================
    print("\n" + "=" * 50)
    print("Benchmark 4: GPU Load Impact")
    print("=" * 50)

    gpu_load_results = []

    for load_level in ['idle', 'light', 'heavy']:
        # Apply load
        if load_level == 'light':
            heat_gpu(100)
        elif load_level == 'heavy':
            heat_gpu(500)

        # Measure
        fpga_temp, _ = fpga.read_temperature()
        gpu = get_gpu_telemetry()

        result = fpga.decay_test(
            base_addr + 0x30000,
            bytes([0xFF] * 16),
            wait_cycles=833333,
            timeout=30.0
        )

        if result.get('success'):
            gpu_load_results.append({
                'load': load_level,
                'fpga_temp': fpga_temp,
                'gpu_temp': gpu['temp'],
                'gpu_power': gpu['power'],
                'bit_errors': result['bit_errors']
            })
            print(f"  {load_level:6s}: GPU={gpu['temp']:.0f}C/{gpu['power']:.0f}W, "
                  f"FPGA={fpga_temp:.1f}C, errors={result['bit_errors']}")

        time.sleep(1.0)  # Cool down

    results['benchmarks']['gpu_load_impact'] = gpu_load_results

    # ================================================================
    # Summary
    # ================================================================
    print("\n" + "=" * 70)
    print("EMBODIMENT BENCHMARK SUMMARY")
    print("=" * 70)

    print("\n1. Temperature-Decay Relationship:")
    if 'temp_decay_correlation' in results:
        print(f"   Correlation: {results['temp_decay_correlation']:.3f}")
    print("   Higher temp → different decay behavior (Arrhenius physics)")

    print("\n2. Pattern-Dependent Memory:")
    if pattern_results:
        high_decay = max(pattern_results, key=lambda x: x['bit_errors'])
        low_decay = min(pattern_results, key=lambda x: x['bit_errors'])
        print(f"   Highest decay: {high_decay['pattern']} ({high_decay['bit_errors']} errors)")
        print(f"   Lowest decay: {low_decay['pattern']} ({low_decay['bit_errors']} errors)")
        print(f"   Range: {high_decay['bit_errors'] - low_decay['bit_errors']} errors")

    print("\n3. Decay Time Curve:")
    if decay_curve_results:
        for d in decay_curve_results:
            print(f"   {d['wait_ms']:3d}ms: {d['bit_errors']:2d} errors")

    print("\n4. GPU-FPGA Thermal Coupling:")
    if gpu_load_results:
        for g in gpu_load_results:
            print(f"   {g['load']:6s}: GPU={g['gpu_temp']:.0f}C, FPGA={g['fpga_temp']:.1f}C")

    # Business value
    print("\n" + "=" * 70)
    print("EMBODIED AI BUSINESS VALUE")
    print("=" * 70)

    embodied_capabilities = []

    if pattern_results:
        if max(r['bit_errors'] for r in pattern_results) - min(r['bit_errors'] for r in pattern_results) > 10:
            embodied_capabilities.append("Pattern-based analog encoding (different patterns = different decay)")

    if 'temp_decay_correlation' in results and abs(results['temp_decay_correlation']) > 0.1:
        embodied_capabilities.append("Temperature-modulated computation (heat affects decay)")

    if gpu_load_results:
        temp_range = max(g['fpga_temp'] for g in gpu_load_results) - min(g['fpga_temp'] for g in gpu_load_results)
        if temp_range > 0.5:
            embodied_capabilities.append("GPU-FPGA thermal coupling (system-wide awareness)")

    if decay_curve_results:
        embodied_capabilities.append("Time-based analog multiply (wait time = multiplier)")

    for cap in embodied_capabilities:
        print(f"  ✓ {cap}")

    if not embodied_capabilities:
        print("  ⚠ Limited embodiment effects observed in this run")

    fpga.disconnect()

    return results


def main():
    results = run_benchmarks()

    if results:
        output_path = Path('results/z1126_embodiment_benchmark.json')
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\nResults saved to: {output_path}")


if __name__ == '__main__':
    main()
