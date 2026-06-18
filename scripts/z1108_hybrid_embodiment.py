#!/usr/bin/env python3
"""
z1108: Hybrid GPU-FPGA-DRAM Embodiment

Explores combining:
1. GPU (ikaros) for high-level computation
2. FPGA (Arty A7) for timing control + temperature sensing
3. DRAM for analog computation (charge decay)

Phase 1: Characterize what timing effects we can observe
Phase 2: FPGA integration
Phase 3: True analog compute with DRAM
"""

import sys
import os
import json
import time
import struct
import mmap
import ctypes
from datetime import datetime
from typing import List, Tuple, Optional
import statistics

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Try to import GPU telemetry
try:
    from src.telemetry.sysfs_hwmon import SysfsHwmonTelemetry
    HAS_GPU = True
except ImportError:
    HAS_GPU = False
    print("Warning: GPU telemetry not available")

# Try serial
try:
    import serial
    HAS_SERIAL = True
except ImportError:
    HAS_SERIAL = False


def measure_memory_timing(n_trials: int = 1000) -> dict:
    """
    Measure memory access timing variations.

    While we can't disable DRAM refresh, timing variations
    still reveal some hardware behavior.
    """
    result = {
        'n_trials': n_trials,
        'write_times_ns': [],
        'read_times_ns': [],
        'round_trip_ns': []
    }

    # Allocate a chunk of memory
    size = 1024 * 1024  # 1MB
    buf = bytearray(size)
    view = memoryview(buf)

    # Pattern to write
    pattern = bytes([0xAA, 0x55] * 512)

    write_times = []
    read_times = []
    rt_times = []

    for i in range(n_trials):
        offset = (i * 1024) % (size - 1024)

        # Write timing
        start = time.perf_counter_ns()
        view[offset:offset+1024] = pattern
        write_end = time.perf_counter_ns()

        # Read timing
        _ = bytes(view[offset:offset+1024])
        read_end = time.perf_counter_ns()

        write_times.append(write_end - start)
        read_times.append(read_end - write_end)
        rt_times.append(read_end - start)

    result['write_times_ns'] = write_times
    result['read_times_ns'] = read_times
    result['round_trip_ns'] = rt_times

    result['write_mean_ns'] = statistics.mean(write_times)
    result['write_std_ns'] = statistics.stdev(write_times)
    result['read_mean_ns'] = statistics.mean(read_times)
    result['read_std_ns'] = statistics.stdev(read_times)
    result['rt_cv_pct'] = 100 * statistics.stdev(rt_times) / statistics.mean(rt_times)

    return result


def measure_memory_vs_gpu_temp(n_samples: int = 100) -> dict:
    """
    Measure if memory timing correlates with GPU temperature.

    This tests thermal coupling between components.
    """
    result = {
        'n_samples': n_samples,
        'gpu_temps': [],
        'mem_times_ns': [],
        'correlation': None
    }

    if not HAS_GPU:
        result['error'] = 'No GPU telemetry'
        return result

    telem = SysfsHwmonTelemetry()

    # Allocate memory
    buf = bytearray(1024 * 1024)
    view = memoryview(buf)
    pattern = bytes([0xAA] * 1024)

    temps = []
    times = []

    for i in range(n_samples):
        # Get GPU temp
        readings = telem.read_sample()
        temp = readings.temp_edge_c if readings else 0

        # Memory operation
        offset = (i * 1024) % (1024 * 1024 - 1024)
        start = time.perf_counter_ns()
        view[offset:offset+1024] = pattern
        _ = bytes(view[offset:offset+1024])
        elapsed = time.perf_counter_ns() - start

        temps.append(temp)
        times.append(elapsed)

        time.sleep(0.05)  # 50ms between samples

    result['gpu_temps'] = temps
    result['mem_times_ns'] = times

    # Calculate correlation
    if len(temps) > 2 and len(set(temps)) > 1:
        # Simple Pearson correlation
        n = len(temps)
        mean_t = sum(temps) / n
        mean_m = sum(times) / n

        num = sum((t - mean_t) * (m - mean_m) for t, m in zip(temps, times))
        den_t = sum((t - mean_t) ** 2 for t in temps) ** 0.5
        den_m = sum((m - mean_m) ** 2 for m in times) ** 0.5

        if den_t > 0 and den_m > 0:
            result['correlation'] = num / (den_t * den_m)
        else:
            result['correlation'] = 0.0
    else:
        result['correlation'] = 0.0

    result['temp_range'] = (min(temps), max(temps)) if temps else (0, 0)
    result['time_cv_pct'] = 100 * statistics.stdev(times) / statistics.mean(times) if times else 0

    return result


def probe_fpga_connection() -> dict:
    """Check if FPGA is available."""
    result = {
        'connected': False,
        'port': None
    }

    if not HAS_SERIAL:
        result['error'] = 'pyserial not installed'
        return result

    import serial.tools.list_ports

    for port in serial.tools.list_ports.comports():
        if '0403:6010' in port.hwid or 'Digilent' in port.description:
            result['port'] = port.device
            result['connected'] = True
            break

    return result


def test_gpu_fpga_roundtrip() -> dict:
    """
    Test GPU -> FPGA -> GPU round trip.

    This simulates the hybrid architecture where tensors
    pass through FPGA for embodied processing.
    """
    result = {
        'gpu_available': False,
        'fpga_available': False,
        'roundtrip_possible': False
    }

    # Check GPU
    try:
        import torch
        if torch.cuda.is_available() or torch.backends.mps.is_available():
            result['gpu_available'] = True
            result['gpu_device'] = 'cuda' if torch.cuda.is_available() else 'mps'

            # Try ROCm
            if hasattr(torch.version, 'hip') and torch.version.hip:
                result['gpu_device'] = 'rocm'
    except ImportError:
        pass

    # Check FPGA
    fpga = probe_fpga_connection()
    result['fpga_available'] = fpga['connected']
    result['fpga_port'] = fpga.get('port')

    result['roundtrip_possible'] = result['gpu_available'] and result['fpga_available']

    if result['roundtrip_possible']:
        result['architecture'] = 'HYBRID: GPU <-> FPGA'
        result['next_step'] = 'Create custom bitstream with vector protocol'
    else:
        missing = []
        if not result['gpu_available']:
            missing.append('GPU')
        if not result['fpga_available']:
            missing.append('FPGA')
        result['missing'] = missing

    return result


def generate_analog_compute_plan() -> dict:
    """
    Generate plan for analog DRAM computation.
    """
    return {
        'goal': 'Use DRAM charge decay as analog computation',

        'approach_1_fpga_ddr': {
            'description': 'Control DDR3 on Arty A7 at timing level',
            'pros': ['Full timing control', 'Can disable refresh', 'Direct temperature correlation'],
            'cons': ['Limited DDR3 size (256MB)', 'Complex controller', 'FPGA resources'],
            'status': 'Requires custom bitstream'
        },

        'approach_2_host_dram': {
            'description': 'Use host LPDDR5 with kernel module',
            'pros': ['Large memory (32GB)', 'Fast', 'Already present'],
            'cons': ['Need kernel module', 'OS interference', 'Risk of instability'],
            'status': 'Requires privileged access'
        },

        'approach_3_fpga_sram': {
            'description': 'Use FPGA block RAM as pseudo-analog',
            'pros': ['Simple', 'Fast', 'No external memory'],
            'cons': ['Not truly analog', 'Limited size', 'Less embodied'],
            'status': 'Easiest to implement'
        },

        'recommended': 'Start with approach_3_fpga_sram, then move to approach_1_fpga_ddr',

        'key_insight': '''
        True embodiment with DRAM:
        - Charge decay rate follows Arrhenius: τ ∝ exp(Ea/kT)
        - Higher temperature = faster decay = different computation
        - Manufacturing variation = unique per-cell behavior
        - This is PHYSICS, not abstraction
        '''
    }


def main():
    print("=" * 60)
    print("Z1108: Hybrid GPU-FPGA-DRAM Embodiment")
    print("=" * 60)
    print()

    results = {
        'experiment': 'z1108_hybrid_embodiment',
        'timestamp': datetime.now().isoformat(),
        'findings': {}
    }

    # Step 1: Memory timing baseline
    print("Step 1: Memory timing characterization...")
    mem_timing = measure_memory_timing(n_trials=500)
    results['findings']['memory_timing'] = {
        'write_mean_ns': mem_timing['write_mean_ns'],
        'write_std_ns': mem_timing['write_std_ns'],
        'read_mean_ns': mem_timing['read_mean_ns'],
        'rt_cv_pct': mem_timing['rt_cv_pct']
    }
    print(f"  Write: {mem_timing['write_mean_ns']:.0f} ± {mem_timing['write_std_ns']:.0f} ns")
    print(f"  Read: {mem_timing['read_mean_ns']:.0f} ns")
    print(f"  Round-trip CV: {mem_timing['rt_cv_pct']:.2f}%")
    print()

    # Step 2: Memory-temperature correlation
    print("Step 2: Memory timing vs GPU temperature...")
    if HAS_GPU:
        mem_temp = measure_memory_vs_gpu_temp(n_samples=50)
        results['findings']['memory_temp_correlation'] = {
            'correlation': mem_temp.get('correlation', 0),
            'temp_range': mem_temp.get('temp_range', (0, 0)),
            'time_cv_pct': mem_temp.get('time_cv_pct', 0)
        }
        print(f"  Temp range: {mem_temp['temp_range'][0]:.1f}°C - {mem_temp['temp_range'][1]:.1f}°C")
        print(f"  Timing CV: {mem_temp['time_cv_pct']:.2f}%")
        print(f"  Correlation (timing vs temp): {mem_temp.get('correlation', 0):.4f}")

        if abs(mem_temp.get('correlation', 0)) > 0.3:
            print("  → Some thermal coupling detected!")
        else:
            print("  → Minimal thermal coupling (expected with OS buffering)")
    else:
        print("  Skipped (no GPU telemetry)")
        results['findings']['memory_temp_correlation'] = {'error': 'No GPU'}
    print()

    # Step 3: Check hybrid components
    print("Step 3: Checking hybrid architecture components...")
    hybrid = test_gpu_fpga_roundtrip()
    results['findings']['hybrid_check'] = hybrid

    print(f"  GPU available: {hybrid['gpu_available']} ({hybrid.get('gpu_device', 'N/A')})")
    print(f"  FPGA available: {hybrid['fpga_available']} ({hybrid.get('fpga_port', 'N/A')})")
    print(f"  Roundtrip possible: {hybrid['roundtrip_possible']}")
    print()

    # Step 4: Analog compute plan
    print("Step 4: Analog DRAM computation plan...")
    plan = generate_analog_compute_plan()
    results['findings']['analog_plan'] = plan

    print(f"  Recommended approach: {plan['recommended']}")
    print()

    for approach, details in plan.items():
        if isinstance(details, dict) and 'description' in details:
            print(f"  {approach}:")
            print(f"    {details['description']}")
            print(f"    Status: {details['status']}")
            print()

    # Summary
    print("=" * 60)
    print("Summary: Path to True Embodiment")
    print("=" * 60)
    print()

    print("Current timing observations:")
    print(f"  - Memory timing CV: {mem_timing['rt_cv_pct']:.2f}%")
    if HAS_GPU and 'correlation' in mem_temp:
        print(f"  - Temp correlation: {mem_temp['correlation']:.4f}")
    print()

    print("Why DRAM analog compute matters:")
    print("  1. Charge decay is PHYSICS (Arrhenius equation)")
    print("  2. Temperature directly affects decay rate")
    print("  3. Manufacturing variation = unique per-cell behavior")
    print("  4. No abstraction layer - it's truly analog")
    print()

    print("Next steps for hybrid embodiment:")
    print("  1. Create FPGA bitstream with XADC temperature + DDR3 controller")
    print("  2. Disable DDR3 auto-refresh on FPGA")
    print("  3. Measure decay curves at different temperatures")
    print("  4. Integrate with GPU pipeline for hybrid inference")
    print()

    if hybrid['roundtrip_possible']:
        print("✓ Both GPU and FPGA available - ready for hybrid experiments!")
    else:
        print(f"⚠ Missing: {', '.join(hybrid.get('missing', ['unknown']))}")

    # Save results
    results_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'results',
        'z1108_hybrid_embodiment.json'
    )

    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\nResults saved to: {results_path}")

    return results


if __name__ == '__main__':
    main()
