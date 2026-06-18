#!/usr/bin/env python3
"""
z1115: Comprehensive FPGA Validation Suite

Thorough testing of DDR3 decay and partial timing writes before GPU integration.

Tests:
1. Basic read/write reliability (100 iterations)
2. Decay vs wait time (logarithmic sweep)
3. Partial timing offset sweep (0-63)
4. Partial timing + decay combination
5. Temperature correlation
6. Repeatability (same conditions, multiple runs)
7. Pattern sensitivity (all 1s, all 0s, checkerboard)
"""

import sys
import time
import json
import statistics
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.fpga.fpga_interface import FPGAInterface


def test_basic_reliability(fpga: FPGAInterface, iterations: int = 100) -> dict:
    """Test 1: Basic read/write reliability"""
    print("\n" + "="*60)
    print("TEST 1: Basic Read/Write Reliability")
    print("="*60)

    patterns = [
        bytes([0xFF] * 16),
        bytes([0x00] * 16),
        bytes([0xAA] * 16),
        bytes([0x55] * 16),
        bytes([0x12, 0x34, 0x56, 0x78] * 4),
        bytes(range(16)),
    ]

    results = {'total': 0, 'passed': 0, 'failed': 0, 'patterns': {}}
    base_addr = 0x100000

    for pattern in patterns:
        pattern_name = pattern[:4].hex()
        errors = 0

        for i in range(iterations):
            addr = base_addr + i * 0x100
            fpga.ddr_write(addr, pattern)
            readback = fpga.ddr_read(addr)

            if readback != pattern:
                errors += 1

        results['patterns'][pattern_name] = {
            'iterations': iterations,
            'errors': errors,
            'error_rate': errors / iterations
        }
        results['total'] += iterations
        results['passed'] += iterations - errors
        results['failed'] += errors

        status = "PASS" if errors == 0 else f"FAIL ({errors} errors)"
        print(f"  Pattern {pattern_name}: {status}")

    results['overall_error_rate'] = results['failed'] / results['total']
    print(f"\nOverall: {results['passed']}/{results['total']} passed "
          f"({results['overall_error_rate']*100:.2f}% error rate)")

    return results


def test_decay_vs_time(fpga: FPGAInterface) -> dict:
    """Test 2: Decay vs wait time"""
    print("\n" + "="*60)
    print("TEST 2: Decay vs Wait Time")
    print("="*60)

    # Logarithmic sweep of wait times (in ui_clk cycles, ~12ns each)
    # 1K cycles = ~12us, 1M cycles = ~12ms, 100M cycles = ~1.2s
    wait_times = [
        1000,       # ~12us
        10000,      # ~120us
        100000,     # ~1.2ms
        1000000,    # ~12ms
        10000000,   # ~120ms
        50000000,   # ~600ms
        100000000,  # ~1.2s
    ]

    results = []
    pattern = bytes([0xFF] * 16)
    base_addr = 0x200000

    for i, wait in enumerate(wait_times):
        addr = base_addr + i * 0x10000
        wait_ms = wait * 12e-6  # Convert to ms

        print(f"\n  Wait: {wait:,} cycles (~{wait_ms:.1f}ms)...")

        decay_result = fpga.decay_test(addr, pattern, wait_cycles=wait)

        if decay_result.get('success'):
            bit_errors = decay_result['bit_errors']
            temp = decay_result['temperature']

            results.append({
                'wait_cycles': wait,
                'wait_ms': wait_ms,
                'bit_errors': bit_errors,
                'error_rate': bit_errors / 128,
                'temperature': temp
            })

            print(f"    Bit errors: {bit_errors}/128 ({bit_errors/128*100:.1f}%)")
            print(f"    Temperature: {temp:.1f}C")
        else:
            print(f"    FAILED")
            results.append({'wait_cycles': wait, 'success': False})

    return results


def test_partial_timing_sweep(fpga: FPGAInterface) -> dict:
    """Test 3: Partial timing offset sweep"""
    print("\n" + "="*60)
    print("TEST 3: Partial Timing Offset Sweep")
    print("="*60)

    offsets = list(range(0, 64, 4)) + [63]  # 0, 4, 8, ..., 60, 63
    offsets = sorted(set(offsets))

    results = []
    pattern = bytes([0xFF] * 16)
    base_addr = 0x300000

    for i, offset in enumerate(offsets):
        addr = base_addr + i * 0x10000
        timing_ps = offset * 12

        print(f"\n  Offset {offset} (~{timing_ps}ps)...", end=" ", flush=True)

        result = fpga.partial_timing_write(addr, pattern, timing_offset=offset, timeout=10)

        if result.get('success'):
            written = result['written_data']
            readback = result['read_data']
            immediate_errors = sum(bin(a ^ b).count('1') for a, b in zip(written, readback))
            temp = result['temperature']

            results.append({
                'offset': offset,
                'timing_ps': timing_ps,
                'immediate_errors': immediate_errors,
                'temperature': temp,
                'success': True
            })

            print(f"Immediate errors: {immediate_errors}, Temp: {temp:.1f}C")
        else:
            print("FAILED")
            results.append({'offset': offset, 'success': False})

        time.sleep(0.2)  # Small delay between tests

    return results


def test_partial_timing_with_decay(fpga: FPGAInterface) -> dict:
    """Test 4: Partial timing + decay combination"""
    print("\n" + "="*60)
    print("TEST 4: Partial Timing + Decay Combination")
    print("="*60)

    offsets = [0, 20, 40, 55, 63]
    decay_wait_s = 5.0  # 5 second decay

    results = []
    pattern = bytes([0xFF] * 16)
    base_addr = 0x400000

    for i, offset in enumerate(offsets):
        addr = base_addr + i * 0x10000

        print(f"\n  Offset {offset} + {decay_wait_s}s decay:")

        # Write with partial timing
        write_result = fpga.partial_timing_write(addr, pattern, timing_offset=offset, timeout=10)

        if not write_result.get('success'):
            print("    Write FAILED")
            results.append({'offset': offset, 'success': False})
            continue

        immediate_errors = sum(bin(a ^ b).count('1')
            for a, b in zip(write_result['written_data'], write_result['read_data']))

        # Wait for decay
        print(f"    Immediate errors: {immediate_errors}")
        print(f"    Waiting {decay_wait_s}s for decay...", end=" ", flush=True)
        time.sleep(decay_wait_s)

        # Read back
        readback = fpga.ddr_read(addr)

        if readback:
            decay_errors = sum(bin(a ^ b).count('1') for a, b in zip(pattern, readback))
            print(f"Done")
            print(f"    Decay errors: {decay_errors}/128 ({decay_errors/128*100:.1f}%)")

            results.append({
                'offset': offset,
                'immediate_errors': immediate_errors,
                'decay_errors': decay_errors,
                'decay_wait_s': decay_wait_s,
                'success': True
            })
        else:
            print("Read FAILED")
            results.append({'offset': offset, 'success': False})

    # Analysis
    print("\n  Summary:")
    print("  Offset | Immediate | After Decay | Delta")
    print("  -------|-----------|-------------|------")
    for r in results:
        if r.get('success'):
            delta = r['decay_errors'] - r['immediate_errors']
            print(f"    {r['offset']:3d}   |     {r['immediate_errors']:3d}   |     {r['decay_errors']:3d}     | {delta:+4d}")

    return results


def test_temperature_correlation(fpga: FPGAInterface) -> dict:
    """Test 5: Temperature correlation with decay"""
    print("\n" + "="*60)
    print("TEST 5: Temperature Monitoring During Decay")
    print("="*60)

    results = []
    pattern = bytes([0xFF] * 16)
    addr = 0x500000

    # Get initial temperature
    temp_start, _ = fpga.read_temperature()
    print(f"  Starting temperature: {temp_start:.1f}C")

    # Write data
    fpga.ddr_write(addr, pattern)

    # Monitor temperature and check decay at intervals
    check_times = [1, 2, 5, 10, 15, 20]  # seconds

    print("\n  Time(s) | Temp(C) | Bit Errors")
    print("  --------|---------|----------")

    start_time = time.time()
    last_check = 0

    for check_time in check_times:
        # Wait until check time
        while time.time() - start_time < check_time:
            time.sleep(0.1)

        # Read temperature
        temp, _ = fpga.read_temperature()

        # Read data (this will show current decay state)
        readback = fpga.ddr_read(addr)
        if readback:
            errors = sum(bin(a ^ b).count('1') for a, b in zip(pattern, readback))
        else:
            errors = -1

        results.append({
            'time_s': check_time,
            'temperature': temp,
            'bit_errors': errors
        })

        print(f"    {check_time:4d}   |  {temp:5.1f}  |    {errors:3d}")

        # Re-write for next interval (reset decay)
        fpga.ddr_write(addr, pattern)

    return results


def test_repeatability(fpga: FPGAInterface, runs: int = 5) -> dict:
    """Test 6: Repeatability of partial timing effect"""
    print("\n" + "="*60)
    print("TEST 6: Repeatability (Same Conditions, Multiple Runs)")
    print("="*60)

    offset = 50  # Use offset 50 as test case
    decay_wait_s = 5.0
    pattern = bytes([0xFF] * 16)

    results = []
    base_addr = 0x600000

    print(f"  Running {runs} identical tests: offset={offset}, decay={decay_wait_s}s")
    print("\n  Run | Immediate | Decay | Temp")
    print("  ----|-----------|-------|-----")

    for run in range(runs):
        addr = base_addr + run * 0x10000

        # Write with partial timing
        write_result = fpga.partial_timing_write(addr, pattern, timing_offset=offset, timeout=10)

        if write_result.get('success'):
            immediate = sum(bin(a ^ b).count('1')
                for a, b in zip(write_result['written_data'], write_result['read_data']))
            temp = write_result['temperature']

            # Wait for decay
            time.sleep(decay_wait_s)

            # Read back
            readback = fpga.ddr_read(addr)
            decay = sum(bin(a ^ b).count('1') for a, b in zip(pattern, readback)) if readback else -1

            results.append({
                'run': run + 1,
                'immediate_errors': immediate,
                'decay_errors': decay,
                'temperature': temp
            })

            print(f"   {run+1:2d}  |     {immediate:3d}   |  {decay:3d}  | {temp:.1f}")
        else:
            print(f"   {run+1:2d}  | FAILED")
            results.append({'run': run + 1, 'success': False})

    # Statistics
    decay_errors = [r['decay_errors'] for r in results if r.get('decay_errors', -1) >= 0]
    if decay_errors:
        mean = statistics.mean(decay_errors)
        stdev = statistics.stdev(decay_errors) if len(decay_errors) > 1 else 0
        print(f"\n  Decay errors: mean={mean:.1f}, stdev={stdev:.1f}")

    return results


def test_pattern_sensitivity(fpga: FPGAInterface) -> dict:
    """Test 7: Pattern sensitivity for decay"""
    print("\n" + "="*60)
    print("TEST 7: Pattern Sensitivity")
    print("="*60)

    patterns = {
        'all_ones': bytes([0xFF] * 16),
        'all_zeros': bytes([0x00] * 16),
        'checker_AA': bytes([0xAA] * 16),
        'checker_55': bytes([0x55] * 16),
        'alternating': bytes([0xFF, 0x00] * 8),
        'gradient': bytes(range(16)),
    }

    decay_wait_s = 5.0
    results = {}
    base_addr = 0x700000

    print(f"  Testing decay sensitivity for different patterns ({decay_wait_s}s decay)")
    print("\n  Pattern      | Initial | After Decay | Changed")
    print("  -------------|---------|-------------|--------")

    for i, (name, pattern) in enumerate(patterns.items()):
        addr = base_addr + i * 0x10000

        # Write pattern
        fpga.ddr_write(addr, pattern)

        # Wait for decay
        time.sleep(decay_wait_s)

        # Read back
        readback = fpga.ddr_read(addr)

        if readback:
            # Count bits that changed
            changed_bits = sum(bin(a ^ b).count('1') for a, b in zip(pattern, readback))
            # Count 1s in original
            ones_in_pattern = sum(bin(b).count('1') for b in pattern)

            results[name] = {
                'pattern': pattern.hex(),
                'ones_in_pattern': ones_in_pattern,
                'bits_changed': changed_bits,
                'success': True
            }

            print(f"  {name:12s} |   {ones_in_pattern:3d}   |     {changed_bits:3d}     |  {changed_bits:3d}")
        else:
            print(f"  {name:12s} | FAILED")
            results[name] = {'success': False}

    return results


def main():
    print("="*60)
    print("z1115: Comprehensive FPGA Validation Suite")
    print("="*60)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    fpga = FPGAInterface()

    if not fpga.connect():
        print("ERROR: Could not connect to FPGA")
        return 1

    print(f"\nConnected to FPGA")
    status = fpga.get_status()
    print(f"Status: {status}")

    all_results = {
        'timestamp': datetime.now().isoformat(),
        'initial_status': status,
        'tests': {}
    }

    try:
        # Run all tests
        all_results['tests']['1_basic_reliability'] = test_basic_reliability(fpga)
        all_results['tests']['2_decay_vs_time'] = test_decay_vs_time(fpga)
        all_results['tests']['3_partial_timing_sweep'] = test_partial_timing_sweep(fpga)
        all_results['tests']['4_partial_timing_decay'] = test_partial_timing_with_decay(fpga)
        all_results['tests']['5_temperature'] = test_temperature_correlation(fpga)
        all_results['tests']['6_repeatability'] = test_repeatability(fpga)
        all_results['tests']['7_pattern_sensitivity'] = test_pattern_sensitivity(fpga)

        # Final summary
        print("\n" + "="*60)
        print("FINAL SUMMARY")
        print("="*60)

        # Check test 1
        t1 = all_results['tests']['1_basic_reliability']
        print(f"\n1. Basic Reliability: {'PASS' if t1['overall_error_rate'] == 0 else 'ISSUES'}")
        print(f"   Error rate: {t1['overall_error_rate']*100:.2f}%")

        # Check test 4 (key test for partial timing)
        t4 = all_results['tests']['4_partial_timing_decay']
        successful = [r for r in t4 if r.get('success')]
        if successful:
            control = successful[0]  # offset 0
            high_offset = [r for r in successful if r['offset'] >= 55]
            if high_offset:
                delta = high_offset[0]['decay_errors'] - control['decay_errors']
                print(f"\n4. Partial Timing Effect:")
                print(f"   Control (offset 0) decay: {control['decay_errors']} errors")
                print(f"   High offset ({high_offset[0]['offset']}) decay: {high_offset[0]['decay_errors']} errors")
                print(f"   Delta: {delta:+d} errors")
                if delta > 10:
                    print("   CONCLUSION: Partial timing DOES create weaker writes!")
                else:
                    print("   CONCLUSION: Effect marginal, may need longer decay or higher temp")

        # Check test 6 (repeatability)
        t6 = all_results['tests']['6_repeatability']
        decay_errors = [r['decay_errors'] for r in t6 if r.get('decay_errors', -1) >= 0]
        if decay_errors:
            stdev = statistics.stdev(decay_errors) if len(decay_errors) > 1 else 0
            print(f"\n6. Repeatability:")
            print(f"   Std dev of decay errors: {stdev:.1f}")
            print(f"   {'GOOD' if stdev < 10 else 'VARIABLE'} repeatability")

        # Save results
        results_file = Path(__file__).parent.parent / 'results' / 'z1115_comprehensive_validation.json'
        results_file.parent.mkdir(exist_ok=True)
        with open(results_file, 'w') as f:
            json.dump(all_results, f, indent=2, default=str)
        print(f"\nResults saved to: {results_file}")

    finally:
        fpga.disconnect()

    print(f"\nCompleted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
