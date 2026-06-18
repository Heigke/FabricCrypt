#!/usr/bin/env python3
"""
z1127: DRAM Cell Calibration System

Scans DRAM addresses to find cells with predictable decay behavior.
Builds a calibration map for use in physical reservoir computing.

Key goals:
1. Identify cells with consistent decay patterns (good reservoir nodes)
2. Map cell behavior vs timing offset (analog input encoding)
3. Find temperature-sensitive cells for thermal sensing
4. Build "golden set" of addresses for embodied computing
"""

import sys
import os
import json
import time
import numpy as np
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Tuple
import struct

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.fpga.fpga_interface import FPGAInterface


@dataclass
class CellCalibration:
    """Calibration data for a single DRAM cell/address"""
    address: int
    # Decay test results
    decay_errors_mean: float
    decay_errors_std: float
    decay_reproducible: bool  # std < 2 errors
    # Pattern sensitivity
    pattern_aa_errors: int
    pattern_55_errors: int
    pattern_ff_errors: int
    pattern_sensitive: bool  # aa and 55 differ
    # Partial timing response
    timing_offsets_tested: List[int]
    timing_errors: List[int]  # Errors at each offset
    timing_responsive: bool  # Shows offset-error correlation
    # Thermal info
    temp_at_calibration: float
    # Quality score (0-100)
    reservoir_score: int


def run_calibration():
    """Run comprehensive DRAM cell calibration"""
    print("=" * 70)
    print("z1127: DRAM Cell Calibration System")
    print("=" * 70)

    fpga = FPGAInterface()

    print("\nConnecting to FPGA...")
    if not fpga.connect():
        print("ERROR: Could not connect to FPGA")
        return None

    print("FPGA connected, DDR3 ready")

    # Calibration parameters
    NUM_ADDRESSES = 128  # Scan 128 addresses (2KB region)
    BASE_ADDR = 0x500000  # Start in a fresh region
    ADDR_STRIDE = 256  # Space addresses out to reduce row effects
    DECAY_WAIT_CYCLES = 833333  # ~10ms at 83.333 MHz
    DECAY_TRIALS = 5  # Repeat decay tests for consistency
    TIMING_OFFSETS = [0, 8, 12, 14, 15, 16, 17, 18, 20, 24, 32]

    results = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'calibration_params': {
            'num_addresses': NUM_ADDRESSES,
            'base_addr': hex(BASE_ADDR),
            'addr_stride': ADDR_STRIDE,
            'decay_wait_cycles': DECAY_WAIT_CYCLES,
            'decay_trials': DECAY_TRIALS,
            'timing_offsets': TIMING_OFFSETS
        },
        'cell_calibrations': [],
        'summary': {}
    }

    calibrations: List[CellCalibration] = []

    # Get initial temperature
    temp_start, _ = fpga.read_temperature()
    print(f"\nStarting temperature: {temp_start:.1f}°C")

    # ================================================================
    # Phase 1: Scan addresses for decay behavior
    # ================================================================
    print(f"\n{'=' * 50}")
    print(f"Phase 1: Scanning {NUM_ADDRESSES} addresses for decay behavior")
    print(f"{'=' * 50}")

    pattern_ff = bytes([0xFF] * 16)
    pattern_aa = bytes([0xAA] * 16)
    pattern_55 = bytes([0x55] * 16)

    for idx in range(NUM_ADDRESSES):
        addr = BASE_ADDR + (idx * ADDR_STRIDE)

        if idx % 16 == 0:
            print(f"\n  Scanning addresses {idx}-{min(idx+15, NUM_ADDRESSES-1)}...", end='', flush=True)

        # Test decay reproducibility with 0xFF pattern
        decay_errors = []
        for trial in range(DECAY_TRIALS):
            result = fpga.decay_test(addr, pattern_ff, DECAY_WAIT_CYCLES, timeout=15.0)
            if result.get('success'):
                decay_errors.append(result['bit_errors'])
            time.sleep(0.05)  # Brief pause between trials

        if len(decay_errors) < DECAY_TRIALS // 2:
            print('X', end='', flush=True)  # Timeouts
            continue

        decay_mean = np.mean(decay_errors)
        decay_std = np.std(decay_errors)
        decay_reproducible = decay_std < 2.0

        # Test pattern sensitivity
        result_aa = fpga.decay_test(addr + 16, pattern_aa, DECAY_WAIT_CYCLES, timeout=15.0)
        result_55 = fpga.decay_test(addr + 32, pattern_55, DECAY_WAIT_CYCLES, timeout=15.0)

        aa_errors = result_aa.get('bit_errors', 0) if result_aa.get('success') else -1
        ff_errors = int(decay_mean)
        errors_55 = result_55.get('bit_errors', 0) if result_55.get('success') else -1

        pattern_sensitive = aa_errors >= 0 and errors_55 >= 0 and abs(aa_errors - errors_55) > 5

        # Test partial timing response
        timing_errors = []
        for offset in TIMING_OFFSETS:
            result = fpga.partial_timing_write(addr + 48, pattern_ff, offset, timeout=10.0)
            if result.get('success'):
                # Compare written vs read
                written = result['written_data']
                read = result['read_data']
                errors = sum(bin(a ^ b).count('1') for a, b in zip(written, read))
                timing_errors.append(errors)
            else:
                timing_errors.append(-1)  # Timeout (expected for some offsets)

        # Check if timing-responsive (non-zero errors at some offsets)
        valid_timing = [e for e in timing_errors if e >= 0]
        timing_responsive = len(valid_timing) >= 3 and max(valid_timing) > min(valid_timing) + 5

        # Get temperature
        temp, _ = fpga.read_temperature()

        # Calculate reservoir score
        score = 0
        if decay_reproducible:
            score += 30  # Consistency is critical
        if pattern_sensitive:
            score += 25  # Pattern-dependent decay is useful
        if timing_responsive:
            score += 35  # Analog input encoding capability
        if decay_mean > 0:
            score += 10  # Some decay is needed

        cal = CellCalibration(
            address=addr,
            decay_errors_mean=decay_mean,
            decay_errors_std=decay_std,
            decay_reproducible=decay_reproducible,
            pattern_aa_errors=aa_errors,
            pattern_55_errors=errors_55,
            pattern_ff_errors=ff_errors,
            pattern_sensitive=pattern_sensitive,
            timing_offsets_tested=TIMING_OFFSETS,
            timing_errors=timing_errors,
            timing_responsive=timing_responsive,
            temp_at_calibration=temp,
            reservoir_score=score
        )

        calibrations.append(cal)

        # Progress indicator
        if score >= 70:
            print('★', end='', flush=True)  # Excellent cell
        elif score >= 50:
            print('+', end='', flush=True)  # Good cell
        elif score >= 30:
            print('.', end='', flush=True)  # Marginal cell
        else:
            print('-', end='', flush=True)  # Poor cell

    print(f"\n\nCalibration complete: {len(calibrations)} cells tested")

    # ================================================================
    # Phase 2: Analyze and categorize cells
    # ================================================================
    print(f"\n{'=' * 50}")
    print("Phase 2: Analyzing cell categories")
    print(f"{'=' * 50}")

    # Sort by reservoir score
    calibrations.sort(key=lambda c: c.reservoir_score, reverse=True)

    excellent_cells = [c for c in calibrations if c.reservoir_score >= 70]
    good_cells = [c for c in calibrations if 50 <= c.reservoir_score < 70]
    marginal_cells = [c for c in calibrations if 30 <= c.reservoir_score < 50]
    poor_cells = [c for c in calibrations if c.reservoir_score < 30]

    reproducible_cells = [c for c in calibrations if c.decay_reproducible]
    pattern_sensitive_cells = [c for c in calibrations if c.pattern_sensitive]
    timing_responsive_cells = [c for c in calibrations if c.timing_responsive]

    print(f"\nCell quality distribution:")
    print(f"  Excellent (≥70): {len(excellent_cells)}")
    print(f"  Good (50-69):    {len(good_cells)}")
    print(f"  Marginal (30-49): {len(marginal_cells)}")
    print(f"  Poor (<30):      {len(poor_cells)}")

    print(f"\nCell capabilities:")
    print(f"  Reproducible decay: {len(reproducible_cells)}")
    print(f"  Pattern sensitive:  {len(pattern_sensitive_cells)}")
    print(f"  Timing responsive:  {len(timing_responsive_cells)}")

    # ================================================================
    # Phase 3: Build golden set
    # ================================================================
    print(f"\n{'=' * 50}")
    print("Phase 3: Building golden reservoir set")
    print(f"{'=' * 50}")

    # Select top 32 cells for reservoir computing
    golden_set = calibrations[:min(32, len(calibrations))]
    golden_addresses = [c.address for c in golden_set]

    print(f"\nGolden set ({len(golden_set)} cells):")
    for i, cell in enumerate(golden_set[:10]):
        print(f"  {i+1}. 0x{cell.address:07X}: score={cell.reservoir_score}, "
              f"decay_err={cell.decay_errors_mean:.1f}±{cell.decay_errors_std:.1f}, "
              f"timing={'Yes' if cell.timing_responsive else 'No'}")
    if len(golden_set) > 10:
        print(f"  ... and {len(golden_set)-10} more")

    # ================================================================
    # Summary and results
    # ================================================================
    print(f"\n{'=' * 70}")
    print("CALIBRATION SUMMARY")
    print(f"{'=' * 70}")

    temp_end, _ = fpga.read_temperature()
    temp_variation = abs(temp_end - temp_start)

    # Calculate aggregate stats
    if calibrations:
        avg_score = np.mean([c.reservoir_score for c in calibrations])
        max_score = max(c.reservoir_score for c in calibrations)

        # Find most thermally stable cells
        temp_stable = [c for c in calibrations if c.decay_errors_std < 1.0]
    else:
        avg_score = 0
        max_score = 0
        temp_stable = []

    summary = {
        'total_cells_tested': len(calibrations),
        'excellent_cells': len(excellent_cells),
        'good_cells': len(good_cells),
        'reproducible_cells': len(reproducible_cells),
        'pattern_sensitive_cells': len(pattern_sensitive_cells),
        'timing_responsive_cells': len(timing_responsive_cells),
        'avg_reservoir_score': float(avg_score),
        'max_reservoir_score': int(max_score),
        'golden_set_size': len(golden_set),
        'golden_addresses': [hex(a) for a in golden_addresses],
        'temp_start': temp_start,
        'temp_end': temp_end,
        'temp_variation': temp_variation,
        'temp_stable_cells': len(temp_stable)
    }

    results['summary'] = summary
    results['cell_calibrations'] = [asdict(c) for c in calibrations]

    print(f"\nResults:")
    print(f"  Cells tested:        {summary['total_cells_tested']}")
    print(f"  Excellent (≥70):     {summary['excellent_cells']}")
    print(f"  Pattern-sensitive:   {summary['pattern_sensitive_cells']}")
    print(f"  Timing-responsive:   {summary['timing_responsive_cells']}")
    print(f"  Average score:       {summary['avg_reservoir_score']:.1f}")
    print(f"  Golden set size:     {summary['golden_set_size']}")
    print(f"  Temp variation:      {summary['temp_variation']:.2f}°C")

    # ================================================================
    # Business value assessment
    # ================================================================
    print(f"\n{'=' * 70}")
    print("EMBODIED COMPUTING ASSESSMENT")
    print(f"{'=' * 70}")

    embodied_capabilities = []

    if summary['excellent_cells'] >= 10:
        embodied_capabilities.append("✓ Sufficient excellent cells for reservoir computing")

    if summary['pattern_sensitive_cells'] >= 20:
        embodied_capabilities.append("✓ Pattern-dependent encoding demonstrated")

    if summary['timing_responsive_cells'] >= 10:
        embodied_capabilities.append("✓ Analog input via partial timing writes")

    if summary['reproducible_cells'] >= summary['total_cells_tested'] * 0.5:
        embodied_capabilities.append("✓ Majority of cells show reproducible decay")

    if summary['temp_variation'] < 1.0:
        embodied_capabilities.append("✓ Thermal stability during calibration")

    for cap in embodied_capabilities:
        print(f"  {cap}")

    if not embodied_capabilities:
        print("  ⚠ Limited embodiment potential in this DRAM region")
        print("  Consider: different base address, longer decay wait, or thermal cycling")

    fpga.disconnect()
    return results


def main():
    results = run_calibration()

    if results:
        output_path = Path('results/z1127_cell_calibration.json')
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\nResults saved to: {output_path}")

        # Also save golden set as separate file for easy use
        golden_path = Path('results/z1127_golden_addresses.json')
        golden_data = {
            'timestamp': results['timestamp'],
            'addresses': results['summary'].get('golden_addresses', []),
            'description': 'Top-scoring DRAM cells for reservoir computing'
        }
        with open(golden_path, 'w') as f:
            json.dump(golden_data, f, indent=2)
        print(f"Golden set saved to: {golden_path}")


if __name__ == '__main__':
    main()
