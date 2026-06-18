#!/usr/bin/env python3
"""
z1127 (Fast): DRAM Cell Calibration System - Streamlined Version

Optimized for faster scanning:
1. First pass: Quick decay test on all addresses
2. Second pass: Detailed tests only on promising cells
3. Timing tests only on top candidates
"""

import sys
import os
import json
import time
import numpy as np
from pathlib import Path
from dataclasses import dataclass, asdict, field
from typing import List, Dict, Optional, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.fpga.fpga_interface import FPGAInterface


@dataclass
class CellCalibration:
    """Calibration data for a single DRAM cell/address"""
    address: int
    decay_errors: List[int] = field(default_factory=list)
    decay_mean: float = 0.0
    decay_std: float = 0.0
    decay_reproducible: bool = False
    pattern_aa_errors: int = -1
    pattern_55_errors: int = -1
    pattern_sensitive: bool = False
    timing_responsive: bool = False
    temp_at_test: float = 0.0
    reservoir_score: int = 0


def run_fast_calibration():
    """Run streamlined DRAM cell calibration"""
    print("=" * 70)
    print("z1127 (Fast): DRAM Cell Calibration System")
    print("=" * 70)

    fpga = FPGAInterface()

    print("\nConnecting to FPGA...")
    if not fpga.connect():
        print("ERROR: Could not connect to FPGA")
        return None

    print("FPGA connected")

    # Faster parameters
    NUM_ADDRESSES = 64
    BASE_ADDR = 0x600000  # Fresh region
    ADDR_STRIDE = 512  # Wider spacing
    DECAY_WAIT_CYCLES = 416666  # ~5ms (faster)
    DECAY_TRIALS = 3

    results = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'calibration_params': {
            'num_addresses': NUM_ADDRESSES,
            'base_addr': hex(BASE_ADDR),
            'addr_stride': ADDR_STRIDE,
            'decay_wait_ms': DECAY_WAIT_CYCLES / 83333
        },
        'cells': [],
        'summary': {}
    }

    cells: List[CellCalibration] = []

    temp_start, _ = fpga.read_temperature()
    print(f"\nStarting temperature: {temp_start:.1f}°C")

    # ================================================================
    # Phase 1: Quick decay scan
    # ================================================================
    print(f"\n{'=' * 50}")
    print(f"Phase 1: Quick scan of {NUM_ADDRESSES} addresses")
    print(f"{'=' * 50}")

    pattern_ff = bytes([0xFF] * 16)
    pattern_aa = bytes([0xAA] * 16)
    pattern_55 = bytes([0x55] * 16)

    for idx in range(NUM_ADDRESSES):
        addr = BASE_ADDR + (idx * ADDR_STRIDE)

        if idx % 16 == 0:
            print(f"\n  [{idx:3d}/{NUM_ADDRESSES}] ", end='', flush=True)

        cell = CellCalibration(address=addr)

        # Quick decay test with 0xFF (3 trials)
        for trial in range(DECAY_TRIALS):
            result = fpga.decay_test(addr, pattern_ff, DECAY_WAIT_CYCLES, timeout=10.0)
            if result.get('success'):
                cell.decay_errors.append(result['bit_errors'])
                cell.temp_at_test = result['temperature']
            time.sleep(0.02)

        if len(cell.decay_errors) < 2:
            print('X', end='', flush=True)
            continue

        cell.decay_mean = np.mean(cell.decay_errors)
        cell.decay_std = np.std(cell.decay_errors)
        cell.decay_reproducible = cell.decay_std < 3.0

        # Quick pattern test (single shot)
        result_aa = fpga.decay_test(addr, pattern_aa, DECAY_WAIT_CYCLES, timeout=10.0)
        result_55 = fpga.decay_test(addr, pattern_55, DECAY_WAIT_CYCLES, timeout=10.0)

        if result_aa.get('success'):
            cell.pattern_aa_errors = result_aa['bit_errors']
        if result_55.get('success'):
            cell.pattern_55_errors = result_55['bit_errors']

        cell.pattern_sensitive = (
            cell.pattern_aa_errors >= 0 and
            cell.pattern_55_errors >= 0 and
            abs(cell.pattern_aa_errors - cell.pattern_55_errors) > 10
        )

        # Calculate score
        score = 0
        if cell.decay_reproducible:
            score += 30
        if cell.pattern_sensitive:
            score += 30
        if cell.decay_mean > 0:
            score += 20
        if cell.decay_mean > 20:
            score += 20  # More decay = more signal

        cell.reservoir_score = min(score, 100)
        cells.append(cell)

        # Progress indicator
        if score >= 70:
            print('★', end='', flush=True)
        elif score >= 50:
            print('+', end='', flush=True)
        elif score >= 30:
            print('.', end='', flush=True)
        else:
            print('-', end='', flush=True)

    print(f"\n\nScanned {len(cells)} cells")

    # ================================================================
    # Phase 2: Analyze results
    # ================================================================
    print(f"\n{'=' * 50}")
    print("Phase 2: Analysis")
    print(f"{'=' * 50}")

    cells.sort(key=lambda c: c.reservoir_score, reverse=True)

    excellent = [c for c in cells if c.reservoir_score >= 70]
    good = [c for c in cells if 50 <= c.reservoir_score < 70]
    reproducible = [c for c in cells if c.decay_reproducible]
    pattern_sens = [c for c in cells if c.pattern_sensitive]

    print(f"\n  Excellent (≥70): {len(excellent)}")
    print(f"  Good (50-69):    {len(good)}")
    print(f"  Reproducible:    {len(reproducible)}")
    print(f"  Pattern-sens:    {len(pattern_sens)}")

    # Golden set (top 16)
    golden = cells[:min(16, len(cells))]
    golden_addrs = [c.address for c in golden]

    print(f"\n  Golden set ({len(golden)} cells):")
    for i, c in enumerate(golden[:8]):
        print(f"    {i+1}. 0x{c.address:07X}: score={c.reservoir_score}, "
              f"decay={c.decay_mean:.0f}±{c.decay_std:.0f}, "
              f"AA={c.pattern_aa_errors}, 55={c.pattern_55_errors}")

    # ================================================================
    # Summary
    # ================================================================
    temp_end, _ = fpga.read_temperature()

    summary = {
        'total_cells': len(cells),
        'excellent_cells': len(excellent),
        'good_cells': len(good),
        'reproducible_cells': len(reproducible),
        'pattern_sensitive_cells': len(pattern_sens),
        'avg_decay_errors': float(np.mean([c.decay_mean for c in cells])) if cells else 0,
        'max_decay_errors': float(max(c.decay_mean for c in cells)) if cells else 0,
        'golden_set_size': len(golden),
        'golden_addresses': [hex(a) for a in golden_addrs],
        'temp_start': temp_start,
        'temp_end': temp_end
    }

    results['summary'] = summary
    results['cells'] = [asdict(c) for c in cells]

    print(f"\n{'=' * 70}")
    print("CALIBRATION SUMMARY")
    print(f"{'=' * 70}")
    print(f"  Total cells:     {summary['total_cells']}")
    print(f"  Excellent:       {summary['excellent_cells']}")
    print(f"  Reproducible:    {summary['reproducible_cells']}")
    print(f"  Pattern-sens:    {summary['pattern_sensitive_cells']}")
    print(f"  Avg decay:       {summary['avg_decay_errors']:.1f} errors")
    print(f"  Max decay:       {summary['max_decay_errors']:.1f} errors")
    print(f"  Temp range:      {temp_start:.1f}°C → {temp_end:.1f}°C")

    # Assess embodiment potential
    print(f"\n{'=' * 70}")
    print("EMBODIMENT ASSESSMENT")
    print(f"{'=' * 70}")

    if summary['excellent_cells'] >= 5:
        print("  ✓ Good reservoir node candidates found")
    else:
        print("  ⚠ Few excellent cells - try different memory region")

    if summary['pattern_sensitive_cells'] >= 10:
        print("  ✓ Pattern-dependent encoding demonstrated")
    else:
        print("  ⚠ Limited pattern sensitivity - may need longer decay time")

    if summary['reproducible_cells'] >= summary['total_cells'] * 0.6:
        print("  ✓ Majority of cells are reproducible")
    else:
        print("  ⚠ High variability - consider thermal stabilization")

    if summary['avg_decay_errors'] > 10:
        print("  ✓ Significant decay signal for computation")
    else:
        print("  ⚠ Low decay signal - increase wait time or temperature")

    fpga.disconnect()
    return results


def main():
    results = run_fast_calibration()

    if results:
        output_path = Path('results/z1127_cell_calibration.json')
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\nResults saved to: {output_path}")

        # Save golden set
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
