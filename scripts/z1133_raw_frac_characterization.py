#!/usr/bin/env python3
"""
z1133: Raw Frac Characterization for Multi-Level Charging

Tests the new CMD_RAW_FRAC command to see if rapid self-refresh cycling
can produce intermediate charge levels beyond the {0, 32, 64, 128} ones
we observed in z1132.

Key parameters to sweep:
- num_fracs: 1-15 (number of SR enter/exit cycles)
- sr_wait: 50-2000 (cycles to wait in self-refresh state)
- pattern: 0x00, 0xFF, checker patterns

Expected outcomes:
- FAIL: Still only {0, 32, 64, 128} ones -> MIG completely blocks intermediate states
- SUCCESS: 65-127 range seen -> SR cycling achieves partial charge
- PARTIAL: New discrete levels found (e.g., 96, 80) -> Quantized charging behavior
"""

import sys
import time
import json
import struct
import statistics
from pathlib import Path
from datetime import datetime
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent / "src" / "fpga"))
from fpga_interface import FPGAInterface

def count_ones(data: bytes) -> int:
    """Count total ones in byte array."""
    return sum(bin(b).count('1') for b in data)

def run_characterization():
    """Sweep Frac parameters and characterize charge levels."""

    results = {
        "experiment": "z1133_raw_frac_characterization",
        "timestamp": datetime.now().isoformat(),
        "parameters": {},
        "sweeps": [],
        "summary": {},
        "unique_levels": set(),
        "hypothesis": "SR cycling can produce intermediate charge levels"
    }

    print("=" * 70)
    print("z1133: Raw Frac Characterization for Multi-Level Charging")
    print("=" * 70)

    try:
        fpga = FPGAInterface()
        if not fpga.connect():
            print("ERROR: Failed to connect to FPGA")
            return None

        print(f"Connected to FPGA: {fpga.port}")

        # Test if CMD_RAW_FRAC is available
        if not hasattr(fpga, 'raw_frac'):
            print("ERROR: raw_frac() not implemented in fpga_interface.py")
            print("       Need to update Python interface for CMD_RAW_FRAC")
            fpga.disconnect()
            return None

        # Test parameters
        test_addr = 0x00000000
        pattern_ff = bytes([0xFF] * 16)  # All ones pattern
        pattern_00 = bytes([0x00] * 16)  # All zeros pattern
        pattern_aa = bytes([0xAA] * 16)  # Checker pattern

        # Parameter sweeps
        num_fracs_range = [1, 2, 3, 5, 7, 10, 15]
        sr_wait_range = [50, 100, 200, 500, 1000, 2000]
        patterns = [
            ("0xFF", pattern_ff),
            ("0xAA", pattern_aa),
        ]

        trials_per_config = 5
        all_ones_counts = []

        print("\n--- Sweep Parameters ---")
        print(f"num_fracs: {num_fracs_range}")
        print(f"sr_wait: {sr_wait_range}")
        print(f"patterns: {[p[0] for p in patterns]}")
        print(f"trials per config: {trials_per_config}")
        print()

        for pattern_name, pattern_data in patterns:
            print(f"\n=== Pattern: {pattern_name} ===")

            for num_fracs in num_fracs_range:
                for sr_wait in sr_wait_range:
                    config_results = []

                    for trial in range(trials_per_config):
                        try:
                            # Execute raw Frac operation
                            result = fpga.raw_frac(
                                address=test_addr,
                                data=pattern_data,
                                num_fracs=num_fracs,
                                sr_wait=sr_wait,
                                timeout=30.0
                            )

                            if result and 'data' in result:
                                ones = count_ones(result['data'])
                                config_results.append(ones)
                                all_ones_counts.append(ones)
                                results['unique_levels'].add(ones)
                            else:
                                config_results.append(None)

                        except Exception as e:
                            print(f"  Trial {trial} error: {e}")
                            config_results.append(None)

                    # Statistics for this config
                    valid_results = [x for x in config_results if x is not None]
                    if valid_results:
                        mean_ones = statistics.mean(valid_results)
                        std_ones = statistics.stdev(valid_results) if len(valid_results) > 1 else 0
                        unique_vals = sorted(set(valid_results))

                        sweep_result = {
                            "pattern": pattern_name,
                            "num_fracs": num_fracs,
                            "sr_wait": sr_wait,
                            "ones_values": valid_results,
                            "mean": round(mean_ones, 2),
                            "std": round(std_ones, 2),
                            "unique": unique_vals
                        }
                        results['sweeps'].append(sweep_result)

                        # Check for intermediate values
                        has_intermediate = any(0 < v < 128 and v not in [32, 64] for v in valid_results)
                        marker = " ***INTERMEDIATE***" if has_intermediate else ""

                        print(f"  fracs={num_fracs:2d} sr_wait={sr_wait:4d}: "
                              f"ones={unique_vals} mean={mean_ones:.1f} std={std_ones:.2f}{marker}")
                    else:
                        print(f"  fracs={num_fracs:2d} sr_wait={sr_wait:4d}: ALL FAILED")

        # Summary analysis
        print("\n" + "=" * 70)
        print("SUMMARY")
        print("=" * 70)

        unique_levels = sorted(results['unique_levels'])
        print(f"\nUnique charge levels observed: {unique_levels}")

        # Check against expected failure pattern {0, 32, 64, 128}
        expected_binary = {0, 32, 64, 128}
        unexpected_levels = results['unique_levels'] - expected_binary

        if unexpected_levels:
            print(f"\n*** SUCCESS: Found unexpected levels: {sorted(unexpected_levels)} ***")
            results['summary']['status'] = 'SUCCESS'
            results['summary']['intermediate_levels'] = sorted(unexpected_levels)
        else:
            print(f"\nFAIL: Only binary/quantized levels observed: {unique_levels}")
            results['summary']['status'] = 'FAIL'
            results['summary']['message'] = "SR cycling via MIG still produces only quantized levels"

        # Distribution analysis
        if all_ones_counts:
            level_counts = defaultdict(int)
            for ones in all_ones_counts:
                level_counts[ones] += 1

            print("\nDistribution of charge levels:")
            for level in sorted(level_counts.keys()):
                count = level_counts[level]
                pct = 100 * count / len(all_ones_counts)
                bar = '#' * int(pct / 2)
                print(f"  {level:3d} ones: {count:4d} ({pct:5.1f}%) {bar}")

        results['unique_levels'] = list(results['unique_levels'])  # Convert set for JSON

        fpga.disconnect()

    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        results['summary']['status'] = 'ERROR'
        results['summary']['error'] = str(e)

    # Save results
    results_path = Path(__file__).parent.parent / "results" / "z1133_raw_frac_characterization.json"
    results_path.parent.mkdir(exist_ok=True)
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {results_path}")

    return results


def test_connection_only():
    """Just test if FPGA responds before full characterization."""
    print("Testing FPGA connection...")

    try:
        fpga = FPGAInterface()
        if fpga.connect():
            print(f"Connected to: {fpga.port}")

            # Test basic read
            result = fpga.ddr_read(0x00000000)
            if result:
                print(f"DDR read OK: {result['data'][:16].hex()}")

            # Check if raw_frac exists
            if hasattr(fpga, 'raw_frac'):
                print("raw_frac() method: AVAILABLE")
            else:
                print("raw_frac() method: NOT FOUND - need to update fpga_interface.py")

            fpga.disconnect()
            return True
        else:
            print("Connection failed")
            return False
    except Exception as e:
        print(f"Error: {e}")
        return False


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="z1133: Raw Frac Characterization")
    parser.add_argument("--test", action="store_true", help="Just test connection")
    args = parser.parse_args()

    if args.test:
        test_connection_only()
    else:
        run_characterization()
