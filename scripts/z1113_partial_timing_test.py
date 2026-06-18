#!/usr/bin/env python3
"""
z1113: Partial Timing Write Test - True Analog Charge Levels

This test demonstrates TRUE partial timing writes by adjusting the
PHASER_OUT fine delay in the MIG DDR3 PHY. This creates shorter
effective write pulses, resulting in partial charge in DRAM cells.

Theory:
- Fine delay range: 0-63 taps (~12ps/tap = ~0-750ps)
- Calibrated timing has ~30-40ps margin to DRAM timing window
- Shifting timing by 10+ taps significantly reduces write margin
- Less margin = partial charge = analog value

Verification:
- Write with offset 0 (full timing) - cells fully charged
- Write with offset 20 (reduced timing) - cells partially charged
- Write with offset 40 (severely reduced) - cells weakly charged
- Measure decay rates for each - partial charges decay faster!
"""

import sys
import time
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.fpga.fpga_interface import FPGAInterface


def test_partial_timing_sweep(fpga: FPGAInterface, address: int = 0x100000):
    """Test partial timing writes with increasing offset"""
    print("\n" + "="*60)
    print("TEST: Partial Timing Write Sweep")
    print("="*60)

    results = []
    offsets = [0, 5, 10, 15, 20, 25, 30, 35, 40]
    pattern = bytes([0xFF] * 16)

    for offset in offsets:
        print(f"\nOffset: {offset} taps (~{offset*12}ps)")

        result = fpga.partial_timing_write(address, pattern, offset, timeout=15)

        if result.get('success'):
            written = result['written_data']
            readback = result['read_data']

            # Calculate immediate bit errors (should be 0 if write successful)
            immediate_errors = sum(bin(a ^ b).count('1') for a, b in zip(written, readback))

            print(f"  Written:  {written.hex()}")
            print(f"  Readback: {readback.hex()}")
            print(f"  Immediate errors: {immediate_errors}")
            print(f"  Temperature: {result['temperature']:.1f}C")

            results.append({
                'offset': offset,
                'timing_ps': offset * 12,
                'written': written.hex(),
                'readback': readback.hex(),
                'immediate_errors': immediate_errors,
                'temperature': result['temperature'],
                'success': True
            })
        else:
            print("  FAILED")
            results.append({
                'offset': offset,
                'success': False
            })

        time.sleep(0.2)

    return results


def test_partial_vs_decay(fpga: FPGAInterface, address: int = 0x200000):
    """Compare decay rates of partial vs full timing writes"""
    print("\n" + "="*60)
    print("TEST: Partial Timing vs Decay Rate")
    print("="*60)

    results = []
    wait_seconds = 10.0
    wait_cycles = int(wait_seconds * 83333 * 1000)

    test_cases = [
        (0, "Full timing (control)"),
        (20, "Moderate offset"),
        (40, "High offset"),
    ]

    for offset, description in test_cases:
        addr = address + offset * 0x10000
        pattern = bytes([0xFF] * 16)

        print(f"\n{description} (offset={offset}):")

        # Write with partial timing
        write_result = fpga.partial_timing_write(addr, pattern, offset, timeout=15)

        if not write_result.get('success'):
            print("  Write FAILED")
            results.append({'offset': offset, 'success': False})
            continue

        # Wait for decay
        print(f"  Waiting {wait_seconds}s for decay...")
        time.sleep(wait_seconds)

        # Read back
        readback = fpga.ddr_read(addr)

        if readback:
            errors = sum(bin(a ^ b).count('1') for a, b in zip(pattern, readback))
            print(f"  Original: {pattern.hex()}")
            print(f"  Readback: {readback.hex()}")
            print(f"  Bit errors: {errors} ({errors/128*100:.1f}%)")

            results.append({
                'offset': offset,
                'description': description,
                'bit_errors': errors,
                'error_rate': errors / 128.0,
                'readback': readback.hex(),
                'success': True
            })
        else:
            print("  Read FAILED")
            results.append({'offset': offset, 'success': False})

    # Analysis
    print("\n" + "-"*40)
    print("ANALYSIS:")

    successful = [r for r in results if r.get('success')]
    if len(successful) >= 2:
        control = successful[0]
        for r in successful[1:]:
            error_diff = r['bit_errors'] - control['bit_errors']
            print(f"  Offset {r['offset']}: {error_diff:+d} more errors than control")
            if error_diff > 0:
                print(f"    -> Partial timing DID create weaker writes!")
            elif error_diff == 0:
                print(f"    -> No difference (need longer decay or higher temp)")

    return results


def main():
    print("="*60)
    print("z1113: Partial Timing Write Test")
    print("="*60)
    print()
    print("This test uses TRUE partial timing writes to create")
    print("analog charge levels in DDR3 cells by adjusting")
    print("PHASER_OUT fine delay in the MIG PHY.")
    print()

    fpga = FPGAInterface()

    if not fpga.connect():
        print("ERROR: Could not connect to FPGA")
        print("       Make sure the partial timing bitstream is loaded!")
        return 1

    print("Connected to FPGA")
    temp, _ = fpga.read_temperature()
    print(f"Temperature: {temp:.1f}C")

    all_results = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'initial_temp': temp,
        'tests': {}
    }

    try:
        # Test 1: Sweep timing offsets
        all_results['tests']['timing_sweep'] = test_partial_timing_sweep(fpga)

        # Test 2: Compare decay rates
        all_results['tests']['decay_comparison'] = test_partial_vs_decay(fpga)

        # Summary
        print("\n" + "="*60)
        print("SUMMARY")
        print("="*60)

        sweep = all_results['tests']['timing_sweep']
        decay = all_results['tests']['decay_comparison']

        # Check for analog behavior
        sweep_ok = [r for r in sweep if r.get('success', False)]
        if sweep_ok:
            # If immediate readback with high offset shows errors,
            # we're definitely affecting the write
            high_offset_errors = sum(
                r.get('immediate_errors', 0)
                for r in sweep_ok
                if r['offset'] >= 30
            )
            if high_offset_errors > 0:
                print(f"\nANALOG BEHAVIOR CONFIRMED!")
                print(f"  High-offset writes show {high_offset_errors} immediate errors")
                print(f"  This means partial timing IS creating weaker writes!")
            else:
                print(f"\nNo immediate errors from partial timing")
                print(f"  Write margin may be larger than expected")
                print(f"  Try higher offsets or check decay rate differences")

        # Save results
        results_file = Path(__file__).parent.parent / 'results' / 'z1113_partial_timing.json'
        results_file.parent.mkdir(exist_ok=True)
        with open(results_file, 'w') as f:
            json.dump(all_results, f, indent=2)
        print(f"\nResults saved to: {results_file}")

    finally:
        fpga.disconnect()

    return 0


if __name__ == '__main__':
    sys.exit(main())
