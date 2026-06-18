#!/usr/bin/env python3
"""
z1111: Comprehensive FPGA DDR3 Decay and Partial Write Tests

Tests:
1. Multi-pattern decay (all-1s, all-0s, checkerboard patterns)
2. Masked/partial writes
3. Extended decay times
4. Multiple address decay comparison
5. Temperature correlation

Requirements:
- Arty A7-100T with embodied_ddr3_top bitstream
- USER_REFRESH="ON" in MIG for true decay measurement
"""

import sys
import time
import json
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.fpga.fpga_interface import FPGAInterface


def test_masked_write(fpga: FPGAInterface):
    """Test partial/masked writes"""
    print("\n" + "="*60)
    print("TEST: Masked/Partial Writes")
    print("="*60)

    results = []
    base_addr = 0x200000

    # First, write known pattern
    print("\n1. Writing initial pattern 0xDEADBEEF... to all bytes")
    initial = bytes([0xDE, 0xAD, 0xBE, 0xEF] * 4)
    if not fpga.ddr_write(base_addr, initial):
        print("   FAILED to write initial pattern")
        return results

    time.sleep(0.1)

    # Verify initial write
    read_back = fpga.ddr_read(base_addr)
    print(f"   Initial read: {read_back.hex()}")

    # Test: Mask first 8 bytes (only write to last 8)
    print("\n2. Masked write: mask=0x00FF (mask bytes 0-7, write bytes 8-15)")
    new_data = bytes([0x11, 0x22, 0x33, 0x44] * 4)
    result = fpga.masked_write(base_addr, new_data, mask=0x00FF)

    if result.get('success'):
        print(f"   Written data: {result['written_data'].hex()}")
        print(f"   Read back:    {result['read_data'].hex()}")
        print(f"   Mask used:    0x{result['mask_used']:04X}")
        print(f"   Temperature:  {result['temperature']:.1f}C")

        # Verify: first 8 bytes should be original (0xDEADBEEF...)
        # Last 8 bytes should be new (0x11223344...)
        expected_first_8 = bytes([0xDE, 0xAD, 0xBE, 0xEF] * 2)  # original
        expected_last_8 = bytes([0x11, 0x22, 0x33, 0x44] * 2)   # new

        # Note: We only get first 8 bytes in response
        results.append({
            'test': 'mask_first_8',
            'mask': 0x00FF,
            'read_data': result['read_data'].hex(),
            'temperature': result['temperature']
        })
    else:
        print("   FAILED")

    time.sleep(0.1)

    # Test: Mask odd bytes (checkerboard mask)
    print("\n3. Writing fresh pattern for next test")
    if fpga.ddr_write(base_addr, initial):
        print("   OK")

    time.sleep(0.1)

    print("\n4. Masked write: mask=0xAAAA (mask odd bytes, write even bytes)")
    result = fpga.masked_write(base_addr, bytes([0xFF] * 16), mask=0xAAAA)

    if result.get('success'):
        print(f"   Read back: {result['read_data'].hex()}")
        print(f"   Expected:  FF?? FF?? FF?? FF?? (even bytes 0xFF, odd bytes original)")
        results.append({
            'test': 'mask_odd_bytes',
            'mask': 0xAAAA,
            'read_data': result['read_data'].hex(),
            'temperature': result['temperature']
        })
    else:
        print("   FAILED")

    return results


def test_multi_pattern_decay(fpga: FPGAInterface, wait_seconds: float = 5.0):
    """Test decay with multiple patterns"""
    print("\n" + "="*60)
    print(f"TEST: Multi-Pattern Decay ({wait_seconds}s)")
    print("="*60)

    patterns = [
        (FPGAInterface.PATTERN_ALL_ONES, "All 1s (0xFF)", "Most likely to decay to 0"),
        (FPGAInterface.PATTERN_ALL_ZEROS, "All 0s (0x00)", "Should stay 0 (baseline)"),
        (FPGAInterface.PATTERN_CHECKER_AA, "Checker 0xAA", "50% duty cycle pattern"),
        (FPGAInterface.PATTERN_CHECKER_55, "Checker 0x55", "Inverse checker"),
    ]

    # ~83.33 MHz ui_clk = 83333 cycles per ms
    wait_cycles = int(wait_seconds * 83333 * 1000)

    results = []
    base_addr = 0x300000

    for pattern_id, name, description in patterns:
        addr = base_addr + (pattern_id * 0x10000)
        print(f"\n{name}: {description}")
        print(f"   Address: 0x{addr:06X}, Wait: {wait_cycles} cycles ({wait_seconds}s)")

        result = fpga.multi_decay_test(addr, pattern_id, wait_cycles, timeout=wait_seconds + 10)

        if result.get('success'):
            print(f"   Original:  {result['original_data'].hex()}")
            print(f"   Read back: {result['read_data'].hex()}")
            print(f"   Bit errors: {result['bit_errors']} / 64 bits ({result['bit_errors']/64*100:.1f}%)")
            print(f"   Temperature: {result['temperature']:.1f}C")

            results.append({
                'pattern': name,
                'pattern_id': pattern_id,
                'wait_seconds': wait_seconds,
                'original': result['original_data'].hex(),
                'readback': result['read_data'].hex(),
                'bit_errors': result['bit_errors'],
                'error_rate': result['bit_errors'] / 64.0,
                'temperature': result['temperature']
            })
        else:
            print("   FAILED")
            results.append({
                'pattern': name,
                'pattern_id': pattern_id,
                'success': False
            })

        time.sleep(0.5)

    return results


def test_decay_sweep(fpga: FPGAInterface):
    """Sweep decay times to find retention curve"""
    print("\n" + "="*60)
    print("TEST: Decay Time Sweep")
    print("="*60)

    # Test durations in seconds
    durations = [0.1, 0.5, 1.0, 5.0, 10.0, 30.0, 60.0]
    base_addr = 0x400000

    results = []

    print(f"\nTesting with all-1s pattern at address 0x{base_addr:06X}")
    print("-" * 50)

    for duration in durations:
        wait_cycles = int(duration * 83333 * 1000)
        print(f"\nDuration: {duration}s ({wait_cycles} cycles)")

        # Get temperature before
        temp_before, _ = fpga.read_temperature()

        result = fpga.multi_decay_test(base_addr, FPGAInterface.PATTERN_ALL_ONES,
                                       wait_cycles, timeout=duration + 15)

        if result.get('success'):
            print(f"   Temp before: {temp_before:.1f}C, after: {result['temperature']:.1f}C")
            print(f"   Original:  {result['original_data'].hex()}")
            print(f"   Read back: {result['read_data'].hex()}")
            print(f"   Bit errors: {result['bit_errors']} ({result['bit_errors']/64*100:.1f}%)")

            results.append({
                'duration_s': duration,
                'wait_cycles': wait_cycles,
                'temp_before': temp_before,
                'temp_after': result['temperature'],
                'bit_errors': result['bit_errors'],
                'error_rate': result['bit_errors'] / 64.0,
                'original': result['original_data'].hex(),
                'readback': result['read_data'].hex()
            })
        else:
            print("   FAILED")
            results.append({
                'duration_s': duration,
                'success': False
            })

        # Short delay between tests
        time.sleep(0.5)

    return results


def test_address_comparison(fpga: FPGAInterface, wait_seconds: float = 10.0):
    """Compare decay across different addresses (cell variability)"""
    print("\n" + "="*60)
    print(f"TEST: Address Comparison ({wait_seconds}s)")
    print("="*60)

    addresses = [
        0x000000,  # Start of DRAM
        0x100000,  # 1MB offset
        0x400000,  # 4MB offset
        0x800000,  # 8MB offset
        0xC00000,  # 12MB offset
        0xF00000,  # Near end
    ]

    wait_cycles = int(wait_seconds * 83333 * 1000)
    pattern = bytes([0xFF] * 16)

    results = []

    # First, write pattern to all addresses
    print("\nWriting pattern to all addresses...")
    for addr in addresses:
        if fpga.ddr_write(addr, pattern):
            print(f"   0x{addr:06X}: OK")
        else:
            print(f"   0x{addr:06X}: FAILED")

    # Now wait for decay (all addresses aging together)
    print(f"\nWaiting {wait_seconds}s for decay...")
    time.sleep(wait_seconds)

    # Read back all addresses
    print("\nReading back all addresses:")
    for addr in addresses:
        read_data = fpga.ddr_read(addr)
        if read_data:
            errors = sum(bin(a ^ b).count('1') for a, b in zip(pattern, read_data))
            print(f"   0x{addr:06X}: {read_data.hex()} - {errors} errors")
            results.append({
                'address': addr,
                'read_data': read_data.hex(),
                'bit_errors': errors,
                'error_rate': errors / 128.0
            })
        else:
            print(f"   0x{addr:06X}: READ FAILED")
            results.append({
                'address': addr,
                'success': False
            })

    return results


def main():
    """Run all FPGA decay tests"""
    print("="*60)
    print("z1111: FPGA DDR3 Decay and Partial Write Tests")
    print("="*60)

    fpga = FPGAInterface()

    print("\nConnecting to FPGA...")
    if not fpga.connect():
        print("ERROR: Could not connect to FPGA")
        print("       Make sure Arty A7 is connected and bitstream is loaded")
        return 1

    print("Connected!")

    # Get initial status
    status = fpga.get_status()
    temp, _ = fpga.read_temperature()
    print(f"\nInitial Status:")
    print(f"   DDR3 Calibrated: {status.get('ddr3_calibrated', False)}")
    print(f"   Temperature: {temp:.1f}C")

    all_results = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'initial_temp': temp,
        'tests': {}
    }

    try:
        # Run tests
        print("\n" + "="*60)
        print("RUNNING ALL TESTS")
        print("="*60)

        # 1. Masked write tests
        all_results['tests']['masked_write'] = test_masked_write(fpga)

        # 2. Multi-pattern decay (short)
        all_results['tests']['multi_pattern_5s'] = test_multi_pattern_decay(fpga, wait_seconds=5.0)

        # 3. Decay time sweep
        all_results['tests']['decay_sweep'] = test_decay_sweep(fpga)

        # 4. Address comparison
        all_results['tests']['address_comparison'] = test_address_comparison(fpga, wait_seconds=30.0)

        # Summary
        print("\n" + "="*60)
        print("SUMMARY")
        print("="*60)

        # Check for any actual decay
        total_errors = 0
        for test_name, results in all_results['tests'].items():
            if isinstance(results, list):
                for r in results:
                    if isinstance(r, dict) and 'bit_errors' in r:
                        total_errors += r['bit_errors']

        print(f"\nTotal bit errors observed: {total_errors}")

        if total_errors == 0:
            print("\nNO DECAY OBSERVED!")
            print("This is expected if:")
            print("  - Temperature is moderate (<50C)")
            print("  - Wait times are not long enough")
            print("  - DDR3 refresh is still happening (check USER_REFRESH)")
            print("\nTo observe decay:")
            print("  1. Heat the board to 50-60C")
            print("  2. Run longer tests (minutes, not seconds)")
            print("  3. Verify USER_REFRESH=ON in synthesis log")
        else:
            print(f"\nDECAY OBSERVED! {total_errors} bit errors")
            print("This is the embodiment signal we're looking for!")

        # Save results
        results_file = Path(__file__).parent.parent / 'results' / 'z1111_fpga_decay.json'
        results_file.parent.mkdir(exist_ok=True)
        with open(results_file, 'w') as f:
            json.dump(all_results, f, indent=2)
        print(f"\nResults saved to: {results_file}")

    finally:
        fpga.disconnect()

    return 0


if __name__ == '__main__':
    sys.exit(main())
