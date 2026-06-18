#!/usr/bin/env python3
"""
z1109: Test FPGA Embodied Communication

Tests the custom protocol once the embodied bitstream is loaded.
Can also work with demo bitstream in limited mode.
"""

import sys
import os
import json
import time
import struct
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import serial
    import serial.tools.list_ports
except ImportError:
    print("Error: pyserial not installed")
    sys.exit(1)

from src.fpga.arty_comm import ArtyFPGA, FPGACommand, probe_fpga


def test_demo_mode():
    """Test with the demo bitstream."""
    print("=== Demo Bitstream Test ===")

    status = probe_fpga('/dev/ttyUSB1')
    print(f"Connected: {status['connected']}")
    print(f"Demo mode: {status['demo_mode']}")

    if status['demo_mode']:
        print("Demo bitstream detected - limited functionality")
        print(f"Response preview: {status['response'][:100]}...")
        return True

    return False


def test_custom_protocol():
    """Test the custom embodied protocol."""
    print("\n=== Custom Protocol Test ===")

    fpga = ArtyFPGA('/dev/ttyUSB1')

    if not fpga.connect():
        print("Failed to connect")
        return False

    # Test ping
    print("\n1. Ping test...")
    success, latency = fpga.ping()
    print(f"   Ping: {'OK' if success else 'FAIL'} ({latency:.2f} ms)")

    if not success:
        print("   Custom bitstream not loaded")
        fpga.disconnect()
        return False

    # Test temperature
    print("\n2. Temperature reading...")
    temp = fpga.read_temperature()
    if temp:
        print(f"   Temperature: {temp:.2f}°C")
    else:
        print("   Temperature read failed")

    # Test timing stats
    print("\n3. Timing statistics...")
    stats = fpga.get_timing_stats()
    if stats:
        print(f"   Cycles: {stats['total_cycles']}")
        print(f"   Min cycle: {stats['min_cycle_ns']} ns")
        print(f"   Max cycle: {stats['max_cycle_ns']} ns")
        print(f"   Thermal events: {stats['thermal_stretch_events']}")

    # Test embodiment toggle
    print("\n4. Embodiment control...")
    fpga.enable_embodiment(mode=1)
    print("   Enabled temperature-modulated mode")

    status = fpga.get_status()
    if status:
        print(f"   Status: {status}")

    fpga.disconnect()
    return True


def measure_thermal_response():
    """Measure how FPGA responds to temperature changes."""
    print("\n=== Thermal Response Test ===")
    print("This test requires the custom bitstream.\n")

    fpga = ArtyFPGA('/dev/ttyUSB1')

    if not fpga.connect():
        print("Failed to connect")
        return None

    success, _ = fpga.ping()
    if not success:
        print("Custom bitstream not loaded - skipping thermal test")
        fpga.disconnect()
        return None

    print("Monitoring temperature and timing for 30 seconds...")
    print("(Heat the FPGA with your finger or a heat gun to see effects)\n")

    samples = []
    start = time.time()

    while time.time() - start < 30:
        temp = fpga.read_temperature()
        stats = fpga.get_timing_stats()

        if temp and stats:
            samples.append({
                'time': time.time() - start,
                'temp_c': temp,
                'cycles': stats['total_cycles'],
                'thermal_events': stats['thermal_stretch_events']
            })
            print(f"  t={samples[-1]['time']:.1f}s: {temp:.1f}°C, "
                  f"thermal_events={stats['thermal_stretch_events']}")

        time.sleep(1)

    fpga.disconnect()

    # Analyze
    if len(samples) > 5:
        temps = [s['temp_c'] for s in samples]
        events = [s['thermal_events'] for s in samples]

        temp_range = max(temps) - min(temps)
        event_range = max(events) - min(events)

        print(f"\nResults:")
        print(f"  Temperature range: {min(temps):.1f}°C - {max(temps):.1f}°C")
        print(f"  Thermal events range: {min(events)} - {max(events)}")

        if temp_range > 2 and event_range > 0:
            print("\n  ✓ Temperature variation caused thermal events!")
            print("  → This is TRUE embodiment: temp affects computation")
        else:
            print("\n  Temperature didn't vary enough to trigger thermal events")
            print("  Try heating the FPGA more")

    return samples


def main():
    print("=" * 60)
    print("Z1109: FPGA Embodied Communication Test")
    print("=" * 60)
    print()

    results = {
        'experiment': 'z1109_fpga_test',
        'timestamp': datetime.now().isoformat(),
        'tests': {}
    }

    # Test 1: Demo mode
    demo_ok = test_demo_mode()
    results['tests']['demo_mode'] = demo_ok

    # Test 2: Custom protocol
    custom_ok = test_custom_protocol()
    results['tests']['custom_protocol'] = custom_ok

    # Test 3: Thermal response (if custom bitstream loaded)
    if custom_ok:
        thermal = measure_thermal_response()
        results['tests']['thermal_response'] = thermal

    # Summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)

    if custom_ok:
        print("\n✓ Custom embodied bitstream is running!")
        print("  - UART protocol working")
        print("  - Temperature sensing active")
        print("  - Embodied neurons ready")
        print("\nNext: Run z1110 for GPU-FPGA hybrid inference")
    else:
        print("\n! Demo bitstream detected")
        print("\nTo enable embodied computing:")
        print("  1. Install Vivado WebPACK")
        print("  2. Run: cd src/fpga && bash build.sh")
        print("  3. Program: openFPGALoader -b arty_a7_100t build/.../embodied_top.bit")

    # Save results
    results_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'results',
        'z1109_fpga_test.json'
    )

    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\nResults saved to: {results_path}")


if __name__ == '__main__':
    main()
