#!/usr/bin/env python3
"""
z1107: FPGA-Based True Embodiment

This experiment explores FPGA as the path to true hardware-software embodiment,
after z1106 conclusively showed that GPU abstraction is too strong.

Phase 1: Communication and baseline with demo bitstream
Phase 2: Custom bitstream with embodied logic (future)
"""

import sys
import os
import json
import time
import struct
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import serial
import serial.tools.list_ports


def find_arty_ports():
    """Find Arty A7 FPGA serial ports."""
    arty_ports = []
    for port in serial.tools.list_ports.comports():
        if '0403:6010' in port.hwid or 'Digilent' in port.description:
            arty_ports.append({
                'device': port.device,
                'description': port.description,
                'hwid': port.hwid
            })
    return arty_ports


def probe_demo_bitstream(port: str) -> dict:
    """
    Probe the demo bitstream to understand current state.

    Returns info about what's running on the FPGA.
    """
    result = {
        'port': port,
        'connected': False,
        'is_demo': False,
        'demo_response': None
    }

    try:
        ser = serial.Serial(port, baudrate=115200, timeout=1.0)
        ser.reset_input_buffer()

        # Send carriage return to wake up any shell
        ser.write(b'\r\n')
        time.sleep(0.3)

        response = ser.read(1024)
        result['connected'] = True

        if response:
            text = response.decode('utf-8', errors='replace')
            result['demo_response'] = text

            if 'Arty' in text or 'LED' in text or 'GPIO' in text:
                result['is_demo'] = True

        ser.close()

    except Exception as e:
        result['error'] = str(e)

    return result


def measure_uart_timing(port: str, n_trials: int = 100) -> dict:
    """
    Measure UART round-trip timing variations.

    Even with demo bitstream, timing variations might reveal
    some hardware state correlation.
    """
    result = {
        'n_trials': n_trials,
        'timings_us': [],
        'mean_us': 0,
        'std_us': 0,
        'cv_pct': 0
    }

    try:
        ser = serial.Serial(port, baudrate=115200, timeout=0.5)
        ser.reset_input_buffer()

        timings = []
        for i in range(n_trials):
            # Clear buffer
            ser.reset_input_buffer()

            # Send a character and measure response time
            start = time.perf_counter_ns()
            ser.write(b'?')
            response = ser.read(1)
            end = time.perf_counter_ns()

            if response:
                timings.append((end - start) / 1000)  # us

            time.sleep(0.01)  # Small delay between trials

        ser.close()

        if timings:
            import statistics
            result['timings_us'] = timings
            result['mean_us'] = statistics.mean(timings)
            result['std_us'] = statistics.stdev(timings) if len(timings) > 1 else 0
            result['cv_pct'] = 100 * result['std_us'] / result['mean_us'] if result['mean_us'] > 0 else 0

    except Exception as e:
        result['error'] = str(e)

    return result


def demo_button_interaction(port: str) -> dict:
    """
    Test interaction with the demo bitstream's menu.

    The demo responds to button presses, but we can try
    simulating those via serial to see FPGA behavior.
    """
    result = {
        'interactions': [],
        'timing_per_command': {}
    }

    try:
        ser = serial.Serial(port, baudrate=115200, timeout=1.0)
        ser.reset_input_buffer()

        # The demo has a menu. Let's see if we can interact.
        commands = [
            ('newline', b'\r\n'),
            ('0', b'0'),
            ('1', b'1'),
            ('?', b'?'),
        ]

        for name, cmd in commands:
            ser.reset_input_buffer()
            start = time.perf_counter()
            ser.write(cmd)
            time.sleep(0.2)
            response = ser.read(512)
            elapsed_ms = (time.perf_counter() - start) * 1000

            result['interactions'].append({
                'command': name,
                'response_len': len(response),
                'response_preview': response[:100].decode('utf-8', errors='replace') if response else None,
                'elapsed_ms': elapsed_ms
            })
            result['timing_per_command'][name] = elapsed_ms

        ser.close()

    except Exception as e:
        result['error'] = str(e)

    return result


def check_fpga_tools() -> dict:
    """Check available FPGA development tools."""
    tools = {
        'openfpgaloader': False,
        'vivado': False,
        'openocd': False
    }

    import shutil

    tools['openfpgaloader'] = shutil.which('openFPGALoader') is not None
    tools['vivado'] = shutil.which('vivado') is not None
    tools['openocd'] = shutil.which('openocd') is not None

    return tools


def generate_bitstream_requirements() -> dict:
    """
    Generate requirements for custom embodied bitstream.

    This documents what we need to implement in HDL.
    """
    return {
        'uart_module': {
            'description': 'Basic UART at 115200 baud',
            'required': True,
            'complexity': 'low'
        },
        'xadc_reader': {
            'description': 'Read on-chip temperature from XADC',
            'required': True,
            'complexity': 'medium'
        },
        'vector_processor': {
            'description': 'Process 128-element fixed-point vectors',
            'required': True,
            'complexity': 'medium'
        },
        'temp_modulated_threshold': {
            'description': 'Threshold that varies with XADC temperature',
            'required': True,
            'complexity': 'low'
        },
        'thermal_clock_stretch': {
            'description': 'Clock divider that increases when hot',
            'required': False,
            'complexity': 'medium'
        },
        'ring_oscillator_rng': {
            'description': 'True RNG from ring oscillator jitter',
            'required': False,
            'complexity': 'medium'
        },
        'timing_reporter': {
            'description': 'Report cycle counts and timing variations',
            'required': True,
            'complexity': 'low'
        }
    }


def main():
    print("=" * 60)
    print("Z1107: FPGA-Based True Embodiment")
    print("=" * 60)
    print()

    results = {
        'experiment': 'z1107_fpga_embodiment',
        'timestamp': datetime.now().isoformat(),
        'phase': 'exploration',
        'findings': {}
    }

    # Step 1: Find FPGA ports
    print("Step 1: Finding Arty A7 FPGA ports...")
    ports = find_arty_ports()
    results['findings']['ports'] = ports

    if not ports:
        print("  ERROR: No Arty A7 found!")
        print("  Check: lsusb | grep 0403:6010")
        results['findings']['error'] = 'No FPGA found'
        return results

    print(f"  Found {len(ports)} port(s):")
    for p in ports:
        print(f"    {p['device']}: {p['description']}")

    uart_port = None
    for p in ports:
        if 'USB1' in p['device']:  # UART is usually on second interface
            uart_port = p['device']
            break

    if not uart_port and ports:
        uart_port = ports[-1]['device']

    print(f"  Using UART port: {uart_port}")
    print()

    # Step 2: Probe demo bitstream
    print("Step 2: Probing current bitstream...")
    probe = probe_demo_bitstream(uart_port)
    results['findings']['demo_probe'] = probe

    if probe.get('connected'):
        print(f"  Connected: Yes")
        print(f"  Is demo bitstream: {probe.get('is_demo')}")
        if probe.get('demo_response'):
            preview = probe['demo_response'][:200].replace('\n', '\\n')
            print(f"  Response preview: {preview}...")
    else:
        print(f"  Connected: No")
        print(f"  Error: {probe.get('error')}")
    print()

    # Step 3: Measure UART timing
    print("Step 3: Measuring UART timing variations...")
    timing = measure_uart_timing(uart_port, n_trials=50)
    results['findings']['uart_timing'] = {
        'n_trials': timing['n_trials'],
        'mean_us': timing.get('mean_us', 0),
        'std_us': timing.get('std_us', 0),
        'cv_pct': timing.get('cv_pct', 0)
    }

    if 'error' not in timing:
        print(f"  Trials: {timing['n_trials']}")
        print(f"  Mean: {timing['mean_us']:.1f} us")
        print(f"  Std: {timing['std_us']:.1f} us")
        print(f"  CV: {timing['cv_pct']:.1f}%")
    else:
        print(f"  Error: {timing.get('error')}")
    print()

    # Step 4: Demo interaction
    print("Step 4: Testing demo bitstream interaction...")
    interaction = demo_button_interaction(uart_port)
    results['findings']['demo_interaction'] = interaction

    for inter in interaction.get('interactions', []):
        print(f"  Command '{inter['command']}': {inter['response_len']} bytes, {inter['elapsed_ms']:.1f}ms")
    print()

    # Step 5: Check tools
    print("Step 5: Checking FPGA development tools...")
    tools = check_fpga_tools()
    results['findings']['tools'] = tools

    for tool, available in tools.items():
        status = "✓" if available else "✗"
        print(f"  {status} {tool}")
    print()

    # Step 6: Bitstream requirements
    print("Step 6: Custom bitstream requirements...")
    requirements = generate_bitstream_requirements()
    results['findings']['bitstream_requirements'] = requirements

    for module, info in requirements.items():
        req = "REQUIRED" if info['required'] else "optional"
        print(f"  {module}: {info['complexity']} complexity ({req})")
    print()

    # Summary
    print("=" * 60)
    print("Summary")
    print("=" * 60)
    print()

    if probe.get('is_demo'):
        print("Current state: Demo bitstream running")
        print("  - LEDs/switches demo is active")
        print("  - Need custom bitstream for embodiment experiments")
        print()

    print("Why FPGA for embodiment (after z1106 showed GPU fails):")
    print("  1. No driver abstraction hiding hardware state")
    print("  2. XADC provides real on-chip temperature")
    print("  3. We can design logic where temp DIRECTLY affects thresholds")
    print("  4. Clock/timing under our control")
    print("  5. Ring oscillators provide true hardware entropy")
    print()

    print("Next steps:")
    if not tools['vivado']:
        print("  1. Install Vivado for HDL synthesis (free WebPACK edition)")
    print("  2. Create custom bitstream with UART + XADC + embodied logic")
    print("  3. Implement temperature-modulated neuron (z1107a)")
    print("  4. Compare FPGA embodiment vs GPU non-embodiment")
    print()

    # Save results
    results_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'results',
        'z1107_fpga_embodiment.json'
    )

    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print(f"Results saved to: {results_path}")

    return results


if __name__ == '__main__':
    main()
