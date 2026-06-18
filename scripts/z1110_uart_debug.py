#!/usr/bin/env python3
"""
z1110: UART Debug Script for Arty A7-100T Embodied FPGA

Diagnoses UART communication issues with the embodied bitstream.
"""

import serial
import time
import sys

def test_port(port_name: str, verbose: bool = True) -> bool:
    """Test a serial port for FPGA response."""
    try:
        ser = serial.Serial(
            port=port_name,
            baudrate=115200,
            bytesize=serial.EIGHTBITS,
            parity=serial.PARITY_NONE,
            stopbits=serial.STOPBITS_ONE,
            timeout=0.5
        )

        # Flush any garbage
        ser.reset_input_buffer()
        ser.reset_output_buffer()
        time.sleep(0.1)

        # Send PING: [CMD=0x01, LEN=0x00]
        ping_cmd = bytes([0x01, 0x00])
        ser.write(ping_cmd)
        ser.flush()

        if verbose:
            print(f"  Sent PING: {ping_cmd.hex()}")

        # Wait for response
        time.sleep(0.1)
        response = ser.read(32)

        if verbose:
            print(f"  Response ({len(response)} bytes): {response.hex() if response else '(empty)'}")

        ser.close()

        # Expected: [0x01, 0x00] for PING response
        if response == bytes([0x01, 0x00]):
            return True
        elif len(response) > 0:
            print(f"  Got unexpected response!")
            return False
        else:
            return False

    except Exception as e:
        if verbose:
            print(f"  Error: {e}")
        return False


def test_raw_loopback(port_name: str) -> bool:
    """Test if port echoes (some FTDI modes do this)."""
    try:
        ser = serial.Serial(port=port_name, baudrate=115200, timeout=0.5)
        ser.reset_input_buffer()

        # Send some data
        test_data = b'\xAA\x55\x01\x02\x03'
        ser.write(test_data)
        ser.flush()
        time.sleep(0.1)

        response = ser.read(32)
        ser.close()

        if response == test_data:
            print(f"  WARNING: Port appears to be in loopback mode!")
            return True
        return False
    except:
        return False


def main():
    print("=" * 60)
    print("z1110: UART Debug for Embodied FPGA")
    print("=" * 60)

    # Wait for FPGA reset to complete (10ms internal counter)
    print("\nWaiting 500ms for FPGA reset to release...")
    time.sleep(0.5)

    ports = ['/dev/ttyUSB0', '/dev/ttyUSB1']

    # Check which ports exist
    import os
    available_ports = [p for p in ports if os.path.exists(p)]
    print(f"\nAvailable ports: {available_ports}")

    if not available_ports:
        print("ERROR: No USB serial ports found!")
        print("Check if FPGA is connected and detected:")
        print("  ls /dev/ttyUSB*")
        return

    # Test each port
    for port in available_ports:
        print(f"\n--- Testing {port} ---")

        # Check for loopback first
        if test_raw_loopback(port):
            continue

        # Try PING
        success = test_port(port)

        if success:
            print(f"  SUCCESS: FPGA responded correctly on {port}")
        else:
            # Try a few more times
            print(f"  Retrying...")
            for i in range(3):
                time.sleep(0.2)
                if test_port(port, verbose=False):
                    print(f"  SUCCESS on retry {i+1}")
                    success = True
                    break

            if not success:
                print(f"  FAILED: No valid response")

    print("\n" + "=" * 60)
    print("LED Status Check:")
    print("  LED[0]: Should blink when UART RX receives data")
    print("  LED[1]: Should blink when UART TX sends data")
    print("  LED[2]: Embodiment enabled (should be ON by default)")
    print("  LED[3]: Temperature above setpoint")
    print("  RGB LED: Blue=cold, Green=normal, Red=hot")
    print("=" * 60)

    # Extended diagnostics
    print("\nExtended diagnostics - trying different patterns...")

    for port in available_ports:
        print(f"\n--- Extended test on {port} ---")
        try:
            ser = serial.Serial(port=port, baudrate=115200, timeout=1.0)
            ser.reset_input_buffer()

            # Test 1: Send just command byte
            print("  Test 1: Send single byte 0x01...")
            ser.write(bytes([0x01]))
            ser.flush()
            time.sleep(0.2)
            resp = ser.read(32)
            print(f"    Response: {resp.hex() if resp else '(empty)'}")

            # Test 2: Send command + length with delay
            print("  Test 2: Send bytes with delay...")
            ser.reset_input_buffer()
            ser.write(bytes([0x01]))
            ser.flush()
            time.sleep(0.01)
            ser.write(bytes([0x00]))
            ser.flush()
            time.sleep(0.5)
            resp = ser.read(32)
            print(f"    Response: {resp.hex() if resp else '(empty)'}")

            # Test 3: Send READ_TEMP command
            print("  Test 3: Send READ_TEMP (0x02, 0x00)...")
            ser.reset_input_buffer()
            ser.write(bytes([0x02, 0x00]))
            ser.flush()
            time.sleep(0.5)
            resp = ser.read(32)
            print(f"    Response: {resp.hex() if resp else '(empty)'}")
            if len(resp) >= 6:
                temp_x100 = resp[2] | (resp[3] << 8)
                print(f"    Temperature: {temp_x100/100:.2f}°C")

            # Test 4: Send GET_TIMING command
            print("  Test 4: Send GET_TIMING (0x30, 0x00)...")
            ser.reset_input_buffer()
            ser.write(bytes([0x30, 0x00]))
            ser.flush()
            time.sleep(0.5)
            resp = ser.read(32)
            print(f"    Response: {resp.hex() if resp else '(empty)'}")

            ser.close()

        except Exception as e:
            print(f"  Error: {e}")

    print("\n" + "=" * 60)
    print("If no responses, check:")
    print("1. Is the bitstream correctly programmed? (run openFPGALoader)")
    print("2. Are LEDs doing anything? (power, activity)")
    print("3. Try pressing reset button (BTN0) and test again")
    print("4. Check UART TX/RX wiring in constraints file")
    print("=" * 60)


if __name__ == "__main__":
    main()
