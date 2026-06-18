#!/usr/bin/env python3
"""z2149: Vg→spike diagnostic — send known Vg values, measure spike rates.
Prints raw hex bytes for every SET_VG command for debugging.
"""
import struct, time, serial, sys

SYNC = 0x55
CMD_SET_VG = 0x01
CMD_READ_TELEM = 0x02
CMD_SET_KILL = 0x03

def to_q16_16(val: float) -> int:
    return int(val * 65536) & 0xFFFFFFFF

def crc8(data: bytes, poly: int = 0x07) -> int:
    crc = 0x00
    for byte in data:
        crc ^= byte
        for _ in range(8):
            crc = ((crc << 1) ^ poly) if crc & 0x80 else crc << 1
            crc &= 0xFF
    return crc

def find_fpga():
    for p in ['/dev/ttyUSB1', '/dev/ttyUSB0', '/dev/ttyUSB2']:
        try:
            s = serial.Serial(p, 115200, timeout=0.1)
            time.sleep(0.1)
            return s, p
        except:
            continue
    return None, None

def send_cmd(ser, cmd, payload=b''):
    pkt = bytes([SYNC, cmd]) + payload
    print(f"  TX [{len(pkt)}B]: {pkt.hex(' ')}")
    ser.write(pkt)
    ser.flush()

def read_response(ser, timeout=0.3):
    deadline = time.monotonic() + timeout
    buf = bytearray()
    while time.monotonic() < deadline:
        ser.timeout = min(deadline - time.monotonic(), 0.02)
        b = ser.read(1)
        if not b:
            continue
        if b[0] == SYNC:
            buf.append(b[0])
            break
    else:
        return None
    while len(buf) < 3 and time.monotonic() < deadline:
        ser.timeout = min(deadline - time.monotonic(), 0.02)
        chunk = ser.read(3 - len(buf))
        if chunk:
            buf.extend(chunk)
    if len(buf) < 3:
        return None
    payload_len = buf[2]
    if payload_len > 200:
        return None
    need = payload_len + 1
    while len(buf) < 3 + need and time.monotonic() < deadline:
        ser.timeout = min(deadline - time.monotonic(), 0.02)
        chunk = ser.read(3 + need - len(buf))
        if chunk:
            buf.extend(chunk)
    if len(buf) < 3 + need:
        return None
    pkt_body = bytes(buf[:-1])
    rx_crc = buf[-1]
    if crc8(pkt_body) != rx_crc:
        print(f"  CRC FAIL: computed={crc8(pkt_body):02x} rx={rx_crc:02x}")
        return None
    payload = bytes(buf[3:3 + payload_len])
    return payload

def read_spike_rates(ser, n_reads=20, delay=0.01):
    """Read telemetry n_reads times, return avg spike counts per neuron."""
    counts = [[] for _ in range(8)]
    for _ in range(n_reads):
        ser.reset_input_buffer()
        send_cmd(ser, CMD_READ_TELEM)
        payload = read_response(ser, timeout=0.2)
        if payload and len(payload) >= 48:
            for i in range(8):
                off = i * 6
                sc = struct.unpack('>H', payload[off:off+2])[0]
                counts[i].append(sc)
        time.sleep(delay)
    avgs = []
    for i in range(8):
        if counts[i]:
            avgs.append(sum(counts[i]) / len(counts[i]))
        else:
            avgs.append(0.0)
    return avgs

def main():
    ser, port = find_fpga()
    if not ser:
        print("ERROR: No FPGA found")
        sys.exit(1)
    print(f"Connected to {port}")

    # Flush
    ser.reset_input_buffer()
    time.sleep(0.1)

    # Initial read to verify connection
    print("\n=== Initial telemetry read ===")
    send_cmd(ser, CMD_READ_TELEM)
    payload = read_response(ser)
    if payload:
        print(f"  OK: {len(payload)} bytes")
        for i in range(8):
            off = i * 6
            sc = struct.unpack('>H', payload[off:off+2])[0]
            vm = struct.unpack('>H', payload[off+2:off+4])[0]
            bv = struct.unpack('>H', payload[off+4:off+6])[0]
            print(f"  N{i}: spike_cnt={sc:5d}  Vm={vm/256:.2f}  BVpar={bv/256:.2f}")
    else:
        print("  FAIL: no response")
        ser.close()
        sys.exit(1)

    # Vg sweep: test 6 levels across full range
    vg_levels = [0.0, 0.10, 0.20, 0.35, 0.50, 0.80, 1.00]

    print("\n=== Vg sweep: set all 8 neurons, read spike rates ===")
    print(f"{'Vg':>6}  {'Q16.16 hex':>12}  {'N0':>6}  {'N1':>6}  {'N2':>6}  {'N3':>6}  {'N4':>6}  {'N5':>6}  {'N6':>6}  {'N7':>6}  {'Mean':>8}")
    print("-" * 110)

    for vg in vg_levels:
        vg_q = to_q16_16(vg)
        print(f"\n  Setting Vg={vg:.2f} (Q16.16=0x{vg_q:08x})...")

        # Set all 8 neurons
        for nid in range(8):
            payload = bytes([nid & 0x07]) + struct.pack('>I', vg_q)
            send_cmd(ser, CMD_SET_VG, payload)
            time.sleep(0.005)  # 5ms between commands

        # Wait for Vg to settle
        time.sleep(0.1)

        # Discard first few reads (transient)
        for _ in range(3):
            ser.reset_input_buffer()
            send_cmd(ser, CMD_READ_TELEM)
            read_response(ser, timeout=0.1)
            time.sleep(0.01)

        # Read spike rates
        rates = read_spike_rates(ser, n_reads=30, delay=0.01)
        mean_rate = sum(rates) / len(rates)

        print(f"  {vg:6.2f}  0x{vg_q:08x}      {rates[0]:6.1f}  {rates[1]:6.1f}  {rates[2]:6.1f}  {rates[3]:6.1f}  {rates[4]:6.1f}  {rates[5]:6.1f}  {rates[6]:6.1f}  {rates[7]:6.1f}  {mean_rate:8.1f}")

    # Kill switch test
    print("\n=== Kill switch test ===")
    print("  Enabling kill switch...")
    send_cmd(ser, CMD_SET_KILL, bytes([0x01]))
    time.sleep(0.1)
    rates_killed = read_spike_rates(ser, n_reads=10)
    mean_killed = sum(rates_killed) / len(rates_killed)
    print(f"  Killed: mean_rate={mean_killed:.1f} (expect ~0)")

    print("  Disabling kill switch...")
    send_cmd(ser, CMD_SET_KILL, bytes([0x00]))
    time.sleep(0.1)

    # Restore Vg=0.35
    for nid in range(8):
        payload = bytes([nid]) + struct.pack('>I', to_q16_16(0.35))
        send_cmd(ser, CMD_SET_VG, payload)

    ser.close()
    print("\nDone.")

if __name__ == '__main__':
    main()
