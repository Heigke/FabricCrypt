#!/usr/bin/env python3
"""fpga_host_v2.py — Python host library for 128-neuron NS-RAM FPGA bridge (v2 protocol)

Protocol v2 changes:
  - UART baud: 921600 (was 115200)
  - neuron_id: 7-bit / full byte (was 3-bit)
  - Telemetry: [0x55][0x02][LEN_HI][LEN_LO][128×6B][CRC8] = 773 bytes (was 52)
  - New CMD_SET_VG_BATCH (0x08): bulk Vg update in one packet
  - Backward-compatible: auto-detects v1 (8-neuron) vs v2 (128-neuron) firmware

Hardware: AMD gfx1151 GPU + Arty A7-100T FPGA on /dev/ttyUSB{0,1,2}
"""

import os, sys, time, struct
from pathlib import Path

# ─── Protocol Constants ───
SYNC = 0x55
CMD_SET_VG       = 0x01
CMD_READ_TELEM   = 0x02
CMD_SET_KILL     = 0x03
CMD_SET_MAC      = 0x06
CMD_SET_SYNAPSE  = 0x07
CMD_SET_VG_BATCH = 0x08

# v1 firmware: 8 neurons, 115200 baud, 52-byte telemetry
# v2 firmware: 128 neurons, 921600 baud, 773-byte telemetry
V1_BAUD = 115200
V2_BAUD = 921600
V1_NEURONS = 8
V2_NEURONS = 128
V1_TELEM_LEN = 52   # [SYNC][0x02][0x30][48B][CRC8]
V2_TELEM_LEN = 773  # [SYNC][0x02][LEN_HI][LEN_LO][768B][CRC8]
V2_PAYLOAD_LEN = 768  # 128 neurons × 6 bytes each


def crc8(data: bytes) -> int:
    """CRC-8/SMBUS (polynomial 0x07, init 0x00) — matches RTL crc8_byte."""
    crc = 0x00
    for b in data:
        crc ^= b
        for _ in range(8):
            if crc & 0x80:
                crc = ((crc << 1) ^ 0x07) & 0xFF
            else:
                crc = (crc << 1) & 0xFF
    return crc


def to_q16_16(val: float) -> int:
    """Convert float to Q16.16 fixed-point (unsigned 32-bit)."""
    return int(val * 65536) & 0xFFFFFFFF


def from_q16_16(val: int) -> float:
    """Convert Q16.16 fixed-point to float."""
    return val / 65536.0


class FPGABridge:
    """Host-side driver for NS-RAM FPGA bridge (supports v1 and v2 firmware)."""

    def __init__(self, port=None, baud=None, num_neurons=None, timeout=0.2):
        """Connect to FPGA. Auto-detects baud and firmware version if not specified."""
        import serial
        self.serial_mod = serial
        self.ser = None
        self.port = port
        self.baud = baud
        self.num_neurons = num_neurons
        self.fw_version = None  # 1 or 2, set after detection
        self.timeout = timeout

        if port and baud:
            self.ser = serial.Serial(port, baud, timeout=timeout)
            time.sleep(0.1)
            if num_neurons:
                self.num_neurons = num_neurons
                self.fw_version = 2 if num_neurons > V1_NEURONS else 1
            else:
                # Probe actual telemetry to detect neuron count
                self._detect_neuron_count()
        else:
            self._auto_connect()

        if self.ser:
            self.kill_switch(False)

    def _auto_connect(self):
        """Try v2 baud first (921600), then v1 (115200), on multiple ports."""
        import serial
        candidates = ['/dev/ttyUSB1', '/dev/ttyUSB0', '/dev/ttyUSB2']
        bauds = [V2_BAUD, V1_BAUD]

        for baud in bauds:
            for p in candidates:
                try:
                    s = serial.Serial(p, baud, timeout=0.15)
                    time.sleep(0.1)
                    # Try a telemetry read to verify connection
                    s.reset_input_buffer()
                    s.write(bytes([SYNC, CMD_SET_KILL, 0x00]))
                    s.flush()
                    time.sleep(0.05)
                    s.write(bytes([SYNC, CMD_READ_TELEM]))
                    s.flush()
                    time.sleep(0.3)
                    avail = s.in_waiting
                    resp = s.read(max(avail, 4))
                    if len(resp) >= 3 and resp[0] == SYNC and resp[1] == CMD_READ_TELEM:
                        self.ser = s
                        self.port = p
                        self.baud = baud
                        # Detect neuron count from response size
                        if len(resp) >= V2_TELEM_LEN:
                            payload_len = (resp[2] << 8) | resp[3]
                            self.num_neurons = payload_len // 6
                            self.fw_version = 2
                        elif len(resp) >= V1_TELEM_LEN:
                            self.num_neurons = V1_NEURONS
                            self.fw_version = 1
                        else:
                            self.num_neurons = V2_NEURONS
                            self.fw_version = 2
                        s.reset_input_buffer()
                        return
                    s.close()
                except Exception:
                    continue

        # Fallback: just open the first port that works at v2 baud
        for p in candidates:
            try:
                s = serial.Serial(p, V2_BAUD, timeout=self.timeout)
                time.sleep(0.1)
                self.ser = s
                self.port = p
                self.baud = V2_BAUD
                self.num_neurons = V2_NEURONS
                self.fw_version = 2
                return
            except Exception:
                continue

    def _detect_neuron_count(self):
        """Probe telemetry to detect actual neuron count from response size."""
        self.ser.reset_input_buffer()
        self.ser.write(bytes([SYNC, CMD_SET_KILL, 0x00]))
        self.ser.flush()
        time.sleep(0.05)
        self.ser.write(bytes([SYNC, CMD_READ_TELEM]))
        self.ser.flush()
        time.sleep(0.5)
        raw = self.ser.read(self.ser.in_waiting or 800)
        if len(raw) >= V2_TELEM_LEN and raw[0] == SYNC and raw[1] == CMD_READ_TELEM:
            payload_len = (raw[2] << 8) | raw[3]
            self.num_neurons = payload_len // 6
            self.fw_version = 2
        elif len(raw) >= V1_TELEM_LEN and raw[0] == SYNC and raw[1] == CMD_READ_TELEM:
            self.num_neurons = V1_NEURONS
            self.fw_version = 1
        else:
            # Fallback
            self.num_neurons = V2_NEURONS
            self.fw_version = 2
        self.ser.reset_input_buffer()

    @property
    def connected(self):
        return self.ser is not None and self.ser.is_open

    def reconnect(self):
        """Reconnect after serial failure."""
        if self.ser:
            try:
                self.ser.close()
            except Exception:
                pass
        try:
            self.ser = self.serial_mod.Serial(self.port, self.baud, timeout=self.timeout)
            time.sleep(0.1)
            self.kill_switch(False)
            return True
        except Exception:
            self.ser = None
            return False

    def close(self):
        if self.ser:
            try:
                self.kill_switch(True)
                self.ser.close()
            except Exception:
                pass
            self.ser = None

    # ─── Commands ───

    def kill_switch(self, enable: bool):
        """Set kill switch: True = all neurons silenced, False = normal operation."""
        self.ser.write(bytes([SYNC, CMD_SET_KILL, 0x01 if enable else 0x00]))
        self.ser.flush()
        time.sleep(0.005)

    def set_vg(self, neuron_id: int, vg: float):
        """Set gate voltage for a single neuron. vg in [0.0, 1.0]."""
        q16 = to_q16_16(max(0.0, min(1.0, vg)))
        nid = neuron_id & 0x7F
        payload = bytes([nid]) + struct.pack('>I', q16)
        self.ser.write(bytes([SYNC, CMD_SET_VG]) + payload)
        self.ser.flush()

    def set_vg_all(self, vg_values):
        """Set Vg for all neurons individually (works with both v1 and v2)."""
        n = min(len(vg_values), self.num_neurons)
        for nid in range(n):
            q16 = to_q16_16(max(0.0, min(1.0, vg_values[nid])))
            payload = bytes([nid & 0x7F]) + struct.pack('>I', q16)
            self.ser.write(bytes([SYNC, CMD_SET_VG]) + payload)
        self.ser.flush()
        time.sleep(0.005)

    def set_vg_batch(self, start_id: int, vg_values):
        """Bulk Vg update using CMD_SET_VG_BATCH (v2 only).

        Sends: [SYNC][0x08][start_id][count][vg0_BE..vgN_BE]
        Much faster than individual SET_VG for many neurons.
        """
        if self.fw_version == 1:
            # Fallback to individual SET_VG for v1 firmware
            for i, vg in enumerate(vg_values):
                self.set_vg(start_id + i, vg)
            return

        count = len(vg_values)
        if count == 0:
            return
        pkt = bytearray([SYNC, CMD_SET_VG_BATCH, start_id & 0x7F, count & 0xFF])
        for vg in vg_values:
            q16 = to_q16_16(max(0.0, min(1.0, vg)))
            pkt.extend(struct.pack('>I', q16))
        self.ser.write(bytes(pkt))
        self.ser.flush()
        time.sleep(0.005)

    def set_mac(self, mac_value: float):
        """Set MAC signal (feedback from GPU). mac_value in [0.0, 1.0]."""
        q16 = to_q16_16(max(0.0, min(1.0, mac_value)))
        self.ser.write(bytes([SYNC, CMD_SET_MAC]) + struct.pack('>I', q16))
        self.ser.flush()

    def set_synapse(self, neuron_id: int, weights_packed: int):
        """Set packed synapse weights for a neuron. weights_packed is 32-bit {w3,w2,w1,w0}."""
        nid = neuron_id & 0x7F
        self.ser.write(bytes([SYNC, CMD_SET_SYNAPSE, nid]) + struct.pack('>I', weights_packed))
        self.ser.flush()

    # ─── Telemetry ───

    def read_telem(self, timeout=None):
        """Read telemetry from all neurons.

        v1: [SYNC][0x02][0x30][48B][CRC8] = 52 bytes, 8 neurons × 6B
        v2: [SYNC][0x02][LEN_HI][LEN_LO][768B][CRC8] = 773 bytes, 128 neurons × 6B

        Returns list of dicts: [{'spike_count': int, 'vmem': float}, ...]
        """
        if timeout is None:
            timeout = 0.15 if self.fw_version == 1 else 0.5

        # Request telemetry
        self.ser.write(bytes([SYNC, CMD_READ_TELEM]))
        self.ser.flush()

        if self.fw_version == 1:
            return self._read_telem_v1(timeout)
        else:
            return self._read_telem_v2(timeout)

    def _read_telem_v1(self, timeout):
        """Parse v1 telemetry: 52 bytes, 8 neurons."""
        deadline = time.monotonic() + timeout
        buf = bytearray()
        while time.monotonic() < deadline:
            self.ser.timeout = max(0.001, deadline - time.monotonic())
            b = self.ser.read(1)
            if not b:
                continue
            if b[0] == SYNC:
                buf = bytearray([SYNC])
                while len(buf) < V1_TELEM_LEN and time.monotonic() < deadline:
                    self.ser.timeout = max(0.001, deadline - time.monotonic())
                    chunk = self.ser.read(V1_TELEM_LEN - len(buf))
                    if chunk:
                        buf.extend(chunk)
                break
        if len(buf) < V1_TELEM_LEN:
            return None
        payload = bytes(buf[3:51])
        neurons = []
        for i in range(V1_NEURONS):
            off = i * 6
            sc = struct.unpack_from('>H', payload, off)[0]
            vm = struct.unpack_from('>H', payload, off + 2)[0]
            neurons.append({'spike_count': sc, 'vmem': vm / 256.0})
        return neurons

    def _read_telem_v2(self, timeout):
        """Parse v2 telemetry: 773 bytes, 128 neurons."""
        deadline = time.monotonic() + timeout
        buf = bytearray()

        # Find SYNC byte
        while time.monotonic() < deadline:
            self.ser.timeout = max(0.001, deadline - time.monotonic())
            b = self.ser.read(1)
            if not b:
                continue
            if b[0] == SYNC:
                buf = bytearray([SYNC])
                # Read header: CMD + LEN_HI + LEN_LO
                while len(buf) < 4 and time.monotonic() < deadline:
                    self.ser.timeout = max(0.001, deadline - time.monotonic())
                    chunk = self.ser.read(4 - len(buf))
                    if chunk:
                        buf.extend(chunk)

                if len(buf) < 4:
                    return None
                if buf[1] != CMD_READ_TELEM:
                    continue  # Not a telemetry response

                payload_len = (buf[2] << 8) | buf[3]
                total_len = 4 + payload_len + 1  # header + payload + CRC

                while len(buf) < total_len and time.monotonic() < deadline:
                    self.ser.timeout = max(0.001, deadline - time.monotonic())
                    chunk = self.ser.read(total_len - len(buf))
                    if chunk:
                        buf.extend(chunk)
                break

        if len(buf) < 5:
            return None

        payload_len = (buf[2] << 8) | buf[3]
        total_len = 4 + payload_len + 1
        if len(buf) < total_len:
            return None

        payload = bytes(buf[4:4 + payload_len])
        received_crc = buf[4 + payload_len]
        # RTL CRCs header + payload (SYNC, CMD, LEN_HI, LEN_LO, then payload)
        computed_crc = crc8(bytes(buf[0:4 + payload_len]))

        if received_crc != computed_crc:
            return None  # CRC mismatch

        n_neurons = payload_len // 6
        neurons = []
        for i in range(n_neurons):
            off = i * 6
            if off + 6 > len(payload):
                break
            sc = struct.unpack_from('>H', payload, off)[0]
            vm = struct.unpack_from('>H', payload, off + 2)[0]
            ia = struct.unpack_from('>H', payload, off + 4)[0]
            neurons.append({
                'spike_count': sc,
                'vmem': vm / 256.0,
                'i_aval': ia / 256.0,
            })
        return neurons

    # ─── High-Level Helpers ───

    def safe_step(self, vg_values, interval=0.05):
        """Set Vg and read telemetry with automatic reconnection on serial error.

        Returns (neurons, elapsed_s) or (None, elapsed_s) on failure.
        """
        t0 = time.monotonic()
        try:
            if self.fw_version == 2 and len(vg_values) > 4:
                self.set_vg_batch(0, vg_values)
            else:
                self.set_vg_all(vg_values)

            elapsed = time.monotonic() - t0
            remaining = max(0.001, interval - elapsed)
            time.sleep(remaining)

            neurons = self.read_telem()
            return neurons, time.monotonic() - t0

        except Exception:
            self.reconnect()
            return None, time.monotonic() - t0

    def reset_spike_counts(self):
        """Reset all spike counters (handled automatically by telemetry read in v2)."""
        # v2 firmware resets counters on telemetry read
        # For explicit reset, just do a telemetry read and discard
        self.read_telem()

    def get_spike_rates(self, dt_s: float):
        """Read telemetry and return spike rates (Hz) for all neurons."""
        neurons = self.read_telem()
        if neurons is None:
            return None
        return [n['spike_count'] / dt_s for n in neurons]


# ─── Standalone Functions (backward-compatible with existing scripts) ───

def find_fpga(baud=None):
    """Find and connect to FPGA, returning (serial_obj, port) or (None, None).

    If baud is None, tries v2 (921600) first, then v1 (115200).
    """
    try:
        import serial
    except ImportError:
        return None, None

    bauds = [baud] if baud else [V2_BAUD, V1_BAUD]
    for b in bauds:
        for p in ['/dev/ttyUSB1', '/dev/ttyUSB0', '/dev/ttyUSB2']:
            try:
                s = serial.Serial(p, b, timeout=0.2)
                time.sleep(0.1)
                return s, p
            except Exception:
                continue
    return None, None


def reconnect_fpga(port, baud=V2_BAUD):
    """Reconnect to FPGA after serial failure."""
    import serial
    try:
        ser = serial.Serial(port, baud, timeout=0.2)
        time.sleep(0.1)
        ser.write(bytes([SYNC, CMD_SET_KILL, 0x00]))
        ser.flush()
        time.sleep(0.1)
        return ser
    except Exception:
        return None


def set_per_neuron_vg(ser, vg_values, num_neurons=None):
    """Set individual Vg for each neuron (standalone function)."""
    n = num_neurons or len(vg_values)
    for nid in range(min(n, len(vg_values))):
        vg = vg_values[nid]
        q16 = to_q16_16(max(0.0, min(1.0, vg)))
        payload = bytes([nid & 0x7F]) + struct.pack('>I', q16)
        ser.write(bytes([SYNC, CMD_SET_VG]) + payload)
    ser.flush()
    time.sleep(0.005)


def read_telem_v1(ser, timeout=0.15):
    """Read v1 telemetry: [0x55][0x02][0x30][48B][CRC8] = 52 bytes, 8 neurons."""
    deadline = time.monotonic() + timeout
    buf = bytearray()
    while time.monotonic() < deadline:
        ser.timeout = max(0.001, deadline - time.monotonic())
        b = ser.read(1)
        if not b:
            continue
        if b[0] == SYNC:
            buf = bytearray([SYNC])
            while len(buf) < V1_TELEM_LEN and time.monotonic() < deadline:
                ser.timeout = max(0.001, deadline - time.monotonic())
                chunk = ser.read(V1_TELEM_LEN - len(buf))
                if chunk:
                    buf.extend(chunk)
            break
    if len(buf) < V1_TELEM_LEN:
        return None
    payload = bytes(buf[3:51])
    neurons = []
    for i in range(V1_NEURONS):
        off = i * 6
        sc = struct.unpack_from('>H', payload, off)[0]
        vm = struct.unpack_from('>H', payload, off + 2)[0]
        neurons.append({'spike_count': sc, 'vmem': vm / 256.0})
    return neurons


def read_telem_v2(ser, timeout=0.5):
    """Read v2 telemetry: [0x55][0x02][LEN_HI][LEN_LO][768B][CRC8] = 773 bytes, 128 neurons."""
    deadline = time.monotonic() + timeout
    buf = bytearray()

    while time.monotonic() < deadline:
        ser.timeout = max(0.001, deadline - time.monotonic())
        b = ser.read(1)
        if not b:
            continue
        if b[0] == SYNC:
            buf = bytearray([SYNC])
            while len(buf) < 4 and time.monotonic() < deadline:
                ser.timeout = max(0.001, deadline - time.monotonic())
                chunk = ser.read(4 - len(buf))
                if chunk:
                    buf.extend(chunk)

            if len(buf) < 4 or buf[1] != CMD_READ_TELEM:
                buf = bytearray()
                continue

            payload_len = (buf[2] << 8) | buf[3]
            total_len = 4 + payload_len + 1

            while len(buf) < total_len and time.monotonic() < deadline:
                ser.timeout = max(0.001, deadline - time.monotonic())
                chunk = ser.read(total_len - len(buf))
                if chunk:
                    buf.extend(chunk)
            break

    if len(buf) < 5:
        return None

    payload_len = (buf[2] << 8) | buf[3]
    total_len = 4 + payload_len + 1
    if len(buf) < total_len:
        return None

    payload = bytes(buf[4:4 + payload_len])
    received_crc = buf[4 + payload_len]
    if crc8(payload) != received_crc:
        return None

    n_neurons = payload_len // 6
    neurons = []
    for i in range(n_neurons):
        off = i * 6
        if off + 6 > len(payload):
            break
        sc = struct.unpack_from('>H', payload, off)[0]
        vm = struct.unpack_from('>H', payload, off + 2)[0]
        ia = struct.unpack_from('>H', payload, off + 4)[0]
        neurons.append({
            'spike_count': sc,
            'vmem': vm / 256.0,
            'i_aval': ia / 256.0,
        })
    return neurons


def send_set_vg_batch(ser, start_id, vg_values):
    """Send CMD_SET_VG_BATCH: [SYNC][0x08][start_id][count][vg0..vgN big-endian]."""
    count = len(vg_values)
    if count == 0:
        return
    pkt = bytearray([SYNC, CMD_SET_VG_BATCH, start_id & 0x7F, count & 0xFF])
    for vg in vg_values:
        q16 = to_q16_16(max(0.0, min(1.0, vg)))
        pkt.extend(struct.pack('>I', q16))
    ser.write(bytes(pkt))
    ser.flush()
    time.sleep(0.005)


# ─── Self-Test ───

if __name__ == '__main__':
    print("fpga_host_v2 — NS-RAM FPGA Bridge Host Library")
    print("=" * 50)

    bridge = FPGABridge()
    if not bridge.connected:
        print("ERROR: No FPGA found on ttyUSB0/1/2")
        sys.exit(1)

    print(f"Connected: {bridge.port} @ {bridge.baud} baud (fw v{bridge.fw_version}, {bridge.num_neurons} neurons)")

    # Kill switch off
    bridge.kill_switch(False)
    print("Kill switch: OFF")

    # Set all neurons to moderate Vg
    vg_values = [0.55 + 0.005 * i for i in range(bridge.num_neurons)]
    if bridge.fw_version == 2:
        bridge.set_vg_batch(0, vg_values)
        print(f"Set {bridge.num_neurons} neurons via batch (Vg 0.55..{vg_values[-1]:.3f})")
    else:
        bridge.set_vg_all(vg_values)
        print(f"Set {bridge.num_neurons} neurons individually")

    # Wait for activity
    time.sleep(0.5)

    # Read telemetry
    neurons = bridge.read_telem()
    if neurons is None:
        print("ERROR: No telemetry response")
        bridge.close()
        sys.exit(1)

    print(f"\nTelemetry ({len(neurons)} neurons):")
    active = 0
    for i, n in enumerate(neurons):
        if n['spike_count'] > 0:
            active += 1
        if i < 16 or n['spike_count'] > 0:
            extra = f", i_aval={n['i_aval']:.2f}" if 'i_aval' in n else ""
            print(f"  N{i:3d}: spikes={n['spike_count']:5d}  vmem={n['vmem']:7.2f}{extra}")
    if len(neurons) > 16:
        print(f"  ... ({len(neurons) - 16} more neurons)")

    print(f"\nActive neurons (spike_count > 0): {active}/{len(neurons)}")

    # Kill switch test
    bridge.kill_switch(True)
    time.sleep(0.3)
    neurons2 = bridge.read_telem()
    if neurons2:
        killed = sum(1 for n in neurons2 if n['spike_count'] == 0)
        print(f"Kill switch ON: {killed}/{len(neurons2)} neurons at zero spikes")

    bridge.close()
    print("\nDone.")
