#!/usr/bin/env python3
"""NS-RAM FPGA Bridge -- Bidirectional GPU<->FPGA coupling for FEEL experiments."""

import serial
import struct
import time
import numpy as np
from collections import deque
from pathlib import Path


# ---------------------------------------------------------------------------
# Q16.16 helpers
# ---------------------------------------------------------------------------

def to_q16(val: float) -> int:
    """Convert float to Q16.16 unsigned 32-bit."""
    return int(val * 65536) & 0xFFFFFFFF


def from_q16(val: int) -> float:
    """Convert Q16.16 unsigned 32-bit to float."""
    return val / 65536.0


def crc8(data: bytes, poly: int = 0x07) -> int:
    """CRC-8 with polynomial 0x07."""
    crc = 0x00
    for byte in data:
        crc ^= byte
        for _ in range(8):
            if crc & 0x80:
                crc = ((crc << 1) ^ poly) & 0xFF
            else:
                crc = (crc << 1) & 0xFF
    return crc


# ---------------------------------------------------------------------------
# GPU sysfs telemetry
# ---------------------------------------------------------------------------

def read_gpu_temp_c() -> float:
    """Read AMD GPU junction temperature from sysfs hwmon."""
    for p in Path('/sys/class/hwmon').iterdir():
        try:
            name = (p / 'name').read_text().strip()
            if name == 'amdgpu':
                return int((p / 'temp1_input').read_text().strip()) / 1000.0
        except (FileNotFoundError, PermissionError, ValueError):
            continue
    return 0.0


def read_gpu_telemetry() -> dict:
    """Read extended AMD GPU telemetry from sysfs hwmon (same approach as z2134)."""
    result = {'temp_c': 0.0, 'power_w': 0.0, 'fan_rpm': 0, 'vddgfx_mv': 0}
    for p in Path('/sys/class/hwmon').iterdir():
        try:
            name = (p / 'name').read_text().strip()
        except (FileNotFoundError, PermissionError):
            continue
        if name != 'amdgpu':
            continue
        # Junction temperature
        try:
            result['temp_c'] = int((p / 'temp1_input').read_text().strip()) / 1000.0
        except (FileNotFoundError, PermissionError, ValueError):
            pass
        # Power (average)
        try:
            result['power_w'] = int((p / 'power1_average').read_text().strip()) / 1e6
        except (FileNotFoundError, PermissionError, ValueError):
            pass
        # Fan RPM
        try:
            result['fan_rpm'] = int((p / 'fan1_input').read_text().strip())
        except (FileNotFoundError, PermissionError, ValueError):
            pass
        # VDDGFX voltage
        try:
            result['vddgfx_mv'] = int((p / 'in0_input').read_text().strip())
        except (FileNotFoundError, PermissionError, ValueError):
            pass
        break
    return result


# ---------------------------------------------------------------------------
# NS-RAM FPGA Bridge
# ---------------------------------------------------------------------------

class NSRAMFPGABridge:
    """Bidirectional coupling between GPU FEEL model and FPGA NS-RAM emulator."""

    # Command IDs (host -> FPGA) — must match nsram_bridge_top.v
    CMD_SET_VG      = 0x01
    CMD_READ_TELEM  = 0x02
    CMD_SET_KILL    = 0x03
    CMD_SET_SYNAPSE = 0x04
    CMD_SET_TEMP    = 0x05
    CMD_SET_MAC     = 0x06

    # Response types (FPGA -> host)
    RESP_TELEM = 0x02

    # Sync byte — same in both directions (matches FPGA)
    SYNC = 0x55

    def __init__(self, port: str = '/dev/ttyUSB1', baudrate: int = 115200,
                 timeout: float = 0.5):
        self.ser = serial.Serial(port, baudrate, timeout=timeout)
        self.spike_history: deque = deque(maxlen=10000)
        self.telemetry_history: deque = deque(maxlen=1000)
        self.last_telemetry: dict | None = None
        time.sleep(0.1)  # let UART settle
        self.ser.reset_input_buffer()

    # ------------------------------------------------------------------
    # Low-level protocol
    # ------------------------------------------------------------------

    def _send_cmd(self, cmd: int, payload: bytes = b'') -> None:
        """Send command packet: [0x55][CMD][PAYLOAD] — no LEN, no CRC on TX."""
        pkt = bytes([self.SYNC, cmd]) + payload
        self.ser.write(pkt)
        self.ser.flush()

    def _read_response(self, timeout: float = 0.5) -> dict | None:
        """Read response packet: [0x55][0x02][0x30][48 bytes][CRC8] = 52 bytes.

        The FPGA response format is fixed:
          - Sync byte 0x55
          - Type 0x02 (telemetry)
          - Length 0x30 (48)
          - 48 bytes of neuron data
          - CRC-8 covering ALL preceding bytes (sync + type + len + payload)

        Returns None on timeout, framing error, or CRC mismatch.
        """
        deadline = time.monotonic() + timeout
        buf = bytearray()

        # Phase 1 --- find the sync byte 0x55
        while time.monotonic() < deadline:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return None
            self.ser.timeout = min(remaining, 0.05)
            b = self.ser.read(1)
            if not b:
                continue
            if b[0] == self.SYNC:
                buf.append(b[0])
                break
        else:
            return None

        # Phase 2 --- read TYPE and LEN (2 bytes)
        while len(buf) < 3 and time.monotonic() < deadline:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return None
            self.ser.timeout = min(remaining, 0.05)
            chunk = self.ser.read(3 - len(buf))
            if chunk:
                buf.extend(chunk)
        if len(buf) < 3:
            return None

        resp_type = buf[1]
        payload_len = buf[2]

        # Sanity-check: payload should never exceed 200 bytes in our protocol
        if payload_len > 200:
            return None

        # Phase 3 --- read PAYLOAD + CRC (payload_len + 1 bytes)
        need = payload_len + 1
        while len(buf) < 3 + need and time.monotonic() < deadline:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return None
            self.ser.timeout = min(remaining, 0.05)
            chunk = self.ser.read(3 + need - len(buf))
            if chunk:
                buf.extend(chunk)
        if len(buf) < 3 + need:
            return None

        # Phase 4 --- CRC validation (covers sync + type + len + payload)
        pkt_body = bytes(buf[:-1])  # everything except last byte (the CRC)
        rx_crc = buf[-1]
        if crc8(pkt_body) != rx_crc:
            return None

        payload = bytes(buf[3:3 + payload_len])
        return {'type': resp_type, 'len': payload_len, 'payload': payload}

    # ------------------------------------------------------------------
    # High-level commands
    # ------------------------------------------------------------------

    def set_temperature(self, temp_k: float) -> None:
        """Send GPU junction temperature to FPGA for BVpar modulation."""
        payload = struct.pack('>I', to_q16(temp_k))
        self._send_cmd(self.CMD_SET_TEMP, payload)

    def set_gate_voltage(self, neuron_id: int, vg: float) -> None:
        """Set gate voltage for a specific neuron."""
        payload = bytes([neuron_id & 0xFF]) + struct.pack('>I', to_q16(vg))
        self._send_cmd(self.CMD_SET_VG, payload)

    def set_mac_signal(self, mac_val: float) -> None:
        """Send GPU MAC output to FPGA."""
        payload = struct.pack('>I', to_q16(mac_val))
        self._send_cmd(self.CMD_SET_MAC, payload)

    def set_kill_switch(self, enabled: bool) -> None:
        """Hardware kill-shot: disable all avalanche physics."""
        self._send_cmd(self.CMD_SET_KILL, bytes([0x01 if enabled else 0x00]))

    def read_telemetry(self) -> dict | None:
        """Read bulk telemetry from all 8 neurons.

        Response payload layout (48 bytes total, 6 bytes per neuron):
          [spike_count_u16][vmem_q8.8][bvpar_q8.8]  x 8 neurons
        """
        self._send_cmd(self.CMD_READ_TELEM)
        resp = self._read_response(timeout=0.1)
        if resp is None or resp['type'] != self.RESP_TELEM:
            return None
        data = resp['payload']
        if len(data) < 48:
            return None

        neurons = []
        for i in range(8):
            off = i * 6
            sc = struct.unpack('>H', data[off:off + 2])[0]
            vm = struct.unpack('>H', data[off + 2:off + 4])[0]
            bv = struct.unpack('>H', data[off + 4:off + 6])[0]
            neurons.append({
                'spike_count': sc,
                'vmem': vm / 256.0,   # Q8.8 for compact telemetry
                'bvpar': bv / 256.0,
            })

        result = {
            'timestamp': time.time(),
            'neurons': neurons,
            'total_spikes': sum(n['spike_count'] for n in neurons),
            'mean_vmem': float(np.mean([n['vmem'] for n in neurons])),
            'mean_bvpar': float(np.mean([n['bvpar'] for n in neurons])),
        }
        self.telemetry_history.append(result)
        self.last_telemetry = result
        # Record per-spike timestamps for ISI analysis
        self.spike_history.append((result['timestamp'], result['total_spikes']))
        return result

    def set_synapse(self, neuron_id: int, syn_id: int, weight: float) -> None:
        """Set synapse weight for a specific neuron."""
        payload = bytes([neuron_id & 0xFF, syn_id & 0xFF]) + \
                  struct.pack('>I', to_q16(weight))
        self._send_cmd(self.CMD_SET_SYNAPSE, payload)

    # ------------------------------------------------------------------
    # Analysis helpers
    # ------------------------------------------------------------------

    def get_spike_rate_hz(self, window_s: float = 1.0) -> float:
        """Compute windowed spike rate from telemetry history."""
        now = time.time()
        recent = [t for t in self.telemetry_history
                  if now - t['timestamp'] < window_s]
        if len(recent) < 2:
            return 0.0
        total = sum(t['total_spikes'] for t in recent)
        dt = recent[-1]['timestamp'] - recent[0]['timestamp']
        return total / dt if dt > 0 else 0.0

    def get_isi_cv(self) -> float:
        """ISI coefficient of variation from recent spike data.

        Uses telemetry timestamps and spike counts to estimate inter-spike
        intervals, then returns CV = std(ISI) / mean(ISI).
        """
        if len(self.spike_history) < 3:
            return 0.0
        # Approximate ISI from telemetry-interval / spike-count
        isis = []
        items = list(self.spike_history)
        for i in range(1, len(items)):
            dt = items[i][0] - items[i - 1][0]
            spikes = items[i][1]
            if spikes > 0 and dt > 0:
                isi = dt / spikes
                isis.append(isi)
        if len(isis) < 2:
            return 0.0
        arr = np.array(isis)
        mean_isi = np.mean(arr)
        if mean_isi == 0:
            return 0.0
        return float(np.std(arr) / mean_isi)

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def close(self) -> None:
        if self.ser and self.ser.is_open:
            self.ser.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


# ---------------------------------------------------------------------------
# Main demo
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    print("NS-RAM FPGA Bridge Demo")
    print("=" * 50)

    gpu = read_gpu_telemetry()
    print(f"GPU telemetry: temp={gpu['temp_c']:.1f}C  "
          f"power={gpu['power_w']:.1f}W  vddgfx={gpu['vddgfx_mv']}mV")

    bridge = NSRAMFPGABridge()
    bridge.reset()
    bridge.set_regime('coupled')
    print("Regime: coupled | Polling 100 iterations @ 10 Hz\n")

    try:
        for i in range(100):
            t = read_gpu_temp_c()
            bridge.set_temperature(t + 273.15)   # Celsius -> Kelvin
            bridge.set_mac_signal(0.5)            # Nominal MAC
            time.sleep(0.1)
            telem = bridge.read_telemetry()
            if telem:
                print(f"[{i:3d}] T={t:.1f}C  "
                      f"Spikes={telem['total_spikes']:4d}  "
                      f"Rate={bridge.get_spike_rate_hz():.0f} Hz  "
                      f"BVpar={telem['mean_bvpar']:.3f}V  "
                      f"Vmem={telem['mean_vmem']:.3f}V  "
                      f"ISI_CV={bridge.get_isi_cv():.3f}")
            else:
                print(f"[{i:3d}] T={t:.1f}C  (no response)")
    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        bridge.close()
        print("Bridge closed.")
