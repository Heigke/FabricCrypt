#!/usr/bin/env python3
"""NS-RAM FPGA Bridge (Fast) -- 921600 baud with closed-loop support.

Upgraded bridge for fast_uart_bridge.v with:
- 921600 baud default (8x over 115200), auto-negotiation fallback
- CMD_CLOSED_LOOP (0x10): combined write+read in one round-trip
- Per-transaction latency instrumentation
- Thread-safe serial access
"""

import serial
import struct
import time
import threading
import numpy as np
from collections import deque
from pathlib import Path


# ---------------------------------------------------------------------------
# Q16.16 / Q8.8 helpers
# ---------------------------------------------------------------------------

def to_q16(val: float) -> int:
    """Convert float to Q16.16 unsigned 32-bit."""
    return int(val * 65536) & 0xFFFFFFFF


def from_q16(val: int) -> float:
    """Convert Q16.16 unsigned 32-bit to float."""
    return val / 65536.0


def to_q8_8(val: float) -> int:
    """Convert float to Q8.8 unsigned 16-bit."""
    return int(val * 256) & 0xFFFF


def from_q8_8(val: int) -> float:
    """Convert Q8.8 unsigned 16-bit to float."""
    return val / 256.0


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
    """Read extended AMD GPU telemetry from sysfs hwmon."""
    result = {'temp_c': 0.0, 'power_w': 0.0, 'fan_rpm': 0, 'vddgfx_mv': 0}
    for p in Path('/sys/class/hwmon').iterdir():
        try:
            name = (p / 'name').read_text().strip()
        except (FileNotFoundError, PermissionError):
            continue
        if name != 'amdgpu':
            continue
        try:
            result['temp_c'] = int((p / 'temp1_input').read_text().strip()) / 1000.0
        except (FileNotFoundError, PermissionError, ValueError):
            pass
        try:
            result['power_w'] = int((p / 'power1_average').read_text().strip()) / 1e6
        except (FileNotFoundError, PermissionError, ValueError):
            pass
        try:
            result['fan_rpm'] = int((p / 'fan1_input').read_text().strip())
        except (FileNotFoundError, PermissionError, ValueError):
            pass
        try:
            result['vddgfx_mv'] = int((p / 'in0_input').read_text().strip())
        except (FileNotFoundError, PermissionError, ValueError):
            pass
        break
    return result


# ---------------------------------------------------------------------------
# NS-RAM FPGA Bridge (Fast)
# ---------------------------------------------------------------------------

class NSRAMFPGABridgeFast:
    """High-speed bidirectional GPU<->FPGA coupling with closed-loop support.

    Wire protocol (matches fast_uart_bridge.v):
      TX (host->FPGA): [0x55][CMD][LEN][PAYLOAD...]
      RX (FPGA->host): [0x55][TYPE][LEN][PAYLOAD...][CRC8]
    """

    # Command IDs — must match fast_uart_bridge.v
    CMD_SET_VG      = 0x01
    CMD_READ_TELEM  = 0x02
    CMD_SET_KILL    = 0x03
    CMD_SET_MAC     = 0x06
    CMD_CLOSED_LOOP = 0x10
    CMD_SET_BAUD    = 0xF0
    CMD_PING        = 0xAA

    # Sync byte
    SYNC = 0x55

    # Baud rates
    BAUD_FAST = 921600
    BAUD_SLOW = 115200

    def __init__(self, port: str = '/dev/ttyUSB1',
                 baudrate: int = 921600,
                 timeout: float = 0.5,
                 auto_negotiate: bool = True):
        self._lock = threading.Lock()
        self._port = port
        self._timeout = timeout
        self._baudrate = baudrate

        # Telemetry history
        self.spike_history: deque = deque(maxlen=10000)
        self.telemetry_history: deque = deque(maxlen=1000)
        self.last_telemetry: dict | None = None

        # Latency tracking
        self.latency_history: deque = deque(maxlen=1000)
        self.last_latency_ms: float = 0.0

        # Open serial port
        self.ser = serial.Serial(port, baudrate, timeout=timeout)
        time.sleep(0.05)
        self.ser.reset_input_buffer()

        if auto_negotiate:
            self._auto_negotiate_baud()

    def _auto_negotiate_baud(self) -> None:
        """Try 921600 first; if ping fails, fall back to 115200."""
        if self._baudrate == self.BAUD_FAST:
            if self._try_ping():
                return  # 921600 works
            # Fall back to 115200
            self.ser.close()
            self._baudrate = self.BAUD_SLOW
            self.ser = serial.Serial(self._port, self.BAUD_SLOW, timeout=self._timeout)
            time.sleep(0.05)
            self.ser.reset_input_buffer()
            if self._try_ping():
                return  # 115200 works
        elif self._baudrate == self.BAUD_SLOW:
            if self._try_ping():
                return

    def _try_ping(self, retries: int = 3) -> bool:
        """Send ping command, return True if valid response received."""
        for _ in range(retries):
            try:
                self.ser.reset_input_buffer()
                pkt = bytes([self.SYNC, self.CMD_PING, 0x00])
                self.ser.write(pkt)
                self.ser.flush()
                resp = self._read_response_raw(timeout=0.2)
                if resp is not None and resp.get('type') == self.CMD_PING:
                    return True
            except (serial.SerialException, OSError):
                pass
            time.sleep(0.02)
        return False

    @property
    def active_baudrate(self) -> int:
        """Return the currently active baud rate."""
        return self._baudrate

    # ------------------------------------------------------------------
    # Low-level protocol (thread-safe)
    # ------------------------------------------------------------------

    def _send_cmd(self, cmd: int, payload: bytes = b'') -> None:
        """Send command packet: [0x55][CMD][LEN][PAYLOAD]."""
        pkt = bytes([self.SYNC, cmd, len(payload)]) + payload
        self.ser.write(pkt)
        self.ser.flush()

    def _read_response_raw(self, timeout: float = 0.5) -> dict | None:
        """Read response: [0x55][TYPE][LEN][PAYLOAD][CRC8].

        Returns dict with 'type', 'len', 'payload' or None on error.
        """
        deadline = time.monotonic() + timeout
        buf = bytearray()

        # Phase 1: find sync byte
        while time.monotonic() < deadline:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return None
            self.ser.timeout = min(remaining, 0.02)
            b = self.ser.read(1)
            if not b:
                continue
            if b[0] == self.SYNC:
                buf.append(b[0])
                break
        else:
            return None

        # Phase 2: read TYPE and LEN (2 bytes)
        while len(buf) < 3 and time.monotonic() < deadline:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return None
            self.ser.timeout = min(remaining, 0.02)
            chunk = self.ser.read(3 - len(buf))
            if chunk:
                buf.extend(chunk)
        if len(buf) < 3:
            return None

        resp_type = buf[1]
        payload_len = buf[2]

        if payload_len > 200:
            return None

        # Phase 3: read PAYLOAD + CRC
        need = payload_len + 1
        while len(buf) < 3 + need and time.monotonic() < deadline:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return None
            self.ser.timeout = min(remaining, 0.02)
            chunk = self.ser.read(3 + need - len(buf))
            if chunk:
                buf.extend(chunk)
        if len(buf) < 3 + need:
            return None

        # Phase 4: CRC validation
        pkt_body = bytes(buf[:-1])
        rx_crc = buf[-1]
        if crc8(pkt_body) != rx_crc:
            return None

        payload = bytes(buf[3:3 + payload_len])
        return {'type': resp_type, 'len': payload_len, 'payload': payload}

    def _send_and_recv(self, cmd: int, payload: bytes = b'',
                       timeout: float = 0.5) -> dict | None:
        """Thread-safe send command + read response with latency measurement."""
        with self._lock:
            self.ser.reset_input_buffer()
            t0 = time.monotonic()
            self._send_cmd(cmd, payload)
            resp = self._read_response_raw(timeout=timeout)
            t1 = time.monotonic()
            latency_ms = (t1 - t0) * 1000.0
            self.last_latency_ms = latency_ms
            self.latency_history.append(latency_ms)
            return resp

    # ------------------------------------------------------------------
    # High-level commands (preserved from NSRAMFPGABridge)
    # ------------------------------------------------------------------

    def ping(self) -> bool:
        """Send ping, return True if FPGA responds."""
        resp = self._send_and_recv(self.CMD_PING, timeout=0.2)
        return resp is not None and resp.get('type') == self.CMD_PING

    def set_gate_voltage(self, neuron_id: int, vg: float) -> None:
        """Set gate voltage for a specific neuron (Q8.8)."""
        vg_q88 = to_q8_8(vg)
        payload = bytes([neuron_id & 0x07, (vg_q88 >> 8) & 0xFF, vg_q88 & 0xFF])
        self._send_and_recv(self.CMD_SET_VG, payload, timeout=0.1)

    def set_temperature(self, temp_k: float) -> None:
        """Send GPU junction temperature to FPGA for BVpar modulation.

        Preserved from original bridge for backward compatibility.
        Uses CMD_SET_MAC with temperature-derived value.
        """
        # Map temperature to a MAC-like feedback signal
        # Normalize around 300K (27C): deviation / 100
        mac_val = (temp_k - 300.0) / 100.0
        self.set_mac_signal(mac_val)

    def set_mac_signal(self, mac_val: float) -> None:
        """Send GPU MAC output to FPGA (Q8.8)."""
        mac_q88 = to_q8_8(max(0.0, min(255.0, mac_val)))
        payload = bytes([(mac_q88 >> 8) & 0xFF, mac_q88 & 0xFF])
        self._send_and_recv(self.CMD_SET_MAC, payload, timeout=0.1)

    def set_kill_switch(self, enabled: bool) -> None:
        """Hardware kill-shot: disable all avalanche physics."""
        payload = bytes([0x01 if enabled else 0x00])
        self._send_and_recv(self.CMD_SET_KILL, payload, timeout=0.1)

    def set_kill_neuron(self, neuron_id: int, enabled: bool) -> None:
        """Per-neuron kill switch."""
        payload = bytes([neuron_id & 0x07, 0x01 if enabled else 0x00])
        self._send_and_recv(self.CMD_SET_KILL, payload, timeout=0.1)

    def read_telemetry(self) -> dict | None:
        """Read bulk telemetry from all 8 neurons.

        Response payload layout (48 bytes total, 6 bytes per neuron):
          [spike_count_u16][vmem_q8.8][bvpar_q8.8]  x 8 neurons
        """
        resp = self._send_and_recv(self.CMD_READ_TELEM, timeout=0.1)
        if resp is None or resp['type'] != self.CMD_READ_TELEM:
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
                'vmem': from_q8_8(vm),
                'bvpar': from_q8_8(bv),
            })

        result = {
            'timestamp': time.time(),
            'neurons': neurons,
            'total_spikes': sum(n['spike_count'] for n in neurons),
            'mean_vmem': float(np.mean([n['vmem'] for n in neurons])),
            'mean_bvpar': float(np.mean([n['bvpar'] for n in neurons])),
            'latency_ms': self.last_latency_ms,
        }
        self.telemetry_history.append(result)
        self.last_telemetry = result
        self.spike_history.append((result['timestamp'], result['total_spikes']))
        return result

    # ------------------------------------------------------------------
    # NEW: Closed-loop transaction
    # ------------------------------------------------------------------

    def closed_loop_step(self, mac_feedback: float, mode_byte: int = 0x01,
                         alpha: float | None = None,
                         target: float | None = None) -> dict | None:
        """Combined write+read in one UART round-trip.

        Sends MAC feedback + mode to FPGA and receives full telemetry back
        in a single transaction, halving the round-trip time vs separate
        set_mac + read_telemetry calls.

        Args:
            mac_feedback: GPU MAC output scalar (Q8.8, typically 0.0-255.0)
            mode_byte: Control bits. bit0 = closed-loop enable.
            alpha: Optional feedback gain (Q8.8). If None, FPGA keeps current.
            target: Optional target MAC value (Q8.8). If None, FPGA keeps current.

        Returns:
            dict with spike_counts, isi_cv, mean_vmem, fpga_temp_c, latency_ms
            or None on communication error.
        """
        mac_q88 = to_q8_8(max(0.0, min(255.0, mac_feedback)))
        payload = bytes([
            (mac_q88 >> 8) & 0xFF,
            mac_q88 & 0xFF,
            mode_byte & 0xFF,
        ])

        # Optional alpha and target
        if alpha is not None:
            alpha_q88 = to_q8_8(max(0.0, min(255.0, alpha)))
            payload += bytes([(alpha_q88 >> 8) & 0xFF, alpha_q88 & 0xFF])
            if target is not None:
                target_q88 = to_q8_8(max(0.0, min(255.0, target)))
                payload += bytes([(target_q88 >> 8) & 0xFF, target_q88 & 0xFF])

        resp = self._send_and_recv(self.CMD_CLOSED_LOOP, payload, timeout=0.1)
        if resp is None or resp['type'] != self.CMD_CLOSED_LOOP:
            return None

        data = resp['payload']
        if len(data) < 22:
            return None

        # Parse response: 8x spike_count(2) + isi_cv(2) + mean_vmem(2) + temp(2) = 22 bytes
        spike_counts = []
        for i in range(8):
            sc = struct.unpack('>H', data[i * 2:i * 2 + 2])[0]
            spike_counts.append(sc)

        isi_cv_raw = struct.unpack('>H', data[16:18])[0]
        mean_vmem_raw = struct.unpack('>H', data[18:20])[0]
        fpga_temp_raw = struct.unpack('>H', data[20:22])[0]

        result = {
            'timestamp': time.time(),
            'spike_counts': spike_counts,
            'total_spikes': sum(spike_counts),
            'isi_cv': from_q8_8(isi_cv_raw),
            'mean_vmem': from_q8_8(mean_vmem_raw),
            'fpga_temp_c': fpga_temp_raw / 100.0,  # temp_celsius_x100
            'latency_ms': self.last_latency_ms,
            'mode_byte': mode_byte,
        }
        self.telemetry_history.append(result)
        self.last_telemetry = result
        self.spike_history.append((result['timestamp'], result['total_spikes']))
        return result

    # ------------------------------------------------------------------
    # Baud rate control
    # ------------------------------------------------------------------

    def switch_baud(self, fast: bool = True) -> bool:
        """Request FPGA to switch baud rate, then reconnect at new rate.

        Args:
            fast: True for 921600, False for 115200.

        Returns:
            True if switch succeeded (ping at new rate works).
        """
        target_baud = self.BAUD_FAST if fast else self.BAUD_SLOW
        baud_code = 0x01 if fast else 0x00

        with self._lock:
            self._send_cmd(self.CMD_SET_BAUD, bytes([baud_code]))
            # Read ACK at current baud
            resp = self._read_response_raw(timeout=0.2)
            if resp is None or resp['type'] != self.CMD_SET_BAUD:
                return False

            # Close and reopen at new baud
            self.ser.close()
            time.sleep(0.05)  # Let FPGA switch
            self.ser = serial.Serial(self._port, target_baud, timeout=self._timeout)
            time.sleep(0.05)
            self.ser.reset_input_buffer()
            self._baudrate = target_baud

        # Verify with ping
        return self.ping()

    # ------------------------------------------------------------------
    # Analysis helpers (preserved from NSRAMFPGABridge)
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
        """ISI coefficient of variation from recent spike data."""
        if len(self.spike_history) < 3:
            return 0.0
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

    def get_latency_stats(self) -> dict:
        """Return latency statistics for recent transactions."""
        if not self.latency_history:
            return {'mean_ms': 0.0, 'min_ms': 0.0, 'max_ms': 0.0,
                    'p50_ms': 0.0, 'p99_ms': 0.0, 'count': 0}
        arr = np.array(list(self.latency_history))
        return {
            'mean_ms': float(np.mean(arr)),
            'min_ms': float(np.min(arr)),
            'max_ms': float(np.max(arr)),
            'p50_ms': float(np.percentile(arr, 50)),
            'p99_ms': float(np.percentile(arr, 99)),
            'count': len(arr),
        }

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def close(self) -> None:
        with self._lock:
            if self.ser and self.ser.is_open:
                self.ser.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


# ---------------------------------------------------------------------------
# Main demo — closed-loop at 921600 baud
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    print("NS-RAM FPGA Bridge (Fast) Demo")
    print("=" * 60)

    gpu = read_gpu_telemetry()
    print(f"GPU: temp={gpu['temp_c']:.1f}C  power={gpu['power_w']:.1f}W  "
          f"vddgfx={gpu['vddgfx_mv']}mV")

    bridge = NSRAMFPGABridgeFast(auto_negotiate=True)
    print(f"Connected at {bridge.active_baudrate} baud")
    print()

    # Verify connectivity
    if not bridge.ping():
        print("ERROR: FPGA not responding to ping!")
        bridge.close()
        exit(1)

    print("Closed-loop demo: 200 iterations")
    print(f"{'iter':>4s}  {'T_gpu':>6s}  {'spikes':>6s}  {'rate_Hz':>8s}  "
          f"{'isi_cv':>6s}  {'vmem':>6s}  {'fpga_T':>6s}  {'lat_ms':>7s}")
    print("-" * 65)

    try:
        for i in range(200):
            t_gpu = read_gpu_temp_c()
            # MAC feedback ~ normalized GPU temperature
            mac = max(0.0, (t_gpu - 25.0) / 50.0)

            telem = bridge.closed_loop_step(
                mac_feedback=mac,
                mode_byte=0x01,  # closed-loop enabled
                alpha=0.0625,
                target=0.5,
            )

            if telem:
                print(f"[{i:3d}]  {t_gpu:5.1f}C  {telem['total_spikes']:5d}  "
                      f"{bridge.get_spike_rate_hz():7.0f}  "
                      f"{telem['isi_cv']:5.3f}  "
                      f"{telem['mean_vmem']:5.2f}  "
                      f"{telem['fpga_temp_c']:5.1f}  "
                      f"{telem['latency_ms']:6.2f}")
            else:
                print(f"[{i:3d}]  {t_gpu:5.1f}C  (no response)")

            time.sleep(0.005)  # ~200 Hz polling at 921600

    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        stats = bridge.get_latency_stats()
        print()
        print(f"Latency stats ({stats['count']} transactions):")
        print(f"  mean={stats['mean_ms']:.2f}ms  "
              f"p50={stats['p50_ms']:.2f}ms  "
              f"p99={stats['p99_ms']:.2f}ms  "
              f"min={stats['min_ms']:.2f}ms  "
              f"max={stats['max_ms']:.2f}ms")
        bridge.close()
        print("Bridge closed.")
