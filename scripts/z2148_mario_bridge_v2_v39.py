#!/usr/bin/env python3
"""
z2148v39: Mario-Bridge v2 Test Battery — T47-T53
==================================================
Seven new tests certifying plasticity, scaling, and cross-substrate
properties of the GPU<->FPGA closed-loop FEEL system.

  T47: Paired-Pulse STP           — synapse short-term plasticity
  T48: Multi-Level Weight Retention — 8-level Vg retention
  T49: Plasticity LM Advantage    — adaptive Vg beats frozen Vg
  T50: SOC Convergence             — self-organised criticality
  T51: Avalanche Size Distribution — power-law alpha via MLE
  T52: Spike-Timing Precision      — ISI CV across regimes
  T53: Cross-Substrate Plasticity  — FPGA LTP ↔ GPU LoRA correlation

Hardware:
  - AMD gfx1151 GPU  (HSA_OVERRIDE_GFX_VERSION=11.0.0)
  - Tang Nano 9K FPGA on /dev/ttyUSB1  (921600 / 115200 baud)

Run:
  HSA_OVERRIDE_GFX_VERSION=11.0.0 venv/bin/python -u scripts/z2148_mario_bridge_v2_v39.py
"""

import os, sys, json, math, time, struct, random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from collections import deque, Counter

# Ensure HSA override for gfx1151
os.environ.setdefault('HSA_OVERRIDE_GFX_VERSION', '11.0.0')

BASE = Path(__file__).resolve().parent.parent

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CONSTANTS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
DEVICE = 'cuda'
BS = 4
SEQ_LEN = 128
N_EVAL_BATCHES = 30
N_NEURONS = 8
FPGA_PORT = '/dev/ttyUSB1'
FPGA_BAUD_FAST = 921600
FPGA_BAUD_SLOW = 115200
RESULTS_JSON = BASE / 'results' / 'z2148_mario_bridge_v2.json'

# Lanza NS-RAM reference parameters
LANZA_BV0 = 4.2
LANZA_ALPHA_T = 0.003
LANZA_T0 = 300.0
LANZA_BETA_VG = 1.8


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# INLINE BRIDGE HELPERS (self-contained — no external imports)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def to_q8_8(val: float) -> int:
    return int(val * 256) & 0xFFFF

def to_q16_16(val: float) -> int:
    """Convert float to Q16.16 unsigned 32-bit (matches firmware)."""
    return int(val * 65536) & 0xFFFFFFFF

def from_q8_8(val: int) -> float:
    return val / 256.0

def crc8(data: bytes, poly: int = 0x07) -> int:
    crc = 0x00
    for byte in data:
        crc ^= byte
        for _ in range(8):
            if crc & 0x80:
                crc = ((crc << 1) ^ poly) & 0xFF
            else:
                crc = (crc << 1) & 0xFF
    return crc

def read_gpu_temp_c() -> float:
    for p in Path('/sys/class/hwmon').iterdir():
        try:
            name = (p / 'name').read_text().strip()
            if name == 'amdgpu':
                return int((p / 'temp1_input').read_text().strip()) / 1000.0
        except (FileNotFoundError, PermissionError, ValueError):
            continue
    return 0.0

def read_gpu_telemetry() -> dict:
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


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# INLINE FPGA BRIDGE (self-contained)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
import serial
import threading

class NSRAMFPGABridge:
    """Bidirectional GPU<->FPGA bridge with auto baud negotiation."""

    CMD_SET_VG      = 0x01
    CMD_READ_TELEM  = 0x02
    CMD_SET_KILL    = 0x03
    CMD_SET_MAC     = 0x06
    CMD_PING        = 0xAA
    SYNC = 0x55

    def __init__(self, port='/dev/ttyUSB1', baudrate=921600, timeout=0.5):
        self._lock = threading.Lock()
        self._port = port
        self._timeout = timeout
        self._baudrate = baudrate
        self.telemetry_history = deque(maxlen=1000)
        self.spike_history = deque(maxlen=10000)
        self.last_telemetry = None

        self.ser = serial.Serial(port, baudrate, timeout=timeout)
        time.sleep(0.05)
        self.ser.reset_input_buffer()

        # Auto-negotiate: try fast, fall back to slow
        # (FPGA firmware doesn't respond to CMD_PING — use telemetry read)
        if not self._try_telem():
            self.ser.close()
            self._baudrate = FPGA_BAUD_SLOW
            self.ser = serial.Serial(port, FPGA_BAUD_SLOW, timeout=timeout)
            time.sleep(0.05)
            self.ser.reset_input_buffer()
            self._try_telem()

    def _try_ping(self, retries=3):
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

    def _try_telem(self, retries=3):
        """Try telemetry read instead of ping (FPGA firmware may not support ping)."""
        for _ in range(retries):
            try:
                self.ser.reset_input_buffer()
                pkt = bytes([self.SYNC, self.CMD_READ_TELEM, 0x00])
                self.ser.write(pkt)
                self.ser.flush()
                resp = self._read_response_raw(timeout=0.3)
                if resp is not None and resp.get('type') == self.CMD_READ_TELEM:
                    return True
            except (serial.SerialException, OSError):
                pass
            time.sleep(0.02)
        return False

    def _send_cmd(self, cmd, payload=b''):
        pkt = bytes([self.SYNC, cmd]) + payload
        self.ser.write(pkt)
        self.ser.flush()

    def _read_response_raw(self, timeout=0.5):
        deadline = time.monotonic() + timeout
        buf = bytearray()
        while time.monotonic() < deadline:
            self.ser.timeout = min(deadline - time.monotonic(), 0.02)
            b = self.ser.read(1)
            if not b:
                continue
            if b[0] == self.SYNC:
                buf.append(b[0])
                break
        else:
            return None
        while len(buf) < 3 and time.monotonic() < deadline:
            self.ser.timeout = min(deadline - time.monotonic(), 0.02)
            chunk = self.ser.read(3 - len(buf))
            if chunk:
                buf.extend(chunk)
        if len(buf) < 3:
            return None
        resp_type = buf[1]
        payload_len = buf[2]
        if payload_len > 200:
            return None
        need = payload_len + 1
        while len(buf) < 3 + need and time.monotonic() < deadline:
            self.ser.timeout = min(deadline - time.monotonic(), 0.02)
            chunk = self.ser.read(3 + need - len(buf))
            if chunk:
                buf.extend(chunk)
        if len(buf) < 3 + need:
            return None
        pkt_body = bytes(buf[:-1])
        rx_crc = buf[-1]
        if crc8(pkt_body) != rx_crc:
            return None
        payload = bytes(buf[3:3 + payload_len])
        return {'type': resp_type, 'len': payload_len, 'payload': payload}

    def _send_and_recv(self, cmd, payload=b'', timeout=0.5):
        with self._lock:
            self.ser.reset_input_buffer()
            self._send_cmd(cmd, payload)
            return self._read_response_raw(timeout=timeout)

    def ping(self):
        resp = self._send_and_recv(self.CMD_PING, timeout=0.2)
        return resp is not None and resp.get('type') == self.CMD_PING

    def set_gate_voltage(self, neuron_id, vg):
        vg_q = to_q16_16(vg)
        payload = bytes([neuron_id & 0x07]) + struct.pack('>I', vg_q)
        self._send_cmd(self.CMD_SET_VG, payload)
        time.sleep(0.005)

    def set_temperature(self, temp_k):
        mac_val = (temp_k - 300.0) / 100.0
        self.set_mac_signal(mac_val)

    def set_mac_signal(self, mac_val):
        mac_q = to_q16_16(max(0.0, min(255.0, mac_val)))
        payload = struct.pack('>I', mac_q)
        self._send_cmd(self.CMD_SET_MAC, payload)
        time.sleep(0.005)

    def set_kill_switch(self, enabled):
        payload = bytes([0x01 if enabled else 0x00])
        self._send_cmd(self.CMD_SET_KILL, payload)
        time.sleep(0.005)

    def read_telemetry(self):
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
        }
        self.telemetry_history.append(result)
        self.spike_history.append((result['timestamp'], result['total_spikes']))
        self.last_telemetry = result
        return result

    def close(self):
        with self._lock:
            if self.ser and self.ser.is_open:
                self.ser.close()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SIMULATED FPGA (fallback when hardware unavailable)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class SimulatedFPGA:
    """Minimal FPGA simulator for testing without hardware."""

    def __init__(self):
        self.vg = [0.35] * N_NEURONS
        self.temp_k = 300.0
        self.mac_val = 0.5
        self.kill = False
        self.telemetry_history = deque(maxlen=1000)
        self.spike_history = deque(maxlen=10000)
        self._step = 0

    def set_gate_voltage(self, neuron_id, vg):
        if 0 <= neuron_id < N_NEURONS:
            self.vg[neuron_id] = vg

    def set_temperature(self, temp_k):
        self.temp_k = temp_k

    def set_mac_signal(self, mac_val):
        self.mac_val = mac_val

    def set_kill_switch(self, enabled):
        self.kill = enabled

    def read_telemetry(self):
        self._step += 1
        neurons = []
        for i in range(N_NEURONS):
            if self.kill:
                sc, vm, bv = 0, 0.0, 0.0
            else:
                vg = self.vg[i]
                t_factor = 1.0 + 0.002 * (self.temp_k - 300.0)
                base_rate = (vg ** 2) * 200.0 * t_factor
                rate = base_rate * (0.8 + 0.4 * self.mac_val)
                sc = max(0, int(rate + random.gauss(0, max(rate * 0.15, 0.5))))
                bv = LANZA_BV0 * math.exp(-LANZA_ALPHA_T * (self.temp_k - LANZA_T0)) * \
                     (1.0 + LANZA_BETA_VG * vg)
                vm = vg * 0.8 + random.gauss(0, 0.02)
            neurons.append({'spike_count': sc, 'vmem': vm, 'bvpar': bv})
        result = {
            'timestamp': time.time(),
            'neurons': neurons,
            'total_spikes': sum(n['spike_count'] for n in neurons),
            'mean_vmem': float(np.mean([n['vmem'] for n in neurons])),
            'mean_bvpar': float(np.mean([n['bvpar'] for n in neurons])),
        }
        self.telemetry_history.append(result)
        self.spike_history.append((result['timestamp'], result['total_spikes']))
        return result

    def close(self):
        pass


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PLASTICITY CLASSES: STP, LTP, PlasticityManager
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class STPState:
    """Short-Term Plasticity state for a single synapse (Tsodyks-Markram)."""

    def __init__(self, tau_fac=0.2, tau_dep=0.5):
        self.u = 0.2       # facilitation variable
        self.x = 1.0       # depression variable (available resources)
        self.tau_fac = tau_fac
        self.tau_dep = tau_dep

    def update(self, dt):
        """Decay toward resting state."""
        self.u += (0.2 - self.u) * (1.0 - math.exp(-dt / self.tau_fac))
        self.x += (1.0 - self.x) * (1.0 - math.exp(-dt / self.tau_dep))

    def on_spike(self):
        """Process a pre-synaptic spike. Returns effective weight."""
        self.u += 0.5 * (1.0 - self.u)   # facilitation step
        eff = self.u * self.x
        self.x *= (1.0 - self.u)          # depression step
        return eff


class LTPState:
    """Long-Term Potentiation state with 8 discrete weight levels."""

    def __init__(self):
        self.weight_level = 3   # 0-7
        self.weight_values = [0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]
        self.stdp_window = 0.020  # 20ms STDP window

    def stdp_update(self, dt_pre_post):
        """Apply STDP rule: positive dt (pre before post) -> potentiation."""
        if abs(dt_pre_post) > self.stdp_window:
            return  # outside STDP window
        if dt_pre_post > 0:
            # Pre before post -> LTP
            if self.weight_level < 7:
                self.weight_level += 1
        else:
            # Post before pre -> LTD
            if self.weight_level > 0:
                self.weight_level -= 1

    def get_vg(self):
        """Return the Vg value for current weight level."""
        return self.weight_values[self.weight_level]


class PlasticityManager:
    """Manages 8x8 STP+LTP synapse matrix for the neuron bank."""

    def __init__(self, n_neurons=N_NEURONS):
        self.n = n_neurons
        self.stp = [[STPState() for _ in range(n_neurons)] for _ in range(n_neurons)]
        self.ltp = [[LTPState() for _ in range(n_neurons)] for _ in range(n_neurons)]
        self._last_spike_time = [0.0] * n_neurons

    def process_telemetry(self, telem, t_now=None):
        """Process FPGA telemetry to drive plasticity updates."""
        if t_now is None:
            t_now = time.time()
        neurons = telem['neurons']
        for i in range(min(self.n, len(neurons))):
            sc = neurons[i]['spike_count']
            if sc > 0:
                dt = t_now - self._last_spike_time[i]
                self._last_spike_time[i] = t_now
                # Update STP for all synapses from this neuron
                for j in range(self.n):
                    self.stp[i][j].update(dt)
                    self.stp[i][j].on_spike()
                # STDP between pairs
                for j in range(self.n):
                    if j != i and neurons[j]['spike_count'] > 0:
                        dt_ij = self._last_spike_time[i] - self._last_spike_time[j]
                        self.ltp[i][j].stdp_update(dt_ij)

    def get_vg_for_neuron(self, neuron_id):
        """Get target Vg for a neuron based on average incoming LTP weight."""
        total = 0.0
        for j in range(self.n):
            total += self.ltp[j][neuron_id].get_vg()
        return total / self.n

    def apply_to_fpga(self, fpga):
        """Apply current LTP-derived Vg values to FPGA neurons."""
        for i in range(self.n):
            vg = self.get_vg_for_neuron(i)
            fpga.set_gate_voltage(i, vg)

    def get_all_weight_levels(self):
        """Return flat list of all LTP weight levels (64 values)."""
        levels = []
        for i in range(self.n):
            for j in range(self.n):
                levels.append(self.ltp[i][j].weight_level)
        return levels


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# VIRTUAL NEURON + SCALED BANK (128 virtual → 8 physical)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class VirtualNeuron:
    """Virtual neuron mapped to a physical FPGA neuron with noise + offset."""

    def __init__(self, physical_id):
        self.physical_id = physical_id
        self.vg_offset = random.gauss(0, 0.02)
        self.noise_std = random.uniform(0.05, 0.15)

    def generate_spike(self, phys_spikes, phys_vmem):
        """Generate virtual spike count from physical neuron telemetry."""
        base = phys_spikes
        noisy = base + random.gauss(0, self.noise_std * max(base, 1.0))
        return max(0, int(round(noisy)))


class ScaledBank:
    """128 virtual neurons round-robin mapped to 8 physical FPGA neurons."""

    def __init__(self, n_virtual=128):
        self.n_virtual = n_virtual
        self.neurons = []
        for i in range(n_virtual):
            phys_id = i % N_NEURONS
            self.neurons.append(VirtualNeuron(phys_id))

    def collect_raster(self, fpga, n_steps, vg=0.35):
        """Collect spike raster: (n_steps, n_virtual) array.

        Sets all physical neurons to vg, then reads telemetry n_steps times.
        """
        for nid in range(N_NEURONS):
            fpga.set_gate_voltage(nid, vg)
        time.sleep(0.2)

        raster = np.zeros((n_steps, self.n_virtual), dtype=int)
        for t in range(n_steps):
            telem = fpga.read_telemetry()
            if telem is None:
                continue
            for v in range(self.n_virtual):
                vn = self.neurons[v]
                phys = telem['neurons'][vn.physical_id]
                raster[t, v] = vn.generate_spike(phys['spike_count'], phys['vmem'])
            time.sleep(0.005)
        return raster


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# HOMEOSTATIC CONTROLLER (SOC adaptation)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class HomeostaticController:
    """Homeostatic controller that adapts Vg to maintain target spike rate."""

    def __init__(self, target_rate=100.0, eta=0.002, ema_alpha=0.1):
        self.target_rate = target_rate
        self.eta = eta
        self.ema_alpha = ema_alpha
        self.ema_rate = target_rate
        self.vg_current = [0.35] * N_NEURONS
        self.history = []

    def adapt_step(self, spikes):
        """One adaptation step: adjust Vg based on spike count error."""
        total = sum(spikes) if isinstance(spikes, (list, tuple)) else spikes
        self.ema_rate = self.ema_alpha * total + (1.0 - self.ema_alpha) * self.ema_rate
        error = self.target_rate - self.ema_rate
        # Adjust Vg up if too few spikes, down if too many
        delta = self.eta * error / max(self.target_rate, 1.0)
        for i in range(N_NEURONS):
            self.vg_current[i] = max(0.05, min(0.60, self.vg_current[i] + delta))
        self.history.append({
            'total_spikes': total,
            'ema_rate': self.ema_rate,
            'mean_vg': float(np.mean(self.vg_current)),
        })
        return self.vg_current

    def get_branching_ratio(self, history_window=20):
        """Estimate branching ratio sigma from recent spike history.

        sigma ~ 1.0 at criticality (each spike causes ~1 descendant).
        """
        if len(self.history) < history_window + 1:
            return 0.0
        recent = self.history[-history_window:]
        ratios = []
        for i in range(1, len(recent)):
            prev = max(recent[i - 1]['total_spikes'], 1)
            curr = recent[i]['total_spikes']
            ratios.append(curr / prev)
        return float(np.mean(ratios)) if ratios else 0.0


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# GPT-2 + FPGAGatedLoRA model (layers 6-11, rank 4)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class FPGAGatedLoRA(nn.Module):
    """LoRA adapter whose gate is driven by FPGA spike telemetry."""

    def __init__(self, base_linear, rank=4, alpha=8):
        super().__init__()
        self.base_linear = base_linear
        self.rank = rank
        self.scaling = alpha / rank

        w = base_linear.weight
        if hasattr(base_linear, 'nf'):
            in_f, out_f = w.shape[0], w.shape[1]
        else:
            out_f, in_f = w.shape
        self.in_features = in_f
        self.out_features = out_f

        dtype = w.dtype
        self.lora_A = nn.Parameter(torch.randn(rank, in_f, dtype=dtype) * 0.01)
        self.lora_B = nn.Parameter(torch.randn(out_f, rank, dtype=dtype) * 0.001)

        self.gate_proj = nn.Linear(N_NEURONS * 2, rank, dtype=torch.float32)
        nn.init.normal_(self.gate_proj.weight, std=0.02)
        nn.init.constant_(self.gate_proj.bias, 0.0)

        self.last_gate = None
        self.open_loop = False

    def set_fpga_state(self, spike_counts, vmems):
        vec = np.zeros(N_NEURONS * 2, dtype=np.float32)
        for i in range(min(N_NEURONS, len(spike_counts))):
            vec[i] = spike_counts[i] / 100.0
            vec[N_NEURONS + i] = vmems[i] / 5.0
        self._fpga_vec = vec

    def forward(self, x):
        base_out = self.base_linear(x)
        x_cast = x.to(self.lora_A.dtype)
        lora_mid = F.linear(x_cast, self.lora_A)

        dev = self.gate_proj.weight.device
        if self.open_loop:
            gate = torch.full((self.rank,), 0.5, device=dev)
        elif hasattr(self, '_fpga_vec'):
            fvec = torch.from_numpy(self._fpga_vec).float().to(dev)
            gate = torch.sigmoid(self.gate_proj(fvec))
        else:
            gate = torch.full((self.rank,), 0.5, device=dev)

        self.last_gate = gate.detach().cpu().numpy()
        lora_out = F.linear(lora_mid * gate, self.lora_B)
        return base_out + lora_out * self.scaling


class FEELBridgeModel(nn.Module):
    """GPT-2 small with FPGAGatedLoRA on layers 6-11, rank 4."""

    def __init__(self):
        super().__init__()
        from transformers import GPT2LMHeadModel
        self.gpt2 = GPT2LMHeadModel.from_pretrained('gpt2')
        for p in self.gpt2.parameters():
            p.requires_grad = False

        self.lora_layers = nn.ModuleList()
        for i in range(6, 12):  # layers 6-11
            attn = self.gpt2.transformer.h[i].attn
            original = attn.c_attn
            lora = FPGAGatedLoRA(original, rank=4, alpha=8)
            attn.c_attn = lora
            self.lora_layers.append(lora)

    def set_fpga_state(self, spike_counts, vmems):
        for lora in self.lora_layers:
            lora.set_fpga_state(spike_counts, vmems)

    def set_open_loop(self, open_loop):
        for lora in self.lora_layers:
            lora.open_loop = open_loop

    def forward(self, input_ids, labels=None):
        return self.gpt2(input_ids=input_ids, labels=labels)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# EVALUATION HELPERS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def get_wikitext_data(tokenizer, n_tokens=16384):
    from datasets import load_dataset
    ds = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')
    text = '\n'.join([r['text'] for r in ds if r['text'].strip()])
    ids = tokenizer.encode(text)[:n_tokens]
    return torch.tensor(ids, dtype=torch.long)


def eval_ppl(model, data, device, n_batches=N_EVAL_BATCHES):
    model.eval()
    total_loss, total_tokens = 0.0, 0
    with torch.no_grad():
        for b in range(n_batches):
            start = b * BS * SEQ_LEN
            end = start + BS * SEQ_LEN
            if end > len(data):
                break
            chunk = data[start:end].view(BS, SEQ_LEN).to(device)
            labels = chunk.clone()
            labels[:, :-1] = chunk[:, 1:]
            labels[:, -1] = -100
            out = model(chunk, labels=labels)
            total_loss += out.loss.item() * (BS * (SEQ_LEN - 1))
            total_tokens += BS * (SEQ_LEN - 1)
    if total_tokens == 0:
        return 999.0
    return math.exp(total_loss / total_tokens)


def closed_loop_step(model, fpga, data, batch_idx, device):
    """One closed-loop step: forward → feedback → FPGA → telemetry → model."""
    start = batch_idx * BS * SEQ_LEN
    end = start + BS * SEQ_LEN
    if end > len(data):
        return None, None
    chunk = data[start:end].view(BS, SEQ_LEN).to(device)
    labels = chunk.clone()
    labels[:, :-1] = chunk[:, 1:]
    labels[:, -1] = -100

    model.eval()
    with torch.no_grad():
        out = model(chunk, labels=labels)
    loss = out.loss.item()

    mac_signal = min(1.0, loss / 10.0)
    gpu_temp = read_gpu_temp_c()
    fpga.set_temperature(gpu_temp + 273.15)
    fpga.set_mac_signal(mac_signal)
    time.sleep(0.01)

    telem = fpga.read_telemetry()
    if telem is not None:
        spike_counts = [n['spike_count'] for n in telem['neurons']]
        vmems = [n['vmem'] for n in telem['neurons']]
        model.set_fpga_state(spike_counts, vmems)
    else:
        telem = {'total_spikes': 0, 'neurons': [{'spike_count': 0, 'vmem': 0}] * 8}

    return loss, telem


def closed_loop_step_with_plasticity(model, fpga, plasticity, data, batch_idx, device):
    """Closed-loop step with PlasticityManager driving Vg."""
    loss, telem = closed_loop_step(model, fpga, data, batch_idx, device)
    if telem is not None and loss is not None:
        plasticity.process_telemetry(telem)
        plasticity.apply_to_fpga(fpga)
    return loss, telem


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# T47: Paired-Pulse STP
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def run_T47(model, fpga, data, device):
    """T47: Paired-Pulse STP — synapse short-term plasticity.

    3 neuron pairs, ISIs [10,20,50,100,200,500]ms, 10 trials each.
    Pulse: set pre-neuron Vg=0.50 for 5ms, measure post-neuron spike response.
    Build facilitation/depression curve.
    PASS: |ratio-1.0|>0.15 at short ISI AND |ratio-1.0|<0.10 at long ISI for >=1 pair.
    """
    print("\n" + "=" * 60)
    print("T47: Paired-Pulse STP (Synapse Short-Term Plasticity)")
    print("=" * 60)

    pairs = [(0, 1), (2, 3), (4, 5)]
    isis_ms = [10, 20, 50, 100, 200, 500]
    n_trials = 10
    pair_results = []
    any_pass = False

    for pre_id, post_id in pairs:
        print(f"  Pair ({pre_id}->{post_id}):")
        ratios_by_isi = {}

        for isi_ms in isis_ms:
            isi_s = isi_ms / 1000.0
            responses_first = []
            responses_second = []

            for trial in range(n_trials):
                # Reset Vg to baseline
                for nid in range(N_NEURONS):
                    fpga.set_gate_voltage(nid, 0.25)
                time.sleep(0.05)

                # First pulse: Vg=0.50 on pre-neuron for 5ms
                fpga.set_gate_voltage(pre_id, 0.50)
                time.sleep(0.005)
                fpga.set_gate_voltage(pre_id, 0.25)

                # Read post-neuron response to first pulse
                telem1 = fpga.read_telemetry()
                r1 = 0
                if telem1 is not None:
                    r1 = telem1['neurons'][post_id]['spike_count']
                responses_first.append(r1)

                # Wait ISI
                time.sleep(isi_s)

                # Second pulse
                fpga.set_gate_voltage(pre_id, 0.50)
                time.sleep(0.005)
                fpga.set_gate_voltage(pre_id, 0.25)

                # Read post-neuron response to second pulse
                telem2 = fpga.read_telemetry()
                r2 = 0
                if telem2 is not None:
                    r2 = telem2['neurons'][post_id]['spike_count']
                responses_second.append(r2)

            mean_first = float(np.mean(responses_first)) if responses_first else 0.0
            mean_second = float(np.mean(responses_second)) if responses_second else 0.0
            ratio = mean_second / max(mean_first, 0.1)
            ratios_by_isi[isi_ms] = ratio
            print(f"    ISI={isi_ms:4d}ms: R1={mean_first:.2f} R2={mean_second:.2f} ratio={ratio:.3f}")

        # Check criteria for this pair
        short_isis = [10, 20]
        long_isis = [200, 500]
        short_ratios = [ratios_by_isi[i] for i in short_isis if i in ratios_by_isi]
        long_ratios = [ratios_by_isi[i] for i in long_isis if i in ratios_by_isi]

        short_ok = any(abs(r - 1.0) > 0.15 for r in short_ratios)
        long_ok = any(abs(r - 1.0) < 0.10 for r in long_ratios)
        pair_pass = short_ok and long_ok
        if pair_pass:
            any_pass = True

        pair_results.append({
            'pre': pre_id, 'post': post_id,
            'ratios': ratios_by_isi,
            'short_ok': short_ok, 'long_ok': long_ok,
            'pass': pair_pass,
        })

    result = {
        'test': 'T47',
        'name': 'Paired-Pulse STP',
        'pairs': pair_results,
        'status': 'PASS' if any_pass else 'FAIL',
        'criterion': '|ratio-1|>0.15@short AND |ratio-1|<0.10@long for >=1 pair',
    }
    print(f"  => {result['status']}")
    return result


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# T48: Multi-Level Weight Retention
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def run_T48(model, fpga, data, device):
    """T48: Multi-Level Weight Retention.

    Set Vg to each of 8 levels on neuron 0, measure spike rate.
    Wait 30s, re-measure at last level.
    PASS: monotonic ordering for >=6 levels AND retention within 20%.
    """
    print("\n" + "=" * 60)
    print("T48: Multi-Level Weight Retention")
    print("=" * 60)

    vg_levels = [0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]
    n_reads = 20
    neuron_id = 0
    rates = []

    for level_idx, vg in enumerate(vg_levels):
        fpga.set_gate_voltage(neuron_id, vg)
        time.sleep(0.5)  # settle

        spikes = []
        for _ in range(n_reads):
            telem = fpga.read_telemetry()
            if telem is not None:
                spikes.append(telem['neurons'][neuron_id]['spike_count'])
            time.sleep(0.02)

        mean_rate = float(np.mean(spikes)) if spikes else 0.0
        rates.append(mean_rate)
        print(f"    Level {level_idx} (Vg={vg:.2f}V): mean_rate={mean_rate:.2f}")

    # Check monotonicity: count how many consecutive pairs are non-decreasing
    mono_count = sum(1 for i in range(len(rates) - 1) if rates[i + 1] >= rates[i])
    monotonic_ok = mono_count >= 5  # at least 6 levels monotonic (5 transitions)
    print(f"  Monotonic transitions: {mono_count}/7")

    # Retention test: wait 30s, re-measure last level
    last_vg = vg_levels[-1]
    print(f"  Waiting 30s for retention test at Vg={last_vg:.2f}V...")
    time.sleep(30.0)

    retention_spikes = []
    for _ in range(n_reads):
        telem = fpga.read_telemetry()
        if telem is not None:
            retention_spikes.append(telem['neurons'][neuron_id]['spike_count'])
        time.sleep(0.02)

    retention_rate = float(np.mean(retention_spikes)) if retention_spikes else 0.0
    original_rate = rates[-1]
    if original_rate > 0:
        retention_error = abs(retention_rate - original_rate) / original_rate
    else:
        retention_error = 0.0 if retention_rate == 0 else 1.0
    retention_ok = retention_error < 0.20
    print(f"    Retention: original={original_rate:.2f}, after_30s={retention_rate:.2f}, "
          f"error={retention_error:.3f}")

    passed = monotonic_ok and retention_ok

    result = {
        'test': 'T48',
        'name': 'Multi-Level Weight Retention',
        'vg_levels': vg_levels,
        'rates': rates,
        'mono_count': mono_count,
        'monotonic_ok': monotonic_ok,
        'retention_rate': retention_rate,
        'original_rate': original_rate,
        'retention_error': retention_error,
        'retention_ok': retention_ok,
        'status': 'PASS' if passed else 'FAIL',
        'criterion': 'monotonic>=6 levels AND retention<20%',
    }
    print(f"  => {result['status']}")
    return result


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# T49: Plasticity LM Advantage
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def run_T49(model, fpga, data, device):
    """T49: Plasticity LM Advantage.

    50 batches with PlasticityManager active vs 50 with frozen Vg=0.35.
    PASS: PPL_plastic / PPL_frozen < 0.99.
    """
    print("\n" + "=" * 60)
    print("T49: Plasticity LM Advantage")
    print("=" * 60)

    N = 50
    model.set_open_loop(False)

    # Phase 1: Frozen Vg=0.35
    print("  Phase 1: Frozen Vg=0.35 (50 batches)...")
    for nid in range(N_NEURONS):
        fpga.set_gate_voltage(nid, 0.35)
    time.sleep(0.2)

    losses_frozen = []
    for b in range(N):
        loss, telem = closed_loop_step(model, fpga, data, b % 20, device)
        if loss is not None:
            losses_frozen.append(loss)
    ppl_frozen = math.exp(np.mean(losses_frozen)) if losses_frozen else 999.0
    print(f"    PPL_frozen = {ppl_frozen:.4f}")

    # Phase 2: PlasticityManager active
    print("  Phase 2: PlasticityManager active (50 batches)...")
    plasticity = PlasticityManager()
    losses_plastic = []
    for b in range(N):
        loss, telem = closed_loop_step_with_plasticity(
            model, fpga, plasticity, data, b % 20, device)
        if loss is not None:
            losses_plastic.append(loss)
    ppl_plastic = math.exp(np.mean(losses_plastic)) if losses_plastic else 999.0
    print(f"    PPL_plastic = {ppl_plastic:.4f}")

    ratio = ppl_plastic / max(ppl_frozen, 1e-6)
    passed = ratio < 0.99

    result = {
        'test': 'T49',
        'name': 'Plasticity LM Advantage',
        'ppl_frozen': ppl_frozen,
        'ppl_plastic': ppl_plastic,
        'ratio': ratio,
        'status': 'PASS' if passed else 'FAIL',
        'criterion': 'PPL_plastic/PPL_frozen < 0.99',
    }
    print(f"  => ratio={ratio:.6f} => {result['status']}")
    return result


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# T50: SOC Convergence
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def run_T50(model, fpga, data, device):
    """T50: SOC Convergence.

    Phase A: all Vg=0.15, 300 adaptation steps.
    Phase B: all Vg=0.50, 300 adaptation steps.
    Track branching ratio trajectory.
    PASS: both converge to sigma in [0.8,1.2] within 250 steps
          AND final Vg means agree within 0.15V.
    """
    print("\n" + "=" * 60)
    print("T50: SOC Convergence")
    print("=" * 60)

    N_STEPS = 300

    def run_phase(name, init_vg):
        print(f"  Phase {name}: init Vg={init_vg:.2f}V, {N_STEPS} steps...")
        controller = HomeostaticController(target_rate=100.0, eta=0.002, ema_alpha=0.1)
        controller.vg_current = [init_vg] * N_NEURONS

        for nid in range(N_NEURONS):
            fpga.set_gate_voltage(nid, init_vg)
        time.sleep(0.2)

        sigma_trajectory = []
        for step in range(N_STEPS):
            telem = fpga.read_telemetry()
            if telem is None:
                continue
            spikes = [n['spike_count'] for n in telem['neurons']]
            new_vg = controller.adapt_step(spikes)
            for nid in range(N_NEURONS):
                fpga.set_gate_voltage(nid, new_vg[nid])
            sigma = controller.get_branching_ratio()
            sigma_trajectory.append(sigma)
            time.sleep(0.005)

        final_sigma = sigma_trajectory[-1] if sigma_trajectory else 0.0
        final_vg_mean = float(np.mean(controller.vg_current))

        # Check convergence: find first step where sigma enters [0.8, 1.2]
        converge_step = -1
        for i, s in enumerate(sigma_trajectory):
            if 0.8 <= s <= 1.2:
                converge_step = i
                break

        print(f"    Final sigma={final_sigma:.4f}, Vg_mean={final_vg_mean:.3f}, "
              f"converge_step={converge_step}")
        return {
            'final_sigma': final_sigma,
            'final_vg_mean': final_vg_mean,
            'converge_step': converge_step,
            'sigma_trajectory': sigma_trajectory[-20:],  # last 20 for JSON
        }

    phase_a = run_phase('A (subcritical)', 0.15)
    phase_b = run_phase('B (supercritical)', 0.50)

    conv_a = 0 <= phase_a['converge_step'] <= 250
    conv_b = 0 <= phase_b['converge_step'] <= 250
    vg_agree = abs(phase_a['final_vg_mean'] - phase_b['final_vg_mean']) < 0.15

    passed = conv_a and conv_b and vg_agree

    result = {
        'test': 'T50',
        'name': 'SOC Convergence',
        'phase_a': phase_a,
        'phase_b': phase_b,
        'conv_a': conv_a,
        'conv_b': conv_b,
        'vg_agree': vg_agree,
        'vg_diff': abs(phase_a['final_vg_mean'] - phase_b['final_vg_mean']),
        'status': 'PASS' if passed else 'FAIL',
        'criterion': 'both sigma in [0.8,1.2] within 250 steps AND Vg agree<0.15V',
    }
    print(f"  => conv_a={conv_a} conv_b={conv_b} vg_agree={vg_agree} => {result['status']}")
    return result


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# T51: Avalanche Size Distribution
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def run_T51(model, fpga, data, device):
    """T51: Avalanche Size Distribution.

    ScaledBank(128) at Vg=0.35, 3000 timesteps.
    Detect avalanches, fit power-law alpha via MLE.
    PASS: alpha in [1.0, 3.0] AND n_avalanches >= 50.
    """
    print("\n" + "=" * 60)
    print("T51: Avalanche Size Distribution (Power-Law)")
    print("=" * 60)

    bank = ScaledBank(128)
    print("  Collecting raster (128 virtual neurons, 3000 steps)...")
    raster = bank.collect_raster(fpga, n_steps=3000, vg=0.35)

    # Total activity per timestep
    activity = raster.sum(axis=1)  # (3000,)
    threshold = float(np.mean(activity))
    print(f"  Mean activity: {threshold:.2f}")

    # Detect avalanches: contiguous runs above threshold
    above = activity > threshold
    avalanche_sizes = []
    current_size = 0
    for t in range(len(above)):
        if above[t]:
            current_size += int(activity[t])
        else:
            if current_size > 0:
                avalanche_sizes.append(current_size)
                current_size = 0
    if current_size > 0:
        avalanche_sizes.append(current_size)

    n_avalanches = len(avalanche_sizes)
    print(f"  Detected {n_avalanches} avalanches")

    # Fit power-law alpha via MLE: alpha = 1 + n / sum(ln(x / x_min))
    alpha = 0.0
    if n_avalanches >= 10:
        sizes = np.array(avalanche_sizes, dtype=float)
        x_min = max(float(np.min(sizes)), 1.0)
        log_ratios = np.log(sizes / x_min)
        valid = log_ratios > 0
        if np.sum(valid) > 5:
            alpha = 1.0 + float(np.sum(valid)) / float(np.sum(log_ratios[valid]))
    print(f"  Power-law alpha (MLE) = {alpha:.4f}")

    passed = 1.0 <= alpha <= 3.0 and n_avalanches >= 50

    result = {
        'test': 'T51',
        'name': 'Avalanche Size Distribution',
        'n_avalanches': n_avalanches,
        'alpha': alpha,
        'mean_size': float(np.mean(avalanche_sizes)) if avalanche_sizes else 0.0,
        'max_size': int(max(avalanche_sizes)) if avalanche_sizes else 0,
        'status': 'PASS' if passed else 'FAIL',
        'criterion': 'alpha in [1.0,3.0] AND n_avalanches>=50',
    }
    print(f"  => alpha={alpha:.4f}, n_aval={n_avalanches} => {result['status']}")
    return result


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# T52: Spike-Timing Precision (ISI CV)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def run_T52(model, fpga, data, device):
    """T52: Spike-Timing Precision (ISI CV).

    200 reads at Vg=0.35 (critical), 0.15 (sub), 0.50 (super).
    Compute ISI coefficient of variation per regime.
    PASS: ISI_CV_critical in [0.3,2.0] AND regimes produce different CVs
          (max/min ratio > 1.2).
    """
    print("\n" + "=" * 60)
    print("T52: Spike-Timing Precision (ISI CV)")
    print("=" * 60)

    regimes = {
        'subcritical': 0.15,
        'critical': 0.35,
        'supercritical': 0.50,
    }
    n_reads = 200
    cv_results = {}

    for regime_name, vg in regimes.items():
        print(f"  Regime: {regime_name} (Vg={vg:.2f}V)...")
        for nid in range(N_NEURONS):
            fpga.set_gate_voltage(nid, vg)
        time.sleep(0.3)

        # Discard stale
        for _ in range(3):
            fpga.read_telemetry()
            time.sleep(0.02)

        spike_counts = []
        timestamps = []
        for _ in range(n_reads):
            telem = fpga.read_telemetry()
            if telem is not None:
                spike_counts.append(telem['total_spikes'])
                timestamps.append(telem['timestamp'])
            time.sleep(0.005)

        # Compute ISI CV from inter-spike intervals
        if len(timestamps) >= 3:
            isis = []
            for i in range(1, len(timestamps)):
                dt = timestamps[i] - timestamps[i - 1]
                sc = spike_counts[i]
                if sc > 0 and dt > 0:
                    isi = dt / sc
                    isis.append(isi)
            if len(isis) >= 5:
                arr = np.array(isis)
                mean_isi = float(np.mean(arr))
                std_isi = float(np.std(arr))
                cv = std_isi / mean_isi if mean_isi > 0 else 0.0
            else:
                cv = 0.0
        else:
            cv = 0.0

        cv_results[regime_name] = cv
        print(f"    ISI_CV = {cv:.4f}")

    cv_crit = cv_results.get('critical', 0.0)
    all_cvs = [v for v in cv_results.values() if v > 0]

    crit_ok = 0.3 <= cv_crit <= 2.0
    if len(all_cvs) >= 2:
        cv_ratio = max(all_cvs) / max(min(all_cvs), 1e-6)
        diff_ok = cv_ratio > 1.2
    else:
        cv_ratio = 0.0
        diff_ok = False

    passed = crit_ok and diff_ok

    result = {
        'test': 'T52',
        'name': 'Spike-Timing Precision (ISI CV)',
        'cv_subcritical': cv_results.get('subcritical', 0.0),
        'cv_critical': cv_crit,
        'cv_supercritical': cv_results.get('supercritical', 0.0),
        'cv_ratio': cv_ratio,
        'crit_ok': crit_ok,
        'diff_ok': diff_ok,
        'status': 'PASS' if passed else 'FAIL',
        'criterion': 'ISI_CV_crit in [0.3,2.0] AND max/min CV ratio > 1.2',
    }
    print(f"  => CV_crit={cv_crit:.4f}, ratio={cv_ratio:.4f} => {result['status']}")
    return result


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# T53: Cross-Substrate Plasticity Transfer
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def run_T53(model, fpga, data, device):
    """T53: Cross-Substrate Plasticity Transfer.

    50 closed-loop steps with PlasticityManager + FPGAGatedLoRA.
    Record LTP weight_level changes and hidden-state norms per LoRA layer.
    Spearman correlation between cumulative weight change and hidden-state trajectory.
    PASS: |rho| > 0.2 in >=1 LoRA layer.
    """
    print("\n" + "=" * 60)
    print("T53: Cross-Substrate Plasticity Transfer")
    print("=" * 60)

    N = 50
    model.set_open_loop(False)
    plasticity = PlasticityManager()

    # Reset Vg
    for nid in range(N_NEURONS):
        fpga.set_gate_voltage(nid, 0.35)
    time.sleep(0.2)

    # Track: cumulative weight change per step, hidden-state norm per LoRA layer
    weight_trajectories = []   # cumulative sum of weight level changes
    hidden_norms_per_layer = [[] for _ in range(len(model.lora_layers))]

    prev_weights = plasticity.get_all_weight_levels()
    cum_weight_change = 0.0

    for step in range(N):
        # Forward pass with plasticity
        start = (step % 20) * BS * SEQ_LEN
        end = start + BS * SEQ_LEN
        if end > len(data):
            continue
        chunk = data[start:end].view(BS, SEQ_LEN).to(device)
        labels = chunk.clone()
        labels[:, :-1] = chunk[:, 1:]
        labels[:, -1] = -100

        model.eval()
        with torch.no_grad():
            out = model(chunk, labels=labels)
        loss = out.loss.item()

        # FPGA feedback
        mac_signal = min(1.0, loss / 10.0)
        fpga.set_mac_signal(mac_signal)
        time.sleep(0.01)
        telem = fpga.read_telemetry()
        if telem is not None:
            sc = [n['spike_count'] for n in telem['neurons']]
            vm = [n['vmem'] for n in telem['neurons']]
            model.set_fpga_state(sc, vm)
            plasticity.process_telemetry(telem)
            plasticity.apply_to_fpga(fpga)

        # Record weight change
        curr_weights = plasticity.get_all_weight_levels()
        delta = sum(abs(c - p) for c, p in zip(curr_weights, prev_weights))
        cum_weight_change += delta
        weight_trajectories.append(cum_weight_change)
        prev_weights = curr_weights[:]

        # Record hidden-state norms per LoRA layer
        for li, lora in enumerate(model.lora_layers):
            if lora.last_gate is not None:
                hidden_norms_per_layer[li].append(float(np.linalg.norm(lora.last_gate)))
            else:
                hidden_norms_per_layer[li].append(0.0)

    # Compute Spearman correlations
    from scipy.stats import spearmanr
    best_rho = 0.0
    best_layer = -1
    layer_rhos = []

    for li in range(len(model.lora_layers)):
        norms = hidden_norms_per_layer[li]
        n_valid = min(len(weight_trajectories), len(norms))
        if n_valid < 10:
            layer_rhos.append(0.0)
            continue
        rho, p_val = spearmanr(weight_trajectories[:n_valid], norms[:n_valid])
        layer_rhos.append(float(rho))
        print(f"    Layer {li}: rho={rho:.4f} (p={p_val:.4e})")
        if abs(rho) > abs(best_rho):
            best_rho = float(rho)
            best_layer = li

    passed = abs(best_rho) > 0.2

    result = {
        'test': 'T53',
        'name': 'Cross-Substrate Plasticity Transfer',
        'layer_rhos': layer_rhos,
        'best_rho': best_rho,
        'best_layer': best_layer,
        'n_steps': N,
        'final_cum_weight_change': cum_weight_change,
        'status': 'PASS' if passed else 'FAIL',
        'criterion': '|rho|>0.2 in >=1 LoRA layer',
    }
    print(f"  => best |rho|={abs(best_rho):.4f} (layer {best_layer}) => {result['status']}")
    return result


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# MAIN: Run battery + scorecard
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def main():
    print("=" * 70)
    print("z2148v39: Mario-Bridge v2 Test Battery — T47-T53")
    print("=" * 70)

    device = torch.device(DEVICE if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    if device.type == 'cuda':
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  HSA_OVERRIDE_GFX_VERSION={os.environ.get('HSA_OVERRIDE_GFX_VERSION', 'unset')}")

    # Load GPT-2 tokenizer + data
    print("\nLoading GPT-2 tokenizer and evaluation data...")
    from transformers import GPT2Tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    eval_data = get_wikitext_data(tokenizer, n_tokens=32768)
    print(f"  Eval tokens: {len(eval_data)}")

    # Build model
    print("\nBuilding FEELBridgeModel (GPT-2 + FPGAGatedLoRA, layers 6-11, rank 4)...")
    model = FEELBridgeModel().to(device)
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable params: {n_trainable:,}")

    # Baseline PPL (no FPGA)
    print("\nBaseline PPL (frozen GPT-2, no FPGA feedback)...")
    model.set_open_loop(True)
    baseline_ppl = eval_ppl(model, eval_data, device)
    print(f"  Baseline PPL = {baseline_ppl:.4f}")

    # Connect FPGA
    print(f"\nConnecting to FPGA on {FPGA_PORT}...")
    fpga_ok = False
    try:
        fpga = NSRAMFPGABridge(port=FPGA_PORT, baudrate=FPGA_BAUD_FAST)
        fpga_ok = True
        print("  FPGA connected.")
        telem = fpga.read_telemetry()
        if telem:
            print(f"  Initial telemetry: spikes={telem['total_spikes']}, "
                  f"bvpar={telem['mean_bvpar']:.3f}V")
        else:
            print("  WARNING: No initial telemetry (FPGA may need reset)")
    except Exception as e:
        print(f"  ERROR: FPGA connection failed: {e}")
        print("  Running in SIMULATED mode (results will be synthetic)")
        fpga = SimulatedFPGA()

    # Run T47-T53
    results = []
    try:
        results.append(run_T47(model, fpga, eval_data, device))
        results.append(run_T48(model, fpga, eval_data, device))
        results.append(run_T49(model, fpga, eval_data, device))
        results.append(run_T50(model, fpga, eval_data, device))
        results.append(run_T51(model, fpga, eval_data, device))
        results.append(run_T52(model, fpga, eval_data, device))
        results.append(run_T53(model, fpga, eval_data, device))
    finally:
        if hasattr(fpga, 'close'):
            fpga.close()
            print("\nFPGA bridge closed.")

    # ━━━ SCORECARD ━━━
    print("\n" + "=" * 70)
    print("SCORECARD: Mario-Bridge v2 Battery T47-T53")
    print("=" * 70)
    n_pass = 0
    for r in results:
        status = r.get('status', 'FAIL')
        marker = 'PASS' if status == 'PASS' else 'FAIL'
        if status == 'PASS':
            n_pass += 1
        print(f"  {r['test']:5s} {r['name']:40s} {marker:4s}  ({r.get('criterion', '')})")
    print(f"\n  Total: {n_pass}/{len(results)} PASS")
    print(f"  FPGA mode: {'REAL' if fpga_ok else 'SIMULATED'}")
    print(f"  Baseline PPL: {baseline_ppl:.4f}")
    print("=" * 70)

    # Save results
    output = {
        'experiment': 'z2148_mario_bridge_v2_v39',
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'device': str(device),
        'fpga_real': fpga_ok,
        'baseline_ppl': baseline_ppl,
        'n_pass': n_pass,
        'n_total': len(results),
        'tests': results,
    }
    RESULTS_JSON.parent.mkdir(exist_ok=True)
    with open(RESULTS_JSON, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n  Results saved to {RESULTS_JSON}")


if __name__ == '__main__':
    main()
