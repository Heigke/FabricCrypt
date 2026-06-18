#!/usr/bin/env python3
"""
z2147v38: Enhanced GPU→FPGA Coupling — Gradient-Norm + Logit-Entropy Feedback
==============================================================================
Replaces the weak residual-norm GPU→FPGA signal (z2139 rho=0.062) with
high-dynamic-range gradient-norm + logit-entropy signals.

Phases:
  P1: Warmup (20 batches) — establish baselines, measure dynamic range
  P2: Enhanced Closed Loop (500 batches) — run with grad+entropy feedback
  P3: Feedback Correlation Analysis — Spearman rho vs z2139
  P4: Transfer Entropy — bidirectional TE with 500 samples, k=2, 50 shuffles
  P5: Latency Benchmark — 921600 vs 115200 baud round-trip timing

Hardware:
  - AMD gfx1151 GPU (HSA_OVERRIDE_GFX_VERSION=11.0.0)
  - Tang Nano 9K FPGA on /dev/ttyUSB1 (115200 or 921600 baud)

Run:
  HSA_OVERRIDE_GFX_VERSION=11.0.0 venv/bin/python -u scripts/z2147_enhanced_coupling_v38.py
"""

import os, sys, json, math, time, struct, random, threading
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque, Counter
from datetime import datetime
from pathlib import Path
from scipy import stats as scipy_stats

# Ensure HSA override for gfx1151
os.environ.setdefault('HSA_OVERRIDE_GFX_VERSION', '11.0.0')

BASE = Path(__file__).resolve().parent.parent
RESULTS_JSON = BASE / 'results' / 'z2147_enhanced_coupling.json'

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CONSTANTS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
DEVICE = 'cuda'
BS = 4
SEQ_LEN = 128
N_NEURONS = 8
FPGA_PORT = '/dev/ttyUSB1'
FPGA_BAUD = 115200

# FPGA voltage limits
VG_BASE = 0.30
VG_MIN = 0.15
VG_MAX = 0.55
FEEDBACK_SCALE = 0.15  # Vg change per unit feedback deviation

# LoRA config
LORA_RANK = 8
LORA_ALPHA = 16
LORA_LAYERS = list(range(4, 12))  # GPT-2 layers 4-11

# Lanza NS-RAM reference (for SimulatedFPGA)
LANZA_BV0 = 4.2
LANZA_ALPHA_T = 0.003
LANZA_T0 = 300.0
LANZA_BETA_VG = 1.8


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# GPU SYSFS TELEMETRY
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
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
# Q8.8 / CRC HELPERS
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


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SIMULATED FPGA (fallback when no hardware)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class SimulatedFPGA:
    """Minimal FPGA simulator for testing without hardware.

    Produces plausible telemetry based on a simplified NS-RAM model:
    spike rate depends on Vg (gate voltage) and temperature.
    """

    def __init__(self):
        self.vg = [0.35] * N_NEURONS
        self.temp_k = 300.0
        self.mac_val = 0.5
        self.kill = False
        self.telemetry_history = deque(maxlen=1000)
        self.spike_history = deque(maxlen=10000)
        self.latency_history = deque(maxlen=1000)
        self.last_latency_ms = 0.0
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
        t0 = time.monotonic()
        neurons = []
        for i in range(N_NEURONS):
            if self.kill:
                sc, vm, bv = 0, 0.0, 0.0
            else:
                vg = self.vg[i]
                t_factor = 1.0 + 0.002 * (self.temp_k - 300.0)
                base_rate = (vg ** 2) * 200.0 * t_factor
                rate = base_rate * (0.8 + 0.4 * self.mac_val)
                sc = max(0, int(rate + random.gauss(0, rate * 0.15)))
                bv = LANZA_BV0 * math.exp(-LANZA_ALPHA_T * (self.temp_k - LANZA_T0)) * \
                     (1.0 + LANZA_BETA_VG * vg)
                vm = vg * 0.8 + random.gauss(0, 0.02)
            neurons.append({'spike_count': sc, 'vmem': vm, 'bvpar': bv})

        latency_ms = (time.monotonic() - t0) * 1000.0
        self.last_latency_ms = latency_ms
        self.latency_history.append(latency_ms)

        result = {
            'timestamp': time.time(),
            'neurons': neurons,
            'total_spikes': sum(n['spike_count'] for n in neurons),
            'mean_vmem': float(np.mean([n['vmem'] for n in neurons])),
            'mean_bvpar': float(np.mean([n['bvpar'] for n in neurons])),
            'latency_ms': latency_ms,
        }
        self.telemetry_history.append(result)
        self.spike_history.append((result['timestamp'], result['total_spikes']))
        return result

    def ping(self):
        return True

    @property
    def active_baudrate(self):
        return 115200

    def get_latency_stats(self):
        if not self.latency_history:
            return {'mean_ms': 0.0, 'min_ms': 0.0, 'max_ms': 0.0,
                    'p50_ms': 0.0, 'p95_ms': 0.0, 'p99_ms': 0.0, 'count': 0}
        arr = np.array(list(self.latency_history))
        return {
            'mean_ms': float(np.mean(arr)),
            'min_ms': float(np.min(arr)),
            'max_ms': float(np.max(arr)),
            'p50_ms': float(np.percentile(arr, 50)),
            'p95_ms': float(np.percentile(arr, 95)),
            'p99_ms': float(np.percentile(arr, 99)),
            'count': len(arr),
        }

    def close(self):
        pass


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# NS-RAM FPGA Bridge (Fast) — inline
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class NSRAMFPGABridgeFast:
    """High-speed bidirectional GPU<->FPGA coupling with closed-loop support."""

    CMD_SET_VG      = 0x01
    CMD_READ_TELEM  = 0x02
    CMD_SET_KILL    = 0x03
    CMD_SET_MAC     = 0x06
    CMD_CLOSED_LOOP = 0x10
    CMD_SET_BAUD    = 0xF0
    CMD_PING        = 0xAA
    SYNC = 0x55
    BAUD_FAST = 921600
    BAUD_SLOW = 115200

    def __init__(self, port: str = '/dev/ttyUSB1',
                 baudrate: int = 921600,
                 timeout: float = 0.5,
                 auto_negotiate: bool = True):
        import serial
        self._lock = threading.Lock()
        self._port = port
        self._timeout = timeout
        self._baudrate = baudrate
        self.spike_history = deque(maxlen=10000)
        self.telemetry_history = deque(maxlen=1000)
        self.last_telemetry = None
        self.latency_history = deque(maxlen=1000)
        self.last_latency_ms = 0.0
        self.ser = serial.Serial(port, baudrate, timeout=timeout)
        time.sleep(0.05)
        self.ser.reset_input_buffer()
        if auto_negotiate:
            self._auto_negotiate_baud()

    def _auto_negotiate_baud(self):
        import serial
        if self._baudrate == self.BAUD_FAST:
            if self._try_ping():
                return
            self.ser.close()
            self._baudrate = self.BAUD_SLOW
            self.ser = serial.Serial(self._port, self.BAUD_SLOW, timeout=self._timeout)
            time.sleep(0.05)
            self.ser.reset_input_buffer()
            if self._try_ping():
                return
        elif self._baudrate == self.BAUD_SLOW:
            self._try_ping()

    def _try_ping(self, retries: int = 3) -> bool:
        import serial
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
        return self._baudrate

    def _send_cmd(self, cmd: int, payload: bytes = b''):
        pkt = bytes([self.SYNC, cmd]) + payload
        self.ser.write(pkt)
        self.ser.flush()

    def _read_response_raw(self, timeout: float = 0.5):
        deadline = time.monotonic() + timeout
        buf = bytearray()
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
        pkt_body = bytes(buf[:-1])
        rx_crc = buf[-1]
        if crc8(pkt_body) != rx_crc:
            return None
        payload = bytes(buf[3:3 + payload_len])
        return {'type': resp_type, 'len': payload_len, 'payload': payload}

    def _send_and_recv(self, cmd: int, payload: bytes = b'',
                       timeout: float = 0.5):
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

    def ping(self) -> bool:
        resp = self._send_and_recv(self.CMD_PING, timeout=0.2)
        return resp is not None and resp.get('type') == self.CMD_PING

    def set_gate_voltage(self, neuron_id: int, vg: float):
        vg_q = to_q16_16(vg)
        payload = bytes([neuron_id & 0x07]) + struct.pack('>I', vg_q)
        self._send_cmd(self.CMD_SET_VG, payload)
        time.sleep(0.005)

    def set_temperature(self, temp_k: float):
        mac_val = (temp_k - 300.0) / 100.0
        self.set_mac_signal(mac_val)

    def set_mac_signal(self, mac_val: float):
        mac_q = to_q16_16(max(0.0, min(255.0, mac_val)))
        payload = struct.pack('>I', mac_q)
        self._send_cmd(self.CMD_SET_MAC, payload)
        time.sleep(0.005)

    def set_kill_switch(self, enabled: bool):
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
            'latency_ms': self.last_latency_ms,
        }
        self.telemetry_history.append(result)
        self.last_telemetry = result
        self.spike_history.append((result['timestamp'], result['total_spikes']))
        return result

    def get_latency_stats(self):
        if not self.latency_history:
            return {'mean_ms': 0.0, 'min_ms': 0.0, 'max_ms': 0.0,
                    'p50_ms': 0.0, 'p95_ms': 0.0, 'p99_ms': 0.0, 'count': 0}
        arr = np.array(list(self.latency_history))
        return {
            'mean_ms': float(np.mean(arr)),
            'min_ms': float(np.min(arr)),
            'max_ms': float(np.max(arr)),
            'p50_ms': float(np.percentile(arr, 50)),
            'p95_ms': float(np.percentile(arr, 95)),
            'p99_ms': float(np.percentile(arr, 99)),
            'count': len(arr),
        }

    def close(self):
        with self._lock:
            if self.ser and self.ser.is_open:
                self.ser.close()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# FPGA-Gated LoRA Layer
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class FPGAGatedLoRA(nn.Module):
    """LoRA adapter whose gate is driven by FPGA spike telemetry."""

    def __init__(self, base_linear, rank=8, alpha=16):
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


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# FEEL Bridge Model (GPT-2 + FPGAGatedLoRA)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class FEELBridgeModel(nn.Module):
    """GPT-2 small with FPGAGatedLoRA on attention projection layers."""

    def __init__(self):
        super().__init__()
        from transformers import GPT2LMHeadModel
        self.gpt2 = GPT2LMHeadModel.from_pretrained('gpt2')
        for p in self.gpt2.parameters():
            p.requires_grad = False

        self.lora_layers = nn.ModuleList()
        for i in LORA_LAYERS:
            attn = self.gpt2.transformer.h[i].attn
            original = attn.c_attn
            lora = FPGAGatedLoRA(original, rank=LORA_RANK, alpha=LORA_ALPHA)
            attn.c_attn = lora
            self.lora_layers.append(lora)

    def set_fpga_state(self, spike_counts, vmems):
        for lora in self.lora_layers:
            lora.set_fpga_state(spike_counts, vmems)

    def set_open_loop(self, open_loop: bool):
        for lora in self.lora_layers:
            lora.open_loop = open_loop

    def trainable_params(self):
        """Yield only LoRA parameters (for gradient computation)."""
        for lora in self.lora_layers:
            yield lora.lora_A
            yield lora.lora_B
            yield from lora.gate_proj.parameters()

    def forward(self, input_ids, labels=None):
        return self.gpt2(input_ids=input_ids, labels=labels)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TRANSFER ENTROPY
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def transfer_entropy(source, target, k=2, lag=1):
    """Transfer entropy from source -> target time series.

    TE = H(target_future | target_past) - H(target_future | target_past, source_past)
    Uses histogram-based estimation with adaptive bin count.
    Discretizes into 3 bins for interpretability.
    """
    source = np.asarray(source, dtype=float)
    target = np.asarray(target, dtype=float)
    n = min(len(source), len(target))
    if n < k + lag + 2:
        return 0.0

    n_bins = 3  # 3 bins as specified

    def digitise(x):
        lo, hi = np.min(x), np.max(x)
        if hi == lo:
            return np.zeros(len(x), dtype=int)
        return np.clip(((x - lo) / (hi - lo) * n_bins).astype(int), 0, n_bins - 1)

    sd = digitise(source)
    td = digitise(target)

    valid_start = k + lag
    valid_end = n
    if valid_start >= valid_end:
        return 0.0
    N = valid_end - valid_start

    def encode_history(arr, t, k_len):
        val = 0
        for j in range(k_len):
            val = val * n_bins + int(arr[t - k_len + j])
        return val

    cnt_tft_tp_sp = Counter()
    cnt_tp_sp = Counter()
    cnt_tft_tp = Counter()
    cnt_tp = Counter()

    for t in range(valid_start, valid_end):
        tf = int(td[t])
        tp = encode_history(td, t, k)
        sp = encode_history(sd, t - lag, k)
        cnt_tft_tp_sp[(tf, tp, sp)] += 1
        cnt_tp_sp[(tp, sp)] += 1
        cnt_tft_tp[(tf, tp)] += 1
        cnt_tp[tp] += 1

    te = 0.0
    for (tf, tp, sp), c_joint in cnt_tft_tp_sp.items():
        p_joint = c_joint / N
        p_tf_given_tp_sp = c_joint / cnt_tp_sp[(tp, sp)]
        p_tf_given_tp = cnt_tft_tp[(tf, tp)] / cnt_tp[tp]
        if p_tf_given_tp > 0 and p_tf_given_tp_sp > 0:
            te += p_joint * math.log2(p_tf_given_tp_sp / p_tf_given_tp)
    return te


def transfer_entropy_shuffle_baseline(source, target, k=2, lag=1, n_shuffles=50):
    """Compute TE and a shuffle baseline for significance testing."""
    te_real = transfer_entropy(source, target, k=k, lag=lag)
    te_shuffled = []
    src_copy = np.array(source)
    for _ in range(n_shuffles):
        np.random.shuffle(src_copy)
        te_shuffled.append(transfer_entropy(src_copy, target, k=k, lag=lag))
    return te_real, float(np.mean(te_shuffled)), float(np.std(te_shuffled))


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ENHANCED FEEDBACK CONTROLLER
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class EnhancedFeedbackController:
    """Bidirectional GPU⇄FPGA controller using gradient-norm + logit-entropy
    feedback instead of weak residual-norm.

    GPU→FPGA path:
      combined = alpha_grad * (grad_norm / grad_baseline)
              + alpha_entropy * (entropy / entropy_baseline)
      Vg = VG_BASE + FEEDBACK_SCALE * (combined - 1.0), clipped [VG_MIN, VG_MAX]

    FPGA→GPU path:
      spike telemetry → gate modulation on LoRA layers
    """

    def __init__(self, fpga, alpha_grad=0.5, alpha_entropy=0.5, feedback_alpha=0.3):
        self.fpga = fpga
        self.alpha_grad = alpha_grad
        self.alpha_entropy = alpha_entropy
        self.feedback_alpha = feedback_alpha  # EMA smoothing

        # Baselines (established during warmup)
        self.grad_baseline = 1.0
        self.entropy_baseline = 1.0

        # EMA state
        self.feedback_ema = 1.0

        # Per-neuron Vg
        self.vg = np.full(N_NEURONS, VG_BASE, dtype=np.float64)

        # History for analysis
        self.history = []

    def compute_gradient_norm(self, model, batch, device):
        """Forward pass with labels, backward on LoRA params, return grad norm."""
        chunk = batch.to(device)
        labels = chunk.clone()
        labels[:, :-1] = chunk[:, 1:]
        labels[:, -1] = -100

        # Need gradients for this pass
        model.train()
        # Zero existing grads
        for p in model.trainable_params():
            if p.grad is not None:
                p.grad.zero_()

        out = model(chunk, labels=labels)
        loss = out.loss
        loss.backward()

        # Compute gradient norm over LoRA params
        total_norm_sq = 0.0
        for p in model.trainable_params():
            if p.grad is not None:
                total_norm_sq += p.grad.norm().item() ** 2
        grad_norm = math.sqrt(total_norm_sq)

        # Zero grads after
        for p in model.trainable_params():
            if p.grad is not None:
                p.grad.zero_()

        model.eval()
        return grad_norm

    def compute_logit_entropy(self, model, batch, device):
        """Forward pass, compute entropy of output logit distribution."""
        chunk = batch.to(device)
        model.eval()
        with torch.no_grad():
            out = model(chunk)
            logits = out.logits  # [BS, SEQ_LEN, vocab_size]
            probs = F.softmax(logits, dim=-1)
            entropy = -(probs * torch.log(probs + 1e-10)).sum(-1).mean()
        return entropy.item()

    def compute_feedback(self, model, batch, device):
        """Compute combined gradient-norm + logit-entropy feedback signal."""
        grad_norm = self.compute_gradient_norm(model, batch, device)
        entropy = self.compute_logit_entropy(model, batch, device)

        # Normalize by baselines
        grad_ratio = grad_norm / max(self.grad_baseline, 1e-8)
        entropy_ratio = entropy / max(self.entropy_baseline, 1e-8)

        combined = self.alpha_grad * grad_ratio + self.alpha_entropy * entropy_ratio
        combined = float(np.clip(combined, 0.1, 3.0))

        return combined, grad_norm, entropy

    def step(self, model, batch, device):
        """One closed-loop step: compute feedback, update FPGA, read telemetry.

        Returns dict with all signals for time-series recording.
        """
        # Compute enhanced feedback
        combined, grad_norm, entropy = self.compute_feedback(model, batch, device)

        # EMA smooth
        self.feedback_ema = (1.0 - self.feedback_alpha) * self.feedback_ema + \
                            self.feedback_alpha * combined

        # Map feedback to Vg
        vg_new = VG_BASE + FEEDBACK_SCALE * (self.feedback_ema - 1.0)
        vg_new = float(np.clip(vg_new, VG_MIN, VG_MAX))

        # Write Vg to all neurons
        self.vg[:] = vg_new
        for i in range(N_NEURONS):
            self.fpga.set_gate_voltage(i, vg_new)
        time.sleep(0.005)

        # Send GPU temperature for BVpar modulation
        gpu_temp = read_gpu_temp_c()
        if gpu_temp > 0:
            self.fpga.set_temperature(gpu_temp + 273.15)

        # Read telemetry
        telem = self.fpga.read_telemetry()
        if telem is not None:
            spike_counts = [n['spike_count'] for n in telem['neurons']]
            vmems = [n['vmem'] for n in telem['neurons']]
            total_spikes = telem['total_spikes']
            per_neuron_spikes = spike_counts
            model.set_fpga_state(spike_counts, vmems)
        else:
            spike_counts = [0] * N_NEURONS
            vmems = [0.0] * N_NEURONS
            total_spikes = 0
            per_neuron_spikes = [0] * N_NEURONS
            model.set_fpga_state(spike_counts, vmems)

        record = {
            'feedback': combined,
            'feedback_ema': self.feedback_ema,
            'grad_norm': grad_norm,
            'entropy': entropy,
            'vg': vg_new,
            'total_spikes': total_spikes,
            'per_neuron_spikes': per_neuron_spikes,
            'gpu_temp_c': gpu_temp,
        }
        self.history.append(record)
        return record


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# DATA LOADING
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def get_wikitext_data(tokenizer, n_tokens=65536):
    """Load a chunk of WikiText-2 for evaluation."""
    from datasets import load_dataset
    ds = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')
    text = '\n'.join([r['text'] for r in ds if r['text'].strip()])
    ids = tokenizer.encode(text)[:n_tokens]
    return torch.tensor(ids, dtype=torch.long)


def make_batches(data, n_batches):
    """Slice data into (n_batches) chunks of [BS, SEQ_LEN]."""
    batches = []
    for b in range(n_batches):
        start = b * BS * SEQ_LEN
        end = start + BS * SEQ_LEN
        if end > len(data):
            break
        batches.append(data[start:end].view(BS, SEQ_LEN))
    return batches


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# FPGA CONNECTION
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def connect_fpga():
    """Try real FPGA first (fast bridge at 921600, fallback to 115200),
    then SimulatedFPGA if no hardware."""
    # FPGA firmware doesn't respond to CMD_PING — use telemetry read instead
    for baud in [921600, 115200]:
        try:
            bridge = NSRAMFPGABridgeFast(port=FPGA_PORT, baudrate=baud, auto_negotiate=False)
            telem = bridge.read_telemetry()
            if telem is not None:
                print(f"[FPGA] Connected REAL FPGA at {baud} baud")
                return bridge, True
            bridge.close()
        except Exception as e:
            print(f"[FPGA] {baud} baud failed: {e}")

    print("[FPGA] Using SimulatedFPGA fallback")
    return SimulatedFPGA(), False


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# MAIN EXPERIMENT
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def main():
    print("=" * 70)
    print("z2147v38: Enhanced GPU→FPGA Coupling")
    print("  Gradient-Norm + Logit-Entropy Feedback")
    print("=" * 70)

    device = DEVICE if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    gpu_telem = read_gpu_telemetry()
    print(f"GPU: temp={gpu_telem['temp_c']:.1f}C  power={gpu_telem['power_w']:.1f}W")

    # Connect FPGA
    fpga, is_real_fpga = connect_fpga()

    # Load model
    print("\nLoading GPT-2...")
    from transformers import GPT2Tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    model = FEELBridgeModel().to(device)

    # Enable gradients on LoRA params
    for p in model.trainable_params():
        p.requires_grad = True

    print(f"LoRA trainable params: {sum(p.numel() for p in model.trainable_params()):,}")

    # Load data
    print("Loading WikiText-2...")
    data = get_wikitext_data(tokenizer, n_tokens=65536)
    batches = make_batches(data, 600)  # 20 warmup + 500 main + buffer
    print(f"Prepared {len(batches)} batches of [{BS}, {SEQ_LEN}]")

    # Create controller
    controller = EnhancedFeedbackController(fpga)

    results = {
        'experiment': 'z2147_enhanced_coupling_v38',
        'timestamp': datetime.now().isoformat(),
        'device': device,
        'is_real_fpga': is_real_fpga,
        'fpga_baud': fpga.active_baudrate if hasattr(fpga, 'active_baudrate') else 115200,
        'phases': {},
    }

    # ── Phase 1: Warmup ─────────────────────────────────────────
    print("\n" + "=" * 60)
    print("PHASE 1: Warmup (20 batches) — establish baselines")
    print("=" * 60)

    WARMUP_N = 20
    warmup_grad_norms = []
    warmup_entropies = []
    warmup_residual_norms = []  # for dynamic range comparison

    model.eval()
    for b in range(WARMUP_N):
        batch = batches[b]

        # Gradient norm
        grad_norm = controller.compute_gradient_norm(model, batch, device)
        warmup_grad_norms.append(grad_norm)

        # Logit entropy
        entropy = controller.compute_logit_entropy(model, batch, device)
        warmup_entropies.append(entropy)

        # Residual norm (z2139-style baseline comparison)
        with torch.no_grad():
            chunk = batch.to(device)
            out = model(chunk)
            logits = out.logits
            residual_norm = logits.norm(dim=-1).mean().item()
        warmup_residual_norms.append(residual_norm)

        if b % 5 == 0:
            print(f"  [{b:2d}/{WARMUP_N}] grad_norm={grad_norm:.4f}  "
                  f"entropy={entropy:.4f}  residual_norm={residual_norm:.4f}")

    # Establish baselines
    controller.grad_baseline = float(np.mean(warmup_grad_norms))
    controller.entropy_baseline = float(np.mean(warmup_entropies))

    # Dynamic range analysis
    grad_min, grad_max = min(warmup_grad_norms), max(warmup_grad_norms)
    ent_min, ent_max = min(warmup_entropies), max(warmup_entropies)
    res_min, res_max = min(warmup_residual_norms), max(warmup_residual_norms)

    grad_dyn_range = grad_max / max(grad_min, 1e-8)
    ent_dyn_range = ent_max / max(ent_min, 1e-8)
    res_dyn_range = res_max / max(res_min, 1e-8)

    print(f"\n  Baselines:")
    print(f"    grad_norm:     mean={controller.grad_baseline:.4f}  "
          f"range=[{grad_min:.4f}, {grad_max:.4f}]  dyn_range={grad_dyn_range:.3f}x")
    print(f"    entropy:       mean={controller.entropy_baseline:.4f}  "
          f"range=[{ent_min:.4f}, {ent_max:.4f}]  dyn_range={ent_dyn_range:.3f}x")
    print(f"    residual_norm: mean={np.mean(warmup_residual_norms):.4f}  "
          f"range=[{res_min:.4f}, {res_max:.4f}]  dyn_range={res_dyn_range:.3f}x")

    enhanced_dyn_better = grad_dyn_range > 2.0 * res_dyn_range

    results['phases']['P1_warmup'] = {
        'n_batches': WARMUP_N,
        'grad_baseline': controller.grad_baseline,
        'entropy_baseline': controller.entropy_baseline,
        'grad_norms': warmup_grad_norms,
        'entropies': warmup_entropies,
        'residual_norms': warmup_residual_norms,
        'grad_dynamic_range': grad_dyn_range,
        'entropy_dynamic_range': ent_dyn_range,
        'residual_dynamic_range': res_dyn_range,
        'enhanced_dynamic_range_better': enhanced_dyn_better,
    }

    # ── Phase 2: Enhanced Closed Loop ───────────────────────────
    print("\n" + "=" * 60)
    print("PHASE 2: Enhanced Closed Loop (500 batches)")
    print("=" * 60)

    LOOP_N = 500
    loop_start = WARMUP_N
    ts_feedback = []
    ts_grad_norm = []
    ts_entropy = []
    ts_vg = []
    ts_total_spikes = []
    ts_per_neuron_spikes = []

    t0_loop = time.time()
    for b in range(LOOP_N):
        batch_idx = loop_start + b
        if batch_idx >= len(batches):
            print(f"  [WARN] Ran out of batches at step {b}")
            break

        batch = batches[batch_idx]
        record = controller.step(model, batch, device)

        ts_feedback.append(record['feedback'])
        ts_grad_norm.append(record['grad_norm'])
        ts_entropy.append(record['entropy'])
        ts_vg.append(record['vg'])
        ts_total_spikes.append(record['total_spikes'])
        ts_per_neuron_spikes.append(record['per_neuron_spikes'])

        if b % 50 == 0:
            print(f"  [{b:3d}/{LOOP_N}] feedback={record['feedback']:.4f}  "
                  f"grad={record['grad_norm']:.4f}  ent={record['entropy']:.4f}  "
                  f"vg={record['vg']:.3f}  spikes={record['total_spikes']}")

    loop_elapsed = time.time() - t0_loop
    print(f"\n  Closed loop: {len(ts_feedback)} steps in {loop_elapsed:.1f}s "
          f"({len(ts_feedback)/loop_elapsed:.1f} steps/s)")

    results['phases']['P2_closed_loop'] = {
        'n_steps': len(ts_feedback),
        'elapsed_s': loop_elapsed,
        'time_series': {
            'feedback': ts_feedback,
            'grad_norm': ts_grad_norm,
            'entropy': ts_entropy,
            'vg': ts_vg,
            'total_spikes': ts_total_spikes,
            'per_neuron_spikes': ts_per_neuron_spikes,
        },
    }

    # ── Phase 3: Feedback Correlation Analysis ──────────────────
    print("\n" + "=" * 60)
    print("PHASE 3: Feedback Correlation Analysis")
    print("=" * 60)

    n_corr = len(ts_feedback)
    if n_corr >= 10:
        rho_grad, p_grad = scipy_stats.spearmanr(ts_grad_norm[:n_corr],
                                                   ts_total_spikes[:n_corr])
        rho_ent, p_ent = scipy_stats.spearmanr(ts_entropy[:n_corr],
                                                 ts_total_spikes[:n_corr])
        rho_combined, p_combined = scipy_stats.spearmanr(ts_feedback[:n_corr],
                                                          ts_total_spikes[:n_corr])
    else:
        rho_grad, p_grad = 0.0, 1.0
        rho_ent, p_ent = 0.0, 1.0
        rho_combined, p_combined = 0.0, 1.0

    z2139_rho = 0.062  # from z2139 T10 result
    improved = abs(rho_combined) > abs(z2139_rho)

    print(f"  Spearman correlations (n={n_corr}):")
    print(f"    grad_norm vs total_spikes:    rho={rho_grad:.4f}  p={p_grad:.4e}")
    print(f"    entropy vs total_spikes:      rho={rho_ent:.4f}  p={p_ent:.4e}")
    print(f"    combined vs total_spikes:     rho={rho_combined:.4f}  p={p_combined:.4e}")
    print(f"    z2139 baseline:               rho={z2139_rho:.4f}")
    print(f"    Improved over z2139:          {improved} "
          f"(|{rho_combined:.4f}| vs |{z2139_rho:.4f}|)")

    results['phases']['P3_correlation'] = {
        'n_samples': n_corr,
        'rho_grad_vs_spikes': float(rho_grad),
        'p_grad': float(p_grad),
        'rho_entropy_vs_spikes': float(rho_ent),
        'p_entropy': float(p_ent),
        'rho_combined_vs_spikes': float(rho_combined),
        'p_combined': float(p_combined),
        'z2139_rho_baseline': z2139_rho,
        'feedback_correlation_improved': improved,
    }

    # ── Phase 4: Transfer Entropy ───────────────────────────────
    print("\n" + "=" * 60)
    print("PHASE 4: Transfer Entropy (k=2, 50 shuffles)")
    print("=" * 60)

    n_te = len(ts_feedback)
    print(f"  Using {n_te} samples (vs 79 in z2139, 300 in z2143)")

    # TE(GPU→FPGA): GPU signal = combined_feedback, FPGA signal = total_spikes
    print("  Computing TE(GPU→FPGA)...")
    te_gpu_fpga, te_gpu_fpga_mean_shuf, te_gpu_fpga_std_shuf = \
        transfer_entropy_shuffle_baseline(
            ts_feedback[:n_te], ts_total_spikes[:n_te],
            k=2, lag=1, n_shuffles=50)
    te_gpu_fpga_z = (te_gpu_fpga - te_gpu_fpga_mean_shuf) / max(te_gpu_fpga_std_shuf, 1e-8)
    te_gpu_fpga_sig = te_gpu_fpga > te_gpu_fpga_mean_shuf + 2.0 * te_gpu_fpga_std_shuf

    print(f"    TE(GPU→FPGA) = {te_gpu_fpga:.6f}  "
          f"(shuffle: {te_gpu_fpga_mean_shuf:.6f} ± {te_gpu_fpga_std_shuf:.6f})  "
          f"z={te_gpu_fpga_z:.2f}  sig={te_gpu_fpga_sig}")

    # TE(FPGA→GPU): FPGA signal = total_spikes, GPU signal = grad_norm
    print("  Computing TE(FPGA→GPU)...")
    te_fpga_gpu, te_fpga_gpu_mean_shuf, te_fpga_gpu_std_shuf = \
        transfer_entropy_shuffle_baseline(
            ts_total_spikes[:n_te], ts_grad_norm[:n_te],
            k=2, lag=1, n_shuffles=50)
    te_fpga_gpu_z = (te_fpga_gpu - te_fpga_gpu_mean_shuf) / max(te_fpga_gpu_std_shuf, 1e-8)
    te_fpga_gpu_sig = te_fpga_gpu > te_fpga_gpu_mean_shuf + 2.0 * te_fpga_gpu_std_shuf

    print(f"    TE(FPGA→GPU) = {te_fpga_gpu:.6f}  "
          f"(shuffle: {te_fpga_gpu_mean_shuf:.6f} ± {te_fpga_gpu_std_shuf:.6f})  "
          f"z={te_fpga_gpu_z:.2f}  sig={te_fpga_gpu_sig}")

    te_significant = te_gpu_fpga_sig or te_fpga_gpu_sig

    results['phases']['P4_transfer_entropy'] = {
        'n_samples': n_te,
        'k': 2,
        'lag': 1,
        'n_shuffles': 50,
        'gpu_to_fpga': {
            'te': float(te_gpu_fpga),
            'shuffle_mean': float(te_gpu_fpga_mean_shuf),
            'shuffle_std': float(te_gpu_fpga_std_shuf),
            'z_score': float(te_gpu_fpga_z),
            'significant': bool(te_gpu_fpga_sig),
        },
        'fpga_to_gpu': {
            'te': float(te_fpga_gpu),
            'shuffle_mean': float(te_fpga_gpu_mean_shuf),
            'shuffle_std': float(te_fpga_gpu_std_shuf),
            'z_score': float(te_fpga_gpu_z),
            'significant': bool(te_fpga_gpu_sig),
        },
        'te_significant': bool(te_significant),
    }

    # ── Phase 5: Latency Benchmark ──────────────────────────────
    print("\n" + "=" * 60)
    print("PHASE 5: Latency Benchmark")
    print("=" * 60)

    latency_results = {}

    if is_real_fpga:
        # Try 921600 baud
        print("  Testing 921600 baud (100 iterations)...")
        fpga.latency_history.clear()
        for _ in range(100):
            fpga.read_telemetry()
        stats_fast = fpga.get_latency_stats()
        latency_results['baud_921600'] = stats_fast
        print(f"    median={stats_fast['p50_ms']:.2f}ms  "
              f"p95={stats_fast['p95_ms']:.2f}ms  "
              f"p99={stats_fast['p99_ms']:.2f}ms")

        # If we're at 921600, also try 115200 for comparison
        if hasattr(fpga, 'active_baudrate') and fpga.active_baudrate == 921600:
            print("  Testing 115200 baud fallback (100 iterations)...")
            try:
                import serial
                fpga_slow = NSRAMFPGABridgeFast(
                    port=FPGA_PORT, baudrate=115200, auto_negotiate=False)
                if fpga_slow.ping():
                    fpga_slow.latency_history.clear()
                    for _ in range(100):
                        fpga_slow.read_telemetry()
                    stats_slow = fpga_slow.get_latency_stats()
                    latency_results['baud_115200'] = stats_slow
                    print(f"    median={stats_slow['p50_ms']:.2f}ms  "
                          f"p95={stats_slow['p95_ms']:.2f}ms  "
                          f"p99={stats_slow['p99_ms']:.2f}ms")
                    fpga_slow.close()
                else:
                    print("    (115200 ping failed, skipping)")
                    fpga_slow.close()
            except Exception as e:
                print(f"    (115200 test failed: {e})")
        else:
            # Already at 115200, measure it
            fpga.latency_history.clear()
            for _ in range(100):
                fpga.read_telemetry()
            stats_slow = fpga.get_latency_stats()
            latency_results['baud_115200'] = stats_slow
            print(f"    median={stats_slow['p50_ms']:.2f}ms  "
                  f"p95={stats_slow['p95_ms']:.2f}ms  "
                  f"p99={stats_slow['p99_ms']:.2f}ms")
    else:
        # Simulated — just measure simulated latency
        print("  (Simulated FPGA — measuring simulated round-trip)")
        fpga.latency_history.clear()
        for _ in range(100):
            fpga.read_telemetry()
        stats_sim = fpga.get_latency_stats()
        latency_results['simulated'] = stats_sim
        print(f"    median={stats_sim['p50_ms']:.2f}ms  "
              f"p95={stats_sim['p95_ms']:.2f}ms  "
              f"p99={stats_sim['p99_ms']:.2f}ms")

    results['phases']['P5_latency'] = latency_results

    # ── Summary ─────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    summary = {
        'enhanced_dynamic_range': enhanced_dyn_better,
        'grad_dyn_range_vs_residual': f"{grad_dyn_range:.3f}x vs {res_dyn_range:.3f}x",
        'feedback_correlation_improved': improved,
        'rho_combined_vs_z2139': f"|{rho_combined:.4f}| vs |{z2139_rho:.4f}|",
        'te_significant': te_significant,
        'te_gpu_fpga_z': float(te_gpu_fpga_z),
        'te_fpga_gpu_z': float(te_fpga_gpu_z),
    }

    print(f"  enhanced_dynamic_range:      {'PASS' if enhanced_dyn_better else 'FAIL'}  "
          f"(grad {grad_dyn_range:.3f}x > 2*residual {res_dyn_range:.3f}x ?)")
    print(f"  feedback_correlation:         {'PASS' if improved else 'FAIL'}  "
          f"(|rho|={abs(rho_combined):.4f} > z2139 {z2139_rho:.4f} ?)")
    print(f"  transfer_entropy_significant: {'PASS' if te_significant else 'FAIL'}  "
          f"(GPU→FPGA z={te_gpu_fpga_z:.2f}, FPGA→GPU z={te_fpga_gpu_z:.2f})")

    results['summary'] = summary

    # ── Save ────────────────────────────────────────────────────
    RESULTS_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_JSON, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {RESULTS_JSON}")

    # Cleanup
    fpga.close()
    print("\nDone.")


if __name__ == '__main__':
    main()
