#!/usr/bin/env python3
"""
z2145v36: Scaled Avalanche Statistics — Virtual Neuron Bank
============================================================
Scale the physical 8-neuron FPGA bank to 64/128/256 virtual neurons
and run avalanche statistics across subcritical, critical, and
supercritical regimes.

Architecture:
  - VirtualNeuron: maps to one of 8 physical neurons (round-robin),
    adds spatial decorrelation via Vg offset, noise, refractory jitter
  - ScaledBank: holds N VirtualNeurons, produces binary spike rasters
  - Avalanche detection: contiguous active frames
  - Power-law MLE fit with KS goodness-of-fit
  - Branching ratio: sigma(t) = n_active(t+1) / n_active(t)

Hardware:
  - AMD gfx1151 GPU (HSA_OVERRIDE_GFX_VERSION=11.0.0)
  - Tang Nano 9K FPGA on /dev/ttyUSB1 (921600 / 115200 baud)
  - SimulatedFPGA fallback if no real FPGA

Run:
  HSA_OVERRIDE_GFX_VERSION=11.0.0 venv/bin/python -u scripts/z2145_scaled_avalanche_v36.py
"""

import os, sys, json, math, time, struct, random
import numpy as np
from pathlib import Path
from collections import deque, Counter

os.environ.setdefault('HSA_OVERRIDE_GFX_VERSION', '11.0.0')

BASE = Path(__file__).resolve().parent.parent
RESULTS_JSON = BASE / 'results' / 'z2145_scaled_avalanche.json'

N_PHYS_NEURONS = 8
FPGA_PORT = '/dev/ttyUSB1'
FPGA_BAUD_FAST = 921600
FPGA_BAUD_SLOW = 115200

# Lanza NS-RAM reference parameters
LANZA_BV0 = 4.2
LANZA_ALPHA_T = 0.003
LANZA_T0 = 300.0
LANZA_BETA_VG = 1.8


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# INLINE UTILITIES (self-contained — no cross-script imports)
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


def lempel_ziv_complexity(binary_string):
    """LZ76 complexity of a binary string, normalised to ~[0,1]."""
    n = len(binary_string)
    if n == 0:
        return 0.0
    s = binary_string + '0'
    i, k, l, c = 0, 1, 1, 1
    while k + l <= n:
        if s[i + l] == s[k + l]:
            l += 1
        else:
            if l >= k - i:
                k += l + 1
                c += 1
                i = k - 1
                l = 1
            else:
                i += 1
                if i == k:
                    k += 1
                    c += 1
                    i = k - 1
                    l = 1
                else:
                    l = 1
    if l > 0:
        c += 1
    norm = n / math.log2(n) if n > 1 else 1.0
    return c / norm


def spike_train_to_binary(spike_counts, threshold=None):
    arr = np.array(spike_counts, dtype=float)
    if threshold is None:
        threshold = float(np.median(arr))
    return ''.join('1' if s > threshold else '0' for s in arr)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# FPGA BRIDGE (inline, from nsram_fpga_bridge_fast.py)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class NSRAMFPGABridgeFast:
    """High-speed bidirectional GPU<->FPGA bridge (inline copy)."""

    CMD_SET_VG      = 0x01
    CMD_READ_TELEM  = 0x02
    CMD_SET_KILL    = 0x03
    CMD_SET_MAC     = 0x06
    CMD_CLOSED_LOOP = 0x10
    CMD_SET_BAUD    = 0xF0
    CMD_PING        = 0xAA
    SYNC            = 0x55

    def __init__(self, port='/dev/ttyUSB1', baudrate=921600,
                 timeout=0.5, auto_negotiate=True):
        import serial
        import threading
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
        if self._baudrate == 921600:
            if self._try_ping():
                return
            self.ser.close()
            self._baudrate = 115200
            self.ser = serial.Serial(self._port, 115200, timeout=self._timeout)
            time.sleep(0.05)
            self.ser.reset_input_buffer()
            self._try_ping()

    def _try_ping(self, retries=3):
        import serial
        for _ in range(retries):
            try:
                self.ser.reset_input_buffer()
                self.ser.write(bytes([self.SYNC, self.CMD_PING, 0x00]))
                self.ser.flush()
                resp = self._read_response_raw(timeout=0.2)
                if resp is not None and resp.get('type') == self.CMD_PING:
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
            self.ser.timeout = min(deadline - time.monotonic(), 0.02)
            chunk = self.ser.read(3 - len(buf))
            if chunk:
                buf.extend(chunk)
        if len(buf) < 3:
            return None
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
        return {'type': buf[1], 'len': payload_len, 'payload': payload}

    def _send_and_recv(self, cmd, payload=b'', timeout=0.5):
        with self._lock:
            self.ser.reset_input_buffer()
            t0 = time.monotonic()
            self._send_cmd(cmd, payload)
            resp = self._read_response_raw(timeout=timeout)
            self.last_latency_ms = (time.monotonic() - t0) * 1000.0
            self.latency_history.append(self.last_latency_ms)
            return resp

    def ping(self):
        resp = self._send_and_recv(self.CMD_PING, timeout=0.2)
        return resp is not None and resp.get('type') == self.CMD_PING

    def set_gate_voltage(self, neuron_id, vg):
        vg_q = to_q16_16(vg)
        payload = bytes([neuron_id & 0x07]) + struct.pack('>I', vg_q)
        self._send_cmd(self.CMD_SET_VG, payload)
        time.sleep(0.005)

    def set_kill_switch(self, enabled):
        payload = bytes([0x01 if enabled else 0x00])
        self._send_cmd(self.CMD_SET_KILL, payload)
        time.sleep(0.005)

    def set_mac_signal(self, mac_val):
        mac_q = to_q16_16(max(0.0, min(255.0, mac_val)))
        payload = struct.pack('>I', mac_q)
        self._send_cmd(self.CMD_SET_MAC, payload)
        time.sleep(0.005)

    def read_telemetry(self):
        # Fast path: send command, read fixed-size 52-byte response
        with self._lock:
            self.ser.reset_input_buffer()
            self.ser.write(bytes([self.SYNC, self.CMD_READ_TELEM]))
            self.ser.flush()
            self.ser.timeout = 0.05
            raw = self.ser.read(52)
        if len(raw) < 52 or raw[0] != self.SYNC:
            return None
        if raw[1] != self.CMD_READ_TELEM or raw[2] != 0x30:
            return None
        if crc8(bytes(raw[:51])) != raw[51]:
            return None
        data = bytes(raw[3:51])
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
        return result

    def close(self):
        try:
            self.ser.close()
        except Exception:
            pass


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SIMULATED FPGA FALLBACK
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class SimulatedFPGA:
    """Minimal FPGA simulator for testing without hardware."""

    def __init__(self):
        self.vg = [0.35] * N_PHYS_NEURONS
        self.temp_k = 300.0
        self.mac_val = 0.5
        self.kill = False
        self.telemetry_history = deque(maxlen=1000)
        self.spike_history = deque(maxlen=10000)
        self._step = 0

    def set_gate_voltage(self, neuron_id, vg):
        if 0 <= neuron_id < N_PHYS_NEURONS:
            self.vg[neuron_id] = vg

    def set_kill_switch(self, enabled):
        self.kill = enabled

    def set_mac_signal(self, mac_val):
        self.mac_val = mac_val

    def read_telemetry(self):
        self._step += 1
        neurons = []
        for i in range(N_PHYS_NEURONS):
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
# VIRTUAL NEURON & SCALED BANK
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class VirtualNeuron:
    """Virtual neuron mapped to a physical FPGA neuron.

    Inherits REAL avalanche physics from the physical neuron but adds
    spatial decorrelation through:
      - vg_offset: slight gate voltage jitter (N(0, 0.02))
      - noise_std: independent noise on vmem (Uniform(0.05, 0.15))
      - refractory_jitter: random refractory period (Uniform(0, 3) timesteps)
      - spike_prob_base: Bernoulli probability modulated by offset
    """

    def __init__(self, virtual_id: int, n_physical: int = N_PHYS_NEURONS):
        self.virtual_id = virtual_id
        self.physical_id = virtual_id % n_physical  # round-robin mapping
        self.vg_offset = np.random.normal(0.0, 0.02)
        self.noise_std = np.random.uniform(0.05, 0.15)
        self.refractory_jitter = int(np.random.uniform(0, 4))  # 0..3
        self._refractory_counter = 0
        # Base spike probability modulation from offset
        # Neurons with positive offset are slightly more excitable
        self._prob_mod = 1.0 / (1.0 + np.exp(-self.vg_offset * 50.0))  # sigmoid

    def generate_spike(self, physical_spiked: bool, physical_vmem: float,
                       threshold: float = 0.3) -> bool:
        """Decide whether this virtual neuron fires.

        Spike if:
          1. Physical neuron spiked (inherits real physics)
          2. vmem + noise > threshold (independent noise)
          3. Not in refractory period
          4. Independent Bernoulli draw modulated by vg_offset
        """
        # Refractory period
        if self._refractory_counter > 0:
            self._refractory_counter -= 1
            return False

        if not physical_spiked:
            # Can still fire from noise if vmem is close to threshold
            noisy_vmem = physical_vmem + np.random.normal(0, self.noise_std)
            if noisy_vmem > threshold * 1.2:  # higher bar for noise-only spikes
                if np.random.random() < self._prob_mod * 0.3:
                    self._refractory_counter = self.refractory_jitter
                    return True
            return False

        # Physical neuron spiked — apply decorrelation
        noisy_vmem = physical_vmem + np.random.normal(0, self.noise_std)
        if noisy_vmem <= threshold:
            return False

        # Bernoulli gate modulated by offset
        if np.random.random() < self._prob_mod:
            self._refractory_counter = self.refractory_jitter
            return True
        return False


class ScaledBank:
    """Virtual neuron bank scaling N physical neurons to M virtual neurons."""

    def __init__(self, n_virtual: int, n_physical: int = N_PHYS_NEURONS):
        self.n_virtual = n_virtual
        self.n_physical = n_physical
        self.neurons = [VirtualNeuron(i, n_physical) for i in range(n_virtual)]
        # Spike threshold derived from Vg (set per-regime)
        self.threshold = 0.3

    def update(self, telem: dict, spike_threshold: int = 10) -> np.ndarray:
        """Generate binary spike vector from FPGA telemetry.

        Uses spike_count (delta from last read) to determine if physical
        neuron was active. Each virtual neuron inherits from its physical
        parent with independent noise for decorrelation.

        Args:
            telem: telemetry dict with 'neurons' list (spike_count, vmem per neuron)
            spike_threshold: min spike_count to consider physical neuron active

        Returns:
            np.ndarray of shape (n_virtual,), dtype=int, binary spike vector
        """
        spikes = np.zeros(self.n_virtual, dtype=int)
        phys_neurons = telem['neurons']

        for vn in self.neurons:
            pid = vn.physical_id
            pn = phys_neurons[pid]
            # Use spike_count directly — firmware resets on read (delta mode)
            physical_spiked = pn['spike_count'] >= spike_threshold
            # Vmem in Q8.8 scale (0-256), normalize to 0-1 for threshold comparison
            physical_vmem = pn['vmem'] / 256.0
            if vn.generate_spike(physical_spiked, physical_vmem, self.threshold):
                spikes[vn.virtual_id] = 1

        return spikes

    def collect_raster(self, bridge, n_steps: int, vg: float) -> np.ndarray:
        """Collect full spike raster from FPGA.

        Uses heterogeneous Vg per neuron (spread around centre) to create
        variable firing rates and produce genuine avalanche structure.

        Args:
            bridge: FPGA bridge (real or simulated)
            n_steps: number of timesteps
            vg: gate voltage centre for this regime

        Returns:
            np.ndarray of shape (n_steps, n_virtual), binary raster
        """
        # Set HETEROGENEOUS Vg per neuron — spread ±0.08 around centre
        # This creates differential firing rates → avalanche structure
        vg_spread = 0.08
        neuron_vgs = []
        for nid in range(self.n_physical):
            offset = (nid - self.n_physical / 2) * (2 * vg_spread / self.n_physical)
            vg_n = max(0.0, min(1.0, vg + offset))
            neuron_vgs.append(vg_n)
            bridge.set_gate_voltage(nid, vg_n)
        time.sleep(0.05)  # let Vg settle

        # Calibrate: read a few samples to find per-neuron spike rates
        cal_rates = [[] for _ in range(self.n_physical)]
        for _ in range(15):
            telem = bridge.read_telemetry()
            if telem:
                for i, n in enumerate(telem['neurons']):
                    cal_rates[i].append(n['spike_count'])

        # Per-neuron spike thresholds: set above median to create ~30-50% active rate
        per_neuron_thresh = []
        for i in range(self.n_physical):
            if cal_rates[i]:
                med = np.median(cal_rates[i])
                # Threshold slightly above median → ~50% active frames per neuron
                per_neuron_thresh.append(max(1, int(med * 1.1)))
            else:
                per_neuron_thresh.append(5)

        self.threshold = 0.3  # vmem threshold in normalized space

        raster = np.zeros((n_steps, self.n_virtual), dtype=int)
        for t in range(n_steps):
            telem = bridge.read_telemetry()
            if telem is None:
                continue
            # Custom per-neuron thresholding
            spikes = np.zeros(self.n_virtual, dtype=int)
            phys_neurons = telem['neurons']
            for vn in self.neurons:
                pid = vn.physical_id
                pn = phys_neurons[pid]
                physical_spiked = pn['spike_count'] >= per_neuron_thresh[pid]
                physical_vmem = pn['vmem'] / 256.0
                if vn.generate_spike(physical_spiked, physical_vmem, self.threshold):
                    spikes[vn.virtual_id] = 1
            raster[t] = spikes

        return raster


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# AVALANCHE DETECTION
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def detect_avalanches(raster: np.ndarray):
    """Detect neuronal avalanches from a spike raster.

    An avalanche is a contiguous sequence of frames where at least
    one neuron is active.

    Args:
        raster: (n_steps, n_neurons) binary array

    Returns:
        sizes: list of avalanche sizes (total spikes in each avalanche)
        durations: list of avalanche durations (number of frames)
    """
    n_steps = raster.shape[0]
    frame_counts = raster.sum(axis=1)  # spikes per frame

    sizes = []
    durations = []
    in_avalanche = False
    current_size = 0
    current_duration = 0

    for t in range(n_steps):
        if frame_counts[t] > 0:
            if not in_avalanche:
                in_avalanche = True
                current_size = 0
                current_duration = 0
            current_size += int(frame_counts[t])
            current_duration += 1
        else:
            if in_avalanche:
                sizes.append(current_size)
                durations.append(current_duration)
                in_avalanche = False
                current_size = 0
                current_duration = 0

    # Close any trailing avalanche
    if in_avalanche:
        sizes.append(current_size)
        durations.append(current_duration)

    return sizes, durations


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# POWER-LAW FITTING
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def power_law_mle(data, x_min=1):
    """MLE estimate of power-law exponent alpha for P(x) ~ x^(-alpha).

    Uses the Clauset et al. (2009) discrete MLE:
      alpha = 1 + n * [sum_i ln(x_i / (x_min - 0.5))]^(-1)

    Args:
        data: array of positive integers (avalanche sizes)
        x_min: minimum value for the fit

    Returns:
        alpha: estimated exponent
        n_tail: number of data points >= x_min
    """
    data = np.asarray(data, dtype=float)
    tail = data[data >= x_min]
    n = len(tail)
    if n < 5:
        return float('nan'), 0

    alpha = 1.0 + n / np.sum(np.log(tail / (x_min - 0.5)))
    return float(alpha), n


def ks_test_power_law(data, alpha, x_min=1):
    """KS goodness-of-fit test for discrete power-law.

    Computes the KS statistic and an approximate p-value using
    Monte Carlo simulation.

    Returns:
        ks_stat: KS statistic
        p_value: approximate p-value (fraction of synthetic datasets
                 with KS >= observed KS)
    """
    data = np.asarray(data, dtype=float)
    tail = data[data >= x_min]
    n = len(tail)
    if n < 10 or np.isnan(alpha):
        return float('nan'), 0.0

    # Empirical CDF
    sorted_data = np.sort(tail)
    ecdf = np.arange(1, n + 1) / n

    # Theoretical CDF for discrete power-law
    x_vals = np.arange(x_min, int(sorted_data[-1]) + 1)
    pmf = x_vals.astype(float) ** (-alpha)
    pmf /= pmf.sum()
    tcdf_dict = {}
    cumsum = 0.0
    for xv, p in zip(x_vals, pmf):
        cumsum += p
        tcdf_dict[int(xv)] = cumsum

    tcdf = np.array([tcdf_dict.get(int(x), 1.0) for x in sorted_data])
    ks_stat = float(np.max(np.abs(ecdf - tcdf)))

    # Monte Carlo p-value (100 synthetic datasets)
    n_mc = 100
    count_ge = 0
    for _ in range(n_mc):
        # Generate synthetic power-law data
        syn = _sample_discrete_power_law(alpha, x_min, n)
        syn_alpha, _ = power_law_mle(syn, x_min)
        if np.isnan(syn_alpha):
            continue
        syn_sorted = np.sort(syn.astype(float))
        syn_ecdf = np.arange(1, n + 1) / n

        x_max_syn = int(syn_sorted[-1])
        x_vals_syn = np.arange(x_min, x_max_syn + 1)
        pmf_syn = x_vals_syn.astype(float) ** (-syn_alpha)
        pmf_syn /= pmf_syn.sum()
        tcdf_syn_dict = {}
        cs = 0.0
        for xv, p in zip(x_vals_syn, pmf_syn):
            cs += p
            tcdf_syn_dict[int(xv)] = cs
        tcdf_syn = np.array([tcdf_syn_dict.get(int(x), 1.0) for x in syn_sorted])
        ks_syn = float(np.max(np.abs(syn_ecdf - tcdf_syn)))

        if ks_syn >= ks_stat:
            count_ge += 1

    p_value = count_ge / n_mc
    return ks_stat, p_value


def _sample_discrete_power_law(alpha, x_min, n):
    """Sample n values from a discrete power-law P(x) ~ x^(-alpha), x >= x_min."""
    # Inverse CDF method with upper bound
    x_max = max(x_min * 100, 10000)
    x_vals = np.arange(x_min, x_max + 1)
    pmf = x_vals.astype(float) ** (-alpha)
    pmf /= pmf.sum()
    cdf = np.cumsum(pmf)
    u = np.random.random(n)
    samples = np.searchsorted(cdf, u) + x_min
    return samples


def fit_exponential_cutoff(data, x_min=1):
    """Fit P(s) ~ s^(-alpha) * exp(-s/s_cut) via grid search.

    Returns:
        alpha: exponent
        s_cut: cutoff scale
    """
    data = np.asarray(data, dtype=float)
    tail = data[data >= x_min]
    n = len(tail)
    if n < 10:
        return float('nan'), float('nan')

    # Grid search over alpha and s_cut
    best_ll = -np.inf
    best_alpha = 1.5
    best_scut = 100.0

    s_max = float(np.max(tail))
    for alpha_try in np.linspace(0.5, 3.5, 30):
        for scut_try in np.logspace(np.log10(max(1.0, s_max * 0.01)),
                                     np.log10(s_max * 10), 20):
            # Log-likelihood
            x_vals = np.arange(x_min, int(s_max) + 1)
            log_pmf = -alpha_try * np.log(x_vals) - x_vals / scut_try
            log_Z = np.log(np.sum(np.exp(log_pmf - np.max(log_pmf)))) + np.max(log_pmf)
            ll = np.sum(-alpha_try * np.log(tail) - tail / scut_try) - n * log_Z
            if ll > best_ll:
                best_ll = ll
                best_alpha = alpha_try
                best_scut = scut_try

    return float(best_alpha), float(best_scut)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# BRANCHING RATIO
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def compute_branching_ratio(raster: np.ndarray):
    """Compute branching ratio sigma(t) = n_active(t+1) / n_active(t).

    Only computed for frames where n_active(t) > 0.

    Returns:
        mean: mean branching ratio
        std: standard deviation
        ci_95: (lower, upper) 95% CI
        values: list of all sigma(t) values
    """
    frame_counts = raster.sum(axis=1)
    sigmas = []

    for t in range(len(frame_counts) - 1):
        if frame_counts[t] > 0:
            sigmas.append(frame_counts[t + 1] / frame_counts[t])

    if len(sigmas) < 3:
        return float('nan'), float('nan'), (float('nan'), float('nan')), []

    sigmas = np.array(sigmas)
    mean = float(np.mean(sigmas))
    std = float(np.std(sigmas))
    n = len(sigmas)
    se = std / np.sqrt(n)
    ci_95 = (mean - 1.96 * se, mean + 1.96 * se)

    return mean, std, ci_95, sigmas.tolist()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# DYNAMIC RANGE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def compute_dynamic_range_dB(raster: np.ndarray):
    """Dynamic range from spike rate distribution.

    DR = 10 * log10(max_rate / min_rate) where rates are per-neuron
    firing rates over the raster.
    """
    n_steps = raster.shape[0]
    if n_steps == 0:
        return 0.0
    rates = raster.sum(axis=0) / n_steps
    rates_nonzero = rates[rates > 0]
    if len(rates_nonzero) < 2:
        return 0.0
    ratio = np.max(rates_nonzero) / np.min(rates_nonzero)
    return float(10.0 * np.log10(ratio)) if ratio > 0 else 0.0


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# MAIN EXPERIMENT
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def connect_fpga():
    """Try real FPGA, fall back to SimulatedFPGA."""
    # FPGA firmware doesn't respond to CMD_PING (0xAA) — use telemetry read
    for baud in [FPGA_BAUD_FAST, FPGA_BAUD_SLOW]:
        try:
            bridge = NSRAMFPGABridgeFast(
                port=FPGA_PORT, baudrate=baud, auto_negotiate=False
            )
            telem = bridge.read_telemetry()
            if telem is not None:
                print(f"  FPGA connected on {FPGA_PORT} @ {baud} baud")
                return bridge, False
            bridge.close()
        except Exception as e:
            print(f"  FPGA @ {baud}: {e}")

    print("  Using SimulatedFPGA fallback")
    return SimulatedFPGA(), True


def run_experiment():
    """Run scaled avalanche statistics across scales and regimes."""
    print("=" * 70)
    print("z2145v36: Scaled Avalanche Statistics")
    print("=" * 70)

    bridge, simulated = connect_fpga()

    scales = [64, 128, 256]
    regimes = {
        'subcritical':   0.15,
        'critical':      0.35,
        'supercritical': 0.50,
    }
    n_steps = 1000

    all_results = {
        'experiment': 'z2145_scaled_avalanche_v36',
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'simulated': simulated,
        'n_steps': n_steps,
        'scales': scales,
        'regimes': {k: v for k, v in regimes.items()},
        'runs': {},
    }

    # Summary table header
    print(f"\n{'Scale':>6}  {'Regime':>14}  {'Vg':>5}  {'#Aval':>6}  "
          f"{'alpha':>6}  {'KS_p':>6}  {'BR_mu':>6}  {'BR_sd':>6}  "
          f"{'LZc':>6}  {'DR_dB':>6}")
    print("-" * 100)

    for N in scales:
        bank = ScaledBank(n_virtual=N)
        all_results['runs'][str(N)] = {}

        for regime_name, vg in regimes.items():
            print(f"\n  [N={N}, {regime_name}, Vg={vg}] Collecting {n_steps} timesteps...",
                  flush=True)
            t0 = time.monotonic()

            raster = bank.collect_raster(bridge, n_steps, vg)
            collect_time = time.monotonic() - t0

            # Avalanche detection
            sizes, durations = detect_avalanches(raster)
            n_avalanches = len(sizes)

            # Power-law fit
            if n_avalanches >= 5:
                alpha, n_tail = power_law_mle(sizes, x_min=1)
                ks_stat, ks_p = ks_test_power_law(sizes, alpha, x_min=1)
                alpha_cut, s_cut = fit_exponential_cutoff(sizes, x_min=1)
            else:
                alpha, n_tail = float('nan'), 0
                ks_stat, ks_p = float('nan'), 0.0
                alpha_cut, s_cut = float('nan'), float('nan')

            # Branching ratio
            br_mean, br_std, br_ci, br_values = compute_branching_ratio(raster)

            # Lempel-Ziv complexity on total spike count time series
            frame_counts = raster.sum(axis=1)
            binary_str = spike_train_to_binary(frame_counts)
            lzc = lempel_ziv_complexity(binary_str)

            # Dynamic range
            dr_dB = compute_dynamic_range_dB(raster)

            # Size histogram (for plotting)
            if sizes:
                size_counts = Counter(sizes)
                size_hist = sorted(size_counts.items())
            else:
                size_hist = []

            # Raster statistics
            total_spikes = int(raster.sum())
            mean_rate = float(raster.mean())
            active_frac = float((frame_counts > 0).sum() / n_steps)

            run_result = {
                'scale': N,
                'regime': regime_name,
                'vg': vg,
                'n_avalanches': n_avalanches,
                'avalanche_sizes': sizes[:500] if len(sizes) > 500 else sizes,  # cap for JSON
                'avalanche_durations': durations[:500] if len(durations) > 500 else durations,
                'size_histogram': size_hist[:100],
                'power_law_alpha': alpha if not np.isnan(alpha) else None,
                'power_law_n_tail': n_tail,
                'ks_stat': ks_stat if not np.isnan(ks_stat) else None,
                'ks_p_value': ks_p,
                'exp_cutoff_alpha': alpha_cut if not np.isnan(alpha_cut) else None,
                'exp_cutoff_s_cut': s_cut if not np.isnan(s_cut) else None,
                'branching_ratio_mean': br_mean if not np.isnan(br_mean) else None,
                'branching_ratio_std': br_std if not np.isnan(br_std) else None,
                'branching_ratio_ci95': list(br_ci) if not np.isnan(br_ci[0]) else None,
                'lzc': lzc,
                'dynamic_range_dB': dr_dB,
                'total_spikes': total_spikes,
                'mean_firing_rate': mean_rate,
                'active_frame_fraction': active_frac,
                'collect_time_s': round(collect_time, 2),
            }
            all_results['runs'][str(N)][regime_name] = run_result

            # Print summary row
            alpha_s = f"{alpha:.3f}" if not np.isnan(alpha) else "  N/A"
            ks_p_s = f"{ks_p:.3f}" if not np.isnan(ks_p) else "  N/A"
            br_m_s = f"{br_mean:.3f}" if not np.isnan(br_mean) else "  N/A"
            br_s_s = f"{br_std:.3f}" if not np.isnan(br_std) else "  N/A"
            print(f"  {N:>6}  {regime_name:>14}  {vg:>5.2f}  {n_avalanches:>6}  "
                  f"{alpha_s:>6}  {ks_p_s:>6}  {br_m_s:>6}  {br_s_s:>6}  "
                  f"{lzc:>6.3f}  {dr_dB:>6.1f}")

    # ── Pass/Fail checks ──
    print("\n" + "=" * 70)
    print("PASS/FAIL CHECKS (informal)")
    print("=" * 70)

    checks = {}

    # Check 1: avalanche_statistics_valid — n_avalanches >= 100 at critical for N=128
    crit_128 = all_results['runs'].get('128', {}).get('critical', {})
    n_aval_128_crit = crit_128.get('n_avalanches', 0)
    checks['avalanche_statistics_valid'] = n_aval_128_crit >= 100
    print(f"  avalanche_statistics_valid: n_aval(N=128, critical) = {n_aval_128_crit} "
          f"{'PASS' if checks['avalanche_statistics_valid'] else 'FAIL'} (need >= 100)")

    # Check 2: power_law_plausible — alpha in [1.0, 3.0] at critical
    alpha_128_crit = crit_128.get('power_law_alpha')
    if alpha_128_crit is not None:
        checks['power_law_plausible'] = 1.0 <= alpha_128_crit <= 3.0
        print(f"  power_law_plausible: alpha(N=128, critical) = {alpha_128_crit:.3f} "
              f"{'PASS' if checks['power_law_plausible'] else 'FAIL'} (need [1.0, 3.0])")
    else:
        checks['power_law_plausible'] = False
        print(f"  power_law_plausible: alpha=N/A FAIL")

    # Check 3: regime_differentiation — BR_sub < BR_crit < BR_super
    br_sub = all_results['runs'].get('128', {}).get('subcritical', {}).get('branching_ratio_mean')
    br_crit = all_results['runs'].get('128', {}).get('critical', {}).get('branching_ratio_mean')
    br_super = all_results['runs'].get('128', {}).get('supercritical', {}).get('branching_ratio_mean')
    if br_sub is not None and br_crit is not None and br_super is not None:
        checks['regime_differentiation'] = br_sub < br_crit < br_super
        print(f"  regime_differentiation: BR_sub={br_sub:.3f} < BR_crit={br_crit:.3f} "
              f"< BR_super={br_super:.3f} "
              f"{'PASS' if checks['regime_differentiation'] else 'FAIL'}")
    else:
        checks['regime_differentiation'] = False
        print(f"  regime_differentiation: BR values unavailable FAIL")

    all_results['checks'] = checks
    n_pass = sum(1 for v in checks.values() if v)
    n_total = len(checks)
    all_results['summary'] = f"{n_pass}/{n_total} checks passed"

    print(f"\n  SUMMARY: {n_pass}/{n_total} checks passed")

    # ── Detailed analysis ──
    print("\n" + "=" * 70)
    print("DETAILED REGIME COMPARISON (N=128)")
    print("=" * 70)
    for regime_name in ['subcritical', 'critical', 'supercritical']:
        r = all_results['runs'].get('128', {}).get(regime_name, {})
        if not r:
            continue
        print(f"\n  {regime_name} (Vg={r.get('vg', '?')}):")
        print(f"    Avalanches:      {r.get('n_avalanches', 0)}")
        print(f"    PL alpha:        {r.get('power_law_alpha', 'N/A')}")
        print(f"    KS p-value:      {r.get('ks_p_value', 'N/A')}")
        print(f"    Cutoff alpha:    {r.get('exp_cutoff_alpha', 'N/A')}")
        print(f"    Cutoff s_cut:    {r.get('exp_cutoff_s_cut', 'N/A')}")
        print(f"    Branching ratio: {r.get('branching_ratio_mean', 'N/A')} "
              f"+/- {r.get('branching_ratio_std', 'N/A')}")
        print(f"    LZc:             {r.get('lzc', 'N/A')}")
        print(f"    Dynamic range:   {r.get('dynamic_range_dB', 'N/A')} dB")
        print(f"    Mean firing rate:{r.get('mean_firing_rate', 'N/A')}")
        print(f"    Active frames:   {r.get('active_frame_fraction', 'N/A')}")

    # ── Save results ──
    RESULTS_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_JSON, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to {RESULTS_JSON}")

    bridge.close()
    return all_results


if __name__ == '__main__':
    run_experiment()
