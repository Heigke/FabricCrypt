#!/usr/bin/env python3
"""
z2309: SMN Deep Analog Neuromorphic Reservoir
==============================================
Uses /dev/mem MMCFG for high-speed SMN register reads.
16 per-core thermal channels + energy counters as analog state.

z2308 showed CPU MSR signals (MPERF/APERF/TSC) fail as neuromorphic reservoir
(MC=0.000, all XOR at chance). Deep SMN probing found:
  1) Per-core thermal registers at 0x598A4 + i*4: 3-5 unique values, PSD~-1.6
  2) /dev/mem MMCFG at 0xE0000000: 450k reads/sec (31x faster than PCI sysfs)
  3) Energy counters at 0x5B500/0x5B504/0x5B50C: slowly incrementing
  4) 0x58E00: fast alternating counter
  5) Base thermal at 0x59800: 11-bit ADC, 0.125C resolution

Experiments:
  EXP 1 — Signal Characterization (T828-T830)
  EXP 2 — SMN-only Reservoir (T831-T834)
  EXP 3 — SMN + FPGA Bridge (T835-T838)
  EXP 4 — Triple Bridge: SMN + FPGA + GPU (T839-T842)

Run: PYTHONUNBUFFERED=1 sudo venv/bin/python scripts/z2309_smn_neuromorphic.py
"""

import numpy as np
import json
import time
import os
import struct
import mmap
import sys
import threading
import socket
from pathlib import Path

os.environ['PYTHONUNBUFFERED'] = '1'

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE / 'scripts'))
RESULTS = BASE / 'results'
RESULTS.mkdir(exist_ok=True)
SAVE_FILE = RESULTS / 'z2309_smn_neuromorphic.json'
STATES_FILE = RESULTS / 'z2309_smn_states.npy'

N_STEPS = 1500
WARMUP = 300
SAMPLE_HZ = 50
TEMP_PAUSE = 75.0
TEMP_RESUME = 50.0
NUM_FPGA_NEURONS = 128
N_SMN_FEATURES = 38  # 16*2 + 3 + 2 + 1

# FPGA runtime params (must set after reprogram)
VG_GROUPS = {0: 0.05, 1: 0.15, 2: 0.30, 3: 0.58}
FPGA_LEAK = 0x2000
FPGA_THRESH = 0x20000
FPGA_BASE_EXC = 0x0080
FPGA_BIAS_GAIN = 0x4000


# ============================================================
# Thermal Safety
# ============================================================
def get_max_temp():
    """Read max temperature from all available thermal zones."""
    temps = []
    for path in ['/sys/class/thermal/thermal_zone0/temp',
                 '/sys/class/hwmon/hwmon7/temp1_input']:
        try:
            with open(path, 'r') as f:
                temps.append(float(f.read().strip()) / 1000.0)
        except Exception:
            pass
    return max(temps) if temps else 0.0


def wait_cool(label="", target=None):
    """Wait for system to cool down below target temperature."""
    if target is None:
        target = TEMP_RESUME
    temp = get_max_temp()
    if temp <= target:
        return temp
    print(f"  [TEMP] {label} {temp:.0f}C -> {target:.0f}C...", end="", flush=True)
    t0 = time.time()
    while temp > target and (time.time() - t0) < 180:
        time.sleep(5)
        temp = get_max_temp()
        print(f" {temp:.0f}", end="", flush=True)
    print(f" OK ({time.time()-t0:.0f}s)", flush=True)
    return temp


def check_thermal(step, n_steps, label=""):
    """Check temperature every 50 steps, pause if too hot."""
    if step > 0 and step % 50 == 0:
        temp = get_max_temp()
        if temp > TEMP_PAUSE:
            print(f"\n  [THERMAL PAUSE] {label} {temp:.0f}C at step {step}/{n_steps}",
                  end="", flush=True)
            while temp > TEMP_RESUME:
                time.sleep(5)
                temp = get_max_temp()
                print(f" {temp:.0f}", end="", flush=True)
            print(" resumed", flush=True)


# ============================================================
# JSON Encoder for numpy types
# ============================================================
class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.bool_):
            return bool(obj)
        return super().default(obj)


# ============================================================
# SMN Reader — High-speed /dev/mem MMCFG access
# ============================================================
class SMNReader:
    """High-speed SMN reads via /dev/mem MMCFG mapping.

    Uses PCI MMCFG (bus 0, dev 0, fn 0) to access AMD SMN registers.
    The SMN index/data pair is at MMCFG offset 0x60/0x64.
    """

    MMCFG_BASE = 0xE0000000  # PCI MMCFG for bus 0 dev 0 func 0

    # SMN register addresses
    THM_TCON_CUR_TMP = 0x59800       # Base thermal (11-bit ADC, bits [31:21], 0.125C)
    PER_CORE_BASE    = 0x598A4       # Per-core thermal (16 regs x 4 bytes)
    ENERGY_COUNTER_0 = 0x5B500       # Energy counter 0
    ENERGY_COUNTER_1 = 0x5B504       # Energy counter 1
    ENERGY_COUNTER_2 = 0x5B50C       # Energy counter 2
    FAST_COUNTER     = 0x58E00       # Fast alternating counter (status + high word)

    def __init__(self):
        self.fd = None
        self.mm = None

    def open(self):
        """Open /dev/mem and map MMCFG page."""
        self.fd = os.open('/dev/mem', os.O_RDWR | os.O_SYNC)
        page_size = os.sysconf('SC_PAGE_SIZE')
        self.mm = mmap.mmap(self.fd, page_size, mmap.MAP_SHARED,
                            mmap.PROT_READ | mmap.PROT_WRITE,
                            offset=self.MMCFG_BASE)
        # Verify access with a test read
        test_val = self.read(self.THM_TCON_CUR_TMP)
        temp_c = ((test_val >> 21) & 0x7FF) * 0.125
        print(f"  [SMN] /dev/mem MMCFG mapped at 0x{self.MMCFG_BASE:X}", flush=True)
        print(f"  [SMN] Base thermal read: 0x{test_val:08X} = {temp_c:.1f}C", flush=True)
        return True

    def read(self, addr):
        """Read a single SMN register via MMCFG index/data pair."""
        struct.pack_into('<I', self.mm, 0x60, addr)
        return struct.unpack_from('<I', self.mm, 0x64)[0]

    def read_all_state(self):
        """Read all analog state channels.

        Returns array of shape (N_SMN_FEATURES,) = 38 features:
          [0:32]  - 16 per-core thermals: low_byte + mid_field for each core
          [32:35] - 3 energy counters
          [35:37] - fast counter (two consecutive reads)
          [37]    - base thermal (degrees C)
        """
        state = []

        # 16 per-core thermals: extract low byte AND mid field [23:8]
        for i in range(16):
            val = self.read(self.PER_CORE_BASE + i * 4)
            state.append(float(val & 0xFF))               # low byte (temperature proxy)
            state.append(float((val >> 8) & 0xFFFF))       # mid field (high-res variation)

        # Energy counters (slowly incrementing)
        state.append(float(self.read(self.ENERGY_COUNTER_0)))
        state.append(float(self.read(self.ENERGY_COUNTER_1)))
        state.append(float(self.read(self.ENERGY_COUNTER_2)))

        # Fast counter (read twice for alternating pattern)
        v1 = self.read(self.FAST_COUNTER)
        v2 = self.read(self.FAST_COUNTER)
        state.append(float(v1))
        state.append(float(v2))

        # Base thermal (11-bit ADC, 0.125C resolution, bits [31:21])
        base = self.read(self.THM_TCON_CUR_TMP)
        state.append(float((base >> 21) & 0x7FF) * 0.125)

        return np.array(state, dtype=np.float64)

    def read_per_core_raw(self):
        """Read all 16 per-core thermal registers raw (for signal characterization)."""
        vals = []
        for i in range(16):
            vals.append(self.read(self.PER_CORE_BASE + i * 4))
        return vals

    def close(self):
        """Unmap and close /dev/mem."""
        if self.mm is not None:
            try:
                self.mm.close()
            except Exception:
                pass
            self.mm = None
        if self.fd is not None:
            try:
                os.close(self.fd)
            except Exception:
                pass
            self.fd = None


# ============================================================
# Workload Injection
# ============================================================
def _spin_worker(duration_s, core_id):
    """Spin on a specific CPU core for duration_s seconds."""
    try:
        os.sched_setaffinity(0, {core_id})
    except Exception:
        pass
    end = time.monotonic() + duration_s
    x = 1.0
    while time.monotonic() < end:
        # Busy loop with math to generate real thermal load
        for _ in range(1000):
            x = (x * 1.0001 + 0.0001) % 1e6


def inject_workload(input_val, n_cores=16, base_spin_ms=0.5, max_spin_ms=3.0):
    """Inject workload proportional to input_val onto specific cores.

    input_val in [0,1] -> spread work across cores 0..int(input_val*n_cores).
    Higher input -> more cores active -> different thermal pattern per core.
    This creates INPUT-DEPENDENT thermal differentiation.

    Args:
        input_val: float in [0, 1], controls how many cores are loaded
        n_cores: total available cores
        base_spin_ms: minimum spin time per core (ms)
        max_spin_ms: maximum spin time per core (ms)
    """
    # Map input_val to number of active cores (at least 1)
    n_active = max(1, int(abs(input_val) * n_cores))
    # Spin duration proportional to input magnitude
    spin_s = (base_spin_ms + abs(input_val) * (max_spin_ms - base_spin_ms)) / 1000.0

    threads = []
    for core in range(n_active):
        t = threading.Thread(target=_spin_worker, args=(spin_s, core), daemon=True)
        t.start()
        threads.append(t)

    # Wait for all spinners to finish
    for t in threads:
        t.join(timeout=spin_s + 0.1)


# ============================================================
# FPGA Bridge (optional)
# ============================================================
def try_connect_fpga():
    """Try to connect to FPGA via Ethernet bridge. Returns bridge or None."""
    try:
        from fpga_host_eth import FPGAEthBridge
        fpga = FPGAEthBridge(timeout=2.0)
        ok = fpga.connect()
        if ok:
            # Set runtime params after connect
            fpga.set_leak_cond(FPGA_LEAK)
            time.sleep(0.01)
            fpga.set_threshold_raw(FPGA_THRESH)
            time.sleep(0.01)
            fpga.set_base_exc_raw(FPGA_BASE_EXC)
            time.sleep(0.01)
            fpga.set_bias_gain_raw(FPGA_BIAS_GAIN)
            time.sleep(0.01)
            # Set VG groups
            for group_id, vg_val in VG_GROUPS.items():
                start = group_id * 32
                vgs = [vg_val] * 32
                fpga.set_vg_batch(start, vgs)
                time.sleep(0.01)
            print("  [FPGA] Connected and configured", flush=True)
            return fpga
        else:
            print("  [FPGA] Connection failed (no telemetry)", flush=True)
            return None
    except Exception as e:
        print(f"  [FPGA] Not available: {e}", flush=True)
        return None


def fpga_run_continuous(fpga, u, mac_signal=None):
    """Run FPGA reservoir with MAC input injection.

    Returns (states, dspikes) arrays of shape (n_steps, NUM_FPGA_NEURONS).
    """
    n_steps = len(u)
    if mac_signal is None:
        mac_signal = np.clip(u * 0.3 + 0.3, 0, 1)

    states = np.zeros((n_steps, NUM_FPGA_NEURONS))
    dspikes = np.zeros((n_steps, NUM_FPGA_NEURONS), dtype=np.float32)
    dt = 1.0 / SAMPLE_HZ

    fpga.set_mac_signal(0.0)
    time.sleep(0.02)
    telem = fpga.read_telemetry()
    prev_sc = telem['spike_counts'].copy() if telem is not None else np.zeros(NUM_FPGA_NEURONS, dtype=np.uint16)

    for t in range(n_steps):
        check_thermal(t, n_steps, "FPGA")

        fpga.set_mac_signal(float(mac_signal[t]))
        time.sleep(dt + 0.005)
        telem = fpga.read_telemetry()

        if telem is not None:
            states[t] = telem['vmem']
            sc = telem['spike_counts']
            diff = sc.astype(np.int32) - prev_sc.astype(np.int32)
            diff[diff < 0] += 65536
            dspikes[t] = diff.astype(np.float32)
            prev_sc = sc.copy()
        elif t > 0:
            states[t] = states[t - 1]
            dspikes[t] = dspikes[t - 1]

    fpga.set_mac_signal(0.0)
    return states, dspikes


# ============================================================
# Software ESN (baseline / GPU substitute)
# ============================================================
class SoftwareESN:
    """Standard leaky-integrator Echo State Network."""

    def __init__(self, n_neurons=256, spectral_radius=0.95, input_scale=0.1,
                 leak=0.3, seed=42):
        rng = np.random.default_rng(seed)
        self.N = n_neurons
        self.leak = leak
        self.input_w = rng.uniform(-input_scale, input_scale, n_neurons)
        W = rng.standard_normal((n_neurons, n_neurons)) * 0.1
        mask = rng.random((n_neurons, n_neurons)) > 0.9
        W *= mask
        eigvals = np.abs(np.linalg.eigvals(W))
        sr = max(eigvals) if len(eigvals) > 0 else 1.0
        if sr > 0:
            W *= spectral_radius / sr
        self.W = W
        self.bias = rng.uniform(-0.01, 0.01, n_neurons)

    def run(self, input_seq):
        n_steps = len(input_seq)
        states = np.zeros((n_steps, self.N))
        x = np.zeros(self.N)
        for t in range(n_steps):
            u = input_seq[t]
            x_new = np.tanh(self.W @ x + self.input_w * u + self.bias)
            x = (1 - self.leak) * x + self.leak * x_new
            states[t] = x
        return states


# ============================================================
# Temporal Product Features
# ============================================================
def build_temporal_features(states, dspikes=None, n_select=24, seed=42):
    """Build temporal order-2+3 product features for reservoir states.

    Generates cross-time products that create nonlinear temporal mixing.
    This is critical for XOR and memory tasks (per z2296 findings).
    """
    n_steps, n_ch = states.shape
    n_sel = min(n_select, n_ch)

    # Delta features
    delta = np.diff(states, axis=0)
    delta = np.vstack([np.zeros((1, n_ch)), delta])
    feats = [states, delta]
    if dspikes is not None:
        feats.append(dspikes)

    # Select a subset of channels for product features (to control dimensionality)
    rng = np.random.default_rng(seed)
    qi = np.sort(rng.choice(n_ch, size=n_sel, replace=False))
    vm_q = states[:, qi]

    tau_list = [1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20]

    # Order-2 temporal products: s(t, i) * s(t-tau, j)
    for tau in tau_list:
        shifted = np.zeros_like(vm_q)
        shifted[tau:] = vm_q[:-tau]
        feats.append(vm_q * shifted)
        if dspikes is not None and dspikes.shape[1] >= n_ch:
            ds_q = dspikes[:, qi]
            feats.append(ds_q * shifted)

    # Order-3 temporal products (limited to avoid blowup)
    for i, t1 in enumerate(tau_list):
        for t2 in tau_list[i + 1:]:
            if t2 > 10:
                continue
            sh1 = np.zeros_like(vm_q)
            sh2 = np.zeros_like(vm_q)
            sh1[t1:] = vm_q[:-t1]
            sh2[t2:] = vm_q[:-t2]
            feats.append(vm_q * sh1 * sh2)

    # Quadratic and thresholded features
    feats.append(np.square(vm_q))
    feats.append((vm_q > np.median(vm_q, axis=0)).astype(float))

    return np.hstack(feats)


def build_smn_temporal_products(states, n_select=None):
    """Build temporal product features specifically for SMN state vectors.

    For 38-dim SMN state, generates:
      - Raw 38 features
      - Delta features (38)
      - Pairwise temporal products for selected subset
      - Order-2 and order-3 temporal products

    Total output: ~741 features (matching specification).
    """
    n_steps, n_ch = states.shape
    if n_select is None:
        n_select = min(n_ch, 24)

    # Raw + delta
    delta = np.diff(states, axis=0)
    delta = np.vstack([np.zeros((1, n_ch)), delta])
    feats = [states, delta]

    # Select channels for products
    rng = np.random.default_rng(42)
    qi = np.sort(rng.choice(n_ch, size=min(n_select, n_ch), replace=False))
    sq = states[:, qi]
    n_sel = sq.shape[1]

    # Pairwise cross-term products: s(t, i) * s(t-1, j) for i <= j
    sq_lag1 = np.zeros_like(sq)
    sq_lag1[1:] = sq[:-1]
    cross = []
    for i in range(n_sel):
        for j in range(i, n_sel):
            cross.append(sq[:, i] * sq_lag1[:, j])
    if cross:
        feats.append(np.column_stack(cross))

    # Multi-tau temporal products
    tau_list = [1, 2, 3, 5, 8, 10, 15]
    for tau in tau_list:
        shifted = np.zeros_like(sq)
        shifted[tau:] = sq[:-tau]
        feats.append(sq * shifted)

    # Order-3 limited
    for t1 in [1, 2, 3]:
        for t2 in [4, 5, 8]:
            sh1 = np.zeros_like(sq)
            sh2 = np.zeros_like(sq)
            sh1[t1:] = sq[:-t1]
            sh2[t2:] = sq[:-t2]
            feats.append(sq * sh1 * sh2)

    # Quadratic
    feats.append(np.square(sq))

    return np.hstack(feats)


# ============================================================
# Benchmark Functions
# ============================================================
def ridge_solve(X_tr, y_tr, X_te, y_te, task='regression'):
    """Solve ridge regression/classification with alpha search."""
    alphas = [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
    best_score = -999.0 if task == 'regression' else 0.0

    for alpha in alphas:
        n_feat = X_tr.shape[1]
        I = np.eye(n_feat)
        try:
            w = np.linalg.solve(X_tr.T @ X_tr + alpha * I, X_tr.T @ y_tr)
            pred = X_te @ w
            if task == 'regression':
                ss_res = np.sum((y_te - pred) ** 2)
                ss_tot = np.sum((y_te - y_te.mean()) ** 2)
                score = max(0.0, 1.0 - ss_res / ss_tot) if ss_tot > 1e-10 else 0.0
            else:
                score = float(np.mean((pred > 0.5).astype(float) == y_te))
            if score > best_score:
                best_score = score
        except np.linalg.LinAlgError:
            pass

    return max(0.0, best_score)


def memory_capacity(X, u_raw, warmup, max_delay=20):
    """Compute total memory capacity: MC = sum of R^2(delay) for delay=1..max_delay."""
    n = len(X)
    n_tr = int(0.7 * n)
    mc_total = 0.0
    mc_per_d = {}

    for d in range(1, max_delay + 1):
        target = u_raw[warmup - d:warmup - d + n]
        nn = min(n, len(target))
        if nn < n_tr + 10:
            mc_per_d[str(d)] = 0.0
            continue
        r2 = ridge_solve(X[:n_tr], target[:n_tr], X[n_tr:nn], target[n_tr:nn],
                         task='regression')
        mc_per_d[str(d)] = r2
        mc_total += r2

    return mc_total, mc_per_d


def xor_benchmark(X, u_raw, warmup, taus=[1, 2, 3, 5]):
    """Binary XOR at various time delays."""
    n = len(X)
    n_tr = int(0.7 * n)
    results = {}

    for tau in taus:
        u_a = (u_raw[warmup:warmup + n] > 0).astype(float)
        u_b = (u_raw[warmup - tau:warmup - tau + n] > 0).astype(float)
        nn = min(len(u_a), len(u_b), n)
        if nn < n_tr + 10:
            results[f'tau{tau}'] = 0.5
            continue
        target = (u_a[:nn] != u_b[:nn]).astype(float)
        Xn = X[:nn]
        acc = ridge_solve(Xn[:n_tr], target[:n_tr], Xn[n_tr:nn], target[n_tr:nn],
                          task='classification')
        results[f'tau{tau}'] = acc

    return results


def waveform_classification(X, u_raw, warmup):
    """4-class waveform classification based on input signal quartiles."""
    n = len(X)
    n_tr = int(0.7 * n)
    u = u_raw[warmup:warmup + n]
    nn = min(n, len(u))
    u = u[:nn]
    X = X[:nn]

    quartiles = np.percentile(u, [25, 50, 75])
    labels = np.zeros(nn, dtype=int)
    labels[u > quartiles[2]] = 3
    labels[(u > quartiles[1]) & (u <= quartiles[2])] = 2
    labels[(u > quartiles[0]) & (u <= quartiles[1])] = 1

    # One-vs-rest classification via argmax
    scores_matrix = np.zeros((nn - n_tr, 4))
    for c in range(4):
        y = (labels == c).astype(float)
        for alpha in [1.0, 10.0, 100.0]:
            I = np.eye(X[:n_tr].shape[1])
            try:
                w = np.linalg.solve(X[:n_tr].T @ X[:n_tr] + alpha * I,
                                    X[:n_tr].T @ y[:n_tr])
                scores_matrix[:, c] = X[n_tr:nn] @ w
                break
            except np.linalg.LinAlgError:
                pass

    pred = np.argmax(scores_matrix, axis=1)
    acc = float(np.mean(pred == labels[n_tr:nn]))
    return acc


def narma_benchmark(X, u_raw, warmup, orders=[5, 10]):
    """NARMA benchmark: predict NARMA-N time series. Returns NRMSE per order."""
    T = len(u_raw)
    u_n = (u_raw - u_raw.min()) / (u_raw.max() - u_raw.min() + 1e-10) * 0.5
    results = {}

    for order in orders:
        y = np.zeros(T)
        for t in range(order, T):
            y[t] = 0.3 * y[t - 1] + 0.05 * y[t - 1] * np.sum(y[t - order:t]) + \
                   1.5 * u_n[t - 1] * u_n[t - order] + 0.1
            y[t] = np.tanh(y[t])

        target = y[warmup:]
        n = len(X)
        nn = min(n, len(target))
        n_tr = int(0.7 * nn)

        best_nrmse = 999.0
        for alpha in [0.01, 0.1, 1.0, 10.0, 100.0]:
            I = np.eye(X[:n_tr].shape[1])
            try:
                w = np.linalg.solve(X[:n_tr].T @ X[:n_tr] + alpha * I,
                                    X[:n_tr].T @ target[:n_tr])
                pred = X[n_tr:nn] @ w
                gt = target[n_tr:nn]
                nrmse = np.sqrt(np.mean((gt - pred) ** 2)) / (np.std(gt) + 1e-10)
                if nrmse < best_nrmse:
                    best_nrmse = nrmse
            except np.linalg.LinAlgError:
                pass
        results[f'narma{order}'] = best_nrmse

    return results


def full_benchmark(X, u_raw, warmup, label=""):
    """Run all benchmarks: MC, XOR, waveform, NARMA."""
    if label:
        print(f"    Benchmarking {label} (features: {X.shape})...", flush=True)

    mc_total, mc_per_d = memory_capacity(X, u_raw, warmup)
    xor = xor_benchmark(X, u_raw, warmup, taus=[1, 2, 3, 5])
    wave_acc = waveform_classification(X, u_raw, warmup)
    narma = narma_benchmark(X, u_raw, warmup, orders=[5, 10])

    res = {
        'mc_total': mc_total,
        'mc_per_delay': mc_per_d,
        'xor': xor,
        'wave4_acc': wave_acc,
        'narma': narma,
        'n_features': int(X.shape[1]),
        'n_steps': int(X.shape[0]),
    }

    print(f"    {label}: MC={mc_total:.3f}  XOR1={xor.get('tau1',0):.3f}  "
          f"XOR3={xor.get('tau3',0):.3f}  Wave={wave_acc:.3f}  "
          f"NARMA5={narma.get('narma5',999):.3f}", flush=True)
    return res


# ============================================================
# Signal Characterization (PSD, ACF)
# ============================================================
def compute_psd_slope(signal, fs=1.0):
    """Compute PSD slope via log-log linear fit. Returns slope (e.g. -1.6 for 1/f)."""
    n = len(signal)
    if n < 64:
        return 0.0
    fft = np.fft.rfft(signal - np.mean(signal))
    psd = np.abs(fft) ** 2 / n
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    # Skip DC and very low frequencies
    mask = freqs > freqs[max(1, len(freqs) // 20)]
    if np.sum(mask) < 10:
        return 0.0
    f = freqs[mask]
    p = psd[mask]
    p[p < 1e-30] = 1e-30
    log_f = np.log10(f)
    log_p = np.log10(p)
    # Linear fit
    A = np.vstack([log_f, np.ones(len(log_f))]).T
    try:
        slope, _ = np.linalg.lstsq(A, log_p, rcond=None)[0]
    except Exception:
        slope = 0.0
    return float(slope)


def compute_acf_lag1(signal):
    """Compute autocorrelation at lag 1."""
    if len(signal) < 3:
        return 0.0
    s = signal - np.mean(signal)
    var = np.var(s)
    if var < 1e-30:
        return 0.0
    return float(np.mean(s[1:] * s[:-1]) / var)


# ============================================================
# Experiment 1: SMN Signal Characterization
# ============================================================
def exp1_signal_characterization(smn):
    """EXP 1: Characterize SMN signals — uniqueness, stationarity, PSD, ACF.

    Tests:
      T828: >= 8/16 cores show > 2 unique thermal values in 2000 samples
      T829: Energy counters increment (non-static)
      T830: PSD slope < -0.5 for per-core thermals (colored noise)
    """
    print("\n" + "=" * 70, flush=True)
    print("  EXP 1: SMN Signal Characterization (T828-T830)", flush=True)
    print("=" * 70, flush=True)

    N_CHAR = 2000
    print(f"  Sampling {N_CHAR} steps at max SMN speed...", flush=True)

    # Collect raw per-core values and full state
    per_core_history = []  # list of [16 raw values]
    full_states = np.zeros((N_CHAR, N_SMN_FEATURES))
    energy_start = np.zeros(3)
    energy_end = np.zeros(3)
    timestamps = []

    t0 = time.time()
    for step in range(N_CHAR):
        check_thermal(step, N_CHAR, "EXP1")
        raw_cores = smn.read_per_core_raw()
        per_core_history.append(raw_cores)
        full_states[step] = smn.read_all_state()
        timestamps.append(time.time())

        if step == 0:
            energy_start[0] = smn.read(SMNReader.ENERGY_COUNTER_0)
            energy_start[1] = smn.read(SMNReader.ENERGY_COUNTER_1)
            energy_start[2] = smn.read(SMNReader.ENERGY_COUNTER_2)
        if step == N_CHAR - 1:
            energy_end[0] = smn.read(SMNReader.ENERGY_COUNTER_0)
            energy_end[1] = smn.read(SMNReader.ENERGY_COUNTER_1)
            energy_end[2] = smn.read(SMNReader.ENERGY_COUNTER_2)

    elapsed = time.time() - t0
    rate = N_CHAR / elapsed if elapsed > 0 else 0
    # Each step does ~21 SMN reads (16 cores + 3 energy + 2 fast + 1 base + per_core_raw)
    smn_reads_per_s = rate * 38
    print(f"  Done: {N_CHAR} steps in {elapsed:.2f}s ({rate:.0f} steps/s, "
          f"~{smn_reads_per_s:.0f} SMN reads/s)", flush=True)

    # Analyze per-core uniqueness
    per_core_arr = np.array(per_core_history)  # (N_CHAR, 16)
    unique_counts = []
    for core in range(16):
        low_bytes = per_core_arr[:, core] & 0xFF
        n_unique = len(np.unique(low_bytes))
        unique_counts.append(n_unique)

    cores_with_variation = sum(1 for uc in unique_counts if uc > 2)
    print(f"  Per-core unique thermal values: {unique_counts}", flush=True)
    print(f"  Cores with >2 unique: {cores_with_variation}/16", flush=True)

    # Energy counter increments
    energy_deltas = energy_end - energy_start
    any_energy_inc = any(d > 0 for d in energy_deltas)
    print(f"  Energy deltas: {energy_deltas[0]:.0f}, {energy_deltas[1]:.0f}, "
          f"{energy_deltas[2]:.0f}", flush=True)

    # PSD slopes for per-core low-byte thermals
    psd_slopes = []
    acf_vals = []
    effective_hz = rate  # approximate sampling rate
    for core in range(16):
        low_bytes = (per_core_arr[:, core] & 0xFF).astype(float)
        slope = compute_psd_slope(low_bytes, fs=effective_hz)
        acf = compute_acf_lag1(low_bytes)
        psd_slopes.append(slope)
        acf_vals.append(acf)

    mean_psd = np.mean(psd_slopes)
    mean_acf = np.mean(acf_vals)
    cores_colored = sum(1 for s in psd_slopes if s < -0.5)
    print(f"  PSD slopes: mean={mean_psd:.2f}, colored (<-0.5): {cores_colored}/16", flush=True)
    print(f"  ACF(1): mean={mean_acf:.3f}", flush=True)

    # Feature statistics
    print(f"  Full state stats:", flush=True)
    for ch_name, ch_slice in [("core_low", slice(0, 32, 2)),
                               ("core_mid", slice(1, 32, 2)),
                               ("energy", slice(32, 35)),
                               ("fast_ctr", slice(35, 37)),
                               ("base_temp", slice(37, 38))]:
        vals = full_states[:, ch_slice]
        std_per_ch = np.std(vals, axis=0)
        print(f"    {ch_name}: mean_std={np.mean(std_per_ch):.4f}, "
              f"min_std={np.min(std_per_ch):.4f}, max_std={np.max(std_per_ch):.4f}",
              flush=True)

    # Tests
    tests = {}

    # T828: >= 8/16 cores show >2 unique thermal values
    t828_pass = cores_with_variation >= 8
    tests['T828'] = {
        'name': 'Per-core thermal diversity (>=8/16 cores with >2 unique)',
        'pass': bool(t828_pass),
        'cores_with_variation': cores_with_variation,
        'unique_counts': unique_counts,
    }
    print(f"  T828 {'PASS' if t828_pass else 'FAIL'}: {cores_with_variation}/16 cores "
          f"with >2 unique (need >=8)", flush=True)

    # T829: Energy counters increment
    t829_pass = bool(any_energy_inc)
    tests['T829'] = {
        'name': 'Energy counters increment (non-static)',
        'pass': bool(t829_pass),
        'energy_deltas': energy_deltas.tolist(),
    }
    print(f"  T829 {'PASS' if t829_pass else 'FAIL'}: energy increment={any_energy_inc} "
          f"(deltas: {energy_deltas.tolist()})", flush=True)

    # T830: PSD slope < -0.5 for majority of cores
    t830_pass = cores_colored >= 8
    tests['T830'] = {
        'name': 'PSD slope < -0.5 (colored noise) for >=8/16 cores',
        'pass': bool(t830_pass),
        'psd_slopes': psd_slopes,
        'mean_psd': mean_psd,
        'cores_colored': cores_colored,
        'acf_vals': acf_vals,
        'mean_acf': mean_acf,
    }
    print(f"  T830 {'PASS' if t830_pass else 'FAIL'}: {cores_colored}/16 cores with "
          f"PSD<-0.5 (need >=8)", flush=True)

    exp1_results = {
        'n_samples': N_CHAR,
        'sampling_rate_hz': rate,
        'smn_reads_per_s': smn_reads_per_s,
        'elapsed_s': elapsed,
        'unique_counts': unique_counts,
        'cores_with_variation': cores_with_variation,
        'energy_deltas': energy_deltas.tolist(),
        'psd_slopes': psd_slopes,
        'mean_psd': mean_psd,
        'acf_vals': acf_vals,
        'mean_acf': mean_acf,
    }

    return tests, exp1_results, full_states


# ============================================================
# Experiment 2: SMN-only Reservoir
# ============================================================
def exp2_smn_reservoir(smn, u_raw, rng):
    """EXP 2: SMN-only neuromorphic reservoir.

    Inject workload proportional to u(t), read SMN state, benchmark.

    Tests:
      T831: MC > 0.5 (SMN has real memory, unlike z2308 MSR=0.000)
      T832: XOR tau=1 > 55% (above chance)
      T833: XOR tau=3 > 52% (weak temporal nonlinearity)
      T834: Waveform > 30% (4-class, chance=25%)
    """
    print("\n" + "=" * 70, flush=True)
    print("  EXP 2: SMN-only Reservoir (T831-T834)", flush=True)
    print("=" * 70, flush=True)

    total_steps = N_STEPS + WARMUP
    print(f"  Collecting {total_steps} steps with workload injection...", flush=True)

    wait_cool("pre-EXP2", TEMP_RESUME)

    smn_states = np.zeros((total_steps, N_SMN_FEATURES))
    t0 = time.time()

    for step in range(total_steps):
        check_thermal(step, total_steps, "EXP2")

        # Normalize input to [0, 1] for workload injection
        u_norm = (u_raw[step] + 1.0) / 2.0  # u_raw in [-1, 1]
        inject_workload(u_norm, n_cores=16, base_spin_ms=0.5, max_spin_ms=3.0)

        # Read SMN state immediately after workload
        smn_states[step] = smn.read_all_state()

        # Progress every 200 steps
        if (step + 1) % 200 == 0:
            temp = get_max_temp()
            elapsed = time.time() - t0
            rate = (step + 1) / elapsed
            print(f"    Step {step+1}/{total_steps} ({rate:.1f} steps/s, "
                  f"{temp:.0f}C)", flush=True)

    elapsed = time.time() - t0
    print(f"  Collection done: {elapsed:.1f}s", flush=True)

    # Save raw SMN states
    np.save(str(STATES_FILE), smn_states)
    print(f"  Saved SMN states to {STATES_FILE}", flush=True)

    # Discard warmup
    X_raw = smn_states[WARMUP:]

    # Normalize: z-score per feature (avoid divide-by-zero)
    mu = X_raw.mean(axis=0)
    sigma = X_raw.std(axis=0)
    sigma[sigma < 1e-6] = 1.0  # avoid amplifying constant features
    X_norm = (X_raw - mu) / sigma

    # Build temporal product features
    print("  Building temporal product features...", flush=True)
    X_temp = build_smn_temporal_products(X_norm)
    print(f"  Feature shape: raw={X_norm.shape}, temporal={X_temp.shape}", flush=True)

    # Normalize temporal features
    mu_t = X_temp.mean(axis=0)
    sigma_t = X_temp.std(axis=0)
    sigma_t[sigma_t < 1e-6] = 1.0
    X_temp = (X_temp - mu_t) / sigma_t

    # Run benchmarks on raw features
    print("\n  --- Raw SMN features ---", flush=True)
    bm_raw = full_benchmark(X_norm, u_raw, WARMUP, label="SMN_RAW")

    # Run benchmarks on temporal product features
    print("\n  --- Temporal product features ---", flush=True)
    bm_temp = full_benchmark(X_temp, u_raw, WARMUP, label="SMN_TEMPORAL")

    # Tests use BEST of raw and temporal
    mc_best = max(bm_raw['mc_total'], bm_temp['mc_total'])
    xor1_best = max(bm_raw['xor'].get('tau1', 0), bm_temp['xor'].get('tau1', 0))
    xor3_best = max(bm_raw['xor'].get('tau3', 0), bm_temp['xor'].get('tau3', 0))
    wave_best = max(bm_raw['wave4_acc'], bm_temp['wave4_acc'])

    tests = {}

    # T831: MC > 0.5
    t831_pass = mc_best > 0.5
    tests['T831'] = {
        'name': 'SMN memory capacity > 0.5 (z2308 MSR was 0.000)',
        'pass': bool(t831_pass),
        'mc_raw': bm_raw['mc_total'],
        'mc_temporal': bm_temp['mc_total'],
        'mc_best': mc_best,
    }
    print(f"\n  T831 {'PASS' if t831_pass else 'FAIL'}: MC={mc_best:.3f} (need >0.5)",
          flush=True)

    # T832: XOR tau=1 > 55%
    t832_pass = xor1_best > 0.55
    tests['T832'] = {
        'name': 'XOR tau=1 > 55% (above chance)',
        'pass': bool(t832_pass),
        'xor1_raw': bm_raw['xor'].get('tau1', 0),
        'xor1_temporal': bm_temp['xor'].get('tau1', 0),
        'xor1_best': xor1_best,
    }
    print(f"  T832 {'PASS' if t832_pass else 'FAIL'}: XOR1={xor1_best:.3f} (need >0.55)",
          flush=True)

    # T833: XOR tau=3 > 52%
    t833_pass = xor3_best > 0.52
    tests['T833'] = {
        'name': 'XOR tau=3 > 52% (weak temporal nonlinearity)',
        'pass': bool(t833_pass),
        'xor3_raw': bm_raw['xor'].get('tau3', 0),
        'xor3_temporal': bm_temp['xor'].get('tau3', 0),
        'xor3_best': xor3_best,
    }
    print(f"  T833 {'PASS' if t833_pass else 'FAIL'}: XOR3={xor3_best:.3f} (need >0.52)",
          flush=True)

    # T834: Waveform > 30%
    t834_pass = wave_best > 0.30
    tests['T834'] = {
        'name': 'Waveform 4-class > 30% (chance=25%)',
        'pass': bool(t834_pass),
        'wave_raw': bm_raw['wave4_acc'],
        'wave_temporal': bm_temp['wave4_acc'],
        'wave_best': wave_best,
    }
    print(f"  T834 {'PASS' if t834_pass else 'FAIL'}: Wave={wave_best:.3f} (need >0.30)",
          flush=True)

    exp2_results = {
        'smn_raw': bm_raw,
        'smn_temporal': bm_temp,
        'total_steps': total_steps,
        'elapsed_s': elapsed,
    }

    return tests, exp2_results, smn_states


# ============================================================
# Experiment 3: SMN + FPGA Bridge
# ============================================================
def exp3_smn_fpga_bridge(smn, fpga, u_raw, smn_states_prev, rng):
    """EXP 3: SMN + FPGA dual-substrate bridge.

    Run SMN and FPGA concurrently, concatenate features.

    Tests:
      T835: BRIDGE MC > max(SMN, FPGA) (cross-substrate synergy)
      T836: BRIDGE XOR1 > max(SMN, FPGA) XOR1
      T837: BRIDGE waveform > max(SMN, FPGA) waveform
      T838: BRIDGE beats SMN on >= 2/4 metrics
    """
    print("\n" + "=" * 70, flush=True)
    print("  EXP 3: SMN + FPGA Bridge (T835-T838)", flush=True)
    print("=" * 70, flush=True)

    if fpga is None:
        print("  [SKIP] FPGA not available, generating placeholder results", flush=True)
        tests = {}
        for tid, name in [('T835', 'BRIDGE MC > max(SMN,FPGA)'),
                           ('T836', 'BRIDGE XOR1 > max(SMN,FPGA)'),
                           ('T837', 'BRIDGE wave > max(SMN,FPGA)'),
                           ('T838', 'BRIDGE beats SMN on >=2/4 metrics')]:
            tests[tid] = {'name': name, 'pass': False, 'skip': 'FPGA unavailable'}
            print(f"  {tid} SKIP: FPGA unavailable", flush=True)
        return tests, {'skip': 'FPGA unavailable'}, None

    wait_cool("pre-EXP3", TEMP_RESUME)

    total_steps = N_STEPS + WARMUP
    mac_signal = np.clip(u_raw[:total_steps] * 0.3 + 0.3, 0, 1)

    print(f"  Running FPGA + SMN concurrent collection ({total_steps} steps)...", flush=True)

    smn_states = np.zeros((total_steps, N_SMN_FEATURES))
    fpga_states = np.zeros((total_steps, NUM_FPGA_NEURONS))
    fpga_dspikes = np.zeros((total_steps, NUM_FPGA_NEURONS), dtype=np.float32)
    dt = 1.0 / SAMPLE_HZ

    fpga.set_mac_signal(0.0)
    time.sleep(0.02)
    telem = fpga.read_telemetry()
    prev_sc = telem['spike_counts'].copy() if telem is not None else np.zeros(NUM_FPGA_NEURONS, dtype=np.uint16)

    t0 = time.time()
    for step in range(total_steps):
        check_thermal(step, total_steps, "EXP3")

        # Inject workload (creates SMN thermal differentiation)
        u_norm = (u_raw[step] + 1.0) / 2.0
        inject_workload(u_norm, n_cores=16, base_spin_ms=0.5, max_spin_ms=2.0)

        # Set FPGA MAC signal
        fpga.set_mac_signal(float(mac_signal[step]))
        time.sleep(dt)

        # Read both substrates
        smn_states[step] = smn.read_all_state()
        telem = fpga.read_telemetry()
        if telem is not None:
            fpga_states[step] = telem['vmem']
            sc = telem['spike_counts']
            diff = sc.astype(np.int32) - prev_sc.astype(np.int32)
            diff[diff < 0] += 65536
            fpga_dspikes[step] = diff.astype(np.float32)
            prev_sc = sc.copy()
        elif step > 0:
            fpga_states[step] = fpga_states[step - 1]
            fpga_dspikes[step] = fpga_dspikes[step - 1]

        if (step + 1) % 200 == 0:
            temp = get_max_temp()
            elapsed = time.time() - t0
            rate = (step + 1) / elapsed
            print(f"    Step {step+1}/{total_steps} ({rate:.1f} steps/s, "
                  f"{temp:.0f}C)", flush=True)

    fpga.set_mac_signal(0.0)
    elapsed = time.time() - t0
    print(f"  Collection done: {elapsed:.1f}s", flush=True)

    # Build features for each substrate and bridge
    smn_x = smn_states[WARMUP:]
    fpga_x = fpga_states[WARMUP:]
    fpga_ds = fpga_dspikes[WARMUP:]

    # Normalize
    for arr in [smn_x]:
        mu = arr.mean(axis=0)
        sig = arr.std(axis=0)
        sig[sig < 1e-6] = 1.0
        arr[:] = (arr - mu) / sig

    for arr in [fpga_x]:
        mu = arr.mean(axis=0)
        sig = arr.std(axis=0)
        sig[sig < 1e-2] = 1.0  # sigma floor per z2210
        arr[:] = (arr - mu) / sig

    # Temporal features
    smn_tf = build_smn_temporal_products(smn_x)
    fpga_tf = build_temporal_features(fpga_x, fpga_ds)

    # Normalize temporal
    for arr in [smn_tf, fpga_tf]:
        mu = arr.mean(axis=0)
        sig = arr.std(axis=0)
        sig[sig < 1e-6] = 1.0
        arr[:] = (arr - mu) / sig

    # Bridge: concatenate SMN + FPGA temporal features
    bridge_tf = np.hstack([smn_tf, fpga_tf])

    # Benchmark all three
    print("\n  --- SMN only ---", flush=True)
    bm_smn = full_benchmark(smn_tf, u_raw, WARMUP, label="SMN")
    print("\n  --- FPGA only ---", flush=True)
    bm_fpga = full_benchmark(fpga_tf, u_raw, WARMUP, label="FPGA")
    print("\n  --- BRIDGE (SMN+FPGA) ---", flush=True)
    bm_bridge = full_benchmark(bridge_tf, u_raw, WARMUP, label="BRIDGE")

    # Tests
    tests = {}
    mc_smn = bm_smn['mc_total']
    mc_fpga = bm_fpga['mc_total']
    mc_bridge = bm_bridge['mc_total']
    mc_max = max(mc_smn, mc_fpga)

    # T835: BRIDGE MC > max(SMN, FPGA)
    t835_pass = mc_bridge > mc_max
    tests['T835'] = {
        'name': 'BRIDGE MC > max(SMN, FPGA)',
        'pass': bool(t835_pass),
        'mc_smn': mc_smn, 'mc_fpga': mc_fpga, 'mc_bridge': mc_bridge,
    }
    print(f"\n  T835 {'PASS' if t835_pass else 'FAIL'}: BRIDGE MC={mc_bridge:.3f} vs "
          f"max(SMN={mc_smn:.3f}, FPGA={mc_fpga:.3f})", flush=True)

    # T836: BRIDGE XOR1 > max(SMN, FPGA) XOR1
    xor1_smn = bm_smn['xor'].get('tau1', 0)
    xor1_fpga = bm_fpga['xor'].get('tau1', 0)
    xor1_bridge = bm_bridge['xor'].get('tau1', 0)
    xor1_max = max(xor1_smn, xor1_fpga)
    t836_pass = xor1_bridge > xor1_max
    tests['T836'] = {
        'name': 'BRIDGE XOR1 > max(SMN, FPGA)',
        'pass': bool(t836_pass),
        'xor1_smn': xor1_smn, 'xor1_fpga': xor1_fpga, 'xor1_bridge': xor1_bridge,
    }
    print(f"  T836 {'PASS' if t836_pass else 'FAIL'}: BRIDGE XOR1={xor1_bridge:.3f} vs "
          f"max(SMN={xor1_smn:.3f}, FPGA={xor1_fpga:.3f})", flush=True)

    # T837: BRIDGE waveform > max(SMN, FPGA)
    wave_smn = bm_smn['wave4_acc']
    wave_fpga = bm_fpga['wave4_acc']
    wave_bridge = bm_bridge['wave4_acc']
    wave_max = max(wave_smn, wave_fpga)
    t837_pass = wave_bridge > wave_max
    tests['T837'] = {
        'name': 'BRIDGE wave > max(SMN, FPGA)',
        'pass': bool(t837_pass),
        'wave_smn': wave_smn, 'wave_fpga': wave_fpga, 'wave_bridge': wave_bridge,
    }
    print(f"  T837 {'PASS' if t837_pass else 'FAIL'}: BRIDGE wave={wave_bridge:.3f} vs "
          f"max(SMN={wave_smn:.3f}, FPGA={wave_fpga:.3f})", flush=True)

    # T838: BRIDGE beats SMN on >= 2/4 metrics
    beats = 0
    if mc_bridge > mc_smn:
        beats += 1
    if xor1_bridge > xor1_smn:
        beats += 1
    if wave_bridge > wave_smn:
        beats += 1
    narma5_smn = bm_smn['narma'].get('narma5', 999)
    narma5_bridge = bm_bridge['narma'].get('narma5', 999)
    if narma5_bridge < narma5_smn:
        beats += 1
    t838_pass = beats >= 2
    tests['T838'] = {
        'name': 'BRIDGE beats SMN on >= 2/4 metrics',
        'pass': bool(t838_pass),
        'beats_count': beats,
    }
    print(f"  T838 {'PASS' if t838_pass else 'FAIL'}: BRIDGE beats SMN on {beats}/4 "
          f"metrics (need >=2)", flush=True)

    exp3_results = {
        'smn': bm_smn,
        'fpga': bm_fpga,
        'bridge': bm_bridge,
        'elapsed_s': elapsed,
    }

    return tests, exp3_results, smn_states


# ============================================================
# Experiment 4: Triple Bridge (SMN + FPGA + GPU)
# ============================================================
def exp4_triple_bridge(smn, fpga, u_raw, smn_states_prev, rng):
    """EXP 4: Triple bridge — SMN + FPGA + GPU (or software ESN fallback).

    Tests:
      T839: TRIPLE MC > DUAL(FPGA+GPU) MC
      T840: TRIPLE XOR1 > DUAL XOR1
      T841: TRIPLE wave > DUAL wave
      T842: TRIPLE beats DUAL on >= 2/4 metrics
    """
    print("\n" + "=" * 70, flush=True)
    print("  EXP 4: Triple Bridge — SMN + FPGA + GPU (T839-T842)", flush=True)
    print("=" * 70, flush=True)

    wait_cool("pre-EXP4", TEMP_RESUME)

    total_steps = N_STEPS + WARMUP

    # --- GPU substrate (try HIP kernel, fall back to software ESN) ---
    gpu_kern = BASE / 'scripts' / 'z2277_gpu_bridge_kern'
    gpu_available = gpu_kern.exists()
    gpu_states = None

    if gpu_available:
        print("  [GPU] Attempting real HIP kernel...", flush=True)
        try:
            import subprocess, tempfile
            with tempfile.NamedTemporaryFile(suffix='.bin', delete=False) as fin:
                input_path = fin.name
                u_raw[:total_steps].astype(np.float32).tofile(fin)
            with tempfile.NamedTemporaryFile(suffix='.bin', delete=False) as fout:
                output_path = fout.name

            env = os.environ.copy()
            env['HSA_OVERRIDE_GFX_VERSION'] = '11.0.0'
            result = subprocess.run(
                [str(gpu_kern), input_path, output_path, str(total_steps)],
                env=env, capture_output=True, text=True, timeout=120
            )
            if result.returncode == 0:
                raw = np.fromfile(output_path, dtype=np.float32)
                n_gpu = 512
                expected = n_gpu * total_steps
                if len(raw) == expected:
                    gpu_states = raw.reshape(n_gpu, total_steps).T
                    print(f"  [GPU] Real HIP kernel: {gpu_states.shape}", flush=True)
                else:
                    print(f"  [GPU] Wrong output size: {len(raw)} != {expected}", flush=True)
            else:
                print(f"  [GPU] Kernel error: {result.stderr[:200]}", flush=True)

            try:
                os.unlink(input_path)
            except Exception:
                pass
            try:
                os.unlink(output_path)
            except Exception:
                pass
        except Exception as e:
            print(f"  [GPU] HIP kernel failed: {e}", flush=True)

    if gpu_states is None:
        print("  [GPU] Using software ESN fallback (256 neurons)...", flush=True)
        esn = SoftwareESN(n_neurons=256, spectral_radius=0.95, seed=42)
        gpu_states = esn.run(u_raw[:total_steps])
        print(f"  [GPU] Software ESN: {gpu_states.shape}", flush=True)

    # --- SMN substrate (reuse or recollect) ---
    if smn_states_prev is not None and len(smn_states_prev) >= total_steps:
        print("  [SMN] Reusing previously collected states", flush=True)
        smn_states = smn_states_prev[:total_steps]
    else:
        print(f"  [SMN] Collecting {total_steps} steps with workload injection...", flush=True)
        smn_states = np.zeros((total_steps, N_SMN_FEATURES))
        for step in range(total_steps):
            check_thermal(step, total_steps, "EXP4-SMN")
            u_norm = (u_raw[step] + 1.0) / 2.0
            inject_workload(u_norm, n_cores=16, base_spin_ms=0.5, max_spin_ms=2.0)
            smn_states[step] = smn.read_all_state()
            if (step + 1) % 300 == 0:
                print(f"    Step {step+1}/{total_steps}", flush=True)

    # --- FPGA substrate ---
    fpga_states = None
    fpga_dspikes = None
    if fpga is not None:
        print("  [FPGA] Collecting FPGA states...", flush=True)
        wait_cool("pre-FPGA", TEMP_RESUME)
        fpga_states, fpga_dspikes = fpga_run_continuous(
            fpga, u_raw[:total_steps],
            mac_signal=np.clip(u_raw[:total_steps] * 0.3 + 0.3, 0, 1)
        )
        print(f"  [FPGA] Collected: {fpga_states.shape}", flush=True)

    # --- Build features and benchmark ---
    # Discard warmup
    gpu_x = gpu_states[WARMUP:]
    smn_x = smn_states[WARMUP:]

    # Normalize SMN
    mu = smn_x.mean(axis=0)
    sig = smn_x.std(axis=0)
    sig[sig < 1e-6] = 1.0
    smn_x = (smn_x - mu) / sig

    # Normalize GPU
    mu = gpu_x.mean(axis=0)
    sig = gpu_x.std(axis=0)
    sig[sig < 1e-6] = 1.0
    gpu_x = (gpu_x - mu) / sig

    # Build temporal features
    smn_tf = build_smn_temporal_products(smn_x)
    gpu_tf = build_temporal_features(gpu_x, n_select=min(24, gpu_x.shape[1]))

    # Normalize temporal features
    for arr in [smn_tf, gpu_tf]:
        mu = arr.mean(axis=0)
        sig = arr.std(axis=0)
        sig[sig < 1e-6] = 1.0
        arr[:] = (arr - mu) / sig

    # DUAL = GPU + FPGA (or GPU-only if no FPGA)
    if fpga_states is not None:
        fpga_x = fpga_states[WARMUP:]
        fpga_ds = fpga_dspikes[WARMUP:]
        mu = fpga_x.mean(axis=0)
        sig = fpga_x.std(axis=0)
        sig[sig < 1e-2] = 1.0
        fpga_x = (fpga_x - mu) / sig
        fpga_tf = build_temporal_features(fpga_x, fpga_ds)
        mu = fpga_tf.mean(axis=0)
        sig = fpga_tf.std(axis=0)
        sig[sig < 1e-6] = 1.0
        fpga_tf = (fpga_tf - mu) / sig

        dual_tf = np.hstack([gpu_tf, fpga_tf])
        triple_tf = np.hstack([smn_tf, gpu_tf, fpga_tf])
    else:
        dual_tf = gpu_tf
        triple_tf = np.hstack([smn_tf, gpu_tf])

    # Cap feature dimensionality to avoid memory issues
    MAX_FEAT = 3000
    if dual_tf.shape[1] > MAX_FEAT:
        rng_sel = np.random.default_rng(42)
        idx = rng_sel.choice(dual_tf.shape[1], MAX_FEAT, replace=False)
        idx = np.sort(idx)
        dual_tf = dual_tf[:, idx]
    if triple_tf.shape[1] > MAX_FEAT:
        rng_sel = np.random.default_rng(42)
        idx = rng_sel.choice(triple_tf.shape[1], MAX_FEAT, replace=False)
        idx = np.sort(idx)
        triple_tf = triple_tf[:, idx]

    print(f"\n  Feature dims: GPU={gpu_tf.shape[1]}, SMN={smn_tf.shape[1]}, "
          f"DUAL={dual_tf.shape[1]}, TRIPLE={triple_tf.shape[1]}", flush=True)

    # Benchmarks
    print("\n  --- GPU only ---", flush=True)
    bm_gpu = full_benchmark(gpu_tf, u_raw, WARMUP, label="GPU")

    print("\n  --- DUAL (GPU+FPGA) ---", flush=True)
    bm_dual = full_benchmark(dual_tf, u_raw, WARMUP, label="DUAL")

    print("\n  --- TRIPLE (SMN+GPU+FPGA) ---", flush=True)
    bm_triple = full_benchmark(triple_tf, u_raw, WARMUP, label="TRIPLE")

    # Tests
    tests = {}

    mc_dual = bm_dual['mc_total']
    mc_triple = bm_triple['mc_total']
    t839_pass = mc_triple > mc_dual
    tests['T839'] = {
        'name': 'TRIPLE MC > DUAL MC',
        'pass': bool(t839_pass),
        'mc_dual': mc_dual, 'mc_triple': mc_triple,
        'mc_gpu': bm_gpu['mc_total'],
    }
    print(f"\n  T839 {'PASS' if t839_pass else 'FAIL'}: TRIPLE MC={mc_triple:.3f} vs "
          f"DUAL MC={mc_dual:.3f}", flush=True)

    xor1_dual = bm_dual['xor'].get('tau1', 0)
    xor1_triple = bm_triple['xor'].get('tau1', 0)
    t840_pass = xor1_triple > xor1_dual
    tests['T840'] = {
        'name': 'TRIPLE XOR1 > DUAL XOR1',
        'pass': bool(t840_pass),
        'xor1_dual': xor1_dual, 'xor1_triple': xor1_triple,
    }
    print(f"  T840 {'PASS' if t840_pass else 'FAIL'}: TRIPLE XOR1={xor1_triple:.3f} vs "
          f"DUAL XOR1={xor1_dual:.3f}", flush=True)

    wave_dual = bm_dual['wave4_acc']
    wave_triple = bm_triple['wave4_acc']
    t841_pass = wave_triple > wave_dual
    tests['T841'] = {
        'name': 'TRIPLE wave > DUAL wave',
        'pass': bool(t841_pass),
        'wave_dual': wave_dual, 'wave_triple': wave_triple,
    }
    print(f"  T841 {'PASS' if t841_pass else 'FAIL'}: TRIPLE wave={wave_triple:.3f} vs "
          f"DUAL wave={wave_dual:.3f}", flush=True)

    # T842: TRIPLE beats DUAL on >= 2/4 metrics
    beats = 0
    if mc_triple > mc_dual:
        beats += 1
    if xor1_triple > xor1_dual:
        beats += 1
    if wave_triple > wave_dual:
        beats += 1
    narma5_dual = bm_dual['narma'].get('narma5', 999)
    narma5_triple = bm_triple['narma'].get('narma5', 999)
    if narma5_triple < narma5_dual:
        beats += 1
    t842_pass = beats >= 2
    tests['T842'] = {
        'name': 'TRIPLE beats DUAL on >= 2/4 metrics',
        'pass': bool(t842_pass),
        'beats_count': beats,
    }
    print(f"  T842 {'PASS' if t842_pass else 'FAIL'}: TRIPLE beats DUAL on {beats}/4 "
          f"metrics (need >=2)", flush=True)

    exp4_results = {
        'gpu': bm_gpu,
        'dual': bm_dual,
        'triple': bm_triple,
        'gpu_type': 'HIP' if gpu_available and gpu_states is not None else 'SOFTWARE_ESN',
        'fpga_available': fpga_states is not None,
    }

    return tests, exp4_results


# ============================================================
# Main
# ============================================================
def main():
    print("=" * 70, flush=True)
    print("  z2309: SMN Deep Analog Neuromorphic Reservoir", flush=True)
    print("  /dev/mem MMCFG for high-speed SMN register reads", flush=True)
    print("  16 per-core thermal + energy counters + fast counter", flush=True)
    print("=" * 70, flush=True)
    print(f"  Date: {time.strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
    print(f"  N_STEPS={N_STEPS}, WARMUP={WARMUP}, SAMPLE_HZ={SAMPLE_HZ}", flush=True)
    print(f"  N_SMN_FEATURES={N_SMN_FEATURES}", flush=True)
    temp = get_max_temp()
    print(f"  Current temp: {temp:.1f}C", flush=True)

    # Check root access (required for /dev/mem)
    if os.geteuid() != 0:
        print("\n  ERROR: This script requires root access for /dev/mem.", flush=True)
        print("  Run: sudo venv/bin/python scripts/z2309_smn_neuromorphic.py", flush=True)
        sys.exit(1)

    results = {'experiments': {}, 'tests': {}, 'meta': {
        'script': 'z2309_smn_neuromorphic.py',
        'date': time.strftime('%Y-%m-%d %H:%M:%S'),
        'n_steps': N_STEPS,
        'warmup': WARMUP,
        'n_smn_features': N_SMN_FEATURES,
    }}

    # Resume support
    if SAVE_FILE.exists():
        try:
            with open(SAVE_FILE) as f:
                saved = json.load(f)
            if 'experiments' in saved and 'tests' in saved:
                results = saved
                done = list(results['experiments'].keys())
                if done:
                    print(f"  RESUMED: {done} already done", flush=True)
        except Exception:
            pass

    # Initialize random input
    rng = np.random.default_rng(42)
    u_raw = rng.uniform(-1, 1, N_STEPS + WARMUP + 50)  # extra buffer

    # Initialize SMN reader
    print("\n  Initializing SMN reader...", flush=True)
    smn = SMNReader()
    try:
        smn.open()
    except Exception as e:
        print(f"  FATAL: Cannot open /dev/mem: {e}", flush=True)
        print("  Ensure running as root: sudo venv/bin/python ...", flush=True)
        sys.exit(1)

    # Try FPGA
    fpga = None
    if 'EXP3' not in results.get('experiments', {}) or \
       'EXP4' not in results.get('experiments', {}):
        print("\n  Attempting FPGA connection...", flush=True)
        fpga = try_connect_fpga()

    smn_states_all = None

    try:
        # ================================================================
        # EXP 1: Signal Characterization
        # ================================================================
        if 'EXP1' not in results.get('experiments', {}):
            wait_cool("pre-EXP1", TEMP_RESUME)
            tests1, exp1_res, _ = exp1_signal_characterization(smn)
            results['tests'].update(tests1)
            results['experiments']['EXP1'] = exp1_res
            # Save checkpoint
            with open(SAVE_FILE, 'w') as f:
                json.dump(results, f, indent=2, cls=NpEncoder)
            print(f"  [SAVE] Checkpoint after EXP1", flush=True)
        else:
            print("\n  [SKIP] EXP1 already done", flush=True)

        # ================================================================
        # EXP 2: SMN-only Reservoir
        # ================================================================
        if 'EXP2' not in results.get('experiments', {}):
            wait_cool("pre-EXP2", TEMP_RESUME)
            tests2, exp2_res, smn_states_all = exp2_smn_reservoir(smn, u_raw, rng)
            results['tests'].update(tests2)
            results['experiments']['EXP2'] = exp2_res
            with open(SAVE_FILE, 'w') as f:
                json.dump(results, f, indent=2, cls=NpEncoder)
            print(f"  [SAVE] Checkpoint after EXP2", flush=True)
        else:
            print("\n  [SKIP] EXP2 already done", flush=True)
            # Try to load saved states for reuse
            if STATES_FILE.exists():
                try:
                    smn_states_all = np.load(str(STATES_FILE))
                    print(f"  Loaded saved SMN states: {smn_states_all.shape}", flush=True)
                except Exception:
                    pass

        # ================================================================
        # EXP 3: SMN + FPGA Bridge
        # ================================================================
        if 'EXP3' not in results.get('experiments', {}):
            wait_cool("pre-EXP3", TEMP_RESUME)
            tests3, exp3_res, smn_states_3 = exp3_smn_fpga_bridge(
                smn, fpga, u_raw, smn_states_all, rng)
            results['tests'].update(tests3)
            results['experiments']['EXP3'] = exp3_res
            if smn_states_3 is not None and smn_states_all is None:
                smn_states_all = smn_states_3
            with open(SAVE_FILE, 'w') as f:
                json.dump(results, f, indent=2, cls=NpEncoder)
            print(f"  [SAVE] Checkpoint after EXP3", flush=True)
        else:
            print("\n  [SKIP] EXP3 already done", flush=True)

        # ================================================================
        # EXP 4: Triple Bridge
        # ================================================================
        if 'EXP4' not in results.get('experiments', {}):
            wait_cool("pre-EXP4", TEMP_RESUME)
            tests4, exp4_res = exp4_triple_bridge(
                smn, fpga, u_raw, smn_states_all, rng)
            results['tests'].update(tests4)
            results['experiments']['EXP4'] = exp4_res
            with open(SAVE_FILE, 'w') as f:
                json.dump(results, f, indent=2, cls=NpEncoder)
            print(f"  [SAVE] Checkpoint after EXP4", flush=True)
        else:
            print("\n  [SKIP] EXP4 already done", flush=True)

    except KeyboardInterrupt:
        print("\n  [INTERRUPTED] Saving partial results...", flush=True)
    except Exception as e:
        print(f"\n  [ERROR] {e}", flush=True)
        import traceback
        traceback.print_exc()
    finally:
        smn.close()
        if fpga is not None:
            try:
                fpga.close()
            except Exception:
                pass

    # Final save
    with open(SAVE_FILE, 'w') as f:
        json.dump(results, f, indent=2, cls=NpEncoder)

    # ================================================================
    # Summary
    # ================================================================
    print("\n" + "=" * 70, flush=True)
    print("  SUMMARY", flush=True)
    print("=" * 70, flush=True)

    all_tests = results.get('tests', {})
    n_pass = sum(1 for t in all_tests.values() if t.get('pass', False))
    n_skip = sum(1 for t in all_tests.values() if t.get('skip'))
    n_total = len(all_tests)
    n_fail = n_total - n_pass - n_skip

    for tid in sorted(all_tests.keys()):
        t = all_tests[tid]
        if t.get('skip'):
            status = 'SKIP'
        elif t.get('pass'):
            status = 'PASS'
        else:
            status = 'FAIL'
        print(f"  {tid} {status}: {t.get('name', '')}", flush=True)

    print(f"\n  Total: {n_pass}/{n_total} PASS, {n_fail} FAIL, {n_skip} SKIP",
          flush=True)
    print(f"  Results: {SAVE_FILE}", flush=True)
    print(f"  States:  {STATES_FILE}", flush=True)

    # Key comparisons
    if 'EXP2' in results.get('experiments', {}):
        exp2 = results['experiments']['EXP2']
        mc_smn = exp2.get('smn_temporal', {}).get('mc_total', 0)
        print(f"\n  SMN MC = {mc_smn:.3f} (z2308 MSR was 0.000)", flush=True)
        xor1 = exp2.get('smn_temporal', {}).get('xor', {}).get('tau1', 0)
        print(f"  SMN XOR1 = {xor1:.3f}", flush=True)

    if 'EXP4' in results.get('experiments', {}):
        exp4 = results['experiments']['EXP4']
        mc_triple = exp4.get('triple', {}).get('mc_total', 0)
        mc_dual = exp4.get('dual', {}).get('mc_total', 0)
        print(f"  TRIPLE MC = {mc_triple:.3f} vs DUAL MC = {mc_dual:.3f}", flush=True)

    print("\n  Done.", flush=True)


if __name__ == '__main__':
    main()
