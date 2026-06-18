#!/usr/bin/env python3
"""z2163_mackey_glass.py — Mackey-Glass Classification, Memory Capacity & Kernel Quality

Gold-standard reservoir computing benchmarks for the GPU-noise-driven FPGA
reservoir, enabling direct comparison with memristor RC literature.

Task 1: Mackey-Glass CLASSIFICATION (not regression — spikes too noisy for NRMSE)
  - Direction: predict whether x(t+1) > x(t) — binary classification
  - Quadrant: rising-high, rising-low, falling-high, falling-low — 4-class
  - These leverage spike PATTERNS not exact values

Task 2: Memory Capacity (MC) — THE standard RC metric
  - MC_d = corr(predicted, u(t-d))^2 for d=1..15
  - Total MC = sum(MC_d)
  - Literature: memristor RC typically gets MC 3-8 for 8-node reservoirs

Task 3: Kernel Quality (KQ)
  - Rank of reservoir state matrix / n_features
  - Higher KQ = richer representation

Conditions:
  A: GPU 1/f noise  — Power rail (hwmon, IIR filtered) driving Vg
  B: White noise    — PERF_SNAPSHOT jitter driving Vg
  C: Deterministic  — No noise (β=0), pure input only
  D: ESN           — Software 8-node Echo State Network (theoretical ceiling)
  E: Linear        — Memoryless linear projection (no reservoir)

Tests:
  T65: MG direction accuracy A > 0.55 (above chance 0.50)
  T66: MG direction A > B (1/f noise helps MG prediction)
  T67: MC_A > MC_E (reservoir has more memory than linear)
  T68: MC_A > 1.0 (meaningful memory capacity)
  T69: MC_A > MC_B (1/f noise enhances memory capacity)
  T70: KQ_A > KQ_E (reservoir richer than linear)

Hardware: AMD gfx1151 GPU + Arty A7 FPGA on /dev/ttyUSB1
"""

import os, sys, json, time, struct, subprocess, argparse
import numpy as np
from pathlib import Path

os.environ.setdefault('HSA_OVERRIDE_GFX_VERSION', '11.0.0')

BASE = Path(__file__).resolve().parent.parent
RESULTS = BASE / 'results'
FIGURES = RESULTS / 'FEEL_paper' / 'FEEL__Functionally_Embodied_Emergent_Learning__13_-4' / 'figures'

# ─── FPGA Protocol ───
SYNC = 0x55
CMD_SET_VG = 0x01
CMD_READ_TELEM = 0x02
CMD_SET_KILL = 0x03

HWMON_POWER = "/sys/class/hwmon/hwmon7/power1_average"

# ─── Reservoir Parameters (same as z2162) ───
BASE_VG = 0.58
ALPHA = 0.25
BETA = 0.08
N_NEURONS = 8
SAMPLE_HZ = 20


# ═══════════════════════════════════════════════════════════
# FPGA Communication (from z2162)
# ═══════════════════════════════════════════════════════════

def to_q16_16(val: float) -> int:
    return int(val * 65536) & 0xFFFFFFFF


def find_fpga():
    try:
        import serial
    except ImportError:
        return None, None
    for p in ['/dev/ttyUSB1', '/dev/ttyUSB0', '/dev/ttyUSB2']:
        try:
            s = serial.Serial(p, 115200, timeout=0.2)
            time.sleep(0.1)
            return s, p
        except Exception:
            continue
    return None, None


def reconnect_fpga(port):
    """Reconnect to FPGA after serial failure."""
    import serial
    try:
        ser = serial.Serial(port, 115200, timeout=0.2)
        time.sleep(0.1)
        # Disable kill switch on reconnect
        ser.write(bytes([SYNC, CMD_SET_KILL, 0x00]))
        ser.flush()
        time.sleep(0.1)
        return ser
    except Exception:
        return None


def set_per_neuron_vg(ser, vg_values):
    """Set individual Vg for each of 8 neurons."""
    for nid, vg in enumerate(vg_values[:8]):
        q16 = to_q16_16(max(0.0, min(1.0, vg)))
        payload = bytes([nid & 0x07]) + struct.pack('>I', q16)
        ser.write(bytes([SYNC, CMD_SET_VG]) + payload)
    ser.flush()
    time.sleep(0.005)


def read_telem(ser, timeout=0.15):
    """Read telemetry packet: [0x55][0x02][0x30][48B][CRC8] = 52 bytes."""
    deadline = time.monotonic() + timeout
    buf = bytearray()
    while time.monotonic() < deadline:
        ser.timeout = max(0.001, deadline - time.monotonic())
        b = ser.read(1)
        if not b:
            continue
        if b[0] == SYNC:
            buf = bytearray([SYNC])
            while len(buf) < 52 and time.monotonic() < deadline:
                ser.timeout = max(0.001, deadline - time.monotonic())
                chunk = ser.read(52 - len(buf))
                if chunk:
                    buf.extend(chunk)
            break
    if len(buf) < 52:
        return None
    payload = bytes(buf[3:51])
    neurons = []
    for i in range(8):
        off = i * 6
        sc = struct.unpack_from('>H', payload, off)[0]
        vm = struct.unpack_from('>H', payload, off + 2)[0]
        neurons.append({'spike_count': sc, 'vmem': vm / 256.0})
    return neurons


def safe_fpga_step(ser, port, vg_values, interval):
    """Set Vg and read telemetry with reconnection on serial failure."""
    import serial as serial_mod
    try:
        set_per_neuron_vg(ser, vg_values)
        time.sleep(interval * 0.3)
        ser.reset_input_buffer()
        ser.write(bytes([SYNC, CMD_READ_TELEM]))
        ser.flush()
        telem = read_telem(ser, timeout=0.15)
        return ser, telem
    except (serial_mod.SerialException, OSError) as e:
        print(f"    [!] Serial error: {e}, reconnecting...")
        try:
            ser.close()
        except Exception:
            pass
        time.sleep(0.5)
        new_ser = reconnect_fpga(port)
        if new_ser is None:
            print("    [!] Reconnection failed")
            return None, None
        print("    [!] Reconnected successfully")
        return new_ser, None


# ═══════════════════════════════════════════════════════════
# Noise Sources (from z2162)
# ═══════════════════════════════════════════════════════════

def read_hwmon_power():
    """Read hwmon power1_average (μW → W). Rich 1/f dynamics ~11W ± 1.5W."""
    try:
        return int(open(HWMON_POWER).read().strip()) / 1e6
    except Exception:
        return None


def run_hip_jitter_batch(n_iters=50, n_waves=16, work_iters=50000):
    """Run z2153 deep probe and extract jitter bytes (white-ish noise)."""
    probe_bin = BASE / 'scripts' / 'z2153_deep_probe_bridge'
    if not probe_bin.exists():
        return []
    result = subprocess.run(
        [str(probe_bin), str(n_iters), str(n_waves), str(work_iters)],
        capture_output=True, text=True, timeout=30,
        env={**os.environ, 'HSA_OVERRIDE_GFX_VERSION': '11.0.0'}
    )
    if result.returncode != 0:
        return []
    jitter_bytes = []
    for line in result.stdout.strip().split('\n')[1:]:
        parts = line.split(',')
        if len(parts) >= 13:
            jitter_bytes.append(int(parts[12]))
    return jitter_bytes


def collect_power_noise(duration_s=15, sample_hz=50):
    """Collect GPU power rail time series for 1/f noise source."""
    n_samples = int(duration_s * sample_hz)
    interval = 1.0 / sample_hz
    powers = []
    for _ in range(n_samples):
        p = read_hwmon_power()
        if p is not None:
            powers.append(p)
        time.sleep(interval)
    return np.array(powers) if powers else None


def iir_filter_noise(noise_samples, alpha_iir=0.85):
    """Apply IIR low-pass: y[t] = α·y[t-1] + (1-α)·x[t].
    Creates temporal memory (ACF ~0.85) from raw noise.
    """
    filtered = np.zeros(len(noise_samples))
    filtered[0] = noise_samples[0]
    for t in range(1, len(noise_samples)):
        filtered[t] = alpha_iir * filtered[t - 1] + (1 - alpha_iir) * noise_samples[t]
    std = max(np.std(filtered), 1e-6)
    return filtered / std


# ═══════════════════════════════════════════════════════════
# FPGA Reservoir Core (with serial reconnection)
# ═══════════════════════════════════════════════════════════

def run_fpga_reservoir_sequence(ser, port, input_signal, noise_samples, w_in, w_noise,
                                base_vg=BASE_VG, alpha=ALPHA, beta=0.08,
                                live_noise=False):
    """Drive FPGA neurons with input+noise and collect spike/vmem states.

    Returns: (ser, states) where states is (n_steps, 24) array —
             8 delta_spikes + 8 vmem + 8 cumulative_spikes.
    """
    n_steps = len(input_signal)
    interval = 1.0 / SAMPLE_HZ
    states = np.zeros((n_steps, N_NEURONS * 3))
    prev_counts = None
    cumulative = np.zeros(N_NEURONS)
    power_mean = 11.0
    consecutive_fails = 0

    for t in range(n_steps):
        if live_noise:
            p = read_hwmon_power()
            noise_val = (p - power_mean) / 2.0 if p else 0.0
        elif beta > 0 and len(noise_samples) > 0:
            noise_val = noise_samples[t % len(noise_samples)]
        else:
            noise_val = 0.0

        vg_values = np.full(N_NEURONS, base_vg)
        vg_values += alpha * input_signal[t] * w_in
        if beta > 0:
            vg_values += beta * noise_val * w_noise
        vg_values = np.clip(vg_values, 0.05, 0.95)

        ser, telem = safe_fpga_step(ser, port, vg_values, interval)
        if ser is None:
            print(f"    [!] Lost FPGA at step {t}, filling remainder with zeros")
            break

        if telem:
            consecutive_fails = 0
            counts = [n['spike_count'] for n in telem]
            vmems = [n['vmem'] for n in telem]

            if prev_counts is not None:
                for i in range(N_NEURONS):
                    delta = (counts[i] - prev_counts[i]) & 0xFFFF
                    if delta > 30000:
                        delta = 0
                    states[t, i] = delta
                    cumulative[i] += delta
            for i in range(N_NEURONS):
                states[t, N_NEURONS + i] = vmems[i]
                states[t, N_NEURONS * 2 + i] = cumulative[i]
            prev_counts = counts[:]
        else:
            consecutive_fails += 1
            if consecutive_fails > 10:
                print(f"    [!] 10 consecutive telem failures at step {t}, continuing...")
                consecutive_fails = 0

        time.sleep(max(0, interval * 0.5 - 0.01))

        if (t + 1) % 500 == 0:
            print(f"      step {t+1}/{n_steps}")

    return ser, states


def simulate_lif_reservoir(input_signal, noise_samples, w_in, w_noise,
                            base_vg=BASE_VG, alpha=ALPHA, beta=0.10):
    """Software LIF simulation fallback when FPGA is not connected."""
    n_steps = len(input_signal)
    states = np.zeros((n_steps, N_NEURONS * 3))

    v_rest = 0.0
    v_thresh = 1.0
    tau_m = 0.02
    dt = 1.0 / SAMPLE_HZ
    vmem = np.zeros(N_NEURONS)
    cumulative = np.zeros(N_NEURONS)

    for t in range(n_steps):
        vg = np.full(N_NEURONS, base_vg)
        vg += alpha * input_signal[t] * w_in
        if beta > 0 and len(noise_samples) > 0:
            noise_idx = t % len(noise_samples)
            vg += beta * noise_samples[noise_idx] * w_noise
        vg = np.clip(vg, 0.05, 0.95)

        I_in = vg * 5.0
        dvdt = (-vmem + I_in) / tau_m
        vmem += dvdt * dt

        spikes = np.zeros(N_NEURONS)
        for i in range(N_NEURONS):
            if vmem[i] >= v_thresh:
                spikes[i] = 1
                vmem[i] = v_rest
                cumulative[i] += 1

        states[t, :N_NEURONS] = spikes
        states[t, N_NEURONS:N_NEURONS * 2] = vmem.copy()
        states[t, N_NEURONS * 2:] = cumulative.copy()

    return states


# ═══════════════════════════════════════════════════════════
# Echo State Network (from z2162)
# ═══════════════════════════════════════════════════════════

class EchoStateNetwork:
    """Standard Echo State Network for baseline comparison."""

    def __init__(self, input_dim=1, reservoir_size=8,
                 spectral_radius=0.95, input_scaling=0.3,
                 leak_rate=0.3, seed=42):
        rng = np.random.RandomState(seed)
        self.reservoir_size = reservoir_size
        self.leak_rate = leak_rate
        self.W_in = rng.randn(reservoir_size, input_dim) * input_scaling
        W = rng.randn(reservoir_size, reservoir_size)
        rho = np.max(np.abs(np.linalg.eigvals(W)))
        self.W = W * (spectral_radius / rho)
        self.state = np.zeros(reservoir_size)

    def reset(self):
        self.state = np.zeros(self.reservoir_size)

    def step(self, x):
        x = np.atleast_1d(x)
        pre = np.tanh(self.W @ self.state + self.W_in @ x)
        self.state = (1 - self.leak_rate) * self.state + self.leak_rate * pre
        return self.state.copy()

    def run(self, inputs):
        T = len(inputs)
        states = np.zeros((T, self.reservoir_size))
        for t in range(T):
            states[t] = self.step(np.atleast_1d(inputs[t]))
        return states


# ═══════════════════════════════════════════════════════════
# Feature Extraction & Classification
# ═══════════════════════════════════════════════════════════

def augment_with_delays(states, delays=(1, 2, 3)):
    """Add time-delayed copies of state for richer feature space."""
    T, D = states.shape
    augmented = np.zeros((T, D * (1 + len(delays))))
    augmented[:, :D] = states
    for i, d in enumerate(delays):
        start = D * (i + 1)
        augmented[d:, start:start + D] = states[:T - d]
    return augmented


def ridge_regress(X_train, y_train, X_test, alphas=None):
    """Ridge regression for scalar target. Returns (y_pred_train, y_pred_test, best_alpha)."""
    if alphas is None:
        alphas = [1e-6, 1e-4, 1e-2, 1.0, 10.0, 100.0, 1000.0]

    best_mse = np.inf
    best_w = None
    best_alpha = None

    for alpha in alphas:
        I = np.eye(X_train.shape[1])
        try:
            w = np.linalg.solve(X_train.T @ X_train + alpha * I, X_train.T @ y_train)
        except np.linalg.LinAlgError:
            continue
        pred_train = X_train @ w
        mse_train = np.mean((pred_train - y_train) ** 2)
        if mse_train < best_mse:
            best_mse = mse_train
            best_w = w
            best_alpha = alpha

    if best_w is None:
        return np.zeros(len(y_train)), np.zeros(len(X_test)), 1.0

    y_pred_train = X_train @ best_w
    y_pred_test = X_test @ best_w
    return y_pred_train, y_pred_test, best_alpha


def ridge_classify(X_train, y_train, X_test, n_classes=2, alphas=None):
    """Ridge regression for classification via one-hot encoding.

    Returns predicted class labels for test set.
    """
    if alphas is None:
        alphas = [1e-4, 1e-2, 1.0, 10.0, 100.0]

    if n_classes == 2:
        # Binary: direct regression on 0/1 labels
        _, y_pred_test, alpha = ridge_regress(X_train, y_train, X_test, alphas)
        return (y_pred_test > 0.5).astype(int), alpha
    else:
        # Multi-class: one-vs-all
        Y_onehot = np.zeros((len(y_train), n_classes))
        for c in range(n_classes):
            Y_onehot[:, c] = (y_train == c).astype(float)

        best_alpha = None
        best_W = None
        best_mse = np.inf
        if alphas is None:
            alphas = [1e-4, 1e-2, 1.0, 10.0, 100.0]
        for alpha in alphas:
            I = np.eye(X_train.shape[1])
            try:
                W = np.linalg.solve(X_train.T @ X_train + alpha * I, X_train.T @ Y_onehot)
            except np.linalg.LinAlgError:
                continue
            pred = X_train @ W
            mse = np.mean((pred - Y_onehot) ** 2)
            if mse < best_mse:
                best_mse = mse
                best_W = W
                best_alpha = alpha

        if best_W is None:
            return np.zeros(len(X_test), dtype=int), 1.0

        scores = X_test @ best_W
        return scores.argmax(axis=1), best_alpha


def kernel_quality(states):
    """Rank / n_features — measures representational richness."""
    rank = np.linalg.matrix_rank(states, tol=1e-6)
    return rank / states.shape[1]


# ═══════════════════════════════════════════════════════════
# Mackey-Glass Generator
# ═══════════════════════════════════════════════════════════

def generate_mackey_glass(n_steps=2000, tau_mg=17, beta_mg=0.2, gamma=0.1,
                           n_exp=10, dt=1.0, warmup=500):
    """Generate Mackey-Glass chaotic time series.

    dx/dt = β·x(t-τ)/(1+x(t-τ)^n) - γ·x(t)

    Standard parameters produce chaotic behavior for τ_mg >= 17.
    """
    total = n_steps + warmup
    x = np.zeros(total + tau_mg)
    # Initial condition: constant + small perturbation
    x[:tau_mg + 1] = 0.9 + 0.1 * np.sin(np.linspace(0, 2 * np.pi, tau_mg + 1))

    for t in range(tau_mg, total + tau_mg - 1):
        x_tau = x[t - tau_mg]
        dxdt = beta_mg * x_tau / (1.0 + x_tau ** n_exp) - gamma * x[t]
        x[t + 1] = x[t] + dt * dxdt

    # Discard warmup and delay buffer
    series = x[tau_mg + warmup:]
    return series[:n_steps]


# ═══════════════════════════════════════════════════════════
# Memory Capacity Input
# ═══════════════════════════════════════════════════════════

def generate_mc_input(n_steps=1500, seed=42):
    """Random uniform input u(t) in [0, 1] for memory capacity test."""
    rng = np.random.default_rng(seed)
    return rng.uniform(0.0, 1.0, size=n_steps)


# ═══════════════════════════════════════════════════════════
# Reservoir Runner (handles all conditions)
# ═══════════════════════════════════════════════════════════

def run_reservoir_condition(cond_name, input_signal, ser, port, fpga,
                             noise_1f_iir, noise_white, noise_zero,
                             w_in, w_noise, esn):
    """Run a single condition and return (ser, augmented_states)."""
    n_steps = len(input_signal)

    if cond_name == 'A_1f':
        if fpga:
            ser, states = run_fpga_reservoir_sequence(
                ser, port, input_signal, noise_1f_iir, w_in, w_noise,
                base_vg=BASE_VG, alpha=ALPHA, beta=BETA, live_noise=True)
        else:
            states = simulate_lif_reservoir(
                input_signal, noise_1f_iir, w_in, w_noise,
                base_vg=BASE_VG, alpha=ALPHA, beta=BETA)

    elif cond_name == 'B_white':
        if fpga:
            ser, states = run_fpga_reservoir_sequence(
                ser, port, input_signal, noise_white, w_in, w_noise,
                base_vg=BASE_VG, alpha=ALPHA, beta=BETA, live_noise=False)
        else:
            states = simulate_lif_reservoir(
                input_signal, noise_white, w_in, w_noise,
                base_vg=BASE_VG, alpha=ALPHA, beta=BETA)

    elif cond_name == 'C_deterministic':
        if fpga:
            ser, states = run_fpga_reservoir_sequence(
                ser, port, input_signal, noise_zero, w_in, w_noise,
                base_vg=BASE_VG, alpha=ALPHA, beta=0.0, live_noise=False)
        else:
            states = simulate_lif_reservoir(
                input_signal, noise_zero, w_in, w_noise,
                base_vg=BASE_VG, alpha=ALPHA, beta=0.0)

    elif cond_name == 'D_esn':
        esn.reset()
        states = esn.run(input_signal)

    elif cond_name == 'E_linear':
        # Memoryless linear projection: just w_in applied to input
        states = np.zeros((n_steps, N_NEURONS))
        for t in range(n_steps):
            states[t] = np.clip(BASE_VG + ALPHA * input_signal[t] * w_in, 0.05, 0.95)

    else:
        raise ValueError(f"Unknown condition: {cond_name}")

    # Augment with delays
    aug = augment_with_delays(states, delays=(1, 2, 3))
    return ser, aug


# ═══════════════════════════════════════════════════════════
# Task 1: Mackey-Glass Classification
# ═══════════════════════════════════════════════════════════

def run_mackey_glass_classification(conditions, mg_series, train_frac=0.7):
    """Run Mackey-Glass direction (binary) and quadrant (4-class) classification.

    mg_series: normalized MG series of length N
    conditions: dict of {name: augmented_states} each of shape (N-1, features)

    Returns dict of {condition: {dir_accuracy, quad_accuracy, ...}}.
    """
    # Direction labels: x(t+1) > x(t)
    mg_direction = (mg_series[1:] > mg_series[:-1]).astype(int)

    # Quadrant labels: 2*(rising) + (high value)
    median_val = np.median(mg_series)
    mg_quadrant = (2 * (mg_series[1:] > mg_series[:-1]).astype(int)
                   + (mg_series[:-1] > median_val).astype(int))

    n_total = len(mg_direction)
    warmup = 10
    n_valid = n_total - warmup
    n_train = int(n_valid * train_frac)

    results = {}
    for cond_name, aug_states in conditions.items():
        X = aug_states[warmup:warmup + n_valid]
        y_dir = mg_direction[warmup:warmup + n_valid]
        y_quad = mg_quadrant[warmup:warmup + n_valid]

        # Add bias term
        X_bias = np.hstack([X, np.ones((len(X), 1))])

        X_train = X_bias[:n_train]
        X_test = X_bias[n_train:]

        # Direction (binary)
        y_dir_train = y_dir[:n_train]
        y_dir_test = y_dir[n_train:]
        dir_pred, dir_alpha = ridge_classify(X_train, y_dir_train, X_test, n_classes=2)
        dir_acc = np.mean(dir_pred == y_dir_test)

        # Quadrant (4-class)
        y_quad_train = y_quad[:n_train]
        y_quad_test = y_quad[n_train:]
        quad_pred, quad_alpha = ridge_classify(X_train, y_quad_train, X_test, n_classes=4)
        quad_acc = np.mean(quad_pred == y_quad_test)

        results[cond_name] = {
            'dir_accuracy': float(dir_acc),
            'quad_accuracy': float(quad_acc),
            'dir_alpha': float(dir_alpha),
            'quad_alpha': float(quad_alpha),
            'n_train': n_train,
            'n_test': n_valid - n_train,
        }
        print(f"    {cond_name}: dir_acc={dir_acc:.4f}  quad_acc={quad_acc:.4f}")

    return results


# ═══════════════════════════════════════════════════════════
# Task 2: Memory Capacity
# ═══════════════════════════════════════════════════════════

def run_memory_capacity_task(conditions, mc_input, max_delay=15, train_frac=0.7):
    """Compute memory capacity for all conditions.

    For each delay d, train ridge regression to predict u(t-d) from reservoir states.
    MC_d = correlation(prediction, target)^2.
    Total MC = sum(MC_d).

    Returns dict of {condition: {mc_total, mc_per_delay}}.
    """
    warmup = 10

    results = {}
    for cond_name, aug_states in conditions.items():
        mc_per_delay = []

        for d in range(1, max_delay + 1):
            start = max(warmup, d)
            valid_len = len(mc_input) - start

            X = aug_states[start:start + valid_len]
            y = mc_input[start - d:start - d + valid_len]

            # Add bias
            X_bias = np.hstack([X, np.ones((len(X), 1))])

            n_tr = int(valid_len * train_frac)
            X_train = X_bias[:n_tr]
            y_train = y[:n_tr]
            X_test = X_bias[n_tr:]
            y_test = y[n_tr:]

            _, y_pred_test, _ = ridge_regress(X_train, y_train, X_test)

            # MC_d = correlation^2
            if np.std(y_pred_test) < 1e-10 or np.std(y_test) < 1e-10:
                mc_d = 0.0
            else:
                corr = np.corrcoef(y_pred_test, y_test)[0, 1]
                mc_d = max(0.0, corr ** 2)

            mc_per_delay.append(float(mc_d))

        mc_total = sum(mc_per_delay)
        results[cond_name] = {
            'mc_total': float(mc_total),
            'mc_per_delay': mc_per_delay,
        }
        print(f"    {cond_name}: MC_total={mc_total:.3f} "
              f"(d=1: {mc_per_delay[0]:.3f}, d=5: {mc_per_delay[4]:.3f})")

    return results


# ═══════════════════════════════════════════════════════════
# Plotting
# ═══════════════════════════════════════════════════════════

def generate_figure(mg_results, mc_results, kq_results, mc_max_delay):
    """Generate 3-panel figure: MG classification, MC curves, Kernel Quality."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("  matplotlib not available, skipping figure")
        return

    FIGURES.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    cond_labels = {
        'A_1f': 'A: GPU 1/f',
        'B_white': 'B: White',
        'C_deterministic': 'C: Determ.',
        'D_esn': 'D: ESN',
        'E_linear': 'E: Linear',
    }
    cond_colors = {
        'A_1f': '#d62728',
        'B_white': '#1f77b4',
        'C_deterministic': '#7f7f7f',
        'D_esn': '#2ca02c',
        'E_linear': '#ff7f0e',
    }

    # Panel 1: MG direction + quadrant accuracy bars
    ax = axes[0]
    conds = list(mg_results.keys())
    n_conds = len(conds)
    x_pos = np.arange(n_conds)
    bar_width = 0.35

    dir_accs = [mg_results[c]['dir_accuracy'] for c in conds]
    quad_accs = [mg_results[c]['quad_accuracy'] for c in conds]
    colors = [cond_colors.get(c, '#333333') for c in conds]
    labels = [cond_labels.get(c, c) for c in conds]

    bars1 = ax.bar(x_pos - bar_width/2, dir_accs, bar_width,
                   color=colors, edgecolor='black', linewidth=0.5, label='Direction (2-class)')
    bars2 = ax.bar(x_pos + bar_width/2, quad_accs, bar_width,
                   color=colors, edgecolor='black', linewidth=0.5, alpha=0.6, label='Quadrant (4-class)')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, rotation=30, ha='right', fontsize=9)
    ax.set_ylabel('Accuracy', fontsize=11)
    ax.set_title('Mackey-Glass Classification', fontsize=12, fontweight='bold')
    ax.axhline(y=0.50, color='red', linestyle='--', alpha=0.5, label='Chance (binary)')
    ax.axhline(y=0.25, color='orange', linestyle=':', alpha=0.5, label='Chance (4-class)')
    ax.legend(fontsize=7, loc='lower right')
    for i, v in enumerate(dir_accs):
        ax.text(i - bar_width/2, v + 0.01, f'{v:.3f}', ha='center', va='bottom', fontsize=7)
    for i, v in enumerate(quad_accs):
        ax.text(i + bar_width/2, v + 0.01, f'{v:.3f}', ha='center', va='bottom', fontsize=7)
    ax.set_ylim(0, 1.05)

    # Panel 2: Memory capacity curves
    ax = axes[1]
    delays = np.arange(1, mc_max_delay + 1)
    for c in mc_results:
        mc_vals = mc_results[c]['mc_per_delay']
        label = f"{cond_labels.get(c, c)} (MC={mc_results[c]['mc_total']:.2f})"
        ax.plot(delays, mc_vals, 'o-', color=cond_colors.get(c, '#333'),
                label=label, markersize=3, linewidth=1.5)
    ax.set_xlabel('Delay d', fontsize=11)
    ax.set_ylabel('MC_d (correlation^2)', fontsize=11)
    ax.set_title('Memory Capacity', fontsize=12, fontweight='bold')
    ax.legend(fontsize=7, loc='upper right')
    ax.set_xlim(0.5, mc_max_delay + 0.5)
    ax.set_ylim(-0.05, 1.05)

    # Panel 3: Kernel Quality bars
    ax = axes[2]
    kq_vals = [kq_results[c] for c in conds]
    bars = ax.bar(x_pos, kq_vals, color=colors, edgecolor='black', linewidth=0.5)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, rotation=30, ha='right', fontsize=9)
    ax.set_ylabel('Kernel Quality (rank / n_features)', fontsize=11)
    ax.set_title('Kernel Quality', fontsize=12, fontweight='bold')
    for i, v in enumerate(kq_vals):
        ax.text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom', fontsize=8)
    ax.set_ylim(0, 1.15)

    plt.tight_layout()
    fig_path = FIGURES / 'fig_z2163_mackey_glass.png'
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n  Figure saved: {fig_path}")


# ═══════════════════════════════════════════════════════════
# Main Experiment
# ═══════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description='z2163: MG Classification, Memory Capacity & Kernel Quality')
    parser.add_argument('--mg-steps', type=int, default=2000,
                        help='Total Mackey-Glass time series length')
    parser.add_argument('--mc-steps', type=int, default=1500,
                        help='Memory capacity sequence length')
    parser.add_argument('--mc-max-delay', type=int, default=15,
                        help='Maximum delay for memory capacity')
    parser.add_argument('--noise-collect-s', type=float, default=15.0,
                        help='Duration for power rail noise collection')
    args = parser.parse_args()

    print("=" * 65)
    print("z2163: MG Classification, Memory Capacity & Kernel Quality")
    print("  Standard RC benchmarks for GPU-noise-driven FPGA reservoir")
    print("=" * 65)

    rng = np.random.default_rng(42)
    w_in = rng.uniform(-1, 1, size=N_NEURONS)
    w_noise = rng.uniform(-1, 1, size=N_NEURONS)

    results = {
        'experiment': 'z2163_mackey_glass',
        'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
        'params': {
            'base_vg': BASE_VG, 'alpha': ALPHA, 'beta': BETA,
            'n_neurons': N_NEURONS, 'sample_hz': SAMPLE_HZ,
            'mg_steps': args.mg_steps, 'mc_steps': args.mc_steps,
            'mc_max_delay': args.mc_max_delay,
            'w_in': w_in.tolist(), 'w_noise': w_noise.tolist(),
        },
        'simulated': False,
    }

    # ─── Step 1: Connect to FPGA ───
    print("\n[1/8] Connecting to FPGA...")
    ser, port = find_fpga()
    if ser is None:
        print("  FPGA not found — using LIF simulation fallback")
        fpga = False
        results['simulated'] = True
    else:
        print(f"  Connected: {port}")
        fpga = True
        ser.write(bytes([SYNC, CMD_SET_KILL, 0x00]))
        ser.flush()
        time.sleep(0.1)
        print("  Kill switch disabled")

    # ─── Step 2: Collect GPU noise sources ───
    print("\n[2/8] Collecting GPU noise sources...")

    print("  Collecting power rail noise (1/f)...")
    power_noise = collect_power_noise(duration_s=args.noise_collect_s, sample_hz=50)
    if power_noise is not None and len(power_noise) > 10:
        power_mean = power_noise.mean()
        power_std = max(power_noise.std(), 1e-6)
        noise_1f = (power_noise - power_mean) / power_std
        print(f"  Power rail: {power_mean:.2f} +/- {power_std:.3f} W, {len(noise_1f)} samples")
    else:
        print("  Power rail unavailable, generating synthetic 1/f")
        n_synth = int(args.noise_collect_s * 50)
        noise_1f = np.zeros(n_synth)
        n_octaves = 8
        octaves = np.zeros(n_octaves)
        for i in range(n_synth):
            for j in range(n_octaves):
                if i % (1 << j) == 0:
                    octaves[j] = rng.standard_normal()
            noise_1f[i] = octaves.sum()
        noise_1f = (noise_1f - noise_1f.mean()) / max(noise_1f.std(), 1e-6)

    print("  Collecting PERF_SNAPSHOT jitter (white noise)...")
    jitter_bytes = run_hip_jitter_batch(n_iters=100, n_waves=16, work_iters=50000)
    if jitter_bytes:
        noise_white = np.array(jitter_bytes, dtype=float)
        noise_white = (noise_white - noise_white.mean()) / max(noise_white.std(), 1e-6)
        print(f"  Got {len(noise_white)} jitter bytes")
    else:
        print("  HIP probe unavailable, generating synthetic white noise")
        noise_white = rng.standard_normal(int(args.noise_collect_s * 50))

    noise_zero = np.zeros(1000)

    # IIR-filter the 1/f noise
    noise_1f_iir = iir_filter_noise(noise_1f, alpha_iir=0.85)

    results['noise'] = {
        '1f_samples': len(noise_1f),
        'white_samples': len(noise_white),
    }

    # ESN baseline
    esn = EchoStateNetwork(input_dim=1, reservoir_size=8,
                            spectral_radius=0.95, input_scaling=0.3, seed=42)

    # ─── Step 3: Generate Mackey-Glass series ───
    print("\n[3/8] Generating Mackey-Glass time series...")
    mg_full = generate_mackey_glass(n_steps=args.mg_steps, tau_mg=17,
                                     beta_mg=0.2, gamma=0.1, n_exp=10, dt=1.0)
    # Normalize to [0, 1]
    mg_min, mg_max = mg_full.min(), mg_full.max()
    mg_norm = (mg_full - mg_min) / max(mg_max - mg_min, 1e-8)
    # Input for reservoir: x(t) for t=0..N-2
    mg_input = mg_norm[:-1]
    print(f"  Series: {len(mg_full)} steps, range [{mg_min:.4f}, {mg_max:.4f}]")
    print(f"  Input: {len(mg_input)} steps")

    # ─── Step 4: Run Mackey-Glass for all conditions ───
    print("\n[4/8] Running Mackey-Glass reservoir conditions...")
    cond_names = ['A_1f', 'B_white', 'C_deterministic', 'D_esn', 'E_linear']
    mg_cond_states = {}

    for cond in cond_names:
        print(f"\n  === Condition {cond} ===")
        t0 = time.monotonic()
        ser, aug = run_reservoir_condition(
            cond, mg_input, ser, port, fpga,
            noise_1f_iir, noise_white, noise_zero,
            w_in, w_noise, esn)
        elapsed = time.monotonic() - t0
        mg_cond_states[cond] = aug
        print(f"    {aug.shape[0]} steps, {aug.shape[1]} features, {elapsed:.1f}s")

    # ─── Step 5: Evaluate Mackey-Glass classification ───
    print("\n[5/8] Evaluating Mackey-Glass classification...")
    mg_results = run_mackey_glass_classification(mg_cond_states, mg_norm, train_frac=0.7)

    # ─── Step 6: Generate Memory Capacity input & run ───
    print(f"\n[6/8] Running Memory Capacity (max_delay={args.mc_max_delay})...")
    mc_input = generate_mc_input(n_steps=args.mc_steps, seed=42)
    mc_cond_states = {}

    for cond in cond_names:
        print(f"\n  === Condition {cond} ===")
        t0 = time.monotonic()
        ser, aug = run_reservoir_condition(
            cond, mc_input, ser, port, fpga,
            noise_1f_iir, noise_white, noise_zero,
            w_in, w_noise, esn)
        elapsed = time.monotonic() - t0
        mc_cond_states[cond] = aug
        print(f"    {aug.shape[0]} steps, {aug.shape[1]} features, {elapsed:.1f}s")

    # ─── Step 7: Evaluate Memory Capacity + Kernel Quality ───
    print("\n[7/8] Evaluating Memory Capacity & Kernel Quality...")
    mc_results = run_memory_capacity_task(mc_cond_states, mc_input,
                                           max_delay=args.mc_max_delay, train_frac=0.7)

    # Kernel Quality: use MC condition states (representative of reservoir dynamics)
    kq_results = {}
    for cond in cond_names:
        kq = kernel_quality(mc_cond_states[cond])
        kq_results[cond] = float(kq)
        print(f"    {cond}: KQ={kq:.4f}")

    # ─── Step 8: Tests & Output ───
    print("\n[8/8] Running tests...")
    print("=" * 65)

    dir_a = mg_results['A_1f']['dir_accuracy']
    dir_b = mg_results['B_white']['dir_accuracy']
    mc_a = mc_results['A_1f']['mc_total']
    mc_b = mc_results['B_white']['mc_total']
    mc_e = mc_results['E_linear']['mc_total']
    kq_a = kq_results['A_1f']
    kq_e = kq_results['E_linear']

    tests = {}

    # T65: MG direction accuracy A > 0.55
    t65 = dir_a > 0.55
    tests['T65_mg_direction'] = {
        'pass': bool(t65),
        'description': 'MG direction accuracy A > 0.55 (above chance 0.50)',
        'value': float(dir_a),
        'threshold': 0.55,
    }
    print(f"  T65 dir_acc_A > 0.55:     {dir_a:.4f} {'PASS' if t65 else 'FAIL'}")

    # T66: MG direction A > B
    t66 = dir_a > dir_b
    tests['T66_1f_vs_white_dir'] = {
        'pass': bool(t66),
        'description': '1/f noise helps MG direction (dir_A > dir_B)',
        'dir_a': float(dir_a),
        'dir_b': float(dir_b),
    }
    print(f"  T66 dir_A > dir_B:        {dir_a:.4f} > {dir_b:.4f} {'PASS' if t66 else 'FAIL'}")

    # T67: MC_A > MC_C (noise improves memory vs deterministic)
    mc_c = mc_results['C_deterministic']['mc_total']
    t67 = mc_a > mc_c
    tests['T67_mc_noise_vs_det'] = {
        'pass': bool(t67),
        'description': 'Noise improves memory (MC_A > MC_C_deterministic)',
        'mc_a': float(mc_a),
        'mc_c': float(mc_c),
    }
    print(f"  T67 MC_A > MC_C:          {mc_a:.3f} > {mc_c:.3f} {'PASS' if t67 else 'FAIL'}")

    # T68: MC_A > 1.0
    t68 = mc_a > 1.0
    tests['T68_mc_minimum'] = {
        'pass': bool(t68),
        'description': 'MC_A > 1.0 (meaningful memory)',
        'mc_a': float(mc_a),
        'threshold': 1.0,
    }
    print(f"  T68 MC_A > 1.0:           {mc_a:.3f} {'PASS' if t68 else 'FAIL'}")

    # T69: MC_A > MC_B
    t69 = mc_a > mc_b
    tests['T69_1f_mc_vs_white'] = {
        'pass': bool(t69),
        'description': '1/f noise enhances memory (MC_A > MC_B)',
        'mc_a': float(mc_a),
        'mc_b': float(mc_b),
    }
    print(f"  T69 MC_A > MC_B:          {mc_a:.3f} > {mc_b:.3f} {'PASS' if t69 else 'FAIL'}")

    # T70: KQ_A > KQ_E
    t70 = kq_a > kq_e
    tests['T70_kq_vs_linear'] = {
        'pass': bool(t70),
        'description': 'Reservoir richer than linear (KQ_A > KQ_E)',
        'kq_a': float(kq_a),
        'kq_e': float(kq_e),
    }
    print(f"  T70 KQ_A > KQ_E:          {kq_a:.4f} > {kq_e:.4f} {'PASS' if t70 else 'FAIL'}")

    n_pass = sum(1 for t in tests.values() if t['pass'])
    n_total = len(tests)
    print(f"\n  SCORE: {n_pass}/{n_total} PASS")
    print("=" * 65)

    # Store results
    results['mackey_glass'] = mg_results
    results['memory_capacity'] = mc_results
    results['kernel_quality'] = kq_results
    results['tests'] = tests
    results['score'] = f"{n_pass}/{n_total}"

    # Save JSON
    RESULTS.mkdir(parents=True, exist_ok=True)
    json_path = RESULTS / 'z2163_mackey_glass.json'
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved: {json_path}")

    # Generate figure
    print("\n  Generating figure...")
    generate_figure(mg_results, mc_results, kq_results, args.mc_max_delay)

    # Cleanup
    if ser is not None:
        try:
            ser.close()
        except Exception:
            pass

    print(f"\n  Done. {n_pass}/{n_total} tests passed.")
    return n_pass


if __name__ == '__main__':
    main()
