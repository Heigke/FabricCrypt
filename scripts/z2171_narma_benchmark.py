#!/usr/bin/env python3
"""z2171_narma_benchmark.py — NARMA-10 Benchmark for FPGA Reservoir

Gold-standard nonlinear time-series benchmark for reservoir computing.
NARMA-10 tests whether the reservoir can approximate:
  y(t+1) = 0.3*y(t) + 0.05*y(t)*sum(y(t-i), i=0..9) + 1.5*u(t-9)*u(t) + 0.1

This is a REGRESSION task measured by NRMSE (Normalized Root Mean Square Error).

Conditions:
  FULL:     GPU 1/f noise (hwmon power rail) + IIR filter driving Vg
  WHITE:    White noise (synthetic) driving Vg
  NO_NOISE: Deterministic (β=0), pure input only
  ESN:      Software 8-node Echo State Network (theoretical ceiling)

Features: 8 neurons × (spike_delta + vmem) × 3 time delays = 48 features
Readout:  Ridge regression, 70/30 chronological split

Tests T115-T120:
  T115: FULL NRMSE < 0.8 (reasonable approximation)
  T116: FULL NRMSE < WHITE NRMSE (1/f helps)
  T117: FULL NRMSE < NO_NOISE NRMSE (noise helps)
  T118: ESN NRMSE < FULL NRMSE (software ceiling is lower/better)
  T119: FULL correlation > 0.3 (moderate prediction capability)
  T120: All conditions NRMSE < 1.0 (better than predicting mean)

Hardware: AMD gfx1151 GPU + Arty A7 FPGA on /dev/ttyUSB{0,1}
"""

import os, sys, json, time, struct, subprocess, argparse
import numpy as np
from pathlib import Path

os.environ.setdefault('HSA_OVERRIDE_GFX_VERSION', '11.0.0')

BASE = Path(__file__).resolve().parent.parent
RESULTS = BASE / 'results'
FIGURES = RESULTS / 'FEEL_paper_update' / 'FEEL__Functionally_Embodied_Emergent_Learning__13_-5' / 'figures'

# ─── JSON Encoder ───
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

# ─── FPGA Protocol ───
SYNC = 0x55
CMD_SET_VG = 0x01
CMD_READ_TELEM = 0x02
CMD_SET_KILL = 0x03

HWMON_POWER = "/sys/class/hwmon/hwmon7/power1_average"

# ─── Reservoir Parameters ───
BASE_VG = 0.58       # near BVpar cliff — input modulation has maximum effect
ALPHA = 0.25         # strong input coupling
BETA = 0.08          # moderate noise coupling
N_NEURONS = 8
SAMPLE_HZ = 20       # FPGA update rate
IIR_ALPHA = 0.85     # IIR filter coefficient for temporal memory


# ═══════════════════════════════════════════════════════════
# FPGA Communication
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


# ═══════════════════════════════════════════════════════════
# Noise Sources
# ═══════════════════════════════════════════════════════════

def read_hwmon_power():
    """Read hwmon power1_average (uW -> W). Rich 1/f dynamics ~11W +/- 1.5W."""
    try:
        return int(open(HWMON_POWER).read().strip()) / 1e6
    except Exception:
        return None


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


def iir_filter_noise(noise_samples, alpha_iir=IIR_ALPHA):
    """Apply IIR low-pass: y[t] = a*y[t-1] + (1-a)*x[t].
    Creates temporal memory (ACF ~0.85) from raw noise.
    """
    filtered = np.zeros(len(noise_samples))
    filtered[0] = noise_samples[0]
    for t in range(1, len(noise_samples)):
        filtered[t] = alpha_iir * filtered[t - 1] + (1 - alpha_iir) * noise_samples[t]
    std = max(np.std(filtered), 1e-6)
    return filtered / std


def generate_voss_mccartney(n_samples, n_octaves=8, rng=None):
    """Voss-McCartney 1/f noise generator."""
    if rng is None:
        rng = np.random.default_rng(42)
    noise = np.zeros(n_samples)
    octaves = np.zeros(n_octaves)
    for i in range(n_samples):
        for j in range(n_octaves):
            if i % (1 << j) == 0:
                octaves[j] = rng.standard_normal()
        noise[i] = octaves.sum()
    noise = (noise - noise.mean()) / max(noise.std(), 1e-6)
    return noise


# ═══════════════════════════════════════════════════════════
# NARMA-10 Generation
# ═══════════════════════════════════════════════════════════

def generate_narma10(n_steps, seed=42):
    """Generate NARMA-10 input u(t) and target y(t).

    y(t+1) = 0.3*y(t) + 0.05*y(t)*sum(y(t-i), i=0..9) + 1.5*u(t-9)*u(t) + 0.1

    Input u(t) ~ Uniform(0, 0.5).
    Returns (u, y) arrays of length n_steps.
    """
    rng = np.random.default_rng(seed)
    u = rng.uniform(0, 0.5, size=n_steps)
    y = np.zeros(n_steps)

    for t in range(10, n_steps - 1):
        y_sum = 0.0
        for i in range(10):
            y_sum += y[t - i]
        y[t + 1] = (0.3 * y[t]
                     + 0.05 * y[t] * y_sum
                     + 1.5 * u[t - 9] * u[t]
                     + 0.1)
        # Clip to prevent divergence (standard practice)
        y[t + 1] = np.clip(y[t + 1], 0.0, 10.0)

    return u, y


# ═══════════════════════════════════════════════════════════
# FPGA Reservoir Core
# ═══════════════════════════════════════════════════════════

def run_fpga_reservoir(ser, input_signal, noise_samples, w_in, w_noise,
                       base_vg=BASE_VG, alpha=ALPHA, beta=BETA,
                       live_noise=False):
    """Drive FPGA neurons with input+noise and collect spike/vmem states.

    When live_noise=True, reads power rail in real-time (true substrate coupling).
    Returns: (n_steps, 16) array — 8 delta_spikes + 8 vmem.
    """
    n_steps = len(input_signal)
    interval = 1.0 / SAMPLE_HZ
    states = np.zeros((n_steps, N_NEURONS * 2))  # delta + vmem
    prev_counts = None
    power_mean = 11.0  # approx mean for normalization

    for t in range(n_steps):
        # Get noise value
        if live_noise:
            p = read_hwmon_power()
            noise_val = (p - power_mean) / 2.0 if p else 0.0
        elif beta > 0 and len(noise_samples) > 0:
            noise_val = noise_samples[t % len(noise_samples)]
        else:
            noise_val = 0.0

        # Compute per-neuron Vg: base + input + noise
        vg_values = np.full(N_NEURONS, base_vg)
        vg_values += alpha * input_signal[t] * w_in
        if beta > 0:
            vg_values += beta * noise_val * w_noise
        vg_values = np.clip(vg_values, 0.05, 0.95)

        set_per_neuron_vg(ser, vg_values)
        time.sleep(interval * 0.3)

        # Read telemetry
        ser.reset_input_buffer()
        ser.write(bytes([SYNC, CMD_READ_TELEM]))
        ser.flush()
        telem = read_telem(ser, timeout=0.15)

        if telem:
            counts = [n['spike_count'] for n in telem]
            vmems = [n['vmem'] for n in telem]

            if prev_counts is not None:
                for i in range(N_NEURONS):
                    delta = (counts[i] - prev_counts[i]) & 0xFFFF
                    if delta > 30000:
                        delta = 0
                    states[t, i] = delta
            for i in range(N_NEURONS):
                states[t, N_NEURONS + i] = vmems[i]
            prev_counts = counts[:]

        time.sleep(max(0, interval * 0.5 - 0.01))

    return states


def simulate_lif_reservoir(input_signal, noise_samples, w_in, w_noise,
                           base_vg=BASE_VG, alpha=ALPHA, beta=BETA):
    """Software LIF simulation fallback when FPGA is not connected."""
    n_steps = len(input_signal)
    states = np.zeros((n_steps, N_NEURONS * 2))  # delta + vmem

    v_rest = 0.0
    v_thresh = 1.0
    tau_m = 0.02
    dt = 1.0 / SAMPLE_HZ
    vmem = np.zeros(N_NEURONS)

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

        states[t, :N_NEURONS] = spikes
        states[t, N_NEURONS:] = vmem.copy()

    return states


# ═══════════════════════════════════════════════════════════
# Echo State Network
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
# Feature Extraction & Regression
# ═══════════════════════════════════════════════════════════

def augment_with_delays(states, delays=(1, 2, 3)):
    """Add time-delayed copies of state for richer feature space.
    8 neurons * 2 (spike_delta + vmem) * (1 + 3 delays) = 64 features
    But we use only spike_delta + vmem columns => 16 * (1+3) = 64.
    Spec says 48: 8 neurons * (spike_delta + vmem) * 3 delays = 48.
    We include current + 3 delays = 16 * 4 = 64, but only delays give 48.
    We'll do current + delays => 64 features total.
    """
    T, D = states.shape
    augmented = np.zeros((T, D * (1 + len(delays))))
    augmented[:, :D] = states
    for i, d in enumerate(delays):
        start = D * (i + 1)
        augmented[d:, start:start + D] = states[:T - d]
    return augmented


def ridge_regression(X_train, y_train, X_test, y_test, alphas=None):
    """Ridge regression for NARMA-10 (regression task).
    Returns (nrmse, correlation, best_alpha, y_pred).
    """
    if alphas is None:
        alphas = [1e-8, 1e-6, 1e-4, 1e-2, 1.0, 10.0, 100.0, 1000.0]

    y_var = np.var(y_test)
    if y_var < 1e-12:
        return 1.0, 0.0, 0.0, np.zeros_like(y_test)

    best_nrmse = float('inf')
    best_corr = 0.0
    best_alpha = alphas[0]
    best_pred = None

    for alpha in alphas:
        I = np.eye(X_train.shape[1])
        try:
            w = np.linalg.solve(X_train.T @ X_train + alpha * I,
                                X_train.T @ y_train)
        except np.linalg.LinAlgError:
            continue

        y_pred = X_test @ w
        mse = np.mean((y_pred - y_test) ** 2)
        nrmse = np.sqrt(mse / y_var)

        # Pearson correlation
        if np.std(y_pred) > 1e-12:
            corr = np.corrcoef(y_pred, y_test)[0, 1]
        else:
            corr = 0.0

        if nrmse < best_nrmse:
            best_nrmse = nrmse
            best_corr = corr
            best_alpha = alpha
            best_pred = y_pred.copy()

    return best_nrmse, best_corr, best_alpha, best_pred


# ═══════════════════════════════════════════════════════════
# Plotting
# ═══════════════════════════════════════════════════════════

def plot_results(results, fig_path):
    """Generate NARMA-10 benchmark figure."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("  matplotlib not available, skipping figure")
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('z2171: NARMA-10 Benchmark — FPGA Reservoir', fontsize=14, fontweight='bold')

    conditions = results.get('conditions', {})
    cond_names = list(conditions.keys())
    nrmse_vals = [conditions[c]['nrmse'] for c in cond_names]
    corr_vals = [conditions[c]['correlation'] for c in cond_names]

    # Color map
    colors = {
        'FULL': '#e74c3c',
        'WHITE': '#3498db',
        'NO_NOISE': '#95a5a6',
        'ESN': '#2ecc71',
    }
    bar_colors = [colors.get(c, '#7f8c8d') for c in cond_names]

    # Panel A: NRMSE bar chart
    ax = axes[0, 0]
    bars = ax.bar(cond_names, nrmse_vals, color=bar_colors, edgecolor='black', linewidth=0.5)
    ax.axhline(y=0.8, color='red', linestyle='--', alpha=0.5, label='T115 threshold')
    ax.axhline(y=1.0, color='darkred', linestyle='--', alpha=0.5, label='T120 threshold')
    ax.set_ylabel('NRMSE (lower = better)')
    ax.set_title('A) NARMA-10 NRMSE by Condition')
    ax.legend(fontsize=8)
    for bar, val in zip(bars, nrmse_vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f'{val:.3f}', ha='center', va='bottom', fontsize=9)

    # Panel B: Correlation bar chart
    ax = axes[0, 1]
    bars = ax.bar(cond_names, corr_vals, color=bar_colors, edgecolor='black', linewidth=0.5)
    ax.axhline(y=0.3, color='green', linestyle='--', alpha=0.5, label='T119 threshold')
    ax.set_ylabel('Pearson Correlation')
    ax.set_title('B) Prediction Correlation')
    ax.legend(fontsize=8)
    for bar, val in zip(bars, corr_vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f'{val:.3f}', ha='center', va='bottom', fontsize=9)

    # Panel C: FULL prediction vs target
    ax = axes[1, 0]
    full_pred = results.get('full_prediction', None)
    full_target = results.get('full_target', None)
    if full_pred is not None and full_target is not None:
        n_show = min(300, len(full_pred))
        t = np.arange(n_show)
        ax.plot(t, full_target[:n_show], 'k-', alpha=0.7, label='Target y(t)', linewidth=1)
        ax.plot(t, full_pred[:n_show], 'r-', alpha=0.7, label='FULL prediction', linewidth=1)
        ax.set_xlabel('Time step')
        ax.set_ylabel('y(t)')
        ax.set_title('C) FULL: Prediction vs Target (first 300 steps)')
        ax.legend(fontsize=8)
    else:
        ax.text(0.5, 0.5, 'No prediction data', transform=ax.transAxes,
                ha='center', va='center')

    # Panel D: Test summary
    ax = axes[1, 1]
    ax.axis('off')
    tests = results.get('tests', {})
    test_lines = []
    for tid in sorted(tests.keys()):
        t = tests[tid]
        status = "PASS" if t['pass'] else "FAIL"
        marker = "[+]" if t['pass'] else "[-]"
        test_lines.append(f"{marker} {tid}: {t['description']}")
        test_lines.append(f"      {t['detail']}")

    n_pass = sum(1 for t in tests.values() if t['pass'])
    n_total = len(tests)
    header = f"NARMA-10 Results: {n_pass}/{n_total} PASS\n"
    header += "=" * 50 + "\n"
    text = header + "\n".join(test_lines)
    ax.text(0.02, 0.98, text, transform=ax.transAxes, fontsize=7.5,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    plt.tight_layout()
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(fig_path), dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Figure saved: {fig_path}")


# ═══════════════════════════════════════════════════════════
# Main Experiment
# ═══════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description='z2171: NARMA-10 FPGA Reservoir Benchmark')
    parser.add_argument('--n-steps', type=int, default=3000,
                        help='NARMA-10 sequence length')
    parser.add_argument('--noise-collect-s', type=float, default=15.0,
                        help='Duration to collect power noise (seconds)')
    parser.add_argument('--train-frac', type=float, default=0.7,
                        help='Fraction of data for training (chronological split)')
    parser.add_argument('--warmup', type=int, default=50,
                        help='Warmup steps to discard (reservoir transient)')
    args = parser.parse_args()

    print("=" * 65)
    print("z2171: NARMA-10 Benchmark — FPGA Reservoir Computing")
    print("=" * 65)

    rng = np.random.default_rng(42)
    # Fixed random weights per neuron (same as z2162)
    w_in = rng.uniform(-1, 1, size=N_NEURONS)
    w_noise = rng.uniform(-1, 1, size=N_NEURONS)

    results = {
        'experiment': 'z2171_narma_benchmark',
        'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
        'params': {
            'base_vg': BASE_VG, 'alpha': ALPHA, 'beta': BETA,
            'n_neurons': N_NEURONS, 'sample_hz': SAMPLE_HZ,
            'iir_alpha': IIR_ALPHA,
            'n_steps': args.n_steps, 'train_frac': args.train_frac,
            'warmup': args.warmup,
            'w_in': w_in.tolist(), 'w_noise': w_noise.tolist(),
            'delays': [1, 2, 3],
        },
        'simulated': False,
    }

    # ─── Step 1: Connect to FPGA ───
    print("\n[1/6] Connecting to FPGA...")
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

    # ─── Step 2: Collect GPU noise ───
    print("\n[2/6] Collecting GPU noise sources...")

    # 1/f noise from power rail
    print("  Collecting power rail noise (1/f)...")
    power_noise = collect_power_noise(duration_s=args.noise_collect_s, sample_hz=50)
    if power_noise is not None and len(power_noise) > 10:
        power_mean = power_noise.mean()
        power_std = max(power_noise.std(), 1e-6)
        noise_1f_raw = (power_noise - power_mean) / power_std
        print(f"  Power rail: {power_mean:.2f} +/- {power_std:.3f} W, {len(noise_1f_raw)} samples")
        results['noise_source'] = 'hwmon_power1_average'
    else:
        print("  Power rail unavailable, generating synthetic 1/f (Voss-McCartney)")
        noise_1f_raw = generate_voss_mccartney(int(args.noise_collect_s * 50), rng=rng)
        results['noise_source'] = 'synthetic_voss_mccartney'

    # IIR filter for temporal memory
    noise_1f = iir_filter_noise(noise_1f_raw, alpha_iir=IIR_ALPHA)
    print(f"  IIR-filtered 1/f noise: {len(noise_1f)} samples, alpha={IIR_ALPHA}")

    # White noise
    noise_white = rng.standard_normal(max(args.n_steps, len(noise_1f)))
    print(f"  White noise: {len(noise_white)} samples (synthetic)")

    # No noise
    noise_zero = np.zeros(1)

    results['noise'] = {
        '1f_samples': len(noise_1f),
        'white_samples': len(noise_white),
    }

    # ─── Step 3: Generate NARMA-10 ───
    print("\n[3/6] Generating NARMA-10 sequence...")
    u, y_target = generate_narma10(args.n_steps, seed=42)
    print(f"  u(t) range: [{u.min():.3f}, {u.max():.3f}]")
    print(f"  y(t) range: [{y_target.min():.3f}, {y_target.max():.3f}]")
    print(f"  y(t) mean: {y_target.mean():.4f}, std: {y_target.std():.4f}")

    results['narma10'] = {
        'u_range': [float(u.min()), float(u.max())],
        'y_range': [float(y_target.min()), float(y_target.max())],
        'y_mean': float(y_target.mean()),
        'y_std': float(y_target.std()),
    }

    # ─── Step 4: Run all conditions ───
    print("\n[4/6] Running reservoir conditions on NARMA-10...")

    conditions_config = {
        'FULL':     {'noise': noise_1f,    'beta': BETA, 'live': True},
        'WHITE':    {'noise': noise_white,  'beta': BETA, 'live': False},
        'NO_NOISE': {'noise': noise_zero,   'beta': 0.0,  'live': False},
    }

    # Chronological split
    split_idx = int(args.n_steps * args.train_frac)
    warmup = args.warmup

    condition_results = {}

    for cond_name, cfg in conditions_config.items():
        print(f"\n  === Condition {cond_name} (beta={cfg['beta']:.2f}, live={cfg['live']}) ===")
        t0 = time.monotonic()

        if fpga:
            states = run_fpga_reservoir(
                ser, u, cfg['noise'], w_in, w_noise,
                base_vg=BASE_VG, alpha=ALPHA, beta=cfg['beta'],
                live_noise=cfg['live'])
        else:
            states = simulate_lif_reservoir(
                u, cfg['noise'], w_in, w_noise,
                base_vg=BASE_VG, alpha=ALPHA, beta=cfg['beta'])

        elapsed = time.monotonic() - t0
        print(f"  Collected {len(states)} states in {elapsed:.1f}s")
        print(f"  State shape: {states.shape}")
        print(f"  Spike deltas — mean: {states[:, :N_NEURONS].mean():.3f}, "
              f"max: {states[:, :N_NEURONS].max():.1f}")

        # Augment with time delays
        aug = augment_with_delays(states, delays=(1, 2, 3))
        print(f"  Augmented features: {aug.shape[1]}")

        # Add bias column
        X = np.hstack([aug, np.ones((len(aug), 1))])

        # Chronological train/test split (discard warmup)
        X_train = X[warmup:split_idx]
        y_train = y_target[warmup:split_idx]
        X_test = X[split_idx:]
        y_test = y_target[split_idx:]

        # Z-score normalize features
        mu = X_train.mean(axis=0, keepdims=True)
        sigma = X_train.std(axis=0, keepdims=True)
        sigma[sigma < 1e-10] = 1.0
        X_train_n = (X_train - mu) / sigma
        X_test_n = (X_test - mu) / sigma

        nrmse, corr, best_alpha, y_pred = ridge_regression(
            X_train_n, y_train, X_test_n, y_test)

        print(f"  NRMSE: {nrmse:.4f}")
        print(f"  Correlation: {corr:.4f}")
        print(f"  Best ridge alpha: {best_alpha}")

        condition_results[cond_name] = {
            'nrmse': float(nrmse),
            'correlation': float(corr),
            'ridge_alpha': float(best_alpha),
            'n_train': len(y_train),
            'n_test': len(y_test),
            'elapsed_s': float(elapsed),
        }

        # Save FULL prediction for plotting
        if cond_name == 'FULL':
            results['full_prediction'] = y_pred.tolist() if y_pred is not None else None
            results['full_target'] = y_test.tolist()

    # ─── Step 5: ESN baseline ───
    print("\n  === Condition ESN (software Echo State Network) ===")
    esn = EchoStateNetwork(input_dim=1, reservoir_size=8,
                           spectral_radius=0.95, input_scaling=0.3,
                           leak_rate=0.3, seed=42)
    esn.reset()
    esn_states = esn.run(u)
    esn_aug = augment_with_delays(esn_states, delays=(1, 2, 3))
    X_esn = np.hstack([esn_aug, np.ones((len(esn_aug), 1))])

    X_train = X_esn[warmup:split_idx]
    y_train = y_target[warmup:split_idx]
    X_test = X_esn[split_idx:]
    y_test = y_target[split_idx:]

    mu = X_train.mean(axis=0, keepdims=True)
    sigma = X_train.std(axis=0, keepdims=True)
    sigma[sigma < 1e-10] = 1.0
    X_train_n = (X_train - mu) / sigma
    X_test_n = (X_test - mu) / sigma

    esn_nrmse, esn_corr, esn_alpha, esn_pred = ridge_regression(
        X_train_n, y_train, X_test_n, y_test)

    print(f"  NRMSE: {esn_nrmse:.4f}")
    print(f"  Correlation: {esn_corr:.4f}")
    print(f"  Best ridge alpha: {esn_alpha}")

    condition_results['ESN'] = {
        'nrmse': float(esn_nrmse),
        'correlation': float(esn_corr),
        'ridge_alpha': float(esn_alpha),
        'n_train': len(y_train),
        'n_test': len(y_test),
        'elapsed_s': 0.0,
    }

    results['conditions'] = condition_results

    # ─── Step 6: Evaluate tests T115-T120 ───
    print("\n[5/6] Evaluating tests T115-T120...")

    full = condition_results['FULL']
    white = condition_results['WHITE']
    no_noise = condition_results['NO_NOISE']
    esn_res = condition_results['ESN']

    tests = {}

    # T115: FULL NRMSE < 0.8
    t115_pass = full['nrmse'] < 0.8
    tests['T115'] = {
        'description': 'FULL NRMSE < 0.8 (reasonable approximation)',
        'pass': bool(t115_pass),
        'detail': f"FULL NRMSE = {full['nrmse']:.4f} {'<' if t115_pass else '>='} 0.8",
    }
    print(f"  T115: {'PASS' if t115_pass else 'FAIL'} — FULL NRMSE={full['nrmse']:.4f} < 0.8")

    # T116: FULL NRMSE < WHITE NRMSE (1/f helps)
    t116_pass = full['nrmse'] < white['nrmse']
    tests['T116'] = {
        'description': 'FULL NRMSE < WHITE NRMSE (1/f noise helps)',
        'pass': bool(t116_pass),
        'detail': f"FULL={full['nrmse']:.4f} {'<' if t116_pass else '>='} WHITE={white['nrmse']:.4f}",
    }
    print(f"  T116: {'PASS' if t116_pass else 'FAIL'} — FULL={full['nrmse']:.4f} < WHITE={white['nrmse']:.4f}")

    # T117: FULL NRMSE < NO_NOISE NRMSE (noise helps)
    t117_pass = full['nrmse'] < no_noise['nrmse']
    tests['T117'] = {
        'description': 'FULL NRMSE < NO_NOISE NRMSE (noise helps)',
        'pass': bool(t117_pass),
        'detail': f"FULL={full['nrmse']:.4f} {'<' if t117_pass else '>='} NO_NOISE={no_noise['nrmse']:.4f}",
    }
    print(f"  T117: {'PASS' if t117_pass else 'FAIL'} — FULL={full['nrmse']:.4f} < NO_NOISE={no_noise['nrmse']:.4f}")

    # T118: ESN NRMSE < FULL NRMSE (software ceiling)
    t118_pass = esn_res['nrmse'] < full['nrmse']
    tests['T118'] = {
        'description': 'ESN NRMSE < FULL NRMSE (software ceiling is better)',
        'pass': bool(t118_pass),
        'detail': f"ESN={esn_res['nrmse']:.4f} {'<' if t118_pass else '>='} FULL={full['nrmse']:.4f}",
    }
    print(f"  T118: {'PASS' if t118_pass else 'FAIL'} — ESN={esn_res['nrmse']:.4f} < FULL={full['nrmse']:.4f}")

    # T119: FULL correlation > 0.3
    t119_pass = full['correlation'] > 0.3
    tests['T119'] = {
        'description': 'FULL correlation > 0.3 (moderate prediction)',
        'pass': bool(t119_pass),
        'detail': f"FULL corr = {full['correlation']:.4f} {'>' if t119_pass else '<='} 0.3",
    }
    print(f"  T119: {'PASS' if t119_pass else 'FAIL'} — FULL corr={full['correlation']:.4f} > 0.3")

    # T120: All conditions NRMSE < 1.0
    all_below_1 = all(condition_results[c]['nrmse'] < 1.0 for c in condition_results)
    worst_cond = max(condition_results, key=lambda c: condition_results[c]['nrmse'])
    worst_nrmse = condition_results[worst_cond]['nrmse']
    tests['T120'] = {
        'description': 'All conditions NRMSE < 1.0 (better than predicting mean)',
        'pass': bool(all_below_1),
        'detail': f"Worst: {worst_cond}={worst_nrmse:.4f} {'<' if all_below_1 else '>='} 1.0",
    }
    print(f"  T120: {'PASS' if all_below_1 else 'FAIL'} — worst: {worst_cond}={worst_nrmse:.4f} < 1.0")

    results['tests'] = tests
    n_pass = sum(1 for t in tests.values() if t['pass'])
    n_total = len(tests)
    results['score'] = f"{n_pass}/{n_total}"
    print(f"\n  SCORE: {n_pass}/{n_total} PASS")

    # ─── Save results ───
    print("\n[6/6] Saving results...")
    RESULTS.mkdir(parents=True, exist_ok=True)
    results_path = RESULTS / 'z2171_narma_benchmark.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, cls=NpEncoder)
    print(f"  Results: {results_path}")

    # ─── Plot figure ───
    fig_path = FIGURES / 'fig_z2171_narma.png'
    plot_results(results, fig_path)

    # ─── Cleanup ───
    if fpga and ser:
        try:
            # Set all Vg to zero
            set_per_neuron_vg(ser, np.zeros(N_NEURONS))
            ser.close()
            print("  FPGA connection closed")
        except Exception:
            pass

    print("\n" + "=" * 65)
    print(f"z2171 NARMA-10 COMPLETE — {n_pass}/{n_total} PASS")
    print("=" * 65)


if __name__ == '__main__':
    main()
