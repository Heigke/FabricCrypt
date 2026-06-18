#!/usr/bin/env python3
"""z2187_information_integration.py — Information Integration (Φ) in GPU-FPGA System

Measures Integrated Information Theory's Φ in the 8-neuron FPGA reservoir driven
by GPU noise. Directly tests IIT predictions: if the system shows non-trivial Φ,
it has a property IIT claims is necessary for consciousness.

4 conditions:
  FULL     — GPU 1/f noise → FPGA neurons
  WHITE    — White noise → FPGA neurons
  NO_NOISE — Deterministic FPGA (beta=0)
  SHUFFLED — FULL but temporally shuffled noise (destroys temporal structure)

Measures:
  - Φ (Practical): MI between bipartition halves at lag 1, min over 35 bipartitions
  - Transfer Entropy: TE(X→Y) summed across all neuron pairs
  - Coalition Entropy: H(full system) - Σ H(individual neurons)

Tests T213-T218:
  T213: Φ(FULL) > Φ(NO_NOISE) — noise creates information integration
  T214: Φ(FULL) > Φ(WHITE) — 1/f noise creates MORE integration than white
  T215: Φ(FULL) > Φ(SHUFFLED) — temporal structure matters for integration
  T216: Mean TE(FULL) > Mean TE(WHITE) — more directed information flow
  T217: Coalition entropy(FULL) > 0 — synergy exceeds redundancy
  T218: Φ(FULL) > 0.1 bits — non-trivial information integration

Hardware: AMD gfx1151 GPU + Arty A7 FPGA on /dev/ttyUSB*
"""

import os, sys, json, time, struct, argparse
import numpy as np
from pathlib import Path
from itertools import combinations

os.environ.setdefault('HSA_OVERRIDE_GFX_VERSION', '11.0.0')

BASE = Path(__file__).resolve().parent.parent
RESULTS = BASE / 'results'
FIGURES = BASE / 'figures'

# ─── FPGA Protocol ───
SYNC = 0x55
CMD_SET_VG = 0x01
CMD_READ_TELEM = 0x02
CMD_SET_KILL = 0x03

HWMON_POWER = "/sys/class/hwmon/hwmon7/power1_average"

# ─── Reservoir Parameters (defaults) ───
N_NEURONS = 8
SAMPLE_HZ = 20


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


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


def crc8(data):
    crc = 0
    for b in data:
        crc ^= b
        for _ in range(8):
            if crc & 0x80:
                crc = (crc << 1) ^ 0x07
            else:
                crc <<= 1
            crc &= 0xFF
    return crc


# ═══════════════════════════════════════════════════════════
# Noise Sources
# ═══════════════════════════════════════════════════════════

def read_hwmon_power():
    """Read hwmon power1_average (uW -> W)."""
    try:
        return int(open(HWMON_POWER).read().strip()) / 1e6
    except Exception:
        return None


def collect_power_noise(duration_s=60, sample_hz=50):
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
    """Apply IIR low-pass: y[t] = alpha*y[t-1] + (1-alpha)*x[t]."""
    filtered = np.zeros(len(noise_samples))
    filtered[0] = noise_samples[0]
    for t in range(1, len(noise_samples)):
        filtered[t] = alpha_iir * filtered[t - 1] + (1 - alpha_iir) * noise_samples[t]
    std = max(np.std(filtered), 1e-6)
    return filtered / std


class VossMcCartneyFilter:
    """10-octave Voss-McCartney 1/f noise generator."""

    def __init__(self, n_octaves=10):
        self.n_octaves = n_octaves
        self.values = np.zeros(n_octaves)
        self.counters = np.zeros(n_octaves, dtype=int)
        self.step = 0

    def process(self, white_sample):
        x = (white_sample / 127.5) - 1.0
        for k in range(self.n_octaves):
            period = 1 << k
            if self.step % period == 0:
                self.values[k] = x
        self.step += 1
        total = np.sum(self.values) / self.n_octaves
        return np.clip(total, -1.0, 1.0)


# ═══════════════════════════════════════════════════════════
# Information-Theoretic Measures
# ═══════════════════════════════════════════════════════════

def discretize_states(data, n_bins=4):
    """Discretize continuous data into quartile bins per column.

    Args:
        data: (T, N) array of continuous values
        n_bins: number of bins (default 4 = quartiles)

    Returns:
        (T, N) array of integer bin indices
    """
    T, N = data.shape
    discrete = np.zeros((T, N), dtype=int)
    for col in range(N):
        x = data[:, col]
        # Use percentile-based bins for robustness
        edges = np.percentile(x, np.linspace(0, 100, n_bins + 1))
        # Make edges unique
        edges = np.unique(edges)
        if len(edges) <= 1:
            discrete[:, col] = 0
        else:
            discrete[:, col] = np.clip(np.digitize(x, edges[1:-1]), 0, n_bins - 1)
    return discrete


def joint_histogram(x, y, n_bins=4):
    """Compute joint probability distribution of two discrete sequences."""
    p_xy = np.zeros((n_bins, n_bins))
    for xi, yi in zip(x, y):
        p_xy[xi, yi] += 1
    total = p_xy.sum()
    if total > 0:
        p_xy /= total
    return p_xy


def entropy(p):
    """Shannon entropy of a probability distribution (bits)."""
    p = p[p > 0]
    return -np.sum(p * np.log2(p))


def mutual_information_plugin(x, y, n_bins=4):
    """Plugin (maximum likelihood) mutual information estimator (bits).

    Uses Panzeri-Treves bias correction:
        bias ≈ (R_xy - R_x - R_y + 1) / (2 * N * ln(2))
    where R is the number of bins with non-zero probability.
    """
    N_samples = len(x)
    p_xy = joint_histogram(x, y, n_bins)
    p_x = p_xy.sum(axis=1)
    p_y = p_xy.sum(axis=0)

    h_x = entropy(p_x)
    h_y = entropy(p_y)
    h_xy = entropy(p_xy.ravel())

    mi_plugin = h_x + h_y - h_xy

    # Panzeri-Treves bias correction
    r_xy = np.sum(p_xy > 0)
    r_x = np.sum(p_x > 0)
    r_y = np.sum(p_y > 0)
    bias = (r_xy - r_x - r_y + 1) / (2 * N_samples * np.log(2))

    mi_corrected = max(0.0, mi_plugin - bias)
    return float(mi_corrected)


def compute_phi_practical(discrete_states, n_bins=4):
    """Compute practical Φ: minimum MI across all 4+4 bipartitions at lag 1.

    For 8 neurons, C(8,4)/2 = 35 unique bipartitions.
    For each bipartition (A, B):
        - Take states at time t (past) and t+1 (future)
        - Encode each half's state as a single integer (base n_bins)
        - Compute MI(A_future ; B_future | past) ≈ MI(A_{t+1} ; B_{t+1})
          using the joint state at lag 1

    Φ = min over bipartitions of MI(A_future ; B_future)
    """
    T, N = discrete_states.shape
    assert N == 8, f"Expected 8 neurons, got {N}"

    if T < 10:
        return 0.0, []

    # Generate all unique 4+4 bipartitions
    all_indices = list(range(N))
    bipartitions = []
    seen = set()
    for combo in combinations(all_indices, 4):
        complement = tuple(sorted(set(all_indices) - set(combo)))
        key = (min(combo, complement), max(combo, complement))
        if key not in seen:
            seen.add(key)
            bipartitions.append((list(combo), list(complement)))

    assert len(bipartitions) == 35, f"Expected 35 bipartitions, got {len(bipartitions)}"

    # Encode each half's state as a single integer at each timestep
    def encode_half(states_half, n_bins):
        """Encode multi-column discrete state into single integer per row."""
        T, K = states_half.shape
        encoded = np.zeros(T, dtype=int)
        for k in range(K):
            encoded += states_half[:, k] * (n_bins ** k)
        return encoded

    phi_values = []
    for part_a, part_b in bipartitions:
        # Future states (t+1)
        a_future = encode_half(discrete_states[1:, part_a], n_bins)
        b_future = encode_half(discrete_states[1:, part_b], n_bins)

        # MI between A_future and B_future
        # Use larger n_bins for joint encoding: n_bins^4 possible states per half
        n_joint_bins = n_bins ** 4
        mi = mutual_information_plugin(a_future, b_future, n_bins=n_joint_bins)
        phi_values.append(mi)

    phi = min(phi_values) if phi_values else 0.0
    return float(phi), phi_values


def compute_transfer_entropy(x, y, k=3, n_bins=4):
    """Transfer entropy TE(X→Y) with history length k.

    TE(X→Y) = H(Y_future | Y_past) - H(Y_future | Y_past, X_past)
            = H(Y_future, Y_past) - H(Y_past) - H(Y_future, Y_past, X_past) + H(Y_past, X_past)

    Uses plugin estimator with bias correction.
    """
    T = len(x)
    if T < k + 2:
        return 0.0

    # Build history vectors
    y_future = y[k:]
    y_past = np.zeros((T - k, k), dtype=int)
    x_past = np.zeros((T - k, k), dtype=int)
    for lag in range(k):
        y_past[:, lag] = y[k - 1 - lag:T - 1 - lag]
        x_past[:, lag] = x[k - 1 - lag:T - 1 - lag]

    # Encode past vectors as single integers
    def encode_vector(v, n_bins):
        encoded = np.zeros(len(v), dtype=int)
        for col in range(v.shape[1]):
            encoded += v[:, col] * (n_bins ** col)
        return encoded

    y_past_enc = encode_vector(y_past, n_bins)
    x_past_enc = encode_vector(x_past, n_bins)

    # Joint encoding of y_past and x_past
    n_ypast_states = n_bins ** k
    yx_past_enc = y_past_enc + x_past_enc * n_ypast_states

    N = len(y_future)
    n_fut = n_bins
    n_yp = n_bins ** k
    n_yxp = n_yp * (n_bins ** k)

    # H(Y_future, Y_past)
    p_fy = np.zeros(n_fut * n_yp)
    for i in range(N):
        idx = y_future[i] + y_past_enc[i] * n_fut
        if idx < len(p_fy):
            p_fy[idx] += 1
    p_fy /= max(N, 1)
    h_fy = entropy(p_fy)

    # H(Y_past)
    p_yp = np.bincount(y_past_enc, minlength=n_yp).astype(float)
    p_yp /= max(N, 1)
    h_yp = entropy(p_yp)

    # H(Y_future, Y_past, X_past)
    p_fyxp = np.zeros(n_fut * n_yxp)
    for i in range(N):
        idx = y_future[i] + yx_past_enc[i] * n_fut
        if idx < len(p_fyxp):
            p_fyxp[idx] += 1
    p_fyxp /= max(N, 1)
    h_fyxp = entropy(p_fyxp)

    # H(Y_past, X_past)
    p_yxp = np.bincount(yx_past_enc, minlength=n_yxp).astype(float)
    p_yxp /= max(N, 1)
    h_yxp = entropy(p_yxp)

    # TE = H(Y_future, Y_past) - H(Y_past) - H(Y_future, Y_past, X_past) + H(Y_past, X_past)
    te = h_fy - h_yp - h_fyxp + h_yxp

    # Bias correction (approximate)
    bias = max(0, (n_fut * n_yxp - n_fut * n_yp) / (2 * N * np.log(2)))
    te_corrected = max(0.0, te - bias)

    return float(te_corrected)


def compute_coalition_entropy(discrete_states, n_bins=4):
    """Coalition entropy: H(X1,...,X8) - Σ H(Xi).

    Positive = synergistic (system encodes more than sum of parts).
    Negative = redundant.
    """
    T, N = discrete_states.shape

    # Joint entropy H(X1,...,X8): encode full state as single integer
    joint_enc = np.zeros(T, dtype=int)
    for col in range(N):
        joint_enc += discrete_states[:, col] * (n_bins ** col)

    n_joint = n_bins ** N
    p_joint = np.bincount(joint_enc, minlength=n_joint).astype(float)
    p_joint /= max(T, 1)
    h_joint = entropy(p_joint)

    # Sum of individual entropies
    h_sum = 0.0
    for col in range(N):
        p_i = np.bincount(discrete_states[:, col], minlength=n_bins).astype(float)
        p_i /= max(T, 1)
        h_sum += entropy(p_i)

    # Coalition entropy = H_joint - Σ H_i
    # Positive means synergistic: joint state has MORE entropy than expected
    # from independent parts. But we want the OPPOSITE sign for IIT:
    # synergy means parts CONSTRAIN each other, so we measure:
    # Σ H_i - H_joint = total correlation (always >= 0)
    # But per the spec: "H(full system) - sum(H(individual neurons))"
    coalition = h_joint - h_sum

    return float(coalition)


# ═══════════════════════════════════════════════════════════
# FPGA Reservoir Core
# ═══════════════════════════════════════════════════════════

def run_fpga_trial(ser, n_steps, input_signal, noise_samples, w_in, w_noise,
                   base_vg, alpha, beta, sample_hz, live_noise=False):
    """Drive FPGA neurons with input+noise and collect spike/vmem states.

    Returns: (n_steps, 16) array — 8 delta_spikes + 8 vmem.
    """
    interval = 1.0 / sample_hz
    states = np.zeros((n_steps, N_NEURONS * 2))
    prev_counts = None
    power_mean = 11.0

    for t in range(n_steps):
        if live_noise:
            p = read_hwmon_power()
            noise_val = (p - power_mean) / 2.0 if p else 0.0
        elif beta > 0 and len(noise_samples) > 0:
            noise_val = noise_samples[t % len(noise_samples)]
        else:
            noise_val = 0.0

        # Drive signal: low-frequency sine (0.5 Hz) to activate reservoir
        vg_values = np.full(N_NEURONS, base_vg)
        vg_values += alpha * input_signal[t] * w_in
        if beta > 0:
            vg_values += beta * noise_val * w_noise
        vg_values = np.clip(vg_values, 0.05, 0.95)

        set_per_neuron_vg(ser, vg_values)
        time.sleep(interval * 0.3)

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


def simulate_lif_reservoir(n_steps, input_signal, noise_samples, w_in, w_noise,
                           base_vg, alpha, beta, sample_hz):
    """Software LIF simulation fallback when FPGA is not connected."""
    states = np.zeros((n_steps, N_NEURONS * 2))
    v_rest = 0.0
    v_thresh = 1.0
    tau_m = 0.02
    dt = 1.0 / sample_hz
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
        states[t, N_NEURONS:N_NEURONS * 2] = vmem.copy()

    return states


# ═══════════════════════════════════════════════════════════
# Plotting
# ═══════════════════════════════════════════════════════════

def make_figure(results, outpath):
    """2x2 figure: Φ bar chart, TE heatmap, coalition entropy, bipartition distribution."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("  matplotlib not available, skipping figure")
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('z2187: Information Integration (Φ) in GPU-FPGA System', fontsize=14)

    conditions = ['FULL', 'WHITE', 'NO_NOISE', 'SHUFFLED']
    colors = ['#2196F3', '#FF9800', '#9E9E9E', '#9C27B0']

    # (a) Φ by condition
    ax = axes[0, 0]
    phi_vals = [results['conditions'].get(c, {}).get('phi', 0) for c in conditions]
    bars = ax.bar(conditions, phi_vals, color=colors, edgecolor='black', linewidth=0.5)
    ax.set_ylabel('Φ (bits)')
    ax.set_title('(a) Integrated Information by Condition')
    ax.axhline(y=0.1, color='red', linestyle='--', alpha=0.5, label='T218 threshold')
    ax.legend(fontsize=8)
    for bar, val in zip(bars, phi_vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f'{val:.3f}', ha='center', va='bottom', fontsize=9)

    # (b) TE heatmap for FULL condition
    ax = axes[0, 1]
    te_matrix = np.array(results['conditions'].get('FULL', {}).get('te_matrix', np.zeros((8, 8))))
    im = ax.imshow(te_matrix, cmap='hot', aspect='equal')
    ax.set_xlabel('Target neuron')
    ax.set_ylabel('Source neuron')
    ax.set_title('(b) Transfer Entropy (FULL condition)')
    plt.colorbar(im, ax=ax, label='TE (bits)')

    # (c) Coalition entropy comparison
    ax = axes[1, 0]
    ce_vals = [results['conditions'].get(c, {}).get('coalition_entropy', 0) for c in conditions]
    bars = ax.bar(conditions, ce_vals, color=colors, edgecolor='black', linewidth=0.5)
    ax.set_ylabel('Coalition Entropy (bits)')
    ax.set_title('(c) Coalition Entropy by Condition')
    ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    for bar, val in zip(bars, ce_vals):
        yoff = 0.01 if val >= 0 else -0.03
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + yoff,
                f'{val:.3f}', ha='center', va='bottom', fontsize=9)

    # (d) Bipartition distribution for FULL
    ax = axes[1, 1]
    phi_dist = results['conditions'].get('FULL', {}).get('phi_bipartitions', [])
    if phi_dist:
        ax.hist(phi_dist, bins=20, color='#2196F3', edgecolor='black', alpha=0.7)
        ax.axvline(x=min(phi_dist), color='red', linestyle='--',
                   label=f'Φ = min = {min(phi_dist):.4f}')
        ax.legend(fontsize=8)
    ax.set_xlabel('MI (bits)')
    ax.set_ylabel('Count')
    ax.set_title('(d) MI Distribution over 35 Bipartitions (FULL)')

    plt.tight_layout()
    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(outpath), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Figure saved: {outpath}")


# ═══════════════════════════════════════════════════════════
# Main Experiment
# ═══════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description='z2187: Information Integration (Φ)')
    parser.add_argument('--n-steps', type=int, default=1000, help='Timesteps per condition')
    parser.add_argument('--sample-hz', type=int, default=20, help='Sampling rate (Hz)')
    parser.add_argument('--base-vg', type=float, default=0.55, help='Base gate voltage')
    parser.add_argument('--alpha', type=float, default=0.15, help='Input coupling')
    parser.add_argument('--beta', type=float, default=0.10, help='Noise coupling')
    parser.add_argument('--noise-collect-s', type=float, default=60.0,
                        help='Duration to collect power noise (s)')
    args = parser.parse_args()

    n_steps = args.n_steps
    sample_hz = args.sample_hz
    base_vg = args.base_vg
    alpha = args.alpha
    beta = args.beta

    print("=" * 65)
    print("z2187: Information Integration (Φ) in GPU-FPGA System")
    print("=" * 65)
    print(f"  Steps: {n_steps}  Sample Hz: {sample_hz}")
    print(f"  Base Vg: {base_vg}  Alpha: {alpha}  Beta: {beta}")

    rng = np.random.default_rng(42)
    w_in = rng.uniform(-1, 1, size=N_NEURONS)
    w_noise = rng.uniform(-1, 1, size=N_NEURONS)

    results = {
        'experiment': 'z2187_information_integration',
        'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
        'params': {
            'base_vg': base_vg, 'alpha': alpha, 'beta': beta,
            'n_neurons': N_NEURONS, 'sample_hz': sample_hz,
            'n_steps': n_steps,
            'w_in': w_in.tolist(), 'w_noise': w_noise.tolist(),
        },
        'simulated': False,
        'conditions': {},
        'tests': {},
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
    print("  Collecting power rail noise (1/f)...")
    power_noise = collect_power_noise(duration_s=args.noise_collect_s, sample_hz=50)
    if power_noise is not None and len(power_noise) > 10:
        power_mean = power_noise.mean()
        power_std = max(power_noise.std(), 1e-6)
        noise_1f_raw = (power_noise - power_mean) / power_std
        print(f"  Power rail: {power_mean:.2f} +/- {power_std:.3f} W, {len(noise_1f_raw)} samples")
    else:
        print("  Power rail unavailable, generating synthetic 1/f")
        n_synth = int(args.noise_collect_s * 50)
        noise_1f_raw = np.zeros(n_synth)
        n_octaves = 8
        octaves = np.zeros(n_octaves)
        for i in range(n_synth):
            for j in range(n_octaves):
                if i % (1 << j) == 0:
                    octaves[j] = rng.standard_normal()
            noise_1f_raw[i] = octaves.sum()
        noise_1f_raw = (noise_1f_raw - noise_1f_raw.mean()) / max(noise_1f_raw.std(), 1e-6)

    noise_1f = iir_filter_noise(noise_1f_raw, alpha_iir=0.85)
    noise_white = rng.standard_normal(len(noise_1f))
    noise_shuffled = noise_1f.copy()
    rng.shuffle(noise_shuffled)  # Destroys temporal structure, preserves marginals
    noise_zero = np.zeros(n_steps)

    # ─── Step 3: Generate input signal ───
    print("\n[3/6] Generating input signal...")
    # Low-frequency sine wave at 0.5 Hz to drive the reservoir
    t_arr = np.arange(n_steps) / sample_hz
    input_signal = np.sin(2 * np.pi * 0.5 * t_arr) * 0.5

    conditions = {
        'FULL':     {'noise': noise_1f,       'beta': beta, 'label': 'GPU 1/f noise'},
        'WHITE':    {'noise': noise_white,    'beta': beta, 'label': 'White noise'},
        'NO_NOISE': {'noise': noise_zero,     'beta': 0.0,  'label': 'No noise'},
        'SHUFFLED': {'noise': noise_shuffled, 'beta': beta, 'label': 'Shuffled 1/f'},
    }

    # ─── Step 4: Run conditions ───
    print("\n[4/6] Running conditions...")
    condition_states = {}
    for cond_name, cond_cfg in conditions.items():
        print(f"\n  === {cond_name}: {cond_cfg['label']} ===")
        t0 = time.monotonic()

        if fpga:
            states = run_fpga_trial(
                ser, n_steps, input_signal, cond_cfg['noise'],
                w_in, w_noise, base_vg, alpha, cond_cfg['beta'],
                sample_hz, live_noise=(cond_name == 'FULL')
            )
        else:
            states = simulate_lif_reservoir(
                n_steps, input_signal, cond_cfg['noise'],
                w_in, w_noise, base_vg, alpha, cond_cfg['beta'],
                sample_hz
            )

        elapsed = time.monotonic() - t0
        print(f"  Collected {states.shape[0]} steps in {elapsed:.1f}s")

        # Use spike counts (first 8 columns) for information measures
        spike_states = states[:, :N_NEURONS]
        condition_states[cond_name] = spike_states

        # Print basic stats
        total_spikes = spike_states.sum()
        mean_rate = spike_states.mean() * sample_hz
        print(f"  Total spikes: {total_spikes:.0f}, Mean rate: {mean_rate:.2f} Hz")

    # ─── Step 5: Compute information measures ───
    print("\n[5/6] Computing information-theoretic measures...")
    n_bins = 4

    for cond_name, spike_states in condition_states.items():
        print(f"\n  --- {cond_name} ---")

        # Discretize states
        discrete = discretize_states(spike_states, n_bins=n_bins)

        # 5a. Practical Φ
        print("  Computing Φ (35 bipartitions)...")
        phi, phi_dist = compute_phi_practical(discrete, n_bins=n_bins)
        print(f"  Φ = {phi:.4f} bits")

        # 5b. Transfer Entropy (all pairs)
        print("  Computing Transfer Entropy...")
        te_matrix = np.zeros((N_NEURONS, N_NEURONS))
        for src in range(N_NEURONS):
            for tgt in range(N_NEURONS):
                if src != tgt:
                    te_matrix[src, tgt] = compute_transfer_entropy(
                        discrete[:, src], discrete[:, tgt], k=3, n_bins=n_bins
                    )
        mean_te = te_matrix.sum() / (N_NEURONS * (N_NEURONS - 1))
        print(f"  Mean TE = {mean_te:.4f} bits")

        # 5c. Coalition Entropy
        coalition = compute_coalition_entropy(discrete, n_bins=n_bins)
        print(f"  Coalition entropy = {coalition:.4f} bits")

        results['conditions'][cond_name] = {
            'phi': phi,
            'phi_bipartitions': phi_dist,
            'mean_te': mean_te,
            'te_matrix': te_matrix.tolist(),
            'coalition_entropy': coalition,
            'total_spikes': float(spike_states.sum()),
            'mean_spike_rate': float(spike_states.mean() * sample_hz),
        }

    # ─── Step 6: Evaluate tests ───
    print("\n[6/6] Evaluating tests T213-T218...")
    phi_full = results['conditions']['FULL']['phi']
    phi_white = results['conditions']['WHITE']['phi']
    phi_no_noise = results['conditions']['NO_NOISE']['phi']
    phi_shuffled = results['conditions']['SHUFFLED']['phi']
    te_full = results['conditions']['FULL']['mean_te']
    te_white = results['conditions']['WHITE']['mean_te']
    ce_full = results['conditions']['FULL']['coalition_entropy']

    tests = {}

    # T213: Φ(FULL) > Φ(NO_NOISE)
    t213_pass = phi_full > phi_no_noise
    tests['T213'] = {
        'name': 'Phi(FULL) > Phi(NO_NOISE)',
        'description': 'Noise creates information integration',
        'phi_full': phi_full, 'phi_no_noise': phi_no_noise,
        'pass': t213_pass,
    }
    print(f"  T213: Φ(FULL)={phi_full:.4f} > Φ(NO_NOISE)={phi_no_noise:.4f} → {'PASS' if t213_pass else 'FAIL'}")

    # T214: Φ(FULL) > Φ(WHITE)
    t214_pass = phi_full > phi_white
    tests['T214'] = {
        'name': 'Phi(FULL) > Phi(WHITE)',
        'description': '1/f noise creates more integration than white',
        'phi_full': phi_full, 'phi_white': phi_white,
        'pass': t214_pass,
    }
    print(f"  T214: Φ(FULL)={phi_full:.4f} > Φ(WHITE)={phi_white:.4f} → {'PASS' if t214_pass else 'FAIL'}")

    # T215: Φ(FULL) > Φ(SHUFFLED)
    t215_pass = phi_full > phi_shuffled
    tests['T215'] = {
        'name': 'Phi(FULL) > Phi(SHUFFLED)',
        'description': 'Temporal structure matters for integration',
        'phi_full': phi_full, 'phi_shuffled': phi_shuffled,
        'pass': t215_pass,
    }
    print(f"  T215: Φ(FULL)={phi_full:.4f} > Φ(SHUFFLED)={phi_shuffled:.4f} → {'PASS' if t215_pass else 'FAIL'}")

    # T216: Mean TE(FULL) > Mean TE(WHITE)
    t216_pass = te_full > te_white
    tests['T216'] = {
        'name': 'TE(FULL) > TE(WHITE)',
        'description': 'More directed information flow with 1/f',
        'te_full': te_full, 'te_white': te_white,
        'pass': t216_pass,
    }
    print(f"  T216: TE(FULL)={te_full:.4f} > TE(WHITE)={te_white:.4f} → {'PASS' if t216_pass else 'FAIL'}")

    # T217: Coalition entropy(FULL) > 0
    t217_pass = ce_full > 0
    tests['T217'] = {
        'name': 'Coalition entropy(FULL) > 0',
        'description': 'Synergy exceeds redundancy',
        'coalition_entropy': ce_full,
        'pass': t217_pass,
    }
    print(f"  T217: CE(FULL)={ce_full:.4f} > 0 → {'PASS' if t217_pass else 'FAIL'}")

    # T218: Φ(FULL) > 0.1 bits
    t218_pass = phi_full > 0.1
    tests['T218'] = {
        'name': 'Phi(FULL) > 0.1 bits',
        'description': 'Non-trivial information integration',
        'phi_full': phi_full,
        'pass': t218_pass,
    }
    print(f"  T218: Φ(FULL)={phi_full:.4f} > 0.1 → {'PASS' if t218_pass else 'FAIL'}")

    results['tests'] = tests
    n_pass = sum(1 for t in tests.values() if t['pass'])
    n_total = len(tests)
    results['summary'] = {
        'pass': n_pass,
        'total': n_total,
        'score': f'{n_pass}/{n_total}',
    }

    print(f"\n{'=' * 65}")
    print(f"  RESULT: {n_pass}/{n_total} PASS")
    print(f"{'=' * 65}")

    # ─── Save results ───
    RESULTS.mkdir(parents=True, exist_ok=True)
    out_json = RESULTS / 'z2187_information_integration.json'
    with open(out_json, 'w') as f:
        json.dump(results, f, indent=2, cls=NpEncoder)
    print(f"\n  Results saved: {out_json}")

    # ─── Generate figure ───
    fig_path = FIGURES / 'z2187_information_integration.png'
    make_figure(results, fig_path)

    # Cleanup
    if fpga and ser:
        ser.close()
        print("  FPGA serial closed")

    return results


if __name__ == '__main__':
    main()
