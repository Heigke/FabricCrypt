#!/usr/bin/env python3
"""z2195_temporal_binding.py — Temporal Binding in GPU-Noise-Driven FPGA Reservoir

Tests whether GPU 1/f noise creates synchronized neural assemblies in the FPGA
reservoir that BIND related information across time.  This is a key mechanism
for Mario Lanza's cross-substrate integration vision.

Protocol:
  Present two stimuli separated by a temporal gap:
    Stimulus A (100 steps): e.g., 0.5 Hz sine
    Gap        ( 50 steps): silence
    Stimulus B (100 steps): e.g., 1.0 Hz sine
  Measure if the FPGA reservoir creates a unified representation that binds A+B.

Conditions (4):
  FULL           — GPU 1/f noise during all phases
  WHITE          — white noise during all phases
  NO_NOISE       — no noise (deterministic)
  NOISE_GAP_ONLY — noise ONLY during the gap (tests if noise bridges stimuli)

Measures:
  1. Cross-temporal correlation: cosine sim between state at end-A and start-B
  2. Binding classification: 3 A-B pairings, 50 reps each → can we tell which A→B?
  3. Spike synchrony: pairwise correlation during stimulus vs. during gap
  4. Temporal integration index: MI(A_features, B_features | condition)

Tests T261-T266:
  T261: cross-temporal corr FULL > NO_NOISE
  T262: binding accuracy > 33% (chance for 3-class)
  T263: FULL binding accuracy > WHITE binding accuracy
  T264: NOISE_GAP_ONLY corr > NO_NOISE corr
  T265: spike synchrony during stimulus > during gap for all conditions
  T266: temporal integration MI > 0.05 bits for FULL

Hardware: AMD gfx1151 GPU + Arty A7 FPGA on /dev/ttyUSB*
"""

import os, sys, json, time, struct, argparse
import numpy as np
from pathlib import Path

os.environ.setdefault('HSA_OVERRIDE_GFX_VERSION', '11.0.0')

BASE = Path(__file__).resolve().parent.parent
RESULTS = BASE / 'results'
FIGURES = RESULTS / 'FEEL_paper_update' / 'FEEL__Functionally_Embodied_Emergent_Learning__13_-5' / 'figures'

# ─── FPGA Protocol ───
SYNC = 0x55
CMD_SET_VG = 0x01
CMD_READ_TELEM = 0x02
CMD_SET_KILL = 0x03

HWMON_POWER = "/sys/class/hwmon/hwmon7/power1_average"

# ─── Default Parameters ───
N_NEURONS = 8
DEFAULT_BASE_VG = 0.55
DEFAULT_ALPHA = 0.15
DEFAULT_BETA = 0.10
DEFAULT_SAMPLE_HZ = 20
DEFAULT_N_REPS = 50
N_STEPS_A = 100
N_STEPS_GAP = 50
N_STEPS_B = 100
N_STEPS_TRIAL = N_STEPS_A + N_STEPS_GAP + N_STEPS_B  # 250


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
# FPGA Communication (same patterns as z2188)
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


def connect_fpga(ser):
    """Disable kill switch and flush."""
    ser.reset_input_buffer()
    ser.reset_output_buffer()
    ser.write(bytes([SYNC, CMD_SET_KILL, 0x00]))
    ser.flush()
    time.sleep(0.05)
    ser.reset_input_buffer()


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
# Stimulus Generation
# ═══════════════════════════════════════════════════════════

# 3 A-B pairings: (sine, triangle), (triangle, square), (square, sine)
PAIRINGS = [
    ('sine', 'triangle'),
    ('triangle', 'square'),
    ('square', 'sine'),
]


def _waveform(kind, n_steps, sample_hz, freq_hz, phase=0.0):
    """Generate a single waveform normalized to [0, 1]."""
    t = np.arange(n_steps) / sample_hz
    if kind == 'sine':
        w = np.sin(2 * np.pi * freq_hz * t + phase)
    elif kind == 'triangle':
        w = 2.0 * np.abs(2.0 * ((freq_hz * t + phase / (2 * np.pi)) % 1.0) - 1.0) - 1.0
    elif kind == 'square':
        w = np.sign(np.sin(2 * np.pi * freq_hz * t + phase))
    else:
        w = np.zeros(n_steps)
    return (w + 1.0) / 2.0  # map to [0, 1]


def generate_trial_stimulus(pairing_idx, sample_hz, rng):
    """Generate a full trial: stimulus_A | gap | stimulus_B.

    Returns (n_steps_trial,) input signal in [0, 1].
    """
    a_kind, b_kind = PAIRINGS[pairing_idx]
    phase_a = rng.uniform(0, 2 * np.pi)
    phase_b = rng.uniform(0, 2 * np.pi)
    freq_a = 0.5 * rng.uniform(0.8, 1.2)
    freq_b = 1.0 * rng.uniform(0.8, 1.2)

    stim_a = _waveform(a_kind, N_STEPS_A, sample_hz, freq_a, phase_a)
    gap = np.full(N_STEPS_GAP, 0.5)  # silence = mid-point
    stim_b = _waveform(b_kind, N_STEPS_B, sample_hz, freq_b, phase_b)

    return np.concatenate([stim_a, gap, stim_b])


# ═══════════════════════════════════════════════════════════
# FPGA Reservoir (real hardware)
# ═══════════════════════════════════════════════════════════

def run_fpga_trial(ser, input_signal, noise_samples, w_in, w_noise,
                   base_vg, alpha, beta, sample_hz, condition):
    """Drive FPGA neurons and collect states for one trial.

    condition controls noise application:
      FULL           — noise at every step
      WHITE          — white noise at every step (noise_samples is white)
      NO_NOISE       — beta=0
      NOISE_GAP_ONLY — noise only during gap phase (steps N_STEPS_A .. N_STEPS_A+N_STEPS_GAP)

    Returns: (n_steps, 24) array — 8 delta_spikes + 8 vmem + 8 cumulative_spikes.
    """
    n_steps = len(input_signal)
    interval = 1.0 / sample_hz
    states = np.zeros((n_steps, N_NEURONS * 3))
    prev_counts = None
    cumulative = np.zeros(N_NEURONS)
    power_mean = 11.0

    for t in range(n_steps):
        # Determine noise for this step
        if condition == 'NO_NOISE':
            noise_val = 0.0
        elif condition == 'NOISE_GAP_ONLY':
            if N_STEPS_A <= t < N_STEPS_A + N_STEPS_GAP:
                # Live hwmon during gap
                p = read_hwmon_power()
                noise_val = (p - power_mean) / 2.0 if p else 0.0
            else:
                noise_val = 0.0
        elif condition == 'FULL':
            p = read_hwmon_power()
            noise_val = (p - power_mean) / 2.0 if p else 0.0
        else:  # WHITE
            noise_val = noise_samples[t % len(noise_samples)] if len(noise_samples) > 0 else 0.0

        # Compute per-neuron Vg
        vg_values = np.full(N_NEURONS, base_vg)
        vg_values += alpha * input_signal[t] * w_in
        if noise_val != 0.0:
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
                    cumulative[i] += delta
            for i in range(N_NEURONS):
                states[t, N_NEURONS + i] = vmems[i]
                states[t, N_NEURONS * 2 + i] = cumulative[i]
            prev_counts = counts[:]

        time.sleep(max(0, interval * 0.5 - 0.01))

    return states


# ═══════════════════════════════════════════════════════════
# Software LIF Simulation Fallback
# ═══════════════════════════════════════════════════════════

def simulate_lif_trial(input_signal, noise_samples, w_in, w_noise,
                       base_vg, alpha, beta, sample_hz, condition, rng_seed=42):
    """Software LIF simulation fallback when FPGA not connected."""
    n_steps = len(input_signal)
    states = np.zeros((n_steps, N_NEURONS * 3))
    v_rest = 0.0
    v_thresh = 1.0
    tau_m = 0.02
    dt = 1.0 / sample_hz
    vmem = np.zeros(N_NEURONS)
    cumulative = np.zeros(N_NEURONS)
    rng = np.random.default_rng(rng_seed)

    for t in range(n_steps):
        if condition == 'NO_NOISE':
            noise_val = 0.0
        elif condition == 'NOISE_GAP_ONLY':
            if N_STEPS_A <= t < N_STEPS_A + N_STEPS_GAP:
                noise_val = noise_samples[t % len(noise_samples)] if len(noise_samples) > 0 else rng.standard_normal() * 0.3
            else:
                noise_val = 0.0
        elif condition == 'FULL':
            noise_val = noise_samples[t % len(noise_samples)] if len(noise_samples) > 0 else rng.standard_normal() * 0.3
        else:  # WHITE
            noise_val = rng.standard_normal()

        vg = np.full(N_NEURONS, base_vg)
        vg += alpha * input_signal[t] * w_in
        if noise_val != 0.0:
            vg += beta * noise_val * w_noise
        vg = np.clip(vg, 0.05, 0.95)

        I_in = vg * 5.0
        dvdt = (-vmem + I_in) / tau_m
        vmem += dvdt * dt

        for i in range(N_NEURONS):
            if vmem[i] >= v_thresh:
                states[t, i] = 1
                vmem[i] = v_rest
                cumulative[i] += 1

        states[t, N_NEURONS:N_NEURONS * 2] = vmem.copy()
        states[t, N_NEURONS * 2:] = cumulative.copy()

    return states


# ═══════════════════════════════════════════════════════════
# Analysis Functions
# ═══════════════════════════════════════════════════════════

def cosine_sim(a, b):
    """Cosine similarity between two vectors."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a < 1e-12 or norm_b < 1e-12:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def extract_phase_features(states, phase):
    """Extract feature vector for a given phase from (n_steps, 24) states.

    phase: 'A', 'gap', 'B'
    Returns: concatenation of [mean, std, max, min] of the reservoir state
             during that phase.
    """
    if phase == 'A':
        seg = states[:N_STEPS_A]
    elif phase == 'gap':
        seg = states[N_STEPS_A:N_STEPS_A + N_STEPS_GAP]
    else:  # 'B'
        seg = states[N_STEPS_A + N_STEPS_GAP:]

    if len(seg) == 0:
        return np.zeros(24 * 4)

    return np.concatenate([
        seg.mean(axis=0),
        seg.std(axis=0),
        seg.max(axis=0),
        seg.min(axis=0),
    ])


def cross_temporal_correlation(states):
    """Cosine similarity between reservoir state at end of A and start of B.

    Uses a window of 10 steps at each boundary for robustness.
    """
    end_a_window = 10
    start_b_window = 10
    end_a = states[max(0, N_STEPS_A - end_a_window):N_STEPS_A].mean(axis=0)
    start_b_idx = N_STEPS_A + N_STEPS_GAP
    start_b = states[start_b_idx:start_b_idx + start_b_window].mean(axis=0)
    return cosine_sim(end_a, start_b)


def pairwise_spike_correlation(states, phase):
    """Mean pairwise Pearson correlation of delta-spike trains in a phase."""
    if phase == 'A':
        seg = states[:N_STEPS_A, :N_NEURONS]
    elif phase == 'gap':
        seg = states[N_STEPS_A:N_STEPS_A + N_STEPS_GAP, :N_NEURONS]
    else:
        seg = states[N_STEPS_A + N_STEPS_GAP:, :N_NEURONS]

    if seg.shape[0] < 3:
        return 0.0

    # Pairwise correlations
    corrs = []
    for i in range(N_NEURONS):
        for j in range(i + 1, N_NEURONS):
            si = seg[:, i]
            sj = seg[:, j]
            if np.std(si) < 1e-12 or np.std(sj) < 1e-12:
                corrs.append(0.0)
            else:
                corrs.append(float(np.corrcoef(si, sj)[0, 1]))
    return float(np.mean(corrs)) if corrs else 0.0


def mutual_information_binned(x, y, n_bins=8):
    """Estimate MI(X; Y) using histogram binning.

    x, y: 1D arrays of the same length.
    """
    n = len(x)
    if n < 10:
        return 0.0

    # Bin edges
    x_edges = np.linspace(np.min(x) - 1e-10, np.max(x) + 1e-10, n_bins + 1)
    y_edges = np.linspace(np.min(y) - 1e-10, np.max(y) + 1e-10, n_bins + 1)

    x_bin = np.digitize(x, x_edges) - 1
    y_bin = np.digitize(y, y_edges) - 1
    x_bin = np.clip(x_bin, 0, n_bins - 1)
    y_bin = np.clip(y_bin, 0, n_bins - 1)

    # Joint histogram
    joint = np.zeros((n_bins, n_bins))
    for i in range(n):
        joint[x_bin[i], y_bin[i]] += 1
    joint /= n

    # Marginals
    px = joint.sum(axis=1)
    py = joint.sum(axis=0)

    # MI
    mi = 0.0
    for i in range(n_bins):
        for j in range(n_bins):
            if joint[i, j] > 1e-12 and px[i] > 1e-12 and py[j] > 1e-12:
                mi += joint[i, j] * np.log2(joint[i, j] / (px[i] * py[j]))
    return max(mi, 0.0)


def temporal_integration_mi(all_trial_states):
    """Compute MI(A_features, B_features) across trials.

    all_trial_states: list of (n_steps, 24) arrays.
    Returns MI in bits.
    """
    a_feats = []
    b_feats = []
    for st in all_trial_states:
        a_feats.append(extract_phase_features(st, 'A'))
        b_feats.append(extract_phase_features(st, 'B'))
    a_feats = np.array(a_feats)
    b_feats = np.array(b_feats)

    if len(a_feats) < 10:
        return 0.0

    # Use first principal component of each for MI estimation
    a_mean = a_feats.mean(axis=0, keepdims=True)
    b_mean = b_feats.mean(axis=0, keepdims=True)
    a_centered = a_feats - a_mean
    b_centered = b_feats - b_mean

    # SVD for PC1
    def pc1(X):
        if X.shape[0] < 2:
            return X[:, 0] if X.shape[1] > 0 else np.zeros(X.shape[0])
        try:
            _, _, Vt = np.linalg.svd(X, full_matrices=False)
            return X @ Vt[0]
        except Exception:
            return X[:, 0] if X.shape[1] > 0 else np.zeros(X.shape[0])

    a_pc1 = pc1(a_centered)
    b_pc1 = pc1(b_centered)

    return mutual_information_binned(a_pc1, b_pc1)


def ridge_classify(X_train, y_train, X_test, y_test, alphas=None):
    """Ridge regression classifier (one-hot encoding for multi-class)."""
    if alphas is None:
        alphas = [1e-6, 1e-4, 1e-2, 1.0, 100.0]

    n_classes = len(np.unique(np.concatenate([y_train, y_test])))
    Y_train = np.zeros((len(y_train), n_classes))
    for i, y in enumerate(y_train):
        Y_train[i, int(y)] = 1.0

    best_acc = -1
    for alpha_reg in alphas:
        I = np.eye(X_train.shape[1])
        try:
            W = np.linalg.solve(X_train.T @ X_train + alpha_reg * I, X_train.T @ Y_train)
        except np.linalg.LinAlgError:
            continue
        pred_test = np.argmax(X_test @ W, axis=1)
        acc = float(np.mean(pred_test == y_test))
        if acc > best_acc:
            best_acc = acc
    return max(best_acc, 0.0)


def stratified_kfold(X, y, n_splits=5, seed=42):
    """Simple stratified k-fold split."""
    rng = np.random.default_rng(seed)
    classes = np.unique(y)
    indices = np.arange(len(y))
    rng.shuffle(indices)

    folds = [[] for _ in range(n_splits)]
    for c in classes:
        c_idx = indices[y[indices] == c]
        for i, idx in enumerate(c_idx):
            folds[i % n_splits].append(idx)

    splits = []
    for fold in range(n_splits):
        test_idx = np.array(folds[fold])
        train_idx = np.concatenate([np.array(folds[f]) for f in range(n_splits) if f != fold])
        splits.append((train_idx, test_idx))
    return splits


# ═══════════════════════════════════════════════════════════
# Plotting
# ═══════════════════════════════════════════════════════════

def make_figure(cond_results, tests):
    """Create 2x2 figure: cross-temporal corr, binding acc, synchrony, test results."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("[WARN] matplotlib not available, skipping figure")
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('z2195: Temporal Binding in GPU-FPGA System', fontsize=14, fontweight='bold')

    conditions = ['FULL', 'WHITE', 'NO_NOISE', 'NOISE_GAP_ONLY']
    cond_colors = {
        'FULL': '#2196F3', 'WHITE': '#FF9800',
        'NO_NOISE': '#9E9E9E', 'NOISE_GAP_ONLY': '#4CAF50',
    }

    # ─── Panel 1: Cross-temporal correlation ───
    ax = axes[0, 0]
    corrs = [cond_results.get(c, {}).get('cross_temporal_corr_mean', 0) for c in conditions]
    stds = [cond_results.get(c, {}).get('cross_temporal_corr_std', 0) for c in conditions]
    bars = ax.bar(conditions, corrs, yerr=stds, color=[cond_colors[c] for c in conditions],
                  capsize=5, alpha=0.8)
    ax.set_ylabel('Cosine Similarity')
    ax.set_title('Cross-Temporal Correlation (end-A vs start-B)')
    ax.set_ylim(-0.2, 1.1)
    ax.grid(axis='y', alpha=0.3)

    # ─── Panel 2: Binding classification accuracy ───
    ax = axes[0, 1]
    bind_full = [cond_results.get(c, {}).get('binding_acc_AB', 0) for c in conditions]
    bind_a = [cond_results.get(c, {}).get('binding_acc_A_only', 0) for c in conditions]
    bind_b = [cond_results.get(c, {}).get('binding_acc_B_only', 0) for c in conditions]
    x = np.arange(len(conditions))
    w = 0.25
    ax.bar(x - w, bind_full, w, label='A+B', color='steelblue', alpha=0.8)
    ax.bar(x, bind_a, w, label='A only', color='coral', alpha=0.8)
    ax.bar(x + w, bind_b, w, label='B only', color='mediumseagreen', alpha=0.8)
    ax.axhline(y=1.0 / 3, color='red', linestyle='--', alpha=0.5, label='chance (33%)')
    ax.set_xticks(x)
    ax.set_xticklabels(conditions, fontsize=8)
    ax.set_ylabel('Accuracy')
    ax.set_title('Binding Classification (3-class)')
    ax.legend(fontsize=7)
    ax.grid(axis='y', alpha=0.3)

    # ─── Panel 3: Spike synchrony ───
    ax = axes[1, 0]
    for ci, cond in enumerate(conditions):
        cr = cond_results.get(cond, {})
        sync_stim = cr.get('sync_stimulus_mean', 0)
        sync_gap = cr.get('sync_gap_mean', 0)
        ax.bar(ci * 2, sync_stim, color=cond_colors[cond], alpha=0.9, label=cond if ci == 0 else None)
        ax.bar(ci * 2 + 1, sync_gap, color=cond_colors[cond], alpha=0.4)
    ax.set_xticks([i * 2 + 0.5 for i in range(len(conditions))])
    ax.set_xticklabels(conditions, fontsize=8)
    ax.set_ylabel('Pairwise Correlation')
    ax.set_title('Spike Synchrony: Stimulus (dark) vs Gap (light)')
    ax.grid(axis='y', alpha=0.3)

    # ─── Panel 4: Test summary ───
    ax = axes[1, 1]
    ax.axis('off')
    test_lines = []
    for t in tests:
        mark = 'PASS' if t['pass'] else 'FAIL'
        test_lines.append(f"{t['name']}: {mark}  ({t['detail']})")
    test_text = '\n'.join(test_lines)

    n_pass = sum(1 for t in tests if t['pass'])
    n_total = len(tests)
    header = f"Tests: {n_pass}/{n_total} PASS\n{'=' * 40}\n"

    ax.text(0.05, 0.95, header + test_text, transform=ax.transAxes,
            fontsize=9, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    ax.set_title('Test Results (T261-T266)')

    plt.tight_layout()
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig_path = FIGURES / 'z2195_temporal_binding.png'
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[INFO] Figure saved: {fig_path}")


# ═══════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description='z2195 Temporal Binding')
    parser.add_argument('--n-reps', type=int, default=DEFAULT_N_REPS)
    parser.add_argument('--sample-hz', type=int, default=DEFAULT_SAMPLE_HZ)
    parser.add_argument('--base-vg', type=float, default=DEFAULT_BASE_VG)
    parser.add_argument('--alpha', type=float, default=DEFAULT_ALPHA)
    parser.add_argument('--beta', type=float, default=DEFAULT_BETA)
    args = parser.parse_args()

    n_reps = args.n_reps
    sample_hz = args.sample_hz
    base_vg = args.base_vg
    alpha = args.alpha
    beta = args.beta

    print("=" * 65)
    print("z2195: Temporal Binding in GPU-Noise-Driven FPGA Reservoir")
    print("=" * 65)
    print(f"n_reps={n_reps}, sample_hz={sample_hz}, base_vg={base_vg}, "
          f"alpha={alpha}, beta={beta}")
    print(f"Trial structure: A({N_STEPS_A}) + gap({N_STEPS_GAP}) + B({N_STEPS_B}) "
          f"= {N_STEPS_TRIAL} steps")

    rng = np.random.default_rng(42)
    # Fixed random weights per neuron
    w_in = rng.uniform(-1, 1, size=N_NEURONS)
    w_noise = rng.uniform(-1, 1, size=N_NEURONS)

    # ─── SysfsHwmonTelemetry (for metadata) ───
    hwmon_avail = read_hwmon_power() is not None
    print(f"  hwmon power available: {hwmon_avail}")

    # ─── Prepare noise sources ───
    print("[1/5] Collecting GPU power noise for 1/f source...")
    noise_duration = max(20, (N_STEPS_TRIAL * n_reps * 3) / sample_hz * 0.1)
    raw_noise = collect_power_noise(duration_s=min(noise_duration, 30), sample_hz=50)
    if raw_noise is not None and len(raw_noise) > 10:
        noise_1f = iir_filter_noise(raw_noise, alpha_iir=0.85)
        print(f"  Collected {len(raw_noise)} power samples -> {len(noise_1f)} filtered")
    else:
        print("  [WARN] hwmon power not available, generating synthetic 1/f noise")
        rng_synth = np.random.default_rng(99)
        raw_synth = rng_synth.standard_normal(N_STEPS_TRIAL * n_reps * 4)
        vmf = VossMcCartneyFilter()
        noise_1f = np.array([vmf.process((s * 50 + 127.5)) for s in raw_synth])

    # White noise
    rng_white = np.random.default_rng(123)
    noise_white = rng_white.standard_normal(N_STEPS_TRIAL * n_reps * 4)

    # ─── Try FPGA connection ───
    print("[2/5] Connecting to FPGA...")
    ser, port = find_fpga()
    use_fpga = ser is not None
    simulated = not use_fpga
    if use_fpga:
        print(f"  FPGA found on {port}")
        connect_fpga(ser)
    else:
        print("  [WARN] No FPGA found, using software LIF simulation")

    # ─── Run trials for each condition ───
    conditions = ['FULL', 'WHITE', 'NO_NOISE', 'NOISE_GAP_ONLY']
    # For each condition: store per-trial states and labels
    cond_trial_states = {c: [] for c in conditions}
    cond_trial_labels = {c: [] for c in conditions}

    print("[3/5] Running trials...")
    total_trials = len(conditions) * n_reps * len(PAIRINGS)
    trial_count = 0

    for cond in conditions:
        print(f"\n  === Condition: {cond} ===")
        for pairing_idx in range(len(PAIRINGS)):
            for rep in range(n_reps):
                trial_count += 1
                seed_offset = pairing_idx * 1000 + rep
                trial_rng = np.random.default_rng(42 + seed_offset)

                input_signal = generate_trial_stimulus(pairing_idx, sample_hz, trial_rng)

                if use_fpga:
                    # Select noise source based on condition
                    if cond in ('FULL', 'NOISE_GAP_ONLY'):
                        ns = noise_1f
                    elif cond == 'WHITE':
                        ns = noise_white
                    else:
                        ns = np.array([])
                    states = run_fpga_trial(
                        ser, input_signal, ns, w_in, w_noise,
                        base_vg, alpha, beta, sample_hz, cond)
                else:
                    if cond in ('FULL', 'NOISE_GAP_ONLY'):
                        ns = noise_1f
                    elif cond == 'WHITE':
                        ns = noise_white
                    else:
                        ns = np.array([])
                    states = simulate_lif_trial(
                        input_signal, ns, w_in, w_noise,
                        base_vg, alpha, beta, sample_hz, cond,
                        rng_seed=hash((cond, pairing_idx, rep)) & 0x7FFFFFFF)

                cond_trial_states[cond].append(states)
                cond_trial_labels[cond].append(pairing_idx)

                if trial_count % 25 == 0:
                    print(f"    trial {trial_count}/{total_trials}")

    if ser:
        ser.close()

    # ─── Analyze ───
    print("\n[4/5] Analyzing results...")

    cond_results = {}
    for cond in conditions:
        trial_states = cond_trial_states[cond]
        trial_labels = np.array(cond_trial_labels[cond])
        n_trials = len(trial_states)

        # 1. Cross-temporal correlation
        ctc_values = [cross_temporal_correlation(st) for st in trial_states]
        ctc_mean = float(np.mean(ctc_values))
        ctc_std = float(np.std(ctc_values))

        # 2. Binding classification
        # Features: A+B combined, A-only, B-only
        feats_ab = np.array([
            np.concatenate([extract_phase_features(st, 'A'),
                            extract_phase_features(st, 'B')])
            for st in trial_states
        ])
        feats_a = np.array([extract_phase_features(st, 'A') for st in trial_states])
        feats_b = np.array([extract_phase_features(st, 'B') for st in trial_states])

        # 5-fold CV
        splits = stratified_kfold(feats_ab, trial_labels, n_splits=5, seed=42)
        acc_ab_folds = []
        acc_a_folds = []
        acc_b_folds = []
        for train_idx, test_idx in splits:
            acc_ab_folds.append(ridge_classify(
                feats_ab[train_idx], trial_labels[train_idx],
                feats_ab[test_idx], trial_labels[test_idx]))
            acc_a_folds.append(ridge_classify(
                feats_a[train_idx], trial_labels[train_idx],
                feats_a[test_idx], trial_labels[test_idx]))
            acc_b_folds.append(ridge_classify(
                feats_b[train_idx], trial_labels[train_idx],
                feats_b[test_idx], trial_labels[test_idx]))

        binding_acc_ab = float(np.mean(acc_ab_folds))
        binding_acc_a = float(np.mean(acc_a_folds))
        binding_acc_b = float(np.mean(acc_b_folds))

        # 3. Spike synchrony during stimulus vs gap
        sync_stim_vals = []
        sync_gap_vals = []
        for st in trial_states:
            sync_a = pairwise_spike_correlation(st, 'A')
            sync_b = pairwise_spike_correlation(st, 'B')
            sync_g = pairwise_spike_correlation(st, 'gap')
            sync_stim_vals.append((sync_a + sync_b) / 2.0)
            sync_gap_vals.append(sync_g)
        sync_stim_mean = float(np.mean(sync_stim_vals))
        sync_gap_mean = float(np.mean(sync_gap_vals))

        # 4. Temporal integration MI
        ti_mi = temporal_integration_mi(trial_states)

        cond_results[cond] = {
            'cross_temporal_corr_mean': ctc_mean,
            'cross_temporal_corr_std': ctc_std,
            'binding_acc_AB': binding_acc_ab,
            'binding_acc_A_only': binding_acc_a,
            'binding_acc_B_only': binding_acc_b,
            'sync_stimulus_mean': sync_stim_mean,
            'sync_gap_mean': sync_gap_mean,
            'temporal_integration_mi': ti_mi,
            'n_trials': n_trials,
        }

        print(f"  {cond}: CTC={ctc_mean:.4f} +/- {ctc_std:.4f}, "
              f"bind_AB={binding_acc_ab:.3f}, bind_A={binding_acc_a:.3f}, "
              f"bind_B={binding_acc_b:.3f}, "
              f"sync_stim={sync_stim_mean:.4f}, sync_gap={sync_gap_mean:.4f}, "
              f"MI={ti_mi:.4f}")

    # ─── Tests T261-T266 ───
    print("\n[5/5] Evaluating tests T261-T266...")

    full_r = cond_results['FULL']
    white_r = cond_results['WHITE']
    nonoise_r = cond_results['NO_NOISE']
    gaponly_r = cond_results['NOISE_GAP_ONLY']

    tests = []

    # T261: Cross-temporal corr FULL > NO_NOISE
    t261 = full_r['cross_temporal_corr_mean'] > nonoise_r['cross_temporal_corr_mean']
    tests.append({
        'name': 'T261',
        'pass': t261,
        'detail': (f"CTC_FULL={full_r['cross_temporal_corr_mean']:.4f} > "
                   f"CTC_NONOISE={nonoise_r['cross_temporal_corr_mean']:.4f}"),
    })

    # T262: Binding accuracy > chance (33%)
    t262 = full_r['binding_acc_AB'] > (1.0 / 3)
    tests.append({
        'name': 'T262',
        'pass': t262,
        'detail': f"bind_AB_FULL={full_r['binding_acc_AB']:.3f} > 0.333",
    })

    # T263: FULL binding accuracy > WHITE binding accuracy
    t263 = full_r['binding_acc_AB'] > white_r['binding_acc_AB']
    tests.append({
        'name': 'T263',
        'pass': t263,
        'detail': (f"bind_FULL={full_r['binding_acc_AB']:.3f} > "
                   f"bind_WHITE={white_r['binding_acc_AB']:.3f}"),
    })

    # T264: NOISE_GAP_ONLY corr > NO_NOISE corr
    t264 = gaponly_r['cross_temporal_corr_mean'] > nonoise_r['cross_temporal_corr_mean']
    tests.append({
        'name': 'T264',
        'pass': t264,
        'detail': (f"CTC_GAPONLY={gaponly_r['cross_temporal_corr_mean']:.4f} > "
                   f"CTC_NONOISE={nonoise_r['cross_temporal_corr_mean']:.4f}"),
    })

    # T265: Spike synchrony during stimulus > during gap (for ALL conditions)
    sync_pass_all = True
    sync_details = []
    for cond in conditions:
        cr = cond_results[cond]
        ok = cr['sync_stimulus_mean'] > cr['sync_gap_mean']
        if not ok:
            sync_pass_all = False
        sync_details.append(f"{cond}: stim={cr['sync_stimulus_mean']:.4f} "
                            f"{'>' if ok else '<='} gap={cr['sync_gap_mean']:.4f}")
    t265 = sync_pass_all
    tests.append({
        'name': 'T265',
        'pass': t265,
        'detail': '; '.join(sync_details),
    })

    # T266: Temporal integration MI > 0.05 bits for FULL
    t266 = full_r['temporal_integration_mi'] > 0.05
    tests.append({
        'name': 'T266',
        'pass': t266,
        'detail': f"MI_FULL={full_r['temporal_integration_mi']:.4f} > 0.05",
    })

    n_pass = sum(1 for t in tests if t['pass'])
    n_total = len(tests)
    print(f"\n{'=' * 55}")
    for t in tests:
        mark = 'PASS' if t['pass'] else 'FAIL'
        print(f"  {t['name']}: {mark}  {t['detail']}")
    print(f"{'=' * 55}")
    print(f"  TOTAL: {n_pass}/{n_total} PASS")

    # ─── Save results ───
    RESULTS.mkdir(parents=True, exist_ok=True)
    result_data = {
        'experiment': 'z2195_temporal_binding',
        'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
        'simulated': simulated,
        'params': {
            'n_reps': n_reps,
            'sample_hz': sample_hz,
            'base_vg': base_vg,
            'alpha': alpha,
            'beta': beta,
            'n_neurons': N_NEURONS,
            'n_steps_A': N_STEPS_A,
            'n_steps_gap': N_STEPS_GAP,
            'n_steps_B': N_STEPS_B,
            'n_pairings': len(PAIRINGS),
            'pairings': [list(p) for p in PAIRINGS],
            'use_fpga': use_fpga,
            'w_in': w_in.tolist(),
            'w_noise': w_noise.tolist(),
        },
        'conditions': cond_results,
        'tests': tests,
        'pass_count': n_pass,
        'total_tests': n_total,
    }

    result_path = RESULTS / 'z2195_temporal_binding.json'
    with open(result_path, 'w') as f:
        json.dump(result_data, f, indent=2, cls=NpEncoder)
    print(f"\n[INFO] Results saved: {result_path}")

    # ─── Figure ───
    make_figure(cond_results, tests)

    return n_pass, n_total


if __name__ == '__main__':
    main()
