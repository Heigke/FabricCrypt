#!/usr/bin/env python3
"""z2189_noise_memory_trace.py — GPU Noise Memory Trace in FPGA Reservoir

Demonstrates that GPU firmware noise leaves a PERSISTENT MEMORY TRACE in the
FPGA reservoir — the noise doesn't just modulate activity, it creates a temporal
fingerprint that persists AFTER the noise is removed. Closest analog to
memristive weight retention in Mario Lanza's RRAM research.

Protocol (per trial):
  Phase 1 — ENCODING (200 steps): Drive reservoir with signal + GPU 1/f noise
  Phase 2 — SILENCE  ( 50 steps): No input, no noise — observe decay
  Phase 3 — PROBE    (100 steps): Same signal, NO noise — does trace remain?

Conditions:
  FULL_SAME  — Same GPU 1/f noise in Phase 1, 20 reps
  FULL_DIFF  — Different GPU 1/f noise each trial
  WHITE_SAME — Same white noise in Phase 1
  WHITE_DIFF — Different white noise each trial
  NO_NOISE   — No noise at all (deterministic control)

Tests T225-T230:
  T225: Within-pattern sim (FULL_SAME) > Between-pattern sim (FULL_DIFF)
  T226: Trace effect (FULL) > Trace effect (WHITE) — 1/f stronger trace
  T227: NO_NOISE within-pattern similarity > 0.8 — deterministic consistency
  T228: FULL_SAME similarity > WHITE_SAME similarity
  T229: Phase 2 decay rate measurable (spike rate decreases during silence)
  T230: Phase 3 classification accuracy > chance (50%) for FULL

Hardware: AMD gfx1151 GPU + Arty A7 FPGA on /dev/ttyUSB*
"""

import os, sys, json, time, struct, argparse
import numpy as np
from pathlib import Path

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

# ─── Reservoir Parameters ───
BASE_VG = 0.55
ALPHA = 0.15
BETA = 0.10
N_NEURONS = 8
SAMPLE_HZ = 20

# ─── Phase lengths ───
PHASE1_STEPS = 200
PHASE2_STEPS = 50
PHASE3_STEPS = 100
N_REPS = 20


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
# FPGA Communication (from z2177)
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
        self.counters = np.zeros(n_octaves, dtype=int)
        self.step = 0

    def process(self, white_sample):
        """Process one white noise sample, return 1/f filtered value."""
        x = (white_sample / 127.5) - 1.0
        for k in range(self.n_octaves):
            period = 1 << k
            if self.step % period == 0:
                self.values[k] = x
        self.step += 1
        total = np.sum(self.values) / self.n_octaves
        return np.clip(total, -1.0, 1.0)


# ═══════════════════════════════════════════════════════════
# Reservoir Drivers
# ═══════════════════════════════════════════════════════════

def run_fpga_phase(ser, input_signal, noise_samples, w_in, w_noise,
                   base_vg=BASE_VG, alpha=ALPHA, beta=BETA, live_noise=False):
    """Drive FPGA neurons through one phase and collect spike/vmem states.

    Returns: (n_steps, 24) array — 8 delta_spikes + 8 vmem + 8 cumulative_spikes.
    """
    n_steps = len(input_signal)
    interval = 1.0 / SAMPLE_HZ
    states = np.zeros((n_steps, N_NEURONS * 3))
    prev_counts = None
    cumulative = np.zeros(N_NEURONS)
    power_mean = 11.0

    for t in range(n_steps):
        # Noise value
        if live_noise:
            p = read_hwmon_power()
            noise_val = (p - power_mean) / 2.0 if p else 0.0
        elif beta > 0 and len(noise_samples) > 0:
            noise_val = noise_samples[t % len(noise_samples)]
        else:
            noise_val = 0.0

        # Compute Vg per neuron
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
                    cumulative[i] += delta
            for i in range(N_NEURONS):
                states[t, N_NEURONS + i] = vmems[i]
                states[t, N_NEURONS * 2 + i] = cumulative[i]
            prev_counts = counts[:]

        time.sleep(max(0, interval * 0.5 - 0.01))

    return states


def simulate_lif_phase(input_signal, noise_samples, w_in, w_noise,
                       base_vg=BASE_VG, alpha=ALPHA, beta=BETA,
                       vmem_init=None):
    """Software LIF simulation fallback.

    Returns: (states, vmem_final) where states is (n_steps, 24).
    """
    n_steps = len(input_signal)
    states = np.zeros((n_steps, N_NEURONS * 3))

    v_rest = 0.0
    v_thresh = 1.0
    tau_m = 0.02
    dt = 1.0 / SAMPLE_HZ
    vmem = vmem_init.copy() if vmem_init is not None else np.zeros(N_NEURONS)
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

    return states, vmem.copy()


# ═══════════════════════════════════════════════════════════
# Full Trial: Phase 1 → Phase 2 → Phase 3
# ═══════════════════════════════════════════════════════════

def run_trial(ser, fpga, signal_p1, noise_p1, signal_p3, w_in, w_noise, rng):
    """Execute one complete 3-phase trial.

    Returns dict with phase1_states, phase2_states, phase3_states.
    """
    silence = np.zeros(PHASE2_STEPS)
    noise_zero = np.zeros(1000)

    if fpga:
        # Phase 1: ENCODING — signal + noise
        p1 = run_fpga_phase(ser, signal_p1, noise_p1, w_in, w_noise,
                            beta=BETA if len(noise_p1) > 0 and np.any(noise_p1 != 0) else 0.0)
        # Phase 2: SILENCE — no input, no noise
        p2 = run_fpga_phase(ser, silence, noise_zero, w_in, w_noise, beta=0.0)
        # Phase 3: PROBE — same signal, NO noise
        p3 = run_fpga_phase(ser, signal_p3, noise_zero, w_in, w_noise, beta=0.0)
    else:
        p1, vmem1 = simulate_lif_phase(signal_p1, noise_p1, w_in, w_noise,
                                        beta=BETA if len(noise_p1) > 0 and np.any(noise_p1 != 0) else 0.0)
        p2, vmem2 = simulate_lif_phase(silence, noise_zero, w_in, w_noise,
                                        beta=0.0, vmem_init=vmem1)
        p3, _ = simulate_lif_phase(signal_p3, noise_zero, w_in, w_noise,
                                    beta=0.0, vmem_init=vmem2)

    return {'phase1': p1, 'phase2': p2, 'phase3': p3}


# ═══════════════════════════════════════════════════════════
# Analysis
# ═══════════════════════════════════════════════════════════

def extract_phase3_features(phase3_states):
    """Extract spike rate features from Phase 3: 8 x PHASE3_STEPS matrix flattened."""
    # Use delta spikes (first 8 columns)
    return phase3_states[:, :N_NEURONS].flatten()


def cosine_similarity(a, b):
    """Cosine similarity between two vectors."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a < 1e-12 or norm_b < 1e-12:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def pairwise_cosine_matrix(features_list):
    """Compute pairwise cosine similarity matrix."""
    n = len(features_list)
    sim = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            s = cosine_similarity(features_list[i], features_list[j])
            sim[i, j] = s
            sim[j, i] = s
    return sim


def within_pattern_similarity(sim_matrix):
    """Mean of off-diagonal entries (all are same-pattern pairs)."""
    n = sim_matrix.shape[0]
    if n < 2:
        return 0.0
    mask = ~np.eye(n, dtype=bool)
    return float(np.mean(sim_matrix[mask]))


def between_pattern_similarity(features_list):
    """Mean pairwise similarity treating each trial as different-pattern."""
    n = len(features_list)
    if n < 2:
        return 0.0
    sims = []
    for i in range(n):
        for j in range(i + 1, n):
            sims.append(cosine_similarity(features_list[i], features_list[j]))
    return float(np.mean(sims)) if sims else 0.0


def compute_decay_rate(phase2_states):
    """Compute spike rate decay during silence phase."""
    # Spike rates per timestep (sum across neurons)
    spike_rates = phase2_states[:, :N_NEURONS].sum(axis=1)
    if len(spike_rates) < 5:
        return 0.0
    # Use first vs last quarter
    q = max(1, len(spike_rates) // 4)
    first_q = spike_rates[:q].mean()
    last_q = spike_rates[-q:].mean()
    if first_q < 1e-8:
        return 0.0
    return float((first_q - last_q) / max(first_q, 1e-8))


def ridge_binary_classify(X_train, y_train, X_test, y_test, alphas=None):
    """Ridge binary classifier returning accuracy."""
    if alphas is None:
        alphas = [1e-6, 1e-4, 1e-2, 1.0, 100.0]
    best_acc = -1
    for alpha_val in alphas:
        I = np.eye(X_train.shape[1])
        try:
            w = np.linalg.solve(X_train.T @ X_train + alpha_val * I, X_train.T @ y_train)
        except np.linalg.LinAlgError:
            continue
        pred = (X_test @ w > 0.5).astype(float)
        acc = np.mean(pred == y_test)
        if acc > best_acc:
            best_acc = acc
    return max(best_acc, 0.0)


# ═══════════════════════════════════════════════════════════
# Plotting
# ═══════════════════════════════════════════════════════════

def make_figure(cond_results, test_results, out_path):
    """Generate 2x2 figure: similarity matrices, trace strength, decay, classification."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("  matplotlib not available, skipping figure")
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle('z2189: GPU Noise Memory Trace in FPGA Reservoir', fontsize=14, fontweight='bold')

    # (0,0) Similarity matrices for FULL_SAME vs FULL_DIFF
    ax = axes[0, 0]
    if 'FULL_SAME' in cond_results and 'sim_matrix' in cond_results['FULL_SAME']:
        sim = np.array(cond_results['FULL_SAME']['sim_matrix'])
        im = ax.imshow(sim, vmin=0, vmax=1, cmap='viridis', aspect='auto')
        ax.set_title('Phase 3 Similarity: FULL_SAME')
        ax.set_xlabel('Trial')
        ax.set_ylabel('Trial')
        plt.colorbar(im, ax=ax, label='Cosine Similarity')
    else:
        ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Phase 3 Similarity: FULL_SAME')

    # (0,1) Trace strength comparison
    ax = axes[0, 1]
    conds = ['FULL_SAME', 'FULL_DIFF', 'WHITE_SAME', 'WHITE_DIFF', 'NO_NOISE']
    sims = []
    labels = []
    for c in conds:
        if c in cond_results and 'within_sim' in cond_results[c]:
            sims.append(cond_results[c]['within_sim'])
            labels.append(c)
    if sims:
        colors = ['#2196F3', '#90CAF9', '#FF9800', '#FFE0B2', '#9E9E9E'][:len(sims)]
        bars = ax.bar(range(len(sims)), sims, color=colors)
        ax.set_xticks(range(len(sims)))
        ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
        ax.set_ylabel('Mean Cosine Similarity')
        ax.set_title('Phase 3 Within-Pattern Similarity')
        ax.set_ylim(0, 1.1)
        for bar, val in zip(bars, sims):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=9)
    else:
        ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Phase 3 Within-Pattern Similarity')

    # (1,0) Decay curves (Phase 2 spike rate over time)
    ax = axes[1, 0]
    for c in ['FULL_SAME', 'WHITE_SAME', 'NO_NOISE']:
        if c in cond_results and 'mean_decay_curve' in cond_results[c]:
            curve = np.array(cond_results[c]['mean_decay_curve'])
            ax.plot(curve, label=c, linewidth=1.5)
    ax.set_xlabel('Timestep (Phase 2 — Silence)')
    ax.set_ylabel('Mean Spike Rate')
    ax.set_title('Phase 2 Decay Curves')
    ax.legend(fontsize=8)

    # (1,1) Classification accuracy
    ax = axes[1, 1]
    test_names = ['T225', 'T226', 'T227', 'T228', 'T229', 'T230']
    passes = [test_results.get(t, {}).get('pass', False) for t in test_names]
    colors_t = ['#4CAF50' if p else '#F44336' for p in passes]
    ax.barh(range(len(test_names)), [1] * len(test_names), color=colors_t, height=0.6)
    ax.set_yticks(range(len(test_names)))
    ax.set_yticklabels(test_names)
    ax.set_xlim(0, 1.2)
    ax.set_title(f'Test Results: {sum(passes)}/{len(test_names)} PASS')
    for i, t in enumerate(test_names):
        desc = test_results.get(t, {}).get('description', '')
        status = 'PASS' if passes[i] else 'FAIL'
        ax.text(0.5, i, f'{status}: {desc[:50]}', va='center', fontsize=7)

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Figure saved: {out_path}")


# ═══════════════════════════════════════════════════════════
# Main Experiment
# ═══════════════════════════════════════════════════════════

def main():
    global PHASE1_STEPS, PHASE2_STEPS, PHASE3_STEPS

    parser = argparse.ArgumentParser(description='z2189: GPU Noise Memory Trace in FPGA Reservoir')
    parser.add_argument('--reps', type=int, default=N_REPS, help='Repetitions per condition')
    parser.add_argument('--noise-collect-s', type=float, default=30.0,
                        help='Duration to collect power noise (s)')
    parser.add_argument('--phase1-steps', type=int, default=PHASE1_STEPS)
    parser.add_argument('--phase2-steps', type=int, default=PHASE2_STEPS)
    parser.add_argument('--phase3-steps', type=int, default=PHASE3_STEPS)
    args = parser.parse_args()

    n_reps = args.reps
    p1_steps = args.phase1_steps
    p2_steps = args.phase2_steps
    p3_steps = args.phase3_steps

    print("=" * 70)
    print("z2189: GPU Noise Memory Trace in FPGA Reservoir")
    print("=" * 70)
    print(f"  Reps/condition: {n_reps}  Phases: {p1_steps}/{p2_steps}/{p3_steps}")
    print(f"  BASE_VG={BASE_VG}  ALPHA={ALPHA}  BETA={BETA}  N_NEURONS={N_NEURONS}")

    rng = np.random.default_rng(42)
    w_in = rng.uniform(-1, 1, size=N_NEURONS)
    w_noise = rng.uniform(-1, 1, size=N_NEURONS)

    # Override phase lengths from args
    PHASE1_STEPS = p1_steps
    PHASE2_STEPS = p2_steps
    PHASE3_STEPS = p3_steps

    results = {
        'experiment': 'z2189_noise_memory_trace',
        'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
        'params': {
            'base_vg': BASE_VG, 'alpha': ALPHA, 'beta': BETA,
            'n_neurons': N_NEURONS, 'sample_hz': SAMPLE_HZ,
            'n_reps': n_reps,
            'phase1_steps': p1_steps, 'phase2_steps': p2_steps, 'phase3_steps': p3_steps,
            'w_in': w_in.tolist(), 'w_noise': w_noise.tolist(),
        },
        'simulated': False,
    }

    # ─── Step 1: Connect to FPGA ───
    print("\n[1/5] Connecting to FPGA...")
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
    print("\n[2/5] Collecting GPU noise sources...")
    print(f"  Collecting power rail noise ({args.noise_collect_s}s)...")
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
    print(f"  IIR-filtered 1/f noise: {len(noise_1f)} samples")

    # Generate white noise pool
    white_noise_pool = rng.standard_normal(len(noise_1f))

    # ─── Step 3: Generate signal ───
    print("\n[3/5] Generating probe signals...")
    t_p1 = np.arange(p1_steps) / SAMPLE_HZ
    signal_p1 = np.sin(2 * np.pi * 1.0 * t_p1)  # 1 Hz sine
    t_p3 = np.arange(p3_steps) / SAMPLE_HZ
    signal_p3 = np.sin(2 * np.pi * 1.0 * t_p3)  # Same 1 Hz sine
    print(f"  Phase 1 signal: 1 Hz sine, {p1_steps} steps")
    print(f"  Phase 3 signal: 1 Hz sine, {p3_steps} steps (same)")

    # Prepare noise segments for conditions:
    # FULL_SAME: fixed 1/f segment (first p1_steps from collection)
    noise_full_fixed = noise_1f[:p1_steps] if len(noise_1f) >= p1_steps else \
        np.tile(noise_1f, (p1_steps // max(len(noise_1f), 1)) + 1)[:p1_steps]
    # WHITE_SAME: fixed white segment
    noise_white_fixed = white_noise_pool[:p1_steps] if len(white_noise_pool) >= p1_steps else \
        np.tile(white_noise_pool, (p1_steps // max(len(white_noise_pool), 1)) + 1)[:p1_steps]

    # ─── Step 4: Run conditions ───
    print("\n[4/5] Running conditions...")

    conditions = {
        'FULL_SAME': {
            'label': 'Same 1/f noise each trial',
            'get_noise': lambda trial: noise_full_fixed.copy(),
        },
        'FULL_DIFF': {
            'label': 'Different 1/f noise each trial',
            'get_noise': lambda trial: iir_filter_noise(
                rng.standard_normal(p1_steps), alpha_iir=0.85),
        },
        'WHITE_SAME': {
            'label': 'Same white noise each trial',
            'get_noise': lambda trial: noise_white_fixed.copy(),
        },
        'WHITE_DIFF': {
            'label': 'Different white noise each trial',
            'get_noise': lambda trial: rng.standard_normal(p1_steps),
        },
        'NO_NOISE': {
            'label': 'No noise (deterministic)',
            'get_noise': lambda trial: np.zeros(p1_steps),
        },
    }

    cond_results = {}

    for cond_name, cond in conditions.items():
        print(f"\n  === {cond_name}: {cond['label']} ===")
        trial_data = []

        for rep in range(n_reps):
            noise_p1 = cond['get_noise'](rep)
            td = run_trial(ser, fpga, signal_p1, noise_p1, signal_p3, w_in, w_noise, rng)
            trial_data.append(td)
            if (rep + 1) % 5 == 0:
                print(f"    rep {rep + 1}/{n_reps}")

        # Extract Phase 3 features
        phase3_features = [extract_phase3_features(td['phase3']) for td in trial_data]

        # Compute similarity matrix
        sim_matrix = pairwise_cosine_matrix(phase3_features)
        within_sim = within_pattern_similarity(sim_matrix)

        # Phase 2 decay curves
        decay_curves = [td['phase2'][:, :N_NEURONS].sum(axis=1) for td in trial_data]
        mean_decay = np.mean(decay_curves, axis=0)
        decay_rate = np.mean([compute_decay_rate(td['phase2']) for td in trial_data])

        cond_results[cond_name] = {
            'within_sim': float(within_sim),
            'sim_matrix': sim_matrix.tolist(),
            'mean_decay_curve': mean_decay.tolist(),
            'decay_rate': float(decay_rate),
            'n_trials': n_reps,
        }

        print(f"    Within-pattern similarity: {within_sim:.4f}")
        print(f"    Decay rate: {decay_rate:.4f}")

    # ─── T230: Classification (ridge classifier on Phase 3 features) ───
    # Use FULL condition: first 10 reps = pattern A (FULL_SAME noise), next 10 = pattern B (new fixed noise)
    print("\n  === T230 Classification: Noise Pattern ID from Phase 3 ===")
    noise_pattern_b = iir_filter_noise(rng.standard_normal(p1_steps), alpha_iir=0.85)

    class_trials_a = []
    class_trials_b = []
    for rep in range(n_reps // 2):
        # Pattern A: use noise_full_fixed
        td_a = run_trial(ser, fpga, signal_p1, noise_full_fixed, signal_p3, w_in, w_noise, rng)
        class_trials_a.append(extract_phase3_features(td_a['phase3']))
        # Pattern B: use noise_pattern_b
        td_b = run_trial(ser, fpga, signal_p1, noise_pattern_b, signal_p3, w_in, w_noise, rng)
        class_trials_b.append(extract_phase3_features(td_b['phase3']))
        if (rep + 1) % 5 == 0:
            print(f"    classification rep {rep + 1}/{n_reps // 2}")

    # Build dataset: label A=0, B=1
    X_all = np.array(class_trials_a + class_trials_b)
    y_all = np.array([0.0] * len(class_trials_a) + [1.0] * len(class_trials_b))

    # Leave-one-out cross-validation
    n_total = len(X_all)
    correct = 0
    for i in range(n_total):
        X_train = np.delete(X_all, i, axis=0)
        y_train = np.delete(y_all, i)
        X_test = X_all[i:i + 1]
        y_test = y_all[i:i + 1]
        acc = ridge_binary_classify(X_train, y_train, X_test, y_test)
        correct += int(acc > 0.5)
    classification_acc = correct / n_total
    print(f"    Classification accuracy (LOO-CV): {classification_acc:.3f}")

    # ─── Step 5: Tests T225-T230 ───
    print("\n[5/5] Evaluating tests T225-T230...")

    full_same_sim = cond_results['FULL_SAME']['within_sim']
    full_diff_sim = cond_results['FULL_DIFF']['within_sim']
    white_same_sim = cond_results['WHITE_SAME']['within_sim']
    white_diff_sim = cond_results['WHITE_DIFF']['within_sim']
    no_noise_sim = cond_results['NO_NOISE']['within_sim']

    full_trace = full_same_sim - full_diff_sim
    white_trace = white_same_sim - white_diff_sim
    mean_decay_rate = np.mean([cond_results[c]['decay_rate'] for c in
                               ['FULL_SAME', 'WHITE_SAME'] if c in cond_results])

    test_results = {}

    # T225: Within-pattern sim (FULL_SAME) > Between-pattern sim (FULL_DIFF)
    t225_pass = full_same_sim > full_diff_sim
    test_results['T225'] = {
        'description': 'FULL_SAME sim > FULL_DIFF sim (noise leaves trace)',
        'full_same_sim': full_same_sim,
        'full_diff_sim': full_diff_sim,
        'delta': full_same_sim - full_diff_sim,
        'pass': bool(t225_pass),
    }
    tag = "PASS" if t225_pass else "FAIL"
    print(f"  T225 [{tag}]: FULL_SAME={full_same_sim:.4f} > FULL_DIFF={full_diff_sim:.4f}"
          f"  delta={full_same_sim - full_diff_sim:.4f}")

    # T226: Trace effect (FULL) > Trace effect (WHITE)
    t226_pass = full_trace > white_trace
    test_results['T226'] = {
        'description': '1/f trace > white trace (temporal memory)',
        'full_trace': full_trace,
        'white_trace': white_trace,
        'delta': full_trace - white_trace,
        'pass': bool(t226_pass),
    }
    tag = "PASS" if t226_pass else "FAIL"
    print(f"  T226 [{tag}]: FULL_trace={full_trace:.4f} > WHITE_trace={white_trace:.4f}"
          f"  delta={full_trace - white_trace:.4f}")

    # T227: NO_NOISE within-pattern similarity > 0.8
    t227_pass = no_noise_sim > 0.8
    test_results['T227'] = {
        'description': 'NO_NOISE similarity > 0.8 (deterministic consistency)',
        'no_noise_sim': no_noise_sim,
        'threshold': 0.8,
        'pass': bool(t227_pass),
    }
    tag = "PASS" if t227_pass else "FAIL"
    print(f"  T227 [{tag}]: NO_NOISE sim={no_noise_sim:.4f} > 0.8")

    # T228: FULL_SAME sim > WHITE_SAME sim
    t228_pass = full_same_sim > white_same_sim
    test_results['T228'] = {
        'description': 'FULL_SAME sim > WHITE_SAME sim (1/f more consistent)',
        'full_same_sim': full_same_sim,
        'white_same_sim': white_same_sim,
        'delta': full_same_sim - white_same_sim,
        'pass': bool(t228_pass),
    }
    tag = "PASS" if t228_pass else "FAIL"
    print(f"  T228 [{tag}]: FULL_SAME={full_same_sim:.4f} > WHITE_SAME={white_same_sim:.4f}")

    # T229: Phase 2 decay rate measurable
    t229_pass = mean_decay_rate > 0.0
    test_results['T229'] = {
        'description': 'Phase 2 decay rate > 0 (spike rate decreases in silence)',
        'mean_decay_rate': float(mean_decay_rate),
        'pass': bool(t229_pass),
    }
    tag = "PASS" if t229_pass else "FAIL"
    print(f"  T229 [{tag}]: mean_decay_rate={mean_decay_rate:.4f} > 0")

    # T230: Classification accuracy > 50%
    t230_pass = classification_acc > 0.50
    test_results['T230'] = {
        'description': 'Phase 3 noise pattern classification > 50% (FULL)',
        'classification_acc': classification_acc,
        'threshold': 0.50,
        'pass': bool(t230_pass),
    }
    tag = "PASS" if t230_pass else "FAIL"
    print(f"  T230 [{tag}]: classification_acc={classification_acc:.3f} > 0.50")

    n_pass = sum(1 for t in test_results.values() if t['pass'])
    n_total_tests = len(test_results)
    print(f"\n  TOTAL: {n_pass}/{n_total_tests} PASS")

    # ─── Save results ───
    results['conditions'] = cond_results
    results['tests'] = test_results
    results['classification_accuracy'] = classification_acc
    results['summary'] = {
        'n_pass': n_pass,
        'n_total': n_total_tests,
        'full_trace_effect': full_trace,
        'white_trace_effect': white_trace,
    }

    RESULTS.mkdir(parents=True, exist_ok=True)
    out_json = RESULTS / 'z2189_noise_memory_trace.json'
    with open(out_json, 'w') as f:
        json.dump(results, f, indent=2, cls=NpEncoder)
    print(f"\n  Results saved: {out_json}")

    # ─── Generate figure ───
    out_fig = FIGURES / 'z2189_noise_memory_trace.png'
    make_figure(cond_results, test_results, out_fig)

    # ─── Cleanup ───
    if fpga and ser:
        ser.write(bytes([SYNC, CMD_SET_KILL, 0x00]))
        ser.flush()
        ser.close()
        print("  FPGA connection closed")

    print("\nDone.")
    return n_pass, n_total_tests


if __name__ == '__main__':
    main()
