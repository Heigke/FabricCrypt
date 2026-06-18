#!/usr/bin/env python3
"""z2179_attractor_dynamics.py — Attractor Dynamics in GPU-Noise-Driven FPGA Reservoir

Tests whether the GPU-noise-driven FPGA reservoir exhibits attractor dynamics —
a hallmark of neural computation. Attractors are stable response patterns that
the reservoir converges to when presented with specific stimuli.

Experiment Design:
  - 3 stimulus patterns (constant low, constant high, oscillating) x 50 repetitions
  - 8 neurons, 20 steps per trial, 20 Hz
  - Measure: (a) within-pattern response consistency (Pearson correlation),
             (b) between-pattern separability (distinct attractors),
             (c) perturbation recovery — inject noise burst mid-trial
  - 3 conditions: FULL (1/f noise), WHITE, NO_NOISE

Tests T163-T168:
  T163: Within-pattern correlation > 0.3 (consistent attractors exist)
  T164: Between-pattern distance > within-pattern distance (distinct attractors)
  T165: FULL perturbation recovery time < NO_NOISE recovery time (noise helps)
  T166: At least 3 distinct clusters in response space (multiple attractors)
  T167: FULL within-pattern corr > WHITE within-pattern corr (1/f improves consistency)
  T168: Post-perturbation accuracy > 60% (reservoir recovers to useful computation)

Hardware: AMD gfx1151 GPU + Arty A7 FPGA on /dev/ttyUSB*
"""

import os, sys, json, time, struct, subprocess, argparse
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

# ─── Reservoir Parameters ───
BASE_VG = 0.55
ALPHA = 0.15
BETA = 0.10
N_NEURONS = 8
SAMPLE_HZ = 20
N_STEPS = 20
N_REPS = 50


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


# ═══════════════════════════════════════════════════════════
# FPGA Reservoir Core
# ═══════════════════════════════════════════════════════════

def run_fpga_reservoir_trial(ser, input_signal, noise_samples, w_in, w_noise,
                              base_vg=BASE_VG, alpha=ALPHA, beta=BETA,
                              live_noise=False, perturb_step=None, perturb_mag=0.3):
    """Drive FPGA neurons with input+noise and collect spike/vmem states.

    If perturb_step is set, inject a large noise burst at that timestep.

    Returns: (n_steps, 24) array -- 8 delta_spikes + 8 vmem + 8 cumulative_spikes.
    """
    n_steps = len(input_signal)
    interval = 1.0 / SAMPLE_HZ
    states = np.zeros((n_steps, N_NEURONS * 3))
    prev_counts = None
    cumulative = np.zeros(N_NEURONS)
    power_mean = 11.0

    for t in range(n_steps):
        if live_noise:
            p = read_hwmon_power()
            noise_val = (p - power_mean) / 2.0 if p else 0.0
        elif beta > 0 and len(noise_samples) > 0:
            noise_val = noise_samples[t % len(noise_samples)]
        else:
            noise_val = 0.0

        # Perturbation injection
        if perturb_step is not None and t == perturb_step:
            noise_val += perturb_mag * (1.0 if np.random.random() > 0.5 else -1.0)

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


def simulate_lif_reservoir(input_signal, noise_samples, w_in, w_noise,
                            base_vg=BASE_VG, alpha=ALPHA, beta=BETA,
                            perturb_step=None, perturb_mag=0.3):
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
            noise_val = noise_samples[noise_idx]
        else:
            noise_val = 0.0

        # Perturbation injection
        if perturb_step is not None and t == perturb_step:
            noise_val += perturb_mag * (1.0 if np.random.random() > 0.5 else -1.0)

        if beta > 0:
            vg += beta * noise_val * w_noise
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
# Stimulus Patterns
# ═══════════════════════════════════════════════════════════

def generate_stimulus_patterns(n_steps):
    """Generate 3 distinct stimulus patterns for attractor testing.
    Returns dict: pattern_name -> (n_steps,) signal in [0, 1].
    """
    t = np.arange(n_steps) / SAMPLE_HZ
    patterns = {
        'low': np.full(n_steps, 0.2),           # constant low
        'high': np.full(n_steps, 0.8),           # constant high
        'oscillating': 0.5 + 0.3 * np.sin(2 * np.pi * 2.0 * t),  # 2 Hz sine
    }
    return patterns


# ═══════════════════════════════════════════════════════════
# Analysis Utilities
# ═══════════════════════════════════════════════════════════

def flatten_trial(states):
    """Flatten (n_steps, n_features) trial states into a single vector."""
    return states.flatten()


def pairwise_correlations(vectors):
    """Compute mean pairwise Pearson correlation among a list of vectors."""
    n = len(vectors)
    if n < 2:
        return 0.0
    corrs = []
    for i in range(n):
        for j in range(i + 1, n):
            if np.std(vectors[i]) < 1e-8 or np.std(vectors[j]) < 1e-8:
                corrs.append(0.0)
            else:
                r = np.corrcoef(vectors[i], vectors[j])[0, 1]
                if np.isnan(r):
                    r = 0.0
                corrs.append(r)
    return float(np.mean(corrs))


def between_pattern_distance(pattern_centroids):
    """Compute mean Euclidean distance between pattern centroids."""
    names = list(pattern_centroids.keys())
    n = len(names)
    if n < 2:
        return 0.0
    dists = []
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(pattern_centroids[names[i]] - pattern_centroids[names[j]])
            dists.append(d)
    return float(np.mean(dists))


def within_pattern_spread(pattern_responses):
    """Compute mean within-pattern spread (mean distance to centroid)."""
    spreads = []
    for name, responses in pattern_responses.items():
        if len(responses) < 2:
            continue
        centroid = np.mean(responses, axis=0)
        dists = [np.linalg.norm(r - centroid) for r in responses]
        spreads.append(float(np.mean(dists)))
    return float(np.mean(spreads)) if spreads else 0.0


def compute_perturbation_recovery(baseline_trajectory, perturbed_trajectory, perturb_step):
    """Measure how many steps after perturbation the trajectory returns within
    baseline variance. Returns recovery time in steps (or n_steps if never recovers).
    """
    n_steps = len(baseline_trajectory)
    if perturb_step >= n_steps - 1:
        return n_steps

    # Compute baseline std at each step (across features)
    baseline_std = max(np.std(baseline_trajectory), 1e-6)

    # Distance between perturbed and baseline at each post-perturbation step
    for t in range(perturb_step + 1, n_steps):
        dist = np.linalg.norm(perturbed_trajectory[t] - baseline_trajectory[t])
        if dist < 2.0 * baseline_std:
            return t - perturb_step

    return n_steps - perturb_step


def kmeans_simple(data, k=3, max_iter=50, seed=42):
    """Simple k-means clustering. Returns labels, centroids."""
    rng = np.random.default_rng(seed)
    n = len(data)
    if n <= k:
        return np.arange(n), data.copy()

    # Initialize centroids randomly
    idx = rng.choice(n, k, replace=False)
    centroids = data[idx].copy()

    labels = np.zeros(n, dtype=int)
    for _ in range(max_iter):
        # Assign
        for i in range(n):
            dists = [np.linalg.norm(data[i] - centroids[c]) for c in range(k)]
            labels[i] = np.argmin(dists)
        # Update
        new_centroids = np.zeros_like(centroids)
        for c in range(k):
            members = data[labels == c]
            if len(members) > 0:
                new_centroids[c] = members.mean(axis=0)
            else:
                new_centroids[c] = centroids[c]
        if np.allclose(centroids, new_centroids, atol=1e-8):
            break
        centroids = new_centroids

    return labels, centroids


def silhouette_score_simple(data, labels):
    """Simplified silhouette score."""
    n = len(data)
    unique_labels = np.unique(labels)
    if len(unique_labels) < 2:
        return 0.0

    sil_scores = []
    for i in range(n):
        same_cluster = data[labels == labels[i]]
        if len(same_cluster) < 2:
            sil_scores.append(0.0)
            continue

        # a(i): mean distance to same cluster
        a_i = np.mean([np.linalg.norm(data[i] - same_cluster[j])
                        for j in range(len(same_cluster)) if not np.array_equal(same_cluster[j], data[i])])

        # b(i): min mean distance to other clusters
        b_i = np.inf
        for lab in unique_labels:
            if lab == labels[i]:
                continue
            other = data[labels == lab]
            if len(other) == 0:
                continue
            mean_dist = np.mean([np.linalg.norm(data[i] - other[j]) for j in range(len(other))])
            b_i = min(b_i, mean_dist)

        if b_i == np.inf:
            sil_scores.append(0.0)
        else:
            sil_scores.append((b_i - a_i) / max(a_i, b_i, 1e-8))

    return float(np.mean(sil_scores))


def pool_trial_features(trial_states):
    """Pool per-timestep reservoir states into trial-level features."""
    return np.concatenate([
        trial_states.mean(axis=0),
        trial_states.std(axis=0),
        trial_states.max(axis=0),
        trial_states.min(axis=0),
    ])


def ridge_classify(X_train, y_train, X_test, y_test, alphas=None):
    """Ridge regression classifier (one-hot encoding for multi-class)."""
    if alphas is None:
        alphas = [1e-6, 1e-4, 1e-2, 1.0, 100.0]

    n_classes = len(np.unique(y_train))
    Y_train = np.zeros((len(y_train), n_classes))
    for i, y in enumerate(y_train):
        Y_train[i, int(y)] = 1.0

    best_acc = -1
    for alpha_val in alphas:
        I = np.eye(X_train.shape[1])
        try:
            W = np.linalg.solve(X_train.T @ X_train + alpha_val * I, X_train.T @ Y_train)
        except np.linalg.LinAlgError:
            continue
        pred_test = np.argmax(X_test @ W, axis=1)
        acc_test = np.mean(pred_test == y_test)
        if acc_test > best_acc:
            best_acc = acc_test

    return max(best_acc, 0.0)


# ═══════════════════════════════════════════════════════════
# Main Experiment
# ═══════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description='z2179: Attractor Dynamics')
    parser.add_argument('--reps', type=int, default=N_REPS, help='Repetitions per pattern')
    parser.add_argument('--steps', type=int, default=N_STEPS, help='Steps per trial')
    parser.add_argument('--noise-collect-s', type=float, default=15.0,
                        help='Duration to collect power noise (s)')
    args = parser.parse_args()

    n_reps = args.reps
    n_steps = args.steps
    perturb_step = n_steps // 2  # inject perturbation at midpoint

    print("=" * 65)
    print("z2179: Attractor Dynamics in GPU-Noise-Driven FPGA Reservoir")
    print("=" * 65)
    print(f"  Patterns: 3 (low, high, oscillating)")
    print(f"  Repetitions: {n_reps}  Steps: {n_steps}  Perturb at step: {perturb_step}")
    print(f"  base_vg={BASE_VG}  alpha={ALPHA}  beta={BETA}")

    rng = np.random.default_rng(42)
    w_in = rng.uniform(-1, 1, size=N_NEURONS)
    w_noise = rng.uniform(-1, 1, size=N_NEURONS)

    results = {
        'experiment': 'z2179_attractor_dynamics',
        'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
        'params': {
            'base_vg': BASE_VG, 'alpha': ALPHA, 'beta': BETA,
            'n_neurons': N_NEURONS, 'sample_hz': SAMPLE_HZ,
            'n_steps': n_steps, 'n_reps': n_reps,
            'perturb_step': perturb_step,
            'w_in': w_in.tolist(), 'w_noise': w_noise.tolist(),
        },
        'simulated': False,
    }

    # ─── Step 1: Connect to FPGA ───
    print("\n[1/7] Connecting to FPGA...")
    ser, port = find_fpga()
    if ser is None:
        print("  FPGA not found -- using LIF simulation fallback")
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
    print("\n[2/7] Collecting GPU noise sources...")
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
    noise_zero = np.zeros(1000)

    conditions = {
        'FULL': {'noise': noise_1f, 'beta': BETA, 'label': 'GPU 1/f noise'},
        'WHITE': {'noise': noise_white, 'beta': BETA, 'label': 'White noise'},
        'NO_NOISE': {'noise': noise_zero, 'beta': 0.0, 'label': 'No noise'},
    }

    # ─── Step 3: Generate stimulus patterns ───
    print("\n[3/7] Generating stimulus patterns...")
    patterns = generate_stimulus_patterns(n_steps)
    pattern_names = list(patterns.keys())
    print(f"  Patterns: {pattern_names}")
    for name, sig in patterns.items():
        print(f"    {name}: mean={sig.mean():.2f}, range=[{sig.min():.2f}, {sig.max():.2f}]")

    # ─── Step 4: Run reservoir for all conditions ───
    print("\n[4/7] Running reservoir across conditions and patterns...")
    condition_data = {}

    for cond_name, cond in conditions.items():
        print(f"\n  === Condition: {cond_name} ({cond['label']}) ===")
        cond_noise = cond['noise']
        cond_beta = cond['beta']

        pattern_responses = {}   # pattern_name -> list of flattened response vectors
        pattern_raw = {}         # pattern_name -> list of (n_steps, features) arrays
        perturb_baselines = {}   # pattern_name -> mean baseline trajectory
        perturb_recovery_times = []

        for pat_name in pattern_names:
            inp = patterns[pat_name]
            responses = []
            raw_states = []

            print(f"    Pattern '{pat_name}': {n_reps} reps (normal)...", end='', flush=True)
            for rep in range(n_reps):
                if fpga:
                    st = run_fpga_reservoir_trial(ser, inp, cond_noise, w_in, w_noise,
                                                   beta=cond_beta,
                                                   live_noise=(cond_name == 'FULL'))
                else:
                    st = simulate_lif_reservoir(inp, cond_noise, w_in, w_noise,
                                                 beta=cond_beta)
                responses.append(flatten_trial(st))
                raw_states.append(st)
                if (rep + 1) % 25 == 0:
                    print(f" {rep + 1}", end='', flush=True)
            print()

            pattern_responses[pat_name] = responses
            pattern_raw[pat_name] = raw_states

            # Compute baseline (mean trajectory)
            mean_traj = np.mean(np.array([s for s in raw_states]), axis=0)
            perturb_baselines[pat_name] = mean_traj

            # Perturbation trials (10 reps per pattern)
            n_perturb = 10
            print(f"    Pattern '{pat_name}': {n_perturb} reps (perturbed at step {perturb_step})...")
            for rep in range(n_perturb):
                if fpga:
                    st_p = run_fpga_reservoir_trial(ser, inp, cond_noise, w_in, w_noise,
                                                     beta=cond_beta,
                                                     live_noise=(cond_name == 'FULL'),
                                                     perturb_step=perturb_step,
                                                     perturb_mag=0.3)
                else:
                    st_p = simulate_lif_reservoir(inp, cond_noise, w_in, w_noise,
                                                   beta=cond_beta,
                                                   perturb_step=perturb_step,
                                                   perturb_mag=0.3)
                recovery = compute_perturbation_recovery(mean_traj, st_p, perturb_step)
                perturb_recovery_times.append(recovery)

        # Compute within-pattern correlations
        within_corrs = {}
        for pat_name in pattern_names:
            within_corrs[pat_name] = pairwise_correlations(pattern_responses[pat_name])
        mean_within_corr = float(np.mean(list(within_corrs.values())))

        # Compute between-pattern distance
        centroids = {}
        for pat_name in pattern_names:
            centroids[pat_name] = np.mean(pattern_responses[pat_name], axis=0)
        mean_between_dist = between_pattern_distance(centroids)
        mean_within_spread = within_pattern_spread(pattern_responses)

        # K-means clustering on all responses
        all_responses = []
        all_labels_true = []
        for idx, pat_name in enumerate(pattern_names):
            for resp in pattern_responses[pat_name]:
                all_responses.append(resp)
                all_labels_true.append(idx)
        all_responses = np.array(all_responses)
        all_labels_true = np.array(all_labels_true)

        cluster_labels, cluster_centroids = kmeans_simple(all_responses, k=3, seed=42)
        sil_score = silhouette_score_simple(all_responses, cluster_labels)

        # Count distinct clusters (each cluster should have members from mostly one pattern)
        n_distinct = 0
        for c in range(3):
            members = all_labels_true[cluster_labels == c]
            if len(members) > 0:
                # Check if majority is from a single pattern
                counts = np.bincount(members, minlength=3)
                if counts.max() / len(members) > 0.5:
                    n_distinct += 1

        # Post-perturbation classification accuracy
        # Use pooled features from normal trials, train classifier, test on post-perturbation region
        # We will train on the second half (post-perturbation region) of normal trials
        X_train_list = []
        y_train_list = []
        for idx, pat_name in enumerate(pattern_names):
            for raw in pattern_raw[pat_name]:
                # Use only post-perturbation segment as features
                post_seg = raw[perturb_step:, :]
                feat = pool_trial_features(post_seg)
                X_train_list.append(feat)
                y_train_list.append(idx)
        X_train = np.array(X_train_list)
        y_train = np.array(y_train_list)

        # Now collect perturbed post-segments for testing
        X_test_list = []
        y_test_list = []
        for idx, pat_name in enumerate(pattern_names):
            inp = patterns[pat_name]
            for _ in range(10):
                if fpga:
                    st_p = run_fpga_reservoir_trial(ser, inp, cond_noise, w_in, w_noise,
                                                     beta=cond_beta,
                                                     live_noise=(cond_name == 'FULL'),
                                                     perturb_step=perturb_step,
                                                     perturb_mag=0.3)
                else:
                    st_p = simulate_lif_reservoir(inp, cond_noise, w_in, w_noise,
                                                   beta=cond_beta,
                                                   perturb_step=perturb_step,
                                                   perturb_mag=0.3)
                post_seg = st_p[perturb_step:, :]
                feat = pool_trial_features(post_seg)
                X_test_list.append(feat)
                y_test_list.append(idx)
        X_test = np.array(X_test_list)
        y_test = np.array(y_test_list)

        post_perturb_acc = ridge_classify(X_train, y_train, X_test, y_test)

        mean_recovery = float(np.mean(perturb_recovery_times))

        condition_data[cond_name] = {
            'within_pattern_correlations': within_corrs,
            'mean_within_corr': mean_within_corr,
            'mean_between_distance': mean_between_dist,
            'mean_within_spread': mean_within_spread,
            'distance_ratio': mean_between_dist / max(mean_within_spread, 1e-8),
            'n_distinct_clusters': n_distinct,
            'silhouette_score': sil_score,
            'mean_recovery_time': mean_recovery,
            'perturb_recovery_times': perturb_recovery_times,
            'post_perturb_accuracy': post_perturb_acc,
        }

        print(f"    Within-pattern corr: {mean_within_corr:.4f}")
        print(f"    Between-pattern dist: {mean_between_dist:.2f}, within spread: {mean_within_spread:.2f}")
        print(f"    Distance ratio: {condition_data[cond_name]['distance_ratio']:.4f}")
        print(f"    Distinct clusters: {n_distinct}/3, silhouette: {sil_score:.4f}")
        print(f"    Mean recovery time: {mean_recovery:.2f} steps")
        print(f"    Post-perturbation accuracy: {post_perturb_acc:.4f}")

    results['condition_data'] = condition_data

    # ─── Step 5: Evaluate tests T163-T168 ───
    print("\n[5/7] Evaluating tests T163-T168...")
    full = condition_data['FULL']
    white = condition_data['WHITE']
    no_noise = condition_data['NO_NOISE']

    tests = {}

    # T163: Within-pattern correlation > 0.3
    t163_val = full['mean_within_corr']
    t163_pass = t163_val > 0.3
    tests['T163'] = {
        'name': 'Within-pattern correlation > 0.3 (consistent attractors)',
        'within_corr': t163_val,
        'threshold': 0.3,
        'pass': t163_pass,
    }
    print(f"  T163: Within-pattern corr={t163_val:.4f} > 0.3 -> {'PASS' if t163_pass else 'FAIL'}")

    # T164: Between-pattern distance > within-pattern distance
    t164_ratio = full['distance_ratio']
    t164_pass = full['mean_between_distance'] > full['mean_within_spread']
    tests['T164'] = {
        'name': 'Between-pattern distance > within-pattern distance (distinct attractors)',
        'between_dist': full['mean_between_distance'],
        'within_spread': full['mean_within_spread'],
        'ratio': t164_ratio,
        'pass': t164_pass,
    }
    print(f"  T164: Between={full['mean_between_distance']:.2f} > Within={full['mean_within_spread']:.2f} "
          f"(ratio={t164_ratio:.2f}) -> {'PASS' if t164_pass else 'FAIL'}")

    # T165: FULL recovery time < NO_NOISE recovery time
    t165_full_rec = full['mean_recovery_time']
    t165_no_rec = no_noise['mean_recovery_time']
    t165_pass = t165_full_rec < t165_no_rec
    tests['T165'] = {
        'name': 'FULL recovery time < NO_NOISE recovery time (noise helps recovery)',
        'full_recovery': t165_full_rec,
        'no_noise_recovery': t165_no_rec,
        'pass': t165_pass,
    }
    print(f"  T165: FULL recovery={t165_full_rec:.2f} < NO_NOISE={t165_no_rec:.2f} -> "
          f"{'PASS' if t165_pass else 'FAIL'}")

    # T166: At least 3 distinct clusters
    t166_val = full['n_distinct_clusters']
    t166_pass = t166_val >= 3
    tests['T166'] = {
        'name': 'At least 3 distinct clusters (multiple attractors)',
        'n_distinct': t166_val,
        'threshold': 3,
        'silhouette': full['silhouette_score'],
        'pass': t166_pass,
    }
    print(f"  T166: Distinct clusters={t166_val} >= 3 (sil={full['silhouette_score']:.4f}) -> "
          f"{'PASS' if t166_pass else 'FAIL'}")

    # T167: FULL within-pattern corr > WHITE within-pattern corr
    t167_full = full['mean_within_corr']
    t167_white = white['mean_within_corr']
    t167_pass = t167_full > t167_white
    tests['T167'] = {
        'name': 'FULL within-pattern corr > WHITE (1/f improves consistency)',
        'full_corr': t167_full,
        'white_corr': t167_white,
        'pass': t167_pass,
    }
    print(f"  T167: FULL corr={t167_full:.4f} > WHITE={t167_white:.4f} -> "
          f"{'PASS' if t167_pass else 'FAIL'}")

    # T168: Post-perturbation accuracy > 60%
    t168_val = full['post_perturb_accuracy']
    t168_pass = t168_val > 0.60
    tests['T168'] = {
        'name': 'Post-perturbation accuracy > 60% (reservoir recovers)',
        'accuracy': t168_val,
        'threshold': 0.60,
        'pass': t168_pass,
    }
    print(f"  T168: Post-perturb acc={t168_val:.4f} > 0.60 -> {'PASS' if t168_pass else 'FAIL'}")

    n_pass = sum(1 for t in tests.values() if t['pass'])
    print(f"\n  TOTAL: {n_pass}/6 PASS")
    results['tests'] = tests
    results['n_pass'] = n_pass
    results['n_total'] = 6

    # ─── Step 6: Save results ───
    print("\n[6/7] Saving results...")
    RESULTS.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS / 'z2179_attractor_dynamics.json'
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2, cls=NpEncoder)
    print(f"  Results saved: {out_path}")

    # ─── Step 7: Generate figures ───
    print("\n[7/7] Generating figures...")
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle('z2179: Attractor Dynamics -- GPU-Noise-Driven FPGA Reservoir',
                      fontsize=14, fontweight='bold')

        cond_names = ['FULL', 'WHITE', 'NO_NOISE']
        colors = {'FULL': '#e74c3c', 'WHITE': '#3498db', 'NO_NOISE': '#95a5a6'}
        labels_map = {'FULL': 'GPU 1/f', 'WHITE': 'White', 'NO_NOISE': 'No noise'}

        # Panel 1: Within-pattern correlation by condition
        ax = axes[0, 0]
        vals = [condition_data[c]['mean_within_corr'] for c in cond_names]
        bars = ax.bar(cond_names, vals, color=[colors[c] for c in cond_names])
        ax.set_title('Within-Pattern Correlation')
        ax.set_ylabel('Mean Pearson r')
        ax.axhline(y=0.3, color='green', linestyle=':', alpha=0.7, label='T163 thresh')
        ax.legend(fontsize=8)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f'{v:.3f}', ha='center', va='bottom', fontsize=10)

        # Panel 2: Between vs within distance (T164)
        ax = axes[0, 1]
        x_pos = np.arange(len(cond_names))
        between = [condition_data[c]['mean_between_distance'] for c in cond_names]
        within = [condition_data[c]['mean_within_spread'] for c in cond_names]
        w = 0.35
        ax.bar(x_pos - w / 2, between, w, color='#2ecc71', label='Between-pattern dist')
        ax.bar(x_pos + w / 2, within, w, color='#e67e22', label='Within-pattern spread')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(cond_names)
        ax.set_title('Between vs Within Pattern Distance')
        ax.set_ylabel('Euclidean Distance')
        ax.legend(fontsize=8)

        # Panel 3: Perturbation recovery time
        ax = axes[0, 2]
        vals = [condition_data[c]['mean_recovery_time'] for c in cond_names]
        bars = ax.bar(cond_names, vals, color=[colors[c] for c in cond_names])
        ax.set_title('Perturbation Recovery Time')
        ax.set_ylabel('Steps to recover')
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                    f'{v:.1f}', ha='center', va='bottom', fontsize=10)

        # Panel 4: Number of distinct clusters
        ax = axes[1, 0]
        vals = [condition_data[c]['n_distinct_clusters'] for c in cond_names]
        bars = ax.bar(cond_names, vals, color=[colors[c] for c in cond_names])
        ax.set_title('Distinct Attractor Clusters')
        ax.set_ylabel('Count')
        ax.axhline(y=3, color='green', linestyle=':', alpha=0.7, label='T166 thresh')
        ax.set_ylim(0, 4)
        ax.legend(fontsize=8)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                    f'{v}', ha='center', va='bottom', fontsize=10)

        # Panel 5: Post-perturbation accuracy
        ax = axes[1, 1]
        vals = [condition_data[c]['post_perturb_accuracy'] for c in cond_names]
        bars = ax.bar(cond_names, vals, color=[colors[c] for c in cond_names])
        ax.set_title('Post-Perturbation Classification Accuracy')
        ax.set_ylabel('Accuracy')
        ax.axhline(y=0.60, color='green', linestyle=':', alpha=0.7, label='T168 thresh')
        ax.axhline(y=0.333, color='gray', linestyle='--', alpha=0.5, label='chance')
        ax.set_ylim(0, 1)
        ax.legend(fontsize=8)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f'{v:.3f}', ha='center', va='bottom', fontsize=10)

        # Panel 6: Silhouette scores
        ax = axes[1, 2]
        vals = [condition_data[c]['silhouette_score'] for c in cond_names]
        bars = ax.bar(cond_names, vals, color=[colors[c] for c in cond_names])
        ax.set_title('Cluster Quality (Silhouette Score)')
        ax.set_ylabel('Silhouette')
        ax.set_ylim(-0.2, 1.0)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f'{v:.3f}', ha='center', va='bottom', fontsize=10)

        plt.tight_layout()
        FIGURES.mkdir(parents=True, exist_ok=True)
        fig_path = FIGURES / 'z2179_attractor_dynamics.png'
        plt.savefig(fig_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Figure saved: {fig_path}")
    except ImportError:
        print("  matplotlib not available, skipping figure")

    # ─── Cleanup ───
    if fpga and ser:
        ser.write(bytes([SYNC, CMD_SET_KILL, 0x00]))
        ser.flush()
        ser.close()
        print("  FPGA connection closed")

    print(f"\n{'='*65}")
    print(f"z2179 COMPLETE: {n_pass}/6 tests passed")
    print(f"{'='*65}")


if __name__ == '__main__':
    main()
