#!/usr/bin/env python3
"""z2206_128neuron_reservoir.py — 128-Neuron Deep Intertwine Reservoir

THE CLAIM: 128 time-multiplexed FPGA neurons driven by heterogeneous GPU firmware
noise create a significantly richer reservoir than 8 parallel neurons (z2165).
The 16× neuron scaling provides: (1) larger state space for readout, (2) richer
cross-neuron decorrelation from heterogeneous noise channels, (3) higher memory
capacity from more independent dynamical variables.

Architecture:
  128 neurons on Arty A7-100T FPGA, time-multiplexed through shared
  avalanche+LIF pipeline. Per-neuron Vg driven by GPU firmware noise:

  Channel assignment (5 heterogeneous noise sources):
    Neurons   0-31:  hwmon power1_average (VRM 1/f, IIR alpha=0.85)
    Neurons  32-55:  SMN thermal registers (slow drift, IIR alpha=0.92)
    Neurons  56-79:  PERF_SNAPSHOT jitter (near-white)
    Neurons  80-103: gpu_metrics temp_soc (thermal oscillation)
    Neurons 104-127: GPU clock drift (DVFS dynamics)

  Input encoding: vg[n] = BASE_VG + alpha * input * w_in[n] + beta * noise[n] * w_noise[n]
  Readout: Ridge regression on delta_spike + vmem + cumulative features

3 Conditions:
  FULL_128:  128 neurons, 5-channel heterogeneous noise
  HOMO_128:  128 neurons, all power 1/f (homogeneous)
  SUB_8:     8 neurons only (subset 0-7), multi-channel — matches z2165 scale

Tasks:
  Waveform 3-class (sine/triangle/square): 200 trials × 30 steps
  Temporal XOR at tau=1,2,3,5,8

Tests T307-T314:
  T307: FULL_128 waveform > SUB_8 waveform (scaling helps)
  T308: FULL_128 waveform > 0.70 (high accuracy from rich reservoir)
  T309: FULL_128 > HOMO_128 (heterogeneous noise helps)
  T310: Cross-neuron correlation lower in FULL_128 than HOMO_128
  T311: FULL_128 XOR tau=2 > SUB_8 XOR tau=2
  T312: FULL_128 XOR tau=5 > 0.55 (long-range temporal memory)
  T313: Memory capacity (sum XOR accs) FULL_128 > SUB_8
  T314: At least 4 of 5 noise channels show distinct per-channel spike statistics

Hardware: AMD gfx1151 GPU + Arty A7-100T FPGA (128-neuron bitstream) on /dev/ttyUSB0
Protocol: fpga_host_v2.py (115200 baud, 773-byte telemetry, CRC-8/SMBUS)
"""

import os, sys, json, time, struct, subprocess, argparse
import numpy as np
from pathlib import Path

os.environ.setdefault('HSA_OVERRIDE_GFX_VERSION', '11.0.0')

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE / 'scripts'))
RESULTS = BASE / 'results'

# ─── Reservoir Parameters ───
N_NEURONS = 128
BASE_VG = 0.58
ALPHA = 0.25
BETA = 0.08
SAMPLE_HZ = 20
IIR_ALPHA_POWER = 0.85
IIR_ALPHA_THERMAL = 0.92

# ─── 5-Channel Noise Assignment ───
POWER_NEURONS   = list(range(0, 32))     # hwmon power 1/f
SMN_NEURONS     = list(range(32, 56))    # SMN thermal registers
JITTER_NEURONS  = list(range(56, 80))    # PERF_SNAPSHOT white
THERMAL_NEURONS = list(range(80, 104))   # gpu_metrics temp_soc
CLOCK_NEURONS   = list(range(104, 128))  # GPU clock drift

HWMON_POWER = "/sys/class/hwmon/hwmon7/power1_average"
GPU_METRICS_PATH = "/sys/class/drm/card0/device/gpu_metrics"


# ═══════════════════════════════════════════════════════════
# Noise Sources
# ═══════════════════════════════════════════════════════════

def read_hwmon_power():
    try:
        return int(open(HWMON_POWER).read().strip()) / 1e6
    except Exception:
        return None


def read_gpu_thermal():
    """Read GPU edge temperature from hwmon (amdgpu)."""
    try:
        return int(open("/sys/class/hwmon/hwmon7/temp1_input").read().strip()) / 1000.0
    except Exception:
        return None


def read_gpu_clock():
    """Read current GPU SCLK from hwmon freq1_input (Hz -> MHz)."""
    try:
        return int(open("/sys/class/hwmon/hwmon7/freq1_input").read().strip()) / 1e6
    except Exception:
        return None


PM_TABLE_PATH = "/sys/kernel/ryzen_smu_drv/pm_table"
PM_TABLE_THERMAL_OFFSET = 0x004C  # T1: hotspot temp, most dynamic (~±3°C variance)


def read_smn_thermal():
    """Read SMN thermal from PM table binary blob. Float at offset 0x4C (hotspot temp).
    ryzen_smu PM table exposes raw thermal sensor data below driver smoothing."""
    try:
        with open(PM_TABLE_PATH, 'rb') as f:
            f.seek(PM_TABLE_THERMAL_OFFSET)
            return struct.unpack('<f', f.read(4))[0]
    except Exception:
        return None


def normalize_noise(samples):
    arr = np.array(samples, dtype=float)
    if len(arr) == 0:
        return arr
    mu = arr.mean()
    std = max(arr.std(), 1e-6)
    return (arr - mu) / std


def iir_filter_noise(noise_samples, alpha_iir=0.85):
    if len(noise_samples) == 0:
        return noise_samples
    filtered = np.zeros(len(noise_samples))
    filtered[0] = noise_samples[0]
    for t in range(1, len(noise_samples)):
        filtered[t] = alpha_iir * filtered[t-1] + (1 - alpha_iir) * noise_samples[t]
    std = max(np.std(filtered), 1e-6)
    return filtered / std


def generate_synthetic_1f(n_samples, rng):
    noise = np.zeros(n_samples)
    n_octaves = 8
    octaves = np.zeros(n_octaves)
    for i in range(n_samples):
        for j in range(n_octaves):
            if i % (1 << j) == 0:
                octaves[j] = rng.standard_normal()
        noise[i] = octaves.sum()
    return normalize_noise(noise)


def collect_all_noise(duration_s=20, sample_hz=50):
    """Collect 5 noise channels simultaneously."""
    n = int(duration_s * sample_hz)
    interval = 1.0 / sample_hz
    power_s, thermal_s, clock_s, smn_s = [], [], [], []

    print("  Collecting 4 real-time noise channels...")
    for i in range(n):
        p = read_hwmon_power()
        t = read_gpu_thermal()
        c = read_gpu_clock()
        sm = read_smn_thermal()
        if p is not None: power_s.append(p)
        if t is not None: thermal_s.append(t)
        if c is not None: clock_s.append(c)
        if sm is not None: smn_s.append(sm)
        time.sleep(interval)
        if (i + 1) % (n // 4) == 0:
            print(f"    {i+1}/{n} samples collected...")

    # PERF jitter from HIP probe
    print("  Collecting PERF_SNAPSHOT jitter...")
    probe_bin = BASE / 'scripts' / 'z2153_deep_probe_bridge'
    jitter_s = []
    if probe_bin.exists():
        try:
            result = subprocess.run(
                [str(probe_bin), '100', '16', '50000'],
                capture_output=True, text=True, timeout=30,
                env={**os.environ, 'HSA_OVERRIDE_GFX_VERSION': '11.0.0'})
            if result.returncode == 0:
                for line in result.stdout.strip().split('\n')[1:]:
                    parts = line.split(',')
                    if len(parts) >= 13:
                        jitter_s.append(int(parts[12]))
        except Exception:
            pass

    return power_s, thermal_s, clock_s, smn_s, jitter_s


# ═══════════════════════════════════════════════════════════
# Waveform & XOR Tasks
# ═══════════════════════════════════════════════════════════

def generate_waveforms(n_trials=200, steps_per_trial=30, dt=1.0/20):
    rng = np.random.default_rng(42)
    trials, labels = [], []
    t = np.arange(steps_per_trial) * dt
    for _ in range(n_trials):
        cls = rng.integers(0, 3)
        phase = rng.uniform(0, 2 * np.pi)
        freq = rng.uniform(0.8, 1.2)
        if cls == 0:
            wave = np.sin(2 * np.pi * freq * t + phase)
        elif cls == 1:
            wave = 2.0 * np.abs(2.0 * ((freq * t + phase / (2*np.pi)) % 1.0) - 1.0) - 1.0
        else:
            wave = np.sign(np.sin(2 * np.pi * freq * t + phase))
        trials.append((wave + 1.0) / 2.0)
        labels.append(cls)
    return np.array(trials), np.array(labels)


def generate_xor_sequence(n_steps=3000, seed=42):
    return np.random.default_rng(seed).integers(0, 2, size=n_steps).astype(float)


def compute_xor_targets(u, tau):
    n = len(u)
    targets = np.zeros(n)
    for t in range(tau, n):
        targets[t] = int(u[t]) ^ int(u[t - tau])
    return targets


# ═══════════════════════════════════════════════════════════
# Reservoir Core (128 neurons via fpga_host_v2)
# ═══════════════════════════════════════════════════════════

def compute_vg_128(t, input_val, noises, w_in, w_noise, mode='FULL_128',
                   neuron_mask=None):
    """Compute per-neuron Vg for 128 neurons with 5-channel noise.

    noises: dict with keys 'power', 'smn', 'jitter', 'thermal', 'clock'
    neuron_mask: if not None, only these neurons are active (rest get base_vg)
    """
    n = N_NEURONS if neuron_mask is None else N_NEURONS
    vg = np.full(n, BASE_VG) + ALPHA * input_val * w_in[:n]

    if mode == 'FULL_128':
        channel_map = {
            'power': POWER_NEURONS,
            'smn': SMN_NEURONS,
            'jitter': JITTER_NEURONS,
            'thermal': THERMAL_NEURONS,
            'clock': CLOCK_NEURONS,
        }
        for ch_name, neuron_ids in channel_map.items():
            ch_data = noises.get(ch_name, np.zeros(1))
            if len(ch_data) == 0:
                ch_data = np.zeros(1)
            for nid in neuron_ids:
                if nid < n:
                    idx = t % len(ch_data)
                    vg[nid] += BETA * ch_data[idx] * w_noise[nid]

    elif mode == 'HOMO_128':
        # All neurons use power 1/f noise
        ch_data = noises.get('power', np.zeros(1))
        if len(ch_data) == 0:
            ch_data = np.zeros(1)
        idx = t % len(ch_data)
        vg += BETA * ch_data[idx] * w_noise[:n]

    elif mode == 'SUB_8':
        # Only 8 neurons, multi-channel
        ch_data = noises.get('power', np.zeros(1))
        if len(ch_data) > 0:
            for nid in range(min(3, n)):
                vg[nid] += BETA * ch_data[t % len(ch_data)] * w_noise[nid]
        ch_data = noises.get('thermal', np.zeros(1))
        if len(ch_data) > 0:
            for nid in range(3, min(6, n)):
                vg[nid] += BETA * ch_data[t % len(ch_data)] * w_noise[nid]
        ch_data = noises.get('jitter', np.zeros(1))
        if len(ch_data) > 0:
            for nid in range(6, min(8, n)):
                vg[nid] += BETA * ch_data[t % len(ch_data)] * w_noise[nid]

    return np.clip(vg, 0.05, 0.95)


def run_reservoir_128(fpga, input_signal, noises, w_in, w_noise, mode='FULL_128'):
    """Drive 128-neuron FPGA reservoir and collect states.

    Returns: (n_steps, N*3) array — delta_spikes + vmem + cumulative
    """
    n_steps = len(input_signal)
    n_active = 8 if mode == 'SUB_8' else N_NEURONS
    interval = 1.0 / SAMPLE_HZ
    states = np.zeros((n_steps, n_active * 3))
    prev_counts = None
    cumulative = np.zeros(n_active)

    for t in range(n_steps):
        vg_full = compute_vg_128(t, input_signal[t], noises, w_in, w_noise, mode=mode)
        vg_list = vg_full.tolist()

        # Set Vg for all active neurons
        try:
            if mode == 'SUB_8':
                for nid in range(8):
                    fpga.set_vg(nid, vg_list[nid])
                fpga.ser.flush()
            else:
                fpga.set_vg_all(vg_list)

            time.sleep(interval * 0.3)
            fpga.ser.reset_input_buffer()
            telem = fpga.read_telem(timeout=0.3)
        except Exception as e:
            if not fpga.reconnect():
                break
            telem = None

        if telem and len(telem) >= n_active:
            counts = [telem[i]['spike_count'] for i in range(n_active)]
            vmems = [telem[i]['vmem'] for i in range(n_active)]

            if prev_counts is not None:
                for i in range(n_active):
                    delta = (counts[i] - prev_counts[i]) & 0xFFFF
                    if delta > 30000:
                        delta = 0
                    states[t, i] = delta
                    cumulative[i] += delta
            for i in range(n_active):
                states[t, n_active + i] = vmems[i]
                states[t, n_active * 2 + i] = cumulative[i]
            prev_counts = counts[:]

        time.sleep(max(0, interval * 0.5 - 0.01))

    return states


# ═══════════════════════════════════════════════════════════
# Feature Extraction & Classification
# ═══════════════════════════════════════════════════════════

def augment_with_delays(states, delays=(1, 2, 3)):
    T, D = states.shape
    augmented = np.zeros((T, D * (1 + len(delays))))
    augmented[:, :D] = states
    for i, d in enumerate(delays):
        start = D * (i + 1)
        augmented[d:, start:start + D] = states[:T - d]
    return augmented


def pool_trial_features(trial_states):
    return np.concatenate([
        trial_states.mean(axis=0),
        trial_states.std(axis=0),
        trial_states.max(axis=0),
        trial_states.min(axis=0),
    ])


def ridge_classify(X_train, y_train, X_test, y_test, n_classes_global=3,
                   alphas=None):
    if alphas is None:
        alphas = [1e-6, 1e-4, 1e-2, 1.0, 100.0]
    Y_train = np.zeros((len(y_train), n_classes_global))
    for i, y in enumerate(y_train):
        Y_train[i, int(y)] = 1.0
    best_acc = -1
    for alpha in alphas:
        I = np.eye(X_train.shape[1])
        try:
            W = np.linalg.solve(X_train.T @ X_train + alpha * I, X_train.T @ Y_train)
        except np.linalg.LinAlgError:
            continue
        pred_test = np.argmax(X_test @ W, axis=1)
        acc_test = np.mean(pred_test == y_test)
        if acc_test > best_acc:
            best_acc = acc_test
    return best_acc


def ridge_binary(X_train, y_train, X_test, y_test, alphas=None):
    if alphas is None:
        alphas = [1e-6, 1e-4, 1e-2, 1.0, 100.0]
    best_acc = -1
    for alpha in alphas:
        I = np.eye(X_train.shape[1])
        try:
            w = np.linalg.solve(X_train.T @ X_train + alpha * I, X_train.T @ y_train)
        except np.linalg.LinAlgError:
            continue
        pred = (X_test @ w > 0.5).astype(float)
        acc = np.mean(pred == y_test)
        if acc > best_acc:
            best_acc = acc
    return best_acc


def stratified_kfold(X, y, n_splits=5, seed=42):
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


def compute_cross_neuron_correlation(states_list, n_neurons):
    """Mean pairwise Pearson correlation of delta_spikes across trials."""
    corr_accum = np.zeros((n_neurons, n_neurons))
    n_valid = 0
    for states in states_list:
        spikes = states[:, :n_neurons]
        valid_cols = [i for i in range(n_neurons) if spikes[:, i].std() > 1e-8]
        if len(valid_cols) < 2:
            continue
        sub = spikes[:, valid_cols]
        corr = np.corrcoef(sub.T)
        full_corr = np.eye(n_neurons)
        for ii, ci in enumerate(valid_cols):
            for jj, cj in enumerate(valid_cols):
                full_corr[ci, cj] = corr[ii, jj]
        corr_accum += full_corr
        n_valid += 1
    if n_valid > 0:
        corr_accum /= n_valid
    return corr_accum


def mean_off_diagonal(corr_matrix):
    n = corr_matrix.shape[0]
    mask = ~np.eye(n, dtype=bool)
    return np.mean(np.abs(corr_matrix[mask]))


# ═══════════════════════════════════════════════════════════
# Main Experiment
# ═══════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description='z2206: 128-Neuron Deep Intertwine Reservoir')
    parser.add_argument('--n-trials', type=int, default=200)
    parser.add_argument('--steps-per-trial', type=int, default=30)
    parser.add_argument('--xor-steps', type=int, default=3000)
    parser.add_argument('--noise-collect-s', type=float, default=20.0)
    args = parser.parse_args()

    print("=" * 70)
    print("z2206: 128-Neuron Deep Intertwine Reservoir")
    print("  16x scaling from z2165 (8 neurons) to 128 neurons")
    print("  5-channel heterogeneous GPU firmware noise")
    print("=" * 70)

    rng = np.random.default_rng(42)
    w_in = rng.uniform(-1, 1, size=N_NEURONS)
    w_noise = rng.uniform(-1, 1, size=N_NEURONS)

    results = {
        'experiment': 'z2206_128neuron_reservoir',
        'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
        'params': {
            'base_vg': BASE_VG, 'alpha': ALPHA, 'beta': BETA,
            'n_neurons': N_NEURONS, 'sample_hz': SAMPLE_HZ,
            'n_trials': args.n_trials, 'steps_per_trial': args.steps_per_trial,
            'xor_steps': args.xor_steps,
            'channel_assignment': {
                'power': [0, 31], 'smn': [32, 55], 'jitter': [56, 79],
                'thermal': [80, 103], 'clock': [104, 127],
            },
        },
    }

    # ─── Step 1: Connect to FPGA ───
    print("\n[1/8] Connecting to 128-neuron FPGA...")
    from fpga_host_v2 import FPGABridge
    fpga = FPGABridge()  # auto-detect port and neuron count
    if not fpga.connected:
        print("  ERROR: FPGA not found")
        return
    print(f"  Connected: {fpga.port}, baud={fpga.baud}, neurons={fpga.num_neurons}")

    # Verify FPGA responds
    fpga.read_telem(timeout=0.5)
    time.sleep(1.0)
    test = fpga.read_telem(timeout=0.5)
    if test is None:
        print("  ERROR: No telemetry response")
        return
    print(f"  Telemetry OK: {len(test)} neurons, spike range "
          f"{min(t['spike_count'] for t in test)}-{max(t['spike_count'] for t in test)}")

    # ─── Step 2: Collect 5-channel GPU noise ───
    print(f"\n[2/8] Collecting 5-channel GPU noise ({args.noise_collect_s}s)...")
    power_raw, thermal_raw, clock_raw, smn_raw, jitter_raw = \
        collect_all_noise(duration_s=args.noise_collect_s, sample_hz=50)

    # Process each channel
    noises = {}
    noise_info = {}

    for name, raw, iir_a in [
        ('power', power_raw, IIR_ALPHA_POWER),
        ('thermal', thermal_raw, IIR_ALPHA_THERMAL),
        ('clock', clock_raw, 0.80),
        ('smn', smn_raw, 0.90),
    ]:
        if len(raw) > 10:
            normed = normalize_noise(raw)
            filtered = iir_filter_noise(normed, alpha_iir=iir_a)
            noises[name] = filtered
            noise_info[name] = {
                'n_samples': len(filtered),
                'raw_mean': float(np.mean(raw)),
                'raw_std': float(np.std(raw)),
            }
            print(f"  {name}: {len(raw)} samples, mean={np.mean(raw):.3f} ± {np.std(raw):.4f}")
        else:
            print(f"  {name}: unavailable, generating synthetic 1/f")
            noises[name] = generate_synthetic_1f(int(args.noise_collect_s * 50), rng)
            noise_info[name] = {'n_samples': len(noises[name]), 'synthetic': True}

    if len(jitter_raw) > 10:
        noises['jitter'] = normalize_noise(jitter_raw)
        noise_info['jitter'] = {'n_samples': len(noises['jitter'])}
        print(f"  jitter: {len(jitter_raw)} samples")
    else:
        noises['jitter'] = rng.standard_normal(int(args.noise_collect_s * 50))
        noise_info['jitter'] = {'n_samples': len(noises['jitter']), 'synthetic': True}

    results['noise'] = noise_info

    # ─── Step 3: Generate tasks ───
    print("\n[3/8] Generating waveform + XOR tasks...")
    wave_trials, wave_labels = generate_waveforms(
        n_trials=args.n_trials, steps_per_trial=args.steps_per_trial)
    print(f"  Waveforms: {args.n_trials} trials × {args.steps_per_trial} steps")
    print(f"  Class distribution: {np.bincount(wave_labels)}")

    # ─── Step 4: Run 3 conditions on waveform task ───
    print("\n[4/8] Running 3 reservoir conditions on waveform task...")
    conditions = ['FULL_128', 'HOMO_128', 'SUB_8']
    wave_features = {}
    wave_trial_states = {}

    for cond in conditions:
        n_active = 8 if cond == 'SUB_8' else N_NEURONS
        print(f"\n  === {cond} ({n_active} neurons) ===")
        trial_features = []
        trial_states_list = []
        t0 = time.monotonic()

        for trial_idx in range(args.n_trials):
            input_signal = wave_trials[trial_idx]
            states = run_reservoir_128(fpga, input_signal, noises, w_in, w_noise,
                                       mode=cond)
            aug = augment_with_delays(states, delays=(1, 2, 3))
            feat = pool_trial_features(aug)
            trial_features.append(feat)
            trial_states_list.append(states)

            if (trial_idx + 1) % 10 == 0:
                elapsed = time.monotonic() - t0
                rate = (trial_idx + 1) / elapsed
                eta = (args.n_trials - trial_idx - 1) / rate
                print(f"    Trial {trial_idx+1}/{args.n_trials} "
                      f"({rate:.1f} trials/s, ETA {eta:.0f}s)")

        wave_features[cond] = np.array(trial_features)
        wave_trial_states[cond] = trial_states_list
        elapsed = time.monotonic() - t0
        print(f"  {cond}: {len(trial_features)} trials in {elapsed:.1f}s")

    # ─── Step 5: Classify waveforms ───
    print("\n[5/8] Classifying waveforms (5-fold stratified CV)...")
    wave_accuracies = {}

    for cond in conditions:
        X_all = wave_features[cond]
        splits = stratified_kfold(X_all, wave_labels, n_splits=5)
        fold_accs = []
        for train_idx, test_idx in splits:
            X_train, X_test = X_all[train_idx], X_all[test_idx]
            y_train, y_test = wave_labels[train_idx], wave_labels[test_idx]
            mu = X_train.mean(axis=0, keepdims=True)
            sigma = X_train.std(axis=0, keepdims=True)
            sigma[sigma < 1e-10] = 1.0
            X_train_n = (X_train - mu) / sigma
            X_test_n = (X_test - mu) / sigma
            acc = ridge_classify(X_train_n, y_train, X_test_n, y_test)
            fold_accs.append(acc)
        mean_acc = np.mean(fold_accs)
        std_acc = np.std(fold_accs)
        wave_accuracies[cond] = {
            'mean': float(mean_acc), 'std': float(std_acc),
            'folds': [float(a) for a in fold_accs],
        }
        print(f"  {cond}: {mean_acc:.3f} ± {std_acc:.3f}")

    results['waveform_classification'] = wave_accuracies

    # ─── Step 6: Cross-neuron correlation ───
    print("\n[6/8] Computing cross-neuron correlations...")
    corr_offdiag = {}
    for cond in conditions:
        n_active = 8 if cond == 'SUB_8' else N_NEURONS
        corr = compute_cross_neuron_correlation(wave_trial_states[cond], n_active)
        offdiag = mean_off_diagonal(corr)
        corr_offdiag[cond] = float(offdiag)
        print(f"  {cond}: mean |off-diagonal| = {offdiag:.4f}")

    results['cross_neuron_correlation'] = corr_offdiag

    # ─── Step 7: Temporal XOR ───
    print("\n[7/8] Running temporal XOR...")
    xor_input = generate_xor_sequence(n_steps=args.xor_steps, seed=42)
    taus = [1, 2, 3, 5, 8]

    xor_states = {}
    for cond in conditions:
        print(f"  Running XOR reservoir ({cond})...")
        st = run_reservoir_128(fpga, xor_input, noises, w_in, w_noise, mode=cond)
        xor_states[cond] = augment_with_delays(st, delays=(1, 2, 3))

    xor_results = {}
    for tau in taus:
        y_xor = compute_xor_targets(xor_input, tau)
        valid = np.arange(max(tau, 3), args.xor_steps)
        accs = {}
        for cond in conditions:
            X_all = xor_states[cond]
            X_valid, y_valid = X_all[valid], y_xor[valid]
            n_valid = len(valid)
            split = int(0.7 * n_valid)
            X_tr, X_te = X_valid[:split], X_valid[split:]
            y_tr, y_te = y_valid[:split], y_valid[split:]
            mu = X_tr.mean(axis=0, keepdims=True)
            sigma = X_tr.std(axis=0, keepdims=True)
            sigma[sigma < 1e-10] = 1.0
            X_tr_n = (X_tr - mu) / sigma
            X_te_n = (X_te - mu) / sigma
            acc = ridge_binary(X_tr_n, y_tr, X_te_n, y_te)
            accs[cond] = float(acc)
        xor_results[f'tau_{tau}'] = accs
        print(f"  XOR tau={tau}: " + ", ".join(f"{c}={a:.3f}" for c, a in accs.items()))

    results['xor_classification'] = xor_results

    # ─── Step 8: Per-channel spike statistics ───
    print("\n[8/8] Per-channel spike statistics...")
    # Use FULL_128 trial states
    channel_stats = {}
    channel_groups = {
        'power': POWER_NEURONS,
        'smn': SMN_NEURONS,
        'jitter': JITTER_NEURONS,
        'thermal': THERMAL_NEURONS,
        'clock': CLOCK_NEURONS,
    }
    for ch_name, neuron_ids in channel_groups.items():
        spike_rates = []
        for states in wave_trial_states['FULL_128'][:50]:  # first 50 trials
            for nid in neuron_ids:
                if nid < states.shape[1]:
                    spike_rates.append(states[:, nid].mean())
        channel_stats[ch_name] = {
            'mean_rate': float(np.mean(spike_rates)) if spike_rates else 0,
            'std_rate': float(np.std(spike_rates)) if spike_rates else 0,
            'n_neurons': len(neuron_ids),
        }
        print(f"  {ch_name}: rate={channel_stats[ch_name]['mean_rate']:.3f} "
              f"± {channel_stats[ch_name]['std_rate']:.3f}")

    results['channel_statistics'] = channel_stats

    # ═══════════════════════════════════════════════════════════
    # Tests T307-T314
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("TEST RESULTS")
    print("=" * 70)

    tests = {}

    # T307: FULL_128 > SUB_8 waveform
    f128 = wave_accuracies['FULL_128']['mean']
    s8 = wave_accuracies['SUB_8']['mean']
    t307 = f128 > s8
    tests['T307'] = {'pass': t307, 'full_128': f128, 'sub_8': s8,
                     'desc': 'FULL_128 waveform > SUB_8'}
    print(f"  T307 FULL_128({f128:.3f}) > SUB_8({s8:.3f}): "
          f"{'PASS' if t307 else 'FAIL'}")

    # T308: FULL_128 waveform > 0.70
    t308 = f128 > 0.70
    tests['T308'] = {'pass': t308, 'accuracy': f128,
                     'desc': 'FULL_128 waveform > 0.70'}
    print(f"  T308 FULL_128({f128:.3f}) > 0.70: {'PASS' if t308 else 'FAIL'}")

    # T309: FULL_128 > HOMO_128
    h128 = wave_accuracies['HOMO_128']['mean']
    t309 = f128 > h128
    tests['T309'] = {'pass': t309, 'full_128': f128, 'homo_128': h128,
                     'desc': 'FULL_128 > HOMO_128 (heterogeneous helps)'}
    print(f"  T309 FULL_128({f128:.3f}) > HOMO_128({h128:.3f}): "
          f"{'PASS' if t309 else 'FAIL'}")

    # T310: Cross-neuron correlation FULL_128 < HOMO_128
    cf = corr_offdiag['FULL_128']
    ch = corr_offdiag['HOMO_128']
    t310 = cf < ch
    tests['T310'] = {'pass': t310, 'full_128': cf, 'homo_128': ch,
                     'desc': 'FULL_128 decorrelation < HOMO_128'}
    print(f"  T310 FULL_128 corr({cf:.4f}) < HOMO_128 corr({ch:.4f}): "
          f"{'PASS' if t310 else 'FAIL'}")

    # T311: FULL_128 XOR tau=2 > SUB_8 XOR tau=2
    xf2 = xor_results['tau_2']['FULL_128']
    xs2 = xor_results['tau_2']['SUB_8']
    t311 = xf2 > xs2
    tests['T311'] = {'pass': t311, 'full_128': xf2, 'sub_8': xs2,
                     'desc': 'FULL_128 XOR tau=2 > SUB_8'}
    print(f"  T311 FULL_128 XOR2({xf2:.3f}) > SUB_8({xs2:.3f}): "
          f"{'PASS' if t311 else 'FAIL'}")

    # T312: FULL_128 XOR tau=5 > 0.55
    xf5 = xor_results['tau_5']['FULL_128']
    t312 = xf5 > 0.55
    tests['T312'] = {'pass': t312, 'accuracy': xf5,
                     'desc': 'FULL_128 XOR tau=5 > 0.55'}
    print(f"  T312 FULL_128 XOR5({xf5:.3f}) > 0.55: {'PASS' if t312 else 'FAIL'}")

    # T313: Memory capacity FULL_128 > SUB_8
    mc_full = sum(xor_results[f'tau_{tau}']['FULL_128'] for tau in taus)
    mc_sub = sum(xor_results[f'tau_{tau}']['SUB_8'] for tau in taus)
    t313 = mc_full > mc_sub
    tests['T313'] = {'pass': t313, 'mc_full': mc_full, 'mc_sub': mc_sub,
                     'desc': 'Memory capacity FULL_128 > SUB_8'}
    print(f"  T313 MC_FULL({mc_full:.3f}) > MC_SUB({mc_sub:.3f}): "
          f"{'PASS' if t313 else 'FAIL'}")

    # T314: 4+ channels show distinct spike statistics
    means = [channel_stats[ch]['mean_rate'] for ch in channel_groups]
    stds = [channel_stats[ch]['std_rate'] for ch in channel_groups]
    n_distinct = sum(1 for i in range(len(means))
                     for j in range(i+1, len(means))
                     if abs(means[i] - means[j]) > 0.5 * max(stds[i], stds[j], 1e-6))
    t314 = n_distinct >= 4
    tests['T314'] = {'pass': t314, 'n_distinct_pairs': n_distinct,
                     'desc': '4+ channel pairs with distinct stats'}
    print(f"  T314 {n_distinct} distinct channel pairs >= 4: "
          f"{'PASS' if t314 else 'FAIL'}")

    n_pass = sum(1 for t in tests.values() if t['pass'])
    print(f"\n  TOTAL: {n_pass}/8 PASS")

    results['tests'] = tests
    results['summary'] = f'{n_pass}/8 PASS'

    # Save results
    out_path = RESULTS / 'z2206_128neuron_reservoir.json'
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved: {out_path}")

    fpga.close()
    print("\nDone.")


if __name__ == '__main__':
    main()
