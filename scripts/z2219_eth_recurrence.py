#!/usr/bin/env python3
"""z2219_eth_recurrence.py — High-Speed Ethernet Recurrence Sweep

z2217/z2218 FAILURE ANALYSIS:
  At UART 20 Hz (50ms/step), recurrent state decays too fast:
    LEAK_RATE=0.3 → 0.3^5 = 0.002 after 5 steps (250ms)
    LEAK_RATE=0.8 → 0.8^5 = 0.328 after 5 steps (250ms) — still marginal
  Result: ALL recurrence conditions near chance (0.47-0.57)

FIX: UDP Ethernet bridge — 0.82ms round-trip → 200 Hz loop rate
  At 200 Hz, LEAK_RATE=0.80, 5 steps = 25ms:
    0.8^5 = 0.328 — same math but happens in 25ms, not 250ms
  At 200 Hz, LEAK_RATE=0.80, 50 steps = 250ms:
    0.8^50 = 1.4e-5 — same total window, 10× more recurrent updates!
  KEY: More updates in same physical time = more state accumulation

SWEEP: sample_hz × leak_rate × alpha_rec for XOR τ=2,3,5
  sample_hz: [50, 100, 200, 500] — how fast we drive the loop
  leak_rate: [0.80, 0.90, 0.95] — state persistence between updates
  alpha_rec: [0.10, 0.20] — recurrent gain (calibrated from z2217)

Hardware: AMD gfx1151 GPU + Arty A7-100T FPGA (128-neuron, UDP Ethernet)
"""

import os, sys, json, time, struct
import numpy as np
from pathlib import Path

os.environ.setdefault('HSA_OVERRIDE_GFX_VERSION', '11.0.0')

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE / 'scripts'))
RESULTS = BASE / 'results'

N_NEURONS    = 128
BASE_VG      = 0.58
ALPHA_IN     = 0.25
BETA_1F      = 0.08
SPECTRAL_RAD = 0.90
SPARSITY     = 0.10

# Ethernet-enabled: much faster loop
N_TRIALS     = 40       # per condition
N_STEPS      = 60       # per trial

HWMON_POWER = "/sys/class/hwmon/hwmon7/power1_average"
PM_TABLE_PATH = "/sys/kernel/ryzen_smu_drv/pm_table"

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer): return int(obj)
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        if isinstance(obj, np.bool_): return bool(obj)
        return super().default(obj)

def read_hwmon_power():
    try: return int(open(HWMON_POWER).read().strip()) / 1e6
    except: return None

def read_smn_thermal():
    try:
        with open(PM_TABLE_PATH, 'rb') as f:
            f.seek(0x004C)
            return struct.unpack('<f', f.read(4))[0]
    except: return None

def read_perf_jitter():
    t0 = time.perf_counter_ns()
    _ = os.getpid()
    return time.perf_counter_ns() - t0

def normalize_noise(samples):
    arr = np.array(samples, dtype=float)
    if len(arr) == 0: return arr
    mu, std = arr.mean(), max(arr.std(), 1e-6)
    return (arr - mu) / std

def iir_filter_noise(noise_samples, alpha_iir=0.85):
    if len(noise_samples) == 0: return noise_samples
    filtered = np.zeros(len(noise_samples))
    filtered[0] = noise_samples[0]
    for t in range(1, len(noise_samples)):
        filtered[t] = alpha_iir * filtered[t-1] + (1 - alpha_iir) * noise_samples[t]
    std = max(np.std(filtered), 1e-6)
    return filtered / std

def collect_noise_fast(duration_s=10, sample_hz=100):
    n = int(duration_s * sample_hz)
    interval = 1.0 / sample_hz
    power_s, smn_s, jitter_s = [], [], []
    print("  Collecting noise channels...")
    for i in range(n):
        p = read_hwmon_power()
        sm = read_smn_thermal()
        j = read_perf_jitter()
        if p is not None: power_s.append(p)
        if sm is not None: smn_s.append(sm)
        jitter_s.append(j)
        time.sleep(interval)
        if n > 4 and (i + 1) % (n // 4) == 0:
            print(f"    {i+1}/{n} samples")
    return power_s, smn_s, jitter_s

def create_recurrent_matrix(n_neurons, sparsity, spectral_radius, seed=42):
    rng = np.random.default_rng(seed)
    W = rng.standard_normal((n_neurons, n_neurons))
    mask = rng.random((n_neurons, n_neurons)) < sparsity
    np.fill_diagonal(mask, False)
    W *= mask
    eigenvalues = np.linalg.eigvals(W)
    max_eig = np.max(np.abs(eigenvalues))
    if max_eig > 0:
        W = W * (spectral_radius / max_eig)
    return W

def ridge_classify(X_tr, y_tr, X_te, y_te, n_classes=2):
    alphas = [1e-4, 1e-3, 1e-2, 0.1, 1.0, 10.0, 100.0]
    mu = X_tr.mean(axis=0); sigma = X_tr.std(axis=0)
    sigma[sigma < 1e-2] = 1.0
    X_tr_s = (X_tr - mu) / sigma
    X_te_s = (X_te - mu) / sigma
    Y_tr = np.zeros((len(y_tr), n_classes))
    for i, y in enumerate(y_tr): Y_tr[i, int(y)] = 1.0
    best = -1
    for a in alphas:
        I = np.eye(X_tr_s.shape[1])
        try: W = np.linalg.solve(X_tr_s.T @ X_tr_s + a * I, X_tr_s.T @ Y_tr)
        except: continue
        acc = np.mean(np.argmax(X_te_s @ W, axis=1) == y_te)
        if acc > best: best = acc
    return best

def generate_temporal_xor(n_trials, steps, tau, seed=42):
    rng = np.random.default_rng(seed)
    trials, labels = [], []
    for _ in range(n_trials):
        seq = rng.integers(0, 2, size=steps).astype(float)
        target = np.zeros(steps, dtype=int)
        for t_i in range(tau, steps):
            target[t_i] = int(seq[t_i]) ^ int(seq[t_i - tau])
        trials.append(seq)
        labels.append(target)
    return np.array(trials), np.array(labels)

def generate_waveforms(n_trials, steps, sample_hz, seed=42):
    rng = np.random.default_rng(seed)
    dt = 1.0 / sample_hz
    t = np.arange(steps) * dt
    trials, labels = [], []
    for _ in range(n_trials):
        cls = rng.integers(0, 7)
        phase = rng.uniform(0, 2 * np.pi)
        freq = rng.uniform(0.8, 1.2)
        if cls == 0:    wave = np.sin(2 * np.pi * freq * t + phase)
        elif cls == 1:  wave = 2.0 * np.abs(2.0 * ((freq * t + phase/(2*np.pi)) % 1.0) - 1.0) - 1.0
        elif cls == 2:  wave = np.sign(np.sin(2 * np.pi * freq * t + phase))
        elif cls == 3:  wave = 2.0 * ((freq * t + phase/(2*np.pi)) % 1.0) - 1.0
        elif cls == 4:
            f0, f1 = freq * 0.5, freq * 2.0
            inst_f = f0 + (f1 - f0) * t / max(t[-1], 1e-6)
            wave = np.sin(2 * np.pi * np.cumsum(inst_f) * dt + phase)
        elif cls == 5:
            carrier = np.sin(2 * np.pi * freq * 2 * t + phase)
            envelope = 0.5 + 0.5 * np.sin(2 * np.pi * freq * 0.3 * t)
            wave = carrier * envelope
        else:
            decay = np.exp(-2.0 * t)
            wave = np.sin(2 * np.pi * freq * t + phase) * decay
        wave = (wave - wave.min()) / max(wave.max() - wave.min(), 1e-6)
        trials.append(wave)
        labels.append(cls)
    return np.array(trials), np.array(labels)

# Noise channel assignment (128 neurons → 5 channels)
POWER_NEURONS   = list(range(0, 32))
SMN_NEURONS     = list(range(32, 56))
JITTER_NEURONS  = list(range(56, 80))
THERMAL_NEURONS = list(range(80, 104))
CLOCK_NEURONS   = list(range(104, 128))

def run_trial_eth(fpga, input_signal, noises, w_in, w_noise, W_res,
                  alpha_rec, leak_rate, gamma_mod, sample_hz,
                  mode='NO_REC'):
    """Run one trial via Ethernet at target sample_hz.

    KEY DIFFERENCE from z2217/z2218:
    - Uses FPGAEthBridge.set_vg_batch() + read_telemetry_fast()
    - No serial buffer flush needed
    - Loop runs at sample_hz (50-500 Hz) instead of 20 Hz
    """
    n_steps = len(input_signal)
    spike_out = np.zeros((n_steps, N_NEURONS))
    h_out = np.zeros((n_steps, N_NEURONS))

    prev_counts = None
    h = np.zeros(N_NEURONS)
    interval = 1.0 / sample_hz

    channel_assignment = {
        'power': POWER_NEURONS, 'smn': SMN_NEURONS,
        'jitter': JITTER_NEURONS, 'thermal': THERMAL_NEURONS,
        'clock': CLOCK_NEURONS,
    }

    vg_min_seen = 1.0
    vg_max_seen = 0.0
    h_norms = []
    actual_rates = []

    for t in range(n_steps):
        t_start = time.perf_counter()

        inp = input_signal[t]
        vg = np.full(N_NEURONS, BASE_VG) + ALPHA_IN * inp * w_in

        # Recurrent feedback
        if mode != 'NO_REC':
            vg += alpha_rec * h

        # Firmware noise injection for FW_REC
        if mode == 'FW_REC':
            for ch_name, neuron_ids in channel_assignment.items():
                ch_data = noises.get(ch_name, np.zeros(1))
                if len(ch_data) == 0: continue
                idx = t % len(ch_data)
                for nid in neuron_ids:
                    vg[nid] += BETA_1F * ch_data[idx] * w_noise[nid]

        vg = np.clip(vg, 0.05, 0.95)
        vg_min_seen = min(vg_min_seen, float(vg.min()))
        vg_max_seen = max(vg_max_seen, float(vg.max()))

        # Send Vg batch via Ethernet (fire-and-forget UDP, ~0.1ms)
        fpga.set_vg_batch(0, vg.tolist())

        # Brief integration time — neurons need a few ms to respond
        # At 200 Hz → 5ms total per step, use ~2ms integration
        integration_time = max(0.001, interval * 0.4)
        time.sleep(integration_time)

        # Read telemetry via Ethernet (~0.8ms round-trip)
        try:
            counts, vmem, bvpar = fpga.read_telemetry_fast()
        except (TimeoutError, Exception):
            spike_out[t] = 0
            h_out[t] = h
            continue

        # Extract spike deltas
        spike_deltas = np.zeros(N_NEURONS)
        if prev_counts is not None:
            for i in range(N_NEURONS):
                delta = (int(counts[i]) - int(prev_counts[i])) & 0xFFFF
                if delta > 30000: delta = 0
                spike_deltas[i] = delta
        prev_counts = counts.copy()

        spike_out[t] = spike_deltas

        # Update echo state
        if mode != 'NO_REC':
            s_total = max(spike_deltas.sum(), 1.0)
            s_norm = spike_deltas / np.sqrt(s_total)

            if mode == 'FW_REC':
                W_eff = W_res.copy()
                for ch_name, neuron_ids in channel_assignment.items():
                    ch_data = noises.get(ch_name, np.zeros(1))
                    if len(ch_data) > 0:
                        eta = ch_data[t % len(ch_data)]
                    else:
                        eta = 0.0
                    for j in neuron_ids:
                        W_eff[:, j] *= (1.0 + gamma_mod * eta)
                h_raw = np.tanh(W_eff @ s_norm)
            else:
                h_raw = np.tanh(W_res @ s_norm)

            h = leak_rate * h + (1 - leak_rate) * h_raw
            h_norms.append(float(np.linalg.norm(h)))

        h_out[t] = h

        # Pace to target rate
        elapsed = time.perf_counter() - t_start
        remaining = interval - elapsed
        if remaining > 0.0005:
            time.sleep(remaining)
        actual_rates.append(1.0 / max(time.perf_counter() - t_start, 1e-6))

    stability = {
        'vg_min': float(vg_min_seen),
        'vg_max': float(vg_max_seen),
        'vg_stable': bool(vg_min_seen > 0.08 and vg_max_seen < 0.92),
        'h_norm_mean': float(np.mean(h_norms)) if h_norms else 0.0,
        'actual_hz': float(np.mean(actual_rates)) if actual_rates else 0.0,
        'h_bounded': bool(max(h_norms) < 10.0) if h_norms else True,
    }

    return spike_out, h_out, stability

def evaluate_xor(fpga, noises, w_in, w_noise, W_res, alpha_rec, leak_rate,
                 gamma_mod, sample_hz, tau, mode, n_trials=N_TRIALS, n_steps=N_STEPS):
    """XOR τ evaluation at given sample rate."""
    inputs, targets = generate_temporal_xor(n_trials, n_steps, tau)

    all_X, all_y = [], []
    for trial in range(n_trials):
        spikes, h_states, stab = run_trial_eth(
            fpga, inputs[trial], noises, w_in, w_noise, W_res,
            alpha_rec, leak_rate, gamma_mod, sample_hz, mode=mode)

        for t_i in range(tau, n_steps):
            feat = np.concatenate([spikes[t_i], h_states[t_i]])
            all_X.append(feat)
            all_y.append(targets[trial][t_i])

        if trial == 0:
            print(f"      actual_hz={stab['actual_hz']:.0f}, "
                  f"h_norm={stab['h_norm_mean']:.3f}, "
                  f"vg=[{stab['vg_min']:.2f},{stab['vg_max']:.2f}]")

    X = np.array(all_X)
    y = np.array(all_y)

    # 70/30 split
    n = len(y)
    idx = np.random.default_rng(42).permutation(n)
    split = int(0.7 * n)
    acc = ridge_classify(X[idx[:split]], y[idx[:split]], X[idx[split:]], y[idx[split:]])
    return acc

def evaluate_waveform(fpga, noises, w_in, w_noise, W_res, alpha_rec, leak_rate,
                      gamma_mod, sample_hz, mode, n_trials=60, n_steps=N_STEPS):
    """7-class waveform classification."""
    inputs, labels = generate_waveforms(n_trials, n_steps, sample_hz)

    all_feats = []
    for trial in range(n_trials):
        spikes, h_states, _ = run_trial_eth(
            fpga, inputs[trial], noises, w_in, w_noise, W_res,
            alpha_rec, leak_rate, gamma_mod, sample_hz, mode=mode)

        feat = np.concatenate([
            spikes.mean(axis=0), spikes.std(axis=0),
            h_states.mean(axis=0), h_states.std(axis=0),
        ])
        all_feats.append(feat)

        if (trial + 1) % 20 == 0:
            print(f"      trial {trial+1}/{n_trials}")

    X = np.array(all_feats)

    # 70/30 split
    n = len(labels)
    idx = np.random.default_rng(42).permutation(n)
    split = int(0.7 * n)
    acc = ridge_classify(X[idx[:split]], labels[idx[:split]],
                         X[idx[split:]], labels[idx[split:]], n_classes=7)
    return acc


def main():
    from fpga_host_eth import FPGAEthBridge

    print("=" * 70)
    print("z2219: High-Speed Ethernet Recurrence Sweep")
    print("  z2217/z2218: UART 20 Hz → all near chance (0.47-0.57)")
    print("  THIS: Ethernet 50-500 Hz → recurrent state persists!")
    print("  Benchmark: 0.82ms mean round-trip, max 1224 Hz")
    print("=" * 70)

    # ─── Connect FPGA via Ethernet ───
    print("\n[1] Connecting to FPGA via Ethernet...")
    fpga = FPGAEthBridge()
    if not fpga.connect():
        print("  FATAL: Cannot connect to FPGA at 192.168.0.50:7700")
        return
    print(f"  Connected: {fpga.fpga_ip}:{fpga.fpga_port}, {fpga.num_neurons} neurons")

    # ─── Collect firmware noise ───
    print("\n[2] Collecting firmware noise (10s at 100 Hz)...")
    power_raw, smn_raw, jitter_raw = collect_noise_fast(10, 100)
    noises = {
        'power': iir_filter_noise(normalize_noise(power_raw)),
        'smn': iir_filter_noise(normalize_noise(smn_raw)),
        'jitter': iir_filter_noise(normalize_noise(jitter_raw)),
        'thermal': iir_filter_noise(normalize_noise(smn_raw)),  # alias
        'clock': iir_filter_noise(normalize_noise(jitter_raw)),  # alias
    }
    for k, v in noises.items():
        print(f"  {k}: {len(v)} samples")

    # ─── Create matrices ───
    print("\n[3] Creating recurrent weight matrix...")
    rng = np.random.default_rng(42)
    w_in = rng.standard_normal(N_NEURONS)
    w_in /= np.linalg.norm(w_in)
    w_noise = rng.standard_normal(N_NEURONS)
    w_noise /= np.linalg.norm(w_noise)
    W_res = create_recurrent_matrix(N_NEURONS, SPARSITY, SPECTRAL_RAD, seed=42)
    print(f"  W_res: ρ={SPECTRAL_RAD}, sparsity={SPARSITY}")

    results = {}

    # ═══════════════════════════════════════════════════════════════
    # PHASE 1: Rate Sweep — XOR τ=2 baseline at different speeds
    # ═══════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("PHASE 1: Sample Rate Effect on Baseline (NO_REC, XOR τ=2)")
    print("=" * 70)

    for hz in [20, 50, 100, 200]:
        t0 = time.time()
        acc = evaluate_xor(fpga, noises, w_in, w_noise, W_res,
                           alpha_rec=0, leak_rate=0, gamma_mod=0,
                           sample_hz=hz, tau=2, mode='NO_REC')
        dt = time.time() - t0
        print(f"  NO_REC @ {hz:4d} Hz: {acc:.3f}  ({dt:.0f}s)")
        results[f'P1_norec_{hz}hz_tau2'] = acc

    # ═══════════════════════════════════════════════════════════════
    # PHASE 2: Recurrence at High Speed — α × λ sweep
    # ═══════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("PHASE 2: Recurrence Sweep @ 200 Hz (XOR τ=2)")
    print("=" * 70)

    hz_fast = 200
    baseline_200 = results.get(f'P1_norec_{hz_fast}hz_tau2', 0.50)

    for alpha_rec in [0.10, 0.20]:
        for leak in [0.80, 0.90, 0.95]:
            t0 = time.time()
            acc = evaluate_xor(fpga, noises, w_in, w_noise, W_res,
                               alpha_rec=alpha_rec, leak_rate=leak, gamma_mod=0.20,
                               sample_hz=hz_fast, tau=2, mode='STATIC_REC')
            dt = time.time() - t0
            delta = acc - baseline_200
            tag = ">>>" if delta > 0.03 else "   "
            print(f"  {tag} α={alpha_rec:.2f} λ={leak:.2f}: {acc:.3f} "
                  f"(Δ={delta:+.3f}, {dt:.0f}s)")
            results[f'P2_static_a{alpha_rec:.2f}_l{leak:.2f}_tau2'] = acc

    # Find best alpha/leak from phase 2
    best_alpha, best_leak, best_acc = 0.10, 0.90, 0
    for alpha_rec in [0.10, 0.20]:
        for leak in [0.80, 0.90, 0.95]:
            key = f'P2_static_a{alpha_rec:.2f}_l{leak:.2f}_tau2'
            if results.get(key, 0) > best_acc:
                best_acc = results[key]
                best_alpha = alpha_rec
                best_leak = leak
    print(f"\n  Best: α={best_alpha:.2f}, λ={best_leak:.2f} → {best_acc:.3f}")

    # ═══════════════════════════════════════════════════════════════
    # PHASE 3: Harder tasks — XOR τ=3,5 with best params
    # ═══════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print(f"PHASE 3: Harder Tasks @ {hz_fast} Hz (α={best_alpha:.2f}, λ={best_leak:.2f})")
    print("=" * 70)

    for tau in [3, 5]:
        # Baseline
        t0 = time.time()
        acc_nr = evaluate_xor(fpga, noises, w_in, w_noise, W_res,
                               alpha_rec=0, leak_rate=0, gamma_mod=0,
                               sample_hz=hz_fast, tau=tau, mode='NO_REC')
        dt = time.time() - t0
        print(f"  NO_REC τ={tau}: {acc_nr:.3f} ({dt:.0f}s)")
        results[f'P3_norec_{hz_fast}hz_tau{tau}'] = acc_nr

        # Static recurrence
        t0 = time.time()
        acc_sr = evaluate_xor(fpga, noises, w_in, w_noise, W_res,
                               alpha_rec=best_alpha, leak_rate=best_leak,
                               gamma_mod=0.20, sample_hz=hz_fast, tau=tau,
                               mode='STATIC_REC')
        dt = time.time() - t0
        delta = acc_sr - acc_nr
        tag = ">>>" if delta > 0.02 else "   "
        print(f"  {tag} STATIC τ={tau}: {acc_sr:.3f} (Δ={delta:+.3f}, {dt:.0f}s)")
        results[f'P3_static_{hz_fast}hz_tau{tau}'] = acc_sr

        # Firmware-modulated recurrence
        t0 = time.time()
        acc_fw = evaluate_xor(fpga, noises, w_in, w_noise, W_res,
                               alpha_rec=best_alpha, leak_rate=best_leak,
                               gamma_mod=0.20, sample_hz=hz_fast, tau=tau,
                               mode='FW_REC')
        dt = time.time() - t0
        delta_fw = acc_fw - acc_nr
        tag = ">>>" if acc_fw > acc_sr else "   "
        print(f"  {tag} FW_REC τ={tau}: {acc_fw:.3f} (Δ={delta_fw:+.3f}, {dt:.0f}s)")
        results[f'P3_fw_{hz_fast}hz_tau{tau}'] = acc_fw

    # ═══════════════════════════════════════════════════════════════
    # PHASE 4: Rate × Recurrence Interaction (τ=5)
    # ═══════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print(f"PHASE 4: Rate × Recurrence (α={best_alpha:.2f}, λ={best_leak:.2f}, τ=5)")
    print("=" * 70)

    for hz in [50, 100, 200]:
        t0 = time.time()
        acc_nr = evaluate_xor(fpga, noises, w_in, w_noise, W_res,
                               alpha_rec=0, leak_rate=0, gamma_mod=0,
                               sample_hz=hz, tau=5, mode='NO_REC')
        dt1 = time.time() - t0

        t0 = time.time()
        acc_sr = evaluate_xor(fpga, noises, w_in, w_noise, W_res,
                               alpha_rec=best_alpha, leak_rate=best_leak,
                               gamma_mod=0.20, sample_hz=hz, tau=5,
                               mode='STATIC_REC')
        dt2 = time.time() - t0
        delta = acc_sr - acc_nr
        tag = ">>>" if delta > 0.02 else "   "
        print(f"  {tag} {hz:4d} Hz: NO_REC={acc_nr:.3f}, STATIC={acc_sr:.3f}, "
              f"Δ={delta:+.3f} ({dt1+dt2:.0f}s)")
        results[f'P4_norec_{hz}hz_tau5'] = acc_nr
        results[f'P4_static_{hz}hz_tau5'] = acc_sr

    # ═══════════════════════════════════════════════════════════════
    # PHASE 5: Waveform Classification with Recurrence
    # ═══════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print(f"PHASE 5: 7-Class Waveform @ {hz_fast} Hz")
    print("=" * 70)

    for mode_name in ['NO_REC', 'STATIC_REC', 'FW_REC']:
        t0 = time.time()
        ar = best_alpha if mode_name != 'NO_REC' else 0
        lr = best_leak if mode_name != 'NO_REC' else 0
        gm = 0.20 if mode_name != 'NO_REC' else 0
        acc = evaluate_waveform(fpga, noises, w_in, w_noise, W_res,
                                alpha_rec=ar, leak_rate=lr, gamma_mod=gm,
                                sample_hz=hz_fast, mode=mode_name)
        dt = time.time() - t0
        print(f"  {mode_name:12s}: {acc:.3f} ({dt:.0f}s)")
        results[f'P5_wave_{mode_name}_{hz_fast}hz'] = acc

    # ═══════════════════════════════════════════════════════════════
    # SUMMARY
    # ═══════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("SUMMARY — z2219 Ethernet Recurrence")
    print("=" * 70)

    # Phase 1 recap
    print("\nPhase 1 — Rate effect on baseline (NO_REC, τ=2):")
    for hz in [20, 50, 100, 200]:
        k = f'P1_norec_{hz}hz_tau2'
        print(f"  {hz:4d} Hz: {results.get(k, 'N/A')}")

    # Phase 3 recap
    print(f"\nPhase 3 — Best recurrence (α={best_alpha:.2f}, λ={best_leak:.2f}) @ {hz_fast} Hz:")
    for tau in [2, 3, 5]:
        nr_key = f'P3_norec_{hz_fast}hz_tau{tau}' if tau > 2 else f'P1_norec_{hz_fast}hz_tau2'
        sr_key = f'P3_static_{hz_fast}hz_tau{tau}' if tau > 2 else f'P2_static_a{best_alpha:.2f}_l{best_leak:.2f}_tau2'
        fw_key = f'P3_fw_{hz_fast}hz_tau{tau}' if tau > 2 else 'N/A'
        nr = results.get(nr_key, 'N/A')
        sr = results.get(sr_key, 'N/A')
        fw = results.get(fw_key, 'N/A')
        print(f"  τ={tau}: NO_REC={nr}, STATIC={sr}, FW_REC={fw}")

    # Phase 5 recap
    print(f"\nPhase 5 — Waveform @ {hz_fast} Hz:")
    for mode_name in ['NO_REC', 'STATIC_REC', 'FW_REC']:
        k = f'P5_wave_{mode_name}_{hz_fast}hz'
        print(f"  {mode_name:12s}: {results.get(k, 'N/A')}")

    # Key comparison with z2218
    print("\nKey comparison with z2218 (UART 20 Hz):")
    print(f"  z2218 baseline τ=2: 0.536")
    print(f"  z2219 baseline τ=2 @ 200Hz: {results.get(f'P1_norec_200hz_tau2', 'N/A')}")

    tau5_nr = results.get(f'P3_norec_{hz_fast}hz_tau5', 0)
    tau5_sr = results.get(f'P3_static_{hz_fast}hz_tau5', 0)
    tau5_fw = results.get(f'P3_fw_{hz_fast}hz_tau5', 0)
    recurrence_helps = (isinstance(tau5_sr, float) and isinstance(tau5_nr, float)
                        and tau5_sr > tau5_nr + 0.03)
    fw_helps = (isinstance(tau5_fw, float) and isinstance(tau5_sr, float)
                and tau5_fw > tau5_sr + 0.01)

    print(f"\n  RECURRENCE HELPS (τ=5): {'YES' if recurrence_helps else 'NO'}")
    print(f"  FIRMWARE MODULATION HELPS: {'YES' if fw_helps else 'NO'}")

    results['best_alpha'] = best_alpha
    results['best_leak'] = best_leak
    results['sample_hz_fast'] = hz_fast

    with open(RESULTS / 'z2219_eth_recurrence.json', 'w') as f:
        json.dump(results, f, indent=2, cls=NpEncoder)
    print(f"\n  Saved: results/z2219_eth_recurrence.json")

    fpga.close()

if __name__ == '__main__':
    main()
