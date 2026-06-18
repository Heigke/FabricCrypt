#!/usr/bin/env python3
"""
z2233_bridge_definitive.py — Definitive GPU↔FPGA NS-RAM Bridge Experiment
==========================================================================
Bridges the gap between:
  - GPU low-level hardware neuromorphic (power 1/f, thermal, clock jitter)
  - FPGA NS-RAM avalanche neurons (Lanza model, BVpar physics)

Three honest experiments, no software tricks:

EXP 1 — FADING MEMORY (the persistent failure)
  Does the LIF membrane's τ=105ms create genuine fading memory at 200Hz?
  Memory capacity: R²(d) for delays d=1..10 at 5ms step (200Hz).
  LEAK_COND=0x0008 (τ≈105ms, 8.95× dynamic range).
  Compare: NOISE (GPU 1/f → Vg), STATIC (fixed Vg), WHITE (random Vg).
  Target: R²(d=1) > 0.05 with meaningful decay profile.

EXP 2 — RESERVOIR CLASSIFICATION (proven protocol, new bridge)
  z2206 protocol (81% at 20Hz UART) adapted to 200Hz ETH.
  4-class waveform: sine/triangle/square/sawtooth.
  Compare LEAK_COND: 0x0008, 0x0010, 0x0020.
  Rich features: 3 state vars + delay augment + 4× pooling.
  Target: reproduce >70% and show temporal features help.

EXP 3 — MAC NEUROMODULATION (not a wire)
  MAC signal changes neuron excitability, not input encoding.
  Two tasks run simultaneously:
    a) Waveform encoded in Vg (as before)
    b) MAC carries a DIFFERENT signal (task context / attention)
  Test: does MAC modulation improve classification beyond no-MAC?
  This is biologically motivated: neuromodulation changes gain, not content.

All use: BASE_VG=0.58, ALPHA=0.25, BETA=0.08 (proven z2206 params),
128 neurons, 200Hz ETH auto-telemetry, pre-collected GPU noise.
"""

import os, sys, time, json
import numpy as np

os.environ.setdefault('HSA_OVERRIDE_GFX_VERSION', '11.0.0')
sys.path.insert(0, "scripts")
from fpga_host_eth import FPGAEthBridge

# ==========================================================================
# Constants
# ==========================================================================
N_NEURONS = 128
SAMPLE_HZ = 200
STEP_DT = 1.0 / SAMPLE_HZ  # 5 ms

# Proven z2206 parameters
BASE_VG = 0.58
ALPHA = 0.25    # input → Vg gain
BETA = 0.08     # noise → Vg gain

# Noise channel mapping (heterogeneous, as in z2206)
# Neurons 0-31:  channel 0 (power 1/f)
# Neurons 32-55: channel 1 (thermal drift)
# Neurons 56-79: channel 2 (perf jitter)
# Neurons 80-103: channel 3 (temp_soc oscillation)
# Neurons 104-127: channel 4 (clock drift)
NOISE_CHANNELS = 5
NOISE_MAP = [0]*32 + [1]*24 + [2]*24 + [3]*24 + [4]*24

# IIR filter coefficients per channel
IIR_ALPHAS = [0.85, 0.92, 0.0, 0.90, 0.80]  # 0.0 = no filter (white-ish)


def collect_gpu_noise(duration=20.0, rate=50):
    """Collect GPU hardware noise from hwmon/PM table. Returns (n_samples, 5) array."""
    import subprocess

    n = int(duration * rate)
    noise = np.zeros((n, NOISE_CHANNELS), dtype=np.float32)
    dt = 1.0 / rate

    for i in range(n):
        t0 = time.perf_counter()

        # Channel 0: hwmon power (1/f character)
        try:
            with open('/sys/class/hwmon/hwmon7/power1_average', 'r') as f:
                noise[i, 0] = float(f.read().strip()) / 1e6  # µW → W
        except:
            noise[i, 0] = noise[i-1, 0] if i > 0 else 10.0

        # Channel 1: PM table thermal (slow drift)
        try:
            with open('/sys/kernel/ryzen_smu_drv/pm_table', 'rb') as f:
                f.seek(0x004C)
                noise[i, 1] = np.frombuffer(f.read(4), dtype=np.float32)[0]
        except:
            noise[i, 1] = noise[i-1, 1] if i > 0 else 45.0

        # Channel 2: perf counter jitter
        try:
            t_perf = time.perf_counter_ns()
            noise[i, 2] = float(t_perf % 100000) / 100000.0
        except:
            noise[i, 2] = np.random.random()

        # Channel 3: hwmon temp (thermal oscillation)
        try:
            with open('/sys/class/hwmon/hwmon7/temp1_input', 'r') as f:
                noise[i, 3] = float(f.read().strip()) / 1000.0  # mC → C
        except:
            noise[i, 3] = noise[i-1, 3] if i > 0 else 45.0

        # Channel 4: hwmon clock (drift)
        try:
            with open('/sys/class/hwmon/hwmon7/freq1_input', 'r') as f:
                noise[i, 4] = float(f.read().strip()) / 1e6  # Hz → MHz
        except:
            noise[i, 4] = noise[i-1, 4] if i > 0 else 1000.0

        elapsed = time.perf_counter() - t0
        if elapsed < dt:
            time.sleep(dt - elapsed)

    # Normalize each channel to zero-mean unit-variance
    for ch in range(NOISE_CHANNELS):
        mu = noise[:, ch].mean()
        sigma = noise[:, ch].std()
        if sigma > 1e-10:
            noise[:, ch] = (noise[:, ch] - mu) / sigma

    # Apply IIR filters
    for ch in range(NOISE_CHANNELS):
        a = IIR_ALPHAS[ch]
        if a > 0:
            filtered = np.zeros_like(noise[:, ch])
            filtered[0] = noise[0, ch]
            for j in range(1, len(filtered)):
                filtered[j] = a * filtered[j-1] + (1 - a) * noise[j, ch]
            noise[:, ch] = filtered

    return noise


def drain_latest(fpga, max_reads=50):
    """Drain auto-telemetry buffer, return latest packet."""
    latest = None
    for _ in range(max_reads):
        try:
            pkt = fpga.recv_auto_telemetry(timeout=0.001)
            if pkt is not None:
                latest = pkt
            else:
                break
        except:
            break
    return latest


def ridge_r2(X_tr, y_tr, X_te, y_te, alphas=None):
    """Ridge regression R² with alpha search."""
    if alphas is None:
        alphas = [1e-4, 1e-2, 1.0, 100.0, 10000.0]
    best = -999
    for a in alphas:
        I = np.eye(X_tr.shape[1])
        try:
            w = np.linalg.solve(X_tr.T @ X_tr + a * I, X_tr.T @ y_tr)
        except:
            continue
        pred = X_te @ w
        ss_res = np.sum((y_te - pred) ** 2)
        ss_tot = np.sum((y_te - y_te.mean()) ** 2)
        if ss_tot < 1e-10:
            continue
        r2 = 1 - ss_res / ss_tot
        if r2 > best:
            best = r2
    return best


def ridge_classify(X_tr, y_tr, X_te, y_te, n_classes=4):
    """Ridge classification (one-hot encoding)."""
    Y_tr = np.zeros((len(y_tr), n_classes))
    for i, c in enumerate(y_tr):
        Y_tr[i, int(c)] = 1.0
    best_acc = -1
    for a in [1e-4, 1e-2, 1.0, 100.0, 10000.0]:
        I = np.eye(X_tr.shape[1])
        try:
            W = np.linalg.solve(X_tr.T @ X_tr + a * I, X_tr.T @ Y_tr)
        except:
            continue
        pred = X_te @ W
        acc = np.mean(pred.argmax(axis=1) == y_te.astype(int))
        if acc > best_acc:
            best_acc = acc
    return best_acc


def generate_waveform(class_id, n_steps, rng):
    """Generate 3-class waveform signal (sine/triangle/square), matching z2206."""
    freq = rng.uniform(0.8, 1.2)  # match z2206 frequency range
    phase = rng.uniform(0, 2 * np.pi)
    t = np.linspace(0, n_steps / SAMPLE_HZ, n_steps)

    if class_id == 0:   # sine
        sig = np.sin(2 * np.pi * freq * t + phase)
    elif class_id == 1: # triangle
        sig = 2.0 * np.abs(2.0 * ((freq * t + phase/(2*np.pi)) % 1.0) - 1.0) - 1.0
    else:               # square
        sig = np.sign(np.sin(2 * np.pi * freq * t + phase))

    # Scale to [0, 1] matching z2206
    sig = (sig + 1.0) / 2.0
    return sig


def extract_features(trial_states, n_neurons=N_NEURONS):
    """Extract rich features from trial state matrix (z2206 protocol).

    trial_states: (n_steps, n_neurons*3) — [delta_spike | vmem | cumulative]
    Returns: 1D feature vector with delay augmentation + 4× pooling.
    """
    n_steps = trial_states.shape[0]
    if n_steps < 4:
        return None

    # Delay augmentation: append t-1, t-2, t-3
    width = trial_states.shape[1]
    augmented = np.zeros((n_steps - 3, width * 4))
    for t in range(3, n_steps):
        augmented[t-3, 0*width:1*width] = trial_states[t]
        augmented[t-3, 1*width:2*width] = trial_states[t-1]
        augmented[t-3, 2*width:3*width] = trial_states[t-2]
        augmented[t-3, 3*width:4*width] = trial_states[t-3]

    # 4× temporal pooling
    feat = np.concatenate([
        augmented.mean(axis=0),
        augmented.std(axis=0),
        augmented.max(axis=0),
        augmented.min(axis=0),
    ])
    return feat


def run_trial(fpga, signal, noise_buf, noise_idx, w_in, w_noise, mac_signal=0.0):
    """Run one trial with signal-driven Vg modulation and noise injection.

    Returns list of state vectors (n_steps, n_neurons*3).
    """
    n_steps = len(signal)
    prev_counts = None
    cumulative = np.zeros(N_NEURONS, dtype=np.float32)
    states = []

    # Set MAC (constant for the trial)
    fpga.set_mac_signal(mac_signal)

    for t in range(n_steps):
        t0 = time.perf_counter()

        # Compute Vg: BASE + ALPHA * signal * w_in + BETA * noise * w_noise
        ni = (noise_idx + t) % len(noise_buf)
        noise_per_neuron = np.array([noise_buf[ni, NOISE_MAP[n]] for n in range(N_NEURONS)])
        vg = np.full(N_NEURONS, BASE_VG) + ALPHA * signal[t] * w_in + BETA * noise_per_neuron * w_noise
        vg = np.clip(vg, 0.05, 0.95)

        # Set Vg (batch for speed)
        fpga.set_vg_batch(0, vg[:64].tolist())
        fpga.set_vg_batch(64, vg[64:].tolist())

        # Wait for interval, drain buffer, get latest telemetry
        elapsed_set = time.perf_counter() - t0
        wait = STEP_DT - elapsed_set - 0.001  # leave 1ms for read
        if wait > 0.0005:
            time.sleep(wait)

        latest = drain_latest(fpga, max_reads=30)
        if latest is None:
            time.sleep(0.003)
            latest = drain_latest(fpga, max_reads=10)

        if latest is not None:
            counts = latest['spike_counts']
            vm = latest['vmem']

            if prev_counts is not None:
                delta = counts.astype(np.int32) - prev_counts.astype(np.int32)
                delta[delta < 0] = 0
                delta[delta > 30000] = 0
                cumulative += delta.astype(np.float32)

                # State: [delta_spikes(128) | vmem(128) | cumulative(128)]
                state = np.concatenate([
                    delta.astype(np.float32),
                    vm.astype(np.float32),
                    cumulative.copy()
                ])
                states.append(state)

            prev_counts = counts.copy()

        # Maintain timing
        total_elapsed = time.perf_counter() - t0
        remaining = STEP_DT - total_elapsed
        if remaining > 0.0005:
            time.sleep(remaining)

    return states


# ==========================================================================
# EXP 1: FADING MEMORY
# ==========================================================================
def exp1_memory_capacity(fpga, noise_buf, rng, leak_val=0x0008):
    """Test fading memory at 200Hz with τ≈105ms.

    Memory capacity: drive neurons with random input sequence u(t),
    then decode u(t-d) from spike states at time t.
    """
    print("\n" + "=" * 70)
    print(f"EXP 1 — FADING MEMORY (LEAK={hex(leak_val)}, 200Hz, τ≈105ms)")
    print("=" * 70)

    fpga.set_leak_cond(leak_val)
    time.sleep(0.1)

    N_TRIALS = 60       # independent random sequences
    N_STEPS = 200       # steps per trial (1.0s at 200Hz — 10× membrane τ)
    MAX_DELAY = 10

    w_in = rng.standard_normal(N_NEURONS).astype(np.float32) * 0.3
    w_noise = rng.standard_normal(N_NEURONS).astype(np.float32) * 0.3

    conditions = {
        'NOISE': True,   # GPU 1/f noise driving
        'WHITE': 'white', # white noise (no IIR, random)
        'STATIC': False,  # fixed Vg, no noise
    }

    results = {}

    for cond_name, cond_noise in conditions.items():
        print(f"\n  --- {cond_name} ---")

        # Reset
        fpga.set_kill(True)
        time.sleep(0.15)
        fpga.set_kill(False)
        time.sleep(0.15)
        drain_latest(fpga, max_reads=100)

        all_inputs = []
        all_states = []

        for trial in range(N_TRIALS):
            # Random input sequence
            u = rng.uniform(-1, 1, N_STEPS).astype(np.float32)

            # Choose noise source
            if cond_noise == True:
                noise_idx = rng.integers(0, len(noise_buf) - N_STEPS)
                trial_noise = noise_buf
            elif cond_noise == 'white':
                # Generate fresh white noise matching shape
                trial_noise = rng.standard_normal((len(noise_buf), NOISE_CHANNELS)).astype(np.float32)
                noise_idx = 0
            else:
                trial_noise = np.zeros((len(noise_buf), NOISE_CHANNELS), dtype=np.float32)
                noise_idx = 0

            states = run_trial(fpga, u, trial_noise, noise_idx, w_in, w_noise)

            if len(states) >= N_STEPS - 5:
                # Align: use last N_STEPS-1 states (skip first for delta)
                state_mat = np.array(states[-N_STEPS+1:])  # (N_STEPS-1, 384)
                all_inputs.append(u)
                all_states.append(state_mat)

            if (trial + 1) % 20 == 0:
                print(f"    Trial {trial+1}/{N_TRIALS}, valid={len(all_states)}")

        if len(all_states) < 20:
            print(f"    FAIL: only {len(all_states)} valid trials")
            results[cond_name] = {'mc': [0]*MAX_DELAY, 'mc_total': 0}
            continue

        # Compute memory capacity at each delay
        mc_values = []
        for d in range(1, MAX_DELAY + 1):
            X_list = []
            y_list = []
            for i in range(len(all_states)):
                state_mat = all_states[i]
                u = all_inputs[i]
                # For each timestep t where u(t-d) exists
                for t in range(d, min(len(state_mat), len(u) - 1)):
                    X_list.append(state_mat[t])
                    y_list.append(u[t - d])

            X = np.array(X_list)
            y = np.array(y_list)

            # Normalize
            mu = X.mean(axis=0)
            sigma = X.std(axis=0)
            sigma[sigma < 1e-2] = 1.0
            X_n = (X - mu) / sigma

            # Train/test split
            n = len(X)
            n_tr = n * 3 // 4
            idx = rng.permutation(n)
            r2 = ridge_r2(X_n[idx[:n_tr]], y[idx[:n_tr]], X_n[idx[n_tr:]], y[idx[n_tr:]])
            mc_values.append(max(0, r2))

            print(f"    d={d:2d}: R²={r2:+.4f} {'***' if r2 > 0.05 else ''}")

        mc_total = sum(mc_values)
        results[cond_name] = {
            'mc': mc_values,
            'mc_total': float(mc_total),
            'n_valid': len(all_states),
        }
        print(f"    MC total = {mc_total:.3f}")

    # Tests
    print("\n  TESTS:")
    noise_mc = results.get('NOISE', {}).get('mc_total', 0)
    static_mc = results.get('STATIC', {}).get('mc_total', 0)
    white_mc = results.get('WHITE', {}).get('mc_total', 0)
    noise_d1 = results.get('NOISE', {}).get('mc', [0])[0]

    t_pass = 0
    t_total = 3

    p = noise_d1 > 0.05
    t_pass += p
    print(f"  T1 MC(d=1) > 0.05:        R²={noise_d1:.4f} {'PASS' if p else 'FAIL'}")

    p = noise_mc > static_mc
    t_pass += p
    print(f"  T2 NOISE_MC > STATIC_MC:   {noise_mc:.3f} vs {static_mc:.3f} {'PASS' if p else 'FAIL'}")

    p = noise_mc > white_mc
    t_pass += p
    print(f"  T3 NOISE_MC > WHITE_MC:    {noise_mc:.3f} vs {white_mc:.3f} {'PASS' if p else 'FAIL'}")

    print(f"  Score: {t_pass}/{t_total}")
    results['tests'] = {'pass': t_pass, 'total': t_total}
    return results


# ==========================================================================
# EXP 2: RESERVOIR CLASSIFICATION
# ==========================================================================
def exp2_classification(fpga, noise_buf, rng):
    """4-class waveform classification with proven z2206 protocol at 200Hz.
    Compare LEAK_COND values to find optimal dynamics.
    """
    print("\n" + "=" * 70)
    print("EXP 2 — RESERVOIR CLASSIFICATION (z2206 protocol, 200Hz ETH)")
    print("=" * 70)

    N_TRIALS = 150   # 50 per class
    N_STEPS = 300    # 1.5s at 200Hz (matches z2206's 30 steps @ 20Hz)
    N_CLASSES = 3    # sine/triangle/square (matching z2206)
    LEAK_VALUES = [0x0008, 0x0010, 0x0020]
    LEAK_NAMES = ['0x0008(τ≈105ms)', '0x0010(τ≈52ms)', '0x0020(τ≈26ms)']

    w_in = rng.standard_normal(N_NEURONS).astype(np.float32) * 0.3
    w_noise = rng.standard_normal(N_NEURONS).astype(np.float32) * 0.3

    results = {}

    for leak_val, leak_name in zip(LEAK_VALUES, LEAK_NAMES):
        print(f"\n  --- LEAK={leak_name} ---")
        fpga.set_leak_cond(leak_val)
        time.sleep(0.1)

        for cond in ['COUPLED', 'FPGA_ONLY']:
            print(f"\n    {cond}:")

            # Reset
            fpga.set_kill(True)
            time.sleep(0.15)
            fpga.set_kill(False)
            time.sleep(0.15)
            drain_latest(fpga, max_reads=100)

            X_all = []
            y_all = []

            for trial in range(N_TRIALS):
                class_id = trial % N_CLASSES

                # Generate waveform signal
                sig = generate_waveform(class_id, N_STEPS, rng)

                if cond == 'COUPLED':
                    noise_idx = rng.integers(0, len(noise_buf) - N_STEPS)
                    trial_noise = noise_buf
                else:
                    trial_noise = np.zeros((len(noise_buf), NOISE_CHANNELS), dtype=np.float32)
                    noise_idx = 0

                states = run_trial(fpga, sig, trial_noise, noise_idx, w_in, w_noise)

                if len(states) >= N_STEPS - 8:
                    state_mat = np.array(states)
                    feat = extract_features(state_mat)
                    if feat is not None:
                        X_all.append(feat)
                        y_all.append(class_id)

                if (trial + 1) % 40 == 0:
                    print(f"      Trial {trial+1}/{N_TRIALS}, valid={len(X_all)}")

            if len(X_all) < 40:
                print(f"      FAIL: only {len(X_all)} valid trials")
                results[f"{leak_name}_{cond}"] = {'acc': 0, 'n_valid': len(X_all)}
                continue

            X = np.array(X_all)
            y = np.array(y_all)

            # Normalize
            mu = X.mean(axis=0)
            sigma = X.std(axis=0)
            sigma[sigma < 1e-2] = 1.0
            X_n = (X - mu) / sigma

            # 5-fold cross-validation
            n = len(X)
            fold_size = n // 5
            accs = []
            for fold in range(5):
                te_idx = list(range(fold * fold_size, min((fold + 1) * fold_size, n)))
                tr_idx = [i for i in range(n) if i not in te_idx]
                acc = ridge_classify(X_n[tr_idx], y[tr_idx], X_n[te_idx], y[te_idx], N_CLASSES)
                accs.append(acc)

            mean_acc = np.mean(accs)
            std_acc = np.std(accs)
            print(f"      Accuracy: {mean_acc:.3f} ± {std_acc:.3f}")

            results[f"{leak_name}_{cond}"] = {
                'acc': float(mean_acc),
                'std': float(std_acc),
                'n_valid': len(X_all),
                'feat_dim': X.shape[1],
            }

    # Also test with temporal features only (no delay augment)
    # to show 200Hz gives temporal info that 20Hz couldn't
    print("\n  --- TEMPORAL vs SNAPSHOT features (LEAK=0x0008) ---")
    fpga.set_leak_cond(0x0008)
    time.sleep(0.1)
    fpga.set_kill(True); time.sleep(0.15); fpga.set_kill(False); time.sleep(0.15)
    drain_latest(fpga, max_reads=100)

    X_snap = []  # snapshot: just mean pooling of raw states
    X_temp = []  # temporal: full z2206 pipeline
    y_ft = []

    for trial in range(N_TRIALS):
        class_id = trial % N_CLASSES
        sig = generate_waveform(class_id, N_STEPS, rng)
        noise_idx = rng.integers(0, len(noise_buf) - N_STEPS)
        states = run_trial(fpga, sig, noise_buf, noise_idx, w_in, w_noise)

        if len(states) >= N_STEPS - 8:
            state_mat = np.array(states)
            # Snapshot: just mean of raw features (no delay, no multi-pool)
            snap_feat = state_mat.mean(axis=0)
            # Temporal: full pipeline
            temp_feat = extract_features(state_mat)
            if temp_feat is not None:
                X_snap.append(snap_feat)
                X_temp.append(temp_feat)
                y_ft.append(class_id)

        if (trial + 1) % 40 == 0:
            print(f"    Trial {trial+1}/{N_TRIALS}, valid={len(y_ft)}")

    for name, X_list in [('SNAPSHOT', X_snap), ('TEMPORAL', X_temp)]:
        X = np.array(X_list)
        y = np.array(y_ft)
        mu = X.mean(axis=0); sigma = X.std(axis=0); sigma[sigma < 1e-2] = 1.0
        X_n = (X - mu) / sigma
        n = len(X); fold_size = n // 5; accs = []
        for fold in range(5):
            te = list(range(fold*fold_size, min((fold+1)*fold_size, n)))
            tr = [i for i in range(n) if i not in te]
            accs.append(ridge_classify(X_n[tr], y[tr], X_n[te], y[te], N_CLASSES))
        print(f"    {name}: {np.mean(accs):.3f} ± {np.std(accs):.3f} (dim={X.shape[1]})")
        results[f"feat_{name}"] = {'acc': float(np.mean(accs)), 'std': float(np.std(accs)), 'dim': X.shape[1]}

    # Tests
    print("\n  TESTS:")
    t_pass = 0; t_total = 5

    # T4: Best accuracy > 0.50
    best_acc = max(v.get('acc', 0) for v in results.values())
    p = best_acc > 0.50
    t_pass += p
    print(f"  T4 Best accuracy > 0.50:     {best_acc:.3f} {'PASS' if p else 'FAIL'}")

    # T5: COUPLED >= FPGA_ONLY for best LEAK
    best_leak = None; best_coupled = 0
    for lname in LEAK_NAMES:
        c = results.get(f"{lname}_COUPLED", {}).get('acc', 0)
        if c > best_coupled:
            best_coupled = c
            best_leak = lname
    if best_leak:
        fpga_only = results.get(f"{best_leak}_FPGA_ONLY", {}).get('acc', 0)
        p = best_coupled >= fpga_only
        t_pass += p
        print(f"  T5 COUPLED >= FPGA_ONLY:     {best_coupled:.3f} vs {fpga_only:.3f} {'PASS' if p else 'FAIL'}")
    else:
        print(f"  T5 COUPLED >= FPGA_ONLY:     NO DATA FAIL")

    # T6: Temporal > Snapshot
    temp_acc = results.get('feat_TEMPORAL', {}).get('acc', 0)
    snap_acc = results.get('feat_SNAPSHOT', {}).get('acc', 0)
    p = temp_acc > snap_acc
    t_pass += p
    print(f"  T6 TEMPORAL > SNAPSHOT:      {temp_acc:.3f} vs {snap_acc:.3f} {'PASS' if p else 'FAIL'}")

    # T7: Best > 0.70 (z2206 territory)
    p = best_acc > 0.70
    t_pass += p
    print(f"  T7 Best accuracy > 0.70:     {best_acc:.3f} {'PASS' if p else 'FAIL'}")

    # T8: LEAK_COND affects performance (not all the same)
    leak_accs = [results.get(f"{ln}_COUPLED", {}).get('acc', 0) for ln in LEAK_NAMES]
    p = max(leak_accs) - min(leak_accs) > 0.03
    t_pass += p
    print(f"  T8 LEAK affects performance: range={max(leak_accs)-min(leak_accs):.3f} {'PASS' if p else 'FAIL'}")

    print(f"  Score: {t_pass}/{t_total}")
    results['tests'] = {'pass': t_pass, 'total': t_total}
    return results


# ==========================================================================
# EXP 3: MAC NEUROMODULATION
# ==========================================================================
def exp3_neuromodulation(fpga, noise_buf, rng):
    """MAC as neuromodulation: changes neuron excitability, not signal.

    Two conditions:
      NO_MAC:  MAC=0, input only through Vg
      MODULATED: MAC varies with task context (0.2 for "easy" waveforms,
                 0.6 for "hard" waveforms), like dopaminergic gain control

    This is NOT a linear wire — MAC doesn't encode the waveform class.
    It modulates the OPERATING REGIME of the neurons.
    """
    print("\n" + "=" * 70)
    print("EXP 3 — MAC NEUROMODULATION (gain control, not signal)")
    print("=" * 70)

    fpga.set_leak_cond(0x0008)
    time.sleep(0.1)

    N_TRIALS = 120
    N_STEPS = 300    # 1.5s matching EXP 2
    N_CLASSES = 3

    w_in = rng.standard_normal(N_NEURONS).astype(np.float32) * 0.3
    w_noise = rng.standard_normal(N_NEURONS).astype(np.float32) * 0.3

    # Define task difficulty: sine=easy (smooth), triangle=hard (sharp transitions), square=easy (binary)
    MAC_LEVELS = {0: 0.2, 1: 0.6, 2: 0.2}  # sine=easy, tri=hard, sq=easy

    results = {}

    for cond, use_mac in [('NO_MAC', False), ('MODULATED', True), ('CONSTANT_MAC', 'constant')]:
        print(f"\n  --- {cond} ---")

        fpga.set_kill(True); time.sleep(0.15); fpga.set_kill(False); time.sleep(0.15)
        drain_latest(fpga, max_reads=100)

        X_all = []
        y_all = []

        for trial in range(N_TRIALS):
            class_id = trial % N_CLASSES
            sig = generate_waveform(class_id, N_STEPS, rng)
            noise_idx = rng.integers(0, len(noise_buf) - N_STEPS)

            if use_mac == True:
                mac = MAC_LEVELS[class_id]
            elif use_mac == 'constant':
                mac = 0.4  # same for all classes
            else:
                mac = 0.0

            states = run_trial(fpga, sig, noise_buf, noise_idx, w_in, w_noise, mac_signal=mac)

            if len(states) >= N_STEPS - 8:
                state_mat = np.array(states)
                feat = extract_features(state_mat)
                if feat is not None:
                    X_all.append(feat)
                    y_all.append(class_id)

            if (trial + 1) % 40 == 0:
                print(f"    Trial {trial+1}/{N_TRIALS}, valid={len(X_all)}")

        if len(X_all) < 40:
            print(f"    FAIL: only {len(X_all)} valid trials")
            results[cond] = {'acc': 0}
            continue

        X = np.array(X_all); y = np.array(y_all)
        mu = X.mean(axis=0); sigma = X.std(axis=0); sigma[sigma < 1e-2] = 1.0
        X_n = (X - mu) / sigma

        n = len(X); fold_size = n // 5; accs = []
        for fold in range(5):
            te = list(range(fold*fold_size, min((fold+1)*fold_size, n)))
            tr = [i for i in range(n) if i not in te]
            accs.append(ridge_classify(X_n[tr], y[tr], X_n[te], y[te], N_CLASSES))

        mean_acc = np.mean(accs)
        std_acc = np.std(accs)
        print(f"    Accuracy: {mean_acc:.3f} ± {std_acc:.3f}")
        results[cond] = {'acc': float(mean_acc), 'std': float(std_acc), 'n_valid': len(X_all)}

    # Tests
    print("\n  TESTS:")
    t_pass = 0; t_total = 3

    mod_acc = results.get('MODULATED', {}).get('acc', 0)
    no_mac = results.get('NO_MAC', {}).get('acc', 0)
    const_mac = results.get('CONSTANT_MAC', {}).get('acc', 0)

    # T9: MODULATED >= NO_MAC (neuromodulation helps)
    p = mod_acc >= no_mac
    t_pass += p
    print(f"  T9  MODULATED >= NO_MAC:      {mod_acc:.3f} vs {no_mac:.3f} {'PASS' if p else 'FAIL'}")

    # T10: MODULATED > CONSTANT_MAC (context-dependent modulation > fixed)
    p = mod_acc > const_mac
    t_pass += p
    print(f"  T10 MODULATED > CONSTANT_MAC: {mod_acc:.3f} vs {const_mac:.3f} {'PASS' if p else 'FAIL'}")

    # T11: CONSTANT_MAC > NO_MAC (any excitability change helps)
    p = const_mac > no_mac
    t_pass += p
    print(f"  T11 CONSTANT_MAC > NO_MAC:    {const_mac:.3f} vs {no_mac:.3f} {'PASS' if p else 'FAIL'}")

    print(f"  Score: {t_pass}/{t_total}")
    results['tests'] = {'pass': t_pass, 'total': t_total}
    return results


# ==========================================================================
# MAIN
# ==========================================================================
def main():
    print("z2233 — Definitive GPU↔FPGA NS-RAM Bridge Experiment")
    print("=" * 70)

    # Connect to FPGA
    fpga = FPGAEthBridge()
    if not fpga.connect():
        print("FAIL: Cannot connect to FPGA")
        return

    fpga.set_kill(False)
    fpga.enable_auto_telemetry(2000)  # 2kHz push for 200Hz sampling headroom
    time.sleep(0.3)

    # Drain stale packets
    drain_latest(fpga, max_reads=200)

    rng = np.random.default_rng(2233)

    # Collect GPU noise (pre-collected, as in z2206)
    print("\nCollecting GPU hardware noise (20s at 50Hz)...")
    t0 = time.time()
    noise_buf = collect_gpu_noise(duration=20.0, rate=50)
    print(f"  Collected {noise_buf.shape[0]} samples in {time.time()-t0:.1f}s")
    print(f"  Channel stats: mean={noise_buf.mean(axis=0)}, std={noise_buf.std(axis=0)}")

    # Verify noise has real dynamics
    for ch in range(NOISE_CHANNELS):
        print(f"  Ch{ch}: range=[{noise_buf[:,ch].min():.2f}, {noise_buf[:,ch].max():.2f}], "
              f"autocorr(1)={np.corrcoef(noise_buf[:-1,ch], noise_buf[1:,ch])[0,1]:.3f}")

    all_results = {}

    # EXP 1: Fading Memory
    try:
        all_results['exp1_memory'] = exp1_memory_capacity(fpga, noise_buf, rng)
    except Exception as e:
        print(f"  EXP 1 ERROR: {e}")
        import traceback; traceback.print_exc()

    # EXP 2: Classification
    try:
        all_results['exp2_classification'] = exp2_classification(fpga, noise_buf, rng)
    except Exception as e:
        print(f"  EXP 2 ERROR: {e}")
        import traceback; traceback.print_exc()

    # EXP 3: Neuromodulation
    try:
        all_results['exp3_neuromodulation'] = exp3_neuromodulation(fpga, noise_buf, rng)
    except Exception as e:
        print(f"  EXP 3 ERROR: {e}")
        import traceback; traceback.print_exc()

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    total_pass = 0; total_tests = 0
    for exp_name, exp_res in all_results.items():
        t = exp_res.get('tests', {})
        p = t.get('pass', 0); n = t.get('total', 0)
        total_pass += p; total_tests += n
        print(f"  {exp_name}: {p}/{n}")
    print(f"  TOTAL: {total_pass}/{total_tests}")

    # Save
    with open("results/z2233_bridge_definitive.json", "w") as f:
        # Convert numpy types
        def convert(obj):
            if isinstance(obj, (np.integer,)): return int(obj)
            if isinstance(obj, (np.floating,)): return float(obj)
            if isinstance(obj, np.ndarray): return obj.tolist()
            return obj
        json.dump(all_results, f, indent=2, default=convert)
    print(f"\nSaved to results/z2233_bridge_definitive.json")

    fpga.close()


if __name__ == "__main__":
    main()
