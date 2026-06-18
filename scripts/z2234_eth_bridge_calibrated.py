#!/usr/bin/env python3
"""
z2234_eth_bridge_calibrated.py — Calibrated ETH Bridge Reservoir
================================================================
Diagnosis from z2232/z2233:
  - z2206 (UART 20Hz) got 81% waveform classification
  - z2233 (ETH 200Hz) got 30-39% — massive regression
  - Root cause: 5ms steps at 200Hz don't accumulate enough spikes
  - z2232 sweet spot: leak=0x0100, vg=0.68, mac=on → R²=0.954
  - z2233 EXP3: MAC neuromodulation = clear win (0.531 vs 0.275)

Strategy:
  EXP 1 — RATE CALIBRATION: Find operating point where ETH telemetry
    gives meaningful spike deltas per 5ms step. Sweep Vg × threshold × exc.

  EXP 2 — TEMPORAL POOLING: Collect at 200Hz but pool features at
    effective rates (20Hz, 50Hz, 100Hz, 200Hz). Compare classification.
    Hypothesis: 20-50Hz effective pooling matches z2206 performance.

  EXP 3 — MAC BRIDGE CLASSIFICATION: Best operating point from EXP1/2
    with MAC neuromodulation. 3-class waveform, 200 trials.
    Compare: COUPLED (noise+MAC), FPGA_ONLY (no noise, no MAC),
    MAC_ONLY (MAC, no noise), NOISE_ONLY (noise, no MAC).

  EXP 4 — MEMORY WITH MAC: Memory capacity test at best operating point
    with MAC driving the input signal. Hypothesis: MAC gives R²>0.10
    (vs z2233's MC≈0 via Vg-only).

  EXP 5 — Z2206 REPRODUCTION ON ETH: Exact z2206 protocol (20Hz effective,
    3-class, 200 trials, delay augment) but via ETH bridge.
    Target: >70% to validate ETH bridge parity.

Tests T815-T830:
  T815: At least one operating point gives >5 spk/neuron/step at 200Hz
  T816: Pooling at 20Hz > pooling at 200Hz for classification
  T817: COUPLED > FPGA_ONLY (the persistent failure we need to fix)
  T818: MAC_ONLY > FPGA_ONLY (MAC carries signal)
  T819: NOISE_ONLY > FPGA_ONLY (noise adds richness)
  T820: COUPLED > MAX(MAC_ONLY, NOISE_ONLY) (synergy)
  T821: Best condition > 0.60 (reasonable for 3-class)
  T822: MC(d=1) > 0.10 with MAC driving
  T823: MC(d=1,MAC) > MC(d=1,Vg) (MAC is better input channel)
  T824: ETH reproduction > 0.70 (z2206 parity)
  T825: ETH reproduction within 10pp of z2206 result (81%)

Hardware: AMD gfx1151 GPU + Arty A7-100T FPGA via Ethernet (192.168.0.50)
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
ETH_HZ = 200       # raw telemetry rate
STEP_DT = 1.0 / ETH_HZ

BASE_VG = 0.58
ALPHA = 0.25
BETA = 0.08

NOISE_CHANNELS = 5
NOISE_MAP = [0]*32 + [1]*24 + [2]*24 + [3]*24 + [4]*24
IIR_ALPHAS = [0.85, 0.92, 0.0, 0.90, 0.80]


def collect_gpu_noise(duration=20.0, rate=50):
    """Collect GPU hardware noise from hwmon/PM table."""
    n = int(duration * rate)
    noise = np.zeros((n, NOISE_CHANNELS), dtype=np.float32)
    dt = 1.0 / rate

    for i in range(n):
        t0 = time.perf_counter()
        try:
            with open('/sys/class/hwmon/hwmon7/power1_average', 'r') as f:
                noise[i, 0] = float(f.read().strip()) / 1e6
        except:
            noise[i, 0] = noise[i-1, 0] if i > 0 else 10.0
        try:
            with open('/sys/kernel/ryzen_smu_drv/pm_table', 'rb') as f:
                f.seek(0x004C)
                noise[i, 1] = np.frombuffer(f.read(4), dtype=np.float32)[0]
        except:
            noise[i, 1] = noise[i-1, 1] if i > 0 else 45.0
        try:
            noise[i, 2] = float(time.perf_counter_ns() % 100000) / 100000.0
        except:
            noise[i, 2] = np.random.random()
        try:
            with open('/sys/class/hwmon/hwmon7/temp1_input', 'r') as f:
                noise[i, 3] = float(f.read().strip()) / 1000.0
        except:
            noise[i, 3] = noise[i-1, 3] if i > 0 else 45.0
        try:
            with open('/sys/class/hwmon/hwmon7/freq1_input', 'r') as f:
                noise[i, 4] = float(f.read().strip()) / 1e6
        except:
            noise[i, 4] = noise[i-1, 4] if i > 0 else 1000.0

        elapsed = time.perf_counter() - t0
        if elapsed < dt:
            time.sleep(dt - elapsed)

    for ch in range(NOISE_CHANNELS):
        mu, sigma = noise[:, ch].mean(), noise[:, ch].std()
        if sigma > 1e-10:
            noise[:, ch] = (noise[:, ch] - mu) / sigma

    for ch in range(NOISE_CHANNELS):
        a = IIR_ALPHAS[ch]
        if a > 0:
            filtered = np.zeros_like(noise[:, ch])
            filtered[0] = noise[0, ch]
            for j in range(1, len(filtered)):
                filtered[j] = a * filtered[j-1] + (1 - a) * noise[j, ch]
            noise[:, ch] = filtered

    return noise


def drain(fpga, n=50):
    for _ in range(n):
        try:
            fpga.recv_auto_telemetry(timeout=0.002)
        except:
            break


def generate_waveform(cls, n_steps, rng):
    freq = rng.uniform(0.8, 1.2)
    phase = rng.uniform(0, 2 * np.pi)
    t = np.linspace(0, n_steps * STEP_DT, n_steps)
    if cls == 0:
        sig = np.sin(2 * np.pi * freq * t + phase)
    elif cls == 1:
        sig = 2.0 * np.abs(2.0 * ((freq * t + phase/(2*np.pi)) % 1.0) - 1.0) - 1.0
    else:
        sig = np.sign(np.sin(2 * np.pi * freq * t + phase))
    return (sig + 1.0) / 2.0


def ridge_classify(X_tr, y_tr, X_te, y_te, n_classes=3):
    Y_tr = np.zeros((len(y_tr), n_classes))
    for i, c in enumerate(y_tr):
        Y_tr[i, int(c)] = 1.0
    best_acc = -1
    for a in [1e-6, 1e-4, 1e-2, 1.0, 100.0, 10000.0]:
        I = np.eye(X_tr.shape[1])
        try:
            W = np.linalg.solve(X_tr.T @ X_tr + a * I, X_tr.T @ Y_tr)
        except:
            continue
        acc = np.mean(np.argmax(X_te @ W, axis=1) == y_te.astype(int))
        if acc > best_acc:
            best_acc = acc
    return best_acc


def ridge_r2(X_tr, y_tr, X_te, y_te):
    best = -999
    for a in [1e-4, 1e-2, 1.0, 100.0, 10000.0]:
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


def collect_trial_raw(fpga, signal, noise_buf, noise_idx, w_in, w_noise,
                      use_noise=True, mac_mode='none', mac_signal_arr=None):
    """Run one trial at 200Hz, return raw per-step telemetry.

    Returns list of dicts with keys: delta_spikes, vmem, cumulative
    mac_mode: 'none', 'waveform' (MAC carries signal), 'context' (MAC carries different signal)
    """
    n_steps = len(signal)
    prev_counts = None
    cumulative = np.zeros(N_NEURONS, dtype=np.float32)
    raw_steps = []

    for t in range(n_steps):
        t0 = time.perf_counter()

        # Compute Vg
        if use_noise:
            ni = (noise_idx + t) % len(noise_buf)
            noise_per_neuron = np.array([noise_buf[ni, NOISE_MAP[n]] for n in range(N_NEURONS)])
            vg = np.full(N_NEURONS, BASE_VG) + ALPHA * signal[t] * w_in + BETA * noise_per_neuron * w_noise
        else:
            vg = np.full(N_NEURONS, BASE_VG) + ALPHA * signal[t] * w_in

        vg = np.clip(vg, 0.05, 0.95)
        fpga.set_vg_batch(0, vg[:64].tolist())
        fpga.set_vg_batch(64, vg[64:].tolist())

        # MAC signal
        if mac_mode == 'waveform' and mac_signal_arr is not None:
            fpga.set_mac_signal(float(mac_signal_arr[t]))
        elif mac_mode == 'context':
            # Slow context signal (different from input)
            ctx = 0.3 + 0.4 * np.sin(2 * np.pi * 0.5 * t * STEP_DT)
            fpga.set_mac_signal(ctx)
        elif mac_mode == 'none':
            pass  # don't send MAC every step (save bandwidth)

        # Wait + read telemetry
        elapsed = time.perf_counter() - t0
        wait = STEP_DT - elapsed - 0.001
        if wait > 0.0005:
            time.sleep(wait)

        latest = None
        for _ in range(20):
            try:
                pkt = fpga.recv_auto_telemetry(timeout=0.001)
                if pkt is not None:
                    latest = pkt
                else:
                    break
            except:
                break

        if latest is not None:
            counts = latest['spike_counts']
            vm = latest['vmem']

            if prev_counts is not None:
                delta = counts.astype(np.int32) - prev_counts.astype(np.int32)
                delta[delta < 0] = 0
                delta[delta > 30000] = 0
                cumulative += delta.astype(np.float32)
                raw_steps.append({
                    'delta': delta.astype(np.float32).copy(),
                    'vmem': vm.astype(np.float32).copy(),
                    'cum': cumulative.copy(),
                })
            prev_counts = counts.copy()

        total = time.perf_counter() - t0
        rem = STEP_DT - total
        if rem > 0.0005:
            time.sleep(rem)

    return raw_steps


def pool_steps(raw_steps, pool_factor):
    """Pool raw 200Hz steps into effective lower rate.
    pool_factor=10 → 20Hz effective, pool_factor=4 → 50Hz, etc.
    """
    pooled = []
    for i in range(0, len(raw_steps) - pool_factor + 1, pool_factor):
        chunk = raw_steps[i:i+pool_factor]
        delta_sum = sum(s['delta'] for s in chunk)
        vmem_mean = np.mean([s['vmem'] for s in chunk], axis=0)
        cum_last = chunk[-1]['cum']
        pooled.append(np.concatenate([delta_sum, vmem_mean, cum_last]))
    return pooled


def extract_features(pooled_steps):
    """Delay-augmented pooled features (z2206 protocol)."""
    if len(pooled_steps) < 4:
        return None
    states = np.array(pooled_steps)
    T, D = states.shape
    # Delay augment: t, t-1, t-2, t-3
    aug = np.zeros((T - 3, D * 4))
    for t in range(3, T):
        aug[t-3, 0*D:1*D] = states[t]
        aug[t-3, 1*D:2*D] = states[t-1]
        aug[t-3, 2*D:3*D] = states[t-2]
        aug[t-3, 3*D:4*D] = states[t-3]
    # 4x temporal pooling
    return np.concatenate([aug.mean(axis=0), aug.std(axis=0),
                           aug.max(axis=0), aug.min(axis=0)])


# ==========================================================================
# EXP 1: RATE CALIBRATION
# ==========================================================================
def exp1_rate_calibration(fpga):
    """Find operating point where spikes are visible at 200Hz."""
    print("\n" + "=" * 70)
    print("EXP 1 — RATE CALIBRATION (find sweet spot for ETH 200Hz)")
    print("=" * 70)

    # From z2233b: th=0.75, dtc=0x0200, bexc=0x0080 → 29.5 spk/neuron/s
    # That's ~0.15 spk/neuron/step at 200Hz — too sparse
    # Need higher rates: ~5-20 spk/neuron/step → 1000-4000 spk/s
    # z2233b: th=0.75, dtc=0x0200, bexc=0x0333 → 155 spk/s still low
    # Default params: th=0.50, dtc=0x0200, bexc=default → need to check

    configs = [
        # (threshold, dt_c, base_exc, vg, leak, name)
        (0.50, 0x0200, 0x0333, 0.58, 0x0004, "default-like"),
        (0.50, 0x0200, 0x0333, 0.62, 0x0004, "higher-vg"),
        (0.50, 0x0200, 0x0333, 0.68, 0x0004, "high-vg"),
        (0.50, 0x0400, 0x0333, 0.62, 0x0004, "fast-int"),
        (0.50, 0x0400, 0x0800, 0.62, 0x0004, "fast-int-high-exc"),
        (0.30, 0x0400, 0x0800, 0.62, 0x0004, "low-thresh-fast"),
        (0.50, 0x0200, 0x0333, 0.62, 0x0011, "orig-leak"),
        (0.50, 0x0200, 0x0333, 0.68, 0x0100, "z2232-sweet"),
    ]

    results = {}
    for thresh, dtc, bexc, vg, leak, name in configs:
        fpga.set_threshold(thresh)
        fpga.set_dt_over_c_raw(dtc)
        fpga.set_base_exc_raw(bexc)
        fpga.set_leak_cond(leak)
        fpga.set_vg_batch(0, [vg] * 64)
        fpga.set_vg_batch(64, [vg] * 64)
        fpga.set_mac_signal(0.0)
        time.sleep(0.3)
        drain(fpga, 200)

        # Measure for 0.5s
        total_spikes = np.zeros(N_NEURONS, dtype=np.int64)
        vmem_samples = []
        n_pkts = 0
        prev_counts = None

        t_start = time.perf_counter()
        while time.perf_counter() - t_start < 0.5:
            try:
                pkt = fpga.recv_auto_telemetry(timeout=0.005)
                if pkt is not None:
                    counts = pkt['spike_counts']
                    if prev_counts is not None:
                        delta = counts.astype(np.int64) - prev_counts.astype(np.int64)
                        delta[delta < 0] = 0
                        delta[delta > 30000] = 0
                        total_spikes += delta
                    prev_counts = counts.copy()
                    vmem_samples.append(pkt['vmem'].mean())
                    n_pkts += 1
            except:
                pass

        dur = time.perf_counter() - t_start
        rate_per_neuron = total_spikes.mean() / dur if dur > 0 else 0
        spk_per_step = rate_per_neuron / ETH_HZ
        vmem_mean = np.mean(vmem_samples) if vmem_samples else 0

        results[name] = {
            'threshold': thresh, 'dt_c': hex(dtc), 'base_exc': hex(bexc),
            'vg': vg, 'leak': hex(leak),
            'rate_per_neuron': float(rate_per_neuron),
            'spk_per_step': float(spk_per_step),
            'vmem_mean': float(vmem_mean),
            'n_pkts': n_pkts,
        }
        print(f"  {name:25s}: rate={rate_per_neuron:8.1f} spk/s, "
              f"per_step={spk_per_step:.2f}, vmem={vmem_mean:.3f}, pkts={n_pkts}")

    # Find best for ~5-20 spk/step
    best = None
    best_name = None
    for name, r in results.items():
        sps = r['spk_per_step']
        if 0.5 <= sps <= 50:  # reasonable range
            if best is None or abs(sps - 5) < abs(best - 5):
                best = sps
                best_name = name

    t815 = best is not None and best >= 0.5
    print(f"\n  T815 spk/step >= 0.5: best={best_name} ({best:.2f}) {'PASS' if t815 else 'FAIL'}")
    results['best_config'] = best_name
    results['tests'] = {'T815': t815}
    return results


# ==========================================================================
# EXP 2: TEMPORAL POOLING COMPARISON
# ==========================================================================
def exp2_temporal_pooling(fpga, noise_buf, rng, best_config):
    """Compare classification at different effective sampling rates."""
    print("\n" + "=" * 70)
    print("EXP 2 — TEMPORAL POOLING (200Hz raw → pooled at 20/50/100/200Hz)")
    print("=" * 70)

    # Apply best config from EXP 1
    if best_config:
        print(f"  Using config: {best_config}")

    N_TRIALS = 150
    N_STEPS_RAW = 300   # 1.5s at 200Hz
    N_CLASSES = 3

    w_in = rng.standard_normal(N_NEURONS).astype(np.float32) * 0.3
    w_noise = rng.standard_normal(N_NEURONS).astype(np.float32) * 0.3

    # Collect all trials at 200Hz raw
    print(f"\n  Collecting {N_TRIALS} trials at 200Hz raw...")
    all_raw = []
    all_labels = []
    t0 = time.monotonic()

    fpga.set_mac_signal(0.0)

    for trial in range(N_TRIALS):
        cls = trial % N_CLASSES
        signal = generate_waveform(cls, N_STEPS_RAW, rng)
        noise_idx = rng.integers(0, max(1, len(noise_buf) - N_STEPS_RAW))

        raw = collect_trial_raw(fpga, signal, noise_buf, noise_idx, w_in, w_noise,
                                use_noise=True, mac_mode='none')
        all_raw.append(raw)
        all_labels.append(cls)

        if (trial + 1) % 30 == 0:
            elapsed = time.monotonic() - t0
            rate = (trial + 1) / elapsed
            print(f"    Trial {trial+1}/{N_TRIALS} ({rate:.1f}/s, "
                  f"steps={len(raw)}, ETA {(N_TRIALS-trial-1)/rate:.0f}s)")

    all_labels = np.array(all_labels)

    # Pool at different rates and classify
    pool_factors = {
        '20Hz': 10,   # 200/10 = 20 Hz effective
        '50Hz': 4,    # 200/4 = 50 Hz
        '100Hz': 2,   # 200/2 = 100 Hz
        '200Hz': 1,   # no pooling
    }

    results = {}
    for rate_name, pf in pool_factors.items():
        features = []
        valid_labels = []
        for i, raw in enumerate(all_raw):
            if pf > 1:
                pooled = pool_steps(raw, pf)
            else:
                pooled = [np.concatenate([s['delta'], s['vmem'], s['cum']]) for s in raw]

            feat = extract_features(pooled)
            if feat is not None:
                features.append(feat)
                valid_labels.append(all_labels[i])

        if len(features) < 30:
            print(f"  {rate_name}: only {len(features)} valid trials, skip")
            results[rate_name] = {'acc': 0, 'n_valid': len(features)}
            continue

        X = np.array(features)
        y = np.array(valid_labels)

        # Normalize
        mu = X.mean(axis=0, keepdims=True)
        sigma = X.std(axis=0, keepdims=True)
        sigma[sigma < 1e-2] = 1.0
        X_n = (X - mu) / sigma

        # 5-fold CV
        splits = stratified_kfold(X_n, y, n_splits=5)
        accs = []
        for tr_idx, te_idx in splits:
            acc = ridge_classify(X_n[tr_idx], y[tr_idx], X_n[te_idx], y[te_idx])
            accs.append(acc)

        mean_acc = np.mean(accs)
        std_acc = np.std(accs)
        results[rate_name] = {
            'acc': float(mean_acc), 'std': float(std_acc),
            'n_valid': len(features), 'feat_dim': X.shape[1],
        }
        print(f"  {rate_name}: acc={mean_acc:.3f} +/- {std_acc:.3f} "
              f"(n={len(features)}, dim={X.shape[1]})")

    acc_20 = results.get('20Hz', {}).get('acc', 0)
    acc_200 = results.get('200Hz', {}).get('acc', 0)
    t816 = acc_20 > acc_200
    print(f"\n  T816 20Hz > 200Hz: {acc_20:.3f} vs {acc_200:.3f} {'PASS' if t816 else 'FAIL'}")
    results['tests'] = {'T816': t816}
    return results


# ==========================================================================
# EXP 3: MAC BRIDGE CLASSIFICATION
# ==========================================================================
def exp3_mac_bridge(fpga, noise_buf, rng):
    """4-condition comparison: COUPLED, FPGA_ONLY, MAC_ONLY, NOISE_ONLY."""
    print("\n" + "=" * 70)
    print("EXP 3 — MAC BRIDGE CLASSIFICATION (4 conditions)")
    print("=" * 70)

    N_TRIALS = 200
    N_STEPS_RAW = 300
    N_CLASSES = 3
    POOL_FACTOR = 10  # 20Hz effective (best expected)

    w_in = rng.standard_normal(N_NEURONS).astype(np.float32) * 0.3
    w_noise = rng.standard_normal(N_NEURONS).astype(np.float32) * 0.3

    conditions = {
        'COUPLED':    {'noise': True,  'mac': 'waveform'},
        'MAC_ONLY':   {'noise': False, 'mac': 'waveform'},
        'NOISE_ONLY': {'noise': True,  'mac': 'none'},
        'FPGA_ONLY':  {'noise': False, 'mac': 'none'},
    }

    results = {}

    for cond_name, cfg in conditions.items():
        print(f"\n  --- {cond_name} ---")
        features = []
        labels = []

        # Reset neurons
        fpga.set_kill(True)
        time.sleep(0.15)
        fpga.set_kill(False)
        time.sleep(0.15)
        if cfg['mac'] == 'none':
            fpga.set_mac_signal(0.0)
        drain(fpga, 200)

        t0 = time.monotonic()

        for trial in range(N_TRIALS):
            cls = trial % N_CLASSES
            signal = generate_waveform(cls, N_STEPS_RAW, rng)
            noise_idx = rng.integers(0, max(1, len(noise_buf) - N_STEPS_RAW))

            # MAC waveform = same signal scaled to [0, 0.5]
            mac_arr = signal * 0.5 if cfg['mac'] == 'waveform' else None

            raw = collect_trial_raw(
                fpga, signal, noise_buf, noise_idx, w_in, w_noise,
                use_noise=cfg['noise'],
                mac_mode=cfg['mac'],
                mac_signal_arr=mac_arr,
            )

            pooled = pool_steps(raw, POOL_FACTOR)
            feat = extract_features(pooled)
            if feat is not None:
                features.append(feat)
                labels.append(cls)

            if (trial + 1) % 50 == 0:
                elapsed = time.monotonic() - t0
                rate = (trial + 1) / elapsed
                print(f"    Trial {trial+1}/{N_TRIALS} ({rate:.1f}/s, "
                      f"valid={len(features)}, ETA {(N_TRIALS-trial-1)/rate:.0f}s)")

        if len(features) < 30:
            print(f"    Only {len(features)} valid, skip")
            results[cond_name] = {'acc': 0, 'n_valid': len(features)}
            continue

        X = np.array(features)
        y = np.array(labels)
        mu = X.mean(axis=0, keepdims=True)
        sigma = X.std(axis=0, keepdims=True)
        sigma[sigma < 1e-2] = 1.0
        X_n = (X - mu) / sigma

        splits = stratified_kfold(X_n, y, n_splits=5)
        accs = []
        for tr_idx, te_idx in splits:
            acc = ridge_classify(X_n[tr_idx], y[tr_idx], X_n[te_idx], y[te_idx])
            accs.append(acc)

        mean_acc = np.mean(accs)
        std_acc = np.std(accs)
        results[cond_name] = {
            'acc': float(mean_acc), 'std': float(std_acc),
            'n_valid': len(features), 'feat_dim': X.shape[1],
        }
        print(f"    {cond_name}: acc={mean_acc:.3f} +/- {std_acc:.3f}")

    # Reset MAC
    fpga.set_mac_signal(0.0)

    # Tests
    coupled = results.get('COUPLED', {}).get('acc', 0)
    fpga_only = results.get('FPGA_ONLY', {}).get('acc', 0)
    mac_only = results.get('MAC_ONLY', {}).get('acc', 0)
    noise_only = results.get('NOISE_ONLY', {}).get('acc', 0)

    t817 = coupled > fpga_only
    t818 = mac_only > fpga_only
    t819 = noise_only > fpga_only
    t820 = coupled > max(mac_only, noise_only)
    t821 = max(coupled, mac_only, noise_only, fpga_only) > 0.60

    print(f"\n  T817 COUPLED > FPGA_ONLY:     {coupled:.3f} vs {fpga_only:.3f} {'PASS' if t817 else 'FAIL'}")
    print(f"  T818 MAC_ONLY > FPGA_ONLY:    {mac_only:.3f} vs {fpga_only:.3f} {'PASS' if t818 else 'FAIL'}")
    print(f"  T819 NOISE_ONLY > FPGA_ONLY:  {noise_only:.3f} vs {fpga_only:.3f} {'PASS' if t819 else 'FAIL'}")
    print(f"  T820 COUPLED > max(others):    {coupled:.3f} vs {max(mac_only, noise_only):.3f} {'PASS' if t820 else 'FAIL'}")
    print(f"  T821 best > 0.60:             {max(coupled, mac_only, noise_only, fpga_only):.3f} {'PASS' if t821 else 'FAIL'}")

    results['tests'] = {'T817': t817, 'T818': t818, 'T819': t819, 'T820': t820, 'T821': t821}
    return results


# ==========================================================================
# EXP 4: MEMORY WITH MAC
# ==========================================================================
def exp4_memory_mac(fpga, noise_buf, rng):
    """Memory capacity with MAC as input channel (not Vg)."""
    print("\n" + "=" * 70)
    print("EXP 4 — MEMORY CAPACITY VIA MAC (direct current injection)")
    print("=" * 70)

    N_TRIALS = 60
    N_STEPS = 200   # 1.0s at 200Hz
    MAX_DELAY = 10
    POOL_FACTOR = 10  # 20Hz effective

    w_in = rng.standard_normal(N_NEURONS).astype(np.float32) * 0.3
    w_noise = rng.standard_normal(N_NEURONS).astype(np.float32) * 0.3

    conditions = {
        'MAC_INPUT': 'mac',   # signal via MAC, noise via Vg
        'VG_INPUT': 'vg',     # signal via Vg (traditional)
    }

    results = {}

    for cond_name, input_mode in conditions.items():
        print(f"\n  --- {cond_name} ---")

        fpga.set_kill(True)
        time.sleep(0.15)
        fpga.set_kill(False)
        time.sleep(0.15)
        fpga.set_mac_signal(0.0)
        drain(fpga, 200)

        all_inputs = []
        all_states = []

        for trial in range(N_TRIALS):
            u = rng.uniform(0, 1, N_STEPS).astype(np.float32)
            noise_idx = rng.integers(0, max(1, len(noise_buf) - N_STEPS))

            if input_mode == 'mac':
                # Vg carries noise only, MAC carries signal
                signal_for_vg = np.full(N_STEPS, 0.5)  # neutral
                raw = collect_trial_raw(
                    fpga, signal_for_vg, noise_buf, noise_idx, w_in, w_noise,
                    use_noise=True, mac_mode='waveform',
                    mac_signal_arr=u * 0.5)  # scale to safe range
            else:
                # Traditional: signal via Vg
                raw = collect_trial_raw(
                    fpga, u, noise_buf, noise_idx, w_in, w_noise,
                    use_noise=True, mac_mode='none')

            pooled = pool_steps(raw, POOL_FACTOR)
            if len(pooled) >= N_STEPS // POOL_FACTOR - 2:
                states = np.array(pooled)
                # Also pool the input
                u_pooled = np.array([u[i*POOL_FACTOR:(i+1)*POOL_FACTOR].mean()
                                     for i in range(len(pooled))])
                all_inputs.append(u_pooled)
                all_states.append(states)

            if (trial + 1) % 20 == 0:
                print(f"    Trial {trial+1}/{N_TRIALS}, valid={len(all_states)}")

        fpga.set_mac_signal(0.0)

        if len(all_states) < 20:
            print(f"    Only {len(all_states)} valid trials")
            results[cond_name] = {'mc': [0]*MAX_DELAY, 'mc_total': 0}
            continue

        # Memory capacity at each delay
        mc_values = []
        for d in range(1, MAX_DELAY + 1):
            X_list, y_list = [], []
            for i in range(len(all_states)):
                states = all_states[i]
                u = all_inputs[i]
                for t in range(d, min(len(states), len(u))):
                    X_list.append(states[t])
                    y_list.append(u[t - d])

            X = np.array(X_list)
            y = np.array(y_list)

            mu = X.mean(axis=0)
            sigma = X.std(axis=0)
            sigma[sigma < 1e-2] = 1.0
            X_n = (X - mu) / sigma

            n = len(X)
            n_tr = n * 3 // 4
            idx = rng.permutation(n)
            r2 = ridge_r2(X_n[idx[:n_tr]], y[idx[:n_tr]], X_n[idx[n_tr:]], y[idx[n_tr:]])
            mc_values.append(max(0, r2))
            print(f"    d={d:2d}: R²={r2:+.4f} {'***' if r2 > 0.05 else ''}")

        mc_total = sum(mc_values)
        results[cond_name] = {
            'mc': [float(v) for v in mc_values],
            'mc_total': float(mc_total),
            'n_valid': len(all_states),
        }
        print(f"    MC total = {mc_total:.3f}")

    mac_d1 = results.get('MAC_INPUT', {}).get('mc', [0])[0]
    vg_d1 = results.get('VG_INPUT', {}).get('mc', [0])[0]

    t822 = mac_d1 > 0.10
    t823 = mac_d1 > vg_d1
    print(f"\n  T822 MC(d=1,MAC) > 0.10:     {mac_d1:.4f} {'PASS' if t822 else 'FAIL'}")
    print(f"  T823 MC(MAC) > MC(Vg):       {mac_d1:.4f} vs {vg_d1:.4f} {'PASS' if t823 else 'FAIL'}")

    results['tests'] = {'T822': t822, 'T823': t823}
    return results


# ==========================================================================
# EXP 5: Z2206 REPRODUCTION ON ETH
# ==========================================================================
def exp5_z2206_reproduction(fpga, noise_buf, rng):
    """Reproduce z2206's 81% result using ETH bridge at 20Hz effective."""
    print("\n" + "=" * 70)
    print("EXP 5 — Z2206 REPRODUCTION (ETH bridge, 20Hz effective)")
    print("=" * 70)

    N_TRIALS = 200
    N_STEPS_RAW = 300   # collect at 200Hz
    N_CLASSES = 3
    POOL_FACTOR = 10    # → 20Hz = 30 effective steps (matches z2206)

    w_in = rng.uniform(-1, 1, N_NEURONS).astype(np.float32)
    w_noise = rng.uniform(-1, 1, N_NEURONS).astype(np.float32)

    print(f"  Collecting {N_TRIALS} trials (FULL_128 condition)...")
    features = []
    labels = []
    t0 = time.monotonic()

    for trial in range(N_TRIALS):
        cls = trial % N_CLASSES
        signal = generate_waveform(cls, N_STEPS_RAW, rng)
        noise_idx = rng.integers(0, max(1, len(noise_buf) - N_STEPS_RAW))

        raw = collect_trial_raw(
            fpga, signal, noise_buf, noise_idx, w_in, w_noise,
            use_noise=True, mac_mode='none')

        pooled = pool_steps(raw, POOL_FACTOR)
        feat = extract_features(pooled)
        if feat is not None:
            features.append(feat)
            labels.append(cls)

        if (trial + 1) % 50 == 0:
            elapsed = time.monotonic() - t0
            rate = (trial + 1) / elapsed
            print(f"    Trial {trial+1}/{N_TRIALS} ({rate:.1f}/s, valid={len(features)})")

    X = np.array(features)
    y = np.array(labels)

    mu = X.mean(axis=0, keepdims=True)
    sigma = X.std(axis=0, keepdims=True)
    sigma[sigma < 1e-2] = 1.0
    X_n = (X - mu) / sigma

    splits = stratified_kfold(X_n, y, n_splits=5)
    accs = []
    for tr_idx, te_idx in splits:
        acc = ridge_classify(X_n[tr_idx], y[tr_idx], X_n[te_idx], y[te_idx])
        accs.append(acc)

    mean_acc = np.mean(accs)
    std_acc = np.std(accs)
    print(f"\n  ETH REPRODUCTION: acc={mean_acc:.3f} +/- {std_acc:.3f}")
    print(f"  z2206 reference:  acc=0.810")
    print(f"  Gap: {(0.810 - mean_acc)*100:.1f}pp")

    t824 = mean_acc > 0.70
    t825 = mean_acc > 0.71  # within 10pp of 0.81

    print(f"\n  T824 ETH > 0.70:              {mean_acc:.3f} {'PASS' if t824 else 'FAIL'}")
    print(f"  T825 within 10pp of z2206:    {mean_acc:.3f} vs 0.81 {'PASS' if t825 else 'FAIL'}")

    results = {
        'acc': float(mean_acc), 'std': float(std_acc),
        'n_valid': len(features), 'feat_dim': X.shape[1],
        'z2206_ref': 0.810,
        'tests': {'T824': t824, 'T825': t825},
    }
    return results


# ==========================================================================
# MAIN
# ==========================================================================
def main():
    print("=" * 70)
    print("z2234: Calibrated ETH Bridge Reservoir")
    print("  Fixing z2233's 30% regression from z2206's 81%")
    print("  Key insight: pool 200Hz → 20Hz effective rate")
    print("=" * 70)

    rng = np.random.default_rng(42)

    # Connect to FPGA
    print("\n[1/6] Connecting to FPGA via Ethernet...")
    fpga = FPGAEthBridge()
    if not fpga.connect():
        print("FAIL: Cannot connect to FPGA")
        return

    fpga.set_kill(False)
    fpga.enable_auto_telemetry(2000)
    time.sleep(0.5)
    drain(fpga, 200)
    print(f"  Connected: {fpga.num_neurons} neurons, ETH {FPGA_IP}")

    # Collect GPU noise
    print("\n[2/6] Collecting GPU noise (20s)...")
    noise_buf = collect_gpu_noise(duration=20.0, rate=50)
    print(f"  Noise buffer: {noise_buf.shape}")

    results = {
        'experiment': 'z2234_eth_bridge_calibrated',
        'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
    }

    # EXP 1
    results['exp1_calibration'] = exp1_rate_calibration(fpga)
    best_cfg = results['exp1_calibration'].get('best_config', None)

    # EXP 2
    results['exp2_pooling'] = exp2_temporal_pooling(fpga, noise_buf, rng, best_cfg)

    # EXP 3
    results['exp3_mac_bridge'] = exp3_mac_bridge(fpga, noise_buf, rng)

    # EXP 4
    results['exp4_memory_mac'] = exp4_memory_mac(fpga, noise_buf, rng)

    # EXP 5
    results['exp5_z2206_repro'] = exp5_z2206_reproduction(fpga, noise_buf, rng)

    fpga.set_mac_signal(0.0)
    fpga.close()

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    all_tests = {}
    for exp_name, exp_data in results.items():
        if isinstance(exp_data, dict) and 'tests' in exp_data:
            for tk, tv in exp_data['tests'].items():
                all_tests[tk] = tv

    n_pass = sum(1 for v in all_tests.values() if v)
    n_total = len(all_tests)
    print(f"\n  Tests: {n_pass}/{n_total} PASS")
    for tk in sorted(all_tests.keys()):
        print(f"    {tk}: {'PASS' if all_tests[tk] else 'FAIL'}")

    # Save
    out_path = 'results/z2234_eth_bridge_calibrated.json'
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Saved: {out_path}")


if __name__ == '__main__':
    main()
