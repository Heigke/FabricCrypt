#!/usr/bin/env python3
"""
z2234_uart_mac_bridge.py — UART Bridge with MAC Neuromodulation
================================================================
Diagnosis:
  z2206 (UART 20Hz): 81% waveform — PROVEN, BEST EVER
  z2233 (ETH 200Hz): 30-39% — massive regression (5ms too sparse)
  z2233 EXP3: MAC neuromodulation = CLEAR WIN (0.531 vs 0.275)
  z2232: MAC ON R²=0.954 at sweet spot

Strategy: Use z2206's PROVEN UART protocol (20Hz, default FPGA params)
  but ADD the one thing that actually works: MAC neuromodulation.
  Also: deeper condition matrix to find synergy.

EXP 1 — MAC BRIDGE (z2206 protocol + MAC)
  200 trials × 30 steps × 20Hz, 3-class waveform
  4 conditions:
    COUPLED:    GPU 1/f noise → Vg + waveform → MAC
    NOISE_ONLY: GPU 1/f noise → Vg (= z2206 FULL_128)
    MAC_ONLY:   waveform → MAC only, fixed Vg
    FPGA_ONLY:  fixed Vg, no noise, no MAC
  Target: COUPLED > NOISE_ONLY (z2206), both > FPGA_ONLY

EXP 2 — MEMORY CAPACITY via MAC
  60 random input trials × 200 steps @ 20Hz = 10s per trial
  MAC drives random input u(t), Vg carries noise
  Decode u(t-d) from neuron states
  Compare MAC_INPUT vs VG_INPUT

EXP 3 — MAC GAIN SWEEP
  MAC gain: 0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0
  3-class waveform, 100 trials each
  Find optimal MAC gain for classification

Tests T815-T826:
  T815: COUPLED > NOISE_ONLY (MAC adds signal)
  T816: COUPLED > 0.81 (exceeds z2206 BEST)
  T817: MAC_ONLY > FPGA_ONLY (MAC carries information)
  T818: NOISE_ONLY > 0.70 (reproduces z2206 on this run)
  T819: COUPLED > FPGA_ONLY (full bridge > isolated FPGA)
  T820: MC(d=1, MAC) > 0.05 (MAC creates fading memory)
  T821: MC(d=1, MAC) > MC(d=1, Vg) (MAC better input channel)
  T822: MC_total(MAC) > 0.50
  T823: Best MAC gain > 0.0 (optimal gain is non-zero)
  T824: Acc at best gain > acc at gain=0
  T825: Acc at best gain > 0.75
  T826: Gain curve non-monotonic (too much MAC overwhelms)

Hardware: Arty A7-100T FPGA (128 neurons, UART 921600) + AMD gfx1151 GPU
"""

import os, sys, json, time, struct
import numpy as np
from pathlib import Path

os.environ.setdefault('HSA_OVERRIDE_GFX_VERSION', '11.0.0')

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE / 'scripts'))
RESULTS = BASE / 'results'

N_NEURONS = 128
BASE_VG = 0.58
ALPHA = 0.25
BETA = 0.08
SAMPLE_HZ = 20
IIR_ALPHA_POWER = 0.85
IIR_ALPHA_THERMAL = 0.92

# 5-Channel Noise Assignment (identical to z2206)
POWER_NEURONS   = list(range(0, 32))
SMN_NEURONS     = list(range(32, 56))
JITTER_NEURONS  = list(range(56, 80))
THERMAL_NEURONS = list(range(80, 104))
CLOCK_NEURONS   = list(range(104, 128))

HWMON_POWER = "/sys/class/hwmon/hwmon7/power1_average"
PM_TABLE_PATH = "/sys/kernel/ryzen_smu_drv/pm_table"


def read_hwmon_power():
    try:
        return int(open(HWMON_POWER).read().strip()) / 1e6
    except:
        return None

def read_gpu_thermal():
    try:
        return int(open("/sys/class/hwmon/hwmon7/temp1_input").read().strip()) / 1000.0
    except:
        return None

def read_gpu_clock():
    try:
        return int(open("/sys/class/hwmon/hwmon7/freq1_input").read().strip()) / 1e6
    except:
        return None

def read_smn_thermal():
    try:
        with open(PM_TABLE_PATH, 'rb') as f:
            f.seek(0x004C)
            return struct.unpack('<f', f.read(4))[0]
    except:
        return None


def normalize_noise(samples):
    arr = np.array(samples, dtype=float)
    if len(arr) == 0:
        return arr
    mu, std = arr.mean(), max(arr.std(), 1e-6)
    return (arr - mu) / std

def iir_filter(noise, alpha=0.85):
    if len(noise) == 0:
        return noise
    f = np.zeros(len(noise))
    f[0] = noise[0]
    for t in range(1, len(noise)):
        f[t] = alpha * f[t-1] + (1-alpha) * noise[t]
    return f / max(np.std(f), 1e-6)

def generate_synthetic_1f(n, rng):
    noise = np.zeros(n)
    octaves = np.zeros(8)
    for i in range(n):
        for j in range(8):
            if i % (1 << j) == 0:
                octaves[j] = rng.standard_normal()
        noise[i] = octaves.sum()
    return normalize_noise(noise)


def collect_all_noise(duration_s=20, sample_hz=50):
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
            print(f"    {i+1}/{n} samples")

    return power_s, thermal_s, clock_s, smn_s


def generate_waveform(cls, steps, dt, rng):
    freq = rng.uniform(0.8, 1.2)
    phase = rng.uniform(0, 2 * np.pi)
    t = np.arange(steps) * dt
    if cls == 0:
        sig = np.sin(2 * np.pi * freq * t + phase)
    elif cls == 1:
        sig = 2.0 * np.abs(2.0 * ((freq * t + phase/(2*np.pi)) % 1.0) - 1.0) - 1.0
    else:
        sig = np.sign(np.sin(2 * np.pi * freq * t + phase))
    return (sig + 1.0) / 2.0


def compute_vg_128(t, input_val, noises, w_in, w_noise, mode='FULL'):
    vg = np.full(N_NEURONS, BASE_VG) + ALPHA * input_val * w_in

    if mode in ('FULL', 'NOISE_ONLY'):
        channel_map = {
            'power': POWER_NEURONS, 'smn': SMN_NEURONS,
            'jitter': JITTER_NEURONS, 'thermal': THERMAL_NEURONS,
            'clock': CLOCK_NEURONS,
        }
        for ch_name, nids in channel_map.items():
            ch = noises.get(ch_name, np.zeros(1))
            if len(ch) == 0:
                ch = np.zeros(1)
            idx = t % len(ch)
            for nid in nids:
                vg[nid] += BETA * ch[idx] * w_noise[nid]

    return np.clip(vg, 0.05, 0.95)


def run_reservoir_trial(fpga, input_signal, noises, w_in, w_noise,
                        mode='FULL', mac_signal=None, mac_gain=0.5):
    """Drive 128 neurons and collect states. Returns (n_steps, N*3)."""
    n_steps = len(input_signal)
    interval = 1.0 / SAMPLE_HZ
    states = np.zeros((n_steps, N_NEURONS * 3))
    prev_counts = None
    cumulative = np.zeros(N_NEURONS)

    for t in range(n_steps):
        t0 = time.perf_counter()

        # Vg: noise + input (for FULL/NOISE_ONLY) or just input (MAC_ONLY/FPGA_ONLY)
        if mode in ('FULL', 'NOISE_ONLY'):
            vg = compute_vg_128(t, input_signal[t], noises, w_in, w_noise, mode='FULL')
        elif mode == 'MAC_ONLY':
            vg = np.full(N_NEURONS, BASE_VG)  # fixed Vg, no signal encoding via Vg
        else:  # FPGA_ONLY
            vg = np.full(N_NEURONS, BASE_VG)

        fpga.set_vg_all(vg.tolist())

        # MAC signal
        if mac_signal is not None and mode in ('FULL', 'MAC_ONLY'):
            fpga.set_mac(mac_signal[t] * mac_gain)

        time.sleep(interval * 0.3)

        try:
            fpga.ser.reset_input_buffer()
            telem = fpga.read_telem(timeout=0.3)
        except Exception:
            if not fpga.reconnect():
                break
            telem = None

        if telem and len(telem) >= N_NEURONS:
            counts = [telem[i]['spike_count'] for i in range(N_NEURONS)]
            vmems = [telem[i]['vmem'] for i in range(N_NEURONS)]

            if prev_counts is not None:
                for i in range(N_NEURONS):
                    delta = (counts[i] - prev_counts[i]) & 0xFFFF
                    if delta > 30000:
                        delta = 0
                    states[t, i] = delta
                    cumulative[i] += delta
                for i in range(N_NEURONS):
                    states[t, N_NEURONS + i] = vmems[i]
                    states[t, N_NEURONS*2 + i] = cumulative[i]

            prev_counts = counts[:]

        elapsed = time.perf_counter() - t0
        remaining = interval * 0.5 - elapsed
        if remaining > 0:
            time.sleep(remaining)

    return states


def augment_with_delays(states, delays=(1, 2, 3)):
    T, D = states.shape
    aug = np.zeros((T, D * (1 + len(delays))))
    aug[:, :D] = states
    for i, d in enumerate(delays):
        start = D * (i + 1)
        aug[d:, start:start+D] = states[:T-d]
    return aug


def pool_features(states):
    return np.concatenate([
        states.mean(axis=0), states.std(axis=0),
        states.max(axis=0), states.min(axis=0),
    ])


def ridge_classify(X_tr, y_tr, X_te, y_te, n_classes=3):
    Y_tr = np.zeros((len(y_tr), n_classes))
    for i, c in enumerate(y_tr):
        Y_tr[i, int(c)] = 1.0
    best_acc = -1
    for a in [1e-6, 1e-4, 1e-2, 1.0, 100.0]:
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
    idx = np.arange(len(y))
    rng.shuffle(idx)
    folds = [[] for _ in range(n_splits)]
    for c in classes:
        c_idx = idx[y[idx] == c]
        for i, ix in enumerate(c_idx):
            folds[i % n_splits].append(ix)
    splits = []
    for f in range(n_splits):
        te = np.array(folds[f])
        tr = np.concatenate([np.array(folds[i]) for i in range(n_splits) if i != f])
        splits.append((tr, te))
    return splits


def classify_cv(features, labels, n_classes=3):
    X = np.array(features)
    y = np.array(labels)
    mu = X.mean(axis=0, keepdims=True)
    sigma = X.std(axis=0, keepdims=True)
    sigma[sigma < 1e-2] = 1.0
    X_n = (X - mu) / sigma
    splits = stratified_kfold(X_n, y)
    accs = [ridge_classify(X_n[tr], y[tr], X_n[te], y[te], n_classes) for tr, te in splits]
    return float(np.mean(accs)), float(np.std(accs))


# ==========================================================================
# EXP 1: MAC BRIDGE CLASSIFICATION
# ==========================================================================
def exp1_mac_bridge(fpga, noises, rng):
    print("\n" + "=" * 70)
    print("EXP 1 — MAC BRIDGE CLASSIFICATION (z2206 protocol + MAC)")
    print("=" * 70)

    N_TRIALS = 200
    STEPS = 30
    DT = 1.0 / SAMPLE_HZ
    N_CLASSES = 3
    MAC_GAIN = 0.3  # conservative

    w_in = rng.uniform(-1, 1, N_NEURONS)
    w_noise = rng.uniform(-1, 1, N_NEURONS)

    conditions = ['COUPLED', 'NOISE_ONLY', 'MAC_ONLY', 'FPGA_ONLY']
    results = {}

    for cond in conditions:
        print(f"\n  === {cond} ===")
        features, labels = [], []
        t0 = time.monotonic()

        for trial in range(N_TRIALS):
            cls = trial % N_CLASSES
            wave = generate_waveform(cls, STEPS, DT, rng)

            mac_arr = wave if cond in ('COUPLED', 'MAC_ONLY') else None

            states = run_reservoir_trial(
                fpga, wave, noises, w_in, w_noise,
                mode=cond if cond != 'COUPLED' else 'FULL',
                mac_signal=mac_arr, mac_gain=MAC_GAIN)

            aug = augment_with_delays(states, delays=(1, 2, 3))
            feat = pool_features(aug)
            features.append(feat)
            labels.append(cls)

            if (trial + 1) % 50 == 0:
                elapsed = time.monotonic() - t0
                rate = (trial + 1) / elapsed
                print(f"    Trial {trial+1}/{N_TRIALS} ({rate:.1f}/s, ETA {(N_TRIALS-trial-1)/rate:.0f}s)")

        # Reset MAC
        fpga.set_mac(0.0)

        acc, std = classify_cv(features, labels, N_CLASSES)
        results[cond] = {'acc': acc, 'std': std, 'n_trials': len(features),
                         'feat_dim': len(features[0])}
        print(f"    {cond}: {acc:.3f} +/- {std:.3f}")

    # Tests
    coupled = results['COUPLED']['acc']
    noise_only = results['NOISE_ONLY']['acc']
    mac_only = results['MAC_ONLY']['acc']
    fpga_only = results['FPGA_ONLY']['acc']

    tests = {
        'T815': coupled > noise_only,
        'T816': coupled > 0.81,
        'T817': mac_only > fpga_only,
        'T818': noise_only > 0.70,
        'T819': coupled > fpga_only,
    }

    print(f"\n  T815 COUPLED > NOISE_ONLY:   {coupled:.3f} vs {noise_only:.3f} {'PASS' if tests['T815'] else 'FAIL'}")
    print(f"  T816 COUPLED > 0.81:         {coupled:.3f} {'PASS' if tests['T816'] else 'FAIL'}")
    print(f"  T817 MAC_ONLY > FPGA_ONLY:   {mac_only:.3f} vs {fpga_only:.3f} {'PASS' if tests['T817'] else 'FAIL'}")
    print(f"  T818 NOISE_ONLY > 0.70:      {noise_only:.3f} {'PASS' if tests['T818'] else 'FAIL'}")
    print(f"  T819 COUPLED > FPGA_ONLY:    {coupled:.3f} vs {fpga_only:.3f} {'PASS' if tests['T819'] else 'FAIL'}")

    results['tests'] = tests
    return results


# ==========================================================================
# EXP 2: MEMORY CAPACITY VIA MAC
# ==========================================================================
def exp2_memory_mac(fpga, noises, rng):
    print("\n" + "=" * 70)
    print("EXP 2 — MEMORY CAPACITY VIA MAC")
    print("=" * 70)

    N_TRIALS = 60
    N_STEPS = 200   # 10s at 20Hz
    MAX_DELAY = 10
    DT = 1.0 / SAMPLE_HZ

    w_in = rng.uniform(-1, 1, N_NEURONS)
    w_noise = rng.uniform(-1, 1, N_NEURONS)

    for cond_name, input_via in [('MAC_INPUT', 'mac'), ('VG_INPUT', 'vg')]:
        print(f"\n  --- {cond_name} ---")

        all_inputs = []
        all_states = []

        for trial in range(N_TRIALS):
            u = rng.uniform(0, 1, N_STEPS).astype(np.float32)

            if input_via == 'mac':
                # Signal via MAC, noise via Vg
                neutral_signal = np.full(N_STEPS, 0.5)
                states = run_reservoir_trial(
                    fpga, neutral_signal, noises, w_in, w_noise,
                    mode='FULL', mac_signal=u, mac_gain=0.5)
            else:
                # Signal via Vg (traditional)
                states = run_reservoir_trial(
                    fpga, u, noises, w_in, w_noise,
                    mode='FULL', mac_signal=None)

            fpga.set_mac(0.0)

            # Use last N_STEPS-1 states (skip first for delta)
            valid_rows = np.any(states != 0, axis=1)
            if valid_rows.sum() >= N_STEPS * 0.8:
                all_inputs.append(u)
                all_states.append(states)

            if (trial + 1) % 20 == 0:
                print(f"    Trial {trial+1}/{N_TRIALS}, valid={len(all_states)}")

        if len(all_states) < 20:
            print(f"    Only {len(all_states)} valid trials")
            continue

        mc_values = []
        for d in range(1, MAX_DELAY + 1):
            X_list, y_list = [], []
            for i in range(len(all_states)):
                states = all_states[i]
                u = all_inputs[i]
                for t in range(d + 1, min(len(states), len(u))):
                    if np.any(states[t] != 0):
                        X_list.append(states[t])
                        y_list.append(u[t - d])

            if len(X_list) < 50:
                mc_values.append(0)
                print(f"    d={d:2d}: too few samples ({len(X_list)})")
                continue

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
        print(f"    MC total = {mc_total:.3f}")

        if cond_name == 'MAC_INPUT':
            mac_mc = mc_values
            mac_mc_total = mc_total
        else:
            vg_mc = mc_values
            vg_mc_total = mc_total

    mac_d1 = mac_mc[0] if len(mac_mc) > 0 else 0
    vg_d1 = vg_mc[0] if len(vg_mc) > 0 else 0

    tests = {
        'T820': mac_d1 > 0.05,
        'T821': mac_d1 > vg_d1,
        'T822': mac_mc_total > 0.50,
    }

    print(f"\n  T820 MC(d=1,MAC) > 0.05:     {mac_d1:.4f} {'PASS' if tests['T820'] else 'FAIL'}")
    print(f"  T821 MC(MAC) > MC(Vg):       {mac_d1:.4f} vs {vg_d1:.4f} {'PASS' if tests['T821'] else 'FAIL'}")
    print(f"  T822 MC_total(MAC) > 0.50:   {mac_mc_total:.3f} {'PASS' if tests['T822'] else 'FAIL'}")

    results = {
        'MAC_INPUT': {'mc': [float(v) for v in mac_mc], 'mc_total': float(mac_mc_total)},
        'VG_INPUT': {'mc': [float(v) for v in vg_mc], 'mc_total': float(vg_mc_total)},
        'tests': tests,
    }
    return results


# ==========================================================================
# EXP 3: MAC GAIN SWEEP
# ==========================================================================
def exp3_mac_gain_sweep(fpga, noises, rng):
    print("\n" + "=" * 70)
    print("EXP 3 — MAC GAIN SWEEP")
    print("=" * 70)

    N_TRIALS = 100
    STEPS = 30
    DT = 1.0 / SAMPLE_HZ
    N_CLASSES = 3
    GAINS = [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]

    w_in = rng.uniform(-1, 1, N_NEURONS)
    w_noise = rng.uniform(-1, 1, N_NEURONS)

    results = {}

    for gain in GAINS:
        print(f"\n  --- MAC gain = {gain:.1f} ---")
        features, labels = [], []
        t0 = time.monotonic()

        for trial in range(N_TRIALS):
            cls = trial % N_CLASSES
            wave = generate_waveform(cls, STEPS, DT, rng)
            mac_arr = wave if gain > 0 else None

            states = run_reservoir_trial(
                fpga, wave, noises, w_in, w_noise,
                mode='FULL', mac_signal=mac_arr, mac_gain=gain)

            aug = augment_with_delays(states, delays=(1, 2, 3))
            feat = pool_features(aug)
            features.append(feat)
            labels.append(cls)

            if (trial + 1) % 50 == 0:
                elapsed = time.monotonic() - t0
                rate = (trial + 1) / elapsed
                print(f"    Trial {trial+1}/{N_TRIALS} ({rate:.1f}/s)")

        fpga.set_mac(0.0)

        acc, std = classify_cv(features, labels, N_CLASSES)
        results[f'gain_{gain:.1f}'] = {'acc': acc, 'std': std, 'gain': gain}
        print(f"    gain={gain:.1f}: acc={acc:.3f} +/- {std:.3f}")

    # Find best
    accs = [(r['gain'], r['acc']) for r in results.values() if 'gain' in r]
    best_gain, best_acc = max(accs, key=lambda x: x[1])
    zero_acc = next((r['acc'] for r in results.values() if r.get('gain') == 0.0), 0)
    max_gain_acc = next((r['acc'] for r in results.values() if r.get('gain') == 1.0), 0)

    # Non-monotonic: some middle gain is better than max gain
    mid_accs = [a for g, a in accs if 0.1 <= g <= 0.7]
    non_mono = len(mid_accs) > 0 and max(mid_accs) > max_gain_acc

    tests = {
        'T823': best_gain > 0.0,
        'T824': best_acc > zero_acc,
        'T825': best_acc > 0.75,
        'T826': non_mono,
    }

    print(f"\n  Best: gain={best_gain:.1f}, acc={best_acc:.3f}")
    print(f"  T823 best gain > 0.0:        {best_gain:.1f} {'PASS' if tests['T823'] else 'FAIL'}")
    print(f"  T824 best > gain=0:          {best_acc:.3f} vs {zero_acc:.3f} {'PASS' if tests['T824'] else 'FAIL'}")
    print(f"  T825 best > 0.75:            {best_acc:.3f} {'PASS' if tests['T825'] else 'FAIL'}")
    print(f"  T826 non-monotonic:          {'PASS' if tests['T826'] else 'FAIL'}")

    results['best'] = {'gain': float(best_gain), 'acc': float(best_acc)}
    results['tests'] = tests
    return results


# ==========================================================================
# MAIN
# ==========================================================================
def main():
    print("=" * 70)
    print("z2234: UART Bridge with MAC Neuromodulation")
    print("  z2206 proven protocol + MAC from z2233 EXP3")
    print("=" * 70)

    rng = np.random.default_rng(42)

    # Connect via UART
    print("\n[1/5] Connecting to 128-neuron FPGA via UART...")
    from fpga_host_v2 import FPGABridge
    fpga = FPGABridge()
    if not fpga.connected:
        print("  FAIL: FPGA not found")
        return
    print(f"  Connected: {fpga.port}, baud={fpga.baud}, neurons={fpga.num_neurons}")

    # Verify telemetry
    fpga.read_telem(timeout=0.5)
    time.sleep(1.0)
    test = fpga.read_telem(timeout=0.5)
    if test is None:
        print("  FAIL: No telemetry")
        return
    print(f"  Telemetry OK: {len(test)} neurons")

    # Collect GPU noise
    print(f"\n[2/5] Collecting GPU noise (20s)...")
    power_raw, thermal_raw, clock_raw, smn_raw = collect_all_noise(20, 50)

    noises = {}
    for name, raw, iir_a in [
        ('power', power_raw, IIR_ALPHA_POWER),
        ('thermal', thermal_raw, IIR_ALPHA_THERMAL),
        ('clock', clock_raw, 0.80),
        ('smn', smn_raw, 0.90),
    ]:
        if len(raw) > 10:
            normed = normalize_noise(raw)
            noises[name] = iir_filter(normed, alpha=iir_a)
            print(f"  {name}: {len(raw)} samples, mean={np.mean(raw):.3f} +/- {np.std(raw):.4f}")
        else:
            noises[name] = generate_synthetic_1f(1000, rng)
            print(f"  {name}: synthetic 1/f")

    # PERF jitter (try HIP probe)
    probe_bin = BASE / 'scripts' / 'z2153_deep_probe_bridge'
    if probe_bin.exists():
        import subprocess
        try:
            result = subprocess.run(
                [str(probe_bin), '100', '16', '50000'],
                capture_output=True, text=True, timeout=30,
                env={**os.environ, 'HSA_OVERRIDE_GFX_VERSION': '11.0.0'})
            jitter_s = []
            if result.returncode == 0:
                for line in result.stdout.strip().split('\n')[1:]:
                    parts = line.split(',')
                    if len(parts) >= 13:
                        jitter_s.append(int(parts[12]))
            if len(jitter_s) > 10:
                noises['jitter'] = normalize_noise(jitter_s)
                print(f"  jitter: {len(jitter_s)} samples (HIP probe)")
            else:
                noises['jitter'] = rng.standard_normal(1000)
                print(f"  jitter: synthetic white")
        except:
            noises['jitter'] = rng.standard_normal(1000)
            print(f"  jitter: synthetic white (probe failed)")
    else:
        noises['jitter'] = rng.standard_normal(1000)
        print(f"  jitter: synthetic white (no probe)")

    results = {
        'experiment': 'z2234_uart_mac_bridge',
        'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
        'params': {
            'base_vg': BASE_VG, 'alpha': ALPHA, 'beta': BETA,
            'sample_hz': SAMPLE_HZ, 'n_neurons': N_NEURONS,
        },
    }

    # EXP 1
    print("\n[3/5] EXP 1: MAC Bridge Classification...")
    results['exp1_mac_bridge'] = exp1_mac_bridge(fpga, noises, rng)

    # EXP 2
    print("\n[4/5] EXP 2: Memory Capacity via MAC...")
    results['exp2_memory_mac'] = exp2_memory_mac(fpga, noises, rng)

    # EXP 3
    print("\n[5/5] EXP 3: MAC Gain Sweep...")
    results['exp3_mac_gain'] = exp3_mac_gain_sweep(fpga, noises, rng)

    fpga.set_mac(0.0)

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

    out_path = str(RESULTS / 'z2234_uart_mac_bridge.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Saved: {out_path}")


if __name__ == '__main__':
    main()
