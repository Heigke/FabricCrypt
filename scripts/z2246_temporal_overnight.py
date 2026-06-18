#!/usr/bin/env python3
"""
z2246_temporal_overnight.py — Overnight Temporal Experiment Suite
================================================================
Builds on z2239's PROVEN approach (94.4% classification, MC=0.813):
  - 20Hz sampling (proven better SNR than 50Hz)
  - TUNED FPGA params (leak=0x0011, thresh=0.50, bias_gain=0.03125)
  - Delta features (spike[t]-spike[t-1])
  - RidgeClassifier with variance masking
  - GPU power noise coupling (IIR filtered)
  - Conditions: COUPLED, FPGA_ONLY, STATIC

Tests T1096-T1135 (40 tests across 8 experiments):
  EXP 1: Waveform classification baseline (confirm z2239 results)
  EXP 2: Temporal XOR at multiple delays (fading memory)
  EXP 3: Frequency discrimination (temporal vs static)
  EXP 4: Sequence order detection (temporal-only task)
  EXP 5: Memory capacity via classification (confirm z2241)
  EXP 6: Sustained vs transient response
  EXP 7: Multi-timescale integration
  EXP 8: Input reconstruction (temporal regression)
"""

import sys, os, time, json, warnings
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(line_buffering=True)

import numpy as np
from sklearn.linear_model import RidgeClassifier, Ridge
from sklearn.model_selection import StratifiedKFold, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fpga_host_eth import FPGAEthBridge

RESULTS_FILE = "results/z2246_temporal_overnight.json"
N_NEURONS = 128
BASE_VG = 0.58
ALPHA = 0.25
BETA = 0.08
SAMPLE_HZ = 20
STEP_INTERVAL = 1.0 / SAMPLE_HZ

# Tuned FPGA parameters (from z2239 — CRITICAL for input responsiveness)
TUNED_LEAK = 0x0011
TUNED_THRESH = 0.50
TUNED_BIAS_GAIN = 0.03125
TUNED_DT_C = 0.0078
TUNED_REFRACT = 50


def read_gpu_power():
    try:
        with open("/sys/class/drm/card1/device/hwmon/hwmon7/power1_average", "r") as f:
            return float(f.read().strip()) / 1e6
    except Exception:
        return 11.0 + np.random.randn() * 0.5


def configure_fpga(fpga):
    """Configure FPGA with tuned parameters — CRITICAL for input responsiveness."""
    fpga.set_kill(False)
    time.sleep(0.05)
    fpga.set_leak_cond(TUNED_LEAK)
    fpga.set_bias_gain(TUNED_BIAS_GAIN)
    fpga.set_threshold(TUNED_THRESH)
    fpga.set_dt_over_c(TUNED_DT_C)
    fpga.set_refract_cycles(TUNED_REFRACT)
    time.sleep(0.1)
    vg_base = np.array([float(BASE_VG + 0.15 * (i/127 - 0.5)) for i in range(128)])
    fpga.set_vg_batch(0, vg_base.tolist())
    time.sleep(0.1)
    return vg_base


def collect_trial(fpga, input_signal, condition, w_in, vg_base):
    """Collect reservoir states with rich features (z2239 style)."""
    n_steps = len(input_signal)
    all_spikes = []
    all_delta = []
    all_vmem = []
    gpu_power_base = read_gpu_power()
    noise_state = 0.0
    prev_spikes = None

    for step in range(n_steps):
        t0 = time.time()
        inp = float(input_signal[step])
        gpu_power = read_gpu_power()
        power_noise = (gpu_power - gpu_power_base) / 5.0
        noise_state = 0.85 * noise_state + 0.15 * power_noise

        if condition == "COUPLED":
            mac_val = inp + noise_state * 0.3
            fpga.set_mac_signal(float(np.clip(mac_val * 0.5 + 0.5, 0.0, 1.0)))
            vg_mod = vg_base + ALPHA * inp + BETA * w_in * inp + 0.05 * noise_state
            fpga.set_vg_batch(0, [float(np.clip(v, 0.3, 0.9)) for v in vg_mod])
        elif condition == "FPGA_ONLY":
            mac_val = inp * 0.5 + 0.5
            fpga.set_mac_signal(float(np.clip(mac_val, 0.0, 1.0)))
            vg_mod = vg_base + ALPHA * inp
            fpga.set_vg_batch(0, [float(np.clip(v, 0.3, 0.9)) for v in vg_mod])
        elif condition == "STATIC":
            pass  # No modulation — baseline

        telem = fpga.read_telemetry(timeout=0.15)
        if telem is not None:
            sc = telem['spike_counts'].astype(np.float32)
            vm = telem['vmem'].copy()
            delta = sc - prev_spikes if prev_spikes is not None else np.zeros(N_NEURONS, dtype=np.float32)
            all_spikes.append(sc)
            all_delta.append(delta)
            all_vmem.append(vm)
            prev_spikes = sc.copy()
        else:
            all_spikes.append(all_spikes[-1].copy() if all_spikes else np.zeros(N_NEURONS, dtype=np.float32))
            all_delta.append(np.zeros(N_NEURONS, dtype=np.float32))
            all_vmem.append(all_vmem[-1].copy() if all_vmem else np.zeros(N_NEURONS, dtype=np.float32))

        elapsed = time.time() - t0
        if elapsed < STEP_INTERVAL:
            time.sleep(STEP_INTERVAL - elapsed)

    return np.array(all_spikes), np.array(all_delta), np.array(all_vmem)


def build_features(spikes, delta, vmem):
    """Rich feature extraction (z2239 style): 7 × 128 = 896 features."""
    return np.concatenate([
        spikes.mean(axis=0),
        spikes.std(axis=0),
        delta.mean(axis=0),
        delta.std(axis=0),
        vmem.mean(axis=0),
        vmem[-1],
        spikes[-1] - spikes[0],
    ])


def build_static_features(spikes, vmem):
    """Static-only features (no temporal info): 2 × 128 = 256."""
    return np.concatenate([spikes.mean(axis=0), vmem.mean(axis=0)])


def ridge_classify(X, y, n_splits=5):
    scaler = StandardScaler()
    std = X.std(axis=0)
    mask = std > 1e-2
    if mask.sum() < 3:
        return 0.0
    X_f = scaler.fit_transform(X[:, mask])
    classes, counts = np.unique(y, return_counts=True)
    if counts.min() < n_splits:
        n_splits = max(2, counts.min())
    clf = RidgeClassifier(alpha=1.0)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    scores = cross_val_score(clf, X_f, y, cv=skf, scoring='accuracy')
    return float(scores.mean())


def ridge_regress(X, y, n_splits=5):
    scaler = StandardScaler()
    std = X.std(axis=0)
    mask = std > 1e-2
    if mask.sum() < 3:
        return 0.0
    X_f = scaler.fit_transform(X[:, mask])
    reg = Ridge(alpha=1.0)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    scores = cross_val_score(reg, X_f, y, cv=kf, scoring='r2')
    return float(max(0, np.mean(scores)))


def generate_signal(pattern, n_steps):
    t = np.linspace(0, 2 * np.pi, n_steps)
    if pattern == 'sine_slow':
        return np.sin(t)
    elif pattern == 'sine_fast':
        return np.sin(3 * t)
    elif pattern == 'square':
        return np.sign(np.sin(t))
    elif pattern == 'ramp':
        return 2 * (t / (2*np.pi)) - 1
    elif pattern == 'complex':
        return 0.5 * np.sin(2*t) + 0.5 * np.sin(3*t)
    elif pattern == 'noise':
        return np.random.randn(n_steps)
    elif pattern == 'binary':
        return np.random.choice([-1.0, 1.0], n_steps)
    elif pattern == 'pulse_early':
        s = np.zeros(n_steps)
        s[n_steps//4:n_steps//4+n_steps//8] = 1.0
        return s
    elif pattern == 'pulse_late':
        s = np.zeros(n_steps)
        s[3*n_steps//4:3*n_steps//4+n_steps//8] = 1.0
        return s
    elif pattern == 'ramp_up':
        return np.linspace(-1, 1, n_steps)
    elif pattern == 'ramp_down':
        return np.linspace(1, -1, n_steps)
    else:
        return np.zeros(n_steps)


# ==============================================================================
# EXP 1: Waveform Classification Baseline (confirm z2239)
# ==============================================================================
def exp1_baseline(fpga, w_in, results):
    print("\n=== EXP 1: Waveform Classification Baseline ===")

    patterns = ['sine_slow', 'square', 'ramp', 'complex']
    N_TRIALS = 120  # 30 per class
    STEPS = 20

    for cond in ["COUPLED", "FPGA_ONLY", "STATIC"]:
        print(f"  Condition: {cond}")
        vg_base = configure_fpga(fpga)
        time.sleep(0.3)
        features, labels = [], []

        for trial in range(N_TRIALS):
            cls = trial % 4
            signal = generate_signal(patterns[cls], STEPS)
            spk, dlt, vm = collect_trial(fpga, signal, cond, w_in, vg_base)
            features.append(build_features(spk, dlt, vm))
            labels.append(cls)
            if (trial + 1) % 40 == 0:
                print(f"    Trial {trial+1}/{N_TRIALS}")

        acc = ridge_classify(np.array(features), np.array(labels))
        results[f'exp1_{cond}_acc'] = acc
        print(f"    {cond}: {acc:.3f}")

    c = results.get('exp1_COUPLED_acc', 0)
    f = results.get('exp1_FPGA_ONLY_acc', 0)
    s = results.get('exp1_STATIC_acc', 0)
    results['T1096_coupled_gt_50'] = "PASS" if c > 0.50 else "FAIL"
    results['T1097_coupled_gt_80'] = "PASS" if c > 0.80 else "FAIL"
    results['T1098_coupled_gt_static'] = "PASS" if c > s else "FAIL"
    results['T1099_fpga_gt_40'] = "PASS" if f > 0.40 else "FAIL"
    results['T1100_any_gt_50'] = "PASS" if max(c, f) > 0.50 else "FAIL"
    _print_tests(results, 'T109', 'T110')


# ==============================================================================
# EXP 2: Temporal XOR at Multiple Delays
# ==============================================================================
def exp2_xor_delay(fpga, w_in, results):
    print("\n=== EXP 2: Temporal XOR at Multiple Delays ===")

    STEPS = 20
    N_TRIALS = 80  # per delay

    for delay in [1, 3, 5, 10]:
        print(f"\n  Delay={delay} ({delay * 50}ms)")

        for cond in ["COUPLED", "STATIC"]:
            vg_base = configure_fpga(fpga)
            time.sleep(0.3)
            features, labels = [], []

            for trial in range(N_TRIALS):
                signal = generate_signal('binary', STEPS)
                spk, dlt, vm = collect_trial(fpga, signal, cond, w_in, vg_base)

                idx = STEPS - 1
                if idx - delay < 0:
                    continue
                target = 1 if (signal[idx] > 0) != (signal[idx - delay] > 0) else 0
                features.append(build_features(spk, dlt, vm))
                labels.append(target)

                if (trial + 1) % 40 == 0:
                    print(f"    {cond} Trial {trial+1}/{N_TRIALS}")

            acc = ridge_classify(np.array(features), np.array(labels))
            results[f'exp2_xor_d{delay}_{cond}'] = acc
            print(f"    d={delay} {cond}: {acc:.3f}")

    d1 = results.get('exp2_xor_d1_COUPLED', 0)
    d5 = results.get('exp2_xor_d5_COUPLED', 0)
    d10 = results.get('exp2_xor_d10_COUPLED', 0)
    d10_s = results.get('exp2_xor_d10_STATIC', 0)
    d1_s = results.get('exp2_xor_d1_STATIC', 0)

    results['T1101_xor_d1_gt_52'] = "PASS" if d1 > 0.52 else "FAIL"
    results['T1102_xor_d5_gt_52'] = "PASS" if d5 > 0.52 else "FAIL"
    results['T1103_xor_d10_gt_chance'] = "PASS" if d10 > 0.52 else "FAIL"
    results['T1104_coupled_gt_static_d1'] = "PASS" if d1 > d1_s else "FAIL"
    results['T1105_coupled_gt_static_d10'] = "PASS" if d10 > d10_s else "FAIL"
    _print_tests(results, 'T110')


# ==============================================================================
# EXP 3: Frequency Discrimination
# ==============================================================================
def exp3_frequency(fpga, w_in, results):
    print("\n=== EXP 3: Frequency Discrimination ===")

    STEPS = 30  # longer to capture frequency differences
    N_TRIALS = 120  # 30 per freq

    freqs = [1.0, 2.0, 3.0, 5.0]

    for cond in ["COUPLED", "STATIC"]:
        vg_base = configure_fpga(fpga)
        time.sleep(0.3)
        features_full, features_static, labels = [], [], []

        for trial in range(N_TRIALS):
            cls = trial % 4
            freq = freqs[cls]
            t = np.linspace(0, 2 * np.pi * freq, STEPS)
            signal = np.sin(t)

            spk, dlt, vm = collect_trial(fpga, signal, cond, w_in, vg_base)
            features_full.append(build_features(spk, dlt, vm))
            features_static.append(build_static_features(spk, vm))
            labels.append(cls)

            if (trial + 1) % 40 == 0:
                print(f"    {cond} Trial {trial+1}/{N_TRIALS}")

        acc_full = ridge_classify(np.array(features_full), np.array(labels))
        acc_static = ridge_classify(np.array(features_static), np.array(labels))
        results[f'exp3_{cond}_full'] = acc_full
        results[f'exp3_{cond}_static'] = acc_static
        print(f"    {cond} full: {acc_full:.3f}, static: {acc_static:.3f}")

    cf = results.get('exp3_COUPLED_full', 0)
    cs = results.get('exp3_COUPLED_static', 0)
    results['T1106_freq_coupled_gt_50'] = "PASS" if cf > 0.50 else "FAIL"
    results['T1107_freq_full_gt_static'] = "PASS" if cf > cs else "FAIL"
    results['T1108_freq_coupled_gt_30'] = "PASS" if cf > 0.30 else "FAIL"
    _print_tests(results, 'T110')


# ==============================================================================
# EXP 4: Sequence Order Detection (temporal-only task)
# ==============================================================================
def exp4_sequence_order(fpga, w_in, results):
    print("\n=== EXP 4: Sequence Order Detection ===")

    STEPS = 20
    N_TRIALS = 80

    for cond in ["COUPLED", "STATIC"]:
        vg_base = configure_fpga(fpga)
        time.sleep(0.3)
        features, labels = [], []

        for trial in range(N_TRIALS):
            # Class 0: pulse_early then ramp, Class 1: ramp then pulse_late
            cls = trial % 2
            if cls == 0:
                signal = generate_signal('pulse_early', STEPS)
            else:
                signal = generate_signal('pulse_late', STEPS)

            spk, dlt, vm = collect_trial(fpga, signal, cond, w_in, vg_base)
            features.append(build_features(spk, dlt, vm))
            labels.append(cls)

            if (trial + 1) % 40 == 0:
                print(f"    {cond} Trial {trial+1}/{N_TRIALS}")

        acc = ridge_classify(np.array(features), np.array(labels))
        results[f'exp4_{cond}_order'] = acc
        print(f"    {cond}: {acc:.3f}")

    c = results.get('exp4_COUPLED_order', 0)
    s = results.get('exp4_STATIC_order', 0)
    results['T1109_order_coupled_gt_55'] = "PASS" if c > 0.55 else "FAIL"
    results['T1110_order_coupled_gt_static'] = "PASS" if c > s else "FAIL"
    results['T1111_order_coupled_gt_70'] = "PASS" if c > 0.70 else "FAIL"
    _print_tests(results, 'T110', 'T111')


# ==============================================================================
# EXP 5: Memory Capacity via Classification (z2241 style)
# ==============================================================================
def exp5_memory_capacity(fpga, w_in, results):
    print("\n=== EXP 5: Memory Capacity via Classification ===")

    STEPS = 20
    N_TRIALS = 100  # per delay

    mc_total = 0.0
    for delay in [0, 1, 2, 5]:
        print(f"\n  Delay={delay} ({delay * 50}ms)")
        vg_base = configure_fpga(fpga)
        time.sleep(0.3)
        features, labels = [], []

        for trial in range(N_TRIALS):
            signal = generate_signal('binary', STEPS)
            spk, dlt, vm = collect_trial(fpga, signal, "COUPLED", w_in, vg_base)

            # Target: classify what the input was `delay` steps before last
            idx = STEPS - 1 - delay
            if idx < 0:
                continue
            target = 1 if signal[idx] > 0 else 0
            features.append(build_features(spk, dlt, vm))
            labels.append(target)

            if (trial + 1) % 50 == 0:
                print(f"    Trial {trial+1}/{N_TRIALS}")

        acc = ridge_classify(np.array(features), np.array(labels))
        results[f'exp5_mc_d{delay}'] = acc
        mc_total += max(0, acc - 0.5) * 2  # normalize: 50%=0, 100%=1
        print(f"    d={delay}: {acc:.3f}")

    results['exp5_mc_total'] = mc_total
    print(f"  MC total: {mc_total:.3f}")

    d0 = results.get('exp5_mc_d0', 0)
    d1 = results.get('exp5_mc_d1', 0)
    d2 = results.get('exp5_mc_d2', 0)

    results['T1112_mc_d0_gt_55'] = "PASS" if d0 > 0.55 else "FAIL"
    results['T1113_mc_d1_gt_52'] = "PASS" if d1 > 0.52 else "FAIL"
    results['T1114_mc_total_gt_050'] = "PASS" if mc_total > 0.50 else "FAIL"
    results['T1115_mc_decays'] = "PASS" if d0 > d2 else "FAIL"
    _print_tests(results, 'T111')


# ==============================================================================
# EXP 6: Sustained vs Transient Response
# ==============================================================================
def exp6_sustained_transient(fpga, w_in, results):
    print("\n=== EXP 6: Sustained vs Transient Response ===")

    STEPS = 30
    N_TRIALS = 80

    for cond in ["COUPLED", "STATIC"]:
        vg_base = configure_fpga(fpga)
        time.sleep(0.3)
        features, labels = [], []

        for trial in range(N_TRIALS):
            cls = trial % 2
            if cls == 0:  # sustained: ramp_up
                signal = generate_signal('ramp_up', STEPS)
            else:  # transient: pulse then flat
                signal = np.zeros(STEPS)
                signal[2:5] = 1.0

            spk, dlt, vm = collect_trial(fpga, signal, cond, w_in, vg_base)
            features.append(build_features(spk, dlt, vm))
            labels.append(cls)

            if (trial + 1) % 40 == 0:
                print(f"    {cond} Trial {trial+1}/{N_TRIALS}")

        acc = ridge_classify(np.array(features), np.array(labels))
        results[f'exp6_{cond}_sustrans'] = acc
        print(f"    {cond}: {acc:.3f}")

    c = results.get('exp6_COUPLED_sustrans', 0)
    s = results.get('exp6_STATIC_sustrans', 0)
    results['T1116_sustrans_coupled_gt_60'] = "PASS" if c > 0.60 else "FAIL"
    results['T1117_sustrans_coupled_gt_static'] = "PASS" if c > s else "FAIL"
    results['T1118_sustrans_coupled_gt_80'] = "PASS" if c > 0.80 else "FAIL"
    _print_tests(results, 'T111')


# ==============================================================================
# EXP 7: Multi-Timescale Integration
# ==============================================================================
def exp7_multitimescale(fpga, w_in, results):
    print("\n=== EXP 7: Multi-Timescale Integration ===")

    STEPS = 40  # longer for multi-timescale
    N_TRIALS = 120  # 30 per class

    for cond in ["COUPLED", "FPGA_ONLY"]:
        vg_base = configure_fpga(fpga)
        time.sleep(0.3)
        features, labels = [], []

        for trial in range(N_TRIALS):
            cls = trial % 4
            t = np.linspace(0, 4 * np.pi, STEPS)
            if cls == 0:
                signal = np.sin(t)  # slow only
            elif cls == 1:
                signal = np.sin(5 * t)  # fast only
            elif cls == 2:
                signal = np.sin(t) + 0.5 * np.sin(5 * t)  # both
            else:
                signal = np.sin(t) * np.sin(5 * t)  # modulated

            spk, dlt, vm = collect_trial(fpga, signal, cond, w_in, vg_base)
            features.append(build_features(spk, dlt, vm))
            labels.append(cls)

            if (trial + 1) % 40 == 0:
                print(f"    {cond} Trial {trial+1}/{N_TRIALS}")

        acc = ridge_classify(np.array(features), np.array(labels))
        results[f'exp7_{cond}_multiscale'] = acc
        print(f"    {cond}: {acc:.3f}")

    c = results.get('exp7_COUPLED_multiscale', 0)
    f = results.get('exp7_FPGA_ONLY_multiscale', 0)
    results['T1119_multiscale_coupled_gt_40'] = "PASS" if c > 0.40 else "FAIL"
    results['T1120_multiscale_coupled_gt_60'] = "PASS" if c > 0.60 else "FAIL"
    results['T1121_multiscale_coupled_gt_fpga'] = "PASS" if c > f else "FAIL"
    _print_tests(results, 'T111', 'T112')


# ==============================================================================
# EXP 8: Input Reconstruction (Temporal Regression)
# ==============================================================================
def exp8_reconstruction(fpga, w_in, results):
    print("\n=== EXP 8: Input Reconstruction (Regression) ===")

    STEPS = 60  # longer continuous run for regression
    N_REPS = 5

    for cond in ["COUPLED", "STATIC"]:
        vg_base = configure_fpga(fpga)
        time.sleep(0.3)

        all_r2 = []
        for rep in range(N_REPS):
            signal = np.random.randn(STEPS) * 0.5
            signal_smooth = np.convolve(signal, np.ones(3)/3, mode='same')

            spk, dlt, vm = collect_trial(fpga, signal_smooth, cond, w_in, vg_base)

            # Try to reconstruct input from reservoir state
            X = np.hstack([spk[5:], vm[5:]])
            y = signal_smooth[5:]

            r2 = ridge_regress(X, y)
            all_r2.append(r2)
            print(f"    {cond} rep {rep}: R2={r2:.4f}")

        mean_r2 = float(np.mean(all_r2))
        results[f'exp8_{cond}_recon_r2'] = mean_r2
        print(f"    {cond} mean R2: {mean_r2:.4f}")

    # Also try periodic signal (known to work from z2240)
    vg_base = configure_fpga(fpga)
    time.sleep(0.3)
    t = np.linspace(0, 6*np.pi, STEPS)
    signal_periodic = np.sin(t)
    spk, dlt, vm = collect_trial(fpga, signal_periodic, "COUPLED", w_in, vg_base)
    X = np.hstack([spk[5:], vm[5:]])
    y = signal_periodic[5:]
    r2_periodic = ridge_regress(X, y)
    results['exp8_periodic_r2'] = r2_periodic
    print(f"    Periodic R2: {r2_periodic:.4f}")

    cr = results.get('exp8_COUPLED_recon_r2', 0)
    sr = results.get('exp8_STATIC_recon_r2', 0)
    pr = results.get('exp8_periodic_r2', 0)

    results['T1122_recon_coupled_gt_0'] = "PASS" if cr > 0.0 else "FAIL"
    results['T1123_recon_coupled_gt_static'] = "PASS" if cr > sr else "FAIL"
    results['T1124_periodic_gt_010'] = "PASS" if pr > 0.10 else "FAIL"
    results['T1125_periodic_gt_050'] = "PASS" if pr > 0.50 else "FAIL"
    _print_tests(results, 'T112')


def _print_tests(results, *prefixes):
    for k in sorted(results):
        if any(k.startswith(p) for p in prefixes) and k.startswith('T'):
            print(f"  {k}: {results[k]}")


def main():
    t_start = time.time()
    print("=" * 60)
    print("z2246: Temporal Overnight Experiment Suite")
    print("  Building on z2239 proven approach (94.4% classification)")
    print("  20Hz, tuned FPGA params, delta features, RidgeClassifier")
    print("=" * 60)

    fpga = FPGAEthBridge()
    if not fpga.connect():
        print("  WARNING: connect() returned False")
    fpga.set_kill(False)
    time.sleep(0.1)

    telem = fpga.read_telemetry()
    if telem is None:
        print("ERROR: No telemetry from FPGA")
        sys.exit(1)
    print(f"  FPGA: {len(telem['spike_counts'])} neurons")
    print(f"  Spikes: [{telem['spike_counts'].min():.0f}, {telem['spike_counts'].max():.0f}]")
    print(f"  Vmem: [{telem['vmem'].min():.3f}, {telem['vmem'].max():.3f}]")

    vg_base = configure_fpga(fpga)
    print(f"  Vg: [{vg_base.min():.3f}, {vg_base.max():.3f}]")

    rng = np.random.RandomState(42)
    w_in = rng.uniform(-1, 1, N_NEURONS)

    results = {
        'experiment': 'z2246_temporal_overnight',
        'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
        'sample_hz': SAMPLE_HZ,
        'base_vg': BASE_VG,
        'alpha': ALPHA,
        'beta': BETA,
        'bias_gain': TUNED_BIAS_GAIN,
        'threshold': TUNED_THRESH,
    }

    exp1_baseline(fpga, w_in, results)
    exp2_xor_delay(fpga, w_in, results)
    exp3_frequency(fpga, w_in, results)
    exp4_sequence_order(fpga, w_in, results)
    exp5_memory_capacity(fpga, w_in, results)
    exp6_sustained_transient(fpga, w_in, results)
    exp7_multitimescale(fpga, w_in, results)
    exp8_reconstruction(fpga, w_in, results)

    elapsed = time.time() - t_start
    tests = sorted([k for k in results if k.startswith('T')])
    n_pass = sum(1 for t in tests if results[t] == "PASS")
    n_total = len(tests)
    results['summary'] = f"{n_pass}/{n_total} PASS"
    results['elapsed_s'] = elapsed

    print(f"\n{'=' * 60}")
    print(f"RESULTS: {n_pass}/{n_total} PASS ({elapsed:.0f}s)")
    for t in tests:
        print(f"  {t}: {results[t]}")
    print(f"{'=' * 60}")

    with open(RESULTS_FILE, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved: {RESULTS_FILE}")


if __name__ == '__main__':
    main()
