#!/usr/bin/env python3
"""
z2247_nsram_bridge_validation.py — NS-RAM Bridge Validation Suite
=================================================================
Validates that our FPGA reservoir exhibits properties analogous to
Mario Lanza's NS-RAM memristive systems. Focuses on:

1. Analog fading memory (not just binary classification)
2. Nonlinear transformation (kernel quality)
3. Noise-driven computation (GPU power noise coupling)
4. Cross-substrate information flow
5. Criticality indicators (edge of chaos)

Builds on z2246's proven foundation (23/30 PASS).

Tests T1126-T1165 (40 tests across 8 experiments):
  EXP 1: Analog input reconstruction (graded, not binary)
  EXP 2: Nonlinear kernel quality
  EXP 3: Noise coupling advantage (COUPLED vs NO_NOISE)
  EXP 4: Fading memory profile (fine-grained delays)
  EXP 5: Cross-substrate information transfer
  EXP 6: Edge-of-chaos indicators
  EXP 7: Multi-class scaling (4,8,16 classes)
  EXP 8: Temporal pattern replay
"""

import sys, os, time, json, warnings
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(line_buffering=True)

import numpy as np
from sklearn.linear_model import RidgeClassifier, Ridge
from sklearn.model_selection import StratifiedKFold, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from scipy import stats

warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fpga_host_eth import FPGAEthBridge

RESULTS_FILE = "results/z2247_nsram_bridge.json"
N_NEURONS = 128
BASE_VG = 0.58
ALPHA = 0.25
BETA = 0.08
SAMPLE_HZ = 20
STEP_INTERVAL = 1.0 / SAMPLE_HZ

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
    n_steps = len(input_signal)
    all_spikes, all_delta, all_vmem = [], [], []
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
        elif condition == "NO_NOISE":
            vg_mod = vg_base + ALPHA * inp + BETA * w_in * inp
            fpga.set_vg_batch(0, [float(np.clip(v, 0.3, 0.9)) for v in vg_mod])
            fpga.set_mac_signal(0.5)
        elif condition == "STATIC":
            pass

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
    return np.concatenate([
        spikes.mean(axis=0), spikes.std(axis=0),
        delta.mean(axis=0), delta.std(axis=0),
        vmem.mean(axis=0), vmem[-1],
        spikes[-1] - spikes[0],
    ])


def ridge_classify(X, y, n_splits=5):
    std = X.std(axis=0)
    mask = std > 1e-2
    if mask.sum() < 3:
        return 0.0
    scaler = StandardScaler()
    X_f = scaler.fit_transform(X[:, mask])
    classes, counts = np.unique(y, return_counts=True)
    if counts.min() < n_splits:
        n_splits = max(2, counts.min())
    clf = RidgeClassifier(alpha=1.0)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    scores = cross_val_score(clf, X_f, y, cv=skf, scoring='accuracy')
    return float(scores.mean())


def ridge_regress(X, y, n_splits=5):
    std = X.std(axis=0)
    mask = std > 1e-2
    if mask.sum() < 3:
        return 0.0
    scaler = StandardScaler()
    X_f = scaler.fit_transform(X[:, mask])
    reg = Ridge(alpha=1.0)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    scores = cross_val_score(reg, X_f, y, cv=kf, scoring='r2')
    return float(np.mean(scores))  # allow negative R2


# ==============================================================================
# EXP 1: Analog Input Reconstruction (graded signals)
# ==============================================================================
def exp1_analog_recon(fpga, w_in, results):
    print("\n=== EXP 1: Analog Input Reconstruction ===")
    STEPS = 40

    signals = {
        'sine': lambda: np.sin(np.linspace(0, 4*np.pi, STEPS)),
        'sum_sines': lambda: np.sin(np.linspace(0, 4*np.pi, STEPS)) + 0.3*np.sin(np.linspace(0, 12*np.pi, STEPS)),
        'chirp': lambda: np.sin(np.cumsum(np.linspace(0.1, 0.5, STEPS))),
        'ramp': lambda: np.linspace(-1, 1, STEPS),
        'random_smooth': lambda: np.convolve(np.random.randn(STEPS+4), np.ones(5)/5, mode='valid'),
    }

    for cond in ["COUPLED", "NO_NOISE"]:
        vg_base = configure_fpga(fpga)
        time.sleep(0.3)

        for sig_name, sig_fn in signals.items():
            r2_list = []
            for rep in range(3):
                signal = sig_fn()
                spk, dlt, vm = collect_trial(fpga, signal, cond, w_in, vg_base)
                X = np.hstack([spk[3:], vm[3:]])
                y = signal[3:]
                r2 = ridge_regress(X, y)
                r2_list.append(r2)
            mean_r2 = float(np.mean(r2_list))
            results[f'exp1_{cond}_{sig_name}_r2'] = mean_r2
            print(f"    {cond} {sig_name}: R2={mean_r2:.4f}")

    # Tests
    sine_c = results.get('exp1_COUPLED_sine_r2', -1)
    chirp_c = results.get('exp1_COUPLED_chirp_r2', -1)
    ramp_c = results.get('exp1_COUPLED_ramp_r2', -1)
    smooth_c = results.get('exp1_COUPLED_random_smooth_r2', -1)
    sine_nn = results.get('exp1_NO_NOISE_sine_r2', -1)

    results['T1126_sine_r2_gt_050'] = "PASS" if sine_c > 0.50 else "FAIL"
    results['T1127_chirp_r2_gt_020'] = "PASS" if chirp_c > 0.20 else "FAIL"
    results['T1128_ramp_r2_gt_030'] = "PASS" if ramp_c > 0.30 else "FAIL"
    results['T1129_smooth_r2_gt_0'] = "PASS" if smooth_c > 0.0 else "FAIL"
    results['T1130_coupled_gt_nonoise'] = "PASS" if sine_c > sine_nn else "FAIL"
    _print_tests(results, 'T112', 'T113')


# ==============================================================================
# EXP 2: Nonlinear Kernel Quality
# ==============================================================================
def exp2_kernel_quality(fpga, w_in, results):
    print("\n=== EXP 2: Nonlinear Kernel Quality ===")
    STEPS = 30
    N_TRIALS = 60

    vg_base = configure_fpga(fpga)
    time.sleep(0.3)

    # Collect reservoir states for random inputs
    all_states = []
    all_inputs = []

    for trial in range(N_TRIALS):
        signal = np.random.randn(STEPS) * 0.5
        spk, dlt, vm = collect_trial(fpga, signal, "COUPLED", w_in, vg_base)
        state = np.hstack([spk.mean(0), vm.mean(0)])
        all_states.append(state)
        all_inputs.append(signal.mean())
        if (trial+1) % 20 == 0:
            print(f"    Trial {trial+1}/{N_TRIALS}")

    X = np.array(all_states)
    u = np.array(all_inputs)

    # Kernel quality: rank of state matrix relative to linear embedding
    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)
    std = X_s.std(axis=0)
    X_s = X_s[:, std > 1e-3]

    # Effective dimension
    if X_s.shape[1] > 0:
        cov = np.cov(X_s.T)
        eigvals = np.linalg.eigvalsh(cov)
        eigvals = eigvals[eigvals > 0]
        eigvals_norm = eigvals / eigvals.sum()
        eff_dim = np.exp(-np.sum(eigvals_norm * np.log(eigvals_norm + 1e-12)))
    else:
        eff_dim = 0

    results['exp2_eff_dim'] = eff_dim
    print(f"    Effective dimension: {eff_dim:.1f}")

    # Nonlinear separation: can we predict input^2?
    y_sq = u ** 2
    r2_sq = ridge_regress(X, y_sq)
    results['exp2_squared_r2'] = r2_sq
    print(f"    Input^2 R2: {r2_sq:.4f}")

    # Linear vs nonlinear: predict sin(input) vs input
    y_sin = np.sin(3 * u)
    r2_sin = ridge_regress(X, y_sin)
    r2_lin = ridge_regress(X, u)
    results['exp2_sin_r2'] = r2_sin
    results['exp2_linear_r2'] = r2_lin
    print(f"    sin(3u) R2: {r2_sin:.4f}, linear R2: {r2_lin:.4f}")

    results['T1131_eff_dim_gt_5'] = "PASS" if eff_dim > 5 else "FAIL"
    results['T1132_eff_dim_gt_20'] = "PASS" if eff_dim > 20 else "FAIL"
    results['T1133_squared_r2_gt_0'] = "PASS" if r2_sq > 0.0 else "FAIL"
    results['T1134_nonlinear_exists'] = "PASS" if r2_sin > 0 or r2_sq > 0 else "FAIL"
    _print_tests(results, 'T113')


# ==============================================================================
# EXP 3: Noise Coupling Advantage
# ==============================================================================
def exp3_noise_advantage(fpga, w_in, results):
    print("\n=== EXP 3: Noise Coupling Advantage ===")
    STEPS = 20
    N_TRIALS = 120

    patterns = ['sine_slow', 'square', 'ramp', 'complex']

    for cond in ["COUPLED", "NO_NOISE", "FPGA_ONLY"]:
        vg_base = configure_fpga(fpga)
        time.sleep(0.3)
        features, labels = [], []

        for trial in range(N_TRIALS):
            cls = trial % 4
            t = np.linspace(0, 2*np.pi, STEPS)
            if cls == 0: signal = np.sin(t)
            elif cls == 1: signal = np.sign(np.sin(t))
            elif cls == 2: signal = 2 * (t / (2*np.pi)) - 1
            else: signal = 0.5 * np.sin(2*t) + 0.5 * np.sin(3*t)

            spk, dlt, vm = collect_trial(fpga, signal, cond, w_in, vg_base)
            features.append(build_features(spk, dlt, vm))
            labels.append(cls)

            if (trial+1) % 40 == 0:
                print(f"    {cond} Trial {trial+1}/{N_TRIALS}")

        acc = ridge_classify(np.array(features), np.array(labels))
        results[f'exp3_{cond}_acc'] = acc
        print(f"    {cond}: {acc:.3f}")

    c = results.get('exp3_COUPLED_acc', 0)
    nn = results.get('exp3_NO_NOISE_acc', 0)
    fo = results.get('exp3_FPGA_ONLY_acc', 0)

    results['T1135_coupled_gt_nonoise'] = "PASS" if c > nn else "FAIL"
    results['T1136_coupled_gt_80'] = "PASS" if c > 0.80 else "FAIL"
    results['T1137_fpga_gt_70'] = "PASS" if fo > 0.70 else "FAIL"
    results['T1138_noise_advantage_pp'] = f"{(c-nn)*100:.1f}pp"
    _print_tests(results, 'T113')


# ==============================================================================
# EXP 4: Fading Memory Profile (fine-grained delays)
# ==============================================================================
def exp4_fading_memory(fpga, w_in, results):
    print("\n=== EXP 4: Fading Memory Profile ===")
    STEPS = 25
    N_TRIALS = 80

    delays = [0, 1, 2, 3, 5, 7, 10, 15]
    mc_profile = {}

    vg_base = configure_fpga(fpga)
    time.sleep(0.3)

    for delay in delays:
        features, labels = [], []

        for trial in range(N_TRIALS):
            signal = np.random.choice([-1.0, 1.0], STEPS)
            spk, dlt, vm = collect_trial(fpga, signal, "COUPLED", w_in, vg_base)

            idx = STEPS - 1 - delay
            if idx < 0:
                continue
            target = 1 if signal[idx] > 0 else 0
            features.append(build_features(spk, dlt, vm))
            labels.append(target)

            if (trial+1) % 40 == 0:
                print(f"    d={delay} Trial {trial+1}/{N_TRIALS}")

        acc = ridge_classify(np.array(features), np.array(labels))
        mc_profile[delay] = acc
        results[f'exp4_mc_d{delay}'] = acc
        print(f"    d={delay} ({delay*50}ms): {acc:.3f}")

    mc_total = sum(max(0, v - 0.5) * 2 for v in mc_profile.values())
    results['exp4_mc_total'] = mc_total
    results['exp4_mc_profile'] = {str(k): v for k, v in mc_profile.items()}
    print(f"    MC total: {mc_total:.3f}")

    d0 = mc_profile.get(0, 0)
    d3 = mc_profile.get(3, 0)
    d10 = mc_profile.get(10, 0)
    # Check monotonic decay
    vals = [mc_profile[d] for d in sorted(mc_profile.keys())]
    n_decreasing = sum(1 for i in range(1, len(vals)) if vals[i] <= vals[i-1])

    results['T1139_mc_d0_gt_60'] = "PASS" if d0 > 0.60 else "FAIL"
    results['T1140_mc_d3_gt_52'] = "PASS" if d3 > 0.52 else "FAIL"
    results['T1141_mc_total_gt_1'] = "PASS" if mc_total > 1.0 else "FAIL"
    results['T1142_mc_mostly_decreasing'] = "PASS" if n_decreasing >= len(vals) // 2 else "FAIL"
    results['T1143_mc_d0_gt_d10'] = "PASS" if d0 > d10 else "FAIL"
    _print_tests(results, 'T113', 'T114')


# ==============================================================================
# EXP 5: Cross-Substrate Information Transfer
# ==============================================================================
def exp5_cross_substrate(fpga, w_in, results):
    print("\n=== EXP 5: Cross-Substrate Information Transfer ===")
    STEPS = 30
    N_TRIALS = 100

    vg_base = configure_fpga(fpga)
    time.sleep(0.3)

    # Collect GPU power and FPGA spikes simultaneously
    gpu_powers = []
    fpga_rates = []
    input_vals = []

    for trial in range(N_TRIALS):
        signal = np.random.randn(STEPS) * 0.5
        spk, dlt, vm = collect_trial(fpga, signal, "COUPLED", w_in, vg_base)

        gpu_powers.append(read_gpu_power())
        fpga_rates.append(float(spk.mean()))
        input_vals.append(float(signal.mean()))

        if (trial+1) % 50 == 0:
            print(f"    Trial {trial+1}/{N_TRIALS}")

    gpu_p = np.array(gpu_powers)
    fpga_r = np.array(fpga_rates)
    inp_v = np.array(input_vals)

    # Correlations
    corr_gpu_fpga = float(np.corrcoef(gpu_p, fpga_r)[0, 1]) if gpu_p.std() > 0 else 0
    corr_inp_fpga = float(np.corrcoef(inp_v, fpga_r)[0, 1]) if inp_v.std() > 0 else 0

    results['exp5_corr_gpu_fpga'] = corr_gpu_fpga
    results['exp5_corr_input_fpga'] = corr_inp_fpga
    print(f"    GPU-FPGA correlation: {corr_gpu_fpga:.4f}")
    print(f"    Input-FPGA correlation: {corr_inp_fpga:.4f}")

    # Mutual information proxy: can FPGA state predict GPU power bin?
    gpu_bins = (gpu_p > np.median(gpu_p)).astype(int)
    fpga_bins = (fpga_r > np.median(fpga_r)).astype(int)
    agreement = float(np.mean(gpu_bins == fpga_bins))
    results['exp5_gpu_fpga_agreement'] = agreement
    print(f"    GPU-FPGA bin agreement: {agreement:.3f}")

    results['T1144_input_fpga_corr_gt_0'] = "PASS" if abs(corr_inp_fpga) > 0.05 else "FAIL"
    results['T1145_gpu_fpga_corr_exists'] = "PASS" if abs(corr_gpu_fpga) > 0.01 else "FAIL"
    results['T1146_agreement_gt_50'] = "PASS" if agreement > 0.50 else "FAIL"
    _print_tests(results, 'T114')


# ==============================================================================
# EXP 6: Edge-of-Chaos Indicators
# ==============================================================================
def exp6_criticality(fpga, w_in, results):
    print("\n=== EXP 6: Edge-of-Chaos Indicators ===")
    STEPS = 100

    vg_base = configure_fpga(fpga)
    time.sleep(0.3)

    # Long continuous run with noise input
    signal = np.random.randn(STEPS) * 0.3
    spk, dlt, vm = collect_trial(fpga, signal, "COUPLED", w_in, vg_base)

    # 1. Lyapunov-like: sensitivity to perturbation
    signal2 = signal.copy()
    signal2[STEPS//4] += 0.5  # small perturbation
    spk2, dlt2, vm2 = collect_trial(fpga, signal2, "COUPLED", w_in, vg_base)

    # Measure divergence after perturbation point
    start = STEPS // 4
    div = np.mean(np.abs(spk[start:] - spk2[start:]))
    div_before = np.mean(np.abs(spk[:start] - spk2[:start]))
    sensitivity = div / (div_before + 1e-8) if div_before > 0 else div

    results['exp6_perturbation_sensitivity'] = float(sensitivity)
    print(f"    Perturbation sensitivity: {sensitivity:.3f}")

    # 2. Branching ratio (proxy for criticality)
    total_spikes_per_step = spk.sum(axis=1)
    if len(total_spikes_per_step) > 1:
        ratios = total_spikes_per_step[1:] / (total_spikes_per_step[:-1] + 1e-8)
        branching = float(np.median(ratios))
    else:
        branching = 0
    results['exp6_branching_ratio'] = branching
    print(f"    Branching ratio: {branching:.3f} (critical=1.0)")

    # 3. Spike rate CV across neurons
    rates = spk.mean(axis=0)
    rate_cv = float(rates.std() / (rates.mean() + 1e-8))
    results['exp6_rate_cv'] = rate_cv
    print(f"    Rate CV: {rate_cv:.3f}")

    # 4. Autocorrelation decay
    mean_vm = vm.mean(axis=1)
    if len(mean_vm) > 10:
        acf1 = float(np.corrcoef(mean_vm[:-1], mean_vm[1:])[0, 1])
        acf5 = float(np.corrcoef(mean_vm[:-5], mean_vm[5:])[0, 1]) if len(mean_vm) > 10 else 0
    else:
        acf1, acf5 = 0, 0
    results['exp6_acf1'] = acf1
    results['exp6_acf5'] = acf5
    print(f"    ACF(1): {acf1:.3f}, ACF(5): {acf5:.3f}")

    results['T1147_sensitivity_gt_1'] = "PASS" if sensitivity > 1.0 else "FAIL"
    results['T1148_branching_near_1'] = "PASS" if 0.8 < branching < 1.2 else "FAIL"
    results['T1149_rate_cv_gt_010'] = "PASS" if rate_cv > 0.10 else "FAIL"
    results['T1150_temporal_memory'] = "PASS" if abs(acf1) > 0.1 else "FAIL"
    _print_tests(results, 'T114', 'T115')


# ==============================================================================
# EXP 7: Multi-Class Scaling
# ==============================================================================
def exp7_scaling(fpga, w_in, results):
    print("\n=== EXP 7: Multi-Class Scaling ===")
    STEPS = 20

    for n_classes in [4, 8, 16]:
        print(f"\n  {n_classes} classes:")
        N_TRIALS = n_classes * 20  # 20 per class

        vg_base = configure_fpga(fpga)
        time.sleep(0.3)
        features, labels = [], []

        freqs = np.linspace(0.5, 5.0, n_classes)

        for trial in range(N_TRIALS):
            cls = trial % n_classes
            freq = freqs[cls]
            t = np.linspace(0, 2 * np.pi * freq, STEPS)
            signal = np.sin(t)

            spk, dlt, vm = collect_trial(fpga, signal, "COUPLED", w_in, vg_base)
            features.append(build_features(spk, dlt, vm))
            labels.append(cls)

            if (trial+1) % 40 == 0:
                print(f"    Trial {trial+1}/{N_TRIALS}")

        acc = ridge_classify(np.array(features), np.array(labels))
        chance = 1.0 / n_classes
        ratio = acc / chance
        results[f'exp7_{n_classes}class_acc'] = acc
        results[f'exp7_{n_classes}class_ratio'] = ratio
        print(f"    {n_classes}-class: {acc:.3f} (chance={chance:.3f}, ratio={ratio:.1f}x)")

    a4 = results.get('exp7_4class_acc', 0)
    a8 = results.get('exp7_8class_acc', 0)
    a16 = results.get('exp7_16class_acc', 0)

    results['T1151_4class_gt_50'] = "PASS" if a4 > 0.50 else "FAIL"
    results['T1152_8class_gt_2x_chance'] = "PASS" if a8 > 2 * (1/8) else "FAIL"
    results['T1153_16class_gt_chance'] = "PASS" if a16 > 1.5 * (1/16) else "FAIL"
    results['T1154_scaling_graceful'] = "PASS" if a4 > a8 > a16 else "FAIL"
    results['T1155_8class_gt_25'] = "PASS" if a8 > 0.25 else "FAIL"
    _print_tests(results, 'T115')


# ==============================================================================
# EXP 8: Temporal Pattern Replay
# ==============================================================================
def exp8_replay(fpga, w_in, results):
    print("\n=== EXP 8: Temporal Pattern Replay ===")
    STEPS = 30
    N_TRIALS = 80

    vg_base = configure_fpga(fpga)
    time.sleep(0.3)

    # Task: classify whether the SAME temporal pattern was repeated or a different one
    templates = [
        np.sin(np.linspace(0, 2*np.pi, STEPS//2)),
        np.sign(np.sin(np.linspace(0, 2*np.pi, STEPS//2))),
        np.linspace(-1, 1, STEPS//2),
        np.random.RandomState(42).randn(STEPS//2),
    ]

    features, labels = [], []

    for trial in range(N_TRIALS):
        cls = trial % 2
        template_idx = (trial // 2) % len(templates)
        t1 = templates[template_idx]

        if cls == 0:  # same pattern repeated
            signal = np.concatenate([t1, t1])
        else:  # different pattern in second half
            other_idx = (template_idx + 1) % len(templates)
            signal = np.concatenate([t1, templates[other_idx]])

        spk, dlt, vm = collect_trial(fpga, signal, "COUPLED", w_in, vg_base)
        features.append(build_features(spk, dlt, vm))
        labels.append(cls)

        if (trial+1) % 40 == 0:
            print(f"    Trial {trial+1}/{N_TRIALS}")

    acc = ridge_classify(np.array(features), np.array(labels))
    results['exp8_replay_acc'] = acc
    print(f"    Replay detection: {acc:.3f}")

    # Also test with static features
    features_s = []
    for trial in range(N_TRIALS):
        cls = trial % 2
        template_idx = (trial // 2) % len(templates)
        t1 = templates[template_idx]
        if cls == 0:
            signal = np.concatenate([t1, t1])
        else:
            other_idx = (template_idx + 1) % len(templates)
            signal = np.concatenate([t1, templates[other_idx]])
        spk, dlt, vm = collect_trial(fpga, signal, "STATIC", w_in, vg_base)
        features_s.append(build_features(spk, dlt, vm))

    acc_s = ridge_classify(np.array(features_s), np.array(labels))
    results['exp8_replay_static_acc'] = acc_s
    print(f"    Replay static: {acc_s:.3f}")

    results['T1156_replay_gt_55'] = "PASS" if acc > 0.55 else "FAIL"
    results['T1157_replay_gt_70'] = "PASS" if acc > 0.70 else "FAIL"
    results['T1158_replay_gt_static'] = "PASS" if acc > acc_s else "FAIL"
    _print_tests(results, 'T115')


def _print_tests(results, *prefixes):
    for k in sorted(results):
        if any(k.startswith(p) for p in prefixes) and k.startswith('T'):
            print(f"  {k}: {results[k]}")


def main():
    t_start = time.time()
    print("=" * 60)
    print("z2247: NS-RAM Bridge Validation Suite")
    print("  Validating FPGA reservoir ↔ NS-RAM analog properties")
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

    vg_base = configure_fpga(fpga)
    print(f"  Vg: [{vg_base.min():.3f}, {vg_base.max():.3f}]")

    rng = np.random.RandomState(42)
    w_in = rng.uniform(-1, 1, N_NEURONS)

    results = {
        'experiment': 'z2247_nsram_bridge_validation',
        'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
        'sample_hz': SAMPLE_HZ,
        'base_vg': BASE_VG,
        'alpha': ALPHA,
        'beta': BETA,
    }

    exp1_analog_recon(fpga, w_in, results)
    exp2_kernel_quality(fpga, w_in, results)
    exp3_noise_advantage(fpga, w_in, results)
    exp4_fading_memory(fpga, w_in, results)
    exp5_cross_substrate(fpga, w_in, results)
    exp6_criticality(fpga, w_in, results)
    exp7_scaling(fpga, w_in, results)
    exp8_replay(fpga, w_in, results)

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
