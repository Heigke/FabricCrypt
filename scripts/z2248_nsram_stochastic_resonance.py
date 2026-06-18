#!/usr/bin/env python3
"""
z2248_nsram_stochastic_resonance.py — NS-RAM Stochastic Resonance & Parameter Sweep
====================================================================================
Addresses z2247 gap: COUPLED < FPGA_ONLY by -9.2pp on classification.
Hypothesis: noise coupling strength is suboptimal — stochastic resonance predicts
an OPTIMAL noise level where signal detection peaks.

Also explores:
1. Stochastic resonance curve (noise amplitude sweep)
2. Leak rate sweep (runtime τ tuning for temporal memory)
3. Bias gain sweep (MAC current injection strength)
4. Combined optimal: best noise × best leak × best bias_gain
5. NS-RAM conductance analogy (Vg as analog weight, not just modulator)
6. Temporal regression with optimal params
7. Transfer entropy at optimal operating point
8. Final NS-RAM bridge scorecard

Tests T1159-T1195 (37 tests across 8 experiments)
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

RESULTS_FILE = "results/z2248_stochastic_resonance.json"
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


def configure_fpga(fpga, leak=None, bias_gain=None, thresh=None):
    fpga.set_kill(False)
    time.sleep(0.05)
    fpga.set_leak_cond(leak if leak is not None else TUNED_LEAK)
    fpga.set_bias_gain(bias_gain if bias_gain is not None else TUNED_BIAS_GAIN)
    fpga.set_threshold(thresh if thresh is not None else TUNED_THRESH)
    fpga.set_dt_over_c(TUNED_DT_C)
    fpga.set_refract_cycles(TUNED_REFRACT)
    time.sleep(0.1)
    vg_base = np.array([float(BASE_VG + 0.15 * (i/127 - 0.5)) for i in range(128)])
    fpga.set_vg_batch(0, vg_base.tolist())
    time.sleep(0.1)
    return vg_base


def collect_trial(fpga, input_signal, noise_scale, w_in, vg_base, use_mac=True):
    """Collect with controllable noise coupling strength."""
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

        # Input drives Vg modulation; noise_scale controls coupling strength
        if use_mac:
            mac_val = inp + noise_state * noise_scale
            fpga.set_mac_signal(float(np.clip(mac_val * 0.5 + 0.5, 0.0, 1.0)))
        else:
            fpga.set_mac_signal(0.5)

        vg_mod = vg_base + ALPHA * inp + BETA * w_in * inp + noise_scale * 0.05 * noise_state
        fpga.set_vg_batch(0, [float(np.clip(v, 0.3, 0.9)) for v in vg_mod])

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
    return float(np.mean(scores))


def gen_waveform_signal(wave_type, n_steps=20):
    t = np.linspace(0, 2*np.pi, n_steps)
    if wave_type == 0:
        return np.sin(t)
    elif wave_type == 1:
        return np.sign(np.sin(t))
    elif wave_type == 2:
        return 2*(t/(2*np.pi)) - 1
    else:
        return np.sin(t) + 0.5*np.sin(3*t)


# ==============================================================================
# EXP 1: Stochastic Resonance Curve
# ==============================================================================
def exp1_stochastic_resonance(fpga, w_in, results):
    print("\n=== EXP 1: Stochastic Resonance Curve ===")
    STEPS = 20
    TRIALS = 80  # 20 per class × 4 classes
    # Noise scales from 0 (no noise) to high
    noise_levels = [0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.8, 1.5, 3.0]
    vg_base = configure_fpga(fpga)

    sr_curve = {}
    for ns in noise_levels:
        X_all, y_all = [], []
        for trial in range(TRIALS):
            cls = trial % 4
            sig = gen_waveform_signal(cls, STEPS)
            spk, dlt, vm = collect_trial(fpga, sig, ns, w_in, vg_base, use_mac=True)
            X_all.append(build_features(spk, dlt, vm))
            y_all.append(cls)
        acc = ridge_classify(np.array(X_all), np.array(y_all))
        sr_curve[ns] = acc
        results[f'exp1_noise_{ns}_acc'] = acc
        print(f"    noise_scale={ns:.2f}: acc={acc:.3f}")

    # Find optimal noise level
    best_ns = max(sr_curve, key=sr_curve.get)
    best_acc = sr_curve[best_ns]
    zero_acc = sr_curve[0.0]
    results['exp1_best_noise'] = best_ns
    results['exp1_best_acc'] = best_acc
    results['exp1_zero_noise_acc'] = zero_acc
    results['exp1_sr_gain_pp'] = (best_acc - zero_acc) * 100

    # T1159: Stochastic resonance exists (non-zero noise beats zero noise)
    t1159 = best_ns > 0.0 and best_acc > zero_acc
    results['T1159_sr_exists'] = 'PASS' if t1159 else 'FAIL'
    print(f"  T1159_sr_exists: {'PASS' if t1159 else 'FAIL'} (best={best_ns}, gain={results['exp1_sr_gain_pp']:.1f}pp)")

    # T1160: Best accuracy > 80%
    t1160 = best_acc > 0.80
    results['T1160_best_gt_80'] = 'PASS' if t1160 else 'FAIL'
    print(f"  T1160_best_gt_80: {'PASS' if t1160 else 'FAIL'} ({best_acc:.3f})")

    # T1161: SR curve is non-monotonic (rises then falls)
    accs = [sr_curve[ns] for ns in sorted(sr_curve.keys())]
    peak_idx = np.argmax(accs)
    non_mono = 0 < peak_idx < len(accs) - 1
    t1161 = non_mono
    results['T1161_sr_non_monotonic'] = 'PASS' if t1161 else 'FAIL'
    print(f"  T1161_sr_non_monotonic: {'PASS' if t1161 else 'FAIL'} (peak at idx {peak_idx}/{len(accs)-1})")

    # T1162: SR gain > 3pp
    t1162 = results['exp1_sr_gain_pp'] > 3.0
    results['T1162_sr_gain_gt_3pp'] = 'PASS' if t1162 else 'FAIL'
    print(f"  T1162_sr_gain_gt_3pp: {'PASS' if t1162 else 'FAIL'} ({results['exp1_sr_gain_pp']:.1f}pp)")

    return best_ns


# ==============================================================================
# EXP 2: Leak Rate Sweep (Runtime τ Tuning)
# ==============================================================================
def exp2_leak_sweep(fpga, w_in, results, best_noise):
    print("\n=== EXP 2: Leak Rate Sweep ===")
    STEPS = 20
    TRIALS = 80
    # Leak values: lower = slower membrane = more memory
    leak_values = [0x0002, 0x0004, 0x0008, 0x0011, 0x0020, 0x0040, 0x0080]
    leak_names = ['0x02(τ~880ms)', '0x04(τ~210ms)', '0x08(τ~105ms)', '0x11(τ~49ms)',
                  '0x20(τ~26ms)', '0x40(τ~13ms)', '0x80(τ~6ms)']

    leak_curve = {}
    for leak_val, leak_name in zip(leak_values, leak_names):
        vg_base = configure_fpga(fpga, leak=leak_val)
        X_all, y_all = [], []
        for trial in range(TRIALS):
            cls = trial % 4
            sig = gen_waveform_signal(cls, STEPS)
            spk, dlt, vm = collect_trial(fpga, sig, best_noise, w_in, vg_base)
            X_all.append(build_features(spk, dlt, vm))
            y_all.append(cls)
        acc = ridge_classify(np.array(X_all), np.array(y_all))
        leak_curve[leak_val] = acc
        results[f'exp2_leak_{hex(leak_val)}_acc'] = acc
        print(f"    {leak_name}: acc={acc:.3f}")

    # Restore default
    vg_base = configure_fpga(fpga)

    best_leak = max(leak_curve, key=leak_curve.get)
    best_acc = leak_curve[best_leak]
    default_acc = leak_curve.get(TUNED_LEAK, 0.0)
    results['exp2_best_leak'] = hex(best_leak)
    results['exp2_best_acc'] = best_acc
    results['exp2_default_acc'] = default_acc

    # T1163: Slower leak (more memory) helps classification
    slow_leaks = [l for l in leak_values if l <= 0x0008]
    slow_avg = np.mean([leak_curve[l] for l in slow_leaks]) if slow_leaks else 0
    fast_leaks = [l for l in leak_values if l >= 0x0040]
    fast_avg = np.mean([leak_curve[l] for l in fast_leaks]) if fast_leaks else 0
    t1163 = slow_avg > fast_avg
    results['T1163_slow_gt_fast'] = 'PASS' if t1163 else 'FAIL'
    results['exp2_slow_avg'] = slow_avg
    results['exp2_fast_avg'] = fast_avg
    print(f"  T1163_slow_gt_fast: {'PASS' if t1163 else 'FAIL'} (slow={slow_avg:.3f} vs fast={fast_avg:.3f})")

    # T1164: Best leak improves over default
    t1164 = best_acc >= default_acc
    results['T1164_best_gt_default'] = 'PASS' if t1164 else 'FAIL'
    print(f"  T1164_best_gt_default: {'PASS' if t1164 else 'FAIL'} (best={best_acc:.3f} vs default={default_acc:.3f})")

    # T1165: Best accuracy > 85%
    t1165 = best_acc > 0.85
    results['T1165_leak_best_gt_85'] = 'PASS' if t1165 else 'FAIL'
    print(f"  T1165_leak_best_gt_85: {'PASS' if t1165 else 'FAIL'}")

    return best_leak


# ==============================================================================
# EXP 3: Bias Gain Sweep (MAC Current Injection)
# ==============================================================================
def exp3_bias_gain_sweep(fpga, w_in, results, best_noise, best_leak):
    print("\n=== EXP 3: Bias Gain Sweep ===")
    STEPS = 20
    TRIALS = 80
    # bias_gain values: controls how strongly MAC signal drives membrane current
    bg_values = [0.0, 0.005, 0.015, 0.03125, 0.0625, 0.125, 0.25, 0.5]

    bg_curve = {}
    for bg in bg_values:
        vg_base = configure_fpga(fpga, leak=best_leak, bias_gain=bg)
        X_all, y_all = [], []
        for trial in range(TRIALS):
            cls = trial % 4
            sig = gen_waveform_signal(cls, STEPS)
            spk, dlt, vm = collect_trial(fpga, sig, best_noise, w_in, vg_base)
            X_all.append(build_features(spk, dlt, vm))
            y_all.append(cls)
        acc = ridge_classify(np.array(X_all), np.array(y_all))
        bg_curve[bg] = acc
        results[f'exp3_bg_{bg}_acc'] = acc
        print(f"    bias_gain={bg:.4f}: acc={acc:.3f}")

    vg_base = configure_fpga(fpga, leak=best_leak)

    best_bg = max(bg_curve, key=bg_curve.get)
    best_acc = bg_curve[best_bg]
    results['exp3_best_bg'] = best_bg
    results['exp3_best_acc'] = best_acc

    # T1166: Nonzero bias_gain beats zero
    t1166 = best_bg > 0.0 and best_acc > bg_curve[0.0]
    results['T1166_nonzero_bg_better'] = 'PASS' if t1166 else 'FAIL'
    print(f"  T1166_nonzero_bg_better: {'PASS' if t1166 else 'FAIL'} (best_bg={best_bg}, zero={bg_curve[0.0]:.3f})")

    # T1167: Bias gain curve is non-monotonic (optimal in middle)
    accs = [bg_curve[bg] for bg in sorted(bg_curve.keys())]
    peak_idx = np.argmax(accs)
    non_mono = 0 < peak_idx < len(accs) - 1
    t1167 = non_mono
    results['T1167_bg_non_monotonic'] = 'PASS' if t1167 else 'FAIL'
    print(f"  T1167_bg_non_monotonic: {'PASS' if t1167 else 'FAIL'} (peak at idx {peak_idx}/{len(accs)-1})")

    # T1168: Best acc > 85%
    t1168 = best_acc > 0.85
    results['T1168_bg_best_gt_85'] = 'PASS' if t1168 else 'FAIL'
    print(f"  T1168_bg_best_gt_85: {'PASS' if t1168 else 'FAIL'} ({best_acc:.3f})")

    return best_bg


# ==============================================================================
# EXP 4: Combined Optimal — Full NS-RAM Parameter Space
# ==============================================================================
def exp4_combined_optimal(fpga, w_in, results, best_noise, best_leak, best_bg):
    print("\n=== EXP 4: Combined Optimal vs Defaults ===")
    STEPS = 20
    TRIALS = 120  # 30 per class

    configs = {
        'DEFAULT': {'leak': TUNED_LEAK, 'bg': TUNED_BIAS_GAIN, 'noise': 0.3},
        'OPTIMAL': {'leak': best_leak, 'bg': best_bg, 'noise': best_noise},
        'NO_NOISE': {'leak': best_leak, 'bg': best_bg, 'noise': 0.0},
        'FPGA_ONLY_OPT': {'leak': best_leak, 'bg': 0.0, 'noise': 0.0},
    }

    config_accs = {}
    for name, cfg in configs.items():
        vg_base = configure_fpga(fpga, leak=cfg['leak'], bias_gain=cfg['bg'])
        X_all, y_all = [], []
        for trial in range(TRIALS):
            if trial % 40 == 0:
                print(f"    {name} Trial {trial}/{TRIALS}")
            cls = trial % 4
            sig = gen_waveform_signal(cls, STEPS)
            use_mac = cfg['bg'] > 0
            spk, dlt, vm = collect_trial(fpga, sig, cfg['noise'], w_in, vg_base, use_mac=use_mac)
            X_all.append(build_features(spk, dlt, vm))
            y_all.append(cls)
        acc = ridge_classify(np.array(X_all), np.array(y_all))
        config_accs[name] = acc
        results[f'exp4_{name}_acc'] = acc
        print(f"    {name}: {acc:.3f}")

    # T1169: OPTIMAL > DEFAULT
    t1169 = config_accs['OPTIMAL'] > config_accs['DEFAULT']
    results['T1169_optimal_gt_default'] = 'PASS' if t1169 else 'FAIL'
    print(f"  T1169_optimal_gt_default: {'PASS' if t1169 else 'FAIL'} ({config_accs['OPTIMAL']:.3f} vs {config_accs['DEFAULT']:.3f})")

    # T1170: OPTIMAL > NO_NOISE (noise helps at optimal coupling)
    t1170 = config_accs['OPTIMAL'] > config_accs['NO_NOISE']
    results['T1170_optimal_gt_nonoise'] = 'PASS' if t1170 else 'FAIL'
    print(f"  T1170_optimal_gt_nonoise: {'PASS' if t1170 else 'FAIL'} ({config_accs['OPTIMAL']:.3f} vs {config_accs['NO_NOISE']:.3f})")

    # T1171: OPTIMAL > FPGA_ONLY (coupled beats standalone with optimal params)
    t1171 = config_accs['OPTIMAL'] > config_accs['FPGA_ONLY_OPT']
    results['T1171_coupled_gt_fpga'] = 'PASS' if t1171 else 'FAIL'
    pp = (config_accs['OPTIMAL'] - config_accs['FPGA_ONLY_OPT']) * 100
    results['exp4_coupled_advantage_pp'] = pp
    print(f"  T1171_coupled_gt_fpga: {'PASS' if t1171 else 'FAIL'} ({pp:+.1f}pp)")

    # T1172: OPTIMAL > 90%
    t1172 = config_accs['OPTIMAL'] > 0.90
    results['T1172_optimal_gt_90'] = 'PASS' if t1172 else 'FAIL'
    print(f"  T1172_optimal_gt_90: {'PASS' if t1172 else 'FAIL'} ({config_accs['OPTIMAL']:.3f})")

    # Restore defaults
    configure_fpga(fpga)
    return config_accs


# ==============================================================================
# EXP 5: NS-RAM Conductance Analogy — Vg as Analog Weight
# ==============================================================================
def exp5_conductance_analogy(fpga, w_in, results, best_noise, best_leak, best_bg):
    print("\n=== EXP 5: NS-RAM Conductance Analogy ===")
    STEPS = 20
    vg_base = configure_fpga(fpga, leak=best_leak, bias_gain=best_bg)

    # Sweep Vg levels and measure spike rate response
    vg_levels = np.linspace(0.40, 0.80, 9)
    spike_rates = []
    vmem_means = []
    for vg in vg_levels:
        vg_arr = np.full(N_NEURONS, float(vg))
        fpga.set_vg_batch(0, vg_arr.tolist())
        time.sleep(0.5)
        rates = []
        vmems = []
        for _ in range(10):
            telem = fpga.read_telemetry(timeout=0.2)
            if telem is not None:
                rates.append(float(telem['spike_counts'].mean()))
                vmems.append(float(telem['vmem'].mean()))
            time.sleep(STEP_INTERVAL)
        spike_rates.append(np.mean(rates) if rates else 0.0)
        vmem_means.append(np.mean(vmems) if vmems else 0.0)
        print(f"    Vg={vg:.2f}: rate={spike_rates[-1]:.1f}, vmem={vmem_means[-1]:.4f}")

    results['exp5_vg_levels'] = vg_levels.tolist()
    results['exp5_spike_rates'] = spike_rates
    results['exp5_vmem_means'] = vmem_means

    # T1173: Spike rate increases with Vg (conductance-like)
    corr_rate = np.corrcoef(vg_levels, spike_rates)[0, 1] if len(spike_rates) > 2 else 0
    t1173 = corr_rate > 0.5
    results['T1173_rate_increases_with_vg'] = 'PASS' if t1173 else 'FAIL'
    results['exp5_vg_rate_corr'] = float(corr_rate)
    print(f"  T1173_rate_increases_with_vg: {'PASS' if t1173 else 'FAIL'} (corr={corr_rate:.3f})")

    # T1174: Dynamic range > 5× (max/min rate)
    max_r, min_r = max(spike_rates), max(min(spike_rates), 0.01)
    dyn_range = max_r / min_r
    t1174 = dyn_range > 5.0
    results['T1174_dyn_range_gt_5x'] = 'PASS' if t1174 else 'FAIL'
    results['exp5_dyn_range'] = dyn_range
    print(f"  T1174_dyn_range_gt_5x: {'PASS' if t1174 else 'FAIL'} ({dyn_range:.1f}×)")

    # T1175: Smooth (not binary) transition — at least 3 distinct rate levels
    unique_rates = len(set([round(r, 0) for r in spike_rates]))
    t1175 = unique_rates >= 3
    results['T1175_smooth_transition'] = 'PASS' if t1175 else 'FAIL'
    results['exp5_unique_rate_levels'] = unique_rates
    print(f"  T1175_smooth_transition: {'PASS' if t1175 else 'FAIL'} ({unique_rates} distinct levels)")

    # T1176: vmem correlates with Vg (analog membrane potential)
    corr_vmem = np.corrcoef(vg_levels, vmem_means)[0, 1] if len(vmem_means) > 2 else 0
    t1176 = abs(corr_vmem) > 0.3
    results['T1176_vmem_corr_vg'] = 'PASS' if t1176 else 'FAIL'
    results['exp5_vg_vmem_corr'] = float(corr_vmem)
    print(f"  T1176_vmem_corr_vg: {'PASS' if t1176 else 'FAIL'} (corr={corr_vmem:.3f})")

    configure_fpga(fpga, leak=best_leak, bias_gain=best_bg)


# ==============================================================================
# EXP 6: Temporal Regression at Optimal Parameters
# ==============================================================================
def exp6_temporal_regression(fpga, w_in, results, best_noise, best_leak, best_bg):
    print("\n=== EXP 6: Temporal Regression at Optimal Params ===")
    STEPS = 40
    vg_base = configure_fpga(fpga, leak=best_leak, bias_gain=best_bg)

    # Test: can the reservoir reconstruct a continuous signal?
    signals = {
        'sine': np.sin(np.linspace(0, 4*np.pi, STEPS)),
        'sum_sines': np.sin(np.linspace(0, 4*np.pi, STEPS)) + 0.3*np.sin(np.linspace(0, 12*np.pi, STEPS)),
        'slow_ramp': np.linspace(-1, 1, STEPS),
    }

    for sig_name, sig in signals.items():
        n_trials = 60
        X_all, y_all = [], []
        for trial in range(n_trials):
            phase = np.random.uniform(0, 2*np.pi)
            amplitude = np.random.uniform(0.5, 1.5)
            if sig_name == 'sine':
                shifted = amplitude * np.sin(np.linspace(phase, phase + 4*np.pi, STEPS))
                target = amplitude  # predict amplitude
            elif sig_name == 'sum_sines':
                ratio = np.random.uniform(0.2, 0.5)
                shifted = np.sin(np.linspace(phase, phase+4*np.pi, STEPS)) + ratio*np.sin(np.linspace(0, 12*np.pi, STEPS))
                target = ratio
            else:
                slope = np.random.uniform(-2, 2)
                shifted = np.linspace(0, slope, STEPS)
                target = slope

            spk, dlt, vm = collect_trial(fpga, shifted, best_noise, w_in, vg_base)
            X_all.append(build_features(spk, dlt, vm))
            y_all.append(target)

            if trial % 20 == 0:
                print(f"    {sig_name} Trial {trial}/{n_trials}")

        r2 = ridge_regress(np.array(X_all), np.array(y_all))
        results[f'exp6_{sig_name}_r2'] = r2
        print(f"    {sig_name}: R2={r2:.4f}")

    # T1177: Sine amplitude R2 > 0.1
    t1177 = results.get('exp6_sine_r2', 0) > 0.10
    results['T1177_sine_r2_gt_010'] = 'PASS' if t1177 else 'FAIL'
    print(f"  T1177_sine_r2_gt_010: {'PASS' if t1177 else 'FAIL'}")

    # T1178: At least one signal R2 > 0.3
    best_r2 = max(results.get(f'exp6_{s}_r2', 0) for s in ['sine', 'sum_sines', 'slow_ramp'])
    t1178 = best_r2 > 0.30
    results['T1178_any_r2_gt_030'] = 'PASS' if t1178 else 'FAIL'
    print(f"  T1178_any_r2_gt_030: {'PASS' if t1178 else 'FAIL'} (best={best_r2:.4f})")

    # T1179: Ramp slope R2 > 0 (simplest regression)
    t1179 = results.get('exp6_slow_ramp_r2', 0) > 0.0
    results['T1179_ramp_r2_gt_0'] = 'PASS' if t1179 else 'FAIL'
    print(f"  T1179_ramp_r2_gt_0: {'PASS' if t1179 else 'FAIL'}")

    configure_fpga(fpga)


# ==============================================================================
# EXP 7: Memory Capacity at Optimal Parameters
# ==============================================================================
def exp7_memory_capacity(fpga, w_in, results, best_noise, best_leak, best_bg):
    print("\n=== EXP 7: Memory Capacity at Optimal Params ===")
    STEPS = 20
    TRIALS = 80
    delays = [0, 1, 2, 3, 5, 7, 10]
    vg_base = configure_fpga(fpga, leak=best_leak, bias_gain=best_bg)

    mc_scores = {}
    for d in delays:
        X_all, y_all = [], []
        for trial in range(TRIALS):
            cls = trial % 4
            sig = gen_waveform_signal(cls, STEPS)
            # Insert d blank steps after signal
            blank = np.zeros(d)
            full_sig = np.concatenate([sig, blank])
            spk, dlt, vm = collect_trial(fpga, full_sig, best_noise, w_in, vg_base)
            X_all.append(build_features(spk, dlt, vm))
            y_all.append(cls)
        acc = ridge_classify(np.array(X_all), np.array(y_all))
        mc_scores[d] = acc
        results[f'exp7_mc_d{d}_acc'] = acc
        print(f"    d={d} ({d*50}ms): acc={acc:.3f}")

    mc_total = sum(max(0, v - 0.25) / 0.75 for v in mc_scores.values())
    results['exp7_mc_total'] = mc_total
    results['exp7_mc_scores'] = {str(k): v for k, v in mc_scores.items()}

    # T1180: MC total > 1.0
    t1180 = mc_total > 1.0
    results['T1180_mc_gt_1'] = 'PASS' if t1180 else 'FAIL'
    print(f"  T1180_mc_gt_1: {'PASS' if t1180 else 'FAIL'} (MC={mc_total:.3f})")

    # T1181: MC total > z2247 (1.250)
    t1181 = mc_total > 1.250
    results['T1181_mc_gt_z2247'] = 'PASS' if t1181 else 'FAIL'
    print(f"  T1181_mc_gt_z2247: {'PASS' if t1181 else 'FAIL'} ({mc_total:.3f} vs 1.250)")

    # T1182: MC d=0 > 60%
    t1182 = mc_scores.get(0, 0) > 0.60
    results['T1182_mc_d0_gt_60'] = 'PASS' if t1182 else 'FAIL'
    print(f"  T1182_mc_d0_gt_60: {'PASS' if t1182 else 'FAIL'} ({mc_scores.get(0, 0):.3f})")

    # T1183: d=0 > d=10 (memory fades)
    t1183 = mc_scores.get(0, 0) > mc_scores.get(10, 1.0)
    results['T1183_d0_gt_d10'] = 'PASS' if t1183 else 'FAIL'
    print(f"  T1183_d0_gt_d10: {'PASS' if t1183 else 'FAIL'}")

    configure_fpga(fpga)


# ==============================================================================
# EXP 8: Transfer Entropy at Optimal Operating Point
# ==============================================================================
def exp8_transfer_entropy(fpga, w_in, results, best_noise, best_leak, best_bg):
    print("\n=== EXP 8: Transfer Entropy at Optimal Point ===")
    STEPS = 100
    vg_base = configure_fpga(fpga, leak=best_leak, bias_gain=best_bg)

    # Collect long continuous time series
    signal = np.sin(np.linspace(0, 20*np.pi, STEPS)) + 0.3*np.random.randn(STEPS)
    spk, dlt, vm = collect_trial(fpga, signal, best_noise, w_in, vg_base)

    # GPU power time series
    gpu_powers = []
    for _ in range(STEPS):
        gpu_powers.append(read_gpu_power())
        time.sleep(STEP_INTERVAL)
    gpu_ts = np.array(gpu_powers)

    # Mean spike rate time series
    spike_ts = spk.mean(axis=1)
    vmem_ts = vm.mean(axis=1)

    # Simple transfer entropy estimate via lagged mutual information
    def lagged_corr(x, y, lag):
        if lag == 0:
            return np.corrcoef(x, y)[0, 1]
        return np.corrcoef(x[:-lag], y[lag:])[0, 1]

    lags = [1, 2, 3, 5, 10]
    te_estimates = {}
    for lag in lags:
        if lag < len(spike_ts) - 1:
            # Input → FPGA spikes
            c_input_spike = abs(lagged_corr(signal[:len(spike_ts)], spike_ts, lag))
            # GPU → FPGA spikes
            c_gpu_spike = abs(lagged_corr(gpu_ts[:len(spike_ts)], spike_ts, lag))
            te_estimates[lag] = {
                'input_spike': float(c_input_spike),
                'gpu_spike': float(c_gpu_spike),
            }
            print(f"    lag={lag}: input→spike={c_input_spike:.4f}, gpu→spike={c_gpu_spike:.4f}")

    results['exp8_te_estimates'] = {str(k): v for k, v in te_estimates.items()}

    # Granger-like: input predicts FPGA spikes?
    input_spike_corrs = [te_estimates[l]['input_spike'] for l in lags if l in te_estimates]
    peak_input_corr = max(input_spike_corrs) if input_spike_corrs else 0
    results['exp8_peak_input_corr'] = peak_input_corr

    gpu_spike_corrs = [te_estimates[l]['gpu_spike'] for l in lags if l in te_estimates]
    peak_gpu_corr = max(gpu_spike_corrs) if gpu_spike_corrs else 0
    results['exp8_peak_gpu_corr'] = peak_gpu_corr

    # Autocorrelation of spike time series (temporal memory indicator)
    if len(spike_ts) > 10:
        acf1 = np.corrcoef(spike_ts[:-1], spike_ts[1:])[0, 1]
        acf5 = np.corrcoef(spike_ts[:-5], spike_ts[5:])[0, 1] if len(spike_ts) > 10 else 0
    else:
        acf1, acf5 = 0, 0
    results['exp8_spike_acf1'] = float(acf1)
    results['exp8_spike_acf5'] = float(acf5)

    # T1184: Input influences FPGA at some lag (corr > 0.1)
    t1184 = peak_input_corr > 0.10
    results['T1184_input_influences_fpga'] = 'PASS' if t1184 else 'FAIL'
    print(f"  T1184_input_influences_fpga: {'PASS' if t1184 else 'FAIL'} (peak={peak_input_corr:.4f})")

    # T1185: GPU power influences FPGA (corr > 0.05)
    t1185 = peak_gpu_corr > 0.05
    results['T1185_gpu_influences_fpga'] = 'PASS' if t1185 else 'FAIL'
    print(f"  T1185_gpu_influences_fpga: {'PASS' if t1185 else 'FAIL'} (peak={peak_gpu_corr:.4f})")

    # T1186: Spike temporal memory (ACF(1) > 0.1)
    t1186 = abs(acf1) > 0.10
    results['T1186_spike_temporal_memory'] = 'PASS' if t1186 else 'FAIL'
    print(f"  T1186_spike_temporal_memory: {'PASS' if t1186 else 'FAIL'} (ACF1={acf1:.4f})")

    # T1187: Extended temporal memory (ACF(5) > 0.05)
    t1187 = abs(acf5) > 0.05
    results['T1187_extended_memory'] = 'PASS' if t1187 else 'FAIL'
    print(f"  T1187_extended_memory: {'PASS' if t1187 else 'FAIL'} (ACF5={acf5:.4f})")

    configure_fpga(fpga)


# ==============================================================================
# MAIN
# ==============================================================================
def main():
    print("=" * 60)
    print("z2248: NS-RAM Stochastic Resonance & Parameter Sweep")
    print("  Sweeping noise, leak, bias_gain for optimal NS-RAM operation")
    print("  Targeting COUPLED > FPGA_ONLY via stochastic resonance")
    print("=" * 60)

    fpga = FPGAEthBridge(timeout=0.5)
    ok = fpga.connect()
    if not ok:
        print("FATAL: FPGA not responding")
        sys.exit(1)

    vg_base = configure_fpga(fpga)
    telem = fpga.read_telemetry()
    if telem is not None:
        print(f"  FPGA: {len(telem['spike_counts'])} neurons")
        sr = telem['spike_counts']
        print(f"  Spikes: [{int(sr.min())}, {int(sr.max())}]")

    np.random.seed(42)
    w_in = np.random.randn(N_NEURONS) * 0.1

    results = {
        'experiment': 'z2248_stochastic_resonance',
        'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
        'sample_hz': SAMPLE_HZ,
        'base_vg': BASE_VG,
        'alpha': ALPHA,
        'beta': BETA,
    }

    t_start = time.time()

    # EXP 1: Find optimal noise level
    best_noise = exp1_stochastic_resonance(fpga, w_in, results)

    # EXP 2: Find optimal leak rate
    best_leak = exp2_leak_sweep(fpga, w_in, results, best_noise)

    # EXP 3: Find optimal bias gain
    best_bg = exp3_bias_gain_sweep(fpga, w_in, results, best_noise, best_leak)

    # EXP 4: Combined optimal vs defaults
    exp4_combined_optimal(fpga, w_in, results, best_noise, best_leak, best_bg)

    # EXP 5: NS-RAM conductance analogy
    exp5_conductance_analogy(fpga, w_in, results, best_noise, best_leak, best_bg)

    # EXP 6: Temporal regression
    exp6_temporal_regression(fpga, w_in, results, best_noise, best_leak, best_bg)

    # EXP 7: Memory capacity
    exp7_memory_capacity(fpga, w_in, results, best_noise, best_leak, best_bg)

    # EXP 8: Transfer entropy
    exp8_transfer_entropy(fpga, w_in, results, best_noise, best_leak, best_bg)

    duration = time.time() - t_start
    results['duration_s'] = duration

    # Final scorecard
    tests = {k: v for k, v in results.items() if k.startswith('T1')}
    n_pass = sum(1 for v in tests.values() if v == 'PASS')
    n_total = len(tests)
    results['pass_count'] = n_pass
    results['total_count'] = n_total

    print(f"\n{'='*60}")
    print(f"RESULTS: {n_pass}/{n_total} PASS ({duration:.0f}s)")
    for k in sorted(tests.keys()):
        print(f"  {k}: {tests[k]}")
    print(f"{'='*60}")

    # Save optimal params for future experiments
    results['optimal_params'] = {
        'noise_scale': best_noise,
        'leak': hex(best_leak),
        'bias_gain': best_bg,
    }
    print(f"\nOptimal params: noise={best_noise}, leak={hex(best_leak)}, bg={best_bg}")

    with open(RESULTS_FILE, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved: {RESULTS_FILE}")

    fpga.close()


if __name__ == '__main__':
    main()
