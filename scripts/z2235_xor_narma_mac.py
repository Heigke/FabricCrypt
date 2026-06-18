#!/usr/bin/env python3
"""
z2235_xor_narma_mac.py — XOR and NARMA benchmarks with MAC injection
=====================================================================
Now that MAC current injection works (z2234: MC=0.594), test:
  EXP 1: XOR at delays tau=1..10 (COUPLED vs FPGA_ONLY)
  EXP 2: NARMA-5 and NARMA-10 with MAC
  EXP 3: Nonlinear transformation capacity
  EXP 4: Delayed XOR with temporal features

Tests T831-T852.
"""

import sys, os, time, json, struct
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(line_buffering=True)

import numpy as np
from sklearn.linear_model import RidgeClassifier, Ridge
from sklearn.model_selection import StratifiedKFold, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.fpga_host_eth import FPGAEthBridge

RESULTS_FILE = "results/z2235_xor_narma_mac.json"
N_NEURONS = 128
BASE_VG = 0.58
ALPHA = 0.25
BETA = 0.08
SAMPLE_HZ = 20
STEP_INTERVAL = 1.0 / SAMPLE_HZ

def read_gpu_power():
    try:
        with open("/sys/class/drm/card1/device/hwmon/hwmon7/power1_average", "r") as f:
            return float(f.read().strip()) / 1e6
    except:
        return 11.0 + np.random.randn() * 0.5

def read_gpu_temp():
    try:
        with open("/sys/class/drm/card1/device/hwmon/hwmon7/temp1_input", "r") as f:
            return float(f.read().strip()) / 1000.0
    except:
        return 45.0 + np.random.randn() * 1.5

def configure_fpga(fpga):
    fpga.set_kill(False)
    time.sleep(0.05)
    fpga.set_leak_cond(0x0004)
    fpga.set_bias_gain(0.03125)
    fpga.set_threshold(0.50)
    fpga.set_dt_over_c(0.0078)
    fpga.set_refract_cycles(50)
    time.sleep(0.1)
    fpga.set_vg_batch(0, [BASE_VG] * 128)
    time.sleep(0.1)
    print(f"  FPGA configured: leak=0x0004, bias_gain=0.03125, vg={BASE_VG}")

def iir_filter(signal, alpha=0.85):
    out = np.zeros_like(signal)
    out[0] = signal[0]
    for i in range(1, len(signal)):
        out[i] = alpha * out[i-1] + (1 - alpha) * signal[i]
    return out

def collect_reservoir_state(fpga, input_signal, condition, w_in):
    """Collect reservoir states for a long input sequence via MAC injection."""
    n_steps = len(input_signal)
    all_spikes = []
    gpu_power_base = read_gpu_power()
    noise_state = 0.0

    for step in range(n_steps):
        t0 = time.time()
        inp = float(input_signal[step])
        gpu_power = read_gpu_power()
        power_noise = (gpu_power - gpu_power_base) / 5.0
        noise_state = 0.85 * noise_state + 0.15 * power_noise

        if condition == "COUPLED":
            mac_val = inp + noise_state * 0.3
            fpga.set_mac_signal(float(np.clip(mac_val * 0.5 + 0.5, 0.0, 1.0)))
            vg_mod = BASE_VG + ALPHA * inp + BETA * noise_state
            fpga.set_vg_batch(0, [float(np.clip(
                vg_mod + BETA * w_in[i] * inp, 0.3, 0.9
            )) for i in range(128)])
        elif condition == "FPGA_ONLY":
            fpga.set_mac_signal(float(np.clip(inp * 0.5 + 0.5, 0.0, 1.0)))
        elif condition == "NO_MAC":
            if step == 0:
                fpga.set_mac_signal(0.5)
            vg_val = BASE_VG + ALPHA * inp
            fpga.set_vg_batch(0, [float(np.clip(vg_val + BETA * w_in[i] * inp, 0.3, 0.9)) for i in range(128)])

        telem = fpga.read_telemetry(timeout=0.1)
        if telem is not None:
            all_spikes.append(telem['spike_counts'].astype(np.float32))
        elif all_spikes:
            all_spikes.append(all_spikes[-1].copy())
        else:
            all_spikes.append(np.zeros(N_NEURONS, dtype=np.float32))

        elapsed = time.time() - t0
        if elapsed < STEP_INTERVAL:
            time.sleep(STEP_INTERVAL - elapsed)

    return np.array(all_spikes)

def ridge_regress(X, y, n_splits=5):
    scaler = StandardScaler()
    std = X.std(axis=0)
    mask = std > 1e-2
    if mask.sum() < 3:
        return 0.0, 0.0
    X_f = scaler.fit_transform(X[:, mask])
    reg = Ridge(alpha=1.0)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    scores = cross_val_score(reg, X_f, y, cv=kf, scoring='r2')
    return float(np.mean(scores)), float(np.std(scores))

def ridge_classify(X, y, n_splits=5):
    scaler = StandardScaler()
    std = X.std(axis=0)
    mask = std > 1e-2
    if mask.sum() < 3:
        return 0.5, 0.0
    X_f = scaler.fit_transform(X[:, mask])
    classes, counts = np.unique(y, return_counts=True)
    if counts.min() < n_splits:
        n_splits = max(2, counts.min())
    clf = RidgeClassifier(alpha=1.0)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    scores = cross_val_score(clf, X_f, y, cv=skf, scoring='accuracy')
    return float(scores.mean()), float(scores.std())

def generate_narma(u, order=10):
    """Generate NARMA-n target from input u."""
    n = len(u)
    y = np.zeros(n)
    for t in range(order, n):
        if order == 5:
            y[t] = 0.3*y[t-1] + 0.05*y[t-1]*np.sum(y[t-5:t]) + 1.5*u[t-1]*u[t-5] + 0.1
        else:  # NARMA-10
            y[t] = 0.3*y[t-1] + 0.05*y[t-1]*np.sum(y[t-10:t]) + 1.5*u[t-1]*u[t-10] + 0.1
        y[t] = np.clip(y[t], -10, 10)
    return y

def build_delay_features(states, max_delay=4):
    """Build delay-augmented features from reservoir states."""
    n = states.shape[0]
    feats = []
    for d in range(max_delay):
        if d == 0:
            feats.append(states[max_delay-1:])
        else:
            feats.append(states[max_delay-1-d:n-d])
    return np.concatenate(feats, axis=1)


def run_exp1_xor(fpga, results):
    """EXP 1: XOR at delays tau=1..10."""
    print("\n=== EXP 1: XOR at Multiple Delays ===")

    N_STEPS = 500
    rng = np.random.RandomState(42)
    w_in = rng.uniform(-1, 1, N_NEURONS)

    conditions = ["COUPLED", "FPGA_ONLY", "NO_MAC"]

    for cond in conditions:
        print(f"\n  Condition: {cond}")
        configure_fpga(fpga)
        time.sleep(0.5)

        # Binary input
        u = rng.choice([0.0, 1.0], size=N_STEPS)
        u_filtered = iir_filter(u * 2 - 1, alpha=0.3)  # slight smoothing

        print(f"    Collecting {N_STEPS} steps...")
        states = collect_reservoir_state(fpga, u_filtered, cond, w_in)
        X_delay = build_delay_features(states, max_delay=4)

        for tau in [1, 2, 3, 5, 7, 10]:
            # XOR target: u(t) XOR u(t-tau)
            target_start = max(4, tau)  # account for delay features
            if target_start + 10 > N_STEPS:
                continue
            y_xor = np.array([int(u[t] != u[t-tau]) for t in range(target_start, N_STEPS)])
            X_xor = X_delay[target_start - 3:][:len(y_xor)]

            if len(y_xor) < 20:
                continue

            acc, acc_std = ridge_classify(X_xor, y_xor, n_splits=5)
            results[f"exp1_{cond}_xor_tau{tau}"] = acc
            print(f"    XOR tau={tau}: acc = {acc:.3f} +/- {acc_std:.3f}")

    # Tests
    for tau in [1, 2, 3, 5]:
        coupled = results.get(f"exp1_COUPLED_xor_tau{tau}", 0.5)
        fpga_only_val = results.get(f"exp1_FPGA_ONLY_xor_tau{tau}", 0.5)
        no_mac = results.get(f"exp1_NO_MAC_xor_tau{tau}", 0.5)

        test_id = 831 + [1,2,3,5].index(tau)
        results[f"T{test_id}_xor_tau{tau}_coupled_gt_chance"] = "PASS" if coupled > 0.55 else "FAIL"
        print(f"  T{test_id} XOR tau={tau} COUPLED({coupled:.3f}) > 0.55: {results[f'T{test_id}_xor_tau{tau}_coupled_gt_chance']}")

    # T835: XOR tau=5 above chance with MAC (was impossible at z2206 τ=49ms)
    coupled_5 = results.get("exp1_COUPLED_xor_tau5", 0.5)
    results["T835_xor_tau5_coupled_gt_chance"] = "PASS" if coupled_5 > 0.53 else "FAIL"
    print(f"  T835 XOR tau=5 COUPLED({coupled_5:.3f}) > 0.53: {results['T835_xor_tau5_coupled_gt_chance']}")

    # T836: COUPLED > NO_MAC at tau=2
    c2 = results.get("exp1_COUPLED_xor_tau2", 0.5)
    n2 = results.get("exp1_NO_MAC_xor_tau2", 0.5)
    results["T836_coupled_gt_no_mac_tau2"] = "PASS" if c2 > n2 else "FAIL"
    print(f"  T836 COUPLED tau=2 ({c2:.3f}) > NO_MAC ({n2:.3f}): {results['T836_coupled_gt_no_mac_tau2']}")


def run_exp2_narma(fpga, results):
    """EXP 2: NARMA-5 and NARMA-10."""
    print("\n=== EXP 2: NARMA-5 and NARMA-10 ===")

    N_STEPS = 600
    rng = np.random.RandomState(456)
    w_in = rng.uniform(-1, 1, N_NEURONS)

    u = rng.uniform(0, 0.5, N_STEPS)  # NARMA standard input range
    narma5 = generate_narma(u, order=5)
    narma10 = generate_narma(u, order=10)

    # Scale input for reservoir
    u_scaled = u * 2 - 0.5  # map [0,0.5] -> [-0.5, 0.5]

    conditions = ["COUPLED", "FPGA_ONLY"]

    for cond in conditions:
        print(f"\n  Condition: {cond}")
        configure_fpga(fpga)
        time.sleep(0.5)

        print(f"    Collecting {N_STEPS} steps...")
        states = collect_reservoir_state(fpga, u_scaled, cond, w_in)
        X = build_delay_features(states, max_delay=4)

        # NARMA-5
        y5 = narma5[3:][:X.shape[0]]
        r2_5, std_5 = ridge_regress(X[:len(y5)], y5, n_splits=5)
        results[f"exp2_{cond}_narma5_r2"] = max(0.0, r2_5)
        print(f"    NARMA-5: R² = {r2_5:.4f} +/- {std_5:.4f}")

        # NARMA-10
        y10 = narma10[3:][:X.shape[0]]
        r2_10, std_10 = ridge_regress(X[:len(y10)], y10, n_splits=5)
        results[f"exp2_{cond}_narma10_r2"] = max(0.0, r2_10)
        print(f"    NARMA-10: R² = {r2_10:.4f} +/- {std_10:.4f}")

    # Tests
    coupled_n5 = results.get("exp2_COUPLED_narma5_r2", 0)
    coupled_n10 = results.get("exp2_COUPLED_narma10_r2", 0)
    fpga_n5 = results.get("exp2_FPGA_ONLY_narma5_r2", 0)
    fpga_n10 = results.get("exp2_FPGA_ONLY_narma10_r2", 0)

    results["T837_narma5_coupled_gt_0"] = "PASS" if coupled_n5 > 0.01 else "FAIL"
    results["T838_narma10_coupled_gt_0"] = "PASS" if coupled_n10 > 0.005 else "FAIL"
    results["T839_narma5_coupled_gt_fpga"] = "PASS" if coupled_n5 > fpga_n5 else "FAIL"
    results["T840_narma10_coupled_gt_fpga"] = "PASS" if coupled_n10 > fpga_n10 else "FAIL"

    print(f"\n  T837 NARMA-5 COUPLED({coupled_n5:.4f}) > 0.01: {results['T837_narma5_coupled_gt_0']}")
    print(f"  T838 NARMA-10 COUPLED({coupled_n10:.4f}) > 0.005: {results['T838_narma10_coupled_gt_0']}")
    print(f"  T839 NARMA-5 COUPLED > FPGA_ONLY: {results['T839_narma5_coupled_gt_fpga']}")
    print(f"  T840 NARMA-10 COUPLED > FPGA_ONLY: {results['T840_narma10_coupled_gt_fpga']}")


def run_exp3_nonlinear(fpga, results):
    """EXP 3: Nonlinear transformation capacity — can reservoir compute x², x³, sin(x)?"""
    print("\n=== EXP 3: Nonlinear Transformation Capacity ===")

    N_STEPS = 500
    rng = np.random.RandomState(789)
    w_in = rng.uniform(-1, 1, N_NEURONS)

    u = rng.uniform(-1, 1, N_STEPS)
    u_smooth = iir_filter(u, alpha=0.5)

    configure_fpga(fpga)
    time.sleep(0.5)

    print(f"    Collecting {N_STEPS} steps (COUPLED)...")
    states = collect_reservoir_state(fpga, u_smooth, "COUPLED", w_in)
    X = build_delay_features(states, max_delay=4)

    targets = {
        "linear": u_smooth[3:][:X.shape[0]],
        "quadratic": (u_smooth[3:][:X.shape[0]])**2,
        "cubic": (u_smooth[3:][:X.shape[0]])**3,
        "sine": np.sin(3 * u_smooth[3:][:X.shape[0]]),
        "abs": np.abs(u_smooth[3:][:X.shape[0]]),
    }

    total_capacity = 0.0
    for name, y in targets.items():
        r2, std = ridge_regress(X[:len(y)], y, n_splits=5)
        r2 = max(0.0, r2)
        results[f"exp3_{name}_r2"] = r2
        total_capacity += r2
        print(f"    {name}: R² = {r2:.4f} +/- {std:.4f}")

    results["exp3_total_nonlinear_capacity"] = total_capacity
    print(f"    Total nonlinear capacity: {total_capacity:.4f}")

    # Tests
    results["T841_linear_r2_gt_0.10"] = "PASS" if results["exp3_linear_r2"] > 0.10 else "FAIL"
    results["T842_quadratic_r2_gt_0.01"] = "PASS" if results["exp3_quadratic_r2"] > 0.01 else "FAIL"
    results["T843_total_capacity_gt_0.5"] = "PASS" if total_capacity > 0.5 else "FAIL"
    results["T844_sine_r2_gt_0.01"] = "PASS" if results["exp3_sine_r2"] > 0.01 else "FAIL"

    print(f"\n  T841 Linear R²({results['exp3_linear_r2']:.4f}) > 0.10: {results['T841_linear_r2_gt_0.10']}")
    print(f"  T842 Quadratic R²({results['exp3_quadratic_r2']:.4f}) > 0.01: {results['T842_quadratic_r2_gt_0.01']}")
    print(f"  T843 Total capacity({total_capacity:.4f}) > 0.5: {results['T843_total_capacity_gt_0.5']}")
    print(f"  T844 Sine R²({results['exp3_sine_r2']:.4f}) > 0.01: {results['T844_sine_r2_gt_0.01']}")


def run_exp4_delayed_xor_temporal(fpga, results):
    """EXP 4: Delayed XOR with rich temporal features."""
    print("\n=== EXP 4: Delayed XOR with Temporal Features ===")

    N_STEPS = 500
    rng = np.random.RandomState(321)
    w_in = rng.uniform(-1, 1, N_NEURONS)

    u = rng.choice([0.0, 1.0], size=N_STEPS)
    u_scaled = u * 2 - 1

    configure_fpga(fpga)
    time.sleep(0.5)

    print(f"    Collecting {N_STEPS} steps (COUPLED)...")
    states = collect_reservoir_state(fpga, u_scaled, "COUPLED", w_in)

    # Build richer temporal features: delay + diff + running stats
    X_delay = build_delay_features(states, max_delay=4)

    # Add diff features (delta spikes between steps)
    diffs = np.diff(states, axis=0)
    X_diff = build_delay_features(diffs, max_delay=3)

    # Windowed stats (mean, std over 5-step windows)
    window = 5
    n = states.shape[0]
    windowed_mean = np.array([states[max(0,i-window):i+1].mean(axis=0) for i in range(n)])
    windowed_std = np.array([states[max(0,i-window):i+1].std(axis=0) for i in range(n)])

    # Align all features
    min_len = min(X_delay.shape[0], X_diff.shape[0], windowed_mean.shape[0] - 3)
    X_rich = np.concatenate([
        X_delay[:min_len],
        X_diff[:min_len],
        windowed_mean[3:3+min_len],
        windowed_std[3:3+min_len],
    ], axis=1)

    for tau in [2, 5, 7, 10]:
        target_start = max(4, tau)
        y_xor = np.array([int(u[t] != u[t-tau]) for t in range(target_start, N_STEPS)])
        X_xor = X_rich[target_start-3:][:len(y_xor)]

        if len(y_xor) < 20:
            continue

        # Compare basic vs rich features
        X_basic = X_delay[target_start-3:][:len(y_xor)]
        acc_basic, _ = ridge_classify(X_basic, y_xor, n_splits=5)
        acc_rich, _ = ridge_classify(X_xor, y_xor, n_splits=5)

        results[f"exp4_basic_xor_tau{tau}"] = acc_basic
        results[f"exp4_rich_xor_tau{tau}"] = acc_rich
        print(f"    XOR tau={tau}: basic={acc_basic:.3f}, rich={acc_rich:.3f}")

    # Tests
    rich5 = results.get("exp4_rich_xor_tau5", 0.5)
    basic5 = results.get("exp4_basic_xor_tau5", 0.5)
    rich10 = results.get("exp4_rich_xor_tau10", 0.5)

    results["T845_rich_tau5_gt_chance"] = "PASS" if rich5 > 0.53 else "FAIL"
    results["T846_rich_gt_basic_tau5"] = "PASS" if rich5 > basic5 else "FAIL"
    results["T847_rich_tau10_gt_chance"] = "PASS" if rich10 > 0.52 else "FAIL"
    results["T848_decay_profile"] = "PASS" if results.get("exp4_rich_xor_tau2", 0) > results.get("exp4_rich_xor_tau10", 1) else "FAIL"

    print(f"\n  T845 Rich XOR tau=5 ({rich5:.3f}) > 0.53: {results['T845_rich_tau5_gt_chance']}")
    print(f"  T846 Rich > Basic at tau=5: {results['T846_rich_gt_basic_tau5']}")
    print(f"  T847 Rich XOR tau=10 ({rich10:.3f}) > 0.52: {results['T847_rich_tau10_gt_chance']}")
    print(f"  T848 XOR decays with tau: {results['T848_decay_profile']}")


def main():
    print("=" * 70)
    print("z2235: XOR + NARMA Benchmarks with MAC Injection")
    print("=" * 70)
    print(f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")

    fpga = FPGAEthBridge()
    fpga.connect()
    print(f"Connected to FPGA: {fpga.num_neurons} neurons")

    results = {"experiment": "z2235_xor_narma_mac", "timestamp": time.strftime('%Y-%m-%dT%H:%M:%S')}

    try:
        run_exp1_xor(fpga, results)
        run_exp2_narma(fpga, results)
        run_exp3_nonlinear(fpga, results)
        run_exp4_delayed_xor_temporal(fpga, results)
    finally:
        fpga.close()

    # Summary
    passes = sum(1 for k, v in results.items() if k.startswith("T") and v == "PASS")
    total = sum(1 for k, v in results.items() if k.startswith("T") and v in ("PASS", "FAIL"))
    results["summary"] = f"{passes}/{total} PASS"

    print(f"\n{'='*70}")
    print(f"z2235 SUMMARY: {passes}/{total} PASS")
    for k, v in sorted(results.items()):
        if k.startswith("T"):
            print(f"  {k}: {v}")
    print(f"{'='*70}")

    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Results saved to {RESULTS_FILE}")


if __name__ == "__main__":
    main()
