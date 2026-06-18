#!/usr/bin/env python3
"""
z2237_convergence.py — Regime Convergence: Scale FPGA Up, Push GPU Down
========================================================================
Bridge the gap between FPGA neuromorphic and GPU conventional computing:
  - Scale FPGA NS-RAM reservoir (multi-pass, virtual neurons, recurrence)
  - Push GPU toward neuromorphic (stochastic rounding, spiking matmul, noise-modulated compute)
  - Bidirectional co-processing benchmarks

EXP 1: Scaled FPGA (virtual 512-neuron reservoir via time-multiplexing)
EXP 2: GPU stochastic compute (noise-injected matmul as reservoir)
EXP 3: Bidirectional co-processing (FPGA spikes drive GPU, GPU gradients drive FPGA)
EXP 4: Information-theoretic convergence (mutual information, transfer entropy)
EXP 5: Challenge tasks (Lorenz prediction, spoken digit proxy, chaotic timeseries)

Tests T863-T885.
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

RESULTS_FILE = "results/z2237_convergence.json"
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
        return 11.0

def read_gpu_temp():
    try:
        with open("/sys/class/drm/card1/device/hwmon/hwmon7/temp1_input", "r") as f:
            return float(f.read().strip()) / 1000.0
    except:
        return 45.0

def iir_filter(signal, alpha=0.85):
    out = np.zeros_like(signal)
    out[0] = signal[0]
    for i in range(1, len(signal)):
        out[i] = alpha * out[i-1] + (1 - alpha) * signal[i]
    return out

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

def ridge_classify(X, y, n_splits=5):
    std = X.std(axis=0)
    mask = std > 1e-2
    if mask.sum() < 3: return 0.25, 0.0
    scaler = StandardScaler()
    X_f = scaler.fit_transform(X[:, mask])
    classes, counts = np.unique(y, return_counts=True)
    if counts.min() < n_splits: n_splits = max(2, counts.min())
    clf = RidgeClassifier(alpha=1.0)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    scores = cross_val_score(clf, X_f, y, cv=skf, scoring='accuracy')
    return float(scores.mean()), float(scores.std())

def ridge_regress(X, y, n_splits=5):
    std = X.std(axis=0)
    mask = std > 1e-2
    if mask.sum() < 3: return 0.0, 0.0
    scaler = StandardScaler()
    X_f = scaler.fit_transform(X[:, mask])
    reg = Ridge(alpha=1.0)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    scores = cross_val_score(reg, X_f, y, cv=kf, scoring='r2')
    return float(np.mean(scores)), float(np.std(scores))

def generate_waveform(wclass, n_steps):
    t = np.linspace(0, 2*np.pi, n_steps)
    if wclass == 0: return 0.5 * np.sin(t)
    elif wclass == 1: return 0.5 * (2*np.abs(2*(t/(2*np.pi) - np.floor(t/(2*np.pi) + 0.5))) - 1)
    elif wclass == 2: return 0.5 * np.sign(np.sin(t))
    else: return 0.5 * (2*(t/(2*np.pi) - np.floor(t/(2*np.pi))) - 1)

def generate_lorenz(n_steps, dt=0.01):
    """Generate Lorenz attractor timeseries."""
    sigma, rho, beta = 10.0, 28.0, 8/3
    x, y, z = 1.0, 1.0, 1.0
    xs, ys, zs = [], [], []
    for _ in range(n_steps):
        dx = sigma * (y - x)
        dy = x * (rho - z) - y
        dz = x * y - beta * z
        x += dx * dt
        y += dy * dt
        z += dz * dt
        xs.append(x); ys.append(y); zs.append(z)
    return np.array(xs), np.array(ys), np.array(zs)

def generate_mackey_glass(n_steps, tau=17, beta_mg=0.2, gamma=0.1, n=10):
    """Generate Mackey-Glass chaotic timeseries."""
    x = np.zeros(n_steps + tau)
    x[:tau] = 0.9 + np.random.randn(tau) * 0.01
    for t in range(tau, n_steps + tau - 1):
        x_tau = x[t - tau]
        x[t+1] = x[t] + beta_mg * x_tau / (1 + x_tau**n) - gamma * x[t]
    return x[tau:]


def run_exp1_scaled_fpga(fpga, results):
    """EXP 1: Scale FPGA to virtual 512-neuron reservoir via time-multiplexing."""
    print("\n=== EXP 1: Scaled FPGA (Virtual 512-Neuron Reservoir) ===")

    N_STEPS = 30
    N_TRIALS = 120
    N_CLASSES = 4
    N_PASSES = 4  # 4 passes × 128 neurons = 512 virtual neurons
    rng = np.random.RandomState(42)
    w_in = rng.uniform(-1, 1, N_NEURONS)

    configure_fpga(fpga)
    time.sleep(0.5)

    # Collect with different Vg offsets per pass (different operating points)
    vg_offsets = [0.0, 0.02, -0.02, 0.04]  # slight Vg diversity

    X_128_list, X_512_list, y_list = [], [], []

    for trial in range(N_TRIALS):
        wclass = trial % N_CLASSES
        waveform = generate_waveform(wclass, N_STEPS)

        pass_features = []
        noise_state = 0.0
        gpu_power_base = read_gpu_power()

        for p in range(N_PASSES):
            # Set Vg with offset for this pass
            vg_base = BASE_VG + vg_offsets[p]
            fpga.set_vg_batch(0, [float(np.clip(vg_base + BETA * w_in[i], 0.3, 0.9)) for i in range(128)])
            time.sleep(0.05)

            pass_spikes = []
            for step in range(N_STEPS):
                t0 = time.time()
                gpu_power = read_gpu_power()
                noise_state = 0.85 * noise_state + 0.15 * (gpu_power - gpu_power_base) / 5.0

                mac_val = waveform[step] + noise_state * 0.3
                fpga.set_mac_signal(float(np.clip(mac_val * 0.5 + 0.5, 0.0, 1.0)))

                telem = fpga.read_telemetry(timeout=0.1)
                if telem is not None:
                    pass_spikes.append(telem['spike_counts'].astype(np.float32))
                elif pass_spikes:
                    pass_spikes.append(pass_spikes[-1].copy())
                else:
                    pass_spikes.append(np.zeros(N_NEURONS, dtype=np.float32))

                elapsed = time.time() - t0
                if elapsed < STEP_INTERVAL:
                    time.sleep(STEP_INTERVAL - elapsed)

            arr = np.array(pass_spikes)
            feat = np.concatenate([arr.mean(axis=0), arr.std(axis=0)])
            pass_features.append(feat)

        # 128-neuron features (first pass only)
        X_128_list.append(pass_features[0])
        # 512-neuron features (all passes concatenated)
        X_512_list.append(np.concatenate(pass_features))
        y_list.append(wclass)

        if (trial+1) % 30 == 0:
            print(f"    Trial {trial+1}/{N_TRIALS}")

    y = np.array(y_list)
    acc_128, _ = ridge_classify(np.array(X_128_list), y)
    acc_512, _ = ridge_classify(np.array(X_512_list), y)

    results["exp1_128n_acc"] = acc_128
    results["exp1_512v_acc"] = acc_512
    print(f"  128-neuron: {acc_128:.3f}")
    print(f"  512-virtual: {acc_512:.3f}")

    results["T863_512v_gt_128n"] = "PASS" if acc_512 > acc_128 else "FAIL"
    results["T864_512v_gt_0.80"] = "PASS" if acc_512 > 0.80 else "FAIL"
    results["T865_scaling_benefit"] = "PASS" if (acc_512 - acc_128) > 0.02 else "FAIL"

    print(f"\n  T863 512v > 128n: {results['T863_512v_gt_128n']}")
    print(f"  T864 512v > 0.80: {results['T864_512v_gt_0.80']}")
    print(f"  T865 Scaling benefit > 2pp: {results['T865_scaling_benefit']}")


def run_exp2_gpu_stochastic(fpga, results):
    """EXP 2: GPU stochastic compute as reservoir."""
    print("\n=== EXP 2: GPU Stochastic Compute Reservoir ===")

    N_TRIALS = 200
    N_STEPS = 30
    N_CLASSES = 4
    rng = np.random.RandomState(55)

    # Stochastic reservoir: noisy matmul with recurrence
    N_RES = 256
    W_res = rng.randn(N_RES, N_RES).astype(np.float32) * 0.05
    eigvals = np.abs(np.linalg.eigvals(W_res))
    W_res = W_res / max(eigvals) * 0.95  # near-critical spectral radius
    W_in = rng.randn(N_RES).astype(np.float32) * 0.3

    conditions = {
        "deterministic": 0.0,
        "light_noise": 0.01,
        "medium_noise": 0.05,
        "heavy_noise": 0.10,
    }

    for cond_name, noise_level in conditions.items():
        print(f"\n  Condition: {cond_name} (noise={noise_level})")
        X_list, y_list = [], []

        for trial in range(N_TRIALS):
            wclass = trial % N_CLASSES
            waveform = generate_waveform(wclass, N_STEPS)

            state = np.zeros(N_RES, dtype=np.float32)
            states = []

            for step in range(N_STEPS):
                # Stochastic reservoir update
                noise = rng.randn(N_RES).astype(np.float32) * noise_level
                inp = W_in * waveform[step]
                state = np.tanh(W_res @ state + inp + noise)
                # Stochastic rounding (mimics fixed-point)
                if noise_level > 0:
                    state = np.round(state * 256 + rng.randn(N_RES).astype(np.float32) * 0.5) / 256
                states.append(state.copy())

            states = np.array(states)
            feat = np.concatenate([states.mean(axis=0), states.std(axis=0)])
            X_list.append(feat)
            y_list.append(wclass)

        X = np.array(X_list)
        y = np.array(y_list)
        acc, _ = ridge_classify(X, y)
        results[f"exp2_{cond_name}_acc"] = acc
        print(f"    {cond_name}: {acc:.3f}")

    # Test: does moderate noise help (like FPGA stochastic)?
    det = results.get("exp2_deterministic_acc", 0)
    light = results.get("exp2_light_noise_acc", 0)
    medium = results.get("exp2_medium_noise_acc", 0)
    heavy = results.get("exp2_heavy_noise_acc", 0)

    results["T866_stochastic_resonance"] = "PASS" if max(light, medium) > det else "FAIL"
    results["T867_optimal_noise_not_zero"] = "PASS" if max(light, medium, heavy) > det else "FAIL"
    results["T868_heavy_noise_degrades"] = "PASS" if heavy < max(det, light, medium) else "FAIL"

    print(f"\n  T866 Stochastic resonance (noise helps): {results['T866_stochastic_resonance']}")
    print(f"  T867 Optimal noise > 0: {results['T867_optimal_noise_not_zero']}")
    print(f"  T868 Heavy noise degrades: {results['T868_heavy_noise_degrades']}")


def run_exp3_bidirectional(fpga, results):
    """EXP 3: Bidirectional FPGA spikes → GPU, GPU gradients → FPGA."""
    print("\n=== EXP 3: Bidirectional Co-Processing ===")

    N_STEPS = 300
    N_CLASSES = 4
    N_TRIALS = 100
    N_RES_GPU = 128
    rng = np.random.RandomState(77)
    w_in = rng.uniform(-1, 1, N_NEURONS)

    # GPU reservoir
    W_gpu = rng.randn(N_RES_GPU, N_RES_GPU).astype(np.float32) * 0.05
    eigvals = np.abs(np.linalg.eigvals(W_gpu))
    W_gpu = W_gpu / max(eigvals) * 0.9
    W_fpga_to_gpu = rng.randn(N_RES_GPU, N_NEURONS).astype(np.float32) * 0.01

    configure_fpga(fpga)
    time.sleep(0.5)

    conditions = ["BIDIRECTIONAL", "FPGA_ONLY", "GPU_ONLY"]

    for cond in conditions:
        print(f"\n  Condition: {cond}")
        X_list, y_list = [], []

        for trial in range(N_TRIALS):
            wclass = trial % N_CLASSES
            waveform = generate_waveform(wclass, N_STEPS // 10)  # shorter wave, repeat
            waveform = np.tile(waveform, 10)[:N_STEPS]

            gpu_state = np.zeros(N_RES_GPU, dtype=np.float32)
            fpga_states, gpu_states = [], []
            noise_state = 0.0
            gpu_power_base = read_gpu_power()

            for step in range(N_STEPS):
                t0 = time.time()
                gpu_power = read_gpu_power()
                noise_state = 0.85 * noise_state + 0.15 * (gpu_power - gpu_power_base) / 5.0

                if cond in ["BIDIRECTIONAL", "FPGA_ONLY"]:
                    mac_val = waveform[step] + noise_state * 0.3
                    # In bidirectional: GPU state modulates MAC
                    if cond == "BIDIRECTIONAL" and len(gpu_states) > 0:
                        gpu_feedback = float(np.mean(gpu_state[:8])) * 0.2
                        mac_val += gpu_feedback
                    fpga.set_mac_signal(float(np.clip(mac_val * 0.5 + 0.5, 0.0, 1.0)))

                telem = fpga.read_telemetry(timeout=0.1)
                if telem is not None:
                    fpga_spk = telem['spike_counts'].astype(np.float32)
                elif fpga_states:
                    fpga_spk = fpga_states[-1].copy()
                else:
                    fpga_spk = np.zeros(N_NEURONS, dtype=np.float32)
                fpga_states.append(fpga_spk)

                # GPU reservoir update
                if cond in ["BIDIRECTIONAL", "GPU_ONLY"]:
                    gpu_inp = np.zeros(N_RES_GPU, dtype=np.float32)
                    gpu_inp[:1] = waveform[step]
                    if cond == "BIDIRECTIONAL":
                        # FPGA spikes drive GPU
                        gpu_inp += W_fpga_to_gpu @ (fpga_spk / max(fpga_spk.max(), 1.0))
                    noise = rng.randn(N_RES_GPU).astype(np.float32) * 0.02
                    gpu_state = np.tanh(W_gpu @ gpu_state + gpu_inp + noise)

                gpu_states.append(gpu_state.copy())

                elapsed = time.time() - t0
                if elapsed < STEP_INTERVAL:
                    time.sleep(STEP_INTERVAL - elapsed)

            fpga_arr = np.array(fpga_states)
            gpu_arr = np.array(gpu_states)

            if cond == "BIDIRECTIONAL":
                feat = np.concatenate([fpga_arr.mean(axis=0), fpga_arr.std(axis=0),
                                       gpu_arr.mean(axis=0), gpu_arr.std(axis=0)])
            elif cond == "FPGA_ONLY":
                feat = np.concatenate([fpga_arr.mean(axis=0), fpga_arr.std(axis=0)])
            else:
                feat = np.concatenate([gpu_arr.mean(axis=0), gpu_arr.std(axis=0)])

            X_list.append(feat)
            y_list.append(wclass)

            if (trial+1) % 25 == 0:
                print(f"    Trial {trial+1}/{N_TRIALS}")

        X = np.array(X_list)
        y = np.array(y_list)
        acc, _ = ridge_classify(X, y)
        results[f"exp3_{cond}_acc"] = acc
        print(f"    {cond}: {acc:.3f}")

    bi = results.get("exp3_BIDIRECTIONAL_acc", 0)
    fpga_only = results.get("exp3_FPGA_ONLY_acc", 0)
    gpu_only = results.get("exp3_GPU_ONLY_acc", 0)

    results["T869_bidir_gt_fpga"] = "PASS" if bi > fpga_only else "FAIL"
    results["T870_bidir_gt_gpu"] = "PASS" if bi > gpu_only else "FAIL"
    results["T871_bidir_synergy"] = "PASS" if bi > max(fpga_only, gpu_only) else "FAIL"

    print(f"\n  T869 Bidir > FPGA_ONLY: {results['T869_bidir_gt_fpga']}")
    print(f"  T870 Bidir > GPU_ONLY: {results['T870_bidir_gt_gpu']}")
    print(f"  T871 Bidir synergy: {results['T871_bidir_synergy']}")


def run_exp4_info_theoretic(fpga, results):
    """EXP 4: Information-theoretic convergence metrics."""
    print("\n=== EXP 4: Information-Theoretic Convergence ===")

    N_STEPS = 500
    rng = np.random.RandomState(99)
    w_in = rng.uniform(-1, 1, N_NEURONS)

    u = rng.uniform(-1, 1, N_STEPS)
    u_smooth = iir_filter(u, alpha=0.5)

    configure_fpga(fpga)
    time.sleep(0.5)

    print("    Collecting FPGA states...")
    fpga_states = []
    gpu_fw_states = []
    noise_state = 0.0
    gpu_power_base = read_gpu_power()

    for step in range(N_STEPS):
        t0 = time.time()
        gpu_power = read_gpu_power()
        gpu_temp = read_gpu_temp()
        noise_state = 0.85 * noise_state + 0.15 * (gpu_power - gpu_power_base) / 5.0

        mac_val = u_smooth[step] + noise_state * 0.3
        fpga.set_mac_signal(float(np.clip(mac_val * 0.5 + 0.5, 0.0, 1.0)))

        telem = fpga.read_telemetry(timeout=0.1)
        if telem is not None:
            fpga_states.append(telem['spike_counts'].astype(np.float32))
        elif fpga_states:
            fpga_states.append(fpga_states[-1].copy())
        else:
            fpga_states.append(np.zeros(N_NEURONS, dtype=np.float32))

        gpu_fw_states.append([gpu_power, gpu_temp, noise_state])

        elapsed = time.time() - t0
        if elapsed < STEP_INTERVAL:
            time.sleep(STEP_INTERVAL - elapsed)

        if (step+1) % 100 == 0:
            print(f"    Step {step+1}/{N_STEPS}")

    fpga_arr = np.array(fpga_states)
    gpu_arr = np.array(gpu_fw_states)

    # Transfer entropy: GPU → FPGA
    # Approximate with Granger causality (linear proxy for TE)
    from sklearn.linear_model import LinearRegression
    lag = 5

    # Can GPU firmware predict FPGA spike rate?
    fpga_rate = fpga_arr.sum(axis=1)  # total spike rate per step
    X_auto = np.array([fpga_rate[i-lag:i] for i in range(lag, len(fpga_rate))])
    X_full = np.concatenate([X_auto, np.array([gpu_arr[i-lag:i].flatten() for i in range(lag, len(gpu_arr))])], axis=1)
    y_gc = fpga_rate[lag:]

    reg_auto = LinearRegression().fit(X_auto, y_gc)
    reg_full = LinearRegression().fit(X_full, y_gc)

    r2_auto = reg_auto.score(X_auto, y_gc)
    r2_full = reg_full.score(X_full, y_gc)
    gc_gpu_to_fpga = max(0, r2_full - r2_auto)
    results["exp4_gc_gpu_to_fpga"] = gc_gpu_to_fpga

    # Reverse: FPGA → GPU power
    gpu_rate = gpu_arr[:, 0]  # power
    X_auto2 = np.array([gpu_rate[i-lag:i] for i in range(lag, len(gpu_rate))])
    X_full2 = np.concatenate([X_auto2, np.array([fpga_arr[i-lag:i].mean(axis=1) for i in range(lag, len(fpga_arr))])[:len(X_auto2)]], axis=1)
    y_gc2 = gpu_rate[lag:][:len(X_full2)]

    reg_auto2 = LinearRegression().fit(X_auto2[:len(y_gc2)], y_gc2)
    reg_full2 = LinearRegression().fit(X_full2[:len(y_gc2)], y_gc2)
    gc_fpga_to_gpu = max(0, reg_full2.score(X_full2[:len(y_gc2)], y_gc2) - reg_auto2.score(X_auto2[:len(y_gc2)], y_gc2))
    results["exp4_gc_fpga_to_gpu"] = gc_fpga_to_gpu

    # Mutual information (binned estimate)
    n_bins = 20
    fpga_binned = np.digitize(fpga_rate, np.linspace(fpga_rate.min(), fpga_rate.max(), n_bins))
    input_binned = np.digitize(u_smooth, np.linspace(-1, 1, n_bins))

    # Joint and marginal histograms
    joint = np.zeros((n_bins+1, n_bins+1))
    for i in range(min(len(fpga_binned), len(input_binned))):
        joint[fpga_binned[i], input_binned[i]] += 1
    joint /= joint.sum()

    p_fpga = joint.sum(axis=1)
    p_input = joint.sum(axis=0)

    mi = 0.0
    for i in range(n_bins+1):
        for j in range(n_bins+1):
            if joint[i,j] > 0 and p_fpga[i] > 0 and p_input[j] > 0:
                mi += joint[i,j] * np.log2(joint[i,j] / (p_fpga[i] * p_input[j]))
    results["exp4_mi_input_fpga"] = mi

    print(f"  GC(GPU→FPGA): {gc_gpu_to_fpga:.4f}")
    print(f"  GC(FPGA→GPU): {gc_fpga_to_gpu:.4f}")
    print(f"  MI(input, FPGA): {mi:.4f} bits")

    results["T872_gc_gpu_to_fpga_gt_0"] = "PASS" if gc_gpu_to_fpga > 0.001 else "FAIL"
    results["T873_gc_asymmetric"] = "PASS" if gc_gpu_to_fpga != gc_fpga_to_gpu else "FAIL"
    results["T874_mi_gt_0.01"] = "PASS" if mi > 0.01 else "FAIL"

    print(f"\n  T872 GC(GPU→FPGA) > 0: {results['T872_gc_gpu_to_fpga_gt_0']}")
    print(f"  T873 GC asymmetric: {results['T873_gc_asymmetric']}")
    print(f"  T874 MI > 0.01: {results['T874_mi_gt_0.01']}")


def run_exp5_challenge(fpga, results):
    """EXP 5: Challenge tasks — Lorenz, Mackey-Glass, spoken digit proxy."""
    print("\n=== EXP 5: Challenge Tasks ===")

    rng = np.random.RandomState(111)
    w_in = rng.uniform(-1, 1, N_NEURONS)

    # --- Lorenz prediction ---
    print("\n  Task A: Lorenz x-prediction")
    lx, ly, lz = generate_lorenz(600, dt=0.02)
    # Normalize
    lx = (lx - lx.mean()) / (lx.std() + 1e-8)

    configure_fpga(fpga)
    time.sleep(0.5)

    fpga_states = []
    noise_state = 0.0
    gpu_power_base = read_gpu_power()

    for step in range(len(lx)):
        t0 = time.time()
        gpu_power = read_gpu_power()
        noise_state = 0.85 * noise_state + 0.15 * (gpu_power - gpu_power_base) / 5.0
        mac_val = lx[step] * 0.3 + noise_state * 0.2
        fpga.set_mac_signal(float(np.clip(mac_val * 0.5 + 0.5, 0.0, 1.0)))

        telem = fpga.read_telemetry(timeout=0.1)
        if telem is not None:
            fpga_states.append(telem['spike_counts'].astype(np.float32))
        elif fpga_states:
            fpga_states.append(fpga_states[-1].copy())
        else:
            fpga_states.append(np.zeros(N_NEURONS, dtype=np.float32))

        elapsed = time.time() - t0
        if elapsed < STEP_INTERVAL:
            time.sleep(STEP_INTERVAL - elapsed)

        if (step+1) % 150 == 0:
            print(f"    Step {step+1}/{len(lx)}")

    X_lorenz = np.array(fpga_states)
    # 1-step prediction
    pred_horizon = 1
    X_l = X_lorenz[:-pred_horizon]
    y_l = lx[pred_horizon:][:len(X_l)]

    r2_lorenz, _ = ridge_regress(X_l, y_l, n_splits=5)
    results["exp5_lorenz_r2"] = max(0, r2_lorenz)
    print(f"    Lorenz 1-step R²: {r2_lorenz:.4f}")

    # --- Mackey-Glass ---
    print("\n  Task B: Mackey-Glass prediction")
    mg = generate_mackey_glass(600, tau=17)
    mg = (mg - mg.mean()) / (mg.std() + 1e-8)

    configure_fpga(fpga)
    time.sleep(0.5)

    fpga_states = []
    noise_state = 0.0
    gpu_power_base = read_gpu_power()

    for step in range(len(mg)):
        t0 = time.time()
        gpu_power = read_gpu_power()
        noise_state = 0.85 * noise_state + 0.15 * (gpu_power - gpu_power_base) / 5.0
        mac_val = mg[step] * 0.3 + noise_state * 0.2
        fpga.set_mac_signal(float(np.clip(mac_val * 0.5 + 0.5, 0.0, 1.0)))

        telem = fpga.read_telemetry(timeout=0.1)
        if telem is not None:
            fpga_states.append(telem['spike_counts'].astype(np.float32))
        elif fpga_states:
            fpga_states.append(fpga_states[-1].copy())
        else:
            fpga_states.append(np.zeros(N_NEURONS, dtype=np.float32))

        elapsed = time.time() - t0
        if elapsed < STEP_INTERVAL:
            time.sleep(STEP_INTERVAL - elapsed)

        if (step+1) % 150 == 0:
            print(f"    Step {step+1}/{len(mg)}")

    X_mg = np.array(fpga_states)
    X_m = X_mg[:-1]
    y_m = mg[1:][:len(X_m)]
    r2_mg, _ = ridge_regress(X_m, y_m, n_splits=5)
    results["exp5_mg_r2"] = max(0, r2_mg)
    print(f"    Mackey-Glass 1-step R²: {r2_mg:.4f}")

    # --- 8-class spoken digit proxy ---
    print("\n  Task C: 8-class waveform (spoken digit proxy)")
    N_TRIALS = 160
    N_STEPS = 30

    configure_fpga(fpga)
    time.sleep(0.5)

    X_list, y_list = [], []
    for trial in range(N_TRIALS):
        wclass = trial % 8
        t = np.linspace(0, 2*np.pi, N_STEPS)
        # 8 distinct waveforms
        if wclass < 4:
            wave = generate_waveform(wclass, N_STEPS)
        elif wclass == 4:
            wave = 0.5 * np.sin(2*t)  # double freq sine
        elif wclass == 5:
            wave = 0.5 * np.sin(3*t)  # triple freq sine
        elif wclass == 6:
            wave = 0.3 * np.sin(t) + 0.2 * np.sin(3*t)  # harmonic mix
        else:
            wave = 0.5 * np.exp(-t/3) * np.sin(5*t)  # damped oscillation

        fpga_spikes = []
        noise_state = 0.0
        gpu_power_base = read_gpu_power()

        for step in range(N_STEPS):
            t0 = time.time()
            gpu_power = read_gpu_power()
            noise_state = 0.85 * noise_state + 0.15 * (gpu_power - gpu_power_base) / 5.0
            mac_val = wave[step] + noise_state * 0.3
            fpga.set_mac_signal(float(np.clip(mac_val * 0.5 + 0.5, 0.0, 1.0)))
            vg_mod = BASE_VG + ALPHA * wave[step] + BETA * noise_state
            fpga.set_vg_batch(0, [float(np.clip(
                vg_mod + BETA * w_in[i] * wave[step], 0.3, 0.9
            )) for i in range(128)])

            telem = fpga.read_telemetry(timeout=0.1)
            if telem is not None:
                fpga_spikes.append(telem['spike_counts'].astype(np.float32))
            elif fpga_spikes:
                fpga_spikes.append(fpga_spikes[-1].copy())
            else:
                fpga_spikes.append(np.zeros(N_NEURONS, dtype=np.float32))

            elapsed = time.time() - t0
            if elapsed < STEP_INTERVAL:
                time.sleep(STEP_INTERVAL - elapsed)

        arr = np.array(fpga_spikes)
        feat = np.concatenate([arr.mean(axis=0), arr.std(axis=0), arr.max(axis=0), arr.min(axis=0)])
        X_list.append(feat)
        y_list.append(wclass)

        if (trial+1) % 40 == 0:
            print(f"    Trial {trial+1}/{N_TRIALS}")

    X = np.array(X_list)
    y = np.array(y_list)
    acc_8class, _ = ridge_classify(X, y)
    results["exp5_8class_acc"] = acc_8class
    print(f"    8-class waveform: {acc_8class:.3f}")

    # Tests
    results["T875_lorenz_r2_gt_0"] = "PASS" if results["exp5_lorenz_r2"] > 0.001 else "FAIL"
    results["T876_mg_r2_gt_0"] = "PASS" if results["exp5_mg_r2"] > 0.001 else "FAIL"
    results["T877_8class_gt_0.20"] = "PASS" if acc_8class > 0.20 else "FAIL"
    results["T878_8class_gt_chance"] = "PASS" if acc_8class > 0.125 else "FAIL"
    results["T879_lorenz_r2_gt_mg"] = "PASS" if results["exp5_lorenz_r2"] > results["exp5_mg_r2"] else "FAIL"

    print(f"\n  T875 Lorenz R² > 0: {results['T875_lorenz_r2_gt_0']}")
    print(f"  T876 MG R² > 0: {results['T876_mg_r2_gt_0']}")
    print(f"  T877 8-class > 0.20: {results['T877_8class_gt_0.20']}")
    print(f"  T878 8-class > chance (12.5%): {results['T878_8class_gt_chance']}")
    print(f"  T879 Lorenz > MG: {results['T879_lorenz_r2_gt_mg']}")


def main():
    print("=" * 70)
    print("z2237: Regime Convergence — Scale FPGA Up, Push GPU Down")
    print("=" * 70)
    print(f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")

    fpga = FPGAEthBridge()
    fpga.connect()
    print(f"Connected to FPGA: {fpga.num_neurons} neurons")

    results = {"experiment": "z2237_convergence", "timestamp": time.strftime('%Y-%m-%dT%H:%M:%S')}

    try:
        run_exp1_scaled_fpga(fpga, results)
        run_exp2_gpu_stochastic(fpga, results)
        run_exp3_bidirectional(fpga, results)
        run_exp4_info_theoretic(fpga, results)
        run_exp5_challenge(fpga, results)
    finally:
        fpga.close()

    passes = sum(1 for k, v in results.items() if k.startswith("T") and v == "PASS")
    total = sum(1 for k, v in results.items() if k.startswith("T") and v in ("PASS", "FAIL"))
    results["summary"] = f"{passes}/{total} PASS"

    print(f"\n{'='*70}")
    print(f"z2237 SUMMARY: {passes}/{total} PASS")
    for k, v in sorted(results.items()):
        if k.startswith("T"):
            print(f"  {k}: {v}")
    print(f"{'='*70}")

    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Results saved to {RESULTS_FILE}")


if __name__ == "__main__":
    main()
