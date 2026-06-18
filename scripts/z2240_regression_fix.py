#!/usr/bin/env python3
"""z2240: Fix FPGA regression (NARMA, Memory Capacity, Streaming XOR).

Root cause from z2239: Classification hits 94% but ALL regression R²=0.
Hypothesis: per-step spike counts are low integers (3-30), vmem might be
near-constant across neurons. Need to understand what features actually
carry input information and build regression on those.

Approach:
1. DIAG: Characterize spike count and vmem distributions under varying input
2. EXP 1: NARMA-5 with vmem-only features + sliding window
3. EXP 2: NARMA-5 with echo state approach (recurrent readout)
4. EXP 3: Memory capacity with vmem features
5. EXP 4: Linear/nonlinear separation capacity
6. EXP 5: Streaming XOR with vmem + windowed features
7. EXP 6: Full hybrid regression (GPU LIF states + FPGA vmem)

Tests T960-T985
"""

import sys, os, time, json
import numpy as np
from datetime import datetime
from sklearn.linear_model import Ridge, RidgeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.fpga_host_eth import FPGAEthBridge

# --- Parameters (tuned from z2239) ---
N_NEURONS = 128
BASE_VG = 0.58
ALPHA = 0.25
BETA = 0.08
SAMPLE_HZ = 20
STEP_INTERVAL = 1.0 / SAMPLE_HZ
TUNED_LEAK = 0x0011
TUNED_THRESH_F = 0.50
TUNED_BIAS_GAIN_F = 0.03125


def read_gpu_power():
    try:
        with open('/sys/class/hwmon/hwmon7/power1_average', 'r') as f:
            return float(f.read().strip()) / 1e6
    except:
        return 11.0


def configure_fpga(fpga):
    """Set tuned parameters + heterogeneous Vg."""
    fpga.set_leak_cond(TUNED_LEAK)
    fpga.set_threshold(TUNED_THRESH_F)
    fpga.set_bias_gain(TUNED_BIAS_GAIN_F)
    vg_base = np.array([BASE_VG + 0.15 * (i/127 - 0.5) for i in range(N_NEURONS)])
    fpga.set_vg_batch(0, [float(v) for v in vg_base])
    return vg_base


def collect_step(fpga, inp, condition, w_in, vg_base, noise_state, gpu_power_base):
    """Single step: set input, read telemetry, return (spikes, vmem, noise_state)."""
    gpu_power = read_gpu_power()
    power_noise = (gpu_power - gpu_power_base) / 5.0
    noise_state = 0.85 * noise_state + 0.15 * power_noise

    if condition == "COUPLED":
        mac_val = inp + noise_state * 0.3
        fpga.set_mac_signal(float(np.clip(mac_val * 0.5 + 0.5, 0.0, 1.0)))
        vg_mod = vg_base + ALPHA * inp + BETA * w_in * inp + 0.05 * noise_state
    elif condition == "FPGA_ONLY":
        mac_val = inp * 0.5 + 0.5
        fpga.set_mac_signal(float(np.clip(mac_val, 0.0, 1.0)))
        vg_mod = vg_base + ALPHA * inp
    else:  # STATIC
        fpga.set_mac_signal(0.5)
        vg_mod = vg_base.copy()

    fpga.set_vg_batch(0, [float(np.clip(v, 0.3, 0.9)) for v in vg_mod])

    telem = fpga.read_telemetry(timeout=0.15)
    if telem is not None:
        sc = telem['spike_counts'].astype(np.float32)
        vm = telem['vmem'].copy()
    else:
        sc = np.zeros(N_NEURONS, dtype=np.float32)
        vm = np.zeros(N_NEURONS, dtype=np.float32)

    return sc, vm, noise_state


def collect_sequence(fpga, input_signal, condition, w_in, vg_base):
    """Collect spike counts and vmem for a sequence of inputs."""
    all_spikes = []
    all_vmem = []
    gpu_power_base = read_gpu_power()
    noise_state = 0.0

    for step in range(len(input_signal)):
        t0 = time.time()
        inp = float(input_signal[step])
        sc, vm, noise_state = collect_step(
            fpga, inp, condition, w_in, vg_base, noise_state, gpu_power_base)
        all_spikes.append(sc)
        all_vmem.append(vm)

        elapsed = time.time() - t0
        if elapsed < STEP_INTERVAL:
            time.sleep(STEP_INTERVAL - elapsed)

    return np.array(all_spikes), np.array(all_vmem)


def ridge_regress(X, y, n_splits=5, alpha=1.0):
    """Ridge regression with R² scoring, returns (mean_r2, std_r2, raw_scores)."""
    std = X.std(axis=0)
    mask = std > 1e-6  # looser threshold
    if mask.sum() < 3:
        return 0.0, 0.0, np.array([0.0])
    scaler = StandardScaler()
    X_f = scaler.fit_transform(X[:, mask])
    reg = Ridge(alpha=alpha)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    scores = cross_val_score(reg, X_f, y, cv=kf, scoring='r2')
    return float(np.mean(scores)), float(np.std(scores)), scores


def generate_narma(u, order=5):
    """Generate NARMA-order target from input u."""
    y = np.zeros(len(u))
    for t in range(order, len(u)):
        s = sum(y[t-j-1] for j in range(order))
        y[t] = (0.3 * y[t-1] + 0.05 * y[t-1] * s
                + 1.5 * u[t-order] * u[t] + 0.1)
        y[t] = np.clip(y[t], -5, 5)
    return y


# ==========================================================================
# DIAGNOSTIC: What do spike counts and vmem actually look like?
# ==========================================================================
def run_diagnostic(fpga, results):
    """Characterize FPGA output under varying input."""
    print("\n=== DIAGNOSTIC: Feature Characterization ===")
    rng = np.random.RandomState(42)
    w_in = rng.uniform(-1, 1, N_NEURONS)
    vg_base = configure_fpga(fpga)
    time.sleep(1.0)

    # Collect 100 steps with sinusoidal input
    input_signal = np.sin(2 * np.pi * np.arange(100) / 20) * 0.5
    spikes, vmem = collect_sequence(fpga, input_signal, "COUPLED", w_in, vg_base)

    # Spike statistics
    spike_mean = spikes.mean()
    spike_std = spikes.std()
    spike_range = spikes.max() - spikes.min()
    spike_unique = len(np.unique(spikes.ravel()))

    # Vmem statistics
    vmem_mean = vmem.mean()
    vmem_std = vmem.std()
    vmem_range = vmem.max() - vmem.min()

    # Per-neuron temporal correlation with input
    input_corr_spikes = []
    input_corr_vmem = []
    for n in range(N_NEURONS):
        if spikes[:, n].std() > 0:
            c = np.corrcoef(input_signal, spikes[:, n])[0, 1]
            input_corr_spikes.append(abs(c) if not np.isnan(c) else 0)
        if vmem[:, n].std() > 0:
            c = np.corrcoef(input_signal, vmem[:, n])[0, 1]
            input_corr_vmem.append(abs(c) if not np.isnan(c) else 0)

    mean_corr_spikes = np.mean(input_corr_spikes) if input_corr_spikes else 0
    mean_corr_vmem = np.mean(input_corr_vmem) if input_corr_vmem else 0

    # Effective dimensionality
    for name, data in [("spikes", spikes), ("vmem", vmem)]:
        std_per_col = data.std(axis=0)
        active = (std_per_col > 1e-6).sum()
        if active > 1:
            from sklearn.decomposition import PCA
            pca = PCA(n_components=min(active, 50))
            pca.fit(data[:, std_per_col > 1e-6])
            cumvar = np.cumsum(pca.explained_variance_ratio_)
            eff_dim = np.searchsorted(cumvar, 0.90) + 1
        else:
            eff_dim = active
        results[f"diag_{name}_eff_dim"] = int(eff_dim)
        print(f"  {name}: mean={data.mean():.3f}, std={data.std():.4f}, "
              f"range={data.max()-data.min():.3f}, eff_dim(90%)={eff_dim}")

    print(f"  spike_unique_values: {spike_unique}")
    print(f"  mean |corr(input, spikes)|: {mean_corr_spikes:.4f}")
    print(f"  mean |corr(input, vmem)|:   {mean_corr_vmem:.4f}")

    results["diag_spike_mean"] = float(spike_mean)
    results["diag_spike_std"] = float(spike_std)
    results["diag_spike_unique"] = int(spike_unique)
    results["diag_vmem_mean"] = float(vmem_mean)
    results["diag_vmem_std"] = float(vmem_std)
    results["diag_input_corr_spikes"] = float(mean_corr_spikes)
    results["diag_input_corr_vmem"] = float(mean_corr_vmem)

    # Quick regression test: can we predict input from features?
    # Using vmem only
    X_vm = vmem
    y_inp = input_signal
    r2_vm, _, _ = ridge_regress(X_vm, y_inp)
    # Using spikes only
    X_sp = spikes
    r2_sp, _, _ = ridge_regress(X_sp, y_inp)
    # Using both
    X_both = np.hstack([spikes, vmem])
    r2_both, _, _ = ridge_regress(X_both, y_inp)

    print(f"  input prediction R² — vmem: {r2_vm:.4f}, spikes: {r2_sp:.4f}, both: {r2_both:.4f}")
    results["diag_input_r2_vmem"] = float(r2_vm)
    results["diag_input_r2_spikes"] = float(r2_sp)
    results["diag_input_r2_both"] = float(r2_both)

    # Can we predict input[t-1] from features[t]? (memory test)
    X_t = np.hstack([spikes[1:], vmem[1:]])
    y_prev = input_signal[:-1]
    r2_mem, _, _ = ridge_regress(X_t, y_prev)
    print(f"  memory prediction R² (input[t-1] from features[t]): {r2_mem:.4f}")
    results["diag_memory_r2"] = float(r2_mem)


# ==========================================================================
# EXP 1: NARMA-5 with vmem-centric features + sliding window
# ==========================================================================
def run_exp1_narma_vmem(fpga, results):
    """NARMA-5 using vmem as primary feature with windowed regression."""
    print("\n=== EXP 1: NARMA-5 (vmem-centric) ===")

    rng = np.random.RandomState(42)
    w_in = rng.uniform(-1, 1, N_NEURONS)
    N_STEPS = 500
    WINDOW_SIZES = [1, 3, 5, 10]

    for cond in ["COUPLED", "FPGA_ONLY"]:
        print(f"\n  Condition: {cond}")
        vg_base = configure_fpga(fpga)
        time.sleep(0.5)

        # Generate input and NARMA target
        u = rng.uniform(0, 0.5, N_STEPS + 5)
        y_narma = generate_narma(u, order=5)
        input_signal = u[5:]
        target = y_narma[5:]

        # Collect FPGA response
        spikes, vmem = collect_sequence(fpga, input_signal, cond, w_in, vg_base)

        # Try different feature strategies
        for wsize in WINDOW_SIZES:
            features = []
            targets = []
            for t in range(wsize, len(input_signal)):
                # Window of vmem values (primary)
                vm_window = vmem[t-wsize+1:t+1].ravel()  # wsize × 128
                # Add spike delta
                sp_delta = spikes[t] - spikes[max(0, t-1)]
                # Add input history (reservoir should learn to represent this)
                feat = np.concatenate([vm_window, sp_delta])
                features.append(feat)
                targets.append(target[t])

            X = np.array(features)
            y = np.array(targets)
            r2, r2_std, raw = ridge_regress(X, y, alpha=10.0)
            key = f"exp1_{cond}_narma5_w{wsize}"
            results[key] = float(r2)
            raw_mean = float(np.mean(raw))
            print(f"    w={wsize}: R² = {r2:.4f} ± {r2_std:.4f} (raw_mean={raw_mean:.4f})")

    # Best across conditions and windows
    best_r2 = 0
    for k, v in results.items():
        if k.startswith("exp1_") and "narma5" in k and isinstance(v, (int, float)):
            best_r2 = max(best_r2, v)

    results["T960_narma5_vmem_any_gt_0"] = "PASS" if best_r2 > 0 else "FAIL"
    results["T961_narma5_vmem_any_gt_005"] = "PASS" if best_r2 > 0.05 else "FAIL"
    results["T962_narma5_vmem_any_gt_010"] = "PASS" if best_r2 > 0.10 else "FAIL"
    for tid, name, val in [
        ("T960", f"any_narma5_vmem({best_r2:.4f})>0", best_r2 > 0),
        ("T961", f"any_narma5_vmem({best_r2:.4f})>0.05", best_r2 > 0.05),
        ("T962", f"any_narma5_vmem({best_r2:.4f})>0.10", best_r2 > 0.10),
    ]:
        print(f"  {tid} {name}: {'PASS' if val else 'FAIL'}")


# ==========================================================================
# EXP 2: NARMA-5 echo state approach (train readout on reservoir states)
# ==========================================================================
def run_exp2_esn_narma(fpga, results):
    """NARMA-5 with ESN-style approach: accumulate reservoir state matrix."""
    print("\n=== EXP 2: NARMA-5 (ESN-style) ===")

    rng = np.random.RandomState(123)
    w_in = rng.uniform(-1, 1, N_NEURONS)
    N_STEPS = 800
    WASHOUT = 50

    for cond in ["COUPLED", "FPGA_ONLY"]:
        print(f"\n  Condition: {cond}")
        vg_base = configure_fpga(fpga)
        time.sleep(0.5)

        u = rng.uniform(0, 0.5, N_STEPS + 5)
        y_narma = generate_narma(u, order=5)
        input_signal = u[5:]
        target = y_narma[5:]

        spikes, vmem = collect_sequence(fpga, input_signal, cond, w_in, vg_base)

        # ESN-style: use vmem as state vector, add bias and input
        # State matrix: [vmem(t), vmem(t)², input(t), 1]
        states = []
        for t in range(len(input_signal)):
            vm = vmem[t]
            inp_t = input_signal[t]
            state = np.concatenate([
                vm,                    # 128: membrane voltages
                vm ** 2,               # 128: nonlinear transform
                spikes[t],             # 128: spike counts
                [inp_t, inp_t**2, 1.0] # 3: input + bias
            ])
            states.append(state)

        X = np.array(states)[WASHOUT:]
        y = target[WASHOUT:]

        # Ridge regression with multiple alphas
        best_r2 = -999
        best_alpha = 1.0
        for alpha in [0.01, 0.1, 1.0, 10.0, 100.0]:
            r2, r2_std, raw = ridge_regress(X, y, alpha=alpha)
            if r2 > best_r2:
                best_r2 = r2
                best_alpha = alpha

        results[f"exp2_{cond}_narma5_r2"] = float(best_r2)
        results[f"exp2_{cond}_best_alpha"] = float(best_alpha)
        print(f"    Best R² = {best_r2:.4f} (alpha={best_alpha})")

    c_r2 = results.get("exp2_COUPLED_narma5_r2", 0)
    f_r2 = results.get("exp2_FPGA_ONLY_narma5_r2", 0)
    results["T963_esn_narma5_coupled_gt_0"] = "PASS" if c_r2 > 0 else "FAIL"
    results["T964_esn_narma5_any_gt_005"] = "PASS" if max(c_r2, f_r2) > 0.05 else "FAIL"
    results["T965_esn_narma5_any_gt_010"] = "PASS" if max(c_r2, f_r2) > 0.10 else "FAIL"
    for tid, name, val in [
        ("T963", f"COUPLED({c_r2:.4f})>0", c_r2 > 0),
        ("T964", f"any({max(c_r2, f_r2):.4f})>0.05", max(c_r2, f_r2) > 0.05),
        ("T965", f"any({max(c_r2, f_r2):.4f})>0.10", max(c_r2, f_r2) > 0.10),
    ]:
        print(f"  {tid} {name}: {'PASS' if val else 'FAIL'}")


# ==========================================================================
# EXP 3: Memory Capacity with vmem
# ==========================================================================
def run_exp3_memory_capacity(fpga, results):
    """Memory capacity using vmem features."""
    print("\n=== EXP 3: Memory Capacity (vmem) ===")

    rng = np.random.RandomState(77)
    w_in = rng.uniform(-1, 1, N_NEURONS)
    N_STEPS = 600
    MAX_DELAY = 10
    WASHOUT = 20

    for cond in ["COUPLED", "FPGA_ONLY", "STATIC"]:
        print(f"\n  Condition: {cond}")
        vg_base = configure_fpga(fpga)
        time.sleep(0.5)

        # IID uniform input
        input_signal = rng.uniform(-1, 1, N_STEPS)
        spikes, vmem = collect_sequence(fpga, input_signal, cond, w_in, vg_base)

        mc_total = 0.0
        for delay in range(1, MAX_DELAY + 1):
            X = vmem[WASHOUT + delay:]
            y = input_signal[WASHOUT: -delay]
            n = min(len(X), len(y))
            X, y = X[:n], y[:n]

            r2, _, _ = ridge_regress(X, y, alpha=1.0)
            results[f"exp3_{cond}_mc_d{delay}"] = float(r2)
            mc_total += max(0, r2)
            if delay <= 5:
                print(f"    d={delay}: R² = {r2:.4f}")

        results[f"exp3_{cond}_mc_total"] = float(mc_total)
        print(f"    MC total (d=1..{MAX_DELAY}): {mc_total:.4f}")

    c_mc = results.get("exp3_COUPLED_mc_total", 0)
    f_mc = results.get("exp3_FPGA_ONLY_mc_total", 0)
    s_mc = results.get("exp3_STATIC_mc_total", 0)
    c_d1 = results.get("exp3_COUPLED_mc_d1", 0)

    results["T966_mc_d1_vmem_gt_0"] = "PASS" if c_d1 > 0 else "FAIL"
    results["T967_mc_total_coupled_gt_01"] = "PASS" if c_mc > 0.1 else "FAIL"
    results["T968_mc_coupled_gt_static"] = "PASS" if c_mc > s_mc else "FAIL"
    results["T969_mc_any_gt_05"] = "PASS" if max(c_mc, f_mc) > 0.5 else "FAIL"
    for tid, name, val in [
        ("T966", f"MC(d=1)({c_d1:.4f})>0", c_d1 > 0),
        ("T967", f"MC_total_COUPLED({c_mc:.4f})>0.1", c_mc > 0.1),
        ("T968", f"COUPLED({c_mc:.4f})>STATIC({s_mc:.4f})", c_mc > s_mc),
        ("T969", f"any({max(c_mc, f_mc):.4f})>0.5", max(c_mc, f_mc) > 0.5),
    ]:
        print(f"  {tid} {name}: {'PASS' if val else 'FAIL'}")


# ==========================================================================
# EXP 4: Nonlinear Separation Capacity
# ==========================================================================
def run_exp4_separation(fpga, results):
    """Test if reservoir provides nonlinear separation of inputs."""
    print("\n=== EXP 4: Nonlinear Separation ===")

    rng = np.random.RandomState(99)
    w_in = rng.uniform(-1, 1, N_NEURONS)
    N_STEPS = 500
    WASHOUT = 20

    vg_base = configure_fpga(fpga)
    time.sleep(0.5)

    # Uniform input
    input_signal = rng.uniform(-1, 1, N_STEPS)
    spikes, vmem = collect_sequence(fpga, input_signal, "COUPLED", w_in, vg_base)

    X_vm = vmem[WASHOUT:]
    X_sp = spikes[WASHOUT:]
    X_both = np.hstack([vmem[WASHOUT:], spikes[WASHOUT:]])
    inp = input_signal[WASHOUT:]

    targets = {
        "linear": inp,
        "quadratic": inp ** 2,
        "cubic": inp ** 3,
        "sine": np.sin(np.pi * inp),
        "abs": np.abs(inp),
    }

    total_cap = 0.0
    for name, y in targets.items():
        # Try vmem, spikes, and both
        r2_vm, _, _ = ridge_regress(X_vm, y, alpha=1.0)
        r2_sp, _, _ = ridge_regress(X_sp, y, alpha=1.0)
        r2_both, _, _ = ridge_regress(X_both, y, alpha=1.0)
        best = max(r2_vm, r2_sp, r2_both)
        results[f"exp4_{name}_r2_vmem"] = float(r2_vm)
        results[f"exp4_{name}_r2_spikes"] = float(r2_sp)
        results[f"exp4_{name}_r2_both"] = float(r2_both)
        total_cap += max(0, best)
        src = "vmem" if r2_vm >= r2_sp and r2_vm >= r2_both else ("spikes" if r2_sp >= r2_both else "both")
        print(f"  {name}: vmem={r2_vm:.4f}, spikes={r2_sp:.4f}, both={r2_both:.4f} (best={best:.4f} from {src})")

    results["exp4_total_capacity"] = float(total_cap)
    results["T970_linear_gt_0"] = "PASS" if results.get("exp4_linear_r2_both", 0) > 0 else "FAIL"
    results["T971_quadratic_gt_0"] = "PASS" if results.get("exp4_quadratic_r2_both", 0) > 0 else "FAIL"
    results["T972_total_cap_gt_05"] = "PASS" if total_cap > 0.5 else "FAIL"
    results["T973_vmem_gt_spikes_linear"] = "PASS" if results.get("exp4_linear_r2_vmem", 0) > results.get("exp4_linear_r2_spikes", 0) else "FAIL"

    for tid, name, val in [
        ("T970", f"linear({results.get('exp4_linear_r2_both', 0):.4f})>0",
         results.get("exp4_linear_r2_both", 0) > 0),
        ("T971", f"quadratic({results.get('exp4_quadratic_r2_both', 0):.4f})>0",
         results.get("exp4_quadratic_r2_both", 0) > 0),
        ("T972", f"total_cap({total_cap:.4f})>0.5", total_cap > 0.5),
        ("T973", f"vmem>spikes (linear)",
         results.get("exp4_linear_r2_vmem", 0) > results.get("exp4_linear_r2_spikes", 0)),
    ]:
        print(f"  {tid} {name}: {'PASS' if val else 'FAIL'}")


# ==========================================================================
# EXP 5: Streaming XOR with vmem + windowed features
# ==========================================================================
def run_exp5_streaming_xor(fpga, results):
    """Streaming XOR using vmem features and sliding windows."""
    print("\n=== EXP 5: Streaming XOR (vmem) ===")

    rng = np.random.RandomState(55)
    w_in = rng.uniform(-1, 1, N_NEURONS)
    N_STEPS = 600
    WASHOUT = 10

    for tau in [1, 2, 3, 5]:
        vg_base = configure_fpga(fpga)
        time.sleep(0.3)

        bits = rng.randint(0, 2, N_STEPS)
        input_signal = bits.astype(np.float32) * 2 - 1  # {-1, +1}
        xor_target = np.array([bits[t] ^ bits[t - tau] if t >= tau else 0
                               for t in range(N_STEPS)])

        spikes, vmem = collect_sequence(fpga, input_signal, "COUPLED", w_in, vg_base)

        # Features: vmem window of size tau+1, plus spike deltas
        features = []
        targets = []
        for t in range(max(tau, 3), N_STEPS):
            vm_window = vmem[t-tau:t+1].ravel()
            sp_window = spikes[t-tau:t+1].ravel()
            feat = np.concatenate([vm_window, sp_window])
            features.append(feat)
            targets.append(xor_target[t])

        X = np.array(features)[WASHOUT:]
        y = np.array(targets)[WASHOUT:]

        # Classification
        scaler = StandardScaler()
        std = X.std(axis=0)
        mask = std > 1e-6
        if mask.sum() < 3:
            acc = 0.5
        else:
            X_f = scaler.fit_transform(X[:, mask])
            clf = RidgeClassifier(alpha=1.0)
            kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            scores = cross_val_score(clf, X_f, y, cv=kf, scoring='accuracy')
            acc = float(scores.mean())

        results[f"exp5_xor_tau{tau}"] = acc
        print(f"  tau={tau}: acc={acc:.4f}")

    best_xor = max(results.get(f"exp5_xor_tau{t}", 0.5) for t in [1, 2, 3, 5])
    results["T974_xor_tau1_gt_52"] = "PASS" if results.get("exp5_xor_tau1", 0.5) > 0.52 else "FAIL"
    results["T975_xor_tau2_gt_52"] = "PASS" if results.get("exp5_xor_tau2", 0.5) > 0.52 else "FAIL"
    results["T976_any_xor_gt_55"] = "PASS" if best_xor > 0.55 else "FAIL"
    results["T977_any_xor_gt_60"] = "PASS" if best_xor > 0.60 else "FAIL"

    for tid, name, val in [
        ("T974", f"tau1({results.get('exp5_xor_tau1', 0.5):.4f})>0.52",
         results.get("exp5_xor_tau1", 0.5) > 0.52),
        ("T975", f"tau2({results.get('exp5_xor_tau2', 0.5):.4f})>0.52",
         results.get("exp5_xor_tau2", 0.5) > 0.52),
        ("T976", f"best({best_xor:.4f})>0.55", best_xor > 0.55),
        ("T977", f"best({best_xor:.4f})>0.60", best_xor > 0.60),
    ]:
        print(f"  {tid} {name}: {'PASS' if val else 'FAIL'}")


# ==========================================================================
# EXP 6: GPU LIF + FPGA vmem hybrid for NARMA
# ==========================================================================
def run_exp6_hybrid_narma(fpga, results):
    """Hybrid GPU LIF states + FPGA vmem for NARMA-5 regression."""
    print("\n=== EXP 6: Hybrid GPU+FPGA NARMA-5 ===")

    # Compile HIP LIF kernel
    hip_src = "/tmp/lif_narma.hip"
    hip_bin = "/tmp/lif_narma"
    with open(hip_src, 'w') as f:
        f.write("""
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cmath>
#include <cstdlib>

#define N_NEURONS 64
#define TAU 0.95f

__global__ void lif_step(float* vmem, float* spikes, const float* input,
                         const float* w_in, int n_steps) {
    int nid = threadIdx.x;
    if (nid >= N_NEURONS) return;

    float v = 0.0f;
    float threshold = 1.0f;
    // Substrate noise from clock
    unsigned long long t0 = clock64();

    for (int t = 0; t < n_steps; t++) {
        float noise = (float)((clock64() ^ (t0 + t*7919)) % 10000) / 50000.0f;
        float I = input[t] * w_in[nid] + noise;
        v = TAU * v + I;

        if (v >= threshold) {
            spikes[t * N_NEURONS + nid] = 1.0f;
            v = 0.0f;
        } else {
            spikes[t * N_NEURONS + nid] = 0.0f;
        }
        vmem[t * N_NEURONS + nid] = v;
    }
}

int main(int argc, char** argv) {
    int n_steps = atoi(argv[1]);
    float* h_input = new float[n_steps];
    float* h_w_in = new float[N_NEURONS];
    float* h_vmem = new float[n_steps * N_NEURONS];
    float* h_spikes = new float[n_steps * N_NEURONS];

    // Read input from stdin
    for (int i = 0; i < n_steps; i++) scanf("%f", &h_input[i]);

    // Random weights (seeded)
    srand(42);
    for (int i = 0; i < N_NEURONS; i++)
        h_w_in[i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;

    float *d_input, *d_w_in, *d_vmem, *d_spikes;
    hipMalloc(&d_input, n_steps * sizeof(float));
    hipMalloc(&d_w_in, N_NEURONS * sizeof(float));
    hipMalloc(&d_vmem, n_steps * N_NEURONS * sizeof(float));
    hipMalloc(&d_spikes, n_steps * N_NEURONS * sizeof(float));

    hipMemcpy(d_input, h_input, n_steps * sizeof(float), hipMemcpyHostToDevice);
    hipMemcpy(d_w_in, h_w_in, N_NEURONS * sizeof(float), hipMemcpyHostToDevice);

    lif_step<<<1, N_NEURONS>>>(d_vmem, d_spikes, d_input, d_w_in, n_steps);
    hipDeviceSynchronize();

    hipMemcpy(h_vmem, d_vmem, n_steps * N_NEURONS * sizeof(float), hipMemcpyDeviceToHost);
    hipMemcpy(h_spikes, d_spikes, n_steps * N_NEURONS * sizeof(float), hipMemcpyDeviceToHost);

    // Output vmem values
    for (int t = 0; t < n_steps; t++) {
        for (int n = 0; n < N_NEURONS; n++)
            printf("%.6f ", h_vmem[t * N_NEURONS + n]);
        printf("\\n");
    }

    hipFree(d_input); hipFree(d_w_in); hipFree(d_vmem); hipFree(d_spikes);
    delete[] h_input; delete[] h_w_in; delete[] h_vmem; delete[] h_spikes;
    return 0;
}
""")

    import subprocess
    comp = subprocess.run(
        ["hipcc", "--offload-arch=gfx1100", "-O2", "-o", hip_bin, hip_src],
        capture_output=True, text=True, timeout=60)

    if comp.returncode != 0:
        print(f"  HIP compile failed: {comp.stderr[:200]}")
        results["T978_hybrid_narma5_gt_0"] = "FAIL"
        results["T979_hybrid_gt_fpga_alone"] = "FAIL"
        return

    print("  HIP LIF compiled")

    rng = np.random.RandomState(42)
    w_in_fpga = rng.uniform(-1, 1, N_NEURONS)
    N_STEPS = 500
    WASHOUT = 50

    u = rng.uniform(0, 0.5, N_STEPS + 5)
    y_narma = generate_narma(u, order=5)
    input_signal = u[5:]
    target = y_narma[5:]

    # Collect FPGA response
    vg_base = configure_fpga(fpga)
    time.sleep(0.5)
    fpga_spikes, fpga_vmem = collect_sequence(
        fpga, input_signal, "COUPLED", w_in_fpga, vg_base)

    # Run GPU LIF
    inp_str = "\n".join(f"{float(x):.6f}" for x in input_signal)
    try:
        proc = subprocess.run(
            [hip_bin, str(N_STEPS)],
            input=inp_str, capture_output=True, text=True, timeout=30,
            env={**os.environ, "HSA_OVERRIDE_GFX_VERSION": "11.0.0"})
        gpu_lines = proc.stdout.strip().split("\n")
        gpu_vmem = np.array([[float(x) for x in line.split()] for line in gpu_lines])
    except Exception as e:
        print(f"  GPU LIF failed: {e}")
        gpu_vmem = np.zeros((N_STEPS, 64))

    # FPGA only regression
    X_fpga = np.hstack([fpga_vmem[WASHOUT:], fpga_spikes[WASHOUT:]])
    y_tgt = target[WASHOUT:]
    r2_fpga, _, _ = ridge_regress(X_fpga, y_tgt, alpha=10.0)

    # GPU only regression
    n_gpu = min(len(gpu_vmem), N_STEPS)
    X_gpu = gpu_vmem[WASHOUT:n_gpu]
    y_gpu = target[WASHOUT:n_gpu]
    r2_gpu, _, _ = ridge_regress(X_gpu, y_gpu, alpha=10.0)

    # Hybrid: concat FPGA + GPU features
    n_min = min(len(X_fpga), len(X_gpu))
    X_hybrid = np.hstack([X_fpga[:n_min], X_gpu[:n_min]])
    y_hyb = y_tgt[:n_min]
    r2_hybrid, _, _ = ridge_regress(X_hybrid, y_hyb, alpha=10.0)

    results["exp6_fpga_narma5_r2"] = float(r2_fpga)
    results["exp6_gpu_narma5_r2"] = float(r2_gpu)
    results["exp6_hybrid_narma5_r2"] = float(r2_hybrid)

    print(f"  FPGA only:  R² = {r2_fpga:.4f}")
    print(f"  GPU only:   R² = {r2_gpu:.4f}")
    print(f"  Hybrid:     R² = {r2_hybrid:.4f}")

    results["T978_hybrid_narma5_gt_0"] = "PASS" if r2_hybrid > 0 else "FAIL"
    results["T979_hybrid_gt_fpga_alone"] = "PASS" if r2_hybrid > r2_fpga else "FAIL"
    results["T980_gpu_narma5_gt_0"] = "PASS" if r2_gpu > 0 else "FAIL"
    results["T981_any_narma5_gt_005"] = "PASS" if max(r2_fpga, r2_gpu, r2_hybrid) > 0.05 else "FAIL"

    for tid, name, val in [
        ("T978", f"hybrid({r2_hybrid:.4f})>0", r2_hybrid > 0),
        ("T979", f"hybrid({r2_hybrid:.4f})>fpga({r2_fpga:.4f})", r2_hybrid > r2_fpga),
        ("T980", f"gpu({r2_gpu:.4f})>0", r2_gpu > 0),
        ("T981", f"any({max(r2_fpga, r2_gpu, r2_hybrid):.4f})>0.05",
         max(r2_fpga, r2_gpu, r2_hybrid) > 0.05),
    ]:
        print(f"  {tid} {name}: {'PASS' if val else 'FAIL'}")


# ==========================================================================
# EXP 7: Vmem-based input reconstruction (sanity check)
# ==========================================================================
def run_exp7_reconstruction(fpga, results):
    """Can we reconstruct the input signal from FPGA vmem? Fundamental test."""
    print("\n=== EXP 7: Input Reconstruction ===")

    rng = np.random.RandomState(33)
    w_in = rng.uniform(-1, 1, N_NEURONS)

    for sig_type in ["sine", "random", "step"]:
        vg_base = configure_fpga(fpga)
        time.sleep(0.3)

        N = 300
        if sig_type == "sine":
            input_signal = np.sin(2 * np.pi * np.arange(N) / 40) * 0.5
        elif sig_type == "random":
            input_signal = rng.uniform(-0.5, 0.5, N)
        else:  # step
            input_signal = np.zeros(N)
            for i in range(0, N, 50):
                input_signal[i:i+50] = rng.choice([-0.5, -0.25, 0, 0.25, 0.5])

        spikes, vmem = collect_sequence(fpga, input_signal, "COUPLED", w_in, vg_base)

        # Predict current input
        r2_vm, _, _ = ridge_regress(vmem[10:], input_signal[10:])
        r2_sp, _, _ = ridge_regress(spikes[10:], input_signal[10:])
        r2_both, _, _ = ridge_regress(np.hstack([vmem[10:], spikes[10:]]), input_signal[10:])

        results[f"exp7_{sig_type}_r2_vmem"] = float(r2_vm)
        results[f"exp7_{sig_type}_r2_spikes"] = float(r2_sp)
        results[f"exp7_{sig_type}_r2_both"] = float(r2_both)
        print(f"  {sig_type}: vmem={r2_vm:.4f}, spikes={r2_sp:.4f}, both={r2_both:.4f}")

    # Any signal reconstructed?
    best_recon = max(
        results.get(f"exp7_{s}_r2_both", 0) for s in ["sine", "random", "step"])

    results["T982_sine_recon_gt_0"] = "PASS" if results.get("exp7_sine_r2_both", 0) > 0 else "FAIL"
    results["T983_random_recon_gt_0"] = "PASS" if results.get("exp7_random_r2_both", 0) > 0 else "FAIL"
    results["T984_any_recon_gt_02"] = "PASS" if best_recon > 0.2 else "FAIL"
    results["T985_vmem_gt_spikes"] = "PASS" if (
        results.get("exp7_sine_r2_vmem", 0) > results.get("exp7_sine_r2_spikes", 0)
    ) else "FAIL"

    for tid, name, val in [
        ("T982", f"sine_recon({results.get('exp7_sine_r2_both', 0):.4f})>0",
         results.get("exp7_sine_r2_both", 0) > 0),
        ("T983", f"random_recon({results.get('exp7_random_r2_both', 0):.4f})>0",
         results.get("exp7_random_r2_both", 0) > 0),
        ("T984", f"best_recon({best_recon:.4f})>0.2", best_recon > 0.2),
        ("T985", f"vmem>spikes (sine)",
         results.get("exp7_sine_r2_vmem", 0) > results.get("exp7_sine_r2_spikes", 0)),
    ]:
        print(f"  {tid} {name}: {'PASS' if val else 'FAIL'}")


# ==========================================================================
# Main
# ==========================================================================
def main():
    print("=" * 70)
    print("z2240: Regression Fix — vmem-centric features")
    print("=" * 70)
    print(f"Time: {datetime.now().isoformat()}")

    fpga = FPGAEthBridge()
    fpga.connect()
    print(f"Connected to FPGA: {fpga.num_neurons} neurons")

    results = {
        "experiment": "z2240_regression_fix",
        "timestamp": datetime.now().isoformat(),
    }

    run_diagnostic(fpga, results)
    run_exp1_narma_vmem(fpga, results)
    run_exp2_esn_narma(fpga, results)
    run_exp3_memory_capacity(fpga, results)
    run_exp4_separation(fpga, results)
    run_exp5_streaming_xor(fpga, results)
    run_exp6_hybrid_narma(fpga, results)
    run_exp7_reconstruction(fpga, results)

    # Summary
    n_pass = sum(1 for k, v in results.items() if k.startswith("T") and v == "PASS")
    n_total = sum(1 for k, v in results.items() if k.startswith("T") and v in ("PASS", "FAIL"))
    results["summary"] = f"{n_pass}/{n_total} PASS"

    print(f"\n{'=' * 70}")
    print(f"Summary: {results['summary']}")
    print(f"{'=' * 70}")

    out_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "results", "z2240_regression_fix.json")
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Results saved to {out_path}")

    fpga.close()


if __name__ == "__main__":
    main()
