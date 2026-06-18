#!/usr/bin/env python3
"""z2243: Temporal Challenge Suite — Tasks where FPGA fading memory matters.

z2242 showed GPU LIF saturates at 100% on simple classification. This experiment
targets tasks requiring temporal integration over many steps, where:
- GPU LIF (stateless between subprocess calls) should struggle
- FPGA reservoir (persistent neuron state) should excel
- Cross-substrate bridge should show additive value

Architecture:
- GPU LIF: 64 neurons, 50-step trials (mode=3 full), but each call is independent
- FPGA: 128 neurons, 50 steps/trial at 20Hz, state persists across steps
- Bridge: GPU vmem→FPGA MAC, FPGA spikes→GPU features

Tests T1043-T1065 (23 tests)
"""

import sys, os, time, json, subprocess, struct
import numpy as np
from datetime import datetime
from sklearn.linear_model import RidgeClassifier, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_score, KFold

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.fpga_host_eth import FPGAEthBridge

N_FPGA = 128
N_GPU = 64
BASE_VG = 0.58
ALPHA = 0.25
BETA = 0.08
SAMPLE_HZ = 20
STEP_INTERVAL = 1.0 / SAMPLE_HZ
TUNED_LEAK = 0x0011
TUNED_THRESH_F = 0.50
TUNED_BIAS_GAIN_F = 0.03125

HIP_SRC = "/tmp/deep_gpu_neuro.hip"
HIP_BIN = "/tmp/deep_gpu_neuro"

RESULTS = {}


def read_gpu_power():
    try:
        with open('/sys/class/hwmon/hwmon7/power1_average', 'r') as f:
            return float(f.read().strip()) / 1e6
    except:
        return 11.0


def configure_fpga(fpga):
    fpga.set_leak_cond(TUNED_LEAK)
    fpga.set_threshold(TUNED_THRESH_F)
    fpga.set_bias_gain(TUNED_BIAS_GAIN_F)
    vg_base = np.array([BASE_VG + 0.15 * (i/127 - 0.5) for i in range(N_FPGA)])
    fpga.set_vg_batch(0, [float(v) for v in vg_base])
    return vg_base


def ridge_classify(X, y, n_splits=5):
    std = X.std(axis=0)
    mask = std > 1e-2
    if mask.sum() < 3:
        return 0.5, 0.0
    scaler = StandardScaler()
    X_f = scaler.fit_transform(X[:, mask])
    classes, counts = np.unique(y, return_counts=True)
    if len(classes) < 2:
        return 0.0, 0.0
    if counts.min() < n_splits:
        n_splits = max(2, counts.min())
    clf = RidgeClassifier(alpha=1.0)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    scores = cross_val_score(clf, X_f, y, cv=skf, scoring='accuracy')
    return float(scores.mean()), float(scores.std())


def ridge_regress(X, y, n_splits=5):
    std = X.std(axis=0)
    mask = std > 1e-2
    if mask.sum() < 3:
        return 0.0
    scaler = StandardScaler()
    X_f = scaler.fit_transform(X[:, mask])
    if len(X_f) < n_splits * 2:
        n_splits = max(2, len(X_f) // 2)
    clf = Ridge(alpha=10.0)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    scores = cross_val_score(clf, X_f, y, cv=kf, scoring='r2')
    return float(scores.mean())


def run_gpu_lif(input_signal, mode=3):
    """Run GPU LIF (full mode) and return (vmem, spikes, adaptation)."""
    n_steps = len(input_signal)
    inp_str = "\n".join(f"{float(x):.6f}" for x in input_signal)
    try:
        proc = subprocess.run(
            [HIP_BIN, str(n_steps), str(mode)],
            input=inp_str, capture_output=True, text=True, timeout=30,
            env={**os.environ, "HSA_OVERRIDE_GFX_VERSION": "11.0.0"})
        lines = proc.stdout.strip().split("\n")
        vmem = np.array([[float(x) for x in lines[t].split()] for t in range(n_steps)])
        spikes = np.array([[float(x) for x in lines[n_steps + t].split()] for t in range(n_steps)])
        adapt = np.array([[float(x) for x in lines[2*n_steps + t].split()] for t in range(n_steps)])
        return vmem, spikes, adapt
    except Exception as e:
        print(f"  GPU LIF error: {e}")
        return (np.zeros((n_steps, N_GPU)),
                np.zeros((n_steps, N_GPU)),
                np.zeros((n_steps, N_GPU)))


def collect_fpga_extended(fpga, input_signal, condition, w_in, vg_base, n_steps):
    """Collect FPGA response with extended step count for temporal tasks."""
    all_spikes = []
    all_vmem = []
    gpu_power_base = read_gpu_power()
    noise_state = 0.0

    for step in range(n_steps):
        t0 = time.time()
        inp = float(input_signal[step]) if step < len(input_signal) else 0.0
        gpu_power = read_gpu_power()
        power_noise = (gpu_power - gpu_power_base) / 5.0
        noise_state = 0.85 * noise_state + 0.15 * power_noise

        if condition == "COUPLED":
            mac_val = inp + noise_state * 0.3
            fpga.set_mac_signal(float(np.clip(mac_val * 0.5 + 0.5, 0.0, 1.0)))
            vg_mod = vg_base + ALPHA * inp + BETA * w_in * inp + 0.05 * noise_state
        elif condition == "FPGA_ONLY":
            fpga.set_mac_signal(float(np.clip(inp * 0.5 + 0.5, 0.0, 1.0)))
            vg_mod = vg_base + ALPHA * inp
        else:  # STATIC
            fpga.set_mac_signal(0.5)
            vg_mod = vg_base.copy()

        fpga.set_vg_batch(0, [float(np.clip(v, 0.3, 0.9)) for v in vg_mod])

        telem = fpga.read_telemetry(timeout=0.15)
        if telem is not None:
            sc = np.clip(telem['spike_counts'].astype(np.float32), 0, 65000)
            vm = telem['vmem'].copy()
        else:
            sc = np.zeros(N_FPGA, dtype=np.float32)
            vm = np.zeros(N_FPGA, dtype=np.float32)

        all_spikes.append(sc)
        all_vmem.append(vm)

        elapsed = time.time() - t0
        if elapsed < STEP_INTERVAL:
            time.sleep(STEP_INTERVAL - elapsed)

    return np.array(all_spikes), np.array(all_vmem)


def temporal_features(spikes, vmem):
    """Rich temporal features from time series — deltas, trends, moments."""
    n_steps = spikes.shape[0]
    feats = []
    # Snapshot features
    feats.append(spikes.mean(axis=0))
    feats.append(spikes.std(axis=0))
    feats.append(vmem.mean(axis=0))
    feats.append(vmem.std(axis=0))
    # Temporal: first half vs second half
    half = n_steps // 2
    feats.append(spikes[half:].mean(axis=0) - spikes[:half].mean(axis=0))
    feats.append(vmem[half:].mean(axis=0) - vmem[:half].mean(axis=0))
    # Temporal: last vs first
    feats.append(spikes[-1] - spikes[0])
    feats.append(vmem[-1] - vmem[0])
    # Temporal: max and argmax
    feats.append(vmem.max(axis=0))
    feats.append(vmem.argmax(axis=0).astype(np.float32) / n_steps)
    return np.concatenate(feats)


def gpu_snapshot_features(vmem, spikes, adapt):
    """GPU features from single subprocess run."""
    feats = []
    feats.append(vmem.mean(axis=0))
    feats.append(vmem.std(axis=0))
    feats.append(spikes.sum(axis=0))
    feats.append(adapt[-1])
    return np.concatenate(feats)


# ==========================================================================
# EXP 1: Long-Delay XOR (temporal memory required)
# ==========================================================================
def run_exp1_long_xor(fpga, vg_base, w_in, results):
    """XOR at delays 1,2,5,10 — tests temporal memory.

    Input: binary stream. Label = input[t] XOR input[t-delay].
    FPGA maintains state across steps → should have memory.
    GPU LIF sees entire stream at once → can learn XOR within a call.
    Key test: does FPGA memory persist at long delays?
    """
    print("\n=== EXP 1: Long-Delay XOR ===")
    rng = np.random.RandomState(42)

    N_STEPS = 50  # steps per trial — long enough for FPGA memory
    N_TRIALS = 120
    delays = [1, 2, 5, 10]

    for delay in delays:
        print(f"\n  Delay={delay}")

        fpga_feats_coupled = []
        fpga_feats_only = []
        gpu_feats = []
        labels = []

        for trial in range(N_TRIALS):
            if trial % 40 == 0:
                print(f"  Trial {trial}/{N_TRIALS}")

            # Binary input stream
            stream = rng.randint(0, 2, size=N_STEPS).astype(np.float32)
            # XOR label based on last pair in stream
            t_eval = N_STEPS - 1
            t_ref = t_eval - delay
            if t_ref >= 0:
                label = int(stream[t_eval]) ^ int(stream[t_ref])
            else:
                label = 0
            labels.append(label)

            # Input signal: convert binary to [-1, 1]
            input_signal = stream * 2.0 - 1.0

            # GPU LIF: sees full stream, can learn temporal pattern within run
            vmem_g, spikes_g, adapt_g = run_gpu_lif(input_signal, mode=3)
            gpu_feats.append(gpu_snapshot_features(vmem_g, spikes_g, adapt_g))

            # FPGA COUPLED: persistent neurons, state builds over 50 steps
            spk_c, vm_c = collect_fpga_extended(fpga, input_signal, "COUPLED",
                                                 w_in, vg_base, N_STEPS)
            fpga_feats_coupled.append(temporal_features(spk_c, vm_c))

            # FPGA ONLY
            spk_f, vm_f = collect_fpga_extended(fpga, input_signal, "FPGA_ONLY",
                                                 w_in, vg_base, N_STEPS)
            fpga_feats_only.append(temporal_features(spk_f, vm_f))

        X_gpu = np.array(gpu_feats)
        X_fpga_c = np.array(fpga_feats_coupled)
        X_fpga_o = np.array(fpga_feats_only)
        y = np.array(labels)

        acc_gpu, _ = ridge_classify(X_gpu, y)
        acc_coupled, _ = ridge_classify(X_fpga_c, y)
        acc_fpga, _ = ridge_classify(X_fpga_o, y)
        # Combined: GPU + FPGA features
        X_combined = np.hstack([X_gpu, X_fpga_c])
        acc_combined, _ = ridge_classify(X_combined, y)

        print(f"  GPU: {acc_gpu:.3f}, COUPLED: {acc_coupled:.3f}, FPGA: {acc_fpga:.3f}, COMBINED: {acc_combined:.3f}")

        results[f"exp1_xor_d{delay}_gpu"] = acc_gpu
        results[f"exp1_xor_d{delay}_coupled"] = acc_coupled
        results[f"exp1_xor_d{delay}_fpga"] = acc_fpga
        results[f"exp1_xor_d{delay}_combined"] = acc_combined

    # Tests
    # T1043: FPGA coupled > chance (0.50) at d=1
    p = results.get("exp1_xor_d1_coupled", 0)
    results["T1043_xor_d1_coupled_gt_chance"] = "PASS" if p > 0.52 else "FAIL"
    print(f"  T1043 xor_d1_coupled({p:.3f})>0.52: {results['T1043_xor_d1_coupled_gt_chance']}")

    # T1044: FPGA coupled > chance at d=2
    p = results.get("exp1_xor_d2_coupled", 0)
    results["T1044_xor_d2_coupled_gt_chance"] = "PASS" if p > 0.52 else "FAIL"
    print(f"  T1044 xor_d2_coupled({p:.3f})>0.52: {results['T1044_xor_d2_coupled_gt_chance']}")

    # T1045: COMBINED > GPU alone at any delay
    any_combined_better = any(
        results.get(f"exp1_xor_d{d}_combined", 0) > results.get(f"exp1_xor_d{d}_gpu", 1)
        for d in delays)
    results["T1045_combined_gt_gpu_any"] = "PASS" if any_combined_better else "FAIL"
    print(f"  T1045 combined>gpu any delay: {results['T1045_combined_gt_gpu_any']}")

    # T1046: FPGA accuracy decreases with delay (memory fading)
    accs = [results.get(f"exp1_xor_d{d}_coupled", 0) for d in delays]
    monotone = sum(1 for i in range(len(accs)-1) if accs[i] >= accs[i+1])
    results["T1046_memory_fading"] = "PASS" if monotone >= 2 else "FAIL"
    print(f"  T1046 memory fading (monotone {monotone}/3): {results['T1046_memory_fading']}")


# ==========================================================================
# EXP 2: Temporal Pattern Discrimination
# ==========================================================================
def run_exp2_temporal_patterns(fpga, vg_base, w_in, results):
    """Classify temporal PATTERNS, not waveform shapes.

    All patterns use same frequency but different temporal structures:
    - RAMP_UP: amplitude increases over time
    - RAMP_DOWN: amplitude decreases
    - PULSE_EARLY: strong pulse in first half, quiet second half
    - PULSE_LATE: quiet first half, strong pulse in second half

    These are IMPOSSIBLE to distinguish from snapshot statistics alone —
    require temporal integration.
    """
    print("\n=== EXP 2: Temporal Pattern Discrimination ===")
    rng = np.random.RandomState(43)

    N_STEPS = 50
    N_TRIALS = 160
    N_CLASSES = 4

    conditions = ["COUPLED", "FPGA_ONLY", "STATIC"]
    feats_by_cond = {c: [] for c in conditions}
    gpu_feats = []
    labels = []

    for trial in range(N_TRIALS):
        if trial % 40 == 0:
            print(f"  Trial {trial}/{N_TRIALS}")

        cls = trial % N_CLASSES
        t_arr = np.linspace(0, 1, N_STEPS)
        freq = 3.0  # same for all

        if cls == 0:  # RAMP_UP
            envelope = t_arr
        elif cls == 1:  # RAMP_DOWN
            envelope = 1.0 - t_arr
        elif cls == 2:  # PULSE_EARLY
            envelope = np.where(t_arr < 0.3, 1.0, 0.1)
        else:  # PULSE_LATE
            envelope = np.where(t_arr > 0.7, 1.0, 0.1)

        input_signal = (envelope * np.sin(2 * np.pi * freq * t_arr)).astype(np.float32)
        labels.append(cls)

        # GPU LIF
        vmem_g, spikes_g, adapt_g = run_gpu_lif(input_signal, mode=3)
        gpu_feats.append(gpu_snapshot_features(vmem_g, spikes_g, adapt_g))

        # FPGA conditions
        for cond in conditions:
            spk, vm = collect_fpga_extended(fpga, input_signal, cond, w_in, vg_base, N_STEPS)
            feats_by_cond[cond].append(temporal_features(spk, vm))

    y = np.array(labels)
    X_gpu = np.array(gpu_feats)

    acc_gpu, _ = ridge_classify(X_gpu, y)
    results["exp2_gpu_temporal"] = acc_gpu
    print(f"  GPU: {acc_gpu:.3f}")

    for cond in conditions:
        X = np.array(feats_by_cond[cond])
        acc, _ = ridge_classify(X, y)
        results[f"exp2_{cond}_temporal"] = acc
        print(f"  {cond}: {acc:.3f}")

    # Combined
    X_combined = np.hstack([X_gpu, np.array(feats_by_cond["COUPLED"])])
    acc_comb, _ = ridge_classify(X_combined, y)
    results["exp2_COMBINED_temporal"] = acc_comb
    print(f"  COMBINED: {acc_comb:.3f}")

    # T1047: COUPLED > STATIC (temporal features matter)
    c = results.get("exp2_COUPLED_temporal", 0)
    s = results.get("exp2_STATIC_temporal", 0)
    results["T1047_coupled_gt_static"] = "PASS" if c > s else "FAIL"
    print(f"  T1047 COUPLED({c:.3f})>STATIC({s:.3f}): {results['T1047_coupled_gt_static']}")

    # T1048: COUPLED > chance (0.25 for 4-class)
    results["T1048_coupled_gt_chance"] = "PASS" if c > 0.30 else "FAIL"
    print(f"  T1048 COUPLED({c:.3f})>0.30: {results['T1048_coupled_gt_chance']}")

    # T1049: COMBINED > GPU alone (FPGA temporal adds value)
    g = results.get("exp2_gpu_temporal", 0)
    cb = results.get("exp2_COMBINED_temporal", 0)
    results["T1049_combined_gt_gpu"] = "PASS" if cb > g else "FAIL"
    print(f"  T1049 COMBINED({cb:.3f})>GPU({g:.3f}): {results['T1049_combined_gt_gpu']}")

    # T1050: any condition > 0.50
    best = max(c, results.get("exp2_FPGA_ONLY_temporal", 0), cb, g)
    results["T1050_any_gt_50"] = "PASS" if best > 0.50 else "FAIL"
    print(f"  T1050 best({best:.3f})>0.50: {results['T1050_any_gt_50']}")


# ==========================================================================
# EXP 3: N-Back Memory Task
# ==========================================================================
def run_exp3_nback(fpga, vg_base, w_in, results):
    """N-back: classify what the input was N steps ago.

    Pure memory test. GPU LIF sees all steps but has to learn temporal mapping.
    FPGA has fading membrane memory.
    """
    print("\n=== EXP 3: N-Back Memory Task ===")
    rng = np.random.RandomState(44)

    N_STEPS = 50
    N_TRIALS = 120
    N_CLASSES = 4  # 4 possible input values
    n_backs = [1, 2, 5, 10]

    for n_back in n_backs:
        print(f"\n  N-back={n_back}")

        fpga_feats = []
        gpu_feats = []
        labels = []

        for trial in range(N_TRIALS):
            if trial % 40 == 0:
                print(f"  Trial {trial}/{N_TRIALS}")

            # Random sequence of discrete values
            seq = rng.randint(0, N_CLASSES, size=N_STEPS)
            # Convert to analog signal
            input_signal = (seq.astype(np.float32) / (N_CLASSES - 1)) * 2.0 - 1.0

            # Label: what was the input n_back steps before the end?
            t_query = N_STEPS - 1
            t_target = t_query - n_back
            if t_target >= 0:
                label = seq[t_target]
            else:
                label = 0
            labels.append(label)

            # GPU
            vmem_g, spikes_g, adapt_g = run_gpu_lif(input_signal, mode=3)
            gpu_feats.append(gpu_snapshot_features(vmem_g, spikes_g, adapt_g))

            # FPGA COUPLED
            spk, vm = collect_fpga_extended(fpga, input_signal, "COUPLED",
                                             w_in, vg_base, N_STEPS)
            fpga_feats.append(temporal_features(spk, vm))

        X_gpu = np.array(gpu_feats)
        X_fpga = np.array(fpga_feats)
        X_combined = np.hstack([X_gpu, X_fpga])
        y = np.array(labels)

        acc_gpu, _ = ridge_classify(X_gpu, y)
        acc_fpga, _ = ridge_classify(X_fpga, y)
        acc_combined, _ = ridge_classify(X_combined, y)

        print(f"  GPU: {acc_gpu:.3f}, FPGA: {acc_fpga:.3f}, COMBINED: {acc_combined:.3f}")

        results[f"exp3_nback{n_back}_gpu"] = acc_gpu
        results[f"exp3_nback{n_back}_fpga"] = acc_fpga
        results[f"exp3_nback{n_back}_combined"] = acc_combined

    # T1051: FPGA > chance (0.25) at n=1
    p = results.get("exp3_nback1_fpga", 0)
    results["T1051_nback1_fpga_gt_chance"] = "PASS" if p > 0.28 else "FAIL"
    print(f"  T1051 nback1_fpga({p:.3f})>0.28: {results['T1051_nback1_fpga_gt_chance']}")

    # T1052: FPGA > chance at n=2
    p = results.get("exp3_nback2_fpga", 0)
    results["T1052_nback2_fpga_gt_chance"] = "PASS" if p > 0.28 else "FAIL"
    print(f"  T1052 nback2_fpga({p:.3f})>0.28: {results['T1052_nback2_fpga_gt_chance']}")

    # T1053: Memory decay — n=1 > n=10
    a1 = results.get("exp3_nback1_fpga", 0)
    a10 = results.get("exp3_nback10_fpga", 0)
    results["T1053_memory_decay"] = "PASS" if a1 > a10 else "FAIL"
    print(f"  T1053 nback1({a1:.3f})>nback10({a10:.3f}): {results['T1053_memory_decay']}")

    # T1054: COMBINED > either alone at n=2
    comb2 = results.get("exp3_nback2_combined", 0)
    gpu2 = results.get("exp3_nback2_gpu", 0)
    fpga2 = results.get("exp3_nback2_fpga", 0)
    results["T1054_combined_synergy"] = "PASS" if comb2 > max(gpu2, fpga2) else "FAIL"
    print(f"  T1054 combined({comb2:.3f})>max(gpu:{gpu2:.3f},fpga:{fpga2:.3f}): {results['T1054_combined_synergy']}")


# ==========================================================================
# EXP 4: NARMA-5 with Extended Steps
# ==========================================================================
def run_exp4_narma(fpga, vg_base, w_in, results):
    """NARMA-5 with 100 steps — enough for FPGA dynamics to develop.

    Previous attempts used 5-10 steps and R²=0. With 100 steps:
    - FPGA has time to build up temporal correlations
    - Readout uses temporal features, not just snapshot
    - Compare spike-based vs vmem-based regression
    """
    print("\n=== EXP 4: NARMA-5 Extended ===")
    rng = np.random.RandomState(45)

    N_STEPS = 100
    N_TRIALS = 80
    ORDER = 5

    # Generate NARMA-5 targets
    def narma5(u, n):
        y = np.zeros(n)
        for t in range(ORDER, n):
            y[t] = 0.3 * y[t-1] + 0.05 * y[t-1] * sum(y[t-j] for j in range(1, ORDER+1)) + \
                   1.5 * u[t-1] * u[t-ORDER] + 0.1
            y[t] = np.clip(y[t], -5, 5)
        return y

    fpga_feats = []
    gpu_feats = []
    targets = []

    for trial in range(N_TRIALS):
        if trial % 20 == 0:
            print(f"  Trial {trial}/{N_TRIALS}")

        # Random input
        u = rng.uniform(0, 0.5, size=N_STEPS).astype(np.float32)
        y_target = narma5(u, N_STEPS)
        # Target: final NARMA value
        targets.append(y_target[-1])

        # Input signal scaled to [-1, 1]
        input_signal = u * 2.0 - 0.5

        # GPU LIF
        vmem_g, spikes_g, adapt_g = run_gpu_lif(input_signal, mode=3)
        gpu_feats.append(gpu_snapshot_features(vmem_g, spikes_g, adapt_g))

        # FPGA COUPLED with 100 steps
        spk, vm = collect_fpga_extended(fpga, input_signal, "COUPLED",
                                         w_in, vg_base, N_STEPS)
        fpga_feats.append(temporal_features(spk, vm))

    X_gpu = np.array(gpu_feats)
    X_fpga = np.array(fpga_feats)
    X_combined = np.hstack([X_gpu, X_fpga])
    y = np.array(targets)

    r2_gpu = ridge_regress(X_gpu, y)
    r2_fpga = ridge_regress(X_fpga, y)
    r2_combined = ridge_regress(X_combined, y)

    print(f"  GPU R²: {r2_gpu:.4f}")
    print(f"  FPGA R²: {r2_fpga:.4f}")
    print(f"  COMBINED R²: {r2_combined:.4f}")

    results["exp4_narma5_r2_gpu"] = r2_gpu
    results["exp4_narma5_r2_fpga"] = r2_fpga
    results["exp4_narma5_r2_combined"] = r2_combined

    # T1055: any R² > 0 (first positive regression!)
    best_r2 = max(r2_gpu, r2_fpga, r2_combined)
    results["T1055_narma5_any_r2_gt_0"] = "PASS" if best_r2 > 0.0 else "FAIL"
    print(f"  T1055 best_r2({best_r2:.4f})>0: {results['T1055_narma5_any_r2_gt_0']}")

    # T1056: FPGA R² > -0.5 (not catastrophic)
    results["T1056_fpga_r2_not_catastrophic"] = "PASS" if r2_fpga > -0.5 else "FAIL"
    print(f"  T1056 fpga_r2({r2_fpga:.4f})>-0.5: {results['T1056_fpga_r2_not_catastrophic']}")

    # T1057: COMBINED ≥ best single
    single_best = max(r2_gpu, r2_fpga)
    results["T1057_combined_ge_single"] = "PASS" if r2_combined >= single_best - 0.01 else "FAIL"
    print(f"  T1057 combined({r2_combined:.4f})>=single({single_best:.4f}): {results['T1057_combined_ge_single']}")


# ==========================================================================
# EXP 5: Sequence Order Discrimination
# ==========================================================================
def run_exp5_sequence_order(fpga, vg_base, w_in, results):
    """Discriminate sequences by their ORDER of events.

    All sequences contain the same elements but in different orders.
    E.g., [A,B,C] vs [C,A,B] vs [B,C,A] — same histogram, different temporal structure.
    IMPOSSIBLE with snapshot features alone.
    """
    print("\n=== EXP 5: Sequence Order Discrimination ===")
    rng = np.random.RandomState(46)

    N_STEPS = 30  # 3 segments × 10 steps
    N_TRIALS = 120
    SEGMENT = 10
    # 3 amplitude levels
    AMPS = [0.2, 0.6, 1.0]
    # 6 permutations of 3 elements → 6 classes
    from itertools import permutations
    perms = list(permutations(range(3)))  # 6 orderings
    N_CLASSES = len(perms)

    fpga_feats_c = []
    fpga_feats_snap = []  # snapshot-only features for comparison
    gpu_feats = []
    labels = []

    for trial in range(N_TRIALS):
        if trial % 40 == 0:
            print(f"  Trial {trial}/{N_TRIALS}")

        cls = trial % N_CLASSES
        perm = perms[cls]

        # Build signal: 3 segments with amplitudes in permuted order
        input_signal = np.zeros(N_STEPS, dtype=np.float32)
        for seg_i, amp_idx in enumerate(perm):
            start = seg_i * SEGMENT
            end = start + SEGMENT
            t_seg = np.linspace(0, 1, SEGMENT)
            input_signal[start:end] = AMPS[amp_idx] * np.sin(2 * np.pi * 2 * t_seg)

        labels.append(cls)

        # GPU
        vmem_g, spikes_g, adapt_g = run_gpu_lif(input_signal, mode=3)
        gpu_feats.append(gpu_snapshot_features(vmem_g, spikes_g, adapt_g))

        # FPGA COUPLED — temporal features
        spk, vm = collect_fpga_extended(fpga, input_signal, "COUPLED",
                                         w_in, vg_base, N_STEPS)
        fpga_feats_c.append(temporal_features(spk, vm))

        # FPGA — snapshot only (mean, std — no temporal info)
        snap = np.concatenate([spk.mean(axis=0), spk.std(axis=0),
                               vm.mean(axis=0), vm.std(axis=0)])
        fpga_feats_snap.append(snap)

    y = np.array(labels)
    X_gpu = np.array(gpu_feats)
    X_fpga_t = np.array(fpga_feats_c)
    X_fpga_s = np.array(fpga_feats_snap)
    X_combined = np.hstack([X_gpu, X_fpga_t])

    chance = 1.0 / N_CLASSES

    acc_gpu, _ = ridge_classify(X_gpu, y)
    acc_fpga_t, _ = ridge_classify(X_fpga_t, y)
    acc_fpga_s, _ = ridge_classify(X_fpga_s, y)
    acc_combined, _ = ridge_classify(X_combined, y)

    print(f"  GPU: {acc_gpu:.3f}")
    print(f"  FPGA temporal: {acc_fpga_t:.3f}")
    print(f"  FPGA snapshot: {acc_fpga_s:.3f}")
    print(f"  COMBINED: {acc_combined:.3f}")
    print(f"  Chance: {chance:.3f}")

    results["exp5_gpu_order"] = acc_gpu
    results["exp5_fpga_temporal_order"] = acc_fpga_t
    results["exp5_fpga_snapshot_order"] = acc_fpga_s
    results["exp5_combined_order"] = acc_combined

    # T1058: FPGA temporal > FPGA snapshot (temporal features add value)
    results["T1058_temporal_gt_snapshot"] = "PASS" if acc_fpga_t > acc_fpga_s else "FAIL"
    print(f"  T1058 temporal({acc_fpga_t:.3f})>snapshot({acc_fpga_s:.3f}): {results['T1058_temporal_gt_snapshot']}")

    # T1059: any > chance (1/6)
    best = max(acc_gpu, acc_fpga_t, acc_combined)
    results["T1059_order_gt_chance"] = "PASS" if best > chance + 0.05 else "FAIL"
    print(f"  T1059 best({best:.3f})>chance+0.05({chance+0.05:.3f}): {results['T1059_order_gt_chance']}")

    # T1060: COMBINED > GPU alone
    results["T1060_combined_gt_gpu_order"] = "PASS" if acc_combined > acc_gpu else "FAIL"
    print(f"  T1060 COMBINED({acc_combined:.3f})>GPU({acc_gpu:.3f}): {results['T1060_combined_gt_gpu_order']}")


# ==========================================================================
# EXP 6: Sustained vs Transient Response
# ==========================================================================
def run_exp6_sustained_transient(fpga, vg_base, w_in, results):
    """Classify signals that differ ONLY in duration, not amplitude/frequency.

    - SHORT: 5-step pulse then silence (25 steps total)
    - MEDIUM: 12-step pulse then silence
    - LONG: 20-step pulse then silence
    - CONTINUOUS: pulse for all 25 steps

    Same amplitude, same frequency. Only temporal extent differs.
    After the stimulus ends, the READOUT happens at step 25 — must remember.
    """
    print("\n=== EXP 6: Sustained vs Transient ===")
    rng = np.random.RandomState(47)

    N_TOTAL = 30  # stimulus + readout delay
    N_TRIALS = 160
    N_CLASSES = 4
    durations = [5, 12, 20, 30]

    fpga_feats = []
    gpu_feats = []
    labels = []

    for trial in range(N_TRIALS):
        if trial % 40 == 0:
            print(f"  Trial {trial}/{N_TRIALS}")

        cls = trial % N_CLASSES
        dur = durations[cls]

        # Build signal: sine pulse of length `dur`, then silence
        input_signal = np.zeros(N_TOTAL, dtype=np.float32)
        t_stim = np.arange(dur) * 0.05
        input_signal[:dur] = 0.8 * np.sin(2 * np.pi * 3.0 * t_stim)

        labels.append(cls)

        # GPU
        vmem_g, spikes_g, adapt_g = run_gpu_lif(input_signal, mode=3)
        gpu_feats.append(gpu_snapshot_features(vmem_g, spikes_g, adapt_g))

        # FPGA COUPLED
        spk, vm = collect_fpga_extended(fpga, input_signal, "COUPLED",
                                         w_in, vg_base, N_TOTAL)
        fpga_feats.append(temporal_features(spk, vm))

    y = np.array(labels)
    X_gpu = np.array(gpu_feats)
    X_fpga = np.array(fpga_feats)
    X_combined = np.hstack([X_gpu, X_fpga])

    acc_gpu, _ = ridge_classify(X_gpu, y)
    acc_fpga, _ = ridge_classify(X_fpga, y)
    acc_combined, _ = ridge_classify(X_combined, y)

    print(f"  GPU: {acc_gpu:.3f}")
    print(f"  FPGA: {acc_fpga:.3f}")
    print(f"  COMBINED: {acc_combined:.3f}")

    results["exp6_gpu_duration"] = acc_gpu
    results["exp6_fpga_duration"] = acc_fpga
    results["exp6_combined_duration"] = acc_combined

    # T1061: FPGA > chance (0.25)
    results["T1061_duration_fpga_gt_chance"] = "PASS" if acc_fpga > 0.30 else "FAIL"
    print(f"  T1061 FPGA({acc_fpga:.3f})>0.30: {results['T1061_duration_fpga_gt_chance']}")

    # T1062: COMBINED > 0.40
    results["T1062_combined_gt_40"] = "PASS" if acc_combined > 0.40 else "FAIL"
    print(f"  T1062 COMBINED({acc_combined:.3f})>0.40: {results['T1062_combined_gt_40']}")

    # T1063: FPGA > GPU on this task (GPU should struggle with duration)
    results["T1063_fpga_gt_gpu_duration"] = "PASS" if acc_fpga > acc_gpu else "FAIL"
    print(f"  T1063 FPGA({acc_fpga:.3f})>GPU({acc_gpu:.3f}): {results['T1063_fpga_gt_gpu_duration']}")


# ==========================================================================
# EXP 7: Cross-Substrate Temporal Integration
# ==========================================================================
def run_exp7_cross_temporal(fpga, vg_base, w_in, results):
    """Final test: tasks where cross-substrate integration is essential.

    Stimulus: two-part signal. Part 1 encodes a parameter (via amplitude),
    Part 2 encodes another (via frequency). Classification depends on BOTH.

    With gap: 10-step silence between parts. Must remember part 1 while processing part 2.
    """
    print("\n=== EXP 7: Cross-Substrate Temporal Integration ===")
    rng = np.random.RandomState(48)

    N_STEPS = 40  # part1(10) + gap(10) + part2(10) + readout(10)
    N_TRIALS = 160
    # 2 amplitudes × 2 frequencies = 4 classes
    AMPS = [0.3, 0.9]
    FREQS = [1.0, 4.0]
    N_CLASSES = 4

    fpga_feats = []
    gpu_feats = []
    labels = []

    for trial in range(N_TRIALS):
        if trial % 40 == 0:
            print(f"  Trial {trial}/{N_TRIALS}")

        cls = trial % N_CLASSES
        amp_idx = cls // 2
        freq_idx = cls % 2

        input_signal = np.zeros(N_STEPS, dtype=np.float32)
        # Part 1: amplitude cue (steps 0-9)
        t1 = np.arange(10) * 0.05
        input_signal[:10] = AMPS[amp_idx] * np.sin(2 * np.pi * 2.0 * t1)
        # Gap: silence (steps 10-19) — already zeros
        # Part 2: frequency cue (steps 20-29)
        t2 = np.arange(10) * 0.05
        input_signal[20:30] = 0.5 * np.sin(2 * np.pi * FREQS[freq_idx] * t2)
        # Readout delay (steps 30-39) — zeros

        labels.append(cls)

        # GPU
        vmem_g, spikes_g, adapt_g = run_gpu_lif(input_signal, mode=3)
        gpu_feats.append(gpu_snapshot_features(vmem_g, spikes_g, adapt_g))

        # FPGA COUPLED
        spk, vm = collect_fpga_extended(fpga, input_signal, "COUPLED",
                                         w_in, vg_base, N_STEPS)
        fpga_feats.append(temporal_features(spk, vm))

    y = np.array(labels)
    X_gpu = np.array(gpu_feats)
    X_fpga = np.array(fpga_feats)
    X_combined = np.hstack([X_gpu, X_fpga])

    acc_gpu, _ = ridge_classify(X_gpu, y)
    acc_fpga, _ = ridge_classify(X_fpga, y)
    acc_combined, _ = ridge_classify(X_combined, y)

    print(f"  GPU: {acc_gpu:.3f}")
    print(f"  FPGA: {acc_fpga:.3f}")
    print(f"  COMBINED: {acc_combined:.3f}")

    results["exp7_gpu_integration"] = acc_gpu
    results["exp7_fpga_integration"] = acc_fpga
    results["exp7_combined_integration"] = acc_combined

    # T1064: COMBINED > 0.50 (must integrate both parts)
    results["T1064_integration_gt_50"] = "PASS" if acc_combined > 0.50 else "FAIL"
    print(f"  T1064 COMBINED({acc_combined:.3f})>0.50: {results['T1064_integration_gt_50']}")

    # T1065: COMBINED > max(GPU, FPGA) — synergy
    single_best = max(acc_gpu, acc_fpga)
    results["T1065_integration_synergy"] = "PASS" if acc_combined > single_best else "FAIL"
    print(f"  T1065 COMBINED({acc_combined:.3f})>max({single_best:.3f}): {results['T1065_integration_synergy']}")


# ==========================================================================
# MAIN
# ==========================================================================
def main():
    print("=" * 70)
    print("z2243: Temporal Challenge Suite")
    print("=" * 70)
    results = {"experiment": "z2243_temporal_challenge",
               "timestamp": datetime.now().isoformat()}

    # Check HIP binary exists (compiled by z2242)
    if not os.path.exists(HIP_BIN):
        print("  HIP binary not found, compiling...")
        # Inline compile
        hip_src = HIP_SRC
        if not os.path.exists(hip_src):
            print("  ERROR: HIP source not found. Run z2242 first.")
            return
        r = subprocess.run(
            ["hipcc", "--offload-arch=gfx1100", "-O2", "-o", HIP_BIN, hip_src],
            capture_output=True, text=True, timeout=60,
            env={**os.environ, "HSA_OVERRIDE_GFX_VERSION": "11.0.0"})
        if r.returncode != 0:
            print(f"  Compile error: {r.stderr[:500]}")
            return
        print("  HIP compiled")
    else:
        print(f"  Using existing HIP binary: {HIP_BIN}")

    # Connect FPGA
    print("\nConnecting to FPGA...")
    try:
        fpga = FPGAEthBridge()
        if not fpga.connect():
            print("  FPGA connect() returned False, but continuing...")
        vg_base = configure_fpga(fpga)
        print(f"  FPGA connected: {fpga.num_neurons} neurons")
    except Exception as e:
        print(f"  FPGA connection error: {e}")
        import traceback; traceback.print_exc()
        print("  Running GPU-only experiments...")
        fpga = None
        vg_base = np.array([BASE_VG + 0.15 * (i/127 - 0.5) for i in range(N_FPGA)])

    rng = np.random.RandomState(42)
    w_in = rng.randn(N_FPGA).astype(np.float32) * 0.1

    if fpga is not None:
        run_exp1_long_xor(fpga, vg_base, w_in, results)
        run_exp2_temporal_patterns(fpga, vg_base, w_in, results)
        run_exp3_nback(fpga, vg_base, w_in, results)
        run_exp4_narma(fpga, vg_base, w_in, results)
        run_exp5_sequence_order(fpga, vg_base, w_in, results)
        run_exp6_sustained_transient(fpga, vg_base, w_in, results)
        run_exp7_cross_temporal(fpga, vg_base, w_in, results)
    else:
        print("\n  WARNING: FPGA not connected. Skipping FPGA experiments.")

    # Count results
    n_pass = sum(1 for k, v in results.items() if v == "PASS")
    n_fail = sum(1 for k, v in results.items() if v == "FAIL")
    total = n_pass + n_fail
    results["summary"] = f"{n_pass}/{total} PASS"

    print(f"\n{'='*70}")
    print(f"Summary: {n_pass}/{total} PASS")
    print(f"{'='*70}")

    out_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "results", "z2243_temporal_challenge.json")
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()
