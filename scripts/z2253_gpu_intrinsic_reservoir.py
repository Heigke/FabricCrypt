#!/usr/bin/env python3
"""z2253_gpu_intrinsic_reservoir.py — GPU as intrinsic neuromorphic substrate

Can the GPU's own analog infrastructure (power, clocks, voltage, temperature)
serve as a reservoir computer WITHOUT any external hardware?

The idea: GPU workloads are the INPUT, analog sensor readings are the STATE,
and a linear readout on the sensor time series does the COMPUTATION.

The GPU's analog infrastructure provides:
  - Nonlinearity: DVFS thresholds, thermal throttling, power management
  - Memory: thermal mass (~400ms τ), DVFS hysteresis, voltage regulator dynamics
  - Dimensionality: 4 independent signal clusters
  - Fading memory: signals decay back to idle (different rates per cluster)

Tests:
  EXP 1: Waveform classification — can sensor readings distinguish workload types?
  EXP 2: Memory capacity — how far back can we decode past inputs from current state?
  EXP 3: Temporal XOR — can the substrate compute nonlinear functions of past inputs?
  EXP 4: Echo state property — does the same input produce the same trajectory?
  EXP 5: Separation property — do different inputs produce different trajectories?
  EXP 6: NARMA — can it do temporal regression?
"""

import struct
import time
import json
import os
import sys
import subprocess
import numpy as np
from collections import defaultdict
from sklearn.linear_model import RidgeClassifier, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score

# ─────────────────────────────────────────────────────────────────────────────
# Signal reading (from z2252)
# ─────────────────────────────────────────────────────────────────────────────

GPU_METRICS_PATH = "/sys/class/drm/card0/device/gpu_metrics"
HWMON_PATH = "/sys/class/drm/card0/device/hwmon/hwmon7"

# The 14 dynamic signals from z2252
SIGNAL_DEFS = [
    (4,   "temp_gfx"),
    (6,   "temp_soc"),
    (94,  "gfxclk"),
    (96,  "socclk"),
    (104, "socket_power"),
    (106, "gfx_power"),
    (108, "soc_power"),
    (112, "soc_voltage"),
    (132, "coreclk"),
    (136, "gfxclk_2"),
    (138, "socclk_2"),
    (168, "throttle"),
    (182, "vclk"),
    (190, "lclk"),
]

HWMON_PATHS = {
    "power_avg":  f"{HWMON_PATH}/power1_average",
    "power_inst": f"{HWMON_PATH}/power1_input",
    "temp_edge":  f"{HWMON_PATH}/temp1_input",
}


def read_state():
    """Read all analog signals, return numpy array."""
    vals = []
    try:
        with open(GPU_METRICS_PATH, "rb") as f:
            data = f.read()
        for offset, name in SIGNAL_DEFS:
            if offset + 2 <= len(data):
                vals.append(struct.unpack_from("<H", data, offset)[0])
            else:
                vals.append(0)
    except:
        vals = [0] * len(SIGNAL_DEFS)

    for name, path in HWMON_PATHS.items():
        try:
            with open(path) as f:
                vals.append(int(f.read().strip()))
        except:
            vals.append(0)

    return np.array(vals, dtype=np.float64)


SIGNAL_NAMES = [n for _, n in SIGNAL_DEFS] + list(HWMON_PATHS.keys())
N_SIGNALS = len(SIGNAL_NAMES)


# ─────────────────────────────────────────────────────────────────────────────
# Workload injection — this is our "input" to the reservoir
# ─────────────────────────────────────────────────────────────────────────────

def make_workload_script(workload_type, duration):
    """Generate a Python script string for a GPU workload."""
    return f'''
import torch, time, os
os.environ["HSA_OVERRIDE_GFX_VERSION"] = "11.0.0"
d = torch.device("cuda")
t0 = time.monotonic()

if "{workload_type}" == "matmul_small":
    a = torch.randn(256, 256, device=d)
    while time.monotonic() - t0 < {duration}:
        a = a @ a; a = a / a.norm()
        torch.cuda.synchronize()
elif "{workload_type}" == "matmul_large":
    a = torch.randn(2048, 2048, device=d)
    while time.monotonic() - t0 < {duration}:
        a = a @ a; a = a / a.norm()
        torch.cuda.synchronize()
elif "{workload_type}" == "memory":
    a = torch.randn(16*1024*1024, device=d)
    b = torch.empty_like(a)
    while time.monotonic() - t0 < {duration}:
        b.copy_(a); a.copy_(b)
        torch.cuda.synchronize()
elif "{workload_type}" == "fft":
    a = torch.randn(1024, 1024, device=d, dtype=torch.complex64)
    while time.monotonic() - t0 < {duration}:
        b = torch.fft.fft2(a)
        torch.cuda.synchronize()
elif "{workload_type}" == "conv":
    inp = torch.randn(8, 64, 128, 128, device=d)
    w = torch.randn(128, 64, 3, 3, device=d)
    while time.monotonic() - t0 < {duration}:
        out = torch.nn.functional.conv2d(inp, w, padding=1)
        torch.cuda.synchronize()
elif "{workload_type}" == "reduce":
    a = torch.randn(16*1024*1024, device=d)
    while time.monotonic() - t0 < {duration}:
        s = a.sum(); v = a.var()
        torch.cuda.synchronize()
elif "{workload_type}" == "idle":
    time.sleep({duration})
'''


def inject_workload(workload_type, duration):
    """Start workload, return subprocess handle."""
    script = make_workload_script(workload_type, duration)
    env = os.environ.copy()
    env["HSA_OVERRIDE_GFX_VERSION"] = "11.0.0"
    return subprocess.Popen(
        [sys.executable, "-c", script],
        env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )


def collect_trajectory(workload_type, duration, sample_hz=100):
    """Inject workload and collect state trajectory."""
    proc = inject_workload(workload_type, duration + 0.5)
    time.sleep(0.3)  # let workload start

    states = []
    interval = 1.0 / sample_hz
    t0 = time.monotonic()
    while time.monotonic() - t0 < duration:
        t_s = time.monotonic()
        states.append(read_state())
        elapsed = time.monotonic() - t_s
        if elapsed < interval:
            time.sleep(interval - elapsed)

    proc.terminate()
    proc.wait()
    return np.array(states)  # shape: (n_samples, n_signals)


# ─────────────────────────────────────────────────────────────────────────────
# Feature extraction from state trajectories
# ─────────────────────────────────────────────────────────────────────────────

def extract_features(trajectory, window=None):
    """Extract features from a state trajectory window.
    Returns: mean, std, min, max, delta (first→last), slope for each signal.
    """
    if window is not None:
        traj = trajectory[window[0]:window[1]]
    else:
        traj = trajectory

    if len(traj) < 2:
        return np.zeros(N_SIGNALS * 6)

    feats = []
    for col in range(traj.shape[1]):
        s = traj[:, col]
        feats.extend([
            s.mean(),
            s.std(),
            s.min(),
            s.max(),
            s[-1] - s[0],  # delta
            np.polyfit(np.arange(len(s)), s, 1)[0] if len(s) > 2 else 0,  # slope
        ])
    return np.array(feats)


# ─────────────────────────────────────────────────────────────────────────────
# Experiments
# ─────────────────────────────────────────────────────────────────────────────

def exp1_classification(n_reps=8):
    """EXP 1: Can GPU analog state distinguish workload types?"""
    print("\n═══ EXP 1: Workload Classification from Analog State ═══", file=sys.stderr)

    classes = ["matmul_small", "matmul_large", "memory", "fft", "conv", "reduce"]
    duration = 3.0
    sample_hz = 100

    X, y = [], []
    for rep in range(n_reps):
        for ci, cls in enumerate(classes):
            print(f"  Rep {rep+1}/{n_reps}, class={cls}...", end="", file=sys.stderr, flush=True)
            traj = collect_trajectory(cls, duration, sample_hz)
            feats = extract_features(traj)
            X.append(feats)
            y.append(ci)
            print(f" {len(traj)} samples", file=sys.stderr)
            time.sleep(1.0)  # cooldown

    X = np.array(X)
    y = np.array(y)

    # Remove NaN/inf
    X = np.nan_to_num(X, nan=0, posinf=0, neginf=0)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    clf = RidgeClassifier(alpha=1.0)
    scores = cross_val_score(clf, X_scaled, y, cv=min(5, n_reps), scoring="accuracy")

    result = {
        "n_classes": len(classes),
        "classes": classes,
        "n_samples": len(y),
        "accuracy_mean": float(scores.mean()),
        "accuracy_std": float(scores.std()),
        "n_features": X.shape[1],
    }

    print(f"\n  Result: {scores.mean()*100:.1f}% ± {scores.std()*100:.1f}% "
          f"({len(classes)}-class, chance={100/len(classes):.1f}%)", file=sys.stderr)
    print(f"  PASS: {scores.mean() > 1.5/len(classes)}", file=sys.stderr)

    return result


def exp2_memory_capacity(n_reps=5):
    """EXP 2: Memory capacity — decode past input from current state.

    Inject a SEQUENCE of random workloads, then try to decode workload[t-d]
    from state[t]. How far back can we decode?
    """
    print("\n═══ EXP 2: Memory Capacity (Fading Memory) ═══", file=sys.stderr)

    workload_types = ["matmul_small", "matmul_large", "memory", "fft"]
    step_duration = 2.0  # seconds per step
    n_steps = 20
    sample_hz = 50

    all_features = []
    all_labels = []

    for rep in range(n_reps):
        print(f"  Rep {rep+1}/{n_reps}: ", end="", file=sys.stderr, flush=True)
        # Generate random sequence
        seq = [workload_types[i] for i in np.random.randint(0, len(workload_types), n_steps)]
        step_features = []
        step_labels = []

        for step, wl in enumerate(seq):
            traj = collect_trajectory(wl, step_duration, sample_hz)
            feats = extract_features(traj)
            step_features.append(feats)
            step_labels.append(workload_types.index(wl))
            print(f"{wl[0]}", end="", file=sys.stderr, flush=True)
            time.sleep(0.3)

        all_features.append(step_features)
        all_labels.append(step_labels)
        print("", file=sys.stderr)

    # Test memory at delays d=0..8
    mc_results = {}
    max_delay = 8
    for delay in range(max_delay + 1):
        X, y = [], []
        for rep in range(n_reps):
            for t in range(delay, n_steps):
                X.append(all_features[rep][t])
                y.append(all_labels[rep][t - delay])

        X = np.nan_to_num(np.array(X), nan=0, posinf=0, neginf=0)
        y = np.array(y)

        if len(X) < 10:
            continue

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        clf = RidgeClassifier(alpha=1.0)
        n_folds = min(5, len(X) // max(len(set(y)), 2))
        if n_folds < 2:
            continue
        scores = cross_val_score(clf, X_scaled, y, cv=n_folds, scoring="accuracy")

        mc_results[delay] = {
            "accuracy": float(scores.mean()),
            "std": float(scores.std()),
            "n_samples": len(y),
        }
        chance = 1.0 / len(workload_types)
        above = "█" if scores.mean() > chance * 1.5 else "░"
        print(f"  {above} delay={delay}: acc={scores.mean()*100:.1f}% ± {scores.std()*100:.1f}% "
              f"(chance={chance*100:.1f}%)", file=sys.stderr)

    # Memory capacity = sum of R² at each delay
    total_mc = sum(max(0, r["accuracy"] - 1.0/len(workload_types))
                   for r in mc_results.values())

    print(f"  Total memory capacity: {total_mc:.3f}", file=sys.stderr)

    return {"delays": mc_results, "total_mc": total_mc, "n_classes": len(workload_types)}


def exp3_temporal_xor(n_reps=5):
    """EXP 3: Temporal XOR — nonlinear computation on past inputs.

    Input: binary sequence (matmul_small=0, matmul_large=1)
    Target: XOR of input[t] and input[t-1]
    This requires BOTH memory AND nonlinearity.
    """
    print("\n═══ EXP 3: Temporal XOR ═══", file=sys.stderr)

    types = ["matmul_small", "matmul_large"]
    step_duration = 2.0
    n_steps = 24
    sample_hz = 50

    all_features = []
    all_inputs = []

    for rep in range(n_reps):
        print(f"  Rep {rep+1}/{n_reps}: ", end="", file=sys.stderr, flush=True)
        inputs = np.random.randint(0, 2, n_steps)
        step_features = []

        for step, inp in enumerate(inputs):
            traj = collect_trajectory(types[inp], step_duration, sample_hz)
            feats = extract_features(traj)
            step_features.append(feats)
            print(f"{inp}", end="", file=sys.stderr, flush=True)
            time.sleep(0.3)

        all_features.append(step_features)
        all_inputs.append(inputs)
        print("", file=sys.stderr)

    # XOR at different delays
    xor_results = {}
    for tau in [1, 2, 3]:
        X, y = [], []
        for rep in range(n_reps):
            for t in range(tau, n_steps):
                X.append(all_features[rep][t])
                y.append(int(all_inputs[rep][t] ^ all_inputs[rep][t - tau]))

        X = np.nan_to_num(np.array(X), nan=0, posinf=0, neginf=0)
        y = np.array(y)

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        clf = RidgeClassifier(alpha=1.0)
        n_folds = min(5, len(X) // 4)
        if n_folds < 2:
            continue
        scores = cross_val_score(clf, X_scaled, y, cv=n_folds, scoring="accuracy")

        xor_results[tau] = {
            "accuracy": float(scores.mean()),
            "std": float(scores.std()),
        }
        above = "█" if scores.mean() > 0.6 else "░"
        print(f"  {above} XOR(t, t-{tau}): acc={scores.mean()*100:.1f}% ± {scores.std()*100:.1f}% "
              f"(chance=50%)", file=sys.stderr)

    return xor_results


def exp4_echo_state(n_reps=3):
    """EXP 4: Echo State Property — same input → same trajectory?"""
    print("\n═══ EXP 4: Echo State Property (Reproducibility) ═══", file=sys.stderr)

    workloads = ["matmul_small", "matmul_large", "memory", "fft"]
    duration = 3.0
    sample_hz = 50

    results = {}
    for wl in workloads:
        trajectories = []
        for rep in range(n_reps):
            print(f"  {wl} rep {rep+1}/{n_reps}...", file=sys.stderr, flush=True)
            # Cool down first
            time.sleep(2.0)
            traj = collect_trajectory(wl, duration, sample_hz)
            trajectories.append(traj)

        # Compare trajectories pairwise (Pearson correlation per signal)
        corrs = []
        for i in range(n_reps):
            for j in range(i + 1, n_reps):
                min_len = min(len(trajectories[i]), len(trajectories[j]))
                t1 = trajectories[i][:min_len]
                t2 = trajectories[j][:min_len]
                # Per-signal correlation, averaged
                signal_corrs = []
                for s in range(N_SIGNALS):
                    if t1[:, s].std() > 0 and t2[:, s].std() > 0:
                        r = np.corrcoef(t1[:, s], t2[:, s])[0, 1]
                        if not np.isnan(r):
                            signal_corrs.append(r)
                if signal_corrs:
                    corrs.append(np.mean(signal_corrs))

        mean_corr = np.mean(corrs) if corrs else 0
        results[wl] = {
            "mean_reproducibility": float(mean_corr),
            "n_comparisons": len(corrs),
        }
        above = "█" if mean_corr > 0.7 else "░"
        print(f"  {above} {wl}: reproducibility r={mean_corr:.3f}", file=sys.stderr)

    return results


def exp5_separation(n_reps=5):
    """EXP 5: Separation Property — different inputs → different states?"""
    print("\n═══ EXP 5: Separation Property ═══", file=sys.stderr)

    workloads = ["matmul_small", "matmul_large", "memory", "fft", "conv", "reduce"]
    duration = 3.0
    sample_hz = 50

    # Collect feature vectors for each workload
    features = defaultdict(list)
    for rep in range(n_reps):
        for wl in workloads:
            print(f"  Rep {rep+1}/{n_reps} {wl}...", end="", file=sys.stderr, flush=True)
            traj = collect_trajectory(wl, duration, sample_hz)
            feats = extract_features(traj)
            features[wl].append(feats)
            print(f" ok", file=sys.stderr)
            time.sleep(0.5)

    # Compute pairwise separation (distance between class centroids / within-class spread)
    centroids = {}
    spreads = {}
    for wl in workloads:
        f = np.nan_to_num(np.array(features[wl]), nan=0, posinf=0, neginf=0)
        centroids[wl] = f.mean(axis=0)
        spreads[wl] = np.linalg.norm(f.std(axis=0))

    separations = {}
    for i, w1 in enumerate(workloads):
        for j, w2 in enumerate(workloads):
            if i >= j:
                continue
            dist = np.linalg.norm(centroids[w1] - centroids[w2])
            spread = (spreads[w1] + spreads[w2]) / 2
            sep = dist / spread if spread > 0 else 0
            separations[f"{w1}_vs_{w2}"] = float(sep)
            level = "██" if sep > 2 else "░░" if sep > 1 else "  "
            print(f"  {level} {w1:15s} vs {w2:15s}: separation={sep:.2f}", file=sys.stderr)

    mean_sep = np.mean(list(separations.values()))
    print(f"\n  Mean separation: {mean_sep:.2f} (>1 = distinguishable, >2 = well separated)", file=sys.stderr)

    return {"pairs": separations, "mean_separation": float(mean_sep)}


def exp6_narma(order=3, n_steps=40, n_reps=3):
    """EXP 6: NARMA-like temporal regression.

    Input: random workload intensity (matmul size)
    Target: nonlinear function of past inputs
    y(t) = 0.3*y(t-1) + 0.05*y(t-1)*sum(y(t-1..t-order)) + 1.5*u(t-1)*u(t-order) + 0.1
    """
    print(f"\n═══ EXP 6: NARMA-{order} Temporal Regression ═══", file=sys.stderr)

    sizes = [256, 512, 1024, 2048, 4096]
    step_duration = 1.5
    sample_hz = 50

    all_features = []
    all_targets = []

    for rep in range(n_reps):
        print(f"  Rep {rep+1}/{n_reps}: ", end="", file=sys.stderr, flush=True)

        # Random input sequence (normalized to [0,1])
        u = np.random.rand(n_steps)
        sz_indices = (u * (len(sizes) - 1)).astype(int)

        # Generate NARMA target
        y_target = np.zeros(n_steps)
        for t in range(order, n_steps):
            y_sum = sum(y_target[t-k-1] for k in range(order))
            y_target[t] = (0.3 * y_target[t-1]
                          + 0.05 * y_target[t-1] * y_sum
                          + 1.5 * u[t-1] * u[t-order]
                          + 0.1)
            y_target[t] = np.tanh(y_target[t])  # bound

        # Collect state for each step
        step_features = []
        for step in range(n_steps):
            sz = sizes[sz_indices[step]]
            wl_code = f"""
import torch, time, os
os.environ["HSA_OVERRIDE_GFX_VERSION"] = "11.0.0"
d = torch.device("cuda")
a = torch.randn({sz}, {sz}, device=d)
t0 = time.monotonic()
while time.monotonic() - t0 < {step_duration}:
    a = a @ a; a = a / a.norm()
    torch.cuda.synchronize()
"""
            env = os.environ.copy()
            env["HSA_OVERRIDE_GFX_VERSION"] = "11.0.0"
            proc = subprocess.Popen([sys.executable, "-c", wl_code],
                                    env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            time.sleep(0.3)
            traj = collect_trajectory.__wrapped__(step_duration - 0.3, sample_hz) if hasattr(collect_trajectory, '__wrapped__') else None

            # Simplified: just read state during workload
            states = []
            t0 = time.monotonic()
            while time.monotonic() - t0 < step_duration - 0.3:
                states.append(read_state())
                time.sleep(1.0 / sample_hz)

            proc.terminate()
            proc.wait()

            if states:
                traj = np.array(states)
                feats = extract_features(traj)
            else:
                feats = np.zeros(N_SIGNALS * 6)

            step_features.append(feats)
            print(f"{sz_indices[step]}", end="", file=sys.stderr, flush=True)
            time.sleep(0.2)

        all_features.append(step_features)
        all_targets.append(y_target)
        print("", file=sys.stderr)

    # Regression: predict y_target from state features
    X, y = [], []
    for rep in range(n_reps):
        for t in range(order, n_steps):
            X.append(all_features[rep][t])
            y.append(all_targets[rep][t])

    X = np.nan_to_num(np.array(X), nan=0, posinf=0, neginf=0)
    y = np.array(y)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    reg = Ridge(alpha=1.0)
    scores = cross_val_score(reg, X_scaled, y, cv=min(5, len(X) // 4), scoring="r2")

    result = {
        "r2_mean": float(scores.mean()),
        "r2_std": float(scores.std()),
        "n_samples": len(y),
        "order": order,
    }
    above = "█" if scores.mean() > 0.1 else "░"
    print(f"  {above} NARMA-{order} R²={scores.mean():.3f} ± {scores.std():.3f}", file=sys.stderr)

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    results = {
        "experiment": "z2253_gpu_intrinsic_reservoir",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "gpu": "gfx1151 (Radeon 8060S)",
        "n_signals": N_SIGNALS,
        "signal_names": SIGNAL_NAMES,
    }

    print("═══ z2253: GPU Intrinsic Neuromorphic Substrate ═══", file=sys.stderr)
    print(f"  {N_SIGNALS} analog signals, reading at ~100Hz", file=sys.stderr)
    print(f"  Question: Can the GPU's own analog infrastructure compute?", file=sys.stderr)

    # Run experiments
    results["exp1_classification"] = exp1_classification(n_reps=8)
    results["exp4_echo_state"] = exp4_echo_state(n_reps=3)
    results["exp5_separation"] = exp5_separation(n_reps=5)
    results["exp2_memory_capacity"] = exp2_memory_capacity(n_reps=4)
    results["exp3_temporal_xor"] = exp3_temporal_xor(n_reps=4)
    results["exp6_narma"] = exp6_narma(order=3, n_steps=30, n_reps=3)

    # Summary
    print("\n═══ SUMMARY ═══", file=sys.stderr)
    cls_acc = results["exp1_classification"]["accuracy_mean"]
    mc = results["exp2_memory_capacity"]["total_mc"]
    xor1 = results["exp3_temporal_xor"].get(1, {}).get("accuracy", 0)
    echo = np.mean([v["mean_reproducibility"] for v in results["exp4_echo_state"].values()])
    sep = results["exp5_separation"]["mean_separation"]
    narma = results["exp6_narma"]["r2_mean"]

    print(f"  Classification (6-class): {cls_acc*100:.1f}% (chance=16.7%)", file=sys.stderr)
    print(f"  Memory capacity:          {mc:.3f}", file=sys.stderr)
    print(f"  Temporal XOR(t,t-1):      {xor1*100:.1f}% (chance=50%)", file=sys.stderr)
    print(f"  Echo state (reprod.):      {echo:.3f}", file=sys.stderr)
    print(f"  Separation:               {sep:.2f}", file=sys.stderr)
    print(f"  NARMA-3 R²:               {narma:.3f}", file=sys.stderr)

    gpu_is_reservoir = (cls_acc > 0.5 and mc > 0.5 and xor1 > 0.55)
    print(f"\n  GPU IS intrinsic reservoir: {'YES' if gpu_is_reservoir else 'NOT YET'}", file=sys.stderr)

    results["summary"] = {
        "classification_acc": cls_acc,
        "memory_capacity": mc,
        "xor_acc": xor1,
        "echo_reproducibility": echo,
        "separation": sep,
        "narma_r2": narma,
        "is_reservoir": gpu_is_reservoir,
    }

    out_path = "results/z2253_gpu_intrinsic_reservoir.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Results saved to {out_path}", file=sys.stderr)
    print("Done.", file=sys.stderr)


if __name__ == "__main__":
    main()
