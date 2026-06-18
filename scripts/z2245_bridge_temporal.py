#!/usr/bin/env python3
"""
z2245_bridge_temporal.py — Temporal experiments with VG modulation at ~50Hz
===========================================================================
Uses FPGAEthBridge with real-time VG modulation (input-dependent gate voltage).
Tests whether FPGA fading memory provides measurable advantage for temporal tasks.

Tests T1076-T1095 (20 tests):
  EXP 1: Waveform Classification with VG modulation (T1076-T1079)
  EXP 2: Long-Delay XOR (T1080-T1083)
  EXP 3: Temporal Pattern Discrimination (T1084-T1087)
  EXP 4: Memory Capacity via regression (T1088-T1091)
  EXP 5: NARMA-5 Regression (T1092-T1095)
"""

import os, sys, json, time, warnings
import numpy as np
from pathlib import Path
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline

warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings('ignore', category=ConvergenceWarning)

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from fpga_host_eth import FPGAEthBridge

RESULTS_FILE = ROOT / "results" / "z2245_bridge_temporal.json"

# Tuned parameters from z2239 (94.4% classification)
BASE_VG = 0.58
VG_SPREAD = 0.075
ALPHA = 0.25     # input -> Vg coupling
SAMPLE_HZ = 50   # telemetry polling rate


def generate_input_sequence(pattern, freq, n_steps, dt=0.02):
    """Generate input signal sequence."""
    t = np.arange(n_steps) * dt
    if pattern == 'sine':
        return 0.5 * np.sin(2 * np.pi * freq * t) + 0.5
    elif pattern == 'square':
        return (np.sin(2 * np.pi * freq * t) > 0).astype(float)
    elif pattern == 'triangle':
        return np.abs(2 * (t * freq % 1) - 1)
    elif pattern == 'noise':
        return np.random.rand(n_steps)
    elif pattern == 'binary':
        return np.random.randint(0, 2, n_steps).astype(float)
    else:
        return np.zeros(n_steps)


def configure_fpga(fpga):
    """Set heterogeneous Vg and initial state."""
    vg_base = np.array([BASE_VG + VG_SPREAD * ((i % 17) / 16.0 - 0.5)
                        for i in range(128)])
    fpga.set_vg_batch(0, vg_base.tolist())
    time.sleep(0.05)
    return vg_base


def collect_trial(fpga, vg_base, inputs):
    """Collect one trial: modulate VG with inputs, read FPGA state each step."""
    n_steps = len(inputs)
    spikes = np.zeros((n_steps, 128))
    vmems = np.zeros((n_steps, 128))
    dt = 1.0 / SAMPLE_HZ

    for step in range(n_steps):
        t0 = time.time()

        # Input-dependent VG modulation
        inp = inputs[step]
        vg_mod = vg_base + ALPHA * (inp - 0.5)
        vg_mod = np.clip(vg_mod, 0.0, 1.0)

        # Send VG batch
        fpga.set_vg_batch(0, vg_mod.tolist())

        # Read telemetry
        telem = fpga.read_telemetry(timeout=0.1)
        if telem and telem['spike_counts'] is not None:
            spikes[step] = telem['spike_counts']
            vmems[step] = telem['vmem']

        # Rate control
        elapsed = time.time() - t0
        if elapsed < dt:
            time.sleep(dt - elapsed)

    return spikes, vmems


def extract_features(spikes, vmems):
    """Extract temporal features from trial data."""
    n = len(spikes)
    mid = n // 2
    q1 = n // 4
    q3 = 3 * n // 4

    spike_early = spikes[:mid].mean(axis=0)
    spike_late = spikes[mid:].mean(axis=0)
    spike_delta = spike_late - spike_early

    vmem_early = vmems[:mid].mean(axis=0)
    vmem_late = vmems[mid:].mean(axis=0)
    vmem_delta = vmem_late - vmem_early

    vmem_var = vmems.var(axis=0)

    spike_q1 = spikes[:q1].mean(axis=0)
    spike_q3 = spikes[q3:].mean(axis=0)
    spike_trend = spike_q3 - spike_q1

    total_spikes = spikes.sum(axis=1)
    spk_slope = np.polyfit(np.arange(n), total_spikes, 1)[0] if n > 2 else 0
    vmem_mean_ts = vmems.mean(axis=1)
    vmem_slope = np.polyfit(np.arange(n), vmem_mean_ts, 1)[0] if n > 2 else 0

    return np.concatenate([
        spike_delta, vmem_delta, vmem_var, spike_trend,
        [spk_slope, vmem_slope],
    ])  # 514


def extract_static_features(spikes, vmems):
    """Extract only static (non-temporal) features."""
    return np.concatenate([spikes.mean(axis=0), vmems.mean(axis=0)])  # 256


def classify(X, y, n_components=30):
    """PCA + LogisticRegression classifier with CV."""
    if len(np.unique(y)) < 2 or len(y) < 10:
        return 0.0
    try:
        n_comp = min(n_components, X.shape[0] - 1, X.shape[1])
        pipe = Pipeline([
            ('scaler', StandardScaler()),
            ('pca', PCA(n_components=n_comp)),
            ('clf', LogisticRegression(max_iter=1000, C=1.0, solver='lbfgs')),
        ])
        scores = cross_val_score(pipe, X, y, cv=min(5, len(np.unique(y))),
                                 scoring='accuracy')
        return float(scores.mean())
    except Exception as e:
        print(f"    classify error: {e}")
        return 0.0


def exp1_classification(fpga, vg_base, results):
    """EXP 1: 4-class waveform classification."""
    print("\n=== EXP 1: Waveform Classification with VG Modulation ===")

    classes = [
        ('sine', 2.0),
        ('sine', 5.0),
        ('square', 2.0),
        ('triangle', 2.0),
    ]
    n_trials = 30
    n_steps = 60

    all_feat = {'temporal': [], 'static': []}
    all_labels = []

    for cls_idx, (pat, freq) in enumerate(classes):
        print(f"  Class {cls_idx} ({pat}_{freq}Hz): ", end="", flush=True)
        for trial in range(n_trials):
            inputs = generate_input_sequence(pat, freq, n_steps)
            spk, vm = collect_trial(fpga, vg_base, inputs)
            all_feat['temporal'].append(extract_features(spk, vm))
            all_feat['static'].append(extract_static_features(spk, vm))
            all_labels.append(cls_idx)
        print(f"{n_trials} done")

    y = np.array(all_labels)
    for cond in ['temporal', 'static']:
        X = np.array(all_feat[cond])
        acc = classify(X, y)
        results[f'exp1_{cond}_acc'] = acc
        print(f"  {cond}: {acc:.3f}")

    temp = results['exp1_temporal_acc']
    stat = results['exp1_static_acc']

    results['T1076_temporal_gt_50'] = "PASS" if temp > 0.50 else "FAIL"
    results['T1077_temporal_gt_80'] = "PASS" if temp > 0.80 else "FAIL"
    results['T1078_temporal_gt_chance'] = "PASS" if temp > 0.30 else "FAIL"
    results['T1079_temporal_gt_static'] = "PASS" if temp > stat else "FAIL"

    for t in ['T1076', 'T1077', 'T1078', 'T1079']:
        k = [k for k in results if k.startswith(t)][0]
        print(f"  {k}: {results[k]}")


def exp2_xor_delay(fpga, vg_base, results):
    """EXP 2: Long-delay XOR with binary stream."""
    print("\n=== EXP 2: Long-Delay XOR ===")

    delays = [1, 2, 5, 10]
    n_trials = 60
    n_steps = 40

    for delay in delays:
        print(f"\n  Delay={delay} ({delay * 20}ms)")
        feats = {'temporal': [], 'static': []}
        labels = []

        for trial in range(n_trials):
            if trial % 20 == 0:
                print(f"    Trial {trial}/{n_trials}")
            inputs = generate_input_sequence('binary', 1.0, n_steps)
            spk, vm = collect_trial(fpga, vg_base, inputs)

            if n_steps - 1 - delay < 0:
                continue
            target = int(inputs[-1]) ^ int(inputs[-1 - delay])
            feats['temporal'].append(extract_features(spk, vm))
            feats['static'].append(extract_static_features(spk, vm))
            labels.append(target)

        y = np.array(labels)
        for cond in ['temporal', 'static']:
            X = np.array(feats[cond])
            acc = classify(X, y, n_components=20)
            results[f'exp2_xor_d{delay}_{cond}'] = acc
            print(f"    d={delay} {cond}: {acc:.3f}")

    d1 = results.get('exp2_xor_d1_temporal', 0)
    d5 = results.get('exp2_xor_d5_temporal', 0)
    d10 = results.get('exp2_xor_d10_temporal', 0)
    d10_s = results.get('exp2_xor_d10_static', 0)

    results['T1080_xor_d1_gt_52'] = "PASS" if d1 > 0.52 else "FAIL"
    results['T1081_xor_d5_gt_52'] = "PASS" if d5 > 0.52 else "FAIL"
    results['T1082_xor_d10_gt_52'] = "PASS" if d10 > 0.52 else "FAIL"
    results['T1083_temporal_gt_static_d10'] = "PASS" if d10 > d10_s else "FAIL"

    for t in ['T1080', 'T1081', 'T1082', 'T1083']:
        k = [k for k in results if k.startswith(t)][0]
        print(f"  {k}: {results[k]}")


def exp3_temporal_discrimination(fpga, vg_base, results):
    """EXP 3: Discriminate temporal patterns."""
    print("\n=== EXP 3: Temporal Pattern Discrimination ===")

    patterns = [('sine', 2.0), ('square', 2.0), ('triangle', 2.0), ('noise', 1.0)]
    n_trials = 30
    n_steps = 60

    feats = {'temporal': [], 'static': []}
    labels = []

    for cls_idx, (pat, freq) in enumerate(patterns):
        print(f"  Class {cls_idx} ({pat}): ", end="", flush=True)
        for trial in range(n_trials):
            inputs = generate_input_sequence(pat, freq, n_steps)
            spk, vm = collect_trial(fpga, vg_base, inputs)
            feats['temporal'].append(extract_features(spk, vm))
            feats['static'].append(extract_static_features(spk, vm))
            labels.append(cls_idx)
        print(f"{n_trials} done")

    y = np.array(labels)
    for cond in ['temporal', 'static']:
        X = np.array(feats[cond])
        acc = classify(X, y)
        results[f'exp3_{cond}_acc'] = acc
        print(f"  {cond}: {acc:.3f}")

    temp = results['exp3_temporal_acc']
    stat = results['exp3_static_acc']

    results['T1084_temporal_gt_static'] = "PASS" if temp > stat + 0.02 else "FAIL"
    results['T1085_temporal_gt_70'] = "PASS" if temp > 0.70 else "FAIL"
    results['T1086_temporal_gt_50'] = "PASS" if temp > 0.50 else "FAIL"
    results['T1087_temporal_gt_chance'] = "PASS" if temp > 0.30 else "FAIL"

    for t in ['T1084', 'T1085', 'T1086', 'T1087']:
        k = [k for k in results if k.startswith(t)][0]
        print(f"  {k}: {results[k]}")


def exp4_memory_capacity(fpga, vg_base, results):
    """EXP 4: Memory capacity -- reconstruct delayed input."""
    print("\n=== EXP 4: Memory Capacity ===")

    n_steps = 300
    inputs = generate_input_sequence('noise', 1.0, n_steps)

    print(f"  Collecting {n_steps} steps ({n_steps/SAMPLE_HZ:.0f}s)...")
    spk, vm = collect_trial(fpga, vg_base, inputs)

    start = 50
    inputs_use = inputs[start:]
    spk_use = spk[start:]
    vm_use = vm[start:]
    N = len(inputs_use)

    delays = [1, 2, 5, 10, 15, 20]
    mc = {}

    for d in delays:
        if d + 10 >= N:
            continue
        y = inputs_use[:N - d]
        X = np.hstack([spk_use[d:][:len(y)], vm_use[d:][:len(y)]])

        try:
            pipe = Pipeline([
                ('scaler', StandardScaler()),
                ('ridge', Ridge(alpha=10.0)),
            ])
            scores = cross_val_score(pipe, X, y, cv=5, scoring='r2')
            r2 = float(max(scores.mean(), 0.0))
        except Exception:
            r2 = 0.0
        mc[d] = r2
        print(f"  d={d:2d} ({d*20}ms): R2={r2:.4f}")

    mc_total = sum(mc.values())
    results['exp4_mc_total'] = mc_total
    results['exp4_mc_details'] = {str(k): v for k, v in mc.items()}
    print(f"  MC total: {mc_total:.3f}")

    r2_d1 = mc.get(1, 0)
    results['T1088_mc_d1_gt_005'] = "PASS" if r2_d1 > 0.05 else "FAIL"
    results['T1089_mc_d5_gt_003'] = "PASS" if mc.get(5, 0) > 0.03 else "FAIL"
    results['T1090_mc_total_gt_020'] = "PASS" if mc_total > 0.20 else "FAIL"
    results['T1091_mc_decays'] = "PASS" if r2_d1 > mc.get(10, 0) else "FAIL"

    for t in ['T1088', 'T1089', 'T1090', 'T1091']:
        k = [k for k in results if k.startswith(t)][0]
        print(f"  {k}: {results[k]}")


def exp5_narma(fpga, vg_base, results):
    """EXP 5: NARMA-5 regression."""
    print("\n=== EXP 5: NARMA-5 Regression ===")

    n_steps = 300
    inputs = 0.5 * generate_input_sequence('noise', 1.0, n_steps)

    print(f"  Collecting {n_steps} steps...")
    spk, vm = collect_trial(fpga, vg_base, inputs)

    narma = np.zeros(n_steps)
    for t in range(5, n_steps):
        narma[t] = 0.3 * narma[t-1] + 0.05 * narma[t-1] * sum(narma[t-j] for j in range(1, 6)) \
                   + 1.5 * inputs[t-1] * inputs[t-5] + 0.1
        narma[t] = np.clip(narma[t], -10, 10)

    start = 50
    y = narma[start:]
    X = np.hstack([spk[start:], vm[start:]])

    try:
        pipe = Pipeline([('scaler', StandardScaler()), ('ridge', Ridge(alpha=10.0))])
        scores = cross_val_score(pipe, X, y, cv=5, scoring='r2')
        r2 = float(scores.mean())
    except Exception:
        r2 = -1.0
    results['exp5_narma5_r2'] = r2
    print(f"  NARMA-5 R2: {r2:.4f}")

    y_input = inputs[start:]
    try:
        pipe2 = Pipeline([('scaler', StandardScaler()), ('ridge', Ridge(alpha=10.0))])
        scores2 = cross_val_score(pipe2, X, y_input, cv=5, scoring='r2')
        r2_input = float(scores2.mean())
    except Exception:
        r2_input = -1.0
    results['exp5_input_recon_r2'] = r2_input
    print(f"  Input reconstruction R2: {r2_input:.4f}")

    results['T1092_narma_gt_neg'] = "PASS" if r2 > -0.5 else "FAIL"
    results['T1093_narma_gt_0'] = "PASS" if r2 > 0.0 else "FAIL"
    results['T1094_input_recon_gt_0'] = "PASS" if r2_input > 0.0 else "FAIL"
    results['T1095_input_recon_gt_010'] = "PASS" if r2_input > 0.10 else "FAIL"

    for t in ['T1092', 'T1093', 'T1094', 'T1095']:
        k = [k for k in results if k.startswith(t)][0]
        print(f"  {k}: {results[k]}")


def main():
    print("=" * 60)
    print("z2245: Bridge Temporal Experiments (VG Modulation, 50Hz)")
    print("=" * 60)

    print("\nConnecting to FPGA...")
    fpga = FPGAEthBridge()
    if not fpga.connect():
        print("  connect() returned False, continuing anyway...")

    fpga.set_kill(False)
    time.sleep(0.1)

    telem = fpga.read_telemetry()
    if telem is None:
        print("ERROR: No telemetry from FPGA")
        sys.exit(1)
    print(f"  FPGA connected: {len(telem['spike_counts'])} neurons")
    print(f"  Spike range: [{telem['spike_counts'].min():.0f}, {telem['spike_counts'].max():.0f}]")
    print(f"  Vmem range: [{telem['vmem'].min():.3f}, {telem['vmem'].max():.3f}]")

    vg_base = configure_fpga(fpga)
    print(f"  Vg range: [{vg_base.min():.3f}, {vg_base.max():.3f}]")

    results = {
        'experiment': 'z2245_bridge_temporal',
        'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
        'sample_hz': SAMPLE_HZ,
        'alpha': ALPHA,
        'base_vg': BASE_VG,
        'vg_spread': VG_SPREAD,
        'n_neurons': 128,
    }

    exp1_classification(fpga, vg_base, results)
    exp2_xor_delay(fpga, vg_base, results)
    exp3_temporal_discrimination(fpga, vg_base, results)
    exp4_memory_capacity(fpga, vg_base, results)
    exp5_narma(fpga, vg_base, results)

    tests = sorted([k for k in results if k.startswith('T')])
    n_pass = sum(1 for t in tests if results[t] == "PASS")
    n_total = len(tests)
    results['summary'] = f"{n_pass}/{n_total} PASS"

    print(f"\n{'=' * 60}")
    print(f"RESULTS: {n_pass}/{n_total} PASS")
    for t in tests:
        print(f"  {t}: {results[t]}")
    print(f"{'=' * 60}")

    RESULTS_FILE.parent.mkdir(exist_ok=True)
    with open(RESULTS_FILE, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved: {RESULTS_FILE}")


if __name__ == '__main__':
    main()
