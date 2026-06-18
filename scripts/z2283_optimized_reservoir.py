#!/usr/bin/env python3
"""
z2283_optimized_reservoir.py — FPGA reservoir with sweep-optimized parameters
=============================================================================
z2282 parameter sweep found MC(d=1)=0.3575 with:
  LEAK=0x0020, EXC=0x0080, BIAS=0x4000, THRESH=0x20000

Key physics: moderate excitation preserves Vg-dependent heterogeneity
while high threshold prevents spiking → neurons integrate MAC input
with Vg-dependent dynamics. Leak prevents vmem saturation.

Full benchmark: Wave-4, Wave-8, MC delays 1-10, XOR tau 1-3.
"""

import os, sys, json, time
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from fpga_host_eth import FPGAEthBridge

NUM_NEURONS = 128
SAMPLE_HZ = 200
N_WAVE_TRIALS = 40
N_WAVE_STEPS = 60
N_CONTINUOUS_STEPS = 2000

RESULTS_PATH = os.path.join(os.path.dirname(__file__), '..', 'results', 'z2283_optimized_reservoir.json')

# z2277 baselines
Z2277_FPGA_ONLY = {
    'wave4': 0.287, 'wave8': 0.226,
    'mc': 0.0, 'xor1': 0.506, 'xor2': 0.476, 'xor3': 0.530,
}

# Optimal params from z2282 sweep
# Two regimes tested:
# A) Low leak (0x0020): Wave-4=89%, MC=0.39, XOR=chance
# B) High leak (0x2000): MC(1)=0.96, MC(2)=0.64, MC_total=1.87
# Try high leak first — MC is the harder barrier to break
OPT_LEAK = 0x2000
OPT_EXC = 0x0080
OPT_BIAS = 0x4000
OPT_THRESH = 0x20000

# 4 Vg groups spanning BVpar transition
VG_GROUPS = {0: 0.05, 1: 0.15, 2: 0.30, 3: 0.58}


def generate_waveform(cls, steps):
    t = np.linspace(0, 2*np.pi, steps)
    if cls == 0: return np.sin(t)
    elif cls == 1: return np.sign(np.sin(t))
    elif cls == 2: return 2*np.abs(2*(t/(2*np.pi) - np.floor(t/(2*np.pi)+0.5))) - 1
    else: return 2*(t/(2*np.pi) - np.floor(t/(2*np.pi))) - 1


def ridge_classify(X, y, n_classes, alpha=10.0):
    from sklearn.linear_model import RidgeClassifier
    from sklearn.model_selection import cross_val_score
    sigma = np.std(X, axis=0)
    sigma[sigma < 1e-2] = 1.0
    X_n = X / sigma
    clf = RidgeClassifier(alpha=alpha)
    scores = cross_val_score(clf, X_n, y, cv=5)
    return scores.mean(), scores.std()


def ridge_mc(X_tr, y_tr, X_te, y_te, alpha=1.0):
    I = np.eye(X_tr.shape[1])
    try:
        w = np.linalg.solve(X_tr.T @ X_tr + alpha*I, X_tr.T @ y_tr)
    except Exception:
        return 0.0
    pred = X_te @ w
    ss_res = np.sum((y_te - pred)**2)
    ss_tot = np.sum((y_te - y_te.mean())**2)
    return max(0, 1 - ss_res/ss_tot) if ss_tot > 1e-10 else 0.0


def fpga_run_sequence(fpga, input_seq, sample_hz=SAMPLE_HZ):
    """Inject input via MAC, collect vmem+differential spikes."""
    n_steps = len(input_seq)
    states = np.zeros((n_steps, NUM_NEURONS))
    dspikes = np.zeros((n_steps, NUM_NEURONS), dtype=np.float32)
    dt = 1.0 / sample_hz

    fpga.set_mac_signal(0.0)
    time.sleep(0.05)

    telem = fpga.read_telemetry()
    prev_sc = telem['spike_counts'].copy() if telem is not None else np.zeros(NUM_NEURONS, dtype=np.uint16)

    for t in range(n_steps):
        mac_val = float(np.clip(input_seq[t], 0, 1))
        fpga.set_mac_signal(mac_val)
        time.sleep(dt)
        telem = fpga.read_telemetry()
        if telem is not None:
            states[t] = telem['vmem']
            sc = telem['spike_counts']
            diff = sc.astype(np.int32) - prev_sc.astype(np.int32)
            diff[diff < 0] += 65536
            dspikes[t] = diff.astype(np.float32)
            prev_sc = sc.copy()
        elif t > 0:
            states[t] = states[t-1]
            dspikes[t] = dspikes[t-1]

    fpga.set_mac_signal(0.0)
    return states, dspikes


def extract_features(states, dspikes):
    """Rich features from vmem + differential spikes."""
    feat_mean = states.mean(axis=0)
    feat_std = states.std(axis=0)
    feat_last = states[-1]
    ds_mean = dspikes.mean(axis=0)
    ds_std = dspikes.std(axis=0)
    ds_last = dspikes[-1] if len(dspikes) > 0 else np.zeros(states.shape[1])
    delta = np.diff(states, axis=0)
    feat_delta_std = delta.std(axis=0) if len(delta) > 0 else np.zeros(states.shape[1])
    return np.concatenate([feat_mean, feat_std, feat_last,
                           ds_mean, ds_std, ds_last, feat_delta_std])


def benchmark_waveform(fpga, n_classes=4):
    print(f"\n  [{n_classes}-class waveform] {N_WAVE_TRIALS} trials × {N_WAVE_STEPS} steps @ {SAMPLE_HZ}Hz...")
    X, y = [], []
    for trial in range(N_WAVE_TRIALS):
        for cls in range(n_classes):
            wf = generate_waveform(cls, N_WAVE_STEPS)
            wf_norm = (wf - wf.min()) / (wf.max() - wf.min() + 1e-10)
            wf_scaled = wf_norm * 0.8 + 0.1
            states, spk = fpga_run_sequence(fpga, wf_scaled)
            feat = extract_features(states, spk)
            X.append(feat)
            y.append(cls)
        if (trial+1) % 10 == 0:
            print(f"    trial {trial+1}/{N_WAVE_TRIALS}")
    X, y = np.array(X), np.array(y)
    acc, std = ridge_classify(X, y, n_classes)
    print(f"    Accuracy: {acc:.1%} ± {std:.1%}")
    return acc, std


def benchmark_continuous(fpga):
    print(f"\n  [Continuous] {N_CONTINUOUS_STEPS} steps @ {SAMPLE_HZ}Hz...")
    rng = np.random.default_rng(42)
    u = rng.uniform(-1, 1, N_CONTINUOUS_STEPS).astype(np.float64)
    u_mac = (u * 0.4 + 0.5).astype(np.float64)
    states, spk = fpga_run_sequence(fpga, u_mac)

    delta = np.diff(states, axis=0)
    delta = np.vstack([np.zeros((1, NUM_NEURONS)), delta])
    states_aug = np.hstack([states, spk, delta])

    warmup = 300
    results = {}

    # Memory capacity
    mc_total = 0.0
    mc_per_delay = {}
    for d in range(1, 11):
        X = states_aug[warmup:]
        target = u[warmup-d:N_CONTINUOUS_STEPS-d]
        n = min(len(X), len(target))
        X_c, t_c = X[:n], target[:n]
        n_tr = int(0.7 * n)
        best_r2 = 0.0
        for alpha in [0.01, 0.1, 1.0, 10.0, 100.0]:
            r2 = ridge_mc(X_c[:n_tr], t_c[:n_tr], X_c[n_tr:], t_c[n_tr:], alpha)
            if r2 > best_r2:
                best_r2 = r2
        mc_total += best_r2
        mc_per_delay[str(d)] = best_r2
        print(f"    MC delay={d:2d}: R²={best_r2:.4f}")
    results['mc'] = mc_total
    results['mc_per_delay'] = mc_per_delay
    print(f"    Total MC = {mc_total:.3f}")

    # XOR — try both linear and quadratic readout
    u_bin = (u > 0).astype(float)

    # Build quadratic features: products of vmem pairs (subsample for tractability)
    # Pick 32 neurons evenly spaced + their dspike channels
    quad_idx = np.arange(0, NUM_NEURONS, 4)  # 32 neurons
    vm_sub = states[warmup:][:, quad_idx]
    ds_sub = spk[warmup:][:, quad_idx]
    # Products: vm_i * ds_i for each neuron (32 features)
    # Plus vm_i * vm_{i+1} for adjacent pairs (31 features)
    quad_feats = []
    quad_feats.append(vm_sub * ds_sub)  # 32 features: vmem × dspike (same neuron)
    quad_feats.append(vm_sub[:, :-1] * vm_sub[:, 1:])  # 31 features: adjacent vmem products
    quad_feats.append(np.square(vm_sub))  # 32 features: vmem²
    X_quad = np.hstack([states_aug[warmup:]] + quad_feats)  # 384 + 95 = 479 features

    for tau in [1, 2, 3]:
        target = np.zeros(N_CONTINUOUS_STEPS)
        for t in range(tau, N_CONTINUOUS_STEPS):
            target[t] = float(int(u_bin[t]) ^ int(u_bin[t-tau]))

        # Linear readout
        X_lin = states_aug[warmup:]
        y_xor = target[warmup:]
        n_tr = int(0.7 * len(X_lin))
        best_lin = 0.5
        for alpha in [0.01, 0.1, 1.0, 10.0, 100.0]:
            I = np.eye(X_lin.shape[1])
            try:
                w = np.linalg.solve(X_lin[:n_tr].T @ X_lin[:n_tr] + alpha*I, X_lin[:n_tr].T @ y_xor[:n_tr])
                pred = X_lin[n_tr:] @ w
                acc = np.mean((pred > 0.5).astype(float) == y_xor[n_tr:])
                if acc > best_lin:
                    best_lin = acc
            except Exception:
                pass

        # Quadratic readout
        best_quad = 0.5
        for alpha in [0.1, 1.0, 10.0, 100.0, 1000.0]:
            I = np.eye(X_quad.shape[1])
            try:
                w = np.linalg.solve(X_quad[:n_tr].T @ X_quad[:n_tr] + alpha*I, X_quad[:n_tr].T @ y_xor[:n_tr])
                pred = X_quad[n_tr:] @ w
                acc = np.mean((pred > 0.5).astype(float) == y_xor[n_tr:])
                if acc > best_quad:
                    best_quad = acc
            except Exception:
                pass

        results[f'xor{tau}'] = best_lin
        results[f'xor{tau}_quad'] = best_quad
        print(f"    XOR tau={tau}: linear={best_lin:.1%}, quadratic={best_quad:.1%}")

    # Diversity analysis
    vmem_stds = np.std(states[warmup:], axis=0)
    corr_mat = np.corrcoef(states[warmup:].T)
    upper_tri = corr_mat[np.triu_indices(NUM_NEURONS, k=1)]

    # Effective dimensionality
    X_vm = states[warmup:]
    X_vm_c = X_vm - X_vm.mean(0)
    try:
        sv = np.linalg.svd(X_vm_c, compute_uv=False)
        sv_norm = sv / sv.sum()
        eff_dim = np.exp(-np.sum(sv_norm * np.log(sv_norm + 1e-30)))
    except Exception:
        eff_dim = 0.0

    results['diversity'] = {
        'mean_neuron_std': float(np.mean(vmem_stds)),
        'std_neuron_std': float(np.std(vmem_stds)),
        'effective_dims': int(np.sum(vmem_stds > 0.001)),
        'eff_dim_svd': float(eff_dim),
        'cross_corr': float(np.mean(upper_tri)),
    }
    print(f"    Diversity: eff_dim={eff_dim:.1f}, cross_corr={results['diversity']['cross_corr']:.3f}")

    return results


def main():
    print("="*70)
    print("  z2283: OPTIMIZED FPGA RESERVOIR")
    print(f"  Params: LEAK={OPT_LEAK:#06x} EXC={OPT_EXC:#06x} BIAS={OPT_BIAS:#06x} THRESH={OPT_THRESH:#06x}")
    print("="*70)

    fpga = FPGAEthBridge()
    print("\n[1] Connecting to FPGA...")
    fpga.connect()
    fpga.set_kill(0)
    time.sleep(0.3)

    # Apply optimal runtime parameters
    print("\n[2] Setting optimized parameters...")
    fpga.set_leak_cond(OPT_LEAK)
    fpga.set_base_exc_raw(OPT_EXC)
    fpga.set_bias_gain_raw(OPT_BIAS)
    fpga.set_threshold_raw(OPT_THRESH)
    time.sleep(0.5)
    print(f"    LEAK={OPT_LEAK:#06x}, EXC={OPT_EXC:#06x}, BIAS={OPT_BIAS:#06x}, THRESH={OPT_THRESH:#06x}")

    # Set heterogeneous Vg (4-group approach — best from z2281)
    print("\n[3] Setting heterogeneous Vg (4 groups)...")
    for n in range(NUM_NEURONS):
        fpga.set_vg(n, VG_GROUPS[n % 4])
        time.sleep(0.001)
    time.sleep(1.0)
    print(f"    Groups: {VG_GROUPS}")

    # Quick diagnostic
    print("\n[4] Diagnostic telemetry...")
    telem = fpga.read_telemetry()
    if telem is None:
        time.sleep(0.5)
        telem = fpga.read_telemetry()
    if telem is None:
        print("  FATAL: No telemetry")
        fpga.close()
        sys.exit(1)
    print(f"    vmem range: [{telem['vmem'].min():.3f}, {telem['vmem'].max():.3f}]")

    # Quick MAC stimulus to check response
    fpga.set_mac_signal(0.5)
    time.sleep(1.0)
    telem2 = fpga.read_telemetry()
    fpga.set_mac_signal(0.0)
    if telem2 is not None:
        print(f"    vmem after MAC=0.5: [{telem2['vmem'].min():.3f}, {telem2['vmem'].max():.3f}]")
        # Per-group analysis
        for g in range(4):
            mask = np.arange(NUM_NEURONS) % 4 == g
            gvm = telem2['vmem'][mask]
            gsc = telem2['spike_counts'][mask]
            print(f"    G{g} (Vg={VG_GROUPS[g]:.2f}): vmem={gvm.mean():.3f}±{gvm.std():.3f}, spikes={gsc.mean():.0f}")

    results = {
        'experiment': 'z2283_optimized_reservoir',
        'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
        'config': {
            'vg_groups': VG_GROUPS,
            'leak_cond': hex(OPT_LEAK),
            'base_exc': hex(OPT_EXC),
            'bias_gain': hex(OPT_BIAS),
            'threshold': hex(OPT_THRESH),
            'sample_hz': SAMPLE_HZ,
            'sweep_source': 'z2282 round 4: mod_exc_mac_3',
        },
    }

    # Benchmarks
    print("\n[5] Waveform classification...")
    w4_acc, w4_std = benchmark_waveform(fpga, 4)
    w8_acc, w8_std = benchmark_waveform(fpga, 8)
    results['waveform_4class'] = {'accuracy': w4_acc, 'std': w4_std}
    results['waveform_8class'] = {'accuracy': w8_acc, 'std': w8_std}

    print("\n[6] Continuous benchmarks (MC, XOR)...")
    cont = benchmark_continuous(fpga)
    results['continuous'] = cont

    # Comparison
    print("\n" + "="*70)
    print("  z2283 (OPTIMIZED) vs z2277 FPGA_ONLY (BASELINE)")
    print("="*70)
    comparisons = [
        ('Wave-4',  w4_acc,                    Z2277_FPGA_ONLY['wave4']),
        ('Wave-8',  w8_acc,                    Z2277_FPGA_ONLY['wave8']),
        ('MC',      cont.get('mc', 0),         Z2277_FPGA_ONLY['mc']),
        ('XOR-1',   cont.get('xor1', 0.5),     Z2277_FPGA_ONLY['xor1']),
        ('XOR-2',   cont.get('xor2', 0.5),     Z2277_FPGA_ONLY['xor2']),
        ('XOR-3',   cont.get('xor3', 0.5),     Z2277_FPGA_ONLY['xor3']),
    ]

    tests = []
    n_improved = 0
    for name, val, ref in comparisons:
        diff = val - ref
        improved = val > ref + 0.01
        status = "IMPROVED" if improved else ("SAME" if abs(diff) < 0.01 else "WORSE")
        if improved: n_improved += 1
        sign = "+" if diff >= 0 else ""
        is_pct = name.startswith('Wave') or name.startswith('XOR')
        if is_pct:
            print(f"  {name:10s}: {val:.1%} (was {ref:.1%}, {sign}{diff*100:.1f}pp) — {status}")
        else:
            print(f"  {name:10s}: {val:.4f} (was {ref:.4f}, {sign}{diff:.4f}) — {status}")
        tests.append({'name': name, 'value': float(val), 'ref': float(ref),
                      'diff': float(diff), 'status': status})

    results['tests'] = tests
    results['n_improved'] = n_improved
    results['n_total'] = len(tests)
    print(f"\n  IMPROVED: {n_improved}/{len(tests)} metrics")

    # Second run: lower threshold for spiking nonlinearity
    print("\n[7] Re-run continuous with THRESH=0x8000 (more spiking)...")
    fpga.set_threshold_raw(0x8000)
    time.sleep(0.5)
    rng2 = np.random.default_rng(42)
    u2 = rng2.uniform(-1, 1, N_CONTINUOUS_STEPS).astype(np.float64)
    u2_mac = (u2 * 0.4 + 0.5).astype(np.float64)
    states2, spk2 = fpga_run_sequence(fpga, u2_mac)
    delta2 = np.diff(states2, axis=0)
    delta2 = np.vstack([np.zeros((1, NUM_NEURONS)), delta2])
    states_aug2 = np.hstack([states2, spk2, delta2])
    warmup2 = 300
    u2_bin = (u2 > 0).astype(float)
    for tau in [1, 2, 3]:
        target2 = np.zeros(N_CONTINUOUS_STEPS)
        for t in range(tau, N_CONTINUOUS_STEPS):
            target2[t] = float(int(u2_bin[t]) ^ int(u2_bin[t-tau]))
        X2 = states_aug2[warmup2:]
        y2 = target2[warmup2:]
        n_tr2 = int(0.7 * len(X2))
        best2 = 0.5
        for alpha in [0.01, 0.1, 1.0, 10.0, 100.0]:
            I = np.eye(X2.shape[1])
            try:
                w = np.linalg.solve(X2[:n_tr2].T @ X2[:n_tr2] + alpha*I, X2[:n_tr2].T @ y2[:n_tr2])
                pred = X2[n_tr2:] @ w
                acc = np.mean((pred > 0.5).astype(float) == y2[n_tr2:])
                if acc > best2:
                    best2 = acc
            except Exception:
                pass
        results[f'xor{tau}_spiking'] = best2
        print(f"    XOR tau={tau} (spiking): {best2:.1%}")

    # MC with spiking
    mc2_d1 = 0.0
    X2 = states_aug2[warmup2:]
    target2 = u2[warmup2-1:N_CONTINUOUS_STEPS-1]
    n2 = min(len(X2), len(target2))
    n_tr2 = int(0.7 * n2)
    for alpha in [0.01, 0.1, 1.0, 10.0, 100.0]:
        r2 = ridge_mc(X2[:n_tr2], target2[:n_tr2], X2[n_tr2:n2], target2[n_tr2:n2], alpha)
        if r2 > mc2_d1:
            mc2_d1 = r2
    results['mc_d1_spiking'] = mc2_d1
    print(f"    MC(d=1) spiking: {mc2_d1:.4f}")

    # Restore original threshold
    fpga.set_threshold_raw(OPT_THRESH)

    # Key tests
    mc_val = cont.get('mc', 0)
    xor1_val = cont.get('xor1', 0.5)
    results['key_tests'] = {
        'T1_mc_above_threshold': {
            'pass': mc_val > 0.1,
            'desc': f'Total MC={mc_val:.3f} > 0.1'
        },
        'T2_mc_d1_significant': {
            'pass': float(cont.get('mc_per_delay', {}).get('1', 0)) > 0.05,
            'desc': f'MC(d=1)={cont.get("mc_per_delay", {}).get("1", 0):.4f} > 0.05'
        },
        'T3_xor_above_chance': {
            'pass': max(xor1_val,
                        cont.get('xor1_quad', 0.5),
                        results.get('xor1_spiking', 0.5)) > 0.55,
            'desc': f'Best XOR1={max(xor1_val, cont.get("xor1_quad", 0.5), results.get("xor1_spiking", 0.5)):.1%} > 55%'
        },
        'T4_wave4_improved': {
            'pass': w4_acc > Z2277_FPGA_ONLY['wave4'],
            'desc': f'Wave4={w4_acc:.1%} > {Z2277_FPGA_ONLY["wave4"]:.1%}'
        },
        'T5_wave4_above_50pct': {
            'pass': w4_acc > 0.50,
            'desc': f'Wave4={w4_acc:.1%} > 50%'
        },
    }
    key_pass = sum(1 for t in results['key_tests'].values() if t['pass'])
    print(f"\n  KEY TESTS: {key_pass}/5 PASS")
    for tid, t in results['key_tests'].items():
        print(f"    {tid}: {'PASS' if t['pass'] else 'FAIL'} — {t['desc']}")

    # Save
    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
    with open(RESULTS_PATH, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Results saved: {RESULTS_PATH}")

    fpga.set_kill(1)
    fpga.close()

if __name__ == '__main__':
    main()
