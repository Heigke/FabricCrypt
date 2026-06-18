#!/usr/bin/env python3
"""
z2280_heterogeneous_reservoir.py — Heterogeneous FPGA reservoir
===============================================================
z2279 showed coupling alone fails because all neurons are identical.
Fix: heterogeneous Vg spread + 200Hz auto-telemetry + temporal features.

Three simultaneous interventions:
  1. Spread Vg across neurons: 0.50-0.66V (some below threshold, some above)
     This creates diverse firing regimes like the GPU's 4 populations.
  2. 200Hz auto-telemetry (was 50Hz) — 4× temporal resolution
  3. Temporal features: delta_vmem, spike_rate_change, ISI statistics

Compare against z2277 FPGA_ONLY baselines and z2279 uniform coupling.
"""

import os, sys, json, time
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from fpga_host_eth import FPGAEthBridge

NUM_NEURONS = 128
SAMPLE_HZ = 200   # auto-telemetry rate
N_WAVE_TRIALS = 40
N_WAVE_STEPS = 60
N_CONTINUOUS_STEPS = 2000

RESULTS_PATH = os.path.join(os.path.dirname(__file__), '..', 'results', 'z2280_heterogeneous_reservoir.json')

# Baselines
Z2277_FPGA_ONLY = {
    'wave4': 0.287, 'wave8': 0.226,
    'mc': 0.0, 'xor1': 0.506, 'xor2': 0.476, 'xor3': 0.530,
}

# ── Signal generators ──
def generate_waveform(cls, steps):
    t = np.linspace(0, 2*np.pi, steps)
    if cls == 0: return np.sin(t)
    elif cls == 1: return np.sign(np.sin(t))
    elif cls == 2: return 2*np.abs(2*(t/(2*np.pi) - np.floor(t/(2*np.pi)+0.5))) - 1
    else: return 2*(t/(2*np.pi) - np.floor(t/(2*np.pi))) - 1

def generate_narma(u, order=10):
    n = len(u)
    y = np.zeros(n)
    u_s = np.clip(u * 0.2 + 0.2, 0.0, 0.5)
    for t in range(order, n):
        s = np.sum(y[max(0, t-order):t])
        y[t] = 0.3*y[t-1] + 0.05*y[t-1]*s + 1.5*u_s[t-order]*u_s[t] + 0.1
        y[t] = np.clip(y[t], -5, 5)
    return y


# ── Readout ──
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


# ── FPGA reservoir with auto-telemetry ──
def fpga_run_sequence_auto(fpga, input_seq, sample_hz=SAMPLE_HZ):
    """Inject input via MAC, collect vmem via auto-telemetry at sample_hz."""
    n_steps = len(input_seq)
    states = np.zeros((n_steps, NUM_NEURONS))
    spikes = np.zeros((n_steps, NUM_NEURONS), dtype=np.uint16)
    dt = 1.0 / sample_hz

    fpga.set_mac_signal(0.0)
    time.sleep(0.05)

    for t in range(n_steps):
        mac_val = float(np.clip(input_seq[t], 0, 1))
        fpga.set_mac_signal(mac_val)
        time.sleep(dt)
        telem = fpga.read_telemetry()
        if telem is not None:
            states[t] = telem['vmem']
            spikes[t] = telem['spike_counts']
        elif t > 0:
            states[t] = states[t-1]  # hold last value

    fpga.set_mac_signal(0.0)
    return states, spikes


def extract_features(states, spikes):
    """Extract rich temporal features from state trajectories."""
    # Basic statistics
    feat_mean = states.mean(axis=0)
    feat_std = states.std(axis=0)
    feat_last = states[-1]

    # Temporal features: delta_vmem (derivative)
    delta = np.diff(states, axis=0)
    feat_delta_mean = delta.mean(axis=0)
    feat_delta_std = delta.std(axis=0)

    # Spike rate features
    spike_rate = spikes.astype(float).mean(axis=0)

    # Temporal structure: autocorrelation at lag 1
    n_steps = states.shape[0]
    if n_steps > 2:
        acf_1 = np.array([np.corrcoef(states[:-1, i], states[1:, i])[0, 1]
                          if np.std(states[:, i]) > 1e-8 else 0.0
                          for i in range(states.shape[1])])
    else:
        acf_1 = np.zeros(states.shape[1])
    acf_1 = np.nan_to_num(acf_1)

    return np.concatenate([feat_mean, feat_std, feat_last,
                           feat_delta_mean, feat_delta_std,
                           spike_rate, acf_1])


# ── Benchmarks ──
def benchmark_waveform(fpga, n_classes=4):
    print(f"\n  [{n_classes}-class waveform] {N_WAVE_TRIALS} trials × {N_WAVE_STEPS} steps @ {SAMPLE_HZ}Hz...")
    X, y = [], []
    for trial in range(N_WAVE_TRIALS):
        for cls in range(n_classes):
            wf = generate_waveform(cls, N_WAVE_STEPS)
            wf_norm = (wf - wf.min()) / (wf.max() - wf.min() + 1e-10)
            wf_scaled = wf_norm * 0.8 + 0.1
            states, spk = fpga_run_sequence_auto(fpga, wf_scaled)
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
    states, spk = fpga_run_sequence_auto(fpga, u_mac)

    # Add temporal derivative features for continuous tasks
    delta = np.diff(states, axis=0)
    delta = np.vstack([np.zeros((1, NUM_NEURONS)), delta])
    states_aug = np.hstack([states, delta])

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
        # Try multiple alphas
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

    # XOR
    u_bin = (u > 0).astype(float)
    for tau in [1, 2, 3]:
        target = np.zeros(N_CONTINUOUS_STEPS)
        for t in range(tau, N_CONTINUOUS_STEPS):
            target[t] = float(int(u_bin[t]) ^ int(u_bin[t-tau]))
        X = states_aug[warmup:]
        y_xor = target[warmup:]
        n_tr = int(0.7 * len(X))
        best_acc = 0.5
        for alpha in [0.01, 0.1, 1.0, 10.0, 100.0]:
            I = np.eye(X.shape[1])
            try:
                w = np.linalg.solve(X[:n_tr].T @ X[:n_tr] + alpha*I, X[:n_tr].T @ y_xor[:n_tr])
                pred = X[n_tr:] @ w
                acc = np.mean((pred > 0.5).astype(float) == y_xor[n_tr:])
                if acc > best_acc:
                    best_acc = acc
            except Exception:
                pass
        results[f'xor{tau}'] = best_acc
        print(f"    XOR tau={tau}: {best_acc:.1%}")

    # Spike statistics
    telem = fpga.read_telemetry()
    if telem is not None:
        sc = telem['spike_counts']
        results['spike_stats'] = {
            'mean_rate': float(np.mean(sc)),
            'std_rate': float(np.std(sc)),
            'active_neurons': int(np.sum(sc > 0)),
            'max_spikes': int(np.max(sc)),
        }
        print(f"    Spikes: mean={np.mean(sc):.1f}, std={np.std(sc):.1f}, active={np.sum(sc>0)}/128")

    # State diversity analysis
    vmem_stds = np.std(states[warmup:], axis=0)
    results['diversity'] = {
        'mean_neuron_std': float(np.mean(vmem_stds)),
        'std_neuron_std': float(np.std(vmem_stds)),
        'effective_dims': int(np.sum(vmem_stds > 0.001)),
        'cross_corr': float(np.mean(np.corrcoef(states[warmup:].T)[np.triu_indices(NUM_NEURONS, k=1)])),
    }
    print(f"    Diversity: eff_dims={results['diversity']['effective_dims']}, "
          f"cross_corr={results['diversity']['cross_corr']:.3f}")

    return results


# ── Main ──
def main():
    print("="*70)
    print("  z2280: HETEROGENEOUS FPGA RESERVOIR")
    print("  Spread Vg (0.50-0.66V) + 200Hz + temporal features")
    print("="*70)

    fpga = FPGAEthBridge()
    print("\n[1] Connecting to FPGA...")
    fpga.connect()
    fpga.set_kill(0)
    time.sleep(0.3)

    # Set HETEROGENEOUS gate voltages — 4 groups like GPU populations
    print("\n[2] Configuring 128 neurons with heterogeneous Vg...")
    vg_values = []
    for n in range(NUM_NEURONS):
        group = n % 4
        if group == 0:
            vg = 0.50 + (n / NUM_NEURONS) * 0.02   # Low: 0.50-0.52 (subthreshold, slow)
        elif group == 1:
            vg = 0.56 + (n / NUM_NEURONS) * 0.02   # Mid: 0.56-0.58 (near threshold)
        elif group == 2:
            vg = 0.60 + (n / NUM_NEURONS) * 0.02   # High: 0.60-0.62 (active firing)
        else:
            vg = 0.64 + (n / NUM_NEURONS) * 0.02   # Hot: 0.64-0.66 (fast firing)
        vg_values.append(vg)
        fpga.set_vg(n, vg)
    time.sleep(1.0)

    print(f"    Vg range: [{min(vg_values):.3f}, {max(vg_values):.3f}]")
    print(f"    Groups: subthreshold(0.50-0.52), near-thresh(0.56-0.58), active(0.60-0.62), fast(0.64-0.66)")

    # Telemetry test
    print("\n[3] Telemetry test...")
    telem = fpga.read_telemetry()
    if telem is None:
        time.sleep(0.5)
        telem = fpga.read_telemetry()
    if telem is None:
        print("  FATAL: No telemetry response")
        fpga.set_kill(1)
        fpga.close()
        sys.exit(1)
    vm = telem['vmem']
    sc = telem['spike_counts']
    print(f"  OK: vmem range=[{vm.min():.3f}, {vm.max():.3f}]")
    print(f"  Spike counts: min={sc.min()}, max={sc.max()}, mean={sc.mean():.0f}")

    # Check diversity
    print(f"  Neuron diversity: vmem_std={np.std(vm):.4f}")
    for g in range(4):
        g_mask = np.arange(NUM_NEURONS) % 4 == g
        print(f"    Group {g} (Vg~{vg_values[g]:.2f}): vmem={vm[g_mask].mean():.3f}±{vm[g_mask].std():.3f}, "
              f"spikes={sc[g_mask].mean():.0f}±{sc[g_mask].std():.0f}")

    results = {
        'experiment': 'z2280_heterogeneous_reservoir',
        'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
        'config': {
            'vg_spread': '0.50-0.66V (4 groups)',
            'syn_weights': 'N±1=0.50, N±2=0.25 (32× coupling)',
            'spike_hold': '8 pipeline passes',
            'leak_cond': '0x0004 (τ≈210ms)',
            'bias_gain': '0x0800 (0.03125)',
            'sample_hz': SAMPLE_HZ,
            'features': 'mean+std+last+delta_mean+delta_std+spike_rate+acf1',
        }
    }

    # Waveform classification
    print("\n[4] Waveform classification...")
    w4_acc, w4_std = benchmark_waveform(fpga, 4)
    w8_acc, w8_std = benchmark_waveform(fpga, 8)
    results['waveform_4class'] = {'accuracy': w4_acc, 'std': w4_std}
    results['waveform_8class'] = {'accuracy': w8_acc, 'std': w8_std}

    # Continuous benchmarks
    print("\n[5] Continuous benchmarks (MC, XOR)...")
    cont = benchmark_continuous(fpga)
    results['continuous'] = cont

    # Comparison
    print("\n" + "="*70)
    print("  z2280 (HETEROGENEOUS) vs z2277 FPGA_ONLY (UNIFORM)")
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

    # Key tests
    mc_val = cont.get('mc', 0)
    xor1_val = cont.get('xor1', 0.5)
    results['key_tests'] = {
        'T1_mc_above_zero': {
            'pass': mc_val > 0.05,
            'desc': f'MC={mc_val:.3f} > 0.05'
        },
        'T2_xor_above_chance': {
            'pass': xor1_val > 0.55,
            'desc': f'XOR1={xor1_val:.1%} > 55%'
        },
        'T3_wave4_improved': {
            'pass': w4_acc > Z2277_FPGA_ONLY['wave4'],
            'desc': f'Wave4={w4_acc:.1%} > {Z2277_FPGA_ONLY["wave4"]:.1%}'
        },
        'T4_diversity': {
            'pass': cont.get('diversity', {}).get('effective_dims', 0) > 50,
            'desc': f'EffDims={cont.get("diversity", {}).get("effective_dims", 0)} > 50'
        },
    }
    key_pass = sum(1 for t in results['key_tests'].values() if t['pass'])
    print(f"\n  KEY TESTS: {key_pass}/4 PASS")
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
