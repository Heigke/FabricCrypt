#!/usr/bin/env python3
"""
z2279_lateral_coupling_validation.py — Validate enhanced lateral coupling
=========================================================================
Tests the FPGA with 32× stronger lateral coupling (4× weights + 8-pass hold):
  - Synaptic weights: N±1=0.50, N±2=0.25 (was 0.125/0.0625)
  - Spike hold: 8 pipeline passes (was 1)

Benchmarks:
  1. Waveform classification (4-class, 8-class)
  2. Memory capacity (delays 1..10)
  3. Temporal XOR (tau=1,2,3)
  4. Spike statistics (firing rate, correlation, ISI)

Compare against z2277 FPGA_ONLY baselines (MC=0, XOR≈chance, Wave4=28.7%)
"""

import os, sys, json, time
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from fpga_host_eth import FPGAEthBridge

# ── Configuration ──
NUM_NEURONS = 128
SAMPLE_HZ = 50
N_WAVE_TRIALS = 40
N_WAVE_STEPS = 60
N_CONTINUOUS_STEPS = 2000

RESULTS_PATH = os.path.join(os.path.dirname(__file__), '..', 'results', 'z2279_lateral_coupling.json')

# z2277 FPGA_ONLY baselines for comparison
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


# ── Run reservoir on FPGA ──
def fpga_run_sequence(fpga, input_seq, sample_hz=SAMPLE_HZ):
    """Inject input sequence via MAC, collect vmem states."""
    n_steps = len(input_seq)
    states = np.zeros((n_steps, NUM_NEURONS))
    dt = 1.0 / sample_hz

    fpga.set_mac_signal(0.0)
    time.sleep(0.1)

    for t in range(n_steps):
        mac_val = float(np.clip(input_seq[t], 0, 1))
        fpga.set_mac_signal(mac_val)
        time.sleep(dt)
        telem = fpga.read_telemetry()
        if telem is not None:
            states[t] = telem['vmem']
        else:
            # Retry once
            time.sleep(0.02)
            telem = fpga.read_telemetry()
            if telem is not None:
                states[t] = telem['vmem']

    fpga.set_mac_signal(0.0)
    return states


# ── Benchmarks ──
def benchmark_waveform(fpga, n_classes=4):
    print(f"\n  [{n_classes}-class waveform] {N_WAVE_TRIALS} trials × {N_WAVE_STEPS} steps...")
    X, y = [], []
    for trial in range(N_WAVE_TRIALS):
        for cls in range(n_classes):
            wf = generate_waveform(cls, N_WAVE_STEPS)
            wf_norm = (wf - wf.min()) / (wf.max() - wf.min() + 1e-10)
            wf_scaled = wf_norm * 0.8 + 0.1  # scale to 0.1..0.9 for MAC
            states = fpga_run_sequence(fpga, wf_scaled)
            feat = np.concatenate([states.mean(0), states.std(0), states[-1]])
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
    # Scale to MAC range 0..1
    u_mac = (u * 0.4 + 0.5).astype(np.float64)
    states = fpga_run_sequence(fpga, u_mac)
    warmup = 300
    results = {}

    # Memory capacity
    mc_total = 0.0
    mc_per_delay = {}
    for d in range(1, 11):
        X = states[warmup:]
        target = u[warmup-d:N_CONTINUOUS_STEPS-d]
        n = min(len(X), len(target))
        X_c, t_c = X[:n], target[:n]
        n_tr = int(0.7 * n)
        r2 = ridge_mc(X_c[:n_tr], t_c[:n_tr], X_c[n_tr:], t_c[n_tr:])
        mc_total += r2
        mc_per_delay[str(d)] = r2
        print(f"    MC delay={d:2d}: R²={r2:.4f}")
    results['mc'] = mc_total
    results['mc_per_delay'] = mc_per_delay
    print(f"    Total MC = {mc_total:.3f}")

    # XOR
    u_bin = (u > 0).astype(float)
    xor_results = {}
    for tau in [1, 2, 3]:
        target = np.zeros(N_CONTINUOUS_STEPS)
        for t in range(tau, N_CONTINUOUS_STEPS):
            target[t] = float(int(u_bin[t]) ^ int(u_bin[t-tau]))
        X = states[warmup:]
        y_xor = target[warmup:]
        n_tr = int(0.7 * len(X))
        I = np.eye(X.shape[1])
        try:
            w = np.linalg.solve(X[:n_tr].T @ X[:n_tr] + 1.0*I, X[:n_tr].T @ y_xor[:n_tr])
            pred = X[n_tr:] @ w
            acc = np.mean((pred > 0.5).astype(float) == y_xor[n_tr:])
        except Exception:
            acc = 0.5
        xor_results[str(tau)] = acc
        results[f'xor{tau}'] = acc
        print(f"    XOR tau={tau}: {acc:.1%}")

    # Spike statistics from last telemetry
    telem = fpga.read_telemetry()
    if telem is not None:
        sc = telem['spike_counts']
        results['spike_stats'] = {
            'mean_rate': float(np.mean(sc)),
            'std_rate': float(np.std(sc)),
            'active_neurons': int(np.sum(sc > 0)),
            'max_spikes': int(np.max(sc)),
        }
        print(f"    Spikes: mean={np.mean(sc):.1f}, active={np.sum(sc>0)}/128")

    return results


# ── Main ──
def main():
    print("="*70)
    print("  z2279: LATERAL COUPLING VALIDATION")
    print("  Enhanced: 4× weights + 8-pass spike hold (~32× coupling)")
    print("="*70)

    fpga = FPGAEthBridge()
    print("\n[1] Connecting to FPGA...")
    fpga.connect()
    fpga.set_kill(0)
    time.sleep(0.2)

    # Set gate voltages for all neurons
    print("\n[2] Configuring 128 neurons (Vg=0.58)...")
    for n in range(NUM_NEURONS):
        fpga.set_vg(n, 0.58)
    time.sleep(1.0)

    # Quick telemetry test
    print("\n[3] Telemetry test...")
    telem = fpga.read_telemetry()
    if telem is None:
        # Retry with longer wait
        time.sleep(0.5)
        telem = fpga.read_telemetry()
    if telem is None:
        print("  FATAL: No telemetry response")
        fpga.close()
        sys.exit(1)
    print(f"  OK: vmem range=[{telem['vmem'].min():.3f}, {telem['vmem'].max():.3f}]")
    print(f"  Spikes: {telem['spike_counts'][:8]}")

    results = {
        'experiment': 'z2279_lateral_coupling',
        'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
        'config': {
            'syn_weights': 'N±1=0.50, N±2=0.25 (4× increase)',
            'spike_hold': '8 pipeline passes (was 1)',
            'coupling_boost': '~32×',
            'leak_cond': '0x0004 (τ≈210ms)',
            'bias_gain': '0x0800 (0.03125)',
            'base_vg': 0.58,
            'sample_hz': SAMPLE_HZ,
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
    print("  COMPARISON: z2279 (enhanced coupling) vs z2277 FPGA_ONLY")
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
    n_pass = 0
    for name, val, ref in comparisons:
        improved = val > ref + 0.01  # at least 1pp improvement
        status = "IMPROVED" if improved else ("SAME" if abs(val-ref) < 0.01 else "WORSE")
        diff = val - ref
        sign = "+" if diff >= 0 else ""
        is_pct = name.startswith('Wave') or name.startswith('XOR')
        if is_pct:
            print(f"  {name:10s}: {val:.1%} (was {ref:.1%}, {sign}{diff*100:.1f}pp) — {status}")
        else:
            print(f"  {name:10s}: {val:.4f} (was {ref:.4f}, {sign}{diff:.4f}) — {status}")
        ok = val > ref
        if ok: n_pass += 1
        tests.append({'name': name, 'value': float(val), 'ref': float(ref),
                      'diff': float(diff), 'status': status})

    results['tests'] = tests
    results['n_improved'] = n_pass
    results['n_total'] = len(tests)
    print(f"\n  IMPROVED: {n_pass}/{len(tests)} metrics")

    # Key test: did coupling break the MC=0 and XOR=chance barrier?
    mc_val = cont.get('mc', 0)
    xor1_val = cont.get('xor1', 0.5)
    results['key_tests'] = {
        'T1_mc_above_zero': {
            'pass': mc_val > 0.1,
            'desc': f'MC={mc_val:.3f} > 0.1 (was 0.0)'
        },
        'T2_xor_above_chance': {
            'pass': xor1_val > 0.55,
            'desc': f'XOR1={xor1_val:.1%} > 55% (was 50.6%)'
        },
        'T3_wave4_maintained': {
            'pass': w4_acc > 0.25,
            'desc': f'Wave4={w4_acc:.1%} > 25% (was 28.7%)'
        },
        'T4_wave4_improved': {
            'pass': w4_acc > Z2277_FPGA_ONLY['wave4'] + 0.02,
            'desc': f'Wave4={w4_acc:.1%} > {Z2277_FPGA_ONLY["wave4"]:.1%}+2pp'
        },
    }
    key_pass = sum(1 for t in results['key_tests'].values() if t['pass'])
    print(f"\n  KEY TESTS: {key_pass}/4 PASS")
    for tid, t in results['key_tests'].items():
        print(f"    {tid}: {'PASS' if t['pass'] else 'FAIL'} — {t['desc']}")

    # Save results
    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
    with open(RESULTS_PATH, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Results saved: {RESULTS_PATH}")

    fpga.set_kill(1)
    fpga.close()

if __name__ == '__main__':
    main()
