#!/usr/bin/env python3
"""
z2327_fpga_criticality.py — FPGA Edge-of-Criticality Parameter Tuning
======================================================================
z2325 showed two extremes: rank=1.2 (no spiking) vs rank=110 (saturated).
This script explores the transition zone where moderate spiking enables
both high rank AND input-dependent dynamics.

Key insight from z2325: T=1.5/E=0.002 had 36/128 active at rate 91.8 —
this is the "edge of criticality" where synaptic dynamics emerge without
drowning the input signal.

Fine-grained sweep around transition:
  - Threshold: [0x16000, 0x17000, 0x18000, 0x19000, 0x1A000, 0x1C000, 0x1E000, 0x20000]
  - Base_exc:  [0x0020, 0x0040, 0x0060, 0x0080, 0x00A0, 0x00C0, 0x0100, 0x0200]
  - Bias_gain: [0x0000, 0x0400, 0x0800, 0x1000]
  - Leak:      [0x2000 (original), 0x0004 (slow)]

EXP 1: Fine-grained sweep (threshold × base_exc, ~64 configs)
  Quick probe: 200 steps, score by rank + MC + n_active
  Tests: T1104-T1107

EXP 2: Top-5 configs → full evaluation (2000 steps each)
  Rank, MC, spike stats, classification, XOR
  Tests: T1108-T1119

EXP 3: Best config comparison vs z2324 (non-spiking) baseline
  Tests: T1120-T1125

Total: 22 tests (T1104-T1125)

Run:
  HSA_OVERRIDE_GFX_VERSION=11.0.0 PYTHONUNBUFFERED=1 venv/bin/python scripts/z2327_fpga_criticality.py
"""

import os, sys, time, json, struct, socket
import numpy as np
from pathlib import Path
from collections import OrderedDict

os.environ['PYTHONUNBUFFERED'] = '1'

BASE = Path(__file__).resolve().parent.parent
RESULTS = BASE / 'results'
RESULTS.mkdir(exist_ok=True)
SAVE_FILE = RESULTS / 'z2327_fpga_criticality.json'
sys.path.insert(0, str(BASE / 'scripts'))
from fpga_host_eth import FPGAEthBridge

NUM_NEURONS = 128
SAMPLE_HZ = 50

# ======================================================================
# Thermal safety
# ======================================================================
def get_temp():
    try:
        with open('/sys/class/thermal/thermal_zone0/temp') as f:
            return int(f.read().strip()) // 1000
    except:
        return 50

def wait_cool(target=50, timeout=120):
    t0 = time.time()
    while get_temp() > target and time.time() - t0 < timeout:
        time.sleep(2)
    return get_temp()

def check_thermal(pause_at=75, resume_at=50):
    t = get_temp()
    if t >= pause_at:
        print(f"  [TEMP] {t}C >= {pause_at}C, cooling...", end='', flush=True)
        t2 = wait_cool(resume_at)
        print(f" {t2}C OK")
    return get_temp()

# ======================================================================
# FPGA setup helper
# ======================================================================
def setup_fpga(fpga, thresh, base_exc, bias_gain=0x0800, leak=0x2000):
    """Configure FPGA runtime parameters."""
    fpga.set_threshold_raw(thresh)
    fpga.set_base_exc_raw(base_exc)
    fpga.set_bias_gain_raw(bias_gain)
    fpga.set_leak_cond(leak)
    # Set Vg groups (from z2324 pattern)
    for g in range(4):
        vg = [0x9000, 0x9800, 0xA000, 0xA800][g]
        fpga.set_vg(g * 32, (g + 1) * 32 - 1, vg)
    time.sleep(0.2)

def fpga_quick_probe(fpga, n_steps=200, mac_pattern='random'):
    """Quick probe: collect states and compute basic metrics."""
    rng = np.random.RandomState(42)
    states = np.zeros((n_steps, NUM_NEURONS), dtype=np.float32)
    dspikes = np.zeros((n_steps, NUM_NEURONS), dtype=np.float32)
    inputs = rng.uniform(0, 1, n_steps)

    for t in range(n_steps):
        mac = inputs[t] if mac_pattern == 'random' else 0.5
        fpga.set_mac_signal(int(mac * 65536))
        telem = fpga.read_telemetry()
        if telem is not None:
            vmem = np.array(telem['vmem'], dtype=np.float32) / 65536.0
            scnt = np.array(telem['spike_cnt'], dtype=np.float32)
            states[t] = vmem
            dspikes[t] = scnt
        time.sleep(1.0 / SAMPLE_HZ)

    # Delta spikes
    delta = np.diff(dspikes, axis=0)
    delta[delta < 0] = 0  # Handle counter resets

    # Effective rank
    X = states[50:]  # Skip warmup
    X = X - X.mean(axis=0, keepdims=True)
    cov = X.T @ X / len(X)
    sv = np.linalg.svd(cov, compute_uv=False)
    sv = sv / (sv.sum() + 1e-20)
    sv = sv[sv > 1e-12]
    rank = float(np.exp(-np.sum(sv * np.log(sv + 1e-20))))

    # Spike stats
    total_spikes = delta.sum()
    spike_rates = delta.sum(axis=0)
    n_active = int((spike_rates > 0).sum())
    rate_mean = float(spike_rates.mean())
    rate_std = float(spike_rates.std())

    # Mean correlation
    if X.shape[0] > 10:
        corr = np.corrcoef(X.T)
        np.fill_diagonal(corr, 0)
        mean_corr = float(np.abs(corr).mean())
    else:
        mean_corr = 1.0

    # Memory capacity (quick, d=1..10)
    mc = compute_mc_quick(X, inputs[50:], max_delay=10)

    return {
        'rank': rank, 'n_active': n_active, 'rate_mean': rate_mean,
        'rate_std': rate_std, 'total_spikes': total_spikes,
        'mean_corr': mean_corr, 'mc': mc,
        'states': states, 'dspikes': delta, 'inputs': inputs,
    }

def compute_mc_quick(states, inputs, max_delay=10):
    """Quick memory capacity."""
    n = min(len(states), len(inputs))
    states = states[:n]
    inputs = inputs[:n]
    split = n // 2
    mc = 0.0
    for d in range(1, max_delay + 1):
        if d >= n:
            break
        y_train = inputs[:split - d]
        y_test = inputs[split:n - d]
        X_train = states[d:d + len(y_train)]
        X_test = states[split + d:split + d + len(y_test)]
        if len(X_test) < 10 or len(X_train) < 10:
            continue
        # Ridge
        alpha = 1e-4
        XtX = X_train.T @ X_train + alpha * np.eye(X_train.shape[1])
        Xty = X_train.T @ y_train
        try:
            w = np.linalg.solve(XtX, Xty)
            y_pred = X_test @ w
            cc = np.corrcoef(y_test.flatten(), y_pred.flatten())[0, 1]
            if not np.isnan(cc):
                mc += max(0, cc**2)
        except:
            pass
    return mc

def compute_mc_full(states, inputs, max_delay=20):
    """Full memory capacity with per-delay breakdown."""
    n = min(len(states), len(inputs))
    states = states[:n]
    inputs = inputs[:n]
    split = n // 2
    mc_total = 0.0
    mc_per = {}
    for d in range(1, max_delay + 1):
        if d >= n:
            mc_per[d] = 0.0
            continue
        y_train = inputs[:split - d]
        y_test = inputs[split:n - d]
        X_train = states[d:d + len(y_train)]
        X_test = states[split + d:split + d + len(y_test)]
        if len(X_test) < 10 or len(X_train) < 10:
            mc_per[d] = 0.0
            continue
        alpha = 1e-4
        XtX = X_train.T @ X_train + alpha * np.eye(X_train.shape[1])
        Xty = X_train.T @ y_train
        try:
            w = np.linalg.solve(XtX, Xty)
            y_pred = X_test @ w
            cc = np.corrcoef(y_test.flatten(), y_pred.flatten())[0, 1]
            r2 = max(0, cc**2) if not np.isnan(cc) else 0.0
        except:
            r2 = 0.0
        mc_per[d] = r2
        mc_total += r2
    return mc_total, mc_per

def compute_xor(states, inputs, tau):
    """XOR temporal task."""
    n = min(len(states), len(inputs))
    u_bin = (inputs[:n] > np.median(inputs[:n])).astype(float)
    targets = np.zeros(n)
    for t in range(tau, n):
        targets[t] = float(int(u_bin[t]) ^ int(u_bin[t - tau]))
    valid = slice(max(tau, 50), n)
    X = states[valid]
    y = targets[valid]
    split = len(X) // 2
    if split < 10:
        return 0.5
    alpha = 1e-4
    XtX = X[:split].T @ X[:split] + alpha * np.eye(X.shape[1])
    Xty = X[:split].T @ y[:split]
    try:
        w = np.linalg.solve(XtX, Xty)
        y_pred = X[split:] @ w
        acc = float(np.mean((y_pred > 0.5).astype(float) == y[split:]))
    except:
        acc = 0.5
    return acc

def classify_waveforms(fpga, n_samples=200, n_classes=4, n_steps=50):
    """Classify waveforms using FPGA reservoir."""
    from sklearn.linear_model import RidgeClassifier
    rng = np.random.RandomState(42)
    t = np.linspace(0, 2*np.pi, n_steps)
    signals, labels = [], []
    for _ in range(n_samples):
        c = rng.randint(n_classes)
        if c == 0: s = np.sin(t)
        elif c == 1: s = np.sign(np.sin(t))
        elif c == 2: s = 2*(t/(2*np.pi) - np.floor(t/(2*np.pi) + 0.5))
        else: s = np.sin(t) + 0.5*np.sin(3*t)
        s += rng.randn(n_steps) * 0.1
        signals.append(s)
        labels.append(c)
    labels = np.array(labels)

    features = []
    for i in range(n_samples):
        sample_states = np.zeros((n_steps, NUM_NEURONS), dtype=np.float32)
        for st in range(n_steps):
            mac = int(np.clip((signals[i][st] + 1.5) / 3.0, 0, 1) * 65536)
            fpga.set_mac_signal(mac)
            telem = fpga.read_telemetry()
            if telem is not None:
                sample_states[st] = np.array(telem['vmem'], dtype=np.float32) / 65536.0
            time.sleep(1.0 / SAMPLE_HZ)
        feat = np.concatenate([
            sample_states[-20:].mean(0), sample_states[-20:].std(0),
            np.diff(sample_states[-10:], axis=0).mean(0),
        ])
        features.append(feat)
        if (i+1) % 50 == 0:
            check_thermal()
            print(f"    sample {i+1}/{n_samples}, temp={get_temp()}C")

    features = np.array(features)
    split = n_samples // 2
    clf = RidgeClassifier(alpha=1.0)
    clf.fit(features[:split], labels[:split])
    return clf.score(features[split:], labels[split:])

# ======================================================================
# Scoring function — composite of rank + MC + decorrelation
# ======================================================================
def composite_score(probe):
    """Higher is better. Balances rank, MC, and decorrelation."""
    rank = probe['rank']
    mc = probe['mc']
    n_active = probe['n_active']
    mean_corr = probe['mean_corr']

    # Decorrelation score (0 = perfect, 1 = all same)
    decorr = max(0, 1.0 - mean_corr)

    # Active fraction
    active_frac = n_active / 128.0

    # Composite: we want HIGH rank, HIGH MC, LOW correlation, MODERATE activity
    # Penalize extremes: 0 or 128 active is bad
    activity_bonus = 4 * active_frac * (1 - active_frac)  # peaks at 50%

    score = rank * 0.1 + mc * 10.0 + decorr * 5.0 + activity_bonus * 3.0
    return score

# ======================================================================
# Results
# ======================================================================
results = {'experiments': {}, 'tests': {}, 'meta': {
    'script': 'z2327_fpga_criticality.py',
    'start_time': time.strftime('%Y-%m-%d %H:%M:%S'),
}}

def save():
    with open(SAVE_FILE, 'w') as f:
        json.dump(results, f, indent=2, default=lambda x: float(x) if isinstance(x, np.floating) else str(x))
    print(f"  [SAVED] {SAVE_FILE}")

print("=" * 70)
print("  z2327: FPGA Edge-of-Criticality Parameter Tuning")
print("  Synapse topology: N±1 (exc), N^32/N^64 (inh), 128-cycle hold")
print("=" * 70)
print(f"  Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
print(f"  Temp: {get_temp()}C")

# ======================================================================
# EXP 1: Fine-grained Sweep
# ======================================================================
print("\n" + "=" * 70)
print("  EXP 1: Fine-grained Threshold × Base_Exc Sweep")
print("=" * 70)

check_thermal()

thresholds = [0x16000, 0x17000, 0x18000, 0x19000, 0x1A000, 0x1C000, 0x1E000, 0x20000]
base_excs  = [0x0020, 0x0040, 0x0060, 0x0080, 0x00A0, 0x00C0, 0x0100, 0x0200]

sweep_results = []

for thresh in thresholds:
    for bexc in base_excs:
        check_thermal()
        key = f"T{thresh/65536:.3f}_E{bexc/65536:.4f}"

        try:
            fpga = FPGAEthBridge(timeout=2.0)
            setup_fpga(fpga, thresh, bexc, bias_gain=0x0800, leak=0x2000)
            probe = fpga_quick_probe(fpga, n_steps=200)
            score = composite_score(probe)
            fpga.close()
        except Exception as e:
            print(f"  {key}: ERROR {e}")
            score = 0
            probe = {'rank': 0, 'n_active': 0, 'rate_mean': 0, 'mc': 0, 'mean_corr': 1, 'total_spikes': 0}

        entry = {
            'thresh': thresh, 'base_exc': bexc, 'key': key, 'score': score,
            'rank': probe['rank'], 'n_active': probe['n_active'],
            'rate_mean': probe['rate_mean'], 'mc': probe['mc'],
            'mean_corr': probe['mean_corr'], 'total_spikes': probe['total_spikes'],
        }
        sweep_results.append(entry)
        print(f"  {key}: rank={probe['rank']:.1f}, active={probe['n_active']}, "
              f"rate={probe['rate_mean']:.0f}, MC={probe['mc']:.3f}, "
              f"corr={probe['mean_corr']:.4f}, score={score:.2f}")

# Sort by composite score
sweep_results.sort(key=lambda x: x['score'], reverse=True)
results['experiments']['EXP1_SWEEP'] = {'configs': sweep_results, 'top5': sweep_results[:5]}
save()

print(f"\n  TOP 5 configs:")
for i, cfg in enumerate(sweep_results[:5]):
    print(f"    {i+1}. {cfg['key']}: score={cfg['score']:.2f}, rank={cfg['rank']:.1f}, "
          f"MC={cfg['mc']:.3f}, active={cfg['n_active']}, corr={cfg['mean_corr']:.4f}")

# ======================================================================
# EXP 2: Full evaluation of top-5 configs
# ======================================================================
print("\n" + "=" * 70)
print("  EXP 2: Full Evaluation of Top-5 Configs")
print("=" * 70)

exp2 = {}
N_STEPS = 2000
WARMUP = 300

for i, cfg in enumerate(sweep_results[:5]):
    check_thermal()
    print(f"\n  --- Config {i+1}: {cfg['key']} (score={cfg['score']:.2f}) ---")

    fpga = FPGAEthBridge(timeout=2.0)
    setup_fpga(fpga, cfg['thresh'], cfg['base_exc'])

    rng = np.random.RandomState(42)
    inputs = rng.uniform(0, 1, N_STEPS)
    states = np.zeros((N_STEPS, NUM_NEURONS), dtype=np.float32)
    dspikes = np.zeros((N_STEPS, NUM_NEURONS), dtype=np.float32)

    for t in range(N_STEPS):
        fpga.set_mac_signal(int(inputs[t] * 65536))
        telem = fpga.read_telemetry()
        if telem is not None:
            states[t] = np.array(telem['vmem'], dtype=np.float32) / 65536.0
            dspikes[t] = np.array(telem['spike_cnt'], dtype=np.float32)
        time.sleep(1.0 / SAMPLE_HZ)
        if (t+1) % 500 == 0:
            check_thermal()
            print(f"    step {t+1}/{N_STEPS}, temp={get_temp()}C")

    fpga.close()

    # Delta spikes
    delta = np.diff(dspikes, axis=0)
    delta[delta < 0] = 0

    # Metrics on warmup-stripped data
    X = states[WARMUP:]
    inp = inputs[WARMUP:]
    X_c = X - X.mean(axis=0, keepdims=True)

    # Rank
    cov = X_c.T @ X_c / len(X_c)
    sv = np.linalg.svd(cov, compute_uv=False)
    sv = sv / (sv.sum() + 1e-20)
    sv = sv[sv > 1e-12]
    rank = float(np.exp(-np.sum(sv * np.log(sv + 1e-20))))

    # Correlation
    corr = np.corrcoef(X_c.T)
    np.fill_diagonal(corr, 0)
    mean_corr = float(np.abs(corr).mean())

    # Spike stats
    spike_rates = delta[WARMUP:].sum(axis=0)
    n_active = int((spike_rates > 0).sum())
    rate_mean = float(spike_rates.mean())
    rate_std = float(spike_rates.std())

    # MC
    mc, mc_per = compute_mc_full(X, inp, max_delay=20)

    # XOR
    xor1 = compute_xor(X, inp, tau=1)
    xor3 = compute_xor(X, inp, tau=3)
    xor5 = compute_xor(X, inp, tau=5)

    exp2[cfg['key']] = {
        'thresh': cfg['thresh'], 'base_exc': cfg['base_exc'],
        'rank': rank, 'mc': mc, 'mc_per_delay': mc_per,
        'mean_corr': mean_corr, 'n_active': n_active,
        'rate_mean': rate_mean, 'rate_std': rate_std,
        'xor1': xor1, 'xor3': xor3, 'xor5': xor5,
    }
    print(f"    rank={rank:.1f}, MC={mc:.3f}, corr={mean_corr:.4f}")
    print(f"    active={n_active}, rate_mean={rate_mean:.0f}")
    print(f"    XOR: tau1={xor1*100:.1f}%, tau3={xor3*100:.1f}%, tau5={xor5*100:.1f}%")

results['experiments']['EXP2_TOP5'] = exp2
save()

# Select best config
best_key = max(exp2, key=lambda k: exp2[k]['mc'] * 10 + exp2[k]['rank'] * 0.1 + (1-exp2[k]['mean_corr']) * 5)
best_cfg = exp2[best_key]
print(f"\n  BEST overall: {best_key}")
print(f"    rank={best_cfg['rank']:.1f}, MC={best_cfg['mc']:.3f}, corr={best_cfg['mean_corr']:.4f}")

# ======================================================================
# EXP 3: Best Config → Classification + Full Comparison
# ======================================================================
print("\n" + "=" * 70)
print(f"  EXP 3: Classification + Comparison ({best_key})")
print("=" * 70)

check_thermal()
exp3 = {}

# Classification at best config
fpga = FPGAEthBridge(timeout=2.0)
setup_fpga(fpga, best_cfg['thresh'], best_cfg['base_exc'])
print("  Running classification (200 waveforms)...")
acc_best = classify_waveforms(fpga, n_samples=200, n_classes=4, n_steps=50)
fpga.close()
print(f"  Classification accuracy = {acc_best*100:.1f}%")

check_thermal()

# Non-spiking baseline (T=0x20000, E=0x0080)
fpga2 = FPGAEthBridge(timeout=2.0)
setup_fpga(fpga2, 0x20000, 0x0080)
print("  Running non-spiking baseline classification...")
acc_nospi = classify_waveforms(fpga2, n_samples=200, n_classes=4, n_steps=50)
fpga2.close()
print(f"  Non-spiking accuracy = {acc_nospi*100:.1f}%")

exp3['classification'] = {
    'best_config_acc': acc_best,
    'non_spiking_acc': acc_nospi,
    'best_key': best_key,
}

# Comparison summary
exp3['comparison'] = {
    'non_spiking': {
        'rank': 1.21, 'mc': 0.995, 'mean_corr': 0.9999,
        'classify': acc_nospi, 'source': 'z2324/z2325'
    },
    'saturated_spiking': {
        'rank': 110.32, 'mc': 0.0, 'mean_corr': 0.064,
        'classify': 0.667, 'source': 'z2325 T=1.5/E=0.031'
    },
    'critical': {
        'rank': best_cfg['rank'], 'mc': best_cfg['mc'],
        'mean_corr': best_cfg['mean_corr'],
        'classify': acc_best, 'source': f'z2327 {best_key}'
    },
}

results['experiments']['EXP3_COMPARE'] = exp3
save()

# ======================================================================
# Test evaluation
# ======================================================================
print("\n" + "=" * 70)
print("  TEST RESULTS")
print("=" * 70)

e1 = results['experiments']['EXP1_SWEEP']
e2 = results['experiments']['EXP2_TOP5']
e3 = results['experiments']['EXP3_COMPARE']
top5 = e1['top5']
tests = {}

# EXP1: Sweep (T1104-T1107)
tests['T1104'] = {'desc': 'At least 5 configs with rank > 2',
                  'pass': sum(1 for c in e1['configs'] if c['rank'] > 2) >= 5,
                  'val': f"{sum(1 for c in e1['configs'] if c['rank'] > 2)} configs"}
tests['T1105'] = {'desc': 'At least 3 configs with MC > 0.5',
                  'pass': sum(1 for c in e1['configs'] if c['mc'] > 0.5) >= 3,
                  'val': f"{sum(1 for c in e1['configs'] if c['mc'] > 0.5)} configs"}
tests['T1106'] = {'desc': 'Top config composite score > 10',
                  'pass': top5[0]['score'] > 10,
                  'val': f"score={top5[0]['score']:.2f}"}
tests['T1107'] = {'desc': 'Transition zone exists (configs with 10 < active < 100)',
                  'pass': sum(1 for c in e1['configs'] if 10 < c['n_active'] < 100) >= 3,
                  'val': f"{sum(1 for c in e1['configs'] if 10 < c['n_active'] < 100)} configs"}

# EXP2: Full eval (T1108-T1119)
best_e2 = e2[best_key]
all_mc = [e2[k]['mc'] for k in e2]
all_rank = [e2[k]['rank'] for k in e2]
all_xor1 = [e2[k]['xor1'] for k in e2]

tests['T1108'] = {'desc': f'Best rank > 4 (significant recurrence)',
                  'pass': max(all_rank) > 4,
                  'val': f"rank={max(all_rank):.1f}"}
tests['T1109'] = {'desc': f'Best MC > 1.0',
                  'pass': max(all_mc) > 1.0,
                  'val': f"MC={max(all_mc):.3f}"}
tests['T1110'] = {'desc': f'Best MC > 3.0 (good memory)',
                  'pass': max(all_mc) > 3.0,
                  'val': f"MC={max(all_mc):.3f}"}
tests['T1111'] = {'desc': f'Best XOR-1 > 60%',
                  'pass': max(all_xor1) > 0.60,
                  'val': f"XOR-1={max(all_xor1)*100:.1f}%"}
tests['T1112'] = {'desc': f'Any config with rank > 2 AND MC > 0.5',
                  'pass': any(e2[k]['rank'] > 2 and e2[k]['mc'] > 0.5 for k in e2),
                  'val': str({k: f"rank={e2[k]['rank']:.1f},MC={e2[k]['mc']:.3f}" for k in e2})}
tests['T1113'] = {'desc': f'Best corr < 0.5 (some decorrelation)',
                  'pass': min(e2[k]['mean_corr'] for k in e2) < 0.5,
                  'val': f"corr={min(e2[k]['mean_corr'] for k in e2):.4f}"}
tests['T1114'] = {'desc': f'Any config with n_active between 16 and 96',
                  'pass': any(16 <= e2[k]['n_active'] <= 96 for k in e2),
                  'val': str({k: e2[k]['n_active'] for k in e2})}
tests['T1115'] = {'desc': f'Best XOR-3 > 55%',
                  'pass': max(e2[k]['xor3'] for k in e2) > 0.55,
                  'val': f"XOR-3={max(e2[k]['xor3'] for k in e2)*100:.1f}%"}
tests['T1116'] = {'desc': f'Best XOR-5 > 55%',
                  'pass': max(e2[k]['xor5'] for k in e2) > 0.55,
                  'val': f"XOR-5={max(e2[k]['xor5'] for k in e2)*100:.1f}%"}
tests['T1117'] = {'desc': f'Spike rate diversity (std > 0.1 × mean)',
                  'pass': any(e2[k]['rate_std'] > 0.1 * e2[k]['rate_mean'] for k in e2 if e2[k]['rate_mean'] > 0),
                  'val': str({k: f"std={e2[k]['rate_std']:.0f}/mean={e2[k]['rate_mean']:.0f}" for k in list(e2.keys())[:3]})}
tests['T1118'] = {'desc': f'MC improvement over non-spiking (>1.0 baseline)',
                  'pass': max(all_mc) > 1.0,
                  'val': f"best={max(all_mc):.3f} vs baseline=0.995"}
tests['T1119'] = {'desc': f'Rank × MC product > 5 for any config',
                  'pass': max(e2[k]['rank'] * e2[k]['mc'] for k in e2) > 5,
                  'val': f"best_product={max(e2[k]['rank'] * e2[k]['mc'] for k in e2):.2f}"}

# EXP3: Comparison (T1120-T1125)
crit = e3['comparison']['critical']
tests['T1120'] = {'desc': f'Critical accuracy > 50% (above chance)',
                  'pass': crit['classify'] > 0.50,
                  'val': f"acc={crit['classify']*100:.1f}%"}
tests['T1121'] = {'desc': f'Critical accuracy > 70%',
                  'pass': crit['classify'] > 0.70,
                  'val': f"acc={crit['classify']*100:.1f}%"}
tests['T1122'] = {'desc': f'Critical accuracy > non-spiking',
                  'pass': crit['classify'] > e3['comparison']['non_spiking']['classify'],
                  'val': f"critical={crit['classify']*100:.1f}% vs non-spike={e3['comparison']['non_spiking']['classify']*100:.1f}%"}
tests['T1123'] = {'desc': f'Critical rank > 2× non-spiking rank',
                  'pass': crit['rank'] > 2 * e3['comparison']['non_spiking']['rank'],
                  'val': f"critical={crit['rank']:.1f} vs non-spike={e3['comparison']['non_spiking']['rank']:.1f}"}
tests['T1124'] = {'desc': f'Critical achieves both rank > 2 AND MC > 0.5',
                  'pass': crit['rank'] > 2 and crit['mc'] > 0.5,
                  'val': f"rank={crit['rank']:.1f}, MC={crit['mc']:.3f}"}
tests['T1125'] = {'desc': f'Critical beats saturated on classification',
                  'pass': crit['classify'] > e3['comparison']['saturated_spiking']['classify'],
                  'val': f"critical={crit['classify']*100:.1f}% vs saturated={e3['comparison']['saturated_spiking']['classify']*100:.1f}%"}

results['tests'] = tests

n_pass = sum(1 for t in tests.values() if t['pass'])
n_total = len(tests)

for tid, t in sorted(tests.items()):
    status = "PASS" if t['pass'] else "FAIL"
    print(f"  {tid} {status}: {t['desc']} [{t['val']}]")

print(f"\n{'=' * 70}")
print(f"  SUMMARY: {n_pass}/{n_total} PASS ({100*n_pass/n_total:.0f}%)")
print(f"{'=' * 70}")

results['meta']['end_time'] = time.strftime('%Y-%m-%d %H:%M:%S')
results['meta']['n_pass'] = n_pass
results['meta']['n_total'] = n_total
results['meta']['best_config'] = best_key
save()

print(f"\n  Finished at {time.strftime('%Y-%m-%d %H:%M:%S')}")
print(f"  Results: {SAVE_FILE}")
