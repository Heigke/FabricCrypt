#!/usr/bin/env python3
"""
z2290_synapse_clusters.py — Synapse-mediated local clusters at intermediate leak
================================================================================
z2287: Sharp phase transition — low leak (eff_dim=80, MC=0) vs high leak (eff_dim=1.5, MC=1.9).
z2289: FPGA XOR=75.4% (best nonlinearity), bridge MC=2.3 (best memory).

Hypothesis: At intermediate leak, strong synapse coupling creates LOCAL clusters of
4-8 neurons that share temporal memory, while remaining independent across clusters.
This gives eff_dim ≈ 16-32 (128/4 to 128/8) with per-cluster MC > 0.

Plan:
  Phase 1: Fine leak sweep 0x0800-0x2000 (gap from z2287)
  Phase 2: At each promising leak, test synapse patterns:
    - ISOLATED: zero synapses
    - CLUSTER4: groups of 4 neurons strongly coupled, isolated between groups
    - CLUSTER8: groups of 8
    - CHAIN: forward-only propagation (asymmetric)
    - RING: circular coupling within groups
  Phase 3: Best combination → full benchmark (MC, XOR, NARMA, wave classification)

Run:
  HSA_OVERRIDE_GFX_VERSION=11.0.0 venv/bin/python scripts/z2290_synapse_clusters.py
"""

import os, sys, time, json
import numpy as np
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE / 'scripts'))
RESULTS = BASE / 'results'
RESULTS.mkdir(exist_ok=True)
SAVE_FILE = RESULTS / 'z2290_synapse_clusters.json'

from fpga_host_eth import FPGAEthBridge

NUM_NEURONS = 128
SAMPLE_HZ = 200
VG_GROUPS = {0: 0.05, 1: 0.15, 2: 0.30, 3: 0.58}
N_STEPS = 2000
WARMUP = 300
TEMP_SAFE = 55.0


def get_max_temp():
    temps = []
    for path in ['/sys/class/thermal/thermal_zone0/temp',
                 '/sys/class/hwmon/hwmon7/temp1_input']:
        try:
            with open(path, 'r') as f:
                temps.append(float(f.read().strip()) / 1000.0)
        except Exception:
            pass
    return max(temps) if temps else 0.0


def wait_cool(label="", target=None):
    if target is None:
        target = TEMP_SAFE
    temp = get_max_temp()
    if temp <= target:
        return temp
    print(f"  [TEMP] {label} {temp:.0f}°C → {target:.0f}°C...", end="", flush=True)
    t0 = time.time()
    while temp > target and (time.time() - t0) < 180:
        time.sleep(5)
        temp = get_max_temp()
        print(f" {temp:.0f}", end="", flush=True)
    print(f" OK ({time.time()-t0:.0f}s)")
    return temp


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer): return int(obj)
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        if isinstance(obj, np.bool_): return bool(obj)
        return super().default(obj)


def pack_synapse(w_nm2, w_np2, w_nm1, w_np1):
    b_nm2 = max(0, min(255, int(w_nm2 * 256)))
    b_np2 = max(0, min(255, int(w_np2 * 256)))
    b_nm1 = max(0, min(255, int(w_nm1 * 256)))
    b_np1 = max(0, min(255, int(w_np1 * 256)))
    return (b_nm2 << 24) | (b_np2 << 16) | (b_nm1 << 8) | b_np1


def apply_synapse_pattern(fpga, pattern, strength=0.5):
    """Apply synapse pattern. Strength 0.0-1.0."""
    for n in range(NUM_NEURONS):
        if pattern == 'ISOLATED':
            packed = 0x00000000
        elif pattern == 'CLUSTER4':
            # Groups of 4: strong coupling within group, zero outside
            pos_in_group = n % 4
            # Connect to neighbors within group
            w_nm1 = strength if pos_in_group > 0 else 0.0
            w_np1 = strength if pos_in_group < 3 else 0.0
            w_nm2 = strength * 0.5 if pos_in_group > 1 else 0.0
            w_np2 = strength * 0.5 if pos_in_group < 2 else 0.0
            packed = pack_synapse(w_nm2, w_np2, w_nm1, w_np1)
        elif pattern == 'CLUSTER8':
            pos_in_group = n % 8
            w_nm1 = strength if pos_in_group > 0 else 0.0
            w_np1 = strength if pos_in_group < 7 else 0.0
            w_nm2 = strength * 0.5 if pos_in_group > 1 else 0.0
            w_np2 = strength * 0.5 if pos_in_group < 6 else 0.0
            packed = pack_synapse(w_nm2, w_np2, w_nm1, w_np1)
        elif pattern == 'CHAIN':
            # Forward-only: each neuron drives next but not previous
            w_nm1 = 0.0
            w_np1 = strength
            w_nm2 = 0.0
            w_np2 = strength * 0.3
            # Break at group boundaries every 8 neurons
            if n % 8 == 7:
                w_np1 = 0.0
                w_np2 = 0.0
            packed = pack_synapse(w_nm2, w_np2, w_nm1, w_np1)
        elif pattern == 'RING':
            # Circular within groups of 8
            pos_in_group = n % 8
            w_nm1 = strength
            w_np1 = strength
            # Wrap: first and last in group connect
            if pos_in_group == 0:
                w_nm1 = 0.0  # Can't wrap with linear addressing
            if pos_in_group == 7:
                w_np1 = 0.0
            w_nm2 = 0.0
            w_np2 = 0.0
            packed = pack_synapse(w_nm2, w_np2, w_nm1, w_np1)
        elif pattern == 'HETERO_CLUSTER':
            # Heterogeneous clusters: different Vg groups get different coupling
            grp = n % 4
            pos_in_grp8 = n % 8
            if grp == 0:  # Low Vg: strong coupling
                w_nm1 = strength * 0.8 if pos_in_grp8 > 0 else 0.0
                w_np1 = strength * 0.8 if pos_in_grp8 < 7 else 0.0
            elif grp == 1:  # Medium Vg: asymmetric
                w_nm1 = strength * 0.3 if pos_in_grp8 > 0 else 0.0
                w_np1 = strength * 0.6 if pos_in_grp8 < 7 else 0.0
            elif grp == 2:  # Higher Vg: weak
                w_nm1 = strength * 0.2 if pos_in_grp8 > 0 else 0.0
                w_np1 = strength * 0.2 if pos_in_grp8 < 7 else 0.0
            else:  # Highest Vg: isolated (nonlinear nodes)
                w_nm1 = 0.0
                w_np1 = 0.0
            w_nm2 = 0.0
            w_np2 = 0.0
            packed = pack_synapse(w_nm2, w_np2, w_nm1, w_np1)
        else:
            packed = 0x00000000
        fpga.set_synapse(n, packed)
        time.sleep(0.001)
    time.sleep(0.5)


def fpga_run_continuous(fpga, u, mac_signal=None, sample_hz=SAMPLE_HZ):
    n_steps = len(u)
    if mac_signal is None:
        mac_signal = np.clip(u * 0.4 + 0.5, 0, 1)
    states = np.zeros((n_steps, NUM_NEURONS))
    dspikes = np.zeros((n_steps, NUM_NEURONS), dtype=np.float32)
    dt = 1.0 / sample_hz

    fpga.set_mac_signal(0.0)
    time.sleep(0.02)
    telem = fpga.read_telemetry()
    prev_sc = telem['spike_counts'].copy() if telem is not None else np.zeros(NUM_NEURONS, dtype=np.uint16)

    for t in range(n_steps):
        fpga.set_mac_signal(float(mac_signal[t]))
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
            states[t] = states[t - 1]
            dspikes[t] = dspikes[t - 1]
    fpga.set_mac_signal(0.0)
    return states, dspikes


def build_features(states, dspikes):
    delta = np.diff(states, axis=0)
    delta = np.vstack([np.zeros((1, states.shape[1])), delta])
    X = np.hstack([states, dspikes, delta])
    n_cols = states.shape[1]
    qi = np.arange(0, n_cols, max(1, n_cols // 32))[:32]
    vm = states[:, qi]
    ds = dspikes[:, qi]
    X = np.hstack([X, vm * ds, vm[:, :-1] * vm[:, 1:], np.square(vm)])
    return X


def ridge_solve(X_tr, y_tr, X_te, y_te, task='regression'):
    alphas = [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
    best_score = 0.0 if task == 'regression' else 0.5
    for alpha in alphas:
        I = np.eye(X_tr.shape[1])
        try:
            w = np.linalg.solve(X_tr.T @ X_tr + alpha * I, X_tr.T @ y_tr)
            pred = X_te @ w
            if task == 'regression':
                ss_res = np.sum((y_te - pred) ** 2)
                ss_tot = np.sum((y_te - y_te.mean()) ** 2)
                score = max(0, 1 - ss_res / ss_tot) if ss_tot > 1e-10 else 0.0
            else:
                score = np.mean((pred > 0.5).astype(float) == y_te)
            if score > best_score:
                best_score = score
        except Exception:
            pass
    return best_score


def quick_benchmark(X, u_raw):
    """Quick MC + XOR + diversity."""
    n = len(X)
    n_tr = int(0.7 * n)
    # MC at d=1..5
    mc_total = 0.0
    mc_d = {}
    for d in range(1, 6):
        target = u_raw[WARMUP - d:len(u_raw) - d]
        nn = min(n, len(target))
        if nn < n_tr + 20:
            mc_d[d] = 0.0
            continue
        r2 = ridge_solve(X[:n_tr], target[:n_tr], X[n_tr:nn], target[n_tr:nn])
        mc_d[d] = r2
        mc_total += r2

    # XOR τ=1
    u_a = (u_raw[WARMUP:] > 0).astype(float)
    u_b = (u_raw[WARMUP - 1:len(u_raw) - 1] > 0).astype(float)
    nn = min(len(u_a), len(u_b), n)
    target = (u_a[:nn] != u_b[:nn]).astype(float)
    xor1 = ridge_solve(X[:n_tr], target[:n_tr], X[n_tr:nn], target[n_tr:nn], 'classification')

    return mc_total, mc_d, xor1


def compute_diversity(states):
    vm = states[WARMUP:]
    vm_c = vm - vm.mean(0)
    try:
        sv = np.linalg.svd(vm_c, compute_uv=False)
        sv_n = sv / (sv.sum() + 1e-30)
        eff_dim = float(np.exp(-np.sum(sv_n * np.log(sv_n + 1e-30))))
    except:
        eff_dim = 0.0
    corr_mat = np.corrcoef(vm.T)
    mask = np.triu(np.ones_like(corr_mat, dtype=bool), k=1)
    xcorr = float(np.mean(np.abs(corr_mat[mask])))
    return eff_dim, xcorr


def main():
    print("=" * 70)
    print("  z2290: SYNAPSE CLUSTERS — Local memory at intermediate leak")
    print("  z2287: Phase transition at leak 0x0800↔0x2000")
    print("=" * 70)

    results = {
        'experiment': 'z2290_synapse_clusters',
        'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
    }

    fpga = FPGAEthBridge()
    fpga.connect()
    fpga.set_kill(0)
    time.sleep(0.3)

    fpga.set_base_exc_raw(0x0080)
    fpga.set_bias_gain_raw(0x4000)
    fpga.set_threshold_raw(0x20000)

    for n in range(NUM_NEURONS):
        fpga.set_vg(n, VG_GROUPS[n % 4])
        time.sleep(0.001)
    time.sleep(0.5)

    telem = fpga.read_telemetry()
    if telem is None:
        time.sleep(0.5)
        telem = fpga.read_telemetry()
    if telem is None:
        print("  FATAL: No FPGA telemetry")
        fpga.close()
        sys.exit(1)
    print(f"  FPGA online: vmem [{telem['vmem'].min():.3f}, {telem['vmem'].max():.3f}]")

    rng = np.random.default_rng(42)
    u = rng.uniform(-1, 1, N_STEPS).astype(np.float64)

    # ═══════════════════════════════════════════════════════════
    # Phase 1: Fine leak sweep in the gap (0x0800 to 0x2000)
    # ═══════════════════════════════════════════════════════════
    print("\n[Phase 1] Fine leak sweep 0x0800-0x2000 with ISOLATED synapses")
    print(f"  {'Leak':<10s} {'MC':>7s} {'MC(1)':>7s} {'XOR1':>7s} {'EffDim':>7s} {'xCorr':>7s}")
    print(f"  {'-'*10} {'-'*7} {'-'*7} {'-'*7} {'-'*7} {'-'*7}")

    apply_synapse_pattern(fpga, 'ISOLATED')

    leak_values = [0x0800, 0x0A00, 0x0C00, 0x0E00, 0x1000,
                   0x1200, 0x1400, 0x1600, 0x1800, 0x1A00, 0x1C00, 0x1E00, 0x2000]

    phase1 = []
    for leak in leak_values:
        fpga.set_leak_cond(leak)
        time.sleep(0.5)

        states, dspikes = fpga_run_continuous(fpga, u)
        X = build_features(states, dspikes)[WARMUP:]
        mc_tot, mc_d, xor1 = quick_benchmark(X, u)
        ed, xc = compute_diversity(states)

        phase1.append({
            'leak': hex(leak), 'mc_total': mc_tot, 'mc_d1': mc_d.get(1, 0),
            'xor1': xor1, 'eff_dim': ed, 'xcorr': xc,
        })
        print(f"  {hex(leak):<10s} {mc_tot:7.3f} {mc_d.get(1,0):7.4f} {xor1*100:6.1f}% {ed:7.1f} {xc:7.4f}")

    results['phase1_leak_sweep'] = phase1

    # Find best leak with MC > 0.5 AND eff_dim > 5
    balanced = [(p['leak'], p['mc_total'], p['eff_dim'])
                for p in phase1 if p['mc_total'] > 0.5 and p['eff_dim'] > 5.0]
    if balanced:
        # Score: MC × log(eff_dim)
        scored = [(l, mc * np.log(ed + 1)) for l, mc, ed in balanced]
        scored.sort(key=lambda x: -x[1])
        best_leak_hex = scored[0][0]
        best_leak = int(best_leak_hex, 16)
        print(f"\n  BEST balanced leak: {best_leak_hex} (MC×log(eff_dim)={scored[0][1]:.3f})")
    else:
        # Fall back to best MC
        best_p = max(phase1, key=lambda x: x['mc_total'])
        best_leak_hex = best_p['leak']
        best_leak = int(best_leak_hex, 16)
        print(f"\n  No balanced point found. BEST MC leak: {best_leak_hex}")

    results['best_leak'] = best_leak_hex

    # ═══════════════════════════════════════════════════════════
    # Phase 2: Synapse patterns at best leak + one below transition
    # ═══════════════════════════════════════════════════════════
    wait_cool("pre-Phase2")
    print(f"\n[Phase 2] Synapse patterns at leak={best_leak_hex}")

    patterns = ['ISOLATED', 'CLUSTER4', 'CLUSTER8', 'CHAIN', 'RING', 'HETERO_CLUSTER']
    strengths = [0.3, 0.6, 0.9]

    print(f"  {'Pattern':<18s} {'Str':>4s} {'MC':>7s} {'MC(1)':>7s} {'XOR1':>7s} {'EffDim':>7s}")
    print(f"  {'-'*18} {'-'*4} {'-'*7} {'-'*7} {'-'*7} {'-'*7}")

    phase2 = []
    fpga.set_leak_cond(best_leak)
    time.sleep(0.3)

    for pattern in patterns:
        for strength in strengths:
            if pattern == 'ISOLATED' and strength > 0.3:
                continue  # Only test ISOLATED once
            apply_synapse_pattern(fpga, pattern, strength)

            states, dspikes = fpga_run_continuous(fpga, u)
            X = build_features(states, dspikes)[WARMUP:]
            mc_tot, mc_d, xor1 = quick_benchmark(X, u)
            ed, xc = compute_diversity(states)

            row = {
                'pattern': pattern, 'strength': strength,
                'mc_total': mc_tot, 'mc_d1': mc_d.get(1, 0),
                'xor1': xor1, 'eff_dim': ed, 'xcorr': xc,
            }
            phase2.append(row)
            print(f"  {pattern:<18s} {strength:4.1f} {mc_tot:7.3f} {mc_d.get(1,0):7.4f} {xor1*100:6.1f}% {ed:7.1f}")

    results['phase2_synapse'] = phase2

    # Find best synapse config
    best_syn = max(phase2, key=lambda x: x['mc_total'] * (1 + 0.1 * np.log(x['eff_dim'] + 1)))
    print(f"\n  BEST: {best_syn['pattern']} str={best_syn['strength']:.1f} MC={best_syn['mc_total']:.3f} eff_dim={best_syn['eff_dim']:.1f}")
    results['best_synapse'] = {'pattern': best_syn['pattern'], 'strength': best_syn['strength']}

    # ═══════════════════════════════════════════════════════════
    # Phase 3: Full benchmark with best config
    # ═══════════════════════════════════════════════════════════
    wait_cool("pre-Phase3")
    print(f"\n[Phase 3] Full benchmark — leak={best_leak_hex}, syn={best_syn['pattern']} str={best_syn['strength']:.1f}")

    fpga.set_leak_cond(best_leak)
    apply_synapse_pattern(fpga, best_syn['pattern'], best_syn['strength'])

    # Extended run with longer sequence
    u_long = rng.uniform(-1, 1, 2500).astype(np.float64)
    states, dspikes = fpga_run_continuous(fpga, u_long)
    X = build_features(states, dspikes)[WARMUP:]

    # Full MC d=1..10
    mc_total = 0.0
    mc_per_d = {}
    n = len(X)
    n_tr = int(0.7 * n)
    for d in range(1, 11):
        target = u_long[WARMUP - d:len(u_long) - d]
        nn = min(n, len(target))
        r2 = ridge_solve(X[:n_tr], target[:n_tr], X[n_tr:nn], target[n_tr:nn])
        mc_per_d[str(d)] = r2
        mc_total += r2

    # XOR τ=1,2,3,5
    xor_results = {}
    for tau in [1, 2, 3, 5]:
        u_a = (u_long[WARMUP:] > 0).astype(float)
        u_b = (u_long[WARMUP - tau:len(u_long) - tau] > 0).astype(float)
        nn = min(len(u_a), len(u_b), n)
        target = (u_a[:nn] != u_b[:nn]).astype(float)
        Xn = X[:nn]
        acc = ridge_solve(Xn[:n_tr], target[:n_tr], Xn[n_tr:nn], target[n_tr:nn], 'classification')
        xor_results[f'xor_tau{tau}'] = acc

    # NARMA-10
    T = len(u_long)
    u_n = (u_long - u_long.min()) / (u_long.max() - u_long.min() + 1e-10) * 0.5
    y = np.zeros(T)
    for t in range(10, T):
        y[t] = 0.3 * y[t-1] + 0.05 * y[t-1] * np.sum(y[t-10:t]) + 1.5 * u_n[t-1] * u_n[t-10] + 0.1
        y[t] = np.tanh(y[t])
    target_narma = y[WARMUP:]
    nn = min(len(X), len(target_narma))
    narma_nrmse = 999.0
    for alpha in [0.01, 0.1, 1.0, 10.0, 100.0]:
        I = np.eye(X[:n_tr].shape[1])
        try:
            w = np.linalg.solve(X[:n_tr].T @ X[:n_tr] + alpha * I, X[:n_tr].T @ target_narma[:n_tr])
            pred = X[n_tr:nn] @ w
            gt = target_narma[n_tr:nn]
            nrmse = np.sqrt(np.mean((gt - pred)**2)) / (np.std(gt) + 1e-10)
            if nrmse < narma_nrmse:
                narma_nrmse = nrmse
        except Exception:
            pass

    ed, xc = compute_diversity(states)

    results['phase3'] = {
        'mc_total': mc_total, 'mc_per_delay': mc_per_d,
        'xor': xor_results,
        'narma10_nrmse': narma_nrmse,
        'eff_dim': ed, 'xcorr': xc,
    }

    print(f"  MC_total={mc_total:.3f}")
    for d in range(1, 11):
        print(f"    d={d:2d}: {mc_per_d[str(d)]:.4f}")
    print(f"  XOR: τ1={xor_results['xor_tau1']*100:.1f}% τ2={xor_results['xor_tau2']*100:.1f}% τ3={xor_results['xor_tau3']*100:.1f}% τ5={xor_results['xor_tau5']*100:.1f}%")
    print(f"  NARMA-10: NRMSE={narma_nrmse:.3f}")
    print(f"  eff_dim={ed:.1f} xcorr={xc:.4f}")

    # Compare with z2287 FPGA baseline (LEAK=0x2000, no synapses)
    wait_cool("pre-baseline")
    print("\n  [Baseline] LEAK=0x2000, ISOLATED synapses")
    fpga.set_leak_cond(0x2000)
    apply_synapse_pattern(fpga, 'ISOLATED')
    bl_s, bl_ds = fpga_run_continuous(fpga, u_long)
    bl_X = build_features(bl_s, bl_ds)[WARMUP:]
    bl_mc = 0.0
    for d in range(1, 11):
        target = u_long[WARMUP - d:len(u_long) - d]
        nn = min(len(bl_X), len(target))
        r2 = ridge_solve(bl_X[:n_tr], target[:n_tr], bl_X[n_tr:nn], target[n_tr:nn])
        bl_mc += r2
    bl_u_a = (u_long[WARMUP:] > 0).astype(float)
    bl_u_b = (u_long[WARMUP - 1:len(u_long) - 1] > 0).astype(float)
    nn = min(len(bl_u_a), len(bl_u_b), len(bl_X))
    bl_target = (bl_u_a[:nn] != bl_u_b[:nn]).astype(float)
    bl_xor1 = ridge_solve(bl_X[:n_tr], bl_target[:n_tr], bl_X[n_tr:nn], bl_target[n_tr:nn], 'classification')
    bl_ed, bl_xc = compute_diversity(bl_s)
    print(f"  Baseline: MC={bl_mc:.3f} XOR1={bl_xor1*100:.1f}% eff_dim={bl_ed:.1f}")

    results['baseline'] = {
        'mc_total': bl_mc, 'xor1': bl_xor1, 'eff_dim': bl_ed, 'xcorr': bl_xc,
    }

    # ═══════════════════════════════════════════════════════════
    # Tests
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("  KEY TESTS")
    print("=" * 70)

    tests = {}

    # T1: Best config eff_dim > 5 (meaningful independence)
    t1 = ed > 5.0
    tests['T1_eff_dim_above_5'] = {'pass': t1, 'desc': f'eff_dim={ed:.1f} > 5.0'}

    # T2: Best config MC > 0.5
    t2 = mc_total > 0.5
    tests['T2_mc_above_05'] = {'pass': t2, 'desc': f'MC={mc_total:.3f} > 0.5'}

    # T3: Synapses improve MC vs ISOLATED at same leak
    iso_at_best = [p for p in phase2 if p['pattern'] == 'ISOLATED']
    iso_mc = iso_at_best[0]['mc_total'] if iso_at_best else 0
    t3 = best_syn['mc_total'] if best_syn['pattern'] != 'ISOLATED' else False
    t3 = mc_total > iso_mc * 1.05 if best_syn['pattern'] != 'ISOLATED' else False
    tests['T3_synapses_help'] = {'pass': bool(t3), 'desc': f'Best MC={mc_total:.3f} > ISO MC={iso_mc:.3f}×1.05'}

    # T4: XOR τ=1 > 55%
    t4 = xor_results['xor_tau1'] > 0.55
    tests['T4_xor1_above_55'] = {'pass': t4, 'desc': f'XOR1={xor_results["xor_tau1"]*100:.1f}% > 55%'}

    # T5: Balanced score (MC × eff_dim) beats baseline
    score_best = mc_total * np.log(ed + 1)
    score_bl = bl_mc * np.log(bl_ed + 1)
    t5 = score_best > score_bl
    tests['T5_balanced_gt_baseline'] = {'pass': t5, 'desc': f'Score={score_best:.3f} > Baseline={score_bl:.3f}'}

    # T6: NARMA-10 < 1.0
    t6 = narma_nrmse < 1.0
    tests['T6_narma10_useful'] = {'pass': t6, 'desc': f'NARMA10={narma_nrmse:.3f} < 1.0'}

    # T7: Any synapse pattern creates eff_dim > 10 with MC > 0.3
    balanced_configs = [p for p in phase2 if p['eff_dim'] > 10 and p['mc_total'] > 0.3]
    t7 = len(balanced_configs) > 0
    tests['T7_balanced_exists'] = {'pass': t7, 'desc': f'{len(balanced_configs)} configs with eff_dim>10 AND MC>0.3'}

    # T8: Phase transition has intermediate points (not just binary)
    intermediate = [p for p in phase1 if 2.0 < p['eff_dim'] < 50.0 and p['mc_total'] > 0.1]
    t8 = len(intermediate) > 0
    tests['T8_intermediate_regime'] = {'pass': t8, 'desc': f'{len(intermediate)} leak values in intermediate regime'}

    n_pass = sum(1 for t in tests.values() if t['pass'])
    n_tests = len(tests)

    for k, v in tests.items():
        tag = k.split('_', 1)[0]
        print(f"  {tag} {'PASS' if v['pass'] else 'FAIL'}: {v['desc']}")

    print(f"\n  TOTAL: {n_pass}/{n_tests} PASS")

    results['key_tests'] = tests
    results['n_pass'] = n_pass
    results['n_tests'] = n_tests

    with open(SAVE_FILE, 'w') as f:
        json.dump(results, f, indent=2, cls=NpEncoder)
    print(f"  Results saved: {SAVE_FILE}")

    fpga.close()


if __name__ == '__main__':
    main()
