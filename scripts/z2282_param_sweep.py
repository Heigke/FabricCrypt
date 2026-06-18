#!/usr/bin/env python3
"""
z2282_param_sweep.py — Parameter sweep for FPGA reservoir memory capacity
==========================================================================
z2281 showed Vg heterogeneity gives Wave-4=70% (+41pp!) but MC=0 persists.
Root cause: even "subthreshold" neurons fire rapidly because:
  i_exc_min = BASE_EXC × exp(0) = 0.0125/cycle >> i_leak = 3e-5/cycle

Fix: sweep runtime parameters to find a regime where:
  1. Some neurons are truly subthreshold (integrate without spiking)
  2. Others spike at input-dependent rates
  3. The readout can recover past inputs from neuron states

Parameters to sweep:
  - LEAK_COND: {0x0004, 0x0020, 0x0100, 0x0800}
  - BASE_EXC: {0x0100, 0x0333, 0x0800}
  - BIAS_GAIN: {0x0200, 0x0800, 0x2000}
  - THRESHOLD: {0x4000, 0x8000, 0x20000}

Quick metric: MC at delay=1 (R²) from 500-step continuous task.
"""

import os, sys, json, time
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from fpga_host_eth import FPGAEthBridge

NUM_NEURONS = 128
SAMPLE_HZ = 200
N_QUICK_STEPS = 1500
WARMUP = 300

RESULTS_PATH = os.path.join(os.path.dirname(__file__), '..', 'results', 'z2282_param_sweep.json')


def ridge_mc(X_tr, y_tr, X_te, y_te):
    best_r2 = 0.0
    for alpha in [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]:
        I = np.eye(X_tr.shape[1])
        try:
            w = np.linalg.solve(X_tr.T @ X_tr + alpha*I, X_tr.T @ y_tr)
            pred = X_te @ w
            ss_res = np.sum((y_te - pred)**2)
            ss_tot = np.sum((y_te - y_te.mean())**2)
            r2 = max(0, 1 - ss_res/ss_tot) if ss_tot > 1e-10 else 0.0
            if r2 > best_r2:
                best_r2 = r2
        except Exception:
            pass
    return best_r2


def quick_mc_test(fpga, u, sample_hz=SAMPLE_HZ):
    """Run quick MC test. Returns R² at delay 1 + diagnostics."""
    n_steps = len(u)
    u_mac = (u * 0.4 + 0.5).astype(np.float64)

    states = np.zeros((n_steps, NUM_NEURONS))
    dt = 1.0 / sample_hz

    # Read baseline spike counts for differential
    telem = fpga.read_telemetry()
    prev_sc = telem['spike_counts'].copy() if telem is not None else np.zeros(NUM_NEURONS, dtype=np.uint16)

    fpga.set_mac_signal(0.0)
    time.sleep(0.05)

    dspike_states = np.zeros((n_steps, NUM_NEURONS))

    for t in range(n_steps):
        mac_val = float(np.clip(u_mac[t], 0, 1))
        fpga.set_mac_signal(mac_val)
        time.sleep(dt)
        telem = fpga.read_telemetry()
        if telem is not None:
            states[t] = telem['vmem']
            sc = telem['spike_counts']
            dspike = sc.astype(np.int32) - prev_sc.astype(np.int32)
            dspike[dspike < 0] += 65536  # counter wrap
            dspike_states[t] = dspike
            prev_sc = sc.copy()
        elif t > 0:
            states[t] = states[t-1]

    fpga.set_mac_signal(0.0)

    # Combine vmem + dspike features for readout
    X_vmem = states[WARMUP:]
    X_dspike = dspike_states[WARMUP:]
    delta = np.diff(states, axis=0)
    delta = np.vstack([np.zeros((1, NUM_NEURONS)), delta])
    X = np.hstack([X_vmem, X_dspike, delta[WARMUP:]])  # 384 features

    # MC at delays 1-5
    mc_delays = {}
    mc_total = 0.0
    for d in range(1, 6):
        target = u[WARMUP-d:n_steps-d]
        n = min(len(X), len(target))
        X_c, t_c = X[:n], target[:n]
        n_tr = int(0.7 * n)
        best_r2 = 0.0
        for alpha in [0.01, 0.1, 1.0, 10.0, 100.0]:
            r2_a = ridge_mc(X_c[:n_tr], t_c[:n_tr], X_c[n_tr:], t_c[n_tr:])
            if r2_a > best_r2:
                best_r2 = r2_a
        mc_delays[d] = best_r2
        mc_total += best_r2
    r2 = mc_delays[1]

    # Effective dimensionality of vmem states
    X_vm = states[WARMUP:]
    X_vm_c = X_vm - X_vm.mean(0)
    try:
        sv = np.linalg.svd(X_vm_c, compute_uv=False)
        sv_norm = sv / sv.sum()
        eff_dim = np.exp(-np.sum(sv_norm * np.log(sv_norm + 1e-30)))
    except Exception:
        eff_dim = 0.0

    # Per-group vmem variance (4 groups of 32 neurons each)
    group_stds = []
    for g in range(4):
        gidx = list(range(g, NUM_NEURONS, 4))
        group_stds.append(float(np.std(X_vm[:, gidx].mean(1))))

    # Also get spike stats
    telem = fpga.read_telemetry()
    sc = telem['spike_counts'] if telem is not None else np.zeros(NUM_NEURONS)

    return r2, {
        'vmem_std': float(np.std(states[WARMUP:])),
        'spike_mean': float(np.mean(sc)),
        'spike_std': float(np.std(sc)),
        'eff_dim': float(eff_dim),
        'group_stds': group_stds,
        'mc_delays': mc_delays,
        'mc_total': mc_total,
    }


def main():
    print("="*70)
    print("  z2282: PARAMETER SWEEP FOR MEMORY CAPACITY")
    print("="*70)

    fpga = FPGAEthBridge()
    fpga.connect()
    fpga.set_kill(0)
    time.sleep(0.3)

    # Set heterogeneous Vg (best from z2281)
    print("\n[1] Setting heterogeneous Vg (0.05, 0.15, 0.30, 0.58)...")
    vg_groups = {0: 0.05, 1: 0.15, 2: 0.30, 3: 0.58}
    for n in range(NUM_NEURONS):
        fpga.set_vg(n, vg_groups[n % 4])
        time.sleep(0.001)
    time.sleep(1.0)

    # Generate test signal (reusable)
    rng = np.random.default_rng(42)
    u = rng.uniform(-1, 1, N_QUICK_STEPS).astype(np.float64)

    # Parameter sweep
    sweep_configs = [
        # (name, leak_q16, base_exc_q16, bias_gain_q16, threshold_q16)
        # ROUND 5: HIGH LEAK for multi-delay memory capacity
        # Current LEAK=0x0020 → τ≈3.3s (too slow, vmem saturates)
        # Need τ≈5 steps (25ms) → LEAK≈0x1000
        # decay/update = LEAK*DT_OVER_C/2^32, ~390 updates/step at 200Hz
        #
        # Best base: EXC=0x0080, BIAS=0x4000, THRESH=0x20000
        # Sweep leak from 0x0200 to 0x4000
        ("leak_0200",  0x0200, 0x0080, 0x4000, 0x20000),
        ("leak_0400",  0x0400, 0x0080, 0x4000, 0x20000),
        ("leak_0800",  0x0800, 0x0080, 0x4000, 0x20000),
        ("leak_1000",  0x1000, 0x0080, 0x4000, 0x20000),
        ("leak_2000",  0x2000, 0x0080, 0x4000, 0x20000),
        ("leak_4000",  0x4000, 0x0080, 0x4000, 0x20000),
        # Also try with stronger MAC to compensate for faster leak
        ("leak_0800_hb", 0x0800, 0x0080, 0x8000, 0x20000),
        ("leak_1000_hb", 0x1000, 0x0080, 0x8000, 0x20000),
        ("leak_2000_hb", 0x2000, 0x0080, 0x8000, 0x20000),
        # Reference: best so far
        ("best_r4",    0x0020, 0x0080, 0x4000, 0x20000),
    ]

    results_list = []
    best_r2 = 0.0
    best_name = ""

    print(f"\n[2] Sweeping {len(sweep_configs)} configurations...")
    print(f"    {'Config':<20s} {'MC(1)':>7s} {'MC(2)':>7s} {'MC(3)':>7s} {'MC(5)':>7s} {'MCsum':>7s} {'EffDim':>7s} {'vm_std':>7s}")
    print(f"    {'-'*20} {'-'*7} {'-'*7} {'-'*7} {'-'*7} {'-'*7} {'-'*7} {'-'*7}")

    for name, leak, exc, bias, thresh in sweep_configs:
        # Apply parameters
        fpga.set_leak_cond(leak)
        fpga.set_base_exc_raw(exc)
        fpga.set_bias_gain_raw(bias)
        fpga.set_threshold_raw(thresh)
        time.sleep(0.5)  # let neurons settle

        # Run quick MC test
        r2, stats = quick_mc_test(fpga, u)
        md = stats.get('mc_delays', {})

        print(f"    {name:<20s} {md.get(1,0):7.4f} {md.get(2,0):7.4f} {md.get(3,0):7.4f} {md.get(5,0):7.4f} {stats.get('mc_total',0):7.3f} {stats['eff_dim']:7.1f} {stats['vmem_std']:7.4f}")

        results_list.append({
            'name': name,
            'params': {'leak': hex(leak), 'base_exc': hex(exc), 'bias_gain': hex(bias), 'threshold': hex(thresh)},
            'mc_d1': r2,
            'mc_total': stats.get('mc_total', 0),
            'mc_delays': {str(k): v for k, v in stats.get('mc_delays', {}).items()},
            'stats': stats,
        })

        mc_t = stats.get('mc_total', 0)
        if mc_t > best_r2:
            best_r2 = mc_t
            best_name = name

    print(f"\n    BEST: {best_name} with MC_total={best_r2:.4f}")

    # Save
    results = {
        'experiment': 'z2282_param_sweep',
        'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
        'vg_groups': {str(k): v for k, v in vg_groups.items()},
        'sweep': results_list,
        'best': {'name': best_name, 'mc_d1': best_r2},
    }
    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
    with open(RESULTS_PATH, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Results saved: {RESULTS_PATH}")

    fpga.set_kill(1)
    fpga.close()

if __name__ == '__main__':
    main()
