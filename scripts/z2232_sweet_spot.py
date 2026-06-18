#!/usr/bin/env python3
"""
z2232_sweet_spot.py — Find optimal LEAK_COND × BASE_VG × MAC sweet spots
=========================================================================
Quick sweep: for each parameter combo, run 10 short trials with varying
input signal, measure:
  1. Spike rate dynamic range (low vs high input)
  2. Mini R² (can readout distinguish input from spike state?)
  3. MAC effect size (how much does MAC modulate firing?)

Goal: find the combo that gives best R² > 0.
"""

import os, sys, time, json, struct
import numpy as np

os.environ.setdefault('HSA_OVERRIDE_GFX_VERSION', '11.0.0')
sys.path.insert(0, "scripts")
from fpga_host_eth import FPGAEthBridge

N_NEURONS = 128
N_TRIALS = 15       # quick trials per combo
N_STEPS = 80        # steps per trial (0.4s at 200Hz)
SAMPLE_HZ = 200

# Sweep ranges
LEAK_VALUES = [0x0004, 0x0010, 0x0040, 0x0100, 0x0400]
LEAK_NAMES  = ["0x0004", "0x0010", "0x0040", "0x0100", "0x0400"]

VG_VALUES = [0.50, 0.55, 0.58, 0.62, 0.68]

MAC_MODES = ['off', 'on']  # off=0, on=varies with input

def ridge_r2(X_tr, y_tr, X_te, y_te):
    """Quick ridge regression R²."""
    best = -999
    for a in [1e-2, 1.0, 100.0, 10000.0]:
        I = np.eye(X_tr.shape[1])
        try:
            w = np.linalg.solve(X_tr.T @ X_tr + a * I, X_tr.T @ y_tr)
        except:
            continue
        pred = X_te @ w
        ss_res = np.sum((y_te - pred) ** 2)
        ss_tot = np.sum((y_te - y_te.mean()) ** 2)
        if ss_tot < 1e-10:
            continue
        r2 = 1 - ss_res / ss_tot
        if r2 > best:
            best = r2
    return best


def drain_to_latest(fpga, max_reads=20):
    """Drain auto-telemetry buffer, return the LATEST packet."""
    latest = None
    for _ in range(max_reads):
        try:
            pkt = fpga.recv_auto_telemetry(timeout=0.001)
            if pkt is not None:
                latest = pkt
            else:
                break
        except:
            break
    return latest


def run_mini_trial(fpga, base_vg, input_val, mac_mode, rng, w_in):
    """Run one short trial, return spike state vector."""
    interval = 1.0 / SAMPLE_HZ
    prev_counts = None
    states = []

    # Set Vg with input modulation
    vg = np.full(N_NEURONS, base_vg) + 0.25 * input_val * w_in
    vg = np.clip(vg, 0.10, 0.85)
    fpga.set_vg_batch(0, vg[:64].tolist())
    fpga.set_vg_batch(64, vg[64:].tolist())

    # Set MAC
    if mac_mode == 'on':
        mac_val = 0.3 + 0.4 * (input_val + 1) / 2  # map [-1,1] → [0.1, 0.7]
        fpga.set_mac_signal(mac_val)
    else:
        fpga.set_mac_signal(0.0)

    time.sleep(0.02)  # let Vg settle

    # Get initial baseline by draining buffer
    drain_to_latest(fpga)
    time.sleep(0.006)  # wait for fresh packet

    for t in range(N_STEPS):
        t0 = time.perf_counter()

        # Drain buffer and get LATEST packet (spans full 5ms since last step)
        latest = drain_to_latest(fpga)
        if latest is None:
            # No packets, wait a bit and try once more
            time.sleep(0.003)
            latest = drain_to_latest(fpga)

        if latest is not None:
            counts = latest['spike_counts']
            vm = latest['vmem']
            if prev_counts is not None:
                delta = counts.astype(np.int32) - prev_counts.astype(np.int32)
                delta[delta < 0] = 0
                delta[delta > 30000] = 0
                states.append(np.concatenate([delta.astype(np.float32),
                                               vm.astype(np.float32)]))
            prev_counts = counts.copy()

        elapsed = time.perf_counter() - t0
        remaining = interval - elapsed
        if remaining > 0.0005:
            time.sleep(remaining)

    if len(states) < 10:
        return None

    # Return mean state over the trial
    return np.array(states).mean(axis=0)


def evaluate_combo(fpga, leak_val, base_vg, mac_mode, rng):
    """Run N_TRIALS trials with different inputs, compute R²."""
    fpga.set_leak_cond(leak_val)
    time.sleep(0.05)

    w_in = rng.standard_normal(N_NEURONS) * 0.3

    # Generate random input values
    inputs = rng.uniform(-1, 1, N_TRIALS)

    X = []
    y = []
    rates = []

    for i, inp in enumerate(inputs):
        state = run_mini_trial(fpga, base_vg, inp, mac_mode, rng, w_in)
        if state is not None:
            X.append(state)
            y.append(inp)
            # Mean spike rate
            rates.append(state[:N_NEURONS].mean())

    if len(X) < 8:
        return {'r2': -1, 'rate_mean': 0, 'rate_std': 0, 'rate_range': 0, 'rate_corr': 0, 'n_valid': len(X)}

    X = np.array(X)
    y = np.array(y)
    rates = np.array(rates)

    # Normalize features
    mu = X.mean(axis=0)
    sigma = X.std(axis=0)
    sigma[sigma < 1e-3] = 1.0
    X_n = (X - mu) / sigma

    # Split train/test
    n_tr = len(X) * 3 // 4
    r2 = ridge_r2(X_n[:n_tr], y[:n_tr], X_n[n_tr:], y[n_tr:])

    # Rate correlation with input
    rate_corr = np.corrcoef(rates, y)[0, 1] if len(rates) > 2 else 0

    return {
        'r2': float(r2),
        'rate_mean': float(rates.mean()),
        'rate_std': float(rates.std()),
        'rate_range': float(rates.max() - rates.min()),
        'rate_corr': float(rate_corr) if np.isfinite(rate_corr) else 0,
        'n_valid': len(X),
    }


def main():
    fpga = FPGAEthBridge()
    if not fpga.connect():
        print("FAIL: Cannot connect to FPGA")
        return

    fpga.set_kill(False)
    fpga.enable_auto_telemetry(2000)
    time.sleep(0.2)

    # Drain stale packets
    for _ in range(50):
        try: fpga.recv_auto_telemetry(timeout=0.005)
        except: break

    rng = np.random.default_rng(42)
    results = {}

    # Header
    print(f"{'LEAK':>8} {'VG':>6} {'MAC':>4} | {'R²':>8} {'RateM':>8} {'RateSD':>8} {'Range':>8} {'Corr':>8} {'N':>4}")
    print("-" * 80)

    for leak_val, leak_name in zip(LEAK_VALUES, LEAK_NAMES):
        for base_vg in VG_VALUES:
            for mac_mode in MAC_MODES:
                # Reset between combos
                fpga.set_kill(True)
                time.sleep(0.15)
                fpga.set_kill(False)
                time.sleep(0.15)

                # Drain
                for _ in range(30):
                    try: fpga.recv_auto_telemetry(timeout=0.003)
                    except: break

                r = evaluate_combo(fpga, leak_val, base_vg, mac_mode, rng)

                key = f"leak={leak_name}_vg={base_vg}_mac={mac_mode}"
                results[key] = r

                tag = "***" if r['r2'] > 0.05 else "   "
                print(f"{leak_name:>8} {base_vg:>6.2f} {mac_mode:>4} | "
                      f"{r['r2']:>8.4f} {r['rate_mean']:>8.1f} {r['rate_std']:>8.3f} "
                      f"{r['rate_range']:>8.3f} {r['rate_corr']:>8.4f} {r['n_valid']:>4d} {tag}")

    # Best combos
    print("\n" + "=" * 80)
    print("TOP 10 by R²:")
    sorted_r = sorted(results.items(), key=lambda x: x[1]['r2'], reverse=True)
    for i, (k, v) in enumerate(sorted_r[:10]):
        print(f"  {i+1}. {k}: R²={v['r2']:.4f}, rate={v['rate_mean']:.1f}±{v['rate_std']:.3f}, corr={v['rate_corr']:.4f}")

    with open("results/z2232_sweet_spot.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to results/z2232_sweet_spot.json")

    fpga.close()


if __name__ == "__main__":
    main()
