#!/usr/bin/env python3
"""
z2232_vg_transfer.py — Map the Vg→spike transfer function at fine resolution
=============================================================================
Two sweeps, MAC=0 throughout (pure avalanche physics, no software tricks):

SWEEP 1: Coarse full-range (Vg 0.10 to 0.85 in 0.01 steps)
  - Find where spikes start, where they saturate, where the cliff is
  - Measure mean rate, std across neurons, vmem distribution

SWEEP 2: Fine cliff-edge (±0.05 around cliff center, 0.002 steps)
  - 50 points around the steepest part of the transfer curve
  - Measure per-neuron rate heterogeneity
  - Test if small Vg perturbations create distinguishable states

SWEEP 3: Reservoir test at cliff edge
  - At each of 5 Vg operating points near the cliff, inject ±0.02 perturbations
  - Run 20 trials per point, measure R² from spike states (no MAC)
  - This tests: can the avalanche physics encode input through Vg alone?

All sweeps repeated for LEAK_COND = [0x0004, 0x0010, 0x0040, 0x0100]
"""

import sys, time, json
import numpy as np
sys.path.insert(0, "scripts")
from fpga_host_eth import FPGAEthBridge

N_NEURONS = 128
MEASURE_TIME = 0.3   # seconds per Vg point (coarse)
FINE_TIME = 0.5      # seconds per Vg point (fine)
TRIAL_STEPS = 60     # steps per reservoir trial
TRIAL_HZ = 200       # sampling rate for reservoir trials

LEAK_VALUES = [0x0004, 0x0010, 0x0040, 0x0100]
LEAK_NAMES  = ["0x0004", "0x0010", "0x0040", "0x0100"]


def drain_latest(fpga, max_reads=20):
    """Drain auto-telemetry buffer, return latest packet."""
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


def measure_rate_at_vg(fpga, vg, measure_time=MEASURE_TIME):
    """Set all neurons to vg, MAC=0, measure spike rate via cumulative counters."""
    fpga.set_vg_batch(0, [vg] * 64)
    fpga.set_vg_batch(64, [vg] * 64)
    time.sleep(0.08)  # let Vg settle

    # Disable auto-telem, use request-response for clean delta measurement
    fpga.disable_auto_telemetry()
    time.sleep(0.01)
    # Flush any remaining auto-telem packets
    drain_latest(fpga, max_reads=100)

    # Read baseline via request-response
    t0 = fpga.read_telemetry()
    if t0 is None:
        return None
    sc0 = t0["spike_counts"].copy()
    vm0 = t0["vmem"].copy()

    time.sleep(measure_time)

    # Read final via request-response
    t1 = fpga.read_telemetry()
    if t1 is None:
        return None
    sc1 = t1["spike_counts"].copy()
    vm1 = t1["vmem"].copy()

    # Re-enable auto-telem for reservoir sweep later
    fpga.enable_auto_telemetry(2000)
    time.sleep(0.005)

    delta = sc1.astype(np.int32) - sc0.astype(np.int32)
    delta[delta < 0] = 0  # wrap protection
    delta[delta > 60000] = 0  # saturation protection

    rates = delta.astype(np.float64) / measure_time  # per-neuron rates (spk/s)
    saturated = int(np.sum(sc1 == 65535))

    return {
        'rates': rates,
        'mean_rate': float(rates.mean()),
        'std_rate': float(rates.std()),
        'min_rate': float(rates.min()),
        'max_rate': float(rates.max()),
        'median_rate': float(np.median(rates)),
        'active_neurons': int(np.sum(rates > 1.0)),
        'vmem_mean': float(vm1.mean()),
        'vmem_std': float(vm1.std()),
        'vmem_min': float(vm1.min()),
        'vmem_max': float(vm1.max()),
        'saturated': saturated,
    }


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


def reservoir_trial(fpga, base_vg, perturbation, rng):
    """Run one trial: set Vg = base + perturbation*w_in, collect spike/vmem states.
    Returns mean state vector or None."""
    w_in = rng.standard_normal(N_NEURONS)  # unit normal per-neuron weights
    vg = np.full(N_NEURONS, base_vg) + perturbation * w_in
    vg = np.clip(vg, 0.05, 0.90)

    fpga.set_vg_batch(0, vg[:64].tolist())
    fpga.set_vg_batch(64, vg[64:].tolist())
    time.sleep(0.05)

    # Drain stale
    drain_latest(fpga)
    time.sleep(0.006)

    interval = 1.0 / TRIAL_HZ
    prev_counts = None
    states = []

    for t in range(TRIAL_STEPS):
        t0 = time.perf_counter()
        latest = drain_latest(fpga)
        if latest is None:
            time.sleep(0.003)
            latest = drain_latest(fpga)

        if latest is not None:
            counts = latest['spike_counts']
            vm = latest['vmem']
            if prev_counts is not None:
                delta = counts.astype(np.int32) - prev_counts.astype(np.int32)
                delta[delta < 0] = 0
                delta[delta > 30000] = 0
                states.append(np.concatenate([
                    delta.astype(np.float32),
                    vm.astype(np.float32)
                ]))
            prev_counts = counts.copy()

        elapsed = time.perf_counter() - t0
        remaining = interval - elapsed
        if remaining > 0.0005:
            time.sleep(remaining)

    if len(states) < 15:
        return None
    return np.array(states).mean(axis=0)


def sweep_coarse(fpga, leak_val, leak_name):
    """SWEEP 1: Vg from 0.10 to 0.85 in 0.01 steps."""
    print(f"\n{'='*70}")
    print(f"SWEEP 1 (COARSE): LEAK={leak_name}, Vg 0.10→0.85, step=0.01, MAC=0")
    print(f"{'='*70}")
    print(f"{'Vg':>6} | {'Mean':>8} {'Std':>8} {'Min':>8} {'Max':>8} {'Med':>8} {'Active':>7} {'VmM':>6} {'VmS':>6} {'Sat':>4}")
    print("-" * 90)

    fpga.set_leak_cond(leak_val)
    fpga.set_mac_signal(0.0)
    time.sleep(0.1)

    vg_values = np.arange(0.10, 0.86, 0.01)
    results = []

    for vg in vg_values:
        r = measure_rate_at_vg(fpga, float(vg))
        if r is None:
            print(f"{vg:>6.2f} | TIMEOUT")
            results.append({'vg': float(vg), 'mean_rate': 0})
            continue

        tag = " <--" if r['std_rate'] > 5 and r['mean_rate'] > 5 else ""
        print(f"{vg:>6.2f} | {r['mean_rate']:>8.1f} {r['std_rate']:>8.1f} {r['min_rate']:>8.1f} "
              f"{r['max_rate']:>8.1f} {r['median_rate']:>8.1f} {r['active_neurons']:>7d} "
              f"{r['vmem_mean']:>6.0f} {r['vmem_std']:>6.0f} {r['saturated']:>4d}{tag}")

        results.append({
            'vg': float(vg),
            'mean_rate': r['mean_rate'],
            'std_rate': r['std_rate'],
            'min_rate': r['min_rate'],
            'max_rate': r['max_rate'],
            'median_rate': r['median_rate'],
            'active_neurons': r['active_neurons'],
            'vmem_mean': r['vmem_mean'],
            'vmem_std': r['vmem_std'],
            'saturated': r['saturated'],
        })

    return results


def find_cliff_center(coarse_results):
    """Find Vg where the steepest rate increase occurs."""
    rates = [r['mean_rate'] for r in coarse_results]
    vgs = [r['vg'] for r in coarse_results]

    # Find max gradient
    best_grad = 0
    best_vg = 0.50
    for i in range(1, len(rates)):
        grad = rates[i] - rates[i-1]
        if grad > best_grad:
            best_grad = grad
            best_vg = (vgs[i] + vgs[i-1]) / 2

    # Also find the Vg where rate first exceeds 10% of max rate
    max_rate = max(rates)
    threshold_vg = best_vg
    for i, r in enumerate(rates):
        if r > 0.1 * max_rate:
            threshold_vg = vgs[i]
            break

    # Use midpoint between threshold and steepest gradient
    center = (best_vg + threshold_vg) / 2
    print(f"  Cliff analysis: steepest gradient at Vg={best_vg:.3f} ({best_grad:.1f} spk/s/0.01V)")
    print(f"  10% threshold at Vg={threshold_vg:.3f}")
    print(f"  Cliff center estimate: Vg={center:.3f}")

    return center


def sweep_fine(fpga, leak_val, leak_name, cliff_center):
    """SWEEP 2: Fine resolution around cliff edge."""
    vg_lo = max(0.05, cliff_center - 0.05)
    vg_hi = min(0.90, cliff_center + 0.05)
    step = 0.002

    print(f"\n{'='*70}")
    print(f"SWEEP 2 (FINE): LEAK={leak_name}, Vg {vg_lo:.3f}→{vg_hi:.3f}, step={step}, MAC=0")
    print(f"{'='*70}")
    print(f"{'Vg':>7} | {'Mean':>8} {'Std':>8} {'Min':>8} {'Max':>8} {'Active':>7} {'VmM':>6} {'VmS':>6}")
    print("-" * 80)

    fpga.set_leak_cond(leak_val)
    fpga.set_mac_signal(0.0)

    # Kill and reset for clean state
    fpga.set_kill(True)
    time.sleep(0.2)
    fpga.set_kill(False)
    time.sleep(0.2)

    vg_values = np.arange(vg_lo, vg_hi + step/2, step)
    results = []

    for vg in vg_values:
        r = measure_rate_at_vg(fpga, float(vg), measure_time=FINE_TIME)
        if r is None:
            print(f"{vg:>7.3f} | TIMEOUT")
            results.append({'vg': float(vg), 'mean_rate': 0})
            continue

        print(f"{vg:>7.3f} | {r['mean_rate']:>8.1f} {r['std_rate']:>8.1f} {r['min_rate']:>8.1f} "
              f"{r['max_rate']:>8.1f} {r['active_neurons']:>7d} "
              f"{r['vmem_mean']:>6.0f} {r['vmem_std']:>6.0f}")

        results.append({
            'vg': float(vg),
            'mean_rate': r['mean_rate'],
            'std_rate': r['std_rate'],
            'min_rate': r['min_rate'],
            'max_rate': r['max_rate'],
            'active_neurons': r['active_neurons'],
            'vmem_mean': r['vmem_mean'],
            'vmem_std': r['vmem_std'],
        })

    return results


def sweep_reservoir(fpga, leak_val, leak_name, cliff_center, rng):
    """SWEEP 3: Reservoir R² test at cliff edge using Vg perturbations only (no MAC)."""
    # Pick 5 operating points spanning the cliff
    op_points = [cliff_center - 0.03, cliff_center - 0.015, cliff_center,
                 cliff_center + 0.015, cliff_center + 0.03]
    op_points = [max(0.10, min(0.85, v)) for v in op_points]

    # Perturbation amplitudes to test
    perturb_amps = [0.005, 0.01, 0.02, 0.04, 0.08]

    print(f"\n{'='*70}")
    print(f"SWEEP 3 (RESERVOIR): LEAK={leak_name}, cliff={cliff_center:.3f}, MAC=0")
    print(f"  Testing Vg perturbation as input channel (pure physics)")
    print(f"{'='*70}")
    print(f"{'BaseVg':>7} {'Perturb':>8} | {'R²':>8} {'RateCorr':>9} {'RateMean':>9} {'RateStd':>8} {'N':>4}")
    print("-" * 70)

    fpga.set_leak_cond(leak_val)
    fpga.set_mac_signal(0.0)
    time.sleep(0.1)

    results = []
    n_trials = 20

    for base_vg in op_points:
        for perturb in perturb_amps:
            # Reset between combos
            fpga.set_kill(True)
            time.sleep(0.12)
            fpga.set_kill(False)
            time.sleep(0.12)
            drain_latest(fpga)

            # Run trials with random input values
            inputs = rng.uniform(-1, 1, n_trials)
            X = []
            y = []
            trial_rates = []

            for inp in inputs:
                state = reservoir_trial(fpga, base_vg, perturb * inp, rng)
                if state is not None:
                    X.append(state)
                    y.append(inp)
                    trial_rates.append(state[:N_NEURONS].mean())

            if len(X) < 10:
                print(f"{base_vg:>7.3f} {perturb:>8.3f} | {'FAIL':>8} {'':>9} {'':>9} {'':>8} {len(X):>4}")
                results.append({
                    'base_vg': float(base_vg), 'perturb': float(perturb),
                    'r2': -1, 'n_valid': len(X)
                })
                continue

            X = np.array(X)
            y = np.array(y)
            rates = np.array(trial_rates)

            # Normalize
            mu = X.mean(axis=0)
            sigma = X.std(axis=0)
            sigma[sigma < 1e-3] = 1.0
            X_n = (X - mu) / sigma

            n_tr = len(X) * 3 // 4
            r2 = ridge_r2(X_n[:n_tr], y[:n_tr], X_n[n_tr:], y[n_tr:])
            rate_corr = float(np.corrcoef(rates, y)[0, 1]) if len(rates) > 2 else 0
            if not np.isfinite(rate_corr):
                rate_corr = 0

            tag = " ***" if r2 > 0.05 else ""
            print(f"{base_vg:>7.3f} {perturb:>8.3f} | {r2:>8.4f} {rate_corr:>9.4f} "
                  f"{rates.mean():>9.2f} {rates.std():>8.4f} {len(X):>4}{tag}")

            results.append({
                'base_vg': float(base_vg),
                'perturb': float(perturb),
                'r2': float(r2),
                'rate_corr': float(rate_corr),
                'rate_mean': float(rates.mean()),
                'rate_std': float(rates.std()),
                'n_valid': len(X),
            })

    return results


def main():
    fpga = FPGAEthBridge()
    if not fpga.connect():
        print("FAIL: Cannot connect to FPGA")
        return

    fpga.set_kill(False)
    fpga.set_mac_signal(0.0)
    fpga.enable_auto_telemetry(2000)
    time.sleep(0.3)

    # Drain stale
    for _ in range(50):
        try: fpga.recv_auto_telemetry(timeout=0.005)
        except: break

    rng = np.random.default_rng(42)
    all_results = {}

    for leak_val, leak_name in zip(LEAK_VALUES, LEAK_NAMES):
        # Kill/reset between leak values
        fpga.set_kill(True)
        time.sleep(0.3)
        fpga.set_kill(False)
        time.sleep(0.3)

        # SWEEP 1: Coarse
        coarse = sweep_coarse(fpga, leak_val, leak_name)
        cliff = find_cliff_center(coarse)

        # SWEEP 2: Fine around cliff
        fine = sweep_fine(fpga, leak_val, leak_name, cliff)

        # SWEEP 3: Reservoir R² at cliff edge
        reservoir = sweep_reservoir(fpga, leak_val, leak_name, cliff, rng)

        all_results[leak_name] = {
            'leak_hex': hex(leak_val),
            'cliff_center': cliff,
            'coarse': coarse,
            'fine': fine,
            'reservoir': reservoir,
        }

    # Summary
    print(f"\n{'='*70}")
    print("GRAND SUMMARY")
    print(f"{'='*70}")

    for leak_name in LEAK_NAMES:
        r = all_results[leak_name]
        print(f"\nLEAK={leak_name} (cliff @ Vg={r['cliff_center']:.3f}):")

        # Best coarse dynamic range
        coarse_rates = [c['mean_rate'] for c in r['coarse']]
        max_rate = max(coarse_rates)
        min_nonzero = min([x for x in coarse_rates if x > 1] or [0])
        dr = max_rate / min_nonzero if min_nonzero > 0 else 0
        print(f"  Coarse: rate range {min_nonzero:.1f}→{max_rate:.1f} spk/s ({dr:.1f}x dynamic range)")

        # Best reservoir R²
        if r['reservoir']:
            best_res = max(r['reservoir'], key=lambda x: x.get('r2', -1))
            print(f"  Best reservoir R²={best_res.get('r2', -1):.4f} "
                  f"at Vg={best_res.get('base_vg', 0):.3f}, perturb={best_res.get('perturb', 0):.3f}")

            # Show all R² > 0
            good = [x for x in r['reservoir'] if x.get('r2', -1) > 0]
            if good:
                print(f"  {len(good)} combos with R²>0:")
                for g in sorted(good, key=lambda x: x['r2'], reverse=True)[:5]:
                    print(f"    Vg={g['base_vg']:.3f} perturb={g['perturb']:.3f}: "
                          f"R²={g['r2']:.4f} corr={g.get('rate_corr', 0):.4f}")

    with open("results/z2232_vg_transfer.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nSaved to results/z2232_vg_transfer.json")

    fpga.close()


if __name__ == "__main__":
    main()
