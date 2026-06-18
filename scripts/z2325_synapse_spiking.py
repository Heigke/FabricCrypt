#!/usr/bin/env python3
"""
z2325_synapse_spiking.py — FPGA Synapse Enhancement with Spiking-Tuned Params
===============================================================================
z2324 showed zero spikes: THRESH=0x20000 (2.0V) above vmem range (~1.3V max).
Synaptic connections are invisible without spiking (syn_contrib requires pre_spikes=1).

This experiment:
  EXP1: Sweep threshold to find spiking regime
  EXP2: At optimal threshold, measure rank, MC, spike diversity, correlations
  EXP3: Classification at optimal threshold
  EXP4: XOR at optimal threshold
  EXP5: Compare spiking vs non-spiking (does the synapse topology help?)

Tests: T1050-T1075 (26 total)

Run:
  HSA_OVERRIDE_GFX_VERSION=11.0.0 PYTHONUNBUFFERED=1 venv/bin/python scripts/z2325_synapse_spiking.py
"""

import os, sys, time, json
import numpy as np
from pathlib import Path

os.environ['PYTHONUNBUFFERED'] = '1'

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE / 'scripts'))
RESULTS = BASE / 'results'
RESULTS.mkdir(exist_ok=True)
SAVE_FILE = RESULTS / 'z2325_synapse_spiking.json'
STATES_FILE = RESULTS / 'z2325_fpga_states.npy'
DSPIKES_FILE = RESULTS / 'z2325_fpga_dspikes.npy'

from fpga_host_eth import FPGAEthBridge

NUM_NEURONS = 128
SAMPLE_HZ = 50
TEMP_PAUSE = 75.0
TEMP_RESUME = 50.0
TEMP_SAFE = 42.0
VG_GROUPS = {0: 0.05, 1: 0.15, 2: 0.30, 3: 0.58}
RIDGE_ALPHAS = [1e-4, 1e-3, 1e-2, 0.1, 1.0, 10.0]
N_STEPS = 2000
WARMUP = 200

# Threshold sweep values (Q16.16): from well above vmem to well below
THRESH_SWEEP = [
    (0x20000, '2.000'),   # baseline (no spikes expected)
    (0x18000, '1.500'),
    (0x14000, '1.250'),
    (0x10000, '1.000'),
    (0x0C000, '0.750'),
    (0x08000, '0.500'),
    (0x04000, '0.250'),
]

# BASE_EXC sweep values
BASE_EXC_SWEEP = [
    (0x0080, '0.002'),   # baseline
    (0x0200, '0.008'),
    (0x0800, '0.031'),
    (0x1000, '0.063'),
]


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer): return int(obj)
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        if isinstance(obj, np.bool_): return bool(obj)
        return super().default(obj)


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
    print(f"  [TEMP] {label} {temp:.0f}C -> {target:.0f}C...", end="", flush=True)
    t0 = time.time()
    while temp > target and (time.time() - t0) < 180:
        time.sleep(5)
        temp = get_max_temp()
        print(f" {temp:.0f}", end="", flush=True)
    print(f" OK ({time.time()-t0:.0f}s)")
    return temp


def setup_fpga(thresh=0x10000, base_exc=0x0080, bias_gain=0x4000, leak=0x2000):
    """Connect and configure FPGA with given params."""
    fpga = FPGAEthBridge(timeout=2.0)
    fpga.connect()
    fpga.set_kill(0)
    time.sleep(1.0)
    fpga.set_leak_cond(leak)
    fpga.set_threshold_raw(thresh)
    fpga.set_base_exc_raw(base_exc)
    fpga.set_bias_gain_raw(bias_gain)
    for n in range(NUM_NEURONS):
        fpga.set_vg(n, VG_GROUPS[n % 4])
        time.sleep(0.001)
    time.sleep(0.5)
    return fpga


def fpga_run_continuous(fpga, u):
    """Drive FPGA with input signal u, return (states, dspikes)."""
    n_steps = len(u)
    mac_signal = np.clip(u * 0.3 + 0.3, 0, 1)
    states = np.zeros((n_steps, NUM_NEURONS))
    dspikes = np.zeros((n_steps, NUM_NEURONS), dtype=np.float32)
    dt = 1.0 / SAMPLE_HZ
    fpga.set_mac_signal(0.0)
    time.sleep(0.02)
    telem = fpga.read_telemetry()
    prev_sc = telem['spike_counts'].copy() if telem is not None else np.zeros(NUM_NEURONS, dtype=np.uint16)
    for t in range(n_steps):
        if t > 0 and t % 50 == 0:
            temp = get_max_temp()
            if temp > TEMP_PAUSE:
                fpga.set_mac_signal(0.0)
                print(f"\n  [THERMAL PAUSE] {temp:.0f}C at step {t}/{n_steps}", end="", flush=True)
                while temp > TEMP_RESUME:
                    time.sleep(5)
                    temp = get_max_temp()
                    print(f" {temp:.0f}", end="", flush=True)
                print(" resumed", flush=True)
        fpga.set_mac_signal(float(mac_signal[t]))
        time.sleep(dt + 0.005)
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
        if t > 0 and t % 500 == 0:
            print(f"    step {t}/{n_steps}, temp={get_max_temp():.0f}C", flush=True)
    fpga.set_mac_signal(0.0)
    return states, dspikes


def fpga_quick_probe(fpga, n_steps=200):
    """Quick probe: run n_steps with random input, return spike stats."""
    rng = np.random.default_rng(42)
    u = rng.uniform(0, 1, n_steps)
    mac_signal = np.clip(u * 0.3 + 0.3, 0, 1)
    dt = 1.0 / SAMPLE_HZ

    fpga.set_mac_signal(0.0)
    time.sleep(0.02)
    telem = fpga.read_telemetry()
    prev_sc = telem['spike_counts'].copy() if telem is not None else np.zeros(NUM_NEURONS, dtype=np.uint16)

    total_spikes = np.zeros(NUM_NEURONS)
    vmem_all = np.zeros((n_steps, NUM_NEURONS))

    for t in range(n_steps):
        if t > 0 and t % 50 == 0:
            temp = get_max_temp()
            if temp > TEMP_PAUSE:
                fpga.set_mac_signal(0.0)
                while temp > TEMP_RESUME:
                    time.sleep(5)
                    temp = get_max_temp()
        fpga.set_mac_signal(float(mac_signal[t]))
        time.sleep(dt + 0.005)
        telem = fpga.read_telemetry()
        if telem is not None:
            vmem_all[t] = telem['vmem']
            sc = telem['spike_counts']
            diff = sc.astype(np.int32) - prev_sc.astype(np.int32)
            diff[diff < 0] += 65536
            total_spikes += diff
            prev_sc = sc.copy()

    fpga.set_mac_signal(0.0)
    spike_rate = total_spikes / n_steps
    vmem_mean = vmem_all[50:].mean()  # skip warmup
    vmem_max = vmem_all[50:].max()
    return {
        'spike_rate_mean': float(spike_rate.mean()),
        'spike_rate_max': float(spike_rate.max()),
        'n_active': int(np.sum(total_spikes > 0)),
        'total_spikes': float(total_spikes.sum()),
        'vmem_mean': float(vmem_mean),
        'vmem_max': float(vmem_max),
    }


def effective_rank(X):
    X_c = X - X.mean(axis=0)
    S = np.linalg.svd(X_c, compute_uv=False)
    S = S[S > 1e-10]
    if len(S) == 0:
        return 1.0
    p = S / S.sum()
    H = -np.sum(p * np.log(p))
    return float(np.exp(H))


def ridge_fast(X_train, y_train, X_test, alpha=0.01):
    d = X_train.shape[1]
    w = np.linalg.solve(X_train.T @ X_train + alpha * np.eye(d), X_train.T @ y_train)
    return X_test @ w


def compute_mc(X, u, max_delay=20):
    n = min(len(X), len(u))
    n_tr = int(0.7 * n)
    mc = 0.0
    per_delay = {}
    for d in range(1, max_delay + 1):
        target = u[max_delay - d:max_delay - d + n][:n]
        best_r2 = 0.0
        for alpha in RIDGE_ALPHAS:
            try:
                pred = ridge_fast(X[:n_tr], target[:n_tr], X[n_tr:], alpha=alpha)
                y_test = target[n_tr:]
                ss_res = np.sum((y_test - pred) ** 2)
                ss_tot = np.sum((y_test - y_test.mean()) ** 2)
                r2 = max(0.0, 1.0 - ss_res / ss_tot) if ss_tot > 1e-10 else 0.0
                if r2 > best_r2:
                    best_r2 = r2
            except Exception:
                pass
        mc += best_r2
        per_delay[d] = best_r2
    return mc, per_delay


def compute_xor(X, u, tau=1):
    n = min(len(X), len(u))
    u_bin = (u[:n] > 0.5).astype(float)
    target = np.zeros(n)
    target[tau:] = np.abs(u_bin[tau:] - u_bin[:-tau])
    n_tr = int(0.7 * n)
    best_acc = 0.5
    for alpha in RIDGE_ALPHAS:
        try:
            pred = ridge_fast(X[:n_tr], target[:n_tr], X[n_tr:], alpha=alpha)
            acc = float(np.mean((pred > 0.5).astype(float) == target[n_tr:]))
            if acc > best_acc:
                best_acc = acc
        except Exception:
            pass
    return best_acc


def build_temporal_features(states, dspikes=None, n_select=24, seed=42):
    n_steps, n_ch = states.shape
    delta = np.diff(states, axis=0)
    delta = np.vstack([np.zeros((1, n_ch)), delta])
    feats = [states, delta]
    if dspikes is not None:
        feats.append(dspikes)
    rng = np.random.default_rng(seed)
    qi = np.sort(rng.choice(n_ch, size=min(n_select, n_ch), replace=False))
    vm_q = states[:, qi]
    tau_list = [1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20]
    for tau in tau_list:
        shifted = np.zeros_like(vm_q)
        shifted[tau:] = vm_q[:-tau]
        feats.append(vm_q * shifted)
        if dspikes is not None:
            ds_q = dspikes[:, qi]
            feats.append(ds_q * shifted)
    for i, t1 in enumerate(tau_list):
        for t2 in tau_list[i+1:]:
            if t2 > 10:
                continue
            sh1 = np.zeros_like(vm_q)
            sh2 = np.zeros_like(vm_q)
            sh1[t1:] = vm_q[:-t1]
            sh2[t2:] = vm_q[:-t2]
            feats.append(vm_q * sh1 * sh2)
    feats.append(np.square(vm_q))
    feats.append((vm_q > np.median(vm_q, axis=0)).astype(float))
    return np.hstack(feats)


def pca_reduce(X, n_components=128):
    if X.shape[1] <= n_components:
        return X
    X_c = X - X.mean(axis=0)
    U, S, Vt = np.linalg.svd(X_c, full_matrices=False)
    return X_c @ Vt[:n_components].T


def generate_waveform_dataset(n_samples=200, n_steps_per=50, seed=42):
    rng = np.random.default_rng(seed)
    signals, labels = [], []
    freqs = [1.0, 2.0, 3.0, 5.0]
    n_per_class = n_samples // 4
    t = np.linspace(0, 1, n_steps_per)
    for cls in range(4):
        for i in range(n_per_class):
            f = rng.choice(freqs) * (1.0 + 0.1 * rng.standard_normal())
            phase = rng.uniform(0, 2 * np.pi)
            if cls == 0:
                sig = np.sin(2 * np.pi * f * t + phase)
            elif cls == 1:
                sig = np.sign(np.sin(2 * np.pi * f * t + phase))
            elif cls == 2:
                sig = 2 * np.abs(2 * (f * t + phase / (2*np.pi)) % 1 - 0.5) * 2 - 1
            else:
                sig = 2 * ((f * t + phase / (2*np.pi)) % 1) - 1
            sig = sig * 0.3 + 0.5
            signals.append(sig)
            labels.append(cls)
    idx = rng.permutation(len(signals))
    return [signals[i] for i in idx], [labels[i] for i in idx]


def classify_waveforms(fpga, signals, labels, use_temporal=False, seed=42):
    n_samples = len(signals)
    n_steps_per = len(signals[0])
    all_features = []
    print(f"    Collecting {n_samples} waveform responses...", flush=True)
    for idx, sig in enumerate(signals):
        if idx > 0 and idx % 20 == 0:
            temp = get_max_temp()
            if temp > TEMP_PAUSE:
                fpga.set_mac_signal(0.0)
                while temp > TEMP_RESUME:
                    time.sleep(5)
                    temp = get_max_temp()
            if idx % 100 == 0:
                print(f"    sample {idx}/{n_samples}, temp={get_max_temp():.0f}C", flush=True)
        states_block = np.zeros((n_steps_per, NUM_NEURONS))
        dspikes_block = np.zeros((n_steps_per, NUM_NEURONS), dtype=np.float32)
        dt = 1.0 / SAMPLE_HZ
        telem = fpga.read_telemetry()
        prev_sc = telem['spike_counts'].copy() if telem is not None else np.zeros(NUM_NEURONS, dtype=np.uint16)
        for t in range(n_steps_per):
            fpga.set_mac_signal(float(sig[t]))
            time.sleep(dt + 0.005)
            telem = fpga.read_telemetry()
            if telem is not None:
                states_block[t] = telem['vmem']
                sc = telem['spike_counts']
                diff = sc.astype(np.int32) - prev_sc.astype(np.int32)
                diff[diff < 0] += 65536
                dspikes_block[t] = diff.astype(np.float32)
                prev_sc = sc.copy()
            elif t > 0:
                states_block[t] = states_block[t-1]
                dspikes_block[t] = dspikes_block[t-1]
        feat_mean = states_block.mean(axis=0)
        feat_std = states_block.std(axis=0)
        feat_last = states_block[-1]
        feat_dspike_sum = dspikes_block.sum(axis=0)
        if use_temporal:
            tf = build_temporal_features(states_block, dspikes_block, n_select=16, seed=seed)
            tf_mean = tf.mean(axis=0)
            tf_std = tf.std(axis=0)
            feat = np.concatenate([feat_mean, feat_std, feat_last, feat_dspike_sum, tf_mean, tf_std])
        else:
            feat = np.concatenate([feat_mean, feat_std, feat_last, feat_dspike_sum])
        all_features.append(feat)
    fpga.set_mac_signal(0.0)
    X = np.array(all_features)
    y = np.array(labels)
    mu = X.mean(axis=0)
    sigma = X.std(axis=0)
    sigma[sigma < 1e-2] = 1.0
    X = (X - mu) / sigma
    n_tr = int(0.7 * n_samples)
    X_tr, y_tr = X[:n_tr], y[:n_tr]
    X_te, y_te = X[n_tr:], y[n_tr:]
    best_acc = 0.0
    for alpha in RIDGE_ALPHAS:
        preds = np.zeros((len(X_te), 4))
        for c in range(4):
            y_bin = (y_tr == c).astype(float)
            try:
                pred = ridge_fast(X_tr, y_bin, X_te, alpha=alpha)
                preds[:, c] = pred
            except Exception:
                pass
        pred_labels = np.argmax(preds, axis=1)
        acc = float(np.mean(pred_labels == y_te))
        if acc > best_acc:
            best_acc = acc
    return best_acc


def save_results(results):
    with open(SAVE_FILE, 'w') as f:
        json.dump(results, f, indent=2, cls=NpEncoder)
    print(f"  [SAVED] {SAVE_FILE}", flush=True)


def main():
    print("=" * 70)
    print("  z2325: FPGA Synapse Enhancement — Spiking Parameter Sweep")
    print("  Goal: Enable spiking to activate new synapse topology")
    print("=" * 70)
    print(f"  Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Temp: {get_max_temp():.0f}C")

    results = {'experiments': {}, 'tests': {}, 'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S')}
    if SAVE_FILE.exists():
        try:
            with open(SAVE_FILE) as f:
                results = json.load(f)
            done = list(results.get('experiments', {}).keys())
            if done:
                print(f"  RESUMED: {done} already done")
        except Exception:
            results = {'experiments': {}, 'tests': {}, 'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S')}

    fpga = None
    try:
        # ==============================================================
        # EXP 1 — Threshold × Base_Exc Sweep (find spiking regime)
        # ==============================================================
        if 'EXP1_SWEEP' not in results.get('experiments', {}):
            print("\n" + "=" * 70)
            print("  EXP 1: Threshold × Base_Exc Sweep")
            print("  Goal: find parameter combo that produces spiking")
            print("=" * 70)

            sweep_results = {}
            best_config = None
            best_score = -1  # score = n_active × spike_rate_mean (want many neurons spiking moderately)

            for thresh_val, thresh_label in THRESH_SWEEP:
                for bexc_val, bexc_label in BASE_EXC_SWEEP:
                    key = f"T{thresh_label}_E{bexc_label}"
                    wait_cool(f"pre-{key}", target=TEMP_SAFE)

                    fpga = setup_fpga(thresh=thresh_val, base_exc=bexc_val)
                    probe = fpga_quick_probe(fpga, n_steps=200)

                    try:
                        fpga.set_kill(1)
                        fpga.close()
                    except Exception:
                        pass
                    fpga = None

                    sweep_results[key] = {
                        'thresh_hex': hex(thresh_val),
                        'base_exc_hex': hex(bexc_val),
                        **probe
                    }

                    # Score: want many neurons active with moderate spike rate
                    # Penalize saturation (all neurons max firing) and silence (no spikes)
                    rate = probe['spike_rate_mean']
                    n_active = probe['n_active']
                    # Ideal: 50-100 active neurons, rate 0.01-0.5 per step
                    score = n_active * min(rate, 1.0) if rate > 0 else 0

                    print(f"  {key}: n_active={n_active:3d}, rate_mean={rate:.4f}, "
                          f"vmem_max={probe['vmem_max']:.3f}, total={probe['total_spikes']:.0f}, "
                          f"score={score:.2f}", flush=True)

                    if score > best_score:
                        best_score = score
                        best_config = {
                            'thresh': thresh_val,
                            'base_exc': bexc_val,
                            'key': key,
                            'score': score,
                        }

            # If no spiking found at all, try highest excitation with lowest threshold
            if best_score <= 0:
                print("\n  [FALLBACK] No spiking found! Trying extreme params...")
                wait_cool("pre-fallback", target=TEMP_SAFE)
                fpga = setup_fpga(thresh=0x02000, base_exc=0x2000, bias_gain=0x8000)
                probe = fpga_quick_probe(fpga, n_steps=200)
                try:
                    fpga.set_kill(1)
                    fpga.close()
                except Exception:
                    pass
                fpga = None
                key = "FALLBACK"
                sweep_results[key] = {
                    'thresh_hex': '0x2000', 'base_exc_hex': '0x2000', **probe
                }
                rate = probe['spike_rate_mean']
                n_active = probe['n_active']
                score = n_active * min(rate, 1.0) if rate > 0 else 0
                print(f"  {key}: n_active={n_active}, rate={rate:.4f}, score={score:.2f}")
                if score > best_score:
                    best_score = score
                    best_config = {
                        'thresh': 0x02000, 'base_exc': 0x2000,
                        'key': key, 'score': score,
                    }

            print(f"\n  BEST CONFIG: {best_config}")
            results['experiments']['EXP1_SWEEP'] = {
                'sweep': sweep_results,
                'best_config': best_config,
            }
            save_results(results)
        else:
            print("\n  EXP1_SWEEP -- already done, skipping")
            best_config = results['experiments']['EXP1_SWEEP']['best_config']

        # ==============================================================
        # EXP 2 — Full Evaluation at Optimal Threshold
        # ==============================================================
        if 'EXP2_EVAL' not in results.get('experiments', {}):
            print("\n" + "=" * 70)
            print("  EXP 2: Full Evaluation at Optimal Threshold")
            print(f"  Config: thresh={hex(best_config['thresh'])}, base_exc={hex(best_config['base_exc'])}")
            print("=" * 70)
            wait_cool("pre-EXP2", target=TEMP_SAFE)

            fpga = setup_fpga(thresh=best_config['thresh'], base_exc=best_config['base_exc'])
            telem = fpga.read_telemetry()
            if telem is not None:
                print(f"  FPGA online: vmem [{telem['vmem'].min():.4f}, {telem['vmem'].max():.4f}]")

            rng = np.random.default_rng(42)
            u_rand = rng.uniform(0, 1, N_STEPS)
            print(f"  Collecting {N_STEPS} steps...", flush=True)
            states, dspikes = fpga_run_continuous(fpga, u_rand)
            np.save(STATES_FILE, states)
            np.save(DSPIKES_FILE, dspikes)

            try:
                fpga.set_kill(1)
                fpga.close()
            except Exception:
                pass
            fpga = None

            st_w = states[WARMUP:]
            ds_w = dspikes[WARMUP:]
            u_w = u_rand[WARMUP:]

            # Effective rank
            eff_r = effective_rank(st_w)
            print(f"  Effective rank = {eff_r:.3f} (old: 1.15, z2324: 1.21)")

            # Spike stats
            spike_rates = ds_w.mean(axis=0)
            sr_mean = float(spike_rates.mean())
            sr_std = float(spike_rates.std())
            n_active = int(np.sum(ds_w.sum(axis=0) > 0))
            n_spiking = int(np.sum(spike_rates > 0.001))
            print(f"  Spikes: mean_rate={sr_mean:.4f}, std={sr_std:.4f}, active={n_active}, spiking={n_spiking}")

            # Correlation
            sub = st_w[::max(1, len(st_w)//500)]
            corr = np.corrcoef(sub.T)
            mask = ~np.eye(NUM_NEURONS, dtype=bool)
            mean_corr = float(np.nanmean(np.abs(corr[mask])))
            print(f"  Mean |correlation| = {mean_corr:.4f}")

            # MC raw
            mc_raw, mc_per = compute_mc(st_w, u_w, max_delay=20)
            print(f"  MC_raw = {mc_raw:.3f}")

            # MC with temporal features
            X_temp = build_temporal_features(st_w, ds_w, n_select=24, seed=42)
            X_temp = pca_reduce(X_temp, n_components=128)
            mc_temp, mc_per_temp = compute_mc(X_temp, u_w, max_delay=20)
            print(f"  MC_temporal = {mc_temp:.3f}")

            # XOR raw
            xor1_raw = compute_xor(st_w, u_w, tau=1)
            xor1_temp = compute_xor(X_temp, u_w, tau=1)
            print(f"  XOR_tau1: raw={xor1_raw*100:.1f}%, temporal={xor1_temp*100:.1f}%")

            # Spike diversity
            n_distinct = 0
            for i in range(NUM_NEURONS):
                diffs = np.abs(spike_rates - spike_rates[i])
                if np.sum(diffs > 0.01) >= NUM_NEURONS // 2:
                    n_distinct += 1

            results['experiments']['EXP2_EVAL'] = {
                'effective_rank': eff_r,
                'mc_raw': mc_raw,
                'mc_temporal': mc_temp,
                'mc_per_delay_raw': mc_per,
                'mc_per_delay_temporal': mc_per_temp,
                'spike_rate_mean': sr_mean,
                'spike_rate_std': sr_std,
                'n_active': n_active,
                'n_spiking': n_spiking,
                'n_distinct': n_distinct,
                'mean_abs_correlation': mean_corr,
                'xor_tau1_raw': xor1_raw,
                'xor_tau1_temporal': xor1_temp,
                'vmem_range': [float(states.min()), float(states.max())],
                'config': best_config,
            }
            save_results(results)
        else:
            print("\n  EXP2_EVAL -- already done, skipping")

        # ==============================================================
        # EXP 3 — Classification at Optimal Threshold
        # ==============================================================
        if 'EXP3_CLASSIFY' not in results.get('experiments', {}):
            print("\n" + "=" * 70)
            print("  EXP 3: 4-Class Waveform Classification (spiking regime)")
            print("=" * 70)
            wait_cool("pre-EXP3", target=TEMP_SAFE)

            fpga = setup_fpga(thresh=best_config['thresh'], base_exc=best_config['base_exc'])
            signals, labels = generate_waveform_dataset(n_samples=200, n_steps_per=50, seed=42)
            print(f"  {len(signals)} waveforms, 4 classes")

            print("\n  [RAW] Classifying...", flush=True)
            acc_raw = classify_waveforms(fpga, signals, labels, use_temporal=False, seed=42)
            print(f"  Accuracy (raw) = {acc_raw*100:.1f}%")

            wait_cool("pre-temporal-classify", target=TEMP_SAFE)

            print("\n  [TEMPORAL] Classifying...", flush=True)
            acc_temp = classify_waveforms(fpga, signals, labels, use_temporal=True, seed=42)
            print(f"  Accuracy (temporal) = {acc_temp*100:.1f}%")

            try:
                fpga.set_kill(1)
                fpga.close()
            except Exception:
                pass
            fpga = None

            results['experiments']['EXP3_CLASSIFY'] = {
                'accuracy_raw': acc_raw,
                'accuracy_temporal': acc_temp,
                'n_samples': len(signals),
                'z2264_baseline': 0.46,
            }
            save_results(results)
        else:
            print("\n  EXP3_CLASSIFY -- already done, skipping")

        # ==============================================================
        # EXP 4 — XOR at Optimal Threshold
        # ==============================================================
        if 'EXP4_XOR' not in results.get('experiments', {}):
            print("\n" + "=" * 70)
            print("  EXP 4: XOR Temporal Nonlinearity (spiking regime)")
            print("=" * 70)

            if STATES_FILE.exists() and DSPIKES_FILE.exists():
                states = np.load(STATES_FILE)
                dspikes = np.load(DSPIKES_FILE)
                rng = np.random.default_rng(42)
                u_rand = rng.uniform(0, 1, N_STEPS)
                print(f"  Using cached states: {states.shape}")
            else:
                print("  ERROR: no cached states")
                return

            st_w = states[WARMUP:]
            ds_w = dspikes[WARMUP:]
            u_w = u_rand[WARMUP:]
            X_temp = build_temporal_features(st_w, ds_w, n_select=24, seed=42)
            X_temp = pca_reduce(X_temp, n_components=128)

            xor_results = {}
            for tau in [1, 3, 5, 10]:
                xor_raw = compute_xor(st_w, u_w, tau=tau)
                xor_temp = compute_xor(X_temp, u_w, tau=tau)
                xor_results[f'tau{tau}'] = {
                    'raw': xor_raw, 'temporal': xor_temp, 'best': max(xor_raw, xor_temp),
                }
                print(f"  XOR tau={tau}: raw={xor_raw*100:.1f}%, temporal={xor_temp*100:.1f}%")

            results['experiments']['EXP4_XOR'] = xor_results
            save_results(results)
        else:
            print("\n  EXP4_XOR -- already done, skipping")

        # ==============================================================
        # EXP 5 — Compare: spiking vs non-spiking (same bitstream)
        # ==============================================================
        if 'EXP5_COMPARE' not in results.get('experiments', {}):
            print("\n" + "=" * 70)
            print("  EXP 5: Spiking vs Non-Spiking Comparison")
            print("  Same bitstream, different THRESH (spiking vs silent)")
            print("=" * 70)
            wait_cool("pre-EXP5", target=TEMP_SAFE)

            # Non-spiking run (old params: high threshold)
            fpga = setup_fpga(thresh=0x20000, base_exc=0x0080)
            rng = np.random.default_rng(42)
            u_rand = rng.uniform(0, 1, N_STEPS)
            print("  [NON-SPIKING] Collecting...", flush=True)
            states_ns, dspikes_ns = fpga_run_continuous(fpga, u_rand)
            try:
                fpga.set_kill(1)
                fpga.close()
            except Exception:
                pass
            fpga = None

            st_ns = states_ns[WARMUP:]
            ds_ns = dspikes_ns[WARMUP:]
            u_w = u_rand[WARMUP:]

            eff_r_ns = effective_rank(st_ns)
            mc_ns, _ = compute_mc(st_ns, u_w, max_delay=20)
            xor1_ns = compute_xor(st_ns, u_w, tau=1)
            spike_ns = float(ds_ns.sum())
            corr_ns = float(np.nanmean(np.abs(np.corrcoef(st_ns[::4].T)[~np.eye(NUM_NEURONS, dtype=bool)])))
            print(f"  NON-SPIKING: rank={eff_r_ns:.2f}, MC={mc_ns:.3f}, XOR={xor1_ns*100:.1f}%, "
                  f"spikes={spike_ns:.0f}, corr={corr_ns:.4f}")

            # Spiking run (optimal params)
            wait_cool("pre-spiking", target=TEMP_SAFE)
            if STATES_FILE.exists():
                states_sp = np.load(STATES_FILE)
                dspikes_sp = np.load(DSPIKES_FILE)
            else:
                fpga = setup_fpga(thresh=best_config['thresh'], base_exc=best_config['base_exc'])
                states_sp, dspikes_sp = fpga_run_continuous(fpga, u_rand)
                try:
                    fpga.set_kill(1)
                    fpga.close()
                except Exception:
                    pass
                fpga = None

            st_sp = states_sp[WARMUP:]
            ds_sp = dspikes_sp[WARMUP:]
            eff_r_sp = effective_rank(st_sp)
            mc_sp, _ = compute_mc(st_sp, u_w, max_delay=20)
            xor1_sp = compute_xor(st_sp, u_w, tau=1)
            spike_sp = float(ds_sp.sum())
            corr_sp = float(np.nanmean(np.abs(np.corrcoef(st_sp[::4].T)[~np.eye(NUM_NEURONS, dtype=bool)])))
            print(f"  SPIKING:     rank={eff_r_sp:.2f}, MC={mc_sp:.3f}, XOR={xor1_sp*100:.1f}%, "
                  f"spikes={spike_sp:.0f}, corr={corr_sp:.4f}")

            results['experiments']['EXP5_COMPARE'] = {
                'non_spiking': {
                    'effective_rank': eff_r_ns, 'mc_raw': mc_ns, 'xor_tau1': xor1_ns,
                    'total_spikes': spike_ns, 'mean_corr': corr_ns,
                    'thresh': '0x20000', 'base_exc': '0x0080',
                },
                'spiking': {
                    'effective_rank': eff_r_sp, 'mc_raw': mc_sp, 'xor_tau1': xor1_sp,
                    'total_spikes': spike_sp, 'mean_corr': corr_sp,
                    'thresh': hex(best_config['thresh']),
                    'base_exc': hex(best_config['base_exc']),
                },
                'rank_improvement': eff_r_sp - eff_r_ns,
                'mc_improvement': mc_sp - mc_ns,
                'xor_improvement': xor1_sp - xor1_ns,
                'corr_improvement': corr_ns - corr_sp,  # positive = decorrelated (better)
            }
            save_results(results)
        else:
            print("\n  EXP5_COMPARE -- already done, skipping")

        # ==============================================================
        # Evaluate all tests
        # ==============================================================
        print("\n" + "=" * 70)
        print("  TEST RESULTS")
        print("=" * 70)

        exp = results.get('experiments', {})
        tests = {}
        n_pass = 0
        n_total = 0

        def test(tid, passed, desc, **kwargs):
            nonlocal n_pass, n_total
            n_total += 1
            if passed:
                n_pass += 1
            status = "PASS" if passed else "FAIL"
            tests[tid] = {'pass': bool(passed), 'desc': desc, **kwargs}
            print(f"  {tid} {status}: {desc}", flush=True)

        # EXP1 tests
        e1 = exp.get('EXP1_SWEEP', {})
        bc = e1.get('best_config', {})
        best_score = bc.get('score', 0)
        n_configs_spiking = sum(1 for v in e1.get('sweep', {}).values() if v.get('n_active', 0) > 0)
        test('T1050', best_score > 0,
             f"Any config produces spiking (best_score={best_score:.2f})")
        test('T1051', n_configs_spiking >= 3,
             f"At least 3 configs spike ({n_configs_spiking} found)")
        test('T1052', bc.get('score', 0) > 1.0,
             f"Best config has good spiking (score={best_score:.2f} > 1.0)")

        # EXP2 tests
        e2 = exp.get('EXP2_EVAL', {})
        er = e2.get('effective_rank', 0)
        mc_r = e2.get('mc_raw', 0)
        mc_t = e2.get('mc_temporal', 0)
        n_act = e2.get('n_active', 0)
        n_spk = e2.get('n_spiking', 0)
        mean_c = e2.get('mean_abs_correlation', 1.0)
        sr_m = e2.get('spike_rate_mean', 0)
        sr_s = e2.get('spike_rate_std', 0)
        xor1_r = e2.get('xor_tau1_raw', 0.5)
        xor1_t = e2.get('xor_tau1_temporal', 0.5)
        n_dist = e2.get('n_distinct', 0)

        test('T1053', er > 2.0,
             f"eff_rank({er:.2f}) > 2.0 (break rank-1)")
        test('T1054', er > 4.0,
             f"eff_rank({er:.2f}) > 4.0 (significant recurrence)")
        test('T1055', er > 8.0,
             f"eff_rank({er:.2f}) > 8.0 (strong recurrence)")
        test('T1056', mc_r > 1.0,
             f"MC_raw({mc_r:.3f}) > 1.0 (memory beyond 1-step)")
        test('T1057', mc_r > 3.0,
             f"MC_raw({mc_r:.3f}) > 3.0 (good reservoir memory)")
        test('T1058', mc_t > 5.0,
             f"MC_temporal({mc_t:.3f}) > 5.0")
        test('T1059', n_act >= 32,
             f"n_active({n_act}) >= 32 (at least 25% neurons spike)")
        test('T1060', n_spk >= 16,
             f"n_spiking({n_spk}) >= 16 (sustained spiking)")
        test('T1061', mean_c < 0.9,
             f"mean_corr({mean_c:.4f}) < 0.9 (some decorrelation)")
        test('T1062', mean_c < 0.7,
             f"mean_corr({mean_c:.4f}) < 0.7 (significant decorrelation)")
        test('T1063', sr_s > 0.05 * sr_m if sr_m > 0 else False,
             f"spike_rate_diversity: std={sr_s:.4f} > 0.05*mean={sr_m:.4f}")
        test('T1064', xor1_r > 0.55 or xor1_t > 0.55,
             f"XOR_tau1: raw={xor1_r*100:.1f}%, temp={xor1_t*100:.1f}% > 55%")

        # EXP3 tests
        e3 = exp.get('EXP3_CLASSIFY', {})
        acc_r = e3.get('accuracy_raw', 0)
        acc_t = e3.get('accuracy_temporal', 0)
        test('T1065', acc_r > 0.50,
             f"classify_raw({acc_r*100:.1f}%) > 50% (above chance)")
        test('T1066', acc_r > 0.70,
             f"classify_raw({acc_r*100:.1f}%) > 70%")
        test('T1067', acc_t > 0.80,
             f"classify_temporal({acc_t*100:.1f}%) > 80%")
        test('T1068', max(acc_r, acc_t) > 0.46,
             f"best_classify({max(acc_r,acc_t)*100:.1f}%) > 46% (z2264 baseline)")

        # EXP4 tests
        e4 = exp.get('EXP4_XOR', {})
        xor1 = e4.get('tau1', {}).get('best', 0.5)
        xor3 = e4.get('tau3', {}).get('best', 0.5)
        xor5 = e4.get('tau5', {}).get('best', 0.5)
        xor10 = e4.get('tau10', {}).get('best', 0.5)
        test('T1069', xor1 > 0.60,
             f"XOR_tau1({xor1*100:.1f}%) > 60%")
        test('T1070', xor3 > 0.55,
             f"XOR_tau3({xor3*100:.1f}%) > 55%")
        test('T1071', xor5 > 0.52,
             f"XOR_tau5({xor5*100:.1f}%) > 52%")
        test('T1072', xor10 > 0.51,
             f"XOR_tau10({xor10*100:.1f}%) > 51%")

        # EXP5 tests
        e5 = exp.get('EXP5_COMPARE', {})
        sp = e5.get('spiking', {})
        ns = e5.get('non_spiking', {})
        rank_imp = e5.get('rank_improvement', 0)
        mc_imp = e5.get('mc_improvement', 0)
        corr_imp = e5.get('corr_improvement', 0)
        test('T1073', rank_imp > 0.5,
             f"rank_improvement({rank_imp:.2f}) > 0.5 (spiking breaks rank-1)")
        test('T1074', corr_imp > 0.05,
             f"corr_improvement({corr_imp:.4f}) > 0.05 (spiking decorrelates)")
        test('T1075', sp.get('total_spikes', 0) > 100,
             f"spiking_total_spikes({sp.get('total_spikes', 0):.0f}) > 100")

        results['tests'] = tests
        results['summary'] = {
            'total': n_total,
            'passed': n_pass,
            'failed': n_total - n_pass,
            'pass_rate': f"{n_pass}/{n_total} ({100*n_pass/n_total:.0f}%)" if n_total > 0 else "0/0",
        }
        save_results(results)

        print("\n" + "=" * 70)
        print(f"  SUMMARY: {n_pass}/{n_total} PASS ({100*n_pass/n_total:.0f}%)" if n_total > 0 else "  SUMMARY: 0/0")
        print("=" * 70)

    except Exception as e:
        print(f"\n  [FATAL ERROR] {e}", flush=True)
        import traceback; traceback.print_exc()
        results['error'] = str(e)
        save_results(results)
    finally:
        if fpga is not None:
            try:
                fpga.set_kill(1)
                fpga.close()
            except Exception:
                pass
        print(f"\n  Finished at {time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"  Results: {SAVE_FILE}")


if __name__ == '__main__':
    main()
