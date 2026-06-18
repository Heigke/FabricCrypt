#!/usr/bin/env python3
"""
z2284_optimized_bridge.py — GPU↔FPGA bridge with optimized FPGA parameters
==========================================================================
Combines z2283's breakthrough FPGA parameters with GPU fourpop reservoir:
  - FPGA: LEAK=0x2000, EXC=0x0080, BIAS=0x4000, THRESH=0x20000 (MC=1.94, Wave=100%)
  - GPU: fourpop ESN (branch/L1/ESN/wavefront) — XOR=77%, MC=0.62, Wave=97.7%
  - Bridge: GPU state → mean(abs(state)) → set_mac_signal() → FPGA bias current
  - 200Hz sampling, heterogeneous Vg (4 groups), quadratic XOR features

Conditions:
  1. FPGA_OPT:  FPGA alone with z2283 optimal params
  2. GPU_FOURPOP: GPU fourpop ESN alone (Python model)
  3. BRIDGE: GPU fourpop drives FPGA via MAC bridge (cross-substrate)

Expected: BRIDGE gets best-of-both — MC>1.5 AND XOR>75% simultaneously.

Run:
  HSA_OVERRIDE_GFX_VERSION=11.0.0 venv/bin/python scripts/z2284_optimized_bridge.py
"""

import os, sys, time, json, signal, subprocess
import numpy as np
from pathlib import Path

# ─── Paths ───
BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE / 'scripts'))
RESULTS = BASE / 'results'
RESULTS.mkdir(exist_ok=True)
SAVE_FILE = RESULTS / 'z2284_optimized_bridge.json'

from fpga_host_eth import FPGAEthBridge

# ─── FPGA Optimal Parameters (from z2283) ───
NUM_NEURONS = 128
SAMPLE_HZ = 200
OPT_LEAK = 0x2000
OPT_EXC = 0x0080
OPT_BIAS = 0x4000
OPT_THRESH = 0x20000
VG_GROUPS = {0: 0.05, 1: 0.15, 2: 0.30, 3: 0.58}

# ─── Benchmark parameters ───
N_WAVE_TRIALS = 40
N_WAVE_STEPS = 60
N_CONTINUOUS_STEPS = 2000
WARMUP = 300

# ─── Thermal Safety ───
TEMP_ABORT = 90.0
TEMP_SAFE = 55.0
TEMP_PAUSE = 75.0


# ═══════════════════════════════════════════════════════════
# Temperature Monitoring
# ═══════════════════════════════════════════════════════════

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
    print(f"  [TEMP] {label} {temp:.0f}°C — cooling to {target:.0f}°C...", end="", flush=True)
    t0 = time.time()
    while temp > target and (time.time() - t0) < 180:
        time.sleep(5)
        temp = get_max_temp()
        print(f" {temp:.0f}", end="", flush=True)
    print(f" — OK ({time.time()-t0:.0f}s)", flush=True)
    return temp


def check_abort():
    return get_max_temp() > TEMP_ABORT


# ═══════════════════════════════════════════════════════════
# GPU Fourpop ESN (Python model matching z2268 architecture)
# ═══════════════════════════════════════════════════════════

class GPUFourpopESN:
    """4-population ESN modeling GPU hardware mechanisms."""

    def __init__(self, n_per_pop=64):
        self.pp = n_per_pop
        self.N = 4 * n_per_pop
        rng = np.random.default_rng(7777)

        # Per-neuron parameters
        self.leak = np.zeros(self.N)
        self.input_w = np.zeros(self.N)
        self.thr = np.zeros(self.N)
        self.bias = np.zeros(self.N)

        for pop in range(4):
            s, e = pop * n_per_pop, (pop + 1) * n_per_pop
            self.leak[s:e] = 0.05 + 0.15 * rng.random(n_per_pop)
            self.input_w[s:e] = 0.05 + 0.20 * rng.random(n_per_pop)
            self.thr[s:e] = 0.4 + 0.5 * rng.random(n_per_pop)
            self.bias[s:e] = 0.02 * (rng.random(n_per_pop) - 0.5)

        # Recurrent weights (sparse ~10%)
        self.W_rec = rng.standard_normal((self.N, self.N)) * 0.04
        mask = rng.random((self.N, self.N)) > 0.9
        self.W_rec *= mask

        # Pop C (ESN): dense recurrent with spectral radius ~1.05
        sc, ec = 2 * n_per_pop, 3 * n_per_pop
        W_c = rng.standard_normal((n_per_pop, n_per_pop)) * 0.08
        mask_c = rng.random((n_per_pop, n_per_pop)) > 0.7
        W_c *= mask_c
        eigvals = np.abs(np.linalg.eigvals(W_c))
        sr = max(eigvals) if len(eigvals) > 0 else 1.0
        if sr > 0:
            W_c *= 1.05 / sr
        self.W_rec[sc:ec, sc:ec] = W_c

        # Pop A: branch divergence thresholds
        self.bthr = 0.5 + 0.3 * np.arange(n_per_pop) / max(n_per_pop - 1, 1)
        self.temp_c = 0.65

    def run(self, input_seq):
        """Run fourpop ESN. Returns (n_steps, N) state matrix."""
        n_steps = len(input_seq)
        pp = self.pp
        states = np.zeros((n_steps, self.N))
        v = np.zeros(self.N)
        h = np.zeros(self.N)
        slow = np.zeros(self.N)
        rng = np.random.default_rng(42)

        for t in range(n_steps):
            u = input_seq[t]
            rec = self.W_rec @ v

            # Pop A: branch divergence (memory)
            sa, ea = 0, pp
            branch_val = np.where(v[sa:ea] > self.bthr, 1.0, -1.0)
            v_a = np.tanh((1 - self.leak[sa:ea]) * v[sa:ea]
                          + self.input_w[sa:ea] * u + rec[sa:ea]
                          + self.bias[sa:ea] + 0.02 * branch_val)

            # Pop B: L1 cache conflicts (XOR nonlinearity)
            sb, eb = pp, 2 * pp
            v_b = v[sb:eb].copy()
            n_swap = max(1, pp // 10)
            swap_idx = rng.choice(pp, size=n_swap * 2, replace=False)
            for k in range(0, n_swap * 2 - 1, 2):
                v_b[swap_idx[k]], v_b[swap_idx[k + 1]] = v_b[swap_idx[k + 1]], v_b[swap_idx[k]]
            v_b = np.tanh((1 - self.leak[sb:eb]) * v_b
                          + self.input_w[sb:eb] * u + rec[sb:eb] + self.bias[sb:eb])

            # Pop C: dense ESN (NARMA)
            sc, ec = 2 * pp, 3 * pp
            v_c = np.tanh(((1 - self.leak[sc:ec]) * v[sc:ec]
                           + self.input_w[sc:ec] * u + rec[sc:ec]
                           + self.bias[sc:ec]) / self.temp_c)

            # Pop D: wavefront scheduling (classification)
            sd, ed = 3 * pp, 4 * pp
            sched_noise = rng.uniform(-1, 1, pp) * 0.01
            v_d = np.tanh((1 - self.leak[sd:ed]) * v[sd:ed]
                          + self.input_w[sd:ed] * u + rec[sd:ed]
                          + self.bias[sd:ed] + sched_noise)

            v_new = np.concatenate([v_a, v_b, v_c, v_d])
            v_new += rng.uniform(-1, 1, self.N) * 0.003  # PLL jitter

            spike_mask = v_new > self.thr
            v_new[spike_mask] -= self.thr[spike_mask]
            v = v_new

            h = 0.93 * h + 0.07 * v
            slow = 0.99 * slow + 0.01 * v
            states[t] = v + 0.3 * h + 0.1 * slow

        return states


# ═══════════════════════════════════════════════════════════
# FPGA Reservoir (optimized parameters)
# ═══════════════════════════════════════════════════════════

def fpga_setup(fpga):
    """Configure FPGA with optimal z2283 parameters."""
    fpga.set_kill(0)
    time.sleep(0.3)
    fpga.set_leak_cond(OPT_LEAK)
    fpga.set_base_exc_raw(OPT_EXC)
    fpga.set_bias_gain_raw(OPT_BIAS)
    fpga.set_threshold_raw(OPT_THRESH)
    time.sleep(0.3)

    for n in range(NUM_NEURONS):
        fpga.set_vg(n, VG_GROUPS[n % 4])
        time.sleep(0.001)
    time.sleep(1.0)

    print(f"  FPGA: LEAK={OPT_LEAK:#06x} EXC={OPT_EXC:#06x} "
          f"BIAS={OPT_BIAS:#06x} THRESH={OPT_THRESH:#06x}")
    print(f"  Vg groups: {VG_GROUPS}")


def fpga_run_sequence(fpga, input_seq, mac_override=None):
    """Run input sequence through FPGA. mac_override replaces input_seq for MAC."""
    n_steps = len(input_seq)
    states = np.zeros((n_steps, NUM_NEURONS))
    dspikes = np.zeros((n_steps, NUM_NEURONS), dtype=np.float32)
    dt = 1.0 / SAMPLE_HZ

    fpga.set_mac_signal(0.0)
    time.sleep(0.02)

    telem = fpga.read_telemetry()
    prev_sc = telem['spike_counts'].copy() if telem is not None else np.zeros(NUM_NEURONS, dtype=np.uint16)

    for t in range(n_steps):
        if mac_override is not None:
            mac_val = float(np.clip(mac_override[t], 0, 1))
        else:
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
            states[t] = states[t - 1]
            dspikes[t] = dspikes[t - 1]

    fpga.set_mac_signal(0.0)
    return states, dspikes


# ═══════════════════════════════════════════════════════════
# Bridge: GPU fourpop → MAC → FPGA
# ═══════════════════════════════════════════════════════════

def bridge_run_sequence(fpga, gpu_esn, input_seq):
    """Run GPU fourpop + FPGA with MAC bridge.
    GPU processes input, its state modulates FPGA via MAC signal.
    FPGA also receives input (scaled) via MAC baseline.
    Returns combined features from both substrates.
    """
    n_steps = len(input_seq)

    # GPU processes input
    gpu_states = gpu_esn.run(input_seq)  # (n_steps, 256)

    # Compute MAC signal from GPU state: mean(abs(state)) scaled to [0, 1]
    gpu_mac = np.zeros(n_steps)
    for t in range(n_steps):
        # Blend: 60% input + 40% GPU state
        gpu_activity = np.mean(np.abs(gpu_states[t]))
        input_component = (input_seq[t] * 0.4 + 0.5)  # map [-1,1] → [0.1, 0.9]
        gpu_component = np.clip(gpu_activity, 0, 1)
        gpu_mac[t] = 0.6 * input_component + 0.4 * gpu_component
    gpu_mac = np.clip(gpu_mac, 0, 1)

    # FPGA processes with GPU-modulated MAC
    fpga_states, fpga_dspikes = fpga_run_sequence(fpga, input_seq, mac_override=gpu_mac)

    return gpu_states, fpga_states, fpga_dspikes, gpu_mac


# ═══════════════════════════════════════════════════════════
# Feature Extraction
# ═══════════════════════════════════════════════════════════

def extract_trial_features(states, dspikes):
    """Extract features for classification (per-trial)."""
    feat_mean = states.mean(axis=0)
    feat_std = states.std(axis=0)
    feat_last = states[-1]
    ds_mean = dspikes.mean(axis=0)
    ds_std = dspikes.std(axis=0)
    ds_last = dspikes[-1]
    delta = np.diff(states, axis=0)
    feat_delta_std = delta.std(axis=0) if len(delta) > 0 else np.zeros(states.shape[1])
    return np.concatenate([feat_mean, feat_std, feat_last,
                           ds_mean, ds_std, ds_last, feat_delta_std])


def extract_gpu_trial_features(states):
    """Extract per-trial features from GPU states."""
    return np.concatenate([
        states.mean(axis=0), states.std(axis=0),
        states.max(axis=0), states.min(axis=0),
    ])


def build_continuous_features(states, dspikes, include_quad=False):
    """Build time-series features for continuous benchmarks."""
    delta = np.diff(states, axis=0)
    delta = np.vstack([np.zeros((1, states.shape[1])), delta])
    X = np.hstack([states, dspikes, delta])

    if include_quad:
        n_cols = states.shape[1]
        quad_idx = np.arange(0, n_cols, max(1, n_cols // 32))[:32]
        vm_sub = states[:, quad_idx]
        ds_sub = dspikes[:, quad_idx]
        quad_feats = [
            vm_sub * ds_sub,                        # vmem × dspike
            vm_sub[:, :-1] * vm_sub[:, 1:],         # adjacent vmem products
            np.square(vm_sub),                       # vmem²
        ]
        X = np.hstack([X] + quad_feats)

    return X


# ═══════════════════════════════════════════════════════════
# Ridge Utilities
# ═══════════════════════════════════════════════════════════

def ridge_classify(X, y, n_classes, n_splits=5):
    """Stratified k-fold ridge classification."""
    from sklearn.linear_model import RidgeClassifier
    from sklearn.model_selection import cross_val_score
    sigma = np.std(X, axis=0)
    sigma[sigma < 1e-2] = 1.0
    X_n = X / sigma
    clf = RidgeClassifier(alpha=10.0)
    scores = cross_val_score(clf, X_n, y, cv=n_splits)
    return float(scores.mean()), float(scores.std())


def ridge_mc(X_tr, y_tr, X_te, y_te):
    """Ridge regression for memory capacity R²."""
    best_r2 = 0.0
    for alpha in [0.01, 0.1, 1.0, 10.0, 100.0]:
        I = np.eye(X_tr.shape[1])
        try:
            w = np.linalg.solve(X_tr.T @ X_tr + alpha * I, X_tr.T @ y_tr)
            pred = X_te @ w
            ss_res = np.sum((y_te - pred) ** 2)
            ss_tot = np.sum((y_te - y_te.mean()) ** 2)
            r2 = max(0, 1 - ss_res / ss_tot) if ss_tot > 1e-10 else 0.0
            if r2 > best_r2:
                best_r2 = r2
        except Exception:
            pass
    return best_r2


def ridge_xor(X_tr, y_tr, X_te, y_te):
    """Ridge binary classification for XOR."""
    best_acc = 0.5
    for alpha in [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]:
        I = np.eye(X_tr.shape[1])
        try:
            w = np.linalg.solve(X_tr.T @ X_tr + alpha * I, X_tr.T @ y_tr)
            pred = X_te @ w
            acc = np.mean((pred > 0.5).astype(float) == y_te)
            if acc > best_acc:
                best_acc = acc
        except Exception:
            pass
    return best_acc


# ═══════════════════════════════════════════════════════════
# Waveform Generation
# ═══════════════════════════════════════════════════════════

def generate_waveform(cls, steps):
    t = np.linspace(0, 2 * np.pi, steps)
    if cls == 0:
        return np.sin(t)
    elif cls == 1:
        return np.sign(np.sin(t))
    elif cls == 2:
        return 2 * np.abs(2 * (t / (2 * np.pi) - np.floor(t / (2 * np.pi) + 0.5))) - 1
    else:
        return 2 * (t / (2 * np.pi) - np.floor(t / (2 * np.pi))) - 1


# ═══════════════════════════════════════════════════════════
# Main Experiment
# ═══════════════════════════════════════════════════════════

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer): return int(obj)
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        if isinstance(obj, np.bool_): return bool(obj)
        return super().default(obj)


def main():
    print("=" * 70)
    print("  z2284: OPTIMIZED GPU↔FPGA BRIDGE")
    print("  FPGA: z2283 optimal | GPU: fourpop ESN | Bridge: MAC coupling")
    print("=" * 70)

    results = {
        'experiment': 'z2284_optimized_bridge',
        'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
    }

    # ─── Connect FPGA ───
    print("\n[1] Connecting to FPGA...")
    fpga = FPGAEthBridge()
    fpga.connect()
    fpga_setup(fpga)

    # Diagnostic
    telem = fpga.read_telemetry()
    if telem is None:
        time.sleep(0.5)
        telem = fpga.read_telemetry()
    if telem is None:
        print("  FATAL: No FPGA telemetry")
        fpga.close()
        sys.exit(1)
    print(f"  vmem: [{telem['vmem'].min():.3f}, {telem['vmem'].max():.3f}]")
    print(f"  spikes: mean={telem['spike_counts'].mean():.0f}")

    # ─── Initialize GPU fourpop ESN ───
    print("\n[2] Initializing GPU fourpop ESN (Python model)...")
    gpu_esn = GPUFourpopESN(n_per_pop=64)
    print(f"  {gpu_esn.N} neurons (4 pops × 64)")

    # ═══════════════════════════════════════
    # BENCHMARK 1: Waveform Classification
    # ═══════════════════════════════════════

    for n_classes in [4, 8]:
        label = f"{n_classes}-class"
        print(f"\n{'=' * 50}")
        print(f"  BENCHMARK: {label} Waveform Classification")
        print(f"{'=' * 50}")

        wave_results = {}
        for cond in ['FPGA_OPT', 'GPU_FOURPOP', 'BRIDGE']:
            if check_abort():
                print(f"  *** THERMAL ABORT — skipping {cond} ***")
                continue
            wait_cool(f"Before {label} {cond}")

            print(f"\n  [{cond}] {N_WAVE_TRIALS} trials × {N_WAVE_STEPS} steps @ {SAMPLE_HZ}Hz...")
            X_all, y_all = [], []

            for trial in range(N_WAVE_TRIALS):
                for cls in range(n_classes):
                    wf = generate_waveform(cls % 4, N_WAVE_STEPS)
                    if n_classes == 8:
                        # 8-class: vary frequency for classes 4-7
                        if cls >= 4:
                            wf = generate_waveform(cls % 4, N_WAVE_STEPS)
                            wf = np.interp(np.linspace(0, 1, N_WAVE_STEPS),
                                           np.linspace(0, 1, N_WAVE_STEPS * 2), np.tile(wf, 2))
                    wf_norm = (wf - wf.min()) / (wf.max() - wf.min() + 1e-10)
                    wf_scaled = wf_norm * 0.8 + 0.1
                    wf_input = wf_norm * 2 - 1  # [-1, 1] for GPU

                    if cond == 'FPGA_OPT':
                        states, spk = fpga_run_sequence(fpga, wf_scaled)
                        feat = extract_trial_features(states, spk)
                    elif cond == 'GPU_FOURPOP':
                        gpu_st = gpu_esn.run(wf_input)
                        feat = extract_gpu_trial_features(gpu_st)
                    elif cond == 'BRIDGE':
                        gpu_st, fpga_st, fpga_spk, mac = bridge_run_sequence(
                            fpga, gpu_esn, wf_input)
                        feat_fpga = extract_trial_features(fpga_st, fpga_spk)
                        feat_gpu = extract_gpu_trial_features(gpu_st)
                        feat = np.concatenate([feat_fpga, feat_gpu])

                    X_all.append(feat)
                    y_all.append(cls)

                if (trial + 1) % 10 == 0:
                    print(f"    trial {trial + 1}/{N_WAVE_TRIALS} (T={get_max_temp():.0f}°C)")

                # Brief thermal check between trials
                if get_max_temp() > TEMP_PAUSE:
                    print(f"    thermal pause...", flush=True)
                    time.sleep(10)

            X_all = np.array(X_all)
            y_all = np.array(y_all)
            acc, std = ridge_classify(X_all, y_all, n_classes)
            print(f"  [{cond}] {label}: {acc:.1%} ± {std:.1%}")
            wave_results[cond] = {'accuracy': float(acc), 'std': float(std)}

        results[f'wave_{n_classes}class'] = wave_results

    # ═══════════════════════════════════════
    # BENCHMARK 2: Continuous (MC, XOR)
    # ═══════════════════════════════════════

    print(f"\n{'=' * 50}")
    print(f"  BENCHMARK: Continuous (MC + XOR)")
    print(f"{'=' * 50}")

    rng = np.random.default_rng(42)
    u = rng.uniform(-1, 1, N_CONTINUOUS_STEPS).astype(np.float64)
    u_mac = (u * 0.4 + 0.5).astype(np.float64)
    u_bin = (u > 0).astype(float)

    cont_results = {}

    for cond in ['FPGA_OPT', 'GPU_FOURPOP', 'BRIDGE']:
        if check_abort():
            print(f"  *** THERMAL ABORT — skipping {cond} ***")
            continue
        wait_cool(f"Before continuous {cond}")

        print(f"\n  [{cond}] {N_CONTINUOUS_STEPS} steps @ {SAMPLE_HZ}Hz...")
        cond_res = {}

        if cond == 'FPGA_OPT':
            states, spk = fpga_run_sequence(fpga, u_mac)
            X_lin = build_continuous_features(states, spk, include_quad=False)
            X_quad = build_continuous_features(states, spk, include_quad=True)
        elif cond == 'GPU_FOURPOP':
            gpu_st = gpu_esn.run(u)
            # GPU has no dspikes — use zeros
            zero_spk = np.zeros_like(gpu_st)
            X_lin = build_continuous_features(gpu_st, zero_spk, include_quad=False)
            X_quad = build_continuous_features(gpu_st, zero_spk, include_quad=True)
        elif cond == 'BRIDGE':
            gpu_st, fpga_st, fpga_spk, mac = bridge_run_sequence(fpga, gpu_esn, u)
            # Combine both substrates
            fpga_X = build_continuous_features(fpga_st, fpga_spk, include_quad=True)
            gpu_zero_spk = np.zeros_like(gpu_st)
            gpu_X = build_continuous_features(gpu_st, gpu_zero_spk, include_quad=False)
            X_lin = np.hstack([fpga_X, gpu_X])  # large feature set
            X_quad = X_lin  # already includes quadratic from FPGA side

        # Memory Capacity (delays 1-10)
        mc_total = 0.0
        mc_per_delay = {}
        for d in range(1, 11):
            X = X_lin[WARMUP:]
            target = u[WARMUP - d:N_CONTINUOUS_STEPS - d]
            n = min(len(X), len(target))
            X_c, t_c = X[:n], target[:n]
            n_tr = int(0.7 * n)
            r2 = ridge_mc(X_c[:n_tr], t_c[:n_tr], X_c[n_tr:], t_c[n_tr:])
            mc_total += r2
            mc_per_delay[str(d)] = float(r2)
            print(f"    MC delay={d:2d}: R²={r2:.4f}")
        cond_res['mc_total'] = mc_total
        cond_res['mc_per_delay'] = mc_per_delay
        print(f"    Total MC = {mc_total:.3f}")

        # XOR (linear + quadratic readout)
        for tau in [1, 2, 3]:
            target = np.zeros(N_CONTINUOUS_STEPS)
            for t in range(tau, N_CONTINUOUS_STEPS):
                target[t] = float(int(u_bin[t]) ^ int(u_bin[t - tau]))

            y_xor = target[WARMUP:]
            n_tr = int(0.7 * len(y_xor))

            # Linear
            X_l = X_lin[WARMUP:]
            acc_lin = ridge_xor(X_l[:n_tr], y_xor[:n_tr], X_l[n_tr:], y_xor[n_tr:])

            # Quadratic
            X_q = X_quad[WARMUP:]
            acc_quad = ridge_xor(X_q[:n_tr], y_xor[:n_tr], X_q[n_tr:], y_xor[n_tr:])

            cond_res[f'xor{tau}_lin'] = float(acc_lin)
            cond_res[f'xor{tau}_quad'] = float(acc_quad)
            print(f"    XOR tau={tau}: linear={acc_lin:.1%}, quadratic={acc_quad:.1%}")

        cont_results[cond] = cond_res
        results['continuous'] = cont_results

        # Save intermediate
        with open(SAVE_FILE, 'w') as f:
            json.dump(results, f, cls=NpEncoder, indent=2)

    # ═══════════════════════════════════════
    # Summary
    # ═══════════════════════════════════════

    print("\n" + "=" * 70)
    print("  z2284: SUMMARY")
    print("=" * 70)

    def g(key, cond, metric, default=0):
        return results.get(key, {}).get(cond, {}).get(metric, default)

    print(f"\n  {'Condition':<16} {'W4':>7} {'W8':>7} {'MC':>7} {'XOR1q':>7} {'XOR2q':>7}")
    print(f"  {'-' * 16} {'-' * 7} {'-' * 7} {'-' * 7} {'-' * 7} {'-' * 7}")
    for cond in ['FPGA_OPT', 'GPU_FOURPOP', 'BRIDGE']:
        w4 = g('wave_4class', cond, 'accuracy')
        w8 = g('wave_8class', cond, 'accuracy')
        mc = cont_results.get(cond, {}).get('mc_total', 0)
        x1 = cont_results.get(cond, {}).get('xor1_quad', 0)
        x2 = cont_results.get(cond, {}).get('xor2_quad', 0)
        print(f"  {cond:<16} {w4:>6.1%} {w8:>6.1%} {mc:>7.3f} {x1:>6.1%} {x2:>6.1%}")

    # Key tests
    print(f"\n  KEY TESTS:")
    tests = {}
    n_pass = 0

    def check(name, cond, desc):
        nonlocal n_pass
        s = "PASS" if cond else "FAIL"
        if cond:
            n_pass += 1
        print(f"    {name}: {s} — {desc}")
        tests[name] = {'pass': bool(cond), 'desc': desc}

    fpga_mc = cont_results.get('FPGA_OPT', {}).get('mc_total', 0)
    gpu_mc = cont_results.get('GPU_FOURPOP', {}).get('mc_total', 0)
    bridge_mc = cont_results.get('BRIDGE', {}).get('mc_total', 0)

    fpga_xor1 = max(cont_results.get('FPGA_OPT', {}).get('xor1_lin', 0.5),
                    cont_results.get('FPGA_OPT', {}).get('xor1_quad', 0.5))
    gpu_xor1 = max(cont_results.get('GPU_FOURPOP', {}).get('xor1_lin', 0.5),
                   cont_results.get('GPU_FOURPOP', {}).get('xor1_quad', 0.5))
    bridge_xor1 = max(cont_results.get('BRIDGE', {}).get('xor1_lin', 0.5),
                      cont_results.get('BRIDGE', {}).get('xor1_quad', 0.5))

    fpga_w4 = g('wave_4class', 'FPGA_OPT', 'accuracy')
    gpu_w4 = g('wave_4class', 'GPU_FOURPOP', 'accuracy')
    bridge_w4 = g('wave_4class', 'BRIDGE', 'accuracy')

    check("T1_bridge_mc_above_1",
          bridge_mc > 1.0,
          f"BRIDGE MC={bridge_mc:.3f} > 1.0")
    check("T2_bridge_mc_vs_gpu",
          bridge_mc > gpu_mc,
          f"BRIDGE MC={bridge_mc:.3f} > GPU MC={gpu_mc:.3f}")
    check("T3_bridge_xor_above_60",
          bridge_xor1 > 0.60,
          f"BRIDGE XOR1={bridge_xor1:.1%} > 60%")
    check("T4_bridge_xor_vs_fpga",
          bridge_xor1 > fpga_xor1,
          f"BRIDGE XOR1={bridge_xor1:.1%} > FPGA XOR1={fpga_xor1:.1%}")
    check("T5_bridge_w4_above_80",
          bridge_w4 > 0.80,
          f"BRIDGE Wave4={bridge_w4:.1%} > 80%")
    check("T6_bridge_w4_best",
          bridge_w4 >= max(fpga_w4, gpu_w4) - 0.05,
          f"BRIDGE Wave4={bridge_w4:.1%} >= max(FPGA={fpga_w4:.1%}, GPU={gpu_w4:.1%})-5pp")
    check("T7_simultaneous_mc_xor",
          bridge_mc > 1.0 and bridge_xor1 > 0.55,
          f"MC={bridge_mc:.3f}>1.0 AND XOR1={bridge_xor1:.1%}>55% simultaneously")
    check("T8_bridge_beats_both_somewhere",
          (bridge_mc > max(fpga_mc, gpu_mc) or
           bridge_xor1 > max(fpga_xor1, gpu_xor1) or
           bridge_w4 > max(fpga_w4, gpu_w4)),
          "BRIDGE best on at least one metric")

    results['key_tests'] = tests
    results['n_pass'] = n_pass
    results['n_tests'] = len(tests)
    print(f"\n  KEY TESTS: {n_pass}/{len(tests)} PASS")

    # Save final
    with open(SAVE_FILE, 'w') as f:
        json.dump(results, f, cls=NpEncoder, indent=2)
    print(f"\n  Results saved: {SAVE_FILE}")

    fpga.set_kill(1)
    fpga.close()


if __name__ == '__main__':
    main()
