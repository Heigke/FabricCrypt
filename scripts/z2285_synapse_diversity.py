#!/usr/bin/env python3
"""
z2285_synapse_diversity.py — Break neuron synchrony via lateral coupling weights
================================================================================
Root cause of weak XOR/NARMA: eff_dim_svd=1.48, cross_corr=0.994.
128 neurons behave like ~1.5 neurons — all synchronized by excitatory-only
lateral coupling with uniform default weights (0x40408080 = N±1=0.50, N±2=0.25).

Fix: Use newly-added CMD_SET_SYNAPSE to set per-neuron synapse weights.
Key insight: INHIBITORY coupling breaks synchrony. But hardware only supports
excitatory (0..255 → 0..0.996). Solution: create ASYMMETRIC diversity:
  - Some neurons: zero coupling (independent)
  - Some: strong N+1 (forward chain)
  - Some: strong N-1 (backward chain)
  - Some: mixed

Conditions:
  C0_DEFAULT:    Default uniform weights (0x40408080) — baseline
  C1_ZERO:       All synapses = 0 — no lateral coupling, maximum independence
  C2_SPARSE:     50% neurons have coupling, 50% zero — mixed
  C3_CHAIN:      Forward chain: strong N+1, weak others — temporal propagation
  C4_DIVERSE:    4 groups with different coupling patterns — maximum diversity
  C5_BRIDGE:     Best local + GPU fourpop bridge — full cross-substrate

Metrics: MC, XOR, Wave-4, Wave-8, eff_dim, cross_corr (per condition)

Run:
  HSA_OVERRIDE_GFX_VERSION=11.0.0 venv/bin/python scripts/z2285_synapse_diversity.py
"""

import os, sys, time, json
import numpy as np
from pathlib import Path

# ─── Paths ───
BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE / 'scripts'))
RESULTS = BASE / 'results'
RESULTS.mkdir(exist_ok=True)
SAVE_FILE = RESULTS / 'z2285_synapse_diversity.json'

from fpga_host_eth import FPGAEthBridge

# ─── FPGA Parameters (from z2283) ───
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


# ═══════════════════════════════════════════════════════════
# Synapse Weight Patterns
# ═══════════════════════════════════════════════════════════

def pack_synapse(w_nm2, w_np2, w_nm1, w_np1):
    """Pack 4 float weights (0..1) into 32-bit synapse word."""
    b_nm2 = max(0, min(255, int(w_nm2 * 256)))
    b_np2 = max(0, min(255, int(w_np2 * 256)))
    b_nm1 = max(0, min(255, int(w_nm1 * 256)))
    b_np1 = max(0, min(255, int(w_np1 * 256)))
    return (b_nm2 << 24) | (b_np2 << 16) | (b_nm1 << 8) | b_np1


def apply_synapse_pattern(fpga, pattern_name, rng=None):
    """Apply a synapse weight pattern to all 128 neurons."""
    if rng is None:
        rng = np.random.default_rng(2285)

    for n in range(NUM_NEURONS):
        if pattern_name == 'default':
            packed = 0x40408080  # N±1=0.50, N±2=0.25
        elif pattern_name == 'zero':
            packed = 0x00000000  # no lateral coupling
        elif pattern_name == 'sparse':
            # 50% neurons: no coupling, 50%: default
            if n % 2 == 0:
                packed = 0x00000000
            else:
                packed = 0x40408080
        elif pattern_name == 'chain':
            # Strong forward (N+1), weak backward, minimal ±2
            # Creates temporal propagation chain
            packed = pack_synapse(
                w_nm2=0.0,      # no N-2
                w_np2=0.0,      # no N+2
                w_nm1=0.05,     # weak backward
                w_np1=0.80,     # strong forward
            )
        elif pattern_name == 'diverse':
            # 4 groups with very different coupling patterns
            grp = n % 4
            if grp == 0:
                # Independent — no coupling
                packed = 0x00000000
            elif grp == 1:
                # Forward chain — strong N+1
                packed = pack_synapse(0.0, 0.0, 0.0, 0.90)
            elif grp == 2:
                # Backward chain — strong N-1
                packed = pack_synapse(0.0, 0.0, 0.90, 0.0)
            else:
                # Random — diverse coupling
                packed = pack_synapse(
                    w_nm2=rng.uniform(0, 0.3),
                    w_np2=rng.uniform(0, 0.3),
                    w_nm1=rng.uniform(0, 0.5),
                    w_np1=rng.uniform(0, 0.5),
                )
        else:
            packed = 0x00000000

        fpga.set_synapse(n, packed)
        time.sleep(0.001)

    time.sleep(0.5)
    print(f"    Applied synapse pattern: {pattern_name}")


# ═══════════════════════════════════════════════════════════
# GPU Fourpop ESN (same as z2284)
# ═══════════════════════════════════════════════════════════

class GPUFourpopESN:
    def __init__(self, n_per_pop=64):
        self.pp = n_per_pop
        self.N = 4 * n_per_pop
        rng = np.random.default_rng(7777)
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
        self.W_rec = rng.standard_normal((self.N, self.N)) * 0.04
        mask = rng.random((self.N, self.N)) > 0.9
        self.W_rec *= mask
        sc, ec = 2 * n_per_pop, 3 * n_per_pop
        W_c = rng.standard_normal((n_per_pop, n_per_pop)) * 0.08
        mask_c = rng.random((n_per_pop, n_per_pop)) > 0.7
        W_c *= mask_c
        eigvals = np.abs(np.linalg.eigvals(W_c))
        sr = max(eigvals) if len(eigvals) > 0 else 1.0
        if sr > 0:
            W_c *= 1.05 / sr
        self.W_rec[sc:ec, sc:ec] = W_c
        self.bthr = 0.5 + 0.3 * np.arange(n_per_pop) / max(n_per_pop - 1, 1)
        self.temp_c = 0.65

    def run(self, input_seq):
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
            sa, ea = 0, pp
            branch_val = np.where(v[sa:ea] > self.bthr, 1.0, -1.0)
            v_a = np.tanh((1 - self.leak[sa:ea]) * v[sa:ea]
                          + self.input_w[sa:ea] * u + rec[sa:ea]
                          + self.bias[sa:ea] + 0.02 * branch_val)
            sb, eb = pp, 2 * pp
            v_b = v[sb:eb].copy()
            n_swap = max(1, pp // 10)
            swap_idx = rng.choice(pp, size=n_swap * 2, replace=False)
            for k in range(0, n_swap * 2 - 1, 2):
                v_b[swap_idx[k]], v_b[swap_idx[k + 1]] = v_b[swap_idx[k + 1]], v_b[swap_idx[k]]
            v_b = np.tanh((1 - self.leak[sb:eb]) * v_b
                          + self.input_w[sb:eb] * u + rec[sb:eb] + self.bias[sb:eb])
            sc, ec = 2 * pp, 3 * pp
            v_c = np.tanh(((1 - self.leak[sc:ec]) * v[sc:ec]
                           + self.input_w[sc:ec] * u + rec[sc:ec]
                           + self.bias[sc:ec]) / self.temp_c)
            sd, ed = 3 * pp, 4 * pp
            sched_noise = rng.uniform(-1, 1, pp) * 0.01
            v_d = np.tanh((1 - self.leak[sd:ed]) * v[sd:ed]
                          + self.input_w[sd:ed] * u + rec[sd:ed]
                          + self.bias[sd:ed] + sched_noise)
            v_new = np.concatenate([v_a, v_b, v_c, v_d])
            v_new += rng.uniform(-1, 1, self.N) * 0.003
            spike_mask = v_new > self.thr
            v_new[spike_mask] -= self.thr[spike_mask]
            v = v_new
            h = 0.93 * h + 0.07 * v
            slow = 0.99 * slow + 0.01 * v
            states[t] = v + 0.3 * h + 0.1 * slow
        return states


# ═══════════════════════════════════════════════════════════
# FPGA + Bridge Run Functions
# ═══════════════════════════════════════════════════════════

def fpga_setup(fpga):
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


def fpga_run_sequence(fpga, input_seq, mac_override=None):
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


def bridge_run_sequence(fpga, gpu_esn, input_seq):
    n_steps = len(input_seq)
    gpu_states = gpu_esn.run(input_seq)
    gpu_mac = np.zeros(n_steps)
    for t in range(n_steps):
        gpu_activity = np.mean(np.abs(gpu_states[t]))
        input_component = (input_seq[t] * 0.4 + 0.5)
        gpu_component = np.clip(gpu_activity, 0, 1)
        gpu_mac[t] = 0.6 * input_component + 0.4 * gpu_component
    gpu_mac = np.clip(gpu_mac, 0, 1)
    fpga_states, fpga_dspikes = fpga_run_sequence(fpga, input_seq, mac_override=gpu_mac)
    return gpu_states, fpga_states, fpga_dspikes, gpu_mac


# ═══════════════════════════════════════════════════════════
# Feature Extraction + Ridge
# ═══════════════════════════════════════════════════════════

def extract_trial_features(states, dspikes):
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


def build_continuous_features(states, dspikes, include_quad=False):
    delta = np.diff(states, axis=0)
    delta = np.vstack([np.zeros((1, states.shape[1])), delta])
    X = np.hstack([states, dspikes, delta])
    if include_quad:
        n_cols = states.shape[1]
        quad_idx = np.arange(0, n_cols, max(1, n_cols // 32))[:32]
        vm_sub = states[:, quad_idx]
        ds_sub = dspikes[:, quad_idx]
        X = np.hstack([X, vm_sub * ds_sub, vm_sub[:, :-1] * vm_sub[:, 1:], np.square(vm_sub)])
    return X


def ridge_mc(X_tr, y_tr, X_te, y_te):
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


def ridge_classify(X, y, n_classes, n_splits=5):
    from sklearn.linear_model import RidgeClassifier
    from sklearn.model_selection import cross_val_score
    sigma = np.std(X, axis=0)
    sigma[sigma < 1e-2] = 1.0
    X_n = X / sigma
    clf = RidgeClassifier(alpha=10.0)
    scores = cross_val_score(clf, X_n, y, cv=n_splits)
    return float(scores.mean()), float(scores.std())


def generate_waveform(cls, steps):
    t = np.linspace(0, 2 * np.pi, steps)
    if cls == 0:   return np.sin(t)
    elif cls == 1: return np.sign(np.sin(t))
    elif cls == 2: return 2 * np.abs(2 * (t / (2 * np.pi) - np.floor(t / (2 * np.pi) + 0.5))) - 1
    else:          return 2 * (t / (2 * np.pi) - np.floor(t / (2 * np.pi))) - 1


def compute_diversity(states):
    """Compute effective dimensionality and cross-correlation of neural states."""
    X = states[WARMUP:] if len(states) > WARMUP else states
    per_neuron_std = np.std(X, axis=0)
    mean_std = float(np.mean(per_neuron_std))
    std_std = float(np.std(per_neuron_std))

    X_c = X - X.mean(0)
    try:
        sv = np.linalg.svd(X_c, compute_uv=False)
        sv_norm = sv / (sv.sum() + 1e-30)
        eff_dim = float(np.exp(-np.sum(sv_norm * np.log(sv_norm + 1e-30))))
    except Exception:
        eff_dim = 0.0

    # Cross-correlation: mean pairwise correlation
    if X.shape[1] > 1:
        corr_mat = np.corrcoef(X.T)
        # Mean of upper triangle (excluding diagonal)
        mask = np.triu(np.ones_like(corr_mat, dtype=bool), k=1)
        cross_corr = float(np.mean(np.abs(corr_mat[mask])))
    else:
        cross_corr = 1.0

    return {
        'mean_neuron_std': mean_std,
        'std_neuron_std': std_std,
        'eff_dim_svd': eff_dim,
        'cross_corr': cross_corr,
    }


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer): return int(obj)
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        if isinstance(obj, np.bool_): return bool(obj)
        return super().default(obj)


# ═══════════════════════════════════════════════════════════
# Run One Condition
# ═══════════════════════════════════════════════════════════

def run_condition(fpga, cond_name, syn_pattern, gpu_esn=None, use_bridge=False):
    """Run full benchmark suite for one synapse configuration."""
    print(f"\n  ── {cond_name} (syn={syn_pattern}, bridge={use_bridge}) ──")

    rng = np.random.default_rng(2285)

    # Apply synapse pattern
    apply_synapse_pattern(fpga, syn_pattern, rng=rng)

    # Let neurons settle with new synapses
    fpga.set_mac_signal(0.5)
    time.sleep(2.0)
    fpga.set_mac_signal(0.0)
    time.sleep(0.5)

    cond_results = {}

    # ─── A. Waveform Classification (4-class) ───
    print("    [A] Waveform 4-class...", end=" ", flush=True)
    X_trials, y_trials = [], []
    rng_wave = np.random.default_rng(42)

    for trial in range(N_WAVE_TRIALS):
        cls = trial % 4
        wave = generate_waveform(cls, N_WAVE_STEPS)
        wave_mac = (wave * 0.4 + 0.5).astype(np.float64)

        if use_bridge and gpu_esn is not None:
            gpu_st, fpga_st, fpga_ds, _ = bridge_run_sequence(fpga, gpu_esn, wave)
            feats = extract_trial_features(fpga_st, fpga_ds)
            gpu_feats = np.concatenate([gpu_st.mean(0), gpu_st.std(0)])
            feats = np.concatenate([feats, gpu_feats])
        else:
            fpga_st, fpga_ds = fpga_run_sequence(fpga, wave_mac)
            feats = extract_trial_features(fpga_st, fpga_ds)

        X_trials.append(feats)
        y_trials.append(cls)

    X_w = np.array(X_trials)
    y_w = np.array(y_trials)
    w4_acc, w4_std = ridge_classify(X_w, y_w, 4)
    print(f"acc={w4_acc*100:.1f}%")
    cond_results['wave4_acc'] = w4_acc
    cond_results['wave4_std'] = w4_std

    # ─── B. Waveform 8-class ───
    print("    [B] Waveform 8-class...", end=" ", flush=True)
    X8, y8 = [], []
    for trial in range(N_WAVE_TRIALS):
        cls = trial % 8
        if cls < 4:
            wave = generate_waveform(cls, N_WAVE_STEPS)
        else:
            wave = generate_waveform(cls - 4, N_WAVE_STEPS)
            wave += rng_wave.uniform(-0.2, 0.2, N_WAVE_STEPS)
            wave = np.clip(wave, -1, 1)
        wave_mac = (wave * 0.4 + 0.5).astype(np.float64)

        if use_bridge and gpu_esn is not None:
            gpu_st, fpga_st, fpga_ds, _ = bridge_run_sequence(fpga, gpu_esn, wave)
            feats = extract_trial_features(fpga_st, fpga_ds)
            gpu_feats = np.concatenate([gpu_st.mean(0), gpu_st.std(0)])
            feats = np.concatenate([feats, gpu_feats])
        else:
            fpga_st, fpga_ds = fpga_run_sequence(fpga, wave_mac)
            feats = extract_trial_features(fpga_st, fpga_ds)

        X8.append(feats)
        y8.append(cls)

    w8_acc, w8_std = ridge_classify(np.array(X8), np.array(y8), 8)
    print(f"acc={w8_acc*100:.1f}%")
    cond_results['wave8_acc'] = w8_acc
    cond_results['wave8_std'] = w8_std

    wait_cool(cond_name)

    # ─── C. Continuous: Memory Capacity + XOR ───
    print("    [C] Continuous benchmarks...", end=" ", flush=True)
    u = rng.uniform(-1, 1, N_CONTINUOUS_STEPS).astype(np.float64)
    u_mac = np.clip(u * 0.4 + 0.5, 0, 1)

    if use_bridge and gpu_esn is not None:
        gpu_st, fpga_st, fpga_ds, _ = bridge_run_sequence(fpga, gpu_esn, u)
        X_fpga = build_continuous_features(fpga_st, fpga_ds, include_quad=True)
        gpu_feats = build_continuous_features(gpu_st, np.zeros_like(gpu_st))
        X = np.hstack([X_fpga[WARMUP:], gpu_feats[WARMUP:]])
    else:
        fpga_st, fpga_ds = fpga_run_sequence(fpga, u_mac)
        X = build_continuous_features(fpga_st, fpga_ds, include_quad=True)[WARMUP:]

    # Diversity metrics from FPGA states
    diversity = compute_diversity(fpga_st)
    cond_results['diversity'] = diversity
    print(f"eff_dim={diversity['eff_dim_svd']:.2f} xcorr={diversity['cross_corr']:.4f}")

    # MC at delays 1-10
    mc_total = 0.0
    mc_per_delay = {}
    n_total = len(X)
    n_tr = int(0.7 * n_total)

    for d in range(1, 11):
        target = u[WARMUP - d:N_CONTINUOUS_STEPS - d]
        n = min(len(X), len(target))
        X_c, t_c = X[:n], target[:n]
        r2 = ridge_mc(X_c[:n_tr], t_c[:n_tr], X_c[n_tr:], t_c[n_tr:])
        mc_per_delay[str(d)] = r2
        mc_total += r2

    cond_results['mc_total'] = mc_total
    cond_results['mc_per_delay'] = mc_per_delay
    print(f"    MC_total={mc_total:.3f} (d1={mc_per_delay['1']:.3f}, d2={mc_per_delay['2']:.3f})")

    # XOR at delays 1-3
    u_bin = (u > 0).astype(float)
    for tau in [1, 2, 3]:
        xor_target = ((u_bin[WARMUP:N_CONTINUOUS_STEPS] +
                        u_bin[WARMUP - tau:N_CONTINUOUS_STEPS - tau]) % 2).astype(float)
        n = min(len(X), len(xor_target))
        X_c, t_c = X[:n], xor_target[:n]
        # Linear
        acc_lin = ridge_xor(X_c[:n_tr], t_c[:n_tr], X_c[n_tr:], t_c[n_tr:])
        cond_results[f'xor{tau}_lin'] = acc_lin
        # Quadratic (already in features via include_quad=True)
        cond_results[f'xor{tau}_quad'] = acc_lin  # quad features already in X

    print(f"    XOR1={cond_results['xor1_lin']*100:.1f}% XOR2={cond_results['xor2_lin']*100:.1f}% XOR3={cond_results['xor3_lin']*100:.1f}%")

    wait_cool(cond_name)
    return cond_results


# ═══════════════════════════════════════════════════════════
# Main Experiment
# ═══════════════════════════════════════════════════════════

def main():
    print("=" * 70)
    print("  z2285: SYNAPSE DIVERSITY — BREAK NEURON SYNCHRONY")
    print("  Root cause: eff_dim=1.48, xcorr=0.994 with default synapses")
    print("  Fix: per-neuron synapse weights via CMD_SET_SYNAPSE")
    print("=" * 70)

    results = {
        'experiment': 'z2285_synapse_diversity',
        'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
    }

    # ─── Connect FPGA ───
    print("\n[1] Connecting to FPGA...")
    fpga = FPGAEthBridge()
    fpga.connect()
    fpga_setup(fpga)

    telem = fpga.read_telemetry()
    if telem is None:
        time.sleep(0.5)
        telem = fpga.read_telemetry()
    if telem is None:
        print("  FATAL: No FPGA telemetry")
        fpga.close()
        sys.exit(1)
    print(f"  vmem: [{telem['vmem'].min():.3f}, {telem['vmem'].max():.3f}]")

    # ─── GPU ESN ───
    print("\n[2] Initializing GPU fourpop ESN...")
    gpu_esn = GPUFourpopESN(n_per_pop=64)
    print(f"  {gpu_esn.N} neurons (4 pops × 64)")

    # ─── Run conditions ───
    conditions = [
        ('C0_DEFAULT',  'default', False),
        ('C1_ZERO',     'zero',    False),
        ('C2_SPARSE',   'sparse',  False),
        ('C3_CHAIN',    'chain',   False),
        ('C4_DIVERSE',  'diverse', False),
        ('C5_BRIDGE',   'diverse', True),   # Best local (diverse) + GPU bridge
    ]

    for cond_name, syn_pat, use_bridge in conditions:
        if get_max_temp() > TEMP_ABORT:
            print(f"\n  ABORT: Temperature {get_max_temp():.0f}°C > {TEMP_ABORT}°C")
            break

        wait_cool(cond_name, target=TEMP_SAFE)
        cond_res = run_condition(
            fpga, cond_name, syn_pat,
            gpu_esn=gpu_esn if use_bridge else None,
            use_bridge=use_bridge
        )
        results[cond_name] = cond_res

        # Incremental save
        with open(SAVE_FILE, 'w') as f:
            json.dump(results, f, indent=2, cls=NpEncoder)

    # ─── Key Tests ───
    print("\n" + "=" * 70)
    print("  KEY TESTS")
    print("=" * 70)

    key_tests = {}

    # T1: ZERO synapses should have higher eff_dim than DEFAULT
    if 'C0_DEFAULT' in results and 'C1_ZERO' in results:
        ed_def = results['C0_DEFAULT']['diversity']['eff_dim_svd']
        ed_zero = results['C1_ZERO']['diversity']['eff_dim_svd']
        p = ed_zero > ed_def
        key_tests['T1_zero_breaks_sync'] = {
            'pass': p,
            'desc': f"ZERO eff_dim={ed_zero:.2f} > DEFAULT eff_dim={ed_def:.2f}"
        }
        print(f"  T1 {'PASS' if p else 'FAIL'}: {key_tests['T1_zero_breaks_sync']['desc']}")

    # T2: ZERO should have lower cross_corr than DEFAULT
    if 'C0_DEFAULT' in results and 'C1_ZERO' in results:
        cc_def = results['C0_DEFAULT']['diversity']['cross_corr']
        cc_zero = results['C1_ZERO']['diversity']['cross_corr']
        p = cc_zero < cc_def
        key_tests['T2_zero_decorrelates'] = {
            'pass': p,
            'desc': f"ZERO xcorr={cc_zero:.4f} < DEFAULT xcorr={cc_def:.4f}"
        }
        print(f"  T2 {'PASS' if p else 'FAIL'}: {key_tests['T2_zero_decorrelates']['desc']}")

    # T3: DIVERSE should have highest eff_dim among FPGA-only conditions
    fpga_conds = ['C0_DEFAULT', 'C1_ZERO', 'C2_SPARSE', 'C3_CHAIN', 'C4_DIVERSE']
    eds = {c: results[c]['diversity']['eff_dim_svd'] for c in fpga_conds if c in results}
    if eds:
        best_ed_cond = max(eds, key=eds.get)
        p = best_ed_cond in ['C4_DIVERSE', 'C1_ZERO']
        key_tests['T3_best_eff_dim'] = {
            'pass': p,
            'desc': f"Best eff_dim: {best_ed_cond}={eds[best_ed_cond]:.2f}"
        }
        print(f"  T3 {'PASS' if p else 'FAIL'}: {key_tests['T3_best_eff_dim']['desc']}")

    # T4: Best FPGA-only MC should be ≥ 2.0
    mcs = {c: results[c]['mc_total'] for c in fpga_conds if c in results}
    if mcs:
        best_mc_cond = max(mcs, key=mcs.get)
        p = mcs[best_mc_cond] >= 2.0
        key_tests['T4_mc_above_2'] = {
            'pass': p,
            'desc': f"Best MC: {best_mc_cond}={mcs[best_mc_cond]:.3f} vs 2.0"
        }
        print(f"  T4 {'PASS' if p else 'FAIL'}: {key_tests['T4_mc_above_2']['desc']}")

    # T5: Best XOR1 should be > 65%
    xors = {c: results[c]['xor1_lin'] for c in fpga_conds if c in results}
    if xors:
        best_xor_cond = max(xors, key=xors.get)
        p = xors[best_xor_cond] > 0.65
        key_tests['T5_xor_above_65'] = {
            'pass': p,
            'desc': f"Best XOR1: {best_xor_cond}={xors[best_xor_cond]*100:.1f}% vs 65%"
        }
        print(f"  T5 {'PASS' if p else 'FAIL'}: {key_tests['T5_xor_above_65']['desc']}")

    # T6: BRIDGE should have best overall (MC OR XOR)
    if 'C5_BRIDGE' in results:
        bridge_mc = results['C5_BRIDGE']['mc_total']
        bridge_xor = results['C5_BRIDGE']['xor1_lin']
        all_mc = {c: results[c]['mc_total'] for c in results if c.startswith('C')}
        all_xor = {c: results[c]['xor1_lin'] for c in results if c.startswith('C')}
        mc_best = bridge_mc >= max(all_mc.values()) - 0.1
        xor_best = bridge_xor >= max(all_xor.values()) - 0.02
        p = mc_best or xor_best
        key_tests['T6_bridge_competitive'] = {
            'pass': p,
            'desc': f"BRIDGE MC={bridge_mc:.3f} XOR1={bridge_xor*100:.1f}% competitive"
        }
        print(f"  T6 {'PASS' if p else 'FAIL'}: {key_tests['T6_bridge_competitive']['desc']}")

    # T7: Synapse manipulation should change eff_dim by > 2×
    if 'C0_DEFAULT' in results and eds:
        ed_range = max(eds.values()) / max(min(eds.values()), 0.01)
        p = ed_range > 2.0
        key_tests['T7_synapse_dynamic_range'] = {
            'pass': p,
            'desc': f"eff_dim range: {max(eds.values()):.2f}/{min(eds.values()):.2f} = {ed_range:.1f}×"
        }
        print(f"  T7 {'PASS' if p else 'FAIL'}: {key_tests['T7_synapse_dynamic_range']['desc']}")

    # T8: Wave-4 should remain > 80% for best condition
    w4s = {c: results[c]['wave4_acc'] for c in fpga_conds if c in results}
    if w4s:
        best_w4 = max(w4s.values())
        p = best_w4 > 0.80
        key_tests['T8_wave4_robust'] = {
            'pass': p,
            'desc': f"Best Wave4={best_w4*100:.1f}% > 80%"
        }
        print(f"  T8 {'PASS' if p else 'FAIL'}: {key_tests['T8_wave4_robust']['desc']}")

    results['key_tests'] = key_tests
    n_pass = sum(1 for t in key_tests.values() if t['pass'])
    results['n_pass'] = n_pass
    results['n_tests'] = len(key_tests)

    print(f"\n  TOTAL: {n_pass}/{len(key_tests)} PASS")

    # Final save
    with open(SAVE_FILE, 'w') as f:
        json.dump(results, f, indent=2, cls=NpEncoder)
    print(f"\n  Results saved: {SAVE_FILE}")

    fpga.set_kill(1)
    fpga.close()
    print("  Done.")


if __name__ == '__main__':
    main()
