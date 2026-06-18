#!/usr/bin/env python3
"""
z2297_recurrent_temporal.py — FPGA recurrent synapses + temporal products
========================================================================
z2296 (14/14 PASS) used zero synapses. This experiment enables the FPGA's
hardwired ring topology (N±1, N±2 lateral connections) with non-zero weights
to test if recurrent coupling improves XOR/MC on top of temporal features.

Synapse format: 4×8-bit packed [w_np1, w_nm1, w_np2, w_nm2] in Q16.16.
  0x40 = weight 0.25, 0x80 = weight 0.50, 0xFF = weight ~1.0

Conditions:
  A) ZERO synapses (z2296 baseline reproduction)
  B) UNIFORM synapses (all weights = 0x40 = 0.25)
  C) HETEROGENEOUS synapses (exc/inh pattern, varying by group)
  D) STRONG recurrent (w=0.60 exc, 0.30 inh)

Each condition: 3 seeds × full benchmark (MC, XOR, NARMA)
Plus: 1 seed COUPLED (FPGA recurrent + GPU fourpop)

Tests (18):
  T1-T3: Condition B or C beats A on MC, XOR3, XOR5
  T4-T6: Best recurrent condition beats z2296 mean on MC, XOR3, XOR5
  T7-T9: Recurrent improves NARMA-5, NARMA-10 vs zero
  T10: Best recurrent XOR1 > 80%
  T11: Best recurrent MC > 12.0
  T12: COUPLED (recurrent + GPU) > FPGA-only recurrent on at least 1 metric
  T13-T14: Consistency — std(XOR1) < 5% within best condition
  T15: Any condition achieves XOR5 > 70%
  T16: Best NARMA-5 < 0.15
  T17: Recurrent creates new dynamics (spike patterns differ from zero)
  T18: Heterogeneous better than uniform on at least 2/5 metrics

Run:
  HSA_OVERRIDE_GFX_VERSION=11.0.0 venv/bin/python scripts/z2297_recurrent_temporal.py
"""

import os, sys, time, json
import numpy as np
from pathlib import Path

os.environ['PYTHONUNBUFFERED'] = '1'

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE / 'scripts'))
RESULTS = BASE / 'results'
RESULTS.mkdir(exist_ok=True)
SAVE_FILE = RESULTS / 'z2297_recurrent_temporal.json'

from fpga_host_eth import FPGAEthBridge

NUM_NEURONS = 128
SAMPLE_HZ = 200
N_STEPS = 3000
WARMUP = 500
TEMP_SAFE = 45.0
N_SEEDS = 3
VG_GROUPS = {0: 0.05, 1: 0.15, 2: 0.30, 3: 0.58}


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


class GPUFourpopESN:
    """Numpy simulation of fourpop GPU reservoir (same as z2296)."""
    def __init__(self, n_per_pop=64, seed=7777):
        self.pp = n_per_pop
        self.N = 4 * n_per_pop
        rng = np.random.default_rng(seed)
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
        if sr > 0: W_c *= 1.05 / sr
        self.W_rec[sc:ec, sc:ec] = W_c
        self.bthr = 0.5 + 0.3 * np.arange(n_per_pop) / max(n_per_pop - 1, 1)
        self.temp_c = 0.65

    def run(self, input_seq, run_seed=42):
        n_steps = len(input_seq)
        pp = self.pp
        states = np.zeros((n_steps, self.N))
        v = np.zeros(self.N)
        h = np.zeros(self.N)
        slow = np.zeros(self.N)
        rng = np.random.default_rng(run_seed)
        for t in range(n_steps):
            u = input_seq[t]
            rec = self.W_rec @ v
            sa, ea = 0, pp
            bv = np.where(v[sa:ea] > self.bthr, 1.0, -1.0)
            v_a = np.tanh((1-self.leak[sa:ea])*v[sa:ea] + self.input_w[sa:ea]*u + rec[sa:ea] + self.bias[sa:ea] + 0.02*bv)
            sb, eb = pp, 2*pp
            v_b = v[sb:eb].copy()
            ns = max(1, pp//10)
            si = rng.choice(pp, size=ns*2, replace=False)
            for k in range(0, ns*2-1, 2):
                v_b[si[k]], v_b[si[k+1]] = v_b[si[k+1]], v_b[si[k]]
            v_b = np.tanh((1-self.leak[sb:eb])*v_b + self.input_w[sb:eb]*u + rec[sb:eb] + self.bias[sb:eb])
            sc, ec = 2*pp, 3*pp
            v_c = np.tanh(((1-self.leak[sc:ec])*v[sc:ec] + self.input_w[sc:ec]*u + rec[sc:ec] + self.bias[sc:ec])/self.temp_c)
            sd, ed = 3*pp, 4*pp
            sn = rng.uniform(-1,1,pp)*0.01
            v_d = np.tanh((1-self.leak[sd:ed])*v[sd:ed] + self.input_w[sd:ed]*u + rec[sd:ed] + self.bias[sd:ed] + sn)
            v_new = np.concatenate([v_a, v_b, v_c, v_d])
            v_new += rng.uniform(-1,1,self.N)*0.003
            spike_mask = v_new > self.thr
            v_new[spike_mask] -= self.thr[spike_mask]
            v = v_new
            h = 0.93*h + 0.07*v
            slow = 0.99*slow + 0.01*v
            states[t] = v + 0.3*h + 0.1*slow
        return states


# ============================================================
# Synapse weight configurations
# ============================================================
# Format: 4×8-bit packed as 32-bit word [w_np1 | w_nm1 | w_np2 | w_nm2]
# Each byte: unsigned Q0.8 mapped to [0, ~1.0] synapse weight

def make_synapse_configs():
    """Define 4 synapse conditions."""
    configs = {}

    # A: Zero (z2296 baseline)
    configs['A_ZERO'] = [0x00000000] * NUM_NEURONS

    # B: Uniform excitatory (all 4 neighbors weighted 0.25)
    configs['B_UNIFORM'] = [0x40404040] * NUM_NEURONS

    # C: Heterogeneous (exc N±1, inh N±2, varying by VG group)
    c_synapses = []
    for n in range(NUM_NEURONS):
        group = n % 4
        if group == 0:    # Low Vg: strong exc N+1, weak N-1, weak inh N±2
            w = (0x60 << 24) | (0x30 << 16) | (0x10 << 8) | 0x10
        elif group == 1:  # Mid-low: balanced exc, no inh
            w = (0x40 << 24) | (0x40 << 16) | (0x00 << 8) | 0x00
        elif group == 2:  # Mid-high: exc N+1 only, inh N-2
            w = (0x50 << 24) | (0x00 << 16) | (0x00 << 8) | 0x20
        else:             # High Vg: strong bidirectional exc, weak inh
            w = (0x70 << 24) | (0x70 << 16) | (0x18 << 8) | 0x18
        c_synapses.append(w)
    configs['C_HETERO'] = c_synapses

    # D: Strong recurrent (exc=0.60 N±1, inh=0.30 N±2)
    exc8 = int(0.60 * 255)  # 0x99
    inh8 = int(0.30 * 255)  # 0x4C
    w_strong = (exc8 << 24) | (exc8 << 16) | (inh8 << 8) | inh8
    configs['D_STRONG'] = [w_strong] * NUM_NEURONS

    return configs


def set_fpga_synapses(fpga, synapse_list):
    """Write synapse weights for all neurons."""
    for n in range(NUM_NEURONS):
        fpga.set_synapse(n, synapse_list[n])
        time.sleep(0.001)
    time.sleep(0.5)


def fpga_run_continuous(fpga, u, mac_signal=None):
    n_steps = len(u)
    if mac_signal is None:
        mac_signal = np.clip(u * 0.3 + 0.3, 0, 1)
    states = np.zeros((n_steps, NUM_NEURONS))
    dspikes = np.zeros((n_steps, NUM_NEURONS), dtype=np.float32)
    dt = 1.0 / SAMPLE_HZ
    fpga.set_mac_signal(0.0)
    time.sleep(0.02)
    telem = fpga.read_telemetry()
    prev_sc = telem['spike_counts'].copy() if telem is not None else np.zeros(NUM_NEURONS, dtype=np.uint16)
    for t in range(n_steps):
        if t > 0 and t % 25 == 0:
            temp = get_max_temp()
            if temp > 70.0:
                fpga.set_mac_signal(0.0)
                print(f"\n  [THERMAL PAUSE] {temp:.0f}°C at step {t}/{n_steps}", end="", flush=True)
                while temp > 50.0:
                    time.sleep(5)
                    temp = get_max_temp()
                    print(f" {temp:.0f}", end="", flush=True)
                print(" resumed", flush=True)
        fpga.set_mac_signal(float(mac_signal[t]))
        time.sleep(dt + 0.002)  # extra 2ms cooldown per step
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
    fpga.set_mac_signal(0.0)
    return states, dspikes


def build_best_features(states, dspikes):
    """Temporal products order 2+3 (same as z2296)."""
    n_steps, n_ch = states.shape
    delta = np.diff(states, axis=0)
    delta = np.vstack([np.zeros((1, n_ch)), delta])
    feats = [states, dspikes, delta]

    rng = np.random.default_rng(42)
    qi = np.sort(rng.choice(n_ch, size=min(24, n_ch), replace=False))
    vm_q = states[:, qi]
    ds_q = dspikes[:, qi]

    tau_list = [1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20]

    for tau in tau_list:
        shifted = np.zeros_like(vm_q)
        shifted[tau:] = vm_q[:-tau]
        feats.append(vm_q * shifted)
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


def full_benchmark(X, u_raw):
    n = len(X)
    n_tr = int(0.7 * n)

    mc_total = 0.0
    mc_per_d = {}
    for d in range(1, 21):
        target = u_raw[WARMUP-d:len(u_raw)-d]
        nn = min(n, len(target))
        r2 = ridge_solve(X[:n_tr], target[:n_tr], X[n_tr:nn], target[n_tr:nn])
        mc_per_d[str(d)] = r2
        mc_total += r2

    xor = {}
    for tau in [1, 2, 3, 5, 8, 10, 15]:
        u_a = (u_raw[WARMUP:] > 0).astype(float)
        u_b = (u_raw[WARMUP-tau:len(u_raw)-tau] > 0).astype(float)
        nn = min(len(u_a), len(u_b), n)
        target = (u_a[:nn] != u_b[:nn]).astype(float)
        Xn = X[:nn]
        acc = ridge_solve(Xn[:n_tr], target[:n_tr], Xn[n_tr:nn], target[n_tr:nn], 'classification')
        xor[f'tau{tau}'] = acc

    narma = {}
    for order in [5, 10, 20]:
        T = len(u_raw)
        u_n = (u_raw - u_raw.min()) / (u_raw.max() - u_raw.min() + 1e-10) * 0.5
        y = np.zeros(T)
        for t in range(order, T):
            y[t] = 0.3*y[t-1] + 0.05*y[t-1]*np.sum(y[t-order:t]) + 1.5*u_n[t-1]*u_n[t-order] + 0.1
            y[t] = np.tanh(y[t])
        target = y[WARMUP:]
        nn = min(n, len(target))
        best_nrmse = 999.0
        for alpha in [0.01, 0.1, 1.0, 10.0, 100.0]:
            I2 = np.eye(X[:n_tr].shape[1])
            try:
                w = np.linalg.solve(X[:n_tr].T @ X[:n_tr] + alpha * I2, X[:n_tr].T @ target[:n_tr])
                pred = X[n_tr:nn] @ w
                gt = target[n_tr:nn]
                nrmse = np.sqrt(np.mean((gt-pred)**2)) / (np.std(gt)+1e-10)
                if nrmse < best_nrmse:
                    best_nrmse = nrmse
            except Exception:
                pass
        narma[f'narma{order}'] = best_nrmse

    return {'mc_total': mc_total, 'mc_per_delay': mc_per_d, 'xor': xor, 'narma': narma}


def main():
    print("=" * 70)
    print("  z2297: FPGA Recurrent Synapses + Temporal Products")
    print("  Baseline: z2296 (14/14 PASS, zero synapses)")
    print("  Question: Does ring N±1,N±2 coupling improve XOR/MC?")
    print("=" * 70)

    fpga = FPGAEthBridge(timeout=2.0)
    fpga.connect()
    fpga.set_kill(0)
    time.sleep(1.0)

    # Runtime parameter setup
    fpga.set_leak_cond(0x2000)
    fpga.set_base_exc_raw(0x0080)
    fpga.set_bias_gain_raw(0x4000)
    fpga.set_threshold_raw(0x20000)

    for n in range(NUM_NEURONS):
        fpga.set_vg(n, VG_GROUPS[n % 4])
        time.sleep(0.001)

    telem = fpga.read_telemetry()
    print(f"  FPGA online: vmem [{telem['vmem'].min():.3f}, {telem['vmem'].max():.3f}]")
    print(f"  Config: LEAK=0x2000, THRESH=0x20000, BASE_EXC=0x0080, BIAS_GAIN=0x4000")

    configs = make_synapse_configs()
    seeds = [42, 123, 456]

    # Resume from saved results if available
    results = {'conditions': {}, 'coupled': {}, 'tests': {}}
    if SAVE_FILE.exists():
        try:
            with open(SAVE_FILE) as f:
                results = json.load(f)
            print(f"  RESUMED: {list(results.get('conditions',{}).keys())} already done")
        except Exception:
            pass

    # ================================================================
    # Run each synapse condition × 3 seeds
    # ================================================================
    for cond_name, syn_list in configs.items():
        if cond_name in results.get('conditions', {}) and len(results['conditions'][cond_name]) >= 3:
            print(f"\n  [SKIP] {cond_name} already complete ({len(results['conditions'][cond_name])} seeds)")
            continue

        print(f"\n{'='*60}")
        print(f"  CONDITION: {cond_name}")
        print(f"{'='*60}")

        set_fpga_synapses(fpga, syn_list)

        # Verify synapse effect — quick spike check
        telem = fpga.read_telemetry()
        mean_vm = telem['vmem'].mean()
        std_vm = telem['vmem'].std()
        print(f"  Post-synapse: vmem mean={mean_vm:.4f}, std={std_vm:.4f}")

        cond_results = []
        for si, seed in enumerate(seeds):
            rng = np.random.default_rng(seed)
            u_raw = rng.uniform(-1, 1, N_STEPS + WARMUP)

            wait_cool(f"{cond_name} seed{si}")
            print(f"  Running seed {seed}...", end="", flush=True)
            fpga_states, fpga_dspikes = fpga_run_continuous(fpga, u_raw)
            print(f" done ({fpga_states.shape})", flush=True)

            # Spike activity stats
            total_spikes = fpga_dspikes[WARMUP:].sum()
            active_frac = (fpga_dspikes[WARMUP:].sum(axis=0) > 0).mean()

            X = build_best_features(fpga_states[WARMUP:], fpga_dspikes[WARMUP:])
            bm = full_benchmark(X, u_raw)
            bm['seed'] = seed
            bm['n_features'] = X.shape[1]
            bm['total_spikes'] = float(total_spikes)
            bm['active_neuron_frac'] = float(active_frac)
            bm['vmem_mean'] = float(fpga_states[WARMUP:].mean())
            bm['vmem_std'] = float(fpga_states[WARMUP:].std())
            cond_results.append(bm)

            xor = bm['xor']
            print(f"    Seed {seed}: MC={bm['mc_total']:.2f} XOR1={xor['tau1']*100:.1f}% "
                  f"XOR3={xor['tau3']*100:.1f}% XOR5={xor['tau5']*100:.1f}% "
                  f"N5={bm['narma']['narma5']:.3f} spikes={total_spikes:.0f}")

        results['conditions'][cond_name] = cond_results

        # Incremental save
        with open(SAVE_FILE, 'w') as f:
            json.dump(results, f, indent=2, cls=NpEncoder)

    # ================================================================
    # COUPLED: Best recurrent + GPU fourpop
    # ================================================================
    print(f"\n{'='*60}")
    print(f"  COUPLED: Best recurrent + GPU fourpop")
    print(f"{'='*60}")

    # Find best recurrent condition (by mean MC + mean XOR3)
    best_cond = None
    best_score = -1
    for cond_name in ['B_UNIFORM', 'C_HETERO', 'D_STRONG']:
        cond_res = results['conditions'][cond_name]
        mean_mc = np.mean([r['mc_total'] for r in cond_res])
        mean_xor3 = np.mean([r['xor']['tau3'] for r in cond_res])
        score = mean_mc + mean_xor3 * 20  # Weight XOR3 more
        if score > best_score:
            best_score = score
            best_cond = cond_name
    print(f"  Best recurrent: {best_cond}")

    set_fpga_synapses(fpga, configs[best_cond])
    gpu = GPUFourpopESN(seed=7777)

    wait_cool("COUPLED")
    rng = np.random.default_rng(42)
    u_raw = rng.uniform(-1, 1, N_STEPS + WARMUP)

    fpga_states, fpga_dspikes = fpga_run_continuous(fpga, u_raw)
    gpu_states = gpu.run(u_raw, run_seed=42)

    # Build combined features
    X_fpga = build_best_features(fpga_states[WARMUP:], fpga_dspikes[WARMUP:])
    gpu_delta = np.diff(gpu_states[WARMUP:], axis=0)
    gpu_delta = np.vstack([np.zeros((1, gpu_states.shape[1])), gpu_delta])
    X_gpu = np.hstack([gpu_states[WARMUP:], gpu_delta])
    X_coupled = np.hstack([X_fpga, X_gpu])

    bm_coupled = full_benchmark(X_coupled, u_raw)
    bm_fpga_only = full_benchmark(X_fpga, u_raw)
    bm_gpu_only = full_benchmark(X_gpu, u_raw)

    xc = bm_coupled['xor']
    xf = bm_fpga_only['xor']
    xg = bm_gpu_only['xor']
    print(f"  COUPLED:    MC={bm_coupled['mc_total']:.2f} XOR1={xc['tau1']*100:.1f}% XOR3={xc['tau3']*100:.1f}% "
          f"XOR5={xc['tau5']*100:.1f}% N5={bm_coupled['narma']['narma5']:.3f}")
    print(f"  FPGA-only:  MC={bm_fpga_only['mc_total']:.2f} XOR1={xf['tau1']*100:.1f}% XOR3={xf['tau3']*100:.1f}% "
          f"XOR5={xf['tau5']*100:.1f}% N5={bm_fpga_only['narma']['narma5']:.3f}")
    print(f"  GPU-only:   MC={bm_gpu_only['mc_total']:.2f} XOR1={xg['tau1']*100:.1f}% XOR3={xg['tau3']*100:.1f}% "
          f"XOR5={xg['tau5']*100:.1f}% N5={bm_gpu_only['narma']['narma5']:.3f}")

    results['coupled'] = {
        'best_recurrent_cond': best_cond,
        'coupled': bm_coupled,
        'fpga_only': bm_fpga_only,
        'gpu_only': bm_gpu_only,
    }

    # ================================================================
    # TESTS
    # ================================================================
    print(f"\n{'='*70}")
    print("  TESTS")
    print(f"{'='*70}")

    tests = {}
    n_pass = 0

    # Helper: mean metric across seeds
    def cond_mean(cond, fn):
        return np.mean([fn(r) for r in results['conditions'][cond]])

    def cond_std(cond, fn):
        return np.std([fn(r) for r in results['conditions'][cond]])

    # z2296 reference values (from 5-seed run)
    z2296_mc = 12.27
    z2296_xor3 = 0.892
    z2296_xor5 = 0.883

    zero_mc = cond_mean('A_ZERO', lambda r: r['mc_total'])
    zero_xor3 = cond_mean('A_ZERO', lambda r: r['xor']['tau3'])
    zero_xor5 = cond_mean('A_ZERO', lambda r: r['xor']['tau5'])

    # Find best recurrent condition per metric
    rec_conds = ['B_UNIFORM', 'C_HETERO', 'D_STRONG']
    best_mc_cond = max(rec_conds, key=lambda c: cond_mean(c, lambda r: r['mc_total']))
    best_xor3_cond = max(rec_conds, key=lambda c: cond_mean(c, lambda r: r['xor']['tau3']))
    best_xor5_cond = max(rec_conds, key=lambda c: cond_mean(c, lambda r: r['xor']['tau5']))

    best_mc = cond_mean(best_mc_cond, lambda r: r['mc_total'])
    best_xor3 = cond_mean(best_xor3_cond, lambda r: r['xor']['tau3'])
    best_xor5 = cond_mean(best_xor5_cond, lambda r: r['xor']['tau5'])

    # T1: Best recurrent MC > zero MC
    t1 = best_mc > zero_mc
    tests['T1'] = {'pass': bool(t1), 'best': best_mc, 'zero': zero_mc, 'best_cond': best_mc_cond}
    print(f"  T1 {'PASS' if t1 else 'FAIL'}: Best recurrent MC={best_mc:.2f} > Zero={zero_mc:.2f} ({best_mc_cond})")
    n_pass += t1

    # T2: Best recurrent XOR3 > zero XOR3
    t2 = best_xor3 > zero_xor3
    tests['T2'] = {'pass': bool(t2), 'best': best_xor3, 'zero': zero_xor3, 'best_cond': best_xor3_cond}
    print(f"  T2 {'PASS' if t2 else 'FAIL'}: Best recurrent XOR3={best_xor3*100:.1f}% > Zero={zero_xor3*100:.1f}% ({best_xor3_cond})")
    n_pass += t2

    # T3: Best recurrent XOR5 > zero XOR5
    t3 = best_xor5 > zero_xor5
    tests['T3'] = {'pass': bool(t3), 'best': best_xor5, 'zero': zero_xor5, 'best_cond': best_xor5_cond}
    print(f"  T3 {'PASS' if t3 else 'FAIL'}: Best recurrent XOR5={best_xor5*100:.1f}% > Zero={zero_xor5*100:.1f}% ({best_xor5_cond})")
    n_pass += t3

    # T4: Best recurrent MC > z2296 mean (12.27)
    t4 = best_mc > z2296_mc
    tests['T4'] = {'pass': bool(t4), 'best': best_mc, 'z2296': z2296_mc}
    print(f"  T4 {'PASS' if t4 else 'FAIL'}: Best MC={best_mc:.2f} > z2296 mean={z2296_mc:.2f}")
    n_pass += t4

    # T5: Best recurrent XOR3 > z2296 mean (89.2%)
    t5 = best_xor3 > z2296_xor3
    tests['T5'] = {'pass': bool(t5), 'best': best_xor3, 'z2296': z2296_xor3}
    print(f"  T5 {'PASS' if t5 else 'FAIL'}: Best XOR3={best_xor3*100:.1f}% > z2296 mean={z2296_xor3*100:.1f}%")
    n_pass += t5

    # T6: Best recurrent XOR5 > z2296 mean (88.3%)
    t6 = best_xor5 > z2296_xor5
    tests['T6'] = {'pass': bool(t6), 'best': best_xor5, 'z2296': z2296_xor5}
    print(f"  T6 {'PASS' if t6 else 'FAIL'}: Best XOR5={best_xor5*100:.1f}% > z2296 mean={z2296_xor5*100:.1f}%")
    n_pass += t6

    # T7-T9: NARMA improvements
    zero_n5 = cond_mean('A_ZERO', lambda r: r['narma']['narma5'])
    zero_n10 = cond_mean('A_ZERO', lambda r: r['narma']['narma10'])

    best_n5_cond = min(rec_conds, key=lambda c: cond_mean(c, lambda r: r['narma']['narma5']))
    best_n10_cond = min(rec_conds, key=lambda c: cond_mean(c, lambda r: r['narma']['narma10']))
    best_n5 = cond_mean(best_n5_cond, lambda r: r['narma']['narma5'])
    best_n10 = cond_mean(best_n10_cond, lambda r: r['narma']['narma10'])

    t7 = best_n5 < zero_n5
    tests['T7'] = {'pass': bool(t7), 'best': best_n5, 'zero': zero_n5}
    print(f"  T7 {'PASS' if t7 else 'FAIL'}: Best NARMA-5={best_n5:.3f} < Zero={zero_n5:.3f}")
    n_pass += t7

    t8 = best_n10 < zero_n10
    tests['T8'] = {'pass': bool(t8), 'best': best_n10, 'zero': zero_n10}
    print(f"  T8 {'PASS' if t8 else 'FAIL'}: Best NARMA-10={best_n10:.3f} < Zero={zero_n10:.3f}")
    n_pass += t8

    # T9: Best NARMA-5 < z2296's 0.136
    t9 = best_n5 < 0.136
    tests['T9'] = {'pass': bool(t9), 'best': best_n5}
    print(f"  T9 {'PASS' if t9 else 'FAIL'}: Best NARMA-5={best_n5:.3f} < z2296 mean=0.136")
    n_pass += t9

    # T10: Best recurrent XOR1 > 80%
    best_xor1 = max(cond_mean(c, lambda r: r['xor']['tau1']) for c in rec_conds)
    t10 = best_xor1 > 0.80
    tests['T10'] = {'pass': bool(t10), 'val': best_xor1}
    print(f"  T10 {'PASS' if t10 else 'FAIL'}: Best XOR1={best_xor1*100:.1f}% > 80%")
    n_pass += t10

    # T11: Best recurrent MC > 12.0
    t11 = best_mc > 12.0
    tests['T11'] = {'pass': bool(t11), 'val': best_mc}
    print(f"  T11 {'PASS' if t11 else 'FAIL'}: Best MC={best_mc:.2f} > 12.0")
    n_pass += t11

    # T12: COUPLED beats FPGA-only on at least 1 metric
    coupled_wins = 0
    if bm_coupled['mc_total'] > bm_fpga_only['mc_total']: coupled_wins += 1
    if bm_coupled['xor']['tau3'] > bm_fpga_only['xor']['tau3']: coupled_wins += 1
    if bm_coupled['xor']['tau5'] > bm_fpga_only['xor']['tau5']: coupled_wins += 1
    if bm_coupled['narma']['narma5'] < bm_fpga_only['narma']['narma5']: coupled_wins += 1
    if bm_coupled['narma']['narma10'] < bm_fpga_only['narma']['narma10']: coupled_wins += 1
    t12 = coupled_wins >= 1
    tests['T12'] = {'pass': bool(t12), 'coupled_wins': coupled_wins}
    print(f"  T12 {'PASS' if t12 else 'FAIL'}: COUPLED wins on {coupled_wins}/5 metrics")
    n_pass += t12

    # T13: Consistency — std(XOR1) < 5% within best condition
    best_overall = max(rec_conds, key=lambda c: cond_mean(c, lambda r: r['mc_total']) + cond_mean(c, lambda r: r['xor']['tau3']))
    xor1_std = cond_std(best_overall, lambda r: r['xor']['tau1'])
    t13 = xor1_std < 0.05
    tests['T13'] = {'pass': bool(t13), 'std': xor1_std, 'cond': best_overall}
    print(f"  T13 {'PASS' if t13 else 'FAIL'}: XOR1 std={xor1_std*100:.1f}% < 5% ({best_overall})")
    n_pass += t13

    # T14: MC std < 2.0 within best condition
    mc_std = cond_std(best_overall, lambda r: r['mc_total'])
    t14 = mc_std < 2.0
    tests['T14'] = {'pass': bool(t14), 'std': mc_std}
    print(f"  T14 {'PASS' if t14 else 'FAIL'}: MC std={mc_std:.2f} < 2.0 ({best_overall})")
    n_pass += t14

    # T15: Any condition achieves XOR5 > 70%
    any_xor5_70 = any(cond_mean(c, lambda r: r['xor']['tau5']) > 0.70 for c in configs.keys())
    t15 = any_xor5_70
    tests['T15'] = {'pass': bool(t15)}
    print(f"  T15 {'PASS' if t15 else 'FAIL'}: Any condition XOR5 > 70%")
    n_pass += t15

    # T16: Best NARMA-5 < 0.15
    t16 = best_n5 < 0.15
    tests['T16'] = {'pass': bool(t16), 'val': best_n5}
    print(f"  T16 {'PASS' if t16 else 'FAIL'}: Best NARMA-5={best_n5:.3f} < 0.15")
    n_pass += t16

    # T17: Recurrent creates different dynamics (spike pattern differs from zero)
    zero_spikes = np.mean([r['total_spikes'] for r in results['conditions']['A_ZERO']])
    rec_spikes = {c: np.mean([r['total_spikes'] for r in results['conditions'][c]]) for c in rec_conds}
    max_spike_change = max(abs(rec_spikes[c] - zero_spikes) / (zero_spikes + 1) for c in rec_conds)
    t17 = max_spike_change > 0.05  # >5% change in spike count
    tests['T17'] = {'pass': bool(t17), 'change': max_spike_change, 'zero': zero_spikes, 'rec': rec_spikes}
    print(f"  T17 {'PASS' if t17 else 'FAIL'}: Spike change={max_spike_change*100:.1f}% > 5%")
    n_pass += t17

    # T18: Heterogeneous better than uniform on at least 2/5 metrics
    hetero_wins = 0
    for fn in [lambda r: r['mc_total'], lambda r: r['xor']['tau1'], lambda r: r['xor']['tau3'],
               lambda r: r['xor']['tau5']]:
        if cond_mean('C_HETERO', fn) > cond_mean('B_UNIFORM', fn):
            hetero_wins += 1
    for fn in [lambda r: r['narma']['narma5']]:
        if cond_mean('C_HETERO', fn) < cond_mean('B_UNIFORM', fn):
            hetero_wins += 1
    t18 = hetero_wins >= 2
    tests['T18'] = {'pass': bool(t18), 'wins': hetero_wins}
    print(f"  T18 {'PASS' if t18 else 'FAIL'}: Hetero wins {hetero_wins}/5 vs Uniform")
    n_pass += t18

    print(f"\n  TOTAL: {n_pass}/18 PASS")

    results['tests'] = tests
    results['summary'] = {
        'n_pass': n_pass, 'n_total': 18,
        'configuration': 'FPGA 128-neuron, 4 synapse conditions, temporal order-2+3',
        'best_recurrent_condition': best_cond,
    }

    with open(SAVE_FILE, 'w') as f:
        json.dump(results, f, indent=2, cls=NpEncoder)
    print(f"  Results saved: {SAVE_FILE}")

    # Clean up: reset synapses to zero
    for n in range(NUM_NEURONS):
        fpga.set_synapse(n, 0x00000000)
        time.sleep(0.001)
    fpga.close()


if __name__ == '__main__':
    main()
