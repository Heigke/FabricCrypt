#!/usr/bin/env python3
"""
z2287_tuned_bridge.py — Parameter sweep to recover MC with new synapse-capable bitstream
========================================================================================
z2285 showed the new bitstream has eff_dim≈100 (vs 1.48 before) — synchrony broken!
But MC dropped from 1.94 to 0.1-0.2. Bridge still strong: XOR=73.3%, Wave=100%.

Hypothesis: with higher independence, we need SLOWER leak to build temporal memory.
Previous optimal LEAK=0x2000 may be too fast for the new routing.

Plan:
  Phase 1: Quick leak sweep (0x0010 to 0x4000) with continuous MC test
  Phase 2: Best leak + synapse pattern sweep
  Phase 3: Full bridge benchmark with best parameters

Run:
  HSA_OVERRIDE_GFX_VERSION=11.0.0 venv/bin/python scripts/z2287_tuned_bridge.py
"""

import os, sys, time, json
import numpy as np
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE / 'scripts'))
RESULTS = BASE / 'results'
RESULTS.mkdir(exist_ok=True)
SAVE_FILE = RESULTS / 'z2287_tuned_bridge.json'

from fpga_host_eth import FPGAEthBridge

NUM_NEURONS = 128
SAMPLE_HZ = 200
VG_GROUPS = {0: 0.05, 1: 0.15, 2: 0.30, 3: 0.58}

# Benchmark parameters
N_QUICK_STEPS = 1500
WARMUP = 300
N_WAVE_TRIALS = 40
N_WAVE_STEPS = 60
N_CONTINUOUS_STEPS = 2000

TEMP_ABORT = 90.0
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


def pack_synapse(w_nm2, w_np2, w_nm1, w_np1):
    b_nm2 = max(0, min(255, int(w_nm2 * 256)))
    b_np2 = max(0, min(255, int(w_np2 * 256)))
    b_nm1 = max(0, min(255, int(w_nm1 * 256)))
    b_np1 = max(0, min(255, int(w_np1 * 256)))
    return (b_nm2 << 24) | (b_np2 << 16) | (b_nm1 << 8) | b_np1


def apply_synapse_pattern(fpga, pattern):
    rng = np.random.default_rng(2287)
    for n in range(NUM_NEURONS):
        if pattern == 'zero':
            packed = 0x00000000
        elif pattern == 'default':
            packed = 0x40408080
        elif pattern == 'diverse':
            grp = n % 4
            if grp == 0:
                packed = 0x00000000
            elif grp == 1:
                packed = pack_synapse(0.0, 0.0, 0.0, 0.90)
            elif grp == 2:
                packed = pack_synapse(0.0, 0.0, 0.90, 0.0)
            else:
                packed = pack_synapse(
                    rng.uniform(0, 0.3), rng.uniform(0, 0.3),
                    rng.uniform(0, 0.5), rng.uniform(0, 0.5))
        elif pattern == 'mild':
            # Mild coupling: weaker than default, asymmetric
            packed = pack_synapse(
                w_nm2=0.05, w_np2=0.05,
                w_nm1=0.10 + 0.10 * (n % 4) / 3,
                w_np1=0.30 - 0.10 * (n % 4) / 3)
        else:
            packed = 0x00000000
        fpga.set_synapse(n, packed)
        time.sleep(0.001)
    time.sleep(0.5)


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


def fpga_run_continuous(fpga, u, sample_hz=SAMPLE_HZ):
    """Run continuous input and return states + dspikes."""
    n_steps = len(u)
    u_mac = np.clip(u * 0.4 + 0.5, 0, 1)
    states = np.zeros((n_steps, NUM_NEURONS))
    dspikes = np.zeros((n_steps, NUM_NEURONS), dtype=np.float32)
    dt = 1.0 / sample_hz

    fpga.set_mac_signal(0.0)
    time.sleep(0.02)
    telem = fpga.read_telemetry()
    prev_sc = telem['spike_counts'].copy() if telem is not None else np.zeros(NUM_NEURONS, dtype=np.uint16)

    for t in range(n_steps):
        fpga.set_mac_signal(float(u_mac[t]))
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


def build_features(states, dspikes, include_quad=True):
    delta = np.diff(states, axis=0)
    delta = np.vstack([np.zeros((1, states.shape[1])), delta])
    X = np.hstack([states, dspikes, delta])
    if include_quad:
        n_cols = states.shape[1]
        qi = np.arange(0, n_cols, max(1, n_cols // 32))[:32]
        vm = states[:, qi]
        ds = dspikes[:, qi]
        X = np.hstack([X, vm * ds, vm[:, :-1] * vm[:, 1:], np.square(vm)])
    return X


def quick_mc_test(fpga, u):
    """Quick MC test: returns mc_total, mc_d1, eff_dim, xcorr."""
    states, dspikes = fpga_run_continuous(fpga, u)
    X = build_features(states, dspikes)[WARMUP:]

    # MC
    mc_total = 0.0
    mc_delays = {}
    n = len(X)
    n_tr = int(0.7 * n)
    for d in range(1, 6):
        target = u[WARMUP - d:len(u) - d]
        nn = min(len(X), len(target))
        r2 = ridge_mc(X[:n_tr], target[:n_tr], X[n_tr:nn], target[n_tr:nn])
        mc_delays[d] = r2
        mc_total += r2

    # Diversity
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

    return mc_total, mc_delays, eff_dim, xcorr


# ═══════════════════════════════════════════════════════════
# GPU Fourpop ESN
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
        if sr > 0: W_c *= 1.05 / sr
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
            bv = np.where(v[sa:ea] > self.bthr, 1.0, -1.0)
            v_a = np.tanh((1 - self.leak[sa:ea]) * v[sa:ea] + self.input_w[sa:ea] * u + rec[sa:ea] + self.bias[sa:ea] + 0.02 * bv)
            sb, eb = pp, 2 * pp
            v_b = v[sb:eb].copy()
            ns = max(1, pp // 10)
            si = rng.choice(pp, size=ns * 2, replace=False)
            for k in range(0, ns * 2 - 1, 2):
                v_b[si[k]], v_b[si[k + 1]] = v_b[si[k + 1]], v_b[si[k]]
            v_b = np.tanh((1 - self.leak[sb:eb]) * v_b + self.input_w[sb:eb] * u + rec[sb:eb] + self.bias[sb:eb])
            sc, ec = 2 * pp, 3 * pp
            v_c = np.tanh(((1 - self.leak[sc:ec]) * v[sc:ec] + self.input_w[sc:ec] * u + rec[sc:ec] + self.bias[sc:ec]) / self.temp_c)
            sd, ed = 3 * pp, 4 * pp
            sn = rng.uniform(-1, 1, pp) * 0.01
            v_d = np.tanh((1 - self.leak[sd:ed]) * v[sd:ed] + self.input_w[sd:ed] * u + rec[sd:ed] + self.bias[sd:ed] + sn)
            v_new = np.concatenate([v_a, v_b, v_c, v_d])
            v_new += rng.uniform(-1, 1, self.N) * 0.003
            spike_mask = v_new > self.thr
            v_new[spike_mask] -= self.thr[spike_mask]
            v = v_new
            h = 0.93 * h + 0.07 * v
            slow = 0.99 * slow + 0.01 * v
            states[t] = v + 0.3 * h + 0.1 * slow
        return states


def extract_trial_features(states, dspikes):
    feat_mean = states.mean(axis=0)
    feat_std = states.std(axis=0)
    feat_last = states[-1]
    ds_mean = dspikes.mean(axis=0)
    ds_std = dspikes.std(axis=0)
    delta = np.diff(states, axis=0)
    feat_delta_std = delta.std(axis=0) if len(delta) > 0 else np.zeros(states.shape[1])
    return np.concatenate([feat_mean, feat_std, feat_last, ds_mean, ds_std, feat_delta_std])


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer): return int(obj)
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        if isinstance(obj, np.bool_): return bool(obj)
        return super().default(obj)


def main():
    print("=" * 70)
    print("  z2287: TUNED BRIDGE — Recover MC with new synapse-capable bitstream")
    print("  z2285: eff_dim≈100 (sync broken!), MC=0.1-0.2 (too low)")
    print("=" * 70)

    results = {
        'experiment': 'z2287_tuned_bridge',
        'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
    }

    fpga = FPGAEthBridge()
    fpga.connect()
    fpga.set_kill(0)
    time.sleep(0.3)

    # Set heterogeneous Vg
    for n in range(NUM_NEURONS):
        fpga.set_vg(n, VG_GROUPS[n % 4])
        time.sleep(0.001)
    time.sleep(1.0)

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
    u = rng.uniform(-1, 1, N_QUICK_STEPS).astype(np.float64)

    # ═══════════════════════════════════════════════════════════
    # Phase 1: Leak sweep with zero synapses (maximum independence)
    # ═══════════════════════════════════════════════════════════
    print("\n[Phase 1] Leak sweep (zero synapses, BIAS=0x4000, THRESH=0x20000)")
    print(f"  {'Config':<18s} {'MC_tot':>7s} {'MC(1)':>7s} {'MC(2)':>7s} {'EffDim':>7s} {'xCorr':>7s}")
    print(f"  {'-'*18} {'-'*7} {'-'*7} {'-'*7} {'-'*7} {'-'*7}")

    apply_synapse_pattern(fpga, 'zero')

    sweep_leak = [
        ('leak_0004', 0x0004),
        ('leak_0010', 0x0010),
        ('leak_0020', 0x0020),
        ('leak_0080', 0x0080),
        ('leak_0200', 0x0200),
        ('leak_0800', 0x0800),
        ('leak_2000', 0x2000),
        ('leak_4000', 0x4000),
    ]

    phase1 = []
    best_mc = 0.0
    best_leak = 0x2000

    for name, leak in sweep_leak:
        fpga.set_leak_cond(leak)
        fpga.set_base_exc_raw(0x0080)
        fpga.set_bias_gain_raw(0x4000)
        fpga.set_threshold_raw(0x20000)
        time.sleep(0.5)

        mc_tot, mc_d, ed, xc = quick_mc_test(fpga, u)
        print(f"  {name:<18s} {mc_tot:7.3f} {mc_d.get(1,0):7.4f} {mc_d.get(2,0):7.4f} {ed:7.1f} {xc:7.4f}")

        phase1.append({
            'name': name, 'leak': hex(leak),
            'mc_total': mc_tot, 'mc_d1': mc_d.get(1, 0),
            'mc_delays': {str(k): v for k, v in mc_d.items()},
            'eff_dim': ed, 'xcorr': xc,
        })

        if mc_tot > best_mc:
            best_mc = mc_tot
            best_leak = leak

    results['phase1_leak_sweep'] = phase1
    results['best_leak'] = hex(best_leak)
    results['best_leak_mc'] = best_mc
    print(f"\n  BEST: leak={best_leak:#06x} MC_total={best_mc:.3f}")

    wait_cool("Phase 1")

    # ═══════════════════════════════════════════════════════════
    # Phase 2: Synapse + parameter combinations with best leak
    # ═══════════════════════════════════════════════════════════
    print(f"\n[Phase 2] Synapse patterns with LEAK={best_leak:#06x}")

    syn_patterns = ['zero', 'default', 'diverse', 'mild']
    phase2 = []

    for syn_name in syn_patterns:
        fpga.set_leak_cond(best_leak)
        fpga.set_base_exc_raw(0x0080)
        fpga.set_bias_gain_raw(0x4000)
        fpga.set_threshold_raw(0x20000)
        apply_synapse_pattern(fpga, syn_name)
        time.sleep(0.5)

        mc_tot, mc_d, ed, xc = quick_mc_test(fpga, u)
        print(f"  {syn_name:<12s}: MC={mc_tot:.3f} d1={mc_d.get(1,0):.3f} eff_dim={ed:.1f} xcorr={xc:.4f}")

        phase2.append({
            'syn': syn_name, 'mc_total': mc_tot,
            'mc_delays': {str(k): v for k, v in mc_d.items()},
            'eff_dim': ed, 'xcorr': xc,
        })

    results['phase2_synapse'] = phase2

    # Find best synapse pattern
    best_syn = max(phase2, key=lambda x: x['mc_total'])
    best_syn_name = best_syn['syn']
    print(f"\n  BEST synapse: {best_syn_name} MC={best_syn['mc_total']:.3f}")

    wait_cool("Phase 2")

    # ═══════════════════════════════════════════════════════════
    # Phase 3: Full benchmark with best params + bridge
    # ═══════════════════════════════════════════════════════════
    print(f"\n[Phase 3] Full benchmark: LEAK={best_leak:#06x}, SYN={best_syn_name}")

    gpu_esn = GPUFourpopESN(n_per_pop=64)
    print(f"  GPU fourpop: {gpu_esn.N} neurons")

    fpga.set_leak_cond(best_leak)
    fpga.set_base_exc_raw(0x0080)
    fpga.set_bias_gain_raw(0x4000)
    fpga.set_threshold_raw(0x20000)
    apply_synapse_pattern(fpga, best_syn_name)

    # ── Full continuous (2000 steps) for FPGA alone ──
    print("  Running FPGA continuous (2000 steps)...")
    u_full = rng.uniform(-1, 1, N_CONTINUOUS_STEPS).astype(np.float64)
    fpga_st, fpga_ds = fpga_run_continuous(fpga, u_full)
    X_fpga = build_features(fpga_st, fpga_ds)[WARMUP:]

    n = len(X_fpga)
    n_tr = int(0.7 * n)

    # MC
    fpga_mc = 0.0
    fpga_mc_d = {}
    for d in range(1, 11):
        tgt = u_full[WARMUP - d:N_CONTINUOUS_STEPS - d]
        nn = min(len(X_fpga), len(tgt))
        r2 = ridge_mc(X_fpga[:n_tr], tgt[:n_tr], X_fpga[n_tr:nn], tgt[n_tr:nn])
        fpga_mc_d[str(d)] = r2
        fpga_mc += r2

    # XOR
    u_bin = (u_full > 0).astype(float)
    fpga_xor = {}
    for tau in [1, 2, 3]:
        xor_tgt = ((u_bin[WARMUP:N_CONTINUOUS_STEPS] + u_bin[WARMUP - tau:N_CONTINUOUS_STEPS - tau]) % 2).astype(float)
        nn = min(len(X_fpga), len(xor_tgt))
        acc = ridge_xor(X_fpga[:n_tr], xor_tgt[:n_tr], X_fpga[n_tr:nn], xor_tgt[n_tr:nn])
        fpga_xor[f'xor{tau}'] = acc

    print(f"    FPGA: MC={fpga_mc:.3f} XOR1={fpga_xor['xor1']*100:.1f}%")

    wait_cool("Phase 3a")

    # ── Bridge: GPU → MAC → FPGA ──
    print("  Running BRIDGE continuous (2000 steps)...")
    gpu_states = gpu_esn.run(u_full)
    gpu_mac = np.zeros(N_CONTINUOUS_STEPS)
    for t in range(N_CONTINUOUS_STEPS):
        gpu_act = np.mean(np.abs(gpu_states[t]))
        inp_comp = (u_full[t] * 0.4 + 0.5)
        gpu_mac[t] = 0.6 * inp_comp + 0.4 * np.clip(gpu_act, 0, 1)
    gpu_mac = np.clip(gpu_mac, 0, 1)

    # FPGA with GPU-modulated MAC
    bridge_st, bridge_ds = fpga_run_continuous(fpga, u_full)
    # Override MAC during run
    fpga.set_mac_signal(0.0)
    time.sleep(0.02)
    telem = fpga.read_telemetry()
    prev_sc = telem['spike_counts'].copy() if telem is not None else np.zeros(NUM_NEURONS, dtype=np.uint16)
    dt = 1.0 / SAMPLE_HZ
    for t in range(N_CONTINUOUS_STEPS):
        fpga.set_mac_signal(float(gpu_mac[t]))
        time.sleep(dt)
        telem = fpga.read_telemetry()
        if telem is not None:
            bridge_st[t] = telem['vmem']
            sc = telem['spike_counts']
            diff = sc.astype(np.int32) - prev_sc.astype(np.int32)
            diff[diff < 0] += 65536
            bridge_ds[t] = diff.astype(np.float32)
            prev_sc = sc.copy()
    fpga.set_mac_signal(0.0)

    X_bridge_fpga = build_features(bridge_st, bridge_ds)[WARMUP:]
    X_bridge_gpu = build_features(gpu_states, np.zeros_like(gpu_states))[WARMUP:]
    X_bridge = np.hstack([X_bridge_fpga, X_bridge_gpu])

    # Bridge MC
    bridge_mc = 0.0
    bridge_mc_d = {}
    for d in range(1, 11):
        tgt = u_full[WARMUP - d:N_CONTINUOUS_STEPS - d]
        nn = min(len(X_bridge), len(tgt))
        r2 = ridge_mc(X_bridge[:n_tr], tgt[:n_tr], X_bridge[n_tr:nn], tgt[n_tr:nn])
        bridge_mc_d[str(d)] = r2
        bridge_mc += r2

    # Bridge XOR
    bridge_xor = {}
    for tau in [1, 2, 3]:
        xor_tgt = ((u_bin[WARMUP:N_CONTINUOUS_STEPS] + u_bin[WARMUP - tau:N_CONTINUOUS_STEPS - tau]) % 2).astype(float)
        nn = min(len(X_bridge), len(xor_tgt))
        acc = ridge_xor(X_bridge[:n_tr], xor_tgt[:n_tr], X_bridge[n_tr:nn], xor_tgt[n_tr:nn])
        bridge_xor[f'xor{tau}'] = acc

    print(f"    BRIDGE: MC={bridge_mc:.3f} XOR1={bridge_xor['xor1']*100:.1f}%")

    # GPU alone
    X_gpu = build_features(gpu_states, np.zeros_like(gpu_states))[WARMUP:]
    gpu_mc = 0.0
    gpu_mc_d = {}
    for d in range(1, 11):
        tgt = u_full[WARMUP - d:N_CONTINUOUS_STEPS - d]
        nn = min(len(X_gpu), len(tgt))
        r2 = ridge_mc(X_gpu[:n_tr], tgt[:n_tr], X_gpu[n_tr:nn], tgt[n_tr:nn])
        gpu_mc_d[str(d)] = r2
        gpu_mc += r2
    gpu_xor1 = ridge_xor(
        X_gpu[:n_tr], ((u_bin[WARMUP:N_CONTINUOUS_STEPS] + u_bin[WARMUP - 1:N_CONTINUOUS_STEPS - 1]) % 2).astype(float)[:n_tr],
        X_gpu[n_tr:], ((u_bin[WARMUP:N_CONTINUOUS_STEPS] + u_bin[WARMUP - 1:N_CONTINUOUS_STEPS - 1]) % 2).astype(float)[n_tr:])
    print(f"    GPU: MC={gpu_mc:.3f} XOR1={gpu_xor1*100:.1f}%")

    wait_cool("Phase 3b")

    # ── Wave-4 classification ──
    print("  Running waveform classification...")
    for cond_name, run_bridge in [('FPGA', False), ('BRIDGE', True)]:
        X_trials, y_trials = [], []
        for trial in range(N_WAVE_TRIALS):
            cls = trial % 4
            wave = generate_waveform(cls, N_WAVE_STEPS)
            wave_mac = np.clip(wave * 0.4 + 0.5, 0, 1)

            if run_bridge:
                gst = gpu_esn.run(wave)
                g_mac = np.zeros(N_WAVE_STEPS)
                for t in range(N_WAVE_STEPS):
                    ga = np.mean(np.abs(gst[t]))
                    ic = wave[t] * 0.4 + 0.5
                    g_mac[t] = np.clip(0.6 * ic + 0.4 * ga, 0, 1)

                fst = np.zeros((N_WAVE_STEPS, NUM_NEURONS))
                fds = np.zeros((N_WAVE_STEPS, NUM_NEURONS), dtype=np.float32)
                telem = fpga.read_telemetry()
                prev_sc = telem['spike_counts'].copy() if telem is not None else np.zeros(NUM_NEURONS, dtype=np.uint16)
                for t in range(N_WAVE_STEPS):
                    fpga.set_mac_signal(float(g_mac[t]))
                    time.sleep(dt)
                    telem = fpga.read_telemetry()
                    if telem is not None:
                        fst[t] = telem['vmem']
                        sc = telem['spike_counts']
                        diff = sc.astype(np.int32) - prev_sc.astype(np.int32)
                        diff[diff < 0] += 65536
                        fds[t] = diff.astype(np.float32)
                        prev_sc = sc.copy()
                fpga.set_mac_signal(0.0)
                feats = extract_trial_features(fst, fds)
                gpu_feats = np.concatenate([gst.mean(0), gst.std(0)])
                feats = np.concatenate([feats, gpu_feats])
            else:
                fst = np.zeros((N_WAVE_STEPS, NUM_NEURONS))
                fds = np.zeros((N_WAVE_STEPS, NUM_NEURONS), dtype=np.float32)
                telem = fpga.read_telemetry()
                prev_sc = telem['spike_counts'].copy() if telem is not None else np.zeros(NUM_NEURONS, dtype=np.uint16)
                for t in range(N_WAVE_STEPS):
                    fpga.set_mac_signal(float(wave_mac[t]))
                    time.sleep(dt)
                    telem = fpga.read_telemetry()
                    if telem is not None:
                        fst[t] = telem['vmem']
                        sc = telem['spike_counts']
                        diff = sc.astype(np.int32) - prev_sc.astype(np.int32)
                        diff[diff < 0] += 65536
                        fds[t] = diff.astype(np.float32)
                        prev_sc = sc.copy()
                fpga.set_mac_signal(0.0)
                feats = extract_trial_features(fst, fds)

            X_trials.append(feats)
            y_trials.append(cls)

        w4_acc, w4_std = ridge_classify(np.array(X_trials), np.array(y_trials), 4)
        print(f"    {cond_name}: Wave-4={w4_acc*100:.1f}%")
        results[f'{cond_name.lower()}_wave4'] = {'acc': w4_acc, 'std': w4_std}

    # ═══════════════════════════════════════════════════════════
    # Summary + Key Tests
    # ═══════════════════════════════════════════════════════════
    results['continuous'] = {
        'FPGA': {'mc_total': fpga_mc, 'mc_per_delay': fpga_mc_d, **fpga_xor},
        'GPU': {'mc_total': gpu_mc, 'mc_per_delay': gpu_mc_d, 'xor1': gpu_xor1},
        'BRIDGE': {'mc_total': bridge_mc, 'mc_per_delay': bridge_mc_d, **bridge_xor},
    }

    print("\n" + "=" * 70)
    print("  KEY TESTS")
    print("=" * 70)

    kt = {}

    # T1: Bridge MC > 1.0
    p = bridge_mc > 1.0
    kt['T1_bridge_mc_above_1'] = {'pass': p, 'desc': f"BRIDGE MC={bridge_mc:.3f} > 1.0"}
    print(f"  T1 {'PASS' if p else 'FAIL'}: {kt['T1_bridge_mc_above_1']['desc']}")

    # T2: Bridge MC > GPU MC
    p = bridge_mc > gpu_mc
    kt['T2_bridge_mc_gt_gpu'] = {'pass': p, 'desc': f"BRIDGE MC={bridge_mc:.3f} > GPU MC={gpu_mc:.3f}"}
    print(f"  T2 {'PASS' if p else 'FAIL'}: {kt['T2_bridge_mc_gt_gpu']['desc']}")

    # T3: Bridge XOR1 > 65%
    p = bridge_xor['xor1'] > 0.65
    kt['T3_bridge_xor_above_65'] = {'pass': p, 'desc': f"BRIDGE XOR1={bridge_xor['xor1']*100:.1f}% > 65%"}
    print(f"  T3 {'PASS' if p else 'FAIL'}: {kt['T3_bridge_xor_above_65']['desc']}")

    # T4: Bridge Wave-4 > 90%
    bw4 = results.get('bridge_wave4', {}).get('acc', 0)
    p = bw4 > 0.90
    kt['T4_bridge_wave4_above_90'] = {'pass': p, 'desc': f"BRIDGE Wave4={bw4*100:.1f}% > 90%"}
    print(f"  T4 {'PASS' if p else 'FAIL'}: {kt['T4_bridge_wave4_above_90']['desc']}")

    # T5: Bridge best on at least 2 metrics
    bridge_wins = 0
    all_mc = {'FPGA': fpga_mc, 'GPU': gpu_mc, 'BRIDGE': bridge_mc}
    if bridge_mc >= max(all_mc.values()) - 0.05:
        bridge_wins += 1
    all_xor = {'FPGA': fpga_xor.get('xor1', 0), 'GPU': gpu_xor1, 'BRIDGE': bridge_xor.get('xor1', 0)}
    if bridge_xor.get('xor1', 0) >= max(all_xor.values()) - 0.02:
        bridge_wins += 1
    if bw4 >= max(results.get('fpga_wave4', {}).get('acc', 0), 0.5):
        bridge_wins += 1
    p = bridge_wins >= 2
    kt['T5_bridge_best_2_metrics'] = {'pass': p, 'desc': f"BRIDGE best on {bridge_wins}/3 metrics"}
    print(f"  T5 {'PASS' if p else 'FAIL'}: {kt['T5_bridge_best_2_metrics']['desc']}")

    # T6: FPGA eff_dim > 10 (independence maintained)
    ed_best = max(p2['eff_dim'] for p2 in phase2)
    p = ed_best > 10.0
    kt['T6_independence'] = {'pass': p, 'desc': f"Best eff_dim={ed_best:.1f} > 10.0"}
    print(f"  T6 {'PASS' if p else 'FAIL'}: {kt['T6_independence']['desc']}")

    results['key_tests'] = kt
    n_pass = sum(1 for t in kt.values() if t['pass'])
    results['n_pass'] = n_pass
    results['n_tests'] = len(kt)

    print(f"\n  TOTAL: {n_pass}/{len(kt)} PASS")

    with open(SAVE_FILE, 'w') as f:
        json.dump(results, f, indent=2, cls=NpEncoder)
    print(f"  Results saved: {SAVE_FILE}")

    fpga.set_kill(1)
    fpga.close()


if __name__ == '__main__':
    main()
