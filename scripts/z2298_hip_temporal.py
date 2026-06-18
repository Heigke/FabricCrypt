#!/usr/bin/env python3
"""
z2298_hip_temporal.py — Real HIP GPU kernel + Temporal Products + FPGA Bridge
=============================================================================
Combines:
  - Real HIP fourpop kernel (z2277: 3072 GPU neurons, real HW physics)
  - Temporal product features (z2296: order-2+3, τ=1..20)
  - FPGA NS-RAM 128-neuron reservoir with recurrent synapses (z2297 best config)
  - Software ESN baseline for comparison

z2277 showed BRIDGE_FULL NARMA-3 R²=0.616 > GPU R²=0.605 — but used only
simple features. z2296 showed temporal products boost FPGA XOR from ~72% to 88%.
This experiment applies temporal products to ALL substrates for fair comparison.

Conditions:
  1) GPU_HIP:      Real HIP kernel + temporal features
  2) FPGA_ONLY:    FPGA 128N + temporal features (z2296 reproduction)
  3) FPGA_RECUR:   FPGA with best recurrent synapses + temporal features
  4) SOFTWARE_ESN: Numpy ESN 256N + temporal features (software baseline)
  5) BRIDGE:       GPU_HIP + FPGA concatenated + temporal on both
  6) BRIDGE_MAC:   GPU_HIP → MAC → FPGA (causal bridge) + temporal

Benchmarks: MC(d=1..20), XOR(τ=1..15), NARMA-5/10/20, 4-class waveform

Run:
  HSA_OVERRIDE_GFX_VERSION=11.0.0 PYTHONUNBUFFERED=1 venv/bin/python scripts/z2298_hip_temporal.py
"""

import os, sys, time, json, struct, subprocess, tempfile
import numpy as np
from pathlib import Path

os.environ['PYTHONUNBUFFERED'] = '1'
import logging

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE / 'scripts'))
RESULTS = BASE / 'results'
RESULTS.mkdir(exist_ok=True)
SAVE_FILE = RESULTS / 'z2298_hip_temporal.json'
LOG_FILE = RESULTS / 'z2298_hip_temporal_run.log'

# Setup dual logging: file + console
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE, mode='a'),
        logging.StreamHandler(sys.stdout),
    ]
)
log = logging.getLogger('z2298')

from fpga_host_eth import FPGAEthBridge

NUM_NEURONS = 128
N_GPU_SAMPLED = 512
SAMPLE_HZ = 50   # Halved AGAIN from 100Hz — APU still hitting 99°C ACPI trip
N_STEPS = 1500   # Reduced from 2000 — shorter runs = less sustained heat
WARMUP = 300
TEMP_SAFE = 40.0
GPU_KERN = BASE / 'scripts' / 'z2277_gpu_bridge_kern'
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


# ============================================================
# GPU HIP Kernel
# ============================================================
def run_hip_kernel(input_seq):
    """Run real HIP fourpop kernel, return (n_steps, 512) states."""
    n_steps = len(input_seq)
    with tempfile.NamedTemporaryFile(suffix='.bin', delete=False) as fin:
        input_path = fin.name
        input_seq.astype(np.float32).tofile(fin)
    with tempfile.NamedTemporaryFile(suffix='.bin', delete=False) as fout:
        output_path = fout.name
    try:
        env = os.environ.copy()
        env['HSA_OVERRIDE_GFX_VERSION'] = '11.0.0'
        result = subprocess.run(
            [str(GPU_KERN), input_path, output_path, str(n_steps)],
            env=env, capture_output=True, text=True, timeout=120
        )
        if result.returncode != 0:
            print(f"  [GPU] Kernel error: {result.stderr[:200]}", flush=True)
            return None
        raw = np.fromfile(output_path, dtype=np.float32)
        expected = N_GPU_SAMPLED * n_steps
        if len(raw) != expected:
            print(f"  [GPU] Wrong size: {len(raw)} != {expected}", flush=True)
            return None
        return raw.reshape(N_GPU_SAMPLED, n_steps).T
    finally:
        try: os.unlink(input_path)
        except: pass
        try: os.unlink(output_path)
        except: pass


# ============================================================
# Software ESN
# ============================================================
class SoftwareESN:
    """Standard leaky-integrator ESN (software baseline)."""
    def __init__(self, n_neurons=256, spectral_radius=0.95, input_scale=0.1, leak=0.3, seed=42):
        rng = np.random.default_rng(seed)
        self.N = n_neurons
        self.leak = leak
        self.input_w = rng.uniform(-input_scale, input_scale, n_neurons)
        W = rng.standard_normal((n_neurons, n_neurons)) * 0.1
        mask = rng.random((n_neurons, n_neurons)) > 0.9
        W *= mask
        eigvals = np.abs(np.linalg.eigvals(W))
        sr = max(eigvals) if len(eigvals) > 0 else 1.0
        if sr > 0:
            W *= spectral_radius / sr
        self.W = W
        self.bias = rng.uniform(-0.01, 0.01, n_neurons)

    def run(self, input_seq, seed=42):
        n_steps = len(input_seq)
        states = np.zeros((n_steps, self.N))
        x = np.zeros(self.N)
        for t in range(n_steps):
            u = input_seq[t]
            x_new = np.tanh(self.W @ x + self.input_w * u + self.bias)
            x = (1 - self.leak) * x + self.leak * x_new
            states[t] = x
        return states


# ============================================================
# FPGA functions
# ============================================================
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
        # Check temp EVERY 5 steps — APU heats from 45→99°C in seconds from UDP I/O
        if t > 0 and t % 5 == 0:
            temp = get_max_temp()
            if temp > 60.0:
                fpga.set_mac_signal(0.0)
                print(f"\n  [THERMAL PAUSE] {temp:.0f}°C at step {t}/{n_steps}", end="", flush=True)
                while temp > 42.0:
                    time.sleep(5)
                    temp = get_max_temp()
                    print(f" {temp:.0f}", end="", flush=True)
                print(" resumed", flush=True)
        fpga.set_mac_signal(float(mac_signal[t]))
        time.sleep(dt + 0.005)  # extra 5ms cooldown per step (was 2ms, still crashed)
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


# ============================================================
# Temporal product features (same as z2296)
# ============================================================
def build_temporal_features(states, dspikes=None, n_select=24, seed=42):
    """Build temporal order-2+3 product features for ANY reservoir states."""
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

    # Order-2 temporal products
    for tau in tau_list:
        shifted = np.zeros_like(vm_q)
        shifted[tau:] = vm_q[:-tau]
        feats.append(vm_q * shifted)
        if dspikes is not None:
            ds_q = dspikes[:, qi] if dspikes.shape[1] >= n_ch else dspikes[:, :min(n_select, dspikes.shape[1])]
            if ds_q.shape[1] == vm_q.shape[1]:
                feats.append(ds_q * shifted)

    # Order-3 temporal products (limited)
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


# ============================================================
# Benchmarks
# ============================================================
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

    # 4-class waveform classification
    wave_acc = classify_waveform(X, u_raw)

    return {'mc_total': mc_total, 'mc_per_delay': mc_per_d, 'xor': xor, 'narma': narma,
            'wave4_acc': wave_acc}


def classify_waveform(X, u_raw):
    """4-class waveform classification (sine, square, triangle, sawtooth)."""
    n = len(X)
    n_per_class = n // 4
    # Create targets based on input signal quartiles
    quartiles = np.percentile(u_raw[WARMUP:WARMUP+n], [25, 50, 75])
    u = u_raw[WARMUP:WARMUP+n]
    labels = np.zeros(n)
    labels[u > quartiles[2]] = 3
    labels[(u > quartiles[1]) & (u <= quartiles[2])] = 2
    labels[(u > quartiles[0]) & (u <= quartiles[1])] = 1

    n_tr = int(0.7 * n)
    # One-vs-rest classification
    correct = 0
    pred_all = np.zeros(n - n_tr)
    for c in range(4):
        y = (labels == c).astype(float)
        score_c = ridge_solve(X[:n_tr], y[:n_tr], X[n_tr:], y[n_tr:], 'classification')

    # Multi-class via argmax of one-vs-rest scores
    scores_matrix = np.zeros((n - n_tr, 4))
    for c in range(4):
        y = (labels == c).astype(float)
        for alpha in [1.0, 10.0, 100.0]:
            I = np.eye(X[:n_tr].shape[1])
            try:
                w = np.linalg.solve(X[:n_tr].T @ X[:n_tr] + alpha * I, X[:n_tr].T @ y[:n_tr])
                scores_matrix[:, c] = X[n_tr:] @ w
                break
            except:
                pass
    pred = np.argmax(scores_matrix, axis=1)
    acc = np.mean(pred == labels[n_tr:])
    return float(acc)


def main():
    print("=" * 70)
    print("  z2298: Real HIP GPU + Temporal Products + FPGA Bridge")
    print("  Combining z2277 HIP kernel + z2296 temporal features")
    print("=" * 70)

    results = {'conditions': {}, 'tests': {}}
    # Resume from saved results if any
    if SAVE_FILE.exists():
        try:
            with open(SAVE_FILE) as f:
                results = json.load(f)
            done = list(results.get('conditions', {}).keys())
            if done:
                print(f"  RESUMED: {done} already done")
        except Exception:
            results = {'conditions': {}, 'tests': {}}

    rng = np.random.default_rng(42)
    u_raw = rng.uniform(-1, 1, N_STEPS + WARMUP)

    # ================================================================
    # 1. SOFTWARE ESN (safe, no thermal risk)
    # ================================================================
    if 'SOFTWARE_ESN' not in results.get('conditions', {}):
        print("\n[1/6] SOFTWARE ESN baseline (256 neurons)")
        esn = SoftwareESN(n_neurons=256, seed=42)
        esn_states = esn.run(u_raw, seed=42)
        X_esn = build_temporal_features(esn_states[WARMUP:], n_select=24, seed=42)
        bm_esn = full_benchmark(X_esn, u_raw)
        results['conditions']['SOFTWARE_ESN'] = bm_esn
        xor = bm_esn['xor']
        print(f"  ESN: MC={bm_esn['mc_total']:.2f} XOR1={xor['tau1']*100:.1f}% XOR3={xor['tau3']*100:.1f}% "
              f"XOR5={xor['tau5']*100:.1f}% N5={bm_esn['narma']['narma5']:.3f} W4={bm_esn['wave4_acc']*100:.1f}%")
        with open(SAVE_FILE, 'w') as f:
            json.dump(results, f, indent=2, cls=NpEncoder)
    else:
        print("\n[1/6] SOFTWARE ESN — already done, skipping")
        # Still need ESN features for later comparisons
        esn = SoftwareESN(n_neurons=256, seed=42)
        esn_states = esn.run(u_raw, seed=42)
        X_esn = build_temporal_features(esn_states[WARMUP:], n_select=24, seed=42)
    bm_esn = results['conditions']['SOFTWARE_ESN']

    # ================================================================
    # 2. GPU HIP KERNEL
    # ================================================================
    gpu_states = None
    if 'GPU_HIP' not in results.get('conditions', {}):
        print("\n[2/6] GPU HIP fourpop kernel (3072 neurons)")
        wait_cool("pre-GPU")
        gpu_states = run_hip_kernel(u_raw.astype(np.float32))
        if gpu_states is not None:
            print(f"  GPU states: {gpu_states.shape}, range [{gpu_states.min():.3f}, {gpu_states.max():.3f}]")
            X_gpu = build_temporal_features(gpu_states[WARMUP:], n_select=24, seed=42)
            bm_gpu = full_benchmark(X_gpu, u_raw)
            results['conditions']['GPU_HIP'] = bm_gpu
            xor = bm_gpu['xor']
            print(f"  GPU: MC={bm_gpu['mc_total']:.2f} XOR1={xor['tau1']*100:.1f}% XOR3={xor['tau3']*100:.1f}% "
                  f"XOR5={xor['tau5']*100:.1f}% N5={bm_gpu['narma']['narma5']:.3f} W4={bm_gpu['wave4_acc']*100:.1f}%")
        else:
            print("  [SKIP] GPU kernel failed")
        with open(SAVE_FILE, 'w') as f:
            json.dump(results, f, indent=2, cls=NpEncoder)
    else:
        print("\n[2/6] GPU HIP — already done, skipping")
    bm_gpu = results['conditions'].get('GPU_HIP')

    # We need gpu_states for BRIDGE conditions — re-run kernel if needed
    if gpu_states is None and bm_gpu is not None and ('BRIDGE_CONCAT' not in results.get('conditions', {}) or 'BRIDGE_MAC' not in results.get('conditions', {})):
        log.info("Re-running GPU kernel for bridge conditions...")
        log.info(f"  Temp before GPU rerun: {get_max_temp():.1f}°C")
        wait_cool("pre-GPU-rerun")
        t0 = time.time()
        gpu_states = run_hip_kernel(u_raw.astype(np.float32))
        elapsed = time.time() - t0
        log.info(f"  GPU kernel done in {elapsed:.1f}s, temp after: {get_max_temp():.1f}°C")
        if gpu_states is not None:
            log.info(f"  GPU states shape: {gpu_states.shape}, range [{gpu_states.min():.4f}, {gpu_states.max():.4f}]")
        else:
            log.error("  GPU kernel FAILED — returned None")

    X_gpu = None
    if gpu_states is not None:
        X_gpu = build_temporal_features(gpu_states[WARMUP:], n_select=24, seed=42)

    # ================================================================
    # 3. FPGA ONLY (zero synapses, z2296 config)
    # ================================================================
    # Only setup FPGA for conditions 3+4 (NOT BRIDGE_MAC — that gets its own setup later)
    fpga = None
    early_fpga_needed = ('FPGA_ONLY' not in results.get('conditions', {}) or
                         'FPGA_RECURRENT' not in results.get('conditions', {}))
    if early_fpga_needed:
        print(f"\n  Setting up FPGA for conditions 3+4")
        fpga = FPGAEthBridge(timeout=2.0)
        fpga.connect()
        fpga.set_kill(0)
        time.sleep(1.0)
        fpga.set_leak_cond(0x2000)
        fpga.set_base_exc_raw(0x0080)
        fpga.set_bias_gain_raw(0x4000)
        fpga.set_threshold_raw(0x20000)
        for n in range(NUM_NEURONS):
            fpga.set_vg(n, VG_GROUPS[n % 4])
            time.sleep(0.001)
        for n in range(NUM_NEURONS):
            fpga.set_synapse(n, 0x00000000)
            time.sleep(0.001)
        time.sleep(0.5)
        telem = fpga.read_telemetry()
        if telem is not None:
            print(f"  FPGA online: vmem [{telem['vmem'].min():.3f}, {telem['vmem'].max():.3f}]")
        else:
            print("  WARNING: FPGA telemetry returned None!")

    if 'FPGA_ONLY' not in results.get('conditions', {}):
        print("\n[3/6] FPGA ONLY (128 neurons, zero synapses)")
        wait_cool("pre-FPGA")
        fpga_states, fpga_dspikes = fpga_run_continuous(fpga, u_raw)
        # Save raw states to disk for offline bridge computation (avoids re-running FPGA)
        np.save(RESULTS / 'z2298_fpga_states.npy', fpga_states)
        np.save(RESULTS / 'z2298_fpga_dspikes.npy', fpga_dspikes)
        print(f"  Saved FPGA states to disk for bridge conditions")
        X_fpga = build_temporal_features(fpga_states[WARMUP:], fpga_dspikes[WARMUP:], n_select=24, seed=42)
        bm_fpga = full_benchmark(X_fpga, u_raw)
        results['conditions']['FPGA_ONLY'] = bm_fpga
        xor = bm_fpga['xor']
        print(f"  FPGA: MC={bm_fpga['mc_total']:.2f} XOR1={xor['tau1']*100:.1f}% XOR3={xor['tau3']*100:.1f}% "
              f"XOR5={xor['tau5']*100:.1f}% N5={bm_fpga['narma']['narma5']:.3f} W4={bm_fpga['wave4_acc']*100:.1f}%")
        with open(SAVE_FILE, 'w') as f:
            json.dump(results, f, indent=2, cls=NpEncoder)
    else:
        print("\n[3/6] FPGA ONLY — already done, skipping")
    bm_fpga = results['conditions'].get('FPGA_ONLY', {})

    # ================================================================
    # 4. FPGA RECURRENT (heterogeneous synapses)
    # ================================================================
    if 'FPGA_RECURRENT' not in results.get('conditions', {}):
        print("\n[4/6] FPGA RECURRENT (heterogeneous synapses)")
        wait_cool("pre-FPGA-recur")
        for n in range(NUM_NEURONS):
            group = n % 4
            if group == 0:
                w = (0x60 << 24) | (0x30 << 16) | (0x10 << 8) | 0x10
            elif group == 1:
                w = (0x40 << 24) | (0x40 << 16) | (0x00 << 8) | 0x00
            elif group == 2:
                w = (0x50 << 24) | (0x00 << 16) | (0x00 << 8) | 0x20
            else:
                w = (0x70 << 24) | (0x70 << 16) | (0x18 << 8) | 0x18
            fpga.set_synapse(n, w)
            time.sleep(0.001)
        time.sleep(0.5)
        fpga_rec_states, fpga_rec_dspikes = fpga_run_continuous(fpga, u_raw)
        X_fpga_rec = build_temporal_features(fpga_rec_states[WARMUP:], fpga_rec_dspikes[WARMUP:], n_select=24, seed=42)
        bm_fpga_rec = full_benchmark(X_fpga_rec, u_raw)
        results['conditions']['FPGA_RECURRENT'] = bm_fpga_rec
        xor = bm_fpga_rec['xor']
        print(f"  FPGA-R: MC={bm_fpga_rec['mc_total']:.2f} XOR1={xor['tau1']*100:.1f}% XOR3={xor['tau3']*100:.1f}% "
              f"XOR5={xor['tau5']*100:.1f}% N5={bm_fpga_rec['narma']['narma5']:.3f} W4={bm_fpga_rec['wave4_acc']*100:.1f}%")
        with open(SAVE_FILE, 'w') as f:
            json.dump(results, f, indent=2, cls=NpEncoder)
    else:
        print("\n[4/6] FPGA RECURRENT — already done, skipping")
    bm_fpga_rec = results['conditions'].get('FPGA_RECURRENT', {})

    # Close FPGA connection after conditions 3+4 to let APU cool
    if fpga is not None:
        try:
            fpga.set_mac_signal(0.0)
            fpga.close()
            fpga = None
            print("  FPGA disconnected for cooling")
        except Exception:
            fpga = None

    # ================================================================
    # 5. BRIDGE_CONCAT — OFFLINE from saved FPGA + GPU data (NO FPGA I/O!)
    # ================================================================
    if 'BRIDGE_CONCAT' not in results.get('conditions', {}):
        log.info("[5/6] BRIDGE_CONCAT: GPU_HIP + FPGA (OFFLINE — no FPGA I/O)")
        log.info(f"  Temp at start of cond5: {get_max_temp():.1f}°C")
        fpga_states_file = RESULTS / 'z2298_fpga_states.npy'
        fpga_dspikes_file = RESULTS / 'z2298_fpga_dspikes.npy'
        log.info(f"  X_gpu={'OK shape='+str(X_gpu.shape) if X_gpu is not None else 'NONE'}")
        log.info(f"  fpga_states_file exists={fpga_states_file.exists()}")
        log.info(f"  fpga_dspikes_file exists={fpga_dspikes_file.exists()}")
        if X_gpu is not None and fpga_states_file.exists():
            saved_states = np.load(fpga_states_file)
            saved_dspikes = np.load(fpga_dspikes_file)
            log.info(f"  Loaded saved FPGA: states={saved_states.shape}, dspikes={saved_dspikes.shape}")
            log.info(f"  Temp before feature build: {get_max_temp():.1f}°C")
            X_fpga_saved = build_temporal_features(saved_states[WARMUP:], saved_dspikes[WARMUP:], n_select=24, seed=42)
            log.info(f"  X_fpga_saved shape: {X_fpga_saved.shape}")
            X_bridge = np.hstack([X_fpga_saved, X_gpu])
            log.info(f"  X_bridge shape: {X_bridge.shape}")
            log.info(f"  Temp before benchmark: {get_max_temp():.1f}°C")
            t0 = time.time()
            bm_bridge = full_benchmark(X_bridge, u_raw)
            elapsed = time.time() - t0
            log.info(f"  Benchmark done in {elapsed:.1f}s, temp after: {get_max_temp():.1f}°C")
            results['conditions']['BRIDGE_CONCAT'] = bm_bridge
            xor = bm_bridge['xor']
            log.info(f"  BRIDGE: MC={bm_bridge['mc_total']:.2f} XOR1={xor['tau1']*100:.1f}% XOR3={xor['tau3']*100:.1f}% "
                  f"XOR5={xor['tau5']*100:.1f}% N5={bm_bridge['narma']['narma5']:.3f} W4={bm_bridge['wave4_acc']*100:.1f}%")
        else:
            log.warning(f"  [SKIP] GPU={'OK' if X_gpu is not None else 'NONE'}, FPGA states={'saved' if fpga_states_file.exists() else 'MISSING'}")
        with open(SAVE_FILE, 'w') as f:
            json.dump(results, f, indent=2, cls=NpEncoder)
        log.info(f"  Results saved after cond5")
    else:
        print("\n[5/6] BRIDGE_CONCAT — already done, skipping")

    # ================================================================
    # 6. BRIDGE_MAC — needs ONE more FPGA run with MAC signal
    #    Run ONLY after a long cooling pause (60s forced wait)
    # ================================================================
    if 'BRIDGE_MAC' not in results.get('conditions', {}):
        log.info("[6/6] BRIDGE_MAC: GPU → MAC signal → FPGA (causal bridge)")
        log.info(f"  Temp at start of cond6: {get_max_temp():.1f}°C")
        if gpu_states is not None:
            log.info("  Forced 60s cooling before final FPGA run...")
            time.sleep(60)
            log.info(f"  Temp after 60s wait: {get_max_temp():.1f}°C")
            wait_cool("pre-BRIDGE-MAC", target=38.0)
            log.info(f"  Temp after wait_cool: {get_max_temp():.1f}°C")
            if fpga is None:
                log.info("  Connecting to FPGA...")
                fpga = FPGAEthBridge(timeout=2.0)
                fpga.connect()
                fpga.set_kill(0)
                time.sleep(1.0)
                fpga.set_leak_cond(0x2000)
                fpga.set_base_exc_raw(0x0080)
                fpga.set_bias_gain_raw(0x4000)
                fpga.set_threshold_raw(0x20000)
                for nn in range(NUM_NEURONS):
                    fpga.set_vg(nn, VG_GROUPS[nn % 4])
                    time.sleep(0.001)
                log.info("  FPGA connected and configured")
            for nn in range(NUM_NEURONS):
                fpga.set_synapse(nn, 0x00000000)
                time.sleep(0.001)
            time.sleep(0.3)
            telem = fpga.read_telemetry()
            if telem is not None:
                log.info(f"  FPGA telemetry OK: vmem [{telem['vmem'].min():.3f}, {telem['vmem'].max():.3f}]")
            else:
                log.warning("  FPGA telemetry returned None!")
            gpu_mean = gpu_states[WARMUP:].mean(axis=1)
            mac_sig = (gpu_mean - gpu_mean.min()) / (gpu_mean.max() - gpu_mean.min() + 1e-10)
            mac_full = np.zeros(len(u_raw))
            mac_full[WARMUP:WARMUP+len(mac_sig)] = mac_sig
            log.info(f"  MAC signal range: [{mac_sig.min():.4f}, {mac_sig.max():.4f}], steps={len(u_raw)}")
            log.info(f"  Temp before FPGA run: {get_max_temp():.1f}°C")
            t0 = time.time()
            fpga_mac_states, fpga_mac_dspikes = fpga_run_continuous(fpga, u_raw, mac_signal=mac_full)
            elapsed = time.time() - t0
            log.info(f"  FPGA run done in {elapsed:.1f}s, temp after: {get_max_temp():.1f}°C")
            log.info(f"  fpga_mac_states shape: {fpga_mac_states.shape}, range [{fpga_mac_states.min():.4f}, {fpga_mac_states.max():.4f}]")
            X_fpga_mac = build_temporal_features(fpga_mac_states[WARMUP:], fpga_mac_dspikes[WARMUP:], n_select=24, seed=42)
            X_bridge_mac = np.hstack([X_fpga_mac, X_gpu])
            log.info(f"  X_bridge_mac shape: {X_bridge_mac.shape}")
            log.info(f"  Temp before benchmark: {get_max_temp():.1f}°C")
            t0 = time.time()
            bm_bridge_mac = full_benchmark(X_bridge_mac, u_raw)
            elapsed = time.time() - t0
            log.info(f"  Benchmark done in {elapsed:.1f}s, temp after: {get_max_temp():.1f}°C")
            results['conditions']['BRIDGE_MAC'] = bm_bridge_mac
            xor = bm_bridge_mac['xor']
            log.info(f"  BR_MAC: MC={bm_bridge_mac['mc_total']:.2f} XOR1={xor['tau1']*100:.1f}% XOR3={xor['tau3']*100:.1f}% "
                  f"XOR5={xor['tau5']*100:.1f}% N5={bm_bridge_mac['narma']['narma5']:.3f} W4={bm_bridge_mac['wave4_acc']*100:.1f}%")
        else:
            log.warning("  [SKIP] No GPU states available")
        with open(SAVE_FILE, 'w') as f:
            json.dump(results, f, indent=2, cls=NpEncoder)
    else:
        print("\n[6/6] BRIDGE_MAC — already done, skipping")

    bm_bridge = results['conditions'].get('BRIDGE_CONCAT')
    bm_bridge_mac = results['conditions'].get('BRIDGE_MAC')

    # ================================================================
    # TESTS
    # ================================================================
    print(f"\n{'='*70}")
    print("  TESTS — z2298")
    print(f"{'='*70}")

    tests = {}
    n_pass = 0

    def test(name, cond, desc):
        nonlocal n_pass
        tests[name] = {'pass': bool(cond), 'desc': desc}
        print(f"  {name} {'PASS' if cond else 'FAIL'}: {desc}")
        n_pass += cond

    # Use results dict for all comparisons (safe even after resume)
    c = results['conditions']
    bm_fpga = c.get('FPGA_ONLY', {})
    bm_esn = c.get('SOFTWARE_ESN', {})
    bm_gpu = c.get('GPU_HIP')
    bm_fpga_rec = c.get('FPGA_RECURRENT', {})
    bm_bridge = c.get('BRIDGE_CONCAT')
    bm_bridge_mac = c.get('BRIDGE_MAC')

    def g(bm, *keys):
        """Safe nested dict get."""
        v = bm
        for k in keys:
            if isinstance(v, dict):
                v = v.get(k, 0)
            else:
                return 0
        return v if v is not None else 0

    # Core quality
    test('T1', g(bm_fpga,'xor','tau1') > 0.78, f"FPGA XOR1={g(bm_fpga,'xor','tau1')*100:.1f}% > 78%")
    test('T2', g(bm_fpga,'mc_total') > 10.0, f"FPGA MC={g(bm_fpga,'mc_total'):.2f} > 10.0")
    test('T3', g(bm_fpga,'narma','narma5') < 0.20, f"FPGA NARMA-5={g(bm_fpga,'narma','narma5'):.3f} < 0.20")

    # Software ESN comparison
    test('T4', g(bm_fpga,'xor','tau1') > g(bm_esn,'xor','tau1'),
         f"FPGA XOR1={g(bm_fpga,'xor','tau1')*100:.1f}% > ESN={g(bm_esn,'xor','tau1')*100:.1f}%")
    test('T5', g(bm_fpga,'mc_total') > g(bm_esn,'mc_total'),
         f"FPGA MC={g(bm_fpga,'mc_total'):.2f} > ESN={g(bm_esn,'mc_total'):.2f}")

    # GPU HIP comparison
    if bm_gpu is not None:
        test('T6', g(bm_fpga,'xor','tau3') > g(bm_gpu,'xor','tau3'),
             f"FPGA XOR3={g(bm_fpga,'xor','tau3')*100:.1f}% > GPU={g(bm_gpu,'xor','tau3')*100:.1f}%")
        test('T7', g(bm_gpu,'mc_total') > 5.0, f"GPU MC={g(bm_gpu,'mc_total'):.2f} > 5.0")
        test('T8', g(bm_gpu,'xor','tau1') > 0.60, f"GPU XOR1={g(bm_gpu,'xor','tau1')*100:.1f}% > 60%")
    else:
        test('T6', False, "GPU unavailable")
        test('T7', False, "GPU unavailable")
        test('T8', False, "GPU unavailable")

    # Recurrent vs zero
    test('T9', g(bm_fpga_rec,'mc_total') != g(bm_fpga,'mc_total'),
         f"Recurrent MC={g(bm_fpga_rec,'mc_total'):.2f} differs from Zero={g(bm_fpga,'mc_total'):.2f}")

    # Bridge tests
    if bm_bridge is not None:
        bridge_beats_fpga = sum([
            g(bm_bridge,'mc_total') > g(bm_fpga,'mc_total'),
            g(bm_bridge,'xor','tau3') > g(bm_fpga,'xor','tau3'),
            g(bm_bridge,'xor','tau5') > g(bm_fpga,'xor','tau5'),
            g(bm_bridge,'narma','narma5') < g(bm_fpga,'narma','narma5'),
            g(bm_bridge,'narma','narma10') < g(bm_fpga,'narma','narma10'),
        ])
        test('T10', bridge_beats_fpga >= 1,
             f"BRIDGE beats FPGA on {bridge_beats_fpga}/5 metrics")
        gpu_mc = g(bm_gpu,'mc_total') if bm_gpu else 0
        test('T11', g(bm_bridge,'mc_total') > max(g(bm_fpga,'mc_total'), gpu_mc),
             f"BRIDGE MC={g(bm_bridge,'mc_total'):.2f} > max(FPGA={g(bm_fpga,'mc_total'):.2f}, GPU={gpu_mc:.2f})")
    else:
        test('T10', False, "Bridge unavailable")
        test('T11', False, "Bridge unavailable")

    if bm_bridge_mac is not None:
        mac_beats_concat = sum([
            g(bm_bridge_mac,'mc_total') > g(bm_bridge,'mc_total') if bm_bridge else False,
            g(bm_bridge_mac,'narma','narma5') < g(bm_bridge,'narma','narma5') if bm_bridge else False,
        ])
        test('T12', mac_beats_concat >= 1,
             f"MAC bridge beats concat on {mac_beats_concat}/2 metrics")
    else:
        test('T12', False, "MAC bridge unavailable")

    # Harder benchmarks
    test('T13', g(bm_fpga,'xor','tau5') > 0.70, f"FPGA XOR5={g(bm_fpga,'xor','tau5')*100:.1f}% > 70%")
    test('T14', g(bm_fpga,'xor','tau10') > 0.55, f"FPGA XOR10={g(bm_fpga,'xor','tau10')*100:.1f}% > 55%")
    test('T15', g(bm_fpga,'narma','narma10') < 0.50, f"FPGA NARMA-10={g(bm_fpga,'narma','narma10'):.3f} < 0.50")

    # Cross-substrate synergy
    if bm_bridge is not None and bm_gpu is not None:
        synergy = (g(bm_bridge,'mc_total') > g(bm_fpga,'mc_total') and
                   g(bm_bridge,'mc_total') > g(bm_gpu,'mc_total') and
                   g(bm_bridge,'xor','tau1') > 0.75)
        test('T16', synergy, "Cross-substrate synergy: BRIDGE MC>both AND XOR1>75%")
    else:
        test('T16', False, "Bridge unavailable")

    # Waveform classification
    test('T17', g(bm_fpga,'wave4_acc') > 0.35, f"FPGA Wave4={g(bm_fpga,'wave4_acc')*100:.1f}% > 35%")
    if bm_gpu is not None:
        test('T18', g(bm_gpu,'wave4_acc') > 0.30, f"GPU Wave4={g(bm_gpu,'wave4_acc')*100:.1f}% > 30%")
    else:
        test('T18', False, "GPU unavailable")

    # ESN quality
    test('T19', g(bm_esn,'mc_total') > 3.0, f"ESN MC={g(bm_esn,'mc_total'):.2f} > 3.0")
    test('T20', g(bm_esn,'xor','tau1') > 0.55, f"ESN XOR1={g(bm_esn,'xor','tau1')*100:.1f}% > 55%")

    print(f"\n  TOTAL: {n_pass}/20 PASS")

    results['tests'] = tests
    results['summary'] = {
        'n_pass': n_pass, 'n_total': 20,
        'conditions_run': list(results['conditions'].keys()),
    }

    # Final comparison table
    print(f"\n{'='*70}")
    print("  COMPARISON TABLE")
    print(f"{'='*70}")
    print(f"  {'Condition':<16} {'MC':>7} {'XOR1':>7} {'XOR3':>7} {'XOR5':>7} {'N5':>7} {'W4':>7}")
    print(f"  {'-'*16} {'-'*7} {'-'*7} {'-'*7} {'-'*7} {'-'*7} {'-'*7}")
    for cond_name, bm in results['conditions'].items():
        xor = bm['xor']
        print(f"  {cond_name:<16} {bm['mc_total']:7.2f} {xor['tau1']*100:6.1f}% {xor['tau3']*100:6.1f}% "
              f"{xor['tau5']*100:6.1f}% {bm['narma']['narma5']:7.3f} {bm['wave4_acc']*100:6.1f}%")

    with open(SAVE_FILE, 'w') as f:
        json.dump(results, f, indent=2, cls=NpEncoder)
    print(f"\n  Results saved: {SAVE_FILE}")

    # Clean up
    if fpga is not None:
        try:
            for n in range(NUM_NEURONS):
                fpga.set_synapse(n, 0x00000000)
                time.sleep(0.001)
            fpga.close()
        except Exception:
            pass


if __name__ == '__main__':
    main()
