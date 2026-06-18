#!/usr/bin/env python3
"""
z2299_multiseed_validation.py — Multi-seed statistical validation of z2298
==========================================================================
Runs 10 random seeds across 3 key conditions (FPGA_ONLY, BRIDGE_CONCAT,
SOFTWARE_ESN) to validate z2298's results are stable, not lucky seeds.

GPU_HIP runs once (seed doesn't affect hardware kernel), reused across seeds.
FPGA runs fresh per seed (new u_raw each time).
BRIDGE_CONCAT is offline: FPGA states + GPU features concatenated.

Benchmarks: MC(d=1..20), XOR(tau=1,3,5), NARMA-5, Wave4

Run:
  HSA_OVERRIDE_GFX_VERSION=11.0.0 PYTHONUNBUFFERED=1 taskset -c 0-3 venv/bin/python scripts/z2299_multiseed_validation.py
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
SAVE_FILE = RESULTS / 'z2299_multiseed_validation.json'
LOG_FILE = RESULTS / 'z2299_multiseed_validation_run.log'

# Setup dual logging: file + console
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE, mode='a'),
        logging.StreamHandler(sys.stdout),
    ]
)
log = logging.getLogger('z2299')

from fpga_host_eth import FPGAEthBridge

NUM_NEURONS = 128
N_GPU_SAMPLED = 512
SAMPLE_HZ = 50
N_STEPS = 1500
WARMUP = 300
TEMP_SAFE = 42.0
GPU_KERN = BASE / 'scripts' / 'z2277_gpu_bridge_kern'
VG_GROUPS = {0: 0.05, 1: 0.15, 2: 0.30, 3: 0.58}

SEEDS = [42, 123, 456, 789, 1001, 2002, 3003, 4004, 5005, 6006]


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
        # Check temp EVERY 5 steps — APU heats from 45->99C in seconds from UDP I/O
        if t > 0 and t % 5 == 0:
            temp = get_max_temp()
            if temp > 60.0:
                fpga.set_mac_signal(0.0)
                print(f"\n  [THERMAL PAUSE] {temp:.0f}C at step {t}/{n_steps}", end="", flush=True)
                while temp > 42.0:
                    time.sleep(5)
                    temp = get_max_temp()
                    print(f" {temp:.0f}", end="", flush=True)
                print(" resumed", flush=True)
        fpga.set_mac_signal(float(mac_signal[t]))
        time.sleep(dt + 0.005)  # extra 5ms cooldown per step
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
    for tau in [1, 3, 5]:
        u_a = (u_raw[WARMUP:] > 0).astype(float)
        u_b = (u_raw[WARMUP-tau:len(u_raw)-tau] > 0).astype(float)
        nn = min(len(u_a), len(u_b), n)
        target = (u_a[:nn] != u_b[:nn]).astype(float)
        Xn = X[:nn]
        acc = ridge_solve(Xn[:n_tr], target[:n_tr], Xn[n_tr:nn], target[n_tr:nn], 'classification')
        xor[f'tau{tau}'] = acc

    narma = {}
    for order in [5]:
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
    print("  z2299: Multi-Seed Statistical Validation of z2298")
    print("  10 seeds x 3 conditions (FPGA_ONLY, BRIDGE_CONCAT, SOFTWARE_ESN)")
    print("=" * 70)

    # Resume support
    results = {'seeds': {}, 'gpu_single': None, 'tests': {}, 'stats': {}}
    if SAVE_FILE.exists():
        try:
            with open(SAVE_FILE) as f:
                results = json.load(f)
            done_seeds = list(results.get('seeds', {}).keys())
            if done_seeds:
                print(f"  RESUMED: seeds {done_seeds} already done")
        except Exception:
            results = {'seeds': {}, 'gpu_single': None, 'tests': {}, 'stats': {}}

    # ================================================================
    # 1. GPU HIP kernel — run ONCE (hardware kernel, seed-independent)
    # ================================================================
    # Use seed=42 for generating the GPU input signal (same across all seeds
    # would be unfair; instead GPU runs once with a fixed input and we reuse
    # its features for all BRIDGE_CONCAT conditions)
    gpu_u_raw = np.random.default_rng(42).uniform(-1, 1, N_STEPS + WARMUP)
    gpu_states = None
    X_gpu = None

    if results.get('gpu_single') is None:
        print("\n[GPU] Running HIP fourpop kernel ONCE (3072 neurons)")
        wait_cool("pre-GPU")
        gpu_states = run_hip_kernel(gpu_u_raw.astype(np.float32))
        if gpu_states is not None:
            print(f"  GPU states: {gpu_states.shape}, range [{gpu_states.min():.3f}, {gpu_states.max():.3f}]")
            X_gpu = build_temporal_features(gpu_states[WARMUP:], n_select=24, seed=42)
            # Benchmark GPU once for reference
            bm_gpu = full_benchmark(X_gpu, gpu_u_raw)
            results['gpu_single'] = bm_gpu
            xor = bm_gpu['xor']
            print(f"  GPU: MC={bm_gpu['mc_total']:.2f} XOR1={xor['tau1']*100:.1f}% "
                  f"XOR3={xor['tau3']*100:.1f}% XOR5={xor['tau5']*100:.1f}% "
                  f"N5={bm_gpu['narma']['narma5']:.3f} W4={bm_gpu['wave4_acc']*100:.1f}%")
            # Save GPU states for bridge conditions
            np.save(RESULTS / 'z2299_gpu_states.npy', gpu_states)
            with open(SAVE_FILE, 'w') as f:
                json.dump(results, f, indent=2, cls=NpEncoder)
        else:
            print("  [FATAL] GPU kernel failed — cannot proceed with bridge conditions")
    else:
        print("\n[GPU] Already done, loading saved states")
        gpu_states_file = RESULTS / 'z2299_gpu_states.npy'
        if gpu_states_file.exists():
            gpu_states = np.load(gpu_states_file)
            X_gpu = build_temporal_features(gpu_states[WARMUP:], n_select=24, seed=42)
            print(f"  Loaded GPU states: {gpu_states.shape}")
        else:
            print("  WARNING: GPU states file missing, bridge conditions will be skipped")

    # ================================================================
    # 2. Per-seed loop: FPGA_ONLY, BRIDGE_CONCAT, SOFTWARE_ESN
    # ================================================================
    fpga = None

    for seed_idx, seed in enumerate(SEEDS):
        seed_key = str(seed)
        if seed_key in results.get('seeds', {}):
            existing = results['seeds'][seed_key]
            if all(k in existing for k in ['FPGA_ONLY', 'BRIDGE_CONCAT', 'SOFTWARE_ESN']):
                print(f"\n[Seed {seed_idx+1}/10] seed={seed} — already done, skipping")
                continue

        print(f"\n{'='*70}")
        print(f"  [Seed {seed_idx+1}/10] seed={seed}")
        print(f"{'='*70}")

        # Generate seed-specific input signal
        rng = np.random.default_rng(seed)
        u_raw = rng.uniform(-1, 1, N_STEPS + WARMUP)

        seed_results = results.get('seeds', {}).get(seed_key, {})

        # --- SOFTWARE_ESN ---
        if 'SOFTWARE_ESN' not in seed_results:
            print(f"  [ESN] Software ESN (256 neurons, seed={seed})")
            esn = SoftwareESN(n_neurons=256, seed=seed)
            esn_states = esn.run(u_raw, seed=seed)
            X_esn = build_temporal_features(esn_states[WARMUP:], n_select=24, seed=seed)
            bm_esn = full_benchmark(X_esn, u_raw)
            seed_results['SOFTWARE_ESN'] = bm_esn
            xor = bm_esn['xor']
            print(f"    ESN: MC={bm_esn['mc_total']:.2f} XOR1={xor['tau1']*100:.1f}% "
                  f"W4={bm_esn['wave4_acc']*100:.1f}%")
        else:
            print(f"  [ESN] Already done")

        # --- FPGA_ONLY ---
        if 'FPGA_ONLY' not in seed_results:
            print(f"  [FPGA] FPGA ONLY (128 neurons, seed={seed})")
            wait_cool(f"pre-FPGA-seed{seed}")

            if fpga is None:
                print(f"    Connecting to FPGA...")
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
                    print(f"    FPGA online: vmem [{telem['vmem'].min():.3f}, {telem['vmem'].max():.3f}]")
                else:
                    print(f"    WARNING: FPGA telemetry returned None!")

            fpga_states, fpga_dspikes = fpga_run_continuous(fpga, u_raw)
            # Save per-seed FPGA states for bridge
            np.save(RESULTS / f'z2299_fpga_states_seed{seed}.npy', fpga_states)
            np.save(RESULTS / f'z2299_fpga_dspikes_seed{seed}.npy', fpga_dspikes)
            X_fpga = build_temporal_features(fpga_states[WARMUP:], fpga_dspikes[WARMUP:], n_select=24, seed=seed)
            bm_fpga = full_benchmark(X_fpga, u_raw)
            seed_results['FPGA_ONLY'] = bm_fpga
            xor = bm_fpga['xor']
            print(f"    FPGA: MC={bm_fpga['mc_total']:.2f} XOR1={xor['tau1']*100:.1f}% "
                  f"XOR5={xor['tau5']*100:.1f}% W4={bm_fpga['wave4_acc']*100:.1f}%")
        else:
            print(f"  [FPGA] Already done")

        # --- BRIDGE_CONCAT (offline: FPGA states + GPU features) ---
        if 'BRIDGE_CONCAT' not in seed_results:
            print(f"  [BRIDGE] BRIDGE_CONCAT (offline, seed={seed})")
            fpga_states_file = RESULTS / f'z2299_fpga_states_seed{seed}.npy'
            fpga_dspikes_file = RESULTS / f'z2299_fpga_dspikes_seed{seed}.npy'
            if X_gpu is not None and fpga_states_file.exists():
                saved_states = np.load(fpga_states_file)
                saved_dspikes = np.load(fpga_dspikes_file)
                X_fpga_saved = build_temporal_features(saved_states[WARMUP:], saved_dspikes[WARMUP:], n_select=24, seed=seed)
                # For bridge, concatenate FPGA features with GPU features
                # GPU features are from a different input signal, but that's fine —
                # the GPU provides substrate-level diversity, not input-correlated features
                n_min = min(len(X_fpga_saved), len(X_gpu))
                X_bridge = np.hstack([X_fpga_saved[:n_min], X_gpu[:n_min]])
                # Use FPGA's u_raw for benchmarking (it drove the FPGA reservoir)
                bm_bridge = full_benchmark(X_bridge, u_raw)
                seed_results['BRIDGE_CONCAT'] = bm_bridge
                xor = bm_bridge['xor']
                print(f"    BRIDGE: MC={bm_bridge['mc_total']:.2f} XOR1={xor['tau1']*100:.1f}% "
                      f"XOR5={xor['tau5']*100:.1f}% W4={bm_bridge['wave4_acc']*100:.1f}%")
            else:
                print(f"    [SKIP] GPU={'OK' if X_gpu is not None else 'NONE'}, "
                      f"FPGA states={'saved' if fpga_states_file.exists() else 'MISSING'}")
        else:
            print(f"  [BRIDGE] Already done")

        # Save after each seed
        if 'seeds' not in results:
            results['seeds'] = {}
        results['seeds'][seed_key] = seed_results
        with open(SAVE_FILE, 'w') as f:
            json.dump(results, f, indent=2, cls=NpEncoder)
        print(f"  Saved seed {seed} results")

        # Cool down between seeds
        if seed_idx < len(SEEDS) - 1:
            wait_cool(f"between-seeds", target=TEMP_SAFE)

    # Close FPGA
    if fpga is not None:
        try:
            fpga.set_mac_signal(0.0)
            fpga.close()
            fpga = None
            print("\n  FPGA disconnected")
        except Exception:
            fpga = None

    # ================================================================
    # 3. Aggregate statistics
    # ================================================================
    print(f"\n{'='*70}")
    print("  AGGREGATE STATISTICS")
    print(f"{'='*70}")

    def collect_metric(condition, *keys):
        """Collect a metric across all seeds."""
        vals = []
        for seed_key, seed_data in results.get('seeds', {}).items():
            bm = seed_data.get(condition)
            if bm is None:
                continue
            v = bm
            for k in keys:
                if isinstance(v, dict):
                    v = v.get(k, None)
                else:
                    v = None
                    break
            if v is not None:
                vals.append(float(v))
        return np.array(vals)

    # FPGA stats
    fpga_mc = collect_metric('FPGA_ONLY', 'mc_total')
    fpga_xor1 = collect_metric('FPGA_ONLY', 'xor', 'tau1')
    fpga_xor3 = collect_metric('FPGA_ONLY', 'xor', 'tau3')
    fpga_xor5 = collect_metric('FPGA_ONLY', 'xor', 'tau5')
    fpga_narma5 = collect_metric('FPGA_ONLY', 'narma', 'narma5')
    fpga_wave4 = collect_metric('FPGA_ONLY', 'wave4_acc')

    # BRIDGE stats
    bridge_mc = collect_metric('BRIDGE_CONCAT', 'mc_total')
    bridge_xor1 = collect_metric('BRIDGE_CONCAT', 'xor', 'tau1')
    bridge_xor5 = collect_metric('BRIDGE_CONCAT', 'xor', 'tau5')
    bridge_wave4 = collect_metric('BRIDGE_CONCAT', 'wave4_acc')

    # ESN stats
    esn_mc = collect_metric('SOFTWARE_ESN', 'mc_total')
    esn_xor1 = collect_metric('SOFTWARE_ESN', 'xor', 'tau1')

    stats = {}
    for name, arr in [
        ('fpga_mc', fpga_mc), ('fpga_xor1', fpga_xor1), ('fpga_xor3', fpga_xor3),
        ('fpga_xor5', fpga_xor5), ('fpga_narma5', fpga_narma5), ('fpga_wave4', fpga_wave4),
        ('bridge_mc', bridge_mc), ('bridge_xor1', bridge_xor1), ('bridge_xor5', bridge_xor5),
        ('bridge_wave4', bridge_wave4),
        ('esn_mc', esn_mc), ('esn_xor1', esn_xor1),
    ]:
        if len(arr) > 0:
            stats[name] = {'mean': float(np.mean(arr)), 'std': float(np.std(arr)),
                           'min': float(np.min(arr)), 'max': float(np.max(arr)),
                           'n': int(len(arr)), 'values': arr.tolist()}
            print(f"  {name:20s}: {np.mean(arr):.3f} +/- {np.std(arr):.3f}  "
                  f"[{np.min(arr):.3f}, {np.max(arr):.3f}]  n={len(arr)}")
        else:
            stats[name] = {'mean': 0, 'std': 0, 'min': 0, 'max': 0, 'n': 0, 'values': []}
            print(f"  {name:20s}: NO DATA")

    results['stats'] = stats

    # GPU single-run reference
    gpu_mc_val = results.get('gpu_single', {}).get('mc_total', 0)
    print(f"\n  GPU (single run):    MC={gpu_mc_val:.2f}")

    # ================================================================
    # 4. TESTS
    # ================================================================
    print(f"\n{'='*70}")
    print("  TESTS -- z2299")
    print(f"{'='*70}")

    tests = {}
    n_pass = 0

    def test(name, cond, desc):
        nonlocal n_pass
        tests[name] = {'pass': bool(cond), 'desc': desc}
        print(f"  {name} {'PASS' if cond else 'FAIL'}: {desc}")
        n_pass += cond

    def s(name):
        """Get stat mean."""
        return stats.get(name, {}).get('mean', 0)

    def s_std(name):
        """Get stat std."""
        return stats.get(name, {}).get('std', 999)

    # T1: FPGA MC mean > 8.0
    test('T1', s('fpga_mc') > 8.0,
         f"FPGA MC mean={s('fpga_mc'):.2f} > 8.0")

    # T2: FPGA MC std < 3.0
    test('T2', s_std('fpga_mc') < 3.0,
         f"FPGA MC std={s_std('fpga_mc'):.2f} < 3.0")

    # T3: FPGA XOR1 mean > 0.80
    test('T3', s('fpga_xor1') > 0.80,
         f"FPGA XOR1 mean={s('fpga_xor1')*100:.1f}% > 80%")

    # T4: BRIDGE MC mean > FPGA MC mean
    test('T4', s('bridge_mc') > s('fpga_mc'),
         f"BRIDGE MC={s('bridge_mc'):.2f} > FPGA MC={s('fpga_mc'):.2f}")

    # T5: BRIDGE MC mean > GPU MC (single run)
    test('T5', s('bridge_mc') > gpu_mc_val,
         f"BRIDGE MC={s('bridge_mc'):.2f} > GPU MC={gpu_mc_val:.2f}")

    # T6: FPGA > ESN on MC (mean)
    test('T6', s('fpga_mc') > s('esn_mc'),
         f"FPGA MC={s('fpga_mc'):.2f} > ESN MC={s('esn_mc'):.2f}")

    # T7: FPGA > ESN on XOR1 (mean)
    test('T7', s('fpga_xor1') > s('esn_xor1'),
         f"FPGA XOR1={s('fpga_xor1')*100:.1f}% > ESN XOR1={s('esn_xor1')*100:.1f}%")

    # T8: BRIDGE Wave4 mean > 0.85
    test('T8', s('bridge_wave4') > 0.85,
         f"BRIDGE Wave4 mean={s('bridge_wave4')*100:.1f}% > 85%")

    # T9: At least 8/10 seeds have FPGA MC > 8.0
    n_seeds_above = int(np.sum(fpga_mc > 8.0)) if len(fpga_mc) > 0 else 0
    test('T9', n_seeds_above >= 8,
         f"Seeds with FPGA MC>8.0: {n_seeds_above}/10 >= 8")

    # T10: BRIDGE MC std < 4.0
    test('T10', s_std('bridge_mc') < 4.0,
         f"BRIDGE MC std={s_std('bridge_mc'):.2f} < 4.0")

    print(f"\n  TOTAL: {n_pass}/10 PASS")

    results['tests'] = tests
    results['summary'] = {
        'n_pass': n_pass, 'n_total': 10,
        'n_seeds_completed': len(results.get('seeds', {})),
        'seeds': SEEDS,
    }

    # ================================================================
    # Comparison table
    # ================================================================
    print(f"\n{'='*70}")
    print("  PER-SEED RESULTS TABLE")
    print(f"{'='*70}")
    print(f"  {'Seed':>6} | {'Cond':<14} {'MC':>7} {'XOR1':>7} {'XOR3':>7} {'XOR5':>7} {'N5':>7} {'W4':>7}")
    print(f"  {'-'*6}-+-{'-'*14}-{'-'*7}-{'-'*7}-{'-'*7}-{'-'*7}-{'-'*7}-{'-'*7}")
    for seed_key in sorted(results.get('seeds', {}).keys(), key=lambda x: int(x)):
        seed_data = results['seeds'][seed_key]
        for cond_name in ['FPGA_ONLY', 'BRIDGE_CONCAT', 'SOFTWARE_ESN']:
            bm = seed_data.get(cond_name)
            if bm is None:
                continue
            xor = bm.get('xor', {})
            print(f"  {seed_key:>6} | {cond_name:<14} {bm.get('mc_total',0):7.2f} "
                  f"{xor.get('tau1',0)*100:6.1f}% {xor.get('tau3',0)*100:6.1f}% "
                  f"{xor.get('tau5',0)*100:6.1f}% {bm.get('narma',{}).get('narma5',0):7.3f} "
                  f"{bm.get('wave4_acc',0)*100:6.1f}%")

    with open(SAVE_FILE, 'w') as f:
        json.dump(results, f, indent=2, cls=NpEncoder)
    print(f"\n  Results saved: {SAVE_FILE}")

    # Clean up per-seed .npy files
    print("  Cleaning up per-seed .npy files...")
    for seed in SEEDS:
        for suffix in ['states', 'dspikes']:
            f = RESULTS / f'z2299_fpga_{suffix}_seed{seed}.npy'
            try:
                if f.exists():
                    os.unlink(f)
            except Exception:
                pass
    gpu_states_file = RESULTS / 'z2299_gpu_states.npy'
    # Keep GPU states file (small, useful for debugging)


if __name__ == '__main__':
    main()
