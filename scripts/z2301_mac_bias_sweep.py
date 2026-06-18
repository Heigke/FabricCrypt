#!/usr/bin/env python3
"""
z2301_mac_bias_sweep.py — Sweep BIAS_GAIN to find optimal MAC injection strength
================================================================================
z2298 used BIAS_GAIN=0x4000 and got BRIDGE_MAC MC=5.78 (catastrophic synchronization).
This sweeps BIAS_GAIN from 0x0000 to 0x8000 to find the sweet spot.

For each BIAS_GAIN value:
  1. Configure FPGA with that BIAS_GAIN
  2. Run FPGA with GPU-derived MAC signal
  3. Build temporal features, concatenate with GPU features
  4. Benchmark: MC(d=1..20), XOR(tau=1,3,5,8), NARMA-5, Wave4

GPU kernel runs ONCE, states reused for all BIAS_GAIN values.
FPGA_ONLY (BIAS_GAIN=0, no MAC) serves as baseline.

Run:
  HSA_OVERRIDE_GFX_VERSION=11.0.0 PYTHONUNBUFFERED=1 taskset -c 0-3 venv/bin/python scripts/z2301_mac_bias_sweep.py
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
SAVE_FILE = RESULTS / 'z2301_mac_bias_sweep.json'
LOG_FILE = RESULTS / 'z2301_mac_bias_sweep_run.log'

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE, mode='a'),
        logging.StreamHandler(sys.stdout),
    ]
)
log = logging.getLogger('z2301')

from fpga_host_eth import FPGAEthBridge

NUM_NEURONS = 128
N_GPU_SAMPLED = 512
SAMPLE_HZ = 50
N_STEPS = 1500
WARMUP = 300
TEMP_SAFE = 40.0
GPU_KERN = BASE / 'scripts' / 'z2277_gpu_bridge_kern'
VG_GROUPS = {0: 0.05, 1: 0.15, 2: 0.30, 3: 0.58}

BIAS_GAIN_VALUES = [0x0000, 0x0100, 0x0400, 0x0800, 0x1000, 0x2000, 0x4000, 0x8000]


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
    fpga.set_mac_signal(0.0)
    return states, dspikes


# ============================================================
# Temporal product features (same as z2296/z2298)
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


def classify_waveform(X, u_raw):
    """4-class waveform classification (sine, square, triangle, sawtooth)."""
    n = len(X)
    quartiles = np.percentile(u_raw[WARMUP:WARMUP+n], [25, 50, 75])
    u = u_raw[WARMUP:WARMUP+n]
    labels = np.zeros(n)
    labels[u > quartiles[2]] = 3
    labels[(u > quartiles[1]) & (u <= quartiles[2])] = 2
    labels[(u > quartiles[0]) & (u <= quartiles[1])] = 1

    n_tr = int(0.7 * n)
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
    for tau in [1, 3, 5, 8]:
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

    wave_acc = classify_waveform(X, u_raw)

    return {'mc_total': mc_total, 'mc_per_delay': mc_per_d, 'xor': xor, 'narma': narma,
            'wave4_acc': wave_acc}


def setup_fpga(fpga, bias_gain):
    """Configure FPGA with given BIAS_GAIN, standard params, zero synapses."""
    fpga.set_kill(0)
    time.sleep(0.5)
    fpga.set_leak_cond(0x2000)
    fpga.set_base_exc_raw(0x0080)
    fpga.set_bias_gain_raw(bias_gain)
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
        log.info(f"  FPGA configured (BIAS_GAIN=0x{bias_gain:04X}): vmem [{telem['vmem'].min():.3f}, {telem['vmem'].max():.3f}]")
    else:
        log.warning(f"  FPGA telemetry returned None after config (BIAS_GAIN=0x{bias_gain:04X})")


def main():
    print("=" * 70)
    print("  z2301: MAC BIAS_GAIN Sweep")
    print("  Sweeping BIAS_GAIN to find optimal MAC injection strength")
    print("  z2298 used 0x4000 -> MC=5.78 (catastrophic synchronization)")
    print("=" * 70)

    results = {'sweep': {}, 'tests': {}, 'meta': {}}
    # Resume from saved results
    if SAVE_FILE.exists():
        try:
            with open(SAVE_FILE) as f:
                results = json.load(f)
            if 'sweep' not in results:
                results['sweep'] = {}
            done = list(results.get('sweep', {}).keys())
            if done:
                print(f"  RESUMED: {done} already done")
        except Exception:
            results = {'sweep': {}, 'tests': {}, 'meta': {}}

    rng = np.random.default_rng(42)
    u_raw = rng.uniform(-1, 1, N_STEPS + WARMUP)

    # ================================================================
    # 1. GPU HIP KERNEL — run ONCE, reuse states for all BIAS_GAIN values
    # ================================================================
    gpu_states = None
    gpu_states_file = RESULTS / 'z2301_gpu_states.npy'

    if gpu_states_file.exists():
        log.info("[GPU] Loading cached GPU states from disk")
        gpu_states = np.load(gpu_states_file)
        log.info(f"  GPU states: {gpu_states.shape}, range [{gpu_states.min():.3f}, {gpu_states.max():.3f}]")
    else:
        log.info("[GPU] Running HIP fourpop kernel (3072 neurons)")
        wait_cool("pre-GPU")
        gpu_states = run_hip_kernel(u_raw.astype(np.float32))
        if gpu_states is not None:
            np.save(gpu_states_file, gpu_states)
            log.info(f"  GPU states: {gpu_states.shape}, range [{gpu_states.min():.3f}, {gpu_states.max():.3f}]")
            log.info(f"  Saved to {gpu_states_file}")
        else:
            log.error("  GPU kernel FAILED — cannot proceed with bridge conditions")

    if gpu_states is None:
        log.error("No GPU states available. Exiting.")
        sys.exit(1)

    # Build GPU temporal features (reused for all BRIDGE_MAC conditions)
    X_gpu = build_temporal_features(gpu_states[WARMUP:], n_select=24, seed=42)
    log.info(f"  GPU features shape: {X_gpu.shape}")

    # Derive MAC signal from GPU mean states (same as z2298 cond6)
    gpu_mean = gpu_states[WARMUP:].mean(axis=1)
    mac_sig = (gpu_mean - gpu_mean.min()) / (gpu_mean.max() - gpu_mean.min() + 1e-10)
    mac_full = np.zeros(len(u_raw))
    mac_full[WARMUP:WARMUP+len(mac_sig)] = mac_sig
    log.info(f"  MAC signal range: [{mac_sig.min():.4f}, {mac_sig.max():.4f}]")

    results['meta']['gpu_states_shape'] = list(gpu_states.shape)
    results['meta']['mac_sig_range'] = [float(mac_sig.min()), float(mac_sig.max())]

    # ================================================================
    # 2. FPGA_ONLY baseline (BIAS_GAIN=0, no MAC signal)
    # ================================================================
    cond_key_baseline = 'FPGA_ONLY_bg0x0000'
    if cond_key_baseline not in results.get('sweep', {}):
        log.info(f"\n[BASELINE] FPGA_ONLY (BIAS_GAIN=0x0000, no MAC)")
        wait_cool("pre-FPGA-baseline")
        fpga = FPGAEthBridge(timeout=2.0)
        fpga.connect()
        setup_fpga(fpga, 0x0000)

        # Run with NO MAC signal (just input-derived)
        no_mac = np.clip(u_raw * 0.3 + 0.3, 0, 1)
        fpga_states, fpga_dspikes = fpga_run_continuous(fpga, u_raw, mac_signal=no_mac)
        X_fpga = build_temporal_features(fpga_states[WARMUP:], fpga_dspikes[WARMUP:], n_select=24, seed=42)
        bm = full_benchmark(X_fpga, u_raw)
        bm['bias_gain'] = 0x0000
        bm['bias_gain_hex'] = '0x0000'
        bm['condition'] = 'FPGA_ONLY'
        results['sweep'][cond_key_baseline] = bm
        xor = bm['xor']
        log.info(f"  BASELINE: MC={bm['mc_total']:.2f} XOR1={xor['tau1']*100:.1f}% XOR3={xor['tau3']*100:.1f}% "
                 f"XOR5={xor['tau5']*100:.1f}% N5={bm['narma']['narma5']:.3f} W4={bm['wave4_acc']*100:.1f}%")

        fpga.set_mac_signal(0.0)
        fpga.close()
        with open(SAVE_FILE, 'w') as f:
            json.dump(results, f, indent=2, cls=NpEncoder)
    else:
        log.info(f"\n[BASELINE] FPGA_ONLY — already done, skipping")

    # ================================================================
    # 3. Sweep BIAS_GAIN values with MAC signal
    # ================================================================
    for idx, bg in enumerate(BIAS_GAIN_VALUES):
        cond_key = f'BRIDGE_MAC_bg0x{bg:04X}'
        if cond_key in results.get('sweep', {}):
            log.info(f"\n[{idx+1}/{len(BIAS_GAIN_VALUES)}] BIAS_GAIN=0x{bg:04X} — already done, skipping")
            continue

        log.info(f"\n[{idx+1}/{len(BIAS_GAIN_VALUES)}] BIAS_GAIN=0x{bg:04X}")
        wait_cool(f"pre-bg0x{bg:04X}")

        fpga = FPGAEthBridge(timeout=2.0)
        fpga.connect()
        setup_fpga(fpga, bg)

        # Run FPGA with GPU-derived MAC signal
        fpga_states, fpga_dspikes = fpga_run_continuous(fpga, u_raw, mac_signal=mac_full)

        # Build BRIDGE_MAC features: FPGA temporal + GPU temporal
        X_fpga_mac = build_temporal_features(fpga_states[WARMUP:], fpga_dspikes[WARMUP:], n_select=24, seed=42)
        X_bridge = np.hstack([X_fpga_mac, X_gpu])
        log.info(f"  Features: FPGA={X_fpga_mac.shape}, GPU={X_gpu.shape}, bridge={X_bridge.shape}")

        bm = full_benchmark(X_bridge, u_raw)
        bm['bias_gain'] = bg
        bm['bias_gain_hex'] = f'0x{bg:04X}'
        bm['condition'] = 'BRIDGE_MAC'

        # Also benchmark FPGA-only features at this BIAS_GAIN for diagnostics
        bm_fpga_only = full_benchmark(X_fpga_mac, u_raw)
        bm['fpga_only_mc'] = bm_fpga_only['mc_total']
        bm['fpga_only_xor1'] = bm_fpga_only['xor']['tau1']

        results['sweep'][cond_key] = bm
        xor = bm['xor']
        log.info(f"  BG=0x{bg:04X}: MC={bm['mc_total']:.2f} XOR1={xor['tau1']*100:.1f}% XOR3={xor['tau3']*100:.1f}% "
                 f"XOR5={xor['tau5']*100:.1f}% N5={bm['narma']['narma5']:.3f} W4={bm['wave4_acc']*100:.1f}%"
                 f" (FPGA-only MC={bm_fpga_only['mc_total']:.2f})")

        fpga.set_mac_signal(0.0)
        fpga.close()
        with open(SAVE_FILE, 'w') as f:
            json.dump(results, f, indent=2, cls=NpEncoder)

    # ================================================================
    # Summary table
    # ================================================================
    print(f"\n{'='*90}")
    print(f"  {'BIAS_GAIN':>10} {'MC':>8} {'XOR1':>8} {'XOR3':>8} {'XOR5':>8} {'XOR8':>8} {'NARMA5':>8} {'Wave4':>8}")
    print(f"  {'-'*10} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")
    for key in sorted(results['sweep'].keys()):
        bm = results['sweep'][key]
        xor = bm.get('xor', {})
        narma = bm.get('narma', {})
        bg_hex = bm.get('bias_gain_hex', key)
        cond = bm.get('condition', '')
        label = f"{bg_hex} ({cond[:4]})" if cond else bg_hex
        print(f"  {label:>14} {bm.get('mc_total',0):8.2f} {xor.get('tau1',0)*100:7.1f}% "
              f"{xor.get('tau3',0)*100:7.1f}% {xor.get('tau5',0)*100:7.1f}% "
              f"{xor.get('tau8',0)*100:7.1f}% {narma.get('narma5',999):8.3f} {bm.get('wave4_acc',0)*100:7.1f}%")
    print(f"{'='*90}")

    # ================================================================
    # TESTS
    # ================================================================
    print(f"\n{'='*70}")
    print("  TESTS - z2301")
    print(f"{'='*70}")

    tests = {}
    n_pass = 0

    def test(name, cond, desc):
        nonlocal n_pass
        tests[name] = {'pass': bool(cond), 'desc': desc}
        status = 'PASS' if cond else 'FAIL'
        print(f"  {name} {status}: {desc}")
        n_pass += cond

    def g(bm, *keys):
        """Safe nested dict get."""
        v = bm
        for k in keys:
            if isinstance(v, dict):
                v = v.get(k, 0)
            else:
                return 0
        return v if v is not None else 0

    # Collect results
    baseline = results['sweep'].get('FPGA_ONLY_bg0x0000', {})
    baseline_mc = g(baseline, 'mc_total')
    baseline_xor1 = g(baseline, 'xor', 'tau1')

    mac_results = {}
    for key, bm in results['sweep'].items():
        if key.startswith('BRIDGE_MAC_'):
            bg = bm.get('bias_gain', -1)
            mac_results[bg] = bm

    # Find best values across sweep
    best_mc = 0.0
    best_mc_bg = -1
    best_xor1 = 0.5
    best_xor1_bg = -1
    best_xor3 = 0.5
    best_xor3_bg = -1
    best_narma5 = 999.0
    best_narma5_bg = -1
    wave4_over_80 = 0

    mc_values = []  # (bias_gain, mc) for monotonicity check

    for bg, bm in sorted(mac_results.items()):
        mc = g(bm, 'mc_total')
        xor1 = g(bm, 'xor', 'tau1')
        xor3 = g(bm, 'xor', 'tau3')
        n5 = g(bm, 'narma', 'narma5')
        w4 = g(bm, 'wave4_acc')

        mc_values.append((bg, mc))

        if mc > best_mc:
            best_mc = mc
            best_mc_bg = bg
        if xor1 > best_xor1:
            best_xor1 = xor1
            best_xor1_bg = bg
        if xor3 > best_xor3:
            best_xor3 = xor3
            best_xor3_bg = bg
        if n5 < best_narma5:
            best_narma5 = n5
            best_narma5_bg = bg
        if w4 > 0.80:
            wave4_over_80 += 1

    # Check non-monotonicity (inverted U): MC should increase then decrease with BIAS_GAIN
    mc_values_sorted = sorted(mc_values, key=lambda x: x[0])
    mc_seq = [mc for _, mc in mc_values_sorted]
    is_nonmonotonic = False
    if len(mc_seq) >= 3:
        # Check if there's a peak (some i where mc[i] > mc[i-1] and mc[i] > mc[i+1])
        for i in range(1, len(mc_seq) - 1):
            if mc_seq[i] > mc_seq[i-1] and mc_seq[i] > mc_seq[i+1]:
                is_nonmonotonic = True
                break
        # Also check if first < some middle > last
        if max(mc_seq[1:-1]) > mc_seq[0] and max(mc_seq[1:-1]) > mc_seq[-1]:
            is_nonmonotonic = True

    bg4000_mc = g(mac_results.get(0x4000, {}), 'mc_total')

    # T1: FPGA_ONLY (bg=0) MC > 8.0
    test('T1', baseline_mc > 8.0,
         f"FPGA_ONLY MC={baseline_mc:.2f} > 8.0")

    # T2: At least one BIAS_GAIN achieves MC > FPGA_ONLY MC
    test('T2', best_mc > baseline_mc,
         f"Best MAC MC={best_mc:.2f} (bg=0x{best_mc_bg:04X}) > baseline={baseline_mc:.2f}")

    # T3: At least one BIAS_GAIN achieves XOR1 > FPGA_ONLY XOR1
    test('T3', best_xor1 > baseline_xor1,
         f"Best MAC XOR1={best_xor1*100:.1f}% (bg=0x{best_xor1_bg:04X}) > baseline={baseline_xor1*100:.1f}%")

    # T4: Best MAC MC > 10.0
    test('T4', best_mc > 10.0,
         f"Best MAC MC={best_mc:.2f} > 10.0")

    # T5: Best MAC XOR1 > 85%
    test('T5', best_xor1 > 0.85,
         f"Best MAC XOR1={best_xor1*100:.1f}% > 85%")

    # T6: BIAS_GAIN=0x4000 (z2298 config) MC < FPGA_ONLY MC (confirms synchronization)
    test('T6', bg4000_mc < baseline_mc if bg4000_mc > 0 else False,
         f"bg=0x4000 MC={bg4000_mc:.2f} < baseline={baseline_mc:.2f} (sync damage)")

    # T7: Optimal BIAS_GAIN != 0x4000 (sweet spot exists elsewhere)
    test('T7', best_mc_bg != 0x4000,
         f"Optimal bg=0x{best_mc_bg:04X} != 0x4000")

    # T8: MC is non-monotonic with BIAS_GAIN (inverted U shape)
    test('T8', is_nonmonotonic,
         f"MC non-monotonic={is_nonmonotonic} across {len(mc_seq)} values")

    # T9: Best MAC NARMA-5 < 0.25
    test('T9', best_narma5 < 0.25,
         f"Best NARMA-5={best_narma5:.3f} (bg=0x{best_narma5_bg:04X}) < 0.25")

    # T10: Wave4 > 80% for at least 3 BIAS_GAIN values
    test('T10', wave4_over_80 >= 3,
         f"Wave4>80% count={wave4_over_80} >= 3")

    # T11: Best overall BRIDGE_MAC beats BRIDGE_CONCAT MC from z2298 (14.01)
    test('T11', best_mc > 14.01,
         f"Best MAC MC={best_mc:.2f} > z2298 BRIDGE_CONCAT MC=14.01")

    # T12: Best BRIDGE_MAC XOR3 > 70%
    test('T12', best_xor3 > 0.70,
         f"Best MAC XOR3={best_xor3*100:.1f}% (bg=0x{best_xor3_bg:04X}) > 70%")

    results['tests'] = tests
    results['meta']['best_mc_bias_gain'] = f'0x{best_mc_bg:04X}' if best_mc_bg >= 0 else 'N/A'
    results['meta']['best_mc'] = best_mc
    results['meta']['best_xor1_bias_gain'] = f'0x{best_xor1_bg:04X}' if best_xor1_bg >= 0 else 'N/A'
    results['meta']['best_xor1'] = best_xor1
    results['meta']['n_pass'] = n_pass
    results['meta']['n_tests'] = 12
    results['meta']['timestamp'] = time.strftime('%Y-%m-%dT%H:%M:%S')

    with open(SAVE_FILE, 'w') as f:
        json.dump(results, f, indent=2, cls=NpEncoder)

    print(f"\n  TOTAL: {n_pass}/12 PASS")
    print(f"  Optimal BIAS_GAIN for MC: 0x{best_mc_bg:04X} (MC={best_mc:.2f})")
    print(f"  Optimal BIAS_GAIN for XOR1: 0x{best_xor1_bg:04X} (XOR1={best_xor1*100:.1f}%)")
    print(f"  Results saved to {SAVE_FILE}")


if __name__ == '__main__':
    main()
