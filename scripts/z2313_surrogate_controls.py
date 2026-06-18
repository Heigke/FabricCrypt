#!/usr/bin/env python3
"""
z2313_surrogate_controls.py — Surrogate Noise Controls for Causal Evidence
===========================================================================
Generates 5 types of surrogate noise matched to GPU characteristics, plus
real GPU noise, to test whether specific GPU dynamics (not just spectral
properties) matter for cross-substrate reservoir computing.

Surrogate conditions:
  1) WHITE:           Gaussian, matched mean/var to real GPU thermal
  2) IAAFT:           Preserves spectrum + amplitude distribution, destroys phase
  3) PHASE_RANDOM:    FFT phase-randomized, preserves spectrum only
  4) SHUFFLED:        Random permutation, preserves histogram only
  5) BLOCK_SHUFFLED:  50-sample block shuffle, preserves local structure
  6) REAL_GPU:        Actual hwmon7 thermal readings during FPGA operation

Benchmarks per condition: MC(d=1..20), XOR(tau=1,3,5), Wave4, NARMA-5
3 repetitions per condition.

Tests (15):
  T869: Real GPU MC > White noise MC
  T870: Real GPU MC > IAAFT MC (phase relationships matter)
  T871: Real GPU MC > Shuffled MC
  T872: Real GPU XOR5 > White noise XOR5
  T873: Real GPU XOR5 > IAAFT XOR5
  T874: Real GPU Wave4 > White noise Wave4
  T875: Real GPU Wave4 > IAAFT Wave4
  T876: IAAFT MC > White noise MC (spectrum helps even without phase)
  T877: Block-shuffled MC > Shuffled MC (local structure has value)
  T878: Phase-randomized PSD matches Real GPU PSD within 5%
  T879: IAAFT histogram matches Real GPU histogram (KS p > 0.05)
  T880: Real GPU > IAAFT by >= 5% on >= 2 of 4 metrics
  T881: Shuffled MC < 50% of Real GPU MC
  T882: Real GPU NARMA < White noise NARMA (NRMSE, lower is better)
  T883: Cohen's d for Real vs IAAFT > 0.5 on >= 1 metric

Run:
  PYTHONUNBUFFERED=1 venv/bin/python scripts/z2313_surrogate_controls.py
"""

import os, sys, time, json, struct
import numpy as np
from pathlib import Path
from scipy import stats as sp_stats

os.environ['PYTHONUNBUFFERED'] = '1'

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE / 'scripts'))
RESULTS = BASE / 'results'
RESULTS.mkdir(exist_ok=True)
SAVE_FILE = RESULTS / 'z2313_surrogate_controls.json'
STATES_FILE = RESULTS / 'z2313_fpga_states.npy'

from fpga_host_eth import FPGAEthBridge

NUM_NEURONS = 128
SAMPLE_HZ = 50
N_STEPS = 2000
WARMUP = 300
N_NOISE_SAMPLES = 2000
N_REPS = 3
TEMP_PAUSE = 60.0
TEMP_RESUME = 42.0
TEMP_SAFE = 42.0
VG_GROUPS = {0: 0.05, 1: 0.15, 2: 0.30, 3: 0.58}
BLOCK_SIZE = 50  # for block-shuffled surrogate

CONDITIONS = ['REAL_GPU', 'WHITE', 'IAAFT', 'PHASE_RANDOM', 'SHUFFLED', 'BLOCK_SHUFFLED']


# ============================================================
# Thermal helpers
# ============================================================
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
# Surrogate generation
# ============================================================
def iaaft(signal, n_iterations=100):
    """Iterative Amplitude Adjusted Fourier Transform surrogate."""
    sorted_signal = np.sort(signal)
    # Initialize with shuffled original
    surrogate = signal.copy()
    np.random.shuffle(surrogate)
    for _ in range(n_iterations):
        # Step 1: Match spectrum
        s_fft = np.fft.rfft(surrogate)
        s_fft = np.abs(np.fft.rfft(signal)) * np.exp(1j * np.angle(s_fft))
        surrogate = np.fft.irfft(s_fft, n=len(signal))
        # Step 2: Match amplitude distribution
        ranks = np.argsort(np.argsort(surrogate))
        surrogate = sorted_signal[ranks]
    return surrogate


def phase_randomize(signal, rng):
    """FFT phase-randomization: preserves power spectrum, randomizes phases."""
    fft_vals = np.fft.rfft(signal)
    amplitudes = np.abs(fft_vals)
    random_phases = rng.uniform(0, 2 * np.pi, len(fft_vals))
    # Keep DC and Nyquist real
    random_phases[0] = 0
    if len(signal) % 2 == 0:
        random_phases[-1] = 0
    new_fft = amplitudes * np.exp(1j * random_phases)
    return np.fft.irfft(new_fft, n=len(signal))


def block_shuffle(signal, block_size, rng):
    """Shuffle signal in blocks of block_size samples."""
    n = len(signal)
    n_blocks = n // block_size
    blocks = [signal[i*block_size:(i+1)*block_size] for i in range(n_blocks)]
    # Include remainder
    remainder = signal[n_blocks*block_size:]
    perm = rng.permutation(n_blocks)
    shuffled = np.concatenate([blocks[i] for i in perm])
    if len(remainder) > 0:
        shuffled = np.concatenate([shuffled, remainder])
    return shuffled


def generate_surrogates(real_gpu_noise, rng):
    """Generate all 5 surrogate types from real GPU noise."""
    mu = np.mean(real_gpu_noise)
    sigma = np.std(real_gpu_noise)
    n = len(real_gpu_noise)

    surrogates = {}

    # 1. White noise: matched mean/var
    surrogates['WHITE'] = rng.normal(mu, sigma, n)

    # 2. IAAFT: preserves spectrum + amplitude distribution
    surrogates['IAAFT'] = iaaft(real_gpu_noise, n_iterations=100)

    # 3. Phase-randomized: preserves spectrum, not amplitude distribution
    surrogates['PHASE_RANDOM'] = phase_randomize(real_gpu_noise, rng)

    # 4. Shuffled: random permutation
    shuffled = real_gpu_noise.copy()
    rng.shuffle(shuffled)
    surrogates['SHUFFLED'] = shuffled

    # 5. Block-shuffled: 50-sample blocks
    surrogates['BLOCK_SHUFFLED'] = block_shuffle(real_gpu_noise, BLOCK_SIZE, rng)

    return surrogates


def noise_to_base_exc(noise_signal):
    """Convert noise signal to BASE_EXC uint16 values centered on 0x0080.

    Scales noise to modulate BASE_EXC around the nominal value 0x0080 (128).
    Range: [0x0020, 0x00E0] = [32, 224] centered on 128.
    """
    # Normalize to [-1, 1]
    mn, mx = noise_signal.min(), noise_signal.max()
    rng_val = mx - mn
    if rng_val < 1e-10:
        return np.full(len(noise_signal), 0x0080, dtype=np.uint16)
    normalized = 2.0 * (noise_signal - mn) / rng_val - 1.0
    # Map to [0x0020, 0x00E0] centered on 0x0080
    base_exc = 0x0080 + (normalized * 0x0060).astype(int)
    base_exc = np.clip(base_exc, 0x0020, 0x00E0).astype(np.uint16)
    return base_exc


def compute_noise_stats(signal, label=""):
    """Compute PSD slope, ACF(1), histogram stats."""
    from scipy.signal import welch
    fs = SAMPLE_HZ
    freqs, psd = welch(signal, fs=fs, nperseg=min(256, len(signal)))
    # PSD slope (log-log fit excluding DC)
    mask = freqs > 0
    if np.sum(mask) > 2:
        log_f = np.log10(freqs[mask])
        log_p = np.log10(psd[mask] + 1e-30)
        slope = np.polyfit(log_f, log_p, 1)[0]
    else:
        slope = 0.0
    # ACF(1)
    if len(signal) > 1:
        acf1 = np.corrcoef(signal[:-1], signal[1:])[0, 1]
    else:
        acf1 = 0.0
    return {
        'mean': float(np.mean(signal)),
        'std': float(np.std(signal)),
        'psd_slope': float(slope),
        'acf1': float(acf1),
        'min': float(np.min(signal)),
        'max': float(np.max(signal)),
        'psd_freqs': freqs.tolist(),
        'psd_values': psd.tolist(),
    }


# ============================================================
# Collect real GPU thermal noise
# ============================================================
def collect_gpu_noise(n_samples):
    """Collect real GPU thermal noise from hwmon7 at ~20Hz."""
    print(f"  Collecting {n_samples} real GPU thermal samples at ~20Hz...", flush=True)
    noise = []
    dt = 1.0 / 20.0
    for i in range(n_samples):
        try:
            with open('/sys/class/hwmon/hwmon7/temp1_input', 'r') as f:
                temp_mc = float(f.read().strip())
            noise.append(temp_mc / 1000.0)  # millicelsius -> celsius
        except Exception:
            if noise:
                noise.append(noise[-1])
            else:
                noise.append(50.0)
        time.sleep(dt)
        if (i + 1) % 500 == 0:
            print(f"    {i+1}/{n_samples} samples collected", flush=True)
    return np.array(noise, dtype=np.float64)


# ============================================================
# FPGA setup and run
# ============================================================
def setup_fpga(fpga):
    """Configure FPGA with standard parameters."""
    fpga.set_leak_cond(0x2000)
    fpga.set_base_exc_raw(0x0080)
    fpga.set_bias_gain_raw(0x4000)
    fpga.set_threshold_raw(0x20000)
    time.sleep(0.05)
    # Set Vg groups
    for n in range(NUM_NEURONS):
        fpga.set_vg(n, VG_GROUPS[n % 4])
        time.sleep(0.001)
    # Clear synapses
    for n in range(NUM_NEURONS):
        fpga.set_synapse(n, 0x00000000)
        time.sleep(0.001)
    time.sleep(0.1)
    print("  FPGA configured: LEAK=0x2000 THRESH=0x20000 BASE_EXC=0x0080 BIAS_GAIN=0x4000")


def fpga_run_with_noise(fpga, u_input, noise_base_exc):
    """Run FPGA with time-varying BASE_EXC (noise injection) and MAC input.

    noise_base_exc: uint16 array of BASE_EXC values per step.
    u_input: float array, the reservoir input signal driving MAC.
    """
    n_steps = len(u_input)
    mac_signal = np.clip(u_input * 0.3 + 0.3, 0, 1)
    states = np.zeros((n_steps, NUM_NEURONS))
    dspikes = np.zeros((n_steps, NUM_NEURONS), dtype=np.float32)
    dt = 1.0 / SAMPLE_HZ

    fpga.set_mac_signal(0.0)
    time.sleep(0.02)
    telem = fpga.read_telemetry()
    prev_sc = telem['spike_counts'].copy() if telem is not None else np.zeros(NUM_NEURONS, dtype=np.uint16)

    for t in range(n_steps):
        # Thermal check every 50 steps
        if t > 0 and t % 50 == 0:
            temp = get_max_temp()
            if temp > 75.0:
                fpga.set_mac_signal(0.0)
                print(f"\n  [THERMAL PAUSE] {temp:.0f}C at step {t}/{n_steps}", end="", flush=True)
                while temp > 50.0:
                    time.sleep(5)
                    temp = get_max_temp()
                    print(f" {temp:.0f}", end="", flush=True)
                print(" resumed", flush=True)

        # Inject noise as BASE_EXC modulation
        fpga.set_base_exc_raw(int(noise_base_exc[t % len(noise_base_exc)]))
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
    fpga.set_base_exc_raw(0x0080)  # Restore default
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
    """4-class waveform classification."""
    n = len(X)
    n_tr = int(0.7 * n)
    quartiles = np.percentile(u_raw[WARMUP:WARMUP+n], [25, 50, 75])
    u = u_raw[WARMUP:WARMUP+n]
    labels = np.zeros(n)
    labels[u > quartiles[2]] = 3
    labels[(u > quartiles[1]) & (u <= quartiles[2])] = 2
    labels[(u > quartiles[0]) & (u <= quartiles[1])] = 1

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
    """Run full benchmark battery: MC, XOR, NARMA-5, Wave4."""
    n = len(X)
    n_tr = int(0.7 * n)

    # Memory Capacity
    mc_total = 0.0
    mc_per_d = {}
    for d in range(1, 21):
        target = u_raw[WARMUP-d:len(u_raw)-d]
        nn = min(n, len(target))
        r2 = ridge_solve(X[:n_tr], target[:n_tr], X[n_tr:nn], target[n_tr:nn])
        mc_per_d[str(d)] = r2
        mc_total += r2

    # XOR at various tau
    xor = {}
    for tau in [1, 3, 5]:
        u_a = (u_raw[WARMUP:] > 0).astype(float)
        u_b = (u_raw[WARMUP-tau:len(u_raw)-tau] > 0).astype(float)
        nn = min(len(u_a), len(u_b), n)
        target = (u_a[:nn] != u_b[:nn]).astype(float)
        Xn = X[:nn]
        acc = ridge_solve(Xn[:n_tr], target[:n_tr], Xn[n_tr:nn], target[n_tr:nn], 'classification')
        xor[f'tau{tau}'] = acc

    # NARMA-5
    T = len(u_raw)
    order = 5
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
    narma5 = best_nrmse

    # 4-class waveform
    wave4 = classify_waveform(X, u_raw)

    return {
        'mc_total': mc_total,
        'mc_per_delay': mc_per_d,
        'xor': xor,
        'narma5': narma5,
        'wave4_acc': wave4,
    }


# ============================================================
# Tests
# ============================================================
def mean_metric(results, cond, metric_path):
    """Extract mean of a metric across repetitions."""
    vals = []
    for rep_key, rep_data in results['conditions'].get(cond, {}).items():
        if not rep_key.startswith('rep'):
            continue
        bm = rep_data.get('benchmark', {})
        parts = metric_path.split('.')
        v = bm
        for p in parts:
            if isinstance(v, dict):
                v = v.get(p)
            else:
                v = None
                break
        if v is not None:
            vals.append(float(v))
    return np.mean(vals) if vals else None


def all_metric_vals(results, cond, metric_path):
    """Extract all rep values of a metric."""
    vals = []
    for rep_key, rep_data in results['conditions'].get(cond, {}).items():
        if not rep_key.startswith('rep'):
            continue
        bm = rep_data.get('benchmark', {})
        parts = metric_path.split('.')
        v = bm
        for p in parts:
            if isinstance(v, dict):
                v = v.get(p)
            else:
                v = None
                break
        if v is not None:
            vals.append(float(v))
    return vals


def cohens_d(vals_a, vals_b):
    """Compute Cohen's d effect size."""
    if len(vals_a) < 2 or len(vals_b) < 2:
        return 0.0
    na, nb = len(vals_a), len(vals_b)
    ma, mb = np.mean(vals_a), np.mean(vals_b)
    sa, sb = np.std(vals_a, ddof=1), np.std(vals_b, ddof=1)
    pooled_std = np.sqrt(((na-1)*sa**2 + (nb-1)*sb**2) / (na+nb-2))
    if pooled_std < 1e-10:
        return 0.0
    return (ma - mb) / pooled_std


def run_tests(results):
    """Evaluate all 15 tests."""
    tests = {}

    real_mc = mean_metric(results, 'REAL_GPU', 'mc_total')
    white_mc = mean_metric(results, 'WHITE', 'mc_total')
    iaaft_mc = mean_metric(results, 'IAAFT', 'mc_total')
    shuffled_mc = mean_metric(results, 'SHUFFLED', 'mc_total')
    block_mc = mean_metric(results, 'BLOCK_SHUFFLED', 'mc_total')

    real_xor5 = mean_metric(results, 'REAL_GPU', 'xor.tau5')
    white_xor5 = mean_metric(results, 'WHITE', 'xor.tau5')
    iaaft_xor5 = mean_metric(results, 'IAAFT', 'xor.tau5')

    real_wave4 = mean_metric(results, 'REAL_GPU', 'wave4_acc')
    white_wave4 = mean_metric(results, 'WHITE', 'wave4_acc')
    iaaft_wave4 = mean_metric(results, 'IAAFT', 'wave4_acc')

    real_narma = mean_metric(results, 'REAL_GPU', 'narma5')
    white_narma = mean_metric(results, 'WHITE', 'narma5')

    def safe_gt(a, b):
        if a is None or b is None:
            return False
        return a > b

    def safe_lt(a, b):
        if a is None or b is None:
            return False
        return a < b

    # T869-T871: Real GPU MC comparisons
    tests['T869'] = {'desc': 'Real GPU MC > White MC', 'pass': safe_gt(real_mc, white_mc),
                     'real': real_mc, 'white': white_mc}
    tests['T870'] = {'desc': 'Real GPU MC > IAAFT MC', 'pass': safe_gt(real_mc, iaaft_mc),
                     'real': real_mc, 'iaaft': iaaft_mc}
    tests['T871'] = {'desc': 'Real GPU MC > Shuffled MC', 'pass': safe_gt(real_mc, shuffled_mc),
                     'real': real_mc, 'shuffled': shuffled_mc}

    # T872-T873: XOR5
    tests['T872'] = {'desc': 'Real GPU XOR5 > White XOR5', 'pass': safe_gt(real_xor5, white_xor5),
                     'real': real_xor5, 'white': white_xor5}
    tests['T873'] = {'desc': 'Real GPU XOR5 > IAAFT XOR5', 'pass': safe_gt(real_xor5, iaaft_xor5),
                     'real': real_xor5, 'iaaft': iaaft_xor5}

    # T874-T875: Wave4
    tests['T874'] = {'desc': 'Real GPU Wave4 > White Wave4', 'pass': safe_gt(real_wave4, white_wave4),
                     'real': real_wave4, 'white': white_wave4}
    tests['T875'] = {'desc': 'Real GPU Wave4 > IAAFT Wave4', 'pass': safe_gt(real_wave4, iaaft_wave4),
                     'real': real_wave4, 'iaaft': iaaft_wave4}

    # T876: IAAFT MC > White MC
    tests['T876'] = {'desc': 'IAAFT MC > White MC', 'pass': safe_gt(iaaft_mc, white_mc),
                     'iaaft': iaaft_mc, 'white': white_mc}

    # T877: Block-shuffled MC > Shuffled MC
    tests['T877'] = {'desc': 'Block-shuffled MC > Shuffled MC', 'pass': safe_gt(block_mc, shuffled_mc),
                     'block': block_mc, 'shuffled': shuffled_mc}

    # T878: Phase-randomized PSD matches Real GPU PSD within 5%
    real_stats = results.get('noise_stats', {}).get('REAL_GPU', {})
    phase_stats = results.get('noise_stats', {}).get('PHASE_RANDOM', {})
    psd_match = False
    if real_stats and phase_stats:
        real_psd = np.array(real_stats.get('psd_values', []))
        phase_psd = np.array(phase_stats.get('psd_values', []))
        if len(real_psd) == len(phase_psd) and len(real_psd) > 0:
            # Relative error on total power
            real_power = np.sum(real_psd)
            phase_power = np.sum(phase_psd)
            if real_power > 1e-10:
                rel_err = abs(phase_power - real_power) / real_power
                psd_match = rel_err < 0.05
                tests['T878'] = {'desc': 'Phase-random PSD matches Real PSD (5%)',
                                 'pass': psd_match, 'rel_error': float(rel_err)}
            else:
                tests['T878'] = {'desc': 'Phase-random PSD matches Real PSD (5%)',
                                 'pass': False, 'note': 'real power ~0'}
        else:
            tests['T878'] = {'desc': 'Phase-random PSD matches Real PSD (5%)',
                             'pass': False, 'note': 'PSD length mismatch'}
    else:
        tests['T878'] = {'desc': 'Phase-random PSD matches Real PSD (5%)',
                         'pass': False, 'note': 'missing stats'}

    # T879: IAAFT histogram matches Real GPU (KS test p > 0.05)
    real_noise = np.array(results.get('raw_noise', {}).get('REAL_GPU', []))
    iaaft_noise = np.array(results.get('raw_noise', {}).get('IAAFT', []))
    if len(real_noise) > 10 and len(iaaft_noise) > 10:
        ks_stat, ks_p = sp_stats.ks_2samp(real_noise, iaaft_noise)
        tests['T879'] = {'desc': 'IAAFT histogram ~ Real GPU (KS p>0.05)',
                         'pass': ks_p > 0.05, 'ks_stat': float(ks_stat), 'ks_p': float(ks_p)}
    else:
        tests['T879'] = {'desc': 'IAAFT histogram ~ Real GPU (KS p>0.05)',
                         'pass': False, 'note': 'insufficient noise data'}

    # T880: Real > IAAFT by >= 5% on >= 2 of 4 metrics
    metrics_better = 0
    for m_real, m_iaaft, higher_better in [
        (real_mc, iaaft_mc, True), (real_xor5, iaaft_xor5, True),
        (real_wave4, iaaft_wave4, True), (real_narma, mean_metric(results, 'IAAFT', 'narma5'), False)
    ]:
        if m_real is not None and m_iaaft is not None:
            if higher_better:
                if m_iaaft > 1e-10 and (m_real - m_iaaft) / abs(m_iaaft) >= 0.05:
                    metrics_better += 1
            else:
                if m_real > 1e-10 and (m_iaaft - m_real) / abs(m_real) >= 0.05:
                    metrics_better += 1
    tests['T880'] = {'desc': 'Real > IAAFT by >=5% on >=2 metrics',
                     'pass': metrics_better >= 2, 'n_better': metrics_better}

    # T881: Shuffled MC < 50% of Real GPU MC
    if real_mc is not None and shuffled_mc is not None and real_mc > 1e-10:
        ratio = shuffled_mc / real_mc
        tests['T881'] = {'desc': 'Shuffled MC < 50% of Real MC',
                         'pass': ratio < 0.5, 'ratio': float(ratio)}
    else:
        tests['T881'] = {'desc': 'Shuffled MC < 50% of Real MC', 'pass': False, 'note': 'missing data'}

    # T882: Real NARMA < White NARMA (lower NRMSE is better)
    tests['T882'] = {'desc': 'Real NARMA5 < White NARMA5 (NRMSE)',
                     'pass': safe_lt(real_narma, white_narma),
                     'real': real_narma, 'white': white_narma}

    # T883: Cohen's d Real vs IAAFT > 0.5 on at least 1 metric
    any_large_d = False
    d_values = {}
    for metric_path, label in [('mc_total', 'MC'), ('xor.tau5', 'XOR5'),
                                ('wave4_acc', 'Wave4'), ('narma5', 'NARMA5')]:
        vals_real = all_metric_vals(results, 'REAL_GPU', metric_path)
        vals_iaaft = all_metric_vals(results, 'IAAFT', metric_path)
        d = cohens_d(vals_real, vals_iaaft)
        d_values[label] = float(d)
        if abs(d) > 0.5:
            any_large_d = True
    tests['T883'] = {'desc': "Cohen's d Real vs IAAFT > 0.5 on >=1 metric",
                     'pass': any_large_d, 'cohens_d': d_values}

    return tests


# ============================================================
# Main
# ============================================================
def main():
    print("=" * 70)
    print("  z2313: Surrogate Noise Controls for Causal Evidence")
    print("  6 conditions x 3 reps = 18 FPGA runs")
    print("=" * 70)

    results = {'conditions': {}, 'noise_stats': {}, 'raw_noise': {}, 'tests': {}}
    # Resume from saved results
    if SAVE_FILE.exists():
        try:
            with open(SAVE_FILE) as f:
                results = json.load(f)
            done = list(results.get('conditions', {}).keys())
            if done:
                print(f"  RESUMED: {done} already done")
        except Exception:
            results = {'conditions': {}, 'noise_stats': {}, 'raw_noise': {}, 'tests': {}}

    rng = np.random.default_rng(42)
    u_raw = rng.uniform(-1, 1, N_STEPS + WARMUP)  # 2300 total

    all_states = {}

    # ================================================================
    # Step 1: Collect real GPU thermal noise
    # ================================================================
    if 'REAL_GPU' not in results.get('raw_noise', {}):
        wait_cool("pre-collection")
        real_gpu_noise = collect_gpu_noise(N_NOISE_SAMPLES)
        results.setdefault('raw_noise', {})['REAL_GPU'] = real_gpu_noise.tolist()
        results.setdefault('noise_stats', {})['REAL_GPU'] = compute_noise_stats(real_gpu_noise, 'REAL_GPU')
        print(f"  Real GPU noise: mean={np.mean(real_gpu_noise):.2f}C "
              f"std={np.std(real_gpu_noise):.3f}C "
              f"ACF(1)={results['noise_stats']['REAL_GPU']['acf1']:.3f} "
              f"PSD_slope={results['noise_stats']['REAL_GPU']['psd_slope']:.2f}")
        with open(SAVE_FILE, 'w') as f:
            json.dump(results, f, indent=2, cls=NpEncoder)
    else:
        real_gpu_noise = np.array(results['raw_noise']['REAL_GPU'])
        print(f"  Real GPU noise loaded from cache ({len(real_gpu_noise)} samples)")

    # ================================================================
    # Step 2: Generate surrogates
    # ================================================================
    surrogates = generate_surrogates(real_gpu_noise, rng)
    surrogates['REAL_GPU'] = real_gpu_noise

    # Compute and save noise stats for all conditions
    for cond_name, noise in surrogates.items():
        if cond_name not in results.get('noise_stats', {}):
            results.setdefault('noise_stats', {})[cond_name] = compute_noise_stats(noise, cond_name)
        if cond_name not in results.get('raw_noise', {}):
            results.setdefault('raw_noise', {})[cond_name] = noise.tolist()

    print("\n  Noise characteristics:")
    print(f"  {'Condition':<18} {'Mean':>8} {'Std':>8} {'ACF(1)':>8} {'PSD slope':>10}")
    print(f"  {'-'*52}")
    for cond_name in CONDITIONS:
        s = results['noise_stats'].get(cond_name, {})
        print(f"  {cond_name:<18} {s.get('mean',0):8.2f} {s.get('std',0):8.3f} "
              f"{s.get('acf1',0):8.3f} {s.get('psd_slope',0):10.2f}")

    with open(SAVE_FILE, 'w') as f:
        json.dump(results, f, indent=2, cls=NpEncoder)

    # ================================================================
    # Step 3: Connect FPGA and run conditions
    # ================================================================
    print("\n  Connecting to FPGA...", flush=True)
    fpga = FPGAEthBridge(timeout=2.0)
    fpga.connect()
    fpga.set_kill(0)
    time.sleep(1.0)
    print(f"  FPGA connected")
    setup_fpga(fpga)

    for ci, cond_name in enumerate(CONDITIONS):
        cond_results = results.setdefault('conditions', {}).setdefault(cond_name, {})

        # Check how many reps are done
        done_reps = [k for k in cond_results if k.startswith('rep')]
        if len(done_reps) >= N_REPS:
            print(f"\n[{ci+1}/{len(CONDITIONS)}] {cond_name} -- all {N_REPS} reps done, skipping")
            continue

        noise_signal = surrogates[cond_name]
        noise_base_exc = noise_to_base_exc(noise_signal)

        print(f"\n[{ci+1}/{len(CONDITIONS)}] {cond_name} ({len(done_reps)}/{N_REPS} reps done)")

        for rep in range(N_REPS):
            rep_key = f'rep{rep}'
            if rep_key in cond_results:
                print(f"  rep{rep}: already done, skipping")
                continue

            try:
                wait_cool(f"{cond_name} rep{rep}")

                # For REAL_GPU, re-collect fresh noise for each rep (different temporal realization)
                if cond_name == 'REAL_GPU' and rep > 0:
                    print(f"  rep{rep}: collecting fresh GPU noise...", flush=True)
                    fresh_noise = collect_gpu_noise(N_NOISE_SAMPLES)
                    noise_base_exc = noise_to_base_exc(fresh_noise)

                print(f"  rep{rep}: running FPGA ({N_STEPS} steps)...", end="", flush=True)
                t0 = time.time()

                # Reset FPGA params before each run
                setup_fpga(fpga)
                time.sleep(0.1)

                states, dspikes = fpga_run_with_noise(fpga, u_raw[:N_STEPS+WARMUP], noise_base_exc)
                elapsed = time.time() - t0
                print(f" {elapsed:.0f}s", flush=True)

                # Build features and benchmark
                X = build_temporal_features(states[WARMUP:], dspikes[WARMUP:], n_select=24, seed=42)
                bm = full_benchmark(X, u_raw)

                cond_results[rep_key] = {
                    'benchmark': bm,
                    'elapsed_s': elapsed,
                }

                # Store states for first rep only (save disk)
                if rep == 0:
                    all_states[cond_name] = states

                xor = bm['xor']
                print(f"  rep{rep}: MC={bm['mc_total']:.2f} "
                      f"XOR1={xor['tau1']*100:.1f}% XOR3={xor['tau3']*100:.1f}% "
                      f"XOR5={xor['tau5']*100:.1f}% "
                      f"N5={bm['narma5']:.3f} W4={bm['wave4_acc']*100:.1f}%")

            except Exception as e:
                print(f"  rep{rep}: EXCEPTION: {e}")
                cond_results[rep_key] = {'error': str(e)}

            # Incremental save
            results['conditions'][cond_name] = cond_results
            with open(SAVE_FILE, 'w') as f:
                json.dump(results, f, indent=2, cls=NpEncoder)

    # Save states
    if all_states:
        np.save(STATES_FILE, all_states, allow_pickle=True)
        print(f"\n  States saved to {STATES_FILE}")

    # ================================================================
    # Step 4: Run tests
    # ================================================================
    print("\n" + "=" * 70)
    print("  TESTS")
    print("=" * 70)

    tests = run_tests(results)
    results['tests'] = tests

    n_pass = 0
    n_total = len(tests)
    for tid in sorted(tests.keys()):
        t = tests[tid]
        passed = t.get('pass', False)
        if passed:
            n_pass += 1
        status = "PASS" if passed else "FAIL"
        desc = t.get('desc', '')
        # Format details
        detail_parts = []
        for k, v in t.items():
            if k in ('desc', 'pass', 'note', 'psd_freqs', 'psd_values'):
                continue
            if isinstance(v, float):
                detail_parts.append(f"{k}={v:.4f}")
            elif isinstance(v, dict):
                detail_parts.append(f"{k}={v}")
            else:
                detail_parts.append(f"{k}={v}")
        detail = ", ".join(detail_parts)
        print(f"  {tid} [{status}] {desc}")
        if detail:
            print(f"         {detail}")

    print(f"\n  TOTAL: {n_pass}/{n_total} PASS")

    # ================================================================
    # Step 5: Summary table
    # ================================================================
    print("\n" + "=" * 70)
    print("  SUMMARY TABLE (mean across reps)")
    print("=" * 70)
    print(f"  {'Condition':<18} {'MC':>8} {'XOR1':>8} {'XOR3':>8} {'XOR5':>8} {'NARMA5':>8} {'Wave4':>8}")
    print(f"  {'-'*66}")
    for cond_name in CONDITIONS:
        mc = mean_metric(results, cond_name, 'mc_total')
        x1 = mean_metric(results, cond_name, 'xor.tau1')
        x3 = mean_metric(results, cond_name, 'xor.tau3')
        x5 = mean_metric(results, cond_name, 'xor.tau5')
        n5 = mean_metric(results, cond_name, 'narma5')
        w4 = mean_metric(results, cond_name, 'wave4_acc')
        fmt = lambda v, pct=False: f"{v*100:.1f}%" if v is not None and pct else (f"{v:.3f}" if v is not None else "N/A")
        print(f"  {cond_name:<18} {fmt(mc):>8} {fmt(x1,True):>8} {fmt(x3,True):>8} "
              f"{fmt(x5,True):>8} {fmt(n5):>8} {fmt(w4,True):>8}")

    # Final save
    with open(SAVE_FILE, 'w') as f:
        json.dump(results, f, indent=2, cls=NpEncoder)
    print(f"\n  Results saved to {SAVE_FILE}")
    print(f"  Done.")


if __name__ == '__main__':
    main()
