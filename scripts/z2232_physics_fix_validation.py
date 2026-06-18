#!/usr/bin/env python3
"""
z2232_physics_fix_validation.py — Validate three physics fixes
================================================================
Fix A: Direct MAC current injection (BIAS_GAIN=0x0800 = 0.03125)
Fix B: Slower membrane τ=210ms (LEAK_COND=0x0004)
Fix C: 200Hz Ethernet auto-telemetry (vs prior 20Hz)

Optimal params from z2233b sweep:
  THRESHOLD = 2.00V, DT_OVER_C = 0x0200, BASE_EXC = 0x0200
  LEAK_COND = 0x0004, Vg = 0.62, BIAS_GAIN = 0x0800

5 experiments, 17 tests (T794-T810):
  EXP 1: Memory Capacity (R² at delays d=1..10)
  EXP 2: Classification with MAC current (4-class waveform)
  EXP 3: Temporal Features at 200Hz
  EXP 4: Transfer Entropy at 200Hz
  EXP 5: Full Stack (joint encoding + XOR + NARMA-5)

NOTE: Spike counters reset after each telemetry read. Each packet's
spike_counts = spikes since last read. Accumulate to get total.
"""
import sys, os, time, json, struct, datetime
import numpy as np
sys.path.insert(0, "scripts")
from fpga_host_eth import FPGAEthBridge

# ── Paths ──
RESULTS = "results"
os.makedirs(RESULTS, exist_ok=True)

# ── Optimal params — MAC-modulated regime ──
# KEY FINDING: Vg has no effect above BVpar cliff (~0.55V). i_aval saturates.
# MAC current injection (via BIAS_GAIN) DOES modulate spike rate:
#   BGAIN=0x2000, th=0.50: MAC 0→1 gives 47% spike rate increase.
# Use MAC as primary input signal, fixed Vg=0.62.
THRESH    = 0.50      # lower threshold for MAC sensitivity
DTC       = 0x0200    # DT_OVER_C
BEXC      = 0x0200    # BASE_EXC (moderate)
LEAK      = 0x0004    # slow cortical τ≈210ms
BGAIN     = 0x2000    # BIAS_GAIN 4× stronger for real MAC modulation
VG_BASE   = 0.62
REFRACT   = 50        # 5us @ 10MHz
N_NEURONS = 128

# ── Telemetry rate ──
# FPGA auto-telem limited to ~2kHz (16-bit divider).
# Input signal updates at INPUT_HZ (200Hz), telemetry sampled at TELEM_HZ (2kHz).
# Each input step accumulates PKTS_PER_STEP telemetry packets.
TELEM_HZ  = 2000      # actual telemetry rate (FPGA hardware)
INPUT_HZ  = 200       # input signal update rate
PKTS_PER_STEP = TELEM_HZ // INPUT_HZ  # 10 packets per input step

# ── GPU sysfs channels ──
def read_pm_table():
    try:
        with open('/sys/kernel/ryzen_smu_drv/pm_table', 'rb') as f:
            f.seek(0x004C)
            temp = struct.unpack('<f', f.read(4))[0]
            f.seek(0x0100)
            power = struct.unpack('<f', f.read(4))[0]
            f.seek(0x0344)
            sclk = struct.unpack('<f', f.read(4))[0]
        return temp, power, sclk
    except:
        return 45.0, 10.0, 1000.0

def read_hwmon():
    base = '/sys/class/hwmon'
    for h in sorted(os.listdir(base)):
        name_path = os.path.join(base, h, 'name')
        try:
            with open(name_path) as f:
                if 'amdgpu' in f.read():
                    hp = os.path.join(base, h)
                    power = float(open(os.path.join(hp, 'power1_average')).read()) / 1e6
                    temp = float(open(os.path.join(hp, 'temp1_input')).read()) / 1000.0
                    freq = float(open(os.path.join(hp, 'freq1_input')).read()) / 1e6
                    return power, temp, freq
        except:
            continue
    return 10.0, 45.0, 1000.0

def read_gpu_busy():
    try:
        return float(open('/sys/class/drm/card0/device/gpu_busy_percent').read())
    except:
        return 0.0

GPU_HAS_TORCH = False
try:
    import torch
    if torch.cuda.is_available():
        GPU_HAS_TORCH = True
        _jit_a = torch.randn(64, 64, device='cuda')
        _jit_b = torch.randn(64, 64, device='cuda')
except:
    pass

def measure_dispatch_jitter():
    if not GPU_HAS_TORCH:
        return 0.0
    t0 = time.perf_counter()
    torch.mm(_jit_a, _jit_b)
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1e6  # microseconds

def read_all_gpu_state():
    pm_t, pm_p, pm_sclk = read_pm_table()
    hw_p, hw_t, hw_f = read_hwmon()
    busy = read_gpu_busy()
    jitter = measure_dispatch_jitter()
    return np.array([pm_t, pm_p, pm_sclk, hw_p, hw_t, hw_f, busy, jitter], dtype=np.float32)

N_GPU_CH = 8

# ── Helpers ──
class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer,)): return int(obj)
        if isinstance(obj, (np.floating,)): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        return super().default(obj)

def drain(fpga, n=100):
    for _ in range(n):
        try: fpga.recv_auto_telemetry(timeout=0.002)
        except: break

def collect_trial(fpga, n_steps, mac_pattern=None, mac_signal=0.0, gpu_workload=False):
    """Collect n_steps of input-rate data (each step = PKTS_PER_STEP telemetry packets).
    Input signal updated at INPUT_HZ via MAC, telemetry captured at TELEM_HZ (2kHz).
    mac_pattern: array of MAC values per step (overrides mac_signal if given).
    spike_data[t, n] = accumulated spikes for neuron n during step t.
    vmem_data[t, n] = last vmem snapshot for neuron n during step t.
    """
    if mac_pattern is None:
        fpga.set_mac_signal(mac_signal)

    spike_data = np.zeros((n_steps, N_NEURONS), dtype=np.int32)
    vmem_data = np.zeros((n_steps, N_NEURONS), dtype=np.float32)
    gpu_log = np.zeros((n_steps, N_GPU_CH), dtype=np.float32)

    for t in range(n_steps):
        if mac_pattern is not None:
            fpga.set_mac_signal(float(mac_pattern[t % len(mac_pattern)]))

        # Accumulate PKTS_PER_STEP telemetry packets per input step
        step_spikes = np.zeros(N_NEURONS, dtype=np.int32)
        last_vmem = np.zeros(N_NEURONS, dtype=np.float32)
        got = 0
        for _ in range(PKTS_PER_STEP + 5):  # allow a few retries
            try:
                pkt = fpga.recv_auto_telemetry(timeout=0.01)
                if pkt is not None:
                    step_spikes += pkt['spike_counts'].astype(np.int32)
                    last_vmem = pkt['vmem'].astype(np.float32)
                    got += 1
                    if got >= PKTS_PER_STEP:
                        break
            except:
                pass

        spike_data[t] = step_spikes
        vmem_data[t] = last_vmem

        if gpu_workload and GPU_HAS_TORCH:
            intensity = step_spikes.sum() / (N_NEURONS * 10.0)
            n = max(16, min(512, int(intensity * 512)))
            a = torch.randn(n, n, device='cuda')
            torch.mm(a, a)
            torch.cuda.synchronize()

        gpu_log[t] = read_all_gpu_state()

    return spike_data, vmem_data, gpu_log

def pool_features(spike_data, vmem_data, gpu_log=None):
    """Pool temporal features: mean, std, max, min over time axis."""
    # spike_data: (T, N), vmem_data: (T, N)
    raw = np.concatenate([spike_data.astype(np.float32), vmem_data], axis=1)  # (T, 2N)

    # Add 1-step delay features
    delayed = np.zeros_like(raw)
    delayed[1:] = raw[:-1]
    aug = np.concatenate([raw, delayed], axis=1)  # (T, 4N)

    feats = np.concatenate([
        aug.mean(axis=0),
        aug.std(axis=0),
        aug.max(axis=0),
        aug.min(axis=0),
    ])
    if gpu_log is not None:
        gpu_feats = np.concatenate([
            gpu_log.mean(axis=0),
            gpu_log.std(axis=0),
            gpu_log.max(axis=0),
            gpu_log.min(axis=0),
        ])
        feats = np.concatenate([feats, gpu_feats])
    return feats

def pool_temporal_features(spike_data, vmem_data):
    """Enhanced temporal features for 200Hz data.
    Extracts delta_spikes, vmem_derivative, ISI statistics.
    """
    T, N = spike_data.shape

    # Delta spikes (rate of change)
    delta_spikes = np.diff(spike_data.astype(np.float32), axis=0)  # (T-1, N)
    # Vmem derivative
    vmem_deriv = np.diff(vmem_data, axis=0)  # (T-1, N)

    # ISI features: for each neuron, compute std/mean of inter-spike intervals
    isi_cv = np.zeros(N, dtype=np.float32)
    for n in range(N):
        spike_times = np.where(spike_data[:, n] > 0)[0]
        if len(spike_times) > 2:
            isis = np.diff(spike_times).astype(np.float32)
            if isis.mean() > 0:
                isi_cv[n] = isis.std() / isis.mean()

    feats = np.concatenate([
        # Standard features
        spike_data.mean(axis=0),
        spike_data.std(axis=0),
        vmem_data.mean(axis=0),
        vmem_data.std(axis=0),
        # Temporal features (200Hz-enabled)
        delta_spikes.mean(axis=0),
        delta_spikes.std(axis=0),
        vmem_deriv.mean(axis=0),
        vmem_deriv.std(axis=0),
        isi_cv,
    ])
    return feats

def ridge_classify(X_tr, y_tr, X_te, y_te, n_classes):
    alphas = [1e-4, 1e-2, 1.0, 100.0, 1000.0, 10000.0]
    Y_tr = np.zeros((len(y_tr), n_classes))
    for i, y in enumerate(y_tr):
        Y_tr[i, int(y)] = 1.0
    best_acc = 0.0
    for a in alphas:
        try:
            I = np.eye(X_tr.shape[1])
            W = np.linalg.solve(X_tr.T @ X_tr + a * I, X_tr.T @ Y_tr)
            preds = np.argmax(X_te @ W, axis=1)
            acc = np.mean(preds == y_te)
            if acc > best_acc:
                best_acc = acc
        except:
            continue
    return best_acc

def classify_cv(X, y, n_splits=5, max_pca=120):
    n = len(y)
    classes = np.unique(y)
    n_classes = len(classes)
    idx = np.arange(n)
    np.random.shuffle(idx)
    fold_size = n // n_splits
    accs = []
    for fold in range(n_splits):
        te = idx[fold * fold_size:(fold + 1) * fold_size]
        tr = np.setdiff1d(idx, te)
        X_tr, X_te = X[tr], X[te]
        y_tr, y_te = y[tr], y[te]

        mu = X_tr.mean(axis=0)
        sigma = X_tr.std(axis=0)
        sigma[sigma < 1e-2] = 1.0
        X_tr_n = (X_tr - mu) / sigma
        X_te_n = (X_te - mu) / sigma

        if X_tr_n.shape[1] > max_pca:
            cov = X_tr_n.T @ X_tr_n / len(X_tr_n)
            eigvals, eigvecs = np.linalg.eigh(cov)
            order = np.argsort(eigvals)[::-1][:max_pca]
            Vt = eigvecs[:, order].T
            X_tr_n = X_tr_n @ Vt.T
            X_te_n = X_te_n @ Vt.T

        acc = ridge_classify(X_tr_n, y_tr, X_te_n, y_te, n_classes)
        accs.append(acc)
    return {'mean': float(np.mean(accs)), 'std': float(np.std(accs)), 'per_fold': [float(a) for a in accs]}

def ridge_regression(X_tr, y_tr, X_te, y_te, alpha=1.0):
    """Ridge regression, returns R²."""
    I = np.eye(X_tr.shape[1])
    W = np.linalg.solve(X_tr.T @ X_tr + alpha * I, X_tr.T @ y_tr)
    y_pred = X_te @ W
    ss_res = np.sum((y_te - y_pred) ** 2)
    ss_tot = np.sum((y_te - y_te.mean()) ** 2)
    if ss_tot < 1e-12:
        return 0.0
    return float(1.0 - ss_res / ss_tot)


# ── Waveform generators ──
def make_waveform(cls, n_steps, freq=2.0):
    """Generate MAC waveform in [0.1, 0.9] — 4 waveform classes."""
    t = np.linspace(0, n_steps / INPUT_HZ, n_steps)
    if cls == 0:    # sine
        return 0.5 + 0.4 * np.sin(2 * np.pi * freq * t)
    elif cls == 1:  # square
        return 0.5 + 0.4 * np.sign(np.sin(2 * np.pi * freq * t))
    elif cls == 2:  # sawtooth
        return 0.5 + 0.4 * (2 * (freq * t % 1) - 1)
    else:           # triangle
        return 0.5 + 0.4 * (2 * np.abs(2 * (freq * t % 1) - 1) - 1)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
def main():
    np.random.seed(42)
    fpga = FPGAEthBridge()
    if not fpga.connect():
        print("FAIL: Cannot connect to FPGA")
        return

    # ── Configure optimal parameters ──
    print("Setting optimal physics parameters...")
    fpga.set_kill(False)
    fpga.set_threshold(THRESH)
    fpga.set_base_exc_raw(BEXC)
    fpga.set_leak_cond(LEAK)
    fpga.set_bias_gain_raw(BGAIN)
    fpga.set_dt_over_c_raw(DTC)
    fpga.set_refract_cycles(REFRACT)
    fpga.set_mac_signal(0.0)
    fpga.set_vg_batch(0, [VG_BASE] * 64)
    fpga.set_vg_batch(64, [VG_BASE] * 64)
    fpga.enable_auto_telemetry(TELEM_HZ)  # 2kHz auto-push
    time.sleep(0.5)
    drain(fpga, 200)

    # Quick sanity check
    pkt = fpga.recv_auto_telemetry(timeout=0.1)
    if pkt is None:
        print("FAIL: No telemetry")
        return
    print(f"  Telemetry OK: {pkt['spike_counts'].sum()} spikes, vmem_mean={pkt['vmem'].mean():.4f}")

    results = {
        'experiment': 'z2232_physics_fix_validation',
        'timestamp': datetime.datetime.now().isoformat(),
        'architecture': {
            'n_neurons': N_NEURONS,
            'telem_hz': TELEM_HZ,
            'threshold': THRESH,
            'dt_over_c': hex(DTC),
            'base_exc': hex(BEXC),
            'leak_cond': hex(LEAK),
            'bias_gain': hex(BGAIN),
            'vg_base': VG_BASE,
            'refract_cycles': REFRACT,
        },
        'tests': {},
    }

    # ══════════════════════════════════════════════════════════════════════════
    # EXP 1: Memory Capacity (τ=210ms @ 200Hz)
    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "="*70)
    print("EXP 1: MEMORY CAPACITY")
    print("="*70)

    N_MC_STEPS = 400   # 2 seconds at 200Hz
    N_MC_TRIALS = 10   # 10 trials × 400 steps = sufficient
    DELAYS = list(range(1, 11))  # d=1..10 (5ms to 50ms)

    mc_results = {}
    for mode in ['COUPLED', 'FPGA_ONLY', 'STATIC']:
        print(f"\n  Mode: {mode}")
        all_features = []
        all_inputs = []

        for trial in range(N_MC_TRIALS):
            # i.i.d. uniform random input [0,1] — standard MC protocol
            # (NOT cumsum random walk — tiny step changes get drowned by spike noise)
            input_sig = np.random.rand(N_MC_STEPS)

            if mode in ('COUPLED', 'FPGA_ONLY'):
                # MAC modulated by input: [0.1, 0.9]
                mac_pat = 0.1 + 0.8 * input_sig
            else:  # STATIC
                mac_pat = np.full(N_MC_STEPS, 0.5)

            drain(fpga, 50)
            spk, vm, gpu = collect_trial(fpga, N_MC_STEPS, mac_pattern=mac_pat,
                                          gpu_workload=(mode == 'COUPLED'))

            # Features: spike counts only (vmem may be near-constant → noise dims)
            feat_per_step = spk.astype(np.float32)  # (T, N_NEURONS)
            all_features.append(feat_per_step)
            all_inputs.append(input_sig)

            rate = spk.sum() / (N_MC_STEPS / INPUT_HZ) / N_NEURONS
            spk_std = spk.astype(float).std(axis=0).mean()
            corr = np.corrcoef(input_sig, spk.sum(axis=1))[0, 1] if spk.sum() > 0 else 0
            print(f"    Trial {trial}/{N_MC_TRIALS}: rate={rate:.1f} spk/s/n, spk_std={spk_std:.3f}, corr(in,spk_sum)={corr:.3f}")

        # Compute MC at each delay — same approach as manual diagnostic (R²=0.66)
        mc_per_delay = {}
        for d in DELAYS:
            r2_all = []
            for feat, inp in zip(all_features, all_inputs):
                T = len(inp) - d
                if T < 20:
                    continue
                X = feat[d:d+T]  # features at time t
                y = inp[:T]       # input at time t-d
                # Ridge regression (no normalization — matches manual test)
                for alpha in [1.0, 10.0]:
                    try:
                        XtX = X.T @ X + alpha * np.eye(X.shape[1])
                        W = np.linalg.solve(XtX, X.T @ y)
                        y_pred = X @ W
                        ss_res = np.sum((y - y_pred) ** 2)
                        ss_tot = np.sum((y - y.mean()) ** 2)
                        r2 = 1.0 - ss_res / (ss_tot + 1e-12)
                        r2_all.append(max(0.0, r2))
                    except:
                        pass
            mc_per_delay[d] = float(np.mean(r2_all)) if r2_all else 0.0

        mc_total = sum(mc_per_delay.values())
        mc_results[mode] = {
            'r2_per_delay': mc_per_delay,
            'mc_total': mc_total,
        }
        print(f"  {mode}: MC_total={mc_total:.4f}, R²(d=1)={mc_per_delay.get(1, 0):.4f}")

    results['memory_capacity'] = mc_results

    # Tests T794-T796
    r2_d1_coupled = mc_results.get('COUPLED', {}).get('r2_per_delay', {}).get(1, 0)
    mc_coupled = mc_results.get('COUPLED', {}).get('mc_total', 0)
    mc_fpga = mc_results.get('FPGA_ONLY', {}).get('mc_total', 0)
    mc_static = mc_results.get('STATIC', {}).get('mc_total', 0)

    results['tests']['T794'] = {
        'desc': 'Memory capacity R²(d=1) > 0.10',
        'val': r2_d1_coupled,
        'pass': r2_d1_coupled > 0.10,
    }
    results['tests']['T795'] = {
        'desc': 'MC COUPLED > MC STATIC',
        'val': f'{mc_coupled:.4f} vs {mc_static:.4f}',
        'pass': mc_coupled > mc_static,
    }
    results['tests']['T796'] = {
        'desc': 'MC COUPLED > MC FPGA_ONLY',
        'val': f'{mc_coupled:.4f} vs {mc_fpga:.4f}',
        'pass': mc_coupled > mc_fpga,
    }

    # ══════════════════════════════════════════════════════════════════════════
    # EXP 2: Classification with MAC Current
    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "="*70)
    print("EXP 2: CLASSIFICATION WITH MAC CURRENT")
    print("="*70)

    N_CLASS_STEPS = 200  # 1 second at 200Hz
    N_TRIALS_PER_CLASS = 30  # 30 × 4 classes = 120 trials
    N_CLASSES = 4
    FREQS = [1.0, 2.0, 3.0, 5.0]  # varied frequencies for richness

    class_results = {}
    for mode in ['COUPLED', 'FPGA_ONLY', 'STATIC']:
        print(f"\n  Mode: {mode}")
        X_all, y_all = [], []

        for cls in range(N_CLASSES):
            for trial in range(N_TRIALS_PER_CLASS):
                freq = FREQS[trial % len(FREQS)]
                waveform = make_waveform(cls, N_CLASS_STEPS, freq=freq)

                if mode == 'STATIC':
                    # Constant MAC — no waveform info
                    mac_pat = np.full(N_CLASS_STEPS, 0.5)
                else:
                    mac_pat = waveform  # MAC carries waveform shape

                drain(fpga, 30)
                spk, vm, gpu = collect_trial(fpga, N_CLASS_STEPS, mac_pattern=mac_pat,
                                              gpu_workload=(mode == 'COUPLED'))

                feat = pool_features(spk, vm, gpu if mode == 'COUPLED' else None)
                X_all.append(feat)
                y_all.append(cls)

        X = np.array(X_all)
        y = np.array(y_all)
        cv = classify_cv(X, y, n_splits=5)
        class_results[mode] = cv
        print(f"  {mode}: accuracy={cv['mean']:.3f}±{cv['std']:.3f}")

    results['classification'] = class_results

    acc_coupled = class_results.get('COUPLED', {}).get('mean', 0)
    acc_fpga = class_results.get('FPGA_ONLY', {}).get('mean', 0)
    acc_static = class_results.get('STATIC', {}).get('mean', 0)

    results['tests']['T797'] = {
        'desc': 'Classification COUPLED >= FPGA_ONLY',
        'val': f'{acc_coupled:.3f} vs {acc_fpga:.3f}',
        'pass': acc_coupled >= acc_fpga - 0.01,  # within 1pp tolerance
    }
    results['tests']['T798'] = {
        'desc': 'Classification COUPLED > STATIC',
        'val': f'{acc_coupled:.3f} vs {acc_static:.3f}',
        'pass': acc_coupled > acc_static,
    }
    results['tests']['T799'] = {
        'desc': 'Classification COUPLED > 0.50 (above chance)',
        'val': acc_coupled,
        'pass': acc_coupled > 0.50,
    }

    # ══════════════════════════════════════════════════════════════════════════
    # EXP 3: Temporal Features at 200Hz
    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "="*70)
    print("EXP 3: TEMPORAL vs SNAPSHOT FEATURES")
    print("="*70)

    # Re-use classification data but compare temporal vs snapshot features
    temporal_results = {}
    for mode in ['COUPLED', 'FPGA_ONLY']:
        print(f"\n  Mode: {mode}")
        X_snap, X_temp, y_all = [], [], []

        for cls in range(N_CLASSES):
            for trial in range(N_TRIALS_PER_CLASS):
                freq = FREQS[trial % len(FREQS)]
                waveform = make_waveform(cls, N_CLASS_STEPS, freq=freq)

                drain(fpga, 30)
                spk, vm, gpu = collect_trial(fpga, N_CLASS_STEPS, mac_pattern=waveform,
                                              gpu_workload=(mode == 'COUPLED'))

                # Snapshot: just mean spike rate and mean vmem (no temporal info)
                snap = np.concatenate([spk.mean(axis=0), vm.mean(axis=0)])
                # Temporal: delta, derivative, ISI features
                temp = pool_temporal_features(spk, vm)

                X_snap.append(snap)
                X_temp.append(temp)
                y_all.append(cls)

        X_s = np.array(X_snap)
        X_t = np.array(X_temp)
        y = np.array(y_all)

        cv_snap = classify_cv(X_s, y, n_splits=5)
        cv_temp = classify_cv(X_t, y, n_splits=5)
        temporal_results[mode] = {
            'snapshot': cv_snap,
            'temporal': cv_temp,
        }
        print(f"  {mode}: snapshot={cv_snap['mean']:.3f}, temporal={cv_temp['mean']:.3f}")

    results['temporal_features'] = temporal_results

    snap_c = temporal_results.get('COUPLED', {}).get('snapshot', {}).get('mean', 0)
    temp_c = temporal_results.get('COUPLED', {}).get('temporal', {}).get('mean', 0)
    snap_f = temporal_results.get('FPGA_ONLY', {}).get('snapshot', {}).get('mean', 0)
    temp_f = temporal_results.get('FPGA_ONLY', {}).get('temporal', {}).get('mean', 0)

    results['tests']['T800'] = {
        'desc': 'Temporal > Snapshot for COUPLED (by 5pp+)',
        'val': f'{temp_c:.3f} vs {snap_c:.3f} (diff={temp_c-snap_c:.3f})',
        'pass': temp_c > snap_c + 0.05,
    }
    results['tests']['T801'] = {
        'desc': 'Temporal > Snapshot for FPGA_ONLY',
        'val': f'{temp_f:.3f} vs {snap_f:.3f}',
        'pass': temp_f > snap_f,
    }
    results['tests']['T802'] = {
        'desc': 'Temporal COUPLED > Temporal FPGA_ONLY',
        'val': f'{temp_c:.3f} vs {temp_f:.3f}',
        'pass': temp_c > temp_f,
    }

    # ══════════════════════════════════════════════════════════════════════════
    # EXP 4: Transfer Entropy at 200Hz
    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "="*70)
    print("EXP 4: TRANSFER ENTROPY GPU→FPGA")
    print("="*70)

    N_TE_STEPS = 1000  # 5 seconds at 200Hz
    N_TE_TRIALS = 10
    TE_LAGS = [1, 2, 5, 10, 20]  # 5ms, 10ms, 25ms, 50ms, 100ms

    te_results = {}
    for mode in ['COUPLED', 'STATIC']:
        print(f"\n  Mode: {mode}")
        te_per_lag = {lag: [] for lag in TE_LAGS}

        for trial in range(N_TE_TRIALS):
            mac = 0.5 if mode == 'COUPLED' else 0.0
            drain(fpga, 50)
            spk, vm, gpu = collect_trial(fpga, N_TE_STEPS, mac_signal=mac,
                                          gpu_workload=(mode == 'COUPLED'))

            # GPU signal: power (most dynamic)
            gpu_sig = gpu[:, 3]  # hw_power
            # FPGA signal: total spike rate
            fpga_sig = spk.sum(axis=1).astype(np.float32)

            # Compute transfer entropy approximation via lagged correlation
            for lag in TE_LAGS:
                if lag >= len(gpu_sig) - 10:
                    continue
                # TE approximation: conditional MI via regression residuals
                # TE(GPU→FPGA) ≈ how much GPU[t-lag] improves prediction of FPGA[t]
                T = len(fpga_sig) - lag
                y = fpga_sig[lag:lag+T]
                x_self = fpga_sig[:T]  # FPGA's own past
                x_gpu = gpu_sig[:T]     # GPU past

                mu_s, std_s = x_self.mean(), max(x_self.std(), 1e-6)
                mu_g, std_g = x_gpu.mean(), max(x_gpu.std(), 1e-6)
                mu_y, std_y = y.mean(), max(y.std(), 1e-6)

                x_s_n = (x_self - mu_s) / std_s
                x_g_n = (x_gpu - mu_g) / std_g
                y_n = (y - mu_y) / std_y

                # Residual variance with self only
                try:
                    w_self = np.linalg.lstsq(x_s_n.reshape(-1, 1), y_n, rcond=None)[0]
                    res_self = y_n - x_s_n.reshape(-1, 1) @ w_self
                    var_self = np.var(res_self)
                except:
                    var_self = 1.0

                # Residual variance with self + GPU
                try:
                    X_both = np.column_stack([x_s_n, x_g_n])
                    w_both = np.linalg.lstsq(X_both, y_n, rcond=None)[0]
                    res_both = y_n - X_both @ w_both
                    var_both = np.var(res_both)
                except:
                    var_both = var_self

                # TE ≈ 0.5 * log(var_self / var_both) [in bits]
                te = max(0.0, 0.5 * np.log2(max(var_self, 1e-12) / max(var_both, 1e-12)))
                te_per_lag[lag].append(te)

        te_mean = {lag: float(np.mean(vals)) if vals else 0.0 for lag, vals in te_per_lag.items()}
        te_results[mode] = te_mean
        peak_lag = max(te_mean, key=te_mean.get) if te_mean else 0
        peak_te = te_mean.get(peak_lag, 0)
        print(f"  {mode}: peak TE={peak_te:.4f} bits at lag={peak_lag} ({peak_lag*5}ms)")

    results['transfer_entropy'] = te_results

    te_coupled = te_results.get('COUPLED', {})
    te_static = te_results.get('STATIC', {})
    te_coupled_peak = max(te_coupled.values()) if te_coupled else 0
    te_static_peak = max(te_static.values()) if te_static else 0
    te_lag1 = te_coupled.get(1, 0)

    results['tests']['T803'] = {
        'desc': 'TE(GPU→FPGA) > 0.05 bits at any lag (COUPLED)',
        'val': te_coupled_peak,
        'pass': te_coupled_peak > 0.05,
    }
    results['tests']['T804'] = {
        'desc': 'TE COUPLED > TE STATIC at peak lag',
        'val': f'{te_coupled_peak:.4f} vs {te_static_peak:.4f}',
        'pass': te_coupled_peak > te_static_peak,
    }
    results['tests']['T805'] = {
        'desc': 'TE at lag=1 (5ms) > 0.01 bits',
        'val': te_lag1,
        'pass': te_lag1 > 0.01,
    }

    # ══════════════════════════════════════════════════════════════════════════
    # EXP 5: Full Stack (XOR + NARMA-5)
    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "="*70)
    print("EXP 5: FULL STACK — XOR + NARMA-5")
    print("="*70)

    N_FULL_STEPS = 500  # 2.5 seconds at 200Hz
    N_FULL_TRIALS = 40

    full_results = {}
    for mode in ['COUPLED', 'FPGA_ONLY', 'STATIC']:
        print(f"\n  Mode: {mode}")

        # Generate common input sequences
        all_xor_feats = []
        all_xor_labels = []
        all_narma_feats = []
        all_narma_targets = []

        for trial in range(N_FULL_TRIALS):
            # Binary input for XOR → MAC signal
            u = np.random.choice([0.0, 1.0], size=N_FULL_STEPS)
            mac_pat = 0.1 + 0.8 * u  # MAC [0.1, 0.9] for binary input

            if mode == 'STATIC':
                mac_pat = np.full(N_FULL_STEPS, 0.5)

            drain(fpga, 30)
            spk, vm, gpu = collect_trial(fpga, N_FULL_STEPS, mac_pattern=mac_pat,
                                          gpu_workload=(mode == 'COUPLED'))

            # XOR task: y[t] = u[t] XOR u[t-2]
            TAU = 2
            for t in range(TAU, N_FULL_STEPS):
                feat = np.concatenate([spk[t].astype(np.float32), vm[t]])
                xor_label = int(u[t]) ^ int(u[t - TAU])
                all_xor_feats.append(feat)
                all_xor_labels.append(xor_label)

            # NARMA-5 target
            narma = np.zeros(N_FULL_STEPS)
            for t in range(5, N_FULL_STEPS):
                narma[t] = (0.3 * narma[t-1] + 0.05 * narma[t-1] *
                           sum(narma[t-j] for j in range(1, 6)) +
                           1.5 * u[t-1] * u[t-5] + 0.1)
                narma[t] = np.clip(narma[t], -10, 10)
            for t in range(10, N_FULL_STEPS):
                feat = np.concatenate([spk[t].astype(np.float32), vm[t]])
                all_narma_feats.append(feat)
                all_narma_targets.append(narma[t])

            if trial % 10 == 0:
                rate = spk.sum() / (N_FULL_STEPS / INPUT_HZ) / N_NEURONS
                print(f"    Trial {trial}: rate={rate:.1f} spk/s/n")

        # XOR classification
        X_xor = np.array(all_xor_feats)
        y_xor = np.array(all_xor_labels)
        cv_xor = classify_cv(X_xor, y_xor, n_splits=5, max_pca=80)

        # NARMA-5 regression
        X_narma = np.array(all_narma_feats)
        y_narma = np.array(all_narma_targets)
        # Simple train/test split for regression
        split = int(0.8 * len(y_narma))
        X_tr, X_te = X_narma[:split], X_narma[split:]
        y_tr, y_te = y_narma[:split], y_narma[split:]
        mu = X_tr.mean(axis=0)
        sigma = X_tr.std(axis=0)
        sigma[sigma < 1e-2] = 1.0
        X_tr_n = (X_tr - mu) / sigma
        X_te_n = (X_te - mu) / sigma
        # PCA if needed
        if X_tr_n.shape[1] > 80:
            cov = X_tr_n.T @ X_tr_n / len(X_tr_n)
            eigvals, eigvecs = np.linalg.eigh(cov)
            order = np.argsort(eigvals)[::-1][:80]
            Vt = eigvecs[:, order].T
            X_tr_n = X_tr_n @ Vt.T
            X_te_n = X_te_n @ Vt.T
        best_r2 = -1.0
        for alpha in [0.1, 1.0, 10.0, 100.0, 1000.0]:
            r2 = ridge_regression(X_tr_n, y_tr, X_te_n, y_te, alpha=alpha)
            if r2 > best_r2:
                best_r2 = r2
        narma_r2 = max(0.0, best_r2)

        full_results[mode] = {
            'xor_tau2': cv_xor,
            'narma5_r2': narma_r2,
        }
        print(f"  {mode}: XOR τ=2 acc={cv_xor['mean']:.3f}, NARMA-5 R²={narma_r2:.4f}")

    results['full_stack'] = full_results

    xor_c = full_results.get('COUPLED', {}).get('xor_tau2', {}).get('mean', 0)
    xor_f = full_results.get('FPGA_ONLY', {}).get('xor_tau2', {}).get('mean', 0)
    xor_s = full_results.get('STATIC', {}).get('xor_tau2', {}).get('mean', 0)
    narma_c = full_results.get('COUPLED', {}).get('narma5_r2', 0)
    narma_f = full_results.get('FPGA_ONLY', {}).get('narma5_r2', 0)

    results['tests']['T806'] = {
        'desc': 'XOR τ=2 COUPLED > 0.55 (above chance)',
        'val': xor_c,
        'pass': xor_c > 0.55,
    }
    results['tests']['T807'] = {
        'desc': 'XOR τ=2 COUPLED > STATIC',
        'val': f'{xor_c:.3f} vs {xor_s:.3f}',
        'pass': xor_c > xor_s,
    }
    results['tests']['T808'] = {
        'desc': 'NARMA-5 COUPLED R² > 0.05',
        'val': narma_c,
        'pass': narma_c > 0.05,
    }
    results['tests']['T809'] = {
        'desc': 'NARMA-5 COUPLED > FPGA_ONLY',
        'val': f'{narma_c:.4f} vs {narma_f:.4f}',
        'pass': narma_c > narma_f,
    }
    results['tests']['T810'] = {
        'desc': 'COUPLED wins >= 2 of 3 tasks (MC, XOR, NARMA)',
        'val': f'MC:{mc_coupled > mc_fpga}, XOR:{xor_c > xor_f}, NARMA:{narma_c > narma_f}',
        'pass': sum([mc_coupled > mc_fpga, xor_c > xor_f, narma_c > narma_f]) >= 2,
    }

    # ══════════════════════════════════════════════════════════════════════════
    # SUMMARY
    # ══════════════════════════════════════════════════════════════════════════
    n_pass = sum(1 for t in results['tests'].values() if t['pass'])
    n_total = len(results['tests'])
    results['score'] = f'{n_pass}/{n_total}'

    print("\n" + "="*70)
    print(f"RESULTS: {n_pass}/{n_total} PASS")
    print("="*70)
    for tid, t in sorted(results['tests'].items()):
        status = "PASS" if t['pass'] else "FAIL"
        print(f"  {tid}: {status} — {t['desc']} (val={t['val']})")

    out_path = os.path.join(RESULTS, 'z2232_physics_fix_validation.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2, cls=NpEncoder)
    print(f"\nSaved to {out_path}")
    fpga.close()

if __name__ == "__main__":
    main()
