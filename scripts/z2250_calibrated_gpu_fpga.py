#!/usr/bin/env python3
"""
z2250: Calibrated GPU-FPGA Deep Integration
=============================================
Fix z2249's Vg overdrive: use z2248's proven collection approach (IIR-filtered
power noise, controlled MAC coupling) but with z2249's deep GPU feature extraction
as READOUT features (not input modulation).

Key insight: GPU telemetry should be READ (for features) not WRITTEN (to Vg).
Vg modulation uses only calibrated power noise at SR-optimal level.

8 experiments, 30 tests (T1231-T1260):

EXP 1 — Baseline Sanity (T1231-T1233)
  Confirm z2248 optimal params still work: leak=0x02, bg=0.0625, noise=0.05.
  4-class waveform with proven collect_trial approach. Target: >90%.

EXP 2 — GPU Features as Readout (T1234-T1237)
  Use deep gpu_metrics (36D) as ADDITIONAL readout features alongside spike/vmem.
  Compare: spike-only vs spike+gpu vs gpu-only. Target: spike+gpu > spike-only.

EXP 3 — Multi-Source Noise Comparison (T1238-T1241)
  Compare noise sources at z2248 optimal operating point:
  a) hwmon power (z2248 proven), b) gpu_metrics thermal, c) PM table hotspot,
  d) composite (all three IIR-filtered). All at noise_scale=0.05.

EXP 4 — Vg Calibration Sweep (T1242-T1245)
  Sweep BETA from 0.001 to 0.20 with gpu_metrics noise.
  Find the SR-optimal BETA for high-dimensional GPU coupling.
  z2249 used BETA=0.08 (too high with 36D); z2248 used 0.05*noise_state (scalar).

EXP 5 — Deep Stack with Calibrated Coupling (T1246-T1249)
  All GPU layers active (gpu_metrics+hwmon+PM) but at calibrated coupling.
  Compare DEEP_CALIBRATED vs SHALLOW (power-only) vs NO_GPU.

EXP 6 — Memory Capacity at Optimal Point (T1250-T1253)
  MC with z2248 optimal params (leak=0x02 τ=880ms).
  Compare COUPLED (power noise) vs DEEP (gpu_metrics noise at optimal BETA).

EXP 7 — Temporal XOR with GPU Features (T1254-T1257)
  XOR at delays d=1..5. Use gpu_metrics as additional readout.
  Target: gpu_features help temporal tasks where spike features plateau.

EXP 8 — Full Benchmark Scorecard (T1258-T1260)
  Best configs from EXP 1-7. Classification + memory + temporal.
  Compare against z2248 best (98.8%, MC=5.333).
"""

import sys, os, time, json, struct, warnings
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(line_buffering=True)

import numpy as np
from sklearn.linear_model import RidgeClassifier, Ridge
from sklearn.model_selection import StratifiedKFold, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from scipy import stats

warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fpga_host_eth import FPGAEthBridge

RESULTS_FILE = "results/z2250_calibrated_gpu_fpga.json"
N_NEURONS = 128
BASE_VG = 0.58
ALPHA = 0.25
SAMPLE_HZ = 20
STEP_INTERVAL = 1.0 / SAMPLE_HZ

# z2248 optimal params (SR peak)
OPT_LEAK = 0x0002      # τ≈880ms
OPT_THRESH = 0.50
OPT_BIAS_GAIN = 0.0625
OPT_NOISE = 0.05
OPT_DT_C = 0.0078
OPT_REFRACT = 50

# ─── GPU Access Layer (from z2249) ─────────────────────────────────────────

GPU_SYSFS = "/sys/class/drm/card0/device"
HWMON = "/sys/class/hwmon/hwmon7"
GPU_METRICS = f"{GPU_SYSFS}/gpu_metrics"
PM_TABLE = "/sys/kernel/ryzen_smu_drv/pm_table"


def read_gpu_metrics():
    """Parse gpu_metrics v3.0 binary into feature dict."""
    try:
        with open(GPU_METRICS, 'rb') as f:
            data = f.read()
        if len(data) < 100:
            return {}
        m = {}
        m['temp_gfx'] = struct.unpack_from('<H', data, 4)[0] / 100.0
        m['temp_soc'] = struct.unpack_from('<H', data, 6)[0] / 100.0
        m['dpm_flags'] = struct.unpack_from('<H', data, 62)[0]
        m['activity_gfx'] = struct.unpack_from('<H', data, 64)[0]
        m['voltage_soc'] = struct.unpack_from('<H', data, 94)[0]
        m['voltage_gfx'] = struct.unpack_from('<H', data, 96)[0]
        m['cur_sclk'] = struct.unpack_from('<H', data, 174)[0]
        m['cur_mclk'] = struct.unpack_from('<H', data, 176)[0]
        m['throttle_status'] = struct.unpack_from('<I', data, 108)[0]
        m['max_sclk'] = struct.unpack_from('<H', data, 224)[0]
        for i in range(130, 170, 2):
            val = struct.unpack_from('<H', data, i)[0]
            if val > 0:
                m[f'field_{i}'] = val
        return m
    except Exception:
        return {}


def read_hwmon():
    """Read hwmon7 telemetry."""
    m = {}
    for name, path in [
        ('power_uW', f'{HWMON}/power1_average'),
        ('temp_mC', f'{HWMON}/temp1_input'),
        ('freq_Hz', f'{HWMON}/freq1_input'),
        ('vddgfx_mV', f'{HWMON}/in0_input'),
        ('vddnb_mV', f'{HWMON}/in1_input'),
    ]:
        try:
            with open(path) as f:
                m[name] = int(f.read().strip())
        except Exception:
            m[name] = 0
    return m


def read_pm_table():
    """Read ryzen_smu PM table."""
    try:
        with open(PM_TABLE, 'rb') as f:
            data = f.read()
        m = {}
        if len(data) >= 0x50:
            m['hotspot_temp'] = struct.unpack_from('<f', data, 0x4C)[0]
        if len(data) >= 0x60:
            m['pm_power'] = struct.unpack_from('<f', data, 0x5C)[0]
        return m
    except Exception:
        return {}


def gpu_metrics_vector():
    """36D feature vector from all GPU sources (for readout, NOT modulation)."""
    gm = read_gpu_metrics()
    hw = read_hwmon()
    pm = read_pm_table()

    vec = []
    vec.append(gm.get('temp_gfx', 0) / 100.0)
    vec.append(gm.get('temp_soc', 0) / 100.0)
    vec.append(gm.get('activity_gfx', 0) / 100.0)
    vec.append(gm.get('voltage_soc', 0) / 10000.0)
    vec.append(gm.get('voltage_gfx', 0) / 10000.0)
    vec.append(gm.get('cur_sclk', 0) / 3000.0)
    vec.append(gm.get('cur_mclk', 0) / 3000.0)
    vec.append(gm.get('dpm_flags', 0) / 100.0)
    vec.append(gm.get('throttle_status', 0) / 1e6)
    for i in range(130, 170, 2):
        vec.append(gm.get(f'field_{i}', 0) / 10000.0)
    vec.append(hw.get('power_uW', 0) / 1e7)
    vec.append(hw.get('temp_mC', 0) / 1e5)
    vec.append(hw.get('freq_Hz', 0) / 3e9)
    vec.append(hw.get('vddgfx_mV', 0) / 2000.0)
    vec.append(hw.get('vddnb_mV', 0) / 2000.0)
    vec.append(pm.get('hotspot_temp', 0) / 100.0)
    vec.append(pm.get('pm_power', 0) / 100.0)
    return np.array(vec, dtype=np.float32)


def read_gpu_power():
    """Single scalar power reading (z2248 proven approach)."""
    try:
        with open(f"{HWMON}/power1_average", "r") as f:
            return float(f.read().strip()) / 1e6
    except Exception:
        return 11.0 + np.random.randn() * 0.5


# ─── FPGA Helpers ──────────────────────────────────────────────────────────

def configure_fpga(fpga, leak=None, bias_gain=None, thresh=None):
    """Configure FPGA with z2248 optimal params by default."""
    fpga.set_kill(False)
    time.sleep(0.05)
    fpga.set_leak_cond(leak if leak is not None else OPT_LEAK)
    fpga.set_bias_gain(bias_gain if bias_gain is not None else OPT_BIAS_GAIN)
    fpga.set_threshold(thresh if thresh is not None else OPT_THRESH)
    fpga.set_dt_over_c(OPT_DT_C)
    fpga.set_refract_cycles(OPT_REFRACT)
    time.sleep(0.1)
    vg_base = np.array([float(BASE_VG + 0.15 * (i/127 - 0.5)) for i in range(128)])
    fpga.set_vg_batch(0, vg_base.tolist())
    time.sleep(0.1)
    return vg_base


def collect_trial(fpga, input_signal, noise_scale, w_in, vg_base,
                  use_mac=True, noise_source='power', beta_vg=0.05,
                  capture_gpu=False):
    """
    Proven z2248 collection approach with configurable noise source.

    noise_source: 'power' (scalar hwmon), 'gpu_thermal' (gpu_metrics temp),
                  'pm_hotspot' (PM table), 'composite' (all three IIR-filtered),
                  'gpu_metrics' (36D but reduced to scalar via PCA-like projection)
    capture_gpu: if True, also capture gpu_metrics_vector per step (for readout)
    """
    n_steps = len(input_signal)
    all_spikes, all_delta, all_vmem = [], [], []
    gpu_readout = [] if capture_gpu else None

    power_base = read_gpu_power()
    noise_state = 0.0
    prev_spikes = None

    for step in range(n_steps):
        t0 = time.time()
        inp = float(input_signal[step])

        # Get noise from selected source
        if noise_source == 'power':
            raw_noise = (read_gpu_power() - power_base) / 5.0
        elif noise_source == 'gpu_thermal':
            gm = read_gpu_metrics()
            raw_noise = (gm.get('temp_soc', 27.0) - 27.0) / 10.0
        elif noise_source == 'pm_hotspot':
            pm = read_pm_table()
            raw_noise = (pm.get('hotspot_temp', 40.0) - 40.0) / 20.0
        elif noise_source == 'composite':
            p = (read_gpu_power() - power_base) / 5.0
            gm = read_gpu_metrics()
            t_noise = (gm.get('temp_soc', 27.0) - 27.0) / 10.0
            pm = read_pm_table()
            h_noise = (pm.get('hotspot_temp', 40.0) - 40.0) / 20.0
            raw_noise = (p + t_noise + h_noise) / 3.0
        elif noise_source == 'gpu_metrics':
            gvec = gpu_metrics_vector()
            raw_noise = float(np.mean(gvec[gvec != 0])) if np.any(gvec != 0) else 0.0
        else:
            raw_noise = 0.0

        # IIR filter (temporal memory in noise)
        noise_state = 0.85 * noise_state + 0.15 * raw_noise

        # MAC coupling (z2248 proven)
        if use_mac:
            mac_val = inp + noise_state * noise_scale
            fpga.set_mac_signal(float(np.clip(mac_val * 0.5 + 0.5, 0.0, 1.0)))
        else:
            fpga.set_mac_signal(0.5)

        # Vg modulation: input + SMALL noise (not 36D overdrive!)
        vg_mod = vg_base + ALPHA * inp + beta_vg * w_in * inp + noise_scale * beta_vg * noise_state
        fpga.set_vg_batch(0, [float(np.clip(v, 0.3, 0.9)) for v in vg_mod])

        # Capture GPU features for readout (NOT for modulation)
        if capture_gpu:
            gpu_readout.append(gpu_metrics_vector().copy())

        # Read telemetry
        telem = fpga.read_telemetry(timeout=0.15)
        if telem is not None:
            sc = telem['spike_counts'].astype(np.float32)
            vm = telem['vmem'].copy()
            delta = sc - prev_spikes if prev_spikes is not None else np.zeros(N_NEURONS, dtype=np.float32)
            all_spikes.append(sc)
            all_delta.append(delta)
            all_vmem.append(vm)
            prev_spikes = sc.copy()
        else:
            all_spikes.append(all_spikes[-1].copy() if all_spikes else np.zeros(N_NEURONS, dtype=np.float32))
            all_delta.append(np.zeros(N_NEURONS, dtype=np.float32))
            all_vmem.append(all_vmem[-1].copy() if all_vmem else np.zeros(N_NEURONS, dtype=np.float32))

        elapsed = time.time() - t0
        if elapsed < STEP_INTERVAL:
            time.sleep(STEP_INTERVAL - elapsed)

    spikes = np.array(all_spikes)
    delta = np.array(all_delta)
    vmem = np.array(all_vmem)
    gpu_feat = np.array(gpu_readout) if capture_gpu else None
    return spikes, delta, vmem, gpu_feat


def build_features(spikes, delta, vmem):
    """z2248 proven feature extraction."""
    return np.concatenate([
        spikes.mean(axis=0), spikes.std(axis=0),
        delta.mean(axis=0), delta.std(axis=0),
        vmem.mean(axis=0), vmem[-1],
        spikes[-1] - spikes[0],
    ])


def build_features_with_gpu(spikes, delta, vmem, gpu_feat):
    """Features augmented with GPU telemetry readout."""
    base = build_features(spikes, delta, vmem)
    if gpu_feat is not None and len(gpu_feat) > 0:
        gpu_stats = np.concatenate([
            gpu_feat.mean(axis=0),
            gpu_feat.std(axis=0),
            gpu_feat[-1] - gpu_feat[0],  # temporal derivative
        ])
        return np.concatenate([base, gpu_stats])
    return base


def ridge_classify(X, y, n_splits=5):
    """Cross-validated ridge classification."""
    std = X.std(axis=0)
    mask = std > 1e-2
    if mask.sum() < 3:
        return 0.0
    scaler = StandardScaler()
    X_f = scaler.fit_transform(X[:, mask])
    classes, counts = np.unique(y, return_counts=True)
    if counts.min() < n_splits:
        n_splits = max(2, counts.min())
    clf = RidgeClassifier(alpha=1.0)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    scores = cross_val_score(clf, X_f, y, cv=skf, scoring='accuracy')
    return float(scores.mean())


def ridge_regress(X, y, n_splits=5):
    """Cross-validated ridge regression."""
    std = X.std(axis=0)
    mask = std > 1e-2
    if mask.sum() < 3:
        return 0.0
    scaler = StandardScaler()
    X_f = scaler.fit_transform(X[:, mask])
    reg = Ridge(alpha=1.0)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    scores = cross_val_score(reg, X_f, y, cv=kf, scoring='r2')
    return float(scores.mean())


def make_waveform_data(n_trials=120, n_steps=60, n_classes=4, seed=42):
    """Generate waveform classification dataset."""
    rng = np.random.RandomState(seed)
    signals, labels = [], []
    t = np.linspace(0, 2*np.pi, n_steps)
    for _ in range(n_trials):
        c = rng.randint(n_classes)
        if c == 0:
            s = np.sin(t)
        elif c == 1:
            s = np.sign(np.sin(t))
        elif c == 2:
            s = np.linspace(-1, 1, n_steps)
        else:
            s = np.sin(2*t) * np.sin(t)
        s += rng.randn(n_steps) * 0.1
        signals.append(s)
        labels.append(c)
    return signals, np.array(labels)


def make_xor_data(n_trials=80, n_steps=60, delay=1, seed=42):
    """XOR temporal task."""
    rng = np.random.RandomState(seed)
    signals, labels = [], []
    for _ in range(n_trials):
        seq = rng.choice([-1, 1], size=n_steps + delay)
        xor_target = (seq[:n_steps] * seq[delay:delay+n_steps] > 0).astype(int)
        label = int(xor_target.sum() > n_steps // 2)
        signals.append(seq[:n_steps].astype(float))
        labels.append(label)
    return signals, np.array(labels)


def make_mc_data(n_trials=80, n_steps=60, delay=0, seed=42):
    """Memory capacity: classify delayed random binary."""
    rng = np.random.RandomState(seed)
    signals, labels = [], []
    for _ in range(n_trials):
        full = rng.choice([-1, 1], size=n_steps + delay + 1)
        signal = full[delay+1:].astype(float)
        label = 1 if full[0] > 0 else 0
        signals.append(signal)
        labels.append(label)
    return signals, np.array(labels)


def convert_for_json(obj):
    """Convert numpy types to JSON-serializable Python types."""
    if isinstance(obj, dict):
        return {k: convert_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_for_json(v) for v in obj]
    elif isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        return float(obj)
    elif isinstance(obj, (np.bool_,)):
        return bool(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("z2250: Calibrated GPU-FPGA Deep Integration")
    print("=" * 60)

    results = {
        'experiment': 'z2250_calibrated_gpu_fpga',
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
    }

    rng = np.random.RandomState(42)
    w_in = rng.randn(N_NEURONS) * 0.3

    # Connect FPGA
    fpga = FPGAEthBridge()
    fpga.connect()
    print("FPGA connected")

    pass_count = 0
    total_count = 0

    # ───────────────────────────────────────────────────────────────────────
    # EXP 1: Baseline Sanity — Confirm z2248 optimal params
    # ───────────────────────────────────────────────────────────────────────
    print("\n" + "─" * 60)
    print("EXP 1: Baseline Sanity (z2248 optimal params)")
    print("─" * 60)

    vg_base = configure_fpga(fpga, leak=OPT_LEAK, bias_gain=OPT_BIAS_GAIN)
    signals, labels = make_waveform_data(n_trials=80, n_steps=50)

    X_coupled = []
    for i, sig in enumerate(signals):
        sp, dl, vm, _ = collect_trial(fpga, sig, OPT_NOISE, w_in, vg_base)
        X_coupled.append(build_features(sp, dl, vm))
        if (i+1) % 20 == 0:
            print(f"  Trial {i+1}/{len(signals)}")
    X_coupled = np.array(X_coupled)
    acc_baseline = ridge_classify(X_coupled, labels)
    print(f"  Baseline accuracy: {acc_baseline*100:.1f}%")

    # Static control (no noise)
    X_static = []
    for i, sig in enumerate(signals):
        sp, dl, vm, _ = collect_trial(fpga, sig, 0.0, w_in, vg_base, use_mac=False)
        X_static.append(build_features(sp, dl, vm))
        if (i+1) % 20 == 0:
            print(f"  Static {i+1}/{len(signals)}")
    X_static = np.array(X_static)
    acc_static = ridge_classify(X_static, labels)
    print(f"  Static accuracy: {acc_static*100:.1f}%")

    t1231 = acc_baseline > 0.80
    t1232 = acc_baseline > acc_static
    t1233 = acc_baseline > 0.90

    results['exp1'] = {
        'acc_baseline': acc_baseline, 'acc_static': acc_static,
        'T1231': t1231, 'T1232': t1232, 'T1233': t1233,
    }
    for t, v in [('T1231', t1231), ('T1232', t1232), ('T1233', t1233)]:
        r = "PASS" if v else "FAIL"
        total_count += 1
        if v: pass_count += 1
        print(f"  {t}: {r}")

    # ───────────────────────────────────────────────────────────────────────
    # EXP 2: GPU Features as Readout
    # ───────────────────────────────────────────────────────────────────────
    print("\n" + "─" * 60)
    print("EXP 2: GPU Features as Readout (NOT input modulation)")
    print("─" * 60)

    vg_base = configure_fpga(fpga)
    signals, labels = make_waveform_data(n_trials=80, n_steps=50, seed=99)

    X_spike_only = []
    X_spike_gpu = []
    X_gpu_only = []

    for i, sig in enumerate(signals):
        sp, dl, vm, gf = collect_trial(fpga, sig, OPT_NOISE, w_in, vg_base,
                                        capture_gpu=True)
        X_spike_only.append(build_features(sp, dl, vm))
        X_spike_gpu.append(build_features_with_gpu(sp, dl, vm, gf))
        if gf is not None:
            X_gpu_only.append(np.concatenate([gf.mean(axis=0), gf.std(axis=0), gf[-1]-gf[0]]))
        if (i+1) % 20 == 0:
            print(f"  Trial {i+1}/{len(signals)}")

    X_spike_only = np.array(X_spike_only)
    X_spike_gpu = np.array(X_spike_gpu)
    X_gpu_only = np.array(X_gpu_only)

    acc_spike = ridge_classify(X_spike_only, labels)
    acc_spike_gpu = ridge_classify(X_spike_gpu, labels)
    acc_gpu = ridge_classify(X_gpu_only, labels) if len(X_gpu_only) > 0 else 0.0

    print(f"  Spike-only: {acc_spike*100:.1f}%")
    print(f"  Spike+GPU:  {acc_spike_gpu*100:.1f}%")
    print(f"  GPU-only:   {acc_gpu*100:.1f}%")

    t1234 = acc_spike > 0.80
    t1235 = acc_spike_gpu >= acc_spike
    t1236 = acc_spike_gpu > acc_gpu
    t1237 = acc_gpu < acc_spike  # GPU alone should be worse (no reservoir)

    results['exp2'] = {
        'acc_spike': acc_spike, 'acc_spike_gpu': acc_spike_gpu, 'acc_gpu': acc_gpu,
        'T1234': t1234, 'T1235': t1235, 'T1236': t1236, 'T1237': t1237,
    }
    for t, v in [('T1234', t1234), ('T1235', t1235), ('T1236', t1236), ('T1237', t1237)]:
        r = "PASS" if v else "FAIL"
        total_count += 1
        if v: pass_count += 1
        print(f"  {t}: {r}")

    # ───────────────────────────────────────────────────────────────────────
    # EXP 3: Multi-Source Noise Comparison
    # ───────────────────────────────────────────────────────────────────────
    print("\n" + "─" * 60)
    print("EXP 3: Multi-Source Noise Comparison")
    print("─" * 60)

    vg_base = configure_fpga(fpga)
    signals, labels = make_waveform_data(n_trials=80, n_steps=50, seed=77)

    noise_sources = ['power', 'gpu_thermal', 'pm_hotspot', 'composite']
    noise_accs = {}

    for ns in noise_sources:
        print(f"  Testing noise source: {ns}")
        X_ns = []
        for i, sig in enumerate(signals):
            sp, dl, vm, _ = collect_trial(fpga, sig, OPT_NOISE, w_in, vg_base,
                                           noise_source=ns)
            X_ns.append(build_features(sp, dl, vm))
            if (i+1) % 20 == 0:
                print(f"    Trial {i+1}/{len(signals)}")
        X_ns = np.array(X_ns)
        acc = ridge_classify(X_ns, labels)
        noise_accs[ns] = acc
        print(f"    {ns}: {acc*100:.1f}%")

    best_source = max(noise_accs, key=noise_accs.get)
    t1238 = noise_accs['power'] > 0.80          # proven source still works
    t1239 = max(noise_accs.values()) > 0.85      # at least one source >85%
    t1240 = noise_accs['composite'] >= noise_accs['power'] * 0.95  # composite competitive
    t1241 = len([v for v in noise_accs.values() if v > 0.70]) >= 3  # 3+ sources above 70%

    results['exp3'] = {
        'noise_accs': noise_accs, 'best_source': best_source,
        'T1238': t1238, 'T1239': t1239, 'T1240': t1240, 'T1241': t1241,
    }
    for t, v in [('T1238', t1238), ('T1239', t1239), ('T1240', t1240), ('T1241', t1241)]:
        r = "PASS" if v else "FAIL"
        total_count += 1
        if v: pass_count += 1
        print(f"  {t}: {r}")

    # ───────────────────────────────────────────────────────────────────────
    # EXP 4: Vg BETA Calibration Sweep
    # ───────────────────────────────────────────────────────────────────────
    print("\n" + "─" * 60)
    print("EXP 4: BETA Calibration Sweep (gpu_metrics noise)")
    print("─" * 60)

    vg_base = configure_fpga(fpga)
    signals, labels = make_waveform_data(n_trials=60, n_steps=50, seed=55)

    betas = [0.001, 0.005, 0.01, 0.02, 0.05, 0.08, 0.15, 0.25]
    beta_accs = {}

    for beta in betas:
        X_b = []
        for sig in signals:
            sp, dl, vm, _ = collect_trial(fpga, sig, OPT_NOISE, w_in, vg_base,
                                           noise_source='gpu_metrics', beta_vg=beta)
            X_b.append(build_features(sp, dl, vm))
        X_b = np.array(X_b)
        acc = ridge_classify(X_b, labels)
        beta_accs[beta] = acc
        print(f"  BETA={beta:.3f}: {acc*100:.1f}%")

    best_beta = max(beta_accs, key=beta_accs.get)
    worst_beta = min(beta_accs, key=beta_accs.get)

    t1242 = best_beta <= 0.05           # optimal BETA should be small
    t1243 = beta_accs[best_beta] > 0.80  # best BETA above 80%
    t1244 = beta_accs[best_beta] > beta_accs[0.08] + 0.01  # better than z2249's 0.08
    t1245 = beta_accs[0.25] < beta_accs[best_beta]  # high BETA worse than optimal

    results['exp4'] = {
        'beta_accs': {str(k): v for k, v in beta_accs.items()},
        'best_beta': best_beta, 'worst_beta': worst_beta,
        'T1242': t1242, 'T1243': t1243, 'T1244': t1244, 'T1245': t1245,
    }
    for t, v in [('T1242', t1242), ('T1243', t1243), ('T1244', t1244), ('T1245', t1245)]:
        r = "PASS" if v else "FAIL"
        total_count += 1
        if v: pass_count += 1
        print(f"  {t}: {r}")

    # ───────────────────────────────────────────────────────────────────────
    # EXP 5: Deep Stack with Calibrated Coupling
    # ───────────────────────────────────────────────────────────────────────
    print("\n" + "─" * 60)
    print("EXP 5: Deep Stack (calibrated coupling)")
    print("─" * 60)

    vg_base = configure_fpga(fpga)
    signals, labels = make_waveform_data(n_trials=80, n_steps=50, seed=33)

    conditions = {
        'DEEP': {'noise_source': 'composite', 'beta_vg': best_beta, 'capture_gpu': True},
        'SHALLOW': {'noise_source': 'power', 'beta_vg': 0.05, 'capture_gpu': False},
        'NO_GPU': {'noise_source': 'power', 'beta_vg': 0.0, 'capture_gpu': False},
    }

    cond_accs = {}
    for name, params in conditions.items():
        print(f"  Condition: {name}")
        X_c = []
        for i, sig in enumerate(signals):
            sp, dl, vm, gf = collect_trial(fpga, sig, OPT_NOISE, w_in, vg_base,
                                            noise_source=params['noise_source'],
                                            beta_vg=params['beta_vg'],
                                            capture_gpu=params['capture_gpu'])
            if params['capture_gpu']:
                X_c.append(build_features_with_gpu(sp, dl, vm, gf))
            else:
                X_c.append(build_features(sp, dl, vm))
            if (i+1) % 20 == 0:
                print(f"    Trial {i+1}/{len(signals)}")
        X_c = np.array(X_c)
        acc = ridge_classify(X_c, labels)
        cond_accs[name] = acc
        print(f"    {name}: {acc*100:.1f}%")

    t1246 = cond_accs['DEEP'] > 0.80
    t1247 = cond_accs['DEEP'] >= cond_accs['SHALLOW']
    t1248 = cond_accs['DEEP'] > cond_accs['NO_GPU']
    t1249 = cond_accs['SHALLOW'] > cond_accs['NO_GPU']

    results['exp5'] = {
        'cond_accs': cond_accs,
        'T1246': t1246, 'T1247': t1247, 'T1248': t1248, 'T1249': t1249,
    }
    for t, v in [('T1246', t1246), ('T1247', t1247), ('T1248', t1248), ('T1249', t1249)]:
        r = "PASS" if v else "FAIL"
        total_count += 1
        if v: pass_count += 1
        print(f"  {t}: {r}")

    # ───────────────────────────────────────────────────────────────────────
    # EXP 6: Memory Capacity at Optimal Point
    # ───────────────────────────────────────────────────────────────────────
    print("\n" + "─" * 60)
    print("EXP 6: Memory Capacity (z2248 optimal + GPU readout)")
    print("─" * 60)

    vg_base = configure_fpga(fpga)
    delays = [0, 1, 2, 3, 5, 7, 10]
    mc_coupled = {}
    mc_deep = {}

    for d in delays:
        # COUPLED (power noise, proven)
        sigs, labs = make_mc_data(n_trials=80, n_steps=50, delay=d)
        X_c = []
        for sig in sigs:
            sp, dl, vm, _ = collect_trial(fpga, sig, OPT_NOISE, w_in, vg_base)
            X_c.append(build_features(sp, dl, vm))
        X_c = np.array(X_c)
        acc_c = ridge_classify(X_c, labs)
        mc_coupled[d] = acc_c

        # DEEP (power noise + GPU readout features)
        X_d = []
        for sig in sigs:
            sp, dl, vm, gf = collect_trial(fpga, sig, OPT_NOISE, w_in, vg_base,
                                            capture_gpu=True)
            X_d.append(build_features_with_gpu(sp, dl, vm, gf))
        X_d = np.array(X_d)
        acc_d = ridge_classify(X_d, labs)
        mc_deep[d] = acc_d

        print(f"  d={d}: COUPLED={acc_c*100:.1f}%, DEEP={acc_d*100:.1f}%")

    mc_total_coupled = sum(max(0, 2*(v-0.5)) for v in mc_coupled.values())
    mc_total_deep = sum(max(0, 2*(v-0.5)) for v in mc_deep.values())

    t1250 = mc_coupled[0] > 0.70         # d=0 strong
    t1251 = mc_coupled[1] > 0.60         # d=1 above threshold
    t1252 = mc_total_coupled > 3.0       # total MC > 3 (z2248 got 5.333)
    t1253 = mc_total_deep >= mc_total_coupled * 0.95  # GPU readout doesn't hurt MC

    results['exp6'] = {
        'mc_coupled': {str(k): v for k, v in mc_coupled.items()},
        'mc_deep': {str(k): v for k, v in mc_deep.items()},
        'mc_total_coupled': mc_total_coupled,
        'mc_total_deep': mc_total_deep,
        'T1250': t1250, 'T1251': t1251, 'T1252': t1252, 'T1253': t1253,
    }
    for t, v in [('T1250', t1250), ('T1251', t1251), ('T1252', t1252), ('T1253', t1253)]:
        r = "PASS" if v else "FAIL"
        total_count += 1
        if v: pass_count += 1
        print(f"  {t}: {r}")

    # ───────────────────────────────────────────────────────────────────────
    # EXP 7: Temporal XOR with GPU Features
    # ───────────────────────────────────────────────────────────────────────
    print("\n" + "─" * 60)
    print("EXP 7: Temporal XOR with GPU Readout")
    print("─" * 60)

    vg_base = configure_fpga(fpga)
    xor_delays = [1, 2, 3, 5]
    xor_spike = {}
    xor_deep = {}

    for d in xor_delays:
        sigs, labs = make_xor_data(n_trials=80, n_steps=50, delay=d)
        X_s, X_d = [], []
        for sig in sigs:
            sp, dl, vm, gf = collect_trial(fpga, sig, OPT_NOISE, w_in, vg_base,
                                            capture_gpu=True)
            X_s.append(build_features(sp, dl, vm))
            X_d.append(build_features_with_gpu(sp, dl, vm, gf))

        acc_s = ridge_classify(np.array(X_s), labs)
        acc_d = ridge_classify(np.array(X_d), labs)
        xor_spike[d] = acc_s
        xor_deep[d] = acc_d
        print(f"  d={d}: Spike={acc_s*100:.1f}%, Deep={acc_d*100:.1f}%")

    t1254 = xor_spike[1] > 0.55           # XOR d=1 above chance+5%
    t1255 = any(xor_deep[d] > xor_spike[d] for d in xor_delays)  # GPU helps at some delay
    t1256 = xor_spike[1] > xor_spike[5]   # memory fading with delay
    t1257 = max(xor_deep.values()) > 0.55  # deep approach helps somewhere

    results['exp7'] = {
        'xor_spike': {str(k): v for k, v in xor_spike.items()},
        'xor_deep': {str(k): v for k, v in xor_deep.items()},
        'T1254': t1254, 'T1255': t1255, 'T1256': t1256, 'T1257': t1257,
    }
    for t, v in [('T1254', t1254), ('T1255', t1255), ('T1256', t1256), ('T1257', t1257)]:
        r = "PASS" if v else "FAIL"
        total_count += 1
        if v: pass_count += 1
        print(f"  {t}: {r}")

    # ───────────────────────────────────────────────────────────────────────
    # EXP 8: Full Benchmark Scorecard
    # ───────────────────────────────────────────────────────────────────────
    print("\n" + "─" * 60)
    print("EXP 8: Full Benchmark Scorecard")
    print("─" * 60)

    # Use best configs from previous experiments
    best_noise_source = best_source  # from EXP 3
    print(f"  Best noise source: {best_noise_source}")
    print(f"  Best BETA: {best_beta}")

    # Final classification with best config
    vg_base = configure_fpga(fpga)
    signals, labels = make_waveform_data(n_trials=100, n_steps=50, seed=111)

    X_best = []
    for i, sig in enumerate(signals):
        sp, dl, vm, gf = collect_trial(fpga, sig, OPT_NOISE, w_in, vg_base,
                                        noise_source=best_noise_source,
                                        beta_vg=best_beta, capture_gpu=True)
        X_best.append(build_features_with_gpu(sp, dl, vm, gf))
        if (i+1) % 25 == 0:
            print(f"  Trial {i+1}/{len(signals)}")
    X_best = np.array(X_best)
    acc_final = ridge_classify(X_best, labels)
    print(f"  Final classification: {acc_final*100:.1f}%")

    t1258 = acc_final > 0.90               # final score >90%
    t1259 = acc_final > acc_baseline * 0.95  # competitive with EXP 1 baseline
    t1260 = mc_total_coupled > 3.0 and acc_final > 0.85  # both MC and classification strong

    results['exp8'] = {
        'acc_final': acc_final,
        'best_noise_source': best_noise_source,
        'best_beta': best_beta,
        'mc_total_coupled': mc_total_coupled,
        'T1258': t1258, 'T1259': t1259, 'T1260': t1260,
    }
    for t, v in [('T1258', t1258), ('T1259', t1259), ('T1260', t1260)]:
        r = "PASS" if v else "FAIL"
        total_count += 1
        if v: pass_count += 1
        print(f"  {t}: {r}")

    # ───────────────────────────────────────────────────────────────────────
    # Summary
    # ───────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    results['pass_count'] = pass_count
    results['total_count'] = total_count
    print(f"  Total: {pass_count}/{total_count} PASS")

    # Save results
    os.makedirs(os.path.dirname(RESULTS_FILE), exist_ok=True)
    with open(RESULTS_FILE, 'w') as f:
        json.dump(convert_for_json(results), f, indent=2)
    print(f"  Saved to {RESULTS_FILE}")

    fpga.close()
    print("Done.")


if __name__ == '__main__':
    main()
