#!/usr/bin/env python3
"""
z2249: Deep GPU-FPGA Integration — Firmware-Level Coupling
===========================================================
Push GPU access as deep as possible (PM4-level, DPM forcing, gpu_metrics
binary parsing, clock/power manipulation) and couple to FPGA reservoir.

8 experiments, 35 tests (T1196-T1230):

EXP 1 — GPU Metrics Deep Parse (T1196-T1199)
  Parse full 264-byte gpu_metrics v3.0 binary at high rate.
  Extract thermal, voltage, DPM state, clock gating flags.
  Use as 30+ dimensional feature vector for reservoir input.

EXP 2 — DPM State Forcing (T1200-T1203)
  Force GPU between DPM states (600/1100/2900 MHz) via pp_dpm_sclk.
  Measure FPGA response to DPM transitions.
  Classify DPM state from spike patterns alone.

EXP 3 — Power Dynamics as Noise Source (T1204-T1207)
  Use hwmon power1_average as real-time noise injection.
  Compare power-driven noise vs synthetic 1/f vs white noise.
  Power fluctuations are native 1/f from VRM switching.

EXP 4 — Clock Gating Coupling (T1208-T1211)
  Toggle GPU compute load to change clock gating state.
  Measure how CG transitions affect power→FPGA coupling.
  GFXOFF transitions create sharp power transients.

EXP 5 — Multi-Layer Firmware Telemetry (T1212-T1215)
  Fuse gpu_metrics + hwmon + PM table (ryzen_smu) into
  multi-scale firmware feature vector.
  Compare single-layer vs multi-layer as reservoir input.

EXP 6 — PM4-Level Register Probing (T1216-T1219)
  Read GPU register state via debugfs amdgpu_regs_smc.
  Extract SMC firmware state as additional input features.
  Deepest read-only GPU access without kernel modification.

EXP 7 — Cross-Substrate Temporal Encoding (T1220-T1225)
  Encode temporal patterns in DPM state sequences.
  FPGA decodes GPU state history from spike dynamics.
  Tests information flow from firmware→FPGA→classification.

EXP 8 — Full Deep Stack Benchmark (T1226-T1230)
  All layers active: gpu_metrics + DPM + power + FPGA reservoir.
  Compare against shallow (hwmon only) and no-GPU baselines.
  Target: deep > shallow > no-GPU.
"""

import sys, os, time, json, struct, socket, subprocess
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from fpga_host_eth import FPGAEthBridge

# ─── GPU Access Layer ───────────────────────────────────────────────────────

GPU_SYSFS = "/sys/class/drm/card0/device"
HWMON = "/sys/class/hwmon/hwmon7"
GPU_METRICS = f"{GPU_SYSFS}/gpu_metrics"
PP_DPM_SCLK = f"{GPU_SYSFS}/pp_dpm_sclk"
DPM_FORCE = f"{GPU_SYSFS}/power_dpm_force_performance_level"
PM_TABLE = "/sys/kernel/ryzen_smu_drv/pm_table"

def read_gpu_metrics():
    """Parse 264-byte gpu_metrics v3.0 binary into feature dict."""
    try:
        with open(GPU_METRICS, 'rb') as f:
            data = f.read()
        if len(data) < 100:
            return {}

        m = {}
        m['temp_gfx'] = struct.unpack_from('<H', data, 4)[0] / 100.0
        m['temp_soc'] = struct.unpack_from('<H', data, 6)[0] / 100.0
        # DPM state indicators at various offsets
        m['dpm_flags'] = struct.unpack_from('<H', data, 62)[0]
        m['activity_gfx'] = struct.unpack_from('<H', data, 64)[0]
        # Voltage/power fields
        m['voltage_soc'] = struct.unpack_from('<H', data, 94)[0]
        m['voltage_gfx'] = struct.unpack_from('<H', data, 96)[0]
        # Current clocks
        m['cur_sclk'] = struct.unpack_from('<H', data, 174)[0]
        m['cur_mclk'] = struct.unpack_from('<H', data, 176)[0]
        # Throttle/power status
        m['throttle_status'] = struct.unpack_from('<I', data, 108)[0]
        # Max clocks for normalization
        m['max_sclk'] = struct.unpack_from('<H', data, 224)[0]
        # Padding/additional features from raw bytes
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
    """Read ryzen_smu PM table for deep thermal/power data."""
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

def force_dpm_state(state_idx):
    """Force GPU to specific DPM state (0=600MHz, 1=1100MHz, 2=2900MHz)."""
    try:
        with open(DPM_FORCE, 'w') as f:
            f.write('manual')
        with open(PP_DPM_SCLK, 'w') as f:
            f.write(str(state_idx))
        return True
    except PermissionError:
        return False

def release_dpm():
    """Release DPM back to auto."""
    try:
        with open(DPM_FORCE, 'w') as f:
            f.write('auto')
    except Exception:
        pass

def get_current_dpm():
    """Get current DPM state index."""
    try:
        with open(PP_DPM_SCLK) as f:
            for line in f:
                if '*' in line:
                    return int(line.split(':')[0].strip())
    except Exception:
        pass
    return -1

def gpu_metrics_vector():
    """Get a high-dimensional feature vector from all GPU sources."""
    gm = read_gpu_metrics()
    hw = read_hwmon()
    pm = read_pm_table()

    vec = []
    # gpu_metrics fields (normalized)
    vec.append(gm.get('temp_gfx', 0) / 100.0)
    vec.append(gm.get('temp_soc', 0) / 100.0)
    vec.append(gm.get('activity_gfx', 0) / 100.0)
    vec.append(gm.get('voltage_soc', 0) / 10000.0)
    vec.append(gm.get('voltage_gfx', 0) / 10000.0)
    vec.append(gm.get('cur_sclk', 0) / 3000.0)
    vec.append(gm.get('cur_mclk', 0) / 3000.0)
    vec.append(gm.get('dpm_flags', 0) / 100.0)
    vec.append(gm.get('throttle_status', 0) / 1e6)
    # Dynamic fields from gpu_metrics
    for i in range(130, 170, 2):
        vec.append(gm.get(f'field_{i}', 0) / 10000.0)
    # hwmon fields
    vec.append(hw.get('power_uW', 0) / 1e7)  # normalize ~10W
    vec.append(hw.get('temp_mC', 0) / 1e5)
    vec.append(hw.get('freq_Hz', 0) / 3e9)
    vec.append(hw.get('vddgfx_mV', 0) / 2000.0)
    vec.append(hw.get('vddnb_mV', 0) / 2000.0)
    # PM table
    vec.append(pm.get('hotspot_temp', 0) / 100.0)
    vec.append(pm.get('pm_power', 0) / 100.0)

    return np.array(vec, dtype=np.float32)


# ─── FPGA Helpers ───────────────────────────────────────────────────────────

TUNED_LEAK = 0x0011
TUNED_THRESH = 0.50
TUNED_BIAS_GAIN = 0.03125
BASE_VG = 0.58
VG_SPREAD = 0.075
N_NEURONS = 128
ALPHA = 0.25
BETA = 0.08

def configure_fpga(fpga, leak=None, bias_gain=None, thresh=None):
    """Configure FPGA parameters via Ethernet commands."""
    if leak is not None:
        fpga.set_leak_cond(leak)
        time.sleep(0.05)
    if bias_gain is not None:
        fpga.set_bias_gain(bias_gain)
        time.sleep(0.05)
    if thresh is not None:
        fpga.set_threshold(thresh)
        time.sleep(0.05)

def collect_reservoir(fpga, input_signal, noise_source='gpu_metrics', w_in=None,
                      vg_base=None, use_mac=True, sample_hz=20):
    """Collect reservoir states with configurable noise source."""
    n_steps = len(input_signal)
    if w_in is None:
        rng = np.random.RandomState(42)
        w_in = rng.randn(N_NEURONS) * 0.3
    if vg_base is None:
        rng = np.random.RandomState(123)
        vg_base = BASE_VG + rng.uniform(-VG_SPREAD, VG_SPREAD, N_NEURONS)

    spike_matrix = np.zeros((n_steps, N_NEURONS))
    vmem_matrix = np.zeros((n_steps, N_NEURONS))
    gpu_features = []
    dt = 1.0 / sample_hz

    for step in range(n_steps):
        x = input_signal[step]

        # Get noise from selected source
        if noise_source == 'gpu_metrics':
            gvec = gpu_metrics_vector()
            noise = gvec[:N_NEURONS] if len(gvec) >= N_NEURONS else np.pad(gvec, (0, max(0, N_NEURONS - len(gvec))))
            gpu_features.append(gvec.copy())
        elif noise_source == 'hwmon_power':
            hw = read_hwmon()
            power_w = hw['power_uW'] / 1e6
            noise = np.full(N_NEURONS, power_w / 15.0)  # normalize around ~10W
        elif noise_source == 'white':
            noise = np.random.randn(N_NEURONS) * 0.05
        elif noise_source == 'none':
            noise = np.zeros(N_NEURONS)
        else:
            noise = np.zeros(N_NEURONS)

        # Modulate Vg
        vg_new = vg_base + ALPHA * x * w_in + BETA * noise[:N_NEURONS]
        vg_new = np.clip(vg_new, 0.01, 0.99)

        # Send to FPGA
        if use_mac:
            mac_val = float(np.clip(x, -1, 1))
            fpga.set_mac_signal(mac_val)

        # Send Vg batch
        for nid in range(0, N_NEURONS, 8):
            batch = vg_new[nid:nid+8].tolist()
            fpga.set_vg_batch(nid, batch)

        # Read telemetry
        time.sleep(dt * 0.8)
        t = fpga.read_telemetry()
        if t and 'spike_counts' in t:
            spike_matrix[step] = np.array(t['spike_counts'][:N_NEURONS])
            vmem_matrix[step] = np.array(t['vmem'][:N_NEURONS]) / 65536.0

        if step % 20 == 0 and step > 0:
            print(f"    Trial {step}/{n_steps}")

    return spike_matrix, vmem_matrix, np.array(gpu_features) if gpu_features else None

def ridge_classify(X_train, y_train, X_test, y_test, alpha=1.0):
    """Ridge regression classifier."""
    from sklearn.linear_model import RidgeClassifier
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_train)
    X_te = scaler.transform(X_test)

    clf = RidgeClassifier(alpha=alpha)
    clf.fit(X_tr, y_train)
    return clf.score(X_te, y_test)

def make_waveform_data(n_trials=120, n_steps=60, n_classes=4, seed=42):
    """Generate waveform classification dataset."""
    rng = np.random.RandomState(seed)
    signals = []
    labels = []
    t = np.linspace(0, 2*np.pi, n_steps)

    for _ in range(n_trials):
        c = rng.randint(n_classes)
        if c == 0:
            s = np.sin(t)
        elif c == 1:
            s = np.sign(np.sin(t))
        elif c == 2:
            s = np.abs(np.sin(t)) * 2 - 1
        else:
            s = (t / (2*np.pi)) * 2 - 1
        s += rng.randn(n_steps) * 0.1
        signals.append(s)
        labels.append(c)

    return signals, np.array(labels)


# ─── Experiments ────────────────────────────────────────────────────────────

def exp1_gpu_metrics_deep(fpga, results):
    """EXP 1: GPU Metrics Deep Parse — high-rate multi-dimensional sampling."""
    print("\n=== EXP 1: GPU Metrics Deep Parse ===")

    # Sample gpu_metrics at max rate for 5 seconds
    samples = []
    t0 = time.time()
    while time.time() - t0 < 5.0:
        gm = read_gpu_metrics()
        hw = read_hwmon()
        gm['_hw_power'] = hw.get('power_uW', 0)
        gm['_hw_temp'] = hw.get('temp_mC', 0)
        gm['_time'] = time.time() - t0
        samples.append(gm)

    rate = len(samples) / 5.0
    n_fields = len([k for k in samples[0] if not k.startswith('_')])

    print(f"  Sampling rate: {rate:.0f} Hz")
    print(f"  Fields parsed: {n_fields}")

    # Check power dynamics
    powers = [s['_hw_power'] for s in samples]
    power_std = np.std(powers) / np.mean(powers) if np.mean(powers) > 0 else 0
    print(f"  Power CV: {power_std:.4f}")

    # Check temperature variation
    temps = [s.get('temp_gfx', 0) for s in samples]
    temp_range = max(temps) - min(temps)
    print(f"  Temp range: {temp_range:.2f}°C")

    # Feature vector dimensionality
    vec = gpu_metrics_vector()
    n_nonzero = np.count_nonzero(vec)
    print(f"  Feature vector: {len(vec)}D, {n_nonzero} non-zero")

    # T1196: Sampling rate > 100 Hz
    t1196 = rate > 100
    print(f"  T1196_sample_rate_gt_100hz: {'PASS' if t1196 else 'FAIL'} ({rate:.0f} Hz)")

    # T1197: >10 non-zero features
    t1197 = n_nonzero > 10
    print(f"  T1197_features_gt_10: {'PASS' if t1197 else 'FAIL'} ({n_nonzero})")

    # T1198: Power has measurable dynamics
    t1198 = power_std > 0.001
    print(f"  T1198_power_dynamics: {'PASS' if t1198 else 'FAIL'} (CV={power_std:.4f})")

    # T1199: Temperature readable
    t1199 = temps[0] > 10 and temps[0] < 120
    print(f"  T1199_temp_readable: {'PASS' if t1199 else 'FAIL'} ({temps[0]:.1f}°C)")

    results['exp1'] = {
        'rate_hz': rate, 'n_fields': n_fields, 'n_nonzero': n_nonzero,
        'power_cv': power_std, 'temp_range': temp_range,
        'T1196': t1196, 'T1197': t1197, 'T1198': t1198, 'T1199': t1199
    }


def exp2_dpm_state_forcing(fpga, results):
    """EXP 2: DPM State Forcing — force GPU clocks and classify from spikes."""
    print("\n=== EXP 2: DPM State Forcing ===")

    # Check if we can force DPM
    can_force = force_dpm_state(0)
    if not can_force:
        print("  Cannot force DPM (need write access to pp_dpm_sclk)")
        print("  Running in observation mode...")

    # Collect spike data at each DPM state
    dpm_states = [0, 1, 2]  # 600, 1100, 2900 MHz
    dpm_names = ['600MHz', '1100MHz', '2900MHz']
    all_spikes = []
    all_labels = []

    for dpm_idx, dpm_name in zip(dpm_states, dpm_names):
        if can_force:
            force_dpm_state(dpm_idx)
            time.sleep(1.0)  # Let DPM settle

        # Collect 30 samples at this state
        for trial in range(30):
            # Use power as input signal (one sample)
            hw = read_hwmon()
            power_norm = hw['power_uW'] / 1e7

            # Single-step MAC + telemetry
            fpga.set_mac_signal(power_norm)
            time.sleep(0.05)
            t = fpga.read_telemetry()

            if t and 'spike_counts' in t:
                spikes = np.array(t['spike_counts'][:N_NEURONS])
                all_spikes.append(spikes)
                all_labels.append(dpm_idx)

    if can_force:
        release_dpm()

    if len(all_spikes) < 30:
        print("  Not enough data collected")
        results['exp2'] = {'T1200': False, 'T1201': False, 'T1202': False, 'T1203': False}
        return

    X = np.array(all_spikes)
    y = np.array(all_labels)

    # Split train/test
    n = len(X)
    idx = np.random.RandomState(42).permutation(n)
    split = int(0.7 * n)
    X_tr, X_te = X[idx[:split]], X[idx[split:]]
    y_tr, y_te = y[idx[:split]], y[idx[split:]]

    acc = ridge_classify(X_tr, y_tr, X_te, y_te)
    print(f"  DPM classification accuracy: {acc:.1%}")

    # Power difference between states
    if can_force:
        powers = {}
        for dpm_idx, dpm_name in zip(dpm_states, dpm_names):
            force_dpm_state(dpm_idx)
            time.sleep(1.5)
            pw = []
            for _ in range(20):
                hw = read_hwmon()
                pw.append(hw['power_uW'] / 1e6)
                time.sleep(0.05)
            powers[dpm_name] = np.mean(pw)
            print(f"    {dpm_name}: {powers[dpm_name]:.2f}W")
        release_dpm()
        power_range = max(powers.values()) - min(powers.values())
    else:
        power_range = 0
        print("  (DPM forcing unavailable, power range unknown)")

    # T1200: Can read DPM state
    cur_dpm = get_current_dpm()
    t1200 = cur_dpm >= 0
    print(f"  T1200_dpm_readable: {'PASS' if t1200 else 'FAIL'} (state={cur_dpm})")

    # T1201: DPM classification > chance (33%)
    t1201 = acc > 0.40
    print(f"  T1201_dpm_classify_gt_chance: {'PASS' if t1201 else 'FAIL'} ({acc:.1%} vs 33%)")

    # T1202: Power varies with DPM (if forced)
    t1202 = power_range > 1.0 if can_force else False
    print(f"  T1202_power_varies_with_dpm: {'PASS' if t1202 else 'FAIL'} (range={power_range:.1f}W)")

    # T1203: Spike patterns differ between states
    if can_force and len(all_spikes) >= 60:
        state0 = X[y == 0].mean(axis=0)
        state2 = X[y == 2].mean(axis=0)
        diff = np.abs(state0 - state2).mean()
        t1203 = diff > 0.5
        print(f"  T1203_spike_diff_between_states: {'PASS' if t1203 else 'FAIL'} (diff={diff:.2f})")
    else:
        t1203 = False
        print(f"  T1203_spike_diff_between_states: FAIL (no DPM forcing)")

    results['exp2'] = {
        'acc': acc, 'power_range': power_range, 'can_force': can_force,
        'T1200': t1200, 'T1201': t1201, 'T1202': t1202, 'T1203': t1203
    }


def exp3_power_noise(fpga, results):
    """EXP 3: Power Dynamics as Noise Source — native 1/f from VRM."""
    print("\n=== EXP 3: Power Dynamics as Noise Source ===")

    configure_fpga(fpga, leak=TUNED_LEAK, bias_gain=TUNED_BIAS_GAIN, thresh=TUNED_THRESH)
    time.sleep(0.2)

    signals, labels = make_waveform_data(n_trials=120, n_steps=60)

    conditions = ['gpu_metrics', 'hwmon_power', 'white', 'none']
    accs = {}

    rng = np.random.RandomState(42)
    w_in = rng.randn(N_NEURONS) * 0.3
    vg_base = BASE_VG + rng.uniform(-VG_SPREAD, VG_SPREAD, N_NEURONS)

    for cond in conditions:
        print(f"  Condition: {cond}")
        features = []

        for trial_idx in range(len(signals)):
            spikes, vmem, _ = collect_reservoir(
                fpga, signals[trial_idx], noise_source=cond,
                w_in=w_in, vg_base=vg_base, use_mac=True, sample_hz=20
            )
            # Delta features
            delta = np.diff(spikes, axis=0)
            feat = np.concatenate([spikes.mean(axis=0), spikes.std(axis=0),
                                   delta.mean(axis=0) if len(delta) > 0 else np.zeros(N_NEURONS)])
            features.append(feat)

            if trial_idx % 40 == 0 and trial_idx > 0:
                print(f"    Trial {trial_idx}/{len(signals)}")

        X = np.array(features)
        n = len(X)
        idx = rng.permutation(n)
        split = int(0.7 * n)
        acc = ridge_classify(X[idx[:split]], labels[idx[:split]],
                           X[idx[split:]], labels[idx[split:]])
        accs[cond] = acc
        print(f"    {cond}: {acc:.1%}")

    # T1204: gpu_metrics coupling > none
    t1204 = accs['gpu_metrics'] > accs['none']
    print(f"  T1204_gpu_metrics_gt_none: {'PASS' if t1204 else 'FAIL'} ({accs['gpu_metrics']:.1%} vs {accs['none']:.1%})")

    # T1205: hwmon_power > none
    t1205 = accs['hwmon_power'] > accs['none']
    print(f"  T1205_hwmon_gt_none: {'PASS' if t1205 else 'FAIL'} ({accs['hwmon_power']:.1%} vs {accs['none']:.1%})")

    # T1206: Best coupled > 85%
    best = max(accs.values())
    t1206 = best > 0.85
    print(f"  T1206_best_gt_85: {'PASS' if t1206 else 'FAIL'} ({best:.1%})")

    # T1207: Power noise at least competitive with white noise
    t1207 = accs['hwmon_power'] >= accs['white'] - 0.05
    print(f"  T1207_power_competitive_with_white: {'PASS' if t1207 else 'FAIL'} ({accs['hwmon_power']:.1%} vs {accs['white']:.1%})")

    results['exp3'] = {
        'accs': accs,
        'T1204': t1204, 'T1205': t1205, 'T1206': t1206, 'T1207': t1207
    }


def exp4_clock_gating(fpga, results):
    """EXP 4: Clock Gating Coupling — GPU load transitions."""
    print("\n=== EXP 4: Clock Gating Coupling ===")

    # Collect power traces during idle and load
    idle_powers = []
    load_powers = []

    # Idle baseline
    print("  Collecting idle baseline...")
    for _ in range(50):
        hw = read_hwmon()
        idle_powers.append(hw['power_uW'] / 1e6)
        time.sleep(0.05)

    # Generate GPU load with a simple HIP kernel (or fallback to numpy)
    print("  Generating GPU load...")
    try:
        # Try to create GPU load via numpy/torch
        import subprocess
        proc = subprocess.Popen(
            ['python3', '-c',
             'import time; import numpy as np; '
             'a = np.random.randn(4096, 4096); '
             '[np.dot(a, a) for _ in range(100)]; '
             'time.sleep(5)'],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
        time.sleep(1.0)
        for _ in range(50):
            hw = read_hwmon()
            load_powers.append(hw['power_uW'] / 1e6)
            time.sleep(0.05)
        proc.wait(timeout=30)
    except Exception as e:
        print(f"  Load generation failed: {e}")
        load_powers = idle_powers.copy()

    idle_mean = np.mean(idle_powers)
    load_mean = np.mean(load_powers)
    power_delta = load_mean - idle_mean

    print(f"  Idle power: {idle_mean:.2f}W")
    print(f"  Load power: {load_mean:.2f}W")
    print(f"  Delta: {power_delta:.2f}W")

    # Collect FPGA spikes during transitions
    transition_spikes = []
    print("  Collecting transition spike data...")
    for rep in range(10):
        # Idle phase
        time.sleep(0.3)
        t = fpga.read_telemetry()
        if t and 'spike_counts' in t:
            transition_spikes.append(('idle', np.array(t['spike_counts'][:N_NEURONS])))

        # Brief CPU load
        a = np.random.randn(2048, 2048)
        _ = np.dot(a, a)

        t = fpga.read_telemetry()
        if t and 'spike_counts' in t:
            transition_spikes.append(('load', np.array(t['spike_counts'][:N_NEURONS])))

    # Check if spike patterns differ
    idle_spk = np.array([s[1] for s in transition_spikes if s[0] == 'idle'])
    load_spk = np.array([s[1] for s in transition_spikes if s[0] == 'load'])

    if len(idle_spk) > 0 and len(load_spk) > 0:
        spike_diff = np.abs(idle_spk.mean(axis=0) - load_spk.mean(axis=0)).mean()
    else:
        spike_diff = 0

    # T1208: Power increases under load
    t1208 = power_delta > 0.5
    print(f"  T1208_power_increases: {'PASS' if t1208 else 'FAIL'} (delta={power_delta:.2f}W)")

    # T1209: Idle power readable and stable
    idle_cv = np.std(idle_powers) / np.mean(idle_powers) if idle_mean > 0 else 1
    t1209 = idle_mean > 1.0 and idle_cv < 0.3
    print(f"  T1209_idle_stable: {'PASS' if t1209 else 'FAIL'} ({idle_mean:.2f}W, CV={idle_cv:.3f})")

    # T1210: Load creates measurable power change
    t1210 = power_delta > 1.0
    print(f"  T1210_load_power_change: {'PASS' if t1210 else 'FAIL'} ({power_delta:.2f}W)")

    # T1211: Spike patterns show any load sensitivity
    t1211 = spike_diff > 0.1
    print(f"  T1211_spike_load_sensitivity: {'PASS' if t1211 else 'FAIL'} (diff={spike_diff:.3f})")

    results['exp4'] = {
        'idle_W': idle_mean, 'load_W': load_mean, 'delta_W': power_delta,
        'spike_diff': spike_diff,
        'T1208': t1208, 'T1209': t1209, 'T1210': t1210, 'T1211': t1211
    }


def exp5_multi_layer_firmware(fpga, results):
    """EXP 5: Multi-Layer Firmware Telemetry fusion."""
    print("\n=== EXP 5: Multi-Layer Firmware Telemetry ===")

    configure_fpga(fpga, leak=TUNED_LEAK, bias_gain=TUNED_BIAS_GAIN, thresh=TUNED_THRESH)

    signals, labels = make_waveform_data(n_trials=100, n_steps=50)
    rng = np.random.RandomState(42)
    w_in = rng.randn(N_NEURONS) * 0.3
    vg_base = BASE_VG + rng.uniform(-VG_SPREAD, VG_SPREAD, N_NEURONS)

    # Collect with full GPU metrics coupling
    print("  Collecting multi-layer features...")
    spike_features = []
    gpu_feat_list = []

    for trial_idx in range(len(signals)):
        spikes, vmem, gpu_feats = collect_reservoir(
            fpga, signals[trial_idx], noise_source='gpu_metrics',
            w_in=w_in, vg_base=vg_base, use_mac=True, sample_hz=20
        )

        # Spike features
        delta = np.diff(spikes, axis=0)
        sfeat = np.concatenate([spikes.mean(axis=0), spikes.std(axis=0),
                                delta.mean(axis=0) if len(delta) > 0 else np.zeros(N_NEURONS)])
        spike_features.append(sfeat)

        # GPU features (mean across steps)
        if gpu_feats is not None and len(gpu_feats) > 0:
            gfeat = gpu_feats.mean(axis=0)
            gpu_feat_list.append(gfeat)
        else:
            gpu_feat_list.append(np.zeros(30))

        if trial_idx % 25 == 0 and trial_idx > 0:
            print(f"    Trial {trial_idx}/{len(signals)}")

    X_spike = np.array(spike_features)
    X_gpu = np.array(gpu_feat_list)
    X_fused = np.hstack([X_spike, X_gpu])

    n = len(X_spike)
    idx = rng.permutation(n)
    split = int(0.7 * n)

    acc_spike = ridge_classify(X_spike[idx[:split]], labels[idx[:split]],
                               X_spike[idx[split:]], labels[idx[split:]])
    acc_gpu = ridge_classify(X_gpu[idx[:split]], labels[idx[:split]],
                             X_gpu[idx[split:]], labels[idx[split:]])
    acc_fused = ridge_classify(X_fused[idx[:split]], labels[idx[:split]],
                               X_fused[idx[split:]], labels[idx[split:]])

    print(f"  Spike-only: {acc_spike:.1%}")
    print(f"  GPU-only: {acc_gpu:.1%}")
    print(f"  Fused: {acc_fused:.1%}")

    # T1212: Fused >= spike-only
    t1212 = acc_fused >= acc_spike - 0.02
    print(f"  T1212_fused_ge_spike: {'PASS' if t1212 else 'FAIL'}")

    # T1213: GPU features alone > chance (25%)
    t1213 = acc_gpu > 0.30
    print(f"  T1213_gpu_alone_gt_chance: {'PASS' if t1213 else 'FAIL'} ({acc_gpu:.1%})")

    # T1214: Spike features > 80%
    t1214 = acc_spike > 0.80
    print(f"  T1214_spike_gt_80: {'PASS' if t1214 else 'FAIL'} ({acc_spike:.1%})")

    # T1215: Feature dimensionality increase with GPU
    dim_spike = X_spike.shape[1]
    dim_fused = X_fused.shape[1]
    t1215 = dim_fused > dim_spike * 1.1
    print(f"  T1215_dim_increase: {'PASS' if t1215 else 'FAIL'} ({dim_spike}→{dim_fused})")

    results['exp5'] = {
        'acc_spike': acc_spike, 'acc_gpu': acc_gpu, 'acc_fused': acc_fused,
        'dim_spike': dim_spike, 'dim_fused': dim_fused,
        'T1212': t1212, 'T1213': t1213, 'T1214': t1214, 'T1215': t1215
    }


def exp6_pm4_register_probe(fpga, results):
    """EXP 6: PM4-Level Register Probing — deepest read-only GPU access."""
    print("\n=== EXP 6: PM4-Level Register Probing ===")

    # Read SMC registers via debugfs (read-only, safe)
    smc_data = {}
    try:
        import subprocess
        out = subprocess.run(['sudo', 'cat', '/sys/kernel/debug/dri/0/amdgpu_regs_smc'],
                           capture_output=True, timeout=5)
        if out.returncode == 0:
            lines = out.stdout.decode('utf-8', errors='replace').strip().split('\n')
            for line in lines[:100]:
                parts = line.strip().split()
                if len(parts) >= 2:
                    try:
                        addr = int(parts[0], 16) if parts[0].startswith('0x') else int(parts[0], 16)
                        val = int(parts[1], 16) if '0x' in parts[1].lower() else int(parts[1])
                        smc_data[addr] = val
                    except ValueError:
                        pass
            print(f"  SMC registers read: {len(smc_data)}")
        else:
            print(f"  SMC read failed (might need root)")
    except Exception as e:
        print(f"  SMC access error: {e}")

    # Read firmware info
    fw_info = {}
    try:
        out = subprocess.run(['sudo', 'cat', '/sys/kernel/debug/dri/0/amdgpu_firmware_info'],
                           capture_output=True, timeout=5)
        if out.returncode == 0:
            lines = out.stdout.decode('utf-8', errors='replace').strip().split('\n')
            fw_info['n_entries'] = len(lines)
            fw_info['mes_present'] = any('MES' in l for l in lines)
            fw_info['rlc_present'] = any('RLC' in l for l in lines)
            fw_info['smc_present'] = any('SMC' in l for l in lines)
            print(f"  Firmware entries: {fw_info['n_entries']}")
    except Exception as e:
        print(f"  Firmware info error: {e}")

    # Read PM info for deeper power state
    pm_info = {}
    try:
        out = subprocess.run(['sudo', 'cat', '/sys/kernel/debug/dri/0/amdgpu_pm_info'],
                           capture_output=True, timeout=5)
        if out.returncode == 0:
            text = out.stdout.decode('utf-8', errors='replace')
            pm_info['has_clocks'] = 'SCLK' in text
            pm_info['has_power'] = 'W' in text
            pm_info['has_temp'] = 'Temperature' in text
            pm_info['cg_flags'] = text.count('On')
            print(f"  PM info: clocks={pm_info['has_clocks']}, power={pm_info['has_power']}, CG_on={pm_info['cg_flags']}")
    except Exception as e:
        print(f"  PM info error: {e}")

    # Sample SMC/PM at rate for temporal features
    print("  Sampling firmware state vectors...")
    fw_vectors = []
    for _ in range(30):
        vec = []
        gm = read_gpu_metrics()
        hw = read_hwmon()
        pm = read_pm_table()

        vec.append(gm.get('temp_gfx', 0))
        vec.append(gm.get('temp_soc', 0))
        vec.append(gm.get('activity_gfx', 0))
        vec.append(gm.get('voltage_soc', 0))
        vec.append(gm.get('voltage_gfx', 0))
        vec.append(hw.get('power_uW', 0) / 1e6)
        vec.append(pm.get('hotspot_temp', 0))
        vec.append(pm.get('pm_power', 0))

        fw_vectors.append(vec)
        time.sleep(0.1)

    fw_arr = np.array(fw_vectors)
    # Effective dimensionality
    from numpy.linalg import svd
    if fw_arr.shape[0] > 2:
        centered = fw_arr - fw_arr.mean(axis=0)
        try:
            _, s, _ = svd(centered, full_matrices=False)
            var_explained = np.cumsum(s**2) / np.sum(s**2) if np.sum(s**2) > 0 else np.ones(len(s))
            eff_dim = np.searchsorted(var_explained, 0.90) + 1
        except Exception:
            eff_dim = 1
    else:
        eff_dim = 1

    print(f"  Firmware feature eff_dim(90%): {eff_dim}")

    # T1216: Can read firmware info
    t1216 = fw_info.get('n_entries', 0) > 10
    print(f"  T1216_fw_info_readable: {'PASS' if t1216 else 'FAIL'} ({fw_info.get('n_entries', 0)} entries)")

    # T1217: MES + RLC + SMC all present
    t1217 = fw_info.get('mes_present', False) and fw_info.get('rlc_present', False) and fw_info.get('smc_present', False)
    print(f"  T1217_all_fw_present: {'PASS' if t1217 else 'FAIL'}")

    # T1218: PM info has clock + power + temp
    t1218 = pm_info.get('has_clocks', False) and pm_info.get('has_power', False) and pm_info.get('has_temp', False)
    print(f"  T1218_pm_info_complete: {'PASS' if t1218 else 'FAIL'}")

    # T1219: Firmware features have eff_dim > 2
    t1219 = eff_dim > 2
    print(f"  T1219_fw_eff_dim_gt_2: {'PASS' if t1219 else 'FAIL'} ({eff_dim})")

    results['exp6'] = {
        'smc_registers': len(smc_data), 'fw_entries': fw_info.get('n_entries', 0),
        'eff_dim': eff_dim, 'pm_cg_flags': pm_info.get('cg_flags', 0),
        'T1216': t1216, 'T1217': t1217, 'T1218': t1218, 'T1219': t1219
    }


def exp7_cross_substrate_temporal(fpga, results):
    """EXP 7: Cross-Substrate Temporal Encoding — DPM sequences → FPGA decoding."""
    print("\n=== EXP 7: Cross-Substrate Temporal Encoding ===")

    configure_fpga(fpga, leak=TUNED_LEAK, bias_gain=TUNED_BIAS_GAIN, thresh=TUNED_THRESH)

    rng = np.random.RandomState(42)
    w_in = rng.randn(N_NEURONS) * 0.3
    vg_base = BASE_VG + rng.uniform(-VG_SPREAD, VG_SPREAD, N_NEURONS)

    # Encode temporal patterns via power modulation
    n_patterns = 4
    n_steps = 40
    n_trials = 80

    patterns = []
    for p in range(n_patterns):
        t = np.linspace(0, 2*np.pi, n_steps)
        if p == 0:
            sig = np.sin(t)
        elif p == 1:
            sig = np.sign(np.sin(t))
        elif p == 2:
            sig = np.sin(2*t) * np.sin(t)
        else:
            sig = np.linspace(-1, 1, n_steps)
        patterns.append(sig)

    # Conditions: FULL (gpu+fpga), FPGA_ONLY (no gpu noise), STATIC (no dynamics)
    conditions = {
        'FULL': {'noise': 'gpu_metrics', 'mac': True},
        'FPGA_ONLY': {'noise': 'none', 'mac': True},
        'STATIC': {'noise': 'none', 'mac': False},
    }

    cond_accs = {}

    for cond_name, cond_cfg in conditions.items():
        print(f"  Condition: {cond_name}")
        features = []
        labels = []

        for trial in range(n_trials):
            p = rng.randint(n_patterns)
            sig = patterns[p] + rng.randn(n_steps) * 0.1

            spikes, vmem, gpu_feats = collect_reservoir(
                fpga, sig, noise_source=cond_cfg['noise'],
                w_in=w_in, vg_base=vg_base, use_mac=cond_cfg['mac'],
                sample_hz=20
            )

            delta = np.diff(spikes, axis=0)
            feat = np.concatenate([spikes.mean(axis=0), spikes.std(axis=0),
                                   delta.mean(axis=0) if len(delta) > 0 else np.zeros(N_NEURONS)])

            # Add GPU features if available
            if gpu_feats is not None and len(gpu_feats) > 0:
                gfeat = gpu_feats.mean(axis=0)
                feat = np.concatenate([feat, gfeat])

            features.append(feat)
            labels.append(p)

            if trial % 20 == 0 and trial > 0:
                print(f"    Trial {trial}/{n_trials}")

        X = np.array(features)
        y = np.array(labels)

        idx = rng.permutation(len(X))
        split = int(0.7 * len(X))
        acc = ridge_classify(X[idx[:split]], y[idx[:split]], X[idx[split:]], y[idx[split:]])
        cond_accs[cond_name] = acc
        print(f"    {cond_name}: {acc:.1%}")

    # Memory test: classify with delay
    print("  Delayed readout test...")
    delay_accs = {}
    for delay in [0, 2, 5]:
        features = []
        labels = []

        for trial in range(60):
            p = rng.randint(n_patterns)
            # Pattern + blank delay
            sig = np.concatenate([patterns[p], np.zeros(delay)])
            sig += rng.randn(len(sig)) * 0.1

            spikes, vmem, _ = collect_reservoir(
                fpga, sig, noise_source='gpu_metrics',
                w_in=w_in, vg_base=vg_base, use_mac=True, sample_hz=20
            )

            # Use only the LAST n_steps of spikes (after blank period)
            if len(spikes) > delay and delay > 0:
                late_spikes = spikes[-delay:]
                feat = np.concatenate([late_spikes.mean(axis=0), late_spikes.std(axis=0)])
            else:
                feat = np.concatenate([spikes.mean(axis=0), spikes.std(axis=0)])

            features.append(feat)
            labels.append(p)

        X = np.array(features)
        y = np.array(labels)
        idx = rng.permutation(len(X))
        split = int(0.7 * len(X))
        acc = ridge_classify(X[idx[:split]], y[idx[:split]], X[idx[split:]], y[idx[split:]])
        delay_accs[delay] = acc
        print(f"    Delay={delay}: {acc:.1%}")

    # T1220: FULL > STATIC
    t1220 = cond_accs['FULL'] > cond_accs['STATIC'] + 0.05
    print(f"  T1220_full_gt_static: {'PASS' if t1220 else 'FAIL'} ({cond_accs['FULL']:.1%} vs {cond_accs['STATIC']:.1%})")

    # T1221: FULL >= FPGA_ONLY
    t1221 = cond_accs['FULL'] >= cond_accs['FPGA_ONLY'] - 0.02
    print(f"  T1221_full_ge_fpga: {'PASS' if t1221 else 'FAIL'} ({cond_accs['FULL']:.1%} vs {cond_accs['FPGA_ONLY']:.1%})")

    # T1222: FULL > 80%
    t1222 = cond_accs['FULL'] > 0.80
    print(f"  T1222_full_gt_80: {'PASS' if t1222 else 'FAIL'} ({cond_accs['FULL']:.1%})")

    # T1223: Memory persists (delay=2 > chance)
    t1223 = delay_accs.get(2, 0) > 0.30
    print(f"  T1223_memory_d2_gt_chance: {'PASS' if t1223 else 'FAIL'} ({delay_accs.get(2, 0):.1%})")

    # T1224: Memory fades (delay=0 > delay=5)
    t1224 = delay_accs.get(0, 0) > delay_accs.get(5, 0)
    print(f"  T1224_memory_fades: {'PASS' if t1224 else 'FAIL'} (d0={delay_accs.get(0,0):.1%} vs d5={delay_accs.get(5,0):.1%})")

    # T1225: STATIC < 50% (no temporal encoding without dynamics)
    t1225 = cond_accs['STATIC'] < 0.55
    print(f"  T1225_static_lt_55: {'PASS' if t1225 else 'FAIL'} ({cond_accs['STATIC']:.1%})")

    results['exp7'] = {
        'cond_accs': cond_accs, 'delay_accs': delay_accs,
        'T1220': t1220, 'T1221': t1221, 'T1222': t1222,
        'T1223': t1223, 'T1224': t1224, 'T1225': t1225
    }


def exp8_full_deep_stack(fpga, results):
    """EXP 8: Full Deep Stack Benchmark — all layers active."""
    print("\n=== EXP 8: Full Deep Stack Benchmark ===")

    configure_fpga(fpga, leak=TUNED_LEAK, bias_gain=TUNED_BIAS_GAIN, thresh=TUNED_THRESH)

    signals, labels = make_waveform_data(n_trials=120, n_steps=60, n_classes=4)
    rng = np.random.RandomState(42)
    w_in = rng.randn(N_NEURONS) * 0.3
    vg_base = BASE_VG + rng.uniform(-VG_SPREAD, VG_SPREAD, N_NEURONS)

    # Three conditions: DEEP (all layers), SHALLOW (hwmon only), NO_GPU
    conditions = {
        'DEEP': 'gpu_metrics',
        'SHALLOW': 'hwmon_power',
        'NO_GPU': 'none',
    }

    cond_results = {}

    for cond_name, noise_src in conditions.items():
        print(f"  Condition: {cond_name}")
        features = []

        for trial_idx in range(len(signals)):
            spikes, vmem, gpu_feats = collect_reservoir(
                fpga, signals[trial_idx], noise_source=noise_src,
                w_in=w_in, vg_base=vg_base, use_mac=True, sample_hz=20
            )

            delta = np.diff(spikes, axis=0)
            feat = np.concatenate([
                spikes.mean(axis=0), spikes.std(axis=0),
                delta.mean(axis=0) if len(delta) > 0 else np.zeros(N_NEURONS),
                vmem.mean(axis=0), vmem.std(axis=0),
            ])

            # Add GPU features for DEEP condition
            if gpu_feats is not None and len(gpu_feats) > 0:
                gfeat_mean = gpu_feats.mean(axis=0)
                gfeat_std = gpu_feats.std(axis=0)
                feat = np.concatenate([feat, gfeat_mean, gfeat_std])

            features.append(feat)

            if trial_idx % 40 == 0 and trial_idx > 0:
                print(f"    Trial {trial_idx}/{len(signals)}")

        X = np.array(features)
        n = len(X)
        idx = rng.permutation(n)
        split = int(0.7 * n)
        acc = ridge_classify(X[idx[:split]], labels[idx[:split]],
                           X[idx[split:]], labels[idx[split:]])
        cond_results[cond_name] = acc
        print(f"    {cond_name}: {acc:.1%}")

    # XOR temporal test
    print("  XOR temporal test...")
    xor_signals = []
    xor_labels = []
    n_xor = 80
    for _ in range(n_xor):
        a, b = rng.randint(2), rng.randint(2)
        delay = 3
        sig = np.zeros(20)
        sig[0:5] = a * 2 - 1
        sig[5+delay:10+delay] = b * 2 - 1
        xor_signals.append(sig)
        xor_labels.append(a ^ b)

    xor_features = []
    for sig in xor_signals:
        spikes, vmem, _ = collect_reservoir(
            fpga, sig, noise_source='gpu_metrics',
            w_in=w_in, vg_base=vg_base, use_mac=True, sample_hz=20
        )
        delta = np.diff(spikes, axis=0)
        feat = np.concatenate([spikes.mean(axis=0), delta.mean(axis=0) if len(delta) > 0 else np.zeros(N_NEURONS)])
        xor_features.append(feat)

    X_xor = np.array(xor_features)
    y_xor = np.array(xor_labels)
    idx = rng.permutation(len(X_xor))
    split = int(0.7 * len(X_xor))
    xor_acc = ridge_classify(X_xor[idx[:split]], y_xor[idx[:split]],
                             X_xor[idx[split:]], y_xor[idx[split:]])
    print(f"  XOR accuracy: {xor_acc:.1%}")

    # T1226: DEEP > NO_GPU
    t1226 = cond_results['DEEP'] > cond_results['NO_GPU']
    print(f"  T1226_deep_gt_nogpu: {'PASS' if t1226 else 'FAIL'} ({cond_results['DEEP']:.1%} vs {cond_results['NO_GPU']:.1%})")

    # T1227: DEEP >= SHALLOW
    t1227 = cond_results['DEEP'] >= cond_results['SHALLOW'] - 0.02
    print(f"  T1227_deep_ge_shallow: {'PASS' if t1227 else 'FAIL'} ({cond_results['DEEP']:.1%} vs {cond_results['SHALLOW']:.1%})")

    # T1228: DEEP > 85%
    t1228 = cond_results['DEEP'] > 0.85
    print(f"  T1228_deep_gt_85: {'PASS' if t1228 else 'FAIL'} ({cond_results['DEEP']:.1%})")

    # T1229: XOR > chance (50%)
    t1229 = xor_acc > 0.55
    print(f"  T1229_xor_gt_chance: {'PASS' if t1229 else 'FAIL'} ({xor_acc:.1%})")

    # T1230: Feature dimensionality with deep stack
    deep_dim = len(features[0]) if features else 0
    t1230 = deep_dim > 500
    print(f"  T1230_deep_dim_gt_500: {'PASS' if t1230 else 'FAIL'} ({deep_dim}D)")

    results['exp8'] = {
        'cond_results': cond_results, 'xor_acc': xor_acc, 'deep_dim': deep_dim,
        'T1226': t1226, 'T1227': t1227, 'T1228': t1228, 'T1229': t1229, 'T1230': t1230
    }


# ─── Main ───────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("z2249: Deep GPU-FPGA Integration — Firmware-Level Coupling")
    print("  Pushing GPU access deep: gpu_metrics, DPM, power, PM4")
    print("  Coupling all firmware layers to FPGA reservoir")
    print("=" * 60)

    # Connect to FPGA
    fpga = FPGAEthBridge()
    fpga.connect()

    t = fpga.read_telemetry()
    if t and 'spike_counts' in t:
        n_neurons = len(t['spike_counts'])
        print(f"  FPGA: {n_neurons} neurons")
        print(f"  Spikes: [{min(t['spike_counts'])}, {max(t['spike_counts'])}]")

    results = {'experiment': 'z2249_deep_gpu_fpga_integration', 'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')}

    # Run all experiments
    exp1_gpu_metrics_deep(fpga, results)
    exp2_dpm_state_forcing(fpga, results)
    exp3_power_noise(fpga, results)
    exp4_clock_gating(fpga, results)
    exp5_multi_layer_firmware(fpga, results)
    exp6_pm4_register_probe(fpga, results)
    exp7_cross_substrate_temporal(fpga, results)
    exp8_full_deep_stack(fpga, results)

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    n_pass = 0
    n_total = 0
    for exp_name, exp_data in results.items():
        if not isinstance(exp_data, dict):
            continue
        for k, v in exp_data.items():
            if k.startswith('T1') and isinstance(v, bool):
                n_total += 1
                if v:
                    n_pass += 1

    print(f"  Total: {n_pass}/{n_total} PASS")

    # Save results
    results_path = 'results/z2249_deep_gpu_fpga.json'

    # Convert numpy types for JSON
    def convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [convert(x) for x in obj]
        return obj

    with open(results_path, 'w') as f:
        json.dump(convert(results), f, indent=2)
    print(f"  Results saved to {results_path}")

    fpga.close()
    print("Done.")

if __name__ == '__main__':
    main()
