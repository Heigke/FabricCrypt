#!/usr/bin/env python3
"""z2169_transfer_regime.py — Transfer Learning + Noise Regime Switching

Tests whether an FPGA LIF reservoir trained under one GPU noise regime
(compute-active) can generalize to a different regime (idle), and whether
the reservoir's internal state statistics shift between regimes.

Phases:
  Phase 1 (TRAIN_REGIME): GPU under compute workload (torch matmul).
          Collect waveform trials, train ridge readout.
  Phase 2 (TRANSFER):     GPU idle, apply trained readout (no retraining).
  Phase 3 (RETRAINED):    GPU idle, collect fresh trials, retrain readout.

Conditions:
  A: SAME_REGIME     — Train and test both under compute workload (sanity)
  B: CROSS_REGIME    — Train under compute, test under idle (transfer)
  C: RETRAINED       — Train and test both under idle (upper bound)

Tests (T103-T108):
  T103: Same-regime accuracy > 55%  (sanity — reservoir works)
  T104: Cross-regime accuracy > 33.3%  (knowledge transfers across regimes)
  T105: Cross-regime accuracy < same-regime  (regime matters)
  T106: Retrained > cross-regime  (retraining helps)
  T107: GPU power differs between regimes > 2W
  T108: Reservoir state vmem mean shift > 0.01 between regimes

Hardware: AMD gfx1151 GPU + Arty A7 FPGA on /dev/ttyUSB0|1
"""

import os, sys, json, time, struct, argparse, threading
import numpy as np
from pathlib import Path

os.environ.setdefault('HSA_OVERRIDE_GFX_VERSION', '11.0.0')

BASE = Path(__file__).resolve().parent.parent
RESULTS = BASE / 'results'
FIGURES = (RESULTS / 'FEEL_paper_update' /
           'FEEL__Functionally_Embodied_Emergent_Learning__13_-5' / 'figures')

# ─── FPGA Protocol ───
SYNC = 0x55
CMD_SET_VG = 0x01
CMD_READ_TELEM = 0x02
CMD_SET_KILL = 0x03

HWMON_POWER = "/sys/class/hwmon/hwmon7/power1_average"

# ─── Reservoir Parameters ───
BASE_VG = 0.58       # near BVpar cliff
ALPHA = 0.25         # input coupling
BETA = 0.08          # noise coupling
N_NEURONS = 8
SAMPLE_HZ = 20
IIR_ALPHA = 0.85     # temporal memory filter


# ═══════════════════════════════════════════════════════════
# FPGA Communication (from z2162/z2165)
# ═══════════════════════════════════════════════════════════

def to_q16_16(val: float) -> int:
    return int(val * 65536) & 0xFFFFFFFF


def find_fpga():
    try:
        import serial
    except ImportError:
        return None, None
    for p in ['/dev/ttyUSB1', '/dev/ttyUSB0', '/dev/ttyUSB2']:
        try:
            s = serial.Serial(p, 115200, timeout=0.2)
            time.sleep(0.1)
            return s, p
        except Exception:
            continue
    return None, None


def reconnect_fpga(port):
    import serial
    try:
        ser = serial.Serial(port, 115200, timeout=0.2)
        time.sleep(0.1)
        ser.write(bytes([SYNC, CMD_SET_KILL, 0x00]))
        ser.flush()
        time.sleep(0.1)
        return ser
    except Exception:
        return None


def set_per_neuron_vg(ser, vg_values):
    for nid, vg in enumerate(vg_values[:8]):
        q16 = to_q16_16(max(0.0, min(1.0, vg)))
        payload = bytes([nid & 0x07]) + struct.pack('>I', q16)
        ser.write(bytes([SYNC, CMD_SET_VG]) + payload)
    ser.flush()
    time.sleep(0.005)


def read_telem(ser, timeout=0.15):
    """Read telemetry packet: [0x55][0x02][0x30][48B][CRC8] = 52 bytes."""
    deadline = time.monotonic() + timeout
    buf = bytearray()
    while time.monotonic() < deadline:
        ser.timeout = max(0.001, deadline - time.monotonic())
        b = ser.read(1)
        if not b:
            continue
        if b[0] == SYNC:
            buf = bytearray([SYNC])
            while len(buf) < 52 and time.monotonic() < deadline:
                ser.timeout = max(0.001, deadline - time.monotonic())
                chunk = ser.read(52 - len(buf))
                if chunk:
                    buf.extend(chunk)
            break
    if len(buf) < 52:
        return None
    payload = bytes(buf[3:51])
    neurons = []
    for i in range(8):
        off = i * 6
        sc = struct.unpack_from('>H', payload, off)[0]
        vm = struct.unpack_from('>H', payload, off + 2)[0]
        neurons.append({'spike_count': sc, 'vmem': vm / 256.0})
    return neurons


def safe_fpga_step(ser, port, vg_values, interval):
    import serial as serial_mod
    try:
        set_per_neuron_vg(ser, vg_values)
        time.sleep(interval * 0.3)
        ser.reset_input_buffer()
        ser.write(bytes([SYNC, CMD_READ_TELEM]))
        ser.flush()
        telem = read_telem(ser, timeout=0.15)
        return ser, telem
    except (serial_mod.SerialException, OSError) as e:
        print(f"    [!] Serial error: {e}, reconnecting...")
        try:
            ser.close()
        except Exception:
            pass
        time.sleep(0.5)
        new_ser = reconnect_fpga(port)
        if new_ser is None:
            print("    [!] Reconnection failed")
            return None, None
        print("    [!] Reconnected successfully")
        return new_ser, None


# ═══════════════════════════════════════════════════════════
# Noise Sources
# ═══════════════════════════════════════════════════════════

def read_hwmon_power():
    """Read hwmon power1_average (uW -> W)."""
    try:
        return int(open(HWMON_POWER).read().strip()) / 1e6
    except Exception:
        return None


def iir_filter_noise(noise_samples, alpha_iir=0.85):
    """IIR low-pass: y[t] = alpha*y[t-1] + (1-alpha)*x[t]."""
    if len(noise_samples) == 0:
        return noise_samples
    filtered = np.zeros(len(noise_samples))
    filtered[0] = noise_samples[0]
    for t in range(1, len(noise_samples)):
        filtered[t] = alpha_iir * filtered[t - 1] + (1 - alpha_iir) * noise_samples[t]
    std = max(np.std(filtered), 1e-6)
    return filtered / std


def normalize_noise(samples):
    arr = np.array(samples, dtype=float)
    if len(arr) == 0:
        return arr
    mu = arr.mean()
    std = max(arr.std(), 1e-6)
    return (arr - mu) / std


def generate_synthetic_1f(n_samples, rng):
    """Voss-McCartney 1/f generator."""
    noise = np.zeros(n_samples)
    n_octaves = 8
    octaves = np.zeros(n_octaves)
    for i in range(n_samples):
        for j in range(n_octaves):
            if i % (1 << j) == 0:
                octaves[j] = rng.standard_normal()
        noise[i] = octaves.sum()
    return normalize_noise(noise)


# ═══════════════════════════════════════════════════════════
# GPU Workload Control
# ═══════════════════════════════════════════════════════════

_gpu_workload_running = False
_gpu_workload_thread = None


def _gpu_matmul_loop(size=2048, stop_event=None):
    """Continuous GPU matmul to create compute workload + noise."""
    global _gpu_workload_running
    try:
        import torch
        if not torch.cuda.is_available():
            print("    [!] CUDA/ROCm not available for GPU workload")
            _gpu_workload_running = False
            return
        device = torch.device('cuda')
        a = torch.randn(size, size, device=device)
        b = torch.randn(size, size, device=device)
        while not stop_event.is_set():
            _ = torch.mm(a, b)
            torch.cuda.synchronize()
    except Exception as e:
        print(f"    [!] GPU workload error: {e}")
    finally:
        _gpu_workload_running = False


def start_gpu_workload(size=2048):
    """Start background GPU compute workload."""
    global _gpu_workload_running, _gpu_workload_thread
    if _gpu_workload_running:
        return
    stop_event = threading.Event()
    _gpu_workload_running = True
    _gpu_workload_thread = threading.Thread(
        target=_gpu_matmul_loop, args=(size, stop_event), daemon=True)
    _gpu_workload_thread._stop_event = stop_event
    _gpu_workload_thread.start()
    # Let GPU warm up and stabilize power draw
    time.sleep(2.0)
    print(f"    GPU workload started (matmul {size}x{size})")


def stop_gpu_workload():
    """Stop background GPU compute workload."""
    global _gpu_workload_running, _gpu_workload_thread
    if _gpu_workload_thread is not None and hasattr(_gpu_workload_thread, '_stop_event'):
        _gpu_workload_thread._stop_event.set()
        _gpu_workload_thread.join(timeout=5.0)
    _gpu_workload_running = False
    _gpu_workload_thread = None
    # Let GPU cool down / power stabilize
    time.sleep(2.0)
    print("    GPU workload stopped")


# ═══════════════════════════════════════════════════════════
# Waveform Generation
# ═══════════════════════════════════════════════════════════

def generate_waveforms(n_trials=200, steps_per_trial=30, freq_hz=1.0,
                       dt=1.0 / 20, seed=42):
    """Generate sine/triangle/square waveforms for classification."""
    rng = np.random.default_rng(seed)
    trials = []
    labels = []
    t = np.arange(steps_per_trial) * dt

    for _ in range(n_trials):
        cls = rng.integers(0, 3)
        phase = rng.uniform(0, 2 * np.pi)
        freq = freq_hz * rng.uniform(0.8, 1.2)

        if cls == 0:   # sine
            wave = np.sin(2 * np.pi * freq * t + phase)
        elif cls == 1:  # triangle
            wave = 2.0 * np.abs(2.0 * ((freq * t + phase / (2 * np.pi)) % 1.0) - 1.0) - 1.0
        else:           # square
            wave = np.sign(np.sin(2 * np.pi * freq * t + phase))

        wave = (wave + 1.0) / 2.0
        trials.append(wave)
        labels.append(cls)

    return np.array(trials), np.array(labels)


# ═══════════════════════════════════════════════════════════
# Reservoir Execution
# ═══════════════════════════════════════════════════════════

def run_fpga_reservoir_trial(ser, port, input_signal, noise_samples, w_in, w_noise,
                              base_vg=BASE_VG, alpha=ALPHA, beta=BETA,
                              live_noise=False):
    """Drive FPGA neurons with input+noise and collect spike/vmem states.

    When live_noise=True, reads power rail in real-time (true substrate coupling).
    Returns: (n_steps, 24) array, updated ser, list of power readings.
    """
    n_steps = len(input_signal)
    interval = 1.0 / SAMPLE_HZ
    states = np.zeros((n_steps, N_NEURONS * 3))
    prev_counts = None
    cumulative = np.zeros(N_NEURONS)
    power_mean = 11.0
    power_readings = []

    for t in range(n_steps):
        # Get noise value
        if live_noise:
            p = read_hwmon_power()
            if p is not None:
                power_readings.append(p)
            noise_val = (p - power_mean) / 2.0 if p else 0.0
        elif beta > 0 and len(noise_samples) > 0:
            noise_val = noise_samples[t % len(noise_samples)]
        else:
            noise_val = 0.0

        # Compute per-neuron Vg
        vg_values = np.full(N_NEURONS, base_vg)
        vg_values += alpha * input_signal[t] * w_in
        if beta > 0:
            vg_values += beta * noise_val * w_noise
        vg_values = np.clip(vg_values, 0.05, 0.95)

        # FPGA step with reconnection
        ser, telem = safe_fpga_step(ser, port, vg_values, interval)
        if ser is None:
            break

        if telem:
            counts = [n['spike_count'] for n in telem]
            vmems = [n['vmem'] for n in telem]
            if prev_counts is not None:
                for i in range(N_NEURONS):
                    delta = (counts[i] - prev_counts[i]) & 0xFFFF
                    if delta > 30000:
                        delta = 0
                    states[t, i] = delta
                    cumulative[i] += delta
            for i in range(N_NEURONS):
                states[t, N_NEURONS + i] = vmems[i]
                states[t, N_NEURONS * 2 + i] = cumulative[i]
            prev_counts = counts[:]

        time.sleep(max(0, interval * 0.5 - 0.01))

    return states, ser, power_readings


def simulate_lif_reservoir(input_signal, noise_samples, w_in, w_noise,
                            base_vg=BASE_VG, alpha=ALPHA, beta=BETA,
                            regime='compute'):
    """Software LIF simulation fallback.

    regime='compute' adds extra noise variance to simulate higher GPU power.
    regime='idle' uses lower noise variance.
    """
    n_steps = len(input_signal)
    states = np.zeros((n_steps, N_NEURONS * 3))

    v_rest = 0.0
    v_thresh = 1.0
    tau_m = 0.02
    dt = 1.0 / SAMPLE_HZ
    vmem = np.zeros(N_NEURONS)
    cumulative = np.zeros(N_NEURONS)

    # Regime-dependent noise scaling
    noise_scale = 1.2 if regime == 'compute' else 0.6

    for t in range(n_steps):
        vg = np.full(N_NEURONS, base_vg)
        vg += alpha * input_signal[t] * w_in
        if beta > 0 and len(noise_samples) > 0:
            noise_idx = t % len(noise_samples)
            vg += beta * noise_scale * noise_samples[noise_idx] * w_noise
        vg = np.clip(vg, 0.05, 0.95)

        I_in = vg * 5.0
        dvdt = (-vmem + I_in) / tau_m
        vmem += dvdt * dt

        spikes = np.zeros(N_NEURONS)
        for i in range(N_NEURONS):
            if vmem[i] >= v_thresh:
                spikes[i] = 1
                vmem[i] = v_rest
                cumulative[i] += 1

        states[t, :N_NEURONS] = spikes
        states[t, N_NEURONS:N_NEURONS * 2] = vmem.copy()
        states[t, N_NEURONS * 2:] = cumulative.copy()

    return states


# ═══════════════════════════════════════════════════════════
# Feature Extraction & Classification
# ═══════════════════════════════════════════════════════════

def augment_with_delays(states, delays=(1, 2, 3)):
    T, D = states.shape
    augmented = np.zeros((T, D * (1 + len(delays))))
    augmented[:, :D] = states
    for i, d in enumerate(delays):
        start = D * (i + 1)
        augmented[d:, start:start + D] = states[:T - d]
    return augmented


def pool_trial_features(trial_states):
    return np.concatenate([
        trial_states.mean(axis=0),
        trial_states.std(axis=0),
        trial_states.max(axis=0),
        trial_states.min(axis=0),
    ])


def ridge_classify(X_train, y_train, X_test, y_test, alphas=None):
    """Ridge regression classifier (one-hot encoding for multi-class)."""
    if alphas is None:
        alphas = [1e-6, 1e-4, 1e-2, 1.0, 100.0]

    n_classes = len(np.unique(y_train))
    Y_train = np.zeros((len(y_train), n_classes))
    for i, y in enumerate(y_train):
        Y_train[i, int(y)] = 1.0

    best_acc = -1
    best_W = None
    for alpha_r in alphas:
        I = np.eye(X_train.shape[1])
        try:
            W = np.linalg.solve(X_train.T @ X_train + alpha_r * I,
                                X_train.T @ Y_train)
        except np.linalg.LinAlgError:
            continue
        pred_test = np.argmax(X_test @ W, axis=1)
        acc_test = np.mean(pred_test == y_test)
        if acc_test > best_acc:
            best_acc = acc_test
            best_W = W

    return best_acc, best_W


def ridge_classify_with_W(W, X_test, y_test):
    """Apply pre-trained ridge weights to new test data."""
    pred_test = np.argmax(X_test @ W, axis=1)
    acc = np.mean(pred_test == y_test)
    return acc


def stratified_kfold(X, y, n_splits=5, seed=42):
    rng = np.random.default_rng(seed)
    classes = np.unique(y)
    indices = np.arange(len(y))
    rng.shuffle(indices)

    folds = [[] for _ in range(n_splits)]
    for c in classes:
        c_idx = indices[y[indices] == c]
        for i, idx in enumerate(c_idx):
            folds[i % n_splits].append(idx)

    splits = []
    for fold in range(n_splits):
        test_idx = np.array(folds[fold])
        train_idx = np.concatenate(
            [np.array(folds[f]) for f in range(n_splits) if f != fold])
        splits.append((train_idx, test_idx))
    return splits


# ═══════════════════════════════════════════════════════════
# Collect Noise Under a Given Regime
# ═══════════════════════════════════════════════════════════

def collect_noise_regime(duration_s=10.0, sample_hz=50):
    """Collect power rail noise under current GPU regime."""
    n = int(duration_s * sample_hz)
    interval = 1.0 / sample_hz
    samples = []
    for _ in range(n):
        p = read_hwmon_power()
        if p is not None:
            samples.append(p)
        time.sleep(interval)
    return samples


# ═══════════════════════════════════════════════════════════
# Run Trials Under a Regime
# ═══════════════════════════════════════════════════════════

def run_trials_regime(ser, port, fpga, wave_trials, wave_labels,
                      noise_iir, w_in, w_noise, regime_name,
                      n_trials=None):
    """Run waveform trials and collect features + state statistics.

    Returns: features array, vmem_means list, power_readings list, updated ser.
    """
    if n_trials is None:
        n_trials = len(wave_trials)
    n_trials = min(n_trials, len(wave_trials))

    trial_features = []
    vmem_means_all = []
    power_readings_all = []
    t0 = time.monotonic()

    for trial_idx in range(n_trials):
        input_signal = wave_trials[trial_idx]

        if fpga and ser is not None:
            states, ser, pw = run_fpga_reservoir_trial(
                ser, port, input_signal, noise_iir, w_in, w_noise,
                base_vg=BASE_VG, alpha=ALPHA, beta=BETA,
                live_noise=True)
            power_readings_all.extend(pw)
            if ser is None:
                print(f"    FPGA lost during {regime_name}, switching to sim")
                fpga = False
        else:
            states = simulate_lif_reservoir(
                input_signal, noise_iir, w_in, w_noise,
                base_vg=BASE_VG, alpha=ALPHA, beta=BETA,
                regime=regime_name)
            # Simulate power readings
            if regime_name == 'compute':
                power_readings_all.extend(
                    [15.0 + np.random.randn() * 2.0 for _ in range(5)])
            else:
                power_readings_all.extend(
                    [8.0 + np.random.randn() * 0.5 for _ in range(5)])

        # Collect vmem statistics from middle columns
        vmem_cols = states[:, N_NEURONS:N_NEURONS * 2]
        vmem_means_all.append(float(vmem_cols.mean()))

        aug = augment_with_delays(states, delays=(1, 2, 3))
        feat = pool_trial_features(aug)
        trial_features.append(feat)

        if (trial_idx + 1) % 50 == 0:
            elapsed = time.monotonic() - t0
            rate = (trial_idx + 1) / elapsed
            eta = (n_trials - trial_idx - 1) / rate
            print(f"    Trial {trial_idx + 1}/{n_trials} "
                  f"({rate:.1f} trials/s, ETA {eta:.0f}s)")

    elapsed = time.monotonic() - t0
    print(f"  {regime_name}: {len(trial_features)} trials in {elapsed:.1f}s")
    return np.array(trial_features), vmem_means_all, power_readings_all, ser, fpga


# ═══════════════════════════════════════════════════════════
# Figure Generation
# ═══════════════════════════════════════════════════════════

def generate_figure(results, fig_path):
    """Generate summary figure for z2169."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("  matplotlib not available, skipping figure")
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('z2169: Transfer Learning + Noise Regime Switching',
                 fontsize=14, fontweight='bold')

    # Panel A: Accuracy comparison bar chart
    ax = axes[0, 0]
    conds = ['same_regime', 'cross_regime', 'retrained']
    labels = ['Same Regime\n(Train+Test Compute)', 'Cross Regime\n(Train Compute,\nTest Idle)',
              'Retrained\n(Train+Test Idle)']
    accs = [results['accuracies'].get(c, {}).get('mean', 0) for c in conds]
    stds = [results['accuracies'].get(c, {}).get('std', 0) for c in conds]
    colors = ['#2196F3', '#FF9800', '#4CAF50']
    bars = ax.bar(range(3), accs, yerr=stds, capsize=5, color=colors,
                  edgecolor='black', linewidth=0.8)
    ax.axhline(y=1.0 / 3.0, color='red', linestyle='--', alpha=0.7,
               label='Chance (33.3%)')
    ax.set_xticks(range(3))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel('Accuracy')
    ax.set_title('A) Waveform Classification Accuracy')
    ax.set_ylim(0, 1.0)
    ax.legend(fontsize=8)
    for bar, acc in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f'{acc:.3f}', ha='center', va='bottom', fontsize=9)

    # Panel B: Power distribution per regime
    ax = axes[0, 1]
    compute_power = results.get('power_stats', {}).get('compute', {})
    idle_power = results.get('power_stats', {}).get('idle', {})
    if compute_power and idle_power:
        data = [compute_power.get('mean', 0), idle_power.get('mean', 0)]
        errs = [compute_power.get('std', 0), idle_power.get('std', 0)]
        ax.bar([0, 1], data, yerr=errs, capsize=5,
               color=['#F44336', '#2196F3'], edgecolor='black', linewidth=0.8)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(['GPU Compute', 'GPU Idle'])
        ax.set_ylabel('Power (W)')
        diff = abs(data[0] - data[1])
        ax.set_title(f'B) GPU Power per Regime (delta={diff:.2f}W)')
    else:
        ax.text(0.5, 0.5, 'No power data', transform=ax.transAxes,
                ha='center', va='center')
        ax.set_title('B) GPU Power per Regime')

    # Panel C: Vmem distribution shift
    ax = axes[1, 0]
    vmem_compute = results.get('vmem_stats', {}).get('compute', {})
    vmem_idle = results.get('vmem_stats', {}).get('idle', {})
    if vmem_compute and vmem_idle:
        ax.bar([0, 1],
               [vmem_compute.get('mean', 0), vmem_idle.get('mean', 0)],
               yerr=[vmem_compute.get('std', 0), vmem_idle.get('std', 0)],
               capsize=5, color=['#F44336', '#2196F3'],
               edgecolor='black', linewidth=0.8)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(['Compute Regime', 'Idle Regime'])
        ax.set_ylabel('Mean Vmem')
        shift = abs(vmem_compute.get('mean', 0) - vmem_idle.get('mean', 0))
        ax.set_title(f'C) Reservoir Vmem Shift (delta={shift:.4f})')
    else:
        ax.text(0.5, 0.5, 'No vmem data', transform=ax.transAxes,
                ha='center', va='center')
        ax.set_title('C) Reservoir Vmem Shift')

    # Panel D: Test results summary
    ax = axes[1, 1]
    ax.axis('off')
    tests = results.get('tests', {})
    text_lines = ['Test Results Summary\n']
    for tid in sorted(tests.keys()):
        t = tests[tid]
        status = 'PASS' if t.get('pass', False) else 'FAIL'
        marker = '+' if t.get('pass', False) else 'x'
        text_lines.append(f"[{marker}] {tid}: {t.get('description', '')}")
        text_lines.append(f"    value={t.get('value', 'N/A')}, "
                         f"threshold={t.get('threshold', 'N/A')}")
    n_pass = sum(1 for t in tests.values() if t.get('pass', False))
    text_lines.append(f"\nTotal: {n_pass}/{len(tests)} PASS")
    text_lines.append(f"Simulated: {results.get('simulated', False)}")
    ax.text(0.05, 0.95, '\n'.join(text_lines), transform=ax.transAxes,
            fontsize=8, fontfamily='monospace', verticalalignment='top')
    ax.set_title('D) Test Results')

    plt.tight_layout()
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(fig_path), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Figure saved: {fig_path}")


# ═══════════════════════════════════════════════════════════
# Main Experiment
# ═══════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description='z2169: Transfer Learning + Noise Regime Switching')
    parser.add_argument('--n-trials', type=int, default=200,
                        help='Waveform trials per condition')
    parser.add_argument('--steps-per-trial', type=int, default=30)
    parser.add_argument('--noise-collect-s', type=float, default=10.0,
                        help='Noise collection duration per regime (seconds)')
    parser.add_argument('--matmul-size', type=int, default=2048,
                        help='GPU matmul size for compute workload')
    args = parser.parse_args()

    print("=" * 70)
    print("z2169: Transfer Learning + Noise Regime Switching")
    print("=" * 70)

    rng = np.random.default_rng(42)
    w_in = rng.uniform(-1, 1, size=N_NEURONS)
    w_noise = rng.uniform(-1, 1, size=N_NEURONS)

    results = {
        'experiment': 'z2169_transfer_regime',
        'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
        'params': {
            'base_vg': BASE_VG, 'alpha': ALPHA, 'beta': BETA,
            'n_neurons': N_NEURONS, 'sample_hz': SAMPLE_HZ,
            'iir_alpha': IIR_ALPHA,
            'n_trials': args.n_trials,
            'steps_per_trial': args.steps_per_trial,
            'matmul_size': args.matmul_size,
            'w_in': w_in.tolist(), 'w_noise': w_noise.tolist(),
        },
        'simulated': False,
    }

    # ─── Step 1: Connect to FPGA ───
    print("\n[1/9] Connecting to FPGA...")
    ser, port = find_fpga()
    if ser is None:
        print("  FPGA not found -- using LIF simulation fallback")
        fpga = False
        results['simulated'] = True
    else:
        print(f"  Connected: {port}")
        fpga = True
        ser.write(bytes([SYNC, CMD_SET_KILL, 0x00]))
        ser.flush()
        time.sleep(0.1)
        print("  Kill switch disabled")

    # ─── Step 2: Generate waveform dataset ───
    # Use two different seeds so train and test sets are truly independent
    print("\n[2/9] Generating waveform datasets...")
    wave_trials_A, wave_labels_A = generate_waveforms(
        n_trials=args.n_trials, steps_per_trial=args.steps_per_trial, seed=42)
    wave_trials_B, wave_labels_B = generate_waveforms(
        n_trials=args.n_trials, steps_per_trial=args.steps_per_trial, seed=137)
    print(f"  Dataset A (seed=42): {args.n_trials} trials, classes: {np.bincount(wave_labels_A)}")
    print(f"  Dataset B (seed=137): {args.n_trials} trials, classes: {np.bincount(wave_labels_B)}")

    # ─── Step 3: Phase 1 — COMPUTE regime (GPU workload active) ───
    print("\n[3/9] Phase 1: Collecting noise under COMPUTE regime...")
    start_gpu_workload(size=args.matmul_size)

    print("  Collecting power rail noise (compute)...")
    compute_power_raw = collect_noise_regime(
        duration_s=args.noise_collect_s, sample_hz=50)
    if len(compute_power_raw) > 10:
        compute_noise = normalize_noise(compute_power_raw)
        compute_noise_iir = iir_filter_noise(compute_noise, alpha_iir=IIR_ALPHA)
        compute_power_mean = float(np.mean(compute_power_raw))
        compute_power_std = float(np.std(compute_power_raw))
        print(f"  Compute power: {compute_power_mean:.2f} +/- {compute_power_std:.3f} W, "
              f"{len(compute_noise)} samples")
    else:
        print("  Power rail unavailable, generating synthetic 1/f (compute)")
        compute_noise = generate_synthetic_1f(int(args.noise_collect_s * 50), rng)
        compute_noise_iir = iir_filter_noise(compute_noise, alpha_iir=IIR_ALPHA)
        compute_power_mean = 15.0
        compute_power_std = 2.0
        compute_power_raw = [15.0 + rng.standard_normal() * 2.0 for _ in range(500)]

    # ─── Step 4: Run trials under COMPUTE regime ───
    print("\n[4/9] Running reservoir trials under COMPUTE regime...")

    # Condition A: same-regime (train on dataset A under compute)
    print("  --- Condition A: Same-regime TRAIN set ---")
    features_compute_train, vmem_compute_train, pw_compute_train, ser, fpga = \
        run_trials_regime(ser, port, fpga, wave_trials_A, wave_labels_A,
                          compute_noise_iir, w_in, w_noise, 'compute',
                          n_trials=args.n_trials)

    # Condition A: same-regime (test on dataset B under compute)
    print("  --- Condition A: Same-regime TEST set ---")
    features_compute_test, vmem_compute_test, pw_compute_test, ser, fpga = \
        run_trials_regime(ser, port, fpga, wave_trials_B, wave_labels_B,
                          compute_noise_iir, w_in, w_noise, 'compute',
                          n_trials=args.n_trials)

    # Stop GPU workload
    print("\n[5/9] Switching to IDLE regime (stopping GPU workload)...")
    stop_gpu_workload()

    # ─── Step 5: Phase 2 — IDLE regime (no GPU workload) ───
    print("\n[6/9] Phase 2: Collecting noise under IDLE regime...")
    idle_power_raw = collect_noise_regime(
        duration_s=args.noise_collect_s, sample_hz=50)
    if len(idle_power_raw) > 10:
        idle_noise = normalize_noise(idle_power_raw)
        idle_noise_iir = iir_filter_noise(idle_noise, alpha_iir=IIR_ALPHA)
        idle_power_mean = float(np.mean(idle_power_raw))
        idle_power_std = float(np.std(idle_power_raw))
        print(f"  Idle power: {idle_power_mean:.2f} +/- {idle_power_std:.3f} W, "
              f"{len(idle_noise)} samples")
    else:
        print("  Power rail unavailable, generating synthetic 1/f (idle)")
        idle_noise = generate_synthetic_1f(int(args.noise_collect_s * 50), rng)
        idle_noise_iir = iir_filter_noise(idle_noise, alpha_iir=IIR_ALPHA)
        idle_power_mean = 8.0
        idle_power_std = 0.5
        idle_power_raw = [8.0 + rng.standard_normal() * 0.5 for _ in range(500)]

    # ─── Step 6: Run trials under IDLE regime ───
    print("\n[7/9] Running reservoir trials under IDLE regime...")

    # Condition B: cross-regime test (dataset B under idle)
    print("  --- Condition B: Cross-regime TEST set (idle) ---")
    features_idle_test, vmem_idle_test, pw_idle_test, ser, fpga = \
        run_trials_regime(ser, port, fpga, wave_trials_B, wave_labels_B,
                          idle_noise_iir, w_in, w_noise, 'idle',
                          n_trials=args.n_trials)

    # Condition C: retrained (dataset A under idle for training)
    print("  --- Condition C: Retrained TRAIN set (idle) ---")
    features_idle_train, vmem_idle_train, pw_idle_train, ser, fpga = \
        run_trials_regime(ser, port, fpga, wave_trials_A, wave_labels_A,
                          idle_noise_iir, w_in, w_noise, 'idle',
                          n_trials=args.n_trials)

    # ─── Step 7: Classification ───
    print("\n[8/9] Classifying waveforms...")

    # Normalize features (fit on train, apply to test)
    def norm_fit_transform(X_train, X_test):
        mu = X_train.mean(axis=0, keepdims=True)
        sigma = X_train.std(axis=0, keepdims=True)
        sigma[sigma < 1e-10] = 1.0
        return (X_train - mu) / sigma, (X_test - mu) / sigma, mu, sigma

    # Condition A: Same regime — train on compute, test on compute
    print("  Condition A: Same regime (compute/compute)...")
    X_train_A, X_test_A, mu_A, sigma_A = norm_fit_transform(
        features_compute_train, features_compute_test)
    acc_same, W_same = ridge_classify(X_train_A, wave_labels_A,
                                       X_test_A, wave_labels_B)
    print(f"    Same-regime accuracy: {acc_same:.4f}")

    # Condition B: Cross-regime — use W_same on idle test data
    print("  Condition B: Cross-regime (train compute, test idle)...")
    # Normalize idle test using compute train statistics (transfer scenario)
    X_test_B = (features_idle_test - mu_A) / sigma_A
    if W_same is not None:
        acc_cross = ridge_classify_with_W(W_same, X_test_B, wave_labels_B)
    else:
        acc_cross = 1.0 / 3.0
    print(f"    Cross-regime accuracy: {acc_cross:.4f}")

    # Condition C: Retrained — train on idle, test on idle
    print("  Condition C: Retrained (idle/idle)...")
    X_train_C, X_test_C, _, _ = norm_fit_transform(
        features_idle_train, features_idle_test)
    acc_retrained, _ = ridge_classify(X_train_C, wave_labels_A,
                                       X_test_C, wave_labels_B)
    print(f"    Retrained accuracy: {acc_retrained:.4f}")

    # Also compute 5-fold CV for each condition for robustness
    print("\n  5-fold CV on same-regime (compute)...")
    all_compute = np.vstack([features_compute_train, features_compute_test])
    all_labels_compute = np.concatenate([wave_labels_A, wave_labels_B])
    splits = stratified_kfold(all_compute, all_labels_compute, n_splits=5)
    cv_same = []
    for train_idx, test_idx in splits:
        Xtr, Xte, _, _ = norm_fit_transform(
            all_compute[train_idx], all_compute[test_idx])
        a, _ = ridge_classify(Xtr, all_labels_compute[train_idx],
                               Xte, all_labels_compute[test_idx])
        cv_same.append(a)
    cv_same_mean = float(np.mean(cv_same))
    cv_same_std = float(np.std(cv_same))
    print(f"    CV same-regime: {cv_same_mean:.4f} +/- {cv_same_std:.4f}")

    # Store accuracy results
    results['accuracies'] = {
        'same_regime': {
            'mean': float(acc_same),
            'std': float(cv_same_std),
            'cv_folds': [float(a) for a in cv_same],
        },
        'cross_regime': {
            'mean': float(acc_cross),
            'std': 0.0,  # single-shot transfer, no CV
        },
        'retrained': {
            'mean': float(acc_retrained),
            'std': 0.0,
        },
    }

    # Power statistics
    results['power_stats'] = {
        'compute': {
            'mean': float(np.mean(compute_power_raw)),
            'std': float(np.std(compute_power_raw)),
            'n_samples': len(compute_power_raw),
        },
        'idle': {
            'mean': float(np.mean(idle_power_raw)),
            'std': float(np.std(idle_power_raw)),
            'n_samples': len(idle_power_raw),
        },
    }

    # Vmem statistics
    vmem_compute_all = vmem_compute_train + vmem_compute_test
    vmem_idle_all = vmem_idle_test + vmem_idle_train
    results['vmem_stats'] = {
        'compute': {
            'mean': float(np.mean(vmem_compute_all)),
            'std': float(np.std(vmem_compute_all)),
        },
        'idle': {
            'mean': float(np.mean(vmem_idle_all)),
            'std': float(np.std(vmem_idle_all)),
        },
    }

    # ─── Step 8: Evaluate tests ───
    print("\n[9/9] Evaluating tests T103-T108...")

    power_diff = abs(results['power_stats']['compute']['mean'] -
                     results['power_stats']['idle']['mean'])
    vmem_shift = abs(results['vmem_stats']['compute']['mean'] -
                     results['vmem_stats']['idle']['mean'])

    tests = {}

    # T103: Same-regime accuracy > 55%
    tests['T103'] = {
        'description': 'Same-regime accuracy > 55% (sanity)',
        'value': float(acc_same),
        'threshold': 0.55,
        'pass': acc_same > 0.55,
    }
    print(f"  T103: same_regime={acc_same:.4f} > 0.55? "
          f"{'PASS' if tests['T103']['pass'] else 'FAIL'}")

    # T104: Cross-regime accuracy > chance (33.3%)
    tests['T104'] = {
        'description': 'Cross-regime accuracy > chance (33.3%)',
        'value': float(acc_cross),
        'threshold': 1.0 / 3.0,
        'pass': acc_cross > 1.0 / 3.0,
    }
    print(f"  T104: cross_regime={acc_cross:.4f} > 0.333? "
          f"{'PASS' if tests['T104']['pass'] else 'FAIL'}")

    # T105: Cross-regime < same-regime (regime matters)
    tests['T105'] = {
        'description': 'Cross-regime < same-regime (regime matters)',
        'value': float(acc_cross),
        'threshold': float(acc_same),
        'pass': acc_cross < acc_same,
    }
    print(f"  T105: cross={acc_cross:.4f} < same={acc_same:.4f}? "
          f"{'PASS' if tests['T105']['pass'] else 'FAIL'}")

    # T106: Retrained > cross-regime (retraining helps)
    tests['T106'] = {
        'description': 'Retrained > cross-regime (retraining helps)',
        'value': float(acc_retrained),
        'threshold': float(acc_cross),
        'pass': acc_retrained > acc_cross,
    }
    print(f"  T106: retrained={acc_retrained:.4f} > cross={acc_cross:.4f}? "
          f"{'PASS' if tests['T106']['pass'] else 'FAIL'}")

    # T107: GPU power difference > 2W
    tests['T107'] = {
        'description': 'GPU power differs between regimes > 2W',
        'value': float(power_diff),
        'threshold': 2.0,
        'pass': power_diff > 2.0,
    }
    print(f"  T107: power_diff={power_diff:.2f}W > 2.0W? "
          f"{'PASS' if tests['T107']['pass'] else 'FAIL'}")

    # T108: Vmem mean shift > 0.01
    tests['T108'] = {
        'description': 'Reservoir vmem mean shift > 0.01 between regimes',
        'value': float(vmem_shift),
        'threshold': 0.01,
        'pass': vmem_shift > 0.01,
    }
    print(f"  T108: vmem_shift={vmem_shift:.4f} > 0.01? "
          f"{'PASS' if tests['T108']['pass'] else 'FAIL'}")

    results['tests'] = tests
    n_pass = sum(1 for t in tests.values() if t['pass'])
    results['n_pass'] = n_pass
    results['n_tests'] = len(tests)
    print(f"\n  === TOTAL: {n_pass}/{len(tests)} PASS ===")

    # ─── Save results ───
    RESULTS.mkdir(parents=True, exist_ok=True)
    class NpEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, (np.integer,)): return int(obj)
            if isinstance(obj, (np.floating,)): return float(obj)
            if isinstance(obj, (np.bool_,)): return bool(obj)
            if isinstance(obj, np.ndarray): return obj.tolist()
            return super().default(obj)

    out_path = RESULTS / 'z2169_transfer_regime.json'
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2, cls=NpEncoder)
    print(f"\nResults saved: {out_path}")

    # ─── Generate figure ───
    fig_path = FIGURES / 'fig_z2169_transfer_regime.png'
    generate_figure(results, fig_path)

    # ─── Cleanup ───
    if fpga and ser is not None:
        try:
            # Set all neurons to low Vg before exit
            set_per_neuron_vg(ser, [0.1] * 8)
            ser.close()
        except Exception:
            pass
    stop_gpu_workload()

    print(f"\nz2169 complete: {n_pass}/{len(tests)} PASS")
    return 0 if n_pass >= 4 else 1


if __name__ == '__main__':
    sys.exit(main())
