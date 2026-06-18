#!/usr/bin/env python3
"""z2211_thermal_coupling.py — Thermal Cross-Substrate Coupling

PHYSICAL SETUP: Arty A7-100T FPGA placed directly on HP Z2 Mini G1a heatsink,
in thermal contact with GPU/VRM thermal environment.

HYPOTHESIS: GPU thermal fluctuations propagate to FPGA die via conduction,
modulating avalanche noise characteristics and spike rates. This creates
a genuine physical coupling channel between substrates — temperature IS
the cross-substrate language.

PROTOCOL:
  Phase 1: Baseline — record FPGA spike rates + GPU temp at steady state (60s)
  Phase 2: Thermal stress — GPU matmul bursts (10s ON / 10s OFF × 6 cycles)
  Phase 3: Cool-down — record during natural cooling (60s)

MEASUREMENTS:
  - GPU die temperature (hwmon7/temp1_input, 1kHz readout)
  - GPU power draw (hwmon7/power1_average)
  - FPGA spike rates per neuron (telemetry at ~20Hz)
  - Ambient temp (ACPI thermal zone)
  - SMN PM table thermal (hotspot)

ANALYSIS:
  T367: Cross-correlation between GPU temp and FPGA mean spike rate > 0.15
  T368: Phase lag between GPU temp peak and FPGA spike rate peak = 2-30s (thermal mass)
  T369: GPU thermal stress increases FPGA spike rate variance > 2× baseline
  T370: Spike rate during GPU-hot periods differs from GPU-cool periods (t-test p<0.05)
  T371: FPGA spike rate PSD during stress has lower-frequency power than baseline
  T372: Mutual information between GPU temp series and FPGA spike series > 0.01 bits

Hardware: AMD gfx1151 GPU + Arty A7-100T FPGA (128-neuron) physically on heatsink
"""

import os, sys, json, time, struct, threading
import numpy as np
from pathlib import Path
from scipy import signal, stats

os.environ.setdefault('HSA_OVERRIDE_GFX_VERSION', '11.0.0')

# ═══════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════

SCRIPT_DIR = Path(__file__).parent
ROOT = SCRIPT_DIR.parent
RESULTS_DIR = ROOT / 'results'
RESULTS_DIR.mkdir(exist_ok=True)

N_NEURONS = 128
BASE_VG = 0.58  # Suprathreshold — sensitive to thermal modulation
SAMPLE_HZ = 10  # FPGA telemetry sample rate (limited by UART bandwidth)

PM_TABLE_PATH = '/sys/kernel/ryzen_smu_drv/pm_table'
HWMON_POWER = '/sys/class/hwmon/hwmon7/power1_average'
HWMON_TEMP = '/sys/class/hwmon/hwmon7/temp1_input'
ACPI_TEMP = '/sys/class/hwmon/hwmon0/temp1_input'

# Timing
BASELINE_SECS = 60
STRESS_ON_SECS = 10
STRESS_OFF_SECS = 10
STRESS_CYCLES = 6
COOLDOWN_SECS = 60

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer): return int(obj)
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        if isinstance(obj, np.bool_): return bool(obj)
        return super().default(obj)


# ═══════════════════════════════════════════════════════════
# Sensor reads
# ═══════════════════════════════════════════════════════════

def read_gpu_temp():
    try: return int(open(HWMON_TEMP).read().strip()) / 1000.0
    except: return None

def read_gpu_power():
    try: return int(open(HWMON_POWER).read().strip()) / 1e6
    except: return None

def read_acpi_temp():
    try: return int(open(ACPI_TEMP).read().strip()) / 1000.0
    except: return None

def read_smn_thermal():
    try:
        with open(PM_TABLE_PATH, 'rb') as f:
            f.seek(0x4C)
            return struct.unpack('<f', f.read(4))[0]
    except: return None


# ═══════════════════════════════════════════════════════════
# FPGA connection
# ═══════════════════════════════════════════════════════════

def connect_fpga():
    """Connect to 128-neuron FPGA."""
    sys.path.insert(0, str(SCRIPT_DIR))
    from fpga_host_v2 import FPGABridge

    for port in ['/dev/ttyUSB1', '/dev/ttyUSB0']:
        try:
            fpga = FPGABridge(port)
            t = fpga.read_telem(timeout=0.3)
            if t and len(t) >= N_NEURONS:
                print(f"  Connected: {port}, {len(t)} neurons")
                return fpga
        except:
            continue
    return None


# ═══════════════════════════════════════════════════════════
# GPU thermal stress generator
# ═══════════════════════════════════════════════════════════

gpu_stress_active = False

def gpu_stress_worker(duration_secs):
    """Run heavy GPU matmul to generate heat."""
    import torch
    global gpu_stress_active
    gpu_stress_active = True
    device = torch.device('cuda')
    end_time = time.time() + duration_secs

    # Large matrix multiply — maximize thermal output
    a = torch.randn(2048, 2048, device=device)
    b = torch.randn(2048, 2048, device=device)

    while time.time() < end_time and gpu_stress_active:
        _ = torch.mm(a, b)
        torch.cuda.synchronize()

    gpu_stress_active = False


# ═══════════════════════════════════════════════════════════
# Data collection
# ═══════════════════════════════════════════════════════════

def collect_thermal_trace(fpga, duration_secs, label, gpu_stress=False):
    """Collect synchronized GPU temp + FPGA spike rate trace.

    Returns dict with arrays:
      timestamps, gpu_temp, gpu_power, acpi_temp, smn_temp,
      fpga_mean_rate, fpga_rates[n_samples, N_NEURONS], phase
    """
    print(f"  [{label}] Collecting {duration_secs}s of data...")

    timestamps = []
    gpu_temps = []
    gpu_powers = []
    acpi_temps = []
    smn_temps = []
    fpga_rates_all = []
    phases = []

    # Get initial spike counts
    prev_telem = fpga.read_telem(timeout=0.5)
    if not prev_telem or len(prev_telem) < N_NEURONS:
        print(f"    WARNING: No initial telemetry")
        return None

    prev_counts = np.array([prev_telem[i]['spike_count'] for i in range(N_NEURONS)])
    t_start = time.time()
    t_prev = t_start
    sample_interval = 1.0 / SAMPLE_HZ

    # Optional GPU stress thread
    stress_thread = None
    if gpu_stress:
        stress_thread = threading.Thread(target=gpu_stress_worker, args=(duration_secs,))
        stress_thread.start()

    n_samples = 0
    while (time.time() - t_start) < duration_secs:
        t_now = time.time()

        # Read all sensors
        gpu_t = read_gpu_temp()
        gpu_p = read_gpu_power()
        acpi_t = read_acpi_temp()
        smn_t = read_smn_thermal()

        # Read FPGA telemetry
        telem = fpga.read_telem(timeout=0.3)
        if telem and len(telem) >= N_NEURONS:
            counts = np.array([telem[i]['spike_count'] for i in range(N_NEURONS)])
            dt = t_now - t_prev
            if dt > 0:
                deltas = (counts - prev_counts) & 0xFFFF
                deltas[deltas > 30000] = 0
                rates = deltas / max(dt, 0.01)

                timestamps.append(t_now - t_start)
                gpu_temps.append(gpu_t)
                gpu_powers.append(gpu_p)
                acpi_temps.append(acpi_t)
                smn_temps.append(smn_t)
                fpga_rates_all.append(rates)
                phases.append('stress' if gpu_stress else 'idle')

                prev_counts = counts.copy()
                t_prev = t_now
                n_samples += 1

                if n_samples % (SAMPLE_HZ * 10) == 0:
                    mean_rate = np.mean(rates)
                    print(f"    {n_samples} samples, gpu_temp={gpu_t:.1f}°C, "
                          f"fpga_mean_rate={mean_rate:.1f}, power={gpu_p:.1f}W")

        # Pace at sample rate
        elapsed = time.time() - t_now
        if elapsed < sample_interval:
            time.sleep(sample_interval - elapsed)

    if stress_thread:
        global gpu_stress_active
        gpu_stress_active = False
        stress_thread.join(timeout=5)

    if n_samples == 0:
        return None

    return {
        'label': label,
        'timestamps': np.array(timestamps),
        'gpu_temp': np.array(gpu_temps, dtype=float),
        'gpu_power': np.array(gpu_powers, dtype=float),
        'acpi_temp': np.array(acpi_temps, dtype=float),
        'smn_temp': np.array(smn_temps, dtype=float),
        'fpga_rates': np.array(fpga_rates_all),
        'fpga_mean_rate': np.array([r.mean() for r in fpga_rates_all]),
        'phases': phases,
        'n_samples': n_samples,
    }


def collect_stress_cycles(fpga, n_cycles, on_secs, off_secs):
    """Alternating GPU stress ON/OFF cycles with continuous FPGA recording."""
    print(f"  Stress protocol: {n_cycles} cycles × ({on_secs}s ON / {off_secs}s OFF)")

    all_data = {
        'timestamps': [], 'gpu_temp': [], 'gpu_power': [],
        'acpi_temp': [], 'smn_temp': [],
        'fpga_rates': [], 'fpga_mean_rate': [],
        'phases': [], 'cycle_markers': [],
    }

    prev_telem = fpga.read_telem(timeout=0.5)
    if not prev_telem or len(prev_telem) < N_NEURONS:
        print("    WARNING: No initial telemetry")
        return None

    prev_counts = np.array([prev_telem[i]['spike_count'] for i in range(N_NEURONS)])
    t_global_start = time.time()
    t_prev = t_global_start
    sample_interval = 1.0 / SAMPLE_HZ

    for cycle in range(n_cycles):
        for phase_name, phase_secs in [('stress_on', on_secs), ('stress_off', off_secs)]:
            print(f"    Cycle {cycle+1}/{n_cycles} — {phase_name} ({phase_secs}s)")

            stress_thread = None
            if phase_name == 'stress_on':
                import torch
                stress_thread = threading.Thread(target=gpu_stress_worker, args=(phase_secs,))
                stress_thread.start()

            phase_start = time.time()
            while (time.time() - phase_start) < phase_secs:
                t_now = time.time()

                gpu_t = read_gpu_temp()
                gpu_p = read_gpu_power()
                acpi_t = read_acpi_temp()
                smn_t = read_smn_thermal()

                telem = fpga.read_telem(timeout=0.3)
                if telem and len(telem) >= N_NEURONS:
                    counts = np.array([telem[i]['spike_count'] for i in range(N_NEURONS)])
                    dt = t_now - t_prev
                    if dt > 0:
                        deltas = (counts - prev_counts) & 0xFFFF
                        deltas[deltas > 30000] = 0
                        rates = deltas / max(dt, 0.01)

                        all_data['timestamps'].append(t_now - t_global_start)
                        all_data['gpu_temp'].append(gpu_t)
                        all_data['gpu_power'].append(gpu_p)
                        all_data['acpi_temp'].append(acpi_t)
                        all_data['smn_temp'].append(smn_t)
                        all_data['fpga_rates'].append(rates)
                        all_data['fpga_mean_rate'].append(rates.mean())
                        all_data['phases'].append(phase_name)
                        all_data['cycle_markers'].append(cycle)

                        prev_counts = counts.copy()
                        t_prev = t_now

                elapsed = time.time() - t_now
                if elapsed < sample_interval:
                    time.sleep(sample_interval - elapsed)

            if stress_thread:
                global gpu_stress_active
                gpu_stress_active = False
                stress_thread.join(timeout=5)
                time.sleep(0.5)  # brief settle

    # Convert to arrays
    for key in ['timestamps', 'gpu_temp', 'gpu_power', 'acpi_temp', 'smn_temp', 'fpga_mean_rate']:
        all_data[key] = np.array(all_data[key], dtype=float)
    all_data['fpga_rates'] = np.array(all_data['fpga_rates'])
    all_data['n_samples'] = len(all_data['timestamps'])

    return all_data


# ═══════════════════════════════════════════════════════════
# Analysis
# ═══════════════════════════════════════════════════════════

def compute_cross_correlation(gpu_temp, fpga_rate, sample_hz):
    """Compute normalized cross-correlation and find peak lag."""
    # Detrend both signals
    gpu_dt = gpu_temp - np.mean(gpu_temp)
    fpga_dt = fpga_rate - np.mean(fpga_rate)

    # Normalize
    gpu_norm = gpu_dt / (np.std(gpu_dt) + 1e-10)
    fpga_norm = fpga_dt / (np.std(fpga_dt) + 1e-10)

    # Cross-correlation
    n = len(gpu_norm)
    xcorr = np.correlate(fpga_norm, gpu_norm, mode='full') / n
    lags = np.arange(-(n-1), n) / sample_hz  # Convert to seconds

    # Find peak in positive lag range (FPGA responds AFTER GPU temp change)
    pos_mask = (lags >= 0) & (lags <= 60)  # 0-60s lag
    if pos_mask.sum() > 0:
        pos_xcorr = xcorr[pos_mask]
        pos_lags = lags[pos_mask]
        peak_idx = np.argmax(np.abs(pos_xcorr))
        peak_lag = pos_lags[peak_idx]
        peak_corr = pos_xcorr[peak_idx]
    else:
        peak_lag = 0
        peak_corr = 0

    # Overall max correlation
    max_corr = np.max(np.abs(xcorr))

    return {
        'peak_corr': float(peak_corr),
        'peak_lag_s': float(peak_lag),
        'max_corr': float(max_corr),
    }


def compute_mutual_information(x, y, n_bins=16):
    """Compute mutual information between two time series."""
    # Digitize into bins
    x_bins = np.digitize(x, np.linspace(x.min() - 1e-10, x.max() + 1e-10, n_bins + 1))
    y_bins = np.digitize(y, np.linspace(y.min() - 1e-10, y.max() + 1e-10, n_bins + 1))

    # Joint histogram
    joint = np.zeros((n_bins + 2, n_bins + 2))
    for xi, yi in zip(x_bins, y_bins):
        joint[xi, yi] += 1
    joint /= joint.sum()

    # Marginals
    px = joint.sum(axis=1)
    py = joint.sum(axis=0)

    # MI
    mi = 0
    for i in range(joint.shape[0]):
        for j in range(joint.shape[1]):
            if joint[i, j] > 0 and px[i] > 0 and py[j] > 0:
                mi += joint[i, j] * np.log2(joint[i, j] / (px[i] * py[j]))
    return max(mi, 0)


def analyze_stress_response(stress_data):
    """Analyze FPGA response to GPU thermal stress cycles."""
    phases = np.array(stress_data['phases'])
    fpga_rates = stress_data['fpga_mean_rate']

    on_mask = phases == 'stress_on'
    off_mask = phases == 'stress_off'

    rates_on = fpga_rates[on_mask]
    rates_off = fpga_rates[off_mask]

    # T-test: do spike rates differ between stress ON and OFF?
    if len(rates_on) > 5 and len(rates_off) > 5:
        t_stat, p_val = stats.ttest_ind(rates_on, rates_off)
    else:
        t_stat, p_val = 0, 1.0

    return {
        'mean_rate_stress_on': float(np.mean(rates_on)) if len(rates_on) > 0 else 0,
        'mean_rate_stress_off': float(np.mean(rates_off)) if len(rates_off) > 0 else 0,
        'std_rate_stress_on': float(np.std(rates_on)) if len(rates_on) > 0 else 0,
        'std_rate_stress_off': float(np.std(rates_off)) if len(rates_off) > 0 else 0,
        'ttest_t': float(t_stat),
        'ttest_p': float(p_val),
        'rate_ratio': float(np.mean(rates_on) / max(np.mean(rates_off), 1e-10)),
    }


# ═══════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════

def main():
    print("=" * 70)
    print("z2211: Thermal Cross-Substrate Coupling")
    print("  FPGA physically on GPU heatsink — testing thermal cross-talk")
    print(f"  Protocol: {BASELINE_SECS}s baseline → {STRESS_CYCLES}× ({STRESS_ON_SECS}s stress/"
          f"{STRESS_OFF_SECS}s rest) → {COOLDOWN_SECS}s cooldown")
    print("=" * 70)

    results = {
        'experiment': 'z2211_thermal_coupling',
        'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
        'physical_setup': 'FPGA on GPU heatsink (thermal contact)',
    }

    # ─── Connect FPGA ───
    print("\n[1/6] Connecting to FPGA...")
    fpga = connect_fpga()
    if not fpga:
        print("  FATAL: No FPGA found")
        return

    # Set all neurons to suprathreshold Vg
    vg_all = [BASE_VG] * N_NEURONS
    fpga.set_vg_all(vg_all)
    time.sleep(1.0)

    # Quick health check
    t1 = fpga.read_telem(timeout=0.5)
    time.sleep(0.5)
    t2 = fpga.read_telem(timeout=0.5)
    if t1 and t2 and len(t1) >= N_NEURONS:
        deltas = [(t2[i]['spike_count'] - t1[i]['spike_count']) & 0xFFFF for i in range(N_NEURONS)]
        active = sum(1 for d in deltas if 0 < d < 30000)
        mean_rate = np.mean([d for d in deltas if d < 30000])
        print(f"  Health: mean_rate={mean_rate:.1f}, active={active}/{N_NEURONS}")
        results['health'] = {'mean_rate': float(mean_rate), 'active': active}

    # Initial thermal state
    gpu_t0 = read_gpu_temp()
    acpi_t0 = read_acpi_temp()
    smn_t0 = read_smn_thermal()
    print(f"  Initial temps: GPU={gpu_t0}°C, ACPI={acpi_t0}°C, SMN={smn_t0}°C")
    results['initial_temps'] = {'gpu': gpu_t0, 'acpi': acpi_t0, 'smn': smn_t0}

    # ─── Phase 1: Baseline ───
    print(f"\n[2/6] Phase 1: Baseline ({BASELINE_SECS}s, GPU idle)...")
    baseline = collect_thermal_trace(fpga, BASELINE_SECS, "baseline")
    if baseline:
        results['baseline'] = {
            'n_samples': baseline['n_samples'],
            'gpu_temp_mean': float(np.nanmean(baseline['gpu_temp'])),
            'gpu_temp_std': float(np.nanstd(baseline['gpu_temp'])),
            'fpga_rate_mean': float(np.mean(baseline['fpga_mean_rate'])),
            'fpga_rate_std': float(np.std(baseline['fpga_mean_rate'])),
            'fpga_rate_cv': float(np.std(baseline['fpga_mean_rate']) /
                                   max(np.mean(baseline['fpga_mean_rate']), 1e-10)),
        }
        print(f"  Baseline: GPU={results['baseline']['gpu_temp_mean']:.1f}±"
              f"{results['baseline']['gpu_temp_std']:.2f}°C, "
              f"FPGA rate={results['baseline']['fpga_rate_mean']:.1f}±"
              f"{results['baseline']['fpga_rate_std']:.1f}")

    # ─── Phase 2: Stress cycles ───
    print(f"\n[3/6] Phase 2: Thermal stress cycles...")
    stress = collect_stress_cycles(fpga, STRESS_CYCLES, STRESS_ON_SECS, STRESS_OFF_SECS)
    if stress:
        stress_analysis = analyze_stress_response(stress)
        results['stress'] = {
            'n_samples': stress['n_samples'],
            'gpu_temp_range': [float(np.nanmin(stress['gpu_temp'])),
                              float(np.nanmax(stress['gpu_temp']))],
            **stress_analysis,
        }
        print(f"  Stress ON:  rate={stress_analysis['mean_rate_stress_on']:.1f}±"
              f"{stress_analysis['std_rate_stress_on']:.1f}")
        print(f"  Stress OFF: rate={stress_analysis['mean_rate_stress_off']:.1f}±"
              f"{stress_analysis['std_rate_stress_off']:.1f}")
        print(f"  T-test: t={stress_analysis['ttest_t']:.3f}, p={stress_analysis['ttest_p']:.4f}")

    # ─── Phase 3: Cooldown ───
    print(f"\n[4/6] Phase 3: Cooldown ({COOLDOWN_SECS}s)...")
    cooldown = collect_thermal_trace(fpga, COOLDOWN_SECS, "cooldown")
    if cooldown:
        results['cooldown'] = {
            'n_samples': cooldown['n_samples'],
            'gpu_temp_start': float(cooldown['gpu_temp'][0]) if len(cooldown['gpu_temp']) > 0 else None,
            'gpu_temp_end': float(cooldown['gpu_temp'][-1]) if len(cooldown['gpu_temp']) > 0 else None,
            'fpga_rate_mean': float(np.mean(cooldown['fpga_mean_rate'])),
        }

    # ─── Analysis ───
    print(f"\n[5/6] Cross-substrate analysis...")

    # Combine stress data for cross-correlation
    if stress and stress['n_samples'] > 20:
        gpu_temp_series = stress['gpu_temp']
        fpga_rate_series = stress['fpga_mean_rate']

        # Replace NaN
        gpu_temp_series = np.nan_to_num(gpu_temp_series, nan=np.nanmean(gpu_temp_series))

        # T367: Cross-correlation
        xcorr = compute_cross_correlation(gpu_temp_series, fpga_rate_series, SAMPLE_HZ)
        results['cross_correlation'] = xcorr
        print(f"  Cross-corr: peak={xcorr['peak_corr']:.4f} at lag={xcorr['peak_lag_s']:.1f}s, "
              f"max={xcorr['max_corr']:.4f}")

        # T372: Mutual information
        mi = compute_mutual_information(gpu_temp_series, fpga_rate_series)
        results['mutual_information'] = float(mi)
        print(f"  MI(GPU_temp, FPGA_rate) = {mi:.4f} bits")

        # T369: Variance comparison
        if baseline and baseline['n_samples'] > 10:
            baseline_var = np.var(baseline['fpga_mean_rate'])
            stress_var = np.var(fpga_rate_series)
            var_ratio = stress_var / max(baseline_var, 1e-10)
            results['variance_ratio'] = float(var_ratio)
            print(f"  Variance ratio (stress/baseline): {var_ratio:.2f}×")

        # T371: PSD comparison
        if baseline and baseline['n_samples'] > 20:
            # Baseline PSD
            f_b, psd_b = signal.welch(baseline['fpga_mean_rate'], fs=SAMPLE_HZ, nperseg=min(64, len(baseline['fpga_mean_rate'])))
            # Stress PSD
            f_s, psd_s = signal.welch(fpga_rate_series, fs=SAMPLE_HZ, nperseg=min(64, len(fpga_rate_series)))

            # Low-frequency power ratio (< 0.1 Hz)
            lf_mask_b = f_b < 0.1
            lf_mask_s = f_s < 0.1
            lf_power_baseline = np.sum(psd_b[lf_mask_b]) if lf_mask_b.sum() > 0 else 0
            lf_power_stress = np.sum(psd_s[lf_mask_s]) if lf_mask_s.sum() > 0 else 0
            lf_ratio = lf_power_stress / max(lf_power_baseline, 1e-10)
            results['lf_power_ratio'] = float(lf_ratio)
            print(f"  LF power ratio (stress/baseline): {lf_ratio:.2f}×")

        # Per-neuron thermal sensitivity
        # Find neurons most/least correlated with GPU temp
        n_check = min(N_NEURONS, stress['fpga_rates'].shape[1])
        neuron_corrs = []
        for ni in range(n_check):
            r, _ = stats.pearsonr(gpu_temp_series, stress['fpga_rates'][:, ni])
            neuron_corrs.append(float(r) if np.isfinite(r) else 0)

        neuron_corrs = np.array(neuron_corrs)
        results['neuron_thermal_corr'] = {
            'mean': float(np.mean(neuron_corrs)),
            'std': float(np.std(neuron_corrs)),
            'max': float(np.max(neuron_corrs)),
            'min': float(np.min(neuron_corrs)),
            'n_positive': int(np.sum(neuron_corrs > 0)),
            'n_significant': int(np.sum(np.abs(neuron_corrs) > 0.15)),
            'top5_neurons': np.argsort(np.abs(neuron_corrs))[-5:][::-1].tolist(),
            'top5_corrs': neuron_corrs[np.argsort(np.abs(neuron_corrs))[-5:][::-1]].tolist(),
        }
        print(f"  Neuron-temp correlation: mean={np.mean(neuron_corrs):.4f}, "
              f"max={np.max(np.abs(neuron_corrs)):.4f}, "
              f"{np.sum(np.abs(neuron_corrs) > 0.15)} neurons |r|>0.15")

    # ─── Tests ───
    print(f"\n[6/6] Tests...")
    tests = []

    # T367: Cross-correlation > 0.15
    xcorr_val = results.get('cross_correlation', {}).get('max_corr', 0)
    tests.append({
        'id': 'T367',
        'description': f'GPU-FPGA thermal xcorr({xcorr_val:.3f}) > 0.15',
        'passed': xcorr_val > 0.15,
    })

    # T368: Phase lag 2-30s
    lag = results.get('cross_correlation', {}).get('peak_lag_s', 0)
    tests.append({
        'id': 'T368',
        'description': f'Thermal phase lag({lag:.1f}s) in [2, 30]',
        'passed': 2 <= lag <= 30,
    })

    # T369: Variance ratio > 2
    var_r = results.get('variance_ratio', 0)
    tests.append({
        'id': 'T369',
        'description': f'Stress variance ratio({var_r:.2f}) > 2.0',
        'passed': var_r > 2.0,
    })

    # T370: T-test p < 0.05
    p_val = results.get('stress', {}).get('ttest_p', 1.0)
    tests.append({
        'id': 'T370',
        'description': f'Stress ON vs OFF t-test p({p_val:.4f}) < 0.05',
        'passed': p_val < 0.05,
    })

    # T371: LF power ratio > 1.5
    lf_r = results.get('lf_power_ratio', 0)
    tests.append({
        'id': 'T371',
        'description': f'LF power ratio({lf_r:.2f}) > 1.5 [stress adds LF]',
        'passed': lf_r > 1.5,
    })

    # T372: MI > 0.01 bits
    mi_val = results.get('mutual_information', 0)
    tests.append({
        'id': 'T372',
        'description': f'MI(GPU_temp, FPGA_rate)({mi_val:.4f}) > 0.01 bits',
        'passed': mi_val > 0.01,
    })

    results['tests'] = tests
    n_pass = sum(1 for t in tests if t['passed'])
    results['summary'] = {'pass': n_pass, 'total': len(tests)}

    print()
    for t in tests:
        tag = "PASS" if t['passed'] else "FAIL"
        print(f"  [{tag}] {t['id']}: {t['description']}")
    print(f"\n  TOTAL: {n_pass}/{len(tests)} PASS")

    # ─── Save ───
    out_path = RESULTS_DIR / 'z2211_thermal_coupling.json'
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2, cls=NpEncoder)
    print(f"\n  Saved: {out_path}")

    # Summary
    print(f"\n{'='*70}")
    print(f"z2211 THERMAL COUPLING: {n_pass}/{len(tests)} PASS")
    if xcorr_val > 0.15:
        print(f"  THERMAL CROSS-TALK DETECTED! peak_corr={xcorr_val:.3f}, lag={lag:.1f}s")
    else:
        print(f"  No significant thermal coupling detected (corr={xcorr_val:.3f})")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
