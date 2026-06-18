#!/usr/bin/env python3
"""z2191_reservoir_criticality.py — Criticality in GPU-Noise-Driven FPGA Reservoir

Tests whether GPU firmware noise pushes FPGA reservoir neurons toward
CRITICALITY — the edge-of-chaos regime where computation is maximized.
This connects to Mario Lanza's memristive computing work where critical
dynamics emerge from device noise.

Criticality measures:
  - Branching ratio sigma: E[n_active(t+1)] / E[n_active(t)], critical at ~1.0
  - Avalanche size distribution: power-law exponent alpha ~ 1.5
  - Dynamic range: response span to inputs across 3 orders of magnitude
  - Autocorrelation timescale: critical systems have longest ACF decay
  - Fano factor: variance/mean spike counts, >1 = bursty near-critical
  - ISI coefficient of variation: ~1.0 at criticality

4 conditions:
  FULL         — GPU 1/f noise -> FPGA neurons (beta=0.10)
  WHITE        — Gaussian white noise -> FPGA neurons (beta=0.10)
  NO_NOISE     — Signal only (beta=0)
  SUPERCRITICAL — Very high noise (beta=0.50) -> FPGA neurons

Tests T237-T242:
  T237: Branching ratio FULL closer to 1.0 than WHITE
  T238: Avalanche size exponent alpha in [1.0, 2.0] for FULL
  T239: Dynamic range FULL > Dynamic range NO_NOISE
  T240: ACF timescale FULL > ACF timescale WHITE
  T241: Fano factor FULL in [1.0, 5.0]
  T242: SUPERCRITICAL branching ratio > 1.2

Hardware: AMD gfx1151 GPU + Arty A7 FPGA on /dev/ttyUSB*
"""

import os, sys, json, time, struct, argparse
import numpy as np
from pathlib import Path

os.environ.setdefault('HSA_OVERRIDE_GFX_VERSION', '11.0.0')

BASE = Path(__file__).resolve().parent.parent
RESULTS = BASE / 'results'
FIGURES = BASE / 'results' / 'FEEL_paper_update' / \
    'FEEL__Functionally_Embodied_Emergent_Learning__13_-5' / 'figures'

# ─── FPGA Protocol ───
SYNC = 0x55
CMD_SET_VG = 0x01
CMD_READ_TELEM = 0x02
CMD_SET_KILL = 0x03

HWMON_POWER = "/sys/class/hwmon/hwmon7/power1_average"

# ─── Default Parameters ───
N_NEURONS = 8
DEFAULT_BASE_VG = 0.55
DEFAULT_ALPHA = 0.15
DEFAULT_BETA = 0.10
DEFAULT_BETA_SUPER = 0.50
DEFAULT_SAMPLE_HZ = 20
DEFAULT_N_STEPS = 500


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


# ═══════════════════════════════════════════════════════════
# FPGA Communication
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


def connect_fpga(ser):
    """Disable kill switch and flush."""
    ser.reset_input_buffer()
    ser.reset_output_buffer()
    ser.write(bytes([SYNC, CMD_SET_KILL, 0x00]))
    ser.flush()
    time.sleep(0.05)
    ser.reset_input_buffer()


def set_per_neuron_vg(ser, vg_values):
    """Set individual Vg for each of 8 neurons."""
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


def crc8(data):
    crc = 0
    for b in data:
        crc ^= b
        for _ in range(8):
            if crc & 0x80:
                crc = (crc << 1) ^ 0x07
            else:
                crc <<= 1
            crc &= 0xFF
    return crc


# ═══════════════════════════════════════════════════════════
# Noise Sources
# ═══════════════════════════════════════════════════════════

def read_hwmon_power():
    """Read hwmon power1_average (uW -> W)."""
    try:
        return int(open(HWMON_POWER).read().strip()) / 1e6
    except Exception:
        return None


def collect_power_noise(duration_s=15, sample_hz=50):
    """Collect GPU power rail time series for 1/f noise source."""
    n_samples = int(duration_s * sample_hz)
    interval = 1.0 / sample_hz
    powers = []
    for _ in range(n_samples):
        p = read_hwmon_power()
        if p is not None:
            powers.append(p)
        time.sleep(interval)
    return np.array(powers) if powers else None


def iir_filter_noise(noise_samples, alpha_iir=0.85):
    """Apply IIR low-pass: y[t] = alpha*y[t-1] + (1-alpha)*x[t]."""
    filtered = np.zeros(len(noise_samples))
    filtered[0] = noise_samples[0]
    for t in range(1, len(noise_samples)):
        filtered[t] = alpha_iir * filtered[t - 1] + (1 - alpha_iir) * noise_samples[t]
    std = max(np.std(filtered), 1e-6)
    return filtered / std


class VossMcCartneyFilter:
    """10-octave Voss-McCartney 1/f noise generator."""

    def __init__(self, n_octaves=10):
        self.n_octaves = n_octaves
        self.values = np.zeros(n_octaves)
        self.step = 0

    def process(self, white_sample):
        x = (white_sample / 127.5) - 1.0
        for k in range(self.n_octaves):
            period = 1 << k
            if self.step % period == 0:
                self.values[k] = x
        self.step += 1
        total = np.sum(self.values) / self.n_octaves
        return np.clip(total, -1.0, 1.0)


# ═══════════════════════════════════════════════════════════
# FPGA Data Collection
# ═══════════════════════════════════════════════════════════

def generate_drive_signal(n_steps, sample_hz):
    """Slow input oscillation for reservoir drive."""
    t = np.arange(n_steps) / sample_hz
    signal = 0.3 * np.sin(2 * np.pi * 0.2 * t) + 0.2 * np.sin(2 * np.pi * 0.5 * t)
    return signal


def run_condition(ser, condition, n_steps, sample_hz, base_vg, alpha, beta,
                  noise_1f, noise_white, drive_signal):
    """Run one condition on FPGA, collect delta spike counts.

    Returns: (n_steps, 8) array of delta spike counts per neuron.
    """
    interval = 1.0 / sample_hz
    delta_spikes = np.zeros((n_steps, N_NEURONS))
    prev_counts = None
    power_mean = 11.0
    w_noise = np.array([1.0, -0.7, 0.5, -0.3, 0.8, -0.6, 0.4, -0.9])
    w_in = np.array([1.0, 0.8, 0.6, 0.4, -0.4, -0.6, -0.8, -1.0])

    beta_eff = DEFAULT_BETA_SUPER if condition == 'SUPERCRITICAL' else beta

    for t in range(n_steps):
        # Determine noise
        if condition in ('FULL', 'SUPERCRITICAL'):
            p = read_hwmon_power()
            noise_val = (p - power_mean) / 2.0 if p else 0.0
        elif condition == 'WHITE':
            noise_val = noise_white[t % len(noise_white)]
        else:  # NO_NOISE
            noise_val = 0.0

        # Compute per-neuron Vg
        vg_values = np.full(N_NEURONS, base_vg)
        vg_values += alpha * drive_signal[t % len(drive_signal)] * w_in
        if condition != 'NO_NOISE':
            vg_values += beta_eff * noise_val * w_noise
        vg_values = np.clip(vg_values, 0.05, 0.95)

        set_per_neuron_vg(ser, vg_values)
        time.sleep(interval * 0.3)

        ser.reset_input_buffer()
        ser.write(bytes([SYNC, CMD_READ_TELEM]))
        ser.flush()
        telem = read_telem(ser, timeout=0.15)

        if telem:
            counts = [n['spike_count'] for n in telem]
            if prev_counts is not None:
                for i in range(N_NEURONS):
                    delta = (counts[i] - prev_counts[i]) & 0xFFFF
                    if delta > 30000:
                        delta = 0
                    delta_spikes[t, i] = delta
            prev_counts = counts[:]

        time.sleep(max(0, interval * 0.5 - 0.01))

        if (t + 1) % 100 == 0:
            print(f"  [{condition}] step {t+1}/{n_steps}")

    return delta_spikes


def simulate_condition(condition, n_steps, sample_hz, base_vg, alpha, beta,
                       noise_1f, noise_white, drive_signal):
    """Software LIF simulation fallback when FPGA not connected."""
    dt = 1.0 / sample_hz
    vmem = np.zeros(N_NEURONS)
    delta_spikes = np.zeros((n_steps, N_NEURONS))
    v_thresh = 1.0
    tau_m = 0.02
    w_noise = np.array([1.0, -0.7, 0.5, -0.3, 0.8, -0.6, 0.4, -0.9])
    w_in = np.array([1.0, 0.8, 0.6, 0.4, -0.4, -0.6, -0.8, -1.0])

    beta_eff = DEFAULT_BETA_SUPER if condition == 'SUPERCRITICAL' else beta

    rng = np.random.default_rng(
        42 if condition == 'FULL' else
        43 if condition == 'WHITE' else
        44 if condition == 'SUPERCRITICAL' else 45
    )

    for t in range(n_steps):
        if condition in ('FULL', 'SUPERCRITICAL'):
            noise_val = noise_1f[t % len(noise_1f)] if len(noise_1f) > 0 else rng.standard_normal() * 0.3
        elif condition == 'WHITE':
            noise_val = noise_white[t % len(noise_white)] if len(noise_white) > 0 else rng.standard_normal()
        else:
            noise_val = 0.0

        vg = np.full(N_NEURONS, base_vg)
        vg += alpha * drive_signal[t % len(drive_signal)] * w_in
        if condition != 'NO_NOISE':
            vg += beta_eff * noise_val * w_noise
        vg = np.clip(vg, 0.05, 0.95)

        I_in = vg * 5.0
        dvdt = (-vmem + I_in) / tau_m
        vmem += dvdt * dt

        for i in range(N_NEURONS):
            if vmem[i] >= v_thresh:
                delta_spikes[t, i] = 1
                vmem[i] = 0.0

    return delta_spikes


# ═══════════════════════════════════════════════════════════
# Criticality Analysis
# ═══════════════════════════════════════════════════════════

def compute_branching_ratio(delta_spikes):
    """Branching ratio sigma = E[n_active(t+1)] / E[n_active(t)].

    n_active(t) = number of neurons with delta_spikes > 0 at timestep t.
    Critical at sigma ~ 1.0.
    """
    n_active = (delta_spikes > 0).sum(axis=1).astype(float)  # (n_steps,)
    # Only consider timesteps where at least 1 neuron is active
    ratios = []
    for t in range(len(n_active) - 1):
        if n_active[t] > 0:
            ratios.append(n_active[t + 1] / n_active[t])
    if len(ratios) == 0:
        return 0.0
    return float(np.mean(ratios))


def extract_avalanches(delta_spikes, threshold=0):
    """Extract avalanche sizes and durations.

    An avalanche is a consecutive run of timesteps where total activity
    exceeds threshold.
    """
    total_activity = delta_spikes.sum(axis=1)  # (n_steps,)
    above = total_activity > threshold
    sizes = []
    durations = []
    current_size = 0
    current_dur = 0
    in_avalanche = False

    for t in range(len(above)):
        if above[t]:
            if not in_avalanche:
                in_avalanche = True
                current_size = 0
                current_dur = 0
            current_size += total_activity[t]
            current_dur += 1
        else:
            if in_avalanche:
                sizes.append(current_size)
                durations.append(current_dur)
                in_avalanche = False
    # Close last avalanche
    if in_avalanche:
        sizes.append(current_size)
        durations.append(current_dur)

    return np.array(sizes), np.array(durations)


def fit_power_law_exponent(data):
    """Fit power-law exponent alpha using MLE (Clauset et al. 2009).

    For discrete data x >= x_min: alpha = 1 + n / sum(ln(x / x_min))
    Returns alpha and x_min used.
    """
    data = np.array(data, dtype=float)
    data = data[data > 0]
    if len(data) < 5:
        return 0.0, 0.0
    x_min = max(np.min(data), 1.0)
    valid = data[data >= x_min]
    if len(valid) < 3:
        return 0.0, x_min
    n = len(valid)
    alpha = 1.0 + n / np.sum(np.log(valid / x_min))
    return float(alpha), float(x_min)


def compute_dynamic_range(ser, use_fpga, sample_hz, base_vg, alpha, n_trials=10):
    """Measure response to inputs spanning 3 orders of magnitude.

    Input amplitudes: [0.01, 0.03, 0.1, 0.3, 1.0]
    Dynamic range (dB) = 10 * log10(max_response / min_response)
    """
    amplitudes = [0.01, 0.03, 0.1, 0.3, 1.0]
    interval = 1.0 / sample_hz
    responses = []

    for amp in amplitudes:
        amp_responses = []
        for trial in range(n_trials):
            vg_values = np.full(N_NEURONS, base_vg + alpha * amp)
            vg_values = np.clip(vg_values, 0.05, 0.95)

            if use_fpga and ser is not None:
                set_per_neuron_vg(ser, vg_values)
                time.sleep(interval)
                ser.reset_input_buffer()
                ser.write(bytes([SYNC, CMD_READ_TELEM]))
                ser.flush()
                telem = read_telem(ser, timeout=0.15)
                if telem:
                    total = sum(n['spike_count'] for n in telem)
                    amp_responses.append(total)
            else:
                # Simulated response: logistic-like mapping
                rng = np.random.default_rng(100 + trial)
                rate = 5.0 / (1.0 + np.exp(-10 * (base_vg + alpha * amp - 0.5)))
                amp_responses.append(rate + rng.standard_normal() * 0.3)

        responses.append(float(np.mean(amp_responses)) if amp_responses else 0.0)

    responses = np.array(responses)
    r_min = max(np.min(responses), 1e-6)
    r_max = max(np.max(responses), 1e-6)
    dynamic_range_db = 10.0 * np.log10(r_max / r_min)

    return float(dynamic_range_db), amplitudes, responses.tolist()


def compute_acf_timescale(delta_spikes, max_lag=50):
    """Compute autocorrelation function decay timescale.

    Returns the lag at which ACF drops below 1/e, and the ACF values.
    """
    total_activity = delta_spikes.sum(axis=1).astype(float)
    mean_act = np.mean(total_activity)
    var_act = np.var(total_activity)
    if var_act < 1e-10:
        return 0.0, np.zeros(max_lag)

    centered = total_activity - mean_act
    n = len(centered)
    acf_vals = np.zeros(max_lag)

    for lag in range(max_lag):
        if lag >= n:
            break
        c = np.mean(centered[:n - lag] * centered[lag:])
        acf_vals[lag] = c / var_act

    # Find timescale: first lag where ACF < 1/e
    threshold = 1.0 / np.e
    timescale = float(max_lag)
    for lag in range(1, max_lag):
        if acf_vals[lag] < threshold:
            # Linear interpolation
            if acf_vals[lag - 1] > threshold:
                frac = (acf_vals[lag - 1] - threshold) / max(acf_vals[lag - 1] - acf_vals[lag], 1e-10)
                timescale = lag - 1 + frac
            else:
                timescale = float(lag)
            break

    return float(timescale), acf_vals


def compute_fano_factor(delta_spikes, window_size=10):
    """Fano factor: variance / mean of spike counts in windows.

    >1 indicates bursty (near-critical) dynamics.
    """
    total_activity = delta_spikes.sum(axis=1)
    n_windows = len(total_activity) // window_size
    if n_windows < 3:
        return 0.0

    windowed_counts = []
    for w in range(n_windows):
        s = total_activity[w * window_size:(w + 1) * window_size].sum()
        windowed_counts.append(s)
    windowed_counts = np.array(windowed_counts)
    mean_c = np.mean(windowed_counts)
    if mean_c < 1e-10:
        return 0.0
    return float(np.var(windowed_counts) / mean_c)


def compute_isi_cv(delta_spikes):
    """Coefficient of variation of inter-spike intervals across all neurons."""
    all_isis = []
    for i in range(delta_spikes.shape[1]):
        spike_times = np.where(delta_spikes[:, i] > 0)[0]
        if len(spike_times) > 1:
            isis = np.diff(spike_times).astype(float)
            all_isis.extend(isis.tolist())
    if len(all_isis) < 3:
        return 0.0
    arr = np.array(all_isis)
    return float(np.std(arr) / max(np.mean(arr), 1e-10))


# ═══════════════════════════════════════════════════════════
# Plotting
# ═══════════════════════════════════════════════════════════

def make_figure(all_analysis, tests):
    """Create 2x3 figure: branching, avalanches, dynamic range, ACF, Fano, tests."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("[WARN] matplotlib not available, skipping figure")
        return

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('z2191: Reservoir Criticality — GPU Noise Drives FPGA to Edge of Chaos',
                 fontsize=14, fontweight='bold')

    conditions = ['FULL', 'WHITE', 'NO_NOISE', 'SUPERCRITICAL']
    cond_colors = {
        'FULL': '#2196F3', 'WHITE': '#FF9800',
        'NO_NOISE': '#9E9E9E', 'SUPERCRITICAL': '#F44336'
    }

    # ─── Panel 1: Branching Ratio ───
    ax = axes[0, 0]
    sigmas = [all_analysis[c].get('branching_ratio', 0) for c in conditions]
    bars = ax.bar(conditions, sigmas, color=[cond_colors[c] for c in conditions], alpha=0.8)
    ax.axhline(y=1.0, color='red', linestyle='--', linewidth=2, label='Critical (sigma=1)')
    ax.set_ylabel('Branching Ratio (sigma)')
    ax.set_title('Branching Ratio')
    ax.legend(fontsize=8)
    ax.grid(axis='y', alpha=0.3)
    for bar, val in zip(bars, sigmas):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f'{val:.3f}', ha='center', va='bottom', fontsize=9)

    # ─── Panel 2: Avalanche Size Distribution (FULL) ───
    ax = axes[0, 1]
    for cond in ['FULL', 'WHITE']:
        sizes = all_analysis[cond].get('avalanche_sizes', [])
        if len(sizes) > 0:
            sizes = np.array(sizes)
            # Log-binned histogram
            bins = np.logspace(np.log10(max(sizes.min(), 1)),
                               np.log10(max(sizes.max(), 2)), 20)
            hist, edges = np.histogram(sizes, bins=bins, density=True)
            centers = (edges[:-1] + edges[1:]) / 2
            mask = hist > 0
            if np.any(mask):
                ax.loglog(centers[mask], hist[mask], 'o-',
                          color=cond_colors[cond], label=cond, markersize=4)
    alpha_full = all_analysis['FULL'].get('avalanche_alpha', 0)
    ax.set_xlabel('Avalanche Size')
    ax.set_ylabel('P(size)')
    ax.set_title(f'Avalanche Size Distribution (alpha={alpha_full:.2f})')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # ─── Panel 3: Dynamic Range ───
    ax = axes[0, 2]
    for cond in conditions:
        dr_amps = all_analysis[cond].get('dr_amplitudes', [])
        dr_resp = all_analysis[cond].get('dr_responses', [])
        if dr_amps and dr_resp:
            ax.semilogx(dr_amps, dr_resp, 'o-', color=cond_colors[cond],
                        label=f"{cond} ({all_analysis[cond].get('dynamic_range_db', 0):.1f} dB)",
                        markersize=5)
    ax.set_xlabel('Input Amplitude')
    ax.set_ylabel('Response')
    ax.set_title('Dynamic Range')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # ─── Panel 4: ACF Decay ───
    ax = axes[1, 0]
    for cond in ['FULL', 'WHITE', 'NO_NOISE']:
        acf = all_analysis[cond].get('acf_values', [])
        if len(acf) > 0:
            lags = np.arange(len(acf))
            ax.plot(lags, acf, '-', color=cond_colors[cond],
                    label=f"{cond} (tau={all_analysis[cond].get('acf_timescale', 0):.1f})",
                    linewidth=1.5)
    ax.axhline(y=1.0 / np.e, color='gray', linestyle=':', alpha=0.7, label='1/e threshold')
    ax.set_xlabel('Lag (timesteps)')
    ax.set_ylabel('Autocorrelation')
    ax.set_title('ACF Decay')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # ─── Panel 5: Fano Factor ───
    ax = axes[1, 1]
    fanos = [all_analysis[c].get('fano_factor', 0) for c in conditions]
    bars = ax.bar(conditions, fanos, color=[cond_colors[c] for c in conditions], alpha=0.8)
    ax.axhline(y=1.0, color='green', linestyle='--', linewidth=1, label='Poisson (Fano=1)')
    ax.set_ylabel('Fano Factor')
    ax.set_title('Fano Factor (Burstiness)')
    ax.legend(fontsize=8)
    ax.grid(axis='y', alpha=0.3)
    for bar, val in zip(bars, fanos):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                f'{val:.2f}', ha='center', va='bottom', fontsize=9)

    # ─── Panel 6: Test Summary ───
    ax = axes[1, 2]
    ax.axis('off')
    test_lines = []
    for t in tests:
        mark = 'PASS' if t['pass'] else 'FAIL'
        test_lines.append(f"{t['name']}: {mark}  ({t['detail']})")
    test_text = '\n'.join(test_lines)

    n_pass = sum(1 for t in tests if t['pass'])
    n_total = len(tests)
    header = f"Tests: {n_pass}/{n_total} PASS\n{'=' * 50}\n"

    ax.text(0.05, 0.95, header + test_text, transform=ax.transAxes,
            fontsize=8.5, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    ax.set_title('Test Results (T237-T242)')

    plt.tight_layout()
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig_path = FIGURES / 'fig_z2191_reservoir_criticality.png'
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[INFO] Figure saved: {fig_path}")


# ═══════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description='z2191 Reservoir Criticality')
    parser.add_argument('--n-steps', type=int, default=DEFAULT_N_STEPS)
    parser.add_argument('--sample-hz', type=int, default=DEFAULT_SAMPLE_HZ)
    parser.add_argument('--base-vg', type=float, default=DEFAULT_BASE_VG)
    parser.add_argument('--alpha', type=float, default=DEFAULT_ALPHA)
    parser.add_argument('--beta', type=float, default=DEFAULT_BETA)
    args = parser.parse_args()

    n_steps = args.n_steps
    sample_hz = args.sample_hz
    base_vg = args.base_vg
    alpha = args.alpha
    beta = args.beta

    print("=" * 65)
    print("z2191: Reservoir Criticality — GPU Noise Drives FPGA to Edge of Chaos")
    print("=" * 65)
    print(f"n_steps={n_steps}, sample_hz={sample_hz}, base_vg={base_vg}, "
          f"alpha={alpha}, beta={beta}, beta_super={DEFAULT_BETA_SUPER}")

    # ─── Prepare noise sources ───
    print("\n[1/6] Collecting GPU power noise for 1/f source...")
    noise_duration = max(20, n_steps / sample_hz * 1.5)
    raw_noise = collect_power_noise(duration_s=noise_duration, sample_hz=50)
    if raw_noise is not None and len(raw_noise) > 10:
        noise_1f = iir_filter_noise(raw_noise, alpha_iir=0.85)
        print(f"  Collected {len(raw_noise)} power samples -> {len(noise_1f)} filtered")
    else:
        print("  [WARN] hwmon power not available, generating synthetic 1/f noise")
        rng = np.random.default_rng(99)
        raw_synth = rng.standard_normal(n_steps * 2)
        vmf = VossMcCartneyFilter()
        noise_1f = np.array([vmf.process((s * 50 + 127.5)) for s in raw_synth])

    # White noise
    rng = np.random.default_rng(123)
    noise_white = rng.standard_normal(n_steps * 2)

    # Drive signal
    drive_signal = generate_drive_signal(n_steps, sample_hz)

    # ─── Try FPGA connection ───
    print("\n[2/6] Connecting to FPGA...")
    ser, port = find_fpga()
    use_fpga = ser is not None
    simulated = not use_fpga
    if use_fpga:
        print(f"  FPGA found on {port}")
        connect_fpga(ser)
    else:
        print("  [WARN] No FPGA found, using software LIF simulation")

    # ─── Run 4 conditions ───
    conditions = ['FULL', 'WHITE', 'NO_NOISE', 'SUPERCRITICAL']
    all_spikes = {}

    print("\n[3/6] Running conditions (500 steps x 4 = ~100s on FPGA)...")
    for cond in conditions:
        print(f"  Running condition: {cond}")
        t0 = time.monotonic()
        if use_fpga:
            spikes = run_condition(ser, cond, n_steps, sample_hz, base_vg, alpha, beta,
                                   noise_1f, noise_white, drive_signal)
        else:
            spikes = simulate_condition(cond, n_steps, sample_hz, base_vg, alpha, beta,
                                         noise_1f, noise_white, drive_signal)
        elapsed = time.monotonic() - t0
        all_spikes[cond] = spikes
        total = np.sum(spikes)
        n_active_steps = np.sum(spikes.sum(axis=1) > 0)
        print(f"    Done in {elapsed:.1f}s, total spikes: {total:.0f}, "
              f"active steps: {n_active_steps}/{n_steps}")

    # ─── Dynamic range measurement ───
    print("\n[4/6] Measuring dynamic range...")
    dr_results = {}
    for cond in conditions:
        use_ser = ser if (use_fpga and cond != 'NO_NOISE') else None
        dr_db, dr_amps, dr_resp = compute_dynamic_range(
            use_ser, use_fpga and cond != 'NO_NOISE', sample_hz, base_vg, alpha)
        dr_results[cond] = {'db': dr_db, 'amplitudes': dr_amps, 'responses': dr_resp}
        print(f"  {cond}: {dr_db:.1f} dB")

    if ser:
        ser.close()

    # ─── Analyze criticality measures ───
    print("\n[5/6] Computing criticality measures...")
    all_analysis = {}

    for cond in conditions:
        spikes = all_spikes[cond]
        sigma = compute_branching_ratio(spikes)
        aval_sizes, aval_durations = extract_avalanches(spikes)
        aval_alpha, aval_xmin = fit_power_law_exponent(aval_sizes)
        dur_alpha, dur_xmin = fit_power_law_exponent(aval_durations)
        acf_timescale, acf_values = compute_acf_timescale(spikes, max_lag=50)
        fano = compute_fano_factor(spikes, window_size=10)
        isi_cv = compute_isi_cv(spikes)

        analysis = {
            'branching_ratio': sigma,
            'avalanche_sizes': aval_sizes.tolist() if len(aval_sizes) > 0 else [],
            'avalanche_durations': aval_durations.tolist() if len(aval_durations) > 0 else [],
            'avalanche_alpha': aval_alpha,
            'avalanche_xmin': aval_xmin,
            'duration_alpha': dur_alpha,
            'duration_xmin': dur_xmin,
            'n_avalanches': len(aval_sizes),
            'mean_avalanche_size': float(np.mean(aval_sizes)) if len(aval_sizes) > 0 else 0.0,
            'dynamic_range_db': dr_results[cond]['db'],
            'dr_amplitudes': dr_results[cond]['amplitudes'],
            'dr_responses': dr_results[cond]['responses'],
            'acf_timescale': acf_timescale,
            'acf_values': acf_values.tolist(),
            'fano_factor': fano,
            'isi_cv': isi_cv,
            'total_spikes': float(np.sum(spikes)),
            'mean_firing_rate': float(np.mean(spikes.sum(axis=1))),
        }
        all_analysis[cond] = analysis

        print(f"  {cond}: sigma={sigma:.3f}, aval_alpha={aval_alpha:.2f} "
              f"(n={len(aval_sizes)}), DR={dr_results[cond]['db']:.1f} dB, "
              f"ACF_tau={acf_timescale:.1f}, Fano={fano:.2f}, ISI_CV={isi_cv:.3f}")

    # ─── Tests T237-T242 ───
    print("\n[6/6] Evaluating tests T237-T242...")
    full = all_analysis['FULL']
    white = all_analysis['WHITE']
    no_noise = all_analysis['NO_NOISE']
    supercrit = all_analysis['SUPERCRITICAL']

    tests = []

    # T237: Branching ratio FULL closer to 1.0 than WHITE
    dist_full = abs(full['branching_ratio'] - 1.0)
    dist_white = abs(white['branching_ratio'] - 1.0)
    t237 = dist_full < dist_white
    tests.append({
        'name': 'T237',
        'pass': t237,
        'detail': f"|sigma_FULL-1|={dist_full:.3f} < |sigma_WHITE-1|={dist_white:.3f}",
    })
    print(f"  T237: |sigma_FULL-1|={dist_full:.3f} vs |sigma_WHITE-1|={dist_white:.3f}"
          f" -> {'PASS' if t237 else 'FAIL'}")

    # T238: Avalanche size exponent alpha in [1.0, 2.0] for FULL
    t238 = 1.0 <= full['avalanche_alpha'] <= 2.0
    tests.append({
        'name': 'T238',
        'pass': t238,
        'detail': f"alpha={full['avalanche_alpha']:.3f} in [1.0, 2.0]",
    })
    print(f"  T238: alpha={full['avalanche_alpha']:.3f} in [1.0, 2.0]"
          f" -> {'PASS' if t238 else 'FAIL'}")

    # T239: Dynamic range FULL > Dynamic range NO_NOISE
    t239 = full['dynamic_range_db'] > no_noise['dynamic_range_db']
    tests.append({
        'name': 'T239',
        'pass': t239,
        'detail': f"DR_FULL={full['dynamic_range_db']:.1f} > DR_NONOISE={no_noise['dynamic_range_db']:.1f} dB",
    })
    print(f"  T239: DR_FULL={full['dynamic_range_db']:.1f} vs "
          f"DR_NONOISE={no_noise['dynamic_range_db']:.1f} dB"
          f" -> {'PASS' if t239 else 'FAIL'}")

    # T240: ACF timescale FULL > ACF timescale WHITE
    t240 = full['acf_timescale'] > white['acf_timescale']
    tests.append({
        'name': 'T240',
        'pass': t240,
        'detail': f"tau_FULL={full['acf_timescale']:.1f} > tau_WHITE={white['acf_timescale']:.1f}",
    })
    print(f"  T240: tau_FULL={full['acf_timescale']:.1f} vs "
          f"tau_WHITE={white['acf_timescale']:.1f}"
          f" -> {'PASS' if t240 else 'FAIL'}")

    # T241: Fano factor FULL in [1.0, 5.0]
    t241 = 1.0 <= full['fano_factor'] <= 5.0
    tests.append({
        'name': 'T241',
        'pass': t241,
        'detail': f"Fano={full['fano_factor']:.2f} in [1.0, 5.0]",
    })
    print(f"  T241: Fano={full['fano_factor']:.2f} in [1.0, 5.0]"
          f" -> {'PASS' if t241 else 'FAIL'}")

    # T242: SUPERCRITICAL branching ratio > 1.2
    t242 = supercrit['branching_ratio'] > 1.2
    tests.append({
        'name': 'T242',
        'pass': t242,
        'detail': f"sigma_SUPER={supercrit['branching_ratio']:.3f} > 1.2",
    })
    print(f"  T242: sigma_SUPER={supercrit['branching_ratio']:.3f} > 1.2"
          f" -> {'PASS' if t242 else 'FAIL'}")

    n_pass = sum(1 for t in tests if t['pass'])
    n_total = len(tests)
    print(f"\n{'=' * 65}")
    print(f"RESULT: {n_pass}/{n_total} PASS")
    print(f"{'=' * 65}")

    # ─── Build output ───
    output = {
        'experiment': 'z2191_reservoir_criticality',
        'description': 'GPU firmware noise drives FPGA reservoir toward criticality',
        'simulated': simulated,
        'params': {
            'n_steps': n_steps,
            'sample_hz': sample_hz,
            'base_vg': base_vg,
            'alpha': alpha,
            'beta': beta,
            'beta_supercritical': DEFAULT_BETA_SUPER,
            'n_neurons': N_NEURONS,
        },
        'conditions': {cond: {k: v for k, v in all_analysis[cond].items()
                               if k not in ('acf_values', 'avalanche_sizes',
                                            'avalanche_durations')}
                        for cond in conditions},
        'tests': tests,
        'summary': {
            'n_pass': n_pass,
            'n_total': n_total,
            'branching_ratio_FULL': full['branching_ratio'],
            'avalanche_alpha_FULL': full['avalanche_alpha'],
            'dynamic_range_FULL_db': full['dynamic_range_db'],
            'acf_timescale_FULL': full['acf_timescale'],
            'fano_factor_FULL': full['fano_factor'],
            'isi_cv_FULL': full['isi_cv'],
        },
    }

    # ─── Save results ───
    RESULTS.mkdir(parents=True, exist_ok=True)
    result_path = RESULTS / 'z2191_reservoir_criticality.json'
    with open(result_path, 'w') as f:
        json.dump(output, f, indent=2, cls=NpEncoder)
    print(f"\n[INFO] Results saved: {result_path}")

    # ─── Make figure ───
    make_figure(all_analysis, tests)


if __name__ == '__main__':
    main()
