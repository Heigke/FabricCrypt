#!/usr/bin/env python3
"""z2184_cross_timescale_coupling.py — Cross-Timescale Coupling in GPU-Noise-Driven FPGA Neurons

Tests whether GPU firmware noise creates CROSS-TIMESCALE COUPLING in FPGA neurons —
a key property of biological neural systems where fast and slow dynamics interact
(theta-gamma coupling, nested oscillations).

4 conditions:
  FULL      — GPU power 1/f noise (hwmon) -> FPGA neurons (standard bridge)
  WHITE     — White noise -> FPGA neurons (control)
  NO_NOISE  — Deterministic FPGA (no noise injection)
  PINK_IIR  — Software 1/f noise via IIR filter (synthetic control)

3 analyses:
  1. Phase-Amplitude Coupling (PAC): MI between slow (0.5-2 Hz) phase and fast (4-8 Hz) amplitude
  2. Detrended Fluctuation Analysis (DFA): Long-range temporal correlations across scales
  3. Multi-scale entropy (MSE): Complexity across timescales (coarse-graining)

Tests T195-T200:
  T195: PAC MI(FULL) > PAC MI(WHITE)
  T196: PAC MI(FULL) > PAC MI(NO_NOISE)
  T197: DFA alpha(FULL) in [0.6, 1.0]
  T198: DFA alpha(FULL) > DFA alpha(WHITE)
  T199: MSE(FULL) > MSE(WHITE) at coarse scales (tau >= 4)
  T200: MSE(FULL) complexity index > MSE(NO_NOISE)

Hardware: AMD gfx1151 GPU + Arty A7 FPGA on /dev/ttyUSB*
"""

import os, sys, json, time, struct, argparse
import numpy as np
from pathlib import Path

os.environ.setdefault('HSA_OVERRIDE_GFX_VERSION', '11.0.0')

BASE = Path(__file__).resolve().parent.parent
RESULTS = BASE / 'results'
FIGURES = BASE / 'figures'

# ─── FPGA Protocol ───
SYNC = 0x55
CMD_SET_VG = 0x01
CMD_READ_TELEM = 0x02
CMD_SET_KILL = 0x03

HWMON_POWER = "/sys/class/hwmon/hwmon7/power1_average"

# ─── Parameters ───
BASE_VG = 0.55
ALPHA = 0.15
BETA = 0.10
N_NEURONS = 8
SAMPLE_HZ = 20


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
    """10-octave Voss-McCartney 1/f noise generator.

    Takes white noise input and produces 1/f-like output by summing
    10 random generators that update at geometrically spaced intervals.
    """

    def __init__(self, n_octaves=10):
        self.n_octaves = n_octaves
        self.values = np.zeros(n_octaves)
        self.counters = np.zeros(n_octaves, dtype=int)
        self.step = 0

    def process(self, white_sample):
        """Process one white noise sample, return 1/f filtered value."""
        x = (white_sample / 127.5) - 1.0
        for k in range(self.n_octaves):
            period = 1 << k
            if self.step % period == 0:
                self.values[k] = x
        self.step += 1
        total = np.sum(self.values) / self.n_octaves
        return np.clip(total, -1.0, 1.0)


# ═══════════════════════════════════════════════════════════
# FPGA Reservoir Core
# ═══════════════════════════════════════════════════════════

def run_fpga_trial(ser, noise_samples, base_vg, alpha, beta,
                   n_steps, sample_hz, live_noise=False):
    """Drive FPGA neurons with noise and collect spike rate time series.

    Returns: (n_steps,) array of total spike rate (sum across 8 neurons per timestep).
    """
    interval = 1.0 / sample_hz
    spike_rates = np.zeros(n_steps)
    prev_counts = None
    power_mean = 11.0
    rng = np.random.default_rng()

    for t in range(n_steps):
        # Determine noise value
        if live_noise:
            p = read_hwmon_power()
            noise_val = (p - power_mean) / 2.0 if p else 0.0
        elif beta > 0 and len(noise_samples) > 0:
            noise_val = noise_samples[t % len(noise_samples)]
        else:
            noise_val = 0.0

        # Compute per-neuron Vg
        vg_values = np.full(N_NEURONS, base_vg)
        if beta > 0:
            # Heterogeneous noise injection across neurons
            w_noise = np.array([1.0, -0.8, 0.6, -0.4, 0.9, -0.7, 0.5, -0.3])
            vg_values += beta * noise_val * w_noise
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
                total_delta = 0
                for i in range(N_NEURONS):
                    delta = (counts[i] - prev_counts[i]) & 0xFFFF
                    if delta > 30000:
                        delta = 0
                    total_delta += delta
                spike_rates[t] = total_delta
            prev_counts = counts[:]

        time.sleep(max(0, interval * 0.5 - 0.01))

    return spike_rates


def simulate_lif_trial(noise_samples, base_vg, alpha, beta,
                       n_steps, sample_hz):
    """Software LIF simulation fallback when FPGA is not connected."""
    dt = 1.0 / sample_hz
    v_thresh = 1.0
    tau_m = 0.02
    vmem = np.zeros(N_NEURONS)
    spike_rates = np.zeros(n_steps)
    w_noise = np.array([1.0, -0.8, 0.6, -0.4, 0.9, -0.7, 0.5, -0.3])

    for t in range(n_steps):
        vg = np.full(N_NEURONS, base_vg)
        if beta > 0 and len(noise_samples) > 0:
            noise_val = noise_samples[t % len(noise_samples)]
            vg += beta * noise_val * w_noise
        vg = np.clip(vg, 0.05, 0.95)

        I_in = vg * 5.0
        dvdt = (-vmem + I_in) / tau_m
        vmem += dvdt * dt

        total_spikes = 0
        for i in range(N_NEURONS):
            if vmem[i] >= v_thresh:
                total_spikes += 1
                vmem[i] = 0.0
        spike_rates[t] = total_spikes

    return spike_rates


# ═══════════════════════════════════════════════════════════
# Analysis Functions
# ═══════════════════════════════════════════════════════════

def bandpass_filter(signal, fs, low, high, order=3):
    """Simple FIR bandpass filter using windowed sinc."""
    nyq = fs / 2.0
    if high >= nyq:
        high = nyq * 0.95
    if low >= high:
        low = high * 0.5
    # Design FIR filter via frequency-domain windowing
    n_taps = min(len(signal) - 1, max(31, int(4 * fs / low)))
    if n_taps % 2 == 0:
        n_taps += 1
    t = np.arange(n_taps) - (n_taps - 1) / 2
    t[t == 0] = 1e-10
    # Bandpass = highpass(low) via lowpass(high) - lowpass(low)
    h_high = np.sin(2 * np.pi * high / fs * t) / (np.pi * t)
    h_low = np.sin(2 * np.pi * low / fs * t) / (np.pi * t)
    h = h_high - h_low
    # Fix center tap
    center = (n_taps - 1) // 2
    h[center] = 2.0 * (high - low) / fs
    # Apply Hamming window
    window = np.hamming(n_taps)
    h *= window
    # Normalize
    h /= np.sum(np.abs(h)) + 1e-10
    # Apply via convolution (zero-phase: forward + reverse)
    padded = np.pad(signal, n_taps, mode='reflect')
    filtered = np.convolve(padded, h, mode='same')
    filtered = filtered[n_taps:n_taps + len(signal)]
    return filtered


def hilbert_transform(signal):
    """Compute analytic signal via FFT-based Hilbert transform."""
    N = len(signal)
    X = np.fft.fft(signal)
    h = np.zeros(N)
    if N % 2 == 0:
        h[0] = 1
        h[N // 2] = 1
        h[1:N // 2] = 2
    else:
        h[0] = 1
        h[1:(N + 1) // 2] = 2
    analytic = np.fft.ifft(X * h)
    return analytic


def compute_pac_mi(spike_rates, fs, slow_band=(0.5, 2.0), fast_band=(4.0, 8.0), n_bins=18):
    """Compute Phase-Amplitude Coupling Modulation Index.

    MI = KL divergence of amplitude distribution from uniform, normalized.
    """
    if len(spike_rates) < 50:
        return 0.0, np.zeros(n_bins), np.zeros(n_bins)

    # Filter into slow and fast bands
    slow = bandpass_filter(spike_rates, fs, slow_band[0], slow_band[1])
    fast = bandpass_filter(spike_rates, fs, fast_band[0], fast_band[1])

    # Get phase of slow oscillation
    analytic_slow = hilbert_transform(slow)
    phase_slow = np.angle(analytic_slow)

    # Get amplitude envelope of fast oscillation
    analytic_fast = hilbert_transform(fast)
    amp_fast = np.abs(analytic_fast)

    # Bin phase into n_bins
    bin_edges = np.linspace(-np.pi, np.pi, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    mean_amp = np.zeros(n_bins)

    for b in range(n_bins):
        mask = (phase_slow >= bin_edges[b]) & (phase_slow < bin_edges[b + 1])
        if np.sum(mask) > 0:
            mean_amp[b] = np.mean(amp_fast[mask])

    # Normalize to probability distribution
    total = np.sum(mean_amp)
    if total < 1e-10:
        return 0.0, bin_centers, mean_amp

    p = mean_amp / total
    # KL divergence from uniform
    uniform = np.ones(n_bins) / n_bins
    # Avoid log(0)
    p_safe = np.clip(p, 1e-10, None)
    kl = np.sum(p_safe * np.log(p_safe / uniform))
    mi = kl / np.log(n_bins)  # Normalize to [0, 1]

    return float(mi), bin_centers, mean_amp


def compute_dfa(signal, min_box=4, max_box=None, n_scales=15):
    """Detrended Fluctuation Analysis.

    Returns: alpha (scaling exponent), scales, fluctuations.
    """
    N = len(signal)
    if N < 20:
        return 0.5, np.array([]), np.array([])

    # Integrate signal (cumulative sum of mean-subtracted)
    y = np.cumsum(signal - np.mean(signal))

    if max_box is None:
        max_box = N // 4

    # Generate log-spaced box sizes
    scales = np.unique(np.logspace(
        np.log10(max(min_box, 4)),
        np.log10(max(max_box, min_box + 1)),
        n_scales
    ).astype(int))
    scales = scales[scales >= 4]

    fluctuations = []
    valid_scales = []

    for n in scales:
        n_segments = N // n
        if n_segments < 2:
            continue

        rms_list = []
        for seg in range(n_segments):
            start = seg * n
            end = start + n
            segment = y[start:end]
            # Linear detrend
            x_axis = np.arange(n)
            coeffs = np.polyfit(x_axis, segment, 1)
            trend = np.polyval(coeffs, x_axis)
            residual = segment - trend
            rms_list.append(np.sqrt(np.mean(residual ** 2)))

        if rms_list:
            fluctuations.append(np.mean(rms_list))
            valid_scales.append(n)

    if len(valid_scales) < 3:
        return 0.5, np.array(valid_scales), np.array(fluctuations)

    scales_arr = np.array(valid_scales, dtype=float)
    fluct_arr = np.array(fluctuations, dtype=float)

    # Fit log-log slope
    mask = fluct_arr > 0
    if np.sum(mask) < 3:
        return 0.5, scales_arr, fluct_arr

    log_s = np.log10(scales_arr[mask])
    log_f = np.log10(fluct_arr[mask])
    coeffs = np.polyfit(log_s, log_f, 1)
    alpha = coeffs[0]

    return float(alpha), scales_arr, fluct_arr


def sample_entropy(signal, m=2, r_frac=0.15):
    """Compute Sample Entropy for a time series.

    m: embedding dimension
    r_frac: tolerance as fraction of std
    """
    N = len(signal)
    if N < m + 2:
        return 0.0

    r = r_frac * np.std(signal)
    if r < 1e-10:
        return 0.0

    def count_matches(templates):
        n_templates = len(templates)
        count = 0
        for i in range(n_templates):
            for j in range(i + 1, n_templates):
                if np.max(np.abs(templates[i] - templates[j])) <= r:
                    count += 1
        return count

    # Build templates of length m
    templates_m = np.array([signal[i:i + m] for i in range(N - m)])
    # Build templates of length m+1
    templates_m1 = np.array([signal[i:i + m + 1] for i in range(N - m)])

    B = count_matches(templates_m)
    A = count_matches(templates_m1)

    if B == 0:
        return 0.0

    return -np.log(A / B) if A > 0 else float(np.log(B))


def compute_mse(signal, max_scale=10, m=2, r_frac=0.15):
    """Multi-Scale Entropy: SampleEntropy at multiple coarse-grain scales."""
    entropies = []
    scales = list(range(1, max_scale + 1))

    for tau in scales:
        # Coarse-grain: average non-overlapping windows of size tau
        n_coarse = len(signal) // tau
        if n_coarse < m + 10:
            entropies.append(0.0)
            continue
        coarse = np.array([
            np.mean(signal[i * tau:(i + 1) * tau])
            for i in range(n_coarse)
        ])
        se = sample_entropy(coarse, m=m, r_frac=r_frac)
        entropies.append(float(se))

    return np.array(scales), np.array(entropies)


# ═══════════════════════════════════════════════════════════
# Main Experiment
# ═══════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description='z2184: Cross-Timescale Coupling')
    parser.add_argument('--n-steps', type=int, default=500,
                        help='Timesteps per condition (default 500)')
    parser.add_argument('--sample-hz', type=int, default=20,
                        help='Sample rate in Hz (default 20)')
    parser.add_argument('--base-vg', type=float, default=BASE_VG,
                        help=f'Base Vg (default {BASE_VG})')
    parser.add_argument('--alpha', type=float, default=ALPHA,
                        help=f'Alpha gain (default {ALPHA})')
    parser.add_argument('--beta', type=float, default=BETA,
                        help=f'Beta noise gain (default {BETA})')
    parser.add_argument('--noise-collect-s', type=float, default=15.0,
                        help='Duration to collect power noise (s)')
    args = parser.parse_args()

    n_steps = args.n_steps
    sample_hz = args.sample_hz
    base_vg = args.base_vg
    alpha = args.alpha
    beta = args.beta

    print("=" * 65)
    print("z2184: Cross-Timescale Coupling in GPU-Noise-Driven FPGA Neurons")
    print("=" * 65)
    print(f"  Steps: {n_steps}  Sample Hz: {sample_hz}  Duration: {n_steps/sample_hz:.1f}s")
    print(f"  base_vg={base_vg}  alpha={alpha}  beta={beta}")

    rng = np.random.default_rng(42)

    results = {
        'experiment': 'z2184_cross_timescale_coupling',
        'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
        'params': {
            'base_vg': base_vg, 'alpha': alpha, 'beta': beta,
            'n_neurons': N_NEURONS, 'sample_hz': sample_hz,
            'n_steps': n_steps,
        },
        'simulated': False,
    }

    # ─── Step 1: Connect to FPGA ───
    print("\n[1/5] Connecting to FPGA...")
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

    # ─── Step 2: Collect GPU noise ───
    print("\n[2/5] Collecting GPU noise sources...")
    print("  Collecting power rail noise (1/f)...")
    power_noise = collect_power_noise(duration_s=args.noise_collect_s, sample_hz=50)
    if power_noise is not None and len(power_noise) > 10:
        power_mean = power_noise.mean()
        power_std = max(power_noise.std(), 1e-6)
        noise_1f_raw = (power_noise - power_mean) / power_std
        print(f"  Power rail: {power_mean:.2f} +/- {power_std:.3f} W, {len(noise_1f_raw)} samples")
    else:
        print("  Power rail unavailable, generating synthetic 1/f")
        n_synth = int(args.noise_collect_s * 50)
        noise_1f_raw = np.zeros(n_synth)
        n_octaves = 8
        octaves = np.zeros(n_octaves)
        for i in range(n_synth):
            for j in range(n_octaves):
                if i % (1 << j) == 0:
                    octaves[j] = rng.standard_normal()
            noise_1f_raw[i] = octaves.sum()
        noise_1f_raw = (noise_1f_raw - noise_1f_raw.mean()) / max(noise_1f_raw.std(), 1e-6)

    noise_1f = iir_filter_noise(noise_1f_raw, alpha_iir=0.85)
    noise_white = rng.standard_normal(len(noise_1f))
    noise_zero = np.zeros(1000)

    # Generate pink IIR noise from white via VossMcCartney
    print("  Generating PINK_IIR noise via Voss-McCartney...")
    vmf = VossMcCartneyFilter(n_octaves=10)
    white_bytes = rng.integers(0, 256, size=len(noise_1f))
    noise_pink_iir = np.array([vmf.process(wb) for wb in white_bytes])
    noise_pink_iir = (noise_pink_iir - noise_pink_iir.mean()) / max(noise_pink_iir.std(), 1e-6)
    print(f"  PINK_IIR noise: {len(noise_pink_iir)} samples, std={noise_pink_iir.std():.3f}")

    conditions = {
        'FULL':     {'noise': noise_1f,       'beta': beta, 'live_noise': True,
                     'label': 'GPU 1/f noise'},
        'WHITE':    {'noise': noise_white,    'beta': beta, 'live_noise': False,
                     'label': 'White noise'},
        'NO_NOISE': {'noise': noise_zero,     'beta': 0.0,  'live_noise': False,
                     'label': 'No noise'},
        'PINK_IIR': {'noise': noise_pink_iir, 'beta': beta, 'live_noise': False,
                     'label': 'Pink IIR noise'},
    }

    # ─── Step 3: Run FPGA trials for each condition ───
    print("\n[3/5] Running FPGA trials across conditions...")
    spike_data = {}

    for cond_name, cond in conditions.items():
        print(f"\n  === Condition: {cond_name} ({cond['label']}) ===")
        if fpga:
            spike_rates = run_fpga_trial(
                ser, cond['noise'], base_vg, alpha, cond['beta'],
                n_steps, sample_hz, live_noise=cond['live_noise']
            )
        else:
            spike_rates = simulate_lif_trial(
                cond['noise'], base_vg, alpha, cond['beta'],
                n_steps, sample_hz
            )
        total_spikes = spike_rates.sum()
        mean_rate = spike_rates.mean()
        print(f"    Total spikes: {total_spikes:.0f}, mean rate: {mean_rate:.2f}/step")
        spike_data[cond_name] = spike_rates

    # ─── Step 4: Analyses ───
    print("\n[4/5] Running cross-timescale analyses...")

    analysis_results = {}
    for cond_name, rates in spike_data.items():
        print(f"\n  === Analyzing: {cond_name} ===")

        # PAC
        mi, bin_centers, mean_amp = compute_pac_mi(rates, sample_hz,
                                                     slow_band=(0.5, 2.0),
                                                     fast_band=(4.0, 8.0),
                                                     n_bins=18)
        print(f"    PAC MI = {mi:.6f}")

        # DFA
        dfa_alpha, dfa_scales, dfa_fluct = compute_dfa(rates, min_box=4,
                                                         max_box=n_steps // 4,
                                                         n_scales=15)
        print(f"    DFA alpha = {dfa_alpha:.4f}")

        # MSE
        mse_scales, mse_entropies = compute_mse(rates, max_scale=10, m=2, r_frac=0.15)
        complexity_index = float(np.sum(mse_entropies))
        coarse_mse = float(np.mean(mse_entropies[3:])) if len(mse_entropies) > 3 else 0.0
        print(f"    MSE complexity index = {complexity_index:.4f}")
        print(f"    MSE coarse (tau>=4) mean = {coarse_mse:.4f}")

        analysis_results[cond_name] = {
            'pac_mi': mi,
            'pac_bin_centers': bin_centers.tolist() if isinstance(bin_centers, np.ndarray) else bin_centers,
            'pac_mean_amp': mean_amp.tolist() if isinstance(mean_amp, np.ndarray) else mean_amp,
            'dfa_alpha': dfa_alpha,
            'dfa_scales': dfa_scales.tolist() if isinstance(dfa_scales, np.ndarray) else [],
            'dfa_fluctuations': dfa_fluct.tolist() if isinstance(dfa_fluct, np.ndarray) else [],
            'mse_scales': mse_scales.tolist() if isinstance(mse_scales, np.ndarray) else [],
            'mse_entropies': mse_entropies.tolist() if isinstance(mse_entropies, np.ndarray) else [],
            'mse_complexity_index': complexity_index,
            'mse_coarse_mean': coarse_mse,
            'spike_rate_mean': float(rates.mean()),
            'spike_rate_std': float(rates.std()),
            'total_spikes': float(rates.sum()),
        }

    results['analysis'] = analysis_results
    results['spike_data'] = {k: v.tolist() for k, v in spike_data.items()}

    # ─── Step 5: Evaluate tests T195-T200 ───
    print("\n[5/5] Evaluating tests T195-T200...")
    full = analysis_results['FULL']
    white = analysis_results['WHITE']
    no_noise = analysis_results['NO_NOISE']

    tests = {}

    # T195: PAC MI(FULL) > PAC MI(WHITE)
    t195_pass = full['pac_mi'] > white['pac_mi']
    tests['T195'] = {
        'name': 'PAC MI(FULL) > PAC MI(WHITE)',
        'mi_full': full['pac_mi'],
        'mi_white': white['pac_mi'],
        'pass': t195_pass,
    }
    print(f"  T195: PAC MI FULL={full['pac_mi']:.6f} vs WHITE={white['pac_mi']:.6f}"
          f" -> {'PASS' if t195_pass else 'FAIL'}")

    # T196: PAC MI(FULL) > PAC MI(NO_NOISE)
    t196_pass = full['pac_mi'] > no_noise['pac_mi']
    tests['T196'] = {
        'name': 'PAC MI(FULL) > PAC MI(NO_NOISE)',
        'mi_full': full['pac_mi'],
        'mi_no_noise': no_noise['pac_mi'],
        'pass': t196_pass,
    }
    print(f"  T196: PAC MI FULL={full['pac_mi']:.6f} vs NO_NOISE={no_noise['pac_mi']:.6f}"
          f" -> {'PASS' if t196_pass else 'FAIL'}")

    # T197: DFA alpha(FULL) in [0.6, 1.0]
    t197_pass = 0.6 <= full['dfa_alpha'] <= 1.0
    tests['T197'] = {
        'name': 'DFA alpha(FULL) in [0.6, 1.0]',
        'dfa_alpha_full': full['dfa_alpha'],
        'range': [0.6, 1.0],
        'pass': t197_pass,
    }
    print(f"  T197: DFA alpha FULL={full['dfa_alpha']:.4f} in [0.6,1.0]"
          f" -> {'PASS' if t197_pass else 'FAIL'}")

    # T198: DFA alpha(FULL) > DFA alpha(WHITE)
    t198_pass = full['dfa_alpha'] > white['dfa_alpha']
    tests['T198'] = {
        'name': 'DFA alpha(FULL) > DFA alpha(WHITE)',
        'dfa_alpha_full': full['dfa_alpha'],
        'dfa_alpha_white': white['dfa_alpha'],
        'pass': t198_pass,
    }
    print(f"  T198: DFA alpha FULL={full['dfa_alpha']:.4f} vs WHITE={white['dfa_alpha']:.4f}"
          f" -> {'PASS' if t198_pass else 'FAIL'}")

    # T199: MSE(FULL) > MSE(WHITE) at coarse scales (tau >= 4)
    t199_pass = full['mse_coarse_mean'] > white['mse_coarse_mean']
    tests['T199'] = {
        'name': 'MSE(FULL) > MSE(WHITE) at coarse scales (tau>=4)',
        'mse_coarse_full': full['mse_coarse_mean'],
        'mse_coarse_white': white['mse_coarse_mean'],
        'pass': t199_pass,
    }
    print(f"  T199: MSE coarse FULL={full['mse_coarse_mean']:.4f} vs WHITE={white['mse_coarse_mean']:.4f}"
          f" -> {'PASS' if t199_pass else 'FAIL'}")

    # T200: MSE(FULL) complexity index > MSE(NO_NOISE)
    t200_pass = full['mse_complexity_index'] > no_noise['mse_complexity_index']
    tests['T200'] = {
        'name': 'MSE(FULL) complexity > MSE(NO_NOISE)',
        'ci_full': full['mse_complexity_index'],
        'ci_no_noise': no_noise['mse_complexity_index'],
        'pass': t200_pass,
    }
    print(f"  T200: MSE CI FULL={full['mse_complexity_index']:.4f} vs NO_NOISE="
          f"{no_noise['mse_complexity_index']:.4f} -> {'PASS' if t200_pass else 'FAIL'}")

    n_pass = sum(1 for t in tests.values() if t['pass'])
    print(f"\n  TOTAL: {n_pass}/6 PASS")
    results['tests'] = tests
    results['n_pass'] = n_pass
    results['n_total'] = 6

    # ─── Save results ───
    RESULTS.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS / 'z2184_cross_timescale_coupling.json'
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2, cls=NpEncoder)
    print(f"\n  Results saved: {out_path}")

    # ─── Generate figure ───
    print("\n  Generating figure...")
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('z2184: Cross-Timescale Coupling -- GPU-Noise-Driven FPGA Neurons',
                     fontsize=14, fontweight='bold')

        cond_names = ['FULL', 'WHITE', 'NO_NOISE', 'PINK_IIR']
        colors = {'FULL': '#e74c3c', 'WHITE': '#3498db',
                  'NO_NOISE': '#95a5a6', 'PINK_IIR': '#9b59b6'}
        labels = {'FULL': 'GPU 1/f', 'WHITE': 'White',
                  'NO_NOISE': 'No noise', 'PINK_IIR': 'Pink IIR'}

        # Panel 1: PAC polar plot
        ax = axes[0, 0]
        ax.set_title('Phase-Amplitude Coupling (PAC)')
        for cond in cond_names:
            ar = analysis_results[cond]
            bc = np.array(ar['pac_bin_centers'])
            ma = np.array(ar['pac_mean_amp'])
            if len(bc) > 0 and np.sum(ma) > 0:
                ma_norm = ma / (np.sum(ma) + 1e-10)
                ax.plot(bc, ma_norm, 'o-', color=colors[cond],
                        label=f"{labels[cond]} MI={ar['pac_mi']:.4f}", markersize=3)
        ax.set_xlabel('Slow phase (rad)')
        ax.set_ylabel('Normalized fast amplitude')
        ax.legend(fontsize=8)
        ax.axhline(y=1.0 / 18, color='gray', linestyle='--', alpha=0.3, label='uniform')

        # Panel 2: DFA log-log plot
        ax = axes[0, 1]
        ax.set_title('Detrended Fluctuation Analysis (DFA)')
        for cond in cond_names:
            ar = analysis_results[cond]
            scales = np.array(ar['dfa_scales'])
            fluct = np.array(ar['dfa_fluctuations'])
            if len(scales) > 0 and len(fluct) > 0:
                mask = fluct > 0
                if np.any(mask):
                    ax.loglog(scales[mask], fluct[mask], 'o-', color=colors[cond],
                              label=f"{labels[cond]} a={ar['dfa_alpha']:.3f}", markersize=4)
        ax.set_xlabel('Box size n')
        ax.set_ylabel('Fluctuation F(n)')
        ax.legend(fontsize=8)
        # Reference lines
        if len(analysis_results['FULL']['dfa_scales']) > 0:
            ref_scales = np.array(analysis_results['FULL']['dfa_scales'])
            ref_scales = ref_scales[ref_scales > 0]
            if len(ref_scales) > 1:
                ax.loglog(ref_scales, ref_scales ** 0.5 * 0.1, '--', color='gray',
                          alpha=0.3, label='a=0.5 (white)')
                ax.loglog(ref_scales, ref_scales ** 1.0 * 0.01, ':', color='gray',
                          alpha=0.3, label='a=1.0 (1/f)')

        # Panel 3: MSE curves
        ax = axes[1, 0]
        ax.set_title('Multi-Scale Entropy (MSE)')
        for cond in cond_names:
            ar = analysis_results[cond]
            scales = np.array(ar['mse_scales'])
            ent = np.array(ar['mse_entropies'])
            if len(scales) > 0:
                ax.plot(scales, ent, 'o-', color=colors[cond],
                        label=f"{labels[cond]} CI={ar['mse_complexity_index']:.2f}",
                        markersize=4)
        ax.axvline(x=4, color='gray', linestyle=':', alpha=0.3, label='tau=4 threshold')
        ax.set_xlabel('Scale factor tau')
        ax.set_ylabel('Sample Entropy')
        ax.legend(fontsize=8)

        # Panel 4: Summary bar chart
        ax = axes[1, 1]
        ax.set_title('Test Summary (T195-T200)')
        test_names = ['T195\nPAC>W', 'T196\nPAC>N', 'T197\nDFA\nrange',
                      'T198\nDFA>W', 'T199\nMSE>W', 'T200\nMSE>N']
        test_keys = ['T195', 'T196', 'T197', 'T198', 'T199', 'T200']
        pass_vals = [1 if tests[k]['pass'] else 0 for k in test_keys]
        bar_colors = ['#27ae60' if v else '#e74c3c' for v in pass_vals]
        bars = ax.bar(test_names, pass_vals, color=bar_colors)
        ax.set_ylim(-0.1, 1.3)
        ax.set_ylabel('PASS (1) / FAIL (0)')
        for bar, v, k in zip(bars, pass_vals, test_keys):
            label = 'PASS' if v else 'FAIL'
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                    label, ha='center', va='bottom', fontsize=9, fontweight='bold')
        ax.text(0.5, 0.95, f'{n_pass}/6 PASS', transform=ax.transAxes,
                ha='center', va='top', fontsize=14, fontweight='bold',
                color='#27ae60' if n_pass >= 4 else '#e74c3c')

        plt.tight_layout()
        FIGURES.mkdir(parents=True, exist_ok=True)
        fig_path = FIGURES / 'z2184_cross_timescale_coupling.png'
        plt.savefig(fig_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Figure saved: {fig_path}")
    except ImportError:
        print("  matplotlib not available, skipping figure")

    # ─── Cleanup ───
    if fpga and ser:
        ser.write(bytes([SYNC, CMD_SET_KILL, 0x00]))
        ser.flush()
        ser.close()
        print("  FPGA connection closed")

    print(f"\n{'=' * 65}")
    print(f"z2184 COMPLETE: {n_pass}/6 tests passed")
    print(f"{'=' * 65}")


if __name__ == '__main__':
    main()
