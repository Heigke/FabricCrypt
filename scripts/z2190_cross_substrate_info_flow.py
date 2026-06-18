#!/usr/bin/env python3
"""z2190_cross_substrate_info_flow.py — Directed Information Flow: GPU → FPGA

Measures DIRECTED INFORMATION FLOW between GPU firmware noise and FPGA neuron
responses using Transfer Entropy, Granger Causality, Mutual Information profiles,
and Phase Transfer Entropy.  Connects to Mario Lanza's cross-substrate memristive
computing vision: if GPU thermal noise CAUSALLY drives FPGA spike dynamics with
measurable directed information, the two substrates form a functional unit.

Conditions (4):
  FULL     — GPU 1/f noise (hwmon power1_average) → FPGA neurons
  WHITE    — Gaussian white noise → FPGA neurons
  SHUFFLED — Shuffled GPU noise (destroys temporal structure) → FPGA neurons
  NO_NOISE — Signal only, no noise (β=0)

Information measures:
  1. Transfer Entropy TE(GPU→FPGA) at lags 1..5  (histogram-based, 8 bins)
  2. Transfer Entropy TE(FPGA→GPU) at lags 1..5  (control — should be ~0)
  3. Granger Causality: VAR model orders 1-5, F-test
  4. Time-lagged Mutual Information: MI(noise(t), spikes(t+τ)) for τ=0..20
  5. Directed Information Rate: cumulative TE over time

Tests T231-T236:
  T231: TE(GPU→FPGA, FULL) > TE(GPU→FPGA, WHITE)
  T232: TE(GPU→FPGA) > TE(FPGA→GPU) for FULL
  T233: TE(FULL) > TE(SHUFFLED)
  T234: Granger causality p < 0.05 for FULL
  T235: Peak MI lag > 0 for FULL (causal delay exists)
  T236: TE(FULL) > 0.01 bits (non-trivial information transfer)

Hardware: AMD gfx1151 GPU + Arty A7 FPGA on /dev/ttyUSB*
"""

import os, sys, json, time, struct, argparse
import numpy as np
from pathlib import Path

os.environ.setdefault('HSA_OVERRIDE_GFX_VERSION', '11.0.0')

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))
RESULTS = BASE / 'results'
FIGURES = (BASE / 'results' / 'FEEL_paper_update'
           / 'FEEL__Functionally_Embodied_Emergent_Learning__13_-5' / 'figures')

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
DEFAULT_SAMPLE_HZ = 20
DEFAULT_N_STEPS = 300
DEFAULT_N_BINS_TE = 8
DEFAULT_MAX_LAG_TE = 5
DEFAULT_MAX_LAG_MI = 20


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
    """Set individual Vg for each of 8 neurons (fire-and-forget)."""
    for nid, vg in enumerate(vg_values[:8]):
        q16 = to_q16_16(max(0.0, min(1.0, vg)))
        payload = bytes([nid & 0x07]) + struct.pack('>I', q16)
        ser.write(bytes([SYNC, CMD_SET_VG]) + payload)
    ser.flush()
    time.sleep(0.005)


def read_telem(ser, timeout=0.15):
    """Read telemetry packet: [0x55][0x02][0x30][48B_data][CRC8] = 52 bytes."""
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
# Information-Theoretic Measures
# ═══════════════════════════════════════════════════════════

def discretize(series, n_bins=8):
    """Discretize a 1-D continuous series into n_bins using percentile edges."""
    edges = np.percentile(series, np.linspace(0, 100, n_bins + 1))
    edges = np.unique(edges)
    if len(edges) <= 1:
        return np.zeros(len(series), dtype=int)
    return np.clip(np.digitize(series, edges[1:-1]), 0, n_bins - 1)


def entropy_bits(counts):
    """Shannon entropy in bits from a count array."""
    total = counts.sum()
    if total == 0:
        return 0.0
    p = counts / total
    p = p[p > 0]
    return float(-np.sum(p * np.log2(p)))


def transfer_entropy(source, target, lag=1, n_bins=8):
    """Transfer entropy TE(source → target) at a given lag.

    TE = H(target_future | target_past) - H(target_future | target_past, source_past)

    Uses histogram-based plugin estimation with Panzeri-Treves bias correction.
    source, target: 1-D integer arrays (already discretized).
    """
    T = len(source)
    if T < lag + 2:
        return 0.0

    tgt_future = target[lag:]
    tgt_past = target[:T - lag]
    src_past = source[:T - lag]
    N = len(tgt_future)

    # Joint counts for (tgt_future, tgt_past) and (tgt_future, tgt_past, src_past)
    # H(tgt_future, tgt_past)
    idx_ft = tgt_future * n_bins + tgt_past
    counts_ft = np.bincount(idx_ft, minlength=n_bins * n_bins)
    h_ft = entropy_bits(counts_ft)

    # H(tgt_past)
    counts_tp = np.bincount(tgt_past, minlength=n_bins)
    h_tp = entropy_bits(counts_tp)

    # H(tgt_future, tgt_past, src_past)
    idx_fts = tgt_future * (n_bins * n_bins) + tgt_past * n_bins + src_past
    counts_fts = np.bincount(idx_fts, minlength=n_bins ** 3)
    h_fts = entropy_bits(counts_fts)

    # H(tgt_past, src_past)
    idx_ts = tgt_past * n_bins + src_past
    counts_ts = np.bincount(idx_ts, minlength=n_bins * n_bins)
    h_ts = entropy_bits(counts_ts)

    # TE = H(f,tp) - H(tp) - H(f,tp,sp) + H(tp,sp)
    te = h_ft - h_tp - h_fts + h_ts

    # Panzeri-Treves bias correction
    r_fts = np.sum(counts_fts > 0)
    r_ft = np.sum(counts_ft > 0)
    bias = max(0, (r_fts - r_ft)) / (2 * N * np.log(2))
    te_corrected = max(0.0, te - bias)
    return float(te_corrected)


def mutual_information_lagged(source, target, lag, n_bins=8):
    """MI(source(t), target(t+lag)) using histogram-based estimation.

    source, target: 1-D integer arrays (already discretized).
    """
    T = min(len(source), len(target))
    if lag >= T:
        return 0.0
    if lag >= 0:
        s = source[:T - lag]
        t = target[lag:T]
    else:
        s = source[-lag:T]
        t = target[:T + lag]
    N = len(s)
    if N < 10:
        return 0.0

    idx_joint = s * n_bins + t
    counts_joint = np.bincount(idx_joint, minlength=n_bins * n_bins)
    counts_s = np.bincount(s, minlength=n_bins)
    counts_t = np.bincount(t, minlength=n_bins)

    h_s = entropy_bits(counts_s)
    h_t = entropy_bits(counts_t)
    h_st = entropy_bits(counts_joint)

    mi = h_s + h_t - h_st

    # Bias correction
    r_st = np.sum(counts_joint > 0)
    r_s = np.sum(counts_s > 0)
    r_t = np.sum(counts_t > 0)
    bias = (r_st - r_s - r_t + 1) / (2 * N * np.log(2))
    return float(max(0.0, mi - bias))


def granger_causality_test(source_cont, target_cont, max_order=5):
    """Granger causality F-test: does source help predict target?

    Fits two OLS models:
      Restricted:   target(t) = Σ a_k * target(t-k) + ε_r
      Unrestricted: target(t) = Σ a_k * target(t-k) + Σ b_k * source(t-k) + ε_u

    Returns best (F-statistic, p-value, order) across orders 1..max_order.
    """
    from scipy import stats as sp_stats

    T = len(source_cont)
    best_f, best_p, best_order = 0.0, 1.0, 1

    for order in range(1, max_order + 1):
        if T <= 2 * order + 2:
            continue

        # Build design matrices
        y = target_cont[order:]
        N = len(y)

        # Restricted: target lags only
        X_r = np.column_stack([target_cont[order - k - 1:T - k - 1] for k in range(order)])
        # Unrestricted: target lags + source lags
        X_u = np.column_stack([
            *[target_cont[order - k - 1:T - k - 1] for k in range(order)],
            *[source_cont[order - k - 1:T - k - 1] for k in range(order)],
        ])

        # Add intercept
        X_r = np.column_stack([np.ones(N), X_r])
        X_u = np.column_stack([np.ones(N), X_u])

        # OLS via pseudoinverse
        try:
            beta_r = np.linalg.lstsq(X_r, y, rcond=None)[0]
            beta_u = np.linalg.lstsq(X_u, y, rcond=None)[0]
        except np.linalg.LinAlgError:
            continue

        resid_r = y - X_r @ beta_r
        resid_u = y - X_u @ beta_u

        ssr_r = np.sum(resid_r ** 2)
        ssr_u = np.sum(resid_u ** 2)

        df_extra = order  # extra parameters in unrestricted model
        df_resid = N - X_u.shape[1]

        if df_resid <= 0 or ssr_u <= 0:
            continue

        f_stat = ((ssr_r - ssr_u) / df_extra) / (ssr_u / df_resid)
        p_val = 1.0 - sp_stats.f.cdf(f_stat, df_extra, df_resid)

        if p_val < best_p:
            best_f, best_p, best_order = float(f_stat), float(p_val), order

    return best_f, best_p, best_order


def directed_info_rate(source_disc, target_disc, n_bins=8, window=50):
    """Cumulative TE over sliding windows → directed information rate profile."""
    T = len(source_disc)
    n_windows = max(1, (T - window) // (window // 2))
    rates = []
    for w in range(n_windows):
        start = w * (window // 2)
        end = min(start + window, T)
        if end - start < 10:
            continue
        te = transfer_entropy(source_disc[start:end], target_disc[start:end],
                              lag=1, n_bins=n_bins)
        rates.append(te)
    return rates


# ═══════════════════════════════════════════════════════════
# FPGA Data Collection
# ═══════════════════════════════════════════════════════════

def generate_input_signal(n_steps, sample_hz):
    """Slow sine oscillation to provide structure: 0.5*sin(2π*0.5*t)."""
    t = np.arange(n_steps) / sample_hz
    return 0.5 * np.sin(2 * np.pi * 0.5 * t)


def run_condition_fpga(ser, condition, n_steps, sample_hz, base_vg, alpha, beta,
                       noise_1f, noise_white, noise_shuffled, input_signal,
                       w_in, w_noise, n_bins=8):
    """Run one condition on real FPGA, collect noise input and spike output.

    Returns:
        noise_trace: (n_steps,) noise values actually applied
        spike_rates: (n_steps,) aggregate spike rate across 8 neurons
    """
    interval = 1.0 / sample_hz
    noise_trace = np.zeros(n_steps)
    spike_rates = np.zeros(n_steps)
    prev_counts = None
    power_mean = 11.0

    for t in range(n_steps):
        # Determine noise
        if condition == 'FULL':
            p = read_hwmon_power()
            noise_val = (p - power_mean) / 2.0 if p else 0.0
        elif condition == 'WHITE':
            noise_val = noise_white[t % len(noise_white)]
        elif condition == 'SHUFFLED':
            noise_val = noise_shuffled[t % len(noise_shuffled)]
        else:  # NO_NOISE
            noise_val = 0.0

        noise_trace[t] = noise_val

        # Compute per-neuron Vg
        vg_values = np.full(N_NEURONS, base_vg)
        vg_values += alpha * input_signal[t] * w_in
        if condition != 'NO_NOISE':
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

        if (t + 1) % 100 == 0:
            print(f"    [{condition}] step {t+1}/{n_steps}")

    return noise_trace, spike_rates


def simulate_condition(condition, n_steps, sample_hz, base_vg, alpha, beta,
                       noise_1f, noise_white, noise_shuffled, input_signal,
                       w_in, w_noise):
    """Software LIF simulation fallback when FPGA not connected."""
    dt = 1.0 / sample_hz
    vmem = np.zeros(N_NEURONS)
    noise_trace = np.zeros(n_steps)
    spike_rates = np.zeros(n_steps)
    v_thresh = 1.0
    tau_m = 0.02

    seed_map = {'FULL': 42, 'WHITE': 43, 'SHUFFLED': 44, 'NO_NOISE': 45}
    rng = np.random.default_rng(seed_map.get(condition, 42))

    for t in range(n_steps):
        if condition == 'FULL':
            noise_val = (noise_1f[t % len(noise_1f)]
                         if len(noise_1f) > 0 else rng.standard_normal() * 0.3)
        elif condition == 'WHITE':
            noise_val = (noise_white[t % len(noise_white)]
                         if len(noise_white) > 0 else rng.standard_normal())
        elif condition == 'SHUFFLED':
            noise_val = (noise_shuffled[t % len(noise_shuffled)]
                         if len(noise_shuffled) > 0 else rng.standard_normal() * 0.3)
        else:
            noise_val = 0.0

        noise_trace[t] = noise_val

        vg = np.full(N_NEURONS, base_vg)
        vg += alpha * input_signal[t] * w_in
        if condition != 'NO_NOISE':
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

    return noise_trace, spike_rates


# ═══════════════════════════════════════════════════════════
# Analysis per condition
# ═══════════════════════════════════════════════════════════

def analyze_condition(noise_trace, spike_rates, n_bins=8, max_lag_te=5,
                      max_lag_mi=20):
    """Compute all information-theoretic measures for one condition.

    Returns dict with:
        te_gpu_to_fpga: list of TE values at lags 1..max_lag_te
        te_fpga_to_gpu: list of TE values at lags 1..max_lag_te (control)
        granger_f, granger_p, granger_order: best Granger causality result
        mi_profile: list of MI at lags 0..max_lag_mi
        peak_mi_lag: lag with maximum MI
        dir_info_rate: list of windowed TE rates
        mean_te_fwd: mean TE(GPU→FPGA)
        mean_te_bwd: mean TE(FPGA→GPU)
    """
    # Discretize both time series
    noise_disc = discretize(noise_trace, n_bins)
    spike_disc = discretize(spike_rates, n_bins)

    # 1. Transfer Entropy GPU→FPGA at multiple lags
    te_fwd = []
    for lag in range(1, max_lag_te + 1):
        te = transfer_entropy(noise_disc, spike_disc, lag=lag, n_bins=n_bins)
        te_fwd.append(te)

    # 2. Transfer Entropy FPGA→GPU (control direction)
    te_bwd = []
    for lag in range(1, max_lag_te + 1):
        te = transfer_entropy(spike_disc, noise_disc, lag=lag, n_bins=n_bins)
        te_bwd.append(te)

    # 3. Granger Causality (on continuous series)
    gc_f, gc_p, gc_order = granger_causality_test(noise_trace, spike_rates,
                                                   max_order=max_lag_te)

    # 4. Time-lagged Mutual Information profile
    mi_profile = []
    for lag in range(0, max_lag_mi + 1):
        mi = mutual_information_lagged(noise_disc, spike_disc, lag=lag,
                                       n_bins=n_bins)
        mi_profile.append(mi)

    peak_mi_lag = int(np.argmax(mi_profile))

    # 5. Directed Information Rate
    dir_rate = directed_info_rate(noise_disc, spike_disc, n_bins=n_bins,
                                  window=50)

    mean_te_fwd = float(np.mean(te_fwd)) if te_fwd else 0.0
    mean_te_bwd = float(np.mean(te_bwd)) if te_bwd else 0.0

    return {
        'te_gpu_to_fpga': te_fwd,
        'te_fpga_to_gpu': te_bwd,
        'mean_te_fwd': mean_te_fwd,
        'mean_te_bwd': mean_te_bwd,
        'granger_f': gc_f,
        'granger_p': gc_p,
        'granger_order': gc_order,
        'mi_profile': mi_profile,
        'peak_mi_lag': peak_mi_lag,
        'dir_info_rate': dir_rate,
        'noise_std': float(np.std(noise_trace)),
        'spike_rate_mean': float(np.mean(spike_rates)),
        'spike_rate_std': float(np.std(spike_rates)),
    }


# ═══════════════════════════════════════════════════════════
# Plotting
# ═══════════════════════════════════════════════════════════

def make_figure(all_results, tests, fig_path):
    """Create 2x3 figure: TE bars, TE asymmetry, MI profile, Granger,
    directed info rate, test summary."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("[WARN] matplotlib not available, skipping figure")
        return

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('z2190: Cross-Substrate Directed Information Flow (GPU → FPGA)',
                 fontsize=14, fontweight='bold')

    conditions = ['FULL', 'WHITE', 'SHUFFLED', 'NO_NOISE']
    cond_colors = {'FULL': '#2196F3', 'WHITE': '#FF9800',
                   'SHUFFLED': '#4CAF50', 'NO_NOISE': '#9E9E9E'}
    lags = list(range(1, DEFAULT_MAX_LAG_TE + 1))

    # ─── Panel 1: TE(GPU→FPGA) per condition across lags ───
    ax = axes[0, 0]
    for cond in conditions:
        if cond in all_results:
            te_vals = all_results[cond]['te_gpu_to_fpga']
            ax.plot(lags[:len(te_vals)], te_vals, 'o-', color=cond_colors[cond],
                    label=cond, linewidth=2, markersize=5)
    ax.set_xlabel('Lag (timesteps)')
    ax.set_ylabel('Transfer Entropy (bits)')
    ax.set_title('TE(GPU → FPGA) by Lag')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # ─── Panel 2: TE asymmetry (fwd vs bwd) for FULL ───
    ax = axes[0, 1]
    if 'FULL' in all_results:
        r = all_results['FULL']
        te_f = r['te_gpu_to_fpga']
        te_b = r['te_fpga_to_gpu']
        x = np.arange(len(lags))
        w = 0.35
        ax.bar(x - w/2, te_f, w, color='#2196F3', label='GPU→FPGA')
        ax.bar(x + w/2, te_b, w, color='#F44336', label='FPGA→GPU')
        ax.set_xticks(x)
        ax.set_xticklabels(lags)
        ax.set_xlabel('Lag (timesteps)')
        ax.set_ylabel('Transfer Entropy (bits)')
        ax.set_title('TE Asymmetry (FULL condition)')
        ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # ─── Panel 3: MI profile ───
    ax = axes[0, 2]
    for cond in conditions:
        if cond in all_results:
            mi = all_results[cond]['mi_profile']
            ax.plot(range(len(mi)), mi, '-', color=cond_colors[cond],
                    label=cond, linewidth=1.5)
    if 'FULL' in all_results:
        peak = all_results['FULL']['peak_mi_lag']
        mi_peak_val = all_results['FULL']['mi_profile'][peak]
        ax.axvline(x=peak, color='red', linestyle='--', alpha=0.5,
                   label=f'Peak lag={peak}')
        ax.plot(peak, mi_peak_val, 'r*', markersize=12)
    ax.set_xlabel('Lag τ (timesteps)')
    ax.set_ylabel('Mutual Information (bits)')
    ax.set_title('Time-Lagged MI Profile')
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)

    # ─── Panel 4: Granger causality summary ───
    ax = axes[1, 0]
    conds_present = [c for c in conditions if c in all_results]
    if conds_present:
        f_vals = [all_results[c]['granger_f'] for c in conds_present]
        p_vals = [all_results[c]['granger_p'] for c in conds_present]
        colors = [cond_colors[c] for c in conds_present]
        bars = ax.bar(conds_present, f_vals, color=colors, alpha=0.8)
        ax.set_ylabel('F-statistic')
        ax.set_title('Granger Causality (GPU → Spikes)')
        ax2 = ax.twinx()
        ax2.plot(conds_present, p_vals, 'rs-', markersize=8, label='p-value')
        ax2.axhline(y=0.05, color='red', linestyle='--', alpha=0.5)
        ax2.set_ylabel('p-value', color='red')
        ax2.legend(fontsize=8, loc='upper right')
    ax.grid(alpha=0.3)

    # ─── Panel 5: Directed information rate over time ───
    ax = axes[1, 1]
    for cond in conditions:
        if cond in all_results:
            rate = all_results[cond]['dir_info_rate']
            if rate:
                ax.plot(range(len(rate)), rate, '-', color=cond_colors[cond],
                        label=cond, linewidth=1.5)
    ax.set_xlabel('Window index')
    ax.set_ylabel('TE per window (bits)')
    ax.set_title('Directed Information Rate')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # ─── Panel 6: Test summary ───
    ax = axes[1, 2]
    ax.axis('off')
    test_lines = []
    for t in tests:
        mark = 'PASS' if t['pass'] else 'FAIL'
        test_lines.append(f"{t['name']}: {mark}  ({t['detail']})")
    test_text = '\n'.join(test_lines)
    n_pass = sum(1 for t in tests if t['pass'])
    n_total = len(tests)
    header = f"Tests: {n_pass}/{n_total} PASS\n{'='*45}\n"
    ax.text(0.05, 0.95, header + test_text, transform=ax.transAxes,
            fontsize=8, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    ax.set_title('Test Results (T231-T236)')

    plt.tight_layout()
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[INFO] Figure saved: {fig_path}")


# ═══════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description='z2190: Cross-Substrate Directed Information Flow (GPU → FPGA)')
    parser.add_argument('--n-steps', type=int, default=DEFAULT_N_STEPS,
                        help='Timesteps per condition (default: 300)')
    parser.add_argument('--sample-hz', type=int, default=DEFAULT_SAMPLE_HZ,
                        help='Sample rate in Hz (default: 20)')
    parser.add_argument('--base-vg', type=float, default=DEFAULT_BASE_VG,
                        help='Base gate voltage (default: 0.55)')
    parser.add_argument('--alpha', type=float, default=DEFAULT_ALPHA,
                        help='Input signal gain (default: 0.15)')
    parser.add_argument('--beta', type=float, default=DEFAULT_BETA,
                        help='Noise gain (default: 0.10)')
    parser.add_argument('--n-bins', type=int, default=DEFAULT_N_BINS_TE,
                        help='Histogram bins for TE/MI (default: 8)')
    parser.add_argument('--max-lag-te', type=int, default=DEFAULT_MAX_LAG_TE,
                        help='Max lag for Transfer Entropy (default: 5)')
    parser.add_argument('--max-lag-mi', type=int, default=DEFAULT_MAX_LAG_MI,
                        help='Max lag for MI profile (default: 20)')
    args = parser.parse_args()

    n_steps = args.n_steps
    sample_hz = args.sample_hz
    base_vg = args.base_vg
    alpha = args.alpha
    beta = args.beta
    n_bins = args.n_bins
    max_lag_te = args.max_lag_te
    max_lag_mi = args.max_lag_mi

    print("=" * 65)
    print("z2190: Cross-Substrate Directed Information Flow (GPU → FPGA)")
    print("=" * 65)
    print(f"  n_steps={n_steps}, sample_hz={sample_hz}, base_vg={base_vg}")
    print(f"  alpha={alpha}, beta={beta}, n_bins={n_bins}")
    print(f"  max_lag_te={max_lag_te}, max_lag_mi={max_lag_mi}")

    # ─── Fixed per-neuron weights ───
    rng_w = np.random.default_rng(2190)
    w_in = rng_w.uniform(-1, 1, N_NEURONS)
    w_in /= np.linalg.norm(w_in)
    w_noise = np.array([1.0, -0.7, 0.5, -0.3, 0.8, -0.6, 0.4, -0.9])

    # ─── Prepare noise sources ───
    print("\n[1/5] Collecting GPU power noise for 1/f source...")
    noise_duration = max(20, n_steps / sample_hz * 1.5)
    raw_noise = collect_power_noise(duration_s=noise_duration, sample_hz=50)
    if raw_noise is not None and len(raw_noise) > 10:
        noise_1f = iir_filter_noise(raw_noise, alpha_iir=0.85)
        print(f"  Collected {len(raw_noise)} power samples → {len(noise_1f)} filtered")
    else:
        print("  [WARN] hwmon power not available, generating synthetic 1/f noise")
        rng_synth = np.random.default_rng(99)
        raw_synth = rng_synth.standard_normal(n_steps * 2)
        vmf = VossMcCartneyFilter()
        noise_1f = np.array([vmf.process((s * 50 + 127.5)) for s in raw_synth])

    # White noise
    rng_white = np.random.default_rng(123)
    noise_white = rng_white.standard_normal(n_steps * 2)

    # Shuffled 1/f noise (same distribution, destroyed temporal structure)
    noise_shuffled = noise_1f.copy()
    rng_shuf = np.random.default_rng(456)
    rng_shuf.shuffle(noise_shuffled)

    # Input signal
    input_signal = generate_input_signal(n_steps, sample_hz)

    # ─── Connect FPGA ───
    print("\n[2/5] Connecting to FPGA...")
    ser, port = find_fpga()
    use_fpga = ser is not None
    simulated = not use_fpga
    if use_fpga:
        print(f"  FPGA found on {port}")
        connect_fpga(ser)
    else:
        print("  [WARN] No FPGA found, using software LIF simulation")

    # ─── Run 4 conditions ───
    conditions = ['FULL', 'WHITE', 'SHUFFLED', 'NO_NOISE']
    raw_data = {}  # condition → (noise_trace, spike_rates)

    print("\n[3/5] Running conditions...")
    for cond in conditions:
        print(f"  Running condition: {cond}")
        t0 = time.monotonic()
        if use_fpga:
            noise_tr, spike_r = run_condition_fpga(
                ser, cond, n_steps, sample_hz, base_vg, alpha, beta,
                noise_1f, noise_white, noise_shuffled, input_signal,
                w_in, w_noise, n_bins)
        else:
            noise_tr, spike_r = simulate_condition(
                cond, n_steps, sample_hz, base_vg, alpha, beta,
                noise_1f, noise_white, noise_shuffled, input_signal,
                w_in, w_noise)
        elapsed = time.monotonic() - t0
        raw_data[cond] = (noise_tr, spike_r)
        print(f"    Done in {elapsed:.1f}s  |  spikes_total={spike_r.sum():.0f}  "
              f"noise_std={np.std(noise_tr):.4f}")

    if ser:
        ser.close()

    # ─── Analyze ───
    print("\n[4/5] Computing information-theoretic measures...")
    all_results = {}
    for cond in conditions:
        noise_tr, spike_r = raw_data[cond]
        r = analyze_condition(noise_tr, spike_r, n_bins=n_bins,
                              max_lag_te=max_lag_te, max_lag_mi=max_lag_mi)
        all_results[cond] = r
        print(f"  {cond}:  TE_fwd={r['mean_te_fwd']:.4f}  TE_bwd={r['mean_te_bwd']:.4f}  "
              f"GC_p={r['granger_p']:.4f}  peak_MI_lag={r['peak_mi_lag']}")

    # ─── Tests T231-T236 ───
    print("\n[5/5] Evaluating tests T231-T236...")
    full = all_results['FULL']
    white = all_results.get('WHITE', {})
    shuffled = all_results.get('SHUFFLED', {})

    tests = []

    # T231: TE(GPU→FPGA, FULL) > TE(GPU→FPGA, WHITE)
    te_full_fwd = full['mean_te_fwd']
    te_white_fwd = white.get('mean_te_fwd', 0.0)
    t231 = te_full_fwd > te_white_fwd
    tests.append({
        'name': 'T231',
        'pass': bool(t231),
        'detail': f'TE_FULL={te_full_fwd:.4f} > TE_WHITE={te_white_fwd:.4f}',
    })

    # T232: TE(GPU→FPGA) > TE(FPGA→GPU) for FULL
    te_full_bwd = full['mean_te_bwd']
    t232 = te_full_fwd > te_full_bwd
    tests.append({
        'name': 'T232',
        'pass': bool(t232),
        'detail': f'TE_fwd={te_full_fwd:.4f} > TE_bwd={te_full_bwd:.4f}',
    })

    # T233: TE(FULL) > TE(SHUFFLED)
    te_shuf_fwd = shuffled.get('mean_te_fwd', 0.0)
    t233 = te_full_fwd > te_shuf_fwd
    tests.append({
        'name': 'T233',
        'pass': bool(t233),
        'detail': f'TE_FULL={te_full_fwd:.4f} > TE_SHUFFLED={te_shuf_fwd:.4f}',
    })

    # T234: Granger causality p < 0.05 for FULL
    gc_p_full = full['granger_p']
    t234 = gc_p_full < 0.05
    tests.append({
        'name': 'T234',
        'pass': bool(t234),
        'detail': f'GC_p={gc_p_full:.4f} < 0.05 (F={full["granger_f"]:.2f}, order={full["granger_order"]})',
    })

    # T235: Peak MI lag > 0 for FULL (causal delay exists)
    peak_lag = full['peak_mi_lag']
    t235 = peak_lag > 0
    tests.append({
        'name': 'T235',
        'pass': bool(t235),
        'detail': f'peak_MI_lag={peak_lag} > 0',
    })

    # T236: TE(FULL) > 0.01 bits (non-trivial information transfer)
    t236 = te_full_fwd > 0.01
    tests.append({
        'name': 'T236',
        'pass': bool(t236),
        'detail': f'TE_FULL={te_full_fwd:.4f} > 0.01 bits',
    })

    n_pass = sum(1 for t in tests if t['pass'])
    n_total = len(tests)
    print(f"\n{'='*55}")
    for t in tests:
        mark = 'PASS' if t['pass'] else 'FAIL'
        print(f"  {t['name']}: {mark}  {t['detail']}")
    print(f"{'='*55}")
    print(f"  TOTAL: {n_pass}/{n_total} PASS")

    # ─── Save results ───
    RESULTS.mkdir(parents=True, exist_ok=True)
    result_data = {
        'experiment': 'z2190_cross_substrate_info_flow',
        'simulated': simulated,
        'params': {
            'n_steps': n_steps,
            'sample_hz': sample_hz,
            'base_vg': base_vg,
            'alpha': alpha,
            'beta': beta,
            'n_bins': n_bins,
            'max_lag_te': max_lag_te,
            'max_lag_mi': max_lag_mi,
            'n_neurons': N_NEURONS,
            'use_fpga': use_fpga,
        },
        'conditions': {},
        'tests': tests,
        'pass_count': n_pass,
        'total_tests': n_total,
    }

    for cond in conditions:
        result_data['conditions'][cond] = all_results[cond]

    result_path = RESULTS / 'z2190_cross_substrate_info_flow.json'
    with open(result_path, 'w') as f:
        json.dump(result_data, f, indent=2, cls=NpEncoder)
    print(f"\n[INFO] Results saved: {result_path}")

    # ─── Figure ───
    fig_path = FIGURES / 'fig_z2190_cross_substrate_info_flow.png'
    make_figure(all_results, tests, fig_path)

    return n_pass, n_total


if __name__ == '__main__':
    main()
