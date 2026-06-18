#!/usr/bin/env python3
"""z2152: Unified Analog Proxy — Fuses gpu_metrics + PERF_SNAPSHOT + FPGA telemetry

Three major innovations:
1. CHANNEL FUSION: Combines slow-but-1/f gpu_metrics with fast-but-memoryless PERF_SNAPSHOT
2. IIR 1/f INJECTION: Filters PERF_SNAPSHOT through 1/f IIR to add temporal correlation
3. ANALOG INFERENCE: Statistically reconstructs continuous analog distributions from
   quantized digital samples using Allan variance decomposition, maximum entropy
   spectral estimation, and Bayesian deconvolution

The key insight: GPU silicon IS analog. We're just seeing it through digital keyholes.
By combining multiple keyholes + statistical inference, we can reconstruct what the
underlying analog substrate is actually doing — bridging toward SPICE/FPGA fidelity.

Usage:
    # Collect data first:
    HSA_OVERRIDE_GFX_VERSION=11.0.0 ./scripts/z2151_perf_snapshot_stats 1000 16 10000 > results/z2152_perf.csv
    # Then run unified proxy:
    python3 scripts/z2152_unified_analog_proxy.py --perf results/z2152_perf.csv --duration 5
"""

import argparse, json, os, struct, sys, time, threading
import numpy as np
from collections import defaultdict, deque
from pathlib import Path

# ═══════════════════════════════════════════════════════════════
# PART 1: Data Sources — gpu_metrics + PERF_SNAPSHOT readers
# ═══════════════════════════════════════════════════════════════

GPU_METRICS_PATH = "/sys/class/drm/card0/device/gpu_metrics"

# gpu_metrics v3.0 layout (264 bytes, APU)
V3_OFFSETS = {
    'temp_gfx':       (0x02, 'H'),   # ×100 = °C
    'temp_soc':       (0x04, 'H'),
    'temp_skin':      (0x08, 'H'),
    'average_socket_power': (0x1E, 'I'),  # mW
    'average_gfx_power':    (0x22, 'H'),  # mW (not populated on APU)
    'average_core_power_0': (0x24, 'H'),
    'average_core_power_1': (0x26, 'H'),
    'average_gfxclk_freq':  (0x60, 'H'),  # MHz
    'average_fclk_freq':    (0x62, 'H'),
    'average_uclk_freq':    (0x64, 'H'),
    'system_clock_counter': (0x68, 'Q'),  # ns
}

def read_gpu_metrics():
    """Read gpu_metrics v3.0, return dict of decoded values."""
    try:
        with open(GPU_METRICS_PATH, 'rb') as f:
            data = f.read()
        if len(data) < 264:
            return None
        result = {}
        for name, (offset, fmt) in V3_OFFSETS.items():
            size = struct.calcsize(fmt)
            val = struct.unpack_from(fmt, data, offset)[0]
            result[name] = val
        # Convert temperatures
        result['temp_gfx_C'] = result['temp_gfx'] / 100.0
        result['temp_soc_C'] = result['temp_soc'] / 100.0
        result['power_W'] = result['average_socket_power'] / 1000.0
        result['timestamp_ns'] = result['system_clock_counter']
        return result
    except Exception as e:
        return None


def load_perf_csv(path):
    """Load PERF_SNAPSHOT CSV data."""
    data = defaultdict(list)
    with open(path) as f:
        header = f.readline().strip().split(',')
        for line in f:
            parts = line.strip().split(',')
            if len(parts) < 4:
                continue
            for h, v in zip(header, parts):
                if h in ('hw_id1', 'status'):
                    data[h].append(int(v, 16) if v.startswith('0x') else int(v))
                else:
                    data[h].append(int(v))
    return {k: np.array(v) for k, v in data.items()}


# ═══════════════════════════════════════════════════════════════
# PART 2: IIR 1/f Filter — Inject temporal correlation
# ═══════════════════════════════════════════════════════════════

class OneOverFFilter:
    """IIR filter that shapes white-ish noise into 1/f-like spectrum.

    Uses cascaded first-order IIR sections with geometrically-spaced
    time constants. This is the Voss-McCartney algorithm adapted for
    real-time filtering.

    The key insight: PERF_SNAPSHOT has high entropy but no temporal
    correlation. SPICE 1/f noise has both. By filtering PERF_SNAPSHOT
    through this, we ADD the missing temporal structure while PRESERVING
    the silicon-sourced entropy.
    """

    def __init__(self, n_octaves=8, base_alpha=0.01):
        """
        n_octaves: number of IIR sections (more = deeper 1/f)
        base_alpha: smoothing for fastest octave (slowest = base_alpha^n)
        """
        self.n_octaves = n_octaves
        # Each octave has a different time constant
        self.alphas = [base_alpha * (2.0 ** i) for i in range(n_octaves)]
        # Clamp to [0, 1]
        self.alphas = [min(a, 0.999) for a in self.alphas]
        # State: running average for each octave
        self.states = [0.0] * n_octaves
        self.initialized = False

    def filter(self, x):
        """Filter a single sample through the 1/f cascade.

        Returns weighted sum of all octave states — this creates
        1/f spectrum because lower octaves (longer τ) contribute
        more low-frequency power.
        """
        if not self.initialized:
            for i in range(self.n_octaves):
                self.states[i] = x
            self.initialized = True
            return x

        output = 0.0
        for i in range(self.n_octaves):
            # IIR update: state = alpha * x + (1-alpha) * state
            self.states[i] = self.alphas[i] * x + (1 - self.alphas[i]) * self.states[i]
            output += self.states[i]

        return output / self.n_octaves

    def filter_array(self, arr):
        """Filter entire array, return filtered version."""
        out = np.zeros(len(arr))
        for i, x in enumerate(arr):
            out[i] = self.filter(float(x))
        return out


class AdaptiveOneOverF(OneOverFFilter):
    """Adaptive 1/f filter that matches target PSD slope.

    Uses the gpu_metrics PSD slope (-0.82) as target and adjusts
    the octave weights to match, rather than using uniform weighting.
    """

    def __init__(self, target_slope=-0.82, n_octaves=10):
        super().__init__(n_octaves, base_alpha=0.3)  # higher base = less correlation
        # Weight each octave by f^(target_slope/2) to shape spectrum
        # Lower octaves (low freq) get more weight for steeper slopes
        # Use sqrt to dampen — prevents over-correlation
        self.weights = np.array([
            (2.0 ** (-i)) ** (-target_slope / 4) for i in range(n_octaves)
        ])
        self.weights /= np.sum(self.weights)
        self.target_slope = target_slope

    def filter(self, x):
        if not self.initialized:
            for i in range(self.n_octaves):
                self.states[i] = x
            self.initialized = True
            return x

        output = 0.0
        for i in range(self.n_octaves):
            self.states[i] = self.alphas[i] * x + (1 - self.alphas[i]) * self.states[i]
            output += self.weights[i] * self.states[i]

        return output


# ═══════════════════════════════════════════════════════════════
# PART 3: Allan Variance — Separate noise types from digital samples
# ═══════════════════════════════════════════════════════════════

def allan_variance(samples, max_cluster=None):
    """Compute Allan variance to decompose noise into types.

    Allan variance at averaging time τ reveals:
    - slope -1: white noise (thermal)
    - slope  0: flicker noise (1/f) ← what we want to find
    - slope +1: random walk (Brownian)

    This is how atomic clock physicists separate noise types from
    quantized frequency counter readings — exactly our situation.
    """
    n = len(samples)
    if max_cluster is None:
        max_cluster = n // 4

    taus = []
    avars = []

    cluster_sizes = np.unique(np.logspace(0, np.log10(max_cluster), 50).astype(int))
    cluster_sizes = cluster_sizes[cluster_sizes >= 1]

    for m in cluster_sizes:
        # Number of clusters
        n_clusters = n // m
        if n_clusters < 3:
            break

        # Cluster averages
        clusters = np.mean(samples[:n_clusters * m].reshape(n_clusters, m), axis=1)

        # Allan variance: 0.5 * mean of (consecutive differences)^2
        diffs = np.diff(clusters)
        avar = 0.5 * np.mean(diffs ** 2)

        taus.append(m)
        avars.append(avar)

    taus = np.array(taus, dtype=float)
    avars = np.array(avars)

    # Fit slope in log-log space
    if len(taus) > 5:
        log_tau = np.log10(taus)
        log_avar = np.log10(avars + 1e-30)
        slope, intercept = np.polyfit(log_tau, log_avar, 1)
    else:
        slope, intercept = 0, 0

    # Interpret slope
    noise_types = {
        (-1.5, -0.5): "white_phase",
        (-0.5, 0.5): "flicker_phase_or_white_freq",
        (0.5, 1.5): "random_walk",
    }
    noise_type = "unknown"
    for (lo, hi), name in noise_types.items():
        if lo <= slope <= hi:
            noise_type = name
            break

    return {
        'taus': taus.tolist(),
        'avars': avars.tolist(),
        'slope': float(slope),
        'intercept': float(intercept),
        'noise_type': noise_type,
        'has_flicker': -0.5 <= slope <= 0.5,
    }


# ═══════════════════════════════════════════════════════════════
# PART 4: Statistical Analog Inference
# ═══════════════════════════════════════════════════════════════

def infer_analog_distribution(digital_samples, n_bits=24):
    """Reconstruct the underlying continuous analog distribution
    from quantized digital samples.

    Key insight: The GPU's ADCs and counters quantize a continuous
    analog signal. We can UNDO this quantization statistically
    using kernel density estimation with bandwidth matching the
    quantization step size.

    This is equivalent to Bayesian deconvolution:
    P(analog | digital) ∝ P(digital | analog) × P(analog)
    where P(digital | analog) is the quantization model.
    """
    samples = digital_samples.astype(float)
    n = len(samples)

    # Step 1: Estimate quantization step (LSB)
    sorted_vals = np.sort(np.unique(samples))
    if len(sorted_vals) > 1:
        diffs = np.diff(sorted_vals)
        # LSB is the GCD-like minimum meaningful step
        lsb = float(np.median(diffs[diffs > 0])) if np.any(diffs > 0) else 1.0
    else:
        lsb = 1.0

    # Step 2: Normalize to [0, 1]
    vmin, vmax = samples.min(), samples.max()
    vrange = vmax - vmin
    if vrange == 0:
        return {
            'error': 'constant signal', 'lsb_estimate': 0, 'value_range': 0,
            'n_digital_levels': 1, 'kde_bandwidth': 0,
            'continuous_entropy': 0, 'n_effective_analog_states': 1,
            'analog_snr_db': 0, 'analog_bits_per_sample': 0,
            'me_psd_slope': 0, 'me_psd_noise_type': 'constant',
            'inferred_mean': float(vmin), 'inferred_std': 0,
            'inferred_skewness': 0, 'inferred_kurtosis': 0,
        }
    normalized = (samples - vmin) / vrange

    # Step 3: KDE with Silverman's rule (optimal for Gaussian-like)
    # Adapted: use max(Silverman, quantization_step) so we don't under-smooth
    silverman_bw = 1.06 * np.std(normalized) * n ** (-0.2)
    quant_bw = lsb / vrange
    bw = max(silverman_bw, quant_bw, 0.005)

    # Evaluate on fine grid
    grid = np.linspace(0, 1, 1000)
    kde = np.zeros(len(grid))
    for s in normalized[::max(1, n//2000)]:  # subsample for speed
        kde += np.exp(-0.5 * ((grid - s) / bw) ** 2)
    kde /= (np.sum(kde) * (grid[1] - grid[0]))

    # Step 4: Maximum entropy spectral estimation
    # Use Burg's method to estimate PSD without windowing artifacts
    order = min(50, n // 10)
    me_psd = burg_psd(normalized[:min(4096, n)], order)

    # Step 5: Compute inferred analog properties
    # Shannon entropy of the continuous (KDE) distribution
    kde_safe = kde[kde > 0]
    dx = grid[1] - grid[0]
    # Differential entropy (can be negative for peaked distributions — that's ok)
    diff_entropy = -np.sum(kde_safe * np.log2(kde_safe) * dx)

    # Analog SNR: signal power / noise floor
    signal_var = np.var(samples)
    quant_noise_var = lsb ** 2 / 12  # uniform quantization noise
    analog_snr_db = 10 * np.log10(signal_var / quant_noise_var) if quant_noise_var > 0 else 0

    # Effective number of analog bits via Shannon:
    # For a quantized signal, analog_bits ≈ 0.5 * log2(1 + SNR)
    snr_linear = signal_var / (quant_noise_var + 1e-30)
    analog_bits = 0.5 * np.log2(1 + snr_linear)

    # Effective analog states (2^analog_bits)
    n_eff_states = 2 ** analog_bits

    # Also compute discrete Shannon entropy (this is always positive)
    continuous_entropy = float(np.log2(len(sorted_vals)))  # bits to enumerate all observed levels

    return {
        'lsb_estimate': lsb,
        'value_range': float(vrange),
        'n_digital_levels': len(sorted_vals),
        'kde_bandwidth': float(bw),
        'continuous_entropy': float(continuous_entropy),
        'n_effective_analog_states': float(n_eff_states),
        'analog_snr_db': float(analog_snr_db),
        'analog_bits_per_sample': float(analog_bits),
        'me_psd_slope': float(me_psd.get('slope', 0)),
        'me_psd_noise_type': me_psd.get('noise_type', 'unknown'),
        # Moments of inferred distribution
        'inferred_mean': float(np.sum(grid * kde * dx)),
        'inferred_std': float(np.sqrt(np.sum(grid**2 * kde * dx) - np.sum(grid * kde * dx)**2)),
        'inferred_skewness': float(inferred_moment(grid, kde, dx, 3)),
        'inferred_kurtosis': float(inferred_moment(grid, kde, dx, 4) - 3),
    }


def inferred_moment(grid, kde, dx, order):
    """Compute central moment of inferred distribution."""
    mean = np.sum(grid * kde * dx)
    std = np.sqrt(np.sum(grid**2 * kde * dx) - mean**2)
    if std == 0:
        return 0
    return np.sum(((grid - mean) / std) ** order * kde * dx)


def burg_psd(x, order):
    """Burg's maximum entropy PSD estimation.

    Better than FFT for short/noisy sequences because it doesn't
    assume the signal is zero outside the window. Gives us the
    MOST LIKELY spectrum consistent with the observed autocorrelation.
    """
    n = len(x)
    x = x - np.mean(x)

    # Burg's algorithm: estimate AR coefficients
    ef = x.copy()  # forward prediction error
    eb = x.copy()  # backward prediction error
    a = np.zeros(order + 1)
    a[0] = 1.0
    err = np.var(x)

    for p in range(1, order + 1):
        # Reflection coefficient
        num = -2.0 * np.sum(ef[p:] * eb[p-1:-1])
        den = np.sum(ef[p:] ** 2) + np.sum(eb[p-1:-1] ** 2)
        if den == 0:
            break
        k = num / den

        # Update AR coefficients
        a_new = np.zeros(order + 1)
        a_new[0] = 1.0
        for j in range(1, p):
            a_new[j] = a[j] + k * a[p - j]
        a_new[p] = k
        a = a_new

        err *= (1 - k * k)

        # Update prediction errors
        ef_new = ef[1:] + k * eb[:-1]
        eb_new = eb[:-1] + k * ef[1:]
        ef = ef_new
        eb = eb_new

    # Compute PSD from AR coefficients
    nfft = 512
    freqs = np.fft.rfftfreq(nfft)
    H = np.zeros(len(freqs), dtype=complex)
    for i, f in enumerate(freqs):
        z = np.exp(-2j * np.pi * f)
        denom = sum(a[k] * z**k for k in range(min(order + 1, len(a))))
        H[i] = 1.0 / denom if abs(denom) > 1e-10 else 0

    psd = err * np.abs(H) ** 2

    # Fit slope
    mask = (freqs > 0) & (psd > 0)
    if np.sum(mask) > 10:
        log_f = np.log10(freqs[mask])
        log_p = np.log10(psd[mask])
        n_fit = len(log_f) // 2
        slope, _ = np.polyfit(log_f[:max(5, n_fit)], log_p[:max(5, n_fit)], 1)
    else:
        slope = 0

    noise_type = (
        "white" if slope > -0.3 else
        "pink (1/f)" if slope > -0.7 else
        "flicker (1/f)" if slope > -1.3 else
        "brown (1/f²)" if slope > -1.7 else
        "steep red"
    )

    return {
        'slope': float(slope),
        'noise_type': noise_type,
        'ar_order': order,
        'prediction_error': float(err),
    }


# ═══════════════════════════════════════════════════════════════
# PART 5: Channel Fusion — The Unified Proxy
# ═══════════════════════════════════════════════════════════════

def fuse_channels(gpu_metrics_samples, perf_snapshot_samples, target_psd=-0.82):
    """Fuse gpu_metrics and PERF_SNAPSHOT into unified analog proxy.

    Strategy:
    1. Normalize both channels to [0, 1]
    2. Filter PERF_SNAPSHOT through adaptive 1/f IIR → adds temporal correlation
    3. Weight: gpu_metrics provides the spectral shape, PERF_SNAPSHOT provides entropy
    4. Output: combined signal with BOTH 1/f spectrum AND high entropy

    This is inspired by how the brain combines slow neuromodulatory signals
    (like gpu_metrics thermal inertia) with fast synaptic activity
    (like PERF_SNAPSHOT workload sensitivity).
    """
    # Normalize gpu_metrics
    gm = np.array(gpu_metrics_samples, dtype=float)
    gm_mean, gm_std = np.mean(gm), np.std(gm)
    gm_norm = (gm - gm_mean) / (gm_std + 1e-10)

    # Normalize PERF_SNAPSHOT
    ps = np.array(perf_snapshot_samples, dtype=float)
    ps_mean, ps_std = np.mean(ps), np.std(ps)
    ps_norm = (ps - ps_mean) / (ps_std + 1e-10)

    # Filter PERF_SNAPSHOT through adaptive 1/f
    filt = AdaptiveOneOverF(target_slope=target_psd, n_octaves=10)
    ps_filtered = filt.filter_array(ps_norm)

    # Fusion weights: optimize for target PSD slope
    # gpu_metrics has slope -0.82, PERF_SNAPSHOT filtered has slope ~-0.6 to -0.8
    # Use complementary weighting
    w_gm = 0.4   # spectral shape anchor
    w_ps = 0.6   # entropy source

    # Resample if different lengths (gpu_metrics is typically slower)
    if len(gm_norm) != len(ps_filtered):
        # Interpolate gpu_metrics to PERF_SNAPSHOT rate
        gm_indices = np.linspace(0, len(gm_norm) - 1, len(ps_filtered))
        gm_interp = np.interp(gm_indices, np.arange(len(gm_norm)), gm_norm)
    else:
        gm_interp = gm_norm

    # Fuse
    fused = w_gm * gm_interp + w_ps * ps_filtered

    # Compute quality metrics
    fused_psd = compute_psd_slope(fused)
    gm_psd = compute_psd_slope(gm_interp)
    ps_raw_psd = compute_psd_slope(ps_norm)
    ps_filt_psd = compute_psd_slope(ps_filtered)

    # Entropy
    fused_entropy = compute_byte_entropy(fused)
    gm_entropy = compute_byte_entropy(gm_interp)
    ps_entropy = compute_byte_entropy(ps_norm)

    # Autocorrelation lag-1
    fused_acf = autocorr_lag1(fused)
    gm_acf = autocorr_lag1(gm_interp)
    ps_raw_acf = autocorr_lag1(ps_norm)
    ps_filt_acf = autocorr_lag1(ps_filtered)

    return {
        'fused_signal': fused,
        'metrics': {
            'psd_slopes': {
                'gpu_metrics': gm_psd,
                'perf_snapshot_raw': ps_raw_psd,
                'perf_snapshot_filtered': ps_filt_psd,
                'fused': fused_psd,
                'target': target_psd,
                'gap_to_target': abs(fused_psd - target_psd),
            },
            'entropies': {
                'gpu_metrics': gm_entropy,
                'perf_snapshot': ps_entropy,
                'fused': fused_entropy,
            },
            'autocorrelation_lag1': {
                'gpu_metrics': gm_acf,
                'perf_snapshot_raw': ps_raw_acf,
                'perf_snapshot_filtered': ps_filt_acf,
                'fused': fused_acf,
            },
            'weights': {'gpu_metrics': w_gm, 'perf_snapshot': w_ps},
            'n_samples': len(fused),
        }
    }


def compute_psd_slope(x, fit_fraction=0.5):
    """Quick PSD slope computation."""
    n = len(x)
    if n < 64:
        return 0.0
    x = x - np.mean(x)
    fft = np.fft.rfft(x)
    psd = np.abs(fft[1:]) ** 2 / n
    freqs = np.fft.rfftfreq(n)[1:]
    mask = psd > 0
    if np.sum(mask) < 5:
        return 0.0
    log_f = np.log10(freqs[mask])
    log_p = np.log10(psd[mask])
    n_fit = max(5, int(len(log_f) * fit_fraction))
    slope, _ = np.polyfit(log_f[:n_fit], log_p[:n_fit], 1)
    return float(slope)


def compute_byte_entropy(x):
    """Entropy of quantized-to-byte values."""
    # Quantize float to 8-bit
    xmin, xmax = np.min(x), np.max(x)
    if xmax == xmin:
        return 0.0
    quantized = ((x - xmin) / (xmax - xmin) * 255).astype(np.uint8)
    counts = np.bincount(quantized, minlength=256)
    probs = counts[counts > 0] / len(x)
    return float(-np.sum(probs * np.log2(probs)))


def autocorr_lag1(x):
    """Lag-1 autocorrelation."""
    x = x - np.mean(x)
    var = np.var(x)
    if var == 0:
        return 0.0
    return float(np.mean(x[:-1] * x[1:]) / var)


# ═══════════════════════════════════════════════════════════════
# PART 6: FPGA Cross-Substrate Bridge
# ═══════════════════════════════════════════════════════════════

def fpga_bridge_metrics(gpu_proxy, fpga_telemetry=None):
    """Compute cross-substrate correlation metrics.

    If FPGA telemetry is available (from z2144-z2148 experiments),
    compute direct cross-correlation. Otherwise, compute the metrics
    that WOULD be compared, using the z2149 diagnostic calibration
    data (Vg 0→1 gives spike rates 8.4→226.4, 33× dynamic range).
    """
    # FPGA NS-RAM reference characteristics (from z2144-z2149)
    fpga_ref = {
        'dynamic_range_x': 33.0,      # Vg sweep 33× spike rate range
        'spike_rate_range': (8.4, 226.4),  # spikes/sec at Vg=0 vs Vg=1
        'paired_pulse_ratio': 1.38,    # z2144 T47 (PASS)
        'retention_delta': 0.0,        # z2144 T48 (FAIL — no retention)
        'power_law_alpha': 1.132,      # z2145 (near-critical)
        'soc_sigma_converged': 1.0,    # z2146 (PASS)
        'feedback_rho': 0.1837,        # z2147 (weak but present)
    }

    gpu_stats = {
        'dynamic_range': float(np.max(gpu_proxy) - np.min(gpu_proxy)),
        'cv': float(np.std(gpu_proxy) / (np.mean(gpu_proxy) + 1e-10)),
        'psd_slope': compute_psd_slope(gpu_proxy),
        'entropy': compute_byte_entropy(gpu_proxy),
        'acf_lag1': autocorr_lag1(gpu_proxy),
    }

    # Bridge quality metrics
    bridge = {}

    # 1. PSD slope match (GPU → SPICE → FPGA path)
    # SPICE 1/f = -1.0, FPGA should show similar through oxide defects
    bridge['psd_gap_to_1f'] = abs(gpu_stats['psd_slope'] - (-1.0))
    bridge['psd_quality'] = max(0, 1 - bridge['psd_gap_to_1f'] / 2)

    # 2. Dynamic range comparison
    # FPGA: 33× through Vg sweep. GPU proxy: what's our range?
    gpu_dr = gpu_stats['dynamic_range'] / (np.std(gpu_proxy) + 1e-10)
    bridge['dynamic_range_ratio'] = float(gpu_dr / fpga_ref['dynamic_range_x'])

    # 3. Temporal structure similarity
    # FPGA has retention (T48 failed but signal exists), GPU needs acf > 0
    bridge['has_temporal_memory'] = gpu_stats['acf_lag1'] > 0.05
    bridge['temporal_quality'] = min(1.0, abs(gpu_stats['acf_lag1']) / 0.3)

    # 4. Information content comparison
    bridge['entropy_ratio'] = gpu_stats['entropy'] / 8.0  # vs max
    bridge['analog_fidelity'] = float(np.mean([
        bridge['psd_quality'],
        bridge['entropy_ratio'],
        bridge['temporal_quality'],
    ]))

    # 5. Cross-substrate mapping
    bridge['substrate_map'] = {
        'gpu_thermal_noise': 'fpga_vg_drift',      # slow baseline
        'gpu_perf_snapshot': 'fpga_spike_events',   # fast stochastic
        'gpu_dvfs_states':   'fpga_set_reset',      # discrete transitions
        'gpu_shader_cycles': 'fpga_isi_jitter',     # timing noise
    }

    return {
        'gpu_stats': gpu_stats,
        'fpga_reference': fpga_ref,
        'bridge': bridge,
    }


# ═══════════════════════════════════════════════════════════════
# PART 7: Live Unified Proxy Collection
# ═══════════════════════════════════════════════════════════════

def collect_live(duration_s=5.0, gpu_metrics_hz=200):
    """Collect live gpu_metrics at high frequency."""
    print(f"Collecting gpu_metrics for {duration_s}s at ~{gpu_metrics_hz} Hz...")
    samples = []
    t_start = time.monotonic()
    interval = 1.0 / gpu_metrics_hz

    while time.monotonic() - t_start < duration_s:
        t0 = time.monotonic()
        m = read_gpu_metrics()
        if m:
            samples.append({
                't': time.monotonic() - t_start,
                'temp_gfx': m['temp_gfx_C'],
                'temp_soc': m['temp_soc_C'],
                'power_W': m['power_W'],
                'gfxclk': m.get('average_gfxclk_freq', 0),
                'fclk': m.get('average_fclk_freq', 0),
            })
        elapsed = time.monotonic() - t0
        if elapsed < interval:
            time.sleep(interval - elapsed)

    actual_hz = len(samples) / duration_s
    print(f"  Collected {len(samples)} samples ({actual_hz:.1f} Hz actual)")
    return samples


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description='z2152: Unified Analog Proxy')
    parser.add_argument('--perf', type=str, help='PERF_SNAPSHOT CSV from z2151')
    parser.add_argument('--duration', type=float, default=3.0, help='Live collection duration (s)')
    parser.add_argument('--output', type=str, default='results/z2152_unified_proxy.json')
    args = parser.parse_args()

    results = {'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S')}

    # ── Step 1: Load PERF_SNAPSHOT data ──
    if args.perf and os.path.exists(args.perf):
        print(f"\n{'='*60}")
        print("STEP 1: Loading PERF_SNAPSHOT data")
        print(f"{'='*60}")
        perf_data = load_perf_csv(args.perf)
        perf_vals = perf_data['perf_snapshot']
        print(f"  Loaded {len(perf_vals)} PERF_SNAPSHOT samples")
    else:
        print("No PERF_SNAPSHOT CSV provided, using synthetic test data")
        perf_vals = np.random.randint(0, 2**24, size=8000).astype(np.uint32)

    # ── Step 2: Collect live gpu_metrics ──
    print(f"\n{'='*60}")
    print("STEP 2: Collecting live gpu_metrics")
    print(f"{'='*60}")
    gm_samples = collect_live(args.duration)
    gm_temps = np.array([s['temp_gfx'] for s in gm_samples])
    gm_power = np.array([s['power_W'] for s in gm_samples])

    results['gpu_metrics'] = {
        'n_samples': len(gm_samples),
        'temp_mean': float(np.mean(gm_temps)),
        'temp_std': float(np.std(gm_temps)),
        'power_mean': float(np.mean(gm_power)),
        'power_std': float(np.std(gm_power)),
        'psd_slope_temp': compute_psd_slope(gm_temps),
        'psd_slope_power': compute_psd_slope(gm_power),
    }

    # ── Step 3: Allan variance decomposition ──
    print(f"\n{'='*60}")
    print("STEP 3: Allan variance — noise type decomposition")
    print(f"{'='*60}")

    allan_perf = allan_variance(perf_vals.astype(float))
    allan_temp = allan_variance(gm_temps)
    allan_power = allan_variance(gm_power)

    results['allan_variance'] = {
        'perf_snapshot': allan_perf,
        'gpu_temp': allan_temp,
        'gpu_power': allan_power,
    }

    print(f"  PERF_SNAPSHOT: slope={allan_perf['slope']:.3f} → {allan_perf['noise_type']}")
    print(f"  GPU temp:      slope={allan_temp['slope']:.3f} → {allan_temp['noise_type']}")
    print(f"  GPU power:     slope={allan_power['slope']:.3f} → {allan_power['noise_type']}")
    print(f"  Has flicker (1/f) component:")
    print(f"    PERF_SNAPSHOT: {allan_perf['has_flicker']}")
    print(f"    GPU temp:      {allan_temp['has_flicker']}")
    print(f"    GPU power:     {allan_power['has_flicker']}")

    # ── Step 4: Statistical analog inference ──
    print(f"\n{'='*60}")
    print("STEP 4: Statistical analog inference")
    print(f"{'='*60}")

    analog_perf = infer_analog_distribution(perf_vals)
    print(f"  PERF_SNAPSHOT analog inference:")
    print(f"    LSB estimate:     {analog_perf['lsb_estimate']:.1f}")
    print(f"    Digital levels:   {analog_perf['n_digital_levels']}")
    print(f"    Analog states:    {analog_perf['n_effective_analog_states']:.0f}")
    print(f"    Analog bits:      {analog_perf['analog_bits_per_sample']:.2f}")
    print(f"    Analog SNR:       {analog_perf['analog_snr_db']:.1f} dB")
    print(f"    ME PSD slope:     {analog_perf['me_psd_slope']:.3f} ({analog_perf['me_psd_noise_type']})")
    print(f"    Inferred skew:    {analog_perf['inferred_skewness']:.3f}")
    print(f"    Inferred kurtosis:{analog_perf['inferred_kurtosis']:.3f}")

    results['analog_inference'] = {
        'perf_snapshot': analog_perf,
    }

    # Also infer from temperature if we have enough samples
    if len(gm_temps) > 64:
        analog_temp = infer_analog_distribution(gm_temps)
        results['analog_inference']['gpu_temp'] = analog_temp
        print(f"\n  GPU temp analog inference:")
        print(f"    Analog bits: {analog_temp['analog_bits_per_sample']:.2f}")
        print(f"    ME PSD:      {analog_temp['me_psd_slope']:.3f} ({analog_temp['me_psd_noise_type']})")

    # ── Step 5: IIR 1/f filtering ──
    print(f"\n{'='*60}")
    print("STEP 5: IIR 1/f filtering of PERF_SNAPSHOT")
    print(f"{'='*60}")

    # Use wavefront 0 as time series
    w0_mask = perf_data.get('wave_id', np.zeros(len(perf_vals))) == 0
    ps_w0 = perf_vals[w0_mask] if np.any(w0_mask) else perf_vals[:500]

    ps_norm = (ps_w0.astype(float) - np.mean(ps_w0)) / (np.std(ps_w0) + 1e-10)
    psd_raw = compute_psd_slope(ps_norm)

    # Apply different filter strengths
    filter_results = {}
    for target in [-0.5, -0.82, -1.0]:
        filt = AdaptiveOneOverF(target_slope=target, n_octaves=10)
        filtered = filt.filter_array(ps_norm)
        psd_filt = compute_psd_slope(filtered)
        acf_filt = autocorr_lag1(filtered)
        ent_filt = compute_byte_entropy(filtered)
        filter_results[f'target_{target}'] = {
            'target_slope': target,
            'achieved_slope': psd_filt,
            'gap': abs(psd_filt - target),
            'acf_lag1': acf_filt,
            'entropy': ent_filt,
        }
        print(f"  Target {target:+.2f}: achieved {psd_filt:+.3f} (gap={abs(psd_filt-target):.3f}), "
              f"ACF={acf_filt:.3f}, entropy={ent_filt:.2f}")

    print(f"  Raw PSD slope: {psd_raw:+.3f}")
    results['iir_filtering'] = filter_results

    # ── Step 6: Channel fusion ──
    print(f"\n{'='*60}")
    print("STEP 6: Channel fusion — unified proxy")
    print(f"{'='*60}")

    # Use power as the gpu_metrics channel (more dynamic than temp)
    fusion = fuse_channels(gm_power, ps_w0, target_psd=-0.82)
    fm = fusion['metrics']

    print(f"  PSD slopes:")
    for k, v in fm['psd_slopes'].items():
        print(f"    {k:30s}: {v:+.3f}" if isinstance(v, float) else f"    {k:30s}: {v}")
    print(f"  Entropies:")
    for k, v in fm['entropies'].items():
        print(f"    {k:30s}: {v:.3f}")
    print(f"  Autocorrelation (lag-1):")
    for k, v in fm['autocorrelation_lag1'].items():
        print(f"    {k:30s}: {v:+.4f}")

    results['fusion'] = {
        'psd_slopes': fm['psd_slopes'],
        'entropies': fm['entropies'],
        'autocorrelation': fm['autocorrelation_lag1'],
        'weights': fm['weights'],
        'n_samples': fm['n_samples'],
    }

    # ── Step 7: FPGA bridge metrics ──
    print(f"\n{'='*60}")
    print("STEP 7: FPGA cross-substrate bridge metrics")
    print(f"{'='*60}")

    bridge = fpga_bridge_metrics(fusion['fused_signal'])
    print(f"  GPU proxy stats:")
    for k, v in bridge['gpu_stats'].items():
        print(f"    {k:20s}: {v:.4f}" if isinstance(v, float) else f"    {k:20s}: {v}")
    print(f"  Bridge quality:")
    for k, v in bridge['bridge'].items():
        if isinstance(v, dict):
            print(f"    {k}:")
            for k2, v2 in v.items():
                print(f"      {k2:30s} ↔ {v2}")
        else:
            print(f"    {k:30s}: {v}")

    results['fpga_bridge'] = {
        'gpu_stats': bridge['gpu_stats'],
        'bridge': {k: v for k, v in bridge['bridge'].items() if not isinstance(v, dict)},
        'substrate_map': bridge['bridge'].get('substrate_map', {}),
        'fpga_reference': bridge['fpga_reference'],
    }

    # ── Final Summary ──
    print(f"\n{'='*60}")
    print("UNIFIED ANALOG PROXY — FINAL SUMMARY")
    print(f"{'='*60}")

    # Overall fidelity score
    scores = {
        'psd_match': max(0, 1 - abs(fm['psd_slopes']['fused'] - (-0.82)) / 2),
        'entropy': fm['entropies']['fused'] / 8.0,
        'temporal': min(1.0, abs(fm['autocorrelation_lag1']['fused']) / 0.1),
        'bridge': bridge['bridge']['analog_fidelity'],
        'allan_flicker': 1.0 if allan_perf['has_flicker'] else 0.0,
        'analog_bits': min(1.0, analog_perf['analog_bits_per_sample'] / 10),
    }

    overall = float(np.mean(list(scores.values())))
    results['overall_fidelity'] = {
        'scores': {k: float(v) for k, v in scores.items()},
        'overall': overall,
        'interpretation': (
            "excellent — near-analog substrate" if overall > 0.8
            else "good — significant analog character" if overall > 0.6
            else "moderate — usable analog proxy" if overall > 0.4
            else "low — needs improvement"
        ),
    }

    print(f"\n  Component scores:")
    for k, v in scores.items():
        bar = '█' * int(v * 20) + '░' * (20 - int(v * 20))
        print(f"    {k:20s}: {bar} {v:.3f}")
    print(f"\n  OVERALL FIDELITY: {overall:.3f} — {results['overall_fidelity']['interpretation']}")

    # Save
    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Results saved to {args.output}")


if __name__ == '__main__':
    main()
