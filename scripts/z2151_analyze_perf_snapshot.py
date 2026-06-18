#!/usr/bin/env python3
"""z2151: Statistical analysis of PERF_SNAPSHOT samples from GPU silicon.

Reads CSV from z2151_perf_snapshot_stats, computes:
1. Distribution (histogram, normality test, heavy tails)
2. Power spectral density (PSD slope for 1/f noise comparison)
3. Entropy (Shannon, sample entropy)
4. Autocorrelation (temporal structure)
5. Per-wavefront vs cross-wavefront variation
6. SPICE comparison metrics

Usage:
    python z2151_analyze_perf_snapshot.py results/z2151_perf_raw.csv
"""

import sys, json, os
import numpy as np
from collections import defaultdict

def load_csv(path):
    """Load CSV, return dict of arrays."""
    data = defaultdict(list)
    with open(path) as f:
        header = f.readline().strip().split(',')
        for line in f:
            parts = line.strip().split(',')
            if len(parts) != len(header):
                continue
            for h, v in zip(header, parts):
                if h in ('hw_id1', 'status'):
                    data[h].append(int(v, 16) if v.startswith('0x') else int(v))
                else:
                    data[h].append(int(v))
    return {k: np.array(v) for k, v in data.items()}

def compute_distribution(values, name="perf_snapshot"):
    """Histogram, moments, normality."""
    n = len(values)
    mean = np.mean(values)
    std = np.std(values)
    median = np.median(values)
    skew = float(np.mean(((values - mean) / std) ** 3)) if std > 0 else 0
    kurtosis = float(np.mean(((values - mean) / std) ** 4) - 3) if std > 0 else 0

    # Unique values
    unique = len(np.unique(values))

    # Range
    vmin, vmax = int(np.min(values)), int(np.max(values))

    # IQR
    q25, q75 = np.percentile(values, [25, 75])
    iqr = q75 - q25

    # Log-scale histogram for heavy tail detection
    log_vals = np.log1p(values.astype(float))
    log_mean = float(np.mean(log_vals))
    log_std = float(np.std(log_vals))

    return {
        "name": name,
        "n": n,
        "unique": unique,
        "unique_ratio": unique / n,
        "mean": float(mean),
        "std": float(std),
        "cv": float(std / mean) if mean > 0 else 0,
        "median": float(median),
        "min": vmin,
        "max": vmax,
        "range": vmax - vmin,
        "iqr": float(iqr),
        "skewness": skew,
        "excess_kurtosis": kurtosis,
        "log_mean": log_mean,
        "log_std": log_std,
        "is_heavy_tailed": kurtosis > 1.0,
        "is_symmetric": abs(skew) < 0.5,
    }

def compute_psd(values, name="perf_snapshot"):
    """Power spectral density via FFT, compute slope in log-log space."""
    n = len(values)
    if n < 64:
        return {"name": name, "error": "too few samples for PSD"}

    # Detrend
    x = values.astype(float)
    x = x - np.mean(x)

    # FFT
    fft = np.fft.rfft(x)
    psd = np.abs(fft) ** 2 / n
    freqs = np.fft.rfftfreq(n)

    # Skip DC component
    freqs = freqs[1:]
    psd = psd[1:]

    # Remove zeros
    mask = psd > 0
    freqs = freqs[mask]
    psd = psd[mask]

    if len(freqs) < 10:
        return {"name": name, "error": "too few valid PSD points"}

    # Log-log fit for slope
    log_f = np.log10(freqs)
    log_p = np.log10(psd)

    # Fit in lower half of spectrum (most relevant for 1/f)
    n_fit = len(log_f) // 2
    if n_fit < 5:
        n_fit = len(log_f)

    coeffs = np.polyfit(log_f[:n_fit], log_p[:n_fit], 1)
    slope = float(coeffs[0])
    intercept = float(coeffs[1])

    # R² of fit
    predicted = np.polyval(coeffs, log_f[:n_fit])
    ss_res = np.sum((log_p[:n_fit] - predicted) ** 2)
    ss_tot = np.sum((log_p[:n_fit] - np.mean(log_p[:n_fit])) ** 2)
    r_squared = float(1 - ss_res / ss_tot) if ss_tot > 0 else 0

    return {
        "name": name,
        "psd_slope": slope,
        "psd_intercept": intercept,
        "psd_r_squared": r_squared,
        "psd_n_points": len(freqs),
        "psd_n_fit": n_fit,
        "noise_type": classify_noise(slope),
        "spice_1f_match": abs(slope - (-1.0)),
        "gpu_metrics_match": abs(slope - (-0.82)),
    }

def classify_noise(slope):
    """Classify noise type from PSD slope."""
    if slope > -0.3:
        return "white (flat)"
    elif slope > -0.7:
        return "pink-ish (weak 1/f)"
    elif slope > -1.3:
        return "1/f (flicker)"
    elif slope > -1.7:
        return "brown-ish (1/f²)"
    else:
        return "brown/red (1/f² or steeper)"

def compute_entropy(values):
    """Shannon entropy and sample entropy."""
    # Shannon entropy of binned distribution
    n = len(values)
    unique, counts = np.unique(values, return_counts=True)
    probs = counts / n
    shannon = float(-np.sum(probs * np.log2(probs)))
    max_entropy = np.log2(n)  # max if all unique

    # Byte-level entropy (treat 32-bit values as 4 bytes)
    byte_arr = values.astype(np.uint32).tobytes()
    byte_counts = np.bincount(np.frombuffer(byte_arr, dtype=np.uint8), minlength=256)
    byte_probs = byte_counts / len(byte_arr)
    byte_probs = byte_probs[byte_probs > 0]
    byte_entropy = float(-np.sum(byte_probs * np.log2(byte_probs)))

    return {
        "shannon_entropy": shannon,
        "max_entropy": float(max_entropy),
        "entropy_ratio": shannon / max_entropy if max_entropy > 0 else 0,
        "byte_entropy": byte_entropy,
        "byte_entropy_max": 8.0,
        "byte_entropy_ratio": byte_entropy / 8.0,
    }

def compute_autocorrelation(values, max_lag=50):
    """Autocorrelation function for temporal structure detection."""
    n = len(values)
    x = values.astype(float)
    x = x - np.mean(x)
    var = np.var(x)
    if var == 0:
        return {"error": "zero variance"}

    lags = list(range(1, min(max_lag + 1, n // 4)))
    acf = []
    for lag in lags:
        c = np.mean(x[:n-lag] * x[lag:]) / var
        acf.append(float(c))

    # Find decay constant (lag where ACF drops below 1/e)
    decay_lag = None
    for i, c in enumerate(acf):
        if c < 1/np.e:
            decay_lag = lags[i]
            break

    return {
        "lags": lags[:20],
        "acf": acf[:20],
        "decay_lag": decay_lag,
        "acf_lag1": acf[0] if acf else None,
        "has_temporal_structure": acf[0] > 0.1 if acf else False,
    }

def per_wavefront_analysis(data):
    """Analyze PERF_SNAPSHOT variation within vs across wavefronts."""
    iters = data['iteration']
    waves = data['wave_id']
    perfs = data['perf_snapshot']

    unique_waves = np.unique(waves)
    unique_iters = np.unique(iters)

    # Per-wavefront stats
    wave_stats = {}
    for w in unique_waves:
        mask = waves == w
        vals = perfs[mask]
        wave_stats[int(w)] = {
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals)),
            "cv": float(np.std(vals) / np.mean(vals)) if np.mean(vals) > 0 else 0,
            "n": int(len(vals)),
        }

    # Cross-wavefront variation per iteration
    cross_wave_cvs = []
    for it in unique_iters[:100]:  # first 100 iterations
        mask = iters == it
        vals = perfs[mask]
        if len(vals) > 1 and np.mean(vals) > 0:
            cross_wave_cvs.append(float(np.std(vals) / np.mean(vals)))

    # Within-wavefront temporal variation
    temporal_cvs = []
    for w in unique_waves:
        mask = waves == w
        vals = perfs[mask]
        if len(vals) > 1 and np.mean(vals) > 0:
            temporal_cvs.append(float(np.std(vals) / np.mean(vals)))

    return {
        "n_wavefronts": int(len(unique_waves)),
        "n_iterations": int(len(unique_iters)),
        "per_wavefront": wave_stats,
        "cross_wavefront_cv_mean": float(np.mean(cross_wave_cvs)) if cross_wave_cvs else 0,
        "cross_wavefront_cv_std": float(np.std(cross_wave_cvs)) if cross_wave_cvs else 0,
        "temporal_cv_mean": float(np.mean(temporal_cvs)) if temporal_cvs else 0,
        "temporal_cv_std": float(np.std(temporal_cvs)) if temporal_cvs else 0,
        "temporal_dominates": (
            float(np.mean(temporal_cvs)) > float(np.mean(cross_wave_cvs))
            if temporal_cvs and cross_wave_cvs else None
        ),
    }

def shader_cycles_analysis(data):
    """Analyze SHADER_CYCLES jitter across wavefronts."""
    cycles = data['shader_cycles']
    iters = data['iteration']
    waves = data['wave_id']

    unique_iters = np.unique(iters)

    # Per-iteration spread
    spreads = []
    for it in unique_iters:
        mask = iters == it
        vals = cycles[mask]
        if len(vals) > 1:
            spreads.append(int(np.max(vals) - np.min(vals)))

    return {
        "mean_spread": float(np.mean(spreads)) if spreads else 0,
        "std_spread": float(np.std(spreads)) if spreads else 0,
        "min_spread": int(np.min(spreads)) if spreads else 0,
        "max_spread": int(np.max(spreads)) if spreads else 0,
        "global_mean": float(np.mean(cycles)),
        "global_std": float(np.std(cycles)),
    }

def spice_comparison(dist, psd, entropy):
    """Compare GPU silicon metrics to SPICE/analog expectations."""
    comparisons = {}

    # 1. PSD slope: SPICE 1/f noise = -1.0
    if 'psd_slope' in psd:
        comparisons["psd_slope_gpu"] = psd['psd_slope']
        comparisons["psd_slope_spice_1f"] = -1.0
        comparisons["psd_slope_gpu_metrics"] = -0.82
        comparisons["psd_gap_to_spice"] = abs(psd['psd_slope'] - (-1.0))
        comparisons["psd_gap_to_gpu_metrics"] = abs(psd['psd_slope'] - (-0.82))

    # 2. Entropy: ideal TRNG = 8.0 bits/byte, SPICE thermal noise ≈ 7.5-7.9
    comparisons["byte_entropy_gpu"] = entropy.get('byte_entropy', 0)
    comparisons["byte_entropy_trng"] = 8.0
    comparisons["byte_entropy_spice_thermal"] = 7.8  # typical
    comparisons["entropy_quality"] = (
        "excellent" if entropy.get('byte_entropy', 0) > 7.5
        else "good" if entropy.get('byte_entropy', 0) > 6.0
        else "moderate" if entropy.get('byte_entropy', 0) > 4.0
        else "poor"
    )

    # 3. Distribution: SPICE thermal noise is Gaussian, shot noise is Poisson
    comparisons["is_gaussian"] = dist.get('is_symmetric', False) and abs(dist.get('excess_kurtosis', 0)) < 1.0
    comparisons["is_heavy_tailed"] = dist.get('is_heavy_tailed', False)
    comparisons["likely_distribution"] = (
        "Gaussian (thermal-like)" if comparisons["is_gaussian"]
        else "heavy-tailed (shot/avalanche-like)" if comparisons["is_heavy_tailed"]
        else "non-Gaussian (mixed)"
    )

    # 4. CV comparison: SMU power CV = 0.083, SPICE leakage CV ≈ 0.1-0.3
    comparisons["cv_gpu_perf_snapshot"] = dist.get('cv', 0)
    comparisons["cv_gpu_metrics_power"] = 0.083  # from Phase 1
    comparisons["cv_spice_leakage_typical"] = 0.15

    # 5. Overall silicon-physics fidelity score (0-1)
    scores = []
    if 'psd_slope' in psd:
        # How close to 1/f: score = 1 - |slope - (-1)| / 2
        scores.append(max(0, 1 - abs(psd['psd_slope'] - (-1.0)) / 2))
    if entropy.get('byte_entropy', 0) > 0:
        scores.append(entropy['byte_entropy'] / 8.0)
    if dist.get('unique_ratio', 0) > 0:
        scores.append(min(1.0, dist['unique_ratio']))

    comparisons["silicon_physics_fidelity"] = float(np.mean(scores)) if scores else 0
    comparisons["fidelity_interpretation"] = (
        "excellent — near-analog physics" if comparisons["silicon_physics_fidelity"] > 0.8
        else "good — significant analog content" if comparisons["silicon_physics_fidelity"] > 0.6
        else "moderate — digital + analog mix" if comparisons["silicon_physics_fidelity"] > 0.4
        else "low — mostly digital artifacts"
    )

    return comparisons

def hw_topology_analysis(data):
    """Decode HW_ID1 to understand physical wavefront distribution."""
    hw_id1s = data['hw_id1']
    unique = np.unique(hw_id1s)

    topology = []
    for h in unique:
        wave_id = (h >> 0) & 0xF
        simd_id = (h >> 4) & 0x3
        wgp_id  = (h >> 6) & 0xF
        sa_id   = (h >> 10) & 0x7
        se_id   = (h >> 13) & 0x7
        topology.append({
            "hw_id1": f"0x{h:08X}",
            "SE": int(se_id), "SA": int(sa_id), "WGP": int(wgp_id),
            "SIMD": int(simd_id), "WAVE": int(wave_id),
        })

    # Count unique locations
    ses = set(t['SE'] for t in topology)
    sas = set((t['SE'], t['SA']) for t in topology)
    wgps = set((t['SE'], t['SA'], t['WGP']) for t in topology)

    return {
        "unique_hw_id1": len(unique),
        "unique_SEs": len(ses),
        "unique_SAs": len(sas),
        "unique_WGPs": len(wgps),
        "topology": topology[:32],  # first 32
    }

def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <csv_file>")
        sys.exit(1)

    csv_path = sys.argv[1]
    print(f"Loading {csv_path}...")
    data = load_csv(csv_path)
    print(f"  Loaded {len(data['perf_snapshot'])} samples")

    perfs = data['perf_snapshot']
    cycles = data['shader_cycles']

    results = {}

    # 1. Distribution analysis
    print("Computing distribution...")
    results['distribution'] = compute_distribution(perfs, "perf_snapshot")
    results['distribution_cycles'] = compute_distribution(cycles, "shader_cycles")

    # 2. PSD (use sequential samples — concatenate all wavefront 0 readings)
    print("Computing PSD...")
    # Use wavefront 0 as time series
    w0_mask = data['wave_id'] == 0
    w0_perfs = perfs[w0_mask]
    results['psd_wavefront0'] = compute_psd(w0_perfs, "perf_snapshot_w0")

    # Also PSD of all samples in order
    results['psd_all'] = compute_psd(perfs, "perf_snapshot_all")

    # 3. Entropy
    print("Computing entropy...")
    results['entropy'] = compute_entropy(perfs)
    results['entropy_cycles'] = compute_entropy(cycles)

    # 4. Autocorrelation
    print("Computing autocorrelation...")
    results['autocorrelation'] = compute_autocorrelation(w0_perfs)

    # 5. Per-wavefront analysis
    print("Computing per-wavefront analysis...")
    results['wavefront_analysis'] = per_wavefront_analysis(data)

    # 6. Shader cycles jitter
    print("Computing shader cycles analysis...")
    results['shader_cycles'] = shader_cycles_analysis(data)

    # 7. HW topology
    print("Computing topology analysis...")
    results['topology'] = hw_topology_analysis(data)

    # 8. SPICE comparison
    print("Computing SPICE comparison...")
    results['spice_comparison'] = spice_comparison(
        results['distribution'], results['psd_wavefront0'], results['entropy']
    )

    # Output
    out_path = csv_path.replace('.csv', '.json')
    if out_path == csv_path:
        out_path = csv_path + '.analysis.json'

    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")

    # Print summary
    print("\n" + "=" * 60)
    print("PERF_SNAPSHOT STATISTICAL SUMMARY")
    print("=" * 60)

    d = results['distribution']
    print(f"\nDistribution ({d['n']} samples):")
    print(f"  Mean: {d['mean']:.1f}")
    print(f"  Std:  {d['std']:.1f}")
    print(f"  CV:   {d['cv']:.4f}")
    print(f"  Range: [{d['min']}, {d['max']}]")
    print(f"  Unique: {d['unique']} / {d['n']} ({d['unique_ratio']:.4f})")
    print(f"  Skewness: {d['skewness']:.3f}")
    print(f"  Excess Kurtosis: {d['excess_kurtosis']:.3f}")
    print(f"  Heavy-tailed: {d['is_heavy_tailed']}")

    p = results['psd_wavefront0']
    if 'psd_slope' in p:
        print(f"\nPSD (wavefront 0 time series):")
        print(f"  Slope: {p['psd_slope']:.3f}")
        print(f"  R²: {p['psd_r_squared']:.3f}")
        print(f"  Noise type: {p['noise_type']}")
        print(f"  Gap to SPICE 1/f: {p['spice_1f_match']:.3f}")
        print(f"  Gap to gpu_metrics: {p['gpu_metrics_match']:.3f}")

    e = results['entropy']
    print(f"\nEntropy:")
    print(f"  Shannon: {e['shannon_entropy']:.2f} bits")
    print(f"  Byte entropy: {e['byte_entropy']:.3f} / 8.0")
    print(f"  Entropy ratio: {e['entropy_ratio']:.4f}")

    sc = results['spice_comparison']
    print(f"\nSPICE/Analog Comparison:")
    print(f"  Silicon physics fidelity: {sc['silicon_physics_fidelity']:.3f}")
    print(f"  Interpretation: {sc['fidelity_interpretation']}")
    print(f"  Likely distribution: {sc['likely_distribution']}")

    wa = results['wavefront_analysis']
    print(f"\nWavefront Analysis:")
    print(f"  Cross-wavefront CV: {wa['cross_wavefront_cv_mean']:.4f}")
    print(f"  Temporal CV: {wa['temporal_cv_mean']:.4f}")
    print(f"  Temporal dominates: {wa['temporal_dominates']}")

    ac = results['autocorrelation']
    print(f"\nAutocorrelation:")
    print(f"  Lag-1 ACF: {ac.get('acf_lag1', 'N/A')}")
    print(f"  Decay lag: {ac.get('decay_lag', 'N/A')}")
    print(f"  Has temporal structure: {ac.get('has_temporal_structure', 'N/A')}")

if __name__ == "__main__":
    main()
