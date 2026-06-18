#!/usr/bin/env python3
"""z2153_bridge_analysis.py — Deep probe + bridge analysis

Part 1: Analyze ALL hwreg signals from z2153 deep probe (new registers)
Part 2: Compute stochastic jitter statistics for FPGA ISI injection
Part 3: Sample amdgpu_pm_info + fence_info for host-side signals
Part 4: Fuse all channels into enhanced bridge metrics
Part 5: Compare against Mario Lanza NS-RAM reference values
"""

import json
import csv
import sys
import os
import re
import subprocess
import time
import struct
import numpy as np
from pathlib import Path
from collections import Counter, defaultdict

# ─── Paths ───
PROJ = Path("/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy")
RESULTS = PROJ / "results"
GPU_METRICS_PATH = "/sys/class/drm/card0/device/gpu_metrics"
PM_INFO_PATH = "/sys/kernel/debug/dri/0/amdgpu_pm_info"
FENCE_INFO_PATH = "/sys/kernel/debug/dri/0/amdgpu_fence_info"

# ─── Mario Lanza NS-RAM reference constants ───
LANZA_REF = {
    "psd_slope": -1.0,         # Pure 1/f from avalanche process
    "entropy_bits": 7.5,       # Near-maximum from quantum stochastic
    "isi_cv_range": (0.3, 2.0),  # Stochastic switching
    "avalanche_alpha": 1.5,    # Beggs-Plenz cortical reference
    "retention_hours": 24,     # Non-volatile state
    "energy_per_spike_fj": (0.2, 21.0),  # Pazos et al. range
    "bv0": 3.5,                # Barrier voltage at Vg=0
    "alpha_t": 21.3e-6,        # Thermal coefficient V/K
    "dynamic_range_x": 100,    # Vg 0->1 conductance ratio
    "temperature_bv_coefficient": -21.3e-6,  # V/K
}


def parse_deep_probe_csv(csv_path):
    """Parse z2153 deep probe CSV."""
    data = defaultdict(list)
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            for col in ['perf_snapshot', 'shader_cycles', 'work_result', 'jitter_byte']:
                if col in row:
                    data[col].append(int(row[col]))
            for col in ['hw_id1', 'status', 'ib_sts2', 'mem_bases', 'gpr_alloc', 'ib_sts', 'mode']:
                if col in row:
                    data[col].append(int(row[col], 16) if row[col].startswith('0x') else int(row[col]))
            data['iteration'].append(int(row['iteration']))
            data['wave_id'].append(int(row['wave_id']))
    return {k: np.array(v) for k, v in data.items()}


def analyze_register(name, values):
    """Per-register statistical analysis."""
    unique = len(np.unique(values))
    result = {
        "name": name,
        "samples": len(values),
        "unique_values": unique,
        "unique_ratio": unique / max(len(values), 1),
        "min": int(np.min(values)),
        "max": int(np.max(values)),
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "is_dynamic": unique > 1,
    }

    if unique > 1 and np.std(values) > 0:
        # Byte entropy
        byte_counts = Counter(values & 0xFF)
        total = sum(byte_counts.values())
        probs = np.array([c / total for c in byte_counts.values()])
        result["byte_entropy"] = float(-np.sum(probs * np.log2(probs + 1e-15)))

        # PSD slope
        sig = values.astype(float) - np.mean(values)
        if len(sig) >= 64:
            fft = np.fft.rfft(sig)
            psd = np.abs(fft) ** 2
            freqs = np.fft.rfftfreq(len(sig))
            mask = freqs > 0
            if np.sum(mask) > 10:
                log_f = np.log10(freqs[mask][:len(psd[mask]) // 2])
                log_p = np.log10(psd[mask][:len(psd[mask]) // 2] + 1e-15)
                coeffs = np.polyfit(log_f, log_p, 1)
                result["psd_slope"] = float(coeffs[0])

        # Autocorrelation at lag 1
        if len(sig) > 2:
            acf1 = np.corrcoef(sig[:-1], sig[1:])[0, 1]
            result["autocorrelation_lag1"] = float(acf1)

        # Shannon capacity (SNR-based)
        signal_var = np.var(values.astype(float))
        # Estimate noise as high-freq component
        diff = np.diff(values.astype(float))
        noise_var = np.var(diff) / 2  # difference noise
        if noise_var > 0:
            snr = signal_var / noise_var
            result["shannon_capacity_bits"] = float(0.5 * np.log2(1 + snr))
            result["snr_db"] = float(10 * np.log10(snr + 1e-15))
    else:
        result["byte_entropy"] = 0.0
        result["psd_slope"] = 0.0
        result["autocorrelation_lag1"] = 0.0
        result["shannon_capacity_bits"] = 0.0

    return result


def analyze_jitter_for_fpga(jitter_bytes):
    """Analyze jitter byte quality for FPGA ISI injection (z2153 bridge)."""
    byte_counts = Counter(jitter_bytes)
    total = len(jitter_bytes)
    probs = np.array([byte_counts.get(i, 0) / total for i in range(256)])
    probs_nonzero = probs[probs > 0]
    entropy = -np.sum(probs_nonzero * np.log2(probs_nonzero))

    # Uniformity test: chi-squared against uniform
    expected = total / 256.0
    observed = np.array([byte_counts.get(i, 0) for i in range(256)])
    chi2 = np.sum((observed - expected) ** 2 / max(expected, 1e-10))
    # p-value approximation (df=255)
    from scipy import stats as sp_stats
    p_value = 1 - sp_stats.chi2.cdf(chi2, df=255)

    # ISI CV prediction: if jitter maps to +-threshold_pct
    # For threshold modulation of +-X%, the CV of resulting ISI is ~X/100
    # With 8-bit jitter (0-255), mapping to +-5% gives CV ~ jitter_std/128 * 0.05
    jitter_std = np.std(jitter_bytes)
    predicted_isi_cv = jitter_std / 128.0  # normalized jitter amplitude

    return {
        "entropy_bits": float(entropy),
        "max_entropy": 8.0,
        "efficiency": float(entropy / 8.0),
        "unique_values": len(byte_counts),
        "chi2_statistic": float(chi2),
        "chi2_p_value": float(p_value),
        "is_uniform": p_value > 0.01,
        "jitter_std": float(jitter_std),
        "predicted_isi_cv": float(predicted_isi_cv),
        "target_isi_cv_range": list(LANZA_REF["isi_cv_range"]),
        "jitter_amplitude_for_cv_0.5": "map jitter [0-255] -> Vg offset [-0.10, +0.10]V",
    }


def sample_pm_info(n_samples=20, interval_ms=50):
    """Sample amdgpu_pm_info for host-side analog signals."""
    samples = []
    for i in range(n_samples):
        try:
            result = subprocess.run(
                ["sudo", "cat", PM_INFO_PATH],
                capture_output=True, text=True, timeout=2
            )
            if result.returncode == 0:
                text = result.stdout
                sample = {"timestamp": time.time()}
                # Parse key fields
                for line in text.splitlines():
                    if "SCLK" in line and "MHz" in line:
                        m = re.search(r'(\d+)\s*MHz\s*\(SCLK\)', line)
                        if m: sample["sclk_mhz"] = int(m.group(1))
                    elif "MCLK" in line and "MHz" in line and "PSTATE" not in line:
                        m = re.search(r'(\d+)\s*MHz\s*\(MCLK\)', line)
                        if m: sample["mclk_mhz"] = int(m.group(1))
                    elif "VDDGFX" in line:
                        m = re.search(r'(\d+)\s*mV\s*\(VDDGFX\)', line)
                        if m: sample["vddgfx_mv"] = int(m.group(1))
                    elif "average SoC" in line:
                        m = re.search(r'([\d.]+)\s*W\s*\(average', line)
                        if m: sample["soc_power_w"] = float(m.group(1))
                    elif "current SoC" in line:
                        m = re.search(r'([\d.]+)\s*W\s*\(current', line)
                        if m: sample["soc_current_w"] = float(m.group(1))
                    elif "GPU Temperature" in line:
                        m = re.search(r'(\d+)\s*C', line)
                        if m: sample["gpu_temp_c"] = int(m.group(1))
                    elif "GPU Load" in line:
                        m = re.search(r'(\d+)\s*%', line)
                        if m: sample["gpu_load_pct"] = int(m.group(1))
                samples.append(sample)
        except Exception as e:
            pass
        time.sleep(interval_ms / 1000.0)
    return samples


def sample_fence_deltas(n_samples=10, interval_ms=100):
    """Sample fence_info for ring activity deltas."""
    def read_fence():
        try:
            result = subprocess.run(
                ["sudo", "cat", FENCE_INFO_PATH],
                capture_output=True, text=True, timeout=2
            )
            if result.returncode != 0:
                return {}
            rings = {}
            current_ring = None
            for line in result.stdout.splitlines():
                if line.startswith("--- ring"):
                    m = re.search(r'ring (\d+) \((\w+)', line)
                    if m:
                        current_ring = m.group(2)
                elif "Last emitted" in line and current_ring and "trailing" not in line.lower():
                    m = re.search(r'0x([0-9a-fA-F]+)', line)
                    if m:
                        rings[current_ring] = int(m.group(1), 16)
            return rings
        except:
            return {}

    snapshots = []
    for _ in range(n_samples):
        snapshots.append({"timestamp": time.time(), "fences": read_fence()})
        time.sleep(interval_ms / 1000.0)

    # Compute deltas
    deltas = defaultdict(list)
    for i in range(1, len(snapshots)):
        for ring in snapshots[i]["fences"]:
            if ring in snapshots[i-1]["fences"]:
                d = snapshots[i]["fences"][ring] - snapshots[i-1]["fences"][ring]
                deltas[ring].append(d)
    return dict(deltas)


def sample_gpu_metrics_raw(n_samples=100, interval_ms=2):
    """Sample gpu_metrics binary for fast telemetry."""
    samples = []
    for _ in range(n_samples):
        try:
            with open(GPU_METRICS_PATH, "rb") as f:
                raw = f.read()
            if len(raw) >= 264:
                # Parse key fields from gpu_metrics v3.0
                header = struct.unpack_from("<HBB", raw, 0)
                temp_gfx = struct.unpack_from("<H", raw, 60)[0]  # temperature_gfx
                temp_soc = struct.unpack_from("<H", raw, 62)[0]  # temperature_soc
                avg_power = struct.unpack_from("<H", raw, 80)[0]  # average_socket_power
                gfx_clk = struct.unpack_from("<H", raw, 92)[0]   # current_gfxclk
                soc_clk = struct.unpack_from("<H", raw, 94)[0]   # current_socclk
                throttle = struct.unpack_from("<I", raw, 100)[0]  # throttle_status
                gfx_activity = struct.unpack_from("<H", raw, 104)[0]  # current_gfxclk_activity

                samples.append({
                    "timestamp": time.time(),
                    "temp_gfx": temp_gfx / 100.0 if temp_gfx < 20000 else 0,
                    "temp_soc": temp_soc / 100.0 if temp_soc < 20000 else 0,
                    "avg_power_w": avg_power,
                    "gfx_clk_mhz": gfx_clk,
                    "soc_clk_mhz": soc_clk,
                    "throttle_status": throttle,
                    "gfx_activity": gfx_activity,
                })
        except:
            pass
        time.sleep(interval_ms / 1000.0)
    return samples


def compute_bridge_fidelity(reg_results, jitter_analysis, pm_samples, fence_deltas):
    """Compute enhanced bridge fidelity vs Mario Lanza NS-RAM reference."""

    scores = {}

    # 1. PSD slope proximity to -1.0 (Lanza 1/f)
    # Use best PSD from all registers
    best_psd = 0.0
    for reg in reg_results:
        if reg.get("psd_slope", 0) != 0:
            dist = abs(reg["psd_slope"] - LANZA_REF["psd_slope"])
            score = max(0, 1 - dist / 2.0)
            if score > best_psd:
                best_psd = score
    scores["psd_match"] = best_psd

    # 2. Total entropy across ALL channels
    total_entropy = sum(r.get("byte_entropy", 0) for r in reg_results if r.get("is_dynamic"))
    n_dynamic = sum(1 for r in reg_results if r.get("is_dynamic"))
    avg_entropy = total_entropy / max(n_dynamic, 1)
    scores["entropy"] = min(1.0, avg_entropy / LANZA_REF["entropy_bits"])

    # 3. Total analog bits (Shannon capacity sum)
    total_bits = sum(r.get("shannon_capacity_bits", 0) for r in reg_results if r.get("is_dynamic"))
    scores["total_analog_bits"] = total_bits
    scores["analog_bit_score"] = min(1.0, total_bits / 24.0)  # 24 = target bits

    # 4. Jitter quality for ISI injection
    jitter_eff = jitter_analysis.get("efficiency", 0)
    isi_cv = jitter_analysis.get("predicted_isi_cv", 0)
    target_cv_min, target_cv_max = LANZA_REF["isi_cv_range"]
    if target_cv_min <= isi_cv <= target_cv_max:
        cv_score = 1.0
    elif isi_cv < target_cv_min:
        cv_score = isi_cv / target_cv_min
    else:
        cv_score = max(0, 1 - (isi_cv - target_cv_max) / target_cv_max)
    scores["jitter_quality"] = (jitter_eff + cv_score) / 2.0

    # 5. Temporal memory (best autocorrelation from any register)
    best_acf = 0.0
    for reg in reg_results:
        acf = abs(reg.get("autocorrelation_lag1", 0))
        if acf > best_acf:
            best_acf = acf
    scores["temporal_memory"] = min(1.0, best_acf / 0.3)  # target: 0.3 ACF

    # 6. Dynamic range from pm_info
    if pm_samples:
        powers = [s.get("soc_power_w", 0) for s in pm_samples if "soc_power_w" in s]
        if powers and max(powers) > 0:
            dr = max(powers) / max(min(powers), 0.01)
            scores["power_dynamic_range"] = min(1.0, dr / LANZA_REF["dynamic_range_x"])
        else:
            scores["power_dynamic_range"] = 0.0
    else:
        scores["power_dynamic_range"] = 0.0

    # 7. Number of dynamic signal dimensions
    scores["signal_dimensions"] = n_dynamic
    scores["dimension_score"] = min(1.0, n_dynamic / 8.0)  # target: 8 dynamic registers

    # Overall fidelity (weighted)
    weights = {
        "psd_match": 0.20,
        "entropy": 0.15,
        "analog_bit_score": 0.15,
        "jitter_quality": 0.15,
        "temporal_memory": 0.15,
        "power_dynamic_range": 0.05,
        "dimension_score": 0.15,
    }
    overall = sum(scores.get(k, 0) * w for k, w in weights.items())
    scores["overall_bridge_fidelity"] = overall

    # Gap analysis vs Lanza
    gaps = []
    if scores["psd_match"] < 0.5:
        gaps.append("PSD slope far from -1.0 (need IIR 1/f filter)")
    if scores["temporal_memory"] < 0.3:
        gaps.append("Low temporal memory (PERF_SNAPSHOT is white noise)")
    if scores["jitter_quality"] < 0.5:
        gaps.append("Jitter quality insufficient for ISI injection")
    if n_dynamic < 4:
        gaps.append(f"Only {n_dynamic} dynamic dimensions (need ≥4)")
    scores["gaps_to_lanza"] = gaps
    scores["lanza_reference"] = LANZA_REF

    return scores


def main():
    import argparse
    parser = argparse.ArgumentParser(description="z2153 Deep Probe Bridge Analysis")
    parser.add_argument("--csv", type=str, help="Deep probe CSV path")
    parser.add_argument("--pm-samples", type=int, default=20, help="Number of pm_info samples")
    parser.add_argument("--fence-samples", type=int, default=10, help="Number of fence_info samples")
    parser.add_argument("--gpu-metrics-samples", type=int, default=100, help="gpu_metrics samples")
    parser.add_argument("--output", type=str, default=str(RESULTS / "z2153_bridge_analysis.json"))
    args = parser.parse_args()

    results = {
        "experiment": "z2153_deep_probe_bridge",
        "description": "Enhanced hwreg probe + bridge fidelity vs Mario Lanza NS-RAM",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }

    # ─── Part 1: Analyze deep probe CSV if available ───
    reg_results = []
    jitter_analysis = {}
    if args.csv and Path(args.csv).exists():
        print(f"\n{'='*60}")
        print("PART 1: Deep Probe Register Analysis")
        print(f"{'='*60}")
        data = parse_deep_probe_csv(args.csv)
        print(f"  Loaded {len(data.get('perf_snapshot', []))} samples")

        for regname in ['perf_snapshot', 'shader_cycles', 'status', 'ib_sts2',
                        'mem_bases', 'gpr_alloc', 'ib_sts', 'mode']:
            if regname in data:
                r = analyze_register(regname, data[regname])
                reg_results.append(r)
                dyn = "DYNAMIC" if r["is_dynamic"] else "STATIC"
                print(f"\n  [{dyn}] {regname}:")
                print(f"    Unique: {r['unique_values']}/{r['samples']} ({r['unique_ratio']:.3f})")
                if r["is_dynamic"]:
                    print(f"    Entropy: {r.get('byte_entropy', 0):.3f} bits")
                    print(f"    PSD slope: {r.get('psd_slope', 0):.3f}")
                    print(f"    ACF lag-1: {r.get('autocorrelation_lag1', 0):.4f}")
                    print(f"    Shannon capacity: {r.get('shannon_capacity_bits', 0):.2f} bits")
                else:
                    print(f"    Value: 0x{r['min']:08X}")

        # Part 2: Jitter analysis
        if 'jitter_byte' in data:
            print(f"\n{'='*60}")
            print("PART 2: Stochastic Jitter for FPGA ISI Injection")
            print(f"{'='*60}")
            jitter_analysis = analyze_jitter_for_fpga(data['jitter_byte'])
            print(f"  Entropy: {jitter_analysis['entropy_bits']:.3f} / 8.0 bits "
                  f"(efficiency: {jitter_analysis['efficiency']:.1%})")
            print(f"  Unique bytes: {jitter_analysis['unique_values']} / 256")
            print(f"  Chi-squared p-value: {jitter_analysis['chi2_p_value']:.4f} "
                  f"({'uniform' if jitter_analysis['is_uniform'] else 'NOT uniform'})")
            print(f"  Predicted ISI CV: {jitter_analysis['predicted_isi_cv']:.3f}")
            print(f"  Target ISI CV: {jitter_analysis['target_isi_cv_range']}")

    results["register_analysis"] = reg_results
    results["jitter_analysis"] = jitter_analysis

    # ─── Part 3: Host-side signal sampling ───
    print(f"\n{'='*60}")
    print("PART 3: Host-Side Signal Sampling")
    print(f"{'='*60}")

    # PM info
    print(f"  Sampling pm_info ({args.pm_samples} samples, 50ms interval)...")
    pm_samples = sample_pm_info(n_samples=args.pm_samples, interval_ms=50)
    if pm_samples:
        powers = [s.get("soc_power_w", 0) for s in pm_samples if "soc_power_w" in s]
        temps = [s.get("gpu_temp_c", 0) for s in pm_samples if "gpu_temp_c" in s]
        print(f"    Power: {np.mean(powers):.1f}W (std={np.std(powers):.2f}W)")
        print(f"    Temp: {np.mean(temps):.1f}C (std={np.std(temps):.2f}C)")
    else:
        print("    (pm_info not accessible — need sudo)")
    results["pm_info_samples"] = pm_samples

    # Fence deltas
    print(f"  Sampling fence_info ({args.fence_samples} samples, 100ms interval)...")
    fence_deltas = sample_fence_deltas(n_samples=args.fence_samples, interval_ms=100)
    active_rings = {k: v for k, v in fence_deltas.items() if any(d != 0 for d in v)}
    print(f"    Active rings: {list(active_rings.keys()) if active_rings else 'none (idle)'}")
    for ring, deltas in active_rings.items():
        print(f"      {ring}: total delta={sum(deltas)}, mean={np.mean(deltas):.1f}")
    results["fence_deltas"] = {k: [int(x) for x in v] for k, v in fence_deltas.items()}

    # gpu_metrics fast sample
    print(f"  Sampling gpu_metrics ({args.gpu_metrics_samples} samples, 2ms interval)...")
    gm_samples = sample_gpu_metrics_raw(n_samples=args.gpu_metrics_samples, interval_ms=2)
    if gm_samples:
        gm_temps = [s["temp_gfx"] for s in gm_samples if s["temp_gfx"] > 0]
        gm_powers = [s["avg_power_w"] for s in gm_samples]
        gm_clks = [s["gfx_clk_mhz"] for s in gm_samples]
        print(f"    Temp GFX: {np.mean(gm_temps):.1f}C" if gm_temps else "    Temp: N/A")
        print(f"    Avg power: {np.mean(gm_powers):.0f}W, Clk: {np.mean(gm_clks):.0f}MHz")
    results["gpu_metrics_samples_count"] = len(gm_samples)

    # ─── Part 4: Bridge Fidelity ───
    print(f"\n{'='*60}")
    print("PART 4: Enhanced Bridge Fidelity vs Mario Lanza NS-RAM")
    print(f"{'='*60}")

    fidelity = compute_bridge_fidelity(reg_results, jitter_analysis, pm_samples, fence_deltas)
    results["bridge_fidelity"] = fidelity

    print(f"\n  Component scores:")
    for key in ["psd_match", "entropy", "analog_bit_score", "jitter_quality",
                "temporal_memory", "power_dynamic_range", "dimension_score"]:
        if key in fidelity:
            bar = "█" * int(fidelity[key] * 20) + "░" * (20 - int(fidelity[key] * 20))
            print(f"    {key:25s}: {fidelity[key]:.3f} [{bar}]")

    overall = fidelity["overall_bridge_fidelity"]
    if overall >= 0.8:
        grade = "EXCELLENT — near-analog substrate character"
    elif overall >= 0.6:
        grade = "GOOD — significant analog character"
    elif overall >= 0.4:
        grade = "MODERATE — some analog features"
    else:
        grade = "WEAK — mostly digital character"

    print(f"\n  ★ Overall Bridge Fidelity: {overall:.3f} — {grade}")
    print(f"  ★ Total analog bits: {fidelity.get('total_analog_bits', 0):.1f}")
    print(f"  ★ Dynamic dimensions: {fidelity.get('signal_dimensions', 0)}")

    if fidelity.get("gaps_to_lanza"):
        print(f"\n  Gaps to close:")
        for gap in fidelity["gaps_to_lanza"]:
            print(f"    ⚠ {gap}")

    # ─── Part 5: Lanza Comparison Table ───
    print(f"\n{'='*60}")
    print("PART 5: GPU Proxy vs Mario Lanza NS-RAM Comparison")
    print(f"{'='*60}")

    best_psd = min((r.get("psd_slope", 0) for r in reg_results), default=0, key=lambda x: abs(x + 1))
    best_entropy = max((r.get("byte_entropy", 0) for r in reg_results), default=0)
    best_acf = max((abs(r.get("autocorrelation_lag1", 0)) for r in reg_results), default=0)

    comparisons = [
        ("PSD slope",     f"{best_psd:.3f}",           f"{LANZA_REF['psd_slope']:.1f}",    "1/f noise"),
        ("Byte entropy",  f"{best_entropy:.3f}",       f"{LANZA_REF['entropy_bits']:.1f}",  "bits"),
        ("ACF lag-1",     f"{best_acf:.4f}",           "0.3+",                              "temporal memory"),
        ("ISI CV",        f"{jitter_analysis.get('predicted_isi_cv', 0):.3f}",
                          f"{LANZA_REF['isi_cv_range']}",                                   "stochasticity"),
        ("Analog bits",   f"{fidelity.get('total_analog_bits', 0):.1f}",
                          "24+",                                                            "Shannon capacity"),
        ("Dimensions",    f"{fidelity.get('signal_dimensions', 0)}",
                          "∞ (continuous)",                                                  "signal channels"),
    ]

    print(f"\n  {'Metric':<18} {'GPU Proxy':<14} {'NS-RAM Ref':<16} {'Notes'}")
    print(f"  {'-'*18} {'-'*14} {'-'*16} {'-'*20}")
    for metric, gpu, nsram, notes in comparisons:
        print(f"  {metric:<18} {gpu:<14} {nsram:<16} {notes}")

    # Save results
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Results saved to {args.output}")

    return results


if __name__ == "__main__":
    main()
