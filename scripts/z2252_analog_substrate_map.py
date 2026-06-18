#!/usr/bin/env python3
"""z2252_analog_substrate_map.py — Systematic mapping of GPU analog substrate signals

Maps ALL readable low-level analog signals from the GPU and characterizes:
  1. Which signals exist and their dynamic range
  2. How each signal responds to different GPU workload types
  3. Signal independence (correlation matrix)
  4. Temporal dynamics (response time, ACF, spectral character)
  5. Controllability: can we SET a target signal level?
  6. Multi-signal state space: how many independent dimensions?

This is the foundation for using GPU firmware/hardware as a neuromorphic substrate.

Prior findings integrated:
  - L1 cache jitter: CU-wide, thermally controllable 0.1%→82%, 1 stream per chip
  - Memory contention: per-wavefront, CV 0.003→0.195, multiple independent streams
  - Power (hwmon): native 1/f character, ~11W ± 1.5W
  - gpu_metrics: 10+ dynamic fields at ~6500Hz read rate
  - DVFS: power_dpm_force_performance_level high/low/auto works
  - Voltage sensors: broken (read 0) on kernel 6.14
"""

import struct
import time
import json
import os
import sys
import signal
import statistics
import subprocess
import threading
import numpy as np
from collections import defaultdict
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# Signal readers
# ─────────────────────────────────────────────────────────────────────────────

GPU_METRICS_PATH = "/sys/class/drm/card0/device/gpu_metrics"
HWMON_PATH = "/sys/class/drm/card0/device/hwmon/hwmon7"
DPM_PATH = "/sys/class/drm/card0/device/power_dpm_force_performance_level"

# All gpu_metrics fields with known offsets (v3.0 APU)
GPU_METRICS_FIELDS = [
    (4,   "temp_gfx",        "cC",   0.01),   # centi-Celsius → C
    (6,   "temp_soc",        "cC",   0.01),
    (50,  "gfx_activity",    "c%",   0.01),   # centi-percent → %
    (52,  "mm_activity",     "c%",   0.01),
    (94,  "gfxclk",          "MHz",  1.0),
    (96,  "socclk",          "MHz",  1.0),
    (104, "socket_power",    "mW",   0.001),  # mW → W
    (106, "gfx_power",       "mW",   0.001),
    (108, "soc_power",       "mW",   0.001),
    (112, "soc_voltage",     "mV",   1.0),
    (120, "cpu_voltage",     "mV",   1.0),
    (132, "coreclk",         "MHz",  1.0),
    (136, "gfxclk_2",        "MHz",  1.0),
    (138, "socclk_2",        "MHz",  1.0),
    (168, "throttle",        "bits", 1.0),
    (174, "gfxclk_3",        "MHz",  1.0),
    (178, "uclk",            "MHz",  1.0),
    (182, "vclk",            "MHz",  1.0),
    (184, "dclk",            "MHz",  1.0),
    (190, "lclk",            "MHz",  1.0),
]

HWMON_FIELDS = [
    ("power_avg",  f"{HWMON_PATH}/power1_average",  "uW", 1e-6),  # µW → W
    ("power_inst", f"{HWMON_PATH}/power1_input",     "uW", 1e-6),
    ("temp_edge",  f"{HWMON_PATH}/temp1_input",      "mC", 0.001), # mC → C
    ("freq_gfx",   f"{HWMON_PATH}/freq1_input",      "Hz", 1e-6),  # Hz → MHz
]


def read_gpu_metrics():
    """Read all gpu_metrics fields, return dict of raw uint16 values."""
    try:
        with open(GPU_METRICS_PATH, "rb") as f:
            data = f.read()
        result = {}
        for offset, name, unit, scale in GPU_METRICS_FIELDS:
            if offset + 2 <= len(data):
                raw = struct.unpack_from("<H", data, offset)[0]
                result[name] = raw
        return result
    except:
        return {}


def read_hwmon():
    """Read hwmon sysfs values."""
    result = {}
    for name, path, unit, scale in HWMON_FIELDS:
        try:
            with open(path) as f:
                result[name] = int(f.read().strip())
        except:
            result[name] = 0
    return result


def read_all_signals():
    """Read ALL analog signals, return dict of raw values."""
    d = read_gpu_metrics()
    d.update(read_hwmon())
    return d


def signal_to_physical(name, raw):
    """Convert raw value to physical units."""
    for offset, n, unit, scale in GPU_METRICS_FIELDS:
        if n == name:
            return raw * scale
    for n, path, unit, scale in HWMON_FIELDS:
        if n == name:
            return raw * scale
    return raw


# ─────────────────────────────────────────────────────────────────────────────
# GPU workload generators (run in background threads via torch)
# ─────────────────────────────────────────────────────────────────────────────

WORKLOAD_SCRIPT_TEMPLATE = '''
import torch, time, sys, os
os.environ["HSA_OVERRIDE_GFX_VERSION"] = "11.0.0"
d = torch.device("cuda")
workload = sys.argv[1]
duration = float(sys.argv[2])

t0 = time.monotonic()
try:
    if workload == "idle":
        time.sleep(duration)

    elif workload == "matmul_light":
        a = torch.randn(256, 256, device=d)
        while time.monotonic() - t0 < duration:
            b = a @ a
            torch.cuda.synchronize()
            time.sleep(0.01)

    elif workload == "matmul_heavy":
        a = torch.randn(4096, 4096, device=d)
        while time.monotonic() - t0 < duration:
            b = a @ a
            torch.cuda.synchronize()

    elif workload == "matmul_medium":
        a = torch.randn(1024, 1024, device=d)
        while time.monotonic() - t0 < duration:
            b = a @ a
            torch.cuda.synchronize()

    elif workload == "memory_bandwidth":
        a = torch.randn(16*1024*1024, device=d)
        b = torch.empty_like(a)
        while time.monotonic() - t0 < duration:
            b.copy_(a)
            a.copy_(b)
            torch.cuda.synchronize()

    elif workload == "memory_random":
        a = torch.randn(4*1024*1024, device=d)
        idx = torch.randint(0, len(a), (1024*1024,), device=d)
        while time.monotonic() - t0 < duration:
            b = a[idx]
            torch.cuda.synchronize()
            time.sleep(0.001)

    elif workload == "fft":
        a = torch.randn(1024, 1024, device=d, dtype=torch.complex64)
        while time.monotonic() - t0 < duration:
            b = torch.fft.fft2(a)
            torch.cuda.synchronize()

    elif workload == "conv":
        inp = torch.randn(8, 64, 128, 128, device=d)
        w = torch.randn(128, 64, 3, 3, device=d)
        while time.monotonic() - t0 < duration:
            out = torch.nn.functional.conv2d(inp, w, padding=1)
            torch.cuda.synchronize()

    elif workload == "reduce":
        a = torch.randn(16*1024*1024, device=d)
        while time.monotonic() - t0 < duration:
            s = a.sum()
            v = a.var()
            torch.cuda.synchronize()

    elif workload == "mixed_oscillate":
        a_small = torch.randn(256, 256, device=d)
        a_big = torch.randn(2048, 2048, device=d)
        while time.monotonic() - t0 < duration:
            # Heavy burst
            for _ in range(5):
                b = a_big @ a_big
            torch.cuda.synchronize()
            time.sleep(0.05)
            # Light burst
            for _ in range(5):
                b = a_small @ a_small
            torch.cuda.synchronize()
            time.sleep(0.05)

    elif workload == "dvfs_sweep":
        sizes = [128, 256, 512, 1024, 2048, 4096]
        step_time = duration / len(sizes)
        for sz in sizes:
            a = torch.randn(sz, sz, device=d)
            t_step = time.monotonic()
            while time.monotonic() - t_step < step_time:
                b = a @ a
                torch.cuda.synchronize()

except Exception as e:
    print(f"workload error: {e}", file=sys.stderr)
'''


def run_workload(name, duration):
    """Start a GPU workload subprocess, return Popen object."""
    env = os.environ.copy()
    env["HSA_OVERRIDE_GFX_VERSION"] = "11.0.0"
    proc = subprocess.Popen(
        [sys.executable, "-c", WORKLOAD_SCRIPT_TEMPLATE, name, str(duration)],
        env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    return proc


def sample_signals(duration_s, rate_hz=200):
    """Sample all signals for given duration at given rate."""
    interval = 1.0 / rate_hz
    samples = defaultdict(list)
    timestamps = []

    t0 = time.monotonic()
    while time.monotonic() - t0 < duration_s:
        t_sample = time.monotonic()
        d = read_all_signals()
        for k, v in d.items():
            samples[k].append(v)
        timestamps.append(t_sample - t0)
        # Busy-wait for precise timing
        while time.monotonic() - t_sample < interval:
            pass

    return dict(samples), timestamps


# ─────────────────────────────────────────────────────────────────────────────
# Analysis functions
# ─────────────────────────────────────────────────────────────────────────────

def compute_stats(values):
    """Compute comprehensive statistics for a signal."""
    if not values or len(values) < 2:
        return {}
    v = np.array(values, dtype=np.float64)
    # Filter out uint16 overflow artifacts (values near 65535 or 0 that are clearly wrong)
    mn, mx = float(v.min()), float(v.max())
    mean = float(v.mean())
    std = float(v.std())
    cv = std / mean if mean != 0 else 0
    unique = len(np.unique(v))

    # Autocorrelation at lags 1, 5, 10, 50
    acf = {}
    centered = v - mean
    var = float(np.sum(centered ** 2))
    for lag in [1, 2, 5, 10, 20, 50]:
        if lag < len(v) and var > 0:
            cov = float(np.sum(centered[:-lag] * centered[lag:]))
            acf[lag] = cov / var
        else:
            acf[lag] = 0

    # PSD slope estimate (first 64 frequencies)
    if len(v) > 128 and std > 0:
        # Use FFT
        fft = np.fft.rfft(centered)
        power = np.abs(fft) ** 2
        freqs = np.arange(1, min(65, len(power)))
        if len(freqs) > 2:
            log_f = np.log(freqs)
            log_p = np.log(power[freqs] + 1e-30)
            # Linear regression
            A = np.vstack([log_f, np.ones_like(log_f)]).T
            slope, _ = np.linalg.lstsq(A, log_p, rcond=None)[0]
            psd_slope = float(slope)
        else:
            psd_slope = 0
    else:
        psd_slope = 0

    # Shannon entropy (binned)
    if unique > 1:
        hist, _ = np.histogram(v, bins=min(unique, 256))
        hist = hist[hist > 0]
        probs = hist / hist.sum()
        entropy = float(-np.sum(probs * np.log2(probs)))
    else:
        entropy = 0

    return {
        "min": mn, "max": mx, "mean": mean, "std": std, "cv": cv,
        "unique": unique, "acf": acf, "psd_slope": psd_slope,
        "entropy": entropy
    }


def correlation_matrix(samples):
    """Compute pairwise Pearson correlation between all dynamic signals."""
    # Filter to signals with >3 unique values
    dynamic = {k: np.array(v, dtype=np.float64) for k, v in samples.items()
               if len(set(v)) > 3 and len(v) > 10}

    names = sorted(dynamic.keys())
    n = len(names)
    corr = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            if len(dynamic[names[i]]) == len(dynamic[names[j]]):
                a = dynamic[names[i]]
                b = dynamic[names[j]]
                if a.std() > 0 and b.std() > 0:
                    corr[i, j] = float(np.corrcoef(a, b)[0, 1])
                else:
                    corr[i, j] = 0
            else:
                corr[i, j] = 0

    return names, corr


def effective_dimensions(corr_matrix):
    """Estimate effective dimensionality from correlation matrix eigenvalues."""
    eigvals = np.linalg.eigvalsh(corr_matrix)
    eigvals = eigvals[eigvals > 0.01]  # threshold noise
    if len(eigvals) == 0:
        return 0
    # Participation ratio: (sum λ)² / sum(λ²)
    pr = (eigvals.sum() ** 2) / (eigvals ** 2).sum()
    return float(pr)


# ─────────────────────────────────────────────────────────────────────────────
# Main experiments
# ─────────────────────────────────────────────────────────────────────────────

def main():
    results = {
        "experiment": "z2252_analog_substrate_map",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "gpu": "gfx1151 (Radeon 8060S)",
    }

    print("═══ z2252: GPU Analog Substrate Signal Map ═══", file=sys.stderr)

    # ─── EXP 1: Baseline signal inventory (idle GPU) ─────────────────────
    print("\n═══ EXP 1: Signal Inventory (idle) ═══", file=sys.stderr)
    samples_idle, ts_idle = sample_signals(3.0, rate_hz=500)

    print(f"  Sampled {len(ts_idle)} points at ~{len(ts_idle)/3:.0f}Hz", file=sys.stderr)
    print(f"  {'SIGNAL':20s} {'min':>10s} {'max':>10s} {'mean':>10s} {'std':>10s} {'CV':>8s} {'unique':>6s} {'ACF(1)':>8s} {'PSD':>6s} {'H':>6s}", file=sys.stderr)
    print("  " + "─" * 96, file=sys.stderr)

    exp1 = {}
    for name, vals in sorted(samples_idle.items()):
        s = compute_stats(vals)
        if not s:
            continue
        exp1[name] = s
        phys_mean = signal_to_physical(name, s["mean"])
        phys_std = signal_to_physical(name, s["std"])
        dynamic = "██" if s["unique"] > 10 else "░░" if s["unique"] > 2 else "  "
        print(f"  {dynamic} {name:18s} {s['min']:10.1f} {s['max']:10.1f} {s['mean']:10.1f} {s['std']:10.1f} {s['cv']:8.4f} {s['unique']:6d} {s['acf'].get(1,0):8.4f} {s['psd_slope']:6.2f} {s['entropy']:6.2f}", file=sys.stderr)
    results["exp1_idle"] = exp1

    # ─── EXP 2: Workload response mapping ────────────────────────────────
    print("\n═══ EXP 2: Workload Response Map ═══", file=sys.stderr)

    workloads = [
        "idle", "matmul_light", "matmul_medium", "matmul_heavy",
        "memory_bandwidth", "memory_random", "fft", "conv", "reduce",
        "mixed_oscillate"
    ]

    exp2 = {}
    workload_duration = 5.0
    settle_time = 1.0

    for wl in workloads:
        print(f"  Testing: {wl}...", end="", file=sys.stderr, flush=True)
        if wl == "idle":
            time.sleep(settle_time)
            samples, ts = sample_signals(workload_duration, rate_hz=200)
        else:
            proc = run_workload(wl, workload_duration + settle_time + 1)
            time.sleep(settle_time)  # let workload stabilize
            samples, ts = sample_signals(workload_duration, rate_hz=200)
            proc.terminate()
            proc.wait()
            time.sleep(0.5)  # cool down

        stats = {}
        for name, vals in samples.items():
            s = compute_stats(vals)
            if s:
                stats[name] = s

        exp2[wl] = stats

        # Show key signals
        keys = ["gfx_power", "gfxclk", "soc_voltage", "temp_gfx", "power_avg", "coreclk"]
        vals_str = " ".join(f"{k}={stats.get(k,{}).get('mean',0):.0f}" for k in keys if k in stats)
        print(f" {vals_str}", file=sys.stderr)

    results["exp2_workload_response"] = exp2

    # ─── EXP 3: Control curves — how does each signal scale with workload intensity? ─
    print("\n═══ EXP 3: Workload Intensity Control Curves ═══", file=sys.stderr)

    # Use matmul size as intensity proxy
    sizes = [0, 128, 256, 512, 1024, 2048, 3072, 4096]
    exp3 = {}

    for sz in sizes:
        if sz == 0:
            label = "idle"
            time.sleep(0.5)
            samples, ts = sample_signals(3.0, rate_hz=200)
        else:
            label = f"matmul_{sz}"
            wl_code = f"""
import torch, time, os
os.environ["HSA_OVERRIDE_GFX_VERSION"] = "11.0.0"
d = torch.device("cuda")
a = torch.randn({sz}, {sz}, device=d)
t0 = time.monotonic()
while time.monotonic() - t0 < 6:
    b = a @ a
    torch.cuda.synchronize()
"""
            env = os.environ.copy()
            env["HSA_OVERRIDE_GFX_VERSION"] = "11.0.0"
            proc = subprocess.Popen([sys.executable, "-c", wl_code],
                                    env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            time.sleep(1.5)
            samples, ts = sample_signals(3.0, rate_hz=200)
            proc.terminate()
            proc.wait()
            time.sleep(0.5)

        stats = {}
        for name, vals in samples.items():
            s = compute_stats(vals)
            if s:
                stats[name] = {"mean": s["mean"], "std": s["std"], "cv": s["cv"]}

        exp3[label] = stats
        print(f"  {label:15s}: power={stats.get('gfx_power',{}).get('mean',0):.0f} "
              f"clk={stats.get('gfxclk',{}).get('mean',0):.0f} "
              f"volt={stats.get('soc_voltage',{}).get('mean',0):.0f} "
              f"temp={stats.get('temp_gfx',{}).get('mean',0):.0f}", file=sys.stderr)

    results["exp3_control_curves"] = exp3

    # ─── EXP 4: Signal independence (correlation matrix) ─────────────────
    print("\n═══ EXP 4: Signal Independence (under mixed workload) ═══", file=sys.stderr)

    proc = run_workload("mixed_oscillate", 15)
    time.sleep(2)
    samples_mixed, ts_mixed = sample_signals(10.0, rate_hz=200)
    proc.terminate()
    proc.wait()

    names, corr = correlation_matrix(samples_mixed)
    eff_dim = effective_dimensions(corr)

    print(f"  Dynamic signals: {len(names)}", file=sys.stderr)
    print(f"  Effective dimensions: {eff_dim:.1f}", file=sys.stderr)
    print(f"  Correlation matrix (|r| > 0.5 shown):", file=sys.stderr)

    exp4_corr = {}
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            r = corr[i, j]
            if abs(r) > 0.5:
                print(f"    {names[i]:18s} ↔ {names[j]:18s}: r={r:+.3f}", file=sys.stderr)
            exp4_corr[f"{names[i]}__x__{names[j]}"] = float(r)

    # Show independent groups
    print(f"\n  Independent signal groups (|r| < 0.3):", file=sys.stderr)
    for i in range(len(names)):
        independent_of = []
        for j in range(len(names)):
            if i != j and abs(corr[i, j]) < 0.3:
                independent_of.append(names[j])
        if independent_of:
            print(f"    {names[i]:18s} independent of: {', '.join(independent_of[:5])}", file=sys.stderr)

    results["exp4_independence"] = {
        "signals": names,
        "correlations": exp4_corr,
        "effective_dimensions": eff_dim,
    }

    # ─── EXP 5: Temporal dynamics — response time to workload change ─────
    print("\n═══ EXP 5: Temporal Response Dynamics ═══", file=sys.stderr)

    # Record signal transition: idle → heavy → idle
    # Start idle sampling
    print("  Phase 1: idle baseline (2s)...", file=sys.stderr)
    samples_pre, ts_pre = sample_signals(2.0, rate_hz=200)

    # Start heavy workload
    print("  Phase 2: heavy workload onset (3s)...", file=sys.stderr)
    proc = run_workload("matmul_heavy", 5)
    samples_onset, ts_onset = sample_signals(3.0, rate_hz=200)

    # Stop workload
    print("  Phase 3: workload offset (3s)...", file=sys.stderr)
    proc.terminate()
    proc.wait()
    samples_offset, ts_offset = sample_signals(3.0, rate_hz=200)

    # For each dynamic signal, find time to 90% of transition
    exp5 = {}
    for name in samples_pre:
        if name not in samples_onset or name not in samples_offset:
            continue
        pre_vals = np.array(samples_pre[name], dtype=np.float64)
        onset_vals = np.array(samples_onset[name], dtype=np.float64)
        offset_vals = np.array(samples_offset[name], dtype=np.float64)

        if pre_vals.std() + onset_vals.std() == 0:
            continue

        pre_mean = float(pre_vals.mean())
        onset_mean = float(onset_vals.mean())
        delta = onset_mean - pre_mean

        if abs(delta) < 1:  # skip static signals
            continue

        # Find time to 90% of transition in onset phase
        target_90 = pre_mean + 0.9 * delta
        t90_idx = None
        for i, v in enumerate(onset_vals):
            if delta > 0 and v >= target_90:
                t90_idx = i
                break
            elif delta < 0 and v <= target_90:
                t90_idx = i
                break

        t90_ms = t90_idx * (1000.0 / 200) if t90_idx is not None else None

        # Spectral character of onset phase
        s = compute_stats(list(onset_vals))
        psd = s.get("psd_slope", 0) if s else 0

        exp5[name] = {
            "idle_mean": pre_mean,
            "active_mean": onset_mean,
            "delta": delta,
            "delta_pct": abs(delta / pre_mean * 100) if pre_mean != 0 else 0,
            "t90_ms": t90_ms,
            "onset_psd_slope": psd,
            "onset_acf1": s.get("acf", {}).get(1, 0) if s else 0,
        }

        t90_str = f"{t90_ms:.0f}ms" if t90_ms is not None else ">3000ms"
        print(f"  {name:18s}: idle={pre_mean:.0f} → active={onset_mean:.0f} "
              f"(Δ={delta:+.0f}, {abs(delta/pre_mean*100) if pre_mean else 0:.1f}%), "
              f"t90={t90_str}, PSD={psd:.2f}", file=sys.stderr)

    results["exp5_temporal"] = exp5

    # ─── EXP 6: DVFS as control knob ─────────────────────────────────────
    print("\n═══ EXP 6: DVFS Control Knob ═══", file=sys.stderr)

    dpm_levels = ["auto", "low", "high", "auto"]
    exp6 = {}

    for level in dpm_levels:
        print(f"  Setting DPM={level}...", end="", file=sys.stderr, flush=True)
        try:
            with open(DPM_PATH, "w") as f:
                f.write(level)
        except PermissionError:
            # Try with subprocess
            subprocess.run(["sudo", "tee", DPM_PATH],
                           input=level.encode(), capture_output=True)

        # Run medium workload to force GPU to target DPM
        proc = run_workload("matmul_medium", 5)
        time.sleep(2)
        samples, ts = sample_signals(2.0, rate_hz=200)
        proc.terminate()
        proc.wait()

        stats = {}
        for name, vals in samples.items():
            s = compute_stats(vals)
            if s:
                stats[name] = {"mean": s["mean"], "std": s["std"]}

        exp6[level] = stats
        print(f" power={stats.get('gfx_power',{}).get('mean',0):.0f} "
              f"clk={stats.get('gfxclk',{}).get('mean',0):.0f} "
              f"volt={stats.get('soc_voltage',{}).get('mean',0):.0f}", file=sys.stderr)
        time.sleep(1)

    # Restore auto
    try:
        with open(DPM_PATH, "w") as f:
            f.write("auto")
    except:
        subprocess.run(["sudo", "tee", DPM_PATH], input=b"auto", capture_output=True)

    results["exp6_dvfs"] = exp6

    # ─── EXP 7: Noise character at each workload level ───────────────────
    print("\n═══ EXP 7: Noise Character Summary ═══", file=sys.stderr)
    print(f"  {'SIGNAL':18s} {'idle_CV':>8s} {'light_CV':>8s} {'heavy_CV':>8s} "
          f"{'idle_PSD':>8s} {'heavy_PSD':>8s} {'ctrl_range':>10s}", file=sys.stderr)
    print("  " + "─" * 75, file=sys.stderr)

    exp7 = {}
    for name in sorted(exp1.keys()):
        idle_s = exp1.get(name, {})
        light_s = exp2.get("matmul_light", {}).get(name, {})
        heavy_s = exp2.get("matmul_heavy", {}).get(name, {})

        if not idle_s or idle_s.get("unique", 0) <= 1:
            continue

        idle_cv = idle_s.get("cv", 0)
        light_cv = light_s.get("cv", 0)
        heavy_cv = heavy_s.get("cv", 0)

        idle_psd = idle_s.get("psd_slope", 0)
        heavy_psd = heavy_s.get("psd_slope", 0)

        # Control range: difference between idle and heavy means
        idle_mean = idle_s.get("mean", 0)
        heavy_mean = heavy_s.get("mean", 0)
        ctrl_range = abs(heavy_mean - idle_mean)
        ctrl_pct = (ctrl_range / idle_mean * 100) if idle_mean != 0 else 0

        exp7[name] = {
            "idle_cv": idle_cv, "light_cv": light_cv, "heavy_cv": heavy_cv,
            "idle_psd": idle_psd, "heavy_psd": heavy_psd,
            "control_range": ctrl_range, "control_pct": ctrl_pct,
        }

        if ctrl_pct > 1:
            print(f"  {name:18s} {idle_cv:8.4f} {light_cv:8.4f} {heavy_cv:8.4f} "
                  f"{idle_psd:8.2f} {heavy_psd:8.2f} {ctrl_pct:9.1f}%", file=sys.stderr)

    results["exp7_noise_character"] = exp7

    # ─── Summary ─────────────────────────────────────────────────────────
    print("\n═══ SUMMARY ═══", file=sys.stderr)

    # Count usable signals
    n_dynamic = sum(1 for s in exp1.values() if s.get("unique", 0) > 10)
    n_controllable = sum(1 for s in exp7.values() if s.get("control_pct", 0) > 5)
    n_noisy = sum(1 for s in exp7.values() if s.get("heavy_cv", 0) > 0.01)

    print(f"  Total signals read: {len(exp1)}", file=sys.stderr)
    print(f"  Dynamic (>10 unique values): {n_dynamic}", file=sys.stderr)
    print(f"  Controllable (>5% range): {n_controllable}", file=sys.stderr)
    print(f"  Noisy (CV>1%): {n_noisy}", file=sys.stderr)
    print(f"  Effective independent dimensions: {eff_dim:.1f}", file=sys.stderr)
    print(f"  Read rate: ~{len(ts_idle)/3:.0f} Hz", file=sys.stderr)

    results["summary"] = {
        "total_signals": len(exp1),
        "dynamic_signals": n_dynamic,
        "controllable_signals": n_controllable,
        "noisy_signals": n_noisy,
        "effective_dimensions": eff_dim,
        "read_rate_hz": len(ts_idle) / 3,
    }

    # Save results
    out_path = "results/z2252_analog_substrate_map.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Results saved to {out_path}", file=sys.stderr)
    print("Done.", file=sys.stderr)


if __name__ == "__main__":
    main()
