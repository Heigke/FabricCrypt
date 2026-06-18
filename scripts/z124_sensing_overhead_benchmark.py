#!/usr/bin/env python3
"""
z124 Sensing Overhead Benchmark

Measures the REAL cost of body sensing:
1. Inline sysfs read (current bad approach)
2. Background sampling with RAM read (new good approach)

This is critical because z121 showed 10% sensing overhead.
If background sampling reduces this to <1%, adaptive mode becomes viable.
"""

import os
import sys
import time
import argparse
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def find_amd_hwmon() -> Optional[Path]:
    """Find AMD hwmon path."""
    for i in range(20):
        hwmon = Path(f"/sys/class/hwmon/hwmon{i}")
        if not hwmon.exists():
            continue
        name_file = hwmon / "name"
        if name_file.exists() and "amdgpu" in name_file.read_text():
            if (hwmon / "power1_average").exists():
                return hwmon
    return None


def benchmark_inline_read(path: Path, iterations: int = 1000) -> Dict:
    """Benchmark inline sysfs file read (the bad way)."""
    power_file = path / "power1_average"

    times = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        with open(power_file) as f:
            _ = f.read()
        times.append(time.perf_counter() - t0)

    return {
        "method": "inline_read",
        "iterations": iterations,
        "total_ms": sum(times) * 1000,
        "mean_us": np.mean(times) * 1_000_000,
        "std_us": np.std(times) * 1_000_000,
        "min_us": np.min(times) * 1_000_000,
        "max_us": np.max(times) * 1_000_000,
        "p99_us": np.percentile(times, 99) * 1_000_000,
    }


def benchmark_pread(path: Path, iterations: int = 1000) -> Dict:
    """Benchmark os.pread on pre-opened fd (the good way)."""
    power_file = path / "power1_average"
    fd = os.open(str(power_file), os.O_RDONLY)

    times = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        _ = os.pread(fd, 32, 0)
        times.append(time.perf_counter() - t0)

    os.close(fd)

    return {
        "method": "pread",
        "iterations": iterations,
        "total_ms": sum(times) * 1000,
        "mean_us": np.mean(times) * 1_000_000,
        "std_us": np.std(times) * 1_000_000,
        "min_us": np.min(times) * 1_000_000,
        "max_us": np.max(times) * 1_000_000,
        "p99_us": np.percentile(times, 99) * 1_000_000,
    }


def benchmark_background_sampler(iterations: int = 1000) -> Dict:
    """Benchmark reading from background sampler (RAM only)."""
    from sensing.background_telemetry import BackgroundTelemetrySampler, TelemetryConfig

    config = TelemetryConfig(sample_rate_hz=100)
    sampler = BackgroundTelemetrySampler(config)
    sampler.start()
    time.sleep(0.2)  # Let sampler stabilize

    times = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        _ = sampler.get_latest_body_vec()
        times.append(time.perf_counter() - t0)

    sampler.stop()

    return {
        "method": "background_sampler",
        "iterations": iterations,
        "total_ms": sum(times) * 1000,
        "mean_us": np.mean(times) * 1_000_000,
        "std_us": np.std(times) * 1_000_000,
        "min_us": np.min(times) * 1_000_000,
        "max_us": np.max(times) * 1_000_000,
        "p99_us": np.percentile(times, 99) * 1_000_000,
    }


def benchmark_numpy_copy(iterations: int = 1000) -> Dict:
    """Benchmark pure numpy copy (theoretical minimum)."""
    arr = np.random.randn(12).astype(np.float32)

    times = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        _ = arr.copy()
        times.append(time.perf_counter() - t0)

    return {
        "method": "numpy_copy",
        "iterations": iterations,
        "total_ms": sum(times) * 1000,
        "mean_us": np.mean(times) * 1_000_000,
        "std_us": np.std(times) * 1_000_000,
        "min_us": np.min(times) * 1_000_000,
        "max_us": np.max(times) * 1_000_000,
        "p99_us": np.percentile(times, 99) * 1_000_000,
    }


def estimate_overhead_impact(inline_us: float, background_us: float, tokens_per_sec: float = 150) -> Dict:
    """Estimate impact on inference."""
    token_time_us = 1_000_000 / tokens_per_sec  # ~6667 us at 150 TPS

    inline_overhead_pct = (inline_us / token_time_us) * 100
    background_overhead_pct = (background_us / token_time_us) * 100

    return {
        "token_time_us": token_time_us,
        "inline_overhead_pct": inline_overhead_pct,
        "background_overhead_pct": background_overhead_pct,
        "improvement_factor": inline_us / background_us if background_us > 0 else float('inf'),
    }


def main():
    parser = argparse.ArgumentParser(description="Sensing Overhead Benchmark")
    parser.add_argument("--iterations", type=int, default=1000)
    args = parser.parse_args()

    print("=" * 60)
    print("SENSING OVERHEAD BENCHMARK (z124)")
    print("=" * 60)

    # Find hwmon
    hwmon = find_amd_hwmon()
    if not hwmon:
        print("ERROR: No AMD hwmon found!")
        return

    print(f"\nUsing hwmon: {hwmon}")
    print(f"Iterations: {args.iterations}")

    # Run benchmarks
    print("\n" + "-" * 60)
    print("1. Inline sysfs read (current bad approach)")
    inline_result = benchmark_inline_read(hwmon, args.iterations)
    print(f"   Mean: {inline_result['mean_us']:.1f} μs")
    print(f"   P99:  {inline_result['p99_us']:.1f} μs")

    print("\n2. os.pread on pre-opened fd")
    pread_result = benchmark_pread(hwmon, args.iterations)
    print(f"   Mean: {pread_result['mean_us']:.1f} μs")
    print(f"   P99:  {pread_result['p99_us']:.1f} μs")
    print(f"   Speedup vs inline: {inline_result['mean_us']/pread_result['mean_us']:.1f}x")

    print("\n3. Background sampler (RAM read only)")
    background_result = benchmark_background_sampler(args.iterations)
    print(f"   Mean: {background_result['mean_us']:.1f} μs")
    print(f"   P99:  {background_result['p99_us']:.1f} μs")
    print(f"   Speedup vs inline: {inline_result['mean_us']/background_result['mean_us']:.1f}x")

    print("\n4. Pure numpy copy (theoretical minimum)")
    numpy_result = benchmark_numpy_copy(args.iterations)
    print(f"   Mean: {numpy_result['mean_us']:.2f} μs")

    # Impact analysis
    print("\n" + "-" * 60)
    print("IMPACT ANALYSIS (at 150 TPS)")
    print("-" * 60)

    impact = estimate_overhead_impact(
        inline_result['mean_us'],
        background_result['mean_us'],
        150
    )

    print(f"\nToken generation time: {impact['token_time_us']:.0f} μs")
    print(f"\nInline sysfs read overhead:")
    print(f"  {inline_result['mean_us']:.1f} μs per read = {impact['inline_overhead_pct']:.2f}% of token time")

    print(f"\nBackground sampler overhead:")
    print(f"  {background_result['mean_us']:.1f} μs per read = {impact['background_overhead_pct']:.4f}% of token time")

    print(f"\nImprovement: {impact['improvement_factor']:.0f}x faster")
    print(f"Overhead reduction: {impact['inline_overhead_pct']:.2f}% → {impact['background_overhead_pct']:.4f}%")

    # Verdict
    print("\n" + "=" * 60)
    print("VERDICT")
    print("=" * 60)

    if impact['background_overhead_pct'] < 0.1:
        print("✓ Background sampler reduces sensing overhead to NEGLIGIBLE (<0.1%)")
        print("✓ Adaptive mode is now viable!")
    else:
        print("⚠ Background sampler still has measurable overhead")
        print(f"  Consider reducing body_dim or sampling rate")


if __name__ == "__main__":
    main()
