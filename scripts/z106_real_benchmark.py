#!/usr/bin/env python3
"""
z106_real_benchmark.py - Real Energy Benchmark for FEEL-SLM

Proper A/B comparison between Baseline and FEEL-SLM with:
1. Real energy measurement from AMD/NVIDIA hardware
2. Warmup runs to stabilize
3. Multiple trials with 95% confidence intervals
4. Paired comparison (same prompts)
5. Statistical significance testing

Run: HSA_OVERRIDE_GFX_VERSION=11.0.0 python scripts/z106_real_benchmark.py
"""

import sys
import os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import time
import statistics
import math
import argparse
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Optional
import subprocess

import torch
import torch.nn.functional as F


@dataclass
class EnergyReading:
    """Single energy reading from hardware."""
    timestamp: float
    energy_uj: int  # Microjoules cumulative
    power_w: float
    temp_c: float
    util_pct: float


class AMDEnergyMeter:
    """
    Real energy measurement from AMD GPU using sysfs, rocm-smi, or amd-smi.

    Uses power * time integration since APUs don't have energy counters.
    """

    def __init__(self, device_id: int = 0):
        self.device_id = device_id
        self.hwmon_path = self._find_hwmon_path()
        self.tool = self._detect_tool()
        self._prev_energy_uj = 0
        self._prev_time = 0.0
        print(f"AMD Energy Meter initialized: hwmon={self.hwmon_path}, tool={self.tool}")

    def _find_hwmon_path(self) -> Optional[str]:
        """Find the hwmon path for the GPU."""
        import glob
        # Try card0 first (typically the GPU)
        patterns = [
            "/sys/class/drm/card0/device/hwmon/hwmon*/",
            "/sys/class/drm/card1/device/hwmon/hwmon*/",
            "/sys/class/hwmon/hwmon*/",
        ]
        for pattern in patterns:
            matches = glob.glob(pattern)
            for path in matches:
                if os.path.exists(os.path.join(path, "power1_average")):
                    return path.rstrip('/')
        return None

    def _detect_tool(self) -> str:
        """Detect available AMD tool."""
        # Try amd-smi first (newer)
        try:
            result = subprocess.run(['amd-smi', 'version'], capture_output=True, timeout=5)
            if result.returncode == 0:
                return 'amd-smi'
        except:
            pass

        # Fall back to rocm-smi
        try:
            result = subprocess.run(['rocm-smi', '--version'], capture_output=True, timeout=5)
            if result.returncode == 0:
                return 'rocm-smi'
        except:
            pass

        return 'none'

    def get_reading(self) -> EnergyReading:
        """Get current energy reading."""
        # Prefer sysfs (most reliable for APUs)
        if self.hwmon_path:
            return self._read_sysfs()
        elif self.tool == 'rocm-smi':
            return self._read_rocm_smi()
        elif self.tool == 'amd-smi':
            return self._read_amd_smi()
        else:
            return self._read_fallback()

    def _read_sysfs(self) -> EnergyReading:
        """Read directly from sysfs (most reliable)."""
        try:
            # Power (microwatts -> watts)
            power_path = os.path.join(self.hwmon_path, "power1_average")
            if os.path.exists(power_path):
                with open(power_path) as f:
                    power_uw = int(f.read().strip())
                power_w = power_uw / 1e6
            else:
                power_w = 0.0

            # Temperature (millicelsius -> celsius)
            temp_path = os.path.join(self.hwmon_path, "temp1_input")
            if os.path.exists(temp_path):
                with open(temp_path) as f:
                    temp_mc = int(f.read().strip())
                temp_c = temp_mc / 1000.0
            else:
                temp_c = 0.0

            # Frequency (Hz -> MHz)
            freq_path = os.path.join(self.hwmon_path, "freq1_input")
            if os.path.exists(freq_path):
                with open(freq_path) as f:
                    freq_hz = int(f.read().strip())
                freq_mhz = freq_hz / 1e6
            else:
                freq_mhz = 0.0

            return EnergyReading(
                timestamp=time.time(),
                energy_uj=0,  # No energy counter on APUs
                power_w=power_w,
                temp_c=temp_c,
                util_pct=0.0  # Not available in sysfs
            )
        except Exception as e:
            print(f"sysfs read error: {e}")
            return self._read_fallback()

    def _read_amd_smi(self) -> EnergyReading:
        """Read from amd-smi."""
        try:
            # Get power and energy
            result = subprocess.run(
                ['amd-smi', 'metric', '-g', str(self.device_id), '--json'],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                data = json.loads(result.stdout)
                gpu_data = data.get('gpu', [{}])[0] if isinstance(data.get('gpu'), list) else data

                # Extract metrics
                power_w = float(gpu_data.get('power', {}).get('socket_power', 0) or
                               gpu_data.get('power_usage', 0) or 0)
                temp_c = float(gpu_data.get('temperature', {}).get('edge', 0) or
                              gpu_data.get('temperature', 0) or 0)
                util = float(gpu_data.get('utilization', {}).get('gfx', 0) or
                            gpu_data.get('gpu_use', 0) or 0)

                # Energy counter (if available)
                energy_uj = int(gpu_data.get('energy', {}).get('total', 0) or 0)

                return EnergyReading(
                    timestamp=time.time(),
                    energy_uj=energy_uj,
                    power_w=power_w,
                    temp_c=temp_c,
                    util_pct=util
                )
        except Exception as e:
            print(f"amd-smi error: {e}")

        return self._read_fallback()

    def _read_rocm_smi(self) -> EnergyReading:
        """Read from rocm-smi."""
        try:
            # Power
            power_result = subprocess.run(
                ['rocm-smi', '-d', str(self.device_id), '--showpower'],
                capture_output=True, text=True, timeout=5
            )
            power_w = 0.0
            for line in power_result.stdout.split('\n'):
                if 'Average' in line or 'Power' in line:
                    parts = line.split()
                    for i, p in enumerate(parts):
                        if p.replace('.', '').isdigit():
                            power_w = float(p)
                            break

            # Temperature
            temp_result = subprocess.run(
                ['rocm-smi', '-d', str(self.device_id), '--showtemp'],
                capture_output=True, text=True, timeout=5
            )
            temp_c = 0.0
            for line in temp_result.stdout.split('\n'):
                if 'edge' in line.lower() or 'Temperature' in line:
                    parts = line.split()
                    for p in parts:
                        try:
                            temp_c = float(p.replace('c', '').replace('C', ''))
                            break
                        except:
                            continue

            # Utilization
            util_result = subprocess.run(
                ['rocm-smi', '-d', str(self.device_id), '--showuse'],
                capture_output=True, text=True, timeout=5
            )
            util_pct = 0.0
            for line in util_result.stdout.split('\n'):
                if 'GPU use' in line:
                    parts = line.split()
                    for p in parts:
                        if '%' in p:
                            util_pct = float(p.replace('%', ''))
                            break

            return EnergyReading(
                timestamp=time.time(),
                energy_uj=0,  # rocm-smi doesn't have energy counter
                power_w=power_w,
                temp_c=temp_c,
                util_pct=util_pct
            )
        except Exception as e:
            print(f"rocm-smi error: {e}")

        return self._read_fallback()

    def _read_fallback(self) -> EnergyReading:
        """Fallback when no tool available."""
        return EnergyReading(
            timestamp=time.time(),
            energy_uj=0,
            power_w=0.0,
            temp_c=0.0,
            util_pct=0.0
        )

    def measure_interval(self, duration_s: float = None) -> Tuple[float, float]:
        """
        Measure energy over an interval.

        Returns:
            energy_mj: Total energy in millijoules
            avg_power_w: Average power in watts
        """
        start_reading = self.get_reading()
        start_time = time.time()

        if duration_s:
            time.sleep(duration_s)

        end_reading = self.get_reading()
        end_time = time.time()

        dt = max(end_time - start_time, 0.001)

        # If we have energy counter, use ΔE
        if start_reading.energy_uj > 0 and end_reading.energy_uj > 0:
            delta_uj = end_reading.energy_uj - start_reading.energy_uj
            energy_mj = delta_uj / 1000.0
            avg_power_w = (delta_uj / 1e6) / dt
        else:
            # Fall back to power * time
            avg_power_w = (start_reading.power_w + end_reading.power_w) / 2
            energy_mj = avg_power_w * dt * 1000

        return energy_mj, avg_power_w


@dataclass
class BenchmarkResult:
    """Result from a single benchmark run."""
    condition: str  # "baseline" or "feel"
    tokens_generated: int
    latency_ms: float
    energy_mj: float
    mj_per_token: float
    avg_power_w: float
    tokens_per_second: float


@dataclass
class BenchmarkSummary:
    """Statistical summary of benchmark results."""
    condition: str
    n_runs: int
    mj_per_token_mean: float
    mj_per_token_std: float
    mj_per_token_ci95: Tuple[float, float]
    tokens_per_sec_mean: float
    latency_ms_mean: float
    power_w_mean: float


def compute_ci95(values: List[float]) -> Tuple[float, float]:
    """Compute 95% confidence interval."""
    if len(values) < 2:
        return (values[0], values[0]) if values else (0, 0)

    mean = statistics.mean(values)
    std = statistics.stdev(values)
    n = len(values)

    # t-value for 95% CI
    t = 2.0 if n < 30 else 1.96
    margin = t * std / math.sqrt(n)

    return (mean - margin, mean + margin)


def run_benchmark(
    output_dir: str = "results/z106_benchmark",
    model_size: str = "tiny",
    n_runs: int = 10,
    warmup_runs: int = 3,
    max_tokens: int = 100,
    prompt_length: int = 32,
    device: str = "auto",
):
    """
    Run comprehensive A/B benchmark.

    Args:
        output_dir: Where to save results
        model_size: tiny/small/medium
        n_runs: Number of measurement runs per condition
        warmup_runs: Warmup runs (not counted)
        max_tokens: Tokens to generate per run
        prompt_length: Input prompt length
        device: Device to use
    """
    from src.feel_slm import FEELConfig, FEELSLM, BaselineSLM

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Setup device
    if device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)

    print("=" * 70)
    print("FEEL-SLM Real Energy Benchmark")
    print("=" * 70)
    print(f"Model size: {model_size}")
    print(f"Device: {device}")
    print(f"Runs: {n_runs} (+ {warmup_runs} warmup)")
    print(f"Tokens per run: {max_tokens}")

    # Initialize energy meter
    meter = AMDEnergyMeter()

    # Get baseline reading
    baseline_reading = meter.get_reading()
    print(f"\nBaseline reading:")
    print(f"  Power: {baseline_reading.power_w:.1f}W")
    print(f"  Temp: {baseline_reading.temp_c:.1f}C")
    print(f"  Util: {baseline_reading.util_pct:.1f}%")

    # Create models
    config_fn = {"tiny": FEELConfig.tiny, "small": FEELConfig.small, "medium": FEELConfig.medium}
    config = config_fn.get(model_size, FEELConfig.tiny)()

    print(f"\nLoading models...")
    baseline_model = BaselineSLM(config).to(device).eval()
    feel_model = FEELSLM(config).to(device).eval()

    baseline_params = sum(p.numel() for p in baseline_model.parameters())
    feel_params = sum(p.numel() for p in feel_model.parameters())
    print(f"  Baseline: {baseline_params:,} params")
    print(f"  FEEL: {feel_params:,} params (+{100*(feel_params-baseline_params)/baseline_params:.1f}%)")

    # Create test prompts (same for both models)
    prompts = [
        torch.randint(0, config.vocab_size, (1, prompt_length), device=device)
        for _ in range(n_runs + warmup_runs)
    ]

    # Telemetry for FEEL model
    telemetry_samples = [
        torch.rand(1, config.body_dim, device=device)
        for _ in range(n_runs + warmup_runs)
    ]

    results: Dict[str, List[BenchmarkResult]] = {"baseline": [], "feel": []}

    # === Warmup ===
    print(f"\nWarming up ({warmup_runs} runs per model)...")
    for i in range(warmup_runs):
        with torch.no_grad():
            _ = baseline_model.generate(prompts[i], max_new_tokens=10)
            _ = feel_model.generate(prompts[i], telemetry_samples[i], max_new_tokens=10)

    torch.cuda.synchronize() if torch.cuda.is_available() else None
    time.sleep(2)  # Let GPU settle

    # === Benchmark Baseline ===
    print(f"\nBenchmarking Baseline ({n_runs} runs)...")
    for i in range(n_runs):
        prompt = prompts[warmup_runs + i]

        # Start measurement
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        start_reading = meter.get_reading()
        start_time = time.perf_counter()

        # Generate
        with torch.no_grad():
            output = baseline_model.generate(prompt, max_new_tokens=max_tokens)

        # End measurement
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        end_time = time.perf_counter()
        end_reading = meter.get_reading()

        # Compute metrics
        tokens_generated = output.shape[1] - prompt.shape[1]
        latency_ms = (end_time - start_time) * 1000

        dt = end_time - start_time
        if start_reading.energy_uj > 0 and end_reading.energy_uj > 0:
            energy_mj = (end_reading.energy_uj - start_reading.energy_uj) / 1000.0
            avg_power_w = (end_reading.energy_uj - start_reading.energy_uj) / 1e6 / dt
        else:
            avg_power_w = (start_reading.power_w + end_reading.power_w) / 2
            energy_mj = avg_power_w * dt * 1000

        mj_per_token = energy_mj / max(tokens_generated, 1)
        tps = tokens_generated / dt if dt > 0 else 0

        result = BenchmarkResult(
            condition="baseline",
            tokens_generated=tokens_generated,
            latency_ms=latency_ms,
            energy_mj=energy_mj,
            mj_per_token=mj_per_token,
            avg_power_w=avg_power_w,
            tokens_per_second=tps
        )
        results["baseline"].append(result)

        print(f"  [{i+1}/{n_runs}] {tokens_generated} tok, {mj_per_token:.1f} mJ/tok, "
              f"{avg_power_w:.0f}W, {tps:.1f} tok/s")

        time.sleep(0.5)  # Brief pause between runs

    # === Benchmark FEEL ===
    print(f"\nBenchmarking FEEL ({n_runs} runs)...")
    for i in range(n_runs):
        prompt = prompts[warmup_runs + i]
        telemetry = telemetry_samples[warmup_runs + i]

        # Start measurement
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        start_reading = meter.get_reading()
        start_time = time.perf_counter()

        # Generate
        with torch.no_grad():
            output, gate_history = feel_model.generate(prompt, telemetry, max_new_tokens=max_tokens)

        # End measurement
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        end_time = time.perf_counter()
        end_reading = meter.get_reading()

        # Compute metrics
        tokens_generated = output.shape[1] - prompt.shape[1]
        latency_ms = (end_time - start_time) * 1000

        dt = end_time - start_time
        if start_reading.energy_uj > 0 and end_reading.energy_uj > 0:
            energy_mj = (end_reading.energy_uj - start_reading.energy_uj) / 1000.0
            avg_power_w = (end_reading.energy_uj - start_reading.energy_uj) / 1e6 / dt
        else:
            avg_power_w = (start_reading.power_w + end_reading.power_w) / 2
            energy_mj = avg_power_w * dt * 1000

        mj_per_token = energy_mj / max(tokens_generated, 1)
        tps = tokens_generated / dt if dt > 0 else 0

        result = BenchmarkResult(
            condition="feel",
            tokens_generated=tokens_generated,
            latency_ms=latency_ms,
            energy_mj=energy_mj,
            mj_per_token=mj_per_token,
            avg_power_w=avg_power_w,
            tokens_per_second=tps
        )
        results["feel"].append(result)

        # Gate info
        avg_gate = sum(gate_history) / len(gate_history) if gate_history else 0

        print(f"  [{i+1}/{n_runs}] {tokens_generated} tok, {mj_per_token:.1f} mJ/tok, "
              f"{avg_power_w:.0f}W, {tps:.1f} tok/s, gate={avg_gate:.3f}")

        time.sleep(0.5)

    # === Compute Summaries ===
    summaries = {}
    for condition, res_list in results.items():
        mj_values = [r.mj_per_token for r in res_list]
        tps_values = [r.tokens_per_second for r in res_list]
        lat_values = [r.latency_ms for r in res_list]
        pow_values = [r.avg_power_w for r in res_list]

        summaries[condition] = BenchmarkSummary(
            condition=condition,
            n_runs=len(res_list),
            mj_per_token_mean=statistics.mean(mj_values),
            mj_per_token_std=statistics.stdev(mj_values) if len(mj_values) > 1 else 0,
            mj_per_token_ci95=compute_ci95(mj_values),
            tokens_per_sec_mean=statistics.mean(tps_values),
            latency_ms_mean=statistics.mean(lat_values),
            power_w_mean=statistics.mean(pow_values)
        )

    # === Print Summary ===
    print("\n" + "=" * 70)
    print("BENCHMARK RESULTS")
    print("=" * 70)

    print(f"\n{'Condition':<12} {'n':>4} {'mJ/tok':>10} {'95% CI':>20} {'tok/s':>8} {'Power':>8}")
    print("-" * 70)

    for condition, s in summaries.items():
        ci_str = f"[{s.mj_per_token_ci95[0]:.1f}, {s.mj_per_token_ci95[1]:.1f}]"
        print(f"{s.condition:<12} {s.n_runs:>4} {s.mj_per_token_mean:>10.1f} "
              f"{ci_str:>20} {s.tokens_per_sec_mean:>8.1f} {s.power_w_mean:>7.0f}W")

    # === Compute Comparison ===
    if "baseline" in summaries and "feel" in summaries:
        base_mj = summaries["baseline"].mj_per_token_mean
        feel_mj = summaries["feel"].mj_per_token_mean

        if base_mj > 0:
            savings_pct = (base_mj - feel_mj) / base_mj * 100

            # Check statistical significance (simple t-test approximation)
            base_values = [r.mj_per_token for r in results["baseline"]]
            feel_values = [r.mj_per_token for r in results["feel"]]

            # Paired difference test
            if len(base_values) == len(feel_values):
                diffs = [b - f for b, f in zip(base_values, feel_values)]
                diff_mean = statistics.mean(diffs)
                diff_std = statistics.stdev(diffs) if len(diffs) > 1 else 0

                # t-statistic
                if diff_std > 0:
                    t_stat = diff_mean / (diff_std / math.sqrt(len(diffs)))
                else:
                    t_stat = 0

                # Approximate p-value (two-tailed)
                # For df > 30, t ~ N(0,1)
                significant = abs(t_stat) > 2.0  # ~95% confidence
            else:
                significant = False
                t_stat = 0

            print(f"\n{'='*70}")
            print(f"COMPARISON: FEEL vs Baseline")
            print(f"{'='*70}")
            print(f"  Energy savings: {savings_pct:+.1f}%")
            print(f"  Baseline: {base_mj:.1f} mJ/tok")
            print(f"  FEEL:     {feel_mj:.1f} mJ/tok")
            print(f"  Difference: {base_mj - feel_mj:.1f} mJ/tok")
            print(f"  t-statistic: {t_stat:.2f}")
            print(f"  Significant (p<0.05): {'YES' if significant else 'NO'}")

    # === Save Results ===
    results_data = {
        "config": {
            "model_size": model_size,
            "n_runs": n_runs,
            "warmup_runs": warmup_runs,
            "max_tokens": max_tokens,
            "prompt_length": prompt_length,
            "device": str(device),
            "baseline_params": baseline_params,
            "feel_params": feel_params,
        },
        "results": {
            condition: [asdict(r) for r in res_list]
            for condition, res_list in results.items()
        },
        "summaries": {
            condition: asdict(s) for condition, s in summaries.items()
        },
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    results_file = output_path / "benchmark_results.json"
    with open(results_file, 'w') as f:
        json.dump(results_data, f, indent=2)
    print(f"\nResults saved to {results_file}")

    # === Generate LaTeX Table ===
    latex_file = output_path / "benchmark_table.tex"
    with open(latex_file, 'w') as f:
        f.write("\\begin{table}[h]\n")
        f.write("\\centering\n")
        f.write("\\caption{FEEL-SLM Energy Benchmark Results}\n")
        f.write("\\begin{tabular}{lrrrr}\n")
        f.write("\\toprule\n")
        f.write("Model & mJ/tok & 95\\% CI & tok/s & Power (W) \\\\\n")
        f.write("\\midrule\n")
        for condition, s in summaries.items():
            ci_str = f"[{s.mj_per_token_ci95[0]:.1f}, {s.mj_per_token_ci95[1]:.1f}]"
            f.write(f"{condition.capitalize()} & {s.mj_per_token_mean:.1f} & {ci_str} & "
                   f"{s.tokens_per_sec_mean:.1f} & {s.power_w_mean:.0f} \\\\\n")
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n")
    print(f"LaTeX table saved to {latex_file}")

    return results, summaries


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FEEL-SLM Real Energy Benchmark")
    parser.add_argument("--output-dir", default="results/z106_benchmark")
    parser.add_argument("--model-size", choices=["tiny", "small", "medium"], default="tiny")
    parser.add_argument("--n-runs", type=int, default=10)
    parser.add_argument("--warmup-runs", type=int, default=3)
    parser.add_argument("--max-tokens", type=int, default=100)
    parser.add_argument("--prompt-length", type=int, default=32)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    run_benchmark(
        output_dir=args.output_dir,
        model_size=args.model_size,
        n_runs=args.n_runs,
        warmup_runs=args.warmup_runs,
        max_tokens=args.max_tokens,
        prompt_length=args.prompt_length,
        device=args.device,
    )
