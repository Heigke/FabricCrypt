#!/usr/bin/env python3
"""FEEL-SLM Real Energy Benchmark for NVIDIA GPUs.

Uses nvidia-smi for power measurement with proper A/B testing methodology.
"""

import os
import sys
import json
import time
import subprocess
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import List, Optional, Tuple
from pathlib import Path

import torch
import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from feel_slm import FEELConfig, FEELSLM, BaselineSLM
from feel_slm.body_encoder import TelemetrySnapshot


@dataclass
class EnergyReading:
    """Single energy measurement reading."""
    timestamp: float
    power_w: float
    temp_c: float
    util_pct: float
    memory_used_mb: float


@dataclass
class BenchmarkRun:
    """Result from a single benchmark run."""
    condition: str
    tokens_generated: int
    latency_ms: float
    energy_mj: float
    mj_per_token: float
    avg_power_w: float
    tokens_per_second: float


class NVIDIAEnergyMeter:
    """Energy measurement using nvidia-smi."""

    def __init__(self, device_idx: int = 0, sample_interval_ms: int = 50):
        self.device_idx = device_idx
        self.sample_interval = sample_interval_ms / 1000.0
        self.readings: List[EnergyReading] = []
        self._sampling = False

    def _query_nvidia_smi(self) -> Optional[EnergyReading]:
        """Query nvidia-smi for current power and metrics."""
        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    f"--id={self.device_idx}",
                    "--query-gpu=power.draw,temperature.gpu,utilization.gpu,memory.used",
                    "--format=csv,noheader,nounits"
                ],
                capture_output=True,
                text=True,
                timeout=1.0
            )

            if result.returncode == 0:
                parts = result.stdout.strip().split(", ")
                if len(parts) >= 4:
                    power_w = float(parts[0]) if parts[0] != "[N/A]" else 0.0
                    temp_c = float(parts[1]) if parts[1] != "[N/A]" else 0.0
                    util_pct = float(parts[2]) if parts[2] != "[N/A]" else 0.0
                    memory_mb = float(parts[3]) if parts[3] != "[N/A]" else 0.0

                    return EnergyReading(
                        timestamp=time.time(),
                        power_w=power_w,
                        temp_c=temp_c,
                        util_pct=util_pct,
                        memory_used_mb=memory_mb
                    )
        except Exception as e:
            print(f"nvidia-smi query failed: {e}")

        return None

    def start_sampling(self):
        """Start continuous power sampling in background."""
        self.readings = []
        self._sampling = True

    def sample_once(self):
        """Take a single sample."""
        if self._sampling:
            reading = self._query_nvidia_smi()
            if reading:
                self.readings.append(reading)

    def stop_sampling(self) -> Tuple[float, float]:
        """Stop sampling and return (total_energy_mj, avg_power_w)."""
        self._sampling = False

        if len(self.readings) < 2:
            return 0.0, 0.0

        total_energy_mj = 0.0
        powers = []

        for i in range(1, len(self.readings)):
            dt = self.readings[i].timestamp - self.readings[i-1].timestamp
            avg_power = (self.readings[i].power_w + self.readings[i-1].power_w) / 2
            total_energy_mj += avg_power * dt * 1000  # W * s * 1000 = mJ
            powers.append(avg_power)

        avg_power_w = np.mean(powers) if powers else 0.0
        return total_energy_mj, avg_power_w


def run_single_benchmark(
    model: torch.nn.Module,
    condition: str,
    input_ids: torch.Tensor,
    max_tokens: int,
    meter: NVIDIAEnergyMeter,
    device: torch.device,
    telemetry: Optional[torch.Tensor] = None
) -> BenchmarkRun:
    """Run a single benchmark measurement."""

    # Warmup
    torch.cuda.synchronize()

    # Start measurement
    meter.start_sampling()
    meter.sample_once()  # Initial sample

    start_time = time.perf_counter()

    # Generation loop with power sampling
    generated_ids = input_ids.clone()
    tokens_generated = 0

    with torch.no_grad():
        for _ in range(max_tokens):
            # Sample power periodically
            meter.sample_once()

            if telemetry is not None:
                # FEEL model - pass telemetry tensor
                outputs = model(generated_ids, telemetry)
            else:
                # Baseline model
                outputs = model(generated_ids)

            if isinstance(outputs, dict):
                logits = outputs["logits"]
            else:
                logits = outputs

            next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            generated_ids = torch.cat([generated_ids, next_token], dim=1)
            tokens_generated += 1

    torch.cuda.synchronize()
    end_time = time.perf_counter()

    # Final sample and stop
    meter.sample_once()
    total_energy_mj, avg_power_w = meter.stop_sampling()

    latency_ms = (end_time - start_time) * 1000
    tokens_per_sec = tokens_generated / (latency_ms / 1000)
    mj_per_token = total_energy_mj / tokens_generated if tokens_generated > 0 else 0

    return BenchmarkRun(
        condition=condition,
        tokens_generated=tokens_generated,
        latency_ms=latency_ms,
        energy_mj=total_energy_mj,
        mj_per_token=mj_per_token,
        avg_power_w=avg_power_w,
        tokens_per_second=tokens_per_sec
    )


def compute_summary_stats(runs: List[BenchmarkRun]) -> dict:
    """Compute summary statistics with 95% CI."""
    mj_values = [r.mj_per_token for r in runs]
    tps_values = [r.tokens_per_second for r in runs]
    latency_values = [r.latency_ms for r in runs]
    power_values = [r.avg_power_w for r in runs]

    n = len(mj_values)
    mj_mean = np.mean(mj_values)
    mj_std = np.std(mj_values, ddof=1)
    mj_se = mj_std / np.sqrt(n)
    t_crit = stats.t.ppf(0.975, n - 1)
    mj_ci = (mj_mean - t_crit * mj_se, mj_mean + t_crit * mj_se)

    return {
        "condition": runs[0].condition,
        "n_runs": n,
        "mj_per_token_mean": float(mj_mean),
        "mj_per_token_std": float(mj_std),
        "mj_per_token_ci95": [float(mj_ci[0]), float(mj_ci[1])],
        "tokens_per_sec_mean": float(np.mean(tps_values)),
        "latency_ms_mean": float(np.mean(latency_values)),
        "power_w_mean": float(np.mean(power_values))
    }


def run_benchmark(
    model_size: str = "tiny",
    n_runs: int = 10,
    warmup_runs: int = 3,
    max_tokens: int = 100,
    prompt_length: int = 32,
    output_dir: str = "results/z107_nvidia_benchmark"
):
    """Run full A/B benchmark comparison."""

    print("=" * 70)
    print("FEEL-SLM NVIDIA Energy Benchmark")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Create models
    print("\nCreating models...")

    if model_size == "tiny":
        config = FEELConfig(
            vocab_size=32000,
            hidden_dim=256,
            num_layers=4,
            num_heads=4,
            num_kv_heads=2,
            head_dim=64,
            intermediate_dim=512,
            max_seq_len=512
        )
    else:
        config = FEELConfig()

    baseline_model = BaselineSLM(config).to(device).eval()
    feel_model = FEELSLM(config).to(device).eval()

    baseline_params = sum(p.numel() for p in baseline_model.parameters())
    feel_params = sum(p.numel() for p in feel_model.parameters())

    print(f"Baseline parameters: {baseline_params:,}")
    print(f"FEEL parameters: {feel_params:,}")
    print(f"Overhead: {(feel_params - baseline_params) / baseline_params * 100:.1f}%")

    # Create input
    input_ids = torch.randint(0, config.vocab_size, (1, prompt_length), device=device)

    # Create body state for FEEL and convert to tensor
    telemetry_tensor = TelemetrySnapshot(
        power_watts=50.0,
        temp_c=45.0,
        utilization=0.5,
        clock_mhz=1800.0,
        mem_util=0.25,
        mem_used_gb=2.0,
        throttle_state=False,
        fan_speed=0.5,
        profile="balanced"
    )
    # Convert to tensor: [1, body_dim]
    telemetry_tensor = telemetry_tensor.to_vector().unsqueeze(0).to(device)

    # Initialize energy meter
    meter = NVIDIAEnergyMeter(device_idx=0, sample_interval_ms=50)

    # Test power reading
    test_reading = meter._query_nvidia_smi()
    if test_reading is None or test_reading.power_w == 0:
        print("\nWARNING: Cannot read GPU power via nvidia-smi")
        print("Benchmark will still measure latency and throughput")
    else:
        print(f"\nPower reading OK: {test_reading.power_w:.1f}W")

    # Warmup runs
    print(f"\nRunning {warmup_runs} warmup iterations...")
    for i in range(warmup_runs):
        with torch.no_grad():
            _ = baseline_model(input_ids)
            _ = feel_model(input_ids, telemetry_tensor)
        torch.cuda.synchronize()

    # Benchmark runs - interleaved A/B design
    print(f"\nRunning {n_runs} benchmark iterations (interleaved A/B)...")

    baseline_runs: List[BenchmarkRun] = []
    feel_runs: List[BenchmarkRun] = []

    for i in range(n_runs):
        print(f"  Run {i+1}/{n_runs}...", end=" ", flush=True)

        # Alternate starting condition to reduce order effects
        if i % 2 == 0:
            # Baseline first
            baseline_run = run_single_benchmark(
                baseline_model, "baseline", input_ids, max_tokens, meter, device
            )
            baseline_runs.append(baseline_run)

            feel_run = run_single_benchmark(
                feel_model, "feel", input_ids, max_tokens, meter, device, telemetry_tensor
            )
            feel_runs.append(feel_run)
        else:
            # FEEL first
            feel_run = run_single_benchmark(
                feel_model, "feel", input_ids, max_tokens, meter, device, telemetry_tensor
            )
            feel_runs.append(feel_run)

            baseline_run = run_single_benchmark(
                baseline_model, "baseline", input_ids, max_tokens, meter, device
            )
            baseline_runs.append(baseline_run)

        print(f"baseline={baseline_run.mj_per_token:.1f} mJ/tok, "
              f"feel={feel_run.mj_per_token:.1f} mJ/tok")

    # Compute statistics
    baseline_summary = compute_summary_stats(baseline_runs)
    feel_summary = compute_summary_stats(feel_runs)

    # Print results
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)

    print(f"\n{'Condition':<12} {'n':>5} {'mJ/tok':>10} {'95% CI':>20} {'tok/s':>8} {'Power':>8}")
    print("-" * 70)

    for summary in [baseline_summary, feel_summary]:
        ci_str = f"[{summary['mj_per_token_ci95'][0]:.1f}, {summary['mj_per_token_ci95'][1]:.1f}]"
        print(f"{summary['condition']:<12} {summary['n_runs']:>5} "
              f"{summary['mj_per_token_mean']:>10.1f} {ci_str:>20} "
              f"{summary['tokens_per_sec_mean']:>8.1f} "
              f"{summary['power_w_mean']:>7.0f}W")

    # Statistical comparison
    baseline_mj = [r.mj_per_token for r in baseline_runs]
    feel_mj = [r.mj_per_token for r in feel_runs]

    t_stat, p_value = stats.ttest_ind(baseline_mj, feel_mj)
    energy_savings = (baseline_summary["mj_per_token_mean"] - feel_summary["mj_per_token_mean"]) / baseline_summary["mj_per_token_mean"] * 100

    print(f"\nCOMPARISON: FEEL vs Baseline")
    print(f"  Energy savings: {energy_savings:+.1f}%")
    print(f"  t-statistic: {t_stat:.2f}")
    print(f"  Significant (p<0.05): {'YES' if abs(t_stat) > 2.0 else 'NO'}")

    # Save results
    os.makedirs(output_dir, exist_ok=True)

    results = {
        "config": {
            "model_size": model_size,
            "n_runs": n_runs,
            "warmup_runs": warmup_runs,
            "max_tokens": max_tokens,
            "prompt_length": prompt_length,
            "device": str(device),
            "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A",
            "baseline_params": baseline_params,
            "feel_params": feel_params
        },
        "results": {
            "baseline": [asdict(r) for r in baseline_runs],
            "feel": [asdict(r) for r in feel_runs]
        },
        "summaries": {
            "baseline": baseline_summary,
            "feel": feel_summary
        },
        "comparison": {
            "energy_savings_pct": float(energy_savings),
            "t_statistic": float(t_stat),
            "significant": bool(abs(t_stat) > 2.0)
        },
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

    results_path = os.path.join(output_dir, "benchmark_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {results_path}")

    # Generate LaTeX table
    latex = r"""\begin{table}[h]
\centering
\caption{FEEL-SLM NVIDIA Energy Benchmark Results}
\begin{tabular}{lrrrr}
\toprule
Model & mJ/tok & 95\% CI & tok/s & Power (W) \\
\midrule
"""
    for summary in [baseline_summary, feel_summary]:
        ci_str = f"[{summary['mj_per_token_ci95'][0]:.1f}, {summary['mj_per_token_ci95'][1]:.1f}]"
        latex += f"{summary['condition'].title()} & {summary['mj_per_token_mean']:.1f} & {ci_str} & {summary['tokens_per_sec_mean']:.1f} & {summary['power_w_mean']:.0f} \\\\\n"
    latex += r"""\bottomrule
\end{tabular}
\end{table}
"""

    table_path = os.path.join(output_dir, "benchmark_table.tex")
    with open(table_path, "w") as f:
        f.write(latex)
    print(f"LaTeX table saved to: {table_path}")

    return results


if __name__ == "__main__":
    run_benchmark()
