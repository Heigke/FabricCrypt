#!/usr/bin/env python3
"""
Real A/B Benchmark for FEEL-SLM v2

Proper statistical comparison between:
- Baseline (fixed eco, balanced, perf)
- FEEL (fixed eco, balanced, perf)
- FEEL (adaptive mode)

Features:
- Real telemetry via sysfs/NVML (no subprocess)
- Interleaved A/B design to reduce order effects
- 95% confidence intervals
- Statistical significance testing (t-test)
- Energy from power integration at 100Hz
"""

import os
import sys
import json
import time
import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict, field

import torch
import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from feel_slm.telemetry_source import create_telemetry_source, TelemetrySampler
from feel_slm.model_v2 import FEELConfigV2, FEELSLMV2, BaselineSLMV2


# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class BenchmarkRun:
    """Single benchmark run."""
    condition: str
    mode: str
    run_idx: int
    tokens_generated: int
    duration_s: float
    energy_mj: float
    mj_per_token: float
    tokens_per_second: float
    avg_power_w: float
    max_power_w: float
    avg_temp_c: float
    n_samples: int


@dataclass
class ConditionSummary:
    """Summary statistics for a condition."""
    condition: str
    mode: str
    n_runs: int
    mj_per_token_mean: float
    mj_per_token_std: float
    mj_per_token_ci95: Tuple[float, float]
    tokens_per_sec_mean: float
    tokens_per_sec_std: float
    power_w_mean: float
    temp_c_mean: float


# =============================================================================
# Benchmark Runner
# =============================================================================

def run_single_benchmark(
    model: torch.nn.Module,
    sampler: TelemetrySampler,
    condition: str,
    mode: str,
    run_idx: int,
    num_tokens: int,
    device: torch.device,
    control_window: int = 32,
) -> BenchmarkRun:
    """Run a single benchmark iteration."""

    is_feel = hasattr(model, 'body_encoder')

    # Set mode
    if is_feel:
        model.set_mode(mode if mode != "adaptive" else "balanced")

    # Clear and prepare
    sampler.buffer.clear()
    torch.cuda.synchronize() if device.type == "cuda" else None

    # Input
    input_ids = torch.randint(0, 32000, (1, 32), device=device)
    generated_ids = input_ids.clone()

    # Timing
    start_time = time.perf_counter()
    tokens_since_control = 0

    # Generate tokens
    for i in range(num_tokens):
        # Adaptive control for FEEL
        if is_feel and mode == "adaptive" and tokens_since_control >= control_window:
            tokens_since_control = 0
            window = sampler.get_window(0.2)
            if window and window.n_samples > 2:
                telemetry = torch.tensor([
                    window.power_w_mean / 300.0,
                    window.temp_c_mean / 100.0,
                    window.gpu_util_mean / 100.0,
                    0.5, 0.5, 0.5,
                    window.power_delta / 300.0,
                    window.temp_delta / 100.0,
                    window.util_delta / 100.0,
                    0.0, 0.5, 0.5,
                ], device=device, dtype=torch.float32).unsqueeze(0)

                with torch.no_grad():
                    body_embed = model.body_encoder(telemetry)
                    policy = model.policy_head(body_embed)
                    profile_idx = policy["profile_idx"].item()
                    modes = ["eco", "balanced", "perf"]
                    model.set_mode(modes[profile_idx])

        # Forward pass
        with torch.no_grad():
            if is_feel:
                latest = sampler.get_latest()
                if latest:
                    telemetry = torch.tensor(
                        latest.to_vector(), device=device, dtype=torch.float32
                    ).unsqueeze(0)
                else:
                    telemetry = torch.zeros(1, 12, device=device)
                outputs = model(generated_ids, telemetry)
            else:
                outputs = model(generated_ids)

            logits = outputs["logits"][:, -1, :]
            next_token = logits.argmax(dim=-1, keepdim=True)

        generated_ids = torch.cat([generated_ids, next_token], dim=1)
        tokens_since_control += 1

    torch.cuda.synchronize() if device.type == "cuda" else None
    end_time = time.perf_counter()
    duration_s = end_time - start_time

    # Get telemetry window
    window = sampler.buffer.aggregate_window(duration_s)

    energy_mj = window.energy_mj if window else 0.0
    avg_power = window.power_w_mean if window else 0.0
    max_power = window.power_w_max if window else 0.0
    avg_temp = window.temp_c_mean if window else 0.0
    n_samples = window.n_samples if window else 0

    return BenchmarkRun(
        condition=condition,
        mode=mode,
        run_idx=run_idx,
        tokens_generated=num_tokens,
        duration_s=duration_s,
        energy_mj=energy_mj,
        mj_per_token=energy_mj / num_tokens if num_tokens > 0 else 0,
        tokens_per_second=num_tokens / duration_s,
        avg_power_w=avg_power,
        max_power_w=max_power,
        avg_temp_c=avg_temp,
        n_samples=n_samples,
    )


def compute_summary(runs: List[BenchmarkRun]) -> ConditionSummary:
    """Compute summary statistics with 95% CI."""
    mj_values = [r.mj_per_token for r in runs]
    tps_values = [r.tokens_per_second for r in runs]
    power_values = [r.avg_power_w for r in runs]
    temp_values = [r.avg_temp_c for r in runs]

    n = len(mj_values)
    mj_mean = np.mean(mj_values)
    mj_std = np.std(mj_values, ddof=1) if n > 1 else 0.0
    mj_se = mj_std / np.sqrt(n) if n > 1 else 0.0
    t_crit = stats.t.ppf(0.975, n - 1) if n > 1 else 0.0
    mj_ci = (mj_mean - t_crit * mj_se, mj_mean + t_crit * mj_se)

    return ConditionSummary(
        condition=runs[0].condition,
        mode=runs[0].mode,
        n_runs=n,
        mj_per_token_mean=float(mj_mean),
        mj_per_token_std=float(mj_std),
        mj_per_token_ci95=(float(mj_ci[0]), float(mj_ci[1])),
        tokens_per_sec_mean=float(np.mean(tps_values)),
        tokens_per_sec_std=float(np.std(tps_values, ddof=1)) if n > 1 else 0.0,
        power_w_mean=float(np.mean(power_values)),
        temp_c_mean=float(np.mean(temp_values)),
    )


# =============================================================================
# Main Benchmark
# =============================================================================

def run_benchmark(
    platform: str = "auto",
    n_runs: int = 10,
    num_tokens: int = 100,
    warmup_runs: int = 3,
    output_dir: str = "results/z111_real_benchmark",
):
    """Run full A/B benchmark."""

    print("=" * 70)
    print("FEEL-SLM v2 Real A/B Energy Benchmark")
    print("=" * 70)

    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")

    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Telemetry
    print("\n--- Setting up telemetry ---")
    source, sampler = create_telemetry_source(platform)
    print(f"Platform: {type(source).__name__}")
    print(f"Capabilities: {source.get_capabilities()}")

    sampler.start()
    time.sleep(0.5)

    latest = sampler.get_latest()
    if latest:
        print(f"Telemetry OK: {latest.power_w:.1f}W, {latest.temp_c:.1f}°C")

    # Models
    print("\n--- Creating models ---")
    config = FEELConfigV2(
        hidden_dim=256,
        num_layers=4,
        num_heads=4,
        num_kv_heads=2,
        intermediate_dim=512,
        phase=1,
        enable_film=False,
        enable_gating=False,
        enable_layerdrop=True,
        layerdrop_layers=[1, 2],
    )

    baseline = BaselineSLMV2(config).to(device).eval()
    feel = FEELSLMV2(config).to(device).eval()

    baseline_params = sum(p.numel() for p in baseline.parameters())
    feel_params = sum(p.numel() for p in feel.parameters())
    print(f"Baseline params: {baseline_params:,}")
    print(f"FEEL params: {feel_params:,}")
    print(f"Overhead: {(feel_params - baseline_params) / baseline_params * 100:.2f}%")

    # Define conditions
    conditions = [
        ("baseline", baseline, "balanced"),
        ("baseline", baseline, "eco"),
        ("baseline", baseline, "perf"),
        ("feel", feel, "balanced"),
        ("feel", feel, "eco"),
        ("feel", feel, "perf"),
        ("feel", feel, "adaptive"),
    ]

    # Warmup
    print(f"\n--- Warmup ({warmup_runs} runs) ---")
    input_ids = torch.randint(0, 32000, (1, 32), device=device)
    for _ in range(warmup_runs):
        with torch.no_grad():
            _ = baseline(input_ids)
            _ = feel(input_ids, torch.zeros(1, 12, device=device))
        torch.cuda.synchronize() if device.type == "cuda" else None

    # Run benchmark (interleaved)
    print(f"\n--- Running benchmark ({n_runs} runs per condition) ---")

    all_runs: Dict[str, List[BenchmarkRun]] = {
        f"{cond}_{mode}": [] for cond, _, mode in conditions
    }

    for run_idx in range(n_runs):
        print(f"\nRun {run_idx + 1}/{n_runs}")

        # Shuffle order to reduce systematic bias
        shuffled = conditions.copy()
        if run_idx % 2 == 1:
            shuffled = list(reversed(shuffled))

        for cond_name, model, mode in shuffled:
            key = f"{cond_name}_{mode}"

            # Brief pause between conditions
            time.sleep(0.3)

            result = run_single_benchmark(
                model=model,
                sampler=sampler,
                condition=cond_name,
                mode=mode,
                run_idx=run_idx,
                num_tokens=num_tokens,
                device=device,
            )
            all_runs[key].append(result)

            print(f"  {key}: {result.mj_per_token:.1f} mJ/tok, "
                  f"{result.tokens_per_second:.1f} tok/s")

    # Stop sampler
    sampler.stop()

    # Compute summaries
    summaries = {key: compute_summary(runs) for key, runs in all_runs.items()}

    # Print results
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)

    print(f"\n{'Condition':<20} {'n':>5} {'mJ/tok':>10} {'95% CI':>22} {'tok/s':>10} {'Power':>8}")
    print("-" * 80)

    for key in sorted(summaries.keys()):
        s = summaries[key]
        ci_str = f"[{s.mj_per_token_ci95[0]:.1f}, {s.mj_per_token_ci95[1]:.1f}]"
        print(f"{key:<20} {s.n_runs:>5} {s.mj_per_token_mean:>10.1f} "
              f"{ci_str:>22} {s.tokens_per_sec_mean:>10.1f} {s.power_w_mean:>7.1f}W")

    # Statistical comparisons
    print("\n--- Statistical Comparisons ---")

    comparisons = [
        ("baseline_balanced", "baseline_eco", "Baseline: balanced vs eco"),
        ("baseline_balanced", "baseline_perf", "Baseline: balanced vs perf"),
        ("feel_balanced", "feel_eco", "FEEL: balanced vs eco (LayerDrop effect)"),
        ("feel_balanced", "feel_perf", "FEEL: balanced vs perf"),
        ("baseline_balanced", "feel_balanced", "Baseline vs FEEL (balanced)"),
        ("baseline_eco", "feel_eco", "Baseline vs FEEL (eco)"),
        ("feel_balanced", "feel_adaptive", "FEEL: balanced vs adaptive"),
    ]

    print(f"\n{'Comparison':<45} {'Δ mJ/tok':>12} {'t-stat':>10} {'Sig?':>8}")
    print("-" * 80)

    for key1, key2, desc in comparisons:
        if key1 in all_runs and key2 in all_runs:
            vals1 = [r.mj_per_token for r in all_runs[key1]]
            vals2 = [r.mj_per_token for r in all_runs[key2]]

            mean1, mean2 = np.mean(vals1), np.mean(vals2)
            delta = mean2 - mean1
            t_stat, p_val = stats.ttest_ind(vals1, vals2)
            sig = "YES" if p_val < 0.05 else "NO"

            print(f"{desc:<45} {delta:>+12.1f} {t_stat:>10.2f} {sig:>8}")

    # Key insights
    print("\n--- Key Insights ---")

    feel_eco = summaries.get("feel_eco")
    feel_perf = summaries.get("feel_perf")
    feel_adaptive = summaries.get("feel_adaptive")
    baseline_balanced = summaries.get("baseline_balanced")
    feel_balanced = summaries.get("feel_balanced")

    if feel_eco and feel_perf:
        eco_savings = (feel_perf.mj_per_token_mean - feel_eco.mj_per_token_mean) / feel_perf.mj_per_token_mean * 100
        eco_speedup = feel_eco.tokens_per_sec_mean / feel_perf.tokens_per_sec_mean
        print(f"1. LayerDrop energy savings (eco vs perf): {eco_savings:.1f}%")
        print(f"2. LayerDrop speedup: {eco_speedup:.2f}x")

    if baseline_balanced and feel_balanced:
        overhead = (feel_balanced.mj_per_token_mean - baseline_balanced.mj_per_token_mean) / baseline_balanced.mj_per_token_mean * 100
        print(f"3. FEEL overhead (balanced mode): {overhead:+.1f}%")

    if feel_adaptive:
        print(f"4. FEEL adaptive energy: {feel_adaptive.mj_per_token_mean:.1f} mJ/tok")

    # Save results
    os.makedirs(output_dir, exist_ok=True)

    results_data = {
        "config": {
            "platform": platform,
            "n_runs": n_runs,
            "num_tokens": num_tokens,
            "warmup_runs": warmup_runs,
            "device": str(device),
            "gpu_name": torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU",
            "baseline_params": baseline_params,
            "feel_params": feel_params,
        },
        "runs": {key: [asdict(r) for r in runs] for key, runs in all_runs.items()},
        "summaries": {key: asdict(s) for key, s in summaries.items()},
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

    results_path = os.path.join(output_dir, "benchmark_results.json")
    with open(results_path, "w") as f:
        json.dump(results_data, f, indent=2, default=lambda x: list(x) if isinstance(x, tuple) else x)
    print(f"\nResults saved to: {results_path}")

    # Generate LaTeX table
    latex = r"""\begin{table}[t]
\centering
\caption{FEEL-SLM v2 Energy Benchmark Results}
\label{tab:feel-v2-benchmark}
\begin{tabular}{llrrrr}
\toprule
Model & Mode & mJ/tok & 95\% CI & tok/s & Power (W) \\
\midrule
"""
    for key in sorted(summaries.keys()):
        s = summaries[key]
        cond, mode = key.rsplit("_", 1)
        ci_str = f"[{s.mj_per_token_ci95[0]:.1f}, {s.mj_per_token_ci95[1]:.1f}]"
        latex += f"{cond.title()} & {mode} & {s.mj_per_token_mean:.1f} & {ci_str} & {s.tokens_per_sec_mean:.1f} & {s.power_w_mean:.0f} \\\\\n"

    latex += r"""\bottomrule
\end{tabular}
\end{table}
"""

    table_path = os.path.join(output_dir, "benchmark_table.tex")
    with open(table_path, "w") as f:
        f.write(latex)
    print(f"LaTeX table saved to: {table_path}")

    return summaries, all_runs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--platform", default="auto", choices=["auto", "amd", "nvidia"])
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--tokens", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--output", default="results/z111_real_benchmark")
    args = parser.parse_args()

    run_benchmark(
        platform=args.platform,
        n_runs=args.runs,
        num_tokens=args.tokens,
        warmup_runs=args.warmup,
        output_dir=args.output,
    )
