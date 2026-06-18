#!/usr/bin/env python3
"""
Comprehensive Regime Map Experiment Matrix.

Runs experiments across multiple dimensions to identify when active DVFS control
helps vs when auto is sufficient.

Dimensions:
- Model size: 0.5B, 1.5B, 3B, (7B if VRAM permits)
- Prompt length: 256, 2048, 8192
- Decode length: 64, 256, 1024
- Concurrency: 1, 2, 4
- Temperature: 0.0 (greedy), 0.7 (sampling)
- Controller: auto, controller, windowed_controller

Statistical rigor:
- N>=5 repeats per condition
- Bootstrap confidence intervals
- t-tests for pairwise comparisons
- Effect size (Cohen's d) calculation
"""

import argparse
import json
import subprocess
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import scipy.stats as stats


@dataclass
class ExperimentCondition:
    """Single experiment condition."""
    model: str
    model_size_b: float  # Model size in billions
    prompt_tokens: int
    decode_tokens: int
    concurrency: int
    temperature: float
    policy: str
    repeats: int = 5

    @property
    def name(self) -> str:
        temp_str = "greedy" if self.temperature == 0.0 else f"t{self.temperature}"
        return f"{self.model_size_b}B_p{self.prompt_tokens}_d{self.decode_tokens}_c{self.concurrency}_{temp_str}_{self.policy}"


@dataclass
class ExperimentResult:
    """Results from a single experiment run."""
    condition: ExperimentCondition
    run_idx: int
    success: bool = False
    total_s: float = 0.0
    prefill_s: float = 0.0
    ttft_s: float = 0.0
    tpot_mean_ms: float = 0.0
    tpot_p50_ms: float = 0.0
    tpot_p95_ms: float = 0.0
    tpot_std_ms: float = 0.0
    energy_j: float = 0.0
    tok_per_j: float = 0.0
    avg_power_w: float = 0.0
    peak_power_w: float = 0.0
    avg_temp_c: float = 0.0
    peak_temp_c: float = 0.0
    avg_sclk_mhz: float = 0.0
    policy_switches: int = 0
    error_msg: str = ""


@dataclass
class AggregatedResult:
    """Aggregated results with statistics."""
    condition: ExperimentCondition
    n: int = 0
    success_rate: float = 0.0

    # Means and standard errors
    total_s_mean: float = 0.0
    total_s_se: float = 0.0
    energy_j_mean: float = 0.0
    energy_j_se: float = 0.0
    tok_per_j_mean: float = 0.0
    tok_per_j_se: float = 0.0
    tpot_p95_ms_mean: float = 0.0
    tpot_p95_ms_se: float = 0.0
    avg_power_w_mean: float = 0.0
    avg_power_w_se: float = 0.0

    # Confidence intervals (95%)
    tok_per_j_ci_lo: float = 0.0
    tok_per_j_ci_hi: float = 0.0
    tpot_p95_ms_ci_lo: float = 0.0
    tpot_p95_ms_ci_hi: float = 0.0

    # Raw values for later analysis
    raw_tok_per_j: List[float] = None
    raw_tpot_p95_ms: List[float] = None
    raw_energy_j: List[float] = None


def compute_ci_bootstrap(values: List[float], n_bootstrap: int = 1000, ci: float = 0.95) -> Tuple[float, float]:
    """Compute bootstrap confidence interval."""
    if len(values) < 2:
        return (values[0] if values else 0.0, values[0] if values else 0.0)

    arr = np.array(values)
    boot_means = []
    for _ in range(n_bootstrap):
        sample = np.random.choice(arr, size=len(arr), replace=True)
        boot_means.append(np.mean(sample))

    alpha = 1 - ci
    lo = np.percentile(boot_means, alpha / 2 * 100)
    hi = np.percentile(boot_means, (1 - alpha / 2) * 100)
    return (lo, hi)


def cohens_d(group1: List[float], group2: List[float]) -> float:
    """Compute Cohen's d effect size."""
    n1, n2 = len(group1), len(group2)
    if n1 < 2 or n2 < 2:
        return 0.0

    mean1, mean2 = np.mean(group1), np.mean(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)

    # Pooled standard deviation
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    if pooled_std == 0:
        return 0.0

    return (mean1 - mean2) / pooled_std


def pairwise_ttest(group1: List[float], group2: List[float]) -> Tuple[float, float]:
    """Perform independent t-test, return (t_stat, p_value)."""
    if len(group1) < 2 or len(group2) < 2:
        return (0.0, 1.0)

    t_stat, p_val = stats.ttest_ind(group1, group2)
    return (t_stat, p_val)


def aggregate_results(results: List[ExperimentResult]) -> AggregatedResult:
    """Aggregate multiple experiment results with statistics."""
    if not results:
        return None

    condition = results[0].condition
    successful = [r for r in results if r.success]
    n = len(successful)

    if n == 0:
        return AggregatedResult(condition=condition, n=0, success_rate=0.0)

    # Extract values
    tok_per_j = [r.tok_per_j for r in successful]
    energy_j = [r.energy_j for r in successful]
    total_s = [r.total_s for r in successful]
    tpot_p95 = [r.tpot_p95_ms for r in successful]
    power_w = [r.avg_power_w for r in successful]

    # Compute means and standard errors
    def mean_se(vals):
        m = np.mean(vals)
        se = np.std(vals, ddof=1) / np.sqrt(len(vals)) if len(vals) > 1 else 0.0
        return m, se

    tok_per_j_mean, tok_per_j_se = mean_se(tok_per_j)
    energy_j_mean, energy_j_se = mean_se(energy_j)
    total_s_mean, total_s_se = mean_se(total_s)
    tpot_p95_mean, tpot_p95_se = mean_se(tpot_p95)
    power_w_mean, power_w_se = mean_se(power_w)

    # Bootstrap CIs
    tok_per_j_ci = compute_ci_bootstrap(tok_per_j)
    tpot_p95_ci = compute_ci_bootstrap(tpot_p95)

    return AggregatedResult(
        condition=condition,
        n=n,
        success_rate=n / len(results),
        total_s_mean=total_s_mean,
        total_s_se=total_s_se,
        energy_j_mean=energy_j_mean,
        energy_j_se=energy_j_se,
        tok_per_j_mean=tok_per_j_mean,
        tok_per_j_se=tok_per_j_se,
        tpot_p95_ms_mean=tpot_p95_mean,
        tpot_p95_ms_se=tpot_p95_se,
        avg_power_w_mean=power_w_mean,
        avg_power_w_se=power_w_se,
        tok_per_j_ci_lo=tok_per_j_ci[0],
        tok_per_j_ci_hi=tok_per_j_ci[1],
        tpot_p95_ms_ci_lo=tpot_p95_ci[0],
        tpot_p95_ms_ci_hi=tpot_p95_ci[1],
        raw_tok_per_j=tok_per_j,
        raw_tpot_p95_ms=tpot_p95,
        raw_energy_j=energy_j,
    )


def run_single_experiment(
    condition: ExperimentCondition,
    run_idx: int,
    output_dir: Path,
    timeout_s: int = 600,
) -> ExperimentResult:
    """Run a single experiment and return results."""
    result = ExperimentResult(condition=condition, run_idx=run_idx)

    # Get project root (where scripts/ directory is)
    project_root = Path(__file__).parent.parent.resolve()
    script_path = project_root / "scripts" / "hf_infer_extended.py"
    run_output_dir = output_dir.resolve() / f"run_{run_idx}"

    # Build command
    cmd = [
        sys.executable,
        str(script_path),
        "--model", condition.model,
        "--policy", condition.policy,
        "--prompt-tokens", str(condition.prompt_tokens),
        "--decode-tokens", str(condition.decode_tokens),
        "--concurrency", str(condition.concurrency),
        "--repeats", "1",  # Single run
        "--output-dir", str(run_output_dir),
    ]

    if condition.temperature > 0:
        cmd.extend(["--temperature", str(condition.temperature)])

    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout_s,
            cwd=str(project_root),
        )

        if proc.returncode != 0:
            result.error_msg = proc.stderr[:500] if proc.stderr else "Unknown error"
            return result

        # Parse output JSON (use the same resolved path we passed to the command)
        result_files = list(run_output_dir.glob("comparison*.json")) + list(run_output_dir.glob("results*.json"))

        if result_files:
            with open(result_files[0]) as f:
                data = json.load(f)

            # Handle different JSON structures
            if condition.policy in data:
                policy_data = data[condition.policy]
            elif "results" in data:
                policy_data = data["results"].get(condition.policy, {})
            else:
                policy_data = data

            result.success = True
            result.total_s = policy_data.get("total_s_mean", policy_data.get("total_s", 0))
            result.prefill_s = policy_data.get("prefill_s_mean", policy_data.get("prefill_s", 0))
            result.ttft_s = policy_data.get("ttft_s_mean", policy_data.get("ttft_s", 0))
            result.tpot_mean_ms = policy_data.get("tpot_s_mean_mean", 0) * 1000
            result.tpot_p50_ms = policy_data.get("tpot_s_p50_mean", 0) * 1000
            result.tpot_p95_ms = policy_data.get("tpot_s_p95_mean", 0) * 1000
            result.tpot_std_ms = policy_data.get("tpot_s_std_mean", 0) * 1000
            result.energy_j = policy_data.get("total_energy_j_mean", policy_data.get("total_energy_j", 0))
            result.tok_per_j = policy_data.get("total_tok_per_j_mean", policy_data.get("total_tok_per_j", 0))
            result.avg_power_w = policy_data.get("avg_power_w_mean", policy_data.get("avg_power_w", 0))
            result.peak_power_w = policy_data.get("peak_power_w_mean", policy_data.get("peak_power_w", 0))
            result.avg_sclk_mhz = policy_data.get("avg_sclk_mhz", 0)
        else:
            result.error_msg = "No result file found"

    except subprocess.TimeoutExpired:
        result.error_msg = "Timeout"
    except Exception as e:
        result.error_msg = str(e)[:500]

    return result


def generate_experiment_matrix(
    models: List[str],
    prompt_lengths: List[int],
    decode_lengths: List[int],
    concurrencies: List[int],
    temperatures: List[float],
    policies: List[str],
    repeats: int,
) -> List[ExperimentCondition]:
    """Generate full experiment matrix."""
    model_sizes = {
        "Qwen/Qwen2.5-0.5B-Instruct": 0.5,
        "Qwen/Qwen2.5-1.5B-Instruct": 1.5,
        "Qwen/Qwen2.5-3B-Instruct": 3.0,
        "Qwen/Qwen2.5-7B-Instruct": 7.0,
    }

    conditions = []
    for model in models:
        for prompt in prompt_lengths:
            for decode in decode_lengths:
                for conc in concurrencies:
                    for temp in temperatures:
                        for policy in policies:
                            conditions.append(ExperimentCondition(
                                model=model,
                                model_size_b=model_sizes.get(model, 0),
                                prompt_tokens=prompt,
                                decode_tokens=decode,
                                concurrency=conc,
                                temperature=temp,
                                policy=policy,
                                repeats=repeats,
                            ))

    return conditions


def run_experiment_matrix(
    conditions: List[ExperimentCondition],
    output_dir: Path,
    max_workers: int = 1,  # Sequential by default for GPU
) -> Dict[str, AggregatedResult]:
    """Run all experiments and return aggregated results."""
    output_dir.mkdir(parents=True, exist_ok=True)

    all_results: Dict[str, List[ExperimentResult]] = {}
    total_runs = sum(c.repeats for c in conditions)
    completed = 0

    print(f"Running {len(conditions)} conditions with {total_runs} total runs")
    print("=" * 60)

    for condition in conditions:
        condition_name = condition.name
        condition_dir = output_dir / condition_name
        condition_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n[{completed}/{total_runs}] {condition_name}")

        condition_results = []
        for run_idx in range(condition.repeats):
            print(f"  Run {run_idx + 1}/{condition.repeats}...", end=" ", flush=True)
            start = time.time()

            result = run_single_experiment(condition, run_idx, condition_dir)
            condition_results.append(result)

            elapsed = time.time() - start
            if result.success:
                print(f"OK ({elapsed:.1f}s, {result.tok_per_j:.2f} tok/J)")
            else:
                print(f"FAIL ({result.error_msg[:50]})")

            completed += 1

        all_results[condition_name] = condition_results

        # Save intermediate results
        agg = aggregate_results(condition_results)
        if agg:
            with open(condition_dir / "aggregated.json", "w") as f:
                json.dump({
                    "condition": asdict(condition),
                    "n": agg.n,
                    "success_rate": agg.success_rate,
                    "tok_per_j_mean": agg.tok_per_j_mean,
                    "tok_per_j_se": agg.tok_per_j_se,
                    "tok_per_j_ci": [agg.tok_per_j_ci_lo, agg.tok_per_j_ci_hi],
                    "tpot_p95_ms_mean": agg.tpot_p95_ms_mean,
                    "tpot_p95_ms_se": agg.tpot_p95_ms_se,
                    "energy_j_mean": agg.energy_j_mean,
                    "avg_power_w_mean": agg.avg_power_w_mean,
                }, f, indent=2)

    # Aggregate all results
    aggregated = {}
    for name, results in all_results.items():
        agg = aggregate_results(results)
        if agg:
            aggregated[name] = agg

    return aggregated


def compare_policies(
    results: Dict[str, AggregatedResult],
    baseline_policy: str = "auto",
) -> List[Dict]:
    """Compare policies against baseline with statistical tests."""
    comparisons = []

    # Group by condition (excluding policy)
    condition_groups = {}
    for name, agg in results.items():
        key = (
            agg.condition.model_size_b,
            agg.condition.prompt_tokens,
            agg.condition.decode_tokens,
            agg.condition.concurrency,
            agg.condition.temperature,
        )
        if key not in condition_groups:
            condition_groups[key] = {}
        condition_groups[key][agg.condition.policy] = agg

    for key, policies in condition_groups.items():
        if baseline_policy not in policies:
            continue

        baseline = policies[baseline_policy]

        for policy_name, agg in policies.items():
            if policy_name == baseline_policy:
                continue

            # Efficiency comparison
            eff_diff = agg.tok_per_j_mean - baseline.tok_per_j_mean
            eff_pct = (eff_diff / baseline.tok_per_j_mean * 100) if baseline.tok_per_j_mean > 0 else 0

            # Statistical tests
            if agg.raw_tok_per_j and baseline.raw_tok_per_j:
                t_stat, p_val = pairwise_ttest(agg.raw_tok_per_j, baseline.raw_tok_per_j)
                d = cohens_d(agg.raw_tok_per_j, baseline.raw_tok_per_j)
            else:
                t_stat, p_val, d = 0.0, 1.0, 0.0

            # Determine winner (convert numpy bool to Python bool for JSON)
            significant = bool(p_val < 0.05)
            winner = policy_name if eff_diff > 0 and significant else (
                baseline_policy if eff_diff < 0 and significant else "tie"
            )

            comparisons.append({
                "model_size_b": key[0],
                "prompt_tokens": key[1],
                "decode_tokens": key[2],
                "concurrency": key[3],
                "temperature": key[4],
                "baseline": baseline_policy,
                "policy": policy_name,
                "baseline_tok_per_j": baseline.tok_per_j_mean,
                "baseline_tok_per_j_ci": [baseline.tok_per_j_ci_lo, baseline.tok_per_j_ci_hi],
                "policy_tok_per_j": agg.tok_per_j_mean,
                "policy_tok_per_j_ci": [agg.tok_per_j_ci_lo, agg.tok_per_j_ci_hi],
                "efficiency_diff": eff_diff,
                "efficiency_pct": eff_pct,
                "t_stat": t_stat,
                "p_value": p_val,
                "cohens_d": d,
                "significant": significant,
                "winner": winner,
            })

    return comparisons


def generate_regime_map(comparisons: List[Dict], output_path: Path):
    """Generate regime map showing when each policy wins."""
    import matplotlib.pyplot as plt
    import pandas as pd

    df = pd.DataFrame(comparisons)

    # Create regime map heatmap
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Subplot 1: Model size vs Prompt length
    for ax_idx, policy in enumerate(["controller", "windowed"]):
        ax = axes[ax_idx]
        policy_df = df[df["policy"] == policy] if policy in df["policy"].values else df

        if policy_df.empty:
            continue

        # Pivot for heatmap
        pivot = policy_df.pivot_table(
            values="efficiency_pct",
            index="model_size_b",
            columns="prompt_tokens",
            aggfunc="mean",
        )

        im = ax.imshow(pivot.values, cmap="RdYlGn", aspect="auto", vmin=-20, vmax=20)
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels(pivot.columns)
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels([f"{x}B" for x in pivot.index])
        ax.set_xlabel("Prompt Length")
        ax.set_ylabel("Model Size")
        ax.set_title(f"{policy} vs auto (% efficiency gain)")

        # Add text annotations
        for i in range(len(pivot.index)):
            for j in range(len(pivot.columns)):
                val = pivot.values[i, j]
                color = "white" if abs(val) > 10 else "black"
                ax.text(j, i, f"{val:+.1f}%", ha="center", va="center", color=color, fontsize=9)

        plt.colorbar(im, ax=ax, label="Efficiency gain (%)")

    plt.tight_layout()
    plt.savefig(output_path / "regime_map_v2.png", dpi=150, bbox_inches="tight")
    plt.savefig(output_path / "regime_map_v2.pdf", bbox_inches="tight")
    plt.close()

    print(f"Saved regime map to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Run comprehensive regime map experiments")
    parser.add_argument("--output-dir", type=Path, default=Path("results/regime_matrix"),
                       help="Output directory")
    parser.add_argument("--repeats", type=int, default=5,
                       help="Number of repeats per condition")

    # Dimension control
    parser.add_argument("--models", nargs="+",
                       default=["Qwen/Qwen2.5-0.5B-Instruct", "Qwen/Qwen2.5-3B-Instruct"],
                       help="Models to test")
    parser.add_argument("--prompt-lengths", nargs="+", type=int,
                       default=[256, 2048],
                       help="Prompt lengths")
    parser.add_argument("--decode-lengths", nargs="+", type=int,
                       default=[64, 256],
                       help="Decode lengths")
    parser.add_argument("--concurrencies", nargs="+", type=int,
                       default=[1],
                       help="Concurrency levels")
    parser.add_argument("--temperatures", nargs="+", type=float,
                       default=[0.0],
                       help="Temperatures (0.0=greedy)")
    parser.add_argument("--policies", nargs="+",
                       default=["auto", "controller", "windowed"],
                       help="Policies to compare")

    # Quick mode for testing
    parser.add_argument("--quick", action="store_true",
                       help="Quick test mode (1 repeat, limited conditions)")

    args = parser.parse_args()

    if args.quick:
        args.repeats = 1
        args.prompt_lengths = [256]
        args.decode_lengths = [32]
        args.policies = ["auto", "controller"]

    print("=" * 60)
    print("REGIME MAP EXPERIMENT MATRIX")
    print("=" * 60)
    print(f"Models: {args.models}")
    print(f"Prompt lengths: {args.prompt_lengths}")
    print(f"Decode lengths: {args.decode_lengths}")
    print(f"Concurrencies: {args.concurrencies}")
    print(f"Temperatures: {args.temperatures}")
    print(f"Policies: {args.policies}")
    print(f"Repeats: {args.repeats}")
    print("=" * 60)

    # Generate conditions
    conditions = generate_experiment_matrix(
        models=args.models,
        prompt_lengths=args.prompt_lengths,
        decode_lengths=args.decode_lengths,
        concurrencies=args.concurrencies,
        temperatures=args.temperatures,
        policies=args.policies,
        repeats=args.repeats,
    )

    total_runs = sum(c.repeats for c in conditions)
    print(f"\nTotal conditions: {len(conditions)}")
    print(f"Total runs: {total_runs}")
    print(f"Estimated time: {total_runs * 0.5:.0f} - {total_runs * 2:.0f} minutes")
    print("=" * 60)

    # Run experiments
    results = run_experiment_matrix(conditions, args.output_dir)

    # Compare policies
    comparisons = compare_policies(results, baseline_policy="auto")

    # Save results
    with open(args.output_dir / "regime_map_comparisons.json", "w") as f:
        json.dump(comparisons, f, indent=2)

    # Print summary
    print("\n" + "=" * 60)
    print("REGIME MAP SUMMARY")
    print("=" * 60)
    for comp in comparisons:
        sig = "*" if comp["significant"] else ""
        print(f"{comp['model_size_b']}B p{comp['prompt_tokens']} d{comp['decode_tokens']} "
              f"c{comp['concurrency']}: {comp['policy']} vs {comp['baseline']} "
              f"→ {comp['efficiency_pct']:+.1f}%{sig} (p={comp['p_value']:.3f}, d={comp['cohens_d']:.2f}) "
              f"[{comp['winner']}]")

    # Generate plots
    try:
        generate_regime_map(comparisons, args.output_dir)
    except ImportError:
        print("Note: matplotlib/pandas not available for plotting")

    # Save final summary
    summary = {
        "experiment_config": {
            "models": args.models,
            "prompt_lengths": args.prompt_lengths,
            "decode_lengths": args.decode_lengths,
            "concurrencies": args.concurrencies,
            "temperatures": args.temperatures,
            "policies": args.policies,
            "repeats": args.repeats,
        },
        "total_conditions": len(conditions),
        "total_runs": total_runs,
        "comparisons": comparisons,
        "regime_rules": [],  # Will be populated by analysis
    }

    # Extract regime rules
    for comp in comparisons:
        if comp["significant"]:
            summary["regime_rules"].append({
                "model_size": f"{comp['model_size_b']}B",
                "prompt": f"p{comp['prompt_tokens']}",
                "winner": comp["winner"],
                "gain": f"{abs(comp['efficiency_pct']):.1f}%",
            })

    with open(args.output_dir / "regime_map_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nResults saved to {args.output_dir}")


if __name__ == "__main__":
    main()
