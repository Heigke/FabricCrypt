#!/usr/bin/env python3
"""Generate paper-quality comparison table for AMD vs NVIDIA benchmarks."""

import json
import os
from pathlib import Path


def load_results():
    """Load benchmark results from both platforms."""
    base_dir = Path(__file__).parent.parent / "results"

    # AMD results (Ikaros - Radeon 8060S)
    amd_path = base_dir / "z106_benchmark" / "benchmark_results.json"
    with open(amd_path) as f:
        amd_results = json.load(f)

    # NVIDIA results (Minos - RTX A6000)
    nvidia_path = base_dir / "z107_nvidia_benchmark" / "benchmark_results.json"
    with open(nvidia_path) as f:
        nvidia_results = json.load(f)

    return amd_results, nvidia_results


def generate_comparison_table():
    """Generate comprehensive comparison table."""
    amd, nvidia = load_results()

    print("=" * 80)
    print("FEEL-SLM Cross-Platform Energy Benchmark Comparison")
    print("=" * 80)

    # Platform info
    print("\n## Platform Configuration")
    print("-" * 80)
    print(f"{'Platform':<20} {'GPU':<25} {'Model Params (B/F)':<25}")
    print("-" * 80)
    print(f"{'AMD (Ikaros)':<20} {'Radeon 8060S (APU)':<25} "
          f"{amd['config']['baseline_params']:,} / {amd['config']['feel_params']:,}")
    print(f"{'NVIDIA (Minos)':<20} {nvidia['config']['gpu_name']:<25} "
          f"{nvidia['config']['baseline_params']:,} / {nvidia['config']['feel_params']:,}")

    # Main comparison table
    print("\n## Energy Efficiency Comparison (n=10 runs, 100 tokens each)")
    print("-" * 80)
    print(f"{'Platform':<15} {'Model':<10} {'mJ/tok':>12} {'95% CI':>22} {'tok/s':>10} {'Power':>10}")
    print("-" * 80)

    for platform_name, results in [("AMD", amd), ("NVIDIA", nvidia)]:
        for condition in ["baseline", "feel"]:
            summary = results["summaries"][condition]
            ci = summary["mj_per_token_ci95"]
            ci_str = f"[{ci[0]:.1f}, {ci[1]:.1f}]"
            print(f"{platform_name:<15} {condition:<10} "
                  f"{summary['mj_per_token_mean']:>12.1f} "
                  f"{ci_str:>22} "
                  f"{summary['tokens_per_sec_mean']:>10.1f} "
                  f"{summary['power_w_mean']:>9.0f}W")

    # Calculate overhead percentages
    print("\n## FEEL Overhead Analysis")
    print("-" * 80)

    amd_overhead = (amd['summaries']['feel']['mj_per_token_mean'] -
                   amd['summaries']['baseline']['mj_per_token_mean']) / \
                  amd['summaries']['baseline']['mj_per_token_mean'] * 100

    nvidia_overhead = (nvidia['summaries']['feel']['mj_per_token_mean'] -
                      nvidia['summaries']['baseline']['mj_per_token_mean']) / \
                     nvidia['summaries']['baseline']['mj_per_token_mean'] * 100

    amd_params_overhead = (amd['config']['feel_params'] - amd['config']['baseline_params']) / \
                         amd['config']['baseline_params'] * 100

    nvidia_params_overhead = (nvidia['config']['feel_params'] - nvidia['config']['baseline_params']) / \
                            nvidia['config']['baseline_params'] * 100

    print(f"{'Platform':<15} {'Param Overhead':>15} {'Energy Overhead':>18} {'Significant':>15}")
    print("-" * 80)
    print(f"{'AMD':<15} {amd_params_overhead:>14.1f}% {amd_overhead:>17.1f}% "
          f"{'YES' if abs(amd.get('comparison', {}).get('t_statistic', 0)) > 2.0 else 'NO':>15}")
    print(f"{'NVIDIA':<15} {nvidia_params_overhead:>14.1f}% {nvidia_overhead:>17.1f}% "
          f"{'YES' if nvidia['comparison']['significant'] else 'NO':>15}")

    # Speed comparison
    print("\n## Throughput Comparison")
    print("-" * 80)

    amd_speed_loss = (amd['summaries']['baseline']['tokens_per_sec_mean'] -
                     amd['summaries']['feel']['tokens_per_sec_mean']) / \
                    amd['summaries']['baseline']['tokens_per_sec_mean'] * 100

    nvidia_speed_loss = (nvidia['summaries']['baseline']['tokens_per_sec_mean'] -
                        nvidia['summaries']['feel']['tokens_per_sec_mean']) / \
                       nvidia['summaries']['baseline']['tokens_per_sec_mean'] * 100

    print(f"{'Platform':<15} {'Baseline tok/s':>15} {'FEEL tok/s':>15} {'Speed Loss':>15}")
    print("-" * 80)
    print(f"{'AMD':<15} {amd['summaries']['baseline']['tokens_per_sec_mean']:>15.1f} "
          f"{amd['summaries']['feel']['tokens_per_sec_mean']:>15.1f} {amd_speed_loss:>14.1f}%")
    print(f"{'NVIDIA':<15} {nvidia['summaries']['baseline']['tokens_per_sec_mean']:>15.1f} "
          f"{nvidia['summaries']['feel']['tokens_per_sec_mean']:>15.1f} {nvidia_speed_loss:>14.1f}%")

    # Generate LaTeX table
    latex = r"""\begin{table}[t]
\centering
\caption{FEEL-SLM Cross-Platform Energy Benchmark (Untrained Models)}
\label{tab:feel-benchmark}
\begin{tabular}{llrrrrr}
\toprule
Platform & Model & Params & mJ/tok & 95\% CI & tok/s & Power (W) \\
\midrule
"""

    for platform_name, results in [("AMD Radeon 8060S", amd), ("NVIDIA RTX A6000", nvidia)]:
        for condition in ["baseline", "feel"]:
            summary = results["summaries"][condition]
            ci = summary["mj_per_token_ci95"]
            params = results['config']['baseline_params'] if condition == 'baseline' else results['config']['feel_params']
            latex += f"{platform_name} & {condition.title()} & {params:,} & "
            latex += f"{summary['mj_per_token_mean']:.1f} & "
            latex += f"[{ci[0]:.1f}, {ci[1]:.1f}] & "
            latex += f"{summary['tokens_per_sec_mean']:.1f} & "
            latex += f"{summary['power_w_mean']:.0f} \\\\\n"
        latex += r"\midrule" + "\n"

    # Remove last \midrule
    latex = latex.rsplit(r"\midrule", 1)[0]

    latex += r"""\bottomrule
\end{tabular}
\vspace{1mm}
\footnotesize{Note: FEEL shows +7.5\% (NVIDIA) and +62\% (AMD) energy overhead for untrained models. \\
After training, FEEL should demonstrate energy savings through learned adaptive computation.}
\end{table}
"""

    output_dir = Path(__file__).parent.parent / "results" / "comparison"
    output_dir.mkdir(exist_ok=True)

    # Save LaTeX table
    table_path = output_dir / "cross_platform_comparison.tex"
    with open(table_path, "w") as f:
        f.write(latex)
    print(f"\nLaTeX table saved to: {table_path}")

    # Save summary JSON
    summary = {
        "platforms": {
            "amd": {
                "name": "AMD Radeon 8060S (APU)",
                "host": "Ikaros",
                "baseline_mj_per_token": amd['summaries']['baseline']['mj_per_token_mean'],
                "feel_mj_per_token": amd['summaries']['feel']['mj_per_token_mean'],
                "energy_overhead_pct": amd_overhead,
                "params_overhead_pct": amd_params_overhead,
                "baseline_tok_per_sec": amd['summaries']['baseline']['tokens_per_sec_mean'],
                "feel_tok_per_sec": amd['summaries']['feel']['tokens_per_sec_mean'],
                "speed_loss_pct": amd_speed_loss
            },
            "nvidia": {
                "name": "NVIDIA RTX A6000",
                "host": "Minos",
                "baseline_mj_per_token": nvidia['summaries']['baseline']['mj_per_token_mean'],
                "feel_mj_per_token": nvidia['summaries']['feel']['mj_per_token_mean'],
                "energy_overhead_pct": nvidia_overhead,
                "params_overhead_pct": nvidia_params_overhead,
                "baseline_tok_per_sec": nvidia['summaries']['baseline']['tokens_per_sec_mean'],
                "feel_tok_per_sec": nvidia['summaries']['feel']['tokens_per_sec_mean'],
                "speed_loss_pct": nvidia_speed_loss
            }
        },
        "conclusion": "FEEL shows consistent overhead for untrained models (expected). "
                     "Energy savings require training to learn adaptive computation."
    }

    summary_path = output_dir / "comparison_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary JSON saved to: {summary_path}")

    return summary


if __name__ == "__main__":
    generate_comparison_table()
