#!/usr/bin/env python3
"""
Generate Report: Plots and Tables for Energy-Efficient LLM Inference Research

Creates publication-quality figures and LaTeX tables from experiment results.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime

import numpy as np
import pandas as pd

# Optional plotting imports
try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.ticker import MultipleLocator
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("Warning: matplotlib not available. Install with: pip install matplotlib")

try:
    import seaborn as sns
    SEABORN_AVAILABLE = True
except ImportError:
    SEABORN_AVAILABLE = False


def set_plot_style():
    """Set publication-quality plot style."""
    if not MATPLOTLIB_AVAILABLE:
        return

    plt.rcParams.update({
        'font.size': 11,
        'font.family': 'serif',
        'axes.labelsize': 12,
        'axes.titlesize': 13,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'figure.figsize': (8, 5),
        'figure.dpi': 150,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'axes.grid': True,
        'grid.alpha': 0.3,
    })

    if SEABORN_AVAILABLE:
        sns.set_palette("colorblind")


def load_comparison_results(path: Path) -> Dict[str, Any]:
    """Load comparison JSON results."""
    with open(path, 'r') as f:
        return json.load(f)


def load_all_results(results_dir: Path) -> Dict[str, Dict]:
    """Load all result files from a directory."""
    results = {}

    # Find comparison files
    for json_file in results_dir.rglob("*comparison*.json"):
        key = json_file.stem
        results[key] = load_comparison_results(json_file)

    # Find aggregated CSVs
    for csv_file in results_dir.rglob("*aggregated*.csv"):
        key = csv_file.stem
        results[f"csv_{key}"] = pd.read_csv(csv_file)

    return results


def plot_energy_vs_latency_pareto(
    results: Dict[str, Any],
    output_path: Path,
    title: str = "Energy vs Latency Pareto Front"
):
    """
    Plot energy vs latency with Pareto front.

    X-axis: Total latency (s)
    Y-axis: Energy consumption (J)
    Points: Different policies
    """
    if not MATPLOTLIB_AVAILABLE:
        print("Skipping plot: matplotlib not available")
        return

    fig, ax = plt.subplots(figsize=(8, 6))

    policies = list(results.keys())
    colors = plt.cm.Set1(np.linspace(0, 1, len(policies)))
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p']

    for i, (policy, data) in enumerate(results.items()):
        latency = data.get('total_s_mean', 0)
        latency_std = data.get('total_s_std', 0)
        energy = data.get('total_energy_j_mean', 0)
        energy_std = data.get('total_energy_j_std', 0)

        ax.errorbar(
            latency, energy,
            xerr=latency_std, yerr=energy_std,
            fmt=markers[i % len(markers)],
            color=colors[i],
            markersize=10,
            capsize=4,
            label=policy,
            linewidth=2
        )

    # Add Pareto front line
    points = [(results[p]['total_s_mean'], results[p]['total_energy_j_mean'], p)
              for p in policies if 'total_s_mean' in results[p]]
    points.sort(key=lambda x: x[0])

    pareto_points = []
    min_energy = float('inf')
    for lat, eng, p in points:
        if eng < min_energy:
            pareto_points.append((lat, eng))
            min_energy = eng

    if len(pareto_points) > 1:
        pareto_x, pareto_y = zip(*pareto_points)
        ax.plot(pareto_x, pareto_y, 'k--', alpha=0.5, linewidth=1.5, label='Pareto Front')

    ax.set_xlabel('Total Latency (s)')
    ax.set_ylabel('Energy Consumption (J)')
    ax.set_title(title)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path)
    plt.savefig(output_path.with_suffix('.pdf'))
    plt.close()
    print(f"Saved: {output_path}")


def plot_efficiency_comparison(
    results: Dict[str, Any],
    output_path: Path,
    title: str = "Energy Efficiency by DVFS Policy"
):
    """
    Bar chart comparing tok/J across policies.
    """
    if not MATPLOTLIB_AVAILABLE:
        return

    policies = list(results.keys())
    efficiencies = [results[p].get('total_tok_per_j_mean', 0) for p in policies]
    errors = [results[p].get('total_tok_per_j_std', 0) for p in policies]

    fig, ax = plt.subplots(figsize=(8, 5))

    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(policies)))
    bars = ax.bar(policies, efficiencies, yerr=errors, capsize=5, color=colors, edgecolor='black')

    # Add value labels on bars
    for bar, eff in zip(bars, efficiencies):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                f'{eff:.2f}', ha='center', va='bottom', fontsize=10)

    ax.set_xlabel('DVFS Policy')
    ax.set_ylabel('Energy Efficiency (tokens/Joule)')
    ax.set_title(title)
    ax.grid(True, axis='y', alpha=0.3)

    # Highlight best
    max_idx = np.argmax(efficiencies)
    bars[max_idx].set_edgecolor('red')
    bars[max_idx].set_linewidth(2)

    plt.tight_layout()
    plt.savefig(output_path)
    plt.savefig(output_path.with_suffix('.pdf'))
    plt.close()
    print(f"Saved: {output_path}")


def plot_power_timeline(
    trace_path: Path,
    output_path: Path,
    title: str = "Power Consumption Over Time"
):
    """
    Plot power consumption timeline from trace CSV.
    """
    if not MATPLOTLIB_AVAILABLE:
        return

    df = pd.read_csv(trace_path)

    if 'timestamp' not in df.columns or 'power_watts' not in df.columns:
        print(f"Skipping {trace_path}: missing required columns")
        return

    fig, ax = plt.subplots(figsize=(10, 4))

    # Normalize timestamp to start at 0
    t = df['timestamp'] - df['timestamp'].iloc[0]
    power = df['power_watts']

    ax.plot(t, power, 'b-', linewidth=1.5, alpha=0.8)
    ax.fill_between(t, power, alpha=0.3)

    # Mark phases if available
    if 'phase' in df.columns:
        phases = df['phase'].unique()
        colors = plt.cm.Set2(np.linspace(0, 1, len(phases)))
        for phase, color in zip(phases, colors):
            mask = df['phase'] == phase
            if mask.any():
                start = t[mask].iloc[0]
                end = t[mask].iloc[-1]
                ax.axvspan(start, end, alpha=0.2, color=color, label=phase)

    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Power (W)')
    ax.set_title(title)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved: {output_path}")


def plot_tpot_distribution(
    results: Dict[str, Any],
    output_path: Path,
    title: str = "TPOT Distribution by Policy"
):
    """
    Box/violin plot of TPOT distributions.
    """
    if not MATPLOTLIB_AVAILABLE:
        return

    fig, ax = plt.subplots(figsize=(8, 5))

    policies = list(results.keys())
    data_to_plot = []

    for policy in policies:
        # Use p50 and p95 to estimate distribution
        p50 = results[policy].get('tpot_s_p50_mean', 0) * 1000  # ms
        p95 = results[policy].get('tpot_s_p95_mean', 0) * 1000
        mean = results[policy].get('tpot_s_mean_mean', 0) * 1000

        data_to_plot.append({
            'policy': policy,
            'p50': p50,
            'p95': p95,
            'mean': mean
        })

    df = pd.DataFrame(data_to_plot)

    x = np.arange(len(policies))
    width = 0.35

    ax.bar(x - width/2, df['p50'], width, label='p50', color='steelblue')
    ax.bar(x + width/2, df['p95'], width, label='p95', color='coral')

    # Add jitter markers
    for i, policy in enumerate(policies):
        jitter = results[policy].get('tpot_s_std_mean', 0) * 1000
        ax.annotate(f'σ={jitter:.2f}ms', (i, df.iloc[i]['p95'] + 0.5),
                    ha='center', fontsize=8, color='gray')

    ax.set_xlabel('DVFS Policy')
    ax.set_ylabel('TPOT (ms)')
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(policies)
    ax.legend()
    ax.grid(True, axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path)
    plt.savefig(output_path.with_suffix('.pdf'))
    plt.close()
    print(f"Saved: {output_path}")


def plot_sustained_thermal(
    sustained_path: Path,
    output_path: Path
):
    """
    Plot sustained test results showing thermal behavior over time.
    """
    if not MATPLOTLIB_AVAILABLE:
        return

    with open(sustained_path, 'r') as f:
        data = json.load(f)

    time_series = data.get('time_series', [])
    if not time_series:
        return

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    elapsed = [p['elapsed_s'] for p in time_series]

    # Temperature
    ax = axes[0, 0]
    temps = [p.get('temp_c', 0) for p in time_series]
    ax.plot(elapsed, temps, 'r-', linewidth=1.5)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Temperature (°C)')
    ax.set_title('GPU Temperature')
    ax.grid(True, alpha=0.3)

    # Power
    ax = axes[0, 1]
    powers = [p.get('power_w', 0) for p in time_series]
    ax.plot(elapsed, powers, 'b-', linewidth=1.5)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Power (W)')
    ax.set_title('Power Consumption')
    ax.grid(True, alpha=0.3)

    # TPOT p95
    ax = axes[1, 0]
    tpots = [p.get('tpot_s_p95', 0) * 1000 for p in time_series]
    ax.plot(elapsed, tpots, 'g-', linewidth=1.5)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('TPOT p95 (ms)')
    ax.set_title('Latency (p95)')
    ax.grid(True, alpha=0.3)

    # SCLK
    ax = axes[1, 1]
    sclks = [p.get('sclk_mhz', 0) for p in time_series]
    ax.plot(elapsed, sclks, 'm-', linewidth=1.5)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('SCLK (MHz)')
    ax.set_title('GPU Clock')
    ax.grid(True, alpha=0.3)

    plt.suptitle(f"Sustained Test: {data.get('policy', 'unknown')} ({data.get('duration_s', 0)}s)")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.savefig(output_path.with_suffix('.pdf'))
    plt.close()
    print(f"Saved: {output_path}")


def generate_latex_table(
    results: Dict[str, Any],
    output_path: Path,
    caption: str = "DVFS Policy Comparison"
):
    """
    Generate LaTeX table from results.
    """
    policies = list(results.keys())

    # Build table data
    rows = []
    for policy in policies:
        d = results[policy]
        row = {
            'Policy': policy,
            'Latency (s)': f"{d.get('total_s_mean', 0):.3f} ± {d.get('total_s_std', 0):.3f}",
            'TPOT p95 (ms)': f"{d.get('tpot_s_p95_mean', 0)*1000:.1f} ± {d.get('tpot_s_p95_std', 0)*1000:.1f}",
            'Jitter (ms)': f"{d.get('tpot_s_std_mean', 0)*1000:.2f}",
            'Energy (J)': f"{d.get('total_energy_j_mean', 0):.1f} ± {d.get('total_energy_j_std', 0):.1f}",
            'Tok/J': f"{d.get('total_tok_per_j_mean', 0):.2f} ± {d.get('total_tok_per_j_std', 0):.2f}",
            'Power (W)': f"{d.get('avg_power_w_mean', 0):.0f} ± {d.get('avg_power_w_std', 0):.0f}",
        }
        rows.append(row)

    df = pd.DataFrame(rows)

    # Generate LaTeX
    latex = df.to_latex(index=False, escape=False, column_format='l' + 'c' * (len(df.columns) - 1))

    # Add caption and label
    latex = f"""\\begin{{table}}[htbp]
\\centering
\\caption{{{caption}}}
\\label{{tab:dvfs_comparison}}
{latex}
\\end{{table}}
"""

    with open(output_path, 'w') as f:
        f.write(latex)

    print(f"Saved: {output_path}")
    return df


def generate_markdown_table(
    results: Dict[str, Any],
    output_path: Path
):
    """
    Generate Markdown table from results.
    """
    policies = list(results.keys())

    lines = [
        "| Policy | Latency (s) | TPOT p95 (ms) | Jitter (ms) | Energy (J) | Tok/J | Power (W) |",
        "|--------|-------------|---------------|-------------|------------|-------|-----------|"
    ]

    for policy in policies:
        d = results[policy]
        line = f"| {policy} | " \
               f"{d.get('total_s_mean', 0):.3f}±{d.get('total_s_std', 0):.3f} | " \
               f"{d.get('tpot_s_p95_mean', 0)*1000:.1f}±{d.get('tpot_s_p95_std', 0)*1000:.1f} | " \
               f"{d.get('tpot_s_std_mean', 0)*1000:.2f} | " \
               f"{d.get('total_energy_j_mean', 0):.1f}±{d.get('total_energy_j_std', 0):.1f} | " \
               f"{d.get('total_tok_per_j_mean', 0):.2f}±{d.get('total_tok_per_j_std', 0):.2f} | " \
               f"{d.get('avg_power_w_mean', 0):.0f}±{d.get('avg_power_w_std', 0):.0f} |"
        lines.append(line)

    with open(output_path, 'w') as f:
        f.write('\n'.join(lines))

    print(f"Saved: {output_path}")


def generate_summary_report(
    results_dir: Path,
    output_dir: Path
):
    """
    Generate complete summary report with all plots and tables.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    set_plot_style()

    print(f"\n{'='*60}")
    print("Generating Report")
    print(f"Input: {results_dir}")
    print(f"Output: {output_dir}")
    print(f"{'='*60}\n")

    # Find comparison results
    comparison_files = list(results_dir.rglob("*comparison*.json"))

    for comp_file in comparison_files:
        print(f"\nProcessing: {comp_file}")
        results = load_comparison_results(comp_file)

        base_name = comp_file.stem

        # Generate plots
        plot_energy_vs_latency_pareto(
            results,
            output_dir / f"{base_name}_pareto.png",
            title="Energy vs Latency Trade-off"
        )

        plot_efficiency_comparison(
            results,
            output_dir / f"{base_name}_efficiency.png",
            title="Energy Efficiency Comparison"
        )

        plot_tpot_distribution(
            results,
            output_dir / f"{base_name}_tpot.png",
            title="TPOT Distribution"
        )

        # Generate tables
        generate_latex_table(
            results,
            output_dir / f"{base_name}_table.tex",
            caption=f"DVFS Policy Comparison - {base_name}"
        )

        generate_markdown_table(
            results,
            output_dir / f"{base_name}_table.md"
        )

    # Find sustained test results
    sustained_files = list(results_dir.rglob("sustained_*.json"))
    for sus_file in sustained_files:
        print(f"\nProcessing sustained: {sus_file}")
        plot_sustained_thermal(
            sus_file,
            output_dir / f"{sus_file.stem}_thermal.png"
        )

    # Find trace files for power timeline
    trace_files = list(results_dir.rglob("trace_*.csv"))[:3]  # Limit to 3
    for trace_file in trace_files:
        print(f"\nProcessing trace: {trace_file}")
        plot_power_timeline(
            trace_file,
            output_dir / f"{trace_file.stem}_timeline.png",
            title=f"Power Timeline - {trace_file.stem}"
        )

    print(f"\n{'='*60}")
    print(f"Report generation complete!")
    print(f"Output directory: {output_dir}")
    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(description="Generate plots and tables for energy efficiency report")
    parser.add_argument("--results-dir", type=Path, default=Path("results"),
                        help="Directory containing experiment results")
    parser.add_argument("--output-dir", type=Path, default=Path("reports"),
                        help="Directory for output plots and tables")
    parser.add_argument("--comparison-file", type=Path,
                        help="Specific comparison JSON file to process")

    args = parser.parse_args()

    if args.comparison_file:
        # Process single file
        output_dir = args.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        set_plot_style()

        results = load_comparison_results(args.comparison_file)
        base_name = args.comparison_file.stem

        plot_energy_vs_latency_pareto(results, output_dir / f"{base_name}_pareto.png")
        plot_efficiency_comparison(results, output_dir / f"{base_name}_efficiency.png")
        plot_tpot_distribution(results, output_dir / f"{base_name}_tpot.png")
        generate_latex_table(results, output_dir / f"{base_name}_table.tex")
        generate_markdown_table(results, output_dir / f"{base_name}_table.md")
    else:
        # Process entire results directory
        generate_summary_report(args.results_dir, args.output_dir)


if __name__ == "__main__":
    main()
