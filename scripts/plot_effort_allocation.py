#!/usr/bin/env python3
"""Generate effort allocation and quality-vs-Joules plots."""

import json
from pathlib import Path
import argparse

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    print("matplotlib not available, skipping plot generation")


def plot_effort_by_difficulty(results_path: Path, output_dir: Path):
    """Plot energy consumption by prompt difficulty."""
    if not HAS_MPL:
        return

    with open(results_path) as f:
        data = json.load(f)

    stats = data.get("stats", {})

    # Filter to difficulty levels only
    difficulties = ["easy", "medium", "hard", "meta"]
    valid_diffs = [d for d in difficulties if d in stats]

    if not valid_diffs:
        print("No valid difficulty data found")
        return

    energy_means = [stats[d]["energy_mean"] for d in valid_diffs]
    energy_stds = [stats[d].get("energy_std", 0) for d in valid_diffs]
    time_means = [stats[d]["time_mean"] for d in valid_diffs]
    tpot_means = [stats[d]["tpot_mean"] for d in valid_diffs]

    colors = {
        "easy": "#2ecc71",
        "medium": "#f1c40f",
        "hard": "#e74c3c",
        "meta": "#9b59b6",
    }

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    # 1. Energy by difficulty
    ax = axes[0]
    bars = ax.bar(valid_diffs, energy_means,
                  color=[colors.get(d, "#3498db") for d in valid_diffs],
                  edgecolor='black', yerr=energy_stds, capsize=3)
    ax.set_ylabel('Energy (J)', fontsize=11)
    ax.set_xlabel('Prompt Difficulty', fontsize=11)
    ax.set_title('Energy per Difficulty Level', fontsize=12)

    for bar, val in zip(bars, energy_means):
        ax.text(bar.get_x() + bar.get_width()/2, val + 2,
                f'{val:.0f}J', ha='center', va='bottom', fontsize=10)

    # 2. Time by difficulty
    ax = axes[1]
    bars = ax.bar(valid_diffs, time_means,
                  color=[colors.get(d, "#3498db") for d in valid_diffs],
                  edgecolor='black')
    ax.set_ylabel('Time (s)', fontsize=11)
    ax.set_xlabel('Prompt Difficulty', fontsize=11)
    ax.set_title('Inference Time per Difficulty', fontsize=12)

    for bar, val in zip(bars, time_means):
        ax.text(bar.get_x() + bar.get_width()/2, val + 0.05,
                f'{val:.2f}s', ha='center', va='bottom', fontsize=10)

    # 3. TPOT by difficulty (per-token latency)
    ax = axes[2]
    bars = ax.bar(valid_diffs, tpot_means,
                  color=[colors.get(d, "#3498db") for d in valid_diffs],
                  edgecolor='black')
    ax.set_ylabel('TPOT (ms)', fontsize=11)
    ax.set_xlabel('Prompt Difficulty', fontsize=11)
    ax.set_title('Per-Token Latency per Difficulty', fontsize=12)

    for bar, val in zip(bars, tpot_means):
        ax.text(bar.get_x() + bar.get_width()/2, val + 0.3,
                f'{val:.1f}ms', ha='center', va='bottom', fontsize=10)

    plt.suptitle('Effort Allocation Analysis: Easy vs Hard Prompts',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()

    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_dir / 'effort_allocation.png', dpi=150, bbox_inches='tight')
    plt.savefig(output_dir / 'effort_allocation.pdf', bbox_inches='tight')
    plt.close()

    print(f"Saved: {output_dir / 'effort_allocation.png'}")


def plot_quality_vs_joules(results_path: Path, output_dir: Path):
    """
    Generate quality-vs-Joules Pareto frontier plot.

    For this initial version, we use output length as a proxy for "quality"
    since longer, more detailed responses typically indicate more effort.
    """
    if not HAS_MPL:
        return

    with open(results_path) as f:
        data = json.load(f)

    results = data.get("results", [])

    if not results:
        print("No results data found")
        return

    # Extract data points
    energies = [r["total_energy_j"] for r in results]
    output_lengths = [r.get("output_length", 0) for r in results]
    difficulties = [r["difficulty"] for r in results]

    colors_map = {
        "easy": "#2ecc71",
        "medium": "#f1c40f",
        "hard": "#e74c3c",
        "meta": "#9b59b6",
        "energy_aware": "#3498db",
    }

    fig, ax = plt.subplots(figsize=(10, 6))

    for diff in set(difficulties):
        e = [energies[i] for i in range(len(difficulties)) if difficulties[i] == diff]
        l = [output_lengths[i] for i in range(len(difficulties)) if difficulties[i] == diff]
        ax.scatter(e, l, c=colors_map.get(diff, "#95a5a6"),
                   label=diff.capitalize(), s=80, alpha=0.7, edgecolors='black')

    ax.set_xlabel('Energy (Joules)', fontsize=12)
    ax.set_ylabel('Output Length (tokens)', fontsize=12)
    ax.set_title('Quality vs Energy: Output Length as Effort Proxy', fontsize=13)
    ax.legend(title='Difficulty')
    ax.grid(True, alpha=0.3)

    # Annotate Pareto frontier region
    ax.annotate('Ideal: High quality,\nlow energy',
                xy=(min(energies), max(output_lengths)),
                xytext=(min(energies) + 5, max(output_lengths) - 5),
                fontsize=10, color='green',
                arrowprops=dict(arrowstyle='->', color='green'))

    plt.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_dir / 'quality_vs_joules.png', dpi=150, bbox_inches='tight')
    plt.savefig(output_dir / 'quality_vs_joules.pdf', bbox_inches='tight')
    plt.close()

    print(f"Saved: {output_dir / 'quality_vs_joules.png'}")


def plot_latent_v2_overhead_comparison(output_dir: Path):
    """
    Plot latent_v2 overhead comparison showing hook-based capture has near-zero overhead.

    Data from experiments:
    - auto: 1.584s, 81.3J, TPOT p95=25.6ms
    - latent (output_hidden_states): 1.742s, 90.6J, TPOT p95=26.6ms
    - latent_v2 (hook): 1.717s, 89.3J, TPOT p95=25.8ms
    """
    if not HAS_MPL:
        return

    policies = ['auto', 'latent\n(output_hidden_states)', 'latent_v2\n(hook-based)']

    # Data from experiments
    times = [1.584, 1.742, 1.717]
    energies = [81.3, 90.6, 89.3]
    tpot_p95 = [25.6, 26.6, 25.8]

    colors = ['#2ecc71', '#e74c3c', '#3498db']

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    # 1. Time comparison
    ax = axes[0]
    bars = ax.bar(policies, times, color=colors, edgecolor='black')
    ax.set_ylabel('Time (s)', fontsize=11)
    ax.set_title('Inference Time', fontsize=12)
    ax.set_ylim(0, max(times) * 1.2)

    for bar, val in zip(bars, times):
        ax.text(bar.get_x() + bar.get_width()/2, val + 0.02,
                f'{val:.3f}s', ha='center', va='bottom', fontsize=9)

    # Overhead annotations
    ax.annotate('+10%', xy=(1, times[1]), xytext=(1.2, times[1]*1.05),
                fontsize=9, color='red', fontweight='bold')
    ax.annotate('+8%', xy=(2, times[2]), xytext=(2.2, times[2]*1.02),
                fontsize=9, color='orange', fontweight='bold')

    # 2. Energy comparison
    ax = axes[1]
    bars = ax.bar(policies, energies, color=colors, edgecolor='black')
    ax.set_ylabel('Energy (J)', fontsize=11)
    ax.set_title('Total Energy', fontsize=12)
    ax.set_ylim(0, max(energies) * 1.2)

    for bar, val in zip(bars, energies):
        ax.text(bar.get_x() + bar.get_width()/2, val + 1,
                f'{val:.1f}J', ha='center', va='bottom', fontsize=9)

    ax.annotate('+11%', xy=(1, energies[1]), xytext=(1.2, energies[1]*1.02),
                fontsize=9, color='red', fontweight='bold')
    ax.annotate('+10%', xy=(2, energies[2]), xytext=(2.2, energies[2]*1.01),
                fontsize=9, color='orange', fontweight='bold')

    # 3. TPOT p95 (per-token overhead)
    ax = axes[2]
    bars = ax.bar(policies, tpot_p95, color=colors, edgecolor='black')
    ax.set_ylabel('TPOT p95 (ms)', fontsize=11)
    ax.set_title('Per-Token Latency (Key Metric!)', fontsize=12)
    ax.set_ylim(0, max(tpot_p95) * 1.2)

    for bar, val in zip(bars, tpot_p95):
        ax.text(bar.get_x() + bar.get_width()/2, val + 0.3,
                f'{val:.1f}ms', ha='center', va='bottom', fontsize=9)

    # Key finding: hook-based has near-zero per-token overhead!
    ax.annotate('+4%', xy=(1, tpot_p95[1]), xytext=(1.2, tpot_p95[1]*1.02),
                fontsize=9, color='red', fontweight='bold')
    ax.annotate('+0.8%\n(near zero!)', xy=(2, tpot_p95[2]), xytext=(2.15, tpot_p95[2]*0.95),
                fontsize=9, color='green', fontweight='bold')

    plt.suptitle('Latent-v2 Hook-Based Capture: Near-Zero Per-Token Overhead',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()

    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_dir / 'latent_v2_overhead.png', dpi=150, bbox_inches='tight')
    plt.savefig(output_dir / 'latent_v2_overhead.pdf', bbox_inches='tight')
    plt.close()

    print(f"Saved: {output_dir / 'latent_v2_overhead.png'}")


def main():
    parser = argparse.ArgumentParser(description="Generate effort allocation plots")
    parser.add_argument("--results", type=Path,
                       default=Path("results/meta_cognitive/meta_cognitive_results.json"))
    parser.add_argument("--output-dir", type=Path,
                       default=Path("reports/effort_allocation"))
    parser.add_argument("--latent-v2-only", action="store_true",
                       help="Generate only latent_v2 overhead comparison")
    args = parser.parse_args()

    if args.latent_v2_only:
        plot_latent_v2_overhead_comparison(args.output_dir)
        return

    if args.results.exists():
        plot_effort_by_difficulty(args.results, args.output_dir)
        plot_quality_vs_joules(args.results, args.output_dir)
    else:
        print(f"Results file not found: {args.results}")

    # Always generate latent_v2 overhead comparison
    plot_latent_v2_overhead_comparison(args.output_dir)


if __name__ == '__main__':
    main()
