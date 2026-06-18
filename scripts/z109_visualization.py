#!/usr/bin/env python3
"""Generate visualization comparing AMD vs NVIDIA FEEL-SLM benchmarks."""

import json
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path


def load_results():
    """Load benchmark results from both platforms."""
    base_dir = Path(__file__).parent.parent / "results"

    # AMD results
    amd_path = base_dir / "z106_benchmark" / "benchmark_results.json"
    with open(amd_path) as f:
        amd_results = json.load(f)

    # NVIDIA results
    nvidia_path = base_dir / "z107_nvidia_benchmark" / "benchmark_results.json"
    with open(nvidia_path) as f:
        nvidia_results = json.load(f)

    return amd_results, nvidia_results


def create_visualization():
    """Create comprehensive visualization."""
    amd, nvidia = load_results()

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("FEEL-SLM Cross-Platform Energy Benchmark\n(Untrained Models)", fontsize=14, fontweight='bold')

    colors = {'baseline': '#2ecc71', 'feel': '#e74c3c'}

    # =========================================================================
    # 1. Energy per Token (Absolute)
    # =========================================================================
    ax1 = axes[0, 0]

    # Use log scale due to huge difference between AMD and NVIDIA
    platforms = ['AMD\n(Radeon 8060S)', 'NVIDIA\n(RTX A6000)']
    baseline_mj = [amd['summaries']['baseline']['mj_per_token_mean'],
                   nvidia['summaries']['baseline']['mj_per_token_mean']]
    feel_mj = [amd['summaries']['feel']['mj_per_token_mean'],
               nvidia['summaries']['feel']['mj_per_token_mean']]

    x = np.arange(len(platforms))
    width = 0.35

    bars1 = ax1.bar(x - width/2, baseline_mj, width, label='Baseline', color=colors['baseline'], edgecolor='black')
    bars2 = ax1.bar(x + width/2, feel_mj, width, label='FEEL', color=colors['feel'], edgecolor='black')

    ax1.set_ylabel('Energy per Token (mJ)', fontsize=11)
    ax1.set_title('Energy Consumption (mJ/token)', fontsize=12)
    ax1.set_xticks(x)
    ax1.set_xticklabels(platforms)
    ax1.set_yscale('log')
    ax1.legend(loc='upper left')
    ax1.grid(axis='y', alpha=0.3)

    # Add value labels
    for bar in bars1 + bars2:
        height = bar.get_height()
        ax1.annotate(f'{height:.0f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=9)

    # =========================================================================
    # 2. Overhead Comparison (%)
    # =========================================================================
    ax2 = axes[0, 1]

    amd_overhead = (amd['summaries']['feel']['mj_per_token_mean'] -
                   amd['summaries']['baseline']['mj_per_token_mean']) / \
                  amd['summaries']['baseline']['mj_per_token_mean'] * 100

    nvidia_overhead = (nvidia['summaries']['feel']['mj_per_token_mean'] -
                      nvidia['summaries']['baseline']['mj_per_token_mean']) / \
                     nvidia['summaries']['baseline']['mj_per_token_mean'] * 100

    param_overhead_amd = (amd['config']['feel_params'] - amd['config']['baseline_params']) / \
                        amd['config']['baseline_params'] * 100
    param_overhead_nvidia = (nvidia['config']['feel_params'] - nvidia['config']['baseline_params']) / \
                           nvidia['config']['baseline_params'] * 100

    x = np.arange(2)
    width = 0.35

    bars_energy = ax2.bar(x - width/2, [amd_overhead, nvidia_overhead], width,
                         label='Energy Overhead', color='#e74c3c', edgecolor='black')
    bars_param = ax2.bar(x + width/2, [param_overhead_amd, param_overhead_nvidia], width,
                        label='Parameter Overhead', color='#3498db', edgecolor='black')

    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax2.set_ylabel('Overhead (%)', fontsize=11)
    ax2.set_title('FEEL Overhead vs Baseline', fontsize=12)
    ax2.set_xticks(x)
    ax2.set_xticklabels(['AMD', 'NVIDIA'])
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)

    # Add value labels
    for bar in bars_energy:
        height = bar.get_height()
        ax2.annotate(f'{height:.1f}%',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=9, fontweight='bold')
    for bar in bars_param:
        height = bar.get_height()
        ax2.annotate(f'{height:.1f}%',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=9)

    # =========================================================================
    # 3. Throughput Comparison
    # =========================================================================
    ax3 = axes[1, 0]

    baseline_tps = [amd['summaries']['baseline']['tokens_per_sec_mean'],
                   nvidia['summaries']['baseline']['tokens_per_sec_mean']]
    feel_tps = [amd['summaries']['feel']['tokens_per_sec_mean'],
               nvidia['summaries']['feel']['tokens_per_sec_mean']]

    x = np.arange(len(platforms))
    bars1 = ax3.bar(x - width/2, baseline_tps, width, label='Baseline', color=colors['baseline'], edgecolor='black')
    bars2 = ax3.bar(x + width/2, feel_tps, width, label='FEEL', color=colors['feel'], edgecolor='black')

    ax3.set_ylabel('Tokens per Second', fontsize=11)
    ax3.set_title('Generation Throughput', fontsize=12)
    ax3.set_xticks(x)
    ax3.set_xticklabels(platforms)
    ax3.legend()
    ax3.grid(axis='y', alpha=0.3)

    for bar in bars1 + bars2:
        height = bar.get_height()
        ax3.annotate(f'{height:.1f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=9)

    # =========================================================================
    # 4. Power Draw Comparison
    # =========================================================================
    ax4 = axes[1, 1]

    baseline_power = [amd['summaries']['baseline']['power_w_mean'],
                     nvidia['summaries']['baseline']['power_w_mean']]
    feel_power = [amd['summaries']['feel']['power_w_mean'],
                 nvidia['summaries']['feel']['power_w_mean']]

    x = np.arange(len(platforms))
    bars1 = ax4.bar(x - width/2, baseline_power, width, label='Baseline', color=colors['baseline'], edgecolor='black')
    bars2 = ax4.bar(x + width/2, feel_power, width, label='FEEL', color=colors['feel'], edgecolor='black')

    ax4.set_ylabel('Average Power (W)', fontsize=11)
    ax4.set_title('Power Consumption', fontsize=12)
    ax4.set_xticks(x)
    ax4.set_xticklabels(platforms)
    ax4.legend()
    ax4.grid(axis='y', alpha=0.3)

    for bar in bars1 + bars2:
        height = bar.get_height()
        ax4.annotate(f'{height:.0f}W',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=9)

    # Add notes
    fig.text(0.5, 0.02,
             "Note: Higher energy overhead on AMD APU vs NVIDIA discrete GPU suggests different optimization strategies may be needed.\n"
             "Training is expected to reduce FEEL overhead by enabling adaptive computation based on body signals.",
             ha='center', fontsize=9, style='italic')

    plt.tight_layout(rect=[0, 0.05, 1, 0.95])

    # Save figure
    output_dir = Path(__file__).parent.parent / "results" / "comparison"
    output_dir.mkdir(exist_ok=True)

    output_path = output_dir / "cross_platform_comparison.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Visualization saved to: {output_path}")

    # Also save as PDF for paper
    pdf_path = output_dir / "cross_platform_comparison.pdf"
    plt.savefig(pdf_path, bbox_inches='tight')
    print(f"PDF saved to: {pdf_path}")

    plt.close()

    return output_path


if __name__ == "__main__":
    create_visualization()
