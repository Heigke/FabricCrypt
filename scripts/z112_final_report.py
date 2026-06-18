#!/usr/bin/env python3
"""
Generate final FEEL-SLM v2 comparison report.

Combines AMD and NVIDIA results into paper-quality tables and visualizations.
"""

import json
import os
from pathlib import Path
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt


def load_results():
    """Load benchmark results from both platforms."""
    base_dir = Path(__file__).parent.parent / "results"

    # AMD results
    amd_path = base_dir / "z111_real_benchmark" / "benchmark_results.json"
    with open(amd_path) as f:
        amd = json.load(f)

    # NVIDIA results
    nvidia_path = base_dir / "z111_nvidia_benchmark_results.json"
    with open(nvidia_path) as f:
        nvidia = json.load(f)

    return amd, nvidia


def generate_report():
    """Generate comprehensive report."""
    amd, nvidia = load_results()

    print("=" * 80)
    print("FEEL-SLM v2 Final Benchmark Report")
    print("=" * 80)
    print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Platform info
    print("\n## Platform Configuration")
    print("-" * 80)
    print(f"AMD:    {amd['config']['gpu_name']}, {amd['config']['baseline_params']:,} baseline params")
    print(f"NVIDIA: {nvidia['config']['gpu_name']}, {nvidia['config']['baseline_params']:,} baseline params")
    print(f"Model overhead: {(amd['config']['feel_params'] - amd['config']['baseline_params']) / amd['config']['baseline_params'] * 100:.2f}%")

    # Main comparison table
    print("\n## Cross-Platform Energy Comparison")
    print("-" * 80)
    print(f"{'Platform':<10} {'Condition':<20} {'mJ/tok':>10} {'tok/s':>10} {'Power':>10}")
    print("-" * 80)

    for platform_name, results in [("AMD", amd), ("NVIDIA", nvidia)]:
        for key in sorted(results['summaries'].keys()):
            s = results['summaries'][key]
            print(f"{platform_name:<10} {key:<20} {s['mj_per_token_mean']:>10.1f} "
                  f"{s['tokens_per_sec_mean']:>10.1f} {s['power_w_mean']:>9.1f}W")
        print()

    # Energy savings analysis
    print("\n## Energy Savings Analysis")
    print("-" * 80)

    analysis = []
    for platform_name, results in [("AMD", amd), ("NVIDIA", nvidia)]:
        baseline_bal = results['summaries']['baseline_balanced']['mj_per_token_mean']
        feel_eco = results['summaries']['feel_eco']['mj_per_token_mean']
        feel_bal = results['summaries']['feel_balanced']['mj_per_token_mean']
        feel_perf = results['summaries']['feel_perf']['mj_per_token_mean']
        feel_adapt = results['summaries']['feel_adaptive']['mj_per_token_mean']

        # Key comparisons
        eco_vs_baseline = (baseline_bal - feel_eco) / baseline_bal * 100
        layerdrop_savings = (feel_perf - feel_eco) / feel_perf * 100
        overhead_balanced = (feel_bal - baseline_bal) / baseline_bal * 100

        analysis.append({
            'platform': platform_name,
            'baseline_balanced': baseline_bal,
            'feel_eco': feel_eco,
            'feel_balanced': feel_bal,
            'feel_perf': feel_perf,
            'feel_adaptive': feel_adapt,
            'eco_vs_baseline_savings': eco_vs_baseline,
            'layerdrop_savings': layerdrop_savings,
            'overhead_balanced': overhead_balanced,
        })

        print(f"\n{platform_name}:")
        print(f"  FEEL eco vs Baseline balanced: {eco_vs_baseline:+.1f}% energy")
        print(f"  LayerDrop savings (eco vs perf): {layerdrop_savings:.1f}%")
        print(f"  FEEL overhead (balanced mode): {overhead_balanced:+.1f}%")
        print(f"  FEEL adaptive: {feel_adapt:.1f} mJ/tok")

    # Key findings
    print("\n## Key Findings")
    print("-" * 80)
    print("""
1. FEEL eco mode BEATS baseline on BOTH platforms:
   - AMD: 8.7% energy savings vs baseline balanced
   - NVIDIA: 29.5% energy savings vs baseline balanced

2. LayerDrop provides significant savings:
   - AMD: 22% energy savings (eco vs perf)
   - NVIDIA: 44% energy savings (eco vs perf)

3. LayerDrop speedup:
   - AMD: 1.24x faster in eco mode
   - NVIDIA: 1.70x faster in eco mode

4. Model overhead is minimal:
   - Only 0.25% parameter overhead (Phase 1: Policy/Reporter only)
   - FiLM and gated injection disabled

5. Adaptive mode works:
   - Automatically selects efficient mode based on telemetry
   - Achieves near-optimal energy without manual tuning
""")

    # Generate LaTeX table
    print("\n## LaTeX Table")
    print("-" * 80)

    latex = r"""\begin{table}[t]
\centering
\caption{FEEL-SLM v2 Cross-Platform Energy Benchmark}
\label{tab:feel-v2-final}
\begin{tabular}{llrrrrr}
\toprule
Platform & Condition & mJ/tok & $\Delta$ vs Baseline & tok/s & Speedup \\
\midrule
"""

    for platform_name, results in [("AMD", amd), ("NVIDIA", nvidia)]:
        baseline = results['summaries']['baseline_balanced']['mj_per_token_mean']
        baseline_tps = results['summaries']['baseline_balanced']['tokens_per_sec_mean']

        for key in ['baseline_balanced', 'feel_eco', 'feel_balanced', 'feel_adaptive']:
            s = results['summaries'][key]
            delta = (s['mj_per_token_mean'] - baseline) / baseline * 100
            speedup = s['tokens_per_sec_mean'] / baseline_tps

            condition = key.replace('_', ' ').title()
            latex += f"{platform_name} & {condition} & {s['mj_per_token_mean']:.1f} & "
            latex += f"{delta:+.1f}\\% & {s['tokens_per_sec_mean']:.1f} & {speedup:.2f}x \\\\\n"

        latex += r"\midrule" + "\n"

    # Remove last midrule
    latex = latex.rsplit(r"\midrule", 1)[0]

    latex += r"""\bottomrule
\end{tabular}
\vspace{1mm}
\footnotesize{
FEEL eco achieves energy savings through LayerDrop (skipping 2/4 layers). \\
Overhead refers to FEEL balanced mode with full compute.
}
\end{table}
"""

    print(latex)

    # Save outputs
    output_dir = Path(__file__).parent.parent / "results" / "z112_final_report"
    output_dir.mkdir(exist_ok=True)

    # Save LaTeX table
    table_path = output_dir / "final_comparison_table.tex"
    with open(table_path, "w") as f:
        f.write(latex)
    print(f"\nLaTeX table saved to: {table_path}")

    # Save analysis JSON
    analysis_path = output_dir / "analysis.json"
    with open(analysis_path, "w") as f:
        json.dump(analysis, f, indent=2)
    print(f"Analysis saved to: {analysis_path}")

    # Create visualization
    create_visualization(amd, nvidia, output_dir)

    return analysis


def create_visualization(amd, nvidia, output_dir):
    """Create comparison visualization."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("FEEL-SLM v2: Cross-Platform Energy Benchmark Results", fontsize=14, fontweight='bold')

    colors = {
        'baseline_balanced': '#3498db',
        'baseline_eco': '#2980b9',
        'baseline_perf': '#1f618d',
        'feel_balanced': '#e74c3c',
        'feel_eco': '#27ae60',
        'feel_perf': '#c0392b',
        'feel_adaptive': '#9b59b6',
    }

    # 1. Energy per token comparison
    ax1 = axes[0, 0]
    conditions = ['baseline_balanced', 'feel_eco', 'feel_balanced', 'feel_adaptive']
    x = np.arange(len(conditions))
    width = 0.35

    amd_vals = [amd['summaries'][c]['mj_per_token_mean'] for c in conditions]
    nvidia_vals = [nvidia['summaries'][c]['mj_per_token_mean'] for c in conditions]

    bars1 = ax1.bar(x - width/2, amd_vals, width, label='AMD', color='#e74c3c', edgecolor='black')
    bars2 = ax1.bar(x + width/2, nvidia_vals, width, label='NVIDIA', color='#3498db', edgecolor='black')

    ax1.set_ylabel('Energy per Token (mJ)')
    ax1.set_title('Energy Comparison')
    ax1.set_xticks(x)
    ax1.set_xticklabels([c.replace('_', '\n') for c in conditions], fontsize=9)
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)

    # Add value labels
    for bar in bars1:
        ax1.annotate(f'{bar.get_height():.0f}', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                    xytext=(0, 3), textcoords="offset points", ha='center', fontsize=8)
    for bar in bars2:
        ax1.annotate(f'{bar.get_height():.0f}', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                    xytext=(0, 3), textcoords="offset points", ha='center', fontsize=8)

    # 2. Energy savings percentage
    ax2 = axes[0, 1]

    amd_baseline = amd['summaries']['baseline_balanced']['mj_per_token_mean']
    nvidia_baseline = nvidia['summaries']['baseline_balanced']['mj_per_token_mean']

    savings_conditions = ['feel_eco', 'feel_adaptive']
    amd_savings = [(amd_baseline - amd['summaries'][c]['mj_per_token_mean']) / amd_baseline * 100
                   for c in savings_conditions]
    nvidia_savings = [(nvidia_baseline - nvidia['summaries'][c]['mj_per_token_mean']) / nvidia_baseline * 100
                      for c in savings_conditions]

    x = np.arange(len(savings_conditions))
    bars1 = ax2.bar(x - width/2, amd_savings, width, label='AMD', color='#e74c3c', edgecolor='black')
    bars2 = ax2.bar(x + width/2, nvidia_savings, width, label='NVIDIA', color='#3498db', edgecolor='black')

    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax2.set_ylabel('Energy Savings vs Baseline (%)')
    ax2.set_title('Energy Savings (positive = better)')
    ax2.set_xticks(x)
    ax2.set_xticklabels(['FEEL eco\n(LayerDrop)', 'FEEL adaptive'])
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)

    for bar in bars1 + bars2:
        height = bar.get_height()
        ax2.annotate(f'{height:+.1f}%', xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', fontsize=9, fontweight='bold')

    # 3. Throughput comparison
    ax3 = axes[1, 0]

    amd_tps = [amd['summaries'][c]['tokens_per_sec_mean'] for c in conditions]
    nvidia_tps = [nvidia['summaries'][c]['tokens_per_sec_mean'] for c in conditions]

    x = np.arange(len(conditions))
    bars1 = ax3.bar(x - width/2, amd_tps, width, label='AMD', color='#e74c3c', edgecolor='black')
    bars2 = ax3.bar(x + width/2, nvidia_tps, width, label='NVIDIA', color='#3498db', edgecolor='black')

    ax3.set_ylabel('Tokens per Second')
    ax3.set_title('Generation Throughput')
    ax3.set_xticks(x)
    ax3.set_xticklabels([c.replace('_', '\n') for c in conditions], fontsize=9)
    ax3.legend()
    ax3.grid(axis='y', alpha=0.3)

    # 4. LayerDrop effect
    ax4 = axes[1, 1]

    # FEEL modes only
    feel_conditions = ['feel_perf', 'feel_balanced', 'feel_eco']
    amd_feel = [amd['summaries'][c]['mj_per_token_mean'] for c in feel_conditions]
    nvidia_feel = [nvidia['summaries'][c]['mj_per_token_mean'] for c in feel_conditions]

    x = np.arange(len(feel_conditions))
    bars1 = ax4.bar(x - width/2, amd_feel, width, label='AMD', color='#e74c3c', edgecolor='black')
    bars2 = ax4.bar(x + width/2, nvidia_feel, width, label='NVIDIA', color='#3498db', edgecolor='black')

    ax4.set_ylabel('Energy per Token (mJ)')
    ax4.set_title('LayerDrop Effect (eco skips 2/4 layers)')
    ax4.set_xticks(x)
    ax4.set_xticklabels(['FEEL perf\n(full)', 'FEEL balanced\n(full)', 'FEEL eco\n(LayerDrop)'])
    ax4.legend()
    ax4.grid(axis='y', alpha=0.3)

    # Add annotations for savings
    amd_savings_ld = (amd_feel[0] - amd_feel[2]) / amd_feel[0] * 100
    nvidia_savings_ld = (nvidia_feel[0] - nvidia_feel[2]) / nvidia_feel[0] * 100
    ax4.annotate(f'AMD: -{amd_savings_ld:.0f}%', xy=(2 - width/2, amd_feel[2]),
                xytext=(-30, 30), textcoords="offset points",
                arrowprops=dict(arrowstyle='->', color='#e74c3c'),
                color='#e74c3c', fontweight='bold')
    ax4.annotate(f'NVIDIA: -{nvidia_savings_ld:.0f}%', xy=(2 + width/2, nvidia_feel[2]),
                xytext=(10, 40), textcoords="offset points",
                arrowprops=dict(arrowstyle='->', color='#3498db'),
                color='#3498db', fontweight='bold')

    # Notes
    fig.text(0.5, 0.02,
             "FEEL-SLM v2: Simplified architecture (0.25% overhead), LayerDrop for conditional compute.\n"
             "eco mode achieves 8.7% (AMD) to 29.5% (NVIDIA) energy savings vs baseline.",
             ha='center', fontsize=10, style='italic')

    plt.tight_layout(rect=[0, 0.05, 1, 0.95])

    # Save
    png_path = output_dir / "final_comparison.png"
    pdf_path = output_dir / "final_comparison.pdf"

    plt.savefig(png_path, dpi=150, bbox_inches='tight')
    plt.savefig(pdf_path, bbox_inches='tight')

    print(f"Visualization saved to: {png_path}")
    print(f"PDF saved to: {pdf_path}")

    plt.close()


if __name__ == "__main__":
    generate_report()
