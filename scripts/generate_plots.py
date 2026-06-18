#!/usr/bin/env python3
"""
Generate correlation plots for PowerTraceLLM-AMD research.

Plots:
1. Entropy vs Power vs TPOT correlation
2. Regime map (model size vs efficiency delta)
3. Policy comparison across models
4. Signal trace visualization
"""

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
import seaborn as sns

# Set style for publication-quality plots
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("colorblind")


def load_comparison_results(results_dir: Path) -> dict:
    """Load comparison results from JSON."""
    json_path = results_dir / "comparison_extended.json"
    if json_path.exists():
        with open(json_path) as f:
            return json.load(f)
    return {}


def load_signal_trace(trace_path: Path) -> pd.DataFrame:
    """Load signal trace CSV."""
    if trace_path.exists():
        return pd.read_csv(trace_path)
    return pd.DataFrame()


def plot_regime_map(results_3b: dict, results_05b: dict, output_dir: Path):
    """
    Plot regime map showing which models benefit from active control.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    models = ['0.5B', '3B']
    auto_efficiency = []
    controller_efficiency = []

    for results in [results_05b, results_3b]:
        if results:
            # Handle both flat and nested JSON structures
            auto_data = results.get('auto', {})
            ctrl_data = results.get('controller', {})
            auto_efficiency.append(auto_data.get('total_tok_per_j_mean', 0))
            controller_efficiency.append(ctrl_data.get('total_tok_per_j_mean', 0))

    x = np.arange(len(models))
    width = 0.35

    bars1 = ax.bar(x - width/2, auto_efficiency, width, label='Auto DVFS', color='#1f77b4')
    bars2 = ax.bar(x + width/2, controller_efficiency, width, label='Active Controller', color='#ff7f0e')

    ax.set_xlabel('Model Size', fontsize=12)
    ax.set_ylabel('Energy Efficiency (tok/J)', fontsize=12)
    ax.set_title('Regime Map: When Does Active DVFS Control Help?', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.legend()

    # Add percentage difference annotations
    for i, (auto, ctrl) in enumerate(zip(auto_efficiency, controller_efficiency)):
        if auto > 0:
            diff = (ctrl - auto) / auto * 100
            color = 'green' if diff > 0 else 'red'
            ax.annotate(f'{diff:+.1f}%',
                       xy=(i + width/2, ctrl),
                       xytext=(5, 5),
                       textcoords='offset points',
                       fontsize=10, color=color)

    # Add regime boundary annotation
    ax.axhline(y=3.5, color='gray', linestyle='--', alpha=0.5)
    ax.text(0.02, 3.6, 'Regime Boundary: ≤1B → auto wins', fontsize=10, color='gray')

    plt.tight_layout()
    plt.savefig(output_dir / 'regime_map.png', dpi=150, bbox_inches='tight')
    plt.savefig(output_dir / 'regime_map.pdf', bbox_inches='tight')
    plt.close()
    print(f"Saved regime_map.png/pdf")


def plot_policy_comparison(results: dict, model_name: str, output_dir: Path):
    """
    Plot comparison of all policies for a single model.
    """
    if not results:
        return

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    policies = ['auto', 'peak', 'phase_split', 'controller', 'signal_controller']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

    # Prepare data - handle flat JSON structure
    efficiency = []
    energy = []
    tpot_p95 = []

    for policy in policies:
        if policy in results:
            policy_data = results[policy]
            efficiency.append(policy_data.get('total_tok_per_j_mean', 0))
            energy.append(policy_data.get('total_energy_j_mean', 0))
            tpot_p95.append(policy_data.get('tpot_s_p95_mean', 0) * 1000)  # Convert to ms
        else:
            efficiency.append(0)
            energy.append(0)
            tpot_p95.append(0)

    x = np.arange(len(policies))

    # Plot 1: Energy Efficiency
    axes[0].bar(x, efficiency, color=colors)
    axes[0].set_xlabel('Policy')
    axes[0].set_ylabel('Efficiency (tok/J)')
    axes[0].set_title(f'{model_name}: Energy Efficiency')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(policies, rotation=45, ha='right')

    # Plot 2: Total Energy
    axes[1].bar(x, energy, color=colors)
    axes[1].set_xlabel('Policy')
    axes[1].set_ylabel('Energy (J)')
    axes[1].set_title(f'{model_name}: Total Energy')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(policies, rotation=45, ha='right')

    # Plot 3: TPOT P95
    axes[2].bar(x, tpot_p95, color=colors)
    axes[2].set_xlabel('Policy')
    axes[2].set_ylabel('TPOT P95 (ms)')
    axes[2].set_title(f'{model_name}: Latency Tail')
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(policies, rotation=45, ha='right')

    plt.tight_layout()
    filename = f'policy_comparison_{model_name.replace("/", "_").lower()}'
    plt.savefig(output_dir / f'{filename}.png', dpi=150, bbox_inches='tight')
    plt.savefig(output_dir / f'{filename}.pdf', bbox_inches='tight')
    plt.close()
    print(f"Saved {filename}.png/pdf")


def plot_signal_trace(trace_dir: Path, output_dir: Path):
    """
    Plot signal trace showing entropy, power, and policy over time.
    """
    # Find a signal trace file
    signal_files = list(trace_dir.glob("*_signals.csv"))
    power_files = list(trace_dir.glob("trace_signal_controller*.csv"))

    if not signal_files:
        print("No signal trace files found")
        return

    # Load first signal trace
    signal_df = pd.read_csv(signal_files[0])

    # Find matching power trace
    matching_power = None
    for pf in power_files:
        if "_signals" not in pf.name:
            matching_power = pd.read_csv(pf)
            break

    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

    # Plot 1: Entropy over tokens
    axes[0].plot(signal_df['token_idx'], signal_df['entropy'], 'b-', linewidth=0.8)
    axes[0].fill_between(signal_df['token_idx'], 0, signal_df['entropy'], alpha=0.3)
    axes[0].set_ylabel('Entropy (bits)')
    axes[0].set_title('Internal Signals During Decode')
    axes[0].axhline(y=signal_df['entropy'].mean(), color='r', linestyle='--', alpha=0.5, label=f'Mean: {signal_df["entropy"].mean():.2f}')
    axes[0].legend()

    # Plot 2: Difficulty EMA (Z-score in V2)
    axes[1].plot(signal_df['token_idx'], signal_df['difficulty_ema'], 'g-', linewidth=0.8)
    axes[1].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    axes[1].set_ylabel('Difficulty (EMA)')

    # Shade policy regions
    policies = signal_df['policy'].unique()
    policy_colors = {'auto': 'yellow', 'peak': 'red', 'min_sclk': 'green'}
    for i in range(len(signal_df) - 1):
        policy = signal_df['policy'].iloc[i]
        axes[1].axvspan(i, i+1, alpha=0.2, color=policy_colors.get(policy, 'gray'))

    # Plot 3: Power if available
    if matching_power is not None:
        axes[2].plot(matching_power['elapsed_ms'], matching_power['power_watts'], 'r-', linewidth=0.8)
        axes[2].set_ylabel('Power (W)')
        axes[2].set_xlabel('Time (ms)')
    else:
        axes[2].plot(signal_df['token_idx'], signal_df['margin'], 'm-', linewidth=0.8)
        axes[2].set_ylabel('Margin (p1-p2)')
        axes[2].set_xlabel('Token Index')

    plt.tight_layout()
    plt.savefig(output_dir / 'signal_trace.png', dpi=150, bbox_inches='tight')
    plt.savefig(output_dir / 'signal_trace.pdf', bbox_inches='tight')
    plt.close()
    print(f"Saved signal_trace.png/pdf")


def plot_entropy_power_correlation(trace_dir: Path, output_dir: Path):
    """
    Plot correlation between entropy and power consumption.
    """
    signal_files = list(trace_dir.glob("*_signals.csv"))
    power_files = [f for f in trace_dir.glob("trace_signal_controller*.csv") if "_signals" not in f.name]

    if not signal_files or not power_files:
        print("Missing files for correlation plot")
        return

    # Load data
    signal_df = pd.read_csv(signal_files[0])
    power_df = pd.read_csv(power_files[0])

    # Resample power to match token count (approximate)
    n_tokens = len(signal_df)
    n_samples = len(power_df)

    if n_samples > n_tokens:
        # Downsample power to token resolution
        indices = np.linspace(0, n_samples-1, n_tokens, dtype=int)
        power_sampled = power_df['power_watts'].iloc[indices].values
    else:
        power_sampled = power_df['power_watts'].values[:n_tokens]

    if len(power_sampled) != len(signal_df):
        min_len = min(len(power_sampled), len(signal_df))
        power_sampled = power_sampled[:min_len]
        entropy = signal_df['entropy'].values[:min_len]
    else:
        entropy = signal_df['entropy'].values

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Scatter plot
    axes[0].scatter(entropy, power_sampled, alpha=0.5, s=10)
    axes[0].set_xlabel('Entropy (bits)')
    axes[0].set_ylabel('Power (W)')
    axes[0].set_title('Entropy vs Power Correlation')

    # Calculate correlation
    corr = np.corrcoef(entropy, power_sampled)[0, 1]
    axes[0].text(0.05, 0.95, f'r = {corr:.3f}', transform=axes[0].transAxes,
                 fontsize=12, verticalalignment='top')

    # Time series comparison
    token_idx = np.arange(len(entropy))
    ax1 = axes[1]
    ax2 = ax1.twinx()

    line1, = ax1.plot(token_idx, entropy, 'b-', alpha=0.7, label='Entropy')
    line2, = ax2.plot(token_idx, power_sampled, 'r-', alpha=0.7, label='Power')

    ax1.set_xlabel('Token Index')
    ax1.set_ylabel('Entropy (bits)', color='b')
    ax2.set_ylabel('Power (W)', color='r')
    ax1.set_title('Entropy and Power Over Time')
    ax1.legend([line1, line2], ['Entropy', 'Power'], loc='upper right')

    plt.tight_layout()
    plt.savefig(output_dir / 'entropy_power_correlation.png', dpi=150, bbox_inches='tight')
    plt.savefig(output_dir / 'entropy_power_correlation.pdf', bbox_inches='tight')
    plt.close()
    print(f"Saved entropy_power_correlation.png/pdf")


def create_summary_table(results_3b: dict, results_05b: dict, output_dir: Path):
    """
    Create a summary table of key results.
    """
    rows = []

    for model_name, results in [('Qwen2.5-0.5B', results_05b), ('Qwen2.5-3B', results_3b)]:
        if not results:
            continue

        for policy in ['auto', 'controller', 'signal_controller']:
            if policy in results:
                policy_data = results[policy]
                rows.append({
                    'Model': model_name,
                    'Policy': policy,
                    'Total (s)': f"{policy_data.get('total_s_mean', 0):.2f}",
                    'TPOT P95 (ms)': f"{policy_data.get('tpot_s_p95_mean', 0)*1000:.1f}",
                    'Energy (J)': f"{policy_data.get('total_energy_j_mean', 0):.0f}",
                    'Efficiency (tok/J)': f"{policy_data.get('total_tok_per_j_mean', 0):.2f}",
                })

    if rows:
        df = pd.DataFrame(rows)
        df.to_csv(output_dir / 'results_summary.csv', index=False)
        print(f"Saved results_summary.csv")
        print("\n" + df.to_string(index=False))


def main():
    parser = argparse.ArgumentParser(description='Generate research plots')
    parser.add_argument('--results-3b', type=Path,
                       default=Path('results/signal_matrix/qwen3b_p2k_d256'),
                       help='Path to 3B model results')
    parser.add_argument('--results-05b', type=Path,
                       default=Path('results/signal_matrix/qwen05b_p2k_d256'),
                       help='Path to 0.5B model results')
    parser.add_argument('--output', type=Path,
                       default=Path('results/plots'),
                       help='Output directory for plots')
    args = parser.parse_args()

    # Create output directory
    args.output.mkdir(parents=True, exist_ok=True)

    # Load results
    results_3b = load_comparison_results(args.results_3b)
    results_05b = load_comparison_results(args.results_05b)

    print("Generating plots...")

    # Generate plots
    plot_regime_map(results_3b, results_05b, args.output)
    plot_policy_comparison(results_3b, 'Qwen2.5-3B', args.output)
    plot_policy_comparison(results_05b, 'Qwen2.5-0.5B', args.output)

    if args.results_3b.exists():
        plot_signal_trace(args.results_3b, args.output)
        plot_entropy_power_correlation(args.results_3b, args.output)

    create_summary_table(results_3b, results_05b, args.output)

    print(f"\nAll plots saved to {args.output}")


if __name__ == '__main__':
    main()
