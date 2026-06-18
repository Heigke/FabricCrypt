#!/usr/bin/env python3
"""Generate plots for regime matrix V2 results."""

import json
from pathlib import Path
import numpy as np

# Check for matplotlib
try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    print("matplotlib not available, skipping plot generation")


def load_results(results_dir: Path):
    """Load all results from subdirectories."""
    results = {}

    for subdir in results_dir.iterdir():
        if not subdir.is_dir():
            continue

        parts = subdir.name.split('_')
        model_size = float(parts[0].replace('B', ''))
        prompt_tokens = int(parts[1].replace('p', ''))
        decode_tokens = int(parts[2].replace('d', ''))
        policy = parts[5]

        tok_per_j_values = []

        for run_dir in subdir.glob('run_*'):
            json_file = run_dir / 'comparison_extended.json'
            if json_file.exists():
                with open(json_file) as f:
                    data = json.load(f)
                    policy_data = data.get(policy, {})
                    if policy_data and 'total_tok_per_j_mean' in policy_data:
                        tok_per_j_values.append(policy_data['total_tok_per_j_mean'])

        if tok_per_j_values:
            key = (model_size, prompt_tokens, decode_tokens, policy)
            results[key] = {
                'mean': np.mean(tok_per_j_values),
                'std': np.std(tok_per_j_values),
                'values': tok_per_j_values
            }

    return results


def plot_regime_heatmap(results, output_dir: Path):
    """Create heatmap showing efficiency by condition."""
    if not HAS_MPL:
        return

    # Organize data
    models = sorted(set(k[0] for k in results.keys()))
    prompts = sorted(set(k[1] for k in results.keys()))
    decodes = sorted(set(k[2] for k in results.keys()))
    policies = ['auto', 'controller', 'windowed']

    fig, axes = plt.subplots(1, len(models), figsize=(14, 6), sharey=False)
    if len(models) == 1:
        axes = [axes]

    for idx, model in enumerate(models):
        ax = axes[idx]

        # Create matrix for this model
        conditions = [(p, d) for p in prompts for d in decodes]
        n_conditions = len(conditions)
        n_policies = len(policies)

        matrix = np.zeros((n_conditions, n_policies))

        for i, (prompt, decode) in enumerate(conditions):
            for j, policy in enumerate(policies):
                key = (model, prompt, decode, policy)
                if key in results:
                    matrix[i, j] = results[key]['mean']

        # Plot
        im = ax.imshow(matrix, aspect='auto', cmap='RdYlGn')

        # Labels
        ax.set_xticks(range(n_policies))
        ax.set_xticklabels(policies, fontsize=10)
        ax.set_yticks(range(n_conditions))
        ax.set_yticklabels([f"p{p} d{d}" for p, d in conditions], fontsize=9)

        ax.set_xlabel('Policy', fontsize=11)
        ax.set_ylabel('Condition (prompt/decode)', fontsize=11)
        ax.set_title(f'{model}B Model\nEfficiency (tok/J)', fontsize=12)

        # Add values
        for i in range(n_conditions):
            for j in range(n_policies):
                val = matrix[i, j]
                color = 'white' if val < matrix.max() * 0.5 else 'black'
                ax.text(j, i, f'{val:.2f}', ha='center', va='center', color=color, fontsize=9)

        plt.colorbar(im, ax=ax, label='tok/J')

    plt.tight_layout()
    plt.savefig(output_dir / 'regime_matrix_heatmap.png', dpi=150, bbox_inches='tight')
    plt.savefig(output_dir / 'regime_matrix_heatmap.pdf', bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir / 'regime_matrix_heatmap.png'}")


def plot_policy_comparison_bars(results, output_dir: Path):
    """Create grouped bar chart comparing policies."""
    if not HAS_MPL:
        return

    models = sorted(set(k[0] for k in results.keys()))
    prompts = sorted(set(k[1] for k in results.keys()))
    decodes = sorted(set(k[2] for k in results.keys()))
    policies = ['auto', 'controller', 'windowed']
    colors = {'auto': '#2ecc71', 'controller': '#3498db', 'windowed': '#e74c3c'}

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    plot_idx = 0
    for model in models:
        for decode in decodes:
            ax = axes[plot_idx]

            x = np.arange(len(prompts))
            width = 0.25

            for i, policy in enumerate(policies):
                values = []
                errors = []
                for prompt in prompts:
                    key = (model, prompt, decode, policy)
                    if key in results:
                        values.append(results[key]['mean'])
                        errors.append(results[key]['std'])
                    else:
                        values.append(0)
                        errors.append(0)

                ax.bar(x + i * width, values, width, label=policy, color=colors[policy],
                       yerr=errors, capsize=3)

            ax.set_xlabel('Prompt Tokens', fontsize=11)
            ax.set_ylabel('Efficiency (tok/J)', fontsize=11)
            ax.set_title(f'{model}B Model, Decode={decode}', fontsize=12)
            ax.set_xticks(x + width)
            ax.set_xticklabels([str(p) for p in prompts])
            ax.legend()
            ax.grid(axis='y', alpha=0.3)

            plot_idx += 1

    plt.tight_layout()
    plt.savefig(output_dir / 'regime_matrix_bars.png', dpi=150, bbox_inches='tight')
    plt.savefig(output_dir / 'regime_matrix_bars.pdf', bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir / 'regime_matrix_bars.png'}")


def plot_windowed_penalty(results, output_dir: Path):
    """Plot windowed controller penalty across conditions."""
    if not HAS_MPL:
        return

    conditions = []
    penalties = []

    models = sorted(set(k[0] for k in results.keys()))
    prompts = sorted(set(k[1] for k in results.keys()))
    decodes = sorted(set(k[2] for k in results.keys()))

    for model in models:
        for prompt in prompts:
            for decode in decodes:
                auto_key = (model, prompt, decode, 'auto')
                wind_key = (model, prompt, decode, 'windowed')

                if auto_key in results and wind_key in results:
                    auto_eff = results[auto_key]['mean']
                    wind_eff = results[wind_key]['mean']
                    penalty = (wind_eff - auto_eff) / auto_eff * 100

                    conditions.append(f"{model}B\np{prompt}\nd{decode}")
                    penalties.append(penalty)

    fig, ax = plt.subplots(figsize=(12, 6))

    colors = ['#e74c3c' if p < 0 else '#2ecc71' for p in penalties]
    bars = ax.bar(range(len(conditions)), penalties, color=colors)

    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.set_xticks(range(len(conditions)))
    ax.set_xticklabels(conditions, fontsize=9)
    ax.set_ylabel('Efficiency Change vs Auto (%)', fontsize=11)
    ax.set_title('Windowed Controller Penalty: Always Loses 6-17% vs Auto', fontsize=12)
    ax.grid(axis='y', alpha=0.3)

    # Add value labels
    for i, (bar, val) in enumerate(zip(bars, penalties)):
        ax.text(bar.get_x() + bar.get_width()/2, val + (1 if val > 0 else -2),
                f'{val:.1f}%', ha='center', va='bottom' if val > 0 else 'top', fontsize=9)

    plt.tight_layout()
    plt.savefig(output_dir / 'windowed_penalty.png', dpi=150, bbox_inches='tight')
    plt.savefig(output_dir / 'windowed_penalty.pdf', bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir / 'windowed_penalty.png'}")


def main():
    results_dir = Path('/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy/results/regime_matrix_v2')
    output_dir = Path('/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy/reports/regime_matrix_v2')
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading results...")
    results = load_results(results_dir)
    print(f"Loaded {len(results)} conditions")

    if not HAS_MPL:
        print("Skipping plots - matplotlib not available")
        return

    print("Generating plots...")
    plot_regime_heatmap(results, output_dir)
    plot_policy_comparison_bars(results, output_dir)
    plot_windowed_penalty(results, output_dir)

    print(f"\nAll plots saved to {output_dir}")


if __name__ == '__main__':
    main()
