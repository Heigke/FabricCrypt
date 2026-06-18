#!/usr/bin/env python3
"""Analyze existing regime matrix results."""

import json
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict
import numpy as np
import scipy.stats as stats


def compute_ci_bootstrap(values: List[float], n_bootstrap: int = 1000, ci: float = 0.95):
    if len(values) < 2:
        return (values[0] if values else 0.0, values[0] if values else 0.0)
    arr = np.array(values)
    boot_means = [np.mean(np.random.choice(arr, size=len(arr), replace=True)) for _ in range(n_bootstrap)]
    alpha = 1 - ci
    return (np.percentile(boot_means, alpha / 2 * 100), np.percentile(boot_means, (1 - alpha / 2) * 100))


def cohens_d(group1: List[float], group2: List[float]) -> float:
    if len(group1) < 2 or len(group2) < 2:
        return 0.0
    n1, n2 = len(group1), len(group2)
    mean1, mean2 = np.mean(group1), np.mean(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    if pooled_std == 0:
        return 0.0
    return (mean1 - mean2) / pooled_std


def pairwise_ttest(group1: List[float], group2: List[float]):
    if len(group1) < 2 or len(group2) < 2:
        return (0.0, 1.0)
    t_stat, p_val = stats.ttest_ind(group1, group2)
    return (t_stat, p_val)


def parse_condition_name(name: str):
    """Parse condition name like '0.5B_p256_d64_c1_greedy_auto'"""
    parts = name.split('_')
    model_size = float(parts[0].replace('B', ''))
    prompt_tokens = int(parts[1].replace('p', ''))
    decode_tokens = int(parts[2].replace('d', ''))
    concurrency = int(parts[3].replace('c', ''))
    temp = parts[4]
    policy = parts[5]
    return {
        'model_size_b': model_size,
        'prompt_tokens': prompt_tokens,
        'decode_tokens': decode_tokens,
        'concurrency': concurrency,
        'temperature': 0.0 if temp == 'greedy' else float(temp.replace('t', '')),
        'policy': policy
    }


def load_results(results_dir: Path):
    """Load all results from subdirectories."""
    results = {}

    for subdir in results_dir.iterdir():
        if not subdir.is_dir():
            continue

        condition = parse_condition_name(subdir.name)
        tok_per_j_values = []
        tpot_p95_values = []
        energy_values = []

        # Read each run's comparison_extended.json
        for run_dir in subdir.glob('run_*'):
            json_file = run_dir / 'comparison_extended.json'
            if json_file.exists():
                with open(json_file) as f:
                    data = json.load(f)
                    # Get the policy's data
                    policy_data = data.get(condition['policy'], {})
                    if policy_data and 'total_tok_per_j_mean' in policy_data:
                        tok_per_j_values.append(policy_data['total_tok_per_j_mean'])
                        tpot_p95_values.append(policy_data.get('tpot_s_p95_mean', 0) * 1000)  # Convert to ms
                        energy_values.append(policy_data.get('total_energy_j_mean', 0))

        if tok_per_j_values:
            results[subdir.name] = {
                'condition': condition,
                'n': len(tok_per_j_values),
                'tok_per_j_mean': np.mean(tok_per_j_values),
                'tok_per_j_std': np.std(tok_per_j_values),
                'tok_per_j_ci': compute_ci_bootstrap(tok_per_j_values),
                'tpot_p95_mean': np.mean(tpot_p95_values),
                'energy_mean': np.mean(energy_values),
                'raw_tok_per_j': tok_per_j_values
            }

    return results


def compare_policies(results: Dict, baseline_policy: str = 'auto'):
    """Compare policies with statistical tests."""
    comparisons = []

    # Group by condition
    condition_groups = {}
    for name, data in results.items():
        cond = data['condition']
        key = (cond['model_size_b'], cond['prompt_tokens'], cond['decode_tokens'], cond['concurrency'])
        if key not in condition_groups:
            condition_groups[key] = {}
        condition_groups[key][cond['policy']] = data

    for key, policies in condition_groups.items():
        if baseline_policy not in policies:
            continue

        baseline = policies[baseline_policy]

        for policy_name, data in policies.items():
            if policy_name == baseline_policy:
                continue

            eff_diff = data['tok_per_j_mean'] - baseline['tok_per_j_mean']
            eff_pct = (eff_diff / baseline['tok_per_j_mean'] * 100) if baseline['tok_per_j_mean'] > 0 else 0

            t_stat, p_val = pairwise_ttest(data['raw_tok_per_j'], baseline['raw_tok_per_j'])
            d = cohens_d(data['raw_tok_per_j'], baseline['raw_tok_per_j'])

            significant = bool(p_val < 0.05)
            winner = policy_name if eff_diff > 0 and significant else (
                baseline_policy if eff_diff < 0 and significant else "tie"
            )

            comparisons.append({
                'model_size_b': key[0],
                'prompt_tokens': key[1],
                'decode_tokens': key[2],
                'concurrency': key[3],
                'baseline': baseline_policy,
                'policy': policy_name,
                'baseline_tok_per_j': baseline['tok_per_j_mean'],
                'baseline_ci': baseline['tok_per_j_ci'],
                'policy_tok_per_j': data['tok_per_j_mean'],
                'policy_ci': data['tok_per_j_ci'],
                'efficiency_pct': eff_pct,
                'p_value': float(p_val),
                'cohens_d': float(d),
                'significant': significant,
                'winner': winner
            })

    return comparisons


def main():
    results_dir = Path('results/regime_matrix_v2')

    print("Loading results...")
    results = load_results(results_dir)
    print(f"Loaded {len(results)} conditions")

    print("\nComparing policies...")
    comparisons = compare_policies(results)

    # Save comparisons
    with open(results_dir / 'regime_map_comparisons.json', 'w') as f:
        json.dump(comparisons, f, indent=2)

    print("\n" + "=" * 80)
    print("REGIME MAP SUMMARY - AMD gfx1151 Energy-Efficient LLM Inference")
    print("=" * 80)
    print(f"{'Model':>5} {'Prompt':>6} {'Decode':>6} {'Policy':>12} vs auto | {'Δ%':>7} | {'p-val':>7} | {'d':>6} | Winner")
    print("-" * 80)

    for comp in sorted(comparisons, key=lambda x: (x['model_size_b'], x['prompt_tokens'], x['decode_tokens'], x['policy'])):
        sig = "*" if comp['significant'] else " "
        print(f"{comp['model_size_b']:>5}B {comp['prompt_tokens']:>6} {comp['decode_tokens']:>6} "
              f"{comp['policy']:>12} | {comp['efficiency_pct']:>+6.1f}%{sig} | {comp['p_value']:>7.4f} | "
              f"{comp['cohens_d']:>6.2f} | {comp['winner']}")

    print("=" * 80)
    print("* = statistically significant at p<0.05")

    # Summary statistics
    controller_wins = sum(1 for c in comparisons if c['winner'] == 'controller' and c['policy'] == 'controller')
    windowed_wins = sum(1 for c in comparisons if c['winner'] == 'windowed' and c['policy'] == 'windowed')
    auto_wins = sum(1 for c in comparisons if c['winner'] == 'auto')
    ties = sum(1 for c in comparisons if c['winner'] == 'tie')

    print(f"\nSummary:")
    print(f"  Auto wins: {auto_wins}")
    print(f"  Controller wins: {controller_wins}")
    print(f"  Windowed wins: {windowed_wins}")
    print(f"  Ties: {ties}")

    # Key findings per model
    print("\n" + "=" * 80)
    print("KEY FINDINGS BY MODEL SIZE")
    print("=" * 80)

    for model_size in sorted(set(c['model_size_b'] for c in comparisons)):
        model_comps = [c for c in comparisons if c['model_size_b'] == model_size]
        avg_controller_delta = np.mean([c['efficiency_pct'] for c in model_comps if c['policy'] == 'controller'])
        avg_windowed_delta = np.mean([c['efficiency_pct'] for c in model_comps if c['policy'] == 'windowed'])
        print(f"\n{model_size}B Model:")
        print(f"  Controller avg vs auto: {avg_controller_delta:+.1f}%")
        print(f"  Windowed avg vs auto: {avg_windowed_delta:+.1f}%")

    print(f"\nResults saved to {results_dir / 'regime_map_comparisons.json'}")


if __name__ == '__main__':
    main()
