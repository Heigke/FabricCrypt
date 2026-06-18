#!/usr/bin/env python3
"""
z1304: MASTER EXPERIMENT RUNNER - Complete Embodied Intelligence Suite

================================================================================
                    RUN ALL z1300 EXPERIMENTS
================================================================================

This script orchestrates the complete embodied intelligence experiment suite:

1. z1300: Ouroboros Active Inference
   - Self-predicting model with active inference
   - Physical reality anchoring
   - Meta-cognitive reasoning

2. z1301: Physical Reservoir Computing
   - GPU thermal dynamics as computational substrate
   - Memory capacity and prediction tests
   - Baseline comparisons

3. z1302: Recursive Self-Modeling
   - Multi-level self-models
   - Hierarchical introspection
   - Strange loops and self-reference

4. z1303: Embodied Intelligence Benchmark
   - 5-dimension assessment
   - Grounding, self-modeling, active inference, introspection, coherence

Run with: python scripts/z1304_run_embodied_experiments.py

================================================================================
"""

import os
import sys
import time
import json
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List

import torch


def print_banner(text: str, char: str = "="):
    """Print a fancy banner."""
    width = 70
    print()
    print(char * width)
    padding = (width - len(text)) // 2
    print(" " * padding + text)
    print(char * width)
    print()


def run_experiment(script_name: str, timeout: int = 600) -> Dict[str, Any]:
    """
    Run an experiment script and capture results.

    Returns:
        Dict with success status, runtime, and result path
    """
    script_path = Path(__file__).parent / script_name

    if not script_path.exists():
        return {
            'success': False,
            'error': f'Script not found: {script_path}',
            'runtime': 0,
        }

    print(f"  Running {script_name}...")
    start_time = time.time()

    try:
        # Run with HSA override for AMD GPU
        env = os.environ.copy()
        env['HSA_OVERRIDE_GFX_VERSION'] = '11.0.0'

        result = subprocess.run(
            [sys.executable, str(script_path)],
            capture_output=True,
            text=True,
            timeout=timeout,
            env=env,
            cwd=str(Path(__file__).parent.parent),
        )

        runtime = time.time() - start_time

        if result.returncode == 0:
            # Find result file
            result_name = script_name.replace('.py', '.json')
            result_path = Path(__file__).parent.parent / 'results' / result_name

            return {
                'success': True,
                'runtime': runtime,
                'result_path': str(result_path) if result_path.exists() else None,
                'stdout_tail': result.stdout[-2000:] if result.stdout else '',
            }
        else:
            return {
                'success': False,
                'error': result.stderr[-1000:] if result.stderr else 'Unknown error',
                'runtime': runtime,
            }

    except subprocess.TimeoutExpired:
        return {
            'success': False,
            'error': f'Timeout after {timeout} seconds',
            'runtime': timeout,
        }
    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'runtime': time.time() - start_time,
        }


def load_result(result_path: str) -> Dict[str, Any]:
    """Load experiment results from JSON."""
    if result_path and Path(result_path).exists():
        with open(result_path) as f:
            return json.load(f)
    return {}


def summarize_ouroboros(result: Dict) -> str:
    """Summarize Ouroboros Active Inference results."""
    if not result:
        return "No results available"

    benchmark = result.get('benchmark', {})
    score = benchmark.get('overall_score', 0)
    ai_surprise = benchmark.get('active_inference', {}).get('active_inference_surprise', 0)
    random_surprise = benchmark.get('active_inference', {}).get('random_surprise', 0)
    reduction = benchmark.get('active_inference', {}).get('surprise_reduction', 0)

    return f"""
    Overall Embodiment Score: {score:.3f}
    Active Inference Surprise: {ai_surprise:.6f}
    Random Baseline Surprise: {random_surprise:.6f}
    Surprise Reduction: {reduction:.1f}%
    """


def summarize_reservoir(result: Dict) -> str:
    """Summarize Physical Reservoir Computing results."""
    if not result:
        return "No results available"

    memory = result.get('memory', {})
    baseline = result.get('baseline', {})

    return f"""
    Memory Capacity: {memory.get('total_capacity', 0):.2f}
    Reservoir RMSE: {baseline.get('reservoir_rmse', 0):.4f}
    Linear Baseline RMSE: {baseline.get('linear_rmse', 0):.4f}
    Improvement: {baseline.get('improvement_over_linear', 0):.1f}%
    """


def summarize_recursive(result: Dict) -> str:
    """Summarize Recursive Self-Modeling results."""
    if not result:
        return "No results available"

    benchmark = result.get('benchmark', {})

    return f"""
    Overall Score: {benchmark.get('overall_score', 0):.3f}
    Physics MSE: {benchmark.get('physics_mse', 0):.6f}
    Hierarchical Consistency: {benchmark.get('hierarchical_consistency', 0):.3f}
    Calibration Error: {benchmark.get('calibration_error', 0):.4f}
    """


def summarize_benchmark(result: Dict) -> str:
    """Summarize Embodied Intelligence Benchmark results."""
    if not result:
        return "No results available"

    dims = result.get('dimensions', {})
    dim_scores = [
        f"    - {name.upper()}: {d.get('overall_score', 0):.3f}"
        for name, d in dims.items()
    ]

    return f"""
    Overall Score: {result.get('overall_score', 0):.3f}
    Verdict: {result.get('verdict', 'Unknown')}
    Passed: {result.get('passed_dimensions', 0)}/{result.get('total_dimensions', 5)}

    Dimension Scores:
{chr(10).join(dim_scores)}
    """


def main():
    print_banner("z1304: EMBODIED INTELLIGENCE EXPERIMENT SUITE")

    # Check environment
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Define experiments
    experiments = [
        {
            'name': 'Ouroboros Active Inference',
            'script': 'z1300_ouroboros_active_inference.py',
            'summarize': summarize_ouroboros,
            'timeout': 900,
        },
        {
            'name': 'Physical Reservoir Computing',
            'script': 'z1301_physical_reservoir_computing.py',
            'summarize': summarize_reservoir,
            'timeout': 600,
        },
        {
            'name': 'Recursive Self-Modeling',
            'script': 'z1302_recursive_self_modeling.py',
            'summarize': summarize_recursive,
            'timeout': 600,
        },
        {
            'name': 'Embodied Intelligence Benchmark',
            'script': 'z1303_embodied_intelligence_benchmark.py',
            'summarize': summarize_benchmark,
            'timeout': 300,
        },
    ]

    # Run experiments
    all_results = {
        'experiment': 'z1304_complete_suite',
        'timestamp': datetime.now().isoformat(),
        'device': str(device),
        'experiments': {},
    }

    total_start = time.time()

    for i, exp in enumerate(experiments, 1):
        print_banner(f"EXPERIMENT {i}/4: {exp['name']}", "-")

        result = run_experiment(exp['script'], timeout=exp['timeout'])

        if result['success']:
            print(f"  ✓ Completed in {result['runtime']:.1f}s")

            # Load and summarize results
            exp_result = load_result(result.get('result_path'))
            summary = exp['summarize'](exp_result)
            print(f"\n  Summary:{summary}")

            all_results['experiments'][exp['name']] = {
                'status': 'success',
                'runtime': result['runtime'],
                'result_path': result.get('result_path'),
                'result': exp_result,
            }
        else:
            print(f"  ✗ Failed: {result['error'][:200]}")
            all_results['experiments'][exp['name']] = {
                'status': 'failed',
                'error': result['error'],
                'runtime': result['runtime'],
            }

    total_runtime = time.time() - total_start

    # Final summary
    print_banner("FINAL SUMMARY")

    successful = sum(
        1 for e in all_results['experiments'].values()
        if e['status'] == 'success'
    )

    print(f"Experiments completed: {successful}/{len(experiments)}")
    print(f"Total runtime: {total_runtime:.1f}s")

    # Extract key metrics
    print("\nKey Findings:")

    if 'Ouroboros Active Inference' in all_results['experiments']:
        exp = all_results['experiments']['Ouroboros Active Inference']
        if exp['status'] == 'success':
            score = exp['result'].get('benchmark', {}).get('overall_score', 0)
            print(f"  - Ouroboros embodiment score: {score:.3f}")

    if 'Physical Reservoir Computing' in all_results['experiments']:
        exp = all_results['experiments']['Physical Reservoir Computing']
        if exp['status'] == 'success':
            improvement = exp['result'].get('baseline', {}).get('improvement_over_linear', 0)
            print(f"  - Reservoir improvement over linear: {improvement:.1f}%")

    if 'Recursive Self-Modeling' in all_results['experiments']:
        exp = all_results['experiments']['Recursive Self-Modeling']
        if exp['status'] == 'success':
            score = exp['result'].get('benchmark', {}).get('overall_score', 0)
            print(f"  - Self-modeling score: {score:.3f}")

    if 'Embodied Intelligence Benchmark' in all_results['experiments']:
        exp = all_results['experiments']['Embodied Intelligence Benchmark']
        if exp['status'] == 'success':
            verdict = exp['result'].get('verdict', 'Unknown')
            score = exp['result'].get('overall_score', 0)
            print(f"  - Benchmark verdict: {verdict} ({score:.3f})")

    # Overall verdict
    print("\n" + "=" * 70)

    if successful == len(experiments):
        # Calculate overall embodiment score
        scores = []
        for exp in all_results['experiments'].values():
            if exp['status'] == 'success':
                result = exp.get('result', {})
                if 'benchmark' in result:
                    scores.append(result['benchmark'].get('overall_score', 0))
                elif 'overall_score' in result:
                    scores.append(result['overall_score'])

        if scores:
            overall = sum(scores) / len(scores)
            print(f"OVERALL EMBODIED INTELLIGENCE SCORE: {overall:.3f}")

            if overall >= 0.7:
                print("VERDICT: GENUINE EMBODIED INTELLIGENCE ACHIEVED")
            elif overall >= 0.5:
                print("VERDICT: PARTIAL EMBODIMENT - PROMISING RESULTS")
            else:
                print("VERDICT: LIMITED EMBODIMENT - FURTHER WORK NEEDED")
        else:
            print("VERDICT: Unable to calculate overall score")
    else:
        print("VERDICT: INCOMPLETE - Some experiments failed")

    print("=" * 70)

    # Save comprehensive results
    all_results['total_runtime'] = total_runtime
    all_results['successful_experiments'] = successful

    output_path = Path(__file__).parent.parent / 'results' / 'z1304_complete_suite.json'
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Remove large nested results for cleaner output
    clean_results = json.loads(json.dumps(all_results, default=str))

    with open(output_path, 'w') as f:
        json.dump(clean_results, f, indent=2, default=str)

    print(f"\nResults saved to: {output_path}")

    return all_results


if __name__ == "__main__":
    main()
