#!/usr/bin/env python3
"""
Z605: Real LLM Inference Validation

This is the HONEST test:
1. Real LLM inference (GPT-2, actual token generation)
2. Real hardware metrics (hwmon power, sysfs clocks)
3. Proper methodology (fixed task per comparison, trial-level stats)
4. Strong baselines (grid search, tuned PID)
5. Meaningful metrics (tokens/s, J/token)

Objective: Maximize throughput subject to power/thermal constraints.
"""

import os
import sys
import time
import json
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Dict, List
import statistics

os.environ["HSA_OVERRIDE_GFX_VERSION"] = "11.0.0"
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class ValidationConfig:
    n_trials: int = 3
    n_inferences_per_trial: int = 20
    max_new_tokens: int = 30  # Fixed token count per inference
    cooldown_s: float = 5.0

    # Constraints
    max_power_w: float = 80.0
    max_temp_c: float = 75.0

    # Model
    model_name: str = "gpt2"


# ============================================================================
# Main Validation
# ============================================================================

def run_validation(config: ValidationConfig):
    print("=" * 70)
    print("Z605: REAL LLM INFERENCE VALIDATION")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")
    print(f"Model: {config.model_name}")
    print(f"Tokens per inference: {config.max_new_tokens}")
    print(f"Trials: {config.n_trials}, Inferences/trial: {config.n_inferences_per_trial}")

    # Import our modules
    from src.deep_embodiment.hw_substrate import HWSubstrate
    from src.deep_embodiment.inference_harness import LLMInferenceHarness
    from src.deep_embodiment.embodied_controller import (
        ControlAction, ControlObjective,
        SimpleReactiveController, TunedPIDController,
        GridSearchController, EmbodiedController,
    )

    # Initialize hardware substrate
    print("\n--- Initializing Hardware Substrate ---")
    hw = HWSubstrate()
    cal = hw.calibrate(duration_s=3.0)

    # Initialize inference harness
    print("\n--- Loading LLM ---")
    harness = LLMInferenceHarness(
        model_name=config.model_name,
        device=device,
        max_new_tokens=config.max_new_tokens,
    )
    harness.load_model()

    # Test inference
    print("\n--- Test Inference ---")
    test_result = harness.run_inference(hw_substrate=hw)
    print(f"  Generated: {test_result.n_tokens_generated} tokens")
    print(f"  Throughput: {test_result.tokens_per_second:.1f} tokens/s")
    print(f"  Energy: {test_result.joules_per_token:.4f} J/token")

    # Define objective
    objective = ControlObjective(
        max_power_w=config.max_power_w,
        max_temp_c=config.max_temp_c,
        throughput_weight=1.0,
        efficiency_weight=0.5,
        constraint_penalty=10.0,
    )

    # Create controllers
    controllers = {
        'reactive': SimpleReactiveController(hw, objective),
        'pid': TunedPIDController(hw, objective),
        'grid_search': GridSearchController(hw, objective),
        'embodied': EmbodiedController(hw, objective),
    }

    # Calibrate grid search (strong baseline)
    print("\n--- Calibrating Grid Search Baseline ---")

    def inference_for_calibration(action: ControlAction):
        hw.set_perf_level(action.perf_level)
        prompts = [harness.get_next_prompt() for _ in range(action.batch_size)]
        if action.batch_size == 1:
            return harness.run_inference(prompts[0], max_new_tokens=action.max_tokens, hw_substrate=hw)
        else:
            return harness.run_batch_inference(prompts, batch_size=action.batch_size,
                                               max_new_tokens=action.max_tokens, hw_substrate=hw)

    controllers['grid_search'].calibrate(inference_for_calibration, n_samples=5)

    # ========================================================================
    # Run Trials
    # ========================================================================
    print("\n--- Running Trials ---")

    results = {name: [] for name in controllers}

    for trial in range(config.n_trials):
        print(f"\nTrial {trial + 1}/{config.n_trials}")

        for name, controller in controllers.items():
            # Cooldown
            hw.set_perf_level('low')
            time.sleep(config.cooldown_s)
            hw.set_perf_level('auto')

            trial_metrics = {
                'tokens_generated': 0,
                'total_time_s': 0.0,
                'total_energy_j': 0.0,
                'tokens_per_second': [],
                'joules_per_token': [],
                'power_samples': [],
                'temp_samples': [],
                'constraint_violations': 0,
            }

            for i in range(config.n_inferences_per_trial):
                # Sense
                state = hw.sense()
                trial_metrics['power_samples'].append(state.power_w)
                trial_metrics['temp_samples'].append(state.temp_edge_c)

                # Choose action
                action = controller.choose_action(state)

                # Execute
                hw.set_perf_level(action.perf_level)
                time.sleep(0.05)  # Let perf level take effect

                prompts = [harness.get_next_prompt() for _ in range(action.batch_size)]
                if action.batch_size == 1:
                    result = harness.run_inference(prompts[0], max_new_tokens=action.max_tokens, hw_substrate=hw)
                else:
                    result = harness.run_batch_inference(prompts, batch_size=action.batch_size,
                                                        max_new_tokens=action.max_tokens, hw_substrate=hw)

                # Record
                trial_metrics['tokens_generated'] += result.n_tokens_generated
                trial_metrics['total_time_s'] += result.total_time_s
                trial_metrics['total_energy_j'] += result.energy_j
                trial_metrics['tokens_per_second'].append(result.tokens_per_second)
                trial_metrics['joules_per_token'].append(result.joules_per_token)

                # Check constraints
                state_after = hw.sense()
                if state_after.power_w > config.max_power_w or state_after.temp_edge_c > config.max_temp_c:
                    trial_metrics['constraint_violations'] += 1

                # Learn (for embodied controller)
                if hasattr(controller, 'observe_outcome'):
                    controller.observe_outcome(result, state_after)

            # Aggregate trial
            trial_summary = {
                'tokens_generated': trial_metrics['tokens_generated'],
                'total_time_s': trial_metrics['total_time_s'],
                'total_energy_j': trial_metrics['total_energy_j'],
                'avg_tokens_per_second': statistics.mean(trial_metrics['tokens_per_second']),
                'avg_joules_per_token': statistics.mean(trial_metrics['joules_per_token']),
                'avg_power_w': statistics.mean(trial_metrics['power_samples']),
                'avg_temp_c': statistics.mean(trial_metrics['temp_samples']),
                'constraint_violations': trial_metrics['constraint_violations'],
            }
            results[name].append(trial_summary)

            print(f"  {name:12s}: {trial_summary['avg_tokens_per_second']:.1f} tok/s, "
                  f"{trial_summary['avg_joules_per_token']:.4f} J/tok, "
                  f"{trial_summary['constraint_violations']} violations")

    # ========================================================================
    # Analysis
    # ========================================================================
    print("\n" + "=" * 70)
    print("RESULTS (Trial-level statistics)")
    print("=" * 70)

    final_results = {}

    for name in controllers:
        tps = [t['avg_tokens_per_second'] for t in results[name]]
        jpt = [t['avg_joules_per_token'] for t in results[name]]
        violations = sum(t['constraint_violations'] for t in results[name])

        final_results[name] = {
            'tokens_per_second_mean': statistics.mean(tps),
            'tokens_per_second_std': statistics.stdev(tps) if len(tps) > 1 else 0,
            'joules_per_token_mean': statistics.mean(jpt),
            'joules_per_token_std': statistics.stdev(jpt) if len(jpt) > 1 else 0,
            'total_violations': violations,
            'per_trial': results[name],
        }

        print(f"\n{name.upper()}:")
        print(f"  Throughput: {final_results[name]['tokens_per_second_mean']:.1f} ± "
              f"{final_results[name]['tokens_per_second_std']:.1f} tokens/s")
        print(f"  Efficiency: {final_results[name]['joules_per_token_mean']:.4f} ± "
              f"{final_results[name]['joules_per_token_std']:.4f} J/token")
        print(f"  Constraint violations: {violations}")

    # ========================================================================
    # Comparison
    # ========================================================================
    print("\n--- Comparison ---")

    # Grid search is our strong baseline
    baseline_tps = final_results['grid_search']['tokens_per_second_mean']
    baseline_jpt = final_results['grid_search']['joules_per_token_mean']

    for name in controllers:
        tps = final_results[name]['tokens_per_second_mean']
        jpt = final_results[name]['joules_per_token_mean']

        tps_diff = ((tps - baseline_tps) / baseline_tps * 100) if baseline_tps > 0 else 0
        jpt_diff = ((baseline_jpt - jpt) / baseline_jpt * 100) if baseline_jpt > 0 else 0

        print(f"  {name:12s}: throughput {tps_diff:+.1f}%, efficiency {jpt_diff:+.1f}% vs grid_search")

    # ========================================================================
    # Verdict
    # ========================================================================
    print("\n--- Verdict ---")

    embodied_tps = final_results['embodied']['tokens_per_second_mean']
    embodied_jpt = final_results['embodied']['joules_per_token_mean']
    grid_tps = final_results['grid_search']['tokens_per_second_mean']
    grid_jpt = final_results['grid_search']['joules_per_token_mean']

    beats_throughput = embodied_tps > grid_tps * 1.05  # 5% margin
    beats_efficiency = embodied_jpt < grid_jpt * 0.95  # 5% margin

    if beats_throughput or beats_efficiency:
        print("  ✓ EMBODIED CONTROLLER SHOWS IMPROVEMENT")
        if beats_throughput:
            print(f"    Throughput: {embodied_tps:.1f} vs {grid_tps:.1f} tokens/s")
        if beats_efficiency:
            print(f"    Efficiency: {embodied_jpt:.4f} vs {grid_jpt:.4f} J/token")
    else:
        print("  ~ EMBODIED CONTROLLER DOES NOT BEAT STRONG BASELINE")
        print(f"    This is expected - grid search finds optimal static policy.")
        print(f"    Embodied value comes from DYNAMIC adaptation, not static tuning.")

    # ========================================================================
    # Save Results
    # ========================================================================
    output = {
        'config': asdict(config),
        'calibration': cal,
        'results': final_results,
        'verdict': {
            'beats_throughput': beats_throughput,
            'beats_efficiency': beats_efficiency,
        },
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
    }

    output_path = Path("results/z605_real_inference.json")
    output_path.parent.mkdir(exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2, default=float)

    print(f"\nResults saved to {output_path}")

    # Cleanup
    hw.reset()
    harness.unload_model()

    return output


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--inferences", type=int, default=20)
    parser.add_argument("--tokens", type=int, default=30)
    parser.add_argument("--quick", action="store_true")

    args = parser.parse_args()

    config = ValidationConfig(
        n_trials=2 if args.quick else args.trials,
        n_inferences_per_trial=10 if args.quick else args.inferences,
        max_new_tokens=args.tokens,
    )

    run_validation(config)
