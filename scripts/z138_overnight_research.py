#!/usr/bin/env python3
"""
z138_overnight_research.py

OVERNIGHT RESEARCH: ENHANCE & FALSIFY EMBODIMENT
=================================================

Runs comprehensive experiments to strengthen and falsify embodiment claims:
1. Multi-seed validation (reproducibility)
2. Ablation studies (what breaks it)
3. Stress tests (scaling)
4. Baseline comparisons

Run with:
  HSA_OVERRIDE_GFX_VERSION=11.0.0 python scripts/z138_overnight_research.py
"""

import argparse
import json
import os
import sys
import time
import random
import numpy as np
import torch
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.z135_deep_embodiment_pipeline import (
    DeepEmbodimentConfig,
    RobustTelemetryReader,
    InterventionDatasetGenerator,
    WorldModelTrainer,
    RigorousValidator,
    BodyWorldModel,
    Episode,
    load_model
)
from transformers import AutoTokenizer


def run_experiment(name: str, config: DeepEmbodimentConfig, modifications: dict = None) -> dict:
    """Run a single experiment with optional modifications."""

    print(f"\n{'='*60}")
    print(f"EXPERIMENT: {name}")
    print(f"{'='*60}")

    # Apply modifications
    if modifications:
        for key, value in modifications.items():
            if hasattr(config, key):
                setattr(config, key, value)
                print(f"  Modified {key} = {value}")

    os.makedirs(config.output_dir, exist_ok=True)

    # Set seeds
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-125M")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = load_model(config)
    telem_reader = RobustTelemetryReader(gpu_id=1)

    # Generate episodes
    print(f"\nGenerating {config.n_episodes} episodes...")
    generator = InterventionDatasetGenerator(model, tokenizer, config, telem_reader)

    episodes = []
    for i in range(config.n_episodes):
        episode = generator.generate_episode(i)
        episodes.append(episode)
        if (i + 1) % 50 == 0:
            print(f"  Episode {i+1}/{config.n_episodes}")

    # Train world model
    print(f"\nTraining world model for {config.world_model_epochs} epochs...")
    trainer = WorldModelTrainer(config)

    n_train = int(len(episodes) * 0.8)
    train_episodes = episodes[:n_train]
    test_episodes = episodes[n_train:]

    history = trainer.train(train_episodes)

    # Validate
    print("\nRunning validation...")
    validator = RigorousValidator(trainer.world_model, config)
    results = validator.validate(test_episodes)

    # Summary
    verdict = results['verdict']
    print(f"\n{'-'*40}")
    print(f"RESULT: {verdict['tests_passed']}/{verdict['total_tests']} tests passed")
    for test, status in verdict['details'].items():
        print(f"  {test}: {status}")
    print(f"VERDICT: {verdict['verdict']}")

    return {
        'name': name,
        'config': {k: v for k, v in config.__dict__.items() if not k.startswith('_')},
        'modifications': modifications,
        'results': results,
        'training_final_loss': history['loss'][-1] if history['loss'] else None
    }


def run_ablation_no_timing(base_config: DeepEmbodimentConfig) -> dict:
    """Ablation: Zero out timing signal - should FAIL."""

    # Monkey-patch the telemetry reader to zero timing
    original_read = RobustTelemetryReader.read

    def read_no_timing(self):
        telem = original_read(self)
        telem[8] = 0.5  # Zero out efficiency/timing channel
        return telem

    RobustTelemetryReader.read = read_no_timing

    config = DeepEmbodimentConfig(
        n_episodes=200,
        world_model_epochs=50,
        device=base_config.device,
        seed=42,
        output_dir="results/z138_ablation_no_timing"
    )

    result = run_experiment("ABLATION: No Timing Signal", config)

    # Restore original
    RobustTelemetryReader.read = original_read

    return result


def run_ablation_random_actions(base_config: DeepEmbodimentConfig) -> dict:
    """Ablation: Random actions unrelated to telemetry - should FAIL."""

    # Monkey-patch to shuffle actions
    original_generate = InterventionDatasetGenerator.generate_episode

    def generate_shuffled(self, episode_id):
        episode = original_generate(self, episode_id)
        # Shuffle the action schedule (break causal link)
        random.shuffle(episode.action_schedule)
        return episode

    InterventionDatasetGenerator.generate_episode = generate_shuffled

    config = DeepEmbodimentConfig(
        n_episodes=200,
        world_model_epochs=50,
        device=base_config.device,
        seed=42,
        output_dir="results/z138_ablation_random_actions"
    )

    result = run_experiment("ABLATION: Random Actions", config)

    # Restore original
    InterventionDatasetGenerator.generate_episode = original_generate

    return result


def run_ablation_constant_depth(base_config: DeepEmbodimentConfig) -> dict:
    """Ablation: Constant depth (no variation) - should FAIL counterfactual."""

    config = DeepEmbodimentConfig(
        n_episodes=200,
        world_model_epochs=50,
        device=base_config.device,
        seed=42,
        depth_levels=[4],  # Only depth 4
        output_dir="results/z138_ablation_constant_depth"
    )

    return run_experiment("ABLATION: Constant Depth (4 only)", config)


def run_multi_seed(base_config: DeepEmbodimentConfig, seeds: list) -> list:
    """Run with multiple seeds for reproducibility."""

    results = []
    for seed in seeds:
        config = DeepEmbodimentConfig(
            n_episodes=300,
            world_model_epochs=50,
            device=base_config.device,
            seed=seed,
            output_dir=f"results/z138_seed_{seed}"
        )

        result = run_experiment(f"SEED {seed}", config)
        results.append(result)

    return results


def run_stress_test_episodes(base_config: DeepEmbodimentConfig) -> dict:
    """Stress test with more episodes."""

    config = DeepEmbodimentConfig(
        n_episodes=1000,
        world_model_epochs=100,
        device=base_config.device,
        seed=42,
        output_dir="results/z138_stress_1000ep"
    )

    return run_experiment("STRESS: 1000 Episodes", config)


def main():
    parser = argparse.ArgumentParser(description="Overnight Research")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--quick", action="store_true", help="Quick mode (fewer experiments)")
    args = parser.parse_args()

    print("="*70)
    print("OVERNIGHT RESEARCH: ENHANCE & FALSIFY EMBODIMENT")
    print("="*70)
    print(f"Started: {datetime.now().isoformat()}")
    print(f"Device: {args.device}")

    base_config = DeepEmbodimentConfig(device=args.device)
    all_results = []

    # 1. Multi-seed validation
    print("\n" + "="*70)
    print("PHASE 1: MULTI-SEED VALIDATION")
    print("="*70)

    seeds = [42, 123, 456] if args.quick else [42, 123, 456, 789, 1000]
    seed_results = run_multi_seed(base_config, seeds)
    all_results.extend(seed_results)

    # Analyze seed results
    passes = [r['results']['verdict']['tests_passed'] for r in seed_results]
    print(f"\nSeed Summary: {passes}")
    print(f"Mean tests passed: {np.mean(passes):.2f} ± {np.std(passes):.2f}")

    # 2. Ablation: No timing
    print("\n" + "="*70)
    print("PHASE 2: ABLATION - NO TIMING SIGNAL")
    print("="*70)

    ablation_timing = run_ablation_no_timing(base_config)
    all_results.append(ablation_timing)

    expected_fail = ablation_timing['results']['verdict']['tests_passed'] < 3
    print(f"Expected to FAIL (proves timing is key): {'YES' if expected_fail else 'NO - UNEXPECTED!'}")

    # 3. Ablation: Random actions
    print("\n" + "="*70)
    print("PHASE 3: ABLATION - RANDOM ACTIONS")
    print("="*70)

    ablation_random = run_ablation_random_actions(base_config)
    all_results.append(ablation_random)

    expected_fail = ablation_random['results']['verdict']['tests_passed'] < 3
    print(f"Expected to FAIL (proves action coupling): {'YES' if expected_fail else 'NO - UNEXPECTED!'}")

    # 4. Ablation: Constant depth
    print("\n" + "="*70)
    print("PHASE 4: ABLATION - CONSTANT DEPTH")
    print("="*70)

    ablation_constant = run_ablation_constant_depth(base_config)
    all_results.append(ablation_constant)

    expected_fail = ablation_constant['results']['verdict']['tests_passed'] < 3
    print(f"Expected to FAIL (proves depth variation needed): {'YES' if expected_fail else 'NO - UNEXPECTED!'}")

    # 5. Stress test (skip in quick mode)
    if not args.quick:
        print("\n" + "="*70)
        print("PHASE 5: STRESS TEST - 1000 EPISODES")
        print("="*70)

        stress_result = run_stress_test_episodes(base_config)
        all_results.append(stress_result)

    # Final summary
    print("\n" + "="*70)
    print("FINAL SUMMARY")
    print("="*70)

    summary = {
        'timestamp': datetime.now().isoformat(),
        'experiments': []
    }

    for r in all_results:
        verdict = r['results']['verdict']
        exp_summary = {
            'name': r['name'],
            'tests_passed': verdict['tests_passed'],
            'verdict': verdict['verdict'],
            'details': verdict['details']
        }
        summary['experiments'].append(exp_summary)

        status = "✓" if verdict['tests_passed'] >= 3 else "✗"
        print(f"{status} {r['name']}: {verdict['tests_passed']}/4 - {verdict['verdict']}")

    # Analysis
    print("\n" + "-"*40)
    print("FALSIFICATION ANALYSIS")
    print("-"*40)

    # Check if ablations fail as expected
    seed_passes = [r['results']['verdict']['tests_passed'] for r in seed_results]
    ablation_results = [ablation_timing, ablation_random, ablation_constant]
    ablation_passes = [r['results']['verdict']['tests_passed'] for r in ablation_results]

    if min(seed_passes) >= 3:
        print("✓ Multi-seed: REPRODUCIBLE (all seeds pass)")
    else:
        print("✗ Multi-seed: NOT REPRODUCIBLE")

    if max(ablation_passes) < 3:
        print("✓ Ablations: FAIL AS EXPECTED (proves each component needed)")
    else:
        print("⚠ Ablations: Some unexpectedly pass (investigate!)")

    # Save results
    results_path = "results/z138_overnight_summary.json"
    os.makedirs("results", exist_ok=True)

    def convert_for_json(obj):
        if isinstance(obj, (np.floating, np.integer)):
            return float(obj) if isinstance(obj, np.floating) else int(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        elif isinstance(obj, dict):
            return {k: convert_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_for_json(v) for v in obj]
        return obj

    with open(results_path, 'w') as f:
        json.dump(convert_for_json(summary), f, indent=2)

    print(f"\nResults saved to: {results_path}")
    print(f"Completed: {datetime.now().isoformat()}")


if __name__ == "__main__":
    main()
