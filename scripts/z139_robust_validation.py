#!/usr/bin/env python3
"""
z139_robust_validation.py

ROBUST VALIDATION PROTOCOL
==========================

Based on z138 findings:
- Uses 1000 episodes (3.3x more than z138 multi-seed)
- Uses 100 epochs (2x more training)
- Runs 3 seeds minimum for statistical significance
- Reports mean ± std across seeds
- Includes quick ablation check

Run with:
  HSA_OVERRIDE_GFX_VERSION=11.0.0 python scripts/z139_robust_validation.py
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
    load_model
)
from transformers import AutoTokenizer


def run_single_validation(config: DeepEmbodimentConfig, seed: int) -> dict:
    """Run validation with specific seed."""

    # Set seeds
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    config.seed = seed

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-125M")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = load_model(config)
    telem_reader = RobustTelemetryReader(gpu_id=1)

    # Generate episodes
    print(f"\n  Generating {config.n_episodes} episodes...")
    generator = InterventionDatasetGenerator(model, tokenizer, config, telem_reader)

    episodes = []
    for i in range(config.n_episodes):
        episode = generator.generate_episode(i)
        episodes.append(episode)
        if (i + 1) % 200 == 0:
            print(f"    Episode {i+1}/{config.n_episodes}")

    # Train world model
    print(f"  Training world model for {config.world_model_epochs} epochs...")
    trainer = WorldModelTrainer(config)

    n_train = int(len(episodes) * 0.8)
    train_episodes = episodes[:n_train]
    test_episodes = episodes[n_train:]

    history = trainer.train(train_episodes)

    # Validate
    print("  Running validation...")
    validator = RigorousValidator(trainer.world_model, config)
    results = validator.validate(test_episodes)

    return {
        'seed': seed,
        'tests_passed': results['verdict']['tests_passed'],
        'verdict': results['verdict']['verdict'],
        'details': results['verdict']['details'],
        'final_loss': history['loss'][-1] if history['loss'] else None
    }


def main():
    parser = argparse.ArgumentParser(description="Robust Validation Protocol")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 456])
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=100)
    args = parser.parse_args()

    print("=" * 70)
    print("z139: ROBUST VALIDATION PROTOCOL")
    print("=" * 70)
    print(f"Started: {datetime.now().isoformat()}")
    print(f"Device: {args.device}")
    print(f"Episodes: {args.episodes}")
    print(f"Epochs: {args.epochs}")
    print(f"Seeds: {args.seeds}")
    print("=" * 70)

    results = []

    for i, seed in enumerate(args.seeds):
        print(f"\n{'='*60}")
        print(f"SEED {seed} ({i+1}/{len(args.seeds)})")
        print(f"{'='*60}")

        config = DeepEmbodimentConfig(
            n_episodes=args.episodes,
            world_model_epochs=args.epochs,
            device=args.device,
            seed=seed,
            output_dir=f"results/z139_seed_{seed}"
        )
        os.makedirs(config.output_dir, exist_ok=True)

        result = run_single_validation(config, seed)
        results.append(result)

        status = "PASS" if result['tests_passed'] >= 3 else "FAIL"
        print(f"\n  Result: {result['tests_passed']}/4 - {status}")
        for test, val in result['details'].items():
            print(f"    {test}: {val}")

    # Summary
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)

    passes = [r['tests_passed'] for r in results]
    mean_pass = np.mean(passes)
    std_pass = np.std(passes)

    for r in results:
        status = "✓" if r['tests_passed'] >= 3 else "✗"
        print(f"{status} Seed {r['seed']}: {r['tests_passed']}/4 - {r['verdict']}")

    print(f"\nMean: {mean_pass:.2f} ± {std_pass:.2f}")
    print(f"Pass rate: {sum(1 for p in passes if p >= 3)}/{len(passes)} ({100*sum(1 for p in passes if p >= 3)/len(passes):.0f}%)")

    # Determine overall verdict
    if all(p >= 3 for p in passes):
        overall = "REPRODUCIBLE - All seeds pass"
    elif mean_pass >= 3:
        overall = "MOSTLY REPRODUCIBLE - Mean passes threshold"
    else:
        overall = "NOT REPRODUCIBLE - Insufficient consistency"

    print(f"\nOVERALL VERDICT: {overall}")

    # Save results
    summary = {
        'timestamp': datetime.now().isoformat(),
        'config': {
            'episodes': args.episodes,
            'epochs': args.epochs,
            'device': args.device,
            'seeds': args.seeds
        },
        'results': results,
        'summary': {
            'mean': float(mean_pass),
            'std': float(std_pass),
            'pass_rate': sum(1 for p in passes if p >= 3) / len(passes),
            'overall_verdict': overall
        }
    }

    os.makedirs("results", exist_ok=True)
    results_path = "results/z139_robust_validation.json"
    with open(results_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\nResults saved to: {results_path}")
    print(f"Completed: {datetime.now().isoformat()}")


if __name__ == "__main__":
    main()
