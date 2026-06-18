#!/usr/bin/env python3
"""
FEEL Training v10 - Fixed Gradient Flow + Stronger Teacher

Key fixes from v9:
1. Gradients flow to FEEL (removed torch.no_grad)
2. Teacher is actually stronger (self-consistency voting)
3. Action-conditioned interoception

Usage:
    python scripts/train_feel_v10.py --epochs 10 --alpha 0.15
"""

import sys
import time
import json
import argparse
from pathlib import Path
from typing import Dict, List
import numpy as np

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.feel_projector import FEELProjectorFull
from src.feel_trainer_v10 import FEELTrainerV10, TrainerConfigV10, TRAINER_VERSION
from src.telemetry_sampler import TelemetrySampler

# Training prompts
TRAINING_PROMPTS = [
    # Math (high entropy - teacher should help)
    "The derivative of x^3 + 2x is",
    "If a train travels at 60 mph for 2 hours, the distance is",
    "What is 15% of 80?",
    "Solve for x: 2x + 5 = 15, so x =",
    "The sum of angles in a triangle is",

    # Factual (low entropy - less need for teacher)
    "The capital of France is",
    "Water boils at",
    "The largest planet in our solar system is",
    "The chemical formula for water is",
    "The speed of light is approximately",

    # Open-ended (medium entropy)
    "The best way to learn programming is",
    "A healthy breakfast might include",
    "To improve sleep quality, one should",
    "The future of AI technology will",
    "Climate change can be addressed by",

    # Creative (high entropy)
    "Once upon a time, in a distant",
    "The robot looked at the sunset and felt",
    "If I could have any superpower, I would choose",
    "The strangest thing happened when",
    "In the year 2100, humans will",

    # Technical (medium entropy)
    "To implement a binary search tree in Python,",
    "The time complexity of quicksort is",
    "REST APIs should follow the principle of",
    "Machine learning models are trained using",
    "The difference between TCP and UDP is",

    # Completions
    "The quick brown fox",
    "Hello, my name is",
    "I think, therefore",
    "To be or not to",
    "All roads lead to",
]


def main():
    parser = argparse.ArgumentParser(description="FEEL v10 Training")
    parser.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--output_dir", default="results/feel_training_v10")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--alpha", type=float, default=0.15)
    parser.add_argument("--teacher_samples", type=int, default=8)
    parser.add_argument("--teacher_mode", default="consistency")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    print("=" * 70)
    print(f"  FEEL TRAINING {TRAINER_VERSION} - GRADIENT FLOW FIXED")
    print("=" * 70)
    print(f"  Alpha: {args.alpha}")
    print(f"  Teacher mode: {args.teacher_mode}")
    print(f"  Teacher samples: {args.teacher_samples}")
    print("=" * 70)

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # Load model
    print(f"\nLoading model {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map=args.device,
        trust_remote_code=True,
    )
    model.eval()

    embed_dim = model.config.hidden_size
    print(f"  Embed dim: {embed_dim}")

    # Create projector
    projector = FEELProjectorFull(embed_dim=embed_dim).to(args.device)

    # Config
    config = TrainerConfigV10(
        alpha=args.alpha,
        teacher_samples=args.teacher_samples,
        teacher_mode=args.teacher_mode,
        train_interoception=True,
        learning_rate=1e-4,
    )

    # Trainer
    trainer = FEELTrainerV10(model, projector, config)
    if trainer.interoception_head is not None:
        trainer.interoception_head = trainer.interoception_head.to(args.device)

    # Telemetry
    try:
        sampler = TelemetrySampler(sample_hz=30)
        sampler.start()
        trainer.set_telemetry_sampler(sampler)
        print(f"  Telemetry: {sampler.source}")
    except Exception as e:
        print(f"  Telemetry unavailable: {e}")
        sampler = None

    # Training
    print(f"\n  Training on {len(TRAINING_PROMPTS)} prompts for {args.epochs} epochs")
    print("-" * 70)

    all_metrics = []

    for epoch in range(args.epochs):
        print(f"\n[Epoch {epoch + 1}/{args.epochs}]")
        epoch_metrics = []
        prev_feel = None

        np.random.seed(42 + epoch)
        prompts = np.random.permutation(TRAINING_PROMPTS).tolist()

        for i, prompt in enumerate(prompts):
            input_ids = tokenizer.encode(prompt, return_tensors='pt').to(args.device)

            loss, metrics, feel_embed = trainer.train_step(input_ids, prev_feel)
            prev_feel = feel_embed
            epoch_metrics.append(metrics)

            if (i + 1) % 10 == 0:
                print(f"    [{i+1}/{len(prompts)}] loss={metrics['total_loss']:.4f}, "
                      f"grad={metrics['grad_norm']:.4f}, kl={metrics['kl_baseline_feel']:.4f}")

        # Epoch summary
        summary = {}
        for key in epoch_metrics[0].keys():
            vals = [m[key] for m in epoch_metrics]
            summary[f"mean_{key}"] = np.mean(vals)
            summary[f"std_{key}"] = np.std(vals)

        all_metrics.append(summary)

        print(f"  Epoch {epoch + 1} summary:")
        print(f"    loss: {summary['mean_total_loss']:.4f} ± {summary['std_total_loss']:.4f}")
        print(f"    grad_norm: {summary['mean_grad_norm']:.4f}")
        print(f"    kl_baseline_feel: {summary['mean_kl_baseline_feel']:.4f}")
        print(f"    distillation: {summary['mean_distillation_loss']:.4f}")

        if 'mean_r2_horizon_1' in summary:
            print(f"    R² (1-step): {summary['mean_r2_horizon_1']:.4f}")

        # Save checkpoint
        ckpt = {
            'version': TRAINER_VERSION,
            'epoch': epoch + 1,
            'projector_state_dict': projector.state_dict(),
            'alpha': args.alpha,
            'config': config.__dict__,
            'metrics': summary,
        }
        if trainer.interoception_head is not None:
            ckpt['interoception_state_dict'] = trainer.interoception_head.state_dict()

        torch.save(ckpt, f"{args.output_dir}/checkpoint_epoch_{epoch+1}.pt")

    # Final
    torch.save(ckpt, f"{args.output_dir}/final_checkpoint.pt")

    # History
    with open(f"{args.output_dir}/training_history.json", 'w') as f:
        json.dump({
            'version': TRAINER_VERSION,
            'config': {k: str(v) if not isinstance(v, (int, float, str, bool, list)) else v
                       for k, v in config.__dict__.items()},
            'epochs': [{k: float(v) if isinstance(v, (np.floating,)) else v
                        for k, v in m.items()} for m in all_metrics],
        }, f, indent=2)

    if sampler:
        sampler.stop()

    print("\n" + "=" * 70)
    print("  Training complete!")
    print(f"  Checkpoints: {args.output_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
