#!/usr/bin/env python3
"""
FEEL Training v3.0 - Teacher Distillation with One Forward Per Token
=====================================================================

This script trains the FEEL projector for UTILITY via teacher distillation.
Uses FEELStreamV3 (t→t+1 injection) which fixes the double-forward artifact.

Key improvements over v6:
1. One forward per token (t→t+1 injection pattern)
2. Teacher distillation: multi-sample teacher vs single-sample student with FEEL
3. Entropy-weighted distillation: focus on "hard" regions
4. KL budget constraint: FEEL shouldn't diverge too much from baseline
5. Interoception training: z_feel predicts future hardware state

Usage:
    python scripts/train_feel_v3.py --epochs 10 --batch_size 4
    python scripts/train_feel_v3.py --checkpoint results/feel_training/canonical_v6_checkpoint.pt
"""

import sys
import time
import json
import argparse
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent.parent))

from src import (
    FEELProjectorFull,
    FEELTrainer,
    FEELTrainerConfig,
    TelemetrySampler,
    FEEL_STREAM_V3_VERSION,
)


TRAINER_VERSION = "v3.0.0"

# Training prompts - diverse to cover different scenarios
TRAINING_PROMPTS = [
    # Math/reasoning (high entropy expected)
    "The derivative of x^3 + 2x is",
    "If a train travels at 60 mph for 2 hours, the distance is",
    "The sum of angles in a triangle is",
    "What is 15% of 80?",
    "Solve for x: 2x + 5 = 15, so x =",

    # Factual (lower entropy)
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

    # Completions (varies)
    "The quick brown fox",
    "Hello, my name is",
    "I think, therefore",
    "To be or not to",
    "All roads lead to",
]


def create_model_and_tokenizer(model_name: str, device: str):
    """Load model and tokenizer."""
    print(f"Loading model {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.bfloat16,
        device_map=device,
        trust_remote_code=True,
    )
    model.eval()

    return model, tokenizer


def load_checkpoint(checkpoint_path: str, projector: FEELProjectorFull) -> float:
    """Load projector checkpoint."""
    if not Path(checkpoint_path).exists():
        print(f"  No checkpoint at {checkpoint_path}, starting fresh")
        return 0.15

    print(f"  Loading checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

    # Handle different checkpoint formats
    loaded = False

    if 'projector_state_dict' in ckpt:
        projector.load_state_dict(ckpt['projector_state_dict'])
        loaded = True
    elif 'state_dict' in ckpt:
        projector.load_state_dict(ckpt['state_dict'])
        loaded = True
    elif 'feel_stream_state' in ckpt:
        # v6 checkpoint format - extract projector weights
        fs_state = ckpt['feel_stream_state']
        projector_state = {}
        for k, v in fs_state.items():
            if k.startswith('projector.encoder.'):
                new_k = k.replace('projector.', '')
                projector_state[new_k] = v
            elif k.startswith('projector.log_scale'):
                projector_state['log_scale'] = v

        if projector_state:
            try:
                projector.load_state_dict(projector_state, strict=False)
                print(f"  Loaded {len(projector_state)} projector weights from v6 checkpoint")
                loaded = True
            except Exception as e:
                print(f"  Warning: Partial load from v6: {e}")

    if not loaded:
        print(f"  No compatible weights found, starting fresh")

    alpha = ckpt.get('alpha', 0.15)
    print(f"  Checkpoint alpha: {alpha:.6f} (using 0.15)")

    return 0.15  # Use recommended alpha


def save_checkpoint(
    save_path: str,
    projector: FEELProjectorFull,
    trainer: FEELTrainer,
    epoch: int,
    metrics: Dict,
):
    """Save training checkpoint."""
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    ckpt = {
        'version': TRAINER_VERSION,
        'epoch': epoch,
        'projector_state_dict': projector.state_dict(),
        'alpha': trainer.config.alpha,
        'config': trainer.config.__dict__,
        'metrics': metrics,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
    }

    if trainer.interoception_head is not None:
        ckpt['interoception_state_dict'] = trainer.interoception_head.state_dict()

    torch.save(ckpt, save_path)
    print(f"  Saved checkpoint: {save_path}")


def train_epoch(
    trainer: FEELTrainer,
    tokenizer,
    prompts: List[str],
    device: str,
    verbose: bool = True,
) -> Dict:
    """Run one training epoch."""
    metrics_history = []
    prev_feel_embed = None

    for i, prompt in enumerate(prompts):
        # Tokenize
        input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)

        # Training step
        loss, metrics, feel_embed = trainer.train_step(input_ids, prev_feel_embed)
        prev_feel_embed = feel_embed

        metrics_history.append(metrics)

        if verbose and (i + 1) % 10 == 0:
            print(f"    [{i+1}/{len(prompts)}] loss={metrics['total_loss']:.4f}, "
                  f"kl={metrics['kl_baseline_feel']:.4f}, norm={metrics['feel_norm']:.4f}")

    # Aggregate metrics
    aggregated = {}
    for key in metrics_history[0].keys():
        values = [m[key] for m in metrics_history if key in m]
        if values:
            aggregated[f"mean_{key}"] = np.mean(values)
            aggregated[f"std_{key}"] = np.std(values)

    return aggregated


def main():
    parser = argparse.ArgumentParser(description="FEEL v3 Training - Teacher Distillation")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--checkpoint", type=str,
                        default="results/feel_training/canonical_v6_checkpoint.pt")
    parser.add_argument("--output_dir", type=str, default="results/feel_training_v3")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--alpha", type=float, default=0.15)
    parser.add_argument("--teacher_samples", type=int, default=5)
    parser.add_argument("--distillation_weight", type=float, default=1.0)
    parser.add_argument("--kl_budget", type=float, default=0.1)
    parser.add_argument("--train_interoception", action="store_true", default=True)
    parser.add_argument("--no_interoception", action="store_true")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--verbose", action="store_true", default=True)
    args = parser.parse_args()

    print("=" * 70)
    print(f"  FEEL TRAINING {TRAINER_VERSION} - Teacher Distillation")
    print("=" * 70)
    print(f"  FEELStreamV3: {FEEL_STREAM_V3_VERSION}")
    print(f"  Alpha: {args.alpha}")
    print(f"  Teacher samples: {args.teacher_samples}")
    print(f"  KL budget: {args.kl_budget}")
    print(f"  Train interoception: {not args.no_interoception}")
    print("=" * 70)

    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # Load model
    model, tokenizer = create_model_and_tokenizer(args.model, args.device)
    embed_dim = model.config.hidden_size
    print(f"  Model embed_dim: {embed_dim}")

    # Create projector
    projector = FEELProjectorFull(embed_dim=embed_dim).to(args.device)

    # Load checkpoint
    load_checkpoint(args.checkpoint, projector)

    # Check projector output
    test_sensors = torch.randn(1, 16, device=args.device)
    test_out = projector(test_sensors)
    print(f"  Projector output norm: {test_out.norm().item():.4f}")

    # Create trainer config
    config = FEELTrainerConfig(
        alpha=args.alpha,
        teacher_samples=args.teacher_samples,
        distillation_weight=args.distillation_weight,
        kl_budget=args.kl_budget,
        train_interoception=not args.no_interoception,
        learning_rate=1e-4,
    )

    # Create trainer and move to device
    trainer = FEELTrainer(model, projector, config)
    if trainer.interoception_head is not None:
        trainer.interoception_head = trainer.interoception_head.to(args.device)

    # Setup telemetry
    try:
        sampler = TelemetrySampler(sample_hz=30)
        sampler.start()  # Must start the sampling thread!
        trainer.set_telemetry_sampler(sampler)
        backend = getattr(sampler, 'source', 'unknown')
        print(f"  Telemetry: {backend}")
    except Exception as e:
        print(f"  Telemetry unavailable: {e}")
        sampler = None

    # Training loop
    print(f"\n  Training on {len(TRAINING_PROMPTS)} prompts for {args.epochs} epochs")
    print("-" * 70)

    all_metrics = []

    for epoch in range(args.epochs):
        print(f"\n[Epoch {epoch + 1}/{args.epochs}]")

        # Shuffle prompts each epoch
        np.random.seed(42 + epoch)
        epoch_prompts = np.random.permutation(TRAINING_PROMPTS).tolist()

        # Train
        epoch_metrics = train_epoch(
            trainer, tokenizer, epoch_prompts, args.device, args.verbose
        )

        all_metrics.append(epoch_metrics)

        # Summary
        print(f"  Epoch {epoch + 1} summary:")
        print(f"    loss: {epoch_metrics['mean_total_loss']:.4f} ± {epoch_metrics['std_total_loss']:.4f}")
        print(f"    kl_baseline_feel: {epoch_metrics['mean_kl_baseline_feel']:.4f}")
        print(f"    feel_norm: {epoch_metrics['mean_feel_norm']:.4f}")

        if 'mean_r2_horizon_1' in epoch_metrics:
            print(f"    R² (1-step): {epoch_metrics['mean_r2_horizon_1']:.4f}")

        # Save checkpoint every epoch
        ckpt_path = f"{args.output_dir}/checkpoint_epoch_{epoch + 1}.pt"
        save_checkpoint(ckpt_path, projector, trainer, epoch + 1, epoch_metrics)

    # Save final checkpoint
    final_path = f"{args.output_dir}/final_checkpoint.pt"
    save_checkpoint(final_path, projector, trainer, args.epochs, all_metrics[-1])

    # Save training history
    history_path = f"{args.output_dir}/training_history.json"
    with open(history_path, 'w') as f:
        # Convert numpy types to Python types
        history = []
        for m in all_metrics:
            epoch_m = {}
            for k, v in m.items():
                if isinstance(v, (np.floating, np.integer)):
                    epoch_m[k] = float(v)
                else:
                    epoch_m[k] = v
            history.append(epoch_m)
        json.dump({
            'version': TRAINER_VERSION,
            'config': config.__dict__,
            'epochs': history,
        }, f, indent=2, default=str)
    print(f"\n  Training history: {history_path}")

    # Cleanup
    if sampler:
        sampler.stop()

    print("\n" + "=" * 70)
    print("  Training complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
