#!/usr/bin/env python3
"""
Z114: LayerDrop Training with KL Anchoring

This script trains LayerDrop as a semantic-preserving energy knob.

Key insight: To make "math actuate hardware" without semantic garbage,
we need to train the model so that dropping layers preserves outputs.

Training recipe:
1. Train baseline LM normally (creates semantic anchor)
2. Train FEEL LM with:
   - LM loss (language modeling objective)
   - KL divergence to baseline logits (semantic anchoring)
   - LayerDrop "eco mode" active some % of batches

This makes "shallower compute" produce stable language while saving energy.

Reference: "Reducing Transformer Depth on Demand with Structured Dropout"
https://arxiv.org/abs/1909.11556

Usage:
    HSA_OVERRIDE_GFX_VERSION=11.0.0 python scripts/z114_layerdrop_training.py

Author: FEEL Research Team
Date: 2026-01-21
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional, Dict, List, Tuple
import math

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from src.feel_slm.model_v2 import FEELSLMV2, FEELConfigV2, BaselineSLMV2


# =============================================================================
# Training Configuration
# =============================================================================

@dataclass
class TrainingConfig:
    """Training hyperparameters."""
    # Model
    vocab_size: int = 32000
    hidden_dim: int = 256
    num_layers: int = 4
    num_heads: int = 4
    max_seq_len: int = 256

    # LayerDrop
    layerdrop_layers: List[int] = None  # Which layers can be dropped
    layerdrop_prob: float = 0.3         # Probability of dropping in eco mode

    # Training
    batch_size: int = 8
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    epochs: int = 10
    warmup_steps: int = 100
    gradient_clip: float = 1.0

    # KL anchoring
    kl_weight: float = 0.1              # Weight for KL divergence loss
    kl_temperature: float = 2.0         # Temperature for KL (higher = softer)
    eco_batch_prob: float = 0.5         # Prob of using eco mode per batch

    # Paths
    output_dir: str = "results/z114_layerdrop"
    checkpoint_interval: int = 100

    def __post_init__(self):
        if self.layerdrop_layers is None:
            self.layerdrop_layers = [1, 2]  # Default: drop middle layers


# =============================================================================
# Synthetic Dataset (for demonstration)
# =============================================================================

class SyntheticTextDataset(Dataset):
    """
    Synthetic dataset for training demonstration.

    In practice, replace with real text data (e.g., from Ouroboros dataset).
    """

    def __init__(self, vocab_size: int, seq_len: int, n_samples: int = 1000):
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.n_samples = n_samples

        # Generate synthetic sequences with patterns
        self.data = []
        for i in range(n_samples):
            # Create sequences with some structure (not pure random)
            base = torch.randint(100, vocab_size - 100, (1,)).item()
            seq = torch.zeros(seq_len, dtype=torch.long)

            for j in range(seq_len):
                # Add some predictable patterns
                if j % 10 == 0:
                    seq[j] = base
                elif j % 5 == 0:
                    seq[j] = (base + 100) % vocab_size
                else:
                    seq[j] = torch.randint(100, vocab_size - 100, (1,)).item()

            self.data.append(seq)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        seq = self.data[idx]
        # Input is seq[:-1], target is seq[1:]
        return seq[:-1], seq[1:]


# =============================================================================
# Trainer with KL Anchoring
# =============================================================================

class LayerDropTrainer:
    """
    Trainer for LayerDrop with KL anchoring.

    Phase 1: Train baseline (semantic anchor)
    Phase 2: Train FEEL with KL constraint
    """

    def __init__(self, config: TrainingConfig, device: torch.device):
        self.config = config
        self.device = device

        # Build model config
        self.model_config = FEELConfigV2(
            vocab_size=config.vocab_size,
            hidden_dim=config.hidden_dim,
            num_layers=config.num_layers,
            num_heads=config.num_heads,
            num_kv_heads=config.num_heads // 2,
            intermediate_dim=config.hidden_dim * 2,
            max_seq_len=config.max_seq_len,
            phase=1,
            enable_film=False,
            enable_gating=False,
            enable_layerdrop=True,
            layerdrop_layers=config.layerdrop_layers,
        )

        # Models
        self.baseline_model: Optional[BaselineSLMV2] = None
        self.feel_model: Optional[FEELSLMV2] = None

        # Metrics
        self.metrics = {
            "baseline_loss": [],
            "feel_loss": [],
            "kl_loss": [],
            "eco_loss": [],
        }

    def create_baseline(self) -> BaselineSLMV2:
        """Create baseline model (semantic anchor)."""
        self.baseline_model = BaselineSLMV2(self.model_config).to(self.device)
        return self.baseline_model

    def create_feel(self) -> FEELSLMV2:
        """Create FEEL model for LayerDrop training."""
        self.feel_model = FEELSLMV2(self.model_config).to(self.device)

        # Initialize from baseline if available
        if self.baseline_model is not None:
            self._copy_lm_weights()

        return self.feel_model

    def _copy_lm_weights(self):
        """Copy LM weights from baseline to FEEL model."""
        baseline_state = self.baseline_model.state_dict()
        feel_state = self.feel_model.state_dict()

        # Copy matching keys (LM backbone)
        copied = 0
        for key in baseline_state:
            if key in feel_state:
                if baseline_state[key].shape == feel_state[key].shape:
                    feel_state[key] = baseline_state[key].clone()
                    copied += 1

        self.feel_model.load_state_dict(feel_state)
        print(f"Copied {copied} parameters from baseline to FEEL")

    def train_baseline(
        self,
        train_loader: DataLoader,
        epochs: int = None
    ) -> Dict:
        """
        Phase 1: Train baseline model as semantic anchor.
        """
        epochs = epochs or self.config.epochs
        model = self.baseline_model
        model.train()

        optimizer = AdamW(
            model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )

        total_steps = len(train_loader) * epochs
        scheduler = CosineAnnealingLR(optimizer, T_max=total_steps)

        print(f"\n{'='*60}")
        print("Phase 1: Training Baseline (Semantic Anchor)")
        print(f"{'='*60}")
        print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
        print(f"Epochs: {epochs}, Steps per epoch: {len(train_loader)}")

        step = 0
        for epoch in range(epochs):
            epoch_loss = 0.0
            epoch_steps = 0

            for batch_idx, (input_ids, labels) in enumerate(train_loader):
                input_ids = input_ids.to(self.device)
                labels = labels.to(self.device)

                optimizer.zero_grad()

                # Forward
                outputs = model(input_ids)
                logits = outputs["logits"]

                # LM loss
                loss = F.cross_entropy(
                    logits.view(-1, self.config.vocab_size),
                    labels.view(-1),
                    ignore_index=-100,
                )

                # Backward
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), self.config.gradient_clip)
                optimizer.step()
                scheduler.step()

                epoch_loss += loss.item()
                epoch_steps += 1
                step += 1

                if step % 50 == 0:
                    print(f"  Step {step}: loss={loss.item():.4f}, lr={scheduler.get_last_lr()[0]:.2e}")

            avg_loss = epoch_loss / epoch_steps
            self.metrics["baseline_loss"].append(avg_loss)
            print(f"Epoch {epoch+1}/{epochs}: avg_loss={avg_loss:.4f}")

        return {"final_loss": avg_loss, "steps": step}

    def train_feel_with_kl(
        self,
        train_loader: DataLoader,
        epochs: int = None
    ) -> Dict:
        """
        Phase 2: Train FEEL model with KL anchoring.

        Loss = LM_loss + kl_weight * KL(FEEL || Baseline)

        For eco-mode batches, additionally penalize deviation when layers are dropped.
        """
        epochs = epochs or self.config.epochs

        # Ensure baseline is frozen (no gradients)
        self.baseline_model.eval()
        for p in self.baseline_model.parameters():
            p.requires_grad = False

        model = self.feel_model
        model.train()

        optimizer = AdamW(
            model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )

        total_steps = len(train_loader) * epochs
        scheduler = CosineAnnealingLR(optimizer, T_max=total_steps)

        print(f"\n{'='*60}")
        print("Phase 2: Training FEEL with KL Anchoring")
        print(f"{'='*60}")
        print(f"KL weight: {self.config.kl_weight}")
        print(f"KL temperature: {self.config.kl_temperature}")
        print(f"Eco batch probability: {self.config.eco_batch_prob}")
        print(f"LayerDrop layers: {self.config.layerdrop_layers}")

        step = 0
        for epoch in range(epochs):
            epoch_lm_loss = 0.0
            epoch_kl_loss = 0.0
            epoch_eco_loss = 0.0
            epoch_steps = 0
            eco_batches = 0

            for batch_idx, (input_ids, labels) in enumerate(train_loader):
                input_ids = input_ids.to(self.device)
                labels = labels.to(self.device)

                optimizer.zero_grad()

                # Decide if this is an eco-mode batch
                use_eco = torch.rand(1).item() < self.config.eco_batch_prob

                if use_eco:
                    model.set_mode("eco")
                    eco_batches += 1
                else:
                    model.set_mode("balanced")

                # Create dummy telemetry (not used in phase 1)
                telemetry = torch.zeros(input_ids.shape[0], 12, device=self.device)

                # Forward FEEL model
                feel_outputs = model(input_ids, telemetry=telemetry)
                feel_logits = feel_outputs["logits"]

                # Forward baseline model (no grad)
                with torch.no_grad():
                    baseline_outputs = self.baseline_model(input_ids)
                    baseline_logits = baseline_outputs["logits"]

                # LM loss
                lm_loss = F.cross_entropy(
                    feel_logits.view(-1, self.config.vocab_size),
                    labels.view(-1),
                    ignore_index=-100,
                )

                # KL divergence loss (anchoring to baseline)
                # Use temperature to soften distributions
                T = self.config.kl_temperature
                feel_probs = F.log_softmax(feel_logits / T, dim=-1)
                baseline_probs = F.softmax(baseline_logits / T, dim=-1)

                kl_loss = F.kl_div(
                    feel_probs.view(-1, self.config.vocab_size),
                    baseline_probs.view(-1, self.config.vocab_size),
                    reduction="batchmean",
                ) * (T * T)  # Scale by T^2 per KD convention

                # Total loss
                total_loss = lm_loss + self.config.kl_weight * kl_loss

                # Backward
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), self.config.gradient_clip)
                optimizer.step()
                scheduler.step()

                epoch_lm_loss += lm_loss.item()
                epoch_kl_loss += kl_loss.item()
                if use_eco:
                    epoch_eco_loss += lm_loss.item()
                epoch_steps += 1
                step += 1

                if step % 50 == 0:
                    mode = "eco" if use_eco else "balanced"
                    print(f"  Step {step} [{mode}]: lm={lm_loss.item():.4f}, "
                          f"kl={kl_loss.item():.4f}, total={total_loss.item():.4f}")

            # Epoch summary
            avg_lm = epoch_lm_loss / epoch_steps
            avg_kl = epoch_kl_loss / epoch_steps
            avg_eco = epoch_eco_loss / eco_batches if eco_batches > 0 else 0

            self.metrics["feel_loss"].append(avg_lm)
            self.metrics["kl_loss"].append(avg_kl)
            self.metrics["eco_loss"].append(avg_eco)

            print(f"Epoch {epoch+1}/{epochs}: lm={avg_lm:.4f}, kl={avg_kl:.4f}, "
                  f"eco_loss={avg_eco:.4f}, eco_batches={eco_batches}/{epoch_steps}")

        return {
            "final_lm_loss": avg_lm,
            "final_kl_loss": avg_kl,
            "final_eco_loss": avg_eco,
            "steps": step,
        }

    def evaluate_layerdrop_quality(self, eval_loader: DataLoader) -> Dict:
        """
        Evaluate semantic preservation when using LayerDrop.

        Compares:
        - Baseline perplexity
        - FEEL balanced mode perplexity
        - FEEL eco mode perplexity (with LayerDrop)
        - KL divergence between modes
        """
        self.baseline_model.eval()
        self.feel_model.eval()

        results = {
            "baseline_ppl": 0.0,
            "feel_balanced_ppl": 0.0,
            "feel_eco_ppl": 0.0,
            "kl_balanced_vs_baseline": 0.0,
            "kl_eco_vs_baseline": 0.0,
            "kl_eco_vs_balanced": 0.0,
        }

        total_tokens = 0
        baseline_loss = 0.0
        balanced_loss = 0.0
        eco_loss = 0.0

        kl_balanced = 0.0
        kl_eco_base = 0.0
        kl_eco_balanced = 0.0

        with torch.no_grad():
            for input_ids, labels in eval_loader:
                input_ids = input_ids.to(self.device)
                labels = labels.to(self.device)
                batch_tokens = labels.numel()
                total_tokens += batch_tokens

                telemetry = torch.zeros(input_ids.shape[0], 12, device=self.device)

                # Baseline
                baseline_out = self.baseline_model(input_ids)
                baseline_logits = baseline_out["logits"]
                baseline_loss += F.cross_entropy(
                    baseline_logits.view(-1, self.config.vocab_size),
                    labels.view(-1),
                    reduction="sum",
                ).item()

                # FEEL balanced
                self.feel_model.set_mode("balanced")
                balanced_out = self.feel_model(input_ids, telemetry=telemetry)
                balanced_logits = balanced_out["logits"]
                balanced_loss += F.cross_entropy(
                    balanced_logits.view(-1, self.config.vocab_size),
                    labels.view(-1),
                    reduction="sum",
                ).item()

                # FEEL eco (with LayerDrop)
                self.feel_model.set_mode("eco")
                eco_out = self.feel_model(input_ids, telemetry=telemetry)
                eco_logits = eco_out["logits"]
                eco_loss += F.cross_entropy(
                    eco_logits.view(-1, self.config.vocab_size),
                    labels.view(-1),
                    reduction="sum",
                ).item()

                # KL divergences
                baseline_probs = F.softmax(baseline_logits, dim=-1)
                balanced_probs = F.softmax(balanced_logits, dim=-1)
                eco_probs = F.softmax(eco_logits, dim=-1)

                kl_balanced += F.kl_div(
                    F.log_softmax(balanced_logits, dim=-1).view(-1, self.config.vocab_size),
                    baseline_probs.view(-1, self.config.vocab_size),
                    reduction="sum",
                ).item()

                kl_eco_base += F.kl_div(
                    F.log_softmax(eco_logits, dim=-1).view(-1, self.config.vocab_size),
                    baseline_probs.view(-1, self.config.vocab_size),
                    reduction="sum",
                ).item()

                kl_eco_balanced += F.kl_div(
                    F.log_softmax(eco_logits, dim=-1).view(-1, self.config.vocab_size),
                    balanced_probs.view(-1, self.config.vocab_size),
                    reduction="sum",
                ).item()

        # Compute perplexities
        results["baseline_ppl"] = math.exp(baseline_loss / total_tokens)
        results["feel_balanced_ppl"] = math.exp(balanced_loss / total_tokens)
        results["feel_eco_ppl"] = math.exp(eco_loss / total_tokens)

        # Average KL
        results["kl_balanced_vs_baseline"] = kl_balanced / total_tokens
        results["kl_eco_vs_baseline"] = kl_eco_base / total_tokens
        results["kl_eco_vs_balanced"] = kl_eco_balanced / total_tokens

        return results

    def save_checkpoint(self, path: Path, epoch: int):
        """Save training checkpoint."""
        path.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            "epoch": epoch,
            "config": asdict(self.config),
            "model_config": asdict(self.model_config),
            "metrics": self.metrics,
        }

        if self.baseline_model:
            checkpoint["baseline_state"] = self.baseline_model.state_dict()

        if self.feel_model:
            checkpoint["feel_state"] = self.feel_model.state_dict()

        torch.save(checkpoint, path / f"checkpoint_epoch{epoch}.pt")

        # Save metrics separately for easy access
        with open(path / "metrics.json", "w") as f:
            json.dump(self.metrics, f, indent=2)

    def load_checkpoint(self, path: Path):
        """Load training checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)

        if "baseline_state" in checkpoint and self.baseline_model:
            self.baseline_model.load_state_dict(checkpoint["baseline_state"])

        if "feel_state" in checkpoint and self.feel_model:
            self.feel_model.load_state_dict(checkpoint["feel_state"])

        self.metrics = checkpoint.get("metrics", self.metrics)
        return checkpoint.get("epoch", 0)


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="LayerDrop Training with KL Anchoring")
    parser.add_argument("--epochs", type=int, default=5, help="Training epochs per phase")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--kl-weight", type=float, default=0.1, help="KL divergence weight")
    parser.add_argument("--eco-prob", type=float, default=0.5, help="Eco mode batch probability")
    parser.add_argument("--output-dir", type=str, default="results/z114_layerdrop")
    parser.add_argument("--skip-baseline", action="store_true", help="Skip baseline training")
    args = parser.parse_args()

    # Check device
    if not torch.cuda.is_available():
        print("ERROR: CUDA not available")
        sys.exit(1)

    device = torch.device("cuda")
    print(f"Device: {torch.cuda.get_device_name(0)}")

    # Configuration
    config = TrainingConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        kl_weight=args.kl_weight,
        eco_batch_prob=args.eco_prob,
        output_dir=args.output_dir,
    )

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create trainer
    trainer = LayerDropTrainer(config, device)

    # Create dataset
    print("\nCreating synthetic dataset...")
    train_dataset = SyntheticTextDataset(
        vocab_size=config.vocab_size,
        seq_len=config.max_seq_len,
        n_samples=2000,
    )
    eval_dataset = SyntheticTextDataset(
        vocab_size=config.vocab_size,
        seq_len=config.max_seq_len,
        n_samples=200,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=0,
    )
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=config.batch_size,
        shuffle=False,
    )

    # Phase 1: Train baseline
    if not args.skip_baseline:
        trainer.create_baseline()
        baseline_results = trainer.train_baseline(train_loader)
        print(f"\nBaseline training complete: {baseline_results}")

    # Phase 2: Train FEEL with KL anchoring
    trainer.create_feel()
    feel_results = trainer.train_feel_with_kl(train_loader)
    print(f"\nFEEL training complete: {feel_results}")

    # Evaluate LayerDrop quality
    print("\n" + "=" * 60)
    print("Evaluating LayerDrop Semantic Preservation")
    print("=" * 60)

    eval_results = trainer.evaluate_layerdrop_quality(eval_loader)

    print(f"\nPerplexity comparison:")
    print(f"  Baseline:       {eval_results['baseline_ppl']:.2f}")
    print(f"  FEEL balanced:  {eval_results['feel_balanced_ppl']:.2f}")
    print(f"  FEEL eco:       {eval_results['feel_eco_ppl']:.2f}")

    print(f"\nKL divergence (semantic drift):")
    print(f"  Balanced vs Baseline: {eval_results['kl_balanced_vs_baseline']:.4f}")
    print(f"  Eco vs Baseline:      {eval_results['kl_eco_vs_baseline']:.4f}")
    print(f"  Eco vs Balanced:      {eval_results['kl_eco_vs_balanced']:.4f}")

    # Compute quality metrics
    ppl_increase = (eval_results['feel_eco_ppl'] - eval_results['baseline_ppl']) / eval_results['baseline_ppl'] * 100

    print(f"\n📊 Semantic quality score:")
    print(f"  PPL increase in eco mode: {ppl_increase:+.1f}%")

    if ppl_increase < 5:
        print("  ✅ EXCELLENT: LayerDrop preserves semantics well (<5% PPL increase)")
    elif ppl_increase < 15:
        print("  ⚠️ ACCEPTABLE: Some semantic drift (5-15% PPL increase)")
    else:
        print("  ❌ NEEDS WORK: Significant semantic drift (>15% PPL increase)")

    # Save final checkpoint
    trainer.save_checkpoint(output_dir, config.epochs)

    # Save evaluation results
    with open(output_dir / "eval_results.json", "w") as f:
        json.dump(eval_results, f, indent=2)

    print(f"\n✅ Results saved to {output_dir}")
    print(f"\nNext steps:")
    print(f"  1. Use trained model for inference with phase-separated controller")
    print(f"  2. Collect trajectory data for dynamics model training")
    print(f"  3. Train PolicyHead to select eco mode optimally")


if __name__ == "__main__":
    main()
