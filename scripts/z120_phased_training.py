#!/usr/bin/env python3
"""
z120 - Proper 3-Phase FEEL Training

Phase 1: Body is "read-only" to language
    - Train LM normally (next-token prediction)
    - Add Reporter head: predict telemetry from hidden states
    - Add Policy head: choose action from body latent + hidden
    - Body does NOT modulate logits yet

Phase 2: Gated injection in last N layers
    - Turn on FiLM in last 1-2 layers only
    - Gate bias strongly closed + invariance KL
    - Train with LM loss + invariance loss (KL to baseline)

Phase 3: LayerDrop compute actuation
    - Train with stochastic depth (LayerDrop)
    - Add distillation: shallow path matches full path logits

Dataset: TinyStories (fast, coherent, widely used)

Usage:
    python scripts/z120_phased_training.py --phase 1 --epochs 3
    python scripts/z120_phased_training.py --phase 2 --epochs 2 --checkpoint results/z120/phase1_final.pt
    python scripts/z120_phased_training.py --phase 3 --epochs 2 --checkpoint results/z120/phase2_final.pt
"""

import os
import sys
import time
import json
import argparse
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np

# HuggingFace
from transformers import AutoTokenizer


# =============================================================================
# Clamped Tokenizer (GPT-2 has 50k tokens, model has 32k)
# =============================================================================

class ClampedTokenizer:
    """
    Tokenizer wrapper that clamps token IDs to a max vocab size.

    GPT-2 tokenizer has 50257 tokens but our model uses 32000 vocab.
    """

    def __init__(self, base_tokenizer, max_vocab_size: int = 32000):
        self.base = base_tokenizer
        self.max_vocab_size = max_vocab_size
        self.pad_token = base_tokenizer.pad_token
        self.eos_token = base_tokenizer.eos_token
        self.eos_token_id = min(base_tokenizer.eos_token_id or 0, max_vocab_size - 1)
        self.pad_token_id = self.eos_token_id
        self.vocab_size = max_vocab_size

    def _clamp_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        return torch.clamp(tensor, min=0, max=self.max_vocab_size - 1)

    def _clamp_list(self, ids: list) -> list:
        if not ids:
            return ids
        if isinstance(ids[0], list):
            return [[min(max(0, id), self.max_vocab_size - 1) for id in row] for row in ids]
        return [min(max(0, id), self.max_vocab_size - 1) for id in ids]

    def __call__(self, text, **kwargs):
        result = self.base(text, **kwargs)
        output = {}
        for key in result.keys():
            value = result[key]
            if key == "input_ids":
                if isinstance(value, torch.Tensor):
                    output[key] = self._clamp_tensor(value)
                elif isinstance(value, list):
                    output[key] = self._clamp_list(value)
                else:
                    output[key] = value
            else:
                output[key] = value
        return output

    def encode(self, text, **kwargs):
        ids = self.base.encode(text, **kwargs)
        return self._clamp_list(ids) if isinstance(ids, list) else ids

    def decode(self, ids, **kwargs):
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()
        if isinstance(ids, list):
            ids = [min(max(0, id), self.base.vocab_size - 1) for id in ids]
        return self.base.decode(ids, **kwargs)


# =============================================================================
# Training Configuration
# =============================================================================

@dataclass
class TrainingConfig:
    """Training configuration."""
    # Phase
    phase: int = 1  # 1, 2, or 3

    # Model
    model_size: str = "30m"
    checkpoint: Optional[str] = None

    # Data
    dataset: str = "tinystories"
    max_seq_len: int = 256
    train_samples: int = 10000
    val_samples: int = 500

    # Training
    epochs: int = 3
    batch_size: int = 8
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    warmup_steps: int = 100
    grad_clip: float = 1.0

    # Phase-specific
    # Phase 1: Body read-only
    reporter_weight: float = 0.1
    policy_weight: float = 0.05

    # Phase 2: Gated injection
    film_layers: List[int] = None  # Which layers get FiLM (None = last 2)
    invariance_weight: float = 0.5
    invariance_temperature: float = 2.0

    # Phase 3: LayerDrop
    layerdrop_prob: float = 0.2
    distill_weight: float = 0.3

    # Telemetry simulation (for training without real GPU)
    simulate_telemetry: bool = True

    # Output
    output_dir: str = "results/z120_phased_training"
    save_every_n_steps: int = 500
    eval_every_n_steps: int = 100

    def __post_init__(self):
        if self.film_layers is None:
            # Default: FiLM only in last 2 layers
            if self.model_size == "30m":
                self.film_layers = [6, 7]  # 8 layers total
            else:
                self.film_layers = [10, 11]  # 12 layers total


# =============================================================================
# Dataset
# =============================================================================

class TinyStoriesDataset(Dataset):
    """TinyStories dataset for training."""

    def __init__(
        self,
        tokenizer,
        max_len: int = 256,
        split: str = "train",
        max_samples: int = 10000,
    ):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.samples = []

        print(f"Loading TinyStories ({split})...")
        try:
            from datasets import load_dataset
            ds = load_dataset("roneneldan/TinyStories", split=split, streaming=True)

            for i, item in enumerate(ds):
                if i >= max_samples:
                    break
                text = item["text"].strip()
                if len(text) > 50:  # Skip very short samples
                    self.samples.append(text)

            print(f"  Loaded {len(self.samples)} samples")

        except Exception as e:
            print(f"Could not load TinyStories: {e}")
            print("Using synthetic data for testing...")
            # Fallback synthetic data
            for i in range(max_samples):
                self.samples.append(
                    f"Once upon a time, there was a story number {i}. "
                    "It was a very nice story about things that happened. "
                    "The end was happy."
                )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        text = self.samples[idx]

        # Tokenize
        encoding = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_len,
            padding="max_length",
        )

        input_ids = encoding["input_ids"].squeeze(0)
        attention_mask = encoding["attention_mask"].squeeze(0)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": input_ids.clone(),
        }


# =============================================================================
# Telemetry Simulation
# =============================================================================

def simulate_telemetry(batch_size: int, device: str = "cuda") -> torch.Tensor:
    """
    Simulate telemetry for training without real GPU sensing.

    Returns realistic-looking telemetry vectors.
    """
    # Simulate: [power, temp, util, clock, ...]
    # Normalized values
    telemetry = torch.zeros(batch_size, 12, device=device)

    # Power: 10-40W range normalized by 50W
    telemetry[:, 0] = torch.rand(batch_size, device=device) * 0.6 + 0.2

    # Temp: 30-70°C range normalized by 100°C
    telemetry[:, 1] = torch.rand(batch_size, device=device) * 0.4 + 0.3

    # Util: 0-100%
    telemetry[:, 2] = torch.rand(batch_size, device=device)

    # Clock: normalized
    telemetry[:, 3] = torch.rand(batch_size, device=device) * 0.8 + 0.2

    return telemetry


# =============================================================================
# Phase 1: Body Read-Only Training
# =============================================================================

class Phase1Trainer:
    """
    Phase 1: Body is "read-only" to language.

    - Train LM normally
    - Add Reporter head (predict telemetry from hidden)
    - Add Policy head (predict action from body + hidden)
    - Body does NOT modulate logits
    """

    def __init__(self, model, config: TrainingConfig, device: str = "cuda"):
        self.model = model
        self.config = config
        self.device = device

        # Ensure baseline mode (no FiLM modulation)
        self.model.base_model.set_training_phase("baseline")

        # Optimizer
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

        # Metrics
        self.step = 0
        self.metrics = {"lm_loss": [], "reporter_loss": [], "policy_loss": []}

    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Single training step."""
        self.model.train()
        self.optimizer.zero_grad()

        input_ids = batch["input_ids"].to(self.device)
        labels = batch["labels"].to(self.device)

        # Simulate telemetry (or use real if available)
        telemetry = simulate_telemetry(input_ids.shape[0], self.device)

        # Forward (baseline mode - FiLM inactive)
        # But we still pass telemetry so body encoder trains
        self.model.base_model.set_training_phase("adapter")  # Temporarily enable body
        out = self.model(input_ids, telemetry, return_policy=True)
        self.model.base_model.set_training_phase("baseline")  # Back to baseline

        logits = out["logits"]

        # LM loss
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        lm_loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=self.model.base_model.config.vocab_size,  # Pad token
        )

        # Reporter loss (body head predicts telemetry)
        reporter_loss = torch.tensor(0.0, device=self.device)
        if "body_pred" in out:
            reporter_loss = F.mse_loss(out["body_pred"], telemetry)

        # Policy loss (cross-entropy to "ground truth" action)
        # For now, use simple heuristic: high power/temp → ECO (0), low → PERF (2)
        policy_loss = torch.tensor(0.0, device=self.device)
        if "policy_logits" in out:
            # Simple target: based on power level
            power = telemetry[:, 0]
            targets = torch.zeros(input_ids.shape[0], dtype=torch.long, device=self.device)
            targets[power > 0.6] = 0  # ECO for high power
            targets[power < 0.3] = 2  # PERF for low power
            targets[(power >= 0.3) & (power <= 0.6)] = 1  # BALANCED
            policy_loss = F.cross_entropy(out["policy_logits"], targets)

        # Total loss
        total_loss = (
            lm_loss +
            self.config.reporter_weight * reporter_loss +
            self.config.policy_weight * policy_loss
        )

        # Backward
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
        self.optimizer.step()

        self.step += 1

        return {
            "total_loss": total_loss.item(),
            "lm_loss": lm_loss.item(),
            "reporter_loss": reporter_loss.item(),
            "policy_loss": policy_loss.item(),
        }


# =============================================================================
# Phase 2: Gated Injection Training
# =============================================================================

class Phase2Trainer:
    """
    Phase 2: Gated injection in last N layers.

    - FiLM active in last 1-2 layers only
    - Gate bias strongly closed
    - Invariance KL loss to keep language stable
    """

    def __init__(
        self,
        model,
        baseline_model,  # Frozen copy for KL
        config: TrainingConfig,
        device: str = "cuda",
    ):
        self.model = model
        self.baseline_model = baseline_model
        self.config = config
        self.device = device

        # Set which layers have FiLM active
        self._setup_film_layers()

        # Enable full mode
        self.model.base_model.set_training_phase("full")

        # Freeze baseline
        self.baseline_model.eval()
        for p in self.baseline_model.parameters():
            p.requires_grad = False

        # Optimizer
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=config.learning_rate * 0.5,  # Lower LR for fine-tuning
            weight_decay=config.weight_decay,
        )

        self.step = 0

    def _setup_film_layers(self):
        """Configure which layers have FiLM active."""
        for i, layer in enumerate(self.model.base_model.layers):
            if hasattr(layer, "has_film"):
                if i in self.config.film_layers:
                    layer.has_film = True
                    # Reset FiLM to near-identity
                    if hasattr(layer, "film"):
                        nn.init.zeros_(layer.film.film_proj[0].weight)
                        nn.init.zeros_(layer.film.film_proj[0].bias)
                else:
                    layer.has_film = False

    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Single training step with invariance KL."""
        self.model.train()
        self.optimizer.zero_grad()

        input_ids = batch["input_ids"].to(self.device)
        labels = batch["labels"].to(self.device)
        telemetry = simulate_telemetry(input_ids.shape[0], self.device)

        # Forward with body
        out = self.model(input_ids, telemetry, return_policy=True)
        logits = out["logits"]

        # Baseline forward (no body)
        with torch.no_grad():
            baseline_out = self.baseline_model.base_model(input_ids)
            baseline_logits = baseline_out["logits"]

        # LM loss
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        lm_loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
        )

        # Invariance KL loss
        T = self.config.invariance_temperature
        log_probs = F.log_softmax(logits / T, dim=-1)
        baseline_probs = F.softmax(baseline_logits / T, dim=-1)
        kl_loss = F.kl_div(log_probs, baseline_probs, reduction="batchmean") * (T * T)

        # Total loss
        total_loss = lm_loss + self.config.invariance_weight * kl_loss

        # Backward
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
        self.optimizer.step()

        self.step += 1

        return {
            "total_loss": total_loss.item(),
            "lm_loss": lm_loss.item(),
            "kl_loss": kl_loss.item(),
        }


# =============================================================================
# Phase 3: LayerDrop Training
# =============================================================================

class Phase3Trainer:
    """
    Phase 3: LayerDrop compute actuation.

    - Train with stochastic depth
    - Distillation: shallow path matches full path logits
    """

    def __init__(
        self,
        model,
        config: TrainingConfig,
        device: str = "cuda",
    ):
        self.model = model
        self.config = config
        self.device = device

        # Enable LayerDrop
        self.model.base_model.set_layerdrop(True, prob=config.layerdrop_prob)
        self.model.base_model.set_training_phase("full")

        # Optimizer
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=config.learning_rate * 0.3,
            weight_decay=config.weight_decay,
        )

        self.step = 0

    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Single training step with LayerDrop + distillation."""
        self.model.train()
        self.optimizer.zero_grad()

        input_ids = batch["input_ids"].to(self.device)
        labels = batch["labels"].to(self.device)
        telemetry = simulate_telemetry(input_ids.shape[0], self.device)

        # Forward with LayerDrop (stochastic depth during training)
        self.model.base_model.set_layerdrop(True, prob=self.config.layerdrop_prob)
        out_dropped = self.model(input_ids, telemetry, return_policy=True)
        logits_dropped = out_dropped["logits"]

        # Forward without LayerDrop (full model - for distillation target)
        with torch.no_grad():
            self.model.base_model.set_layerdrop(False)
            out_full = self.model(input_ids, telemetry, return_policy=False)
            logits_full = out_full["logits"]
            self.model.base_model.set_layerdrop(True, prob=self.config.layerdrop_prob)

        # LM loss
        shift_logits = logits_dropped[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        lm_loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
        )

        # Distillation loss (dropped matches full)
        T = 2.0
        log_probs = F.log_softmax(logits_dropped / T, dim=-1)
        full_probs = F.softmax(logits_full / T, dim=-1)
        distill_loss = F.kl_div(log_probs, full_probs, reduction="batchmean") * (T * T)

        # Total loss
        total_loss = lm_loss + self.config.distill_weight * distill_loss

        # Backward
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
        self.optimizer.step()

        self.step += 1

        return {
            "total_loss": total_loss.item(),
            "lm_loss": lm_loss.item(),
            "distill_loss": distill_loss.item(),
        }


# =============================================================================
# Main Training Loop
# =============================================================================

def train_phase(config: TrainingConfig, device: str = "cuda"):
    """Run training for specified phase."""

    print("=" * 60)
    print(f"FEEL Phase {config.phase} Training")
    print("=" * 60)

    # Set HSA override for AMD
    os.environ["HSA_OVERRIDE_GFX_VERSION"] = "11.0.0"

    # Output directory
    output_dir = Path(config.output_dir) / f"phase{config.phase}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Tokenizer (clamped to 32k vocab)
    print("\n1. Loading tokenizer...")
    base_tokenizer = AutoTokenizer.from_pretrained("gpt2")
    base_tokenizer.pad_token = base_tokenizer.eos_token
    tokenizer = ClampedTokenizer(base_tokenizer, max_vocab_size=32000)

    # Dataset
    print("\n2. Loading dataset...")
    train_ds = TinyStoriesDataset(
        tokenizer, config.max_seq_len, "train", config.train_samples
    )
    val_ds = TinyStoriesDataset(
        tokenizer, config.max_seq_len, "validation", config.val_samples
    )

    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=config.batch_size)

    # Model
    print(f"\n3. Creating model ({config.model_size})...")
    from feel_slm.embodied_slm import create_embodied_slm_30m, create_embodied_slm_125m
    from feel_slm.feel_runtime import create_feel_runtime, FEELConfig

    if config.model_size == "30m":
        base_model = create_embodied_slm_30m()
    else:
        base_model = create_embodied_slm_125m()

    # Load checkpoint if provided
    if config.checkpoint:
        print(f"   Loading checkpoint: {config.checkpoint}")
        checkpoint = torch.load(config.checkpoint, map_location=device)
        base_model.load_state_dict(checkpoint["model_state_dict"])

    base_model = base_model.to(device)

    # Create FEEL model wrapper
    feel_config = FEELConfig(
        layerdrop_eco=0.4,
        layerdrop_balanced=0.0,
        min_layers=4,
    )
    from feel_slm.feel_runtime import FEELModel
    model = FEELModel(base_model, feel_config).to(device)

    print(f"   Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")

    # Create trainer for phase
    if config.phase == 1:
        trainer = Phase1Trainer(model, config, device)
    elif config.phase == 2:
        # Need frozen baseline for KL
        baseline_model = create_embodied_slm_30m() if config.model_size == "30m" else create_embodied_slm_125m()
        if config.checkpoint:
            baseline_model.load_state_dict(checkpoint["model_state_dict"])
        baseline_model = baseline_model.to(device)
        baseline_feel = FEELModel(baseline_model, feel_config).to(device)
        trainer = Phase2Trainer(model, baseline_feel, config, device)
    else:
        trainer = Phase3Trainer(model, config, device)

    # Training loop
    print(f"\n4. Training Phase {config.phase}...")
    print(f"   Epochs: {config.epochs}")
    print(f"   Batch size: {config.batch_size}")
    print(f"   Steps per epoch: {len(train_loader)}")

    best_val_loss = float("inf")
    history = {"train": [], "val": []}

    for epoch in range(config.epochs):
        print(f"\n--- Epoch {epoch + 1}/{config.epochs} ---")

        # Training
        model.train()
        epoch_losses = []

        for batch_idx, batch in enumerate(train_loader):
            losses = trainer.train_step(batch)
            epoch_losses.append(losses["total_loss"])

            if (batch_idx + 1) % 50 == 0:
                avg_loss = np.mean(epoch_losses[-50:])
                print(f"  Step {trainer.step}: loss={avg_loss:.4f}")

            # Save checkpoint periodically
            if trainer.step % config.save_every_n_steps == 0:
                save_checkpoint(
                    model, trainer.optimizer, trainer.step, epoch,
                    output_dir / f"checkpoint_step{trainer.step}.pt"
                )

        # Validation
        val_loss = evaluate(model, val_loader, device)
        print(f"  Epoch {epoch + 1} - train_loss={np.mean(epoch_losses):.4f}, val_loss={val_loss:.4f}")

        history["train"].append(float(np.mean(epoch_losses)))
        history["val"].append(float(val_loss))

        # Save best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(
                model, trainer.optimizer, trainer.step, epoch,
                output_dir / f"phase{config.phase}_best.pt"
            )

    # Save final
    save_checkpoint(
        model, trainer.optimizer, trainer.step, config.epochs,
        output_dir / f"phase{config.phase}_final.pt"
    )

    # Save history
    with open(output_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    print(f"\n5. Training complete!")
    print(f"   Best val loss: {best_val_loss:.4f}")
    print(f"   Checkpoints saved to: {output_dir}")


def evaluate(model, dataloader, device) -> float:
    """Evaluate model on validation set."""
    model.eval()
    total_loss = 0
    total_samples = 0

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            telemetry = simulate_telemetry(input_ids.shape[0], device)

            out = model(input_ids, telemetry, return_policy=False)
            logits = out["logits"]

            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
            )

            total_loss += loss.item() * input_ids.shape[0]
            total_samples += input_ids.shape[0]

    return total_loss / total_samples if total_samples > 0 else float("inf")


def save_checkpoint(model, optimizer, step, epoch, path):
    """Save training checkpoint."""
    torch.save({
        "step": step,
        "epoch": epoch,
        "model_state_dict": model.base_model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }, path)
    print(f"  Saved checkpoint: {path}")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="FEEL Phased Training")
    parser.add_argument("--phase", type=int, required=True, choices=[1, 2, 3])
    parser.add_argument("--model-size", type=str, default="30m", choices=["30m", "125m"])
    parser.add_argument("--checkpoint", type=str, default=None, help="Previous phase checkpoint")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--train-samples", type=int, default=5000)
    parser.add_argument("--val-samples", type=int, default=500)
    parser.add_argument("--output-dir", type=str, default="results/z120_phased_training")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    config = TrainingConfig(
        phase=args.phase,
        model_size=args.model_size,
        checkpoint=args.checkpoint,
        epochs=args.epochs,
        batch_size=args.batch_size,
        train_samples=args.train_samples,
        val_samples=args.val_samples,
        output_dir=args.output_dir,
    )

    train_phase(config, args.device)


if __name__ == "__main__":
    main()
