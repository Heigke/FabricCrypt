#!/usr/bin/env python3
"""
z116 - EmbodiedSLM 3-Step Training

Training regimen for semantics-safe embodiment:

Step 1 - Language Baseline (pure LM):
  - Train on text data without body signals
  - Establishes language semantics baseline

Step 2 - Adapter Warmup (frozen trunk):
  - Freeze trunk + LM head
  - Train body encoder + FiLM layers + body head
  - Goal: body channel becomes meaningful without moving language

Step 3 - Semantics Lock (KL distillation):
  - Unfreeze everything
  - Add KL term to keep LM distribution close to Step 1 baseline
  - Model can "feel" body while language remains stable

Usage:
  python z116_embodied_training.py --step 1 --epochs 10  # Baseline
  python z116_embodied_training.py --step 2 --epochs 5   # Adapter
  python z116_embodied_training.py --step 3 --epochs 5   # Full + KL
  python z116_embodied_training.py --all                 # All steps
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.feel_slm.embodied_slm import (
    EmbodiedSLM, EmbodiedSLMConfig, EmbodiedLoss,
    create_embodied_slm_30m, create_embodied_slm_125m,
)


@dataclass
class TrainingConfig:
    """Training configuration."""
    # Model
    model_size: str = "30m"  # "30m" or "125m"

    # Data
    data_path: str = "data/ouroboros/train_responses.jsonl"
    max_seq_len: int = 512
    batch_size: int = 8

    # Training
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    warmup_steps: int = 100
    gradient_clip: float = 1.0

    # Step-specific
    step1_epochs: int = 10
    step2_epochs: int = 5
    step3_epochs: int = 5

    # Loss weights
    lm_weight: float = 1.0
    body_weight: float = 0.1
    kl_weight: float = 0.5
    kl_temperature: float = 2.0

    # Checkpoints
    output_dir: str = "results/z116_embodied"
    save_every: int = 1000
    log_every: int = 100

    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class TextDataset(Dataset):
    """Simple text dataset for language modeling."""

    def __init__(self, data_path: str, tokenizer, max_seq_len: int = 512, max_samples: int = None):
        self.max_seq_len = max_seq_len
        self.tokenizer = tokenizer
        self.samples = []

        # Load data
        data_file = Path(data_path)
        if data_file.exists():
            with open(data_file) as f:
                for i, line in enumerate(f):
                    if max_samples and i >= max_samples:
                        break
                    try:
                        item = json.loads(line)
                        text = item.get("response", item.get("text", ""))
                        if text:
                            self.samples.append(text)
                    except json.JSONDecodeError:
                        continue

        if not self.samples:
            # Fallback: synthetic data for testing
            self.samples = [
                "The quick brown fox jumps over the lazy dog.",
                "Machine learning models can learn patterns from data.",
                "Energy efficiency is important for sustainable computing.",
            ] * 100

        print(f"Loaded {len(self.samples)} samples from {data_path}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        text = self.samples[idx]
        tokens = self.tokenizer.encode(text)

        # Truncate or pad
        if len(tokens) > self.max_seq_len:
            tokens = tokens[:self.max_seq_len]
        else:
            tokens = tokens + [0] * (self.max_seq_len - len(tokens))

        return torch.tensor(tokens, dtype=torch.long)


class SimpleTokenizer:
    """Simple character-level tokenizer for testing."""

    def __init__(self, vocab_size: int = 32000):
        self.vocab_size = vocab_size

    def encode(self, text: str) -> List[int]:
        # Simple: use ord() mod vocab_size
        return [ord(c) % self.vocab_size for c in text]

    def decode(self, tokens: List[int]) -> str:
        return "".join(chr(t % 128) for t in tokens)


class TelemetrySimulator:
    """Simulates telemetry for training when real hardware unavailable."""

    def __init__(self, body_dim: int = 12):
        self.body_dim = body_dim

    def generate(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Generate synthetic telemetry."""
        # Realistic ranges:
        # [power_w, temp_c, gpu_util, clock_mhz, mem_util, fan_rpm, ...]
        telemetry = torch.zeros(batch_size, self.body_dim, device=device)

        telemetry[:, 0] = torch.rand(batch_size, device=device) * 100 + 20  # power: 20-120W
        telemetry[:, 1] = torch.rand(batch_size, device=device) * 40 + 40   # temp: 40-80C
        telemetry[:, 2] = torch.rand(batch_size, device=device) * 100       # gpu_util: 0-100%
        telemetry[:, 3] = torch.rand(batch_size, device=device) * 1500 + 500  # clock: 500-2000MHz
        telemetry[:, 4] = torch.rand(batch_size, device=device) * 100       # mem_util: 0-100%
        telemetry[:, 5] = torch.rand(batch_size, device=device) * 3000      # fan: 0-3000RPM
        # Rest are misc sensors
        telemetry[:, 6:] = torch.rand(batch_size, self.body_dim - 6, device=device)

        return telemetry


class EmbodiedTrainer:
    """Trainer for EmbodiedSLM with 3-step regimen."""

    def __init__(self, config: TrainingConfig):
        self.config = config
        self.device = torch.device(config.device)

        # Create output directory
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Create model
        if config.model_size == "125m":
            self.model = create_embodied_slm_125m()
        else:
            self.model = create_embodied_slm_30m()
        self.model = self.model.to(self.device)

        print(f"Model: {self.model.num_parameters / 1e6:.1f}M parameters")

        # Baseline model (for KL in step 3)
        self.baseline_model = None

        # Loss function
        self.loss_fn = EmbodiedLoss(
            lm_weight=config.lm_weight,
            body_weight=config.body_weight,
            kl_weight=config.kl_weight,
            kl_temperature=config.kl_temperature,
        )

        # Tokenizer
        self.tokenizer = SimpleTokenizer()

        # Telemetry simulator
        self.telemetry_sim = TelemetrySimulator()

        # Metrics
        self.metrics = {
            "step": [],
            "epoch": [],
            "loss": [],
            "lm_loss": [],
            "body_loss": [],
            "kl_loss": [],
        }

    def create_optimizer(self, lr: float) -> AdamW:
        """Create optimizer for current training phase."""
        params = [p for p in self.model.parameters() if p.requires_grad]
        return AdamW(params, lr=lr, weight_decay=self.config.weight_decay)

    def train_step1_baseline(self, epochs: int = None):
        """
        Step 1: Train language baseline (pure LM, no body signals).

        This establishes the language semantics baseline that we'll
        preserve in later steps.
        """
        print("\n" + "=" * 60)
        print("STEP 1: Language Baseline Training")
        print("=" * 60)

        epochs = epochs or self.config.step1_epochs

        # Set training phase
        self.model.set_training_phase("baseline")
        print(f"Trainable parameters: {self.model.num_trainable_parameters / 1e6:.2f}M")

        # Data
        dataset = TextDataset(
            self.config.data_path,
            self.tokenizer,
            self.config.max_seq_len,
        )
        dataloader = DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=0,
        )

        # Optimizer
        optimizer = self.create_optimizer(self.config.learning_rate)
        scheduler = CosineAnnealingLR(optimizer, T_max=epochs * len(dataloader))

        # Training loop
        self.model.train()
        global_step = 0

        for epoch in range(epochs):
            epoch_loss = 0
            epoch_samples = 0

            for batch in dataloader:
                batch = batch.to(self.device)

                # Forward (no telemetry in baseline)
                outputs = self.model(batch)
                logits = outputs["logits"]

                # Loss
                losses = self.loss_fn(logits, batch)
                loss = losses["lm"]  # Only LM loss in step 1

                # Backward
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip)
                optimizer.step()
                scheduler.step()

                epoch_loss += loss.item() * batch.size(0)
                epoch_samples += batch.size(0)
                global_step += 1

                if global_step % self.config.log_every == 0:
                    print(f"  Step {global_step}: loss={loss.item():.4f}")

            avg_loss = epoch_loss / epoch_samples
            print(f"Epoch {epoch + 1}/{epochs}: loss={avg_loss:.4f}")

            self.metrics["step"].append(1)
            self.metrics["epoch"].append(epoch + 1)
            self.metrics["loss"].append(avg_loss)
            self.metrics["lm_loss"].append(avg_loss)
            self.metrics["body_loss"].append(0)
            self.metrics["kl_loss"].append(0)

        # Save baseline checkpoint
        self._save_checkpoint("step1_baseline.pt")

        # Create frozen copy for KL in step 3
        self.baseline_model = create_embodied_slm_30m() if self.config.model_size == "30m" else create_embodied_slm_125m()
        self.baseline_model.load_state_dict(self.model.state_dict())
        self.baseline_model = self.baseline_model.to(self.device)
        self.baseline_model.eval()
        for param in self.baseline_model.parameters():
            param.requires_grad = False

        print("Step 1 complete. Baseline model frozen for KL constraint.")

    def train_step2_adapter(self, epochs: int = None):
        """
        Step 2: Adapter warmup (frozen trunk, train body components).

        Goal: Make the body channel meaningful without moving the language model.
        """
        print("\n" + "=" * 60)
        print("STEP 2: Adapter Warmup Training")
        print("=" * 60)

        epochs = epochs or self.config.step2_epochs

        # Set training phase (freezes trunk)
        self.model.set_training_phase("adapter")
        print(f"Trainable parameters: {self.model.num_trainable_parameters / 1e6:.2f}M")

        # Data
        dataset = TextDataset(
            self.config.data_path,
            self.tokenizer,
            self.config.max_seq_len,
        )
        dataloader = DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=0,
        )

        # Optimizer (only unfrozen params)
        optimizer = self.create_optimizer(self.config.learning_rate * 0.5)  # Lower LR for adapters

        # Training loop
        self.model.train()
        global_step = 0

        for epoch in range(epochs):
            epoch_loss = 0
            epoch_body_loss = 0
            epoch_samples = 0

            for batch in dataloader:
                batch = batch.to(self.device)

                # Generate telemetry
                telemetry = self.telemetry_sim.generate(batch.size(0), self.device)

                # Forward with telemetry
                outputs = self.model(batch, telemetry=telemetry, return_body_pred=True)
                logits = outputs["logits"]
                body_pred = outputs.get("body_pred")

                # Loss: body prediction (trunk is frozen, so LM loss doesn't contribute)
                losses = self.loss_fn(
                    logits, batch,
                    body_pred=body_pred,
                    body_target=telemetry,
                )

                # In step 2, focus on body loss (LM frozen anyway)
                loss = losses["body"]

                # Backward
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip)
                optimizer.step()

                epoch_loss += losses["total"].item() * batch.size(0)
                epoch_body_loss += losses["body"].item() * batch.size(0)
                epoch_samples += batch.size(0)
                global_step += 1

                if global_step % self.config.log_every == 0:
                    print(f"  Step {global_step}: body_loss={losses['body'].item():.4f}")

            avg_loss = epoch_loss / epoch_samples
            avg_body_loss = epoch_body_loss / epoch_samples
            print(f"Epoch {epoch + 1}/{epochs}: loss={avg_loss:.4f}, body_loss={avg_body_loss:.4f}")

            self.metrics["step"].append(2)
            self.metrics["epoch"].append(epoch + 1)
            self.metrics["loss"].append(avg_loss)
            self.metrics["lm_loss"].append(0)
            self.metrics["body_loss"].append(avg_body_loss)
            self.metrics["kl_loss"].append(0)

        # Save checkpoint
        self._save_checkpoint("step2_adapter.pt")
        print("Step 2 complete. Body channel is now meaningful.")

    def train_step3_semantics_lock(self, epochs: int = None):
        """
        Step 3: Full training with KL semantics lock.

        Train everything but add KL term to keep LM distribution
        close to the Step 1 baseline. This allows the model to
        "feel" body state while language remains stable.
        """
        print("\n" + "=" * 60)
        print("STEP 3: Semantics Lock Training (KL)")
        print("=" * 60)

        epochs = epochs or self.config.step3_epochs

        if self.baseline_model is None:
            print("WARNING: No baseline model. Loading from checkpoint...")
            checkpoint_path = self.output_dir / "step1_baseline.pt"
            if checkpoint_path.exists():
                self.baseline_model = create_embodied_slm_30m() if self.config.model_size == "30m" else create_embodied_slm_125m()
                self.baseline_model.load_state_dict(torch.load(checkpoint_path))
                self.baseline_model = self.baseline_model.to(self.device)
                self.baseline_model.eval()
                for param in self.baseline_model.parameters():
                    param.requires_grad = False
            else:
                print("ERROR: No baseline checkpoint found. Run step 1 first.")
                return

        # Set training phase (all params trainable)
        self.model.set_training_phase("full")
        print(f"Trainable parameters: {self.model.num_trainable_parameters / 1e6:.2f}M")

        # Data
        dataset = TextDataset(
            self.config.data_path,
            self.tokenizer,
            self.config.max_seq_len,
        )
        dataloader = DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=0,
        )

        # Optimizer
        optimizer = self.create_optimizer(self.config.learning_rate * 0.1)  # Lower LR for fine-tuning
        scheduler = CosineAnnealingLR(optimizer, T_max=epochs * len(dataloader))

        # Training loop
        self.model.train()
        global_step = 0

        for epoch in range(epochs):
            epoch_loss = 0
            epoch_lm_loss = 0
            epoch_body_loss = 0
            epoch_kl_loss = 0
            epoch_samples = 0

            for batch in dataloader:
                batch = batch.to(self.device)

                # Generate telemetry
                telemetry = self.telemetry_sim.generate(batch.size(0), self.device)

                # Forward with telemetry
                outputs = self.model(batch, telemetry=telemetry, return_body_pred=True)
                logits = outputs["logits"]
                body_pred = outputs.get("body_pred")

                # Get baseline logits (for KL)
                with torch.no_grad():
                    baseline_outputs = self.baseline_model(batch)
                    baseline_logits = baseline_outputs["logits"]

                # Loss: LM + body + KL
                losses = self.loss_fn(
                    logits, batch,
                    body_pred=body_pred,
                    body_target=telemetry,
                    baseline_logits=baseline_logits,
                )

                loss = losses["total"]

                # Backward
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip)
                optimizer.step()
                scheduler.step()

                epoch_loss += losses["total"].item() * batch.size(0)
                epoch_lm_loss += losses["lm"].item() * batch.size(0)
                epoch_body_loss += losses["body"].item() * batch.size(0)
                epoch_kl_loss += losses["kl"].item() * batch.size(0)
                epoch_samples += batch.size(0)
                global_step += 1

                if global_step % self.config.log_every == 0:
                    print(f"  Step {global_step}: total={losses['total'].item():.4f}, "
                          f"lm={losses['lm'].item():.4f}, body={losses['body'].item():.4f}, "
                          f"kl={losses['kl'].item():.4f}")

            avg_loss = epoch_loss / epoch_samples
            avg_lm = epoch_lm_loss / epoch_samples
            avg_body = epoch_body_loss / epoch_samples
            avg_kl = epoch_kl_loss / epoch_samples

            print(f"Epoch {epoch + 1}/{epochs}: loss={avg_loss:.4f}, "
                  f"lm={avg_lm:.4f}, body={avg_body:.4f}, kl={avg_kl:.4f}")

            self.metrics["step"].append(3)
            self.metrics["epoch"].append(epoch + 1)
            self.metrics["loss"].append(avg_loss)
            self.metrics["lm_loss"].append(avg_lm)
            self.metrics["body_loss"].append(avg_body)
            self.metrics["kl_loss"].append(avg_kl)

        # Save final checkpoint
        self._save_checkpoint("step3_final.pt")
        self._save_checkpoint("embodied_slm_final.pt")
        print("Step 3 complete. Model is now embodied with stable semantics.")

    def train_all_steps(self):
        """Run all three training steps."""
        self.train_step1_baseline()
        self.train_step2_adapter()
        self.train_step3_semantics_lock()

        # Save metrics
        self._save_metrics()
        print("\n" + "=" * 60)
        print("ALL STEPS COMPLETE")
        print("=" * 60)
        print(f"Final model saved to: {self.output_dir / 'embodied_slm_final.pt'}")

    def _save_checkpoint(self, filename: str):
        """Save model checkpoint."""
        path = self.output_dir / filename
        torch.save(self.model.state_dict(), path)
        print(f"Saved checkpoint: {path}")

    def _save_metrics(self):
        """Save training metrics."""
        path = self.output_dir / "training_metrics.json"
        with open(path, "w") as f:
            json.dump(self.metrics, f, indent=2)
        print(f"Saved metrics: {path}")

    def load_checkpoint(self, filename: str):
        """Load model checkpoint."""
        path = self.output_dir / filename
        if path.exists():
            self.model.load_state_dict(torch.load(path))
            print(f"Loaded checkpoint: {path}")
        else:
            print(f"Checkpoint not found: {path}")


def main():
    parser = argparse.ArgumentParser(description="EmbodiedSLM 3-Step Training")
    parser.add_argument("--step", type=int, choices=[1, 2, 3], help="Run specific step")
    parser.add_argument("--all", action="store_true", help="Run all steps")
    parser.add_argument("--epochs", type=int, help="Override epochs for this step")
    parser.add_argument("--model-size", type=str, default="30m", choices=["30m", "125m"])
    parser.add_argument("--data", type=str, default="data/ouroboros/train_responses.jsonl")
    parser.add_argument("--output-dir", type=str, default="results/z116_embodied")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    args = parser.parse_args()

    # Create config
    config = TrainingConfig(
        model_size=args.model_size,
        data_path=args.data,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        learning_rate=args.lr,
    )

    # HSA override for AMD
    os.environ["HSA_OVERRIDE_GFX_VERSION"] = "11.0.0"

    # Create trainer
    trainer = EmbodiedTrainer(config)

    if args.all:
        trainer.train_all_steps()
    elif args.step == 1:
        trainer.train_step1_baseline(args.epochs)
    elif args.step == 2:
        # Need to load step 1 checkpoint first
        trainer.load_checkpoint("step1_baseline.pt")
        trainer.train_step2_adapter(args.epochs)
    elif args.step == 3:
        # Need to load step 2 checkpoint
        trainer.load_checkpoint("step2_adapter.pt")
        trainer.train_step3_semantics_lock(args.epochs)
    else:
        print("Specify --step [1|2|3] or --all")


if __name__ == "__main__":
    main()
