#!/usr/bin/env python3
"""
z151: Training Script for Early Exit Embodied Compute

Multi-objective training for early exit with:
- L_quality: CE + KL(early || final)
- L_exit: encourage early exits
- L_energy: homeostatic energy target

Mixed data: 70% TinyStories, 30% FineWeb-Edu
Curriculum: gradually enable earlier exits

Usage:
    python z151_train_early_exit.py --model deepseek-coder-1.3b --epochs 3
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional, Iterator, Tuple
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, IterableDataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
from tqdm import tqdm

# HuggingFace
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

# Local
from src.modeling.early_exit_transformer import EarlyExitTransformer, EarlyExitTrainer
from src.modeling.z24_sensor_hub import SimulatedSensorHub
from src.energy_harness.nvml_energy import create_energy_meter
from src.memory.control_memory import ControlMemory, ControlOutcome


@dataclass
class TrainingConfig:
    """Training configuration"""
    # Model
    model_name: str = "deepseek-ai/deepseek-coder-1.3b-instruct"
    num_layers: int = 24

    # Data
    tinystories_ratio: float = 0.7
    fineweb_ratio: float = 0.3
    max_seq_len: int = 512
    batch_size: int = 4

    # Training
    num_epochs: int = 3
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    warmup_steps: int = 100
    gradient_accumulation: int = 4
    max_grad_norm: float = 1.0

    # Early exit
    exit_layers: List[int] = None
    target_exit_layer: float = 12.0
    lambda_exit: float = 0.1
    lambda_energy: float = 0.01
    lambda_kl: float = 0.1

    # Curriculum
    curriculum_steps: int = 1000
    curriculum_min_exit: int = 16  # Start allowing exits from layer 16

    # Logging
    log_interval: int = 10
    eval_interval: int = 100
    save_interval: int = 500
    output_dir: str = "checkpoints/z151_early_exit"

    def __post_init__(self):
        if self.exit_layers is None:
            self.exit_layers = [4, 8, 12, 16, 20, 24]


class MixedDataset(IterableDataset):
    """
    Mixed dataset combining TinyStories and FineWeb-Edu.

    Difficulty stratification:
    - Easy: short, simple sentences (low perplexity)
    - Medium: standard complexity
    - Hard: long, complex content (high perplexity)
    """

    def __init__(
        self,
        tokenizer,
        max_seq_len: int = 512,
        tinystories_ratio: float = 0.7,
        seed: int = 42
    ):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.tinystories_ratio = tinystories_ratio
        self.seed = seed

        self.tinystories = None
        self.fineweb = None

    def _load_datasets(self):
        """Lazy load datasets"""
        if self.tinystories is None:
            print("Loading TinyStories...")
            self.tinystories = load_dataset(
                "roneneldan/TinyStories",
                split="train",
                streaming=True
            )

        if self.fineweb is None:
            print("Loading FineWeb-Edu (sample)...")
            try:
                self.fineweb = load_dataset(
                    "HuggingFaceFW/fineweb-edu",
                    name="sample-10BT",
                    split="train",
                    streaming=True
                )
            except Exception as e:
                print(f"FineWeb load failed: {e}, using TinyStories only")
                self.fineweb = None

    def _tokenize(self, text: str) -> Dict[str, torch.Tensor]:
        """Tokenize and prepare for training"""
        # Truncate long text
        if len(text) > self.max_seq_len * 4:  # Rough char estimate
            text = text[:self.max_seq_len * 4]

        encoded = self.tokenizer(
            text,
            max_length=self.max_seq_len,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )

        input_ids = encoded["input_ids"].squeeze(0)
        attention_mask = encoded["attention_mask"].squeeze(0)

        # Labels = input_ids shifted (for causal LM)
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100  # Ignore padding

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }

    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        self._load_datasets()

        rng = np.random.RandomState(self.seed)
        ts_iter = iter(self.tinystories)
        fw_iter = iter(self.fineweb) if self.fineweb else None

        while True:
            # Choose dataset based on ratio
            use_tinystories = rng.random() < self.tinystories_ratio or fw_iter is None

            try:
                if use_tinystories:
                    item = next(ts_iter)
                    text = item.get("text", "")
                else:
                    item = next(fw_iter)
                    text = item.get("text", "")

                if len(text.strip()) < 20:  # Skip very short
                    continue

                yield self._tokenize(text)

            except StopIteration:
                # Restart iterators
                ts_iter = iter(self.tinystories)
                if self.fineweb:
                    fw_iter = iter(self.fineweb)


class DifficultyEstimator:
    """
    Estimates prompt difficulty based on various heuristics.
    Used for stratified evaluation.
    """

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def estimate(self, text: str) -> str:
        """
        Estimate difficulty: easy, medium, hard

        Heuristics:
        - Length
        - Vocabulary complexity (rare words)
        - Sentence structure
        """
        # Length-based
        tokens = self.tokenizer.encode(text)
        num_tokens = len(tokens)

        if num_tokens < 50:
            return "easy"
        elif num_tokens < 200:
            return "medium"
        else:
            return "hard"


class EarlyExitTrainingLoop:
    """
    Complete training loop for early exit models.
    """

    def __init__(self, config: TrainingConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = None
        self.tokenizer = None
        self.optimizer = None
        self.scheduler = None
        self.energy_meter = None
        self.control_memory = None

        self.global_step = 0
        self.metrics_history = []

    def setup(self):
        """Initialize everything"""
        print(f"\n{'='*60}")
        print("Training Setup")
        print(f"{'='*60}")

        # Tokenizer
        print(f"\nLoading tokenizer: {self.config.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Base model
        print(f"Loading base model...")
        base_model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            torch_dtype=torch.float16,
        ).to(self.device)

        # Determine num_layers
        if hasattr(base_model.config, 'num_hidden_layers'):
            self.config.num_layers = base_model.config.num_hidden_layers

        # Wrap with early exit
        print(f"Creating early exit wrapper (layers: {self.config.exit_layers})...")
        sensor_hub = SimulatedSensorHub()
        self.model = EarlyExitTransformer(
            base_model,
            sensor_hub=sensor_hub,
            exit_layers=self.config.exit_layers,
            target_exit_layer=self.config.target_exit_layer,
            train_base_model=False  # Freeze base, train exit heads
        )

        # Optimizer (only trainable params)
        trainable_params = self.model.get_trainable_parameters()
        print(f"Trainable parameters: {sum(p.numel() for p in trainable_params):,}")

        self.optimizer = AdamW(
            trainable_params,
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )

        # Scheduler
        total_steps = self.config.num_epochs * 10000  # Estimate
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=total_steps,
            eta_min=self.config.learning_rate * 0.1
        )

        # Energy meter (optional)
        try:
            self.energy_meter = create_energy_meter()
            print(f"Energy meter: {type(self.energy_meter).__name__}")
        except:
            self.energy_meter = None
            print("Energy meter: disabled")

        # Control memory
        self.control_memory = ControlMemory(num_layers=self.config.num_layers)

        print("\nSetup complete!")

    def create_dataloader(self) -> DataLoader:
        """Create mixed dataloader"""
        dataset = MixedDataset(
            self.tokenizer,
            max_seq_len=self.config.max_seq_len,
            tinystories_ratio=self.config.tinystories_ratio
        )

        return DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            num_workers=0  # Streaming doesn't support multiprocess
        )

    def get_curriculum_min_exit(self) -> int:
        """Get minimum allowed exit layer based on curriculum"""
        if self.global_step >= self.config.curriculum_steps:
            return min(self.config.exit_layers)  # Allow all exits

        # Linear interpolation from curriculum_min_exit to first exit layer
        progress = self.global_step / self.config.curriculum_steps
        min_exit = int(
            self.config.curriculum_min_exit * (1 - progress) +
            min(self.config.exit_layers) * progress
        )
        return min_exit

    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Single training step"""
        self.model.train_mode(train_base=False)
        self.optimizer.zero_grad()

        input_ids = batch["input_ids"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)
        labels = batch["labels"].to(self.device)

        # Get curriculum constraint
        min_exit = self.get_curriculum_min_exit()

        # Forward pass
        output = self.model(
            input_ids,
            attention_mask=attention_mask,
            labels=labels
        )

        # Backward
        loss = output.loss
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(
            self.model.get_trainable_parameters(),
            self.config.max_grad_norm
        )

        # Optimizer step
        self.optimizer.step()
        self.scheduler.step()

        self.global_step += 1

        # Collect metrics
        metrics = {
            'loss': loss.item(),
            'exit_layer': output.exit_layer,
            'flops_saved_pct': output.flops_saved / (12 * self.model.hidden_dim ** 2 * input_ids.size(1) * self.model.num_layers) * 100,
            'lr': self.scheduler.get_last_lr()[0],
            'curriculum_min_exit': min_exit
        }

        if output.loss_breakdown:
            metrics.update(output.loss_breakdown)

        return metrics

    def train_epoch(self, dataloader: DataLoader, epoch: int) -> Dict[str, float]:
        """Train for one epoch"""
        epoch_metrics = []

        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}")
        for batch_idx, batch in enumerate(pbar):
            metrics = self.train_step(batch)
            epoch_metrics.append(metrics)

            # Update progress bar
            pbar.set_postfix({
                'loss': f"{metrics['loss']:.4f}",
                'exit': metrics['exit_layer'],
                'flops_saved': f"{metrics['flops_saved_pct']:.1f}%"
            })

            # Logging
            if self.global_step % self.config.log_interval == 0:
                self.log_metrics(metrics)

            # Save checkpoint
            if self.global_step % self.config.save_interval == 0:
                self.save_checkpoint()

            # Stop after reasonable number of batches per epoch
            if batch_idx >= 10000:
                break

        # Aggregate epoch metrics
        avg_metrics = {}
        for key in epoch_metrics[0].keys():
            values = [m[key] for m in epoch_metrics if isinstance(m.get(key), (int, float))]
            if values:
                avg_metrics[f"epoch_{key}"] = np.mean(values)

        return avg_metrics

    def log_metrics(self, metrics: Dict[str, float]):
        """Log metrics"""
        self.metrics_history.append({
            'step': self.global_step,
            'timestamp': datetime.now().isoformat(),
            **metrics
        })

    def save_checkpoint(self):
        """Save model checkpoint"""
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        checkpoint_path = output_dir / f"checkpoint_step{self.global_step}.pt"

        torch.save({
            'global_step': self.global_step,
            'model_state_dict': {
                k: v for k, v in self.model.state_dict().items()
                if 'exit_head' in k or 'exit_decision' in k
            },
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'config': asdict(self.config),
            'exit_stats': asdict(self.model.get_exit_stats())
        }, checkpoint_path)

        print(f"\nCheckpoint saved: {checkpoint_path}")

    def train(self):
        """Full training loop"""
        print(f"\n{'='*60}")
        print("Starting Training")
        print(f"{'='*60}")
        print(f"Epochs: {self.config.num_epochs}")
        print(f"Batch size: {self.config.batch_size}")
        print(f"Learning rate: {self.config.learning_rate}")
        print(f"Target exit layer: {self.config.target_exit_layer}")

        dataloader = self.create_dataloader()

        for epoch in range(self.config.num_epochs):
            print(f"\n--- Epoch {epoch + 1}/{self.config.num_epochs} ---")

            epoch_metrics = self.train_epoch(dataloader, epoch)

            print(f"\nEpoch {epoch + 1} Summary:")
            for key, value in epoch_metrics.items():
                print(f"  {key}: {value:.4f}")

            # Save end-of-epoch checkpoint
            self.save_checkpoint()

        # Final save
        self.save_final()

        print(f"\n{'='*60}")
        print("Training Complete!")
        print(f"{'='*60}")

    def save_final(self):
        """Save final model and metrics"""
        output_dir = Path(self.config.output_dir)

        # Save metrics history
        metrics_path = output_dir / "metrics_history.json"
        with open(metrics_path, 'w') as f:
            json.dump(self.metrics_history, f, indent=2)

        # Save control memory
        memory_path = output_dir / "control_memory.json"
        self.control_memory.save(memory_path)

        # Save final model
        final_path = output_dir / "final_model.pt"
        torch.save({
            'global_step': self.global_step,
            'model_state_dict': {
                k: v for k, v in self.model.state_dict().items()
                if 'exit_head' in k or 'exit_decision' in k
            },
            'config': asdict(self.config),
            'exit_stats': asdict(self.model.get_exit_stats())
        }, final_path)

        print(f"\nFinal model saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Train early exit model")
    parser.add_argument("--model", type=str, default="deepseek-ai/deepseek-coder-1.3b-instruct")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--target-exit", type=float, default=12.0)
    parser.add_argument("--output-dir", type=str, default="checkpoints/z151_early_exit")
    args = parser.parse_args()

    config = TrainingConfig(
        model_name=args.model,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        target_exit_layer=args.target_exit,
        output_dir=args.output_dir
    )

    trainer = EarlyExitTrainingLoop(config)
    trainer.setup()
    trainer.train()


if __name__ == "__main__":
    main()
