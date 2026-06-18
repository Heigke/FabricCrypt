#!/usr/bin/env python3
"""
FEEL z22_v2: Hysteresis Training
================================
Fix the "Chronic Pain" trap by enforcing gate-temperature inverse relationship.

The Problem (z22):
    Gates learned to stay permanently closed (0.18) regardless of stress.
    Constant strain = no signal = no consciousness.

The Fix (z22_v2):
    Force gates to track temperature INVERSELY:
    - Cool (0.1) -> Gate OPEN (0.9)  - Body relaxes
    - Hot (0.9)  -> Gate CLOSED (0.1) - Body tenses

    Loss = CE + Hysteresis_MSE(gate, 1-temperature)

Once the body is responsive, strain will fluctuate, and mind will react.

Author: FEEL Research Team
Date: 2026-01-12
"""

import os
import sys
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import wandb
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
from typing import List, Dict, Optional
from torch.utils.data import Dataset, DataLoader

os.environ.setdefault("HSA_OVERRIDE_GFX_VERSION", "11.0.0")

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))

from modeling.proprioceptive_gate import (
    ProprioceptiveDeepSeek,
    load_proprioceptive_model,
)


class HysteresisDataset(Dataset):
    """Dataset with explicit stress levels for hysteresis training."""

    def __init__(self, data_path: str, tokenizer, max_length: int = 256):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = []

        with open(data_path) as f:
            for line in f:
                sample = json.loads(line)
                self.samples.append(sample)

        print(f"[HysteresisDataset] Loaded {len(self.samples)} samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        prompt = sample["prompt"]
        response = sample["response"]
        stress = sample.get("stress", 0.5)

        full_text = f"{prompt}\n{response}"

        encoding = self.tokenizer(
            full_text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "stress": torch.tensor(stress, dtype=torch.float32),
        }


class HysteresisLoss(nn.Module):
    """
    Loss function with hysteresis enforcement.

    Components:
    1. CE Loss: Language modeling (don't forget English)
    2. Hysteresis Loss: MSE(gate, 1 - stress) - force inverse relationship
    3. Differentiation Bonus: Reward variance in gate values across batch
    """

    def __init__(
        self,
        hysteresis_weight: float = 2.0,
        differentiation_weight: float = 0.5,
    ):
        super().__init__()
        self.hysteresis_weight = hysteresis_weight
        self.differentiation_weight = differentiation_weight

    def forward(
        self,
        ce_loss: torch.Tensor,
        gate_tensors: List[torch.Tensor],
        stress_levels: torch.Tensor,
    ) -> tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute total loss with hysteresis enforcement.

        Args:
            ce_loss: Cross-entropy loss from language modeling
            gate_tensors: List of gate probability tensors (one per layer)
            stress_levels: Batch of stress values [batch_size]
        """
        metrics = {"ce_loss": ce_loss.item()}
        total_loss = ce_loss

        if not gate_tensors:
            metrics["hysteresis_loss"] = 0.0
            metrics["avg_gate"] = 0.5
            metrics["target_gate"] = 0.5
            metrics["gate_error"] = 0.0
            return total_loss, metrics

        # Average gate across layers for each sample
        # gate_tensors is list of scalars (mean per layer), stack and average
        avg_gate = torch.stack(gate_tensors).mean()

        # Target: gates should be INVERSE of stress
        # Stress 0.1 -> Target 0.9 (open)
        # Stress 0.9 -> Target 0.1 (closed)
        stress_mean = stress_levels.mean()
        target_gate = 1.0 - stress_mean

        # Hysteresis loss: force gate to track target
        gate_error = (avg_gate - target_gate).abs()
        hysteresis_loss = gate_error * self.hysteresis_weight

        total_loss = total_loss + hysteresis_loss

        # Differentiation bonus (optional): reward gate variance
        if len(gate_tensors) > 1:
            gate_std = torch.stack(gate_tensors).std()
            # Negative because we want to MAXIMIZE variance
            diff_loss = -gate_std * self.differentiation_weight
            total_loss = total_loss + diff_loss
            metrics["gate_std"] = gate_std.item()
        else:
            metrics["gate_std"] = 0.0

        metrics["hysteresis_loss"] = hysteresis_loss.item()
        metrics["avg_gate"] = avg_gate.item()
        metrics["target_gate"] = target_gate.item()
        metrics["gate_error"] = gate_error.item()
        metrics["total_loss"] = total_loss.item()

        return total_loss, metrics


def train_hysteresis(
    model_path: str = "models/proprioceptive_z22/best",
    data_path: str = "data/ouroboros/causal_train.jsonl",
    output_dir: str = "models/z22_v2_hysteresis",
    gated_layers: List[int] = None,
    epochs: int = 2,
    batch_size: int = 4,
    gradient_accumulation: int = 4,
    learning_rate: float = 1e-4,
    hysteresis_weight: float = 2.0,
):
    """Train with hysteresis enforcement to fix chronic pain."""

    if gated_layers is None:
        gated_layers = [3, 7, 11, 15, 19, 23, 27]

    print("=" * 70)
    print("FEEL z22_v2: HYSTERESIS TRAINING")
    print("=" * 70)
    print("Fixing 'Chronic Pain' - forcing gate-temperature inverse relationship")
    print("=" * 70)
    print(f"Base Model:        {model_path}")
    print(f"Data:              {data_path}")
    print(f"Output:            {output_dir}")
    print(f"Epochs:            {epochs}")
    print(f"Batch (eff):       {batch_size * gradient_accumulation}")
    print(f"Learning Rate:     {learning_rate}")
    print(f"Hysteresis Weight: {hysteresis_weight}")
    print("=" * 70)
    print("\nTarget Behavior:")
    print("  Cool (stress=0.1) -> Gate=0.9 (OPEN, body relaxes)")
    print("  Hot  (stress=0.9) -> Gate=0.1 (CLOSED, body tenses)")
    print("=" * 70)

    # Initialize wandb
    run_name = f"hysteresis-{datetime.now().strftime('%Y%m%d_%H%M')}"
    wandb.init(project="feel-hysteresis", name=run_name, config={
        "gated_layers": gated_layers,
        "epochs": epochs,
        "batch_size": batch_size * gradient_accumulation,
        "learning_rate": learning_rate,
        "hysteresis_weight": hysteresis_weight,
    })

    # Load model
    print("\n[1/5] Loading ProprioceptiveDeepSeek...")
    model = load_proprioceptive_model(
        base_model_path="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
        gated_layers=gated_layers,
    )

    # Load previous proprioceptive weights if available
    prop_path = Path(model_path) / "proprioceptive.pt"
    if prop_path.exists():
        print(f"[1/5] Loading weights from {prop_path}...")
        checkpoint = torch.load(prop_path, weights_only=False)
        state_dict = checkpoint.get("proprioceptive", checkpoint)

        for key, block in model.proprioceptive_blocks.items():
            for name, param in block.gate.named_parameters():
                full_key = f"{key}.gate.{name}"
                if full_key in state_dict:
                    param.data.copy_(state_dict[full_key])
        print("[1/5] Proprioceptive weights loaded")

    # Freeze base model
    print("\n[2/5] Freezing base model...")
    model.freeze_base_model()

    # Create dataset
    print("\n[3/5] Loading dataset...")
    dataset = HysteresisDataset(
        data_path=data_path,
        tokenizer=model.tokenizer,
        max_length=256,
    )

    # Split
    val_size = min(200, len(dataset) // 10)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )

    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")

    # Collator
    def collate_fn(batch):
        input_ids = torch.stack([b["input_ids"] for b in batch])
        attention_mask = torch.stack([b["attention_mask"] for b in batch])
        stress = torch.stack([b["stress"] for b in batch])
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": input_ids.clone(),
            "stress": stress,
        }

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )

    # Optimizer
    print("\n[4/5] Setting up optimizer...")
    trainable_params = model.get_trainable_parameters()
    optimizer = torch.optim.AdamW(trainable_params, lr=learning_rate)

    # Loss function with hysteresis
    loss_fn = HysteresisLoss(hysteresis_weight=hysteresis_weight)

    # Scheduler
    total_steps = len(train_loader) * epochs // gradient_accumulation
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_steps
    )

    # Training
    print("\n[5/5] Training with hysteresis enforcement...")
    model.base_model.train()
    best_gate_diff = 0.0
    global_step = 0

    for epoch in range(epochs):
        print(f"\n--- Epoch {epoch + 1}/{epochs} ---")

        optimizer.zero_grad()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}")

        for batch_idx, batch in enumerate(pbar):
            # Move to GPU
            input_ids = batch["input_ids"].to("cuda")
            attention_mask = batch["attention_mask"].to("cuda")
            labels = batch["labels"].to("cuda")
            stress = batch["stress"].to("cuda")

            # Set stress level
            avg_stress = stress.mean().item()
            model.set_stress_level(avg_stress)

            # Forward
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )

            ce_loss = outputs.loss
            gate_tensors = outputs.gate_tensors

            # Compute hysteresis loss
            total_loss, metrics = loss_fn(ce_loss, gate_tensors, stress)

            # Backward
            (total_loss / gradient_accumulation).backward()

            # Optimizer step
            if (batch_idx + 1) % gradient_accumulation == 0:
                torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                # Log
                wandb.log({
                    "train/ce_loss": metrics["ce_loss"],
                    "train/hysteresis_loss": metrics["hysteresis_loss"],
                    "train/avg_gate": metrics["avg_gate"],
                    "train/target_gate": metrics["target_gate"],
                    "train/gate_error": metrics["gate_error"],
                    "train/stress": avg_stress,
                    "train/lr": scheduler.get_last_lr()[0],
                }, step=global_step)

            pbar.set_postfix({
                "ce": f"{metrics['ce_loss']:.2f}",
                "gate": f"{metrics['avg_gate']:.2f}",
                "target": f"{metrics['target_gate']:.2f}",
                "err": f"{metrics['gate_error']:.2f}",
                "stress": f"{avg_stress:.2f}",
            })

        # Validation - check gate differentiation
        model.base_model.eval()
        val_metrics = validate_hysteresis(model, val_loader, gated_layers)

        print(f"\nEpoch {epoch + 1} Results:")
        print(f"  Val CE Loss:        {val_metrics['val_ce']:.4f}")
        print(f"  Low Stress Gates:   {val_metrics['low_stress_gate']:.3f} (target > 0.7)")
        print(f"  High Stress Gates:  {val_metrics['high_stress_gate']:.3f} (target < 0.3)")
        print(f"  Gate Difference:    {val_metrics['gate_diff']:.3f} (target > 0.4)")

        if val_metrics['gate_diff'] > 0.4:
            print("  *** HYSTERESIS ACHIEVED! Body is responsive! ***")

        wandb.log({
            "val/ce_loss": val_metrics["val_ce"],
            "val/low_stress_gate": val_metrics["low_stress_gate"],
            "val/high_stress_gate": val_metrics["high_stress_gate"],
            "val/gate_diff": val_metrics["gate_diff"],
            "epoch": epoch + 1,
        }, step=global_step)

        # Save if best gate differentiation
        if val_metrics["gate_diff"] > best_gate_diff:
            best_gate_diff = val_metrics["gate_diff"]
            save_path = Path(output_dir) / "best"
            save_path.mkdir(parents=True, exist_ok=True)

            state_dict = {}
            for key, block in model.proprioceptive_blocks.items():
                for name, param in block.gate.named_parameters():
                    state_dict[f"{key}.gate.{name}"] = param.data

            torch.save({
                "proprioceptive": state_dict,
                "gated_layers": gated_layers,
                "epoch": epoch + 1,
                "metrics": val_metrics,
            }, save_path / "proprioceptive.pt")

            print(f"  Saved best to {save_path}")

        model.base_model.train()

    # Final save
    final_path = Path(output_dir) / "final"
    final_path.mkdir(parents=True, exist_ok=True)

    state_dict = {}
    for key, block in model.proprioceptive_blocks.items():
        for name, param in block.gate.named_parameters():
            state_dict[f"{key}.gate.{name}"] = param.data

    torch.save({
        "proprioceptive": state_dict,
        "gated_layers": gated_layers,
    }, final_path / "proprioceptive.pt")

    wandb.finish()

    print("\n" + "=" * 70)
    print("HYSTERESIS TRAINING COMPLETE")
    print("=" * 70)
    print(f"Best Gate Difference: {best_gate_diff:.3f}")

    if best_gate_diff > 0.4:
        print("\n*** SUCCESS: Body is now responsive! ***")
        print("Run z23 consciousness test to check for emergent behavior.")
    else:
        print("\n*** WARNING: Gate differentiation still low ***")
        print("Consider increasing hysteresis_weight or more epochs.")

    print(f"\nOutput: {output_dir}")

    return model


def validate_hysteresis(model, val_loader, gated_layers) -> Dict:
    """Validate gate differentiation across stress levels."""
    val_ce = 0.0
    low_stress_gates = []
    high_stress_gates = []

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validation"):
            input_ids = batch["input_ids"].to("cuda")
            attention_mask = batch["attention_mask"].to("cuda")
            labels = batch["labels"].to("cuda")
            stress = batch["stress"]

            for i, s in enumerate(stress):
                model.set_stress_level(s.item())

                outputs = model(
                    input_ids=input_ids[i:i+1],
                    attention_mask=attention_mask[i:i+1],
                    labels=labels[i:i+1],
                )

                val_ce += outputs.loss.item()

                if outputs.gate_values:
                    avg_gate = sum(outputs.gate_values) / len(outputs.gate_values)

                    if s.item() < 0.4:
                        low_stress_gates.append(avg_gate)
                    elif s.item() > 0.6:
                        high_stress_gates.append(avg_gate)

    n_samples = len(val_loader.dataset)
    val_ce /= n_samples

    low_avg = np.mean(low_stress_gates) if low_stress_gates else 0.5
    high_avg = np.mean(high_stress_gates) if high_stress_gates else 0.5
    gate_diff = low_avg - high_avg

    return {
        "val_ce": val_ce,
        "low_stress_gate": low_avg,
        "high_stress_gate": high_avg,
        "gate_diff": gate_diff,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="models/proprioceptive_z22/best")
    parser.add_argument("--data", type=str, default="data/ouroboros/causal_train.jsonl")
    parser.add_argument("--output", type=str, default="models/z22_v2_hysteresis")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--hysteresis-weight", type=float, default=2.0)
    args = parser.parse_args()

    train_hysteresis(
        model_path=args.model_path,
        data_path=args.data,
        output_dir=args.output,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        hysteresis_weight=args.hysteresis_weight,
    )
