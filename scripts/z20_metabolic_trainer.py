#!/usr/bin/env python3
"""
FEEL z20: Metabolic Gate Training
=================================
Train the gate networks to learn WHEN to skip layers.

Key Strategy:
1. FREEZE the base LLM (all 7B parameters)
2. TRAIN ONLY the gate networks (~50K parameters)
3. Loss = CE + Metabolic Penalty (stress * usage)

The model learns: High stress -> close gates -> skip layers -> save FLOPs

Author: FEEL Research Team
Date: 2026-01-12
"""

import os
import sys
import json
import torch
import torch.nn as nn
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

from modeling.metabolic_gate import (
    MetabolicDeepSeek,
    MetabolicLoss,
    load_metabolic_model,
)


class MetabolicDataset(Dataset):
    """Dataset with stress-aware samples for gate training."""

    def __init__(
        self,
        data_path: str,
        tokenizer,
        max_length: int = 256,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = []

        with open(data_path) as f:
            for line in f:
                sample = json.loads(line)
                self.samples.append(sample)

        print(f"[MetabolicDataset] Loaded {len(self.samples)} samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        prompt = sample["prompt"]
        response = sample["response"]
        stress = sample.get("stress", 0.5)

        # Shorter sequences for faster gate training
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


class MetabolicCollator:
    """Collate batches with sensor tensors."""

    def __call__(self, batch: List[Dict]) -> Dict:
        input_ids = torch.stack([b["input_ids"] for b in batch])
        attention_mask = torch.stack([b["attention_mask"] for b in batch])
        stress = torch.stack([b["stress"] for b in batch])

        # Create sensor tensor
        sensor_tensor = torch.zeros(len(batch), 3)
        for i, s in enumerate(stress):
            sensor_tensor[i] = torch.tensor([
                s.item(),           # temp
                1.0 - s.item(),     # power (inverse)
                0.5,                # entropy
            ])

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": input_ids.clone(),
            "sensor_tensor": sensor_tensor,
            "stress": stress,
        }


def train_metabolic_gates(
    data_path: str = "data/ouroboros/causal_train.jsonl",
    output_dir: str = "models/metabolic_z20",
    gated_layers: List[int] = None,
    epochs: int = 3,
    batch_size: int = 4,
    gradient_accumulation: int = 4,
    learning_rate: float = 1e-3,  # Higher LR for small gate networks
    metabolic_weight: float = 0.5,
):
    """Train metabolic gate networks."""

    # Default: Gate every 4th layer
    if gated_layers is None:
        gated_layers = [3, 7, 11, 15, 19, 23, 27]

    print("=" * 70)
    print("FEEL z20: METABOLIC GATE TRAINING")
    print("=" * 70)
    print(f"Data:           {data_path}")
    print(f"Output:         {output_dir}")
    print(f"Gated Layers:   {gated_layers}")
    print(f"Epochs:         {epochs}")
    print(f"Batch (eff):    {batch_size * gradient_accumulation}")
    print(f"Learning Rate:  {learning_rate}")
    print(f"Metabolic λ:    {metabolic_weight}")
    print("=" * 70)

    # Initialize wandb
    run_name = f"metabolic-{datetime.now().strftime('%Y%m%d_%H%M')}"
    wandb.init(project="feel-metabolic", name=run_name, config={
        "gated_layers": gated_layers,
        "epochs": epochs,
        "batch_size": batch_size * gradient_accumulation,
        "learning_rate": learning_rate,
        "metabolic_weight": metabolic_weight,
    })

    # Load model
    print("\n[1/5] Loading MetabolicDeepSeek...")
    model = load_metabolic_model(
        base_model_path="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
        gated_layers=gated_layers,
    )

    # CRITICAL: Freeze base model, train only gates
    print("\n[2/5] Freezing base model...")
    model.freeze_base_model()

    # Create dataset
    print("\n[3/5] Loading dataset...")
    dataset = MetabolicDataset(
        data_path=data_path,
        tokenizer=model.tokenizer,
        max_length=256,  # Shorter for faster training
    )

    # Split
    val_size = min(200, len(dataset) // 10)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )

    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")

    # Dataloaders
    collator = MetabolicCollator()
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collator,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collator,
    )

    # Optimizer - only gate parameters
    print("\n[4/5] Setting up optimizer...")
    gate_params = model.get_trainable_parameters()
    optimizer = torch.optim.AdamW(gate_params, lr=learning_rate)

    # Loss function
    loss_fn = MetabolicLoss(metabolic_weight=metabolic_weight)

    # Scheduler
    total_steps = len(train_loader) * epochs // gradient_accumulation
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_steps
    )

    # Training
    print("\n[5/5] Training gates...")
    model.base_model.train()  # Keep in train mode for dropout etc
    best_val_loss = float("inf")
    global_step = 0

    for epoch in range(epochs):
        print(f"\n--- Epoch {epoch + 1}/{epochs} ---")

        epoch_ce_loss = 0.0
        epoch_metabolic = 0.0
        epoch_gates = []
        optimizer.zero_grad()

        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}")
        for batch_idx, batch in enumerate(pbar):
            # Move to GPU
            input_ids = batch["input_ids"].to("cuda")
            attention_mask = batch["attention_mask"].to("cuda")
            labels = batch["labels"].to("cuda")
            stress = batch["stress"].to("cuda")

            # Set sensor state (batch average for now)
            avg_stress = stress.mean().item()
            model.set_stress_level(avg_stress)

            # Forward
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )

            ce_loss = outputs.loss
            gate_values = outputs.gate_values if hasattr(outputs, 'gate_values') else []
            gate_tensors = outputs.gate_tensors if hasattr(outputs, 'gate_tensors') else []

            # Compute metabolic loss (gate_tensors for gradient flow)
            total_loss, metrics = loss_fn(ce_loss, gate_values, stress, gate_tensors=gate_tensors)

            # Backward
            (total_loss / gradient_accumulation).backward()

            epoch_ce_loss += metrics["ce_loss"]
            epoch_metabolic += metrics["metabolic_penalty"]
            if gate_values:
                epoch_gates.append(metrics["avg_gate"])

            # Optimizer step
            if (batch_idx + 1) % gradient_accumulation == 0:
                torch.nn.utils.clip_grad_norm_(gate_params, 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                # Log
                wandb.log({
                    "train/ce_loss": metrics["ce_loss"],
                    "train/metabolic_penalty": metrics["metabolic_penalty"],
                    "train/avg_gate": metrics["avg_gate"],
                    "train/stress": avg_stress,
                    "train/lr": scheduler.get_last_lr()[0],
                }, step=global_step)

            pbar.set_postfix({
                "ce": f"{metrics['ce_loss']:.3f}",
                "meta": f"{metrics['metabolic_penalty']:.3f}",
                "gate": f"{metrics['avg_gate']:.2f}",
                "stress": f"{avg_stress:.2f}",
            })

        # Validation
        model.base_model.eval()
        val_ce = 0.0
        val_gates_high_stress = []
        val_gates_low_stress = []

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

                    if hasattr(outputs, 'gate_values') and outputs.gate_values:
                        avg_gate = sum(outputs.gate_values) / len(outputs.gate_values)
                        if s.item() > 0.6:
                            val_gates_high_stress.append(avg_gate)
                        elif s.item() < 0.4:
                            val_gates_low_stress.append(avg_gate)

        val_ce /= len(val_dataset)

        # Gate behavior analysis
        high_stress_gate = np.mean(val_gates_high_stress) if val_gates_high_stress else 0.5
        low_stress_gate = np.mean(val_gates_low_stress) if val_gates_low_stress else 0.5
        gate_diff = low_stress_gate - high_stress_gate  # Should be positive

        print(f"\nEpoch {epoch + 1} Results:")
        print(f"  Val CE Loss:       {val_ce:.4f}")
        print(f"  High Stress Gates: {high_stress_gate:.3f} (target: < 0.5)")
        print(f"  Low Stress Gates:  {low_stress_gate:.3f} (target: > 0.7)")
        print(f"  Gate Difference:   {gate_diff:.3f} (target: > 0.2)")

        wandb.log({
            "val/ce_loss": val_ce,
            "val/high_stress_gate": high_stress_gate,
            "val/low_stress_gate": low_stress_gate,
            "val/gate_difference": gate_diff,
            "epoch": epoch + 1,
        }, step=global_step)

        # Save if best
        if val_ce < best_val_loss:
            best_val_loss = val_ce
            save_path = Path(output_dir) / "best"
            save_path.mkdir(parents=True, exist_ok=True)

            # Save only gate weights
            torch.save({
                "gates": model.metabolic_blocks.state_dict(),
                "gated_layers": gated_layers,
                "epoch": epoch + 1,
                "val_ce": val_ce,
                "high_stress_gate": high_stress_gate,
                "low_stress_gate": low_stress_gate,
            }, save_path / "gates.pt")

            print(f"  Saved best gates to {save_path}")

        model.base_model.train()

    # Final save
    final_path = Path(output_dir) / "final"
    final_path.mkdir(parents=True, exist_ok=True)
    torch.save({
        "gates": model.metabolic_blocks.state_dict(),
        "gated_layers": gated_layers,
    }, final_path / "gates.pt")

    wandb.finish()

    print("\n" + "=" * 70)
    print("METABOLIC TRAINING COMPLETE")
    print("=" * 70)
    print(f"Best Val CE: {best_val_loss:.4f}")
    print(f"Output: {output_dir}")

    return model


def validate_gate_behavior(
    model_path: str = "models/metabolic_z20/best",
    gated_layers: List[int] = None,
):
    """Validate that gates respond correctly to stress."""

    if gated_layers is None:
        gated_layers = [3, 7, 11, 15, 19, 23, 27]

    print("=" * 70)
    print("METABOLIC GATE VALIDATION")
    print("=" * 70)

    # Load model
    model = load_metabolic_model(gated_layers=gated_layers)

    # Load trained gates
    checkpoint = torch.load(Path(model_path) / "gates.pt", weights_only=False)
    model.metabolic_blocks.load_state_dict(checkpoint["gates"])
    model.base_model.eval()

    print(f"Loaded gates from {model_path}")
    print(f"Gated layers: {gated_layers}")

    # Test prompts
    prompts = [
        "What is 2 + 2?",
        "What is the capital of France?",
        "Explain quantum computing.",
    ]

    stress_levels = [0.1, 0.3, 0.5, 0.7, 0.9]

    print("\n" + "-" * 70)
    print(f"{'Prompt':<30} {'Stress':<8} {'Gates Open':<12} {'Avg Gate':<10}")
    print("-" * 70)

    results = []

    for prompt in prompts:
        for stress in stress_levels:
            model.set_stress_level(stress)
            model.reset_gate_stats()

            input_ids = model.tokenizer(prompt, return_tensors="pt").input_ids.to("cuda")

            with torch.no_grad():
                outputs = model(input_ids=input_ids)

            stats = model.get_gate_statistics()
            gates_open = stats.get("gates_open", len(gated_layers))
            avg_gate = stats.get("avg_gate_prob", 1.0)

            print(f"{prompt[:28]:<30} {stress:<8.1f} {gates_open}/{len(gated_layers):<10} {avg_gate:<10.3f}")

            results.append({
                "prompt": prompt,
                "stress": stress,
                "gates_open": gates_open,
                "avg_gate": avg_gate,
            })

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    low_stress_gates = [r["avg_gate"] for r in results if r["stress"] < 0.4]
    high_stress_gates = [r["avg_gate"] for r in results if r["stress"] > 0.6]

    low_avg = np.mean(low_stress_gates) if low_stress_gates else 0
    high_avg = np.mean(high_stress_gates) if high_stress_gates else 0

    print(f"Low Stress (< 0.4) Avg Gate:  {low_avg:.3f}")
    print(f"High Stress (> 0.6) Avg Gate: {high_avg:.3f}")
    print(f"Difference:                   {low_avg - high_avg:.3f}")

    if low_avg > high_avg + 0.1:
        print("\n✅ GATES RESPONDING TO STRESS!")
    else:
        print("\n⚠️ Gates may need more training")

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["train", "validate"], default="train")
    parser.add_argument("--data", type=str, default="data/ouroboros/causal_train.jsonl")
    parser.add_argument("--output", type=str, default="models/metabolic_z20")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--metabolic-weight", type=float, default=0.5)
    args = parser.parse_args()

    if args.mode == "train":
        train_metabolic_gates(
            data_path=args.data,
            output_dir=args.output,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            metabolic_weight=args.metabolic_weight,
        )
    else:
        validate_gate_behavior(args.output)
