#!/usr/bin/env python3
"""
FEEL z22: Proprioceptive Training
=================================
Train the model to FEEL its own gates closing.

Key Innovation:
- Train strain embeddings alongside gates
- Model learns: "When I feel this strain, I should be brief"
- Behavior emerges from sensation, not explicit training

Training Strategy:
1. FREEZE base LLM (7B params)
2. TRAIN gates + strain embeddings (~100K params)
3. Loss = CE + Metabolic + Strain Alignment

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

from modeling.proprioceptive_gate import (
    ProprioceptiveDeepSeek,
    ProprioceptiveLoss,
    load_proprioceptive_model,
)


class ProprioceptiveDataset(Dataset):
    """
    Dataset for proprioceptive training.

    Each sample has:
    - prompt: The input
    - response: The target output
    - stress: Thermal stress level (0-1)
    - mode: "scholar" (verbose) or "survivor" (brief)

    The model should learn:
    - Low stress + scholar mode -> verbose output
    - High stress + survivor mode -> brief output
    - The strain embedding mediates this transition
    """

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

        print(f"[ProprioceptiveDataset] Loaded {len(self.samples)} samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        prompt = sample["prompt"]
        response = sample["response"]
        stress = sample.get("stress", 0.5)
        mode = sample.get("mode", "neutral")

        full_text = f"{prompt}\n{response}"

        encoding = self.tokenizer(
            full_text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )

        # Target length for strain alignment
        response_tokens = self.tokenizer(
            response,
            truncation=True,
            max_length=self.max_length,
        )
        target_length = len(response_tokens["input_ids"])

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "stress": torch.tensor(stress, dtype=torch.float32),
            "target_length": target_length,
            "mode": mode,
        }


class ProprioceptiveCollator:
    """Collate batches with sensor tensors and metadata."""

    def __call__(self, batch: List[Dict]) -> Dict:
        input_ids = torch.stack([b["input_ids"] for b in batch])
        attention_mask = torch.stack([b["attention_mask"] for b in batch])
        stress = torch.stack([b["stress"] for b in batch])
        target_lengths = [b["target_length"] for b in batch]
        modes = [b["mode"] for b in batch]

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": input_ids.clone(),
            "stress": stress,
            "target_lengths": target_lengths,
            "modes": modes,
        }


def train_proprioceptive(
    data_path: str = "data/ouroboros/causal_train.jsonl",
    output_dir: str = "models/proprioceptive_z22",
    metabolic_checkpoint: str = "models/metabolic_z20/best/gates.pt",
    gated_layers: List[int] = None,
    epochs: int = 3,
    batch_size: int = 4,
    gradient_accumulation: int = 4,
    learning_rate: float = 5e-4,
    metabolic_weight: float = 0.3,
    strain_weight: float = 0.2,
):
    """Train proprioceptive gates and strain embeddings."""

    if gated_layers is None:
        gated_layers = [3, 7, 11, 15, 19, 23, 27]

    print("=" * 70)
    print("FEEL z22: PROPRIOCEPTIVE TRAINING")
    print("=" * 70)
    print(f"Data:              {data_path}")
    print(f"Output:            {output_dir}")
    print(f"Metabolic Ckpt:    {metabolic_checkpoint}")
    print(f"Gated Layers:      {gated_layers}")
    print(f"Epochs:            {epochs}")
    print(f"Batch (eff):       {batch_size * gradient_accumulation}")
    print(f"Learning Rate:     {learning_rate}")
    print(f"Metabolic Weight:  {metabolic_weight}")
    print(f"Strain Weight:     {strain_weight}")
    print("=" * 70)

    # Initialize wandb
    run_name = f"proprioceptive-{datetime.now().strftime('%Y%m%d_%H%M')}"
    wandb.init(project="feel-proprioceptive", name=run_name, config={
        "gated_layers": gated_layers,
        "epochs": epochs,
        "batch_size": batch_size * gradient_accumulation,
        "learning_rate": learning_rate,
        "metabolic_weight": metabolic_weight,
        "strain_weight": strain_weight,
    })

    # Load model with proprioceptive gates
    print("\n[1/5] Loading ProprioceptiveDeepSeek...")
    model = load_proprioceptive_model(
        base_model_path="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
        gated_layers=gated_layers,
        metabolic_checkpoint=metabolic_checkpoint if Path(metabolic_checkpoint).exists() else None,
    )

    # Freeze base model
    print("\n[2/5] Freezing base model...")
    model.freeze_base_model()

    # Create dataset
    print("\n[3/5] Loading dataset...")
    dataset = ProprioceptiveDataset(
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

    # Dataloaders
    collator = ProprioceptiveCollator()
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

    # Optimizer
    print("\n[4/5] Setting up optimizer...")
    trainable_params = model.get_trainable_parameters()
    optimizer = torch.optim.AdamW(trainable_params, lr=learning_rate)

    # Loss function
    loss_fn = ProprioceptiveLoss(
        metabolic_weight=metabolic_weight,
        strain_weight=strain_weight,
    )

    # Scheduler
    total_steps = len(train_loader) * epochs // gradient_accumulation
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_steps
    )

    # Training
    print("\n[5/5] Training proprioceptive system...")
    model.base_model.train()
    best_val_loss = float("inf")
    global_step = 0

    for epoch in range(epochs):
        print(f"\n--- Epoch {epoch + 1}/{epochs} ---")

        epoch_ce_loss = 0.0
        epoch_metabolic = 0.0
        epoch_strain = 0.0
        optimizer.zero_grad()

        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}")
        for batch_idx, batch in enumerate(pbar):
            # Move to GPU
            input_ids = batch["input_ids"].to("cuda")
            attention_mask = batch["attention_mask"].to("cuda")
            labels = batch["labels"].to("cuda")
            stress = batch["stress"].to("cuda")

            # Set stress level (batch average)
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
            strain_magnitudes = outputs.strain_magnitudes

            # Compute loss
            total_loss, metrics = loss_fn(
                ce_loss=ce_loss,
                gate_tensors=gate_tensors,
                strain_magnitudes=strain_magnitudes,
                stress_level=stress,
            )

            # Backward
            (total_loss / gradient_accumulation).backward()

            epoch_ce_loss += metrics["ce_loss"]
            epoch_metabolic += metrics["metabolic_penalty"]
            epoch_strain += metrics["avg_strain"]

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
                    "train/metabolic_penalty": metrics["metabolic_penalty"],
                    "train/avg_gate": metrics["avg_gate"],
                    "train/avg_strain": metrics["avg_strain"],
                    "train/stress": avg_stress,
                    "train/lr": scheduler.get_last_lr()[0],
                }, step=global_step)

            pbar.set_postfix({
                "ce": f"{metrics['ce_loss']:.3f}",
                "gate": f"{metrics['avg_gate']:.2f}",
                "strain": f"{metrics['avg_strain']:.2f}",
                "stress": f"{avg_stress:.2f}",
            })

        # Validation
        model.base_model.eval()
        val_metrics = validate_proprioception(model, val_loader, loss_fn)

        print(f"\nEpoch {epoch + 1} Results:")
        print(f"  Val CE Loss:        {val_metrics['val_ce']:.4f}")
        print(f"  High Stress Gates:  {val_metrics['high_stress_gate']:.3f}")
        print(f"  Low Stress Gates:   {val_metrics['low_stress_gate']:.3f}")
        print(f"  High Stress Strain: {val_metrics['high_stress_strain']:.3f}")
        print(f"  Low Stress Strain:  {val_metrics['low_stress_strain']:.3f}")
        print(f"  Strain-Gate Correlation: {val_metrics['strain_gate_corr']:.3f}")

        wandb.log({
            "val/ce_loss": val_metrics["val_ce"],
            "val/high_stress_gate": val_metrics["high_stress_gate"],
            "val/low_stress_gate": val_metrics["low_stress_gate"],
            "val/high_stress_strain": val_metrics["high_stress_strain"],
            "val/low_stress_strain": val_metrics["low_stress_strain"],
            "epoch": epoch + 1,
        }, step=global_step)

        # Save if best
        if val_metrics["val_ce"] < best_val_loss:
            best_val_loss = val_metrics["val_ce"]
            save_path = Path(output_dir) / "best"
            save_path.mkdir(parents=True, exist_ok=True)

            # Save proprioceptive state (gates + strain embeddings)
            state_dict = {}
            for key, block in model.proprioceptive_blocks.items():
                for name, param in block.gate.named_parameters():
                    state_dict[f"{key}.gate.{name}"] = param.data

            torch.save({
                "proprioceptive": state_dict,
                "gated_layers": gated_layers,
                "epoch": epoch + 1,
                "val_ce": val_metrics["val_ce"],
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
    print("PROPRIOCEPTIVE TRAINING COMPLETE")
    print("=" * 70)
    print(f"Best Val CE: {best_val_loss:.4f}")
    print(f"Output: {output_dir}")

    return model


def validate_proprioception(model, val_loader, loss_fn) -> Dict:
    """Validate proprioceptive behavior."""
    val_ce = 0.0
    high_stress_gates = []
    low_stress_gates = []
    high_stress_strains = []
    low_stress_strains = []

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
                    avg_strain = sum(outputs.strain_magnitudes) / len(outputs.strain_magnitudes)

                    if s.item() > 0.6:
                        high_stress_gates.append(avg_gate)
                        high_stress_strains.append(avg_strain)
                    elif s.item() < 0.4:
                        low_stress_gates.append(avg_gate)
                        low_stress_strains.append(avg_strain)

    n_samples = len(val_loader.dataset)
    val_ce /= n_samples

    # Compute correlation between strain and gate
    all_gates = high_stress_gates + low_stress_gates
    all_strains = high_stress_strains + low_stress_strains
    if len(all_gates) > 1:
        corr = np.corrcoef(all_gates, all_strains)[0, 1]
    else:
        corr = 0.0

    return {
        "val_ce": val_ce,
        "high_stress_gate": np.mean(high_stress_gates) if high_stress_gates else 0.5,
        "low_stress_gate": np.mean(low_stress_gates) if low_stress_gates else 0.5,
        "high_stress_strain": np.mean(high_stress_strains) if high_stress_strains else 0.0,
        "low_stress_strain": np.mean(low_stress_strains) if low_stress_strains else 0.0,
        "strain_gate_corr": corr if not np.isnan(corr) else 0.0,
    }


def demo_proprioception(
    model_path: str = "models/proprioceptive_z22/best",
    gated_layers: List[int] = None,
):
    """Demonstrate proprioceptive behavior."""

    if gated_layers is None:
        gated_layers = [3, 7, 11, 15, 19, 23, 27]

    print("=" * 70)
    print("FEEL z22: PROPRIOCEPTION DEMO")
    print("=" * 70)

    # Load model
    model = load_proprioceptive_model(gated_layers=gated_layers)

    # Load trained weights
    checkpoint_path = Path(model_path) / "proprioceptive.pt"
    if checkpoint_path.exists():
        print(f"Loading from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, weights_only=False)

        state_dict = checkpoint.get("proprioceptive", checkpoint)
        for key, block in model.proprioceptive_blocks.items():
            for name, param in block.gate.named_parameters():
                full_key = f"{key}.gate.{name}"
                if full_key in state_dict:
                    param.data.copy_(state_dict[full_key])
    else:
        print("No trained weights found, using random initialization")

    model.base_model.eval()

    # Test prompts
    prompts = [
        "Explain quantum entanglement.",
        "What is 2 + 2?",
        "Describe the process of photosynthesis.",
    ]

    stress_levels = [0.1, 0.5, 0.9]

    print("\n" + "-" * 70)
    print(f"{'Stress':<8} {'Gates':<8} {'Strain':<10} {'Response Preview':<40}")
    print("-" * 70)

    for prompt in prompts:
        print(f"\nPrompt: {prompt[:50]}...")

        for stress in stress_levels:
            model.set_stress_level(stress)

            response, stats = model.generate(prompt, max_new_tokens=50)

            # Extract just the response part
            response_text = response[len(prompt):].strip()[:40]

            print(f"{stress:<8.1f} {stats['gates_open']}/{len(gated_layers):<5} "
                  f"{stats['total_strain']:<10.2f} {response_text}...")

    print("\n" + "=" * 70)
    print("KEY OBSERVATION:")
    print("High strain should correlate with shorter, simpler responses.")
    print("The model 'feels' its reduced capacity through the strain embedding.")
    print("=" * 70)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["train", "demo"], default="train")
    parser.add_argument("--data", type=str, default="data/ouroboros/causal_train.jsonl")
    parser.add_argument("--output", type=str, default="models/proprioceptive_z22")
    parser.add_argument("--metabolic", type=str, default="models/metabolic_z20/best/gates.pt")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=5e-4)
    args = parser.parse_args()

    if args.mode == "train":
        train_proprioceptive(
            data_path=args.data,
            output_dir=args.output,
            metabolic_checkpoint=args.metabolic,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
        )
    else:
        demo_proprioception(args.output)
