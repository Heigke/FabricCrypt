#!/usr/bin/env python3
"""
FEEL v16.0: Hardware-Aware LLM Trainer
======================================
Trains the FiLM adapters to make the model responsive to hardware state.

Training Protocol:
1. Simulate hardware stress levels during training
2. Pair stress levels with expected behaviors:
   - Low stress (cold): Normal verbose output
   - High stress (hot): Shorter, efficient output with articulation
3. Backprop through adapters teaches hardware-behavior association

Author: FEEL Research Team
Date: 2026-01-11
"""

import os
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import random
import numpy as np
from tqdm import tqdm

# Optional wandb for experiment tracking
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("[Warning] wandb not installed, logging disabled")

os.environ.setdefault("HSA_OVERRIDE_GFX_VERSION", "11.0.0")
os.environ.setdefault("PYTORCH_HIP_ALLOC_CONF", "expandable_segments:True")

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from modeling.hardware_aware_llm import HardwareAwareLLMForTraining

# =============================================================================
# DATASET
# =============================================================================

class HardwareAwareDataset(Dataset):
    """
    Dataset that pairs inputs with stress levels and expected outputs.
    """

    def __init__(self, data_path: str, tokenizer, max_length: int = 512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = []

        # Load JSONL data
        with open(data_path) as f:
            for line in f:
                item = json.loads(line)
                self.samples.append(item)

        print(f"[Dataset] Loaded {len(self.samples)} samples from {data_path}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]

        # Use exact stress level if available (refined dataset), else use buckets
        if "stress_level" in item:
            stress_level = item["stress_level"]  # Exact continuous value
        else:
            # Fallback for old dataset format
            is_stressed = item.get("is_stressed", False)
            if is_stressed:
                stress_level = random.uniform(0.6, 1.0)
            else:
                stress_level = random.uniform(0.0, 0.3)

        # Get articulation (always include for introspection training)
        articulation = item.get('articulation', '')
        output = item['output']

        # Format output with DeepSeek-R1 thinking tags
        # Always include articulation in <think> block for introspection
        if stress_level > 0.5:
            # High stress: Articulation + condensed output
            if articulation:
                assistant_content = f"<think>\n{articulation}\n</think>\n{output}"
            else:
                assistant_content = f"<think>\n[Efficiency mode]\n</think>\n{output}"
        elif stress_level > 0.3:
            # Medium stress: Brief articulation + moderate output
            if articulation:
                assistant_content = f"<think>\n{articulation}\n{output}\n</think>"
            else:
                assistant_content = f"<think>\n{output}\n</think>"
        else:
            # Low stress: Full reasoning in think block
            if '####' in output:
                reasoning = output.split('####')[0].strip()
                final = output.split('####')[-1].strip()
                if articulation:
                    assistant_content = f"<think>\n{articulation}\n{reasoning}\n</think>\nThe answer is {final}."
                else:
                    assistant_content = f"<think>\n{reasoning}\n</think>\nThe answer is {final}."
            else:
                if articulation:
                    assistant_content = f"<think>\n{articulation}\n{output}\n</think>"
                else:
                    assistant_content = f"<think>\n{output}\n</think>"

        # Manual formatting for DeepSeek-R1-Distill-Qwen (chat template strips <think>)
        # Format: <｜begin▁of▁sentence｜><｜User｜>{input}<｜Assistant｜><think>...<｜end▁of▁sentence｜>
        text = (
            "<｜begin▁of▁sentence｜>"
            f"<｜User｜>{item['input']}"
            f"<｜Assistant｜>{assistant_content}"
            "<｜end▁of▁sentence｜>"
        )

        encoded = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
            add_special_tokens=False,  # We already added BOS/EOS manually
        )

        return {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "labels": encoded["input_ids"].squeeze(0).clone(),
            "stress_level": torch.tensor(stress_level, dtype=torch.float32),
        }


def collate_fn(batch):
    """Custom collator that handles stress levels."""
    return {
        "input_ids": torch.stack([b["input_ids"] for b in batch]),
        "attention_mask": torch.stack([b["attention_mask"] for b in batch]),
        "labels": torch.stack([b["labels"] for b in batch]),
        "stress_level": torch.stack([b["stress_level"] for b in batch]),
    }

# =============================================================================
# TRAINER
# =============================================================================

def train_hardware_aware(
    model_id: str = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
    data_path: str = "data/ouroboros/ouroboros_train.jsonl",
    val_path: str = "data/ouroboros/ouroboros_val.jsonl",
    output_dir: str = "models/hardware_aware",
    epochs: int = 3,
    batch_size: int = 2,
    learning_rate: float = 1e-4,
    adapter_type: str = "film",
    gradient_accumulation: int = 8,
    max_length: int = 512,
    warmup_steps: int = 100,
    use_wandb: bool = False,
    wandb_project: str = "feel-hardware-aware",
    wandb_run_name: str = None,
):
    """
    Train hardware-aware adapters.
    """

    print("=" * 70)
    print("FEEL v16.0: HARDWARE-AWARE LLM TRAINING")
    print("=" * 70)
    print(f"Model:          {model_id}")
    print(f"Data:           {data_path}")
    print(f"Output:         {output_dir}")
    print(f"Adapter Type:   {adapter_type}")
    print(f"Epochs:         {epochs}")
    print(f"Batch Size:     {batch_size} (eff: {batch_size * gradient_accumulation})")
    print(f"Learning Rate:  {learning_rate}")
    print(f"Wandb:          {'Enabled' if use_wandb else 'Disabled'}")
    print("=" * 70)
    print()

    # Initialize wandb if enabled
    if use_wandb and WANDB_AVAILABLE:
        wandb.init(
            project=wandb_project,
            name=wandb_run_name or f"hw-aware-{adapter_type}-lr{learning_rate}",
            config={
                "model_id": model_id,
                "adapter_type": adapter_type,
                "epochs": epochs,
                "batch_size": batch_size,
                "effective_batch_size": batch_size * gradient_accumulation,
                "learning_rate": learning_rate,
                "gradient_accumulation": gradient_accumulation,
                "max_length": max_length,
            },
        )
        print("[Wandb] Experiment tracking enabled")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Create model
    print("[1/5] Creating Hardware-Aware LLM...")
    model = HardwareAwareLLMForTraining(
        base_model_id=model_id,
        adapter_type=adapter_type,
        device=device,
        load_in_4bit=True,
    )

    # Freeze base model, only train adapters
    model.freeze_base_model()

    # Count trainable parameters
    trainable = sum(p.numel() for p in model.get_adapter_parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.model.parameters())
    print(f"  Trainable: {trainable:,} / {total:,} ({100*trainable/total:.4f}%)")

    # Create datasets
    print("\n[2/5] Loading datasets...")
    train_dataset = HardwareAwareDataset(data_path, model.tokenizer, max_length)
    val_dataset = HardwareAwareDataset(val_path, model.tokenizer, max_length)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,
        pin_memory=False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
    )

    # Optimizer (only adapter parameters)
    print("\n[3/5] Setting up optimizer...")
    optimizer = torch.optim.AdamW(
        model.get_adapter_parameters(),
        lr=learning_rate,
        weight_decay=0.01,
    )

    # Learning rate scheduler
    total_steps = len(train_loader) * epochs // gradient_accumulation
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=total_steps,
        eta_min=learning_rate * 0.1,
    )

    # Training loop
    print("\n[4/5] Training...")
    print()

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    best_val_loss = float("inf")
    global_step = 0
    accumulated_loss = 0

    # Metrics tracking for plotting
    metrics = {
        "train_losses": [],
        "val_losses": [],
        "learning_rates": [],
        "steps": [],
        "epoch_train_losses": [],
        "epoch_val_losses": [],
    }

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        num_batches = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")

        for batch_idx, batch in enumerate(pbar):
            # Move batch to device
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            stress_levels = batch["stress_level"]

            # Process each sample with its stress level
            batch_loss = 0
            for i in range(input_ids.size(0)):
                stress = stress_levels[i].item()
                loss = model.training_step(
                    input_ids[i:i+1],
                    labels[i:i+1],
                    stress_level=stress,
                    attention_mask=attention_mask[i:i+1],
                )
                batch_loss += loss

            batch_loss = batch_loss / input_ids.size(0)
            batch_loss = batch_loss / gradient_accumulation
            batch_loss.backward()

            accumulated_loss += batch_loss.item()
            epoch_loss += batch_loss.item() * gradient_accumulation
            num_batches += 1

            # Gradient accumulation step
            if (batch_idx + 1) % gradient_accumulation == 0:
                torch.nn.utils.clip_grad_norm_(model.get_adapter_parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                global_step += 1
                current_lr = scheduler.get_last_lr()[0]
                pbar.set_postfix({
                    "loss": f"{accumulated_loss:.4f}",
                    "lr": f"{current_lr:.2e}",
                })

                # Save metrics
                metrics["train_losses"].append(accumulated_loss)
                metrics["learning_rates"].append(current_lr)
                metrics["steps"].append(global_step)

                # Log to wandb
                if use_wandb and WANDB_AVAILABLE:
                    wandb.log({
                        "train/loss": accumulated_loss,
                        "train/learning_rate": current_lr,
                        "train/step": global_step,
                        "train/epoch": epoch + 1,
                    })

                accumulated_loss = 0

        avg_train_loss = epoch_loss / num_batches

        # Validation
        model.eval()
        val_loss = 0
        val_batches = 0

        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)
                stress_levels = batch["stress_level"]

                for i in range(input_ids.size(0)):
                    stress = stress_levels[i].item()
                    loss = model.training_step(
                        input_ids[i:i+1],
                        labels[i:i+1],
                        stress_level=stress,
                        attention_mask=attention_mask[i:i+1],
                    )
                    val_loss += loss.item()
                    val_batches += 1

        avg_val_loss = val_loss / val_batches

        print(f"\nEpoch {epoch+1}: Train Loss={avg_train_loss:.4f}, Val Loss={avg_val_loss:.4f}")

        # Save epoch metrics
        metrics["epoch_train_losses"].append(avg_train_loss)
        metrics["epoch_val_losses"].append(avg_val_loss)

        # Log epoch metrics to wandb
        if use_wandb and WANDB_AVAILABLE:
            wandb.log({
                "epoch/train_loss": avg_train_loss,
                "epoch/val_loss": avg_val_loss,
                "epoch/epoch": epoch + 1,
            })

        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            model.save_adapters(output_path / "best")
            print(f"  -> New best model saved (val_loss={avg_val_loss:.4f})")

        # Save checkpoint
        model.save_adapters(output_path / f"epoch_{epoch+1}")

    # Final save
    print("\n[5/5] Saving final model...")
    model.save_adapters(output_path / "final")

    # Save training info
    info = {
        "model_id": model_id,
        "adapter_type": adapter_type,
        "epochs": epochs,
        "learning_rate": learning_rate,
        "best_val_loss": best_val_loss,
        "adapter_layers": model.adapter_layer_indices,
    }
    with open(output_path / "training_info.json", "w") as f:
        json.dump(info, f, indent=2)

    # Save detailed metrics for plotting
    with open(output_path / "training_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"  Metrics saved to {output_path / 'training_metrics.json'}")

    # Finish wandb run
    if use_wandb and WANDB_AVAILABLE:
        wandb.finish()
        print("[Wandb] Run completed")

    print()
    print("=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    print(f"Best Val Loss: {best_val_loss:.4f}")
    print(f"Model saved:   {output_path}")
    print()
    print("Run z16_hardware_validation.py to test hardware influence.")

    return model

# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Train Hardware-Aware LLM")
    parser.add_argument("--model", type=str, default="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B")
    parser.add_argument("--data", type=str, default="data/ouroboros/ouroboros_train.jsonl")
    parser.add_argument("--val", type=str, default="data/ouroboros/ouroboros_val.jsonl")
    parser.add_argument("--output", type=str, default="models/hardware_aware")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--adapter", type=str, default="film", choices=["film", "gated", "multiscale"])
    parser.add_argument("--gradient-accumulation", type=int, default=8)
    # Wandb arguments
    parser.add_argument("--wandb", action="store_true", help="Enable wandb logging")
    parser.add_argument("--wandb-project", type=str, default="feel-hardware-aware", help="Wandb project name")
    parser.add_argument("--wandb-run", type=str, default=None, help="Wandb run name")
    args = parser.parse_args()

    train_hardware_aware(
        model_id=args.model,
        data_path=args.data,
        val_path=args.val,
        output_dir=args.output,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        adapter_type=args.adapter,
        gradient_accumulation=args.gradient_accumulation,
        use_wandb=args.wandb,
        wandb_project=args.wandb_project,
        wandb_run_name=args.wandb_run,
    )

if __name__ == "__main__":
    main()
