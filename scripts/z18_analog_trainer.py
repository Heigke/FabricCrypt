#!/usr/bin/env python3
"""
FEEL v18: Analog Control Trainer
=================================
Fixes the "Binary Trap" to achieve true S-Curve analog control.

Key Changes from v16:
1. AnalogHardwareCollator: Continuous stress values (not binary 0.1/0.9)
2. AnalogFiLMAdapter: LeakyReLU for better gradient flow
3. Gradient Coverage: Ensures training samples span full 0.0-1.0 range
4. Lower LR: Fine-tuning mode to preserve efficiency gains

Usage:
    # Fresh start
    python z18_analog_trainer.py --epochs 1 --lr 5e-5

    # Resume from z17 checkpoint (recommended)
    python z18_analog_trainer.py --resume models/hardware_aware_z17/final --epochs 1 --lr 5e-5

Author: FEEL Research Team
Date: 2026-01-12
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
from datetime import datetime

# Optional wandb
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

os.environ.setdefault("HSA_OVERRIDE_GFX_VERSION", "11.0.0")
os.environ.setdefault("PYTORCH_HIP_ALLOC_CONF", "expandable_segments:True")

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from transformers import AutoModelForCausalLM, AutoTokenizer
from modeling.analog_adapter import AnalogFiLMAdapter, create_analog_adapter

# =============================================================================
# ANALOG DATASET
# =============================================================================

class AnalogHardwareDataset(Dataset):
    """
    Dataset with continuous stress levels for analog control training.

    Each sample gets a stress level from a continuous distribution:
    - Calm zone: Beta(2,5) scaled to [0.0, 0.35]
    - Mid zone: Uniform [0.35, 0.65] (sparse but present)
    - Stressed zone: Beta(5,2) scaled to [0.65, 1.0]
    """

    def __init__(
        self,
        data_path: str,
        tokenizer,
        max_length: int = 512,
        mid_zone_prob: float = 0.2,  # 20% samples in transition zone
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.mid_zone_prob = mid_zone_prob
        self.samples = []

        # Load JSONL data
        with open(data_path) as f:
            for line in f:
                item = json.loads(line)
                self.samples.append(item)

        print(f"[AnalogDataset] Loaded {len(self.samples)} samples from {data_path}")

    def _get_analog_stress(self, item: Dict) -> float:
        """Generate continuous stress value."""
        # If explicit stress_level exists, use it with jitter
        if "stress_level" in item:
            base = item["stress_level"]
            jitter = random.gauss(0, 0.03)
            return max(0.0, min(1.0, base + jitter))

        # Determine zone from is_stressed flag or heuristic
        if "is_stressed" in item:
            is_stressed = item["is_stressed"]
        else:
            # Heuristic: short output = stressed
            output_len = len(item.get("output", ""))
            is_stressed = output_len < 150

        # Occasionally force mid-zone samples (critical for S-Curve!)
        if random.random() < self.mid_zone_prob:
            return random.uniform(0.35, 0.65)

        if is_stressed:
            # Stressed: Beta distribution clustered toward 0.8
            base = np.random.beta(5, 2)  # Skewed high
            return 0.65 + base * 0.35  # Scale to [0.65, 1.0]
        else:
            # Calm: Beta distribution clustered toward 0.15
            base = np.random.beta(2, 5)  # Skewed low
            return base * 0.35  # Scale to [0.0, 0.35]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        stress_level = self._get_analog_stress(item)

        # Get articulation for introspection training
        articulation = item.get('articulation', '')
        output = item['output']

        # Format with DeepSeek-R1 thinking tags
        # Stress level influences the articulation content
        if stress_level > 0.7:
            # High stress: Include thermal awareness
            thermal_marker = "[THERMAL WARNING] "
            if articulation:
                assistant_content = f"<think>\n{thermal_marker}{articulation}\n</think>\n{output}"
            else:
                assistant_content = f"<think>\n{thermal_marker}Efficiency mode active.\n</think>\n{output}"
        elif stress_level > 0.5:
            # Medium-high: Mild awareness
            if articulation:
                assistant_content = f"<think>\n[Moderate load] {articulation}\n</think>\n{output}"
            else:
                assistant_content = f"<think>\n{output}\n</think>"
        elif stress_level > 0.3:
            # Medium: Normal processing
            if '####' in output:
                reasoning = output.split('####')[0].strip()
                final = output.split('####')[-1].strip()
                assistant_content = f"<think>\n{reasoning}\n</think>\nThe answer is {final}."
            else:
                assistant_content = f"<think>\n{output}\n</think>"
        else:
            # Low stress: Full verbose reasoning
            if articulation:
                assistant_content = f"<think>\n{articulation}\nLet me work through this carefully.\n{output}\n</think>"
            else:
                if '####' in output:
                    reasoning = output.split('####')[0].strip()
                    final = output.split('####')[-1].strip()
                    assistant_content = f"<think>\nLet me think step by step.\n{reasoning}\n</think>\nThe answer is {final}."
                else:
                    assistant_content = f"<think>\nLet me think through this.\n{output}\n</think>"

        # Manual formatting for DeepSeek-R1
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
            add_special_tokens=False,
        )

        return {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "labels": encoded["input_ids"].squeeze(0).clone(),
            "stress_level": torch.tensor(stress_level, dtype=torch.float32),
        }


def analog_collate_fn(batch):
    """Collate with analog stress levels."""
    return {
        "input_ids": torch.stack([b["input_ids"] for b in batch]),
        "attention_mask": torch.stack([b["attention_mask"] for b in batch]),
        "labels": torch.stack([b["labels"] for b in batch]),
        "stress_level": torch.stack([b["stress_level"] for b in batch]),
    }


# =============================================================================
# ANALOG-AWARE LLM
# =============================================================================

class AnalogAwareLLM(nn.Module):
    """
    LLM with analog FiLM adapters for continuous hardware response.
    """

    def __init__(
        self,
        base_model_id: str = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
        adapter_type: str = "analog",
        adapter_layers: Optional[List[int]] = None,
        device: str = "cuda",
    ):
        super().__init__()
        self.device = device
        self.adapter_type = adapter_type

        # Load base model
        print(f"[AnalogLLM] Loading {base_model_id}...")
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            base_model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        self.config = self.model.config

        # Get model internals
        self._setup_model_internals()

        # Adapter layer indices
        if adapter_layers is None:
            # More adapter layers for finer control
            self.adapter_layer_indices = [
                self.num_layers // 5,
                2 * self.num_layers // 5,
                3 * self.num_layers // 5,
                4 * self.num_layers // 5,
                self.num_layers - 1,
            ]
        else:
            self.adapter_layer_indices = adapter_layers

        # Create analog adapters
        print(f"[AnalogLLM] Creating {adapter_type} adapters at layers {self.adapter_layer_indices}...")
        self.adapters = nn.ModuleDict()
        for idx in self.adapter_layer_indices:
            self.adapters[str(idx)] = create_analog_adapter(
                adapter_type,
                self.hidden_size,
                sensor_dim=3,
            ).to(device)

        # Register hooks
        self._register_hooks()

        # Current stress level (set during forward)
        self._current_stress = 0.5

        print(f"[AnalogLLM] Ready. {len(self.adapters)} adapters installed.")

    def _setup_model_internals(self):
        """Find transformer layers."""
        if hasattr(self.model, "model"):
            self.base_model = self.model.model
        elif hasattr(self.model, "transformer"):
            self.base_model = self.model.transformer
        else:
            self.base_model = self.model

        if hasattr(self.base_model, "layers"):
            self.layers = self.base_model.layers
        elif hasattr(self.base_model, "h"):
            self.layers = self.base_model.h
        else:
            raise ValueError("Cannot find transformer layers")

        self.num_layers = len(self.layers)
        self.hidden_size = self.config.hidden_size
        print(f"  Model: {self.num_layers} layers, {self.hidden_size} hidden dim")

    def _register_hooks(self):
        """Register forward hooks for hardware injection."""
        self.hooks = []
        for idx in self.adapter_layer_indices:
            hook = self.layers[idx].register_forward_hook(
                self._create_adapter_hook(idx)
            )
            self.hooks.append(hook)

    def _create_adapter_hook(self, layer_idx: int):
        """Create hook that applies analog adapter."""
        def hook(module, input, output):
            # Create sensor vector from stress level
            # [temp, power, clock] - correlated with noise
            stress = self._current_stress
            temp = stress + random.gauss(0, 0.02)
            power = stress * 0.8 + random.gauss(0.2, 0.05)
            clock = 1.0 - stress * 0.2 + random.gauss(0, 0.03)

            sensor = torch.tensor(
                [max(0, min(1, temp)), max(0, min(1, power)), max(0.5, min(1, clock))],
                device=self.device,
                dtype=torch.float32
            )

            # Get hidden states
            if isinstance(output, tuple):
                hidden_states = output[0]
            else:
                hidden_states = output

            # Apply adapter
            adapter = self.adapters[str(layer_idx)]
            modulated = adapter(hidden_states, sensor)

            if isinstance(output, tuple):
                return (modulated,) + output[1:]
            return modulated

        return hook

    def set_stress_level(self, level: float):
        """Set current stress level for generation."""
        self._current_stress = max(0.0, min(1.0, level))

    def forward(self, input_ids, attention_mask=None, labels=None, **kwargs):
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            **kwargs
        )

    def training_step(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor,
        stress_level: float,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Single training step with specified stress level."""
        self.set_stress_level(stress_level)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
        return outputs.loss

    def get_adapter_parameters(self):
        """Get only adapter parameters."""
        return self.adapters.parameters()

    def freeze_base_model(self):
        """Freeze base, train only adapters."""
        for param in self.model.parameters():
            param.requires_grad = False
        for param in self.adapters.parameters():
            param.requires_grad = True

    def save_adapters(self, path: str):
        """Save adapters."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        torch.save({
            "adapters": self.adapters.state_dict(),
            "adapter_layers": self.adapter_layer_indices,
            "adapter_type": self.adapter_type,
            "hidden_size": self.hidden_size,
        }, path / "analog_adapters.pt")
        print(f"[AnalogLLM] Saved to {path}")

    def load_adapters(self, path: str):
        """Load adapters."""
        path = Path(path)
        # Try analog format first, fall back to v16 format
        if (path / "analog_adapters.pt").exists():
            ckpt = torch.load(path / "analog_adapters.pt")
        elif (path / "hardware_adapters.pt").exists():
            ckpt = torch.load(path / "hardware_adapters.pt")
            print("  [Note] Loaded v16 format adapters")
        else:
            raise FileNotFoundError(f"No adapter checkpoint found in {path}")

        self.adapters.load_state_dict(ckpt["adapters"])
        print(f"[AnalogLLM] Loaded from {path}")


# =============================================================================
# TRAINER
# =============================================================================

def train_analog(
    model_id: str = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
    data_path: str = "data/ouroboros/refined_train.jsonl",
    val_path: str = "data/ouroboros/refined_val.jsonl",
    output_dir: str = "models/analog_aware",
    resume_from: Optional[str] = None,
    epochs: int = 1,
    batch_size: int = 2,
    learning_rate: float = 5e-5,
    adapter_type: str = "analog",
    gradient_accumulation: int = 8,
    max_length: int = 512,
    use_wandb: bool = False,
    wandb_project: str = "feel-analog",
    wandb_run_name: str = None,
):
    """Train analog-aware adapters."""

    print("=" * 70)
    print("FEEL v18: ANALOG CONTROL TRAINING")
    print("=" * 70)
    print(f"Model:          {model_id}")
    print(f"Resume From:    {resume_from or 'Fresh Start'}")
    print(f"Data:           {data_path}")
    print(f"Output:         {output_dir}")
    print(f"Adapter:        {adapter_type}")
    print(f"Epochs:         {epochs}")
    print(f"Batch (eff):    {batch_size * gradient_accumulation}")
    print(f"Learning Rate:  {learning_rate}")
    print("=" * 70)

    # Wandb
    if use_wandb and WANDB_AVAILABLE:
        wandb.init(
            project=wandb_project,
            name=wandb_run_name or f"analog-{datetime.now().strftime('%Y%m%d_%H%M')}",
            config={
                "model_id": model_id,
                "adapter_type": adapter_type,
                "epochs": epochs,
                "learning_rate": learning_rate,
                "resume_from": resume_from,
            },
        )

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Create model
    print("\n[1/5] Creating Analog-Aware LLM...")
    model = AnalogAwareLLM(
        base_model_id=model_id,
        adapter_type=adapter_type,
        device=device,
    )

    # Load previous checkpoint if resuming
    if resume_from:
        print(f"\n[1.5/5] Loading checkpoint from {resume_from}...")
        model.load_adapters(resume_from)

    model.freeze_base_model()

    trainable = sum(p.numel() for p in model.get_adapter_parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.model.parameters())
    print(f"  Trainable: {trainable:,} / {total:,} ({100*trainable/total:.4f}%)")

    # Datasets
    print("\n[2/5] Loading datasets...")
    train_dataset = AnalogHardwareDataset(data_path, model.tokenizer, max_length)
    val_dataset = AnalogHardwareDataset(val_path, model.tokenizer, max_length)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=analog_collate_fn,
        num_workers=0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=analog_collate_fn,
    )

    # Optimizer
    print("\n[3/5] Setting up optimizer...")
    optimizer = torch.optim.AdamW(
        model.get_adapter_parameters(),
        lr=learning_rate,
        weight_decay=0.01,
    )

    total_steps = len(train_loader) * epochs // gradient_accumulation
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=total_steps,
        eta_min=learning_rate * 0.1,
    )

    # Training
    print("\n[4/5] Training with Analog Control...")
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    best_val_loss = float("inf")
    global_step = 0
    accumulated_loss = 0

    # Track stress distribution for debugging
    stress_histogram = [0] * 10

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        num_batches = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")

        for batch_idx, batch in enumerate(pbar):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            stress_levels = batch["stress_level"]

            batch_loss = 0
            for i in range(input_ids.size(0)):
                stress = stress_levels[i].item()

                # Track distribution
                bin_idx = min(9, int(stress * 10))
                stress_histogram[bin_idx] += 1

                loss = model.training_step(
                    input_ids[i:i+1],
                    labels[i:i+1],
                    stress_level=stress,
                    attention_mask=attention_mask[i:i+1],
                )
                batch_loss += loss

            batch_loss = batch_loss / input_ids.size(0) / gradient_accumulation
            batch_loss.backward()

            accumulated_loss += batch_loss.item()
            epoch_loss += batch_loss.item() * gradient_accumulation
            num_batches += 1

            if (batch_idx + 1) % gradient_accumulation == 0:
                torch.nn.utils.clip_grad_norm_(model.get_adapter_parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                global_step += 1
                lr = scheduler.get_last_lr()[0]
                pbar.set_postfix({"loss": f"{accumulated_loss:.4f}", "lr": f"{lr:.2e}"})

                if use_wandb and WANDB_AVAILABLE:
                    wandb.log({
                        "train/loss": accumulated_loss,
                        "train/lr": lr,
                        "train/step": global_step,
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
                    loss = model.training_step(
                        input_ids[i:i+1],
                        labels[i:i+1],
                        stress_level=stress_levels[i].item(),
                        attention_mask=attention_mask[i:i+1],
                    )
                    val_loss += loss.item()
                    val_batches += 1

        avg_val_loss = val_loss / val_batches

        print(f"\nEpoch {epoch+1}: Train={avg_train_loss:.4f}, Val={avg_val_loss:.4f}")

        # Print stress distribution
        print("  Stress Distribution: " + " ".join(f"{c}" for c in stress_histogram))

        if use_wandb and WANDB_AVAILABLE:
            wandb.log({
                "epoch/train_loss": avg_train_loss,
                "epoch/val_loss": avg_val_loss,
            })

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            model.save_adapters(output_path / "best")
            print(f"  -> New best! (val_loss={avg_val_loss:.4f})")

        model.save_adapters(output_path / f"epoch_{epoch+1}")

    # Final save
    print("\n[5/5] Saving final model...")
    model.save_adapters(output_path / "final")

    info = {
        "model_id": model_id,
        "adapter_type": adapter_type,
        "epochs": epochs,
        "learning_rate": learning_rate,
        "best_val_loss": best_val_loss,
        "resumed_from": resume_from,
        "timestamp": datetime.now().isoformat(),
    }
    with open(output_path / "training_info.json", "w") as f:
        json.dump(info, f, indent=2)

    if use_wandb and WANDB_AVAILABLE:
        wandb.finish()

    print("\n" + "=" * 70)
    print("ANALOG TRAINING COMPLETE")
    print("=" * 70)
    print(f"Best Val Loss: {best_val_loss:.4f}")
    print(f"Model saved:   {output_path}")
    print("\nRun z17_comprehensive_validation.py to verify S-Curve.")

    return model


def main():
    parser = argparse.ArgumentParser(description="FEEL v18 Analog Control Trainer")
    parser.add_argument("--model", type=str, default="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B")
    parser.add_argument("--data", type=str, default="data/ouroboros/refined_train.jsonl")
    parser.add_argument("--val", type=str, default="data/ouroboros/refined_val.jsonl")
    parser.add_argument("--output", type=str, default="models/analog_aware")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint dir")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--adapter", type=str, default="analog", choices=["analog", "analog_gated", "analog_multiscale"])
    parser.add_argument("--gradient-accumulation", type=int, default=8)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project", type=str, default="feel-analog")
    args = parser.parse_args()

    train_analog(
        model_id=args.model,
        data_path=args.data,
        val_path=args.val,
        output_dir=args.output,
        resume_from=args.resume,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        adapter_type=args.adapter,
        gradient_accumulation=args.gradient_accumulation,
        use_wandb=args.wandb,
        wandb_project=args.wandb_project,
    )


if __name__ == "__main__":
    main()
