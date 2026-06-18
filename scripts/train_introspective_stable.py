#!/usr/bin/env python3
"""
Stable Introspective Fine-Tuning (IFT) Training Script

This fixes the NaN losses from the original train_introspective.py by:
1. bf16-first training (better numerical stability than fp16)
2. Bounded FiLM modulation (tanh to prevent explosion)
3. Identity-safe initialization (starts near identity transform)
4. Proper gradient scaling for mixed precision

The goal: teach the model to recognize its own internal state WITHOUT
explicit prompting, using stable numerics that actually converge.

Usage:
    python scripts/train_introspective_stable.py \
        --model Qwen/Qwen2.5-1.5B \
        --dataset data/ift_dataset.json \
        --output models/ift_stable \
        --dtype bf16
"""

import sys
import json
import random
import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


@dataclass
class StableIFTConfig:
    """Configuration for stable introspective fine-tuning."""
    model_name: str = "Qwen/Qwen2.5-1.5B"
    use_lora: bool = True
    lora_r: int = 16
    lora_alpha: int = 32

    # Training
    batch_size: int = 2
    learning_rate: float = 5e-5
    num_epochs: int = 3
    warmup_steps: int = 50
    max_length: int = 512
    gradient_accumulation_steps: int = 4

    # z_feel injection
    z_feel_dim: int = 8

    # Numerical stability
    dtype: str = "bf16"  # bf16 or fp32
    max_grad_norm: float = 1.0

    # Bounded FiLM
    film_scale_bound: float = 0.3  # tanh output scaled by this
    film_shift_bound: float = 0.1  # tanh output scaled by this


class BoundedFiLM(nn.Module):
    """
    Bounded Feature-wise Linear Modulation.

    Key stability features:
    - tanh activation to bound outputs
    - Small scale factors to prevent explosion
    - Identity-safe initialization
    """

    def __init__(
        self,
        z_dim: int,
        hidden_dim: int,
        scale_bound: float = 0.3,
        shift_bound: float = 0.1,
        dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__()

        self.scale_bound = scale_bound
        self.shift_bound = shift_bound

        # Projection layers
        self.gamma_proj = nn.Linear(z_dim, hidden_dim, dtype=dtype)
        self.beta_proj = nn.Linear(z_dim, hidden_dim, dtype=dtype)

        # Identity-safe initialization
        self._init_identity_safe()

    def _init_identity_safe(self):
        """Initialize near identity transform."""
        with torch.no_grad():
            # Small random init for gamma (will be scaled by tanh * scale_bound)
            nn.init.normal_(self.gamma_proj.weight, mean=0.0, std=0.01)
            nn.init.zeros_(self.gamma_proj.bias)

            # Small random init for beta
            nn.init.normal_(self.beta_proj.weight, mean=0.0, std=0.01)
            nn.init.zeros_(self.beta_proj.bias)

    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute bounded gamma and beta.

        Returns:
            gamma: Scale factor, bounded to [1 - scale_bound, 1 + scale_bound]
            beta: Shift factor, bounded to [-shift_bound, shift_bound]
        """
        # Project z to hidden dim
        gamma_raw = self.gamma_proj(z)
        beta_raw = self.beta_proj(z)

        # Bound with tanh and scale
        # gamma = 1 + tanh(raw) * scale_bound  → [1-bound, 1+bound]
        gamma = 1.0 + torch.tanh(gamma_raw) * self.scale_bound

        # beta = tanh(raw) * shift_bound → [-bound, bound]
        beta = torch.tanh(beta_raw) * self.shift_bound

        return gamma, beta


class StableFiLMInjector:
    """
    Stable z_feel injection via bounded FiLM.

    Key differences from original:
    - Uses BoundedFiLM for numerical stability
    - bf16 by default
    - Proper hook cleanup
    """

    def __init__(
        self,
        model: nn.Module,
        config: StableIFTConfig,
    ):
        self.model = model
        self.config = config
        self.device = next(model.parameters()).device

        # Determine dtype
        if config.dtype == "bf16":
            self.dtype = torch.bfloat16
        elif config.dtype == "fp32":
            self.dtype = torch.float32
        else:
            self.dtype = torch.float16

        # Get hidden dim
        hidden_dim = model.config.hidden_size

        # Create bounded FiLM
        self.film = BoundedFiLM(
            z_dim=config.z_feel_dim,
            hidden_dim=hidden_dim,
            scale_bound=config.film_scale_bound,
            shift_bound=config.film_shift_bound,
            dtype=self.dtype,
        ).to(self.device)

        # Determine target layers (middle third)
        n_layers = len(model.model.layers)
        start = n_layers // 3
        end = 2 * n_layers // 3
        self.target_layers = list(range(start, end))

        self._hooks = []
        self._current_z = None

    def generate_z_feel(self, condition: str) -> torch.Tensor:
        """Generate z_feel tensor for condition."""
        z = torch.zeros(self.config.z_feel_dim, device=self.device, dtype=self.dtype)

        if condition == "hot_focused":
            z[0:4] = torch.rand(4, device=self.device, dtype=self.dtype) * 0.4 + 0.6
        elif condition == "memory_fragmented":
            z[4:8] = torch.rand(4, device=self.device, dtype=self.dtype) * 0.4 + 0.6
        elif condition == "critical":
            z[0:4] = torch.rand(4, device=self.device, dtype=self.dtype) * 0.4 + 0.6
            z[4:8] = torch.rand(4, device=self.device, dtype=self.dtype) * 0.4 + 0.6
        elif condition == "warm":
            z[0:4] = torch.rand(4, device=self.device, dtype=self.dtype) * 0.3 + 0.3
        elif condition == "very_hot":
            z[0:4] = torch.rand(4, device=self.device, dtype=self.dtype) * 0.2 + 0.8
        else:  # cool_clear
            z = torch.rand(self.config.z_feel_dim, device=self.device, dtype=self.dtype) * 0.3

        return z

    def _create_hook(self, gamma: torch.Tensor, beta: torch.Tensor):
        """Create modulation hook with pre-computed gamma/beta."""
        def hook(module, input, output):
            if isinstance(output, tuple):
                hidden = output[0]
                # FiLM: h' = gamma * h + beta
                modulated = gamma.unsqueeze(0).unsqueeze(1) * hidden + beta.unsqueeze(0).unsqueeze(1)
                return (modulated,) + output[1:]
            else:
                return gamma.unsqueeze(0).unsqueeze(1) * output + beta.unsqueeze(0).unsqueeze(1)
        return hook

    def activate(self, condition: str):
        """Activate FiLM injection for condition."""
        self.deactivate()

        z = self.generate_z_feel(condition)
        self._current_z = z

        # Compute gamma/beta
        gamma, beta = self.film(z)

        # Register hooks
        hook = self._create_hook(gamma, beta)
        for layer_idx in self.target_layers:
            layer = self.model.model.layers[layer_idx]
            handle = layer.register_forward_hook(hook)
            self._hooks.append(handle)

    def deactivate(self):
        """Remove all hooks."""
        for handle in self._hooks:
            handle.remove()
        self._hooks.clear()
        self._current_z = None

    def get_parameters(self):
        """Return FiLM parameters for optimizer."""
        return list(self.film.parameters())


class StableIFTTrainer:
    """Stable trainer for Introspective Fine-Tuning."""

    def __init__(self, config: StableIFTConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Determine dtype
        if config.dtype == "bf16":
            self.dtype = torch.bfloat16
        elif config.dtype == "fp32":
            self.dtype = torch.float32
        else:
            self.dtype = torch.float16

        self._load_model()
        self._setup_film()

        if config.use_lora:
            self._setup_lora()

    def _load_model(self):
        """Load model and tokenizer."""
        from transformers import AutoModelForCausalLM, AutoTokenizer

        print(f"Loading model: {self.config.model_name}")
        print(f"Using dtype: {self.config.dtype}")

        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            torch_dtype=self.dtype,
            device_map="auto",
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def _setup_film(self):
        """Setup stable FiLM injector."""
        self.injector = StableFiLMInjector(self.model, self.config)
        print(f"FiLM injector ready, targeting layers {self.injector.target_layers}")

    def _setup_lora(self):
        """Setup LoRA for efficient fine-tuning."""
        try:
            from peft import get_peft_model, LoraConfig, TaskType

            lora_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=self.config.lora_r,
                lora_alpha=self.config.lora_alpha,
                lora_dropout=0.05,
                target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
            )

            self.model = get_peft_model(self.model, lora_config)
            print("LoRA enabled")

        except ImportError:
            print("PEFT not installed, training full model")

    def load_dataset(self, path: str) -> List[Dict]:
        """Load dataset."""
        with open(path) as f:
            data = json.load(f)
        return data["examples"]

    def prepare_batch(self, examples: List[Dict]) -> Tuple:
        """Prepare training batch."""
        texts = []
        conditions = []

        for ex in examples:
            prompt = ex["prompt"]
            target = ex["target_response"]
            condition = ex["condition"]

            full_text = f"<|user|>\n{prompt}\n<|assistant|>\n{target}"
            texts.append(full_text)
            conditions.append(condition)

        encodings = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.config.max_length,
            return_tensors="pt",
        )

        input_ids = encodings["input_ids"].to(self.device)
        attention_mask = encodings["attention_mask"].to(self.device)

        labels = input_ids.clone()
        labels[labels == self.tokenizer.pad_token_id] = -100

        return input_ids, attention_mask, labels, conditions

    def train_step(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
        conditions: List[str],
        optimizer: torch.optim.Optimizer,
        scaler: Optional[torch.cuda.amp.GradScaler] = None,
    ) -> float:
        """Single training step with stable numerics."""
        total_loss = 0.0
        valid_samples = 0

        for i in range(len(conditions)):
            condition = conditions[i]

            # Activate FiLM for this condition
            self.injector.activate(condition)

            try:
                # Forward pass
                outputs = self.model(
                    input_ids=input_ids[i:i+1],
                    attention_mask=attention_mask[i:i+1],
                    labels=labels[i:i+1],
                )

                loss = outputs.loss

                # Check for NaN
                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"  Warning: NaN/Inf loss for condition {condition}, skipping")
                    self.injector.deactivate()
                    continue

                # Backward
                if scaler is not None:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()

                total_loss += loss.item()
                valid_samples += 1

            except RuntimeError as e:
                print(f"  Warning: Runtime error for condition {condition}: {e}")

            finally:
                self.injector.deactivate()

        if valid_samples == 0:
            return float('nan')

        # Gradient clipping
        if scaler is not None:
            scaler.unscale_(optimizer)

        torch.nn.utils.clip_grad_norm_(
            list(self.model.parameters()) + self.injector.get_parameters(),
            max_norm=self.config.max_grad_norm
        )

        # Optimizer step
        if scaler is not None:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()

        optimizer.zero_grad()

        return total_loss / valid_samples

    def train(self, dataset_path: str, output_dir: str):
        """Run full training loop."""
        print(f"\n{'='*60}")
        print("  STABLE INTROSPECTIVE FINE-TUNING")
        print(f"  dtype: {self.config.dtype}, bounded FiLM, identity-safe init")
        print(f"{'='*60}\n")

        examples = self.load_dataset(dataset_path)
        print(f"Loaded {len(examples)} examples")

        # Setup optimizer
        params = list(self.model.parameters()) + self.injector.get_parameters()
        optimizer = torch.optim.AdamW(params, lr=self.config.learning_rate)

        # Setup gradient scaler for fp16 (not needed for bf16)
        scaler = None
        if self.config.dtype == "fp16":
            scaler = torch.cuda.amp.GradScaler()

        # Training loop
        for epoch in range(self.config.num_epochs):
            print(f"\nEpoch {epoch + 1}/{self.config.num_epochs}")

            random.shuffle(examples)

            epoch_loss = 0.0
            n_batches = 0
            nan_count = 0

            pbar = tqdm(range(0, len(examples), self.config.batch_size))
            for i in pbar:
                batch = examples[i:i + self.config.batch_size]

                input_ids, attention_mask, labels, conditions = self.prepare_batch(batch)

                loss = self.train_step(
                    input_ids, attention_mask, labels, conditions,
                    optimizer, scaler
                )

                if not torch.isnan(torch.tensor(loss)):
                    epoch_loss += loss
                    n_batches += 1
                    pbar.set_postfix({"loss": f"{loss:.4f}"})
                else:
                    nan_count += 1

            if n_batches > 0:
                avg_loss = epoch_loss / n_batches
                print(f"  Average loss: {avg_loss:.4f} (NaN batches: {nan_count})")
            else:
                print(f"  All batches had NaN loss!")

        # Save
        self._save(output_dir)

    def _save(self, output_dir: str):
        """Save model and FiLM parameters."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        self.model.save_pretrained(output_path)
        self.tokenizer.save_pretrained(output_path)

        # Save FiLM
        torch.save({
            "film_state_dict": self.injector.film.state_dict(),
            "config": {
                "z_dim": self.config.z_feel_dim,
                "scale_bound": self.config.film_scale_bound,
                "shift_bound": self.config.film_shift_bound,
            }
        }, output_path / "film_params.pt")

        print(f"\nModel saved to: {output_path}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Stable IFT Training")
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B")
    parser.add_argument("--dataset", default="data/ift_dataset.json")
    parser.add_argument("--output", default="models/ift_stable")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--no-lora", action="store_true")

    args = parser.parse_args()

    config = StableIFTConfig(
        model_name=args.model,
        use_lora=not args.no_lora,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        dtype=args.dtype,
    )

    trainer = StableIFTTrainer(config)
    trainer.train(args.dataset, args.output)


if __name__ == "__main__":
    main()
