#!/usr/bin/env python3
"""
Hook-Free Introspective Fine-Tuning with Additive Embedding Injection

This training script avoids the ROCm crash by NOT using forward hooks.
Instead, it injects the bodily signal by adding a bounded, learned offset
to the input embeddings.

Architecture:
    input_ids → embeddings → embeddings + z_offset → model forward

The z_offset is computed from z_feel via a small projection network.
No hooks means no HSA_STATUS_ERROR_EXCEPTION on gfx1151.

This is "clean" in the embodied sense:
- The model only gets a continuous non-semantic perturbation
- It must LEARN to map this perturbation to language
- No concept injection, no steering vectors

Usage:
    python scripts/train_introspective_additive.py \
        --model Qwen/Qwen2.5-1.5B \
        --dataset data/ift_dataset_with_actions.json \
        --output models/ift_additive \
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

from action_tokens import add_feel_tokens_to_tokenizer, resize_model_embeddings


@dataclass
class AdditiveIFTConfig:
    """Configuration for additive (hook-free) IFT."""
    model_name: str = "Qwen/Qwen2.5-1.5B"
    use_lora: bool = True
    lora_r: int = 16
    lora_alpha: int = 32

    # Training
    batch_size: int = 2
    learning_rate: float = 5e-5
    num_epochs: int = 3
    max_length: int = 512
    gradient_accumulation_steps: int = 4

    # z_feel
    z_feel_dim: int = 8

    # Additive injection
    injection_scale: float = 0.05  # Scale of the additive offset

    # Numerical stability
    dtype: str = "bf16"
    max_grad_norm: float = 1.0


class AdditiveZFeelInjector(nn.Module):
    """
    Additive z_feel injection - NO HOOKS.

    Instead of using forward hooks (which crash on ROCm),
    we compute an offset to add to the input embeddings.

    The offset is:
        z_offset = scale * tanh(proj(z_feel))

    This is bounded and smooth, preventing numerical issues.
    """

    def __init__(
        self,
        z_dim: int,
        embed_dim: int,
        scale: float = 0.05,
        dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__()

        self.scale = scale

        # Project z_feel to embedding dimension
        self.proj = nn.Sequential(
            nn.Linear(z_dim, embed_dim // 4, dtype=dtype),
            nn.GELU(),
            nn.Linear(embed_dim // 4, embed_dim, dtype=dtype),
        )

        # Initialize small
        self._init_small()

    def _init_small(self):
        """Initialize with small weights for stability."""
        with torch.no_grad():
            for layer in self.proj:
                if isinstance(layer, nn.Linear):
                    nn.init.normal_(layer.weight, mean=0.0, std=0.01)
                    nn.init.zeros_(layer.bias)

    def forward(self, z_feel: torch.Tensor) -> torch.Tensor:
        """
        Compute additive offset from z_feel.

        Args:
            z_feel: [z_dim] tensor

        Returns:
            offset: [embed_dim] tensor to add to embeddings
        """
        raw = self.proj(z_feel)
        # Bound with tanh and scale
        offset = self.scale * torch.tanh(raw)
        return offset


class AdditiveIFTTrainer:
    """
    Hook-free trainer using additive embedding injection.

    This is safe for ROCm/gfx1151 because it doesn't use forward hooks.
    """

    def __init__(self, config: AdditiveIFTConfig):
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
        self._setup_injector()

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

        # Add action tokens
        num_added = add_feel_tokens_to_tokenizer(self.tokenizer)
        if num_added > 0:
            resize_model_embeddings(self.model, self.tokenizer)

    def _setup_injector(self):
        """Setup additive z_feel injector."""
        embed_dim = self.model.config.hidden_size

        self.injector = AdditiveZFeelInjector(
            z_dim=self.config.z_feel_dim,
            embed_dim=embed_dim,
            scale=self.config.injection_scale,
            dtype=self.dtype,
        ).to(self.device)

        print(f"Additive injector ready (scale={self.config.injection_scale})")

    def _setup_lora(self):
        """Setup LoRA."""
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

    def generate_z_feel(self, condition: str) -> torch.Tensor:
        """Generate z_feel for condition."""
        z = torch.zeros(self.config.z_feel_dim, device=self.device, dtype=self.dtype)

        if condition == "hot_focused":
            z[0:4] = torch.rand(4, device=self.device, dtype=self.dtype) * 0.4 + 0.6
        elif condition == "memory_fragmented":
            z[4:8] = torch.rand(4, device=self.device, dtype=self.dtype) * 0.4 + 0.6
        elif condition == "critical":
            z[:] = torch.rand(8, device=self.device, dtype=self.dtype) * 0.4 + 0.6
        elif condition == "warm":
            z[0:4] = torch.rand(4, device=self.device, dtype=self.dtype) * 0.3 + 0.3
        elif condition == "very_hot":
            z[0:4] = torch.rand(4, device=self.device, dtype=self.dtype) * 0.2 + 0.8
        else:  # cool_clear
            z = torch.rand(self.config.z_feel_dim, device=self.device, dtype=self.dtype) * 0.3

        return z

    def forward_with_injection(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
        z_feel: torch.Tensor,
    ):
        """
        Forward pass with additive z_feel injection.

        NO HOOKS - just embedding manipulation.
        """
        # Get embeddings - handle different model structures
        if hasattr(self.model, 'base_model'):
            # PEFT model - navigate through base_model
            base = self.model.base_model
            if hasattr(base, 'model') and hasattr(base.model, 'embed_tokens'):
                embed_layer = base.model.embed_tokens
            elif hasattr(base, 'model') and hasattr(base.model, 'model'):
                embed_layer = base.model.model.embed_tokens
            else:
                embed_layer = self.model.get_input_embeddings()
        else:
            # Standard model
            embed_layer = self.model.get_input_embeddings()

        embeddings = embed_layer(input_ids)

        # Compute additive offset
        offset = self.injector(z_feel)  # [embed_dim]

        # Add offset to all token embeddings
        # offset: [embed_dim] → [1, 1, embed_dim]
        injected_embeddings = embeddings + offset.unsqueeze(0).unsqueeze(0)

        # Forward with injected embeddings
        outputs = self.model(
            inputs_embeds=injected_embeddings,
            attention_mask=attention_mask,
            labels=labels,
        )

        return outputs

    def load_dataset(self, path: str) -> List[Dict]:
        """Load dataset."""
        with open(path) as f:
            data = json.load(f)
        return data["examples"]

    def prepare_example(self, example: Dict) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, str]:
        """Prepare single example."""
        prompt = example["prompt"]
        target = example["target_response"]
        condition = example["condition"]

        full_text = f"<|user|>\n{prompt}\n<|assistant|>\n{target}"

        encodings = self.tokenizer(
            full_text,
            truncation=True,
            max_length=self.config.max_length,
            return_tensors="pt",
        )

        input_ids = encodings["input_ids"].to(self.device)
        attention_mask = encodings["attention_mask"].to(self.device)

        labels = input_ids.clone()
        labels[labels == self.tokenizer.pad_token_id] = -100

        return input_ids, attention_mask, labels, condition

    def train_step(
        self,
        examples: List[Dict],
        optimizer: torch.optim.Optimizer,
    ) -> float:
        """Training step over batch of examples."""
        total_loss = 0.0
        valid_samples = 0

        for example in examples:
            input_ids, attention_mask, labels, condition = self.prepare_example(example)

            # Generate z_feel for this condition
            z_feel = self.generate_z_feel(condition)

            try:
                # Forward with injection (NO HOOKS!)
                outputs = self.forward_with_injection(
                    input_ids, attention_mask, labels, z_feel
                )

                loss = outputs.loss

                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"  Warning: NaN/Inf loss for {condition}, skipping")
                    continue

                loss.backward()
                total_loss += loss.item()
                valid_samples += 1

            except RuntimeError as e:
                print(f"  Warning: Error for {condition}: {e}")
                continue

        if valid_samples == 0:
            return float('nan')

        # Gradient clipping
        all_params = list(self.model.parameters()) + list(self.injector.parameters())
        torch.nn.utils.clip_grad_norm_(all_params, max_norm=self.config.max_grad_norm)

        # Optimizer step
        optimizer.step()
        optimizer.zero_grad()

        return total_loss / valid_samples

    def train(self, dataset_path: str, output_dir: str):
        """Full training loop."""
        print(f"\n{'='*60}")
        print("  ADDITIVE (HOOK-FREE) INTROSPECTIVE FINE-TUNING")
        print("  Safe for ROCm/gfx1151 - no forward hooks")
        print(f"{'='*60}\n")

        examples = self.load_dataset(dataset_path)
        print(f"Loaded {len(examples)} examples")

        # Optimizer includes injector params
        all_params = list(self.model.parameters()) + list(self.injector.parameters())
        optimizer = torch.optim.AdamW(all_params, lr=self.config.learning_rate)

        # Training
        for epoch in range(self.config.num_epochs):
            print(f"\nEpoch {epoch + 1}/{self.config.num_epochs}")

            random.shuffle(examples)

            epoch_loss = 0.0
            n_batches = 0

            pbar = tqdm(range(0, len(examples), self.config.batch_size))
            for i in pbar:
                batch = examples[i:i + self.config.batch_size]

                loss = self.train_step(batch, optimizer)

                if not torch.isnan(torch.tensor(loss)):
                    epoch_loss += loss
                    n_batches += 1
                    pbar.set_postfix({"loss": f"{loss:.4f}"})

            if n_batches > 0:
                avg_loss = epoch_loss / n_batches
                print(f"  Average loss: {avg_loss:.4f}")

        # Save
        self._save(output_dir)

    def _save(self, output_dir: str):
        """Save model and injector."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        self.model.save_pretrained(output_path)
        self.tokenizer.save_pretrained(output_path)

        # Save additive injector
        torch.save({
            "injector_state_dict": self.injector.state_dict(),
            "config": {
                "z_dim": self.config.z_feel_dim,
                "scale": self.config.injection_scale,
            }
        }, output_path / "additive_injector.pt")

        print(f"\nModel saved to: {output_path}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Additive (Hook-Free) IFT Training")
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B")
    parser.add_argument("--dataset", default="data/ift_dataset_with_actions.json")
    parser.add_argument("--output", default="models/ift_additive")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--scale", type=float, default=0.05,
                       help="Scale of additive injection")
    parser.add_argument("--no-lora", action="store_true")

    args = parser.parse_args()

    config = AdditiveIFTConfig(
        model_name=args.model,
        use_lora=not args.no_lora,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        dtype=args.dtype,
        injection_scale=args.scale,
    )

    trainer = AdditiveIFTTrainer(config)
    trainer.train(args.dataset, args.output)


if __name__ == "__main__":
    main()
