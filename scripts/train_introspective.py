#!/usr/bin/env python3
"""
Introspective Fine-Tuning (IFT) Training Script

This trains the model to recognize its own internal state WITHOUT explicit prompting.
The key: during training, we DYNAMICALLY inject z_feel (via FiLM), and the model
must learn to diagnose itself based on how its hidden states "feel".

The Training Loop:
1. Load a prompt
2. RANDOMLY set z_feel to HOT, COLD, MEMORY_FULL, or MEMORY_OK
3. Forward pass with FiLM active (warping hidden states)
4. Compute loss against the CONDITION-APPROPRIATE target
5. Model learns: "When my hidden states warp in direction X, I should say 'I am hot'"

This is MACHINE PROPRIOCEPTION - the model builds a semantic map of its physical body.

Usage:
    python scripts/train_introspective.py \
        --model Qwen/Qwen2.5-1.5B \
        --dataset data/ift_dataset.json \
        --output models/ift_lora
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

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


@dataclass
class IFTConfig:
    """Configuration for introspective fine-tuning."""
    # Model
    model_name: str = "Qwen/Qwen2.5-1.5B"
    use_lora: bool = True
    lora_r: int = 16
    lora_alpha: int = 32

    # Training
    batch_size: int = 4
    learning_rate: float = 2e-5
    num_epochs: int = 3
    warmup_steps: int = 100
    max_length: int = 512

    # z_feel injection
    z_feel_dim: int = 8
    film_layers: List[int] = None  # Which layers to apply FiLM

    # Condition simulation
    hot_z_range: Tuple[float, float] = (0.6, 1.0)
    cold_z_range: Tuple[float, float] = (0.0, 0.3)
    memory_full_z_range: Tuple[float, float] = (0.6, 1.0)
    memory_ok_z_range: Tuple[float, float] = (0.0, 0.3)


class DynamicZFeelInjector:
    """
    Injects z_feel dynamically during training based on the target condition.

    This creates the "dizzy" effect - the model's hidden states are warped
    by z_feel, and it must learn to recognize which type of warping it's experiencing.
    """

    def __init__(
        self,
        model: nn.Module,
        z_feel_dim: int = 8,
        target_layers: Optional[List[int]] = None,
    ):
        self.model = model
        self.z_feel_dim = z_feel_dim
        self.device = next(model.parameters()).device

        # Determine target layers (middle layers are most effective)
        if target_layers is None:
            n_layers = len(model.model.layers) if hasattr(model, 'model') else 24
            # Apply to middle third of layers
            start = n_layers // 3
            end = 2 * n_layers // 3
            self.target_layers = list(range(start, end))
        else:
            self.target_layers = target_layers

        # Create FiLM parameters in FLOAT32 (full precision) to avoid NaN gradients
        # The model stays in half precision, but FiLM training needs stability
        hidden_dim = model.config.hidden_size
        self.model_dtype = next(model.parameters()).dtype  # Save for later conversion

        # FiLM in float32 for stable gradients
        self.film_gamma = nn.Linear(z_feel_dim, hidden_dim).to(self.device, dtype=torch.float32)
        self.film_beta = nn.Linear(z_feel_dim, hidden_dim).to(self.device, dtype=torch.float32)

        # Initialize near-identity with small perturbation for learning signal
        with torch.no_grad():
            nn.init.normal_(self.film_gamma.weight, mean=1.0, std=0.01)
            nn.init.zeros_(self.film_gamma.bias)
            nn.init.normal_(self.film_beta.weight, mean=0.0, std=0.01)
            nn.init.zeros_(self.film_beta.bias)

        self._hooks = []
        self._current_z_feel = None

    def generate_z_feel(self, condition: str) -> torch.Tensor:
        """
        Generate z_feel tensor based on target condition.

        Different conditions create different z_feel patterns:
        - HOT: High values in thermal dimensions (0-3)
        - MEMORY: High values in memory dimensions (4-7)
        """
        # Use float32 for FiLM computation stability
        z = torch.zeros(self.z_feel_dim, device=self.device, dtype=torch.float32)

        if condition == "hot_focused":
            # Thermal stress: high in dims 0-3
            z[0:4] = torch.rand(4, device=self.device, dtype=torch.float32) * 0.4 + 0.6

        elif condition == "memory_fragmented":
            # Memory stress: high in dims 4-7
            z[4:8] = torch.rand(4, device=self.device, dtype=torch.float32) * 0.4 + 0.6

        elif condition == "critical":
            # Both stressors
            z[0:4] = torch.rand(4, device=self.device, dtype=torch.float32) * 0.4 + 0.6
            z[4:8] = torch.rand(4, device=self.device, dtype=torch.float32) * 0.4 + 0.6

        else:  # cool_clear
            # Normal operation: low across all dims
            z = torch.rand(self.z_feel_dim, device=self.device, dtype=torch.float32) * 0.3

        return z

    def _create_film_hook(self, z_feel: torch.Tensor):
        """Create FiLM modulation hook."""
        # Compute in float32 for stability
        gamma = self.film_gamma(z_feel)  # [hidden_dim] in float32
        beta = self.film_beta(z_feel)    # [hidden_dim] in float32

        # Convert to model dtype for application
        gamma_cast = gamma.to(self.model_dtype)
        beta_cast = beta.to(self.model_dtype)

        def hook(module, input, output):
            if isinstance(output, tuple):
                hidden = output[0]
                # FiLM: h' = gamma * h + beta (all in model dtype now)
                modulated = gamma_cast.unsqueeze(0).unsqueeze(1) * hidden + beta_cast.unsqueeze(0).unsqueeze(1)
                return (modulated,) + output[1:]
            else:
                return gamma_cast.unsqueeze(0).unsqueeze(1) * output + beta_cast.unsqueeze(0).unsqueeze(1)

        return hook

    def activate(self, condition: str):
        """Activate z_feel injection for a condition."""
        self.deactivate()  # Clear any existing hooks

        z_feel = self.generate_z_feel(condition)
        self._current_z_feel = z_feel

        hook = self._create_film_hook(z_feel)

        # Register hooks on target layers
        for layer_idx in self.target_layers:
            try:
                layer = self.model.model.layers[layer_idx]
            except:
                layer = self.model.transformer.h[layer_idx]

            handle = layer.register_forward_hook(hook)
            self._hooks.append(handle)

    def deactivate(self):
        """Remove all injection hooks."""
        for handle in self._hooks:
            handle.remove()
        self._hooks.clear()
        self._current_z_feel = None

    def get_film_parameters(self):
        """Return FiLM parameters for optimizer."""
        return list(self.film_gamma.parameters()) + list(self.film_beta.parameters())


class IFTTrainer:
    """
    Trainer for Introspective Fine-Tuning.

    This implements the "Mirror Training" approach:
    1. Randomly select a condition (HOT, COLD, MEMORY, OK)
    2. Inject z_feel to warp the model's hidden states
    3. Train to produce condition-appropriate self-diagnosis
    """

    def __init__(
        self,
        config: IFTConfig,
        model=None,
        tokenizer=None,
    ):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load model and tokenizer
        if model is None:
            from transformers import AutoModelForCausalLM, AutoTokenizer

            print(f"Loading model: {config.model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                config.model_name,
                torch_dtype=torch.float16,
                device_map="auto",
            )

            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
        else:
            self.model = model
            self.tokenizer = tokenizer

        # Setup LoRA if enabled
        if config.use_lora:
            self._setup_lora()

        # Create z_feel injector
        self.injector = DynamicZFeelInjector(
            self.model,
            z_feel_dim=config.z_feel_dim,
        )

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
            print("PEFT not installed. Training full model.")

    def load_dataset(self, path: str) -> List[Dict]:
        """Load the IFT dataset."""
        with open(path) as f:
            data = json.load(f)
        return data["examples"]

    def prepare_batch(
        self,
        examples: List[Dict],
    ) -> Tuple[torch.Tensor, torch.Tensor, List[str]]:
        """
        Prepare a training batch with dynamic z_feel conditions.

        Returns input_ids, labels, and conditions for z_feel injection.
        """
        prompts = []
        targets = []
        conditions = []

        for ex in examples:
            prompt = ex["prompt"]
            condition = ex["condition"]
            target = ex["target_response"]

            # Format as conversation
            full_text = f"<|user|>\n{prompt}\n<|assistant|>\n{target}"

            prompts.append(full_text)
            conditions.append(condition)

        # Tokenize
        encodings = self.tokenizer(
            prompts,
            padding=True,
            truncation=True,
            max_length=self.config.max_length,
            return_tensors="pt",
        )

        input_ids = encodings["input_ids"].to(self.device)
        attention_mask = encodings["attention_mask"].to(self.device)

        # Labels are same as input_ids for causal LM
        labels = input_ids.clone()
        # Mask padding
        labels[labels == self.tokenizer.pad_token_id] = -100

        return input_ids, attention_mask, labels, conditions

    def train_step(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
        conditions: List[str],
        optimizer: torch.optim.Optimizer,
    ) -> float:
        """
        Single training step with dynamic z_feel injection.

        The key: we activate z_feel based on the target condition BEFORE
        the forward pass, so the model experiences warped hidden states.
        """
        total_loss = 0.0

        # Process each example with its condition
        # (In practice, would batch by condition for efficiency)
        for i in range(len(conditions)):
            condition = conditions[i]

            # Activate z_feel for this condition
            self.injector.activate(condition)

            # Forward pass with z_feel active
            outputs = self.model(
                input_ids=input_ids[i:i+1],
                attention_mask=attention_mask[i:i+1],
                labels=labels[i:i+1],
            )

            loss = outputs.loss
            total_loss += loss.item()

            # Backward
            loss.backward()

            # Deactivate before next example
            self.injector.deactivate()

        # Gradient clipping to prevent NaN
        torch.nn.utils.clip_grad_norm_(
            list(self.model.parameters()) + self.injector.get_film_parameters(),
            max_norm=1.0
        )

        # Optimizer step
        optimizer.step()
        optimizer.zero_grad()

        return total_loss / len(conditions)

    def train(
        self,
        dataset_path: str,
        output_dir: str = "models/ift_lora",
    ):
        """
        Run full training loop.
        """
        print(f"\n{'='*60}")
        print("  INTROSPECTIVE FINE-TUNING")
        print("  Teaching the model to recognize its own internal state")
        print(f"{'='*60}\n")

        # Load dataset
        examples = self.load_dataset(dataset_path)
        print(f"Loaded {len(examples)} examples")

        # Setup optimizer (include FiLM parameters)
        params = list(self.model.parameters()) + self.injector.get_film_parameters()
        optimizer = torch.optim.AdamW(params, lr=self.config.learning_rate)

        # Training loop
        for epoch in range(self.config.num_epochs):
            print(f"\nEpoch {epoch + 1}/{self.config.num_epochs}")

            # Shuffle
            random.shuffle(examples)

            # Batch
            epoch_loss = 0.0
            n_batches = 0

            for i in tqdm(range(0, len(examples), self.config.batch_size)):
                batch = examples[i:i + self.config.batch_size]

                input_ids, attention_mask, labels, conditions = self.prepare_batch(batch)

                loss = self.train_step(
                    input_ids, attention_mask, labels, conditions, optimizer
                )

                epoch_loss += loss
                n_batches += 1

            avg_loss = epoch_loss / n_batches
            print(f"  Average loss: {avg_loss:.4f}")

        # Save
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        self.model.save_pretrained(output_path)
        self.tokenizer.save_pretrained(output_path)

        # Save FiLM parameters
        torch.save({
            "film_gamma": self.injector.film_gamma.state_dict(),
            "film_beta": self.injector.film_beta.state_dict(),
        }, output_path / "film_params.pt")

        print(f"\nModel saved to: {output_path}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Introspective Fine-Tuning")
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B")
    parser.add_argument("--dataset", default="data/ift_dataset.json")
    parser.add_argument("--output", default="models/ift_lora")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--no-lora", action="store_true")

    args = parser.parse_args()

    config = IFTConfig(
        model_name=args.model,
        use_lora=not args.no_lora,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
    )

    trainer = IFTTrainer(config)
    trainer.train(args.dataset, args.output)


if __name__ == "__main__":
    main()
