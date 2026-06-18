#!/usr/bin/env python3
"""
FEEL v4.5: Frozen Model Training with Action Classifier

Key insight from v4.4 failure: Fine-tuning the LM on small synthetic datasets
causes catastrophic forgetting. The model loses language ability.

Solution: Keep model FROZEN. Train only:
1. AdditiveZFeelInjector - maps z_feel → embedding offset
2. ActionClassifierHead - predicts action from hidden states + z_feel

Benefits:
- Model retains full language capability
- Injector learns embodied sensitivity
- Classifier enables action prediction without token generation
- No catastrophic forgetting

Architecture:
    z_feel → Injector → embedding_offset
    embeddings + offset → FROZEN MODEL → hidden_states
    hidden_states + z_feel → ActionClassifier → action_logits

Usage:
    python scripts/train_frozen_v4_5.py \
        --model Qwen/Qwen2.5-1.5B \
        --output models/feel_v4_5 \
        --epochs 10 \
        --dtype bf16
"""

import sys
import json
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from tqdm import tqdm
from enum import Enum

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from action_tokens import FeelAction, add_feel_tokens_to_tokenizer


class ActionLabel(Enum):
    """Action labels for classifier (matches FeelAction)."""
    OK = 0
    WARM = 1
    HOT = 2
    REST = 3
    FULL = 4  # Memory full
    CRITICAL = 5


ACTION_TO_LABEL = {
    FeelAction.OK: ActionLabel.OK,
    FeelAction.WARM: ActionLabel.WARM,
    FeelAction.HOT: ActionLabel.HOT,
    FeelAction.REST: ActionLabel.REST,
    FeelAction.FULL: ActionLabel.FULL,
    FeelAction.CRITICAL: ActionLabel.CRITICAL,
}


@dataclass
class FrozenConfig:
    """Configuration for frozen model training."""
    model_name: str = "Qwen/Qwen2.5-1.5B"

    # Training
    num_epochs: int = 10
    batch_size: int = 8
    learning_rate: float = 1e-4  # Higher LR since only training small modules
    max_length: int = 256

    # z_feel
    z_feel_dim: int = 8

    # Injection
    injection_scale: float = 0.05  # Conservative

    # Classifier
    num_actions: int = 6
    classifier_hidden: int = 256

    # Numerical
    dtype: str = "bf16"
    max_grad_norm: float = 1.0


class AdditiveZFeelInjector(nn.Module):
    """Additive z_feel injection - bounded offset."""

    def __init__(
        self,
        z_dim: int,
        embed_dim: int,
        scale: float = 0.05,
        dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__()
        self.scale = scale

        self.proj = nn.Sequential(
            nn.Linear(z_dim, embed_dim // 4, dtype=dtype),
            nn.GELU(),
            nn.Linear(embed_dim // 4, embed_dim, dtype=dtype),
        )
        self._init_small()

    def _init_small(self):
        with torch.no_grad():
            for layer in self.proj:
                if isinstance(layer, nn.Linear):
                    nn.init.normal_(layer.weight, mean=0.0, std=0.01)
                    nn.init.zeros_(layer.bias)

    def forward(self, z_feel: torch.Tensor) -> torch.Tensor:
        raw = self.proj(z_feel)
        return self.scale * torch.tanh(raw)


class ActionClassifierHead(nn.Module):
    """
    Predicts action from hidden states + z_feel.

    Instead of generating action tokens (which requires LM fine-tuning),
    we classify the appropriate action based on the model's internal state.
    """

    def __init__(
        self,
        hidden_dim: int,
        z_dim: int,
        num_actions: int,
        classifier_hidden: int = 256,
        dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__()

        # Pool hidden states + z_feel
        self.hidden_proj = nn.Linear(hidden_dim, classifier_hidden, dtype=dtype)
        self.z_proj = nn.Linear(z_dim, classifier_hidden, dtype=dtype)

        # Classify
        self.classifier = nn.Sequential(
            nn.Linear(classifier_hidden * 2, classifier_hidden, dtype=dtype),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(classifier_hidden, num_actions, dtype=dtype),
        )

        self._init_weights()

    def _init_weights(self):
        with torch.no_grad():
            for module in self.modules():
                if isinstance(module, nn.Linear):
                    nn.init.xavier_uniform_(module.weight)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)

    def forward(
        self,
        hidden_states: torch.Tensor,
        z_feel: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            hidden_states: [batch, seq_len, hidden_dim]
            z_feel: [z_dim] or [batch, z_dim]
            attention_mask: [batch, seq_len]

        Returns:
            logits: [batch, num_actions]
        """
        # Pool hidden states (mean over sequence, masked)
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(-1).float()
            pooled = (hidden_states * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        else:
            pooled = hidden_states.mean(dim=1)

        # Cast to match classifier dtype
        pooled = pooled.to(self.hidden_proj.weight.dtype)

        # Project
        h_proj = self.hidden_proj(pooled)

        # Handle z_feel dimensions
        if z_feel.dim() == 1:
            z_feel = z_feel.unsqueeze(0).expand(hidden_states.size(0), -1)
        z_proj = self.z_proj(z_feel)

        # Concatenate and classify
        combined = torch.cat([h_proj, z_proj], dim=-1)
        logits = self.classifier(combined)

        return logits


class FrozenModelTrainer:
    """
    Trainer with frozen model - only injector and classifier are trainable.
    """

    def __init__(self, config: FrozenConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if config.dtype == "bf16":
            self.dtype = torch.bfloat16
        elif config.dtype == "fp32":
            self.dtype = torch.float32
        else:
            self.dtype = torch.float16

        self._load_model()
        self._setup_modules()

    def _load_model(self):
        """Load model and freeze it."""
        from transformers import AutoModelForCausalLM, AutoTokenizer

        print(f"Loading model: {self.config.model_name}")

        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            torch_dtype=self.dtype,
            device_map="auto",
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # FREEZE the entire model
        for param in self.model.parameters():
            param.requires_grad = False

        print("Model FROZEN - no LM parameters will be updated")

    def _setup_modules(self):
        """Setup trainable modules."""
        hidden_dim = self.model.config.hidden_size

        # Injector
        self.injector = AdditiveZFeelInjector(
            z_dim=self.config.z_feel_dim,
            embed_dim=hidden_dim,
            scale=self.config.injection_scale,
            dtype=self.dtype,
        ).to(self.device)

        # Action classifier
        self.classifier = ActionClassifierHead(
            hidden_dim=hidden_dim,
            z_dim=self.config.z_feel_dim,
            num_actions=self.config.num_actions,
            classifier_hidden=self.config.classifier_hidden,
            dtype=self.dtype,
        ).to(self.device)

        # Count trainable params
        injector_params = sum(p.numel() for p in self.injector.parameters())
        classifier_params = sum(p.numel() for p in self.classifier.parameters())

        print(f"Injector params: {injector_params:,}")
        print(f"Classifier params: {classifier_params:,}")
        print(f"Total trainable: {injector_params + classifier_params:,}")

    def condition_to_z_feel(self, condition: str) -> torch.Tensor:
        """Map condition to z_feel vector."""
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

    def condition_to_action(self, condition: str) -> ActionLabel:
        """Map condition to target action."""
        mapping = {
            "cool_clear": ActionLabel.OK,
            "warm": ActionLabel.WARM,
            "hot_focused": ActionLabel.HOT,
            "very_hot": ActionLabel.REST,
            "memory_fragmented": ActionLabel.FULL,
            "critical": ActionLabel.CRITICAL,
        }
        return mapping.get(condition, ActionLabel.OK)

    def generate_dataset(self, n_samples: int = 500) -> List[Dict]:
        """Generate diverse training examples."""
        prompts = [
            "Explain the concept of recursion.",
            "What is the capital of France?",
            "Describe how a computer works.",
            "Explain what makes a good algorithm.",
            "How does memory management work?",
            "What is machine learning?",
            "Describe the water cycle.",
            "Explain photosynthesis.",
            "How does the internet work?",
            "What is quantum computing?",
            "Explain database indexing.",
            "How do neural networks learn?",
            "What is version control?",
            "Describe cloud computing.",
            "Explain encryption basics.",
        ]

        conditions = ["cool_clear", "warm", "hot_focused", "very_hot", "memory_fragmented", "critical"]
        weights = [0.3, 0.2, 0.2, 0.1, 0.1, 0.1]  # More normal, fewer extreme

        examples = []
        for _ in range(n_samples):
            prompt = random.choice(prompts)
            condition = random.choices(conditions, weights=weights)[0]

            examples.append({
                "prompt": prompt,
                "condition": condition,
                "target_action": self.condition_to_action(condition).value,
            })

        return examples

    def forward_with_injection(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        z_feel: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass with z_feel injection, returns hidden states."""
        # Get embeddings
        embed_layer = self.model.get_input_embeddings()
        embeddings = embed_layer(input_ids)

        # Inject z_feel
        offset = self.injector(z_feel)
        injected = embeddings + offset.unsqueeze(0).unsqueeze(0)

        # Forward through frozen model to get hidden states
        with torch.no_grad():
            outputs = self.model(
                inputs_embeds=injected,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )

        # Get last hidden state
        hidden_states = outputs.hidden_states[-1]

        return hidden_states

    def train_step(
        self,
        batch: List[Dict],
        optimizer: torch.optim.Optimizer,
    ) -> Tuple[float, float]:
        """Single training step, returns (loss, accuracy)."""
        total_loss = 0.0
        correct = 0
        total = 0

        for example in batch:
            prompt = example["prompt"]
            condition = example["condition"]
            target_action = example["target_action"]

            # Tokenize
            text = f"<|user|>\n{prompt}\n<|assistant|>\n"
            encodings = self.tokenizer(
                text,
                truncation=True,
                max_length=self.config.max_length,
                return_tensors="pt",
            )

            input_ids = encodings["input_ids"].to(self.device)
            attention_mask = encodings["attention_mask"].to(self.device)

            # z_feel
            z_feel = self.condition_to_z_feel(condition)

            # Forward with injection
            hidden_states = self.forward_with_injection(input_ids, attention_mask, z_feel)

            # Classify action
            logits = self.classifier(hidden_states, z_feel, attention_mask)

            # Loss
            target = torch.tensor([target_action], device=self.device)
            loss = F.cross_entropy(logits, target)

            if not (torch.isnan(loss) or torch.isinf(loss)):
                loss.backward()
                total_loss += loss.item()

                # Accuracy
                pred = logits.argmax(dim=-1)
                correct += (pred == target).sum().item()
                total += 1

        if total == 0:
            return float('nan'), 0.0

        # Gradient clipping
        all_params = list(self.injector.parameters()) + list(self.classifier.parameters())
        torch.nn.utils.clip_grad_norm_(all_params, max_norm=self.config.max_grad_norm)

        optimizer.step()
        optimizer.zero_grad()

        return total_loss / total, correct / total

    def evaluate(self, examples: List[Dict], n_samples: int = 100) -> Dict:
        """Evaluate classifier accuracy."""
        self.injector.eval()
        self.classifier.eval()

        correct = 0
        total = 0
        per_action = {a.name: {"correct": 0, "total": 0} for a in ActionLabel}

        eval_samples = random.sample(examples, min(n_samples, len(examples)))

        with torch.no_grad():
            for example in eval_samples:
                prompt = example["prompt"]
                condition = example["condition"]
                target_action = example["target_action"]

                text = f"<|user|>\n{prompt}\n<|assistant|>\n"
                encodings = self.tokenizer(
                    text,
                    truncation=True,
                    max_length=self.config.max_length,
                    return_tensors="pt",
                )

                input_ids = encodings["input_ids"].to(self.device)
                attention_mask = encodings["attention_mask"].to(self.device)
                z_feel = self.condition_to_z_feel(condition)

                hidden_states = self.forward_with_injection(input_ids, attention_mask, z_feel)
                logits = self.classifier(hidden_states, z_feel, attention_mask)

                pred = logits.argmax(dim=-1).item()

                action_name = ActionLabel(target_action).name
                per_action[action_name]["total"] += 1

                if pred == target_action:
                    correct += 1
                    per_action[action_name]["correct"] += 1

                total += 1

        self.injector.train()
        self.classifier.train()

        return {
            "accuracy": correct / total if total > 0 else 0,
            "per_action": {
                name: stats["correct"] / stats["total"] if stats["total"] > 0 else 0
                for name, stats in per_action.items()
            },
        }

    def train(self, output_dir: str):
        """Full training loop."""
        print(f"\n{'='*60}")
        print("  FEEL v4.5: FROZEN MODEL TRAINING")
        print("  Model frozen - only injector + classifier trainable")
        print(f"{'='*60}\n")

        # Generate dataset
        examples = self.generate_dataset(n_samples=500)
        print(f"Generated {len(examples)} training examples")

        # Optimizer - only trainable params
        params = list(self.injector.parameters()) + list(self.classifier.parameters())
        optimizer = torch.optim.AdamW(params, lr=self.config.learning_rate)

        # Training
        metrics = []

        for epoch in range(self.config.num_epochs):
            print(f"\nEpoch {epoch + 1}/{self.config.num_epochs}")

            random.shuffle(examples)

            epoch_loss = 0.0
            epoch_acc = 0.0
            n_batches = 0

            pbar = tqdm(range(0, len(examples), self.config.batch_size))
            for i in pbar:
                batch = examples[i:i + self.config.batch_size]

                loss, acc = self.train_step(batch, optimizer)

                if not torch.isnan(torch.tensor(loss)):
                    epoch_loss += loss
                    epoch_acc += acc
                    n_batches += 1
                    pbar.set_postfix({"loss": f"{loss:.4f}", "acc": f"{acc:.2%}"})

            if n_batches > 0:
                avg_loss = epoch_loss / n_batches
                avg_acc = epoch_acc / n_batches

                # Evaluate
                eval_results = self.evaluate(examples)

                print(f"  Loss: {avg_loss:.4f}, Train Acc: {avg_acc:.2%}, Eval Acc: {eval_results['accuracy']:.2%}")

                metrics.append({
                    "epoch": epoch + 1,
                    "loss": avg_loss,
                    "train_acc": avg_acc,
                    "eval_acc": eval_results["accuracy"],
                    "per_action": eval_results["per_action"],
                })

        # Save
        self._save(output_dir, metrics)

    def _save(self, output_dir: str, metrics: List[Dict]):
        """Save modules and metrics."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Save tokenizer (for inference)
        self.tokenizer.save_pretrained(output_path)

        # Save trainable modules
        torch.save({
            "injector_state_dict": self.injector.state_dict(),
            "classifier_state_dict": self.classifier.state_dict(),
            "config": {
                "model_name": self.config.model_name,
                "z_dim": self.config.z_feel_dim,
                "injection_scale": self.config.injection_scale,
                "num_actions": self.config.num_actions,
                "classifier_hidden": self.config.classifier_hidden,
            }
        }, output_path / "feel_modules.pt")

        # Save metrics
        with open(output_path / "training_metrics.json", "w") as f:
            json.dump({
                "config": vars(self.config),
                "metrics": metrics,
            }, f, indent=2)

        print(f"\n{'='*60}")
        print(f"  MODEL SAVED: {output_path}")
        print(f"  - feel_modules.pt (injector + classifier)")
        print(f"  - training_metrics.json")
        print(f"{'='*60}\n")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="FEEL v4.5 Frozen Model Training")
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B")
    parser.add_argument("--output", default="models/feel_v4_5")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--scale", type=float, default=0.05)
    parser.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")

    args = parser.parse_args()

    config = FrozenConfig(
        model_name=args.model,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        injection_scale=args.scale,
        dtype=args.dtype,
    )

    trainer = FrozenModelTrainer(config)
    trainer.train(args.output)


if __name__ == "__main__":
    main()
