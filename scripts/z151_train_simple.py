#!/usr/bin/env python3
"""
z151 Simple Training - Robust Early Exit Training

Uses base model with output_hidden_states=True for compatibility.
Simpler architecture that works with any HuggingFace model.
"""

import os
import sys
import json
import time
from pathlib import Path
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Dict, List, Optional, Any

sys.path.insert(0, str(Path(__file__).parent.parent))

os.environ.setdefault("HSA_OVERRIDE_GFX_VERSION", "11.0.0")

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset


@dataclass
class TrainConfig:
    model_name: str = "gpt2"
    batch_size: int = 4
    max_seq_len: int = 256
    num_steps: int = 500
    lr: float = 1e-4
    exit_layers: List[int] = None
    target_exit: float = 6.0  # For GPT-2 (12 layers)
    lambda_exit: float = 0.1
    output_dir: str = "checkpoints/z151_simple"

    def __post_init__(self):
        if self.exit_layers is None:
            self.exit_layers = [3, 6, 9, 12]  # For GPT-2


class SimpleExitHead(nn.Module):
    """Simple exit head: project hidden state to vocab"""

    def __init__(self, hidden_dim: int, vocab_size: int):
        super().__init__()
        self.proj = nn.Linear(hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.head = nn.Linear(hidden_dim, vocab_size, bias=False)

    def forward(self, hidden_state):
        h = self.norm(self.proj(hidden_state) + hidden_state)
        return self.head(h)


class SimpleExitDecision(nn.Module):
    """Simple exit decision based on hidden state entropy"""

    def __init__(self, hidden_dim: int, num_layers: int):
        super().__init__()
        self.num_layers = num_layers

        self.uncertainty_net = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

        # Per-layer threshold (learnable)
        self.thresholds = nn.Parameter(torch.linspace(0.3, 0.7, num_layers))

    def forward(self, hidden_state, layer_idx):
        """Returns exit probability"""
        # Use mean pooled hidden state
        if hidden_state.dim() == 3:
            h = hidden_state.mean(dim=1)
        else:
            h = hidden_state

        uncertainty = self.uncertainty_net(h)
        threshold = torch.sigmoid(self.thresholds[layer_idx])

        # Lower uncertainty = higher exit prob
        exit_prob = (1 - uncertainty) * (layer_idx / self.num_layers)
        return exit_prob, uncertainty


class EarlyExitModel(nn.Module):
    """Simple early exit wrapper"""

    def __init__(self, base_model, tokenizer, exit_layers: List[int]):
        super().__init__()
        self.base_model = base_model
        self.tokenizer = tokenizer
        self.exit_layers = exit_layers

        # Freeze base model
        for param in base_model.parameters():
            param.requires_grad = False

        # Get dimensions
        self.hidden_dim = base_model.config.hidden_size
        self.vocab_size = base_model.config.vocab_size
        self.num_layers = base_model.config.num_hidden_layers

        # Create exit heads
        self.exit_heads = nn.ModuleDict({
            str(l): SimpleExitHead(self.hidden_dim, self.vocab_size)
            for l in exit_layers
        })

        # Exit decision module
        self.exit_decision = SimpleExitDecision(self.hidden_dim, self.num_layers)

        # Share lm_head weights where possible
        if hasattr(base_model, 'lm_head'):
            for head in self.exit_heads.values():
                head.head.weight = base_model.lm_head.weight

    def forward(self, input_ids, labels=None, return_all_exits=False):
        """Forward with early exit"""

        # Get all hidden states from base model
        with torch.no_grad():
            outputs = self.base_model(
                input_ids,
                output_hidden_states=True,
                return_dict=True
            )

        hidden_states = outputs.hidden_states  # tuple of [batch, seq, hidden]
        final_logits = outputs.logits

        # Compute exit logits and decisions for each exit layer
        exit_logits_list = {}
        exit_probs = {}
        uncertainties = {}

        for layer in self.exit_layers:
            if layer <= len(hidden_states) - 1:
                h = hidden_states[layer]

                # Get exit logits
                logits = self.exit_heads[str(layer)](h)
                exit_logits_list[layer] = logits

                # Get exit probability
                exit_prob, uncertainty = self.exit_decision(h, layer)
                exit_probs[layer] = exit_prob
                uncertainties[layer] = uncertainty

        # During training, compute loss for all exits
        if labels is not None:
            total_loss = 0.0
            ce_losses = {}

            # Shift labels for causal LM
            shift_labels = labels[..., 1:].contiguous()

            for layer, logits in exit_logits_list.items():
                shift_logits = logits[..., :-1, :].contiguous()

                ce_loss = F.cross_entropy(
                    shift_logits.view(-1, self.vocab_size),
                    shift_labels.view(-1),
                    ignore_index=-100
                )
                ce_losses[layer] = ce_loss

                # Weight by layer (later exits weighted more during early training)
                weight = layer / self.num_layers
                total_loss += weight * ce_loss

            # Normalize
            total_loss = total_loss / len(exit_logits_list)

            # KL loss to match final distribution
            final_shift = final_logits[..., :-1, :].contiguous()
            final_probs = F.softmax(final_shift, dim=-1).detach()

            kl_loss = 0.0
            for layer, logits in exit_logits_list.items():
                shift_logits = logits[..., :-1, :].contiguous()
                exit_probs_dist = F.log_softmax(shift_logits, dim=-1)
                kl = F.kl_div(exit_probs_dist, final_probs, reduction='batchmean')
                kl_loss += kl
            kl_loss = kl_loss / len(exit_logits_list)

            # Exit cost (encourage early exits)
            exit_cost = sum(
                (l / self.num_layers) * exit_probs[l].mean()
                for l in exit_probs
            ) / len(exit_probs)

            total_loss = total_loss + 0.1 * kl_loss + 0.05 * (1 - exit_cost)

            return {
                'loss': total_loss,
                'ce_losses': ce_losses,
                'kl_loss': kl_loss.item(),
                'exit_probs': {l: p.mean().item() for l, p in exit_probs.items()},
                'uncertainties': {l: u.mean().item() for l, u in uncertainties.items()}
            }

        # During inference, return best exit
        if return_all_exits:
            return exit_logits_list, exit_probs

        # Find first layer where we're confident enough
        for layer in sorted(self.exit_layers):
            if layer in exit_probs:
                if exit_probs[layer].mean() > 0.5:
                    return exit_logits_list[layer], layer

        # Default to last exit
        last_layer = max(self.exit_layers)
        return exit_logits_list[last_layer], last_layer


def create_dataloader(tokenizer, batch_size: int, max_len: int):
    """Create simple dataloader from TinyStories"""
    print("Loading TinyStories...")
    dataset = load_dataset("roneneldan/TinyStories", split="train", streaming=True)

    def collate(examples):
        texts = [ex['text'][:max_len * 4] for ex in examples]
        encoded = tokenizer(
            texts,
            max_length=max_len,
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        )
        encoded['labels'] = encoded['input_ids'].clone()
        return encoded

    # Simple iterator
    batch = []
    for item in dataset:
        batch.append(item)
        if len(batch) >= batch_size:
            yield collate(batch)
            batch = []


def train(config: TrainConfig):
    print(f"\n{'='*60}")
    print("Early Exit Training (Simple)")
    print(f"{'='*60}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load model
    print(f"\nLoading {config.model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        torch_dtype=torch.float16
    ).to(device)

    # Adjust exit layers for model
    num_layers = base_model.config.num_hidden_layers
    config.exit_layers = [l for l in [num_layers//4, num_layers//2, 3*num_layers//4, num_layers] if l > 0]
    print(f"Exit layers: {config.exit_layers}")

    # Create exit model (use same dtype as base model)
    model = EarlyExitModel(base_model, tokenizer, config.exit_layers)
    model = model.to(device).to(torch.float16)

    # Count trainable params
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {trainable:,}")

    # Optimizer
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=config.lr
    )

    # Training
    print(f"\nTraining for {config.num_steps} steps...")
    model.train()

    dataloader = create_dataloader(tokenizer, config.batch_size, config.max_seq_len)

    metrics_history = []
    pbar = tqdm(range(config.num_steps), desc="Training")

    for step in pbar:
        try:
            batch = next(dataloader)
        except StopIteration:
            dataloader = create_dataloader(tokenizer, config.batch_size, config.max_seq_len)
            batch = next(dataloader)

        input_ids = batch['input_ids'].to(device)
        labels = batch['labels'].to(device)

        optimizer.zero_grad()

        # Use autocast for mixed precision
        with torch.amp.autocast('cuda', dtype=torch.float16):
            output = model(input_ids, labels=labels)
            loss = output['loss']

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        # Log
        metrics = {
            'step': step,
            'loss': loss.item(),
            'kl_loss': output['kl_loss'],
            **{f'ce_{l}': v.item() for l, v in output['ce_losses'].items()},
            **{f'exit_prob_{l}': v for l, v in output['exit_probs'].items()},
            **{f'uncertainty_{l}': v for l, v in output['uncertainties'].items()}
        }
        metrics_history.append(metrics)

        pbar.set_postfix({
            'loss': f"{loss.item():.4f}",
            'exit_probs': [f"{v:.2f}" for v in output['exit_probs'].values()]
        })

        # Checkpoint
        if (step + 1) % 100 == 0:
            save_checkpoint(model, config, step, metrics_history)

    # Final save
    save_checkpoint(model, config, config.num_steps, metrics_history, final=True)

    print("\nTraining complete!")
    return model, metrics_history


def save_checkpoint(model, config, step, metrics, final=False):
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    name = "final" if final else f"step_{step}"

    # Save model state (only trainable parts)
    torch.save({
        'step': step,
        'exit_heads': model.exit_heads.state_dict(),
        'exit_decision': model.exit_decision.state_dict(),
        'config': asdict(config)
    }, output_dir / f"checkpoint_{name}.pt")

    # Save metrics
    with open(output_dir / f"metrics_{name}.json", 'w') as f:
        json.dump(metrics, f, indent=2)

    print(f"\nSaved checkpoint: {name}")


def evaluate(model, tokenizer, prompts: List[str], device):
    """Evaluate model on prompts"""
    model.eval()

    results = []
    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors='pt').to(device)

        with torch.no_grad():
            logits, exit_layer = model(inputs['input_ids'])

        # Get prediction
        next_token = logits[0, -1].argmax().item()
        next_word = tokenizer.decode([next_token])

        results.append({
            'prompt': prompt,
            'exit_layer': exit_layer,
            'next_token': next_word
        })

    return results


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="gpt2")
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--output-dir", default="checkpoints/z151_simple")
    args = parser.parse_args()

    config = TrainConfig(
        model_name=args.model,
        num_steps=args.steps,
        batch_size=args.batch_size,
        output_dir=args.output_dir
    )

    model, metrics = train(config)

    # Quick evaluation
    print("\n" + "="*60)
    print("Evaluation")
    print("="*60)

    device = next(model.parameters()).device
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)

    test_prompts = [
        "The cat sat on the",  # Easy
        "Once upon a time",     # Easy
        "The meaning of life is", # Medium
        "Quantum mechanics describes", # Hard
    ]

    results = evaluate(model, tokenizer, test_prompts, device)

    for r in results:
        print(f"\nPrompt: {r['prompt']}")
        print(f"  Exit layer: {r['exit_layer']}/{model.num_layers}")
        print(f"  Next: {r['next_token']}")


if __name__ == "__main__":
    main()
