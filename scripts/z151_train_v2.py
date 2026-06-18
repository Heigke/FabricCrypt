#!/usr/bin/env python3
"""
Early Exit Training v2 - With exit incentive loss

Key changes from v1:
1. Exit incentive loss to encourage early exits when confident
2. Uncertainty targets based on perplexity difference from final layer
3. Longer training (1000 steps)
4. Better uncertainty calibration
"""

import os
os.environ["HSA_OVERRIDE_GFX_VERSION"] = "11.0.0"

import sys
import json
import time
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset


class ExitHead(nn.Module):
    """Simple exit head"""
    def __init__(self, hidden_dim, vocab_size):
        super().__init__()
        self.proj = nn.Linear(hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.out = nn.Linear(hidden_dim, vocab_size, bias=False)

    def forward(self, x):
        return self.out(self.norm(F.gelu(self.proj(x)) + x))


class EarlyExitGPT2(nn.Module):
    def __init__(self, model_name="gpt2"):
        super().__init__()

        # Load base model
        self.base = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32)

        # Freeze base
        for p in self.base.parameters():
            p.requires_grad = False

        # Config
        self.hidden_dim = self.base.config.hidden_size
        self.vocab_size = self.base.config.vocab_size
        self.num_layers = self.base.config.num_hidden_layers

        # Exit layers (at 1/4, 1/2, 3/4, full)
        self.exit_layer_indices = [
            self.num_layers // 4,
            self.num_layers // 2,
            3 * self.num_layers // 4,
            self.num_layers
        ]

        # Exit heads
        self.exit_heads = nn.ModuleList([
            ExitHead(self.hidden_dim, self.vocab_size)
            for _ in self.exit_layer_indices
        ])

        # Uncertainty estimator - predict if early exit will match final layer
        self.uncertainty = nn.Sequential(
            nn.Linear(self.hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, input_ids, labels=None, compute_exit_targets=True):
        # Get all hidden states
        with torch.no_grad():
            out = self.base(input_ids, output_hidden_states=True)

        hidden_states = out.hidden_states
        final_logits = out.logits

        # Compute exit outputs
        exit_outputs = []
        uncertainties = []

        for i, layer_idx in enumerate(self.exit_layer_indices):
            h = hidden_states[layer_idx]
            logits = self.exit_heads[i](h)
            exit_outputs.append(logits)

            # Uncertainty from last token
            u = self.uncertainty(h[:, -1, :])
            uncertainties.append(u)

        # Training loss
        if labels is not None:
            total_loss = torch.tensor(0.0, device=input_ids.device)
            ce_losses = []

            shift_labels = labels[..., 1:].contiguous()

            # CE loss for each exit head
            for i, logits in enumerate(exit_outputs):
                shift_logits = logits[..., :-1, :].contiguous()
                ce = F.cross_entropy(
                    shift_logits.view(-1, self.vocab_size),
                    shift_labels.view(-1),
                    ignore_index=-100
                )
                ce_losses.append(ce.item())

                # Weight: later layers get higher weight
                w = (i + 1) / len(self.exit_layer_indices)
                total_loss = total_loss + w * ce

            total_loss = total_loss / len(exit_outputs)

            # KL to final
            final_shift = final_logits[..., :-1, :].contiguous()
            final_probs = F.softmax(final_shift.detach(), dim=-1)

            kl_loss = torch.tensor(0.0, device=input_ids.device)
            for logits in exit_outputs:
                shift_logits = logits[..., :-1, :].contiguous()
                log_probs = F.log_softmax(shift_logits, dim=-1)
                kl = F.kl_div(log_probs, final_probs, reduction='batchmean')
                kl_loss = kl_loss + kl
            kl_loss = kl_loss / len(exit_outputs)

            total_loss = total_loss + 0.1 * kl_loss

            # Uncertainty calibration loss
            # Target: uncertainty should be HIGH when exit disagrees with final layer
            # and LOW when exit agrees with final layer
            if compute_exit_targets:
                uncertainty_loss = torch.tensor(0.0, device=input_ids.device)

                final_pred = final_logits[:, -1, :].argmax(dim=-1)  # [batch]

                for i, (logits, u) in enumerate(zip(exit_outputs, uncertainties)):
                    exit_pred = logits[:, -1, :].argmax(dim=-1)  # [batch]

                    # Target: 0 if exit matches final (confident), 1 if doesn't match (uncertain)
                    target = (exit_pred != final_pred).float().unsqueeze(-1)

                    # BCE loss: uncertainty should match target
                    u_loss = F.binary_cross_entropy(u, target)
                    uncertainty_loss = uncertainty_loss + u_loss

                uncertainty_loss = uncertainty_loss / len(exit_outputs)
                total_loss = total_loss + 0.5 * uncertainty_loss

            # Exit incentive: reward early exits when uncertainty is low
            exit_incentive = torch.tensor(0.0, device=input_ids.device)
            for i, u in enumerate(uncertainties):
                # Normalized layer position (0=early, 1=late)
                layer_pos = (i + 1) / len(self.exit_layer_indices)
                # Reward: low uncertainty at early layers, high uncertainty at late layers
                # If confident (low u) at early layer (low pos), reward is high
                reward = (1 - u.mean()) * (1 - layer_pos)
                exit_incentive = exit_incentive + reward

            # Maximize exit incentive (minimize negative)
            total_loss = total_loss - 0.1 * exit_incentive

            return {
                'loss': total_loss,
                'ce_losses': ce_losses,
                'kl_loss': kl_loss.item(),
                'uncertainties': [u.mean().item() for u in uncertainties],
                'exit_incentive': exit_incentive.item()
            }

        return exit_outputs, uncertainties


def main():
    print("="*60)
    print("Early Exit Training v2 (with exit incentive)")
    print("="*60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Model
    print("\nLoading GPT-2...")
    model = EarlyExitGPT2("gpt2").to(device)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Exit layers: {model.exit_layer_indices}")
    print(f"Trainable params: {trainable:,}")

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    # Optimizer
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=5e-5  # Lower LR for stability
    )

    # LR scheduler with warmup
    num_steps = 1000
    warmup_steps = 100

    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        return max(0.1, 1 - (step - warmup_steps) / (num_steps - warmup_steps))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Data
    print("\nLoading TinyStories...")
    dataset = load_dataset("roneneldan/TinyStories", split="train", streaming=True)

    # Training
    print(f"\nTraining for {num_steps} steps...")
    model.train()

    metrics = []
    data_iter = iter(dataset)
    pbar = tqdm(range(num_steps))

    for step in pbar:
        # Get batch
        texts = []
        for _ in range(8):  # batch size 8
            try:
                item = next(data_iter)
                texts.append(item['text'][:512])
            except StopIteration:
                data_iter = iter(dataset)
                item = next(data_iter)
                texts.append(item['text'][:512])

        # Tokenize
        encoded = tokenizer(
            texts,
            max_length=128,
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        )
        input_ids = encoded['input_ids'].to(device)
        labels = input_ids.clone()

        # Forward
        optimizer.zero_grad()
        out = model(input_ids, labels=labels)
        loss = out['loss']

        # Backward
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        # Log
        metrics.append({
            'step': step,
            'loss': loss.item(),
            'ce': out['ce_losses'],
            'kl': out['kl_loss'],
            'uncertainty': out['uncertainties'],
            'exit_incentive': out['exit_incentive'],
            'lr': scheduler.get_last_lr()[0]
        })

        pbar.set_postfix({
            'loss': f"{loss.item():.4f}",
            'u': [f"{u:.2f}" for u in out['uncertainties']],
            'exit_inc': f"{out['exit_incentive']:.3f}"
        })

        # Checkpoint every 250 steps
        if (step + 1) % 250 == 0:
            output_dir = Path("checkpoints/z151_v2")
            output_dir.mkdir(parents=True, exist_ok=True)

            torch.save({
                'step': step,
                'exit_heads': model.exit_heads.state_dict(),
                'uncertainty': model.uncertainty.state_dict(),
                'exit_layers': model.exit_layer_indices
            }, output_dir / f"checkpoint_step{step+1}.pt")
            print(f"\n  Saved checkpoint at step {step+1}")

    # Final save
    output_dir = Path("checkpoints/z151_v2")
    output_dir.mkdir(parents=True, exist_ok=True)

    torch.save({
        'exit_heads': model.exit_heads.state_dict(),
        'uncertainty': model.uncertainty.state_dict(),
        'exit_layers': model.exit_layer_indices
    }, output_dir / "checkpoint.pt")

    with open(output_dir / "metrics.json", 'w') as f:
        json.dump(metrics, f, indent=2)

    print(f"\nSaved to {output_dir}")

    # Eval
    print("\n" + "="*60)
    print("Evaluation")
    print("="*60)

    model.eval()
    test_prompts = [
        "Once upon a time",        # Easy (common pattern)
        "The cat sat on the",      # Easy
        "Scientists discovered",    # Medium
        "The epistemological",     # Hard
    ]

    for prompt in test_prompts:
        inputs = tokenizer(prompt, return_tensors='pt').to(device)
        with torch.no_grad():
            exits, uncertainties = model(inputs['input_ids'])

        u_values = [u.item() for u in uncertainties]
        best_exit = np.argmin(u_values)
        best_layer = model.exit_layer_indices[best_exit]

        next_token = exits[best_exit][0, -1].argmax().item()
        next_word = tokenizer.decode([next_token])

        print(f"\nPrompt: '{prompt}'")
        print(f"  Uncertainties: {[f'{u:.3f}' for u in u_values]}")
        print(f"  Best exit: layer {best_layer}/{model.num_layers}")
        print(f"  Next: '{next_word}'")

    # Exit distribution summary
    print("\n" + "-"*40)
    print("Exit layer preferences:")
    exit_counts = {l: 0 for l in model.exit_layer_indices}
    for prompt in test_prompts:
        inputs = tokenizer(prompt, return_tensors='pt').to(device)
        with torch.no_grad():
            exits, uncertainties = model(inputs['input_ids'])
        u_values = [u.item() for u in uncertainties]
        best_exit = np.argmin(u_values)
        best_layer = model.exit_layer_indices[best_exit]
        exit_counts[best_layer] += 1

    for layer, count in exit_counts.items():
        pct = 100 * count / len(test_prompts)
        print(f"  Layer {layer:2d}: {count}/{len(test_prompts)} ({pct:.0f}%)")

    print("\n" + "="*60)
    print("Done!")
    print("="*60)


if __name__ == "__main__":
    main()
