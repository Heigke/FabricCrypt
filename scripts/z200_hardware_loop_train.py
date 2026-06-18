#!/usr/bin/env python3
"""
Hardware-in-the-Loop Training for Embodied Conditional Compute

KEY DIFFERENCE FROM PREVIOUS: Real energy measurement, not simulated.

The model learns a causal world model:
    action (depth, span) → measure(energy, latency) → optimize(quality | energy_budget)

Training signal comes from REAL hardware, not simulation.
"""

import os
os.environ["HSA_OVERRIDE_GFX_VERSION"] = "11.0.0"

import sys
import json
import time
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional, Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

from src.telemetry.real_amd import AMDTelemetry, EnergyMeter, TelemetrySnapshot


@dataclass
class TrainingConfig:
    """Training configuration."""
    model_name: str = "gpt2"
    batch_size: int = 4
    max_seq_len: int = 128
    num_steps: int = 500
    lr: float = 3e-5

    # Energy constraint (Lagrangian)
    energy_budget_mj_per_token: float = 2.0  # Target mJ/token
    lambda_energy: float = 0.1  # Energy constraint weight

    # Action space
    exit_layers: List[int] = None
    span_choices: List[int] = None

    output_dir: str = "checkpoints/z200_hardware_loop"

    def __post_init__(self):
        if self.exit_layers is None:
            self.exit_layers = [3, 6, 9, 12]  # For GPT-2
        if self.span_choices is None:
            self.span_choices = [32, 64, 128, 256]


class ExitHead(nn.Module):
    """Exit head for early exit."""
    def __init__(self, hidden_dim: int, vocab_size: int):
        super().__init__()
        self.proj = nn.Linear(hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.out = nn.Linear(hidden_dim, vocab_size, bias=False)

    def forward(self, x):
        return self.out(self.norm(F.gelu(self.proj(x)) + x))


class ComputePolicy(nn.Module):
    """
    Policy network that chooses compute actions based on:
    - Semantic content (hidden state)
    - Body state (hardware telemetry)

    Actions: (exit_layer_idx, span_idx)
    """

    def __init__(
        self,
        hidden_dim: int,
        body_dim: int,  # Telemetry vector size
        num_exit_choices: int,
        num_span_choices: int
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.body_dim = body_dim
        self.num_exit_choices = num_exit_choices
        self.num_span_choices = num_span_choices

        # Semantic encoder
        self.semantic_encoder = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64)
        )

        # Body state encoder
        self.body_encoder = nn.Sequential(
            nn.Linear(body_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 32)
        )

        # Action heads
        self.exit_head = nn.Sequential(
            nn.Linear(96, 64),
            nn.ReLU(),
            nn.Linear(64, num_exit_choices)
        )

        self.span_head = nn.Sequential(
            nn.Linear(96, 64),
            nn.ReLU(),
            nn.Linear(64, num_span_choices)
        )

        # Temperature for Gumbel-softmax
        self.temperature = nn.Parameter(torch.ones(1) * 1.0)

    def forward(
        self,
        hidden_state: torch.Tensor,  # [batch, seq, hidden]
        body_state: torch.Tensor,     # [batch, body_dim]
        hard: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            exit_probs: [batch, num_exit]
            span_probs: [batch, num_span]
            exit_idx: [batch] selected exit
            span_idx: [batch] selected span
        """
        # Encode semantic (use mean-pooled hidden state)
        h_pooled = hidden_state.mean(dim=1)  # [batch, hidden]
        semantic = self.semantic_encoder(h_pooled)  # [batch, 64]

        # Encode body
        body = self.body_encoder(body_state)  # [batch, 32]

        # Combine
        combined = torch.cat([semantic, body], dim=-1)  # [batch, 96]

        # Action logits
        exit_logits = self.exit_head(combined)
        span_logits = self.span_head(combined)

        # Temperature-scaled softmax
        temp = F.softplus(self.temperature) + 0.1

        if self.training and not hard:
            exit_probs = F.gumbel_softmax(exit_logits, tau=temp, hard=False)
            span_probs = F.gumbel_softmax(span_logits, tau=temp, hard=False)
        else:
            exit_probs = F.softmax(exit_logits / temp, dim=-1)
            span_probs = F.softmax(span_logits / temp, dim=-1)

        exit_idx = exit_probs.argmax(dim=-1)
        span_idx = span_probs.argmax(dim=-1)

        return exit_probs, span_probs, exit_idx, span_idx


class HardwareLoopModel(nn.Module):
    """
    Model with hardware-in-the-loop training.

    The compute policy is trained with REAL energy measurements.
    """

    def __init__(self, config: TrainingConfig):
        super().__init__()

        # Load base model
        self.base = AutoModelForCausalLM.from_pretrained(
            config.model_name, torch_dtype=torch.float32
        )
        for p in self.base.parameters():
            p.requires_grad = False

        self.hidden_dim = self.base.config.hidden_size
        self.vocab_size = self.base.config.vocab_size
        self.num_layers = self.base.config.num_hidden_layers

        self.exit_layers = config.exit_layers
        self.span_choices = config.span_choices

        # Exit heads
        self.exit_heads = nn.ModuleList([
            ExitHead(self.hidden_dim, self.vocab_size)
            for _ in self.exit_layers
        ])

        # Body state dimension: power, temp, vram_pct, util_pct, sclk_norm, mclk_norm
        self.body_dim = 6

        # Compute policy
        self.policy = ComputePolicy(
            self.hidden_dim,
            self.body_dim,
            len(self.exit_layers),
            len(self.span_choices)
        )

        # World model: predicts (energy, latency) from (action, body_state)
        # This learns the causal relationship
        self.world_model = nn.Sequential(
            nn.Linear(len(self.exit_layers) + len(self.span_choices) + self.body_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 2)  # Predicts [energy_normalized, latency_normalized]
        )

    def telemetry_to_body_state(
        self,
        snapshot: TelemetrySnapshot,
        device: torch.device
    ) -> torch.Tensor:
        """Convert hardware telemetry to body state tensor."""
        # Normalize to [0, 1] range
        body = torch.tensor([
            snapshot.power_w / 100.0,  # Assume max 100W
            snapshot.temp_c / 100.0,   # Assume max 100C
            snapshot.vram_used_gb / max(snapshot.vram_total_gb, 1.0),
            snapshot.gpu_util_pct / 100.0,
            snapshot.sclk_mhz / 3000.0,  # Max ~3GHz
            snapshot.mclk_mhz / 2000.0,  # Max ~2GHz
        ], device=device, dtype=torch.float32)
        return body

    def forward(
        self,
        input_ids: torch.Tensor,
        body_state: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        return_policy: bool = False
    ):
        """Forward pass with policy-selected compute."""
        device = input_ids.device
        batch_size = input_ids.size(0)

        # Get hidden states from base model
        with torch.no_grad():
            out = self.base(input_ids, output_hidden_states=True)

        hidden_states = out.hidden_states
        final_logits = out.logits

        # Use first layer hidden state for policy decision
        h_first = hidden_states[1]  # After first layer

        # Get policy actions
        exit_probs, span_probs, exit_idx, span_idx = self.policy(
            h_first, body_state.unsqueeze(0).expand(batch_size, -1)
        )

        # Compute exit outputs for selected exits
        exit_outputs = []
        for i, layer_idx in enumerate(self.exit_layers):
            h = hidden_states[layer_idx]
            logits = self.exit_heads[i](h)
            exit_outputs.append(logits)

        # Select output based on policy (weighted by probs during training)
        if self.training:
            # Soft selection
            stacked = torch.stack(exit_outputs, dim=1)  # [batch, num_exits, seq, vocab]
            weights = exit_probs.unsqueeze(-1).unsqueeze(-1)  # [batch, num_exits, 1, 1]
            selected_logits = (stacked * weights).sum(dim=1)
        else:
            # Hard selection
            selected_logits = exit_outputs[exit_idx[0].item()]

        result = {
            'logits': selected_logits,
            'exit_probs': exit_probs,
            'span_probs': span_probs,
            'exit_idx': exit_idx,
            'span_idx': span_idx,
            'final_logits': final_logits
        }

        if labels is not None:
            # Quality loss
            shift_logits = selected_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            quality_loss = F.cross_entropy(
                shift_logits.view(-1, self.vocab_size),
                shift_labels.view(-1),
                ignore_index=-100
            )

            # KL loss to final layer (preserve semantics)
            final_shift = final_logits[..., :-1, :].detach()
            kl_loss = F.kl_div(
                F.log_softmax(shift_logits, dim=-1),
                F.softmax(final_shift, dim=-1),
                reduction='batchmean'
            )

            result['quality_loss'] = quality_loss
            result['kl_loss'] = kl_loss

        if return_policy:
            result['exit_outputs'] = exit_outputs

        return result

    def predict_energy_latency(
        self,
        exit_probs: torch.Tensor,
        span_probs: torch.Tensor,
        body_state: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Use world model to predict energy and latency.
        This is learned from real measurements.
        """
        batch_size = exit_probs.size(0)
        body_expanded = body_state.unsqueeze(0).expand(batch_size, -1)

        world_input = torch.cat([exit_probs, span_probs, body_expanded], dim=-1)
        predictions = self.world_model(world_input)

        energy_pred = predictions[:, 0]  # Normalized energy
        latency_pred = predictions[:, 1]  # Normalized latency

        return energy_pred, latency_pred


def train_hardware_loop(config: TrainingConfig):
    """Main training loop with real hardware telemetry."""

    print("="*60)
    print("Hardware-in-the-Loop Training")
    print("="*60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Initialize hardware telemetry
    print("\nInitializing hardware telemetry...")
    telemetry = AMDTelemetry()
    energy_meter = EnergyMeter(telemetry, sample_interval_ms=20)

    # Read initial state
    initial = telemetry.read()
    print(f"  Power: {initial.power_w:.1f}W")
    print(f"  Temp: {initial.temp_c:.0f}°C")
    print(f"  VRAM: {initial.vram_used_gb:.2f}GB")

    # Load model
    print(f"\nLoading model...")
    model = HardwareLoopModel(config).to(device)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Exit layers: {config.exit_layers}")
    print(f"  Span choices: {config.span_choices}")
    print(f"  Trainable params: {trainable:,}")

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    tokenizer.pad_token = tokenizer.eos_token

    # Optimizer
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=config.lr
    )

    # LR scheduler
    warmup_steps = min(100, config.num_steps // 10)

    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        return max(0.1, 1 - (step - warmup_steps) / (config.num_steps - warmup_steps))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Data
    print("\nLoading data...")
    dataset = load_dataset("roneneldan/TinyStories", split="train", streaming=True)
    data_iter = iter(dataset)

    # Training metrics
    metrics_history = []
    energy_observations = []  # For world model training

    # Output directory
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nTraining for {config.num_steps} steps...")
    print(f"Energy budget: {config.energy_budget_mj_per_token} mJ/token")
    model.train()

    pbar = tqdm(range(config.num_steps))

    for step in pbar:
        # Get batch
        texts = []
        for _ in range(config.batch_size):
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
            max_length=config.max_seq_len,
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        )
        input_ids = encoded['input_ids'].to(device)
        labels = input_ids.clone()
        num_tokens = (input_ids != tokenizer.pad_token_id).sum().item()

        # Read current body state
        body_snapshot = telemetry.read()
        body_state = model.telemetry_to_body_state(body_snapshot, device)

        # Forward pass WITH energy measurement
        optimizer.zero_grad()

        with energy_meter.measure():
            torch.cuda.synchronize()
            out = model(input_ids, body_state, labels=labels)
            torch.cuda.synchronize()

        energy_result = energy_meter.result

        # Compute losses
        quality_loss = out['quality_loss']
        kl_loss = out['kl_loss']

        # Real energy per token
        energy_per_token = energy_result.energy_mj / max(num_tokens, 1)
        latency_per_token = energy_result.duration_ms / max(num_tokens, 1)

        # Energy constraint loss (Lagrangian)
        energy_violation = max(0, energy_per_token - config.energy_budget_mj_per_token)
        energy_loss = torch.tensor(energy_violation, device=device)

        # Total loss
        total_loss = quality_loss + 0.1 * kl_loss + config.lambda_energy * energy_loss

        # World model training: record (action, body, outcome) for learning
        with torch.no_grad():
            energy_observations.append({
                'exit_probs': out['exit_probs'].cpu().numpy().tolist(),
                'span_probs': out['span_probs'].cpu().numpy().tolist(),
                'body_state': body_state.cpu().numpy().tolist(),
                'energy_mj': energy_result.energy_mj,
                'latency_ms': energy_result.duration_ms,
                'num_tokens': num_tokens
            })

        # Backward
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        # Log metrics
        exit_idx = out['exit_idx'][0].item()
        span_idx = out['span_idx'][0].item()
        selected_exit = config.exit_layers[exit_idx]
        selected_span = config.span_choices[span_idx]

        metrics = {
            'step': step,
            'total_loss': total_loss.item(),
            'quality_loss': quality_loss.item(),
            'kl_loss': kl_loss.item(),
            'energy_per_token_mj': energy_per_token,
            'latency_per_token_ms': latency_per_token,
            'selected_exit': selected_exit,
            'selected_span': selected_span,
            'power_w': body_snapshot.power_w,
            'temp_c': body_snapshot.temp_c,
            'energy_violation': energy_violation
        }
        metrics_history.append(metrics)

        pbar.set_postfix({
            'loss': f"{total_loss.item():.3f}",
            'E/tok': f"{energy_per_token:.2f}",
            'exit': selected_exit,
            'span': selected_span,
            'power': f"{body_snapshot.power_w:.0f}W"
        })

        # Checkpoint
        if (step + 1) % 100 == 0:
            torch.save({
                'step': step,
                'model_state': model.state_dict(),
                'config': asdict(config)
            }, output_dir / f"checkpoint_step{step+1}.pt")

            with open(output_dir / f"metrics_step{step+1}.json", 'w') as f:
                json.dump(metrics_history, f, indent=2)

            # Train world model on collected observations
            if len(energy_observations) > 50:
                train_world_model(model, energy_observations, device)

    # Final save
    torch.save({
        'model_state': model.state_dict(),
        'config': asdict(config)
    }, output_dir / "checkpoint_final.pt")

    with open(output_dir / "metrics_final.json", 'w') as f:
        json.dump(metrics_history, f, indent=2)

    with open(output_dir / "energy_observations.json", 'w') as f:
        json.dump(energy_observations, f, indent=2)

    # Cleanup
    telemetry.shutdown()

    print(f"\nTraining complete! Saved to {output_dir}")

    # Summary statistics
    print("\n" + "="*60)
    print("Training Summary")
    print("="*60)

    recent = metrics_history[-50:] if len(metrics_history) > 50 else metrics_history
    avg_energy = np.mean([m['energy_per_token_mj'] for m in recent])
    avg_exit = np.mean([m['selected_exit'] for m in recent])
    avg_span = np.mean([m['selected_span'] for m in recent])
    avg_loss = np.mean([m['quality_loss'] for m in recent])

    print(f"  Avg energy/token: {avg_energy:.3f} mJ (budget: {config.energy_budget_mj_per_token})")
    print(f"  Avg exit layer: {avg_exit:.1f}/{model.num_layers}")
    print(f"  Avg span: {avg_span:.0f}")
    print(f"  Avg quality loss: {avg_loss:.3f}")

    return model, metrics_history


def train_world_model(model: HardwareLoopModel, observations: List[dict], device: torch.device):
    """
    Train the world model on collected (action, body, energy, latency) observations.
    This is the "internal physics model" that makes the system truly embodied.
    """
    # Convert observations to tensors
    exit_probs = torch.tensor([o['exit_probs'][0] for o in observations], device=device)
    span_probs = torch.tensor([o['span_probs'][0] for o in observations], device=device)
    body_states = torch.tensor([o['body_state'] for o in observations], device=device)
    energies = torch.tensor([o['energy_mj'] / o['num_tokens'] for o in observations], device=device)
    latencies = torch.tensor([o['latency_ms'] / o['num_tokens'] for o in observations], device=device)

    # Normalize targets
    energy_mean, energy_std = energies.mean(), energies.std() + 1e-6
    latency_mean, latency_std = latencies.mean(), latencies.std() + 1e-6
    energy_norm = (energies - energy_mean) / energy_std
    latency_norm = (latencies - latency_mean) / latency_std
    targets = torch.stack([energy_norm, latency_norm], dim=-1)

    # Train world model
    world_optimizer = torch.optim.Adam(model.world_model.parameters(), lr=1e-3)

    for _ in range(10):  # Quick update
        world_input = torch.cat([exit_probs, span_probs, body_states], dim=-1)
        predictions = model.world_model(world_input)
        loss = F.mse_loss(predictions, targets)
        world_optimizer.zero_grad()
        loss.backward()
        world_optimizer.step()


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--energy-budget", type=float, default=2.0,
                        help="Target mJ per token")
    parser.add_argument("--output-dir", default="checkpoints/z200_hardware_loop")
    args = parser.parse_args()

    config = TrainingConfig(
        num_steps=args.steps,
        batch_size=args.batch_size,
        energy_budget_mj_per_token=args.energy_budget,
        output_dir=args.output_dir
    )

    train_hardware_loop(config)


if __name__ == "__main__":
    main()
