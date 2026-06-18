#!/usr/bin/env python3
"""
z1137: Embodied Training - Learning Shaped by Hardware State
=============================================================

The model learns differently based on its physical state:
- High energy consumption → amplified gradients (pressure to be efficient)
- Low DRAM charge → regularization toward simpler patterns
- High temperature → reduced learning rate (thermal throttling)

This is TRUE embodiment: the learning process itself is hardware-aware.

Author: FEEL Research Team
Date: 2026-01-31
"""

import sys
import os
import json
import time
import math
from pathlib import Path
from datetime import datetime
from typing import Optional, Tuple, List

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW

# Import from z1136
from scripts.z1136_embodied_loop import (
    SensorySystem,
    SensorySnapshot,
    FeelSystem,
    ExpressSystem,
    EmbodiedLoop,
)


class EmbodiedLoss(nn.Module):
    """
    Loss function modulated by hardware state.

    L_embodied = L_task * energy_pressure * charge_regularization

    Where:
    - energy_pressure = 1 + β * max(0, E - E_budget) / E_budget
    - charge_regularization = 1 + γ * (1 - charge)

    This creates pressure to find efficient solutions when resources are constrained.
    """

    def __init__(
        self,
        energy_budget_j: float = 0.01,  # Per-token energy budget
        energy_pressure_beta: float = 0.5,
        charge_pressure_gamma: float = 0.2,
        temp_lr_reduction_alpha: float = 0.01,
    ):
        super().__init__()
        self.energy_budget = energy_budget_j
        self.beta = energy_pressure_beta
        self.gamma = charge_pressure_gamma
        self.alpha = temp_lr_reduction_alpha

        # Tracking
        self.energy_history = []
        self.loss_history = []
        self.modulation_history = []

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        snapshot: SensorySnapshot,
        energy_this_step_j: float,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Compute embodied loss.

        Returns: (loss, info_dict)
        """
        # Base task loss
        task_loss = F.cross_entropy(logits, targets)

        # === Energy Pressure ===
        # When using more energy than budget, amplify loss
        energy_ratio = energy_this_step_j / max(self.energy_budget, 1e-6)
        energy_pressure = 1.0 + self.beta * max(0, energy_ratio - 1.0)

        # === Charge Regularization ===
        # When DRAM charge is low, penalize complex computations
        charge_pressure = 1.0 + self.gamma * (1.0 - snapshot.dram_charge)

        # === Combined Modulation ===
        modulation = energy_pressure * charge_pressure

        # Embodied loss
        embodied_loss = task_loss * modulation

        # Track
        info = {
            'task_loss': task_loss.item(),
            'embodied_loss': embodied_loss.item(),
            'energy_pressure': energy_pressure,
            'charge_pressure': charge_pressure,
            'modulation': modulation,
            'energy_j': energy_this_step_j,
        }

        self.energy_history.append(energy_this_step_j)
        self.loss_history.append(embodied_loss.item())
        self.modulation_history.append(modulation)

        return embodied_loss, info

    def get_lr_multiplier(self, snapshot: SensorySnapshot) -> float:
        """
        Get learning rate multiplier based on temperature.

        High temp → reduce LR (thermal throttling for learning)
        """
        temp_excess = max(0, snapshot.gpu_temp_c - 70.0)  # Above 70C
        return 1.0 / (1.0 + self.alpha * temp_excess)


class EmbodiedTrainer:
    """
    Train a model with embodied awareness.

    The training loop:
    1. Sense hardware state
    2. Forward pass
    3. Compute embodied loss
    4. Backward with hardware-modulated gradients
    5. Regulate hardware based on training state
    """

    def __init__(
        self,
        model: ExpressSystem,
        feel_system: FeelSystem,
        sense_system: SensorySystem,
        learning_rate: float = 1e-3,
        device: str = 'cuda',
    ):
        self.model = model.to(device)
        self.feel = feel_system.to(device)
        self.sense = sense_system
        self.device = device

        # Optimizer
        self.optimizer = AdamW(
            list(model.parameters()) + list(feel_system.parameters()),
            lr=learning_rate,
        )

        # Embodied loss
        self.loss_fn = EmbodiedLoss()

        # Training state
        self.step = 0
        self.epoch = 0
        self.metrics = {
            'steps': [],
            'task_loss': [],
            'embodied_loss': [],
            'energy_j': [],
            'charge': [],
            'modulation': [],
        }

    def train_step(
        self,
        tokens: torch.Tensor,
        targets: torch.Tensor,
    ) -> dict:
        """
        Execute one training step with embodiment.
        """
        self.model.train()
        self.feel.train()

        # 1. SENSE
        snapshot_before = self.sense.sense()
        energy_before = snapshot_before.gpu_energy_j

        # 2. FEEL
        z_feel = self.feel.feel(snapshot_before).detach()  # Detach to avoid double backward

        # 3. EXPRESS (forward)
        tokens = tokens.to(self.device)
        targets = targets.to(self.device)

        logits, exit_layer, confidence = self.model(tokens, z_feel)

        # 4. SENSE again (to measure energy)
        snapshot_after = self.sense.sense()
        energy_this_step = snapshot_after.gpu_energy_j - energy_before

        # 5. COMPUTE EMBODIED LOSS
        loss, loss_info = self.loss_fn(logits, targets, snapshot_after, energy_this_step)

        # 6. BACKWARD with embodiment
        self.optimizer.zero_grad()
        loss.backward()

        # Modulate gradients by temperature
        lr_mult = self.loss_fn.get_lr_multiplier(snapshot_after)
        if lr_mult < 1.0:
            for param in self.model.parameters():
                if param.grad is not None:
                    param.grad *= lr_mult

        self.optimizer.step()

        # 7. REGULATE (via feel system recording)
        if snapshot_after.dram_charge < 0.3:
            self.sense.fpga.record_frac(4)  # Replenish

        # Update metrics
        self.step += 1
        self.metrics['steps'].append(self.step)
        self.metrics['task_loss'].append(loss_info['task_loss'])
        self.metrics['embodied_loss'].append(loss_info['embodied_loss'])
        self.metrics['energy_j'].append(energy_this_step)
        self.metrics['charge'].append(snapshot_after.dram_charge)
        self.metrics['modulation'].append(loss_info['modulation'])

        return {
            'step': self.step,
            'task_loss': loss_info['task_loss'],
            'embodied_loss': loss_info['embodied_loss'],
            'energy_j': energy_this_step,
            'charge': snapshot_after.dram_charge,
            'lr_mult': lr_mult,
            'exit_layer': exit_layer,
        }

    def train_epoch(
        self,
        data: str,
        seq_len: int = 64,
        batch_size: int = 4,
    ) -> dict:
        """
        Train for one epoch on character data.
        """
        self.epoch += 1
        epoch_losses = []

        # Convert to tokens
        all_tokens = [ord(c) for c in data if 0 <= ord(c) < 256]

        # Create batches
        n_batches = (len(all_tokens) - seq_len - 1) // (seq_len * batch_size)

        print(f"\nEpoch {self.epoch}: {n_batches} batches")

        for batch_idx in range(min(n_batches, 100)):  # Cap at 100 batches
            # Build batch
            batch_tokens = []
            batch_targets = []

            for b in range(batch_size):
                start = (batch_idx * batch_size + b) * seq_len
                seq = all_tokens[start:start + seq_len]
                tgt = all_tokens[start + 1:start + seq_len + 1]

                if len(seq) == seq_len and len(tgt) == seq_len:
                    batch_tokens.append(seq)
                    batch_targets.append(tgt[-1])  # Predict last token

            if not batch_tokens:
                continue

            tokens = torch.tensor(batch_tokens, dtype=torch.long)
            targets = torch.tensor(batch_targets, dtype=torch.long)

            # Train step
            info = self.train_step(tokens, targets)
            epoch_losses.append(info['embodied_loss'])

            if (batch_idx + 1) % 20 == 0:
                avg_loss = sum(epoch_losses[-20:]) / len(epoch_losses[-20:])
                print(f"  Batch {batch_idx+1}/{min(n_batches, 100)}: "
                      f"loss={avg_loss:.4f}, energy={info['energy_j']*1000:.2f}mJ, "
                      f"charge={info['charge']:.2f}, lr_mult={info['lr_mult']:.3f}")

        return {
            'epoch': self.epoch,
            'avg_loss': sum(epoch_losses) / len(epoch_losses) if epoch_losses else 0,
            'n_batches': len(epoch_losses),
        }


def main():
    print("=" * 70)
    print("z1137: EMBODIED TRAINING - Learning Shaped by Hardware")
    print("=" * 70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nDevice: {device}")

    # Initialize systems
    print("\nInitializing embodied systems...")
    sense = SensorySystem()
    feel = FeelSystem()
    model = ExpressSystem(vocab_size=256, hidden_dim=128, n_layers=4)

    trainer = EmbodiedTrainer(model, feel, sense, learning_rate=1e-3, device=device)

    # Load training data
    data_path = Path('data/tiny_shakespeare.txt')
    if data_path.exists():
        data = data_path.read_text()[:50000]  # First 50K chars
        print(f"Loaded {len(data)} characters from {data_path}")
    else:
        # Fallback to generated data
        data = "The embodied machine learns through feeling. " * 1000
        print("Using generated training data")

    # Train for a few epochs
    print("\n" + "=" * 70)
    print("TRAINING WITH EMBODIMENT")
    print("=" * 70)

    sense.start_continuous()

    try:
        for epoch in range(3):
            epoch_info = trainer.train_epoch(data, seq_len=64, batch_size=4)
            print(f"\nEpoch {epoch_info['epoch']} complete: avg_loss={epoch_info['avg_loss']:.4f}")

            # Report embodiment effects
            recent = 20
            if len(trainer.metrics['modulation']) >= recent:
                avg_mod = sum(trainer.metrics['modulation'][-recent:]) / recent
                avg_energy = sum(trainer.metrics['energy_j'][-recent:]) / recent
                avg_charge = sum(trainer.metrics['charge'][-recent:]) / recent
                print(f"  Embodiment: modulation={avg_mod:.3f}, energy={avg_energy*1000:.2f}mJ, charge={avg_charge:.2f}")

    finally:
        sense.stop_continuous()

    # Test generation after training
    print("\n" + "=" * 70)
    print("POST-TRAINING GENERATION")
    print("=" * 70)

    # Create full loop for generation
    loop = EmbodiedLoop(vocab_size=256, hidden_dim=128, device=device)
    loop.feel = feel
    loop.express = model
    loop.sense = sense

    prompt = "The embodied machine"
    generated, history = loop.run_loop(prompt, num_tokens=50, temperature=0.8)
    print(f"\nGenerated: {generated}")

    # Save results
    results = {
        'experiment': 'z1137_embodied_training',
        'timestamp': datetime.now().isoformat(),
        'device': device,
        'epochs': trainer.epoch,
        'total_steps': trainer.step,
        'final_loss': trainer.metrics['task_loss'][-1] if trainer.metrics['task_loss'] else 0,
        'metrics_sample': {
            'task_loss': trainer.metrics['task_loss'][-20:],
            'embodied_loss': trainer.metrics['embodied_loss'][-20:],
            'energy_j': trainer.metrics['energy_j'][-20:],
            'modulation': trainer.metrics['modulation'][-20:],
        },
        'generated': generated,
    }

    results_path = Path('results/z1137_embodied_training.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}")

    # Summary
    print("\n" + "=" * 70)
    print("EMBODIED TRAINING COMPLETE")
    print("=" * 70)
    print(f"  Total steps: {trainer.step}")
    print(f"  Final task loss: {results['final_loss']:.4f}")

    if trainer.metrics['modulation']:
        print(f"  Average modulation: {sum(trainer.metrics['modulation'])/len(trainer.metrics['modulation']):.3f}")
        print(f"  Max modulation: {max(trainer.metrics['modulation']):.3f}")

    return 0


if __name__ == '__main__':
    sys.exit(main())
