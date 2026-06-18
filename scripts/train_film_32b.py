#!/usr/bin/env python3
"""
Train FiLM and InteroceptiveBody for DeepSeek 32B

The 32B model has:
- hidden_size: 5120
- num_hidden_layers: 64

We train the small controller modules (FiLM generator + Body GRU) on CPU
with synthetic data. This takes ~2 minutes and produces weights that work
for the 32B model's larger hidden dimension.

Usage:
    python scripts/train_film_32b.py
"""

import argparse
import random
import time
from pathlib import Path
from dataclasses import dataclass
from enum import Enum
from typing import List, Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# ============================================================================
# INTEROCEPTIVE BODY (same as parallel_worlds_demo but standalone for training)
# ============================================================================

class InteroceptiveBody(nn.Module):
    """GRU-based body that feels - maintains z_feel state."""

    def __init__(self, signal_dim: int = 8, z_dim: int = 32, device='cpu'):
        super().__init__()
        self.z_dim = z_dim
        self.device = device

        # Signal encoder
        self.signal_encoder = nn.Sequential(
            nn.Linear(signal_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
        )

        # Learned regime embeddings
        self.z_cool = nn.Parameter(torch.randn(z_dim) * 0.1)
        self.z_warm = nn.Parameter(torch.randn(z_dim) * 0.1)
        self.z_hot = nn.Parameter(torch.randn(z_dim) * 0.1)

        # GRU for temporal memory
        self.gru = nn.GRU(64, z_dim, batch_first=True)

        # Projections
        self.stress_proj = nn.Linear(z_dim, 1)
        self.error_predictor = nn.Linear(z_dim, 1)
        self.recovery_gate = nn.Linear(z_dim, 1)
        self.regime_head = nn.Linear(z_dim, 4)

        self.to(device)
        self.h = None

    def init_state(self, batch_size: int = 1):
        return torch.zeros(batch_size, self.z_dim, device=self.device)

    def forward(self, signals: torch.Tensor, prev_state: torch.Tensor = None):
        """Process signals and update state."""
        if prev_state is None:
            prev_state = self.init_state(signals.size(0))

        encoded = self.signal_encoder(signals).unsqueeze(1)

        # GRU expects [batch, seq, features] for input, [num_layers, batch, hidden] for hidden
        h = prev_state.unsqueeze(0)  # [1, batch, z_dim]
        z_out, h_new = self.gru(encoded, h)
        z = z_out.squeeze(1)  # [batch, z_dim]

        # Regime logits
        regime_logits = self.regime_head(z)
        stress = torch.sigmoid(self.stress_proj(z)).squeeze(-1)
        error_risk = torch.sigmoid(self.error_predictor(z)).squeeze(-1)
        recovery = torch.sigmoid(self.recovery_gate(z)).squeeze(-1)

        return z, {
            'regime_logits': regime_logits,
            'regime': regime_logits.argmax(dim=-1),
            'stress': stress,
            'error_risk': error_risk,
            'recovery': recovery,
        }


# ============================================================================
# FiLM GENERATOR FOR 32B MODEL
# ============================================================================

class FiLMGenerator(nn.Module):
    """
    Generates FiLM parameters (gamma, beta) for 32B model.

    Key change: hidden_dim = 5120 for DeepSeek 32B
    """

    def __init__(
        self,
        hidden_dim: int = 5120,  # DeepSeek 32B hidden size
        z_dim: int = 32,
        n_layers: int = 4,
    ):
        super().__init__()
        self.z_dim = z_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        # Shared z encoder
        self.z_encoder = nn.Sequential(
            nn.Linear(z_dim, 128),
            nn.GELU(),
            nn.Linear(128, 128),
            nn.GELU(),
        )

        # Per-layer FiLM heads
        # These are the big ones: 128 -> 5120 each
        self.gamma_heads = nn.ModuleList([
            nn.Linear(128, hidden_dim) for _ in range(n_layers)
        ])
        self.beta_heads = nn.ModuleList([
            nn.Linear(128, hidden_dim) for _ in range(n_layers)
        ])

        # Initialize for stable identity-ish transform
        for gamma_head in self.gamma_heads:
            nn.init.normal_(gamma_head.weight, std=0.02)
            nn.init.ones_(gamma_head.bias)
        for beta_head in self.beta_heads:
            nn.init.normal_(beta_head.weight, std=0.02)
            nn.init.zeros_(beta_head.bias)

    def forward(self, z: torch.Tensor, ablate: bool = False):
        """Generate (gamma, beta) tuples for each layer."""
        batch_size = z.size(0)

        if ablate:
            return [
                (torch.ones(batch_size, self.hidden_dim, device=z.device),
                 torch.zeros(batch_size, self.hidden_dim, device=z.device))
                for _ in range(self.n_layers)
            ]

        z_enc = self.z_encoder(z)

        film_params = []
        for i in range(self.n_layers):
            gamma = self.gamma_heads[i](z_enc)
            beta = self.beta_heads[i](z_enc)
            film_params.append((gamma, beta))

        return film_params


# ============================================================================
# SYNTHETIC TRAINING DATA
# ============================================================================

class FeltRegime(Enum):
    COMFORTABLE = 0
    WARM = 1
    HOT = 2
    DISTRESSED = 3


def generate_regime_signals(regime: FeltRegime, signal_dim: int = 8) -> torch.Tensor:
    """Generate signals typical of a regime."""
    # Base signal: [entropy, margin, stress, confidence, throughput, attn_entropy, grad_norm, mem_pressure]
    base = torch.tensor([0.4, 0.7, 0.2, 0.8, 0.7, 0.4, 0.1, 0.2])

    if regime == FeltRegime.COMFORTABLE:
        pass  # base is comfortable
    elif regime == FeltRegime.WARM:
        base[0] += 0.2  # higher entropy
        base[1] -= 0.1  # lower margin
        base[2] += 0.2  # higher stress
        base[4] -= 0.15  # lower throughput
    elif regime == FeltRegime.HOT:
        base[0] += 0.4
        base[1] -= 0.3
        base[2] += 0.5
        base[3] -= 0.3
        base[4] -= 0.3
    elif regime == FeltRegime.DISTRESSED:
        base[0] += 0.5
        base[1] -= 0.5
        base[2] += 0.7
        base[3] -= 0.5
        base[4] -= 0.5

    # Add noise
    noise = torch.randn_like(base) * 0.1
    signals = (base + noise).clamp(0, 1)

    return signals


def generate_training_data(
    n_samples: int = 500,
    seq_len: int = 16,
    signal_dim: int = 8,
    device: str = 'cpu',
) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    """Generate sequences of (signals, regime_labels) for training."""
    data = []

    for _ in range(n_samples):
        # Random regime trajectory with transitions
        regime_seq = []
        current = random.randint(0, 3)

        for t in range(seq_len):
            if random.random() < 0.2:
                # Transition
                current = max(0, min(3, current + random.choice([-1, 0, 1])))
            regime_seq.append(current)

        # Generate signals for each regime
        signals_list = []
        for r in regime_seq:
            signals = generate_regime_signals(FeltRegime(r), signal_dim)
            signals_list.append(signals)

        signals_tensor = torch.stack(signals_list).to(device)
        regime_tensor = torch.tensor(regime_seq, dtype=torch.long, device=device)

        data.append((signals_tensor, regime_tensor))

    return data


# ============================================================================
# TRAINING
# ============================================================================

def train(
    hidden_dim: int = 5120,
    z_dim: int = 32,
    n_layers: int = 4,
    n_samples: int = 500,
    epochs: int = 50,
    lr: float = 1e-3,
    output_dir: Path = None,
):
    """Train InteroceptiveBody and FiLMGenerator."""
    print("="*60)
    print(f"Training FiLM for DeepSeek 32B (hidden_dim={hidden_dim})")
    print("="*60)

    device = 'cpu'  # Training is on CPU, inference on GPU

    # Create modules
    body = InteroceptiveBody(signal_dim=8, z_dim=z_dim, device=device)
    film = FiLMGenerator(hidden_dim=hidden_dim, z_dim=z_dim, n_layers=n_layers)

    # Count parameters
    body_params = sum(p.numel() for p in body.parameters())
    film_params = sum(p.numel() for p in film.parameters())
    print(f"\nBody parameters: {body_params:,}")
    print(f"FiLM parameters: {film_params:,}")
    print(f"Total: {body_params + film_params:,}")

    # Generate training data
    print(f"\nGenerating {n_samples} training sequences...")
    t0 = time.time()
    data = generate_training_data(n_samples=n_samples, device=device)
    print(f"Generated in {time.time() - t0:.1f}s")

    # Optimizer
    params = list(body.parameters()) + list(film.parameters())
    optimizer = torch.optim.Adam(params, lr=lr)

    print(f"\nTraining for {epochs} epochs...")
    history = {'loss': [], 'regime_acc': []}

    for epoch in range(epochs):
        body.train()
        film.train()

        total_loss = 0
        correct = 0
        total = 0

        for signals_seq, regime_labels in data:
            optimizer.zero_grad()

            # Initialize state
            z = body.init_state(1)
            seq_losses = []

            for t in range(signals_seq.size(0)):
                signal = signals_seq[t].unsqueeze(0)
                label = regime_labels[t]

                # Forward body
                z, outputs = body(signal, z)

                # Regime loss
                loss_regime = F.cross_entropy(outputs['regime_logits'], label.unsqueeze(0))
                seq_losses.append(loss_regime)

                # Track accuracy
                pred = outputs['regime_logits'].argmax(dim=-1)
                correct += (pred == label).sum().item()
                total += 1

            # FiLM stability loss (once per sequence)
            film_params_list = film(z)
            loss_film = 0
            for gamma, beta in film_params_list:
                # gamma should stay near 1
                loss_gamma = F.mse_loss(gamma.mean(), torch.tensor(1.0))
                # beta should stay small
                loss_beta = beta.abs().mean()
                loss_film += 0.05 * loss_gamma + 0.05 * loss_beta
            seq_losses.append(loss_film)

            # Backprop
            loss = sum(seq_losses) / len(seq_losses)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(data)
        acc = correct / total if total > 0 else 0
        history['loss'].append(avg_loss)
        history['regime_acc'].append(acc)

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1:3d}: loss={avg_loss:.4f}, regime_acc={acc:.1%}")

    # Save models
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)

        body_path = output_dir / "recurrent_state_32b.pt"
        film_path = output_dir / "film_generator_32b.pt"

        torch.save(body.state_dict(), body_path)
        torch.save(film.state_dict(), film_path)

        print(f"\nSaved: {body_path}")
        print(f"Saved: {film_path}")

        # Save config
        config = {
            'hidden_dim': hidden_dim,
            'z_dim': z_dim,
            'n_layers': n_layers,
            'signal_dim': 8,
            'model': 'deepseek-ai/DeepSeek-R1-Distill-Qwen-32B',
        }
        import json
        with open(output_dir / "config_32b.json", 'w') as f:
            json.dump(config, f, indent=2)
        print(f"Saved: {output_dir / 'config_32b.json'}")

    print("\nTraining complete!")
    print(f"Final loss: {history['loss'][-1]:.4f}")
    print(f"Final accuracy: {history['regime_acc'][-1]:.1%}")

    return body, film, history


def main():
    parser = argparse.ArgumentParser(description='Train FiLM for DeepSeek 32B')
    parser.add_argument('--output-dir', type=str, default='results/closed_loop_interoception')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--samples', type=int, default=500)
    args = parser.parse_args()

    train(
        hidden_dim=5120,  # DeepSeek 32B
        z_dim=32,
        n_layers=4,
        n_samples=args.samples,
        epochs=args.epochs,
        output_dir=Path(args.output_dir),
    )


if __name__ == '__main__':
    main()
