#!/usr/bin/env python3
"""
z1000: Predictive Coding Baseline - Free Energy Minimization
============================================================

Proves that combined free energy loss (task + energy prediction + state prediction)
leads to better outcomes than task loss alone.

Hypothesis: Minimizing prediction error across ALL modalities (text AND hardware)
produces models that are both smarter AND more energy-efficient.

Reuses:
- src/metabolic/film_transformer.py (FiLM backbone)
- src/atom/feel.py (BodyStateTracker)
- src/telemetry/sysfs_hwmon.py (EnergyMeter)

New:
- Energy prediction head with uncertainty
- Next-state prediction head (self-modeling)
- Combined free energy loss

Conditions:
A: Task loss only (baseline)
B: Task + energy prediction
C: Task + energy + state prediction (full free energy)

Author: FEEL Research Team
Date: 2026-01-29
"""

import os
import sys
import json
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import Dict, Any, List, Optional, Tuple
from collections import deque

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from telemetry.sysfs_hwmon import SysfsHwmonTelemetry, EnergyMeter, GpuSample


# ============================================================================
# Body State Tracker (from src/atom/feel.py - simplified for this experiment)
# ============================================================================

@dataclass
class BodyState:
    """Processed body state."""
    power_w: float = 0.0
    temp_c: float = 0.0
    util_pct: float = 0.0
    power_ema: float = 0.0
    temp_ema: float = 0.0
    power_deriv: float = 0.0
    temp_deriv: float = 0.0

    def to_tensor(self, device='cpu') -> torch.Tensor:
        """Convert to normalized tensor [7 dims]."""
        return torch.tensor([
            self.power_w / 300.0,
            self.temp_c / 100.0,
            self.util_pct / 100.0,
            self.power_ema / 300.0,
            self.temp_ema / 100.0,
            self.power_deriv / 100.0,
            self.temp_deriv / 10.0,
        ], dtype=torch.float32, device=device)


class BodyStateTracker:
    """Tracks and processes hardware state."""

    def __init__(self, ema_alpha: float = 0.1):
        self.alpha = ema_alpha
        self.state = BodyState()
        self.prev_power = 0.0
        self.prev_temp = 0.0
        self.prev_time = time.time()
        self.initialized = False

    def update(self, sample: GpuSample) -> BodyState:
        """Update state from raw sample."""
        now = time.time()
        dt = max(now - self.prev_time, 0.001)

        if not self.initialized:
            self.state.power_w = sample.power_w
            self.state.temp_c = sample.temp_edge_c
            self.state.util_pct = sample.gpu_busy_pct
            self.state.power_ema = sample.power_w
            self.state.temp_ema = sample.temp_edge_c
            self.prev_power = sample.power_w
            self.prev_temp = sample.temp_edge_c
            self.initialized = True
        else:
            # Update raw
            self.state.power_w = sample.power_w
            self.state.temp_c = sample.temp_edge_c
            self.state.util_pct = sample.gpu_busy_pct

            # EMA
            self.state.power_ema = self.alpha * sample.power_w + (1 - self.alpha) * self.state.power_ema
            self.state.temp_ema = self.alpha * sample.temp_edge_c + (1 - self.alpha) * self.state.temp_ema

            # Derivatives
            self.state.power_deriv = (sample.power_w - self.prev_power) / dt
            self.state.temp_deriv = (sample.temp_edge_c - self.prev_temp) / dt

            self.prev_power = sample.power_w
            self.prev_temp = sample.temp_edge_c

        self.prev_time = now
        return self.state


# ============================================================================
# Predictive FiLM Transformer
# ============================================================================

class FiLMLayer(nn.Module):
    """Feature-wise Linear Modulation layer."""

    def __init__(self, hidden_size: int, body_dim: int = 7):
        super().__init__()
        self.gamma = nn.Linear(body_dim, hidden_size)
        self.beta = nn.Linear(body_dim, hidden_size)

    def forward(self, x: torch.Tensor, body_state: torch.Tensor) -> torch.Tensor:
        gamma = self.gamma(body_state).unsqueeze(1)  # [B, 1, H]
        beta = self.beta(body_state).unsqueeze(1)
        return x * (1 + gamma) + beta


class PredictiveTransformerBlock(nn.Module):
    """Transformer block with FiLM conditioning."""

    def __init__(self, hidden_size: int, n_heads: int = 4, body_dim: int = 7):
        super().__init__()
        self.attn = nn.MultiheadAttention(hidden_size, n_heads, batch_first=True)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.GELU(),
            nn.Linear(hidden_size * 4, hidden_size),
        )
        self.ln1 = nn.LayerNorm(hidden_size)
        self.ln2 = nn.LayerNorm(hidden_size)
        self.film = FiLMLayer(hidden_size, body_dim)

    def forward(self, x: torch.Tensor, body_state: torch.Tensor) -> torch.Tensor:
        # Self-attention
        residual = x
        x = self.ln1(x)
        x = self.film(x, body_state)  # FiLM conditioning
        x, _ = self.attn(x, x, x, need_weights=False)
        x = residual + x

        # MLP
        residual = x
        x = self.ln2(x)
        x = self.mlp(x)
        x = residual + x

        return x


class PredictiveTransformer(nn.Module):
    """
    Transformer with predictive coding heads.

    Outputs:
    - logits: Next token prediction
    - energy_mean: Predicted energy cost
    - energy_logvar: Uncertainty in energy prediction
    - next_state: Predicted next body state
    """

    def __init__(
        self,
        vocab_size: int,
        hidden_size: int = 256,
        n_layers: int = 4,
        n_heads: int = 4,
        body_dim: int = 7,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.body_dim = body_dim

        # Token embedding
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.pos_embedding = nn.Embedding(512, hidden_size)

        # Transformer blocks with FiLM
        self.blocks = nn.ModuleList([
            PredictiveTransformerBlock(hidden_size, n_heads, body_dim)
            for _ in range(n_layers)
        ])

        self.ln_out = nn.LayerNorm(hidden_size)

        # Output heads
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)

        # Predictive coding heads
        self.energy_head = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Linear(64, 2),  # [mean, log_var]
        )

        self.state_head = nn.Sequential(
            nn.Linear(hidden_size + body_dim, 64),
            nn.ReLU(),
            nn.Linear(64, body_dim),  # Predict next body state
        )

    def forward(
        self,
        tokens: torch.Tensor,
        body_state: torch.Tensor,
        return_predictions: bool = False,
    ) -> Dict[str, torch.Tensor]:
        B, T = tokens.shape
        device = tokens.device

        # Embeddings
        x = self.embedding(tokens)
        positions = torch.arange(T, device=device).unsqueeze(0)
        x = x + self.pos_embedding(positions)

        # Transformer blocks
        for block in self.blocks:
            x = block(x, body_state)

        x = self.ln_out(x)

        # Language modeling head
        logits = self.lm_head(x)

        result = {'logits': logits}

        if return_predictions:
            # Pool hidden states
            pooled = x.mean(dim=1)  # [B, H]

            # Energy prediction with uncertainty
            energy_out = self.energy_head(pooled)
            result['energy_mean'] = energy_out[:, 0]
            result['energy_logvar'] = energy_out[:, 1]

            # Next state prediction
            state_input = torch.cat([pooled, body_state], dim=-1)
            result['next_state'] = self.state_head(state_input)

        return result


# ============================================================================
# Free Energy Loss
# ============================================================================

class FreeEnergyLoss(nn.Module):
    """
    Combined free energy loss.

    F = α_task * L_task + α_energy * L_energy + α_state * L_state

    Where:
    - L_task: Cross-entropy for next token
    - L_energy: Gaussian NLL for energy prediction
    - L_state: MSE for next state prediction
    """

    def __init__(
        self,
        alpha_task: float = 1.0,
        alpha_energy: float = 0.1,
        alpha_state: float = 0.1,
    ):
        super().__init__()
        self.alpha_task = alpha_task
        self.alpha_energy = alpha_energy
        self.alpha_state = alpha_state

    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, float]]:

        # Task loss
        logits = outputs['logits'][:, :-1].contiguous()  # [B, T-1, V]
        target_tokens = targets['tokens'][:, 1:].contiguous()  # [B, T-1]
        task_loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            target_tokens.view(-1),
        )

        # Energy prediction loss (if available)
        if 'energy_mean' in outputs and 'energy_j' in targets:
            energy_mean = outputs['energy_mean']
            energy_logvar = outputs['energy_logvar']
            energy_target = targets['energy_j']

            # Gaussian NLL
            energy_var = energy_logvar.exp()
            energy_loss = 0.5 * (
                energy_logvar +
                (energy_target - energy_mean).pow(2) / energy_var
            ).mean()
        else:
            energy_loss = torch.tensor(0.0, device=logits.device)

        # State prediction loss (if available)
        if 'next_state' in outputs and 'next_body_state' in targets:
            state_loss = F.mse_loss(outputs['next_state'], targets['next_body_state'])
        else:
            state_loss = torch.tensor(0.0, device=logits.device)

        # Combined free energy
        free_energy = (
            self.alpha_task * task_loss +
            self.alpha_energy * energy_loss +
            self.alpha_state * state_loss
        )

        metrics = {
            'task_loss': task_loss.item(),
            'energy_loss': energy_loss.item(),
            'state_loss': state_loss.item(),
            'free_energy': free_energy.item(),
        }

        return free_energy, metrics


# ============================================================================
# Data
# ============================================================================

def load_tinyshakespeare() -> str:
    """Load TinyShakespeare dataset."""
    data_paths = [
        Path(__file__).parent.parent / "data" / "tinyshakespeare.txt",
        Path(__file__).parent.parent / "data" / "tiny_shakespeare.txt",
        Path(__file__).parent.parent / "data" / "tinyshakespeare" / "input.txt",
    ]
    for path in data_paths:
        if path.exists():
            return path.read_text()

    # Download if not found
    import urllib.request
    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    data_dir = Path(__file__).parent.parent / "data"
    data_dir.mkdir(exist_ok=True)
    path = data_dir / "tinyshakespeare.txt"
    urllib.request.urlretrieve(url, path)
    return path.read_text()


def make_char_tokenizer(text: str):
    """Create character-level tokenizer."""
    chars = sorted(list(set(text)))
    char_to_idx = {c: i for i, c in enumerate(chars)}
    idx_to_char = {i: c for c, i in char_to_idx.items()}
    return char_to_idx, idx_to_char, len(chars)


# ============================================================================
# Training
# ============================================================================

def train_condition(
    condition: str,
    model: PredictiveTransformer,
    text: str,
    char_to_idx: Dict[str, int],
    device: torch.device,
    telemetry: SysfsHwmonTelemetry,
    n_steps: int = 500,
    batch_size: int = 32,
    seq_len: int = 64,
    lr: float = 3e-4,
) -> Dict[str, Any]:
    """
    Train under one condition.

    Conditions:
    A: Task loss only (alpha_energy=0, alpha_state=0)
    B: Task + energy (alpha_energy=0.1, alpha_state=0)
    C: Full free energy (alpha_energy=0.1, alpha_state=0.1)
    """

    # Configure loss
    if condition == 'A':
        loss_fn = FreeEnergyLoss(alpha_task=1.0, alpha_energy=0.0, alpha_state=0.0)
        use_predictions = False
    elif condition == 'B':
        loss_fn = FreeEnergyLoss(alpha_task=1.0, alpha_energy=0.1, alpha_state=0.0)
        use_predictions = True
    elif condition == 'C':
        loss_fn = FreeEnergyLoss(alpha_task=1.0, alpha_energy=0.1, alpha_state=0.1)
        use_predictions = True
    else:
        raise ValueError(f"Unknown condition: {condition}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    tracker = BodyStateTracker()

    # Encode text
    data = torch.tensor([char_to_idx[c] for c in text], dtype=torch.long, device=device)

    # Training loop
    metrics_history = []
    energy_predictions = []
    energy_actuals = []

    model.train()

    for step in range(n_steps):
        # Sample batch
        starts = torch.randint(0, len(data) - seq_len - 1, (batch_size,))
        batch = torch.stack([data[s:s+seq_len] for s in starts])

        # Get body state
        sample = telemetry.read_sample()
        body = tracker.update(sample)
        body_tensor = body.to_tensor(device).unsqueeze(0).expand(batch_size, -1)

        # Forward with energy measurement
        with EnergyMeter(telemetry) as meter:
            outputs = model(batch, body_tensor, return_predictions=use_predictions)

        actual_energy_j = meter.energy_j / batch_size  # Per-sample

        # Get next body state (for state prediction target)
        time.sleep(0.001)  # Small delay for state change
        next_sample = telemetry.read_sample()
        next_body = tracker.update(next_sample)
        next_body_tensor = next_body.to_tensor(device).unsqueeze(0).expand(batch_size, -1)

        # Targets
        targets = {
            'tokens': batch,
            'energy_j': torch.full((batch_size,), actual_energy_j, device=device),
            'next_body_state': next_body_tensor,
        }

        # Loss
        loss, step_metrics = loss_fn(outputs, targets)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        # Track metrics
        step_metrics['actual_energy_j'] = actual_energy_j
        step_metrics['step'] = step
        metrics_history.append(step_metrics)

        if use_predictions and 'energy_mean' in outputs:
            pred = outputs['energy_mean'].mean().item()
            energy_predictions.append(pred)
            energy_actuals.append(actual_energy_j)

        if step % 50 == 0:
            ppl = np.exp(step_metrics['task_loss'])
            print(f"    Step {step}: PPL={ppl:.2f}, F={step_metrics['free_energy']:.4f}")

    # Final evaluation
    model.eval()
    eval_losses = []
    with torch.no_grad():
        for _ in range(20):
            starts = torch.randint(0, len(data) - seq_len - 1, (batch_size,))
            batch = torch.stack([data[s:s+seq_len] for s in starts])
            sample = telemetry.read_sample()
            body = tracker.update(sample)
            body_tensor = body.to_tensor(device).unsqueeze(0).expand(batch_size, -1)

            outputs = model(batch, body_tensor, return_predictions=False)
            task_loss = F.cross_entropy(
                outputs['logits'][:, :-1].contiguous().view(-1, outputs['logits'].size(-1)),
                batch[:, 1:].contiguous().view(-1)
            )
            eval_losses.append(task_loss.item())

    final_ppl = np.exp(np.mean(eval_losses))

    # Energy prediction accuracy
    if energy_predictions:
        energy_mape = np.mean([
            abs(p - a) / max(a, 1e-6)
            for p, a in zip(energy_predictions, energy_actuals)
        ])
    else:
        energy_mape = None

    return {
        'condition': condition,
        'final_ppl': final_ppl,
        'energy_mape': energy_mape,
        'metrics_history': metrics_history,
        'n_steps': n_steps,
    }


# ============================================================================
# Main
# ============================================================================

def main():
    print("=" * 70)
    print("z1000: PREDICTIVE CODING BASELINE")
    print("=" * 70)
    print("\nHypothesis: Combined free energy loss (task + energy + state)")
    print("produces better models than task loss alone.\n")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load data
    print("\nLoading TinyShakespeare...")
    text = load_tinyshakespeare()
    char_to_idx, idx_to_char, vocab_size = make_char_tokenizer(text)
    print(f"  Vocab size: {vocab_size}")
    print(f"  Text length: {len(text):,} chars")

    # Initialize telemetry
    telemetry = SysfsHwmonTelemetry()
    print(f"  Telemetry: {telemetry}")

    results = {}
    n_steps = 300

    for condition in ['A', 'B', 'C']:
        label = {
            'A': 'Task loss only',
            'B': 'Task + energy prediction',
            'C': 'Full free energy',
        }[condition]

        print(f"\n{'='*60}")
        print(f"CONDITION {condition}: {label}")
        print("=" * 60)

        # Fresh model for each condition
        model = PredictiveTransformer(
            vocab_size=vocab_size,
            hidden_size=256,
            n_layers=4,
            n_heads=4,
            body_dim=7,
        ).to(device)

        result = train_condition(
            condition=condition,
            model=model,
            text=text,
            char_to_idx=char_to_idx,
            device=device,
            telemetry=telemetry,
            n_steps=n_steps,
        )

        results[condition] = result
        print(f"\n  Final PPL: {result['final_ppl']:.2f}")
        if result['energy_mape'] is not None:
            print(f"  Energy MAPE: {result['energy_mape']*100:.1f}%")

    # Summary
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)

    print("\n| Condition | Description | Final PPL | Energy MAPE |")
    print("|-----------|-------------|-----------|-------------|")
    for c in ['A', 'B', 'C']:
        r = results[c]
        label = {
            'A': 'Task only',
            'B': 'Task + energy',
            'C': 'Full free energy',
        }[c]
        mape = f"{r['energy_mape']*100:.1f}%" if r['energy_mape'] else "N/A"
        print(f"| {c} | {label:15} | {r['final_ppl']:9.2f} | {mape:11} |")

    # Verdict
    print("\n" + "=" * 70)
    print("VERDICT")
    print("=" * 70)

    ppl_a = results['A']['final_ppl']
    ppl_c = results['C']['final_ppl']

    if ppl_c < ppl_a:
        print(f"\n✅ FREE ENERGY WINS: PPL {ppl_c:.2f} < {ppl_a:.2f}")
        print("   Combined loss produces better language model!")
    else:
        print(f"\n⚠️ BASELINE WINS: PPL {ppl_a:.2f} < {ppl_c:.2f}")
        print("   Need to tune alpha weights or architecture.")

    # Save results
    results_path = Path(__file__).parent.parent / "results" / "z1000_predictive_baseline.json"
    results_path.parent.mkdir(exist_ok=True)

    # Convert to serializable format
    save_results = {
        'experiment': 'z1000_predictive_baseline',
        'timestamp': datetime.now().isoformat(),
        'n_steps': n_steps,
        'conditions': {
            c: {
                'condition': c,
                'final_ppl': r['final_ppl'],
                'energy_mape': r['energy_mape'],
            }
            for c, r in results.items()
        },
        'verdict': 'FREE_ENERGY_WINS' if ppl_c < ppl_a else 'BASELINE_WINS',
    }

    with open(results_path, 'w') as f:
        json.dump(save_results, f, indent=2)

    print(f"\nResults saved to: {results_path}")


if __name__ == "__main__":
    main()
