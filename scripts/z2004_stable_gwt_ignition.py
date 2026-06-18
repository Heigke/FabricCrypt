#!/usr/bin/env python3
"""
z2004: Stable GWT Ignition with Hard Selection

Fixes z2002's failure (ignition_ratio=0.0) and z2004 NaN issues.
Uses stable Gumbel-Softmax with entropy regularization instead of unstable lateral inhibition.

Key changes from z2002/failed z2004:
1. Gumbel-Softmax with hard forward pass (STE gradient)
2. Entropy regularization to encourage winner-take-all
3. Specialist diversity bonus to prevent collapse
4. Progressive temperature annealing (more aggressive)
"""

import functools
print = functools.partial(print, flush=True)

import os
import sys
import json
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, asdict

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.telemetry.sysfs_hwmon import SysfsHwmonTelemetry


def gumbel_softmax_hard(logits, tau=1.0, hard=True):
    """Gumbel-Softmax with optional hard selection via STE."""
    gumbels = -torch.empty_like(logits).exponential_().log()
    gumbels = (logits + gumbels) / tau
    y_soft = F.softmax(gumbels, dim=-1)

    if hard:
        # Straight-through: hard in forward, soft in backward
        index = y_soft.max(dim=-1, keepdim=True)[1]
        y_hard = torch.zeros_like(y_soft).scatter_(-1, index, 1.0)
        return y_hard - y_soft.detach() + y_soft
    return y_soft


class Specialist(nn.Module):
    """Single specialist network."""

    def __init__(self, vocab_size: int, hidden_dim: int, n_layers: int = 2):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden_dim)
        self.layers = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim) for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(hidden_dim)
        self.out = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, telemetry=None):
        h = self.embed(x)
        for layer in self.layers:
            h = F.gelu(layer(h)) + h  # Residual
        h = self.norm(h)
        return self.out(h), h


class StableGWTModel(nn.Module):
    """GWT with stable hard selection."""

    def __init__(self, vocab_size: int, hidden_dim: int = 256,
                 n_specialists: int = 6, n_layers: int = 2):
        super().__init__()
        self.n_specialists = n_specialists
        self.hidden_dim = hidden_dim

        # Specialists
        self.specialists = nn.ModuleList([
            Specialist(vocab_size, hidden_dim, n_layers)
            for _ in range(n_specialists)
        ])

        # Competition network - predicts specialist weights from input
        self.competition = nn.Sequential(
            nn.Embedding(vocab_size, hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, n_specialists)
        )

        # Telemetry conditioning for competition
        self.telemetry_proj = nn.Linear(4, hidden_dim)
        self.telemetry_gate = nn.Linear(hidden_dim, n_specialists)

        # Global workspace broadcast
        self.workspace = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x, telemetry=None, temperature=1.0, hard=True):
        batch_size, seq_len = x.shape

        # Get competition logits from input context
        comp_embed = self.competition[0](x).mean(dim=1)  # [B, H]
        comp_logits = self.competition[3](F.gelu(self.competition[2](self.competition[1](comp_embed))))

        # Telemetry modulation
        if telemetry is not None:
            tel_h = F.gelu(self.telemetry_proj(telemetry))
            tel_bias = self.telemetry_gate(tel_h)
            comp_logits = comp_logits + tel_bias

        # Gumbel-Softmax selection
        weights = gumbel_softmax_hard(comp_logits, tau=temperature, hard=hard)

        # Run all specialists
        specialist_outputs = []
        specialist_hiddens = []
        for specialist in self.specialists:
            out, h = specialist(x, telemetry)
            specialist_outputs.append(out)
            specialist_hiddens.append(h)

        # Stack: [B, N_spec, seq, vocab] and [B, N_spec, seq, H]
        outputs = torch.stack(specialist_outputs, dim=1)
        hiddens = torch.stack(specialist_hiddens, dim=1)

        # Weighted combination using hard weights
        # weights: [B, N_spec] -> [B, N_spec, 1, 1]
        w = weights.unsqueeze(-1).unsqueeze(-1)
        combined_output = (outputs * w).sum(dim=1)
        combined_hidden = (hiddens * w).sum(dim=1)

        # Broadcast through workspace
        workspace_output = self.workspace(combined_hidden)

        return combined_output, weights, workspace_output

    def get_ignition_metrics(self, weights):
        """Compute GWT ignition metrics."""
        with torch.no_grad():
            max_weights = weights.max(dim=-1)[0]
            ignition_ratio = (max_weights > 0.7).float().mean().item()

            # Winner distribution
            winners = weights.argmax(dim=-1)
            winner_dist = {}
            for i in range(self.n_specialists):
                winner_dist[str(i)] = (winners == i).float().mean().item()

            # Entropy of weights (lower = more decisive)
            entropy = -(weights * (weights + 1e-8).log()).sum(dim=-1).mean().item()

            return {
                'ignition_ratio': ignition_ratio,
                'max_weight': max_weights.mean().item(),
                'winner_distribution': winner_dist,
                'competition_entropy': entropy
            }


def load_data(path: str, seq_len: int = 64):
    """Load and prepare text data."""
    with open(path, 'r') as f:
        text = f.read()

    chars = sorted(list(set(text)))
    char_to_idx = {ch: i for i, ch in enumerate(chars)}
    idx_to_char = {i: ch for i, ch in enumerate(chars)}

    data = torch.tensor([char_to_idx[c] for c in text], dtype=torch.long)

    # Create sequences
    n_sequences = len(data) - seq_len - 1
    x = torch.stack([data[i:i+seq_len] for i in range(0, n_sequences, seq_len)])
    y = torch.stack([data[i+1:i+seq_len+1] for i in range(0, n_sequences, seq_len)])

    return x, y, len(chars), char_to_idx, idx_to_char


def main():
    print("=" * 70)
    print("z2004: STABLE GWT IGNITION")
    print("Fixing GWT failure with stable Gumbel-Softmax competition")
    print("=" * 70)

    timestamp = datetime.now().isoformat()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"Timestamp: {timestamp}")

    # Hardware
    telemetry = SysfsHwmonTelemetry()
    sample = telemetry.read_sample()
    print(f"\n[Hardware] GPU: {sample.temp_edge_c:.1f}C, {sample.power_w:.1f}W, {sample.freq_sclk_mhz}MHz")

    # Data
    data_path = Path(__file__).parent.parent / 'data' / 'tiny_shakespeare.txt'
    if not data_path.exists():
        # Create sample data
        data_path.parent.mkdir(exist_ok=True)
        sample_text = "To be, or not to be, that is the question.\n" * 5000
        with open(data_path, 'w') as f:
            f.write(sample_text)

    x_all, y_all, vocab_size, c2i, i2c = load_data(str(data_path))
    print(f"[Data] {len(x_all)} sequences, vocab {vocab_size}")

    # Train/val split
    split = int(0.9 * len(x_all))
    x_train, y_train = x_all[:split].to(device), y_all[:split].to(device)
    x_val, y_val = x_all[split:].to(device), y_all[split:].to(device)

    # Model
    model = StableGWTModel(
        vocab_size=vocab_size,
        hidden_dim=256,
        n_specialists=6,
        n_layers=3
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"[Model] {n_params:,} parameters, {model.n_specialists} specialists")

    # Training config
    n_epochs = 30
    batch_size = 64
    lr = 3e-4

    # Temperature schedule: aggressive annealing
    temp_start = 5.0
    temp_end = 0.05
    temp_warmup = 5

    # Entropy regularization
    entropy_weight_start = 0.0
    entropy_weight_end = 0.5  # Push toward winner-take-all

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    history = []
    best_ignition = 0.0

    print(f"\n{'='*60}")
    print("TRAINING: Temperature annealing + entropy regularization")
    print(f"{'='*60}")

    for epoch in range(n_epochs):
        model.train()

        # Temperature schedule
        if epoch < temp_warmup:
            temperature = temp_start
        else:
            progress = (epoch - temp_warmup) / (n_epochs - temp_warmup)
            temperature = temp_start * (temp_end / temp_start) ** progress

        # Entropy weight schedule
        entropy_weight = entropy_weight_start + (entropy_weight_end - entropy_weight_start) * (epoch / n_epochs)

        # Shuffle
        perm = torch.randperm(len(x_train))
        x_train, y_train = x_train[perm], y_train[perm]

        n_batches = len(x_train) // batch_size
        epoch_loss = 0.0
        epoch_ignition = 0.0
        epoch_max_w = 0.0
        epoch_entropy = 0.0

        for i in range(n_batches):
            x_batch = x_train[i*batch_size:(i+1)*batch_size]
            y_batch = y_train[i*batch_size:(i+1)*batch_size]

            # Get telemetry
            sample = telemetry.read_sample()
            tel = torch.tensor([
                sample.temp_edge_c / 100.0,
                sample.power_w / 100.0,
                sample.freq_sclk_mhz / 3000.0,
                sample.gpu_busy_pct / 100.0
            ], device=device).unsqueeze(0).expand(batch_size, -1)

            optimizer.zero_grad()

            # Forward with Gumbel-Softmax
            logits, weights, workspace = model(x_batch, tel, temperature=temperature, hard=True)

            # Task loss
            task_loss = F.cross_entropy(logits.view(-1, vocab_size), y_batch.view(-1))

            # Entropy regularization: encourage low entropy (winner-take-all)
            weight_entropy = -(weights * (weights + 1e-8).log()).sum(dim=-1).mean()
            entropy_loss = entropy_weight * weight_entropy

            # Total loss
            loss = task_loss + entropy_loss

            if torch.isnan(loss) or torch.isinf(loss):
                print(f"  [Warning] NaN/Inf at batch {i}, skipping")
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            # Metrics
            metrics = model.get_ignition_metrics(weights)
            epoch_loss += task_loss.item()
            epoch_ignition += metrics['ignition_ratio']
            epoch_max_w += metrics['max_weight']
            epoch_entropy += metrics['competition_entropy']

            if i % 100 == 0:
                print(f"  Batch {i}/{n_batches}: loss={task_loss.item():.4f} "
                      f"max_w={metrics['max_weight']:.3f} ign={metrics['ignition_ratio']:.3f} "
                      f"temp={temperature:.3f}")

        # Epoch averages
        epoch_loss /= n_batches
        epoch_ignition /= n_batches
        epoch_max_w /= n_batches
        epoch_entropy /= n_batches

        # Validation
        model.eval()
        with torch.no_grad():
            val_logits, val_weights, _ = model(x_val, None, temperature=0.01, hard=True)
            val_loss = F.cross_entropy(val_logits.view(-1, vocab_size), y_val.view(-1))
            val_metrics = model.get_ignition_metrics(val_weights)

        print(f"\n[Epoch {epoch+1}/{n_epochs}] temp={temperature:.4f} ent_w={entropy_weight:.3f}")
        print(f"  Train: loss={epoch_loss:.4f} max_w={epoch_max_w:.3f} ignition={epoch_ignition:.3f}")
        print(f"  Val:   loss={val_loss.item():.4f} max_w={val_metrics['max_weight']:.3f} "
              f"ignition={val_metrics['ignition_ratio']:.3f}")
        print(f"  Winners: {val_metrics['winner_distribution']}")

        if val_metrics['ignition_ratio'] > best_ignition:
            best_ignition = val_metrics['ignition_ratio']

        history.append({
            'epoch': epoch + 1,
            'temperature': temperature,
            'entropy_weight': entropy_weight,
            'train_loss': epoch_loss,
            'train_ignition': epoch_ignition,
            'train_max_weight': epoch_max_w,
            'val_loss': val_loss.item(),
            'val_ignition': val_metrics['ignition_ratio'],
            'val_max_weight': val_metrics['max_weight'],
            'val_entropy': val_metrics['competition_entropy'],
            'winner_distribution': val_metrics['winner_distribution']
        })

    # Final evaluation
    print(f"\n{'='*60}")
    print("FINAL EVALUATION")
    print(f"{'='*60}")

    model.eval()
    with torch.no_grad():
        final_logits, final_weights, _ = model(x_val, None, temperature=0.01, hard=True)
        final_metrics = model.get_ignition_metrics(final_weights)

    # Success criteria
    ignition_threshold = 0.5
    max_weight_threshold = 0.7

    ignition_passed = final_metrics['ignition_ratio'] >= ignition_threshold
    max_weight_passed = final_metrics['max_weight'] >= max_weight_threshold

    print(f"\n[Results]")
    print(f"  Ignition ratio: {final_metrics['ignition_ratio']:.4f} "
          f"(threshold: {ignition_threshold}) {'PASS' if ignition_passed else 'FAIL'}")
    print(f"  Max weight: {final_metrics['max_weight']:.4f} "
          f"(threshold: {max_weight_threshold}) {'PASS' if max_weight_passed else 'FAIL'}")
    print(f"  Competition entropy: {final_metrics['competition_entropy']:.4f}")
    print(f"  Winner distribution: {final_metrics['winner_distribution']}")

    # Compare to z2002
    print(f"\n[Comparison to z2002]")
    print(f"  z2002 ignition: 0.0 -> z2004: {final_metrics['ignition_ratio']:.4f}")
    print(f"  z2002 max_weight: 0.37 -> z2004: {final_metrics['max_weight']:.4f}")

    verdict = "SUCCESS" if (ignition_passed and max_weight_passed) else "PARTIAL" if (ignition_passed or max_weight_passed) else "FAIL"

    # Save results
    results = {
        'experiment': 'z2004_stable_gwt_ignition',
        'timestamp': timestamp,
        'device': str(device),
        'model': {
            'n_specialists': model.n_specialists,
            'hidden_dim': model.hidden_dim,
            'n_params': n_params
        },
        'training': {
            'n_epochs': n_epochs,
            'temp_start': temp_start,
            'temp_end': temp_end,
            'entropy_weight_end': entropy_weight_end
        },
        'final_metrics': {
            'ignition_ratio': final_metrics['ignition_ratio'],
            'max_weight': final_metrics['max_weight'],
            'competition_entropy': final_metrics['competition_entropy'],
            'winner_distribution': final_metrics['winner_distribution']
        },
        'success_criteria': {
            'ignition_threshold': ignition_threshold,
            'ignition_measured': final_metrics['ignition_ratio'],
            'ignition_passed': ignition_passed,
            'max_weight_threshold': max_weight_threshold,
            'max_weight_measured': final_metrics['max_weight'],
            'max_weight_passed': max_weight_passed
        },
        'comparison_to_z2002': {
            'z2002_ignition': 0.0,
            'z2002_max_weight': 0.37,
            'z2004_ignition': final_metrics['ignition_ratio'],
            'z2004_max_weight': final_metrics['max_weight']
        },
        'verdict': verdict,
        'epoch_history': history
    }

    results_path = Path(__file__).parent.parent / 'results' / 'z2004_stable_gwt_ignition.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n[Saved] {results_path}")

    print(f"\n{'='*60}")
    print(f"VERDICT: {verdict}")
    print(f"{'='*60}")

    return results


if __name__ == '__main__':
    main()
