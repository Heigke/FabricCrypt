#!/usr/bin/env python3
"""
z2005: Diverse GWT Competition

Fixes z2004's specialist collapse (100% specialist 1) with:
1. Diversity regularization - penalize single specialist dominance
2. Capacity limits - each specialist has max utilization
3. Input-dependent routing - different inputs should use different specialists
4. Entropy target - aim for moderate entropy (not zero)

Target: ignition_ratio > 0.5 AND diversity > 0.5 (multiple specialists used)
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
from collections import Counter

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.telemetry.sysfs_hwmon import SysfsHwmonTelemetry


def gumbel_softmax_hard(logits, tau=1.0):
    """Gumbel-Softmax with hard selection via STE."""
    gumbels = -torch.empty_like(logits).exponential_().log()
    gumbels = (logits + gumbels) / tau
    y_soft = F.softmax(gumbels, dim=-1)
    index = y_soft.max(dim=-1, keepdim=True)[1]
    y_hard = torch.zeros_like(y_soft).scatter_(-1, index, 1.0)
    return y_hard - y_soft.detach() + y_soft


class Specialist(nn.Module):
    """Single specialist with domain specialization."""

    def __init__(self, vocab_size: int, hidden_dim: int, specialist_id: int, n_specialists: int):
        super().__init__()
        self.specialist_id = specialist_id

        # Each specialist has slightly different initialization
        # This breaks symmetry and encourages specialization
        self.embed = nn.Embedding(vocab_size, hidden_dim)
        nn.init.normal_(self.embed.weight, std=0.02 * (1 + specialist_id * 0.1))

        self.layers = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim) for _ in range(2)
        ])
        self.norm = nn.LayerNorm(hidden_dim)
        self.out = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        h = self.embed(x)
        for layer in self.layers:
            h = F.gelu(layer(h)) + h
        h = self.norm(h)
        return self.out(h), h


class DiverseGWTModel(nn.Module):
    """GWT with diversity-preserving competition."""

    def __init__(self, vocab_size: int, hidden_dim: int = 256,
                 n_specialists: int = 6, n_layers: int = 2):
        super().__init__()
        self.n_specialists = n_specialists
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size

        # Specialists with different initializations
        self.specialists = nn.ModuleList([
            Specialist(vocab_size, hidden_dim, i, n_specialists)
            for i in range(n_specialists)
        ])

        # Content-based router - routes based on input patterns
        self.router_embed = nn.Embedding(vocab_size, hidden_dim)
        self.router = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, n_specialists)
        )

        # Telemetry modulation
        self.telemetry_proj = nn.Linear(4, n_specialists)

        # Usage tracking for capacity limits (not learnable)
        self.register_buffer('usage_counts', torch.zeros(n_specialists))
        self.capacity_limit = 0.4  # Each specialist max 40% of samples

    def forward(self, x, telemetry=None, temperature=1.0):
        batch_size, seq_len = x.shape

        # Content-based routing - use mean of embeddings
        route_embed = self.router_embed(x).mean(dim=1)  # [B, H]
        route_logits = self.router(route_embed)  # [B, N_spec]

        # Telemetry modulation
        if telemetry is not None:
            tel_bias = self.telemetry_proj(telemetry)
            route_logits = route_logits + 0.5 * tel_bias

        # Gumbel-Softmax selection
        weights = gumbel_softmax_hard(route_logits, tau=temperature)

        # Run all specialists
        specialist_outputs = []
        specialist_hiddens = []
        for specialist in self.specialists:
            out, h = specialist(x)
            specialist_outputs.append(out)
            specialist_hiddens.append(h)

        outputs = torch.stack(specialist_outputs, dim=1)  # [B, N_spec, seq, vocab]
        hiddens = torch.stack(specialist_hiddens, dim=1)  # [B, N_spec, seq, H]

        # Weighted combination
        w = weights.unsqueeze(-1).unsqueeze(-1)
        combined_output = (outputs * w).sum(dim=1)

        return combined_output, weights, route_logits

    def compute_diversity_loss(self, weights):
        """
        Encourage diverse specialist usage.

        Instead of minimizing entropy (which causes collapse),
        we aim for a TARGET entropy that represents healthy competition.
        """
        # Average weights across batch
        avg_weights = weights.mean(dim=0)  # [N_spec]

        # Target: uniform distribution (each specialist used equally on average)
        target_weights = torch.ones_like(avg_weights) / self.n_specialists

        # KL divergence from uniform - penalize deviation from uniform usage
        diversity_loss = F.kl_div(
            (avg_weights + 1e-8).log(),
            target_weights,
            reduction='sum'
        )

        return diversity_loss

    def compute_capacity_loss(self, weights):
        """Penalize over-utilization of any single specialist."""
        # Per-specialist usage in this batch
        usage = weights.mean(dim=0)  # [N_spec]

        # Penalize exceeding capacity limit
        over_capacity = F.relu(usage - self.capacity_limit)
        capacity_loss = over_capacity.sum()

        return capacity_loss

    def get_metrics(self, weights):
        """Compute GWT metrics including diversity."""
        with torch.no_grad():
            max_weights = weights.max(dim=-1)[0]
            ignition_ratio = (max_weights > 0.7).float().mean().item()

            # Winner distribution
            winners = weights.argmax(dim=-1)
            winner_counts = Counter(winners.cpu().numpy())
            total = len(winners)
            winner_dist = {str(i): winner_counts.get(i, 0) / total
                          for i in range(self.n_specialists)}

            # Diversity: count specialists used > 5% of the time
            usage_counts = list(winner_dist.values())
            specialists_used = sum(1 for u in usage_counts if u > 0.05)
            diversity = specialists_used / self.n_specialists

            # Entropy of usage distribution
            usage_tensor = torch.tensor(usage_counts)
            entropy = -(usage_tensor * (usage_tensor + 1e-8).log()).sum().item()

            return {
                'ignition_ratio': ignition_ratio,
                'max_weight': max_weights.mean().item(),
                'winner_distribution': winner_dist,
                'diversity': diversity,
                'specialists_used': specialists_used,
                'usage_entropy': entropy
            }


def load_data(path: str, seq_len: int = 64):
    """Load text data."""
    with open(path, 'r') as f:
        text = f.read()

    chars = sorted(list(set(text)))
    char_to_idx = {ch: i for i, ch in enumerate(chars)}

    data = torch.tensor([char_to_idx[c] for c in text], dtype=torch.long)
    n_sequences = len(data) - seq_len - 1
    x = torch.stack([data[i:i+seq_len] for i in range(0, n_sequences, seq_len)])
    y = torch.stack([data[i+1:i+seq_len+1] for i in range(0, n_sequences, seq_len)])

    return x, y, len(chars)


def main():
    print("=" * 70)
    print("z2005: DIVERSE GWT COMPETITION")
    print("Fixing specialist collapse with diversity regularization")
    print("=" * 70)

    timestamp = datetime.now().isoformat()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Hardware
    telemetry = SysfsHwmonTelemetry()
    sample = telemetry.read_sample()
    print(f"[Hardware] GPU: {sample.temp_edge_c:.1f}C, {sample.power_w:.1f}W")

    # Data
    data_path = Path(__file__).parent.parent / 'data' / 'tiny_shakespeare.txt'
    if not data_path.exists():
        data_path.parent.mkdir(exist_ok=True)
        sample_text = "To be, or not to be, that is the question.\n" * 5000
        with open(data_path, 'w') as f:
            f.write(sample_text)

    x_all, y_all, vocab_size = load_data(str(data_path))
    print(f"[Data] {len(x_all)} sequences, vocab {vocab_size}")

    split = int(0.9 * len(x_all))
    x_train, y_train = x_all[:split].to(device), y_all[:split].to(device)
    x_val, y_val = x_all[split:].to(device), y_all[split:].to(device)

    # Model
    model = DiverseGWTModel(
        vocab_size=vocab_size,
        hidden_dim=256,
        n_specialists=6
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"[Model] {n_params:,} parameters, {model.n_specialists} specialists")

    # Training config
    n_epochs = 25
    batch_size = 64
    lr = 3e-4

    # Loss weights
    diversity_weight = 0.1  # Encourage diverse usage
    capacity_weight = 0.5   # Penalize over-utilization

    # Temperature schedule
    temp_start = 2.0
    temp_end = 0.1

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    history = []

    print(f"\n{'='*60}")
    print("TRAINING: Diversity + Capacity regularization")
    print(f"{'='*60}")

    for epoch in range(n_epochs):
        model.train()

        # Temperature schedule
        progress = epoch / (n_epochs - 1)
        temperature = temp_start * (temp_end / temp_start) ** progress

        # Shuffle
        perm = torch.randperm(len(x_train))
        x_train, y_train = x_train[perm], y_train[perm]

        n_batches = len(x_train) // batch_size
        epoch_task_loss = 0.0
        epoch_div_loss = 0.0
        epoch_cap_loss = 0.0
        all_weights = []

        for i in range(n_batches):
            x_batch = x_train[i*batch_size:(i+1)*batch_size]
            y_batch = y_train[i*batch_size:(i+1)*batch_size]

            # Telemetry
            sample = telemetry.read_sample()
            tel = torch.tensor([
                sample.temp_edge_c / 100.0,
                sample.power_w / 100.0,
                sample.freq_sclk_mhz / 3000.0,
                sample.gpu_busy_pct / 100.0
            ], device=device).unsqueeze(0).expand(batch_size, -1)

            optimizer.zero_grad()

            logits, weights, _ = model(x_batch, tel, temperature=temperature)
            all_weights.append(weights.detach())

            # Task loss
            task_loss = F.cross_entropy(logits.view(-1, vocab_size), y_batch.view(-1))

            # Diversity loss - encourage uniform usage
            div_loss = model.compute_diversity_loss(weights)

            # Capacity loss - prevent single specialist dominance
            cap_loss = model.compute_capacity_loss(weights)

            # Total loss
            loss = task_loss + diversity_weight * div_loss + capacity_weight * cap_loss

            if torch.isnan(loss) or torch.isinf(loss):
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_task_loss += task_loss.item()
            epoch_div_loss += div_loss.item()
            epoch_cap_loss += cap_loss.item()

            if i % 100 == 0:
                metrics = model.get_metrics(weights)
                print(f"  Batch {i}/{n_batches}: task={task_loss.item():.4f} "
                      f"div={div_loss.item():.4f} ign={metrics['ignition_ratio']:.3f} "
                      f"used={metrics['specialists_used']}")

        # Epoch metrics
        epoch_task_loss /= n_batches
        epoch_div_loss /= n_batches
        epoch_cap_loss /= n_batches

        all_weights = torch.cat(all_weights, dim=0)
        train_metrics = model.get_metrics(all_weights)

        # Validation
        model.eval()
        with torch.no_grad():
            val_logits, val_weights, _ = model(x_val, None, temperature=0.1)
            val_loss = F.cross_entropy(val_logits.view(-1, vocab_size), y_val.view(-1))
            val_metrics = model.get_metrics(val_weights)

        print(f"\n[Epoch {epoch+1}/{n_epochs}] temp={temperature:.4f}")
        print(f"  Train: task={epoch_task_loss:.4f} div={epoch_div_loss:.4f} "
              f"ignition={train_metrics['ignition_ratio']:.3f}")
        print(f"  Val:   task={val_loss.item():.4f} ignition={val_metrics['ignition_ratio']:.3f} "
              f"diversity={val_metrics['diversity']:.3f}")
        print(f"  Specialists used: {val_metrics['specialists_used']}/6")
        print(f"  Usage: {val_metrics['winner_distribution']}")

        history.append({
            'epoch': epoch + 1,
            'temperature': temperature,
            'train_task_loss': epoch_task_loss,
            'train_div_loss': epoch_div_loss,
            'val_loss': val_loss.item(),
            'val_ignition': val_metrics['ignition_ratio'],
            'val_diversity': val_metrics['diversity'],
            'val_specialists_used': val_metrics['specialists_used'],
            'winner_distribution': val_metrics['winner_distribution']
        })

    # Final evaluation
    print(f"\n{'='*60}")
    print("FINAL EVALUATION")
    print(f"{'='*60}")

    model.eval()
    with torch.no_grad():
        final_logits, final_weights, _ = model(x_val, None, temperature=0.1)
        final_metrics = model.get_metrics(final_weights)

    # Success criteria
    ignition_passed = final_metrics['ignition_ratio'] >= 0.5
    diversity_passed = final_metrics['diversity'] >= 0.5  # At least 3 specialists used

    print(f"\n[Results]")
    print(f"  Ignition ratio: {final_metrics['ignition_ratio']:.4f} "
          f"(threshold: 0.5) {'PASS' if ignition_passed else 'FAIL'}")
    print(f"  Diversity: {final_metrics['diversity']:.4f} "
          f"(threshold: 0.5) {'PASS' if diversity_passed else 'FAIL'}")
    print(f"  Specialists used: {final_metrics['specialists_used']}/6")
    print(f"  Usage distribution: {final_metrics['winner_distribution']}")

    # Compare to z2004
    print(f"\n[Comparison to z2004]")
    print(f"  z2004 diversity: 0.17 (1 specialist) -> z2005: {final_metrics['diversity']:.3f}")
    print(f"  z2004 ignition: 1.0 (collapsed) -> z2005: {final_metrics['ignition_ratio']:.3f}")

    verdict = "SUCCESS" if (ignition_passed and diversity_passed) else "PARTIAL" if (ignition_passed or diversity_passed) else "FAIL"

    # Save results
    results = {
        'experiment': 'z2005_diverse_gwt_competition',
        'timestamp': timestamp,
        'device': str(device),
        'model': {
            'n_specialists': model.n_specialists,
            'hidden_dim': model.hidden_dim,
            'n_params': n_params
        },
        'training': {
            'n_epochs': n_epochs,
            'diversity_weight': diversity_weight,
            'capacity_weight': capacity_weight,
            'temp_start': temp_start,
            'temp_end': temp_end
        },
        'final_metrics': {
            'ignition_ratio': final_metrics['ignition_ratio'],
            'diversity': final_metrics['diversity'],
            'specialists_used': final_metrics['specialists_used'],
            'max_weight': final_metrics['max_weight'],
            'usage_entropy': final_metrics['usage_entropy'],
            'winner_distribution': final_metrics['winner_distribution']
        },
        'success_criteria': {
            'ignition_threshold': 0.5,
            'ignition_measured': final_metrics['ignition_ratio'],
            'ignition_passed': ignition_passed,
            'diversity_threshold': 0.5,
            'diversity_measured': final_metrics['diversity'],
            'diversity_passed': diversity_passed
        },
        'comparison_to_z2004': {
            'z2004_diversity': 0.17,
            'z2004_specialists_used': 1,
            'z2005_diversity': final_metrics['diversity'],
            'z2005_specialists_used': final_metrics['specialists_used']
        },
        'verdict': verdict,
        'epoch_history': history
    }

    results_path = Path(__file__).parent.parent / 'results' / 'z2005_diverse_gwt_competition.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n[Saved] {results_path}")

    print(f"\n{'='*60}")
    print(f"VERDICT: {verdict}")
    print(f"{'='*60}")

    return results


if __name__ == '__main__':
    main()
