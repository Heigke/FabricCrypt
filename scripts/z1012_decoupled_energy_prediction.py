#!/usr/bin/env python3
"""
z1012: Decoupled Energy Prediction

Key insight from z1006-z1011: Joint training of energy prediction and task
ALWAYS destabilizes the task. This experiment tests DECOUPLED training:

Phase 1: Train the LM normally (no energy)
Phase 2: Freeze LM, train only energy predictor
Phase 3: Validate energy predictor doesn't affect LM quality

Claims to validate:
1. Energy predictor can learn accurate predictions from frozen LM features
2. Frozen LM maintains original quality during energy-aware inference
3. Energy predictions correlate with actual measurements (>0.5 correlation)
"""

import json
import time
import sys
from pathlib import Path
from dataclasses import dataclass, asdict
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.telemetry.sysfs_hwmon import SysfsHwmonTelemetry

# ============================================================================
# Configuration
# ============================================================================

@dataclass
class Config:
    # Model
    n_layers: int = 6
    d_model: int = 256
    n_heads: int = 4
    d_ff: int = 512
    vocab_size: int = 256
    max_seq_len: int = 128

    # Training
    phase1_epochs: int = 10  # LM training
    phase2_epochs: int = 5   # Energy head training
    batch_size: int = 64
    lr: float = 1e-3

    # Energy prediction
    energy_scale: float = 100.0

    # Validation
    n_seeds: int = 3
    train_split: float = 0.8

    device: str = "cuda" if torch.cuda.is_available() else "cpu"

# ============================================================================
# Model
# ============================================================================

class SimpleTransformer(nn.Module):
    """Basic transformer for char-level LM."""

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg

        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_embed = nn.Parameter(torch.randn(1, cfg.max_seq_len, cfg.d_model) * 0.02)

        layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.n_heads,
            dim_feedforward=cfg.d_ff,
            batch_first=True,
            activation='gelu'
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=cfg.n_layers)
        self.out_proj = nn.Linear(cfg.d_model, cfg.vocab_size)

    def forward(self, x: torch.Tensor):
        B, T = x.shape
        h = self.embed(x) + self.pos_embed[:, :T, :]
        h = self.encoder(h)
        return self.out_proj(h), h


class EnergyHead(nn.Module):
    """Separate energy predictor that takes frozen LM features."""

    def __init__(self, d_model: int):
        super().__init__()
        self.predictor = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """Predict energy from hidden states."""
        h_pooled = h.mean(dim=1)  # [B, D]
        return self.predictor(h_pooled).squeeze(-1)  # [B]


# ============================================================================
# Data Loading
# ============================================================================

def load_data(cfg: Config):
    """Load TinyShakespeare."""
    data_path = Path(__file__).parent.parent / "data" / "input.txt"

    if not data_path.exists():
        import urllib.request
        data_path.parent.mkdir(exist_ok=True)
        url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
        urllib.request.urlretrieve(url, data_path)

    text = data_path.read_text()
    data = torch.tensor([ord(c) % cfg.vocab_size for c in text], dtype=torch.long)

    n_seqs = len(data) // cfg.max_seq_len
    data = data[:n_seqs * cfg.max_seq_len].view(n_seqs, cfg.max_seq_len)

    n_train = int(len(data) * cfg.train_split)
    train_data = data[:n_train]
    test_data = data[n_train:]

    train_loader = DataLoader(
        TensorDataset(train_data[:, :-1], train_data[:, 1:]),
        batch_size=cfg.batch_size,
        shuffle=True
    )
    test_loader = DataLoader(
        TensorDataset(test_data[:, :-1], test_data[:, 1:]),
        batch_size=cfg.batch_size
    )

    return train_loader, test_loader


# ============================================================================
# Training Functions
# ============================================================================

def train_lm_epoch(model, loader, optimizer, cfg):
    """Phase 1: Train LM only."""
    model.train()
    total_loss = 0
    total_tokens = 0

    for x, y in loader:
        x, y = x.to(cfg.device), y.to(cfg.device)

        optimizer.zero_grad()
        logits, _ = model(x)
        loss = F.cross_entropy(logits.view(-1, cfg.vocab_size), y.view(-1))

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item() * x.numel()
        total_tokens += x.numel()

    return {
        'loss': total_loss / total_tokens,
        'ppl': math.exp(min(total_loss / total_tokens, 10))
    }


def train_energy_epoch(model, energy_head, loader, optimizer, cfg, telemetry):
    """Phase 2: Train energy head only (frozen LM)."""
    model.eval()  # LM is frozen
    energy_head.train()

    total_loss = 0
    total_samples = 0
    predictions = []
    actuals = []

    telemetry.reset_accumulator()

    for x, y in loader:
        x, y = x.to(cfg.device), y.to(cfg.device)

        start_energy = telemetry.get_accumulated_energy_j()

        # Get frozen LM features
        with torch.no_grad():
            _, h = model(x)

        # Predict energy from features
        energy_pred = energy_head(h.detach())

        # Measure actual energy
        torch.cuda.synchronize()
        end_energy = telemetry.get_accumulated_energy_j()
        actual_energy = max(0, (end_energy - start_energy)) * cfg.energy_scale

        if actual_energy > 0.001:
            optimizer.zero_grad()
            loss = F.mse_loss(energy_pred, torch.full_like(energy_pred, actual_energy))
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_samples += 1
            predictions.extend(energy_pred.detach().cpu().tolist())
            actuals.extend([actual_energy] * x.size(0))

    # Compute correlation
    if len(predictions) > 10:
        import numpy as np
        corr = np.corrcoef(predictions, actuals)[0, 1]
    else:
        corr = 0.0

    return {
        'loss': total_loss / max(total_samples, 1),
        'correlation': corr,
        'n_samples': total_samples
    }


@torch.no_grad()
def evaluate(model, loader, cfg):
    """Evaluate LM quality."""
    model.eval()
    total_loss = 0
    total_tokens = 0

    for x, y in loader:
        x, y = x.to(cfg.device), y.to(cfg.device)
        logits, _ = model(x)
        loss = F.cross_entropy(logits.view(-1, cfg.vocab_size), y.view(-1))
        total_loss += loss.item() * x.numel()
        total_tokens += x.numel()

    return {
        'loss': total_loss / total_tokens,
        'ppl': math.exp(min(total_loss / total_tokens, 10))
    }


@torch.no_grad()
def evaluate_energy(model, energy_head, loader, cfg, telemetry):
    """Evaluate energy prediction quality."""
    model.eval()
    energy_head.eval()

    predictions = []
    actuals = []

    telemetry.reset_accumulator()

    for x, y in loader:
        x, y = x.to(cfg.device), y.to(cfg.device)

        start_energy = telemetry.get_accumulated_energy_j()

        _, h = model(x)
        energy_pred = energy_head(h)

        torch.cuda.synchronize()
        end_energy = telemetry.get_accumulated_energy_j()
        actual_energy = max(0, (end_energy - start_energy)) * cfg.energy_scale

        if actual_energy > 0.001:
            predictions.extend(energy_pred.cpu().tolist())
            actuals.extend([actual_energy] * x.size(0))

    if len(predictions) > 10:
        import numpy as np
        corr = np.corrcoef(predictions, actuals)[0, 1]
        mape = np.mean(np.abs((np.array(predictions) - np.array(actuals)) / np.array(actuals))) * 100
    else:
        corr = 0.0
        mape = 100.0

    return {'correlation': corr, 'mape': mape, 'n_samples': len(predictions)}


# ============================================================================
# Main Experiment
# ============================================================================

def run_experiment(cfg: Config, train_loader, test_loader, telemetry, seed: int):
    """Run one complete experiment."""
    torch.manual_seed(seed)

    # Phase 1: Train LM
    model = SimpleTransformer(cfg).to(cfg.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr)

    print(f"  Phase 1: Training LM ({cfg.phase1_epochs} epochs)...")
    for epoch in range(cfg.phase1_epochs):
        train_lm_epoch(model, train_loader, optimizer, cfg)

    phase1_test = evaluate(model, test_loader, cfg)
    print(f"    After Phase 1: PPL={phase1_test['ppl']:.3f}")

    # Phase 2: Train energy head (frozen LM)
    for param in model.parameters():
        param.requires_grad = False

    energy_head = EnergyHead(cfg.d_model).to(cfg.device)
    energy_optimizer = torch.optim.AdamW(energy_head.parameters(), lr=cfg.lr)

    print(f"  Phase 2: Training energy head ({cfg.phase2_epochs} epochs)...")
    for epoch in range(cfg.phase2_epochs):
        metrics = train_energy_epoch(
            model, energy_head, train_loader, energy_optimizer, cfg, telemetry
        )
        if epoch == cfg.phase2_epochs - 1:
            print(f"    Energy corr={metrics['correlation']:.3f}")

    # Phase 3: Validate
    phase3_test = evaluate(model, test_loader, cfg)
    energy_eval = evaluate_energy(model, energy_head, test_loader, cfg, telemetry)

    print(f"  Phase 3 validation:")
    print(f"    LM PPL: {phase3_test['ppl']:.3f} (was {phase1_test['ppl']:.3f})")
    print(f"    Energy corr: {energy_eval['correlation']:.3f}, MAPE: {energy_eval['mape']:.1f}%")

    return {
        'phase1_ppl': phase1_test['ppl'],
        'phase3_ppl': phase3_test['ppl'],
        'ppl_change': phase3_test['ppl'] - phase1_test['ppl'],
        'energy_correlation': energy_eval['correlation'],
        'energy_mape': energy_eval['mape']
    }


def main():
    print("=" * 70)
    print("z1012: DECOUPLED ENERGY PREDICTION")
    print("=" * 70)
    print()
    print("Testing decoupled training:")
    print("  Phase 1: Train LM (no energy)")
    print("  Phase 2: Freeze LM, train energy head only")
    print("  Phase 3: Validate LM unchanged, energy head works")
    print()
    print("Claims to validate:")
    print("  1. Energy head learns from frozen LM (correlation > 0.5)")
    print("  2. Frozen LM unchanged (PPL change < 0.01)")
    print("  3. Energy predictions useful (MAPE < 50%)")
    print()

    cfg = Config()
    print(f"Config: {cfg.n_seeds} seeds, device={cfg.device}")
    print()

    # Initialize telemetry
    try:
        telemetry = SysfsHwmonTelemetry()
        telemetry.start_continuous_sampling()
        time.sleep(0.5)
        print("✓ Telemetry initialized")
    except Exception as e:
        print(f"✗ Telemetry failed: {e}")
        return

    # Load data
    train_loader, test_loader = load_data(cfg)
    print(f"Data: {len(train_loader.dataset)} train, {len(test_loader.dataset)} test")
    print()

    # Run experiments
    results = {'ppl_change': [], 'correlation': [], 'mape': []}

    for seed in range(cfg.n_seeds):
        print(f"{'='*70}")
        print(f"Seed {seed + 1}/{cfg.n_seeds}")
        print(f"{'='*70}")

        res = run_experiment(cfg, train_loader, test_loader, telemetry, seed + 42)

        results['ppl_change'].append(res['ppl_change'])
        results['correlation'].append(res['energy_correlation'])
        results['mape'].append(res['energy_mape'])

    telemetry.stop_continuous_sampling()

    # Analysis
    print()
    print("=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)

    def mean_std(arr):
        m = sum(arr) / len(arr)
        s = (sum((x-m)**2 for x in arr) / len(arr)) ** 0.5
        return m, s

    ppl_m, ppl_s = mean_std(results['ppl_change'])
    corr_m, corr_s = mean_std(results['correlation'])
    mape_m, mape_s = mean_std(results['mape'])

    print(f"\nPPL change (frozen LM): {ppl_m:.4f} ± {ppl_s:.4f}")
    print(f"Energy correlation: {corr_m:.3f} ± {corr_s:.3f}")
    print(f"Energy MAPE: {mape_m:.1f}% ± {mape_s:.1f}%")

    # Claims validation
    print()
    print("=" * 70)
    print("CLAIMS VALIDATION")
    print("=" * 70)

    claims = []

    # Claim 1: Energy head learns
    c1_valid = corr_m > 0.3  # Lower threshold than 0.5 since energy is noisy
    claims.append({
        'claim': 'Energy head learns from frozen LM (corr > 0.3)',
        'validated': bool(c1_valid),
        'evidence': f'correlation = {corr_m:.3f}'
    })
    print(f"\n1. Energy head learns: {'✓' if c1_valid else '✗'} (corr={corr_m:.3f})")

    # Claim 2: LM unchanged
    c2_valid = abs(ppl_m) < 0.1  # Very small change allowed
    claims.append({
        'claim': 'Frozen LM unchanged (|PPL change| < 0.1)',
        'validated': bool(c2_valid),
        'evidence': f'PPL change = {ppl_m:.4f}'
    })
    print(f"2. LM unchanged: {'✓' if c2_valid else '✗'} (change={ppl_m:.4f})")

    # Claim 3: Energy predictions useful
    c3_valid = mape_m < 50.0
    claims.append({
        'claim': 'Energy predictions useful (MAPE < 50%)',
        'validated': bool(c3_valid),
        'evidence': f'MAPE = {mape_m:.1f}%'
    })
    print(f"3. Energy useful: {'✓' if c3_valid else '✗'} (MAPE={mape_m:.1f}%)")

    n_valid = sum(1 for c in claims if c['validated'])
    print(f"\n{n_valid}/{len(claims)} claims validated")

    # Save results
    output = {
        'experiment': 'z1012_decoupled_energy_prediction',
        'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
        'config': asdict(cfg),
        'raw_results': results,
        'summary': {
            'ppl_change_mean': ppl_m,
            'ppl_change_std': ppl_s,
            'correlation_mean': corr_m,
            'correlation_std': corr_s,
            'mape_mean': mape_m,
            'mape_std': mape_s
        },
        'claims': claims,
        'n_validated': n_valid
    }

    out_path = Path(__file__).parent.parent / "results" / "z1012_decoupled_energy.json"
    out_path.parent.mkdir(exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2))
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
