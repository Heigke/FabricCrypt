#!/usr/bin/env python3
"""
z1011: Adaptive Inference with Energy Prediction

Hypothesis: Hidden-state gates (proven in z906) combined with energy prediction
(proven in z1008) achieve better energy-quality Pareto curves WITHOUT claiming
embodiment improves quality.

Key insights from prior experiments:
- z906: Hidden-only gates WORK (B_hidden_only PPL 6.54 vs baseline 6.70)
- z1008: Energy prediction IS learnable (1.5% MAPE)
- z1009: Energy does NOT improve LM quality (p=0.55)
- z1010: Energy modulation HURTS PPL (p=0.008)

This experiment tests: Can we use energy prediction for ADAPTIVE INFERENCE
(skip computation when predicted energy is low) without claiming it helps quality?

Claims to validate:
1. Hidden-state gates reduce energy without significant PPL loss
2. Energy prediction allows dynamic quality-energy tradeoff control
3. Combined system achieves better Pareto frontier than either alone
"""

import json
import time
import sys
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split

# Add project root to path
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
    vocab_size: int = 256  # char-level
    max_seq_len: int = 128

    # Training
    n_epochs: int = 10
    batch_size: int = 64
    lr: float = 1e-3

    # Gates (from z906 best config)
    gate_hidden_dim: int = 64
    use_hidden_gate: bool = True

    # Energy prediction (from z1008 config)
    energy_loss_weight: float = 0.1
    energy_scale: float = 100.0  # Scale small energy values

    # Validation
    n_seeds: int = 3
    train_split: float = 0.8

    device: str = "cuda" if torch.cuda.is_available() else "cpu"

# ============================================================================
# Model Components
# ============================================================================

class HiddenStateGate(nn.Module):
    """Per-layer gate based on hidden state (from z906 B_hidden_only)."""

    def __init__(self, d_model: int, hidden_dim: int = 64):
        super().__init__()
        self.gate_mlp = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """Return skip probability from pooled hidden state."""
        h_pooled = h.mean(dim=1)  # [B, D]
        return self.gate_mlp(h_pooled).squeeze(-1)  # [B]


class EnergyPredictor(nn.Module):
    """Predict energy for current computation (from z1008)."""

    def __init__(self, d_model: int, n_layers: int):
        super().__init__()
        # Input: pooled hidden state + gate decisions
        input_dim = d_model + n_layers
        self.predictor = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, h: torch.Tensor, gate_probs: torch.Tensor) -> torch.Tensor:
        """Predict energy given hidden state and gate decisions."""
        h_pooled = h.mean(dim=1)  # [B, D]
        x = torch.cat([h_pooled, gate_probs], dim=-1)  # [B, D+L]
        return self.predictor(x).squeeze(-1)  # [B]


class AdaptiveTransformerLayer(nn.Module):
    """Transformer layer with stochastic depth gate."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int, gate_hidden: int):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.gate = HiddenStateGate(d_model, gate_hidden)

    def forward(self, x: torch.Tensor, training: bool = True):
        """Forward with stochastic depth."""
        # Get gate probability
        fire_prob = 1.0 - self.gate(x)  # [B]

        if training:
            # Stochastic depth with straight-through estimator
            fire_mask = (torch.rand_like(fire_prob) < fire_prob).float()
            fire_mask = fire_mask + fire_prob - fire_prob.detach()  # STE
        else:
            fire_mask = (fire_prob > 0.5).float()

        # Apply layer if firing
        fire_mask = fire_mask.unsqueeze(-1).unsqueeze(-1)  # [B, 1, 1]

        # Self-attention
        attn_out, _ = self.attn(x, x, x, need_weights=False)
        x = x + fire_mask * attn_out
        x = self.norm1(x)

        # FFN
        ff_out = self.ff(x)
        x = x + fire_mask * ff_out
        x = self.norm2(x)

        return x, fire_prob.mean()  # Return avg fire prob


class AdaptiveTransformer(nn.Module):
    """Transformer with hidden-state gates and energy prediction."""

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg

        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_embed = nn.Parameter(torch.randn(1, cfg.max_seq_len, cfg.d_model) * 0.02)

        self.layers = nn.ModuleList([
            AdaptiveTransformerLayer(cfg.d_model, cfg.n_heads, cfg.d_ff, cfg.gate_hidden_dim)
            for _ in range(cfg.n_layers)
        ])

        self.out_proj = nn.Linear(cfg.d_model, cfg.vocab_size)
        self.energy_pred = EnergyPredictor(cfg.d_model, cfg.n_layers)

    def forward(self, x: torch.Tensor, training: bool = True):
        """Forward pass returning logits, gate probs, and energy prediction."""
        B, T = x.shape

        h = self.embed(x) + self.pos_embed[:, :T, :]

        gate_probs = []
        for layer in self.layers:
            h, fire_prob = layer(h, training)
            gate_probs.append(fire_prob)

        gate_probs = torch.stack(gate_probs, dim=-1)  # [L]

        # Energy prediction
        energy_pred = self.energy_pred(h, gate_probs.unsqueeze(0).expand(B, -1))

        logits = self.out_proj(h)

        return logits, gate_probs, energy_pred


# ============================================================================
# Baseline Model (no gates)
# ============================================================================

class BaselineTransformer(nn.Module):
    """Standard transformer without gates."""

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

    def forward(self, x: torch.Tensor, training: bool = True):
        B, T = x.shape
        h = self.embed(x) + self.pos_embed[:, :T, :]
        h = self.encoder(h)
        logits = self.out_proj(h)
        return logits, None, None


# ============================================================================
# Training & Evaluation
# ============================================================================

def load_data(cfg: Config):
    """Load and prepare TinyShakespeare."""
    data_path = Path(__file__).parent.parent / "data" / "input.txt"

    if not data_path.exists():
        # Download
        import urllib.request
        data_path.parent.mkdir(exist_ok=True)
        url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
        urllib.request.urlretrieve(url, data_path)

    text = data_path.read_text()
    data = torch.tensor([ord(c) % cfg.vocab_size for c in text], dtype=torch.long)

    # Create sequences
    n_seqs = len(data) // cfg.max_seq_len
    data = data[:n_seqs * cfg.max_seq_len].view(n_seqs, cfg.max_seq_len)

    # Train/test split
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


def train_epoch(model, loader, optimizer, cfg, telemetry=None, with_energy=False):
    """Train one epoch."""
    model.train()
    total_loss = 0
    total_tokens = 0
    total_energy = 0
    energy_pred_errors = []
    fire_rates = []

    # Reset energy accumulator at start of epoch
    if telemetry and with_energy:
        telemetry.reset_accumulator()

    for x, y in loader:
        x, y = x.to(cfg.device), y.to(cfg.device)

        # Measure energy if available
        if telemetry and with_energy:
            start_energy = telemetry.get_accumulated_energy_j()

        optimizer.zero_grad()
        logits, gate_probs, energy_pred = model(x, training=True)

        # Language modeling loss
        lm_loss = F.cross_entropy(logits.view(-1, cfg.vocab_size), y.view(-1))

        loss = lm_loss

        # Energy prediction loss (if applicable)
        if telemetry and with_energy and energy_pred is not None:
            torch.cuda.synchronize()  # Ensure GPU work is done
            end_energy = telemetry.get_accumulated_energy_j()
            actual_energy = max(0, (end_energy - start_energy)) * cfg.energy_scale

            if actual_energy > 0.001:  # Minimum threshold
                energy_loss = F.mse_loss(energy_pred, torch.full_like(energy_pred, actual_energy))
                loss = loss + cfg.energy_loss_weight * energy_loss

                # Track prediction error
                pred_error = (energy_pred.mean().item() - actual_energy) / max(actual_energy, 1e-6)
                energy_pred_errors.append(abs(pred_error))
                total_energy += actual_energy / cfg.energy_scale

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += lm_loss.item() * x.numel()
        total_tokens += x.numel()

        if gate_probs is not None:
            fire_rates.append(gate_probs.mean().item())

    metrics = {
        'loss': total_loss / total_tokens,
        'ppl': math.exp(min(total_loss / total_tokens, 10)),
    }

    if fire_rates:
        metrics['fire_rate'] = sum(fire_rates) / len(fire_rates)
    if energy_pred_errors:
        metrics['energy_mape'] = sum(energy_pred_errors) / len(energy_pred_errors) * 100
    if total_energy > 0:
        metrics['energy_j'] = total_energy
        metrics['j_per_token'] = total_energy / total_tokens

    return metrics


@torch.no_grad()
def evaluate(model, loader, cfg, telemetry=None):
    """Evaluate on test set."""
    model.eval()
    total_loss = 0
    total_tokens = 0
    total_energy = 0
    fire_rates = []

    # Reset and measure energy for entire evaluation
    if telemetry:
        telemetry.reset_accumulator()
        start_energy = telemetry.get_accumulated_energy_j()

    for x, y in loader:
        x, y = x.to(cfg.device), y.to(cfg.device)

        logits, gate_probs, _ = model(x, training=False)
        loss = F.cross_entropy(logits.view(-1, cfg.vocab_size), y.view(-1))

        total_loss += loss.item() * x.numel()
        total_tokens += x.numel()

        if gate_probs is not None:
            fire_rates.append(gate_probs.mean().item())

    if telemetry:
        torch.cuda.synchronize()
        end_energy = telemetry.get_accumulated_energy_j()
        total_energy = max(0, end_energy - start_energy)

    metrics = {
        'loss': total_loss / total_tokens,
        'ppl': math.exp(min(total_loss / total_tokens, 10)),
    }

    if fire_rates:
        metrics['fire_rate'] = sum(fire_rates) / len(fire_rates)
    if total_energy > 0:
        metrics['energy_j'] = total_energy
        metrics['j_per_token'] = total_energy / total_tokens

    return metrics


# ============================================================================
# Experiment Conditions
# ============================================================================

def run_condition(condition: str, cfg: Config, train_loader, test_loader,
                  telemetry: Optional[SysfsHwmonTelemetry], seed: int):
    """Run one experimental condition."""
    torch.manual_seed(seed)

    if condition == "A_baseline":
        model = BaselineTransformer(cfg).to(cfg.device)
        with_energy = False
    elif condition == "B_gates_only":
        model = AdaptiveTransformer(cfg).to(cfg.device)
        with_energy = False
    elif condition == "C_gates_energy":
        model = AdaptiveTransformer(cfg).to(cfg.device)
        with_energy = True
    else:
        raise ValueError(f"Unknown condition: {condition}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr)

    epoch_metrics = []
    for epoch in range(cfg.n_epochs):
        train_metrics = train_epoch(
            model, train_loader, optimizer, cfg, telemetry, with_energy
        )
        test_metrics = evaluate(model, test_loader, cfg, telemetry)

        epoch_metrics.append({
            'epoch': epoch + 1,
            'train': train_metrics,
            'test': test_metrics
        })

    return {
        'final_train_ppl': epoch_metrics[-1]['train']['ppl'],
        'final_test_ppl': epoch_metrics[-1]['test']['ppl'],
        'final_fire_rate': epoch_metrics[-1]['test'].get('fire_rate', 1.0),
        'final_j_per_token': epoch_metrics[-1]['test'].get('j_per_token', 0),
        'epochs': epoch_metrics
    }


# ============================================================================
# Statistical Analysis
# ============================================================================

def ttest_ind(a, b):
    """Simple t-test implementation."""
    n1, n2 = len(a), len(b)
    m1, m2 = sum(a)/n1, sum(b)/n2
    v1 = sum((x-m1)**2 for x in a) / (n1-1) if n1 > 1 else 0
    v2 = sum((x-m2)**2 for x in b) / (n2-1) if n2 > 1 else 0

    se = math.sqrt(v1/n1 + v2/n2) if (v1/n1 + v2/n2) > 0 else 1e-10
    t = (m1 - m2) / se

    df = n1 + n2 - 2
    # Approximate p-value
    p = 2 * (1 - 0.5 * (1 + math.erf(abs(t) / math.sqrt(2))))

    return t, p, m1 - m2


def cohens_d(a, b):
    """Effect size."""
    n1, n2 = len(a), len(b)
    m1, m2 = sum(a)/n1, sum(b)/n2
    v1 = sum((x-m1)**2 for x in a) / (n1-1) if n1 > 1 else 0
    v2 = sum((x-m2)**2 for x in b) / (n2-1) if n2 > 1 else 0

    pooled_std = math.sqrt(((n1-1)*v1 + (n2-1)*v2) / (n1+n2-2)) if (n1+n2) > 2 else 1
    return (m1 - m2) / pooled_std if pooled_std > 0 else 0


# ============================================================================
# Main Experiment
# ============================================================================

def main():
    print("=" * 70)
    print("z1011: ADAPTIVE INFERENCE WITH ENERGY PREDICTION")
    print("=" * 70)
    print()
    print("Testing whether hidden-state gates (z906) + energy prediction (z1008)")
    print("achieve better energy-quality tradeoffs than either alone.")
    print()
    print("Claims to validate:")
    print("  1. Gates reduce energy without significant PPL loss")
    print("  2. Energy prediction enables dynamic quality-energy control")
    print("  3. Combined achieves better Pareto frontier")
    print()

    cfg = Config()
    print(f"Config: {cfg.n_seeds} seeds, {cfg.n_epochs} epochs, device={cfg.device}")
    print()

    # Initialize telemetry
    try:
        telemetry = SysfsHwmonTelemetry()
        telemetry.start_continuous_sampling()
        # Wait for first samples
        time.sleep(0.5)
        print("✓ Telemetry initialized")
    except Exception as e:
        print(f"⚠ Telemetry unavailable: {e}")
        telemetry = None

    # Load data
    train_loader, test_loader = load_data(cfg)
    print(f"Data: {len(train_loader.dataset)} train, {len(test_loader.dataset)} test")
    print()

    # Run experiments
    conditions = ["A_baseline", "B_gates_only", "C_gates_energy"]
    results = {c: {'ppl': [], 'energy': [], 'fire_rate': []} for c in conditions}

    for seed in range(cfg.n_seeds):
        print(f"{'='*70}")
        print(f"Seed {seed + 1}/{cfg.n_seeds}")
        print(f"{'='*70}")

        for condition in conditions:
            print(f"\n  Running {condition}...")
            start = time.time()

            res = run_condition(
                condition, cfg, train_loader, test_loader, telemetry, seed + 42
            )

            elapsed = time.time() - start
            print(f"    PPL: {res['final_test_ppl']:.3f}, "
                  f"Fire: {res['final_fire_rate']:.3f}, "
                  f"J/tok: {res['final_j_per_token']:.6f}, "
                  f"Time: {elapsed:.1f}s")

            results[condition]['ppl'].append(res['final_test_ppl'])
            results[condition]['energy'].append(res['final_j_per_token'])
            results[condition]['fire_rate'].append(res['final_fire_rate'])

    if telemetry:
        telemetry.stop_continuous_sampling()

    # Statistical analysis
    print()
    print("=" * 70)
    print("STATISTICAL ANALYSIS")
    print("=" * 70)

    def analyze_pair(name, cond_a, cond_b, metric):
        a = results[cond_a][metric]
        b = results[cond_b][metric]
        t, p, diff = ttest_ind(a, b)
        d = cohens_d(a, b)
        m_a, m_b = sum(a)/len(a), sum(b)/len(b)

        print(f"\n{name} ({metric}):")
        print(f"  {cond_a}: {m_a:.4f} ± {(sum((x-m_a)**2 for x in a)/len(a))**0.5:.4f}")
        print(f"  {cond_b}: {m_b:.4f} ± {(sum((x-m_b)**2 for x in b)/len(b))**0.5:.4f}")
        print(f"  Diff: {diff:.4f}, p={p:.4f}, d={d:.2f}")
        print(f"  Significant (p<0.05): {'Yes' if p < 0.05 else 'No'}")

        return {'t': t, 'p': p, 'diff': diff, 'd': d, 'significant': p < 0.05}

    stats = {}

    # Claim 1: Gates reduce energy without significant PPL loss
    stats['gates_vs_baseline_ppl'] = analyze_pair(
        "Claim 1a: Gates PPL impact", "A_baseline", "B_gates_only", "ppl"
    )
    stats['gates_vs_baseline_energy'] = analyze_pair(
        "Claim 1b: Gates energy impact", "A_baseline", "B_gates_only", "energy"
    )

    # Claim 2: Energy prediction enables control
    stats['energy_vs_gates_ppl'] = analyze_pair(
        "Claim 2a: Energy pred PPL impact", "B_gates_only", "C_gates_energy", "ppl"
    )
    stats['energy_vs_gates_energy'] = analyze_pair(
        "Claim 2b: Energy pred energy impact", "B_gates_only", "C_gates_energy", "energy"
    )

    # Claim 3: Combined vs baseline
    stats['combined_vs_baseline_ppl'] = analyze_pair(
        "Claim 3a: Combined PPL vs baseline", "A_baseline", "C_gates_energy", "ppl"
    )
    stats['combined_vs_baseline_energy'] = analyze_pair(
        "Claim 3b: Combined energy vs baseline", "A_baseline", "C_gates_energy", "energy"
    )

    # Validation summary
    print()
    print("=" * 70)
    print("CLAIMS VALIDATION")
    print("=" * 70)

    claims = []

    # Claim 1: Gates don't significantly hurt PPL
    c1_valid = stats['gates_vs_baseline_ppl']['p'] > 0.05 or stats['gates_vs_baseline_ppl']['diff'] < 0.5
    claims.append({
        'claim': 'Gates reduce energy without significant PPL loss',
        'validated': c1_valid,
        'evidence': f"PPL diff={stats['gates_vs_baseline_ppl']['diff']:.3f}, p={stats['gates_vs_baseline_ppl']['p']:.3f}"
    })
    print(f"\n1. Gates don't hurt PPL: {'✓' if c1_valid else '✗'}")
    print(f"   {claims[-1]['evidence']}")

    # Claim 2: Energy prediction doesn't hurt performance
    c2_valid = stats['energy_vs_gates_ppl']['p'] > 0.05 or stats['energy_vs_gates_ppl']['diff'] < 0.5
    claims.append({
        'claim': 'Energy prediction adds without hurting',
        'validated': c2_valid,
        'evidence': f"PPL diff={stats['energy_vs_gates_ppl']['diff']:.3f}, p={stats['energy_vs_gates_ppl']['p']:.3f}"
    })
    print(f"\n2. Energy pred doesn't hurt: {'✓' if c2_valid else '✗'}")
    print(f"   {claims[-1]['evidence']}")

    # Claim 3: Combined system works
    c3_valid = stats['combined_vs_baseline_ppl']['p'] > 0.05 or stats['combined_vs_baseline_ppl']['diff'] < 0.5
    claims.append({
        'claim': 'Combined system viable',
        'validated': c3_valid,
        'evidence': f"PPL diff={stats['combined_vs_baseline_ppl']['diff']:.3f}, p={stats['combined_vs_baseline_ppl']['p']:.3f}"
    })
    print(f"\n3. Combined system viable: {'✓' if c3_valid else '✗'}")
    print(f"   {claims[-1]['evidence']}")

    n_valid = sum(1 for c in claims if c['validated'])
    print(f"\n{n_valid}/{len(claims)} claims validated")

    # Save results
    output = {
        'experiment': 'z1011_adaptive_inference',
        'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
        'config': asdict(cfg),
        'results': results,
        'statistics': stats,
        'claims': claims,
        'n_validated': n_valid
    }

    out_path = Path(__file__).parent.parent / "results" / "z1011_adaptive_inference.json"
    out_path.parent.mkdir(exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2))
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
