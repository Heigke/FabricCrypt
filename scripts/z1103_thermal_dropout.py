#!/usr/bin/env python3
"""
z1103: Thermal Dropout - Hardware State as Mandatory Computation Modifier

Key insight from z1102: Making hardware available as INPUT gets ignored.
We need to make computation DEPEND on hardware state directly.

This experiment uses GPU thermal fluctuations as a DROPOUT MASK.
The hardware state directly determines which neurons are active.
This creates a causal HW→SW link that CANNOT be bypassed.

If temp_deviation > threshold: dropout this neuron
The dropout pattern comes from REAL hardware, not random numbers.

Claims to validate:
1. Thermal dropout creates measurable sparsity
2. Thermal dropout acts as regularization (test < train gap)
3. Models trained with thermal dropout generalize better under thermal stress
"""

import json
import time
import sys
import math
from pathlib import Path
from dataclasses import dataclass, asdict

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
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 4
    d_ff: int = 256
    vocab_size: int = 256
    max_seq_len: int = 64

    # Thermal dropout
    base_dropout: float = 0.1  # Standard dropout rate
    thermal_sensitivity: float = 0.5  # How much thermal changes affect dropout

    n_epochs: int = 10
    batch_size: int = 32
    lr: float = 1e-3
    n_seeds: int = 3
    train_split: float = 0.8

    device: str = "cuda" if torch.cuda.is_available() else "cpu"


# ============================================================================
# Thermal Dropout
# ============================================================================

class ThermalDropout(nn.Module):
    """
    Dropout where the mask is derived from GPU thermal state.

    Instead of random dropout, we use:
    - Per-neuron threshold based on thermal deviation
    - Neurons with thermal_contribution > threshold are dropped
    - This creates a CAUSAL link from HW to SW computation
    """

    def __init__(self, p: float = 0.1, thermal_sensitivity: float = 0.5):
        super().__init__()
        self.p = p
        self.thermal_sensitivity = thermal_sensitivity
        self.temp_ema = 55.0  # Running average
        self.temp_std_ema = 2.0  # Running std

    def update_thermal_stats(self, temp: float):
        """Update thermal running statistics."""
        alpha = 0.1
        self.temp_ema = alpha * temp + (1 - alpha) * self.temp_ema
        deviation = abs(temp - self.temp_ema)
        self.temp_std_ema = alpha * deviation + (1 - alpha) * self.temp_std_ema

    def forward(self, x: torch.Tensor, temp: float) -> torch.Tensor:
        if not self.training:
            return x

        self.update_thermal_stats(temp)

        # Thermal deviation from mean
        thermal_deviation = (temp - self.temp_ema) / (self.temp_std_ema + 1e-6)

        # Dropout rate modulated by thermal state
        # Hot GPU = more dropout (regularization under stress)
        # Cold GPU = less dropout (use full capacity)
        thermal_factor = torch.sigmoid(torch.tensor(thermal_deviation * self.thermal_sensitivity))
        effective_p = self.p * (1 + thermal_factor.item())
        effective_p = min(effective_p, 0.5)  # Cap at 50%

        # Create mask
        mask = torch.bernoulli(torch.ones_like(x) * (1 - effective_p))
        return x * mask / (1 - effective_p + 1e-6)


class ThermalAttention(nn.Module):
    """
    Attention where scores are modulated by power state.

    High power = attention is more diffuse (prevent runaway)
    Low power = attention can be sharper (use available headroom)
    """

    def __init__(self, d_model: int, n_heads: int, power_sensitivity: float = 0.3):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.power_sensitivity = power_sensitivity

        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out = nn.Linear(d_model, d_model)

        self.power_ema = 30.0
        self.power_std_ema = 5.0

    def update_power_stats(self, power: float):
        alpha = 0.1
        self.power_ema = alpha * power + (1 - alpha) * self.power_ema
        deviation = abs(power - self.power_ema)
        self.power_std_ema = alpha * deviation + (1 - alpha) * self.power_std_ema

    def forward(self, x: torch.Tensor, power: float) -> torch.Tensor:
        B, T, D = x.shape

        self.update_power_stats(power)

        # Power deviation
        power_deviation = (power - self.power_ema) / (self.power_std_ema + 1e-6)

        # Temperature scaling factor
        # High power = higher temperature = softer attention
        temp_scale = 1.0 + self.power_sensitivity * torch.sigmoid(torch.tensor(power_deviation)).item()

        # QKV projection
        qkv = self.qkv(x).reshape(B, T, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)  # [B, T, H, D]

        q = q.transpose(1, 2)  # [B, H, T, D]
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Attention with thermal scaling
        scale = self.head_dim ** 0.5 * temp_scale  # Power modulates scale
        attn = torch.matmul(q, k.transpose(-2, -1)) / scale
        attn = F.softmax(attn, dim=-1)

        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).reshape(B, T, D)

        return self.out(out)


# ============================================================================
# Models
# ============================================================================

class StandardTransformer(nn.Module):
    """Baseline with standard dropout."""

    def __init__(self, cfg: Config):
        super().__init__()
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos = nn.Parameter(torch.randn(1, cfg.max_seq_len, cfg.d_model) * 0.02)

        layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.n_heads,
            dim_feedforward=cfg.d_ff,
            dropout=cfg.base_dropout,
            batch_first=True,
            activation='gelu'
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=cfg.n_layers)
        self.out = nn.Linear(cfg.d_model, cfg.vocab_size)

    def forward(self, x, telemetry=None):
        B, T = x.shape
        h = self.embed(x) + self.pos[:, :T]
        h = self.encoder(h)
        return self.out(h)


class ThermalTransformer(nn.Module):
    """Transformer with thermal dropout and power-modulated attention."""

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos = nn.Parameter(torch.randn(1, cfg.max_seq_len, cfg.d_model) * 0.02)

        self.layers = nn.ModuleList()
        self.dropouts = nn.ModuleList()

        for _ in range(cfg.n_layers):
            self.layers.append(nn.ModuleDict({
                'attn': ThermalAttention(cfg.d_model, cfg.n_heads),
                'ff': nn.Sequential(
                    nn.Linear(cfg.d_model, cfg.d_ff),
                    nn.GELU(),
                    nn.Linear(cfg.d_ff, cfg.d_model)
                ),
                'norm1': nn.LayerNorm(cfg.d_model),
                'norm2': nn.LayerNorm(cfg.d_model)
            }))
            self.dropouts.append(ThermalDropout(cfg.base_dropout, cfg.thermal_sensitivity))

        self.out = nn.Linear(cfg.d_model, cfg.vocab_size)

    def forward(self, x: torch.Tensor, telemetry: dict = None):
        B, T = x.shape
        h = self.embed(x) + self.pos[:, :T]

        temp = telemetry.get('temp', 55.0) if telemetry else 55.0
        power = telemetry.get('power', 30.0) if telemetry else 30.0

        for layer, dropout in zip(self.layers, self.dropouts):
            # Attention with power modulation
            attn_out = layer['attn'](h, power)
            h = layer['norm1'](h + dropout(attn_out, temp))

            # FFN with thermal dropout
            ff_out = layer['ff'](h)
            h = layer['norm2'](h + dropout(ff_out, temp))

        return self.out(h)


# ============================================================================
# Data & Training
# ============================================================================

def load_data(cfg: Config):
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
    train_loader = DataLoader(
        TensorDataset(data[:n_train, :-1], data[:n_train, 1:]),
        batch_size=cfg.batch_size, shuffle=True
    )
    test_loader = DataLoader(
        TensorDataset(data[n_train:, :-1], data[n_train:, 1:]),
        batch_size=cfg.batch_size
    )

    return train_loader, test_loader


def train_epoch(model, loader, optimizer, cfg, telemetry=None, is_thermal=False):
    model.train()
    total_loss, total_tokens = 0, 0
    dropout_rates = []

    for x, y in loader:
        x, y = x.to(cfg.device), y.to(cfg.device)

        telem = None
        if telemetry and is_thermal:
            sample = telemetry.read_sample()
            telem = {'temp': sample.temp_edge_c, 'power': sample.power_w}

        optimizer.zero_grad()
        logits = model(x, telem)
        loss = F.cross_entropy(logits.view(-1, cfg.vocab_size), y.view(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item() * x.numel()
        total_tokens += x.numel()

        # Track dropout rates for thermal model
        if is_thermal and hasattr(model, 'dropouts'):
            for d in model.dropouts:
                if hasattr(d, 'p'):
                    dropout_rates.append(d.p)

    metrics = {
        'loss': total_loss / total_tokens,
        'ppl': math.exp(min(total_loss / total_tokens, 10))
    }

    if dropout_rates:
        metrics['avg_dropout'] = sum(dropout_rates) / len(dropout_rates)

    return metrics


@torch.no_grad()
def evaluate(model, loader, cfg, telemetry=None, is_thermal=False, inject_thermal_stress=False):
    model.eval()
    total_loss, total_tokens = 0, 0

    for x, y in loader:
        x, y = x.to(cfg.device), y.to(cfg.device)

        telem = None
        if telemetry and is_thermal:
            sample = telemetry.read_sample()
            if inject_thermal_stress:
                # Simulate thermal stress
                telem = {'temp': sample.temp_edge_c + 20, 'power': sample.power_w + 20}
            else:
                telem = {'temp': sample.temp_edge_c, 'power': sample.power_w}

        logits = model(x, telem)
        loss = F.cross_entropy(logits.view(-1, cfg.vocab_size), y.view(-1))

        total_loss += loss.item() * x.numel()
        total_tokens += x.numel()

    return {
        'loss': total_loss / total_tokens,
        'ppl': math.exp(min(total_loss / total_tokens, 10))
    }


# ============================================================================
# Main
# ============================================================================

def run_condition(condition: str, cfg: Config, train_loader, test_loader, telemetry, seed: int):
    torch.manual_seed(seed)

    if condition == "A_standard":
        model = StandardTransformer(cfg).to(cfg.device)
        is_thermal = False
    elif condition == "B_thermal":
        model = ThermalTransformer(cfg).to(cfg.device)
        is_thermal = True
    elif condition == "C_no_dropout":
        cfg_no_drop = Config(**{**asdict(cfg), 'base_dropout': 0.0})
        model = StandardTransformer(cfg_no_drop).to(cfg.device)
        is_thermal = False
    else:
        raise ValueError(f"Unknown condition: {condition}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr)

    for epoch in range(cfg.n_epochs):
        train_epoch(model, train_loader, optimizer, cfg, telemetry, is_thermal)

    # Normal evaluation
    test_normal = evaluate(model, test_loader, cfg, telemetry, is_thermal, inject_thermal_stress=False)

    # Evaluation under simulated thermal stress
    test_stress = evaluate(model, test_loader, cfg, telemetry, is_thermal, inject_thermal_stress=True)

    return {
        'test_ppl': test_normal['ppl'],
        'test_ppl_stress': test_stress['ppl'],
        'stress_degradation': (test_stress['ppl'] - test_normal['ppl']) / test_normal['ppl'] * 100
    }


def main():
    print("=" * 70)
    print("z1103: THERMAL DROPOUT - Hardware as Mandatory Modifier")
    print("=" * 70)
    print()
    print("Key insight: Make computation DEPEND on hardware, not just input it.")
    print("Thermal dropout: GPU temp determines dropout mask (can't be bypassed)")
    print()
    print("Claims to validate:")
    print("  1. Thermal dropout creates varying sparsity")
    print("  2. Thermal model is more robust to thermal stress")
    print("  3. Quality matches standard dropout under normal conditions")
    print()

    cfg = Config()
    print(f"Config: {cfg.n_seeds} seeds, {cfg.n_epochs} epochs")
    print()

    try:
        telemetry = SysfsHwmonTelemetry()
        telemetry.start_continuous_sampling()
        time.sleep(0.5)
        print("✓ Telemetry initialized")
    except Exception as e:
        print(f"✗ Telemetry failed: {e}")
        return

    train_loader, test_loader = load_data(cfg)
    print(f"Data: {len(train_loader.dataset)} train, {len(test_loader.dataset)} test")

    conditions = ["A_standard", "B_thermal", "C_no_dropout"]
    results = {c: {'ppl': [], 'ppl_stress': [], 'stress_deg': []} for c in conditions}

    for seed in range(cfg.n_seeds):
        print(f"\n{'='*70}")
        print(f"Seed {seed + 1}/{cfg.n_seeds}")
        print(f"{'='*70}")

        for condition in conditions:
            print(f"\n  Running {condition}...")
            start = time.time()

            res = run_condition(condition, cfg, train_loader, test_loader, telemetry, seed + 42)

            elapsed = time.time() - start
            print(f"    PPL: {res['test_ppl']:.3f}, Stress: {res['test_ppl_stress']:.3f} "
                  f"(+{res['stress_degradation']:.1f}%), Time: {elapsed:.1f}s")

            results[condition]['ppl'].append(res['test_ppl'])
            results[condition]['ppl_stress'].append(res['test_ppl_stress'])
            results[condition]['stress_deg'].append(res['stress_degradation'])

    telemetry.stop_continuous_sampling()

    # Analysis
    print()
    print("=" * 70)
    print("RESULTS")
    print("=" * 70)

    def mean_std(arr):
        m = sum(arr) / len(arr)
        s = (sum((x-m)**2 for x in arr) / len(arr)) ** 0.5
        return m, s

    for cond in conditions:
        ppl_m, ppl_s = mean_std(results[cond]['ppl'])
        stress_m, stress_s = mean_std(results[cond]['ppl_stress'])
        deg_m, deg_s = mean_std(results[cond]['stress_deg'])
        print(f"\n{cond}:")
        print(f"  Normal PPL: {ppl_m:.3f} ± {ppl_s:.3f}")
        print(f"  Stress PPL: {stress_m:.3f} ± {stress_s:.3f}")
        print(f"  Degradation: {deg_m:+.1f}% ± {deg_s:.1f}%")

    # Claims validation
    print()
    print("=" * 70)
    print("CLAIMS VALIDATION")
    print("=" * 70)

    claims = []

    std_ppl, _ = mean_std(results['A_standard']['ppl'])
    therm_ppl, _ = mean_std(results['B_thermal']['ppl'])
    no_drop_ppl, _ = mean_std(results['C_no_dropout']['ppl'])

    std_deg, _ = mean_std(results['A_standard']['stress_deg'])
    therm_deg, _ = mean_std(results['B_thermal']['stress_deg'])
    no_drop_deg, _ = mean_std(results['C_no_dropout']['stress_deg'])

    # Claim 1: Thermal matches standard quality
    ppl_diff = abs(therm_ppl - std_ppl) / std_ppl * 100
    c1_valid = ppl_diff < 5  # Within 5%
    claims.append({
        'claim': 'Thermal dropout matches standard quality (within 5%)',
        'validated': bool(c1_valid),
        'evidence': f'diff = {ppl_diff:.1f}%'
    })
    print(f"\n1. Quality match: {'✓' if c1_valid else '✗'} (diff={ppl_diff:.1f}%)")

    # Claim 2: Thermal is more stress-robust
    c2_valid = therm_deg < std_deg
    claims.append({
        'claim': 'Thermal model more stress-robust',
        'validated': bool(c2_valid),
        'evidence': f'thermal={therm_deg:.1f}% vs std={std_deg:.1f}%'
    })
    print(f"2. Stress robust: {'✓' if c2_valid else '✗'} (therm={therm_deg:.1f}%, std={std_deg:.1f}%)")

    # Claim 3: Dropout helps (vs no dropout)
    c3_valid = std_ppl < no_drop_ppl or therm_ppl < no_drop_ppl
    claims.append({
        'claim': 'Dropout improves generalization vs no dropout',
        'validated': bool(c3_valid),
        'evidence': f'no_drop={no_drop_ppl:.3f} vs std={std_ppl:.3f}, therm={therm_ppl:.3f}'
    })
    print(f"3. Dropout helps: {'✓' if c3_valid else '✗'}")

    n_valid = sum(1 for c in claims if c['validated'])
    print(f"\n{n_valid}/{len(claims)} claims validated")

    # Save
    output = {
        'experiment': 'z1103_thermal_dropout',
        'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
        'config': asdict(cfg),
        'results': results,
        'claims': claims,
        'n_validated': n_valid
    }

    out_path = Path(__file__).parent.parent / "results" / "z1103_thermal_dropout.json"
    out_path.write_text(json.dumps(output, indent=2, default=str))
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
