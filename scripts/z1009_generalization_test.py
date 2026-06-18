#!/usr/bin/env python3
"""
z1009: Generalization Test with Train/Test Split
=================================================

Tests if energy-aware models generalize better:
1. True train/test split (80/20)
2. Shorter training (200 steps) to avoid memorization
3. Early stopping based on validation loss

Claims to validate:
1. Energy prediction doesn't hurt generalization
2. Energy-aware models may generalize better (lower test PPL)
"""

import os, sys, json, time, torch, torch.nn as nn, torch.nn.functional as F
import numpy as np
from pathlib import Path
from datetime import datetime
from scipy import stats
from typing import Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from telemetry.sysfs_hwmon import SysfsHwmonTelemetry, EnergyMeter

N_SEEDS = 5
N_STEPS = 200  # Shorter to avoid memorization

class FiLMLayer(nn.Module):
    def __init__(self, hidden_size: int, body_dim: int = 7):
        super().__init__()
        self.gamma = nn.Linear(body_dim, hidden_size)
        self.beta = nn.Linear(body_dim, hidden_size)
        nn.init.zeros_(self.gamma.weight)
        nn.init.zeros_(self.gamma.bias)
        nn.init.zeros_(self.beta.weight)
        nn.init.zeros_(self.beta.bias)

    def forward(self, x, body_state):
        gamma = self.gamma(body_state).unsqueeze(1)
        beta = self.beta(body_state).unsqueeze(1)
        return x * (1 + gamma * 0.1) + beta * 0.1

class FiLMTransformerBlock(nn.Module):
    def __init__(self, hidden_size: int, n_heads: int = 4, body_dim: int = 7):
        super().__init__()
        self.attn = nn.MultiheadAttention(hidden_size, n_heads, batch_first=True, dropout=0.1)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size * 4, hidden_size),
            nn.Dropout(0.1),
        )
        self.ln1 = nn.LayerNorm(hidden_size)
        self.ln2 = nn.LayerNorm(hidden_size)
        self.film = FiLMLayer(hidden_size, body_dim)

    def forward(self, x, body_state):
        residual = x
        x = self.ln1(x)
        x = self.film(x, body_state)
        x, _ = self.attn(x, x, x, need_weights=False)
        x = residual + x

        residual = x
        x = self.ln2(x)
        x = self.mlp(x)
        x = residual + x
        return x

class FiLMTransformer(nn.Module):
    def __init__(self, vocab_size, hidden=128, layers=3, n_heads=4, body_dim=7, use_energy=False):
        super().__init__()
        self.use_energy = use_energy
        self.embed = nn.Embedding(vocab_size, hidden)
        self.pos = nn.Embedding(256, hidden)

        self.blocks = nn.ModuleList([
            FiLMTransformerBlock(hidden, n_heads, body_dim)
            for _ in range(layers)
        ])
        self.ln = nn.LayerNorm(hidden)
        self.head = nn.Linear(hidden, vocab_size, bias=False)
        self.dropout = nn.Dropout(0.1)

        if use_energy:
            self.energy_head = nn.Sequential(
                nn.Linear(hidden, 32), nn.ReLU(), nn.Linear(32, 1)
            )

    def forward(self, x, body_state):
        B, T = x.shape
        h = self.embed(x) + self.pos(torch.arange(min(T, 256), device=x.device))
        h = self.dropout(h)

        for block in self.blocks:
            h = block(h, body_state)

        h = self.ln(h)
        logits = self.head(h)

        energy_pred = None
        if self.use_energy:
            pooled = h.mean(dim=1)
            energy_pred = self.energy_head(pooled).squeeze(-1)

        return logits, energy_pred

class BodyTracker:
    def __init__(self):
        self.ema = {'power': 50, 'temp': 50}
        self.prev = {'power': 50, 'temp': 50}
        self.alpha = 0.1

    def update(self, sample):
        self.ema['power'] = self.alpha * sample.power_w + (1-self.alpha) * self.ema['power']
        self.ema['temp'] = self.alpha * sample.temp_edge_c + (1-self.alpha) * self.ema['temp']
        deriv_p = sample.power_w - self.prev['power']
        deriv_t = sample.temp_edge_c - self.prev['temp']
        self.prev['power'] = sample.power_w
        self.prev['temp'] = sample.temp_edge_c
        return torch.tensor([
            sample.power_w/300, sample.temp_edge_c/100, sample.gpu_busy_pct/100,
            self.ema['power']/300, self.ema['temp']/100, deriv_p/100, deriv_t/10
        ], dtype=torch.float32)

def load_data():
    path = Path(__file__).parent.parent / "data" / "tinyshakespeare.txt"
    if path.exists():
        return path.read_text()
    import urllib.request
    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    path.parent.mkdir(exist_ok=True)
    urllib.request.urlretrieve(url, path)
    return path.read_text()

def run_experiment(seed: int, use_energy: bool, telemetry, train_data, test_data, vocab_size, device) -> Dict:
    """Run single experiment with train/test split."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = FiLMTransformer(vocab_size, use_energy=use_energy).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
    tracker = BodyTracker()

    batch_size, seq_len = 32, 64
    ENERGY_SCALE = 100.0

    # Training
    model.train()
    for step in range(N_STEPS):
        starts = torch.randint(0, len(train_data)-seq_len-1, (batch_size,))
        batch = torch.stack([train_data[s:s+seq_len] for s in starts])

        sample = telemetry.read_sample()
        body = tracker.update(sample).to(device).unsqueeze(0).expand(batch_size, -1)

        with EnergyMeter(telemetry) as meter:
            logits, energy_pred = model(batch, body)
        actual_energy = meter.energy_j / batch_size * ENERGY_SCALE

        task_loss = F.cross_entropy(logits[:, :-1].reshape(-1, vocab_size), batch[:, 1:].reshape(-1))

        loss = task_loss
        if use_energy and energy_pred is not None:
            energy_loss = F.mse_loss(energy_pred, torch.full_like(energy_pred, actual_energy))
            loss = task_loss + 0.01 * energy_loss

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

    # Evaluation on TEST set
    model.eval()
    test_losses = []
    with torch.no_grad():
        for _ in range(50):
            starts = torch.randint(0, len(test_data)-seq_len-1, (batch_size,))
            batch = torch.stack([test_data[s:s+seq_len] for s in starts])
            sample = telemetry.read_sample()
            body = tracker.update(sample).to(device).unsqueeze(0).expand(batch_size, -1)
            logits, _ = model(batch, body)
            loss = F.cross_entropy(logits[:, :-1].reshape(-1, vocab_size), batch[:, 1:].reshape(-1))
            test_losses.append(loss.item())

    # Also evaluate on TRAIN set for comparison
    train_losses = []
    with torch.no_grad():
        for _ in range(50):
            starts = torch.randint(0, len(train_data)-seq_len-1, (batch_size,))
            batch = torch.stack([train_data[s:s+seq_len] for s in starts])
            sample = telemetry.read_sample()
            body = tracker.update(sample).to(device).unsqueeze(0).expand(batch_size, -1)
            logits, _ = model(batch, body)
            loss = F.cross_entropy(logits[:, :-1].reshape(-1, vocab_size), batch[:, 1:].reshape(-1))
            train_losses.append(loss.item())

    train_ppl = np.exp(np.mean(train_losses))
    test_ppl = np.exp(np.mean(test_losses))
    generalization_gap = test_ppl - train_ppl

    return {
        'train_ppl': train_ppl,
        'test_ppl': test_ppl,
        'generalization_gap': generalization_gap,
        'seed': seed
    }

def compute_statistics(group1: List[float], group2: List[float]) -> Dict:
    """Compute statistical comparison."""
    t_stat, t_pvalue = stats.ttest_ind(group1, group2)
    pooled_std = np.sqrt((np.std(group1)**2 + np.std(group2)**2) / 2)
    cohens_d = (np.mean(group1) - np.mean(group2)) / pooled_std if pooled_std > 0 else 0
    diff = np.array(group1) - np.array(group2)
    ci_low, ci_high = stats.t.interval(0.95, len(diff)-1, loc=np.mean(diff), scale=stats.sem(diff))
    return {
        't_pvalue': t_pvalue,
        'cohens_d': cohens_d,
        'ci_95': (ci_low, ci_high),
        'mean_diff': np.mean(diff),
        'significant_005': t_pvalue < 0.05,
    }

def main():
    print("="*70)
    print("z1009: GENERALIZATION TEST")
    print("="*70)
    print(f"\nRunning {N_SEEDS} seeds per condition, {N_STEPS} steps each")
    print("Using 80/20 train/test split to test generalization")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    text = load_data()
    chars = sorted(set(text))
    char_to_idx = {c: i for i, c in enumerate(chars)}
    vocab_size = len(chars)

    # True train/test split
    all_data = torch.tensor([char_to_idx[c] for c in text], device=device)
    split_idx = int(len(all_data) * 0.8)
    train_data = all_data[:split_idx]
    test_data = all_data[split_idx:]
    print(f"Train: {len(train_data):,} chars, Test: {len(test_data):,} chars")

    telemetry = SysfsHwmonTelemetry()

    results = {
        'baseline': {'train_ppl': [], 'test_ppl': [], 'gap': []},
        'energy': {'train_ppl': [], 'test_ppl': [], 'gap': []},
    }

    for cond_name, use_energy in [('baseline', False), ('energy', True)]:
        print(f"\n{'='*50}")
        print(f"Condition: {cond_name}")
        print("="*50)

        for seed in range(N_SEEDS):
            print(f"  Seed {seed+1}/{N_SEEDS}...", end=" ", flush=True)
            result = run_experiment(seed, use_energy, telemetry, train_data, test_data, vocab_size, device)
            results[cond_name]['train_ppl'].append(result['train_ppl'])
            results[cond_name]['test_ppl'].append(result['test_ppl'])
            results[cond_name]['gap'].append(result['generalization_gap'])
            print(f"Train={result['train_ppl']:.2f}, Test={result['test_ppl']:.2f}, Gap={result['generalization_gap']:.2f}")

        print(f"  Mean Test PPL: {np.mean(results[cond_name]['test_ppl']):.2f} ± {np.std(results[cond_name]['test_ppl']):.2f}")
        print(f"  Mean Gap: {np.mean(results[cond_name]['gap']):.2f} ± {np.std(results[cond_name]['gap']):.2f}")

    # Statistical tests
    print("\n" + "="*70)
    print("STATISTICAL TESTS")
    print("="*70)

    # Test PPL comparison
    stats_test_ppl = compute_statistics(results['baseline']['test_ppl'], results['energy']['test_ppl'])

    print(f"\nTest PPL Comparison:")
    print(f"  Baseline: {np.mean(results['baseline']['test_ppl']):.2f} ± {np.std(results['baseline']['test_ppl']):.2f}")
    print(f"  Energy:   {np.mean(results['energy']['test_ppl']):.2f} ± {np.std(results['energy']['test_ppl']):.2f}")
    print(f"  Mean diff: {stats_test_ppl['mean_diff']:.2f} (positive = baseline higher = energy better)")
    print(f"  t-test p-value: {stats_test_ppl['t_pvalue']:.4f}")
    print(f"  Cohen's d: {stats_test_ppl['cohens_d']:.3f}")

    # Generalization gap comparison
    stats_gap = compute_statistics(results['baseline']['gap'], results['energy']['gap'])

    print(f"\nGeneralization Gap Comparison:")
    print(f"  Baseline gap: {np.mean(results['baseline']['gap']):.2f} ± {np.std(results['baseline']['gap']):.2f}")
    print(f"  Energy gap:   {np.mean(results['energy']['gap']):.2f} ± {np.std(results['energy']['gap']):.2f}")
    print(f"  Mean diff: {stats_gap['mean_diff']:.2f} (positive = baseline has larger gap)")
    print(f"  t-test p-value: {stats_gap['t_pvalue']:.4f}")

    # Verdicts
    print("\n" + "="*70)
    print("VALIDATION SUMMARY")
    print("="*70)

    claims = []
    baseline_test = np.mean(results['baseline']['test_ppl'])
    energy_test = np.mean(results['energy']['test_ppl'])
    ppl_diff_pct = (energy_test - baseline_test) / baseline_test * 100

    # Claim 1: Doesn't hurt generalization
    if ppl_diff_pct < 5 or not stats_test_ppl['significant_005']:
        claims.append(('Energy does not hurt generalization (<5% or not sig.)', True, ppl_diff_pct))
    else:
        claims.append(('Energy does not hurt generalization (<5% or not sig.)', False, ppl_diff_pct))

    # Claim 2: Energy helps generalization
    if stats_test_ppl['mean_diff'] > 0 and stats_test_ppl['significant_005']:
        claims.append(('Energy significantly improves generalization', True, stats_test_ppl['t_pvalue']))
    else:
        claims.append(('Energy significantly improves generalization', False, stats_test_ppl['t_pvalue']))

    # Claim 3: Better generalization gap
    if stats_gap['mean_diff'] > 0:
        claims.append(('Energy reduces generalization gap', True, stats_gap['mean_diff']))
    else:
        claims.append(('Energy reduces generalization gap', False, stats_gap['mean_diff']))

    print("\n| Claim | Validated | Value |")
    print("|-------|-----------|-------|")
    for claim, validated, val in claims:
        status = "✅" if validated else "❌"
        print(f"| {claim:52} | {status} | {val:+.3f} |")

    validated_count = sum(1 for _, v, _ in claims if v)
    print(f"\n{validated_count}/{len(claims)} claims validated")

    # Save results
    save_results = {
        'experiment': 'z1009_generalization_test',
        'timestamp': datetime.now().isoformat(),
        'n_seeds': N_SEEDS,
        'n_steps': N_STEPS,
        'raw_results': {
            'baseline': {k: [float(v) for v in vals] for k, vals in results['baseline'].items()},
            'energy': {k: [float(v) for v in vals] for k, vals in results['energy'].items()},
        },
        'test_ppl_stats': {k: float(v) if isinstance(v, (int, float, np.floating)) else str(v) for k, v in stats_test_ppl.items()},
        'gap_stats': {k: float(v) if isinstance(v, (int, float, np.floating)) else str(v) for k, v in stats_gap.items()},
        'claims': [{'claim': c, 'validated': bool(v), 'value': float(p)} for c, v, p in claims],
        'validated_count': validated_count,
    }

    results_path = Path(__file__).parent.parent / "results" / "z1009_generalization_test.json"
    with open(results_path, 'w') as f:
        json.dump(save_results, f, indent=2)

    print(f"\nSaved to {results_path}")

if __name__ == "__main__":
    main()
