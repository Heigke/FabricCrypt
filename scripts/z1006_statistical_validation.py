#!/usr/bin/env python3
"""
z1006: Statistical Validation
=============================

Rigorous statistical validation of FEEL V2 claims with:
- Multiple runs (5 seeds)
- Statistical significance tests (t-test, Wilcoxon)
- Effect size (Cohen's d)
- Confidence intervals

Claims to validate:
1. Energy prediction improves language modeling
2. Self-modeling improves hidden state prediction
3. Active inference improves coherence
4. Embodiment improves reasoning
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
N_STEPS = 300

class SimpleTransformer(nn.Module):
    def __init__(self, vocab_size, hidden=128, layers=4, body_dim=7, use_body=True, use_energy=False):
        super().__init__()
        self.use_body = use_body
        self.use_energy = use_energy
        self.embed = nn.Embedding(vocab_size, hidden)
        self.pos = nn.Embedding(256, hidden)
        if use_body:
            self.body_proj = nn.Linear(body_dim, hidden)
        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(hidden, 4, hidden*4, batch_first=True)
            for _ in range(layers)
        ])
        self.ln = nn.LayerNorm(hidden)
        self.head = nn.Linear(hidden, vocab_size)
        if use_energy:
            self.energy_head = nn.Sequential(nn.Linear(hidden, 32), nn.GELU(), nn.Linear(32, 2))

    def forward(self, x, body_state=None):
        B, T = x.shape
        h = self.embed(x) + self.pos(torch.arange(min(T, 256), device=x.device))
        if self.use_body and body_state is not None:
            h = h + self.body_proj(body_state).unsqueeze(1) * 0.1
        for block in self.blocks:
            h = block(h)
        h = self.ln(h)
        logits = self.head(h)
        energy_out = None
        if self.use_energy:
            energy_out = self.energy_head(h.mean(dim=1))
        return logits, h, energy_out

class BodyTracker:
    def __init__(self):
        self.ema = 50.0
        self.alpha = 0.1
    def update(self, sample):
        self.ema = self.alpha * sample.power_w + (1-self.alpha) * self.ema
        return torch.tensor([
            sample.power_w/300, sample.temp_edge_c/100, sample.gpu_busy_pct/100,
            self.ema/300, sample.temp_edge_c/100, 0.0, 0.0
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

def run_experiment(seed: int, use_body: bool, use_energy: bool, telemetry, data, char_to_idx, vocab_size, device) -> Dict:
    """Run single experiment with given config."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = SimpleTransformer(vocab_size, use_body=use_body, use_energy=use_energy).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    tracker = BodyTracker()

    batch_size, seq_len = 32, 64
    perplexities = []

    for step in range(N_STEPS):
        starts = torch.randint(0, len(data)-seq_len-1, (batch_size,))
        batch = torch.stack([data[s:s+seq_len] for s in starts])

        sample = telemetry.read_sample()
        body = tracker.update(sample).to(device).unsqueeze(0).expand(batch_size, -1)

        with EnergyMeter(telemetry) as meter:
            logits, hidden, energy_out = model(batch, body if use_body else None)
        actual_energy = meter.energy_j / batch_size

        task_loss = F.cross_entropy(logits[:, :-1].reshape(-1, vocab_size), batch[:, 1:].reshape(-1))

        loss = task_loss
        if use_energy and energy_out is not None:
            energy_mean = energy_out[:, 0]
            energy_logvar = energy_out[:, 1]
            energy_target = torch.full((batch_size,), actual_energy * 10, device=device)
            energy_var = energy_logvar.exp()
            energy_loss = 0.5 * (energy_logvar + (energy_target - energy_mean).pow(2) / (energy_var + 1e-6)).mean()
            loss = loss + 0.1 * energy_loss

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        perplexities.append(torch.exp(task_loss).item())

    final_ppl = np.mean(perplexities[-50:])
    return {'final_ppl': final_ppl, 'seed': seed}

def compute_statistics(group1: List[float], group2: List[float]) -> Dict:
    """Compute statistical comparison between two groups."""
    # T-test
    t_stat, t_pvalue = stats.ttest_ind(group1, group2)

    # Wilcoxon (non-parametric)
    try:
        w_stat, w_pvalue = stats.wilcoxon([a-b for a, b in zip(group1, group2)])
    except:
        w_stat, w_pvalue = 0, 1.0

    # Effect size (Cohen's d)
    pooled_std = np.sqrt((np.std(group1)**2 + np.std(group2)**2) / 2)
    cohens_d = (np.mean(group1) - np.mean(group2)) / pooled_std if pooled_std > 0 else 0

    # Confidence interval
    diff = np.array(group1) - np.array(group2)
    ci_low, ci_high = stats.t.interval(0.95, len(diff)-1, loc=np.mean(diff), scale=stats.sem(diff))

    return {
        't_stat': t_stat,
        't_pvalue': t_pvalue,
        'wilcoxon_pvalue': w_pvalue,
        'cohens_d': cohens_d,
        'ci_95': (ci_low, ci_high),
        'mean_diff': np.mean(diff),
        'significant_005': t_pvalue < 0.05,
        'significant_001': t_pvalue < 0.01,
    }

def main():
    print("="*70)
    print("z1006: STATISTICAL VALIDATION")
    print("="*70)
    print(f"\nRunning {N_SEEDS} seeds per condition, {N_STEPS} steps each")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    text = load_data()
    chars = sorted(set(text))
    char_to_idx = {c: i for i, c in enumerate(chars)}
    vocab_size = len(chars)
    data = torch.tensor([char_to_idx[c] for c in text], device=device)

    telemetry = SysfsHwmonTelemetry()

    # Run experiments
    results = {
        'baseline': [],      # No body, no energy
        'body_only': [],     # Body, no energy
        'energy_only': [],   # No body, energy
        'full': [],          # Body + energy
    }

    conditions = [
        ('baseline', False, False),
        ('body_only', True, False),
        ('energy_only', False, True),
        ('full', True, True),
    ]

    for cond_name, use_body, use_energy in conditions:
        print(f"\n{'='*50}")
        print(f"Condition: {cond_name}")
        print("="*50)

        for seed in range(N_SEEDS):
            print(f"  Seed {seed+1}/{N_SEEDS}...", end=" ", flush=True)
            result = run_experiment(seed, use_body, use_energy, telemetry, data, char_to_idx, vocab_size, device)
            results[cond_name].append(result['final_ppl'])
            print(f"PPL={result['final_ppl']:.2f}")

        print(f"  Mean PPL: {np.mean(results[cond_name]):.2f} ± {np.std(results[cond_name]):.2f}")

    # Statistical tests
    print("\n" + "="*70)
    print("STATISTICAL TESTS")
    print("="*70)

    comparisons = [
        ('Baseline vs Body', 'baseline', 'body_only'),
        ('Baseline vs Energy', 'baseline', 'energy_only'),
        ('Baseline vs Full', 'baseline', 'full'),
        ('Body vs Full', 'body_only', 'full'),
    ]

    test_results = {}
    for name, g1, g2 in comparisons:
        stats_result = compute_statistics(results[g1], results[g2])
        test_results[name] = stats_result

        print(f"\n{name}:")
        print(f"  {g1}: {np.mean(results[g1]):.2f} ± {np.std(results[g1]):.2f}")
        print(f"  {g2}: {np.mean(results[g2]):.2f} ± {np.std(results[g2]):.2f}")
        print(f"  Mean diff: {stats_result['mean_diff']:.2f}")
        print(f"  t-test p-value: {stats_result['t_pvalue']:.4f}")
        print(f"  Cohen's d: {stats_result['cohens_d']:.3f}")
        print(f"  95% CI: [{stats_result['ci_95'][0]:.2f}, {stats_result['ci_95'][1]:.2f}]")

        if stats_result['significant_001']:
            print(f"  ✅ HIGHLY SIGNIFICANT (p < 0.01)")
        elif stats_result['significant_005']:
            print(f"  ✅ SIGNIFICANT (p < 0.05)")
        else:
            print(f"  ⚠️ NOT SIGNIFICANT (p = {stats_result['t_pvalue']:.3f})")

    # Summary
    print("\n" + "="*70)
    print("VALIDATION SUMMARY")
    print("="*70)

    claims = []

    # Claim 1: Energy prediction helps
    if test_results['Baseline vs Energy']['mean_diff'] > 0 and test_results['Baseline vs Energy']['significant_005']:
        claims.append(('Energy prediction improves LM', True, test_results['Baseline vs Energy']['t_pvalue']))
    else:
        claims.append(('Energy prediction improves LM', False, test_results['Baseline vs Energy']['t_pvalue']))

    # Claim 2: Body awareness helps
    if test_results['Baseline vs Body']['mean_diff'] > 0 and test_results['Baseline vs Body']['significant_005']:
        claims.append(('Body awareness improves LM', True, test_results['Baseline vs Body']['t_pvalue']))
    else:
        claims.append(('Body awareness improves LM', False, test_results['Baseline vs Body']['t_pvalue']))

    # Claim 3: Combined is best
    if test_results['Baseline vs Full']['mean_diff'] > 0 and test_results['Baseline vs Full']['significant_005']:
        claims.append(('Full embodiment improves LM', True, test_results['Baseline vs Full']['t_pvalue']))
    else:
        claims.append(('Full embodiment improves LM', False, test_results['Baseline vs Full']['t_pvalue']))

    print("\n| Claim | Validated | p-value |")
    print("|-------|-----------|---------|")
    for claim, validated, pval in claims:
        status = "✅" if validated else "❌"
        print(f"| {claim:30} | {status} | {pval:.4f} |")

    validated_count = sum(1 for _, v, _ in claims if v)
    print(f"\n{validated_count}/{len(claims)} claims validated")

    # Save results
    save_results = {
        'experiment': 'z1006_statistical_validation',
        'timestamp': datetime.now().isoformat(),
        'n_seeds': N_SEEDS,
        'n_steps': N_STEPS,
        'raw_results': {k: [float(v) for v in vals] for k, vals in results.items()},
        'test_results': {k: {kk: float(vv) if isinstance(vv, (int, float, np.floating)) else str(vv) for kk, vv in v.items()} for k, v in test_results.items()},
        'claims': [{'claim': c, 'validated': v, 'pvalue': float(p)} for c, v, p in claims],
        'validated_count': validated_count,
        'total_claims': len(claims),
    }

    results_path = Path(__file__).parent.parent / "results" / "z1006_statistical_validation.json"
    with open(results_path, 'w') as f:
        json.dump(save_results, f, indent=2)

    print(f"\nSaved to {results_path}")

if __name__ == "__main__":
    main()
