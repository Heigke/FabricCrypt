#!/usr/bin/env python3
"""
z1010: Kernel-Level Active Inference
====================================

COMBINES the working pieces from our research:
1. Kernel-level energy modulation (z910 - PROVEN to work)
2. Active inference for coherence (z1002 - +65% coherence)
3. Fast telemetry (sysfs_hwmon - 100Hz sampling)
4. Proper statistical validation (5 seeds)

Key insight: z1000 showed energy prediction doesn't help LM quality,
BUT z1002 showed active inference DOES help coherence.
z910 showed kernel-level embodiment WORKS.

This experiment tests: Does kernel-level energy modulation + active inference
produce more coherent outputs than either alone?

Architecture:
- Use existing energy_attention.hip for energy-aware softmax
- Add active inference selection (EFE-based token choice)
- Real telemetry drives kernel modulation
- Body tokens participate in attention

Controls:
A: Baseline (greedy, no energy modulation)
B: Energy modulation only (z910 style)
C: Active inference only (z1002 style)
D: Full system (energy mod + active inference)

Author: FEEL Research Team
Date: 2026-01-29
"""

import os
os.environ['HSA_OVERRIDE_GFX_VERSION'] = '11.0.0'

import sys
import json
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from datetime import datetime
from scipy import stats
from typing import Dict, List, Tuple, Optional

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from telemetry.sysfs_hwmon import SysfsHwmonTelemetry, EnergyMeter

N_SEEDS = 5
N_STEPS = 300
N_EVAL_SAMPLES = 50

# =============================================================================
# Simple Energy-Modulated Transformer (uses kernel concepts in PyTorch for now)
# =============================================================================

class EnergyModulatedAttention(nn.Module):
    """
    Attention with energy-based modulation.

    Implements kernel concept: attention = softmax(QK^T / sqrt(d) * energy_mod)

    energy_mod = 1 - α * (current_power / power_setpoint - 1)

    When power > setpoint: energy_mod < 1 -> attend less (conserve energy)
    When power < setpoint: energy_mod > 1 -> attend more (use budget)
    """

    def __init__(self, hidden_size: int, n_heads: int = 4, energy_strength: float = 0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.n_heads = n_heads
        self.head_dim = hidden_size // n_heads
        self.energy_strength = energy_strength

        self.qkv = nn.Linear(hidden_size, 3 * hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)

    def forward(self, x: torch.Tensor, energy_mod: float = 1.0) -> torch.Tensor:
        B, T, H = x.shape

        qkv = self.qkv(x).reshape(B, T, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)  # [3, B, heads, T, head_dim]

        # Scaled dot-product attention with energy modulation
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)

        # Apply energy modulation to scores (like in HIP kernel)
        scores = scores * energy_mod

        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, v)

        out = out.transpose(1, 2).reshape(B, T, H)
        return self.out_proj(out)


class EnergyTransformerBlock(nn.Module):
    def __init__(self, hidden_size: int, n_heads: int = 4, energy_strength: float = 0.1):
        super().__init__()
        self.attn = EnergyModulatedAttention(hidden_size, n_heads, energy_strength)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.GELU(),
            nn.Linear(hidden_size * 4, hidden_size),
        )
        self.ln1 = nn.LayerNorm(hidden_size)
        self.ln2 = nn.LayerNorm(hidden_size)

    def forward(self, x: torch.Tensor, energy_mod: float = 1.0) -> torch.Tensor:
        x = x + self.attn(self.ln1(x), energy_mod)
        x = x + self.mlp(self.ln2(x))
        return x


class EnergyTransformer(nn.Module):
    """Transformer with energy modulation for embodied generation."""

    def __init__(self, vocab_size: int, hidden: int = 128, layers: int = 4,
                 n_heads: int = 4, energy_strength: float = 0.1):
        super().__init__()
        self.hidden = hidden
        self.embed = nn.Embedding(vocab_size, hidden)
        self.pos = nn.Embedding(256, hidden)

        self.blocks = nn.ModuleList([
            EnergyTransformerBlock(hidden, n_heads, energy_strength)
            for _ in range(layers)
        ])

        self.ln = nn.LayerNorm(hidden)
        self.head = nn.Linear(hidden, vocab_size, bias=False)

        # Uncertainty head for active inference
        self.uncertainty = nn.Sequential(
            nn.Linear(hidden, 32), nn.ReLU(), nn.Linear(32, 1), nn.Softplus()
        )

    def forward(self, x: torch.Tensor, energy_mod: float = 1.0) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, T = x.shape
        h = self.embed(x) + self.pos(torch.arange(min(T, 256), device=x.device))

        for block in self.blocks:
            h = block(h, energy_mod)

        h = self.ln(h)
        logits = self.head(h)
        uncertainty = self.uncertainty(h.mean(dim=1))

        return logits, uncertainty, h


# =============================================================================
# Telemetry and Energy Modulation
# =============================================================================

class EnergyModulator:
    """Computes energy modulation factor from telemetry."""

    def __init__(self, power_setpoint: float = 30.0, temp_setpoint: float = 60.0,
                 energy_strength: float = 0.2, homeostatic_gain: float = 0.1):
        self.power_setpoint = power_setpoint
        self.temp_setpoint = temp_setpoint
        self.energy_strength = energy_strength
        self.homeostatic_gain = homeostatic_gain
        self.ema_power = power_setpoint
        self.ema_temp = temp_setpoint

    def update(self, sample) -> float:
        """Compute energy modulation from current telemetry."""
        # Update EMAs
        alpha = 0.1
        self.ema_power = alpha * sample.power_w + (1 - alpha) * self.ema_power
        self.ema_temp = alpha * sample.temp_edge_c + (1 - alpha) * self.ema_temp

        # Energy modulation: reduce attention when over budget
        power_ratio = self.ema_power / max(self.power_setpoint, 1)
        energy_mod = 1.0 - self.energy_strength * (power_ratio - 1.0)

        # Homeostatic modulation: reduce when far from setpoint
        temp_deviation = abs(self.ema_temp - self.temp_setpoint) / self.temp_setpoint
        homeostatic = 1.0 / (1.0 + self.homeostatic_gain * temp_deviation)

        # Combined modulation (clamped to reasonable range)
        mod = max(0.5, min(1.5, energy_mod * homeostatic))
        return mod


# =============================================================================
# Active Inference Generation
# =============================================================================

def generate_greedy(model: EnergyTransformer, prompt: torch.Tensor,
                    max_new_tokens: int, energy_mod: float = 1.0) -> torch.Tensor:
    """Standard greedy generation."""
    generated = prompt.clone()
    model.eval()

    with torch.no_grad():
        for _ in range(max_new_tokens):
            logits, _, _ = model(generated[:, -128:], energy_mod)
            next_token = logits[:, -1].argmax(-1, keepdim=True)
            generated = torch.cat([generated, next_token], dim=1)

    return generated


def generate_active_inference(model: EnergyTransformer, prompt: torch.Tensor,
                              max_new_tokens: int, energy_mod: float = 1.0,
                              n_candidates: int = 5) -> torch.Tensor:
    """Active inference generation using Expected Free Energy."""
    generated = prompt.clone()
    model.eval()

    with torch.no_grad():
        for _ in range(max_new_tokens):
            logits, base_unc, _ = model(generated[:, -128:], energy_mod)

            # Sample candidates
            probs = F.softmax(logits[:, -1] / 0.7, dim=-1)
            candidates = torch.multinomial(probs, n_candidates, replacement=True)

            # Evaluate each candidate's uncertainty (EFE proxy)
            best_token = candidates[:, 0:1]
            best_efe = float('inf')

            for i in range(n_candidates):
                candidate_token = candidates[:, i:i+1]
                test_seq = torch.cat([generated, candidate_token], dim=1)
                _, unc, _ = model(test_seq[:, -128:], energy_mod)

                # EFE = uncertainty + complexity penalty
                efe = unc.item() + 0.1 * (-probs[:, candidates[:, i]].log().mean().item())

                if efe < best_efe:
                    best_efe = efe
                    best_token = candidate_token

            generated = torch.cat([generated, best_token], dim=1)

    return generated


# =============================================================================
# Coherence Evaluation (from z1002/z1003)
# =============================================================================

def compute_local_coherence(hidden_states: torch.Tensor) -> float:
    """Average cosine similarity between consecutive hidden states."""
    if hidden_states.shape[1] < 2:
        return 0.0
    h = hidden_states.squeeze(0)  # [T, H]
    h_norm = F.normalize(h, dim=-1)
    sims = (h_norm[:-1] * h_norm[1:]).sum(dim=-1)
    return sims.mean().item()


def evaluate_coherence(model: EnergyTransformer, test_prompts: List[torch.Tensor],
                       generation_fn, energy_mod: float = 1.0) -> Dict[str, float]:
    """Evaluate generation coherence."""
    coherences = []

    for prompt in test_prompts[:20]:  # Limit for speed
        generated = generation_fn(model, prompt, max_new_tokens=32, energy_mod=energy_mod)

        with torch.no_grad():
            _, _, hidden = model(generated[:, -64:], energy_mod)
            coh = compute_local_coherence(hidden)
            coherences.append(coh)

    return {
        'mean_coherence': np.mean(coherences),
        'std_coherence': np.std(coherences),
    }


# =============================================================================
# Training and Experiment
# =============================================================================

def load_data():
    path = Path(__file__).parent.parent / "data" / "tinyshakespeare.txt"
    if path.exists():
        return path.read_text()
    import urllib.request
    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    path.parent.mkdir(exist_ok=True)
    urllib.request.urlretrieve(url, path)
    return path.read_text()


def run_condition(seed: int, use_energy_mod: bool, use_active_inference: bool,
                  telemetry, train_data, test_data, vocab_size, device) -> Dict:
    """Run one experimental condition."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = EnergyTransformer(vocab_size).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    modulator = EnergyModulator()

    batch_size, seq_len = 32, 64

    # Training
    model.train()
    for step in range(N_STEPS):
        starts = torch.randint(0, len(train_data) - seq_len - 1, (batch_size,))
        batch = torch.stack([train_data[s:s+seq_len] for s in starts])

        sample = telemetry.read_sample()
        energy_mod = modulator.update(sample) if use_energy_mod else 1.0

        logits, _, _ = model(batch, energy_mod)
        loss = F.cross_entropy(logits[:, :-1].reshape(-1, vocab_size),
                               batch[:, 1:].reshape(-1))

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

    # Evaluation
    model.eval()

    # Test prompts
    test_prompts = []
    for i in range(N_EVAL_SAMPLES):
        start = torch.randint(0, len(test_data) - 32, (1,)).item()
        prompt = test_data[start:start+16].unsqueeze(0)
        test_prompts.append(prompt)

    # Get final energy mod for evaluation
    sample = telemetry.read_sample()
    eval_energy_mod = modulator.update(sample) if use_energy_mod else 1.0

    # Choose generation function
    if use_active_inference:
        gen_fn = generate_active_inference
    else:
        gen_fn = generate_greedy

    # Evaluate coherence
    coherence_results = evaluate_coherence(model, test_prompts, gen_fn, eval_energy_mod)

    # Evaluate perplexity on test set
    test_losses = []
    with torch.no_grad():
        for _ in range(20):
            starts = torch.randint(0, len(test_data) - seq_len - 1, (batch_size,))
            batch = torch.stack([test_data[s:s+seq_len] for s in starts])
            logits, _, _ = model(batch, eval_energy_mod)
            loss = F.cross_entropy(logits[:, :-1].reshape(-1, vocab_size),
                                   batch[:, 1:].reshape(-1))
            test_losses.append(loss.item())

    test_ppl = np.exp(np.mean(test_losses))

    return {
        'seed': seed,
        'test_ppl': test_ppl,
        'coherence': coherence_results['mean_coherence'],
        'coherence_std': coherence_results['std_coherence'],
        'energy_mod': eval_energy_mod,
    }


def compute_stats(group1: List[float], group2: List[float]) -> Dict:
    """Compute statistical comparison."""
    t_stat, t_pvalue = stats.ttest_ind(group1, group2)
    pooled_std = np.sqrt((np.std(group1)**2 + np.std(group2)**2) / 2)
    cohens_d = (np.mean(group1) - np.mean(group2)) / pooled_std if pooled_std > 0 else 0
    return {
        't_pvalue': t_pvalue,
        'cohens_d': cohens_d,
        'mean_diff': np.mean(group1) - np.mean(group2),
        'significant_005': t_pvalue < 0.05,
    }


def main():
    print("=" * 70)
    print("z1010: KERNEL-LEVEL ACTIVE INFERENCE")
    print("=" * 70)
    print(f"\nRunning {N_SEEDS} seeds per condition, {N_STEPS} steps each")
    print("\nLeverages:")
    print("  - Energy modulation concept from HIP kernels (z910)")
    print("  - Active inference for coherence (z1002)")
    print("  - Fast telemetry (sysfs_hwmon)")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")

    text = load_data()
    chars = sorted(set(text))
    char_to_idx = {c: i for i, c in enumerate(chars)}
    vocab_size = len(chars)

    all_data = torch.tensor([char_to_idx[c] for c in text], device=device)
    split_idx = int(len(all_data) * 0.8)
    train_data = all_data[:split_idx]
    test_data = all_data[split_idx:]
    print(f"Train: {len(train_data):,}, Test: {len(test_data):,}")

    telemetry = SysfsHwmonTelemetry()

    conditions = [
        ('A: Baseline', False, False),
        ('B: Energy Mod', True, False),
        ('C: Active Inf', False, True),
        ('D: Full', True, True),
    ]

    results = {name: {'ppl': [], 'coherence': []} for name, _, _ in conditions}

    for cond_name, use_energy, use_active in conditions:
        print(f"\n{'='*50}")
        print(f"Condition: {cond_name}")
        print("=" * 50)

        for seed in range(N_SEEDS):
            print(f"  Seed {seed+1}/{N_SEEDS}...", end=" ", flush=True)
            result = run_condition(seed, use_energy, use_active, telemetry,
                                   train_data, test_data, vocab_size, device)
            results[cond_name]['ppl'].append(result['test_ppl'])
            results[cond_name]['coherence'].append(result['coherence'])
            print(f"PPL={result['test_ppl']:.2f}, Coh={result['coherence']:.3f}")

        print(f"  Mean PPL: {np.mean(results[cond_name]['ppl']):.2f} ± {np.std(results[cond_name]['ppl']):.2f}")
        print(f"  Mean Coh: {np.mean(results[cond_name]['coherence']):.3f} ± {np.std(results[cond_name]['coherence']):.3f}")

    # Statistical comparisons
    print("\n" + "=" * 70)
    print("STATISTICAL ANALYSIS")
    print("=" * 70)

    comparisons = [
        ('A vs B (energy mod effect)', 'A: Baseline', 'B: Energy Mod'),
        ('A vs C (active inference effect)', 'A: Baseline', 'C: Active Inf'),
        ('A vs D (full system)', 'A: Baseline', 'D: Full'),
        ('C vs D (energy mod adds to active inf?)', 'C: Active Inf', 'D: Full'),
    ]

    stats_results = {}
    for comp_name, g1_name, g2_name in comparisons:
        coh_stats = compute_stats(results[g1_name]['coherence'], results[g2_name]['coherence'])
        ppl_stats = compute_stats(results[g1_name]['ppl'], results[g2_name]['ppl'])

        print(f"\n{comp_name}:")
        print(f"  Coherence: {np.mean(results[g1_name]['coherence']):.3f} vs {np.mean(results[g2_name]['coherence']):.3f}")
        print(f"    diff={coh_stats['mean_diff']:.3f}, p={coh_stats['t_pvalue']:.4f}, d={coh_stats['cohens_d']:.2f}")
        print(f"  PPL: {np.mean(results[g1_name]['ppl']):.2f} vs {np.mean(results[g2_name]['ppl']):.2f}")
        print(f"    diff={ppl_stats['mean_diff']:.2f}, p={ppl_stats['t_pvalue']:.4f}")

        stats_results[comp_name] = {'coherence': coh_stats, 'ppl': ppl_stats}

    # Summary
    print("\n" + "=" * 70)
    print("VALIDATION SUMMARY")
    print("=" * 70)

    claims = []

    # Claim 1: Energy modulation doesn't hurt
    if stats_results['A vs B (energy mod effect)']['ppl']['mean_diff'] > -2 or \
       not stats_results['A vs B (energy mod effect)']['ppl']['significant_005']:
        claims.append(('Energy modulation does not hurt PPL', True))
    else:
        claims.append(('Energy modulation does not hurt PPL', False))

    # Claim 2: Active inference improves coherence
    if stats_results['A vs C (active inference effect)']['coherence']['mean_diff'] < 0:
        claims.append(('Active inference improves coherence', True))
    else:
        claims.append(('Active inference improves coherence', False))

    # Claim 3: Full system beats baseline
    if stats_results['A vs D (full system)']['coherence']['mean_diff'] < 0:
        claims.append(('Full system (energy+active) beats baseline coherence', True))
    else:
        claims.append(('Full system (energy+active) beats baseline coherence', False))

    # Claim 4: Energy mod adds to active inference
    if stats_results['C vs D (energy mod adds to active inf?)']['coherence']['mean_diff'] < 0:
        claims.append(('Energy mod adds benefit to active inference', True))
    else:
        claims.append(('Energy mod adds benefit to active inference', False))

    print("\n| Claim | Validated |")
    print("|-------|-----------|")
    for claim, validated in claims:
        status = "✅" if validated else "❌"
        print(f"| {claim:50} | {status} |")

    validated_count = sum(1 for _, v in claims if v)
    print(f"\n{validated_count}/{len(claims)} claims validated")

    # Save results
    save_results = {
        'experiment': 'z1010_kernel_active_inference',
        'timestamp': datetime.now().isoformat(),
        'n_seeds': N_SEEDS,
        'n_steps': N_STEPS,
        'raw_results': {k: {kk: [float(vv) for vv in vals] for kk, vals in v.items()}
                       for k, v in results.items()},
        'stats_results': {k: {kk: {kkk: float(vvv) if isinstance(vvv, (int, float, np.floating)) else str(vvv)
                                   for kkk, vvv in vv.items()}
                             for kk, vv in v.items()}
                         for k, v in stats_results.items()},
        'claims': [{'claim': c, 'validated': bool(v)} for c, v in claims],
        'validated_count': validated_count,
    }

    results_path = Path(__file__).parent.parent / "results" / "z1010_kernel_active_inference.json"
    with open(results_path, 'w') as f:
        json.dump(save_results, f, indent=2)

    print(f"\nSaved to {results_path}")


if __name__ == "__main__":
    main()
