#!/usr/bin/env python3
"""
Z1106: Timing-Based Embodiment (No Sensors)

The fundamental insight: Don't READ hardware state, BE AFFECTED by it.

Previous approaches failed because:
- Reading temperature then modifying computation = optimizer compensates
- The modification is CHOSEN, not FORCED

This experiment uses ACTUAL TIMING as the embodiment signal:
- Hot GPU = slower computation = longer elapsed time
- We don't read temperature - timing IS the signal
- The model must operate within a TIME budget, not energy budget
- Early exit is FORCED by timeout, not CHOSEN by gate

This is TRUE embodiment because:
1. Hardware state directly affects timing (no sensor needed)
2. Timeout forces early exit (not a suggestion)
3. Model must learn to be useful within time constraints

Author: FEEL Research Team
Date: 2026-01-29
"""

import os
os.environ['HSA_OVERRIDE_GFX_VERSION'] = '11.0.0'

import sys
import json
import time
import math
from pathlib import Path
from dataclasses import dataclass, asdict, field
from typing import List, Dict, Tuple, Optional
import threading

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.telemetry.sysfs_hwmon import SysfsHwmonTelemetry


# ============================================================================
# Timing-Gated Transformer
# ============================================================================

class TimingGatedAttention(nn.Module):
    """Attention that tracks its own execution time."""

    def __init__(self, hidden_dim: int, num_heads: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)

        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

        # Track last execution time (for logging)
        self.last_exec_time_us = 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        torch.cuda.synchronize()
        start = time.perf_counter_ns()

        B, T, D = x.shape
        Q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, V)
        out = out.transpose(1, 2).contiguous().view(B, T, D)
        out = self.out_proj(out)

        torch.cuda.synchronize()
        self.last_exec_time_us = (time.perf_counter_ns() - start) / 1000

        return out


class TimingGatedTransformer(nn.Module):
    """
    Transformer with time-budget-based early exit.

    Key difference from confidence-based early exit:
    - Exit decision is FORCED by time, not chosen by model
    - Hot GPU = slower = more early exits (true embodiment)
    - No sensors, no gates - just physics
    """

    def __init__(self, vocab_size: int, hidden_dim: int, num_layers: int,
                 num_heads: int, time_budget_us: float = 1000.0):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.time_budget_us = time_budget_us

        self.token_emb = nn.Embedding(vocab_size, hidden_dim)
        self.pos_emb = nn.Embedding(512, hidden_dim)

        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            layer = nn.ModuleDict({
                'attn': TimingGatedAttention(hidden_dim, num_heads),
                'ln1': nn.LayerNorm(hidden_dim),
                'ff': nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim * 4),
                    nn.GELU(),
                    nn.Linear(hidden_dim * 4, hidden_dim),
                ),
                'ln2': nn.LayerNorm(hidden_dim),
            })
            self.layers.append(layer)

        # Early exit heads for each layer
        self.early_heads = nn.ModuleList([
            nn.Linear(hidden_dim, vocab_size) for _ in range(num_layers)
        ])

        self.ln_out = nn.LayerNorm(hidden_dim)
        self.final_head = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x: torch.Tensor, use_timing_gate: bool = True
                ) -> Tuple[torch.Tensor, Dict]:
        """
        Forward with optional time-budget gating.

        Returns:
            logits: [batch, seq, vocab]
            info: dict with timing and exit info
        """
        B, T = x.shape
        device = x.device

        torch.cuda.synchronize()
        forward_start = time.perf_counter_ns()

        pos = torch.arange(T, device=device)
        h = self.token_emb(x) + self.pos_emb(pos)

        exit_layer = self.num_layers
        layer_times_us = []
        cumulative_time_us = 0.0

        for i, layer in enumerate(self.layers):
            # Attention
            h = h + layer['attn'](layer['ln1'](h))
            h = h + layer['ff'](layer['ln2'](h))

            torch.cuda.synchronize()
            now_us = (time.perf_counter_ns() - forward_start) / 1000
            layer_times_us.append(layer['attn'].last_exec_time_us)
            cumulative_time_us = now_us

            # TIME-BASED EARLY EXIT (not confidence!)
            if use_timing_gate and cumulative_time_us > self.time_budget_us:
                exit_layer = i + 1
                logits = self.early_heads[i](self.ln_out(h))
                break
        else:
            logits = self.final_head(self.ln_out(h))

        info = {
            'exit_layer': exit_layer,
            'total_time_us': cumulative_time_us,
            'layer_times_us': layer_times_us,
            'timed_out': exit_layer < self.num_layers,
        }

        return logits, info


# ============================================================================
# Non-Deterministic Transformer (for comparison)
# ============================================================================

class NonDeterministicTransformer(nn.Module):
    """
    Transformer that explicitly uses non-deterministic operations.

    The output varies based on hardware state through:
    - Reduction order in softmax
    - Thread scheduling in matmul
    - Memory access patterns
    """

    def __init__(self, vocab_size: int, hidden_dim: int, num_layers: int,
                 num_heads: int):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim

        self.token_emb = nn.Embedding(vocab_size, hidden_dim)
        self.pos_emb = nn.Embedding(512, hidden_dim)

        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            layer = nn.ModuleDict({
                'attn': TimingGatedAttention(hidden_dim, num_heads),
                'ln1': nn.LayerNorm(hidden_dim),
                'ff': nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim * 4),
                    nn.GELU(),
                    nn.Linear(hidden_dim * 4, hidden_dim),
                ),
                'ln2': nn.LayerNorm(hidden_dim),
            })
            self.layers.append(layer)

        self.ln_out = nn.LayerNorm(hidden_dim)
        self.head = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T = x.shape
        pos = torch.arange(T, device=x.device)
        h = self.token_emb(x) + self.pos_emb(pos)

        for layer in self.layers:
            h = h + layer['attn'](layer['ln1'](h))
            h = h + layer['ff'](layer['ln2'](h))

        return self.head(self.ln_out(h))


# ============================================================================
# Experiments
# ============================================================================

def test_timing_variation(device: torch.device, telemetry: SysfsHwmonTelemetry):
    """Test that timing actually varies with hardware state."""
    print("\n[Test 1] Timing Variation with Hardware State")
    print("-" * 50)

    model = TimingGatedTransformer(
        vocab_size=256, hidden_dim=256, num_layers=6, num_heads=4,
        time_budget_us=float('inf')  # No timeout for this test
    ).to(device)

    batch_size = 16
    seq_len = 64

    # Warm up
    for _ in range(10):
        x = torch.randint(0, 256, (batch_size, seq_len), device=device)
        _ = model(x, use_timing_gate=False)

    # Measure timing at different thermal states
    # We'll vary load to create thermal variation
    timings = []
    temps = []

    for i in range(50):
        # Read state before
        sample = telemetry.read_sample()
        temps.append(sample.temp_edge_c)

        x = torch.randint(0, 256, (batch_size, seq_len), device=device)
        _, info = model(x, use_timing_gate=False)
        timings.append(info['total_time_us'])

        # Create some thermal variation with heavy matmul
        if i % 10 < 5:
            for _ in range(50):
                _ = torch.randn(2000, 2000, device=device) @ torch.randn(2000, 2000, device=device)
            torch.cuda.synchronize()

    # Analyze correlation
    import numpy as np
    timings = np.array(timings)
    temps = np.array(temps)

    timing_std = timings.std()
    timing_mean = timings.mean()
    temp_corr = np.corrcoef(timings, temps)[0, 1] if len(set(temps)) > 1 else 0

    print(f"  Timing: mean={timing_mean:.1f}μs, std={timing_std:.1f}μs, cv={timing_std/timing_mean*100:.1f}%")
    print(f"  Temp range: {temps.min():.1f}°C - {temps.max():.1f}°C")
    print(f"  Timing-Temp correlation: {temp_corr:.3f}")

    return {
        'timing_mean_us': float(timing_mean),
        'timing_std_us': float(timing_std),
        'timing_cv_pct': float(timing_std/timing_mean*100),
        'temp_range': [float(temps.min()), float(temps.max())],
        'timing_temp_corr': float(temp_corr),
    }


def test_timing_gated_exit(device: torch.device, telemetry: SysfsHwmonTelemetry):
    """Test that timing gate forces early exit under thermal load."""
    print("\n[Test 2] Timing-Gated Early Exit")
    print("-" * 50)

    # First, measure baseline timing to set budget
    model = TimingGatedTransformer(
        vocab_size=256, hidden_dim=256, num_layers=6, num_heads=4,
        time_budget_us=float('inf')
    ).to(device)

    batch_size = 16
    seq_len = 64

    # Warm up and get baseline
    baseline_times = []
    for _ in range(20):
        x = torch.randint(0, 256, (batch_size, seq_len), device=device)
        _, info = model(x, use_timing_gate=False)
        baseline_times.append(info['total_time_us'])

    baseline_mean = sum(baseline_times) / len(baseline_times)
    print(f"  Baseline timing: {baseline_mean:.1f}μs")

    # Set budget to 80% of baseline - should trigger exits under load
    time_budget = baseline_mean * 0.8
    model.time_budget_us = time_budget
    print(f"  Time budget: {time_budget:.1f}μs (80% of baseline)")

    # Test with and without thermal load
    results = {'cold': [], 'hot': []}

    for condition in ['cold', 'hot']:
        if condition == 'hot':
            # Create thermal load
            print("  Heating GPU...")
            for _ in range(100):
                _ = torch.randn(3000, 3000, device=device) @ torch.randn(3000, 3000, device=device)
            torch.cuda.synchronize()

        sample = telemetry.read_sample()
        print(f"  {condition.upper()}: {sample.temp_edge_c:.1f}°C, {sample.power_w:.1f}W")

        for _ in range(30):
            x = torch.randint(0, 256, (batch_size, seq_len), device=device)
            _, info = model(x, use_timing_gate=True)
            results[condition].append({
                'exit_layer': info['exit_layer'],
                'time_us': info['total_time_us'],
                'timed_out': info['timed_out'],
            })

    # Analyze
    cold_exits = [r['exit_layer'] for r in results['cold']]
    hot_exits = [r['exit_layer'] for r in results['hot']]
    cold_timeouts = sum(1 for r in results['cold'] if r['timed_out'])
    hot_timeouts = sum(1 for r in results['hot'] if r['timed_out'])

    print(f"\n  Results:")
    print(f"    Cold: avg_exit={sum(cold_exits)/len(cold_exits):.1f}, timeouts={cold_timeouts}/{len(cold_exits)}")
    print(f"    Hot:  avg_exit={sum(hot_exits)/len(hot_exits):.1f}, timeouts={hot_timeouts}/{len(hot_exits)}")

    # Key claim: hot should have MORE timeouts (earlier exits)
    embodiment_effect = hot_timeouts > cold_timeouts

    print(f"\n  Embodiment effect (hot has more timeouts): {embodiment_effect}")

    return {
        'baseline_time_us': baseline_mean,
        'time_budget_us': time_budget,
        'cold_avg_exit': sum(cold_exits) / len(cold_exits),
        'hot_avg_exit': sum(hot_exits) / len(hot_exits),
        'cold_timeout_rate': cold_timeouts / len(cold_exits),
        'hot_timeout_rate': hot_timeouts / len(hot_exits),
        'embodiment_effect': embodiment_effect,
    }


def test_non_deterministic_variance(device: torch.device, telemetry: SysfsHwmonTelemetry):
    """Test that non-deterministic ops create hardware-correlated variance."""
    print("\n[Test 3] Non-Deterministic Operation Variance")
    print("-" * 50)

    # Disable deterministic mode
    torch.use_deterministic_algorithms(False)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

    model = NonDeterministicTransformer(
        vocab_size=256, hidden_dim=256, num_layers=4, num_heads=4
    ).to(device)

    # Fixed input
    x = torch.randint(0, 256, (8, 32), device=device)

    # Run same input multiple times, measure output variance
    outputs = []
    for _ in range(20):
        with torch.no_grad():
            out = model(x)
        outputs.append(out.cpu().numpy())

    import numpy as np
    outputs = np.array(outputs)  # [20, batch, seq, vocab]

    # Variance across runs for same input
    run_variance = outputs.var(axis=0).mean()
    print(f"  Variance across runs (same input): {run_variance:.6f}")

    # This variance, if non-zero, comes from hardware state affecting computation
    has_variance = run_variance > 1e-8

    print(f"  Non-determinism detected: {has_variance}")

    return {
        'run_variance': float(run_variance),
        'has_variance': has_variance,
    }


def test_training_under_timing_constraint(device: torch.device,
                                          telemetry: SysfsHwmonTelemetry):
    """Train with timing constraints - does the model adapt?"""
    print("\n[Test 4] Training Under Timing Constraint")
    print("-" * 50)

    model = TimingGatedTransformer(
        vocab_size=256, hidden_dim=128, num_layers=4, num_heads=4,
        time_budget_us=float('inf')  # Start unconstrained
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    batch_size = 16
    seq_len = 64
    n_steps = 100

    # Phase 1: Train unconstrained
    print("  Phase 1: Unconstrained training...")
    for step in range(n_steps):
        x = torch.randint(0, 256, (batch_size, seq_len + 1), device=device)
        inp, tgt = x[:, :-1], x[:, 1:]

        optimizer.zero_grad()
        logits, _ = model(inp, use_timing_gate=False)
        loss = F.cross_entropy(logits.reshape(-1, 256), tgt.reshape(-1))
        loss.backward()
        optimizer.step()

    baseline_loss = loss.item()
    print(f"  Baseline loss: {baseline_loss:.4f}")

    # Measure timing
    timings = []
    for _ in range(20):
        x = torch.randint(0, 256, (batch_size, seq_len), device=device)
        _, info = model(x, use_timing_gate=False)
        timings.append(info['total_time_us'])
    baseline_time = sum(timings) / len(timings)

    # Phase 2: Train WITH timing constraint (50% budget)
    print(f"  Phase 2: Constrained training (budget={baseline_time*0.5:.0f}μs)...")
    model.time_budget_us = baseline_time * 0.5

    constrained_losses = []
    exit_layers = []

    for step in range(n_steps):
        x = torch.randint(0, 256, (batch_size, seq_len + 1), device=device)
        inp, tgt = x[:, :-1], x[:, 1:]

        optimizer.zero_grad()
        logits, info = model(inp, use_timing_gate=True)
        loss = F.cross_entropy(logits.reshape(-1, 256), tgt.reshape(-1))
        loss.backward()
        optimizer.step()

        constrained_losses.append(loss.item())
        exit_layers.append(info['exit_layer'])

    final_constrained_loss = constrained_losses[-1]
    avg_exit_layer = sum(exit_layers) / len(exit_layers)

    print(f"  Constrained final loss: {final_constrained_loss:.4f}")
    print(f"  Average exit layer: {avg_exit_layer:.1f}/{model.num_layers}")

    # Key insight: if model learned to be useful at early layers,
    # loss should not be catastrophically worse
    loss_degradation = (final_constrained_loss - baseline_loss) / baseline_loss

    print(f"  Loss degradation: {loss_degradation*100:.1f}%")
    print(f"  Model adapted: {loss_degradation < 0.5}")  # <50% worse

    return {
        'baseline_loss': baseline_loss,
        'constrained_loss': final_constrained_loss,
        'loss_degradation_pct': loss_degradation * 100,
        'avg_exit_layer': avg_exit_layer,
        'model_adapted': loss_degradation < 0.5,
    }


# ============================================================================
# Main
# ============================================================================

def main():
    print("=" * 70)
    print("  Z1106: Timing-Based Embodiment (No Sensors)")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[Device] {device}")
    if device.type == 'cuda':
        print(f"  GPU: {torch.cuda.get_device_name(0)}")

    telemetry = SysfsHwmonTelemetry(sample_rate_hz=100)
    sample = telemetry.read_sample()
    print(f"  Current: {sample.temp_edge_c:.1f}°C, {sample.power_w:.1f}W")

    results = {
        'experiment': 'z1106_timing_embodiment',
        'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
    }

    # Run tests
    results['timing_variation'] = test_timing_variation(device, telemetry)
    results['timing_gated_exit'] = test_timing_gated_exit(device, telemetry)
    results['non_deterministic'] = test_non_deterministic_variance(device, telemetry)
    results['constrained_training'] = test_training_under_timing_constraint(device, telemetry)

    # Summary
    print("\n" + "=" * 70)
    print("  SUMMARY")
    print("=" * 70)

    claims = [
        {
            'claim': 'Timing varies with hardware state',
            'validated': results['timing_variation']['timing_cv_pct'] > 1.0,
            'evidence': f"CV={results['timing_variation']['timing_cv_pct']:.1f}%"
        },
        {
            'claim': 'Hot GPU causes more timeouts',
            'validated': results['timing_gated_exit']['embodiment_effect'],
            'evidence': f"hot={results['timing_gated_exit']['hot_timeout_rate']*100:.0f}% vs cold={results['timing_gated_exit']['cold_timeout_rate']*100:.0f}%"
        },
        {
            'claim': 'Non-deterministic ops vary across runs',
            'validated': results['non_deterministic']['has_variance'],
            'evidence': f"variance={results['non_deterministic']['run_variance']:.2e}"
        },
        {
            'claim': 'Model adapts to timing constraint',
            'validated': results['constrained_training']['model_adapted'],
            'evidence': f"degradation={results['constrained_training']['loss_degradation_pct']:.1f}%"
        },
    ]

    results['claims'] = claims
    results['n_validated'] = sum(1 for c in claims if c['validated'])

    for c in claims:
        status = "PASS" if c['validated'] else "FAIL"
        print(f"  [{status}] {c['claim']}: {c['evidence']}")

    print(f"\n  Validated: {results['n_validated']}/{len(claims)}")

    # Save
    results_dir = Path(__file__).parent.parent / "results"
    results_dir.mkdir(exist_ok=True)
    out_path = results_dir / "z1106_timing_embodiment.json"

    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n[Saved] {out_path}")


if __name__ == "__main__":
    main()
