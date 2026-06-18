#!/usr/bin/env python3
"""
Z912c: Homeostatic Batch Accumulation Benchmark

This experiment tests temperature-adaptive batch accumulation:
- Accumulate 1-8 mini-batches based on GPU temperature
- Cool GPU (<45C): accumulate 4x (maximum efficiency)
- Warm GPU (45-50C): accumulate 2x (balanced)
- Hot GPU (>50C): accumulate 1x (protect hardware)

Hypothesis:
Larger effective batch sizes are more energy-efficient (better GPU utilization),
but require more memory and increase thermal load. Homeostatic accumulation
finds the sweet spot dynamically.

Metrics:
- J/token across batch accumulation factors (1x, 2x, 4x, 8x)
- Tokens/joule curve vs batch size
- Latency per effective batch
- Perplexity (quality preservation)
- Business value calculation

Target: 30% better tokens/joule at optimal batch size vs naive batching.

Author: FEEL Research Team
Date: 2026-01-29
"""

import os
os.environ['HSA_OVERRIDE_GFX_VERSION'] = '11.0.0'

import sys
import json
import time
import math
import argparse
from pathlib import Path
from dataclasses import dataclass, asdict, field
from typing import List, Dict, Tuple, Optional
from collections import deque

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.telemetry.sysfs_hwmon import SysfsHwmonTelemetry, EnergyMeter


# ============================================================================
# Homeostatic Batch Accumulator
# ============================================================================

class HomeostaticBatchAccumulator:
    """
    Accumulates mini-batches based on GPU temperature.

    Logic:
    - temp < 45C: accumulate 4x (GPU is cool, maximize efficiency)
    - temp < 50C: accumulate 2x (moderate thermal load)
    - temp >= 50C: accumulate 1x (GPU is hot, process immediately)

    This creates a homeostatic feedback loop where:
    - Cool GPU -> larger batches -> higher throughput -> GPU heats up
    - Hot GPU -> smaller batches -> lower throughput -> GPU cools down
    """

    def __init__(self,
                 telemetry: SysfsHwmonTelemetry,
                 temp_threshold_cool: float = 45.0,
                 temp_threshold_warm: float = 50.0,
                 max_accumulation: int = 8):
        self.telemetry = telemetry
        self.temp_threshold_cool = temp_threshold_cool
        self.temp_threshold_warm = temp_threshold_warm
        self.max_accumulation = max_accumulation

        # Statistics
        self.accumulation_history: List[int] = []
        self.temp_history: List[float] = []

    def get_accumulation_factor(self) -> Tuple[int, float]:
        """
        Get current accumulation factor based on temperature.

        Returns:
            (accumulation_factor, current_temp_c)
        """
        sample = self.telemetry.read_sample()
        temp_c = sample.temp_edge_c

        if temp_c < self.temp_threshold_cool:
            factor = 4
        elif temp_c < self.temp_threshold_warm:
            factor = 2
        else:
            factor = 1

        # Clamp to max
        factor = min(factor, self.max_accumulation)

        self.accumulation_history.append(factor)
        self.temp_history.append(temp_c)

        return factor, temp_c

    def get_stats(self) -> Dict:
        """Get accumulator statistics."""
        if not self.accumulation_history:
            return {}

        return {
            'avg_accumulation': np.mean(self.accumulation_history),
            'accumulation_distribution': {
                str(k): self.accumulation_history.count(k)
                for k in set(self.accumulation_history)
            },
            'avg_temp_c': np.mean(self.temp_history),
            'min_temp_c': min(self.temp_history),
            'max_temp_c': max(self.temp_history),
            'n_decisions': len(self.accumulation_history),
        }

    def reset(self):
        """Reset statistics."""
        self.accumulation_history.clear()
        self.temp_history.clear()


# ============================================================================
# Simple Language Model for Testing
# ============================================================================

class SimpleLM(nn.Module):
    """Simple transformer LM for batch accumulation testing."""

    def __init__(self,
                 vocab_size: int = 256,
                 hidden_dim: int = 256,
                 num_layers: int = 4,
                 num_heads: int = 4,
                 ff_dim: int = 1024):
        super().__init__()

        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim

        self.token_emb = nn.Embedding(vocab_size, hidden_dim)
        self.pos_emb = nn.Embedding(512, hidden_dim)

        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(nn.ModuleDict({
                'attn': nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True),
                'ln1': nn.LayerNorm(hidden_dim),
                'ff': nn.Sequential(
                    nn.Linear(hidden_dim, ff_dim),
                    nn.GELU(),
                    nn.Linear(ff_dim, hidden_dim),
                ),
                'ln2': nn.LayerNorm(hidden_dim),
            }))

        self.ln_out = nn.LayerNorm(hidden_dim)
        self.head = nn.Linear(hidden_dim, vocab_size)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        B, T = input_ids.shape
        device = input_ids.device

        x = self.token_emb(input_ids) + self.pos_emb(torch.arange(T, device=device))

        # Causal mask
        causal_mask = torch.triu(torch.ones(T, T, device=device), diagonal=1).bool()

        for layer in self.layers:
            x_norm = layer['ln1'](x)
            attn_out, _ = layer['attn'](x_norm, x_norm, x_norm, attn_mask=causal_mask)
            x = x + attn_out
            x = x + layer['ff'](layer['ln2'](x))

        return self.head(self.ln_out(x))


# ============================================================================
# Data Loading
# ============================================================================

def load_tiny_shakespeare(data_dir: Path) -> str:
    """Load or download TinyShakespeare dataset."""
    fpath = data_dir / "tiny_shakespeare.txt"

    if fpath.exists():
        return fpath.read_text()

    print("[z912c] Downloading TinyShakespeare...")
    import urllib.request
    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    data_dir.mkdir(parents=True, exist_ok=True)

    try:
        urllib.request.urlretrieve(url, str(fpath))
        return fpath.read_text()
    except Exception as e:
        print(f"[z912c] Download failed: {e}, using synthetic data")
        text = "To be or not to be that is the question " * 10000
        fpath.write_text(text)
        return text


def make_batch(data: torch.Tensor, batch_size: int, seq_len: int,
               device: torch.device, offset: int = 0) -> Tuple[torch.Tensor, torch.Tensor]:
    """Create a random batch from data."""
    n = len(data) - seq_len - 1
    starts = (torch.randint(0, n, (batch_size,)) + offset) % n
    batch = torch.stack([data[s:s+seq_len+1] for s in starts]).to(device)
    return batch[:, :-1], batch[:, 1:]


# ============================================================================
# Benchmark Results
# ============================================================================

@dataclass
class BatchSizeResult:
    """Results for a specific batch accumulation factor."""
    accumulation_factor: int
    effective_batch_size: int
    n_effective_batches: int
    total_tokens: int
    total_time_s: float
    total_energy_j: float
    avg_loss: float
    avg_ppl: float
    j_per_token: float
    mj_per_token: float
    tokens_per_sec: float
    tokens_per_joule: float
    avg_power_w: float
    avg_temp_c: float
    min_temp_c: float
    max_temp_c: float
    latency_per_batch_ms: float


@dataclass
class AdaptiveResult:
    """Results for adaptive batch accumulation."""
    n_effective_batches: int
    total_tokens: int
    total_time_s: float
    total_energy_j: float
    avg_loss: float
    avg_ppl: float
    j_per_token: float
    mj_per_token: float
    tokens_per_sec: float
    tokens_per_joule: float
    avg_power_w: float
    avg_temp_c: float
    min_temp_c: float
    max_temp_c: float
    avg_accumulation: float
    accumulation_distribution: Dict[str, int]


@dataclass
class BusinessMetrics:
    """Business value calculations."""
    best_fixed_tpj: float
    best_fixed_factor: int
    adaptive_tpj: float
    improvement_vs_1x_pct: float
    improvement_vs_best_fixed_pct: float
    daily_tokens_1gpu: int
    daily_kwh_1x: float
    daily_kwh_adaptive: float
    daily_savings_kwh: float
    daily_cost_savings_usd: float
    annual_savings_100gpu_usd: float
    co2_reduction_kg_year: float


# ============================================================================
# Benchmark Functions
# ============================================================================

def run_fixed_batch_benchmark(
    model: nn.Module,
    data: torch.Tensor,
    telemetry: SysfsHwmonTelemetry,
    mini_batch_size: int,
    seq_len: int,
    accumulation_factor: int,
    n_effective_batches: int,
    device: torch.device,
) -> BatchSizeResult:
    """Run benchmark with fixed batch accumulation factor."""

    print(f"\n  [Fixed {accumulation_factor}x] Running {n_effective_batches} effective batches...")

    model.eval()

    effective_batch_size = mini_batch_size * accumulation_factor
    total_loss = 0.0
    total_energy = 0.0
    total_tokens = 0
    temps = []
    powers = []
    latencies = []

    start_time = time.time()

    with torch.no_grad():
        for i in range(n_effective_batches):
            batch_start = time.time()

            # Accumulate mini-batches
            accumulated_loss = 0.0
            batch_tokens = 0

            with EnergyMeter(telemetry) as meter:
                for j in range(accumulation_factor):
                    inp, tgt = make_batch(data, mini_batch_size, seq_len, device,
                                         offset=(i * accumulation_factor + j) * mini_batch_size * seq_len)

                    logits = model(inp)
                    loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), tgt.reshape(-1))
                    accumulated_loss += loss.item()
                    batch_tokens += mini_batch_size * seq_len

            # Record metrics
            sample = telemetry.read_sample()
            temps.append(sample.temp_edge_c)
            powers.append(sample.power_w)

            total_energy += meter.energy_j
            total_loss += accumulated_loss / accumulation_factor
            total_tokens += batch_tokens
            latencies.append((time.time() - batch_start) * 1000)

            if (i + 1) % 25 == 0:
                print(f"    batch {i+1}/{n_effective_batches}  "
                      f"loss={accumulated_loss/accumulation_factor:.4f}  "
                      f"temp={sample.temp_edge_c:.1f}C  "
                      f"power={sample.power_w:.1f}W")

    end_time = time.time()
    total_time = end_time - start_time

    avg_loss = total_loss / n_effective_batches

    return BatchSizeResult(
        accumulation_factor=accumulation_factor,
        effective_batch_size=effective_batch_size,
        n_effective_batches=n_effective_batches,
        total_tokens=total_tokens,
        total_time_s=total_time,
        total_energy_j=total_energy,
        avg_loss=avg_loss,
        avg_ppl=math.exp(avg_loss),
        j_per_token=total_energy / total_tokens,
        mj_per_token=(total_energy * 1000) / total_tokens,
        tokens_per_sec=total_tokens / total_time,
        tokens_per_joule=total_tokens / total_energy if total_energy > 0 else 0,
        avg_power_w=np.mean(powers),
        avg_temp_c=np.mean(temps),
        min_temp_c=min(temps),
        max_temp_c=max(temps),
        latency_per_batch_ms=np.mean(latencies),
    )


def run_adaptive_batch_benchmark(
    model: nn.Module,
    data: torch.Tensor,
    accumulator: HomeostaticBatchAccumulator,
    telemetry: SysfsHwmonTelemetry,
    mini_batch_size: int,
    seq_len: int,
    target_tokens: int,
    device: torch.device,
) -> AdaptiveResult:
    """Run benchmark with adaptive/homeostatic batch accumulation."""

    print(f"\n  [Adaptive] Running until {target_tokens:,} tokens processed...")

    model.eval()
    accumulator.reset()

    total_loss = 0.0
    total_energy = 0.0
    total_tokens = 0
    n_effective_batches = 0
    temps = []
    powers = []
    offset = 0

    start_time = time.time()

    with torch.no_grad():
        while total_tokens < target_tokens:
            # Get temperature-adaptive accumulation factor
            factor, temp_c = accumulator.get_accumulation_factor()
            temps.append(temp_c)

            # Accumulate mini-batches
            accumulated_loss = 0.0
            batch_tokens = 0

            with EnergyMeter(telemetry) as meter:
                for j in range(factor):
                    inp, tgt = make_batch(data, mini_batch_size, seq_len, device, offset=offset)
                    offset += mini_batch_size * seq_len

                    logits = model(inp)
                    loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), tgt.reshape(-1))
                    accumulated_loss += loss.item()
                    batch_tokens += mini_batch_size * seq_len

            sample = telemetry.read_sample()
            powers.append(sample.power_w)

            total_energy += meter.energy_j
            total_loss += accumulated_loss / factor
            total_tokens += batch_tokens
            n_effective_batches += 1

            if n_effective_batches % 25 == 0:
                print(f"    batch {n_effective_batches}  factor={factor}x  "
                      f"tokens={total_tokens:,}/{target_tokens:,}  "
                      f"temp={temp_c:.1f}C  loss={accumulated_loss/factor:.4f}")

    end_time = time.time()
    total_time = end_time - start_time

    avg_loss = total_loss / n_effective_batches
    stats = accumulator.get_stats()

    return AdaptiveResult(
        n_effective_batches=n_effective_batches,
        total_tokens=total_tokens,
        total_time_s=total_time,
        total_energy_j=total_energy,
        avg_loss=avg_loss,
        avg_ppl=math.exp(avg_loss),
        j_per_token=total_energy / total_tokens,
        mj_per_token=(total_energy * 1000) / total_tokens,
        tokens_per_sec=total_tokens / total_time,
        tokens_per_joule=total_tokens / total_energy if total_energy > 0 else 0,
        avg_power_w=np.mean(powers),
        avg_temp_c=np.mean(temps),
        min_temp_c=min(temps),
        max_temp_c=max(temps),
        avg_accumulation=stats.get('avg_accumulation', 1.0),
        accumulation_distribution=stats.get('accumulation_distribution', {}),
    )


def calculate_business_metrics(
    fixed_results: List[BatchSizeResult],
    adaptive_result: AdaptiveResult,
    electricity_cost_per_kwh: float = 0.10,
    co2_per_kwh: float = 0.4,
    cluster_size: int = 100,
) -> BusinessMetrics:
    """Calculate business value from benchmark results."""

    # Find best fixed configuration
    best_fixed = max(fixed_results, key=lambda r: r.tokens_per_joule)
    result_1x = next(r for r in fixed_results if r.accumulation_factor == 1)

    # Improvements
    improvement_vs_1x = (adaptive_result.tokens_per_joule / result_1x.tokens_per_joule - 1) * 100
    improvement_vs_best = (adaptive_result.tokens_per_joule / best_fixed.tokens_per_joule - 1) * 100

    # Daily projections (24h)
    seconds_per_day = 86400
    daily_tokens = int(result_1x.tokens_per_sec * seconds_per_day)

    # Energy for same token count
    daily_energy_1x_j = daily_tokens * result_1x.j_per_token
    daily_energy_adaptive_j = daily_tokens * adaptive_result.j_per_token

    daily_kwh_1x = daily_energy_1x_j / 3_600_000
    daily_kwh_adaptive = daily_energy_adaptive_j / 3_600_000
    daily_savings_kwh = daily_kwh_1x - daily_kwh_adaptive

    daily_cost_savings = daily_savings_kwh * electricity_cost_per_kwh
    annual_savings_100gpu = daily_cost_savings * 365 * cluster_size
    co2_reduction = daily_savings_kwh * 365 * cluster_size * co2_per_kwh

    return BusinessMetrics(
        best_fixed_tpj=best_fixed.tokens_per_joule,
        best_fixed_factor=best_fixed.accumulation_factor,
        adaptive_tpj=adaptive_result.tokens_per_joule,
        improvement_vs_1x_pct=improvement_vs_1x,
        improvement_vs_best_fixed_pct=improvement_vs_best,
        daily_tokens_1gpu=daily_tokens,
        daily_kwh_1x=daily_kwh_1x,
        daily_kwh_adaptive=daily_kwh_adaptive,
        daily_savings_kwh=daily_savings_kwh,
        daily_cost_savings_usd=daily_cost_savings,
        annual_savings_100gpu_usd=annual_savings_100gpu,
        co2_reduction_kg_year=co2_reduction,
    )


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Z912c: Homeostatic Batch Accumulation Benchmark")
    parser.add_argument("--mini-batch-size", type=int, default=16,
                        help="Base mini-batch size")
    parser.add_argument("--seq-len", type=int, default=128,
                        help="Sequence length")
    parser.add_argument("--n-effective-batches", type=int, default=100,
                        help="Number of effective batches per fixed config")
    parser.add_argument("--temp-cool", type=float, default=45.0,
                        help="Temperature threshold for cool GPU (4x accumulation)")
    parser.add_argument("--temp-warm", type=float, default=50.0,
                        help="Temperature threshold for warm GPU (2x accumulation)")
    parser.add_argument("--device", type=str, default="auto",
                        help="Device: auto, cuda, or cpu")
    parser.add_argument("--train-steps", type=int, default=100,
                        help="Training steps before benchmark")
    args = parser.parse_args()

    print("=" * 70)
    print("  Z912c: Homeostatic Batch Accumulation Benchmark")
    print("=" * 70)

    # Device setup
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"\n  Device: {device}")

    # Telemetry
    telemetry = SysfsHwmonTelemetry(sample_rate_hz=100)
    sample = telemetry.read_sample()
    print(f"  GPU: {sample.temp_edge_c:.1f}C, {sample.power_w:.1f}W")

    # Homeostatic accumulator
    accumulator = HomeostaticBatchAccumulator(
        telemetry=telemetry,
        temp_threshold_cool=args.temp_cool,
        temp_threshold_warm=args.temp_warm,
        max_accumulation=8,
    )
    print(f"  Accumulation thresholds: cool<{args.temp_cool}C (4x), "
          f"warm<{args.temp_warm}C (2x), hot>=50C (1x)")

    # Load data
    data_dir = Path(__file__).parent.parent / "data"
    text = load_tiny_shakespeare(data_dir)
    data = torch.tensor([ord(c) % 256 for c in text], dtype=torch.long)
    print(f"  Data: {len(text):,} chars")

    # Create model
    print("\n  Creating model...")
    model = SimpleLM(
        vocab_size=256,
        hidden_dim=256,
        num_layers=4,
        num_heads=4,
        ff_dim=1024,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {n_params:,}")

    # Quick training for reasonable weights
    if args.train_steps > 0:
        print(f"\n  Training for {args.train_steps} steps...")
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
        model.train()

        for step in range(args.train_steps):
            inp, tgt = make_batch(data, args.mini_batch_size, args.seq_len, device)
            optimizer.zero_grad()
            logits = model(inp)
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), tgt.reshape(-1))
            loss.backward()
            optimizer.step()

            if (step + 1) % 25 == 0:
                print(f"    step {step+1}/{args.train_steps}  loss={loss.item():.4f}")

    # ========================================================================
    # BENCHMARK 1: Fixed Batch Sizes
    # ========================================================================
    print("\n" + "=" * 70)
    print("  BENCHMARK 1: Fixed Batch Accumulation Factors")
    print("=" * 70)

    fixed_factors = [1, 2, 4, 8]
    fixed_results: List[BatchSizeResult] = []

    for factor in fixed_factors:
        # Cool down between tests
        print(f"\n  Cooling down (5s)...")
        time.sleep(5)

        result = run_fixed_batch_benchmark(
            model=model,
            data=data,
            telemetry=telemetry,
            mini_batch_size=args.mini_batch_size,
            seq_len=args.seq_len,
            accumulation_factor=factor,
            n_effective_batches=args.n_effective_batches,
            device=device,
        )
        fixed_results.append(result)

        print(f"    Result: {result.tokens_per_joule:.1f} tok/J, "
              f"{result.mj_per_token:.4f} mJ/tok, "
              f"PPL={result.avg_ppl:.2f}")

    # ========================================================================
    # BENCHMARK 2: Adaptive/Homeostatic Batch Accumulation
    # ========================================================================
    print("\n" + "=" * 70)
    print("  BENCHMARK 2: Adaptive Batch Accumulation (Homeostatic)")
    print("=" * 70)

    # Match token count from 1x benchmark
    target_tokens = fixed_results[0].total_tokens

    print(f"\n  Cooling down (10s) before adaptive test...")
    time.sleep(10)

    adaptive_result = run_adaptive_batch_benchmark(
        model=model,
        data=data,
        accumulator=accumulator,
        telemetry=telemetry,
        mini_batch_size=args.mini_batch_size,
        seq_len=args.seq_len,
        target_tokens=target_tokens,
        device=device,
    )

    print(f"\n    Result: {adaptive_result.tokens_per_joule:.1f} tok/J, "
          f"{adaptive_result.mj_per_token:.4f} mJ/tok, "
          f"PPL={adaptive_result.avg_ppl:.2f}")
    print(f"    Avg accumulation: {adaptive_result.avg_accumulation:.2f}x")
    print(f"    Distribution: {adaptive_result.accumulation_distribution}")

    # ========================================================================
    # Business Metrics
    # ========================================================================
    print("\n" + "=" * 70)
    print("  BUSINESS METRICS")
    print("=" * 70)

    business = calculate_business_metrics(fixed_results, adaptive_result)

    # ========================================================================
    # Results Summary
    # ========================================================================
    print("\n" + "=" * 70)
    print("  RESULTS SUMMARY")
    print("=" * 70)

    print("\n  TOKENS/JOULE CURVE vs BATCH ACCUMULATION")
    print("-" * 70)
    print(f"  {'Factor':<10} {'Eff.Batch':<12} {'Tok/J':>12} {'mJ/tok':>12} {'PPL':>10} {'Temp(C)':>10}")
    print("-" * 70)

    for r in fixed_results:
        marker = " *" if r.tokens_per_joule == business.best_fixed_tpj else ""
        print(f"  {r.accumulation_factor}x{marker:<8} {r.effective_batch_size:<12} "
              f"{r.tokens_per_joule:>12.1f} {r.mj_per_token:>12.4f} "
              f"{r.avg_ppl:>10.2f} {r.avg_temp_c:>10.1f}")

    print("-" * 70)
    print(f"  {'Adaptive':<10} {'variable':<12} "
          f"{adaptive_result.tokens_per_joule:>12.1f} {adaptive_result.mj_per_token:>12.4f} "
          f"{adaptive_result.avg_ppl:>10.2f} {adaptive_result.avg_temp_c:>10.1f}")
    print("-" * 70)
    print(f"  * = best fixed configuration ({business.best_fixed_factor}x)")

    print("\n  EFFICIENCY COMPARISON")
    print("-" * 70)
    result_1x = fixed_results[0]

    print(f"  Improvement vs 1x (baseline):     {business.improvement_vs_1x_pct:>+.1f}%")
    print(f"  Improvement vs best fixed ({business.best_fixed_factor}x):  {business.improvement_vs_best_fixed_pct:>+.1f}%")

    print("\n  QUALITY PRESERVATION")
    print("-" * 70)
    ppl_1x = result_1x.avg_ppl
    ppl_adaptive = adaptive_result.avg_ppl
    ppl_diff = (ppl_adaptive / ppl_1x - 1) * 100
    print(f"  PPL (1x baseline):    {ppl_1x:.2f}")
    print(f"  PPL (adaptive):       {ppl_adaptive:.2f}")
    print(f"  PPL delta:            {ppl_diff:+.1f}%")

    print("\n  LATENCY BREAKDOWN")
    print("-" * 70)
    print(f"  {'Factor':<10} {'Latency/batch (ms)':>20}")
    print("-" * 70)
    for r in fixed_results:
        print(f"  {r.accumulation_factor}x{'':<8} {r.latency_per_batch_ms:>20.1f}")

    print("\n  THERMAL BEHAVIOR")
    print("-" * 70)
    print(f"  Adaptive accumulation distribution:")
    for factor, count in sorted(adaptive_result.accumulation_distribution.items()):
        pct = count / adaptive_result.n_effective_batches * 100
        print(f"    {factor}x: {count} decisions ({pct:.1f}%)")
    print(f"  Average accumulation: {adaptive_result.avg_accumulation:.2f}x")
    print(f"  Temperature range: {adaptive_result.min_temp_c:.1f}C - {adaptive_result.max_temp_c:.1f}C")

    print("\n  BUSINESS VALUE (100 GPU Cluster, 24/7)")
    print("-" * 70)
    print(f"  Daily tokens (1 GPU):           {business.daily_tokens_1gpu:,}")
    print(f"  Daily kWh (1x baseline):        {business.daily_kwh_1x:.2f}")
    print(f"  Daily kWh (adaptive):           {business.daily_kwh_adaptive:.2f}")
    print(f"  Daily savings (kWh):            {business.daily_savings_kwh:.2f}")
    print(f"  Daily cost savings (1 GPU):     ${business.daily_cost_savings_usd:.2f}")
    print(f"  Annual savings (100 GPUs):      ${business.annual_savings_100gpu_usd:,.0f}")
    print(f"  CO2 reduction (kg/year):        {business.co2_reduction_kg_year:,.0f}")

    # ========================================================================
    # Validation Checks
    # ========================================================================
    print("\n" + "=" * 70)
    print("  VALIDATION CHECKS")
    print("=" * 70)

    checks = []

    # Check 1: Adaptive beats 1x baseline
    if business.improvement_vs_1x_pct > 0:
        checks.append(("Adaptive beats 1x baseline", True, f"{business.improvement_vs_1x_pct:+.1f}%"))
    else:
        checks.append(("Adaptive beats 1x baseline", False, f"{business.improvement_vs_1x_pct:+.1f}%"))

    # Check 2: Target 30% improvement
    target_improvement = 30.0
    if business.improvement_vs_1x_pct >= target_improvement:
        checks.append((f"Target {target_improvement:.0f}% improvement", True,
                      f"{business.improvement_vs_1x_pct:.1f}%"))
    else:
        checks.append((f"Target {target_improvement:.0f}% improvement", False,
                      f"{business.improvement_vs_1x_pct:.1f}% (need {target_improvement:.0f}%)"))

    # Check 3: Quality preserved (PPL within 5%)
    if abs(ppl_diff) < 5:
        checks.append(("Quality preserved (<5% PPL diff)", True, f"{ppl_diff:+.1f}%"))
    else:
        checks.append(("Quality preserved (<5% PPL diff)", False, f"{ppl_diff:+.1f}%"))

    # Check 4: Adaptive used multiple accumulation factors
    n_factors_used = len(adaptive_result.accumulation_distribution)
    if n_factors_used >= 2:
        checks.append(("Homeostatic adaptation active", True, f"{n_factors_used} factors used"))
    else:
        checks.append(("Homeostatic adaptation active", False, f"only {n_factors_used} factor(s)"))

    # Check 5: Positive business value
    if business.annual_savings_100gpu_usd > 0:
        checks.append(("Positive business value", True,
                      f"${business.annual_savings_100gpu_usd:,.0f}/year"))
    else:
        checks.append(("Positive business value", False,
                      f"${business.annual_savings_100gpu_usd:,.0f}/year"))

    n_passed = sum(1 for _, passed, _ in checks if passed)
    print(f"\n  Passed {n_passed}/{len(checks)} checks:")
    for name, passed, detail in checks:
        status = "PASS" if passed else "FAIL"
        print(f"    [{status}] {name}: {detail}")

    # Verdict
    if n_passed >= 4 and business.improvement_vs_1x_pct >= target_improvement:
        verdict = "TARGET ACHIEVED - 30%+ improvement demonstrated"
    elif n_passed >= 4:
        verdict = "HOMEOSTATIC BATCHING WORKS - improvement below target"
    elif n_passed >= 3:
        verdict = "PARTIAL SUCCESS - needs optimization"
    else:
        verdict = "NEEDS INVESTIGATION"

    print(f"\n  VERDICT: {verdict}")

    # ========================================================================
    # Save Results
    # ========================================================================
    results_dir = Path(__file__).parent.parent / "results"
    results_dir.mkdir(exist_ok=True)

    output = {
        "experiment": "z912c_batch_accumulation",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "config": {
            "mini_batch_size": args.mini_batch_size,
            "seq_len": args.seq_len,
            "n_effective_batches": args.n_effective_batches,
            "temp_threshold_cool": args.temp_cool,
            "temp_threshold_warm": args.temp_warm,
            "train_steps": args.train_steps,
        },
        "fixed_results": [asdict(r) for r in fixed_results],
        "adaptive_result": asdict(adaptive_result),
        "business_metrics": asdict(business),
        "tokens_per_joule_curve": {
            str(r.accumulation_factor): r.tokens_per_joule for r in fixed_results
        } | {"adaptive": adaptive_result.tokens_per_joule},
        "checks": [{"name": n, "passed": p, "detail": d} for n, p, d in checks],
        "verdict": verdict,
    }

    out_path = results_dir / "z912c_batch_accumulation.json"
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\n  Results saved to {out_path}")

    print("\n" + "=" * 70)
    print("  Z912c COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
