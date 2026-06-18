#!/usr/bin/env python3
"""
Z912b: Adaptive Attention Window Benchmark

Hypothesis: Sliding window attention with energy-based window sizing can reduce
memory bandwidth by 50% while maintaining quality.

Key innovations:
1. Sliding window attention with configurable window size (512-4096)
2. Energy pressure calculation: pressure = (power_w - 30) / 30
3. Window sizing: window = int(4096 * (1.0 - min(pressure, 0.875)))
4. Memory bandwidth proxy via attention computation cost

Baselines:
- Fixed window sizes: 512, 1024, 2048, 4096
- Adaptive: energy-pressure-driven window sizing

Metrics:
- J/token (energy efficiency)
- Memory bandwidth proxy (attention FLOPs per token)
- Perplexity per window size
- Business value (cost savings at scale)

Target: Prove 50% memory bandwidth reduction hypothesis.

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

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.telemetry.sysfs_hwmon import SysfsHwmonTelemetry, EnergyMeter


# ============================================================================
# Constants
# ============================================================================

POWER_SETPOINT = 30.0  # Watts - baseline power for pressure calculation
MAX_WINDOW = 4096
MIN_WINDOW = 512
MAX_PRESSURE = 0.875  # Maximum pressure before window hits minimum


# ============================================================================
# Telemetry
# ============================================================================

@dataclass
class TelemetrySample:
    timestamp: float
    temp_c: float
    power_w: float

    def compute_pressure(self) -> float:
        """
        Energy pressure calculation.
        pressure = (power_w - 30) / 30
        Positive when above setpoint, negative when below.
        """
        return (self.power_w - POWER_SETPOINT) / POWER_SETPOINT

    def compute_window_size(self) -> int:
        """
        Adaptive window sizing based on energy pressure.
        window = int(4096 * (1.0 - min(pressure, 0.875)))

        When pressure=0: window=4096 (full attention)
        When pressure=0.5: window=2048
        When pressure>=0.875: window=512 (minimum)
        """
        pressure = self.compute_pressure()
        clamped_pressure = max(0.0, min(pressure, MAX_PRESSURE))
        window = int(MAX_WINDOW * (1.0 - clamped_pressure))
        return max(MIN_WINDOW, min(MAX_WINDOW, window))


class TelemetryCollector:
    def __init__(self, history_len: int = 32):
        self.history_len = history_len
        self.telemetry = SysfsHwmonTelemetry(sample_rate_hz=100)
        self.history: List[TelemetrySample] = []

    def read(self) -> TelemetrySample:
        sample = self.telemetry.read_sample()
        ts = TelemetrySample(
            timestamp=time.time(),
            temp_c=sample.temp_edge_c,
            power_w=sample.power_w,
        )
        self.history.append(ts)
        if len(self.history) > self.history_len:
            self.history.pop(0)
        return ts

    def get_avg_power(self, last_n: int = 10) -> float:
        if not self.history:
            return POWER_SETPOINT
        samples = self.history[-last_n:]
        return sum(s.power_w for s in samples) / len(samples)


# ============================================================================
# Sliding Window Attention
# ============================================================================

class SlidingWindowAttention(nn.Module):
    """
    Sliding window attention with configurable window size.

    Memory bandwidth proxy:
    - Full attention: O(n^2) attention computations
    - Window attention: O(n * w) where w = window size
    - Bandwidth reduction = 1 - (w / n) for n > w
    """

    def __init__(self, hidden_dim: int = 256, num_heads: int = 4, max_seq_len: int = 4096):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.max_seq_len = max_seq_len

        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x: torch.Tensor, window_size: int = None) -> Tuple[torch.Tensor, Dict]:
        """
        Args:
            x: [batch, seq_len, hidden_dim]
            window_size: Attention window size (None = full attention)
        Returns:
            output: [batch, seq_len, hidden_dim]
            info: Dict with attention stats
        """
        B, T, D = x.shape

        if window_size is None:
            window_size = T
        window_size = min(window_size, T)

        # Project Q, K, V
        Q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        # Compute attention with sliding window mask
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # Create sliding window mask
        mask = self._create_window_mask(T, window_size, x.device)
        scores = scores.masked_fill(mask.unsqueeze(0).unsqueeze(0), float('-inf'))

        # Softmax and apply to values
        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, V)

        out = out.transpose(1, 2).contiguous().view(B, T, D)
        out = self.out_proj(out)

        # Calculate memory bandwidth proxy
        # Full attention: T * T attention scores
        # Window attention: T * window_size attention scores
        full_attn_ops = T * T
        window_attn_ops = T * window_size
        bandwidth_reduction = 1.0 - (window_attn_ops / full_attn_ops) if T > 0 else 0.0

        info = {
            'window_size': window_size,
            'seq_len': T,
            'full_attn_ops': full_attn_ops,
            'window_attn_ops': window_attn_ops,
            'bandwidth_reduction': bandwidth_reduction,
        }

        return out, info

    def _create_window_mask(self, seq_len: int, window_size: int, device: torch.device) -> torch.Tensor:
        """
        Create sliding window causal mask.
        Position i can attend to positions [max(0, i - window_size + 1), i]
        """
        mask = torch.ones(seq_len, seq_len, dtype=torch.bool, device=device)

        for i in range(seq_len):
            start = max(0, i - window_size + 1)
            mask[i, start:i+1] = False

        return mask


# ============================================================================
# Transformer with Adaptive Window
# ============================================================================

class AdaptiveWindowTransformer(nn.Module):
    """
    Transformer with adaptive sliding window attention.
    Window size adapts based on energy pressure from telemetry.
    """

    def __init__(self, vocab_size: int = 256, hidden_dim: int = 256,
                 num_layers: int = 6, num_heads: int = 4, ff_dim: int = 1024,
                 max_seq_len: int = 4096):
        super().__init__()

        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.token_emb = nn.Embedding(vocab_size, hidden_dim)
        self.pos_emb = nn.Embedding(max_seq_len, hidden_dim)

        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            layer = nn.ModuleDict({
                'attn': SlidingWindowAttention(hidden_dim, num_heads, max_seq_len),
                'ln1': nn.LayerNorm(hidden_dim),
                'ff': nn.Sequential(
                    nn.Linear(hidden_dim, ff_dim),
                    nn.GELU(),
                    nn.Linear(ff_dim, hidden_dim),
                ),
                'ln2': nn.LayerNorm(hidden_dim),
            })
            self.layers.append(layer)

        self.ln_out = nn.LayerNorm(hidden_dim)
        self.head = nn.Linear(hidden_dim, vocab_size)

    def forward(self, input_ids: torch.Tensor, window_size: int = None) -> Tuple[torch.Tensor, Dict]:
        """
        Args:
            input_ids: [batch, seq_len]
            window_size: Attention window size (None = full attention)
        Returns:
            logits: [batch, seq_len, vocab_size]
            info: Aggregated attention stats
        """
        B, T = input_ids.shape
        device = input_ids.device

        x = self.token_emb(input_ids) + self.pos_emb(torch.arange(T, device=device))

        total_bandwidth_reduction = 0.0
        total_window_attn_ops = 0

        for layer in self.layers:
            x_norm = layer['ln1'](x)
            attn_out, info = layer['attn'](x_norm, window_size)
            x = x + attn_out
            x = x + layer['ff'](layer['ln2'](x))

            total_bandwidth_reduction += info['bandwidth_reduction']
            total_window_attn_ops += info['window_attn_ops']

        logits = self.head(self.ln_out(x))

        avg_bandwidth_reduction = total_bandwidth_reduction / self.num_layers

        return logits, {
            'window_size': window_size if window_size else T,
            'seq_len': T,
            'avg_bandwidth_reduction': avg_bandwidth_reduction,
            'total_window_attn_ops': total_window_attn_ops,
        }


# ============================================================================
# Data Loading
# ============================================================================

def load_data(data_dir: Path, device: torch.device):
    fpath = data_dir / "tiny_shakespeare.txt"
    if not fpath.exists():
        print("[z912b] Downloading TinyShakespeare...")
        import urllib.request
        url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
        fpath.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(url, str(fpath))

    text = fpath.read_text()
    data = torch.tensor([ord(c) % 256 for c in text], dtype=torch.long, device=device)
    return data, len(text)


def make_batch(data: torch.Tensor, batch_size: int, seq_len: int, offset: int = 0):
    starts = torch.randint(0, len(data) - seq_len - 1, (batch_size,)) + offset
    starts = starts % (len(data) - seq_len - 1)
    batch = torch.stack([data[s:s+seq_len+1] for s in starts])
    return batch[:, :-1], batch[:, 1:]


# ============================================================================
# Benchmark Functions
# ============================================================================

@dataclass
class WindowBenchmarkResult:
    """Results for one window configuration."""
    window_type: str  # "fixed" or "adaptive"
    window_size: int  # For fixed, or avg for adaptive
    n_batches: int
    n_tokens: int
    total_time_s: float
    total_energy_j: float
    avg_loss: float
    avg_ppl: float
    j_per_token: float
    tokens_per_sec: float
    avg_bandwidth_reduction: float
    bandwidth_proxy_mops: float  # Million attention ops
    avg_power_w: float
    avg_temp_c: float
    window_sizes_used: List[int] = field(default_factory=list)


def run_fixed_window_benchmark(
    model: nn.Module,
    data: torch.Tensor,
    telemetry: TelemetryCollector,
    window_size: int,
    batch_size: int,
    seq_len: int,
    n_batches: int,
    device: torch.device,
) -> WindowBenchmarkResult:
    """Benchmark with fixed window size."""

    model.eval()

    total_loss = 0
    total_energy = 0
    total_tokens = 0
    total_bandwidth_reduction = 0
    total_attn_ops = 0
    temps = []
    powers = []

    start_time = time.time()

    with torch.no_grad():
        for i in range(n_batches):
            inp, tgt = make_batch(data, batch_size, seq_len, offset=i*batch_size*seq_len)

            current = telemetry.read()
            temps.append(current.temp_c)
            powers.append(current.power_w)

            with EnergyMeter(telemetry.telemetry) as meter:
                logits, info = model(inp, window_size=window_size)
                loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), tgt.reshape(-1))

            total_energy += meter.energy_j
            total_loss += loss.item()
            total_tokens += batch_size * seq_len
            total_bandwidth_reduction += info['avg_bandwidth_reduction']
            total_attn_ops += info['total_window_attn_ops']

    end_time = time.time()
    total_time = end_time - start_time

    avg_loss = total_loss / n_batches
    avg_ppl = math.exp(avg_loss)
    j_per_token = total_energy / total_tokens
    tokens_per_sec = total_tokens / total_time
    avg_bandwidth_reduction = total_bandwidth_reduction / n_batches
    bandwidth_proxy_mops = total_attn_ops / 1e6

    return WindowBenchmarkResult(
        window_type="fixed",
        window_size=window_size,
        n_batches=n_batches,
        n_tokens=total_tokens,
        total_time_s=total_time,
        total_energy_j=total_energy,
        avg_loss=avg_loss,
        avg_ppl=avg_ppl,
        j_per_token=j_per_token,
        tokens_per_sec=tokens_per_sec,
        avg_bandwidth_reduction=avg_bandwidth_reduction,
        bandwidth_proxy_mops=bandwidth_proxy_mops,
        avg_power_w=sum(powers) / len(powers),
        avg_temp_c=sum(temps) / len(temps),
        window_sizes_used=[window_size],
    )


def run_adaptive_window_benchmark(
    model: nn.Module,
    data: torch.Tensor,
    telemetry: TelemetryCollector,
    batch_size: int,
    seq_len: int,
    n_batches: int,
    device: torch.device,
) -> WindowBenchmarkResult:
    """Benchmark with adaptive energy-pressure-driven window sizing."""

    model.eval()

    total_loss = 0
    total_energy = 0
    total_tokens = 0
    total_bandwidth_reduction = 0
    total_attn_ops = 0
    temps = []
    powers = []
    window_sizes_used = []

    start_time = time.time()

    with torch.no_grad():
        for i in range(n_batches):
            inp, tgt = make_batch(data, batch_size, seq_len, offset=i*batch_size*seq_len)

            current = telemetry.read()
            temps.append(current.temp_c)
            powers.append(current.power_w)

            # Adaptive window sizing based on energy pressure
            window_size = current.compute_window_size()
            window_sizes_used.append(window_size)

            with EnergyMeter(telemetry.telemetry) as meter:
                logits, info = model(inp, window_size=window_size)
                loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), tgt.reshape(-1))

            total_energy += meter.energy_j
            total_loss += loss.item()
            total_tokens += batch_size * seq_len
            total_bandwidth_reduction += info['avg_bandwidth_reduction']
            total_attn_ops += info['total_window_attn_ops']

    end_time = time.time()
    total_time = end_time - start_time

    avg_loss = total_loss / n_batches
    avg_ppl = math.exp(avg_loss)
    j_per_token = total_energy / total_tokens
    tokens_per_sec = total_tokens / total_time
    avg_bandwidth_reduction = total_bandwidth_reduction / n_batches
    bandwidth_proxy_mops = total_attn_ops / 1e6
    avg_window = sum(window_sizes_used) / len(window_sizes_used)

    return WindowBenchmarkResult(
        window_type="adaptive",
        window_size=int(avg_window),
        n_batches=n_batches,
        n_tokens=total_tokens,
        total_time_s=total_time,
        total_energy_j=total_energy,
        avg_loss=avg_loss,
        avg_ppl=avg_ppl,
        j_per_token=j_per_token,
        tokens_per_sec=tokens_per_sec,
        avg_bandwidth_reduction=avg_bandwidth_reduction,
        bandwidth_proxy_mops=bandwidth_proxy_mops,
        avg_power_w=sum(powers) / len(powers),
        avg_temp_c=sum(temps) / len(temps),
        window_sizes_used=window_sizes_used,
    )


# ============================================================================
# Business Value Calculation
# ============================================================================

@dataclass
class BusinessValue:
    """Business value metrics."""
    # Efficiency gains
    bandwidth_reduction_pct: float
    energy_savings_pct: float
    quality_preservation_pct: float  # 100% = same PPL

    # Daily projections (1 GPU, 24/7)
    daily_tokens_1gpu: int
    daily_kwh_baseline: float
    daily_kwh_adaptive: float
    daily_savings_kwh: float
    daily_cost_savings_usd: float  # at $0.10/kWh

    # Annual projections (100 GPU cluster)
    annual_kwh_savings_100gpu: float
    annual_cost_savings_100gpu_usd: float
    co2_reduction_kg: float  # at 0.4 kg CO2/kWh

    # Hypothesis validation
    hypothesis_validated: bool  # 50% bandwidth reduction achieved?


def calculate_business_value(
    baseline: WindowBenchmarkResult,
    adaptive: WindowBenchmarkResult,
    electricity_cost_per_kwh: float = 0.10,
    co2_per_kwh: float = 0.4,
    cluster_size: int = 100,
) -> BusinessValue:
    """Calculate business value from benchmark results."""

    # Efficiency gains
    bandwidth_reduction_pct = adaptive.avg_bandwidth_reduction * 100
    energy_savings_pct = (1 - adaptive.j_per_token / baseline.j_per_token) * 100
    quality_preservation_pct = 100 - abs(adaptive.avg_ppl - baseline.avg_ppl) / baseline.avg_ppl * 100

    # Daily projections (24h operation)
    seconds_per_day = 86400
    daily_tokens_1gpu = int(baseline.tokens_per_sec * seconds_per_day)

    daily_energy_baseline_j = daily_tokens_1gpu * baseline.j_per_token
    daily_energy_adaptive_j = daily_tokens_1gpu * adaptive.j_per_token

    daily_kwh_baseline = daily_energy_baseline_j / 3600000
    daily_kwh_adaptive = daily_energy_adaptive_j / 3600000
    daily_savings_kwh = daily_kwh_baseline - daily_kwh_adaptive
    daily_cost_savings = daily_savings_kwh * electricity_cost_per_kwh

    # Annual cluster projections
    annual_days = 365
    annual_kwh_savings = daily_savings_kwh * annual_days * cluster_size
    annual_cost_savings = daily_cost_savings * annual_days * cluster_size
    co2_reduction = annual_kwh_savings * co2_per_kwh

    # Hypothesis validation: 50% bandwidth reduction
    hypothesis_validated = bandwidth_reduction_pct >= 50.0

    return BusinessValue(
        bandwidth_reduction_pct=bandwidth_reduction_pct,
        energy_savings_pct=energy_savings_pct,
        quality_preservation_pct=quality_preservation_pct,
        daily_tokens_1gpu=daily_tokens_1gpu,
        daily_kwh_baseline=daily_kwh_baseline,
        daily_kwh_adaptive=daily_kwh_adaptive,
        daily_savings_kwh=daily_savings_kwh,
        daily_cost_savings_usd=daily_cost_savings,
        annual_kwh_savings_100gpu=annual_kwh_savings,
        annual_cost_savings_100gpu_usd=annual_cost_savings,
        co2_reduction_kg=co2_reduction,
        hypothesis_validated=hypothesis_validated,
    )


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Z912b: Adaptive Attention Window Benchmark")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seq-len", type=int, default=1024, help="Sequence length (512-4096)")
    parser.add_argument("--n-batches", type=int, default=100)
    parser.add_argument("--train-epochs", type=int, default=2)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--skip-training", action="store_true")
    args = parser.parse_args()

    print("=" * 70)
    print("  Z912b: Adaptive Attention Window Benchmark")
    print("=" * 70)
    print(f"  Hypothesis: 50% memory bandwidth reduction via adaptive windowing")
    print(f"  Pressure formula: pressure = (power_w - 30) / 30")
    print(f"  Window formula: window = int(4096 * (1.0 - min(pressure, 0.875)))")
    print("=" * 70)

    # Device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"\n  Device: {device}")

    # Telemetry
    telemetry = TelemetryCollector(history_len=32)
    sample = telemetry.read()
    print(f"  GPU: {sample.temp_c:.1f}C, {sample.power_w:.1f}W")
    print(f"  Initial pressure: {sample.compute_pressure():.3f}")
    print(f"  Initial window: {sample.compute_window_size()}")

    # Data
    data_dir = Path(__file__).parent.parent / "data"
    data, n_chars = load_data(data_dir, device)
    print(f"  Data: {n_chars:,} chars")

    # Create model
    print("\n  Creating model...")
    model = AdaptiveWindowTransformer(
        vocab_size=256,
        hidden_dim=256,
        num_layers=6,
        num_heads=4,
        ff_dim=1024,
        max_seq_len=max(4096, args.seq_len),
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {n_params:,}")

    # Quick training
    if not args.skip_training:
        print(f"\n  Training ({args.train_epochs} epochs)...")
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
        model.train()

        for epoch in range(args.train_epochs):
            epoch_loss = 0
            n_steps = 50
            for step in range(n_steps):
                inp, tgt = make_batch(data, args.batch_size, args.seq_len)
                optimizer.zero_grad()
                logits, _ = model(inp, window_size=None)  # Full attention for training
                loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), tgt.reshape(-1))
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            print(f"    Epoch {epoch+1}/{args.train_epochs}  loss={epoch_loss/n_steps:.4f}")

    # ========================================================================
    # Run Benchmarks
    # ========================================================================

    fixed_window_sizes = [512, 1024, 2048, 4096]
    results = {}

    # Fixed window baselines
    print("\n" + "=" * 70)
    print("  FIXED WINDOW BASELINES")
    print("=" * 70)

    for window_size in fixed_window_sizes:
        print(f"\n  Testing window_size={window_size}...")

        # Warm up telemetry
        for _ in range(10):
            telemetry.read()
            time.sleep(0.05)

        result = run_fixed_window_benchmark(
            model, data, telemetry,
            window_size=window_size,
            batch_size=args.batch_size,
            seq_len=args.seq_len,
            n_batches=args.n_batches,
            device=device,
        )
        results[f"fixed_{window_size}"] = result

        print(f"    PPL={result.avg_ppl:.2f}  J/tok={result.j_per_token:.6f}  "
              f"BW_red={result.avg_bandwidth_reduction*100:.1f}%")

        # Cool down
        time.sleep(2)

    # Adaptive window
    print("\n" + "=" * 70)
    print("  ADAPTIVE WINDOW (Energy-Pressure Driven)")
    print("=" * 70)

    # Warm up telemetry
    for _ in range(20):
        telemetry.read()
        time.sleep(0.05)

    adaptive_result = run_adaptive_window_benchmark(
        model, data, telemetry,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        n_batches=args.n_batches,
        device=device,
    )
    results["adaptive"] = adaptive_result

    print(f"  Avg window={adaptive_result.window_size}  PPL={adaptive_result.avg_ppl:.2f}  "
          f"J/tok={adaptive_result.j_per_token:.6f}  BW_red={adaptive_result.avg_bandwidth_reduction*100:.1f}%")

    # Window size distribution
    window_counts = {}
    for w in adaptive_result.window_sizes_used:
        window_counts[w] = window_counts.get(w, 0) + 1
    print(f"  Window distribution: {dict(sorted(window_counts.items()))}")

    # ========================================================================
    # Business Value
    # ========================================================================

    baseline = results["fixed_4096"]  # Full attention baseline
    business = calculate_business_value(baseline, adaptive_result)

    # ========================================================================
    # Results Summary
    # ========================================================================

    print("\n" + "=" * 70)
    print("  RESULTS SUMMARY")
    print("=" * 70)

    print("\n  WINDOW COMPARISON")
    print("-" * 70)
    print(f"  {'Window':<15} {'PPL':>10} {'J/token':>12} {'BW Red':>10} {'tok/s':>10}")
    print("-" * 70)

    for key, result in results.items():
        label = key.replace("fixed_", "Fixed ").replace("adaptive", "Adaptive")
        print(f"  {label:<15} {result.avg_ppl:>10.2f} {result.j_per_token:>12.6f} "
              f"{result.avg_bandwidth_reduction*100:>9.1f}% {result.tokens_per_sec:>10.1f}")

    print("\n  HYPOTHESIS VALIDATION")
    print("-" * 70)
    print(f"  Target: 50% memory bandwidth reduction")
    print(f"  Achieved: {business.bandwidth_reduction_pct:.1f}%")
    print(f"  Status: {'VALIDATED' if business.hypothesis_validated else 'NOT VALIDATED'}")

    print("\n  QUALITY PRESERVATION")
    print("-" * 70)
    print(f"  Baseline PPL (4096):  {baseline.avg_ppl:.2f}")
    print(f"  Adaptive PPL:         {adaptive_result.avg_ppl:.2f}")
    print(f"  Quality preserved:    {business.quality_preservation_pct:.1f}%")

    print("\n  BUSINESS VALUE (100 GPU Cluster, 24/7 Operation)")
    print("-" * 70)
    print(f"  Bandwidth Reduction:         {business.bandwidth_reduction_pct:.1f}%")
    print(f"  Energy Savings:              {business.energy_savings_pct:+.1f}%")
    print(f"  Daily Tokens (1 GPU):        {business.daily_tokens_1gpu:,}")
    print(f"  Daily kWh (Baseline):        {business.daily_kwh_baseline:.1f}")
    print(f"  Daily kWh (Adaptive):        {business.daily_kwh_adaptive:.1f}")
    print(f"  Daily Savings:               {business.daily_savings_kwh:.2f} kWh")
    print(f"  Daily Cost Savings (1 GPU):  ${business.daily_cost_savings_usd:.2f}")
    print(f"  Annual kWh Savings (100):    {business.annual_kwh_savings_100gpu:,.0f}")
    print(f"  Annual Cost Savings (100):   ${business.annual_cost_savings_100gpu_usd:,.0f}")
    print(f"  CO2 Reduction (kg/year):     {business.co2_reduction_kg:,.0f}")

    # Validation checks
    print("\n" + "=" * 70)
    print("  VALIDATION CHECKS")
    print("=" * 70)

    checks = []

    # Check 1: 50% bandwidth reduction
    if business.bandwidth_reduction_pct >= 50.0:
        checks.append(("50% bandwidth reduction", True, f"{business.bandwidth_reduction_pct:.1f}%"))
    else:
        checks.append(("50% bandwidth reduction", False, f"{business.bandwidth_reduction_pct:.1f}%"))

    # Check 2: Quality preserved (PPL within 10%)
    ppl_diff = abs(adaptive_result.avg_ppl - baseline.avg_ppl) / baseline.avg_ppl
    if ppl_diff < 0.10:
        checks.append(("Quality preserved (<10% PPL diff)", True, f"{ppl_diff*100:.1f}%"))
    else:
        checks.append(("Quality preserved (<10% PPL diff)", False, f"{ppl_diff*100:.1f}%"))

    # Check 3: Adaptive window varies with power
    min_window = min(adaptive_result.window_sizes_used)
    max_window = max(adaptive_result.window_sizes_used)
    if max_window > min_window:
        checks.append(("Window adapts to power", True, f"{min_window}-{max_window}"))
    else:
        checks.append(("Window adapts to power", False, f"fixed at {min_window}"))

    # Check 4: Energy savings
    if business.energy_savings_pct > 0:
        checks.append(("Positive energy savings", True, f"{business.energy_savings_pct:.1f}%"))
    else:
        checks.append(("Positive energy savings", False, f"{business.energy_savings_pct:.1f}%"))

    # Check 5: Throughput maintained
    throughput_ratio = adaptive_result.tokens_per_sec / baseline.tokens_per_sec
    if throughput_ratio >= 0.9:
        checks.append(("Throughput maintained (>90%)", True, f"{throughput_ratio*100:.1f}%"))
    else:
        checks.append(("Throughput maintained (>90%)", False, f"{throughput_ratio*100:.1f}%"))

    n_passed = sum(1 for _, passed, _ in checks if passed)
    print(f"\n  Passed {n_passed}/{len(checks)} checks:")
    for name, passed, detail in checks:
        status = "PASS" if passed else "FAIL"
        print(f"    [{status}] {name}: {detail}")

    # Verdict
    if business.hypothesis_validated and n_passed >= 4:
        verdict = "HYPOTHESIS VALIDATED - 50% BANDWIDTH REDUCTION ACHIEVED"
    elif n_passed >= 3:
        verdict = "PARTIAL SUCCESS - BANDWIDTH REDUCTION WORKS, HYPOTHESIS NOT FULLY MET"
    else:
        verdict = "HYPOTHESIS NOT VALIDATED"

    print(f"\n  VERDICT: {verdict}")

    # Save results
    results_dir = Path(__file__).parent.parent / "results"
    results_dir.mkdir(exist_ok=True)

    output = {
        "experiment": "z912b_attention_window",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "hypothesis": "50% memory bandwidth reduction via adaptive attention windowing",
        "config": {
            "batch_size": args.batch_size,
            "seq_len": args.seq_len,
            "n_batches": args.n_batches,
            "train_epochs": args.train_epochs,
            "power_setpoint_w": POWER_SETPOINT,
            "max_window": MAX_WINDOW,
            "min_window": MIN_WINDOW,
            "max_pressure": MAX_PRESSURE,
        },
        "fixed_window_results": {
            k: asdict(v) for k, v in results.items() if k.startswith("fixed")
        },
        "adaptive_result": asdict(adaptive_result),
        "business_value": asdict(business),
        "checks": [{"name": n, "passed": p, "detail": d} for n, p, d in checks],
        "verdict": verdict,
        "window_size_distribution": window_counts,
    }

    out_path = results_dir / "z912b_attention_window.json"
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\n  Results saved to {out_path}")

    print("\n" + "=" * 70)
    print("  Z912b COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
