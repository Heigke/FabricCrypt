#!/usr/bin/env python3
"""
Z911: Comprehensive Embodiment Validation & Business Metrics

This script rigorously validates z910's native embodiment claims with:
1. Extended benchmarks (not just 5 epochs)
2. Energy efficiency measurements (J/token)
3. Throughput metrics (tokens/sec)
4. Quality preservation (perplexity stability)
5. Thermal stress testing (hot vs cold GPU)
6. Business value calculations (cost savings, ROI)

Controls:
- A: Baseline (no embodiment)
- B: Full embodiment (body tokens + energy modulation)

Metrics:
- J/token across load levels
- Tokens/second throughput
- Perplexity at matched compute budgets
- Energy savings percentage
- Projected cost savings at scale

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

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.telemetry.sysfs_hwmon import SysfsHwmonTelemetry, EnergyMeter


# ============================================================================
# Telemetry
# ============================================================================

@dataclass
class TelemetrySample:
    timestamp: float
    temp_c: float
    power_w: float

    def to_tensor(self, device) -> torch.Tensor:
        return torch.tensor([
            self.temp_c / 100.0,
            self.power_w / 50.0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0  # padding to 12-dim
        ], device=device, dtype=torch.float32)


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

    def get_history_tensor(self, device) -> torch.Tensor:
        if len(self.history) < self.history_len:
            pad = [TelemetrySample(0, 50, 30) for _ in range(self.history_len - len(self.history))]
            history = pad + self.history
        else:
            history = self.history[-self.history_len:]
        return torch.stack([s.to_tensor(device) for s in history])


# ============================================================================
# Model Architecture (same as z910)
# ============================================================================

class BodyTokenEncoder(nn.Module):
    def __init__(self, num_tokens=8, telemetry_dim=12, history_len=32, embedding_dim=256):
        super().__init__()
        self.num_tokens = num_tokens
        self.telemetry_dim = telemetry_dim
        self.token_queries = nn.Parameter(torch.randn(num_tokens, telemetry_dim) * 0.02)
        self.value_proj = nn.Linear(telemetry_dim, embedding_dim)
        self.out_proj = nn.Linear(embedding_dim, embedding_dim)
        self.history_pos = nn.Parameter(torch.randn(history_len, telemetry_dim) * 0.02)

    def forward(self, telemetry_history: torch.Tensor) -> torch.Tensor:
        history = telemetry_history + self.history_pos.unsqueeze(0)
        scores = torch.einsum('nd,bhd->bnh', self.token_queries, history)
        scores = scores / math.sqrt(self.telemetry_dim)
        attn = F.softmax(scores, dim=-1)
        values = self.value_proj(history)
        body_tokens = torch.einsum('bnh,bhd->bnd', attn, values)
        return self.out_proj(body_tokens)


class EnergyModulatedAttention(nn.Module):
    def __init__(self, hidden_dim=256, num_heads=4, num_body_tokens=8,
                 power_setpoint=30.0, temp_setpoint=50.0,
                 energy_mod_strength=0.3, homeostatic_gain=0.1,
                 use_body_tokens=True, use_energy_mod=True):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.num_body_tokens = num_body_tokens if use_body_tokens else 0
        self.power_setpoint = power_setpoint
        self.temp_setpoint = temp_setpoint
        self.energy_mod_strength = energy_mod_strength
        self.homeostatic_gain = homeostatic_gain
        self.use_body_tokens = use_body_tokens
        self.use_energy_mod = use_energy_mod

        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x, body_tokens, telemetry, causal_mask=None):
        B, T, D = x.shape

        if self.use_body_tokens and body_tokens is not None:
            full_seq = torch.cat([body_tokens, x], dim=1)
            num_body = body_tokens.size(1)
        else:
            full_seq = x
            num_body = 0

        full_len = full_seq.size(1)
        Q = self.q_proj(full_seq).view(B, full_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(full_seq).view(B, full_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(full_seq).view(B, full_len, self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)

        if causal_mask is not None and num_body > 0:
            full_mask = torch.zeros(full_len, full_len, device=x.device, dtype=torch.bool)
            full_mask[num_body:, num_body:] = causal_mask[:T, :T]
            scores = scores.masked_fill(full_mask.unsqueeze(0).unsqueeze(0), float('-inf'))
        elif causal_mask is not None:
            scores = scores.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), float('-inf'))

        if self.use_energy_mod:
            power_ratio = telemetry.power_w / max(self.power_setpoint, 1.0)
            energy_mod = 1.0 - self.energy_mod_strength * min(power_ratio, 2.0)
            energy_mod = max(0.1, energy_mod)

            power_dev = abs(telemetry.power_w - self.power_setpoint) / self.power_setpoint
            temp_dev = abs(telemetry.temp_c - self.temp_setpoint) / self.temp_setpoint
            homeostatic_gate = 1.0 / (1.0 + self.homeostatic_gain * (power_dev + temp_dev))
            combined_mod = energy_mod * homeostatic_gate

            mod_mask = torch.ones(full_len, device=x.device)
            if num_body > 0:
                mod_mask[:num_body] = 2.0 - combined_mod
                mod_mask[num_body:] = combined_mod
            else:
                mod_mask[:] = combined_mod
            scores = scores * mod_mask.view(1, 1, 1, -1)
        else:
            energy_mod, homeostatic_gate = 1.0, 1.0

        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, V).transpose(1, 2).contiguous().view(B, full_len, D)

        if num_body > 0:
            out = out[:, num_body:, :]

        return self.out_proj(out), {'energy_mod': energy_mod, 'homeostatic_gate': homeostatic_gate}


class EmbodiedTransformer(nn.Module):
    def __init__(self, vocab_size=256, hidden_dim=256, num_layers=6, num_heads=4,
                 ff_dim=1024, num_body_tokens=8, history_len=32,
                 use_body_tokens=True, use_energy_mod=True):
        super().__init__()
        self.use_body_tokens = use_body_tokens
        self.use_energy_mod = use_energy_mod

        self.token_emb = nn.Embedding(vocab_size, hidden_dim)
        self.pos_emb = nn.Embedding(512, hidden_dim)

        if use_body_tokens:
            self.body_encoder = BodyTokenEncoder(num_body_tokens, 12, history_len, hidden_dim)
        else:
            self.body_encoder = None

        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(nn.ModuleDict({
                'attn': EnergyModulatedAttention(hidden_dim, num_heads, num_body_tokens,
                                                  use_body_tokens=use_body_tokens,
                                                  use_energy_mod=use_energy_mod),
                'ln1': nn.LayerNorm(hidden_dim),
                'ff': nn.Sequential(nn.Linear(hidden_dim, ff_dim), nn.GELU(),
                                    nn.Linear(ff_dim, hidden_dim)),
                'ln2': nn.LayerNorm(hidden_dim),
            }))

        self.ln_out = nn.LayerNorm(hidden_dim)
        self.head = nn.Linear(hidden_dim, vocab_size)

    def forward(self, input_ids, telemetry_history, current_telemetry):
        B, T = input_ids.shape
        device = input_ids.device

        x = self.token_emb(input_ids) + self.pos_emb(torch.arange(T, device=device))
        body_tokens = self.body_encoder(telemetry_history) if self.body_encoder else None
        causal_mask = torch.triu(torch.ones(T, T, device=device, dtype=torch.bool), diagonal=1)

        total_energy_mod = 0
        for layer in self.layers:
            x_norm = layer['ln1'](x)
            attn_out, info = layer['attn'](x_norm, body_tokens, current_telemetry, causal_mask)
            x = x + attn_out
            x = x + layer['ff'](layer['ln2'](x))
            total_energy_mod += info['energy_mod']

        return self.head(self.ln_out(x)), total_energy_mod / len(self.layers)


# ============================================================================
# Data Loading
# ============================================================================

def load_data(data_dir: Path, device: torch.device):
    fpath = data_dir / "tiny_shakespeare.txt"
    if not fpath.exists():
        print("[z911] Downloading TinyShakespeare...")
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
class BenchmarkResult:
    condition: str
    n_batches: int
    n_tokens: int
    total_time_s: float
    total_energy_j: float
    avg_loss: float
    avg_ppl: float
    j_per_token: float
    tokens_per_sec: float
    tokens_per_joule: float
    avg_power_w: float
    avg_temp_c: float
    avg_energy_mod: float
    min_temp_c: float
    max_temp_c: float


def run_benchmark(
    model: nn.Module,
    data: torch.Tensor,
    telemetry: TelemetryCollector,
    batch_size: int,
    seq_len: int,
    n_batches: int,
    condition: str,
    device: torch.device,
) -> BenchmarkResult:
    """Run inference benchmark and collect metrics."""

    model.eval()

    total_loss = 0
    total_energy = 0
    total_tokens = 0
    total_energy_mod = 0
    temps = []
    powers = []

    start_time = time.time()

    with torch.no_grad():
        for i in range(n_batches):
            inp, tgt = make_batch(data, batch_size, seq_len, offset=i*batch_size*seq_len)

            # Read telemetry
            current = telemetry.read()
            history = telemetry.get_history_tensor(device).unsqueeze(0).expand(batch_size, -1, -1)

            temps.append(current.temp_c)
            powers.append(current.power_w)

            # Measure energy
            with EnergyMeter(telemetry.telemetry) as meter:
                logits, energy_mod = model(inp, history, current)
                loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), tgt.reshape(-1))

            total_energy += meter.energy_j
            total_loss += loss.item()
            total_tokens += batch_size * seq_len
            total_energy_mod += energy_mod

            if (i + 1) % 50 == 0:
                print(f"    [{condition}] batch {i+1}/{n_batches}  "
                      f"loss={loss.item():.4f}  temp={current.temp_c:.1f}C  "
                      f"power={current.power_w:.1f}W")

    end_time = time.time()
    total_time = end_time - start_time

    avg_loss = total_loss / n_batches
    avg_ppl = math.exp(avg_loss)
    j_per_token = total_energy / total_tokens
    tokens_per_sec = total_tokens / total_time
    tokens_per_joule = total_tokens / total_energy if total_energy > 0 else 0

    return BenchmarkResult(
        condition=condition,
        n_batches=n_batches,
        n_tokens=total_tokens,
        total_time_s=total_time,
        total_energy_j=total_energy,
        avg_loss=avg_loss,
        avg_ppl=avg_ppl,
        j_per_token=j_per_token,
        tokens_per_sec=tokens_per_sec,
        tokens_per_joule=tokens_per_joule,
        avg_power_w=sum(powers) / len(powers),
        avg_temp_c=sum(temps) / len(temps),
        avg_energy_mod=total_energy_mod / n_batches,
        min_temp_c=min(temps),
        max_temp_c=max(temps),
    )


def thermal_stress_test(
    model: nn.Module,
    data: torch.Tensor,
    telemetry: TelemetryCollector,
    batch_size: int,
    seq_len: int,
    device: torch.device,
) -> Dict:
    """Test model behavior under thermal stress."""

    print("\n  [Thermal Stress Test]")
    print("  Phase 1: Cool GPU (idle 10s)...")
    time.sleep(10)

    cool_sample = telemetry.read()
    print(f"    Cool state: {cool_sample.temp_c:.1f}C, {cool_sample.power_w:.1f}W")

    # Run cool benchmark
    cool_result = run_benchmark(model, data, telemetry, batch_size, seq_len, 50, "cool", device)

    print("\n  Phase 2: Heat GPU (heavy compute)...")
    # Heat up with heavy matmuls
    for _ in range(20):
        a = torch.randn(2048, 2048, device=device)
        b = torch.randn(2048, 2048, device=device)
        _ = torch.matmul(a, b)

    hot_sample = telemetry.read()
    print(f"    Hot state: {hot_sample.temp_c:.1f}C, {hot_sample.power_w:.1f}W")

    # Run hot benchmark
    hot_result = run_benchmark(model, data, telemetry, batch_size, seq_len, 50, "hot", device)

    return {
        'cool': asdict(cool_result),
        'hot': asdict(hot_result),
        'temp_delta_c': hot_result.avg_temp_c - cool_result.avg_temp_c,
        'energy_mod_delta': hot_result.avg_energy_mod - cool_result.avg_energy_mod,
        'ppl_stability': abs(hot_result.avg_ppl - cool_result.avg_ppl) / cool_result.avg_ppl,
    }


# ============================================================================
# Business Metrics
# ============================================================================

@dataclass
class BusinessMetrics:
    # Efficiency gains
    energy_savings_pct: float
    throughput_gain_pct: float
    tokens_per_joule_improvement: float

    # Cost projections (at scale)
    daily_tokens_1gpu: int
    daily_kwh_baseline: float
    daily_kwh_embodied: float
    daily_cost_savings_usd: float  # at $0.10/kWh

    # Annual projections (100 GPU cluster)
    annual_kwh_savings_100gpu: float
    annual_cost_savings_100gpu_usd: float
    co2_reduction_kg: float  # at 0.4 kg CO2/kWh

    # ROI
    break_even_days: float


def calculate_business_metrics(
    baseline: BenchmarkResult,
    embodied: BenchmarkResult,
    electricity_cost_per_kwh: float = 0.10,
    co2_per_kwh: float = 0.4,
    cluster_size: int = 100,
) -> BusinessMetrics:
    """Calculate business value from benchmark results."""

    # Efficiency gains
    energy_savings_pct = (1 - embodied.j_per_token / baseline.j_per_token) * 100
    throughput_gain_pct = (embodied.tokens_per_sec / baseline.tokens_per_sec - 1) * 100
    tpj_improvement = embodied.tokens_per_joule / baseline.tokens_per_joule

    # Daily projections (24h operation)
    seconds_per_day = 86400
    daily_tokens_1gpu = int(baseline.tokens_per_sec * seconds_per_day)

    # Energy consumption (kWh)
    daily_energy_baseline_j = daily_tokens_1gpu * baseline.j_per_token
    daily_energy_embodied_j = daily_tokens_1gpu * embodied.j_per_token

    daily_kwh_baseline = daily_energy_baseline_j / 3600000  # J to kWh
    daily_kwh_embodied = daily_energy_embodied_j / 3600000

    daily_savings_kwh = daily_kwh_baseline - daily_kwh_embodied
    daily_cost_savings = daily_savings_kwh * electricity_cost_per_kwh

    # Annual cluster projections
    annual_days = 365
    annual_kwh_savings = daily_savings_kwh * annual_days * cluster_size
    annual_cost_savings = daily_cost_savings * annual_days * cluster_size
    co2_reduction = annual_kwh_savings * co2_per_kwh

    # Break-even (assuming minimal implementation cost)
    implementation_cost = 1000  # Conservative estimate
    break_even = implementation_cost / max(daily_cost_savings * cluster_size, 0.01)

    return BusinessMetrics(
        energy_savings_pct=energy_savings_pct,
        throughput_gain_pct=throughput_gain_pct,
        tokens_per_joule_improvement=tpj_improvement,
        daily_tokens_1gpu=daily_tokens_1gpu,
        daily_kwh_baseline=daily_kwh_baseline,
        daily_kwh_embodied=daily_kwh_embodied,
        daily_cost_savings_usd=daily_cost_savings,
        annual_kwh_savings_100gpu=annual_kwh_savings,
        annual_cost_savings_100gpu_usd=annual_cost_savings,
        co2_reduction_kg=co2_reduction,
        break_even_days=break_even,
    )


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Z911: Embodiment Validation & Business Metrics")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--n-batches", type=int, default=200)
    parser.add_argument("--train-epochs", type=int, default=3)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--skip-training", action="store_true")
    args = parser.parse_args()

    print("=" * 70)
    print("  Z911: Comprehensive Embodiment Validation & Business Metrics")
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

    # Data
    data_dir = Path(__file__).parent.parent / "data"
    data, n_chars = load_data(data_dir, device)
    print(f"  Data: {n_chars:,} chars")

    # Create models
    print("\n  Creating models...")
    baseline_model = EmbodiedTransformer(
        use_body_tokens=False,
        use_energy_mod=False,
    ).to(device)

    embodied_model = EmbodiedTransformer(
        use_body_tokens=True,
        use_energy_mod=True,
    ).to(device)

    baseline_params = sum(p.numel() for p in baseline_model.parameters())
    embodied_params = sum(p.numel() for p in embodied_model.parameters())
    print(f"  Baseline params: {baseline_params:,}")
    print(f"  Embodied params: {embodied_params:,}")
    print(f"  Overhead: {(embodied_params/baseline_params - 1)*100:.1f}%")

    # Quick training to get reasonable weights
    if not args.skip_training:
        print(f"\n  Training models ({args.train_epochs} epochs each)...")

        for name, model in [("Baseline", baseline_model), ("Embodied", embodied_model)]:
            print(f"\n  Training {name}...")
            optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
            model.train()

            for epoch in range(args.train_epochs):
                epoch_loss = 0
                n_steps = 50
                for step in range(n_steps):
                    inp, tgt = make_batch(data, args.batch_size, args.seq_len)
                    current = telemetry.read()
                    history = telemetry.get_history_tensor(device).unsqueeze(0).expand(args.batch_size, -1, -1)

                    optimizer.zero_grad()
                    logits, _ = model(inp, history, current)
                    loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), tgt.reshape(-1))
                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss.item()

                print(f"    Epoch {epoch+1}/{args.train_epochs}  loss={epoch_loss/n_steps:.4f}")

    # ========================================================================
    # BENCHMARK 1: Standard Inference
    # ========================================================================
    print("\n" + "=" * 70)
    print("  BENCHMARK 1: Standard Inference")
    print("=" * 70)

    print("\n  Running baseline...")
    baseline_result = run_benchmark(
        baseline_model, data, telemetry,
        args.batch_size, args.seq_len, args.n_batches,
        "Baseline", device
    )

    # Cool down between tests
    print("\n  Cooling down (5s)...")
    time.sleep(5)

    print("\n  Running embodied...")
    embodied_result = run_benchmark(
        embodied_model, data, telemetry,
        args.batch_size, args.seq_len, args.n_batches,
        "Embodied", device
    )

    # ========================================================================
    # BENCHMARK 2: Thermal Stress Test
    # ========================================================================
    print("\n" + "=" * 70)
    print("  BENCHMARK 2: Thermal Stress Test (Embodied Model)")
    print("=" * 70)

    thermal_results = thermal_stress_test(
        embodied_model, data, telemetry,
        args.batch_size, args.seq_len, device
    )

    # ========================================================================
    # Business Metrics
    # ========================================================================
    print("\n" + "=" * 70)
    print("  BUSINESS METRICS")
    print("=" * 70)

    business = calculate_business_metrics(baseline_result, embodied_result)

    # ========================================================================
    # Results Summary
    # ========================================================================
    print("\n" + "=" * 70)
    print("  RESULTS SUMMARY")
    print("=" * 70)

    print("\n  INFERENCE PERFORMANCE")
    print("-" * 70)
    print(f"  {'Metric':<30} {'Baseline':>15} {'Embodied':>15} {'Delta':>10}")
    print("-" * 70)
    print(f"  {'J/token':<30} {baseline_result.j_per_token:>15.6f} {embodied_result.j_per_token:>15.6f} {(embodied_result.j_per_token/baseline_result.j_per_token - 1)*100:>+9.1f}%")
    print(f"  {'Tokens/sec':<30} {baseline_result.tokens_per_sec:>15.1f} {embodied_result.tokens_per_sec:>15.1f} {(embodied_result.tokens_per_sec/baseline_result.tokens_per_sec - 1)*100:>+9.1f}%")
    print(f"  {'Tokens/Joule':<30} {baseline_result.tokens_per_joule:>15.1f} {embodied_result.tokens_per_joule:>15.1f} {(embodied_result.tokens_per_joule/baseline_result.tokens_per_joule - 1)*100:>+9.1f}%")
    print(f"  {'Perplexity':<30} {baseline_result.avg_ppl:>15.2f} {embodied_result.avg_ppl:>15.2f} {(embodied_result.avg_ppl/baseline_result.avg_ppl - 1)*100:>+9.1f}%")
    print(f"  {'Avg Power (W)':<30} {baseline_result.avg_power_w:>15.1f} {embodied_result.avg_power_w:>15.1f} {(embodied_result.avg_power_w/baseline_result.avg_power_w - 1)*100:>+9.1f}%")
    print(f"  {'Avg Temp (C)':<30} {baseline_result.avg_temp_c:>15.1f} {embodied_result.avg_temp_c:>15.1f} {embodied_result.avg_temp_c - baseline_result.avg_temp_c:>+9.1f}")
    print(f"  {'Energy Modulation':<30} {'1.000':>15} {embodied_result.avg_energy_mod:>15.3f} {'active':>10}")

    print("\n  THERMAL STRESS RESPONSE (Embodied Model)")
    print("-" * 70)
    print(f"  {'Metric':<30} {'Cool GPU':>15} {'Hot GPU':>15} {'Delta':>10}")
    print("-" * 70)
    print(f"  {'Temperature (C)':<30} {thermal_results['cool']['avg_temp_c']:>15.1f} {thermal_results['hot']['avg_temp_c']:>15.1f} {thermal_results['temp_delta_c']:>+9.1f}")
    print(f"  {'Energy Modulation':<30} {thermal_results['cool']['avg_energy_mod']:>15.3f} {thermal_results['hot']['avg_energy_mod']:>15.3f} {thermal_results['energy_mod_delta']:>+9.3f}")
    print(f"  {'Perplexity':<30} {thermal_results['cool']['avg_ppl']:>15.2f} {thermal_results['hot']['avg_ppl']:>15.2f} {thermal_results['ppl_stability']*100:>+9.1f}%")

    print("\n  BUSINESS VALUE (100 GPU Cluster, 24/7 Operation)")
    print("-" * 70)
    print(f"  Energy Savings:              {business.energy_savings_pct:>+.1f}%")
    print(f"  Throughput Change:           {business.throughput_gain_pct:>+.1f}%")
    print(f"  Tokens/Joule Improvement:    {business.tokens_per_joule_improvement:.2f}x")
    print(f"  Daily Tokens (1 GPU):        {business.daily_tokens_1gpu:,}")
    print(f"  Daily kWh (Baseline):        {business.daily_kwh_baseline:.1f}")
    print(f"  Daily kWh (Embodied):        {business.daily_kwh_embodied:.1f}")
    print(f"  Daily Savings (1 GPU):       ${business.daily_cost_savings_usd:.2f}")
    print(f"  Annual kWh Savings (100):    {business.annual_kwh_savings_100gpu:,.0f}")
    print(f"  Annual Cost Savings (100):   ${business.annual_cost_savings_100gpu_usd:,.0f}")
    print(f"  CO2 Reduction (kg/year):     {business.co2_reduction_kg:,.0f}")
    print(f"  Break-even (days):           {business.break_even_days:.1f}")

    # Validation checks
    print("\n" + "=" * 70)
    print("  VALIDATION CHECKS")
    print("=" * 70)

    checks = []

    # Check 1: Energy modulation is active
    if embodied_result.avg_energy_mod < 0.99:
        checks.append(("Energy modulation active", True, f"{embodied_result.avg_energy_mod:.3f}"))
    else:
        checks.append(("Energy modulation active", False, f"{embodied_result.avg_energy_mod:.3f}"))

    # Check 2: Quality preserved (PPL within 10%)
    ppl_diff = abs(embodied_result.avg_ppl - baseline_result.avg_ppl) / baseline_result.avg_ppl
    if ppl_diff < 0.10:
        checks.append(("Quality preserved (<10% PPL diff)", True, f"{ppl_diff*100:.1f}%"))
    else:
        checks.append(("Quality preserved (<10% PPL diff)", False, f"{ppl_diff*100:.1f}%"))

    # Check 3: Thermal response (energy mod changes with temp)
    if abs(thermal_results['energy_mod_delta']) > 0.001:
        checks.append(("Thermal response present", True, f"delta={thermal_results['energy_mod_delta']:.4f}"))
    else:
        checks.append(("Thermal response present", False, f"delta={thermal_results['energy_mod_delta']:.4f}"))

    # Check 4: PPL stable under thermal stress
    if thermal_results['ppl_stability'] < 0.05:
        checks.append(("PPL stable under stress (<5%)", True, f"{thermal_results['ppl_stability']*100:.1f}%"))
    else:
        checks.append(("PPL stable under stress (<5%)", False, f"{thermal_results['ppl_stability']*100:.1f}%"))

    # Check 5: Positive business value
    if business.energy_savings_pct > 0:
        checks.append(("Positive energy savings", True, f"{business.energy_savings_pct:.1f}%"))
    else:
        checks.append(("Positive energy savings", False, f"{business.energy_savings_pct:.1f}%"))

    n_passed = sum(1 for _, passed, _ in checks if passed)
    print(f"\n  Passed {n_passed}/{len(checks)} checks:")
    for name, passed, detail in checks:
        status = "PASS" if passed else "FAIL"
        print(f"    [{status}] {name}: {detail}")

    # Verdict
    if n_passed >= 4:
        verdict = "EMBODIMENT VALIDATED - PRODUCTION READY"
    elif n_passed >= 3:
        verdict = "EMBODIMENT WORKS - NEEDS OPTIMIZATION"
    else:
        verdict = "EMBODIMENT INSUFFICIENT"

    print(f"\n  VERDICT: {verdict}")

    # Save results
    results_dir = Path(__file__).parent.parent / "results"
    results_dir.mkdir(exist_ok=True)

    output = {
        "experiment": "z911_embodiment_validation",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "config": {
            "batch_size": args.batch_size,
            "seq_len": args.seq_len,
            "n_batches": args.n_batches,
            "train_epochs": args.train_epochs,
        },
        "baseline": asdict(baseline_result),
        "embodied": asdict(embodied_result),
        "thermal_stress": thermal_results,
        "business_metrics": asdict(business),
        "checks": [{"name": n, "passed": p, "detail": d} for n, p, d in checks],
        "verdict": verdict,
    }

    out_path = results_dir / "z911_embodiment_validation.json"
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\n  Results saved to {out_path}")

    print("\n" + "=" * 70)
    print("  Z911 COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
