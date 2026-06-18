#!/usr/bin/env python3
"""
Z912A: Dynamic Precision Adaptation Benchmark
==============================================

Implements telemetry-driven precision switching to validate the hypothesis:
    "Dynamic precision adaptation can achieve 2-4x energy reduction with <5% quality loss"

Precision Modes:
    - FP32: Full precision (quality baseline)
    - FP16: Half precision (standard mixed precision)
    - INT8: Simulated 8-bit quantization (attention only or FFN only)

Adaptation Policy (telemetry-driven):
    - If power > 40W OR temp > 55C: INT8 attention + FP16 FFN (aggressive savings)
    - If power > 30W OR temp > 50C: FP16 for both (balanced)
    - Else: FP32 for quality (cool/low-power state)

Benchmarks:
    1. Fixed FP32 baseline
    2. Fixed FP16 baseline
    3. Fixed INT8-sim baseline
    4. Embodied adaptive (telemetry-driven switching)

Metrics per condition:
    - J/token (energy efficiency)
    - Tokens/sec (throughput)
    - Perplexity (quality)
    - Precision distribution (how often each mode used)

Business Value:
    - Energy savings vs FP32 baseline
    - Quality preservation (PPL delta)
    - Cost projections at scale

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
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.telemetry.sysfs_hwmon import SysfsHwmonTelemetry, EnergyMeter


# ============================================================================
# Constants & Thresholds
# ============================================================================

# Precision adaptation thresholds
POWER_HIGH_THRESHOLD = 40.0   # W - switch to aggressive savings
POWER_MED_THRESHOLD = 30.0    # W - switch to balanced mode
TEMP_HIGH_THRESHOLD = 55.0    # C
TEMP_MED_THRESHOLD = 50.0     # C

# Precision modes
PRECISION_FP32 = "fp32"
PRECISION_FP16 = "fp16"
PRECISION_INT8_ATT = "int8_att"  # INT8 attention, FP16 FFN


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


# ============================================================================
# Precision Adaptation Policy
# ============================================================================

class PrecisionPolicy:
    """Telemetry-driven precision selection policy."""

    def __init__(
        self,
        power_high: float = POWER_HIGH_THRESHOLD,
        power_med: float = POWER_MED_THRESHOLD,
        temp_high: float = TEMP_HIGH_THRESHOLD,
        temp_med: float = TEMP_MED_THRESHOLD,
    ):
        self.power_high = power_high
        self.power_med = power_med
        self.temp_high = temp_high
        self.temp_med = temp_med

        # Tracking
        self.precision_counts = defaultdict(int)
        self.decision_history: List[Tuple[str, float, float]] = []

    def decide(self, telemetry: TelemetrySample) -> str:
        """
        Decide precision mode based on current telemetry.

        Returns:
            Precision mode: 'fp32', 'fp16', or 'int8_att'
        """
        power = telemetry.power_w
        temp = telemetry.temp_c

        # Decision logic
        if power > self.power_high or temp > self.temp_high:
            # Aggressive savings: INT8 attention, FP16 FFN
            mode = PRECISION_INT8_ATT
        elif power > self.power_med or temp > self.temp_med:
            # Balanced: FP16 for both
            mode = PRECISION_FP16
        else:
            # Quality mode: FP32
            mode = PRECISION_FP32

        self.precision_counts[mode] += 1
        self.decision_history.append((mode, power, temp))
        return mode

    def get_distribution(self) -> Dict[str, float]:
        """Get precision mode distribution as percentages."""
        total = sum(self.precision_counts.values())
        if total == 0:
            return {PRECISION_FP32: 0, PRECISION_FP16: 0, PRECISION_INT8_ATT: 0}
        return {k: v / total * 100 for k, v in self.precision_counts.items()}

    def reset(self):
        self.precision_counts.clear()
        self.decision_history.clear()


# ============================================================================
# Precision-Switchable Attention
# ============================================================================

class PrecisionSwitchableAttention(nn.Module):
    """
    Attention module with runtime-switchable precision.

    Supports:
        - FP32: Full precision (highest quality)
        - FP16: Half precision (balanced)
        - INT8-sim: Simulated 8-bit quantization via aggressive rounding
    """

    def __init__(self, hidden_dim: int, num_heads: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

        self.current_precision = PRECISION_FP32

    def set_precision(self, precision: str):
        """Set the precision mode for forward pass."""
        self.current_precision = precision

    def _quantize_int8_sim(self, x: torch.Tensor) -> torch.Tensor:
        """
        Simulate INT8 quantization by scaling to [-128, 127] range and back.
        This approximates the behavior of true INT8 without requiring special kernels.
        """
        # Find scale to fit in INT8 range
        max_val = x.abs().max() + 1e-8
        scale = 127.0 / max_val

        # Quantize and dequantize
        x_int = torch.round(x * scale).clamp(-128, 127)
        x_dequant = x_int / scale
        return x_dequant

    def forward(self, x: torch.Tensor, causal_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, T, D = x.shape

        # Choose precision
        if self.current_precision == PRECISION_FP32:
            use_autocast = False
            use_int8_sim = False
        elif self.current_precision == PRECISION_FP16:
            use_autocast = True
            use_int8_sim = False
        else:  # INT8 simulation
            use_autocast = True  # Base in FP16, but quantize activations
            use_int8_sim = True

        # Store original dtype
        orig_dtype = x.dtype
        device_type = 'cuda' if x.is_cuda else 'cpu'

        # Use autocast for FP16/INT8 modes
        with torch.amp.autocast(device_type=device_type, enabled=use_autocast, dtype=torch.float16):
            # Project Q, K, V
            Q = self.q_proj(x)
            K = self.k_proj(x)
            V = self.v_proj(x)

            # Apply INT8 simulation to attention matrices
            if use_int8_sim:
                Q = self._quantize_int8_sim(Q)
                K = self._quantize_int8_sim(K)

            # Reshape for multi-head attention
            Q = Q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
            K = K.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
            V = V.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

            # Attention scores
            scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)

            if causal_mask is not None:
                scores = scores.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), float('-inf'))

            attn = F.softmax(scores, dim=-1)

            # Apply INT8 simulation to attention weights
            if use_int8_sim:
                attn = self._quantize_int8_sim(attn)

            # Output
            out = torch.matmul(attn, V)
            out = out.transpose(1, 2).contiguous().view(B, T, D)
            out = self.out_proj(out)

        # Cast back to original dtype
        return out.to(orig_dtype)


# ============================================================================
# Precision-Switchable FFN
# ============================================================================

class PrecisionSwitchableFFN(nn.Module):
    """Feed-forward network with runtime-switchable precision."""

    def __init__(self, hidden_dim: int, ff_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(hidden_dim, ff_dim)
        self.fc2 = nn.Linear(ff_dim, hidden_dim)
        self.current_precision = PRECISION_FP32

    def set_precision(self, precision: str):
        self.current_precision = precision

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Choose precision (FFN uses FP16 even in INT8_ATT mode)
        use_autocast = self.current_precision != PRECISION_FP32

        orig_dtype = x.dtype
        device_type = 'cuda' if x.is_cuda else 'cpu'

        with torch.amp.autocast(device_type=device_type, enabled=use_autocast, dtype=torch.float16):
            out = self.fc2(F.gelu(self.fc1(x)))
        return out.to(orig_dtype)


# ============================================================================
# Precision-Adaptive Transformer
# ============================================================================

class PrecisionAdaptiveTransformer(nn.Module):
    """
    Transformer with telemetry-driven dynamic precision adaptation.

    Each layer can independently switch between FP32/FP16/INT8-sim based
    on real-time GPU telemetry.
    """

    def __init__(
        self,
        vocab_size: int = 256,
        hidden_dim: int = 256,
        num_layers: int = 6,
        num_heads: int = 4,
        ff_dim: int = 1024,
        fixed_precision: Optional[str] = None,  # None = adaptive
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.fixed_precision = fixed_precision

        self.token_emb = nn.Embedding(vocab_size, hidden_dim)
        self.pos_emb = nn.Embedding(512, hidden_dim)

        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(nn.ModuleDict({
                'attn': PrecisionSwitchableAttention(hidden_dim, num_heads),
                'ffn': PrecisionSwitchableFFN(hidden_dim, ff_dim),
                'ln1': nn.LayerNorm(hidden_dim),
                'ln2': nn.LayerNorm(hidden_dim),
            }))

        self.ln_out = nn.LayerNorm(hidden_dim)
        self.head = nn.Linear(hidden_dim, vocab_size)

        # Precision policy (used in adaptive mode)
        self.policy = PrecisionPolicy()

    def set_all_precision(self, precision: str):
        """Set precision for all layers (used for fixed baseline modes)."""
        for layer in self.layers:
            layer['attn'].set_precision(precision)
            layer['ffn'].set_precision(precision)

    def forward(
        self,
        input_ids: torch.Tensor,
        telemetry: Optional[TelemetrySample] = None,
    ) -> Tuple[torch.Tensor, str]:
        """
        Forward pass with optional telemetry-driven precision adaptation.

        Returns:
            (logits, precision_used)
        """
        B, T = input_ids.shape
        device = input_ids.device

        # Determine precision
        if self.fixed_precision is not None:
            precision = self.fixed_precision
        elif telemetry is not None:
            precision = self.policy.decide(telemetry)
        else:
            precision = PRECISION_FP32

        # Set precision for all layers
        self.set_all_precision(precision)

        # Embeddings (always FP32)
        x = self.token_emb(input_ids) + self.pos_emb(torch.arange(T, device=device))

        # Causal mask
        causal_mask = torch.triu(torch.ones(T, T, device=device, dtype=torch.bool), diagonal=1)

        # Forward through layers
        for layer in self.layers:
            # Attention
            x_norm = layer['ln1'](x)
            attn_out = layer['attn'](x_norm, causal_mask)
            x = x + attn_out

            # FFN
            x_norm = layer['ln2'](x)
            ffn_out = layer['ffn'](x_norm)
            x = x + ffn_out

        # Output
        logits = self.head(self.ln_out(x))
        return logits, precision


# ============================================================================
# Data Loading
# ============================================================================

def load_data(data_dir: Path, device: torch.device) -> Tuple[torch.Tensor, int]:
    """Load TinyShakespeare dataset."""
    fpath = data_dir / "tiny_shakespeare.txt"
    if not fpath.exists():
        print("[z912a] Downloading TinyShakespeare...")
        import urllib.request
        url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
        fpath.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(url, str(fpath))

    text = fpath.read_text()
    data = torch.tensor([ord(c) % 256 for c in text], dtype=torch.long, device=device)
    return data, len(text)


def make_batch(data: torch.Tensor, batch_size: int, seq_len: int, offset: int = 0):
    """Create a batch of sequences."""
    starts = torch.randint(0, len(data) - seq_len - 1, (batch_size,)) + offset
    starts = starts % (len(data) - seq_len - 1)
    batch = torch.stack([data[s:s+seq_len+1] for s in starts])
    return batch[:, :-1], batch[:, 1:]


# ============================================================================
# Benchmark Results
# ============================================================================

@dataclass
class PrecisionBenchmarkResult:
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
    precision_distribution: Dict[str, float]


@dataclass
class BusinessMetrics:
    energy_savings_pct: float
    quality_loss_pct: float  # PPL increase
    throughput_change_pct: float
    tokens_per_joule_improvement: float
    daily_kwh_baseline: float
    daily_kwh_adaptive: float
    annual_savings_100gpu_usd: float
    co2_reduction_kg: float
    hypothesis_validated: bool  # 2-4x energy, <5% quality loss


# ============================================================================
# Benchmark Runner
# ============================================================================

def run_precision_benchmark(
    model: PrecisionAdaptiveTransformer,
    data: torch.Tensor,
    telemetry: TelemetryCollector,
    batch_size: int,
    seq_len: int,
    n_batches: int,
    condition: str,
    device: torch.device,
    use_adaptive: bool = False,
) -> PrecisionBenchmarkResult:
    """Run benchmark for a specific precision condition."""

    model.eval()

    # Reset policy tracking
    model.policy.reset()

    total_loss = 0.0
    total_energy = 0.0
    total_tokens = 0
    temps = []
    powers = []

    start_time = time.time()

    with torch.no_grad():
        for i in range(n_batches):
            inp, tgt = make_batch(data, batch_size, seq_len, offset=i * batch_size * seq_len)

            # Read telemetry
            current = telemetry.read()
            temps.append(current.temp_c)
            powers.append(current.power_w)

            # Measure energy
            with EnergyMeter(telemetry.telemetry) as meter:
                if use_adaptive:
                    logits, precision = model(inp, current)
                else:
                    logits, precision = model(inp, None)
                loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), tgt.reshape(-1))

            total_energy += meter.energy_j
            total_loss += loss.item()
            total_tokens += batch_size * seq_len

            if (i + 1) % 50 == 0:
                print(f"    [{condition}] batch {i+1}/{n_batches}  "
                      f"loss={loss.item():.4f}  temp={current.temp_c:.1f}C  "
                      f"power={current.power_w:.1f}W  precision={precision}")

    end_time = time.time()
    total_time = end_time - start_time

    avg_loss = total_loss / n_batches
    avg_ppl = math.exp(min(avg_loss, 20))  # Clamp to avoid overflow
    j_per_token = total_energy / total_tokens
    tokens_per_sec = total_tokens / total_time
    tokens_per_joule = total_tokens / total_energy if total_energy > 0 else 0

    return PrecisionBenchmarkResult(
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
        precision_distribution=model.policy.get_distribution(),
    )


def calculate_business_metrics(
    baseline: PrecisionBenchmarkResult,
    adaptive: PrecisionBenchmarkResult,
) -> BusinessMetrics:
    """Calculate business value metrics."""

    # Energy savings
    energy_savings_pct = (1 - adaptive.j_per_token / baseline.j_per_token) * 100

    # Quality loss (PPL increase)
    quality_loss_pct = (adaptive.avg_ppl / baseline.avg_ppl - 1) * 100

    # Throughput change
    throughput_change_pct = (adaptive.tokens_per_sec / baseline.tokens_per_sec - 1) * 100

    # Tokens per joule improvement
    tpj_improvement = adaptive.tokens_per_joule / baseline.tokens_per_joule

    # Daily projections (24h operation)
    seconds_per_day = 86400
    daily_tokens = baseline.tokens_per_sec * seconds_per_day

    daily_kwh_baseline = (daily_tokens * baseline.j_per_token) / 3600000
    daily_kwh_adaptive = (daily_tokens * adaptive.j_per_token) / 3600000

    daily_savings_kwh = daily_kwh_baseline - daily_kwh_adaptive

    # Annual cluster (100 GPU) projections
    annual_savings_kwh = daily_savings_kwh * 365 * 100
    annual_savings_usd = annual_savings_kwh * 0.10  # $0.10/kWh
    co2_reduction = annual_savings_kwh * 0.4  # 0.4 kg CO2/kWh

    # Hypothesis validation: 2-4x energy reduction with <5% quality loss
    energy_reduction_factor = baseline.j_per_token / adaptive.j_per_token if adaptive.j_per_token > 0 else 1.0
    hypothesis_validated = (energy_reduction_factor >= 2.0) and (quality_loss_pct < 5.0)

    return BusinessMetrics(
        energy_savings_pct=energy_savings_pct,
        quality_loss_pct=quality_loss_pct,
        throughput_change_pct=throughput_change_pct,
        tokens_per_joule_improvement=tpj_improvement,
        daily_kwh_baseline=daily_kwh_baseline,
        daily_kwh_adaptive=daily_kwh_adaptive,
        annual_savings_100gpu_usd=annual_savings_usd,
        co2_reduction_kg=co2_reduction,
        hypothesis_validated=hypothesis_validated,
    )


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Z912A: Dynamic Precision Adaptation Benchmark")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--n-batches", type=int, default=150, help="Batches per condition (100+ for statistical significance)")
    parser.add_argument("--train-steps", type=int, default=100, help="Training steps to initialize weights")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--skip-training", action="store_true")
    args = parser.parse_args()

    print("=" * 70)
    print("  Z912A: Dynamic Precision Adaptation Benchmark")
    print("  Hypothesis: 2-4x energy reduction with <5% quality loss")
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
    print(f"  Data: {n_chars:,} chars (TinyShakespeare)")

    # Create models for each condition
    print("\n  Creating models...")

    models = {
        "FP32_Fixed": PrecisionAdaptiveTransformer(fixed_precision=PRECISION_FP32).to(device),
        "FP16_Fixed": PrecisionAdaptiveTransformer(fixed_precision=PRECISION_FP16).to(device),
        "INT8_Fixed": PrecisionAdaptiveTransformer(fixed_precision=PRECISION_INT8_ATT).to(device),
        "Adaptive": PrecisionAdaptiveTransformer(fixed_precision=None).to(device),
    }

    # Share weights across models for fair comparison
    base_state = models["FP32_Fixed"].state_dict()
    for name, model in models.items():
        if name != "FP32_Fixed":
            model.load_state_dict(base_state)

    n_params = sum(p.numel() for p in models["FP32_Fixed"].parameters())
    print(f"  Model params: {n_params:,}")

    # Quick training
    if not args.skip_training:
        print(f"\n  Training model ({args.train_steps} steps)...")
        model = models["FP32_Fixed"]
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
        model.train()

        for step in range(args.train_steps):
            inp, tgt = make_batch(data, args.batch_size, args.seq_len)

            optimizer.zero_grad()
            logits, _ = model(inp)
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), tgt.reshape(-1))
            loss.backward()
            optimizer.step()

            if (step + 1) % 25 == 0:
                print(f"    Step {step+1}/{args.train_steps}  loss={loss.item():.4f}")

        # Copy trained weights to all models
        trained_state = model.state_dict()
        for name, m in models.items():
            if name != "FP32_Fixed":
                m.load_state_dict(trained_state)

    # ========================================================================
    # Run Benchmarks
    # ========================================================================

    results = {}

    print("\n" + "=" * 70)
    print("  BENCHMARK: Fixed Precision Baselines + Adaptive")
    print("=" * 70)

    conditions = [
        ("FP32_Fixed", False),
        ("FP16_Fixed", False),
        ("INT8_Fixed", False),
        ("Adaptive", True),
    ]

    for condition_name, use_adaptive in conditions:
        print(f"\n  Running {condition_name}...")

        # Cool down between conditions
        if condition_name != "FP32_Fixed":
            print("    Cooling down (3s)...")
            time.sleep(3)

        model = models[condition_name]
        result = run_precision_benchmark(
            model, data, telemetry,
            args.batch_size, args.seq_len, args.n_batches,
            condition_name, device,
            use_adaptive=use_adaptive,
        )
        results[condition_name] = result

    # ========================================================================
    # Calculate Business Metrics
    # ========================================================================

    fp32_baseline = results["FP32_Fixed"]
    adaptive_result = results["Adaptive"]
    business = calculate_business_metrics(fp32_baseline, adaptive_result)

    # ========================================================================
    # Results Summary
    # ========================================================================

    print("\n" + "=" * 70)
    print("  RESULTS SUMMARY")
    print("=" * 70)

    print("\n  PRECISION BENCHMARK RESULTS")
    print("-" * 80)
    print(f"  {'Condition':<15} {'J/token':>12} {'tok/s':>10} {'PPL':>10} {'Avg Power':>10} {'Avg Temp':>10}")
    print("-" * 80)

    for name, result in results.items():
        print(f"  {name:<15} {result.j_per_token:>12.6f} {result.tokens_per_sec:>10.1f} "
              f"{result.avg_ppl:>10.2f} {result.avg_power_w:>9.1f}W {result.avg_temp_c:>9.1f}C")

    print("\n  PRECISION DISTRIBUTION (Adaptive)")
    print("-" * 40)
    dist = results["Adaptive"].precision_distribution
    print(f"    FP32:     {dist.get(PRECISION_FP32, 0):>6.1f}%")
    print(f"    FP16:     {dist.get(PRECISION_FP16, 0):>6.1f}%")
    print(f"    INT8_ATT: {dist.get(PRECISION_INT8_ATT, 0):>6.1f}%")

    print("\n  COMPARISON VS FP32 BASELINE")
    print("-" * 70)
    print(f"  {'Condition':<15} {'Energy Savings':>15} {'PPL Delta':>12} {'Throughput':>12}")
    print("-" * 70)

    for name, result in results.items():
        if name == "FP32_Fixed":
            continue
        energy_savings = (1 - result.j_per_token / fp32_baseline.j_per_token) * 100
        ppl_delta = (result.avg_ppl / fp32_baseline.avg_ppl - 1) * 100
        throughput_delta = (result.tokens_per_sec / fp32_baseline.tokens_per_sec - 1) * 100
        print(f"  {name:<15} {energy_savings:>+14.1f}% {ppl_delta:>+11.1f}% {throughput_delta:>+11.1f}%")

    print("\n  BUSINESS VALUE (Adaptive vs FP32, 100 GPU Cluster)")
    print("-" * 60)
    print(f"  Energy Savings:              {business.energy_savings_pct:>+.1f}%")
    print(f"  Quality Loss (PPL increase): {business.quality_loss_pct:>+.1f}%")
    print(f"  Throughput Change:           {business.throughput_change_pct:>+.1f}%")
    print(f"  Tokens/Joule Improvement:    {business.tokens_per_joule_improvement:.2f}x")
    print(f"  Daily kWh (Baseline):        {business.daily_kwh_baseline:.1f}")
    print(f"  Daily kWh (Adaptive):        {business.daily_kwh_adaptive:.1f}")
    print(f"  Annual Savings (100 GPU):    ${business.annual_savings_100gpu_usd:,.0f}")
    print(f"  CO2 Reduction (kg/year):     {business.co2_reduction_kg:,.0f}")

    # Hypothesis validation
    print("\n  HYPOTHESIS VALIDATION")
    print("-" * 60)
    energy_reduction = fp32_baseline.j_per_token / adaptive_result.j_per_token if adaptive_result.j_per_token > 0 else 1.0
    print(f"  Energy Reduction Factor:     {energy_reduction:.2f}x (target: 2-4x)")
    print(f"  Quality Loss:                {business.quality_loss_pct:.2f}% (target: <5%)")

    if business.hypothesis_validated:
        verdict = "HYPOTHESIS VALIDATED - 2-4x energy reduction with <5% quality loss achieved!"
    elif energy_reduction >= 1.5:
        verdict = "PARTIAL SUCCESS - Significant energy reduction, but below 2x target"
    else:
        verdict = "HYPOTHESIS NOT VALIDATED - Insufficient energy reduction"

    print(f"\n  VERDICT: {verdict}")

    # ========================================================================
    # Save Results
    # ========================================================================

    results_dir = Path(__file__).parent.parent / "results"
    results_dir.mkdir(exist_ok=True)

    output = {
        "experiment": "z912a_precision_adaptation",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "config": {
            "batch_size": args.batch_size,
            "seq_len": args.seq_len,
            "n_batches": args.n_batches,
            "train_steps": args.train_steps,
            "thresholds": {
                "power_high": POWER_HIGH_THRESHOLD,
                "power_med": POWER_MED_THRESHOLD,
                "temp_high": TEMP_HIGH_THRESHOLD,
                "temp_med": TEMP_MED_THRESHOLD,
            }
        },
        "results": {name: asdict(result) for name, result in results.items()},
        "business_metrics": asdict(business),
        "hypothesis": {
            "target": "2-4x energy reduction with <5% quality loss",
            "energy_reduction_factor": energy_reduction,
            "quality_loss_pct": business.quality_loss_pct,
            "validated": business.hypothesis_validated,
        },
        "verdict": verdict,
    }

    out_path = results_dir / "z912a_precision_adaptation.json"
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\n  Results saved to {out_path}")

    print("\n" + "=" * 70)
    print("  Z912A COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
