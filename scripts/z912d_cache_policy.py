#!/usr/bin/env python3
"""
Z912D: Memory vs Recompute Tradeoff Benchmark - KV Cache Policy Optimization

This script implements and benchmarks adaptive KV cache policies based on
bottleneck detection (compute-bound vs memory-bound workloads).

Core Hypothesis:
- When memory-bound: recompute KV pairs (trade compute for memory bandwidth)
- When compute-bound: cache KV pairs (classic KV cache approach)
- Adaptive policy switches based on real-time telemetry

Bottleneck Detection Proxy:
- GPU utilization high + memory bandwidth saturated -> memory-bound
- GPU utilization low + memory available -> compute-bound
- We use gpu_busy_pct and vram_used as proxies

Target: Prove 20% energy savings by avoiding wrong bottleneck strategy.

Metrics:
1. J/token across different policies
2. Memory usage patterns
3. Compute utilization
4. Perplexity preservation
5. Business value calculation

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
from typing import List, Dict, Tuple, Optional, Literal
from collections import deque

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.telemetry.sysfs_hwmon import SysfsHwmonTelemetry, EnergyMeter


# ============================================================================
# Telemetry & Bottleneck Detection
# ============================================================================

@dataclass
class BottleneckState:
    """Current bottleneck classification based on telemetry."""
    timestamp: float
    gpu_busy_pct: float
    vram_used_gb: float
    vram_total_gb: float
    power_w: float
    temp_c: float

    # Derived
    memory_pressure: float = 0.0  # 0-1 scale
    compute_pressure: float = 0.0  # 0-1 scale
    bottleneck: str = "unknown"  # "memory", "compute", or "balanced"

    def __post_init__(self):
        # Memory pressure: how full is VRAM
        if self.vram_total_gb > 0:
            self.memory_pressure = self.vram_used_gb / self.vram_total_gb

        # Compute pressure: GPU utilization
        self.compute_pressure = self.gpu_busy_pct / 100.0

        # Bottleneck classification
        if self.memory_pressure > 0.8 and self.compute_pressure > 0.7:
            self.bottleneck = "memory"  # Memory is limiting
        elif self.compute_pressure < 0.5 and self.memory_pressure < 0.6:
            self.bottleneck = "compute"  # Compute is limiting
        else:
            self.bottleneck = "balanced"


class BottleneckDetector:
    """Detects compute vs memory bottleneck using sysfs telemetry."""

    def __init__(self, vram_total_gb: float = 16.0):
        self.telemetry = SysfsHwmonTelemetry(sample_rate_hz=100)
        self.vram_total_gb = vram_total_gb
        self.history: deque = deque(maxlen=32)

        # Thresholds for adaptive policy
        self.memory_threshold = 0.75  # Switch to recompute above this
        self.compute_threshold = 0.5   # Switch to cache below this

    def read(self) -> BottleneckState:
        """Read current bottleneck state."""
        sample = self.telemetry.read_sample()

        state = BottleneckState(
            timestamp=time.time(),
            gpu_busy_pct=sample.gpu_busy_pct,
            vram_used_gb=sample.vram_used_gb,
            vram_total_gb=self.vram_total_gb,
            power_w=sample.power_w,
            temp_c=sample.temp_edge_c,
        )

        self.history.append(state)
        return state

    def get_recommendation(self) -> Literal["cache", "recompute"]:
        """Get policy recommendation based on current state."""
        if not self.history:
            return "cache"  # Default to caching

        # Use average of recent samples for stability
        recent = list(self.history)[-8:]
        avg_memory_pressure = sum(s.memory_pressure for s in recent) / len(recent)
        avg_compute_pressure = sum(s.compute_pressure for s in recent) / len(recent)

        if avg_memory_pressure > self.memory_threshold:
            return "recompute"  # Memory-bound: trade compute for memory
        elif avg_compute_pressure < self.compute_threshold:
            return "cache"  # Compute-bound: use memory for caching
        else:
            # Balanced: prefer caching (lower latency)
            return "cache"


# ============================================================================
# KV Cache Implementation
# ============================================================================

class KVCache:
    """Key-Value cache for attention layers."""

    def __init__(self, max_seq_len: int, num_layers: int, num_heads: int,
                 head_dim: int, device: torch.device):
        self.max_seq_len = max_seq_len
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.device = device

        # Allocate cache
        self.k_cache = None
        self.v_cache = None
        self.seq_len = 0

    def allocate(self, batch_size: int):
        """Allocate cache memory."""
        self.k_cache = torch.zeros(
            self.num_layers, batch_size, self.num_heads,
            self.max_seq_len, self.head_dim,
            device=self.device, dtype=torch.float32
        )
        self.v_cache = torch.zeros_like(self.k_cache)
        self.seq_len = 0

    def update(self, layer_idx: int, k: torch.Tensor, v: torch.Tensor, pos: int):
        """Update cache at position."""
        seq_len = k.size(2)
        self.k_cache[layer_idx, :, :, pos:pos+seq_len, :] = k
        self.v_cache[layer_idx, :, :, pos:pos+seq_len, :] = v
        self.seq_len = max(self.seq_len, pos + seq_len)

    def get(self, layer_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get cached K, V up to current sequence length."""
        return (
            self.k_cache[layer_idx, :, :, :self.seq_len, :],
            self.v_cache[layer_idx, :, :, :self.seq_len, :]
        )

    def clear(self):
        """Clear cache."""
        if self.k_cache is not None:
            self.k_cache.zero_()
            self.v_cache.zero_()
        self.seq_len = 0

    def memory_bytes(self) -> int:
        """Return memory used by cache in bytes."""
        if self.k_cache is None:
            return 0
        return self.k_cache.numel() * 4 * 2  # float32, K + V


# ============================================================================
# Attention with Policy-Aware KV Handling
# ============================================================================

class PolicyAwareAttention(nn.Module):
    """Attention layer that can switch between caching and recomputing KV."""

    def __init__(self, hidden_dim: int, num_heads: int, layer_idx: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.layer_idx = layer_idx

        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

        # Track compute statistics
        self.last_flops = 0
        self.last_memory_access = 0

    def forward(
        self,
        x: torch.Tensor,
        kv_cache: Optional[KVCache] = None,
        policy: Literal["cache", "recompute"] = "cache",
        position: int = 0,
        causal_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Forward pass with policy-aware KV handling.

        Args:
            x: Input tensor [B, T, D]
            kv_cache: Optional KV cache
            policy: "cache" uses KV cache, "recompute" always recomputes
            position: Current position for cache update
            causal_mask: Causal attention mask

        Returns:
            output: Attention output
            info: Statistics dictionary
        """
        B, T, D = x.shape

        # Always compute Q
        Q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        info = {"policy": policy, "cache_hit": False, "recomputed": False}

        if policy == "cache" and kv_cache is not None and position > 0:
            # Use cached K, V for past positions
            K_past, V_past = kv_cache.get(self.layer_idx)

            # Compute new K, V for current position
            K_new = self.k_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
            V_new = self.v_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

            # Update cache
            kv_cache.update(self.layer_idx, K_new, V_new, position)

            # Concatenate past and new
            K = torch.cat([K_past, K_new], dim=2)
            V = torch.cat([V_past, V_new], dim=2)

            info["cache_hit"] = True
            self.last_memory_access = K_past.numel() * 4 * 2  # Read past K, V

        else:
            # Always recompute (or first position)
            K = self.k_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
            V = self.v_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

            if kv_cache is not None:
                kv_cache.update(self.layer_idx, K, V, position)

            info["recomputed"] = True
            self.last_flops = B * T * D * D * 2  # K and V projections

        # Attention computation
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)

        if causal_mask is not None:
            full_len = K.size(2)
            if causal_mask.size(0) < full_len:
                # Extend mask for cached positions
                extended_mask = torch.triu(
                    torch.ones(full_len, full_len, device=x.device, dtype=torch.bool),
                    diagonal=1
                )
                scores = scores.masked_fill(extended_mask.unsqueeze(0).unsqueeze(0), float('-inf'))
            else:
                scores = scores.masked_fill(causal_mask[:T, :K.size(2)].unsqueeze(0).unsqueeze(0), float('-inf'))

        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, V)
        out = out.transpose(1, 2).contiguous().view(B, T, D)

        return self.out_proj(out), info


# ============================================================================
# Transformer with Configurable Cache Policy
# ============================================================================

class CachePolicyTransformer(nn.Module):
    """Transformer that supports different KV cache policies."""

    def __init__(
        self,
        vocab_size: int = 256,
        hidden_dim: int = 256,
        num_layers: int = 6,
        num_heads: int = 4,
        ff_dim: int = 1024,
        max_seq_len: int = 512,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.max_seq_len = max_seq_len

        self.token_emb = nn.Embedding(vocab_size, hidden_dim)
        self.pos_emb = nn.Embedding(max_seq_len, hidden_dim)

        self.layers = nn.ModuleList()
        for i in range(num_layers):
            self.layers.append(nn.ModuleDict({
                'attn': PolicyAwareAttention(hidden_dim, num_heads, i),
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

    def create_kv_cache(self, batch_size: int, device: torch.device) -> KVCache:
        """Create a new KV cache."""
        cache = KVCache(
            self.max_seq_len, self.num_layers, self.num_heads,
            self.head_dim, device
        )
        cache.allocate(batch_size)
        return cache

    def forward(
        self,
        input_ids: torch.Tensor,
        kv_cache: Optional[KVCache] = None,
        policy: Literal["cache", "recompute", "adaptive"] = "cache",
        bottleneck_detector: Optional[BottleneckDetector] = None,
        position: int = 0,
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Forward pass with policy-aware KV handling.

        Args:
            input_ids: Input token IDs [B, T]
            kv_cache: Optional KV cache
            policy: "cache", "recompute", or "adaptive"
            bottleneck_detector: Required for adaptive policy
            position: Current position in sequence

        Returns:
            logits: Output logits
            info: Statistics dictionary
        """
        B, T = input_ids.shape
        device = input_ids.device

        # Embeddings
        positions = torch.arange(position, position + T, device=device)
        x = self.token_emb(input_ids) + self.pos_emb(positions)

        # Create causal mask
        causal_mask = torch.triu(
            torch.ones(T, T, device=device, dtype=torch.bool),
            diagonal=1
        )

        # Determine effective policy
        if policy == "adaptive" and bottleneck_detector is not None:
            effective_policy = bottleneck_detector.get_recommendation()
        else:
            effective_policy = policy if policy != "adaptive" else "cache"

        layer_infos = []
        for layer in self.layers:
            x_norm = layer['ln1'](x)
            attn_out, info = layer['attn'](
                x_norm, kv_cache, effective_policy, position, causal_mask
            )
            x = x + attn_out
            x = x + layer['ff'](layer['ln2'](x))
            layer_infos.append(info)

        logits = self.head(self.ln_out(x))

        return logits, {
            "effective_policy": effective_policy,
            "layer_infos": layer_infos,
            "cache_memory_bytes": kv_cache.memory_bytes() if kv_cache else 0,
        }


# ============================================================================
# Data Loading
# ============================================================================

def load_data(data_dir: Path, device: torch.device) -> Tuple[torch.Tensor, int]:
    """Load TinyShakespeare dataset."""
    candidates = [
        data_dir / "tiny_shakespeare.txt",
        data_dir / "tinyshakespeare" / "tiny_shakespeare.txt",
        data_dir / "tinyshakespeare.txt",
    ]

    fpath = None
    for path in candidates:
        if path.exists():
            fpath = path
            break

    if fpath is None:
        print("[z912d] Downloading TinyShakespeare...")
        import urllib.request
        url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
        fpath = data_dir / "tiny_shakespeare.txt"
        fpath.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(url, str(fpath))

    text = fpath.read_text()
    data = torch.tensor([ord(c) % 256 for c in text], dtype=torch.long, device=device)
    return data, len(text)


def make_batch(data: torch.Tensor, batch_size: int, seq_len: int, offset: int = 0):
    """Create a batch of sequences."""
    max_start = len(data) - seq_len - 1
    starts = torch.randint(0, max_start, (batch_size,)) + (offset % max_start)
    starts = starts % max_start
    batch = torch.stack([data[s:s+seq_len+1] for s in starts])
    return batch[:, :-1], batch[:, 1:]


# ============================================================================
# Benchmark Infrastructure
# ============================================================================

@dataclass
class PolicyBenchmarkResult:
    """Results from a policy benchmark run."""
    policy: str
    n_batches: int
    n_tokens: int
    total_time_s: float
    total_energy_j: float

    # Core metrics
    j_per_token: float
    tokens_per_sec: float
    tokens_per_joule: float

    # Quality
    avg_loss: float
    avg_ppl: float

    # Resource utilization
    avg_power_w: float
    avg_gpu_busy_pct: float
    avg_memory_pressure: float
    peak_vram_gb: float

    # Cache statistics
    cache_hits: int = 0
    recomputes: int = 0

    # Adaptive-specific
    policy_switches: int = 0
    time_in_cache_mode: float = 0.0
    time_in_recompute_mode: float = 0.0


def run_policy_benchmark(
    model: CachePolicyTransformer,
    data: torch.Tensor,
    detector: BottleneckDetector,
    policy: Literal["cache", "recompute", "adaptive"],
    batch_size: int,
    seq_len: int,
    n_batches: int,
    device: torch.device,
) -> PolicyBenchmarkResult:
    """Run benchmark with specific cache policy."""

    print(f"\n  Running [{policy.upper()}] policy benchmark...")
    model.eval()

    # Metrics accumulators
    total_loss = 0.0
    total_energy = 0.0
    total_tokens = 0
    cache_hits = 0
    recomputes = 0
    policy_switches = 0
    time_cache = 0.0
    time_recompute = 0.0

    powers = []
    gpu_busys = []
    memory_pressures = []
    peak_vram = 0.0

    last_effective_policy = None

    start_time = time.time()

    with torch.no_grad():
        for batch_idx in range(n_batches):
            inp, tgt = make_batch(data, batch_size, seq_len, offset=batch_idx)

            # Read bottleneck state
            state = detector.read()
            powers.append(state.power_w)
            gpu_busys.append(state.gpu_busy_pct)
            memory_pressures.append(state.memory_pressure)
            peak_vram = max(peak_vram, state.vram_used_gb)

            # Create fresh cache for this batch
            kv_cache = model.create_kv_cache(batch_size, device) if policy != "recompute" else None

            # Measure energy
            batch_start = time.time()
            with EnergyMeter(detector.telemetry) as meter:
                logits, info = model(
                    inp,
                    kv_cache=kv_cache,
                    policy=policy,
                    bottleneck_detector=detector,
                    position=0,
                )
                loss = F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)),
                    tgt.reshape(-1)
                )
            batch_time = time.time() - batch_start

            total_energy += meter.energy_j
            total_loss += loss.item()
            total_tokens += batch_size * seq_len

            # Track policy statistics
            effective = info.get("effective_policy", policy)
            if last_effective_policy is not None and effective != last_effective_policy:
                policy_switches += 1
            last_effective_policy = effective

            if effective == "cache":
                time_cache += batch_time
            else:
                time_recompute += batch_time

            # Count cache hits/recomputes from layer infos
            for layer_info in info.get("layer_infos", []):
                if layer_info.get("cache_hit"):
                    cache_hits += 1
                if layer_info.get("recomputed"):
                    recomputes += 1

            if (batch_idx + 1) % 50 == 0:
                print(f"    batch {batch_idx+1}/{n_batches}  "
                      f"loss={loss.item():.4f}  "
                      f"policy={effective}  "
                      f"gpu={state.gpu_busy_pct:.0f}%  "
                      f"mem={state.memory_pressure*100:.0f}%")

    end_time = time.time()
    total_time = end_time - start_time

    avg_loss = total_loss / n_batches

    return PolicyBenchmarkResult(
        policy=policy,
        n_batches=n_batches,
        n_tokens=total_tokens,
        total_time_s=total_time,
        total_energy_j=total_energy,
        j_per_token=total_energy / total_tokens,
        tokens_per_sec=total_tokens / total_time,
        tokens_per_joule=total_tokens / total_energy if total_energy > 0 else 0,
        avg_loss=avg_loss,
        avg_ppl=math.exp(avg_loss),
        avg_power_w=sum(powers) / len(powers),
        avg_gpu_busy_pct=sum(gpu_busys) / len(gpu_busys),
        avg_memory_pressure=sum(memory_pressures) / len(memory_pressures),
        peak_vram_gb=peak_vram,
        cache_hits=cache_hits,
        recomputes=recomputes,
        policy_switches=policy_switches,
        time_in_cache_mode=time_cache,
        time_in_recompute_mode=time_recompute,
    )


# ============================================================================
# Memory Stress Test
# ============================================================================

def run_memory_stress_test(
    model: CachePolicyTransformer,
    data: torch.Tensor,
    detector: BottleneckDetector,
    device: torch.device,
) -> Dict:
    """
    Run test under artificial memory pressure to validate adaptive policy.

    We allocate dummy tensors to create memory pressure, then see if
    adaptive policy correctly switches to recompute mode.
    """
    print("\n  [Memory Stress Test]")

    results = {}

    # Test 1: Low memory pressure
    print("  Phase 1: Low memory pressure...")
    torch.cuda.empty_cache()
    time.sleep(1)

    low_result = run_policy_benchmark(
        model, data, detector, "adaptive",
        batch_size=16, seq_len=64, n_batches=30, device=device
    )
    results["low_pressure"] = asdict(low_result)

    # Test 2: High memory pressure (allocate dummy tensors)
    print("\n  Phase 2: High memory pressure (artificial)...")

    # Allocate ~8GB of dummy tensors to create memory pressure
    dummy_tensors = []
    try:
        for i in range(8):
            dummy_tensors.append(torch.randn(256, 1024, 1024, device=device))
            torch.cuda.synchronize()
    except RuntimeError:
        print(f"    Allocated {len(dummy_tensors)} dummy tensors before OOM")

    time.sleep(1)
    state = detector.read()
    print(f"    Memory pressure: {state.memory_pressure*100:.1f}%")

    high_result = run_policy_benchmark(
        model, data, detector, "adaptive",
        batch_size=16, seq_len=64, n_batches=30, device=device
    )
    results["high_pressure"] = asdict(high_result)

    # Clean up
    del dummy_tensors
    torch.cuda.empty_cache()

    # Analysis
    results["analysis"] = {
        "low_recompute_ratio": low_result.time_in_recompute_mode / max(low_result.total_time_s, 0.001),
        "high_recompute_ratio": high_result.time_in_recompute_mode / max(high_result.total_time_s, 0.001),
        "policy_adaptation_detected": high_result.time_in_recompute_mode > low_result.time_in_recompute_mode,
        "energy_under_pressure_j_per_token": high_result.j_per_token,
        "energy_normal_j_per_token": low_result.j_per_token,
    }

    return results


# ============================================================================
# Business Value Calculation
# ============================================================================

@dataclass
class BusinessValue:
    """Business value metrics from cache policy optimization."""
    # Efficiency
    energy_savings_vs_always_cache_pct: float
    energy_savings_vs_always_recompute_pct: float
    optimal_policy_savings_pct: float

    # Throughput impact
    throughput_vs_cache_pct: float
    throughput_vs_recompute_pct: float

    # Quality preservation
    ppl_vs_cache_pct: float
    ppl_vs_recompute_pct: float

    # Daily projections (1 GPU, 24h)
    daily_tokens: int
    daily_kwh_cache: float
    daily_kwh_recompute: float
    daily_kwh_adaptive: float
    daily_savings_usd: float

    # Annual cluster projections (100 GPUs)
    annual_kwh_savings: float
    annual_cost_savings_usd: float
    co2_reduction_kg: float


def calculate_business_value(
    cache_result: PolicyBenchmarkResult,
    recompute_result: PolicyBenchmarkResult,
    adaptive_result: PolicyBenchmarkResult,
    electricity_cost_per_kwh: float = 0.10,
    co2_per_kwh: float = 0.4,
    cluster_size: int = 100,
) -> BusinessValue:
    """Calculate business value from policy comparison."""

    # Find best fixed policy
    best_fixed = cache_result if cache_result.j_per_token < recompute_result.j_per_token else recompute_result

    # Savings calculations
    energy_vs_cache = (1 - adaptive_result.j_per_token / cache_result.j_per_token) * 100
    energy_vs_recompute = (1 - adaptive_result.j_per_token / recompute_result.j_per_token) * 100
    optimal_savings = (1 - adaptive_result.j_per_token / best_fixed.j_per_token) * 100

    # Throughput changes
    throughput_vs_cache = (adaptive_result.tokens_per_sec / cache_result.tokens_per_sec - 1) * 100
    throughput_vs_recompute = (adaptive_result.tokens_per_sec / recompute_result.tokens_per_sec - 1) * 100

    # Quality preservation
    ppl_vs_cache = (adaptive_result.avg_ppl / cache_result.avg_ppl - 1) * 100
    ppl_vs_recompute = (adaptive_result.avg_ppl / recompute_result.avg_ppl - 1) * 100

    # Daily projections
    seconds_per_day = 86400
    daily_tokens = int(adaptive_result.tokens_per_sec * seconds_per_day)

    daily_energy_cache_j = daily_tokens * cache_result.j_per_token
    daily_energy_recompute_j = daily_tokens * recompute_result.j_per_token
    daily_energy_adaptive_j = daily_tokens * adaptive_result.j_per_token

    daily_kwh_cache = daily_energy_cache_j / 3600000
    daily_kwh_recompute = daily_energy_recompute_j / 3600000
    daily_kwh_adaptive = daily_energy_adaptive_j / 3600000

    # Use worst-case fixed policy as baseline for savings
    worst_fixed_kwh = max(daily_kwh_cache, daily_kwh_recompute)
    daily_savings_kwh = worst_fixed_kwh - daily_kwh_adaptive
    daily_savings_usd = daily_savings_kwh * electricity_cost_per_kwh

    # Annual cluster projections
    annual_days = 365
    annual_kwh_savings = daily_savings_kwh * annual_days * cluster_size
    annual_cost_savings = daily_savings_usd * annual_days * cluster_size
    co2_reduction = annual_kwh_savings * co2_per_kwh

    return BusinessValue(
        energy_savings_vs_always_cache_pct=energy_vs_cache,
        energy_savings_vs_always_recompute_pct=energy_vs_recompute,
        optimal_policy_savings_pct=optimal_savings,
        throughput_vs_cache_pct=throughput_vs_cache,
        throughput_vs_recompute_pct=throughput_vs_recompute,
        ppl_vs_cache_pct=ppl_vs_cache,
        ppl_vs_recompute_pct=ppl_vs_recompute,
        daily_tokens=daily_tokens,
        daily_kwh_cache=daily_kwh_cache,
        daily_kwh_recompute=daily_kwh_recompute,
        daily_kwh_adaptive=daily_kwh_adaptive,
        daily_savings_usd=daily_savings_usd,
        annual_kwh_savings=annual_kwh_savings,
        annual_cost_savings_usd=annual_cost_savings,
        co2_reduction_kg=co2_reduction,
    )


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Z912D: KV Cache Policy Benchmark")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--n-batches", type=int, default=100)
    parser.add_argument("--train-epochs", type=int, default=2)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--skip-training", action="store_true")
    parser.add_argument("--skip-stress-test", action="store_true")
    args = parser.parse_args()

    print("=" * 70)
    print("  Z912D: Memory vs Recompute Tradeoff Benchmark")
    print("  KV Cache Policy Optimization via Bottleneck Detection")
    print("=" * 70)

    # Device setup
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"\n  Device: {device}")

    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.1f} GB")

    # Initialize bottleneck detector
    vram_total = torch.cuda.get_device_properties(0).total_memory / (1024**3) if device.type == "cuda" else 16.0
    detector = BottleneckDetector(vram_total_gb=vram_total)

    state = detector.read()
    print(f"  Initial state: GPU={state.gpu_busy_pct:.0f}%, "
          f"VRAM={state.vram_used_gb:.1f}/{state.vram_total_gb:.0f}GB, "
          f"Power={state.power_w:.1f}W")

    # Load data
    data_dir = Path(__file__).parent.parent / "data"
    data, n_chars = load_data(data_dir, device)
    print(f"  Data: {n_chars:,} chars (TinyShakespeare)")

    # Create model
    print("\n  Creating model...")
    model = CachePolicyTransformer(
        vocab_size=256,
        hidden_dim=256,
        num_layers=6,
        num_heads=4,
        ff_dim=1024,
        max_seq_len=512,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {n_params:,}")

    # Quick training
    if not args.skip_training:
        print(f"\n  Training model ({args.train_epochs} epochs)...")
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
        model.train()

        for epoch in range(args.train_epochs):
            epoch_loss = 0.0
            n_steps = 50
            for step in range(n_steps):
                inp, tgt = make_batch(data, args.batch_size, args.seq_len)
                optimizer.zero_grad()
                logits, _ = model(inp, policy="cache")
                loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), tgt.reshape(-1))
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            print(f"    Epoch {epoch+1}/{args.train_epochs}  loss={epoch_loss/n_steps:.4f}")

    # ========================================================================
    # Benchmark: Compare Policies
    # ========================================================================
    print("\n" + "=" * 70)
    print("  POLICY COMPARISON BENCHMARK")
    print("=" * 70)

    results = {}

    # Benchmark 1: Always Cache
    torch.cuda.empty_cache()
    time.sleep(2)
    cache_result = run_policy_benchmark(
        model, data, detector, "cache",
        args.batch_size, args.seq_len, args.n_batches, device
    )
    results["always_cache"] = asdict(cache_result)

    # Benchmark 2: Always Recompute
    torch.cuda.empty_cache()
    time.sleep(2)
    recompute_result = run_policy_benchmark(
        model, data, detector, "recompute",
        args.batch_size, args.seq_len, args.n_batches, device
    )
    results["always_recompute"] = asdict(recompute_result)

    # Benchmark 3: Adaptive
    torch.cuda.empty_cache()
    time.sleep(2)
    adaptive_result = run_policy_benchmark(
        model, data, detector, "adaptive",
        args.batch_size, args.seq_len, args.n_batches, device
    )
    results["adaptive"] = asdict(adaptive_result)

    # ========================================================================
    # Memory Stress Test
    # ========================================================================
    if not args.skip_stress_test:
        print("\n" + "=" * 70)
        print("  MEMORY STRESS TEST")
        print("=" * 70)

        stress_results = run_memory_stress_test(model, data, detector, device)
        results["stress_test"] = stress_results

    # ========================================================================
    # Business Value Calculation
    # ========================================================================
    print("\n" + "=" * 70)
    print("  BUSINESS VALUE ANALYSIS")
    print("=" * 70)

    business = calculate_business_value(cache_result, recompute_result, adaptive_result)
    results["business_value"] = asdict(business)

    # ========================================================================
    # Results Summary
    # ========================================================================
    print("\n" + "=" * 70)
    print("  RESULTS SUMMARY")
    print("=" * 70)

    print("\n  ENERGY EFFICIENCY (J/token - lower is better)")
    print("-" * 70)
    print(f"  {'Policy':<20} {'J/token':>12} {'vs Best':>12} {'Tokens/J':>12}")
    print("-" * 70)

    best_j = min(cache_result.j_per_token, recompute_result.j_per_token, adaptive_result.j_per_token)
    for name, res in [("Always Cache", cache_result),
                      ("Always Recompute", recompute_result),
                      ("Adaptive", adaptive_result)]:
        delta = (res.j_per_token / best_j - 1) * 100
        marker = " <-- BEST" if res.j_per_token == best_j else ""
        print(f"  {name:<20} {res.j_per_token:>12.6f} {delta:>+11.1f}% {res.tokens_per_joule:>12.1f}{marker}")

    print("\n  THROUGHPUT (tokens/sec - higher is better)")
    print("-" * 70)
    best_tps = max(cache_result.tokens_per_sec, recompute_result.tokens_per_sec, adaptive_result.tokens_per_sec)
    for name, res in [("Always Cache", cache_result),
                      ("Always Recompute", recompute_result),
                      ("Adaptive", adaptive_result)]:
        delta = (res.tokens_per_sec / best_tps - 1) * 100
        print(f"  {name:<20} {res.tokens_per_sec:>12.1f} {delta:>+11.1f}%")

    print("\n  QUALITY (Perplexity - lower is better)")
    print("-" * 70)
    best_ppl = min(cache_result.avg_ppl, recompute_result.avg_ppl, adaptive_result.avg_ppl)
    for name, res in [("Always Cache", cache_result),
                      ("Always Recompute", recompute_result),
                      ("Adaptive", adaptive_result)]:
        delta = (res.avg_ppl / best_ppl - 1) * 100
        print(f"  {name:<20} {res.avg_ppl:>12.2f} {delta:>+11.1f}%")

    print("\n  RESOURCE UTILIZATION")
    print("-" * 70)
    print(f"  {'Policy':<20} {'Avg Power':>12} {'GPU Busy':>12} {'Mem Press':>12}")
    print("-" * 70)
    for name, res in [("Always Cache", cache_result),
                      ("Always Recompute", recompute_result),
                      ("Adaptive", adaptive_result)]:
        print(f"  {name:<20} {res.avg_power_w:>10.1f}W {res.avg_gpu_busy_pct:>10.0f}% {res.avg_memory_pressure*100:>10.0f}%")

    print("\n  ADAPTIVE POLICY DETAILS")
    print("-" * 70)
    print(f"  Policy switches: {adaptive_result.policy_switches}")
    print(f"  Time in cache mode: {adaptive_result.time_in_cache_mode:.1f}s ({adaptive_result.time_in_cache_mode/adaptive_result.total_time_s*100:.1f}%)")
    print(f"  Time in recompute mode: {adaptive_result.time_in_recompute_mode:.1f}s ({adaptive_result.time_in_recompute_mode/adaptive_result.total_time_s*100:.1f}%)")

    print("\n  BUSINESS VALUE (100 GPU Cluster, 24/7)")
    print("-" * 70)
    print(f"  Energy vs Always Cache:      {business.energy_savings_vs_always_cache_pct:>+.1f}%")
    print(f"  Energy vs Always Recompute:  {business.energy_savings_vs_always_recompute_pct:>+.1f}%")
    print(f"  Throughput vs Cache:         {business.throughput_vs_cache_pct:>+.1f}%")
    print(f"  PPL vs Cache:                {business.ppl_vs_cache_pct:>+.1f}%")
    print(f"  Daily kWh (Cache):           {business.daily_kwh_cache:.1f}")
    print(f"  Daily kWh (Recompute):       {business.daily_kwh_recompute:.1f}")
    print(f"  Daily kWh (Adaptive):        {business.daily_kwh_adaptive:.1f}")
    print(f"  Daily Savings (1 GPU):       ${business.daily_savings_usd:.2f}")
    print(f"  Annual Savings (100 GPUs):   ${business.annual_cost_savings_usd:,.0f}")
    print(f"  CO2 Reduction (kg/year):     {business.co2_reduction_kg:,.0f}")

    # ========================================================================
    # Validation Checks
    # ========================================================================
    print("\n" + "=" * 70)
    print("  VALIDATION CHECKS")
    print("=" * 70)

    checks = []

    # Check 1: Adaptive is better than worst fixed policy
    worst_fixed_j = max(cache_result.j_per_token, recompute_result.j_per_token)
    savings_vs_worst = (1 - adaptive_result.j_per_token / worst_fixed_j) * 100
    if savings_vs_worst > 0:
        checks.append(("Adaptive beats worst fixed policy", True, f"{savings_vs_worst:.1f}% savings"))
    else:
        checks.append(("Adaptive beats worst fixed policy", False, f"{savings_vs_worst:.1f}%"))

    # Check 2: Target 20% savings (vs wrong bottleneck)
    if savings_vs_worst >= 20:
        checks.append(("20% energy savings target", True, f"{savings_vs_worst:.1f}% achieved"))
    else:
        checks.append(("20% energy savings target", False, f"{savings_vs_worst:.1f}% (need 20%)"))

    # Check 3: Quality preserved
    max_ppl = max(cache_result.avg_ppl, recompute_result.avg_ppl)
    ppl_degradation = (adaptive_result.avg_ppl / max_ppl - 1) * 100
    if ppl_degradation < 5:
        checks.append(("Quality preserved (<5% PPL increase)", True, f"{ppl_degradation:.1f}%"))
    else:
        checks.append(("Quality preserved (<5% PPL increase)", False, f"{ppl_degradation:.1f}%"))

    # Check 4: Policy switching detected
    if adaptive_result.policy_switches > 0:
        checks.append(("Adaptive policy switching detected", True, f"{adaptive_result.policy_switches} switches"))
    else:
        checks.append(("Adaptive policy switching detected", False, "no switches"))

    # Check 5: Business value positive
    if business.annual_cost_savings_usd > 0:
        checks.append(("Positive business value", True, f"${business.annual_cost_savings_usd:,.0f}/year"))
    else:
        checks.append(("Positive business value", False, f"${business.annual_cost_savings_usd:,.0f}/year"))

    n_passed = sum(1 for _, passed, _ in checks if passed)
    print(f"\n  Passed {n_passed}/{len(checks)} checks:")
    for name, passed, detail in checks:
        status = "PASS" if passed else "FAIL"
        print(f"    [{status}] {name}: {detail}")

    # Verdict
    if n_passed >= 4:
        verdict = "CACHE POLICY OPTIMIZATION VALIDATED"
    elif n_passed >= 3:
        verdict = "PARTIAL SUCCESS - NEEDS TUNING"
    else:
        verdict = "INSUFFICIENT BENEFIT"

    print(f"\n  VERDICT: {verdict}")

    # ========================================================================
    # Save Results
    # ========================================================================
    results_dir = Path(__file__).parent.parent / "results"
    results_dir.mkdir(exist_ok=True)

    output = {
        "experiment": "z912d_cache_policy",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "hypothesis": "Adaptive KV cache policy saves energy by avoiding wrong bottleneck",
        "target": "20% energy savings vs worst fixed policy",
        "config": {
            "batch_size": args.batch_size,
            "seq_len": args.seq_len,
            "n_batches": args.n_batches,
            "train_epochs": args.train_epochs,
            "device": str(device),
        },
        "results": results,
        "checks": [{"name": n, "passed": p, "detail": d} for n, p, d in checks],
        "verdict": verdict,
    }

    out_path = results_dir / "z912d_cache_policy.json"
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\n  Results saved to {out_path}")

    print("\n" + "=" * 70)
    print("  Z912D COMPLETE")
    print("=" * 70)

    return n_passed >= 4


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
