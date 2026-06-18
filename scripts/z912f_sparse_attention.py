#!/usr/bin/env python3
"""
z912f_sparse_attention.py - Hardware-Gated Sparse Attention Benchmark

Tests whether hardware state (power/temp) can dynamically gate attention sparsity.
Compares 4 conditions:
1. Fixed dense (full attention)
2. Fixed sparse (k=64)
3. Embodied sparse (k varies with hardware state)
4. Random sparse (k varies randomly)

Architecture:
- Small transformer (6 layers, 256 hidden)
- Top-k attention selection
- Hardware-gated k values: k = min_k + (max_k - min_k) * (1 - power_normalized)
"""

import sys
import os
import json
import time
import math
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.telemetry.sysfs_hwmon import SysfsHwmonTelemetry, EnergyMeter


@dataclass
class AttentionMetrics:
    """Metrics for a single attention configuration."""
    condition: str
    total_energy_j: float
    total_tokens: int
    j_per_token: float
    perplexity: float
    avg_k: float
    min_k: int
    max_k: int
    k_std: float
    avg_power_w: float
    avg_temp_c: float
    inference_time_s: float
    tokens_per_sec: float


class SparseAttention(nn.Module):
    """Attention with top-k selection for sparsity."""

    def __init__(self, hidden_dim: int, num_heads: int, max_seq_len: int = 512):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.max_seq_len = max_seq_len

        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"

        self.qkv_proj = nn.Linear(hidden_dim, 3 * hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.scale = self.head_dim ** -0.5

        # For tracking k values
        self.last_k = None

    def forward(self, x: torch.Tensor, k: Optional[int] = None,
                causal_mask: bool = True) -> torch.Tensor:
        """
        Args:
            x: [batch, seq_len, hidden_dim]
            k: top-k selection (None = dense attention)
            causal_mask: apply causal masking for autoregressive modeling
        """
        batch_size, seq_len, _ = x.shape

        # Project to Q, K, V
        qkv = self.qkv_proj(x)  # [batch, seq_len, 3*hidden_dim]
        qkv = qkv.reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, batch, num_heads, seq_len, head_dim]
        q, k_proj, v = qkv[0], qkv[1], qkv[2]

        # Compute attention scores
        scores = torch.matmul(q, k_proj.transpose(-2, -1)) * self.scale  # [batch, num_heads, seq_len, seq_len]

        # Apply causal mask if needed
        if causal_mask:
            causal_mask_tensor = torch.triu(
                torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool),
                diagonal=1
            )
            scores = scores.masked_fill(causal_mask_tensor, float('-inf'))

        # Apply top-k sparse selection if k is provided
        if k is not None and k < seq_len:
            # Keep track of k value
            self.last_k = k

            # For each query position, select top-k key positions
            # scores: [batch, num_heads, seq_len, seq_len]
            topk_values, topk_indices = torch.topk(scores, k=k, dim=-1, sorted=False)

            # Create sparse scores tensor
            sparse_scores = torch.full_like(scores, float('-inf'))
            sparse_scores.scatter_(-1, topk_indices, topk_values)
            scores = sparse_scores
        else:
            self.last_k = seq_len

        # Compute attention weights and output
        attn_weights = F.softmax(scores, dim=-1)  # [batch, num_heads, seq_len, seq_len]
        attn_output = torch.matmul(attn_weights, v)  # [batch, num_heads, seq_len, head_dim]

        # Reshape and project
        attn_output = attn_output.transpose(1, 2).reshape(batch_size, seq_len, self.hidden_dim)
        output = self.out_proj(attn_output)

        return output


class TransformerBlock(nn.Module):
    """Single transformer block with sparse attention."""

    def __init__(self, hidden_dim: int, num_heads: int, ff_dim: int, dropout: float = 0.1):
        super().__init__()
        self.attention = SparseAttention(hidden_dim, num_heads)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

        self.ff = nn.Sequential(
            nn.Linear(hidden_dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, hidden_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor, k: Optional[int] = None) -> torch.Tensor:
        # Attention with residual
        attn_out = self.attention(self.norm1(x), k=k)
        x = x + attn_out

        # Feedforward with residual
        ff_out = self.ff(self.norm2(x))
        x = x + ff_out

        return x


class SparseTransformer(nn.Module):
    """Small transformer with sparse attention capability."""

    def __init__(
        self,
        vocab_size: int,
        hidden_dim: int = 256,
        num_layers: int = 6,
        num_heads: int = 4,
        ff_dim: int = 1024,
        max_seq_len: int = 512,
        dropout: float = 0.1
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.max_seq_len = max_seq_len

        self.token_embedding = nn.Embedding(vocab_size, hidden_dim)
        self.pos_embedding = nn.Parameter(torch.zeros(1, max_seq_len, hidden_dim))
        self.dropout = nn.Dropout(dropout)

        self.blocks = nn.ModuleList([
            TransformerBlock(hidden_dim, num_heads, ff_dim, dropout)
            for _ in range(num_layers)
        ])

        self.norm = nn.LayerNorm(hidden_dim)
        self.lm_head = nn.Linear(hidden_dim, vocab_size)

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, input_ids: torch.Tensor, k: Optional[int] = None) -> torch.Tensor:
        """
        Args:
            input_ids: [batch, seq_len]
            k: top-k for sparse attention (None = dense)
        Returns:
            logits: [batch, seq_len, vocab_size]
        """
        batch_size, seq_len = input_ids.shape

        # Embeddings
        token_emb = self.token_embedding(input_ids)  # [batch, seq_len, hidden_dim]
        pos_emb = self.pos_embedding[:, :seq_len, :]  # [1, seq_len, hidden_dim]
        x = self.dropout(token_emb + pos_emb)

        # Transformer blocks with sparse attention
        for block in self.blocks:
            x = block(x, k=k)

        # Output
        x = self.norm(x)
        logits = self.lm_head(x)  # [batch, seq_len, vocab_size]

        return logits

    def get_last_k_values(self) -> List[int]:
        """Get the k values used by each attention layer."""
        k_values = []
        for block in self.blocks:
            if hasattr(block.attention, 'last_k') and block.attention.last_k is not None:
                k_values.append(block.attention.last_k)
        return k_values


class TinyShakespeareDataset(Dataset):
    """Simple character-level dataset from TinyShakespeare."""

    def __init__(self, text: str, seq_len: int = 128):
        self.seq_len = seq_len

        # Build vocabulary
        chars = sorted(list(set(text)))
        self.vocab_size = len(chars)
        self.char_to_idx = {ch: i for i, ch in enumerate(chars)}
        self.idx_to_char = {i: ch for i, ch in enumerate(chars)}

        # Encode text
        self.data = [self.char_to_idx[ch] for ch in text]

    def __len__(self):
        return len(self.data) - self.seq_len - 1

    def __getitem__(self, idx):
        # Get sequence and target
        x = torch.tensor(self.data[idx:idx + self.seq_len], dtype=torch.long)
        y = torch.tensor(self.data[idx + 1:idx + self.seq_len + 1], dtype=torch.long)
        return x, y


class HardwareGate:
    """Hardware-based gate for dynamic k selection."""

    def __init__(
        self,
        sensor: SysfsHwmonTelemetry,
        min_k: int = 64,
        max_k: int = 512,
        power_threshold_w: float = 15.0,
        temp_threshold_c: float = 70.0
    ):
        self.sensor = sensor
        self.min_k = min_k
        self.max_k = max_k
        self.power_threshold = power_threshold_w
        self.temp_threshold = temp_threshold_c

        # Tracking
        self.k_history = []
        self.power_history = []
        self.temp_history = []

    def get_k(self) -> int:
        """Compute k based on current hardware state."""
        # Read hardware state
        sample = self.sensor.read_sample()
        power_w = sample.power_w
        temp_c = sample.temp_edge_c

        # Normalize power and temp to [0, 1]
        # Higher values = worse conditions = lower k
        power_normalized = min(power_w / self.power_threshold, 1.0)
        temp_normalized = min(temp_c / self.temp_threshold, 1.0)

        # Combine (take max for conservative approach)
        stress_factor = max(power_normalized, temp_normalized)

        # Compute k: high stress → low k
        k_range = self.max_k - self.min_k
        k = self.max_k - int(stress_factor * k_range)
        k = max(self.min_k, min(k, self.max_k))

        # Track
        self.k_history.append(k)
        self.power_history.append(power_w)
        self.temp_history.append(temp_c)

        return k

    def get_stats(self) -> Dict:
        """Get statistics about k selection."""
        if not self.k_history:
            return {
                'avg_k': 0, 'min_k': 0, 'max_k': 0, 'std_k': 0,
                'avg_power': 0, 'avg_temp': 0
            }

        k_array = torch.tensor(self.k_history, dtype=torch.float32)
        return {
            'avg_k': k_array.mean().item(),
            'min_k': int(k_array.min().item()),
            'max_k': int(k_array.max().item()),
            'std_k': k_array.std().item(),
            'avg_power': sum(self.power_history) / len(self.power_history),
            'avg_temp': sum(self.temp_history) / len(self.temp_history)
        }


def load_tiny_shakespeare() -> str:
    """Load TinyShakespeare dataset."""
    # Try to download if not exists
    data_path = project_root / "data" / "tiny_shakespeare.txt"

    if not data_path.exists():
        print("Downloading TinyShakespeare...")
        import urllib.request
        data_path.parent.mkdir(parents=True, exist_ok=True)
        url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
        urllib.request.urlretrieve(url, data_path)
        print(f"Downloaded to {data_path}")

    return data_path.read_text()


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    device: torch.device,
    num_epochs: int = 3,
    lr: float = 3e-4
) -> None:
    """Train the model."""
    print(f"\n{'='*80}")
    print("TRAINING MODEL")
    print(f"{'='*80}")

    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    total_steps = 0
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        epoch_steps = 0

        for batch_idx, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)

            # Forward pass (dense attention during training)
            logits = model(x, k=None)

            # Compute loss
            loss = criterion(logits.reshape(-1, logits.size(-1)), y.reshape(-1))

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()
            epoch_steps += 1
            total_steps += 1

            if (batch_idx + 1) % 50 == 0:
                avg_loss = epoch_loss / epoch_steps
                perplexity = math.exp(min(avg_loss, 10))
                print(f"Epoch {epoch+1}/{num_epochs} | Step {batch_idx+1}/{len(train_loader)} | "
                      f"Loss: {avg_loss:.4f} | Perplexity: {perplexity:.2f}")

        avg_loss = epoch_loss / epoch_steps
        perplexity = math.exp(min(avg_loss, 10))
        print(f"Epoch {epoch+1}/{num_epochs} complete | Loss: {avg_loss:.4f} | Perplexity: {perplexity:.2f}")


def compute_perplexity(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    k: Optional[int] = None,
    gate: Optional[HardwareGate] = None,
    random_k: bool = False,
    min_k: int = 64,
    max_k: int = 512
) -> float:
    """Compute perplexity on the dataset."""
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        for x, y in data_loader:
            x, y = x.to(device), y.to(device)

            # Determine k value
            if gate is not None:
                current_k = gate.get_k()
            elif random_k:
                current_k = random.randint(min_k, max_k)
            else:
                current_k = k

            # Forward pass
            logits = model(x, k=current_k)

            # Compute loss
            loss = criterion(logits.reshape(-1, logits.size(-1)), y.reshape(-1))

            total_loss += loss.item() * y.numel()
            total_tokens += y.numel()

    avg_loss = total_loss / total_tokens
    perplexity = math.exp(min(avg_loss, 10))
    return perplexity


def benchmark_condition(
    condition_name: str,
    model: nn.Module,
    test_loader: DataLoader,
    sensor: SysfsHwmonTelemetry,
    device: torch.device,
    k: Optional[int] = None,
    gate: Optional[HardwareGate] = None,
    random_k: bool = False,
    min_k: int = 64,
    max_k: int = 512,
    num_warmup: int = 10
) -> AttentionMetrics:
    """Benchmark a single attention condition."""
    print(f"\n{'='*80}")
    print(f"BENCHMARKING: {condition_name}")
    print(f"{'='*80}")

    model.eval()

    # Warmup
    print("Warming up...")
    with torch.no_grad():
        for i, (x, _) in enumerate(test_loader):
            if i >= num_warmup:
                break
            x = x.to(device)
            _ = model(x, k=k if k else max_k)

    # Reset sensor
    sensor.reset_accumulator()
    if gate:
        gate.k_history.clear()
        gate.power_history.clear()
        gate.temp_history.clear()

    # Benchmark
    print("Running benchmark...")
    start_time = time.perf_counter()
    total_tokens = 0
    k_values = []
    powers = []
    temps = []

    with torch.no_grad():
        with EnergyMeter(sensor) as meter:
            for batch_idx, (x, _) in enumerate(test_loader):
                x = x.to(device)
                batch_size, seq_len = x.shape

                # Read telemetry
                sample = sensor.read_sample()
                powers.append(sample.power_w)
                temps.append(sample.temp_edge_c)

                # Determine k value
                if gate is not None:
                    current_k = gate.get_k()
                elif random_k:
                    current_k = random.randint(min_k, max_k)
                else:
                    current_k = k

                # Forward pass
                _ = model(x, k=current_k)

                # Track k values used
                layer_k_values = model.get_last_k_values()
                if layer_k_values:
                    k_values.extend(layer_k_values)

                total_tokens += batch_size * seq_len

                if (batch_idx + 1) % 20 == 0:
                    print(f"Processed {batch_idx + 1}/{len(test_loader)} batches "
                          f"({total_tokens} tokens) | k={current_k if current_k else 'full'}")

    end_time = time.perf_counter()
    inference_time = end_time - start_time

    # Get energy metrics
    energy_j = meter.energy_j
    avg_power = sum(powers) / len(powers) if powers else 0
    avg_temp = sum(temps) / len(temps) if temps else 0

    # Compute perplexity
    print("Computing perplexity...")
    perplexity = compute_perplexity(
        model, test_loader, device,
        k=k, gate=gate, random_k=random_k,
        min_k=min_k, max_k=max_k
    )

    # Get k statistics
    if gate:
        k_stats = gate.get_stats()
        avg_k = k_stats['avg_k']
        min_k_val = k_stats['min_k']
        max_k_val = k_stats['max_k']
        k_std = k_stats['std_k']
    elif k_values:
        k_tensor = torch.tensor(k_values, dtype=torch.float32)
        avg_k = k_tensor.mean().item()
        min_k_val = int(k_tensor.min().item())
        max_k_val = int(k_tensor.max().item())
        k_std = k_tensor.std().item()
    else:
        avg_k = k if k else max_k
        min_k_val = k if k else max_k
        max_k_val = k if k else max_k
        k_std = 0.0

    # Compute metrics
    j_per_token = energy_j / total_tokens if total_tokens > 0 else 0
    tokens_per_sec = total_tokens / inference_time if inference_time > 0 else 0

    metrics = AttentionMetrics(
        condition=condition_name,
        total_energy_j=energy_j,
        total_tokens=total_tokens,
        j_per_token=j_per_token,
        perplexity=perplexity,
        avg_k=avg_k,
        min_k=min_k_val,
        max_k=max_k_val,
        k_std=k_std,
        avg_power_w=avg_power,
        avg_temp_c=avg_temp,
        inference_time_s=inference_time,
        tokens_per_sec=tokens_per_sec
    )

    # Print results
    print(f"\nResults for {condition_name}:")
    print(f"  Total Energy: {energy_j:.4f} J")
    print(f"  Total Tokens: {total_tokens}")
    print(f"  J/Token: {j_per_token:.6f}")
    print(f"  Perplexity: {perplexity:.2f}")
    print(f"  Avg k: {avg_k:.1f} (min={min_k_val}, max={max_k_val}, std={k_std:.1f})")
    print(f"  Avg Power: {avg_power:.2f} W")
    print(f"  Avg Temp: {avg_temp:.1f} °C")
    print(f"  Inference Time: {inference_time:.2f} s")
    print(f"  Tokens/sec: {tokens_per_sec:.1f}")

    return metrics


def main():
    print("="*80)
    print("Hardware-Gated Sparse Attention Benchmark")
    print("="*80)

    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if device.type != "cuda":
        print("WARNING: CUDA not available, running on CPU")
        print("Energy measurements may not be accurate")

    # Initialize sensor
    sensor = SysfsHwmonTelemetry()
    sample = sensor.read_sample()
    print(f"Sensor initialized: GPU at {sample.temp_edge_c:.1f}C, {sample.power_w:.1f}W")

    # Load data
    print("\nLoading TinyShakespeare dataset...")
    text = load_tiny_shakespeare()
    print(f"Dataset size: {len(text)} characters")

    # Create dataset and loaders
    seq_len = 128
    dataset = TinyShakespeareDataset(text, seq_len=seq_len)
    vocab_size = dataset.vocab_size
    print(f"Vocabulary size: {vocab_size}")
    print(f"Sequence length: {seq_len}")

    # Split into train/test (90/10)
    train_size = int(0.9 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )

    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    print(f"Train batches: {len(train_loader)}, Test batches: {len(test_loader)}")

    # Create model
    print("\nCreating model...")
    model = SparseTransformer(
        vocab_size=vocab_size,
        hidden_dim=256,
        num_layers=6,
        num_heads=4,
        ff_dim=1024,
        max_seq_len=seq_len,
        dropout=0.1
    ).to(device)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")

    # Train model
    train_model(model, train_loader, device, num_epochs=3, lr=3e-4)

    # Save model
    model_path = project_root / "checkpoints" / "z912f_sparse_model.pt"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), model_path)
    print(f"\nModel saved to {model_path}")

    # Setup hardware gate
    min_k = 64
    max_k = seq_len  # Full attention possible
    gate = HardwareGate(
        sensor=sensor,
        min_k=min_k,
        max_k=max_k,
        power_threshold_w=15.0,
        temp_threshold_c=70.0
    )

    # Benchmark all conditions
    print("\n" + "="*80)
    print("RUNNING BENCHMARKS")
    print("="*80)

    results = []

    # 1. Fixed dense (full attention)
    metrics_dense = benchmark_condition(
        condition_name="Fixed Dense (Full Attention)",
        model=model,
        test_loader=test_loader,
        sensor=sensor,
        device=device,
        k=None,  # No top-k selection
        num_warmup=10
    )
    results.append(metrics_dense)

    time.sleep(2)  # Cool down

    # 2. Fixed sparse (k=64)
    metrics_sparse = benchmark_condition(
        condition_name="Fixed Sparse (k=64)",
        model=model,
        test_loader=test_loader,
        sensor=sensor,
        device=device,
        k=min_k,
        num_warmup=10
    )
    results.append(metrics_sparse)

    time.sleep(2)

    # 3. Embodied sparse (hardware-gated)
    metrics_embodied = benchmark_condition(
        condition_name="Embodied Sparse (Hardware-Gated)",
        model=model,
        test_loader=test_loader,
        sensor=sensor,
        device=device,
        gate=gate,
        min_k=min_k,
        max_k=max_k,
        num_warmup=10
    )
    results.append(metrics_embodied)

    time.sleep(2)

    # 4. Random sparse (baseline)
    metrics_random = benchmark_condition(
        condition_name="Random Sparse (k varies randomly)",
        model=model,
        test_loader=test_loader,
        sensor=sensor,
        device=device,
        random_k=True,
        min_k=min_k,
        max_k=max_k,
        num_warmup=10
    )
    results.append(metrics_random)

    # Compute correlations for embodied condition
    print(f"\n{'='*80}")
    print("CORRELATION ANALYSIS (Embodied Condition)")
    print(f"{'='*80}")

    if gate.k_history and gate.power_history and gate.temp_history:
        k_tensor = torch.tensor(gate.k_history, dtype=torch.float32)
        power_tensor = torch.tensor(gate.power_history, dtype=torch.float32)
        temp_tensor = torch.tensor(gate.temp_history, dtype=torch.float32)

        # Compute correlations
        k_power_corr = torch.corrcoef(torch.stack([k_tensor, power_tensor]))[0, 1].item()
        k_temp_corr = torch.corrcoef(torch.stack([k_tensor, temp_tensor]))[0, 1].item()

        print(f"Correlation(k, power): {k_power_corr:.3f}")
        print(f"Correlation(k, temp): {k_temp_corr:.3f}")
        print(f"Expected: negative correlation (higher power/temp → lower k)")

        correlation_stats = {
            'k_power_correlation': k_power_corr,
            'k_temp_correlation': k_temp_corr,
            'k_distribution': {
                'mean': k_tensor.mean().item(),
                'std': k_tensor.std().item(),
                'min': k_tensor.min().item(),
                'max': k_tensor.max().item(),
                'median': k_tensor.median().item()
            },
            'power_distribution': {
                'mean': power_tensor.mean().item(),
                'std': power_tensor.std().item(),
                'min': power_tensor.min().item(),
                'max': power_tensor.max().item()
            },
            'temp_distribution': {
                'mean': temp_tensor.mean().item(),
                'std': temp_tensor.std().item(),
                'min': temp_tensor.min().item(),
                'max': temp_tensor.max().item()
            }
        }
    else:
        correlation_stats = None

    # Print summary comparison
    print(f"\n{'='*80}")
    print("SUMMARY COMPARISON")
    print(f"{'='*80}")
    print(f"{'Condition':<35} {'J/Token':<12} {'Perplexity':<12} {'Avg k':<10}")
    print("-" * 80)

    for m in results:
        print(f"{m.condition:<35} {m.j_per_token:<12.6f} {m.perplexity:<12.2f} {m.avg_k:<10.1f}")

    # Compute savings vs dense
    dense_energy = metrics_dense.j_per_token
    print(f"\n{'Condition':<35} {'Energy Savings':<20} {'Quality Delta':<15}")
    print("-" * 80)

    for m in results[1:]:  # Skip dense baseline
        energy_savings = ((dense_energy - m.j_per_token) / dense_energy) * 100
        quality_delta = m.perplexity - metrics_dense.perplexity
        print(f"{m.condition:<35} {energy_savings:>18.1f}% {quality_delta:>13.2f}")

    # Save results
    output_path = project_root / "results" / "z912f_sparse_attention.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    output_data = {
        'metadata': {
            'device': str(device),
            'sensor': 'AMD Radeon 8060S',
            'model_params': num_params,
            'vocab_size': vocab_size,
            'seq_len': seq_len,
            'num_layers': 6,
            'hidden_dim': 256,
            'num_heads': 4,
            'min_k': min_k,
            'max_k': max_k,
            'test_batches': len(test_loader),
            'batch_size': batch_size
        },
        'results': [asdict(m) for m in results],
        'correlation_stats': correlation_stats,
        'comparison': {
            'dense_baseline_j_per_token': dense_energy,
            'dense_baseline_perplexity': metrics_dense.perplexity,
            'savings_vs_dense': {
                m.condition: {
                    'energy_savings_percent': ((dense_energy - m.j_per_token) / dense_energy) * 100,
                    'perplexity_delta': m.perplexity - metrics_dense.perplexity,
                    'tokens_per_sec_ratio': m.tokens_per_sec / metrics_dense.tokens_per_sec if metrics_dense.tokens_per_sec > 0 else 0
                }
                for m in results[1:]
            }
        }
    }

    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"\nResults saved to {output_path}")
    print("="*80)
    print("Benchmark complete!")
    print("="*80)


if __name__ == "__main__":
    main()
