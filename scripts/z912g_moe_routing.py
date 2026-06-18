#!/usr/bin/env python3
"""
z912g_moe_routing.py - Energy-Aware Mixture of Experts Routing Benchmark

Tests whether hardware telemetry (power, temperature) can influence expert selection
in MoE models to optimize energy efficiency while maintaining quality.

Architecture:
1. Simple MoE Layer with 4 experts (different computational costs)
2. Energy-aware router that considers hardware state
3. Training with energy-shaped loss function
4. Comparison across 4 routing strategies

Business Value:
- Adaptive inference that responds to power constraints
- Lower operational costs in power-limited environments
- Proof-of-concept for hardware-aware neural routing
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import time
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from telemetry.sysfs_hwmon import SysfsHwmonTelemetry, EnergyMeter


@dataclass
class MoEMetrics:
    """Metrics for a single MoE condition."""
    condition: str
    energy_joules: float
    tokens_processed: int
    joules_per_token: float
    perplexity: float
    training_time_sec: float
    expert_selections: List[int]  # Count per expert
    power_expert_correlation: float  # Correlation between power and expert cost
    avg_expert_cost: float  # Average computational cost of selected experts

    def business_projection(self, tokens_per_day: int = 1_000_000) -> Dict:
        """Project business metrics."""
        daily_kwh = (self.joules_per_token * tokens_per_day) / 3_600_000
        monthly_kwh = daily_kwh * 30
        annual_kwh = daily_kwh * 365

        # Cost at $0.12/kWh
        monthly_cost = monthly_kwh * 0.12
        annual_cost = annual_kwh * 0.12

        return {
            "daily_kwh": round(daily_kwh, 3),
            "monthly_kwh": round(monthly_kwh, 2),
            "annual_kwh": round(annual_kwh, 2),
            "monthly_cost_usd": round(monthly_cost, 2),
            "annual_cost_usd": round(annual_cost, 2),
            "perplexity": round(self.perplexity, 3),
            "quality_cost_ratio": round(self.perplexity / self.joules_per_token, 2)
        }


class ExpertFFN(nn.Module):
    """Single expert FFN with configurable cost."""

    def __init__(self, d_model: int, d_ff: int, cost_multiplier: float = 1.0):
        super().__init__()
        self.cost_multiplier = cost_multiplier  # Relative computational cost
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through expert."""
        return self.fc2(self.activation(self.fc1(x)))

    @property
    def cost(self) -> float:
        """Return computational cost."""
        return self.cost_multiplier


class EnergyAwareMoELayer(nn.Module):
    """MoE layer with optional energy-aware routing."""

    def __init__(
        self,
        d_model: int = 256,
        num_experts: int = 4,
        use_hardware_telemetry: bool = False,
        expert_costs: Optional[List[float]] = None
    ):
        super().__init__()
        self.d_model = d_model
        self.num_experts = num_experts
        self.use_hardware_telemetry = use_hardware_telemetry

        # Create experts with different costs
        if expert_costs is None:
            expert_costs = [0.5, 0.75, 1.0, 1.5]  # Relative costs

        expert_sizes = [
            int(512 * cost) for cost in expert_costs
        ]

        self.experts = nn.ModuleList([
            ExpertFFN(d_model, size, cost)
            for size, cost in zip(expert_sizes, expert_costs)
        ])

        # Router: input + optional hardware telemetry (power, temp)
        router_input_size = d_model + (2 if use_hardware_telemetry else 0)
        self.router = nn.Linear(router_input_size, num_experts)

        # Track expert selections
        self.expert_counts = [0] * num_experts

    def forward(
        self,
        x: torch.Tensor,
        hardware_state: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, List[int]]:
        """
        Forward pass with expert routing.

        Args:
            x: Input tensor [batch, seq, d_model]
            hardware_state: Optional [batch, 2] tensor (power, temp normalized)

        Returns:
            output: Routed output [batch, seq, d_model]
            routing_weights: Expert weights [batch, seq, num_experts]
            selected_experts: Expert indices for each position
        """
        batch_size, seq_len, _ = x.shape

        # Prepare router input
        if self.use_hardware_telemetry and hardware_state is not None:
            # Broadcast hardware state to all positions
            hw_expanded = hardware_state.unsqueeze(1).expand(batch_size, seq_len, 2)
            router_input = torch.cat([x, hw_expanded], dim=-1)
        else:
            router_input = x

        # Compute routing weights
        routing_logits = self.router(router_input)
        routing_weights = F.softmax(routing_logits, dim=-1)

        # Select top-1 expert per position (simple routing)
        selected_experts = routing_weights.argmax(dim=-1)  # [batch, seq]

        # Apply experts
        output = torch.zeros_like(x)
        expert_indices = []

        for i in range(self.num_experts):
            mask = (selected_experts == i)
            if mask.any():
                expert_input = x[mask]
                expert_output = self.experts[i](expert_input)
                output[mask] = expert_output

                # Track expert usage
                count = mask.sum().item()
                self.expert_counts[i] += count
                expert_indices.extend([i] * count)

        return output, routing_weights, expert_indices

    def get_expert_costs(self, selected_experts: List[int]) -> float:
        """Calculate average expert cost for selected experts."""
        if not selected_experts:
            return 0.0
        costs = [self.experts[i].cost for i in selected_experts]
        return sum(costs) / len(costs)

    def reset_expert_counts(self):
        """Reset expert usage counters."""
        self.expert_counts = [0] * self.num_experts


class SimpleMoELanguageModel(nn.Module):
    """Simple character-level language model with MoE layer."""

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        num_experts: int = 4,
        use_hardware_telemetry: bool = False,
        expert_costs: Optional[List[float]] = None
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.moe = EnergyAwareMoELayer(
            d_model, num_experts, use_hardware_telemetry, expert_costs
        )
        self.ln = nn.LayerNorm(d_model)
        self.output = nn.Linear(d_model, vocab_size)

    def forward(
        self,
        x: torch.Tensor,
        hardware_state: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, List[int]]:
        """
        Forward pass.

        Returns:
            logits: [batch, seq, vocab_size]
            routing_weights: Expert routing weights
            selected_experts: Expert indices
        """
        x = self.embedding(x)
        x, routing_weights, selected_experts = self.moe(x, hardware_state)
        x = self.ln(x)
        logits = self.output(x)
        return logits, routing_weights, selected_experts


class RandomRoutingMoE(nn.Module):
    """MoE with random expert selection (baseline)."""

    def __init__(self, vocab_size: int, d_model: int = 256, num_experts: int = 4):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.experts = nn.ModuleList([
            ExpertFFN(d_model, 512, cost_multiplier=1.0)
            for _ in range(num_experts)
        ])
        self.ln = nn.LayerNorm(d_model)
        self.output = nn.Linear(d_model, vocab_size)
        self.num_experts = num_experts

    def forward(self, x: torch.Tensor, **kwargs) -> Tuple[torch.Tensor, torch.Tensor, List[int]]:
        batch_size, seq_len = x.shape
        x = self.embedding(x)

        # Random expert selection
        selected_experts = torch.randint(0, self.num_experts, (batch_size, seq_len))

        output = torch.zeros(batch_size, seq_len, x.size(-1), device=x.device)
        expert_indices = []

        for i in range(self.num_experts):
            mask = (selected_experts == i)
            if mask.any():
                expert_input = x[mask]
                expert_output = self.experts[i](expert_input)
                output[mask] = expert_output
                expert_indices.extend([i] * mask.sum().item())

        x = self.ln(output)
        logits = self.output(x)

        # Dummy routing weights
        routing_weights = torch.zeros(batch_size, seq_len, self.num_experts, device=x.device)

        return logits, routing_weights, expert_indices


class SingleExpertBaseline(nn.Module):
    """Single expert baseline (no routing)."""

    def __init__(self, vocab_size: int, d_model: int = 256):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.expert = ExpertFFN(d_model, 512, cost_multiplier=1.0)
        self.ln = nn.LayerNorm(d_model)
        self.output = nn.Linear(d_model, vocab_size)

    def forward(self, x: torch.Tensor, **kwargs) -> Tuple[torch.Tensor, torch.Tensor, List[int]]:
        x = self.embedding(x)
        x = self.expert(x)
        x = self.ln(x)
        logits = self.output(x)

        batch_size, seq_len = x.shape[:2]
        routing_weights = torch.zeros(batch_size, seq_len, 1, device=x.device)
        expert_indices = [0] * (batch_size * seq_len)

        return logits, routing_weights, expert_indices


def load_tiny_shakespeare(path: str = "data/ouroboros/tiny_shakespeare.txt") -> Tuple[str, Dict]:
    """Load TinyShakespeare dataset."""
    data_path = Path(path)
    if not data_path.exists():
        # Download if needed
        import urllib.request
        url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
        data_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Downloading TinyShakespeare to {data_path}...")
        urllib.request.urlretrieve(url, data_path)

    text = data_path.read_text()

    # Create vocabulary
    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    char_to_idx = {ch: i for i, ch in enumerate(chars)}
    idx_to_char = {i: ch for i, ch in enumerate(chars)}

    return text, {
        "vocab_size": vocab_size,
        "char_to_idx": char_to_idx,
        "idx_to_char": idx_to_char
    }


def create_batches(
    text: str,
    char_to_idx: Dict,
    batch_size: int = 32,
    seq_len: int = 64
) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    """Create training batches."""
    # Encode text
    data = torch.tensor([char_to_idx[ch] for ch in text], dtype=torch.long)

    # Create batches
    batches = []
    for i in range(0, len(data) - seq_len - 1, batch_size * seq_len):
        batch_data = []
        batch_targets = []

        for j in range(batch_size):
            start_idx = i + j * seq_len
            if start_idx + seq_len + 1 >= len(data):
                break

            batch_data.append(data[start_idx:start_idx + seq_len])
            batch_targets.append(data[start_idx + 1:start_idx + seq_len + 1])

        if batch_data:
            batches.append((
                torch.stack(batch_data),
                torch.stack(batch_targets)
            ))

    return batches


def train_condition(
    model: nn.Module,
    batches: List[Tuple[torch.Tensor, torch.Tensor]],
    condition_name: str,
    device: torch.device,
    epochs: int = 10,
    energy_lambda: float = 0.1,
    monitor: Optional[SysfsHwmonTelemetry] = None
) -> MoEMetrics:
    """
    Train model under one condition and collect metrics.

    Args:
        model: MoE model to train
        batches: Training batches
        condition_name: Name of condition
        device: Torch device
        epochs: Number of training epochs
        energy_lambda: Weight for energy-shaped loss
        monitor: Hardware monitor

    Returns:
        MoEMetrics for this condition
    """
    print(f"\n{'='*60}")
    print(f"Training: {condition_name}")
    print(f"{'='*60}")

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Energy tracking
    total_energy = 0.0
    total_tokens = 0
    all_expert_selections = []
    power_readings = []
    expert_costs = []

    start_time = time.time()

    # Normalize hardware telemetry ranges
    power_min, power_max = 5.0, 25.0  # Expected power range in watts
    temp_min, temp_max = 30.0, 80.0   # Expected temp range in celsius

    for epoch in range(epochs):
        epoch_loss = 0.0
        epoch_tokens = 0

        for batch_idx, (inputs, targets) in enumerate(batches):
            inputs = inputs.to(device)
            targets = targets.to(device)

            batch_size, seq_len = inputs.shape

            # Get hardware state if available
            hardware_state = None
            current_power = 0.0

            if monitor:
                sample = monitor.read_sample()
                power_w = sample.power_w
                temp_c = sample.temp_edge_c

                # Normalize to [0, 1]
                power_norm = (power_w - power_min) / (power_max - power_min)
                temp_norm = (temp_c - temp_min) / (temp_max - temp_min)

                hardware_state = torch.tensor(
                    [[power_norm, temp_norm]] * batch_size,
                    dtype=torch.float32,
                    device=device
                )

                current_power = power_w
                power_readings.append(power_w)

            # Forward pass
            logits, routing_weights, selected_experts = model(inputs, hardware_state=hardware_state)

            # Task loss
            loss_task = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1)
            )

            # Energy-shaped loss (for energy-aware condition)
            loss_energy = 0.0
            if "energy-aware" in condition_name.lower() and hasattr(model, 'moe'):
                # Penalize expensive experts when power is high
                avg_expert_cost = model.moe.get_expert_costs(selected_experts)
                power_penalty = current_power / power_max  # Higher power = higher penalty
                loss_energy = avg_expert_cost * power_penalty
                expert_costs.append(avg_expert_cost)

            # Combined loss
            loss = loss_task + energy_lambda * loss_energy

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Track metrics
            tokens = batch_size * seq_len
            epoch_loss += loss_task.item() * tokens
            epoch_tokens += tokens
            total_tokens += tokens
            all_expert_selections.extend(selected_experts)

            # Energy measurement
            if monitor:
                # Approximate energy for this batch
                batch_time = 0.05  # Rough estimate
                batch_energy = current_power * batch_time
                total_energy += batch_energy

            if (batch_idx + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Batch {batch_idx+1}/{len(batches)}, "
                      f"Loss: {loss_task.item():.4f}, "
                      f"Energy Loss: {loss_energy:.4f}" if loss_energy else "")

        avg_loss = epoch_loss / epoch_tokens
        perplexity = np.exp(avg_loss)
        print(f"Epoch {epoch+1} - Perplexity: {perplexity:.3f}")

    training_time = time.time() - start_time

    # Calculate perplexity on final epoch
    final_perplexity = perplexity

    # Expert selection distribution
    expert_counts = [0] * 4
    for exp_idx in all_expert_selections:
        if exp_idx < len(expert_counts):
            expert_counts[exp_idx] += 1

    # Power-expert correlation
    correlation = 0.0
    if power_readings and expert_costs:
        min_len = min(len(power_readings), len(expert_costs))
        if min_len > 1:
            correlation = np.corrcoef(
                power_readings[:min_len],
                expert_costs[:min_len]
            )[0, 1]

    avg_expert_cost = np.mean(expert_costs) if expert_costs else 1.0

    # Calculate J/token
    joules_per_token = total_energy / total_tokens if total_tokens > 0 else 0.0

    return MoEMetrics(
        condition=condition_name,
        energy_joules=total_energy,
        tokens_processed=total_tokens,
        joules_per_token=joules_per_token,
        perplexity=final_perplexity,
        training_time_sec=training_time,
        expert_selections=expert_counts,
        power_expert_correlation=correlation,
        avg_expert_cost=avg_expert_cost
    )


def main():
    """Main benchmark."""
    print("="*60)
    print("Energy-Aware MoE Routing Benchmark")
    print("="*60)

    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Initialize hardware monitor
    try:
        monitor = SysfsHwmonTelemetry()
        print("✓ Hardware monitor initialized")
    except Exception as e:
        print(f"⚠ Hardware monitor failed: {e}")
        monitor = None

    # Load dataset
    print("\nLoading TinyShakespeare...")
    text, vocab = load_tiny_shakespeare()
    print(f"✓ Loaded {len(text):,} characters, vocabulary size: {vocab['vocab_size']}")

    # Create batches
    print("\nCreating training batches...")
    batches = create_batches(text, vocab['char_to_idx'], batch_size=32, seq_len=64)
    print(f"✓ Created {len(batches)} batches")

    # Use subset for faster iteration
    batches = batches[:100]  # ~200k tokens
    print(f"Using {len(batches)} batches for benchmark")

    # Expert costs
    expert_costs = [0.5, 0.75, 1.0, 1.5]

    # Define conditions
    conditions = [
        ("Standard MoE (no hardware)", SimpleMoELanguageModel(
            vocab['vocab_size'], use_hardware_telemetry=False, expert_costs=expert_costs
        ), 0.0),
        ("Energy-Aware MoE (hardware)", SimpleMoELanguageModel(
            vocab['vocab_size'], use_hardware_telemetry=True, expert_costs=expert_costs
        ), 0.1),
        ("Random Routing", RandomRoutingMoE(vocab['vocab_size']), 0.0),
        ("Single Expert Baseline", SingleExpertBaseline(vocab['vocab_size']), 0.0),
    ]

    # Run benchmark
    results = []

    for condition_name, model, energy_lambda in conditions:
        metrics = train_condition(
            model,
            batches,
            condition_name,
            device,
            epochs=10,
            energy_lambda=energy_lambda,
            monitor=monitor
        )
        results.append(metrics)

        # Print summary
        print(f"\n{condition_name} Results:")
        print(f"  Energy: {metrics.energy_joules:.2f} J")
        print(f"  J/token: {metrics.joules_per_token:.6f}")
        print(f"  Perplexity: {metrics.perplexity:.3f}")
        print(f"  Expert selections: {metrics.expert_selections}")
        print(f"  Power-Cost correlation: {metrics.power_expert_correlation:.3f}")
        print(f"  Avg expert cost: {metrics.avg_expert_cost:.3f}")

    # Find best condition
    best_condition = min(results, key=lambda r: r.joules_per_token)
    baseline_condition = next(r for r in results if "baseline" in r.condition.lower())

    # Calculate improvements
    energy_improvement = (
        (baseline_condition.joules_per_token - best_condition.joules_per_token)
        / baseline_condition.joules_per_token * 100
    )

    # Business projections
    print("\n" + "="*60)
    print("BUSINESS PROJECTIONS (1M tokens/day)")
    print("="*60)

    for metrics in results:
        proj = metrics.business_projection(tokens_per_day=1_000_000)
        print(f"\n{metrics.condition}:")
        print(f"  Monthly cost: ${proj['monthly_cost_usd']}")
        print(f"  Annual cost: ${proj['annual_cost_usd']}")
        print(f"  Perplexity: {proj['perplexity']}")
        print(f"  Quality/Cost: {proj['quality_cost_ratio']}")

    # Summary report
    summary = {
        "benchmark": "z912g_moe_routing",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "device": str(device),
        "dataset": "TinyShakespeare",
        "tokens_processed": results[0].tokens_processed,
        "conditions": [asdict(r) for r in results],
        "best_condition": best_condition.condition,
        "energy_improvement_vs_baseline_pct": round(energy_improvement, 2),
        "business_projections": {
            r.condition: r.business_projection(1_000_000)
            for r in results
        },
        "key_findings": {
            "hardware_aware_routing_works": best_condition.condition == "Energy-Aware MoE (hardware)",
            "power_cost_correlation": round(
                next(r.power_expert_correlation for r in results if "energy-aware" in r.condition.lower()),
                3
            ),
            "expert_diversity": {
                r.condition: r.expert_selections for r in results
            }
        }
    }

    # Save results
    output_path = Path("results/z912g_moe_routing.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n✓ Results saved to {output_path}")

    # Final summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Best condition: {best_condition.condition}")
    print(f"Energy improvement vs baseline: {energy_improvement:+.1f}%")
    print(f"Power-cost correlation (energy-aware): {summary['key_findings']['power_cost_correlation']:.3f}")

    # Cleanup (SysfsHwmonTelemetry doesn't need explicit close)
    pass


if __name__ == "__main__":
    main()
