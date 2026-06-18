#!/usr/bin/env python3
"""
z142_mpc_controller.py

MODEL PREDICTIVE CONTROL (MPC) CLOSED-LOOP CONTROLLER
======================================================

Implements proper MPC for energy-aware inference:
1. Sense body state (b_t)
2. Compute LM hidden state (h_t)
3. Simulate candidate actions through world model
4. Choose action minimizing cost (energy) subject to constraints (latency/quality)
5. Actuate
6. Repeat

This replaces hand-tuned heuristics with model-based anticipation.

Run with:
  HSA_OVERRIDE_GFX_VERSION=11.0.0 python scripts/z142_mpc_controller.py
"""

import argparse
import json
import os
import sys
import time
import random
import numpy as np
import torch
import torch.nn as nn
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
from enum import Enum

sys.path.insert(0, str(Path(__file__).parent.parent))


class ControlMode(Enum):
    """Controller modes for comparison."""
    BASELINE = "baseline"  # Fixed maximum performance
    EXTERNAL = "external"  # GreenLLM-style external DVFS
    MPC = "mpc"  # Model Predictive Control


@dataclass
class MPCConfig:
    """Configuration for MPC controller."""
    # Action space
    depth_levels: List[int] = None
    power_modes: List[str] = None

    # Constraints
    latency_target_ms: float = 50.0  # SLO target
    latency_penalty_weight: float = 10.0

    # Energy optimization
    energy_weight: float = 1.0

    # Quality (optional)
    quality_weight: float = 0.5

    # MPC horizon
    horizon: int = 1  # Look-ahead steps

    device: str = "cuda"

    def __post_init__(self):
        if self.depth_levels is None:
            self.depth_levels = [2, 3, 4, 5, 6]
        if self.power_modes is None:
            self.power_modes = ["eco", "balanced", "perf"]


class WorldModel(nn.Module):
    """World model for predicting body state from current state + action + LM hidden."""

    def __init__(self, telem_dim: int = 9, action_dim: int = 16,
                 lm_hidden_dim: int = 256, hidden_dim: int = 256):
        super().__init__()
        self.action_embed = nn.Embedding(30, action_dim)
        self.lm_proj = nn.Linear(lm_hidden_dim, 64)

        input_dim = telem_dim + action_dim + 64
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, telem_dim)
        )

    def forward(self, telem: torch.Tensor, action: torch.Tensor,
                lm_hidden: torch.Tensor) -> torch.Tensor:
        action_emb = self.action_embed(action)
        lm_feat = self.lm_proj(lm_hidden)
        x = torch.cat([telem, action_emb, lm_feat], dim=-1)
        return self.net(x)

    def predict_batch(self, telem: torch.Tensor, actions: torch.Tensor,
                      lm_hidden: torch.Tensor) -> torch.Tensor:
        """Predict for multiple candidate actions."""
        # telem: [telem_dim], actions: [n_actions], lm_hidden: [lm_dim]
        n_actions = actions.shape[0]
        telem_batch = telem.unsqueeze(0).expand(n_actions, -1)
        lm_batch = lm_hidden.unsqueeze(0).expand(n_actions, -1)
        return self.forward(telem_batch, actions, lm_batch)


def encode_action(depth: int, mode: str) -> int:
    """Encode depth + mode into single action index."""
    mode_idx = {"eco": 0, "balanced": 1, "perf": 2}.get(mode, 1)
    return (depth - 2) * 3 + mode_idx


def decode_action(action_idx: int) -> Tuple[int, str]:
    """Decode action index to depth + mode."""
    modes = ["eco", "balanced", "perf"]
    depth = (action_idx // 3) + 2
    mode = modes[action_idx % 3]
    return depth, mode


class TelemetryReader:
    """Read GPU telemetry with caching."""

    def __init__(self, gpu_id: int = 1):
        self.gpu_id = gpu_id
        self.drm_path = f"/sys/class/drm/card{gpu_id}/device"

        self.gpu_hwmon = None
        hwmon_path = f"{self.drm_path}/hwmon"
        if os.path.exists(hwmon_path):
            hwmons = os.listdir(hwmon_path)
            if hwmons:
                self.gpu_hwmon = f"{hwmon_path}/{hwmons[0]}"

        self.last_read_time = 0
        self.cached_telem = np.zeros(9, dtype=np.float32)
        self.cache_ttl = 0.01  # 10ms cache

    def read(self, inference_time_ms: float = 0) -> np.ndarray:
        """Read telemetry with optional timing."""
        now = time.time()

        if now - self.last_read_time > self.cache_ttl:
            telem = np.zeros(9, dtype=np.float32)

            try:
                if self.gpu_hwmon:
                    # Power
                    power_path = f"{self.gpu_hwmon}/power1_average"
                    if os.path.exists(power_path):
                        with open(power_path) as f:
                            power_uw = int(f.read().strip())
                            telem[0] = min(power_uw / 100_000_000, 1.0)

                    # Temperature
                    temp_path = f"{self.gpu_hwmon}/temp1_input"
                    if os.path.exists(temp_path):
                        with open(temp_path) as f:
                            temp_mc = int(f.read().strip())
                            telem[1] = min(temp_mc / 100_000, 1.0)

                # GPU utilization
                util_path = f"{self.drm_path}/gpu_busy_percent"
                if os.path.exists(util_path):
                    with open(util_path) as f:
                        util = int(f.read().strip())
                        telem[2] = util / 100.0

            except Exception:
                pass

            self.cached_telem = telem
            self.last_read_time = now

        # Add timing if provided
        result = self.cached_telem.copy()
        if inference_time_ms > 0:
            result[8] = min(inference_time_ms / 2.0, 1.0)

        return result


class MPCController:
    """Model Predictive Controller for energy-aware inference."""

    def __init__(self, world_model: WorldModel, config: MPCConfig):
        self.world_model = world_model
        self.config = config
        self.world_model.eval()

        # Precompute all action indices
        self.all_actions = []
        for depth in config.depth_levels:
            for mode in config.power_modes:
                self.all_actions.append(encode_action(depth, mode))

        self.action_tensor = torch.tensor(
            self.all_actions, dtype=torch.long, device=config.device
        )

    def compute_cost(self, predicted_telem: torch.Tensor, action_idx: int) -> float:
        """Compute cost for a predicted body state."""
        # Energy cost (power channel)
        energy = predicted_telem[0].item()

        # Latency cost (timing channel)
        timing = predicted_telem[8].item()
        latency_ms = timing * 2.0  # Denormalize

        # SLO violation penalty
        if latency_ms > self.config.latency_target_ms:
            latency_penalty = (latency_ms - self.config.latency_target_ms) * \
                              self.config.latency_penalty_weight
        else:
            latency_penalty = 0

        # Quality penalty (lower depth = lower quality)
        depth, _ = decode_action(action_idx)
        quality_penalty = (6 - depth) * self.config.quality_weight

        # Total cost
        cost = (
            self.config.energy_weight * energy +
            latency_penalty +
            quality_penalty
        )

        return cost

    def select_action(self, current_telem: torch.Tensor,
                      lm_hidden: torch.Tensor) -> Tuple[int, str, Dict]:
        """Select optimal action using MPC."""

        with torch.no_grad():
            # Predict future body state for all candidate actions
            predictions = self.world_model.predict_batch(
                current_telem, self.action_tensor, lm_hidden
            )

            # Compute costs
            costs = []
            for i, action_idx in enumerate(self.all_actions):
                cost = self.compute_cost(predictions[i], action_idx)
                costs.append((cost, action_idx, predictions[i]))

            # Select minimum cost action
            costs.sort(key=lambda x: x[0])
            best_cost, best_action, best_pred = costs[0]

            depth, mode = decode_action(best_action)

            info = {
                'cost': best_cost,
                'predicted_power': best_pred[0].item(),
                'predicted_timing': best_pred[8].item(),
                'all_costs': [(c, decode_action(a)) for c, a, _ in costs[:5]]
            }

        return depth, mode, info


class ExternalController:
    """GreenLLM-style external DVFS controller (for comparison)."""

    def __init__(self, config: MPCConfig):
        self.config = config
        self.last_latency = 0
        self.current_depth = 4  # Start medium

    def select_action(self, current_telem: np.ndarray, **kwargs) -> Tuple[int, str, Dict]:
        """Simple reactive controller based on latency."""
        timing = current_telem[8]
        latency_ms = timing * 2.0

        # Reactive adjustment
        if latency_ms > self.config.latency_target_ms * 0.9:
            # Approaching SLO, increase performance
            self.current_depth = min(self.current_depth + 1, 6)
        elif latency_ms < self.config.latency_target_ms * 0.5:
            # Well under SLO, reduce for energy
            self.current_depth = max(self.current_depth - 1, 2)

        self.last_latency = latency_ms

        return self.current_depth, "balanced", {'latency': latency_ms}


class BaselineController:
    """Fixed maximum performance baseline."""

    def __init__(self, config: MPCConfig):
        self.config = config

    def select_action(self, **kwargs) -> Tuple[int, str, Dict]:
        """Always use maximum depth and performance mode."""
        return 6, "perf", {}


class VariableDepthModel(nn.Module):
    """Model with variable depth for testing."""

    def __init__(self, vocab_size: int = 50257, hidden_dim: int = 256, n_layers: int = 6):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.current_depth = n_layers

        self.embed = nn.Embedding(vocab_size, hidden_dim)
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=hidden_dim, nhead=4, dim_feedforward=hidden_dim*4,
                batch_first=True
            ) for _ in range(n_layers)
        ])
        self.lm_head = nn.Linear(hidden_dim, vocab_size)

    def set_depth(self, depth: int):
        self.current_depth = min(max(depth, 1), self.n_layers)

    def forward(self, input_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.embed(input_ids)
        for i, layer in enumerate(self.layers):
            if i >= self.current_depth:
                break
            x = layer(x)
        hidden = x.mean(dim=1)
        logits = self.lm_head(x)
        return logits, hidden


def run_benchmark(controller, lm_model, tokenizer, telem_reader,
                  n_tokens: int, config: MPCConfig, mode_name: str) -> Dict:
    """Run benchmark with given controller."""

    print(f"\n  Running {mode_name} controller...")

    prompt = "The future of artificial intelligence and machine learning"
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(config.device)

    results = {
        'total_energy': 0,
        'total_time_ms': 0,
        'latencies': [],
        'slo_violations': 0,
        'depths_used': [],
        'actions': []
    }

    for i in range(n_tokens):
        # Read current telemetry
        telem_before = telem_reader.read()

        # Run inference
        start_time = time.perf_counter()

        with torch.no_grad():
            logits, hidden = lm_model(input_ids)

        inference_time = (time.perf_counter() - start_time) * 1000

        # Read telemetry after (with timing)
        telem_after = telem_reader.read(inference_time)
        telem_tensor = torch.tensor(telem_after, dtype=torch.float32, device=config.device)

        # Select next action
        if isinstance(controller, MPCController):
            hidden_tensor = hidden.squeeze(0)
            depth, mode, info = controller.select_action(telem_tensor, hidden_tensor)
        elif isinstance(controller, ExternalController):
            depth, mode, info = controller.select_action(telem_after)
        else:
            depth, mode, info = controller.select_action()

        # Apply action for next iteration
        lm_model.set_depth(depth)

        # Record metrics
        results['total_time_ms'] += inference_time
        results['latencies'].append(inference_time)
        results['depths_used'].append(depth)
        results['actions'].append((depth, mode))

        if inference_time > config.latency_target_ms:
            results['slo_violations'] += 1

        # Estimate energy from power reading
        power_w = telem_after[0] * 100  # Denormalize (approx)
        energy_j = power_w * (inference_time / 1000)
        results['total_energy'] += energy_j

        # Generate next token
        next_token = logits[0, -1].argmax().unsqueeze(0).unsqueeze(0)
        input_ids = torch.cat([input_ids, next_token], dim=1)

        if input_ids.shape[1] > 100:
            input_ids = input_ids[:, -100:]

    # Compute summary stats
    results['avg_latency_ms'] = np.mean(results['latencies'])
    results['p95_latency_ms'] = np.percentile(results['latencies'], 95)
    results['p99_latency_ms'] = np.percentile(results['latencies'], 99)
    results['avg_depth'] = np.mean(results['depths_used'])
    results['slo_violation_rate'] = results['slo_violations'] / n_tokens

    return results


def main():
    parser = argparse.ArgumentParser(description="MPC Controller Benchmark")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--tokens", type=int, default=200)
    parser.add_argument("--latency-slo", type=float, default=1.0, help="Latency SLO in ms")
    args = parser.parse_args()

    print("=" * 70)
    print("z142: MPC CLOSED-LOOP CONTROLLER BENCHMARK")
    print("=" * 70)
    print(f"Started: {datetime.now().isoformat()}")
    print(f"Device: {args.device}")
    print(f"Tokens: {args.tokens}")
    print(f"Latency SLO: {args.latency_slo}ms")

    config = MPCConfig(
        device=args.device,
        latency_target_ms=args.latency_slo
    )

    # Create models
    print("\nLoading models...")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-125M")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    lm_model = VariableDepthModel(
        vocab_size=tokenizer.vocab_size,
        hidden_dim=256,
        n_layers=6
    ).to(config.device)

    telem_reader = TelemetryReader(gpu_id=1)

    # Create world model (pretrained or random for demo)
    world_model = WorldModel(
        telem_dim=9,
        action_dim=16,
        lm_hidden_dim=256,
        hidden_dim=256
    ).to(config.device)

    # Create controllers
    mpc_controller = MPCController(world_model, config)
    external_controller = ExternalController(config)
    baseline_controller = BaselineController(config)

    # Run benchmarks
    print("\n" + "=" * 70)
    print("RUNNING BENCHMARKS")
    print("=" * 70)

    results = {}

    # Baseline
    lm_model.set_depth(6)  # Reset
    results['baseline'] = run_benchmark(
        baseline_controller, lm_model, tokenizer, telem_reader,
        args.tokens, config, "BASELINE"
    )

    # External (GreenLLM-style)
    lm_model.set_depth(4)  # Reset
    results['external'] = run_benchmark(
        external_controller, lm_model, tokenizer, telem_reader,
        args.tokens, config, "EXTERNAL (GreenLLM-style)"
    )

    # MPC
    lm_model.set_depth(4)  # Reset
    results['mpc'] = run_benchmark(
        mpc_controller, lm_model, tokenizer, telem_reader,
        args.tokens, config, "MPC"
    )

    # Print comparison
    print("\n" + "=" * 70)
    print("RESULTS COMPARISON")
    print("=" * 70)

    print(f"\n{'Metric':<25} {'Baseline':>12} {'External':>12} {'MPC':>12}")
    print("-" * 65)

    metrics = [
        ('Total Energy (J)', 'total_energy', '.3f'),
        ('Total Time (ms)', 'total_time_ms', '.1f'),
        ('Avg Latency (ms)', 'avg_latency_ms', '.3f'),
        ('P95 Latency (ms)', 'p95_latency_ms', '.3f'),
        ('P99 Latency (ms)', 'p99_latency_ms', '.3f'),
        ('SLO Violations (%)', 'slo_violation_rate', '.1%'),
        ('Avg Depth', 'avg_depth', '.2f'),
    ]

    for name, key, fmt in metrics:
        b = results['baseline'][key]
        e = results['external'][key]
        m = results['mpc'][key]
        print(f"{name:<25} {b:>12{fmt}} {e:>12{fmt}} {m:>12{fmt}}")

    # Compute improvements
    print("\n" + "-" * 65)
    print("IMPROVEMENTS vs BASELINE")
    print("-" * 65)

    base_energy = results['baseline']['total_energy']
    for mode in ['external', 'mpc']:
        energy = results[mode]['total_energy']
        improvement = (base_energy - energy) / base_energy * 100 if base_energy > 0 else 0
        print(f"  {mode.upper()}: {improvement:.1f}% energy reduction")

    # Save results
    os.makedirs("results/z142_mpc", exist_ok=True)
    results_path = "results/z142_mpc/benchmark_results.json"

    # Clean results for JSON
    clean_results = {}
    for mode, data in results.items():
        clean_results[mode] = {
            k: v for k, v in data.items()
            if k not in ['latencies', 'depths_used', 'actions']
        }
        clean_results[mode]['latencies_summary'] = {
            'mean': float(np.mean(data['latencies'])),
            'std': float(np.std(data['latencies'])),
            'min': float(np.min(data['latencies'])),
            'max': float(np.max(data['latencies']))
        }

    with open(results_path, 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'config': {
                'tokens': args.tokens,
                'latency_slo_ms': args.latency_slo
            },
            'results': clean_results
        }, f, indent=2)

    print(f"\nResults saved to: {results_path}")
    print(f"Completed: {datetime.now().isoformat()}")


if __name__ == "__main__":
    main()
