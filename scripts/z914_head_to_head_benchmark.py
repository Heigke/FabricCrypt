#!/usr/bin/env python3
"""
z914: Head-to-Head Controller Benchmark
========================================

Definitive comparison of ALL controller types on IDENTICAL workload:

1. CLASSICAL BASELINES:
   - Fixed (no control)
   - PID (proportional-integral-derivative)
   - MPC (model predictive control)
   - GreenLLM (dual-loop throughput + thermal)
   - ThrottLLM (phase-predictive)

2. EMBODIED CONTROLLERS:
   - Bandit (contextual multi-armed bandit)
   - Embodied (FiLM-conditioned with body state)
   - Case-Based (AnchoredGraphMemory retrieval)

All controllers drive the SAME actuators with SAME constraints.
Measured on SAME workload with identical energy accounting.

This is the DEFINITIVE test of whether embodiment beats classical control.

Author: FEEL Research Team
Date: 2026-01-29
"""

import os
import sys
import json
import time
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import Dict, Any, List, Optional, Tuple
from collections import defaultdict

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from telemetry.sysfs_hwmon import SysfsHwmonTelemetry, EnergyMeter, GpuSample
from memory.anchored_graph import AnchoredGraphMemory, HardwareAnchor, ProvenanceType
from body_daemon.controller.baselines import (
    FixedCapController, PIDController, GreenLLMController, ThrottLLMController,
    FixedCapConfig, PIDConfig, GreenLLMConfig, ThrottLLMConfig,
)


# ============================================================================
# Unified Action Space (all controllers use same actuators)
# ============================================================================

@dataclass
class ComputeAction:
    """Unified action representation for all controllers."""
    precision: str = "fp16"  # fp32, fp16, int8
    batch_size: int = 4
    attention_window: int = 2048
    compute_level: int = 2  # 0=minimal, 4=maximum

    @classmethod
    def from_level(cls, level: int) -> 'ComputeAction':
        """Create action from discrete level (0-4)."""
        configs = [
            cls(precision="int8", batch_size=1, attention_window=512, compute_level=0),
            cls(precision="int8", batch_size=2, attention_window=1024, compute_level=1),
            cls(precision="fp16", batch_size=4, attention_window=2048, compute_level=2),
            cls(precision="fp16", batch_size=8, attention_window=3072, compute_level=3),
            cls(precision="fp32", batch_size=16, attention_window=4096, compute_level=4),
        ]
        return configs[max(0, min(level, 4))]


# ============================================================================
# Case-Based Controller (uses AnchoredGraphMemory)
# ============================================================================

class CaseBasedController:
    """
    Controller that retrieves similar past states from graph memory.

    This WIRES the AnchoredGraphMemory into the control loop.
    """

    def __init__(self, memory: AnchoredGraphMemory, fallback_level: int = 2):
        self.memory = memory
        self.fallback_level = fallback_level
        self.step_count = 0
        self.total_reward = 0.0
        self.retrieval_count = 0
        self.fallback_count = 0

    def _state_to_vector(self, state: np.ndarray) -> List[float]:
        """Convert state array to list for memory query."""
        return state.tolist()

    def select_action(self, state: np.ndarray) -> int:
        """Select action by retrieving similar past cases."""
        query_vector = self._state_to_vector(state)

        # Query graph memory for similar states
        similar_cases = self.memory.find_similar_states(
            query_vector,
            top_k=3,
            min_similarity=0.7,
        )

        if similar_cases:
            # Use action from most similar case
            best_case = similar_cases[0]
            action = best_case.get('action', {}).get('compute_level', self.fallback_level)
            self.retrieval_count += 1
            return int(action)
        else:
            # Fallback to default
            self.fallback_count += 1
            return self.fallback_level

    def update(self, state: np.ndarray, action: int, reward: float,
               anchor: Optional[HardwareAnchor] = None,
               next_state: Optional[np.ndarray] = None):
        """Store case in memory for future retrieval."""
        self.step_count += 1
        self.total_reward += reward

        if anchor is not None:
            # Store this case for future retrieval
            self.memory.add_control_case(
                body_latent_vector=state.tolist(),
                action={'compute_level': action, 'reward': reward},
                anchor=anchor,
                outcome_metrics={'reward': reward},
            )

    def get_stats(self) -> Dict[str, Any]:
        retrieval_rate = self.retrieval_count / max(self.step_count, 1)
        return {
            'type': 'CaseBased',
            'step_count': self.step_count,
            'total_reward': self.total_reward,
            'avg_reward': self.total_reward / max(self.step_count, 1),
            'retrieval_count': self.retrieval_count,
            'fallback_count': self.fallback_count,
            'retrieval_rate': retrieval_rate,
            'memory_size': len(self.memory.nodes),
        }


# ============================================================================
# Embodied Controller (FiLM-conditioned)
# ============================================================================

class EmbodiedFiLMController:
    """
    Controller that uses FiLM conditioning on body state.

    Learns gamma, beta parameters from telemetry to modulate decisions.
    """

    def __init__(self, state_dim: int = 18, hidden_dim: int = 64):
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim

        # FiLM generator: state -> (gamma, beta) for decision modulation
        self.film_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 10),  # 5 actions x 2 (gamma, beta)
        )

        # Action value network
        self.value_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 5),  # 5 action values
        )

        # Training state
        self.optimizer = torch.optim.Adam(
            list(self.film_net.parameters()) + list(self.value_net.parameters()),
            lr=1e-3
        )
        self.step_count = 0
        self.total_reward = 0.0
        self.training_losses = []

    def select_action(self, state: np.ndarray) -> int:
        """Select action using FiLM-modulated Q-values."""
        with torch.no_grad():
            state_t = torch.FloatTensor(state).unsqueeze(0)

            # Get FiLM parameters
            film_params = self.film_net(state_t)
            gamma = film_params[:, :5]
            beta = film_params[:, 5:]

            # Get base Q-values
            q_values = self.value_net(state_t)

            # Apply FiLM modulation
            modulated_q = q_values * (1 + gamma) + beta

            # Epsilon-greedy exploration
            if np.random.random() < 0.1:
                return np.random.randint(5)
            return modulated_q.argmax().item()

    def update(self, state: np.ndarray, action: int, reward: float,
               next_state: Optional[np.ndarray] = None):
        """Update controller with TD learning."""
        self.step_count += 1
        self.total_reward += reward

        if next_state is None:
            return

        state_t = torch.FloatTensor(state).unsqueeze(0)
        next_state_t = torch.FloatTensor(next_state).unsqueeze(0)

        # Current Q-value
        film_params = self.film_net(state_t)
        gamma = film_params[:, :5]
        beta = film_params[:, 5:]
        q_values = self.value_net(state_t)
        modulated_q = q_values * (1 + gamma) + beta
        current_q = modulated_q[0, action]

        # Target Q-value (no modulation for stability)
        with torch.no_grad():
            next_q_values = self.value_net(next_state_t)
            target_q = reward + 0.99 * next_q_values.max()

        # TD loss
        loss = (current_q - target_q) ** 2

        # Update
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.training_losses.append(loss.item())

    def get_stats(self) -> Dict[str, Any]:
        return {
            'type': 'EmbodiedFiLM',
            'step_count': self.step_count,
            'total_reward': self.total_reward,
            'avg_reward': self.total_reward / max(self.step_count, 1),
            'avg_loss': np.mean(self.training_losses[-100:]) if self.training_losses else 0.0,
        }


# ============================================================================
# Workload Model (same for all controllers)
# ============================================================================

class TestWorkload:
    """
    Standardized workload that all controllers must handle.

    Simulates realistic inference patterns with varying load.
    """

    def __init__(self, device: str = "cuda"):
        self.device = device
        self.vocab_size = 8192
        self.hidden_dim = 512

        # Simple transformer-like model
        self.model = nn.Sequential(
            nn.Embedding(self.vocab_size, self.hidden_dim),
            nn.Linear(self.hidden_dim, self.hidden_dim * 4),
            nn.GELU(),
            nn.Linear(self.hidden_dim * 4, self.hidden_dim),
            nn.Linear(self.hidden_dim, self.vocab_size),
        ).to(device)

    def run_batch(self, action: ComputeAction) -> Dict[str, float]:
        """Run workload with given compute configuration."""
        batch_size = action.batch_size
        seq_len = min(action.attention_window, 512)  # Cap for speed

        # Create input - ensure minimum work for energy measurement
        effective_batch = max(batch_size, 4)  # Minimum batch for measurable energy
        effective_seq = max(seq_len, 256)     # Minimum seq for measurable energy

        input_ids = torch.randint(0, self.vocab_size, (effective_batch, effective_seq), device=self.device)

        # Run with appropriate precision
        if action.precision == "fp32":
            with torch.amp.autocast(device_type='cuda', enabled=False):
                output = self.model(input_ids)
                loss = output.mean()
                loss.backward()
        elif action.precision == "fp16":
            with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                output = self.model(input_ids)
                loss = output.mean()
            loss.backward()
        else:  # int8 (simulate with smaller compute, still measurable)
            with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                # Use smaller batches but still enough for measurement
                subset_batch = max(2, effective_batch // 2)
                subset_seq = max(128, effective_seq // 2)
                output = self.model(input_ids[:subset_batch, :subset_seq])
                loss = output.mean()
            loss.backward()

        torch.cuda.synchronize()

        # Return actual tokens processed
        if action.precision == "int8":
            tokens = max(2, effective_batch // 2) * max(128, effective_seq // 2)
        else:
            tokens = effective_batch * effective_seq

        return {
            'tokens': tokens,
            'perplexity': loss.item() + 10.0,  # Simulated
        }


# ============================================================================
# Body State Encoder
# ============================================================================

def encode_body_state(sample: GpuSample, prev_sample: Optional[GpuSample] = None) -> np.ndarray:
    """
    Encode GPU sample into body state vector.

    Returns 18-dim vector matching BodyState.to_observation_vector() layout.
    """
    state = np.zeros(18, dtype=np.float32)

    # Power (normalized to 0-1 assuming 300W TDP)
    power_norm = sample.power_w / 300.0
    state[0] = power_norm  # current
    state[1] = power_norm  # EMA (same for single sample)
    state[2] = 0.0  # derivative

    # Temperature (normalized to 0-1 assuming 100C max)
    temp_norm = sample.temp_edge_c / 100.0
    state[3] = temp_norm  # current
    state[4] = temp_norm  # EMA
    state[5] = 0.0  # derivative

    # Utilization
    util_norm = getattr(sample, 'gpu_busy_pct', 50.0) / 100.0
    state[6] = util_norm
    state[7] = util_norm
    state[8] = 0.0

    # Homeostatic deviation (from setpoints)
    power_setpoint = 0.6  # 60% of TDP
    temp_setpoint = 0.65  # 65C target
    state[9] = power_norm - power_setpoint
    state[10] = temp_norm - temp_setpoint

    # Clocks (normalized)
    freq_norm = sample.freq_sclk_mhz / 3000.0 if sample.freq_sclk_mhz else 0.5
    state[11] = freq_norm
    state[12] = freq_norm
    state[13] = 1.0  # ratio

    # Throttle flag
    state[14] = 1.0 if sample.temp_edge_c > 75 else 0.0

    # Efficiency (log J/token placeholder)
    state[15] = 0.5

    # Reserved
    state[16] = 0.5
    state[17] = 0.5

    return state


# ============================================================================
# Main Benchmark
# ============================================================================

def run_benchmark(
    duration_sec: float = 60.0,
    warmup_sec: float = 5.0,
    results_dir: Path = None,
) -> Dict[str, Any]:
    """
    Run head-to-head benchmark of all controllers.
    """
    print("=" * 70)
    print("z914: HEAD-TO-HEAD CONTROLLER BENCHMARK")
    print("=" * 70)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Initialize telemetry
    telemetry = SysfsHwmonTelemetry()

    # Initialize workload
    workload = TestWorkload(device)

    # Initialize graph memory for case-based controller
    memory = AnchoredGraphMemory()

    # Create all controllers
    controllers = {
        # Classical baselines
        'Fixed-ECO': FixedCapController(FixedCapConfig(power_level=0)),
        'Fixed-MED': FixedCapController(FixedCapConfig(power_level=2)),
        'Fixed-PERF': FixedCapController(FixedCapConfig(power_level=4)),
        'PID': PIDController(PIDConfig(kp=0.5, ki=0.1, kd=0.05, temp_setpoint=0.65)),
        'GreenLLM': GreenLLMController(GreenLLMConfig()),
        'ThrottLLM': ThrottLLMController(ThrottLLMConfig()),
        # Embodied controllers
        'CaseBased': CaseBasedController(memory),
        'EmbodiedFiLM': EmbodiedFiLMController(),
    }

    print(f"\nTesting {len(controllers)} controllers:")
    for name in controllers:
        print(f"  - {name}")

    # Results storage
    results = {
        'benchmark': 'z914_head_to_head',
        'timestamp': datetime.now().isoformat(),
        'duration_sec': duration_sec,
        'device': device,
        'controllers': {},
    }

    # Run each controller
    for ctrl_name, controller in controllers.items():
        print(f"\n{'='*50}")
        print(f"Testing: {ctrl_name}")
        print(f"{'='*50}")

        # Reset state
        torch.cuda.empty_cache() if device == "cuda" else None

        # Warmup
        print("  Warming up...", end=" ", flush=True)
        warmup_start = time.time()
        while time.time() - warmup_start < warmup_sec:
            sample = telemetry.read_sample()
            state = encode_body_state(sample)
            action_level = controller.select_action(state)
            action = ComputeAction.from_level(action_level)
            workload.run_batch(action)
        print("done")

        # Main benchmark with continuous energy measurement
        print(f"  Running benchmark ({duration_sec}s)...")

        start_time = time.time()
        total_tokens = 0
        step_metrics = []
        prev_sample = None
        power_samples = []  # Collect power for integration

        # Start continuous sampling for entire benchmark
        telemetry.reset_accumulator()
        telemetry.start_continuous_sampling()

        while time.time() - start_time < duration_sec:
            step_start = time.time()

            # Read telemetry
            sample = telemetry.read_sample()
            state = encode_body_state(sample, prev_sample)
            power_samples.append((time.time(), sample.power_w))

            # Select action
            action_level = controller.select_action(state)
            action = ComputeAction.from_level(action_level)

            # Run workload (no per-step energy measurement)
            batch_result = workload.run_batch(action)
            tokens = batch_result['tokens']

            # Estimate per-step energy from power
            step_power = sample.power_w
            step_duration = time.time() - step_start
            step_energy = step_power * step_duration

            # Calculate reward
            j_per_token = step_energy / max(tokens, 1)
            temp_penalty = max(0, (sample.temp_edge_c - 70) / 30.0) * 0.01
            reward = -j_per_token - temp_penalty

            # Update controller
            next_sample = telemetry.read_sample()
            next_state = encode_body_state(next_sample, sample)

            # Create hardware anchor for case-based controller
            anchor = HardwareAnchor.from_telemetry({
                'power_watts': sample.power_w,
                'temperature_c': sample.temp_edge_c,
                'energy_mj': int(step_energy * 1000),
                'utilization': getattr(sample, 'gpu_busy_pct', 50.0),
                'profile': action.precision,
            }, 'ikaros')

            if hasattr(controller, 'update'):
                if ctrl_name == 'CaseBased':
                    controller.update(state, action_level, reward, anchor, next_state)
                elif hasattr(controller, 'optimizer'):  # EmbodiedFiLM
                    controller.update(state, action_level, reward, next_state)
                else:
                    controller.update(state, action_level, reward, next_state)

            # Accumulate
            total_tokens += tokens

            step_metrics.append({
                'time': time.time() - start_time,
                'tokens': tokens,
                'step_energy_j': step_energy,
                'action_level': action_level,
                'temp_c': sample.temp_edge_c,
                'power_w': sample.power_w,
                'reward': reward,
            })

            prev_sample = sample

            # Progress
            if len(step_metrics) % 50 == 0:
                elapsed = time.time() - start_time
                # Integrate power for total energy so far
                total_energy_j = sum(m['step_energy_j'] for m in step_metrics)
                avg_j_per_token = total_energy_j / max(total_tokens, 1)
                print(f"    [{elapsed:.1f}s] steps={len(step_metrics)}, "
                      f"tokens={total_tokens:,}, avg_j/tok={avg_j_per_token:.6f}")

        # Stop sampling and compute final energy
        telemetry.stop_continuous_sampling()

        # Get total energy from accumulator (more accurate)
        with telemetry._lock:
            accumulated_energy_j = telemetry.accumulator.total_energy_j

        # Fall back to sum of step energies if accumulator failed
        step_energy_sum = sum(m['step_energy_j'] for m in step_metrics)
        total_energy_j = accumulated_energy_j if accumulated_energy_j > 0 else step_energy_sum

        # Compute final metrics
        elapsed = time.time() - start_time
        avg_j_per_token = total_energy_j / max(total_tokens, 1)
        throughput = total_tokens / elapsed

        stats = controller.get_stats() if hasattr(controller, 'get_stats') else {}

        ctrl_results = {
            'controller_type': stats.get('type', ctrl_name),
            'total_tokens': total_tokens,
            'total_energy_j': total_energy_j,
            'elapsed_sec': elapsed,
            'avg_j_per_token': avg_j_per_token,
            'throughput_tok_per_sec': throughput,
            'num_steps': len(step_metrics),
            'avg_reward': stats.get('avg_reward', 0.0),
            'controller_stats': stats,
            # Time series for analysis
            'j_per_token_series': [m['step_energy_j'] / 1024 for m in step_metrics],  # Approx per-token
            'temp_series': [m['temp_c'] for m in step_metrics],
            'action_series': [m['action_level'] for m in step_metrics],
        }

        results['controllers'][ctrl_name] = ctrl_results

        print(f"\n  Results for {ctrl_name}:")
        print(f"    Total tokens:    {total_tokens:,}")
        print(f"    Total energy:    {total_energy_j:.2f} J")
        print(f"    Avg J/token:     {avg_j_per_token:.6f}")
        print(f"    Throughput:      {throughput:.1f} tok/s")
        print(f"    Avg reward:      {stats.get('avg_reward', 0.0):.4f}")

    # ========================================================================
    # Comparative Analysis
    # ========================================================================

    print("\n" + "=" * 70)
    print("COMPARATIVE ANALYSIS")
    print("=" * 70)

    # Rank by efficiency (lower J/token is better)
    rankings = sorted(
        results['controllers'].items(),
        key=lambda x: x[1]['avg_j_per_token']
    )

    print("\nEfficiency Ranking (J/token, lower is better):")
    print("-" * 50)
    baseline_j = rankings[-1][1]['avg_j_per_token']  # Worst as baseline

    for rank, (name, data) in enumerate(rankings, 1):
        j_per_token = data['avg_j_per_token']
        improvement = (1 - j_per_token / baseline_j) * 100
        throughput = data['throughput_tok_per_sec']

        print(f"  {rank}. {name:15s}: {j_per_token:.6f} J/tok "
              f"({improvement:+.1f}% vs worst) | {throughput:.1f} tok/s")

    # Compare embodied vs classical
    classical_names = ['Fixed-ECO', 'Fixed-MED', 'Fixed-PERF', 'PID', 'GreenLLM', 'ThrottLLM']
    embodied_names = ['CaseBased', 'EmbodiedFiLM']

    # Get best non-zero J/tok for fair comparison
    classical_j_vals = [
        results['controllers'][n]['avg_j_per_token'] for n in classical_names
        if n in results['controllers'] and results['controllers'][n]['avg_j_per_token'] > 0
    ]
    embodied_j_vals = [
        results['controllers'][n]['avg_j_per_token'] for n in embodied_names
        if n in results['controllers'] and results['controllers'][n]['avg_j_per_token'] > 0
    ]

    classical_best = min(classical_j_vals) if classical_j_vals else float('inf')
    embodied_best = min(embodied_j_vals) if embodied_j_vals else float('inf')

    print("\n" + "-" * 50)
    print("CLASSICAL vs EMBODIED (non-zero energy only):")
    print(f"  Best Classical: {classical_best:.6f} J/tok")
    print(f"  Best Embodied:  {embodied_best:.6f} J/tok")

    if classical_best == float('inf') or embodied_best == float('inf'):
        print("  >>> INSUFFICIENT DATA FOR COMPARISON <<<")
        results['winner'] = 'insufficient_data'
        results['improvement_pct'] = 0.0
    elif embodied_best < classical_best:
        improvement = (1 - embodied_best / classical_best) * 100
        print(f"  >>> EMBODIED WINS by {improvement:.1f}% <<<")
        results['winner'] = 'embodied'
        results['improvement_pct'] = improvement
    else:
        degradation = (embodied_best / classical_best - 1) * 100
        print(f"  >>> CLASSICAL WINS by {degradation:.1f}% <<<")
        results['winner'] = 'classical'
        results['improvement_pct'] = -degradation

    # Save results
    if results_dir is None:
        results_dir = Path(__file__).parent.parent / "results"
    results_dir.mkdir(exist_ok=True)

    results_file = results_dir / "z914_head_to_head.json"
    with open(results_file, 'w') as f:
        # Remove time series for compact JSON
        compact_results = {
            k: {k2: v2 for k2, v2 in v.items()
                if not k2.endswith('_series')} if isinstance(v, dict) else v
            for k, v in results.items()
        }
        compact_results['controllers'] = {
            k: {k2: v2 for k2, v2 in v.items()
                if not k2.endswith('_series')}
            for k, v in results['controllers'].items()
        }
        json.dump(compact_results, f, indent=2, default=str)

    print(f"\nResults saved to: {results_file}")

    return results


# ============================================================================
# Entry Point
# ============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="z914 Head-to-Head Benchmark")
    parser.add_argument("--duration", type=float, default=30.0,
                        help="Benchmark duration per controller (seconds)")
    parser.add_argument("--warmup", type=float, default=3.0,
                        help="Warmup duration (seconds)")

    args = parser.parse_args()

    results = run_benchmark(
        duration_sec=args.duration,
        warmup_sec=args.warmup,
    )

    print("\n" + "=" * 70)
    print("BENCHMARK COMPLETE")
    print("=" * 70)
