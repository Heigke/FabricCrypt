#!/usr/bin/env python3
"""
z915: Auto-Tune Classical Controllers
======================================

Tunes PID and GreenLLM parameters through grid search to ensure
fair comparison with embodied controllers.

Approach:
1. Grid search over parameter space
2. Evaluate each configuration for 10s
3. Select best parameters
4. Re-run z914 with tuned controllers

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
from dataclasses import dataclass
from typing import Dict, Any, List, Tuple
from itertools import product

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from telemetry.sysfs_hwmon import SysfsHwmonTelemetry, EnergyMeter, GpuSample
from body_daemon.controller.baselines import (
    PIDController, PIDConfig,
    GreenLLMController, GreenLLMConfig,
)


# ============================================================================
# Workload (same as z914)
# ============================================================================

class TestWorkload:
    """Standardized workload for tuning."""

    def __init__(self, device: str = "cuda"):
        self.device = device
        self.vocab_size = 8192
        self.hidden_dim = 512

        self.model = nn.Sequential(
            nn.Embedding(self.vocab_size, self.hidden_dim),
            nn.Linear(self.hidden_dim, self.hidden_dim * 4),
            nn.GELU(),
            nn.Linear(self.hidden_dim * 4, self.hidden_dim),
            nn.Linear(self.hidden_dim, self.vocab_size),
        ).to(device)

    def run_batch(self, batch_size: int = 4, seq_len: int = 256) -> int:
        """Run workload, return tokens processed."""
        input_ids = torch.randint(0, self.vocab_size, (batch_size, seq_len), device=self.device)
        with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
            output = self.model(input_ids)
            loss = output.mean()
        loss.backward()
        torch.cuda.synchronize()
        return batch_size * seq_len


def encode_state(sample: GpuSample) -> np.ndarray:
    """Encode GPU sample into state vector."""
    state = np.zeros(18, dtype=np.float32)
    state[0] = sample.power_w / 300.0
    state[1] = sample.power_w / 300.0
    state[3] = sample.temp_edge_c / 100.0
    state[4] = sample.temp_edge_c / 100.0
    state[6] = getattr(sample, 'gpu_busy_pct', 50.0) / 100.0
    return state


# ============================================================================
# Tuning Functions
# ============================================================================

def evaluate_controller(controller, workload, telemetry, duration_sec: float = 10.0) -> Dict[str, float]:
    """Evaluate a controller for a fixed duration."""
    torch.cuda.empty_cache()

    total_tokens = 0
    power_samples = []

    telemetry.reset_accumulator()
    telemetry.start_continuous_sampling()
    start_time = time.time()

    while time.time() - start_time < duration_sec:
        sample = telemetry.read_sample()
        state = encode_state(sample)
        power_samples.append((time.time(), sample.power_w))

        action = controller.select_action(state)
        tokens = workload.run_batch()
        total_tokens += tokens

        # Reward
        j_per_token = sample.power_w * 0.02 / max(tokens, 1)  # Approx
        temp_penalty = max(0, (sample.temp_edge_c - 70) / 30.0) * 0.01
        reward = -j_per_token - temp_penalty

        controller.update(state, action, reward)

    telemetry.stop_continuous_sampling()

    # Compute metrics
    elapsed = time.time() - start_time
    with telemetry._lock:
        total_energy_j = telemetry.accumulator.total_energy_j

    if total_energy_j <= 0:
        # Fallback: integrate power samples
        total_energy_j = sum(p * 0.02 for t, p in power_samples)

    j_per_token = total_energy_j / max(total_tokens, 1)
    throughput = total_tokens / elapsed

    return {
        'j_per_token': j_per_token,
        'throughput': throughput,
        'total_tokens': total_tokens,
        'total_energy_j': total_energy_j,
        'stats': controller.get_stats(),
    }


def tune_pid(workload, telemetry) -> Tuple[PIDConfig, Dict]:
    """Tune PID controller parameters."""
    print("\n" + "=" * 50)
    print("Tuning PID Controller")
    print("=" * 50)

    # Parameter grid
    kp_values = [0.1, 0.3, 0.5, 0.7, 1.0]
    ki_values = [0.01, 0.05, 0.1, 0.2]
    kd_values = [0.01, 0.05, 0.1]
    temp_setpoints = [0.55, 0.65, 0.75]

    best_config = None
    best_j_per_token = float('inf')
    results = []

    total_combos = len(kp_values) * len(ki_values) * len(kd_values) * len(temp_setpoints)
    combo_idx = 0

    for kp, ki, kd, setpoint in product(kp_values, ki_values, kd_values, temp_setpoints):
        combo_idx += 1
        print(f"  [{combo_idx}/{total_combos}] kp={kp}, ki={ki}, kd={kd}, setpoint={setpoint}...", end=" ")

        config = PIDConfig(kp=kp, ki=ki, kd=kd, temp_setpoint=setpoint)
        controller = PIDController(config)

        metrics = evaluate_controller(controller, workload, telemetry, duration_sec=5.0)
        j_per_token = metrics['j_per_token']

        print(f"J/tok={j_per_token:.6f}")

        results.append({
            'config': {'kp': kp, 'ki': ki, 'kd': kd, 'temp_setpoint': setpoint},
            'j_per_token': j_per_token,
            'throughput': metrics['throughput'],
        })

        if j_per_token < best_j_per_token:
            best_j_per_token = j_per_token
            best_config = config

    print(f"\nBest PID: kp={best_config.kp}, ki={best_config.ki}, kd={best_config.kd}, setpoint={best_config.temp_setpoint}")
    print(f"  J/token: {best_j_per_token:.6f}")

    return best_config, {'best': best_config, 'results': results}


def tune_greenllm(workload, telemetry) -> Tuple[GreenLLMConfig, Dict]:
    """Tune GreenLLM controller parameters."""
    print("\n" + "=" * 50)
    print("Tuning GreenLLM Controller")
    print("=" * 50)

    # Parameter grid
    throughput_targets = [100.0, 300.0, 500.0]
    throughput_gains = [0.005, 0.01, 0.02]
    temp_thresholds_high = [0.75, 0.85, 0.95]

    best_config = None
    best_j_per_token = float('inf')
    results = []

    total_combos = len(throughput_targets) * len(throughput_gains) * len(temp_thresholds_high)
    combo_idx = 0

    for target, gain, thresh in product(throughput_targets, throughput_gains, temp_thresholds_high):
        combo_idx += 1
        print(f"  [{combo_idx}/{total_combos}] target={target}, gain={gain}, thresh={thresh}...", end=" ")

        config = GreenLLMConfig(
            throughput_target=target,
            throughput_gain=gain,
            temp_threshold_high=thresh,
        )
        controller = GreenLLMController(config)

        metrics = evaluate_controller(controller, workload, telemetry, duration_sec=5.0)
        j_per_token = metrics['j_per_token']

        print(f"J/tok={j_per_token:.6f}")

        results.append({
            'config': {'throughput_target': target, 'throughput_gain': gain, 'temp_threshold_high': thresh},
            'j_per_token': j_per_token,
            'throughput': metrics['throughput'],
        })

        if j_per_token < best_j_per_token:
            best_j_per_token = j_per_token
            best_config = config

    print(f"\nBest GreenLLM: target={best_config.throughput_target}, gain={best_config.throughput_gain}, thresh={best_config.temp_threshold_high}")
    print(f"  J/token: {best_j_per_token:.6f}")

    return best_config, {'best': best_config, 'results': results}


# ============================================================================
# Main
# ============================================================================

def main():
    print("=" * 60)
    print("z915: Classical Controller Tuning")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Initialize
    telemetry = SysfsHwmonTelemetry()
    workload = TestWorkload(device)

    # Warmup
    print("\nWarming up...")
    for _ in range(10):
        workload.run_batch()

    # Tune PID
    best_pid_config, pid_results = tune_pid(workload, telemetry)

    # Tune GreenLLM
    best_greenllm_config, greenllm_results = tune_greenllm(workload, telemetry)

    # Save results
    results = {
        'timestamp': datetime.now().isoformat(),
        'pid': {
            'best_config': {
                'kp': best_pid_config.kp,
                'ki': best_pid_config.ki,
                'kd': best_pid_config.kd,
                'temp_setpoint': best_pid_config.temp_setpoint,
            },
            'tuning_results': pid_results['results'][:10],  # Top 10
        },
        'greenllm': {
            'best_config': {
                'throughput_target': best_greenllm_config.throughput_target,
                'throughput_gain': best_greenllm_config.throughput_gain,
                'temp_threshold_high': best_greenllm_config.temp_threshold_high,
            },
            'tuning_results': greenllm_results['results'][:10],
        },
    }

    results_dir = Path(__file__).parent.parent / "results"
    results_dir.mkdir(exist_ok=True)
    results_file = results_dir / "z915_tuning_results.json"

    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\nResults saved to: {results_file}")

    # Print summary
    print("\n" + "=" * 60)
    print("TUNING SUMMARY")
    print("=" * 60)
    print(f"\nBest PID Config:")
    print(f"  kp={best_pid_config.kp}, ki={best_pid_config.ki}, kd={best_pid_config.kd}")
    print(f"  temp_setpoint={best_pid_config.temp_setpoint}")
    print(f"\nBest GreenLLM Config:")
    print(f"  throughput_target={best_greenllm_config.throughput_target}")
    print(f"  throughput_gain={best_greenllm_config.throughput_gain}")
    print(f"  temp_threshold_high={best_greenllm_config.temp_threshold_high}")

    return results


if __name__ == "__main__":
    main()
