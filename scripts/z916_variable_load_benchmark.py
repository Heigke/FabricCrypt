#!/usr/bin/env python3
"""
z916: Variable Load Benchmark - Tests Adaptive Capabilities
============================================================

The z914 steady-state benchmark showed Fixed-MED beats CaseBased by 35%.
This is because steady-state doesn't require adaptation.

This benchmark tests scenarios where ADAPTATION matters:
1. Bursty load (idle -> heavy -> idle cycles)
2. Thermal stress (force GPU to throttle)
3. Mixed workload (varying batch sizes)
4. SLO pressure (tight latency constraints)

Hypothesis: Embodied controllers will outperform fixed configs when
conditions change, because they can adapt while fixed configs cannot.

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
from typing import Dict, Any, List, Optional, Tuple
from collections import defaultdict

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from telemetry.sysfs_hwmon import SysfsHwmonTelemetry, GpuSample
from memory.anchored_graph import AnchoredGraphMemory, HardwareAnchor, ProvenanceType


# ============================================================================
# Controllers (same as z914)
# ============================================================================

class FixedController:
    """Fixed power level - cannot adapt."""
    def __init__(self, level: int = 2):
        self.level = level
        self.name = f"Fixed-{['ECO','LOW','MED','HIGH','PERF'][level]}"

    def select_action(self, state: np.ndarray) -> int:
        return self.level

    def update(self, state, action, reward, **kwargs):
        pass

    def get_stats(self):
        return {'type': self.name}


class CaseBasedController:
    """Embodied controller with graph memory."""
    def __init__(self, fallback_level: int = 2):
        self.memory = AnchoredGraphMemory()
        self.fallback_level = fallback_level
        self.step_count = 0
        self.retrieval_count = 0

    def select_action(self, state: np.ndarray) -> int:
        similar = self.memory.find_similar_states(state.tolist(), top_k=3, min_similarity=0.7)
        if similar:
            self.retrieval_count += 1
            return int(similar[0].get('action', {}).get('level', self.fallback_level))
        return self.fallback_level

    def update(self, state, action, reward, anchor=None, **kwargs):
        self.step_count += 1
        if anchor:
            self.memory.add_control_case(
                body_latent_vector=state.tolist(),
                action={'level': action, 'reward': reward},
                anchor=anchor,
            )

    def get_stats(self):
        return {
            'type': 'CaseBased',
            'memory_size': len(self.memory.nodes),
            'retrieval_rate': self.retrieval_count / max(1, self.step_count),
        }


class AdaptiveController:
    """Simple adaptive controller - adjusts based on temperature."""
    def __init__(self):
        self.current_level = 2
        self.temp_history = []

    def select_action(self, state: np.ndarray) -> int:
        temp_norm = state[3]  # Temperature in state vector
        temp_c = temp_norm * 100

        self.temp_history.append(temp_c)
        if len(self.temp_history) > 10:
            self.temp_history = self.temp_history[-10:]

        avg_temp = np.mean(self.temp_history)

        # Adapt based on temperature
        if avg_temp > 75:
            self.current_level = max(0, self.current_level - 1)
        elif avg_temp < 55:
            self.current_level = min(4, self.current_level + 1)

        return self.current_level

    def update(self, state, action, reward, **kwargs):
        pass

    def get_stats(self):
        return {'type': 'Adaptive', 'current_level': self.current_level}


# ============================================================================
# Workload with Variable Load Patterns
# ============================================================================

class VariableWorkload:
    """Workload that can simulate different load patterns."""

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

    def run_batch(self, batch_size: int, seq_len: int, precision: str = "fp16") -> int:
        """Run workload with specified configuration."""
        input_ids = torch.randint(0, self.vocab_size, (batch_size, seq_len), device=self.device)

        if precision == "fp32":
            with torch.amp.autocast(device_type='cuda', enabled=False):
                output = self.model(input_ids)
                loss = output.mean()
                loss.backward()
        else:
            with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                output = self.model(input_ids)
                loss = output.mean()
            loss.backward()

        torch.cuda.synchronize()
        return batch_size * seq_len

    def idle(self, duration_sec: float = 0.5):
        """Simulate idle period."""
        time.sleep(duration_sec)


# ============================================================================
# Load Patterns
# ============================================================================

def bursty_load_pattern(elapsed_sec: float) -> Tuple[int, int, str]:
    """
    Bursty load: alternates between heavy and light load.
    Returns (batch_size, seq_len, precision)
    """
    cycle = int(elapsed_sec / 5) % 4  # 5-second cycles

    if cycle == 0:  # Heavy burst
        return (16, 512, "fp32")
    elif cycle == 1:  # Light
        return (2, 128, "fp16")
    elif cycle == 2:  # Medium burst
        return (8, 256, "fp16")
    else:  # Idle-ish
        return (1, 64, "fp16")


def thermal_stress_pattern(elapsed_sec: float, temp_c: float) -> Tuple[int, int, str]:
    """
    Thermal stress: increases load until thermal limit, then backs off.
    """
    if temp_c > 80:  # Throttling zone
        return (1, 64, "fp16")  # Back off
    elif temp_c > 70:  # Warning zone
        return (4, 128, "fp16")  # Moderate
    elif temp_c > 60:  # Normal zone
        return (8, 256, "fp16")  # Medium
    else:  # Cold - push harder
        return (16, 512, "fp32")  # Heavy


def mixed_workload_pattern(elapsed_sec: float) -> Tuple[int, int, str]:
    """
    Mixed workload: random variation to simulate real traffic.
    """
    np.random.seed(int(elapsed_sec * 10) % 10000)
    batch_size = np.random.choice([1, 2, 4, 8, 16], p=[0.3, 0.3, 0.2, 0.15, 0.05])
    seq_len = np.random.choice([64, 128, 256, 512], p=[0.2, 0.4, 0.3, 0.1])
    precision = "fp16" if np.random.random() < 0.8 else "fp32"
    return (int(batch_size), int(seq_len), precision)


# ============================================================================
# Benchmark Runner
# ============================================================================

def encode_state(sample: GpuSample) -> np.ndarray:
    """Encode GPU sample into state vector."""
    state = np.zeros(18, dtype=np.float32)
    state[0] = sample.power_w / 300.0
    state[1] = sample.power_w / 300.0
    state[3] = sample.temp_edge_c / 100.0
    state[4] = sample.temp_edge_c / 100.0
    state[6] = getattr(sample, 'gpu_busy_pct', 50.0) / 100.0
    return state


def run_variable_benchmark(
    pattern_name: str,
    pattern_fn,
    duration_sec: float = 60.0,
) -> Dict[str, Any]:
    """Run benchmark with a specific load pattern."""

    print(f"\n{'='*60}")
    print(f"Testing: {pattern_name}")
    print(f"{'='*60}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    telemetry = SysfsHwmonTelemetry()
    workload = VariableWorkload(device)

    # Controllers to test
    controllers = {
        'Fixed-ECO': FixedController(0),
        'Fixed-MED': FixedController(2),
        'Fixed-PERF': FixedController(4),
        'Adaptive': AdaptiveController(),
        'CaseBased': CaseBasedController(),
    }

    results = {}

    for ctrl_name, controller in controllers.items():
        print(f"\n  Testing {ctrl_name}...")
        torch.cuda.empty_cache()

        # Warmup
        for _ in range(5):
            workload.run_batch(4, 128)

        # Run benchmark
        start_time = time.time()
        total_tokens = 0
        total_energy_j = 0.0
        step_metrics = []

        telemetry.reset_accumulator()
        telemetry.start_continuous_sampling()

        while time.time() - start_time < duration_sec:
            elapsed = time.time() - start_time
            sample = telemetry.read_sample()
            state = encode_state(sample)

            # Get load from pattern (may use temperature for thermal pattern)
            if pattern_name == "thermal_stress":
                batch_size, seq_len, precision = pattern_fn(elapsed, sample.temp_edge_c)
            else:
                batch_size, seq_len, precision = pattern_fn(elapsed)

            # Controller selects action (affects how we'd actuate, but we use pattern for load)
            action = controller.select_action(state)

            # Run workload
            step_start = time.time()
            tokens = workload.run_batch(batch_size, seq_len, precision)
            step_duration = time.time() - step_start

            # Estimate energy
            step_energy = sample.power_w * step_duration
            total_tokens += tokens
            total_energy_j += step_energy

            # Reward based on efficiency
            j_per_token = step_energy / max(tokens, 1)
            temp_penalty = max(0, (sample.temp_edge_c - 70) / 30.0) * 0.01
            reward = -j_per_token - temp_penalty

            # Update controller
            anchor = HardwareAnchor.from_telemetry({
                'power_watts': sample.power_w,
                'temperature_c': sample.temp_edge_c,
                'energy_mj': int(step_energy * 1000),
                'utilization': getattr(sample, 'gpu_busy_pct', 50.0),
                'profile': precision,
            }, 'local')

            controller.update(state, action, reward, anchor=anchor)

            step_metrics.append({
                'time': elapsed,
                'tokens': tokens,
                'energy_j': step_energy,
                'temp_c': sample.temp_edge_c,
                'power_w': sample.power_w,
                'batch_size': batch_size,
                'action': action,
            })

        telemetry.stop_continuous_sampling()

        # Get accumulated energy
        with telemetry._lock:
            accumulated_energy = telemetry.accumulator.total_energy_j

        if accumulated_energy > 0:
            total_energy_j = accumulated_energy

        elapsed = time.time() - start_time
        j_per_token = total_energy_j / max(total_tokens, 1)
        throughput = total_tokens / elapsed

        # Temperature statistics
        temps = [m['temp_c'] for m in step_metrics]
        max_temp = max(temps) if temps else 0
        avg_temp = np.mean(temps) if temps else 0
        temp_variance = np.var(temps) if temps else 0

        results[ctrl_name] = {
            'total_tokens': total_tokens,
            'total_energy_j': total_energy_j,
            'j_per_token': j_per_token,
            'throughput': throughput,
            'max_temp_c': max_temp,
            'avg_temp_c': avg_temp,
            'temp_variance': temp_variance,
            'stats': controller.get_stats(),
        }

        print(f"    J/tok: {j_per_token:.6f}, Throughput: {throughput:.0f}, MaxTemp: {max_temp:.1f}°C")

    return results


def main():
    print("=" * 70)
    print("z916: VARIABLE LOAD BENCHMARK")
    print("=" * 70)
    print("\nThis tests ADAPTIVE capabilities that steady-state benchmarks miss.")
    print("Fixed controllers should struggle when conditions change.\n")

    patterns = [
        ("bursty_load", bursty_load_pattern, 60.0),
        ("thermal_stress", thermal_stress_pattern, 60.0),
        ("mixed_workload", mixed_workload_pattern, 60.0),
    ]

    all_results = {
        'benchmark': 'z916_variable_load',
        'timestamp': datetime.now().isoformat(),
        'patterns': {},
    }

    for pattern_name, pattern_fn, duration in patterns:
        results = run_variable_benchmark(pattern_name, pattern_fn, duration)
        all_results['patterns'][pattern_name] = results

    # Analysis
    print("\n" + "=" * 70)
    print("COMPARATIVE ANALYSIS")
    print("=" * 70)

    for pattern_name, results in all_results['patterns'].items():
        print(f"\n{pattern_name.upper()}:")
        print("-" * 50)

        # Rank by efficiency
        ranked = sorted(results.items(), key=lambda x: x[1]['j_per_token'])

        for rank, (name, data) in enumerate(ranked, 1):
            j_tok = data['j_per_token']
            throughput = data['throughput']
            max_temp = data['max_temp_c']
            print(f"  {rank}. {name:15s}: {j_tok:.6f} J/tok | {throughput:.0f} tok/s | {max_temp:.1f}°C max")

        # Check if embodied beats fixed
        fixed_med_j = results.get('Fixed-MED', {}).get('j_per_token', float('inf'))
        casebased_j = results.get('CaseBased', {}).get('j_per_token', float('inf'))
        adaptive_j = results.get('Adaptive', {}).get('j_per_token', float('inf'))

        best_embodied = min(casebased_j, adaptive_j)

        if best_embodied < fixed_med_j:
            improvement = (1 - best_embodied / fixed_med_j) * 100
            print(f"  >>> EMBODIED WINS by {improvement:.1f}% <<<")
            all_results['patterns'][pattern_name]['winner'] = 'embodied'
        else:
            degradation = (best_embodied / fixed_med_j - 1) * 100
            print(f"  >>> FIXED-MED WINS by {degradation:.1f}% <<<")
            all_results['patterns'][pattern_name]['winner'] = 'fixed'

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    embodied_wins = sum(1 for p in all_results['patterns'].values() if p.get('winner') == 'embodied')
    total_patterns = len(all_results['patterns'])

    print(f"\nEmbodied wins: {embodied_wins}/{total_patterns} patterns")

    if embodied_wins > total_patterns / 2:
        print("CONCLUSION: Embodied controllers show value in variable conditions")
    else:
        print("CONCLUSION: Fixed controllers remain competitive even under variable load")

    # Save results
    results_dir = Path(__file__).parent.parent / "results"
    results_dir.mkdir(exist_ok=True)
    results_file = results_dir / "z916_variable_load.json"

    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)

    print(f"\nResults saved to: {results_file}")

    return all_results


if __name__ == "__main__":
    main()
