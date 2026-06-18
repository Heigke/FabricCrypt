#!/usr/bin/env python3
"""
z919: Workload-Aware Hybrid Controller
======================================

Fixes z917's issue: Controller decisions must CONTROL the workload,
not just react to it.

Key change: Controller selects (batch_size, seq_len, precision) directly,
and the workload executes what the controller decides.

Architecture:
1. Somatic trigger detects regime shift
2. Controller retrieves best policy for current regime
3. Policy DICTATES workload parameters (not just observes)
4. Actual J/token measured and fed back for learning

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
from enum import Enum
from collections import deque

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from telemetry.sysfs_hwmon import SysfsHwmonTelemetry, GpuSample

try:
    from memory.anchored_graph import AnchoredGraphMemory, HardwareAnchor, ProvenanceType
    HAS_MEMORY = True
except ImportError:
    HAS_MEMORY = False


# ============================================================================
# State Encoding
# ============================================================================

def encode_state(sample: GpuSample) -> np.ndarray:
    """Encode GPU sample into 18-dim state vector."""
    state = np.zeros(18, dtype=np.float32)
    state[0] = sample.power_w / 300.0
    state[1] = sample.power_w / 300.0
    state[3] = sample.temp_edge_c / 100.0
    state[4] = sample.temp_edge_c / 100.0
    state[6] = getattr(sample, 'gpu_busy_pct', 50.0) / 100.0
    return state


# ============================================================================
# Workload Configurations (from z916 analysis)
# ============================================================================

@dataclass
class WorkloadConfig:
    """Workload configuration that controller can select."""
    name: str
    batch_size: int
    seq_len: int
    precision: str

    # Expected efficiency characteristics
    expected_j_per_token: float  # From z916 data
    temp_range: Tuple[float, float]  # Optimal temperature range


# Configurations derived from z916 winners
WORKLOAD_CONFIGS = {
    'eco': WorkloadConfig('eco', 2, 128, 'fp16', 0.0004, (40, 70)),
    'balanced': WorkloadConfig('balanced', 4, 256, 'fp16', 0.0005, (50, 75)),
    'throughput': WorkloadConfig('throughput', 8, 384, 'fp16', 0.0007, (55, 80)),
    'performance': WorkloadConfig('performance', 16, 512, 'fp32', 0.0008, (60, 85)),
}


# ============================================================================
# Workload-Aware Hybrid Controller
# ============================================================================

class WorkloadAwareController:
    """
    Controller that directly selects workload configuration based on hardware state.

    Unlike z917, this controller's decisions CONTROL the workload, not just observe it.
    """

    # Regime detection thresholds
    TEMP_HOT = 75.0
    TEMP_WARM = 60.0
    TEMP_COOL = 50.0

    POWER_HIGH = 200.0
    POWER_MED = 100.0

    def __init__(self, use_memory: bool = True):
        self.current_config = WORKLOAD_CONFIGS['balanced']

        # Memory for learning
        self.memory = AnchoredGraphMemory() if (HAS_MEMORY and use_memory) else None

        # Regime tracking
        self.current_regime = 'steady'
        self.regime_history = deque(maxlen=10)

        # Statistics
        self.step_count = 0
        self.regime_changes = 0
        self.config_selections = {name: 0 for name in WORKLOAD_CONFIGS}

        # Seed memory if available
        if self.memory:
            self._seed_memory()

    def _seed_memory(self):
        """Pre-populate with known-good configurations from z916."""
        seed_cases = [
            # (regime, config_name, state_vector, reward)
            ('hot', 'eco', [0.2, 0.2, 0, 0.8, 0.8, 0, 0.3, 0.3, 0] + [0]*9, 0.95),
            ('warm', 'balanced', [0.4, 0.4, 0, 0.65, 0.65, 0, 0.5, 0.5, 0] + [0]*9, 0.90),
            ('cool', 'throughput', [0.5, 0.5, 0, 0.5, 0.5, 0, 0.6, 0.6, 0] + [0]*9, 0.88),
            ('cold', 'performance', [0.6, 0.6, 0, 0.35, 0.35, 0, 0.7, 0.7, 0] + [0]*9, 0.85),
            ('throttle', 'eco', [0.15, 0.15, 0, 0.9, 0.9, 0, 0.2, 0.2, 0] + [0]*9, 0.92),
        ]

        for regime, config_name, state_vec, reward in seed_cases:
            config = WORKLOAD_CONFIGS[config_name]
            anchor = HardwareAnchor.from_telemetry({
                'power_watts': state_vec[0] * 300,
                'temperature_c': state_vec[3] * 100,
                'energy_mj': 100,
                'utilization': state_vec[6] * 100,
                'profile': config_name,
            }, f'z919_seed_{regime}')

            self.memory.add_control_case(
                body_latent_vector=state_vec,
                action={'config': config_name, 'regime': regime},
                anchor=anchor,
                outcome_metrics={'reward': reward},
            )

    def _detect_regime(self, sample: GpuSample) -> str:
        """Detect current operating regime from telemetry."""
        temp = sample.temp_edge_c
        power = sample.power_w

        # Temperature-based regime
        if temp > self.TEMP_HOT:
            return 'hot'
        elif temp > self.TEMP_WARM:
            return 'warm'
        elif temp > self.TEMP_COOL:
            return 'cool'
        else:
            return 'cold'

    def _select_config(self, regime: str, state: np.ndarray) -> WorkloadConfig:
        """Select best workload config for current regime."""

        # Try memory retrieval first
        if self.memory:
            similar = self.memory.find_similar_states(state.tolist(), top_k=3, min_similarity=0.7)
            if similar:
                best = similar[0]
                config_name = best.get('action', {}).get('config', 'balanced')
                if config_name in WORKLOAD_CONFIGS:
                    return WORKLOAD_CONFIGS[config_name]

        # Fallback: rule-based selection
        regime_to_config = {
            'hot': 'eco',
            'warm': 'balanced',
            'cool': 'throughput',
            'cold': 'performance',
        }
        config_name = regime_to_config.get(regime, 'balanced')
        return WORKLOAD_CONFIGS[config_name]

    def step(self, sample: GpuSample) -> Tuple[int, int, str]:
        """
        Select workload configuration based on current state.

        Returns (batch_size, seq_len, precision) that the workload MUST use.
        """
        self.step_count += 1
        state = encode_state(sample)

        # Detect regime
        new_regime = self._detect_regime(sample)

        # Check for regime change
        if new_regime != self.current_regime:
            self.regime_changes += 1
            self.current_regime = new_regime
            # Select new config on regime change
            self.current_config = self._select_config(new_regime, state)

        self.regime_history.append(new_regime)
        self.config_selections[self.current_config.name] += 1

        return (self.current_config.batch_size,
                self.current_config.seq_len,
                self.current_config.precision)

    def update(self, sample: GpuSample, j_per_token: float, tokens: int):
        """Learn from actual measured efficiency."""
        if not self.memory:
            return

        state = encode_state(sample)

        # Reward based on efficiency
        expected_j = self.current_config.expected_j_per_token
        efficiency_ratio = expected_j / max(j_per_token, 1e-6)
        reward = min(1.0, efficiency_ratio)  # Cap at 1.0

        # Store case
        anchor = HardwareAnchor.from_telemetry({
            'power_watts': sample.power_w,
            'temperature_c': sample.temp_edge_c,
            'energy_mj': int(j_per_token * tokens * 1000),
            'utilization': getattr(sample, 'gpu_busy_pct', 50.0),
            'profile': self.current_config.name,
        }, 'z919_online')

        self.memory.add_control_case(
            body_latent_vector=state.tolist(),
            action={'config': self.current_config.name, 'regime': self.current_regime},
            anchor=anchor,
            outcome_metrics={'reward': reward, 'j_per_token': j_per_token},
        )

    def get_stats(self) -> Dict[str, Any]:
        return {
            'type': 'WorkloadAware',
            'step_count': self.step_count,
            'regime_changes': self.regime_changes,
            'current_regime': self.current_regime,
            'current_config': self.current_config.name,
            'config_selections': self.config_selections,
            'memory_size': len(self.memory.nodes) if self.memory else 0,
        }


# ============================================================================
# Baseline Controllers
# ============================================================================

class FixedController:
    """Always uses same config."""
    def __init__(self, config_name: str = 'balanced'):
        self.config = WORKLOAD_CONFIGS[config_name]
        self.step_count = 0

    def step(self, sample: GpuSample) -> Tuple[int, int, str]:
        self.step_count += 1
        return (self.config.batch_size, self.config.seq_len, self.config.precision)

    def update(self, sample, j_per_token, tokens):
        pass

    def get_stats(self):
        return {'type': f'Fixed-{self.config.name}', 'step_count': self.step_count}


class SimpleAdaptiveController:
    """Simple temperature-based adaptation."""
    def __init__(self):
        self.step_count = 0
        self.temp_history = deque(maxlen=5)

    def step(self, sample: GpuSample) -> Tuple[int, int, str]:
        self.step_count += 1
        self.temp_history.append(sample.temp_edge_c)
        avg_temp = np.mean(list(self.temp_history))

        if avg_temp > 75:
            config = WORKLOAD_CONFIGS['eco']
        elif avg_temp > 60:
            config = WORKLOAD_CONFIGS['balanced']
        elif avg_temp > 50:
            config = WORKLOAD_CONFIGS['throughput']
        else:
            config = WORKLOAD_CONFIGS['performance']

        return (config.batch_size, config.seq_len, config.precision)

    def update(self, sample, j_per_token, tokens):
        pass

    def get_stats(self):
        return {'type': 'SimpleAdaptive', 'step_count': self.step_count}


# ============================================================================
# Workload
# ============================================================================

class TestWorkload:
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


# ============================================================================
# Thermal Cycling Scenario
# ============================================================================

def run_thermal_cycle_benchmark(
    controller,
    workload: TestWorkload,
    telemetry: SysfsHwmonTelemetry,
    duration_sec: float = 120.0,
) -> Dict[str, Any]:
    """
    Benchmark with natural thermal cycling.

    The benchmark runs continuously, and the GPU naturally heats up and cools down
    based on workload. The controller must adapt to these changes.
    """

    torch.cuda.empty_cache()

    # Warmup
    for _ in range(5):
        workload.run_batch(4, 256)

    start_time = time.time()
    total_tokens = 0
    total_energy_j = 0.0
    step_metrics = []

    telemetry.reset_accumulator()
    telemetry.start_continuous_sampling()

    while time.time() - start_time < duration_sec:
        elapsed = time.time() - start_time
        sample = telemetry.read_sample()

        # Controller selects workload config
        batch_size, seq_len, precision = controller.step(sample)

        # Execute workload with controller's selection
        step_start = time.time()
        tokens = workload.run_batch(batch_size, seq_len, precision)
        step_duration = time.time() - step_start

        # Measure energy
        step_energy = sample.power_w * step_duration
        total_tokens += tokens
        total_energy_j += step_energy

        # Calculate efficiency
        j_per_token = step_energy / max(tokens, 1)

        # Update controller with actual measured efficiency
        controller.update(sample, j_per_token, tokens)

        step_metrics.append({
            'time': elapsed,
            'batch_size': batch_size,
            'seq_len': seq_len,
            'precision': precision,
            'tokens': tokens,
            'energy_j': step_energy,
            'j_per_token': j_per_token,
            'temp_c': sample.temp_edge_c,
            'power_w': sample.power_w,
        })

    telemetry.stop_continuous_sampling()

    with telemetry._lock:
        accumulated = telemetry.accumulator.total_energy_j

    if accumulated > 0:
        total_energy_j = accumulated

    elapsed = time.time() - start_time
    j_per_token = total_energy_j / max(total_tokens, 1)
    throughput = total_tokens / elapsed

    # Temperature statistics
    temps = [m['temp_c'] for m in step_metrics]

    return {
        'total_tokens': total_tokens,
        'total_energy_j': total_energy_j,
        'j_per_token': j_per_token,
        'throughput': throughput,
        'min_temp': min(temps),
        'max_temp': max(temps),
        'avg_temp': np.mean(temps),
        'stats': controller.get_stats(),
    }


# ============================================================================
# Main
# ============================================================================

def main():
    print("=" * 70)
    print("z919: WORKLOAD-AWARE HYBRID CONTROLLER")
    print("=" * 70)
    print("\nFixes z917: Controller decisions CONTROL workload, not just observe")
    print("\nKey difference:")
    print("  - z917: Controller observes external workload")
    print("  - z919: Controller SELECTS (batch, seq_len, precision)")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nDevice: {device}")

    telemetry = SysfsHwmonTelemetry()
    workload = TestWorkload(device)

    # Controllers to test
    controllers = {
        'Fixed-eco': FixedController('eco'),
        'Fixed-balanced': FixedController('balanced'),
        'Fixed-performance': FixedController('performance'),
        'SimpleAdaptive': SimpleAdaptiveController(),
        'WorkloadAware': WorkloadAwareController(use_memory=HAS_MEMORY),
    }

    duration_sec = 60.0  # 1 minute per controller
    results = {}

    for name, controller in controllers.items():
        print(f"\n{'='*60}")
        print(f"Testing: {name}")
        print(f"{'='*60}")

        metrics = run_thermal_cycle_benchmark(
            controller, workload, telemetry, duration_sec
        )
        results[name] = metrics

        print(f"  J/token: {metrics['j_per_token']*1000:.3f} mJ/tok")
        print(f"  Throughput: {metrics['throughput']:.0f} tok/s")
        print(f"  Temp range: {metrics['min_temp']:.1f}°C - {metrics['max_temp']:.1f}°C")

        stats = metrics['stats']
        if 'regime_changes' in stats:
            print(f"  Regime changes: {stats['regime_changes']}")
            print(f"  Config selections: {stats['config_selections']}")

    # Analysis
    print("\n" + "=" * 70)
    print("RESULTS (sorted by efficiency)")
    print("=" * 70)

    sorted_results = sorted(results.items(), key=lambda x: x[1]['j_per_token'])

    best_j = sorted_results[0][1]['j_per_token']
    for rank, (name, metrics) in enumerate(sorted_results, 1):
        j_tok = metrics['j_per_token'] * 1000
        vs_best = (metrics['j_per_token'] - best_j) / best_j * 100
        print(f"  {rank}. {name:20s}: {j_tok:.3f} mJ/tok ({vs_best:+.1f}% vs best)")

    # Check if WorkloadAware beats fixed
    workload_aware_j = results['WorkloadAware']['j_per_token']
    best_fixed_j = min(
        results['Fixed-eco']['j_per_token'],
        results['Fixed-balanced']['j_per_token'],
        results['Fixed-performance']['j_per_token'],
    )

    print("\n" + "=" * 70)
    print("VERDICT")
    print("=" * 70)

    if workload_aware_j < best_fixed_j:
        improvement = (best_fixed_j - workload_aware_j) / best_fixed_j * 100
        print(f"WorkloadAware BEATS best fixed by {improvement:.1f}%!")
    else:
        gap = (workload_aware_j - best_fixed_j) / best_fixed_j * 100
        print(f"WorkloadAware loses to best fixed by {gap:.1f}%")
        print("(May need more thermal variation or longer run)")

    # Save results
    results_summary = {
        'benchmark': 'z919_workload_aware',
        'timestamp': datetime.now().isoformat(),
        'duration_sec': duration_sec,
        'controllers': {
            name: {
                'j_per_token': m['j_per_token'],
                'throughput': m['throughput'],
                'temp_range': [m['min_temp'], m['max_temp']],
                'stats': m['stats'],
            }
            for name, m in results.items()
        },
    }

    results_dir = Path(__file__).parent.parent / "results"
    results_dir.mkdir(exist_ok=True)
    results_file = results_dir / "z919_workload_aware.json"

    with open(results_file, 'w') as f:
        json.dump(results_summary, f, indent=2, default=str)

    print(f"\nResults saved to: {results_file}")

    return results_summary


if __name__ == "__main__":
    main()
