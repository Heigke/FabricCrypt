#!/usr/bin/env python3
"""
z920: Comprehensive FEEL Validation
====================================

Fills the gaps identified in DEEP_ARCHITECTURE.md:
1. Memory ACTUALLY integrated in control loop
2. Proper sense->feel->regulate->express->feedback loop
3. Fair comparison of all controller types
4. Measures what ACTUALLY works vs claims

Tests:
1. Steady-state efficiency (where Fixed should win)
2. Variable load adaptation (where Adaptive should win)
3. Memory learning over time (does it improve?)
4. Thermal response (does regulation work?)

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
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple
from collections import deque

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from telemetry.sysfs_hwmon import SysfsHwmonTelemetry, GpuSample

try:
    from memory.anchored_graph import AnchoredGraphMemory, HardwareAnchor, ProvenanceType
    HAS_MEMORY = True
except ImportError:
    HAS_MEMORY = False
    print("Warning: AnchoredGraphMemory not available")


# ============================================================================
# Body State (FEEL Layer)
# ============================================================================

@dataclass
class BodyState:
    """Processed body state from raw telemetry."""
    # Raw values
    power_w: float = 0.0
    temp_c: float = 0.0
    util_pct: float = 0.0

    # EMA smoothed
    power_ema: float = 0.0
    temp_ema: float = 0.0
    util_ema: float = 0.0

    # Derivatives
    power_deriv: float = 0.0
    temp_deriv: float = 0.0

    # Homeostatic
    power_deviation: float = 0.0
    temp_deviation: float = 0.0

    # Energy
    j_per_token: float = 0.0

    def to_vector(self) -> np.ndarray:
        """Convert to 18-dim vector for memory."""
        return np.array([
            self.power_w / 300.0, self.power_ema / 300.0, self.power_deriv / 100.0,
            self.temp_c / 100.0, self.temp_ema / 100.0, self.temp_deriv / 10.0,
            self.util_pct / 100.0, self.util_ema / 100.0, 0.0,
            self.power_deviation, self.temp_deviation, 0.0,
            self.j_per_token * 1000, 0.0, 0.0,
            0.0, 0.0, 0.0,
        ], dtype=np.float32)


class BodyStateTracker:
    """Processes raw telemetry into body state (FEEL layer)."""

    def __init__(self, ema_alpha: float = 0.1, power_setpoint: float = 100.0, temp_setpoint: float = 65.0):
        self.ema_alpha = ema_alpha
        self.power_setpoint = power_setpoint
        self.temp_setpoint = temp_setpoint

        self.state = BodyState()
        self.last_time = time.time()
        self.initialized = False

    def update(self, sample: GpuSample, tokens: int = 0, energy_j: float = 0.0) -> BodyState:
        """Update body state from new telemetry."""
        now = time.time()
        dt = now - self.last_time
        self.last_time = now

        # Raw values
        self.state.power_w = sample.power_w
        self.state.temp_c = sample.temp_edge_c
        self.state.util_pct = getattr(sample, 'gpu_busy_pct', 50.0)

        if not self.initialized:
            self.state.power_ema = sample.power_w
            self.state.temp_ema = sample.temp_edge_c
            self.state.util_ema = self.state.util_pct
            self.initialized = True
        else:
            # EMA update
            alpha = self.ema_alpha
            self.state.power_ema = alpha * sample.power_w + (1 - alpha) * self.state.power_ema
            self.state.temp_ema = alpha * sample.temp_edge_c + (1 - alpha) * self.state.temp_ema
            self.state.util_ema = alpha * self.state.util_pct + (1 - alpha) * self.state.util_ema

            # Derivatives
            if dt > 0:
                self.state.power_deriv = (sample.power_w - self.state.power_ema) / dt
                self.state.temp_deriv = (sample.temp_edge_c - self.state.temp_ema) / dt

        # Homeostatic deviation
        self.state.power_deviation = abs(self.state.power_ema - self.power_setpoint) / self.power_setpoint
        self.state.temp_deviation = abs(self.state.temp_ema - self.temp_setpoint) / self.temp_setpoint

        # Energy efficiency
        if tokens > 0:
            self.state.j_per_token = energy_j / tokens

        return self.state


# ============================================================================
# Controllers (REGULATE Layer)
# ============================================================================

@dataclass
class WorkloadConfig:
    batch_size: int
    seq_len: int
    precision: str


CONFIGS = {
    'eco': WorkloadConfig(2, 128, 'fp16'),
    'balanced': WorkloadConfig(4, 256, 'fp16'),
    'performance': WorkloadConfig(16, 512, 'fp32'),
}


class FixedController:
    """Fixed configuration - no adaptation."""
    def __init__(self, config_name: str = 'balanced'):
        self.config = CONFIGS[config_name]
        self.name = f'Fixed-{config_name}'

    def select(self, body: BodyState) -> WorkloadConfig:
        return self.config

    def update(self, body: BodyState, config: WorkloadConfig, reward: float):
        pass

    def get_stats(self) -> Dict:
        return {'type': self.name}


class AdaptiveController:
    """Temperature-based adaptive controller."""
    def __init__(self):
        self.name = 'Adaptive'

    def select(self, body: BodyState) -> WorkloadConfig:
        temp = body.temp_ema
        if temp > 75:
            return CONFIGS['eco']
        elif temp > 60:
            return CONFIGS['balanced']
        else:
            return CONFIGS['performance']

    def update(self, body: BodyState, config: WorkloadConfig, reward: float):
        pass

    def get_stats(self) -> Dict:
        return {'type': self.name}


class MemoryController:
    """Memory-based controller using AnchoredGraphMemory."""
    def __init__(self):
        self.name = 'Memory'
        self.memory = AnchoredGraphMemory() if HAS_MEMORY else None
        self.step_count = 0
        self.retrieval_count = 0
        self.learning_count = 0

        if self.memory:
            self._seed_memory()

    def _seed_memory(self):
        """Seed with known-good configurations."""
        seeds = [
            # (body_vector, config_name, reward)
            (np.array([0.2]*3 + [0.8]*3 + [0.3]*3 + [0.5]*3 + [0.001]*3 + [0]*3), 'eco', 0.9),
            (np.array([0.4]*3 + [0.6]*3 + [0.5]*3 + [0.2]*3 + [0.0005]*3 + [0]*3), 'balanced', 0.95),
            (np.array([0.6]*3 + [0.4]*3 + [0.7]*3 + [0.1]*3 + [0.0008]*3 + [0]*3), 'performance', 0.85),
        ]

        for vec, config_name, reward in seeds:
            anchor = HardwareAnchor.from_telemetry({
                'power_watts': vec[0] * 300,
                'temperature_c': vec[3] * 100,
                'energy_mj': 100,
                'utilization': vec[6] * 100,
                'profile': config_name,
            }, f'seed_{config_name}')

            self.memory.add_control_case(
                body_latent_vector=vec.tolist(),
                action={'config': config_name, 'reward': reward},
                anchor=anchor,
            )

    def select(self, body: BodyState) -> WorkloadConfig:
        self.step_count += 1

        if not self.memory:
            return CONFIGS['balanced']

        vec = body.to_vector()
        similar = self.memory.find_similar_states(vec.tolist(), top_k=3, min_similarity=0.6)

        if similar:
            self.retrieval_count += 1
            config_name = similar[0].get('action', {}).get('config', 'balanced')
            if config_name in CONFIGS:
                return CONFIGS[config_name]

        # Fallback based on temperature
        if body.temp_ema > 70:
            return CONFIGS['eco']
        return CONFIGS['balanced']

    def update(self, body: BodyState, config: WorkloadConfig, reward: float):
        if not self.memory:
            return

        self.learning_count += 1
        vec = body.to_vector()

        # Find config name
        config_name = 'balanced'
        for name, cfg in CONFIGS.items():
            if cfg.batch_size == config.batch_size:
                config_name = name
                break

        anchor = HardwareAnchor.from_telemetry({
            'power_watts': body.power_w,
            'temperature_c': body.temp_c,
            'energy_mj': int(body.j_per_token * 1000 * config.batch_size * config.seq_len),
            'utilization': body.util_pct,
            'profile': config_name,
        }, 'online')

        self.memory.add_control_case(
            body_latent_vector=vec.tolist(),
            action={'config': config_name, 'reward': reward},
            anchor=anchor,
        )

    def get_stats(self) -> Dict:
        return {
            'type': self.name,
            'step_count': self.step_count,
            'retrieval_count': self.retrieval_count,
            'retrieval_rate': self.retrieval_count / max(1, self.step_count),
            'memory_size': len(self.memory.nodes) if self.memory else 0,
        }


# ============================================================================
# Workload (EXPRESS Layer)
# ============================================================================

class Workload:
    def __init__(self, device: str = "cuda"):
        self.device = device
        self.model = nn.Sequential(
            nn.Embedding(8192, 512),
            nn.Linear(512, 2048),
            nn.GELU(),
            nn.Linear(2048, 512),
            nn.Linear(512, 8192),
        ).to(device)

    def run(self, config: WorkloadConfig) -> Tuple[int, float]:
        """Run workload, return (tokens, elapsed_sec)."""
        x = torch.randint(0, 8192, (config.batch_size, config.seq_len), device=self.device)

        start = time.time()
        if config.precision == "fp32":
            with torch.amp.autocast(device_type='cuda', enabled=False):
                y = self.model(x).mean()
                y.backward()
        else:
            with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                y = self.model(x).mean()
            y.backward()

        torch.cuda.synchronize()
        elapsed = time.time() - start

        return config.batch_size * config.seq_len, elapsed


# ============================================================================
# Test Scenarios
# ============================================================================

def run_test(
    name: str,
    controller,
    workload: Workload,
    telemetry: SysfsHwmonTelemetry,
    body_tracker: BodyStateTracker,
    duration_sec: float = 60.0,
) -> Dict[str, Any]:
    """Run a single test."""
    print(f"\n  Running {name}...")

    torch.cuda.empty_cache()

    # Warmup
    for _ in range(5):
        workload.run(CONFIGS['balanced'])

    start = time.time()
    total_tokens = 0
    total_energy = 0.0
    step_data = []

    telemetry.reset_accumulator()
    telemetry.start_continuous_sampling()

    while time.time() - start < duration_sec:
        elapsed = time.time() - start

        # SENSE: Read telemetry
        sample = telemetry.read_sample()

        # FEEL: Process into body state
        body = body_tracker.update(sample)

        # REGULATE: Select configuration
        config = controller.select(body)

        # EXPRESS: Execute workload
        tokens, step_time = workload.run(config)
        step_energy = sample.power_w * step_time

        total_tokens += tokens
        total_energy += step_energy

        # FEEDBACK: Update body state with results
        body = body_tracker.update(sample, tokens, step_energy)

        # Reward: negative J/token (lower is better)
        j_per_tok = step_energy / tokens
        reward = 1.0 / (1.0 + j_per_tok * 1000)  # Higher reward for lower J/tok

        # LEARN: Update controller
        controller.update(body, config, reward)

        step_data.append({
            'time': elapsed,
            'tokens': tokens,
            'energy_j': step_energy,
            'j_per_token': j_per_tok,
            'temp_c': sample.temp_edge_c,
            'config': config.batch_size,
        })

    telemetry.stop_continuous_sampling()

    with telemetry._lock:
        accumulated = telemetry.accumulator.total_energy_j
    if accumulated > 0:
        total_energy = accumulated

    j_per_token = total_energy / max(total_tokens, 1)
    throughput = total_tokens / duration_sec

    temps = [s['temp_c'] for s in step_data]

    return {
        'total_tokens': total_tokens,
        'total_energy_j': total_energy,
        'j_per_token': j_per_token,
        'throughput': throughput,
        'min_temp': min(temps) if temps else 0,
        'max_temp': max(temps) if temps else 0,
        'avg_temp': np.mean(temps) if temps else 0,
        'stats': controller.get_stats(),
    }


def run_thermal_stress_test(
    controller,
    workload: Workload,
    telemetry: SysfsHwmonTelemetry,
    body_tracker: BodyStateTracker,
    duration_sec: float = 60.0,
) -> Dict[str, Any]:
    """Test with artificial thermal cycling."""
    print(f"\n  Running thermal stress test...")

    torch.cuda.empty_cache()

    start = time.time()
    total_tokens = 0
    total_energy = 0.0
    stress_periods = []

    telemetry.reset_accumulator()
    telemetry.start_continuous_sampling()

    # Alternate: 20s heavy -> 10s light -> 20s heavy -> 10s light
    while time.time() - start < duration_sec:
        elapsed = time.time() - start
        cycle = int(elapsed / 30) % 2  # 30-second cycles

        sample = telemetry.read_sample()
        body = body_tracker.update(sample)

        # Controller selects
        config = controller.select(body)

        # But we force heavy load during stress periods
        if cycle == 0 and elapsed % 30 < 20:
            # Heavy period - use what controller says but run extra
            tokens, step_time = workload.run(config)
            # Add extra load
            for _ in range(2):
                workload.run(CONFIGS['performance'])
        else:
            # Light period
            tokens, step_time = workload.run(config)

        step_energy = sample.power_w * step_time
        total_tokens += tokens
        total_energy += step_energy

        body = body_tracker.update(sample, tokens, step_energy)
        j_per_tok = step_energy / tokens
        reward = 1.0 / (1.0 + j_per_tok * 1000)
        controller.update(body, config, reward)

        stress_periods.append({
            'time': elapsed,
            'cycle': cycle,
            'temp_c': sample.temp_edge_c,
            'config': config.batch_size,
        })

    telemetry.stop_continuous_sampling()

    with telemetry._lock:
        accumulated = telemetry.accumulator.total_energy_j
    if accumulated > 0:
        total_energy = accumulated

    j_per_token = total_energy / max(total_tokens, 1)

    # Analyze thermal response
    temps = [s['temp_c'] for s in stress_periods]

    return {
        'total_tokens': total_tokens,
        'j_per_token': j_per_token,
        'throughput': total_tokens / duration_sec,
        'temp_range': [min(temps), max(temps)],
        'stats': controller.get_stats(),
    }


# ============================================================================
# Main
# ============================================================================

def main():
    print("=" * 70)
    print("z920: COMPREHENSIVE FEEL VALIDATION")
    print("=" * 70)
    print("\nFilling gaps identified in DEEP_ARCHITECTURE.md:")
    print("  1. Memory ACTUALLY integrated in control loop")
    print("  2. Proper sense->feel->regulate->express->feedback")
    print("  3. Fair comparison of all controller types")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nDevice: {device}")
    print(f"Memory available: {HAS_MEMORY}")

    telemetry = SysfsHwmonTelemetry()
    workload = Workload(device)

    duration = 45.0  # 45 seconds per test

    results = {}

    # Test 1: Steady-state efficiency
    print("\n" + "=" * 60)
    print("TEST 1: STEADY-STATE EFFICIENCY")
    print("=" * 60)
    print("Expected: Fixed-balanced should win (no adaptation overhead)")

    for name, controller in [
        ('Fixed-eco', FixedController('eco')),
        ('Fixed-balanced', FixedController('balanced')),
        ('Adaptive', AdaptiveController()),
        ('Memory', MemoryController()),
    ]:
        body_tracker = BodyStateTracker()
        results[f'steady_{name}'] = run_test(
            name, controller, workload, telemetry, body_tracker, duration
        )
        print(f"    {name}: {results[f'steady_{name}']['j_per_token']*1000:.3f} mJ/tok")

    # Test 2: Thermal stress
    print("\n" + "=" * 60)
    print("TEST 2: THERMAL STRESS (Variable Load)")
    print("=" * 60)
    print("Expected: Adaptive controllers should show benefit")

    for name, controller in [
        ('Fixed-balanced', FixedController('balanced')),
        ('Adaptive', AdaptiveController()),
        ('Memory', MemoryController()),
    ]:
        body_tracker = BodyStateTracker()
        results[f'thermal_{name}'] = run_thermal_stress_test(
            controller, workload, telemetry, body_tracker, duration
        )
        print(f"    {name}: {results[f'thermal_{name}']['j_per_token']*1000:.3f} mJ/tok")

    # Analysis
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)

    print("\nSteady-State (lower is better):")
    steady_results = [(k, v) for k, v in results.items() if k.startswith('steady_')]
    for name, data in sorted(steady_results, key=lambda x: x[1]['j_per_token']):
        j = data['j_per_token'] * 1000
        print(f"  {name.replace('steady_', ''):20s}: {j:.3f} mJ/tok")

    print("\nThermal Stress (lower is better):")
    thermal_results = [(k, v) for k, v in results.items() if k.startswith('thermal_')]
    for name, data in sorted(thermal_results, key=lambda x: x[1]['j_per_token']):
        j = data['j_per_token'] * 1000
        temp_range = data.get('temp_range', [0, 0])
        print(f"  {name.replace('thermal_', ''):20s}: {j:.3f} mJ/tok (temp: {temp_range[0]:.0f}-{temp_range[1]:.0f}°C)")

    # Check if memory learning helps
    print("\n" + "=" * 70)
    print("MEMORY LEARNING CHECK")
    print("=" * 70)

    memory_stats = results.get('steady_Memory', {}).get('stats', {})
    if memory_stats:
        print(f"  Memory size: {memory_stats.get('memory_size', 0)}")
        print(f"  Retrieval rate: {memory_stats.get('retrieval_rate', 0)*100:.1f}%")
        print(f"  Steps: {memory_stats.get('step_count', 0)}")

    # Verdict
    print("\n" + "=" * 70)
    print("VERDICT")
    print("=" * 70)

    steady_fixed = results.get('steady_Fixed-balanced', {}).get('j_per_token', 1)
    steady_memory = results.get('steady_Memory', {}).get('j_per_token', 1)
    thermal_fixed = results.get('thermal_Fixed-balanced', {}).get('j_per_token', 1)
    thermal_adaptive = results.get('thermal_Adaptive', {}).get('j_per_token', 1)

    if steady_memory < steady_fixed:
        print("  [UNEXPECTED] Memory beats Fixed in steady-state!")
    else:
        overhead = (steady_memory - steady_fixed) / steady_fixed * 100
        print(f"  [EXPECTED] Fixed beats Memory in steady-state by {overhead:.1f}%")

    if thermal_adaptive < thermal_fixed:
        improvement = (thermal_fixed - thermal_adaptive) / thermal_fixed * 100
        print(f"  [EXPECTED] Adaptive beats Fixed in thermal stress by {improvement:.1f}%")
    else:
        print("  [UNEXPECTED] Fixed beats Adaptive in thermal stress")

    # Save results
    output = {
        'benchmark': 'z920_comprehensive',
        'timestamp': datetime.now().isoformat(),
        'duration_per_test_sec': duration,
        'results': {
            k: {
                'j_per_token': v['j_per_token'],
                'throughput': v['throughput'],
                'stats': v['stats'],
            }
            for k, v in results.items()
        },
    }

    results_dir = Path(__file__).parent.parent / "results"
    results_dir.mkdir(exist_ok=True)
    results_file = results_dir / "z920_comprehensive.json"

    with open(results_file, 'w') as f:
        json.dump(output, f, indent=2, default=str)

    print(f"\nResults saved to: {results_file}")

    return output


if __name__ == "__main__":
    main()
