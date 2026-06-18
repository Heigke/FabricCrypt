#!/usr/bin/env python3
"""
z917: Sleep-Wake Hybrid Controller
===================================

Solves the z916 paradox: "Adaptation has value but high overhead"

Key insight from z916:
- Fixed-MED wins in steady-state (zero overhead)
- Adaptive wins in variable load (+19.4% in mixed workload)
- CaseBased fails cold-start (no pre-populated memory)

Solution: Hybrid "Sleep-Wake" Architecture
- SLEEP: Run Fixed policy (zero overhead) when conditions stable
- WAKE: Query CaseBased memory only when regime shift detected
- SEED: Pre-populate memory with z916 winners for instant knowledge

Uses v10 Somatic Nervous System as low-cost "wake trigger":
- DifferentialDiagnosis detects FLOW/FEVER/STRAIN transitions
- Only run expensive retrieval when state actually changes

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

# Import somatic diagnosis if available
try:
    from dsi.diagnosis import DifferentialDiagnosis, SomaticSignature, InternalState
    HAS_DIAGNOSIS = True
except ImportError:
    HAS_DIAGNOSIS = False
    print("Warning: DifferentialDiagnosis not available, using simple delta trigger")

# Import memory if available
try:
    from memory.anchored_graph import AnchoredGraphMemory, HardwareAnchor, ProvenanceType
    HAS_MEMORY = True
except ImportError:
    HAS_MEMORY = False
    print("Warning: AnchoredGraphMemory not available")


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


def sample_to_somatic(sample: GpuSample, variance: float = 0.1) -> 'SomaticSignature':
    """Convert GPU sample to somatic signature for diagnosis."""
    if not HAS_DIAGNOSIS:
        return None
    return SomaticSignature(
        thermal=sample.temp_edge_c / 100.0,
        metabolic=sample.power_w / 300.0,
        cognitive=0.7,  # Assume moderate coherence
        variance=variance,
        fatigue=0.0,
        recovery_rate=0.0,
    )


# ============================================================================
# Fixed Policies (from z916 winners)
# ============================================================================

class PolicyLevel(Enum):
    ECO = 0   # Low power, efficient for thermal stress
    LOW = 1
    MED = 2   # Default, good steady-state
    HIGH = 3
    PERF = 4  # Maximum, good for bursty load


@dataclass
class FixedPolicy:
    """A fixed policy configuration."""
    level: PolicyLevel
    batch_size: int
    seq_len: int
    precision: str

    def execute(self) -> Tuple[int, int, str]:
        return (self.batch_size, self.seq_len, self.precision)


# z916 winners for each regime
Z916_WINNERS = {
    'steady': FixedPolicy(PolicyLevel.MED, 4, 256, 'fp16'),
    'thermal_stress': FixedPolicy(PolicyLevel.ECO, 2, 128, 'fp16'),
    'bursty_heavy': FixedPolicy(PolicyLevel.PERF, 16, 512, 'fp32'),
    'bursty_light': FixedPolicy(PolicyLevel.ECO, 2, 128, 'fp16'),
    'mixed': FixedPolicy(PolicyLevel.MED, 4, 256, 'fp16'),
}


# ============================================================================
# Sleep-Wake Hybrid Controller
# ============================================================================

class SleepWakeController:
    """
    Hybrid controller that sleeps (uses fixed policy) until wake trigger fires.

    Wake triggers (from v10 Somatic System):
    1. Somatic state transition (FLOW -> FEVER, etc.)
    2. Temperature delta > threshold
    3. Power delta > threshold
    4. Throttle flag change
    """

    # Wake trigger thresholds
    TEMP_DELTA_THRESHOLD = 10.0  # °C change
    POWER_DELTA_THRESHOLD = 30.0  # W change
    THROTTLE_TEMP = 80.0  # °C

    def __init__(self, use_memory: bool = True):
        # Current policy (starts with MED)
        self.current_policy = Z916_WINNERS['steady']
        self.policy_name = 'steady'

        # Somatic diagnosis
        self.diagnosis = DifferentialDiagnosis() if HAS_DIAGNOSIS else None
        self.last_state: Optional[InternalState] = None

        # Simple delta tracking (fallback if no diagnosis)
        self.last_temp = 0.0
        self.last_power = 0.0
        self.last_throttle = False

        # Memory for learning
        self.memory = AnchoredGraphMemory() if (HAS_MEMORY and use_memory) else None
        self.memory_seeded = False

        # Statistics
        self.wake_count = 0
        self.sleep_count = 0
        self.step_count = 0

        # Variance tracking for somatic signature
        self.power_history = deque(maxlen=10)

    def seed_memory(self):
        """Pre-populate memory with z916 winning configurations."""
        if not self.memory:
            return

        print("  Seeding memory with z916 winners...")

        # Create synthetic state vectors for each regime
        seed_cases = [
            # (state_vector, policy_name, action_level, reward)
            # Steady state - MED works best
            (np.array([0.3, 0.3, 0, 0.55, 0.55, 0, 0.5, 0.5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
             'steady', 2, 0.9),

            # Thermal stress - ECO works best
            (np.array([0.2, 0.2, 0, 0.75, 0.75, 0, 0.3, 0.3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
             'thermal_stress', 0, 0.95),

            # Bursty heavy - PERF works best
            (np.array([0.8, 0.8, 0, 0.6, 0.6, 0, 0.9, 0.9, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
             'bursty_heavy', 4, 0.85),

            # Bursty light - ECO works best
            (np.array([0.15, 0.15, 0, 0.45, 0.45, 0, 0.2, 0.2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
             'bursty_light', 0, 0.92),

            # Mixed/variable - need adaptive
            (np.array([0.5, 0.5, 0, 0.65, 0.65, 0, 0.6, 0.6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
             'mixed', 2, 0.88),

            # High temp, backing off
            (np.array([0.25, 0.25, 0, 0.85, 0.85, 0, 0.4, 0.4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
             'thermal_throttle', 0, 0.90),

            # Cold, can push harder
            (np.array([0.4, 0.4, 0, 0.35, 0.35, 0, 0.5, 0.5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
             'cold_push', 4, 0.85),
        ]

        for state_vec, regime, action_level, reward in seed_cases:
            anchor = HardwareAnchor.from_telemetry({
                'power_watts': state_vec[0] * 300,
                'temperature_c': state_vec[3] * 100,
                'energy_mj': 100,
                'utilization': state_vec[6] * 100,
                'profile': ['ECO', 'LOW', 'MED', 'HIGH', 'PERF'][action_level],
            }, f'z916_seed_{regime}')

            self.memory.add_control_case(
                body_latent_vector=state_vec.tolist(),
                action={'level': action_level, 'regime': regime},
                anchor=anchor,
                outcome_metrics={'reward': reward},
            )

        self.memory_seeded = True
        print(f"  Seeded {len(seed_cases)} archetype cases")

    def _detect_wake_trigger(self, sample: GpuSample) -> Tuple[bool, str]:
        """
        Detect if we need to wake up and query memory.

        Returns (should_wake, reason)
        """
        reasons = []

        # Track power variance
        self.power_history.append(sample.power_w)
        variance = np.var(list(self.power_history)) if len(self.power_history) > 3 else 0.1

        # Tier 1: Somatic state transition (if available)
        if self.diagnosis:
            sig = sample_to_somatic(sample, variance=min(1.0, variance / 100))
            current_state = self.diagnosis.diagnose(sig)

            if self.last_state is not None and current_state != self.last_state:
                # State transition detected!
                reasons.append(f"somatic:{self.last_state.value}->{current_state.value}")
                self.last_state = current_state
                return True, reasons[0]
            self.last_state = current_state

        # Tier 2: Temperature delta
        temp_delta = abs(sample.temp_edge_c - self.last_temp)
        if temp_delta > self.TEMP_DELTA_THRESHOLD:
            reasons.append(f"temp_delta:{temp_delta:.1f}C")

        # Tier 3: Power delta
        power_delta = abs(sample.power_w - self.last_power)
        if power_delta > self.POWER_DELTA_THRESHOLD:
            reasons.append(f"power_delta:{power_delta:.1f}W")

        # Tier 4: Throttle flag change
        is_throttling = sample.temp_edge_c > self.THROTTLE_TEMP
        if is_throttling != self.last_throttle:
            reasons.append(f"throttle:{'ON' if is_throttling else 'OFF'}")

        # Update history
        self.last_temp = sample.temp_edge_c
        self.last_power = sample.power_w
        self.last_throttle = is_throttling

        if reasons:
            return True, "; ".join(reasons)
        return False, ""

    def _select_policy_from_state(self, sample: GpuSample, state: np.ndarray) -> FixedPolicy:
        """Select best policy based on current state."""

        # Try memory retrieval first
        if self.memory and self.memory_seeded:
            similar = self.memory.find_similar_states(state.tolist(), top_k=3, min_similarity=0.7)
            if similar:
                best = similar[0]
                action = best.get('action', {})
                regime = action.get('regime', 'steady')
                if regime in Z916_WINNERS:
                    return Z916_WINNERS[regime]

        # Fallback: rule-based selection
        temp_c = sample.temp_edge_c
        power_w = sample.power_w

        if temp_c > 80:
            return Z916_WINNERS['thermal_stress']
        elif temp_c < 50 and power_w < 100:
            return Z916_WINNERS['bursty_heavy']
        elif power_w > 200:
            return Z916_WINNERS['bursty_heavy']
        else:
            return Z916_WINNERS['steady']

    def step(self, sample: GpuSample) -> Tuple[int, int, str]:
        """
        Main control step - Sleep/Wake hybrid logic.

        Returns (batch_size, seq_len, precision)
        """
        self.step_count += 1
        state = encode_state(sample)

        # Check wake trigger
        should_wake, reason = self._detect_wake_trigger(sample)

        if should_wake:
            # WAKE: Query memory and select new policy
            self.wake_count += 1
            self.current_policy = self._select_policy_from_state(sample, state)
        else:
            # SLEEP: Keep current policy (zero overhead)
            self.sleep_count += 1

        return self.current_policy.execute()

    def update(self, sample: GpuSample, action: int, reward: float):
        """Store successful action in memory for future retrieval."""
        if not self.memory:
            return

        state = encode_state(sample)
        anchor = HardwareAnchor.from_telemetry({
            'power_watts': sample.power_w,
            'temperature_c': sample.temp_edge_c,
            'energy_mj': 100,
            'utilization': getattr(sample, 'gpu_busy_pct', 50.0),
            'profile': ['ECO', 'LOW', 'MED', 'HIGH', 'PERF'][action],
        }, 'z917_online')

        self.memory.add_control_case(
            body_latent_vector=state.tolist(),
            action={'level': action, 'reward': reward},
            anchor=anchor,
            outcome_metrics={'reward': reward},
        )

    def get_stats(self) -> Dict[str, Any]:
        return {
            'type': 'SleepWake',
            'step_count': self.step_count,
            'wake_count': self.wake_count,
            'sleep_count': self.sleep_count,
            'wake_rate': self.wake_count / max(1, self.step_count),
            'memory_size': len(self.memory.nodes) if self.memory else 0,
            'memory_seeded': self.memory_seeded,
        }


# ============================================================================
# Baseline Controllers for Comparison
# ============================================================================

class FixedController:
    """Fixed policy - no adaptation."""
    def __init__(self, level: int = 2):
        self.level = level
        self.policies = [
            (2, 128, 'fp16'),   # ECO
            (4, 192, 'fp16'),   # LOW
            (4, 256, 'fp16'),   # MED
            (8, 384, 'fp16'),   # HIGH
            (16, 512, 'fp32'),  # PERF
        ]
        self.name = ['Fixed-ECO', 'Fixed-LOW', 'Fixed-MED', 'Fixed-HIGH', 'Fixed-PERF'][level]
        self.step_count = 0

    def step(self, sample: GpuSample) -> Tuple[int, int, str]:
        self.step_count += 1
        return self.policies[self.level]

    def seed_memory(self):
        pass

    def update(self, sample, action, reward):
        pass

    def get_stats(self):
        return {'type': self.name, 'step_count': self.step_count}


class AdaptiveController:
    """Simple adaptive controller - adjusts based on temperature."""
    def __init__(self):
        self.current_level = 2
        self.temp_history = []
        self.step_count = 0
        self.policies = [
            (2, 128, 'fp16'),
            (4, 192, 'fp16'),
            (4, 256, 'fp16'),
            (8, 384, 'fp16'),
            (16, 512, 'fp32'),
        ]

    def step(self, sample: GpuSample) -> Tuple[int, int, str]:
        self.step_count += 1
        temp_c = sample.temp_edge_c

        self.temp_history.append(temp_c)
        if len(self.temp_history) > 10:
            self.temp_history = self.temp_history[-10:]

        avg_temp = np.mean(self.temp_history)

        if avg_temp > 75:
            self.current_level = max(0, self.current_level - 1)
        elif avg_temp < 55:
            self.current_level = min(4, self.current_level + 1)

        return self.policies[self.current_level]

    def seed_memory(self):
        pass

    def update(self, sample, action, reward):
        pass

    def get_stats(self):
        return {'type': 'Adaptive', 'current_level': self.current_level, 'step_count': self.step_count}


# ============================================================================
# Workload Simulation
# ============================================================================

class TestWorkload:
    """Test workload for benchmarking."""

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
# "Day in Life" Benchmark Pattern
# ============================================================================

def day_in_life_pattern(elapsed_sec: float, temp_c: float) -> str:
    """
    10-minute "Day in Life" scenario that tests transitions.

    Returns the current regime name.
    """
    minute = elapsed_sec / 60

    if minute < 2:
        # Morning calm: steady, low-load
        return "steady"
    elif minute < 3:
        # The Burst: sudden spike
        return "bursty_heavy"
    elif minute < 7:
        # The Heatwave: sustained load, thermal throttling
        if temp_c > 75:
            return "thermal_stress"
        return "sustained"
    elif minute < 8:
        # The Cooldown: load drops
        return "cooldown"
    else:
        # Chaos: random mixed workload
        return "mixed"


def get_workload_for_regime(regime: str, temp_c: float) -> Tuple[int, int, str]:
    """Get workload parameters for a regime."""
    if regime == "steady":
        return (4, 256, 'fp16')
    elif regime == "bursty_heavy":
        return (16, 512, 'fp32')
    elif regime == "thermal_stress":
        return (2, 128, 'fp16')
    elif regime == "sustained":
        return (8, 384, 'fp16')
    elif regime == "cooldown":
        return (2, 128, 'fp16')
    elif regime == "mixed":
        # Random variation
        batch = np.random.choice([2, 4, 8, 16])
        seq = np.random.choice([128, 256, 384, 512])
        prec = 'fp32' if np.random.random() < 0.2 else 'fp16'
        return (batch, seq, prec)
    return (4, 256, 'fp16')


# ============================================================================
# Benchmark Runner
# ============================================================================

def run_benchmark(
    controller,
    workload: TestWorkload,
    telemetry: SysfsHwmonTelemetry,
    duration_sec: float = 600.0,  # 10 minutes
    use_day_in_life: bool = True,
) -> Dict[str, Any]:
    """Run benchmark with controller."""

    controller.seed_memory()

    torch.cuda.empty_cache()

    # Warmup
    for _ in range(5):
        workload.run_batch(4, 256)

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

        # Get regime and workload
        if use_day_in_life:
            regime = day_in_life_pattern(elapsed, sample.temp_edge_c)
            external_batch, external_seq, external_prec = get_workload_for_regime(regime, sample.temp_edge_c)
        else:
            external_batch, external_seq, external_prec = 4, 256, 'fp16'

        # Controller decides (may override if adaptive)
        batch_size, seq_len, precision = controller.step(sample)

        # Run workload with external load (simulates request pattern)
        step_start = time.time()
        tokens = workload.run_batch(external_batch, external_seq, external_prec)
        step_duration = time.time() - step_start

        # Energy
        step_energy = sample.power_w * step_duration
        total_tokens += tokens
        total_energy_j += step_energy

        # Reward
        j_per_token = step_energy / max(tokens, 1)
        temp_penalty = max(0, (sample.temp_edge_c - 70) / 30.0) * 0.01
        reward = -j_per_token - temp_penalty

        # Update controller
        controller.update(sample, 2, reward)

        step_metrics.append({
            'time': elapsed,
            'regime': regime if use_day_in_life else 'steady',
            'tokens': tokens,
            'energy_j': step_energy,
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

    return {
        'total_tokens': total_tokens,
        'total_energy_j': total_energy_j,
        'j_per_token': j_per_token,
        'throughput': throughput,
        'stats': controller.get_stats(),
        'step_metrics': step_metrics,
    }


# ============================================================================
# Main
# ============================================================================

def main():
    print("=" * 70)
    print("z917: SLEEP-WAKE HYBRID CONTROLLER")
    print("=" * 70)
    print("\nSolving the z916 paradox: Efficient as Fixed, Smart as Adaptive")
    print("\nKey innovations:")
    print("  1. SEED: Pre-populate memory with z916 winners (no cold-start)")
    print("  2. SLEEP: Use Fixed-MED during steady state (zero overhead)")
    print("  3. WAKE: Query memory only on regime shift (v10 somatic trigger)")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nDevice: {device}")

    telemetry = SysfsHwmonTelemetry()
    workload = TestWorkload(device)

    # Controllers to test
    controllers = {
        'Fixed-MED': FixedController(2),
        'Adaptive': AdaptiveController(),
        'SleepWake': SleepWakeController(use_memory=HAS_MEMORY),
    }

    # Duration per controller
    duration_sec = 120.0  # 2 minutes for quick test

    results = {}

    for name, controller in controllers.items():
        print(f"\n{'='*60}")
        print(f"Testing: {name}")
        print(f"{'='*60}")

        metrics = run_benchmark(
            controller, workload, telemetry,
            duration_sec=duration_sec,
            use_day_in_life=True,
        )

        results[name] = metrics
        stats = metrics['stats']

        print(f"  J/token: {metrics['j_per_token']*1000:.3f} mJ/tok")
        print(f"  Throughput: {metrics['throughput']:.0f} tok/s")

        if 'wake_rate' in stats:
            print(f"  Wake rate: {stats['wake_rate']*100:.1f}% ({stats['wake_count']} wakes)")
            print(f"  Memory size: {stats.get('memory_size', 0)}")

    # Analysis
    print("\n" + "=" * 70)
    print("COMPARATIVE ANALYSIS")
    print("=" * 70)

    sorted_results = sorted(results.items(), key=lambda x: x[1]['j_per_token'])

    for rank, (name, metrics) in enumerate(sorted_results, 1):
        j_tok = metrics['j_per_token'] * 1000
        print(f"  {rank}. {name:15s}: {j_tok:.3f} mJ/tok")

    # Check success criteria
    print("\n" + "=" * 70)
    print("SUCCESS CRITERIA CHECK")
    print("=" * 70)

    fixed_j = results['Fixed-MED']['j_per_token']
    adaptive_j = results['Adaptive']['j_per_token']
    sleepwake_j = results['SleepWake']['j_per_token']

    # Criterion 1: SleepWake overhead < 5% vs Fixed-MED in steady phase
    # (Need to analyze step_metrics for this - simplified here)
    overhead_vs_fixed = (sleepwake_j - fixed_j) / fixed_j * 100
    print(f"\n1. SleepWake vs Fixed-MED overhead: {overhead_vs_fixed:+.1f}%")

    # Criterion 2: SleepWake matches or beats Adaptive
    vs_adaptive = (adaptive_j - sleepwake_j) / adaptive_j * 100
    print(f"2. SleepWake vs Adaptive improvement: {vs_adaptive:+.1f}%")

    # Criterion 3: Total energy savings
    if sleepwake_j < min(fixed_j, adaptive_j):
        print("3. SleepWake BEATS BOTH! Best of both worlds achieved.")
    elif sleepwake_j < max(fixed_j, adaptive_j):
        print("3. SleepWake beats one baseline.")
    else:
        print("3. SleepWake needs tuning.")

    # Save results
    results_summary = {
        'benchmark': 'z917_sleep_wake',
        'timestamp': datetime.now().isoformat(),
        'duration_sec': duration_sec,
        'controllers': {
            name: {
                'j_per_token': m['j_per_token'],
                'throughput': m['throughput'],
                'stats': m['stats'],
            }
            for name, m in results.items()
        },
    }

    results_dir = Path(__file__).parent.parent / "results"
    results_dir.mkdir(exist_ok=True)
    results_file = results_dir / "z917_sleep_wake.json"

    with open(results_file, 'w') as f:
        json.dump(results_summary, f, indent=2, default=str)

    print(f"\nResults saved to: {results_file}")

    return results_summary


if __name__ == "__main__":
    main()
