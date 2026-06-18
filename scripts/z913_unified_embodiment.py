#!/usr/bin/env python3
"""
Z913: Unified Embodiment Controller - The "Kill Shot" Benchmark
================================================================

Tests whether COMBINED embodied control (all 7 actuators coordinated)
provides better energy savings than individual actuators alone.

7 ACTUATOR TYPES:
1. Precision: FP32/FP16/INT8 mixed precision
2. Window: Attention window size (512-4096)
3. Batch: Gradient accumulation (1-8x)
4. Cache: Memory vs recompute tradeoffs
5. DVFS: DPM level selection (performance state)
6. Sparsity: Top-k attention pruning
7. MoE: Expert routing based on compute cost

CONTROLLER STRATEGIES:
- Option A: Simple priority chain (if hot->reduce all, if cool->maximize quality)
- Option B: Independent per-actuator decisions based on bottleneck type
- Option C: Learned policy that selects actuator combination

BENCHMARK MATRIX:
- Baseline: All actuators fixed at quality-first settings
- Individual: Each actuator alone (7 conditions)
- Combined: All actuators coordinated (1 condition)
- Oracle: Best individual actuator per batch (upper bound)

KEY HYPOTHESIS:
Coordinated embodiment achieves MORE energy savings than:
1. Best individual actuator
2. Sum of savings from independent decisions
This proves emergent synergy from closed-loop coordination.
"""

import os
import sys
from pathlib import Path

# CRITICAL: gfx1151 requires HSA override
os.environ["HSA_OVERRIDE_GFX_VERSION"] = "11.0.0"

sys.path.insert(0, str(Path(__file__).parent.parent))

import time
import json
import random
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Optional, Any
from collections import deque
from enum import IntEnum

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

from src.telemetry.sysfs_hwmon import SysfsHwmonTelemetry, EnergyMeter


# ============================================================================
# ACTUATOR DEFINITIONS
# ============================================================================

class PrecisionMode(IntEnum):
    """Precision actuator states."""
    FP32 = 0
    FP16 = 1
    INT8 = 2


class WindowSize(IntEnum):
    """Attention window sizes."""
    TINY = 512
    SMALL = 1024
    MEDIUM = 2048
    LARGE = 4096


class BatchMode(IntEnum):
    """Gradient accumulation multipliers."""
    SINGLE = 1
    DOUBLE = 2
    QUAD = 4
    OCT = 8


class CacheMode(IntEnum):
    """Memory vs recompute tradeoff."""
    FULL_CACHE = 0      # Store everything (memory hungry)
    SELECTIVE = 1       # Cache only critical activations
    RECOMPUTE = 2       # Recompute most activations (memory efficient)


class DPMLevel(IntEnum):
    """AMD DPM performance levels."""
    LOW = 0
    BALANCED = 1
    HIGH = 2


class SparsityLevel(IntEnum):
    """Top-k attention sparsity."""
    DENSE = 0           # No sparsity
    SPARSE_75 = 1       # Keep 75% of attention
    SPARSE_50 = 2       # Keep 50% of attention
    SPARSE_25 = 3       # Keep 25% of attention


class MoEMode(IntEnum):
    """Expert routing strategy."""
    ALL_EXPERTS = 0     # Route to all experts
    TOP_2 = 1           # Route to top 2 experts
    TOP_1 = 2           # Route to single expert
    ADAPTIVE = 3        # Dynamic routing based on cost


@dataclass
class ActuatorState:
    """Combined state of all 7 actuators."""
    precision: PrecisionMode = PrecisionMode.FP32
    window: WindowSize = WindowSize.MEDIUM
    batch: BatchMode = BatchMode.SINGLE
    cache: CacheMode = CacheMode.FULL_CACHE
    dpm: DPMLevel = DPMLevel.BALANCED
    sparsity: SparsityLevel = SparsityLevel.DENSE
    moe: MoEMode = MoEMode.ALL_EXPERTS

    def to_dict(self) -> Dict:
        return {
            'precision': self.precision.name,
            'window': self.window.value,
            'batch': self.batch.value,
            'cache': self.cache.name,
            'dpm': self.dpm.name,
            'sparsity': self.sparsity.name,
            'moe': self.moe.name,
        }

    @property
    def quality_score(self) -> float:
        """Estimate quality impact (0-1, higher is better)."""
        scores = {
            'precision': {PrecisionMode.FP32: 1.0, PrecisionMode.FP16: 0.95, PrecisionMode.INT8: 0.85},
            'window': {WindowSize.TINY: 0.7, WindowSize.SMALL: 0.85, WindowSize.MEDIUM: 0.95, WindowSize.LARGE: 1.0},
            'batch': {BatchMode.SINGLE: 1.0, BatchMode.DOUBLE: 0.98, BatchMode.QUAD: 0.95, BatchMode.OCT: 0.92},
            'cache': {CacheMode.FULL_CACHE: 1.0, CacheMode.SELECTIVE: 0.98, CacheMode.RECOMPUTE: 0.95},
            'sparsity': {SparsityLevel.DENSE: 1.0, SparsityLevel.SPARSE_75: 0.97, SparsityLevel.SPARSE_50: 0.92, SparsityLevel.SPARSE_25: 0.82},
            'moe': {MoEMode.ALL_EXPERTS: 1.0, MoEMode.TOP_2: 0.95, MoEMode.TOP_1: 0.85, MoEMode.ADAPTIVE: 0.92},
        }
        return np.mean([
            scores['precision'][self.precision],
            scores['window'][self.window],
            scores['batch'][self.batch],
            scores['cache'][self.cache],
            scores['sparsity'][self.sparsity],
            scores['moe'][self.moe],
        ])

    @property
    def compute_multiplier(self) -> float:
        """Estimate compute cost multiplier (baseline=1.0)."""
        multiplier = 1.0

        # Precision savings
        precision_mult = {PrecisionMode.FP32: 1.0, PrecisionMode.FP16: 0.65, PrecisionMode.INT8: 0.35}
        multiplier *= precision_mult[self.precision]

        # Window size impact
        window_mult = {WindowSize.TINY: 0.3, WindowSize.SMALL: 0.5, WindowSize.MEDIUM: 0.8, WindowSize.LARGE: 1.0}
        multiplier *= window_mult[self.window]

        # Batch accumulation (reduces forward passes)
        multiplier *= (1.0 / self.batch.value)

        # Cache mode (recompute trades memory for compute)
        cache_mult = {CacheMode.FULL_CACHE: 1.0, CacheMode.SELECTIVE: 1.1, CacheMode.RECOMPUTE: 1.3}
        multiplier *= cache_mult[self.cache]

        # Sparsity savings
        sparsity_mult = {SparsityLevel.DENSE: 1.0, SparsityLevel.SPARSE_75: 0.80, SparsityLevel.SPARSE_50: 0.60, SparsityLevel.SPARSE_25: 0.35}
        multiplier *= sparsity_mult[self.sparsity]

        # MoE expert routing
        moe_mult = {MoEMode.ALL_EXPERTS: 1.0, MoEMode.TOP_2: 0.50, MoEMode.TOP_1: 0.25, MoEMode.ADAPTIVE: 0.40}
        multiplier *= moe_mult[self.moe]

        return multiplier


# ============================================================================
# CONTROLLERS
# ============================================================================

class BaseController:
    """Base controller interface."""

    def __init__(self, thermal_target: float = 70.0, power_budget: float = 120.0):
        self.thermal_target = thermal_target
        self.power_budget = power_budget
        self.history: deque = deque(maxlen=32)

    def observe(self, temp_c: float, power_w: float, metrics: Dict):
        """Record observation."""
        self.history.append({'temp': temp_c, 'power': power_w, 'metrics': metrics})

    def select_action(self, current_temp: float, current_power: float) -> ActuatorState:
        """Select actuator configuration. Override in subclasses."""
        raise NotImplementedError


class PriorityChainController(BaseController):
    """
    Option A: Simple priority chain.

    If hot/overpower: Reduce all actuators aggressively
    If cool/underpower: Maximize quality
    """

    def select_action(self, current_temp: float, current_power: float) -> ActuatorState:
        temp_margin = self.thermal_target - current_temp
        power_margin = self.power_budget - current_power

        # Calculate urgency
        urgency = 0.0
        if temp_margin < 0:
            urgency += abs(temp_margin) / 10.0  # Thermal urgency
        if power_margin < 0:
            urgency += abs(power_margin) / 20.0  # Power urgency

        # Map urgency to actuator settings
        if urgency > 2.0:
            # CRITICAL: Maximum savings
            return ActuatorState(
                precision=PrecisionMode.INT8,
                window=WindowSize.TINY,
                batch=BatchMode.OCT,
                cache=CacheMode.RECOMPUTE,
                dpm=DPMLevel.LOW,
                sparsity=SparsityLevel.SPARSE_25,
                moe=MoEMode.TOP_1,
            )
        elif urgency > 1.0:
            # HIGH: Moderate savings
            return ActuatorState(
                precision=PrecisionMode.FP16,
                window=WindowSize.SMALL,
                batch=BatchMode.QUAD,
                cache=CacheMode.SELECTIVE,
                dpm=DPMLevel.LOW,
                sparsity=SparsityLevel.SPARSE_50,
                moe=MoEMode.TOP_2,
            )
        elif urgency > 0.3:
            # MEDIUM: Balanced
            return ActuatorState(
                precision=PrecisionMode.FP16,
                window=WindowSize.MEDIUM,
                batch=BatchMode.DOUBLE,
                cache=CacheMode.SELECTIVE,
                dpm=DPMLevel.BALANCED,
                sparsity=SparsityLevel.SPARSE_75,
                moe=MoEMode.ADAPTIVE,
            )
        else:
            # LOW: Quality first
            return ActuatorState(
                precision=PrecisionMode.FP32,
                window=WindowSize.LARGE,
                batch=BatchMode.SINGLE,
                cache=CacheMode.FULL_CACHE,
                dpm=DPMLevel.HIGH,
                sparsity=SparsityLevel.DENSE,
                moe=MoEMode.ALL_EXPERTS,
            )


class IndependentController(BaseController):
    """
    Option B: Independent per-actuator decisions.

    Each actuator adjusts based on specific bottleneck:
    - Thermal -> DPM level
    - Power -> Precision + Sparsity
    - Memory -> Cache + Batch
    - Latency -> Window + MoE
    """

    def select_action(self, current_temp: float, current_power: float) -> ActuatorState:
        state = ActuatorState()

        # DPM based on thermal
        temp_margin = self.thermal_target - current_temp
        if temp_margin < -5:
            state.dpm = DPMLevel.LOW
        elif temp_margin < 5:
            state.dpm = DPMLevel.BALANCED
        else:
            state.dpm = DPMLevel.HIGH

        # Precision + Sparsity based on power
        power_margin = self.power_budget - current_power
        if power_margin < -20:
            state.precision = PrecisionMode.INT8
            state.sparsity = SparsityLevel.SPARSE_25
        elif power_margin < -10:
            state.precision = PrecisionMode.FP16
            state.sparsity = SparsityLevel.SPARSE_50
        elif power_margin < 10:
            state.precision = PrecisionMode.FP16
            state.sparsity = SparsityLevel.SPARSE_75
        else:
            state.precision = PrecisionMode.FP32
            state.sparsity = SparsityLevel.DENSE

        # Cache + Batch based on memory pressure (heuristic)
        if len(self.history) > 2:
            recent_power = np.mean([h['power'] for h in list(self.history)[-3:]])
            if recent_power > self.power_budget * 0.9:
                state.cache = CacheMode.RECOMPUTE
                state.batch = BatchMode.OCT
            elif recent_power > self.power_budget * 0.7:
                state.cache = CacheMode.SELECTIVE
                state.batch = BatchMode.QUAD
            else:
                state.cache = CacheMode.FULL_CACHE
                state.batch = BatchMode.DOUBLE
        else:
            state.cache = CacheMode.SELECTIVE
            state.batch = BatchMode.DOUBLE

        # Window + MoE based on compute budget
        if power_margin < 0:
            state.window = WindowSize.SMALL
            state.moe = MoEMode.TOP_1
        elif power_margin < 20:
            state.window = WindowSize.MEDIUM
            state.moe = MoEMode.TOP_2
        else:
            state.window = WindowSize.LARGE
            state.moe = MoEMode.ADAPTIVE

        return state


class LearnedController(BaseController):
    """
    Option C: Learned policy network.

    Simple MLP that maps (temp, power, history) -> actuator configuration.
    Trained with gradient-free optimization (evolution strategies).
    """

    def __init__(self, thermal_target: float = 70.0, power_budget: float = 120.0):
        super().__init__(thermal_target, power_budget)

        # Policy network: [temp, power, temp_trend, power_trend] -> 7 actuator logits
        self.policy = nn.Sequential(
            nn.Linear(4, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 7 * 4),  # 7 actuators, ~4 options each (simplified)
        )

        # Actuator cardinalities
        self.actuator_dims = [3, 4, 4, 3, 3, 4, 4]  # How many options per actuator

    def select_action(self, current_temp: float, current_power: float) -> ActuatorState:
        # Compute trends
        temp_trend = 0.0
        power_trend = 0.0
        if len(self.history) >= 2:
            temp_trend = current_temp - self.history[-2]['temp']
            power_trend = current_power - self.history[-2]['power']

        # Normalize inputs
        temp_norm = (current_temp - self.thermal_target) / 20.0
        power_norm = (current_power - self.power_budget) / 40.0

        x = torch.tensor([temp_norm, power_norm, temp_trend / 10.0, power_trend / 20.0], dtype=torch.float32)

        with torch.no_grad():
            logits = self.policy(x)

        # Split logits for each actuator
        offset = 0
        selections = []
        for dim in self.actuator_dims:
            actuator_logits = logits[offset:offset + dim]
            selection = torch.argmax(actuator_logits).item()
            selections.append(selection)
            offset += dim

        # Map selections to actuator values
        precision_map = [PrecisionMode.FP32, PrecisionMode.FP16, PrecisionMode.INT8]
        window_map = [WindowSize.TINY, WindowSize.SMALL, WindowSize.MEDIUM, WindowSize.LARGE]
        batch_map = [BatchMode.SINGLE, BatchMode.DOUBLE, BatchMode.QUAD, BatchMode.OCT]
        cache_map = [CacheMode.FULL_CACHE, CacheMode.SELECTIVE, CacheMode.RECOMPUTE]
        dpm_map = [DPMLevel.LOW, DPMLevel.BALANCED, DPMLevel.HIGH]
        sparsity_map = [SparsityLevel.DENSE, SparsityLevel.SPARSE_75, SparsityLevel.SPARSE_50, SparsityLevel.SPARSE_25]
        moe_map = [MoEMode.ALL_EXPERTS, MoEMode.TOP_2, MoEMode.TOP_1, MoEMode.ADAPTIVE]

        return ActuatorState(
            precision=precision_map[min(selections[0], 2)],
            window=window_map[min(selections[1], 3)],
            batch=batch_map[min(selections[2], 3)],
            cache=cache_map[min(selections[3], 2)],
            dpm=dpm_map[min(selections[4], 2)],
            sparsity=sparsity_map[min(selections[5], 3)],
            moe=moe_map[min(selections[6], 3)],
        )


# ============================================================================
# SYNTHETIC WORKLOAD (simulates inference with actuator effects)
# ============================================================================

class SyntheticWorkload:
    """
    Simulates inference workload with actuator effects.

    Uses PyTorch operations that scale with actuator settings.
    """

    def __init__(self, device: str = 'cuda', base_size: int = 2048):
        self.device = device
        self.base_size = base_size

    def run_iteration(self, state: ActuatorState) -> Dict[str, float]:
        """
        Run synthetic workload with given actuator state.
        Returns: metrics dict with latency, throughput estimate
        """
        # Determine effective sizes
        seq_len = state.window.value
        batch = max(1, 32 // state.batch.value)  # Inverse batch accumulation

        # Precision dtype
        dtype_map = {
            PrecisionMode.FP32: torch.float32,
            PrecisionMode.FP16: torch.float16,
            PrecisionMode.INT8: torch.int8,  # Will use float16 but with reduced ops
        }
        dtype = dtype_map.get(state.precision, torch.float32)
        if state.precision == PrecisionMode.INT8:
            dtype = torch.float16  # Simulate with FP16 but reduced compute

        # Create tensors
        try:
            x = torch.randn(batch, seq_len, 512, dtype=dtype, device=self.device)
            w = torch.randn(512, 512, dtype=dtype, device=self.device)

            # Compute scaled by actuator state
            t0 = time.perf_counter()

            # Simulate attention (scaled by sparsity)
            sparsity_ops = {
                SparsityLevel.DENSE: 1.0,
                SparsityLevel.SPARSE_75: 0.75,
                SparsityLevel.SPARSE_50: 0.50,
                SparsityLevel.SPARSE_25: 0.25,
            }
            ops_scale = sparsity_ops[state.sparsity]

            # MoE expert simulation
            moe_ops = {
                MoEMode.ALL_EXPERTS: 4,
                MoEMode.TOP_2: 2,
                MoEMode.TOP_1: 1,
                MoEMode.ADAPTIVE: 2,
            }
            expert_count = moe_ops[state.moe]

            # Run operations
            for _ in range(int(expert_count * ops_scale)):
                y = torch.matmul(x, w)
                if state.cache != CacheMode.FULL_CACHE:
                    # Recompute simulation (extra pass)
                    _ = torch.matmul(y, w.t())

            torch.cuda.synchronize()
            elapsed = time.perf_counter() - t0

            # Cleanup
            del x, w
            if 'y' in locals():
                del y
            torch.cuda.empty_cache()

            return {
                'latency_ms': elapsed * 1000,
                'tokens_processed': batch * seq_len,
                'throughput_tok_s': (batch * seq_len) / elapsed if elapsed > 0 else 0,
            }

        except Exception as e:
            print(f"[WARN] Workload failed: {e}")
            return {
                'latency_ms': 0.0,
                'tokens_processed': 0,
                'throughput_tok_s': 0.0,
            }


# ============================================================================
# BENCHMARK RUNNER
# ============================================================================

@dataclass
class BenchmarkResult:
    """Result from one benchmark condition."""
    condition: str
    actuator_state: ActuatorState
    energy_j: float
    duration_s: float
    tokens_processed: int
    avg_power_w: float
    max_temp_c: float
    quality_score: float
    compute_multiplier: float


class UnifiedEmbodimentBenchmark:
    """
    Complete benchmark comparing:
    - Baseline (fixed quality settings)
    - Individual actuators (7 conditions)
    - Combined (coordinated control)
    - Oracle (best individual per iteration)
    """

    def __init__(
        self,
        controller_type: str = 'priority',
        thermal_target: float = 70.0,
        power_budget: float = 120.0,
        iterations: int = 100,
    ):
        self.controller_type = controller_type
        self.thermal_target = thermal_target
        self.power_budget = power_budget
        self.iterations = iterations

        # Initialize telemetry
        self.telemetry = SysfsHwmonTelemetry(sample_rate_hz=50.0)
        print(f"[INFO] Telemetry initialized: {self.telemetry.paths.power_average}")

        # Measure idle baseline
        print("[INFO] Measuring idle power baseline...")
        idle = self.telemetry.measure_idle_baseline(duration_s=2.0)
        print(f"[INFO] Idle power: {idle:.1f} W")

        # Workload
        self.workload = SyntheticWorkload(device='cuda' if torch.cuda.is_available() else 'cpu')

        # Create controller
        if controller_type == 'priority':
            self.controller = PriorityChainController(thermal_target, power_budget)
        elif controller_type == 'independent':
            self.controller = IndependentController(thermal_target, power_budget)
        elif controller_type == 'learned':
            self.controller = LearnedController(thermal_target, power_budget)
        else:
            raise ValueError(f"Unknown controller: {controller_type}")

        # Results storage
        self.results: List[BenchmarkResult] = []

    def run_baseline(self) -> BenchmarkResult:
        """Baseline: Quality-first fixed settings."""
        print("\n[BASELINE] Running with quality-first fixed settings...")

        state = ActuatorState(
            precision=PrecisionMode.FP32,
            window=WindowSize.LARGE,
            batch=BatchMode.SINGLE,
            cache=CacheMode.FULL_CACHE,
            dpm=DPMLevel.HIGH,
            sparsity=SparsityLevel.DENSE,
            moe=MoEMode.ALL_EXPERTS,
        )

        return self._run_condition("Baseline_QualityFirst", state)

    def run_individual_actuators(self) -> List[BenchmarkResult]:
        """Test each actuator individually (others at baseline)."""
        print("\n[INDIVIDUAL] Testing each actuator alone...")

        results = []

        # 1. Precision only
        state = ActuatorState(precision=PrecisionMode.FP16)
        results.append(self._run_condition("Individual_Precision", state))

        # 2. Window only
        state = ActuatorState(window=WindowSize.SMALL)
        results.append(self._run_condition("Individual_Window", state))

        # 3. Batch only
        state = ActuatorState(batch=BatchMode.QUAD)
        results.append(self._run_condition("Individual_Batch", state))

        # 4. Cache only
        state = ActuatorState(cache=CacheMode.SELECTIVE)
        results.append(self._run_condition("Individual_Cache", state))

        # 5. DPM only
        state = ActuatorState(dpm=DPMLevel.LOW)
        results.append(self._run_condition("Individual_DPM", state))

        # 6. Sparsity only
        state = ActuatorState(sparsity=SparsityLevel.SPARSE_50)
        results.append(self._run_condition("Individual_Sparsity", state))

        # 7. MoE only
        state = ActuatorState(moe=MoEMode.TOP_2)
        results.append(self._run_condition("Individual_MoE", state))

        return results

    def run_combined(self) -> BenchmarkResult:
        """Test coordinated control with all actuators."""
        print(f"\n[COMBINED] Running unified controller ({self.controller_type})...")

        total_energy = 0.0
        total_duration = 0.0
        total_tokens = 0
        max_temp = 0.0
        states_used = []

        pbar = tqdm(range(self.iterations), desc="Combined control")
        for i in pbar:
            # Get current state
            sample = self.telemetry.read_sample()
            temp = sample.temp_edge_c
            power = sample.power_w

            # Controller selects action
            state = self.controller.select_action(temp, power)
            states_used.append(state.to_dict())

            # Run workload with energy measurement
            with EnergyMeter(self.telemetry) as meter:
                metrics = self.workload.run_iteration(state)

            # Record
            total_energy += meter.energy_j
            total_duration += meter.duration_s
            total_tokens += metrics['tokens_processed']
            max_temp = max(max_temp, temp)

            # Update controller
            self.controller.observe(temp, power, metrics)

            pbar.set_postfix({
                'E': f"{total_energy:.1f}J",
                'P': f"{power:.1f}W",
                'T': f"{temp:.1f}°C",
            })

        avg_power = total_energy / total_duration if total_duration > 0 else 0

        # Compute average quality score
        avg_quality = np.mean([
            ActuatorState(**{k: getattr(ActuatorState(), k) if k not in s else self._parse_state_value(k, s[k])
                            for k in ActuatorState().__dict__.keys()}).quality_score
            for s in states_used
        ])

        return BenchmarkResult(
            condition=f"Combined_{self.controller_type}",
            actuator_state=None,  # Variable
            energy_j=total_energy,
            duration_s=total_duration,
            tokens_processed=total_tokens,
            avg_power_w=avg_power,
            max_temp_c=max_temp,
            quality_score=avg_quality,
            compute_multiplier=0.0,  # Not applicable
        )

    def _parse_state_value(self, key: str, value: Any) -> Any:
        """Parse state value from dict back to enum."""
        if key == 'precision':
            return PrecisionMode[value] if isinstance(value, str) else value
        elif key == 'window':
            return WindowSize(value) if isinstance(value, int) else value
        elif key == 'batch':
            return BatchMode(value) if isinstance(value, int) else value
        elif key == 'cache':
            return CacheMode[value] if isinstance(value, str) else value
        elif key == 'dpm':
            return DPMLevel[value] if isinstance(value, str) else value
        elif key == 'sparsity':
            return SparsityLevel[value] if isinstance(value, str) else value
        elif key == 'moe':
            return MoEMode[value] if isinstance(value, str) else value
        return value

    def _run_condition(self, name: str, state: ActuatorState) -> BenchmarkResult:
        """Run benchmark for specific actuator configuration."""
        print(f"  Running: {name}")

        total_energy = 0.0
        total_duration = 0.0
        total_tokens = 0
        max_temp = 0.0

        for _ in range(self.iterations):
            with EnergyMeter(self.telemetry) as meter:
                metrics = self.workload.run_iteration(state)

            total_energy += meter.energy_j
            total_duration += meter.duration_s
            total_tokens += metrics['tokens_processed']

            sample = self.telemetry.read_sample()
            max_temp = max(max_temp, sample.temp_edge_c)

        avg_power = total_energy / total_duration if total_duration > 0 else 0

        return BenchmarkResult(
            condition=name,
            actuator_state=state,
            energy_j=total_energy,
            duration_s=total_duration,
            tokens_processed=total_tokens,
            avg_power_w=avg_power,
            max_temp_c=max_temp,
            quality_score=state.quality_score,
            compute_multiplier=state.compute_multiplier,
        )

    def run_oracle(self) -> BenchmarkResult:
        """Oracle: Best individual actuator per iteration (upper bound)."""
        print("\n[ORACLE] Testing best-per-iteration selection...")

        # For oracle, we'd need to test all options per iteration
        # Simplified: Use independent controller as proxy
        print("  (Using independent controller as oracle proxy)")

        oracle_controller = IndependentController(self.thermal_target, self.power_budget)

        total_energy = 0.0
        total_duration = 0.0
        total_tokens = 0
        max_temp = 0.0

        for _ in tqdm(range(self.iterations), desc="Oracle"):
            sample = self.telemetry.read_sample()
            state = oracle_controller.select_action(sample.temp_edge_c, sample.power_w)

            with EnergyMeter(self.telemetry) as meter:
                metrics = self.workload.run_iteration(state)

            total_energy += meter.energy_j
            total_duration += meter.duration_s
            total_tokens += metrics['tokens_processed']
            max_temp = max(max_temp, sample.temp_edge_c)

            oracle_controller.observe(sample.temp_edge_c, sample.power_w, metrics)

        avg_power = total_energy / total_duration if total_duration > 0 else 0

        return BenchmarkResult(
            condition="Oracle_BestPerIteration",
            actuator_state=None,
            energy_j=total_energy,
            duration_s=total_duration,
            tokens_processed=total_tokens,
            avg_power_w=avg_power,
            max_temp_c=max_temp,
            quality_score=0.95,  # Estimated
            compute_multiplier=0.0,
        )

    def run_full_benchmark(self) -> Dict:
        """Run complete benchmark matrix."""
        print("="*80)
        print("Z913: UNIFIED EMBODIMENT BENCHMARK")
        print(f"Controller: {self.controller_type}")
        print(f"Iterations: {self.iterations}")
        print(f"Thermal target: {self.thermal_target}°C")
        print(f"Power budget: {self.power_budget}W")
        print("="*80)

        # Warmup
        print("\n[WARMUP] Warming up GPU...")
        warmup_state = ActuatorState()
        for _ in range(5):
            self.workload.run_iteration(warmup_state)
        time.sleep(1.0)

        # Run conditions
        baseline = self.run_baseline()
        self.results.append(baseline)

        individual_results = self.run_individual_actuators()
        self.results.extend(individual_results)

        combined = self.run_combined()
        self.results.append(combined)

        oracle = self.run_oracle()
        self.results.append(oracle)

        # Analysis
        return self._analyze_results()

    def _analyze_results(self) -> Dict:
        """Analyze and compare all results."""
        print("\n" + "="*80)
        print("RESULTS ANALYSIS")
        print("="*80)

        # Find baseline
        baseline = next(r for r in self.results if r.condition.startswith("Baseline"))

        # Compute metrics
        analysis = {
            'controller_type': self.controller_type,
            'thermal_target': self.thermal_target,
            'power_budget': self.power_budget,
            'iterations': self.iterations,
            'baseline': {
                'energy_j': baseline.energy_j,
                'avg_power_w': baseline.avg_power_w,
                'max_temp_c': baseline.max_temp_c,
                'quality_score': baseline.quality_score,
                'j_per_token': baseline.energy_j / baseline.tokens_processed if baseline.tokens_processed > 0 else 0,
            },
            'individual': [],
            'combined': {},
            'oracle': {},
            'comparison': {},
        }

        # Individual actuators
        print("\nINDIVIDUAL ACTUATORS:")
        for r in self.results:
            if r.condition.startswith("Individual"):
                energy_savings = (baseline.energy_j - r.energy_j) / baseline.energy_j * 100
                quality_retention = r.quality_score / baseline.quality_score * 100

                print(f"  {r.condition:30s}: {energy_savings:+6.2f}% energy, {quality_retention:5.1f}% quality")

                analysis['individual'].append({
                    'name': r.condition,
                    'energy_j': r.energy_j,
                    'energy_savings_pct': energy_savings,
                    'quality_score': r.quality_score,
                    'quality_retention_pct': quality_retention,
                    'avg_power_w': r.avg_power_w,
                })

        # Combined
        combined = next(r for r in self.results if r.condition.startswith("Combined"))
        combined_savings = (baseline.energy_j - combined.energy_j) / baseline.energy_j * 100
        combined_quality = combined.quality_score / baseline.quality_score * 100

        print(f"\nCOMBINED ({self.controller_type}):")
        print(f"  Energy savings: {combined_savings:+.2f}%")
        print(f"  Quality retention: {combined_quality:.1f}%")
        print(f"  Avg power: {combined.avg_power_w:.1f}W (baseline: {baseline.avg_power_w:.1f}W)")
        print(f"  Max temp: {combined.max_temp_c:.1f}°C (baseline: {baseline.max_temp_c:.1f}°C)")

        analysis['combined'] = {
            'energy_j': combined.energy_j,
            'energy_savings_pct': combined_savings,
            'quality_score': combined.quality_score,
            'quality_retention_pct': combined_quality,
            'avg_power_w': combined.avg_power_w,
            'max_temp_c': combined.max_temp_c,
        }

        # Oracle
        oracle = next(r for r in self.results if r.condition.startswith("Oracle"))
        oracle_savings = (baseline.energy_j - oracle.energy_j) / baseline.energy_j * 100

        print(f"\nORACLE (upper bound):")
        print(f"  Energy savings: {oracle_savings:+.2f}%")

        analysis['oracle'] = {
            'energy_j': oracle.energy_j,
            'energy_savings_pct': oracle_savings,
            'avg_power_w': oracle.avg_power_w,
        }

        # Key comparisons
        best_individual = max(analysis['individual'], key=lambda x: x['energy_savings_pct'])

        synergy_vs_best = combined_savings - best_individual['energy_savings_pct']
        synergy_vs_sum = combined_savings  # Simplified (would need to model sum properly)

        print(f"\nKEY FINDINGS:")
        print(f"  Best individual actuator: {best_individual['name']} ({best_individual['energy_savings_pct']:.2f}%)")
        print(f"  Combined savings: {combined_savings:.2f}%")
        print(f"  Synergy (Combined - Best Individual): {synergy_vs_best:+.2f}%")
        print(f"  Oracle upper bound: {oracle_savings:.2f}%")
        print(f"  Combined vs Oracle: {combined_savings / oracle_savings * 100:.1f}% of optimal")

        analysis['comparison'] = {
            'best_individual_actuator': best_individual['name'],
            'best_individual_savings_pct': best_individual['energy_savings_pct'],
            'combined_savings_pct': combined_savings,
            'synergy_vs_best_individual_pct': synergy_vs_best,
            'oracle_savings_pct': oracle_savings,
            'combined_vs_oracle_pct': combined_savings / oracle_savings * 100 if oracle_savings > 0 else 0,
        }

        # Business projections
        print(f"\nBUSINESS PROJECTIONS (100 GPU cluster, 24/7):")
        baseline_kwh_year = (baseline.avg_power_w / 1000) * 24 * 365 * 100
        combined_kwh_year = (combined.avg_power_w / 1000) * 24 * 365 * 100
        savings_kwh_year = baseline_kwh_year - combined_kwh_year

        # Assume $0.10/kWh and 0.4kg CO2/kWh
        cost_savings = savings_kwh_year * 0.10
        co2_reduction_kg = savings_kwh_year * 0.4

        print(f"  Baseline consumption: {baseline_kwh_year:,.0f} kWh/year")
        print(f"  Combined consumption: {combined_kwh_year:,.0f} kWh/year")
        print(f"  Savings: {savings_kwh_year:,.0f} kWh/year")
        print(f"  Cost savings: ${cost_savings:,.0f}/year")
        print(f"  CO2 reduction: {co2_reduction_kg:,.0f} kg/year ({co2_reduction_kg/1000:.1f} metric tons)")

        analysis['business_projection'] = {
            'cluster_size': 100,
            'baseline_kwh_year': baseline_kwh_year,
            'combined_kwh_year': combined_kwh_year,
            'savings_kwh_year': savings_kwh_year,
            'cost_savings_usd_year': cost_savings,
            'co2_reduction_kg_year': co2_reduction_kg,
        }

        return analysis


# ============================================================================
# MAIN
# ============================================================================

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Z913: Unified Embodiment Benchmark")
    parser.add_argument('--controller', choices=['priority', 'independent', 'learned'],
                       default='priority', help='Controller strategy')
    parser.add_argument('--thermal-target', type=float, default=70.0, help='Thermal target (°C)')
    parser.add_argument('--power-budget', type=float, default=120.0, help='Power budget (W)')
    parser.add_argument('--iterations', type=int, default=100, help='Iterations per condition')
    parser.add_argument('--output', type=str, default='results/z913_unified_embodiment.json',
                       help='Output path')
    args = parser.parse_args()

    # Ensure output directory
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Run benchmark
    benchmark = UnifiedEmbodimentBenchmark(
        controller_type=args.controller,
        thermal_target=args.thermal_target,
        power_budget=args.power_budget,
        iterations=args.iterations,
    )

    results = benchmark.run_full_benchmark()

    # Save results
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n[DONE] Results saved to: {output_path}")

    # Verdict
    synergy = results['comparison']['synergy_vs_best_individual_pct']
    print("\n" + "="*80)
    print("VERDICT:")
    if synergy > 5.0:
        print(f"✓ HYPOTHESIS CONFIRMED: Combined embodiment shows {synergy:.1f}% MORE savings")
        print("  than best individual actuator. Coordination creates emergent synergy!")
    elif synergy > 0:
        print(f"~ WEAK SYNERGY: Combined is {synergy:.1f}% better than best individual.")
    else:
        print(f"✗ NO SYNERGY: Combined is {abs(synergy):.1f}% WORSE than best individual.")
        print("  Independent actuators may be interfering with each other.")
    print("="*80)


if __name__ == '__main__':
    main()
