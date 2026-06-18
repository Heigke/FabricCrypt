#!/usr/bin/env python3
"""
Z602: Complete Embodied Closed-Loop Validation

The FULL embodiment:
1. SENSE: Deep telemetry (gpu_metrics, temps, power, clocks, activity)
2. MODEL: Attention-based world model learns hardware dynamics
3. ACT: Hardware actuation (perf level) + Compute modulation (batch, precision, layers)
4. LOOP: Actions affect hardware, hardware affects model, model adjusts actions

Hypothesis: Embodied self-regulation beats reactive baseline.
"""

import os
import sys
import time
import json
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Optional
from collections import deque

# AMD GPU setup
def detect_gpu_vendor() -> str:
    for card in sorted(Path("/sys/class/drm").glob("card[0-9]*")):
        vendor_file = card / "device/vendor"
        if vendor_file.exists():
            try:
                if vendor_file.read_text().strip() == "0x1002":
                    return "amd"
            except:
                pass
    return "cpu"

GPU_VENDOR = detect_gpu_vendor()
if GPU_VENDOR == "amd":
    os.environ["HSA_OVERRIDE_GFX_VERSION"] = "11.0.0"

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm

from src.deep_embodiment import (
    create_deep_telemetry,
    create_actuator,
    HardwareWorldModel,
    HardwareState,
    ComputeAction,
    WorldModelTrainer,
    PerfLevel,
)


@dataclass
class EmbodiedConfig:
    """Configuration for embodied compute."""
    batch_size: int = 32
    tensor_size: int = 1024
    exit_layer: int = 12
    perf_level: int = 1  # 0=LOW, 1=BALANCED, 2=HIGH
    throttle_ms: float = 0.0


@dataclass
class LoopMetrics:
    """Metrics from one loop iteration."""
    temp_c: float
    power_w: float
    throughput: float
    config: EmbodiedConfig


class EmbodiedLoop:
    """
    Complete embodied self-regulation loop.

    The loop:
    1. Sense hardware state via deep telemetry
    2. Feed state history to attention-based world model
    3. Predict outcomes for different actions
    4. Choose action that maximizes throughput within thermal/power budget
    5. Execute action (hardware actuation + compute modulation)
    6. Observe outcome, update world model
    """

    def __init__(
        self,
        telemetry,
        actuator,
        world_model: HardwareWorldModel,
        thermal_target_c: float = 65.0,
        power_budget_w: float = 120.0,
    ):
        self.telemetry = telemetry
        self.actuator = actuator
        self.world_model = world_model
        self.thermal_target = thermal_target_c
        self.power_budget = power_budget_w

        self.device = next(world_model.parameters()).device

        # Current config
        self.config = EmbodiedConfig()

        # Metrics history
        self.metrics_history: List[LoopMetrics] = []

        # State history for training (HardwareState objects)
        self.state_history: deque = deque(maxlen=32)

        # Training buffer for online learning
        self.trainer = WorldModelTrainer(world_model, lr=1e-4)

    def sense(self) -> HardwareState:
        """Read deep telemetry and convert to HardwareState."""
        sample = self.telemetry.read_sample()

        state = HardwareState(
            timestamp=time.time(),
            temp_edge_c=sample.temp_edge_c,
            power_w=sample.power_average_w,
            gpu_busy_pct=sample.gpu_busy_pct,
            sclk_mhz=sample.sclk_current_mhz,
            latency_ms=0.0,
        )

        # Feed to world model
        self.world_model.observe(state)

        # Also keep state history for training
        self.state_history.append(state)

        return state

    def predict_outcome(self, config: EmbodiedConfig) -> Dict:
        """Use world model to predict outcome of config."""
        action = ComputeAction(
            exit_layer=config.exit_layer,
            perf_level=config.perf_level,
        )

        try:
            predicted_state = self.world_model.predict_next_state(action)
            return {
                'temp_c': predicted_state.temp_edge_c,
                'power_w': predicted_state.power_w,
                'valid': True,
            }
        except Exception as e:
            # Fallback heuristic
            base_power = 30 + config.batch_size * 0.5 * (config.perf_level + 1)
            return {
                'temp_c': 50 + config.perf_level * 10,
                'power_w': base_power,
                'valid': False,
            }

    def choose_action(self, current_state: HardwareState) -> EmbodiedConfig:
        """Choose best config based on world model predictions."""
        # Generate candidate configs
        candidates = [
            # Minimal (energy saving)
            EmbodiedConfig(batch_size=8, tensor_size=512, exit_layer=6, perf_level=0),
            # Low
            EmbodiedConfig(batch_size=16, tensor_size=768, exit_layer=9, perf_level=0),
            # Medium-Low
            EmbodiedConfig(batch_size=24, tensor_size=896, exit_layer=9, perf_level=1),
            # Medium
            EmbodiedConfig(batch_size=32, tensor_size=1024, exit_layer=12, perf_level=1),
            # Medium-High
            EmbodiedConfig(batch_size=48, tensor_size=1280, exit_layer=12, perf_level=1),
            # High
            EmbodiedConfig(batch_size=64, tensor_size=1536, exit_layer=12, perf_level=2),
        ]

        best_config = candidates[2]  # Default to medium-low
        best_score = float('-inf')

        for config in candidates:
            outcome = self.predict_outcome(config)

            # Score: throughput within constraints
            throughput = config.batch_size * (config.exit_layer / 12.0)

            # Soft penalties for exceeding budgets
            temp_margin = self.thermal_target - outcome['temp_c']
            power_margin = self.power_budget - outcome['power_w']

            if temp_margin < 0:
                throughput *= (1.0 + temp_margin * 0.1)  # Reduce score if over thermal
            if power_margin < 0:
                throughput *= (1.0 + power_margin * 0.05)  # Reduce score if over power

            # Bonus for staying well within limits
            if temp_margin > 10:
                throughput *= 1.1
            if power_margin > 30:
                throughput *= 1.05

            if throughput > best_score:
                best_score = throughput
                best_config = config

        return best_config

    def execute(self, config: EmbodiedConfig) -> float:
        """Execute compute with given config, return throughput."""
        # Apply hardware actuation
        perf_map = {0: PerfLevel.LOW, 1: PerfLevel.BALANCED, 2: PerfLevel.HIGH}
        self.actuator.set_perf_level(perf_map[config.perf_level])

        # Apply compute throttle if specified
        if config.throttle_ms > 0:
            time.sleep(config.throttle_ms / 1000.0)

        # Execute compute
        a = torch.randn(config.batch_size, config.tensor_size, device=self.device)
        b = torch.randn(config.tensor_size, config.tensor_size, device=self.device)

        t0 = time.perf_counter()
        for _ in range(config.exit_layer):
            c = torch.matmul(a, b)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0

        throughput = config.batch_size * config.exit_layer / elapsed
        return throughput

    def step(self) -> LoopMetrics:
        """One complete embodied loop iteration."""
        # 1. Sense
        state = self.sense()

        # 2. Choose action
        self.config = self.choose_action(state)

        # 3. Execute
        throughput = self.execute(self.config)

        # 4. Observe outcome
        outcome_state = self.sense()

        # 5. Learn (add to training buffer)
        if len(self.state_history) >= 10:
            history = list(self.state_history)[-10:]
            action = ComputeAction(
                exit_layer=self.config.exit_layer,
                perf_level=self.config.perf_level,
            )
            self.trainer.add_experience(history, action, outcome_state)

        # Record metrics
        metrics = LoopMetrics(
            temp_c=outcome_state.temp_edge_c,
            power_w=outcome_state.power_w,
            throughput=throughput,
            config=self.config,
        )
        self.metrics_history.append(metrics)

        return metrics

    def train_online(self, batch_size: int = 16):
        """Online learning step."""
        if len(self.trainer.buffer) >= batch_size:
            self.trainer.train_step(batch_size)


class BaselineController:
    """Simple reactive baseline for comparison."""

    def __init__(self, telemetry, actuator, thermal_threshold: float = 60.0):
        self.telemetry = telemetry
        self.actuator = actuator
        self.thermal_threshold = thermal_threshold
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.metrics_history: List[LoopMetrics] = []

    def step(self) -> LoopMetrics:
        """One baseline control step."""
        sample = self.telemetry.read_sample()
        temp = sample.temp_edge_c

        # Simple reactive: reduce performance if hot
        if temp > self.thermal_threshold + 10:
            config = EmbodiedConfig(batch_size=16, tensor_size=768, exit_layer=6, perf_level=0)
            self.actuator.set_perf_level(PerfLevel.LOW)
        elif temp > self.thermal_threshold:
            config = EmbodiedConfig(batch_size=32, tensor_size=1024, exit_layer=9, perf_level=1)
            self.actuator.set_perf_level(PerfLevel.BALANCED)
        else:
            config = EmbodiedConfig(batch_size=48, tensor_size=1280, exit_layer=12, perf_level=2)
            self.actuator.set_perf_level(PerfLevel.HIGH)

        # Execute
        a = torch.randn(config.batch_size, config.tensor_size, device=self.device)
        b = torch.randn(config.tensor_size, config.tensor_size, device=self.device)

        t0 = time.perf_counter()
        for _ in range(config.exit_layer):
            c = torch.matmul(a, b)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0

        throughput = config.batch_size * config.exit_layer / elapsed

        # Re-read after compute
        sample = self.telemetry.read_sample()

        metrics = LoopMetrics(
            temp_c=sample.temp_edge_c,
            power_w=sample.power_average_w,
            throughput=throughput,
            config=config,
        )
        self.metrics_history.append(metrics)

        return metrics


def run_experiment(n_warmup: int = 10, n_steps: int = 30):
    """Run complete embodied vs baseline comparison."""
    print("=" * 70)
    print("Z602: EMBODIED CLOSED-LOOP VALIDATION")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")

    # Initialize components
    print("\n--- Initializing ---")
    telemetry = create_deep_telemetry()
    actuator = create_actuator()

    world_model = HardwareWorldModel(
        state_dim=10,
        action_dim=3,
        hidden_dim=64,
        n_heads=4,
        n_layers=2,
    ).to(device)

    print(f"  World model: {sum(p.numel() for p in world_model.parameters()):,} params")

    # Pre-train world model with synthetic causality data
    print("\n--- Pre-training World Model ---")
    trainer = WorldModelTrainer(world_model)

    for _ in range(50):
        # LOW perf -> stays cool
        history_cool = [HardwareState(temp_edge_c=45+j*0.3, power_w=40, sclk_mhz=600) for j in range(10)]
        trainer.add_experience(history_cool, ComputeAction(exit_layer=6, perf_level=0),
                              HardwareState(temp_edge_c=44, power_w=35, sclk_mhz=600))

        # HIGH perf -> heats up
        history_hot = [HardwareState(temp_edge_c=60+j*0.3, power_w=80, sclk_mhz=1200) for j in range(10)]
        trainer.add_experience(history_hot, ComputeAction(exit_layer=12, perf_level=2),
                              HardwareState(temp_edge_c=70, power_w=110, sclk_mhz=1500))

        # From hot, LOW -> cools
        history_cooling = [HardwareState(temp_edge_c=70-j*0.3, power_w=90, sclk_mhz=1000) for j in range(10)]
        trainer.add_experience(history_cooling, ComputeAction(exit_layer=6, perf_level=0),
                              HardwareState(temp_edge_c=58, power_w=45, sclk_mhz=700))

    for epoch in range(100):
        loss = trainer.train_step(batch_size=32)
        if epoch % 25 == 0:
            print(f"  Epoch {epoch}: loss={loss['loss']:.6f}")

    # Warmup
    print("\n--- Warmup ---")
    for i in range(n_warmup):
        _ = torch.randn(64, 1024, device=device) @ torch.randn(1024, 1024, device=device)
        torch.cuda.synchronize()
    time.sleep(2)

    # Run baseline
    print("\n--- Running Baseline Controller ---")
    actuator.set_perf_level(PerfLevel.BALANCED)
    baseline = BaselineController(telemetry, actuator, thermal_threshold=60.0)

    telemetry.start_continuous_sampling()
    for i in tqdm(range(n_steps), desc="  Baseline"):
        baseline.step()
        time.sleep(0.1)
    telemetry.stop_continuous_sampling()

    # Cool down
    print("  Cooling down...")
    actuator.set_perf_level(PerfLevel.LOW)
    time.sleep(3)

    # Run embodied
    print("\n--- Running Embodied Controller ---")
    actuator.set_perf_level(PerfLevel.BALANCED)
    embodied = EmbodiedLoop(
        telemetry, actuator, world_model,
        thermal_target_c=60.0,
        power_budget_w=100.0,
    )

    telemetry.start_continuous_sampling()
    for i in tqdm(range(n_steps), desc="  Embodied"):
        embodied.step()
        if i % 5 == 0:
            embodied.train_online(batch_size=8)
        time.sleep(0.1)
    telemetry.stop_continuous_sampling()

    # Reset
    actuator.set_perf_level(PerfLevel.BALANCED)

    # Analyze results
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)

    baseline_temps = [m.temp_c for m in baseline.metrics_history]
    baseline_powers = [m.power_w for m in baseline.metrics_history]
    baseline_throughputs = [m.throughput for m in baseline.metrics_history]

    embodied_temps = [m.temp_c for m in embodied.metrics_history]
    embodied_powers = [m.power_w for m in embodied.metrics_history]
    embodied_throughputs = [m.throughput for m in embodied.metrics_history]

    print(f"\nBaseline Controller:")
    print(f"  Avg Temperature: {np.mean(baseline_temps):.1f}°C (max: {np.max(baseline_temps):.1f}°C)")
    print(f"  Avg Power:       {np.mean(baseline_powers):.1f}W")
    print(f"  Avg Throughput:  {np.mean(baseline_throughputs):.1f} ops/s")

    print(f"\nEmbodied Controller:")
    print(f"  Avg Temperature: {np.mean(embodied_temps):.1f}°C (max: {np.max(embodied_temps):.1f}°C)")
    print(f"  Avg Power:       {np.mean(embodied_powers):.1f}W")
    print(f"  Avg Throughput:  {np.mean(embodied_throughputs):.1f} ops/s")

    # Compute improvements
    temp_improvement = np.mean(baseline_temps) - np.mean(embodied_temps)
    power_improvement = np.mean(baseline_powers) - np.mean(embodied_powers)
    throughput_ratio = np.mean(embodied_throughputs) / max(np.mean(baseline_throughputs), 1)

    # Efficiency: throughput per watt
    baseline_efficiency = np.mean(baseline_throughputs) / max(np.mean(baseline_powers), 1)
    embodied_efficiency = np.mean(embodied_throughputs) / max(np.mean(embodied_powers), 1)
    efficiency_improvement = (embodied_efficiency / baseline_efficiency - 1) * 100

    print(f"\n--- Improvements ---")
    print(f"  Temperature: {temp_improvement:+.1f}°C {'(cooler)' if temp_improvement > 0 else '(hotter)'}")
    print(f"  Power:       {power_improvement:+.1f}W {'(less)' if power_improvement > 0 else '(more)'}")
    print(f"  Throughput:  {(throughput_ratio-1)*100:+.1f}%")
    print(f"  Efficiency:  {efficiency_improvement:+.1f}% (throughput/watt)")

    # Verdict
    print("\n--- Verdict ---")
    success = (temp_improvement > 0 or efficiency_improvement > 0) and temp_improvement > -5
    if success:
        print("  ✓ EMBODIED CONTROL OUTPERFORMS BASELINE")
    else:
        print("  ✗ Embodied control did not outperform baseline")

    # Save results (convert numpy values to native Python types)
    results = {
        'baseline': {
            'avg_temp': float(np.mean(baseline_temps)),
            'max_temp': float(np.max(baseline_temps)),
            'avg_power': float(np.mean(baseline_powers)),
            'avg_throughput': float(np.mean(baseline_throughputs)),
            'efficiency': float(baseline_efficiency),
        },
        'embodied': {
            'avg_temp': float(np.mean(embodied_temps)),
            'max_temp': float(np.max(embodied_temps)),
            'avg_power': float(np.mean(embodied_powers)),
            'avg_throughput': float(np.mean(embodied_throughputs)),
            'efficiency': float(embodied_efficiency),
        },
        'improvements': {
            'temp_c': float(temp_improvement),
            'power_w': float(power_improvement),
            'throughput_pct': float((throughput_ratio - 1) * 100),
            'efficiency_pct': float(efficiency_improvement),
        },
        'success': bool(success),
    }

    output_path = Path("results/z602_embodied_loop.json")
    output_path.parent.mkdir(exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {output_path}")

    return results


if __name__ == "__main__":
    run_experiment(n_warmup=10, n_steps=25)
