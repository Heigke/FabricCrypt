#!/usr/bin/env python3
"""
Z72: Embodied Bandit - Proper Hardware Control Experiment
=========================================================

This experiment implements the correct architecture for embodied AI:
- Separate language channel from control channel
- Window-based control (not per-token)
- Contextual bandit controller (fast convergence)
- Proper baselines (fixed cap, PID, GreenLLM, throttLL'eM)
- Real hardware actuation (power cap via NVML/sysfs)
- J/token efficiency metric

Key Design Principles (from critique):
1. Never inject body state into text token stream
2. Act on windows (time/token chunks), not per-token
3. Rate limit actuation (max 1 change per 200-500ms)
4. Semantics-safe: only change hardware operating point

Author: FEEL Research Team
Date: 2026-01-19
"""

import os
import sys
import time
import json
import logging
import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
import numpy as np

# Add project root to path
SCRIPT_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn as nn
import torch.optim as optim

# Import body daemon components
from src.body_daemon.telemetry.unified import UnifiedBodyTelemetry, TelemetrySnapshot, BodyState
from src.body_daemon.actuators.unified import UnifiedActuator, ActuatorAction
from src.body_daemon.controller.bandit import ContextualBanditController, ControllerConfig
from src.body_daemon.controller.baselines import (
    FixedCapController, PIDController, GreenLLMController, ThrottLLMController,
    FixedCapConfig, PIDConfig, GreenLLMConfig, ThrottLLMConfig
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


# ============================================================================
# Simple Language Model (for generating real workload)
# ============================================================================

class SimpleTransformer(nn.Module):
    """
    Simple transformer for generating realistic inference workload.

    This is NOT the embodied model - it's just a workload generator.
    The embodied aspect is the separate controller that adjusts hardware.
    """

    def __init__(self, vocab_size: int = 32000, d_model: int = 256, n_heads: int = 4,
                 n_layers: int = 4, max_seq_len: int = 512):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_seq_len = max_seq_len

        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Embedding(max_seq_len, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=0.1,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.output = nn.Linear(d_model, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, seq_len = x.shape
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(b, -1)

        h = self.embedding(x) + self.pos_embedding(positions)
        h = self.transformer(h)
        logits = self.output(h)

        return logits

    def generate(self, prompt: torch.Tensor, max_new_tokens: int = 100) -> Tuple[torch.Tensor, List[float]]:
        """Generate tokens and return entropies."""
        self.eval()
        tokens = prompt.clone()
        entropies = []

        with torch.no_grad():
            for _ in range(max_new_tokens):
                logits = self.forward(tokens[:, -self.max_seq_len:])
                next_logits = logits[:, -1, :]

                # Compute entropy (average across batch)
                probs = torch.softmax(next_logits, dim=-1)
                entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=-1).mean().item()
                entropies.append(entropy)

                # Sample next token
                next_token = torch.multinomial(probs, 1)
                tokens = torch.cat([tokens, next_token], dim=1)

        return tokens, entropies


# ============================================================================
# Experiment Configuration
# ============================================================================

@dataclass
class ExperimentConfig:
    """Configuration for the embodied bandit experiment."""
    # Device
    device: str = "cuda"

    # Model (workload generator)
    vocab_size: int = 32000
    d_model: int = 256
    n_heads: int = 4
    n_layers: int = 4
    batch_size: int = 8
    seq_len: int = 256

    # Control
    control_interval_sec: float = 0.25  # 250ms control windows
    total_duration_sec: float = 60.0    # Total experiment duration
    warmup_sec: float = 5.0             # Warmup before measurements

    # Controllers to compare
    controllers: List[str] = None

    # Output
    output_dir: str = "results/z72_embodied_bandit"
    save_history: bool = True

    def __post_init__(self):
        if self.controllers is None:
            self.controllers = ['bandit', 'fixed_eco', 'fixed_med', 'fixed_perf', 'pid', 'greenllm', 'throttllm']


# ============================================================================
# Metrics Collection
# ============================================================================

@dataclass
class WindowMetrics:
    """Metrics for a single control window."""
    timestamp: float
    window_idx: int

    # Energy
    energy_joules: float
    power_watts_avg: float

    # Performance
    tokens_generated: int
    throughput_tps: float
    latency_ms_avg: float

    # Efficiency
    j_per_token: float
    tokens_per_watt: float

    # Temperature
    temp_c_avg: float
    temp_c_max: float

    # Control
    action: int
    action_name: str

    # Quality (entropy as proxy)
    entropy_avg: float


@dataclass
class ExperimentResults:
    """Results from a complete experiment run."""
    controller_name: str
    total_duration_sec: float

    # Aggregate metrics
    total_energy_joules: float
    total_tokens: int
    avg_throughput_tps: float
    avg_j_per_token: float
    avg_power_watts: float
    avg_temp_c: float
    max_temp_c: float

    # SLO performance
    latency_slo_violations: int
    thermal_slo_violations: int

    # Action distribution
    action_distribution: Dict[str, float]

    # Window history
    windows: List[Dict]


# ============================================================================
# Experiment Runner
# ============================================================================

class EmbodiedBanditExperiment:
    """
    Main experiment class that runs comparative evaluation of controllers.

    Architecture:
    - Model generates tokens (workload)
    - Controller observes hardware state + inference metrics
    - Controller selects power level
    - Actuator applies power cap
    - Metrics collected per window
    """

    ACTION_NAMES = ['ECO', 'LOW', 'MED', 'HIGH', 'PERF']

    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.device = torch.device(config.device if torch.cuda.is_available() else "cpu")

        # Initialize components
        logger.info(f"Initializing experiment on {self.device}")

        # Telemetry (body sensing)
        self.telemetry = UnifiedBodyTelemetry(device_id=0)
        logger.info(f"Telemetry: {self.telemetry.get_vendor()} - {self.telemetry.get_device_info().get('device_name', 'unknown')}")

        # Actuator (body control)
        self.actuator = UnifiedActuator(
            device_id=0,
            rate_limit_sec=config.control_interval_sec * 0.9,  # Slightly less than control interval
        )
        logger.info(f"Actuator: {self.actuator.get_vendor()}, power range: {self.actuator.get_power_range()}")

        # Model (workload generator)
        self.model = SimpleTransformer(
            vocab_size=config.vocab_size,
            d_model=config.d_model,
            n_heads=config.n_heads,
            n_layers=config.n_layers,
            max_seq_len=config.seq_len,
        ).to(self.device)

        n_params = sum(p.numel() for p in self.model.parameters())
        logger.info(f"Model: {n_params:,} parameters")

        # Controllers
        self.controllers = self._create_controllers()

    def _create_controllers(self) -> Dict[str, Any]:
        """Create all controllers for comparison."""
        controllers = {}

        for name in self.config.controllers:
            if name == 'bandit':
                ctrl_config = ControllerConfig(
                    control_interval_sec=self.config.control_interval_sec,
                )
                controllers[name] = ContextualBanditController(ctrl_config)

            elif name == 'fixed_eco':
                controllers[name] = FixedCapController(FixedCapConfig(power_level=0))

            elif name == 'fixed_med':
                controllers[name] = FixedCapController(FixedCapConfig(power_level=2))

            elif name == 'fixed_perf':
                controllers[name] = FixedCapController(FixedCapConfig(power_level=4))

            elif name == 'pid':
                controllers[name] = PIDController(PIDConfig(temp_setpoint=0.75))

            elif name == 'greenllm':
                controllers[name] = GreenLLMController()

            elif name == 'throttllm':
                controllers[name] = ThrottLLMController()

        return controllers

    def _generate_batch(self) -> Tuple[torch.Tensor, List[float], float]:
        """
        Generate a batch of tokens and measure performance.

        Returns:
            (output_tokens, entropies, latency_ms)
        """
        # Random prompt
        prompt = torch.randint(0, self.config.vocab_size,
                               (self.config.batch_size, self.config.seq_len // 4),
                               device=self.device)

        start_time = time.time()

        # Generate
        output, entropies = self.model.generate(prompt, max_new_tokens=32)

        # Sync CUDA
        if self.device.type == 'cuda':
            torch.cuda.synchronize()

        latency_ms = (time.time() - start_time) * 1000

        return output, entropies, latency_ms

    def run_controller_experiment(self, controller_name: str) -> ExperimentResults:
        """
        Run experiment with a specific controller.

        Args:
            controller_name: Name of controller to use

        Returns:
            ExperimentResults with all metrics
        """
        controller = self.controllers[controller_name]
        logger.info(f"\n{'='*60}")
        logger.info(f"Running experiment with controller: {controller_name}")
        logger.info(f"{'='*60}")

        # Reset
        self.telemetry.reset_energy_accounting()
        self.actuator.reset_to_default()

        # Window tracking
        windows: List[WindowMetrics] = []
        window_idx = 0

        # Metrics accumulators for current window
        window_start_time = time.time()
        window_energy_start = 0.0
        window_tokens = 0
        window_temps = []
        window_powers = []
        window_latencies = []
        window_entropies = []

        # Warmup
        logger.info(f"Warmup for {self.config.warmup_sec}s...")
        warmup_end = time.time() + self.config.warmup_sec
        while time.time() < warmup_end:
            self._generate_batch()

        # Reset after warmup
        self.telemetry.reset_energy_accounting()

        # Main experiment loop
        experiment_start = time.time()
        experiment_end = experiment_start + self.config.total_duration_sec

        logger.info(f"Running for {self.config.total_duration_sec}s...")

        while time.time() < experiment_end:
            # Read current body state
            body_state = self.telemetry.read_body_state()

            # Generate workload batch
            _, entropies, latency_ms = self._generate_batch()
            tokens_generated = self.config.batch_size * 32

            # Update telemetry with inference metrics
            self.telemetry.update_inference_state(
                tokens_generated=tokens_generated,
                latency_ms=latency_ms,
                throughput_tps=tokens_generated / (latency_ms / 1000),
                entropy=np.mean(entropies) if entropies else 0,
            )

            # Accumulate window metrics
            window_tokens += tokens_generated
            window_temps.append(body_state.telemetry.temp_c)
            window_powers.append(body_state.telemetry.power_watts)
            window_latencies.append(latency_ms)
            window_entropies.extend(entropies)

            # Check if control window is complete
            now = time.time()
            if now - window_start_time >= self.config.control_interval_sec:
                # Read final state
                body_state = self.telemetry.read_body_state()
                obs = body_state.to_observation_vector()

                # Select action
                action = controller.select_action(obs)

                # Apply action
                self.actuator.take_action_int(action)

                # Compute window metrics
                window_duration = now - window_start_time
                window_energy = body_state.energy_joules - window_energy_start

                metrics = WindowMetrics(
                    timestamp=now,
                    window_idx=window_idx,
                    energy_joules=window_energy,
                    power_watts_avg=np.mean(window_powers) if window_powers else 0,
                    tokens_generated=window_tokens,
                    throughput_tps=window_tokens / window_duration if window_duration > 0 else 0,
                    latency_ms_avg=np.mean(window_latencies) if window_latencies else 0,
                    j_per_token=window_energy / window_tokens if window_tokens > 0 else 0,
                    tokens_per_watt=window_tokens / np.mean(window_powers) if window_powers and np.mean(window_powers) > 0 else 0,
                    temp_c_avg=np.mean(window_temps) if window_temps else 0,
                    temp_c_max=max(window_temps) if window_temps else 0,
                    action=action,
                    action_name=self.ACTION_NAMES[action],
                    entropy_avg=np.mean(window_entropies) if window_entropies else 0,
                )

                # Compute reward and update controller
                reward = self._compute_reward(metrics)
                controller.update(obs, action, reward)

                windows.append(metrics)

                # Log progress
                if window_idx % 10 == 0:
                    logger.info(
                        f"  W{window_idx:3d} | A:{metrics.action_name:4s} | "
                        f"P:{metrics.power_watts_avg:5.0f}W | T:{metrics.temp_c_avg:4.1f}C | "
                        f"J/tok:{metrics.j_per_token*1000:.3f}mJ | "
                        f"tps:{metrics.throughput_tps:.0f}"
                    )

                # Reset window
                window_start_time = now
                window_energy_start = body_state.energy_joules
                window_tokens = 0
                window_temps = []
                window_powers = []
                window_latencies = []
                window_entropies = []
                window_idx += 1

        # Compile results
        if not windows:
            logger.warning("No windows recorded!")
            return None

        total_energy = sum(w.energy_joules for w in windows)
        total_tokens = sum(w.tokens_generated for w in windows)

        action_counts = np.zeros(5)
        for w in windows:
            action_counts[w.action] += 1
        action_dist = {self.ACTION_NAMES[i]: action_counts[i] / len(windows) for i in range(5)}

        results = ExperimentResults(
            controller_name=controller_name,
            total_duration_sec=time.time() - experiment_start,
            total_energy_joules=total_energy,
            total_tokens=total_tokens,
            avg_throughput_tps=total_tokens / (time.time() - experiment_start),
            avg_j_per_token=total_energy / total_tokens if total_tokens > 0 else 0,
            avg_power_watts=np.mean([w.power_watts_avg for w in windows]),
            avg_temp_c=np.mean([w.temp_c_avg for w in windows]),
            max_temp_c=max(w.temp_c_max for w in windows),
            latency_slo_violations=sum(1 for w in windows if w.latency_ms_avg > 50),
            thermal_slo_violations=sum(1 for w in windows if w.temp_c_max > 80),
            action_distribution=action_dist,
            windows=[asdict(w) for w in windows] if self.config.save_history else [],
        )

        # Log summary
        logger.info(f"\n{controller_name} Results:")
        logger.info(f"  Total energy: {results.total_energy_joules:.1f}J")
        logger.info(f"  Total tokens: {results.total_tokens}")
        logger.info(f"  Avg J/token: {results.avg_j_per_token*1000:.3f}mJ")
        logger.info(f"  Avg throughput: {results.avg_throughput_tps:.0f} tok/s")
        logger.info(f"  Avg power: {results.avg_power_watts:.0f}W")
        logger.info(f"  Avg/Max temp: {results.avg_temp_c:.1f}C / {results.max_temp_c:.1f}C")
        logger.info(f"  Action distribution: {action_dist}")

        return results

    def _compute_reward(self, metrics: WindowMetrics) -> float:
        """Compute reward for a control window."""
        # Energy efficiency (primary objective)
        energy_cost = metrics.j_per_token * 100  # Scale up for visibility

        # SLO violations
        latency_violation = max(0, metrics.latency_ms_avg - 50) / 50
        thermal_violation = max(0, (metrics.temp_c_avg - 75) / 25)

        # Total reward (negative = minimize)
        reward = -(energy_cost + 10 * latency_violation + 5 * thermal_violation)

        return reward

    def run_all(self) -> Dict[str, ExperimentResults]:
        """Run experiment with all controllers."""
        results = {}

        for controller_name in self.config.controllers:
            result = self.run_controller_experiment(controller_name)
            if result:
                results[controller_name] = result

            # Cool down between controllers
            logger.info("Cooling down for 10s...")
            self.actuator.take_action(ActuatorAction.LEVEL_0)
            time.sleep(10)

        return results

    def save_results(self, results: Dict[str, ExperimentResults]):
        """Save results to files."""
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save summary
        summary = {}
        for name, result in results.items():
            summary[name] = {
                'total_energy_joules': result.total_energy_joules,
                'total_tokens': result.total_tokens,
                'avg_j_per_token_mj': result.avg_j_per_token * 1000,
                'avg_throughput_tps': result.avg_throughput_tps,
                'avg_power_watts': result.avg_power_watts,
                'avg_temp_c': result.avg_temp_c,
                'max_temp_c': result.max_temp_c,
                'latency_slo_violations': result.latency_slo_violations,
                'thermal_slo_violations': result.thermal_slo_violations,
                'action_distribution': result.action_distribution,
            }

        summary_path = output_dir / f"summary_{timestamp}.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        logger.info(f"Saved summary to {summary_path}")

        # Save detailed results
        for name, result in results.items():
            result_path = output_dir / f"{name}_{timestamp}.json"
            with open(result_path, 'w') as f:
                json.dump(asdict(result), f, indent=2)

        # Print comparison table
        self._print_comparison(results)

    def _print_comparison(self, results: Dict[str, ExperimentResults]):
        """Print comparison table."""
        print("\n" + "=" * 80)
        print("CONTROLLER COMPARISON")
        print("=" * 80)
        print(f"{'Controller':<15} {'J/token(mJ)':<12} {'tps':<10} {'Power(W)':<10} {'Temp(C)':<10} {'SLO Viol':<10}")
        print("-" * 80)

        # Sort by J/token (lower is better)
        sorted_results = sorted(results.items(), key=lambda x: x[1].avg_j_per_token)

        best_j_per_token = sorted_results[0][1].avg_j_per_token

        for name, result in sorted_results:
            improvement = (1 - result.avg_j_per_token / best_j_per_token) * 100 if best_j_per_token > 0 else 0
            slo_viol = result.latency_slo_violations + result.thermal_slo_violations

            print(f"{name:<15} {result.avg_j_per_token*1000:<12.3f} {result.avg_throughput_tps:<10.0f} "
                  f"{result.avg_power_watts:<10.0f} {result.avg_temp_c:<10.1f} {slo_viol:<10}")

        print("=" * 80)
        print(f"Best efficiency: {sorted_results[0][0]} ({sorted_results[0][1].avg_j_per_token*1000:.3f} mJ/token)")

    def cleanup(self):
        """Clean up resources."""
        self.actuator.reset_to_default()
        self.actuator.shutdown()
        self.telemetry.shutdown()


def main():
    parser = argparse.ArgumentParser(description="Z72: Embodied Bandit Experiment")
    parser.add_argument('--duration', type=float, default=60.0, help='Experiment duration per controller (seconds)')
    parser.add_argument('--warmup', type=float, default=5.0, help='Warmup duration (seconds)')
    parser.add_argument('--controllers', nargs='+', default=None, help='Controllers to test')
    parser.add_argument('--output-dir', type=str, default='results/z72_embodied_bandit', help='Output directory')
    parser.add_argument('--batch-size', type=int, default=8, help='Batch size')
    parser.add_argument('--d-model', type=int, default=256, help='Model dimension')
    args = parser.parse_args()

    config = ExperimentConfig(
        total_duration_sec=args.duration,
        warmup_sec=args.warmup,
        controllers=args.controllers,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        d_model=args.d_model,
    )

    logger.info("=" * 60)
    logger.info("Z72: EMBODIED BANDIT EXPERIMENT")
    logger.info("=" * 60)
    logger.info(f"Duration per controller: {config.total_duration_sec}s")
    logger.info(f"Controllers: {config.controllers}")
    logger.info(f"Device: {config.device}")

    experiment = EmbodiedBanditExperiment(config)

    try:
        results = experiment.run_all()
        experiment.save_results(results)
    finally:
        experiment.cleanup()

    logger.info("\nExperiment complete!")


if __name__ == "__main__":
    main()
