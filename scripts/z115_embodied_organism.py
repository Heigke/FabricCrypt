#!/usr/bin/env python3
"""
Z115: Embodied Organism - Full FEEL-SLM Integration

This script integrates all components into a "one organism" system:
1. Truth-grade telemetry (versioned gpu_metrics + hwmon validation)
2. Phase-separated controller (prefill vs decode)
3. LayerDrop as endogenous actuator (math-as-actuation)
4. Hardware dynamics model (learned world model)
5. MPC control (predictive action selection)

The goal is to prove genuine embodiment where:
- Exogenous actuation works (DVFS power/clocks change)
- Endogenous actuation works (model reduces compute via LayerDrop)
- World-model competence (predicts energy/latency, chooses better than static)

Usage:
    HSA_OVERRIDE_GFX_VERSION=11.0.0 python scripts/z115_embodied_organism.py --mode collect
    HSA_OVERRIDE_GFX_VERSION=11.0.0 python scripts/z115_embodied_organism.py --mode train
    HSA_OVERRIDE_GFX_VERSION=11.0.0 python scripts/z115_embodied_organism.py --mode ablation

Author: FEEL Research Team
Date: 2026-01-21
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional, List, Dict
from datetime import datetime
import threading

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn.functional as F
import numpy as np

# FEEL components
from src.feel_slm.model_v2 import FEELSLMV2, FEELConfigV2, BaselineSLMV2
from src.feel_slm.gpu_metrics_v2 import TruthGradeTelemetry
from src.feel_slm.phase_controller import PhaseSeparatedController, InferencePhase
from src.feel_slm.dynamics_model import (
    HardwareDynamicsModel,
    DynamicsTrainer,
    TrajectoryBuffer,
    TrajectoryPoint,
    MPCController,
)
from src.actuator.client import ActuatorClient


# =============================================================================
# Organism State
# =============================================================================

@dataclass
class OrganismState:
    """Complete state of the embodied organism."""
    # Identity
    machine: str = "ikaros"
    session_id: str = ""

    # Hardware state
    power_w: float = 0.0
    temp_c: float = 0.0
    gpu_util: float = 0.0
    clock_mhz: int = 0

    # Control state
    phase: str = "idle"
    current_profile: str = "balanced"
    layerdrop_active: bool = False

    # Performance metrics
    tokens_generated: int = 0
    total_energy_j: float = 0.0
    mj_per_token: float = 0.0
    avg_tpot_ms: float = 0.0

    # Controller state
    slo_violations: int = 0
    dynamics_predictions: Dict = None


# =============================================================================
# Embodied Organism
# =============================================================================

class EmbodiedOrganism:
    """
    The complete FEEL-SLM embodied organism.

    Integrates:
    - Sensing (truth-grade telemetry)
    - Actuation (DVFS + LayerDrop)
    - Prediction (dynamics model)
    - Control (phase-separated + MPC)
    """

    def __init__(
        self,
        device: torch.device,
        actuator_port: int = 8770,
        use_mpc: bool = False,
    ):
        self.device = device
        self.use_mpc = use_mpc

        # State
        self.state = OrganismState(
            session_id=datetime.now().strftime("%Y%m%d_%H%M%S")
        )

        # Components
        self.telemetry = TruthGradeTelemetry()  # Auto-detect card
        self.actuator = ActuatorClient("localhost", actuator_port, auto_heartbeat=True)
        self.model: Optional[FEELSLMV2] = None
        self.baseline_model: Optional[BaselineSLMV2] = None
        self.dynamics_model: Optional[HardwareDynamicsModel] = None

        # Controller
        self.controller: Optional[PhaseSeparatedController] = None

        # Trajectory collection
        self.trajectory_buffer = TrajectoryBuffer()

        # Power integration
        self._power_samples: List[tuple] = []
        self._sampling = False
        self._sample_thread = None

        # Check actuator
        if not self.actuator.is_available():
            print("WARNING: Actuator not available. Running in sensor-only mode.")
            self.actuator = None

    def create_model(self, config: FEELConfigV2) -> FEELSLMV2:
        """Create FEEL model."""
        self.model = FEELSLMV2(config).to(self.device)
        self.model.eval()

        # Also create baseline for comparison
        self.baseline_model = BaselineSLMV2(config).to(self.device)
        self.baseline_model.eval()

        return self.model

    def create_controller(self):
        """Create phase-separated controller."""
        def on_phase_change(phase: InferencePhase):
            self.state.phase = phase.name
            if phase == InferencePhase.DECODE and self.model:
                # Enable LayerDrop in decode phase
                self.model.set_mode("eco")
                self.state.layerdrop_active = True
            else:
                if self.model:
                    self.model.set_mode("balanced")
                self.state.layerdrop_active = False

        self.controller = PhaseSeparatedController(
            actuator=self.actuator,
            telemetry_source=self.telemetry,
            model=self.model,
            on_phase_change=on_phase_change,
        )

        return self.controller

    def load_dynamics_model(self, path: Path):
        """Load trained dynamics model."""
        self.dynamics_model = HardwareDynamicsModel()
        checkpoint = torch.load(path, map_location=self.device)
        self.dynamics_model.load_state_dict(checkpoint["model_state"])
        self.dynamics_model.to(self.device)
        self.dynamics_model.eval()

        if self.use_mpc:
            self.mpc = MPCController(self.dynamics_model)

    def _start_power_sampling(self, interval: float = 0.02):
        """Start background power sampling."""
        self._power_samples.clear()
        self._sampling = True

        def sample_loop():
            while self._sampling:
                reading = self.telemetry.read()
                self._power_samples.append((time.time(), reading["power_w"]))
                time.sleep(interval)

        self._sample_thread = threading.Thread(target=sample_loop, daemon=True)
        self._sample_thread.start()

    def _stop_power_sampling(self) -> float:
        """Stop sampling and return integrated energy (J)."""
        self._sampling = False
        if self._sample_thread:
            self._sample_thread.join(timeout=1.0)

        if len(self._power_samples) < 2:
            return 0.0

        # Trapezoid integration
        total_energy = 0.0
        for i in range(1, len(self._power_samples)):
            t_prev, p_prev = self._power_samples[i-1]
            t_curr, p_curr = self._power_samples[i]
            dt = t_curr - t_prev
            avg_power = (p_prev + p_curr) / 2.0
            total_energy += avg_power * dt

        return total_energy

    def run_generation(
        self,
        prompt_tokens: torch.Tensor,
        max_new_tokens: int = 64,
        use_feel: bool = True,
        fixed_profile: Optional[str] = None,
    ) -> Dict:
        """
        Run token generation with full organism control.

        Returns metrics including energy, latency, and control decisions.
        """
        model = self.model if use_feel else self.baseline_model
        model.eval()

        # Start sampling
        self._start_power_sampling()

        input_ids = prompt_tokens.to(self.device)
        generated = input_ids.clone()

        # Initialize controller
        if self.controller and use_feel:
            self.controller.start_request(prompt_length=input_ids.shape[1])

        # Set fixed profile if specified
        if fixed_profile and self.actuator:
            self.actuator.set_profile(fixed_profile)

        # Set LayerDrop mode (only FEEL model has set_mode)
        if use_feel:
            if fixed_profile == "eco":
                model.set_mode("eco")
            else:
                model.set_mode("balanced")

        tokens_generated = 0
        all_tpots = []
        trajectory_points = []
        run_start = time.time()

        try:
            for token_idx in range(max_new_tokens):
                token_start = time.time()

                # Get current telemetry
                telem = self.telemetry.read()
                self.state.power_w = telem["power_w"]
                self.state.temp_c = telem["temp_c"]
                self.state.gpu_util = telem["gpu_util"]
                self.state.clock_mhz = telem["clock_gfx_mhz"]

                # Forward pass
                with torch.no_grad():
                    if use_feel:
                        # Create telemetry vector
                        telemetry_vec = torch.tensor([
                            telem["power_w"] / 100.0,
                            telem["temp_c"] / 100.0,
                            telem["gpu_util"] / 100.0,
                            telem["clock_gfx_mhz"] / 2500.0,
                            0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 0.5, 0.5,
                        ], dtype=torch.float32, device=self.device).unsqueeze(0)

                        outputs = model(generated, telemetry_vec, return_all=True)
                    else:
                        outputs = model(generated)

                    logits = outputs["logits"][:, -1, :]
                    probs = F.softmax(logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)

                generated = torch.cat([generated, next_token], dim=1)
                tokens_generated += 1

                token_end = time.time()
                tpot_ms = (token_end - token_start) * 1000
                all_tpots.append(tpot_ms)

                # Controller updates
                if self.controller and use_feel:
                    if token_idx == 0:
                        ttft_ms = (token_end - run_start) * 1000
                        self.controller.on_first_token(ttft_ms)
                    else:
                        self.controller.on_token_generated(tpot_ms, telem["power_w"])

                    # Control window check
                    if (token_idx + 1) % 32 == 0:
                        decision = self.controller.on_control_window()

                        # MPC override if available
                        if self.use_mpc and self.dynamics_model:
                            profile, layerdrop, preds = self.mpc.select_action(telem)
                            self.state.dynamics_predictions = preds

                            if profile != self.state.current_profile:
                                if self.actuator:
                                    self.actuator.set_profile(profile)
                                self.state.current_profile = profile

                            if layerdrop != self.state.layerdrop_active:
                                model.set_mode("eco" if layerdrop else "balanced")
                                self.state.layerdrop_active = layerdrop

                # Collect trajectory point
                trajectory_points.append(TrajectoryPoint(
                    timestamp=token_end,
                    power_w=telem["power_w"],
                    temp_c=telem["temp_c"],
                    gpu_util=telem["gpu_util"],
                    clock_mhz=telem["clock_gfx_mhz"],
                    profile=self.state.current_profile,
                    layerdrop_active=self.state.layerdrop_active,
                    tokens_in_window=tokens_generated % 32,
                    prompt_length=input_ids.shape[1],
                    phase=self.state.phase,
                    energy_mj=tpot_ms * telem["power_w"],  # Approximate
                    latency_ms=tpot_ms,
                ))

        finally:
            # Stop sampling and get energy
            total_energy_j = self._stop_power_sampling()

            # End controller request
            if self.controller and use_feel:
                self.controller.end_request()

        # Update trajectory with next-state info
        for i in range(len(trajectory_points) - 1):
            trajectory_points[i].next_power_w = trajectory_points[i+1].power_w
            trajectory_points[i].next_temp_c = trajectory_points[i+1].temp_c
            trajectory_points[i].next_gpu_util = trajectory_points[i+1].gpu_util

        # Add to buffer
        for p in trajectory_points:
            self.trajectory_buffer.add(p)

        # Compute metrics
        duration = time.time() - run_start
        avg_tpot = sum(all_tpots) / len(all_tpots) if all_tpots else 0
        mj_per_token = (total_energy_j * 1000) / tokens_generated if tokens_generated > 0 else 0

        # Update state
        self.state.tokens_generated += tokens_generated
        self.state.total_energy_j += total_energy_j
        self.state.mj_per_token = mj_per_token
        self.state.avg_tpot_ms = avg_tpot

        return {
            "tokens": tokens_generated,
            "energy_j": total_energy_j,
            "mj_per_token": mj_per_token,
            "duration_s": duration,
            "avg_power_w": total_energy_j / duration if duration > 0 else 0,
            "avg_tpot_ms": avg_tpot,
            "tpot_p95_ms": sorted(all_tpots)[int(len(all_tpots) * 0.95)] if all_tpots else 0,
            "layerdrop_active": self.state.layerdrop_active,
            "profile": self.state.current_profile,
        }

    def run_ablation(
        self,
        n_prompts: int = 20,
        max_tokens: int = 64,
    ) -> Dict:
        """
        Run ablation study comparing all configurations.

        Configurations:
        1. Baseline + fixed eco (DVFS only)
        2. Baseline + fixed perf (DVFS only)
        3. FEEL + fixed eco + LayerDrop (compute only)
        4. FEEL + fixed perf (neither)
        5. FEEL + phase-separated (both)
        6. FEEL + MPC (both + learned)
        """
        results = {}
        vocab_size = self.model.config.vocab_size

        configs = [
            ("baseline_eco", False, "eco"),
            ("baseline_perf", False, "performance"),
            ("feel_eco_layerdrop", True, "eco"),
            ("feel_perf", True, "performance"),
            ("feel_phased", True, None),  # Phase-separated
        ]

        if self.dynamics_model:
            configs.append(("feel_mpc", True, None))  # MPC control

        for name, use_feel, fixed_profile in configs:
            print(f"\n{'='*50}")
            print(f"Running: {name}")
            print(f"{'='*50}")

            # Set MPC mode
            self.use_mpc = (name == "feel_mpc")

            metrics = {
                "tokens": 0,
                "energy_j": 0.0,
                "duration_s": 0.0,
                "tpots": [],
            }

            # Reset to known state
            if self.actuator:
                time.sleep(0.6)
                self.actuator.set_profile(fixed_profile or "balanced")
                time.sleep(1.0)

            for i in range(n_prompts):
                # Random prompt
                prompt = torch.randint(100, vocab_size - 100, (1, 32))

                result = self.run_generation(
                    prompt_tokens=prompt,
                    max_new_tokens=max_tokens,
                    use_feel=use_feel,
                    fixed_profile=fixed_profile,
                )

                metrics["tokens"] += result["tokens"]
                metrics["energy_j"] += result["energy_j"]
                metrics["duration_s"] += result["duration_s"]
                metrics["tpots"].append(result["avg_tpot_ms"])

                if (i + 1) % 5 == 0:
                    print(f"  Progress: {i+1}/{n_prompts}, "
                          f"mJ/tok={result['mj_per_token']:.1f}, "
                          f"power={result['avg_power_w']:.1f}W")

            # Aggregate
            results[name] = {
                "tokens": metrics["tokens"],
                "energy_j": metrics["energy_j"],
                "mj_per_token": (metrics["energy_j"] * 1000) / metrics["tokens"],
                "avg_power_w": metrics["energy_j"] / metrics["duration_s"],
                "avg_tpot_ms": sum(metrics["tpots"]) / len(metrics["tpots"]),
                "duration_s": metrics["duration_s"],
            }

        return results

    def get_state(self) -> Dict:
        """Get current organism state."""
        return asdict(self.state)

    def save_trajectory(self, path: Path):
        """Save collected trajectory data."""
        self.trajectory_buffer.save(path)

    def shutdown(self):
        """Clean shutdown."""
        if self.actuator:
            self.actuator.stop()


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Embodied Organism")
    parser.add_argument("--mode", choices=["collect", "train", "ablation", "demo"],
                       default="demo", help="Operation mode")
    parser.add_argument("--duration", type=int, default=60, help="Collection duration (seconds)")
    parser.add_argument("--output-dir", type=str, default="results/z115_organism")
    parser.add_argument("--dynamics-model", type=str, default=None, help="Path to trained dynamics model")
    args = parser.parse_args()

    # Check device
    if not torch.cuda.is_available():
        print("ERROR: CUDA not available")
        sys.exit(1)

    device = torch.device("cuda")
    print(f"Device: {torch.cuda.get_device_name(0)}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create organism
    organism = EmbodiedOrganism(device, use_mpc=(args.dynamics_model is not None))

    # Create model
    config = FEELConfigV2(
        vocab_size=32000,
        hidden_dim=256,
        num_layers=4,
        num_heads=4,
        num_kv_heads=2,
        intermediate_dim=512,
        max_seq_len=256,
        phase=1,
        enable_layerdrop=True,
        layerdrop_layers=[1, 2],
    )
    organism.create_model(config)
    organism.create_controller()

    # Load dynamics model if provided
    if args.dynamics_model:
        organism.load_dynamics_model(Path(args.dynamics_model))
        print(f"Loaded dynamics model from {args.dynamics_model}")

    print(f"\nTelemetry capabilities: {organism.telemetry.get_capabilities()}")
    print(f"Actuator available: {organism.actuator is not None}")

    if args.mode == "collect":
        # Collect trajectory data
        print(f"\n{'='*60}")
        print("Collecting Trajectory Data")
        print(f"{'='*60}")

        start_time = time.time()
        prompt_count = 0

        while time.time() - start_time < args.duration:
            prompt = torch.randint(100, config.vocab_size - 100, (1, 32))
            result = organism.run_generation(prompt, max_new_tokens=64, use_feel=True)
            prompt_count += 1

            print(f"[{time.time() - start_time:.1f}s] Prompt {prompt_count}: "
                  f"{result['mj_per_token']:.1f} mJ/tok, {result['avg_power_w']:.1f}W")

        # Save trajectory
        traj_path = output_dir / "trajectory.json"
        organism.save_trajectory(traj_path)
        print(f"\n✅ Saved {len(organism.trajectory_buffer)} trajectory points to {traj_path}")

    elif args.mode == "train":
        # Train dynamics model on collected data
        print(f"\n{'='*60}")
        print("Training Dynamics Model")
        print(f"{'='*60}")

        # Load trajectory
        traj_path = output_dir / "trajectory.json"
        if not traj_path.exists():
            print(f"ERROR: No trajectory data at {traj_path}")
            print("Run with --mode collect first")
            sys.exit(1)

        organism.trajectory_buffer.load(traj_path)
        print(f"Loaded {len(organism.trajectory_buffer)} trajectory points")

        # Train
        dynamics = HardwareDynamicsModel()
        trainer = DynamicsTrainer(dynamics, device)

        for epoch in range(50):
            result = trainer.train_epoch(organism.trajectory_buffer)
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}: loss={result['loss']:.4f}")

        # Save
        model_path = output_dir / "dynamics_model.pt"
        trainer.save(model_path)
        print(f"\n✅ Saved dynamics model to {model_path}")

    elif args.mode == "ablation":
        # Run full ablation study
        print(f"\n{'='*60}")
        print("Running Ablation Study")
        print(f"{'='*60}")

        results = organism.run_ablation(n_prompts=20, max_tokens=64)

        # Print results
        print(f"\n{'='*60}")
        print("ABLATION RESULTS")
        print(f"{'='*60}")

        for name, metrics in results.items():
            print(f"\n{name}:")
            print(f"  mJ/token: {metrics['mj_per_token']:.1f}")
            print(f"  Avg power: {metrics['avg_power_w']:.1f}W")
            print(f"  Avg TPOT: {metrics['avg_tpot_ms']:.2f}ms")
            print(f"  Tokens: {metrics['tokens']}")

        # Compute relative improvements
        if "baseline_eco" in results and "feel_eco_layerdrop" in results:
            baseline = results["baseline_eco"]["mj_per_token"]
            feel = results["feel_eco_layerdrop"]["mj_per_token"]
            improvement = (baseline - feel) / baseline * 100
            print(f"\n📊 FEEL+LayerDrop vs Baseline: {improvement:+.1f}% mJ/token")

        if "feel_phased" in results:
            print(f"📊 Phase-separated: {results['feel_phased']['mj_per_token']:.1f} mJ/token")

        if "feel_mpc" in results:
            print(f"📊 MPC-controlled: {results['feel_mpc']['mj_per_token']:.1f} mJ/token")

        # Save results
        results_path = output_dir / "ablation_results.json"
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\n✅ Results saved to {results_path}")

    else:  # demo mode
        # Quick demo
        print(f"\n{'='*60}")
        print("Demo: Single Generation")
        print(f"{'='*60}")

        prompt = torch.randint(100, config.vocab_size - 100, (1, 32))

        print("\nBaseline (no FEEL):")
        result = organism.run_generation(prompt, max_new_tokens=64, use_feel=False)
        print(f"  mJ/token: {result['mj_per_token']:.1f}")
        print(f"  Avg power: {result['avg_power_w']:.1f}W")

        print("\nFEEL with LayerDrop:")
        result = organism.run_generation(prompt, max_new_tokens=64, use_feel=True, fixed_profile="eco")
        print(f"  mJ/token: {result['mj_per_token']:.1f}")
        print(f"  Avg power: {result['avg_power_w']:.1f}W")
        print(f"  LayerDrop active: {result['layerdrop_active']}")

    organism.shutdown()
    print("\nDone.")


if __name__ == "__main__":
    main()
