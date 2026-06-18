#!/usr/bin/env python3 -u
"""
Track B: Sustained Regulation Tests with Perturbations

Runs 10-20 minute sustained decode loops under:
- Background load on/off (GPU stress)
- Forced DVFS changes at scheduled times

Measures:
- Temp overshoot above target
- Time above threshold
- Oscillation count
- Settling time
- Energy/token
- p95 latency stability

Usage:
    python scripts/regulation_test.py --duration 300  # 5 min test
    python scripts/regulation_test.py --duration 600 --with_perturbations  # 10 min with load
"""

import sys
import time
import json
import random
import argparse
import threading
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from collections import deque
import numpy as np

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.stdout.reconfigure(line_buffering=True)

sys.path.insert(0, str(Path(__file__).parent.parent))

from src import (
    FEELProjectorFull,
    TelemetrySampler,
    CanonicalSensorBank,
    RuntimeContext,
    HardwareContext,
    HysteresisController,
    ControllerConfig,
    DVFSController,
    RegulationMetrics,
    compute_regulation_metrics,
    CONTROLLER_VERSION,
)

VERSION = f"regulation-test-{CONTROLLER_VERSION}"


PROMPTS = [
    "Explain how neural networks learn through backpropagation in detail.",
    "Write a comprehensive guide to quantum computing for beginners.",
    "Describe the complete process of photosynthesis at the molecular level.",
    "Explain the theory of general relativity and its implications.",
    "Write a detailed analysis of climate change causes and solutions.",
    "Describe how modern CPUs execute instructions with pipelining.",
    "Explain the human immune system and how vaccines work.",
    "Write about the history and future of artificial intelligence.",
    "Describe the process of protein synthesis in cells.",
    "Explain how blockchain technology works in detail.",
]


class GPULoadGenerator:
    """Generate GPU load for perturbation testing."""

    def __init__(self):
        self._process = None
        self._running = False

    def start(self):
        """Start GPU stress load."""
        if self._running:
            return

        # Use rocm-smi or a simple GPU compute kernel
        # For now, use a simple PyTorch operation
        self._running = True
        self._thread = threading.Thread(target=self._run_load)
        self._thread.daemon = True
        self._thread.start()
        print("  [LOAD] GPU stress started")

    def _run_load(self):
        """Run GPU compute load in background."""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Large matrix multiplications to stress GPU
        size = 4096
        a = torch.randn(size, size, device=device, dtype=torch.float16)
        b = torch.randn(size, size, device=device, dtype=torch.float16)

        while self._running:
            _ = torch.matmul(a, b)
            torch.cuda.synchronize()
            time.sleep(0.01)  # Small sleep to prevent complete lockup

    def stop(self):
        """Stop GPU stress load."""
        self._running = False
        if hasattr(self, '_thread'):
            self._thread.join(timeout=2.0)
        print("  [LOAD] GPU stress stopped")


class RegulationTester:
    """
    Run sustained regulation tests with perturbations.
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-3B-Instruct",
        checkpoint_path: str = None,
        device: str = "cuda",
    ):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        print(f"Loading model {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16 if self.device.type == "cuda" else torch.float32,
            device_map="auto",
            trust_remote_code=True,
        )
        self.model.eval()

        self.embed_dim = self.model.config.hidden_size

        # Projector
        self.projector = FEELProjectorFull(embed_dim=self.embed_dim).to(self.device)
        if checkpoint_path and Path(checkpoint_path).exists():
            ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
            if 'projector_state_dict' in ckpt:
                self.projector.load_state_dict(ckpt['projector_state_dict'])
            print(f"  Loaded projector: {ckpt.get('version', 'unknown')}")

        # Sensor bank
        self.sensor_bank = CanonicalSensorBank(mode="full")

        # Telemetry
        self.telemetry = None
        try:
            self.telemetry = TelemetrySampler(sample_hz=30)
            self.telemetry.start()
            print(f"  Telemetry: {self.telemetry.source}")
        except Exception as e:
            print(f"  Telemetry unavailable: {e}")

        # Controller
        self.controller = HysteresisController(ControllerConfig())

        # Load generator
        self.load_gen = GPULoadGenerator()

    def run_sustained_generation(
        self,
        duration_seconds: float,
        with_controller: bool = True,
        perturbation_schedule: List[Tuple[float, str]] = None,
    ) -> Dict:
        """
        Run sustained token generation for specified duration.

        Args:
            duration_seconds: how long to run
            with_controller: use hysteresis controller
            perturbation_schedule: list of (time_offset, action) tuples
                actions: "load_on", "load_off", "dvfs_low", "dvfs_auto", "dvfs_high"

        Returns:
            Dict with traces and metrics
        """
        print(f"\n{'='*60}")
        print(f"  SUSTAINED GENERATION TEST")
        print(f"  Duration: {duration_seconds}s, Controller: {with_controller}")
        print(f"{'='*60}")

        start_time = time.time()
        self.controller.reset()

        # Traces
        temps = []
        powers = []
        utils = []
        latencies = []
        actions = []
        timestamps = []
        tokens_generated = 0

        # Perturbation tracking
        perturbation_idx = 0
        perturbations_applied = []

        prompt_idx = 0

        while (time.time() - start_time) < duration_seconds:
            elapsed = time.time() - start_time

            # Check for scheduled perturbations
            if perturbation_schedule and perturbation_idx < len(perturbation_schedule):
                next_time, next_action = perturbation_schedule[perturbation_idx]
                if elapsed >= next_time:
                    self._apply_perturbation(next_action)
                    perturbations_applied.append((elapsed, next_action))
                    perturbation_idx += 1

            # Get prompt
            prompt = PROMPTS[prompt_idx % len(PROMPTS)]
            prompt_idx += 1

            # Generate one sequence
            input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
            current_ids = input_ids.clone()

            for step in range(100):  # Max tokens per prompt
                if (time.time() - start_time) >= duration_seconds:
                    break

                t0 = time.time()
                with torch.no_grad():
                    outputs = self.model(current_ids, use_cache=False)
                    logits = outputs.logits[:, -1, :].float()
                t1 = time.time()

                # Get telemetry
                hw = self.telemetry.get_token_aligned(t0, t1) if self.telemetry else {}
                temp = hw.get("temp")
                power = hw.get("power")
                util = hw.get("util")

                # Record traces
                temps.append(temp)
                powers.append(power)
                utils.append(util)
                latencies.append(t1 - t0)
                timestamps.append(time.time() - start_time)

                # Controller step
                if with_controller and temp is not None:
                    dvfs, K = self.controller.step(temp, power, util, phase="decode")
                    actions.append({"dvfs": dvfs, "K": K, "state": self.controller.state.value})
                else:
                    actions.append({"dvfs": "auto", "K": 1, "state": "none"})

                tokens_generated += 1

                # Sample next token
                probs = F.softmax(logits / 0.7, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                current_ids = torch.cat([current_ids, next_token], dim=-1)

                if next_token.item() == self.tokenizer.eos_token_id:
                    break

            # Progress
            if tokens_generated % 100 == 0:
                current_temp = temps[-1] if temps else 0
                current_state = actions[-1]["state"] if actions else "none"
                print(f"  [{elapsed:.1f}s] {tokens_generated} tokens, "
                      f"T={current_temp:.1f}°C, state={current_state}")

        # Compute metrics
        total_duration = time.time() - start_time
        metrics = compute_regulation_metrics(
            temps=temps,
            powers=powers,
            latencies=latencies,
            transitions=self.controller.get_history(),
            target_temp=self.controller.config.T_high,
            duration_seconds=total_duration,
        )

        results = {
            "duration_seconds": total_duration,
            "tokens_generated": tokens_generated,
            "with_controller": with_controller,
            "perturbations": perturbations_applied,
            "traces": {
                "temps": temps,
                "powers": powers,
                "utils": utils,
                "latencies": latencies,
                "timestamps": timestamps,
                "actions": actions,
            },
            "metrics": {
                "temp_mean": metrics.temp_mean,
                "temp_max": metrics.temp_max,
                "temp_overshoot": metrics.temp_overshoot,
                "time_above_target": metrics.time_above_target,
                "oscillation_count": metrics.oscillation_count,
                "power_mean": metrics.power_mean,
                "energy_total": metrics.energy_total,
                "tokens_per_second": metrics.tokens_per_second,
                "latency_p95": metrics.latency_p95,
            },
            "controller_history": self.controller.get_history(),
        }

        return results

    def _apply_perturbation(self, action: str):
        """Apply a perturbation action."""
        print(f"  [PERTURBATION] {action}")

        if action == "load_on":
            self.load_gen.start()
        elif action == "load_off":
            self.load_gen.stop()
        elif action == "dvfs_low":
            self.controller.dvfs.set_mode("low")
        elif action == "dvfs_auto":
            self.controller.dvfs.set_mode("auto")
        elif action == "dvfs_high":
            self.controller.dvfs.set_mode("high")

    def run_comparison(
        self,
        duration_seconds: float,
        with_perturbations: bool = False,
    ) -> Dict:
        """
        Compare controller vs baseline.
        """
        print("\n" + "=" * 60)
        print("  REGULATION COMPARISON TEST")
        print("=" * 60)

        # Define perturbation schedule if requested
        if with_perturbations:
            # Alternate load every 60 seconds
            schedule = []
            t = 30
            while t < duration_seconds:
                schedule.append((t, "load_on"))
                t += 30
                if t < duration_seconds:
                    schedule.append((t, "load_off"))
                    t += 30
            print(f"  Perturbation schedule: {len(schedule)} events")
        else:
            schedule = None

        # Run without controller (baseline)
        print("\n--- Baseline (no controller) ---")
        baseline = self.run_sustained_generation(
            duration_seconds=duration_seconds / 2,
            with_controller=False,
            perturbation_schedule=schedule[:len(schedule)//2] if schedule else None,
        )

        # Cool down
        print("\n  Cooling down (30s)...")
        time.sleep(30)

        # Run with controller
        print("\n--- With Controller ---")
        controlled = self.run_sustained_generation(
            duration_seconds=duration_seconds / 2,
            with_controller=True,
            perturbation_schedule=schedule[len(schedule)//2:] if schedule else None,
        )

        # Compare
        print("\n" + "=" * 60)
        print("  COMPARISON RESULTS")
        print("=" * 60)

        comparison = {
            "baseline": baseline["metrics"],
            "controlled": controlled["metrics"],
            "improvement": {},
        }

        print(f"\n  {'Metric':<25} {'Baseline':>12} {'Controlled':>12} {'Change':>12}")
        print(f"  {'-'*61}")

        for key in baseline["metrics"]:
            b_val = baseline["metrics"][key]
            c_val = controlled["metrics"][key]

            # For most metrics, lower is better
            if key in ["tokens_per_second"]:
                # Higher is better
                improvement = (c_val - b_val) / b_val * 100 if b_val else 0
            else:
                # Lower is better
                improvement = (b_val - c_val) / b_val * 100 if b_val else 0

            comparison["improvement"][key] = improvement

            print(f"  {key:<25} {b_val:>12.3f} {c_val:>12.3f} {improvement:>+11.1f}%")

        return {
            "baseline": baseline,
            "controlled": controlled,
            "comparison": comparison,
        }


def main():
    parser = argparse.ArgumentParser(description="Regulation Test")
    parser.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--checkpoint", default="results/feel_training_v10/final_checkpoint.pt")
    parser.add_argument("--duration", type=int, default=120, help="Duration in seconds")
    parser.add_argument("--with_perturbations", action="store_true")
    parser.add_argument("--mode", choices=["single", "compare"], default="compare")
    parser.add_argument("--output_dir", default="results/regulation")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    tester = RegulationTester(
        model_name=args.model,
        checkpoint_path=args.checkpoint,
    )

    if args.mode == "single":
        results = tester.run_sustained_generation(
            duration_seconds=args.duration,
            with_controller=True,
        )
    else:
        results = tester.run_comparison(
            duration_seconds=args.duration,
            with_perturbations=args.with_perturbations,
        )

    # Save results (traces can be large, save separately)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = f"{args.output_dir}/regulation_{timestamp}.json"

    # Save summary (without full traces)
    summary = {
        "version": VERSION,
        "timestamp": datetime.now().isoformat(),
        "duration": args.duration,
        "with_perturbations": args.with_perturbations,
    }

    if args.mode == "compare":
        summary["comparison"] = results["comparison"]
        summary["baseline_metrics"] = results["baseline"]["metrics"]
        summary["controlled_metrics"] = results["controlled"]["metrics"]
    else:
        summary["metrics"] = results["metrics"]

    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nSaved: {output_path}")

    # Save full traces separately
    traces_path = f"{args.output_dir}/traces_{timestamp}.npz"
    if args.mode == "compare":
        np.savez_compressed(
            traces_path,
            baseline_temps=results["baseline"]["traces"]["temps"],
            baseline_latencies=results["baseline"]["traces"]["latencies"],
            controlled_temps=results["controlled"]["traces"]["temps"],
            controlled_latencies=results["controlled"]["traces"]["latencies"],
        )
    else:
        np.savez_compressed(
            traces_path,
            temps=results["traces"]["temps"],
            latencies=results["traces"]["latencies"],
        )

    print(f"Saved traces: {traces_path}")

    # Cleanup
    tester.load_gen.stop()
    if tester.telemetry:
        tester.telemetry.stop()


if __name__ == "__main__":
    main()
