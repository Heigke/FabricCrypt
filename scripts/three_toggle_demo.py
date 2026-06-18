#!/usr/bin/env python3 -u
"""
Three-Toggle Demo Framework for FEEL Loop Demonstration

Shows the closed-loop homeostasis in action with falsification controls.

THREE MODES (toggleable):
1. LIVE - Real-time telemetry → z_feel → actions (closed loop)
2. CROSS-SWAP - Telemetry from different prompt (breaks causal link)
3. DISABLED - z_feel always zero (open loop baseline)

FOUR PLOTS:
1. Temperature/Power/Util time series
2. z_feel projection (2D PCA of embedding)
3. Action timeline (K, ponder_steps, DVFS actions)
4. Output behavior proxy (entropy, token rate)

Usage:
    python scripts/three_toggle_demo.py --mode live
    python scripts/three_toggle_demo.py --mode cross_swap
    python scripts/three_toggle_demo.py --mode disabled
    python scripts/three_toggle_demo.py --all  # Run all three for comparison
"""

import sys
import time
import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import numpy as np

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, str(Path(__file__).parent.parent))

from src import (
    TelemetrySampler,
    CanonicalSensorBank,
    RuntimeContext,
    HardwareContext,
    FEELProjectorFull,
)
from src.dual_axis_policy import DualAxisPolicyRuleBased, DualAxisConfig, DualAxisAction
from src.compute_effector import KSamplingEffector
from src.perturbation_scheduler import PerturbationSchedule, PerturbationController, ScheduleExecutor

VERSION = "three-toggle-demo-v1.0.0"


class DemoMode(Enum):
    LIVE = "live"           # Real telemetry → z_feel → actions
    CROSS_SWAP = "cross_swap"  # Telemetry from different context
    DISABLED = "disabled"   # z_feel always zero


@dataclass
class DemoTrace:
    """Traces collected during demo run."""
    timestamps: List[float] = field(default_factory=list)
    temps: List[float] = field(default_factory=list)
    powers: List[float] = field(default_factory=list)
    utils: List[float] = field(default_factory=list)

    z_feel_norms: List[float] = field(default_factory=list)
    z_feel_projections: List[Tuple[float, float]] = field(default_factory=list)

    K_values: List[int] = field(default_factory=list)
    ponder_steps: List[int] = field(default_factory=list)
    dvfs_actions: List[str] = field(default_factory=list)

    entropies: List[float] = field(default_factory=list)
    token_times: List[float] = field(default_factory=list)

    difficulty_scores: List[float] = field(default_factory=list)
    stress_scores: List[float] = field(default_factory=list)
    net_efforts: List[float] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            "timestamps": self.timestamps,
            "temps": self.temps,
            "powers": self.powers,
            "utils": self.utils,
            "z_feel_norms": self.z_feel_norms,
            "z_feel_projections": self.z_feel_projections,
            "K_values": self.K_values,
            "ponder_steps": self.ponder_steps,
            "dvfs_actions": self.dvfs_actions,
            "entropies": self.entropies,
            "token_times": self.token_times,
            "difficulty_scores": self.difficulty_scores,
            "stress_scores": self.stress_scores,
            "net_efforts": self.net_efforts,
        }


class ThreeToggleDemo:
    """
    Interactive demo showing FEEL loop with three toggle modes.
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-3B-Instruct",
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

        # Sensor bank + FEEL projector (sensor vector → z_feel)
        self.sensor_bank = CanonicalSensorBank(mode="full")  # 16-dim with hardware
        self.projector = FEELProjectorFull(embed_dim=self.embed_dim).to(self.device)

        # Dual-axis policy (rule-based for demo)
        self.policy = DualAxisPolicyRuleBased(DualAxisConfig(
            K_difficulty_threshold=0.4,
            ponder_difficulty_threshold=0.7,
            dvfs_stress_threshold=0.6,
            concise_stress_threshold=0.5,
        ))

        # Compute effector
        self.k_effector = KSamplingEffector(self.model, self.tokenizer, self.device)

        # Perturbation controller
        self.perturb = PerturbationController(device)

        # Telemetry
        self.telemetry = TelemetrySampler(sample_hz=30)
        self.telemetry.start()
        print(f"  Telemetry: {self.telemetry.source}")

        # PCA for z_feel projection (2D)
        self._pca_components = None
        self._init_pca()

        # Cross-swap replay buffer
        self._replay_buffer: List[Dict] = []

    def _init_pca(self):
        """Initialize random projection for z_feel visualization."""
        # Use random projection (proper PCA would need samples)
        torch.manual_seed(42)
        self._pca_components = torch.randn(2, self.embed_dim, device=self.device)
        self._pca_components = F.normalize(self._pca_components, dim=1)

    def _project_z_feel(self, z_feel: torch.Tensor) -> Tuple[float, float]:
        """Project z_feel to 2D for visualization."""
        proj = torch.matmul(self._pca_components, z_feel.squeeze())
        return (proj[0].item(), proj[1].item())

    def collect_cross_swap_data(self, prompt: str, duration: float = 60.0):
        """
        Collect telemetry data for cross-swap mode.

        This runs a separate generation to build a replay buffer
        that will be used when mode=CROSS_SWAP.
        """
        print(f"\n  Collecting cross-swap replay data ({duration}s)...")

        self._replay_buffer = []
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        current_ids = input_ids.clone()

        start_time = time.time()
        while (time.time() - start_time) < duration:
            t0 = time.time()
            with torch.no_grad():
                outputs = self.model(current_ids, use_cache=False)
                logits = outputs.logits[:, -1, :].float()
            t1 = time.time()

            hw = self.telemetry.get_token_aligned(t0, t1)
            self._replay_buffer.append(hw)

            probs = F.softmax(logits / 0.7, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            current_ids = torch.cat([current_ids, next_token], dim=-1)

            if current_ids.shape[1] > 512:
                current_ids = input_ids.clone()

        print(f"  Collected {len(self._replay_buffer)} replay samples")

    def run_demo(
        self,
        prompt: str,
        mode: DemoMode,
        duration: float = 120.0,
        with_perturbations: bool = True,
    ) -> DemoTrace:
        """
        Run demo in specified mode.

        Args:
            prompt: Generation prompt
            mode: LIVE, CROSS_SWAP, or DISABLED
            duration: Run duration in seconds
            with_perturbations: Whether to apply perturbation schedule
        """
        print(f"\n{'='*60}")
        print(f"  THREE-TOGGLE DEMO: {mode.value.upper()}")
        print(f"{'='*60}")

        self.policy.reset()
        trace = DemoTrace()

        # Setup perturbation schedule (aggressive by default for validation)
        if with_perturbations:
            schedule = PerturbationSchedule.aggressive_validation(duration=duration)
            executor = ScheduleExecutor(schedule, self.perturb)
            executor.start()
            print(f"  Schedule: {schedule.description}")

        # Cross-swap needs replay buffer
        if mode == DemoMode.CROSS_SWAP and not self._replay_buffer:
            print("  WARNING: No replay buffer for cross-swap, using live data")
            mode = DemoMode.LIVE

        replay_idx = 0

        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        current_ids = input_ids.clone()

        start_time = time.time()
        token_count = 0

        while (time.time() - start_time) < duration:
            elapsed = time.time() - start_time

            # Check perturbation events
            if with_perturbations:
                executor.check_events(elapsed)

            # Generate token
            t0 = time.time()
            with torch.no_grad():
                outputs = self.model(current_ids, use_cache=False)
                logits = outputs.logits[:, -1, :].float()
            t1 = time.time()
            token_time = t1 - t0

            # Get telemetry based on mode
            if mode == DemoMode.CROSS_SWAP:
                # Use replay buffer (different context)
                hw = self._replay_buffer[replay_idx % len(self._replay_buffer)]
                replay_idx += 1
            else:
                # Live telemetry
                hw = self.telemetry.get_token_aligned(t0, t1)

            temp = hw.get("temp", 50)
            power = hw.get("power", 0)
            util = hw.get("util", 0)
            vram = hw.get("vram_used", 0)

            # Encode z_feel based on mode
            if mode == DemoMode.DISABLED:
                # z_feel always zero (open loop)
                z_feel = torch.zeros(1, self.embed_dim, device=self.device)
            else:
                # Create sensor context from telemetry
                hw_ctx = HardwareContext(
                    temp=temp,
                    power=power,
                    util=util,
                    vram_used_pct=(vram / 1024 / 24.0) if vram > 100 else (vram / 24.0),  # ~24GB VRAM
                )
                runtime = RuntimeContext(
                    token_latency=token_time,
                    kv_cache_tokens=current_ids.shape[1],
                    generation_depth=token_count,
                )
                # Sensor bank → projector → z_feel
                sensors = self.sensor_bank(logits.detach(), runtime=runtime, hardware=hw_ctx)
                z_feel = self.projector(sensors.float())

            # Get policy action
            action = self.policy.step(logits, z_feel)

            # Execute K-sampling if K > 1
            if action.K > 1:
                next_token, _ = self.k_effector.sample_with_K(logits, K=action.K)
            else:
                probs = F.softmax(logits / 0.7, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

            # Compute output entropy
            probs = F.softmax(logits / 0.7, dim=-1)
            entropy = -torch.sum(probs * torch.log(probs + 1e-10)).item()

            # Record trace
            trace.timestamps.append(elapsed)
            trace.temps.append(temp)
            trace.powers.append(power)
            trace.utils.append(util)
            trace.z_feel_norms.append(z_feel.norm().item())
            trace.z_feel_projections.append(self._project_z_feel(z_feel))
            trace.K_values.append(action.K)
            trace.ponder_steps.append(action.ponder_steps)
            trace.dvfs_actions.append(action.dvfs.value)
            trace.entropies.append(entropy)
            trace.token_times.append(token_time)
            trace.difficulty_scores.append(action.difficulty_score)
            trace.stress_scores.append(action.stress_score)
            trace.net_efforts.append(action.net_effort)

            # Update context
            current_ids = torch.cat([current_ids, next_token], dim=-1)
            if current_ids.shape[1] > 512:
                current_ids = input_ids.clone()

            token_count += 1

            # Progress
            if token_count % 50 == 0:
                print(f"  [{elapsed:5.1f}s] T={temp:.1f}°C K={action.K} "
                      f"stress={action.stress_score:.2f} mode={mode.value}")

        if with_perturbations:
            executor.stop()

        print(f"\n  Completed: {token_count} tokens in {duration:.0f}s")
        return trace

    def run_all_modes(
        self,
        prompt: str,
        cross_swap_prompt: str,
        duration: float = 90.0,
    ) -> Dict[str, DemoTrace]:
        """
        Run all three modes for comparison.
        """
        results = {}

        # First collect cross-swap data
        self.collect_cross_swap_data(cross_swap_prompt, duration=60.0)

        # Wait for cooldown
        print("\n  Cooling down before trials...")
        time.sleep(30)

        # Run each mode
        for mode in [DemoMode.LIVE, DemoMode.CROSS_SWAP, DemoMode.DISABLED]:
            print(f"\n  Waiting for cooldown...")
            time.sleep(20)

            trace = self.run_demo(prompt, mode, duration=duration)
            results[mode.value] = trace

        return results

    def compute_comparison_metrics(self, results: Dict[str, DemoTrace]) -> Dict:
        """
        Compute comparison metrics across modes.
        """
        metrics = {}

        for mode_name, trace in results.items():
            temps = np.array(trace.temps)
            K_vals = np.array(trace.K_values)
            stress = np.array(trace.stress_scores)

            # Homeostasis metrics
            time_above_60 = np.mean(temps > 60)
            temp_std = np.std(temps)

            # Controller activity
            K_mean = np.mean(K_vals)
            K_2_rate = np.mean(K_vals == 2)

            # Correlations
            if len(temps) > 10:
                temp_stress_corr = np.corrcoef(temps, stress)[0, 1]
                stress_K_corr = np.corrcoef(stress, K_vals)[0, 1]
            else:
                temp_stress_corr = 0
                stress_K_corr = 0

            metrics[mode_name] = {
                "temp_mean": float(np.mean(temps)),
                "temp_std": float(temp_std),
                "time_above_60": float(time_above_60),
                "K_mean": float(K_mean),
                "K_2_rate": float(K_2_rate),
                "temp_stress_corr": float(temp_stress_corr) if not np.isnan(temp_stress_corr) else 0,
                "stress_K_corr": float(stress_K_corr) if not np.isnan(stress_K_corr) else 0,
            }

        # Compute relative improvements
        if "live" in metrics and "disabled" in metrics:
            live = metrics["live"]
            disabled = metrics["disabled"]

            metrics["live_vs_disabled"] = {
                "temp_std_reduction": (disabled["temp_std"] - live["temp_std"]) / max(disabled["temp_std"], 0.01),
                "time_above_reduction": (disabled["time_above_60"] - live["time_above_60"]),
                "K_activity_increase": live["K_2_rate"] - disabled["K_2_rate"],
            }

        if "live" in metrics and "cross_swap" in metrics:
            live = metrics["live"]
            cross = metrics["cross_swap"]

            # Cross-swap should break correlation
            metrics["falsification"] = {
                "live_temp_stress_corr": live["temp_stress_corr"],
                "cross_temp_stress_corr": cross["temp_stress_corr"],
                "correlation_collapse": live["temp_stress_corr"] - cross["temp_stress_corr"],
            }

        return metrics

    def stop(self):
        self.perturb.cleanup()
        self.telemetry.stop()


def print_demo_summary(metrics: Dict):
    """Print formatted summary of demo results."""
    print("\n" + "=" * 60)
    print("  THREE-TOGGLE DEMO RESULTS")
    print("=" * 60)

    for mode_name in ["live", "cross_swap", "disabled"]:
        if mode_name not in metrics:
            continue
        m = metrics[mode_name]
        print(f"\n  {mode_name.upper()}:")
        print(f"    Temperature: {m['temp_mean']:.1f}°C ± {m['temp_std']:.1f}°C")
        print(f"    Time above 60°C: {m['time_above_60']:.1%}")
        print(f"    K=2 rate: {m['K_2_rate']:.1%}")
        print(f"    temp↔stress corr: {m['temp_stress_corr']:.3f}")

    if "live_vs_disabled" in metrics:
        print(f"\n  LIVE vs DISABLED:")
        lvd = metrics["live_vs_disabled"]
        print(f"    Temp std reduction: {lvd['temp_std_reduction']:.1%}")
        print(f"    Time above 60 reduction: {lvd['time_above_reduction']:.1%}")
        print(f"    K activity increase: {lvd['K_activity_increase']:.1%}")

    if "falsification" in metrics:
        print(f"\n  FALSIFICATION (Cross-swap):")
        f = metrics["falsification"]
        print(f"    Live temp↔stress: {f['live_temp_stress_corr']:.3f}")
        print(f"    Cross temp↔stress: {f['cross_temp_stress_corr']:.3f}")
        print(f"    Correlation collapse: {f['correlation_collapse']:.3f}")

        if f["correlation_collapse"] > 0.3:
            print(f"    ✓ PASS: Cross-swap breaks causal link")
        else:
            print(f"    ✗ WEAK: Cross-swap correlation not sufficiently collapsed")


def main():
    parser = argparse.ArgumentParser(description="Three-Toggle FEEL Demo")
    parser.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--mode", choices=["live", "cross_swap", "disabled", "all"], default="all")
    parser.add_argument("--duration", type=float, default=90, help="Duration per mode in seconds")
    parser.add_argument("--output_dir", default="results/three_toggle_demo")
    args = parser.parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    demo = ThreeToggleDemo(model_name=args.model)

    prompt = "Explain the mathematical foundations of neural network backpropagation."
    cross_swap_prompt = "Write a story about a robot learning to feel emotions."

    if args.mode == "all":
        results = demo.run_all_modes(prompt, cross_swap_prompt, duration=args.duration)
        metrics = demo.compute_comparison_metrics(results)

        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"{args.output_dir}/three_toggle_{timestamp}.json"

        # Convert traces to dicts
        results_dict = {k: v.to_dict() for k, v in results.items()}

        with open(output_path, "w") as f:
            json.dump({
                "version": VERSION,
                "prompt": prompt,
                "cross_swap_prompt": cross_swap_prompt,
                "duration": args.duration,
                "results": results_dict,
                "metrics": metrics,
            }, f, indent=2)

        print(f"\nSaved: {output_path}")
        print_demo_summary(metrics)

    else:
        mode = DemoMode(args.mode)

        if mode == DemoMode.CROSS_SWAP:
            demo.collect_cross_swap_data(cross_swap_prompt, duration=60.0)
            time.sleep(20)

        trace = demo.run_demo(prompt, mode, duration=args.duration)

        # Save single trace
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"{args.output_dir}/demo_{mode.value}_{timestamp}.json"

        with open(output_path, "w") as f:
            json.dump({
                "version": VERSION,
                "mode": mode.value,
                "prompt": prompt,
                "duration": args.duration,
                "trace": trace.to_dict(),
            }, f, indent=2)

        print(f"\nSaved: {output_path}")

    demo.stop()


if __name__ == "__main__":
    main()
