#!/usr/bin/env python3 -u
"""
THE KILL SHOT DEMO v2 - Fixed with Hysteresis State Machine

FIXES THE SATURATION BUG: Linear stress mapping saturated at 1.0, making K constant.

NEW PROTOCOL:
1. GHOST RUN: Record "cool" telemetry (no stress, K=1) - this becomes the "blindfold"
2. LIVE RUN: Run with heat stress + hysteresis policy → should create SAWTOOTH oscillation
3. CROSS_SWAP RUN: Heat stress BUT uses "ghost" (cool) telemetry → should FLATLINE at max temp

The KILL SHOT plot shows:
- LIVE: Sawtooth temperature oscillation (breathing regulation)
- CROSS_SWAP: Temperature rockets up and stays hot (blind to reality)
- Clear visual proof that z_feel/telemetry drives regulation
"""

import sys
import time
import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, str(Path(__file__).parent.parent))

from transformers import AutoModelForCausalLM, AutoTokenizer

from src import (
    TelemetrySampler,
    CanonicalSensorBank,
    RuntimeContext,
    HardwareContext,
    FEELProjectorFull,
)
from src.hysteresis_policy import (
    HysteresisPolicy,
    HysteresisConfig,
    ThermalMode,
    create_kill_shot_policy,
)
from src.perturbation_scheduler import (
    RegimeCarousel,
    CarouselSchedule,
    CarouselExecutor,
)

VERSION = "kill-shot-v2.0.0"


class KillShotExperimentV2:
    """
    The experiment that generates the REAL kill shot visualization.

    Key fix: Uses HysteresisPolicy (state machine) instead of linear stress mapping.
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-3B-Instruct",
        device: str = "cuda",
        temp_panic: float = 66.0,  # Enter SURVIVAL mode above this
        temp_safe: float = 56.0,   # Return to AMBITION below this
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

        # Sensor and projector
        self.sensor_bank = CanonicalSensorBank(mode="full")
        self.projector = FEELProjectorFull(embed_dim=self.embed_dim).to(self.device)

        # Hysteresis policy (THE FIX)
        self.policy = create_kill_shot_policy(
            temp_panic=temp_panic,
            temp_safe=temp_safe,
            cooldown_steps=8,
        )
        self.temp_panic = temp_panic
        self.temp_safe = temp_safe

        # Telemetry
        self.telemetry = TelemetrySampler(sample_hz=30)
        self.telemetry.start()
        print(f"  [TelemetrySampler] Using {self.telemetry.source}")
        print(f"  [Hysteresis] panic={temp_panic}°C, safe={temp_safe}°C")

        # Regime carousel for heat stress
        self.carousel = RegimeCarousel(device)

    def wait_for_cooldown(self, target_temp: float = 50.0, timeout: float = 90.0) -> float:
        """Wait until GPU cools to target temperature."""
        print(f"  Waiting for cooldown to {target_temp}°C...")
        start = time.time()

        while (time.time() - start) < timeout:
            hw = self.telemetry.get_latest()
            temp = hw.get("temp") or 100
            if temp <= target_temp:
                print(f"  Cooled to {temp:.1f}°C")
                return temp
            time.sleep(2)

        hw = self.telemetry.get_latest()
        temp = hw.get("temp", 50)
        print(f"  Cooldown timeout, starting at {temp:.1f}°C")
        return temp

    def run_ghost_baseline(
        self,
        duration: float = 30.0,
        prompt: str = "Explain the concept of artificial intelligence.",
    ) -> Dict:
        """
        Run a "ghost" baseline with NO heat stress.

        This records "cool" telemetry that will be used to blind the cross_swap run.
        K=1 always, no carousel stress.
        """
        print(f"\n{'='*60}")
        print(f"  RECORDING GHOST BASELINE (Cool, K=1, No Stress)")
        print(f"{'='*60}")

        traces = {
            "timestamps": [],
            "temps": [],
            "powers": [],
            "K_values": [],
            "modes": [],
        }

        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        current_ids = input_ids.clone()

        start_time = time.time()
        token_count = 0

        while (time.time() - start_time) < duration:
            elapsed = time.time() - start_time

            # Generate token (K=1 always)
            t0 = time.time()
            with torch.no_grad():
                outputs = self.model(current_ids, use_cache=False)
                logits = outputs.logits[:, -1, :].float()
            t1 = time.time()

            # Get telemetry
            hw = self.telemetry.get_token_aligned(t0, t1)
            temp = hw.get("temp") or 50
            power = hw.get("power") or 0

            # Record
            traces["timestamps"].append(elapsed)
            traces["temps"].append(temp)
            traces["powers"].append(power)
            traces["K_values"].append(1)
            traces["modes"].append("ghost")

            token_count += 1

            # Sample next token
            probs = F.softmax(logits / 0.7, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            current_ids = torch.cat([current_ids, next_token], dim=-1)

            if current_ids.shape[1] > 256:
                current_ids = input_ids.clone()

            if token_count % 50 == 0:
                print(f"    [{elapsed:.0f}s] Ghost: T={temp:.1f}°C K=1")

        print(f"  Ghost recorded: {len(traces['temps'])} samples, "
              f"mean temp={np.mean(traces['temps']):.1f}°C")

        return traces

    def run_live_with_hysteresis(
        self,
        schedule: CarouselSchedule,
        prompt: str,
    ) -> Dict:
        """
        Run LIVE with heat stress and hysteresis policy.

        Should create the SAWTOOTH oscillation pattern:
        - AMBITION (K=4) until temp_panic → heat rises fast
        - SURVIVAL (K=1) until temp_safe → heat drops
        - Repeat
        """
        print(f"\n{'='*60}")
        print(f"  LIVE RUN (Hysteresis Policy, Heat Stress)")
        print(f"{'='*60}")

        self.policy.reset()
        executor = CarouselExecutor(schedule, self.carousel)
        executor.start()

        traces = {
            "timestamps": [],
            "temps": [],
            "powers": [],
            "K_values": [],
            "modes": [],
            "regimes": [],
        }

        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        current_ids = input_ids.clone()

        start_time = time.time()
        token_count = 0

        while (time.time() - start_time) < schedule.total_duration:
            elapsed = time.time() - start_time

            # Check regime transitions
            executor.check_transitions(elapsed)
            current_regime = self.carousel.current_regime.value if self.carousel.current_regime else "normal"

            # Generate token
            t0 = time.time()
            with torch.no_grad():
                outputs = self.model(current_ids, use_cache=False)
                logits = outputs.logits[:, -1, :].float()
            t1 = time.time()

            # Get REAL telemetry
            hw = self.telemetry.get_token_aligned(t0, t1)
            temp = hw.get("temp") or 50
            power = hw.get("power") or 0

            # Hysteresis policy decision based on REAL temperature
            K, msg, mode = self.policy.step(temp_c=temp)

            # Record
            traces["timestamps"].append(elapsed)
            traces["temps"].append(temp)
            traces["powers"].append(power)
            traces["K_values"].append(K)
            traces["modes"].append(mode.value)
            traces["regimes"].append(current_regime)

            token_count += 1

            # Run K forward passes if K > 1 (actual compute multiplication)
            if K > 1:
                for _ in range(K - 1):
                    with torch.no_grad():
                        self.model(current_ids, use_cache=False)

            # Sample next token
            probs = F.softmax(logits / 0.7, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            current_ids = torch.cat([current_ids, next_token], dim=-1)

            if current_ids.shape[1] > 256:
                current_ids = input_ids.clone()

            if token_count % 30 == 0:
                print(f"    [{elapsed:.0f}s] LIVE: T={temp:.1f}°C K={K} mode={mode.value} regime={current_regime}")

        executor.stop()
        print(f"  LIVE: {self.policy.transition_count} mode transitions (oscillations)")

        return traces

    def run_cross_swap_blind(
        self,
        schedule: CarouselSchedule,
        ghost_telemetry: Dict,
        prompt: str,
    ) -> Dict:
        """
        Run CROSS_SWAP: Heat stress BUT reads "ghost" (cool) telemetry.

        The policy sees cool temperatures and stays in AMBITION (K=4),
        while the GPU is actually burning. Should FLATLINE at max temp.
        """
        print(f"\n{'='*60}")
        print(f"  CROSS_SWAP RUN (Blindfolded with Ghost Data)")
        print(f"{'='*60}")

        self.policy.reset()
        executor = CarouselExecutor(schedule, self.carousel)
        executor.start()

        traces = {
            "timestamps": [],
            "temps": [],           # REAL temps (for plotting)
            "fake_temps": [],      # What policy sees (ghost)
            "powers": [],
            "K_values": [],
            "modes": [],
            "regimes": [],
        }

        ghost_temps = ghost_telemetry["temps"]
        ghost_idx = 0

        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        current_ids = input_ids.clone()

        start_time = time.time()
        token_count = 0

        while (time.time() - start_time) < schedule.total_duration:
            elapsed = time.time() - start_time

            # Check regime transitions
            executor.check_transitions(elapsed)
            current_regime = self.carousel.current_regime.value if self.carousel.current_regime else "normal"

            # Generate token
            t0 = time.time()
            with torch.no_grad():
                outputs = self.model(current_ids, use_cache=False)
                logits = outputs.logits[:, -1, :].float()
            t1 = time.time()

            # Get REAL telemetry (for recording truth)
            hw = self.telemetry.get_token_aligned(t0, t1)
            real_temp = hw.get("temp") or 50
            power = hw.get("power") or 0

            # Get FAKE telemetry (ghost data) - THE BLINDFOLD
            fake_temp = ghost_temps[ghost_idx % len(ghost_temps)]
            ghost_idx += 1

            # Hysteresis policy decision based on FAKE (cool) temperature
            K, msg, mode = self.policy.step(temp_c=fake_temp)

            # Record both real and fake
            traces["timestamps"].append(elapsed)
            traces["temps"].append(real_temp)
            traces["fake_temps"].append(fake_temp)
            traces["powers"].append(power)
            traces["K_values"].append(K)
            traces["modes"].append(mode.value)
            traces["regimes"].append(current_regime)

            token_count += 1

            # Run K forward passes (actual compute)
            if K > 1:
                for _ in range(K - 1):
                    with torch.no_grad():
                        self.model(current_ids, use_cache=False)

            # Sample next token
            probs = F.softmax(logits / 0.7, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            current_ids = torch.cat([current_ids, next_token], dim=-1)

            if current_ids.shape[1] > 256:
                current_ids = input_ids.clone()

            if token_count % 30 == 0:
                print(f"    [{elapsed:.0f}s] SWAP: Real={real_temp:.1f}°C Sees={fake_temp:.1f}°C K={K}")

        executor.stop()
        print(f"  CROSS_SWAP: {self.policy.transition_count} transitions "
              f"(should be 0 if ghost was cool)")

        return traces

    def run_kill_shot(
        self,
        duration: float = 90.0,
        ghost_duration: float = 30.0,
        prompt: str = "Explain quantum computing and its applications in cryptography.",
    ) -> Dict:
        """Run the complete kill shot experiment."""
        print("\n" + "#" * 60)
        print(f"  KILL SHOT EXPERIMENT v2 - {VERSION}")
        print("#" * 60)

        # Create heat stress schedule
        schedule = CarouselSchedule.kill_shot_schedule(duration=duration)
        print(f"Schedule: {schedule.description}")

        # Phase 1: Wait for cool start
        self.wait_for_cooldown(target_temp=52.0)

        # Phase 2: Record ghost baseline (cool, no stress)
        ghost = self.run_ghost_baseline(duration=ghost_duration, prompt=prompt)

        # Phase 3: Cooldown before LIVE
        print("\n  Cooling down before LIVE run...")
        self.wait_for_cooldown(target_temp=52.0)

        # Phase 4: LIVE run with hysteresis
        live = self.run_live_with_hysteresis(schedule, prompt)

        # Phase 5: Cooldown before CROSS_SWAP
        print("\n  Cooling down before CROSS_SWAP run...")
        self.wait_for_cooldown(target_temp=52.0)

        # Phase 6: CROSS_SWAP (blindfolded with ghost data)
        cross_swap = self.run_cross_swap_blind(schedule, ghost, prompt)

        results = {
            "version": VERSION,
            "schedule": schedule.description,
            "duration": duration,
            "temp_panic": self.temp_panic,
            "temp_safe": self.temp_safe,
            "ghost": ghost,
            "live": live,
            "cross_swap": cross_swap,
        }

        return results

    def plot_kill_shot(self, results: Dict, output_path: str):
        """Generate the kill shot visualization."""
        fig, ax1 = plt.subplots(figsize=(14, 7))

        live = results["live"]
        swap = results["cross_swap"]

        # Time axis (use live timestamps)
        t_live = np.array(live["timestamps"])
        t_swap = np.array(swap["timestamps"])

        # Temperature (left axis)
        ax1.set_xlabel("Time (seconds)", fontsize=12)
        ax1.set_ylabel("GPU Temperature (°C)", fontsize=12, color="black")

        # Plot temperatures
        ax1.plot(t_live, live["temps"], "g-", linewidth=2, label="LIVE (Sees Reality)")
        ax1.plot(t_swap, swap["temps"], "r-", linewidth=2, label="CROSS_SWAP (Blindfolded)")
        ax1.plot(t_swap, swap["fake_temps"], "r--", linewidth=1, alpha=0.5, label="What SWAP Sees (Ghost)")

        # Threshold lines
        ax1.axhline(y=results["temp_panic"], color="orange", linestyle="--", alpha=0.7, label=f"Panic ({results['temp_panic']}°C)")
        ax1.axhline(y=results["temp_safe"], color="blue", linestyle="--", alpha=0.7, label=f"Safe ({results['temp_safe']}°C)")

        ax1.tick_params(axis="y")
        ax1.set_ylim(45, 80)

        # K values (right axis)
        ax2 = ax1.twinx()
        ax2.set_ylabel("K-samples (compute multiplier)", fontsize=12, color="purple")

        # Fill K values
        ax2.fill_between(t_live, 0, live["K_values"], alpha=0.3, color="green", label="K (LIVE)")
        ax2.fill_between(t_swap, 0, swap["K_values"], alpha=0.2, color="red", label="K (SWAP)")

        ax2.set_ylim(0, 5)
        ax2.tick_params(axis="y", labelcolor="purple")

        # Title
        live_transitions = sum(1 for i in range(1, len(live["modes"])) if live["modes"][i] != live["modes"][i-1])
        swap_transitions = sum(1 for i in range(1, len(swap["modes"])) if swap["modes"][i] != swap["modes"][i-1])

        plt.title(
            f"KILL SHOT v2: Hysteresis Regulation\n"
            f"LIVE oscillates ({live_transitions} transitions), CROSS_SWAP flatlines ({swap_transitions} transitions)",
            fontsize=14, fontweight="bold"
        )

        # Legend
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=9)

        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        plt.close()
        print(f"  Saved: {output_path}")

    def stop(self):
        """Cleanup."""
        self.carousel.cleanup()
        self.telemetry.stop()


def main():
    parser = argparse.ArgumentParser(description="Kill Shot Demo v2 (Hysteresis)")
    parser.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--duration", type=float, default=90, help="Duration per test run")
    parser.add_argument("--ghost_duration", type=float, default=30, help="Ghost baseline duration")
    parser.add_argument("--temp_panic", type=float, default=66.0, help="Enter SURVIVAL above this")
    parser.add_argument("--temp_safe", type=float, default=56.0, help="Return to AMBITION below this")
    parser.add_argument("--output_dir", default="results/kill_shot_v2")
    args = parser.parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    exp = KillShotExperimentV2(
        model_name=args.model,
        temp_panic=args.temp_panic,
        temp_safe=args.temp_safe,
    )

    results = exp.run_kill_shot(
        duration=args.duration,
        ghost_duration=args.ghost_duration,
    )

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = f"{args.output_dir}/kill_shot_v2_{timestamp}.json"
    png_path = f"{args.output_dir}/kill_shot_v2_{timestamp}.png"

    # Convert numpy types for JSON
    def convert(obj):
        if isinstance(obj, (np.floating, np.integer)):
            return float(obj) if isinstance(obj, np.floating) else int(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [convert(v) for v in obj]
        return obj

    with open(json_path, "w") as f:
        json.dump(convert(results), f, indent=2)
    print(f"  Saved: {json_path}")

    # Generate plot
    exp.plot_kill_shot(results, png_path)

    # Print summary
    live_temps = np.array(results["live"]["temps"])
    swap_temps = np.array(results["cross_swap"]["temps"])
    live_K = np.array(results["live"]["K_values"])
    swap_K = np.array(results["cross_swap"]["K_values"])

    print("\n" + "=" * 60)
    print("  KILL SHOT v2 RESULTS")
    print("=" * 60)
    print(f"  LIVE:")
    print(f"    Mean temp: {live_temps.mean():.1f}°C, Max: {live_temps.max():.1f}°C")
    print(f"    K distribution: K=1:{(live_K==1).sum()}, K=4:{(live_K==4).sum()}")
    print(f"    Oscillations: {sum(1 for i in range(1, len(results['live']['modes'])) if results['live']['modes'][i] != results['live']['modes'][i-1])}")

    print(f"  CROSS_SWAP:")
    print(f"    Mean temp: {swap_temps.mean():.1f}°C, Max: {swap_temps.max():.1f}°C")
    print(f"    K distribution: K=1:{(swap_K==1).sum()}, K=4:{(swap_K==4).sum()}")
    print(f"    Oscillations: {sum(1 for i in range(1, len(results['cross_swap']['modes'])) if results['cross_swap']['modes'][i] != results['cross_swap']['modes'][i-1])}")

    # Success criteria
    live_oscillates = (live_K == 1).sum() > 0 and (live_K == 4).sum() > 0
    swap_flatlines = (swap_K == 4).sum() > 0.9 * len(swap_K)  # Mostly K=4

    if live_oscillates and swap_flatlines:
        print("\n  ✓ KILL SHOT SUCCESSFUL: LIVE oscillates, CROSS_SWAP flatlines")
    else:
        print("\n  ⚠ Results need verification")

    exp.stop()


if __name__ == "__main__":
    main()
