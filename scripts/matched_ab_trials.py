#!/usr/bin/env python3 -u
"""
Matched A/B Trials for Track B: Homeostasis Regulation

The old evaluation was confounded: first-half vs second-half comparison
isn't valid because thermal state drifts over time.

Proper design:
1. Run N matched pairs of trials
2. Each pair: same prompt, same starting conditions
3. Condition A: Hysteresis controller ON
4. Condition B: No controller (DVFS=auto always)
5. Compare oscillation count, temp variance, recovery time

Statistical test:
- Paired t-test or Wilcoxon signed-rank on oscillation counts
- Controller should significantly reduce oscillations

Reset protocol between trials:
- Cool GPU to baseline (<50°C)
- Reset controller state
- Same random seed for sampling
"""

import sys
import time
import json
import argparse
import threading
from pathlib import Path
from datetime import datetime
from typing import List, Tuple, Dict
import numpy as np
from scipy import stats

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, str(Path(__file__).parent.parent))

from src import (
    TelemetrySampler,
    HysteresisController,
    ControllerConfig,
)

VERSION = "matched-ab-v1.0.0"


class GPUStressor:
    """Controllable GPU load."""

    def __init__(self):
        self._running = False
        self._thread = None

    def start(self):
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._run)
        self._thread.daemon = True
        self._thread.start()

    def _run(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        a = torch.randn(4096, 4096, device=device, dtype=torch.float16)
        b = torch.randn(4096, 4096, device=device, dtype=torch.float16)
        while self._running:
            _ = torch.matmul(a, b)
            torch.cuda.synchronize()
            time.sleep(0.01)

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)

    @property
    def is_running(self):
        return self._running


def wait_for_cooldown(telemetry: TelemetrySampler, target_temp: float = 50.0, timeout: float = 120.0):
    """Wait for GPU to cool below target temperature."""
    print(f"  Waiting for cooldown to {target_temp}°C...")
    start = time.time()
    while (time.time() - start) < timeout:
        snap = telemetry.snapshot()
        temp = snap.get("temp")
        if temp is not None and temp < target_temp:
            print(f"  Cooled to {temp:.1f}°C")
            return True
        time.sleep(2.0)
    print(f"  Timeout waiting for cooldown")
    return False


def count_oscillations(dvfs_modes: List[str]) -> int:
    """Count DVFS mode changes (oscillations)."""
    if len(dvfs_modes) < 2:
        return 0
    oscillations = 0
    for i in range(1, len(dvfs_modes)):
        if dvfs_modes[i] != dvfs_modes[i-1]:
            oscillations += 1
    return oscillations


def compute_temp_variance(temps: List[float]) -> float:
    """Compute temperature variance."""
    if len(temps) < 2:
        return 0.0
    return float(np.var(temps))


class MatchedABTrials:
    """
    Run matched A/B trials for homeostasis regulation.
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

        # Telemetry
        self.telemetry = TelemetrySampler(sample_hz=30)
        self.telemetry.start()
        print(f"  Telemetry: {self.telemetry.source}")

        # Controller
        self.controller = HysteresisController(ControllerConfig())

        # Stressor
        self.stressor = GPUStressor()

    def run_single_trial(
        self,
        prompt: str,
        duration: float,
        use_controller: bool,
        seed: int,
    ) -> Dict:
        """
        Run a single trial.

        Args:
            prompt: Generation prompt
            duration: Trial duration in seconds
            use_controller: Whether to use hysteresis controller
            seed: Random seed for reproducibility
        """
        torch.manual_seed(seed)

        self.controller.reset()

        traces = {
            "use_controller": use_controller,
            "seed": seed,
            "timestamps": [],
            "temps": [],
            "dvfs_modes": [],
            "controller_states": [],
            "stressor_active": [],
        }

        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        current_ids = input_ids.clone()

        start_time = time.time()
        token_count = 0

        while (time.time() - start_time) < duration:
            elapsed = time.time() - start_time

            # Perturbation: stress on from 1/4 to 3/4 duration
            if elapsed > duration / 4 and elapsed < 3 * duration / 4:
                if not self.stressor.is_running:
                    self.stressor.start()
            else:
                if self.stressor.is_running:
                    self.stressor.stop()

            # Forward pass
            t0 = time.time()
            with torch.no_grad():
                outputs = self.model(current_ids, use_cache=False)
                logits = outputs.logits[:, -1, :].float()
            t1 = time.time()

            # Get telemetry
            hw = self.telemetry.get_token_aligned(t0, t1)
            temp = hw.get("temp")
            power = hw.get("power")
            util = hw.get("util")

            # Controller step
            if use_controller and temp is not None:
                dvfs, _ = self.controller.step(temp, power, util, phase="decode")
            else:
                dvfs = "auto"

            # Record
            traces["timestamps"].append(elapsed)
            traces["temps"].append(temp or 0)
            traces["dvfs_modes"].append(dvfs)
            traces["controller_states"].append(self.controller.state.value if use_controller else "none")
            traces["stressor_active"].append(self.stressor.is_running)

            token_count += 1

            # Sample next token (deterministic with seed)
            probs = F.softmax(logits / 0.7, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            current_ids = torch.cat([current_ids, next_token], dim=-1)

            if current_ids.shape[1] > 512:
                current_ids = input_ids.clone()

        self.stressor.stop()

        # Compute metrics
        traces["tokens"] = token_count
        traces["oscillations"] = count_oscillations(traces["dvfs_modes"])
        traces["temp_variance"] = compute_temp_variance(traces["temps"])
        traces["temp_max"] = max(traces["temps"]) if traces["temps"] else 0
        traces["temp_mean"] = float(np.mean(traces["temps"])) if traces["temps"] else 0

        return traces

    def run_matched_trials(
        self,
        n_pairs: int = 5,
        trial_duration: float = 60.0,
        cooldown_target: float = 50.0,
    ) -> Dict:
        """
        Run N matched pairs of trials.

        Each pair:
        - Trial A: With hysteresis controller
        - Trial B: Without controller (baseline)
        - Same prompt, same seed, cooled between trials
        """
        print("\n" + "=" * 60)
        print("  MATCHED A/B TRIALS: Homeostasis Regulation")
        print(f"  {n_pairs} pairs × {trial_duration}s each")
        print("=" * 60)

        prompts = [
            "Explain quantum computing and its applications.",
            "Describe the process of photosynthesis in plants.",
            "What are the main causes and effects of climate change?",
            "Explain how neural networks learn through backpropagation.",
            "Describe the history and evolution of the internet.",
        ]

        pairs = []

        for i in range(n_pairs):
            print(f"\n--- Pair {i+1}/{n_pairs} ---")
            prompt = prompts[i % len(prompts)]
            seed = 42 + i  # Reproducible seeds

            # Trial A: WITH controller
            wait_for_cooldown(self.telemetry, target_temp=cooldown_target)
            print(f"  Running Trial A (controller ON)...")
            trial_a = self.run_single_trial(
                prompt=prompt,
                duration=trial_duration,
                use_controller=True,
                seed=seed,
            )

            # Cooldown between trials
            wait_for_cooldown(self.telemetry, target_temp=cooldown_target)

            # Trial B: WITHOUT controller
            print(f"  Running Trial B (controller OFF)...")
            trial_b = self.run_single_trial(
                prompt=prompt,
                duration=trial_duration,
                use_controller=False,
                seed=seed,
            )

            pair = {
                "pair_id": i,
                "prompt": prompt[:50] + "...",
                "seed": seed,
                "with_controller": trial_a,
                "without_controller": trial_b,
            }
            pairs.append(pair)

            # Summary for this pair
            print(f"    Controller ON:  oscillations={trial_a['oscillations']}, "
                  f"temp_var={trial_a['temp_variance']:.1f}, max={trial_a['temp_max']:.1f}°C")
            print(f"    Controller OFF: oscillations={trial_b['oscillations']}, "
                  f"temp_var={trial_b['temp_variance']:.1f}, max={trial_b['temp_max']:.1f}°C")

        # Statistical analysis
        stats_results = self._compute_statistics(pairs)

        return {
            "version": VERSION,
            "n_pairs": n_pairs,
            "trial_duration": trial_duration,
            "pairs": pairs,
            "statistics": stats_results,
        }

    def _compute_statistics(self, pairs: List[Dict]) -> Dict:
        """
        Compute statistical comparison between conditions.
        """
        # Extract metrics
        oscillations_with = [p["with_controller"]["oscillations"] for p in pairs]
        oscillations_without = [p["without_controller"]["oscillations"] for p in pairs]

        temp_var_with = [p["with_controller"]["temp_variance"] for p in pairs]
        temp_var_without = [p["without_controller"]["temp_variance"] for p in pairs]

        temp_max_with = [p["with_controller"]["temp_max"] for p in pairs]
        temp_max_without = [p["without_controller"]["temp_max"] for p in pairs]

        # Paired tests
        # Wilcoxon signed-rank (non-parametric, small samples)
        try:
            osc_stat, osc_pvalue = stats.wilcoxon(oscillations_without, oscillations_with, alternative='greater')
        except Exception:
            osc_stat, osc_pvalue = 0, 1.0

        try:
            var_stat, var_pvalue = stats.wilcoxon(temp_var_without, temp_var_with, alternative='greater')
        except Exception:
            var_stat, var_pvalue = 0, 1.0

        # Effect sizes (mean difference)
        osc_reduction = np.mean(oscillations_without) - np.mean(oscillations_with)
        var_reduction = np.mean(temp_var_without) - np.mean(temp_var_with)
        max_reduction = np.mean(temp_max_without) - np.mean(temp_max_with)

        return {
            "oscillations": {
                "with_controller_mean": float(np.mean(oscillations_with)),
                "without_controller_mean": float(np.mean(oscillations_without)),
                "reduction": float(osc_reduction),
                "wilcoxon_statistic": float(osc_stat),
                "p_value": float(osc_pvalue),
                "significant": osc_pvalue < 0.05,
            },
            "temp_variance": {
                "with_controller_mean": float(np.mean(temp_var_with)),
                "without_controller_mean": float(np.mean(temp_var_without)),
                "reduction": float(var_reduction),
                "wilcoxon_statistic": float(var_stat),
                "p_value": float(var_pvalue),
                "significant": var_pvalue < 0.05,
            },
            "temp_max": {
                "with_controller_mean": float(np.mean(temp_max_with)),
                "without_controller_mean": float(np.mean(temp_max_without)),
                "reduction": float(max_reduction),
            },
            "interpretation": (
                "PASS: Controller significantly reduces oscillations"
                if osc_pvalue < 0.05 and osc_reduction > 0
                else "INCONCLUSIVE: No significant difference"
            ),
        }

    def stop(self):
        self.stressor.stop()
        if self.telemetry:
            self.telemetry.stop()


def main():
    parser = argparse.ArgumentParser(description="Matched A/B Trials")
    parser.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--n_pairs", type=int, default=5)
    parser.add_argument("--trial_duration", type=float, default=60.0)
    parser.add_argument("--output_dir", default="results/matched_ab")
    args = parser.parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    trial = MatchedABTrials(model_name=args.model)

    results = trial.run_matched_trials(
        n_pairs=args.n_pairs,
        trial_duration=args.trial_duration,
    )

    # Save
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = f"{args.output_dir}/matched_ab_{timestamp}.json"

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

    with open(output_path, "w") as f:
        json.dump(convert(results), f, indent=2)

    print(f"\nSaved: {output_path}")

    # Print summary
    s = results["statistics"]
    print("\n" + "=" * 60)
    print("  MATCHED A/B TRIAL RESULTS")
    print("=" * 60)
    print(f"  Oscillations:")
    print(f"    With controller:    {s['oscillations']['with_controller_mean']:.1f}")
    print(f"    Without controller: {s['oscillations']['without_controller_mean']:.1f}")
    print(f"    Reduction:          {s['oscillations']['reduction']:.1f}")
    print(f"    p-value:            {s['oscillations']['p_value']:.4f}")
    print(f"  Temperature variance:")
    print(f"    With controller:    {s['temp_variance']['with_controller_mean']:.1f}")
    print(f"    Without controller: {s['temp_variance']['without_controller_mean']:.1f}")
    print(f"  Result: {s['interpretation']}")
    print("=" * 60)

    trial.stop()


if __name__ == "__main__":
    main()
