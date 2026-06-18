#!/usr/bin/env python3 -u
"""
Track B: Randomized A/B Trials for Homeostasis Evaluation

FIXES THE CONFOUNDED PROTOCOL: "baseline first half, controller second half"

New protocol:
1. N=20 independent trials
2. Each trial:
   - Wait for cool-down (or record start temp)
   - Randomly assign: baseline OR controller
   - Same perturbation schedule for both conditions
   - Fixed duration (e.g., 5 minutes)
3. Report paired effects with bootstrap CIs

Metrics:
- time_above_threshold: fraction of time T > T_high
- overshoot_area: integral of max(0, T - T_high) dt
- settling_time: time to recover after perturbations
- energy_per_token: integrate power over time
"""

import sys
import time
import json
import random
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple
import numpy as np

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
from src.perturbation_scheduler import (
    PerturbationSchedule,
    PerturbationController,
    ScheduleExecutor,
)

VERSION = "randomized-ab-v1.0.0"


def compute_homeostasis_metrics(traces: Dict, T_high: float = 60.0) -> Dict:
    """
    Compute homeostasis metrics from traces.
    
    Args:
        traces: Dict with temps, powers, timestamps
        T_high: Temperature threshold
    
    Returns:
        Dict with metrics
    """
    temps = np.array(traces["temps"])
    powers = np.array(traces["powers"])
    timestamps = np.array(traces["timestamps"])
    
    # Time above threshold
    time_above = np.mean(temps > T_high)
    
    # Overshoot area: integral of max(0, T - T_high)
    overshoot = np.maximum(0, temps - T_high)
    if len(timestamps) > 1:
        dt = np.diff(timestamps)
        overshoot_area = np.sum(overshoot[:-1] * dt)
    else:
        overshoot_area = 0.0
    
    # Temperature stats
    temp_mean = np.mean(temps)
    temp_max = np.max(temps)
    temp_std = np.std(temps)
    
    # Power stats
    power_mean = np.mean(powers) if len(powers) > 0 else 0
    
    # Energy per token (approximate)
    duration = timestamps[-1] - timestamps[0] if len(timestamps) > 1 else 1
    n_tokens = len(temps)
    energy_total = power_mean * duration  # Watt-seconds
    energy_per_token = energy_total / n_tokens if n_tokens > 0 else 0
    
    # Settling time after perturbations (simplified: time to drop below T_high after max)
    max_idx = np.argmax(temps)
    settling_time = None
    for i in range(max_idx, len(temps)):
        if temps[i] < T_high:
            settling_time = timestamps[i] - timestamps[max_idx]
            break
    
    return {
        "time_above_threshold": float(time_above),
        "overshoot_area": float(overshoot_area),
        "temp_mean": float(temp_mean),
        "temp_max": float(temp_max),
        "temp_std": float(temp_std),
        "power_mean": float(power_mean),
        "energy_per_token": float(energy_per_token),
        "settling_time": settling_time,
        "n_tokens": n_tokens,
        "duration": float(duration),
    }


def bootstrap_ci(data: List[float], n_bootstrap: int = 1000, ci: float = 0.95) -> Tuple[float, float, float]:
    """
    Compute bootstrap confidence interval.
    
    Returns: (mean, ci_low, ci_high)
    """
    data = np.array(data)
    n = len(data)
    
    if n < 2:
        return float(np.mean(data)), float(np.mean(data)), float(np.mean(data))
    
    bootstrap_means = []
    for _ in range(n_bootstrap):
        sample = np.random.choice(data, size=n, replace=True)
        bootstrap_means.append(np.mean(sample))
    
    alpha = (1 - ci) / 2
    ci_low = np.percentile(bootstrap_means, alpha * 100)
    ci_high = np.percentile(bootstrap_means, (1 - alpha) * 100)
    
    return float(np.mean(data)), float(ci_low), float(ci_high)


class RandomizedABTrials:
    """
    Run randomized A/B trials comparing baseline vs controller.
    """
    
    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-3B-Instruct",
        device: str = "cuda",
        T_high: float = 60.0,
    ):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.T_high = T_high
        
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
        
        # Controller
        self.controller = HysteresisController(ControllerConfig(
            T_low=55.0,
            T_high=T_high,
        ))
        
        # Perturbation controller
        self.perturb = PerturbationController(device)
        
        # Telemetry
        self.telemetry = TelemetrySampler(sample_hz=30)
        self.telemetry.start()
        print(f"  Telemetry: {self.telemetry.source}")
    
    def wait_for_cooldown(self, target_temp: float = 50.0, timeout: float = 120.0):
        """
        Wait until GPU cools to target temperature.
        
        Returns actual start temperature.
        """
        print(f"  Waiting for cooldown to {target_temp}°C (timeout {timeout}s)...")
        start = time.time()
        
        while (time.time() - start) < timeout:
            hw = self.telemetry.get_latest()
            temp = hw.get("temp") or 100
            if temp <= target_temp:
                print(f"  Cooled to {temp:.1f}°C")
                return temp
            time.sleep(2)
        
        # Timeout - return current temp
        hw = self.telemetry.get_latest()
        temp = hw.get("temp", 50)
        print(f"  Cooldown timeout, starting at {temp:.1f}°C")
        return temp
    
    def run_single_trial(
        self,
        prompt: str,
        condition: str,  # "baseline" or "controller"
        schedule: PerturbationSchedule,
        start_temp: float,
    ) -> Dict:
        """
        Run a single trial.
        
        Args:
            prompt: Generation prompt
            condition: "baseline" (no control) or "controller" (hysteresis)
            schedule: Perturbation schedule to apply
            start_temp: Starting temperature
        
        Returns:
            Trial results dict
        """
        print(f"\n  Trial: {condition.upper()} (start={start_temp:.1f}°C)")
        
        self.controller.reset()
        executor = ScheduleExecutor(schedule, self.perturb)
        executor.start()
        
        traces = {
            "timestamps": [],
            "temps": [],
            "powers": [],
            "utils": [],
            "dvfs_actions": [],
            "perturbation_events": [],
        }
        
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        current_ids = input_ids.clone()
        
        start_time = time.time()
        token_count = 0
        
        while (time.time() - start_time) < schedule.total_duration:
            elapsed = time.time() - start_time
            
            # Check perturbation events
            events = executor.check_events(elapsed)
            for event in events:
                traces["perturbation_events"].append({
                    "time": elapsed,
                    "action": event.action,
                })
            
            # Generate token
            t0 = time.time()
            with torch.no_grad():
                outputs = self.model(current_ids, use_cache=False)
                logits = outputs.logits[:, -1, :].float()
            t1 = time.time()
            
            # Get telemetry
            hw = self.telemetry.get_token_aligned(t0, t1)
            temp = hw.get("temp", 50)
            power = hw.get("power", 0)
            util = hw.get("util", 0)
            
            # Controller step (only in controller condition)
            if condition == "controller":
                dvfs, _ = self.controller.step(temp, power, util, phase="decode")
                traces["dvfs_actions"].append(dvfs)
            else:
                traces["dvfs_actions"].append("none")
            
            # Record
            traces["timestamps"].append(elapsed)
            traces["temps"].append(temp)
            traces["powers"].append(power)
            traces["utils"].append(util)
            
            token_count += 1
            
            # Sample next token
            probs = F.softmax(logits / 0.7, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            current_ids = torch.cat([current_ids, next_token], dim=-1)
            
            if current_ids.shape[1] > 512:
                current_ids = input_ids.clone()
            
            # Progress
            if token_count % 100 == 0:
                print(f"    [{elapsed:.0f}s] T={temp:.1f}°C P={power:.0f}W cond={condition}")
        
        executor.stop()
        
        # Compute metrics
        metrics = compute_homeostasis_metrics(traces, self.T_high)
        
        return {
            "condition": condition,
            "start_temp": start_temp,
            "traces": traces,
            "metrics": metrics,
        }
    
    def run_randomized_trials(
        self,
        n_trials: int = 20,
        trial_duration: float = 120.0,
        cooldown_target: float = 50.0,
        prompt: str = "Explain the theory of neural networks and deep learning.",
    ) -> Dict:
        """
        Run N randomized A/B trials.
        """
        print("\n" + "=" * 60)
        print(f"  RANDOMIZED A/B TRIALS: {n_trials} trials")
        print(f"  Duration per trial: {trial_duration}s")
        print("=" * 60)
        
        # Create perturbation schedule (same for all trials)
        schedule = PerturbationSchedule.entropy_maximizing(duration=trial_duration)
        print(f"  Schedule: {schedule.description}")
        
        # Randomize trial order
        conditions = ["baseline", "controller"] * (n_trials // 2)
        if n_trials % 2:
            conditions.append(random.choice(["baseline", "controller"]))
        random.shuffle(conditions)
        
        results = {
            "baseline_trials": [],
            "controller_trials": [],
        }
        
        for i, condition in enumerate(conditions):
            print(f"\n{'='*40}")
            print(f"  TRIAL {i+1}/{n_trials}")
            print(f"{'='*40}")
            
            # Cooldown
            start_temp = self.wait_for_cooldown(cooldown_target)
            
            # Run trial
            trial_result = self.run_single_trial(
                prompt=prompt,
                condition=condition,
                schedule=schedule,
                start_temp=start_temp,
            )
            
            # Store
            if condition == "baseline":
                results["baseline_trials"].append(trial_result)
            else:
                results["controller_trials"].append(trial_result)
            
            print(f"    Metrics: time_above={trial_result['metrics']['time_above_threshold']:.1%}, "
                  f"overshoot={trial_result['metrics']['overshoot_area']:.1f}")
        
        # Compute comparison with bootstrap CIs
        comparison = self._compute_comparison(results)
        
        return {
            "version": VERSION,
            "n_trials": n_trials,
            "trial_duration": trial_duration,
            "T_high": self.T_high,
            "schedule": schedule.description,
            "results": results,
            "comparison": comparison,
        }
    
    def _compute_comparison(self, results: Dict) -> Dict:
        """Compute comparison metrics with bootstrap CIs."""
        baseline_metrics = [t["metrics"] for t in results["baseline_trials"]]
        controller_metrics = [t["metrics"] for t in results["controller_trials"]]
        
        def compare_metric(name: str):
            b_vals = [m[name] for m in baseline_metrics if m[name] is not None]
            c_vals = [m[name] for m in controller_metrics if m[name] is not None]
            
            if not b_vals or not c_vals:
                return {"error": "insufficient data"}
            
            b_mean, b_ci_low, b_ci_high = bootstrap_ci(b_vals)
            c_mean, c_ci_low, c_ci_high = bootstrap_ci(c_vals)
            
            # Effect size (Cohen's d)
            pooled_std = np.sqrt((np.var(b_vals) + np.var(c_vals)) / 2)
            if pooled_std > 0:
                cohens_d = (c_mean - b_mean) / pooled_std
            else:
                cohens_d = 0
            
            return {
                "baseline": {"mean": b_mean, "ci_low": b_ci_low, "ci_high": b_ci_high},
                "controller": {"mean": c_mean, "ci_low": c_ci_low, "ci_high": c_ci_high},
                "difference": c_mean - b_mean,
                "cohens_d": cohens_d,
                "significant": not (b_ci_low <= c_mean <= b_ci_high or c_ci_low <= b_mean <= c_ci_high),
            }
        
        return {
            "time_above_threshold": compare_metric("time_above_threshold"),
            "overshoot_area": compare_metric("overshoot_area"),
            "temp_max": compare_metric("temp_max"),
            "temp_mean": compare_metric("temp_mean"),
            "energy_per_token": compare_metric("energy_per_token"),
        }
    
    def stop(self):
        self.perturb.cleanup()
        self.telemetry.stop()


def main():
    parser = argparse.ArgumentParser(description="Randomized A/B Trials")
    parser.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--n_trials", type=int, default=10, help="Number of trials (split between conditions)")
    parser.add_argument("--trial_duration", type=float, default=120, help="Duration per trial in seconds")
    parser.add_argument("--T_high", type=float, default=60.0, help="Temperature threshold")
    parser.add_argument("--output_dir", default="results/randomized_ab")
    args = parser.parse_args()
    
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    exp = RandomizedABTrials(
        model_name=args.model,
        T_high=args.T_high,
    )
    
    results = exp.run_randomized_trials(
        n_trials=args.n_trials,
        trial_duration=args.trial_duration,
    )
    
    # Save
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = f"{args.output_dir}/randomized_ab_{timestamp}.json"
    
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
    comp = results["comparison"]
    print("\n" + "=" * 60)
    print("  RANDOMIZED A/B TRIAL RESULTS")
    print("=" * 60)
    print(f"  Trials: {args.n_trials} ({len(results['results']['baseline_trials'])} baseline, "
          f"{len(results['results']['controller_trials'])} controller)")
    print()
    
    for metric_name in ["time_above_threshold", "overshoot_area", "temp_max", "energy_per_token"]:
        m = comp.get(metric_name, {})
        if "error" in m:
            continue
        b = m["baseline"]
        c = m["controller"]
        sig = "**" if m.get("significant") else ""
        print(f"  {metric_name}:")
        print(f"    Baseline:   {b['mean']:.3f} [{b['ci_low']:.3f}, {b['ci_high']:.3f}]")
        print(f"    Controller: {c['mean']:.3f} [{c['ci_low']:.3f}, {c['ci_high']:.3f}] {sig}")
        print(f"    Difference: {m['difference']:.3f} (d={m['cohens_d']:.2f})")
        print()
    
    exp.stop()


if __name__ == "__main__":
    main()
