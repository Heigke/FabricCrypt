#!/usr/bin/env python3 -u
"""
Benefit-Collapse Falsification Test

The gold standard for proving embodiment:
If the loop is real, then breaking the information flow should
COLLAPSE the benefit (homeostasis improvement should disappear).

Three falsification modes:
1. LIVE: Real telemetry → real z_feel → real policy
2. CROSS-SWAP: Telemetry from different run (breaks sensing)
3. Z_FEEL_SHUFFLE: Real telemetry, shuffled z_feel (breaks mediation)

Pass criteria:
- LIVE shows homeostasis benefit (lower time_above, lower overshoot)
- CROSS-SWAP and Z_FEEL_SHUFFLE show benefit COLLAPSE (no improvement)

This proves:
- Link 1: Actions → Hardware (already demonstrated)
- Link 2: Hardware → z_feel (collapses under cross-swap)
- Link 3: z_feel → Actions (collapses under shuffle)
"""

import sys
import time
import json
import random
import argparse
from pathlib import Path
from datetime import datetime
from collections import deque
from typing import Dict, List, Tuple
import numpy as np

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, str(Path(__file__).parent.parent))

from src import (
    FEELProjectorFull,
    TelemetrySampler,
    TelemetrySamplerWrapper,
    CanonicalSensorBank,
    RuntimeContext,
    HardwareContext,
    HysteresisController,
    ControllerConfig,
)
from src.dual_axis_policy import DualAxisPolicyRuleBased, DualAxisConfig
from src.perturbation_scheduler import (
    PerturbationSchedule,
    PerturbationController,
    ScheduleExecutor,
)

VERSION = "benefit-collapse-v1.0.0"


def compute_homeostasis_metrics(traces: Dict, T_high: float = 60.0) -> Dict:
    """Compute homeostasis metrics."""
    temps = np.array(traces["temps"])
    powers = np.array(traces["powers"])
    timestamps = np.array(traces["timestamps"])
    
    time_above = np.mean(temps > T_high)
    overshoot = np.maximum(0, temps - T_high)
    if len(timestamps) > 1:
        dt = np.diff(timestamps)
        overshoot_area = np.sum(overshoot[:-1] * dt)
    else:
        overshoot_area = 0.0
    
    return {
        "time_above_threshold": float(time_above),
        "overshoot_area": float(overshoot_area),
        "temp_max": float(np.max(temps)),
        "temp_mean": float(np.mean(temps)),
        "power_mean": float(np.mean(powers)),
    }


class BenefitCollapseTest:
    """
    Test whether homeostasis benefit collapses under falsification.
    """
    
    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-3B-Instruct",
        checkpoint_path: str = None,
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
        
        self.embed_dim = self.model.config.hidden_size
        
        # Projector
        self.projector = FEELProjectorFull(embed_dim=self.embed_dim).to(self.device)
        if checkpoint_path and Path(checkpoint_path).exists():
            ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
            if 'projector_state_dict' in ckpt:
                self.projector.load_state_dict(ckpt['projector_state_dict'])
            print(f"  Loaded projector")
        
        # Sensor bank
        self.sensor_bank = CanonicalSensorBank(mode="full")
        
        # Dual-axis policy (rule-based for now)
        self.policy = DualAxisPolicyRuleBased()
        
        # Hysteresis controller
        self.controller = HysteresisController(ControllerConfig(T_high=T_high))
        
        # Perturbation controller
        self.perturb = PerturbationController(device)
        
        # Telemetry
        self.telemetry = TelemetrySampler(sample_hz=30)
        self.telemetry.start()
        self.wrapper = TelemetrySamplerWrapper(self.telemetry, mode="live")
        print(f"  Telemetry: {self.telemetry.source}")
        
        # Z_feel buffer for shuffle mode
        self._z_feel_buffer = deque(maxlen=100)
    
    def run_trial(
        self,
        prompt: str,
        mode: str,  # "live", "cross_swap", "z_feel_shuffle", "baseline"
        schedule: PerturbationSchedule,
        cross_telemetry: Dict = None,  # For cross_swap mode
        shuffle_lag: int = 32,
    ) -> Dict:
        """
        Run a single trial.
        
        Args:
            prompt: Generation prompt
            mode: Falsification mode
            schedule: Perturbation schedule
            cross_telemetry: Pre-recorded telemetry for cross_swap mode
            shuffle_lag: Lag for z_feel shuffle
        """
        print(f"\n  Trial: {mode.upper()}")
        
        self.policy.reset()
        self.controller.reset()
        self._z_feel_buffer.clear()
        
        executor = ScheduleExecutor(schedule, self.perturb)
        executor.start()
        
        traces = {
            "timestamps": [],
            "temps": [],
            "powers": [],
            "utils": [],
            "difficulty_scores": [],
            "stress_scores": [],
            "K_values": [],
            "dvfs_actions": [],
            "z_feel_norms": [],
        }
        
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        current_ids = input_ids.clone()
        
        # Cross-swap index
        cross_idx = 0
        
        start_time = time.time()
        token_count = 0
        
        while (time.time() - start_time) < schedule.total_duration:
            elapsed = time.time() - start_time
            
            # Check perturbation events
            executor.check_events(elapsed)
            
            # Generate token
            t0 = time.time()
            with torch.no_grad():
                outputs = self.model(current_ids, use_cache=False)
                logits = outputs.logits[:, -1, :].float()
            t1 = time.time()
            
            # Get telemetry (depends on mode)
            if mode == "cross_swap" and cross_telemetry:
                # Use telemetry from different run
                idx = cross_idx % len(cross_telemetry["temps"])
                hw = {
                    "temp": cross_telemetry["temps"][idx],
                    "power": cross_telemetry["powers"][idx],
                    "util": cross_telemetry["utils"][idx],
                }
                cross_idx += 1
            else:
                # Use real telemetry
                hw = self.wrapper.get_token_aligned(t0, t1)
            
            temp = hw.get("temp", 50)
            power = hw.get("power", 0)
            util = hw.get("util", 0)
            
            # Compute z_feel
            hw_ctx = HardwareContext.from_dict(hw)
            runtime = RuntimeContext(
                token_latency=t1 - t0,
                kv_cache_tokens=current_ids.shape[1],
                generation_depth=token_count,
            )
            sensors = self.sensor_bank(logits.detach(), runtime=runtime, hardware=hw_ctx)
            real_z_feel = self.projector(sensors.float())
            
            # Z_feel to use (depends on mode)
            self._z_feel_buffer.append(real_z_feel.clone())
            
            if mode == "z_feel_shuffle" and len(self._z_feel_buffer) > shuffle_lag:
                # Use old z_feel (breaks mediation)
                z_feel = self._z_feel_buffer[0]
            else:
                z_feel = real_z_feel
            
            # Policy step (unless baseline)
            if mode == "baseline":
                K = 1
                dvfs = "auto"
                difficulty = 0.5
                stress = 0.5
            else:
                action = self.policy.step(logits, z_feel, temperature=0.7)
                K = action.K
                dvfs = action.dvfs.value
                difficulty = action.difficulty_score
                stress = action.stress_score
                
                # Apply DVFS action
                if dvfs != "auto":
                    from src.perturbation_scheduler import DVFSLevel
                    self.perturb.set_dvfs(DVFSLevel.LOW if dvfs == "low" else DVFSLevel.HIGH)
            
            # Record
            traces["timestamps"].append(elapsed)
            traces["temps"].append(temp)
            traces["powers"].append(power)
            traces["utils"].append(util)
            traces["difficulty_scores"].append(difficulty)
            traces["stress_scores"].append(stress)
            traces["K_values"].append(K)
            traces["dvfs_actions"].append(dvfs)
            traces["z_feel_norms"].append(real_z_feel.norm().item())
            
            token_count += 1
            
            # Sample next token
            probs = F.softmax(logits / 0.7, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            current_ids = torch.cat([current_ids, next_token], dim=-1)
            
            if current_ids.shape[1] > 512:
                current_ids = input_ids.clone()
            
            # Progress
            if token_count % 100 == 0:
                print(f"    [{elapsed:.0f}s] T={temp:.1f}°C K={K} diff={difficulty:.2f} stress={stress:.2f}")
        
        executor.stop()
        
        # Compute metrics
        metrics = compute_homeostasis_metrics(traces, self.T_high)
        
        return {
            "mode": mode,
            "traces": traces,
            "metrics": metrics,
            "tokens": token_count,
        }
    
    def run_benefit_collapse_test(
        self,
        prompt: str = "Explain the principles of machine learning.",
        duration: float = 120.0,
    ) -> Dict:
        """
        Run the full benefit-collapse test.
        
        Compares: baseline, live, cross_swap, z_feel_shuffle
        """
        print("\n" + "=" * 60)
        print("  BENEFIT-COLLAPSE FALSIFICATION TEST")
        print("=" * 60)
        
        schedule = PerturbationSchedule.entropy_maximizing(duration=duration)
        print(f"  Schedule: {schedule.description}")
        
        results = {}
        
        # 1. Baseline (no control)
        print("\n--- BASELINE (no control) ---")
        self.wait_cooldown()
        results["baseline"] = self.run_trial(prompt, "baseline", schedule)
        
        # 2. Live (full control)
        print("\n--- LIVE (full control) ---")
        self.wait_cooldown()
        results["live"] = self.run_trial(prompt, "live", schedule)
        
        # 3. Cross-swap (use baseline telemetry)
        print("\n--- CROSS-SWAP (baseline telemetry) ---")
        self.wait_cooldown()
        results["cross_swap"] = self.run_trial(
            prompt, "cross_swap", schedule,
            cross_telemetry=results["baseline"]["traces"]
        )
        
        # 4. Z_feel shuffle
        print("\n--- Z_FEEL_SHUFFLE ---")
        self.wait_cooldown()
        results["z_feel_shuffle"] = self.run_trial(
            prompt, "z_feel_shuffle", schedule,
            shuffle_lag=32
        )
        
        # Compute collapse metrics
        collapse = self._compute_collapse(results)
        
        return {
            "version": VERSION,
            "duration": duration,
            "T_high": self.T_high,
            "results": results,
            "collapse": collapse,
        }
    
    def wait_cooldown(self, target: float = 50.0, timeout: float = 60.0):
        """Wait for cooldown."""
        print(f"  Cooling down to {target}°C...")
        start = time.time()
        while (time.time() - start) < timeout:
            hw = self.telemetry.get_latest()
            temp = hw.get("temp") or 100  # Handle None
            if temp <= target:
                print(f"  Cooled to {temp:.1f}°C")
                return
            time.sleep(2)
        hw = self.telemetry.get_latest()
        print(f"  Timeout, at {(hw.get('temp') or 50):.1f}°C")
    
    def _compute_collapse(self, results: Dict) -> Dict:
        """Compute benefit collapse metrics."""
        baseline = results["baseline"]["metrics"]
        live = results["live"]["metrics"]
        cross_swap = results["cross_swap"]["metrics"]
        z_shuffle = results["z_feel_shuffle"]["metrics"]
        
        def compute_benefit(metric_name: str, lower_is_better: bool = True):
            b = baseline[metric_name]
            l = live[metric_name]
            cs = cross_swap[metric_name]
            zs = z_shuffle[metric_name]
            
            if lower_is_better:
                live_benefit = b - l  # Positive = improvement
                cross_benefit = b - cs
                shuffle_benefit = b - zs
            else:
                live_benefit = l - b
                cross_benefit = cs - b
                shuffle_benefit = zs - b
            
            # Collapse = benefit that disappears under falsification
            cross_collapse = (live_benefit - cross_benefit) / (abs(live_benefit) + 1e-6)
            shuffle_collapse = (live_benefit - shuffle_benefit) / (abs(live_benefit) + 1e-6)
            
            return {
                "baseline": b,
                "live": l,
                "cross_swap": cs,
                "z_feel_shuffle": zs,
                "live_benefit": live_benefit,
                "cross_benefit": cross_benefit,
                "shuffle_benefit": shuffle_benefit,
                "cross_collapse_pct": cross_collapse * 100,
                "shuffle_collapse_pct": shuffle_collapse * 100,
            }
        
        return {
            "time_above_threshold": compute_benefit("time_above_threshold", lower_is_better=True),
            "overshoot_area": compute_benefit("overshoot_area", lower_is_better=True),
            "temp_max": compute_benefit("temp_max", lower_is_better=True),
            "interpretation": self._interpret_collapse(results),
        }
    
    def _interpret_collapse(self, results: Dict) -> str:
        """Interpret collapse results."""
        baseline = results["baseline"]["metrics"]["time_above_threshold"]
        live = results["live"]["metrics"]["time_above_threshold"]
        cross = results["cross_swap"]["metrics"]["time_above_threshold"]
        shuffle = results["z_feel_shuffle"]["metrics"]["time_above_threshold"]
        
        live_benefit = baseline - live
        cross_benefit = baseline - cross
        shuffle_benefit = baseline - shuffle
        
        if live_benefit > 0.05:  # At least 5% improvement
            if cross_benefit < live_benefit * 0.5 and shuffle_benefit < live_benefit * 0.5:
                return "PASS: Live shows benefit, falsification collapses it"
            elif cross_benefit < live_benefit * 0.5:
                return "PARTIAL: Cross-swap collapses benefit (sensing link proven)"
            elif shuffle_benefit < live_benefit * 0.5:
                return "PARTIAL: Z_feel shuffle collapses benefit (mediation link proven)"
            else:
                return "FAIL: Benefit persists under falsification (information leak)"
        else:
            return "INCONCLUSIVE: No significant live benefit to test collapse"
    
    def stop(self):
        self.perturb.cleanup()
        self.telemetry.stop()


def main():
    parser = argparse.ArgumentParser(description="Benefit Collapse Test")
    parser.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--checkpoint", default="results/feel_training_v10/final_checkpoint.pt")
    parser.add_argument("--duration", type=float, default=120, help="Duration per trial")
    parser.add_argument("--T_high", type=float, default=60.0)
    parser.add_argument("--output_dir", default="results/benefit_collapse")
    args = parser.parse_args()
    
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    test = BenefitCollapseTest(
        model_name=args.model,
        checkpoint_path=args.checkpoint,
        T_high=args.T_high,
    )
    
    results = test.run_benefit_collapse_test(duration=args.duration)
    
    # Save
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = f"{args.output_dir}/benefit_collapse_{timestamp}.json"
    
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
    collapse = results["collapse"]
    print("\n" + "=" * 60)
    print("  BENEFIT-COLLAPSE TEST RESULTS")
    print("=" * 60)
    
    for metric in ["time_above_threshold", "overshoot_area"]:
        m = collapse[metric]
        print(f"\n  {metric}:")
        print(f"    Baseline:      {m['baseline']:.3f}")
        print(f"    Live:          {m['live']:.3f} (benefit: {m['live_benefit']:.3f})")
        print(f"    Cross-swap:    {m['cross_swap']:.3f} (collapse: {m['cross_collapse_pct']:.0f}%)")
        print(f"    Z_feel_shuffle: {m['z_feel_shuffle']:.3f} (collapse: {m['shuffle_collapse_pct']:.0f}%)")
    
    print(f"\n  Interpretation: {collapse['interpretation']}")
    print("=" * 60)
    
    test.stop()


if __name__ == "__main__":
    main()
