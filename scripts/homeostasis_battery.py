#!/usr/bin/env python3
"""
Homeostasis Battery - Collect rollouts for offline policy learning

This script:
1. Runs generations under randomized actions (K, temperature)
2. Logs (z_feel, action, hardware_trace, quality, J) for each rollout
3. Compares conditions: random policy, fixed policy, FEEL policy

Key insight: FEEL should NOT inject into embeddings (causes harm).
Instead, z_feel drives external actions.

Usage:
    python scripts/homeostasis_battery.py --n_rollouts 200
    python scripts/homeostasis_battery.py --mode compare  # Compare policies
"""

import sys
import time
import json
import random
import argparse
import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import numpy as np

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent.parent))

from src import (
    FEELProjectorFull,
    TelemetrySampler,
    TelemetrySamplerWrapper,
)
from src.feel_regulation import (
    FEELRegulator,
    PolicyHead,
    ActionSpace,
    HomeostasisObjective,
    HomeostasisMetrics,
    Rollout,
    RolloutCollector,
    REGULATION_VERSION,
)
from src.canonical_sensors import CanonicalSensorBank, RuntimeContext, HardwareContext

VERSION = f"homeostasis-{REGULATION_VERSION}"

# Test prompts - mix of difficulties
PROMPTS = [
    # Math
    "What is 17 * 23?",
    "The derivative of x^3 + 2x is",
    "Solve for x: 2x + 5 = 15",
    "What is 15% of 80?",
    "The integral of 2x dx is",
    # Factual
    "What is the capital of France?",
    "The largest planet in our solar system is",
    "Water boils at what temperature in Celsius?",
    # Coding
    "What Python keyword defines a function?",
    "What is the time complexity of binary search?",
    "What data structure uses FIFO?",
    # Open-ended (longer generation)
    "Explain how photosynthesis works in one paragraph.",
    "Describe the steps to make a cup of coffee.",
    "What are three benefits of regular exercise?",
]


def check_answer(prompt: str, output: str) -> float:
    """
    Simple quality check - returns 0-1 score.
    For real use, replace with a verifier/grader.
    """
    output_lower = output.lower().strip()

    # Known answers
    answers = {
        "17 * 23": "391",
        "derivative": "3x^2",
        "2x + 5 = 15": "5",
        "15%": "12",
        "integral": "x^2",
        "capital of france": "paris",
        "largest planet": "jupiter",
        "water boils": "100",
        "python keyword": "def",
        "binary search": "log",
        "fifo": "queue",
    }

    for key, ans in answers.items():
        if key.lower() in prompt.lower():
            return 1.0 if ans.lower() in output_lower else 0.0

    # For open-ended, check minimum length and coherence
    if len(output.split()) >= 20:
        return 0.7
    elif len(output.split()) >= 10:
        return 0.5
    elif len(output.split()) >= 5:
        return 0.3
    return 0.1


class HomeostasisBattery:
    """
    Run homeostasis experiments with controlled action variation.
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
            dtype=torch.bfloat16 if self.device.type == "cuda" else torch.float32,
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
            print(f"  Loaded checkpoint: {ckpt.get('version', 'unknown')}")

        # Regulator (no embedding injection!)
        self.regulator = FEELRegulator(
            projector=self.projector,
            action_space=ActionSpace(),
        ).to(self.device)

        # Sensor bank (for z_feel computation)
        self.sensor_bank = CanonicalSensorBank(mode="full")

        # Telemetry
        self.telemetry = None
        try:
            self.telemetry = TelemetrySampler(sample_hz=30)
            self.telemetry.start()
            self.regulator.set_telemetry_sampler(self.telemetry)
            print(f"  Telemetry: {self.telemetry.source}")
        except Exception as e:
            print(f"  Telemetry unavailable: {e}")

        # Objective
        self.objective = HomeostasisObjective()

        # Collector
        self.collector = RolloutCollector(self.objective)

    def _get_hardware(self, t0: float, t1: float) -> Dict:
        """Get hardware context."""
        if self.telemetry:
            return self.telemetry.get_token_aligned(t0, t1)
        return {"temp": None, "power": None, "util": None}

    def generate_with_action(
        self,
        prompt: str,
        action: Tuple[int, float, str],  # (K, temperature, dvfs_mode)
        max_tokens: int = 50,
    ) -> Dict:
        """
        Generate with specified action parameters.

        Key: NO FEEL embedding injection. LM runs normally.
        We just vary K, temperature, and log z_feel.
        """
        K, temperature, dvfs_mode = action

        # TODO: Implement DVFS control
        # For now, just log the intended mode

        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)

        # Storage
        z_feels = []
        hardware_trace = []
        latencies = []
        all_outputs = []

        # Generate K samples and pick best (self-consistency style)
        for k in range(K):
            current_ids = input_ids.clone()
            sample_latencies = []
            sample_hardware = []
            sample_z_feels = []

            for step in range(max_tokens):
                # Normal LM forward (NO FEEL injection!)
                t0 = time.time()
                with torch.no_grad():
                    outputs = self.model(current_ids, use_cache=False)
                    logits = outputs.logits[:, -1, :].float()
                t1 = time.time()

                # Compute z_feel (for logging, not injection)
                hw = self._get_hardware(t0, t1)
                hw_ctx = HardwareContext.from_dict(hw)
                runtime = RuntimeContext(
                    token_latency=t1 - t0,
                    kv_cache_tokens=current_ids.shape[1],
                    generation_depth=step,
                )
                sensors = self.sensor_bank(logits.detach(), runtime=runtime, hardware=hw_ctx)
                z_feel = self.projector(sensors.float())

                sample_z_feels.append(z_feel.detach().cpu())
                sample_hardware.append({
                    "step": step,
                    "temp": hw.get("temp"),
                    "power": hw.get("power"),
                    "util": hw.get("util"),
                })
                sample_latencies.append(t1 - t0)

                # Temperature sampling
                probs = F.softmax(logits / temperature, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                current_ids = torch.cat([current_ids, next_token], dim=-1)

                if next_token.item() == self.tokenizer.eos_token_id:
                    break

            # Decode output
            generated_ids = current_ids[0, input_ids.shape[1]:]
            output_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
            all_outputs.append(output_text)

            # Use first sample's traces (or average later)
            if k == 0:
                z_feels = sample_z_feels
                hardware_trace = sample_hardware
                latencies = sample_latencies

        # For K > 1, we'd vote/select best. For simplicity, use first.
        best_output = all_outputs[0]

        return {
            "output": best_output,
            "z_feels": z_feels,
            "hardware_trace": hardware_trace,
            "latencies": latencies,
            "K": K,
            "temperature": temperature,
            "dvfs_mode": dvfs_mode,
        }

    def collect_rollout(
        self,
        prompt: str,
        policy: str = "random",  # "random", "fixed", "feel"
        fixed_action: Tuple[int, float, str] = None,
    ) -> Rollout:
        """
        Collect a single rollout.

        Args:
            prompt: input prompt
            policy: how to select action
            fixed_action: action to use if policy="fixed"
        """
        self.regulator.reset()

        # Select action
        if policy == "random":
            action = self.regulator.action_space.random_action()
        elif policy == "fixed":
            action = fixed_action or (1, 0.7, "auto")
        elif policy == "feel":
            # Use policy head (requires trained policy)
            # For now, fall back to random
            action = self.regulator.action_space.random_action()
        else:
            action = (1, 0.7, "auto")

        # Generate
        result = self.generate_with_action(prompt, action, max_tokens=50)

        # Compute quality and objective
        quality = check_answer(prompt, result["output"])

        temps = [h["temp"] for h in result["hardware_trace"]]
        powers = [h["power"] for h in result["hardware_trace"]]

        J, metrics = self.objective.compute_from_trace(
            temps=temps,
            powers=powers,
            latencies=result["latencies"],
            quality=quality,
            n_tokens=len(result["latencies"]),
        )

        rollout = Rollout(
            prompt=prompt,
            z_feels=result["z_feels"],
            actions=[(action[0], action[1], action[2])] * len(result["z_feels"]),
            hardware_trace=result["hardware_trace"],
            latencies=result["latencies"],
            quality=quality,
            J=J,
            metrics=metrics,
        )

        return rollout

    def run_collection(
        self,
        prompts: List[str],
        n_rollouts: int = 200,
        policy: str = "random",
    ) -> List[Rollout]:
        """
        Collect multiple rollouts for offline learning.
        """
        print(f"\nCollecting {n_rollouts} rollouts with {policy} policy...")
        print("-" * 60)

        rollouts = []
        for i in range(n_rollouts):
            prompt = random.choice(prompts)
            rollout = self.collect_rollout(prompt, policy=policy)
            rollouts.append(rollout)
            self.collector.add(rollout)

            if (i + 1) % 20 == 0:
                recent_J = np.mean([r.J for r in rollouts[-20:]])
                recent_Q = np.mean([r.quality for r in rollouts[-20:]])
                print(f"  [{i+1}/{n_rollouts}] J={recent_J:.3f}, Q={recent_Q:.3f}")

        return rollouts

    def compare_policies(
        self,
        prompts: List[str],
        n_per_policy: int = 50,
    ) -> Dict:
        """
        Compare different policy conditions.
        """
        conditions = {
            "fixed_K1_T07": (1, 0.7, "auto"),
            "fixed_K2_T07": (2, 0.7, "auto"),
            "fixed_K4_T05": (4, 0.5, "auto"),
            "fixed_K1_T02": (1, 0.2, "auto"),
        }

        results = {}

        for name, action in conditions.items():
            print(f"\nRunning condition: {name}...")
            rollouts = []

            for i in range(n_per_policy):
                prompt = random.choice(prompts)
                rollout = self.collect_rollout(prompt, policy="fixed", fixed_action=action)
                rollouts.append(rollout)

            # Summarize
            Js = [r.J for r in rollouts]
            Qs = [r.quality for r in rollouts]
            temps = [r.metrics.temp_max for r in rollouts if r.metrics.temp_max > 0]
            powers = [r.metrics.power_mean for r in rollouts if r.metrics.power_mean > 0]

            results[name] = {
                "J_mean": float(np.mean(Js)),
                "J_std": float(np.std(Js)),
                "quality_mean": float(np.mean(Qs)),
                "temp_max_mean": float(np.mean(temps)) if temps else 0,
                "power_mean": float(np.mean(powers)) if powers else 0,
                "n_rollouts": len(rollouts),
            }

            print(f"  J={results[name]['J_mean']:.3f}, Q={results[name]['quality_mean']:.3f}")

        return results


def main():
    parser = argparse.ArgumentParser(description="Homeostasis Battery")
    parser.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--checkpoint", default="results/feel_training_v10/final_checkpoint.pt")
    parser.add_argument("--n_rollouts", type=int, default=100)
    parser.add_argument("--mode", choices=["collect", "compare"], default="collect")
    parser.add_argument("--output_dir", default="results/homeostasis")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print(f"  HOMEOSTASIS BATTERY {VERSION}")
    print("=" * 60)

    battery = HomeostasisBattery(
        model_name=args.model,
        checkpoint_path=args.checkpoint,
    )

    if args.mode == "collect":
        # Collect rollouts for offline learning
        rollouts = battery.run_collection(
            prompts=PROMPTS,
            n_rollouts=args.n_rollouts,
            policy="random",
        )

        # Save rollouts
        rollout_path = f"{args.output_dir}/rollouts_{args.n_rollouts}.pkl"
        battery.collector.save(rollout_path)
        print(f"\nSaved rollouts: {rollout_path}")

        # Summary
        Js = [r.J for r in rollouts]
        Qs = [r.quality for r in rollouts]
        print(f"\nSummary ({len(rollouts)} rollouts):")
        print(f"  J: {np.mean(Js):.3f} ± {np.std(Js):.3f}")
        print(f"  Quality: {np.mean(Qs):.3f} ± {np.std(Qs):.3f}")

    elif args.mode == "compare":
        # Compare fixed policies
        results = battery.compare_policies(PROMPTS, n_per_policy=50)

        # Save comparison
        with open(f"{args.output_dir}/policy_comparison.json", "w") as f:
            json.dump({
                "version": VERSION,
                "timestamp": datetime.now().isoformat(),
                "results": results,
            }, f, indent=2)

        # Print table
        print("\n" + "=" * 60)
        print("POLICY COMPARISON")
        print("=" * 60)
        print(f"{'Policy':<20} {'J':>8} {'Quality':>10} {'Temp':>8} {'Power':>8}")
        print("-" * 60)
        for name, r in sorted(results.items(), key=lambda x: -x[1]["J_mean"]):
            print(f"{name:<20} {r['J_mean']:>8.3f} {r['quality_mean']:>10.3f} "
                  f"{r['temp_max_mean']:>8.1f} {r['power_mean']:>8.1f}")

    if battery.telemetry:
        battery.telemetry.stop()


if __name__ == "__main__":
    main()
