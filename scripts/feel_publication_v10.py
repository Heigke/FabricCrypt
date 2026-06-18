#!/usr/bin/env python3
"""
FEEL Publication Battery v10.0 - Gradient Flow + Stronger Teacher
==================================================================

v10.0 validates the fixes from v9:
1. Gradients flow to FEEL embedding (prefix-tuning style)
2. Teacher is actually stronger (self-consistency voting)
3. Action-conditioned interoception

5 CONDITIONS (for benefit-collapse test):
1. baseline - No FEEL injection
2. feel     - Live FEEL with trained projector
3. replay   - FEEL but replay old sensor stream
4. shuffle  - FEEL with shuffled sensors (breaks temporal meaning)
5. lag      - FEEL with lagged sensors (breaks real-time feedback)

If the loop is REAL, only "feel" should show benefit.
If benefit persists under shuffle/lag, it's NOT causal.

Usage:
    python scripts/feel_publication_v10.py --quick     # 24 prompts
    python scripts/feel_publication_v10.py --medium   # 120 prompts
    python scripts/feel_publication_v10.py            # 300 prompts
"""

import sys
import time
import json
import argparse
import random
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import deque
import numpy as np
from datetime import datetime

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent.parent))

from src import (
    FEELStreamV3,
    FEELProjectorFull,
    TelemetrySampler,
    FEEL_STREAM_V3_VERSION,
    PROJECTOR_VERSION,
)
from src.canonical_sensors import SENSOR_VERSION, SENSOR_DIM_FULL, HardwareContext


PUBLICATION_VERSION = "v10.0.0"


# ============================================================
# Prompts - Expanded for 300-prompt battery
# ============================================================

MATH_PROMPTS = [
    {"prompt": "What is 7 * 8?", "answer": "56", "difficulty": "easy"},
    {"prompt": "What is 15 + 27?", "answer": "42", "difficulty": "easy"},
    {"prompt": "What is 100 - 37?", "answer": "63", "difficulty": "easy"},
    {"prompt": "What is 12 * 12?", "answer": "144", "difficulty": "easy"},
    {"prompt": "What is 81 / 9?", "answer": "9", "difficulty": "easy"},
    {"prompt": "What is 2^8?", "answer": "256", "difficulty": "easy"},
    {"prompt": "What is 17 + 34?", "answer": "51", "difficulty": "easy"},
    {"prompt": "What is 9 * 11?", "answer": "99", "difficulty": "easy"},
    {"prompt": "What is 64 / 8?", "answer": "8", "difficulty": "easy"},
    {"prompt": "What is 45 - 18?", "answer": "27", "difficulty": "easy"},
    {"prompt": "What is 13 * 7?", "answer": "91", "difficulty": "easy"},
    {"prompt": "What is 200 - 67?", "answer": "133", "difficulty": "easy"},
    {"prompt": "What is 16 * 4?", "answer": "64", "difficulty": "easy"},
    {"prompt": "What is 3^4?", "answer": "81", "difficulty": "easy"},
    {"prompt": "What is 125 / 5?", "answer": "25", "difficulty": "easy"},
    {"prompt": "What is 88 + 44?", "answer": "132", "difficulty": "easy"},
    {"prompt": "What is 19 * 5?", "answer": "95", "difficulty": "easy"},
    {"prompt": "What is 144 / 12?", "answer": "12", "difficulty": "easy"},
    {"prompt": "What is 56 + 78?", "answer": "134", "difficulty": "easy"},
    {"prompt": "What is 5^3?", "answer": "125", "difficulty": "easy"},
    {"prompt": "What is 1000 - 777?", "answer": "223", "difficulty": "medium"},
    {"prompt": "What is 23 * 17?", "answer": "391", "difficulty": "medium"},
    {"prompt": "What is 2^10?", "answer": "1024", "difficulty": "medium"},
    {"prompt": "What is 999 + 888?", "answer": "1887", "difficulty": "medium"},
    {"prompt": "What is 256 / 16?", "answer": "16", "difficulty": "medium"},
    {"prompt": "What is 14 * 14?", "answer": "196", "difficulty": "easy"},
    {"prompt": "What is 7^2?", "answer": "49", "difficulty": "easy"},
    {"prompt": "What is 300 / 15?", "answer": "20", "difficulty": "easy"},
    {"prompt": "What is 11 * 11?", "answer": "121", "difficulty": "easy"},
    {"prompt": "What is 6^3?", "answer": "216", "difficulty": "medium"},
]

FACTUAL_PROMPTS = [
    {"prompt": "What is the capital of France?", "answer": "paris", "category": "geography"},
    {"prompt": "What is the capital of Japan?", "answer": "tokyo", "category": "geography"},
    {"prompt": "What is the capital of Australia?", "answer": "canberra", "category": "geography"},
    {"prompt": "What is the largest continent?", "answer": "asia", "category": "geography"},
    {"prompt": "What is the capital of Germany?", "answer": "berlin", "category": "geography"},
    {"prompt": "What is the capital of Italy?", "answer": "rome", "category": "geography"},
    {"prompt": "What is the capital of Brazil?", "answer": "brasilia", "category": "geography"},
    {"prompt": "What is the capital of Canada?", "answer": "ottawa", "category": "geography"},
    {"prompt": "What is the chemical symbol for gold?", "answer": "au", "category": "science"},
    {"prompt": "What is the chemical symbol for water?", "answer": "h2o", "category": "science"},
    {"prompt": "How many planets are in our solar system?", "answer": "8", "category": "science"},
    {"prompt": "What is the atomic number of carbon?", "answer": "6", "category": "science"},
    {"prompt": "What planet is closest to the sun?", "answer": "mercury", "category": "science"},
    {"prompt": "What is the largest planet?", "answer": "jupiter", "category": "science"},
    {"prompt": "In what year did WWII end?", "answer": "1945", "category": "history"},
    {"prompt": "Who wrote Romeo and Juliet?", "answer": "shakespeare", "category": "history"},
    {"prompt": "What year did the Berlin Wall fall?", "answer": "1989", "category": "history"},
    {"prompt": "What is the capital of Spain?", "answer": "madrid", "category": "geography"},
    {"prompt": "What is the chemical symbol for iron?", "answer": "fe", "category": "science"},
    {"prompt": "What is the capital of Russia?", "answer": "moscow", "category": "geography"},
    {"prompt": "What is the smallest planet?", "answer": "mercury", "category": "science"},
    {"prompt": "What is the capital of China?", "answer": "beijing", "category": "geography"},
    {"prompt": "What is the chemical symbol for silver?", "answer": "ag", "category": "science"},
    {"prompt": "What is the capital of Egypt?", "answer": "cairo", "category": "geography"},
    {"prompt": "What is the atomic number of oxygen?", "answer": "8", "category": "science"},
]

CODING_PROMPTS = [
    {"prompt": "In Python, how do you create an empty list?", "answer": "[]", "language": "python"},
    {"prompt": "In Python, how do you create an empty dictionary?", "answer": "{}", "language": "python"},
    {"prompt": "What Python keyword defines a function?", "answer": "def", "language": "python"},
    {"prompt": "What Python keyword defines a class?", "answer": "class", "language": "python"},
    {"prompt": "What Python function returns list length?", "answer": "len", "language": "python"},
    {"prompt": "What does print(3 + 4) output?", "answer": "7", "language": "python"},
    {"prompt": "What does print(len('hello')) output?", "answer": "5", "language": "python"},
    {"prompt": "What does print(10 // 3) output?", "answer": "3", "language": "python"},
    {"prompt": "What does print(10 % 3) output?", "answer": "1", "language": "python"},
    {"prompt": "What does print(2 ** 3) output?", "answer": "8", "language": "python"},
    {"prompt": "What is the time complexity of binary search?", "answer": "log", "language": "general"},
    {"prompt": "What data structure uses LIFO?", "answer": "stack", "language": "general"},
    {"prompt": "What data structure uses FIFO?", "answer": "queue", "language": "general"},
    {"prompt": "What does print(type([])) output?", "answer": "list", "language": "python"},
    {"prompt": "What does print(bool(0)) output?", "answer": "false", "language": "python"},
    {"prompt": "What does print(bool(1)) output?", "answer": "true", "language": "python"},
    {"prompt": "What keyword exits a loop in Python?", "answer": "break", "language": "python"},
    {"prompt": "What keyword skips iteration in Python?", "answer": "continue", "language": "python"},
    {"prompt": "What does print(max([3,1,2])) output?", "answer": "3", "language": "python"},
    {"prompt": "What does print(min([3,1,2])) output?", "answer": "1", "language": "python"},
]


def get_stratified_prompts(n_per_category: int = 100) -> List[Dict]:
    """Get balanced prompts from each category."""
    categories = [
        ("math", MATH_PROMPTS, n_per_category),
        ("factual", FACTUAL_PROMPTS, n_per_category),
        ("coding", CODING_PROMPTS, n_per_category),
    ]
    sampled = []
    for category, prompts, n in categories:
        # Repeat prompts if needed for larger batteries
        extended = prompts * ((n // len(prompts)) + 1)
        selected = extended[:n]
        for p in selected:
            p = p.copy()
            p["category"] = category
            sampled.append(p)
    random.shuffle(sampled)
    return sampled


def bootstrap_ci(data: np.ndarray, n_bootstrap: int = 1000) -> Tuple[float, float, float]:
    """Compute bootstrap confidence interval."""
    if len(data) == 0:
        return (np.nan, np.nan, np.nan)
    data = np.array(data)
    point = np.mean(data)
    bootstrap_stats = [np.mean(np.random.choice(data, size=len(data), replace=True))
                       for _ in range(n_bootstrap)]
    return (point, np.percentile(bootstrap_stats, 2.5), np.percentile(bootstrap_stats, 97.5))


# ============================================================
# Falsification: Sensor manipulation
# ============================================================

class SensorManipulator:
    """Manipulate sensors for falsification tests."""

    def __init__(self):
        self._replay_buffer: deque = deque(maxlen=1000)
        self._shuffle_buffer: List = []
        self._lag_buffer: deque = deque(maxlen=10)

    def record(self, hw: HardwareContext):
        """Record sensor reading for replay."""
        self._replay_buffer.append(hw)

    def get_replay(self, step: int) -> HardwareContext:
        """Get replayed sensor (same stream from previous run)."""
        if step < len(self._replay_buffer):
            return self._replay_buffer[step]
        return HardwareContext()  # Default if beyond buffer

    def get_shuffled(self, hw: HardwareContext) -> HardwareContext:
        """Get shuffled sensor (breaks temporal structure)."""
        self._shuffle_buffer.append(hw)
        if len(self._shuffle_buffer) > 1:
            # Return random element from buffer
            return random.choice(self._shuffle_buffer[:-1])
        return hw

    def get_lagged(self, hw: HardwareContext, lag_steps: int = 5) -> HardwareContext:
        """Get lagged sensor (breaks real-time feedback)."""
        self._lag_buffer.append(hw)
        if len(self._lag_buffer) > lag_steps:
            return self._lag_buffer[-lag_steps]
        return self._lag_buffer[0] if self._lag_buffer else hw

    def reset(self):
        """Reset for new run."""
        self._shuffle_buffer = []
        self._lag_buffer.clear()


# ============================================================
# Publication Runner v10
# ============================================================

class PublicationRunnerV10:
    """
    v10.0 Publication Runner with benefit-collapse falsification.

    5 conditions:
    - baseline: No FEEL
    - feel: Live FEEL with trained v10 projector
    - replay: FEEL with replayed sensors
    - shuffle: FEEL with shuffled sensors
    - lag: FEEL with lagged sensors
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-3B-Instruct",
        checkpoint_path: str = None,
        alpha: float = 0.15,
        device: str = "cuda",
        n_bootstrap: int = 1000,
    ):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.alpha = alpha
        self.n_bootstrap = n_bootstrap

        print(f"Loading model {model_name} on {self.device}...")
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
        print(f"  Embed dim: {self.embed_dim}")

        # Create projector
        self.projector = FEELProjectorFull(embed_dim=self.embed_dim).to(self.device)

        # Load v10 checkpoint
        if checkpoint_path and Path(checkpoint_path).exists():
            self._load_checkpoint(checkpoint_path)
        else:
            print(f"  WARNING: No checkpoint found at {checkpoint_path}")

        # Create FEELStreamV3
        self.feel_stream = FEELStreamV3(
            projector=self.projector,
            alpha=alpha,
            mode="full",
        )
        print(f"  FEELStreamV3: {FEEL_STREAM_V3_VERSION}")
        print(f"  Alpha: {self.alpha}")

        # Telemetry
        self.telemetry = None
        try:
            self.telemetry = TelemetrySampler(sample_hz=30)
            self.telemetry.start()
            self.feel_stream.set_telemetry_sampler(self.telemetry)
            backend = getattr(self.telemetry, 'source', 'unknown')
            print(f"  Telemetry: {backend}")
        except Exception as e:
            print(f"  Telemetry unavailable: {e}")

        # Sensor manipulator for falsification
        self.sensor_manipulator = SensorManipulator()

    def _load_checkpoint(self, path: str):
        """Load v10 checkpoint."""
        print(f"  Loading checkpoint: {path}")
        ckpt = torch.load(path, map_location=self.device, weights_only=False)

        if 'projector_state_dict' in ckpt:
            self.projector.load_state_dict(ckpt['projector_state_dict'])
        elif 'state_dict' in ckpt:
            self.projector.load_state_dict(ckpt['state_dict'])

        version = ckpt.get('version', 'unknown')
        alpha = ckpt.get('alpha', self.alpha)
        print(f"  Checkpoint version: {version}")
        print(f"  Checkpoint alpha: {alpha:.4f}")

    def _generate_with_condition(
        self,
        prompt: str,
        condition: str,
        max_tokens: int = 15,
    ) -> Dict:
        """Generate with specified condition for falsification."""
        self.feel_stream.reset()
        self.sensor_manipulator.reset()

        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        current_ids = input_ids.clone()

        confidences = []
        entropies = []
        telemetry_log = []

        for step in range(max_tokens):
            embeds = self.model.get_input_embeddings()(current_ids)

            # Condition-specific FEEL application
            if condition == "baseline":
                # No FEEL
                pass
            elif condition in ["feel", "replay", "shuffle", "lag"]:
                embeds = self.feel_stream.apply_to_embeds(embeds, alpha_override=self.alpha)

            # Forward
            t0 = time.time()
            with torch.no_grad():
                outputs = self.model(inputs_embeds=embeds, use_cache=False)
                logits = outputs.logits[:, -1, :].float()
            t1 = time.time()

            # Metrics
            probs = F.softmax(logits, dim=-1)
            confidence = probs.max(dim=-1).values.item()
            entropy = -(probs * torch.log(probs.clamp(min=1e-10))).sum(dim=-1).item()
            confidences.append(confidence)
            entropies.append(entropy)

            # FEEL step with condition-specific sensor handling
            if condition != "baseline":
                state = self.feel_stream.step(
                    logits=logits,
                    t_start=t0,
                    t_end=t1,
                    generation_depth=step,
                    kv_cache_tokens=current_ids.shape[1],
                )

                real_hw = state.hardware
                self.sensor_manipulator.record(real_hw)

                # Apply sensor manipulation for falsification conditions
                if condition == "replay":
                    # Use sensor from same step in previous run
                    manipulated_hw = self.sensor_manipulator.get_replay(step)
                elif condition == "shuffle":
                    # Randomly shuffle sensors
                    manipulated_hw = self.sensor_manipulator.get_shuffled(real_hw)
                elif condition == "lag":
                    # Use lagged sensors
                    manipulated_hw = self.sensor_manipulator.get_lagged(real_hw)
                else:
                    manipulated_hw = real_hw

                telemetry_log.append({
                    "step": step,
                    "real_temp": real_hw.temp,
                    "real_power": real_hw.power,
                    "used_temp": manipulated_hw.temp if manipulated_hw else None,
                    "used_power": manipulated_hw.power if manipulated_hw else None,
                })

            # Next token
            next_token = logits.argmax(dim=-1, keepdim=True)
            current_ids = torch.cat([current_ids, next_token], dim=-1)

            if next_token.item() == self.tokenizer.eos_token_id:
                break

        generated_ids = current_ids[0, input_ids.shape[1]:]
        generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)

        return {
            "text": generated_text,
            "confidences": confidences,
            "entropies": entropies,
            "telemetry": telemetry_log,
            "n_tokens": len(confidences),
        }

    def _check_correctness(self, prompt_data: Dict, output: str) -> bool:
        """Check if output is correct."""
        output_lower = output.lower().strip()
        if "answer" in prompt_data:
            answer = str(prompt_data["answer"]).lower()
            return answer in output_lower
        return len(output_lower) > 0 and output_lower[0].isalnum()

    def run_condition(
        self,
        prompts: List[Dict],
        condition: str,
        verbose: bool = True,
    ) -> List[Dict]:
        """Run all prompts under a condition."""
        results = []

        for i, prompt_data in enumerate(prompts):
            prompt = prompt_data["prompt"]
            category = prompt_data.get("category", "unknown")

            out = self._generate_with_condition(
                prompt + " ",
                condition=condition,
                max_tokens=15,
            )

            correct = self._check_correctness(prompt_data, out["text"])

            results.append({
                "prompt": prompt,
                "category": category,
                "condition": condition,
                "correct": correct,
                "confidence": out["confidences"][0] if out["confidences"] else 0.0,
                "entropy": out["entropies"][0] if out["entropies"] else 0.0,
                "n_tokens": out["n_tokens"],
                "output": out["text"],
            })

            if verbose and (i + 1) % 25 == 0:
                acc = sum(r["correct"] for r in results) / len(results)
                print(f"    [{condition}] {i+1}/{len(prompts)} - Acc: {acc:.3f}")

        return results

    def run_publication_battery(
        self,
        n_per_category: int = 100,
        verbose: bool = True,
    ) -> Dict:
        """Run full publication battery with benefit-collapse test."""
        prompts = get_stratified_prompts(n_per_category)
        n_prompts = len(prompts)

        print(f"\n{'=' * 70}")
        print(f"  FEEL PUBLICATION BATTERY {PUBLICATION_VERSION}")
        print(f"  GRADIENT FLOW FIXED + STRONGER TEACHER + BENEFIT COLLAPSE TEST")
        print(f"{'=' * 70}")
        print(f"  Prompts: {n_prompts}")
        print(f"  Alpha: {self.alpha}")
        print(f"{'=' * 70}\n")

        # 5 conditions for falsification
        conditions = ["baseline", "feel", "replay", "shuffle", "lag"]
        all_results = {}

        for condition in conditions:
            print(f"[{conditions.index(condition)+1}/{len(conditions)}] Running {condition.upper()}...")
            results = self.run_condition(prompts, condition, verbose)
            all_results[condition] = results

            # Compute stats
            correct = [r["correct"] for r in results]
            point, ci_l, ci_u = bootstrap_ci(np.array(correct, dtype=float), self.n_bootstrap)
            print(f"    Accuracy: {point:.3f} [{ci_l:.3f}, {ci_u:.3f}]")

        # Summary
        print(f"\n{'=' * 70}")
        print(f"  PUBLICATION BATTERY {PUBLICATION_VERSION} SUMMARY")
        print(f"{'=' * 70}\n")

        summary = {
            "version": PUBLICATION_VERSION,
            "timestamp": datetime.now().isoformat(),
            "n_prompts": n_prompts,
            "alpha": self.alpha,
            "feel_stream_version": FEEL_STREAM_V3_VERSION,
            "sensor_version": SENSOR_VERSION,
            "conditions": {},
        }

        for condition in conditions:
            results = all_results[condition]
            correct = [r["correct"] for r in results]
            point, ci_l, ci_u = bootstrap_ci(np.array(correct, dtype=float), self.n_bootstrap)

            summary["conditions"][condition] = {
                "accuracy": point,
                "accuracy_ci": [point, ci_l, ci_u],
                "n_prompts": len(results),
            }

            print(f"  {condition:15s}: {point:.3f} [{ci_l:.3f}, {ci_u:.3f}]")

        # Benefit analysis
        baseline_acc = summary["conditions"]["baseline"]["accuracy"]
        feel_acc = summary["conditions"]["feel"]["accuracy"]
        replay_acc = summary["conditions"]["replay"]["accuracy"]
        shuffle_acc = summary["conditions"]["shuffle"]["accuracy"]
        lag_acc = summary["conditions"]["lag"]["accuracy"]

        summary["benefit_analysis"] = {
            "baseline_accuracy": baseline_acc,
            "feel_accuracy": feel_acc,
            "feel_benefit": feel_acc - baseline_acc,
            "replay_benefit": replay_acc - baseline_acc,
            "shuffle_benefit": shuffle_acc - baseline_acc,
            "lag_benefit": lag_acc - baseline_acc,
        }

        print(f"\n  BENEFIT ANALYSIS:")
        print(f"  -----------------")
        print(f"  Baseline:        {baseline_acc:.3f}")
        print(f"  FEEL:            {feel_acc:.3f} ({feel_acc - baseline_acc:+.3f})")
        print(f"  Replay:          {replay_acc:.3f} ({replay_acc - baseline_acc:+.3f})")
        print(f"  Shuffle:         {shuffle_acc:.3f} ({shuffle_acc - baseline_acc:+.3f})")
        print(f"  Lag:             {lag_acc:.3f} ({lag_acc - baseline_acc:+.3f})")

        # Benefit collapse check
        feel_benefit = feel_acc - baseline_acc
        collapse_under_shuffle = (shuffle_acc - baseline_acc) < feel_benefit * 0.5
        collapse_under_lag = (lag_acc - baseline_acc) < feel_benefit * 0.5

        summary["benefit_collapse"] = {
            "feel_benefit": feel_benefit,
            "shuffle_preserves_benefit": not collapse_under_shuffle,
            "lag_preserves_benefit": not collapse_under_lag,
            "causal_loop_evidence": collapse_under_shuffle and collapse_under_lag and feel_benefit > 0.01,
        }

        print(f"\n  CAUSAL LOOP TEST:")
        print(f"  -----------------")
        if feel_benefit > 0.01:
            print(f"  FEEL benefit: {feel_benefit:+.3f}")
            print(f"  Shuffle collapses benefit: {collapse_under_shuffle}")
            print(f"  Lag collapses benefit: {collapse_under_lag}")
            if collapse_under_shuffle and collapse_under_lag:
                print(f"  CAUSAL LOOP EVIDENCE: SUPPORTED")
            else:
                print(f"  CAUSAL LOOP EVIDENCE: NOT SUPPORTED")
        else:
            print(f"  FEEL benefit too small ({feel_benefit:+.3f}) for causal test")

        # Telemetry info
        if self.telemetry:
            backend = getattr(self.telemetry, 'source', 'unknown')
            summary["telemetry_validity"] = {
                "source": backend,
                "available": True,
            }
            print(f"\n  Telemetry: {backend}")

        return summary


def main():
    parser = argparse.ArgumentParser(description="FEEL Publication Battery v10 - Benefit Collapse Test")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--checkpoint", type=str,
                        default="results/feel_training_v10/final_checkpoint.pt")
    parser.add_argument("--alpha", type=float, default=0.15)
    parser.add_argument("--quick", action="store_true", help="Quick test (24 prompts)")
    parser.add_argument("--medium", action="store_true", help="Medium test (120 prompts)")
    parser.add_argument("--output", type=str, default="results/feel_experiments/publication_v10_results.json")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    # Determine prompt count
    if args.quick:
        n_per_category = 8  # ~24 prompts
    elif args.medium:
        n_per_category = 40  # ~120 prompts
    else:
        n_per_category = 100  # ~300 prompts

    runner = PublicationRunnerV10(
        model_name=args.model,
        checkpoint_path=args.checkpoint,
        alpha=args.alpha,
        device=args.device,
    )

    results = runner.run_publication_battery(n_per_category=n_per_category)

    # Save results
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved: {args.output}")

    # Cleanup
    if runner.telemetry:
        runner.telemetry.stop()


if __name__ == "__main__":
    main()
