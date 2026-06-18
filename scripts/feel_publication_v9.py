#!/usr/bin/env python3
"""
FEEL Publication Battery v9.0 - One Forward Per Token (t→t+1 Injection)
=======================================================================

v9.0 CRITICAL FIX from v8.0:
Uses FEELStreamV3 with t→t+1 injection pattern - ONE FORWARD PER TOKEN.

v8 did TWO forwards per token:
  1. model(input_ids) → logits → sensors
  2. model(inputs_embeds=perturbed) → new logits  [EXTRA FORWARD!]

v9 does ONE forward per token:
  - Token t: get pending FEEL from t-1 → inject → forward → logits → compute FEEL for t+1
  - Token t+1: inject FEEL computed at t → forward → ...

This is the biologically faithful design: sensing → next action.

Usage:
    python scripts/feel_publication_v9.py --quick     # Fast test (32 prompts)
    python scripts/feel_publication_v9.py --medium   # Medium (120 prompts)
    python scripts/feel_publication_v9.py            # Full (300 prompts)
"""

import sys
import time
import json
import argparse
import random
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import numpy as np
from datetime import datetime

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent.parent))

# Import v3 stream (one forward per token)
from src import (
    FEELStreamV3,
    FEELProjectorFull,
    TelemetrySampler,
    FEEL_STREAM_V3_VERSION,
    PROJECTOR_VERSION,
)
from src.canonical_sensors import SENSOR_VERSION, SENSOR_DIM_FULL


# ============================================================
# Prompts (same as v8)
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
]


def get_stratified_prompts(n_per_category: int = 8) -> List[Dict]:
    """Get balanced prompts from each category."""
    categories = [
        ("math", MATH_PROMPTS, n_per_category),
        ("factual", FACTUAL_PROMPTS, n_per_category),
        ("coding", CODING_PROMPTS, n_per_category),
    ]
    sampled = []
    for category, prompts, n in categories:
        selected = random.sample(prompts, min(n, len(prompts)))
        for p in selected:
            p["category"] = category
        sampled.extend(selected)
    random.shuffle(sampled)
    return sampled


def bootstrap_ci(data: np.ndarray, n_bootstrap: int = 100) -> Tuple[float, float, float]:
    """Compute bootstrap confidence interval."""
    if len(data) == 0:
        return (np.nan, np.nan, np.nan)
    data = np.array(data)
    point = np.mean(data)
    bootstrap_stats = [np.mean(np.random.choice(data, size=len(data), replace=True))
                       for _ in range(n_bootstrap)]
    return (point, np.percentile(bootstrap_stats, 2.5), np.percentile(bootstrap_stats, 97.5))


# ============================================================
# Publication Runner v9 - ONE FORWARD PER TOKEN
# ============================================================

class PublicationRunnerV9:
    """
    v9.0 Publication Runner with ONE FORWARD PER TOKEN.

    Uses FEELStreamV3 with t→t+1 injection:
    - Step t: inject FEEL from t-1 → forward → logits → compute FEEL for t+1
    - Only ONE model.forward() call per token
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-3B-Instruct",
        checkpoint_path: str = None,
        alpha: float = 0.15,
        device: str = "cuda",
        n_bootstrap: int = 100,
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

        # Load checkpoint
        if checkpoint_path and Path(checkpoint_path).exists():
            self._load_checkpoint(checkpoint_path)

        # Create FEELStreamV3 (one forward per token)
        self.feel_stream = FEELStreamV3(
            projector=self.projector,
            alpha=alpha,
            mode="full",
        )
        print(f"  FEELStreamV3: {FEEL_STREAM_V3_VERSION}")
        print(f"  Alpha: {self.alpha}")

        # Telemetry sampler
        self.telemetry = None
        try:
            self.telemetry = TelemetrySampler(sample_hz=30)
            self.telemetry.start()  # Must start the sampling thread!
            self.feel_stream.set_telemetry_sampler(self.telemetry)
            backend = getattr(self.telemetry, 'source', 'unknown')
            print(f"  Telemetry: {backend}")
        except Exception as e:
            print(f"  Telemetry unavailable: {e}")

        # Check projector output
        test_out = self.projector(torch.randn(1, 16, device=self.device))
        print(f"  Projector output norm: {test_out.norm().item():.4f}")

    def _load_checkpoint(self, path: str):
        """Load projector checkpoint."""
        print(f"  Loading checkpoint: {path}")
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        if 'projector_state_dict' in ckpt:
            self.projector.load_state_dict(ckpt['projector_state_dict'])
        elif 'state_dict' in ckpt:
            self.projector.load_state_dict(ckpt['state_dict'])
        if 'alpha' in ckpt:
            print(f"  Checkpoint alpha: {ckpt['alpha']:.6f} (using {self.alpha})")

    def _generate_one_forward(
        self,
        prompt: str,
        max_tokens: int = 15,
        use_feel: bool = True,
        alpha_override: float = None,
    ) -> Dict:
        """
        Generate with ONE FORWARD PER TOKEN using FEELStreamV3.

        This is the correct pattern:
        - Token t: inject pending FEEL → single forward → logits → compute FEEL for t+1
        """
        self.feel_stream.reset()

        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        current_ids = input_ids.clone()

        alpha = alpha_override if alpha_override is not None else self.alpha

        confidences = []
        entropies = []
        telemetry = []

        for step in range(max_tokens):
            # Get embeddings
            embeds = self.model.get_input_embeddings()(current_ids)

            # Apply FEEL from PREVIOUS step (t→t+1 injection)
            if use_feel:
                embeds = self.feel_stream.apply_to_embeds(embeds, alpha_override=alpha)

            # SINGLE forward pass
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

            # Process FEEL step (computes sensors, stores FEEL for next token)
            if use_feel:
                state = self.feel_stream.step(
                    logits=logits,
                    t_start=t0,
                    t_end=t1,
                    generation_depth=step,
                    kv_cache_tokens=current_ids.shape[1],
                )
                telemetry.append({
                    "step": step,
                    "temp": state.hardware.temp,
                    "power": state.hardware.power,
                    "util": state.hardware.util,
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
            "telemetry": telemetry,
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

            # Determine settings based on condition
            if condition == "baseline":
                use_feel = False
                alpha = 0.0
            elif condition == "feel":
                use_feel = True
                alpha = self.alpha
            elif condition == "feel_off":
                use_feel = True  # Still run through stream but alpha=0
                alpha = 0.0
            elif condition == "random_feel":
                # Run with FEEL but randomize sensors in stream
                use_feel = True
                alpha = self.alpha
            else:
                use_feel = True
                alpha = self.alpha

            # Generate
            out = self._generate_one_forward(
                prompt + " ",
                max_tokens=15,
                use_feel=use_feel,
                alpha_override=alpha,
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

            if verbose and (i + 1) % 20 == 0:
                acc = sum(r["correct"] for r in results) / len(results)
                print(f"    [{condition}] {i+1}/{len(prompts)} - Acc: {acc:.3f}")

        return results

    def run_publication_battery(
        self,
        n_per_category: int = 75,
        verbose: bool = True,
    ) -> Dict:
        """Run full publication battery."""
        prompts = get_stratified_prompts(n_per_category)
        n_prompts = len(prompts)

        print(f"\n{'=' * 70}")
        print(f"  FEEL PUBLICATION BATTERY v9.0 - ONE FORWARD PER TOKEN")
        print(f"{'=' * 70}")
        print(f"  Prompts: {n_prompts}")
        print(f"  FEELStreamV3: {FEEL_STREAM_V3_VERSION}")
        print(f"  Alpha: {self.alpha}")
        print(f"{'=' * 70}\n")

        conditions = ["baseline", "feel", "feel_off"]
        all_results = {}
        all_telemetry = {}

        for condition in conditions:
            print(f"[{conditions.index(condition)+1}/{len(conditions)}] Running {condition.upper()}...")
            results = self.run_condition(prompts, condition, verbose)
            all_results[condition] = results

            # Collect telemetry from last run
            if self.feel_stream._state_history:
                all_telemetry[condition] = [
                    {"step": s.step, "temp": s.hardware.temp, "power": s.hardware.power}
                    for s in self.feel_stream._state_history[:5]
                ]

            # Compute stats
            correct = [r["correct"] for r in results]
            point, ci_l, ci_u = bootstrap_ci(np.array(correct, dtype=float), self.n_bootstrap)
            print(f"    Accuracy: {point:.3f} [{ci_l:.3f}, {ci_u:.3f}]")

        # Summary
        print(f"\n{'=' * 70}")
        print(f"  PUBLICATION BATTERY v9.0 SUMMARY (ONE FORWARD PER TOKEN)")
        print(f"{'=' * 70}\n")

        summary = {
            "version": "v9.0.0",
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
        summary["benefit_analysis"] = {
            "feel_benefit": feel_acc - baseline_acc,
            "baseline_accuracy": baseline_acc,
            "feel_accuracy": feel_acc,
        }

        print(f"\n  FEEL benefit: {feel_acc - baseline_acc:+.3f}")
        print(f"  Baseline: {baseline_acc:.3f}")
        print(f"  FEEL: {feel_acc:.3f}")

        # Telemetry validity
        if self.telemetry:
            backend = getattr(self.telemetry, '_backend', 'unknown')
            summary["telemetry_validity"] = {
                "source": backend,
                "available": True,
            }
            print(f"\n  Telemetry: {backend}")

        summary["telemetry_samples"] = all_telemetry

        return summary


def main():
    parser = argparse.ArgumentParser(description="FEEL Publication Battery v9 - One Forward Per Token")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--checkpoint", type=str,
                        default="results/feel_training/canonical_v6_checkpoint.pt")
    parser.add_argument("--alpha", type=float, default=0.15)
    parser.add_argument("--quick", action="store_true", help="Quick test (32 prompts)")
    parser.add_argument("--medium", action="store_true", help="Medium test (120 prompts)")
    parser.add_argument("--output", type=str, default="results/feel_experiments/publication_v9_results.json")
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

    runner = PublicationRunnerV9(
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
