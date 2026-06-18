#!/usr/bin/env python3
"""
Correctness Prediction Experiment: Can latent signals predict errors?

Key Research Pivot:
- Result 10 showed: latent signals do NOT predict TPOT spikes (AUC≈0.5)
- New hypothesis: latent signals may predict ERROR PROBABILITY instead

This experiment tests whether internal model signals (entropy, margin,
latent delta) can predict when the model will be WRONG - enabling
"spend more compute when it matters" adaptive inference.

Based on: "Adaptive Token Allocation for Efficient LLM Reasoning"
"""

import json
import re
import sys
import time
import argparse
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.energy_harness.amd_smi_monitor import PowerTraceRecorder

try:
    from sklearn.metrics import roc_auc_score
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


# === EXPANDED QUALITY EVALUATION SUITE ===
# Mix of easy/medium/hard with verifiable answers

EVAL_SUITE = [
    # === MATH (40 items) ===
    # Easy arithmetic
    {"q": "What is 7 + 8?", "a": "15", "cat": "math", "diff": "easy"},
    {"q": "What is 23 - 9?", "a": "14", "cat": "math", "diff": "easy"},
    {"q": "What is 6 * 7?", "a": "42", "cat": "math", "diff": "easy"},
    {"q": "What is 81 / 9?", "a": "9", "cat": "math", "diff": "easy"},
    {"q": "What is 15 + 27?", "a": "42", "cat": "math", "diff": "easy"},
    {"q": "What is 100 - 37?", "a": "63", "cat": "math", "diff": "easy"},
    {"q": "What is 8 * 9?", "a": "72", "cat": "math", "diff": "easy"},
    {"q": "What is 144 / 12?", "a": "12", "cat": "math", "diff": "easy"},

    # Medium arithmetic
    {"q": "What is 17 * 13?", "a": "221", "cat": "math", "diff": "medium"},
    {"q": "What is 256 / 16?", "a": "16", "cat": "math", "diff": "medium"},
    {"q": "What is 15 + 27 + 38?", "a": "80", "cat": "math", "diff": "medium"},
    {"q": "What is (5 + 3) * 7?", "a": "56", "cat": "math", "diff": "medium"},
    {"q": "What is 23 * 11?", "a": "253", "cat": "math", "diff": "medium"},
    {"q": "What is 625 / 25?", "a": "25", "cat": "math", "diff": "medium"},
    {"q": "What is 19 + 23 + 17?", "a": "59", "cat": "math", "diff": "medium"},
    {"q": "What is (12 - 4) * 6?", "a": "48", "cat": "math", "diff": "medium"},

    # Hard multi-step
    {"q": "If x = 5, what is 3x + 7?", "a": "22", "cat": "math", "diff": "hard"},
    {"q": "What is 25% of 80?", "a": "20", "cat": "math", "diff": "hard"},
    {"q": "What is 12 squared?", "a": "144", "cat": "math", "diff": "hard"},
    {"q": "What is the average of 10, 20, and 30?", "a": "20", "cat": "math", "diff": "hard"},
    {"q": "If y = 3, what is 2y² + 1?", "a": "19", "cat": "math", "diff": "hard"},
    {"q": "What is 15% of 200?", "a": "30", "cat": "math", "diff": "hard"},
    {"q": "What is 7 cubed?", "a": "343", "cat": "math", "diff": "hard"},
    {"q": "What is the sum of the first 5 positive integers?", "a": "15", "cat": "math", "diff": "hard"},

    # === FACTUAL QA (40 items) ===
    # Easy factual
    {"q": "What is the capital of France?", "a": "Paris", "cat": "qa", "diff": "easy"},
    {"q": "How many days are in a week?", "a": "7", "cat": "qa", "diff": "easy"},
    {"q": "What planet is closest to the Sun?", "a": "Mercury", "cat": "qa", "diff": "easy"},
    {"q": "What is the chemical symbol for water?", "a": "H2O", "cat": "qa", "diff": "easy"},
    {"q": "What color is the sky on a clear day?", "a": "blue", "cat": "qa", "diff": "easy"},
    {"q": "How many months are in a year?", "a": "12", "cat": "qa", "diff": "easy"},
    {"q": "What is the largest planet in our solar system?", "a": "Jupiter", "cat": "qa", "diff": "easy"},
    {"q": "What is the chemical symbol for gold?", "a": "Au", "cat": "qa", "diff": "easy"},

    # Medium factual
    {"q": "Who wrote Romeo and Juliet?", "a": "Shakespeare", "cat": "qa", "diff": "medium"},
    {"q": "What is the largest ocean?", "a": "Pacific", "cat": "qa", "diff": "medium"},
    {"q": "How many continents are there?", "a": "7", "cat": "qa", "diff": "medium"},
    {"q": "What gas do plants absorb?", "a": "CO2", "cat": "qa", "diff": "medium"},
    {"q": "What is the capital of Japan?", "a": "Tokyo", "cat": "qa", "diff": "medium"},
    {"q": "Who painted the Mona Lisa?", "a": "Leonardo", "cat": "qa", "diff": "medium"},
    {"q": "What is the smallest prime number?", "a": "2", "cat": "qa", "diff": "medium"},
    {"q": "What year did World War II end?", "a": "1945", "cat": "qa", "diff": "medium"},

    # Hard factual
    {"q": "What is the atomic number of carbon?", "a": "6", "cat": "qa", "diff": "hard"},
    {"q": "Who discovered penicillin?", "a": "Fleming", "cat": "qa", "diff": "hard"},
    {"q": "What is the speed of light in km/s (approximately)?", "a": "300000", "cat": "qa", "diff": "hard"},
    {"q": "What element has the symbol Fe?", "a": "Iron", "cat": "qa", "diff": "hard"},

    # === REASONING (20 items) ===
    {"q": "If all roses are flowers and some flowers are red, can we conclude all roses are red?", "a": "No", "cat": "reasoning", "diff": "hard"},
    {"q": "A bat and ball cost $1.10 total. The bat costs $1 more than the ball. What does the ball cost in cents?", "a": "5", "cat": "reasoning", "diff": "hard"},
    {"q": "If it takes 5 machines 5 minutes to make 5 widgets, how many minutes would it take 100 machines to make 100 widgets?", "a": "5", "cat": "reasoning", "diff": "hard"},
    {"q": "In a lake, there's a patch of lily pads. Every day, the patch doubles in size. If it takes 48 days for the patch to cover the entire lake, how many days would it take for the patch to cover half the lake?", "a": "47", "cat": "reasoning", "diff": "hard"},
]


def check_answer(output: str, expected: str) -> bool:
    """Check if output contains the expected answer."""
    output_lower = output.lower().strip()
    expected_lower = expected.lower().strip()

    if expected_lower in output_lower:
        return True

    # Check for number formats
    numbers = re.findall(r'\b\d+(?:\.\d+)?\b', output)
    if expected_lower in [n.lower() for n in numbers]:
        return True

    # For yes/no answers
    if expected_lower in ["yes", "no"]:
        return expected_lower in output_lower

    return False


class SignalCapture:
    """Capture per-token signals during generation."""

    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.signals = []
        self.prev_hidden = None
        self._hook_handle = None
        self._captured_hidden = None

        # Register hook on final norm layer
        self._setup_hook()

    def _setup_hook(self):
        """Setup forward hook on final norm layer."""
        # Find the final norm layer (works for Qwen, Llama, etc.)
        if hasattr(self.model, 'model') and hasattr(self.model.model, 'norm'):
            norm_layer = self.model.model.norm
        elif hasattr(self.model, 'transformer') and hasattr(self.model.transformer, 'ln_f'):
            norm_layer = self.model.transformer.ln_f
        else:
            raise ValueError("Could not find final norm layer")

        def hook_fn(module, inp, out):
            # Capture last token hidden state
            self._captured_hidden = out[:, -1, :].detach()

        self._hook_handle = norm_layer.register_forward_hook(hook_fn)

    def capture_step(self, logits: torch.Tensor) -> Dict[str, float]:
        """Capture signals for current generation step."""
        # Get top logits for margin
        top_logits, _ = torch.topk(logits[0, -1], k=2)
        margin = (top_logits[0] - top_logits[1]).item()

        # Compute entropy (approximate, on top-k)
        top_k = 100
        top_logits_k, _ = torch.topk(logits[0, -1], k=min(top_k, logits.shape[-1]))
        probs = F.softmax(top_logits_k, dim=-1)
        entropy = -(probs * torch.log(probs + 1e-10)).sum().item()

        # Compute hidden state delta
        if self._captured_hidden is not None:
            current = self._captured_hidden
            if self.prev_hidden is not None:
                delta = torch.norm(current - self.prev_hidden) / (torch.norm(current) + 1e-10)
                delta_norm = delta.item()
            else:
                delta_norm = 0.0
            self.prev_hidden = current.clone()
        else:
            delta_norm = 0.0

        signal = {
            "margin": margin,
            "entropy": entropy,
            "delta_norm": delta_norm,
        }
        self.signals.append(signal)
        return signal

    def reset(self):
        """Reset for new generation."""
        self.signals = []
        self.prev_hidden = None

    def get_aggregated_signals(self) -> Dict[str, float]:
        """Get aggregated signals across all tokens."""
        if not self.signals:
            return {"margin_mean": 0, "entropy_mean": 0, "delta_mean": 0}

        return {
            "margin_mean": sum(s["margin"] for s in self.signals) / len(self.signals),
            "margin_min": min(s["margin"] for s in self.signals),
            "entropy_mean": sum(s["entropy"] for s in self.signals) / len(self.signals),
            "entropy_max": max(s["entropy"] for s in self.signals),
            "delta_mean": sum(s["delta_norm"] for s in self.signals) / len(self.signals),
            "delta_max": max(s["delta_norm"] for s in self.signals),
            "n_tokens": len(self.signals),
        }

    def cleanup(self):
        """Remove hooks."""
        if self._hook_handle:
            self._hook_handle.remove()


def generate_with_signals(
    model, tokenizer, prompt: str, max_tokens: int, temperature: float,
    signal_capture: SignalCapture
) -> Tuple[str, Dict[str, float]]:
    """Generate with per-token signal capture."""
    signal_capture.reset()

    messages = [{"role": "user", "content": f"{prompt}\nAnswer with just the final answer, no explanation."}]
    formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(formatted, return_tensors="pt").to(model.device)

    input_len = inputs.input_ids.shape[1]
    generated_ids = inputs.input_ids.clone()

    with torch.no_grad():
        for _ in range(max_tokens):
            outputs = model(generated_ids)
            logits = outputs.logits

            # Capture signals
            signal_capture.capture_step(logits)

            # Sample next token
            if temperature > 0:
                probs = F.softmax(logits[:, -1] / temperature, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = logits[:, -1].argmax(dim=-1, keepdim=True)

            generated_ids = torch.cat([generated_ids, next_token], dim=1)

            # Stop on EOS
            if next_token.item() == tokenizer.eos_token_id:
                break

    # Decode output
    output_ids = generated_ids[0, input_len:]
    output_text = tokenizer.decode(output_ids, skip_special_tokens=True)

    return output_text, signal_capture.get_aggregated_signals()


def run_correctness_prediction_experiment(
    model_name: str = "Qwen/Qwen2.5-0.5B-Instruct",
    temperatures: List[float] = [0.0, 0.7],
    max_tokens: int = 32,
    output_dir: Path = Path("results/correctness_prediction"),
) -> Dict[str, Any]:
    """
    Run experiment: can latent signals predict correctness?

    For each question:
    1. Generate answer with signal capture
    2. Check if correct
    3. Record (signals, is_correct) pair

    Then compute AUC: can signals predict error?
    """
    print(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    signal_capture = SignalCapture(model, tokenizer)

    # Warmup
    print("Warming up...")
    for _ in range(3):
        generate_with_signals(model, tokenizer, "Hello", 5, 0.0, signal_capture)

    results = []

    for temp in temperatures:
        print(f"\n=== Temperature {temp} ===")

        for idx, item in enumerate(EVAL_SUITE):
            output, signals = generate_with_signals(
                model, tokenizer, item["q"], max_tokens, temp, signal_capture
            )

            is_correct = check_answer(output, item["a"])

            result = {
                "question": item["q"],
                "expected": item["a"],
                "output": output[:100],
                "category": item["cat"],
                "difficulty": item["diff"],
                "temperature": temp,
                "is_correct": is_correct,
                **signals
            }
            results.append(result)

            status = "✓" if is_correct else "✗"
            print(f"  [{idx+1}/{len(EVAL_SUITE)}] {status} {item['cat']}/{item['diff']}: "
                  f"margin={signals['margin_mean']:.2f}, entropy={signals['entropy_mean']:.2f}")

    signal_capture.cleanup()

    # Analyze: can signals predict errors?
    analysis = analyze_correctness_prediction(results)

    # Save results
    output_dir.mkdir(parents=True, exist_ok=True)
    model_short = model_name.split("/")[-1]
    output_file = output_dir / f"correctness_prediction_{model_short}.json"

    with open(output_file, "w") as f:
        json.dump({
            "model": model_name,
            "temperatures": temperatures,
            "n_items": len(EVAL_SUITE),
            "analysis": analysis,
            "results": results
        }, f, indent=2)

    print(f"\nSaved: {output_file}")
    return analysis


def analyze_correctness_prediction(results: List[Dict]) -> Dict[str, Any]:
    """Analyze whether signals predict correctness."""
    if not HAS_SKLEARN:
        return {"error": "sklearn not available"}

    analysis = {}

    # Group by temperature
    by_temp = defaultdict(list)
    for r in results:
        by_temp[r["temperature"]].append(r)

    for temp, items in by_temp.items():
        # Labels: 1 = error (we want to predict errors)
        labels = [0 if r["is_correct"] else 1 for r in items]

        # Skip if all same class
        if len(set(labels)) < 2:
            analysis[f"temp_{temp}"] = {"error": "all same class"}
            continue

        temp_analysis = {
            "n_items": len(items),
            "n_correct": sum(1 for r in items if r["is_correct"]),
            "n_errors": sum(1 for r in items if not r["is_correct"]),
            "accuracy": sum(1 for r in items if r["is_correct"]) / len(items),
        }

        # Compute AUC for each signal
        signal_features = ["margin_mean", "margin_min", "entropy_mean", "entropy_max", "delta_mean", "delta_max"]

        for feature in signal_features:
            values = [r.get(feature, 0) for r in items]

            # For margin: LOW margin → likely error (invert for AUC)
            if "margin" in feature:
                values_for_auc = [-v for v in values]  # Invert: low margin = high error risk
            else:
                # For entropy/delta: HIGH values → likely error
                values_for_auc = values

            try:
                auc = roc_auc_score(labels, values_for_auc)
                temp_analysis[f"auc_{feature}"] = auc
            except Exception as e:
                temp_analysis[f"auc_{feature}"] = None

        # Analyze by category
        by_cat = defaultdict(list)
        for r in items:
            by_cat[r["category"]].append(r)

        temp_analysis["by_category"] = {}
        for cat, cat_items in by_cat.items():
            n_correct = sum(1 for r in cat_items if r["is_correct"])
            temp_analysis["by_category"][cat] = {
                "accuracy": n_correct / len(cat_items),
                "n": len(cat_items)
            }

        # Analyze by difficulty
        by_diff = defaultdict(list)
        for r in items:
            by_diff[r["difficulty"]].append(r)

        temp_analysis["by_difficulty"] = {}
        for diff, diff_items in by_diff.items():
            n_correct = sum(1 for r in diff_items if r["is_correct"])
            temp_analysis["by_difficulty"][diff] = {
                "accuracy": n_correct / len(diff_items),
                "n": len(diff_items)
            }

        analysis[f"temp_{temp}"] = temp_analysis

    return analysis


def main():
    parser = argparse.ArgumentParser(description="Correctness prediction experiment")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--output-dir", type=Path, default=Path("results/correctness_prediction"))
    args = parser.parse_args()

    analysis = run_correctness_prediction_experiment(
        model_name=args.model,
        output_dir=args.output_dir,
    )

    print("\n" + "="*60)
    print("CORRECTNESS PREDICTION ANALYSIS")
    print("="*60)

    for temp_key, data in analysis.items():
        if not temp_key.startswith("temp_"):
            continue

        print(f"\n{temp_key}:")
        print(f"  Accuracy: {data.get('accuracy', 0)*100:.1f}% ({data.get('n_correct', 0)}/{data.get('n_items', 0)})")

        print("\n  AUC for predicting ERRORS (>0.5 = signal predicts errors):")
        for key, val in data.items():
            if key.startswith("auc_"):
                signal = key.replace("auc_", "")
                if val is not None:
                    interpretation = "✓ predictive" if val > 0.6 else ("~ marginal" if val > 0.55 else "✗ random")
                    print(f"    {signal}: {val:.3f} {interpretation}")

        print("\n  By category:")
        for cat, cat_data in data.get("by_category", {}).items():
            print(f"    {cat}: {cat_data['accuracy']*100:.1f}%")

        print("\n  By difficulty:")
        for diff, diff_data in data.get("by_difficulty", {}).items():
            print(f"    {diff}: {diff_data['accuracy']*100:.1f}%")


if __name__ == "__main__":
    main()
