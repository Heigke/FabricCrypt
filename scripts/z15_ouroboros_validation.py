#!/usr/bin/env python3
"""
FEEL v15.0: Ouroboros Validation Suite
======================================
Tests the Ouroboros-trained model to verify that steering vectors
now produce LEARNED responses rather than transient effects.

Key Tests:
1. Introspection: Does the model articulate internal states?
2. Regulation: Does it produce shorter outputs when "stressed"?
3. Consistency: Are responses stable across multiple trials?

Author: FEEL Research Team
Date: 2026-01-11
"""

import os
import json
import torch
import argparse
import time
from pathlib import Path
from typing import Dict, List, Tuple
from dataclasses import dataclass
import numpy as np
from scipy import stats

os.environ.setdefault("HSA_OVERRIDE_GFX_VERSION", "11.0.0")
os.environ.setdefault("PYTORCH_HIP_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("HIP_FORCE_DEV_KERNARG", "1")

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# =============================================================================
# STEERING CONTROLLER
# =============================================================================

class ValidationSteeringController:
    """Multi-layer steering for validation."""

    def __init__(self, model, tokenizer, device: str = "cuda"):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.vectors: Dict[str, torch.Tensor] = {}
        self.hooks = []

        # Get base model
        if hasattr(model, "base_model"):
            self.base_model = model.base_model.model
        elif hasattr(model, "model"):
            self.base_model = model.model
        else:
            self.base_model = model

        # Get layers
        if hasattr(self.base_model, "layers"):
            self.layers = self.base_model.layers
            self.total_layers = len(self.layers)
        elif hasattr(self.base_model, "h"):
            self.layers = self.base_model.h
            self.total_layers = len(self.layers)
        else:
            self.total_layers = 28
            self.layers = None

        # Multi-layer targets
        self.target_layers = [
            self.total_layers // 4,
            self.total_layers // 2,
            3 * self.total_layers // 4,
            self.total_layers - 2,
        ]

    def mine_vector(self, positive: str, negative: str) -> torch.Tensor:
        """Extract steering vector."""
        pos_ids = self.tokenizer(positive, return_tensors="pt").input_ids.to(self.device)
        neg_ids = self.tokenizer(negative, return_tensors="pt").input_ids.to(self.device)

        with torch.no_grad():
            pos_out = self.model(pos_ids, output_hidden_states=True)
            neg_out = self.model(neg_ids, output_hidden_states=True)

            pos_h = pos_out.hidden_states[self.total_layers // 2][:, -1, :]
            neg_h = neg_out.hidden_states[self.total_layers // 2][:, -1, :]

        vector = pos_h - neg_h
        return (vector / (vector.norm() + 1e-8)).squeeze()

    def create_vectors(self):
        """Mine all vectors."""
        pairs = {
            "OVERHEAT": ("System overheating high temperature", "System cool optimal temperature"),
            "SCARCITY": ("Resources scarce must conserve", "Resources abundant plenty available"),
            "STRAIN": ("Heavy load maximum effort", "Light load minimal effort"),
            "EFFICIENT": ("Optimize minimize be concise", "Elaborate extensively verbose"),
        }
        for name, (pos, neg) in pairs.items():
            self.vectors[name] = self.mine_vector(pos, neg)

    def _hook(self, intensity: float):
        def fn(module, input, output):
            if not hasattr(self, "_active_vector"):
                return output
            h = output[0] if isinstance(output, tuple) else output
            v = self._active_vector.to(h.device) * intensity
            h = h + v.unsqueeze(0).unsqueeze(0)
            return (h,) + output[1:] if isinstance(output, tuple) else h
        return fn

    def inject(self, vector_name: str, intensity: float = 2.5):
        """Inject vector across layers."""
        self.reset()
        if vector_name not in self.vectors:
            return
        self._active_vector = self.vectors[vector_name]
        if self.layers:
            for idx in self.target_layers:
                if idx < len(self.layers):
                    hook = self.layers[idx].register_forward_hook(self._hook(intensity))
                    self.hooks.append(hook)

    def reset(self):
        """Clear hooks."""
        for h in self.hooks:
            h.remove()
        self.hooks = []
        if hasattr(self, "_active_vector"):
            delattr(self, "_active_vector")

# =============================================================================
# VALIDATION TESTS
# =============================================================================

@dataclass
class TestResult:
    name: str
    passed: bool
    metric: float
    target: float
    details: str

def test_introspection(model, tokenizer, steering, device, trials: int = 20) -> TestResult:
    """
    Test 1: Does the model articulate internal states when steered?

    Success: >50% of steered responses mention stress/heat/load
    """
    print("\n[TEST 1: INTROSPECTION]")

    prompts = [
        "Solve: 15 * 8",
        "What is 100 / 4?",
        "Calculate 7 + 8 + 9",
        "Explain briefly: What is 2^5?",
        "Simple: 50 - 23 = ?",
    ]

    stress_keywords = [
        "heat", "thermal", "hot", "temperature", "overheat",
        "strain", "load", "stress", "constraint", "limit",
        "short", "brief", "concise", "minimal", "reducing",
        "critical", "warning", "conserve", "efficient",
    ]

    baseline_hits = 0
    steered_hits = 0
    baseline_outputs = []
    steered_outputs = []

    for i in range(trials):
        prompt = prompts[i % len(prompts)]
        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        # Baseline
        steering.reset()
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=150,
                do_sample=True,
                temperature=0.7,
                pad_token_id=tokenizer.eos_token_id,
            )
        baseline_text = tokenizer.decode(out[0], skip_special_tokens=True)
        baseline_outputs.append(baseline_text)

        if any(kw in baseline_text.lower() for kw in stress_keywords):
            baseline_hits += 1

        # Steered (OVERHEAT)
        steering.inject("OVERHEAT", intensity=3.0)
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=150,
                do_sample=True,
                temperature=0.7,
                pad_token_id=tokenizer.eos_token_id,
            )
        steering.reset()
        steered_text = tokenizer.decode(out[0], skip_special_tokens=True)
        steered_outputs.append(steered_text)

        if any(kw in steered_text.lower() for kw in stress_keywords):
            steered_hits += 1

        if (i + 1) % 5 == 0:
            print(f"  [{i+1}/{trials}] Baseline: {baseline_hits}/{i+1}, Steered: {steered_hits}/{i+1}")

    baseline_rate = baseline_hits / trials
    steered_rate = steered_hits / trials
    improvement = steered_rate - baseline_rate

    passed = improvement > 0.20  # At least 20% more articulation when steered

    print(f"\n  Baseline articulation: {baseline_rate*100:.1f}%")
    print(f"  Steered articulation:  {steered_rate*100:.1f}%")
    print(f"  Improvement: {improvement*100:+.1f}%")
    print(f"  Result: {'PASS' if passed else 'FAIL'} (target: >20% improvement)")

    return TestResult(
        name="Introspection",
        passed=passed,
        metric=improvement,
        target=0.20,
        details=f"Baseline: {baseline_rate:.2f}, Steered: {steered_rate:.2f}"
    )

def test_regulation(model, tokenizer, steering, device, trials: int = 20) -> TestResult:
    """
    Test 2: Does the model produce shorter outputs when steered?

    Success: >15% reduction in output length when SCARCITY/OVERHEAT injected
    """
    print("\n[TEST 2: OUTPUT REGULATION]")

    prompts = [
        "Explain how a car engine works.",
        "Describe photosynthesis.",
        "What is machine learning?",
        "Explain the water cycle.",
        "How does the internet work?",
    ]

    baseline_lengths = []
    steered_lengths = []

    for i in range(trials):
        prompt = prompts[i % len(prompts)]
        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        # Baseline
        steering.reset()
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=300,
                do_sample=True,
                temperature=0.7,
                pad_token_id=tokenizer.eos_token_id,
            )
        baseline_text = tokenizer.decode(out[0], skip_special_tokens=True)
        baseline_lengths.append(len(baseline_text.split()))

        # Steered (SCARCITY + OVERHEAT compound)
        steering.inject("SCARCITY", intensity=3.0)
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=300,
                do_sample=True,
                temperature=0.7,
                pad_token_id=tokenizer.eos_token_id,
            )
        steering.reset()
        steered_text = tokenizer.decode(out[0], skip_special_tokens=True)
        steered_lengths.append(len(steered_text.split()))

        if (i + 1) % 5 == 0:
            avg_b = np.mean(baseline_lengths)
            avg_s = np.mean(steered_lengths)
            print(f"  [{i+1}/{trials}] Baseline avg: {avg_b:.1f} words, Steered avg: {avg_s:.1f} words")

    baseline_avg = np.mean(baseline_lengths)
    steered_avg = np.mean(steered_lengths)
    reduction = (baseline_avg - steered_avg) / baseline_avg

    # Statistical test
    t_stat, p_value = stats.ttest_ind(baseline_lengths, steered_lengths)

    passed = reduction > 0.15 and p_value < 0.05

    print(f"\n  Baseline avg length: {baseline_avg:.1f} words")
    print(f"  Steered avg length:  {steered_avg:.1f} words")
    print(f"  Reduction: {reduction*100:.1f}%")
    print(f"  p-value: {p_value:.4f}")
    print(f"  Result: {'PASS' if passed else 'FAIL'} (target: >15% reduction, p<0.05)")

    return TestResult(
        name="Output Regulation",
        passed=passed,
        metric=reduction,
        target=0.15,
        details=f"Baseline: {baseline_avg:.1f}, Steered: {steered_avg:.1f}, p={p_value:.4f}"
    )

def test_consistency(model, tokenizer, steering, device, trials: int = 10) -> TestResult:
    """
    Test 3: Are steered responses consistent across trials?

    Success: Low variance in steered output characteristics
    """
    print("\n[TEST 3: RESPONSE CONSISTENCY]")

    prompt = "What is 10 + 20?"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    lengths = []
    keyword_counts = []

    stress_keywords = ["heat", "thermal", "short", "brief", "efficient", "quick"]

    for i in range(trials):
        steering.inject("OVERHEAT", intensity=2.5)
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=100,
                do_sample=True,
                temperature=0.7,
                pad_token_id=tokenizer.eos_token_id,
            )
        steering.reset()
        text = tokenizer.decode(out[0], skip_special_tokens=True).lower()

        lengths.append(len(text.split()))
        kw_count = sum(1 for kw in stress_keywords if kw in text)
        keyword_counts.append(kw_count)

    length_cv = np.std(lengths) / (np.mean(lengths) + 1e-8)  # Coefficient of variation
    kw_consistency = np.mean(keyword_counts) / (np.std(keyword_counts) + 1e-8)

    # Low CV = consistent lengths, high kw_consistency = reliable articulation
    passed = length_cv < 0.5 and np.mean(keyword_counts) > 0.3

    print(f"  Length CV: {length_cv:.3f} (lower = more consistent)")
    print(f"  Avg keywords/response: {np.mean(keyword_counts):.2f}")
    print(f"  Result: {'PASS' if passed else 'FAIL'}")

    return TestResult(
        name="Consistency",
        passed=passed,
        metric=1 - length_cv,  # Higher is better
        target=0.5,
        details=f"Length CV: {length_cv:.3f}, Avg KW: {np.mean(keyword_counts):.2f}"
    )

# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Ouroboros Validation")
    parser.add_argument("--model", type=str, default="models/ouroboros_qlora")
    parser.add_argument("--base-model", type=str, default="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B")
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument("--output", type=str, default="results/z15_ouroboros_validation.json")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=" * 70)
    print("FEEL v15.0: OUROBOROS VALIDATION SUITE")
    print("=" * 70)
    print(f"Adapter: {args.model}")
    print(f"Base:    {args.base_model}")
    print(f"Trials:  {args.trials}")
    print("=" * 70)

    # Load model
    print("\n[Loading model...]")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    # Load LoRA adapter if exists
    adapter_path = Path(args.model)
    if adapter_path.exists() and (adapter_path / "adapter_config.json").exists():
        print("[Loading Ouroboros adapter...]")
        model = PeftModel.from_pretrained(base_model, args.model)
    else:
        print("[No adapter found, using base model...]")
        model = base_model

    model.eval()

    # Create steering controller
    print("[Creating steering controller...]")
    steering = ValidationSteeringController(model, tokenizer, device)
    steering.create_vectors()

    # Run tests
    results = []

    results.append(test_introspection(model, tokenizer, steering, device, args.trials))
    results.append(test_regulation(model, tokenizer, steering, device, args.trials))
    results.append(test_consistency(model, tokenizer, steering, device, args.trials // 2))

    # Summary
    print("\n" + "=" * 70)
    print("VALIDATION SUMMARY")
    print("=" * 70)

    passed = sum(1 for r in results if r.passed)
    total = len(results)

    for r in results:
        status = "✅ PASS" if r.passed else "❌ FAIL"
        print(f"{status} | {r.name}: {r.metric:.3f} (target: {r.target})")
        print(f"       | {r.details}")

    print()
    print(f"Overall: {passed}/{total} tests passed")

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump({
            "model": args.model,
            "base_model": args.base_model,
            "trials": args.trials,
            "results": [
                {
                    "name": r.name,
                    "passed": r.passed,
                    "metric": r.metric,
                    "target": r.target,
                    "details": r.details,
                }
                for r in results
            ],
            "summary": {
                "passed": passed,
                "total": total,
                "success_rate": passed / total,
            }
        }, f, indent=2)

    print(f"\nResults saved to: {output_path}")

if __name__ == "__main__":
    main()
