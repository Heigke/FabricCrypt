#!/usr/bin/env python3
"""
Adaptive Compute Experiment: Test margin-based compute allocation.

Tests whether allocating more compute to uncertain tokens improves
accuracy without proportionally increasing energy.

Goal: Achieve higher accuracy than baseline at similar energy,
or same accuracy at lower energy.
"""

import json
import re
import sys
import time
import argparse
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.energy_harness.amd_smi_monitor import PowerTraceRecorder
from src.energy_harness.adaptive_compute_controller import (
    UnifiedAdaptiveController, ControllerConfig, ComputeMode,
    compute_energy_savings
)

# Import eval suite from correctness prediction
from scripts.correctness_prediction_experiment import EVAL_SUITE, check_answer


def generate_with_adaptive_control(
    model, tokenizer, prompt: str, max_tokens: int,
    controller: UnifiedAdaptiveController,
    baseline_temp: float = 0.3,
) -> Tuple[str, Dict[str, Any]]:
    """Generate with adaptive compute control."""

    messages = [{"role": "user", "content": f"{prompt}\nAnswer with just the final answer, no explanation."}]
    formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(formatted, return_tensors="pt").to(model.device)

    input_len = inputs.input_ids.shape[1]
    generated_ids = inputs.input_ids.clone()

    generation_stats = {
        "modes_used": [],
        "margins": [],
        "samples_taken": [],
    }

    with torch.no_grad():
        for step in range(max_tokens):
            # Forward pass
            outputs = model(generated_ids)
            logits = outputs.logits[:, -1, :]

            # Compute margin
            top_logits, _ = torch.topk(logits[0], k=2)
            margin = (top_logits[0] - top_logits[1]).item()

            # Get adaptive decision
            decision = controller.step(margin=margin)
            mode = decision["mode"]
            params = decision["sampling_params"]

            generation_stats["modes_used"].append(mode.value)
            generation_stats["margins"].append(margin)

            # Sample based on mode
            num_samples = params.get("num_samples", 1)
            generation_stats["samples_taken"].append(num_samples)

            if num_samples > 1:
                # Multi-sample voting
                temp = params.get("temperature", 0.7)
                if temp > 0:
                    probs = F.softmax(logits / temp, dim=-1)
                    samples = torch.multinomial(probs, num_samples=num_samples, replacement=True)
                    # Majority vote
                    next_token = samples.mode().values.unsqueeze(0)
                else:
                    next_token = logits.argmax(dim=-1, keepdim=True)
            else:
                # Single sample
                temp = params.get("temperature", 0)
                if temp > 0 and params.get("do_sample", False):
                    probs = F.softmax(logits / temp, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                else:
                    next_token = logits.argmax(dim=-1, keepdim=True)

            generated_ids = torch.cat([generated_ids, next_token], dim=1)

            if next_token.item() == tokenizer.eos_token_id:
                break

    # Decode
    output_ids = generated_ids[0, input_len:]
    output_text = tokenizer.decode(output_ids, skip_special_tokens=True)

    return output_text, generation_stats


def generate_baseline(
    model, tokenizer, prompt: str, max_tokens: int, temperature: float = 0.0
) -> Tuple[str, Dict[str, Any]]:
    """Generate with fixed baseline strategy."""

    messages = [{"role": "user", "content": f"{prompt}\nAnswer with just the final answer, no explanation."}]
    formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(formatted, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=(temperature > 0),
            temperature=temperature if temperature > 0 else None,
            pad_token_id=tokenizer.pad_token_id,
        )

    output_ids = outputs[0, inputs.input_ids.shape[1]:]
    output_text = tokenizer.decode(output_ids, skip_special_tokens=True)

    return output_text, {"mode": "baseline", "temperature": temperature}


def run_adaptive_experiment(
    model_name: str = "Qwen/Qwen2.5-0.5B-Instruct",
    max_tokens: int = 32,
    output_dir: Path = Path("results/adaptive_compute"),
) -> Dict[str, Any]:
    """
    Run adaptive compute experiment comparing:
    1. Baseline greedy (T=0.0)
    2. Baseline sampling (T=0.7)
    3. Adaptive margin-based control
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

    # Warmup
    print("Warming up...")
    for _ in range(3):
        generate_baseline(model, tokenizer, "Hello", 5, 0.0)

    results = {
        "baseline_greedy": [],
        "baseline_sample": [],
        "adaptive": [],
    }

    # Run experiments
    strategies = [
        ("baseline_greedy", lambda p: generate_baseline(model, tokenizer, p, max_tokens, 0.0)),
        ("baseline_sample", lambda p: generate_baseline(model, tokenizer, p, max_tokens, 0.7)),
    ]

    for strategy_name, gen_fn in strategies:
        print(f"\n=== {strategy_name} ===")

        for idx, item in enumerate(EVAL_SUITE):
            recorder = PowerTraceRecorder(sample_interval_ms=10)
            recorder.start()

            start = time.perf_counter()
            output, stats = gen_fn(item["q"])
            end = time.perf_counter()

            recorder.stop()

            is_correct = check_answer(output, item["a"])
            energy = sum(s.power_watts * 0.01 for s in recorder.samples) if recorder.samples else 0

            results[strategy_name].append({
                "question": item["q"],
                "expected": item["a"],
                "output": output[:100],
                "correct": is_correct,
                "energy_j": energy,
                "time_s": end - start,
                "category": item["cat"],
                "difficulty": item["diff"],
            })

            status = "✓" if is_correct else "✗"
            print(f"  [{idx+1}/{len(EVAL_SUITE)}] {status} {item['cat']}/{item['diff']}: {energy:.1f}J")

    # Adaptive strategy
    print(f"\n=== adaptive ===")
    controller = UnifiedAdaptiveController(ControllerConfig())

    for idx, item in enumerate(EVAL_SUITE):
        recorder = PowerTraceRecorder(sample_interval_ms=10)
        recorder.start()

        start = time.perf_counter()
        output, stats = generate_with_adaptive_control(
            model, tokenizer, item["q"], max_tokens, controller
        )
        end = time.perf_counter()

        recorder.stop()

        is_correct = check_answer(output, item["a"])
        energy = sum(s.power_watts * 0.01 for s in recorder.samples) if recorder.samples else 0

        results["adaptive"].append({
            "question": item["q"],
            "expected": item["a"],
            "output": output[:100],
            "correct": is_correct,
            "energy_j": energy,
            "time_s": end - start,
            "category": item["cat"],
            "difficulty": item["diff"],
            "modes_used": stats["modes_used"],
            "samples_taken": stats["samples_taken"],
        })

        status = "✓" if is_correct else "✗"
        mode_summary = defaultdict(int)
        for m in stats["modes_used"]:
            mode_summary[m] += 1
        print(f"  [{idx+1}/{len(EVAL_SUITE)}] {status} {item['cat']}/{item['diff']}: {energy:.1f}J, modes={dict(mode_summary)}")

    # Analyze results
    analysis = analyze_results(results)
    analysis["controller_summary"] = controller.get_summary()

    # Save
    output_dir.mkdir(parents=True, exist_ok=True)
    model_short = model_name.split("/")[-1]
    output_file = output_dir / f"adaptive_compute_{model_short}.json"

    with open(output_file, "w") as f:
        json.dump({
            "model": model_name,
            "n_items": len(EVAL_SUITE),
            "analysis": analysis,
            "results": results,
        }, f, indent=2, default=str)

    print(f"\nSaved: {output_file}")
    return analysis


def analyze_results(results: Dict[str, List[Dict]]) -> Dict[str, Any]:
    """Analyze and compare strategies."""
    analysis = {}

    for strategy, items in results.items():
        n_correct = sum(1 for r in items if r["correct"])
        total_energy = sum(r["energy_j"] for r in items)
        total_time = sum(r["time_s"] for r in items)

        analysis[strategy] = {
            "accuracy": n_correct / len(items),
            "n_correct": n_correct,
            "n_total": len(items),
            "total_energy_j": total_energy,
            "avg_energy_j": total_energy / len(items),
            "total_time_s": total_time,
            "joules_per_correct": total_energy / n_correct if n_correct > 0 else float('inf'),
        }

        # By category
        by_cat = defaultdict(list)
        for r in items:
            by_cat[r["category"]].append(r)

        analysis[strategy]["by_category"] = {
            cat: sum(1 for r in rs if r["correct"]) / len(rs)
            for cat, rs in by_cat.items()
        }

    # Comparisons
    if "baseline_greedy" in analysis and "adaptive" in analysis:
        baseline = analysis["baseline_greedy"]
        adaptive = analysis["adaptive"]

        analysis["comparison"] = {
            "accuracy_delta": adaptive["accuracy"] - baseline["accuracy"],
            "energy_delta_pct": (adaptive["total_energy_j"] - baseline["total_energy_j"]) / baseline["total_energy_j"] * 100,
            "jpc_improvement_pct": (baseline["joules_per_correct"] - adaptive["joules_per_correct"]) / baseline["joules_per_correct"] * 100 if baseline["joules_per_correct"] < float('inf') else 0,
        }

    return analysis


def main():
    parser = argparse.ArgumentParser(description="Adaptive compute experiment")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--output-dir", type=Path, default=Path("results/adaptive_compute"))
    args = parser.parse_args()

    analysis = run_adaptive_experiment(
        model_name=args.model,
        output_dir=args.output_dir,
    )

    print("\n" + "="*60)
    print("ADAPTIVE COMPUTE ANALYSIS")
    print("="*60)

    for strategy in ["baseline_greedy", "baseline_sample", "adaptive"]:
        if strategy not in analysis:
            continue
        data = analysis[strategy]
        print(f"\n{strategy}:")
        print(f"  Accuracy: {data['accuracy']*100:.1f}% ({data['n_correct']}/{data['n_total']})")
        print(f"  Energy: {data['total_energy_j']:.1f}J total, {data['avg_energy_j']:.1f}J avg")
        print(f"  J/Correct: {data['joules_per_correct']:.1f}")

    if "comparison" in analysis:
        comp = analysis["comparison"]
        print(f"\n--- Adaptive vs Baseline Greedy ---")
        print(f"  Accuracy delta: {comp['accuracy_delta']*100:+.1f}%")
        print(f"  Energy delta: {comp['energy_delta_pct']:+.1f}%")
        print(f"  J/Correct improvement: {comp['jpc_improvement_pct']:+.1f}%")


if __name__ == "__main__":
    main()
