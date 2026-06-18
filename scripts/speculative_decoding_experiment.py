#!/usr/bin/env python3
"""
Speculative Decoding Experiment: Test adaptive draft length.

Compares:
1. Standard greedy decoding
2. Fixed draft length speculative decoding
3. Adaptive draft length speculative decoding

Measures:
- Tokens per second
- Energy per token
- Acceptance rate
- Draft length adaptation behavior
"""

import json
import time
import argparse
from pathlib import Path
from typing import Dict, Any, List
from collections import defaultdict

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.energy_harness.amd_smi_monitor import PowerTraceRecorder
from src.energy_harness.speculative_decoding import (
    AdaptiveSpeculativeDecoder,
    SpeculativeConfig,
)

# Test prompts of varying difficulty
TEST_PROMPTS = [
    # Easy - predictable patterns
    {"prompt": "Count from 1 to 10:", "expected_tokens": 32, "difficulty": "easy"},
    {"prompt": "The capital of France is", "expected_tokens": 16, "difficulty": "easy"},
    {"prompt": "List the days of the week:", "expected_tokens": 32, "difficulty": "easy"},

    # Medium - some reasoning
    {"prompt": "Explain why the sky is blue in one sentence:", "expected_tokens": 48, "difficulty": "medium"},
    {"prompt": "What is 25 * 4?", "expected_tokens": 16, "difficulty": "medium"},
    {"prompt": "Name three programming languages:", "expected_tokens": 24, "difficulty": "medium"},

    # Hard - complex reasoning
    {"prompt": "Solve step by step: If a train travels 60 miles in 1 hour, how far in 2.5 hours?", "expected_tokens": 64, "difficulty": "hard"},
    {"prompt": "Write a Python function to check if a number is prime:", "expected_tokens": 96, "difficulty": "hard"},
    {"prompt": "Explain the difference between supervised and unsupervised learning:", "expected_tokens": 96, "difficulty": "hard"},
]


def run_standard_generation(
    model, tokenizer, prompt: str, max_tokens: int
) -> Dict[str, Any]:
    """Standard autoregressive generation baseline."""
    messages = [{"role": "user", "content": prompt}]
    formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(formatted, return_tensors="pt").to(model.device)
    input_len = inputs.input_ids.shape[1]

    recorder = PowerTraceRecorder(sample_interval_ms=10)
    recorder.start()
    start = time.perf_counter()

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )

    end = time.perf_counter()
    recorder.stop()

    output_ids = outputs[0, input_len:]
    output_text = tokenizer.decode(output_ids, skip_special_tokens=True)

    energy = sum(s.power_watts * 0.01 for s in recorder.samples) if recorder.samples else 0
    n_tokens = len(output_ids)

    return {
        "method": "standard",
        "output": output_text[:100],
        "n_tokens": n_tokens,
        "time_s": end - start,
        "energy_j": energy,
        "tokens_per_second": n_tokens / (end - start) if end > start else 0,
        "joules_per_token": energy / n_tokens if n_tokens > 0 else 0,
    }


def run_speculative_generation(
    model, tokenizer, prompt: str, max_tokens: int,
    decoder: AdaptiveSpeculativeDecoder,
) -> Dict[str, Any]:
    """Speculative decoding with adaptive draft length."""
    messages = [{"role": "user", "content": prompt}]
    formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    recorder = PowerTraceRecorder(sample_interval_ms=10)
    recorder.start()
    start = time.perf_counter()

    output_text, stats = decoder.generate(model, tokenizer, formatted, max_tokens)

    end = time.perf_counter()
    recorder.stop()

    energy = sum(s.power_watts * 0.01 for s in recorder.samples) if recorder.samples else 0

    return {
        "method": "speculative",
        "output": output_text[:100],
        "n_tokens": stats.total_tokens,
        "time_s": end - start,
        "energy_j": energy,
        "tokens_per_second": stats.total_tokens / (end - start) if end > start else 0,
        "joules_per_token": energy / stats.total_tokens if stats.total_tokens > 0 else 0,
        "stats": stats.to_dict(),
    }


def run_experiment(
    model_name: str = "Qwen/Qwen2.5-0.5B-Instruct",
    output_dir: Path = Path("results/speculative_decoding"),
) -> Dict[str, Any]:
    """Run speculative decoding comparison experiment."""
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
        _ = run_standard_generation(model, tokenizer, "Hello", 8)

    # Initialize speculative decoder
    spec_config = SpeculativeConfig(
        initial_draft_length=4,
        min_draft_length=1,
        max_draft_length=8,
        target_acceptance_rate=0.7,
    )
    decoder = AdaptiveSpeculativeDecoder(spec_config)

    results = {
        "model": model_name,
        "config": {
            "initial_draft_length": spec_config.initial_draft_length,
            "min_draft_length": spec_config.min_draft_length,
            "max_draft_length": spec_config.max_draft_length,
            "target_acceptance_rate": spec_config.target_acceptance_rate,
        },
        "standard": [],
        "speculative": [],
    }

    print("\n=== Running Standard Generation ===")
    for idx, item in enumerate(TEST_PROMPTS):
        result = run_standard_generation(
            model, tokenizer, item["prompt"], item["expected_tokens"]
        )
        result["difficulty"] = item["difficulty"]
        result["prompt"] = item["prompt"][:50]
        results["standard"].append(result)
        print(f"  [{idx+1}/{len(TEST_PROMPTS)}] {item['difficulty']}: "
              f"{result['tokens_per_second']:.1f} tok/s, "
              f"{result['joules_per_token']:.2f} J/tok")

    print("\n=== Running Speculative Generation ===")
    for idx, item in enumerate(TEST_PROMPTS):
        result = run_speculative_generation(
            model, tokenizer, item["prompt"], item["expected_tokens"], decoder
        )
        result["difficulty"] = item["difficulty"]
        result["prompt"] = item["prompt"][:50]
        results["speculative"].append(result)
        stats = result.get("stats", {})
        print(f"  [{idx+1}/{len(TEST_PROMPTS)}] {item['difficulty']}: "
              f"{result['tokens_per_second']:.1f} tok/s, "
              f"{result['joules_per_token']:.2f} J/tok, "
              f"accept={stats.get('overall_acceptance_rate', 0):.2f}, "
              f"draft={stats.get('avg_draft_length', 0):.1f}")

    # Analyze
    analysis = analyze_results(results)
    results["analysis"] = analysis

    # Save
    output_dir.mkdir(parents=True, exist_ok=True)
    model_short = model_name.split("/")[-1]
    output_file = output_dir / f"speculative_decoding_{model_short}.json"

    with open(output_file, "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\nSaved: {output_file}")
    return results


def analyze_results(results: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze and compare standard vs speculative."""
    analysis = {}

    for method in ["standard", "speculative"]:
        items = results[method]
        if not items:
            continue

        total_tokens = sum(r["n_tokens"] for r in items)
        total_time = sum(r["time_s"] for r in items)
        total_energy = sum(r["energy_j"] for r in items)

        analysis[method] = {
            "total_tokens": total_tokens,
            "total_time_s": total_time,
            "total_energy_j": total_energy,
            "avg_tokens_per_second": total_tokens / total_time if total_time > 0 else 0,
            "avg_joules_per_token": total_energy / total_tokens if total_tokens > 0 else 0,
        }

        # By difficulty
        by_diff = defaultdict(list)
        for r in items:
            by_diff[r["difficulty"]].append(r)

        analysis[method]["by_difficulty"] = {
            diff: {
                "avg_tokens_per_second": sum(r["tokens_per_second"] for r in rs) / len(rs),
                "avg_joules_per_token": sum(r["joules_per_token"] for r in rs) / len(rs),
            }
            for diff, rs in by_diff.items()
        }

    # Comparison
    if "standard" in analysis and "speculative" in analysis:
        std = analysis["standard"]
        spec = analysis["speculative"]

        analysis["comparison"] = {
            "speedup": spec["avg_tokens_per_second"] / std["avg_tokens_per_second"]
                       if std["avg_tokens_per_second"] > 0 else 1.0,
            "energy_ratio": spec["avg_joules_per_token"] / std["avg_joules_per_token"]
                           if std["avg_joules_per_token"] > 0 else 1.0,
        }

        # Speculative-specific stats
        spec_items = results["speculative"]
        if spec_items and "stats" in spec_items[0]:
            avg_acceptance = sum(
                r["stats"].get("overall_acceptance_rate", 0) for r in spec_items
            ) / len(spec_items)
            avg_speedup = sum(
                r["stats"].get("speedup_ratio", 1) for r in spec_items
            ) / len(spec_items)

            analysis["speculative_stats"] = {
                "avg_acceptance_rate": avg_acceptance,
                "avg_theoretical_speedup": avg_speedup,
            }

    return analysis


def main():
    parser = argparse.ArgumentParser(description="Speculative decoding experiment")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--output-dir", type=Path, default=Path("results/speculative_decoding"))
    args = parser.parse_args()

    results = run_experiment(
        model_name=args.model,
        output_dir=args.output_dir,
    )

    print("\n" + "=" * 60)
    print("SPECULATIVE DECODING ANALYSIS")
    print("=" * 60)

    analysis = results.get("analysis", {})

    for method in ["standard", "speculative"]:
        if method not in analysis:
            continue
        data = analysis[method]
        print(f"\n{method}:")
        print(f"  Tokens/sec: {data['avg_tokens_per_second']:.1f}")
        print(f"  J/token: {data['avg_joules_per_token']:.3f}")
        print(f"  Total time: {data['total_time_s']:.1f}s")
        print(f"  Total energy: {data['total_energy_j']:.1f}J")

    if "comparison" in analysis:
        comp = analysis["comparison"]
        print(f"\n--- Comparison ---")
        print(f"  Speedup: {comp['speedup']:.2f}x")
        print(f"  Energy ratio: {comp['energy_ratio']:.2f}x")

    if "speculative_stats" in analysis:
        spec = analysis["speculative_stats"]
        print(f"\n--- Speculative Stats ---")
        print(f"  Avg acceptance rate: {spec['avg_acceptance_rate']:.2f}")
        print(f"  Theoretical speedup: {spec['avg_theoretical_speedup']:.2f}x")


if __name__ == "__main__":
    main()
