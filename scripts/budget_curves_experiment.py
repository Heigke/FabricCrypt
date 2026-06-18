#!/usr/bin/env python3
"""
Budget Curves Experiment: Quality vs Energy Pareto Frontier

Tests whether energy allocation affects output quality on objective tasks.
Uses tasks with verifiable correctness to build quality-vs-Joules curves.

Key Research Question:
Since latent signals don't predict TPOT spikes (hardware noise dominates),
can we show that compute effort affects OUTPUT QUALITY on reasoning tasks?
"""

import json
import re
import sys
import time
import argparse
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.energy_harness.amd_smi_monitor import PowerTraceRecorder


# === OBJECTIVE TASKS WITH VERIFIABLE ANSWERS ===

MATH_TASKS = [
    # Simple arithmetic - expect high accuracy
    {"prompt": "What is 7 + 8?", "answer": "15", "difficulty": "easy"},
    {"prompt": "What is 23 * 4?", "answer": "92", "difficulty": "easy"},
    {"prompt": "What is 144 / 12?", "answer": "12", "difficulty": "easy"},
    {"prompt": "What is 56 - 29?", "answer": "27", "difficulty": "easy"},

    # Medium arithmetic
    {"prompt": "What is 17 * 13?", "answer": "221", "difficulty": "medium"},
    {"prompt": "What is 256 / 16?", "answer": "16", "difficulty": "medium"},
    {"prompt": "What is 15 + 27 + 38?", "answer": "80", "difficulty": "medium"},
    {"prompt": "What is (5 + 3) * 7?", "answer": "56", "difficulty": "medium"},

    # Harder - multi-step
    {"prompt": "If x = 5, what is 3x + 7?", "answer": "22", "difficulty": "hard"},
    {"prompt": "What is 25% of 80?", "answer": "20", "difficulty": "hard"},
    {"prompt": "What is 12 squared?", "answer": "144", "difficulty": "hard"},
    {"prompt": "What is the average of 10, 20, and 30?", "answer": "20", "difficulty": "hard"},
]

QA_TASKS = [
    # Factual with clear answers
    {"prompt": "What is the capital of France?", "answer": "Paris", "difficulty": "easy"},
    {"prompt": "How many days are in a week?", "answer": "7", "difficulty": "easy"},
    {"prompt": "What planet is closest to the Sun?", "answer": "Mercury", "difficulty": "easy"},
    {"prompt": "What is the chemical symbol for water?", "answer": "H2O", "difficulty": "easy"},

    # Medium factual
    {"prompt": "Who wrote Romeo and Juliet?", "answer": "Shakespeare", "difficulty": "medium"},
    {"prompt": "What is the largest ocean?", "answer": "Pacific", "difficulty": "medium"},
    {"prompt": "How many continents are there?", "answer": "7", "difficulty": "medium"},
    {"prompt": "What gas do plants absorb from the atmosphere?", "answer": "CO2", "difficulty": "medium"},
]

CODE_TASKS = [
    # Simple code understanding
    {"prompt": "What does this Python code print? print(2 + 3)", "answer": "5", "difficulty": "easy"},
    {"prompt": "What does this code return? len([1, 2, 3])", "answer": "3", "difficulty": "easy"},
    {"prompt": "What does this evaluate to? 10 > 5", "answer": "True", "difficulty": "easy"},
    {"prompt": "What does this code print? print('hello'.upper())", "answer": "HELLO", "difficulty": "medium"},
]


def check_answer(output: str, expected: str) -> bool:
    """Check if output contains the expected answer."""
    # Normalize both strings
    output_lower = output.lower().strip()
    expected_lower = expected.lower().strip()

    # Direct containment check
    if expected_lower in output_lower:
        return True

    # Check for number formats (15 vs "fifteen" etc.)
    # Extract all numbers from output
    numbers = re.findall(r'\b\d+(?:\.\d+)?\b', output)
    if expected_lower in [n.lower() for n in numbers]:
        return True

    # For boolean answers
    if expected_lower in ["true", "false"]:
        return expected_lower in output_lower

    return False


@dataclass
class BudgetExperimentResult:
    """Result from a single task under a specific budget/policy."""
    task_prompt: str
    task_category: str
    task_difficulty: str
    expected_answer: str
    model_output: str
    is_correct: bool
    energy_j: float
    time_s: float
    output_tokens: int
    temperature: float
    max_new_tokens: int  # Budget constraint


def run_budget_experiment(
    model_name: str = "Qwen/Qwen2.5-0.5B-Instruct",
    temperatures: List[float] = [0.0, 0.3, 0.7, 1.0],
    token_budgets: List[int] = [16, 32, 64, 128],
    tasks: Optional[List[Dict]] = None,
    output_dir: Path = Path("results/budget_curves"),
) -> Dict[str, Any]:
    """
    Run budget curves experiment varying token budget and temperature.

    Measures:
    - Correctness rate (quality metric)
    - Energy consumption (efficiency metric)
    - Time (latency metric)
    """
    if tasks is None:
        tasks = MATH_TASKS + QA_TASKS + CODE_TASKS

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

    results: List[BudgetExperimentResult] = []

    # Warmup
    print("Warming up...")
    warmup_input = tokenizer("Hello", return_tensors="pt").to(model.device)
    for _ in range(3):
        model.generate(**warmup_input, max_new_tokens=10, do_sample=False)
    torch.cuda.synchronize()

    total_runs = len(temperatures) * len(token_budgets) * len(tasks)
    run_idx = 0

    for temp in temperatures:
        for budget in token_budgets:
            print(f"\n=== Temperature {temp}, Budget {budget} tokens ===")

            for task in tasks:
                run_idx += 1
                prompt = task["prompt"]
                expected = task["answer"]
                category = "math" if task in MATH_TASKS else ("qa" if task in QA_TASKS else "code")
                difficulty = task["difficulty"]

                # Format prompt for instruction model
                messages = [{"role": "user", "content": f"{prompt}\nAnswer with just the final answer, no explanation."}]
                formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                inputs = tokenizer(formatted, return_tensors="pt").to(model.device)

                # Run with energy measurement
                recorder = PowerTraceRecorder(sample_interval_ms=10)
                recorder.start()

                torch.cuda.synchronize()
                start = time.perf_counter()

                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=budget,
                        do_sample=(temp > 0),
                        temperature=temp if temp > 0 else None,
                        pad_token_id=tokenizer.pad_token_id,
                    )

                torch.cuda.synchronize()
                end = time.perf_counter()

                recorder.stop()

                # Decode output
                output_ids = outputs[0][inputs.input_ids.shape[1]:]
                output_text = tokenizer.decode(output_ids, skip_special_tokens=True)
                output_tokens = len(output_ids)

                # Check correctness
                is_correct = check_answer(output_text, expected)

                # Get energy from samples
                samples = recorder.samples
                if samples:
                    total_energy = sum(s.power_watts * 0.01 for s in samples)  # 10ms intervals -> J
                else:
                    total_energy = 0
                total_time = end - start

                result = BudgetExperimentResult(
                    task_prompt=prompt,
                    task_category=category,
                    task_difficulty=difficulty,
                    expected_answer=expected,
                    model_output=output_text[:200],  # Truncate for storage
                    is_correct=is_correct,
                    energy_j=total_energy,
                    time_s=total_time,
                    output_tokens=output_tokens,
                    temperature=temp,
                    max_new_tokens=budget,
                )
                results.append(result)

                status = "✓" if is_correct else "✗"
                print(f"  [{run_idx}/{total_runs}] {status} {category}/{difficulty}: {total_energy:.1f}J, {output_tokens} tokens")

    # Aggregate results
    aggregated = aggregate_results(results)

    # Save results
    output_dir.mkdir(parents=True, exist_ok=True)
    model_short = model_name.split("/")[-1]
    output_file = output_dir / f"budget_curves_{model_short}.json"

    with open(output_file, "w") as f:
        json.dump({
            "model": model_name,
            "temperatures": temperatures,
            "token_budgets": token_budgets,
            "n_tasks": len(tasks),
            "aggregated": aggregated,
            "results": [
                {
                    "prompt": r.task_prompt,
                    "category": r.task_category,
                    "difficulty": r.task_difficulty,
                    "expected": r.expected_answer,
                    "output": r.model_output,
                    "correct": r.is_correct,
                    "energy_j": r.energy_j,
                    "time_s": r.time_s,
                    "tokens": r.output_tokens,
                    "temperature": r.temperature,
                    "budget": r.max_new_tokens,
                }
                for r in results
            ]
        }, f, indent=2)

    print(f"\nSaved: {output_file}")
    return aggregated


def aggregate_results(results: List[BudgetExperimentResult]) -> Dict[str, Any]:
    """Aggregate results by temperature and budget."""
    from collections import defaultdict
    import statistics

    # Group by (temperature, budget)
    groups = defaultdict(list)
    for r in results:
        key = (r.temperature, r.max_new_tokens)
        groups[key].append(r)

    aggregated = {}
    for (temp, budget), group in groups.items():
        key = f"temp_{temp}_budget_{budget}"

        correct_count = sum(1 for r in group if r.is_correct)
        total_count = len(group)
        accuracy = correct_count / total_count if total_count > 0 else 0

        energies = [r.energy_j for r in group]
        times = [r.time_s for r in group]

        aggregated[key] = {
            "temperature": temp,
            "budget": budget,
            "accuracy": accuracy,
            "n_correct": correct_count,
            "n_total": total_count,
            "energy_mean_j": statistics.mean(energies),
            "energy_std_j": statistics.stdev(energies) if len(energies) > 1 else 0,
            "time_mean_s": statistics.mean(times),
            "joules_per_correct": sum(energies) / correct_count if correct_count > 0 else float('inf'),
        }

    # Also aggregate by category
    by_category = defaultdict(list)
    for r in results:
        by_category[r.task_category].append(r)

    aggregated["by_category"] = {}
    for cat, group in by_category.items():
        correct = sum(1 for r in group if r.is_correct)
        total = len(group)
        aggregated["by_category"][cat] = {
            "accuracy": correct / total if total > 0 else 0,
            "n_correct": correct,
            "n_total": total,
            "energy_mean_j": statistics.mean(r.energy_j for r in group),
        }

    return aggregated


def main():
    parser = argparse.ArgumentParser(description="Budget curves experiment")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--output-dir", type=Path, default=Path("results/budget_curves"))
    parser.add_argument("--quick", action="store_true", help="Quick run with fewer configs")
    args = parser.parse_args()

    if args.quick:
        temperatures = [0.0, 0.7]
        token_budgets = [16, 64]
    else:
        temperatures = [0.0, 0.3, 0.7, 1.0]
        token_budgets = [16, 32, 64, 128]

    results = run_budget_experiment(
        model_name=args.model,
        temperatures=temperatures,
        token_budgets=token_budgets,
        output_dir=args.output_dir,
    )

    print("\n" + "="*60)
    print("BUDGET CURVES SUMMARY")
    print("="*60)

    # Print accuracy vs energy trade-off
    configs = [(k, v) for k, v in results.items() if k.startswith("temp_")]
    configs.sort(key=lambda x: (x[1]["temperature"], x[1]["budget"]))

    print(f"\n{'Config':<25} {'Accuracy':>10} {'Energy(J)':>12} {'J/Correct':>12}")
    print("-" * 60)
    for key, data in configs:
        acc = f"{data['accuracy']*100:.1f}%"
        energy = f"{data['energy_mean_j']:.1f}"
        jpc = f"{data['joules_per_correct']:.1f}" if data['joules_per_correct'] < 1000 else "inf"
        print(f"{key:<25} {acc:>10} {energy:>12} {jpc:>12}")

    print("\n--- By Category ---")
    for cat, data in results.get("by_category", {}).items():
        print(f"  {cat}: {data['accuracy']*100:.1f}% ({data['n_correct']}/{data['n_total']})")


if __name__ == "__main__":
    main()
