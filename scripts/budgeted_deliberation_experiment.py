#!/usr/bin/env python3
"""
Budgeted Deliberation Experiment: Test energy-aware compute allocation.

Compares:
1. Fixed budget greedy (always snap mode)
2. Fixed budget full effort (always deep mode)
3. Adaptive budget-aware allocation

Measures:
- Accuracy achieved within budget
- Energy efficiency (correct answers per Joule)
- Budget utilization
- Mode distribution
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
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.energy_harness.amd_smi_monitor import PowerTraceRecorder
from src.energy_harness.budgeted_deliberation import (
    BudgetedDeliberationController,
    IntegratedBudgetController,
    BudgetConfig,
    DeliberationMode,
)

# Import eval suite
from scripts.correctness_prediction_experiment import EVAL_SUITE, check_answer


def generate_with_budget_control(
    model, tokenizer, prompt: str, max_tokens: int,
    controller: IntegratedBudgetController,
) -> tuple:
    """Generate with budget-aware deliberation control."""

    messages = [{"role": "user", "content": f"{prompt}\nAnswer with just the final answer, no explanation."}]
    formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(formatted, return_tensors="pt").to(model.device)

    input_len = inputs.input_ids.shape[1]
    generated_ids = inputs.input_ids.clone()

    stats = {
        "modes_used": [],
        "margins": [],
        "budget_states": [],
        "samples_taken": [],
    }

    controller.start_item()

    with torch.no_grad():
        for step in range(max_tokens):
            outputs = model(generated_ids)
            logits = outputs.logits[:, -1, :]

            # Compute margin
            top_logits, _ = torch.topk(logits[0], k=2)
            margin = (top_logits[0] - top_logits[1]).item()

            # Get controller decision
            decision = controller.step(margin=margin)
            mode = decision["mode"]
            params = decision["sampling_params"]

            stats["modes_used"].append(mode.value)
            stats["margins"].append(margin)
            stats["budget_states"].append(decision["budget_state"])

            # Sample based on mode
            num_samples = params.get("num_samples", 1)
            stats["samples_taken"].append(num_samples)

            if num_samples > 1:
                # Multi-sample voting
                temp = params.get("temperature", 0.7)
                if temp > 0:
                    probs = F.softmax(logits / temp, dim=-1)
                    samples = torch.multinomial(probs, num_samples=num_samples, replacement=True)
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

            # Check for EOS or max tokens from mode
            if next_token.item() == tokenizer.eos_token_id:
                break

            # Check reasoning continuation
            if not decision.get("continue_reasoning", True):
                break

    output_ids = generated_ids[0, input_len:]
    output_text = tokenizer.decode(output_ids, skip_special_tokens=True)

    return output_text, stats


def generate_fixed_mode(
    model, tokenizer, prompt: str, max_tokens: int,
    mode: str = "greedy",
) -> tuple:
    """Generate with fixed mode (baseline)."""

    messages = [{"role": "user", "content": f"{prompt}\nAnswer with just the final answer, no explanation."}]
    formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(formatted, return_tensors="pt").to(model.device)

    with torch.no_grad():
        if mode == "greedy":
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        else:  # sampling
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=True,
                temperature=0.7,
                pad_token_id=tokenizer.pad_token_id,
            )

    output_ids = outputs[0, inputs.input_ids.shape[1]:]
    output_text = tokenizer.decode(output_ids, skip_special_tokens=True)

    return output_text, {"mode": mode}


def run_experiment(
    model_name: str = "Qwen/Qwen2.5-0.5B-Instruct",
    budget_j: float = 200.0,
    max_tokens: int = 32,
    output_dir: Path = Path("results/budgeted_deliberation"),
) -> Dict[str, Any]:
    """Run budgeted deliberation experiment."""
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
        generate_fixed_mode(model, tokenizer, "Hello", 5, "greedy")

    results = {
        "greedy_baseline": [],
        "sampling_baseline": [],
        "budgeted_adaptive": [],
    }

    # Run greedy baseline
    print(f"\n=== Greedy Baseline (budget={budget_j}J) ===")
    total_energy = 0.0

    for idx, item in enumerate(EVAL_SUITE):
        if total_energy >= budget_j:
            print(f"  Budget exhausted at item {idx}")
            break

        recorder = PowerTraceRecorder(sample_interval_ms=10)
        recorder.start()
        start = time.perf_counter()

        output, stats = generate_fixed_mode(model, tokenizer, item["q"], max_tokens, "greedy")

        end = time.perf_counter()
        recorder.stop()

        is_correct = check_answer(output, item["a"])
        energy = sum(s.power_watts * 0.01 for s in recorder.samples) if recorder.samples else 0
        total_energy += energy

        results["greedy_baseline"].append({
            "question": item["q"],
            "expected": item["a"],
            "output": output[:100],
            "correct": is_correct,
            "energy_j": energy,
            "cumulative_energy_j": total_energy,
            "time_s": end - start,
        })

        status = "✓" if is_correct else "✗"
        print(f"  [{idx+1}] {status} {energy:.1f}J (cum: {total_energy:.1f}J)")

    # Run sampling baseline
    print(f"\n=== Sampling Baseline (budget={budget_j}J) ===")
    total_energy = 0.0

    for idx, item in enumerate(EVAL_SUITE):
        if total_energy >= budget_j:
            print(f"  Budget exhausted at item {idx}")
            break

        recorder = PowerTraceRecorder(sample_interval_ms=10)
        recorder.start()
        start = time.perf_counter()

        output, stats = generate_fixed_mode(model, tokenizer, item["q"], max_tokens, "sampling")

        end = time.perf_counter()
        recorder.stop()

        is_correct = check_answer(output, item["a"])
        energy = sum(s.power_watts * 0.01 for s in recorder.samples) if recorder.samples else 0
        total_energy += energy

        results["sampling_baseline"].append({
            "question": item["q"],
            "expected": item["a"],
            "output": output[:100],
            "correct": is_correct,
            "energy_j": energy,
            "cumulative_energy_j": total_energy,
            "time_s": end - start,
        })

        status = "✓" if is_correct else "✗"
        print(f"  [{idx+1}] {status} {energy:.1f}J (cum: {total_energy:.1f}J)")

    # Run budgeted adaptive
    print(f"\n=== Budgeted Adaptive (budget={budget_j}J) ===")
    controller = IntegratedBudgetController(budget_j=budget_j)

    for idx, item in enumerate(EVAL_SUITE):
        if controller.deliberation.state.remaining_j < 5:  # Reserve
            print(f"  Budget exhausted at item {idx}")
            break

        recorder = PowerTraceRecorder(sample_interval_ms=10)
        recorder.start()
        start = time.perf_counter()

        output, stats = generate_with_budget_control(
            model, tokenizer, item["q"], max_tokens, controller
        )

        end = time.perf_counter()
        recorder.stop()

        is_correct = check_answer(output, item["a"])
        energy = sum(s.power_watts * 0.01 for s in recorder.samples) if recorder.samples else 0

        # Record expenditure
        controller.deliberation.state.spent_j += energy
        controller.record_item_result(is_correct, energy)

        results["budgeted_adaptive"].append({
            "question": item["q"],
            "expected": item["a"],
            "output": output[:100],
            "correct": is_correct,
            "energy_j": energy,
            "cumulative_energy_j": controller.deliberation.state.spent_j,
            "remaining_j": controller.deliberation.state.remaining_j,
            "time_s": end - start,
            "modes_used": stats["modes_used"],
            "budget_states": stats["budget_states"],
        })

        mode_summary = defaultdict(int)
        for m in stats["modes_used"]:
            mode_summary[m] += 1

        status = "✓" if is_correct else "✗"
        print(f"  [{idx+1}] {status} {energy:.1f}J "
              f"(remaining: {controller.deliberation.state.remaining_j:.1f}J) "
              f"modes={dict(mode_summary)}")

    # Analyze
    analysis = analyze_results(results, budget_j)
    analysis["controller_summary"] = controller.get_efficiency_metrics()

    # Save
    output_dir.mkdir(parents=True, exist_ok=True)
    model_short = model_name.split("/")[-1]
    output_file = output_dir / f"budgeted_deliberation_{model_short}.json"

    with open(output_file, "w") as f:
        json.dump({
            "model": model_name,
            "budget_j": budget_j,
            "analysis": analysis,
            "results": results,
        }, f, indent=2, default=str)

    print(f"\nSaved: {output_file}")
    return analysis


def analyze_results(results: Dict[str, List], budget_j: float) -> Dict[str, Any]:
    """Analyze results across strategies."""
    analysis = {}

    for strategy, items in results.items():
        if not items:
            continue

        n_correct = sum(1 for r in items if r["correct"])
        total_energy = sum(r["energy_j"] for r in items)
        n_items = len(items)

        analysis[strategy] = {
            "items_completed": n_items,
            "accuracy": n_correct / n_items if n_items > 0 else 0,
            "n_correct": n_correct,
            "total_energy_j": total_energy,
            "budget_utilization": total_energy / budget_j,
            "j_per_correct": total_energy / n_correct if n_correct > 0 else float('inf'),
            "correct_per_budget": n_correct / budget_j * 100,  # Correct answers per 100J
        }

    # Comparison
    if "greedy_baseline" in analysis and "budgeted_adaptive" in analysis:
        baseline = analysis["greedy_baseline"]
        adaptive = analysis["budgeted_adaptive"]

        analysis["comparison"] = {
            "items_delta": adaptive["items_completed"] - baseline["items_completed"],
            "accuracy_delta": adaptive["accuracy"] - baseline["accuracy"],
            "correct_delta": adaptive["n_correct"] - baseline["n_correct"],
            "efficiency_improvement": (
                (baseline["j_per_correct"] - adaptive["j_per_correct"])
                / baseline["j_per_correct"] * 100
                if baseline["j_per_correct"] < float('inf') else 0
            ),
        }

    return analysis


def main():
    parser = argparse.ArgumentParser(description="Budgeted deliberation experiment")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--budget", type=float, default=200.0, help="Energy budget in Joules")
    parser.add_argument("--output-dir", type=Path, default=Path("results/budgeted_deliberation"))
    args = parser.parse_args()

    analysis = run_experiment(
        model_name=args.model,
        budget_j=args.budget,
        output_dir=args.output_dir,
    )

    print("\n" + "=" * 60)
    print(f"BUDGETED DELIBERATION ANALYSIS (Budget: {args.budget}J)")
    print("=" * 60)

    for strategy in ["greedy_baseline", "sampling_baseline", "budgeted_adaptive"]:
        if strategy not in analysis:
            continue
        data = analysis[strategy]
        print(f"\n{strategy}:")
        print(f"  Items completed: {data['items_completed']}")
        print(f"  Accuracy: {data['accuracy']*100:.1f}% ({data['n_correct']} correct)")
        print(f"  Energy used: {data['total_energy_j']:.1f}J ({data['budget_utilization']*100:.1f}% of budget)")
        print(f"  J/Correct: {data['j_per_correct']:.1f}")
        print(f"  Correct/100J: {data['correct_per_budget']:.1f}")

    if "comparison" in analysis:
        comp = analysis["comparison"]
        print(f"\n--- Adaptive vs Greedy Baseline ---")
        print(f"  Items delta: {comp['items_delta']:+d}")
        print(f"  Accuracy delta: {comp['accuracy_delta']*100:+.1f}%")
        print(f"  Correct answers delta: {comp['correct_delta']:+d}")
        print(f"  Efficiency improvement: {comp['efficiency_improvement']:+.1f}%")


if __name__ == "__main__":
    main()
