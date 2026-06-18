#!/usr/bin/env python3
"""
Proprioceptive Conditioning Experiment.

Tests whether FiLM-based proprioceptive conditioning affects model behavior:
1. Baseline: No conditioning
2. Normal state: Cool GPU, low power
3. Stressed state: Hot GPU, high power, near throttling

We measure:
- Generation quality (accuracy on eval suite)
- Token latency
- Energy consumption
- Conditioning statistics
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
from src.energy_harness.proprioceptive_conditioning import (
    ProprioceptiveState,
    ProprioceptiveConditioner,
    ProprioceptiveHook,
    ProprioceptiveMonitor,
    create_proprioceptive_system,
)

# Import eval suite from correctness prediction
from scripts.correctness_prediction_experiment import EVAL_SUITE, check_answer


def generate_with_conditioning(
    model, tokenizer, prompt: str, max_tokens: int,
    conditioner: ProprioceptiveConditioner,
    hook: ProprioceptiveHook,
    state: ProprioceptiveState,
) -> tuple:
    """Generate with proprioceptive conditioning active."""

    # Update conditioning
    conditioner.update(state, model.device)

    messages = [{"role": "user", "content": f"{prompt}\nAnswer with just the final answer, no explanation."}]
    formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(formatted, return_tensors="pt").to(model.device)

    input_len = inputs.input_ids.shape[1]
    generated_ids = inputs.input_ids.clone()

    stats = {"margins": [], "scales": [], "shifts": []}

    with torch.no_grad():
        for step in range(max_tokens):
            outputs = model(generated_ids)
            logits = outputs.logits[:, -1, :]

            # Compute margin for state update
            top_logits, _ = torch.topk(logits[0], k=2)
            margin = (top_logits[0] - top_logits[1]).item()
            stats["margins"].append(margin)

            # Update cognitive state
            state.margin_ema = 0.3 * margin + 0.7 * state.margin_ema
            state.tokens_generated = step + 1
            conditioner.update(state, model.device)

            # Track conditioning
            layer_scale, layer_shift = conditioner.get_modulation(0)
            stats["scales"].append(layer_scale)
            stats["shifts"].append(layer_shift)

            # Greedy selection
            next_token = logits.argmax(dim=-1, keepdim=True)
            generated_ids = torch.cat([generated_ids, next_token], dim=1)

            if next_token.item() == tokenizer.eos_token_id:
                break

    output_ids = generated_ids[0, input_len:]
    output_text = tokenizer.decode(output_ids, skip_special_tokens=True)

    return output_text, stats


def generate_baseline(model, tokenizer, prompt: str, max_tokens: int) -> tuple:
    """Generate without conditioning (baseline)."""
    messages = [{"role": "user", "content": f"{prompt}\nAnswer with just the final answer, no explanation."}]
    formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(formatted, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )

    output_ids = outputs[0, inputs.input_ids.shape[1]:]
    output_text = tokenizer.decode(output_ids, skip_special_tokens=True)

    return output_text, {"mode": "baseline"}


def run_experiment(
    model_name: str = "Qwen/Qwen2.5-0.5B-Instruct",
    max_tokens: int = 32,
    output_dir: Path = Path("results/proprioceptive"),
) -> Dict[str, Any]:
    """Run proprioceptive conditioning experiment."""
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

    # Create proprioceptive system
    print("Creating proprioceptive conditioning system...")
    conditioner, hook = create_proprioceptive_system(model)
    n_hooks = hook.attach()
    print(f"  Attached {n_hooks} layer hooks")

    # Warmup
    print("Warming up...")
    for _ in range(3):
        generate_baseline(model, tokenizer, "Hello", 5)

    # Define test conditions
    conditions = {
        "baseline": None,  # No conditioning
        "normal": ProprioceptiveState(
            temperature_c=50.0,
            power_w=40.0,
            power_budget_w=80.0,
            throttling=False,
            margin_ema=5.0,
        ),
        "stressed": ProprioceptiveState(
            temperature_c=85.0,
            power_w=75.0,
            power_budget_w=80.0,
            throttling=True,
            margin_ema=2.0,
        ),
        "cold_confident": ProprioceptiveState(
            temperature_c=35.0,
            power_w=25.0,
            power_budget_w=80.0,
            throttling=False,
            margin_ema=8.0,
        ),
    }

    results = {cond: [] for cond in conditions}

    for cond_name, state in conditions.items():
        print(f"\n=== {cond_name} ===")

        if state is None:
            # Baseline - disable hooks
            hook.disable()
        else:
            hook.enable()

        for idx, item in enumerate(EVAL_SUITE[:24]):  # Subset for speed
            recorder = PowerTraceRecorder(sample_interval_ms=10)
            recorder.start()
            start = time.perf_counter()

            if state is None:
                output, stats = generate_baseline(model, tokenizer, item["q"], max_tokens)
            else:
                output, stats = generate_with_conditioning(
                    model, tokenizer, item["q"], max_tokens,
                    conditioner, hook, state
                )

            end = time.perf_counter()
            recorder.stop()

            is_correct = check_answer(output, item["a"])
            energy = sum(s.power_watts * 0.01 for s in recorder.samples) if recorder.samples else 0

            results[cond_name].append({
                "question": item["q"],
                "expected": item["a"],
                "output": output[:100],
                "correct": is_correct,
                "energy_j": energy,
                "time_s": end - start,
                "category": item["cat"],
                "difficulty": item["diff"],
                "stats": stats if state is not None else None,
            })

            status = "✓" if is_correct else "✗"
            print(f"  [{idx+1}/24] {status} {item['cat']}/{item['diff']}: {energy:.1f}J")

    # Cleanup
    hook.detach()

    # Analyze
    analysis = analyze_results(results)
    analysis["conditioner_stats"] = conditioner.get_stats()

    # Save
    output_dir.mkdir(parents=True, exist_ok=True)
    model_short = model_name.split("/")[-1]
    output_file = output_dir / f"proprioceptive_{model_short}.json"

    with open(output_file, "w") as f:
        json.dump({
            "model": model_name,
            "n_hooks": n_hooks,
            "conditions": list(conditions.keys()),
            "analysis": analysis,
            "results": results,
        }, f, indent=2, default=str)

    print(f"\nSaved: {output_file}")
    return analysis


def analyze_results(results: Dict[str, List[Dict]]) -> Dict[str, Any]:
    """Analyze and compare conditions."""
    analysis = {}

    for condition, items in results.items():
        if not items:
            continue

        n_correct = sum(1 for r in items if r["correct"])
        total_energy = sum(r["energy_j"] for r in items)
        total_time = sum(r["time_s"] for r in items)

        analysis[condition] = {
            "accuracy": n_correct / len(items),
            "n_correct": n_correct,
            "n_total": len(items),
            "total_energy_j": total_energy,
            "avg_energy_j": total_energy / len(items),
            "total_time_s": total_time,
        }

        # Conditioning stats
        if items[0].get("stats") and items[0]["stats"].get("scales"):
            all_scales = []
            all_shifts = []
            for r in items:
                if r.get("stats"):
                    all_scales.extend(r["stats"].get("scales", []))
                    all_shifts.extend(r["stats"].get("shifts", []))

            if all_scales:
                analysis[condition]["avg_scale"] = sum(all_scales) / len(all_scales)
                analysis[condition]["avg_shift"] = sum(all_shifts) / len(all_shifts)

    # Comparison vs baseline
    if "baseline" in analysis:
        baseline = analysis["baseline"]
        for cond in analysis:
            if cond == "baseline":
                continue
            analysis[cond]["accuracy_delta"] = analysis[cond]["accuracy"] - baseline["accuracy"]
            analysis[cond]["energy_delta_pct"] = (
                (analysis[cond]["total_energy_j"] - baseline["total_energy_j"])
                / baseline["total_energy_j"] * 100
            )

    return analysis


def main():
    parser = argparse.ArgumentParser(description="Proprioceptive conditioning experiment")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--output-dir", type=Path, default=Path("results/proprioceptive"))
    args = parser.parse_args()

    analysis = run_experiment(
        model_name=args.model,
        output_dir=args.output_dir,
    )

    print("\n" + "=" * 60)
    print("PROPRIOCEPTIVE CONDITIONING ANALYSIS")
    print("=" * 60)

    for condition in ["baseline", "normal", "stressed", "cold_confident"]:
        if condition not in analysis:
            continue
        data = analysis[condition]
        print(f"\n{condition}:")
        print(f"  Accuracy: {data['accuracy']*100:.1f}% ({data['n_correct']}/{data['n_total']})")
        print(f"  Energy: {data['total_energy_j']:.1f}J total, {data['avg_energy_j']:.1f}J avg")
        if "avg_scale" in data:
            print(f"  Avg scale: {data['avg_scale']:.4f}, Avg shift: {data['avg_shift']:.4f}")
        if "accuracy_delta" in data:
            print(f"  Δ vs baseline: {data['accuracy_delta']*100:+.1f}% accuracy, "
                  f"{data['energy_delta_pct']:+.1f}% energy")


if __name__ == "__main__":
    main()
