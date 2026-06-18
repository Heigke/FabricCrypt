#!/usr/bin/env python3
"""
Thermal Constraint Pareto Experiment.

Maps the accuracy vs energy Pareto frontier under different thermal/power constraints:
1. Unconstrained (let GPU do what it wants)
2. Power-capped (artificial power limit)
3. Thermal-aware (back off when hot)

This validates the core research claim:
"Conditioning LLM compute allocation on physical state improves quality-at-fixed-Joules"
"""

import json
import time
import argparse
from pathlib import Path
from typing import Dict, Any, List
from collections import defaultdict
from dataclasses import dataclass

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.energy_harness.amd_smi_monitor import PowerTraceRecorder
from src.energy_harness.adaptive_compute_controller import (
    UnifiedAdaptiveController, ControllerConfig, ComputeMode
)
from src.energy_harness.budgeted_deliberation import (
    BudgetedDeliberationController, BudgetConfig
)

from scripts.correctness_prediction_experiment import EVAL_SUITE, check_answer


@dataclass
class ThermalConstraint:
    """Thermal constraint configuration."""
    name: str
    power_limit_w: float = 100.0      # Soft power limit
    temp_threshold_c: float = 85.0     # Thermal throttle threshold
    budget_j: float = 300.0           # Energy budget
    description: str = ""


# Define constraint scenarios
CONSTRAINTS = [
    ThermalConstraint(
        name="unconstrained",
        power_limit_w=150.0,
        temp_threshold_c=95.0,
        budget_j=500.0,
        description="No constraints - let GPU run freely",
    ),
    ThermalConstraint(
        name="moderate_power",
        power_limit_w=80.0,
        temp_threshold_c=85.0,
        budget_j=300.0,
        description="80W power cap, 85°C thermal limit",
    ),
    ThermalConstraint(
        name="tight_power",
        power_limit_w=50.0,
        temp_threshold_c=75.0,
        budget_j=200.0,
        description="50W power cap, 75°C thermal limit",
    ),
    ThermalConstraint(
        name="thermal_stress",
        power_limit_w=100.0,
        temp_threshold_c=70.0,
        budget_j=250.0,
        description="Conservative 70°C limit - thermal priority",
    ),
]


def generate_with_thermal_awareness(
    model, tokenizer, prompt: str, max_tokens: int,
    controller: UnifiedAdaptiveController,
    constraint: ThermalConstraint,
    recorder: PowerTraceRecorder,
) -> tuple:
    """Generate with thermal-aware compute allocation."""

    messages = [{"role": "user", "content": f"{prompt}\nAnswer with just the final answer, no explanation."}]
    formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(formatted, return_tensors="pt").to(model.device)

    input_len = inputs.input_ids.shape[1]
    generated_ids = inputs.input_ids.clone()

    stats = {
        "modes_used": [],
        "margins": [],
        "temperatures": [],
        "powers": [],
    }

    with torch.no_grad():
        for step in range(max_tokens):
            # Get current thermal state from recorder
            if recorder.samples:
                current_power = recorder.samples[-1].power_watts
                # Estimate temperature (placeholder - would come from GPU)
                current_temp = min(85, 50 + current_power * 0.3)
            else:
                current_power = 50.0
                current_temp = 55.0

            stats["temperatures"].append(current_temp)
            stats["powers"].append(current_power)

            outputs = model(generated_ids)
            logits = outputs.logits[:, -1, :]

            # Compute margin
            top_logits, _ = torch.topk(logits[0], k=2)
            margin = (top_logits[0] - top_logits[1]).item()
            stats["margins"].append(margin)

            # Check thermal constraints
            throttling = (
                current_temp > constraint.temp_threshold_c or
                current_power > constraint.power_limit_w
            )

            # Get adaptive decision with thermal awareness
            decision = controller.step(
                margin=margin,
                gpu_temp=current_temp,
                gpu_power=current_power,
                throttling=throttling,
            )
            mode = decision["mode"]
            params = decision["sampling_params"]

            stats["modes_used"].append(mode.value)

            # Sample based on mode
            num_samples = params.get("num_samples", 1)

            if num_samples > 1:
                temp = params.get("temperature", 0.7)
                if temp > 0:
                    probs = F.softmax(logits / temp, dim=-1)
                    samples = torch.multinomial(probs, num_samples=num_samples, replacement=True)
                    next_token = samples.mode().values.unsqueeze(0)
                else:
                    next_token = logits.argmax(dim=-1, keepdim=True)
            else:
                temp = params.get("temperature", 0)
                if temp > 0 and params.get("do_sample", False):
                    probs = F.softmax(logits / temp, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                else:
                    next_token = logits.argmax(dim=-1, keepdim=True)

            generated_ids = torch.cat([generated_ids, next_token], dim=1)

            if next_token.item() == tokenizer.eos_token_id:
                break

    output_ids = generated_ids[0, input_len:]
    output_text = tokenizer.decode(output_ids, skip_special_tokens=True)

    return output_text, stats


def generate_baseline(model, tokenizer, prompt: str, max_tokens: int) -> tuple:
    """Standard greedy generation."""
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
    output_dir: Path = Path("results/thermal_pareto"),
) -> Dict[str, Any]:
    """Run thermal constraint Pareto experiment."""
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
        generate_baseline(model, tokenizer, "Hello", 5)

    all_results = {}
    pareto_points = []

    for constraint in CONSTRAINTS:
        print(f"\n{'='*60}")
        print(f"Constraint: {constraint.name}")
        print(f"  {constraint.description}")
        print(f"  Power limit: {constraint.power_limit_w}W")
        print(f"  Temp threshold: {constraint.temp_threshold_c}°C")
        print(f"  Budget: {constraint.budget_j}J")
        print("="*60)

        results = {"baseline": [], "thermal_aware": []}

        # Baseline run
        print("\n--- Baseline ---")
        total_energy = 0.0

        for idx, item in enumerate(EVAL_SUITE):
            if total_energy >= constraint.budget_j:
                print(f"  Budget exhausted at item {idx}")
                break

            recorder = PowerTraceRecorder(sample_interval_ms=10)
            recorder.start()
            start = time.perf_counter()

            output, stats = generate_baseline(model, tokenizer, item["q"], max_tokens)

            end = time.perf_counter()
            recorder.stop()

            is_correct = check_answer(output, item["a"])
            energy = sum(s.power_watts * 0.01 for s in recorder.samples) if recorder.samples else 0
            total_energy += energy

            results["baseline"].append({
                "correct": is_correct,
                "energy_j": energy,
                "cumulative_energy_j": total_energy,
            })

        baseline_correct = sum(1 for r in results["baseline"] if r["correct"])
        baseline_energy = sum(r["energy_j"] for r in results["baseline"])
        print(f"  Completed: {len(results['baseline'])} items")
        print(f"  Correct: {baseline_correct}")
        print(f"  Energy: {baseline_energy:.1f}J")

        # Thermal-aware run
        print("\n--- Thermal-Aware ---")
        controller = UnifiedAdaptiveController(ControllerConfig(
            temp_throttle_c=constraint.temp_threshold_c,
            power_budget_w=constraint.power_limit_w,
        ))
        total_energy = 0.0

        for idx, item in enumerate(EVAL_SUITE):
            if total_energy >= constraint.budget_j:
                print(f"  Budget exhausted at item {idx}")
                break

            recorder = PowerTraceRecorder(sample_interval_ms=10)
            recorder.start()
            start = time.perf_counter()

            output, stats = generate_with_thermal_awareness(
                model, tokenizer, item["q"], max_tokens,
                controller, constraint, recorder
            )

            end = time.perf_counter()
            recorder.stop()

            is_correct = check_answer(output, item["a"])
            energy = sum(s.power_watts * 0.01 for s in recorder.samples) if recorder.samples else 0
            total_energy += energy

            results["thermal_aware"].append({
                "correct": is_correct,
                "energy_j": energy,
                "cumulative_energy_j": total_energy,
                "modes_used": stats.get("modes_used", []),
            })

        thermal_correct = sum(1 for r in results["thermal_aware"] if r["correct"])
        thermal_energy = sum(r["energy_j"] for r in results["thermal_aware"])
        print(f"  Completed: {len(results['thermal_aware'])} items")
        print(f"  Correct: {thermal_correct}")
        print(f"  Energy: {thermal_energy:.1f}J")

        # Record for Pareto analysis
        pareto_points.append({
            "constraint": constraint.name,
            "baseline_correct": baseline_correct,
            "baseline_energy": baseline_energy,
            "thermal_correct": thermal_correct,
            "thermal_energy": thermal_energy,
            "baseline_efficiency": baseline_correct / baseline_energy if baseline_energy > 0 else 0,
            "thermal_efficiency": thermal_correct / thermal_energy if thermal_energy > 0 else 0,
        })

        all_results[constraint.name] = results

    # Analyze Pareto frontier
    analysis = analyze_pareto(pareto_points)

    # Save
    output_dir.mkdir(parents=True, exist_ok=True)
    model_short = model_name.split("/")[-1]
    output_file = output_dir / f"thermal_pareto_{model_short}.json"

    with open(output_file, "w") as f:
        json.dump({
            "model": model_name,
            "constraints": [c.__dict__ for c in CONSTRAINTS],
            "pareto_points": pareto_points,
            "analysis": analysis,
            "results": all_results,
        }, f, indent=2, default=str)

    print(f"\nSaved: {output_file}")
    return analysis


def analyze_pareto(points: List[Dict]) -> Dict[str, Any]:
    """Analyze Pareto frontier."""
    analysis = {
        "n_constraints": len(points),
        "pareto_frontier": [],
        "summary": {},
    }

    # Find Pareto-optimal points (maximize correct, minimize energy)
    pareto_frontier = []
    for p in points:
        is_dominated = False
        for q in points:
            if q == p:
                continue
            # q dominates p if q has more correct AND less energy (for thermal)
            if (q["thermal_correct"] >= p["thermal_correct"] and
                q["thermal_energy"] <= p["thermal_energy"] and
                (q["thermal_correct"] > p["thermal_correct"] or
                 q["thermal_energy"] < p["thermal_energy"])):
                is_dominated = True
                break
        if not is_dominated:
            pareto_frontier.append(p)

    analysis["pareto_frontier"] = [p["constraint"] for p in pareto_frontier]

    # Summary statistics
    baseline_efficiencies = [p["baseline_efficiency"] for p in points]
    thermal_efficiencies = [p["thermal_efficiency"] for p in points]

    analysis["summary"] = {
        "avg_baseline_efficiency": sum(baseline_efficiencies) / len(baseline_efficiencies),
        "avg_thermal_efficiency": sum(thermal_efficiencies) / len(thermal_efficiencies),
        "efficiency_improvement": (
            (sum(thermal_efficiencies) - sum(baseline_efficiencies)) /
            sum(baseline_efficiencies) * 100
            if sum(baseline_efficiencies) > 0 else 0
        ),
    }

    return analysis


def main():
    parser = argparse.ArgumentParser(description="Thermal constraint Pareto experiment")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--output-dir", type=Path, default=Path("results/thermal_pareto"))
    args = parser.parse_args()

    analysis = run_experiment(
        model_name=args.model,
        output_dir=args.output_dir,
    )

    print("\n" + "=" * 60)
    print("THERMAL PARETO ANALYSIS")
    print("=" * 60)

    print(f"\nPareto-optimal constraints: {analysis['pareto_frontier']}")
    print(f"\nSummary:")
    print(f"  Avg baseline efficiency: {analysis['summary']['avg_baseline_efficiency']:.4f} correct/J")
    print(f"  Avg thermal efficiency: {analysis['summary']['avg_thermal_efficiency']:.4f} correct/J")
    print(f"  Efficiency improvement: {analysis['summary']['efficiency_improvement']:+.1f}%")


if __name__ == "__main__":
    main()
