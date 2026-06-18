#!/usr/bin/env python3
"""
Energy-as-Token-Budget: Hardware-Derived Dynamic Token Budgets

Novel Contribution:
-------------------
Token-budget-aware reasoning (ACL 2025) works, but budgets are typically
static or heuristic. We ground the budget in REAL hardware telemetry:

  remaining_budget_tokens = remaining_joules / J_per_token

Where J_per_token comes from live power measurements.

The model/policy can then:
1. See its "metabolic budget" in real-time
2. Decide: shorter answer, early stop, verify, or abstain
3. Adapt reasoning style based on energy constraints

This creates "organism-like regulation" with hard measurements.

Key hypothesis: Energy-aware token budgeting improves J/correct under
constrained budgets, especially vs fixed-token baselines.
"""

import json
import time
import argparse
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass, asdict
from collections import defaultdict
import math

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.energy_harness.amd_smi_monitor import PowerTraceRecorder
from scripts.eval_suite import EVAL_SUITE_EXPANDED, check_answer

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


@dataclass
class TokenBudgetState:
    """Current token budget state derived from hardware."""
    total_budget_j: float
    spent_j: float
    remaining_j: float
    j_per_token: float  # Rolling average
    remaining_tokens: int  # Estimated
    budget_fraction: float
    is_critical: bool  # < 20% budget remaining


class EnergyBudgetController:
    """Manage token budgets derived from energy measurements."""

    def __init__(self, total_budget_j: float, reserve_fraction: float = 0.1):
        self.total_budget_j = total_budget_j
        self.reserve_fraction = reserve_fraction
        self.spent_j = 0.0
        self.token_energies = []  # J per token history
        self.window_size = 100  # Rolling window for J/token

    def update(self, energy_j: float, tokens: int):
        """Update after generating tokens."""
        self.spent_j += energy_j
        if tokens > 0:
            j_per_token = energy_j / tokens
            self.token_energies.append(j_per_token)
            # Keep rolling window
            if len(self.token_energies) > self.window_size:
                self.token_energies = self.token_energies[-self.window_size:]

    def get_j_per_token(self) -> float:
        """Get current J/token estimate."""
        if not self.token_energies:
            return 0.1  # Default estimate
        return sum(self.token_energies) / len(self.token_energies)

    def get_state(self) -> TokenBudgetState:
        """Get current budget state."""
        remaining_j = max(0, self.total_budget_j - self.spent_j)
        j_per_token = self.get_j_per_token()
        remaining_tokens = int(remaining_j / j_per_token) if j_per_token > 0 else 1000
        budget_fraction = remaining_j / self.total_budget_j

        return TokenBudgetState(
            total_budget_j=self.total_budget_j,
            spent_j=self.spent_j,
            remaining_j=remaining_j,
            j_per_token=j_per_token,
            remaining_tokens=remaining_tokens,
            budget_fraction=budget_fraction,
            is_critical=budget_fraction < self.reserve_fraction,
        )

    def get_max_tokens(self, base_max: int = 64) -> int:
        """Get max tokens for next generation based on budget."""
        state = self.get_state()

        if state.is_critical:
            # Critical: minimal output
            return min(8, state.remaining_tokens // 2)
        elif state.budget_fraction < 0.3:
            # Low: reduce to half
            return min(base_max // 2, state.remaining_tokens // 2)
        elif state.budget_fraction < 0.5:
            # Medium: slight reduction
            return min(int(base_max * 0.75), state.remaining_tokens // 2)
        else:
            # Healthy: full tokens
            return min(base_max, state.remaining_tokens // 2)

    def should_abstain(self, p_error: float, min_tokens_needed: int = 16) -> bool:
        """Decide if we should abstain from answering."""
        state = self.get_state()

        # Abstain if:
        # 1. Very low budget AND high uncertainty
        # 2. Can't fit minimum response
        if state.remaining_tokens < min_tokens_needed:
            return True

        if state.is_critical and p_error > 0.5:
            return True

        return False


def generate_with_budget(
    model, tokenizer, prompt: str,
    budget_controller: EnergyBudgetController,
    p_error: float = 0.5,
    base_max_tokens: int = 64,
) -> Tuple[str, float, int, bool]:
    """Generate with energy-aware token budget."""

    # Check if we should abstain
    if budget_controller.should_abstain(p_error):
        return "[ABSTAIN: insufficient budget]", 0.0, 0, True

    # Get dynamic max tokens
    max_tokens = budget_controller.get_max_tokens(base_max_tokens)
    state = budget_controller.get_state()

    # Construct prompt with budget awareness
    budget_hint = ""
    if state.budget_fraction < 0.3:
        budget_hint = " Answer briefly."
    elif state.budget_fraction < 0.5:
        budget_hint = " Be concise."

    messages = [{"role": "user", "content": f"{prompt}{budget_hint}"}]
    formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(formatted, return_tensors="pt").to(model.device)

    input_len = inputs.input_ids.shape[1]

    recorder = PowerTraceRecorder(sample_interval_ms=10)
    recorder.start()

    with torch.no_grad():
        outputs = model.generate(
            inputs.input_ids,
            max_new_tokens=max_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )

    recorder.stop()

    output_ids = outputs[0, input_len:]
    output_text = tokenizer.decode(output_ids, skip_special_tokens=True)
    n_tokens = len(output_ids)

    energy = sum(s.power_watts * 0.01 for s in recorder.samples) if recorder.samples else 0.1

    # Update controller
    budget_controller.update(energy, n_tokens)

    return output_text, energy, n_tokens, False


def generate_fixed_budget(
    model, tokenizer, prompt: str,
    max_tokens: int = 64,
) -> Tuple[str, float, int]:
    """Generate with fixed token budget (baseline)."""

    messages = [{"role": "user", "content": prompt}]
    formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(formatted, return_tensors="pt").to(model.device)

    input_len = inputs.input_ids.shape[1]

    recorder = PowerTraceRecorder(sample_interval_ms=10)
    recorder.start()

    with torch.no_grad():
        outputs = model.generate(
            inputs.input_ids,
            max_new_tokens=max_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )

    recorder.stop()

    output_ids = outputs[0, input_len:]
    output_text = tokenizer.decode(output_ids, skip_special_tokens=True)
    n_tokens = len(output_ids)

    energy = sum(s.power_watts * 0.01 for s in recorder.samples) if recorder.samples else 0.1

    return output_text, energy, n_tokens


def compute_p_error(model, tokenizer, prompt: str, platt_coef: float, platt_intercept: float) -> float:
    """Quick margin estimation for p(error)."""
    messages = [{"role": "user", "content": prompt}]
    formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(formatted, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model(inputs.input_ids)
        logits = outputs.logits[:, -1, :]
        top_logits, _ = torch.topk(logits[0], k=2)
        margin = (top_logits[0] - top_logits[1]).item()

    logit = platt_coef * margin + platt_intercept
    return 1.0 / (1.0 + math.exp(-logit))


def run_experiment(
    model_name: str = "Qwen/Qwen2.5-0.5B-Instruct",
    total_budget_j: float = 200.0,
    output_dir: Path = Path("results/energy_token_budget"),
    platt_coef: float = -0.6428,
    platt_intercept: float = -0.0008,
) -> Dict[str, Any]:
    """Run energy-as-token-budget experiment."""

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
        generate_fixed_budget(model, tokenizer, "Hello", 16)

    # Get evaluation suite
    suite = [item for item in EVAL_SUITE_EXPANDED if item.get("verify") != "unit_test"][:50]

    # Run three conditions:
    # 1. Fixed budget (baseline) - always uses max_tokens=64
    # 2. Energy-aware budget - dynamic max_tokens based on remaining energy
    # 3. Energy-aware + abstain - can abstain when budget critical

    conditions = {
        "fixed_64": {"dynamic": False, "abstain": False, "max_tokens": 64},
        "fixed_32": {"dynamic": False, "abstain": False, "max_tokens": 32},
        "energy_aware": {"dynamic": True, "abstain": False, "max_tokens": 64},
        "energy_abstain": {"dynamic": True, "abstain": True, "max_tokens": 64},
    }

    results = {}

    for condition_name, config in conditions.items():
        print(f"\n=== Condition: {condition_name} ===")

        controller = EnergyBudgetController(total_budget_j, reserve_fraction=0.1)
        condition_results = []

        for idx, item in enumerate(suite):
            state = controller.get_state()
            if state.remaining_j <= 0:
                print(f"  Budget exhausted at item {idx}")
                break

            # Get p_error for abstain decision
            p_error = compute_p_error(model, tokenizer, item["q"], platt_coef, platt_intercept)

            if config["dynamic"]:
                output, energy, n_tokens, abstained = generate_with_budget(
                    model, tokenizer, item["q"], controller, p_error, config["max_tokens"]
                )
            else:
                output, energy, n_tokens = generate_fixed_budget(
                    model, tokenizer, item["q"], config["max_tokens"]
                )
                abstained = False
                controller.update(energy, n_tokens)

            is_correct, _ = check_answer(output, item) if not abstained else (False, "abstained")

            condition_results.append({
                "correct": is_correct,
                "abstained": abstained,
                "energy_j": energy,
                "n_tokens": n_tokens,
                "budget_fraction": state.budget_fraction,
            })

            status = "✓" if is_correct else ("⊘" if abstained else "✗")
            print(f"  [{idx+1:3d}/{len(suite)}] {status} E={energy:.2f}J tok={n_tokens:2d} "
                  f"budget={state.budget_fraction*100:.0f}%")

        # Aggregate stats
        n_answered = sum(1 for r in condition_results if not r["abstained"])
        n_correct = sum(1 for r in condition_results if r["correct"])
        n_abstained = sum(1 for r in condition_results if r["abstained"])
        total_energy = sum(r["energy_j"] for r in condition_results)
        total_tokens = sum(r["n_tokens"] for r in condition_results)

        results[condition_name] = {
            "items_completed": len(condition_results),
            "n_answered": n_answered,
            "n_correct": n_correct,
            "n_abstained": n_abstained,
            "accuracy_answered": n_correct / n_answered if n_answered > 0 else 0,
            "accuracy_total": n_correct / len(condition_results) if condition_results else 0,
            "total_energy_j": total_energy,
            "total_tokens": total_tokens,
            "j_per_correct": total_energy / n_correct if n_correct > 0 else float('inf'),
            "j_per_token": total_energy / total_tokens if total_tokens > 0 else 0,
            "results": condition_results,
        }

    # Analysis
    analysis = {
        "model": model_name,
        "total_budget_j": total_budget_j,
        "n_items": len(suite),
        "conditions": {k: {kk: vv for kk, vv in v.items() if kk != "results"} for k, v in results.items()},
    }

    # Compare energy-aware vs fixed
    fixed = results["fixed_64"]
    aware = results["energy_aware"]

    analysis["comparison"] = {
        "items_completed_delta": aware["items_completed"] - fixed["items_completed"],
        "correct_delta": aware["n_correct"] - fixed["n_correct"],
        "j_per_correct_delta": aware["j_per_correct"] - fixed["j_per_correct"],
        "energy_aware_wins": aware["j_per_correct"] < fixed["j_per_correct"],
    }

    # Generate plots
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_energy_budget_comparison(results, output_dir)

    # Save results
    model_short = model_name.split("/")[-1]
    output_file = output_dir / f"energy_token_budget_{model_short}.json"

    with open(output_file, "w") as f:
        json.dump({
            "analysis": analysis,
            "results": {k: {kk: vv for kk, vv in v.items() if kk != "results"} for k, v in results.items()},
        }, f, indent=2, default=str)

    print(f"\nSaved: {output_file}")
    return analysis


def plot_energy_budget_comparison(results: Dict, output_dir: Path):
    """Generate plots comparing budget strategies."""

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    conditions = list(results.keys())
    colors = ['blue', 'cyan', 'green', 'orange']

    # Plot 1: Items completed
    ax1 = axes[0, 0]
    completed = [results[c]["items_completed"] for c in conditions]
    ax1.bar(conditions, completed, color=colors[:len(conditions)])
    ax1.set_ylabel('Items Completed')
    ax1.set_title('Coverage (items before budget exhausted)')
    for i, v in enumerate(completed):
        ax1.text(i, v + 0.5, str(v), ha='center')

    # Plot 2: Accuracy
    ax2 = axes[0, 1]
    accuracy = [results[c]["accuracy_answered"] * 100 for c in conditions]
    ax2.bar(conditions, accuracy, color=colors[:len(conditions)])
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Accuracy (answered items only)')
    for i, v in enumerate(accuracy):
        ax2.text(i, v + 1, f'{v:.1f}%', ha='center')

    # Plot 3: J/correct
    ax3 = axes[1, 0]
    j_per_correct = [results[c]["j_per_correct"] for c in conditions]
    ax3.bar(conditions, j_per_correct, color=colors[:len(conditions)])
    ax3.set_ylabel('J/Correct')
    ax3.set_title('Energy Efficiency')
    for i, v in enumerate(j_per_correct):
        ax3.text(i, v + 0.1, f'{v:.2f}', ha='center')

    # Plot 4: Token usage
    ax4 = axes[1, 1]
    tokens = [results[c]["total_tokens"] for c in conditions]
    ax4.bar(conditions, tokens, color=colors[:len(conditions)])
    ax4.set_ylabel('Total Tokens')
    ax4.set_title('Token Usage')
    for i, v in enumerate(tokens):
        ax4.text(i, v + 10, str(v), ha='center')

    plt.tight_layout()
    plt.savefig(output_dir / "energy_budget_comparison.png", dpi=150, bbox_inches='tight')
    plt.close()

    # Plot 5: Budget utilization over time
    fig, ax = plt.subplots(figsize=(10, 5))

    for c, color in zip(conditions, colors):
        budget_fracs = [r["budget_fraction"] * 100 for r in results[c]["results"]]
        ax.plot(range(len(budget_fracs)), budget_fracs, label=c, color=color, alpha=0.7)

    ax.set_xlabel('Item Number')
    ax.set_ylabel('Remaining Budget (%)')
    ax.set_title('Budget Depletion Over Time')
    ax.legend()
    ax.axhline(y=10, color='red', linestyle='--', alpha=0.5, label='Critical (10%)')

    plt.tight_layout()
    plt.savefig(output_dir / "budget_depletion.png", dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Saved plots to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Energy-as-Token-Budget experiment")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--budget", type=float, default=200.0)
    parser.add_argument("--output-dir", type=Path, default=Path("results/energy_token_budget"))
    args = parser.parse_args()

    analysis = run_experiment(
        model_name=args.model,
        total_budget_j=args.budget,
        output_dir=args.output_dir,
    )

    print("\n" + "=" * 70)
    print("ENERGY-AS-TOKEN-BUDGET SUMMARY")
    print("=" * 70)

    for condition, stats in analysis["conditions"].items():
        print(f"\n{condition}:")
        print(f"  Completed: {stats['items_completed']} items")
        print(f"  Accuracy: {stats['accuracy_answered']*100:.1f}% ({stats['n_correct']}/{stats['n_answered']})")
        print(f"  Abstained: {stats['n_abstained']}")
        print(f"  J/correct: {stats['j_per_correct']:.2f}")
        print(f"  Tokens: {stats['total_tokens']}")

    comp = analysis["comparison"]
    print(f"\nEnergy-Aware vs Fixed-64:")
    print(f"  Items completed delta: {comp['items_completed_delta']:+d}")
    print(f"  Correct delta: {comp['correct_delta']:+d}")
    print(f"  J/correct delta: {comp['j_per_correct_delta']:+.2f}")
    print(f"  Energy-aware wins: {comp['energy_aware_wins']}")


if __name__ == "__main__":
    main()
