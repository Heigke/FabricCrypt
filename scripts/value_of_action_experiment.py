#!/usr/bin/env python3
"""
Value of Action Experiment: Thermo-Conditioned Cognitive Action Selection

Novel Contribution:
-------------------
Learn MEASURED benefit curves (Δcorrect, Δjoules) conditioned on hardware state,
rather than assuming "more samples = better" (which fails for 0.5B per Result 17).

The key insight: the VALUE of cognitive actions depends on:
  - Model uncertainty (margin, entropy)
  - Task type (category, difficulty)
  - Hardware state (temp, dT/dt, power, throttle, budget_remaining)

Actions Available:
  A1: greedy (1 sample, T=0)
  A2: vote_3 (3 samples, T=0.5, majority vote)
  A3: vote_5 (5 samples, T=0.7, majority vote)
  A4: verify (unit test / numeric check where applicable)
  A5: concise (max 16 tokens, forces brevity)

We collect (state, action, outcome) tuples and train a tiny model:
  - Predicts Δcorrect (probability improvement vs greedy)
  - Predicts Δjoules (energy cost vs greedy)

Then choose action by: argmax_a [ p_error * Δcorrect(a) - λ * Δjoules(a) ]

This is the "proprioceptive controller that learns hardware-conditioned cognition."
"""

import json
import time
import argparse
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass, field, asdict
from collections import defaultdict
import math

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.energy_harness.amd_smi_monitor import PowerTraceRecorder
from scripts.eval_suite import EVAL_SUITE_EXPANDED, check_answer, check_answer_simple

try:
    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import cross_val_score
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


@dataclass
class HardwareState:
    """Current hardware proprioceptive state."""
    temperature_c: float
    temp_derivative: float  # dT/dt over last window
    power_watts: float
    clock_mhz: int
    throttle_active: bool
    budget_remaining_j: float
    budget_fraction: float


@dataclass
class CognitiveState:
    """Model's internal cognitive state."""
    margin_min: float
    margin_mean: float
    entropy_max: float
    entropy_mean: float
    p_error: float  # Calibrated error probability


@dataclass
class TaskContext:
    """Task metadata."""
    category: str
    difficulty: str
    verify_type: str


@dataclass
class ActionOutcome:
    """Result of taking an action."""
    action: str
    correct: bool
    energy_j: float
    latency_ms: float
    output_tokens: int
    output_text: str


@dataclass
class Experience:
    """Full experience tuple for learning."""
    hw_state: HardwareState
    cog_state: CognitiveState
    task: TaskContext
    outcomes: Dict[str, ActionOutcome]  # action_name -> outcome


# Action configurations
ACTIONS = {
    "greedy": {"samples": 1, "temperature": 0.0, "max_tokens": 64},
    "vote_3": {"samples": 3, "temperature": 0.5, "max_tokens": 64},
    "vote_5": {"samples": 5, "temperature": 0.7, "max_tokens": 64},
    "concise": {"samples": 1, "temperature": 0.0, "max_tokens": 16},
}


class ThermalTracker:
    """Track thermal state over time for derivative calculation."""

    def __init__(self, window_size: int = 10):
        self.temps = []
        self.times = []
        self.window_size = window_size
        self.throttle_events = 0
        self.last_clock = None

    def update(self, temp: float, clock: int, timestamp: float = None):
        """Update with new reading."""
        if timestamp is None:
            timestamp = time.time()

        self.temps.append(temp)
        self.times.append(timestamp)

        # Keep only recent window
        if len(self.temps) > self.window_size * 2:
            self.temps = self.temps[-self.window_size:]
            self.times = self.times[-self.window_size:]

        # Detect throttle event (clock drop under load)
        if self.last_clock is not None and clock < self.last_clock * 0.9:
            self.throttle_events += 1
        self.last_clock = clock

    def get_derivative(self) -> float:
        """Get dT/dt in °C/s."""
        if len(self.temps) < 2:
            return 0.0

        n = min(self.window_size, len(self.temps))
        dt = self.times[-1] - self.times[-n]
        if dt <= 0:
            return 0.0

        dtemp = self.temps[-1] - self.temps[-n]
        return dtemp / dt

    def get_current_temp(self) -> float:
        """Get most recent temperature."""
        return self.temps[-1] if self.temps else 50.0


def estimate_hardware_state(
    recorder: PowerTraceRecorder,
    thermal_tracker: ThermalTracker,
    budget_total: float,
    spent: float,
) -> HardwareState:
    """Estimate current hardware state from telemetry."""
    if recorder.samples:
        latest = recorder.samples[-1]
        power = latest.power_watts
        # Estimate temperature from power (approximate for iGPU)
        temp = min(95, 50 + power * 0.4)
        clock = getattr(latest, 'sclk_mhz', 1800)
    else:
        power = 50.0
        temp = 55.0
        clock = 1800

    thermal_tracker.update(temp, clock)

    return HardwareState(
        temperature_c=temp,
        temp_derivative=thermal_tracker.get_derivative(),
        power_watts=power,
        clock_mhz=clock,
        throttle_active=temp > 85 or power > 90,
        budget_remaining_j=max(0, budget_total - spent),
        budget_fraction=max(0, (budget_total - spent) / budget_total),
    )


def generate_with_action(
    model, tokenizer, prompt: str, action_config: dict
) -> Tuple[str, List[float], float]:
    """Generate response with specified action configuration."""
    samples = action_config["samples"]
    temperature = action_config["temperature"]
    max_tokens = action_config["max_tokens"]

    messages = [{"role": "user", "content": f"{prompt}\nAnswer concisely."}]
    formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(formatted, return_tensors="pt").to(model.device)

    input_len = inputs.input_ids.shape[1]

    recorder = PowerTraceRecorder(sample_interval_ms=10)
    recorder.start()

    all_outputs = []
    all_margins = []

    with torch.no_grad():
        for _ in range(samples):
            generated_ids = inputs.input_ids.clone()
            margins = []

            for step in range(max_tokens):
                outputs = model(generated_ids)
                logits = outputs.logits[:, -1, :]

                # Capture margin
                top_logits, _ = torch.topk(logits[0], k=2)
                margin = (top_logits[0] - top_logits[1]).item()
                margins.append(margin)

                # Sample
                if temperature > 0:
                    probs = F.softmax(logits / temperature, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                else:
                    next_token = logits.argmax(dim=-1, keepdim=True)

                generated_ids = torch.cat([generated_ids, next_token], dim=1)

                if next_token.item() == tokenizer.eos_token_id:
                    break

            output_ids = generated_ids[0, input_len:]
            output_text = tokenizer.decode(output_ids, skip_special_tokens=True)
            all_outputs.append(output_text)
            all_margins.extend(margins)

    recorder.stop()

    # For voting, select most common answer (simplified)
    if samples > 1:
        # Use mode of outputs
        from collections import Counter
        output_counts = Counter(all_outputs)
        final_output = output_counts.most_common(1)[0][0]
    else:
        final_output = all_outputs[0]

    energy = sum(s.power_watts * 0.01 for s in recorder.samples) if recorder.samples else 0.1

    return final_output, all_margins, energy


def compute_cognitive_state(margins: List[float], platt_coef: float, platt_intercept: float) -> CognitiveState:
    """Compute cognitive state from margin signals."""
    if not margins:
        return CognitiveState(0, 0, 0, 0, 0.5)

    margin_min = min(margins)
    margin_mean = sum(margins) / len(margins)

    # Approximate entropy from margin (higher margin = lower entropy)
    entropies = [max(0, 3 - m) for m in margins]
    entropy_max = max(entropies)
    entropy_mean = sum(entropies) / len(entropies)

    # Calibrated p(error)
    logit = platt_coef * margin_min + platt_intercept
    p_error = 1.0 / (1.0 + math.exp(-logit))

    return CognitiveState(
        margin_min=margin_min,
        margin_mean=margin_mean,
        entropy_max=entropy_max,
        entropy_mean=entropy_mean,
        p_error=p_error,
    )


def collect_experiences(
    model, tokenizer,
    items: List[Dict],
    platt_coef: float = -0.6428,
    platt_intercept: float = -0.0008,
    budget_j: float = 500.0,
) -> List[Experience]:
    """Collect experience tuples by running all actions on each item."""

    experiences = []
    thermal_tracker = ThermalTracker()
    spent_j = 0.0

    print(f"\nCollecting experiences from {len(items)} items...")
    print(f"Actions: {list(ACTIONS.keys())}")

    for idx, item in enumerate(items):
        if spent_j >= budget_j * 0.9:  # Reserve 10% budget
            print(f"  Budget limit reached at item {idx}")
            break

        recorder = PowerTraceRecorder(sample_interval_ms=50)
        recorder.start()
        time.sleep(0.1)  # Get initial reading

        # Get initial hardware state
        hw_state = estimate_hardware_state(recorder, thermal_tracker, budget_j, spent_j)

        # Run greedy first to get cognitive state
        greedy_output, greedy_margins, greedy_energy = generate_with_action(
            model, tokenizer, item["q"], ACTIONS["greedy"]
        )

        cog_state = compute_cognitive_state(greedy_margins, platt_coef, platt_intercept)

        task_context = TaskContext(
            category=item.get("cat", "unknown"),
            difficulty=item.get("diff", "unknown"),
            verify_type=item.get("verify", "exact"),
        )

        # Collect outcomes for all actions
        outcomes = {}

        # Greedy (already done)
        greedy_correct, _ = check_answer(greedy_output, item)
        outcomes["greedy"] = ActionOutcome(
            action="greedy",
            correct=greedy_correct,
            energy_j=greedy_energy,
            latency_ms=greedy_energy * 10,  # Approximate
            output_tokens=len(greedy_output.split()),
            output_text=greedy_output[:100],
        )
        spent_j += greedy_energy

        # Other actions
        for action_name, action_config in ACTIONS.items():
            if action_name == "greedy":
                continue

            output, margins, energy = generate_with_action(
                model, tokenizer, item["q"], action_config
            )
            correct, _ = check_answer(output, item)

            outcomes[action_name] = ActionOutcome(
                action=action_name,
                correct=correct,
                energy_j=energy,
                latency_ms=energy * 10,
                output_tokens=len(output.split()),
                output_text=output[:100],
            )
            spent_j += energy

        recorder.stop()

        exp = Experience(
            hw_state=hw_state,
            cog_state=cog_state,
            task=task_context,
            outcomes=outcomes,
        )
        experiences.append(exp)

        # Status
        best_action = max(outcomes.items(), key=lambda x: (x[1].correct, -x[1].energy_j))[0]
        status = "✓" if outcomes["greedy"].correct else "✗"
        print(f"  [{idx+1:3d}/{len(items)}] {status} p_err={cog_state.p_error:.2f} "
              f"T={hw_state.temperature_c:.0f}°C best={best_action} spent={spent_j:.0f}J")

    return experiences


def build_feature_vector(exp: Experience) -> np.ndarray:
    """Build feature vector from experience for ML model."""
    features = [
        # Hardware state
        exp.hw_state.temperature_c / 100,  # Normalize to 0-1
        exp.hw_state.temp_derivative,
        exp.hw_state.power_watts / 100,
        exp.hw_state.throttle_active * 1.0,
        exp.hw_state.budget_fraction,

        # Cognitive state
        exp.cog_state.margin_min,
        exp.cog_state.margin_mean,
        exp.cog_state.entropy_max,
        exp.cog_state.p_error,

        # Task context (one-hot encoding)
        1.0 if exp.task.category == "math" else 0.0,
        1.0 if exp.task.category == "qa" else 0.0,
        1.0 if exp.task.category == "reasoning" else 0.0,
        1.0 if exp.task.difficulty == "easy" else 0.0,
        1.0 if exp.task.difficulty == "medium" else 0.0,
        1.0 if exp.task.difficulty == "hard" else 0.0,
    ]
    return np.array(features)


class ActionValueModel:
    """Learns value (Δcorrect, Δjoules) of actions conditioned on state."""

    def __init__(self):
        self.correct_models = {}  # action -> classifier for P(correct | state)
        self.energy_models = {}   # action -> regressor for E[energy | state]
        self.scaler = StandardScaler()
        self.fitted = False
        self.baseline_correct = {}  # action -> overall P(correct)
        self.baseline_energy = {}   # action -> mean energy

    def fit(self, experiences: List[Experience]):
        """Fit models from collected experiences."""
        if not HAS_SKLEARN:
            print("sklearn not available, using simple baselines")
            return

        # Build dataset
        X = np.array([build_feature_vector(exp) for exp in experiences])
        X_scaled = self.scaler.fit_transform(X)

        for action in ACTIONS.keys():
            # Labels
            y_correct = np.array([exp.outcomes[action].correct for exp in experiences])
            y_energy = np.array([exp.outcomes[action].energy_j for exp in experiences])

            # Store baselines
            self.baseline_correct[action] = y_correct.mean()
            self.baseline_energy[action] = y_energy.mean()

            # Fit correctness classifier
            if len(np.unique(y_correct)) > 1:
                clf = GradientBoostingClassifier(n_estimators=50, max_depth=3, random_state=42)
                clf.fit(X_scaled, y_correct)
                self.correct_models[action] = clf

            # Fit energy regressor
            reg = GradientBoostingRegressor(n_estimators=50, max_depth=3, random_state=42)
            reg.fit(X_scaled, y_energy)
            self.energy_models[action] = reg

        self.fitted = True
        print(f"\nTrained ActionValueModel on {len(experiences)} experiences")
        print(f"  Baseline P(correct): {self.baseline_correct}")
        print(f"  Baseline E[energy]: {self.baseline_energy}")

    def predict(self, exp: Experience) -> Dict[str, Tuple[float, float]]:
        """Predict (P(correct), E[energy]) for each action given state."""
        X = build_feature_vector(exp).reshape(1, -1)
        X_scaled = self.scaler.transform(X)

        predictions = {}
        for action in ACTIONS.keys():
            if self.fitted and action in self.correct_models:
                p_correct = self.correct_models[action].predict_proba(X_scaled)[0, 1]
            else:
                p_correct = self.baseline_correct.get(action, 0.5)

            if self.fitted and action in self.energy_models:
                e_energy = self.energy_models[action].predict(X_scaled)[0]
            else:
                e_energy = self.baseline_energy.get(action, 1.0)

            predictions[action] = (p_correct, e_energy)

        return predictions

    def select_action(
        self, exp: Experience, lambda_energy: float = 0.1
    ) -> Tuple[str, Dict[str, float]]:
        """Select best action by expected utility."""
        predictions = self.predict(exp)

        # Utility = P(correct) - λ * energy
        utilities = {}
        for action, (p_correct, e_energy) in predictions.items():
            # Δcorrect relative to greedy
            greedy_p = predictions["greedy"][0]
            delta_correct = p_correct - greedy_p

            # Δenergy relative to greedy
            greedy_e = predictions["greedy"][1]
            delta_energy = e_energy - greedy_e

            # Expected utility with correction for current error risk
            p_error = exp.cog_state.p_error
            utility = p_error * delta_correct - lambda_energy * delta_energy

            # Greedy is baseline (utility = 0 for greedy)
            if action == "greedy":
                utility = 0

            utilities[action] = utility

        best_action = max(utilities.items(), key=lambda x: x[1])[0]
        return best_action, utilities


def analyze_action_value(experiences: List[Experience]) -> Dict[str, Any]:
    """Analyze the value of different actions from collected data."""

    analysis = {
        "n_experiences": len(experiences),
        "action_stats": {},
        "delta_vs_greedy": {},
        "hw_state_impact": {},
    }

    # Per-action statistics
    for action in ACTIONS.keys():
        correct_list = [exp.outcomes[action].correct for exp in experiences]
        energy_list = [exp.outcomes[action].energy_j for exp in experiences]

        analysis["action_stats"][action] = {
            "accuracy": sum(correct_list) / len(correct_list),
            "mean_energy": sum(energy_list) / len(energy_list),
            "n_correct": sum(correct_list),
        }

    # Delta vs greedy
    greedy_stats = analysis["action_stats"]["greedy"]
    for action in ACTIONS.keys():
        if action == "greedy":
            continue
        stats = analysis["action_stats"][action]
        analysis["delta_vs_greedy"][action] = {
            "delta_accuracy": stats["accuracy"] - greedy_stats["accuracy"],
            "delta_energy": stats["mean_energy"] - greedy_stats["mean_energy"],
            "delta_correct": stats["n_correct"] - greedy_stats["n_correct"],
        }

    # Hardware state impact: split by temperature
    hot_exps = [e for e in experiences if e.hw_state.temperature_c > 70]
    cold_exps = [e for e in experiences if e.hw_state.temperature_c <= 70]

    for regime, exps in [("hot", hot_exps), ("cold", cold_exps)]:
        if not exps:
            continue
        regime_stats = {}
        for action in ACTIONS.keys():
            correct = [e.outcomes[action].correct for e in exps]
            energy = [e.outcomes[action].energy_j for e in exps]
            regime_stats[action] = {
                "accuracy": sum(correct) / len(correct) if correct else 0,
                "mean_energy": sum(energy) / len(energy) if energy else 0,
            }
        analysis["hw_state_impact"][regime] = regime_stats

    # High uncertainty vs low uncertainty
    high_unc = [e for e in experiences if e.cog_state.p_error > 0.3]
    low_unc = [e for e in experiences if e.cog_state.p_error <= 0.3]

    for regime, exps in [("high_uncertainty", high_unc), ("low_uncertainty", low_unc)]:
        if not exps:
            continue
        regime_stats = {}
        for action in ACTIONS.keys():
            correct = [e.outcomes[action].correct for e in exps]
            regime_stats[action] = {
                "accuracy": sum(correct) / len(correct) if correct else 0,
                "n": len(correct),
            }
        analysis["hw_state_impact"][regime] = regime_stats

    return analysis


def plot_value_of_action(experiences: List[Experience], analysis: Dict, output_dir: Path):
    """Generate plots for value of action analysis."""

    # Plot 1: Action comparison
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    actions = list(ACTIONS.keys())
    accuracies = [analysis["action_stats"][a]["accuracy"] * 100 for a in actions]
    energies = [analysis["action_stats"][a]["mean_energy"] for a in actions]

    # Accuracy
    ax1 = axes[0]
    colors = ['blue', 'orange', 'green', 'red']
    ax1.bar(actions, accuracies, color=colors[:len(actions)])
    ax1.set_ylabel('Accuracy (%)')
    ax1.set_title('Accuracy by Action')
    ax1.set_ylim(0, 100)
    for i, v in enumerate(accuracies):
        ax1.text(i, v + 2, f'{v:.1f}%', ha='center')

    # Energy
    ax2 = axes[1]
    ax2.bar(actions, energies, color=colors[:len(actions)])
    ax2.set_ylabel('Energy (J)')
    ax2.set_title('Energy by Action')
    for i, v in enumerate(energies):
        ax2.text(i, v + 0.1, f'{v:.1f}', ha='center')

    # Efficiency (accuracy / energy)
    ax3 = axes[2]
    efficiency = [a / e if e > 0 else 0 for a, e in zip(accuracies, energies)]
    ax3.bar(actions, efficiency, color=colors[:len(actions)])
    ax3.set_ylabel('Accuracy/J')
    ax3.set_title('Efficiency (Acc/J)')
    for i, v in enumerate(efficiency):
        ax3.text(i, v + 0.5, f'{v:.1f}', ha='center')

    plt.tight_layout()
    plt.savefig(output_dir / "action_comparison.png", dpi=150, bbox_inches='tight')
    plt.close()

    # Plot 2: Hardware state impact
    if "hot" in analysis["hw_state_impact"] and "cold" in analysis["hw_state_impact"]:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))

        x = np.arange(len(actions))
        width = 0.35

        cold_acc = [analysis["hw_state_impact"]["cold"][a]["accuracy"] * 100 for a in actions]
        hot_acc = [analysis["hw_state_impact"]["hot"][a]["accuracy"] * 100 for a in actions]

        ax1 = axes[0]
        ax1.bar(x - width/2, cold_acc, width, label='Cold (<70°C)', color='blue', alpha=0.7)
        ax1.bar(x + width/2, hot_acc, width, label='Hot (>70°C)', color='red', alpha=0.7)
        ax1.set_ylabel('Accuracy (%)')
        ax1.set_title('Accuracy by Thermal Regime')
        ax1.set_xticks(x)
        ax1.set_xticklabels(actions)
        ax1.legend()

        cold_e = [analysis["hw_state_impact"]["cold"][a]["mean_energy"] for a in actions]
        hot_e = [analysis["hw_state_impact"]["hot"][a]["mean_energy"] for a in actions]

        ax2 = axes[1]
        ax2.bar(x - width/2, cold_e, width, label='Cold', color='blue', alpha=0.7)
        ax2.bar(x + width/2, hot_e, width, label='Hot', color='red', alpha=0.7)
        ax2.set_ylabel('Energy (J)')
        ax2.set_title('Energy by Thermal Regime')
        ax2.set_xticks(x)
        ax2.set_xticklabels(actions)
        ax2.legend()

        plt.tight_layout()
        plt.savefig(output_dir / "thermal_regime_impact.png", dpi=150, bbox_inches='tight')
        plt.close()

    # Plot 3: Uncertainty impact
    if "high_uncertainty" in analysis["hw_state_impact"] and "low_uncertainty" in analysis["hw_state_impact"]:
        fig, ax = plt.subplots(figsize=(8, 5))

        x = np.arange(len(actions))
        width = 0.35

        low_acc = [analysis["hw_state_impact"]["low_uncertainty"][a]["accuracy"] * 100 for a in actions]
        high_acc = [analysis["hw_state_impact"]["high_uncertainty"][a]["accuracy"] * 100 for a in actions]

        ax.bar(x - width/2, low_acc, width, label='Low Uncertainty (p_err≤0.3)', color='green', alpha=0.7)
        ax.bar(x + width/2, high_acc, width, label='High Uncertainty (p_err>0.3)', color='orange', alpha=0.7)
        ax.set_ylabel('Accuracy (%)')
        ax.set_title('When Does Extra Compute Help?')
        ax.set_xticks(x)
        ax.set_xticklabels(actions)
        ax.legend()

        # Annotate delta
        for i, action in enumerate(actions):
            if action != "greedy":
                delta = high_acc[i] - analysis["hw_state_impact"]["high_uncertainty"]["greedy"]["accuracy"] * 100
                ax.annotate(f'Δ={delta:+.1f}%', (i + width/2, high_acc[i] + 1), ha='center', fontsize=8)

        plt.tight_layout()
        plt.savefig(output_dir / "uncertainty_impact.png", dpi=150, bbox_inches='tight')
        plt.close()

    print(f"Saved plots to {output_dir}")


def run_experiment(
    model_name: str = "Qwen/Qwen2.5-0.5B-Instruct",
    budget_j: float = 300.0,
    output_dir: Path = Path("results/value_of_action"),
    lambda_energy: float = 0.1,
) -> Dict[str, Any]:
    """Run Value of Action experiment."""

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
        generate_with_action(model, tokenizer, "Hello", ACTIONS["greedy"])

    # Collect experiences (use subset of eval suite)
    suite = [item for item in EVAL_SUITE_EXPANDED if item.get("verify") != "unit_test"][:40]

    experiences = collect_experiences(
        model, tokenizer, suite,
        budget_j=budget_j,
    )

    # Analyze
    analysis = analyze_action_value(experiences)

    # Train action value model
    avm = ActionValueModel()
    avm.fit(experiences)

    # Evaluate learned policy vs baselines
    print("\n=== Evaluating Learned Policy ===")
    learned_correct = 0
    greedy_correct = 0
    learned_energy = 0.0
    greedy_energy = 0.0
    action_choices = defaultdict(int)

    for exp in experiences:
        # Learned policy
        best_action, utilities = avm.select_action(exp, lambda_energy)
        action_choices[best_action] += 1

        if exp.outcomes[best_action].correct:
            learned_correct += 1
        learned_energy += exp.outcomes[best_action].energy_j

        # Greedy baseline
        if exp.outcomes["greedy"].correct:
            greedy_correct += 1
        greedy_energy += exp.outcomes["greedy"].energy_j

    n = len(experiences)
    analysis["learned_policy"] = {
        "accuracy": learned_correct / n,
        "total_energy": learned_energy,
        "j_per_correct": learned_energy / max(1, learned_correct),
        "action_distribution": dict(action_choices),
    }
    analysis["greedy_policy"] = {
        "accuracy": greedy_correct / n,
        "total_energy": greedy_energy,
        "j_per_correct": greedy_energy / max(1, greedy_correct),
    }
    analysis["improvement"] = {
        "delta_accuracy": (learned_correct - greedy_correct) / n,
        "delta_energy": learned_energy - greedy_energy,
    }

    # Generate plots
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_value_of_action(experiences, analysis, output_dir)

    # Save results
    model_short = model_name.split("/")[-1]
    output_file = output_dir / f"value_of_action_{model_short}.json"

    # Convert experiences to serializable format
    exp_data = []
    for exp in experiences:
        exp_data.append({
            "hw_state": asdict(exp.hw_state),
            "cog_state": asdict(exp.cog_state),
            "task": asdict(exp.task),
            "outcomes": {k: asdict(v) for k, v in exp.outcomes.items()},
        })

    with open(output_file, "w") as f:
        json.dump({
            "model": model_name,
            "budget_j": budget_j,
            "lambda_energy": lambda_energy,
            "analysis": analysis,
            "experiences": exp_data,
        }, f, indent=2, default=str)

    print(f"\nSaved: {output_file}")
    return analysis


def main():
    parser = argparse.ArgumentParser(description="Value of Action experiment")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--budget", type=float, default=300.0)
    parser.add_argument("--lambda-energy", type=float, default=0.1)
    parser.add_argument("--output-dir", type=Path, default=Path("results/value_of_action"))
    args = parser.parse_args()

    analysis = run_experiment(
        model_name=args.model,
        budget_j=args.budget,
        lambda_energy=args.lambda_energy,
        output_dir=args.output_dir,
    )

    print("\n" + "=" * 70)
    print("VALUE OF ACTION EXPERIMENT SUMMARY")
    print("=" * 70)

    print("\nAction Statistics:")
    for action, stats in analysis["action_stats"].items():
        print(f"  {action:10s}: acc={stats['accuracy']*100:5.1f}%  E={stats['mean_energy']:.2f}J")

    print("\nDelta vs Greedy:")
    for action, delta in analysis["delta_vs_greedy"].items():
        print(f"  {action:10s}: Δacc={delta['delta_accuracy']*100:+5.1f}%  ΔE={delta['delta_energy']:+.2f}J")

    if "learned_policy" in analysis:
        print("\nLearned Policy vs Greedy:")
        lp = analysis["learned_policy"]
        gp = analysis["greedy_policy"]
        imp = analysis["improvement"]
        print(f"  Learned: acc={lp['accuracy']*100:.1f}%  E={lp['total_energy']:.1f}J  J/correct={lp['j_per_correct']:.2f}")
        print(f"  Greedy:  acc={gp['accuracy']*100:.1f}%  E={gp['total_energy']:.1f}J  J/correct={gp['j_per_correct']:.2f}")
        print(f"  Delta:   Δacc={imp['delta_accuracy']*100:+.1f}%  ΔE={imp['delta_energy']:+.1f}J")
        print(f"  Action distribution: {lp['action_distribution']}")


if __name__ == "__main__":
    main()
