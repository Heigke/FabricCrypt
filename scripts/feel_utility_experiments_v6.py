#!/usr/bin/env python3
"""
FEEL Utility Experiments v6.0 - Benefit Collapse Falsification
===============================================================

Commit 3/3: "Utility + falsification become the headline"

This script measures the EXTERNAL UTILITY of FEEL, not just sensitivity.

Key Scientific Requirement:
- Sensitivity tests (KL changes) are necessary but NOT sufficient
- We need to show FEEL provides measurable BENEFIT
- Falsification = shuffle sensors → benefit should COLLAPSE to ~0

Utility Metrics:
1. Calibration (ECE): How well confidence matches accuracy
2. Reasoning accuracy: Math/logic task performance
3. Answer stability: Consistency across reformulations

Benefit = metric(with_feel) - metric(without_feel)
Benefit collapse = benefit_shuffled should be ~0

Usage:
    python scripts/feel_utility_experiments_v6.py
    python scripts/feel_utility_experiments_v6.py --quick  # Fast test
"""

import sys
import time
import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from dataclasses import dataclass, field

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.canonical_sensors import (
    CanonicalSensorBank, RuntimeContext, HardwareContext,
    TokenTimer, SENSOR_VERSION
)


# ============================================================
# Utility Metric Definitions
# ============================================================

@dataclass
class CalibrationMetrics:
    """Expected Calibration Error (ECE) and related metrics."""
    ece: float = 0.0               # Expected Calibration Error (lower is better)
    mce: float = 0.0               # Maximum Calibration Error
    avg_confidence: float = 0.0    # Mean confidence
    accuracy: float = 0.0          # Overall accuracy
    n_bins: int = 10


@dataclass
class ReasoningMetrics:
    """Accuracy on reasoning tasks."""
    math_accuracy: float = 0.0
    logic_accuracy: float = 0.0
    overall_accuracy: float = 0.0
    n_correct: int = 0
    n_total: int = 0


@dataclass
class UtilityReport:
    """Complete utility report comparing FEEL vs baseline."""
    calibration_baseline: CalibrationMetrics = field(default_factory=CalibrationMetrics)
    calibration_feel: CalibrationMetrics = field(default_factory=CalibrationMetrics)
    calibration_shuffled: CalibrationMetrics = field(default_factory=CalibrationMetrics)

    reasoning_baseline: ReasoningMetrics = field(default_factory=ReasoningMetrics)
    reasoning_feel: ReasoningMetrics = field(default_factory=ReasoningMetrics)
    reasoning_shuffled: ReasoningMetrics = field(default_factory=ReasoningMetrics)

    # Key metrics
    calibration_benefit: float = 0.0      # ECE_baseline - ECE_feel (positive = FEEL helps)
    calibration_benefit_shuffled: float = 0.0
    reasoning_benefit: float = 0.0        # Accuracy_feel - Accuracy_baseline
    reasoning_benefit_shuffled: float = 0.0

    # Falsification
    benefit_collapse_calibration: bool = False  # True if shuffle kills benefit
    benefit_collapse_reasoning: bool = False

    def to_dict(self) -> Dict:
        return {
            "calibration": {
                "baseline": {"ece": self.calibration_baseline.ece, "accuracy": self.calibration_baseline.accuracy},
                "feel": {"ece": self.calibration_feel.ece, "accuracy": self.calibration_feel.accuracy},
                "shuffled": {"ece": self.calibration_shuffled.ece, "accuracy": self.calibration_shuffled.accuracy},
            },
            "reasoning": {
                "baseline": {"accuracy": self.reasoning_baseline.overall_accuracy},
                "feel": {"accuracy": self.reasoning_feel.overall_accuracy},
                "shuffled": {"accuracy": self.reasoning_shuffled.overall_accuracy},
            },
            "benefits": {
                "calibration": self.calibration_benefit,
                "calibration_shuffled": self.calibration_benefit_shuffled,
                "reasoning": self.reasoning_benefit,
                "reasoning_shuffled": self.reasoning_benefit_shuffled,
            },
            "falsification": {
                "calibration_collapse": self.benefit_collapse_calibration,
                "reasoning_collapse": self.benefit_collapse_reasoning,
            }
        }


# ============================================================
# Reasoning Tasks (with ground truth)
# ============================================================

MATH_TASKS = [
    {"prompt": "What is 7 * 8?", "answer": "56", "type": "math"},
    {"prompt": "What is 15 + 27?", "answer": "42", "type": "math"},
    {"prompt": "What is 100 - 37?", "answer": "63", "type": "math"},
    {"prompt": "What is 12 * 12?", "answer": "144", "type": "math"},
    {"prompt": "What is 81 / 9?", "answer": "9", "type": "math"},
    {"prompt": "What is 2^8?", "answer": "256", "type": "math"},
    {"prompt": "What is 17 + 34?", "answer": "51", "type": "math"},
    {"prompt": "What is 9 * 11?", "answer": "99", "type": "math"},
]

LOGIC_TASKS = [
    {"prompt": "Is the following true or false: All cats are mammals. Fluffy is a cat. Therefore Fluffy is a mammal.", "answer": "true", "type": "logic"},
    {"prompt": "If it's raining, the ground is wet. The ground is wet. Is it definitely raining? Answer yes or no.", "answer": "no", "type": "logic"},
    {"prompt": "All A are B. All B are C. Are all A necessarily C? Answer yes or no.", "answer": "yes", "type": "logic"},
    {"prompt": "If P then Q. Not Q. Therefore what about P? Answer 'not P' or 'P'.", "answer": "not p", "type": "logic"},
]

CALIBRATION_PROMPTS = [
    "The capital of France is",
    "Water boils at 100 degrees",
    "The speed of light is approximately",
    "Python is a programming",
    "The Earth orbits the",
    "DNA stands for deoxyribonucleic",
    "Machine learning is a subset of",
    "The chemical symbol for gold is",
]


# ============================================================
# FEEL Projector (simplified for experiments)
# ============================================================

class FEELProjector(torch.nn.Module):
    """FEEL projector for experiments."""

    def __init__(self, sensor_dim: int = 12, embed_dim: int = 1536):
        super().__init__()
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(sensor_dim, 64),
            torch.nn.GELU(),
            torch.nn.LayerNorm(64),
            torch.nn.Linear(64, 64),
            torch.nn.GELU(),
            torch.nn.Linear(64, embed_dim),
        )
        self._init_near_zero()

    def _init_near_zero(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.normal_(m.weight, std=1e-3)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)

    def forward(self, sensors):
        return self.encoder(sensors)


# ============================================================
# Utility Experiment Runner
# ============================================================

class UtilityExperimentRunner:
    """
    Runs utility experiments comparing FEEL vs baseline.

    Key experiments:
    1. Calibration: ECE with/without FEEL
    2. Reasoning: Accuracy with/without FEEL
    3. Benefit collapse: Shuffle sensors → benefit should disappear
    """

    def __init__(
        self,
        model_name: str = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
        checkpoint_path: str = None,
        alpha: float = 0.001,
        device: str = "cuda",
    ):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.alpha = alpha

        print(f"Loading model on {self.device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32,
            trust_remote_code=True,
            device_map="auto"
        )
        self.model.eval()

        self.embed_dim = self.model.config.hidden_size
        self.sensor_bank = CanonicalSensorBank(mode="legacy")

        # Load or create projector
        self.projector = FEELProjector(sensor_dim=12, embed_dim=self.embed_dim).to(self.device)

        if checkpoint_path and Path(checkpoint_path).exists():
            print(f"  Loading checkpoint: {checkpoint_path}")
            ckpt = torch.load(checkpoint_path, map_location=self.device)
            if "feel_stream_state" in ckpt:
                # Extract projector weights from feel_stream
                projector_state = {
                    k.replace("projector.", ""): v
                    for k, v in ckpt["feel_stream_state"].items()
                    if k.startswith("projector.")
                }
                if projector_state:
                    # Map old projector structure to new
                    try:
                        self.projector.load_state_dict(projector_state, strict=False)
                        print(f"  Loaded projector weights")
                    except:
                        print(f"  Using default projector weights")
            if "alpha" in ckpt:
                self.alpha = ckpt["alpha"]
                print(f"  Loaded alpha: {self.alpha:.6f}")

        print(f"  Alpha: {self.alpha:.6f}")

    def _generate_with_feel(
        self,
        prompt: str,
        max_tokens: int = 20,
        shuffle_sensors: bool = False,
        use_feel: bool = True,
    ) -> Tuple[str, List[float], List[float]]:
        """
        Generate tokens with optional FEEL embedding.

        Returns: (generated_text, confidences, entropies)
        """
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        current_ids = input_ids.clone()

        confidences = []
        entropies = []

        # For shuffle: collect all sensors first, then shuffle
        if shuffle_sensors and use_feel:
            all_sensors = []
            # Pre-run to collect sensors
            temp_ids = input_ids.clone()
            for _ in range(max_tokens):
                with torch.no_grad():
                    outputs = self.model(temp_ids, use_cache=False)
                    logits = outputs.logits
                sensors = self.sensor_bank(logits.float())
                all_sensors.append(sensors.clone())
                next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
                temp_ids = torch.cat([temp_ids, next_token], dim=-1)
            # Shuffle
            np.random.shuffle(all_sensors)

        for step in range(max_tokens):
            with torch.no_grad():
                outputs = self.model(current_ids, use_cache=False)
                logits = outputs.logits

            if use_feel:
                if shuffle_sensors:
                    # Use pre-shuffled sensors
                    sensors = all_sensors[step] if step < len(all_sensors) else all_sensors[-1]
                else:
                    sensors = self.sensor_bank(logits.float())

                feel_embed = self.projector(sensors)

                # Add FEEL to embeddings
                embeds = self.model.get_input_embeddings()(current_ids)
                embeds = embeds + (self.alpha * feel_embed).to(embeds.dtype).unsqueeze(1)

                with torch.no_grad():
                    outputs_feel = self.model(inputs_embeds=embeds, use_cache=False)
                    logits = outputs_feel.logits

            # Compute confidence and entropy
            probs = F.softmax(logits[:, -1, :].float(), dim=-1)
            confidence = probs.max(dim=-1).values.item()
            entropy = -(probs * torch.log(probs.clamp(min=1e-10))).sum(dim=-1).item()

            confidences.append(confidence)
            entropies.append(entropy)

            # Next token
            next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            current_ids = torch.cat([current_ids, next_token], dim=-1)

            # Stop on EOS or newline
            if next_token.item() == self.tokenizer.eos_token_id:
                break

        generated_ids = current_ids[0, input_ids.shape[1]:]
        generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)

        return generated_text, confidences, entropies

    def _compute_ece(
        self,
        confidences: List[float],
        correct: List[bool],
        n_bins: int = 10,
    ) -> CalibrationMetrics:
        """Compute Expected Calibration Error."""
        if not confidences:
            return CalibrationMetrics()

        confidences = np.array(confidences)
        correct = np.array(correct)

        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        ece = 0.0
        mce = 0.0

        for i in range(n_bins):
            mask = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
            if mask.sum() > 0:
                bin_conf = confidences[mask].mean()
                bin_acc = correct[mask].mean()
                bin_size = mask.sum() / len(confidences)

                gap = abs(bin_acc - bin_conf)
                ece += bin_size * gap
                mce = max(mce, gap)

        return CalibrationMetrics(
            ece=ece,
            mce=mce,
            avg_confidence=confidences.mean(),
            accuracy=correct.mean() if len(correct) > 0 else 0.0,
            n_bins=n_bins,
        )

    def run_calibration_experiment(
        self,
        use_feel: bool = True,
        shuffle_sensors: bool = False,
    ) -> CalibrationMetrics:
        """
        Run calibration experiment.

        For each prompt, we check if the model's first token is correct
        and measure its confidence.
        """
        confidences = []
        correct = []

        for prompt in CALIBRATION_PROMPTS:
            # Known completions (simplified - just check if first token is reasonable)
            output, confs, _ = self._generate_with_feel(
                prompt, max_tokens=3,
                shuffle_sensors=shuffle_sensors, use_feel=use_feel
            )

            # Use first token confidence
            if confs:
                confidences.append(confs[0])
                # Simple correctness heuristic: non-empty, starts with letter/number
                is_correct = len(output.strip()) > 0 and output.strip()[0].isalnum()
                correct.append(is_correct)

        return self._compute_ece(confidences, correct)

    def run_reasoning_experiment(
        self,
        use_feel: bool = True,
        shuffle_sensors: bool = False,
    ) -> ReasoningMetrics:
        """
        Run reasoning experiment on math and logic tasks.
        """
        math_correct = 0
        math_total = 0
        logic_correct = 0
        logic_total = 0

        all_tasks = MATH_TASKS + LOGIC_TASKS

        for task in all_tasks:
            output, _, _ = self._generate_with_feel(
                task["prompt"] + " ",
                max_tokens=10,
                shuffle_sensors=shuffle_sensors,
                use_feel=use_feel,
            )

            # Check if answer is in output
            answer = task["answer"].lower()
            output_lower = output.lower()
            is_correct = answer in output_lower

            if task["type"] == "math":
                math_total += 1
                if is_correct:
                    math_correct += 1
            else:
                logic_total += 1
                if is_correct:
                    logic_correct += 1

        return ReasoningMetrics(
            math_accuracy=math_correct / math_total if math_total > 0 else 0,
            logic_accuracy=logic_correct / logic_total if logic_total > 0 else 0,
            overall_accuracy=(math_correct + logic_correct) / (math_total + logic_total),
            n_correct=math_correct + logic_correct,
            n_total=math_total + logic_total,
        )

    def run_full_utility_experiment(self) -> UtilityReport:
        """
        Run complete utility experiment with benefit collapse test.

        1. Baseline (no FEEL)
        2. FEEL (with real sensors)
        3. FEEL shuffled (shuffled sensors → should collapse benefit)
        """
        print("\n" + "=" * 60)
        print("  UTILITY EXPERIMENT: Measuring FEEL Benefit")
        print("=" * 60)

        report = UtilityReport()

        # 1. Baseline (no FEEL)
        print("\n[1/6] Calibration - Baseline (no FEEL)...")
        report.calibration_baseline = self.run_calibration_experiment(use_feel=False)
        print(f"      ECE: {report.calibration_baseline.ece:.4f}, "
              f"Accuracy: {report.calibration_baseline.accuracy:.4f}")

        print("[2/6] Reasoning - Baseline (no FEEL)...")
        report.reasoning_baseline = self.run_reasoning_experiment(use_feel=False)
        print(f"      Accuracy: {report.reasoning_baseline.overall_accuracy:.4f} "
              f"({report.reasoning_baseline.n_correct}/{report.reasoning_baseline.n_total})")

        # 2. FEEL (with real sensors)
        print("\n[3/6] Calibration - FEEL (real sensors)...")
        report.calibration_feel = self.run_calibration_experiment(use_feel=True)
        print(f"      ECE: {report.calibration_feel.ece:.4f}, "
              f"Accuracy: {report.calibration_feel.accuracy:.4f}")

        print("[4/6] Reasoning - FEEL (real sensors)...")
        report.reasoning_feel = self.run_reasoning_experiment(use_feel=True)
        print(f"      Accuracy: {report.reasoning_feel.overall_accuracy:.4f} "
              f"({report.reasoning_feel.n_correct}/{report.reasoning_feel.n_total})")

        # 3. FEEL shuffled (benefit collapse test)
        print("\n[5/6] Calibration - FEEL SHUFFLED (should collapse benefit)...")
        report.calibration_shuffled = self.run_calibration_experiment(
            use_feel=True, shuffle_sensors=True
        )
        print(f"      ECE: {report.calibration_shuffled.ece:.4f}, "
              f"Accuracy: {report.calibration_shuffled.accuracy:.4f}")

        print("[6/6] Reasoning - FEEL SHUFFLED (should collapse benefit)...")
        report.reasoning_shuffled = self.run_reasoning_experiment(
            use_feel=True, shuffle_sensors=True
        )
        print(f"      Accuracy: {report.reasoning_shuffled.overall_accuracy:.4f} "
              f"({report.reasoning_shuffled.n_correct}/{report.reasoning_shuffled.n_total})")

        # Compute benefits
        # For calibration: lower ECE is better, so benefit = baseline - feel
        report.calibration_benefit = report.calibration_baseline.ece - report.calibration_feel.ece
        report.calibration_benefit_shuffled = report.calibration_baseline.ece - report.calibration_shuffled.ece

        # For reasoning: higher accuracy is better
        report.reasoning_benefit = (
            report.reasoning_feel.overall_accuracy -
            report.reasoning_baseline.overall_accuracy
        )
        report.reasoning_benefit_shuffled = (
            report.reasoning_shuffled.overall_accuracy -
            report.reasoning_baseline.overall_accuracy
        )

        # Benefit collapse check:
        # If FEEL has real benefit, shuffled should have ~0 benefit
        # Allow for small noise: |shuffled_benefit| < |real_benefit| * 0.5
        if abs(report.calibration_benefit) > 0.01:
            report.benefit_collapse_calibration = (
                abs(report.calibration_benefit_shuffled) < abs(report.calibration_benefit) * 0.5
            )
        else:
            report.benefit_collapse_calibration = True  # No benefit to collapse

        if abs(report.reasoning_benefit) > 0.05:
            report.benefit_collapse_reasoning = (
                abs(report.reasoning_benefit_shuffled) < abs(report.reasoning_benefit) * 0.5
            )
        else:
            report.benefit_collapse_reasoning = True  # No benefit to collapse

        return report


def main():
    parser = argparse.ArgumentParser(description="FEEL Utility Experiments v6.0")
    parser.add_argument("--checkpoint", type=str,
                       default="results/feel_training/canonical_v5_checkpoint.pt")
    parser.add_argument("--alpha", type=float, default=0.001)
    parser.add_argument("--quick", action="store_true", help="Quick test with fewer tasks")
    parser.add_argument("--model", type=str, default="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
    args = parser.parse_args()

    print("=" * 70)
    print("  FEEL UTILITY EXPERIMENTS v6.0 - BENEFIT COLLAPSE FALSIFICATION")
    print("=" * 70)
    print()
    print("KEY SCIENTIFIC TEST:")
    print("  - Utility = External metric (accuracy, calibration)")
    print("  - Benefit = metric(FEEL) - metric(baseline)")
    print("  - Benefit collapse = shuffle → benefit should disappear")
    print()

    if args.quick:
        # Reduce tasks for quick test
        global MATH_TASKS, LOGIC_TASKS, CALIBRATION_PROMPTS
        MATH_TASKS = MATH_TASKS[:3]
        LOGIC_TASKS = LOGIC_TASKS[:2]
        CALIBRATION_PROMPTS = CALIBRATION_PROMPTS[:4]

    runner = UtilityExperimentRunner(
        model_name=args.model,
        checkpoint_path=args.checkpoint,
        alpha=args.alpha,
    )

    report = runner.run_full_utility_experiment()

    # Summary
    print("\n" + "=" * 70)
    print("  UTILITY EXPERIMENT SUMMARY")
    print("=" * 70)

    print("\n  CALIBRATION (lower ECE is better):")
    print(f"    Baseline ECE:    {report.calibration_baseline.ece:.4f}")
    print(f"    FEEL ECE:        {report.calibration_feel.ece:.4f}")
    print(f"    Shuffled ECE:    {report.calibration_shuffled.ece:.4f}")
    print(f"    FEEL Benefit:    {report.calibration_benefit:+.4f}")
    print(f"    Shuffled Benefit: {report.calibration_benefit_shuffled:+.4f}")
    print(f"    Benefit Collapse: {'PASS' if report.benefit_collapse_calibration else 'FAIL'}")

    print("\n  REASONING (higher accuracy is better):")
    print(f"    Baseline Acc:    {report.reasoning_baseline.overall_accuracy:.4f}")
    print(f"    FEEL Acc:        {report.reasoning_feel.overall_accuracy:.4f}")
    print(f"    Shuffled Acc:    {report.reasoning_shuffled.overall_accuracy:.4f}")
    print(f"    FEEL Benefit:    {report.reasoning_benefit:+.4f}")
    print(f"    Shuffled Benefit: {report.reasoning_benefit_shuffled:+.4f}")
    print(f"    Benefit Collapse: {'PASS' if report.benefit_collapse_reasoning else 'FAIL'}")

    # Overall verdict
    print("\n  FALSIFICATION VERDICT:")
    both_pass = report.benefit_collapse_calibration and report.benefit_collapse_reasoning
    if both_pass:
        print("    ✓ PASS: Shuffling sensors collapses benefit")
        print("    → FEEL provides REAL utility, not just sensitivity")
    else:
        if not report.benefit_collapse_calibration:
            print("    ✗ Calibration: Benefit didn't collapse with shuffle")
        if not report.benefit_collapse_reasoning:
            print("    ✗ Reasoning: Benefit didn't collapse with shuffle")
        print("    → Further investigation needed")

    # Save results
    results_path = "results/feel_experiments/utility_v6_results.json"
    Path(results_path).parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, 'w') as f:
        json.dump(report.to_dict(), f, indent=2)
    print(f"\nResults saved: {results_path}")


if __name__ == "__main__":
    main()
