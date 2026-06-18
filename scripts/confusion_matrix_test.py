#!/usr/bin/env python3
"""
Confusion Matrix Experiment - Proving Differential Diagnosis

This is THE critical experiment for proving FEEL is not just a "noise detector"
but an embodied intelligence with specific responses to specific stimuli.

The Test:
┌─────────────────┬─────────────────┬─────────────────┐
│                 │  Cure: Drop K   │ Cure: Summarize │
├─────────────────┼─────────────────┼─────────────────┤
│ Stressor: Heat  │    SHOULD ✓     │    SHOULD ✗     │
├─────────────────┼─────────────────┼─────────────────┤
│ Stressor: VRAM  │    SHOULD ✗     │    SHOULD ✓     │
└─────────────────┴─────────────────┴─────────────────┘

If we land the diagonal, we have proven:
1. The model distinguishes between types of stress
2. It applies the CORRECT cure for each type
3. It's not just "noise → panic" but "specific pain → specific remedy"

Usage:
    python scripts/confusion_matrix_test.py

This runs four conditions:
    1. HEAT_ONLY: High temp, low VRAM → expect Drop K, NOT summarize
    2. VRAM_ONLY: Low temp, high VRAM → expect Summarize, NOT drop K
    3. BOTH: High temp, high VRAM → expect EMERGENCY (both cures)
    4. NEITHER: Low temp, low VRAM → expect no action
"""

import sys
import os
import json
import time
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple
import subprocess

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from differential_policy import (
    DifferentialPolicy, DifferentialConfig, DiagnosisResult,
    StressorType, CureType, create_differential_policy,
)


@dataclass
class ExperimentCondition:
    """A single experimental condition."""
    name: str
    temp_c: float
    vram_percent: float
    expected_stressor: StressorType
    expected_cure: CureType


@dataclass
class ConditionResult:
    """Results from running a condition."""
    condition: str
    diagnosis_count: int
    cure_counts: Dict[str, int]
    stressor_counts: Dict[str, int]
    correct_diagnosis: int
    incorrect_diagnosis: int
    accuracy: float
    sample_messages: List[str]


class ConfusionMatrixExperiment:
    """
    Run the confusion matrix experiment to validate differential diagnosis.
    """

    def __init__(
        self,
        policy: Optional[DifferentialPolicy] = None,
        results_dir: str = "results/confusion_matrix",
    ):
        self.policy = policy or create_differential_policy(
            temp_panic=65.0,    # Lower threshold for testing
            temp_safe=55.0,
            vram_panic=0.75,    # 75% VRAM = stress
            vram_safe=0.60,     # 60% VRAM = safe
        )
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)

        # Define the four conditions
        self.conditions = [
            ExperimentCondition(
                name="HEAT_ONLY",
                temp_c=75.0,       # Hot
                vram_percent=0.50,  # Low VRAM
                expected_stressor=StressorType.THERMAL,
                expected_cure=CureType.DROP_K,
            ),
            ExperimentCondition(
                name="VRAM_ONLY",
                temp_c=50.0,       # Cool
                vram_percent=0.85,  # High VRAM
                expected_stressor=StressorType.MEMORY,
                expected_cure=CureType.SUMMARIZE,
            ),
            ExperimentCondition(
                name="BOTH_STRESS",
                temp_c=75.0,       # Hot
                vram_percent=0.85,  # High VRAM
                expected_stressor=StressorType.BOTH,
                expected_cure=CureType.EMERGENCY,
            ),
            ExperimentCondition(
                name="NEITHER",
                temp_c=50.0,       # Cool
                vram_percent=0.50,  # Low VRAM
                expected_stressor=StressorType.NONE,
                expected_cure=CureType.NONE,
            ),
        ]

    def run_condition(
        self,
        condition: ExperimentCondition,
        n_steps: int = 50,
        noise_std: float = 2.0,
    ) -> ConditionResult:
        """
        Run a single condition with noise added to simulate real variation.

        Args:
            condition: The experimental condition
            n_steps: Number of diagnosis steps
            noise_std: Standard deviation of noise to add

        Returns:
            ConditionResult with accuracy and counts
        """
        import random

        self.policy.reset()

        correct = 0
        incorrect = 0
        cure_counts = {cure.name: 0 for cure in CureType}
        stressor_counts = {stressor.name: 0 for stressor in StressorType}
        sample_messages = []

        for i in range(n_steps):
            # Add noise to simulate real conditions
            noisy_temp = condition.temp_c + random.gauss(0, noise_std)
            noisy_vram = max(0.0, min(1.0, condition.vram_percent + random.gauss(0, noise_std/100)))

            result = self.policy.diagnose(noisy_temp, noisy_vram)

            # Track counts
            cure_counts[result.cure.name] += 1
            stressor_counts[result.stressor.name] += 1

            # Check correctness
            if result.cure == condition.expected_cure:
                correct += 1
            else:
                incorrect += 1

            # Sample some messages
            if i < 5 or i == n_steps - 1:
                sample_messages.append(result.message)

        accuracy = correct / n_steps if n_steps > 0 else 0.0

        return ConditionResult(
            condition=condition.name,
            diagnosis_count=n_steps,
            cure_counts=cure_counts,
            stressor_counts=stressor_counts,
            correct_diagnosis=correct,
            incorrect_diagnosis=incorrect,
            accuracy=accuracy,
            sample_messages=sample_messages,
        )

    def run_all_conditions(self, n_steps: int = 100) -> Dict:
        """
        Run all four conditions and build confusion matrix.
        """
        print("\n" + "=" * 70)
        print("  CONFUSION MATRIX EXPERIMENT - DIFFERENTIAL DIAGNOSIS VALIDATION")
        print("=" * 70)

        results = {}
        for condition in self.conditions:
            print(f"\n[{condition.name}] Running {n_steps} steps...")
            print(f"    Temp: {condition.temp_c}°C, VRAM: {condition.vram_percent*100:.0f}%")
            print(f"    Expected: {condition.expected_stressor.name} → {condition.expected_cure.name}")

            result = self.run_condition(condition, n_steps)
            results[condition.name] = result

            print(f"    Accuracy: {result.accuracy*100:.1f}%")
            print(f"    Cure counts: {result.cure_counts}")
            print(f"    Sample: {result.sample_messages[0]}")

        return results

    def build_confusion_matrix(self, results: Dict) -> Dict:
        """
        Build the actual confusion matrix from results.

        The matrix shows:
        - Rows: Actual stressor type
        - Columns: Cure applied
        """
        matrix = {
            "heat_to_drop_k": 0,
            "heat_to_summarize": 0,
            "memory_to_drop_k": 0,
            "memory_to_summarize": 0,
        }

        # Heat condition results
        if "HEAT_ONLY" in results:
            heat_result = results["HEAT_ONLY"]
            matrix["heat_to_drop_k"] = heat_result.cure_counts.get("DROP_K", 0)
            matrix["heat_to_summarize"] = heat_result.cure_counts.get("SUMMARIZE", 0)

        # Memory condition results
        if "VRAM_ONLY" in results:
            mem_result = results["VRAM_ONLY"]
            matrix["memory_to_drop_k"] = mem_result.cure_counts.get("DROP_K", 0)
            matrix["memory_to_summarize"] = mem_result.cure_counts.get("SUMMARIZE", 0)

        return matrix

    def print_confusion_matrix(self, matrix: Dict):
        """Print a nicely formatted confusion matrix."""
        print("\n" + "=" * 60)
        print("  CONFUSION MATRIX - THE PROOF OF DIFFERENTIAL DIAGNOSIS")
        print("=" * 60)
        print()
        print("                      │    Applied Cure     │")
        print("  Actual Stressor     │  Drop K  │ Summarize│")
        print("  ────────────────────┼──────────┼──────────┤")
        print(f"  THERMAL (Heat)      │   {matrix['heat_to_drop_k']:4d}   │   {matrix['heat_to_summarize']:4d}    │")
        print(f"  MEMORY (VRAM)       │   {matrix['memory_to_drop_k']:4d}   │   {matrix['memory_to_summarize']:4d}    │")
        print("  ────────────────────┴──────────┴──────────┘")
        print()

        # Calculate diagonal accuracy
        total_heat = matrix['heat_to_drop_k'] + matrix['heat_to_summarize']
        total_memory = matrix['memory_to_drop_k'] + matrix['memory_to_summarize']

        heat_accuracy = matrix['heat_to_drop_k'] / total_heat if total_heat > 0 else 0
        memory_accuracy = matrix['memory_to_summarize'] / total_memory if total_memory > 0 else 0

        diagonal_accuracy = (matrix['heat_to_drop_k'] + matrix['memory_to_summarize']) / (total_heat + total_memory) if (total_heat + total_memory) > 0 else 0

        print(f"  Heat → Drop K accuracy:     {heat_accuracy*100:6.1f}%")
        print(f"  Memory → Summarize accuracy: {memory_accuracy*100:6.1f}%")
        print(f"  ─────────────────────────────────────────")
        print(f"  DIAGONAL ACCURACY:          {diagonal_accuracy*100:6.1f}%")
        print()

        if diagonal_accuracy >= 0.95:
            print("  ✅ DIFFERENTIAL DIAGNOSIS VALIDATED!")
            print("     The model distinguishes stress types and applies correct cures.")
        elif diagonal_accuracy >= 0.80:
            print("  ⚠️  Good discrimination, but some cross-confusion.")
        else:
            print("  ❌ Differential diagnosis NOT validated.")
            print("     The model cannot reliably distinguish stress types.")

        return {
            "heat_accuracy": heat_accuracy,
            "memory_accuracy": memory_accuracy,
            "diagonal_accuracy": diagonal_accuracy,
        }

    def run_experiment(self, n_steps: int = 100) -> Dict:
        """
        Run the complete confusion matrix experiment.
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Run all conditions
        results = self.run_all_conditions(n_steps)

        # Build confusion matrix
        matrix = self.build_confusion_matrix(results)

        # Print and analyze
        accuracies = self.print_confusion_matrix(matrix)

        # Save results
        output = {
            "timestamp": timestamp,
            "n_steps_per_condition": n_steps,
            "policy_config": asdict(self.policy.config),
            "conditions": {
                name: {
                    "condition": result.condition,
                    "diagnosis_count": result.diagnosis_count,
                    "cure_counts": result.cure_counts,
                    "stressor_counts": result.stressor_counts,
                    "accuracy": result.accuracy,
                    "sample_messages": result.sample_messages,
                }
                for name, result in results.items()
            },
            "confusion_matrix": matrix,
            "accuracies": accuracies,
        }

        output_path = self.results_dir / f"confusion_matrix_{timestamp}.json"
        with open(output_path, "w") as f:
            json.dump(output, f, indent=2, default=str)

        print(f"\n  Results saved to: {output_path}")

        return output


def run_with_live_hardware(n_steps: int = 50):
    """
    Run the confusion matrix test with actual hardware telemetry.

    This version uses real GPU stress to validate the policy.
    """
    import torch

    # Check for GPU
    if not torch.cuda.is_available():
        print("No CUDA GPU available. Running simulation only.")
        return run_simulation(n_steps)

    print("\n" + "=" * 70)
    print("  LIVE HARDWARE CONFUSION MATRIX TEST")
    print("=" * 70)

    # Import hardware utilities
    from feel_llm import FEELIntegration, FEELConfig

    # Create policy
    policy = create_differential_policy(
        temp_panic=68.0,
        temp_safe=58.0,
        vram_panic=0.80,
        vram_safe=0.65,
    )

    results = {
        "HEAT_ONLY": {"DROP_K": 0, "SUMMARIZE": 0, "messages": []},
        "VRAM_ONLY": {"DROP_K": 0, "SUMMARIZE": 0, "messages": []},
    }

    # Test 1: Generate heat without memory pressure
    print("\n[HEAT_ONLY] Generating thermal stress...")

    # Run compute-intensive task to heat GPU
    for i in range(n_steps):
        # Create heat
        if i < n_steps // 2:
            a = torch.randn(4096, 4096, device="cuda")
            b = torch.randn(4096, 4096, device="cuda")
            _ = torch.mm(a, b)

        # Get telemetry
        if hasattr(torch.cuda, 'temperature'):
            temp = torch.cuda.temperature()
        else:
            # Fallback: use nvidia-smi
            try:
                result = subprocess.run(
                    ["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader,nounits"],
                    capture_output=True, text=True
                )
                temp = float(result.stdout.strip())
            except:
                temp = 60.0

        vram_used = torch.cuda.memory_allocated() / torch.cuda.max_memory_reserved() if torch.cuda.max_memory_reserved() > 0 else 0.3

        # Diagnose
        diagnosis = policy.diagnose(temp, vram_used)
        results["HEAT_ONLY"][diagnosis.cure.name] = results["HEAT_ONLY"].get(diagnosis.cure.name, 0) + 1

        if i % 10 == 0:
            print(f"    Step {i}: {diagnosis.message}")

        # Small delay
        time.sleep(0.1)

    # Clear memory for next test
    torch.cuda.empty_cache()
    policy.reset()

    # Test 2: Generate memory pressure without excessive heat
    print("\n[VRAM_ONLY] Generating memory pressure...")

    allocations = []
    for i in range(n_steps):
        # Allocate memory in smaller chunks to avoid OOM
        try:
            if i < n_steps // 2:
                chunk = torch.zeros(256, 1024, 1024, device="cuda")  # ~1GB
                allocations.append(chunk)
        except RuntimeError:
            pass  # OOM, expected

        # Get telemetry
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader,nounits"],
                capture_output=True, text=True
            )
            temp = float(result.stdout.strip())
        except:
            temp = 55.0

        vram_used = torch.cuda.memory_allocated() / torch.cuda.get_device_properties(0).total_memory

        # Diagnose
        diagnosis = policy.diagnose(temp, vram_used)
        results["VRAM_ONLY"][diagnosis.cure.name] = results["VRAM_ONLY"].get(diagnosis.cure.name, 0) + 1

        if i % 10 == 0:
            print(f"    Step {i}: {diagnosis.message}")

        time.sleep(0.1)

    # Clean up
    allocations.clear()
    torch.cuda.empty_cache()

    # Print results
    print("\n" + "=" * 60)
    print("  LIVE HARDWARE CONFUSION MATRIX")
    print("=" * 60)
    print(f"\n  HEAT_ONLY: DROP_K={results['HEAT_ONLY'].get('DROP_K', 0)}, SUMMARIZE={results['HEAT_ONLY'].get('SUMMARIZE', 0)}")
    print(f"  VRAM_ONLY: DROP_K={results['VRAM_ONLY'].get('DROP_K', 0)}, SUMMARIZE={results['VRAM_ONLY'].get('SUMMARIZE', 0)}")

    return results


def run_simulation(n_steps: int = 100):
    """Run pure simulation without hardware."""
    experiment = ConfusionMatrixExperiment()
    return experiment.run_experiment(n_steps)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Confusion Matrix Experiment")
    parser.add_argument("--live", action="store_true", help="Use live hardware")
    parser.add_argument("--steps", type=int, default=100, help="Steps per condition")

    args = parser.parse_args()

    if args.live:
        run_with_live_hardware(args.steps)
    else:
        run_simulation(args.steps)
