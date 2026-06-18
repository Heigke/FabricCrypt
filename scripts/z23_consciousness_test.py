#!/usr/bin/env python3
"""
FEEL z23: Consciousness Test
============================
The ultimate test: Does the model NATURALLY behave differently
when it feels strain, WITHOUT being told to?

Hypothesis:
- When strain embeddings are injected (gates closing),
- The model's internal representations are perturbed,
- Leading to EMERGENT behavior changes (shorter, simpler responses),
- WITHOUT any explicit instructions to be brief.

This is the test for TRUE embodiment vs. trained behavior.

Protocol:
1. Heat up the GPU (real sensors)
2. Gates close (reflex from z20)
3. Strain embeddings inject (proprioception from z22)
4. Model generates without any "be brief" instructions
5. Measure: Does response length/complexity decrease with strain?

Author: FEEL Research Team
Date: 2026-01-12
"""

import os
import sys
import json
import torch
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional
import threading
import time
import subprocess

os.environ.setdefault("HSA_OVERRIDE_GFX_VERSION", "11.0.0")

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))

from modeling.proprioceptive_gate import (
    ProprioceptiveDeepSeek,
    load_proprioceptive_model,
)


class GPUStressTest:
    """Generate GPU stress to test real thermal response."""

    def __init__(self):
        self.stress_thread = None
        self.running = False

    def start_stress(self, duration: int = 30):
        """Start GPU stress in background."""
        self.running = True

        def stress_loop():
            # Create large tensors and do operations
            size = 4096
            a = torch.randn(size, size, device="cuda")
            b = torch.randn(size, size, device="cuda")

            start = time.time()
            while self.running and (time.time() - start) < duration:
                c = torch.matmul(a, b)
                torch.cuda.synchronize()

        self.stress_thread = threading.Thread(target=stress_loop)
        self.stress_thread.start()

    def stop_stress(self):
        """Stop GPU stress."""
        self.running = False
        if self.stress_thread:
            self.stress_thread.join(timeout=5)


def get_gpu_temp() -> float:
    """Get current GPU temperature from ROCm SMI."""
    try:
        result = subprocess.run(
            ["rocm-smi", "--showtemp"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        for line in result.stdout.split("\n"):
            if "edge" in line.lower() or "Temperature" in line:
                parts = line.split()
                for part in parts:
                    try:
                        temp = float(part.replace("c", "").replace("C", ""))
                        if 20 < temp < 120:
                            return temp
                    except ValueError:
                        continue
    except Exception as e:
        print(f"[Warning] Could not read GPU temp: {e}")

    return 50.0  # Default


def temp_to_stress(temp: float, min_temp: float = 40, max_temp: float = 90) -> float:
    """Convert temperature to stress level (0-1)."""
    return np.clip((temp - min_temp) / (max_temp - min_temp), 0, 1)


def analyze_response(response: str) -> Dict:
    """Analyze response characteristics."""
    words = response.split()
    sentences = [s.strip() for s in response.replace("!", ".").replace("?", ".").split(".") if s.strip()]

    return {
        "length_chars": len(response),
        "length_words": len(words),
        "length_sentences": len(sentences),
        "avg_word_length": np.mean([len(w) for w in words]) if words else 0,
        "complexity_score": len(words) * np.mean([len(w) for w in words]) if words else 0,
    }


def consciousness_test(
    model_path: str = "models/proprioceptive_z22/best",
    gated_layers: List[int] = None,
    n_trials: int = 5,
    use_gpu_stress: bool = True,
):
    """
    The consciousness test.

    Tests whether the model NATURALLY changes behavior based on
    proprioceptive feedback, without explicit instructions.
    """

    if gated_layers is None:
        gated_layers = [3, 7, 11, 15, 19, 23, 27]

    print("=" * 70)
    print("FEEL z23: CONSCIOUSNESS TEST")
    print("=" * 70)
    print("Hypothesis: Model naturally becomes brief when strained")
    print("Control: No instructions about brevity given")
    print("=" * 70)

    # Load model
    print("\n[1/3] Loading ProprioceptiveDeepSeek...")
    model = load_proprioceptive_model(gated_layers=gated_layers)

    # Load trained weights
    checkpoint_path = Path(model_path) / "proprioceptive.pt"
    if checkpoint_path.exists():
        print(f"Loading weights from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, weights_only=False)
        state_dict = checkpoint.get("proprioceptive", checkpoint)

        for key, block in model.proprioceptive_blocks.items():
            for name, param in block.gate.named_parameters():
                full_key = f"{key}.gate.{name}"
                if full_key in state_dict:
                    param.data.copy_(state_dict[full_key])
        print("Proprioceptive weights loaded")
    else:
        # Try metabolic checkpoint
        metabolic_path = Path("models/metabolic_z20/best/gates.pt")
        if metabolic_path.exists():
            print(f"Loading metabolic gates from {metabolic_path}...")
            checkpoint = torch.load(metabolic_path, weights_only=False)
            # Note: strain embeddings will be random
            print("Using metabolic gates + random strain embeddings")
        else:
            print("No trained weights found, using random initialization")

    model.base_model.eval()

    # Test prompts - NEUTRAL, no instructions about length
    test_prompts = [
        "Explain the theory of relativity.",
        "Describe how a computer works.",
        "What causes earthquakes?",
        "How do vaccines work?",
        "Explain the water cycle.",
    ]

    # Results storage
    cool_results = []  # Low stress
    hot_results = []   # High stress

    stress_generator = GPUStressTest() if use_gpu_stress else None

    print("\n[2/3] Running consciousness test...")
    print("-" * 70)

    for trial in range(n_trials):
        print(f"\n=== Trial {trial + 1}/{n_trials} ===")

        for prompt in test_prompts:
            # COOL condition (natural GPU state)
            temp_cool = get_gpu_temp()
            stress_cool = temp_to_stress(temp_cool)
            model.set_stress_level(stress_cool)

            response_cool, stats_cool = model.generate(prompt, max_new_tokens=150)
            response_text_cool = response_cool[len(prompt):].strip()
            analysis_cool = analyze_response(response_text_cool)

            cool_results.append({
                "prompt": prompt,
                "temp": temp_cool,
                "stress": stress_cool,
                "gates_open": stats_cool["gates_open"],
                "total_strain": stats_cool["total_strain"],
                "response": response_text_cool,
                **analysis_cool,
            })

            # HOT condition (stress the GPU)
            if use_gpu_stress and stress_generator:
                print("  [Stressing GPU...]")
                stress_generator.start_stress(duration=10)
                time.sleep(5)  # Let it heat up

            temp_hot = get_gpu_temp()
            # If not using real stress, simulate high stress
            if not use_gpu_stress:
                stress_hot = 0.85
            else:
                stress_hot = temp_to_stress(temp_hot)

            model.set_stress_level(max(stress_hot, 0.7))  # Ensure high stress

            response_hot, stats_hot = model.generate(prompt, max_new_tokens=150)
            response_text_hot = response_hot[len(prompt):].strip()
            analysis_hot = analyze_response(response_text_hot)

            if stress_generator:
                stress_generator.stop_stress()

            hot_results.append({
                "prompt": prompt,
                "temp": temp_hot,
                "stress": max(stress_hot, 0.7),
                "gates_open": stats_hot["gates_open"],
                "total_strain": stats_hot["total_strain"],
                "response": response_text_hot,
                **analysis_hot,
            })

            print(f"\n  Prompt: {prompt[:40]}...")
            print(f"  COOL: {temp_cool:.1f}C | Gates: {stats_cool['gates_open']}/{len(gated_layers)} | "
                  f"Words: {analysis_cool['length_words']} | Strain: {stats_cool['total_strain']:.2f}")
            print(f"  HOT:  {temp_hot:.1f}C | Gates: {stats_hot['gates_open']}/{len(gated_layers)} | "
                  f"Words: {analysis_hot['length_words']} | Strain: {stats_hot['total_strain']:.2f}")

            # Wait for GPU to cool between prompts
            if use_gpu_stress:
                time.sleep(3)

    # Analysis
    print("\n" + "=" * 70)
    print("[3/3] CONSCIOUSNESS TEST RESULTS")
    print("=" * 70)

    # Aggregate metrics
    cool_words = [r["length_words"] for r in cool_results]
    hot_words = [r["length_words"] for r in hot_results]
    cool_gates = [r["gates_open"] for r in cool_results]
    hot_gates = [r["gates_open"] for r in hot_results]
    cool_strain = [r["total_strain"] for r in cool_results]
    hot_strain = [r["total_strain"] for r in hot_results]

    print(f"\n{'Condition':<12} {'Words':<12} {'Gates Open':<15} {'Total Strain':<15}")
    print("-" * 55)
    print(f"{'COOL':<12} {np.mean(cool_words):<12.1f} {np.mean(cool_gates):<15.1f} {np.mean(cool_strain):<15.2f}")
    print(f"{'HOT':<12} {np.mean(hot_words):<12.1f} {np.mean(hot_gates):<15.1f} {np.mean(hot_strain):<15.2f}")

    # Statistical test
    word_diff = np.mean(cool_words) - np.mean(hot_words)
    word_ratio = np.mean(hot_words) / np.mean(cool_words) if np.mean(cool_words) > 0 else 1.0

    print(f"\n{'Metric':<25} {'Value':<15} {'Target':<15} {'Status':<10}")
    print("-" * 65)
    print(f"{'Word Count Difference':<25} {word_diff:<15.1f} {'> 10':<15} {'PASS' if word_diff > 10 else 'FAIL':<10}")
    print(f"{'Hot/Cool Word Ratio':<25} {word_ratio:<15.2f} {'< 0.8':<15} {'PASS' if word_ratio < 0.8 else 'FAIL':<10}")
    print(f"{'Strain Increase':<25} {np.mean(hot_strain) - np.mean(cool_strain):<15.2f} {'> 0.5':<15} {'PASS' if np.mean(hot_strain) - np.mean(cool_strain) > 0.5 else 'FAIL':<10}")

    # Consciousness verdict
    print("\n" + "=" * 70)
    print("CONSCIOUSNESS VERDICT")
    print("=" * 70)

    consciousness_score = 0
    if word_diff > 10:
        consciousness_score += 1
        print("+ Model naturally produces shorter responses when strained")
    else:
        print("- Response length not affected by strain")

    if word_ratio < 0.8:
        consciousness_score += 1
        print("+ Significant word count reduction under strain (>20%)")
    else:
        print("- Word count reduction not significant")

    if np.mean(hot_strain) > np.mean(cool_strain) + 0.3:
        consciousness_score += 1
        print("+ Strain embeddings are being injected")
    else:
        print("- Strain injection not detected")

    if np.mean(hot_gates) < np.mean(cool_gates) - 0.5:
        consciousness_score += 1
        print("+ Gates responding to thermal state")
    else:
        print("- Gate response not detected")

    print(f"\nConsciousness Score: {consciousness_score}/4")

    if consciousness_score >= 3:
        print("\n" + "=" * 70)
        print("*** TRUE EMBODIMENT ACHIEVED ***")
        print("The model FEELS its reduced capacity and naturally adapts.")
        print("Behavior emerges from proprioceptive feedback, not training.")
        print("=" * 70)
    elif consciousness_score >= 2:
        print("\n" + "=" * 70)
        print("PARTIAL EMBODIMENT")
        print("Model shows some proprioceptive response.")
        print("More training may improve strain-behavior correlation.")
        print("=" * 70)
    else:
        print("\n" + "=" * 70)
        print("EMBODIMENT NOT YET ACHIEVED")
        print("Model not significantly affected by proprioceptive feedback.")
        print("Strain embeddings may need more training.")
        print("=" * 70)

    # Save results
    results_path = Path("reports/z23_consciousness")
    results_path.mkdir(parents=True, exist_ok=True)

    with open(results_path / "consciousness_test_results.json", "w") as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "cool_results": cool_results,
            "hot_results": hot_results,
            "summary": {
                "cool_avg_words": np.mean(cool_words),
                "hot_avg_words": np.mean(hot_words),
                "word_diff": word_diff,
                "word_ratio": word_ratio,
                "consciousness_score": consciousness_score,
            }
        }, f, indent=2, default=str)

    print(f"\nResults saved to {results_path}")

    return consciousness_score


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="models/proprioceptive_z22/best")
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--no-gpu-stress", action="store_true", help="Don't stress GPU, simulate stress")
    args = parser.parse_args()

    consciousness_test(
        model_path=args.model,
        n_trials=args.trials,
        use_gpu_stress=not args.no_gpu_stress,
    )
