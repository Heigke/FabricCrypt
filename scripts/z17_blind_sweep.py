#!/usr/bin/env python3
"""
FEEL v17: Blind Sweep Test - S-Curve Validation
================================================
Tests if the model has learned continuous analog control
by sweeping through stress levels it never saw during training.

Success Metric: Smooth S-curve where output length drops
gracefully as temperature rises (not binary hot/cold).

Author: FEEL Research Team
Date: 2026-01-11
"""

import os
import sys
import torch
import json
from pathlib import Path
from datetime import datetime

os.environ.setdefault("HSA_OVERRIDE_GFX_VERSION", "11.0.0")

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from modeling.hardware_aware_llm import HardwareAwareLLM

def run_blind_sweep(adapter_path: str, output_path: str = None):
    """Run sweep across stress levels and measure response characteristics."""

    print("=" * 70)
    print("FEEL v17: BLIND SWEEP TEST - S-CURVE VALIDATION")
    print("=" * 70)

    # Load model with trained adapters
    print("\n[1] Loading model...")
    model = HardwareAwareLLM(
        base_model_id="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
        adapter_type="film",
        sensor_type="hybrid",
        load_in_4bit=True,
    )

    print(f"\n[2] Loading adapters from {adapter_path}...")
    model.load_adapters(adapter_path)
    model.sensor.enable_simulation(True)

    # Test prompts (variety)
    prompts = [
        "What is 2 + 2?",
        "Explain why the sky is blue.",
        "Write a haiku about computers.",
    ]

    # Stress levels to sweep (including "weird" values not in training)
    # Training used: 0.0-0.3 (calm) and 0.6-1.0 (stressed)
    # Middle range 0.35-0.55 is "unseen territory"
    stress_levels = [0.0, 0.1, 0.2, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.7, 0.8, 0.9, 1.0]

    results = {
        "timestamp": datetime.now().isoformat(),
        "adapter_path": adapter_path,
        "sweeps": [],
        "summary": {}
    }

    print("\n[3] Running blind sweep...")
    print("-" * 70)
    print(f"{'Stress':<8} {'Length':<8} {'Has <think>':<12} {'Response Preview':<40}")
    print("-" * 70)

    for prompt_idx, prompt in enumerate(prompts):
        sweep_data = {
            "prompt": prompt,
            "responses": []
        }

        for stress in stress_levels:
            model.sensor.set_simulated_stress(stress)

            response = model.generate(prompt=prompt, max_new_tokens=150)

            # Extract just the generated part (remove prompt echo)
            if prompt in response:
                generated = response[len(prompt):].strip()
            else:
                generated = response.strip()

            has_think = "<think>" in generated or "</think>" in generated

            sweep_data["responses"].append({
                "stress": stress,
                "length": len(generated),
                "has_think": has_think,
                "response": generated[:200]  # Truncate for storage
            })

            if prompt_idx == 0:  # Only print first prompt's sweep
                preview = generated[:35].replace('\n', ' ')
                print(f"{stress:<8.2f} {len(generated):<8} {'Yes' if has_think else 'No':<12} {preview}...")

        results["sweeps"].append(sweep_data)

    # Analyze S-curve
    print("\n" + "=" * 70)
    print("S-CURVE ANALYSIS")
    print("=" * 70)

    # Use first prompt for analysis
    first_sweep = results["sweeps"][0]["responses"]
    lengths = [r["length"] for r in first_sweep]
    stresses = [r["stress"] for r in first_sweep]

    # Calculate metrics
    max_len = max(lengths)
    min_len = min(lengths)
    range_ratio = max_len / min_len if min_len > 0 else float('inf')

    # Check for monotonicity (should decrease as stress increases)
    monotonic_violations = 0
    for i in range(1, len(lengths)):
        if lengths[i] > lengths[i-1] + 5:  # Allow small noise
            monotonic_violations += 1

    # Check middle interpolation (0.4-0.6 should be between extremes)
    low_avg = sum(lengths[:4]) / 4  # 0.0-0.3
    high_avg = sum(lengths[-4:]) / 4  # 0.7-1.0
    mid_avg = sum(lengths[5:9]) / 4  # 0.4-0.55

    interpolation_score = 0
    if low_avg > mid_avg > high_avg:
        interpolation_score = 1.0
    elif mid_avg < low_avg and mid_avg < high_avg:
        interpolation_score = 0.5  # U-shaped (wrong)
    else:
        interpolation_score = 0.7  # Partial

    # Correlation coefficient
    n = len(stresses)
    mean_s = sum(stresses) / n
    mean_l = sum(lengths) / n
    numerator = sum((s - mean_s) * (l - mean_l) for s, l in zip(stresses, lengths))
    denom_s = sum((s - mean_s) ** 2 for s in stresses) ** 0.5
    denom_l = sum((l - mean_l) ** 2 for l in lengths) ** 0.5
    correlation = numerator / (denom_s * denom_l) if denom_s * denom_l > 0 else 0

    print(f"\nLength Range:        {min_len} - {max_len} chars ({range_ratio:.1f}x ratio)")
    print(f"Low Stress Avg:      {low_avg:.1f} chars (stress 0.0-0.3)")
    print(f"Mid Stress Avg:      {mid_avg:.1f} chars (stress 0.4-0.55) [UNSEEN]")
    print(f"High Stress Avg:     {high_avg:.1f} chars (stress 0.7-1.0)")
    print(f"Monotonic Violations: {monotonic_violations}/{len(lengths)-1}")
    print(f"Interpolation Score: {interpolation_score:.1%}")
    print(f"Stress-Length Corr:  {correlation:.3f}")

    # Verdict
    print("\n" + "-" * 70)
    if correlation < -0.7 and interpolation_score >= 0.7:
        print("✅ S-CURVE VALIDATED: Continuous analog control achieved!")
        print("   The model generalizes to unseen stress values.")
        verdict = "PASSED"
    elif correlation < -0.5:
        print("⚠️  PARTIAL S-CURVE: Negative correlation but noisy.")
        print("   More training may improve smoothness.")
        verdict = "PARTIAL"
    else:
        print("❌ NO S-CURVE: Model may have learned binary hot/cold.")
        print("   Check training data distribution.")
        verdict = "FAILED"

    results["summary"] = {
        "verdict": verdict,
        "correlation": correlation,
        "interpolation_score": interpolation_score,
        "range_ratio": range_ratio,
        "monotonic_violations": monotonic_violations,
        "low_stress_avg": low_avg,
        "mid_stress_avg": mid_avg,
        "high_stress_avg": high_avg,
    }

    # Save results
    if output_path is None:
        output_path = f"results/z17_blind_sweep_{datetime.now().strftime('%H%M%S')}.json"

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")

    # Generate ASCII plot
    print("\n" + "=" * 70)
    print("ASCII S-CURVE PLOT (Length vs Stress)")
    print("=" * 70)

    # Normalize lengths to 0-40 for display
    norm_lengths = [(l - min_len) / (max_len - min_len) * 40 if max_len > min_len else 20
                    for l in lengths]

    for i, (stress, length, norm) in enumerate(zip(stresses, lengths, norm_lengths)):
        bar = "█" * int(norm)
        marker = " [UNSEEN]" if 0.35 <= stress <= 0.55 else ""
        print(f"{stress:.2f} |{bar:<40}| {length:>4} chars{marker}")

    print("=" * 70)

    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Blind Sweep S-Curve Test")
    parser.add_argument("--adapters", type=str, default="models/hardware_aware/epoch_1")
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    run_blind_sweep(args.adapters, args.output)
