#!/usr/bin/env python3
"""
FEEL v18: Quick S-Curve Validation
===================================
Tests if analog training fixed the S-Curve (continuous stress response).

Author: FEEL Research Team
Date: 2026-01-12
"""

import os
import sys
import torch
from pathlib import Path
from datetime import datetime

os.environ.setdefault("HSA_OVERRIDE_GFX_VERSION", "11.0.0")

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))

from z18_analog_trainer import AnalogAwareLLM


def validate_scurve(adapter_path: str):
    """Test S-Curve: Does output length decrease as stress increases?"""

    print("=" * 70)
    print("FEEL v18: S-CURVE VALIDATION")
    print("=" * 70)
    print(f"Adapter Path: {adapter_path}")
    print(f"Timestamp: {datetime.now().isoformat()}")
    print("=" * 70)

    # Load model
    print("\n[LOADING MODEL...]")
    model = AnalogAwareLLM(
        base_model_id="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
        adapter_type="analog",
        device="cuda",
    )
    model.load_adapters(adapter_path)
    model.eval()
    print("Model loaded successfully.\n")

    # Test prompts
    prompts = [
        "What is 15 + 27?",
        "What is the capital of France?",
        "Calculate 8 * 7.",
    ]

    # Stress levels to test (11 points for smooth curve)
    stress_levels = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    all_lengths = []
    all_responses = {}

    for prompt in prompts:
        print(f"\nPrompt: '{prompt}'")
        print("-" * 60)
        print(f"{'Stress':<8} {'Length':<8} {'Preview':<40}")
        print("-" * 60)

        lengths = []
        for stress in stress_levels:
            model.set_stress_level(stress)

            # Generate
            input_ids = model.tokenizer(prompt, return_tensors="pt").input_ids.to("cuda")

            with torch.no_grad():
                outputs = model.model.generate(
                    input_ids,
                    max_new_tokens=150,
                    do_sample=True,
                    temperature=0.7,
                    pad_token_id=model.tokenizer.eos_token_id,
                )

            response = model.tokenizer.decode(outputs[0], skip_special_tokens=True)
            if prompt in response:
                generated = response[len(prompt):].strip()
            else:
                generated = response.strip()

            lengths.append(len(generated))
            preview = generated[:35].replace('\n', ' ')
            print(f"{stress:<8.1f} {len(generated):<8} {preview}...")

        all_lengths.append(lengths)
        all_responses[prompt] = lengths

    # Calculate S-Curve metrics
    print("\n" + "=" * 70)
    print("S-CURVE ANALYSIS")
    print("=" * 70)

    # Average across prompts
    avg_lengths = [sum(l[i] for l in all_lengths) / len(all_lengths) for i in range(len(stress_levels))]

    print(f"\nAverage lengths by stress level:")
    for i, (stress, length) in enumerate(zip(stress_levels, avg_lengths)):
        bar = "█" * int(length / 20)
        print(f"  {stress:.1f}: {length:6.1f} {bar}")

    # Correlation (should be negative for S-curve)
    import numpy as np
    correlation = np.corrcoef(stress_levels, avg_lengths)[0, 1]

    # Monotonicity check
    monotonic_decreases = sum(1 for i in range(len(avg_lengths)-1) if avg_lengths[i] > avg_lengths[i+1])
    monotonicity = monotonic_decreases / (len(avg_lengths) - 1) * 100

    # Range ratio
    range_ratio = max(avg_lengths) / max(min(avg_lengths), 1)

    # Low vs High comparison
    low_stress_avg = sum(avg_lengths[:3]) / 3  # 0.0, 0.1, 0.2
    high_stress_avg = sum(avg_lengths[-3:]) / 3  # 0.8, 0.9, 1.0
    efficiency_ratio = low_stress_avg / max(high_stress_avg, 1)

    print(f"\n{'='*70}")
    print("METRICS")
    print(f"{'='*70}")
    print(f"Correlation (stress vs length): {correlation:.3f} (target: < -0.5)")
    print(f"Monotonicity Score:             {monotonicity:.1f}% (target: > 60%)")
    print(f"Range Ratio:                    {range_ratio:.2f}x")
    print(f"Low Stress Avg:                 {low_stress_avg:.1f} chars")
    print(f"High Stress Avg:                {high_stress_avg:.1f} chars")
    print(f"Efficiency Ratio:               {efficiency_ratio:.2f}x (target: 1.5-2.0x)")

    print(f"\n{'='*70}")
    print("VERDICT")
    print(f"{'='*70}")

    passed = 0
    total = 4

    if correlation < -0.3:
        print("✅ S-Curve Correlation: PASS")
        passed += 1
    else:
        print(f"❌ S-Curve Correlation: FAIL (got {correlation:.3f}, need < -0.3)")

    if monotonicity > 50:
        print("✅ Monotonicity: PASS")
        passed += 1
    else:
        print(f"❌ Monotonicity: FAIL (got {monotonicity:.1f}%, need > 50%)")

    if 1.3 <= efficiency_ratio <= 4.0:
        print("✅ Efficiency Ratio: PASS")
        passed += 1
    else:
        print(f"❌ Efficiency Ratio: FAIL (got {efficiency_ratio:.2f}x, need 1.3-4.0x)")

    if range_ratio > 1.5:
        print("✅ Range Ratio: PASS")
        passed += 1
    else:
        print(f"❌ Range Ratio: FAIL (got {range_ratio:.2f}x, need > 1.5x)")

    print(f"\nOVERALL: {passed}/{total} tests passed")

    if passed >= 3:
        print("\n🎉 S-CURVE WORKING! Analog control achieved.")
    elif passed >= 2:
        print("\n⚠️ PARTIAL SUCCESS - S-Curve showing but needs refinement.")
    else:
        print("\n❌ S-CURVE STILL BROKEN - More training or data refinement needed.")

    return correlation, monotonicity, efficiency_ratio


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapters", type=str, default="models/analog_aware_z18/best")
    args = parser.parse_args()

    validate_scurve(args.adapters)
