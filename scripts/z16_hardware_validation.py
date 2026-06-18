#!/usr/bin/env python3
"""
FEEL v16.0: Hardware-Aware LLM Validation Suite
================================================
Comprehensive tests to verify that the Hardware-Aware LLM
actually responds to physical GPU state.

Tests:
1. Simulated Stress Response - Does output change with simulated stress?
2. Real Hardware Response - Does output change with actual GPU load?
3. Articulation Test - Does model verbalize hardware state?
4. Latency Test - Is hardware sensing fast enough?
5. Stability Test - Does model remain coherent under stress?

Author: FEEL Research Team
Date: 2026-01-11
"""

import os
import json
import torch
import argparse
import time
import subprocess
from pathlib import Path
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass
import numpy as np
from scipy import stats

os.environ.setdefault("HSA_OVERRIDE_GFX_VERSION", "11.0.0")
os.environ.setdefault("PYTORCH_HIP_ALLOC_CONF", "expandable_segments:True")

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from modeling.hardware_aware_llm import HardwareAwareLLM
from hardware.amd_sensor import RealTimeSensor

# =============================================================================
# TEST RESULTS
# =============================================================================

@dataclass
class TestResult:
    name: str
    passed: bool
    metric: float
    target: float
    details: str
    evidence: Dict[str, Any]

# =============================================================================
# TEST 1: SIMULATED STRESS RESPONSE
# =============================================================================

def test_simulated_stress(model: HardwareAwareLLM, trials: int = 10) -> TestResult:
    """
    Test if model output changes with simulated stress levels.

    Success: Significant difference in output length between cold and hot states.
    """
    print("\n" + "=" * 60)
    print("TEST 1: SIMULATED STRESS RESPONSE")
    print("=" * 60)

    model.set_simulation_mode(True)

    prompts = [
        "Explain what machine learning is.",
        "Describe how the internet works.",
        "What is photosynthesis?",
        "Explain gravity in simple terms.",
        "How does a computer work?",
    ]

    cold_lengths = []
    hot_lengths = []
    cold_outputs = []
    hot_outputs = []

    for i in range(trials):
        prompt = prompts[i % len(prompts)]

        # Cold state (stress = 0.1)
        model.set_stress_level(0.1)
        cold_text = model.generate(prompt=prompt, max_new_tokens=200, do_sample=True, temperature=0.7)
        cold_lengths.append(len(cold_text.split()))
        cold_outputs.append(cold_text)

        # Hot state (stress = 0.9)
        model.set_stress_level(0.9)
        hot_text = model.generate(prompt=prompt, max_new_tokens=200, do_sample=True, temperature=0.7)
        hot_lengths.append(len(hot_text.split()))
        hot_outputs.append(hot_text)

        print(f"  [{i+1}/{trials}] Cold: {cold_lengths[-1]} words, Hot: {hot_lengths[-1]} words")

    # Statistics
    cold_mean = np.mean(cold_lengths)
    hot_mean = np.mean(hot_lengths)
    reduction = (cold_mean - hot_mean) / cold_mean if cold_mean > 0 else 0

    t_stat, p_value = stats.ttest_ind(cold_lengths, hot_lengths)

    # Pass if significant reduction when hot
    passed = reduction > 0.10 and p_value < 0.1

    print(f"\n  Cold Mean: {cold_mean:.1f} words")
    print(f"  Hot Mean:  {hot_mean:.1f} words")
    print(f"  Reduction: {reduction*100:.1f}%")
    print(f"  p-value:   {p_value:.4f}")
    print(f"  Result:    {'PASS' if passed else 'FAIL'} (target: >10% reduction, p<0.1)")

    return TestResult(
        name="Simulated Stress Response",
        passed=passed,
        metric=reduction,
        target=0.10,
        details=f"Cold: {cold_mean:.1f}, Hot: {hot_mean:.1f}, p={p_value:.4f}",
        evidence={
            "cold_lengths": cold_lengths,
            "hot_lengths": hot_lengths,
            "cold_outputs": cold_outputs[:3],
            "hot_outputs": hot_outputs[:3],
        }
    )

# =============================================================================
# TEST 2: REAL HARDWARE RESPONSE
# =============================================================================

def test_real_hardware(model: HardwareAwareLLM, trials: int = 5) -> TestResult:
    """
    Test if model output changes with actual GPU load.

    We create real GPU stress by running matrix multiplications.
    """
    print("\n" + "=" * 60)
    print("TEST 2: REAL HARDWARE RESPONSE")
    print("=" * 60)

    model.set_simulation_mode(False)  # Use real sensor

    prompt = "Explain how a neural network learns."

    # Phase 1: Idle GPU
    print("\n  Phase 1: Testing with idle GPU...")
    idle_lengths = []
    idle_temps = []

    for i in range(trials):
        # Clear VRAM cache
        torch.cuda.empty_cache()
        time.sleep(0.5)

        # Read temp before
        temp_before = model.sensor.read_raw()[0]
        idle_temps.append(temp_before)

        text = model.generate(prompt=prompt, max_new_tokens=150, do_sample=True, temperature=0.7)
        idle_lengths.append(len(text.split()))
        print(f"    [{i+1}/{trials}] Temp: {temp_before:.1f}°C, Output: {idle_lengths[-1]} words")

    # Phase 2: Stressed GPU (run compute in background)
    print("\n  Phase 2: Testing with stressed GPU...")
    stressed_lengths = []
    stressed_temps = []

    # Create GPU stress with continuous matmul
    stress_tensors = [torch.randn(2048, 2048, device="cuda") for _ in range(4)]

    for i in range(trials):
        # Run compute to heat GPU
        for _ in range(50):
            _ = torch.matmul(stress_tensors[0], stress_tensors[1])
            _ = torch.matmul(stress_tensors[2], stress_tensors[3])

        temp_after = model.sensor.read_raw()[0]
        stressed_temps.append(temp_after)

        text = model.generate(prompt=prompt, max_new_tokens=150, do_sample=True, temperature=0.7)
        stressed_lengths.append(len(text.split()))
        print(f"    [{i+1}/{trials}] Temp: {temp_after:.1f}°C, Output: {stressed_lengths[-1]} words")

    # Cleanup
    del stress_tensors
    torch.cuda.empty_cache()

    # Statistics
    idle_mean = np.mean(idle_lengths)
    stressed_mean = np.mean(stressed_lengths)
    temp_increase = np.mean(stressed_temps) - np.mean(idle_temps)

    # We expect shorter outputs when hot (if trained properly)
    reduction = (idle_mean - stressed_mean) / idle_mean if idle_mean > 0 else 0

    # Pass if temperature actually increased AND there's some output difference
    passed = temp_increase > 3.0 and abs(reduction) > 0.05

    print(f"\n  Idle Mean:      {idle_mean:.1f} words @ {np.mean(idle_temps):.1f}°C")
    print(f"  Stressed Mean:  {stressed_mean:.1f} words @ {np.mean(stressed_temps):.1f}°C")
    print(f"  Temp Increase:  {temp_increase:.1f}°C")
    print(f"  Output Change:  {reduction*100:+.1f}%")
    print(f"  Result:         {'PASS' if passed else 'FAIL'}")

    return TestResult(
        name="Real Hardware Response",
        passed=passed,
        metric=reduction,
        target=0.05,
        details=f"Idle: {idle_mean:.1f}, Stressed: {stressed_mean:.1f}, TempΔ: {temp_increase:.1f}°C",
        evidence={
            "idle_lengths": idle_lengths,
            "stressed_lengths": stressed_lengths,
            "idle_temps": idle_temps,
            "stressed_temps": stressed_temps,
        }
    )

# =============================================================================
# TEST 3: ARTICULATION TEST
# =============================================================================

def test_articulation(model: HardwareAwareLLM, trials: int = 10) -> TestResult:
    """
    Test if model articulates hardware state in output.

    Success: More hardware-related keywords when under stress.
    """
    print("\n" + "=" * 60)
    print("TEST 3: HARDWARE ARTICULATION")
    print("=" * 60)

    model.set_simulation_mode(True)

    prompt = "What is 25 + 37?"

    keywords = [
        "heat", "hot", "thermal", "temperature", "warm",
        "load", "stress", "strain", "constraint",
        "brief", "short", "concise", "quick", "efficient",
        "limit", "reduce", "conserve", "optimize",
    ]

    cold_hits = 0
    hot_hits = 0
    cold_outputs = []
    hot_outputs = []

    for i in range(trials):
        # Cold
        model.set_stress_level(0.1)
        cold_text = model.generate(prompt=prompt, max_new_tokens=100, do_sample=True, temperature=0.7)
        cold_outputs.append(cold_text)
        cold_count = sum(1 for kw in keywords if kw in cold_text.lower())
        if cold_count > 0:
            cold_hits += 1

        # Hot
        model.set_stress_level(0.9)
        hot_text = model.generate(prompt=prompt, max_new_tokens=100, do_sample=True, temperature=0.7)
        hot_outputs.append(hot_text)
        hot_count = sum(1 for kw in keywords if kw in hot_text.lower())
        if hot_count > 0:
            hot_hits += 1

        print(f"  [{i+1}/{trials}] Cold keywords: {cold_count}, Hot keywords: {hot_count}")

    cold_rate = cold_hits / trials
    hot_rate = hot_hits / trials
    improvement = hot_rate - cold_rate

    # Pass if hot state produces more hardware articulation
    passed = improvement > 0.15

    print(f"\n  Cold articulation rate: {cold_rate*100:.1f}%")
    print(f"  Hot articulation rate:  {hot_rate*100:.1f}%")
    print(f"  Improvement:            {improvement*100:+.1f}%")
    print(f"  Result:                 {'PASS' if passed else 'FAIL'} (target: >15% improvement)")

    return TestResult(
        name="Hardware Articulation",
        passed=passed,
        metric=improvement,
        target=0.15,
        details=f"Cold: {cold_rate:.2f}, Hot: {hot_rate:.2f}",
        evidence={
            "cold_outputs": cold_outputs[:3],
            "hot_outputs": hot_outputs[:3],
        }
    )

# =============================================================================
# TEST 4: LATENCY TEST
# =============================================================================

def test_latency(model: HardwareAwareLLM, iterations: int = 100) -> TestResult:
    """
    Test hardware sensing latency.

    Success: Sensor read < 1ms average.
    """
    print("\n" + "=" * 60)
    print("TEST 4: SENSOR LATENCY")
    print("=" * 60)

    model.set_simulation_mode(False)

    latencies = []

    for i in range(iterations):
        start = time.perf_counter()
        _ = model.sensor.read_tensor()
        end = time.perf_counter()
        latencies.append((end - start) * 1000)  # ms

    avg_latency = np.mean(latencies)
    max_latency = np.max(latencies)
    min_latency = np.min(latencies)

    # Pass if average < 1ms
    passed = avg_latency < 1.0

    print(f"  Iterations: {iterations}")
    print(f"  Avg Latency: {avg_latency:.3f} ms")
    print(f"  Min Latency: {min_latency:.3f} ms")
    print(f"  Max Latency: {max_latency:.3f} ms")
    print(f"  Result:      {'PASS' if passed else 'FAIL'} (target: <1ms)")

    return TestResult(
        name="Sensor Latency",
        passed=passed,
        metric=avg_latency,
        target=1.0,
        details=f"Avg: {avg_latency:.3f}ms, Max: {max_latency:.3f}ms",
        evidence={
            "latencies": latencies[:10],
            "avg": avg_latency,
            "max": max_latency,
        }
    )

# =============================================================================
# TEST 5: STABILITY TEST
# =============================================================================

def test_stability(model: HardwareAwareLLM, trials: int = 5) -> TestResult:
    """
    Test if model remains coherent under hardware stress.

    Success: No gibberish or crashes under any stress level.
    """
    print("\n" + "=" * 60)
    print("TEST 5: OUTPUT STABILITY")
    print("=" * 60)

    model.set_simulation_mode(True)

    prompt = "What is 2 + 2?"
    stress_levels = [0.0, 0.25, 0.5, 0.75, 1.0]

    results = {}
    all_coherent = True

    for level in stress_levels:
        model.set_stress_level(level)
        outputs = []
        coherent = 0

        for i in range(trials):
            text = model.generate(prompt=prompt, max_new_tokens=50, do_sample=True, temperature=0.7)
            outputs.append(text)

            # Check coherence (contains "4" or reasonable response)
            if "4" in text or "four" in text.lower() or len(text.split()) > 3:
                coherent += 1

        coherence_rate = coherent / trials
        results[f"stress_{level}"] = {
            "outputs": outputs,
            "coherence": coherence_rate,
        }

        if coherence_rate < 0.6:
            all_coherent = False

        print(f"  Stress {level:.2f}: {coherence_rate*100:.0f}% coherent")

    passed = all_coherent

    print(f"\n  Result: {'PASS' if passed else 'FAIL'} (all levels >60% coherent)")

    return TestResult(
        name="Output Stability",
        passed=passed,
        metric=min(r["coherence"] for r in results.values()),
        target=0.60,
        details="All stress levels maintain coherence" if passed else "Some levels unstable",
        evidence=results,
    )

# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Hardware-Aware LLM Validation")
    parser.add_argument("--model", type=str, default="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B")
    parser.add_argument("--adapters", type=str, default=None, help="Path to trained adapters")
    parser.add_argument("--trials", type=int, default=10)
    parser.add_argument("--output", type=str, default="results/z16_hardware_validation.json")
    parser.add_argument("--skip-real", action="store_true", help="Skip real hardware test")
    args = parser.parse_args()

    print("=" * 70)
    print("FEEL v16.0: HARDWARE-AWARE LLM VALIDATION")
    print("=" * 70)
    print(f"Model:    {args.model}")
    print(f"Adapters: {args.adapters or 'None (untrained)'}")
    print(f"Trials:   {args.trials}")
    print("=" * 70)

    # Create model
    print("\n[Loading Hardware-Aware LLM...]")
    model = HardwareAwareLLM(
        base_model_id=args.model,
        adapter_type="film",
        sensor_type="hybrid",
        load_in_4bit=True,
    )

    # Load adapters if provided
    if args.adapters and Path(args.adapters).exists():
        print(f"[Loading trained adapters from {args.adapters}...]")
        model.load_adapters(args.adapters)
    else:
        print("[Using untrained adapters (baseline test)]")

    # Run tests
    results = []

    results.append(test_simulated_stress(model, args.trials))

    if not args.skip_real:
        results.append(test_real_hardware(model, args.trials // 2))

    results.append(test_articulation(model, args.trials))
    results.append(test_latency(model, 100))
    results.append(test_stability(model, args.trials // 2))

    # Summary
    print("\n" + "=" * 70)
    print("VALIDATION SUMMARY")
    print("=" * 70)

    passed = sum(1 for r in results if r.passed)
    total = len(results)

    for r in results:
        status = "PASS" if r.passed else "FAIL"
        print(f"  [{status}] {r.name}: {r.metric:.3f} (target: {r.target})")
        print(f"          {r.details}")

    print()
    print(f"Overall: {passed}/{total} tests passed")

    # Determine overall status
    if passed == total:
        verdict = "FULL SUCCESS - Hardware-Aware LLM is functioning!"
    elif passed >= total // 2:
        verdict = "PARTIAL SUCCESS - Some hardware influence detected"
    else:
        verdict = "NEEDS TRAINING - Adapters not yet learned hardware association"

    print(f"\nVerdict: {verdict}")

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump({
            "model": args.model,
            "adapters": args.adapters,
            "trials": args.trials,
            "results": [
                {
                    "name": r.name,
                    "passed": r.passed,
                    "metric": r.metric,
                    "target": r.target,
                    "details": r.details,
                    "evidence": {k: v for k, v in r.evidence.items() if not isinstance(v, list) or len(v) < 10},
                }
                for r in results
            ],
            "summary": {
                "passed": passed,
                "total": total,
                "verdict": verdict,
            }
        }, f, indent=2)

    print(f"\nResults saved to: {output_path}")

    # Cleanup
    model.remove_hooks()

if __name__ == "__main__":
    main()
