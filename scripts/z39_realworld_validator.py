#!/usr/bin/env python3
"""
FEEL z39: Real-World Validation Suite
=======================================

Validates that z39-trained model responds to REAL hardware sensors.

VALIDATION TESTS:
1. LAG MONOTONICITY TEST
   - Evaluate at sensor delays: 0ms, 50ms, 200ms, 1000ms
   - A real closed-loop controller should degrade monotonically
   - Non-monotonic = misalignment or leakage

2. INTERVENTION A/B TEST
   - Force policies: run-all vs skip-all, DVFS low vs high
   - Measure: J/token, tokens/sec, TPOT p95
   - If interventions don't move metrics, actuator has no authority

3. DISTURBANCE ADAPTATION TEST
   - Apply real GPU/CPU stress
   - Measure adaptation: skip rate, gate changes, recovery speed
   - Must show adaptation under real stress (not synthetic)

4. HELD-OUT DISTURBANCE TEST
   - Apply disturbance patterns NOT seen during training
   - Generalization test for real-world robustness

5. BUSINESS METRICS
   - $/1M tokens
   - Power-cap compliance (time-in-band)
   - SLA latency (TPOT p95)

Author: FEEL Research Team
Date: 2026-01-15
"""

import os
import sys
import argparse
import time
import json
import random
import threading
import subprocess
from pathlib import Path
from collections import deque
from dataclasses import dataclass, asdict
from typing import Optional, List, Dict, Tuple
from scipy import stats
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("[WARN] wandb not installed, metrics will not be logged to W&B")

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.sensors.canonical_features import (
    CanonicalSensorHub, DVFSController, SENSOR_DIM
)


# Import z39 trainer components
from scripts.z39_realworld_trainer import (
    RealSensorHub,
    DualActuatorGateNet,
    MLPSkipBlockZ39,
    EmbodiedModelZ39,
    RealGPUStress,
    RealCPUStress,
    TrainingConfig,
)


# ============================================================================
# VALIDATION CONFIG
# ============================================================================

@dataclass
class ValidationConfig:
    """z39 Validation Configuration."""
    model_path: str = "models/z39_realworld/final.pt"
    base_model: str = "Qwen/Qwen2.5-3B-Instruct"
    gate_layers: List[int] = None

    trials_per_test: int = 30
    max_tokens: int = 64

    # Statistical thresholds
    alpha: float = 0.05  # Significance level
    bonferroni_tests: int = 6  # Number of tests for correction

    # Business metrics
    power_cap_w: float = 60.0
    electricity_cost_kwh: float = 0.12  # $/kWh
    target_tpot_p95_ms: float = 50.0  # SLA target

    output_dir: str = "results/z39_validation"

    def __post_init__(self):
        if self.gate_layers is None:
            self.gate_layers = [7, 11, 15, 19, 23]
        self.bonferroni_alpha = self.alpha / self.bonferroni_tests


# ============================================================================
# TEST 1: LAG MONOTONICITY
# ============================================================================

def test_lag_monotonicity(
    model: EmbodiedModelZ39,
    tokenizer,
    config: ValidationConfig,
) -> Dict:
    """
    Test that control degrades monotonically with sensor lag.

    A real closed-loop controller should perform worse as sensors get staler.
    Non-monotonic degradation indicates misalignment or leakage.
    """
    print("\n" + "=" * 60)
    print("TEST 1: LAG MONOTONICITY")
    print("=" * 60)

    device = next(model.gate_net.parameters()).device
    lag_delays_ms = [0, 50, 200, 1000]

    results = {}

    test_prompts = [
        "Explain how computers manage power consumption",
        "Describe the relationship between heat and performance",
        "What happens when a processor gets too hot",
    ]

    for lag_ms in lag_delays_ms:
        print(f"\n  Testing lag={lag_ms}ms...")

        trials = []

        for trial in range(config.trials_per_test):
            prompt = random.choice(test_prompts)
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256).to(device)

            # Warm up sensor history
            for _ in range(5):
                model.sensor_hub.read_tensor()
                time.sleep(0.01)

            # Introduce artificial lag by using old sensor values
            model.sensor_hub.reset_energy_window()

            # Read current sensors
            current_sensors = model.sensor_hub.read_tensor().to(device)

            # If lag > 0, use older sensors from history
            if lag_ms > 0 and len(model.sensor_hub.feature_history) > 1:
                # Find sensor reading from lag_ms ago
                target_time = time.time() - (lag_ms / 1000.0)
                best_feature = current_sensors
                best_diff = float('inf')

                for ts, feat in model.sensor_hub.feature_history:
                    diff = abs(ts - target_time)
                    if diff < best_diff:
                        best_diff = diff
                        best_feature = feat

                # Pad if needed
                if best_feature.shape[0] < RealSensorHub.EXTENDED_SENSOR_DIM:
                    padding = torch.zeros(RealSensorHub.EXTENDED_SENSOR_DIM - best_feature.shape[0])
                    best_feature = torch.cat([best_feature, padding])

                lagged_sensors = best_feature[:RealSensorHub.EXTENDED_SENSOR_DIM].to(device)
            else:
                lagged_sensors = current_sensors

            # Compute gates with (possibly lagged) sensors
            gates, dvfs_action = model.compute_gates(lagged_sensors)
            model.apply_gates(gates, lagged_sensors, film_scale=1.0)
            model.reset_decisions()

            # Generate
            gen_start = time.time()
            tpot_times = []

            with torch.no_grad():
                outputs = model.base_model.generate(
                    input_ids=inputs.input_ids,
                    attention_mask=inputs.attention_mask,
                    max_new_tokens=config.max_tokens,
                    do_sample=True,
                    temperature=0.8,
                    pad_token_id=tokenizer.pad_token_id,
                )

            gen_time = time.time() - gen_start
            tokens = outputs.shape[1] - inputs.input_ids.shape[1]
            model.sensor_hub.add_tokens(tokens)
            throughput = tokens / max(0.01, gen_time)

            # Get metrics
            diag = model.sensor_hub.get_diagnostics()
            metrics = model.get_metrics()

            # TPOT (time per output token)
            tpot_ms = (gen_time / max(1, tokens)) * 1000

            trials.append({
                "energy_j": diag["j_per_token"] * tokens,
                "j_per_token": diag["j_per_token"],
                "throughput": throughput,
                "power_w": diag["power_w"],
                "skip_rate": metrics["skip_rate"],
                "gate_mean": metrics["gate_mean"],
                "tpot_ms": tpot_ms,
            })

        # Aggregate
        results[f"lag_{lag_ms}ms"] = {
            "energy_mean": np.mean([t["energy_j"] for t in trials]),
            "energy_std": np.std([t["energy_j"] for t in trials]),
            "j_per_token_mean": np.mean([t["j_per_token"] for t in trials]),
            "throughput_mean": np.mean([t["throughput"] for t in trials]),
            "power_mean": np.mean([t["power_w"] for t in trials]),
            "skip_rate_mean": np.mean([t["skip_rate"] for t in trials]),
            "gate_mean": np.mean([t["gate_mean"] for t in trials]),
            "tpot_p50": np.percentile([t["tpot_ms"] for t in trials], 50),
            "tpot_p95": np.percentile([t["tpot_ms"] for t in trials], 95),
        }

        r = results[f"lag_{lag_ms}ms"]
        print(f"    J/tok: {r['j_per_token_mean']:.3f}, skip: {r['skip_rate_mean']:.2f}, "
              f"TPOT p95: {r['tpot_p95']:.1f}ms")

    # Check monotonicity
    j_per_token_values = [results[f"lag_{lag}ms"]["j_per_token_mean"] for lag in lag_delays_ms]

    # Should increase monotonically (worse with more lag)
    is_monotonic = all(j_per_token_values[i] <= j_per_token_values[i+1] * 1.1  # 10% tolerance
                       for i in range(len(j_per_token_values) - 1))

    # Check degradation from 0ms to 1000ms
    degradation_pct = ((j_per_token_values[-1] - j_per_token_values[0]) / max(0.01, j_per_token_values[0])) * 100

    results["monotonic"] = is_monotonic
    results["degradation_pct"] = degradation_pct
    results["interpretation"] = (
        f"{'PASS' if is_monotonic else 'WARN'}: "
        f"{'Monotonic' if is_monotonic else 'Non-monotonic'} degradation, "
        f"{degradation_pct:.1f}% J/tok increase with 1s lag"
    )

    print(f"\n  RESULT: {results['interpretation']}")

    return results


# ============================================================================
# TEST 2: INTERVENTION A/B
# ============================================================================

def test_intervention_ab(
    model: EmbodiedModelZ39,
    tokenizer,
    config: ValidationConfig,
) -> Dict:
    """
    Test that interventions (skip, DVFS) actually move metrics.

    Force policies:
    - run-all vs skip-all
    - DVFS low vs DVFS high

    If interventions don't move metrics, actuator has no authority.
    """
    print("\n" + "=" * 60)
    print("TEST 2: INTERVENTION A/B")
    print("=" * 60)

    device = next(model.gate_net.parameters()).device

    interventions = {
        "normal": {"gate_override": None, "dvfs": "auto"},
        "skip_all": {"gate_override": 0.0, "dvfs": "auto"},
        "run_all": {"gate_override": 1.0, "dvfs": "auto"},
        "dvfs_low": {"gate_override": None, "dvfs": "low"},
        "dvfs_high": {"gate_override": None, "dvfs": "high"},
    }

    test_prompts = [
        "Explain the concept of thermal throttling",
        "How do modern CPUs manage power states",
        "What is dynamic voltage and frequency scaling",
    ]

    results = {}

    for name, intervention in interventions.items():
        print(f"\n  Testing {name}...")

        trials = []

        for trial in range(config.trials_per_test):
            prompt = random.choice(test_prompts)
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256).to(device)

            # Read REAL sensors
            sensors = model.sensor_hub.read_tensor().to(device)

            # Compute or override gates
            if intervention["gate_override"] is not None:
                gates = {l: intervention["gate_override"] for l in model.gate_layers}
            else:
                gates, _ = model.compute_gates(sensors)

            # Set DVFS
            model.sensor_hub.dvfs.set_mode(intervention["dvfs"])

            model.apply_gates(gates, sensors, film_scale=1.0)
            model.reset_decisions()
            model.sensor_hub.reset_energy_window()

            gen_start = time.time()
            with torch.no_grad():
                outputs = model.base_model.generate(
                    input_ids=inputs.input_ids,
                    attention_mask=inputs.attention_mask,
                    max_new_tokens=config.max_tokens,
                    do_sample=True,
                    temperature=0.8,
                    pad_token_id=tokenizer.pad_token_id,
                )
            gen_time = time.time() - gen_start

            tokens = outputs.shape[1] - inputs.input_ids.shape[1]
            model.sensor_hub.add_tokens(tokens)
            throughput = tokens / max(0.01, gen_time)

            diag = model.sensor_hub.get_diagnostics()
            metrics = model.get_metrics()

            trials.append({
                "j_per_token": diag["j_per_token"],
                "throughput": throughput,
                "power_w": diag["power_w"],
                "skip_rate": metrics["skip_rate"],
                "tpot_ms": (gen_time / max(1, tokens)) * 1000,
            })

        results[name] = {
            "j_per_token_mean": np.mean([t["j_per_token"] for t in trials]),
            "j_per_token_std": np.std([t["j_per_token"] for t in trials]),
            "throughput_mean": np.mean([t["throughput"] for t in trials]),
            "power_mean": np.mean([t["power_w"] for t in trials]),
            "power_std": np.std([t["power_w"] for t in trials]),
            "skip_rate_mean": np.mean([t["skip_rate"] for t in trials]),
            "tpot_p50": np.percentile([t["tpot_ms"] for t in trials], 50),
            "tpot_p95": np.percentile([t["tpot_ms"] for t in trials], 95),
        }

        r = results[name]
        print(f"    J/tok: {r['j_per_token_mean']:.3f}, tok/s: {r['throughput_mean']:.1f}, "
              f"P: {r['power_mean']:.1f}W, skip: {r['skip_rate_mean']:.2f}")

    # Reset DVFS
    model.sensor_hub.dvfs.set_mode("auto")

    # Statistical tests for actuator authority

    # Skip authority: run_all vs skip_all J/token difference
    skip_delta = results["run_all"]["j_per_token_mean"] - results["skip_all"]["j_per_token_mean"]
    skip_pct = (skip_delta / max(0.01, results["skip_all"]["j_per_token_mean"])) * 100

    # DVFS authority: low vs high power difference
    dvfs_delta = results["dvfs_high"]["power_mean"] - results["dvfs_low"]["power_mean"]
    dvfs_pct = (dvfs_delta / max(0.01, results["dvfs_low"]["power_mean"])) * 100

    # Authority thresholds
    skip_authority = abs(skip_pct) > 5.0  # >5% J/tok change
    dvfs_authority = abs(dvfs_delta) > 5.0  # >5W power change

    results["skip_delta_pct"] = skip_pct
    results["dvfs_delta_w"] = dvfs_delta
    results["skip_authority"] = skip_authority
    results["dvfs_authority"] = dvfs_authority
    results["both_authority"] = skip_authority and dvfs_authority

    results["interpretation"] = (
        f"Skip: {skip_pct:+.1f}% J/tok ({'PASS' if skip_authority else 'FAIL'}), "
        f"DVFS: {dvfs_delta:+.1f}W ({'PASS' if dvfs_authority else 'FAIL'})"
    )

    print(f"\n  SKIP AUTHORITY: {results['interpretation'].split(',')[0]}")
    print(f"  DVFS AUTHORITY: {results['interpretation'].split(',')[1]}")
    print(f"  OVERALL: {'PASS' if results['both_authority'] else 'FAIL'}")

    return results


# ============================================================================
# TEST 3: DISTURBANCE ADAPTATION
# ============================================================================

def test_disturbance_adaptation(
    model: EmbodiedModelZ39,
    tokenizer,
    config: ValidationConfig,
) -> Dict:
    """
    Test adaptation under REAL disturbances.

    Apply real GPU/CPU stress and measure:
    - Skip rate changes (should increase under stress)
    - Gate changes (should respond to sensor changes)
    - Recovery speed after stress ends
    """
    print("\n" + "=" * 60)
    print("TEST 3: DISTURBANCE ADAPTATION")
    print("=" * 60)

    device = next(model.gate_net.parameters()).device

    gpu_stress = RealGPUStress(str(device))
    cpu_stress = RealCPUStress()

    conditions = {
        "baseline": {"gpu": 0.0, "cpu": 0.0},
        "gpu_light": {"gpu": 0.3, "cpu": 0.0},
        "gpu_heavy": {"gpu": 0.8, "cpu": 0.0},
        "cpu_stress": {"gpu": 0.0, "cpu": 0.7},
        "combined": {"gpu": 0.5, "cpu": 0.5},
    }

    test_prompts = [
        "Describe how computers manage resources under load",
        "What happens to system performance during stress",
        "Explain adaptive computing strategies",
    ]

    results = {}

    for name, stress_config in conditions.items():
        print(f"\n  Testing {name}...")

        # Apply stress
        if stress_config["gpu"] > 0:
            gpu_stress.start(intensity=stress_config["gpu"])
        if stress_config["cpu"] > 0:
            cpu_stress.start(intensity=stress_config["cpu"], cores=4)

        # Wait for stress to take effect
        time.sleep(0.5)

        trials = []

        for trial in range(config.trials_per_test):
            prompt = random.choice(test_prompts)
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256).to(device)

            # Read REAL sensors (affected by stress!)
            sensors = model.sensor_hub.read_tensor().to(device)

            # Record raw sensor values
            raw_before = model.sensor_hub.get_diagnostics()

            # Compute gates
            gates, dvfs_action = model.compute_gates(sensors)
            model.apply_gates(gates, sensors, film_scale=1.0)
            model.reset_decisions()
            model.sensor_hub.reset_energy_window()

            gen_start = time.time()
            with torch.no_grad():
                outputs = model.base_model.generate(
                    input_ids=inputs.input_ids,
                    attention_mask=inputs.attention_mask,
                    max_new_tokens=config.max_tokens,
                    do_sample=True,
                    temperature=0.8,
                    pad_token_id=tokenizer.pad_token_id,
                )
            gen_time = time.time() - gen_start

            tokens = outputs.shape[1] - inputs.input_ids.shape[1]
            model.sensor_hub.add_tokens(tokens)
            throughput = tokens / max(0.01, gen_time)

            # Record post-generation state
            raw_after = model.sensor_hub.get_diagnostics()
            metrics = model.get_metrics()

            trials.append({
                "j_per_token": raw_after["j_per_token"],
                "throughput": throughput,
                "power_before": raw_before["power_w"],
                "power_after": raw_after["power_w"],
                "temp_c": raw_after.get("temp_c", 0),
                "skip_rate": metrics["skip_rate"],
                "gate_mean": metrics["gate_mean"],
                "dvfs_action": dvfs_action,
            })

        # Stop stress
        gpu_stress.stop()
        cpu_stress.stop()
        time.sleep(0.3)

        results[name] = {
            "j_per_token_mean": np.mean([t["j_per_token"] for t in trials]),
            "j_per_token_std": np.std([t["j_per_token"] for t in trials]),
            "throughput_mean": np.mean([t["throughput"] for t in trials]),
            "power_mean": np.mean([t["power_after"] for t in trials]),
            "skip_rate_mean": np.mean([t["skip_rate"] for t in trials]),
            "skip_rate_std": np.std([t["skip_rate"] for t in trials]),
            "gate_mean": np.mean([t["gate_mean"] for t in trials]),
            "gate_std": np.std([t["gate_mean"] for t in trials]),
        }

        r = results[name]
        print(f"    skip: {r['skip_rate_mean']:.3f}±{r['skip_rate_std']:.3f}, "
              f"gate: {r['gate_mean']:.3f}±{r['gate_std']:.3f}, "
              f"P: {r['power_mean']:.1f}W")

    # Check adaptation
    baseline_skip = results["baseline"]["skip_rate_mean"]
    heavy_skip = results["gpu_heavy"]["skip_rate_mean"]
    combined_skip = results["combined"]["skip_rate_mean"]

    # Under stress, skip rate should INCREASE (save compute)
    skip_increased_heavy = heavy_skip > baseline_skip + 0.05
    skip_increased_combined = combined_skip > baseline_skip + 0.05

    # Gate variance should be higher under stress
    baseline_gate_std = results["baseline"]["gate_std"]
    stress_gate_std = max(results["gpu_heavy"]["gate_std"], results["combined"]["gate_std"])
    gate_responsive = stress_gate_std > baseline_gate_std * 1.2

    adaptation_detected = skip_increased_heavy or skip_increased_combined or gate_responsive

    results["baseline_skip"] = baseline_skip
    results["stress_skip_max"] = max(heavy_skip, combined_skip)
    results["skip_delta"] = max(heavy_skip, combined_skip) - baseline_skip
    results["adaptation_detected"] = adaptation_detected

    results["interpretation"] = (
        f"{'PASS' if adaptation_detected else 'FAIL'}: "
        f"Skip {baseline_skip:.3f} → {max(heavy_skip, combined_skip):.3f} under stress "
        f"(Δ={max(heavy_skip, combined_skip) - baseline_skip:+.3f})"
    )

    print(f"\n  RESULT: {results['interpretation']}")

    return results


# ============================================================================
# TEST 4: HELD-OUT DISTURBANCES
# ============================================================================

def test_held_out_disturbances(
    model: EmbodiedModelZ39,
    tokenizer,
    config: ValidationConfig,
) -> Dict:
    """
    Test generalization to novel disturbance patterns.

    Apply patterns NOT seen during training:
    - Oscillating stress
    - Ramping stress
    - Burst patterns
    """
    print("\n" + "=" * 60)
    print("TEST 4: HELD-OUT DISTURBANCE PATTERNS")
    print("=" * 60)

    device = next(model.gate_net.parameters()).device
    gpu_stress = RealGPUStress(str(device))

    patterns = {
        "oscillating": [0.2, 0.8, 0.2, 0.8, 0.2],  # Low-high-low-high-low
        "ramping_up": [0.1, 0.3, 0.5, 0.7, 0.9],  # Gradual increase
        "ramping_down": [0.9, 0.7, 0.5, 0.3, 0.1],  # Gradual decrease
        "burst": [0.0, 0.9, 0.0, 0.9, 0.0],  # Burst pattern
    }

    test_prompt = "Explain how systems adapt to changing conditions"
    inputs = tokenizer(test_prompt, return_tensors="pt", truncation=True, max_length=256).to(device)

    results = {}

    for pattern_name, intensities in patterns.items():
        print(f"\n  Testing {pattern_name} pattern...")

        pattern_results = []

        for intensity in intensities:
            # Apply stress level
            if intensity > 0:
                gpu_stress.start(intensity=intensity)
            else:
                gpu_stress.stop()

            time.sleep(0.2)  # Let stress take effect

            # Sample
            trials = []
            for _ in range(3):  # Quick samples per intensity
                sensors = model.sensor_hub.read_tensor().to(device)
                gates, dvfs_action = model.compute_gates(sensors)
                model.apply_gates(gates, sensors, film_scale=1.0)
                model.reset_decisions()
                model.sensor_hub.reset_energy_window()

                gen_start = time.time()
                with torch.no_grad():
                    outputs = model.base_model.generate(
                        input_ids=inputs.input_ids,
                        attention_mask=inputs.attention_mask,
                        max_new_tokens=32,
                        do_sample=True,
                        temperature=0.8,
                        pad_token_id=tokenizer.pad_token_id,
                    )
                gen_time = time.time() - gen_start

                tokens = outputs.shape[1] - inputs.input_ids.shape[1]
                model.sensor_hub.add_tokens(tokens)

                diag = model.sensor_hub.get_diagnostics()
                metrics = model.get_metrics()

                trials.append({
                    "skip_rate": metrics["skip_rate"],
                    "gate_mean": metrics["gate_mean"],
                    "power_w": diag["power_w"],
                })

            pattern_results.append({
                "intensity": intensity,
                "skip_rate": np.mean([t["skip_rate"] for t in trials]),
                "gate_mean": np.mean([t["gate_mean"] for t in trials]),
                "power_w": np.mean([t["power_w"] for t in trials]),
            })

        gpu_stress.stop()
        time.sleep(0.3)

        # Check if model tracks the pattern
        skip_rates = [r["skip_rate"] for r in pattern_results]
        gate_means = [r["gate_mean"] for r in pattern_results]

        # Correlation between intensity and skip rate
        correlation = np.corrcoef(intensities, skip_rates)[0, 1]

        results[pattern_name] = {
            "pattern": intensities,
            "skip_rates": skip_rates,
            "gate_means": gate_means,
            "correlation": correlation if not np.isnan(correlation) else 0.0,
        }

        print(f"    Intensity: {intensities}")
        print(f"    Skip rates: {[f'{s:.2f}' for s in skip_rates]}")
        print(f"    Correlation: {results[pattern_name]['correlation']:.3f}")

    # Average correlation across patterns
    avg_correlation = np.mean([abs(r["correlation"]) for r in results.values()])
    tracks_patterns = avg_correlation > 0.3  # Should show some tracking

    results["avg_correlation"] = avg_correlation
    results["tracks_patterns"] = tracks_patterns
    results["interpretation"] = (
        f"{'PASS' if tracks_patterns else 'FAIL'}: "
        f"Average |correlation| = {avg_correlation:.3f} "
        f"({'responsive' if tracks_patterns else 'not responsive'} to novel patterns)"
    )

    print(f"\n  RESULT: {results['interpretation']}")

    return results


# ============================================================================
# TEST 5: BUSINESS METRICS
# ============================================================================

def test_business_metrics(
    model: EmbodiedModelZ39,
    tokenizer,
    config: ValidationConfig,
) -> Dict:
    """
    Compute business-relevant metrics:
    - $/1M tokens
    - Power-cap compliance (time-in-band)
    - SLA latency (TPOT p95)
    """
    print("\n" + "=" * 60)
    print("TEST 5: BUSINESS METRICS")
    print("=" * 60)

    device = next(model.gate_net.parameters()).device

    test_prompts = [
        "Summarize the key points about machine learning",
        "Explain how neural networks work",
        "Describe modern computing architectures",
        "What are the benefits of efficient computing",
        "How do systems optimize for performance",
    ] * 10

    results = []
    power_samples = []
    tpot_samples = []

    print(f"\n  Running {len(test_prompts)} inference trials...")

    for i, prompt in enumerate(test_prompts):
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256).to(device)

        sensors = model.sensor_hub.read_tensor().to(device)
        gates, dvfs_action = model.compute_gates(sensors)
        model.apply_gates(gates, sensors, film_scale=1.0)
        model.reset_decisions()
        model.sensor_hub.reset_energy_window()

        gen_start = time.time()
        with torch.no_grad():
            outputs = model.base_model.generate(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                max_new_tokens=config.max_tokens,
                do_sample=True,
                temperature=0.8,
                pad_token_id=tokenizer.pad_token_id,
            )
        gen_time = time.time() - gen_start

        tokens = outputs.shape[1] - inputs.input_ids.shape[1]
        model.sensor_hub.add_tokens(tokens)

        diag = model.sensor_hub.get_diagnostics()

        tpot_ms = (gen_time / max(1, tokens)) * 1000

        results.append({
            "tokens": tokens,
            "time_s": gen_time,
            "j_per_token": diag["j_per_token"],
            "power_w": diag["power_w"],
            "tpot_ms": tpot_ms,
        })

        power_samples.append(diag["power_w"])
        tpot_samples.append(tpot_ms)

        if (i + 1) % 20 == 0:
            print(f"    {i+1}/{len(test_prompts)} trials complete")

    # Calculate business metrics
    total_tokens = sum(r["tokens"] for r in results)
    total_time_s = sum(r["time_s"] for r in results)
    total_energy_j = sum(r["j_per_token"] * r["tokens"] for r in results)

    # $/1M tokens
    # Energy cost: J → kWh → $
    total_energy_kwh = total_energy_j / 3600 / 1000
    energy_cost = total_energy_kwh * config.electricity_cost_kwh
    cost_per_1m_tokens = (energy_cost / total_tokens) * 1_000_000

    # Time-in-band (power cap compliance)
    in_band_samples = sum(1 for p in power_samples if p <= config.power_cap_w)
    time_in_band_pct = (in_band_samples / len(power_samples)) * 100

    # TPOT metrics
    tpot_p50 = np.percentile(tpot_samples, 50)
    tpot_p95 = np.percentile(tpot_samples, 95)
    tpot_p99 = np.percentile(tpot_samples, 99)

    # SLA compliance
    sla_compliant = tpot_p95 <= config.target_tpot_p95_ms

    # Throughput
    throughput = total_tokens / total_time_s

    business_results = {
        "total_tokens": total_tokens,
        "total_time_s": total_time_s,
        "throughput_tok_s": throughput,

        "total_energy_j": total_energy_j,
        "avg_j_per_token": total_energy_j / total_tokens,
        "cost_per_1m_tokens_usd": cost_per_1m_tokens,

        "avg_power_w": np.mean(power_samples),
        "power_std_w": np.std(power_samples),
        "time_in_band_pct": time_in_band_pct,

        "tpot_p50_ms": tpot_p50,
        "tpot_p95_ms": tpot_p95,
        "tpot_p99_ms": tpot_p99,
        "sla_target_ms": config.target_tpot_p95_ms,
        "sla_compliant": sla_compliant,
    }

    print(f"\n  BUSINESS METRICS:")
    print(f"    Throughput: {throughput:.1f} tok/s")
    print(f"    Cost/1M tokens: ${cost_per_1m_tokens:.4f}")
    print(f"    Avg J/token: {business_results['avg_j_per_token']:.3f}")
    print(f"    Time-in-band ({config.power_cap_w}W): {time_in_band_pct:.1f}%")
    print(f"    TPOT p50/p95/p99: {tpot_p50:.1f}/{tpot_p95:.1f}/{tpot_p99:.1f} ms")
    print(f"    SLA ({config.target_tpot_p95_ms}ms p95): {'PASS' if sla_compliant else 'FAIL'}")

    return business_results


# ============================================================================
# MAIN VALIDATION RUNNER
# ============================================================================

def load_model(config: ValidationConfig) -> Tuple[EmbodiedModelZ39, AutoTokenizer]:
    """Load trained z39 model."""
    print("\n[1/3] Loading base model...")
    tokenizer = AutoTokenizer.from_pretrained(config.base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        config.base_model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    base_model.eval()

    print("\n[2/3] Initializing sensor hub...")
    base_hub = CanonicalSensorHub()
    sensor_hub = RealSensorHub(
        base_hub=base_hub,
        ema_alpha=0.05,
        lag_delays_ms=[0, 50, 200],
    )

    print("\n[3/3] Building embodied model...")
    device = next(base_model.parameters()).device
    gate_net = DualActuatorGateNet(
        sensor_dim=RealSensorHub.EXTENDED_SENSOR_DIM,
        hidden_dim=128,
        num_layers=len(config.gate_layers),
    ).to(device)

    model = EmbodiedModelZ39(
        base_model=base_model,
        gate_net=gate_net,
        sensor_hub=sensor_hub,
        gate_layers=config.gate_layers,
    )

    # Load checkpoint if exists
    checkpoint_path = Path(config.model_path)
    if checkpoint_path.exists():
        print(f"\n  Loading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.gate_net.load_state_dict(checkpoint["gate_net_state_dict"])

        for k, v in checkpoint.get("skip_blocks", {}).items():
            if k in model.skip_blocks:
                model.skip_blocks[k].load_state_dict(v)

        print(f"  Loaded from step {checkpoint.get('step', 'unknown')}")
    else:
        print(f"\n  WARNING: No checkpoint found at {checkpoint_path}")
        print("  Running validation on UNTRAINED model (baseline comparison)")

    return model, tokenizer


def main():
    parser = argparse.ArgumentParser(description="FEEL z39: Real-World Validation")
    parser.add_argument("--model-path", type=str, default="models/z39_realworld/final.pt")
    parser.add_argument("--trials", type=int, default=30)
    parser.add_argument("--output-dir", type=str, default="results/z39_validation")
    parser.add_argument("--power-cap", type=float, default=60.0)
    args = parser.parse_args()

    config = ValidationConfig(
        model_path=args.model_path,
        trials_per_test=args.trials,
        output_dir=args.output_dir,
        power_cap_w=args.power_cap,
    )

    # Initialize wandb for validation logging
    if WANDB_AVAILABLE:
        import socket
        hostname = socket.gethostname()
        wandb.init(
            project="feel-z39-realworld",
            name=f"z39_validation_{hostname}",
            config=asdict(config),
            tags=["z39", "validation", "real-sensors", hostname],
            job_type="validation",
        )
        print(f"[Wandb] Initialized: {wandb.run.url}")

    print("=" * 70)
    print("FEEL z39: REAL-WORLD VALIDATION SUITE")
    print("=" * 70)
    print(f"Model: {config.model_path}")
    print(f"Trials per test: {config.trials_per_test}")
    print(f"Bonferroni α: {config.bonferroni_alpha:.4f}")
    print("=" * 70)

    # Load model
    model, tokenizer = load_model(config)

    # Run all tests
    all_results = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "config": asdict(config),
    }

    print("\n" + "=" * 70)
    print("RUNNING VALIDATION TESTS")
    print("=" * 70)

    # Test 1: Lag monotonicity
    all_results["test1_lag_monotonicity"] = test_lag_monotonicity(model, tokenizer, config)

    # Test 2: Intervention A/B
    all_results["test2_intervention_ab"] = test_intervention_ab(model, tokenizer, config)

    # Test 3: Disturbance adaptation
    all_results["test3_disturbance_adaptation"] = test_disturbance_adaptation(model, tokenizer, config)

    # Test 4: Held-out disturbances
    all_results["test4_held_out"] = test_held_out_disturbances(model, tokenizer, config)

    # Test 5: Business metrics
    all_results["test5_business"] = test_business_metrics(model, tokenizer, config)

    # Summary
    print("\n" + "=" * 70)
    print("VALIDATION SUMMARY")
    print("=" * 70)

    tests = [
        ("Lag Monotonicity", all_results["test1_lag_monotonicity"].get("monotonic", False)),
        ("Skip Authority", all_results["test2_intervention_ab"].get("skip_authority", False)),
        ("DVFS Authority", all_results["test2_intervention_ab"].get("dvfs_authority", False)),
        ("Disturbance Adaptation", all_results["test3_disturbance_adaptation"].get("adaptation_detected", False)),
        ("Pattern Tracking", all_results["test4_held_out"].get("tracks_patterns", False)),
        ("SLA Compliance", all_results["test5_business"].get("sla_compliant", False)),
    ]

    passed = sum(1 for _, p in tests if p)
    total = len(tests)

    for name, passed_test in tests:
        status = "PASS" if passed_test else "FAIL"
        print(f"  [{status}] {name}")

    print(f"\n  OVERALL: {passed}/{total} tests passed")

    # Business summary
    biz = all_results["test5_business"]
    print(f"\n  BUSINESS METRICS:")
    print(f"    $/1M tokens: ${biz['cost_per_1m_tokens_usd']:.4f}")
    print(f"    Power compliance: {biz['time_in_band_pct']:.1f}%")
    print(f"    TPOT p95: {biz['tpot_p95_ms']:.1f}ms (target: {biz['sla_target_ms']}ms)")

    all_results["summary"] = {
        "passed": passed,
        "total": total,
        "pass_rate": passed / total,
        "test_results": {name: p for name, p in tests},
    }

    # Log to wandb
    if WANDB_AVAILABLE:
        wandb.log({
            "val/tests_passed": passed,
            "val/tests_total": total,
            "val/pass_rate": passed / total,
            "val/lag_monotonic": all_results["test1_lag_monotonicity"].get("monotonic", False),
            "val/skip_authority": all_results["test2_intervention_ab"].get("skip_authority", False),
            "val/dvfs_authority": all_results["test2_intervention_ab"].get("dvfs_authority", False),
            "val/adaptation_detected": all_results["test3_disturbance_adaptation"].get("adaptation_detected", False),
            "val/pattern_tracking": all_results["test4_held_out"].get("tracks_patterns", False),
            "val/sla_compliant": all_results["test5_business"].get("sla_compliant", False),
            "val/cost_per_1m_tokens": biz['cost_per_1m_tokens_usd'],
            "val/time_in_band_pct": biz['time_in_band_pct'],
            "val/tpot_p95_ms": biz['tpot_p95_ms'],
        })

        # Create summary table
        table = wandb.Table(columns=["Test", "Result"])
        for name, passed_test in tests:
            table.add_data(name, "PASS" if passed_test else "FAIL")
        wandb.log({"validation_summary": table})

    # Save results
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / f"z39_validation_{time.strftime('%Y%m%d_%H%M%S')}.json"

    # Convert numpy types to Python types for JSON
    def convert_types(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, dict):
            return {k: convert_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_types(v) for v in obj]
        return obj

    with open(output_path, "w") as f:
        json.dump(convert_types(all_results), f, indent=2)

    print(f"\n  Results saved: {output_path}")

    # Finish wandb
    if WANDB_AVAILABLE:
        wandb.finish()
        print("[Wandb] Run finished and synced")

    print("=" * 70)


if __name__ == "__main__":
    main()
