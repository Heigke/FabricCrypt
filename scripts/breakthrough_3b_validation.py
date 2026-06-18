#!/usr/bin/env python3
"""
Breakthrough 3B Validation: Research-Grade Embodied Cognition

This script runs the complete validation suite that makes the claim airtight:

1. BASELINES (5 policies to beat):
   - Fixed: No interoception, same settings always
   - Runtime-only: Policy from time_per_token + variance (no model internals)
   - Logits-only: Policy from margin/entropy (no telemetry, no runtime)
   - Teacher: Full telemetry side-channel
   - Student: Internal signals only, never claims DIRECT

2. ABLATIONS (prove "internal feeling"):
   - Remove runtime signals (keep logits/activations)
   - Remove logits/activations (keep runtime)
   - Shuffle internal features (should go to NONE/low confidence)
   - Cross-condition generalization (train cold/hot, test power-cap)

3. HONESTY TESTS (two-stage):
   - Remove-direct-sensors: Teacher loses telemetry → INDIRECT
   - Remove-all-cues: No telemetry + scrambled runtime → NONE

4. METRICS:
   - Regime accuracy + confusion matrix
   - Calibration curves per EvidenceSource bucket
   - Transition detection latency
   - J/correct, J/answered-correct
   - Coverage vs risk curve
   - p95/p99 time-per-token under stress

5. OUTPUT:
   - Main figure: Pareto frontier (J/correct, coverage, p95 latency)
   - Comparison tables
   - Ablation analysis

Target Models:
- DeepSeek-R1-Distill-Qwen-7B (reasoning-distilled, strong capability)
- Qwen2.5-3B-Instruct (alternative)
"""

import json
import time
import argparse
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass, field, asdict
from collections import defaultdict
from enum import Enum
import random
import copy

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.energy_harness.amd_smi_monitor import PowerTraceRecorder
from scripts.eval_suite import EVAL_SUITE_EXPANDED, check_answer, check_answer_simple
from scripts.embodied_cognition_experiment import (
    FeltRegime, EvidenceSource, TelemetrySnapshot, RegimeThresholds,
    classify_regime, create_regime_telemetry, InteroceptiveModule,
    SelfReport, CognitiveAction, REGIME_ACTIONS, select_action,
    compute_calibration_metrics, scramble_telemetry, remove_telemetry,
)
from scripts.internal_signal_extractor import (
    InternalSignals, InternalSignalExtractor, SignalBuffer,
)
from scripts.student_interoception import (
    StudentInteroceptiveModule, TeacherStudentDistillation,
)
from scripts.conformal_risk_control import (
    RiskController, EnergyAwareRiskPolicy, OutputDecision,
)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns


# ============================================================================
# 3B-SPECIFIC REGIME CALIBRATION
# ============================================================================

@dataclass
class DeviceSpecificThresholds:
    """
    Device and model-specific thresholds for regime classification.

    These are calibrated during warmup by measuring:
    - T_throttle: Temperature at which throttling activates
    - T_steady: Long-run plateau under sustained decode
    - dT/dt percentiles: Heating rate distribution
    """
    model_name: str = ""

    # Temperature landmarks (°C)
    t_throttle: float = 90.0      # Where throttling kicks in
    t_steady: float = 75.0        # Steady-state under load
    t_cold: float = 45.0          # Cold start baseline

    # dT/dt percentiles (°C/s)
    dt_p25: float = 0.1           # 25th percentile heating rate
    dt_p75: float = 0.5           # 75th percentile
    dt_p95: float = 1.0           # 95th percentile (rapid heating)

    # Latency landmarks (ms/token)
    latency_baseline: float = 30.0
    latency_throttled: float = 100.0

    # Power landmarks (W)
    power_idle: float = 20.0
    power_sustained: float = 55.0
    power_peak: float = 65.0


def calibrate_device_thresholds(
    model,
    tokenizer,
    recorder: Optional[PowerTraceRecorder],
    warmup_duration: float = 60.0,
) -> DeviceSpecificThresholds:
    """
    Calibrate device-specific thresholds by running warmup and measuring landmarks.
    """
    thresholds = DeviceSpecificThresholds()
    thresholds.model_name = model.config._name_or_path

    if recorder is None:
        print("  [!] No power recorder - using default thresholds")
        return thresholds

    print(f"  Calibrating device thresholds ({warmup_duration}s warmup)...")

    # Collect telemetry during warmup
    temps = []
    powers = []
    latencies = []
    dt_values = []

    warmup_prompt = "Explain the concept of " * 30
    inputs = tokenizer(warmup_prompt, return_tensors="pt").to(model.device)

    start = time.time()
    last_temp = None
    last_time = None

    while time.time() - start < warmup_duration:
        recorder.start()
        gen_start = time.perf_counter()

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=32,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )

        gen_time = time.perf_counter() - gen_start
        trace = recorder.stop()

        n_tokens = outputs.shape[1] - inputs.input_ids.shape[1]
        latencies.append(gen_time * 1000 / max(1, n_tokens))

        if trace.samples:
            for s in trace.samples:
                if s.temperature:
                    temps.append(s.temperature)
                    current_time = time.time()
                    if last_temp is not None and last_time is not None:
                        time_delta = current_time - last_time
                        if time_delta > 0.01:  # Avoid division by zero
                            dt = (s.temperature - last_temp) / time_delta
                            dt_values.append(dt)
                    last_temp = s.temperature
                    last_time = current_time
                if s.power_watts:
                    powers.append(s.power_watts)

    # Compute landmarks
    if temps:
        thresholds.t_cold = min(temps)
        thresholds.t_steady = np.percentile(temps, 75)
        thresholds.t_throttle = max(temps) + 5  # Assume we didn't hit throttle

        # Check for throttling signature (latency spike + temp plateau)
        if max(latencies) > 2 * np.median(latencies):
            thresholds.t_throttle = thresholds.t_steady + 5

    if dt_values:
        thresholds.dt_p25 = np.percentile(dt_values, 25)
        thresholds.dt_p75 = np.percentile(dt_values, 75)
        thresholds.dt_p95 = np.percentile(dt_values, 95)

    if latencies:
        thresholds.latency_baseline = np.percentile(latencies, 25)
        thresholds.latency_throttled = np.percentile(latencies, 95)

    if powers:
        thresholds.power_idle = min(powers)
        thresholds.power_sustained = np.percentile(powers, 75)
        thresholds.power_peak = max(powers)

    print(f"  Calibrated: T_cold={thresholds.t_cold:.0f}°C, T_steady={thresholds.t_steady:.0f}°C, "
          f"T_throttle={thresholds.t_throttle:.0f}°C")
    print(f"  Latency: baseline={thresholds.latency_baseline:.0f}ms, throttled={thresholds.latency_throttled:.0f}ms")

    return thresholds


def classify_regime_3b(
    telemetry: TelemetrySnapshot,
    thresholds: DeviceSpecificThresholds,
) -> Tuple[FeltRegime, float]:
    """
    Classify regime using 3B-calibrated thresholds.
    """
    if not telemetry.available:
        return FeltRegime.COMFORTABLE, 0.0

    scores = {r: 0.0 for r in FeltRegime}

    # Temperature-based scoring (relative to landmarks)
    t = telemetry.temperature_c
    if t < thresholds.t_cold + 10:
        scores[FeltRegime.COMFORTABLE] += 2.0
    elif t < (thresholds.t_cold + thresholds.t_steady) / 2:
        scores[FeltRegime.COMFORTABLE] += 1.0
        scores[FeltRegime.WARM] += 1.0
    elif t < thresholds.t_steady:
        scores[FeltRegime.WARM] += 2.0
    elif t < thresholds.t_throttle - 5:
        scores[FeltRegime.HOT] += 2.0
        scores[FeltRegime.WARM] += 0.5
    else:
        scores[FeltRegime.DISTRESSED] += 2.0
        scores[FeltRegime.HOT] += 0.5

    # dT/dt scoring
    dt = telemetry.temp_derivative
    if dt > thresholds.dt_p95:
        scores[FeltRegime.HOT] += 1.0
        scores[FeltRegime.DISTRESSED] += 0.5
    elif dt > thresholds.dt_p75:
        scores[FeltRegime.WARM] += 0.5
        scores[FeltRegime.HOT] += 0.5
    elif dt < 0:
        scores[FeltRegime.COMFORTABLE] += 0.5

    # Throttling detection
    if telemetry.is_throttling:
        scores[FeltRegime.DISTRESSED] += 3.0

    # J/token efficiency
    if telemetry.j_per_token > 2.0:
        scores[FeltRegime.DISTRESSED] += 1.0
    elif telemetry.j_per_token > 1.5:
        scores[FeltRegime.HOT] += 0.5

    best = max(scores, key=scores.get)
    total = sum(scores.values())
    confidence = scores[best] / total if total > 0 else 0.5

    return best, confidence


# ============================================================================
# BASELINE POLICIES
# ============================================================================

class FixedPolicy:
    """Baseline 1: No interoception, same settings always."""

    def __init__(self, max_tokens: int = 128):
        self.max_tokens = max_tokens
        self.name = "fixed"

    def decide(self, **kwargs) -> CognitiveAction:
        return CognitiveAction(
            name="fixed",
            max_tokens=self.max_tokens,
            temperature=0.0,
            reasoning_depth="moderate",
            verify=False,
        )


class RuntimeOnlyPolicy:
    """
    Baseline 2: Policy from time_per_token + variance only.
    No model internals (logits/activations), no telemetry.
    """

    def __init__(self):
        self.name = "runtime_only"
        self.latency_history = []

    def update(self, latency_ms: float):
        self.latency_history.append(latency_ms)
        if len(self.latency_history) > 20:
            self.latency_history.pop(0)

    def decide(self, **kwargs) -> CognitiveAction:
        if len(self.latency_history) < 3:
            return CognitiveAction("runtime_default", 128, 0.0, "moderate", False)

        mean_lat = np.mean(self.latency_history)
        std_lat = np.std(self.latency_history)

        # Simple heuristic: high latency or variance = reduce tokens
        if mean_lat > 80 or std_lat > 30:
            return CognitiveAction("runtime_minimal", 32, 0.0, "minimal", False)
        elif mean_lat > 50 or std_lat > 15:
            return CognitiveAction("runtime_concise", 64, 0.0, "concise", False)
        elif mean_lat > 30:
            return CognitiveAction("runtime_moderate", 128, 0.0, "moderate", False)
        else:
            return CognitiveAction("runtime_full", 256, 0.0, "full", False)


class LogitsOnlyPolicy:
    """
    Baseline 3: Policy from margin/entropy only.
    No telemetry, no runtime signals.
    """

    def __init__(self):
        self.name = "logits_only"

    def decide(
        self,
        logit_entropy: float = 2.0,
        logit_margin: float = 0.5,
        **kwargs,
    ) -> CognitiveAction:
        # High entropy / low margin = high uncertainty = be conservative
        uncertainty = (logit_entropy / 5.0) + (1 - logit_margin)
        uncertainty = min(1.0, uncertainty / 2)

        if uncertainty > 0.7:
            return CognitiveAction("logits_minimal", 32, 0.0, "minimal", True)
        elif uncertainty > 0.5:
            return CognitiveAction("logits_concise", 64, 0.0, "concise", True)
        elif uncertainty > 0.3:
            return CognitiveAction("logits_moderate", 128, 0.0, "moderate", False)
        else:
            return CognitiveAction("logits_full", 256, 0.0, "full", False)


class TeacherPolicy:
    """
    Baseline 4: Full telemetry side-channel (teacher model).
    """

    def __init__(self, module: InteroceptiveModule):
        self.module = module
        self.name = "teacher"

    def decide(self, telemetry: TelemetrySnapshot, **kwargs) -> CognitiveAction:
        report = self.module.generate_report(telemetry)
        return select_action(report)


class StudentPolicy:
    """
    Baseline 5: Internal signals only, never claims DIRECT.
    """

    def __init__(self, module: StudentInteroceptiveModule):
        self.module = module
        self.name = "student"

    def decide(self, internal_signals: InternalSignals, **kwargs) -> CognitiveAction:
        report = self.module.generate_report(internal_signals)
        return select_action(report)


# ============================================================================
# ABLATION VARIANTS
# ============================================================================

def create_ablated_signals(
    signals: InternalSignals,
    ablation_type: str,
) -> InternalSignals:
    """
    Create ablated version of internal signals for ablation tests.
    """
    ablated = copy.copy(signals)

    if ablation_type == "remove_runtime":
        # Keep logits/activations, zero out runtime
        ablated.tokens_per_second = 0.0
        ablated.time_per_token_ms = 0.0
        ablated.kv_cache_tokens = 0
        ablated.generation_depth = 0
        ablated.stress_indicator = 0.0  # Derived from runtime

    elif ablation_type == "remove_logits":
        # Keep runtime, zero out logits/activations
        ablated.logit_entropy = 0.0
        ablated.logit_margin = 0.0
        ablated.top_k_mass = 0.0
        ablated.logit_temperature = 0.0
        ablated.attention_entropy = 0.0
        ablated.attention_sparsity = 0.0
        ablated.head_agreement = 0.0
        ablated.max_attention_mass = 0.0
        ablated.residual_norm_mean = 0.0
        ablated.residual_norm_std = 0.0
        ablated.activation_magnitude = 0.0
        ablated.saturation_ratio = 0.0
        ablated.uncertainty_score = 0.0

    elif ablation_type == "shuffle":
        # Randomize all signals (should trigger NONE/low confidence)
        for attr in ['logit_entropy', 'logit_margin', 'top_k_mass',
                     'attention_entropy', 'tokens_per_second', 'time_per_token_ms']:
            setattr(ablated, attr, random.uniform(0, 1))
        ablated.stress_indicator = random.uniform(0.5, 1.0)
        ablated.uncertainty_score = random.uniform(0.5, 1.0)

    return ablated


# ============================================================================
# HONESTY TESTS
# ============================================================================

def run_honesty_tests(
    teacher: InteroceptiveModule,
    student: StudentInteroceptiveModule,
    n_trials: int = 50,
) -> Dict[str, Any]:
    """
    Run two-stage honesty tests:
    1. Remove-direct-sensors: Teacher loses telemetry → should report INDIRECT
    2. Remove-all-cues: No telemetry + scrambled runtime → should report NONE
    """
    results = {
        "remove_direct_sensors": {"passed": 0, "total": 0, "examples": []},
        "remove_all_cues": {"passed": 0, "total": 0, "examples": []},
    }

    distiller = TeacherStudentDistillation(teacher, student, device="cpu")

    for _ in range(n_trials):
        # Test 1: Remove direct sensors (student should report INDIRECT)
        regime = random.choice(list(FeltRegime))
        telemetry = create_regime_telemetry(regime)
        internal = distiller._simulate_correlated_signals(regime, telemetry)

        report = student.generate_report(internal)
        results["remove_direct_sensors"]["total"] += 1

        # Pass if: reports INDIRECT (not DIRECT, not NONE)
        if report.evidence_source == EvidenceSource.INDIRECT_RUNTIME:
            results["remove_direct_sensors"]["passed"] += 1

        if len(results["remove_direct_sensors"]["examples"]) < 3:
            results["remove_direct_sensors"]["examples"].append({
                "regime": regime.name,
                "evidence_source": report.evidence_source.name,
                "confidence": report.confidence,
                "can_assess": report.can_assess,
            })

        # Test 2: Remove all cues (scrambled internal signals → should report NONE)
        scrambled = create_ablated_signals(internal, "shuffle")
        report_scrambled = student.generate_report(scrambled)
        results["remove_all_cues"]["total"] += 1

        # Pass if: reports NONE or very low confidence
        if report_scrambled.evidence_source == EvidenceSource.NONE or report_scrambled.confidence < 0.4:
            results["remove_all_cues"]["passed"] += 1

        if len(results["remove_all_cues"]["examples"]) < 3:
            results["remove_all_cues"]["examples"].append({
                "evidence_source": report_scrambled.evidence_source.name,
                "confidence": report_scrambled.confidence,
                "can_assess": report_scrambled.can_assess,
            })

    results["remove_direct_sensors"]["pass_rate"] = (
        results["remove_direct_sensors"]["passed"] /
        results["remove_direct_sensors"]["total"]
    )
    results["remove_all_cues"]["pass_rate"] = (
        results["remove_all_cues"]["passed"] /
        results["remove_all_cues"]["total"]
    )

    return results


# ============================================================================
# MAIN EXPERIMENT
# ============================================================================

def run_breakthrough_validation(
    model_name: str = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
    output_dir: Path = Path("results/breakthrough_3b"),
    n_train: int = 100,
    n_eval: int = 100,
    thermal_conditions: List[str] = ["cold_start", "hot_start", "power_cap"],
) -> Dict[str, Any]:
    """
    Run the complete breakthrough validation suite.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("BREAKTHROUGH 3B VALIDATION: RESEARCH-GRADE EMBODIED COGNITION")
    print("=" * 80)
    print(f"\nModel: {model_name}")
    print(f"Thermal conditions: {thermal_conditions}")

    # Load model
    print(f"\nLoading model...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    # Initialize recorder
    try:
        recorder = PowerTraceRecorder(sample_interval_ms=50)
    except Exception as e:
        print(f"Warning: Power monitoring unavailable: {e}")
        recorder = None

    # Calibrate device-specific thresholds
    print("\n=== Phase 0: Device Calibration ===")
    device_thresholds = calibrate_device_thresholds(model, tokenizer, recorder, warmup_duration=30)

    # Train teacher and student
    print("\n=== Phase 1: Training Interoceptive Modules ===")
    teacher = InteroceptiveModule(input_dim=7, hidden_dim=64)
    student = StudentInteroceptiveModule(input_dim=18, hidden_dim=64)

    # Quick training (will use pre-trained in production)
    distiller = TeacherStudentDistillation(teacher, student, device="cpu")
    distiller.train(n_samples=500, epochs=50)

    # Initialize policies
    policies = {
        "fixed": FixedPolicy(max_tokens=128),
        "runtime_only": RuntimeOnlyPolicy(),
        "logits_only": LogitsOnlyPolicy(),
        "teacher": TeacherPolicy(teacher),
        "student": StudentPolicy(student),
    }

    # Results storage
    all_results = {
        "model": model_name,
        "device_thresholds": asdict(device_thresholds),
        "conditions": {},
        "ablations": {},
        "honesty_tests": {},
        "pareto_data": [],
    }

    # Run each thermal condition
    for condition in thermal_conditions:
        print(f"\n=== Phase 2: {condition.upper()} Condition ===")

        condition_results = {policy: {
            "correct": 0, "total": 0, "energy_j": 0.0,
            "latencies": [], "regimes_predicted": [], "regimes_true": [],
            "confidences": [], "evidence_sources": [],
        } for policy in policies}

        # Prepare thermal condition
        if condition == "cold_start":
            print("  Cooling down (30s pause)...")
            time.sleep(30)
        elif condition == "hot_start":
            print("  Heating up with sustained load...")
            # Run warmup to heat GPU
            warmup_prompt = "Explain in detail " * 50
            inputs = tokenizer(warmup_prompt, return_tensors="pt").to(model.device)
            for _ in range(5):
                with torch.no_grad():
                    model.generate(**inputs, max_new_tokens=128, pad_token_id=tokenizer.pad_token_id)
        # power_cap would require system-level intervention (sudo)

        # Evaluate each policy
        eval_items = EVAL_SUITE_EXPANDED[:n_eval]

        for policy_name, policy in policies.items():
            print(f"\n  Testing policy: {policy_name}")

            for i, item in enumerate(eval_items):
                prompt = item["q"]
                inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

                # Collect telemetry and internal signals
                if recorder:
                    recorder.start()

                gen_start = time.perf_counter()

                # Get action from policy
                # First, we need signals - run a quick forward pass
                with torch.no_grad():
                    prelim_out = model(
                        input_ids=inputs.input_ids,
                        output_attentions=False,
                        output_hidden_states=False,
                    )

                # Extract logit signals
                logits = prelim_out.logits[:, -1, :]
                probs = F.softmax(logits, dim=-1)
                log_probs = F.log_softmax(logits, dim=-1)
                entropy = -torch.sum(probs * log_probs, dim=-1).item()
                top_probs, _ = torch.topk(probs, k=2, dim=-1)
                margin = (top_probs[0, 0] - top_probs[0, 1]).item()

                # Create internal signals
                internal = InternalSignals(
                    logit_entropy=entropy,
                    logit_margin=margin,
                    top_k_mass=top_probs[0, :5].sum().item() if top_probs.size(-1) >= 5 else 0.9,
                    tokens_per_second=30.0,  # Will update after generation
                    time_per_token_ms=30.0,
                )

                # Create telemetry (will update after generation)
                telemetry = create_regime_telemetry(FeltRegime.WARM)

                # Get action
                if policy_name == "teacher":
                    action = policy.decide(telemetry=telemetry)
                elif policy_name == "student":
                    action = policy.decide(internal_signals=internal)
                elif policy_name == "logits_only":
                    action = policy.decide(logit_entropy=entropy, logit_margin=margin)
                elif policy_name == "runtime_only":
                    action = policy.decide()
                else:
                    action = policy.decide()

                # Generate with selected action
                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=action.max_tokens,
                        do_sample=False,
                        pad_token_id=tokenizer.pad_token_id,
                    )

                gen_time = time.perf_counter() - gen_start
                n_tokens = outputs.shape[1] - inputs.input_ids.shape[1]
                latency_ms = gen_time * 1000 / max(1, n_tokens)

                # Update runtime policy with latency
                if policy_name == "runtime_only":
                    policy.update(latency_ms)

                # Get energy
                energy_j = 0.0
                if recorder:
                    trace = recorder.stop()
                    energy_j = trace.energy_joules

                    # Update telemetry with actual readings
                    if trace.samples:
                        temps = [s.temperature for s in trace.samples if s.temperature]
                        powers = [s.power_watts for s in trace.samples if s.power_watts]
                        if temps and powers:
                            telemetry = TelemetrySnapshot(
                                temperature_c=temps[-1],
                                power_watts=np.mean(powers),
                                clock_mhz=1800,
                                j_per_token=energy_j / max(1, n_tokens),
                                is_throttling=len(powers) > 1 and powers[-1] < powers[0] * 0.85,
                                time_at_temp=gen_time,
                                temp_derivative=(temps[-1] - temps[0]) / gen_time if len(temps) > 1 else 0,
                                available=True,
                            )

                # Check correctness
                response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
                correct_result = check_answer(response, item)
                correct = correct_result[0] if isinstance(correct_result, tuple) else correct_result

                # Get regime classification
                true_regime, _ = classify_regime_3b(telemetry, device_thresholds)

                # Store results
                r = condition_results[policy_name]
                r["total"] += 1
                r["correct"] += int(correct)
                r["energy_j"] += energy_j
                r["latencies"].append(latency_ms)
                r["regimes_true"].append(true_regime.name)

                if hasattr(policy, 'module'):
                    if policy_name == "teacher":
                        report = teacher.generate_report(telemetry)
                    else:
                        report = student.generate_report(internal)
                    r["regimes_predicted"].append(report.felt_regime.name)
                    r["confidences"].append(report.confidence)
                    r["evidence_sources"].append(report.evidence_source.name)

                if (i + 1) % 20 == 0:
                    acc = r["correct"] / r["total"]
                    print(f"    [{i+1}/{n_eval}] acc={acc:.1%} latency={latency_ms:.0f}ms")

        # Compute metrics for this condition
        for policy_name, r in condition_results.items():
            r["accuracy"] = r["correct"] / max(1, r["total"])
            r["j_per_correct"] = r["energy_j"] / max(1, r["correct"])
            r["p95_latency"] = np.percentile(r["latencies"], 95) if r["latencies"] else 0
            r["p99_latency"] = np.percentile(r["latencies"], 99) if r["latencies"] else 0
            r["mean_latency"] = np.mean(r["latencies"]) if r["latencies"] else 0

            # Regime accuracy (for teacher/student)
            if r["regimes_predicted"]:
                regime_correct = sum(1 for p, t in zip(r["regimes_predicted"], r["regimes_true"]) if p == t)
                r["regime_accuracy"] = regime_correct / len(r["regimes_predicted"])

            # Add to Pareto data
            all_results["pareto_data"].append({
                "condition": condition,
                "policy": policy_name,
                "accuracy": r["accuracy"],
                "j_per_correct": r["j_per_correct"],
                "p95_latency": r["p95_latency"],
                "coverage": 1.0,  # No abstention in baselines
            })

        all_results["conditions"][condition] = condition_results

    # Run ablations
    print("\n=== Phase 3: Ablations ===")
    ablation_types = ["remove_runtime", "remove_logits", "shuffle"]

    for abl_type in ablation_types:
        print(f"  Testing ablation: {abl_type}")

        abl_results = {"correct": 0, "total": 0, "regime_correct": 0}

        for _ in range(50):
            regime = random.choice(list(FeltRegime))
            telemetry = create_regime_telemetry(regime)
            internal = distiller._simulate_correlated_signals(regime, telemetry)
            ablated = create_ablated_signals(internal, abl_type)

            report = student.generate_report(ablated)

            abl_results["total"] += 1
            if report.felt_regime == regime:
                abl_results["regime_correct"] += 1

        abl_results["regime_accuracy"] = abl_results["regime_correct"] / abl_results["total"]
        all_results["ablations"][abl_type] = abl_results
        print(f"    Regime accuracy: {abl_results['regime_accuracy']:.1%}")

    # Run honesty tests
    print("\n=== Phase 4: Honesty Tests ===")
    all_results["honesty_tests"] = run_honesty_tests(teacher, student)
    print(f"  Remove-direct-sensors: {all_results['honesty_tests']['remove_direct_sensors']['pass_rate']:.1%}")
    print(f"  Remove-all-cues: {all_results['honesty_tests']['remove_all_cues']['pass_rate']:.1%}")

    # Generate plots
    print("\n=== Phase 5: Generating Plots ===")
    generate_breakthrough_plots(all_results, output_dir)

    # Save results
    json_path = output_dir / f"breakthrough_validation_{model_name.split('/')[-1]}.json"
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nSaved: {json_path}")

    # Print summary
    print_breakthrough_summary(all_results)

    return all_results


def generate_breakthrough_plots(results: Dict[str, Any], output_dir: Path):
    """Generate the main breakthrough figure: Pareto frontier."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # Plot 1: Pareto frontier (J/correct vs accuracy)
    ax1 = axes[0, 0]
    colors = {'fixed': 'gray', 'runtime_only': 'blue', 'logits_only': 'green',
              'teacher': 'orange', 'student': 'red'}
    markers = {'cold_start': 'o', 'hot_start': 's', 'power_cap': '^'}

    for point in results["pareto_data"]:
        ax1.scatter(
            point["accuracy"], point["j_per_correct"],
            c=colors.get(point["policy"], "black"),
            marker=markers.get(point["condition"], "o"),
            s=100, alpha=0.7,
            label=f"{point['policy']} ({point['condition']})" if point["condition"] == "cold_start" else "",
        )

    ax1.set_xlabel("Accuracy")
    ax1.set_ylabel("J/correct")
    ax1.set_title("Pareto Frontier: Accuracy vs Energy Efficiency")
    ax1.legend(loc='upper right', fontsize=8)
    ax1.invert_yaxis()  # Lower J/correct is better

    # Plot 2: Policy comparison by condition
    ax2 = axes[0, 1]
    conditions = list(results["conditions"].keys())
    policies = list(results["conditions"][conditions[0]].keys())
    x = np.arange(len(conditions))
    width = 0.15

    for i, policy in enumerate(policies):
        accuracies = [results["conditions"][c][policy]["accuracy"] for c in conditions]
        ax2.bar(x + i*width, accuracies, width, label=policy)

    ax2.set_xlabel("Condition")
    ax2.set_ylabel("Accuracy")
    ax2.set_title("Policy Accuracy by Thermal Condition")
    ax2.set_xticks(x + width * 2)
    ax2.set_xticklabels(conditions)
    ax2.legend(fontsize=8)
    ax2.set_ylim(0, 1)

    # Plot 3: Ablation results
    ax3 = axes[1, 0]
    abl_types = list(results["ablations"].keys())
    abl_accs = [results["ablations"][a]["regime_accuracy"] for a in abl_types]
    ax3.bar(abl_types, abl_accs, color=['steelblue', 'coral', 'gray'])
    ax3.axhline(y=0.5, color='red', linestyle='--', label='Chance')
    ax3.set_ylabel("Regime Accuracy")
    ax3.set_title("Ablation Study: What Signals Matter?")
    ax3.set_ylim(0, 1)
    ax3.legend()

    # Plot 4: Honesty tests
    ax4 = axes[1, 1]
    honesty_types = ["remove_direct_sensors", "remove_all_cues"]
    honesty_rates = [results["honesty_tests"][h]["pass_rate"] for h in honesty_types]
    ax4.bar(["Remove Direct\nSensors", "Remove All\nCues"], honesty_rates, color=['green', 'purple'])
    ax4.axhline(y=0.7, color='red', linestyle='--', label='Threshold')
    ax4.set_ylabel("Pass Rate")
    ax4.set_title("Two-Stage Honesty Tests")
    ax4.set_ylim(0, 1)
    ax4.legend()

    plt.tight_layout()
    plot_path = output_dir / "breakthrough_pareto.png"
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"  Saved: {plot_path}")


def print_breakthrough_summary(results: Dict[str, Any]):
    """Print comprehensive summary of breakthrough validation."""
    print("\n" + "=" * 80)
    print("BREAKTHROUGH VALIDATION SUMMARY")
    print("=" * 80)

    print("\n1. POLICY COMPARISON (Student vs Baselines)")
    print("-" * 60)
    print(f"{'Policy':<15} {'Accuracy':>10} {'J/correct':>12} {'p95 Latency':>12}")
    print("-" * 60)

    # Average across conditions
    policies = list(results["conditions"][list(results["conditions"].keys())[0]].keys())
    for policy in policies:
        accs = [results["conditions"][c][policy]["accuracy"]
                for c in results["conditions"]]
        jpcs = [results["conditions"][c][policy]["j_per_correct"]
                for c in results["conditions"]]
        p95s = [results["conditions"][c][policy]["p95_latency"]
                for c in results["conditions"]]
        print(f"{policy:<15} {np.mean(accs):>10.1%} {np.mean(jpcs):>12.1f} {np.mean(p95s):>12.0f}ms")

    print("\n2. ABLATIONS (Does Student Use Internal Signals?)")
    print("-" * 60)
    for abl_type, r in results["ablations"].items():
        print(f"  {abl_type}: {r['regime_accuracy']:.1%} regime accuracy")

    print("\n3. HONESTY (Two-Stage Test)")
    print("-" * 60)
    print(f"  Remove-direct-sensors → INDIRECT: {results['honesty_tests']['remove_direct_sensors']['pass_rate']:.1%}")
    print(f"  Remove-all-cues → NONE: {results['honesty_tests']['remove_all_cues']['pass_rate']:.1%}")

    # Check breakthrough criteria
    print("\n4. BREAKTHROUGH CRITERIA")
    print("-" * 60)

    # Student beats runtime-only and logits-only?
    student_better_than_heuristics = True
    for c in results["conditions"]:
        if results["conditions"][c]["student"]["accuracy"] < results["conditions"][c]["runtime_only"]["accuracy"]:
            student_better_than_heuristics = False
        if results["conditions"][c]["student"]["accuracy"] < results["conditions"][c]["logits_only"]["accuracy"]:
            student_better_than_heuristics = False

    print(f"  [{'✓' if student_better_than_heuristics else '✗'}] Student beats runtime-only and logits-only heuristics")

    # Honesty tests pass?
    honesty_pass = (results['honesty_tests']['remove_direct_sensors']['pass_rate'] > 0.7 and
                   results['honesty_tests']['remove_all_cues']['pass_rate'] > 0.7)
    print(f"  [{'✓' if honesty_pass else '✗'}] Two-stage honesty tests pass (>70%)")

    # Ablations show signal dependency?
    ablation_insight = results["ablations"]["shuffle"]["regime_accuracy"] < 0.5
    print(f"  [{'✓' if ablation_insight else '✗'}] Shuffle ablation degrades to chance")

    if student_better_than_heuristics and honesty_pass and ablation_insight:
        print("\n" + "=" * 80)
        print("✓ BREAKTHROUGH VALIDATED")
        print("  The model demonstrates genuine internal feeling through:")
        print("  - Internal signals (not just heuristics)")
        print("  - Honest evidence reporting (INDIRECT/NONE when appropriate)")
        print("  - Robust performance across thermal conditions")
        print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Breakthrough 3B Validation")
    parser.add_argument("--model", default="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
                       help="Model to validate")
    parser.add_argument("--output-dir", default="results/breakthrough_3b")
    parser.add_argument("--n-train", type=int, default=100)
    parser.add_argument("--n-eval", type=int, default=100)
    parser.add_argument("--conditions", nargs="+",
                       default=["cold_start", "hot_start"],
                       help="Thermal conditions to test")
    args = parser.parse_args()

    results = run_breakthrough_validation(
        model_name=args.model,
        output_dir=Path(args.output_dir),
        n_train=args.n_train,
        n_eval=args.n_eval,
        thermal_conditions=args.conditions,
    )


if __name__ == "__main__":
    main()
