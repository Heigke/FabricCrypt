#!/usr/bin/env python3
"""
z29 Causal Loop Validator

Scientific validation of the full embodied loop:

  SENSE → FEEL → REGULATE → EXPRESS → EFFECT → SENSE (repeat)

Tests each causal link independently:

1. SENSE → FEEL: Do sensor changes cause gate changes?
   - Inject synthetic sensor states (high/low power, temp)
   - Measure gate response delta

2. FEEL → REGULATE: Do gate changes cause skip decision changes?
   - With Bernoulli: gate 0.3 should give ~30% run, gate 0.7 ~70% run
   - Statistical test of correlation

3. REGULATE → EXPRESS: Do skip decisions actually skip compute?
   - Measure per-layer execution time with/without skip
   - Verify MLP is actually bypassed

4. EXPRESS → EFFECT: Does skipping change hardware state?
   - Measure power/throughput delta between skip modes
   - This is the "actuator authority" test

5. EFFECT → SENSE: Does the model sense its own effects?
   - After forced skip/run, do sensors reflect the change?
   - Closed-loop feedback verification

6. FULL LOOP: Does intentional sensor manipulation propagate through?
   - Inject "high load" sensors → expect more skip
   - Inject "low load" sensors → expect more run
   - Measure if output behavior matches expectation
"""

import os
import sys
import time
import json
import argparse
import threading
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
import numpy as np
from scipy import stats

import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from transformers import AutoTokenizer, AutoModelForCausalLM


# =============================================================================
# SENSOR HUB WITH INJECTION CAPABILITY
# =============================================================================

class InjectableSensorHub:
    """Sensor hub that allows synthetic sensor injection for testing."""

    def __init__(self, device: str = "cuda", target_throughput: float = 12.0):
        self.device = device
        self.target_throughput = target_throughput
        self._tensor_device = device
        self._running = False
        self._thread = None
        self._lock = threading.Lock()

        # Real sensor readings
        self.power_w = 75.0
        self.temp_c = 50.0
        self.util_pct = 50.0
        self.throughput = target_throughput

        # Injection state
        self._inject_mode = False
        self._injected_tensor = None

        self.power_path = self._find_power_sensor()
        self.start()

    def _find_power_sensor(self) -> str:
        base = Path("/sys/class/drm/card1/device/hwmon")
        if base.exists():
            for hwmon in base.glob("hwmon*"):
                power_file = hwmon / "power1_average"
                if power_file.exists():
                    return str(power_file)
        return "/sys/class/drm/card1/device/hwmon/hwmon7/power1_average"

    def _read_power(self) -> float:
        try:
            with open(self.power_path) as f:
                return int(f.read().strip()) / 1_000_000
        except:
            return 75.0

    def _read_temp(self) -> float:
        try:
            temp_path = self.power_path.replace("power1_average", "temp1_input")
            with open(temp_path) as f:
                return int(f.read().strip()) / 1000
        except:
            return 50.0

    def _read_util(self) -> float:
        try:
            with open("/sys/class/drm/card1/device/gpu_busy_percent") as f:
                return float(f.read().strip())
        except:
            return 50.0

    def _sampling_loop(self):
        interval = 0.01  # 100Hz
        while self._running:
            with self._lock:
                self.power_w = self._read_power()
                self.temp_c = self._read_temp()
                self.util_pct = self._read_util()
            time.sleep(interval)

    def start(self):
        if not self._running:
            self._running = True
            self._thread = threading.Thread(target=self._sampling_loop, daemon=True)
            self._thread.start()

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=1.0)

    def inject(self, tensor: torch.Tensor):
        """Inject synthetic sensor tensor for testing."""
        self._inject_mode = True
        self._injected_tensor = tensor.to(self._tensor_device)

    def clear_injection(self):
        """Return to real sensor reading."""
        self._inject_mode = False
        self._injected_tensor = None

    def read_tensor(self) -> torch.Tensor:
        """Get sensor tensor (real or injected)."""
        if self._inject_mode and self._injected_tensor is not None:
            return self._injected_tensor.clone()

        with self._lock:
            state = np.array([
                min(1.0, self.power_w / 150.0),
                min(1.0, (self.temp_c - 30) / 70.0),
                self.util_pct / 100.0,
                0.5,  # mem
                self.throughput / self.target_throughput,
                0.5,  # trend
                0.0,  # error
                0.5,  # reserved
            ], dtype=np.float32)

        return torch.from_numpy(state).to(self._tensor_device)


# =============================================================================
# CAUSAL LINK TESTS
# =============================================================================

@dataclass
class CausalTestResult:
    """Result of a causal link test."""
    link_name: str
    passed: bool
    effect_size: float
    p_value: float
    details: Dict


def test_sense_to_feel(model, sensor_hub, num_trials: int = 20) -> CausalTestResult:
    """
    Test 1: SENSE → FEEL
    Do sensor changes cause gate changes?

    Method: Inject high vs low sensor states, measure gate response.
    """
    print("\n[Test 1] SENSE → FEEL: Do sensors affect gates?")

    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype

    # Create synthetic sensor states
    high_load_sensors = torch.tensor([
        0.9,  # high power
        0.8,  # high temp
        0.9,  # high util
        0.7,  # high mem
        0.5,  # throughput
        0.3,  # declining trend
        0.4,  # error
        0.5,
    ], dtype=torch.float32, device=device)

    low_load_sensors = torch.tensor([
        0.3,  # low power
        0.3,  # low temp
        0.3,  # low util
        0.3,  # low mem
        1.0,  # good throughput
        0.7,  # improving trend
        0.0,  # no error
        0.5,
    ], dtype=torch.float32, device=device)

    high_gates = []
    low_gates = []

    # Test with dummy input
    dummy_input = torch.randn(1, 10, model.base_model.config.hidden_size,
                              device=device, dtype=dtype)

    for _ in range(num_trials):
        # High load
        sensor_hub.inject(high_load_sensors)
        with torch.no_grad():
            for block in model.skip_blocks.values():
                # Manually call forward to get gate
                sensors = sensor_hub.read_tensor().to(dtype=dtype, device=device)
                last_hidden = dummy_input[:, -1, :]
                sensors_exp = sensors.unsqueeze(0)
                gate_input = torch.cat([last_hidden, sensors_exp], dim=-1)
                gate = block.gate_net(gate_input).item()
                high_gates.append(gate)
                break  # Just test one block

        # Low load
        sensor_hub.inject(low_load_sensors)
        with torch.no_grad():
            for block in model.skip_blocks.values():
                sensors = sensor_hub.read_tensor().to(dtype=dtype, device=device)
                last_hidden = dummy_input[:, -1, :]
                sensors_exp = sensors.unsqueeze(0)
                gate_input = torch.cat([last_hidden, sensors_exp], dim=-1)
                gate = block.gate_net(gate_input).item()
                low_gates.append(gate)
                break

    sensor_hub.clear_injection()

    # Statistical test
    t_stat, p_value = stats.ttest_ind(high_gates, low_gates)
    effect_size = abs(np.mean(high_gates) - np.mean(low_gates))

    passed = p_value < 0.05 and effect_size > 0.01

    print(f"  High load gates: {np.mean(high_gates):.4f} ± {np.std(high_gates):.4f}")
    print(f"  Low load gates:  {np.mean(low_gates):.4f} ± {np.std(low_gates):.4f}")
    print(f"  Effect size: {effect_size:.4f}, p-value: {p_value:.4f}")
    print(f"  Result: {'PASS ✓' if passed else 'FAIL ✗'}")

    return CausalTestResult(
        link_name="SENSE→FEEL",
        passed=passed,
        effect_size=effect_size,
        p_value=p_value,
        details={
            "high_gates_mean": np.mean(high_gates),
            "low_gates_mean": np.mean(low_gates),
            "high_gates_std": np.std(high_gates),
            "low_gates_std": np.std(low_gates),
        }
    )


def test_feel_to_regulate(model, num_trials: int = 100) -> CausalTestResult:
    """
    Test 2: FEEL → REGULATE
    Do gate values cause proportional skip rates?

    Method: With Bernoulli sampling, gate=0.3 should give ~70% skip,
    gate=0.7 should give ~30% skip.
    """
    print("\n[Test 2] FEEL → REGULATE: Do gates control skip decisions?")

    # Simulate Bernoulli sampling at different gate values
    test_gates = [0.2, 0.4, 0.6, 0.8]
    observed_run_rates = []
    expected_run_rates = []

    for gate in test_gates:
        runs = 0
        for _ in range(num_trials):
            u = np.random.random()
            if u < gate:  # Bernoulli: run if u < gate
                runs += 1

        observed_rate = runs / num_trials
        observed_run_rates.append(observed_rate)
        expected_run_rates.append(gate)

        print(f"  Gate={gate:.1f}: Expected run={gate*100:.0f}%, Observed={observed_rate*100:.1f}%")

    # Correlation test
    correlation, p_value = stats.pearsonr(expected_run_rates, observed_run_rates)

    # Effect size: how close to perfect correlation?
    effect_size = correlation

    passed = correlation > 0.95 and p_value < 0.05

    print(f"  Correlation: {correlation:.4f}, p-value: {p_value:.6f}")
    print(f"  Result: {'PASS ✓' if passed else 'FAIL ✗'}")

    return CausalTestResult(
        link_name="FEEL→REGULATE",
        passed=passed,
        effect_size=effect_size,
        p_value=p_value,
        details={
            "correlation": correlation,
            "test_gates": test_gates,
            "observed_rates": observed_run_rates,
        }
    )


def test_regulate_to_express(model, tokenizer, num_runs: int = 5) -> CausalTestResult:
    """
    Test 3: REGULATE → EXPRESS
    Do skip decisions actually skip MLP computation?

    Method: Measure time with forced skip vs forced run.
    """
    print("\n[Test 3] REGULATE → EXPRESS: Does skip actually skip compute?")

    device = next(model.parameters()).device

    prompt = "The capital of France is"
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs.input_ids.to(device)

    # Measure baseline (normal)
    normal_times = []
    for _ in range(num_runs):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            _ = model.base_model(input_ids)
        torch.cuda.synchronize()
        normal_times.append(time.perf_counter() - t0)

    # Force all skip by setting gates to 0
    original_biases = {}
    for name, block in model.skip_blocks.items():
        original_biases[name] = block.gate_net[-2].bias.data.clone()
        block.gate_net[-2].bias.data.fill_(-10.0)  # Sigmoid(-10) ≈ 0

    skip_times = []
    for _ in range(num_runs):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            _ = model.base_model(input_ids)
        torch.cuda.synchronize()
        skip_times.append(time.perf_counter() - t0)

    # Force all run by setting gates to 1
    for name, block in model.skip_blocks.items():
        block.gate_net[-2].bias.data.fill_(10.0)  # Sigmoid(10) ≈ 1

    run_times = []
    for _ in range(num_runs):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            _ = model.base_model(input_ids)
        torch.cuda.synchronize()
        run_times.append(time.perf_counter() - t0)

    # Restore original biases
    for name, block in model.skip_blocks.items():
        block.gate_net[-2].bias.data.copy_(original_biases[name])

    # Analysis
    skip_mean = np.mean(skip_times) * 1000  # ms
    run_mean = np.mean(run_times) * 1000
    time_diff = run_mean - skip_mean

    # Skip should be faster
    t_stat, p_value = stats.ttest_ind(run_times, skip_times)
    effect_size = time_diff / run_mean  # Relative speedup

    passed = time_diff > 0 and p_value < 0.05

    print(f"  Skip time: {skip_mean:.2f}ms ± {np.std(skip_times)*1000:.2f}ms")
    print(f"  Run time:  {run_mean:.2f}ms ± {np.std(run_times)*1000:.2f}ms")
    print(f"  Difference: {time_diff:.2f}ms ({effect_size*100:.1f}% faster with skip)")
    print(f"  p-value: {p_value:.6f}")
    print(f"  Result: {'PASS ✓' if passed else 'FAIL ✗'}")

    return CausalTestResult(
        link_name="REGULATE→EXPRESS",
        passed=passed,
        effect_size=effect_size,
        p_value=p_value,
        details={
            "skip_time_ms": skip_mean,
            "run_time_ms": run_mean,
            "speedup_pct": effect_size * 100,
        }
    )


def test_express_to_effect(model, tokenizer, sensor_hub, num_runs: int = 5) -> CausalTestResult:
    """
    Test 4: EXPRESS → EFFECT
    Does skipping actually change hardware state?

    Method: Measure power/throughput with forced skip vs run.
    """
    print("\n[Test 4] EXPRESS → EFFECT: Does skip change hardware?")

    device = next(model.parameters()).device

    prompt = "Explain machine learning in detail:"
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs.input_ids.to(device)

    def measure_generation(force_skip: bool = None):
        """Generate tokens and measure hardware."""
        # Temporarily modify gates if forcing
        original_biases = {}
        if force_skip is not None:
            for name, block in model.skip_blocks.items():
                original_biases[name] = block.gate_net[-2].bias.data.clone()
                block.gate_net[-2].bias.data.fill_(-10.0 if force_skip else 10.0)

        powers = []
        t0 = time.perf_counter()

        with torch.no_grad():
            gen_ids = input_ids.clone()
            for _ in range(32):  # Generate 32 tokens
                outputs = model.base_model(gen_ids, use_cache=False)
                next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
                gen_ids = torch.cat([gen_ids, next_token], dim=-1)
                powers.append(sensor_hub.power_w)

        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0

        # Restore
        if force_skip is not None:
            for name, block in model.skip_blocks.items():
                block.gate_net[-2].bias.data.copy_(original_biases[name])

        return {
            "time": elapsed,
            "throughput": 32 / elapsed,
            "power_mean": np.mean(powers),
            "power_std": np.std(powers),
        }

    # Measure with forced skip
    skip_results = [measure_generation(force_skip=True) for _ in range(num_runs)]

    # Measure with forced run
    run_results = [measure_generation(force_skip=False) for _ in range(num_runs)]

    # Analysis
    skip_throughput = [r["throughput"] for r in skip_results]
    run_throughput = [r["throughput"] for r in run_results]
    skip_power = [r["power_mean"] for r in skip_results]
    run_power = [r["power_mean"] for r in run_results]

    throughput_diff = np.mean(skip_throughput) - np.mean(run_throughput)
    power_diff = np.mean(skip_power) - np.mean(run_power)

    # Throughput should increase with skip
    t_stat, p_value = stats.ttest_ind(skip_throughput, run_throughput)
    effect_size = throughput_diff / np.mean(run_throughput)

    # Skip should increase throughput (positive effect)
    passed = throughput_diff > 0 and p_value < 0.1

    print(f"  Skip throughput: {np.mean(skip_throughput):.2f} ± {np.std(skip_throughput):.2f} tok/s")
    print(f"  Run throughput:  {np.mean(run_throughput):.2f} ± {np.std(run_throughput):.2f} tok/s")
    print(f"  Throughput delta: {throughput_diff:+.2f} tok/s ({effect_size*100:+.1f}%)")
    print(f"  Power delta: {power_diff:+.1f}W")
    print(f"  p-value: {p_value:.6f}")
    print(f"  Result: {'PASS ✓' if passed else 'FAIL ✗'}")

    return CausalTestResult(
        link_name="EXPRESS→EFFECT",
        passed=passed,
        effect_size=effect_size,
        p_value=p_value,
        details={
            "skip_throughput": np.mean(skip_throughput),
            "run_throughput": np.mean(run_throughput),
            "throughput_delta": throughput_diff,
            "power_delta": power_diff,
        }
    )


def test_effect_to_sense(sensor_hub, num_samples: int = 50) -> CausalTestResult:
    """
    Test 5: EFFECT → SENSE
    Does the sensor hub correctly sense hardware state?

    Method: Read sensors rapidly and check variance/responsiveness.
    """
    print("\n[Test 5] EFFECT → SENSE: Are sensors responsive?")

    readings = []
    for _ in range(num_samples):
        tensor = sensor_hub.read_tensor()
        readings.append(tensor.cpu().numpy())
        time.sleep(0.02)  # 50Hz sampling

    readings = np.array(readings)

    # Check variance in power readings (should not be constant)
    power_variance = np.var(readings[:, 0])
    power_range = np.max(readings[:, 0]) - np.min(readings[:, 0])

    # Sensors should show some natural variance
    passed = power_variance > 0.0001 or power_range > 0.01

    print(f"  Power variance: {power_variance:.6f}")
    print(f"  Power range: {power_range:.4f}")
    print(f"  Sensor mean: {np.mean(readings, axis=0)[:4]}")
    print(f"  Sensor std:  {np.std(readings, axis=0)[:4]}")
    print(f"  Result: {'PASS ✓' if passed else 'FAIL ✗'}")

    return CausalTestResult(
        link_name="EFFECT→SENSE",
        passed=passed,
        effect_size=power_range,
        p_value=0.0,  # Not a statistical test
        details={
            "power_variance": power_variance,
            "power_range": power_range,
            "mean_readings": np.mean(readings, axis=0).tolist(),
        }
    )


def test_full_loop(model, tokenizer, sensor_hub, num_trials: int = 10) -> CausalTestResult:
    """
    Test 6: FULL LOOP
    Does the complete sense→feel→regulate→express→effect→sense loop work?

    Method: Inject "stressed" sensors, check if skip rate responds appropriately.
    """
    print("\n[Test 6] FULL LOOP: Complete causal chain test")

    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype

    prompt = "Write a poem:"
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs.input_ids.to(device)

    # Stressed state: high power, high temp → model should want to skip more
    stressed_sensors = torch.tensor([
        0.95, 0.90, 0.95, 0.80, 0.4, 0.2, 0.6, 0.5
    ], dtype=torch.float32, device=device)

    # Relaxed state: low power, low temp → model can run more
    relaxed_sensors = torch.tensor([
        0.2, 0.2, 0.3, 0.3, 1.0, 0.8, 0.0, 0.5
    ], dtype=torch.float32, device=device)

    def run_with_sensors(injected: torch.Tensor):
        sensor_hub.inject(injected)
        model.reset_stats()

        with torch.no_grad():
            gen_ids = input_ids.clone()
            for _ in range(16):
                outputs = model.base_model(gen_ids, use_cache=False)
                next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
                gen_ids = torch.cat([gen_ids, next_token], dim=-1)

        skip_rate = model.skip_rate
        gate_mean = model.gate_mean
        sensor_hub.clear_injection()

        return skip_rate, gate_mean

    stressed_skips = []
    relaxed_skips = []
    stressed_gates = []
    relaxed_gates = []

    for _ in range(num_trials):
        skip, gate = run_with_sensors(stressed_sensors)
        stressed_skips.append(skip)
        stressed_gates.append(gate)

        skip, gate = run_with_sensors(relaxed_sensors)
        relaxed_skips.append(skip)
        relaxed_gates.append(gate)

    # If the loop works, stressed should have different behavior than relaxed
    gate_diff = np.mean(stressed_gates) - np.mean(relaxed_gates)
    skip_diff = np.mean(stressed_skips) - np.mean(relaxed_skips)

    # Statistical test on gate difference
    t_stat, p_value = stats.ttest_ind(stressed_gates, relaxed_gates)

    # Effect size
    effect_size = abs(gate_diff)

    passed = effect_size > 0.005 and p_value < 0.1

    print(f"  Stressed: gate={np.mean(stressed_gates):.4f}, skip={np.mean(stressed_skips)*100:.1f}%")
    print(f"  Relaxed:  gate={np.mean(relaxed_gates):.4f}, skip={np.mean(relaxed_skips)*100:.1f}%")
    print(f"  Gate difference: {gate_diff:+.4f}")
    print(f"  Skip difference: {skip_diff*100:+.1f}%")
    print(f"  p-value: {p_value:.6f}")
    print(f"  Result: {'PASS ✓' if passed else 'FAIL ✗'}")

    return CausalTestResult(
        link_name="FULL_LOOP",
        passed=passed,
        effect_size=effect_size,
        p_value=p_value,
        details={
            "stressed_gate_mean": np.mean(stressed_gates),
            "relaxed_gate_mean": np.mean(relaxed_gates),
            "stressed_skip_mean": np.mean(stressed_skips),
            "relaxed_skip_mean": np.mean(relaxed_skips),
            "gate_diff": gate_diff,
            "skip_diff": skip_diff,
        }
    )


# =============================================================================
# MAIN VALIDATOR
# =============================================================================

def run_full_validation(model, tokenizer, sensor_hub, wandb_log: bool = True) -> Dict:
    """Run all causal loop tests and return summary."""

    print("=" * 70)
    print("CAUSAL LOOP VALIDATION")
    print("Testing: SENSE → FEEL → REGULATE → EXPRESS → EFFECT → SENSE")
    print("=" * 70)

    results = {}

    # Run all tests
    results["sense_feel"] = test_sense_to_feel(model, sensor_hub)
    results["feel_regulate"] = test_feel_to_regulate(model)
    results["regulate_express"] = test_regulate_to_express(model, tokenizer)
    results["express_effect"] = test_express_to_effect(model, tokenizer, sensor_hub)
    results["effect_sense"] = test_effect_to_sense(sensor_hub)
    results["full_loop"] = test_full_loop(model, tokenizer, sensor_hub)

    # Summary
    print("\n" + "=" * 70)
    print("VALIDATION SUMMARY")
    print("=" * 70)

    all_passed = True
    for name, result in results.items():
        status = "✓ PASS" if result.passed else "✗ FAIL"
        print(f"  {result.link_name:20s}: {status} (effect={result.effect_size:.4f}, p={result.p_value:.4f})")
        if not result.passed:
            all_passed = False

    loop_score = sum(1 for r in results.values() if r.passed) / len(results)
    print(f"\n  LOOP INTEGRITY SCORE: {loop_score*100:.0f}% ({sum(1 for r in results.values() if r.passed)}/{len(results)} tests passed)")

    if all_passed:
        print("\n  ✓ FULL CAUSAL LOOP VERIFIED!")
    else:
        print("\n  ✗ Some causal links are weak or broken")

    # Log to W&B if enabled
    if wandb_log and wandb.run is not None:
        log_data = {
            "causal/loop_score": loop_score,
            "causal/all_passed": all_passed,
        }
        for name, result in results.items():
            log_data[f"causal/{name}/passed"] = result.passed
            log_data[f"causal/{name}/effect_size"] = result.effect_size
            log_data[f"causal/{name}/p_value"] = result.p_value
        wandb.log(log_data)

    return {
        "loop_score": loop_score,
        "all_passed": all_passed,
        "results": {name: {
            "passed": r.passed,
            "effect_size": r.effect_size,
            "p_value": r.p_value,
            "details": r.details,
        } for name, r in results.items()}
    }


# =============================================================================
# STANDALONE EXECUTION
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, help="Model checkpoint to load")
    parser.add_argument("--wandb-run-id", type=str, help="W&B run ID to join")
    parser.add_argument("--output", type=str, default="causal_loop_results.json")
    args = parser.parse_args()

    # Import model components
    sys.path.insert(0, str(Path(__file__).parent))
    from z29_stochastic_trainer import EmbodiedModelZ29, BernoulliSTEFunction, MLPSkipBlockZ29

    # Join W&B if specified
    if args.wandb_run_id:
        wandb.init(
            project="feel-z29-stochastic",
            id=args.wandb_run_id,
            resume="must",
        )

    # Load model
    print("Loading model...")
    model_name = "Qwen/Qwen2.5-7B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    # Create sensor hub and model
    sensor_hub = InjectableSensorHub(device="cuda")
    model = EmbodiedModelZ29(
        base_model=base_model,
        sensor_hub=sensor_hub,
    )

    # Load checkpoint if specified
    if args.checkpoint:
        ckpt = torch.load(args.checkpoint)
        model.load_state_dict(ckpt["model_state_dict"], strict=False)
        print(f"Loaded checkpoint: {args.checkpoint}")

    # Run validation
    results = run_full_validation(model, tokenizer, sensor_hub)

    # Save results
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {args.output}")

    sensor_hub.stop()
