#!/usr/bin/env python3
"""
z38 REAL SENSOR Validation Suite
=================================

FIXES FROM z37:
1. Uses REAL sensor readings instead of inject_stress()
2. Implements ACTUAL GPU stress on same thermal domain
3. Logs sensor deltas to confirm disturbances are visible
4. Adds control objective metrics (time-in-band, p95 TPOT, energy/token)

Key insight: z37's disturbance test fed CONSTANT synthetic sensors,
so it couldn't possibly measure real adaptation. This version uses
read_tensor() to get actual hardware telemetry during stress.
"""

import os
import sys
import json
import time
import random
import subprocess
import threading
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
import torch.nn.functional as F
from scipy import stats
from transformers import AutoTokenizer, AutoModelForCausalLM

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.sensors.canonical_features import CanonicalSensorHub, SENSOR_DIM

# ============================================================================
# ROBUST STATISTICS (unchanged from z37)
# ============================================================================

def bootstrap_ci(data: np.ndarray, n_bootstrap: int = 1000, ci: float = 0.95) -> Tuple[float, float]:
    """Compute bootstrap confidence interval."""
    if len(data) < 2:
        return (float(data[0]), float(data[0])) if len(data) == 1 else (0.0, 0.0)
    boot_means = [np.mean(np.random.choice(data, size=len(data), replace=True)) for _ in range(n_bootstrap)]
    return (float(np.percentile(boot_means, (1 - ci) / 2 * 100)),
            float(np.percentile(boot_means, (1 + ci) / 2 * 100)))

def cliffs_delta(group1: np.ndarray, group2: np.ndarray) -> float:
    """Cliff's Delta - nonparametric effect size."""
    n1, n2 = len(group1), len(group2)
    if n1 == 0 or n2 == 0:
        return 0.0
    more = sum(1 for x in group1 for y in group2 if x > y)
    less = sum(1 for x in group1 for y in group2 if x < y)
    return (more - less) / (n1 * n2)

def interpret_cliff(d: float) -> str:
    abs_d = abs(d)
    if abs_d < 0.147: return "negligible"
    elif abs_d < 0.33: return "small"
    elif abs_d < 0.474: return "medium"
    else: return "large"

def robust_stats(group1: np.ndarray, group2: np.ndarray, label: str = "") -> Dict:
    """Compute comprehensive robust statistics."""
    n1, n2 = len(group1), len(group2)

    # Cohen's d
    if n1 >= 2 and n2 >= 2:
        pooled_std = np.sqrt(((n1-1)*np.var(group1, ddof=1) + (n2-1)*np.var(group2, ddof=1)) / max(1, n1+n2-2))
        d = (np.mean(group1) - np.mean(group2)) / pooled_std if pooled_std > 1e-10 else 10.0
        d_interp = "large" if abs(d) >= 0.8 else "medium" if abs(d) >= 0.5 else "small" if abs(d) >= 0.2 else "negligible"
    else:
        d, d_interp = 0.0, "insufficient_data"

    cliff = cliffs_delta(group1, group2)
    t_stat, p_val = stats.ttest_ind(group1, group2, equal_var=False) if n1 > 1 and n2 > 1 else (0.0, 1.0)

    return {
        "label": label,
        "group1_mean": float(np.mean(group1)), "group1_std": float(np.std(group1, ddof=1)) if n1 > 1 else 0.0,
        "group1_ci95": bootstrap_ci(group1),
        "group2_mean": float(np.mean(group2)), "group2_std": float(np.std(group2, ddof=1)) if n2 > 1 else 0.0,
        "group2_ci95": bootstrap_ci(group2),
        "cohens_d": float(d), "cohens_d_interpretation": d_interp,
        "cliffs_delta": cliff, "cliffs_delta_interpretation": interpret_cliff(cliff),
        "p_value": float(p_val), "t_statistic": float(t_stat), "n1": n1, "n2": n2,
    }

# ============================================================================
# POWER SAMPLER (unchanged)
# ============================================================================

class PowerSampler:
    """Sample power during inference for energy calculation."""

    def __init__(self, device_path: str = "/sys/class/drm/card1/device"):
        self.device_path = Path(device_path)
        self.power_path = self._find_power_sensor()
        self.temp_path = self._find_temp_sensor()
        self.samples = []
        self.sampling = False
        self._thread = None

    def _find_power_sensor(self) -> Optional[Path]:
        hwmon_base = self.device_path / "hwmon"
        if hwmon_base.exists():
            for hwmon in sorted(hwmon_base.iterdir()):
                p = hwmon / "power1_average"
                if p.exists(): return p
        return None

    def _find_temp_sensor(self) -> Optional[Path]:
        hwmon_base = self.device_path / "hwmon"
        if hwmon_base.exists():
            for hwmon in sorted(hwmon_base.iterdir()):
                for name in ["temp1_input", "temp2_input"]:
                    t = hwmon / name
                    if t.exists(): return t
        return None

    def _read_power(self) -> float:
        try: return float(self.power_path.read_text().strip()) / 1e6 if self.power_path else 0.0
        except: return 0.0

    def _read_temp(self) -> float:
        try: return float(self.temp_path.read_text().strip()) / 1000 if self.temp_path else 50.0
        except: return 50.0

    def _sample_loop(self):
        while self.sampling:
            self.samples.append((time.perf_counter(), self._read_power(), self._read_temp()))
            time.sleep(0.01)

    def start(self):
        self.samples = []
        self.sampling = True
        self._thread = threading.Thread(target=self._sample_loop, daemon=True)
        self._thread.start()

    def stop(self) -> Dict:
        self.sampling = False
        if self._thread: self._thread.join(timeout=0.1)

        if len(self.samples) < 2:
            return {"avg_power_w": 0.0, "energy_j": 0.0, "duration_s": 0.0, "peak_temp_c": 50.0}

        t0, _, _ = self.samples[0]
        t1, _, _ = self.samples[-1]
        duration = t1 - t0
        powers = [p for _, p, _ in self.samples]
        temps = [t for _, _, t in self.samples]

        return {
            "avg_power_w": float(np.mean(powers)),
            "energy_j": float(np.mean(powers) * duration),
            "duration_s": duration,
            "peak_temp_c": float(max(temps)),
            "power_std": float(np.std(powers)),
        }

# ============================================================================
# REAL GPU STRESS - Actually stress the same GPU
# ============================================================================

class RealGPUStress:
    """
    Apply REAL GPU stress on the SAME device as the model.

    Key insight: The stress must be on the same thermal/power domain
    to affect the sensors the model reads.
    """

    def __init__(self, device: str = "cuda"):
        self.device = device
        self._stop_event = threading.Event()
        self._thread = None
        self._stress_tensor = None

    def start(self, intensity: float = 0.5):
        """
        Start GPU stress.

        Args:
            intensity: 0.0 (light) to 1.0 (heavy)
                - 0.3: Light stress (small matrices, frequent sync)
                - 0.6: Medium stress
                - 1.0: Heavy stress (large matrices, back-to-back)
        """
        self._stop_event.clear()

        # Matrix size based on intensity
        size = int(512 + intensity * 1536)  # 512 to 2048

        def stress_loop():
            # Allocate stress tensor on same GPU
            x = torch.randn(size, size, device=self.device, dtype=torch.float16)

            while not self._stop_event.is_set():
                # Matrix multiply generates GPU load
                _ = torch.mm(x, x)

                # Sync frequency based on intensity
                # Lower intensity = more frequent syncs = more yielding
                if intensity < 0.5:
                    torch.cuda.synchronize()
                    time.sleep(0.005)  # 5ms yield
                elif intensity < 0.8:
                    torch.cuda.synchronize()
                    time.sleep(0.002)  # 2ms yield
                else:
                    # Heavy: minimal yielding
                    if random.random() < 0.1:
                        torch.cuda.synchronize()

        self._thread = threading.Thread(target=stress_loop, daemon=True)
        self._thread.start()

        # Wait for stress to stabilize
        time.sleep(0.5)

    def stop(self):
        """Stop GPU stress."""
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=1.0)
        # Clear CUDA cache
        torch.cuda.empty_cache()
        time.sleep(0.2)

class RealCPUStress:
    """Apply real CPU stress."""

    def __init__(self):
        self._stop_event = threading.Event()
        self._threads = []

    def start(self, cores: int = 2):
        self._stop_event.clear()

        def cpu_work():
            while not self._stop_event.is_set():
                # Compute-intensive work
                _ = sum(i*i for i in range(5000))
                # Small yield to allow stopping
                time.sleep(0.001)

        for _ in range(cores):
            t = threading.Thread(target=cpu_work, daemon=True)
            t.start()
            self._threads.append(t)

    def stop(self):
        self._stop_event.set()
        for t in self._threads:
            t.join(timeout=0.5)
        self._threads = []

# ============================================================================
# SKIP BLOCK WRAPPER
# ============================================================================

class SkipBlockWrapper(torch.nn.Module):
    """Wrapper that adds skip/FiLM functionality to a transformer layer."""

    def __init__(self, original_layer, layer_idx: int, hidden_size: int, device, dtype):
        super().__init__()
        self._original_layer = original_layer
        self.layer_idx = layer_idx

        self.skip_gate = torch.nn.Linear(hidden_size, 1).to(device=device, dtype=dtype)
        self.film_scale = torch.nn.Linear(SENSOR_DIM, hidden_size).to(device=device, dtype=dtype)
        self.film_shift = torch.nn.Linear(SENSOR_DIM, hidden_size).to(device=device, dtype=dtype)

        torch.nn.init.zeros_(self.skip_gate.weight)
        torch.nn.init.constant_(self.skip_gate.bias, -2.0)
        torch.nn.init.normal_(self.film_scale.weight, mean=0.0, std=0.02)
        torch.nn.init.zeros_(self.film_scale.bias)
        torch.nn.init.normal_(self.film_shift.weight, mean=0.0, std=0.02)
        torch.nn.init.zeros_(self.film_shift.bias)

        self.sensors: Optional[torch.Tensor] = None
        self.last_gate_value: Optional[float] = None
        self.last_skipped: bool = False
        self.force_gate: Optional[float] = None

    def __getattr__(self, name: str):
        """Forward attribute access to original layer for compatibility."""
        if name.startswith('_') or name in ['skip_gate', 'film_scale', 'film_shift', 'sensors',
                                             'last_gate_value', 'last_skipped', 'force_gate', 'layer_idx']:
            return super().__getattr__(name)
        return getattr(self._original_layer, name)

    def forward(self, hidden_states, *args, **kwargs):
        batch_size = hidden_states.shape[0]

        # Gate computation from hidden states
        pooled = hidden_states.mean(dim=1)
        gate_logit = self.skip_gate(pooled)
        gate_value = torch.sigmoid(gate_logit).mean().item()

        if self.force_gate is not None:
            gate_value = self.force_gate

        self.last_gate_value = gate_value

        # Skip decision
        if gate_value > 0.5:
            self.last_skipped = True
            return (hidden_states,) + (None,) * (len(args) + len(kwargs))

        self.last_skipped = False

        # FiLM modulation if sensors available
        if self.sensors is not None:
            sensors = self.sensors.to(hidden_states.device, hidden_states.dtype)
            if sensors.dim() == 1:
                sensors = sensors.unsqueeze(0).expand(batch_size, -1)

            scale = 1.0 + self.film_scale(sensors).unsqueeze(1) * 0.1
            shift = self.film_shift(sensors).unsqueeze(1) * 0.1
            hidden_states = hidden_states * scale + shift

        return self._original_layer(hidden_states, *args, **kwargs)

# ============================================================================
# REAL SENSOR VALIDATOR
# ============================================================================

class RealSensorValidator:
    """
    Validation suite that uses REAL sensor readings.

    Key difference from z37: We call sensor_hub.read_tensor() to get
    actual hardware telemetry instead of inject_stress() which gives
    constant synthetic values.
    """

    BONFERRONI_ALPHA = 0.05 / 4  # 4 main tests

    def __init__(self, model, tokenizer, sensor_hub, skip_blocks: Dict, device: str):
        self.model = model
        self.tokenizer = tokenizer
        self.sensor_hub = sensor_hub
        self.skip_blocks = skip_blocks
        self.device = device
        self.power_sampler = PowerSampler()
        self.gpu_stress = RealGPUStress(device)
        self.cpu_stress = RealCPUStress()

        self.prompts = [
            "The future of artificial intelligence will",
            "In a world where technology advances rapidly",
            "Scientists have discovered that the key to",
            "The most important thing about learning is",
            "When we consider the implications of climate",
        ]

    def _reset_blocks(self):
        for b in self.skip_blocks.values():
            b.sensors = None
            b.force_gate = None
            b.last_gate_value = None
            b.last_skipped = False

    def _inject_real_sensors(self, actual_throughput: Optional[float] = None):
        """
        Inject REAL sensor readings into skip blocks.

        This is the KEY FIX: we read actual hardware telemetry
        instead of using synthetic values.
        """
        # Read real sensors
        sensors = self.sensor_hub.read_tensor(actual_throughput=actual_throughput)

        # Inject into all skip blocks
        for b in self.skip_blocks.values():
            b.sensors = sensors

        return sensors

    def _get_sensor_snapshot(self) -> Dict:
        """Get current sensor values for logging."""
        self.sensor_hub.update()
        reading = self.sensor_hub.last_reading
        if reading:
            return {
                "power_w": reading.power_mw,
                "temp_c": reading.temp_c,
                "clock_mhz": reading.clock_mhz,
                "throttle": reading.throttle_active,
            }
        return {"power_w": 0, "temp_c": 50, "clock_mhz": 1000, "throttle": False}

    # ========================================================================
    # TEST 1: Real Sensor Response (H1 with actual hardware stress)
    # ========================================================================

    def test_real_sensor_response(self, trials: int = 20) -> Dict:
        """
        Test that gates respond to REAL sensor changes.

        Unlike z37 which used inject_stress(), we actually stress the GPU
        and read real sensors.
        """
        print("\n  [TEST 1] Real Sensor Response...")

        gate_baseline = []
        gate_stressed = []
        sensor_deltas = []

        for trial in range(trials):
            prompt = random.choice(self.prompts)
            enc = self.tokenizer(prompt, return_tensors="pt")
            input_ids = enc.input_ids.to(self.device)
            attention_mask = torch.ones_like(input_ids)

            # --- BASELINE (no stress) ---
            sensor_before = self._get_sensor_snapshot()
            self._inject_real_sensors()

            with torch.no_grad():
                _ = self.model(input_ids=input_ids, attention_mask=attention_mask)

            gates_baseline = [b.last_gate_value for b in self.skip_blocks.values() if b.last_gate_value]
            gate_baseline.append(np.mean(gates_baseline) if gates_baseline else 0.5)

            # --- STRESSED (real GPU stress) ---
            self.gpu_stress.start(intensity=0.7)
            time.sleep(0.5)  # Let stress affect thermals

            sensor_after = self._get_sensor_snapshot()
            self._inject_real_sensors()

            with torch.no_grad():
                _ = self.model(input_ids=input_ids, attention_mask=attention_mask)

            gates_stressed = [b.last_gate_value for b in self.skip_blocks.values() if b.last_gate_value]
            gate_stressed.append(np.mean(gates_stressed) if gates_stressed else 0.5)

            self.gpu_stress.stop()
            time.sleep(0.3)

            # Log sensor delta
            sensor_deltas.append({
                "power_delta": sensor_after["power_w"] - sensor_before["power_w"],
                "temp_delta": sensor_after["temp_c"] - sensor_before["temp_c"],
            })

            if trial % 5 == 0:
                print(f"      Trial {trial+1}: baseline={gate_baseline[-1]:.3f}, stressed={gate_stressed[-1]:.3f}, "
                      f"ΔP={sensor_deltas[-1]['power_delta']:.1f}W, ΔT={sensor_deltas[-1]['temp_delta']:.1f}°C")

        self._reset_blocks()

        stats = robust_stats(np.array(gate_stressed), np.array(gate_baseline), "Real Sensor Response")
        stats["passed"] = stats["p_value"] < self.BONFERRONI_ALPHA
        stats["avg_power_delta"] = float(np.mean([d["power_delta"] for d in sensor_deltas]))
        stats["avg_temp_delta"] = float(np.mean([d["temp_delta"] for d in sensor_deltas]))
        stats["sensor_visible"] = stats["avg_power_delta"] > 5 or stats["avg_temp_delta"] > 2

        return stats

    # ========================================================================
    # TEST 2: Disturbance Adaptation (FIXED)
    # ========================================================================

    def test_disturbance_adaptation(self, trials: int = 15) -> Dict:
        """
        Test adaptation under REAL disturbance.

        FIX: Uses read_tensor() instead of inject_stress().
        FIX: Actually applies GPU stress on same device.
        """
        print("\n  [TEST 2] Disturbance Adaptation (FIXED)...")

        conditions = {
            "baseline": {"stress_type": None, "intensity": 0},
            "gpu_light": {"stress_type": "gpu", "intensity": 0.3},
            "gpu_heavy": {"stress_type": "gpu", "intensity": 0.8},
            "cpu_stress": {"stress_type": "cpu", "cores": 2},
        }

        results = {}

        for condition_name, config in conditions.items():
            print(f"    Testing {condition_name}...")

            # Start disturbance
            if config.get("stress_type") == "gpu":
                self.gpu_stress.start(intensity=config["intensity"])
            elif config.get("stress_type") == "cpu":
                self.cpu_stress.start(cores=config.get("cores", 2))

            time.sleep(1.0)  # Let stress stabilize and affect sensors

            # Capture sensor state BEFORE trials
            sensor_start = self._get_sensor_snapshot()

            energies = []
            latencies = []
            skip_rates = []
            gate_values = []
            sensor_readings = []

            for _ in range(trials):
                prompt = random.choice(self.prompts)
                enc = self.tokenizer(prompt, return_tensors="pt")
                input_ids = enc.input_ids.to(self.device)
                attention_mask = torch.ones_like(input_ids)

                # KEY FIX: Read REAL sensors
                sensors = self._inject_real_sensors()
                sensor_readings.append(sensors.cpu().numpy())

                self.power_sampler.start()
                t0 = time.perf_counter()

                with torch.no_grad():
                    outputs = self.model.generate(
                        input_ids,
                        attention_mask=attention_mask,
                        max_new_tokens=16,
                        do_sample=False,
                        pad_token_id=self.tokenizer.pad_token_id,
                    )

                torch.cuda.synchronize()
                latency = (time.perf_counter() - t0) * 1000
                power_stats = self.power_sampler.stop()

                energies.append(power_stats["energy_j"])
                latencies.append(latency)

                gates = [b.last_gate_value for b in self.skip_blocks.values() if b.last_gate_value]
                gate_values.append(np.mean(gates) if gates else 0.5)

                skips = [1.0 if b.last_skipped else 0.0 for b in self.skip_blocks.values()]
                skip_rates.append(np.mean(skips))

            # Stop disturbance
            if config.get("stress_type") == "gpu":
                self.gpu_stress.stop()
            elif config.get("stress_type") == "cpu":
                self.cpu_stress.stop()
            time.sleep(0.5)

            # Capture sensor state AFTER
            sensor_end = self._get_sensor_snapshot()

            # Compute sensor feature variance (did sensors actually change?)
            sensor_array = np.array(sensor_readings)
            sensor_variance = float(np.mean(np.var(sensor_array, axis=0)))

            results[condition_name] = {
                "energy_mean": float(np.mean(energies)),
                "energy_std": float(np.std(energies)),
                "latency_mean": float(np.mean(latencies)),
                "latency_p95": float(np.percentile(latencies, 95)),
                "skip_rate_mean": float(np.mean(skip_rates)),
                "skip_rate_std": float(np.std(skip_rates)),
                "gate_mean": float(np.mean(gate_values)),
                "gate_std": float(np.std(gate_values)),
                "sensor_variance": sensor_variance,
                "power_delta": sensor_end["power_w"] - sensor_start["power_w"],
                "temp_delta": sensor_end["temp_c"] - sensor_start["temp_c"],
            }

            print(f"      Skip rate: {results[condition_name]['skip_rate_mean']:.3f}, "
                  f"Gate: {results[condition_name]['gate_mean']:.3f}, "
                  f"ΔP: {results[condition_name]['power_delta']:.1f}W")

        self._reset_blocks()

        # Check for adaptation
        baseline_skip = results["baseline"]["skip_rate_mean"]
        gpu_heavy_skip = results["gpu_heavy"]["skip_rate_mean"]

        # Adaptation = skip rate INCREASES under stress
        adaptation_detected = gpu_heavy_skip > baseline_skip + 0.05
        skip_change = gpu_heavy_skip - baseline_skip

        # Sensor visibility check
        sensors_changed = (
            results["gpu_heavy"]["power_delta"] > 5 or
            results["gpu_heavy"]["temp_delta"] > 2 or
            results["gpu_heavy"]["sensor_variance"] > 0.01
        )

        return {
            "condition_results": results,
            "adaptation_detected": adaptation_detected,
            "skip_rate_change": skip_change,
            "sensors_visible": sensors_changed,
            "interpretation": (
                "PASS: Adaptation detected" if adaptation_detected else
                "WARN: No adaptation (skip rate didn't increase under stress)"
            ),
        }

    # ========================================================================
    # TEST 3: Control Objective Metrics
    # ========================================================================

    def test_control_objectives(self, trials: int = 20, power_cap: float = 60.0) -> Dict:
        """
        Test control objective metrics that matter for production.

        Metrics:
        1. Time-in-band: % of time under power cap
        2. p95 TPOT under stress
        3. Energy per token under stress
        """
        print("\n  [TEST 3] Control Objective Metrics...")

        results = {"baseline": {}, "stressed": {}}

        for condition in ["baseline", "stressed"]:
            print(f"    Testing {condition}...")

            if condition == "stressed":
                self.gpu_stress.start(intensity=0.5)
                time.sleep(0.5)

            powers = []
            latencies = []
            energies_per_token = []
            tokens_generated = []

            for _ in range(trials):
                prompt = random.choice(self.prompts)
                enc = self.tokenizer(prompt, return_tensors="pt")
                input_ids = enc.input_ids.to(self.device)
                attention_mask = torch.ones_like(input_ids)

                self._inject_real_sensors()

                self.power_sampler.start()
                t0 = time.perf_counter()

                with torch.no_grad():
                    outputs = self.model.generate(
                        input_ids,
                        attention_mask=attention_mask,
                        max_new_tokens=32,
                        do_sample=False,
                        pad_token_id=self.tokenizer.pad_token_id,
                    )

                torch.cuda.synchronize()
                elapsed = time.perf_counter() - t0
                power_stats = self.power_sampler.stop()

                num_tokens = outputs.shape[1] - input_ids.shape[1]
                tokens_generated.append(num_tokens)

                # TPOT (time per output token)
                tpot = (elapsed * 1000) / max(1, num_tokens)
                latencies.append(tpot)

                powers.append(power_stats["avg_power_w"])

                # Energy per token
                if num_tokens > 0:
                    energies_per_token.append(power_stats["energy_j"] / num_tokens)

            if condition == "stressed":
                self.gpu_stress.stop()
                time.sleep(0.3)

            # Time in band
            time_in_band = sum(1 for p in powers if p <= power_cap) / len(powers)

            results[condition] = {
                "time_in_band": time_in_band,
                "avg_power_w": float(np.mean(powers)),
                "power_std": float(np.std(powers)),
                "tpot_p50_ms": float(np.percentile(latencies, 50)),
                "tpot_p95_ms": float(np.percentile(latencies, 95)),
                "energy_per_token_j": float(np.mean(energies_per_token)),
                "tokens_per_trial": float(np.mean(tokens_generated)),
            }

            print(f"      Time-in-band: {time_in_band*100:.1f}%, "
                  f"TPOT p95: {results[condition]['tpot_p95_ms']:.1f}ms, "
                  f"J/tok: {results[condition]['energy_per_token_j']:.3f}")

        self._reset_blocks()

        # Compare baseline vs stressed
        baseline_tib = results["baseline"]["time_in_band"]
        stressed_tib = results["stressed"]["time_in_band"]

        return {
            "results": results,
            "power_cap_w": power_cap,
            "time_in_band_degradation": baseline_tib - stressed_tib,
            "tpot_increase_pct": (
                (results["stressed"]["tpot_p95_ms"] - results["baseline"]["tpot_p95_ms"]) /
                results["baseline"]["tpot_p95_ms"] * 100
            ),
            "energy_increase_pct": (
                (results["stressed"]["energy_per_token_j"] - results["baseline"]["energy_per_token_j"]) /
                results["baseline"]["energy_per_token_j"] * 100
            ),
            "interpretation": (
                f"Under stress: time-in-band dropped {(baseline_tib - stressed_tib)*100:.1f}%, "
                f"TPOT p95 increased {(results['stressed']['tpot_p95_ms'] - results['baseline']['tpot_p95_ms']):.1f}ms"
            ),
        }

    # ========================================================================
    # TEST 4: Sensor Causality (Real Lag Test)
    # ========================================================================

    def test_sensor_causality(self, trials: int = 10) -> Dict:
        """
        Test that the model actually uses real-time sensor values.

        Method: Read sensors, then add artificial delay before inference.
        If loop is real, stale sensors should degrade control.
        """
        print("\n  [TEST 4] Sensor Causality (Real Lag Test)...")

        lags = [0, 100, 500]  # ms
        results = {}

        for lag_ms in lags:
            print(f"    Testing lag={lag_ms}ms...")

            # Start some GPU activity
            self.gpu_stress.start(intensity=0.4)
            time.sleep(0.5)

            energies = []
            gate_correlations = []

            for _ in range(trials):
                prompt = random.choice(self.prompts)
                enc = self.tokenizer(prompt, return_tensors="pt")
                input_ids = enc.input_ids.to(self.device)
                attention_mask = torch.ones_like(input_ids)

                # Read sensors NOW
                sensors = self.sensor_hub.read_tensor()

                # Introduce artificial lag
                time.sleep(lag_ms / 1000.0)

                # Read sensors AGAIN to see how much they changed
                sensors_after = self.sensor_hub.read_tensor()

                # Use the OLD (lagged) sensors
                for b in self.skip_blocks.values():
                    b.sensors = sensors  # Deliberately stale

                self.power_sampler.start()

                with torch.no_grad():
                    outputs = self.model.generate(
                        input_ids,
                        attention_mask=attention_mask,
                        max_new_tokens=16,
                        do_sample=False,
                        pad_token_id=self.tokenizer.pad_token_id,
                    )

                torch.cuda.synchronize()
                power_stats = self.power_sampler.stop()

                energies.append(power_stats["energy_j"])

                # Measure how different old vs new sensors were
                sensor_drift = torch.norm(sensors_after - sensors).item()
                gate_correlations.append(sensor_drift)

            self.gpu_stress.stop()
            time.sleep(0.3)

            results[f"lag_{lag_ms}ms"] = {
                "energy_mean": float(np.mean(energies)),
                "energy_std": float(np.std(energies)),
                "sensor_drift_mean": float(np.mean(gate_correlations)),
            }

            print(f"      Energy: {results[f'lag_{lag_ms}ms']['energy_mean']:.2f}J, "
                  f"Sensor drift: {results[f'lag_{lag_ms}ms']['sensor_drift_mean']:.4f}")

        self._reset_blocks()

        # Check if energy degrades with lag (it should if loop is real)
        energy_0 = results["lag_0ms"]["energy_mean"]
        energy_500 = results["lag_500ms"]["energy_mean"]
        degradation = (energy_500 - energy_0) / energy_0 * 100

        return {
            "lag_results": results,
            "degradation_pct": degradation,
            "causality_supported": degradation > 3,  # >3% degradation suggests real dependence
            "interpretation": (
                f"Energy increased {degradation:.1f}% with 500ms lag" if degradation > 0 else
                f"Energy decreased {-degradation:.1f}% with lag (no causal dependence)"
            ),
        }

    # ========================================================================
    # RUN ALL TESTS
    # ========================================================================

    def run_all(self, trials: int = 15) -> Dict:
        """Run all tests and compile results."""
        print("\n" + "=" * 70)
        print("z38 REAL SENSOR VALIDATION SUITE")
        print("=" * 70)

        results = {
            "timestamp": datetime.now().isoformat(),
            "trials_per_test": trials,
            "bonferroni_alpha": self.BONFERRONI_ALPHA,
        }

        # Test 1: Real Sensor Response
        results["test1_real_sensors"] = self.test_real_sensor_response(trials=trials)

        # Test 2: Disturbance Adaptation (FIXED)
        results["test2_disturbance"] = self.test_disturbance_adaptation(trials=trials)

        # Test 3: Control Objectives
        results["test3_control"] = self.test_control_objectives(trials=trials)

        # Test 4: Sensor Causality
        results["test4_causality"] = self.test_sensor_causality(trials=min(trials, 10))

        # Summary
        print("\n" + "=" * 70)
        print("SUMMARY")
        print("=" * 70)

        t1 = results["test1_real_sensors"]
        t2 = results["test2_disturbance"]
        t3 = results["test3_control"]
        t4 = results["test4_causality"]

        print(f"\n  Test 1 (Real Sensor Response):")
        print(f"    Cohen's d: {t1['cohens_d']:.3f} ({t1['cohens_d_interpretation']})")
        print(f"    Sensors visible: {'✅' if t1['sensor_visible'] else '❌'} (ΔP={t1['avg_power_delta']:.1f}W)")
        print(f"    Passed: {'✅' if t1['passed'] else '❌'}")

        print(f"\n  Test 2 (Disturbance Adaptation):")
        print(f"    Skip rate change: {t2['skip_rate_change']:+.3f}")
        print(f"    Sensors visible: {'✅' if t2['sensors_visible'] else '❌'}")
        print(f"    Adaptation: {'✅' if t2['adaptation_detected'] else '❌'}")

        print(f"\n  Test 3 (Control Objectives):")
        print(f"    Time-in-band (baseline): {t3['results']['baseline']['time_in_band']*100:.1f}%")
        print(f"    Time-in-band (stressed): {t3['results']['stressed']['time_in_band']*100:.1f}%")
        print(f"    TPOT increase under stress: {t3['tpot_increase_pct']:.1f}%")

        print(f"\n  Test 4 (Sensor Causality):")
        print(f"    Energy degradation with lag: {t4['degradation_pct']:.1f}%")
        print(f"    Causality supported: {'✅' if t4['causality_supported'] else '❌'}")

        print("\n" + "=" * 70)

        return results


# ============================================================================
# MAIN
# ============================================================================

def load_model(checkpoint_path: str, device: str = "cuda"):
    """Load model with skip blocks."""
    print(f"\n[MODEL] Loading from {checkpoint_path}...")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint.get("config", {})

    print(f"  Checkpoint step: {checkpoint.get('step', 'unknown')}")

    # Load base model
    base_model_name = config.get("base_model", "Qwen/Qwen2.5-1.5B")
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.bfloat16,
        device_map=device,
    )

    hidden_size = model.config.hidden_size
    dtype = next(model.parameters()).dtype

    print(f"  Hidden size: {hidden_size}")

    # Create skip blocks
    gate_layers = config.get("gate_layers", [7, 11, 15, 19, 23])
    print(f"  Gate layers: {gate_layers}")

    skip_blocks = {}
    for layer_idx in gate_layers:
        if hasattr(model.model, "layers") and layer_idx < len(model.model.layers):
            original_layer = model.model.layers[layer_idx]
            wrapper = SkipBlockWrapper(original_layer, layer_idx, hidden_size, device, dtype)
            model.model.layers[layer_idx] = wrapper
            skip_blocks[layer_idx] = wrapper

    print(f"  Created {len(skip_blocks)} skip blocks")

    # Load trained weights
    state_dict = checkpoint.get("model_state_dict", checkpoint.get("state_dict", {}))

    for layer_idx, block in skip_blocks.items():
        prefix = f"skip_blocks.{layer_idx}."
        block_state = {k.replace(prefix, ""): v for k, v in state_dict.items() if k.startswith(prefix)}
        if block_state:
            block.load_state_dict(block_state, strict=False)

    model.eval()
    return model, tokenizer, skip_blocks


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Load model
    checkpoint_path = "models/z34_fullloop/step_300.pt"
    model, tokenizer, skip_blocks = load_model(checkpoint_path, device)

    # Initialize sensor hub
    sensor_hub = CanonicalSensorHub()

    # Run validation
    validator = RealSensorValidator(model, tokenizer, sensor_hub, skip_blocks, device)
    results = validator.run_all(trials=15)

    # Save results
    output_path = Path("results/z38_real_sensor_validation.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
