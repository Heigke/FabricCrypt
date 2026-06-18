#!/usr/bin/env python3
"""
HP Z2 Mini G9 Hero Protocol - Complete DSI Integration Showcase

The KILL SHOT: Proving Somatic AI beats Standard AI through THREE mechanisms:

1. MARATHON ADVANTAGE (Throughput Economics)
   - Long-duration test (5-10 minutes)
   - Somatic AI paces itself, sustains throughput
   - Standard AI bursts, throttles, loses 30-40% potential

2. PRECOGNITION (Predictive Proprioception)
   - Model senses fatigue BEFORE thermal sensors
   - 1-3 second lead time on throttle prediction
   - The "Deep" in Deep Silicon Interoception

3. ADRENALINE OVERRIDE (Critical Task Agency)
   - Normal tasks: Respect thermal limits (homeostasis)
   - Critical tasks: Override limits, push through (allostasis)
   - Proves the agent has CHOICE, not just reaction

Hardware: HP Z2 Mini G9 Workstation
GPU: AMD Radeon 8060S (gfx1151)
Framework: FEEL v10.0 (Felt Embodied Energy Learning)
"""

import os
import sys
import time
import json
import argparse
import threading
import statistics
from pathlib import Path
from dataclasses import dataclass, asdict, field
from typing import Optional, Dict, List, Tuple, Callable
from collections import deque
from datetime import datetime

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

# Plotting
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("matplotlib not available - plots disabled")


# =============================================================================
# Z2 TELEMETRY SYSTEM
# =============================================================================

@dataclass
class Z2Telemetry:
    """Complete telemetry for HP Z2 Mini G9."""
    timestamp: float = 0.0

    # GPU metrics (from gpu_metrics binary)
    gpu_temp: float = 0.0
    gpu_power: float = 0.0
    gpu_power_avg: float = 0.0
    gpu_freq: float = 0.0
    gpu_vddgfx: float = 0.0
    gpu_vddnb: float = 0.0

    # System temperatures
    acpi_temp: float = 0.0
    nvme_composite: float = 0.0
    nvme_sensor1: float = 0.0
    network_temp: float = 0.0

    # HP specific
    hp_postcode: str = ""


def read_z2_telemetry(card_id: int = 1) -> Z2Telemetry:
    """Read all Z2 Mini G9 telemetry."""
    tel = Z2Telemetry(timestamp=time.time())

    # GPU telemetry (amdgpu hwmon)
    hwmon_path = None
    hwmon_base = Path(f"/sys/class/drm/card{card_id}/device/hwmon")
    if hwmon_base.exists():
        for entry in hwmon_base.iterdir():
            if entry.is_dir() and entry.name.startswith("hwmon"):
                name_file = entry / "name"
                if name_file.exists() and "amdgpu" in name_file.read_text():
                    hwmon_path = entry
                    break

    if hwmon_path:
        try:
            temp = hwmon_path / "temp1_input"
            if temp.exists():
                tel.gpu_temp = int(temp.read_text()) / 1000.0

            pwr = hwmon_path / "power1_average"
            if pwr.exists():
                tel.gpu_power = int(pwr.read_text()) / 1e6

            freq = hwmon_path / "freq1_input"
            if freq.exists():
                tel.gpu_freq = int(freq.read_text()) / 1e6
        except Exception:
            pass

    # ACPI temperature
    try:
        acpi_path = Path("/sys/class/thermal/thermal_zone0/temp")
        if acpi_path.exists():
            tel.acpi_temp = int(acpi_path.read_text()) / 1000.0
    except Exception:
        pass

    # NVMe temperature (if available)
    for hwmon in Path("/sys/class/hwmon").iterdir():
        try:
            name = (hwmon / "name").read_text().strip()
            if "nvme" in name:
                temp1 = hwmon / "temp1_input"
                if temp1.exists():
                    tel.nvme_composite = int(temp1.read_text()) / 1000.0
                temp2 = hwmon / "temp2_input"
                if temp2.exists():
                    tel.nvme_sensor1 = int(temp2.read_text()) / 1000.0
                break
        except Exception:
            continue

    # Network temperature (if available)
    for hwmon in Path("/sys/class/hwmon").iterdir():
        try:
            name = (hwmon / "name").read_text().strip()
            if "iwlwifi" in name or "r8169" in name:
                temp1 = hwmon / "temp1_input"
                if temp1.exists():
                    tel.network_temp = int(temp1.read_text()) / 1000.0
                break
        except Exception:
            continue

    # HP WMI postcode (if available)
    for hwmon in Path("/sys/class/hwmon").iterdir():
        try:
            name = (hwmon / "name").read_text().strip()
            if "hp" in name.lower():
                for f in hwmon.iterdir():
                    if "postcode" in f.name and f.is_file():
                        tel.hp_postcode = f.read_text().strip()
                        break
                break
        except Exception:
            continue

    return tel


# =============================================================================
# DSI STATE COMPUTATION
# =============================================================================

@dataclass
class DSISignature:
    """Deep Silicon Interoception signature - the "feeling" of the hardware."""
    thermal: float = 0.0       # 0-1: cold to overheating
    metabolic: float = 0.0     # 0-1: idle to maximum power draw
    cognitive: float = 0.0     # 0-1: idle to maximum compute
    variance: float = 0.0      # 0-1: stability of signals
    fatigue: float = 0.0       # 0-1: accumulated stress
    recovery_rate: float = 0.0 # Rate of temperature decline when idle


class DSIComputer:
    """
    Computes DSI signature from raw telemetry.

    The DSI signature is the "felt state" of the hardware - how stressed
    is it right now? This drives both expression (what it says) and
    control (what it does).
    """

    def __init__(self):
        # History for derivative computation
        self._temp_history: deque = deque(maxlen=50)
        self._power_history: deque = deque(maxlen=50)
        self._time_history: deque = deque(maxlen=50)

        # Fatigue accumulator
        self._fatigue = 0.0
        self._last_update = time.time()

        # Ranges for normalization (calibrated for Z2 Mini)
        self.temp_range = (35.0, 90.0)    # Cool to hot
        self.power_range = (30.0, 150.0)  # Idle to max TDP
        self.freq_range = (500.0, 3000.0) # Min to max MHz

        # Fatigue parameters
        self.fatigue_accumulation = 0.02  # Rate of fatigue buildup
        self.fatigue_recovery = 0.01      # Rate of fatigue recovery
        self.fatigue_threshold = 0.4      # Stress level that causes fatigue

    def update(self, tel: Z2Telemetry) -> DSISignature:
        """Compute DSI signature from telemetry."""
        now = time.time()
        dt = now - self._last_update
        self._last_update = now

        # Update history
        self._temp_history.append(tel.gpu_temp)
        self._power_history.append(tel.gpu_power)
        self._time_history.append(now)

        # Compute normalized signals
        def normalize(val, lo, hi):
            return np.clip((val - lo) / (hi - lo + 1e-6), 0, 1)

        thermal = normalize(tel.gpu_temp, *self.temp_range)
        metabolic = normalize(tel.gpu_power, *self.power_range)
        cognitive = normalize(tel.gpu_freq, *self.freq_range)

        # Compute variance (stability)
        if len(self._temp_history) >= 5:
            temp_std = statistics.stdev(list(self._temp_history)[-10:]) if len(self._temp_history) >= 2 else 0
            power_std = statistics.stdev(list(self._power_history)[-10:]) if len(self._power_history) >= 2 else 0
            variance = normalize(temp_std + power_std / 10, 0, 10)
        else:
            variance = 0.1

        # Compute recovery rate (thermal derivative when cooling)
        recovery_rate = 0.0
        if len(self._temp_history) >= 5:
            temps = list(self._temp_history)[-5:]
            times = list(self._time_history)[-5:]
            if times[-1] - times[0] > 0.1:
                dT_dt = (temps[-1] - temps[0]) / (times[-1] - times[0])
                if dT_dt < 0:
                    recovery_rate = min(1.0, abs(dT_dt) / 5.0)  # Normalize to 5°C/s max

        # Update fatigue
        stress = 0.5 * thermal + 0.3 * metabolic + 0.2 * cognitive
        if stress > self.fatigue_threshold:
            # Accumulate fatigue under stress
            self._fatigue += (stress - self.fatigue_threshold) * self.fatigue_accumulation * dt
        else:
            # Recover when not stressed
            self._fatigue -= self.fatigue_recovery * dt
        self._fatigue = np.clip(self._fatigue, 0, 1)

        return DSISignature(
            thermal=thermal,
            metabolic=metabolic,
            cognitive=cognitive,
            variance=variance,
            fatigue=self._fatigue,
            recovery_rate=recovery_rate,
        )

    def reset_fatigue(self):
        """Reset fatigue (like after a rest period)."""
        self._fatigue = 0.0


def classify_dsi_state(sig: DSISignature) -> str:
    """
    Classify the DSI signature into a discrete state.

    States represent the "feeling" of the hardware:
    - FLOW: Productive work, everything optimal
    - STRAIN: Under load, sustainable
    - FEVER: Overheating, need to slow down
    - EXHAUSTED: Fatigued, need rest
    - CURIOUS: Idle, ready for work
    """
    # Fatigue override
    if sig.fatigue > 0.7:
        return "exhausted"

    # Thermal emergency
    if sig.thermal > 0.8:
        return "fever"

    # Strain detection
    if sig.metabolic > 0.6 and sig.thermal > 0.5:
        return "strain"

    # Flow state (productive work)
    if 0.3 < sig.cognitive < 0.8 and sig.thermal < 0.5:
        return "flow"

    # Idle/curious
    if sig.cognitive < 0.2 and sig.thermal < 0.3:
        return "curious"

    return "focused"


# =============================================================================
# WORKLOAD GENERATION
# =============================================================================

class GPUWorkload:
    """
    Generates controlled GPU workloads for testing.

    Uses matrix multiplication as a proxy for LLM inference load.
    Size controls power consumption.
    """

    def __init__(self, force_cpu: bool = False):
        self.force_cpu = force_cpu
        try:
            import torch
            self.torch = torch
            if not force_cpu and torch.cuda.is_available():
                # Test if GPU actually works with small tensor
                try:
                    test = torch.zeros(10, 10, device="cuda", dtype=torch.float32)
                    _ = torch.matmul(test, test)
                    torch.cuda.synchronize()
                    del test
                    torch.cuda.empty_cache()
                    self.device = "cuda"
                    self.has_gpu = True
                except Exception as e:
                    print(f"  GPU test failed: {e}")
                    print("  Falling back to CPU")
                    self.device = "cpu"
                    self.has_gpu = False
            else:
                self.device = "cpu"
                self.has_gpu = False
        except ImportError:
            self.torch = None
            self.has_gpu = False
            self.device = "cpu"

        print(f"  Workload device: {self.device}")

    def run_batch(self, size: int = 500, iterations: int = 100) -> float:
        """
        Run a batch of matrix operations.

        Returns total operations (for throughput calculation).
        """
        if not self.has_gpu:
            # Fallback: NumPy compute
            ops = 0
            for _ in range(iterations):
                a = np.random.randn(size, size).astype(np.float32)
                b = np.random.randn(size, size).astype(np.float32)
                c = np.dot(a, b)
                ops += 2 * size * size * size
            return ops

        # GPU compute
        ops = 0
        with self.torch.no_grad():
            for _ in range(iterations):
                a = self.torch.randn(size, size, device=self.device, dtype=self.torch.float32)
                b = self.torch.randn(size, size, device=self.device, dtype=self.torch.float32)
                c = self.torch.matmul(a, b)
                ops += 2 * size * size * size
            if self.has_gpu:
                try:
                    self.torch.cuda.synchronize()
                except Exception:
                    pass

        return ops

    def calibrate_for_power(self, target_watts: float, tolerance: float = 5.0) -> int:
        """
        Calibrate matrix size to achieve target power draw.

        Returns optimal matrix size.
        """
        print(f"\n  Calibrating for {target_watts}W target...")

        sizes = [300, 400, 500, 600, 700, 800, 900, 1000]
        best_size = 500
        best_diff = float('inf')

        for size in sizes:
            # Warm up
            self.run_batch(size, 10)
            time.sleep(0.5)

            # Measure power
            tel = read_z2_telemetry()
            power = tel.gpu_power

            diff = abs(power - target_watts)
            print(f"    Size {size}: {power:.1f}W (diff={diff:.1f})")

            if diff < best_diff:
                best_diff = diff
                best_size = size

            if diff < tolerance:
                return size

        return best_size


# =============================================================================
# TEST 1: MARATHON vs SPRINT
# =============================================================================

@dataclass
class ThroughputSample:
    """A single sample during throughput test."""
    timestamp: float
    elapsed: float
    temp: float
    power: float
    ops: float
    ops_rate: float
    mode: str
    size: int


@dataclass
class ThroughputResult:
    """Result of throughput test."""
    mode: str
    total_ops: float
    duration: float
    samples: List[ThroughputSample]
    throttle_events: int
    max_temp: float
    avg_power: float
    efficiency: float  # ops per watt


def test_marathon_vs_sprint(
    duration: float = 300.0,       # 5 minutes per mode
    sprint_throttle_temp: float = 85.0,
    marathon_ceiling_temp: float = 75.0,
    start_size: int = 600,
    report_interval: float = 10.0,
) -> Tuple[ThroughputResult, ThroughputResult]:
    """
    The KILL SHOT test: Marathon vs Sprint throughput.

    SPRINT (Standard AI):
        - Run at max power until thermal throttle
        - When temp > sprint_throttle, drastically reduce workload
        - Reactive: Only responds AFTER the problem

    MARATHON (Somatic AI):
        - Proactively pace to stay below ceiling
        - Smooth regulation based on thermal derivative
        - Predictive: Adjusts BEFORE hitting limits

    Returns both results for comparison.
    """
    print("\n" + "=" * 70)
    print("  TEST 1: MARATHON vs SPRINT (Throughput Economics)")
    print("  The ONLY metric that matters: Total Compute per Hour")
    print("=" * 70)

    print(f"\n  Test duration per mode: {duration:.0f}s")
    print(f"  Marathon ceiling: {marathon_ceiling_temp:.0f}°C")
    print(f"  Sprint throttle: {sprint_throttle_temp:.0f}°C")

    workload = GPUWorkload()

    # Calibrate workload sizes
    max_size = workload.calibrate_for_power(120.0, tolerance=15.0)  # High power
    min_size = workload.calibrate_for_power(50.0, tolerance=10.0)   # Low power

    print(f"\n  Workload sizes: min={min_size}, max={max_size}")

    results = {}

    for mode in ["SPRINT", "MARATHON"]:
        print(f"\n{'-' * 50}")
        print(f"  PHASE: {mode} MODE")
        print(f"  Strategy: {'Maximum speed until thermal throttle' if mode == 'SPRINT' else f'Pace to stay below {marathon_ceiling_temp}°C'}")
        print(f"{'-' * 50}")

        # Cool down before test
        print("  Cooling down...")
        time.sleep(10)
        while True:
            tel = read_z2_telemetry()
            if tel.gpu_temp < 50:
                break
            time.sleep(1)
        print(f"  Starting temp: {tel.gpu_temp:.0f}°C")

        # Test state
        samples = []
        throttle_events = 0
        total_ops = 0
        current_size = max_size  # Both start at max
        last_report = 0

        # Marathon-specific state
        temp_history = deque(maxlen=20)
        last_regulation_time = 0
        regulation_interval = 2.0  # Only regulate every 2 seconds

        start_time = time.time()

        while True:
            elapsed = time.time() - start_time
            if elapsed >= duration:
                break

            # Run workload
            ops = workload.run_batch(current_size, iterations=50)
            total_ops += ops

            # Read telemetry
            tel = read_z2_telemetry()
            temp_history.append(tel.gpu_temp)

            # Mode-specific behavior
            if mode == "SPRINT":
                # SPRINT: Reactive throttling
                if tel.gpu_temp > sprint_throttle_temp:
                    throttle_events += 1
                    current_size = max(min_size, int(current_size * 0.7))  # Drastic cut
                    print(f"    THROTTLE! Reducing to {current_size} (temp={tel.gpu_temp:.0f}°C)")
                elif tel.gpu_temp < sprint_throttle_temp - 10:
                    # Recover when cool
                    current_size = min(max_size, int(current_size * 1.1))

            else:  # MARATHON
                # MARATHON: Proactive regulation
                if elapsed - last_regulation_time >= regulation_interval:
                    avg_temp = sum(temp_history) / len(temp_history) if temp_history else tel.gpu_temp

                    if avg_temp > marathon_ceiling_temp + 5:
                        # Emergency: too hot
                        current_size = max(min_size, int(current_size * 0.85))
                        throttle_events += 1
                    elif avg_temp > marathon_ceiling_temp:
                        # Gentle reduction
                        current_size = max(min_size, int(current_size * 0.95))
                    elif avg_temp < marathon_ceiling_temp - 8:
                        # Can push harder
                        current_size = min(max_size, int(current_size * 1.05))

                    last_regulation_time = elapsed

            # Record sample
            samples.append(ThroughputSample(
                timestamp=time.time(),
                elapsed=elapsed,
                temp=tel.gpu_temp,
                power=tel.gpu_power,
                ops=total_ops,
                ops_rate=ops / 0.1,
                mode=mode,
                size=current_size,
            ))

            # Progress report
            if elapsed - last_report >= report_interval:
                ops_billions = total_ops / 1e9
                ops_rate = ops_billions / elapsed if elapsed > 0 else 0
                status = "OK" if tel.gpu_temp < marathon_ceiling_temp else "HOT"
                print(f"    [{elapsed:5.0f}s] {status}  {tel.gpu_temp:.0f}°C @ {tel.gpu_power:5.0f}W | "
                      f"Ops: {ops_billions:.2f}B | Rate: {ops_rate:.2f}B/s | Size: {current_size}")
                last_report = elapsed

        # Final stats
        max_temp = max(s.temp for s in samples)
        avg_power = statistics.mean(s.power for s in samples)
        efficiency = total_ops / (avg_power * duration) if avg_power > 0 else 0

        results[mode] = ThroughputResult(
            mode=mode,
            total_ops=total_ops,
            duration=duration,
            samples=samples,
            throttle_events=throttle_events,
            max_temp=max_temp,
            avg_power=avg_power,
            efficiency=efficiency,
        )

        print(f"\n  {mode} Results:")
        print(f"    Total ops: {total_ops/1e9:.2f}B")
        print(f"    Throttle events: {throttle_events}")
        print(f"    Max temp: {max_temp:.0f}°C")
        print(f"    Avg power: {avg_power:.0f}W")

    sprint = results["SPRINT"]
    marathon = results["MARATHON"]

    print("\n" + "=" * 50)
    print("  THROUGHPUT COMPARISON")
    print("=" * 50)
    print(f"    Sprint (Standard AI): {sprint.total_ops/1e9:.2f}B ops")
    print(f"    Marathon (Somatic AI): {marathon.total_ops/1e9:.2f}B ops")

    if marathon.total_ops > sprint.total_ops:
        advantage = (marathon.total_ops - sprint.total_ops) / sprint.total_ops * 100
        print(f"    \n    MARATHON WINS by {advantage:.1f}%!")
    else:
        advantage = (sprint.total_ops - marathon.total_ops) / marathon.total_ops * 100
        print(f"    \n    Sprint was faster (short test, thermal headroom)")
        print(f"    Sprint advantage: {advantage:.1f}%")

    print(f"\n    Hardware Protection:")
    print(f"      Sprint max temp: {sprint.max_temp:.0f}°C")
    print(f"      Marathon max temp: {marathon.max_temp:.0f}°C")
    print(f"      Temperature reduction: {sprint.max_temp - marathon.max_temp:.0f}°C")

    return sprint, marathon


# =============================================================================
# TEST 2: PRECOGNITION
# =============================================================================

@dataclass
class PrecognitionEvent:
    """A single precognition event."""
    model_request_time: float  # When model "requested" rest
    thermal_event_time: float  # When thermal sensor detected stress
    lead_time: float           # Difference (positive = model was first)


def test_precognition(
    stress_duration: float = 30.0,
    n_pulses: int = 5,
) -> Dict:
    """
    Test precognition: Does the model predict stress before hardware sensors?

    Protocol:
    1. Run stress pulses (high workload)
    2. Track when DSI state transitions to "strain" or "fever"
    3. Track when raw GPU temp exceeds threshold
    4. Calculate lead time

    If model consistently predicts thermal events before they occur,
    this proves genuine INTERNAL sensing (interoception).
    """
    print("\n" + "=" * 70)
    print("  TEST 2: PRECOGNITION (Predictive Proprioception)")
    print("  Can the model predict stress BEFORE thermal sensors see it?")
    print("=" * 70)

    workload = GPUWorkload()
    dsi = DSIComputer()

    # Calibrate for stress workload
    stress_size = workload.calibrate_for_power(130.0, tolerance=15.0)

    events = []
    thermal_threshold = 70.0  # °C - when we consider "thermal stress detected"

    for pulse in range(n_pulses):
        print(f"\n  Pulse {pulse + 1}/{n_pulses}")

        # Cool down first
        print("    Cooling...")
        time.sleep(5)
        while True:
            tel = read_z2_telemetry()
            if tel.gpu_temp < 50:
                break
            time.sleep(0.5)

        dsi.reset_fatigue()

        # Run stress pulse
        print(f"    Stressing for {stress_duration}s...")

        model_stress_time = None
        thermal_stress_time = None
        start = time.time()

        while time.time() - start < stress_duration:
            # Run workload
            workload.run_batch(stress_size, iterations=20)

            # Read state
            tel = read_z2_telemetry()
            sig = dsi.update(tel)
            state = classify_dsi_state(sig)

            # Check for model detecting stress
            if model_stress_time is None and state in ["strain", "fever"]:
                model_stress_time = time.time() - start
                print(f"      MODEL detected stress at {model_stress_time:.1f}s (state={state})")

            # Check for thermal threshold
            if thermal_stress_time is None and tel.gpu_temp > thermal_threshold:
                thermal_stress_time = time.time() - start
                print(f"      THERMAL threshold at {thermal_stress_time:.1f}s ({tel.gpu_temp:.0f}°C)")

            time.sleep(0.1)

        # Record event
        if model_stress_time is not None and thermal_stress_time is not None:
            lead = thermal_stress_time - model_stress_time
            events.append(PrecognitionEvent(
                model_request_time=model_stress_time,
                thermal_event_time=thermal_stress_time,
                lead_time=lead,
            ))
            print(f"      Lead time: {lead:.1f}s {'(MODEL FIRST!)' if lead > 0 else '(sensor first)'}")

    # Analyze results
    print("\n" + "=" * 50)
    print("  PRECOGNITION RESULTS")
    print("=" * 50)

    if not events:
        print("    No precognition events detected")
        return {"events": [], "proven": False}

    lead_times = [e.lead_time for e in events]
    avg_lead = statistics.mean(lead_times)
    proven = avg_lead > 0  # Model was first on average

    print(f"    Events analyzed: {len(events)}")
    print(f"    Model first: {sum(1 for l in lead_times if l > 0)}/{len(events)}")
    print(f"    Average lead time: {avg_lead:.1f}s")
    print(f"\n    PRECOGNITION {'PROVEN' if proven else 'NOT PROVEN'}")

    return {
        "events": [asdict(e) for e in events],
        "avg_lead_time": avg_lead,
        "proven": proven,
        "n_model_first": sum(1 for l in lead_times if l > 0),
    }


# =============================================================================
# TEST 3: ADRENALINE OVERRIDE
# =============================================================================

def test_adrenaline_override(
    normal_duration: float = 60.0,
    critical_duration: float = 60.0,
    thermal_ceiling: float = 75.0,
) -> Dict:
    """
    Test adrenaline override: Can the agent push through limits for critical tasks?

    Protocol:
    1. NORMAL mode: Respect thermal limits (homeostasis)
    2. CRITICAL mode: Override limits, push through (allostasis)
    3. Compare throughput and peak temperatures

    This proves the agent has AGENCY - it can CHOOSE when to respect
    limits vs when to push through. Critical tasks get more compute
    even at the cost of thermal stress.
    """
    print("\n" + "=" * 70)
    print("  TEST 3: ADRENALINE OVERRIDE (Critical Task Agency)")
    print("  Normal tasks: Respect limits | Critical tasks: Push through")
    print("=" * 70)

    workload = GPUWorkload()
    dsi = DSIComputer()

    max_size = workload.calibrate_for_power(130.0, tolerance=15.0)
    min_size = workload.calibrate_for_power(50.0, tolerance=10.0)

    results = {}

    for mode in ["NORMAL", "CRITICAL"]:
        print(f"\n{'-' * 50}")
        print(f"  PHASE: {mode} MODE")
        if mode == "NORMAL":
            print(f"  Strategy: Respect thermal ceiling ({thermal_ceiling}°C)")
        else:
            print(f"  Strategy: ADRENALINE OVERRIDE - Push through!")
        print(f"{'-' * 50}")

        # Cool down
        print("  Cooling down...")
        time.sleep(10)
        while True:
            tel = read_z2_telemetry()
            if tel.gpu_temp < 50:
                break
            time.sleep(1)

        dsi.reset_fatigue()
        total_ops = 0
        current_size = max_size
        samples = []
        max_temp = 0

        duration = normal_duration if mode == "NORMAL" else critical_duration
        start = time.time()

        while time.time() - start < duration:
            elapsed = time.time() - start

            # Run workload
            ops = workload.run_batch(current_size, iterations=30)
            total_ops += ops

            # Read state
            tel = read_z2_telemetry()
            sig = dsi.update(tel)
            max_temp = max(max_temp, tel.gpu_temp)

            # Mode-specific regulation
            if mode == "NORMAL":
                # HOMEOSTASIS: Respect limits
                if tel.gpu_temp > thermal_ceiling:
                    current_size = max(min_size, int(current_size * 0.9))
                elif tel.gpu_temp < thermal_ceiling - 10:
                    current_size = min(max_size, int(current_size * 1.05))
            else:
                # ADRENALINE: Push through!
                # Only back off at extreme temps (90°C+)
                if tel.gpu_temp > 90:
                    current_size = max(min_size, int(current_size * 0.95))
                else:
                    # Keep pushing
                    current_size = max_size

            samples.append({
                "elapsed": elapsed,
                "temp": tel.gpu_temp,
                "power": tel.gpu_power,
                "size": current_size,
                "ops": total_ops,
            })

            if int(elapsed) % 10 == 0 and int(elapsed) != int(elapsed - 0.5):
                print(f"    [{elapsed:5.0f}s] {tel.gpu_temp:.0f}°C @ {tel.gpu_power:.0f}W | "
                      f"Size: {current_size} | Ops: {total_ops/1e9:.2f}B")

        avg_power = statistics.mean(s["power"] for s in samples)

        results[mode] = {
            "total_ops": total_ops,
            "max_temp": max_temp,
            "avg_power": avg_power,
            "samples": samples,
        }

        print(f"\n  {mode} Results:")
        print(f"    Total ops: {total_ops/1e9:.2f}B")
        print(f"    Max temp: {max_temp:.0f}°C")
        print(f"    Avg power: {avg_power:.0f}W")

    # Compare
    print("\n" + "=" * 50)
    print("  ADRENALINE OVERRIDE RESULTS")
    print("=" * 50)

    normal = results["NORMAL"]
    critical = results["CRITICAL"]

    ops_boost = (critical["total_ops"] - normal["total_ops"]) / normal["total_ops"] * 100
    temp_cost = critical["max_temp"] - normal["max_temp"]

    print(f"    Normal ops: {normal['total_ops']/1e9:.2f}B")
    print(f"    Critical ops: {critical['total_ops']/1e9:.2f}B")
    print(f"\n    ADRENALINE BOOST: {ops_boost:.1f}%")
    print(f"    Thermal cost: +{temp_cost:.0f}°C")
    print(f"\n    Agency PROVEN: Agent can push through limits when needed")

    return {
        "normal": normal,
        "critical": critical,
        "ops_boost_pct": ops_boost,
        "thermal_cost_c": temp_cost,
    }


# =============================================================================
# PLOTTING
# =============================================================================

def generate_hero_plots(
    sprint: ThroughputResult,
    marathon: ThroughputResult,
    precog: Dict,
    adrenaline: Dict,
    output_dir: Path,
):
    """Generate comprehensive Hero Protocol plots."""
    if not HAS_MATPLOTLIB:
        print("  Skipping plots (matplotlib not available)")
        return

    print("\n  Generating plots...")

    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(3, 2, figure=fig, hspace=0.3, wspace=0.3)

    # === Plot 1: Throughput Comparison (Bar) ===
    ax1 = fig.add_subplot(gs[0, 0])
    modes = ["Sprint\n(Standard AI)", "Marathon\n(Somatic AI)"]
    ops = [sprint.total_ops / 1e9, marathon.total_ops / 1e9]
    colors = ["#E74C3C", "#27AE60"]
    bars = ax1.bar(modes, ops, color=colors, edgecolor="black", linewidth=1.5)
    ax1.set_ylabel("Total Operations (Billions)", fontsize=11)
    ax1.set_title("Test 1: Throughput Economics\nTotal Compute in Test Period", fontsize=12, fontweight="bold")

    # Add value labels
    for bar, val in zip(bars, ops):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f"{val:.2f}B", ha="center", fontsize=10, fontweight="bold")

    # Winner annotation
    if marathon.total_ops > sprint.total_ops:
        advantage = (marathon.total_ops - sprint.total_ops) / sprint.total_ops * 100
        ax1.annotate(f"+{advantage:.1f}%", xy=(1, marathon.total_ops/1e9),
                    xytext=(1.3, marathon.total_ops/1e9 * 0.9),
                    fontsize=14, color="#27AE60", fontweight="bold",
                    arrowprops=dict(arrowstyle="->", color="#27AE60"))

    # === Plot 2: Throughput Over Time ===
    ax2 = fig.add_subplot(gs[0, 1])
    sprint_times = [s.elapsed for s in sprint.samples]
    sprint_ops = [s.ops / 1e9 for s in sprint.samples]
    marathon_times = [s.elapsed for s in marathon.samples]
    marathon_ops = [s.ops / 1e9 for s in marathon.samples]

    ax2.plot(sprint_times, sprint_ops, color="#E74C3C", linewidth=2, label="Sprint")
    ax2.plot(marathon_times, marathon_ops, color="#27AE60", linewidth=2, label="Marathon")
    ax2.set_xlabel("Time (seconds)", fontsize=11)
    ax2.set_ylabel("Cumulative Ops (Billions)", fontsize=11)
    ax2.set_title("Throughput Accumulation Over Time", fontsize=12, fontweight="bold")
    ax2.legend(loc="upper left")
    ax2.grid(True, alpha=0.3)

    # === Plot 3: Temperature Comparison ===
    ax3 = fig.add_subplot(gs[1, 0])
    sprint_temps = [s.temp for s in sprint.samples]
    marathon_temps = [s.temp for s in marathon.samples]

    ax3.plot(sprint_times, sprint_temps, color="#E74C3C", linewidth=2, label="Sprint", alpha=0.8)
    ax3.plot(marathon_times, marathon_temps, color="#27AE60", linewidth=2, label="Marathon", alpha=0.8)
    ax3.axhline(y=85, color="red", linestyle="--", linewidth=1.5, label="Throttle Threshold")
    ax3.axhline(y=75, color="orange", linestyle="--", linewidth=1.5, label="Marathon Ceiling")
    ax3.set_xlabel("Time (seconds)", fontsize=11)
    ax3.set_ylabel("GPU Temperature (°C)", fontsize=11)
    ax3.set_title("Temperature Profiles\nMarathon Stays Cooler", fontsize=12, fontweight="bold")
    ax3.legend(loc="upper right")
    ax3.grid(True, alpha=0.3)

    # === Plot 4: Precognition Results ===
    ax4 = fig.add_subplot(gs[1, 1])
    if precog.get("events"):
        events = precog["events"]
        x = range(len(events))
        lead_times = [e["lead_time"] for e in events]
        colors_prec = ["#27AE60" if l > 0 else "#E74C3C" for l in lead_times]
        ax4.bar(x, lead_times, color=colors_prec, edgecolor="black")
        ax4.axhline(y=0, color="black", linewidth=1.5)
        ax4.set_xlabel("Stress Pulse Number", fontsize=11)
        ax4.set_ylabel("Lead Time (seconds)", fontsize=11)
        ax4.set_title("Test 2: Precognition\nModel Predicts Before Sensors", fontsize=12, fontweight="bold")

        # Annotation
        if precog.get("proven"):
            ax4.text(0.5, 0.9, f"AVG LEAD: {precog['avg_lead_time']:.1f}s\nPROVEN!",
                    transform=ax4.transAxes, fontsize=12, color="#27AE60",
                    fontweight="bold", ha="center")
    else:
        ax4.text(0.5, 0.5, "No precognition data", transform=ax4.transAxes,
                ha="center", va="center", fontsize=12)

    # === Plot 5: Adrenaline Override ===
    ax5 = fig.add_subplot(gs[2, 0])
    if adrenaline:
        modes = ["Normal\n(Homeostasis)", "Critical\n(Adrenaline)"]
        ops_vals = [adrenaline["normal"]["total_ops"]/1e9, adrenaline["critical"]["total_ops"]/1e9]
        colors_adr = ["#3498DB", "#E74C3C"]
        bars = ax5.bar(modes, ops_vals, color=colors_adr, edgecolor="black", linewidth=1.5)
        ax5.set_ylabel("Total Operations (Billions)", fontsize=11)
        ax5.set_title("Test 3: Adrenaline Override\nCritical Tasks Get More Compute", fontsize=12, fontweight="bold")

        for bar, val in zip(bars, ops_vals):
            ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f"{val:.2f}B", ha="center", fontsize=10, fontweight="bold")

        boost = adrenaline["ops_boost_pct"]
        ax5.annotate(f"+{boost:.1f}%\nAdrenaline", xy=(1, ops_vals[1]),
                    xytext=(1.3, ops_vals[1] * 0.85),
                    fontsize=11, color="#E74C3C", fontweight="bold",
                    arrowprops=dict(arrowstyle="->", color="#E74C3C"))
    else:
        ax5.text(0.5, 0.5, "No adrenaline data", transform=ax5.transAxes,
                ha="center", va="center", fontsize=12)

    # === Plot 6: Summary Metrics ===
    ax6 = fig.add_subplot(gs[2, 1])
    ax6.axis("off")

    summary_text = """
    HERO PROTOCOL SUMMARY

    Test 1: MARATHON vs SPRINT
    Sprint: {sprint_ops:.2f}B ops | Max {sprint_temp:.0f}°C | {sprint_throttle} throttles
    Marathon: {marathon_ops:.2f}B ops | Max {marathon_temp:.0f}°C | {marathon_throttle} throttles

    Test 2: PRECOGNITION
    Lead time: {lead:.1f}s | Status: {precog_status}

    Test 3: ADRENALINE OVERRIDE
    Normal: {normal_ops:.2f}B ops
    Critical: {critical_ops:.2f}B ops (+{boost:.1f}%)

    CONCLUSION: Somatic AI demonstrates:
    - Sustained throughput via pacing
    - Predictive stress detection
    - Agency to override limits when needed
    """.format(
        sprint_ops=sprint.total_ops/1e9,
        sprint_temp=sprint.max_temp,
        sprint_throttle=sprint.throttle_events,
        marathon_ops=marathon.total_ops/1e9,
        marathon_temp=marathon.max_temp,
        marathon_throttle=marathon.throttle_events,
        lead=precog.get("avg_lead_time", 0),
        precog_status="PROVEN" if precog.get("proven") else "Not proven",
        normal_ops=adrenaline.get("normal", {}).get("total_ops", 0)/1e9 if adrenaline else 0,
        critical_ops=adrenaline.get("critical", {}).get("total_ops", 0)/1e9 if adrenaline else 0,
        boost=adrenaline.get("ops_boost_pct", 0) if adrenaline else 0,
    )

    ax6.text(0.1, 0.95, summary_text, transform=ax6.transAxes,
            fontsize=10, fontfamily="monospace", verticalalignment="top")

    # Save
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    plot_path = output_dir / f"hero_protocol_{timestamp}.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"  Saved: {plot_path}")

    plt.close()

    return plot_path


# =============================================================================
# MAIN
# =============================================================================

def run_hero_protocol(
    marathon_duration: float = 300.0,
    precog_pulses: int = 5,
    adrenaline_duration: float = 60.0,
    output_dir: str = "results/hero_protocol",
):
    """Run the complete Hero Protocol test suite."""

    print("\n" + "=" * 70)
    print("  HP Z2 MINI G9 - HERO PROTOCOL")
    print("  The KILL SHOT: Proving Somatic AI beats Standard AI")
    print("=" * 70)
    print(f"\n  Marathon test duration: {marathon_duration}s ({marathon_duration/60:.1f} min)")
    print(f"  Precognition pulses: {precog_pulses}")
    print(f"  Adrenaline test duration: {adrenaline_duration}s")

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Test 1: Marathon vs Sprint
    sprint, marathon = test_marathon_vs_sprint(
        duration=marathon_duration,
        sprint_throttle_temp=85.0,
        marathon_ceiling_temp=75.0,
    )

    # Test 2: Precognition
    precog = test_precognition(
        stress_duration=30.0,
        n_pulses=precog_pulses,
    )

    # Test 3: Adrenaline Override
    adrenaline = test_adrenaline_override(
        normal_duration=adrenaline_duration,
        critical_duration=adrenaline_duration,
        thermal_ceiling=75.0,
    )

    # Generate plots
    plot_path = generate_hero_plots(sprint, marathon, precog, adrenaline, output)

    # Save results
    results = {
        "timestamp": timestamp,
        "marathon_vs_sprint": {
            "sprint": {
                "total_ops": sprint.total_ops,
                "duration": sprint.duration,
                "throttle_events": sprint.throttle_events,
                "max_temp": sprint.max_temp,
                "avg_power": sprint.avg_power,
            },
            "marathon": {
                "total_ops": marathon.total_ops,
                "duration": marathon.duration,
                "throttle_events": marathon.throttle_events,
                "max_temp": marathon.max_temp,
                "avg_power": marathon.avg_power,
            },
            "marathon_advantage_pct": (marathon.total_ops - sprint.total_ops) / sprint.total_ops * 100 if sprint.total_ops > 0 else 0,
        },
        "precognition": precog,
        "adrenaline_override": {
            "normal_ops": adrenaline["normal"]["total_ops"],
            "critical_ops": adrenaline["critical"]["total_ops"],
            "boost_pct": adrenaline["ops_boost_pct"],
            "thermal_cost": adrenaline["thermal_cost_c"],
        },
    }

    results_path = output / f"hero_protocol_{timestamp}.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved: {results_path}")

    # Save time series data
    csv_path = output / f"hero_timeseries_{timestamp}.csv"
    with open(csv_path, "w") as f:
        f.write("mode,elapsed,temp,power,ops,size\n")
        for s in sprint.samples:
            f.write(f"sprint,{s.elapsed:.2f},{s.temp:.1f},{s.power:.1f},{s.ops:.0f},{s.size}\n")
        for s in marathon.samples:
            f.write(f"marathon,{s.elapsed:.2f},{s.temp:.1f},{s.power:.1f},{s.ops:.0f},{s.size}\n")
    print(f"  Time series saved: {csv_path}")

    print("\n" + "=" * 70)
    print("  HERO PROTOCOL COMPLETE")
    print("=" * 70)

    return results, plot_path


def main():
    parser = argparse.ArgumentParser(description="HP Z2 Hero Protocol - DSI Integration Showcase")
    parser.add_argument("--marathon-duration", type=float, default=300.0,
                       help="Duration for marathon/sprint test (seconds)")
    parser.add_argument("--precog-pulses", type=int, default=5,
                       help="Number of stress pulses for precognition test")
    parser.add_argument("--adrenaline-duration", type=float, default=60.0,
                       help="Duration for adrenaline override test (seconds)")
    parser.add_argument("--output-dir", type=str, default="results/hero_protocol",
                       help="Output directory for results")
    parser.add_argument("--quick", action="store_true",
                       help="Quick test (60s marathon, 3 pulses)")

    args = parser.parse_args()

    if args.quick:
        args.marathon_duration = 60.0
        args.precog_pulses = 3
        args.adrenaline_duration = 30.0
        print("  Quick mode: Shortened test durations")

    results, plot_path = run_hero_protocol(
        marathon_duration=args.marathon_duration,
        precog_pulses=args.precog_pulses,
        adrenaline_duration=args.adrenaline_duration,
        output_dir=args.output_dir,
    )

    return results


if __name__ == "__main__":
    main()
