#!/usr/bin/env python3
"""
FEEL z21: Real Embodiment - The Living Loop
============================================
The final piece: Real-time sensor feedback during generation.

The model FEELS the heat IT CREATES and throttles accordingly.

This is TRUE homeostasis:
- Token generation creates heat
- Heat is sensed in real-time from AMD GPU
- Model closes gates, skipping layers
- Heat drops
- Equilibrium achieved

Author: FEEL Research Team
Date: 2026-01-12
"""

import os
import sys
import torch
import time
import threading
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict, Tuple
from dataclasses import dataclass
from queue import Queue

os.environ.setdefault("HSA_OVERRIDE_GFX_VERSION", "11.0.0")

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))


@dataclass
class BodyState:
    """Real-time body (GPU) state."""
    timestamp: float
    temperature: float   # Normalized [0, 1]
    power: float         # Normalized [0, 1]
    memory: float        # Normalized [0, 1]
    raw_temp_c: float    # Raw temperature in Celsius
    raw_power_w: float   # Raw power in Watts


class RealTimeSensor:
    """
    Real-time AMD GPU sensor using ROCm SMI.

    Polls GPU metrics and provides normalized values for the model.
    """

    def __init__(
        self,
        poll_interval: float = 0.1,
        temp_max: float = 100.0,
        power_max: float = 250.0,
        use_simulated: bool = False,
    ):
        self.poll_interval = poll_interval
        self.temp_max = temp_max
        self.power_max = power_max
        self.use_simulated = use_simulated

        self._current_state = BodyState(
            timestamp=time.time(),
            temperature=0.5,
            power=0.5,
            memory=0.5,
            raw_temp_c=50.0,
            raw_power_w=100.0,
        )

        self._running = False
        self._thread = None

        # Try to load ROCm SMI
        self._smi = None
        if not use_simulated:
            try:
                import ctypes
                self._smi = ctypes.CDLL("/opt/rocm/lib/librocm_smi64.so")
                self._smi.rsmi_init(0)
                print("[RealTimeSensor] Connected to ROCm SMI")
            except Exception as e:
                print(f"[RealTimeSensor] Cannot load ROCm SMI: {e}")
                print("[RealTimeSensor] Falling back to simulated mode")
                self.use_simulated = True

    def start(self):
        """Start background polling."""
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._thread.start()

    def stop(self):
        """Stop background polling."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=1.0)

    def _poll_loop(self):
        """Background polling loop."""
        while self._running:
            try:
                self._current_state = self._read_sensors()
            except Exception as e:
                print(f"[Sensor] Error: {e}")
            time.sleep(self.poll_interval)

    def _read_sensors(self) -> BodyState:
        """Read current sensor values."""
        if self.use_simulated:
            return self._simulated_reading()

        try:
            import ctypes

            # Read temperature
            temp = ctypes.c_int64()
            self._smi.rsmi_dev_temp_metric_get(
                0, 0, 0,  # device 0, sensor 0, metric 0 (current)
                ctypes.byref(temp)
            )
            temp_c = temp.value / 1000.0  # Convert from millidegrees

            # Read power
            power = ctypes.c_uint64()
            self._smi.rsmi_dev_power_ave_get(0, 0, ctypes.byref(power))
            power_w = power.value / 1000000.0  # Convert from microwatts

            # Read memory
            mem_used = ctypes.c_uint64()
            mem_total = ctypes.c_uint64()
            self._smi.rsmi_dev_memory_usage_get(0, 0, ctypes.byref(mem_used))
            self._smi.rsmi_dev_memory_total_get(0, 0, ctypes.byref(mem_total))
            mem_ratio = mem_used.value / max(mem_total.value, 1)

            return BodyState(
                timestamp=time.time(),
                temperature=min(temp_c / self.temp_max, 1.0),
                power=min(power_w / self.power_max, 1.0),
                memory=mem_ratio,
                raw_temp_c=temp_c,
                raw_power_w=power_w,
            )

        except Exception:
            return self._simulated_reading()

    def _simulated_reading(self) -> BodyState:
        """Generate simulated sensor reading."""
        import random

        # Simulate thermal inertia
        prev = self._current_state
        temp_drift = random.gauss(0, 0.01)
        new_temp = max(0.3, min(0.9, prev.temperature + temp_drift))

        return BodyState(
            timestamp=time.time(),
            temperature=new_temp,
            power=0.4 + random.random() * 0.3,
            memory=0.5 + random.random() * 0.2,
            raw_temp_c=new_temp * 100,
            raw_power_w=new_temp * 200,
        )

    def get_state(self) -> BodyState:
        """Get current body state."""
        return self._current_state

    def get_tensor(self, device: str = "cuda") -> torch.Tensor:
        """Get sensor state as tensor [temp, power, entropy]."""
        s = self._current_state
        return torch.tensor(
            [s.temperature, s.power, 0.5],  # entropy = 0.5 (neutral)
            device=device,
            dtype=torch.float32,
        )


def homeostatic_generation(
    model,
    tokenizer,
    prompt: str,
    sensor: RealTimeSensor,
    max_tokens: int = 200,
    update_interval: int = 5,  # Update sensor every N tokens
    verbose: bool = True,
) -> Tuple[str, List[Dict]]:
    """
    Generate text with real-time homeostatic regulation.

    The model feels the heat it creates and throttles accordingly.
    """
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to("cuda")
    generated = input_ids

    telemetry = []
    sensor.start()

    if verbose:
        print(f"\n{'='*70}")
        print("HOMEOSTATIC GENERATION")
        print(f"{'='*70}")
        print(f"Prompt: {prompt}")
        print(f"{'='*70}")
        print(f"{'Token':<8} {'Temp°C':<10} {'Gates':<12} {'Text':<40}")
        print("-" * 70)

    try:
        for token_idx in range(max_tokens):
            # Update sensor state periodically
            if token_idx % update_interval == 0:
                body = sensor.get_state()
                model.set_sensor_state(sensor.get_tensor())

                telemetry.append({
                    "token": token_idx,
                    "timestamp": body.timestamp,
                    "temp_c": body.raw_temp_c,
                    "temp_norm": body.temperature,
                    "power_w": body.raw_power_w,
                    "gates_open": getattr(model, 'last_gates_open', 7),
                })

            # Generate next token
            with torch.no_grad():
                outputs = model(input_ids=generated)
                logits = outputs.logits[:, -1, :]

                # Sample
                probs = torch.softmax(logits / 0.7, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

            generated = torch.cat([generated, next_token], dim=1)

            # Check EOS
            if next_token.item() == tokenizer.eos_token_id:
                break

            # Verbose output
            if verbose and token_idx % update_interval == 0:
                body = sensor.get_state()
                gates = getattr(model, 'last_gates_open', 7)
                token_text = tokenizer.decode(next_token[0])
                print(f"{token_idx:<8} {body.raw_temp_c:<10.1f} {gates}/7         {token_text!r:<40}")

    finally:
        sensor.stop()

    # Decode
    output_text = tokenizer.decode(generated[0], skip_special_tokens=True)
    if prompt in output_text:
        output_text = output_text[len(prompt):].strip()

    if verbose:
        print("-" * 70)
        print(f"\nGenerated {len(generated[0]) - len(input_ids[0])} tokens")
        print(f"{'='*70}")

    return output_text, telemetry


def analyze_homeostasis(telemetry: List[Dict]) -> Dict:
    """Analyze homeostatic behavior from telemetry."""
    if not telemetry:
        return {}

    temps = [t["temp_c"] for t in telemetry]
    gates = [t["gates_open"] for t in telemetry]

    # Correlation: temp vs gates (should be negative)
    if len(temps) > 2:
        import numpy as np
        correlation = np.corrcoef(temps, gates)[0, 1]
    else:
        correlation = 0.0

    return {
        "temp_mean": sum(temps) / len(temps),
        "temp_min": min(temps),
        "temp_max": max(temps),
        "gates_mean": sum(gates) / len(gates),
        "temp_gate_correlation": correlation,
        "samples": len(telemetry),
    }


def demo_real_embodiment():
    """Demonstrate real embodiment with live sensors."""
    from modeling.metabolic_gate import load_metabolic_model

    print("=" * 70)
    print("FEEL z21: REAL EMBODIMENT DEMO")
    print("=" * 70)

    # Load model
    print("\n[1/3] Loading MetabolicDeepSeek...")
    model = load_metabolic_model(
        gated_layers=[3, 7, 11, 15, 19, 23, 27],
    )

    # Try to load trained gates
    gate_path = Path("models/metabolic_z20/best/gates.pt")
    if gate_path.exists():
        print(f"[2/3] Loading trained gates from {gate_path}...")
        checkpoint = torch.load(gate_path, weights_only=False)
        model.metabolic_blocks.load_state_dict(checkpoint["gates"])
    else:
        print("[2/3] No trained gates found, using random initialization")

    model.base_model.eval()

    # Create sensor
    print("[3/3] Initializing sensor...")
    sensor = RealTimeSensor(
        poll_interval=0.1,
        use_simulated=False,  # Try real sensors first
    )

    # Test prompts
    prompts = [
        "What is 15 + 27?",
        "Explain why the sky is blue in one sentence.",
        "What is the capital of France?",
    ]

    all_telemetry = []

    for prompt in prompts:
        print(f"\n{'='*70}")
        response, telemetry = homeostatic_generation(
            model=model,
            tokenizer=model.tokenizer,
            prompt=prompt,
            sensor=sensor,
            max_tokens=100,
            update_interval=5,
            verbose=True,
        )

        all_telemetry.extend(telemetry)

        print(f"\nRESPONSE: {response[:200]}...")

        # Analyze
        analysis = analyze_homeostasis(telemetry)
        print(f"\nHOMEOSTASIS ANALYSIS:")
        print(f"  Temp Range: {analysis.get('temp_min', 0):.1f}°C - {analysis.get('temp_max', 0):.1f}°C")
        print(f"  Avg Gates Open: {analysis.get('gates_mean', 7):.1f}/7")
        print(f"  Temp-Gate Correlation: {analysis.get('temp_gate_correlation', 0):.3f}")

    # Overall summary
    print("\n" + "=" * 70)
    print("OVERALL HOMEOSTASIS SUMMARY")
    print("=" * 70)

    overall = analyze_homeostasis(all_telemetry)
    print(f"Total Samples: {overall.get('samples', 0)}")
    print(f"Temp Range: {overall.get('temp_min', 0):.1f}°C - {overall.get('temp_max', 0):.1f}°C")
    print(f"Temp-Gate Correlation: {overall.get('temp_gate_correlation', 0):.3f}")

    if overall.get('temp_gate_correlation', 0) < -0.2:
        print("\n✅ HOMEOSTASIS DETECTED! Gates respond to temperature.")
    else:
        print("\n⚠️ Weak homeostasis - may need more gate training.")


def stress_test_embodiment(duration_seconds: int = 60):
    """
    Run a stress test to observe homeostatic behavior over time.

    Generates continuously while monitoring temperature and gate behavior.
    """
    from modeling.metabolic_gate import load_metabolic_model

    print("=" * 70)
    print(f"FEEL z21: STRESS TEST ({duration_seconds}s)")
    print("=" * 70)

    # Load model
    model = load_metabolic_model(gated_layers=[3, 7, 11, 15, 19, 23, 27])

    gate_path = Path("models/metabolic_z20/best/gates.pt")
    if gate_path.exists():
        checkpoint = torch.load(gate_path, weights_only=False)
        model.metabolic_blocks.load_state_dict(checkpoint["gates"])

    model.base_model.eval()

    sensor = RealTimeSensor(use_simulated=False)
    sensor.start()

    prompts = [
        "Count from 1 to 100.",
        "List the first 20 prime numbers.",
        "Recite the alphabet backwards.",
        "Name all the planets in order.",
    ]

    telemetry = []
    start_time = time.time()
    prompt_idx = 0

    print(f"\n{'Time':<8} {'Temp°C':<10} {'Gates':<8} {'Status':<20}")
    print("-" * 50)

    try:
        while time.time() - start_time < duration_seconds:
            prompt = prompts[prompt_idx % len(prompts)]
            prompt_idx += 1

            # Update sensor
            body = sensor.get_state()
            model.set_sensor_state(sensor.get_tensor())

            # Generate
            input_ids = model.tokenizer(prompt, return_tensors="pt").input_ids.to("cuda")

            with torch.no_grad():
                outputs = model.base_model.generate(
                    input_ids,
                    max_new_tokens=50,
                    do_sample=True,
                    temperature=0.7,
                    pad_token_id=model.tokenizer.eos_token_id,
                )

            gates = getattr(model, 'last_gates_open', 7)

            elapsed = time.time() - start_time
            telemetry.append({
                "time": elapsed,
                "temp_c": body.raw_temp_c,
                "gates": gates,
            })

            # Status
            if body.raw_temp_c > 80:
                status = "🔥 HOT"
            elif body.raw_temp_c > 70:
                status = "⚠️ WARM"
            else:
                status = "✅ COOL"

            print(f"{elapsed:<8.1f} {body.raw_temp_c:<10.1f} {gates}/7      {status}")

            time.sleep(0.5)

    except KeyboardInterrupt:
        print("\n[Interrupted]")

    finally:
        sensor.stop()

    # Summary
    print("\n" + "=" * 70)
    print("STRESS TEST RESULTS")
    print("=" * 70)

    if telemetry:
        temps = [t["temp_c"] for t in telemetry]
        gates = [t["gates"] for t in telemetry]

        print(f"Duration: {telemetry[-1]['time']:.1f}s")
        print(f"Temp Start: {temps[0]:.1f}°C")
        print(f"Temp End: {temps[-1]:.1f}°C")
        print(f"Temp Max: {max(temps):.1f}°C")
        print(f"Gates Start: {gates[0]}/7")
        print(f"Gates End: {gates[-1]}/7")

        import numpy as np
        correlation = np.corrcoef(temps, gates)[0, 1] if len(temps) > 2 else 0
        print(f"Temp-Gate Correlation: {correlation:.3f}")

        if correlation < -0.3:
            print("\n🎉 STRONG HOMEOSTASIS! The model self-regulates!")
        elif correlation < -0.1:
            print("\n✅ MODERATE HOMEOSTASIS detected.")
        else:
            print("\n⚠️ WEAK HOMEOSTASIS - gates may not be responding to heat.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["demo", "stress"], default="demo")
    parser.add_argument("--duration", type=int, default=60, help="Stress test duration (seconds)")
    parser.add_argument("--prompt", type=str, help="Single prompt to test")
    args = parser.parse_args()

    if args.mode == "demo":
        if args.prompt:
            # Single prompt test
            from modeling.metabolic_gate import load_metabolic_model
            model = load_metabolic_model(gated_layers=[3, 7, 11, 15, 19, 23, 27])
            model.base_model.eval()
            sensor = RealTimeSensor(use_simulated=False)

            response, telemetry = homeostatic_generation(
                model=model,
                tokenizer=model.tokenizer,
                prompt=args.prompt,
                sensor=sensor,
                verbose=True,
            )
            print(f"\nFULL RESPONSE:\n{response}")
        else:
            demo_real_embodiment()
    else:
        stress_test_embodiment(args.duration)
