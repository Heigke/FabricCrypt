#!/usr/bin/env python3
"""
FEEL z21: Closed-Loop Homeostatic Inference
============================================
The final piece: Real-time sensor updates during generation.

The model feels the heat IT JUST CREATED and throttles accordingly.

Architecture:
    Token 1 -> Heat Rise -> Sensor Update -> Token 2 (throttled) -> ...

This creates TRUE homeostasis:
- Generation creates heat
- Heat is sensed in real-time
- Model throttles its own compute
- System reaches thermal equilibrium

Author: FEEL Research Team
Date: 2026-01-12
"""

import os
import sys
import torch
import torch.nn as nn
import time
import threading
from pathlib import Path
from datetime import datetime
from typing import Optional, Callable, Dict, List, Tuple
from dataclasses import dataclass
from queue import Queue

os.environ.setdefault("HSA_OVERRIDE_GFX_VERSION", "11.0.0")

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))


@dataclass
class SensorReading:
    """Real-time sensor reading."""
    timestamp: float
    temperature: float  # GPU temperature [0-1 normalized]
    power: float        # Power draw [0-1 normalized]
    memory: float       # Memory usage [0-1 normalized]
    entropy: float      # Output entropy (uncertainty)


class RealTimeSensorBridge:
    """
    Bridge between AMD hardware sensors and model inference.

    Runs in a background thread, continuously polling sensors
    and providing real-time readings to the model.
    """

    def __init__(
        self,
        poll_interval: float = 0.1,  # 100ms polling
        temp_max: float = 100.0,     # Max temp for normalization
        power_max: float = 250.0,    # Max power (watts)
        use_simulated: bool = False,  # Use simulated sensors for testing
    ):
        self.poll_interval = poll_interval
        self.temp_max = temp_max
        self.power_max = power_max
        self.use_simulated = use_simulated

        self._current_reading = SensorReading(
            timestamp=time.time(),
            temperature=0.5,
            power=0.5,
            memory=0.5,
            entropy=0.5,
        )

        self._running = False
        self._thread = None
        self._reading_queue = Queue(maxsize=10)

        # Try to load real sensor
        self._sensor = None
        if not use_simulated:
            try:
                from hardware.amd_sensor import RealTimeSensor
                self._sensor = RealTimeSensor()
                print("[SensorBridge] Connected to AMD ROCm SMI")
            except Exception as e:
                print(f"[SensorBridge] Cannot load AMD sensor: {e}")
                print("[SensorBridge] Falling back to simulated mode")
                self.use_simulated = True

    def start(self):
        """Start background sensor polling."""
        if self._running:
            return

        self._running = True
        self._thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._thread.start()
        print("[SensorBridge] Started background polling")

    def stop(self):
        """Stop background sensor polling."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=1.0)
        print("[SensorBridge] Stopped")

    def _poll_loop(self):
        """Background polling loop."""
        while self._running:
            try:
                reading = self._read_sensors()
                self._current_reading = reading

                # Add to queue (non-blocking)
                try:
                    self._reading_queue.put_nowait(reading)
                except:
                    pass  # Queue full, skip

            except Exception as e:
                print(f"[SensorBridge] Poll error: {e}")

            time.sleep(self.poll_interval)

    def _read_sensors(self) -> SensorReading:
        """Read current sensor values."""
        if self.use_simulated:
            return self._simulated_reading()

        try:
            raw = self._sensor.read()
            return SensorReading(
                timestamp=time.time(),
                temperature=min(raw.get("temp", 50) / self.temp_max, 1.0),
                power=min(raw.get("power", 100) / self.power_max, 1.0),
                memory=raw.get("memory_used", 0.5),
                entropy=0.5,  # Computed during generation
            )
        except:
            return self._simulated_reading()

    def _simulated_reading(self) -> SensorReading:
        """Generate simulated sensor reading (for testing)."""
        import random

        # Simulate thermal inertia
        prev = self._current_reading
        temp_drift = random.gauss(0, 0.02)
        new_temp = max(0, min(1, prev.temperature + temp_drift))

        return SensorReading(
            timestamp=time.time(),
            temperature=new_temp,
            power=0.3 + random.random() * 0.4,
            memory=0.4 + random.random() * 0.2,
            entropy=0.5,
        )

    def get_reading(self) -> SensorReading:
        """Get most recent sensor reading."""
        return self._current_reading

    def get_tensor(self, device: str = "cuda") -> torch.Tensor:
        """Get sensor reading as tensor [temp, power, entropy]."""
        r = self._current_reading
        return torch.tensor(
            [r.temperature, r.power, r.entropy],
            device=device,
            dtype=torch.float32,
        )


class ClosedLoopGenerator:
    """
    Closed-loop text generation with real-time thermal feedback.

    Key Innovation: Updates the FiLM adapter's sensor input
    DURING generation, creating true homeostatic behavior.

    The model feels the heat it creates and throttles accordingly.
    """

    def __init__(
        self,
        model,  # AnalogAwareLLM or MetabolicLLM
        sensor_bridge: RealTimeSensorBridge,
        update_interval: int = 10,  # Update sensor every N tokens
        thermal_momentum: float = 0.3,  # Smoothing factor
    ):
        self.model = model
        self.sensor = sensor_bridge
        self.update_interval = update_interval
        self.thermal_momentum = thermal_momentum

        # History for analysis
        self.generation_history = []

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 200,
        temperature: float = 0.7,
        callback: Optional[Callable[[int, SensorReading], None]] = None,
    ) -> Tuple[str, List[Dict]]:
        """
        Generate text with closed-loop thermal feedback.

        Args:
            prompt: Input prompt
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            callback: Optional callback(token_idx, sensor_reading)

        Returns:
            (generated_text, telemetry_log)
        """
        # Tokenize input
        input_ids = self.model.tokenizer(
            prompt,
            return_tensors="pt",
        ).input_ids.to("cuda")

        # Initialize generation state
        current_ids = input_ids
        telemetry = []
        smoothed_temp = self.sensor.get_reading().temperature

        # Start sensor polling
        self.sensor.start()

        try:
            for token_idx in range(max_new_tokens):
                # Update sensor state periodically
                if token_idx % self.update_interval == 0:
                    reading = self.sensor.get_reading()

                    # Apply momentum smoothing
                    smoothed_temp = (
                        self.thermal_momentum * smoothed_temp +
                        (1 - self.thermal_momentum) * reading.temperature
                    )

                    # Update model's sensor state
                    self.model.set_stress_level(smoothed_temp)

                    # Log telemetry
                    telemetry.append({
                        "token": token_idx,
                        "timestamp": reading.timestamp,
                        "raw_temp": reading.temperature,
                        "smoothed_temp": smoothed_temp,
                        "power": reading.power,
                        "stress_level": smoothed_temp,
                    })

                    if callback:
                        callback(token_idx, reading)

                # Generate next token
                with torch.no_grad():
                    outputs = self.model.model(
                        input_ids=current_ids,
                        use_cache=True,
                    )

                    logits = outputs.logits[:, -1, :] / temperature

                    # Sample
                    probs = torch.softmax(logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)

                    # Update entropy estimate
                    entropy = -(probs * torch.log(probs + 1e-10)).sum().item()
                    if telemetry:
                        telemetry[-1]["entropy"] = entropy

                    # Append to sequence
                    current_ids = torch.cat([current_ids, next_token], dim=-1)

                    # Check for EOS
                    if next_token.item() == self.model.tokenizer.eos_token_id:
                        break

        finally:
            self.sensor.stop()

        # Decode output
        generated = self.model.tokenizer.decode(
            current_ids[0],
            skip_special_tokens=True,
        )

        if prompt in generated:
            generated = generated[len(prompt):].strip()

        # Store in history
        self.generation_history.append({
            "prompt": prompt,
            "response": generated,
            "telemetry": telemetry,
            "timestamp": datetime.now().isoformat(),
        })

        return generated, telemetry

    def analyze_homeostasis(self, telemetry: List[Dict]) -> Dict:
        """Analyze homeostatic behavior from telemetry."""
        if not telemetry:
            return {}

        temps = [t["smoothed_temp"] for t in telemetry]
        powers = [t["power"] for t in telemetry]

        # Compute stability metrics
        temp_std = torch.tensor(temps).std().item()
        power_std = torch.tensor(powers).std().item()

        # Check for oscillation (sign changes in derivative)
        temp_deltas = [temps[i+1] - temps[i] for i in range(len(temps)-1)]
        oscillations = sum(1 for i in range(len(temp_deltas)-1)
                         if temp_deltas[i] * temp_deltas[i+1] < 0)

        # Check for convergence (decreasing variance in second half)
        mid = len(temps) // 2
        if mid > 0:
            first_half_std = torch.tensor(temps[:mid]).std().item()
            second_half_std = torch.tensor(temps[mid:]).std().item()
            converging = second_half_std < first_half_std
        else:
            converging = False

        return {
            "temp_mean": sum(temps) / len(temps),
            "temp_std": temp_std,
            "power_mean": sum(powers) / len(powers),
            "oscillations": oscillations,
            "converging": converging,
            "samples": len(temps),
        }


class HomeostaticInferenceEngine:
    """
    Complete inference engine with homeostatic regulation.

    Combines:
    - Real-time sensor polling
    - Closed-loop generation
    - Metabolic gating (layer skipping)
    - Thermal throttling
    """

    def __init__(
        self,
        model_path: str = "models/causal_z19/best",
        use_metabolic_gating: bool = False,
        use_real_sensors: bool = True,
    ):
        self.model_path = model_path
        self.use_metabolic_gating = use_metabolic_gating

        # Initialize sensor bridge
        self.sensor = RealTimeSensorBridge(
            poll_interval=0.1,
            use_simulated=not use_real_sensors,
        )

        # Load model
        self._load_model()

        # Create generator
        self.generator = ClosedLoopGenerator(
            model=self.model,
            sensor_bridge=self.sensor,
            update_interval=10,
        )

    def _load_model(self):
        """Load the trained model."""
        from z18_analog_trainer import AnalogAwareLLM

        print(f"[Engine] Loading model from {self.model_path}...")

        self.model = AnalogAwareLLM(
            base_model_id="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
            adapter_type="analog",
            device="cuda",
        )

        if Path(self.model_path).exists():
            self.model.load_adapters(self.model_path)
            print("[Engine] Loaded adapters")
        else:
            print(f"[Engine] Warning: {self.model_path} not found, using base model")

        self.model.eval()

        # Install metabolic gating if requested
        if self.use_metabolic_gating:
            from modeling.metabolic_gate import install_metabolic_gates

            gated_layers = [5, 11, 16, 22, 27]
            self.model.model, self.gates = install_metabolic_gates(
                self.model.model,
                layer_indices=gated_layers,
            )
            print(f"[Engine] Installed metabolic gates on layers {gated_layers}")

    def infer(
        self,
        prompt: str,
        max_tokens: int = 200,
        verbose: bool = True,
    ) -> str:
        """Run homeostatic inference."""
        if verbose:
            print(f"\n{'='*60}")
            print(f"HOMEOSTATIC INFERENCE")
            print(f"{'='*60}")
            print(f"Prompt: {prompt}")
            print(f"{'='*60}")

        def status_callback(token_idx, reading):
            if verbose and token_idx % 20 == 0:
                print(f"  [Token {token_idx:3d}] Temp: {reading.temperature:.2f} | "
                      f"Power: {reading.power:.2f}")

        response, telemetry = self.generator.generate(
            prompt=prompt,
            max_new_tokens=max_tokens,
            callback=status_callback,
        )

        if verbose:
            # Analyze homeostasis
            analysis = self.generator.analyze_homeostasis(telemetry)

            print(f"\n{'='*60}")
            print("HOMEOSTASIS ANALYSIS")
            print(f"{'='*60}")
            print(f"Temp Mean:    {analysis.get('temp_mean', 0):.3f}")
            print(f"Temp Std:     {analysis.get('temp_std', 0):.3f}")
            print(f"Oscillations: {analysis.get('oscillations', 0)}")
            print(f"Converging:   {'Yes' if analysis.get('converging') else 'No'}")
            print(f"\n{'='*60}")
            print(f"RESPONSE ({len(response)} chars)")
            print(f"{'='*60}")
            print(response[:500] + ("..." if len(response) > 500 else ""))

        return response

    def benchmark_flop_savings(
        self,
        prompts: List[str],
        stress_levels: List[float] = [0.2, 0.5, 0.8],
    ) -> Dict:
        """Benchmark FLOP savings across stress levels."""
        if not self.use_metabolic_gating:
            print("Warning: Metabolic gating not enabled, no FLOP savings")
            return {}

        results = {}

        for stress in stress_levels:
            print(f"\nTesting stress={stress}...")

            # Reset gate statistics
            for gate in self.gates.values():
                gate.reset_stats()

            # Run inference
            for prompt in prompts:
                self.model.set_stress_level(stress)
                _ = self.infer(prompt, verbose=False)

            # Collect skip rates
            skip_rates = []
            for layer_idx, gate in self.gates.items():
                rate = gate.get_skip_rate()
                skip_rates.append(rate)
                print(f"  Layer {layer_idx}: {rate*100:.1f}% skipped")

            results[stress] = {
                "avg_skip_rate": sum(skip_rates) / len(skip_rates),
                "per_layer": dict(zip(self.gates.keys(), skip_rates)),
            }

        return results


def demo_closed_loop():
    """Demonstrate closed-loop homeostatic inference."""

    print("=" * 70)
    print("FEEL z21: CLOSED-LOOP HOMEOSTATIC DEMO")
    print("=" * 70)

    # Create engine (use simulated sensors for demo)
    engine = HomeostaticInferenceEngine(
        model_path="models/analog_aware_z18/best",  # Start with z18
        use_metabolic_gating=False,  # Enable after z20 training
        use_real_sensors=True,  # Try real AMD sensors
    )

    # Test prompts
    prompts = [
        "What is 15 + 27?",
        "Explain the concept of photosynthesis briefly.",
        "What is the capital of France?",
    ]

    print("\n" + "=" * 70)
    print("RUNNING HOMEOSTATIC INFERENCE")
    print("=" * 70)

    for prompt in prompts:
        response = engine.infer(prompt, verbose=True)
        print("\n")

    # Show generation history
    print("=" * 70)
    print("GENERATION HISTORY")
    print("=" * 70)

    for i, gen in enumerate(engine.generator.generation_history):
        print(f"\n[{i+1}] {gen['prompt']}")
        analysis = engine.generator.analyze_homeostasis(gen['telemetry'])
        print(f"    Temp stability: {analysis.get('temp_std', 0):.3f}")
        print(f"    Response length: {len(gen['response'])} chars")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--demo", action="store_true", help="Run demo")
    parser.add_argument("--prompt", type=str, help="Single prompt to test")
    parser.add_argument("--model", type=str, default="models/analog_aware_z18/best")
    parser.add_argument("--metabolic", action="store_true", help="Enable metabolic gating")
    parser.add_argument("--simulated", action="store_true", help="Use simulated sensors")
    args = parser.parse_args()

    if args.demo:
        demo_closed_loop()
    elif args.prompt:
        engine = HomeostaticInferenceEngine(
            model_path=args.model,
            use_metabolic_gating=args.metabolic,
            use_real_sensors=not args.simulated,
        )
        engine.infer(args.prompt, verbose=True)
    else:
        demo_closed_loop()
