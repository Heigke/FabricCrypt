#!/usr/bin/env python3
"""
z36 Embodiment Comparison - Base vs Embodied Business Metrics
==============================================================

Compares:
1. Base model (vanilla Qwen without embodiment)
2. Embodied model (with skip gates, sensor feedback, FiLM modulation)

Metrics compared:
- Efficiency (tokens/Joule)
- Cost ($/1M tokens)
- Latency (TTFT, TPOT)
- Thermal (peak temp, headroom)
- Throughput (tokens/sec)
"""

import os
import sys
import json
import time
import random
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any
import threading

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.sensors.canonical_features import CanonicalSensorHub

# ============================================================================
# POWER SAMPLING
# ============================================================================

class RealTimePowerSampler:
    """Background power sampling during generation."""

    def __init__(self, power_path: str = "/sys/class/drm/card1/device/hwmon/hwmon7/power1_average",
                 temp_path: str = "/sys/class/drm/card1/device/hwmon/hwmon7/temp1_input",
                 sample_interval: float = 0.005):
        self.power_path = Path(power_path)
        self.temp_path = Path(temp_path)
        self.sample_interval = sample_interval
        self.samples = []
        self.running = False
        self._thread = None
        self._start_time = 0.0
        self._end_time = 0.0

    def _read_power_watts(self) -> float:
        try:
            if self.power_path.exists():
                return float(self.power_path.read_text().strip()) / 1e6
        except:
            pass
        return 0.0

    def _read_temp_c(self) -> float:
        try:
            if self.temp_path.exists():
                return float(self.temp_path.read_text().strip()) / 1000
        except:
            pass
        return 0.0

    def _sample_loop(self):
        while self.running:
            power = self._read_power_watts()
            temp = self._read_temp_c()
            self.samples.append((time.time(), power, temp))
            time.sleep(self.sample_interval)

    def start(self):
        self.samples = []
        self.running = True
        self._start_time = time.time()
        self._thread = threading.Thread(target=self._sample_loop, daemon=True)
        self._thread.start()

    def stop(self) -> dict:
        self.running = False
        self._end_time = time.time()
        if self._thread:
            self._thread.join(timeout=0.1)

        if not self.samples:
            return {"avg_power_w": 0, "peak_power_w": 0, "total_energy_j": 0,
                    "duration_s": 0, "avg_temp_c": 0, "peak_temp_c": 0}

        powers = [s[1] for s in self.samples]
        temps = [s[2] for s in self.samples]
        duration = self._end_time - self._start_time

        # Integrate power for energy
        total_energy = 0.0
        for i in range(1, len(self.samples)):
            dt = self.samples[i][0] - self.samples[i-1][0]
            avg_p = (self.samples[i][1] + self.samples[i-1][1]) / 2
            total_energy += avg_p * dt

        return {
            "avg_power_w": sum(powers) / len(powers),
            "peak_power_w": max(powers),
            "min_power_w": min(powers),
            "total_energy_j": total_energy,
            "duration_s": duration,
            "avg_temp_c": sum(temps) / len(temps),
            "peak_temp_c": max(temps),
            "samples": len(self.samples),
        }

# ============================================================================
# BUSINESS METRICS COLLECTOR
# ============================================================================

class MetricsCollector:
    """Collects business metrics during benchmark runs."""

    ELECTRICITY_COST_PER_KWH = 0.12  # USD
    CO2_PER_KWH = 0.4  # kg

    def __init__(self):
        self.reset()

    def reset(self):
        self.ttft_samples = []
        self.tpot_samples = []
        self.tokens_generated = 0
        self.total_joules = 0.0
        self.total_time_s = 0.0
        self.peak_temp_c = 0.0
        self.power_samples = []
        self.temp_samples = []

    def record(self, ttft_ms: float, total_time_ms: float, tokens: int,
               joules: float, peak_temp: float, avg_power: float):
        self.ttft_samples.append(ttft_ms)
        if tokens > 1:
            tpot = (total_time_ms - ttft_ms) / (tokens - 1)
            self.tpot_samples.append(tpot)
        self.tokens_generated += tokens
        self.total_joules += joules
        self.total_time_s += total_time_ms / 1000.0
        self.peak_temp_c = max(self.peak_temp_c, peak_temp)
        self.power_samples.append(avg_power)
        self.temp_samples.append(peak_temp)

    def report(self) -> Dict[str, Any]:
        tokens_per_joule = self.tokens_generated / max(0.001, self.total_joules)
        joules_per_token = self.total_joules / max(1, self.tokens_generated)
        tokens_per_second = self.tokens_generated / max(0.001, self.total_time_s)

        kwh = self.total_joules / 3_600_000
        cost_per_1m = (kwh * self.ELECTRICITY_COST_PER_KWH / max(1, self.tokens_generated)) * 1_000_000
        co2_per_1m = (kwh * self.CO2_PER_KWH / max(1, self.tokens_generated)) * 1_000_000 * 1000  # grams

        return {
            "tokens_per_joule": round(tokens_per_joule, 3),
            "joules_per_token": round(joules_per_token, 4),
            "tokens_per_second": round(tokens_per_second, 2),
            "usd_per_1m_tokens": round(cost_per_1m, 4),
            "co2_g_per_1m_tokens": round(co2_per_1m, 2),
            "ttft_p50_ms": round(np.percentile(self.ttft_samples, 50), 2) if self.ttft_samples else 0,
            "ttft_p95_ms": round(np.percentile(self.ttft_samples, 95), 2) if self.ttft_samples else 0,
            "tpot_p50_ms": round(np.percentile(self.tpot_samples, 50), 2) if self.tpot_samples else 0,
            "tpot_p95_ms": round(np.percentile(self.tpot_samples, 95), 2) if self.tpot_samples else 0,
            "peak_temp_c": round(self.peak_temp_c, 1),
            "avg_power_w": round(np.mean(self.power_samples), 1) if self.power_samples else 0,
            "thermal_headroom_c": round(100.0 - self.peak_temp_c, 1),
            "total_tokens": self.tokens_generated,
            "total_joules": round(self.total_joules, 2),
        }

# ============================================================================
# SKIP BLOCK WRAPPER (from z35)
# ============================================================================

class SkipBlockWrapper(torch.nn.Module):
    """
    Wrapper that adds skip/FiLM functionality to a transformer layer.
    Compatible with z34/z35 checkpoint format.
    """

    def __init__(self, original_layer, layer_idx: int, hidden_size: int, device, dtype):
        super().__init__()
        self._original_layer = original_layer
        self.layer_idx = layer_idx
        self.hidden_size = hidden_size

        # Skip gate (learnable)
        self.skip_gate = torch.nn.Linear(hidden_size, 1).to(device=device, dtype=dtype)

        # FiLM modulation
        self.film_scale = torch.nn.Linear(12, hidden_size).to(device=device, dtype=dtype)
        self.film_shift = torch.nn.Linear(12, hidden_size).to(device=device, dtype=dtype)

        # State tracking
        self.sensors = None
        self.force_skip = None
        self.gate_value = None
        self.disable_film = False

        # Metrics
        self.last_gate_value = None
        self.last_skipped = False
        self.last_film_effect = None

    def __getattr__(self, name: str):
        if name.startswith('_'):
            return super().__getattr__(name)
        try:
            return getattr(self._original_layer, name)
        except AttributeError:
            return super().__getattr__(name)

    def forward(self, hidden_states, **kwargs):
        # Determine skip
        if self.force_skip is not None:
            do_skip = self.force_skip
        elif self.gate_value is not None:
            do_skip = self.gate_value < 0.5
        else:
            gate_input = hidden_states.mean(dim=1)
            gate_logit = self.skip_gate(gate_input)
            self.last_gate_value = torch.sigmoid(gate_logit).mean().item()
            do_skip = self.last_gate_value < 0.5

        self.last_skipped = do_skip

        if do_skip:
            self.last_film_effect = 0.0
            return (hidden_states,) if not kwargs else hidden_states

        # Run original layer
        output = self._original_layer(hidden_states, **kwargs)
        if isinstance(output, tuple):
            hidden_out = output[0]
        else:
            hidden_out = output

        # Apply FiLM if sensors available
        if self.sensors is not None and not self.disable_film:
            sensors_t = torch.tensor(self.sensors, device=hidden_out.device, dtype=hidden_out.dtype)
            if sensors_t.dim() == 1:
                sensors_t = sensors_t.unsqueeze(0).expand(hidden_out.shape[0], -1)

            scale = self.film_scale(sensors_t).unsqueeze(1)
            shift = self.film_shift(sensors_t).unsqueeze(1)
            hidden_out = scale * hidden_out + shift
            self.last_film_effect = scale.abs().mean().item()
        else:
            self.last_film_effect = 0.0

        if isinstance(output, tuple):
            return (hidden_out,) + output[1:]
        return hidden_out

# ============================================================================
# EMBODIED MODEL LOADER
# ============================================================================

def load_embodied_model(checkpoint_path: str, base_model: str, device: str):
    """Load the embodied model with skip gates (z35 compatible)."""

    print("\n[EMBODIED] Loading checkpoint...")

    # Load checkpoint
    ckpt_path = Path(checkpoint_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"No checkpoint at {ckpt_path}")

    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    step = checkpoint.get("step", 0)
    print(f"  Checkpoint step: {step}")

    # Load base model
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )

    # Get model dtype and device
    first_param = next(model.parameters())
    dtype = first_param.dtype
    model_device = first_param.device

    hidden_size = model.config.hidden_size
    print(f"  Hidden size: {hidden_size}")

    # Gate layers from checkpoint
    skip_block_data = checkpoint.get("skip_blocks", {})
    gate_layers = [int(k) for k in skip_block_data.keys()]
    if not gate_layers:
        gate_layers = [7, 11, 15, 19, 23]
    print(f"  Gate layers: {gate_layers}")

    # Create skip blocks
    skip_blocks = {}
    layers = model.model.layers

    for layer_idx in gate_layers:
        if layer_idx < len(layers):
            original_layer = layers[layer_idx]
            wrapper = SkipBlockWrapper(
                original_layer=original_layer,
                layer_idx=layer_idx,
                hidden_size=hidden_size,
                device=model_device,
                dtype=dtype
            )

            # Load weights from checkpoint
            layer_key = str(layer_idx)
            if layer_key in skip_block_data:
                layer_state = skip_block_data[layer_key]
                try:
                    wrapper.load_state_dict(layer_state, strict=False)
                except Exception as e:
                    print(f"  Warning: Could not load weights for layer {layer_idx}: {e}")

            layers[layer_idx] = wrapper
            skip_blocks[layer_idx] = wrapper

    print(f"  Created {len(skip_blocks)} skip blocks")

    model.eval()
    return model, skip_blocks

# ============================================================================
# BENCHMARK RUNNER
# ============================================================================

def run_benchmark(model, tokenizer, power_sampler: RealTimePowerSampler,
                  num_trials: int = 30, max_new_tokens: int = 32,
                  label: str = "model") -> Dict:
    """Run benchmark and collect metrics."""

    print(f"\n[{label}] Running {num_trials} trials, {max_new_tokens} tokens each...")

    prompts = [
        "The future of artificial intelligence will",
        "In a world where technology advances",
        "Scientists have discovered that the key",
        "The most important thing about learning is",
        "When we consider the implications of",
        "Research has shown that human creativity",
        "The relationship between mind and body",
        "Looking at the data, we can conclude",
        "The evolution of computing has led to",
        "Understanding consciousness requires us to",
    ]

    device = next(model.parameters()).device
    metrics = MetricsCollector()

    # Warmup
    print(f"  Warmup...")
    enc = tokenizer("Hello world", return_tensors="pt")
    input_ids = enc.input_ids.to(device)
    attention_mask = torch.ones_like(input_ids)
    with torch.no_grad():
        _ = model.generate(input_ids, attention_mask=attention_mask, max_new_tokens=5,
                          do_sample=False, pad_token_id=tokenizer.pad_token_id)
    torch.cuda.synchronize()
    time.sleep(0.5)

    # Actual benchmark
    for i in range(num_trials):
        prompt = prompts[i % len(prompts)]
        enc = tokenizer(prompt, return_tensors="pt")
        input_ids = enc.input_ids.to(device)
        attention_mask = torch.ones_like(input_ids)

        # Start power sampling
        power_sampler.start()

        # Time to first token
        torch.cuda.synchronize()
        t0 = time.perf_counter()

        with torch.no_grad():
            outputs = model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                use_cache=True,
            )

        torch.cuda.synchronize()
        t1 = time.perf_counter()

        # Stop power sampling
        power_stats = power_sampler.stop()

        tokens_generated = outputs.shape[1] - input_ids.shape[1]
        total_time_ms = (t1 - t0) * 1000
        ttft_ms = total_time_ms / max(1, tokens_generated)  # Approximation

        metrics.record(
            ttft_ms=ttft_ms,
            total_time_ms=total_time_ms,
            tokens=tokens_generated,
            joules=power_stats["total_energy_j"],
            peak_temp=power_stats["peak_temp_c"],
            avg_power=power_stats["avg_power_w"],
        )

        if (i + 1) % 10 == 0:
            print(f"    Trial {i+1}/{num_trials}: {tokens_generated} tokens, "
                  f"{power_stats['avg_power_w']:.1f}W, {power_stats['peak_temp_c']:.1f}°C")

    return metrics.report()

# ============================================================================
# MAIN
# ============================================================================

def main():
    import argparse
    parser = argparse.ArgumentParser(description="z36 Embodiment Comparison")
    parser.add_argument("--base-model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--checkpoint", default="models/z34_fullloop/step_300.pt")
    parser.add_argument("--trials", type=int, default=30)
    parser.add_argument("--tokens", type=int, default=32)
    parser.add_argument("--output", default="results/z36_comparison.json")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Initialize power sampler
    power_sampler = RealTimePowerSampler()

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    results = {}

    # ========================================================================
    # TEST 1: BASE MODEL (no embodiment)
    # ========================================================================
    print("\n" + "="*70)
    print("BENCHMARK 1: BASE MODEL (vanilla, no embodiment)")
    print("="*70)

    print("\n[BASE] Loading model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.float16,
        device_map=device,
        trust_remote_code=True,
    )
    base_model.eval()

    results["base"] = run_benchmark(
        base_model, tokenizer, power_sampler,
        num_trials=args.trials,
        max_new_tokens=args.tokens,
        label="BASE"
    )

    # Free memory
    del base_model
    torch.cuda.empty_cache()
    time.sleep(1)

    # ========================================================================
    # TEST 2: EMBODIED MODEL (with skip gates)
    # ========================================================================
    print("\n" + "="*70)
    print("BENCHMARK 2: EMBODIED MODEL (skip gates + sensor feedback)")
    print("="*70)

    # Initialize sensor hub
    sensor_hub = CanonicalSensorHub()

    embodied_model, skip_blocks = load_embodied_model(
        args.checkpoint, args.base_model, device
    )

    # Inject real sensor values before each forward
    # Use inject_stress(0.5) for neutral state
    sensor_values = sensor_hub.inject_stress(0.5)
    for block in skip_blocks.values():
        block.sensors = sensor_values

    results["embodied"] = run_benchmark(
        embodied_model, tokenizer, power_sampler,
        num_trials=args.trials,
        max_new_tokens=args.tokens,
        label="EMBODIED"
    )

    # ========================================================================
    # COMPARISON
    # ========================================================================
    print("\n" + "="*70)
    print("COMPARISON: BASE vs EMBODIED")
    print("="*70)

    base = results["base"]
    emb = results["embodied"]

    # Calculate improvements
    def improvement(base_val, emb_val, higher_is_better=True):
        if base_val == 0:
            return 0
        pct = ((emb_val - base_val) / abs(base_val)) * 100
        return pct if higher_is_better else -pct

    comparison = {
        "efficiency": {
            "base_tokens_per_joule": base["tokens_per_joule"],
            "embodied_tokens_per_joule": emb["tokens_per_joule"],
            "improvement_pct": round(improvement(base["tokens_per_joule"], emb["tokens_per_joule"]), 1),
        },
        "cost": {
            "base_usd_per_1m": base["usd_per_1m_tokens"],
            "embodied_usd_per_1m": emb["usd_per_1m_tokens"],
            "savings_pct": round(improvement(base["usd_per_1m_tokens"], emb["usd_per_1m_tokens"], False), 1),
        },
        "throughput": {
            "base_tokens_per_sec": base["tokens_per_second"],
            "embodied_tokens_per_sec": emb["tokens_per_second"],
            "improvement_pct": round(improvement(base["tokens_per_second"], emb["tokens_per_second"]), 1),
        },
        "latency": {
            "base_tpot_p50_ms": base["tpot_p50_ms"],
            "embodied_tpot_p50_ms": emb["tpot_p50_ms"],
            "improvement_pct": round(improvement(base["tpot_p50_ms"], emb["tpot_p50_ms"], False), 1),
        },
        "power": {
            "base_avg_power_w": base["avg_power_w"],
            "embodied_avg_power_w": emb["avg_power_w"],
            "reduction_pct": round(improvement(base["avg_power_w"], emb["avg_power_w"], False), 1),
        },
        "thermal": {
            "base_peak_temp_c": base["peak_temp_c"],
            "embodied_peak_temp_c": emb["peak_temp_c"],
            "base_headroom_c": base["thermal_headroom_c"],
            "embodied_headroom_c": emb["thermal_headroom_c"],
        },
    }

    results["comparison"] = comparison
    results["timestamp"] = datetime.now().isoformat()

    # Print comparison table
    print("\n┌─────────────────────────────────────────────────────────────────────┐")
    print("│                    BUSINESS METRICS COMPARISON                      │")
    print("├─────────────────────┬─────────────┬─────────────┬───────────────────┤")
    print("│ Metric              │ Base Model  │ Embodied    │ Improvement       │")
    print("├─────────────────────┼─────────────┼─────────────┼───────────────────┤")
    print(f"│ Efficiency (tok/J)  │ {base['tokens_per_joule']:>11.3f} │ {emb['tokens_per_joule']:>11.3f} │ {comparison['efficiency']['improvement_pct']:>+14.1f}%  │")
    print(f"│ Cost ($/1M tok)     │ ${base['usd_per_1m_tokens']:>10.4f} │ ${emb['usd_per_1m_tokens']:>10.4f} │ {comparison['cost']['savings_pct']:>+14.1f}%  │")
    print(f"│ Throughput (tok/s)  │ {base['tokens_per_second']:>11.2f} │ {emb['tokens_per_second']:>11.2f} │ {comparison['throughput']['improvement_pct']:>+14.1f}%  │")
    print(f"│ TPOT p50 (ms)       │ {base['tpot_p50_ms']:>11.2f} │ {emb['tpot_p50_ms']:>11.2f} │ {comparison['latency']['improvement_pct']:>+14.1f}%  │")
    print(f"│ Avg Power (W)       │ {base['avg_power_w']:>11.1f} │ {emb['avg_power_w']:>11.1f} │ {comparison['power']['reduction_pct']:>+14.1f}%  │")
    print(f"│ Peak Temp (°C)      │ {base['peak_temp_c']:>11.1f} │ {emb['peak_temp_c']:>11.1f} │                   │")
    print(f"│ Thermal Headroom    │ {base['thermal_headroom_c']:>11.1f} │ {emb['thermal_headroom_c']:>11.1f} │                   │")
    print("└─────────────────────┴─────────────┴─────────────┴───────────────────┘")

    # CO2 comparison
    print(f"\n  CO2 Impact: {base['co2_g_per_1m_tokens']:.2f}g → {emb['co2_g_per_1m_tokens']:.2f}g per 1M tokens")

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_path}")

    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    eff_imp = comparison['efficiency']['improvement_pct']
    cost_sav = comparison['cost']['savings_pct']

    if eff_imp > 0:
        print(f"  ✅ Embodiment improves energy efficiency by {eff_imp:.1f}%")
    else:
        print(f"  ⚠️  Embodiment reduces energy efficiency by {-eff_imp:.1f}%")

    if cost_sav > 0:
        print(f"  ✅ Embodiment reduces cost by {cost_sav:.1f}%")
    else:
        print(f"  ⚠️  Embodiment increases cost by {-cost_sav:.1f}%")

    print("="*70)

if __name__ == "__main__":
    main()
