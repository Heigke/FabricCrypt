#!/usr/bin/env python3
"""
Energy vs Power Test for MLP Skip

Measures both instantaneous power AND total energy for different skip configs.
This determines the correct reward signal for z28.
"""

import os
import sys
import time
import torch
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from transformers import AutoTokenizer, AutoModelForCausalLM


def read_power_watts():
    """Read GPU power in watts."""
    sensor_path = "/sys/class/drm/card1/device/hwmon/hwmon7/power1_average"
    try:
        with open(sensor_path) as f:
            return int(f.read().strip()) / 1_000_000
    except:
        return 0.0


def measure_energy_and_time(model, input_ids, num_runs: int = 5, skip_layers: list = None):
    """
    Measure energy consumption for forward pass.
    Returns: (avg_time_ms, avg_power_W, avg_energy_mJ, std_time, std_power)
    """
    device = next(model.parameters()).device

    # Store original MLPs if skipping
    original_mlps = {}
    if skip_layers:
        for idx in skip_layers:
            if idx < len(model.model.layers):
                original_mlps[idx] = model.model.layers[idx].mlp.forward
                model.model.layers[idx].mlp.forward = lambda x: x

    # Warmup
    with torch.no_grad():
        for _ in range(3):
            _ = model(input_ids)
    torch.cuda.synchronize() if torch.cuda.is_available() else None

    times = []
    powers_during = []
    powers_after = []

    for _ in range(num_runs):
        # Read power before
        p_before = read_power_watts()

        # Time the forward pass
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        t0 = time.perf_counter()

        with torch.no_grad():
            _ = model(input_ids)

        torch.cuda.synchronize() if torch.cuda.is_available() else None
        t1 = time.perf_counter()

        # Read power after
        p_after = read_power_watts()

        times.append((t1 - t0) * 1000)  # ms
        powers_during.append((p_before + p_after) / 2)  # Approximate
        powers_after.append(p_after)

        time.sleep(0.05)  # Brief pause between measurements

    # Restore MLPs
    for idx, fn in original_mlps.items():
        model.model.layers[idx].mlp.forward = fn

    avg_time = np.mean(times)
    avg_power = np.mean(powers_during)
    avg_energy = avg_power * avg_time  # mW * ms = μJ, but we report mJ

    return {
        'time_ms': avg_time,
        'power_W': avg_power,
        'energy_mJ': avg_energy,
        'std_time': np.std(times),
        'std_power': np.std(powers_during),
    }


def run_generation_energy_test(model, tokenizer, prompt: str, max_tokens: int = 32, skip_layers: list = None):
    """
    Measure energy for actual token generation.
    """
    device = next(model.parameters()).device
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs.input_ids.to(device)

    # Store original MLPs if skipping
    original_mlps = {}
    if skip_layers:
        for idx in skip_layers:
            if idx < len(model.model.layers):
                original_mlps[idx] = model.model.layers[idx].mlp.forward
                model.model.layers[idx].mlp.forward = lambda x: x

    # Measure generation
    power_readings = []
    t_start = time.perf_counter()

    with torch.no_grad():
        past = None
        tokens_generated = 0

        for step in range(max_tokens):
            power_readings.append(read_power_watts())

            if past is None:
                outputs = model(input_ids, use_cache=True)
            else:
                last_token = input_ids[:, -1:]
                outputs = model(last_token, past_key_values=past, use_cache=True)

            logits = outputs.logits[:, -1, :]
            past = outputs.past_key_values

            next_token = logits.argmax(dim=-1, keepdim=True)
            tokens_generated += 1

            if next_token.item() == tokenizer.eos_token_id:
                break

            input_ids = torch.cat([input_ids, next_token], dim=-1)

    torch.cuda.synchronize() if torch.cuda.is_available() else None
    t_end = time.perf_counter()

    # Restore MLPs
    for idx, fn in original_mlps.items():
        model.model.layers[idx].mlp.forward = fn

    total_time = (t_end - t_start) * 1000  # ms
    avg_power = np.mean(power_readings)
    total_energy = avg_power * total_time  # mJ

    return {
        'tokens': tokens_generated,
        'total_time_ms': total_time,
        'avg_power_W': avg_power,
        'total_energy_mJ': total_energy,
        'time_per_token_ms': total_time / max(tokens_generated, 1),
        'energy_per_token_mJ': total_energy / max(tokens_generated, 1),
        'tokens_per_second': tokens_generated / (total_time / 1000),
    }


def main():
    print("="*70)
    print("z27 ENERGY vs POWER ANALYSIS")
    print("="*70)

    # Load model
    print("\nLoading model...")
    model_name = "Qwen/Qwen2.5-7B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()
    device = next(model.parameters()).device

    print(f"Model: {model_name}")
    print(f"Layers: {len(model.model.layers)}")

    # Test input
    prompt = "The capital of France is"
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs.input_ids.to(device)

    print(f"\nTest prompt: '{prompt}'")
    print(f"Input tokens: {input_ids.shape[1]}")

    # Test different skip configurations
    print("\n" + "="*70)
    print("SINGLE FORWARD PASS: Energy vs Power")
    print("="*70)

    configs = [
        ("Baseline (no skip)", []),
        ("z27 config (5)", [7, 11, 15, 19, 23]),
        ("Every 4th (7)", [0, 4, 8, 12, 16, 20, 24]),
        ("First 14", list(range(14))),
        ("All 28", list(range(28))),
    ]

    results = []
    for name, skip_layers in configs:
        r = measure_energy_and_time(model, input_ids, num_runs=10, skip_layers=skip_layers)
        results.append((name, len(skip_layers), r))

        print(f"\n{name}:")
        print(f"  Time: {r['time_ms']:.1f}ms ± {r['std_time']:.1f}ms")
        print(f"  Power: {r['power_W']:.1f}W ± {r['std_power']:.1f}W")
        print(f"  Energy: {r['energy_mJ']:.1f}mJ")

    # Calculate deltas
    baseline = results[0][2]
    print("\n" + "-"*50)
    print("COMPARISON TO BASELINE:")
    for name, skip_count, r in results[1:]:
        time_delta = (r['time_ms'] - baseline['time_ms']) / baseline['time_ms'] * 100
        power_delta = (r['power_W'] - baseline['power_W']) / baseline['power_W'] * 100
        energy_delta = (r['energy_mJ'] - baseline['energy_mJ']) / baseline['energy_mJ'] * 100

        print(f"\n{name} (skip {skip_count}):")
        print(f"  Time: {time_delta:+.1f}%")
        print(f"  Power: {power_delta:+.1f}%")
        print(f"  Energy: {energy_delta:+.1f}%")

    # Test generation
    print("\n" + "="*70)
    print("TOKEN GENERATION: Energy vs Power")
    print("="*70)

    gen_prompt = "Explain machine learning:"
    max_tokens = 32

    print(f"\nPrompt: '{gen_prompt}'")
    print(f"Max tokens: {max_tokens}")

    gen_results = []
    for name, skip_layers in configs[:3]:  # Only test first 3 for speed
        r = run_generation_energy_test(model, tokenizer, gen_prompt, max_tokens, skip_layers)
        gen_results.append((name, r))

        print(f"\n{name}:")
        print(f"  Tokens: {r['tokens']}")
        print(f"  Total time: {r['total_time_ms']:.0f}ms")
        print(f"  Avg power: {r['avg_power_W']:.1f}W")
        print(f"  Total energy: {r['total_energy_mJ']:.0f}mJ")
        print(f"  Time/token: {r['time_per_token_ms']:.1f}ms")
        print(f"  Energy/token: {r['energy_per_token_mJ']:.1f}mJ")
        print(f"  Tokens/sec: {r['tokens_per_second']:.1f}")

    # Analysis
    print("\n" + "="*70)
    print("KEY INSIGHTS")
    print("="*70)

    baseline_gen = gen_results[0][1]
    z27_gen = gen_results[1][1] if len(gen_results) > 1 else None

    if z27_gen:
        time_improvement = (baseline_gen['time_per_token_ms'] - z27_gen['time_per_token_ms']) / baseline_gen['time_per_token_ms'] * 100
        energy_improvement = (baseline_gen['energy_per_token_mJ'] - z27_gen['energy_per_token_mJ']) / baseline_gen['energy_per_token_mJ'] * 100
        throughput_improvement = (z27_gen['tokens_per_second'] - baseline_gen['tokens_per_second']) / baseline_gen['tokens_per_second'] * 100

        print(f"\nz27 config improvements:")
        print(f"  Time per token: {time_improvement:+.1f}%")
        print(f"  Energy per token: {energy_improvement:+.1f}%")
        print(f"  Throughput: {throughput_improvement:+.1f}%")

    print(f"\n" + "-"*50)
    print("RECOMMENDATIONS FOR z28:")
    print("-"*50)

    print("""
1. REWARD SIGNAL OPTIONS (best to worst):
   a) Throughput: reward = tokens_per_second
      - Directly aligned with what MLP skip improves
      - Easy to measure, stable

   b) Energy efficiency: reward = -energy_per_token
      - Captures true benefit of skip
      - Requires integrating power over time

   c) Completion time: reward = -time_to_complete
      - Simple, aligned with skip benefit
      - May encourage early stopping

   d) Instantaneous power: reward = -|power - target|
      - NOT RECOMMENDED: MLP skip may INCREASE power
      - GPU frequency scaling confounds the signal

2. GATE INITIALIZATION:
   - Current: gates ~0.53, threshold 0.55 → 87% skip
   - Fix: Bias final layer to output ~0.6 mean
   - Or: Start threshold at 0.45

3. REGULARIZATION:
   - Replace symmetric reg with skip rate target
   - loss = (skip_rate - 0.3)^2  # Target 30% skip
""")


if __name__ == "__main__":
    main()
