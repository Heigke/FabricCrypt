#!/usr/bin/env python3
"""
z27 Actuator Authority Test

Critical test: Does skipping MLPs actually change power?

This script tests:
1. Force ALL MLPs to RUN → measure power
2. Force ALL MLPs to SKIP → measure power
3. Compare delta - if < 5W, actuator has no authority

Also analyzes:
- Which component dominates power (attention vs MLP)
- How many layers we'd need to skip for meaningful control
"""

import os
import sys
import time
import torch
import numpy as np
from pathlib import Path

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch.nn.functional as F


class PowerMeter:
    """Simple power meter for AMD GPU."""

    def __init__(self):
        self.sensor_path = "/sys/class/drm/card1/device/hwmon/hwmon7/power1_average"
        if not os.path.exists(self.sensor_path):
            # Try to find it
            for hwmon in Path("/sys/class/drm/card1/device/hwmon").glob("hwmon*"):
                power_file = hwmon / "power1_average"
                if power_file.exists():
                    self.sensor_path = str(power_file)
                    break

    def read_watts(self) -> float:
        try:
            with open(self.sensor_path) as f:
                return int(f.read().strip()) / 1_000_000
        except:
            return 0.0

    def measure_for(self, duration_sec: float = 1.0, samples: int = 10) -> tuple:
        """Measure power over duration, return (mean, std)."""
        readings = []
        interval = duration_sec / samples
        for _ in range(samples):
            readings.append(self.read_watts())
            time.sleep(interval)
        return np.mean(readings), np.std(readings)


def run_generation_with_mode(
    model,
    tokenizer,
    prompt: str,
    max_tokens: int = 64,
    mode: str = "normal",  # "normal", "force_run", "force_skip"
    skip_layers: list = None,
) -> tuple:
    """
    Generate tokens with different MLP modes.
    Returns (output_text, avg_power, tokens_generated)
    """
    device = next(model.parameters()).device
    power_meter = PowerMeter()

    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs.input_ids.to(device)

    # Hook to intercept MLP layers
    original_mlps = {}

    if mode == "force_skip":
        # Replace MLP forward with identity
        for layer_idx in (skip_layers or []):
            layer = model.model.layers[layer_idx]
            original_mlps[layer_idx] = layer.mlp.forward
            layer.mlp.forward = lambda x, original=layer.mlp: x  # Identity

    power_readings = []

    with torch.no_grad():
        past = None
        generated = []

        for step in range(max_tokens):
            # Read power
            power_readings.append(power_meter.read_watts())

            if past is None:
                outputs = model(input_ids, use_cache=True)
            else:
                outputs = model(input_ids[:, -1:], past_key_values=past, use_cache=True)

            logits = outputs.logits[:, -1, :]
            past = outputs.past_key_values

            next_token = logits.argmax(dim=-1)
            generated.append(next_token.item())

            if next_token.item() == tokenizer.eos_token_id:
                break

            input_ids = torch.cat([input_ids, next_token.unsqueeze(0).unsqueeze(0)], dim=-1)

    # Restore original MLPs
    if mode == "force_skip":
        for layer_idx, original_fn in original_mlps.items():
            model.model.layers[layer_idx].mlp.forward = original_fn

    output_text = tokenizer.decode(generated, skip_special_tokens=True)
    avg_power = np.mean(power_readings) if power_readings else 0

    return output_text, avg_power, len(generated)


def test_attention_vs_mlp_power(model, tokenizer, device):
    """
    Test power contribution of attention vs MLP.
    """
    print("\n" + "="*70)
    print("TEST 1: Attention vs MLP Power Contribution")
    print("="*70)

    power_meter = PowerMeter()

    # Create test input
    prompt = "The capital of France is"
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs.input_ids.to(device)

    # Warmup
    with torch.no_grad():
        for _ in range(3):
            _ = model(input_ids)

    # Test 1a: Full forward pass
    print("\n[1a] Full forward pass (attention + MLP):")
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    powers = []
    times = []
    for _ in range(10):
        t0 = time.perf_counter()
        with torch.no_grad():
            _ = model(input_ids)
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        times.append(time.perf_counter() - t0)
        powers.append(power_meter.read_watts())

    full_power = np.mean(powers)
    full_time = np.mean(times)
    print(f"  Power: {full_power:.1f}W ± {np.std(powers):.1f}W")
    print(f"  Time: {full_time*1000:.1f}ms ± {np.std(times)*1000:.1f}ms")

    # Test 1b: Skip ALL MLPs (identity)
    print("\n[1b] Skip ALL MLPs (28 layers):")
    original_mlps = []
    for layer in model.model.layers:
        original_mlps.append(layer.mlp.forward)
        layer.mlp.forward = lambda x: x  # Identity

    torch.cuda.synchronize() if torch.cuda.is_available() else None
    powers = []
    times = []
    for _ in range(10):
        t0 = time.perf_counter()
        with torch.no_grad():
            _ = model(input_ids)
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        times.append(time.perf_counter() - t0)
        powers.append(power_meter.read_watts())

    skip_all_power = np.mean(powers)
    skip_all_time = np.mean(times)
    print(f"  Power: {skip_all_power:.1f}W ± {np.std(powers):.1f}W")
    print(f"  Time: {skip_all_time*1000:.1f}ms ± {np.std(times)*1000:.1f}ms")

    # Restore
    for i, layer in enumerate(model.model.layers):
        layer.mlp.forward = original_mlps[i]

    mlp_power_contribution = full_power - skip_all_power
    mlp_time_contribution = full_time - skip_all_time

    print(f"\n  MLP contribution: {mlp_power_contribution:.1f}W ({mlp_power_contribution/full_power*100:.1f}% of total)")
    print(f"  MLP time: {mlp_time_contribution*1000:.1f}ms ({mlp_time_contribution/full_time*100:.1f}% of total)")

    return {
        "full_power": full_power,
        "skip_all_mlp_power": skip_all_power,
        "mlp_power_contribution": mlp_power_contribution,
        "mlp_power_pct": mlp_power_contribution / full_power * 100,
    }


def test_partial_skip_power(model, tokenizer, device):
    """
    Test power when skipping different numbers of MLP layers.
    """
    print("\n" + "="*70)
    print("TEST 2: Partial MLP Skip Power Impact")
    print("="*70)

    power_meter = PowerMeter()

    prompt = "The capital of France is"
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs.input_ids.to(device)

    # Warmup
    with torch.no_grad():
        for _ in range(3):
            _ = model(input_ids)

    results = {}

    # Test different skip configurations
    skip_configs = [
        ("baseline", []),
        ("z27_layers_5", [7, 11, 15, 19, 23]),  # Current z27 config
        ("every_4th_7", [0, 4, 8, 12, 16, 20, 24]),
        ("first_half_14", list(range(14))),
        ("second_half_14", list(range(14, 28))),
        ("all_28", list(range(28))),
    ]

    for name, skip_layers in skip_configs:
        # Store and replace MLP forwards
        original_mlps = {}
        for layer_idx in skip_layers:
            if layer_idx < len(model.model.layers):
                original_mlps[layer_idx] = model.model.layers[layer_idx].mlp.forward
                model.model.layers[layer_idx].mlp.forward = lambda x: x

        torch.cuda.synchronize() if torch.cuda.is_available() else None
        powers = []
        times = []

        for _ in range(10):
            t0 = time.perf_counter()
            with torch.no_grad():
                _ = model(input_ids)
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            times.append(time.perf_counter() - t0)
            powers.append(power_meter.read_watts())

        # Restore
        for layer_idx, fn in original_mlps.items():
            model.model.layers[layer_idx].mlp.forward = fn

        avg_power = np.mean(powers)
        avg_time = np.mean(times)

        print(f"\n[{name}] Skip {len(skip_layers)} MLPs: layers {skip_layers[:5]}{'...' if len(skip_layers) > 5 else ''}")
        print(f"  Power: {avg_power:.1f}W ± {np.std(powers):.1f}W")
        print(f"  Time: {avg_time*1000:.1f}ms ± {np.std(times)*1000:.1f}ms")

        results[name] = {"power": avg_power, "time": avg_time, "skip_count": len(skip_layers)}

    # Analysis
    baseline = results["baseline"]["power"]
    z27_power = results["z27_layers_5"]["power"]
    all_skip = results["all_28"]["power"]

    print("\n" + "-"*50)
    print("ANALYSIS:")
    print(f"  Baseline power: {baseline:.1f}W")
    print(f"  z27 config (5 layers): {z27_power:.1f}W (delta: {baseline - z27_power:.1f}W)")
    print(f"  Skip ALL MLPs: {all_skip:.1f}W (delta: {baseline - all_skip:.1f}W)")
    print(f"  z27 has {(baseline - z27_power)/(baseline - all_skip)*100:.1f}% of max MLP control authority")

    return results


def test_generation_power(model, tokenizer, device):
    """
    Test power during actual generation with run vs skip.
    """
    print("\n" + "="*70)
    print("TEST 3: Generation Power with Force Run vs Force Skip")
    print("="*70)

    prompt = "Explain the concept of machine learning in simple terms:"

    # z27 skip layers
    skip_layers = [7, 11, 15, 19, 23]

    print(f"\nPrompt: '{prompt}'")
    print(f"Skip layers: {skip_layers}")

    # Test normal generation
    print("\n[3a] Normal generation (all MLPs run):")
    _, normal_power, tokens = run_generation_with_mode(
        model, tokenizer, prompt, max_tokens=32, mode="normal"
    )
    print(f"  Avg power: {normal_power:.1f}W")
    print(f"  Tokens: {tokens}")

    # Test with force skip on z27 layers
    print("\n[3b] Force skip on z27 layers:")
    _, skip_power, tokens = run_generation_with_mode(
        model, tokenizer, prompt, max_tokens=32, mode="force_skip", skip_layers=skip_layers
    )
    print(f"  Avg power: {skip_power:.1f}W")
    print(f"  Tokens: {tokens}")

    delta = normal_power - skip_power
    print(f"\n  Power delta: {delta:.1f}W")

    if abs(delta) < 3:
        print("  ⚠️  WARNING: Power delta < 3W - actuator has WEAK authority!")
        print("  The model cannot learn meaningful power regulation with this config.")
    elif abs(delta) < 10:
        print("  ⚡ Moderate authority - power delta is measurable but small")
    else:
        print("  ✓ Good authority - power delta is significant")

    return {"normal": normal_power, "skip": skip_power, "delta": delta}


def test_gate_initialization():
    """
    Test what gate values we get from default initialization.
    """
    print("\n" + "="*70)
    print("TEST 4: Gate Network Initialization Analysis")
    print("="*70)

    import torch.nn as nn

    # Recreate the gate network from z27
    hidden_size = 3584  # Qwen2.5-7B hidden size
    sensor_dim = 8

    gate_net = nn.Sequential(
        nn.Linear(hidden_size + sensor_dim, 256),
        nn.GELU(),
        nn.Dropout(0.1),
        nn.Linear(256, 64),
        nn.GELU(),
        nn.Linear(64, 1),
        nn.Sigmoid(),
    )

    # Test with random inputs
    batch = 16
    test_input = torch.randn(batch, hidden_size + sensor_dim)

    gate_net.eval()
    with torch.no_grad():
        gates = gate_net(test_input).squeeze(-1)

    print(f"\nWith default initialization:")
    print(f"  Gate mean: {gates.mean():.4f}")
    print(f"  Gate std: {gates.std():.4f}")
    print(f"  Gate min: {gates.min():.4f}")
    print(f"  Gate max: {gates.max():.4f}")

    # Test thresholds
    thresholds = [0.35, 0.45, 0.50, 0.55, 0.65]
    print(f"\n  Skip rates at different thresholds (gate < τ → skip):")
    for tau in thresholds:
        skip_rate = (gates < tau).float().mean().item()
        print(f"    τ={tau}: skip_rate={skip_rate*100:.1f}%")

    # What initialization would give us 50% skip at τ=0.55?
    print("\n  To get 50% skip at τ=0.55:")
    print("    Need gate mean ≈ 0.55, which requires biasing the final layer")

    # Test biased initialization
    print("\n  Testing biased initialization (final layer bias = 0.2):")
    gate_net[-2].bias.data.fill_(0.2)  # Before sigmoid

    with torch.no_grad():
        gates_biased = gate_net(test_input).squeeze(-1)

    print(f"  Gate mean: {gates_biased.mean():.4f}")
    print(f"  Gate std: {gates_biased.std():.4f}")

    for tau in thresholds:
        skip_rate = (gates_biased < tau).float().mean().item()
        print(f"    τ={tau}: skip_rate={skip_rate*100:.1f}%")

    return {
        "default_mean": gates.mean().item(),
        "default_std": gates.std().item(),
        "biased_mean": gates_biased.mean().item(),
    }


def main():
    print("="*70)
    print("z27 ACTUATOR AUTHORITY TEST")
    print("="*70)
    print("\nThis test determines if the current z27 skip mechanism")
    print("can actually control power, and if not, what changes are needed.")

    # Check GPU
    if torch.cuda.is_available():
        device = "cuda"
        print(f"\nUsing GPU: {torch.cuda.get_device_name()}")
    else:
        device = "cpu"
        print("\nWARNING: No GPU detected, using CPU")

    # Test gate initialization first (no model needed)
    gate_results = test_gate_initialization()

    # Load model
    print("\n" + "="*70)
    print("Loading model...")
    print("="*70)

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

    print(f"Model loaded: {model_name}")
    print(f"Layers: {len(model.model.layers)}")

    # Run tests
    attn_mlp_results = test_attention_vs_mlp_power(model, tokenizer, device)
    partial_results = test_partial_skip_power(model, tokenizer, device)
    gen_results = test_generation_power(model, tokenizer, device)

    # Summary
    print("\n" + "="*70)
    print("SUMMARY & RECOMMENDATIONS")
    print("="*70)

    z27_delta = partial_results["baseline"]["power"] - partial_results["z27_layers_5"]["power"]
    max_delta = partial_results["baseline"]["power"] - partial_results["all_28"]["power"]

    print(f"\n1. MLP Power Authority:")
    print(f"   - Skipping ALL 28 MLPs saves: {max_delta:.1f}W")
    print(f"   - z27's 5 layers save: {z27_delta:.1f}W ({z27_delta/max_delta*100:.1f}% of max)")

    print(f"\n2. Gate Initialization:")
    print(f"   - Default mean: {gate_results['default_mean']:.3f}")
    print(f"   - With τ=0.55, default init gives ~100% skip (useless)")
    print(f"   - Need to either: lower τ, or bias gates higher")

    print(f"\n3. Recommended Fixes:")

    if z27_delta < 3:
        print("   ⚠️  CRITICAL: z27's 5 MLP layers have WEAK control authority!")
        print("   Consider one of:")
        print("     a) Skip MORE layers (e.g., every 4th layer = 7 layers)")
        print("     b) Skip full blocks (attention + MLP) for stronger effect")
        print("     c) Use early-exit instead of per-layer skip")
    else:
        print(f"   ✓ z27's MLP skip has {z27_delta:.1f}W authority (sufficient)")

    print("\n   Fix threshold/initialization by one of:")
    print("     a) Start τ=0.45 (below default gate mean)")
    print("     b) Bias final gate layer to output ~0.6 mean")
    print("     c) Flip curriculum: start τ=0.35, increase to 0.55")

    print("\n   Fix regularization:")
    print("     Replace symmetric reg with target skip rate reg:")
    print("     loss = (mean_gate - 0.6)^2  # Target 40% skip initially")


if __name__ == "__main__":
    main()
