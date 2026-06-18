#!/usr/bin/env python3
"""
TRUE Early Exit - Actually Stops Computation at Exit Layer

The previous implementation had a critical flaw:
- It ran the FULL base model forward pass (all 12 layers)
- Then just picked which exit head to use
- This gave no real compute/energy savings

This implementation:
- Hooks into GPT-2's transformer blocks
- ACTUALLY stops forward propagation at the exit layer
- Measures REAL energy savings from reduced computation

This is the honest, scientific approach.
"""

import os
os.environ["HSA_OVERRIDE_GFX_VERSION"] = "11.0.0"

import sys
import json
import time
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from transformers import GPT2LMHeadModel, GPT2Config, AutoTokenizer
from datasets import load_dataset

from src.telemetry.real_amd import AMDTelemetry, EnergyMeter, RocmSmiReader


class TrueEarlyExitGPT2(nn.Module):
    """
    GPT-2 with TRUE early exit - actually stops computation.

    How it works:
    1. We manually iterate through transformer blocks
    2. At the exit layer, we stop and apply exit head
    3. Remaining layers are NOT computed
    """

    def __init__(self, model_name: str = "gpt2"):
        super().__init__()

        # Load pretrained model
        self.model = GPT2LMHeadModel.from_pretrained(model_name)

        # Extract components for manual forward
        self.wte = self.model.transformer.wte  # Token embedding
        self.wpe = self.model.transformer.wpe  # Position embedding
        self.drop = self.model.transformer.drop
        self.blocks = self.model.transformer.h  # Transformer blocks
        self.ln_f = self.model.transformer.ln_f  # Final layer norm
        self.lm_head = self.model.lm_head

        self.num_layers = len(self.blocks)
        self.hidden_dim = self.model.config.hidden_size
        self.vocab_size = self.model.config.vocab_size

        # Exit heads for intermediate layers
        self.exit_layers = [3, 6, 9, 12]
        self.exit_heads = nn.ModuleDict()

        for layer in self.exit_layers[:-1]:  # Don't need one for final layer
            self.exit_heads[str(layer)] = nn.Sequential(
                nn.LayerNorm(self.hidden_dim),
                nn.Linear(self.hidden_dim, self.vocab_size, bias=False)
            )
            # Initialize from lm_head
            with torch.no_grad():
                self.exit_heads[str(layer)][1].weight.copy_(self.lm_head.weight)

        # Freeze base model
        for p in self.model.parameters():
            p.requires_grad = False

        # Unfreeze exit heads
        for p in self.exit_heads.parameters():
            p.requires_grad = True

    def forward_to_layer(
        self,
        input_ids: torch.Tensor,
        exit_layer: int,
        attention_mask: Optional[torch.Tensor] = None
    ):
        """
        Forward pass that STOPS at the specified layer.

        This is the TRUE early exit - layers after exit_layer are NOT computed.
        """
        device = input_ids.device
        batch_size, seq_len = input_ids.shape

        # Get embeddings
        position_ids = torch.arange(seq_len, device=device).unsqueeze(0)
        hidden_states = self.wte(input_ids) + self.wpe(position_ids)
        hidden_states = self.drop(hidden_states)

        # Prepare attention mask if needed
        if attention_mask is not None:
            attention_mask = attention_mask.view(batch_size, -1)
            attention_mask = attention_mask[:, None, None, :]
            attention_mask = (1.0 - attention_mask) * torch.finfo(hidden_states.dtype).min

        # Forward through blocks UP TO exit_layer (NOT beyond)
        for i in range(min(exit_layer, self.num_layers)):
            block_output = self.blocks[i](
                hidden_states,
                attention_mask=attention_mask,
            )
            hidden_states = block_output[0]

        # Apply exit head or final layer norm + lm_head
        if exit_layer < self.num_layers:
            # Use exit head
            logits = self.exit_heads[str(exit_layer)](hidden_states)
        else:
            # Use final layer norm and lm_head
            hidden_states = self.ln_f(hidden_states)
            logits = self.lm_head(hidden_states)

        return logits, hidden_states

    def forward(self, input_ids, exit_layer: int = 12, labels=None):
        """Standard forward with optional loss computation."""
        logits, _ = self.forward_to_layer(input_ids, exit_layer)

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, self.vocab_size),
                shift_labels.view(-1),
                ignore_index=-100
            )

        return logits, loss


@dataclass
class BenchmarkResult:
    exit_layer: int
    layers_computed: int
    total_tokens: int
    total_energy_mj: float
    total_time_ms: float
    energy_per_token_mj: float
    tokens_per_second: float
    avg_power_w: float
    quality_loss: float


def run_benchmark(
    model: TrueEarlyExitGPT2,
    batches: List[torch.Tensor],
    exit_layer: int,
    telemetry: AMDTelemetry,
    device: torch.device,
    num_iterations: int = 20,
    warmup_iterations: int = 5
) -> BenchmarkResult:
    """Run benchmark with TRUE early exit."""

    # Warmup
    for _ in range(warmup_iterations):
        for batch in batches[:3]:
            _ = model(batch, exit_layer)
            torch.cuda.synchronize()

    # Count tokens
    total_tokens = sum(
        (batch != 50256).sum().item()  # GPT-2 pad token
        for batch in batches
    ) * num_iterations

    # Measure
    power_samples = []
    torch.cuda.synchronize()

    start_time = time.time()
    for _ in range(num_iterations):
        for batch in batches:
            _ = model(batch, exit_layer)
            power = RocmSmiReader.read_power()
            if power:
                power_samples.append(power)
    torch.cuda.synchronize()
    end_time = time.time()

    duration_ms = (end_time - start_time) * 1000

    # Compute average power and energy
    avg_power = np.mean(power_samples) if power_samples else 0
    total_energy_mj = avg_power * (duration_ms / 1000) * 1000  # W * s * 1000 = mJ

    # Compute quality loss
    labels = batches[0].clone()
    _, loss = model(batches[0], exit_layer, labels=labels)
    quality_loss = loss.item() if loss is not None else 0

    return BenchmarkResult(
        exit_layer=exit_layer,
        layers_computed=min(exit_layer, model.num_layers),
        total_tokens=total_tokens,
        total_energy_mj=total_energy_mj,
        total_time_ms=duration_ms,
        energy_per_token_mj=total_energy_mj / total_tokens if total_tokens > 0 else 0,
        tokens_per_second=total_tokens / (duration_ms / 1000) if duration_ms > 0 else 0,
        avg_power_w=avg_power,
        quality_loss=quality_loss
    )


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-batches", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--output", default="results/z203_true_early_exit.json")
    args = parser.parse_args()

    print("="*60)
    print("TRUE Early Exit - Actually Stops Computation")
    print("="*60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Initialize telemetry
    print("\nInitializing hardware telemetry...")
    telemetry = AMDTelemetry()
    initial = telemetry.read()
    print(f"  Power: {initial.power_w:.1f}W, Temp: {initial.temp_c:.0f}°C")

    # Load model
    print("\nLoading TRUE early exit model...")
    model = TrueEarlyExitGPT2("gpt2").to(device)
    model.eval()

    print(f"  Num layers: {model.num_layers}")
    print(f"  Exit layers: {model.exit_layers}")

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    # Prepare batches
    print(f"\nPreparing {args.num_batches} batches of size {args.batch_size}...")
    dataset = load_dataset("roneneldan/TinyStories", split="validation", streaming=True)

    texts = []
    for item in dataset:
        if len(texts) >= args.num_batches * args.batch_size:
            break
        text = item['text'][:500]
        if len(text) > 50:
            texts.append(text)

    batches = []
    for i in range(0, len(texts), args.batch_size):
        batch_texts = texts[i:i+args.batch_size]
        if len(batch_texts) == args.batch_size:
            encoded = tokenizer(
                batch_texts,
                max_length=128,
                truncation=True,
                padding='max_length',
                return_tensors='pt'
            )
            batches.append(encoded['input_ids'].to(device))

    print(f"  Prepared {len(batches)} batches")

    # Run benchmarks
    print(f"\nRunning TRUE early exit benchmarks ({args.iterations} iterations each)...")
    results = {}

    for exit_layer in [3, 6, 9, 12]:
        print(f"\n--- Exit Layer {exit_layer} (computes {exit_layer} layers) ---")

        result = run_benchmark(
            model, batches, exit_layer,
            telemetry, device, num_iterations=args.iterations
        )

        results[f"exit_{exit_layer}"] = result

        print(f"  Layers computed: {result.layers_computed}/{model.num_layers}")
        print(f"  Tokens: {result.total_tokens:,}")
        print(f"  Time: {result.total_time_ms:.0f} ms")
        print(f"  Energy: {result.total_energy_mj:.1f} mJ ({result.energy_per_token_mj:.3f} mJ/token)")
        print(f"  Throughput: {result.tokens_per_second:.0f} tok/s")
        print(f"  Avg Power: {result.avg_power_w:.1f} W")
        print(f"  Quality: {result.quality_loss:.3f}")

    # Analysis
    print("\n" + "="*60)
    print("ANALYSIS - TRUE EARLY EXIT SAVINGS")
    print("="*60)

    baseline = results["exit_12"]

    print(f"\nBaseline (Full 12 layers):")
    print(f"  {baseline.energy_per_token_mj:.3f} mJ/token")
    print(f"  {baseline.tokens_per_second:.0f} tok/s")
    print(f"  {baseline.avg_power_w:.1f} W")

    print(f"\nTRUE savings from early exit:")
    for exit_layer in [3, 6, 9]:
        r = results[f"exit_{exit_layer}"]
        compute_ratio = exit_layer / 12
        energy_savings = (1 - r.energy_per_token_mj / baseline.energy_per_token_mj) * 100 if baseline.energy_per_token_mj > 0 else 0
        throughput_gain = (r.tokens_per_second / baseline.tokens_per_second - 1) * 100 if baseline.tokens_per_second > 0 else 0
        quality_diff = (r.quality_loss - baseline.quality_loss) / baseline.quality_loss * 100 if baseline.quality_loss > 0 else 0

        print(f"\n  Exit Layer {exit_layer} ({compute_ratio:.0%} of layers):")
        print(f"    Compute ratio: {compute_ratio:.2f}x")
        print(f"    Energy savings: {energy_savings:+.1f}%")
        print(f"    Throughput gain: {throughput_gain:+.1f}%")
        print(f"    Quality change: {quality_diff:+.1f}%")

    # ECC weighted average
    ecc_weights = {3: 0.45, 6: 0.19, 9: 0.36}
    ecc_energy = sum(
        ecc_weights[l] * results[f"exit_{l}"].energy_per_token_mj
        for l in [3, 6, 9]
    )
    ecc_savings = (1 - ecc_energy / baseline.energy_per_token_mj) * 100 if baseline.energy_per_token_mj > 0 else 0

    print(f"\n--- ECC with trained distribution (45% L3, 19% L6, 36% L9) ---")
    print(f"  Expected energy/token: {ecc_energy:.3f} mJ")
    print(f"  Expected savings: {ecc_savings:.1f}%")

    # Quality-adjusted savings
    avg_quality = sum(
        ecc_weights[l] * results[f"exit_{l}"].quality_loss
        for l in [3, 6, 9]
    )
    quality_loss_pct = (avg_quality - baseline.quality_loss) / baseline.quality_loss * 100

    print(f"  Quality change: {quality_loss_pct:+.1f}%")

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    output_data = {
        'config': {
            'batch_size': args.batch_size,
            'num_batches': args.num_batches,
            'iterations': args.iterations,
            'model': 'gpt2',
            'true_early_exit': True
        },
        'results': {k: asdict(v) for k, v in results.items()},
        'analysis': {
            'baseline_energy_mj_per_token': baseline.energy_per_token_mj,
            'baseline_throughput': baseline.tokens_per_second,
            'ecc_weighted_energy_mj_per_token': ecc_energy,
            'ecc_expected_savings_pct': ecc_savings,
            'ecc_quality_change_pct': quality_loss_pct
        }
    }

    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"\nResults saved to {output_path}")

    # Cleanup
    telemetry.shutdown()

    print("\n" + "="*60)
    print("TRUE Early Exit validation complete!")
    print("="*60)


if __name__ == "__main__":
    main()
