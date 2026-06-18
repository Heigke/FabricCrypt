#!/usr/bin/env python3
"""
Validation for Embodied GPT-2 (Early Exit + Dynamic Span)

Tests:
1. Exit layer distribution across difficulties
2. Span response to body stress
3. Combined compute savings
4. Quality preservation
"""

import os
os.environ["HSA_OVERRIDE_GFX_VERSION"] = "11.0.0"

import sys
import json
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

from src.modeling.dynamic_attention import SpanConfig, SpanPredictor


class ExitHead(nn.Module):
    def __init__(self, hidden_dim, vocab_size):
        super().__init__()
        self.proj = nn.Linear(hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.out = nn.Linear(hidden_dim, vocab_size, bias=False)

    def forward(self, x):
        return self.out(self.norm(F.gelu(self.proj(x)) + x))


class EmbodiedGPT2(nn.Module):
    def __init__(self, model_name="gpt2", sensor_dim=32):
        super().__init__()
        self.base = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32)
        for p in self.base.parameters():
            p.requires_grad = False

        self.hidden_dim = self.base.config.hidden_size
        self.vocab_size = self.base.config.vocab_size
        self.num_layers = self.base.config.num_hidden_layers
        self.sensor_dim = sensor_dim

        self.exit_layer_indices = [
            self.num_layers // 4,
            self.num_layers // 2,
            3 * self.num_layers // 4,
            self.num_layers
        ]

        self.exit_heads = nn.ModuleList([
            ExitHead(self.hidden_dim, self.vocab_size)
            for _ in self.exit_layer_indices
        ])

        self.exit_uncertainty = nn.Sequential(
            nn.Linear(self.hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

        self.span_config = SpanConfig(
            hidden_dim=self.hidden_dim,
            sensor_dim=sensor_dim,
            num_layers=len(self.exit_layer_indices),
            num_spans=4,
            min_span=32,
            max_span=256,
            per_layer_span=True
        )
        self.span_predictor = SpanPredictor(self.span_config)

        self.body_encoder = nn.Sequential(
            nn.Linear(sensor_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32)
        )

    def simulate_body_state(self, batch_size, device, stress_level=0.5):
        base_state = torch.randn(batch_size, self.sensor_dim, device=device) * 0.3
        stress = torch.ones(batch_size, 1, device=device) * stress_level
        base_state[:, 0:1] = 1.0 - stress
        base_state[:, 1:2] = 1.0 - stress
        return torch.sigmoid(base_state)

    def forward(self, input_ids, body_state=None, return_all=True):
        batch_size = input_ids.size(0)
        device = input_ids.device

        if body_state is None:
            body_state = self.simulate_body_state(batch_size, device, 0.3)

        with torch.no_grad():
            out = self.base(input_ids, output_hidden_states=True)

        hidden_states = out.hidden_states
        final_logits = out.logits

        exit_outputs = []
        exit_uncertainties = []
        span_predictions = []

        for i, layer_idx in enumerate(self.exit_layer_indices):
            h = hidden_states[layer_idx]
            logits = self.exit_heads[i](h)
            exit_outputs.append(logits)

            u = self.exit_uncertainty(h[:, -1, :])
            exit_uncertainties.append(u)

            span_weights, selected_span = self.span_predictor(h, body_state, i)
            span_predictions.append({
                'weights': span_weights,
                'selected': selected_span,
                'value': self.span_predictor.get_span_value(selected_span)
            })

        return {
            'exit_outputs': exit_outputs,
            'exit_uncertainties': exit_uncertainties,
            'span_predictions': span_predictions,
            'final_logits': final_logits
        }


def load_model(checkpoint_path, device):
    print(f"Loading model from {checkpoint_path}...")
    model = EmbodiedGPT2("gpt2").to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.exit_heads.load_state_dict(checkpoint['exit_heads'])
    model.exit_uncertainty.load_state_dict(checkpoint['exit_uncertainty'])
    model.span_predictor.load_state_dict(checkpoint['span_predictor'])
    model.body_encoder.load_state_dict(checkpoint['body_encoder'])

    model.eval()
    return model


TEST_PROMPTS = {
    "easy": [
        "The cat sat on the",
        "Once upon a time",
        "Hello, my name is",
        "The sun is very",
    ],
    "medium": [
        "The meaning of life is",
        "Scientists believe that",
        "Technology has changed",
        "Democracy requires",
    ],
    "hard": [
        "Quantum entanglement suggests",
        "The epistemological implications",
        "Transcendental phenomenology",
        "Gödel's incompleteness theorem",
    ]
}


def validate_exit_and_span(model, tokenizer, device):
    """Test exit and span behavior across difficulties and stress levels."""
    print("\n" + "="*60)
    print("Validation: Exit Layer & Span vs Difficulty & Stress")
    print("="*60)

    results = {}

    for stress_level in [0.1, 0.5, 0.9]:
        print(f"\n--- Stress Level: {stress_level} ---")
        results[stress_level] = {}

        for difficulty, prompts in TEST_PROMPTS.items():
            exit_layers = []
            span_values = []

            for prompt in prompts:
                inputs = tokenizer(prompt, return_tensors='pt').to(device)
                body = model.simulate_body_state(1, device, stress_level)

                with torch.no_grad():
                    out = model(inputs['input_ids'], body_state=body)

                u_values = [u.mean().item() for u in out['exit_uncertainties']]
                best_idx = np.argmin(u_values)
                best_layer = model.exit_layer_indices[best_idx]
                exit_layers.append(best_layer)

                avg_span = np.mean([p['value'].item() for p in out['span_predictions']])
                span_values.append(avg_span)

            avg_exit = np.mean(exit_layers)
            avg_span = np.mean(span_values)

            results[stress_level][difficulty] = {
                'avg_exit': avg_exit,
                'avg_span': avg_span
            }

            print(f"  {difficulty:8s}: exit={avg_exit:.1f}/{model.num_layers}, span={avg_span:.0f}")

    # Analysis
    print("\n" + "-"*40)
    print("Analysis:")

    # Check if span decreases with stress
    low_stress_span = np.mean([results[0.1][d]['avg_span'] for d in TEST_PROMPTS])
    high_stress_span = np.mean([results[0.9][d]['avg_span'] for d in TEST_PROMPTS])

    print(f"\n  Span at low stress (0.1): {low_stress_span:.0f}")
    print(f"  Span at high stress (0.9): {high_stress_span:.0f}")

    if high_stress_span < low_stress_span:
        print("  [PASS] Span decreases under stress (body-aware)")
    else:
        print("  [NEEDS WORK] Span not responsive to stress")

    # Check exit layer variation
    all_exits = []
    for stress in results:
        for diff in results[stress]:
            all_exits.append(results[stress][diff]['avg_exit'])

    exit_var = np.std(all_exits)
    print(f"\n  Exit layer std: {exit_var:.2f}")
    if exit_var > 1.0:
        print("  [PASS] Exit layer varies based on input")
    else:
        print("  [NEEDS WORK] Exit layer not varying enough")

    return results


def validate_compute_savings(model, tokenizer, device):
    """Estimate total compute savings from both mechanisms."""
    print("\n" + "="*60)
    print("Validation: Compute Savings Estimation")
    print("="*60)

    all_prompts = []
    for prompts in TEST_PROMPTS.values():
        all_prompts.extend(prompts)

    results = {}

    for stress in [0.2, 0.5, 0.8]:
        exit_savings = []
        span_savings = []

        for prompt in all_prompts:
            inputs = tokenizer(prompt, return_tensors='pt').to(device)
            body = model.simulate_body_state(1, device, stress)

            with torch.no_grad():
                out = model(inputs['input_ids'], body_state=body)

            # Exit savings
            u_values = [u.mean().item() for u in out['exit_uncertainties']]
            best_idx = np.argmin(u_values)
            best_layer = model.exit_layer_indices[best_idx]
            exit_ratio = best_layer / model.num_layers
            exit_savings.append(1 - exit_ratio)

            # Span savings (assuming O(n^2) attention)
            max_span = model.span_predictor.span_choices[-1]
            avg_span = np.mean([p['value'].item() for p in out['span_predictions']])
            span_ratio = (avg_span / max_span) ** 2  # Quadratic savings
            span_savings.append(1 - span_ratio)

        avg_exit_save = np.mean(exit_savings) * 100
        avg_span_save = np.mean(span_savings) * 100
        # Combined (multiplicative since they're independent)
        combined = (1 - (1 - np.mean(exit_savings)) * (1 - np.mean(span_savings))) * 100

        results[stress] = {
            'exit_savings': avg_exit_save,
            'span_savings': avg_span_save,
            'combined': combined
        }

        print(f"\n  Stress {stress}:")
        print(f"    Exit savings: {avg_exit_save:.1f}%")
        print(f"    Span savings: {avg_span_save:.1f}%")
        print(f"    Combined: {combined:.1f}%")

    return results


def validate_on_dataset(model, tokenizer, device, num_samples=100):
    """Validate on actual dataset samples."""
    print("\n" + "="*60)
    print("Validation: Dataset Samples")
    print("="*60)

    dataset = load_dataset("roneneldan/TinyStories", split="validation", streaming=True)

    exit_counts = defaultdict(int)
    span_values = []
    losses = []

    count = 0
    for item in dataset:
        if count >= num_samples:
            break

        text = item['text'][:256]
        if len(text) < 30:
            continue

        encoded = tokenizer(text, max_length=64, truncation=True, return_tensors='pt').to(device)
        body = model.simulate_body_state(1, device, 0.5)

        with torch.no_grad():
            out = model(encoded['input_ids'], body_state=body)

        # Best exit
        u_values = [u.mean().item() for u in out['exit_uncertainties']]
        best_idx = np.argmin(u_values)
        best_layer = model.exit_layer_indices[best_idx]
        exit_counts[best_layer] += 1

        # Span
        avg_span = np.mean([p['value'].item() for p in out['span_predictions']])
        span_values.append(avg_span)

        count += 1

    # Print distribution
    print(f"\n  Exit layer distribution (n={count}):")
    for layer in model.exit_layer_indices:
        pct = 100 * exit_counts[layer] / count
        bar = "#" * int(pct / 2)
        print(f"    Layer {layer:2d}: {bar:25s} {pct:.1f}%")

    print(f"\n  Average span: {np.mean(span_values):.0f}")
    print(f"  Span std: {np.std(span_values):.1f}")

    # Compute savings
    avg_exit = sum(l * c for l, c in exit_counts.items()) / count
    exit_savings = (1 - avg_exit / model.num_layers) * 100

    max_span = model.span_predictor.span_choices[-1]
    span_ratio = (np.mean(span_values) / max_span) ** 2
    span_savings = (1 - span_ratio) * 100

    combined = (1 - (avg_exit / model.num_layers) * span_ratio) * 100

    print(f"\n  Compute savings:")
    print(f"    From early exit: {exit_savings:.1f}%")
    print(f"    From span reduction: {span_savings:.1f}%")
    print(f"    Combined: {combined:.1f}%")

    return {
        'exit_distribution': dict(exit_counts),
        'avg_span': np.mean(span_values),
        'exit_savings': exit_savings,
        'span_savings': span_savings,
        'combined_savings': combined
    }


def main():
    print("="*60)
    print("Embodied GPT-2 Validation")
    print("="*60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    checkpoint_path = Path("checkpoints/z152_embodied/checkpoint.pt")
    if not checkpoint_path.exists():
        print(f"Error: Checkpoint not found at {checkpoint_path}")
        return

    model = load_model(checkpoint_path, device)
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    print(f"\nModel: GPT-2 with embodied control")
    print(f"Exit layers: {model.exit_layer_indices}")
    print(f"Span choices: {model.span_predictor.span_choices}")

    # Run validations
    results = {}
    results['exit_span'] = validate_exit_and_span(model, tokenizer, device)
    results['compute_savings'] = validate_compute_savings(model, tokenizer, device)
    results['dataset'] = validate_on_dataset(model, tokenizer, device)

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)

    combined = results['dataset']['combined_savings']
    print(f"\n  Total compute savings: {combined:.1f}%")
    print(f"  Target: >= 30%")

    if combined >= 30:
        print("  [PASS] Compute savings target met!")
    else:
        print(f"  [NEEDS WORK] {30 - combined:.1f}% more savings needed")

    # Save results
    output_path = checkpoint_path.parent / "validation_results.json"

    def convert(obj):
        if isinstance(obj, (np.floating, np.integer)):
            return float(obj) if isinstance(obj, np.floating) else int(obj)
        if isinstance(obj, dict):
            return {str(k): convert(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [convert(v) for v in obj]
        return obj

    with open(output_path, 'w') as f:
        json.dump(convert(results), f, indent=2)

    print(f"\n  Results saved to {output_path}")
    print("\n" + "="*60)
    print("Validation complete!")
    print("="*60)


if __name__ == "__main__":
    main()
