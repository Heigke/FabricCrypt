#!/usr/bin/env python3
"""
Validation script for z151 minimal early exit model.

Tests:
1. Exit layer distribution across different prompt types
2. Quality comparison (early exit vs final layer)
3. Uncertainty calibration
4. Compute savings estimation
"""

import os
os.environ["HSA_OVERRIDE_GFX_VERSION"] = "11.0.0"

import sys
import json
import time
from pathlib import Path
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


class ExitHead(nn.Module):
    """Simple exit head"""
    def __init__(self, hidden_dim, vocab_size):
        super().__init__()
        self.proj = nn.Linear(hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.out = nn.Linear(hidden_dim, vocab_size, bias=False)

    def forward(self, x):
        return self.out(self.norm(F.gelu(self.proj(x)) + x))


class EarlyExitGPT2(nn.Module):
    def __init__(self, model_name="gpt2"):
        super().__init__()
        self.base = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32)

        for p in self.base.parameters():
            p.requires_grad = False

        self.hidden_dim = self.base.config.hidden_size
        self.vocab_size = self.base.config.vocab_size
        self.num_layers = self.base.config.num_hidden_layers

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

        self.uncertainty = nn.Sequential(
            nn.Linear(self.hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, input_ids, labels=None):
        with torch.no_grad():
            out = self.base(input_ids, output_hidden_states=True)

        hidden_states = out.hidden_states
        final_logits = out.logits

        exit_outputs = []
        uncertainties = []

        for i, layer_idx in enumerate(self.exit_layer_indices):
            h = hidden_states[layer_idx]
            logits = self.exit_heads[i](h)
            exit_outputs.append(logits)
            u = self.uncertainty(h[:, -1, :])
            uncertainties.append(u)

        if labels is not None:
            # Compute per-exit losses
            ce_losses = []
            shift_labels = labels[..., 1:].contiguous()

            for logits in exit_outputs:
                shift_logits = logits[..., :-1, :].contiguous()
                ce = F.cross_entropy(
                    shift_logits.view(-1, self.vocab_size),
                    shift_labels.view(-1),
                    ignore_index=-100
                )
                ce_losses.append(ce.item())

            return exit_outputs, uncertainties, ce_losses, final_logits

        return exit_outputs, uncertainties, final_logits


def load_model(checkpoint_path, device):
    """Load trained model from checkpoint"""
    print(f"Loading model from {checkpoint_path}...")

    model = EarlyExitGPT2("gpt2").to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.exit_heads.load_state_dict(checkpoint['exit_heads'])
    model.uncertainty.load_state_dict(checkpoint['uncertainty'])

    model.eval()
    return model


# Test prompts categorized by expected difficulty
TEST_PROMPTS = {
    "easy": [
        "The cat sat on the",
        "Once upon a time",
        "Hello, my name is",
        "The sun rises in the",
        "One plus one equals",
        "The dog ran to the",
        "She said hello and",
        "The sky is very",
    ],
    "medium": [
        "The meaning of life is",
        "In the year 2050",
        "The economy depends on",
        "Scientists believe that",
        "The most important thing about",
        "Democracy requires that",
        "Technology has changed how we",
        "Education should focus on",
    ],
    "hard": [
        "Quantum entanglement suggests that",
        "The epistemological implications of",
        "Transcendental phenomenology argues that",
        "The synthesis of heterogeneous",
        "Gödel's incompleteness theorem proves",
        "The hermeneutic circle involves",
        "Differential topology studies how",
        "The categorical imperative requires",
    ]
}


def validate_exit_distribution(model, tokenizer, device):
    """Test exit layer distribution across prompt difficulties"""
    print("\n" + "="*60)
    print("Validation 1: Exit Layer Distribution")
    print("="*60)

    results = {}

    for difficulty, prompts in TEST_PROMPTS.items():
        exit_layers = []
        uncertainties_by_exit = defaultdict(list)

        for prompt in prompts:
            inputs = tokenizer(prompt, return_tensors='pt').to(device)

            with torch.no_grad():
                exits, uncertainties, _ = model(inputs['input_ids'])

            u_values = [u.item() for u in uncertainties]
            best_exit_idx = np.argmin(u_values)
            best_layer = model.exit_layer_indices[best_exit_idx]

            exit_layers.append(best_layer)
            for i, u in enumerate(u_values):
                uncertainties_by_exit[model.exit_layer_indices[i]].append(u)

        avg_exit = np.mean(exit_layers)
        exit_distribution = {l: exit_layers.count(l) / len(exit_layers) for l in model.exit_layer_indices}

        results[difficulty] = {
            'avg_exit_layer': avg_exit,
            'exit_distribution': exit_distribution,
            'avg_uncertainty': {l: np.mean(u) for l, u in uncertainties_by_exit.items()}
        }

        print(f"\n{difficulty.upper()} prompts:")
        print(f"  Average exit layer: {avg_exit:.2f}/{model.num_layers}")
        print(f"  Distribution: {exit_distribution}")

    # Check if harder prompts exit later (as expected)
    easy_avg = results['easy']['avg_exit_layer']
    hard_avg = results['hard']['avg_exit_layer']

    print(f"\n  Expected: Hard prompts should exit later than easy prompts")
    print(f"  Result: Easy={easy_avg:.2f}, Hard={hard_avg:.2f}")

    if hard_avg >= easy_avg:
        print("  [PASS] Model exits later for harder prompts")
    else:
        print("  [WARN] Model exits earlier for harder prompts (may need more training)")

    return results


def validate_quality(model, tokenizer, device):
    """Compare quality between early exits and final layer"""
    print("\n" + "="*60)
    print("Validation 2: Quality Comparison")
    print("="*60)

    # Use a subset of prompts with known continuations
    test_cases = [
        ("The cat sat on the", " mat"),
        ("Once upon a time there was a", " princess"),
        ("The sun rises in the", " east"),
        ("One plus one equals", " two"),
    ]

    results = []

    for prompt, expected_token in test_cases:
        inputs = tokenizer(prompt, return_tensors='pt').to(device)
        # Use same input_ids as labels (for loss computation on next token prediction)
        labels = inputs['input_ids'].clone()

        with torch.no_grad():
            exits, uncertainties, ce_losses, final_logits = model(inputs['input_ids'], labels=labels)

        # Get predictions from each exit
        predictions = []
        for i, logits in enumerate(exits):
            next_token = logits[0, -1].argmax().item()
            pred_word = tokenizer.decode([next_token])
            predictions.append((model.exit_layer_indices[i], pred_word))

        # Final layer prediction
        final_pred = tokenizer.decode([final_logits[0, -1].argmax().item()])

        # Best exit by uncertainty
        u_values = [u.item() for u in uncertainties]
        best_idx = np.argmin(u_values)
        best_layer = model.exit_layer_indices[best_idx]
        best_pred = tokenizer.decode([exits[best_idx][0, -1].argmax().item()])

        results.append({
            'prompt': prompt,
            'expected': expected_token,
            'final_pred': final_pred,
            'best_exit_pred': best_pred,
            'best_exit_layer': best_layer,
            'ce_losses': ce_losses
        })

        print(f"\nPrompt: '{prompt}'")
        print(f"  Expected: '{expected_token}'")
        print(f"  Final layer ({model.num_layers}): '{final_pred}'")
        print(f"  Best exit (layer {best_layer}): '{best_pred}'")
        print(f"  CE losses by layer: {[f'{l:.3f}' for l in ce_losses]}")

    # Check quality preservation
    matches_final = sum(1 for r in results if r['best_exit_pred'].strip() == r['final_pred'].strip())
    print(f"\n  Best exit matches final: {matches_final}/{len(results)} ({100*matches_final/len(results):.0f}%)")

    return results


def validate_compute_savings(model, tokenizer, device):
    """Estimate compute savings from early exit"""
    print("\n" + "="*60)
    print("Validation 3: Compute Savings Estimation")
    print("="*60)

    all_prompts = []
    for prompts in TEST_PROMPTS.values():
        all_prompts.extend(prompts)

    exit_layers = []

    for prompt in all_prompts:
        inputs = tokenizer(prompt, return_tensors='pt').to(device)

        with torch.no_grad():
            exits, uncertainties, _ = model(inputs['input_ids'])

        u_values = [u.item() for u in uncertainties]
        best_exit_idx = np.argmin(u_values)
        best_layer = model.exit_layer_indices[best_exit_idx]
        exit_layers.append(best_layer)

    avg_exit = np.mean(exit_layers)
    compute_ratio = avg_exit / model.num_layers
    savings = (1 - compute_ratio) * 100

    print(f"\n  Average exit layer: {avg_exit:.2f}/{model.num_layers}")
    print(f"  Compute ratio: {compute_ratio:.2%}")
    print(f"  Estimated compute savings: {savings:.1f}%")

    # Distribution
    print(f"\n  Exit layer distribution:")
    for layer in model.exit_layer_indices:
        count = exit_layers.count(layer)
        pct = 100 * count / len(exit_layers)
        bar = "#" * int(pct / 2)
        print(f"    Layer {layer:2d}: {bar:25s} {pct:.1f}%")

    return {
        'avg_exit_layer': avg_exit,
        'compute_ratio': compute_ratio,
        'savings_pct': savings,
        'distribution': {l: exit_layers.count(l) for l in model.exit_layer_indices}
    }


def validate_uncertainty_calibration(model, tokenizer, device):
    """Check if uncertainty correlates with actual prediction error"""
    print("\n" + "="*60)
    print("Validation 4: Uncertainty Calibration")
    print("="*60)

    # Test on a larger set
    from datasets import load_dataset
    print("Loading validation data...")
    dataset = load_dataset("roneneldan/TinyStories", split="validation", streaming=True)

    uncertainties_by_bin = defaultdict(list)
    errors_by_bin = defaultdict(list)

    count = 0
    for item in dataset:
        if count >= 100:
            break

        text = item['text'][:256]
        if len(text) < 50:
            continue

        encoded = tokenizer(
            text,
            max_length=64,
            truncation=True,
            return_tensors='pt'
        ).to(device)

        labels = encoded['input_ids'].clone()

        with torch.no_grad():
            exits, uncertainties, ce_losses, _ = model(encoded['input_ids'], labels=labels)

        # For each exit, record uncertainty and CE loss
        for i, (u, ce) in enumerate(zip(uncertainties, ce_losses)):
            u_val = u.item()
            # Bin uncertainty into 10 bins
            bin_idx = min(int(u_val * 10), 9)
            uncertainties_by_bin[bin_idx].append(u_val)
            errors_by_bin[bin_idx].append(ce)

        count += 1

    print(f"\n  Uncertainty vs Error correlation:")
    print(f"  {'Bin':>6} | {'Uncertainty':>12} | {'Avg CE Loss':>12} | Count")
    print(f"  {'-'*6}-+-{'-'*12}-+-{'-'*12}-+------")

    for bin_idx in sorted(uncertainties_by_bin.keys()):
        avg_u = np.mean(uncertainties_by_bin[bin_idx])
        avg_err = np.mean(errors_by_bin[bin_idx])
        count = len(uncertainties_by_bin[bin_idx])
        print(f"  {bin_idx:6d} | {avg_u:12.3f} | {avg_err:12.3f} | {count}")

    # Compute correlation
    all_u = []
    all_err = []
    for bin_idx in uncertainties_by_bin:
        all_u.extend(uncertainties_by_bin[bin_idx])
        all_err.extend(errors_by_bin[bin_idx])

    if len(all_u) > 1:
        correlation = np.corrcoef(all_u, all_err)[0, 1]
        print(f"\n  Correlation (uncertainty vs error): {correlation:.3f}")
        if correlation > 0:
            print("  [PASS] Positive correlation - higher uncertainty = higher error")
        else:
            print("  [WARN] Negative or zero correlation - uncertainty not well calibrated")


def main():
    print("="*60)
    print("Early Exit Model Validation")
    print("="*60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Use v2 checkpoint if available, else fall back to minimal
    checkpoint_path = Path("checkpoints/z151_v2/checkpoint.pt")
    if not checkpoint_path.exists():
        checkpoint_path = Path("checkpoints/z151_minimal/checkpoint.pt")
    if not checkpoint_path.exists():
        print(f"Error: Checkpoint not found at {checkpoint_path}")
        return

    model = load_model(checkpoint_path, device)
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    print(f"\nModel: GPT-2 with early exits at layers {model.exit_layer_indices}")
    print(f"Total layers: {model.num_layers}")

    # Run validations
    results = {}

    results['exit_distribution'] = validate_exit_distribution(model, tokenizer, device)
    results['quality'] = validate_quality(model, tokenizer, device)
    results['compute_savings'] = validate_compute_savings(model, tokenizer, device)
    validate_uncertainty_calibration(model, tokenizer, device)

    # Summary
    print("\n" + "="*60)
    print("VALIDATION SUMMARY")
    print("="*60)

    savings = results['compute_savings']['savings_pct']
    print(f"\n  Compute savings: {savings:.1f}%")
    print(f"  Target: >= 30%")

    if savings >= 30:
        print("  [PASS] Compute savings target met!")
    else:
        print(f"  [NEEDS WORK] {30 - savings:.1f}% more savings needed")

    # Save results
    output_path = checkpoint_path.parent / "validation_results.json"

    # Convert numpy types for JSON serialization
    def convert(obj):
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [convert(v) for v in obj]
        return obj

    results = convert(results)

    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n  Results saved to {output_path}")
    print("\n" + "="*60)
    print("Validation complete!")
    print("="*60)


if __name__ == "__main__":
    main()
