#!/usr/bin/env python3
"""
z133_rigorous_embodiment_validation.py

HONEST embodiment validation with proper baselines and control tests.
Implements the 4 tests that make embodiment claims falsifiable:

1. MISMATCH TEST: Score probe with telemetry A against target B
   - If probe still works → information leakage/bug
   - If probe fails → body signal is actually represented

2. NO-BODY TEST: Run with telemetry=zeros
   - Should drop to mean baseline
   - If it doesn't → model learned spurious correlations

3. PROPER BASELINES:
   - Mean predictor (always predict dataset mean)
   - Shuffled labels (train probe on permuted telemetry)
   - Last-layer only vs all-layers (selectivity)

4. SELECTIVITY TEST:
   - Random control task (predict random labels)
   - If probe learns random task equally well → probe has too much capacity

References:
- Hewitt & Liang (2019): "Designing and Interpreting Probes with Control Tasks"
- https://aclanthology.org/D19-1275/
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from feel_slm.embodied_slm import create_embodied_slm_30m


class TelemetryDataset(Dataset):
    """Dataset that provides text + telemetry pairs."""

    def __init__(self, texts: list, telemetry: np.ndarray):
        self.texts = texts
        self.telemetry = telemetry  # [N, 12] body vectors

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return {
            "text": self.texts[idx],
            "telemetry": self.telemetry[idx]
        }


class LinearProbe(nn.Module):
    """Simple linear probe for body state prediction."""

    def __init__(self, hidden_dim: int, body_dim: int = 12):
        super().__init__()
        self.probe = nn.Linear(hidden_dim, body_dim)

    def forward(self, hidden_states):
        # hidden_states: [batch, hidden] (already pooled)
        return self.probe(hidden_states)  # [batch, body_dim]


def generate_synthetic_telemetry(n_samples: int, seed: int = 42) -> np.ndarray:
    """Generate realistic synthetic telemetry data."""
    rng = np.random.RandomState(seed)

    telemetry = np.zeros((n_samples, 12), dtype=np.float32)

    for i in range(n_samples):
        # Simulate correlated GPU metrics
        base_load = rng.uniform(0.2, 0.9)

        # Power (normalized 0-1, based on load)
        telemetry[i, 0] = base_load * 0.8 + rng.uniform(-0.1, 0.1)

        # Temperature (correlated with power)
        telemetry[i, 1] = 0.3 + telemetry[i, 0] * 0.5 + rng.uniform(-0.05, 0.05)

        # Memory util
        telemetry[i, 2] = base_load * 0.7 + rng.uniform(-0.1, 0.2)

        # GPU util
        telemetry[i, 3] = base_load + rng.uniform(-0.1, 0.1)

        # Clock speeds (normalized)
        telemetry[i, 4] = 0.5 + base_load * 0.4 + rng.uniform(-0.05, 0.05)
        telemetry[i, 5] = 0.5 + base_load * 0.3 + rng.uniform(-0.05, 0.05)

        # Derived features
        telemetry[i, 6] = telemetry[i, 0] - telemetry[i, 1]  # power - temp
        telemetry[i, 7] = telemetry[i, 0] * telemetry[i, 3]  # power * util
        telemetry[i, 8] = telemetry[i, 1] / (telemetry[i, 0] + 0.1)  # efficiency proxy

        # Time-varying components
        telemetry[i, 9] = np.sin(i / 100) * 0.1  # cyclic
        telemetry[i, 10] = rng.uniform(-0.1, 0.1)  # noise
        telemetry[i, 11] = base_load  # load indicator

    # Clip to valid range
    telemetry = np.clip(telemetry, -1, 1)

    return telemetry


def load_tinystories_texts(n_samples: int = 1000) -> list:
    """Load TinyStories validation texts."""
    try:
        from datasets import load_dataset
        ds = load_dataset("roneneldan/TinyStories", split="validation")
        texts = [ds[i]["text"][:512] for i in range(min(n_samples, len(ds)))]
        return texts
    except Exception as e:
        print(f"Warning: Could not load TinyStories: {e}")
        print("Using synthetic texts...")
        return [f"This is sample text number {i}." for i in range(n_samples)]


def extract_hidden_states(
    model: nn.Module,
    tokenizer,
    texts: list,
    telemetry: np.ndarray,
    device: torch.device,
    batch_size: int = 8,
    use_body: bool = True
) -> torch.Tensor:
    """Extract hidden states from model with optional body conditioning."""

    model.eval()
    all_hidden = []

    with torch.no_grad():
        for i in tqdm(range(0, len(texts), batch_size), desc="Extracting hidden states"):
            batch_texts = texts[i:i+batch_size]
            batch_telem = telemetry[i:i+batch_size]

            # Tokenize
            encodings = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=256,
                return_tensors="pt"
            )
            input_ids = encodings["input_ids"].to(device)

            # Clamp token IDs to model's vocab size (tokenizer vocab > model vocab)
            vocab_size = model.config.vocab_size
            input_ids = torch.clamp(input_ids, max=vocab_size - 1)

            # Prepare telemetry
            if use_body:
                telem_tensor = torch.tensor(batch_telem, dtype=torch.float32, device=device)
            else:
                telem_tensor = torch.zeros(len(batch_texts), 12, dtype=torch.float32, device=device)

            # Forward pass - get hidden states using hook on final norm
            hidden_capture = []

            def hook_fn(module, input, output):
                hidden_capture.append(output.detach())

            # Register hook on final layer norm
            if hasattr(model, 'trunk'):
                handle = model.trunk.norm.register_forward_hook(hook_fn)
                _ = model(input_ids, telemetry=telem_tensor)
                handle.remove()
                hidden = hidden_capture[0] if hidden_capture else None
            else:
                # Fallback for other model structures
                outputs = model(input_ids, telemetry=telem_tensor)
                hidden = outputs.get("hidden_states", None)

            if hidden is not None:
                # Mean pool across sequence dimension to handle variable lengths
                # hidden: [batch, seq, hidden_dim] -> [batch, hidden_dim]
                pooled_hidden = hidden.mean(dim=1)
                all_hidden.append(pooled_hidden.cpu())

    if all_hidden:
        return torch.cat(all_hidden, dim=0)  # [total_samples, hidden_dim]
    else:
        raise RuntimeError("Could not extract hidden states")


def train_probe(
    hidden_states: torch.Tensor,
    targets: np.ndarray,
    hidden_dim: int,
    epochs: int = 20,
    lr: float = 1e-3,
    device: torch.device = None
) -> tuple:
    """Train a linear probe and return final MSE."""

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    probe = LinearProbe(hidden_dim, body_dim=targets.shape[1]).to(device)
    optimizer = torch.optim.Adam(probe.parameters(), lr=lr)
    criterion = nn.MSELoss()

    # Prepare data
    hidden_states = hidden_states.to(device)
    targets_tensor = torch.tensor(targets, dtype=torch.float32, device=device)

    # Train/val split
    n = len(hidden_states)
    n_train = int(0.8 * n)

    train_hidden = hidden_states[:n_train]
    train_targets = targets_tensor[:n_train]
    val_hidden = hidden_states[n_train:]
    val_targets = targets_tensor[n_train:]

    best_val_mse = float('inf')

    for epoch in range(epochs):
        probe.train()

        # Simple full-batch training
        optimizer.zero_grad()
        preds = probe(train_hidden)
        loss = criterion(preds, train_targets)
        loss.backward()
        optimizer.step()

        # Validation
        probe.eval()
        with torch.no_grad():
            val_preds = probe(val_hidden)
            val_mse = criterion(val_preds, val_targets).item()

        if val_mse < best_val_mse:
            best_val_mse = val_mse

    return probe, best_val_mse


def run_test_1_mismatch(
    model, tokenizer, texts, telemetry_a, telemetry_b, device, hidden_dim
) -> dict:
    """
    TEST 1: Mismatch test

    Extract hidden states with telemetry A, but train probe to predict telemetry B.
    If probe succeeds → BUG (information shouldn't transfer)
    If probe fails → Body signal is genuinely represented
    """
    print("\n" + "="*60)
    print("TEST 1: MISMATCH TEST")
    print("="*60)
    print("Hidden states from telemetry A, probe predicts telemetry B")
    print("Expected: HIGH MSE (probe should fail)")

    # Extract hidden states with telemetry A
    hidden_a = extract_hidden_states(model, tokenizer, texts, telemetry_a, device, use_body=True)

    # Train probe to predict telemetry B (should fail!)
    probe, mse_mismatch = train_probe(hidden_a, telemetry_b, hidden_dim, device=device)

    # Also train probe to predict telemetry A (should succeed)
    probe_match, mse_match = train_probe(hidden_a, telemetry_a, hidden_dim, device=device)

    result = {
        "test": "mismatch",
        "mse_matched": mse_match,
        "mse_mismatched": mse_mismatch,
        "ratio": mse_mismatch / (mse_match + 1e-8),
        "passed": mse_mismatch > mse_match * 5  # Mismatch should be much worse
    }

    print(f"  MSE (matched A→A):    {mse_match:.6f}")
    print(f"  MSE (mismatched A→B): {mse_mismatch:.6f}")
    print(f"  Ratio (should be >>1): {result['ratio']:.2f}")
    print(f"  PASSED: {result['passed']}")

    return result


def run_test_2_no_body(
    model, tokenizer, texts, telemetry, device, hidden_dim
) -> dict:
    """
    TEST 2: No-body test

    Run model with telemetry=zeros, probe should fail.
    If probe still works → Model learned spurious correlations
    """
    print("\n" + "="*60)
    print("TEST 2: NO-BODY TEST")
    print("="*60)
    print("Hidden states from telemetry=zeros, probe predicts real telemetry")
    print("Expected: HIGH MSE (no body info to decode)")

    # Extract hidden states with real body
    hidden_body = extract_hidden_states(model, tokenizer, texts, telemetry, device, use_body=True)

    # Extract hidden states with NO body (zeros)
    hidden_nobody = extract_hidden_states(model, tokenizer, texts, telemetry, device, use_body=False)

    # Train probe on body-conditioned hidden → should work
    probe_body, mse_body = train_probe(hidden_body, telemetry, hidden_dim, device=device)

    # Train probe on no-body hidden → should fail
    probe_nobody, mse_nobody = train_probe(hidden_nobody, telemetry, hidden_dim, device=device)

    result = {
        "test": "no_body",
        "mse_with_body": mse_body,
        "mse_without_body": mse_nobody,
        "ratio": mse_nobody / (mse_body + 1e-8),
        "passed": mse_nobody > mse_body * 3  # No-body should be much worse
    }

    print(f"  MSE (with body):    {mse_body:.6f}")
    print(f"  MSE (without body): {mse_nobody:.6f}")
    print(f"  Ratio (should be >>1): {result['ratio']:.2f}")
    print(f"  PASSED: {result['passed']}")

    return result


def run_test_3_baselines(
    model, tokenizer, texts, telemetry, device, hidden_dim
) -> dict:
    """
    TEST 3: Proper baselines

    Compare probe MSE against:
    - Mean predictor (always predict dataset mean)
    - Shuffled labels (train probe on permuted telemetry)
    """
    print("\n" + "="*60)
    print("TEST 3: PROPER BASELINES")
    print("="*60)
    print("Compare against mean predictor and shuffled labels")
    print("Expected: Probe MSE << Mean MSE, Probe MSE << Shuffled MSE")

    # Extract hidden states
    hidden = extract_hidden_states(model, tokenizer, texts, telemetry, device, use_body=True)

    # Trained probe MSE
    probe, mse_probe = train_probe(hidden, telemetry, hidden_dim, device=device)

    # Mean predictor baseline
    mean_pred = telemetry.mean(axis=0, keepdims=True)
    mse_mean = ((telemetry - mean_pred) ** 2).mean()

    # Shuffled labels baseline
    rng = np.random.RandomState(123)
    shuffled_telemetry = telemetry.copy()
    rng.shuffle(shuffled_telemetry)
    _, mse_shuffled = train_probe(hidden, shuffled_telemetry, hidden_dim, device=device)

    result = {
        "test": "baselines",
        "mse_probe": mse_probe,
        "mse_mean_predictor": float(mse_mean),
        "mse_shuffled_labels": mse_shuffled,
        "improvement_vs_mean": (mse_mean - mse_probe) / mse_mean * 100,
        "improvement_vs_shuffled": (mse_shuffled - mse_probe) / mse_shuffled * 100,
        "passed": mse_probe < mse_mean * 0.5 and mse_probe < mse_shuffled * 0.5
    }

    print(f"  MSE (trained probe):    {mse_probe:.6f}")
    print(f"  MSE (mean predictor):   {mse_mean:.6f}")
    print(f"  MSE (shuffled labels):  {mse_shuffled:.6f}")
    print(f"  Improvement vs mean:    {result['improvement_vs_mean']:.1f}%")
    print(f"  Improvement vs shuffled: {result['improvement_vs_shuffled']:.1f}%")
    print(f"  PASSED: {result['passed']}")

    return result


def run_test_4_selectivity(
    model, tokenizer, texts, telemetry, device, hidden_dim
) -> dict:
    """
    TEST 4: Selectivity test (control task)

    Train probe on random labels. If it learns equally well,
    the probe has too much capacity (not measuring representation).

    Based on Hewitt & Liang (2019).
    """
    print("\n" + "="*60)
    print("TEST 4: SELECTIVITY (Control Task)")
    print("="*60)
    print("Train probe on random labels - should NOT learn well")
    print("If it does, probe capacity is too high")

    # Extract hidden states
    hidden = extract_hidden_states(model, tokenizer, texts, telemetry, device, use_body=True)

    # Real task
    _, mse_real = train_probe(hidden, telemetry, hidden_dim, device=device)

    # Random control task (random labels, same shape)
    rng = np.random.RandomState(456)
    random_labels = rng.randn(*telemetry.shape).astype(np.float32)
    _, mse_random = train_probe(hidden, random_labels, hidden_dim, device=device)

    # Selectivity = how much better real task is vs random
    selectivity = (mse_random - mse_real) / mse_random

    result = {
        "test": "selectivity",
        "mse_real_task": mse_real,
        "mse_random_task": mse_random,
        "selectivity": selectivity,
        "passed": selectivity > 0.5  # Real task should be much better
    }

    print(f"  MSE (real task):   {mse_real:.6f}")
    print(f"  MSE (random task): {mse_random:.6f}")
    print(f"  Selectivity:       {selectivity:.3f} (should be >0.5)")
    print(f"  PASSED: {result['passed']}")

    return result


def main():
    parser = argparse.ArgumentParser(description="Rigorous embodiment validation")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--n-samples", type=int, default=500, help="Number of samples")
    parser.add_argument("--output-dir", type=str, default="results/z133_validation")
    args = parser.parse_args()

    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    os.makedirs(args.output_dir, exist_ok=True)

    # Load model
    print("\nLoading model...")
    model = create_embodied_slm_30m().to(device)

    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model.eval()
    hidden_dim = model.config.hidden_size
    print(f"Model loaded. Hidden dim: {hidden_dim}")

    # Load tokenizer (must match training - gpt-neo-125M has vocab 50257 but model uses 32000)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-125M")
    tokenizer.pad_token = tokenizer.eos_token

    # Load data
    print(f"\nLoading {args.n_samples} samples...")
    texts = load_tinystories_texts(args.n_samples)

    # Generate two different telemetry sets for mismatch test
    telemetry_a = generate_synthetic_telemetry(len(texts), seed=42)
    telemetry_b = generate_synthetic_telemetry(len(texts), seed=999)  # Different seed!

    print(f"Loaded {len(texts)} texts")
    print(f"Telemetry A shape: {telemetry_a.shape}")
    print(f"Telemetry B shape: {telemetry_b.shape}")

    # Run all tests
    results = {
        "timestamp": datetime.now().isoformat(),
        "checkpoint": args.checkpoint,
        "n_samples": len(texts),
        "tests": {}
    }

    try:
        results["tests"]["test_1_mismatch"] = run_test_1_mismatch(
            model, tokenizer, texts, telemetry_a, telemetry_b, device, hidden_dim
        )
    except Exception as e:
        print(f"TEST 1 FAILED: {e}")
        results["tests"]["test_1_mismatch"] = {"error": str(e), "passed": False}

    try:
        results["tests"]["test_2_no_body"] = run_test_2_no_body(
            model, tokenizer, texts, telemetry_a, device, hidden_dim
        )
    except Exception as e:
        print(f"TEST 2 FAILED: {e}")
        results["tests"]["test_2_no_body"] = {"error": str(e), "passed": False}

    try:
        results["tests"]["test_3_baselines"] = run_test_3_baselines(
            model, tokenizer, texts, telemetry_a, device, hidden_dim
        )
    except Exception as e:
        print(f"TEST 3 FAILED: {e}")
        results["tests"]["test_3_baselines"] = {"error": str(e), "passed": False}

    try:
        results["tests"]["test_4_selectivity"] = run_test_4_selectivity(
            model, tokenizer, texts, telemetry_a, device, hidden_dim
        )
    except Exception as e:
        print(f"TEST 4 FAILED: {e}")
        results["tests"]["test_4_selectivity"] = {"error": str(e), "passed": False}

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)

    all_passed = all(
        t.get("passed", False)
        for t in results["tests"].values()
        if isinstance(t, dict)
    )

    results["all_tests_passed"] = all_passed
    results["verdict"] = "EMBODIMENT VERIFIED" if all_passed else "EMBODIMENT NOT PROVEN"

    for name, test in results["tests"].items():
        status = "✅ PASS" if test.get("passed", False) else "❌ FAIL"
        print(f"  {name}: {status}")

    print(f"\nFINAL VERDICT: {results['verdict']}")

    # Save results (convert numpy types to native Python)
    def convert_to_serializable(obj):
        if isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [convert_to_serializable(item) for item in obj]
        elif isinstance(obj, (np.floating, np.integer)):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, bool):
            return bool(obj)
        return obj

    output_file = os.path.join(args.output_dir, "rigorous_validation.json")
    with open(output_file, "w") as f:
        json.dump(convert_to_serializable(results), f, indent=2)
    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    main()
