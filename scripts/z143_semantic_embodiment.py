#!/usr/bin/env python3
"""
z143_semantic_embodiment.py

REVISED EMBODIMENT TEST: Semantic Content → Body State
======================================================

Previous test (z141) FAILED because:
1. Used random model (no semantic content in hidden states)
2. Depth action created trivial timing correlation
3. Hidden state was noise, not signal

This test properly controls for these issues:
1. Uses PRE-TRAINED model (GPT-2) with meaningful hidden states
2. FIXES depth - vary only text difficulty
3. Tests if semantic difficulty predicts hardware state

Hypothesis:
- At SAME depth, "hard" text should use more compute
- Hidden state from "hard" text should predict higher power/time
- If h_t adds predictive power over b_t alone, embodiment is real

Run with:
  HSA_OVERRIDE_GFX_VERSION=11.0.0 python scripts/z143_semantic_embodiment.py
"""

import argparse
import json
import os
import sys
import time
import random
import numpy as np
import torch
import torch.nn as nn
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

from transformers import AutoTokenizer, AutoModelForCausalLM


@dataclass
class SemanticConfig:
    """Configuration for semantic embodiment test."""
    n_samples: int = 500
    sequence_length: int = 64
    fixed_depth: int = 6  # FIX depth to control for trivial causality
    world_hidden: int = 256
    world_epochs: int = 100
    batch_size: int = 32
    lr: float = 1e-3
    device: str = "cuda"
    seed: int = 42
    output_dir: str = "results/z143_semantic"


# Text difficulty categories with examples
DIFFICULTY_TEXTS = {
    "easy": [
        "The cat sat on the mat.",
        "The sun is bright today.",
        "I like to eat apples.",
        "The dog runs in the park.",
        "Birds fly in the sky.",
        "Water is wet and cold.",
        "The book is on the table.",
        "She has a red dress.",
        "The car is very fast.",
        "He walks to the store.",
    ],
    "medium": [
        "The quantum mechanical behavior of electrons determines the chemical properties of atoms.",
        "Economic policies must balance fiscal responsibility with social welfare programs.",
        "The mitochondria is the powerhouse of the cell, generating ATP through oxidative phosphorylation.",
        "Machine learning models require careful hyperparameter tuning for optimal performance.",
        "Climate change impacts biodiversity through habitat destruction and altered migration patterns.",
        "Recursive algorithms solve problems by breaking them into smaller subproblems.",
        "The Renaissance period marked a cultural rebirth in art, literature, and science.",
        "Neurotransmitters facilitate communication between neurons across synaptic gaps.",
        "Constitutional law governs the fundamental principles of governmental organization.",
        "Statistical inference allows us to draw conclusions about populations from samples.",
    ],
    "hard": [
        "Prove that the square root of 2 is irrational using proof by contradiction.",
        "Derive the Navier-Stokes equations from first principles of continuum mechanics.",
        "Explain why P versus NP is considered the most important open problem in computer science.",
        "Analyze the implications of Gödel's incompleteness theorems for mathematical foundations.",
        "Describe the mechanism by which CRISPR-Cas9 achieves precise genome editing.",
        "Calculate the expected value of a quantum observable given the density matrix formalism.",
        "Prove that every vector space has a basis using Zorn's lemma.",
        "Explain the holographic principle and its implications for black hole information paradox.",
        "Derive the Black-Scholes equation for option pricing from Itô calculus.",
        "Analyze the computational complexity of matrix multiplication using Strassen's algorithm.",
    ]
}


class TelemetryReader:
    """Read GPU telemetry."""

    def __init__(self, gpu_id: int = 1):
        self.gpu_id = gpu_id
        self.drm_path = f"/sys/class/drm/card{gpu_id}/device"
        self.gpu_hwmon = None

        hwmon_path = f"{self.drm_path}/hwmon"
        if os.path.exists(hwmon_path):
            hwmons = os.listdir(hwmon_path)
            if hwmons:
                self.gpu_hwmon = f"{hwmon_path}/{hwmons[0]}"

    def read(self, inference_time_ms: float = 0) -> np.ndarray:
        """Read telemetry."""
        telem = np.zeros(9, dtype=np.float32)

        try:
            if self.gpu_hwmon:
                power_path = f"{self.gpu_hwmon}/power1_average"
                if os.path.exists(power_path):
                    with open(power_path) as f:
                        power_uw = int(f.read().strip())
                        telem[0] = min(power_uw / 100_000_000, 1.0)

                temp_path = f"{self.gpu_hwmon}/temp1_input"
                if os.path.exists(temp_path):
                    with open(temp_path) as f:
                        temp_mc = int(f.read().strip())
                        telem[1] = min(temp_mc / 100_000, 1.0)

            util_path = f"{self.drm_path}/gpu_busy_percent"
            if os.path.exists(util_path):
                with open(util_path) as f:
                    util = int(f.read().strip())
                    telem[2] = util / 100.0

        except Exception:
            pass

        telem[8] = min(inference_time_ms / 5.0, 1.0)  # Timing channel
        return telem


class DynamicsOnlyWorld(nn.Module):
    """World model using ONLY telemetry (no hidden state)."""

    def __init__(self, telem_dim: int = 9, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(telem_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, telem_dim)
        )

    def forward(self, telem: torch.Tensor) -> torch.Tensor:
        return self.net(telem)


class SemanticEmbodiedWorld(nn.Module):
    """World model using telemetry + LM hidden state."""

    def __init__(self, telem_dim: int = 9, lm_hidden_dim: int = 768, hidden_dim: int = 256):
        super().__init__()

        # Project LM hidden to smaller dimension
        self.lm_proj = nn.Linear(lm_hidden_dim, 128)

        input_dim = telem_dim + 128
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, telem_dim)
        )

    def forward(self, telem: torch.Tensor, lm_hidden: torch.Tensor) -> torch.Tensor:
        lm_feat = self.lm_proj(lm_hidden.detach())
        x = torch.cat([telem, lm_feat], dim=-1)
        return self.net(x)


def collect_semantic_data(model, tokenizer, telem_reader, config: SemanticConfig) -> Dict:
    """Collect data with semantic difficulty variation."""

    print("Collecting semantic difficulty data...")

    data = {
        'telemetry': [],
        'hidden_states': [],
        'future_telemetry': [],
        'difficulty': [],
        'inference_times': []
    }

    difficulties = list(DIFFICULTY_TEXTS.keys())

    for i in range(config.n_samples):
        # Random difficulty
        diff = random.choice(difficulties)
        text = random.choice(DIFFICULTY_TEXTS[diff])

        # Encode
        inputs = tokenizer(
            text,
            return_tensors="pt",
            max_length=config.sequence_length,
            padding="max_length",
            truncation=True
        ).to(config.device)

        # Read telemetry BEFORE
        telem_before = telem_reader.read()

        # Run inference
        start_time = time.perf_counter()
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
        inference_time = (time.perf_counter() - start_time) * 1000

        # Get hidden state (last layer, mean pooled)
        hidden = outputs.hidden_states[-1].mean(dim=1).cpu().numpy().squeeze()

        # Read telemetry AFTER (with timing)
        telem_after = telem_reader.read(inference_time)

        # Store
        data['telemetry'].append(telem_before)
        data['hidden_states'].append(hidden)
        data['future_telemetry'].append(telem_after)
        data['difficulty'].append(diff)
        data['inference_times'].append(inference_time)

        if (i + 1) % 100 == 0:
            print(f"  Sample {i+1}/{config.n_samples}")

    return data


def analyze_difficulty_correlation(data: Dict) -> Dict:
    """Analyze if difficulty correlates with hardware state."""

    print("\nAnalyzing difficulty → hardware correlation...")

    difficulties = ['easy', 'medium', 'hard']
    diff_to_idx = {d: i for i, d in enumerate(difficulties)}

    # Group by difficulty
    by_difficulty = {d: {'times': [], 'power': [], 'temp': []} for d in difficulties}

    for i, diff in enumerate(data['difficulty']):
        by_difficulty[diff]['times'].append(data['inference_times'][i])
        by_difficulty[diff]['power'].append(data['future_telemetry'][i][0])
        by_difficulty[diff]['temp'].append(data['future_telemetry'][i][1])

    results = {}
    print("\n  Difficulty | Avg Time (ms) | Avg Power | Avg Temp")
    print("  " + "-" * 55)

    for diff in difficulties:
        times = np.array(by_difficulty[diff]['times'])
        power = np.array(by_difficulty[diff]['power'])
        temp = np.array(by_difficulty[diff]['temp'])

        results[diff] = {
            'avg_time': float(np.mean(times)),
            'std_time': float(np.std(times)),
            'avg_power': float(np.mean(power)),
            'avg_temp': float(np.mean(temp)),
            'n_samples': len(times)
        }

        print(f"  {diff:10} | {np.mean(times):13.3f} | {np.mean(power):9.4f} | {np.mean(temp):8.4f}")

    # Test if hard > medium > easy
    times_by_diff = [np.mean(by_difficulty[d]['times']) for d in difficulties]
    monotonic = times_by_diff[0] < times_by_diff[1] < times_by_diff[2]

    results['monotonic_times'] = monotonic
    print(f"\n  Monotonic (easy < medium < hard): {monotonic}")

    return results


def train_and_compare_models(data: Dict, config: SemanticConfig) -> Dict:
    """Train both models and compare."""

    print("\nTraining world models...")

    # Prepare tensors
    telem = torch.tensor(np.array(data['telemetry']), dtype=torch.float32, device=config.device)
    hidden = torch.tensor(np.array(data['hidden_states']), dtype=torch.float32, device=config.device)
    future = torch.tensor(np.array(data['future_telemetry']), dtype=torch.float32, device=config.device)

    # Split
    n_train = int(len(data['telemetry']) * 0.8)
    indices = torch.randperm(len(data['telemetry']))
    train_idx = indices[:n_train]
    test_idx = indices[n_train:]

    # Create models
    dynamics_model = DynamicsOnlyWorld(
        telem_dim=9, hidden_dim=config.world_hidden
    ).to(config.device)

    embodied_model = SemanticEmbodiedWorld(
        telem_dim=9, lm_hidden_dim=hidden.shape[1], hidden_dim=config.world_hidden
    ).to(config.device)

    opt_dyn = torch.optim.Adam(dynamics_model.parameters(), lr=config.lr)
    opt_emb = torch.optim.Adam(embodied_model.parameters(), lr=config.lr)

    # Training loop
    for epoch in range(config.world_epochs):
        dynamics_model.train()
        embodied_model.train()

        perm = torch.randperm(n_train)
        epoch_loss_dyn = 0
        epoch_loss_emb = 0
        n_batches = 0

        for i in range(0, n_train, config.batch_size):
            batch_idx = train_idx[perm[i:i+config.batch_size]]

            b_telem = telem[batch_idx]
            b_hidden = hidden[batch_idx]
            b_future = future[batch_idx]

            # Dynamics
            opt_dyn.zero_grad()
            pred_dyn = dynamics_model(b_telem)
            loss_dyn = nn.functional.mse_loss(pred_dyn, b_future)
            loss_dyn.backward()
            opt_dyn.step()

            # Embodied
            opt_emb.zero_grad()
            pred_emb = embodied_model(b_telem, b_hidden)
            loss_emb = nn.functional.mse_loss(pred_emb, b_future)
            loss_emb.backward()
            opt_emb.step()

            epoch_loss_dyn += loss_dyn.item()
            epoch_loss_emb += loss_emb.item()
            n_batches += 1

        if (epoch + 1) % 20 == 0:
            # Test evaluation
            dynamics_model.eval()
            embodied_model.eval()

            with torch.no_grad():
                test_telem = telem[test_idx]
                test_hidden = hidden[test_idx]
                test_future = future[test_idx]

                pred_dyn = dynamics_model(test_telem)
                pred_emb = embodied_model(test_telem, test_hidden)

                test_loss_dyn = nn.functional.mse_loss(pred_dyn, test_future).item()
                test_loss_emb = nn.functional.mse_loss(pred_emb, test_future).item()

            print(f"  Epoch {epoch+1}/{config.world_epochs}: "
                  f"Dyn={epoch_loss_dyn/n_batches:.6f} Emb={epoch_loss_emb/n_batches:.6f} "
                  f"Test: Dyn={test_loss_dyn:.6f} Emb={test_loss_emb:.6f}")

    # Final evaluation
    dynamics_model.eval()
    embodied_model.eval()

    with torch.no_grad():
        test_telem = telem[test_idx]
        test_hidden = hidden[test_idx]
        test_future = future[test_idx]

        pred_dyn = dynamics_model(test_telem)
        pred_emb = embodied_model(test_telem, test_hidden)

        final_dyn_mse = nn.functional.mse_loss(pred_dyn, test_future).item()
        final_emb_mse = nn.functional.mse_loss(pred_emb, test_future).item()

    improvement = (final_dyn_mse - final_emb_mse) / final_dyn_mse * 100 if final_dyn_mse > 0 else 0

    return {
        'dynamics_mse': final_dyn_mse,
        'embodied_mse': final_emb_mse,
        'improvement_pct': improvement,
        'embodied_wins': final_emb_mse < final_dyn_mse
    }


def main():
    parser = argparse.ArgumentParser(description="Semantic Embodiment Test")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--samples", type=int, default=500)
    parser.add_argument("--epochs", type=int, default=100)
    args = parser.parse_args()

    print("=" * 70)
    print("z143: SEMANTIC EMBODIMENT TEST")
    print("=" * 70)
    print("Does semantic content (text difficulty) affect hardware state?")
    print("Does h_t from PRE-TRAINED model add predictive power?")
    print("=" * 70)
    print(f"Started: {datetime.now().isoformat()}")

    config = SemanticConfig(
        device=args.device,
        n_samples=args.samples,
        world_epochs=args.epochs
    )

    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)

    os.makedirs(config.output_dir, exist_ok=True)

    # Load PRE-TRAINED model
    print("\nLoading pre-trained GPT-2...")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained("gpt2").to(config.device)
    model.eval()

    telem_reader = TelemetryReader(gpu_id=1)

    # Collect data
    data = collect_semantic_data(model, tokenizer, telem_reader, config)

    # Analyze difficulty correlation
    diff_results = analyze_difficulty_correlation(data)

    # Train and compare models
    model_results = train_and_compare_models(data, config)

    # Final verdict
    print("\n" + "=" * 70)
    print("VERDICT")
    print("=" * 70)

    print(f"\n1. DIFFICULTY → HARDWARE CORRELATION")
    print(f"   Monotonic timing (easy < medium < hard): {diff_results['monotonic_times']}")

    print(f"\n2. EMBODIED vs DYNAMICS")
    print(f"   Dynamics MSE: {model_results['dynamics_mse']:.6f}")
    print(f"   Embodied MSE: {model_results['embodied_mse']:.6f}")
    print(f"   Improvement:  {model_results['improvement_pct']:.2f}%")

    if model_results['embodied_wins'] and model_results['improvement_pct'] > 5:
        verdict = "SEMANTIC EMBODIMENT LIKELY - h_t adds predictive power"
    elif diff_results['monotonic_times']:
        verdict = "PARTIAL - difficulty affects timing, but h_t doesn't help prediction"
    else:
        verdict = "NOT PROVEN - no clear semantic → hardware relationship"

    print(f"\n   VERDICT: {verdict}")

    # Save results
    summary = {
        'timestamp': datetime.now().isoformat(),
        'config': {
            'samples': config.n_samples,
            'epochs': config.world_epochs,
            'fixed_depth': config.fixed_depth
        },
        'difficulty_analysis': diff_results,
        'model_comparison': model_results,
        'verdict': verdict
    }

    results_path = f"{config.output_dir}/semantic_results.json"
    with open(results_path, 'w') as f:
        json.dump(summary, f, indent=2, default=float)

    print(f"\nResults saved to: {results_path}")
    print(f"Completed: {datetime.now().isoformat()}")


if __name__ == "__main__":
    main()
