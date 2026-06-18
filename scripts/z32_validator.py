#!/usr/bin/env python3
"""
FEEL z32 Validator - Causal Proof on Daedalus
==============================================

Runs on daedalus while training runs on ikaros.
Tests the causal chain: SENSE → FEEL → EXPRESS
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
from typing import List, Dict

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.sensors.canonical_features import CanonicalSensorHub, SENSOR_DIM


class EmbodiedGateNet(nn.Module):
    """Gate network - must match training."""

    def __init__(self, sensor_dim: int = SENSOR_DIM, hidden_dim: int = 64, num_layers: int = 5):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(sensor_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        self.gate_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, 32),
                nn.GELU(),
                nn.Linear(32, 1),
                nn.Sigmoid()
            )
            for _ in range(num_layers)
        ])
        self.dvfs_head = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.GELU(),
            nn.Linear(32, 3),
        )

    def forward(self, sensors):
        if sensors.dim() == 1:
            sensors = sensors.unsqueeze(0)
        h = self.encoder(sensors)
        gates = [head(h) for head in self.gate_heads]
        dvfs_logits = self.dvfs_head(h)
        return gates, dvfs_logits


def run_causal_validation(
    checkpoint_path: str,
    num_samples: int = 50,
    device: str = "cuda"
) -> Dict:
    """
    Run causal validation: inject stress, measure gate response.
    """
    print(f"\n{'='*60}")
    print("FEEL z32 CAUSAL VALIDATION")
    print(f"{'='*60}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Samples: {num_samples}")
    print()

    # Load checkpoint
    ckpt = torch.load(checkpoint_path, map_location=device)
    print(f"Loaded checkpoint from step {ckpt.get('step', '?')}")

    # Initialize gate network
    gate_net = EmbodiedGateNet(sensor_dim=SENSOR_DIM, num_layers=5).to(device)
    gate_net.load_state_dict(ckpt['gate_net_state_dict'])
    gate_net.eval()

    # Initialize sensor hub
    sensor_hub = CanonicalSensorHub()

    # Test causal response
    results = {
        'stressed_gates': [],
        'relaxed_gates': [],
        'gate_diffs': [],
        'causal_scores': []
    }

    print("\nRunning causal tests...")
    print("-" * 50)

    for i in range(num_samples):
        # Get stressed and relaxed sensor vectors
        stressed = sensor_hub.inject_stress(0.9).to(device)
        relaxed = sensor_hub.inject_stress(0.1).to(device)

        with torch.no_grad():
            gates_s, _ = gate_net(stressed)
            gates_r, _ = gate_net(relaxed)

        # Compute mean gates
        mean_s = sum(g.item() for g in gates_s) / len(gates_s)
        mean_r = sum(g.item() for g in gates_r) / len(gates_r)
        diff = mean_r - mean_s

        results['stressed_gates'].append(mean_s)
        results['relaxed_gates'].append(mean_r)
        results['gate_diffs'].append(diff)

        # Causal score: 1 if relaxed > stressed, 0 otherwise
        causal_score = 1.0 if diff > 0 else 0.0
        results['causal_scores'].append(causal_score)

        if (i + 1) % 10 == 0:
            print(f"  [{i+1}/{num_samples}] stressed={mean_s:.4f} relaxed={mean_r:.4f} diff={diff:.4f}")

    # Compute summary
    avg_stressed = sum(results['stressed_gates']) / len(results['stressed_gates'])
    avg_relaxed = sum(results['relaxed_gates']) / len(results['relaxed_gates'])
    avg_diff = sum(results['gate_diffs']) / len(results['gate_diffs'])
    causal_accuracy = sum(results['causal_scores']) / len(results['causal_scores'])

    print()
    print("=" * 60)
    print("CAUSAL VALIDATION RESULTS")
    print("=" * 60)
    print(f"  Avg Gate (stressed): {avg_stressed:.4f}")
    print(f"  Avg Gate (relaxed):  {avg_relaxed:.4f}")
    print(f"  Avg Gate Diff:       {avg_diff:.4f}")
    print(f"  Causal Accuracy:     {causal_accuracy*100:.1f}%")
    print()

    if causal_accuracy >= 0.8:
        print("✅ CAUSAL CHAIN VALIDATED: SENSE → FEEL working!")
    elif causal_accuracy >= 0.6:
        print("⚠️  CAUSAL CHAIN PARTIAL: Learning in progress")
    else:
        print("❌ CAUSAL CHAIN WEAK: More training needed")

    print("=" * 60)

    return {
        'avg_stressed': avg_stressed,
        'avg_relaxed': avg_relaxed,
        'avg_diff': avg_diff,
        'causal_accuracy': causal_accuracy,
        'step': ckpt.get('step', 0)
    }


def watch_and_validate(
    checkpoint_dir: str,
    interval: int = 120,
    num_samples: int = 50
):
    """
    Watch for new checkpoints and validate them.
    """
    print(f"Watching {checkpoint_dir} for new checkpoints...")
    print(f"Validation interval: {interval}s")
    print()

    validated = set()

    while True:
        # Find checkpoints
        ckpt_dir = Path(checkpoint_dir)
        checkpoints = sorted(ckpt_dir.glob("step_*.pt"))

        for ckpt in checkpoints:
            if str(ckpt) not in validated:
                print(f"\n[{time.strftime('%H:%M:%S')}] New checkpoint: {ckpt.name}")
                try:
                    results = run_causal_validation(str(ckpt), num_samples)
                    validated.add(str(ckpt))

                    # Save results
                    results_file = ckpt.with_suffix('.val.json')
                    with open(results_file, 'w') as f:
                        json.dump(results, f, indent=2)
                    print(f"Results saved to {results_file}")

                except Exception as e:
                    print(f"Validation failed: {e}")

        time.sleep(interval)


def main():
    parser = argparse.ArgumentParser(description="FEEL z32 Validator")
    parser.add_argument("--checkpoint", type=str, help="Single checkpoint to validate")
    parser.add_argument("--watch", type=str, help="Watch directory for new checkpoints")
    parser.add_argument("--interval", type=int, default=120, help="Watch interval (seconds)")
    parser.add_argument("--samples", type=int, default=50, help="Number of causal test samples")
    args = parser.parse_args()

    if args.checkpoint:
        run_causal_validation(args.checkpoint, args.samples)
    elif args.watch:
        watch_and_validate(args.watch, args.interval, args.samples)
    else:
        # Default: validate latest checkpoint
        ckpt_dir = Path("models/z32_embodied")
        checkpoints = sorted(ckpt_dir.glob("step_*.pt"))
        if checkpoints:
            run_causal_validation(str(checkpoints[-1]), args.samples)
        else:
            print("No checkpoints found. Use --checkpoint or --watch")


if __name__ == "__main__":
    main()
