#!/usr/bin/env python3
"""
FEEL z31: Daedalus Validator
============================

Runs on daedalus (secondary machine) to validate checkpoints from ikaros training.
Logs results to the SAME wandb project for unified tracking.

Key tests:
1. SENSE→FEEL causal loop (gate_diff per layer)
2. Ablation modes (full/shuffle/frozen)
3. Per-layer gate response

IMPORTANT: Uses bfloat16 for all tensors to match training.
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
from typing import Dict, List
import numpy as np

os.environ.setdefault("HSA_OVERRIDE_GFX_VERSION", "11.0.0")

import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from transformers import AutoConfig


# =============================================================================
# SENSOR HUB (Simplified for validation - injection only)
# =============================================================================

class ValidationSensorHub:
    """Simplified sensor hub for validation - injection only."""

    SENSOR_DIM = 10

    def __init__(self, device: str = "cuda"):
        self.device = device
        self._tensor = torch.ones(self.SENSOR_DIM, device=device) * 0.5

    def inject(self, tensor: torch.Tensor):
        self._tensor = tensor.to(self.device)

    def clear_injection(self):
        self._tensor = torch.ones(self.SENSOR_DIM, device=self.device) * 0.5

    def read_tensor(self) -> torch.Tensor:
        return self._tensor.clone()

    @staticmethod
    def create_stressed_tensor(device: str = "cuda") -> torch.Tensor:
        """Stressed sensor state."""
        return torch.tensor([
            0.95, 0.85, 0.95, 0.80, 0.95,
            0.6, -0.4, 0.5, 0.90, 0.85,
        ], dtype=torch.float32, device=device)

    @staticmethod
    def create_relaxed_tensor(device: str = "cuda") -> torch.Tensor:
        """Relaxed sensor state."""
        return torch.tensor([
            0.25, 0.20, 0.30, 0.20, 0.40,
            1.3, 0.3, -0.3, 0.20, 0.15,
        ], dtype=torch.float32, device=device)


# =============================================================================
# CAUSAL-AWARE GATE NETWORK (Must match training!)
# =============================================================================

class CausalAwareGateNet(nn.Module):
    """Gate network - must match z31_comprehensive_trainer.py exactly."""

    def __init__(
        self,
        hidden_size: int,
        sensor_dim: int = 10,
        gate_hidden: int = 128,
        num_gates: int = 1,
        sensor_weight: float = 0.5,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.sensor_dim = sensor_dim
        self.num_gates = num_gates
        self.sensor_weight = sensor_weight

        self.sensor_encoder = nn.Sequential(
            nn.Linear(sensor_dim, 64),
            nn.GELU(),
            nn.Linear(64, gate_hidden),
            nn.GELU(),
        )

        self.causal_encoder = nn.Sequential(
            nn.Linear(4, 32),
            nn.Tanh(),
            nn.Linear(32, 32),
            nn.Tanh(),
        )

        self.hidden_compressor = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Linear(256, gate_hidden),
            nn.LayerNorm(gate_hidden),
            nn.GELU(),
        )

        self.film_gamma = nn.Linear(sensor_dim, gate_hidden)
        self.film_beta = nn.Linear(sensor_dim, gate_hidden)

        combined_dim = gate_hidden + 32 + gate_hidden
        self.gate_head = nn.Sequential(
            nn.Linear(combined_dim, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Linear(64, num_gates),
        )

    def forward(
        self,
        hidden_state: torch.Tensor,
        sensors: torch.Tensor,
    ) -> torch.Tensor:
        batch = hidden_state.shape[0]

        if sensors.dim() == 1:
            sensors_batch = sensors.unsqueeze(0).expand(batch, -1)
        else:
            sensors_batch = sensors

        sensor_features = self.sensor_encoder(sensors_batch)

        causal_dims = torch.stack([
            sensors_batch[:, 5],
            sensors_batch[:, 6],
            sensors_batch[:, 7],
            sensors_batch[:, 9],
        ], dim=-1)
        causal_features = self.causal_encoder(causal_dims)

        hidden_features = self.hidden_compressor(hidden_state)

        gamma = self.film_gamma(sensors_batch)
        beta = self.film_beta(sensors_batch)
        hidden_modulated = gamma * hidden_features + beta

        combined = torch.cat([sensor_features, causal_features, hidden_modulated], dim=-1)
        gate_raw = self.gate_head(combined)
        gates = torch.sigmoid(gate_raw)

        return gates


# =============================================================================
# VALIDATION FUNCTIONS
# =============================================================================

def run_causal_loop_test(
    gate_net: CausalAwareGateNet,
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
    num_trials: int = 20,
) -> Dict:
    """Test SENSE→FEEL causal loop for a single gate network."""
    gate_net.eval()

    stressed = ValidationSensorHub.create_stressed_tensor(device).to(dtype=dtype)
    relaxed = ValidationSensorHub.create_relaxed_tensor(device).to(dtype=dtype)

    # Fixed seed for reproducibility
    torch.manual_seed(42)

    hidden_size = gate_net.hidden_size
    dummy_hidden = torch.randn(1, hidden_size, device=device, dtype=dtype)

    stressed_gates = []
    relaxed_gates = []

    with torch.no_grad():
        for _ in range(num_trials):
            gate_s = gate_net(dummy_hidden, stressed).mean().item()
            gate_r = gate_net(dummy_hidden, relaxed).mean().item()
            stressed_gates.append(gate_s)
            relaxed_gates.append(gate_r)

    gate_diff = abs(np.mean(stressed_gates) - np.mean(relaxed_gates))

    return {
        "stressed_gate": float(np.mean(stressed_gates)),
        "relaxed_gate": float(np.mean(relaxed_gates)),
        "gate_diff": float(gate_diff),
        "stressed_std": float(np.std(stressed_gates)),
        "relaxed_std": float(np.std(relaxed_gates)),
        "sensor_response": gate_diff > 0.05,
    }


def validate_checkpoint(
    checkpoint_path: Path,
    model_name: str = "Qwen/Qwen2.5-7B-Instruct",
    device: str = "cuda",
) -> Dict:
    """Validate a single checkpoint."""
    print(f"\n[Validating] {checkpoint_path.name}", flush=True)

    # Load checkpoint
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    step = ckpt.get("step", 0)

    # Get hidden size from model config
    config = AutoConfig.from_pretrained(model_name)
    hidden_size = config.hidden_size

    # Create gate networks and load weights
    layer_results = {}
    skip_layers = [7, 11, 15, 19, 23]

    for layer_idx in skip_layers:
        # Create gate network
        gate_net = CausalAwareGateNet(
            hidden_size=hidden_size,
            sensor_dim=10,
            gate_hidden=128,
            num_gates=1,
        ).to(device).to(torch.bfloat16)  # CRITICAL: bfloat16!

        # Load weights from checkpoint
        state_dict = ckpt.get("model_state_dict", {})
        gate_prefix = f"skip_blocks.{layer_idx}.gate_net."

        gate_state = {}
        for key, value in state_dict.items():
            if gate_prefix in key:
                new_key = key.replace(f"skip_blocks.{layer_idx}.gate_net.", "")
                gate_state[new_key] = value

        if gate_state:
            gate_net.load_state_dict(gate_state, strict=False)
            print(f"  Loaded weights for layer {layer_idx}", flush=True)
        else:
            print(f"  WARNING: No weights found for layer {layer_idx}", flush=True)

        # Run causal test
        result = run_causal_loop_test(gate_net, device, torch.bfloat16)
        layer_results[layer_idx] = result

        print(f"  Layer {layer_idx}: gate_diff={result['gate_diff']:.4f} "
              f"stressed={result['stressed_gate']:.4f} relaxed={result['relaxed_gate']:.4f}", flush=True)

    # Aggregate results
    mean_gate_diff = np.mean([r["gate_diff"] for r in layer_results.values()])
    sensor_response = mean_gate_diff > 0.05

    # Convert all numpy types to native Python for JSON serialization
    per_layer_clean = {}
    for k, v in layer_results.items():
        per_layer_clean[int(k)] = {
            "stressed_gate": float(v["stressed_gate"]),
            "relaxed_gate": float(v["relaxed_gate"]),
            "gate_diff": float(v["gate_diff"]),
            "sensor_response": bool(v["sensor_response"]),
        }

    return {
        "checkpoint": checkpoint_path.name,
        "step": int(step),
        "per_layer": per_layer_clean,
        "mean_gate_diff": float(mean_gate_diff),
        "sensor_response": bool(sensor_response),
        "stressed_gate_mean": float(np.mean([r["stressed_gate"] for r in layer_results.values()])),
        "relaxed_gate_mean": float(np.mean([r["relaxed_gate"] for r in layer_results.values()])),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }


def main():
    parser = argparse.ArgumentParser(description="z31 Daedalus Validator")
    parser.add_argument("--checkpoint-dir", type=str, default="~/z31_checkpoints",
                       help="Directory containing checkpoints")
    parser.add_argument("--output", type=str, default="z31_validation_results.json",
                       help="Output JSON file")
    parser.add_argument("--model-name", type=str, default="Qwen/Qwen2.5-7B-Instruct",
                       help="Model name for config")
    parser.add_argument("--watch", action="store_true",
                       help="Watch for new checkpoints")
    parser.add_argument("--interval", type=int, default=60,
                       help="Watch interval in seconds")
    parser.add_argument("--wandb-project", type=str, default="feel-z31-causal",
                       help="W&B project name")
    parser.add_argument("--wandb-run-id", type=str, default=None,
                       help="W&B run ID to resume (for same run as trainer)")
    args = parser.parse_args()

    # Initialize wandb - connect to same project
    print("=" * 70, flush=True)
    print("FEEL z31: DAEDALUS VALIDATOR", flush=True)
    print("=" * 70, flush=True)

    wandb.init(
        project=args.wandb_project,
        name=f"z31-validator-{time.strftime('%Y%m%d_%H%M')}",
        config={
            "role": "validator",
            "machine": "daedalus",
            "checkpoint_dir": args.checkpoint_dir,
        },
        tags=["validator", "daedalus", "z31"],
    )
    print(f"W&B: {wandb.run.url}", flush=True)
    print(flush=True)

    checkpoint_dir = Path(args.checkpoint_dir).expanduser()
    if not checkpoint_dir.exists():
        print(f"Creating checkpoint directory: {checkpoint_dir}", flush=True)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

    output_path = Path(args.output)
    results = []

    # Load existing results
    if output_path.exists():
        with open(output_path) as f:
            results = json.load(f)
        validated_checkpoints = {r["checkpoint"] for r in results}
    else:
        validated_checkpoints = set()

    def validate_new():
        nonlocal results
        checkpoints = sorted(checkpoint_dir.glob("step_*.pt"))

        for ckpt_path in checkpoints:
            if ckpt_path.name in validated_checkpoints:
                continue

            try:
                result = validate_checkpoint(ckpt_path, args.model_name)
                results.append(result)
                validated_checkpoints.add(ckpt_path.name)

                # Log to wandb
                wandb.log({
                    "val/step": result["step"],
                    "val/mean_gate_diff": result["mean_gate_diff"],
                    "val/sensor_response": 1.0 if result["sensor_response"] else 0.0,
                    "val/stressed_gate": result["stressed_gate_mean"],
                    "val/relaxed_gate": result["relaxed_gate_mean"],
                    # Per-layer metrics
                    "val/gate_diff_layer7": result["per_layer"][7]["gate_diff"],
                    "val/gate_diff_layer11": result["per_layer"][11]["gate_diff"],
                    "val/gate_diff_layer15": result["per_layer"][15]["gate_diff"],
                    "val/gate_diff_layer19": result["per_layer"][19]["gate_diff"],
                    "val/gate_diff_layer23": result["per_layer"][23]["gate_diff"],
                })

                # Save after each validation
                with open(output_path, "w") as f:
                    json.dump(results, f, indent=2)

                status = "PASS ✓" if result["sensor_response"] else "FAIL ✗"
                print(f"\n[RESULT] Step {result['step']}: "
                      f"gate_diff={result['mean_gate_diff']:.4f} {status}", flush=True)

            except Exception as e:
                print(f"Error validating {ckpt_path}: {e}", flush=True)
                import traceback
                traceback.print_exc()

    if args.watch:
        print(f"[z31 Validator] Watching {checkpoint_dir}", flush=True)
        print(f"[z31 Validator] Results: {output_path}", flush=True)
        print(f"[z31 Validator] Interval: {args.interval}s", flush=True)
        print(flush=True)

        while True:
            validate_new()
            time.sleep(args.interval)
    else:
        validate_new()

    wandb.finish()
    print(f"\n[Done] Results saved to {output_path}", flush=True)


if __name__ == "__main__":
    main()
