#!/usr/bin/env python3
"""
FEEL z27 Remote Validator - Runs on daedalus (z2)

Connects to the SAME W&B run as the main trainer on ikaros (z1).
Continuously pulls checkpoints and runs comprehensive validation.

Usage on daedalus:
  source venvs/torch-rocm/bin/activate
  python z27_remote_validator.py --run-id <wandb_run_id> --checkpoint-host ikaros@192.168.0.XX

This enables:
- More validation samples (z2 has 103GB memory)
- ROCprofiler analysis
- Continuous ablation without slowing training
"""

import os
import sys
import time
import json
import argparse
import subprocess
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np

import torch
import torch.nn as nn
import wandb
from transformers import AutoTokenizer, AutoModelForCausalLM


# =============================================================================
# DUMMY SENSOR HUB (for validation on different machine)
# =============================================================================

class DummySensorHub:
    """
    Fake sensor hub for validation-only mode.
    Returns synthetic sensor values for testing causality.
    """

    def __init__(self, device: str = "cuda", mode: str = "normal"):
        self.device = device
        self.mode = mode
        self._cached_tensor = None

    def read_tensor(self) -> torch.Tensor:
        if self.mode == "normal":
            # Normal operation sensors
            state = np.array([0.3, 0.5, 0.3, 0.5, 0.0, 0.2, 0.1, 0.3], dtype=np.float32)
        elif self.mode == "stress":
            # High stress sensors
            state = np.array([0.8, 0.9, 0.8, 1.0, 1.0, 0.8, 0.3, 0.9], dtype=np.float32)
        elif self.mode == "random":
            # Random sensors (for shuffle ablation)
            state = np.random.rand(8).astype(np.float32)
        else:
            state = np.array([0.5] * 8, dtype=np.float32)

        return torch.from_numpy(state).to(self.device)

    def set_mode(self, mode: str):
        self.mode = mode


# =============================================================================
# STE SKIP BLOCK (same as trainer, for loading checkpoints)
# =============================================================================

class STESkipFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, gates, threshold):
        ctx.save_for_backward(gates)
        return (gates >= threshold).float()

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None


class MLPSkipBlockSTE(nn.Module):
    def __init__(
        self,
        original_layer: nn.Module,
        hidden_size: int,
        sensor_hub,
        sensor_dim: int = 8,
        skip_threshold: float = 0.5,
    ):
        super().__init__()
        self.original_layer = original_layer
        self.hidden_size = hidden_size
        self.sensor_hub = sensor_hub
        self.skip_threshold = skip_threshold

        self.gate_net = nn.Sequential(
            nn.Linear(hidden_size + sensor_dim, 256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, 64),
            nn.GELU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

        self.film_gamma = nn.Linear(sensor_dim, hidden_size)
        self.film_beta = nn.Linear(sensor_dim, hidden_size)
        self.strain_embed = nn.Parameter(torch.randn(hidden_size) * 0.01)

        self._last_gates = []
        self._last_skip_decisions = []

    def forward(self, hidden_states):
        batch, seq_len, _ = hidden_states.shape
        sensors = self.sensor_hub.read_tensor().to(dtype=hidden_states.dtype)

        last_hidden = hidden_states[:, -1, :]
        sensors_expanded = sensors.unsqueeze(0).expand(batch, -1)
        gate_input = torch.cat([last_hidden, sensors_expanded], dim=-1)

        gates = self.gate_net(gate_input).squeeze(-1)
        self._last_gates = gates.detach().cpu().tolist()

        hard_decisions = STESkipFunction.apply(gates, self.skip_threshold)
        self._last_skip_decisions = (hard_decisions < 0.5).tolist()

        gamma = 1.0 + self.film_gamma(sensors)
        beta = self.film_beta(sensors)

        run_mask = hard_decisions > 0.5
        skip_mask = ~run_mask

        output = hidden_states.clone()

        if run_mask.any():
            run_idx = run_mask.nonzero(as_tuple=True)[0]
            layer_out = self.original_layer(hidden_states[run_idx])
            modulated = gamma.view(1, 1, -1) * layer_out + beta.view(1, 1, -1)
            output[run_idx] = modulated

        if skip_mask.any():
            skip_idx = skip_mask.nonzero(as_tuple=True)[0]
            output[skip_idx] = hidden_states[skip_idx] + self.strain_embed.view(1, 1, -1)

        return output

    @property
    def gate_mean(self):
        return np.mean(self._last_gates) if self._last_gates else 0.5


# =============================================================================
# VALIDATION MODEL WRAPPER
# =============================================================================

class ValidationModel(nn.Module):
    """Wrapper for validation that can load checkpoints from trainer."""

    def __init__(
        self,
        base_model: nn.Module,
        sensor_hub,
        skip_layers: List[int] = [7, 11, 15, 19, 23],
        skip_threshold: float = 0.5,
    ):
        super().__init__()
        self.base_model = base_model
        self.sensor_hub = sensor_hub
        self.skip_threshold = skip_threshold

        config = base_model.config
        hidden_size = config.hidden_size

        self.skip_blocks = nn.ModuleDict()
        layers = base_model.model.layers

        for layer_idx in skip_layers:
            if layer_idx < len(layers):
                original_mlp = layers[layer_idx].mlp
                mlp_param = next(original_mlp.parameters())
                mlp_device = mlp_param.device
                mlp_dtype = mlp_param.dtype

                skip_block = MLPSkipBlockSTE(
                    original_layer=original_mlp,
                    hidden_size=hidden_size,
                    sensor_hub=sensor_hub,
                    skip_threshold=skip_threshold,
                ).to(device=mlp_device, dtype=mlp_dtype)

                self.skip_blocks[str(layer_idx)] = skip_block
                layers[layer_idx].mlp = skip_block

        for param in self.base_model.parameters():
            param.requires_grad = False

    def load_checkpoint(self, checkpoint_path: str):
        """Load checkpoint from trainer."""
        ckpt = torch.load(checkpoint_path, map_location="cuda")
        state_dict = ckpt.get("model_state_dict", ckpt)

        # Load only skip block parameters
        current_state = self.state_dict()
        for k, v in state_dict.items():
            if k in current_state:
                current_state[k] = v

        self.load_state_dict(current_state, strict=False)
        print(f"[ValidationModel] Loaded checkpoint: {checkpoint_path}")

        if "threshold" in ckpt:
            self.skip_threshold = ckpt["threshold"]
            for block in self.skip_blocks.values():
                block.skip_threshold = self.skip_threshold
            print(f"[ValidationModel] Threshold: {self.skip_threshold}")

    def forward(self, input_ids, **kwargs):
        return self.base_model(input_ids, **kwargs)

    def get_stats(self):
        all_gates = []
        skip_rates = []
        for block in self.skip_blocks.values():
            if block._last_gates:
                all_gates.extend(block._last_gates)
            if block._last_skip_decisions:
                skip_rates.append(sum(block._last_skip_decisions) / len(block._last_skip_decisions))
        return {
            "gate_mean": np.mean(all_gates) if all_gates else 0.5,
            "skip_rate": np.mean(skip_rates) if skip_rates else 0.0,
        }


# =============================================================================
# COMPREHENSIVE VALIDATION
# =============================================================================

def generate_greedy(model, tokenizer, prompt: str, max_tokens: int = 64):
    """Greedy generation for deterministic validation."""
    device = next(model.parameters()).device
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model.base_model.generate(
            inputs.input_ids,
            max_new_tokens=max_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )

    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    stats = model.get_stats()
    return text, stats


def run_comprehensive_validation(
    model: ValidationModel,
    tokenizer,
    prompts: List[str],
    num_samples: int = 512,
) -> Dict:
    """
    Comprehensive validation with more samples for stable metrics.

    This runs on z2 (daedalus) with 103GB memory.
    """
    model.eval()
    results = {}

    samples_per_prompt = max(1, num_samples // len(prompts))

    for mode in ["full", "shuffle", "frozen"]:
        mode_lens = []
        mode_gates = []
        mode_skips = []

        # Set sensor mode
        if mode == "shuffle":
            model.sensor_hub.set_mode("random")
        elif mode == "frozen":
            model.sensor_hub.set_mode("normal")
        else:
            model.sensor_hub.set_mode("normal")

        for prompt in prompts[:min(64, len(prompts))]:
            for _ in range(samples_per_prompt):
                if mode == "shuffle":
                    model.sensor_hub.set_mode("random")  # Re-randomize each sample

                text, stats = generate_greedy(model, tokenizer, prompt, max_tokens=64)

                mode_lens.append(len(text))
                mode_gates.append(stats["gate_mean"])
                mode_skips.append(stats["skip_rate"])

        results[mode] = {
            "avg_len": np.mean(mode_lens),
            "gate_mean": np.mean(mode_gates),
            "gate_std": np.std(mode_gates),
            "skip_rate": np.mean(mode_skips),
        }

    # Causal score: does full model behave differently from shuffle?
    gate_diff = results["full"]["gate_mean"] - results["shuffle"]["gate_mean"]
    skip_diff = results["full"]["skip_rate"] - results["shuffle"]["skip_rate"]

    results["causal_score"] = {
        "gate_diff": gate_diff,
        "skip_diff": skip_diff,
        "combined": gate_diff + skip_diff,
    }

    model.sensor_hub.set_mode("normal")
    return results


def run_expression_test(
    model: ValidationModel,
    tokenizer,
    num_samples: int = 128,
) -> Dict:
    """
    Expression test: does gate respond to sensor stress?
    """
    model.eval()
    results = {}

    for stress_level in ["low", "high"]:
        if stress_level == "low":
            model.sensor_hub.set_mode("normal")
        else:
            model.sensor_hub.set_mode("stress")

        gates = []
        skips = []

        prompts = [
            "Explain quantum computing.",
            "Write a poem about nature.",
            "Describe machine learning.",
        ]

        for _ in range(num_samples // len(prompts)):
            for prompt in prompts:
                _, stats = generate_greedy(model, tokenizer, prompt, max_tokens=32)
                gates.append(stats["gate_mean"])
                skips.append(stats["skip_rate"])

        results[stress_level] = {
            "gate_mean": np.mean(gates),
            "gate_std": np.std(gates),
            "skip_rate": np.mean(skips),
        }

    # Expression delta
    results["delta"] = {
        "gate": results["high"]["gate_mean"] - results["low"]["gate_mean"],
        "skip": results["high"]["skip_rate"] - results["low"]["skip_rate"],
    }

    model.sensor_hub.set_mode("normal")
    return results


# =============================================================================
# CHECKPOINT SYNC
# =============================================================================

def sync_checkpoint(host: str, remote_path: str, local_path: str, password: str) -> bool:
    """Pull latest checkpoint from training machine."""
    try:
        cmd = f"sshpass -p '{password}' scp -o StrictHostKeyChecking=no {host}:{remote_path} {local_path}"
        result = subprocess.run(cmd, shell=True, capture_output=True, timeout=30)
        return result.returncode == 0
    except Exception as e:
        print(f"[Sync] Error: {e}")
        return False


def get_latest_checkpoint(host: str, remote_dir: str, password: str) -> Optional[str]:
    """Find latest checkpoint on training machine."""
    try:
        cmd = f"sshpass -p '{password}' ssh -o StrictHostKeyChecking=no {host} 'ls -t {remote_dir}/*.pt 2>/dev/null | head -1'"
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=10)
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip()
    except Exception as e:
        print(f"[Sync] Error finding checkpoint: {e}")
    return None


# =============================================================================
# MAIN VALIDATION LOOP
# =============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", type=str, required=True, help="W&B run ID to join")
    parser.add_argument("--run-path", type=str, default="bergvall-eric/feel-z27-ste", help="W&B project path")
    parser.add_argument("--checkpoint-host", type=str, default="ikaros@192.168.0.1", help="Training machine")
    parser.add_argument("--checkpoint-dir", type=str, default="/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy/models/grpo_z27")
    parser.add_argument("--password", type=str, default="", help="SSH password")
    parser.add_argument("--val-interval", type=int, default=120, help="Validation interval (seconds)")
    parser.add_argument("--val-samples", type=int, default=512, help="Samples per validation")
    parser.add_argument("--base-model", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    args = parser.parse_args()

    # Join existing W&B run
    print(f"[Validator] Joining W&B run: {args.run_id}")
    wandb.init(
        project="feel-z27-ste",
        id=args.run_id,
        resume="allow",
    )

    print(f"[Validator] W&B: {wandb.run.url}")

    # Create local checkpoint dir
    local_ckpt_dir = Path("/tmp/z27_checkpoints")
    local_ckpt_dir.mkdir(exist_ok=True)

    # Initialize dummy sensor hub
    sensor_hub = DummySensorHub(device="cuda", mode="normal")

    # Load model
    print("[Validator] Loading base model...")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    model = ValidationModel(
        base_model=base_model,
        sensor_hub=sensor_hub,
        skip_layers=[7, 11, 15, 19, 23],
    )

    # Load prompts
    prompts = [
        "Explain the concept of machine learning.",
        "Write a short story about discovery.",
        "Describe how neural networks work.",
        "What is the meaning of consciousness?",
        "Explain quantum computing simply.",
    ] * 20

    print(f"[Validator] Ready. Monitoring checkpoints every {args.val_interval}s")
    print()

    last_checkpoint = None
    validation_count = 0

    while True:
        try:
            # Find latest checkpoint
            latest = get_latest_checkpoint(
                args.checkpoint_host,
                args.checkpoint_dir,
                args.password,
            )

            if latest and latest != last_checkpoint:
                print(f"[Validator] New checkpoint: {latest}")

                # Sync checkpoint
                local_path = local_ckpt_dir / Path(latest).name
                success = sync_checkpoint(
                    args.checkpoint_host,
                    latest,
                    str(local_path),
                    args.password,
                )

                if success:
                    # Load and validate
                    model.load_checkpoint(str(local_path))
                    validation_count += 1

                    print(f"\n[Validation #{validation_count}]")

                    # Comprehensive validation
                    val_results = run_comprehensive_validation(
                        model, tokenizer, prompts,
                        num_samples=args.val_samples,
                    )

                    print("  Ablation Results:")
                    for mode in ["full", "shuffle", "frozen"]:
                        r = val_results[mode]
                        print(f"    {mode:8s}: gate={r['gate_mean']:.3f}±{r['gate_std']:.3f} skip={r['skip_rate']*100:.1f}%")
                    print(f"  Causal Score: gate_diff={val_results['causal_score']['gate_diff']:.4f}")

                    # Expression test
                    expr_results = run_expression_test(
                        model, tokenizer,
                        num_samples=128,
                    )

                    print("  Expression Test:")
                    print(f"    low_stress:  gate={expr_results['low']['gate_mean']:.3f}")
                    print(f"    high_stress: gate={expr_results['high']['gate_mean']:.3f}")
                    print(f"    delta: {expr_results['delta']['gate']:.4f}")

                    # Log to W&B (same run!)
                    wandb.log({
                        "z2_val/count": validation_count,
                        "z2_val/full_gate": val_results["full"]["gate_mean"],
                        "z2_val/shuffle_gate": val_results["shuffle"]["gate_mean"],
                        "z2_val/causal_gate_diff": val_results["causal_score"]["gate_diff"],
                        "z2_val/full_skip": val_results["full"]["skip_rate"],
                        "z2_val/shuffle_skip": val_results["shuffle"]["skip_rate"],
                        "z2_val/expr_low_gate": expr_results["low"]["gate_mean"],
                        "z2_val/expr_high_gate": expr_results["high"]["gate_mean"],
                        "z2_val/expr_delta": expr_results["delta"]["gate"],
                    })

                    last_checkpoint = latest
                    print()

            time.sleep(args.val_interval)

        except KeyboardInterrupt:
            print("\n[Validator] Stopped.")
            break
        except Exception as e:
            print(f"[Validator] Error: {e}")
            time.sleep(30)

    wandb.finish()


if __name__ == "__main__":
    main()
