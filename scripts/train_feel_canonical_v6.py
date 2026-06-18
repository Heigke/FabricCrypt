#!/usr/bin/env python3
"""
FEEL Canonical Training v6.0 - Predictive z_feel with Time-Split CV
====================================================================

Commit 2/3: "Predictive z_feel becomes learnable"

Key improvements over v5.1:
1. Multi-horizon prediction: 1, 5, 10 steps ahead
2. Time-split cross-validation with R² reporting
3. Shuffle/lag ablations to verify signal
4. Integration with TelemetrySampler for real hardware data

Scientific Rigor:
- Train on first 70% of sequence, evaluate R² on last 30%
- R² computed per-horizon to show predictive power decay
- Ablations: shuffle sensors → R² should collapse; lag K → R² should degrade

Usage:
    python scripts/train_feel_canonical_v6.py --epochs 30 --ablation shuffle
"""

import sys
import time
import json
import argparse
from pathlib import Path
from typing import Dict, Tuple, Optional, List
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.metrics import r2_score

sys.path.insert(0, str(Path(__file__).parent.parent))

# Import v6 sensor bank with telemetry support
from src.canonical_sensors import (
    CanonicalSensorBank, RuntimeContext, HardwareContext,
    TokenTimer, SENSOR_VERSION, SENSOR_DIM_LEGACY
)

try:
    from src.telemetry_sampler import TelemetrySampler, ValidityReport
    TELEMETRY_AVAILABLE = True
except ImportError:
    TELEMETRY_AVAILABLE = False
    TelemetrySampler = None


# Multi-horizon prediction horizons
PREDICTION_HORIZONS = [1, 5, 10]

# Time-split ratio (70% train, 30% validation for R² reporting)
TRAIN_SPLIT_RATIO = 0.7


class CanonicalFEELProjectorV6(nn.Module):
    """FEEL projector v6: supports full (16-dim) or legacy (12-dim) sensors."""

    def __init__(self, sensor_dim: int = 12, z_dim: int = 64, embed_dim: int = 1536):
        super().__init__()
        self.sensor_dim = sensor_dim
        self.z_dim = z_dim

        self.sensor_encoder = nn.Sequential(
            nn.Linear(sensor_dim, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(64, z_dim),
            nn.GELU(),
        )

        self.z_to_embed = nn.Sequential(
            nn.Linear(z_dim, 128),
            nn.GELU(),
            nn.LayerNorm(128),
            nn.Linear(128, embed_dim),
        )

        self._init_near_zero()

    def _init_near_zero(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=1e-3)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, sensors: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z_feel = self.sensor_encoder(sensors)
        feel_embed = self.z_to_embed(z_feel)
        return z_feel, feel_embed


class CanonicalFEELStreamV6(nn.Module):
    """FEEL stream v6: supports real telemetry from TelemetrySampler."""

    def __init__(self, embed_dim: int = 1536, z_dim: int = 64,
                 sensor_mode: str = "legacy", fixed_alpha: float = None):
        super().__init__()
        self.z_dim = z_dim
        self.sensor_mode = sensor_mode

        sensor_dim = 16 if sensor_mode == "full" else 12

        self.sensor_bank = CanonicalSensorBank(mode=sensor_mode)
        self.projector = CanonicalFEELProjectorV6(
            sensor_dim=sensor_dim, z_dim=z_dim, embed_dim=embed_dim
        )
        self.alpha = nn.Parameter(torch.tensor(-4.0))

        if fixed_alpha is not None:
            raw_alpha = np.log(np.exp(fixed_alpha) - 1 + 1e-8) + 4.0
            with torch.no_grad():
                self.alpha.fill_(raw_alpha)
            self.alpha.requires_grad = False

    def forward(
        self,
        logits: torch.Tensor,
        runtime: Optional[RuntimeContext] = None,
        hardware: Optional[HardwareContext] = None,
        **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        sensors = self.sensor_bank(logits, runtime=runtime, hardware=hardware, **kwargs)
        z_feel, raw_embed = self.projector(sensors)
        alpha = F.softplus(self.alpha - 4.0) + 1e-4
        feel_embed = alpha * raw_embed
        return feel_embed, sensors, z_feel, alpha

    def get_alpha(self) -> float:
        return (F.softplus(self.alpha - 4.0) + 1e-4).item()


class MultiHorizonPredictionHead(nn.Module):
    """
    Predicts future entropy at multiple horizons.

    Key for scientific rigor:
    - Separate prediction per horizon
    - Can measure R² decay with horizon distance
    """

    def __init__(self, z_dim: int = 64, horizons: List[int] = None):
        super().__init__()
        self.horizons = horizons or PREDICTION_HORIZONS

        # Separate head per horizon for cleaner gradient flow
        self.heads = nn.ModuleDict()
        for h in self.horizons:
            self.heads[f"h{h}"] = nn.Sequential(
                nn.Linear(z_dim, 64),
                nn.GELU(),
                nn.LayerNorm(64),
                nn.Linear(64, 1),
            )

    def forward(self, z_feel: torch.Tensor) -> Dict[int, torch.Tensor]:
        """Returns predictions for each horizon."""
        return {
            h: self.heads[f"h{h}"](z_feel).squeeze(-1)
            for h in self.horizons
        }


class HiddenStateAuxHead(nn.Module):
    """Predicts entropy from LM hidden state."""

    def __init__(self, hidden_dim: int = 1536, proj_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, proj_dim),
            nn.GELU(),
            nn.LayerNorm(proj_dim),
            nn.Linear(proj_dim, proj_dim),
            nn.GELU(),
            nn.Linear(proj_dim, 1),
        )

    def forward(self, h_last: torch.Tensor) -> torch.Tensor:
        return self.net(h_last).squeeze(-1)


class DifferentiableKLConstraint(nn.Module):
    """Soft KL constraint via Lagrangian multiplier."""

    def __init__(self, kl_budget: float = 0.01, lambda_init: float = 1.0):
        super().__init__()
        self.kl_budget = kl_budget
        self.log_lambda = nn.Parameter(torch.tensor(np.log(lambda_init)))

    def forward(self, kl: torch.Tensor) -> Tuple[torch.Tensor, float]:
        lambda_val = self.log_lambda.exp()
        violation = F.relu(kl - self.kl_budget)
        return lambda_val * violation ** 2, lambda_val.item()

    def update_lambda(self, avg_kl: float, lr: float = 0.1):
        with torch.no_grad():
            if avg_kl > self.kl_budget:
                self.log_lambda.data += lr
            else:
                self.log_lambda.data -= lr * 0.5
            self.log_lambda.data.clamp_(-2, 5)


TRAINING_PROMPTS = [
    "What is 2 + 2? Let me calculate:",
    "Explain the concept of recursion:",
    "Write a function to reverse a string:",
    "What causes rain to fall?",
    "Describe the scientific method:",
    "How does electricity work?",
    "What is machine learning?",
    "Explain photosynthesis briefly:",
    "What is the speed of light?",
    "How do computers process data?",
    "What is the meaning of consciousness?",
    "Describe the structure of DNA:",
    "What causes earthquakes?",
    "Explain Newton's first law:",
    "How does the internet work?",
    "What is artificial intelligence?",
]


class PredictiveTrainerV6:
    """
    v6 Trainer with predictive z_feel and time-split CV.

    Scientific improvements:
    1. Multi-horizon prediction (1, 5, 10 steps)
    2. Time-split CV: train on 70%, report R² on 30%
    3. Ablation support: shuffle, lag
    4. Real telemetry integration
    """

    def __init__(
        self,
        model_name: str = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
        kl_budget: float = 0.01,
        sensor_mode: str = "legacy",
        fixed_alpha: float = None,
        device: str = "cuda",
    ):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.kl_budget = kl_budget
        self.sensor_mode = sensor_mode

        print(f"Loading model on {self.device}...", flush=True)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32,
            trust_remote_code=True,
            device_map="auto"
        )
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

        self.embed_dim = self.model.config.hidden_size
        self.model_dtype = next(self.model.parameters()).dtype
        print(f"  Embed dim: {self.embed_dim}, dtype: {self.model_dtype}")

        # Trainable components
        self.feel_stream = CanonicalFEELStreamV6(
            embed_dim=self.embed_dim,
            sensor_mode=sensor_mode,
            fixed_alpha=fixed_alpha
        ).to(self.device).float()

        self.aux_head = HiddenStateAuxHead(
            hidden_dim=self.embed_dim
        ).to(self.device).float()

        self.multi_horizon_head = MultiHorizonPredictionHead(
            z_dim=self.feel_stream.z_dim,
            horizons=PREDICTION_HORIZONS
        ).to(self.device).float()

        self.kl_constraint = DifferentiableKLConstraint(
            kl_budget=kl_budget
        ).to(self.device)

        # Telemetry sampler
        self.telemetry_sampler = None
        if TELEMETRY_AVAILABLE:
            self.telemetry_sampler = TelemetrySampler(sample_hz=30)
            print(f"  Telemetry sampler: {self.telemetry_sampler.source}")

        print(f"  Initial alpha: {self.feel_stream.get_alpha():.6f}")
        print(f"  Sensor mode: {sensor_mode}")
        print(f"  Prediction horizons: {PREDICTION_HORIZONS}")

    def _apply_ablation(
        self,
        z_feel_history: List[torch.Tensor],
        entropy_history: List[float],
        ablation: str,
        lag: int = 0
    ) -> Tuple[List[torch.Tensor], List[float]]:
        """
        Apply ablation to test signal validity.

        ablation options:
        - "none": No ablation (normal training)
        - "shuffle": Randomly shuffle sensor readings
        - "lag": Lag sensors by K steps
        """
        if ablation == "none":
            return z_feel_history, entropy_history

        if ablation == "shuffle":
            # Shuffle z_feel tensor indices
            n = len(z_feel_history)
            indices = np.random.permutation(n)
            return [z_feel_history[i] for i in indices], entropy_history

        if ablation == "lag":
            # Lag z_feel by K steps (use z_feel[t-K] at step t)
            n = len(z_feel_history)
            lagged = []
            for i in range(n):
                src_idx = max(0, i - lag)
                lagged.append(z_feel_history[src_idx])
            return lagged, entropy_history

        return z_feel_history, entropy_history

    def _compute_r2_per_horizon(
        self,
        z_feel_history: List[torch.Tensor],
        entropy_history: List[float],
        val_start_idx: int
    ) -> Dict[int, float]:
        """
        Compute R² for each prediction horizon on validation portion.

        This is the key scientific metric:
        - R² should be positive for valid predictions
        - R² should decay with horizon distance
        - R² should collapse with shuffle ablation
        """
        r2_results = {}

        with torch.no_grad():
            for h in PREDICTION_HORIZONS:
                preds = []
                targets = []

                for t in range(val_start_idx, len(z_feel_history)):
                    if t + h < len(entropy_history):
                        z = z_feel_history[t]
                        pred = self.multi_horizon_head.heads[f"h{h}"](z).item()
                        target = entropy_history[t + h]
                        preds.append(pred)
                        targets.append(target)

                if len(preds) >= 5:  # Need enough samples for R²
                    r2 = r2_score(targets, preds)
                    r2_results[h] = r2
                else:
                    r2_results[h] = float('nan')

        return r2_results

    def train(
        self,
        epochs: int = 20,
        lr: float = 1e-3,
        n_tokens: int = 32,
        future_loss_weight: float = 0.5,
        ablation: str = "none",
        ablation_lag: int = 5
    ):
        """
        v6 Training with time-split CV and multi-horizon prediction.

        Key changes:
        1. Train on first 70% of sequence
        2. Evaluate R² on last 30% per epoch
        3. Multi-horizon prediction heads
        4. Optional ablation for falsification
        """
        if self.telemetry_sampler:
            self.telemetry_sampler.start()

        trainable_params = (
            list(self.feel_stream.parameters()) +
            list(self.aux_head.parameters()) +
            list(self.multi_horizon_head.parameters()) +
            list(self.kl_constraint.parameters())
        )
        optimizer = torch.optim.AdamW(trainable_params, lr=lr)

        history = {
            "aux_loss": [], "kl": [], "future_loss": [],
            "alpha": [], "lambda": [],
            "r2_h1": [], "r2_h5": [], "r2_h10": []
        }

        ablation_str = f", ablation={ablation}" if ablation != "none" else ""
        if ablation == "lag":
            ablation_str = f", ablation=lag({ablation_lag})"

        print(f"\nStarting v6 training (epochs={epochs}, kl_budget={self.kl_budget}{ablation_str})...")
        print("  NEW: Time-split CV with R² reporting per horizon")

        total_steps = n_tokens + max(PREDICTION_HORIZONS)

        for epoch in range(epochs):
            epoch_metrics = {
                "aux_loss": [], "kl": [], "future_loss": [], "constraint_loss": [],
                "r2_h1": [], "r2_h5": [], "r2_h10": []
            }

            for prompt in TRAINING_PROMPTS:
                input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
                current_ids = input_ids.clone()

                z_feel_history: List[torch.Tensor] = []
                entropy_history: List[float] = []
                aux_losses: List[torch.Tensor] = []
                kl_losses: List[torch.Tensor] = []
                constraint_losses: List[torch.Tensor] = []

                # Time-split index
                train_end_idx = int(total_steps * TRAIN_SPLIT_RATIO)

                for step in range(total_steps):
                    # Token timing
                    t_token_start = time.time()

                    # Base model forward
                    with torch.no_grad():
                        outputs_base = self.model(current_ids, use_cache=False)
                        logits_base = outputs_base.logits

                    t_token_end = time.time()
                    token_latency = t_token_end - t_token_start

                    # Build contexts
                    runtime = RuntimeContext(
                        token_latency=token_latency,
                        kv_cache_tokens=current_ids.shape[1],
                        generation_depth=step
                    )

                    hardware = None
                    if self.telemetry_sampler and self.sensor_mode == "full":
                        telemetry = self.telemetry_sampler.get_token_aligned(t_token_start, t_token_end)
                        hardware = HardwareContext.from_dict(telemetry)

                    # FEEL stream forward
                    feel_embed, sensors, z_feel, alpha = self.feel_stream(
                        logits_base.float(),
                        runtime=runtime,
                        hardware=hardware
                    )

                    z_feel_history.append(z_feel)

                    # Entropy target
                    with torch.no_grad():
                        probs_base = F.softmax(logits_base[:, -1, :].float(), dim=-1)
                        entropy = -(probs_base * torch.log(probs_base.clamp(min=1e-10))).sum(-1)
                        entropy_history.append(entropy.item())

                    # Only train on first 70%
                    if step < train_end_idx:
                        # Forward with FEEL
                        embeds = self.model.get_input_embeddings()(current_ids)
                        embeds = embeds + feel_embed.to(embeds.dtype).unsqueeze(1)

                        outputs_feel = self.model(
                            inputs_embeds=embeds,
                            output_hidden_states=True,
                            use_cache=False
                        )

                        # Aux loss
                        h_last = outputs_feel.hidden_states[-1][:, -1, :].float()
                        aux_pred = self.aux_head(h_last)
                        aux_loss = F.mse_loss(aux_pred, entropy)
                        aux_losses.append(aux_loss)

                        # KL constraint
                        p_base = F.softmax(logits_base[:, -1, :].float(), dim=-1)
                        p_feel = F.softmax(outputs_feel.logits[:, -1, :].float(), dim=-1)
                        kl = F.kl_div(p_feel.log(), p_base, reduction='batchmean')
                        kl_losses.append(kl)

                        constraint_loss, lambda_val = self.kl_constraint(kl)
                        constraint_losses.append(constraint_loss)

                    # Next token
                    with torch.no_grad():
                        next_token = outputs_base.logits[:, -1, :].argmax(dim=-1, keepdim=True)
                        current_ids = torch.cat([current_ids, next_token], dim=-1)


                # End of sequence: do backward for accumulated losses
                if aux_losses:
                    # Apply ablation for future prediction training
                    z_train, ent_train = self._apply_ablation(
                        z_feel_history[:train_end_idx],
                        entropy_history[:train_end_idx],
                        ablation, ablation_lag
                    )

                    # Compute future losses
                    future_losses = []
                    for t in range(len(z_train) - max(PREDICTION_HORIZONS)):
                        for h in PREDICTION_HORIZONS:
                            if t + h < len(ent_train):
                                pred = self.multi_horizon_head.heads[f"h{h}"](z_train[t]).squeeze()
                                target = torch.tensor(ent_train[t + h], device=self.device)
                                future_losses.append(F.mse_loss(pred, target))

                    # Combined loss
                    total_aux = sum(aux_losses) / len(aux_losses)
                    total_constraint = sum(constraint_losses) / len(constraint_losses)
                    total_loss = total_aux + total_constraint

                    if future_losses:
                        total_future = sum(future_losses) / len(future_losses)
                        total_loss = total_loss + future_loss_weight * total_future
                        epoch_metrics["future_loss"].append(total_future.item())

                    optimizer.zero_grad()
                    total_loss.backward()
                    torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
                    optimizer.step()

                    epoch_metrics["aux_loss"].append(total_aux.item())
                    epoch_metrics["kl"].append(sum(k.item() for k in kl_losses) / len(kl_losses))
                    epoch_metrics["constraint_loss"].append(total_constraint.item())

                # After sequence: compute R² on validation portion
                val_start_idx = train_end_idx
                r2_per_horizon = self._compute_r2_per_horizon(
                    [z.detach() for z in z_feel_history],
                    entropy_history,
                    val_start_idx
                )

                for h, r2 in r2_per_horizon.items():
                    epoch_metrics[f"r2_h{h}"].append(r2)

            # Update lambda
            avg_kl = np.mean(epoch_metrics["kl"]) if epoch_metrics["kl"] else 0
            self.kl_constraint.update_lambda(avg_kl)

            # Log
            history["aux_loss"].append(np.mean(epoch_metrics["aux_loss"]) if epoch_metrics["aux_loss"] else 0)
            history["kl"].append(avg_kl)
            history["future_loss"].append(np.mean(epoch_metrics["future_loss"]) if epoch_metrics["future_loss"] else 0)
            history["alpha"].append(self.feel_stream.get_alpha())
            history["lambda"].append(lambda_val if 'lambda_val' in dir() else 1.0)

            for h in PREDICTION_HORIZONS:
                key = f"r2_h{h}"
                vals = [v for v in epoch_metrics[key] if not np.isnan(v)]
                history[key].append(np.mean(vals) if vals else float('nan'))

            r2_str = ", ".join(f"R²(h={h})={history[f'r2_h{h}'][-1]:.3f}" for h in PREDICTION_HORIZONS)
            print(f"  Epoch {epoch+1:2d}: aux={history['aux_loss'][-1]:.4f}, "
                  f"kl={avg_kl:.6f}, {r2_str}, alpha={history['alpha'][-1]:.6f}")

        if self.telemetry_sampler:
            self.telemetry_sampler.stop()
            validity = self.telemetry_sampler.get_validity_report()
            history["telemetry_validity"] = validity.to_dict()

        return history

    def save_checkpoint(self, path: str = "results/feel_training/canonical_v6_checkpoint.pt"):
        """Save trained checkpoint."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            "feel_stream_state": self.feel_stream.state_dict(),
            "aux_head_state": self.aux_head.state_dict(),
            "multi_horizon_head_state": self.multi_horizon_head.state_dict(),
            "alpha": self.feel_stream.get_alpha(),
            "kl_budget": self.kl_budget,
            "sensor_mode": self.sensor_mode,
            "version": "v6.0",
            "horizons": PREDICTION_HORIZONS,
            "features": [
                "multi-horizon prediction (1, 5, 10)",
                "time-split CV with R² reporting",
                "shuffle/lag ablations",
                "telemetry integration",
            ],
        }

        torch.save(checkpoint, path)
        print(f"\nCheckpoint saved: {path}")
        print(f"  Alpha: {checkpoint['alpha']:.6f}")
        print(f"  Version: {checkpoint['version']}")

        return checkpoint


def run_ablation_study(trainer: PredictiveTrainerV6, epochs: int = 10, n_tokens: int = 32):
    """
    Run ablation study to verify signal validity.

    Scientific test:
    - Normal: R² should be positive
    - Shuffle: R² should collapse (near 0 or negative)
    - Lag: R² should degrade with lag distance
    """
    print("\n" + "=" * 70)
    print("  ABLATION STUDY: Verifying Signal Validity")
    print("=" * 70)

    results = {}

    # Normal training
    print("\n[1/4] Normal training (no ablation)...")
    h_normal = trainer.train(epochs=epochs, n_tokens=n_tokens, ablation="none")
    results["normal"] = {
        "r2_h1": h_normal["r2_h1"][-1] if h_normal["r2_h1"] else 0,
        "r2_h5": h_normal["r2_h5"][-1] if h_normal["r2_h5"] else 0,
        "r2_h10": h_normal["r2_h10"][-1] if h_normal["r2_h10"] else 0,
    }

    # Re-init weights for fair comparison
    trainer.multi_horizon_head = MultiHorizonPredictionHead(
        z_dim=trainer.feel_stream.z_dim, horizons=PREDICTION_HORIZONS
    ).to(trainer.device).float()

    print("\n[2/4] Shuffle ablation (should collapse R²)...")
    h_shuffle = trainer.train(epochs=epochs, n_tokens=n_tokens, ablation="shuffle")
    results["shuffle"] = {
        "r2_h1": h_shuffle["r2_h1"][-1] if h_shuffle["r2_h1"] else 0,
        "r2_h5": h_shuffle["r2_h5"][-1] if h_shuffle["r2_h5"] else 0,
        "r2_h10": h_shuffle["r2_h10"][-1] if h_shuffle["r2_h10"] else 0,
    }

    # Re-init
    trainer.multi_horizon_head = MultiHorizonPredictionHead(
        z_dim=trainer.feel_stream.z_dim, horizons=PREDICTION_HORIZONS
    ).to(trainer.device).float()

    print("\n[3/4] Lag-3 ablation (should degrade R²)...")
    h_lag3 = trainer.train(epochs=epochs, n_tokens=n_tokens, ablation="lag", ablation_lag=3)
    results["lag_3"] = {
        "r2_h1": h_lag3["r2_h1"][-1] if h_lag3["r2_h1"] else 0,
        "r2_h5": h_lag3["r2_h5"][-1] if h_lag3["r2_h5"] else 0,
        "r2_h10": h_lag3["r2_h10"][-1] if h_lag3["r2_h10"] else 0,
    }

    # Re-init
    trainer.multi_horizon_head = MultiHorizonPredictionHead(
        z_dim=trainer.feel_stream.z_dim, horizons=PREDICTION_HORIZONS
    ).to(trainer.device).float()

    print("\n[4/4] Lag-10 ablation (should severely degrade R²)...")
    h_lag10 = trainer.train(epochs=epochs, n_tokens=n_tokens, ablation="lag", ablation_lag=10)
    results["lag_10"] = {
        "r2_h1": h_lag10["r2_h1"][-1] if h_lag10["r2_h1"] else 0,
        "r2_h5": h_lag10["r2_h5"][-1] if h_lag10["r2_h5"] else 0,
        "r2_h10": h_lag10["r2_h10"][-1] if h_lag10["r2_h10"] else 0,
    }

    # Summary
    print("\n" + "=" * 70)
    print("  ABLATION STUDY RESULTS")
    print("=" * 70)
    print(f"\n{'Condition':<15} {'R²(h=1)':<12} {'R²(h=5)':<12} {'R²(h=10)':<12}")
    print("-" * 50)
    for cond, r2s in results.items():
        print(f"{cond:<15} {r2s['r2_h1']:<12.3f} {r2s['r2_h5']:<12.3f} {r2s['r2_h10']:<12.3f}")

    # Validity checks
    print("\n  Signal Validity Checks:")

    # Shuffle should collapse R²
    shuffle_collapsed = (
        results["shuffle"]["r2_h1"] < results["normal"]["r2_h1"] * 0.5
    )
    print(f"    Shuffle collapses R²: {'PASS' if shuffle_collapsed else 'FAIL'}")

    # Lag should degrade R²
    lag_degrades = (
        results["lag_3"]["r2_h1"] < results["normal"]["r2_h1"] and
        results["lag_10"]["r2_h1"] < results["lag_3"]["r2_h1"]
    )
    print(f"    Lag degrades R²: {'PASS' if lag_degrades else 'FAIL'}")

    return results


def main():
    parser = argparse.ArgumentParser(description="FEEL Canonical Training v6.0")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--kl-budget", type=float, default=0.01)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--n-tokens", type=int, default=32)
    parser.add_argument("--future-weight", type=float, default=0.5)
    parser.add_argument("--sensor-mode", type=str, default="legacy",
                       choices=["legacy", "full"])
    parser.add_argument("--fixed-alpha", type=float, default=None)
    parser.add_argument("--ablation", type=str, default="none",
                       choices=["none", "shuffle", "lag"])
    parser.add_argument("--ablation-lag", type=int, default=5)
    parser.add_argument("--run-ablation-study", action="store_true",
                       help="Run full ablation study (shuffle + lag)")
    parser.add_argument("--model", type=str, default="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
    args = parser.parse_args()

    print("=" * 70)
    print("  FEEL CANONICAL TRAINING v6.0 - PREDICTIVE z_feel")
    print("=" * 70)
    print()
    print("NEW FEATURES:")
    print("  1. Multi-horizon prediction: 1, 5, 10 steps")
    print("  2. Time-split CV: Train 70%, R² on 30%")
    print("  3. Ablation support: shuffle, lag")
    print("  4. Telemetry integration for hardware sensors")
    print()

    trainer = PredictiveTrainerV6(
        model_name=args.model,
        kl_budget=args.kl_budget,
        sensor_mode=args.sensor_mode,
        fixed_alpha=args.fixed_alpha,
    )

    if args.run_ablation_study:
        ablation_results = run_ablation_study(trainer, epochs=args.epochs, n_tokens=args.n_tokens)

        # Save ablation results
        ablation_path = "results/feel_training/v6_ablation_study.json"
        Path(ablation_path).parent.mkdir(parents=True, exist_ok=True)
        with open(ablation_path, 'w') as f:
            json.dump(ablation_results, f, indent=2)
        print(f"\nAblation results saved: {ablation_path}")
    else:
        history = trainer.train(
            epochs=args.epochs,
            lr=args.lr,
            n_tokens=args.n_tokens,
            future_loss_weight=args.future_weight,
            ablation=args.ablation,
            ablation_lag=args.ablation_lag,
        )

        trainer.save_checkpoint()

        # Save history
        history_path = "results/feel_training/v6_training_history.json"
        Path(history_path).parent.mkdir(parents=True, exist_ok=True)

        # Convert numpy types for JSON serialization
        def convert_for_json(obj):
            if isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.bool_):
                return bool(obj)
            elif isinstance(obj, dict):
                return {k: convert_for_json(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_for_json(x) for x in obj]
            return obj

        history_json = convert_for_json(history)

        with open(history_path, 'w') as f:
            json.dump(history_json, f, indent=2)
        print(f"\nTraining history saved: {history_path}")

        print("\n" + "=" * 70)
        print("  TRAINING COMPLETE")
        print("=" * 70)
        print(f"  Final alpha: {history['alpha'][-1]:.6f}")
        print(f"  Final aux loss: {history['aux_loss'][-1]:.4f}")
        print(f"  Final future loss: {history['future_loss'][-1]:.4f}")
        print(f"  Final KL: {history['kl'][-1]:.6f}")
        print(f"\n  R² (time-split CV):")
        for h in PREDICTION_HORIZONS:
            print(f"    h={h}: {history[f'r2_h{h}'][-1]:.3f}")


if __name__ == "__main__":
    main()
