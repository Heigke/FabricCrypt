#!/usr/bin/env python3
"""
FEEL Canonical Training v5.1 - Fixed Predictive Loss
=====================================================

v5.1 FIXES the .detach() issue from v5.0:
- v5.0: future_head(z_feel.detach()) → z_feel doesn't learn from prediction
- v5.1: future_head(z_feel) → z_feel learns predictive representations

Key change: Combined loss with gradient flow through z_feel to future prediction.

Training Objective:
    L = L_aux + lambda * max(0, KL - budget)^2 + L_future

Where L_future now backprops through z_feel!

Usage:
    python scripts/train_feel_canonical_v5_1.py --epochs 30 --kl-budget 0.01
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

sys.path.insert(0, str(Path(__file__).parent.parent))

# Import canonical sensors from unified module
try:
    from src.canonical_sensors import CanonicalSensorBank, SENSOR_VERSION
    print(f"Using unified sensor bank: {SENSOR_VERSION}")
except ImportError:
    SENSOR_VERSION = "v5.1.0-inline"

    class CanonicalSensorBank(nn.Module):
        """12-dimensional sensor bank (inline fallback)."""
        SENSOR_DIM = 12

        def __init__(self):
            super().__init__()

        def forward(
            self,
            logits: torch.Tensor,
            chosen_token_id: Optional[int] = None,
            kv_cache_tokens: int = 0,
            generation_depth: int = 0,
        ) -> torch.Tensor:
            device = logits.device
            logits_f32 = logits[:, -1, :].float()
            probs = F.softmax(logits_f32, dim=-1)
            log_probs = F.log_softmax(logits_f32, dim=-1)

            vocab_size = logits.shape[-1]
            max_entropy = np.log(vocab_size)
            entropy = -(probs * log_probs).sum(-1)
            entropy_norm = (entropy / max_entropy).clamp(0, 1)

            top2 = probs.topk(2, dim=-1).values
            logit_margin = (top2[:, 0] - top2[:, 1]).clamp(0, 1)
            top_k_mass = probs.topk(5, dim=-1).values.sum(-1).clamp(0, 1)
            uncertainty = (1 - probs.max(dim=-1).values).clamp(0, 1)

            tps_norm = torch.ones(logits.shape[0], device=device) * 0.5
            latency_norm = torch.ones(logits.shape[0], device=device) * 0.3
            kv_cache_norm = torch.full((logits.shape[0],), min(kv_cache_tokens / 4096.0, 1.0), device=device)

            if chosen_token_id is not None:
                surprisal = -log_probs[0, chosen_token_id]
                surprisal_norm = (surprisal / 15.0).clamp(0, 1).unsqueeze(0)
            else:
                surprisal_norm = entropy_norm

            attn_entropy_norm = entropy_norm * 0.8 + 0.1
            residual_norm = (logits_f32.std(dim=-1) / 10.0).clamp(0, 1)
            stress = ((entropy_norm * 0.5) + (uncertainty * 0.3) + ((1 - logit_margin) * 0.2)).clamp(0, 1)
            depth_norm = torch.full((logits.shape[0],), min(generation_depth / 256.0, 1.0), device=device)

            sensors = torch.stack([
                entropy_norm.squeeze(), logit_margin.squeeze(), top_k_mass.squeeze(), uncertainty.squeeze(),
                tps_norm.squeeze(), latency_norm.squeeze(), kv_cache_norm.squeeze(), surprisal_norm.squeeze(),
                attn_entropy_norm.squeeze(), residual_norm.squeeze(), stress.squeeze(), depth_norm.squeeze()
            ], dim=-1)

            if sensors.dim() == 1:
                sensors = sensors.unsqueeze(0)
            return sensors


class CanonicalFEELProjector(nn.Module):
    """FEEL projector: 12-dim sensors -> 64-dim z_feel -> embed_dim."""

    def __init__(self, sensor_dim: int = 12, z_dim: int = 64, embed_dim: int = 1536):
        super().__init__()

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


class CanonicalFEELStream(nn.Module):
    """Unified FEEL stream with trainable alpha."""

    def __init__(self, embed_dim: int = 1536, z_dim: int = 64, fixed_alpha: float = None):
        super().__init__()
        self.z_dim = z_dim
        self.sensor_bank = CanonicalSensorBank()
        self.projector = CanonicalFEELProjector(sensor_dim=12, z_dim=z_dim, embed_dim=embed_dim)
        self.alpha = nn.Parameter(torch.tensor(-4.0))

        if fixed_alpha is not None:
            raw_alpha = np.log(np.exp(fixed_alpha) - 1 + 1e-8) + 4.0
            with torch.no_grad():
                self.alpha.fill_(raw_alpha)
            self.alpha.requires_grad = False

    def forward(self, logits: torch.Tensor, **kwargs) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        sensors = self.sensor_bank(logits, **kwargs)
        z_feel, raw_embed = self.projector(sensors)
        alpha = F.softplus(self.alpha - 4.0) + 1e-4
        feel_embed = alpha * raw_embed
        return feel_embed, sensors, z_feel, alpha

    def get_alpha(self) -> float:
        return (F.softplus(self.alpha - 4.0) + 1e-4).item()


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


class FuturePredictionHead(nn.Module):
    """Predicts future entropy from z_feel (leak-free)."""

    def __init__(self, z_dim: int = 64, horizon: int = 4):
        super().__init__()
        self.horizon = horizon
        self.net = nn.Sequential(
            nn.Linear(z_dim, 64),
            nn.GELU(),
            nn.LayerNorm(64),
            nn.Linear(64, horizon),
        )

    def forward(self, z_feel: torch.Tensor) -> torch.Tensor:
        return self.net(z_feel)


class DifferentiableKLConstraint(nn.Module):
    """Soft KL constraint via Lagrangian multiplier."""

    def __init__(self, kl_budget: float = 0.01, lambda_init: float = 1.0, lambda_lr: float = 0.1):
        super().__init__()
        self.kl_budget = kl_budget
        self.log_lambda = nn.Parameter(torch.tensor(np.log(lambda_init)))
        self.lambda_lr = lambda_lr

    def forward(self, kl: torch.Tensor) -> Tuple[torch.Tensor, float]:
        lambda_val = self.log_lambda.exp()
        violation = F.relu(kl - self.kl_budget)
        constraint_loss = lambda_val * violation ** 2
        return constraint_loss, lambda_val.item()

    def update_lambda(self, avg_kl: float):
        with torch.no_grad():
            if avg_kl > self.kl_budget:
                self.log_lambda.data += self.lambda_lr
            else:
                self.log_lambda.data -= self.lambda_lr * 0.5
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


class CanonicalFEELTrainerV51:
    """
    v5.1 Trainer - Fixed predictive loss gradient flow.

    Key difference from v5.0:
    - Accumulates all losses over sequence steps
    - Does combined backward() with future loss
    - z_feel learns from future prediction (no .detach())
    """

    def __init__(
        self,
        model_name: str = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
        kl_budget: float = 0.01,
        fixed_alpha: float = None,
        device: str = "cuda",
    ):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.kl_budget = kl_budget

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
        self.feel_stream = CanonicalFEELStream(
            embed_dim=self.embed_dim,
            fixed_alpha=fixed_alpha
        ).to(self.device).float()

        self.aux_head = HiddenStateAuxHead(
            hidden_dim=self.embed_dim
        ).to(self.device).float()

        self.future_head = FuturePredictionHead(
            z_dim=self.feel_stream.z_dim,
            horizon=4
        ).to(self.device).float()

        self.kl_constraint = DifferentiableKLConstraint(
            kl_budget=kl_budget
        ).to(self.device)

        print(f"  Initial alpha: {self.feel_stream.get_alpha():.6f}")

    def train(self, epochs: int = 20, lr: float = 1e-3, n_tokens: int = 32,
              future_loss_weight: float = 0.5):
        """
        v5.1 Training with fixed gradient flow.

        Key changes:
        1. Accumulate losses over chunks of steps (not per-step backward)
        2. future_loss computed WITHOUT .detach() - z_feel learns predictive representations
        3. Combined loss = aux_loss + kl_constraint + future_loss_weight * future_loss
        """

        trainable_params = (
            list(self.feel_stream.parameters()) +
            list(self.aux_head.parameters()) +
            list(self.future_head.parameters()) +
            list(self.kl_constraint.parameters())
        )
        optimizer = torch.optim.AdamW(trainable_params, lr=lr)

        history = {"aux_loss": [], "kl": [], "future_loss": [], "alpha": [], "lambda": []}

        print(f"\nStarting v5.1 training (epochs={epochs}, kl_budget={self.kl_budget}, "
              f"future_weight={future_loss_weight})...")
        print("  KEY FIX: z_feel now learns from future prediction (no .detach())")

        chunk_size = 8  # Accumulate losses over chunks to manage memory

        for epoch in range(epochs):
            epoch_metrics = {"aux_loss": [], "kl": [], "future_loss": [], "constraint_loss": []}

            for prompt in TRAINING_PROMPTS:
                input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
                current_ids = input_ids.clone()

                # Histories for the sequence
                z_feel_history: List[torch.Tensor] = []  # Keep computation graph!
                entropy_history: List[float] = []
                aux_losses: List[torch.Tensor] = []
                kl_losses: List[torch.Tensor] = []
                constraint_losses: List[torch.Tensor] = []

                total_steps = n_tokens + 4

                for step in range(total_steps):
                    # Base model forward (no grad)
                    with torch.no_grad():
                        outputs_base = self.model(current_ids, use_cache=False)
                        logits_base = outputs_base.logits

                    # FEEL stream forward (WITH grad)
                    feel_embed, sensors, z_feel, alpha = self.feel_stream(
                        logits_base.float(),
                        kv_cache_tokens=current_ids.shape[1],
                        generation_depth=step
                    )

                    z_feel_history.append(z_feel)  # NO .detach() - keep graph!

                    # Entropy target
                    with torch.no_grad():
                        probs_base = F.softmax(logits_base[:, -1, :].float(), dim=-1)
                        entropy = -(probs_base * torch.log(probs_base.clamp(min=1e-10))).sum(-1)
                        entropy_history.append(entropy.item())

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
                        next_token = outputs_feel.logits[:, -1, :].argmax(dim=-1, keepdim=True)
                        current_ids = torch.cat([current_ids, next_token], dim=-1)

                    # Backward in chunks to manage memory
                    if (step + 1) % chunk_size == 0 or step == total_steps - 1:
                        # Compute future losses for this chunk
                        # For z_feel at step t, predict entropy at t+1, t+2, t+3, t+4
                        future_losses = []

                        start_idx = max(0, len(z_feel_history) - chunk_size - 4)
                        end_idx = len(z_feel_history) - 4

                        for t in range(start_idx, end_idx):
                            if t + 4 < len(entropy_history):
                                # KEY FIX: No .detach() here!
                                future_pred = self.future_head(z_feel_history[t])
                                future_targets = torch.tensor(
                                    entropy_history[t+1:t+5],
                                    device=self.device
                                )
                                future_loss = F.mse_loss(future_pred.squeeze(), future_targets)
                                future_losses.append(future_loss)

                        # Combined loss for this chunk
                        chunk_start = max(0, len(aux_losses) - chunk_size)
                        total_aux = sum(aux_losses[chunk_start:]) / len(aux_losses[chunk_start:])
                        total_constraint = sum(constraint_losses[chunk_start:]) / len(constraint_losses[chunk_start:])

                        total_loss = total_aux + total_constraint

                        if future_losses:
                            total_future = sum(future_losses) / len(future_losses)
                            total_loss = total_loss + future_loss_weight * total_future
                            epoch_metrics["future_loss"].append(total_future.item())

                        # Backward
                        optimizer.zero_grad()
                        total_loss.backward()
                        torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
                        optimizer.step()

                        # Log metrics
                        epoch_metrics["aux_loss"].append(total_aux.item())
                        epoch_metrics["kl"].append(sum(k.item() for k in kl_losses[chunk_start:]) / len(kl_losses[chunk_start:]))
                        epoch_metrics["constraint_loss"].append(total_constraint.item())

                        # Detach old z_feel tensors to free memory
                        for i in range(len(z_feel_history) - 4):
                            z_feel_history[i] = z_feel_history[i].detach()

            # Update lambda based on average KL
            avg_kl = np.mean(epoch_metrics["kl"])
            self.kl_constraint.update_lambda(avg_kl)

            # Log
            history["aux_loss"].append(np.mean(epoch_metrics["aux_loss"]))
            history["kl"].append(avg_kl)
            history["future_loss"].append(np.mean(epoch_metrics["future_loss"]) if epoch_metrics["future_loss"] else 0)
            history["alpha"].append(self.feel_stream.get_alpha())
            history["lambda"].append(lambda_val)

            print(f"  Epoch {epoch+1:2d}: aux={history['aux_loss'][-1]:.4f}, "
                  f"kl={avg_kl:.6f}, future={history['future_loss'][-1]:.4f}, "
                  f"alpha={history['alpha'][-1]:.6f}, lambda={lambda_val:.2f}")

        return history

    def save_checkpoint(self, path: str = "results/feel_training/canonical_v5_1_checkpoint.pt"):
        """Save trained checkpoint."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            "feel_stream_state": self.feel_stream.state_dict(),
            "aux_head_state": self.aux_head.state_dict(),
            "future_head_state": self.future_head.state_dict(),
            "alpha": self.feel_stream.get_alpha(),
            "kl_budget": self.kl_budget,
            "version": "v5.1",
            "fix": "removed .detach() from future prediction - z_feel now learns",
        }

        torch.save(checkpoint, path)
        print(f"\nCheckpoint saved: {path}")
        print(f"  Alpha: {checkpoint['alpha']:.6f}")
        print(f"  Version: {checkpoint['version']}")

        return checkpoint


def main():
    parser = argparse.ArgumentParser(description="FEEL Canonical Training v5.1")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--kl-budget", type=float, default=0.01)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--n-tokens", type=int, default=32)
    parser.add_argument("--future-weight", type=float, default=0.5,
                       help="Weight for future prediction loss")
    parser.add_argument("--fixed-alpha", type=float, default=None)
    parser.add_argument("--model", type=str, default="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
    args = parser.parse_args()

    print("=" * 70)
    print("  FEEL CANONICAL TRAINING v5.1 - FIXED PREDICTIVE LOSS")
    print("=" * 70)
    print()
    print("KEY FIX: Removed .detach() from future prediction")
    print("  v5.0: future_head(z_feel.detach()) → z_feel doesn't learn")
    print("  v5.1: future_head(z_feel) → z_feel learns predictive representations")
    print()

    trainer = CanonicalFEELTrainerV51(
        model_name=args.model,
        kl_budget=args.kl_budget,
        fixed_alpha=args.fixed_alpha,
    )

    history = trainer.train(
        epochs=args.epochs,
        lr=args.lr,
        n_tokens=args.n_tokens,
        future_loss_weight=args.future_weight,
    )

    trainer.save_checkpoint()

    # Save history
    history_path = "results/feel_training/v5_1_training_history.json"
    Path(history_path).parent.mkdir(parents=True, exist_ok=True)
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"\nTraining history saved: {history_path}")

    print("\n" + "=" * 70)
    print("  TRAINING COMPLETE")
    print("=" * 70)
    print(f"  Final alpha: {history['alpha'][-1]:.6f}")
    print(f"  Final aux loss: {history['aux_loss'][-1]:.4f}")
    print(f"  Final future loss: {history['future_loss'][-1]:.4f}")
    print(f"  Final KL: {history['kl'][-1]:.6f}")


if __name__ == "__main__":
    main()
