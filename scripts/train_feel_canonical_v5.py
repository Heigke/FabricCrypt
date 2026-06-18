#!/usr/bin/env python3
"""
FEEL Canonical Training v5.0 - Differentiable KL Constraint
============================================================

Trains the unified 12-sensor FEEL pipeline with:
1. Aux head that reads from LM hidden state (forces FEEL to modulate network)
2. DIFFERENTIABLE KL constraint via Lagrangian multiplier (not hard clamp)
3. Predictive loss: z_feel must predict FUTURE entropy (not current)

Training Objective:
    L = -L_aux (maximize aux task) + lambda * max(0, KL - budget)^2 + L_predictive

Where lambda adapts to enforce the KL budget softly.

Usage:
    python scripts/train_feel_canonical_v5.py --epochs 30 --kl-budget 0.01
"""

import sys
import time
import json
import argparse
from pathlib import Path
from typing import Dict, Tuple, Optional
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent.parent))


# ============================================================
# Canonical 12-Sensor Pipeline (from feel_breakthrough_v5.py)
# ============================================================

class CanonicalSensorBank(nn.Module):
    """12-dimensional sensor bank matching training pipeline."""

    def __init__(self):
        super().__init__()
        self.sensor_names = [
            "entropy_norm", "logit_margin", "top_k_mass", "uncertainty",
            "tps_norm", "latency_norm", "kv_cache_norm", "surprisal_norm",
            "attn_entropy_norm", "residual_norm", "stress_indicator", "depth_norm"
        ]

    def forward(
        self,
        logits: torch.Tensor,
        chosen_token_id: Optional[int] = None,
        kv_cache_tokens: int = 0,
        generation_depth: int = 0,
    ) -> torch.Tensor:
        orig_dtype = logits.dtype
        device = logits.device
        logits_f32 = logits[:, -1, :].float()

        probs = F.softmax(logits_f32, dim=-1)
        log_probs = F.log_softmax(logits_f32, dim=-1)

        # Sensors (same as v5.0)
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

        return sensors.to(orig_dtype)


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
            nn.Linear(64, horizon),  # Predict entropy at t+1, t+2, ..., t+horizon
        )

    def forward(self, z_feel: torch.Tensor) -> torch.Tensor:
        return self.net(z_feel)


# ============================================================
# Differentiable KL Constraint with Lagrangian
# ============================================================

class DifferentiableKLConstraint(nn.Module):
    """
    Soft KL constraint via Lagrangian multiplier.

    Instead of: if kl > budget: continue (non-differentiable)
    We use:     loss += lambda * max(0, kl - budget)^2

    Lambda adapts during training to enforce the budget.
    """

    def __init__(self, kl_budget: float = 0.01, lambda_init: float = 1.0, lambda_lr: float = 0.1):
        super().__init__()
        self.kl_budget = kl_budget
        self.log_lambda = nn.Parameter(torch.tensor(np.log(lambda_init)))
        self.lambda_lr = lambda_lr

    def forward(self, kl: torch.Tensor) -> Tuple[torch.Tensor, float]:
        """
        Returns (constraint_loss, current_lambda).

        The gradient w.r.t. kl is: 2 * lambda * max(0, kl - budget)
        The gradient w.r.t. lambda adapts it to enforce budget.
        """
        lambda_val = self.log_lambda.exp()
        violation = F.relu(kl - self.kl_budget)
        constraint_loss = lambda_val * violation ** 2

        return constraint_loss, lambda_val.item()

    def update_lambda(self, avg_kl: float):
        """Manual lambda update step (dual ascent)."""
        with torch.no_grad():
            if avg_kl > self.kl_budget:
                # KL too high, increase lambda
                self.log_lambda.data += self.lambda_lr
            else:
                # KL within budget, decrease lambda
                self.log_lambda.data -= self.lambda_lr * 0.5
            # Clamp lambda
            self.log_lambda.data.clamp_(-2, 5)


# ============================================================
# Trainer
# ============================================================

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


class CanonicalFEELTrainer:
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
        print("  Tokenizer loaded", flush=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32,
            trust_remote_code=True,
            device_map="auto"
        )
        print("  Base model loaded", flush=True)
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

        self.embed_dim = self.model.config.hidden_size
        self.model_dtype = next(self.model.parameters()).dtype
        print(f"  Embed dim: {self.embed_dim}", flush=True)

        # Trainable components (float32 for stability)
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

        print(f"  Model loaded (dtype: {self.model_dtype})")
        print(f"  Initial alpha: {self.feel_stream.get_alpha():.4f}")

    def train(self, epochs: int = 20, lr: float = 1e-3, n_tokens: int = 32):
        """Train with differentiable KL constraint."""

        # Optimizer for all trainable components
        trainable_params = (
            list(self.feel_stream.parameters()) +
            list(self.aux_head.parameters()) +
            list(self.future_head.parameters()) +
            list(self.kl_constraint.parameters())
        )
        optimizer = torch.optim.AdamW(trainable_params, lr=lr)

        history = {"aux_loss": [], "kl": [], "future_loss": [], "alpha": [], "lambda": []}

        print(f"\nStarting training (epochs={epochs}, kl_budget={self.kl_budget})...")

        for epoch in range(epochs):
            epoch_metrics = {"aux_loss": [], "kl": [], "future_loss": [], "constraint_loss": []}

            for prompt in TRAINING_PROMPTS:
                input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
                current_ids = input_ids.clone()

                z_feel_history = []
                entropy_history = []

                # Generate sequence and collect data
                for step in range(n_tokens + 4):  # +4 for future prediction
                    with torch.no_grad():
                        outputs_base = self.model(current_ids, use_cache=False)
                        logits_base = outputs_base.logits

                    # Get FEEL embedding (requires grad)
                    feel_embed, sensors, z_feel, alpha = self.feel_stream(
                        logits_base.float(),
                        kv_cache_tokens=current_ids.shape[1],
                        generation_depth=step
                    )

                    z_feel_history.append(z_feel)

                    # Current entropy (target)
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

                    # Get last hidden state
                    h_last = outputs_feel.hidden_states[-1][:, -1, :].float()

                    # Aux loss: predict entropy from hidden state
                    aux_pred = self.aux_head(h_last)
                    aux_loss = F.mse_loss(aux_pred, entropy)

                    # KL divergence
                    p_base = F.softmax(logits_base[:, -1, :].float(), dim=-1)
                    p_feel = F.softmax(outputs_feel.logits[:, -1, :].float(), dim=-1)
                    kl = F.kl_div(p_feel.log(), p_base, reduction='batchmean')

                    # Differentiable KL constraint
                    constraint_loss, lambda_val = self.kl_constraint(kl)

                    # Total loss for this step
                    step_loss = aux_loss + constraint_loss

                    # Backward
                    optimizer.zero_grad()
                    step_loss.backward()
                    torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
                    optimizer.step()

                    epoch_metrics["aux_loss"].append(aux_loss.item())
                    epoch_metrics["kl"].append(kl.item())
                    epoch_metrics["constraint_loss"].append(constraint_loss.item())

                    # Next token
                    with torch.no_grad():
                        next_token = outputs_feel.logits[:, -1, :].argmax(dim=-1, keepdim=True)
                        current_ids = torch.cat([current_ids, next_token], dim=-1)

                # Future prediction loss (after sequence)
                if len(z_feel_history) > 4:
                    for t in range(len(z_feel_history) - 4):
                        future_pred = self.future_head(z_feel_history[t].detach())
                        future_targets = torch.tensor(
                            entropy_history[t+1:t+5],
                            device=self.device
                        )
                        future_loss = F.mse_loss(future_pred.squeeze(), future_targets)

                        optimizer.zero_grad()
                        future_loss.backward()
                        optimizer.step()

                        epoch_metrics["future_loss"].append(future_loss.item())

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
                  f"alpha={history['alpha'][-1]:.4f}, lambda={lambda_val:.2f}")

        return history

    def save_checkpoint(self, path: str = "results/feel_training/canonical_v5_checkpoint.pt"):
        """Save trained checkpoint."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            "feel_stream_state": self.feel_stream.state_dict(),
            "aux_head_state": self.aux_head.state_dict(),
            "future_head_state": self.future_head.state_dict(),
            "alpha": self.feel_stream.get_alpha(),
            "kl_budget": self.kl_budget,
            "version": "v5.0",
        }

        torch.save(checkpoint, path)
        print(f"Checkpoint saved to {path}")


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--kl-budget", type=float, default=0.01)
    parser.add_argument("--fixed-alpha", type=float, default=None)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    trainer = CanonicalFEELTrainer(
        kl_budget=args.kl_budget,
        fixed_alpha=args.fixed_alpha,
    )

    history = trainer.train(epochs=args.epochs, lr=args.lr)
    trainer.save_checkpoint()

    # Save history
    history_path = "results/feel_training/canonical_v5_history.json"
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"History saved to {history_path}")
