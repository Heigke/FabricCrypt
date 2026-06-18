#!/usr/bin/env python3
"""
FEEL Projector Training Script v3.0 (USEFULNESS OBJECTIVE)

Trains the FEEL token projector while keeping the base LLM frozen.

KEY FIXES from v2.0:
- Aux head now reads from LM HIDDEN STATE (not z_feel) - forces FEEL to modulate network
- KL is a CONSTRAINT (budget), not the main objective - FEEL must be useful
- Alpha can be FROZEN for sweeps to establish effect-size vs coherence curves

Training objectives:
1. MAXIMIZE aux task: predict entropy from LM hidden state (FEEL must influence h_last)
2. SUBJECT TO KL budget: keep KL(on||off) ≤ ε via adaptive Lagrange multiplier
3. Alpha scheduling: freeze at fixed values {0.02, 0.05, 0.1, 0.2} for controlled experiments

The key insight: The aux task can ONLY be solved if FEEL modulates the hidden state.
This gives the optimizer a reason to shape FEEL embeddings into something useful.

Usage:
    python scripts/train_feel_projector.py --epochs 20 --fixed-alpha 0.1 --kl-budget 0.05
"""

import sys
import time
import json
import argparse
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass, asdict
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.predictive_interoception import (
    PredictiveZFeel,
    FEELTokenStream,
)
from scripts.internal_signal_extractor import InternalSignalExtractor


# =============================================================================
# AUXILIARY HEADS
# =============================================================================

class UncertaintyPredictor(nn.Module):
    """
    DEPRECATED: Predicts entropy from z_feel (trivial optimum: FEEL stays invisible).
    Kept for backward compatibility only.
    """

    def __init__(self, z_dim: int = 64, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(z_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, z_feel: torch.Tensor) -> torch.Tensor:
        return self.net(z_feel).squeeze(-1)


class HiddenStateAuxHead(nn.Module):
    """
    Predicts next-token entropy from the LM's HIDDEN STATE (not z_feel).

    This is the KEY fix: the aux task can ONLY be solved if FEEL modulates
    the hidden state. With the base model frozen, the ONLY way for h_last
    to carry extra info is if FEEL injection actually changes it.

    This gives the optimizer a reason to shape FEEL embeddings into
    something the LM "uses" - like prompt-tuning/prefix-tuning.
    """

    def __init__(self, hidden_dim: int = 1536, proj_dim: int = 128):
        super().__init__()
        # Project from LM hidden dim to smaller space
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, proj_dim),
            nn.GELU(),
            nn.LayerNorm(proj_dim),
            nn.Linear(proj_dim, proj_dim),
            nn.GELU(),
            nn.Linear(proj_dim, 1),  # Predict entropy
        )

    def forward(self, h_last: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h_last: [batch, hidden_dim] - last hidden state from LM
        Returns:
            entropy prediction [batch]
        """
        return self.net(h_last).squeeze(-1)


# =============================================================================
# TRAINABLE FEEL MODEL
# =============================================================================

class TrainableFEELModel(nn.Module):
    """
    FEEL model with trainable projector and frozen base LLM.

    v3.0 Changes:
    - HiddenStateAuxHead reads from LM hidden state (not z_feel)
    - Supports fixed_alpha for controlled experiments
    - output_hidden_states=True for FEEL-on branch

    Trainable components:
    - z_feel encoder (GRU + codebook)
    - FEEL token projector (z_to_embed)
    - Alpha gate (can be frozen)
    - HiddenStateAuxHead (predicts entropy from LM hidden state)

    Frozen:
    - Base LLM (all parameters)
    """

    def __init__(
        self,
        model: nn.Module,
        tokenizer,
        z_dim: int = 64,
        device: str = "cuda",
        fixed_alpha: Optional[float] = None,  # If set, freeze alpha at this value
    ):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.z_dim = z_dim
        self.fixed_alpha = fixed_alpha

        # Freeze base model
        for param in model.parameters():
            param.requires_grad = False

        # Get model's hidden dimension and dtype
        if hasattr(model.config, 'hidden_size'):
            self.hidden_dim = model.config.hidden_size
        else:
            self.hidden_dim = 1536
        model_dtype = next(model.parameters()).dtype

        # Trainable FEEL components
        self.z_feel_model = PredictiveZFeel(
            sensor_dim=12,
            z_dim=z_dim,
            forecast_horizon=8,
            num_codes=64,
        ).to(device=device, dtype=torch.float32)  # Keep in float32 for stability

        self.feel_stream = FEELTokenStream(
            z_dim=z_dim,
            embed_dim=self.hidden_dim,
            n_feel_tokens=1,
            update_rate=1,
        ).to(device=device, dtype=torch.float32)

        # If fixed_alpha is set, override the alpha parameter
        if fixed_alpha is not None:
            # Compute what raw alpha should be to get fixed_alpha via softplus(alpha-4)
            # softplus(x) = log(1 + exp(x)), so we need to solve: fixed_alpha = softplus(raw - 4)
            # For small fixed_alpha: raw ≈ log(exp(fixed_alpha) - 1) + 4
            raw_alpha = np.log(np.exp(fixed_alpha) - 1 + 1e-8) + 4.0
            with torch.no_grad():
                self.feel_stream.alpha.fill_(raw_alpha)
            self.feel_stream.alpha.requires_grad = False  # Freeze alpha

        # OLD: Uncertainty predictor from z_feel (trivial optimum - deprecated)
        self.uncertainty_predictor = UncertaintyPredictor(
            z_dim=z_dim,
            hidden_dim=128,
        ).to(device=device, dtype=torch.float32)

        # NEW: HiddenStateAuxHead reads from LM hidden state
        # This forces FEEL to modulate the network to solve the aux task
        self.hidden_state_aux_head = HiddenStateAuxHead(
            hidden_dim=self.hidden_dim,
            proj_dim=128,
        ).to(device=device, dtype=torch.float32)

        # Signal extractor (not trainable)
        self.signal_extractor = InternalSignalExtractor(model, device=device)

    def reset(self):
        """Reset state for new sequence."""
        self.z_feel_model.reset_state()
        self.feel_stream.reset()
        self.signal_extractor.reset()

    def extract_signals(
        self,
        logits: torch.Tensor,
        input_length: int = 0,
        chosen_token_id: Optional[int] = None,
    ) -> Tuple[torch.Tensor, float]:
        """Extract signals and return (sensors, actual_entropy)."""
        signals = self.signal_extractor.extract(
            logits=logits,
            attentions=None,
            hidden_states=None,
            input_length=input_length,
        )

        # Compute actual entropy for training target
        logits_f32 = logits[:, -1, :].float()
        probs = F.softmax(logits_f32, dim=-1)
        actual_entropy = -(probs * torch.log(probs.clamp(min=1e-10))).sum(dim=-1).item()

        # Compute surprisal if token provided
        surprisal = 0.0
        if chosen_token_id is not None:
            log_probs = F.log_softmax(logits_f32, dim=-1)
            surprisal = -log_probs[0, chosen_token_id].item()

        # Build 12-dim sensor vector
        sensors = torch.tensor([
            min(signals.logit_entropy / 5.0, 1.0),
            signals.logit_margin,
            signals.top_k_mass,
            signals.uncertainty_score,
            min(signals.tokens_per_second / 100.0, 1.0),
            min(signals.time_per_token_ms / 100.0, 1.0),
            min(signals.kv_cache_tokens / 4096.0, 1.0),
            min(surprisal / 15.0, 1.0),
            min(signals.attention_entropy / 5.0, 1.0),
            min(signals.residual_norm_mean / 100.0, 1.0),
            signals.stress_indicator,
            min(signals.generation_depth / 256.0, 1.0),
        ], dtype=torch.float32, device=self.device)

        return sensors, actual_entropy

    def forward_step(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        z_feel: Optional[torch.Tensor] = None,
        past_key_values=None,
        with_feel: bool = True,
        return_hidden_states: bool = False,  # v3.0: for aux head
    ):
        """
        Single forward step with or without FEEL injection.

        v3.0: Added return_hidden_states for aux head that reads LM hidden state.

        Returns:
            logits: Model output logits
            feel_embed: FEEL embedding (for loss computation)
            past_key_values: Updated KV cache
            h_last (optional): Last hidden state if return_hidden_states=True
        """
        # Get input embeddings
        input_embeds = self.model.get_input_embeddings()(input_ids)

        # Inject FEEL token if z_feel available AND with_feel=True
        feel_embed = None
        if z_feel is not None and with_feel:
            feel_embed = self.feel_stream(z_feel, return_raw=True)
            feel_embed = feel_embed.to(dtype=input_embeds.dtype, device=input_embeds.device)

            # Prepend FEEL embedding
            input_embeds = torch.cat([feel_embed, input_embeds], dim=1)

            # Extend attention mask
            feel_mask = torch.ones(
                attention_mask.size(0), 1,
                device=attention_mask.device,
                dtype=attention_mask.dtype,
            )
            attention_mask = torch.cat([feel_mask, attention_mask], dim=1)

        # Forward through model - NO torch.no_grad()!
        # Gradients flow through feel_embed → feel_stream → z_feel_model
        # Base model weights are frozen via requires_grad=False (set in __init__)
        outputs = self.model(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=True,
            output_hidden_states=return_hidden_states,  # v3.0
        )

        if return_hidden_states:
            # Get last hidden state at last position
            # hidden_states is tuple of (embed, layer1, ..., layerN)
            # We want layerN at position -1 (last token)
            h_last = outputs.hidden_states[-1][:, -1, :]  # [batch, hidden_dim]
            return outputs.logits, feel_embed, outputs.past_key_values, h_last

        return outputs.logits, feel_embed, outputs.past_key_values, None

    def get_trainable_params(self) -> List[nn.Parameter]:
        """Get list of trainable parameters."""
        params = []
        params.extend(self.z_feel_model.parameters())
        params.extend(self.feel_stream.parameters())
        params.extend(self.uncertainty_predictor.parameters())
        params.extend(self.hidden_state_aux_head.parameters())  # v3.0
        return params


# =============================================================================
# TRAINING DATA
# =============================================================================

class PromptDataset(Dataset):
    """Simple dataset of prompts for training."""

    def __init__(self, prompts: List[str]):
        self.prompts = prompts

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        return self.prompts[idx]


def get_training_prompts() -> List[str]:
    """Get diverse training prompts."""
    return [
        # Factual questions (low uncertainty expected)
        "What is 2 + 2?",
        "What color is the sky on a clear day?",
        "How many days are in a week?",

        # Reasoning tasks (medium uncertainty)
        "Explain step by step how to solve 15 * 23.",
        "What are the pros and cons of electric cars?",
        "Describe the process of photosynthesis.",

        # Creative/open-ended (high uncertainty expected)
        "Write a short poem about the ocean.",
        "Imagine a conversation between a cat and a dog.",
        "What would happen if humans could fly?",

        # Introspective (metacognitive)
        "How confident are you in your answer?",
        "What are you uncertain about right now?",
        "Describe your reasoning process.",

        # Technical
        "Write a Python function to reverse a string.",
        "Explain how neural networks learn.",
        "What is the difference between TCP and UDP?",

        # Ambiguous
        "Is it better to be happy or successful?",
        "What is the meaning of life?",
        "Should AI have rights?",
    ]


# =============================================================================
# TRAINING LOOP
# =============================================================================

class FEELTrainer:
    """
    Trainer for FEEL projector v3.0.

    Key changes:
    - Aux head reads from LM hidden state (not z_feel)
    - KL is a constraint (budget), not the main objective
    - Supports fixed_alpha for controlled experiments
    - Adaptive lambda for KL budget constraint
    """

    def __init__(
        self,
        model_id: str = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
        device: str = "cuda",
        z_dim: int = 64,
        lr: float = 1e-4,
        alpha_reg: float = 0.01,
        uncertainty_weight: float = 0.1,
        output_dir: str = "results/feel_training",
        # v3.0 parameters
        fixed_alpha: Optional[float] = None,  # If set, freeze alpha at this value
        kl_budget: float = 0.05,  # Target KL budget (constraint)
        lambda_kl: float = 1.0,  # Initial Lagrange multiplier for KL constraint
        lambda_lr: float = 0.01,  # Learning rate for adaptive lambda
    ):
        self.device = device
        self.lr = lr
        self.alpha_reg = alpha_reg
        self.uncertainty_weight = uncertainty_weight
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # v3.0: KL constraint parameters
        self.fixed_alpha = fixed_alpha
        self.kl_budget = kl_budget
        self.lambda_kl = lambda_kl  # Will be updated adaptively
        self.lambda_lr = lambda_lr

        print(f"Loading model: {model_id}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        base_model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            device_map=device,
            trust_remote_code=True,
        )
        base_model.eval()

        self.model = TrainableFEELModel(
            base_model, self.tokenizer,
            z_dim=z_dim,
            device=device,
            fixed_alpha=fixed_alpha,  # v3.0
        )

        # Optimizer for trainable parameters only
        self.optimizer = torch.optim.AdamW(
            self.model.get_trainable_params(),
            lr=lr,
            weight_decay=0.01,
        )

        self.training_log = []

    def compute_kl_to_baseline(
        self,
        logits_on: torch.Tensor,
        logits_off: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute KL(P_on || P_off) - how much FEEL changes the distribution.

        This is the CORRECT loss: we want FEEL to not hurt predictions,
        measured as minimal divergence from the baseline (FEEL-off) distribution.

        Unlike argmax-based perplexity loss, this:
        - Uses the FULL distribution, not just the argmax
        - Compares to a principled baseline (FEEL-off logits)
        - Allows gradients to flow through FEEL embeddings
        """
        # P_on = softmax(logits_on), P_off = softmax(logits_off)
        # KL(P_on || P_off) = sum(P_on * log(P_on / P_off))
        p_on = F.softmax(logits_on.float(), dim=-1)
        log_p_on = F.log_softmax(logits_on.float(), dim=-1)
        log_p_off = F.log_softmax(logits_off.float(), dim=-1)

        # KL divergence: sum(P_on * (log_P_on - log_P_off))
        kl = (p_on * (log_p_on - log_p_off)).sum(dim=-1).mean()

        return kl

    def _strip_feel_from_cache(self, past_key_values, n_feel_tokens: int = 1):
        """Strip FEEL tokens from KV cache (same logic as experiments)."""
        if past_key_values is None:
            return None

        if hasattr(past_key_values, 'get_seq_length'):
            from transformers.cache_utils import DynamicCache
            new_cache = DynamicCache()
            for layer_idx in range(len(past_key_values)):
                key, value = past_key_values[layer_idx]
                if key.size(2) > n_feel_tokens:
                    key_stripped = torch.cat([key[:, :, :-2, :], key[:, :, -1:, :]], dim=2)
                    value_stripped = torch.cat([value[:, :, :-2, :], value[:, :, -1:, :]], dim=2)
                    new_cache.update(key_stripped, value_stripped, layer_idx)
                else:
                    new_cache.update(key, value, layer_idx)
            return new_cache
        else:
            stripped = []
            for layer_kv in past_key_values:
                key, value = layer_kv
                if key.size(2) > n_feel_tokens:
                    key_stripped = torch.cat([key[:, :, :-2, :], key[:, :, -1:, :]], dim=2)
                    value_stripped = torch.cat([value[:, :, :-2, :], value[:, :, -1:, :]], dim=2)
                    stripped.append((key_stripped, value_stripped))
                else:
                    stripped.append((key, value))
            return tuple(stripped)

    def train_on_prompt(
        self,
        prompt: str,
        max_tokens: int = 32,
    ) -> Dict[str, float]:
        """
        Train on a single prompt using v3.0 methodology.

        KEY CHANGES v3.0:
        - Aux head reads from LM HIDDEN STATE (not z_feel) - forces FEEL to modulate network
        - KL is a CONSTRAINT (budget) not the main objective
        - Adaptive lambda for KL budget

        For each token position:
        1. Forward with FEEL OFF → get baseline logits (with cache)
        2. Forward with FEEL ON → get FEEL-augmented logits + hidden state (with cache + stripping)
        3. Aux head predicts entropy from h_last (FEEL must modulate network to help)
        4. KL is a constraint: λ * max(0, KL - budget)
        """
        self.model.reset()

        # Prepare input
        messages = [{"role": "user", "content": prompt}]
        input_text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(input_text, return_tensors="pt").to(self.device)
        input_ids = inputs["input_ids"]
        attention_mask = inputs.get("attention_mask", torch.ones_like(input_ids))
        input_length = input_ids.shape[1]

        generated_ids = input_ids.clone()

        # Losses
        total_kl_loss = 0.0
        total_aux_loss = 0.0  # v3.0: renamed from uncertainty_loss
        total_alpha_loss = 0.0
        n_steps = 0

        current_z_feel = None

        # KV caches for both branches
        past_key_values_off = None
        past_key_values_on = None

        for step in range(max_tokens):
            # Step 1: Get baseline logits (FEEL OFF) with proper cache
            with torch.no_grad():
                if step == 0:
                    logits_off, _, past_off_new, _ = self.model.forward_step(
                        input_ids=generated_ids,
                        attention_mask=attention_mask,
                        z_feel=current_z_feel,
                        past_key_values=None,
                        with_feel=False,
                    )
                else:
                    logits_off, _, past_off_new, _ = self.model.forward_step(
                        input_ids=generated_ids[:, -1:],
                        attention_mask=attention_mask,
                        z_feel=current_z_feel,
                        past_key_values=past_key_values_off,
                        with_feel=False,
                    )
                past_key_values_off = past_off_new

            # Step 2: Get FEEL-augmented logits + hidden state (FEEL ON)
            # v3.0: Request hidden states for aux head
            if step == 0:
                logits_on, feel_embed, past_on_new, h_last = self.model.forward_step(
                    input_ids=generated_ids,
                    attention_mask=attention_mask,
                    z_feel=current_z_feel,
                    past_key_values=None,
                    with_feel=True,
                    return_hidden_states=True,  # v3.0: for aux head
                )
            else:
                logits_on, feel_embed, past_on_new, h_last = self.model.forward_step(
                    input_ids=generated_ids[:, -1:],
                    attention_mask=attention_mask,
                    z_feel=current_z_feel,
                    past_key_values=past_key_values_on,
                    with_feel=True,
                    return_hidden_states=True,  # v3.0: for aux head
                )

            # Update FEEL-on cache WITH stripping
            if step > 0 and current_z_feel is not None:
                past_key_values_on = self._strip_feel_from_cache(past_on_new, n_feel_tokens=1)
            else:
                past_key_values_on = past_on_new

            # Step 3: KL to baseline (now a CONSTRAINT, not main loss)
            kl_loss = self.compute_kl_to_baseline(
                logits_on[:, -1, :],
                logits_off[:, -1, :].detach(),
            )
            total_kl_loss += kl_loss

            # Sample next token from FEEL-on distribution
            next_token = logits_on[:, -1, :].argmax(dim=-1, keepdim=True)
            next_token_id = next_token.item()

            # Extract signals from FEEL-off baseline for ground truth
            sensors, actual_entropy = self.model.extract_signals(
                logits_off, input_length, next_token_id
            )

            # Update z_feel (with gradients flowing through)
            out = self.model.z_feel_model(sensors)
            current_z_feel = out['z_feel']

            # v3.0: Aux head predicts entropy from LM HIDDEN STATE (not z_feel!)
            # This forces FEEL to modulate the network to solve the aux task
            if h_last is not None:
                predicted_entropy = self.model.hidden_state_aux_head(h_last.float())
                entropy_target = torch.tensor(
                    actual_entropy / 5.0,
                    device=self.device,
                    dtype=torch.float32,
                )
                aux_loss = F.mse_loss(predicted_entropy, entropy_target.unsqueeze(0))
                total_aux_loss += aux_loss

            # Alpha regularization (only if alpha is learnable)
            alpha = self.model.feel_stream.alpha
            if alpha.requires_grad:
                alpha_loss = F.softplus(alpha - 4.0).mean()
                total_alpha_loss += alpha_loss

            # Update for next step
            generated_ids = torch.cat([generated_ids, next_token], dim=-1)
            attention_mask = torch.cat([
                attention_mask,
                torch.ones((1, 1), device=self.device, dtype=attention_mask.dtype)
            ], dim=1)

            n_steps += 1

            if next_token_id == self.tokenizer.eos_token_id:
                break

        # Compute total loss (v3.0: aux is main objective, KL is constraint)
        if n_steps > 0:
            avg_kl_loss = total_kl_loss / n_steps
            avg_aux_loss = total_aux_loss / n_steps
            avg_alpha_loss = total_alpha_loss / n_steps if total_alpha_loss > 0 else torch.tensor(0.0)

            # v3.0: KL is a CONSTRAINT, not the main objective
            # Use adaptive Lagrange multiplier: λ * max(0, KL - budget)
            kl_value = avg_kl_loss.item() if torch.is_tensor(avg_kl_loss) else avg_kl_loss
            kl_violation = max(0, kl_value - self.kl_budget)

            # Main objective: MINIMIZE aux_loss (predict entropy from hidden state)
            # Constraint: keep KL within budget
            total_loss = (
                avg_aux_loss +  # Main objective: aux head must predict entropy
                self.lambda_kl * kl_violation +  # KL constraint (only penalty if over budget)
                self.alpha_reg * avg_alpha_loss  # Alpha regularization (if alpha learnable)
            )

            # Convert kl_violation to tensor for backward if needed
            if not torch.is_tensor(total_loss):
                total_loss = avg_aux_loss + self.lambda_kl * avg_kl_loss

            total_loss.backward()

            # v3.0: Update adaptive lambda after backward
            # If KL > budget, increase lambda; if KL < budget, decrease lambda
            if kl_value > self.kl_budget:
                self.lambda_kl = min(10.0, self.lambda_kl * (1 + self.lambda_lr))
            else:
                self.lambda_kl = max(0.1, self.lambda_kl * (1 - self.lambda_lr * 0.1))

            return {
                'total_loss': total_loss.item() if torch.is_tensor(total_loss) else total_loss,
                'kl_loss': kl_value,
                'aux_loss': avg_aux_loss.item() if torch.is_tensor(avg_aux_loss) else avg_aux_loss,
                'alpha_loss': avg_alpha_loss.item() if torch.is_tensor(avg_alpha_loss) else avg_alpha_loss,
                'alpha_value': F.softplus(alpha - 4.0).item(),
                'lambda_kl': self.lambda_kl,
                'kl_budget': self.kl_budget,
                'n_steps': n_steps,
            }

        return {'total_loss': 0.0, 'n_steps': 0}

    def train_epoch(
        self,
        prompts: List[str],
        batch_size: int = 4,
    ) -> Dict[str, float]:
        """Train one epoch over all prompts (v3.0)."""
        self.model.train()

        epoch_losses = {
            'total_loss': [],
            'kl_loss': [],
            'aux_loss': [],  # v3.0: renamed from uncertainty_loss
            'alpha_loss': [],
            'alpha_value': [],
            'lambda_kl': [],  # v3.0: track adaptive lambda
        }

        for i, prompt in enumerate(prompts):
            self.optimizer.zero_grad()

            try:
                losses = self.train_on_prompt(prompt)

                if losses['n_steps'] > 0:
                    # Gradient clipping
                    torch.nn.utils.clip_grad_norm_(
                        self.model.get_trainable_params(),
                        max_norm=1.0,
                    )

                    self.optimizer.step()

                    for k in epoch_losses:
                        if k in losses:
                            epoch_losses[k].append(losses[k])

                    # v3.0: Updated print to show aux_loss and lambda
                    print(f"  [{i+1}/{len(prompts)}] loss={losses['total_loss']:.4f} "
                          f"kl={losses['kl_loss']:.4f} "
                          f"aux={losses['aux_loss']:.4f} "
                          f"α={losses['alpha_value']:.4f} "
                          f"λ={losses.get('lambda_kl', self.lambda_kl):.2f}")

            except Exception as e:
                print(f"  [{i+1}/{len(prompts)}] ERROR: {e}")
                import traceback
                traceback.print_exc()
                continue

        # Average losses
        avg_losses = {k: np.mean(v) if v else 0.0 for k, v in epoch_losses.items()}
        return avg_losses

    def train(
        self,
        n_epochs: int = 5,
        prompts: Optional[List[str]] = None,
    ):
        """Full training loop (v3.0)."""
        if prompts is None:
            prompts = get_training_prompts()

        print(f"\n{'='*60}")
        print("  FEEL PROJECTOR TRAINING v3.0")
        print(f"{'='*60}")
        print(f"Epochs: {n_epochs}")
        print(f"Prompts: {len(prompts)}")
        print(f"LR: {self.lr}")
        print(f"KL budget: {self.kl_budget}")
        print(f"Fixed alpha: {self.fixed_alpha}")
        print(f"Lambda KL (initial): {self.lambda_kl}")

        for epoch in range(n_epochs):
            print(f"\n--- Epoch {epoch+1}/{n_epochs} ---")
            start_time = time.perf_counter()

            losses = self.train_epoch(prompts)

            elapsed = time.perf_counter() - start_time

            print(f"\nEpoch {epoch+1} Summary:")
            print(f"  Total loss: {losses['total_loss']:.4f}")
            print(f"  KL loss: {losses['kl_loss']:.4f} (budget: {self.kl_budget})")
            print(f"  Aux loss (h_last→entropy): {losses['aux_loss']:.4f}")
            print(f"  Alpha value: {losses['alpha_value']:.4f}")
            print(f"  Lambda KL: {losses.get('lambda_kl', self.lambda_kl):.2f}")
            print(f"  Time: {elapsed:.1f}s")

            self.training_log.append({
                'epoch': epoch + 1,
                **losses,
                'time_seconds': elapsed,
            })

        # Save trained model
        self.save_checkpoint()

        return self.training_log

    def save_checkpoint(self):
        """Save trained FEEL components."""
        checkpoint_path = self.output_dir / "feel_projector_checkpoint.pt"

        checkpoint = {
            'z_feel_model': self.model.z_feel_model.state_dict(),
            'feel_stream': self.model.feel_stream.state_dict(),
            'uncertainty_predictor': self.model.uncertainty_predictor.state_dict(),
            'training_log': self.training_log,
        }

        torch.save(checkpoint, checkpoint_path)
        print(f"\nCheckpoint saved to: {checkpoint_path}")

        # Also save training log as JSON
        log_path = self.output_dir / "training_log.json"
        with open(log_path, 'w') as f:
            json.dump(self.training_log, f, indent=2)


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Train FEEL Projector v3.0")
    parser.add_argument("--model", default="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--alpha-reg", type=float, default=0.01)
    parser.add_argument("--uncertainty-weight", type=float, default=0.1)
    parser.add_argument("--output-dir", default="results/feel_training")
    # v3.0 parameters
    parser.add_argument("--fixed-alpha", type=float, default=None,
                       help="Fix alpha at this value (e.g., 0.02, 0.05, 0.1, 0.2). If None, alpha is learnable.")
    parser.add_argument("--kl-budget", type=float, default=0.05,
                       help="KL budget constraint. KL is penalized only when > budget.")
    parser.add_argument("--lambda-kl", type=float, default=1.0,
                       help="Initial Lagrange multiplier for KL constraint.")
    args = parser.parse_args()

    trainer = FEELTrainer(
        model_id=args.model,
        device=args.device,
        lr=args.lr,
        alpha_reg=args.alpha_reg,
        uncertainty_weight=args.uncertainty_weight,
        output_dir=args.output_dir,
        # v3.0 parameters
        fixed_alpha=args.fixed_alpha,
        kl_budget=args.kl_budget,
        lambda_kl=args.lambda_kl,
    )

    trainer.train(n_epochs=args.epochs)


if __name__ == "__main__":
    main()
