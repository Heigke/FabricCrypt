#!/usr/bin/env python3
"""
FEEL Token Integration Experiments v3.0

Comprehensive evaluation with CORRECTED metrics and falsification battery.

Key changes from v2.0:
- Fixed attention metric: renamed to "local attention share" with correct baseline
- Added CAUSAL influence metric: KL divergence FEEL on vs off (the gold standard)
- Added SALIENCY metric: gradient norm w.r.t. FEEL embedding
- Added FALSIFICATION battery: shuffle, clamp, lag sweep
- Added MI proxy: z_feel predicts future entropy better than raw sensors

Usage:
    python scripts/feel_integration_experiments.py --experiment all
    python scripts/feel_integration_experiments.py --experiment causal_influence
    python scripts/feel_integration_experiments.py --experiment falsification
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
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.predictive_interoception import (
    PredictiveZFeel,
    FEELTokenStream,
    LearnedFeelingReporter,
    ContinuousAxes,
)
from scripts.internal_signal_extractor import InternalSignals, InternalSignalExtractor


# =============================================================================
# CAUSAL INFLUENCE METRICS (THE GOLD STANDARD)
# =============================================================================

def compute_kl_divergence(p_logits: torch.Tensor, q_logits: torch.Tensor) -> float:
    """
    Compute KL(P || Q) where P and Q are softmax distributions from logits.

    This measures how much the distribution changes when FEEL is on vs off.
    Higher KL = FEEL has more influence on predictions.
    """
    p = F.softmax(p_logits, dim=-1)
    q = F.softmax(q_logits, dim=-1)

    # KL(P || Q) = sum(P * log(P/Q))
    kl = F.kl_div(
        F.log_softmax(q_logits, dim=-1),
        p,
        reduction='batchmean'
    )
    return kl.item()


def compute_delta_logit(
    logits_on: torch.Tensor,
    logits_off: torch.Tensor,
    token_id: int,
) -> float:
    """
    Compute change in log-probability of a specific token.

    Δlogit = log p(token | FEEL on) - log p(token | FEEL off)

    Positive = FEEL increases probability of this token
    Negative = FEEL decreases probability
    """
    log_p_on = F.log_softmax(logits_on, dim=-1)
    log_p_off = F.log_softmax(logits_off, dim=-1)

    delta = log_p_on[0, token_id].item() - log_p_off[0, token_id].item()
    return delta


# =============================================================================
# LOCAL ATTENTION METRIC (CORRECTED)
# =============================================================================

class LocalAttentionMetric:
    """
    Measures attention to FEEL token from Q/K hooks.

    IMPORTANT: This only sees NEW K tokens (not full KV cache), so:
    - In decode step: K typically has ~2 positions (FEEL + current token)
    - Random baseline is 1/seq_k_seen, NOT 1/total_context

    This is a LOCAL attention share metric, not global attention.
    """

    def __init__(self, model: nn.Module, n_layers_sample: int = 3):
        self.model = model
        self.n_layers_sample = n_layers_sample
        self.q_outputs = {}
        self.k_outputs = {}
        self.handles = []
        self.scores = []
        self.seq_k_lengths = []  # Track actual K sequence lengths

    def _find_attention_layers(self):
        """Find Q and K projection layers."""
        q_layers = []
        k_layers = []

        for name, module in self.model.named_modules():
            if any(q in name.lower() for q in ['q_proj', 'query', 'q_lin']):
                if isinstance(module, nn.Linear):
                    q_layers.append((name, module))
            elif any(k in name.lower() for k in ['k_proj', 'key', 'k_lin']):
                if isinstance(module, nn.Linear):
                    k_layers.append((name, module))

        return q_layers, k_layers

    def register_hooks(self):
        """Register forward hooks on Q and K layers."""
        q_layers, k_layers = self._find_attention_layers()

        if not q_layers or not k_layers:
            return False

        n = len(q_layers)
        indices = [0, n // 2, n - 1] if n > self.n_layers_sample else list(range(n))

        for idx in indices:
            q_name, q_layer = q_layers[idx]
            k_name, k_layer = k_layers[idx]

            def q_hook(module, input, output, layer_idx=idx):
                self.q_outputs[layer_idx] = output.detach()

            def k_hook(module, input, output, layer_idx=idx):
                self.k_outputs[layer_idx] = output.detach()

            self.handles.append(q_layer.register_forward_hook(q_hook))
            self.handles.append(k_layer.register_forward_hook(k_hook))

        return True

    def compute(self, feel_position: int = 0) -> Tuple[float, float, int]:
        """
        Compute local attention share to FEEL token.

        Returns:
            (attention_share, correct_baseline, seq_k_length)
        """
        if not self.q_outputs or not self.k_outputs:
            return 0.0, 0.5, 2

        attention_scores = []
        seq_k_len = 2  # Default

        for layer_idx in self.q_outputs:
            if layer_idx not in self.k_outputs:
                continue

            q = self.q_outputs[layer_idx].float()
            k = self.k_outputs[layer_idx].float()

            try:
                if q.dim() == 3 and k.dim() == 3:
                    batch_q, seq_q, dim_q = q.shape
                    batch_k, seq_k, dim_k = k.shape
                    seq_k_len = seq_k  # This is the ACTUAL K length seen by hook

                    q_last = q[:, -1:, :]

                    if dim_q != dim_k:
                        min_dim = min(dim_q, dim_k)
                        q_last = q_last[:, :, :min_dim]
                        k_proj = k[:, :, :min_dim]
                    else:
                        k_proj = k

                    d_k = k_proj.size(-1)
                    scale = (d_k ** -0.5)
                    attn_logits = torch.bmm(q_last, k_proj.transpose(-2, -1)) * scale
                    attn_probs = F.softmax(attn_logits, dim=-1)

                    if attn_probs.size(-1) > feel_position:
                        attn_to_feel = attn_probs[0, 0, feel_position].item()
                        attention_scores.append(attn_to_feel)

            except Exception:
                continue

        avg_score = np.mean(attention_scores) if attention_scores else 0.0
        correct_baseline = 1.0 / seq_k_len if seq_k_len > 0 else 0.5

        self.scores.append(avg_score)
        self.seq_k_lengths.append(seq_k_len)

        # Clear for next step
        self.q_outputs.clear()
        self.k_outputs.clear()

        return avg_score, correct_baseline, seq_k_len

    def remove_hooks(self):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        self.q_outputs.clear()
        self.k_outputs.clear()

    def reset(self):
        self.scores = []
        self.seq_k_lengths = []
        self.q_outputs.clear()
        self.k_outputs.clear()


# =============================================================================
# SALIENCY METRIC (GRADIENT-BASED)
# =============================================================================

def compute_feel_saliency(
    model: nn.Module,
    feel_embed: torch.Tensor,
    logits: torch.Tensor,
    token_id: int,
) -> float:
    """
    Compute ||∂ log p(token) / ∂ FEEL_embed||.

    This measures how sensitive the model's prediction is to FEEL.
    Higher saliency = FEEL embedding has more influence.

    Note: Requires feel_embed to have requires_grad=True.
    """
    if feel_embed is None or not feel_embed.requires_grad:
        return 0.0

    try:
        # Get log probability of chosen token
        log_prob = F.log_softmax(logits[:, -1, :], dim=-1)[0, token_id]

        # Compute gradient
        grad = torch.autograd.grad(
            log_prob,
            feel_embed,
            retain_graph=True,
            allow_unused=True,
        )[0]

        if grad is not None:
            saliency = grad.norm().item()
            return saliency
        return 0.0

    except Exception:
        return 0.0


# =============================================================================
# EXPERIMENT RESULTS DATACLASS
# =============================================================================

@dataclass
class ExperimentResult:
    """Structured result for each experiment."""
    experiment_name: str
    timestamp: str
    config: Dict[str, Any]
    metrics: Dict[str, Any]
    raw_data: Optional[List[Dict]] = None
    conclusion: str = ""


# =============================================================================
# FEEL-INTEGRATED MODEL WRAPPER
# =============================================================================

class FEELIntegratedModel(nn.Module):
    """
    Wraps a transformer model with FEEL token injection.

    Supports:
    - Causal influence measurement (FEEL on vs off comparison)
    - Saliency computation (gradient-based)
    - Local attention measurement (Q/K hooks)
    """

    def __init__(
        self,
        model: nn.Module,
        tokenizer,
        z_dim: int = 64,
        injection_strength: float = 1.0,
        device: str = "cuda",
    ):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.injection_strength = injection_strength

        # Get model's hidden dimension
        if hasattr(model.config, 'hidden_size'):
            self.hidden_dim = model.config.hidden_size
        else:
            self.hidden_dim = 1536

        model_dtype = next(model.parameters()).dtype

        # FEEL components
        self.z_feel_model = PredictiveZFeel(
            sensor_dim=12,
            z_dim=z_dim,
            forecast_horizon=8,
            num_codes=64,
        ).to(device=device, dtype=model_dtype)

        self.feel_stream = FEELTokenStream(
            z_dim=z_dim,
            embed_dim=self.hidden_dim,
            n_feel_tokens=1,
            update_rate=1,
        ).to(device=device, dtype=model_dtype)

        self.reporter = LearnedFeelingReporter(
            axes_dim=8,
            hidden_dim=64,
        ).to(device=device, dtype=model_dtype)

        # Signal extractor
        self.signal_extractor = InternalSignalExtractor(model, device=device)

        # Local attention metric
        self.local_attention = LocalAttentionMetric(model, n_layers_sample=3)

        # State tracking
        self.current_z_feel = None
        self.current_axes = None
        self.current_signals = None
        self.z_feel_trajectory = []
        self.sensors_log = []

        # Causal metrics
        self.kl_divergences = []
        self.delta_logits = []
        self.saliencies = []

    def reset(self):
        """Reset state for new sequence."""
        self.z_feel_model.reset_state()
        model_dtype = next(self.z_feel_model.parameters()).dtype
        self.z_feel_model.h = self.z_feel_model.h.to(device=self.device, dtype=model_dtype)
        self.feel_stream.reset()
        self.signal_extractor.reset()
        self.local_attention.reset()

        self.current_z_feel = None
        self.current_axes = None
        self.current_signals = None
        self.z_feel_trajectory = []
        self.sensors_log = []
        self.kl_divergences = []
        self.delta_logits = []
        self.saliencies = []

    def extract_signals(
        self,
        logits: torch.Tensor,
        input_length: int = 0,
        chosen_token_id: Optional[int] = None,
    ) -> torch.Tensor:
        """Extract real signals using InternalSignalExtractor."""
        signals = self.signal_extractor.extract(
            logits=logits,
            attentions=None,
            hidden_states=None,
            input_length=input_length,
        )
        self.current_signals = signals

        # Compute surprisal
        surprisal = 0.0
        if chosen_token_id is not None:
            logits_f32 = logits[:, -1, :].float()
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

        return sensors

    def update_z_feel(self, sensors: torch.Tensor) -> torch.Tensor:
        """Update z_feel state from sensors."""
        model_dtype = next(self.z_feel_model.parameters()).dtype
        sensors = sensors.to(dtype=model_dtype)

        with torch.no_grad():
            out = self.z_feel_model(sensors)

        self.current_z_feel = out['z_feel']
        self.current_axes = out['axes']
        self.z_feel_trajectory.append(self.current_z_feel.detach().cpu().numpy())

        return self.current_z_feel

    def inject_feel_into_embeds(
        self,
        input_embeds: torch.Tensor,
        require_grad: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Inject FEEL token embedding into input embeddings.

        Returns:
            (augmented_embeds, feel_embed) - feel_embed for saliency computation
        """
        if self.current_z_feel is None or self.injection_strength == 0:
            return input_embeds, None

        # Get FEEL embedding (with grad if needed for saliency)
        z_feel = self.current_z_feel
        if require_grad:
            z_feel = z_feel.detach().clone().requires_grad_(True)

        feel_embed = self.feel_stream(z_feel, return_raw=True)
        feel_embed = feel_embed.to(dtype=input_embeds.dtype, device=input_embeds.device)
        feel_embed = feel_embed * self.injection_strength

        # Prepend FEEL embedding
        augmented = torch.cat([feel_embed, input_embeds], dim=1)

        return augmented, feel_embed if require_grad else None

    def forward_with_feel(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor = None,
        past_key_values = None,
        compute_saliency: bool = False,
    ):
        """Forward pass with FEEL token injection."""
        input_embeds = self.model.get_input_embeddings()(input_ids)

        feel_embed = None
        if self.current_z_feel is not None and self.injection_strength > 0:
            input_embeds, feel_embed = self.inject_feel_into_embeds(
                input_embeds,
                require_grad=compute_saliency,
            )

            if attention_mask is not None:
                feel_mask = torch.ones(
                    attention_mask.size(0), 1,
                    device=attention_mask.device,
                    dtype=attention_mask.dtype,
                )
                attention_mask = torch.cat([feel_mask, attention_mask], dim=1)

        outputs = self.model(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=True,
        )

        return outputs, feel_embed

    def forward_without_feel(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor = None,
        past_key_values = None,
    ):
        """Forward pass WITHOUT FEEL injection (for causal comparison)."""
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=True,
        )
        return outputs


# =============================================================================
# FEEL EXPERIMENTS CLASS
# =============================================================================

class FEELExperiments:
    """Run all FEEL integration experiments with corrected metrics."""

    def __init__(
        self,
        model_id: str = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
        device: str = "cuda",
        output_dir: str = "results/feel_experiments",
    ):
        self.model_id = model_id
        self.device = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        print(f"Loading model: {model_id}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.base_model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            device_map=device,
            trust_remote_code=True,
        )
        self.base_model.eval()

        self.results = []

    def _strip_feel_from_cache(self, past_key_values, n_feel_tokens: int = 1):
        """Strip FEEL tokens from KV cache to prevent accumulation."""
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

    # =========================================================================
    # EXPERIMENT 1: CAUSAL INFLUENCE (THE GOLD STANDARD)
    # =========================================================================

    def experiment_causal_influence(self) -> ExperimentResult:
        """
        THE GOLD STANDARD TEST: Does FEEL actually change predictions?

        For each generation step:
        1. Run forward with FEEL on → get logits_on
        2. Run forward with FEEL off (same context) → get logits_off
        3. Compute KL(on || off) and Δlogit for chosen token

        If KL > 0 consistently, FEEL has genuine causal influence.
        """
        print("\n" + "="*60)
        print("EXPERIMENT 1: Causal Influence (KL/Δlogit)")
        print("="*60)

        model = FEELIntegratedModel(
            self.base_model, self.tokenizer,
            injection_strength=1.0,
            device=self.device,
        )

        prompts = [
            "Explain quantum computing in simple terms.",
            "What is 17 * 23? Think step by step.",
            "Write a haiku about artificial intelligence.",
            "Describe the feeling of uncertainty.",
        ]

        results = []

        for prompt in prompts:
            model.reset()

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
            past_key_values_on = None
            past_key_values_off = None

            kl_divs = []
            delta_logits = []

            for step in range(32):
                with torch.no_grad():
                    # Forward WITH FEEL
                    if step == 0:
                        outputs_on, _ = model.forward_with_feel(
                            input_ids=generated_ids,
                            attention_mask=attention_mask,
                        )
                        # Also get FEEL-off baseline for same context
                        outputs_off = model.forward_without_feel(
                            input_ids=generated_ids,
                            attention_mask=attention_mask,
                        )
                    else:
                        outputs_on, _ = model.forward_with_feel(
                            input_ids=generated_ids[:, -1:],
                            attention_mask=attention_mask,
                            past_key_values=past_key_values_on,
                        )
                        outputs_off = model.forward_without_feel(
                            input_ids=generated_ids[:, -1:],
                            attention_mask=attention_mask,
                            past_key_values=past_key_values_off,
                        )

                    logits_on = outputs_on.logits[:, -1, :]
                    logits_off = outputs_off.logits[:, -1, :]

                    # Compute causal metrics
                    kl = compute_kl_divergence(logits_on, logits_off)

                    # Sample next token (from FEEL-on model)
                    next_token = logits_on.argmax(dim=-1, keepdim=True)
                    next_token_id = next_token.item()

                    delta = compute_delta_logit(logits_on, logits_off, next_token_id)

                    kl_divs.append(kl)
                    delta_logits.append(delta)

                    # Update caches
                    if step > 0 and model.injection_strength > 0:
                        past_key_values_on = self._strip_feel_from_cache(
                            outputs_on.past_key_values, n_feel_tokens=1
                        )
                    else:
                        past_key_values_on = outputs_on.past_key_values
                    past_key_values_off = outputs_off.past_key_values

                    # Update z_feel from sensors
                    sensors = model.extract_signals(
                        outputs_on.logits, input_length, next_token_id
                    )
                    model.sensors_log.append(sensors.detach().cpu())
                    model.update_z_feel(sensors)

                    # Update generation state
                    generated_ids = torch.cat([generated_ids, next_token], dim=-1)
                    attention_mask = torch.cat([
                        attention_mask,
                        torch.ones((1, 1), device=self.device, dtype=attention_mask.dtype)
                    ], dim=1)

                    if next_token_id == self.tokenizer.eos_token_id:
                        break

            avg_kl = np.mean(kl_divs) if kl_divs else 0
            avg_delta = np.mean(delta_logits) if delta_logits else 0
            max_kl = np.max(kl_divs) if kl_divs else 0

            results.append({
                'prompt': prompt[:50],
                'avg_kl_divergence': float(avg_kl),
                'max_kl_divergence': float(max_kl),
                'avg_delta_logit': float(avg_delta),
                'n_steps': len(kl_divs),
            })

            print(f"Prompt: {prompt[:40]}...")
            print(f"  Avg KL divergence: {avg_kl:.6f}")
            print(f"  Max KL divergence: {max_kl:.6f}")
            print(f"  Avg Δlogit: {avg_delta:.6f}")

        overall_avg_kl = np.mean([r['avg_kl_divergence'] for r in results])
        overall_max_kl = np.max([r['max_kl_divergence'] for r in results])

        # KL > 0.001 indicates meaningful influence
        has_influence = overall_avg_kl > 0.001

        conclusion = (
            f"FEEL {'HAS' if has_influence else 'has NO'} causal influence. "
            f"Avg KL={overall_avg_kl:.6f}, Max KL={overall_max_kl:.6f}"
        )

        return ExperimentResult(
            experiment_name="causal_influence",
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
            config={
                'injection_strength': 1.0,
                'prompts': len(prompts),
                'metric': 'KL divergence (FEEL on vs off)',
            },
            metrics={
                'avg_kl_divergence': float(overall_avg_kl),
                'max_kl_divergence': float(overall_max_kl),
                'has_causal_influence': has_influence,
                'threshold': 0.001,
            },
            raw_data=results,
            conclusion=conclusion,
        )

    # =========================================================================
    # EXPERIMENT 2: LOCAL ATTENTION (CORRECTED)
    # =========================================================================

    def experiment_local_attention(self) -> ExperimentResult:
        """
        Measure local attention share to FEEL token.

        CORRECTED: The baseline is 1/seq_k_seen (typically ~0.5 for 2 tokens),
        NOT 1/total_context (which was incorrectly ~0.03).
        """
        print("\n" + "="*60)
        print("EXPERIMENT 2: Local Attention Share (CORRECTED BASELINE)")
        print("="*60)

        model = FEELIntegratedModel(
            self.base_model, self.tokenizer,
            injection_strength=1.0,
            device=self.device,
        )

        prompts = [
            "Explain what you're thinking right now.",
            "How confident are you in your answer?",
            "Calculate 15 * 23 step by step.",
        ]

        results = []

        for prompt in prompts:
            model.reset()
            hooks_registered = model.local_attention.register_hooks()

            if not hooks_registered:
                print("  WARNING: Could not register attention hooks")
                continue

            try:
                messages = [{"role": "user", "content": prompt}]
                input_text = self.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                inputs = self.tokenizer(input_text, return_tensors="pt").to(self.device)
                input_ids = inputs["input_ids"]
                attention_mask = inputs.get("attention_mask", torch.ones_like(input_ids))
                input_length = input_ids.shape[1]

                generated_ids = input_ids.clone()
                past_key_values = None

                attention_shares = []
                baselines = []
                seq_k_lengths = []

                for step in range(32):
                    with torch.no_grad():
                        if step == 0:
                            outputs, _ = model.forward_with_feel(
                                input_ids=generated_ids,
                                attention_mask=attention_mask,
                            )
                        else:
                            outputs, _ = model.forward_with_feel(
                                input_ids=generated_ids[:, -1:],
                                attention_mask=attention_mask,
                                past_key_values=past_key_values,
                            )

                        # Compute local attention with CORRECT baseline
                        attn_share, baseline, seq_k = model.local_attention.compute(
                            feel_position=0
                        )
                        attention_shares.append(attn_share)
                        baselines.append(baseline)
                        seq_k_lengths.append(seq_k)

                        # Update cache
                        if step > 0 and model.injection_strength > 0:
                            past_key_values = self._strip_feel_from_cache(
                                outputs.past_key_values, n_feel_tokens=1
                            )
                        else:
                            past_key_values = outputs.past_key_values

                        # Update z_feel
                        next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
                        sensors = model.extract_signals(
                            outputs.logits, input_length, next_token.item()
                        )
                        model.update_z_feel(sensors)

                        generated_ids = torch.cat([generated_ids, next_token], dim=-1)
                        attention_mask = torch.cat([
                            attention_mask,
                            torch.ones((1, 1), device=self.device, dtype=attention_mask.dtype)
                        ], dim=1)

                        if next_token.item() == self.tokenizer.eos_token_id:
                            break

                avg_share = np.mean(attention_shares) if attention_shares else 0
                avg_baseline = np.mean(baselines) if baselines else 0.5
                avg_seq_k = np.mean(seq_k_lengths) if seq_k_lengths else 2
                above_baseline = avg_share > avg_baseline

                results.append({
                    'prompt': prompt[:50],
                    'avg_attention_share': float(avg_share),
                    'correct_baseline': float(avg_baseline),
                    'avg_seq_k_length': float(avg_seq_k),
                    'above_baseline': above_baseline,
                    'ratio_to_baseline': float(avg_share / avg_baseline) if avg_baseline > 0 else 0,
                })

                print(f"Prompt: {prompt[:40]}...")
                print(f"  Avg attention share: {avg_share:.4f}")
                print(f"  CORRECT baseline (1/seq_k): {avg_baseline:.4f}")
                print(f"  Avg seq_k seen by hook: {avg_seq_k:.1f}")
                print(f"  Above baseline: {above_baseline}")

            finally:
                model.local_attention.remove_hooks()

        overall_avg = np.mean([r['avg_attention_share'] for r in results]) if results else 0
        overall_baseline = np.mean([r['correct_baseline'] for r in results]) if results else 0.5

        conclusion = (
            f"Local attention share: {overall_avg:.4f} vs CORRECT baseline {overall_baseline:.4f}. "
            f"Note: This is LOCAL (new tokens only), not global attention."
        )

        return ExperimentResult(
            experiment_name="local_attention_share",
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
            config={
                'metric': 'Local attention share (Q/K hooks)',
                'note': 'Baseline is 1/seq_k_seen, NOT 1/total_context',
            },
            metrics={
                'avg_attention_share': float(overall_avg),
                'correct_baseline': float(overall_baseline),
                'ratio': float(overall_avg / overall_baseline) if overall_baseline > 0 else 0,
            },
            raw_data=results,
            conclusion=conclusion,
        )

    # =========================================================================
    # EXPERIMENT 3: FALSIFICATION BATTERY
    # =========================================================================

    def experiment_falsification_battery(self) -> ExperimentResult:
        """
        Falsification battery to prove FEEL isn't a trick.

        Tests:
        1. Sensor shuffle: Permute sensor sequence in time → should break
        2. Dim shuffle: Permute sensor dimensions → should degrade
        3. Clamp: Force z_feel constant → removes adaptation
        4. Lag sweep: Delay sensors by k steps → should degrade smoothly
        """
        print("\n" + "="*60)
        print("EXPERIMENT 3: Falsification Battery")
        print("="*60)

        prompt = "Explain the concept of entropy in information theory."

        # First: Get baseline closed-loop metrics
        print("\n--- Baseline (normal closed-loop) ---")
        baseline_result = self._run_with_sensor_manipulation(
            prompt, manipulation='none'
        )

        # Test 1: Shuffle sensors in time
        print("\n--- Sensor Shuffle (time permutation) ---")
        shuffle_result = self._run_with_sensor_manipulation(
            prompt, manipulation='time_shuffle'
        )

        # Test 2: Dim shuffle
        print("\n--- Dim Shuffle (permute dimensions) ---")
        dim_shuffle_result = self._run_with_sensor_manipulation(
            prompt, manipulation='dim_shuffle'
        )

        # Test 3: Clamp z_feel constant
        print("\n--- Clamp (constant z_feel) ---")
        clamp_result = self._run_with_sensor_manipulation(
            prompt, manipulation='clamp'
        )

        # Test 4: Lag sweep
        print("\n--- Lag Sweep (delayed sensors) ---")
        lag_results = {}
        for lag in [1, 2, 4, 8]:
            lag_result = self._run_with_sensor_manipulation(
                prompt, manipulation='lag', lag_steps=lag
            )
            lag_results[f'lag_{lag}'] = lag_result
            print(f"  Lag {lag}: KL={lag_result['avg_kl']:.6f}")

        # Compile results
        results = {
            'baseline': baseline_result,
            'time_shuffle': shuffle_result,
            'dim_shuffle': dim_shuffle_result,
            'clamp': clamp_result,
            'lag_sweep': lag_results,
        }

        # Analysis
        baseline_kl = baseline_result['avg_kl']
        shuffle_kl = shuffle_result['avg_kl']
        clamp_kl = clamp_result['avg_kl']

        # If shuffle/clamp have similar KL to baseline, the coupling isn't real
        shuffle_ratio = shuffle_kl / baseline_kl if baseline_kl > 0 else 1.0
        clamp_ratio = clamp_kl / baseline_kl if baseline_kl > 0 else 1.0

        # Real coupling should show: shuffle/clamp have DIFFERENT KL than baseline
        is_genuine = (
            abs(shuffle_ratio - 1.0) > 0.1 or  # Shuffle changes things
            abs(clamp_ratio - 1.0) > 0.1       # Clamp changes things
        )

        print(f"\n--- Falsification Summary ---")
        print(f"Baseline KL: {baseline_kl:.6f}")
        print(f"Shuffle KL: {shuffle_kl:.6f} (ratio: {shuffle_ratio:.2f})")
        print(f"Clamp KL: {clamp_kl:.6f} (ratio: {clamp_ratio:.2f})")
        print(f"Genuine coupling: {is_genuine}")

        conclusion = (
            f"Falsification: shuffle_ratio={shuffle_ratio:.2f}, clamp_ratio={clamp_ratio:.2f}. "
            f"Coupling is {'GENUINE' if is_genuine else 'SUSPECT (may be spurious)'}."
        )

        return ExperimentResult(
            experiment_name="falsification_battery",
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
            config={
                'tests': ['time_shuffle', 'dim_shuffle', 'clamp', 'lag_sweep'],
                'prompt': prompt[:50],
            },
            metrics={
                'baseline_kl': float(baseline_kl),
                'shuffle_kl': float(shuffle_kl),
                'clamp_kl': float(clamp_kl),
                'shuffle_ratio': float(shuffle_ratio),
                'clamp_ratio': float(clamp_ratio),
                'is_genuine_coupling': is_genuine,
            },
            raw_data=[results],
            conclusion=conclusion,
        )

    def _run_with_sensor_manipulation(
        self,
        prompt: str,
        manipulation: str = 'none',
        lag_steps: int = 0,
    ) -> Dict[str, float]:
        """Run generation with manipulated sensors and measure KL."""
        model = FEELIntegratedModel(
            self.base_model, self.tokenizer,
            injection_strength=1.0,
            device=self.device,
        )
        model.reset()

        messages = [{"role": "user", "content": prompt}]
        input_text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(input_text, return_tensors="pt").to(self.device)
        input_ids = inputs["input_ids"]
        attention_mask = inputs.get("attention_mask", torch.ones_like(input_ids))
        input_length = input_ids.shape[1]

        generated_ids = input_ids.clone()
        past_key_values_on = None
        past_key_values_off = None

        kl_divs = []
        sensor_history = []
        dim_permutation = None

        if manipulation == 'dim_shuffle':
            dim_permutation = torch.randperm(12)

        clamped_z_feel = None

        for step in range(32):
            with torch.no_grad():
                # Forward WITH FEEL
                if step == 0:
                    outputs_on, _ = model.forward_with_feel(
                        input_ids=generated_ids,
                        attention_mask=attention_mask,
                    )
                    outputs_off = model.forward_without_feel(
                        input_ids=generated_ids,
                        attention_mask=attention_mask,
                    )
                else:
                    outputs_on, _ = model.forward_with_feel(
                        input_ids=generated_ids[:, -1:],
                        attention_mask=attention_mask,
                        past_key_values=past_key_values_on,
                    )
                    outputs_off = model.forward_without_feel(
                        input_ids=generated_ids[:, -1:],
                        attention_mask=attention_mask,
                        past_key_values=past_key_values_off,
                    )

                logits_on = outputs_on.logits[:, -1, :]
                logits_off = outputs_off.logits[:, -1, :]

                kl = compute_kl_divergence(logits_on, logits_off)
                kl_divs.append(kl)

                next_token = logits_on.argmax(dim=-1, keepdim=True)
                next_token_id = next_token.item()

                # Update caches
                if step > 0 and model.injection_strength > 0:
                    past_key_values_on = self._strip_feel_from_cache(
                        outputs_on.past_key_values, n_feel_tokens=1
                    )
                else:
                    past_key_values_on = outputs_on.past_key_values
                past_key_values_off = outputs_off.past_key_values

                # Extract sensors
                sensors = model.extract_signals(
                    outputs_on.logits, input_length, next_token_id
                )
                sensor_history.append(sensors.detach().cpu())

                # Apply manipulation
                if manipulation == 'time_shuffle' and len(sensor_history) > 1:
                    # Use a random past sensor
                    idx = np.random.randint(0, len(sensor_history))
                    sensors = sensor_history[idx].to(self.device)

                elif manipulation == 'dim_shuffle' and dim_permutation is not None:
                    sensors = sensors[dim_permutation]

                elif manipulation == 'clamp':
                    if clamped_z_feel is None:
                        # Use first z_feel and clamp it
                        model.update_z_feel(sensors)
                        clamped_z_feel = model.current_z_feel.clone()
                    # Set clamped z_feel but DON'T continue - rollout must proceed normally
                    model.current_z_feel = clamped_z_feel
                    # Skip the update_z_feel below, but keep token append + attention mask growth

                elif manipulation == 'lag' and lag_steps > 0:
                    if step >= lag_steps:
                        sensors = sensor_history[step - lag_steps].to(self.device)

                # Update z_feel (unless clamped)
                if manipulation != 'clamp':
                    model.update_z_feel(sensors)

                generated_ids = torch.cat([generated_ids, next_token], dim=-1)
                attention_mask = torch.cat([
                    attention_mask,
                    torch.ones((1, 1), device=self.device, dtype=attention_mask.dtype)
                ], dim=1)

                if next_token_id == self.tokenizer.eos_token_id:
                    break

        return {
            'avg_kl': float(np.mean(kl_divs)) if kl_divs else 0,
            'max_kl': float(np.max(kl_divs)) if kl_divs else 0,
            'n_steps': len(kl_divs),
        }

    # =========================================================================
    # EXPERIMENT 4: MI PROXY (z_feel predicts future entropy)
    # =========================================================================

    def experiment_mi_proxy(self) -> ExperimentResult:
        """
        Mutual information proxy: Can z_feel predict future entropy
        better than raw sensors alone?

        This tests whether z_feel captures useful predictive information.
        """
        print("\n" + "="*60)
        print("EXPERIMENT 4: MI Proxy (z_feel predicts future entropy)")
        print("="*60)

        model = FEELIntegratedModel(
            self.base_model, self.tokenizer,
            injection_strength=1.0,
            device=self.device,
        )

        prompt = "Explain the difference between supervised and unsupervised learning, then give examples of each."
        model.reset()

        messages = [{"role": "user", "content": prompt}]
        input_text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(input_text, return_tensors="pt").to(self.device)
        input_ids = inputs["input_ids"]
        attention_mask = inputs.get("attention_mask", torch.ones_like(input_ids))
        input_length = input_ids.shape[1]

        generated_ids = input_ids.clone()
        past_key_values = None

        z_feel_history = []
        sensor_history = []
        entropy_history = []

        for step in range(64):
            with torch.no_grad():
                if step == 0:
                    outputs, _ = model.forward_with_feel(
                        input_ids=generated_ids,
                        attention_mask=attention_mask,
                    )
                else:
                    outputs, _ = model.forward_with_feel(
                        input_ids=generated_ids[:, -1:],
                        attention_mask=attention_mask,
                        past_key_values=past_key_values,
                    )

                logits = outputs.logits[:, -1, :]

                # Compute actual entropy
                probs = F.softmax(logits.float(), dim=-1)
                entropy = -(probs * torch.log(probs.clamp(min=1e-10))).sum(dim=-1).item()
                entropy_history.append(entropy)

                next_token = logits.argmax(dim=-1, keepdim=True)

                if step > 0 and model.injection_strength > 0:
                    past_key_values = self._strip_feel_from_cache(
                        outputs.past_key_values, n_feel_tokens=1
                    )
                else:
                    past_key_values = outputs.past_key_values

                sensors = model.extract_signals(outputs.logits, input_length, next_token.item())
                model.update_z_feel(sensors)

                sensor_history.append(sensors.detach().cpu().numpy())
                z_feel_history.append(model.current_z_feel.detach().cpu().numpy().flatten())

                generated_ids = torch.cat([generated_ids, next_token], dim=-1)
                attention_mask = torch.cat([
                    attention_mask,
                    torch.ones((1, 1), device=self.device, dtype=attention_mask.dtype)
                ], dim=1)

                if next_token.item() == self.tokenizer.eos_token_id:
                    break

        # Compute prediction correlations
        # Can z_feel[t] predict entropy[t+k]?
        n = len(entropy_history)

        if n < 10:
            return ExperimentResult(
                experiment_name="mi_proxy",
                timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
                config={},
                metrics={'error': 'insufficient_data'},
                conclusion="Insufficient data for MI analysis",
            )

        z_feel_arr = np.array(z_feel_history[:-1])  # t=0 to n-2
        sensor_arr = np.array(sensor_history[:-1])
        future_entropy = np.array(entropy_history[1:])  # t=1 to n-1

        # Correlation: z_feel with future entropy
        # Use mean of z_feel dimensions for simple proxy
        z_feel_mean = z_feel_arr.mean(axis=1)
        sensor_mean = sensor_arr.mean(axis=1)

        z_feel_corr = np.corrcoef(z_feel_mean, future_entropy)[0, 1]
        sensor_corr = np.corrcoef(sensor_mean, future_entropy)[0, 1]

        # Also try first z_feel dimension (often most informative)
        z_feel_d0_corr = np.corrcoef(z_feel_arr[:, 0], future_entropy)[0, 1]

        print(f"Correlation with future entropy:")
        print(f"  z_feel (mean): {z_feel_corr:.4f}")
        print(f"  z_feel (dim 0): {z_feel_d0_corr:.4f}")
        print(f"  sensors (mean): {sensor_corr:.4f}")

        # z_feel should be better than raw sensors if it's learning something
        z_feel_better = abs(z_feel_corr) > abs(sensor_corr)

        conclusion = (
            f"z_feel correlation: {z_feel_corr:.4f}, sensor correlation: {sensor_corr:.4f}. "
            f"z_feel is {'BETTER' if z_feel_better else 'NOT better'} at predicting future entropy."
        )

        return ExperimentResult(
            experiment_name="mi_proxy",
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
            config={
                'prompt': prompt[:50],
                'n_tokens': n,
            },
            metrics={
                'z_feel_mean_corr': float(z_feel_corr) if not np.isnan(z_feel_corr) else 0,
                'z_feel_d0_corr': float(z_feel_d0_corr) if not np.isnan(z_feel_d0_corr) else 0,
                'sensor_mean_corr': float(sensor_corr) if not np.isnan(sensor_corr) else 0,
                'z_feel_better': z_feel_better,
            },
            raw_data=[{
                'entropy_trajectory': entropy_history,
            }],
            conclusion=conclusion,
        )

    # =========================================================================
    # EXPERIMENT 5: INJECTION STRENGTH SWEEP (STABILITY)
    # =========================================================================

    def experiment_injection_sweep(self) -> ExperimentResult:
        """Test stability across injection strengths."""
        print("\n" + "="*60)
        print("EXPERIMENT 5: Injection Strength Sweep")
        print("="*60)

        strengths = [0, 0.25, 0.5, 1.0, 2.0]
        prompt = "Describe your current state of mind."

        results = []
        for g in strengths:
            print(f"\nTesting g={g}...")
            model = FEELIntegratedModel(
                self.base_model, self.tokenizer,
                injection_strength=g,
                device=self.device,
            )
            model.reset()

            messages = [{"role": "user", "content": prompt}]
            input_text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = self.tokenizer(input_text, return_tensors="pt").to(self.device)
            input_ids = inputs["input_ids"]
            attention_mask = inputs.get("attention_mask", torch.ones_like(input_ids))
            input_length = input_ids.shape[1]

            generated_ids = input_ids.clone()
            past_key_values = None
            start_time = time.perf_counter()
            output_tokens = []

            for step in range(32):
                with torch.no_grad():
                    if step == 0:
                        outputs, _ = model.forward_with_feel(
                            input_ids=generated_ids,
                            attention_mask=attention_mask,
                        )
                    else:
                        outputs, _ = model.forward_with_feel(
                            input_ids=generated_ids[:, -1:],
                            attention_mask=attention_mask,
                            past_key_values=past_key_values,
                        )

                    if step > 0 and g > 0:
                        past_key_values = self._strip_feel_from_cache(
                            outputs.past_key_values, n_feel_tokens=1
                        )
                    else:
                        past_key_values = outputs.past_key_values

                    next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
                    sensors = model.extract_signals(
                        outputs.logits, input_length, next_token.item()
                    )
                    model.update_z_feel(sensors)

                    generated_ids = torch.cat([generated_ids, next_token], dim=-1)
                    attention_mask = torch.cat([
                        attention_mask,
                        torch.ones((1, 1), device=self.device, dtype=attention_mask.dtype)
                    ], dim=1)
                    output_tokens.append(next_token.item())

                    if next_token.item() == self.tokenizer.eos_token_id:
                        break

            elapsed = time.perf_counter() - start_time
            tok_per_sec = len(output_tokens) / elapsed if elapsed > 0 else 0

            z_variance = 0.0
            if model.z_feel_trajectory:
                traj = np.array(model.z_feel_trajectory)
                z_variance = np.mean(np.var(traj, axis=0))

            results.append({
                'strength': g,
                'tokens': len(output_tokens),
                'tok_per_sec': float(tok_per_sec),
                'z_variance': float(z_variance),
            })

            print(f"  Tokens: {len(output_tokens)}, tok/s: {tok_per_sec:.1f}, z_var: {z_variance:.6f}")

        return ExperimentResult(
            experiment_name="injection_strength_sweep",
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
            config={'strengths': strengths, 'prompt': prompt[:50]},
            metrics={f'g{g}_toks': r['tokens'] for g, r in zip(strengths, results)},
            raw_data=results,
            conclusion="Sweep complete. Check z_variance for stability.",
        )

    # =========================================================================
    # RUN ALL EXPERIMENTS
    # =========================================================================

    def run_all(self) -> List[ExperimentResult]:
        """Run all experiments and save results."""
        print("\n" + "="*70)
        print("  FEEL INTEGRATION EXPERIMENTS v3.0")
        print("  With CORRECTED metrics and falsification battery")
        print("="*70)

        experiments = [
            self.experiment_causal_influence,
            self.experiment_local_attention,
            self.experiment_falsification_battery,
            self.experiment_mi_proxy,
            self.experiment_injection_sweep,
        ]

        results = []
        for exp_fn in experiments:
            try:
                result = exp_fn()
                results.append(result)
                self.results.append(result)
            except Exception as e:
                print(f"ERROR in {exp_fn.__name__}: {e}")
                import traceback
                traceback.print_exc()

        self._save_results(results)
        self._print_summary(results)

        return results

    def _save_results(self, results: List[ExperimentResult]):
        """Save results to JSON."""
        output_file = self.output_dir / "feel_experiments_v3_results.json"

        serializable = []
        for r in results:
            d = asdict(r)
            serializable.append(d)

        with open(output_file, 'w') as f:
            json.dump(serializable, f, indent=2, default=str)

        print(f"\nResults saved to: {output_file}")

    def _print_summary(self, results: List[ExperimentResult]):
        """Print experiment summary."""
        print("\n" + "="*70)
        print("  EXPERIMENT SUMMARY")
        print("="*70)

        for r in results:
            print(f"\n{r.experiment_name}:")
            print(f"  Conclusion: {r.conclusion}")
            for k, v in r.metrics.items():
                if isinstance(v, float):
                    print(f"  {k}: {v:.6f}")
                else:
                    print(f"  {k}: {v}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="FEEL Integration Experiments v3.0")
    parser.add_argument("--model", default="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-dir", default="results/feel_experiments")
    parser.add_argument("--experiment", choices=[
        'all', 'causal_influence', 'local_attention', 'falsification', 'mi_proxy', 'injection_sweep'
    ], default='all')
    args = parser.parse_args()

    experiments = FEELExperiments(
        model_id=args.model,
        device=args.device,
        output_dir=args.output_dir,
    )

    if args.experiment == 'all':
        experiments.run_all()
    elif args.experiment == 'causal_influence':
        experiments.experiment_causal_influence()
    elif args.experiment == 'local_attention':
        experiments.experiment_local_attention()
    elif args.experiment == 'falsification':
        experiments.experiment_falsification_battery()
    elif args.experiment == 'mi_proxy':
        experiments.experiment_mi_proxy()
    elif args.experiment == 'injection_sweep':
        experiments.experiment_injection_sweep()


if __name__ == "__main__":
    main()
