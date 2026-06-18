#!/usr/bin/env python3
"""
Internal Signal Extractor for Telemetry-Free Interoception

Extracts purely internal signals from the model during inference:
- Logit-space: entropy, margin, top-k mass
- Attention: entropy, sparsity, head agreement
- Activations: residual norms, saturation
- Runtime: tokens/sec, latency (self-observed)

These signals allow the model to "feel" its hardware state without
external telemetry, like a human sensing fever without a thermometer.
"""

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from collections import deque

import torch
import torch.nn.functional as F
import numpy as np


@dataclass
class InternalSignals:
    """Internal signals extracted from a single forward pass."""

    # Logit-space signals
    logit_entropy: float = 0.0          # H(p) = -sum(p * log(p))
    logit_margin: float = 0.0           # p_top1 - p_top2
    top_k_mass: float = 0.0             # sum of top-5 probabilities
    logit_temperature: float = 0.0      # Effective temperature from logit spread

    # Attention signals (averaged across layers/heads)
    attention_entropy: float = 0.0       # How spread out is attention
    attention_sparsity: float = 0.0      # Fraction of near-zero weights
    head_agreement: float = 0.0          # Consensus across attention heads
    max_attention_mass: float = 0.0      # Max attention on any single token

    # Activation signals
    residual_norm_mean: float = 0.0      # Mean ||h|| across layers
    residual_norm_std: float = 0.0       # Std of norms (instability indicator)
    activation_magnitude: float = 0.0    # Mean |activation|
    saturation_ratio: float = 0.0        # Fraction near ±1 (for tanh/sigmoid)

    # Runtime signals (self-observed)
    tokens_per_second: float = 0.0       # Throughput
    time_per_token_ms: float = 0.0       # Latency
    kv_cache_tokens: int = 0             # Context length
    generation_depth: int = 0            # Tokens generated so far

    # Derived signals
    uncertainty_score: float = 0.0       # Combined uncertainty metric
    stress_indicator: float = 0.0        # Inferred hardware stress

    def to_vector(self) -> List[float]:
        """Convert to normalized feature vector for student model."""
        return [
            # Logit signals (normalized)
            min(self.logit_entropy / 5.0, 1.0),       # Entropy ~0-5 for vocab
            self.logit_margin,                         # Already 0-1
            self.top_k_mass,                           # Already 0-1
            min(self.logit_temperature / 2.0, 1.0),   # Normalize temp

            # Attention signals
            min(self.attention_entropy / 5.0, 1.0),
            self.attention_sparsity,
            self.head_agreement,
            self.max_attention_mass,

            # Activation signals
            min(self.residual_norm_mean / 100.0, 1.0),
            min(self.residual_norm_std / 10.0, 1.0),
            min(self.activation_magnitude / 10.0, 1.0),
            self.saturation_ratio,

            # Runtime signals (normalized to reasonable ranges)
            min(self.tokens_per_second / 100.0, 1.0),
            min(self.time_per_token_ms / 100.0, 1.0),
            min(self.kv_cache_tokens / 4096.0, 1.0),
            min(self.generation_depth / 256.0, 1.0),

            # Derived
            self.uncertainty_score,
            self.stress_indicator,
        ]

    @staticmethod
    def vector_dim() -> int:
        return 18


class InternalSignalExtractor:
    """
    Extracts internal signals from a HuggingFace model during inference.

    Usage:
        extractor = InternalSignalExtractor(model)

        # During generation loop:
        signals = extractor.extract(
            input_ids=input_ids,
            outputs=model_outputs,
            attentions=attentions,  # if output_attentions=True
            hidden_states=hidden_states,  # if output_hidden_states=True
        )
    """

    def __init__(
        self,
        model: torch.nn.Module,
        history_len: int = 10,
        device: str = "cuda",
    ):
        self.model = model
        self.device = device
        self.history_len = history_len

        # Rolling history for computing derivatives/trends
        self.entropy_history = deque(maxlen=history_len)
        self.latency_history = deque(maxlen=history_len)
        self.norm_history = deque(maxlen=history_len)

        # Timing state
        self._last_token_time = None
        self._tokens_generated = 0
        self._generation_start = None

    def reset(self):
        """Reset state for new generation."""
        self.entropy_history.clear()
        self.latency_history.clear()
        self.norm_history.clear()
        self._last_token_time = None
        self._tokens_generated = 0
        self._generation_start = time.perf_counter()

    def extract(
        self,
        logits: torch.Tensor,
        attentions: Optional[Tuple[torch.Tensor, ...]] = None,
        hidden_states: Optional[Tuple[torch.Tensor, ...]] = None,
        input_length: int = 0,
    ) -> InternalSignals:
        """
        Extract internal signals from model outputs.

        Args:
            logits: [batch, seq, vocab] output logits
            attentions: Tuple of attention weights per layer
            hidden_states: Tuple of hidden states per layer
            input_length: Original input length (for KV cache estimate)

        Returns:
            InternalSignals dataclass
        """
        signals = InternalSignals()

        # Timing
        now = time.perf_counter()
        if self._last_token_time is not None:
            token_time_ms = (now - self._last_token_time) * 1000
            self.latency_history.append(token_time_ms)
            signals.time_per_token_ms = token_time_ms
        self._last_token_time = now
        self._tokens_generated += 1

        if self._generation_start is not None:
            elapsed = now - self._generation_start
            if elapsed > 0:
                signals.tokens_per_second = self._tokens_generated / elapsed

        signals.generation_depth = self._tokens_generated
        signals.kv_cache_tokens = input_length + self._tokens_generated

        # Extract logit signals (last token)
        if logits is not None:
            signals = self._extract_logit_signals(logits, signals)

        # Extract attention signals
        if attentions is not None:
            signals = self._extract_attention_signals(attentions, signals)

        # Extract activation signals
        if hidden_states is not None:
            signals = self._extract_activation_signals(hidden_states, signals)

        # Compute derived signals
        signals = self._compute_derived_signals(signals)

        return signals

    def _extract_logit_signals(
        self,
        logits: torch.Tensor,
        signals: InternalSignals,
    ) -> InternalSignals:
        """Extract signals from output logits."""
        # Get last token logits
        last_logits = logits[:, -1, :]  # [batch, vocab]

        # Softmax probabilities
        probs = F.softmax(last_logits, dim=-1)

        # Entropy: H(p) = -sum(p * log(p))
        log_probs = F.log_softmax(last_logits, dim=-1)
        entropy = -torch.sum(probs * log_probs, dim=-1).mean().item()
        signals.logit_entropy = entropy
        self.entropy_history.append(entropy)

        # Top-k analysis
        top_probs, _ = torch.topk(probs, k=min(10, probs.size(-1)), dim=-1)

        # Margin: p1 - p2
        if top_probs.size(-1) >= 2:
            signals.logit_margin = (top_probs[:, 0] - top_probs[:, 1]).mean().item()
        else:
            signals.logit_margin = 1.0

        # Top-5 mass
        signals.top_k_mass = top_probs[:, :5].sum(dim=-1).mean().item()

        # Effective temperature from logit spread
        logit_std = last_logits.std(dim=-1).mean().item()
        signals.logit_temperature = logit_std / 10.0  # Normalize

        return signals

    def _extract_attention_signals(
        self,
        attentions: Tuple[torch.Tensor, ...],
        signals: InternalSignals,
    ) -> InternalSignals:
        """Extract signals from attention patterns."""
        if not attentions or len(attentions) == 0:
            return signals

        # Sample a few layers (first, middle, last)
        n_layers = len(attentions)
        sample_layers = [0, n_layers // 2, n_layers - 1]

        entropies = []
        sparsities = []
        max_masses = []
        head_agreements = []

        for layer_idx in sample_layers:
            if layer_idx >= len(attentions):
                continue

            attn = attentions[layer_idx]  # [batch, heads, seq, seq]

            if attn is None or attn.numel() == 0:
                continue

            # Get attention to last token (what influenced the prediction)
            last_attn = attn[:, :, -1, :]  # [batch, heads, seq]

            # Attention entropy per head
            attn_log = torch.log(last_attn + 1e-10)
            head_entropy = -torch.sum(last_attn * attn_log, dim=-1)  # [batch, heads]
            entropies.append(head_entropy.mean().item())

            # Sparsity: fraction of attention weights < 0.01
            sparse_frac = (last_attn < 0.01).float().mean().item()
            sparsities.append(sparse_frac)

            # Max attention mass
            max_attn = last_attn.max(dim=-1)[0].mean().item()
            max_masses.append(max_attn)

            # Head agreement: do heads attend to same tokens?
            # Measured by correlation between heads
            if last_attn.size(1) > 1:  # More than 1 head
                head_corr = self._compute_head_agreement(last_attn)
                head_agreements.append(head_corr)

        if entropies:
            signals.attention_entropy = np.mean(entropies)
        if sparsities:
            signals.attention_sparsity = np.mean(sparsities)
        if max_masses:
            signals.max_attention_mass = np.mean(max_masses)
        if head_agreements:
            signals.head_agreement = np.mean(head_agreements)

        return signals

    def _compute_head_agreement(self, attn: torch.Tensor) -> float:
        """Compute agreement between attention heads."""
        # attn: [batch, heads, seq]
        n_heads = attn.size(1)
        if n_heads < 2:
            return 1.0

        # Flatten batch dimension
        attn_flat = attn.view(-1, attn.size(-1))  # [batch*heads, seq]

        # Compute pairwise cosine similarity between heads
        attn_norm = F.normalize(attn_flat, dim=-1)
        similarity = torch.mm(attn_norm, attn_norm.t())

        # Average off-diagonal similarity
        mask = ~torch.eye(similarity.size(0), dtype=bool, device=similarity.device)
        agreement = similarity[mask].mean().item()

        return max(0.0, agreement)

    def _extract_activation_signals(
        self,
        hidden_states: Tuple[torch.Tensor, ...],
        signals: InternalSignals,
    ) -> InternalSignals:
        """Extract signals from hidden state activations."""
        if not hidden_states or len(hidden_states) == 0:
            return signals

        norms = []
        magnitudes = []
        saturations = []

        # Sample layers
        n_layers = len(hidden_states)
        sample_layers = list(range(0, n_layers, max(1, n_layers // 5)))

        for layer_idx in sample_layers:
            if layer_idx >= len(hidden_states):
                continue

            h = hidden_states[layer_idx]  # [batch, seq, hidden]

            if h is None or h.numel() == 0:
                continue

            # Last token hidden state
            h_last = h[:, -1, :]  # [batch, hidden]

            # Residual norm
            norm = torch.norm(h_last, dim=-1).mean().item()
            norms.append(norm)
            self.norm_history.append(norm)

            # Activation magnitude
            magnitude = h_last.abs().mean().item()
            magnitudes.append(magnitude)

            # Saturation (fraction near ±max for bounded activations)
            # For unbounded activations, check for very large values
            max_val = h_last.abs().max().item()
            if max_val > 0:
                saturation = (h_last.abs() > 0.9 * max_val).float().mean().item()
                saturations.append(saturation)

        if norms:
            signals.residual_norm_mean = np.mean(norms)
            signals.residual_norm_std = np.std(norms) if len(norms) > 1 else 0.0
        if magnitudes:
            signals.activation_magnitude = np.mean(magnitudes)
        if saturations:
            signals.saturation_ratio = np.mean(saturations)

        return signals

    def _compute_derived_signals(self, signals: InternalSignals) -> InternalSignals:
        """Compute derived/composite signals."""

        # Uncertainty score: combination of entropy and margin
        # High entropy + low margin = high uncertainty
        uncertainty = (
            0.4 * min(signals.logit_entropy / 3.0, 1.0) +
            0.4 * (1.0 - signals.logit_margin) +
            0.2 * (1.0 - signals.top_k_mass)
        )
        signals.uncertainty_score = min(1.0, max(0.0, uncertainty))

        # Stress indicator: inferred from runtime signals
        # Slow tokens + high latency variance = hardware stress
        stress = 0.0

        if len(self.latency_history) >= 3:
            latency_mean = np.mean(self.latency_history)
            latency_std = np.std(self.latency_history)

            # High latency = stress
            if latency_mean > 50:  # >50ms per token
                stress += 0.3
            if latency_mean > 100:
                stress += 0.3

            # High variance = unstable (thermal throttling)
            if latency_std > 20:
                stress += 0.2
            if latency_std > 50:
                stress += 0.2

        # Low throughput = stress
        if signals.tokens_per_second > 0 and signals.tokens_per_second < 10:
            stress += 0.3
        elif signals.tokens_per_second < 20:
            stress += 0.1

        # Norm instability = internal stress
        if len(self.norm_history) >= 3:
            norm_std = np.std(self.norm_history)
            if norm_std > 5:
                stress += 0.2

        signals.stress_indicator = min(1.0, stress)

        return signals


class SignalBuffer:
    """
    Accumulates internal signals over a generation sequence.

    Used to provide summary statistics for the student model.
    """

    def __init__(self, max_tokens: int = 256):
        self.max_tokens = max_tokens
        self.signals: List[InternalSignals] = []

    def add(self, signal: InternalSignals):
        """Add a signal observation."""
        if len(self.signals) < self.max_tokens:
            self.signals.append(signal)

    def clear(self):
        """Clear buffer."""
        self.signals = []

    def get_summary(self) -> InternalSignals:
        """Get summary statistics across all collected signals."""
        if not self.signals:
            return InternalSignals()

        # Aggregate each field
        summary = InternalSignals()
        n = len(self.signals)

        # Mean of each signal
        summary.logit_entropy = np.mean([s.logit_entropy for s in self.signals])
        summary.logit_margin = np.mean([s.logit_margin for s in self.signals])
        summary.top_k_mass = np.mean([s.top_k_mass for s in self.signals])
        summary.logit_temperature = np.mean([s.logit_temperature for s in self.signals])

        summary.attention_entropy = np.mean([s.attention_entropy for s in self.signals])
        summary.attention_sparsity = np.mean([s.attention_sparsity for s in self.signals])
        summary.head_agreement = np.mean([s.head_agreement for s in self.signals])
        summary.max_attention_mass = np.mean([s.max_attention_mass for s in self.signals])

        summary.residual_norm_mean = np.mean([s.residual_norm_mean for s in self.signals])
        summary.residual_norm_std = np.std([s.residual_norm_mean for s in self.signals])
        summary.activation_magnitude = np.mean([s.activation_magnitude for s in self.signals])
        summary.saturation_ratio = np.mean([s.saturation_ratio for s in self.signals])

        # Use final values for runtime signals
        summary.tokens_per_second = self.signals[-1].tokens_per_second
        summary.time_per_token_ms = np.mean([s.time_per_token_ms for s in self.signals if s.time_per_token_ms > 0])
        summary.kv_cache_tokens = self.signals[-1].kv_cache_tokens
        summary.generation_depth = self.signals[-1].generation_depth

        summary.uncertainty_score = np.mean([s.uncertainty_score for s in self.signals])
        summary.stress_indicator = np.max([s.stress_indicator for s in self.signals])  # Max stress seen

        return summary

    def get_trajectory(self) -> torch.Tensor:
        """Get full trajectory as tensor for sequence models."""
        if not self.signals:
            return torch.zeros(1, InternalSignals.vector_dim())

        vectors = [s.to_vector() for s in self.signals]
        return torch.tensor(vectors, dtype=torch.float32)


def extract_signals_during_generation(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 64,
    device: str = "cuda",
) -> Tuple[str, InternalSignals, List[InternalSignals]]:
    """
    Generate text while extracting internal signals.

    Returns:
        (generated_text, summary_signals, per_token_signals)
    """
    extractor = InternalSignalExtractor(model, device=device)
    buffer = SignalBuffer(max_tokens=max_new_tokens)

    # Encode input
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_length = inputs.input_ids.shape[1]

    extractor.reset()

    # Generate token by token to extract signals
    generated_ids = inputs.input_ids.clone()

    for _ in range(max_new_tokens):
        with torch.no_grad():
            outputs = model(
                input_ids=generated_ids,
                output_attentions=True,
                output_hidden_states=True,
                use_cache=True,
            )

        # Extract signals from this step
        signals = extractor.extract(
            logits=outputs.logits,
            attentions=outputs.attentions,
            hidden_states=outputs.hidden_states,
            input_length=input_length,
        )
        buffer.add(signals)

        # Sample next token
        next_token_logits = outputs.logits[:, -1, :]
        next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)

        # Check for EOS
        if next_token.item() == tokenizer.eos_token_id:
            break

        generated_ids = torch.cat([generated_ids, next_token], dim=-1)

    # Decode output
    generated_text = tokenizer.decode(
        generated_ids[0, input_length:],
        skip_special_tokens=True,
    )

    return generated_text, buffer.get_summary(), buffer.signals


if __name__ == "__main__":
    # Quick test
    print("Internal Signal Extractor")
    print("=" * 50)

    # Create dummy signals to test vector conversion
    signals = InternalSignals(
        logit_entropy=2.5,
        logit_margin=0.3,
        top_k_mass=0.8,
        logit_temperature=0.5,
        attention_entropy=3.0,
        attention_sparsity=0.7,
        head_agreement=0.6,
        max_attention_mass=0.4,
        residual_norm_mean=50.0,
        residual_norm_std=5.0,
        activation_magnitude=2.0,
        saturation_ratio=0.1,
        tokens_per_second=30.0,
        time_per_token_ms=33.0,
        kv_cache_tokens=512,
        generation_depth=32,
        uncertainty_score=0.4,
        stress_indicator=0.2,
    )

    vec = signals.to_vector()
    print(f"Signal vector dimension: {len(vec)}")
    print(f"Vector: {vec}")
    print(f"\nAll values in [0, 1]: {all(0 <= v <= 1 for v in vec)}")
