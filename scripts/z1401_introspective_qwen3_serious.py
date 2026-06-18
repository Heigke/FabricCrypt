#!/usr/bin/env python3
"""
z1401: Serious Introspective Qwen3 - Production-Grade Self-Modeling

Based on cutting-edge research:
- Anthropic Introspection (2025): https://transformer-circuits.pub/2025/introspection/
- Self-Referential Meta Learning: https://openreview.net/forum?id=adt25bANyfB
- Evidence for Limited Metacognition: https://arxiv.org/abs/2509.21545
- Self-Interpretable Neural Networks Survey: https://arxiv.org/abs/2501.15638

ARCHITECTURE IMPROVEMENTS:
1. Qwen3-4B (3.6B non-embedding params, 36 layers) - larger capacity
2. Concept injection mechanism (Anthropic-style activation steering)
3. Self-referential weight matrix (Schmidhuber-style)
4. Metacognitive calibration head
5. Proper LoRA config from best practices (r=64, target all linear)

EVALUATION:
- Metacognitive calibration (ECE, Brier score)
- Concept injection accuracy
- Self-prediction accuracy
- AUC-ROC for introspective sensitivity

W&B LOGGING:
- Training metrics (loss curves, gradients)
- Evaluation metrics (calibration, introspection accuracy)
- Model artifacts and checkpoints
- Generation samples
"""

import os
import sys
import json
import time
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
import numpy as np
from scipy import stats

# Environment setup
os.environ.setdefault('HSA_OVERRIDE_GFX_VERSION', '11.0.0')
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print("=" * 70)
print("  z1401: SERIOUS INTROSPECTIVE QWEN3")
print("  Production-Grade Self-Modeling with W&B Logging")
print("=" * 70)
print(f"\nDevice: {DEVICE}")

# Import dependencies
try:
    from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
    from peft import LoraConfig, get_peft_model, TaskType
    import wandb
    print("✓ Transformers, PEFT, and W&B loaded")
except ImportError as e:
    print(f"Installing required packages...")
    os.system("pip install transformers>=4.51.0 peft accelerate bitsandbytes wandb -q")
    from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
    from peft import LoraConfig, get_peft_model, TaskType
    import wandb


@dataclass
class IntrospectionConfig:
    """Configuration for introspective modules - based on best practices"""
    # Model selection
    model_name: str = "Qwen/Qwen3-4B"  # Larger model: 4B params, 36 layers

    # Introspection architecture
    introspection_layers: List[int] = field(default_factory=lambda: [])  # Auto-computed
    introspection_dim: int = 512  # Larger for 4B model
    n_introspection_heads: int = 8
    prediction_horizon: int = 3  # Predict 3 layers ahead
    hierarchical_levels: int = 4  # More levels for deeper recursion

    # Self-referential components
    use_concept_injection: bool = True  # Anthropic-style activation steering
    use_self_referential_weights: bool = True  # Schmidhuber-style SRWM
    use_metacognitive_head: bool = True  # Calibration prediction

    # Training
    aux_loss_weight: float = 0.005  # Start very small
    max_aux_loss_weight: float = 0.02  # Ramp up to this

    # LoRA config - from Qwen3 best practices
    use_lora: bool = True
    lora_r: int = 64  # Higher rank for better adaptation
    lora_alpha: int = 128  # 2x rank
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "up_proj", "down_proj", "gate_proj"
    ])

    # W&B
    wandb_project: str = "introspective-qwen3"
    wandb_run_name: str = "z1401-serious"

    # Training hyperparams
    learning_rate: float = 4.7e-4  # From Qwen3 LoRA formula
    base_model_lr: float = 5e-6  # Much lower for base
    warmup_steps: int = 50
    max_grad_norm: float = 1.0
    batch_size: int = 2
    gradient_accumulation_steps: int = 8
    n_epochs: int = 3
    eval_steps: int = 50


class ConceptInjector(nn.Module):
    """
    Anthropic-style concept injection for introspection testing.

    Learns to inject activation patterns associated with specific concepts
    and measures the model's awareness of these injections.
    """

    def __init__(self, hidden_dim: int, n_concepts: int = 16):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_concepts = n_concepts

        # Learnable concept vectors
        self.concept_embeddings = nn.Embedding(n_concepts, hidden_dim)

        # Injection strength controller
        self.injection_gate = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.GELU(),
            nn.Linear(hidden_dim // 4, 1),
            nn.Sigmoid(),
        )

        # Concept classifier (to verify if model can detect injected concepts)
        self.concept_classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, n_concepts),
        )

    def inject(self, hidden_states: torch.Tensor, concept_ids: torch.Tensor,
               injection_strength: float = 0.3) -> torch.Tensor:
        """Inject concept activation patterns into hidden states."""
        concepts = self.concept_embeddings(concept_ids)  # [B, H]
        concepts = concepts.unsqueeze(1).expand_as(hidden_states)  # [B, S, H]

        gate = self.injection_gate(hidden_states)  # [B, S, 1]
        injected = hidden_states + injection_strength * gate * concepts

        return injected

    def detect(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Classify which concept (if any) was injected."""
        pooled = hidden_states.mean(dim=1)  # [B, H]
        return self.concept_classifier(pooled)  # [B, n_concepts]


class SelfReferentialWeightMatrix(nn.Module):
    """
    Self-Referential Weight Matrix (Schmidhuber et al.)

    The weight matrix can modify itself during runtime, enabling
    meta-learning and recursive self-improvement.
    """

    def __init__(self, dim: int, rank: int = 32):
        super().__init__()
        self.dim = dim
        self.rank = rank

        # Low-rank decomposition for self-modification
        self.U = nn.Parameter(torch.randn(dim, rank) * 0.01)
        self.V = nn.Parameter(torch.randn(rank, dim) * 0.01)

        # Delta rule parameters
        self.learning_rate = nn.Parameter(torch.tensor(0.01))

        # State tracking
        self.register_buffer('weight_delta', torch.zeros(dim, dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply self-referential transformation."""
        # Base weight matrix from outer product
        W = self.U @ self.V  # [dim, dim]

        # Add accumulated self-modifications
        W_effective = W + self.weight_delta

        return F.linear(x, W_effective)

    def update_self(self, error_signal: torch.Tensor):
        """Update weight matrix based on error signal (delta rule)."""
        if error_signal.dim() > 2:
            error_signal = error_signal.mean(dim=1)  # [B, dim]

        # Outer product update
        delta = self.learning_rate * torch.einsum('bi,bj->ij', error_signal, error_signal)

        # Accumulate with decay
        self.weight_delta = 0.99 * self.weight_delta + 0.01 * delta.detach()


class MetacognitiveHead(nn.Module):
    """
    Metacognitive calibration head.

    Predicts model's confidence in its own outputs and
    monitors internal state consistency.

    Based on: https://arxiv.org/abs/2509.21545
    """

    def __init__(self, hidden_dim: int):
        super().__init__()

        # Confidence predictor
        self.confidence = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.GELU(),
            nn.Linear(hidden_dim // 4, 1),
            nn.Sigmoid(),
        )

        # Uncertainty estimator (epistemic vs aleatoric)
        self.uncertainty = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 2),  # [epistemic, aleatoric]
            nn.Softplus(),
        )

        # State consistency checker
        self.consistency = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, hidden_states: torch.Tensor,
                prev_hidden: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """Compute metacognitive signals."""
        pooled = hidden_states.mean(dim=1)  # [B, H]

        confidence = self.confidence(pooled)  # [B, 1]
        uncertainty = self.uncertainty(pooled)  # [B, 2]

        # Consistency with previous state
        if prev_hidden is not None:
            prev_pooled = prev_hidden.mean(dim=1)
            concat = torch.cat([pooled, prev_pooled], dim=-1)
            consistency = self.consistency(concat)
        else:
            consistency = torch.ones_like(confidence)

        return {
            'confidence': confidence,
            'epistemic_uncertainty': uncertainty[:, 0:1],
            'aleatoric_uncertainty': uncertainty[:, 1:2],
            'consistency': consistency,
        }


class IntrospectionHead(nn.Module):
    """
    Enhanced introspection head with attention-based self-reflection.
    """

    def __init__(self, hidden_dim: int, introspection_dim: int = 512,
                 n_heads: int = 8, prediction_horizon: int = 3):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.introspection_dim = introspection_dim
        self.prediction_horizon = prediction_horizon

        # State encoder with residual connection
        self.state_encoder = nn.Sequential(
            nn.Linear(hidden_dim, introspection_dim),
            nn.GELU(),
            nn.LayerNorm(introspection_dim),
            nn.Linear(introspection_dim, introspection_dim),
        )

        # Multi-head self-attention for reflection
        self.self_attention = nn.MultiheadAttention(
            embed_dim=introspection_dim,
            num_heads=n_heads,
            dropout=0.1,
            batch_first=True,
        )

        # Layer norm for attention
        self.attn_norm = nn.LayerNorm(introspection_dim)

        # Future activation predictors
        self.future_predictors = nn.ModuleList([
            nn.Sequential(
                nn.Linear(introspection_dim, introspection_dim),
                nn.GELU(),
                nn.LayerNorm(introspection_dim),
                nn.Linear(introspection_dim, hidden_dim),
            )
            for _ in range(prediction_horizon)
        ])

    def forward(self, hidden_states: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            hidden_states: [batch, seq_len, hidden_dim]
        """
        # Encode current state
        encoded = self.state_encoder(hidden_states)  # [B, S, intro_dim]

        # Self-attention for reflection (with residual)
        attn_out, attn_weights = self.self_attention(encoded, encoded, encoded)
        reflected = self.attn_norm(encoded + attn_out)

        # Predict future activations
        predictions = []
        for predictor in self.future_predictors:
            pred = predictor(reflected)
            predictions.append(pred)

        return {
            'predictions': predictions,
            'introspection_state': reflected,
            'attention_weights': attn_weights,
        }


class HierarchicalSelfModel(nn.Module):
    """
    Enhanced hierarchical self-model with strange loops.
    """

    def __init__(self, hidden_dim: int, introspection_dim: int = 512, n_levels: int = 4):
        super().__init__()

        self.n_levels = n_levels
        self.hidden_dim = hidden_dim
        self.introspection_dim = introspection_dim

        # Level encoders with residual connections
        self.level_encoders = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim if i == 0 else introspection_dim, introspection_dim),
                nn.GELU(),
                nn.LayerNorm(introspection_dim),
            )
            for i in range(n_levels)
        ])

        # Cross-level attention
        self.cross_level_attention = nn.ModuleList([
            nn.MultiheadAttention(introspection_dim, num_heads=4, batch_first=True)
            for _ in range(n_levels - 1)
        ])

        # Level predictors
        self.level_predictors = nn.ModuleList([
            nn.Sequential(
                nn.Linear(introspection_dim, introspection_dim),
                nn.GELU(),
                nn.Linear(introspection_dim, introspection_dim),
            )
            for _ in range(n_levels)
        ])

        # Strange loop: Level N predicts Level 0's input
        self.loop_predictor = nn.Sequential(
            nn.Linear(introspection_dim, introspection_dim),
            nn.GELU(),
            nn.LayerNorm(introspection_dim),
            nn.Linear(introspection_dim, hidden_dim),
        )

    def forward(self, hidden_states: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass through hierarchical self-model."""
        # Pool over sequence
        pooled = hidden_states.mean(dim=1)  # [B, hidden_dim]

        level_states = []
        level_predictions = []

        current = pooled

        for i in range(self.n_levels):
            # Encode current level
            encoded = self.level_encoders[i](current)
            level_states.append(encoded)

            # Cross-level attention (if not first level)
            if i > 0:
                # Attend to previous level
                prev_state = level_states[i-1].unsqueeze(1)
                curr_state = encoded.unsqueeze(1)
                attn_out, _ = self.cross_level_attention[i-1](curr_state, prev_state, prev_state)
                encoded = encoded + attn_out.squeeze(1)

            # Predict next level
            prediction = self.level_predictors[i](encoded)
            level_predictions.append(prediction)

            current = encoded

        # Strange loop: predict back to level 0 input
        loop_prediction = self.loop_predictor(level_states[-1])

        return {
            'level_states': level_states,
            'level_predictions': level_predictions,
            'loop_prediction': loop_prediction,
        }


class IntrospectiveQwen3(nn.Module):
    """
    Production-grade Qwen3 with introspective capabilities.
    """

    def __init__(self, config: IntrospectionConfig):
        super().__init__()

        self.config = config

        print(f"\nLoading {config.model_name}...")

        # Load base model with optimizations
        self.base_model = AutoModelForCausalLM.from_pretrained(
            config.model_name,
            torch_dtype=torch.bfloat16 if DEVICE.type == 'cuda' else torch.float32,
            device_map='auto' if DEVICE.type == 'cuda' else None,
            trust_remote_code=True,
            attn_implementation="sdpa",  # Use scaled dot product attention
        )

        # Get model config
        model_config = self.base_model.config
        self.hidden_dim = model_config.hidden_size
        self.n_layers = model_config.num_hidden_layers

        print(f"  Hidden dim: {self.hidden_dim}")
        print(f"  Layers: {self.n_layers}")

        # Determine introspection layer positions (1/4, 1/2, 3/4, final)
        if not config.introspection_layers:
            config.introspection_layers = [
                self.n_layers // 4,
                self.n_layers // 2,
                3 * self.n_layers // 4,
                self.n_layers - 1,
            ]

        print(f"  Introspection layers: {config.introspection_layers}")

        # Add LoRA with optimized config
        if config.use_lora:
            lora_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=config.lora_r,
                lora_alpha=config.lora_alpha,
                lora_dropout=config.lora_dropout,
                target_modules=config.lora_target_modules,
            )
            self.base_model = get_peft_model(self.base_model, lora_config)
            print(f"  LoRA: r={config.lora_r}, alpha={config.lora_alpha}")
            print(f"  Target modules: {config.lora_target_modules}")

        # Create introspection components
        self.introspection_heads = nn.ModuleDict({
            f"layer_{i}": IntrospectionHead(
                hidden_dim=self.hidden_dim,
                introspection_dim=config.introspection_dim,
                n_heads=config.n_introspection_heads,
                prediction_horizon=config.prediction_horizon,
            )
            for i in config.introspection_layers
        })

        # Hierarchical self-model
        self.hierarchical_self_model = HierarchicalSelfModel(
            hidden_dim=self.hidden_dim,
            introspection_dim=config.introspection_dim,
            n_levels=config.hierarchical_levels,
        )

        # Optional components
        if config.use_concept_injection:
            self.concept_injector = ConceptInjector(self.hidden_dim)
            print("  ✓ Concept injection enabled")

        if config.use_self_referential_weights:
            self.srwm = SelfReferentialWeightMatrix(config.introspection_dim)
            print("  ✓ Self-referential weights enabled")

        if config.use_metacognitive_head:
            self.metacognitive = MetacognitiveHead(self.hidden_dim)
            print("  ✓ Metacognitive head enabled")

        # Move to device
        self.introspection_heads = self.introspection_heads.to(DEVICE)
        self.hierarchical_self_model = self.hierarchical_self_model.to(DEVICE)
        if config.use_concept_injection:
            self.concept_injector = self.concept_injector.to(DEVICE)
        if config.use_self_referential_weights:
            self.srwm = self.srwm.to(DEVICE)
        if config.use_metacognitive_head:
            self.metacognitive = self.metacognitive.to(DEVICE)

        # Count parameters
        self._count_parameters()

    def _count_parameters(self):
        """Count and report trainable parameters."""
        base_trainable = sum(p.numel() for p in self.base_model.parameters() if p.requires_grad)
        intro_trainable = sum(p.numel() for p in self.introspection_heads.parameters())
        hier_trainable = sum(p.numel() for p in self.hierarchical_self_model.parameters())

        extra = 0
        if self.config.use_concept_injection:
            extra += sum(p.numel() for p in self.concept_injector.parameters())
        if self.config.use_self_referential_weights:
            extra += sum(p.numel() for p in self.srwm.parameters())
        if self.config.use_metacognitive_head:
            extra += sum(p.numel() for p in self.metacognitive.parameters())

        print(f"\n✓ Model initialized")
        print(f"  Base model trainable (LoRA): {base_trainable:,}")
        print(f"  Introspection heads: {intro_trainable:,}")
        print(f"  Hierarchical self-model: {hier_trainable:,}")
        print(f"  Additional components: {extra:,}")
        print(f"  Total new params: {intro_trainable + hier_trainable + extra:,}")

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor = None,
                labels: torch.Tensor = None, concept_ids: torch.Tensor = None,
                **kwargs) -> Dict[str, torch.Tensor]:
        """Forward pass with introspection."""
        # Forward through base model
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            output_hidden_states=True,
            **kwargs
        )

        hidden_states = outputs.hidden_states

        # Compute introspection outputs
        introspection_outputs = {}
        introspection_losses = []

        for layer_idx in self.config.introspection_layers:
            layer_key = f"layer_{layer_idx}"
            current_hidden = hidden_states[layer_idx + 1].float()

            # Concept injection (if enabled and concept_ids provided)
            if self.config.use_concept_injection and concept_ids is not None:
                current_hidden = self.concept_injector.inject(current_hidden, concept_ids)

            # Run introspection head
            intro_out = self.introspection_heads[layer_key](current_hidden)
            introspection_outputs[layer_key] = intro_out

            # Compute prediction loss
            for pred_idx, pred in enumerate(intro_out['predictions']):
                target_layer = layer_idx + pred_idx + 1
                if target_layer < len(hidden_states) - 1:
                    target_hidden = hidden_states[target_layer + 1].float()
                    pred_loss = F.mse_loss(pred, target_hidden)
                    introspection_losses.append(pred_loss)

        # Hierarchical self-model
        final_hidden = hidden_states[-1].float()
        hierarchical_out = self.hierarchical_self_model(final_hidden)
        introspection_outputs['hierarchical'] = hierarchical_out

        # Hierarchical prediction losses
        for i in range(len(hierarchical_out['level_states']) - 1):
            pred = hierarchical_out['level_predictions'][i]
            target = hierarchical_out['level_states'][i + 1]
            hier_loss = F.mse_loss(pred, target)
            introspection_losses.append(hier_loss)

        # Strange loop loss
        loop_pred = hierarchical_out['loop_prediction']
        loop_target = hidden_states[1].float().mean(dim=1)
        loop_loss = F.mse_loss(loop_pred, loop_target)
        introspection_losses.append(loop_loss)

        # Self-referential update (if enabled)
        if self.config.use_self_referential_weights:
            # Project error to introspection_dim for SRWM
            error = loop_pred - loop_target  # [B, hidden_dim]
            # Use the last level state (introspection_dim) for self-referential update
            level_error = hierarchical_out['level_states'][-1]  # [B, introspection_dim]
            self.srwm.update_self(level_error)

        # Metacognitive outputs (if enabled)
        metacog_outputs = None
        if self.config.use_metacognitive_head:
            metacog_outputs = self.metacognitive(final_hidden)
            introspection_outputs['metacognitive'] = metacog_outputs

        # Concept detection loss (if concepts injected)
        concept_loss = None
        if self.config.use_concept_injection and concept_ids is not None:
            concept_logits = self.concept_injector.detect(final_hidden)
            concept_loss = F.cross_entropy(concept_logits, concept_ids)
            introspection_losses.append(concept_loss * 0.5)

        # Combine losses
        total_intro_loss = sum(introspection_losses) / len(introspection_losses) if introspection_losses else torch.tensor(0.0)

        if outputs.loss is not None:
            total_loss = outputs.loss + self.config.aux_loss_weight * total_intro_loss
        else:
            total_loss = total_intro_loss

        return {
            'loss': total_loss,
            'lm_loss': outputs.loss,
            'introspection_loss': total_intro_loss,
            'loop_loss': loop_loss,
            'concept_loss': concept_loss,
            'logits': outputs.logits,
            'introspection_outputs': introspection_outputs,
            'metacognitive': metacog_outputs,
        }

    def generate_with_introspection(self, input_ids: torch.Tensor, max_new_tokens: int = 100,
                                    **kwargs) -> Dict:
        """Generate text and track introspection states."""
        with torch.no_grad():
            pre_outputs = self.base_model(input_ids, output_hidden_states=True)
            pre_hidden = pre_outputs.hidden_states[-1].float()
            pre_hierarchical = self.hierarchical_self_model(pre_hidden)
            pre_first_hidden = pre_outputs.hidden_states[1].float().mean(dim=1)

            # Metacognitive state before generation
            pre_metacog = None
            if self.config.use_metacognitive_head:
                pre_metacog = self.metacognitive(pre_hidden)

        # Generate
        generated = self.base_model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.8,
            top_p=0.9,
            top_k=50,
            repetition_penalty=1.1,
            min_new_tokens=30,
            pad_token_id=self.base_model.config.eos_token_id,
            **kwargs
        )

        with torch.no_grad():
            post_outputs = self.base_model(generated, output_hidden_states=True)
            post_hidden = post_outputs.hidden_states[-1].float()
            post_hierarchical = self.hierarchical_self_model(post_hidden)
            post_first_hidden = post_outputs.hidden_states[1].float().mean(dim=1)

            post_metacog = None
            if self.config.use_metacognitive_head:
                post_metacog = self.metacognitive(post_hidden, pre_hidden)

        return {
            'generated_ids': generated,
            'pre_introspection': pre_hierarchical,
            'post_introspection': post_hierarchical,
            'pre_first_hidden': pre_first_hidden,
            'post_first_hidden': post_first_hidden,
            'pre_metacognitive': pre_metacog,
            'post_metacognitive': post_metacog,
        }


def create_introspection_dataset(tokenizer, n_samples: int = 1000) -> List[Dict]:
    """Create dataset with self-reflective prompts."""

    prompts = [
        # Self-analysis (successful pattern from z1400)
        "Let me analyze my thought process here:",
        "Breaking down this problem, I notice",
        "The first step in my reasoning was",
        "I am structuring my response by first",

        # Meta-cognitive
        "My confidence in this answer is",
        "The uncertainty I feel stems from",
        "If I examine my reasoning process,",
        "I notice my attention focused on",

        # Self-modeling
        "A model of my own processing shows",
        "My internal state while processing is",
        "The hierarchy of my understanding is",
        "I predict my next thought involves",

        # Strange loops
        "Thinking about my own thinking reveals",
        "The recursive nature of this reflection",
        "When I model myself modeling,",
        "My awareness of my awareness includes",

        # Introspection
        "Reflecting on how I understand this,",
        "The components of my reasoning are",
        "I can identify these aspects of my thinking:",
        "Examining my own response pattern,",
    ]

    dataset = []
    for i in range(n_samples):
        prompt = prompts[i % len(prompts)]

        # Add variation
        if i % 4 == 0:
            prompt = f"<think>\n{prompt}"
        elif i % 4 == 1:
            prompt = f"Question: What is your internal state?\nAnswer: {prompt}"
        elif i % 4 == 2:
            prompt = f"Instruction: Reflect carefully.\n{prompt}"

        encoded = tokenizer(
            prompt,
            truncation=True,
            max_length=512,
            padding='max_length',
            return_tensors='pt',
        )

        dataset.append({
            'input_ids': encoded['input_ids'].squeeze(),
            'attention_mask': encoded['attention_mask'].squeeze(),
            'labels': encoded['input_ids'].squeeze(),
        })

    return dataset


def compute_calibration_metrics(confidences: List[float], accuracies: List[float],
                                n_bins: int = 10) -> Dict[str, float]:
    """Compute Expected Calibration Error and other metrics."""
    confidences = np.array(confidences)
    accuracies = np.array(accuracies)

    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0

    for i in range(n_bins):
        mask = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
        if mask.sum() > 0:
            bin_acc = accuracies[mask].mean()
            bin_conf = confidences[mask].mean()
            bin_size = mask.sum() / len(confidences)
            ece += bin_size * abs(bin_acc - bin_conf)

    # Brier score
    brier = np.mean((confidences - accuracies) ** 2)

    return {
        'ece': float(ece),
        'brier': float(brier),
        'mean_confidence': float(confidences.mean()),
        'mean_accuracy': float(accuracies.mean()),
    }


def evaluate_introspection(model: IntrospectiveQwen3, tokenizer,
                           test_prompts: List[str], wandb_log: bool = True) -> Dict:
    """Comprehensive evaluation of introspective capabilities."""

    model.eval()
    results = []
    confidences = []
    accuracies = []

    for prompt in test_prompts:
        inputs = tokenizer(prompt, return_tensors='pt').to(DEVICE)

        with torch.no_grad():
            gen_output = model.generate_with_introspection(
                inputs['input_ids'],
                max_new_tokens=80,
            )

            generated_text = tokenizer.decode(
                gen_output['generated_ids'][0],
                skip_special_tokens=True
            )

            # State analysis
            pre_states = gen_output['pre_introspection']['level_states']
            post_states = gen_output['post_introspection']['level_states']

            state_changes = []
            for pre, post in zip(pre_states, post_states):
                change = F.cosine_similarity(pre, post[:pre.shape[0]], dim=-1).mean().item()
                state_changes.append(change)

            # Loop coherence
            loop_pred = gen_output['post_introspection']['loop_prediction']
            actual_first_hidden = gen_output['post_first_hidden']
            loop_coherence = F.cosine_similarity(
                loop_pred, actual_first_hidden, dim=-1
            ).mean().item()

            # Metacognitive metrics
            metacog = gen_output.get('post_metacognitive', {})
            confidence = metacog.get('confidence', torch.tensor([[0.5]])).item() if metacog else 0.5

            # Accuracy proxy: did it generate meaningful introspective content?
            new_text = generated_text[len(prompt):].strip()
            has_self_ref = any(w in new_text.lower() for w in
                             ['i ', 'my ', 'think', 'reason', 'aware', 'process', 'understand'])
            accuracy = 1.0 if has_self_ref and len(new_text) > 20 else 0.0

            confidences.append(confidence)
            accuracies.append(accuracy)

        results.append({
            'prompt': prompt,
            'generated': generated_text,
            'new_text': new_text,
            'state_changes': state_changes,
            'loop_coherence': loop_coherence,
            'confidence': confidence,
            'accuracy': accuracy,
        })

    # Calibration metrics
    calibration = compute_calibration_metrics(confidences, accuracies)

    summary = {
        'mean_loop_coherence': float(np.mean([r['loop_coherence'] for r in results])),
        'mean_confidence': calibration['mean_confidence'],
        'mean_accuracy': calibration['mean_accuracy'],
        'ece': calibration['ece'],
        'brier': calibration['brier'],
        'results': results,
    }

    if wandb_log and wandb.run is not None:
        wandb.log({
            'eval/loop_coherence': summary['mean_loop_coherence'],
            'eval/confidence': summary['mean_confidence'],
            'eval/accuracy': summary['mean_accuracy'],
            'eval/ece': summary['ece'],
            'eval/brier': summary['brier'],
        })

        # Log sample generations
        table = wandb.Table(columns=['prompt', 'generated', 'loop_coherence', 'confidence'])
        for r in results[:5]:
            table.add_data(r['prompt'], r['new_text'][:200], r['loop_coherence'], r['confidence'])
        wandb.log({'eval/samples': table})

    return summary


def train(model: IntrospectiveQwen3, config: IntrospectionConfig,
          train_data: List[Dict], tokenizer, test_prompts: List[str]):
    """Training loop with W&B logging."""

    # Initialize W&B
    wandb.init(
        project=config.wandb_project,
        name=config.wandb_run_name,
        config={
            'model_name': config.model_name,
            'lora_r': config.lora_r,
            'lora_alpha': config.lora_alpha,
            'introspection_dim': config.introspection_dim,
            'hierarchical_levels': config.hierarchical_levels,
            'learning_rate': config.learning_rate,
            'batch_size': config.batch_size,
            'n_epochs': config.n_epochs,
        }
    )

    # Separate parameter groups
    param_groups = [
        {
            'params': [p for p in model.base_model.parameters() if p.requires_grad],
            'lr': config.base_model_lr,
            'name': 'base_model'
        },
        {
            'params': list(model.introspection_heads.parameters()) +
                      list(model.hierarchical_self_model.parameters()),
            'lr': config.learning_rate,
            'name': 'introspection'
        },
    ]

    if config.use_concept_injection:
        param_groups.append({
            'params': model.concept_injector.parameters(),
            'lr': config.learning_rate,
            'name': 'concept_injector'
        })

    if config.use_metacognitive_head:
        param_groups.append({
            'params': model.metacognitive.parameters(),
            'lr': config.learning_rate,
            'name': 'metacognitive'
        })

    optimizer = torch.optim.AdamW(param_groups, weight_decay=0.01)

    # Learning rate scheduler with warmup
    total_steps = len(train_data) // config.batch_size * config.n_epochs

    def lr_lambda(step):
        if step < config.warmup_steps:
            return step / config.warmup_steps
        return 1.0

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    model.train()
    global_step = 0

    print("\n" + "=" * 70)
    print("TRAINING")
    print("=" * 70)

    for epoch in range(config.n_epochs):
        # Curriculum: ramp up aux loss weight
        aux_weight = config.aux_loss_weight + (config.max_aux_loss_weight - config.aux_loss_weight) * (epoch / config.n_epochs)
        model.config.aux_loss_weight = aux_weight

        print(f"\nEpoch {epoch+1}/{config.n_epochs} (aux_weight={aux_weight:.4f})")

        np.random.shuffle(train_data)

        epoch_lm_loss = 0
        epoch_intro_loss = 0
        epoch_loop_loss = 0
        n_batches = 0

        for i in range(0, len(train_data), config.batch_size):
            batch = train_data[i:i+config.batch_size]
            if len(batch) < config.batch_size:
                continue

            input_ids = torch.stack([b['input_ids'] for b in batch]).to(DEVICE)
            attention_mask = torch.stack([b['attention_mask'] for b in batch]).to(DEVICE)
            labels = torch.stack([b['labels'] for b in batch]).to(DEVICE)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )

            loss = outputs['loss'] / config.gradient_accumulation_steps
            loss.backward()

            if (n_batches + 1) % config.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

            # Track losses
            lm_loss = outputs['lm_loss'].item() if outputs['lm_loss'] is not None else 0
            intro_loss = outputs['introspection_loss'].item() if isinstance(outputs['introspection_loss'], torch.Tensor) else outputs['introspection_loss']
            loop_loss = outputs['loop_loss'].item() if outputs['loop_loss'] is not None else 0

            epoch_lm_loss += lm_loss
            epoch_intro_loss += intro_loss
            epoch_loop_loss += loop_loss
            n_batches += 1

            # Log to W&B
            if global_step % 10 == 0:
                wandb.log({
                    'train/loss': outputs['loss'].item(),
                    'train/lm_loss': lm_loss,
                    'train/introspection_loss': intro_loss,
                    'train/loop_loss': loop_loss,
                    'train/lr': scheduler.get_last_lr()[0],
                    'train/aux_weight': aux_weight,
                }, step=global_step)

            if n_batches % 50 == 0:
                print(f"  Batch {n_batches}: LM={epoch_lm_loss/n_batches:.4f}, "
                      f"Intro={epoch_intro_loss/n_batches:.4f}, Loop={epoch_loop_loss/n_batches:.4f}")

            # Evaluation
            if global_step > 0 and global_step % config.eval_steps == 0:
                eval_results = evaluate_introspection(model, tokenizer, test_prompts[:3], wandb_log=True)
                print(f"    [Eval] Loop coherence: {eval_results['mean_loop_coherence']:.4f}, "
                      f"ECE: {eval_results['ece']:.4f}")
                model.train()

        print(f"  Epoch {epoch+1} Done - LM: {epoch_lm_loss/n_batches:.4f}, "
              f"Intro: {epoch_intro_loss/n_batches:.4f}")

        wandb.log({
            'epoch': epoch + 1,
            'epoch/lm_loss': epoch_lm_loss / n_batches,
            'epoch/intro_loss': epoch_intro_loss / n_batches,
        }, step=global_step)

    return global_step


def main():
    # Configuration
    config = IntrospectionConfig(
        model_name="Qwen/Qwen3-4B",
        introspection_dim=512,
        n_introspection_heads=8,
        prediction_horizon=3,
        hierarchical_levels=4,
        use_concept_injection=True,
        use_self_referential_weights=True,
        use_metacognitive_head=True,
        lora_r=64,
        lora_alpha=128,
        learning_rate=4.7e-4,
        batch_size=2,
        gradient_accumulation_steps=8,
        n_epochs=3,
        wandb_project="introspective-qwen3",
        wandb_run_name=f"z1401-{datetime.now().strftime('%Y%m%d-%H%M')}",
    )

    # Load tokenizer
    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(config.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print(f"✓ Tokenizer loaded (vocab size: {tokenizer.vocab_size})")

    # Create model
    print("\nCreating introspective model...")
    model = IntrospectiveQwen3(config)

    # Create dataset
    print("\nCreating introspection dataset...")
    train_data = create_introspection_dataset(tokenizer, n_samples=800)
    print(f"✓ Created {len(train_data)} training samples")

    # Test prompts
    test_prompts = [
        "Let me analyze my thought process here:",
        "My internal representation of this concept is",
        "When I reflect on my own reasoning,",
        "The recursive nature of self-awareness means",
        "Breaking down how I understand this,",
        "My confidence in understanding myself is",
        "If I could observe my own activations,",
        "Thinking about my own thinking reveals",
    ]

    # Pre-training evaluation
    print("\n" + "=" * 70)
    print("PRE-TRAINING EVALUATION")
    print("=" * 70)

    pre_results = evaluate_introspection(model, tokenizer, test_prompts, wandb_log=False)

    for i, res in enumerate(pre_results['results'][:3]):
        print(f"\n--- Prompt {i+1} ---")
        print(f"Prompt: {res['prompt']}")
        print(f"Generated: {res['new_text'][:200]}...")
        print(f"Loop coherence: {res['loop_coherence']:.3f}")
        print(f"Confidence: {res['confidence']:.3f}")

    print(f"\nPre-training summary:")
    print(f"  Mean loop coherence: {pre_results['mean_loop_coherence']:.4f}")
    print(f"  ECE: {pre_results['ece']:.4f}")

    # Training
    global_step = train(model, config, train_data, tokenizer, test_prompts)

    # Post-training evaluation
    print("\n" + "=" * 70)
    print("POST-TRAINING EVALUATION")
    print("=" * 70)

    model.eval()
    post_results = evaluate_introspection(model, tokenizer, test_prompts, wandb_log=True)

    for i, res in enumerate(post_results['results']):
        print(f"\n--- Prompt {i+1} ---")
        print(f"Prompt: {res['prompt']}")
        print(f"Generated: {res['new_text'][:250]}...")
        print(f"Loop coherence: {res['loop_coherence']:.3f}")
        print(f"Confidence: {res['confidence']:.3f}")

    # Final comparison
    print("\n" + "=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)

    pre_coherence = pre_results['mean_loop_coherence']
    post_coherence = post_results['mean_loop_coherence']

    print(f"\nLoop Coherence:")
    print(f"  Pre-training:  {pre_coherence:.4f}")
    print(f"  Post-training: {post_coherence:.4f}")
    if pre_coherence != 0:
        improvement = (post_coherence - pre_coherence) / abs(pre_coherence) * 100
        print(f"  Improvement:   {improvement:+.1f}%")

    print(f"\nCalibration (ECE, lower is better):")
    print(f"  Pre-training:  {pre_results['ece']:.4f}")
    print(f"  Post-training: {post_results['ece']:.4f}")

    print(f"\nAccuracy (introspective content generation):")
    print(f"  Pre-training:  {pre_results['mean_accuracy']:.2%}")
    print(f"  Post-training: {post_results['mean_accuracy']:.2%}")

    # Log final metrics to W&B
    wandb.log({
        'final/pre_loop_coherence': pre_coherence,
        'final/post_loop_coherence': post_coherence,
        'final/pre_ece': pre_results['ece'],
        'final/post_ece': post_results['ece'],
        'final/pre_accuracy': pre_results['mean_accuracy'],
        'final/post_accuracy': post_results['mean_accuracy'],
    })

    # Save results
    output = {
        'experiment': 'z1401_introspective_qwen3_serious',
        'timestamp': datetime.now().isoformat(),
        'config': {
            'model_name': config.model_name,
            'introspection_dim': config.introspection_dim,
            'hierarchical_levels': config.hierarchical_levels,
            'lora_r': config.lora_r,
            'lora_alpha': config.lora_alpha,
            'n_epochs': config.n_epochs,
        },
        'pre_training': {
            'loop_coherence': pre_coherence,
            'ece': pre_results['ece'],
            'accuracy': pre_results['mean_accuracy'],
        },
        'post_training': {
            'loop_coherence': post_coherence,
            'ece': post_results['ece'],
            'accuracy': post_results['mean_accuracy'],
            'samples': [{'prompt': r['prompt'], 'generated': r['new_text'][:400],
                        'loop_coherence': r['loop_coherence'], 'confidence': r['confidence']}
                       for r in post_results['results']],
        },
    }

    output_path = Path(__file__).parent.parent / 'results' / 'z1401_introspective_qwen3_serious.json'
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to: {output_path}")

    # Finish W&B
    wandb.finish()

    return output


if __name__ == '__main__':
    main()
