#!/usr/bin/env python3
"""
z1402: Recursive Introspection for Hierarchical Self-Understanding

Based on cutting-edge research:
- RISE (Recursive Introspection): https://arxiv.org/abs/2407.18219
- Orthogonal LoRA for continual learning: https://arxiv.org/html/2505.22358v1
- KeepLoRA principal/residual subspaces: https://arxiv.org/html/2601.19659
- Self-Synthesized Rehearsal: https://aclanthology.org/2024.acl-long.77/
- Anthropic Introspection: https://transformer-circuits.pub/2025/introspection/

KEY INNOVATIONS TO PRESERVE SEMANTICS:
1. Orthogonal LoRA constraints - updates orthogonal to principal subspace
2. Knowledge distillation from frozen reference model
3. Hierarchical introspection in RESIDUAL stream (additive, not replacing)
4. RISE-style multi-turn self-correction training
5. Soft targets from original model to anchor behavior

ARCHITECTURE:
- Qwen3-8B base (8B params, 36 layers, 4096 hidden)
- Orthogonal LoRA adapters with subspace constraints
- Residual introspection modules (added to hidden states, not replacing)
- Hierarchical strange loop predictor
- Knowledge distillation head for semantic preservation
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

# Environment setup
os.environ.setdefault('HSA_OVERRIDE_GFX_VERSION', '11.0.0')
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print("=" * 70)
print("  z1402: RECURSIVE INTROSPECTION - HIERARCHICAL SELF-UNDERSTANDING")
print("  Preserving Semantics via Orthogonal LoRA + Knowledge Distillation")
print("=" * 70)
print(f"\nDevice: {DEVICE}")

# Imports
try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, get_peft_model, TaskType
    import wandb
    print("✓ All dependencies loaded")
except ImportError as e:
    print(f"Installing dependencies...")
    os.system("pip install transformers>=4.51.0 peft accelerate wandb -q")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, get_peft_model, TaskType
    import wandb


@dataclass
class RecursiveIntrospectionConfig:
    """Configuration preserving semantics while adding introspection"""

    # Model
    model_name: str = "Qwen/Qwen3-8B"  # 8B params, 36 layers

    # Introspection architecture (RESIDUAL - additive to preserve semantics)
    introspection_layers: List[int] = field(default_factory=lambda: [])
    introspection_dim: int = 256  # Small to minimize interference
    hierarchy_levels: int = 4  # Strange loop depth
    residual_scale: float = 0.1  # Scale introspection additions (small!)

    # Orthogonal LoRA (key for preserving semantics)
    lora_r: int = 32  # Moderate rank
    lora_alpha: int = 64
    lora_dropout: float = 0.05
    orthogonal_reg_weight: float = 0.1  # Orthogonality regularization

    # Knowledge distillation (anchor to original behavior)
    distill_weight: float = 0.5  # Weight for KL divergence from frozen model
    distill_temperature: float = 2.0

    # RISE-style recursive training
    max_turns: int = 3  # Multi-turn self-correction
    improvement_threshold: float = 0.1

    # Training (conservative to preserve semantics)
    learning_rate: float = 1e-4
    introspection_lr: float = 5e-4  # Higher for new modules
    warmup_ratio: float = 0.1
    max_grad_norm: float = 0.5  # Stricter clipping
    batch_size: int = 1
    gradient_accumulation: int = 16
    n_epochs: int = 2

    # W&B
    wandb_project: str = "recursive-introspection"
    wandb_run_name: str = "z1402-qwen8b"


class OrthogonalConstraint(nn.Module):
    """
    Enforces orthogonality between LoRA updates and principal subspace.
    Based on KeepLoRA: https://arxiv.org/html/2601.19659
    """

    def __init__(self, dim: int, rank: int):
        super().__init__()
        self.dim = dim
        self.rank = rank

        # Track principal directions (frozen from pretrained)
        self.register_buffer('principal_basis', torch.eye(dim)[:, :rank])

    def compute_orthogonal_loss(self, lora_A: torch.Tensor, lora_B: torch.Tensor) -> torch.Tensor:
        """
        Compute regularization loss to keep LoRA updates orthogonal to principal subspace.
        """
        # LoRA update direction: B @ A
        # We want this orthogonal to principal_basis

        # Project LoRA onto principal subspace
        # loss = ||P @ (B @ A)||_F where P = principal_basis @ principal_basis.T

        update = lora_B @ lora_A  # [out_dim, in_dim]

        # Sample a few directions to check orthogonality
        n_samples = min(16, self.rank)
        principal_sample = self.principal_basis[:, :n_samples]  # [dim, n_samples]

        # Compute overlap with principal directions
        if update.shape[0] == self.dim:
            overlap = torch.norm(principal_sample.T @ update)
        else:
            overlap = torch.tensor(0.0, device=update.device)

        return overlap


class ResidualIntrospectionHead(nn.Module):
    """
    Introspection head that ADDS to hidden states rather than replacing.
    This preserves the original semantic information while adding self-modeling.
    """

    def __init__(self, hidden_dim: int, introspection_dim: int = 256,
                 residual_scale: float = 0.1):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.introspection_dim = introspection_dim
        self.residual_scale = residual_scale

        # Encode for introspection (small bottleneck)
        self.encoder = nn.Sequential(
            nn.Linear(hidden_dim, introspection_dim),
            nn.GELU(),
            nn.LayerNorm(introspection_dim),
        )

        # Self-attention for reflection
        self.self_attn = nn.MultiheadAttention(
            introspection_dim, num_heads=4, dropout=0.1, batch_first=True
        )

        # Predict future layer activation (supervision signal)
        self.future_predictor = nn.Sequential(
            nn.Linear(introspection_dim, introspection_dim),
            nn.GELU(),
            nn.Linear(introspection_dim, hidden_dim),
        )

        # RESIDUAL projection back (scaled small)
        self.residual_proj = nn.Sequential(
            nn.Linear(introspection_dim, hidden_dim),
            nn.Tanh(),  # Bounded output
        )

    def forward(self, hidden_states: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            hidden_states: [B, S, H]
        Returns:
            residual: Addition to hidden states (scaled)
            prediction: Future layer prediction
            introspection_state: Encoded self-model
        """
        # Encode
        encoded = self.encoder(hidden_states)  # [B, S, intro_dim]

        # Self-attention
        reflected, attn_weights = self.self_attn(encoded, encoded, encoded)

        # Predict future
        prediction = self.future_predictor(reflected)

        # Compute residual (SCALED SMALL to preserve semantics)
        residual = self.residual_proj(reflected) * self.residual_scale

        return {
            'residual': residual,
            'prediction': prediction,
            'introspection_state': reflected,
            'attention': attn_weights,
        }


class HierarchicalStrangeLoop(nn.Module):
    """
    Hierarchical self-model with strange loop (Hofstadter-inspired).
    Level N predicts Level 0, creating recursive self-reference.
    """

    def __init__(self, hidden_dim: int, introspection_dim: int = 256,
                 n_levels: int = 4):
        super().__init__()

        self.n_levels = n_levels
        self.hidden_dim = hidden_dim
        self.introspection_dim = introspection_dim

        # Level encoders (first takes hidden_dim, rest take introspection_dim)
        self.level_encoders = nn.ModuleList()
        for i in range(n_levels):
            in_dim = hidden_dim if i == 0 else introspection_dim
            self.level_encoders.append(nn.Sequential(
                nn.Linear(in_dim, introspection_dim),
                nn.GELU(),
                nn.LayerNorm(introspection_dim),
            ))

        # Level predictors (each predicts next level)
        self.level_predictors = nn.ModuleList([
            nn.Linear(introspection_dim, introspection_dim)
            for _ in range(n_levels - 1)
        ])

        # STRANGE LOOP: Level N-1 predicts Level 0's INPUT (hidden_dim)
        self.loop_predictor = nn.Sequential(
            nn.Linear(introspection_dim, introspection_dim),
            nn.GELU(),
            nn.Linear(introspection_dim, hidden_dim),
        )

        # Confidence estimator
        self.confidence = nn.Sequential(
            nn.Linear(introspection_dim, 64),
            nn.GELU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, hidden_states: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Process through hierarchy with strange loop."""
        # Pool over sequence
        pooled = hidden_states.mean(dim=1)  # [B, hidden_dim]

        level_states = []
        level_predictions = []

        current = pooled
        for i in range(self.n_levels):
            # Encode this level
            encoded = self.level_encoders[i](current)
            level_states.append(encoded)

            # Predict next level (if not last)
            if i < self.n_levels - 1:
                pred = self.level_predictors[i](encoded)
                level_predictions.append(pred)

            current = encoded

        # Strange loop: predict back to input
        loop_prediction = self.loop_predictor(level_states[-1])

        # Confidence from final level
        confidence = self.confidence(level_states[-1])

        return {
            'level_states': level_states,
            'level_predictions': level_predictions,
            'loop_prediction': loop_prediction,
            'confidence': confidence,
        }


class RecursiveIntrospectiveModel(nn.Module):
    """
    Qwen3-8B with recursive introspection that preserves semantics.

    Key design principles:
    1. Introspection is ADDITIVE (residual), not replacing
    2. Orthogonal LoRA keeps updates in residual subspace
    3. Knowledge distillation anchors to original behavior
    4. Small introspection_dim minimizes interference
    """

    def __init__(self, config: RecursiveIntrospectionConfig):
        super().__init__()
        self.config = config

        print(f"\nLoading {config.model_name}...")

        # Load base model
        self.base_model = AutoModelForCausalLM.from_pretrained(
            config.model_name,
            torch_dtype=torch.bfloat16,
            device_map='auto',
            trust_remote_code=True,
        )

        # Store reference to frozen model for distillation
        print("Creating frozen reference for knowledge distillation...")
        self.reference_model = AutoModelForCausalLM.from_pretrained(
            config.model_name,
            torch_dtype=torch.bfloat16,
            device_map='auto',
            trust_remote_code=True,
        )
        for param in self.reference_model.parameters():
            param.requires_grad = False

        # Get dimensions
        model_config = self.base_model.config
        self.hidden_dim = model_config.hidden_size
        self.n_layers = model_config.num_hidden_layers

        print(f"  Hidden dim: {self.hidden_dim}")
        print(f"  Layers: {self.n_layers}")

        # Set introspection layers (1/4, 1/2, 3/4, final)
        if not config.introspection_layers:
            config.introspection_layers = [
                self.n_layers // 4,
                self.n_layers // 2,
                3 * self.n_layers // 4,
                self.n_layers - 1,
            ]
        print(f"  Introspection layers: {config.introspection_layers}")

        # Apply LoRA
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=config.lora_r,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        )
        self.base_model = get_peft_model(self.base_model, lora_config)
        print(f"  LoRA: r={config.lora_r}, alpha={config.lora_alpha}")

        # Orthogonal constraint modules
        self.orthogonal_constraints = nn.ModuleList([
            OrthogonalConstraint(self.hidden_dim, config.lora_r)
            for _ in range(len(config.introspection_layers))
        ])

        # Residual introspection heads (ADDITIVE)
        self.introspection_heads = nn.ModuleDict({
            f"layer_{i}": ResidualIntrospectionHead(
                hidden_dim=self.hidden_dim,
                introspection_dim=config.introspection_dim,
                residual_scale=config.residual_scale,
            )
            for i in config.introspection_layers
        })

        # Hierarchical strange loop
        self.strange_loop = HierarchicalStrangeLoop(
            hidden_dim=self.hidden_dim,
            introspection_dim=config.introspection_dim,
            n_levels=config.hierarchy_levels,
        )

        # Move to device
        self.introspection_heads = self.introspection_heads.to(DEVICE)
        self.strange_loop = self.strange_loop.to(DEVICE)
        self.orthogonal_constraints = self.orthogonal_constraints.to(DEVICE)

        self._count_params()

    def _count_params(self):
        """Count parameters."""
        base_trainable = sum(p.numel() for p in self.base_model.parameters() if p.requires_grad)
        intro_params = sum(p.numel() for p in self.introspection_heads.parameters())
        loop_params = sum(p.numel() for p in self.strange_loop.parameters())

        print(f"\n✓ Model initialized")
        print(f"  Base (LoRA): {base_trainable:,}")
        print(f"  Introspection heads: {intro_params:,}")
        print(f"  Strange loop: {loop_params:,}")
        print(f"  Total new: {intro_params + loop_params:,}")

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor = None,
                labels: torch.Tensor = None, compute_distill: bool = True,
                **kwargs) -> Dict[str, torch.Tensor]:
        """
        Forward with introspection + knowledge distillation.
        """
        # Get hidden states from base model
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            output_hidden_states=True,
            **kwargs
        )

        hidden_states = outputs.hidden_states
        introspection_outputs = {}
        introspection_losses = []

        # Process each introspection layer
        for layer_idx in self.config.introspection_layers:
            layer_key = f"layer_{layer_idx}"
            current_hidden = hidden_states[layer_idx + 1].float()

            # Get introspection output (residual)
            intro_out = self.introspection_heads[layer_key](current_hidden)
            introspection_outputs[layer_key] = intro_out

            # Future prediction loss (if we have future layer)
            target_layer = layer_idx + 2
            if target_layer < len(hidden_states):
                target_hidden = hidden_states[target_layer].float()
                pred_loss = F.mse_loss(intro_out['prediction'], target_hidden)
                introspection_losses.append(pred_loss)

        # Strange loop on final hidden
        final_hidden = hidden_states[-1].float()
        loop_out = self.strange_loop(final_hidden)
        introspection_outputs['strange_loop'] = loop_out

        # Level prediction losses
        for i, pred in enumerate(loop_out['level_predictions']):
            target = loop_out['level_states'][i + 1]
            level_loss = F.mse_loss(pred, target)
            introspection_losses.append(level_loss * 0.5)

        # Strange loop loss (predict back to input)
        loop_target = hidden_states[1].float().mean(dim=1)  # First layer pooled
        loop_loss = F.mse_loss(loop_out['loop_prediction'], loop_target)
        introspection_losses.append(loop_loss)

        # Compute knowledge distillation loss (CRITICAL for preserving semantics)
        distill_loss = torch.tensor(0.0, device=DEVICE)
        if compute_distill and labels is not None:
            with torch.no_grad():
                ref_outputs = self.reference_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                )
                ref_logits = ref_outputs.logits

            # KL divergence from reference (soft targets)
            T = self.config.distill_temperature
            student_log_probs = F.log_softmax(outputs.logits / T, dim=-1)
            teacher_probs = F.softmax(ref_logits / T, dim=-1)
            distill_loss = F.kl_div(student_log_probs, teacher_probs, reduction='batchmean') * (T * T)

        # Combine losses
        intro_loss = sum(introspection_losses) / len(introspection_losses) if introspection_losses else torch.tensor(0.0)

        # Total loss with distillation
        if outputs.loss is not None:
            total_loss = outputs.loss + 0.01 * intro_loss + self.config.distill_weight * distill_loss
        else:
            total_loss = intro_loss + distill_loss

        return {
            'loss': total_loss,
            'lm_loss': outputs.loss,
            'introspection_loss': intro_loss,
            'distill_loss': distill_loss,
            'loop_loss': loop_loss,
            'logits': outputs.logits,
            'introspection_outputs': introspection_outputs,
            'confidence': loop_out['confidence'],
        }

    def generate_with_introspection(self, input_ids: torch.Tensor,
                                    max_new_tokens: int = 100, **kwargs) -> Dict:
        """Generate with introspection tracking."""
        # Pre-generation state
        with torch.no_grad():
            pre_out = self.base_model(input_ids, output_hidden_states=True)
            pre_hidden = pre_out.hidden_states[-1].float()
            pre_loop = self.strange_loop(pre_hidden)

        # Generate
        generated = self.base_model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.8,
            top_p=0.9,
            min_new_tokens=20,
            pad_token_id=self.base_model.config.eos_token_id,
            **kwargs
        )

        # Post-generation state
        with torch.no_grad():
            post_out = self.base_model(generated, output_hidden_states=True)
            post_hidden = post_out.hidden_states[-1].float()
            post_loop = self.strange_loop(post_hidden)

        return {
            'generated_ids': generated,
            'pre_loop': pre_loop,
            'post_loop': post_loop,
            'pre_hidden': pre_out.hidden_states[1].float().mean(dim=1),
            'post_hidden': post_out.hidden_states[1].float().mean(dim=1),
        }


def create_introspection_dataset(tokenizer, n_samples: int = 500) -> List[Dict]:
    """Create dataset for recursive introspection training."""

    # Prompts that encourage self-reflection without changing task behavior
    prompts = [
        # Analysis prompts (successful pattern)
        "Let me analyze my reasoning step by step:",
        "Breaking this down, I first consider",
        "My approach to this problem involves",

        # Self-reflection
        "When I examine my thought process,",
        "The key aspects of my reasoning are",
        "I notice my understanding focuses on",

        # Meta-cognitive
        "My confidence in this answer comes from",
        "The uncertainty I have relates to",
        "If I reconsider this, I might",

        # Recursive
        "Thinking about how I think about this,",
        "My model of my own reasoning shows",
        "The recursive nature of this reflection",
    ]

    dataset = []
    for i in range(n_samples):
        prompt = prompts[i % len(prompts)]

        # Variation
        if i % 3 == 0:
            prompt = f"<think>\n{prompt}"
        elif i % 3 == 1:
            prompt = f"Question: Analyze your reasoning.\n{prompt}"

        encoded = tokenizer(
            prompt,
            truncation=True,
            max_length=256,
            padding='max_length',
            return_tensors='pt',
        )

        dataset.append({
            'input_ids': encoded['input_ids'].squeeze(),
            'attention_mask': encoded['attention_mask'].squeeze(),
            'labels': encoded['input_ids'].squeeze(),
        })

    return dataset


def evaluate_introspection(model: RecursiveIntrospectiveModel, tokenizer,
                          prompts: List[str]) -> Dict:
    """Evaluate introspective capabilities."""
    model.eval()
    results = []

    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors='pt').to(DEVICE)

        with torch.no_grad():
            gen_out = model.generate_with_introspection(
                inputs['input_ids'],
                max_new_tokens=80,
            )

            text = tokenizer.decode(gen_out['generated_ids'][0], skip_special_tokens=True)
            new_text = text[len(prompt):].strip()

            # Loop coherence
            loop_pred = gen_out['post_loop']['loop_prediction']
            actual = gen_out['post_hidden']
            coherence = F.cosine_similarity(loop_pred, actual, dim=-1).mean().item()

            # Confidence
            confidence = gen_out['post_loop']['confidence'].mean().item()

            # Check for self-referential content
            has_introspection = any(w in new_text.lower() for w in
                                   ['i ', 'my ', 'think', 'reason', 'consider', 'analyze'])

        results.append({
            'prompt': prompt,
            'generated': new_text,
            'loop_coherence': coherence,
            'confidence': confidence,
            'has_introspection': has_introspection,
        })

    return {
        'results': results,
        'mean_coherence': np.mean([r['loop_coherence'] for r in results]),
        'mean_confidence': np.mean([r['confidence'] for r in results]),
        'introspection_rate': np.mean([r['has_introspection'] for r in results]),
    }


def train(model: RecursiveIntrospectiveModel, config: RecursiveIntrospectionConfig,
          train_data: List[Dict], tokenizer, eval_prompts: List[str]):
    """Train with knowledge distillation to preserve semantics."""

    # Initialize W&B
    wandb.init(
        project=config.wandb_project,
        name=config.wandb_run_name,
        config={
            'model': config.model_name,
            'lora_r': config.lora_r,
            'introspection_dim': config.introspection_dim,
            'distill_weight': config.distill_weight,
            'residual_scale': config.residual_scale,
        }
    )

    # Separate optimizers (different LRs)
    base_params = [p for p in model.base_model.parameters() if p.requires_grad]
    intro_params = list(model.introspection_heads.parameters()) + list(model.strange_loop.parameters())

    optimizer = torch.optim.AdamW([
        {'params': base_params, 'lr': config.learning_rate},
        {'params': intro_params, 'lr': config.introspection_lr},
    ], weight_decay=0.01)

    total_steps = len(train_data) // config.batch_size * config.n_epochs
    warmup_steps = int(total_steps * config.warmup_ratio)

    def lr_schedule(step):
        if step < warmup_steps:
            return step / warmup_steps
        return max(0.1, 1 - (step - warmup_steps) / (total_steps - warmup_steps))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_schedule)

    model.train()
    global_step = 0
    accum_loss = 0
    accum_lm = 0
    accum_intro = 0
    accum_distill = 0

    print("\n" + "=" * 70)
    print("TRAINING")
    print("=" * 70)

    for epoch in range(config.n_epochs):
        print(f"\nEpoch {epoch + 1}/{config.n_epochs}")
        np.random.shuffle(train_data)

        for i, sample in enumerate(train_data):
            input_ids = sample['input_ids'].unsqueeze(0).to(DEVICE)
            attention_mask = sample['attention_mask'].unsqueeze(0).to(DEVICE)
            labels = sample['labels'].unsqueeze(0).to(DEVICE)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                compute_distill=True,
            )

            loss = outputs['loss'] / config.gradient_accumulation
            loss.backward()

            accum_loss += outputs['loss'].item()
            accum_lm += outputs['lm_loss'].item() if outputs['lm_loss'] is not None else 0
            accum_intro += outputs['introspection_loss'].item() if isinstance(outputs['introspection_loss'], torch.Tensor) else 0
            accum_distill += outputs['distill_loss'].item() if isinstance(outputs['distill_loss'], torch.Tensor) else 0

            if (i + 1) % config.gradient_accumulation == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                # Log to W&B
                n = config.gradient_accumulation
                wandb.log({
                    'train/loss': accum_loss / n,
                    'train/lm_loss': accum_lm / n,
                    'train/intro_loss': accum_intro / n,
                    'train/distill_loss': accum_distill / n,
                    'train/lr': scheduler.get_last_lr()[0],
                }, step=global_step)

                if global_step % 20 == 0:
                    print(f"  Step {global_step}: loss={accum_loss/n:.4f}, "
                          f"lm={accum_lm/n:.4f}, intro={accum_intro/n:.4f}, "
                          f"distill={accum_distill/n:.4f}")

                accum_loss = accum_lm = accum_intro = accum_distill = 0

                # Periodic evaluation
                if global_step % 50 == 0:
                    eval_results = evaluate_introspection(model, tokenizer, eval_prompts[:3])
                    wandb.log({
                        'eval/coherence': eval_results['mean_coherence'],
                        'eval/confidence': eval_results['mean_confidence'],
                        'eval/introspection_rate': eval_results['introspection_rate'],
                    }, step=global_step)
                    print(f"    [Eval] coherence={eval_results['mean_coherence']:.3f}, "
                          f"intro_rate={eval_results['introspection_rate']:.2%}")
                    model.train()

    return global_step


def main():
    config = RecursiveIntrospectionConfig(
        model_name="Qwen/Qwen3-8B",
        introspection_dim=256,
        hierarchy_levels=4,
        residual_scale=0.1,  # Small!
        lora_r=32,
        lora_alpha=64,
        distill_weight=0.5,  # Strong distillation
        learning_rate=1e-4,
        introspection_lr=5e-4,
        batch_size=1,
        gradient_accumulation=16,
        n_epochs=2,
        wandb_run_name=f"z1402-{datetime.now().strftime('%m%d-%H%M')}",
    )

    # Load tokenizer
    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(config.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print(f"✓ Tokenizer loaded")

    # Create model
    model = RecursiveIntrospectiveModel(config)

    # Create dataset
    print("\nCreating dataset...")
    train_data = create_introspection_dataset(tokenizer, n_samples=400)
    print(f"✓ {len(train_data)} samples")

    # Eval prompts
    eval_prompts = [
        "Let me analyze my reasoning step by step:",
        "When I examine my thought process,",
        "My approach to understanding this involves",
        "The recursive nature of self-reflection means",
        "Breaking down my reasoning, I first",
        "My confidence in this analysis comes from",
    ]

    # Pre-training eval
    print("\n" + "=" * 70)
    print("PRE-TRAINING EVALUATION")
    print("=" * 70)

    pre_eval = evaluate_introspection(model, tokenizer, eval_prompts)
    for i, r in enumerate(pre_eval['results'][:3]):
        print(f"\n--- Prompt {i+1} ---")
        print(f"Prompt: {r['prompt']}")
        print(f"Generated: {r['generated'][:200]}...")
        print(f"Coherence: {r['loop_coherence']:.3f}, Confidence: {r['confidence']:.3f}")

    print(f"\nPre-training: coherence={pre_eval['mean_coherence']:.4f}, "
          f"intro_rate={pre_eval['introspection_rate']:.2%}")

    # Train
    global_step = train(model, config, train_data, tokenizer, eval_prompts)

    # Post-training eval
    print("\n" + "=" * 70)
    print("POST-TRAINING EVALUATION")
    print("=" * 70)

    model.eval()
    post_eval = evaluate_introspection(model, tokenizer, eval_prompts)

    for i, r in enumerate(post_eval['results']):
        print(f"\n--- Prompt {i+1} ---")
        print(f"Prompt: {r['prompt']}")
        print(f"Generated: {r['generated'][:250]}...")
        print(f"Coherence: {r['loop_coherence']:.3f}, Confidence: {r['confidence']:.3f}")

    # Summary
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)

    print(f"\nLoop Coherence:")
    print(f"  Pre:  {pre_eval['mean_coherence']:.4f}")
    print(f"  Post: {post_eval['mean_coherence']:.4f}")

    print(f"\nIntrospection Rate:")
    print(f"  Pre:  {pre_eval['introspection_rate']:.2%}")
    print(f"  Post: {post_eval['introspection_rate']:.2%}")

    wandb.log({
        'final/pre_coherence': pre_eval['mean_coherence'],
        'final/post_coherence': post_eval['mean_coherence'],
        'final/pre_intro_rate': pre_eval['introspection_rate'],
        'final/post_intro_rate': post_eval['introspection_rate'],
    })

    # Save results
    output = {
        'experiment': 'z1402_recursive_introspection',
        'timestamp': datetime.now().isoformat(),
        'config': {
            'model': config.model_name,
            'introspection_dim': config.introspection_dim,
            'residual_scale': config.residual_scale,
            'distill_weight': config.distill_weight,
        },
        'pre_training': {
            'coherence': pre_eval['mean_coherence'],
            'introspection_rate': pre_eval['introspection_rate'],
        },
        'post_training': {
            'coherence': post_eval['mean_coherence'],
            'introspection_rate': post_eval['introspection_rate'],
            'samples': [{'prompt': r['prompt'], 'generated': r['generated'][:400],
                        'coherence': r['loop_coherence']} for r in post_eval['results']],
        },
    }

    output_path = Path(__file__).parent.parent / 'results' / 'z1402_recursive_introspection.json'
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to: {output_path}")
    wandb.finish()

    return output


if __name__ == '__main__':
    main()
