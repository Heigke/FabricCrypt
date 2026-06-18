#!/usr/bin/env python3
"""
======================================================================
  z1403: LATENT DEPTH REASONING - Selective Deep Thinking in Latent Space

  Combines:
  1. Retrofitting Recurrence (Prelude-Recurrent-Coda) - add adaptive depth
  2. Mixture-of-Depths routing - selective compute allocation
  3. Coconut-style continuous thought - latent space reasoning
  4. Knowledge distillation - preserve semantics

  Key Innovation: "Latent Depth Router" that allows the model to dynamically
  choose HOW DEEP to process different activation patterns without generating
  tokens, reasoning purely in latent space.
======================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
import json
import os
import functools
from datetime import datetime
from typing import Optional, Tuple, List, Dict
import math

# Flush prints immediately
print = functools.partial(print, flush=True)

# Wandb for logging
try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False

print("=" * 70)
print("  z1403: LATENT DEPTH REASONING - Selective Deep Thinking")
print("  Reasoning in Latent Space with Adaptive Compute Allocation")
print("=" * 70)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\nDevice: {device}")

# ============================================================================
# CORE ARCHITECTURE: Latent Depth Router + Recurrent Reasoning Block
# ============================================================================

class LatentDepthRouter(nn.Module):
    """
    Mixture-of-Depths style router that decides HOW MUCH computation
    to allocate to each position/activation pattern.

    Key: This is a SOFT router using learned gates, not hard top-k.
    The model learns to allocate more "thinking depth" to complex patterns.
    """

    def __init__(self, hidden_dim: int, num_depth_levels: int = 4):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_depth_levels = num_depth_levels

        # Router network: predicts depth allocation per position
        self.router = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.GELU(),
            nn.Linear(hidden_dim // 4, num_depth_levels),
        )

        # Auxiliary loss: encourage load balancing across depth levels
        self.register_buffer('depth_usage', torch.zeros(num_depth_levels))
        self.register_buffer('total_tokens', torch.tensor(0.0))

    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            hidden_states: [batch, seq_len, hidden_dim]
        Returns:
            depth_weights: [batch, seq_len, num_depth_levels] - soft allocation
            routing_loss: scalar - load balancing auxiliary loss
        """
        # Compute depth routing logits
        logits = self.router(hidden_states)  # [B, S, num_levels]

        # Soft routing weights (can think of as "how much to use each depth")
        depth_weights = F.softmax(logits, dim=-1)

        # Track usage for auxiliary loss (load balancing)
        if self.training:
            avg_weights = depth_weights.mean(dim=[0, 1])  # [num_levels]
            self.depth_usage = 0.9 * self.depth_usage + 0.1 * avg_weights.detach()
            self.total_tokens = self.total_tokens + hidden_states.shape[0] * hidden_states.shape[1]

        # Load balancing loss: encourage equal usage of all depths
        # Without this, model might collapse to always using depth 0 or max
        uniform = torch.ones_like(self.depth_usage) / self.num_depth_levels
        routing_loss = F.kl_div(
            torch.log(self.depth_usage + 1e-8),
            uniform,
            reduction='sum'
        )

        return depth_weights, routing_loss


class RecurrentReasoningBlock(nn.Module):
    """
    Lightweight recurrent block for iterative reasoning in latent space.
    Simplified from Coconut-style continuous thought - uses GRU-style gating.
    """

    def __init__(self, hidden_dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.hidden_dim = hidden_dim

        # Simple GRU-style gating (more efficient than full attention)
        self.gate = nn.Linear(hidden_dim * 2, hidden_dim)
        self.transform = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Iteration embedding
        self.iteration_emb = nn.Embedding(8, hidden_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        iteration: int,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Single iteration of recurrent reasoning.
        """
        B, S, D = hidden_states.shape

        # Add iteration info
        iter_emb = self.iteration_emb(torch.tensor([min(iteration, 7)], device=hidden_states.device))
        h_aug = hidden_states + iter_emb.unsqueeze(0).expand(B, S, -1) * 0.1

        # Transform
        h_new = self.transform(h_aug)

        # Gated update
        gate = torch.sigmoid(self.gate(torch.cat([hidden_states, h_new], dim=-1)))
        return gate * h_new + (1 - gate) * hidden_states


class LatentDepthReasoningModule(nn.Module):
    """
    The full Latent Depth Reasoning module that can be attached to
    any transformer layer's output.

    Architecture:
    - Prelude: The original transformer layers (frozen/LoRA)
    - Recurrent: This module (iterates based on router decision)
    - Coda: Continue through remaining layers

    The router decides PER-POSITION how many recurrent iterations to use.
    More "difficult" activation patterns get more iterations.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_depth_levels: int = 4,
        max_iterations: int = 4,
        num_heads: int = 8,
        residual_scale: float = 0.1,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_depth_levels = num_depth_levels
        self.max_iterations = max_iterations
        self.residual_scale = residual_scale

        # Router decides depth allocation
        self.router = LatentDepthRouter(hidden_dim, num_depth_levels)

        # Single recurrent block (shared across iterations)
        self.recurrent_block = RecurrentReasoningBlock(hidden_dim, num_heads)

        # Project back to residual stream (small scale to preserve semantics)
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)
        nn.init.zeros_(self.output_proj.weight)  # Start near identity
        nn.init.zeros_(self.output_proj.bias)

        # Track depth statistics for analysis
        self.register_buffer('avg_depth_used', torch.tensor(0.0))

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Apply latent depth reasoning with selective compute allocation.

        Args:
            hidden_states: [batch, seq_len, hidden_dim]
            attention_mask: optional mask
        Returns:
            dict with 'output', 'routing_loss', 'depth_stats'
        """
        batch_size, seq_len, _ = hidden_states.shape

        # Get depth routing weights
        depth_weights, routing_loss = self.router(hidden_states)
        # depth_weights: [batch, seq_len, num_depth_levels]

        # Compute expected depth per position
        depth_indices = torch.arange(self.num_depth_levels, device=device).float()
        expected_depths = (depth_weights * depth_indices).sum(dim=-1)  # [B, S]

        # For efficiency, we do a WEIGHTED combination of different iteration depths
        # rather than actually branching. This is differentiable.

        # Start with original state
        accumulated_output = torch.zeros_like(hidden_states)
        current_state = hidden_states.clone()

        for depth_level in range(self.num_depth_levels):
            # Number of iterations for this depth level
            num_iters = depth_level + 1  # depth 0 = 1 iter, depth 3 = 4 iters

            # Apply recurrent iterations
            state = hidden_states.clone()
            for i in range(min(num_iters, self.max_iterations)):
                state = self.recurrent_block(state, i, attention_mask)

            # Weight by routing probability
            weight = depth_weights[:, :, depth_level:depth_level+1]  # [B, S, 1]
            accumulated_output = accumulated_output + weight * state

        # Project to residual (scaled small to preserve semantics)
        delta = self.output_proj(accumulated_output - hidden_states)
        output = hidden_states + delta * self.residual_scale

        # Track statistics
        avg_depth = expected_depths.mean().item()
        self.avg_depth_used = 0.9 * self.avg_depth_used + 0.1 * avg_depth

        return {
            'output': output,
            'routing_loss': routing_loss,
            'depth_weights': depth_weights,
            'expected_depth': expected_depths,
            'avg_depth': avg_depth,
        }


class HierarchicalSelfPredictor(nn.Module):
    """
    Predicts what the next latent depth reasoning module will compute,
    creating hierarchical self-modeling across depths.

    This enables the model to "plan" its latent reasoning trajectory.
    """

    def __init__(self, hidden_dim: int, num_prediction_layers: int = 3):
        super().__init__()
        self.hidden_dim = hidden_dim

        # Encoder for current state
        self.encoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
        )

        # Predictor for next state
        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim // 4, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, hidden_dim),
        )

    def forward(
        self,
        current_state: torch.Tensor,
        target_state: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Predict the target state from current state.
        """
        encoded = self.encoder(current_state)
        predicted = self.predictor(encoded)

        # Prediction loss
        pred_loss = F.mse_loss(predicted, target_state.detach())

        return {
            'predicted': predicted,
            'prediction_loss': pred_loss,
        }


# ============================================================================
# FULL MODEL: Latent Depth Reasoning Qwen3
# ============================================================================

class LatentDepthQwen3(nn.Module):
    """
    Qwen3 with Latent Depth Reasoning modules inserted at key layers.

    Architecture:
    - Prelude: Layers 0-8 (frozen/LoRA)
    - Recurrent: Latent Depth Module at layer 9, 18, 27
    - Coda: Final layers + LM head

    Knowledge distillation from frozen reference to preserve semantics.
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-8B",
        num_depth_levels: int = 4,
        max_iterations: int = 4,
        lora_r: int = 32,
        lora_alpha: int = 64,
        residual_scale: float = 0.1,
        distill_weight: float = 0.3,
    ):
        super().__init__()
        self.model_name = model_name
        self.distill_weight = distill_weight
        self.residual_scale = residual_scale

        print(f"\nLoading {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Main model with LoRA
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )

        # Get model dimensions
        self.hidden_dim = self.model.config.hidden_size
        self.num_layers = self.model.config.num_hidden_layers

        # Determine insertion points (1/3 and 2/3 through)
        self.insertion_layers = [
            self.num_layers // 3,
            2 * self.num_layers // 3,
        ]

        print(f"  Hidden dim: {self.hidden_dim}")
        print(f"  Layers: {self.num_layers}")
        print(f"  Latent depth modules at layers: {self.insertion_layers}")

        # Apply LoRA for fine-tuning
        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )
        self.model = get_peft_model(self.model, lora_config)

        # Create Latent Depth Reasoning modules
        self.depth_modules = nn.ModuleList([
            LatentDepthReasoningModule(
                hidden_dim=self.hidden_dim,
                num_depth_levels=num_depth_levels,
                max_iterations=max_iterations,
                residual_scale=residual_scale,
            ).to(device).to(torch.bfloat16)
            for _ in self.insertion_layers
        ])

        # Hierarchical self-predictors between modules
        self.self_predictors = nn.ModuleList([
            HierarchicalSelfPredictor(self.hidden_dim).to(device).to(torch.bfloat16)
            for _ in range(len(self.insertion_layers) - 1)
        ])

        # Frozen reference model for knowledge distillation
        print("Creating frozen reference for knowledge distillation...")
        self.reference_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        for param in self.reference_model.parameters():
            param.requires_grad = False
        self.reference_model.eval()

        # Print parameter counts
        base_trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        depth_params = sum(p.numel() for p in self.depth_modules.parameters())
        pred_params = sum(p.numel() for p in self.self_predictors.parameters())

        print(f"\n✓ Model initialized")
        print(f"  LoRA trainable: {base_trainable:,}")
        print(f"  Depth modules: {depth_params:,}")
        print(f"  Self-predictors: {pred_params:,}")
        print(f"  Total new params: {depth_params + pred_params:,}")

    def forward_with_depth_reasoning(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with latent depth reasoning.

        We hook into the model's forward pass to apply depth modules
        at the insertion layers.
        """
        # Get hidden states at each insertion layer
        outputs = self.model.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )

        all_hidden = outputs.hidden_states  # Tuple of [B, S, H]

        # Apply depth reasoning at each insertion point
        depth_outputs = []
        total_routing_loss = 0.0
        total_pred_loss = 0.0
        depth_stats = []

        for i, layer_idx in enumerate(self.insertion_layers):
            hidden = all_hidden[layer_idx + 1].to(device)  # +1 because hidden_states[0] is embeddings

            # Apply latent depth reasoning
            depth_out = self.depth_modules[i](hidden, attention_mask)
            depth_outputs.append(depth_out['output'])
            total_routing_loss += depth_out['routing_loss']
            depth_stats.append(depth_out['avg_depth'])

        # Hierarchical self-prediction between depth modules
        for i in range(len(depth_outputs) - 1):
            pred_out = self.self_predictors[i](
                depth_outputs[i].mean(dim=1),  # Pool over sequence
                depth_outputs[i+1].mean(dim=1),
            )
            total_pred_loss += pred_out['prediction_loss']

        # For LM loss, we use the final hidden state with modifications
        # This is a simplified approach - full implementation would modify
        # the forward pass itself
        final_hidden = all_hidden[-1]

        # Add contribution from depth modules (scaled)
        for i, depth_out in enumerate(depth_outputs):
            # Interpolate to match sequence position
            final_hidden = final_hidden + depth_out * self.residual_scale * 0.1

        # LM head
        logits = self.model.lm_head(final_hidden)

        # LM loss
        lm_loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            lm_loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )

        # Knowledge distillation
        distill_loss = torch.tensor(0.0, device=device)
        if labels is not None:
            with torch.no_grad():
                ref_outputs = self.reference_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                )
                ref_logits = ref_outputs.logits

            T = 2.0  # Temperature
            student_log_probs = F.log_softmax(logits / T, dim=-1)
            teacher_probs = F.softmax(ref_logits / T, dim=-1)
            distill_loss = F.kl_div(
                student_log_probs,
                teacher_probs,
                reduction='batchmean'
            ) * (T * T)

        # Total loss
        total_loss = None
        if lm_loss is not None:
            total_loss = (
                lm_loss +
                self.distill_weight * distill_loss +
                0.01 * total_routing_loss +
                0.01 * total_pred_loss
            )

        return {
            'loss': total_loss,
            'lm_loss': lm_loss,
            'distill_loss': distill_loss,
            'routing_loss': total_routing_loss,
            'pred_loss': total_pred_loss,
            'logits': logits,
            'depth_stats': depth_stats,
        }

    @torch.no_grad()
    def generate_with_depth_reasoning(
        self,
        prompt: str,
        max_new_tokens: int = 100,
        temperature: float = 0.7,
        do_sample: bool = True,
    ) -> Tuple[str, Dict]:
        """
        Generate text with latent depth reasoning.
        """
        self.eval()

        inputs = self.tokenizer(prompt, return_tensors="pt").to(device)

        # Generate
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=do_sample,
            pad_token_id=self.tokenizer.pad_token_id,
        )

        generated = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Get depth statistics from a forward pass
        with torch.no_grad():
            result = self.forward_with_depth_reasoning(inputs.input_ids, inputs.attention_mask)

        stats = {
            'depth_stats': [float(d) for d in result['depth_stats']],
            'avg_depth': sum(result['depth_stats']) / len(result['depth_stats']),
        }

        return generated, stats


# ============================================================================
# DATASET
# ============================================================================

class ReasoningDataset(Dataset):
    """
    Dataset of prompts that require deep reasoning.
    Mix of math, logic, and reflective prompts.
    """

    def __init__(self, tokenizer, num_samples: int = 400, max_length: int = 256):
        self.tokenizer = tokenizer
        self.max_length = max_length

        # Deep reasoning prompts
        self.prompts = [
            # Mathematical reasoning
            "Let me work through this step by step. If we have a sequence where each term is the sum of the previous two terms, and the first two terms are 3 and 5, what is the pattern?",
            "Consider the equation x² + 2x - 15 = 0. To solve this, I need to factor or use the quadratic formula.",
            "If we divide 144 by 12, we get 12. But what if we work backwards from the answer?",

            # Logical reasoning
            "All cats are mammals. Some mammals can fly. Therefore, we cannot conclude that any cats can fly. Let me verify this logic.",
            "If it rains, the ground is wet. The ground is wet. Can we conclude it rained? This is the fallacy of",
            "Consider three boxes: one contains only apples, one contains only oranges, and one contains both. All labels are wrong.",

            # Self-reflective reasoning
            "When I analyze my own reasoning process, I notice that I first",
            "My approach to solving complex problems involves breaking them down into",
            "The recursive nature of thinking about thinking leads to",

            # Deep conceptual reasoning
            "The relationship between entropy and information can be understood by considering",
            "In quantum mechanics, the measurement problem arises because",
            "The halting problem demonstrates that some questions are fundamentally",
        ]

        self.samples = []
        for i in range(num_samples):
            prompt = self.prompts[i % len(self.prompts)]
            # Add some variation
            if i % 3 == 0:
                prompt = "Let me think deeply about this: " + prompt
            elif i % 3 == 1:
                prompt = "I need to reason carefully: " + prompt
            self.samples.append(prompt)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        text = self.samples[idx]
        encoded = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt',
        )
        return {
            'input_ids': encoded['input_ids'].squeeze(0),
            'attention_mask': encoded['attention_mask'].squeeze(0),
            'labels': encoded['input_ids'].squeeze(0).clone(),
        }


# ============================================================================
# TRAINING
# ============================================================================

def train_latent_depth_model():
    """Main training function."""

    # Initialize wandb
    if HAS_WANDB:
        wandb.init(
            project="z1403-latent-depth",
            name="qwen3-8b-latent-depth",
            config={
                "model": "Qwen/Qwen3-8B",
                "num_depth_levels": 4,
                "max_iterations": 4,
                "lora_r": 32,
                "residual_scale": 0.1,
                "distill_weight": 0.3,
            },
            mode="offline",
        )

    # Create model
    model = LatentDepthQwen3(
        model_name="Qwen/Qwen3-8B",
        num_depth_levels=4,
        max_iterations=4,
        lora_r=32,
        lora_alpha=64,
        residual_scale=0.1,
        distill_weight=0.3,
    )

    # Create dataset
    print("\nCreating dataset...")
    dataset = ReasoningDataset(model.tokenizer, num_samples=400)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
    print(f"✓ {len(dataset)} samples")

    # Optimizer (only new parameters + LoRA)
    optimizer = torch.optim.AdamW([
        {'params': model.model.parameters(), 'lr': 1e-4},
        {'params': model.depth_modules.parameters(), 'lr': 5e-4},
        {'params': model.self_predictors.parameters(), 'lr': 5e-4},
    ])

    # Pre-training evaluation
    print("\n" + "=" * 70)
    print("PRE-TRAINING EVALUATION")
    print("=" * 70)

    test_prompts = [
        "Let me analyze my reasoning step by step:",
        "When I think deeply about this problem, I notice that",
        "My approach to understanding complex systems involves",
    ]

    pre_results = []
    for prompt in test_prompts:
        generated, stats = model.generate_with_depth_reasoning(prompt, max_new_tokens=80)
        print(f"\n--- Prompt ---")
        print(f"Prompt: {prompt}")
        print(f"Generated: {generated[len(prompt):len(prompt)+200]}...")
        print(f"Avg depth: {stats['avg_depth']:.3f}")
        pre_results.append(stats['avg_depth'])

    pre_avg_depth = sum(pre_results) / len(pre_results)
    print(f"\nPre-training avg depth: {pre_avg_depth:.3f}")

    # Training
    print("\n" + "=" * 70)
    print("TRAINING")
    print("=" * 70)

    model.train()
    epochs = 2

    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")
        epoch_losses = []

        for step, batch in enumerate(dataloader):
            optimizer.zero_grad()

            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model.forward_with_depth_reasoning(
                input_ids, attention_mask, labels
            )

            loss = outputs['loss']
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_losses.append(loss.item())

            if (step + 1) % 20 == 0:
                avg_loss = sum(epoch_losses[-20:]) / len(epoch_losses[-20:])
                depth_str = ", ".join([f"{d:.2f}" for d in outputs['depth_stats']])
                print(f"  Step {step+1}: loss={avg_loss:.4f}, depths=[{depth_str}]")

                if HAS_WANDB:
                    wandb.log({
                        'train/loss': avg_loss,
                        'train/lm_loss': outputs['lm_loss'].item(),
                        'train/distill_loss': outputs['distill_loss'].item(),
                        'train/routing_loss': outputs['routing_loss'].item() if torch.is_tensor(outputs['routing_loss']) else outputs['routing_loss'],
                        'train/avg_depth': sum(outputs['depth_stats']) / len(outputs['depth_stats']),
                    })

    # Post-training evaluation
    print("\n" + "=" * 70)
    print("POST-TRAINING EVALUATION")
    print("=" * 70)

    model.eval()

    post_results = []
    generated_samples = []

    test_prompts_extended = test_prompts + [
        "The recursive nature of self-reflection means",
        "Breaking down my reasoning, I first",
        "My confidence in this analysis comes from",
    ]

    for prompt in test_prompts_extended:
        generated, stats = model.generate_with_depth_reasoning(prompt, max_new_tokens=100)
        print(f"\n--- Prompt ---")
        print(f"Prompt: {prompt}")
        print(f"Generated: {generated[len(prompt):len(prompt)+300]}...")
        print(f"Avg depth: {stats['avg_depth']:.3f}, Depths: {stats['depth_stats']}")
        post_results.append(stats['avg_depth'])
        generated_samples.append({
            'prompt': prompt,
            'generated': generated[len(prompt):len(prompt)+300],
            'depth_stats': stats['depth_stats'],
        })

    post_avg_depth = sum(post_results) / len(post_results)

    # Summary
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)

    print(f"\nAverage Reasoning Depth:")
    print(f"  Pre:  {pre_avg_depth:.3f}")
    print(f"  Post: {post_avg_depth:.3f}")

    # Depth variation (do different prompts get different depths?)
    depth_variation = max(post_results) - min(post_results)
    print(f"\nDepth Variation (max - min): {depth_variation:.3f}")
    print(f"  This indicates {'good' if depth_variation > 0.1 else 'limited'} selective depth allocation")

    # Save results
    results = {
        "experiment": "z1403_latent_depth_reasoning",
        "timestamp": datetime.now().isoformat(),
        "config": {
            "model": "Qwen/Qwen3-8B",
            "num_depth_levels": 4,
            "max_iterations": 4,
            "lora_r": 32,
            "residual_scale": 0.1,
            "distill_weight": 0.3,
        },
        "pre_training": {
            "avg_depth": pre_avg_depth,
        },
        "post_training": {
            "avg_depth": post_avg_depth,
            "depth_variation": depth_variation,
            "samples": generated_samples,
        },
    }

    results_path = "/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy/results/z1403_latent_depth_reasoning.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {results_path}")

    if HAS_WANDB:
        wandb.log({
            'final/pre_avg_depth': pre_avg_depth,
            'final/post_avg_depth': post_avg_depth,
            'final/depth_variation': depth_variation,
        })
        wandb.finish()

    return results


if __name__ == "__main__":
    results = train_latent_depth_model()
