#!/usr/bin/env python3
"""
======================================================================
  z1406: RECURSIVE PREDICTIVE SELF-MODEL (RPSM)

  NOVEL ARCHITECTURE combining:
  1. Predictive Coding - predict own future activations
  2. Strange Loop - predictions feed back into predictor
  3. Prediction-Error-Driven Depth - uncertainty triggers deeper thinking
  4. World Model with Self - model predicts effect of own outputs

  Key Innovation: The model iteratively refines its internal state
  until self-prediction error falls below threshold, creating
  a "thinking until confident" mechanism in latent space.

  Inspired by:
  - TWISTER (ICLR 2025): Contrastive predictive coding for world models
  - RISE: Recursive introspection for self-improvement
  - Hofstadter's Strange Loops: Self-referential processing
  - Predictive Coding Theory: Minimize prediction error

  This is NOVEL because:
  - Self-prediction is the CORE objective (not auxiliary)
  - Depth is DYNAMIC based on prediction error
  - Creates genuine recursive self-reference
  - Model learns to "know when it doesn't know"
======================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
import json
import functools
from datetime import datetime
from typing import Optional, Tuple, Dict, List
import numpy as np

print = functools.partial(print, flush=True)

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False

print("=" * 70)
print("  z1406: RECURSIVE PREDICTIVE SELF-MODEL (RPSM)")
print("  Novel Architecture: Prediction-Error-Driven Recursive Depth")
print("=" * 70)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\nDevice: {device}")


# ============================================================================
# CORE INNOVATION: Predictive Self-Model
# Predicts own future activations and uses error to guide depth
# ============================================================================

class PredictiveSelfModel(nn.Module):
    """
    Predicts the model's own future internal states.

    Key insight from predictive coding: The brain minimizes prediction error.
    We apply this to the model's own activations - if the model can predict
    its own future states accurately, it has a good "self-model".

    Prediction targets:
    1. Next-layer activations (vertical prediction)
    2. Future-token activations (horizontal prediction)
    3. Output logits (what will I say?)
    4. Confidence level (how sure will I be?)
    """

    def __init__(self, hidden_dim: int, num_layers: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Vertical predictor: predict next layer's activations
        self.vertical_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Horizontal predictor: predict future positions
        self.horizontal_predictor = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
        )

        # Meta-predictor: predict own prediction error
        self.error_predictor = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.GELU(),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

        # Confidence predictor: how confident will the output be?
        self.confidence_predictor = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.GELU(),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

    def predict_next_layer(self, current_hidden: torch.Tensor) -> torch.Tensor:
        """Predict what the next layer's activations will be."""
        return self.vertical_predictor(current_hidden)

    def predict_future_positions(
        self,
        current_hidden: torch.Tensor,
        num_future: int = 3
    ) -> torch.Tensor:
        """Predict activations at future sequence positions."""
        B, S, H = current_hidden.shape

        # Use last position to predict future
        last_hidden = current_hidden[:, -1:, :]  # [B, 1, H]

        predictions = []
        h = None
        for _ in range(num_future):
            out, h = self.horizontal_predictor(last_hidden, h)
            predictions.append(out)
            last_hidden = out

        return torch.cat(predictions, dim=1)  # [B, num_future, H]

    def predict_error(self, hidden: torch.Tensor) -> torch.Tensor:
        """Meta-prediction: how large will my prediction error be?"""
        pooled = hidden.mean(dim=1)
        return self.error_predictor(pooled)

    def predict_confidence(self, hidden: torch.Tensor) -> torch.Tensor:
        """Predict how confident the final output will be."""
        pooled = hidden.mean(dim=1)
        return self.confidence_predictor(pooled)


# ============================================================================
# CORE INNOVATION: Error-Driven Recursive Refinement
# More prediction error → more recursive processing
# ============================================================================

class RecursiveRefinementBlock(nn.Module):
    """
    Iteratively refines hidden states until prediction error is low.

    This is the "thinking until confident" mechanism:
    1. Predict next state
    2. Compare to actual (or self-generated target)
    3. If error high, refine and repeat
    4. Stop when error < threshold or max iterations

    Unlike fixed-depth transformers, this creates DYNAMIC depth
    based on the difficulty of the input.
    """

    def __init__(
        self,
        hidden_dim: int,
        max_iterations: int = 5,
        error_threshold: float = 0.1,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.max_iterations = max_iterations
        self.error_threshold = error_threshold

        # Refinement transform
        self.refine = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Error-to-refinement strength mapping
        self.error_to_strength = nn.Sequential(
            nn.Linear(1, 32),
            nn.GELU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

        # Iteration embedding
        self.iter_emb = nn.Embedding(max_iterations, hidden_dim)

        # Track iteration statistics
        self.register_buffer('avg_iterations', torch.tensor(0.0))
        self.register_buffer('iteration_count', torch.tensor(0.0))

    def forward(
        self,
        hidden: torch.Tensor,
        target_hidden: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Refine hidden states recursively based on prediction error.

        Args:
            hidden: Current hidden states [B, S, H]
            target_hidden: Target to predict (if None, uses self-generated target)
        """
        B, S, H = hidden.shape

        # If no target, use a forward-shifted version as pseudo-target
        if target_hidden is None:
            # Predict what "refined" version should look like
            target_hidden = hidden.detach()

        current = hidden
        total_error = 0.0
        iterations_used = 0

        for i in range(self.max_iterations):
            # Compute prediction error
            error = F.mse_loss(current, target_hidden, reduction='none')
            error_magnitude = error.mean(dim=(1, 2), keepdim=True)  # [B, 1, 1]
            avg_error = error_magnitude.mean().item()

            total_error += avg_error
            iterations_used = i + 1

            # Check if error is low enough
            if avg_error < self.error_threshold:
                break

            # Convert error to refinement strength
            strength = self.error_to_strength(error_magnitude.view(B, 1))  # [B, 1]
            strength = strength.unsqueeze(1).expand(B, S, 1)  # [B, S, 1]

            # Add iteration embedding
            iter_emb = self.iter_emb(torch.tensor([i], device=device))
            iter_emb = iter_emb.expand(B, S, H)

            # Refine based on error
            refinement_input = torch.cat([current, current - target_hidden], dim=-1)
            refinement = self.refine(refinement_input)

            # Apply refinement scaled by error (more error → more change)
            current = current + refinement * strength + iter_emb * 0.05

        # Update statistics
        self.avg_iterations = 0.9 * self.avg_iterations + 0.1 * iterations_used
        self.iteration_count = self.iteration_count + 1

        return {
            'refined': current,
            'iterations': iterations_used,
            'final_error': avg_error,
            'total_error': total_error / iterations_used,
        }


# ============================================================================
# CORE INNOVATION: Strange Loop Module
# Level N feeds back to influence Level 0
# ============================================================================

class StrangeLoopModule(nn.Module):
    """
    Creates genuine self-reference: output of processing
    feeds back to influence the input representation.

    This is Hofstadter's Strange Loop in neural form:
    - The highest level of processing affects the lowest level
    - Creating recursive self-reference
    - The system "sees itself seeing"
    """

    def __init__(self, hidden_dim: int, num_levels: int = 3):
        super().__init__()
        self.num_levels = num_levels

        # Level processors
        self.level_processors = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            for _ in range(num_levels)
        ])

        # Loop-back projection (highest → lowest)
        self.loop_back = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, hidden_dim),
        )

        # Cross-level attention
        self.cross_level_attn = nn.MultiheadAttention(
            hidden_dim, num_heads=4, batch_first=True
        )

    def forward(self, hidden: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Process through levels with strange loop feedback.
        """
        B, S, H = hidden.shape

        level_states = [hidden]

        # Forward through levels
        current = hidden
        for i, processor in enumerate(self.level_processors):
            current = processor(current) + current  # Residual
            level_states.append(current)

        # Strange loop: highest level feeds back to lowest
        highest = level_states[-1]
        loop_signal = self.loop_back(highest)

        # Cross-attention: lowest level attends to highest
        lowest_query = level_states[0]
        highest_kv = level_states[-1]

        looped, _ = self.cross_level_attn(
            lowest_query, highest_kv, highest_kv
        )

        # Combine: original + loop signal + cross-attention
        output = hidden + loop_signal * 0.1 + looped * 0.1

        # Compute loop coherence (similarity between levels)
        loop_coherence = F.cosine_similarity(
            level_states[0].mean(dim=1),
            level_states[-1].mean(dim=1),
            dim=-1
        ).mean()

        return {
            'output': output,
            'level_states': level_states,
            'loop_coherence': loop_coherence.item(),
        }


# ============================================================================
# FULL MODEL: Recursive Predictive Self-Model Qwen3
# ============================================================================

class RPSMQwen3(nn.Module):
    """
    Qwen3 with Recursive Predictive Self-Model architecture.

    Novel features:
    1. Self-prediction as core training objective
    2. Error-driven recursive refinement
    3. Strange loop self-reference
    4. Dynamic depth based on prediction confidence
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-4B",
        lora_r: int = 16,
        max_iterations: int = 5,
        error_threshold: float = 0.1,
        distill_weight: float = 0.3,
        self_pred_weight: float = 0.5,  # Weight for self-prediction loss
    ):
        super().__init__()
        self.distill_weight = distill_weight
        self.self_pred_weight = self_pred_weight

        print(f"\nLoading {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )

        self.hidden_dim = self.model.config.hidden_size
        self.num_layers = self.model.config.num_hidden_layers

        print(f"  Hidden: {self.hidden_dim}, Layers: {self.num_layers}")

        # LoRA
        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_r * 2,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )
        self.model = get_peft_model(self.model, lora_config)

        # RPSM Components
        self.self_predictor = PredictiveSelfModel(
            self.hidden_dim, self.num_layers
        ).to(device).to(torch.bfloat16)

        self.recursive_refiner = RecursiveRefinementBlock(
            self.hidden_dim, max_iterations, error_threshold
        ).to(device).to(torch.bfloat16)

        self.strange_loop = StrangeLoopModule(
            self.hidden_dim, num_levels=3
        ).to(device).to(torch.bfloat16)

        # Reference for distillation
        print("Loading frozen reference...")
        self.reference = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        for p in self.reference.parameters():
            p.requires_grad = False
        self.reference.eval()

        # Count params
        lora_p = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        rpsm_p = (
            sum(p.numel() for p in self.self_predictor.parameters()) +
            sum(p.numel() for p in self.recursive_refiner.parameters()) +
            sum(p.numel() for p in self.strange_loop.parameters())
        )

        print(f"\n✓ Model initialized")
        print(f"  LoRA: {lora_p:,}")
        print(f"  RPSM modules: {rpsm_p:,}")

    def forward_with_rpsm(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with Recursive Predictive Self-Model.
        """
        # Get hidden states from all layers
        outputs = self.model.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        all_hidden = outputs.hidden_states

        # Select key layers for processing
        early_hidden = all_hidden[self.num_layers // 3]
        mid_hidden = all_hidden[2 * self.num_layers // 3]
        late_hidden = all_hidden[-1]

        # 1. Self-prediction: predict next layer from current
        early_to_mid_pred = self.self_predictor.predict_next_layer(early_hidden.to(device))
        mid_to_late_pred = self.self_predictor.predict_next_layer(mid_hidden.to(device))

        # Self-prediction losses
        vertical_loss = (
            F.mse_loss(early_to_mid_pred, mid_hidden.to(device).detach()) +
            F.mse_loss(mid_to_late_pred, late_hidden.to(device).detach())
        ) / 2

        # 2. Meta-prediction: can model predict its own error?
        predicted_error = self.self_predictor.predict_error(mid_hidden.to(device))
        actual_error = F.mse_loss(mid_to_late_pred, late_hidden.to(device).detach(), reduction='none')
        actual_error_scalar = actual_error.mean(dim=(1, 2), keepdim=True)
        # Normalize to 0-1 range
        actual_error_normalized = torch.sigmoid(actual_error_scalar.squeeze() * 10)
        meta_pred_loss = F.mse_loss(predicted_error.squeeze(), actual_error_normalized)

        # 3. Recursive refinement based on prediction error
        refine_result = self.recursive_refiner(
            mid_hidden.to(device),
            late_hidden.to(device).detach()
        )

        # 4. Strange loop processing
        loop_result = self.strange_loop(refine_result['refined'])

        # 5. Final output
        final_hidden = late_hidden + loop_result['output'] * 0.1
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

        # Distillation
        distill_loss = torch.tensor(0.0, device=device)
        if labels is not None:
            with torch.no_grad():
                ref_out = self.reference(input_ids=input_ids, attention_mask=attention_mask)
            T = 2.0
            distill_loss = F.kl_div(
                F.log_softmax(logits / T, dim=-1),
                F.softmax(ref_out.logits / T, dim=-1),
                reduction='batchmean',
            ) * (T * T)

        # Total loss
        total_loss = None
        if lm_loss is not None:
            total_loss = (
                lm_loss +
                self.distill_weight * distill_loss +
                self.self_pred_weight * vertical_loss +
                0.1 * meta_pred_loss
            )

        return {
            'loss': total_loss,
            'lm_loss': lm_loss,
            'distill_loss': distill_loss,
            'self_pred_loss': vertical_loss,
            'meta_pred_loss': meta_pred_loss,
            'logits': logits,
            'iterations': refine_result['iterations'],
            'refinement_error': refine_result['final_error'],
            'loop_coherence': loop_result['loop_coherence'],
        }

    @torch.no_grad()
    def evaluate_self_prediction(self, num_trials: int = 20) -> Dict[str, float]:
        """
        Evaluate self-prediction capabilities.

        Metrics:
        1. Vertical prediction accuracy (next layer)
        2. Meta-prediction accuracy (can predict own error)
        3. Refinement efficiency (iterations needed)
        4. Loop coherence (strange loop quality)
        """
        self.eval()

        test_prompts = [
            "The capital of France is",
            "Let me think step by step about this math problem",
            "I am uncertain about",
            "The recursive nature of self-reflection",
            "When I analyze my own reasoning",
        ]

        vertical_errors = []
        meta_pred_accuracies = []
        iterations_used = []
        loop_coherences = []

        for prompt in test_prompts * (num_trials // len(test_prompts) + 1):
            if len(vertical_errors) >= num_trials:
                break

            inputs = self.tokenizer(prompt, return_tensors="pt").to(device)
            out = self.forward_with_rpsm(inputs.input_ids, inputs.attention_mask)

            vertical_errors.append(out['self_pred_loss'].item())
            meta_pred_accuracies.append(1.0 - out['meta_pred_loss'].item())
            iterations_used.append(out['iterations'])
            loop_coherences.append(out['loop_coherence'])

        return {
            'vertical_pred_error': np.mean(vertical_errors),
            'meta_pred_accuracy': np.mean(meta_pred_accuracies),
            'avg_iterations': np.mean(iterations_used),
            'avg_loop_coherence': np.mean(loop_coherences),
        }

    @torch.no_grad()
    def generate_with_rpsm(
        self,
        prompt: str,
        max_tokens: int = 80
    ) -> Tuple[str, Dict]:
        """Generate with RPSM stats."""
        self.eval()
        inputs = self.tokenizer(prompt, return_tensors="pt").to(device)

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=0.7,
            do_sample=True,
            pad_token_id=self.tokenizer.pad_token_id,
        )

        text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        result = self.forward_with_rpsm(inputs.input_ids, inputs.attention_mask)

        stats = {
            'iterations': result['iterations'],
            'refinement_error': result['refinement_error'],
            'loop_coherence': result['loop_coherence'],
        }

        return text, stats


# ============================================================================
# DATASET
# ============================================================================

class SelfPredictionDataset(Dataset):
    """Dataset for training self-prediction."""

    def __init__(self, tokenizer, num_samples: int = 300, max_len: int = 256):
        self.tokenizer = tokenizer
        self.max_len = max_len

        prompts = [
            "Let me predict what I will think next:",
            "My next thought will be about",
            "I anticipate my reasoning will lead to",
            "Predicting my own internal state:",
            "What will my next layer of processing conclude?",
            "I expect my confidence to be",
            "My self-model suggests that",
            "Recursively examining my thoughts:",
            "The strange loop of my reasoning shows",
            "My prediction error indicates that",
        ]

        self.samples = [prompts[i % len(prompts)] for i in range(num_samples)]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.samples[idx],
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt',
        )
        return {
            'input_ids': enc['input_ids'].squeeze(0),
            'attention_mask': enc['attention_mask'].squeeze(0),
            'labels': enc['input_ids'].squeeze(0).clone(),
        }


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Main training and evaluation."""

    if HAS_WANDB:
        wandb.init(
            project="z1406-rpsm",
            name="qwen3-4b-recursive-predictive",
            mode="offline",
        )

    model = RPSMQwen3(
        model_name="Qwen/Qwen3-4B",
        lora_r=16,
        max_iterations=5,
        error_threshold=0.1,
        distill_weight=0.3,
        self_pred_weight=0.5,
    )

    print("\nCreating dataset...")
    dataset = SelfPredictionDataset(model.tokenizer, num_samples=300)
    loader = DataLoader(dataset, batch_size=2, shuffle=True)
    print(f"✓ {len(dataset)} samples")

    optimizer = torch.optim.AdamW([
        {'params': model.model.parameters(), 'lr': 1e-4},
        {'params': model.self_predictor.parameters(), 'lr': 5e-4},
        {'params': model.recursive_refiner.parameters(), 'lr': 5e-4},
        {'params': model.strange_loop.parameters(), 'lr': 5e-4},
    ])

    # ========================================
    # PRE-TRAINING EVALUATION
    # ========================================
    print("\n" + "=" * 70)
    print("PRE-TRAINING EVALUATION")
    print("=" * 70)

    pre_metrics = model.evaluate_self_prediction(num_trials=20)
    print(f"\n[Self-Prediction Metrics]")
    print(f"  Vertical prediction error: {pre_metrics['vertical_pred_error']:.4f}")
    print(f"  Meta-prediction accuracy: {pre_metrics['meta_pred_accuracy']:.1%}")
    print(f"  Avg iterations used: {pre_metrics['avg_iterations']:.2f}")
    print(f"  Loop coherence: {pre_metrics['avg_loop_coherence']:.3f}")

    # Sample generation
    print("\n[Sample Generation]")
    test_prompts = [
        "Let me predict my own reasoning:",
        "My self-model indicates that",
    ]
    for p in test_prompts:
        text, stats = model.generate_with_rpsm(p, max_tokens=60)
        print(f"  Prompt: {p}")
        print(f"  Generated: {text[len(p):len(p)+100]}...")
        print(f"  Iterations: {stats['iterations']}, Loop coherence: {stats['loop_coherence']:.3f}")
        print()

    # ========================================
    # TRAINING
    # ========================================
    print("=" * 70)
    print("TRAINING")
    print("=" * 70)

    model.train()
    epochs = 2

    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")
        losses = []

        for step, batch in enumerate(loader):
            optimizer.zero_grad()

            out = model.forward_with_rpsm(
                batch['input_ids'].to(device),
                batch['attention_mask'].to(device),
                batch['labels'].to(device),
            )

            out['loss'].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            losses.append(out['loss'].item())

            if (step + 1) % 30 == 0:
                avg = sum(losses[-30:]) / len(losses[-30:])
                print(f"  Step {step+1}: loss={avg:.4f}, self_pred={out['self_pred_loss'].item():.4f}, "
                      f"iter={out['iterations']}, coherence={out['loop_coherence']:.3f}")

                if HAS_WANDB:
                    wandb.log({
                        'train/loss': avg,
                        'train/self_pred_loss': out['self_pred_loss'].item(),
                        'train/iterations': out['iterations'],
                        'train/loop_coherence': out['loop_coherence'],
                    })

    # ========================================
    # POST-TRAINING EVALUATION
    # ========================================
    print("\n" + "=" * 70)
    print("POST-TRAINING EVALUATION")
    print("=" * 70)

    model.eval()
    post_metrics = model.evaluate_self_prediction(num_trials=20)

    print(f"\n[Self-Prediction Metrics]")
    print(f"  Vertical prediction error: {post_metrics['vertical_pred_error']:.4f} "
          f"(Δ {post_metrics['vertical_pred_error'] - pre_metrics['vertical_pred_error']:+.4f})")
    print(f"  Meta-prediction accuracy: {post_metrics['meta_pred_accuracy']:.1%} "
          f"(Δ {post_metrics['meta_pred_accuracy'] - pre_metrics['meta_pred_accuracy']:+.1%})")
    print(f"  Avg iterations used: {post_metrics['avg_iterations']:.2f} "
          f"(Δ {post_metrics['avg_iterations'] - pre_metrics['avg_iterations']:+.2f})")
    print(f"  Loop coherence: {post_metrics['avg_loop_coherence']:.3f} "
          f"(Δ {post_metrics['avg_loop_coherence'] - pre_metrics['avg_loop_coherence']:+.3f})")

    # Sample generation
    print("\n[Sample Generation]")
    test_prompts_full = [
        "Let me predict my own reasoning:",
        "My self-model indicates that",
        "Recursively examining my internal state:",
        "I predict my next thought will be",
    ]
    samples = []
    for p in test_prompts_full:
        text, stats = model.generate_with_rpsm(p, max_tokens=80)
        print(f"  Prompt: {p}")
        print(f"  Generated: {text[len(p):len(p)+150]}...")
        print(f"  Iterations: {stats['iterations']}, Loop coherence: {stats['loop_coherence']:.3f}")
        print()
        samples.append({
            'prompt': p,
            'generated': text[len(p):len(p)+150],
            'iterations': stats['iterations'],
            'loop_coherence': stats['loop_coherence'],
        })

    # ========================================
    # SUMMARY
    # ========================================
    print("=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)

    print("\n                              PRE        POST       DELTA")
    print("-" * 60)
    print(f"Vertical Pred Error:        {pre_metrics['vertical_pred_error']:>7.4f}    {post_metrics['vertical_pred_error']:>7.4f}    {post_metrics['vertical_pred_error'] - pre_metrics['vertical_pred_error']:>+7.4f}")
    print(f"Meta-Pred Accuracy:         {pre_metrics['meta_pred_accuracy']:>7.1%}    {post_metrics['meta_pred_accuracy']:>7.1%}    {post_metrics['meta_pred_accuracy'] - pre_metrics['meta_pred_accuracy']:>+7.1%}")
    print(f"Avg Iterations:             {pre_metrics['avg_iterations']:>7.2f}    {post_metrics['avg_iterations']:>7.2f}    {post_metrics['avg_iterations'] - pre_metrics['avg_iterations']:>+7.2f}")
    print(f"Loop Coherence:             {pre_metrics['avg_loop_coherence']:>7.3f}    {post_metrics['avg_loop_coherence']:>7.3f}    {post_metrics['avg_loop_coherence'] - pre_metrics['avg_loop_coherence']:>+7.3f}")

    # Save results
    results = {
        "experiment": "z1406_recursive_predictive_self_model",
        "timestamp": datetime.now().isoformat(),
        "model": "Qwen/Qwen3-4B",
        "architecture": {
            "max_iterations": 5,
            "error_threshold": 0.1,
            "self_pred_weight": 0.5,
        },
        "pre_training": pre_metrics,
        "post_training": post_metrics,
        "improvements": {
            "vertical_pred_error": post_metrics['vertical_pred_error'] - pre_metrics['vertical_pred_error'],
            "meta_pred_accuracy": post_metrics['meta_pred_accuracy'] - pre_metrics['meta_pred_accuracy'],
            "iterations": post_metrics['avg_iterations'] - pre_metrics['avg_iterations'],
            "loop_coherence": post_metrics['avg_loop_coherence'] - pre_metrics['avg_loop_coherence'],
        },
        "samples": samples,
    }

    path = "/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy/results/z1406_recursive_predictive_self_model.json"
    with open(path, 'w') as f:
        json.dump(results, f, indent=2, default=float)
    print(f"\nResults saved to: {path}")

    if HAS_WANDB:
        wandb.log({
            'final/pre_vertical_error': pre_metrics['vertical_pred_error'],
            'final/post_vertical_error': post_metrics['vertical_pred_error'],
            'final/pre_meta_accuracy': pre_metrics['meta_pred_accuracy'],
            'final/post_meta_accuracy': post_metrics['meta_pred_accuracy'],
        })
        wandb.finish()

    return results


if __name__ == "__main__":
    main()
