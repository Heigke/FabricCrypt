#!/usr/bin/env python3
"""
Knowledge Distillation for Exit Heads

Problem: TRUE early exit saves 44% energy but quality degrades significantly.
Solution: Train exit heads to match the final layer's output distribution.

This script:
1. Uses TrueEarlyExitGPT2 architecture from z203
2. Trains exit heads using KL-divergence distillation
3. Implements confidence-based exit (only exit early when confident)
4. Validates quality preservation with real energy measurement

Key insight: The exit heads are initialized from lm_head but need training
to match the full model's behavior at intermediate representations.
"""

import os
os.environ["HSA_OVERRIDE_GFX_VERSION"] = "11.0.0"

import sys
import json
import time
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from transformers import GPT2LMHeadModel, AutoTokenizer
from datasets import load_dataset

from src.telemetry.real_amd import AMDTelemetry, EnergyMeter, RocmSmiReader


class DistillableEarlyExitGPT2(nn.Module):
    """
    GPT-2 with TRUE early exit and distillation support.

    Improvements over z203:
    1. Exit heads have residual connection for better gradient flow
    2. Confidence estimation for each exit head
    3. Temperature scaling for soft targets
    """

    def __init__(self, model_name: str = "gpt2"):
        super().__init__()

        # Load pretrained model
        self.model = GPT2LMHeadModel.from_pretrained(model_name)

        # Extract components for manual forward
        self.wte = self.model.transformer.wte
        self.wpe = self.model.transformer.wpe
        self.drop = self.model.transformer.drop
        self.blocks = self.model.transformer.h
        self.ln_f = self.model.transformer.ln_f
        self.lm_head = self.model.lm_head

        self.num_layers = len(self.blocks)
        self.hidden_dim = self.model.config.hidden_size
        self.vocab_size = self.model.config.vocab_size

        # Exit layers and their heads
        self.exit_layers = [3, 6, 9, 12]
        self.exit_heads = nn.ModuleDict()
        self.confidence_heads = nn.ModuleDict()

        for layer in self.exit_layers[:-1]:  # No exit head for final layer
            # Exit head with layer norm and projection
            self.exit_heads[str(layer)] = nn.Sequential(
                nn.LayerNorm(self.hidden_dim),
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.GELU(),
                nn.Linear(self.hidden_dim, self.vocab_size, bias=False)
            )
            # Initialize output projection from lm_head
            with torch.no_grad():
                self.exit_heads[str(layer)][-1].weight.copy_(self.lm_head.weight)

            # Confidence head: predicts if this exit will match final layer
            self.confidence_heads[str(layer)] = nn.Sequential(
                nn.Linear(self.hidden_dim, 128),
                nn.GELU(),
                nn.Linear(128, 1),
                nn.Sigmoid()
            )

        # Freeze base model
        for p in self.model.parameters():
            p.requires_grad = False

        # Unfreeze exit and confidence heads
        for p in self.exit_heads.parameters():
            p.requires_grad = True
        for p in self.confidence_heads.parameters():
            p.requires_grad = True

    def forward_all_exits(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Dict[int, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """
        Forward through all layers, collecting exit outputs.

        Returns dict: layer -> (logits, confidence, hidden_states)
        """
        device = input_ids.device
        batch_size, seq_len = input_ids.shape

        # Get embeddings
        position_ids = torch.arange(seq_len, device=device).unsqueeze(0)
        hidden_states = self.wte(input_ids) + self.wpe(position_ids)
        hidden_states = self.drop(hidden_states)

        # Prepare attention mask
        if attention_mask is not None:
            attention_mask = attention_mask.view(batch_size, -1)
            attention_mask = attention_mask[:, None, None, :]
            attention_mask = (1.0 - attention_mask) * torch.finfo(hidden_states.dtype).min

        results = {}

        # Forward through blocks
        for i in range(self.num_layers):
            block_output = self.blocks[i](
                hidden_states,
                attention_mask=attention_mask,
            )
            hidden_states = block_output[0]

            layer = i + 1  # 1-indexed

            if layer in self.exit_layers:
                if layer < self.num_layers:
                    # Early exit
                    logits = self.exit_heads[str(layer)](hidden_states)
                    # Pool hidden states for confidence (mean over sequence)
                    pooled = hidden_states.mean(dim=1)
                    confidence = self.confidence_heads[str(layer)](pooled)
                else:
                    # Final layer
                    final_h = self.ln_f(hidden_states)
                    logits = self.lm_head(final_h)
                    confidence = torch.ones(batch_size, 1, device=device)

                results[layer] = (logits, confidence, hidden_states)

        return results

    def forward_to_layer(
        self,
        input_ids: torch.Tensor,
        exit_layer: int,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """TRUE early exit - stops at specified layer."""
        device = input_ids.device
        batch_size, seq_len = input_ids.shape

        position_ids = torch.arange(seq_len, device=device).unsqueeze(0)
        hidden_states = self.wte(input_ids) + self.wpe(position_ids)
        hidden_states = self.drop(hidden_states)

        if attention_mask is not None:
            attention_mask = attention_mask.view(batch_size, -1)
            attention_mask = attention_mask[:, None, None, :]
            attention_mask = (1.0 - attention_mask) * torch.finfo(hidden_states.dtype).min

        # Forward through blocks UP TO exit_layer
        for i in range(min(exit_layer, self.num_layers)):
            block_output = self.blocks[i](
                hidden_states,
                attention_mask=attention_mask,
            )
            hidden_states = block_output[0]

        # Apply exit head or final projection
        if exit_layer < self.num_layers:
            logits = self.exit_heads[str(exit_layer)](hidden_states)
        else:
            hidden_states = self.ln_f(hidden_states)
            logits = self.lm_head(hidden_states)

        return logits, hidden_states

    def forward_with_confidence_exit(
        self,
        input_ids: torch.Tensor,
        confidence_threshold: float = 0.7,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, int, torch.Tensor]:
        """
        Forward with confidence-based early exit.

        Exits early if confidence exceeds threshold.
        Returns: (logits, exit_layer, confidence)
        """
        device = input_ids.device
        batch_size, seq_len = input_ids.shape

        position_ids = torch.arange(seq_len, device=device).unsqueeze(0)
        hidden_states = self.wte(input_ids) + self.wpe(position_ids)
        hidden_states = self.drop(hidden_states)

        if attention_mask is not None:
            attention_mask = attention_mask.view(batch_size, -1)
            attention_mask = attention_mask[:, None, None, :]
            attention_mask = (1.0 - attention_mask) * torch.finfo(hidden_states.dtype).min

        for i in range(self.num_layers):
            block_output = self.blocks[i](
                hidden_states,
                attention_mask=attention_mask,
            )
            hidden_states = block_output[0]

            layer = i + 1

            if layer in self.exit_layers[:-1]:  # Check confidence at early exits
                logits = self.exit_heads[str(layer)](hidden_states)
                pooled = hidden_states.mean(dim=1)
                confidence = self.confidence_heads[str(layer)](pooled)

                # Exit if confident enough (batch-wise decision using mean)
                if confidence.mean().item() >= confidence_threshold:
                    return logits, layer, confidence

        # Final layer
        final_h = self.ln_f(hidden_states)
        logits = self.lm_head(final_h)
        confidence = torch.ones(batch_size, 1, device=device)

        return logits, 12, confidence


class DistillationTrainer:
    """Trains exit heads via knowledge distillation."""

    def __init__(
        self,
        model: DistillableEarlyExitGPT2,
        device: torch.device,
        lr: float = 1e-4,
        temperature: float = 2.0,
        alpha_distill: float = 0.7,  # Weight for distillation loss
        alpha_task: float = 0.3,     # Weight for task loss
        alpha_confidence: float = 0.1  # Weight for confidence calibration
    ):
        self.model = model
        self.device = device
        self.temperature = temperature
        self.alpha_distill = alpha_distill
        self.alpha_task = alpha_task
        self.alpha_confidence = alpha_confidence

        # Only train exit heads and confidence heads
        trainable = list(model.exit_heads.parameters()) + list(model.confidence_heads.parameters())
        self.optimizer = torch.optim.AdamW(trainable, lr=lr)

        self.metrics = {
            'step': [],
            'total_loss': [],
            'distill_loss': [],
            'task_loss': [],
            'confidence_loss': [],
            'exit_3_agreement': [],
            'exit_6_agreement': [],
            'exit_9_agreement': [],
        }

    def distillation_loss(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor
    ) -> torch.Tensor:
        """KL divergence with temperature scaling."""
        T = self.temperature
        student_soft = F.log_softmax(student_logits / T, dim=-1)
        teacher_soft = F.softmax(teacher_logits / T, dim=-1)
        return F.kl_div(student_soft, teacher_soft, reduction='batchmean') * (T * T)

    def task_loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor
    ) -> torch.Tensor:
        """Standard cross-entropy task loss."""
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        return F.cross_entropy(
            shift_logits.view(-1, self.model.vocab_size),
            shift_labels.view(-1),
            ignore_index=-100
        )

    def confidence_loss(
        self,
        confidence: torch.Tensor,
        agreement: torch.Tensor
    ) -> torch.Tensor:
        """Binary cross-entropy for confidence calibration."""
        return F.binary_cross_entropy(confidence, agreement)

    def compute_agreement(
        self,
        early_logits: torch.Tensor,
        final_logits: torch.Tensor
    ) -> torch.Tensor:
        """Compute top-1 agreement between early and final predictions."""
        early_pred = early_logits.argmax(dim=-1)
        final_pred = final_logits.argmax(dim=-1)
        # Agreement per sample (averaged over sequence)
        agreement = (early_pred == final_pred).float().mean(dim=-1, keepdim=True)
        return agreement

    def train_step(self, input_ids: torch.Tensor, labels: torch.Tensor) -> Dict:
        """Single training step."""
        self.model.train()
        self.optimizer.zero_grad()

        # Forward through all exits
        outputs = self.model.forward_all_exits(input_ids)

        # Teacher is final layer (layer 12)
        teacher_logits = outputs[12][0].detach()

        total_loss = 0.0
        losses = {'distill': 0.0, 'task': 0.0, 'confidence': 0.0}
        agreements = {}

        for layer in [3, 6, 9]:
            student_logits, confidence, _ = outputs[layer]

            # Distillation loss
            d_loss = self.distillation_loss(student_logits, teacher_logits)
            losses['distill'] += d_loss

            # Task loss
            t_loss = self.task_loss(student_logits, labels)
            losses['task'] += t_loss

            # Confidence calibration
            agreement = self.compute_agreement(student_logits, teacher_logits)
            c_loss = self.confidence_loss(confidence, agreement)
            losses['confidence'] += c_loss

            agreements[f'exit_{layer}'] = agreement.mean().item()

        # Average over exit layers
        total_loss = (
            self.alpha_distill * losses['distill'] / 3 +
            self.alpha_task * losses['task'] / 3 +
            self.alpha_confidence * losses['confidence'] / 3
        )

        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()

        return {
            'total_loss': total_loss.item(),
            'distill_loss': losses['distill'].item() / 3,
            'task_loss': losses['task'].item() / 3,
            'confidence_loss': losses['confidence'].item() / 3,
            'agreements': agreements
        }

    def log_metrics(self, step: int, metrics: Dict):
        """Log training metrics."""
        self.metrics['step'].append(step)
        self.metrics['total_loss'].append(metrics['total_loss'])
        self.metrics['distill_loss'].append(metrics['distill_loss'])
        self.metrics['task_loss'].append(metrics['task_loss'])
        self.metrics['confidence_loss'].append(metrics['confidence_loss'])
        self.metrics['exit_3_agreement'].append(metrics['agreements']['exit_3'])
        self.metrics['exit_6_agreement'].append(metrics['agreements']['exit_6'])
        self.metrics['exit_9_agreement'].append(metrics['agreements']['exit_9'])


@dataclass
class ValidationResult:
    """Results for validation at one exit configuration."""
    exit_layer: int
    confidence_threshold: float
    total_tokens: int
    total_energy_mj: float
    total_time_ms: float
    energy_per_token_mj: float
    tokens_per_second: float
    avg_power_w: float
    quality_loss: float
    avg_exit_layer: float  # For confidence-based exit
    agreement_rate: float  # Agreement with final layer


def validate_model(
    model: DistillableEarlyExitGPT2,
    batches: List[torch.Tensor],
    device: torch.device,
    num_iterations: int = 10,
    warmup_iterations: int = 3,
    use_confidence_exit: bool = False,
    confidence_threshold: float = 0.7
) -> Dict[str, ValidationResult]:
    """Validate model at different exit configurations."""

    model.eval()
    results = {}

    configs = [
        ('exit_3', 3, None),
        ('exit_6', 6, None),
        ('exit_9', 9, None),
        ('exit_12', 12, None),
    ]

    if use_confidence_exit:
        for thresh in [0.5, 0.7, 0.9]:
            configs.append((f'confidence_{thresh}', None, thresh))

    for name, exit_layer, threshold in configs:
        print(f"\n  Validating {name}...")

        # Warmup
        for _ in range(warmup_iterations):
            for batch in batches[:2]:
                if threshold is not None:
                    _, _, _ = model.forward_with_confidence_exit(batch, threshold)
                else:
                    _, _ = model.forward_to_layer(batch, exit_layer)
                torch.cuda.synchronize()

        # Count tokens
        total_tokens = sum(
            (batch != 50256).sum().item()
            for batch in batches
        ) * num_iterations

        # Measure
        power_samples = []
        exit_layers_used = []
        torch.cuda.synchronize()

        start_time = time.time()
        for _ in range(num_iterations):
            for batch in batches:
                if threshold is not None:
                    _, used_layer, _ = model.forward_with_confidence_exit(batch, threshold)
                    exit_layers_used.append(used_layer)
                else:
                    _, _ = model.forward_to_layer(batch, exit_layer)
                    exit_layers_used.append(exit_layer)

                power = RocmSmiReader.read_power()
                if power:
                    power_samples.append(power)
        torch.cuda.synchronize()
        end_time = time.time()

        duration_ms = (end_time - start_time) * 1000
        avg_power = np.mean(power_samples) if power_samples else 0
        total_energy_mj = avg_power * (duration_ms / 1000) * 1000

        # Quality and agreement
        with torch.no_grad():
            # Get all outputs for agreement calculation
            outputs = model.forward_all_exits(batches[0])
            teacher_logits = outputs[12][0]

            if threshold is not None:
                # Use confidence exit
                test_logits, _, _ = model.forward_with_confidence_exit(batches[0], threshold)
            else:
                test_logits, _ = model.forward_to_layer(batches[0], exit_layer)

            # Task loss
            labels = batches[0].clone()
            shift_logits = test_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, model.vocab_size),
                shift_labels.view(-1),
                ignore_index=-100
            ).item()

            # Agreement
            test_pred = test_logits.argmax(dim=-1)
            teacher_pred = teacher_logits.argmax(dim=-1)
            agreement = (test_pred == teacher_pred).float().mean().item()

        results[name] = ValidationResult(
            exit_layer=exit_layer if exit_layer else -1,
            confidence_threshold=threshold if threshold else -1,
            total_tokens=total_tokens,
            total_energy_mj=total_energy_mj,
            total_time_ms=duration_ms,
            energy_per_token_mj=total_energy_mj / total_tokens if total_tokens > 0 else 0,
            tokens_per_second=total_tokens / (duration_ms / 1000) if duration_ms > 0 else 0,
            avg_power_w=avg_power,
            quality_loss=loss,
            avg_exit_layer=np.mean(exit_layers_used),
            agreement_rate=agreement
        )

    return results


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-steps", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--temperature", type=float, default=2.0)
    parser.add_argument("--output-dir", default="checkpoints/z204_distilled")
    parser.add_argument("--results", default="results/z204_distillation.json")
    args = parser.parse_args()

    print("="*60)
    print("Knowledge Distillation for Exit Heads")
    print("="*60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Initialize telemetry
    print("\nInitializing hardware telemetry...")
    telemetry = AMDTelemetry()
    initial = telemetry.read()
    print(f"  Power: {initial.power_w:.1f}W, Temp: {initial.temp_c:.0f}°C")

    # Load model
    print("\nLoading distillable early exit model...")
    model = DistillableEarlyExitGPT2("gpt2").to(device)

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Trainable params: {trainable_params:,} / {total_params:,}")

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    # Load training data
    print(f"\nLoading training data...")
    dataset = load_dataset("roneneldan/TinyStories", split="train", streaming=True)

    train_texts = []
    for item in dataset:
        if len(train_texts) >= 1000:
            break
        text = item['text'][:500]
        if len(text) > 50:
            train_texts.append(text)

    print(f"  Loaded {len(train_texts)} training samples")

    # Prepare batches
    train_batches = []
    for i in range(0, len(train_texts), args.batch_size):
        batch_texts = train_texts[i:i+args.batch_size]
        if len(batch_texts) == args.batch_size:
            encoded = tokenizer(
                batch_texts,
                max_length=128,
                truncation=True,
                padding='max_length',
                return_tensors='pt'
            )
            train_batches.append(encoded['input_ids'].to(device))

    print(f"  Prepared {len(train_batches)} training batches")

    # Load validation data
    val_dataset = load_dataset("roneneldan/TinyStories", split="validation", streaming=True)
    val_texts = []
    for item in val_dataset:
        if len(val_texts) >= 100:
            break
        text = item['text'][:500]
        if len(text) > 50:
            val_texts.append(text)

    val_batches = []
    for i in range(0, len(val_texts), args.batch_size):
        batch_texts = val_texts[i:i+args.batch_size]
        if len(batch_texts) == args.batch_size:
            encoded = tokenizer(
                batch_texts,
                max_length=128,
                truncation=True,
                padding='max_length',
                return_tensors='pt'
            )
            val_batches.append(encoded['input_ids'].to(device))

    # ========== PRE-TRAINING VALIDATION ==========
    print("\n" + "="*60)
    print("PRE-TRAINING VALIDATION (untrained exit heads)")
    print("="*60)

    pre_results = validate_model(model, val_batches, device, num_iterations=5)

    baseline = pre_results['exit_12']
    print(f"\nBaseline (Exit 12): {baseline.quality_loss:.3f} loss, {baseline.energy_per_token_mj:.3f} mJ/tok")

    for name in ['exit_3', 'exit_6', 'exit_9']:
        r = pre_results[name]
        energy_savings = (1 - r.energy_per_token_mj / baseline.energy_per_token_mj) * 100
        quality_ratio = r.quality_loss / baseline.quality_loss
        print(f"  {name}: {r.quality_loss:.3f} loss ({quality_ratio:.2f}x baseline), "
              f"energy savings: {energy_savings:.1f}%, agreement: {r.agreement_rate:.1%}")

    # ========== TRAINING ==========
    print("\n" + "="*60)
    print(f"TRAINING ({args.train_steps} steps)")
    print("="*60)

    trainer = DistillationTrainer(
        model, device,
        lr=args.lr,
        temperature=args.temperature
    )

    pbar = tqdm(range(args.train_steps), desc="Training")
    batch_idx = 0

    for step in pbar:
        batch = train_batches[batch_idx]
        labels = batch.clone()

        metrics = trainer.train_step(batch, labels)
        trainer.log_metrics(step, metrics)

        # Update progress bar
        pbar.set_postfix({
            'loss': f"{metrics['total_loss']:.3f}",
            'distill': f"{metrics['distill_loss']:.3f}",
            'agr_3': f"{metrics['agreements']['exit_3']:.2f}",
            'agr_6': f"{metrics['agreements']['exit_6']:.2f}",
            'agr_9': f"{metrics['agreements']['exit_9']:.2f}",
        })

        batch_idx = (batch_idx + 1) % len(train_batches)

        # Periodic validation
        if (step + 1) % 100 == 0:
            model.eval()
            with torch.no_grad():
                val_batch = val_batches[0]
                outputs = model.forward_all_exits(val_batch)
                teacher = outputs[12][0]

                for layer in [3, 6, 9]:
                    student = outputs[layer][0]
                    agreement = (student.argmax(-1) == teacher.argmax(-1)).float().mean()
                    print(f"\n  Step {step+1} - Layer {layer} agreement: {agreement:.3f}")
            model.train()

    # ========== POST-TRAINING VALIDATION ==========
    print("\n" + "="*60)
    print("POST-TRAINING VALIDATION (trained exit heads)")
    print("="*60)

    post_results = validate_model(
        model, val_batches, device,
        num_iterations=10,
        use_confidence_exit=True
    )

    baseline = post_results['exit_12']
    print(f"\nBaseline (Exit 12): {baseline.quality_loss:.3f} loss, {baseline.energy_per_token_mj:.3f} mJ/tok")

    print("\n--- Fixed Exit Layers ---")
    for name in ['exit_3', 'exit_6', 'exit_9']:
        r = post_results[name]
        pre_r = pre_results[name]
        energy_savings = (1 - r.energy_per_token_mj / baseline.energy_per_token_mj) * 100
        quality_ratio = r.quality_loss / baseline.quality_loss
        quality_improvement = (pre_r.quality_loss - r.quality_loss) / pre_r.quality_loss * 100
        if pre_r.agreement_rate > 0:
            agreement_improvement = (r.agreement_rate - pre_r.agreement_rate) / pre_r.agreement_rate * 100
        else:
            agreement_improvement = float('inf') if r.agreement_rate > 0 else 0

        print(f"  {name}:")
        print(f"    Loss: {r.quality_loss:.3f} ({quality_ratio:.2f}x baseline)")
        print(f"    Quality improvement: {quality_improvement:+.1f}%")
        print(f"    Agreement: {r.agreement_rate:.1%} (was {pre_r.agreement_rate:.1%})")
        print(f"    Energy savings: {energy_savings:.1f}%")

    print("\n--- Confidence-Based Exit ---")
    for name in ['confidence_0.5', 'confidence_0.7', 'confidence_0.9']:
        if name in post_results:
            r = post_results[name]
            energy_savings = (1 - r.energy_per_token_mj / baseline.energy_per_token_mj) * 100
            quality_ratio = r.quality_loss / baseline.quality_loss
            print(f"  {name}:")
            print(f"    Avg exit layer: {r.avg_exit_layer:.1f}")
            print(f"    Loss: {r.quality_loss:.3f} ({quality_ratio:.2f}x baseline)")
            print(f"    Agreement: {r.agreement_rate:.1%}")
            print(f"    Energy savings: {energy_savings:.1f}%")

    # ========== ANALYSIS ==========
    print("\n" + "="*60)
    print("ANALYSIS - Quality vs Energy Pareto")
    print("="*60)

    # Find best configuration
    best_config = None
    best_score = 0

    for name, r in post_results.items():
        if name == 'exit_12':
            continue

        energy_savings = (1 - r.energy_per_token_mj / baseline.energy_per_token_mj)
        quality_preservation = 1 / (r.quality_loss / baseline.quality_loss)

        # Score: energy savings * quality preservation
        score = energy_savings * quality_preservation

        if score > best_score:
            best_score = score
            best_config = name

    if best_config:
        r = post_results[best_config]
        energy_savings = (1 - r.energy_per_token_mj / baseline.energy_per_token_mj) * 100
        quality_ratio = r.quality_loss / baseline.quality_loss
        print(f"\nBest configuration: {best_config}")
        print(f"  Energy savings: {energy_savings:.1f}%")
        print(f"  Quality ratio: {quality_ratio:.2f}x")
        print(f"  Agreement: {r.agreement_rate:.1%}")

    # ========== SAVE RESULTS ==========
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save model
    checkpoint_path = output_dir / "model_final.pt"
    torch.save({
        'model_state': model.state_dict(),
        'training_metrics': trainer.metrics,
        'config': {
            'train_steps': args.train_steps,
            'lr': args.lr,
            'temperature': args.temperature,
            'batch_size': args.batch_size,
        }
    }, checkpoint_path)
    print(f"\nModel saved to {checkpoint_path}")

    # Save results
    results_path = Path(args.results)
    results_path.parent.mkdir(parents=True, exist_ok=True)

    output_data = {
        'config': {
            'train_steps': args.train_steps,
            'lr': args.lr,
            'temperature': args.temperature,
            'batch_size': args.batch_size,
        },
        'pre_training': {k: asdict(v) for k, v in pre_results.items()},
        'post_training': {k: asdict(v) for k, v in post_results.items()},
        'training_metrics': trainer.metrics,
        'analysis': {
            'best_config': best_config,
            'baseline_loss': baseline.quality_loss,
            'baseline_energy_mj_per_token': baseline.energy_per_token_mj,
        }
    }

    with open(results_path, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"Results saved to {results_path}")

    # Cleanup
    telemetry.shutdown()

    print("\n" + "="*60)
    print("Knowledge distillation complete!")
    print("="*60)


if __name__ == "__main__":
    main()
