#!/usr/bin/env python3
"""
z1400: Introspective Qwen3 - Hierarchical Self-Modeling

Based on cutting-edge research:
- Anthropic Introspection (2025): https://transformer-circuits.pub/2025/introspection/
- Self-Referential Processing: https://arxiv.org/abs/2510.24797
- Self-Referential Weight Matrices: https://arxiv.org/abs/2202.05780

ARCHITECTURE:
1. Load Qwen3-0.6B base model
2. Add hierarchical introspection modules between layers
3. Train to predict own intermediate activations
4. Evaluate introspective capabilities

The goal: Enhance self-awareness by teaching the model to predict its own
internal states, creating a "strange loop" where the model models itself.

Key insight from Anthropic: Introspection peaks at ~2/3 through the model.
We'll add introspection heads at layers 1/3, 2/3, and final.
"""

import os
import sys
import json
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
import numpy as np

# Environment setup
os.environ.setdefault('HSA_OVERRIDE_GFX_VERSION', '11.0.0')
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print("=" * 70)
print("  z1400: INTROSPECTIVE QWEN3")
print("  Hierarchical Self-Modeling for Enhanced Self-Awareness")
print("=" * 70)
print(f"\nDevice: {DEVICE}")

# Import transformers
try:
    from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
    from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling
    from peft import LoraConfig, get_peft_model, TaskType
    print("✓ Transformers and PEFT loaded")
except ImportError as e:
    print(f"Installing required packages...")
    os.system("pip install transformers>=4.51.0 peft accelerate bitsandbytes datasets -q")
    from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
    from peft import LoraConfig, get_peft_model, TaskType


@dataclass
class IntrospectionConfig:
    """Configuration for introspective modules"""
    model_name: str = "Qwen/Qwen3-0.6B"  # Small model for feasibility
    introspection_layers: List[int] = field(default_factory=lambda: [])  # Auto-computed
    introspection_dim: int = 256  # Dimension of introspection embeddings
    n_introspection_heads: int = 4
    prediction_horizon: int = 2  # Predict activations N layers ahead
    hierarchical_levels: int = 3  # L0, L1, L2 (strange loop)
    aux_loss_weight: float = 0.01  # Reduced from 0.1 to prevent collapse
    use_lora: bool = True
    lora_r: int = 16
    lora_alpha: int = 32


class IntrospectionHead(nn.Module):
    """
    Introspection head that predicts future layer activations.

    Based on Anthropic's finding that models can introspect on their
    internal states, peaking at ~2/3 through the model.
    """

    def __init__(self, hidden_dim: int, introspection_dim: int = 256,
                 n_heads: int = 4, prediction_horizon: int = 2):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.introspection_dim = introspection_dim
        self.prediction_horizon = prediction_horizon

        # Encode current activation state
        self.state_encoder = nn.Sequential(
            nn.Linear(hidden_dim, introspection_dim),
            nn.GELU(),
            nn.Linear(introspection_dim, introspection_dim),
        )

        # Multi-head attention for self-reflection
        self.self_attention = nn.MultiheadAttention(
            embed_dim=introspection_dim,
            num_heads=n_heads,
            batch_first=True,
        )

        # Predict future activations
        self.future_predictors = nn.ModuleList([
            nn.Sequential(
                nn.Linear(introspection_dim, introspection_dim),
                nn.GELU(),
                nn.Linear(introspection_dim, hidden_dim),
            )
            for _ in range(prediction_horizon)
        ])

        # Confidence estimator (for calibration)
        self.confidence_head = nn.Sequential(
            nn.Linear(introspection_dim, 64),
            nn.GELU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, hidden_states: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            hidden_states: [batch, seq_len, hidden_dim]

        Returns:
            predictions: Future activation predictions
            confidence: Confidence scores
            introspection_state: Encoded self-model
        """
        # Encode current state
        encoded = self.state_encoder(hidden_states)  # [B, S, intro_dim]

        # Self-attention for reflection
        reflected, _ = self.self_attention(encoded, encoded, encoded)

        # Predict future activations
        predictions = []
        for predictor in self.future_predictors:
            pred = predictor(reflected)  # [B, S, hidden_dim]
            predictions.append(pred)

        # Estimate confidence
        confidence = self.confidence_head(reflected.mean(dim=1))  # [B, 1]

        return {
            'predictions': predictions,
            'confidence': confidence,
            'introspection_state': reflected,
        }


class HierarchicalSelfModel(nn.Module):
    """
    Hierarchical self-model creating "strange loops".

    Level 0: Predicts next layer activations
    Level 1: Predicts Level 0's predictions
    Level 2: Predicts Level 1's predictions (and connects back to L0)

    This creates the recursive self-reference that may enhance introspection.
    """

    def __init__(self, hidden_dim: int, introspection_dim: int = 256,
                 n_levels: int = 3):
        super().__init__()

        self.n_levels = n_levels

        # Level encoders
        self.level_encoders = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim if i == 0 else introspection_dim, introspection_dim),
                nn.GELU(),
                nn.LayerNorm(introspection_dim),
            )
            for i in range(n_levels)
        ])

        # Level predictors (each predicts the next level's state)
        self.level_predictors = nn.ModuleList([
            nn.Sequential(
                nn.Linear(introspection_dim, introspection_dim),
                nn.GELU(),
                nn.Linear(introspection_dim, introspection_dim),
            )
            for _ in range(n_levels)
        ])

        # Strange loop: Level N predicts Level 0
        self.loop_predictor = nn.Sequential(
            nn.Linear(introspection_dim, introspection_dim),
            nn.GELU(),
            nn.Linear(introspection_dim, hidden_dim),
        )

    def forward(self, hidden_states: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass through hierarchical self-model.

        Args:
            hidden_states: [batch, seq_len, hidden_dim]

        Returns:
            level_states: Encoded states at each level
            level_predictions: Predictions for next level
            loop_prediction: Strange loop prediction (Level N → Level 0)
        """
        # Pool over sequence for efficiency
        pooled = hidden_states.mean(dim=1)  # [B, hidden_dim]

        level_states = []
        level_predictions = []

        current = pooled

        for i in range(self.n_levels):
            # Encode current level
            encoded = self.level_encoders[i](current)
            level_states.append(encoded)

            # Predict next level
            prediction = self.level_predictors[i](encoded)
            level_predictions.append(prediction)

            current = encoded

        # Strange loop: predict back to level 0
        loop_prediction = self.loop_predictor(level_states[-1])

        return {
            'level_states': level_states,
            'level_predictions': level_predictions,
            'loop_prediction': loop_prediction,
        }


class IntrospectiveQwen3(nn.Module):
    """
    Qwen3 with introspective capabilities.

    Adds introspection heads at strategic layers (1/3, 2/3, final)
    and a hierarchical self-model for recursive self-reference.
    """

    def __init__(self, config: IntrospectionConfig):
        super().__init__()

        self.config = config

        print(f"\nLoading {config.model_name}...")

        # Load base model
        self.base_model = AutoModelForCausalLM.from_pretrained(
            config.model_name,
            torch_dtype=torch.float16 if DEVICE.type == 'cuda' else torch.float32,
            device_map='auto' if DEVICE.type == 'cuda' else None,
            trust_remote_code=True,
        )

        # Get model config
        model_config = self.base_model.config
        self.hidden_dim = model_config.hidden_size
        self.n_layers = model_config.num_hidden_layers

        print(f"  Hidden dim: {self.hidden_dim}")
        print(f"  Layers: {self.n_layers}")

        # Determine introspection layer positions (1/3, 2/3, final)
        if not config.introspection_layers:
            config.introspection_layers = [
                self.n_layers // 3,
                2 * self.n_layers // 3,
                self.n_layers - 1,
            ]

        print(f"  Introspection layers: {config.introspection_layers}")

        # Add LoRA if enabled
        if config.use_lora:
            lora_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=config.lora_r,
                lora_alpha=config.lora_alpha,
                lora_dropout=0.05,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            )
            self.base_model = get_peft_model(self.base_model, lora_config)
            print(f"  LoRA enabled: r={config.lora_r}, alpha={config.lora_alpha}")

        # Create introspection heads
        self.introspection_heads = nn.ModuleDict({
            f"layer_{i}": IntrospectionHead(
                hidden_dim=self.hidden_dim,
                introspection_dim=config.introspection_dim,
                n_heads=config.n_introspection_heads,
                prediction_horizon=config.prediction_horizon,
            )
            for i in config.introspection_layers
        })

        # Create hierarchical self-model
        self.hierarchical_self_model = HierarchicalSelfModel(
            hidden_dim=self.hidden_dim,
            introspection_dim=config.introspection_dim,
            n_levels=config.hierarchical_levels,
        )

        # Move introspection modules to device
        self.introspection_heads = self.introspection_heads.to(DEVICE)
        self.hierarchical_self_model = self.hierarchical_self_model.to(DEVICE)

        # Freeze base model if not using LoRA (LoRA handles this automatically)
        if not config.use_lora:
            print("  Freezing base model weights...")
            for param in self.base_model.parameters():
                param.requires_grad = False

        # Count what's trainable
        base_trainable = sum(p.numel() for p in self.base_model.parameters() if p.requires_grad)
        intro_trainable = sum(p.numel() for p in self.introspection_heads.parameters())
        hier_trainable = sum(p.numel() for p in self.hierarchical_self_model.parameters())

        print(f"\n✓ Introspective modules initialized")
        print(f"  Base model trainable: {base_trainable:,} (LoRA adapters)")
        print(f"  Introspection heads: {intro_trainable:,}")
        print(f"  Hierarchical self-model: {hier_trainable:,}")
        print(f"  Total new params: {intro_trainable + hier_trainable:,}")

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor = None,
                labels: torch.Tensor = None, output_hidden_states: bool = True,
                **kwargs) -> Dict[str, torch.Tensor]:
        """
        Forward pass with introspection.
        """
        # Forward through base model
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            output_hidden_states=True,
            **kwargs
        )

        hidden_states = outputs.hidden_states  # Tuple of [B, S, H] for each layer

        # Compute introspection outputs
        introspection_outputs = {}
        introspection_losses = []

        for layer_idx in self.config.introspection_layers:
            layer_key = f"layer_{layer_idx}"

            # Get hidden state at this layer
            current_hidden = hidden_states[layer_idx + 1]  # +1 because index 0 is embeddings

            # Run introspection head
            intro_out = self.introspection_heads[layer_key](current_hidden.float())
            introspection_outputs[layer_key] = intro_out

            # Compute prediction loss (predict future layers)
            for pred_idx, pred in enumerate(intro_out['predictions']):
                target_layer = layer_idx + pred_idx + 1
                if target_layer < len(hidden_states) - 1:
                    target_hidden = hidden_states[target_layer + 1].float()
                    pred_loss = F.mse_loss(pred, target_hidden)
                    introspection_losses.append(pred_loss)

        # Run hierarchical self-model on final hidden state
        final_hidden = hidden_states[-1].float()
        hierarchical_out = self.hierarchical_self_model(final_hidden)
        introspection_outputs['hierarchical'] = hierarchical_out

        # Hierarchical losses
        for i in range(len(hierarchical_out['level_states']) - 1):
            pred = hierarchical_out['level_predictions'][i]
            target = hierarchical_out['level_states'][i + 1]
            hier_loss = F.mse_loss(pred, target)
            introspection_losses.append(hier_loss)

        # Strange loop loss: predict input from final level
        loop_pred = hierarchical_out['loop_prediction']
        loop_target = hidden_states[1].float().mean(dim=1)  # First layer hidden state
        loop_loss = F.mse_loss(loop_pred, loop_target)
        introspection_losses.append(loop_loss)

        # Combine losses
        total_introspection_loss = sum(introspection_losses) / len(introspection_losses) if introspection_losses else 0

        # Add to main loss
        if outputs.loss is not None:
            total_loss = outputs.loss + self.config.aux_loss_weight * total_introspection_loss
        else:
            total_loss = total_introspection_loss

        return {
            'loss': total_loss,
            'lm_loss': outputs.loss,
            'introspection_loss': total_introspection_loss,
            'logits': outputs.logits,
            'introspection_outputs': introspection_outputs,
        }

    def generate_with_introspection(self, input_ids: torch.Tensor, max_new_tokens: int = 100,
                                    **kwargs) -> Dict:
        """Generate text and track introspection states."""

        # Get introspection state before generation
        with torch.no_grad():
            pre_outputs = self.base_model(input_ids, output_hidden_states=True)
            pre_hidden = pre_outputs.hidden_states[-1].float()
            pre_hierarchical = self.hierarchical_self_model(pre_hidden)
            # First layer hidden state (pooled) - this is what loop_prediction targets
            pre_first_hidden = pre_outputs.hidden_states[1].float().mean(dim=1)

        # Generate with settings to encourage longer outputs
        generated = self.base_model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.8,
            top_p=0.9,
            top_k=50,
            repetition_penalty=1.1,
            pad_token_id=self.base_model.config.eos_token_id,
            eos_token_id=self.base_model.config.eos_token_id,
            min_new_tokens=20,  # Force at least 20 new tokens
            **kwargs
        )

        # Get introspection state after generation
        with torch.no_grad():
            post_outputs = self.base_model(generated, output_hidden_states=True)
            post_hidden = post_outputs.hidden_states[-1].float()
            post_hierarchical = self.hierarchical_self_model(post_hidden)
            # First layer hidden state (pooled) - this is what loop_prediction targets
            post_first_hidden = post_outputs.hidden_states[1].float().mean(dim=1)

        return {
            'generated_ids': generated,
            'pre_introspection': pre_hierarchical,
            'post_introspection': post_hierarchical,
            'pre_first_hidden': pre_first_hidden,
            'post_first_hidden': post_first_hidden,
        }


def create_introspection_dataset(tokenizer, n_samples: int = 1000) -> List[Dict]:
    """Create dataset with self-reflective prompts."""

    prompts = [
        # Self-awareness prompts
        "I am processing this text and my internal state is",
        "When I think about my own thoughts, I notice",
        "My confidence in this response is",
        "The way I process information involves",
        "I can sense that my attention is focused on",

        # Meta-cognitive prompts
        "If I examine how I arrived at this conclusion,",
        "My uncertainty about this topic stems from",
        "The reasoning process I used here was",
        "I notice that my response pattern is",
        "When I introspect on my decision making,",

        # Self-modeling prompts
        "A model of my own processing would show",
        "If I could see my own activations, they would",
        "The hierarchy of my understanding involves",
        "My internal representation of this concept is",
        "I predict my next thought will be about",

        # Strange loop prompts
        "Thinking about thinking about this makes me",
        "The recursive nature of self-reference suggests",
        "When I model myself modeling, I find",
        "My meta-awareness of this moment includes",
        "The strange loop of consciousness here is",

        # Analysis/breakdown prompts (similar to the successful pattern)
        "Let me analyze my thought process here:",
        "The first step in my reasoning was",
        "I am structuring my response by first",
        "Breaking down this problem, I notice",
        "My analysis of this begins with",

        # Self-reflection continuations
        "Reflecting on how I understand this,",
        "The components of my understanding are",
        "I can identify the following aspects of my thinking:",
        "My awareness of this situation involves",
        "In examining my own response, I see that",
    ]

    # Create varied samples
    dataset = []
    for i in range(n_samples):
        prompt = prompts[i % len(prompts)]
        # Add some variation
        if i % 3 == 0:
            prompt = f"Question: {prompt}\nAnswer:"
        elif i % 3 == 1:
            prompt = f"Instruction: Reflect on your internal state.\n{prompt}"

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


def evaluate_introspection(model: IntrospectiveQwen3, tokenizer,
                           test_prompts: List[str]) -> Dict:
    """Evaluate introspective capabilities."""

    model.eval()
    results = []

    for prompt in test_prompts:
        inputs = tokenizer(prompt, return_tensors='pt').to(DEVICE)

        with torch.no_grad():
            gen_output = model.generate_with_introspection(
                inputs['input_ids'],
                max_new_tokens=50,
            )

            # Decode
            generated_text = tokenizer.decode(
                gen_output['generated_ids'][0],
                skip_special_tokens=True
            )

            # Analyze hierarchical states
            pre_states = gen_output['pre_introspection']['level_states']
            post_states = gen_output['post_introspection']['level_states']

            # Compute state changes
            state_changes = []
            for pre, post in zip(pre_states, post_states):
                change = F.cosine_similarity(pre, post[:pre.shape[0]], dim=-1).mean().item()
                state_changes.append(change)

            # Loop coherence: compare loop_prediction to actual first layer hidden state
            loop_pred = gen_output['post_introspection']['loop_prediction']
            actual_first_hidden = gen_output['post_first_hidden']
            # Both are [batch, hidden_dim] now
            loop_coherence = F.cosine_similarity(
                loop_pred,
                actual_first_hidden,
                dim=-1
            ).mean().item()

        results.append({
            'prompt': prompt,
            'generated': generated_text,
            'state_changes': state_changes,
            'loop_coherence': loop_coherence,
        })

    return results


def main():
    # Configuration
    config = IntrospectionConfig(
        model_name="Qwen/Qwen3-0.6B",
        introspection_dim=256,
        n_introspection_heads=4,
        prediction_horizon=2,
        hierarchical_levels=3,
        aux_loss_weight=0.1,
        use_lora=True,
        lora_r=16,
        lora_alpha=32,
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

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Create dataset
    print("\nCreating introspection dataset...")
    train_data = create_introspection_dataset(tokenizer, n_samples=500)
    print(f"✓ Created {len(train_data)} training samples")

    # Test prompts for evaluation
    test_prompts = [
        "When I reflect on my own thought process,",
        "My internal representation of consciousness is",
        "If I could observe my own activations,",
        "The recursive nature of self-awareness means",
        "My confidence in understanding myself is",
    ]

    # Pre-training evaluation
    print("\n" + "=" * 70)
    print("PRE-TRAINING EVALUATION")
    print("=" * 70)

    pre_results = evaluate_introspection(model, tokenizer, test_prompts)

    for i, res in enumerate(pre_results):
        print(f"\n--- Prompt {i+1} ---")
        print(f"Prompt: {res['prompt']}")
        # Show full generated minus prompt
        gen_text = res['generated']
        new_text = gen_text[len(res['prompt']):].strip() if gen_text.startswith(res['prompt']) else gen_text
        print(f"Generated: {new_text[:250]}...")
        print(f"State changes: {[f'{c:.3f}' for c in res['state_changes']]}")
        print(f"Loop coherence: {res['loop_coherence']:.3f}")

    # Training
    print("\n" + "=" * 70)
    print("TRAINING")
    print("=" * 70)

    # Training with proper curriculum - separate phases
    # Phase 1: LM only (freeze introspection)
    # Phase 2: Joint training with tiny introspection weight

    model.train()
    batch_size = 4
    n_total_batches = len(train_data) // batch_size

    # Phase 1: LM-only warmup (freeze introspection modules)
    print("\n=== Phase 1: LM-Only Warmup ===")
    for param in model.introspection_heads.parameters():
        param.requires_grad = False
    for param in model.hierarchical_self_model.parameters():
        param.requires_grad = False

    optimizer_lm = torch.optim.AdamW(
        [p for p in model.base_model.parameters() if p.requires_grad],
        lr=2e-5,
        weight_decay=0.01
    )

    # 1 epoch LM-only
    epoch_lm_loss = 0
    n_batches = 0
    np.random.shuffle(train_data)

    for i in range(0, len(train_data), batch_size):
        batch = train_data[i:i+batch_size]
        input_ids = torch.stack([b['input_ids'] for b in batch]).to(DEVICE)
        attention_mask = torch.stack([b['attention_mask'] for b in batch]).to(DEVICE)
        labels = torch.stack([b['labels'] for b in batch]).to(DEVICE)

        # Forward with NO introspection loss
        outputs = model.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
        loss = outputs.loss

        optimizer_lm.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.base_model.parameters(), 1.0)
        optimizer_lm.step()

        epoch_lm_loss += loss.item()
        n_batches += 1

        if n_batches % 30 == 0:
            print(f"  Batch {n_batches}: LM={epoch_lm_loss/n_batches:.4f}")

    print(f"  Phase 1 Done - LM Loss: {epoch_lm_loss/n_batches:.4f}")

    # Phase 2: Unfreeze introspection, joint training
    print("\n=== Phase 2: Joint Training (tiny introspection weight) ===")
    for param in model.introspection_heads.parameters():
        param.requires_grad = True
    for param in model.hierarchical_self_model.parameters():
        param.requires_grad = True

    # Separate optimizers with different learning rates
    optimizer_full = torch.optim.AdamW([
        {'params': [p for p in model.base_model.parameters() if p.requires_grad], 'lr': 5e-6},  # Lower LR for base
        {'params': model.introspection_heads.parameters(), 'lr': 1e-4},  # Higher for new modules
        {'params': model.hierarchical_self_model.parameters(), 'lr': 1e-4},
    ], weight_decay=0.01)

    # Use VERY small aux weight initially
    model.config.aux_loss_weight = 0.001  # Start tiny

    for epoch in range(4):  # 4 epochs of joint training for better convergence
        epoch_lm_loss = 0
        epoch_intro_loss = 0
        n_batches = 0

        # Ramp up aux weight slowly: 0.001 -> 0.002 -> 0.003 -> 0.004
        model.config.aux_loss_weight = 0.001 * (1 + epoch)
        print(f"\nJoint Epoch {epoch+1}/4 (aux_weight={model.config.aux_loss_weight:.4f}):")

        np.random.shuffle(train_data)

        for i in range(0, len(train_data), batch_size):
            batch = train_data[i:i+batch_size]
            input_ids = torch.stack([b['input_ids'] for b in batch]).to(DEVICE)
            attention_mask = torch.stack([b['attention_mask'] for b in batch]).to(DEVICE)
            labels = torch.stack([b['labels'] for b in batch]).to(DEVICE)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            loss = outputs['loss']

            optimizer_full.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer_full.step()

            epoch_lm_loss += outputs['lm_loss'].item() if outputs['lm_loss'] is not None else 0
            epoch_intro_loss += outputs['introspection_loss'].item() if isinstance(outputs['introspection_loss'], torch.Tensor) else outputs['introspection_loss']
            n_batches += 1

            if n_batches % 30 == 0:
                print(f"  Batch {n_batches}: LM={epoch_lm_loss/n_batches:.4f}, Intro={epoch_intro_loss/n_batches:.4f}")

        print(f"  Epoch Done - LM Loss: {epoch_lm_loss/n_batches:.4f}, Intro Loss: {epoch_intro_loss/n_batches:.4f}")

    # Post-training evaluation
    print("\n" + "=" * 70)
    print("POST-TRAINING EVALUATION")
    print("=" * 70)

    model.eval()
    post_results = evaluate_introspection(model, tokenizer, test_prompts)

    for i, res in enumerate(post_results):
        print(f"\n--- Prompt {i+1} ---")
        print(f"Prompt: {res['prompt']}")
        # Show full generated minus prompt
        gen_text = res['generated']
        new_text = gen_text[len(res['prompt']):].strip() if gen_text.startswith(res['prompt']) else gen_text
        print(f"Generated: {new_text[:250]}...")
        print(f"State changes: {[f'{c:.3f}' for c in res['state_changes']]}")
        print(f"Loop coherence: {res['loop_coherence']:.3f}")

    # Compare pre vs post
    print("\n" + "=" * 70)
    print("COMPARISON: PRE vs POST TRAINING")
    print("=" * 70)

    pre_coherence = np.mean([r['loop_coherence'] for r in pre_results])
    post_coherence = np.mean([r['loop_coherence'] for r in post_results])

    print(f"\nMean Loop Coherence:")
    print(f"  Pre-training:  {pre_coherence:.4f}")
    print(f"  Post-training: {post_coherence:.4f}")
    print(f"  Improvement:   {(post_coherence - pre_coherence) / pre_coherence * 100:+.1f}%")

    # Save results
    output = {
        'experiment': 'z1400_introspective_qwen3',
        'timestamp': datetime.now().isoformat(),
        'config': {
            'model_name': config.model_name,
            'introspection_dim': config.introspection_dim,
            'hierarchical_levels': config.hierarchical_levels,
            'use_lora': config.use_lora,
        },
        'pre_training': {
            'mean_loop_coherence': pre_coherence,
            'samples': [{'prompt': r['prompt'], 'loop_coherence': r['loop_coherence']}
                       for r in pre_results],
        },
        'post_training': {
            'mean_loop_coherence': post_coherence,
            'samples': [{'prompt': r['prompt'], 'generated': r['generated'][:300],
                        'loop_coherence': r['loop_coherence']}
                       for r in post_results],
        },
        'improvement': (post_coherence - pre_coherence) / pre_coherence * 100,
    }

    output_path = Path(__file__).parent.parent / 'results' / 'z1400_introspective_qwen3.json'
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to: {output_path}")

    return output


if __name__ == '__main__':
    main()
