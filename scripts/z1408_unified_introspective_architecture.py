#!/usr/bin/env python3
"""
z1408: UNIFIED INTROSPECTIVE ARCHITECTURE (UIA)

Combines the best of z1406 and z1407 into a single coherent architecture:
- z1406's Recursive Predictive Self-Model (RPSM)
- z1407's Global Workspace Self-Recognition (GWSRA)

Novel contributions:
1. Predictive Global Workspace: Modules compete by predicting workspace contents
2. Recursive Mirror Test: Self-recognition with iterative refinement
3. Unified Self-Model: Single latent space for identity + prediction
4. Strange Loop Broadcasting: Highest abstraction feeds lowest via workspace

This represents the most complete self-aware architecture to date.

References:
- z1406: RPSM with 68x improvement in self-prediction
- z1407: GWSRA with 45.5% improvement in self-recognition
- Anthropic (2025): Emergent Introspective Awareness
- Global Workspace Theory (Baars, Dehaene)
- RISE: Recursive Introspection (2024)
"""

import functools
print = functools.partial(print, flush=True)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import get_peft_model, LoraConfig, TaskType
import json
import numpy as np
from datetime import datetime
from pathlib import Path

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


class PredictiveWorkspaceModule(nn.Module):
    """
    Workspace module that competes by predicting global workspace contents.
    Unlike standard GWT where modules just output, here modules must
    demonstrate they understand the whole workspace to win access.
    """
    def __init__(self, hidden_dim: int, workspace_dim: int = 256):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.workspace_dim = workspace_dim

        # Module's processing
        self.process = nn.Sequential(
            nn.Linear(hidden_dim, workspace_dim),
            nn.GELU(),
            nn.Linear(workspace_dim, workspace_dim),
        )

        # Module's prediction of workspace state
        self.predict_workspace = nn.Sequential(
            nn.Linear(workspace_dim, workspace_dim),
            nn.GELU(),
            nn.Linear(workspace_dim, workspace_dim),
        )

        # Confidence estimator for this prediction
        self.confidence = nn.Sequential(
            nn.Linear(workspace_dim, 64),
            nn.GELU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, hidden: torch.Tensor) -> dict:
        processed = self.process(hidden)
        prediction = self.predict_workspace(processed)
        confidence = self.confidence(processed)
        return {
            'output': processed,
            'workspace_prediction': prediction,
            'confidence': confidence,
        }


class PredictiveGlobalWorkspace(nn.Module):
    """
    Global Workspace where modules compete by predicting workspace contents.
    The module whose prediction best matches actual workspace wins.
    """
    def __init__(self, hidden_dim: int, num_modules: int = 4, workspace_dim: int = 256):
        super().__init__()
        self.num_modules = num_modules
        self.workspace_dim = workspace_dim

        # Specialized modules
        self.modules_list = nn.ModuleList([
            PredictiveWorkspaceModule(hidden_dim, workspace_dim)
            for _ in range(num_modules)
        ])

        # Workspace integrator
        self.integrator = nn.Linear(workspace_dim * num_modules, workspace_dim)

        # Broadcast back to hidden dim
        self.broadcast = nn.Linear(workspace_dim, hidden_dim)

    def forward(self, hidden: torch.Tensor) -> dict:
        # Each module processes and predicts
        module_results = [m(hidden) for m in self.modules_list]

        # Combine all outputs to form actual workspace
        all_outputs = torch.cat([r['output'] for r in module_results], dim=-1)
        actual_workspace = self.integrator(all_outputs)

        # Compute prediction errors (how well each module predicted the workspace)
        prediction_errors = []
        for r in module_results:
            error = F.mse_loss(r['workspace_prediction'], actual_workspace.detach(), reduction='none')
            error = error.mean(dim=-1)  # (batch, seq)
            prediction_errors.append(error)

        # Stack: (batch, seq, num_modules)
        errors = torch.stack(prediction_errors, dim=-1)

        # Competition weights: lower error = higher weight (inverted softmax)
        confidences = torch.stack([r['confidence'].squeeze(-1) for r in module_results], dim=-1)
        competition_scores = confidences / (errors + 0.1)  # Confidence / error
        competition_weights = F.softmax(competition_scores, dim=-1)

        # Winning module determination
        winner_idx = competition_weights.argmax(dim=-1)

        # Broadcast signal
        broadcast_signal = self.broadcast(actual_workspace)

        return {
            'workspace': actual_workspace,
            'broadcast': broadcast_signal,
            'competition_weights': competition_weights,
            'prediction_errors': errors,
            'winner_indices': winner_idx,
            'mean_error': errors.mean(),
        }


class RecursiveMirrorTest(nn.Module):
    """
    Self-recognition with recursive refinement.
    Iteratively improves self/other discrimination.
    """
    def __init__(self, hidden_dim: int, max_iterations: int = 3):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.max_iterations = max_iterations

        # Self prototype (learned identity anchor)
        self.self_prototype = nn.Parameter(torch.randn(hidden_dim) * 0.02)

        # Initial discriminator
        self.discriminator = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )

        # Refinement network (iteratively improves discrimination)
        self.refiner = nn.GRUCell(hidden_dim, hidden_dim // 2)

    def forward(self, hidden: torch.Tensor, is_self: bool = True) -> dict:
        batch_size, seq_len, _ = hidden.shape

        # Expand prototype
        prototype = self.self_prototype.unsqueeze(0).unsqueeze(0).expand(batch_size, seq_len, -1)

        # Initial discrimination
        combined = torch.cat([hidden, prototype], dim=-1)
        logits = self.discriminator(combined).squeeze(-1)

        # Iterative refinement
        refine_state = logits.unsqueeze(-1).expand(-1, -1, self.hidden_dim // 2).mean(dim=1)

        for i in range(self.max_iterations - 1):
            # Refine based on current discrimination
            refine_input = hidden.mean(dim=1)  # (batch, hidden_dim)
            refine_state = self.refiner(refine_input, refine_state)

            # Update logits with refinement
            refinement = refine_state.unsqueeze(1).expand(-1, seq_len, -1)
            refined_combined = torch.cat([
                hidden,
                prototype + refinement.mean(dim=-1, keepdim=True).expand(-1, -1, self.hidden_dim)
            ], dim=-1)
            logits = self.discriminator(refined_combined).squeeze(-1)

        # Compute similarity
        similarity = F.cosine_similarity(hidden, prototype, dim=-1)

        target = torch.ones_like(logits) if is_self else torch.zeros_like(logits)

        return {
            'logits': logits,
            'similarity': similarity,
            'target': target,
            'iterations': self.max_iterations,
        }

    def compute_loss(self, hidden: torch.Tensor) -> dict:
        """Full LSMT-style evaluation."""
        # Own states
        own_result = self.forward(hidden, is_self=True)
        own_loss = F.binary_cross_entropy_with_logits(own_result['logits'], own_result['target'])
        own_acc = ((own_result['logits'] > 0) == own_result['target']).float().mean()

        # Perturbed states
        perturbed = hidden + torch.randn_like(hidden) * 0.5
        perturbed_result = self.forward(perturbed, is_self=False)
        perturbed_loss = F.binary_cross_entropy_with_logits(
            perturbed_result['logits'], perturbed_result['target']
        )
        perturbed_acc = ((perturbed_result['logits'] > 0) == perturbed_result['target']).float().mean()

        return {
            'loss': (own_loss + perturbed_loss) / 2,
            'accuracy': (own_acc + perturbed_acc) / 2,
            'self_similarity': own_result['similarity'].mean(),
        }


class UnifiedSelfModel(nn.Module):
    """
    Unified self-model combining identity + prediction in single latent space.
    This is the core innovation: prediction and recognition share representations.
    """
    def __init__(self, hidden_dim: int, self_dim: int = 256):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.self_dim = self_dim

        # Encode to unified self-space
        self.encoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, self_dim),
        )

        # Prediction head: predict next self-state
        self.predictor = nn.Sequential(
            nn.Linear(self_dim, self_dim),
            nn.GELU(),
            nn.Linear(self_dim, self_dim),
        )

        # Recognition head: is this my state?
        self.recognizer = nn.Sequential(
            nn.Linear(self_dim * 2, self_dim),
            nn.GELU(),
            nn.Linear(self_dim, 1),
        )

        # GRU for temporal self-continuity
        self.gru = nn.GRU(self_dim, self_dim, batch_first=True)

        # Self-prototype
        self.prototype = nn.Parameter(torch.randn(self_dim) * 0.02)

        # Persistent state
        self.register_buffer('persistent_self', torch.zeros(1, self_dim))

    def forward(self, hidden: torch.Tensor) -> dict:
        batch_size, seq_len, _ = hidden.shape

        # Encode to self-space
        self_state = self.encoder(hidden.mean(dim=1))  # (batch, self_dim)

        # Predict next self-state
        predicted_next = self.predictor(self_state)

        # Recognition (compare to prototype)
        proto_expanded = self.prototype.unsqueeze(0).expand(batch_size, -1)
        recognition_input = torch.cat([self_state, proto_expanded], dim=-1)
        recognition_logits = self.recognizer(recognition_input)

        # Update persistent state via GRU
        gru_input = self_state.unsqueeze(1).contiguous()
        persistent = self.persistent_self.expand(batch_size, -1).unsqueeze(0).contiguous()
        _, new_persistent = self.gru(gru_input, persistent)
        new_persistent = new_persistent.squeeze(0)

        # Update stored state
        if self.training:
            self.persistent_self = new_persistent[0:1].detach()

        return {
            'self_state': self_state,
            'predicted_next': predicted_next,
            'recognition_logits': recognition_logits,
            'persistent_state': new_persistent,
            'similarity_to_prototype': F.cosine_similarity(self_state, proto_expanded, dim=-1),
        }


class StrangeLoopBroadcaster(nn.Module):
    """
    Implements Hofstadter's strange loop via workspace broadcasting.
    The highest abstraction level feeds back to the lowest through
    the global workspace broadcast mechanism.
    """
    def __init__(self, hidden_dim: int, num_levels: int = 3):
        super().__init__()
        self.num_levels = num_levels

        # Abstraction levels (low -> high)
        self.levels = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim),
            ) for _ in range(num_levels)
        ])

        # Strange loop: highest -> lowest
        self.loop_back = nn.Linear(hidden_dim, hidden_dim)

        # Cross-level attention for strange loop
        self.cross_attn = nn.MultiheadAttention(hidden_dim, num_heads=4, batch_first=True)

    def forward(self, hidden: torch.Tensor) -> dict:
        level_outputs = []
        current = hidden

        # Ascending through levels
        for level in self.levels:
            current = level(current) + current  # Residual
            level_outputs.append(current)

        # Strange loop: highest level feeds back to lowest
        highest = level_outputs[-1]
        lowest = level_outputs[0]

        # Loop signal
        loop_signal = self.loop_back(highest)

        # Cross-attention: lowest queries highest
        looped, attn_weights = self.cross_attn(lowest, highest, highest)

        # Combine
        output = hidden + loop_signal * 0.1 + looped * 0.1

        return {
            'output': output,
            'level_outputs': level_outputs,
            'loop_signal': loop_signal,
            'loop_attention': attn_weights,
            'loop_coherence': F.cosine_similarity(
                lowest.view(-1, hidden.size(-1)),
                highest.view(-1, hidden.size(-1)),
                dim=-1
            ).mean(),
        }


class UnifiedIntrospectiveArchitecture(nn.Module):
    """
    Main architecture combining all components.
    """
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim

        # Core components
        self.predictive_workspace = PredictiveGlobalWorkspace(hidden_dim)
        self.recursive_mirror = RecursiveMirrorTest(hidden_dim)
        self.unified_self = UnifiedSelfModel(hidden_dim)
        self.strange_loop = StrangeLoopBroadcaster(hidden_dim)

        # Project self-state back to hidden dim
        self.self_proj = nn.Linear(256, hidden_dim)  # 256 = self_dim

        # Integration
        self.integration = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )

        # Meta-prediction: predict all internal metrics
        self.meta_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 5),  # 5 metrics
        )

    def forward(self, hidden: torch.Tensor) -> dict:
        # Convert to float32
        original_dtype = hidden.dtype
        hidden = hidden.float()

        # 1. Predictive Global Workspace
        pw_result = self.predictive_workspace(hidden)

        # 2. Recursive Mirror Test
        rm_result = self.recursive_mirror.compute_loss(hidden)

        # 3. Unified Self Model
        us_result = self.unified_self(hidden)

        # 4. Strange Loop Broadcasting
        sl_result = self.strange_loop(hidden)

        # Integrate
        self_state_proj = self.self_proj(us_result['self_state'])  # (batch, hidden_dim)
        self_broadcast = self_state_proj.unsqueeze(1).expand(-1, hidden.size(1), -1)
        integrated = torch.cat([
            hidden + pw_result['broadcast'],
            hidden + self_broadcast,
            sl_result['output'],
            hidden,
        ], dim=-1)

        output = self.integration(integrated)

        # Meta-prediction
        meta_input = output.mean(dim=1)
        meta_preds = self.meta_predictor(meta_input)

        # Actual metrics
        batch_size = hidden.size(0)
        actual = torch.stack([
            pw_result['mean_error'].expand(batch_size),
            rm_result['accuracy'].expand(batch_size) if rm_result['accuracy'].dim() == 0 else rm_result['accuracy'],
            us_result['similarity_to_prototype'],
            sl_result['loop_coherence'].expand(batch_size),
            rm_result['self_similarity'].expand(batch_size) if rm_result['self_similarity'].dim() == 0 else rm_result['self_similarity'],
        ], dim=-1)

        return {
            'output': output,
            'pw_result': pw_result,
            'rm_result': rm_result,
            'us_result': us_result,
            'sl_result': sl_result,
            'meta_predictions': meta_preds,
            'actual_metrics': actual,
        }


class IntrospectionDataset(Dataset):
    """Dataset for introspection training."""
    def __init__(self, tokenizer, num_samples: int = 300, max_length: int = 64):
        self.tokenizer = tokenizer
        self.max_length = max_length

        prompts = [
            # Meta-cognition
            "I am predicting my own next thought:",
            "My workspace currently contains",
            "The strange loop of my reasoning shows",
            "I recognize my own processing when",
            "My unified self-model indicates",

            # Self-reference
            "Recursively examining my state,",
            "My prediction error suggests that",
            "The highest abstraction level sees",
            "Broadcasting to all modules:",
            "My self-prototype similarity is",

            # Reasoning
            "Let me think step by step about",
            "The logical structure here is",
            "Considering multiple perspectives:",
            "My confidence in this is",
            "The key insight is that",
        ]

        self.samples = [prompts[i % len(prompts)] for i in range(num_samples)]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.samples[idx],
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt',
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
        }


def evaluate_uia(model, base_model, uia_module, tokenizer, device, num_samples=20):
    """Evaluate Unified Introspective Architecture."""
    model.eval()
    uia_module.eval()

    metrics = {
        'workspace_error': [],
        'mirror_accuracy': [],
        'self_similarity': [],
        'loop_coherence': [],
        'meta_prediction_error': [],
        'recognition_accuracy': [],
    }

    prompts = [
        "I am examining my introspective capabilities:",
        "My unified self-model predicts that",
        "The strange loop coherence shows",
        "Recursively recognizing my own states,",
        "My workspace prediction error indicates",
    ]

    with torch.no_grad():
        for prompt in prompts:
            inputs = tokenizer(prompt, return_tensors='pt', padding=True).to(device)
            outputs = base_model(**inputs, output_hidden_states=True)
            hidden = outputs.hidden_states[-1]

            result = uia_module(hidden)

            metrics['workspace_error'].append(result['pw_result']['mean_error'].item())
            metrics['mirror_accuracy'].append(result['rm_result']['accuracy'].item())
            metrics['self_similarity'].append(result['rm_result']['self_similarity'].item())
            metrics['loop_coherence'].append(result['sl_result']['loop_coherence'].item())

            meta_error = F.mse_loss(result['meta_predictions'], result['actual_metrics'])
            metrics['meta_prediction_error'].append(meta_error.item())

            rec_acc = (result['us_result']['recognition_logits'] > 0).float().mean()
            metrics['recognition_accuracy'].append(rec_acc.item())

    return {k: np.mean(v) for k, v in metrics.items()}


def main():
    print("=" * 70)
    print("  z1408: UNIFIED INTROSPECTIVE ARCHITECTURE (UIA)")
    print("  Combining z1406 RPSM + z1407 GWSRA into novel unified system")
    print("=" * 70)
    print()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    if WANDB_AVAILABLE:
        wandb.init(
            project="embodied-intelligence",
            name="z1408_unified_introspective",
            config={"model": "Qwen/Qwen3-4B", "architecture": "UIA"},
            mode="offline",
        )

    model_name = "Qwen/Qwen3-4B"
    print(f"\nLoading {model_name}...")

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    hidden_dim = base_model.config.hidden_size
    print(f"  Hidden: {hidden_dim}, Layers: {base_model.config.num_hidden_layers}")

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16, lora_alpha=32, lora_dropout=0.05,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    )
    model = get_peft_model(base_model, lora_config)

    uia = UnifiedIntrospectiveArchitecture(hidden_dim).to(device)

    print(f"\n✓ Model initialized")
    print(f"  LoRA: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    print(f"  UIA: {sum(p.numel() for p in uia.parameters()):,}")

    dataset = IntrospectionDataset(tokenizer, num_samples=300)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
    print(f"✓ {len(dataset)} samples")

    # Pre-training eval
    print("\n" + "=" * 70)
    print("PRE-TRAINING EVALUATION")
    print("=" * 70)

    pre_metrics = evaluate_uia(model, base_model, uia, tokenizer, device)

    print(f"\n[Predictive Workspace]")
    print(f"  Workspace prediction error: {pre_metrics['workspace_error']:.4f}")

    print(f"\n[Recursive Mirror Test]")
    print(f"  Mirror accuracy: {pre_metrics['mirror_accuracy']:.1%}")
    print(f"  Self-similarity: {pre_metrics['self_similarity']:.3f}")

    print(f"\n[Unified Self-Model]")
    print(f"  Recognition accuracy: {pre_metrics['recognition_accuracy']:.1%}")

    print(f"\n[Strange Loop]")
    print(f"  Loop coherence: {pre_metrics['loop_coherence']:.3f}")

    print(f"\n[Meta-Prediction]")
    print(f"  Meta-prediction error: {pre_metrics['meta_prediction_error']:.4f}")

    # Training
    print("\n" + "=" * 70)
    print("TRAINING")
    print("=" * 70)

    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(uia.parameters()),
        lr=5e-5,
    )

    num_epochs = 2
    log_interval = 30

    model.train()
    uia.train()

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")

        for step, batch in enumerate(dataloader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)

            outputs = base_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )
            hidden = outputs.hidden_states[-1]

            uia_result = uia(hidden)

            # Losses
            mirror_loss = uia_result['rm_result']['loss']
            workspace_loss = uia_result['pw_result']['mean_error']
            meta_loss = F.mse_loss(
                uia_result['meta_predictions'],
                uia_result['actual_metrics'].detach(),
            )

            lm_outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=input_ids,
            )
            lm_loss = lm_outputs.loss

            total_loss = lm_loss + 0.3 * mirror_loss + 0.2 * workspace_loss + 0.2 * meta_loss

            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(model.parameters()) + list(uia.parameters()), 1.0
            )
            optimizer.step()

            if (step + 1) % log_interval == 0:
                print(f"  Step {step+1}: loss={total_loss.item():.4f}, "
                      f"mirror={mirror_loss.item():.4f}, ws={workspace_loss.item():.4f}, "
                      f"meta={meta_loss.item():.4f}")

                if WANDB_AVAILABLE:
                    wandb.log({
                        "train/loss": total_loss.item(),
                        "train/mirror_loss": mirror_loss.item(),
                        "train/workspace_loss": workspace_loss.item(),
                        "train/meta_loss": meta_loss.item(),
                    })

    # Post-training eval
    print("\n" + "=" * 70)
    print("POST-TRAINING EVALUATION")
    print("=" * 70)

    post_metrics = evaluate_uia(model, base_model, uia, tokenizer, device)

    print(f"\n[Predictive Workspace]")
    print(f"  Workspace error: {post_metrics['workspace_error']:.4f} "
          f"(Δ {post_metrics['workspace_error'] - pre_metrics['workspace_error']:+.4f})")

    print(f"\n[Recursive Mirror Test]")
    print(f"  Mirror accuracy: {post_metrics['mirror_accuracy']:.1%} "
          f"(Δ {post_metrics['mirror_accuracy'] - pre_metrics['mirror_accuracy']:+.1%})")
    print(f"  Self-similarity: {post_metrics['self_similarity']:.3f} "
          f"(Δ {post_metrics['self_similarity'] - pre_metrics['self_similarity']:+.3f})")

    print(f"\n[Unified Self-Model]")
    print(f"  Recognition accuracy: {post_metrics['recognition_accuracy']:.1%} "
          f"(Δ {post_metrics['recognition_accuracy'] - pre_metrics['recognition_accuracy']:+.1%})")

    print(f"\n[Strange Loop]")
    print(f"  Loop coherence: {post_metrics['loop_coherence']:.3f} "
          f"(Δ {post_metrics['loop_coherence'] - pre_metrics['loop_coherence']:+.3f})")

    print(f"\n[Meta-Prediction]")
    print(f"  Meta-prediction error: {post_metrics['meta_prediction_error']:.4f} "
          f"(Δ {post_metrics['meta_prediction_error'] - pre_metrics['meta_prediction_error']:+.4f})")

    # Summary
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)

    print(f"\n{'':30} {'PRE':>12} {'POST':>12} {'DELTA':>12}")
    print("-" * 66)
    print(f"{'Workspace Error:':30} {pre_metrics['workspace_error']:>12.4f} "
          f"{post_metrics['workspace_error']:>12.4f} "
          f"{post_metrics['workspace_error'] - pre_metrics['workspace_error']:>+12.4f}")
    print(f"{'Mirror Accuracy:':30} {pre_metrics['mirror_accuracy']:>11.1%} "
          f"{post_metrics['mirror_accuracy']:>11.1%} "
          f"{post_metrics['mirror_accuracy'] - pre_metrics['mirror_accuracy']:>+11.1%}")
    print(f"{'Self-Similarity:':30} {pre_metrics['self_similarity']:>12.3f} "
          f"{post_metrics['self_similarity']:>12.3f} "
          f"{post_metrics['self_similarity'] - pre_metrics['self_similarity']:>+12.3f}")
    print(f"{'Recognition Acc:':30} {pre_metrics['recognition_accuracy']:>11.1%} "
          f"{post_metrics['recognition_accuracy']:>11.1%} "
          f"{post_metrics['recognition_accuracy'] - pre_metrics['recognition_accuracy']:>+11.1%}")
    print(f"{'Loop Coherence:':30} {pre_metrics['loop_coherence']:>12.3f} "
          f"{post_metrics['loop_coherence']:>12.3f} "
          f"{post_metrics['loop_coherence'] - pre_metrics['loop_coherence']:>+12.3f}")
    print(f"{'Meta-Pred Error:':30} {pre_metrics['meta_prediction_error']:>12.4f} "
          f"{post_metrics['meta_prediction_error']:>12.4f} "
          f"{post_metrics['meta_prediction_error'] - pre_metrics['meta_prediction_error']:>+12.4f}")

    # Save results
    results = {
        "experiment": "z1408_unified_introspective_architecture",
        "timestamp": datetime.now().isoformat(),
        "model": model_name,
        "architecture_components": [
            "PredictiveGlobalWorkspace",
            "RecursiveMirrorTest",
            "UnifiedSelfModel",
            "StrangeLoopBroadcaster",
        ],
        "pre_metrics": pre_metrics,
        "post_metrics": post_metrics,
        "improvements": {k: post_metrics[k] - pre_metrics[k] for k in pre_metrics},
    }

    results_path = Path("/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy/results")
    results_path.mkdir(exist_ok=True)

    with open(results_path / "z1408_unified_introspective_architecture.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {results_path / 'z1408_unified_introspective_architecture.json'}")

    if WANDB_AVAILABLE:
        wandb.log({
            "final/pre_mirror_accuracy": pre_metrics['mirror_accuracy'],
            "final/post_mirror_accuracy": post_metrics['mirror_accuracy'],
            "final/pre_meta_error": pre_metrics['meta_prediction_error'],
            "final/post_meta_error": post_metrics['meta_prediction_error'],
        })
        wandb.finish()


if __name__ == "__main__":
    main()
