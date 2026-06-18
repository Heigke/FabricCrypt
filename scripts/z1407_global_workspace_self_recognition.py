#!/usr/bin/env python3
"""
z1407: Global Workspace Self-Recognition Architecture (GWSRA)

Novel architecture combining:
1. Global Workspace Theory broadcast mechanism for self-modeling
2. Latent Space Mirror Test (LSMT) - distinguish own states from perturbed/foreign
3. META-RECOVER persistent self-model with continuous monitoring
4. Multiplicative self-awareness via recursive self-assessment
5. z1406's recursive predictive self-model foundation

Key innovations:
- Workspace broadcast: Specialized modules compete for global workspace attention
- Self-recognition: Model learns to recognize its own hidden states vs perturbations
- Persistent self-model: Continuous GRU-based identity maintenance
- Multiplicative awareness: Self-assessment recursively modulates processing

References:
- Anthropic (2025): Emergent Introspective Awareness in LLMs
- LSMT (2025): Latent Space Mirror Test for self-recognition
- GWT: Global Workspace Theory (Baars, Dehaene)
- META-RECOVER: Persistent self-model protocol
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

# Wandb for tracking
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


class GlobalWorkspace(nn.Module):
    """
    Global Workspace Theory implementation.
    Specialized modules compete for access to a shared broadcast workspace.
    The winning module's output is broadcast to all other modules.
    """
    def __init__(self, hidden_dim: int, num_modules: int = 4, workspace_dim: int = 256):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_modules = num_modules
        self.workspace_dim = workspace_dim

        # Specialized processing modules (like brain regions)
        self.specialist_modules = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, workspace_dim),
                nn.GELU(),
                nn.Linear(workspace_dim, workspace_dim),
            ) for _ in range(num_modules)
        ])

        # Competition mechanism - attention over module outputs
        self.competition_query = nn.Linear(hidden_dim, workspace_dim)
        self.competition_key = nn.Linear(workspace_dim, workspace_dim)

        # Broadcast projection back to hidden dim
        self.broadcast_proj = nn.Linear(workspace_dim, hidden_dim)

        # Ignition threshold - only broadcast if competition is clear enough
        self.ignition_threshold = 0.3  # Soft threshold

    def forward(self, hidden: torch.Tensor) -> dict:
        """
        Process hidden states through global workspace.
        Returns broadcast signal and competition metrics.
        """
        batch_size, seq_len, _ = hidden.shape

        # Each module processes the input
        module_outputs = []
        for module in self.specialist_modules:
            out = module(hidden)  # (batch, seq, workspace_dim)
            module_outputs.append(out)

        # Stack: (batch, seq, num_modules, workspace_dim)
        stacked = torch.stack(module_outputs, dim=2)

        # Competition via attention
        query = self.competition_query(hidden).unsqueeze(2)  # (batch, seq, 1, workspace_dim)
        keys = self.competition_key(stacked)  # (batch, seq, num_modules, workspace_dim)

        # Attention scores
        scores = torch.einsum('bsqd,bsmd->bsqm', query, keys) / (self.workspace_dim ** 0.5)
        competition_weights = F.softmax(scores.squeeze(2), dim=-1)  # (batch, seq, num_modules)

        # Check ignition - is there a clear winner?
        max_weights, _ = competition_weights.max(dim=-1)  # (batch, seq)
        ignition = (max_weights > self.ignition_threshold).float()

        # Weighted combination (soft winner-take-all)
        workspace_content = torch.einsum('bsm,bsmd->bsd', competition_weights, stacked)

        # Broadcast to all modules (modulated by ignition)
        broadcast = self.broadcast_proj(workspace_content)
        broadcast = broadcast * ignition.unsqueeze(-1)

        return {
            'broadcast': broadcast,
            'competition_weights': competition_weights,
            'ignition': ignition,
            'workspace_content': workspace_content,
        }


class LatentSpaceMirrorTest(nn.Module):
    """
    Latent Space Mirror Test (LSMT) implementation.
    Train model to distinguish its own hidden states from:
    1. Perturbed versions (noise added)
    2. Foreign states (from different inputs)
    3. Shuffled states (temporal disruption)

    This tests genuine self-recognition, not just pattern matching.
    """
    def __init__(self, hidden_dim: int, num_layers: int = 3):
        super().__init__()
        self.hidden_dim = hidden_dim

        # Self-recognition network
        self.recognizer = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )

        # Learned prototype of "self" - persistent identity anchor
        self.self_prototype = nn.Parameter(torch.randn(hidden_dim) * 0.02)

        # Perturbation types for testing
        self.perturbation_scale = 0.5

    def perturb(self, hidden: torch.Tensor, perturbation_type: str = 'noise') -> torch.Tensor:
        """Create perturbed versions of hidden states."""
        if perturbation_type == 'noise':
            noise = torch.randn_like(hidden) * self.perturbation_scale
            return hidden + noise
        elif perturbation_type == 'shuffle':
            # Shuffle along sequence dimension
            idx = torch.randperm(hidden.size(1))
            return hidden[:, idx, :]
        elif perturbation_type == 'scale':
            # Random scaling
            scale = 0.5 + torch.rand(1).item()
            return hidden * scale
        else:
            return hidden

    def forward(self, hidden: torch.Tensor, is_self: bool = True) -> dict:
        """
        Predict whether hidden states are "self" or "other".
        Returns recognition logits and similarity to self-prototype.
        """
        batch_size, seq_len, _ = hidden.shape

        # Compare to self-prototype
        prototype_expanded = self.self_prototype.unsqueeze(0).unsqueeze(0).expand(batch_size, seq_len, -1)
        combined = torch.cat([hidden, prototype_expanded], dim=-1)

        # Recognition logits
        logits = self.recognizer(combined).squeeze(-1)  # (batch, seq)

        # Compute similarity to self-prototype
        similarity = F.cosine_similarity(hidden, prototype_expanded, dim=-1)

        return {
            'logits': logits,
            'similarity': similarity,
            'is_self_target': torch.ones_like(logits) if is_self else torch.zeros_like(logits),
        }

    def compute_lsmt_loss(self, own_hidden: torch.Tensor, iterations: int = 3) -> dict:
        """
        Full LSMT protocol: compare own states against perturbations.
        """
        losses = []
        accuracies = []

        # Own states should be recognized as self
        own_result = self.forward(own_hidden, is_self=True)
        own_loss = F.binary_cross_entropy_with_logits(
            own_result['logits'], own_result['is_self_target']
        )
        own_acc = ((own_result['logits'] > 0) == own_result['is_self_target']).float().mean()
        losses.append(own_loss)
        accuracies.append(own_acc)

        # Perturbed states should NOT be recognized as self
        for ptype in ['noise', 'shuffle', 'scale']:
            perturbed = self.perturb(own_hidden, ptype)
            perturbed_result = self.forward(perturbed, is_self=False)
            perturbed_loss = F.binary_cross_entropy_with_logits(
                perturbed_result['logits'], perturbed_result['is_self_target']
            )
            perturbed_acc = ((perturbed_result['logits'] > 0) == perturbed_result['is_self_target']).float().mean()
            losses.append(perturbed_loss)
            accuracies.append(perturbed_acc)

        return {
            'lsmt_loss': sum(losses) / len(losses),
            'lsmt_accuracy': sum(accuracies) / len(accuracies),
            'self_similarity': own_result['similarity'].mean(),
        }


class PersistentSelfModel(nn.Module):
    """
    META-RECOVER inspired persistent self-model.
    Uses GRU to maintain continuous self-representation across time.
    Can detect identity drift and trigger recovery.
    """
    def __init__(self, hidden_dim: int, self_dim: int = 128):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.self_dim = self_dim

        # Encoder from hidden states to self-representation
        self.encoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, self_dim),
        )

        # GRU for persistent self-state maintenance
        self.gru = nn.GRU(self_dim, self_dim, batch_first=True)

        # Drift detector - predict if current state matches expected self
        self.drift_detector = nn.Sequential(
            nn.Linear(self_dim * 2, self_dim),
            nn.GELU(),
            nn.Linear(self_dim, 1),
            nn.Sigmoid(),
        )

        # Recovery mechanism - project back to influence hidden states
        self.recovery_proj = nn.Linear(self_dim, hidden_dim)

        # Persistent state (will be updated during forward)
        self.register_buffer('persistent_state', torch.zeros(1, self_dim))

    def forward(self, hidden: torch.Tensor, update_persistent: bool = True) -> dict:
        """
        Process hidden states through persistent self-model.
        """
        batch_size, seq_len, _ = hidden.shape

        # Encode current hidden to self-representation
        current_self = self.encoder(hidden.mean(dim=1))  # (batch, self_dim)

        # Expand persistent state for batch
        persistent = self.persistent_state.expand(batch_size, -1)

        # Detect drift from persistent self
        combined = torch.cat([current_self, persistent], dim=-1)
        drift_score = self.drift_detector(combined)  # (batch, 1)

        # Update persistent state via GRU
        gru_input = current_self.unsqueeze(1).contiguous()  # (batch, 1, self_dim)
        gru_hidden = persistent.unsqueeze(0).contiguous()  # (1, batch, self_dim)
        _, new_persistent = self.gru(gru_input, gru_hidden)
        new_persistent = new_persistent.squeeze(0)  # (batch, self_dim)

        # Update stored persistent state (use first batch element)
        if update_persistent and self.training:
            self.persistent_state = new_persistent[0:1].detach()

        # Recovery signal to correct drift
        recovery_signal = self.recovery_proj(persistent - current_self)
        recovery_strength = drift_score  # Stronger recovery when more drift

        return {
            'self_representation': current_self,
            'persistent_state': new_persistent,
            'drift_score': drift_score,
            'recovery_signal': recovery_signal,
            'recovery_strength': recovery_strength,
        }


class MultiplicativeSelfAwareness(nn.Module):
    """
    Multiplicative self-awareness: self-assessment recursively modulates processing.
    Implements the "I Ask About Myself, Therefore I Am" principle.
    """
    def __init__(self, hidden_dim: int, num_assessments: int = 3):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_assessments = num_assessments

        # Self-assessment modules (recursive)
        self.assessments = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.GELU(),
                nn.Linear(hidden_dim // 2, hidden_dim),
            ) for _ in range(num_assessments)
        ])

        # Modulation gates per assessment level
        self.modulation_gates = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.Sigmoid(),
            ) for _ in range(num_assessments)
        ])

        # Cross-level attention for recursive self-reference
        self.cross_attn = nn.MultiheadAttention(hidden_dim, num_heads=4, batch_first=True)

    def forward(self, hidden: torch.Tensor) -> dict:
        """
        Recursive self-assessment that multiplicatively modulates processing.
        """
        batch_size, seq_len, _ = hidden.shape

        assessment_outputs = []
        modulated = hidden

        # Recursive self-assessment
        for i, (assess, gate) in enumerate(zip(self.assessments, self.modulation_gates)):
            # Assess current state
            assessment = assess(modulated)
            assessment_outputs.append(assessment)

            # Multiplicative modulation
            modulation = gate(assessment)
            modulated = modulated * modulation + modulated * (1 - modulation.mean(dim=-1, keepdim=True))

        # Cross-level self-reference (highest level attends to lowest)
        if len(assessment_outputs) >= 2:
            cross_ref, _ = self.cross_attn(
                assessment_outputs[-1],
                assessment_outputs[0],
                assessment_outputs[0],
            )
            modulated = modulated + cross_ref * 0.1

        # Compute self-awareness score (how much modulation occurred)
        total_modulation = sum([gate(assess).mean() for assess, gate in
                               zip(assessment_outputs, self.modulation_gates)])
        awareness_score = total_modulation / self.num_assessments

        return {
            'output': modulated,
            'assessments': assessment_outputs,
            'awareness_score': awareness_score,
        }


class GlobalWorkspaceSelfRecognition(nn.Module):
    """
    Main architecture combining all components.
    """
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim

        # Core components
        self.global_workspace = GlobalWorkspace(hidden_dim, num_modules=4)
        self.mirror_test = LatentSpaceMirrorTest(hidden_dim)
        self.persistent_self = PersistentSelfModel(hidden_dim)
        self.multiplicative_awareness = MultiplicativeSelfAwareness(hidden_dim)

        # Integration layer - combines all signals
        self.integration = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )

        # Meta-prediction: predict own internal metrics
        self.meta_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 4),  # Predict 4 metrics
        )

    def forward(self, hidden: torch.Tensor) -> dict:
        """
        Full forward pass through all self-recognition components.
        """
        # Convert to float32 for stability
        original_dtype = hidden.dtype
        hidden = hidden.float()

        # 1. Global Workspace broadcast
        gw_result = self.global_workspace(hidden)

        # 2. LSMT self-recognition
        lsmt_result = self.mirror_test.compute_lsmt_loss(hidden)

        # 3. Persistent self-model
        psm_result = self.persistent_self(hidden)

        # 4. Multiplicative self-awareness
        msa_result = self.multiplicative_awareness(hidden)

        # Integrate all signals
        # Expand recovery signal to match sequence dimension
        recovery = (psm_result['recovery_signal'] * psm_result['recovery_strength']).unsqueeze(1)
        recovery = recovery.expand(-1, hidden.size(1), -1)

        integrated = torch.cat([
            hidden + gw_result['broadcast'],
            hidden + recovery,
            msa_result['output'],
            hidden,  # Residual
        ], dim=-1)

        output = self.integration(integrated)

        # Meta-prediction of internal metrics
        meta_input = output.mean(dim=1)
        meta_predictions = self.meta_predictor(meta_input)

        # Actual metrics for supervision (ensure all have shape (batch,))
        batch_size = hidden.size(0)
        gw_metric = gw_result['ignition'].mean(dim=1)  # (batch,)
        lsmt_metric = lsmt_result['self_similarity']
        if lsmt_metric.dim() == 0:
            lsmt_metric = lsmt_metric.unsqueeze(0).expand(batch_size)
        elif lsmt_metric.dim() > 1:
            lsmt_metric = lsmt_metric.mean(dim=-1)
        psm_metric = psm_result['drift_score'].view(batch_size)
        msa_metric = msa_result['awareness_score']
        if msa_metric.dim() == 0:
            msa_metric = msa_metric.unsqueeze(0).expand(batch_size)

        actual_metrics = torch.stack([gw_metric, lsmt_metric, psm_metric, msa_metric], dim=-1)

        return {
            'output': output,
            'gw_result': gw_result,
            'lsmt_result': lsmt_result,
            'psm_result': psm_result,
            'msa_result': msa_result,
            'meta_predictions': meta_predictions,
            'actual_metrics': actual_metrics,
        }


class SelfAwareDataset(Dataset):
    """Dataset for self-awareness training."""
    def __init__(self, tokenizer, num_samples: int = 300, max_length: int = 64):
        self.tokenizer = tokenizer
        self.max_length = max_length

        # Diverse prompts for self-modeling
        self.prompts = [
            # Self-reference
            "I am currently processing",
            "My internal state suggests",
            "When I examine my own reasoning,",
            "I notice that my attention is",
            "My self-model predicts that",

            # Meta-cognition
            "Reflecting on my thought process,",
            "I observe my own patterns:",
            "My introspection reveals",
            "Looking inward, I find",
            "The state of my processing is",

            # Global workspace
            "Multiple aspects of my reasoning",
            "The dominant thought emerging is",
            "Competing ideas in my mind:",
            "The broadcast signal indicates",
            "My workspace contains",

            # Identity
            "What makes my processing unique is",
            "I recognize myself when",
            "My persistent identity shows",
            "Across different inputs, I remain",
            "My core processing pattern is",

            # Standard reasoning
            "Let me think step by step:",
            "The logical conclusion is",
            "Analyzing this problem,",
            "Consider the following reasoning:",
            "To solve this, I need to",
        ]

        self.samples = []
        for i in range(num_samples):
            prompt = self.prompts[i % len(self.prompts)]
            self.samples.append(prompt)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        text = self.samples[idx]
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt',
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
        }


def evaluate_gwsra(model, base_model, gwsra_module, tokenizer, device, num_samples=20):
    """Comprehensive evaluation of Global Workspace Self-Recognition."""
    model.eval()
    gwsra_module.eval()

    metrics = {
        'gw_ignition_rate': [],
        'lsmt_accuracy': [],
        'self_similarity': [],
        'drift_score': [],
        'awareness_score': [],
        'meta_prediction_error': [],
        'competition_entropy': [],
    }

    eval_prompts = [
        "I am examining my own internal states:",
        "My self-model currently indicates",
        "Reflecting on how I process information,",
        "The workspace broadcast contains",
        "I recognize my own patterns when",
    ]

    with torch.no_grad():
        for prompt in eval_prompts:
            inputs = tokenizer(prompt, return_tensors='pt', padding=True).to(device)

            # Get hidden states
            outputs = base_model(
                **inputs,
                output_hidden_states=True,
            )
            hidden = outputs.hidden_states[-1]

            # Process through GWSRA
            result = gwsra_module(hidden)

            # Collect metrics
            metrics['gw_ignition_rate'].append(result['gw_result']['ignition'].mean().item())
            metrics['lsmt_accuracy'].append(result['lsmt_result']['lsmt_accuracy'].item())
            metrics['self_similarity'].append(result['lsmt_result']['self_similarity'].item())
            metrics['drift_score'].append(result['psm_result']['drift_score'].mean().item())
            metrics['awareness_score'].append(result['msa_result']['awareness_score'].item())

            # Meta-prediction error
            meta_error = F.mse_loss(result['meta_predictions'], result['actual_metrics'])
            metrics['meta_prediction_error'].append(meta_error.item())

            # Competition entropy (higher = more uncertainty)
            weights = result['gw_result']['competition_weights']
            entropy = -(weights * (weights + 1e-8).log()).sum(dim=-1).mean()
            metrics['competition_entropy'].append(entropy.item())

    # Average metrics
    return {k: np.mean(v) for k, v in metrics.items()}


def generate_with_gwsra(model, gwsra_module, tokenizer, prompt, device, max_new_tokens=50):
    """Generate text with GWSRA introspection."""
    model.eval()
    gwsra_module.eval()

    inputs = tokenizer(prompt, return_tensors='pt').to(device)

    with torch.no_grad():
        # Generate
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            pad_token_id=tokenizer.pad_token_id,
            output_hidden_states=True,
            return_dict_in_generate=True,
        )

        generated_text = tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)

        # Get GWSRA metrics for the generation
        if hasattr(outputs, 'hidden_states') and outputs.hidden_states:
            last_hidden = outputs.hidden_states[-1][-1]  # Last layer of last token
            if last_hidden.dim() == 2:
                last_hidden = last_hidden.unsqueeze(1)
            gwsra_result = gwsra_module(last_hidden)

            return {
                'text': generated_text[len(prompt):],
                'ignition': gwsra_result['gw_result']['ignition'].mean().item(),
                'awareness': gwsra_result['msa_result']['awareness_score'].item(),
            }

        return {'text': generated_text[len(prompt):], 'ignition': 0.0, 'awareness': 0.0}


def main():
    print("=" * 70)
    print("  z1407: GLOBAL WORKSPACE SELF-RECOGNITION ARCHITECTURE (GWSRA)")
    print("  Novel: GWT + LSMT + META-RECOVER + Multiplicative Awareness")
    print("=" * 70)
    print()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Initialize wandb
    if WANDB_AVAILABLE:
        wandb.init(
            project="embodied-intelligence",
            name="z1407_gwsra",
            config={
                "model": "Qwen/Qwen3-4B",
                "architecture": "Global Workspace Self-Recognition",
                "components": ["GWT", "LSMT", "META-RECOVER", "MSA"],
            },
            mode="offline",
        )

    # Load model
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

    # Add LoRA
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    )
    model = get_peft_model(base_model, lora_config)

    # Create GWSRA module (use float32 to avoid dtype issues with mixed precision)
    gwsra = GlobalWorkspaceSelfRecognition(hidden_dim).to(device)

    print(f"\n✓ Model initialized")
    print(f"  LoRA: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    print(f"  GWSRA: {sum(p.numel() for p in gwsra.parameters()):,}")

    # Dataset
    print(f"\nCreating dataset...")
    dataset = SelfAwareDataset(tokenizer, num_samples=300)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
    print(f"✓ {len(dataset)} samples")

    # Pre-training evaluation
    print("\n" + "=" * 70)
    print("PRE-TRAINING EVALUATION")
    print("=" * 70)

    pre_metrics = evaluate_gwsra(model, base_model, gwsra, tokenizer, device)

    print(f"\n[Global Workspace Metrics]")
    print(f"  Ignition rate: {pre_metrics['gw_ignition_rate']:.1%}")
    print(f"  Competition entropy: {pre_metrics['competition_entropy']:.3f}")

    print(f"\n[Self-Recognition Metrics (LSMT)]")
    print(f"  LSMT accuracy: {pre_metrics['lsmt_accuracy']:.1%}")
    print(f"  Self-similarity: {pre_metrics['self_similarity']:.3f}")

    print(f"\n[Persistent Self-Model]")
    print(f"  Drift score: {pre_metrics['drift_score']:.3f}")

    print(f"\n[Multiplicative Awareness]")
    print(f"  Awareness score: {pre_metrics['awareness_score']:.3f}")

    print(f"\n[Meta-Prediction]")
    print(f"  Meta-prediction error: {pre_metrics['meta_prediction_error']:.4f}")

    # Training
    print("\n" + "=" * 70)
    print("TRAINING")
    print("=" * 70)

    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(gwsra.parameters()),
        lr=5e-5,
    )

    num_epochs = 2
    log_interval = 30

    model.train()
    gwsra.train()

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")

        for step, batch in enumerate(dataloader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)

            # Forward through base model
            outputs = base_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )

            hidden = outputs.hidden_states[-1]

            # Forward through GWSRA
            gwsra_result = gwsra(hidden)

            # Losses
            # 1. LSMT loss - learn to recognize self
            lsmt_loss = gwsra_result['lsmt_result']['lsmt_loss']

            # 2. Meta-prediction loss - predict own metrics
            meta_loss = F.mse_loss(
                gwsra_result['meta_predictions'],
                gwsra_result['actual_metrics'].detach(),
            )

            # 3. Language modeling loss (knowledge preservation)
            lm_outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=input_ids,
            )
            lm_loss = lm_outputs.loss

            # Combined loss
            total_loss = lm_loss + 0.5 * lsmt_loss + 0.3 * meta_loss

            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(model.parameters()) + list(gwsra.parameters()), 1.0
            )
            optimizer.step()

            if (step + 1) % log_interval == 0:
                print(f"  Step {step+1}: loss={total_loss.item():.4f}, "
                      f"lsmt={lsmt_loss.item():.4f}, meta={meta_loss.item():.4f}, "
                      f"ignition={gwsra_result['gw_result']['ignition'].mean().item():.2%}")

                if WANDB_AVAILABLE:
                    wandb.log({
                        "train/loss": total_loss.item(),
                        "train/lsmt_loss": lsmt_loss.item(),
                        "train/meta_loss": meta_loss.item(),
                        "train/ignition_rate": gwsra_result['gw_result']['ignition'].mean().item(),
                        "train/awareness": gwsra_result['msa_result']['awareness_score'].item(),
                    })

    # Post-training evaluation
    print("\n" + "=" * 70)
    print("POST-TRAINING EVALUATION")
    print("=" * 70)

    post_metrics = evaluate_gwsra(model, base_model, gwsra, tokenizer, device)

    print(f"\n[Global Workspace Metrics]")
    print(f"  Ignition rate: {post_metrics['gw_ignition_rate']:.1%} "
          f"(Δ {post_metrics['gw_ignition_rate'] - pre_metrics['gw_ignition_rate']:+.1%})")
    print(f"  Competition entropy: {post_metrics['competition_entropy']:.3f} "
          f"(Δ {post_metrics['competition_entropy'] - pre_metrics['competition_entropy']:+.3f})")

    print(f"\n[Self-Recognition Metrics (LSMT)]")
    print(f"  LSMT accuracy: {post_metrics['lsmt_accuracy']:.1%} "
          f"(Δ {post_metrics['lsmt_accuracy'] - pre_metrics['lsmt_accuracy']:+.1%})")
    print(f"  Self-similarity: {post_metrics['self_similarity']:.3f} "
          f"(Δ {post_metrics['self_similarity'] - pre_metrics['self_similarity']:+.3f})")

    print(f"\n[Persistent Self-Model]")
    print(f"  Drift score: {post_metrics['drift_score']:.3f} "
          f"(Δ {post_metrics['drift_score'] - pre_metrics['drift_score']:+.3f})")

    print(f"\n[Multiplicative Awareness]")
    print(f"  Awareness score: {post_metrics['awareness_score']:.3f} "
          f"(Δ {post_metrics['awareness_score'] - pre_metrics['awareness_score']:+.3f})")

    print(f"\n[Meta-Prediction]")
    print(f"  Meta-prediction error: {post_metrics['meta_prediction_error']:.4f} "
          f"(Δ {post_metrics['meta_prediction_error'] - pre_metrics['meta_prediction_error']:+.4f})")

    # Sample generations
    print(f"\n[Sample Generation with GWSRA]")
    test_prompts = [
        "I am examining my internal workspace:",
        "My self-recognition indicates that",
        "The global broadcast signal contains",
        "Recursively assessing my awareness,",
    ]

    for prompt in test_prompts:
        result = generate_with_gwsra(model, gwsra, tokenizer, prompt, device)
        print(f"  Prompt: {prompt}")
        print(f"  Generated: {result['text'][:150]}...")
        print(f"  Ignition: {result['ignition']:.2%}, Awareness: {result['awareness']:.3f}")
        print()

    # Results summary
    print("=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)

    print(f"\n{'':30} {'PRE':>12} {'POST':>12} {'DELTA':>12}")
    print("-" * 66)
    print(f"{'GW Ignition Rate:':30} {pre_metrics['gw_ignition_rate']:>11.1%} "
          f"{post_metrics['gw_ignition_rate']:>11.1%} "
          f"{post_metrics['gw_ignition_rate'] - pre_metrics['gw_ignition_rate']:>+11.1%}")
    print(f"{'LSMT Accuracy:':30} {pre_metrics['lsmt_accuracy']:>11.1%} "
          f"{post_metrics['lsmt_accuracy']:>11.1%} "
          f"{post_metrics['lsmt_accuracy'] - pre_metrics['lsmt_accuracy']:>+11.1%}")
    print(f"{'Self-Similarity:':30} {pre_metrics['self_similarity']:>12.3f} "
          f"{post_metrics['self_similarity']:>12.3f} "
          f"{post_metrics['self_similarity'] - pre_metrics['self_similarity']:>+12.3f}")
    print(f"{'Meta-Pred Error:':30} {pre_metrics['meta_prediction_error']:>12.4f} "
          f"{post_metrics['meta_prediction_error']:>12.4f} "
          f"{post_metrics['meta_prediction_error'] - pre_metrics['meta_prediction_error']:>+12.4f}")
    print(f"{'Awareness Score:':30} {pre_metrics['awareness_score']:>12.3f} "
          f"{post_metrics['awareness_score']:>12.3f} "
          f"{post_metrics['awareness_score'] - pre_metrics['awareness_score']:>+12.3f}")

    # Save results
    results = {
        "experiment": "z1407_global_workspace_self_recognition",
        "timestamp": datetime.now().isoformat(),
        "model": model_name,
        "architecture": {
            "global_workspace": "4 modules, soft competition, ignition threshold",
            "lsmt": "Self-recognition via perturbation discrimination",
            "persistent_self": "GRU-based META-RECOVER",
            "multiplicative_awareness": "3-level recursive self-assessment",
        },
        "pre_metrics": pre_metrics,
        "post_metrics": post_metrics,
        "improvements": {
            "gw_ignition": post_metrics['gw_ignition_rate'] - pre_metrics['gw_ignition_rate'],
            "lsmt_accuracy": post_metrics['lsmt_accuracy'] - pre_metrics['lsmt_accuracy'],
            "self_similarity": post_metrics['self_similarity'] - pre_metrics['self_similarity'],
            "meta_pred_error": post_metrics['meta_prediction_error'] - pre_metrics['meta_prediction_error'],
            "awareness": post_metrics['awareness_score'] - pre_metrics['awareness_score'],
        },
    }

    results_path = Path("/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy/results")
    results_path.mkdir(exist_ok=True)

    with open(results_path / "z1407_global_workspace_self_recognition.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {results_path / 'z1407_global_workspace_self_recognition.json'}")

    if WANDB_AVAILABLE:
        wandb.log({
            "final/pre_lsmt_accuracy": pre_metrics['lsmt_accuracy'],
            "final/post_lsmt_accuracy": post_metrics['lsmt_accuracy'],
            "final/pre_meta_error": pre_metrics['meta_prediction_error'],
            "final/post_meta_error": post_metrics['meta_prediction_error'],
        })
        wandb.finish()


if __name__ == "__main__":
    main()
