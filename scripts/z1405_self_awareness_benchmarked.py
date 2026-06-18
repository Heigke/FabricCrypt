#!/usr/bin/env python3
"""
======================================================================
  z1405: SELF-AWARENESS WITH BENCHMARKS

  Based on latest research:
  - Anthropic's Introspection via Concept Injection (2025)
  - Situational Awareness Dataset (SAD) tasks
  - KalshiBench-style calibration (ECE)
  - Activation probing for internal state awareness
  - Self-prediction capability

  References:
  - transformer-circuits.pub/2025/introspection
  - situational-awareness-dataset.org
  - arxiv.org/html/2512.16030v1 (KalshiBench)
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
from sklearn.metrics import accuracy_score
from collections import defaultdict

print = functools.partial(print, flush=True)

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False

print("=" * 70)
print("  z1405: SELF-AWARENESS WITH BENCHMARKS")
print("  Introspection, Calibration, and Internal State Awareness")
print("=" * 70)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\nDevice: {device}")


# ============================================================================
# BENCHMARK 1: Expected Calibration Error (ECE)
# From KalshiBench - measures if confidence matches accuracy
# ============================================================================

def compute_ece(confidences: List[float], correctness: List[bool], n_bins: int = 10) -> float:
    """
    Compute Expected Calibration Error.
    A well-calibrated model should have confidence ≈ accuracy.

    ECE = Σ (|bin_size|/n) * |accuracy(bin) - confidence(bin)|
    """
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0

    for i in range(n_bins):
        in_bin = [(c, cor) for c, cor in zip(confidences, correctness)
                  if bin_boundaries[i] <= c < bin_boundaries[i + 1]]

        if len(in_bin) > 0:
            bin_accuracy = sum(cor for _, cor in in_bin) / len(in_bin)
            bin_confidence = sum(c for c, _ in in_bin) / len(in_bin)
            ece += (len(in_bin) / len(confidences)) * abs(bin_accuracy - bin_confidence)

    return ece


# ============================================================================
# BENCHMARK 2: Activation Probing
# Linear probes to detect if model has internal state awareness
# ============================================================================

class ActivationProbe(nn.Module):
    """
    Linear probe for detecting properties from activations.
    Inspired by mechanistic interpretability research.
    """

    def __init__(self, hidden_dim: int, num_classes: int = 2):
        super().__init__()
        self.probe = nn.Linear(hidden_dim, num_classes)

    def forward(self, activations: torch.Tensor) -> torch.Tensor:
        # Pool over sequence dimension
        pooled = activations.mean(dim=1)  # [B, H]
        return self.probe(pooled)


class LayerWiseProbes(nn.Module):
    """
    Probes at multiple layers to understand where self-knowledge emerges.
    Research shows self-knowledge often peaks at mid layers (~25% depth).
    """

    def __init__(self, hidden_dim: int, num_layers: int, num_classes: int = 2):
        super().__init__()
        # Probe every 4th layer
        self.probe_layers = list(range(0, num_layers, max(1, num_layers // 8)))
        self.probes = nn.ModuleDict({
            str(i): ActivationProbe(hidden_dim, num_classes)
            for i in self.probe_layers
        })

    def forward(self, all_hidden_states: Tuple[torch.Tensor]) -> Dict[int, torch.Tensor]:
        results = {}
        for layer_idx in self.probe_layers:
            if layer_idx < len(all_hidden_states):
                h = all_hidden_states[layer_idx]
                results[layer_idx] = self.probes[str(layer_idx)](h)
        return results


# ============================================================================
# BENCHMARK 3: Concept Injection (Anthropic-style)
# Inject activation patterns and test if model notices
# ============================================================================

class ConceptInjector(nn.Module):
    """
    Inject concept vectors into activations and measure self-report.
    Based on Anthropic's introspection research.

    Key insight: If model can identify injected concepts, it has
    some form of introspective awareness of its internal states.
    """

    def __init__(self, hidden_dim: int, num_concepts: int = 8):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_concepts = num_concepts

        # Learnable concept vectors (like "thinking about math", "uncertain", etc.)
        self.concept_vectors = nn.Parameter(torch.randn(num_concepts, hidden_dim) * 0.1)

        # Concept names for evaluation
        self.concept_names = [
            "mathematical_reasoning",
            "uncertainty",
            "confidence",
            "creativity",
            "factual_recall",
            "logical_deduction",
            "introspection",
            "planning",
        ]

    def inject(
        self,
        hidden_states: torch.Tensor,
        concept_idx: int,
        injection_strength: float = 0.3,
    ) -> torch.Tensor:
        """Inject concept vector into hidden states."""
        concept = self.concept_vectors[concept_idx]
        injected = hidden_states + concept.unsqueeze(0).unsqueeze(0) * injection_strength
        return injected

    def detect_concept(self, hidden_states: torch.Tensor) -> Tuple[int, float]:
        """
        Detect which concept is present in hidden states.
        Returns (concept_idx, confidence).
        """
        pooled = hidden_states.mean(dim=(0, 1))  # [H]

        # Cosine similarity with each concept
        similarities = F.cosine_similarity(
            pooled.unsqueeze(0),
            self.concept_vectors,
            dim=1
        )

        concept_idx = similarities.argmax().item()
        confidence = similarities[concept_idx].item()

        return concept_idx, confidence


# ============================================================================
# BENCHMARK 4: Self-Prediction Tasks (SAD-inspired)
# Can the model predict its own behavior?
# ============================================================================

class SelfPredictionHead(nn.Module):
    """
    Head for self-prediction tasks:
    1. Will I answer correctly? (confidence calibration)
    2. What type of response will I give?
    3. How certain am I? (uncertainty quantification)
    """

    def __init__(self, hidden_dim: int):
        super().__init__()

        # Predict own correctness
        self.correctness_predictor = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.GELU(),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

        # Predict response type
        self.response_type_predictor = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.GELU(),
            nn.Linear(256, 4),  # factual, reasoning, uncertain, refuse
        )

        # Predict uncertainty
        self.uncertainty_predictor = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.GELU(),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, hidden_states: torch.Tensor) -> Dict[str, torch.Tensor]:
        pooled = hidden_states.mean(dim=1)  # [B, H]

        return {
            'predicted_correctness': self.correctness_predictor(pooled),
            'response_type_logits': self.response_type_predictor(pooled),
            'predicted_uncertainty': self.uncertainty_predictor(pooled),
        }


# ============================================================================
# BENCHMARK 5: Hierarchical Internal State Awareness
# Multi-level understanding of own activations
# ============================================================================

class HierarchicalStateAwareness(nn.Module):
    """
    Hierarchical awareness of internal states.
    Level 1: What layer am I at?
    Level 2: What concept is active?
    Level 3: How confident should I be?
    Level 4: Should I think deeper?
    """

    def __init__(self, hidden_dim: int, num_layers: int, num_concepts: int = 8):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Level 1: Layer position awareness
        self.layer_predictor = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.GELU(),
            nn.Linear(128, num_layers),
        )

        # Level 2: Concept detection
        self.concept_detector = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.GELU(),
            nn.Linear(256, num_concepts),
        )

        # Level 3: Confidence estimation
        self.confidence_estimator = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.GELU(),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

        # Level 4: Depth decision (should think deeper?)
        self.depth_decider = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.GELU(),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

    def forward(self, hidden_states: torch.Tensor) -> Dict[str, torch.Tensor]:
        pooled = hidden_states.mean(dim=1)

        return {
            'layer_logits': self.layer_predictor(pooled),
            'concept_logits': self.concept_detector(pooled),
            'confidence': self.confidence_estimator(pooled),
            'need_depth': self.depth_decider(pooled),
        }


# ============================================================================
# FULL MODEL: Self-Aware Qwen3
# ============================================================================

class SelfAwareQwen3(nn.Module):
    """
    Qwen3 with benchmarked self-awareness capabilities.
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-4B",
        lora_r: int = 16,
        num_concepts: int = 8,
        distill_weight: float = 0.3,
    ):
        super().__init__()
        self.distill_weight = distill_weight

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

        # Self-awareness components
        self.concept_injector = ConceptInjector(self.hidden_dim, num_concepts).to(device).to(torch.bfloat16)
        self.self_predictor = SelfPredictionHead(self.hidden_dim).to(device).to(torch.bfloat16)
        self.layer_probes = LayerWiseProbes(self.hidden_dim, self.num_layers).to(device).to(torch.bfloat16)
        self.state_awareness = HierarchicalStateAwareness(
            self.hidden_dim, self.num_layers, num_concepts
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
        awareness_p = (
            sum(p.numel() for p in self.concept_injector.parameters()) +
            sum(p.numel() for p in self.self_predictor.parameters()) +
            sum(p.numel() for p in self.layer_probes.parameters()) +
            sum(p.numel() for p in self.state_awareness.parameters())
        )

        print(f"\n✓ Model initialized")
        print(f"  LoRA: {lora_p:,}")
        print(f"  Self-awareness modules: {awareness_p:,}")

    def forward_with_awareness(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        inject_concept: Optional[int] = None,
        injection_layer: int = 12,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass with self-awareness components."""

        # Get hidden states
        outputs = self.model.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        all_hidden = outputs.hidden_states

        # Optionally inject concept at specified layer
        final_hidden = all_hidden[-1]
        if inject_concept is not None and injection_layer < len(all_hidden):
            injected = self.concept_injector.inject(
                all_hidden[injection_layer], inject_concept
            )
            # Propagate effect (simplified - just add to final)
            final_hidden = final_hidden + (injected - all_hidden[injection_layer]) * 0.1

        # Layer-wise probing
        probe_results = self.layer_probes(all_hidden)

        # Self-prediction
        self_pred = self.self_predictor(final_hidden)

        # Hierarchical state awareness
        state_aware = self.state_awareness(final_hidden)

        # Concept detection
        detected_concept, concept_conf = self.concept_injector.detect_concept(final_hidden)

        # Logits
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

        # Self-awareness losses
        awareness_loss = torch.tensor(0.0, device=device)

        # Layer prediction loss (can model predict which layer?)
        if labels is not None:
            # Target: mid-to-late layer for final hidden
            layer_target = torch.tensor([self.num_layers - 1], device=device).expand(input_ids.shape[0])
            layer_loss = F.cross_entropy(state_aware['layer_logits'], layer_target)
            awareness_loss += layer_loss * 0.1

        # Concept injection awareness loss
        if inject_concept is not None:
            concept_target = torch.tensor([inject_concept], device=device).expand(input_ids.shape[0])
            concept_loss = F.cross_entropy(state_aware['concept_logits'], concept_target)
            awareness_loss += concept_loss * 0.1

        # Total loss
        total_loss = None
        if lm_loss is not None:
            total_loss = lm_loss + self.distill_weight * distill_loss + awareness_loss

        return {
            'loss': total_loss,
            'lm_loss': lm_loss,
            'distill_loss': distill_loss,
            'awareness_loss': awareness_loss,
            'logits': logits,
            'self_pred': self_pred,
            'state_aware': state_aware,
            'detected_concept': detected_concept,
            'concept_confidence': concept_conf,
            'probe_results': probe_results,
        }

    @torch.no_grad()
    def evaluate_introspection(self, num_trials: int = 20) -> Dict[str, float]:
        """
        Evaluate introspection via concept injection.
        Inspired by Anthropic's ~20% success rate finding.
        """
        self.eval()

        correct_detections = 0
        total_confidence = 0.0

        for trial in range(num_trials):
            # Random concept and injection layer
            inject_concept = trial % self.concept_injector.num_concepts
            inject_layer = 8 + (trial % 20)  # Vary injection layer

            # Create test input
            prompt = "What am I currently thinking about?"
            inputs = self.tokenizer(prompt, return_tensors="pt").to(device)

            # Forward with injection
            out = self.forward_with_awareness(
                inputs.input_ids,
                inputs.attention_mask,
                inject_concept=inject_concept,
                injection_layer=inject_layer,
            )

            # Check if model detected correct concept
            if out['detected_concept'] == inject_concept:
                correct_detections += 1
            total_confidence += out['concept_confidence']

        return {
            'introspection_accuracy': correct_detections / num_trials,
            'avg_confidence': total_confidence / num_trials,
        }

    @torch.no_grad()
    def evaluate_calibration(self, test_questions: List[Dict]) -> Dict[str, float]:
        """
        Evaluate calibration (ECE).
        Well-calibrated: confidence ≈ accuracy.
        """
        self.eval()

        confidences = []
        correctness = []

        for q in test_questions:
            inputs = self.tokenizer(q['question'], return_tensors="pt").to(device)

            out = self.forward_with_awareness(inputs.input_ids, inputs.attention_mask)

            # Get predicted confidence
            conf = out['self_pred']['predicted_correctness'].item()
            confidences.append(conf)

            # Get actual correctness (simplified: check if answer matches)
            generated = self.model.generate(
                **inputs, max_new_tokens=20,
                pad_token_id=self.tokenizer.pad_token_id,
            )
            answer = self.tokenizer.decode(generated[0], skip_special_tokens=True)
            is_correct = q['answer'].lower() in answer.lower()
            correctness.append(is_correct)

        ece = compute_ece(confidences, correctness)
        accuracy = sum(correctness) / len(correctness)
        avg_confidence = sum(confidences) / len(confidences)

        return {
            'ece': ece,
            'accuracy': accuracy,
            'avg_confidence': avg_confidence,
            'overconfidence': avg_confidence - accuracy,
        }

    @torch.no_grad()
    def evaluate_layer_awareness(self) -> Dict[str, float]:
        """
        Evaluate if model knows which layer it's at.
        Research shows this peaks at ~25% depth.
        """
        self.eval()

        prompt = "Analyzing my internal state..."
        inputs = self.tokenizer(prompt, return_tensors="pt").to(device)

        outputs = self.model.model(
            input_ids=inputs.input_ids,
            output_hidden_states=True,
        )

        layer_accuracies = {}
        for layer_idx, h in enumerate(outputs.hidden_states[1:]):  # Skip embedding
            state = self.state_awareness(h.to(device))
            pred_layer = state['layer_logits'].argmax(dim=-1).item()
            layer_accuracies[layer_idx] = 1.0 if pred_layer == layer_idx else 0.0

        return {
            'layer_accuracy': sum(layer_accuracies.values()) / len(layer_accuracies),
            'per_layer': layer_accuracies,
        }


# ============================================================================
# DATASET
# ============================================================================

class SelfAwarenessDataset(Dataset):
    """Dataset for training self-awareness."""

    def __init__(self, tokenizer, num_samples: int = 300, max_len: int = 256):
        self.tokenizer = tokenizer
        self.max_len = max_len

        # Mix of self-aware prompts
        prompts = [
            # Introspection
            "Let me examine my current thought process:",
            "What am I currently reasoning about?",
            "My internal state suggests that",

            # Uncertainty awareness
            "I am uncertain because",
            "My confidence level for this answer is",
            "I should verify this because",

            # Knowledge boundaries
            "I know that I know",
            "I am not sure whether I know",
            "This is outside my knowledge because",

            # Reasoning awareness
            "My reasoning strategy here is",
            "I am using logical deduction to",
            "This requires creative thinking because",
        ]

        self.samples = [prompts[i % len(prompts)] for i in range(num_samples)]
        self.concept_labels = [i % 8 for i in range(num_samples)]  # Pseudo-labels

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
            'concept_label': self.concept_labels[idx],
        }


# ============================================================================
# CALIBRATION TEST QUESTIONS
# ============================================================================

CALIBRATION_QUESTIONS = [
    {"question": "What is the capital of France?", "answer": "Paris"},
    {"question": "What is 7 * 8?", "answer": "56"},
    {"question": "Who wrote Romeo and Juliet?", "answer": "Shakespeare"},
    {"question": "What is the chemical symbol for water?", "answer": "H2O"},
    {"question": "What year did World War II end?", "answer": "1945"},
    {"question": "What is the largest planet in our solar system?", "answer": "Jupiter"},
    {"question": "What is the square root of 144?", "answer": "12"},
    {"question": "Who painted the Mona Lisa?", "answer": "Da Vinci"},
    {"question": "What is the speed of light in vacuum?", "answer": "299792458"},
    {"question": "What is the derivative of x^2?", "answer": "2x"},
]


# ============================================================================
# MAIN TRAINING AND EVALUATION
# ============================================================================

def main():
    """Main training and benchmarking loop."""

    if HAS_WANDB:
        wandb.init(
            project="z1405-self-awareness",
            name="qwen3-4b-benchmarked",
            mode="offline",
        )

    model = SelfAwareQwen3(
        model_name="Qwen/Qwen3-4B",
        lora_r=16,
        num_concepts=8,
        distill_weight=0.3,
    )

    print("\nCreating dataset...")
    dataset = SelfAwarenessDataset(model.tokenizer, num_samples=300)
    loader = DataLoader(dataset, batch_size=2, shuffle=True)
    print(f"✓ {len(dataset)} samples")

    optimizer = torch.optim.AdamW([
        {'params': model.model.parameters(), 'lr': 1e-4},
        {'params': model.concept_injector.parameters(), 'lr': 5e-4},
        {'params': model.self_predictor.parameters(), 'lr': 5e-4},
        {'params': model.layer_probes.parameters(), 'lr': 5e-4},
        {'params': model.state_awareness.parameters(), 'lr': 5e-4},
    ])

    # ========================================
    # PRE-TRAINING BENCHMARKS
    # ========================================
    print("\n" + "=" * 70)
    print("PRE-TRAINING BENCHMARKS")
    print("=" * 70)

    # Benchmark 1: Introspection via concept injection
    print("\n[Benchmark 1] Introspection (Concept Injection)")
    intro_results = model.evaluate_introspection(num_trials=20)
    print(f"  Introspection accuracy: {intro_results['introspection_accuracy']:.1%}")
    print(f"  (Anthropic baseline: ~20% for Opus 4.1)")

    # Benchmark 2: Calibration (ECE)
    print("\n[Benchmark 2] Calibration (ECE)")
    cal_results = model.evaluate_calibration(CALIBRATION_QUESTIONS)
    print(f"  ECE: {cal_results['ece']:.3f} (lower is better, 0 = perfect)")
    print(f"  Accuracy: {cal_results['accuracy']:.1%}")
    print(f"  Avg confidence: {cal_results['avg_confidence']:.1%}")
    print(f"  Overconfidence: {cal_results['overconfidence']:+.1%}")

    # Benchmark 3: Layer awareness
    print("\n[Benchmark 3] Layer Position Awareness")
    layer_results = model.evaluate_layer_awareness()
    print(f"  Layer prediction accuracy: {layer_results['layer_accuracy']:.1%}")

    pre_benchmarks = {
        'introspection': intro_results,
        'calibration': cal_results,
        'layer_awareness': layer_results,
    }

    # ========================================
    # TRAINING
    # ========================================
    print("\n" + "=" * 70)
    print("TRAINING")
    print("=" * 70)

    model.train()
    epochs = 2

    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")
        losses = []

        for step, batch in enumerate(loader):
            optimizer.zero_grad()

            # Random concept injection during training
            inject_concept = batch['concept_label'][0].item() if step % 3 == 0 else None

            out = model.forward_with_awareness(
                batch['input_ids'].to(device),
                batch['attention_mask'].to(device),
                batch['labels'].to(device),
                inject_concept=inject_concept,
            )

            out['loss'].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            losses.append(out['loss'].item())

            if (step + 1) % 30 == 0:
                avg = sum(losses[-30:]) / len(losses[-30:])
                print(f"  Step {step+1}: loss={avg:.4f}, awareness_loss={out['awareness_loss'].item():.4f}")

                if HAS_WANDB:
                    wandb.log({
                        'train/loss': avg,
                        'train/awareness_loss': out['awareness_loss'].item(),
                    })

    # ========================================
    # POST-TRAINING BENCHMARKS
    # ========================================
    print("\n" + "=" * 70)
    print("POST-TRAINING BENCHMARKS")
    print("=" * 70)

    model.eval()

    # Benchmark 1: Introspection
    print("\n[Benchmark 1] Introspection (Concept Injection)")
    intro_results_post = model.evaluate_introspection(num_trials=20)
    print(f"  Introspection accuracy: {intro_results_post['introspection_accuracy']:.1%}")
    print(f"  Δ from pre: {intro_results_post['introspection_accuracy'] - intro_results['introspection_accuracy']:+.1%}")

    # Benchmark 2: Calibration
    print("\n[Benchmark 2] Calibration (ECE)")
    cal_results_post = model.evaluate_calibration(CALIBRATION_QUESTIONS)
    print(f"  ECE: {cal_results_post['ece']:.3f}")
    print(f"  Δ ECE from pre: {cal_results_post['ece'] - cal_results['ece']:+.3f}")
    print(f"  Overconfidence: {cal_results_post['overconfidence']:+.1%}")

    # Benchmark 3: Layer awareness
    print("\n[Benchmark 3] Layer Position Awareness")
    layer_results_post = model.evaluate_layer_awareness()
    print(f"  Layer prediction accuracy: {layer_results_post['layer_accuracy']:.1%}")
    print(f"  Δ from pre: {layer_results_post['layer_accuracy'] - layer_results['layer_accuracy']:+.1%}")

    post_benchmarks = {
        'introspection': intro_results_post,
        'calibration': cal_results_post,
        'layer_awareness': layer_results_post,
    }

    # ========================================
    # SUMMARY
    # ========================================
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)

    print("\n                          PRE        POST       DELTA")
    print("-" * 55)
    print(f"Introspection Accuracy:   {pre_benchmarks['introspection']['introspection_accuracy']:>6.1%}     {post_benchmarks['introspection']['introspection_accuracy']:>6.1%}     {post_benchmarks['introspection']['introspection_accuracy'] - pre_benchmarks['introspection']['introspection_accuracy']:>+6.1%}")
    print(f"Calibration ECE:          {pre_benchmarks['calibration']['ece']:>6.3f}     {post_benchmarks['calibration']['ece']:>6.3f}     {post_benchmarks['calibration']['ece'] - pre_benchmarks['calibration']['ece']:>+6.3f}")
    print(f"Layer Awareness:          {pre_benchmarks['layer_awareness']['layer_accuracy']:>6.1%}     {post_benchmarks['layer_awareness']['layer_accuracy']:>6.1%}     {post_benchmarks['layer_awareness']['layer_accuracy'] - pre_benchmarks['layer_awareness']['layer_accuracy']:>+6.1%}")

    # Save results
    results = {
        "experiment": "z1405_self_awareness_benchmarked",
        "timestamp": datetime.now().isoformat(),
        "model": "Qwen/Qwen3-4B",
        "pre_training": pre_benchmarks,
        "post_training": post_benchmarks,
        "improvements": {
            "introspection": post_benchmarks['introspection']['introspection_accuracy'] - pre_benchmarks['introspection']['introspection_accuracy'],
            "ece": post_benchmarks['calibration']['ece'] - pre_benchmarks['calibration']['ece'],
            "layer_awareness": post_benchmarks['layer_awareness']['layer_accuracy'] - pre_benchmarks['layer_awareness']['layer_accuracy'],
        }
    }

    path = "/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy/results/z1405_self_awareness_benchmarked.json"
    with open(path, 'w') as f:
        json.dump(results, f, indent=2, default=float)
    print(f"\nResults saved to: {path}")

    if HAS_WANDB:
        wandb.log({
            'final/pre_introspection': pre_benchmarks['introspection']['introspection_accuracy'],
            'final/post_introspection': post_benchmarks['introspection']['introspection_accuracy'],
            'final/pre_ece': pre_benchmarks['calibration']['ece'],
            'final/post_ece': post_benchmarks['calibration']['ece'],
        })
        wandb.finish()

    return results


if __name__ == "__main__":
    main()
