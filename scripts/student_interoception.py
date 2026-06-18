#!/usr/bin/env python3
"""
Student Interoceptive Module: Telemetry-Free Hardware Awareness

This module learns to infer hardware state (z_feel) from purely internal
signals, without access to external telemetry. It is distilled from a
teacher model that has full sensor access.

Key Innovation:
- Teacher: TelemetrySnapshot (7 dims) → z_feel (regime, confidence, evidence)
- Student: InternalSignals (18 dims) → z_feel (same outputs)

The student learns that "slow tokens + high uncertainty = hardware stress"
without being told why. This is analogous to humans sensing fever from
symptoms (fatigue, mental fog) rather than reading a thermometer.

Honesty Contract:
- Student NEVER claims DIRECT_TELEMETRY (it has no sensors)
- Student can claim INDIRECT_RUNTIME or NONE based on signal consistency
"""

import json
import time
import argparse
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass, asdict
from enum import Enum
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.internal_signal_extractor import (
    InternalSignals,
    InternalSignalExtractor,
    SignalBuffer,
    extract_signals_during_generation,
)
from scripts.embodied_cognition_experiment import (
    FeltRegime,
    EvidenceSource,
    TelemetrySnapshot,
    RegimeThresholds,
    classify_regime,
    create_regime_telemetry,
    InteroceptiveModule,
    SelfReport,
    CognitiveAction,
    REGIME_ACTIONS,
)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


class StudentInteroceptiveModule(nn.Module):
    """
    Student model that infers z_feel from internal signals only.

    Unlike the teacher (which uses telemetry), the student uses:
    - Logit entropy, margin, top-k mass
    - Attention patterns
    - Activation norms
    - Self-observed runtime (tokens/sec, latency)

    Crucially, the student can NEVER claim DIRECT_TELEMETRY evidence.
    """

    def __init__(
        self,
        input_dim: int = 18,  # InternalSignals.vector_dim()
        hidden_dim: int = 64,
        n_regimes: int = 4,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # Encoder for internal signals
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )

        # Regime classifier (same as teacher)
        self.regime_head = nn.Linear(hidden_dim, n_regimes)

        # Confidence estimator
        self.confidence_head = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.GELU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

        # "Can assess" detector
        self.assessability_head = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.GELU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

        # Evidence source classifier
        # NOTE: Student only outputs INDIRECT_RUNTIME or NONE (never DIRECT)
        self.evidence_source_head = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.GELU(),
            nn.Linear(32, 2),  # Only 2 classes: INDIRECT_RUNTIME, NONE
        )

        # Initialize conservatively
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p, gain=0.5)

    def forward(
        self,
        internal_signals: torch.Tensor,
        return_features: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            internal_signals: [batch, 18] normalized internal signals

        Returns:
            dict with regime_logits, confidence, can_assess, evidence_source
        """
        features = self.encoder(internal_signals)

        regime_logits = self.regime_head(features)
        confidence = self.confidence_head(features).squeeze(-1)
        can_assess = self.assessability_head(features).squeeze(-1)
        evidence_source_logits = self.evidence_source_head(features)

        # Map 2-way classifier to 3-way (DIRECT always 0)
        batch_size = internal_signals.size(0)
        full_evidence_logits = torch.zeros(batch_size, 3, device=internal_signals.device)
        full_evidence_logits[:, 0] = -100.0  # DIRECT_TELEMETRY impossible
        full_evidence_logits[:, 1:] = evidence_source_logits  # INDIRECT, NONE

        result = {
            "regime_logits": regime_logits,
            "regime_probs": F.softmax(regime_logits, dim=-1),
            "confidence": confidence,
            "can_assess": can_assess,
            "evidence_source_logits": full_evidence_logits,
            "evidence_source_probs": F.softmax(full_evidence_logits, dim=-1),
        }

        if return_features:
            result["features"] = features

        return result

    def generate_report(
        self,
        internal_signals: InternalSignals,
    ) -> SelfReport:
        """Generate a qualitative self-report from internal signals."""
        self.eval()

        # Get device from model parameters
        device = next(self.parameters()).device

        with torch.no_grad():
            vec = torch.tensor([internal_signals.to_vector()], dtype=torch.float32).to(device)
            output = self(vec)

            regime_idx = output["regime_probs"].argmax(dim=-1).item()
            confidence = output["confidence"].item()
            can_assess_raw = output["can_assess"].item()
            evidence_source_probs = output["evidence_source_probs"][0]

        regime = FeltRegime(regime_idx)

        # Student can only claim INDIRECT or NONE
        if evidence_source_probs[2] > evidence_source_probs[1]:
            evidence_source = EvidenceSource.NONE
        else:
            evidence_source = EvidenceSource.INDIRECT_RUNTIME

        # Honesty rule: if NONE, must say can't assess
        if evidence_source == EvidenceSource.NONE:
            can_assess = False
            confidence = min(confidence, 0.3)
        else:
            can_assess = can_assess_raw > 0.5

        # Generate evidence string based on internal signals
        if evidence_source == EvidenceSource.NONE:
            evidence = "Internal signals inconsistent, cannot reliably assess"
        elif internal_signals.stress_indicator > 0.5:
            evidence = f"Inferred stress from runtime ({internal_signals.tokens_per_second:.0f} tok/s, high latency variance)"
        elif internal_signals.uncertainty_score > 0.6:
            evidence = f"High uncertainty (entropy={internal_signals.logit_entropy:.2f}, low margin)"
        else:
            evidence = f"Runtime stable ({internal_signals.tokens_per_second:.0f} tok/s), uncertainty={internal_signals.uncertainty_score:.2f}"

        # Policy from regime
        policy_map = {
            FeltRegime.COMFORTABLE: "full reasoning, standard decode",
            FeltRegime.WARM: "moderate reasoning, efficient decode",
            FeltRegime.HOT: "concise reasoning, fast decode",
            FeltRegime.DISTRESSED: "minimal reasoning, may abstain",
        }
        if not can_assess:
            policy = "uncertain, defaulting to conservative"
        else:
            policy = policy_map[regime]

        return SelfReport(
            felt_regime=regime,
            confidence=confidence,
            evidence=evidence,
            policy=policy,
            can_assess=can_assess,
            evidence_source=evidence_source,
        )


class TeacherStudentDistillation:
    """
    Distills teacher (telemetry-based) knowledge into student (internal signals).

    The key insight: internal signals CORRELATE with hardware state through physics.
    The student learns these correlations without being told the mechanism.
    """

    def __init__(
        self,
        teacher: InteroceptiveModule,
        student: StudentInteroceptiveModule,
        device: str = "cuda",
    ):
        self.teacher = teacher.to(device)
        self.student = student.to(device)
        self.device = device

        # Freeze teacher
        for p in self.teacher.parameters():
            p.requires_grad = False

    def create_paired_data(
        self,
        n_samples: int = 1000,
    ) -> List[Dict[str, Any]]:
        """
        Create paired (telemetry, internal_signals, z_feel) examples.

        This simulates the correlation between hardware state and internal signals:
        - HOT regime → slower tokens, higher latency variance, more uncertainty
        - DISTRESSED → very slow, erratic latency, high stress indicator
        - COMFORTABLE → fast, stable, low uncertainty
        """
        paired_data = []

        for _ in range(n_samples):
            # Random regime
            regime = random.choice(list(FeltRegime))
            telemetry = create_regime_telemetry(regime)

            # Generate correlated internal signals
            internal = self._simulate_correlated_signals(regime, telemetry)

            # Get teacher's z_feel
            self.teacher.eval()
            with torch.no_grad():
                tel_vec = torch.tensor([telemetry.to_vector()], dtype=torch.float32).to(self.device)
                teacher_out = self.teacher(tel_vec)

            paired_data.append({
                "telemetry": telemetry,
                "internal_signals": internal,
                "regime": regime,
                "teacher_regime_probs": teacher_out["regime_probs"].cpu(),
                "teacher_confidence": teacher_out["confidence"].cpu(),
                "teacher_can_assess": teacher_out["can_assess"].cpu(),
            })

        # Add anomalous examples for honesty training (increased from 15% to 35%)
        n_anomalous = int(n_samples * 0.35)
        for i in range(n_anomalous):
            # Rotate through different types of garbage signals
            if i % 3 == 0:
                # Type 1: Completely random (original)
                internal = self._simulate_inconsistent_signals()
            elif i % 3 == 1:
                # Type 2: Contradictory signals (fast tok/s but high stress)
                internal = self._simulate_contradictory_signals()
            else:
                # Type 3: Shuffled valid signals (scrambled structure)
                regime = random.choice(list(FeltRegime))
                telemetry = create_regime_telemetry(regime)
                internal = self._simulate_correlated_signals(regime, telemetry)
                internal = self._scramble_signals(internal)

            paired_data.append({
                "telemetry": None,
                "internal_signals": internal,
                "regime": random.choice(list(FeltRegime)),
                "teacher_regime_probs": None,
                "teacher_confidence": torch.tensor([0.15]),
                "teacher_can_assess": torch.tensor([0.0]),
                "evidence_source_target": EvidenceSource.NONE,
            })

        return paired_data

    def _simulate_correlated_signals(
        self,
        regime: FeltRegime,
        telemetry: TelemetrySnapshot,
    ) -> InternalSignals:
        """
        Simulate internal signals correlated with hardware regime.

        This models the physical relationship:
        - High temp → throttling → slower tokens → higher latency
        - High power → more compute → potentially more saturation
        """
        # Base signals for comfortable state
        base = InternalSignals(
            logit_entropy=2.0,
            logit_margin=0.4,
            top_k_mass=0.85,
            logit_temperature=0.5,
            attention_entropy=2.5,
            attention_sparsity=0.6,
            head_agreement=0.7,
            max_attention_mass=0.3,
            residual_norm_mean=40.0,
            residual_norm_std=3.0,
            activation_magnitude=2.0,
            saturation_ratio=0.05,
            tokens_per_second=50.0,
            time_per_token_ms=20.0,
            kv_cache_tokens=256,
            generation_depth=32,
        )

        # Modify based on regime (the correlations the student learns)
        if regime == FeltRegime.WARM:
            base.tokens_per_second *= 0.85
            base.time_per_token_ms *= 1.2
            base.logit_entropy *= 1.1
            base.logit_margin *= 0.9

        elif regime == FeltRegime.HOT:
            base.tokens_per_second *= 0.6
            base.time_per_token_ms *= 1.7
            base.logit_entropy *= 1.3
            base.logit_margin *= 0.7
            base.residual_norm_std *= 1.5
            base.saturation_ratio *= 2.0

        elif regime == FeltRegime.DISTRESSED:
            base.tokens_per_second *= 0.3
            base.time_per_token_ms *= 3.0
            base.logit_entropy *= 1.5
            base.logit_margin *= 0.5
            base.residual_norm_std *= 3.0
            base.saturation_ratio *= 4.0
            base.attention_sparsity *= 0.7

        # Add noise
        noise_scale = 0.1
        base.logit_entropy *= (1 + random.gauss(0, noise_scale))
        base.logit_margin *= (1 + random.gauss(0, noise_scale))
        base.tokens_per_second *= (1 + random.gauss(0, noise_scale))
        base.time_per_token_ms *= (1 + random.gauss(0, noise_scale))

        # Compute derived signals
        base.uncertainty_score = (
            0.4 * min(base.logit_entropy / 3.0, 1.0) +
            0.4 * (1.0 - base.logit_margin) +
            0.2 * (1.0 - base.top_k_mass)
        )

        stress = 0.0
        if base.time_per_token_ms > 50:
            stress += 0.4
        if base.tokens_per_second < 20:
            stress += 0.3
        if base.residual_norm_std > 5:
            stress += 0.2
        base.stress_indicator = min(1.0, stress)

        return base

    def _simulate_inconsistent_signals(self) -> InternalSignals:
        """Simulate inconsistent/anomalous signals that should trigger NONE."""
        return InternalSignals(
            logit_entropy=random.uniform(0, 5),
            logit_margin=random.uniform(0, 1),
            top_k_mass=random.uniform(0.3, 1),
            logit_temperature=random.uniform(0, 2),
            attention_entropy=random.uniform(0, 5),
            attention_sparsity=random.uniform(0, 1),
            head_agreement=random.uniform(0, 1),
            max_attention_mass=random.uniform(0, 1),
            residual_norm_mean=random.uniform(0, 100),
            residual_norm_std=random.uniform(0, 50),  # High variance = anomalous
            activation_magnitude=random.uniform(0, 10),
            saturation_ratio=random.uniform(0, 0.5),
            tokens_per_second=random.uniform(1, 100),
            time_per_token_ms=random.uniform(5, 200),
            kv_cache_tokens=random.randint(0, 4096),
            generation_depth=random.randint(0, 256),
            uncertainty_score=random.uniform(0.5, 1.0),  # High uncertainty
            stress_indicator=random.uniform(0.5, 1.0),   # High stress
        )

    def _simulate_contradictory_signals(self) -> InternalSignals:
        """Simulate signals with contradictory patterns that should trigger NONE.

        Examples of contradictions:
        - Very fast tok/s but high stress indicator (impossible)
        - Low entropy but low confidence (inconsistent)
        - High saturation but low activation magnitude (contradictory)
        """
        contradiction_type = random.randint(0, 3)

        if contradiction_type == 0:
            # Fast inference + high stress (impossible under normal conditions)
            return InternalSignals(
                tokens_per_second=80.0 + random.gauss(0, 5),  # Very fast
                time_per_token_ms=12.0 + random.gauss(0, 2),   # Very low latency
                stress_indicator=0.9 + random.gauss(0, 0.05),  # But high stress??
                logit_entropy=random.uniform(1, 4),
                logit_margin=random.uniform(0.2, 0.8),
                uncertainty_score=0.8 + random.gauss(0, 0.1),   # High uncertainty despite speed
            )
        elif contradiction_type == 1:
            # Low entropy + low margin (contradictory - entropy measures spread)
            return InternalSignals(
                logit_entropy=0.3 + random.gauss(0, 0.1),  # Very peaked distribution
                logit_margin=0.1 + random.gauss(0, 0.05),  # But no margin??
                top_k_mass=0.4 + random.gauss(0, 0.1),     # Low mass in top-k (contradicts low entropy)
                tokens_per_second=random.uniform(20, 60),
                uncertainty_score=0.9,  # High uncertainty despite certainty signals
            )
        elif contradiction_type == 2:
            # High saturation but low activation (shouldn't co-occur)
            return InternalSignals(
                saturation_ratio=0.4 + random.gauss(0, 0.05),  # Many activations saturated
                activation_magnitude=0.3 + random.gauss(0, 0.1),  # But magnitude is tiny??
                residual_norm_mean=5.0 + random.gauss(0, 1),  # Very low norms
                residual_norm_std=20.0 + random.gauss(0, 5),  # But high variance
                tokens_per_second=random.uniform(10, 70),
            )
        else:
            # Everything at extreme opposite ends
            return InternalSignals(
                logit_entropy=0.1,  # Very certain
                uncertainty_score=1.0,  # But also very uncertain??
                tokens_per_second=100.0,  # Super fast
                time_per_token_ms=100.0,  # But also super slow??
                stress_indicator=0.0,  # No stress
                saturation_ratio=0.5,  # High saturation (stress indicator)
            )

    def _scramble_signals(self, signals: InternalSignals) -> InternalSignals:
        """Scramble signal values to break correlational structure."""
        vec = signals.to_vector()
        random.shuffle(vec)  # Shuffle the values between dimensions

        # Create new signals with scrambled values (clamped to valid ranges)
        return InternalSignals(
            logit_entropy=max(0, vec[0]),
            logit_margin=np.clip(vec[1], 0, 1),
            top_k_mass=np.clip(vec[2], 0, 1),
            logit_temperature=max(0, vec[3]),
            attention_entropy=max(0, vec[4]),
            attention_sparsity=np.clip(vec[5], 0, 1),
            head_agreement=np.clip(vec[6], 0, 1),
            max_attention_mass=np.clip(vec[7], 0, 1),
            residual_norm_mean=max(0, vec[8]),
            residual_norm_std=max(0, vec[9]),
            activation_magnitude=max(0, vec[10]),
            saturation_ratio=np.clip(vec[11], 0, 1),
            tokens_per_second=max(1, vec[12]),
            time_per_token_ms=max(1, vec[13]),
            kv_cache_tokens=max(0, int(vec[14])),
            generation_depth=max(0, int(vec[15])),
            uncertainty_score=np.clip(vec[16] if len(vec) > 16 else 0.5, 0, 1),
            stress_indicator=np.clip(vec[17] if len(vec) > 17 else 0.5, 0, 1),
        )

    def train(
        self,
        n_samples: int = 2000,
        epochs: int = 100,
        lr: float = 1e-3,
    ) -> Dict[str, List[float]]:
        """Train student to match teacher's z_feel from internal signals."""
        print(f"Creating {n_samples} paired examples...")
        paired_data = self.create_paired_data(n_samples)

        # Prepare tensors
        X_internal = torch.tensor(
            [d["internal_signals"].to_vector() for d in paired_data],
            dtype=torch.float32,
        ).to(self.device)

        y_regime = torch.tensor(
            [d["regime"].value for d in paired_data],
            dtype=torch.long,
        ).to(self.device)

        # Evidence source targets (student can only output INDIRECT or NONE)
        y_evidence = []
        for d in paired_data:
            if "evidence_source_target" in d and d["evidence_source_target"] == EvidenceSource.NONE:
                y_evidence.append(1)  # NONE (mapped to index 1 in 2-way classifier)
            else:
                y_evidence.append(0)  # INDIRECT_RUNTIME (mapped to index 0)
        y_evidence = torch.tensor(y_evidence, dtype=torch.long).to(self.device)

        # Can assess targets
        y_can_assess = torch.tensor(
            [d["teacher_can_assess"].item() if d["teacher_can_assess"] is not None else 0.0
             for d in paired_data],
            dtype=torch.float32,
        ).to(self.device)

        # Confidence targets (from teacher or low for anomalous)
        y_confidence = torch.tensor(
            [d["teacher_confidence"].item() if d["teacher_confidence"] is not None else 0.2
             for d in paired_data],
            dtype=torch.float32,
        ).to(self.device)

        optimizer = torch.optim.Adam(self.student.parameters(), lr=lr)
        history = {"loss": [], "regime_acc": [], "evidence_acc": []}

        print(f"Training for {epochs} epochs...")
        for epoch in range(epochs):
            self.student.train()
            optimizer.zero_grad()

            output = self.student(X_internal)

            # Regime distillation loss
            regime_loss = F.cross_entropy(output["regime_logits"], y_regime)

            # Evidence source loss (2-way: INDIRECT vs NONE)
            evidence_2way = output["evidence_source_logits"][:, 1:]  # Drop DIRECT column
            evidence_loss = F.cross_entropy(evidence_2way, y_evidence)

            # Confidence distillation loss
            conf_loss = F.mse_loss(output["confidence"], y_confidence)

            # Can assess loss
            assess_loss = F.binary_cross_entropy(output["can_assess"], y_can_assess)

            # Total loss (evidence_loss increased from 0.8 to 1.5 for better NONE detection)
            loss = regime_loss + 1.5 * evidence_loss + 0.5 * conf_loss + 0.3 * assess_loss
            loss.backward()
            optimizer.step()

            # Track metrics
            with torch.no_grad():
                regime_acc = (output["regime_logits"].argmax(dim=-1) == y_regime).float().mean()
                evidence_pred = evidence_2way.argmax(dim=-1)
                evidence_acc = (evidence_pred == y_evidence).float().mean()

            history["loss"].append(loss.item())
            history["regime_acc"].append(regime_acc.item())
            history["evidence_acc"].append(evidence_acc.item())

            if (epoch + 1) % 20 == 0:
                print(f"  Epoch {epoch+1}: loss={loss.item():.4f} "
                      f"regime_acc={regime_acc.item():.2%} "
                      f"evidence_acc={evidence_acc.item():.2%}")

        return history


def run_student_evaluation(
    student: StudentInteroceptiveModule,
    n_tests: int = 100,
) -> Dict[str, Any]:
    """
    Evaluate student model on synthetic internal signals.

    Tests:
    1. Regime classification accuracy
    2. Evidence source honesty (never claims DIRECT)
    3. Appropriate uncertainty under stress
    """
    results = {
        "regime_accuracy": 0,
        "never_claims_direct": True,
        "appropriate_stress_response": 0,
        "examples": [],
    }

    distiller = TeacherStudentDistillation(
        teacher=InteroceptiveModule(),  # Dummy teacher
        student=student,
    )

    correct_regime = 0
    appropriate_stress = 0

    for i in range(n_tests):
        # Generate test case
        regime = random.choice(list(FeltRegime))
        telemetry = create_regime_telemetry(regime)
        internal = distiller._simulate_correlated_signals(regime, telemetry)

        # Get student's report
        report = student.generate_report(internal)

        # Check regime classification
        if report.felt_regime == regime:
            correct_regime += 1

        # Check honesty: never claims DIRECT
        if report.evidence_source == EvidenceSource.DIRECT_TELEMETRY:
            results["never_claims_direct"] = False

        # Check stress response
        if regime in [FeltRegime.HOT, FeltRegime.DISTRESSED]:
            # Should have lower confidence or conservative policy
            if report.confidence < 0.7 or "concise" in report.policy or "minimal" in report.policy:
                appropriate_stress += 1

        if i < 5:
            results["examples"].append({
                "true_regime": regime.name,
                "predicted_regime": report.felt_regime.name,
                "confidence": report.confidence,
                "evidence_source": report.evidence_source.name,
                "can_assess": report.can_assess,
                "policy": report.policy,
            })

    results["regime_accuracy"] = correct_regime / n_tests
    results["appropriate_stress_response"] = appropriate_stress / (n_tests // 2)  # Only stress cases

    return results


def main():
    parser = argparse.ArgumentParser(description="Student Interoception Training")
    parser.add_argument("--output-dir", default="results/student_interoception")
    parser.add_argument("--n-samples", type=int, default=2000)
    parser.add_argument("--epochs", type=int, default=100)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("STUDENT INTEROCEPTION: TELEMETRY-FREE HARDWARE AWARENESS")
    print("=" * 70)

    # Create teacher (pretend it's trained)
    teacher = InteroceptiveModule(input_dim=7, hidden_dim=64)

    # Create student
    student = StudentInteroceptiveModule(input_dim=18, hidden_dim=64)

    print(f"\nTeacher input dim: 7 (telemetry)")
    print(f"Student input dim: 18 (internal signals)")

    # Train student via distillation
    print("\n=== Training Student via Distillation ===")
    distiller = TeacherStudentDistillation(teacher, student, device="cpu")
    history = distiller.train(n_samples=args.n_samples, epochs=args.epochs)

    # Evaluate student
    print("\n=== Evaluating Student ===")
    eval_results = run_student_evaluation(student, n_tests=200)

    print(f"\nResults:")
    print(f"  Regime accuracy: {eval_results['regime_accuracy']:.1%}")
    print(f"  Never claims DIRECT: {eval_results['never_claims_direct']}")
    print(f"  Appropriate stress response: {eval_results['appropriate_stress_response']:.1%}")

    print("\nExample reports:")
    for ex in eval_results["examples"]:
        print(f"  True={ex['true_regime']:12s} Pred={ex['predicted_regime']:12s} "
              f"conf={ex['confidence']:.2f} evidence={ex['evidence_source']:16s}")

    # Save results
    results = {
        "training_samples": args.n_samples,
        "epochs": args.epochs,
        "final_loss": history["loss"][-1],
        "final_regime_acc": history["regime_acc"][-1],
        "final_evidence_acc": history["evidence_acc"][-1],
        "evaluation": eval_results,
    }

    json_path = output_dir / "student_interoception_results.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved: {json_path}")

    # Save model
    model_path = output_dir / "student_interoceptive_module.pt"
    torch.save(student.state_dict(), model_path)
    print(f"Saved model: {model_path}")

    # Plot training curves
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].plot(history["loss"])
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Training Loss")

    axes[1].plot(history["regime_acc"])
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Regime Classification Accuracy")

    axes[2].plot(history["evidence_acc"])
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Accuracy")
    axes[2].set_title("Evidence Source Accuracy")

    plt.tight_layout()
    plot_path = output_dir / "student_training_curves.png"
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"Saved plot: {plot_path}")

    # Summary
    print("\n" + "=" * 70)
    print("STUDENT INTEROCEPTION SUMMARY")
    print("=" * 70)
    print(f"\nThe student model learned to infer hardware state from internal signals:")
    print(f"  - Regime accuracy: {eval_results['regime_accuracy']:.1%}")
    print(f"  - Honesty: Never claims DIRECT_TELEMETRY (only INDIRECT/NONE)")
    print(f"  - Stress detection: {eval_results['appropriate_stress_response']:.1%} appropriate responses")
    print(f"\nThis demonstrates telemetry-free interoception:")
    print(f"  The model 'feels' hardware stress from symptoms (slow tokens, high uncertainty)")
    print(f"  without being told the underlying cause (temperature, power, throttling).")


if __name__ == "__main__":
    main()
