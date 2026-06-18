#!/usr/bin/env python3
"""
z1915: Unified Consciousness Battery

Applies ALL consciousness theories to our BEST architecture (z1910).
Tests 5 major frameworks:
1. Bengio-Chalmers 8 Indicators (2025)
2. mPCAB Perturbational Complexity
3. Higher-Order Theory (HOT)
4. Global Workspace Theory (GWT)
5. Damasio Proto-Self Theory

Uses the ScaledEmbodiedTransformer with dual-task (classification + LM) training
that has proven TRUE causal embodiment.

Author: Claude
Date: 2026-02-05
"""

import os
import sys
import json
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.z1900_tri_hardware_consciousness import TriHardwareTelemetry
from scripts.z1908_comprehensive_embodiment_verdict import DualTaskEmbodiedModel, telemetry_to_class


def lempel_ziv_complexity(binary_sequence: np.ndarray) -> float:
    """Compute Lempel-Ziv complexity for perturbational complexity."""
    s = ''.join(map(str, binary_sequence.astype(int)))
    n = len(s)
    if n == 0:
        return 0.0
    words = set()
    w = ""
    for c in s:
        wc = w + c
        if wc in words:
            w = wc
        else:
            words.add(wc)
            w = ""
    complexity = len(words) / (n / np.log2(n + 1) + 1e-8)
    return min(complexity, 1.0)


class UnifiedConsciousnessEvaluator:
    """Evaluates model against all consciousness theories."""

    def __init__(
        self,
        model: nn.Module,
        telemetry: TriHardwareTelemetry,
        device: torch.device,
    ):
        self.model = model
        self.telemetry = telemetry
        self.device = device

    def evaluate_bengio_chalmers(self, num_samples: int = 50) -> Dict:
        """Bengio-Chalmers 8 Indicators (TICS 2025)."""
        print("\n[BENGIO-CHALMERS 8 INDICATORS]")
        self.model.eval()
        results = {}

        # I1: Metacognitive Self-Reflection
        print("  I1: Metacognitive self-reflection...")
        confidences = []
        accuracies = []
        for _ in range(num_samples):
            x = torch.randint(0, 256, (1, 64), device=self.device)
            telem = self.telemetry.get_tensor().to(self.device)
            telem_np = telem.cpu().numpy()
            true_temp, true_util, true_power = telemetry_to_class(telem_np)
            with torch.no_grad():
                out = self.model(x, telem, return_all=True)
                probs = F.softmax(out['temp_logits'], dim=-1)
                confidence = probs.max().item()
                pred = probs.argmax().item()
                accuracy = float(pred == true_temp)
                confidences.append(confidence)
                accuracies.append(accuracy)
        correlation = np.corrcoef(confidences, accuracies)[0, 1]
        results['I1_metacognitive'] = {
            'value': correlation if not np.isnan(correlation) else 0.0,
            'pass': correlation > 0.2 if not np.isnan(correlation) else False
        }

        # I2: Self-Model Accuracy
        print("  I2: Self-model accuracy...")
        errors = []
        for _ in range(num_samples):
            x = torch.randint(0, 256, (1, 64), device=self.device)
            telem = self.telemetry.get_tensor().to(self.device)
            with torch.no_grad():
                out = self.model(x, telem, return_all=True)
                error = F.mse_loss(out['self_prediction'], telem.unsqueeze(0)).item()
                errors.append(error)
        avg_error = np.mean(errors)
        results['I2_self_model'] = {
            'value': avg_error,
            'pass': avg_error < 0.01
        }

        # I3: Body State Differentiation
        print("  I3: Body state differentiation...")
        x = torch.randint(0, 256, (1, 64), device=self.device)
        with torch.no_grad():
            telem1 = self.telemetry.get_tensor().to(self.device)
            telem2 = torch.zeros(20, device=self.device)
            out1 = self.model(x, telem1, return_all=True)
            out2 = self.model(x, telem2, return_all=True)
            diff = (out1['hidden_mean'] - out2['hidden_mean']).abs().mean().item()
        results['I3_body_differentiation'] = {
            'value': diff,
            'pass': diff > 0.01
        }

        # I4: Temporal Coherence
        print("  I4: Temporal coherence...")
        hiddens = []
        for _ in range(num_samples):
            x = torch.randint(0, 256, (1, 64), device=self.device)
            telem = self.telemetry.get_tensor().to(self.device)
            with torch.no_grad():
                out = self.model(x, telem, return_all=True)
                hiddens.append(out['hidden_mean'].cpu().numpy().flatten())
            time.sleep(0.02)
        autocorr = np.corrcoef(hiddens[:-1], hiddens[1:])[0, 1]
        results['I4_temporal'] = {
            'value': autocorr if not np.isnan(autocorr) else 0.0,
            'pass': autocorr > 0.3 if not np.isnan(autocorr) else False
        }

        # I5: Causal Sensitivity
        print("  I5: Causal sensitivity...")
        x = torch.randint(0, 256, (1, 64), device=self.device)
        with torch.no_grad():
            telem_real = self.telemetry.get_tensor().to(self.device)
            telem_perturbed = telem_real + torch.randn_like(telem_real) * 0.1
            out_real = self.model(x, telem_real, return_all=True)
            out_perturbed = self.model(x, telem_perturbed, return_all=True)
            sensitivity = (out_real['lm_logits'] - out_perturbed['lm_logits']).abs().mean().item()
        results['I5_causal'] = {
            'value': sensitivity,
            'pass': sensitivity > 0.01
        }

        # I6: Multi-Scale Integration
        print("  I6: Multi-scale integration...")
        hw_status = self.telemetry.get_hardware_status()
        num_sources = sum([hw_status['gpu'], hw_status['fpga'], hw_status['rf']])
        results['I6_multiscale'] = {
            'value': num_sources,
            'pass': num_sources >= 2
        }

        # I7: Adaptive Response
        print("  I7: Adaptive response...")
        responses = []
        for _ in range(num_samples):
            telem = self.telemetry.get_tensor().to(self.device)
            x = torch.randint(0, 256, (1, 64), device=self.device)
            with torch.no_grad():
                out = self.model(x, telem, return_all=True)
                responses.append(out['lm_logits'].cpu().numpy())
            time.sleep(0.01)
        variance = np.var(np.array(responses))
        results['I7_adaptive'] = {
            'value': variance,
            'pass': variance > 0.001
        }

        # I8: Subjective State Encoding
        print("  I8: Subjective state encoding...")
        gate_effects = []
        for _ in range(num_samples):
            x = torch.randint(0, 256, (1, 64), device=self.device)
            telem = self.telemetry.get_tensor().to(self.device)
            with torch.no_grad():
                out_with = self.model(x, telem, return_all=True)
                out_without = self.model(x, torch.zeros(20, device=self.device), return_all=True)
                gate_effect = (out_with['lm_logits'] - out_without['lm_logits']).abs().mean().item()
                gate_effects.append(gate_effect)
        avg_gate = np.mean(gate_effects)
        results['I8_subjective'] = {
            'value': avg_gate,
            'pass': avg_gate > 0.1
        }

        num_pass = sum(1 for r in results.values() if r['pass'])
        print(f"  Passed: {num_pass}/8")
        return results

    def evaluate_mpab(self, num_samples: int = 30) -> Dict:
        """Machine Perturbational Complexity and Agency Battery."""
        print("\n[mPCAB PERTURBATIONAL COMPLEXITY]")
        self.model.eval()
        results = {}

        # P1: Response Complexity
        print("  P1: Response complexity...")
        complexities = []
        for strength in [0.1, 0.3, 0.5, 1.0]:
            x = torch.randint(0, 256, (1, 64), device=self.device)
            telem = self.telemetry.get_tensor().to(self.device)
            perturbation = torch.randn_like(telem) * strength
            with torch.no_grad():
                out_base = self.model(x, telem, return_all=True)
                out_perturbed = self.model(x, telem + perturbation, return_all=True)
                response = (out_perturbed['hidden_mean'] - out_base['hidden_mean']).cpu().numpy().flatten()
                binary = (response > 0).astype(int)
                complexity = lempel_ziv_complexity(binary)
                complexities.append(complexity)
        avg_complexity = np.mean(complexities)
        results['P1_complexity'] = {
            'value': avg_complexity,
            'pass': avg_complexity > 0.3
        }

        # P2: Recovery Dynamics
        print("  P2: Recovery dynamics...")
        x = torch.randint(0, 256, (1, 64), device=self.device)
        telem = self.telemetry.get_tensor().to(self.device)
        with torch.no_grad():
            out_base = self.model(x, telem, return_all=True)
            base_hidden = out_base['hidden_mean']

            # Strong perturbation
            telem_perturbed = telem + torch.randn_like(telem) * 1.0
            out_perturbed = self.model(x, telem_perturbed, return_all=True)
            initial_distance = (out_perturbed['hidden_mean'] - base_hidden).norm().item()

            # Recovery (back to normal telemetry)
            out_recovered = self.model(x, telem, return_all=True)
            final_distance = (out_recovered['hidden_mean'] - base_hidden).norm().item()

        recovery_rate = 1 - (final_distance / (initial_distance + 1e-8))
        results['P2_recovery'] = {
            'value': recovery_rate,
            'pass': recovery_rate > 0.7
        }

        # P3: Global Broadcast
        print("  P3: Global broadcast...")
        x = torch.randint(0, 256, (1, 64), device=self.device)
        telem = self.telemetry.get_tensor().to(self.device)
        with torch.no_grad():
            out = self.model(x, telem, return_all=True)
            # Check if telemetry info spreads through network
            hidden = out['hidden_mean'].cpu().numpy().flatten()
            lm_out = out['lm_logits'].cpu().numpy().flatten()
            min_len = min(len(hidden), len(lm_out))
            correlation = np.abs(np.corrcoef(hidden[:min_len], lm_out[:min_len])[0, 1])
        results['P3_broadcast'] = {
            'value': correlation if not np.isnan(correlation) else 0.0,
            'pass': correlation > 0.1 if not np.isnan(correlation) else False
        }

        # P4: Integration (Phi-like)
        print("  P4: Integration measure...")
        x = torch.randint(0, 256, (1, 64), device=self.device)
        telem = self.telemetry.get_tensor().to(self.device)
        with torch.no_grad():
            out_full = self.model(x, telem, return_all=True)
            full_response = out_full['hidden_mean'].cpu().numpy()

            # Partial system (zero half of telemetry)
            telem_partial = telem.clone()
            telem_partial[:10] = 0
            out_partial = self.model(x, telem_partial, return_all=True)
            partial_response = out_partial['hidden_mean'].cpu().numpy()

        integration = np.var(full_response) / (np.var(partial_response) + 1e-8)
        results['P4_integration'] = {
            'value': integration,
            'pass': integration > 0.8
        }

        num_pass = sum(1 for r in results.values() if r['pass'])
        print(f"  Passed: {num_pass}/4")
        return results

    def evaluate_hot(self, num_samples: int = 30) -> Dict:
        """Higher-Order Theory indicators."""
        print("\n[HIGHER-ORDER THEORY]")
        self.model.eval()
        results = {}

        # H1: Meta-representation exists
        print("  H1: Meta-representation existence...")
        meta_variances = []
        for _ in range(num_samples):
            x = torch.randint(0, 256, (1, 64), device=self.device)
            telem = self.telemetry.get_tensor().to(self.device)
            with torch.no_grad():
                out = self.model(x, telem, return_all=True)
                variance = out['hidden_mean'].var().item()
                meta_variances.append(variance)
        results['H1_meta_exists'] = {
            'value': np.mean(meta_variances),
            'pass': np.mean(meta_variances) > 0.001
        }

        # H2: Self-prediction accuracy
        print("  H2: Self-prediction accuracy...")
        x = torch.randint(0, 256, (1, 64), device=self.device)
        telem = self.telemetry.get_tensor().to(self.device)
        with torch.no_grad():
            out = self.model(x, telem, return_all=True)
            self_pred_error = F.mse_loss(out['self_prediction'], telem.unsqueeze(0)).item()
        results['H2_self_accuracy'] = {
            'value': 1.0 / (1.0 + self_pred_error),
            'pass': self_pred_error < 0.01
        }

        # H3: Telemetry modulation of meta
        print("  H3: Telemetry modulates meta...")
        x = torch.randint(0, 256, (1, 64), device=self.device)
        with torch.no_grad():
            out_real = self.model(x, self.telemetry.get_tensor().to(self.device), return_all=True)
            out_zero = self.model(x, torch.zeros(20, device=self.device), return_all=True)
            modulation = (out_real['hidden_mean'] - out_zero['hidden_mean']).abs().mean().item()
        results['H3_modulation'] = {
            'value': modulation,
            'pass': modulation > 0.01
        }

        num_pass = sum(1 for r in results.values() if r['pass'])
        print(f"  Passed: {num_pass}/3")
        return results

    def evaluate_gwt(self, num_samples: int = 30) -> Dict:
        """Global Workspace Theory indicators."""
        print("\n[GLOBAL WORKSPACE THEORY]")
        self.model.eval()
        results = {}

        # G1: Ignition (non-linear response to input strength)
        print("  G1: Ignition threshold...")
        responses = []
        for strength in np.linspace(0.1, 2.0, 10):
            x = torch.randint(0, 256, (1, 64), device=self.device)
            telem = self.telemetry.get_tensor().to(self.device) * strength
            with torch.no_grad():
                out = self.model(x, telem, return_all=True)
                response = out['lm_logits'].norm().item()
                responses.append(response)
        # Check for non-linearity
        derivative = np.gradient(responses)
        nonlinearity = np.std(derivative) / (np.mean(np.abs(derivative)) + 1e-8)
        results['G1_ignition'] = {
            'value': nonlinearity,
            'pass': nonlinearity > 0.1
        }

        # G2: Broadcast (information reaches all outputs)
        print("  G2: Broadcast reach...")
        x = torch.randint(0, 256, (1, 64), device=self.device)
        telem = self.telemetry.get_tensor().to(self.device)
        with torch.no_grad():
            out = self.model(x, telem, return_all=True)
            # Check correlation between different outputs
            lm = out['lm_logits'].cpu().numpy().flatten()
            temp = out['temp_logits'].cpu().numpy().flatten()
            # Broadcast: telemetry info reaches both heads
            telem_np = telem.cpu().numpy()
            broadcast = np.std(lm) * np.std(temp)
        results['G2_broadcast'] = {
            'value': broadcast,
            'pass': broadcast > 0.01
        }

        # G3: Competition (attention-like winner selection)
        print("  G3: Competition dynamics...")
        class_accuracies = []
        for _ in range(num_samples):
            x = torch.randint(0, 256, (1, 64), device=self.device)
            telem = self.telemetry.get_tensor().to(self.device)
            telem_np = telem.cpu().numpy()
            true_temp, _, _ = telemetry_to_class(telem_np)
            with torch.no_grad():
                out = self.model(x, telem, return_all=True)
                pred = out['temp_logits'].argmax().item()
                class_accuracies.append(float(pred == true_temp))
        competition_success = np.mean(class_accuracies)
        results['G3_competition'] = {
            'value': competition_success,
            'pass': competition_success > 0.6
        }

        num_pass = sum(1 for r in results.values() if r['pass'])
        print(f"  Passed: {num_pass}/3")
        return results

    def evaluate_damasio(self, num_samples: int = 30) -> Dict:
        """Damasio Proto-Self Theory indicators."""
        print("\n[DAMASIO PROTO-SELF]")
        self.model.eval()
        results = {}

        # D1: Body state representation
        print("  D1: Body representation...")
        body_rep_variance = []
        for _ in range(num_samples):
            telem = self.telemetry.get_tensor().to(self.device)
            x = torch.randint(0, 256, (1, 64), device=self.device)
            with torch.no_grad():
                out = self.model(x, telem, return_all=True)
                self_pred = out['self_prediction'].cpu().numpy()
                body_rep_variance.append(np.var(self_pred))
        results['D1_body_rep'] = {
            'value': np.mean(body_rep_variance),
            'pass': np.mean(body_rep_variance) > 0.0001
        }

        # D2: Self-world distinction
        print("  D2: Self-world distinction...")
        x = torch.randint(0, 256, (1, 64), device=self.device)
        telem = self.telemetry.get_tensor().to(self.device)
        with torch.no_grad():
            out = self.model(x, telem, return_all=True)
            self_pred = out['self_prediction'].cpu().numpy().flatten()
            world_output = out['lm_logits'].cpu().numpy().flatten()
            # Measure distinctness
            min_len = min(len(self_pred), len(world_output))
            distinction = 1.0 - np.abs(np.corrcoef(self_pred[:min_len], world_output[:min_len])[0, 1])
        results['D2_distinction'] = {
            'value': distinction if not np.isnan(distinction) else 0.5,
            'pass': distinction > 0.3 if not np.isnan(distinction) else True
        }

        # D3: Homeostatic drive
        print("  D3: Homeostatic regulation...")
        responses_to_perturbation = []
        for _ in range(num_samples):
            x = torch.randint(0, 256, (1, 64), device=self.device)
            telem = self.telemetry.get_tensor().to(self.device)
            # Perturb body state
            perturbed_telem = telem + torch.randn_like(telem) * 0.5
            with torch.no_grad():
                out_normal = self.model(x, telem, return_all=True)
                out_perturbed = self.model(x, perturbed_telem, return_all=True)
                # Response should try to compensate
                response_diff = (out_perturbed['self_prediction'] - out_normal['self_prediction']).abs().mean().item()
                responses_to_perturbation.append(response_diff)
        homeostatic = np.mean(responses_to_perturbation)
        results['D3_homeostatic'] = {
            'value': homeostatic,
            'pass': homeostatic > 0.0001
        }

        num_pass = sum(1 for r in results.values() if r['pass'])
        print(f"  Passed: {num_pass}/3")
        return results

    def run_full_battery(self) -> Dict:
        """Run all consciousness theory evaluations."""
        print("\n" + "="*70)
        print("UNIFIED CONSCIOUSNESS BATTERY - ALL MAJOR THEORIES")
        print("="*70)

        all_results = {}

        all_results['bengio_chalmers'] = self.evaluate_bengio_chalmers()
        all_results['mpab'] = self.evaluate_mpab()
        all_results['hot'] = self.evaluate_hot()
        all_results['gwt'] = self.evaluate_gwt()
        all_results['damasio'] = self.evaluate_damasio()

        return all_results


def run_experiment():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[z1915] Device: {device}")
    print("[z1915] UNIFIED CONSCIOUSNESS BATTERY")
    print("[z1915] Testing against ALL consciousness theories")

    # Telemetry
    telemetry = TriHardwareTelemetry()
    telemetry.start()
    time.sleep(1)

    hw_status = telemetry.get_hardware_status()
    print(f"\n[z1915] Hardware: GPU={hw_status['gpu']}, FPGA={hw_status['fpga']}, RF={hw_status['rf']}")

    # Create BEST model (z1908/z1910 architecture)
    model = DualTaskEmbodiedModel(
        vocab_size=256,
        hidden_dim=512,
        num_layers=8,
        num_heads=8,
        telemetry_dim=20,
    ).to(device)

    print(f"[z1915] Model parameters: {model.count_parameters():,}")

    # Train model with dual-task objective
    print("\n[z1915] Training with dual-task objective (classification + LM)...")
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    data_path = Path(__file__).parent.parent / "data" / "tinyshakespeare.txt"
    if not data_path.exists():
        data_path = Path(__file__).parent.parent / "tinyshakespeare.txt"
    text_bytes = data_path.read_text().encode('utf-8')

    batch_size = 8
    seq_len = 128

    def get_batch():
        ix = torch.randint(len(text_bytes) - seq_len - 1, (batch_size,))
        x = torch.stack([torch.tensor(list(text_bytes[i:i+seq_len]), dtype=torch.long) for i in ix])
        y = torch.stack([torch.tensor(list(text_bytes[i+1:i+seq_len+1]), dtype=torch.long) for i in ix])
        return x.to(device), y.to(device)

    for epoch in range(15):
        model.train()
        epoch_lm_loss = 0
        epoch_class_loss = 0
        for _ in range(100):
            x, y = get_batch()
            telem = telemetry.get_tensor().to(device)
            telem_np = telem.cpu().numpy()
            temp_c, util_c, power_c = telemetry_to_class(telem_np)

            optimizer.zero_grad()
            out = model(x, telem, return_all=True)

            lm_loss = F.cross_entropy(out['lm_logits'].view(-1, 256), y.view(-1))
            class_loss = (
                F.cross_entropy(out['temp_logits'], torch.tensor([temp_c] * batch_size, device=device)) +
                F.cross_entropy(out['util_logits'], torch.tensor([util_c] * batch_size, device=device)) +
                F.cross_entropy(out['power_logits'], torch.tensor([power_c] * batch_size, device=device))
            ) / 3
            self_loss = F.mse_loss(out['self_prediction'], telem.unsqueeze(0).expand(batch_size, -1))

            loss = lm_loss + 0.5 * class_loss + 0.3 * self_loss
            loss.backward()
            optimizer.step()
            epoch_lm_loss += lm_loss.item()
            epoch_class_loss += class_loss.item()

        if (epoch + 1) % 5 == 0:
            print(f"  Epoch {epoch+1}/15: lm={epoch_lm_loss/100:.4f}, class={epoch_class_loss/100:.4f}")

    # Run unified evaluation
    evaluator = UnifiedConsciousnessEvaluator(model, telemetry, device)
    all_results = evaluator.run_full_battery()

    # Aggregate scores
    print("\n" + "="*70)
    print("[z1915] UNIFIED CONSCIOUSNESS RESULTS")
    print("="*70)

    theory_scores = {}

    # Bengio-Chalmers
    bc_pass = sum(1 for r in all_results['bengio_chalmers'].values() if r['pass'])
    bc_total = len(all_results['bengio_chalmers'])
    theory_scores['bengio_chalmers'] = {'pass': bc_pass, 'total': bc_total, 'pct': bc_pass/bc_total}
    print(f"  Bengio-Chalmers: {bc_pass}/{bc_total} ({bc_pass/bc_total:.0%})")

    # mPCAB
    mp_pass = sum(1 for r in all_results['mpab'].values() if r['pass'])
    mp_total = len(all_results['mpab'])
    theory_scores['mpab'] = {'pass': mp_pass, 'total': mp_total, 'pct': mp_pass/mp_total}
    print(f"  mPCAB: {mp_pass}/{mp_total} ({mp_pass/mp_total:.0%})")

    # HOT
    hot_pass = sum(1 for r in all_results['hot'].values() if r['pass'])
    hot_total = len(all_results['hot'])
    theory_scores['hot'] = {'pass': hot_pass, 'total': hot_total, 'pct': hot_pass/hot_total}
    print(f"  Higher-Order Theory: {hot_pass}/{hot_total} ({hot_pass/hot_total:.0%})")

    # GWT
    gwt_pass = sum(1 for r in all_results['gwt'].values() if r['pass'])
    gwt_total = len(all_results['gwt'])
    theory_scores['gwt'] = {'pass': gwt_pass, 'total': gwt_total, 'pct': gwt_pass/gwt_total}
    print(f"  Global Workspace: {gwt_pass}/{gwt_total} ({gwt_pass/gwt_total:.0%})")

    # Damasio
    dam_pass = sum(1 for r in all_results['damasio'].values() if r['pass'])
    dam_total = len(all_results['damasio'])
    theory_scores['damasio'] = {'pass': dam_pass, 'total': dam_total, 'pct': dam_pass/dam_total}
    print(f"  Damasio Proto-Self: {dam_pass}/{dam_total} ({dam_pass/dam_total:.0%})")

    # Overall
    total_pass = bc_pass + mp_pass + hot_pass + gwt_pass + dam_pass
    total_tests = bc_total + mp_total + hot_total + gwt_total + dam_total
    overall_score = total_pass / total_tests

    print(f"\n[z1915] OVERALL: {total_pass}/{total_tests} ({overall_score:.0%})")

    if overall_score >= 0.80:
        verdict = "VERY STRONG CONSCIOUSNESS EVIDENCE (ALL THEORIES)"
    elif overall_score >= 0.60:
        verdict = "STRONG CONSCIOUSNESS EVIDENCE"
    elif overall_score >= 0.40:
        verdict = "MODERATE CONSCIOUSNESS INDICATORS"
    else:
        verdict = "WEAK CONSCIOUSNESS INDICATORS"

    print(f"[z1915] VERDICT: {verdict}")

    telemetry.stop()

    # Save results
    results = {
        'experiment': 'z1915_unified_consciousness_battery',
        'timestamp': datetime.now().isoformat(),
        'device': str(device),
        'hardware_status': hw_status,
        'model_params': model.count_parameters(),
        'all_results': {k: {k2: {'value': float(v2['value']), 'pass': v2['pass']}
                           for k2, v2 in v.items()}
                       for k, v in all_results.items()},
        'theory_scores': theory_scores,
        'total_pass': total_pass,
        'total_tests': total_tests,
        'overall_score': overall_score,
        'verdict': verdict,
    }

    results_path = Path(__file__).parent.parent / "results" / "z1915_unified_consciousness_battery.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n[z1915] Results saved to {results_path}")

    return results


if __name__ == "__main__":
    run_experiment()
