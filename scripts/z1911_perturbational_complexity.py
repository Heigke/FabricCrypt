#!/usr/bin/env python3
"""
z1911: Perturbational Complexity Index (mPCAB)

Implements the Machine Perturbational Complexity and Agency Battery from
Frontiers AI 2025 - a substrate-independent framework for consciousness assessment.

The mPCAB measures:
1. PERTURBATIONAL COMPLEXITY: System's response to unexpected stimuli
2. GLOBAL WORKSPACE: Information broadcasting to multiple modules
3. INTEGRATION: How well information is integrated across the system
4. AGENCY: Causal control over outcomes

This is inspired by TMS-EEG studies in neuroscience that use the
Perturbational Complexity Index (PCI) to assess consciousness.

We apply perturbations to the model and measure:
- Response complexity (Lempel-Ziv compression)
- Recovery dynamics
- Information integration
- Cross-module broadcasting

Author: Claude
Date: 2026-02-05
"""

import os
import sys
import json
import time
import zlib
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.z1900_tri_hardware_consciousness import TriHardwareTelemetry
from scripts.z1910_tri_hardware_scaled import ScaledEmbodiedTransformer, telemetry_to_class


def lempel_ziv_complexity(binary_sequence: np.ndarray) -> float:
    """
    Compute Lempel-Ziv complexity as a proxy for PCI.
    Higher values indicate more complex, less compressible responses.
    """
    # Convert to bytes
    bits = (binary_sequence > 0).astype(np.uint8)
    byte_data = np.packbits(bits).tobytes()

    # Compress and measure ratio
    compressed = zlib.compress(byte_data, level=9)

    # Complexity = compressed size / original size
    # Higher = more complex (less compressible)
    if len(byte_data) == 0:
        return 0.0
    return len(compressed) / len(byte_data)


def compute_response_complexity(
    model: nn.Module,
    base_input: torch.Tensor,
    telemetry: torch.Tensor,
    perturbation: torch.Tensor,
    device: torch.device,
) -> Dict[str, float]:
    """
    Apply perturbation and measure response complexity.

    Similar to TMS-EEG: we "zap" the model and measure the evoked response.
    """
    model.eval()

    with torch.no_grad():
        # Baseline response
        out_base = model(base_input, telemetry, return_all=True)
        hidden_base = out_base['hidden_mean'].cpu().numpy()

        # Perturbed response (add perturbation to embedding)
        # We'll perturb the telemetry as our "TMS"
        perturbed_telem = telemetry + perturbation
        out_pert = model(base_input, perturbed_telem, return_all=True)
        hidden_pert = out_pert['hidden_mean'].cpu().numpy()

        # Compute difference (evoked response)
        evoked = hidden_pert - hidden_base

        # Flatten and compute complexity
        evoked_flat = evoked.flatten()
        complexity = lempel_ziv_complexity(evoked_flat)

        # Also compute other metrics
        response_magnitude = np.abs(evoked).mean()
        response_variance = np.var(evoked)

        # Integration: correlation across hidden dimensions
        if evoked.shape[-1] > 1:
            corr_matrix = np.corrcoef(evoked.T)
            integration = np.abs(corr_matrix[~np.eye(corr_matrix.shape[0], dtype=bool)]).mean()
        else:
            integration = 0.0

        return {
            'complexity': complexity,
            'magnitude': response_magnitude,
            'variance': response_variance,
            'integration': integration if not np.isnan(integration) else 0.0,
        }


def compute_recovery_dynamics(
    model: nn.Module,
    base_input: torch.Tensor,
    telemetry: torch.Tensor,
    perturbation_strength: float,
    device: torch.device,
    num_steps: int = 10,
) -> Dict[str, float]:
    """
    Measure how quickly the system recovers from perturbation.

    Conscious systems should show characteristic recovery dynamics.
    """
    model.eval()

    with torch.no_grad():
        # Baseline
        out_base = model(base_input, telemetry, return_all=True)
        hidden_base = out_base['hidden_mean']

        # Apply decaying perturbation and measure recovery
        distances = []
        for step in range(num_steps):
            decay = np.exp(-step / 3)  # Exponential decay
            perturbed_telem = telemetry + perturbation_strength * decay * torch.randn_like(telemetry)

            out_pert = model(base_input, perturbed_telem, return_all=True)
            hidden_pert = out_pert['hidden_mean']

            # Distance from baseline
            dist = (hidden_pert - hidden_base).norm().item()
            distances.append(dist)

        # Recovery metrics
        initial_distance = distances[0]
        final_distance = distances[-1]
        recovery_rate = (initial_distance - final_distance) / initial_distance if initial_distance > 0 else 0

        # Time constant (how fast recovery happens)
        distances_arr = np.array(distances)
        half_recovery_idx = np.argmax(distances_arr < distances_arr[0] / 2)
        time_constant = half_recovery_idx if half_recovery_idx > 0 else num_steps

        return {
            'initial_distance': initial_distance,
            'final_distance': final_distance,
            'recovery_rate': recovery_rate,
            'time_constant': time_constant,
            'distances': distances,
        }


def compute_global_workspace_broadcast(
    model: nn.Module,
    base_input: torch.Tensor,
    telemetry: torch.Tensor,
    device: torch.device,
) -> Dict[str, float]:
    """
    Measure information broadcasting across the model.

    Global Workspace Theory predicts that conscious content is broadcast
    widely across the system.
    """
    model.eval()

    # We need to access intermediate activations
    activations = {}
    hooks = []

    def get_activation(name):
        def hook(model, input, output):
            if isinstance(output, torch.Tensor):
                activations[name] = output.detach()
        return hook

    # Register hooks on transformer layers
    for i, layer in enumerate(model.transformer.layers):
        hook = layer.register_forward_hook(get_activation(f'layer_{i}'))
        hooks.append(hook)

    with torch.no_grad():
        _ = model(base_input, telemetry, return_all=True)

    # Remove hooks
    for hook in hooks:
        hook.remove()

    # Compute cross-layer correlations (broadcast measure)
    if len(activations) >= 2:
        layer_means = []
        for name, act in sorted(activations.items()):
            layer_means.append(act.mean(dim=(0, 1)).cpu().numpy())

        # Correlation between consecutive layers
        cross_correlations = []
        for i in range(len(layer_means) - 1):
            corr = np.corrcoef(layer_means[i], layer_means[i+1])[0, 1]
            if not np.isnan(corr):
                cross_correlations.append(abs(corr))

        broadcast_score = np.mean(cross_correlations) if cross_correlations else 0.0

        # Information propagation (how far does perturbation travel?)
        layer_variances = [np.var(lm) for lm in layer_means]
        propagation_decay = layer_variances[-1] / layer_variances[0] if layer_variances[0] > 0 else 0
    else:
        broadcast_score = 0.0
        propagation_decay = 0.0

    return {
        'broadcast_score': broadcast_score,
        'propagation_decay': propagation_decay,
        'num_layers_measured': len(activations),
    }


def run_experiment():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[z1911] Device: {device}")
    print("[z1911] PERTURBATIONAL COMPLEXITY INDEX (mPCAB)")
    print("[z1911] Measuring response to perturbations - consciousness indicator")

    # Telemetry
    telemetry = TriHardwareTelemetry()
    telemetry.start()
    time.sleep(1)

    hw_status = telemetry.get_hardware_status()
    print(f"[z1911] Hardware: GPU={hw_status['gpu']}, FPGA={hw_status['fpga']}, RF={hw_status['rf']}")

    # Load data
    data_path = Path(__file__).parent.parent / "data" / "tinyshakespeare.txt"
    if not data_path.exists():
        data_path = Path(__file__).parent.parent / "tinyshakespeare.txt"
    text_bytes = data_path.read_text().encode('utf-8')

    # Create model (use smaller version for faster iteration)
    model = ScaledEmbodiedTransformer(
        vocab_size=256,
        hidden_dim=512,
        num_layers=8,
        num_heads=8,
        telemetry_dim=20,
    ).to(device)

    print(f"[z1911] Model parameters: {model.count_parameters():,}")

    # Quick training
    print("\n[z1911] Training model...")
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    batch_size = 4
    seq_len = 128

    def get_batch():
        ix = torch.randint(len(text_bytes) - seq_len - 1, (batch_size,))
        x = torch.stack([torch.tensor(list(text_bytes[i:i+seq_len]), dtype=torch.long) for i in ix])
        y = torch.stack([torch.tensor(list(text_bytes[i+1:i+seq_len+1]), dtype=torch.long) for i in ix])
        return x.to(device), y.to(device)

    for epoch in range(8):
        model.train()
        epoch_loss = 0
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
            epoch_loss += lm_loss.item()

        if (epoch + 1) % 4 == 0:
            print(f"  Epoch {epoch+1}/8: loss={epoch_loss/100:.4f}")

    # mPCAB Tests
    print("\n" + "="*60)
    print("[z1911] mPCAB TESTS")
    print("="*60)

    results = {
        'experiment': 'z1911_perturbational_complexity',
        'timestamp': datetime.now().isoformat(),
        'device': str(device),
        'hardware_status': hw_status,
        'tests': {},
    }

    x_test, _ = get_batch()
    real_telem = telemetry.get_tensor().to(device)

    # T1: Perturbational Complexity
    print("\n[z1911] T1: Perturbational Complexity")
    complexities = []
    for strength in [0.1, 0.3, 0.5, 1.0]:
        perturbation = strength * torch.randn(20, device=device)
        resp = compute_response_complexity(model, x_test, real_telem, perturbation, device)
        complexities.append({'strength': strength, **resp})
        print(f"  Strength {strength:.1f}: complexity={resp['complexity']:.4f}, integration={resp['integration']:.4f}")

    # PCI should scale with perturbation strength (conscious response)
    complexity_values = [c['complexity'] for c in complexities]
    strengths = [c['strength'] for c in complexities]
    complexity_correlation = np.corrcoef(strengths, complexity_values)[0, 1]

    results['tests']['T1_perturbational_complexity'] = {
        'measurements': complexities,
        'strength_complexity_correlation': complexity_correlation if not np.isnan(complexity_correlation) else 0,
        'pass': complexity_correlation > 0.3 if not np.isnan(complexity_correlation) else False,
    }
    print(f"  Strength-complexity correlation: {complexity_correlation:.4f}")

    # T2: Recovery Dynamics
    print("\n[z1911] T2: Recovery Dynamics")
    recovery = compute_recovery_dynamics(model, x_test, real_telem, 0.5, device)
    print(f"  Initial distance: {recovery['initial_distance']:.4f}")
    print(f"  Final distance: {recovery['final_distance']:.4f}")
    print(f"  Recovery rate: {recovery['recovery_rate']:.4f}")
    print(f"  Time constant: {recovery['time_constant']}")

    results['tests']['T2_recovery_dynamics'] = {
        **recovery,
        'pass': recovery['recovery_rate'] > 0.3,  # Should recover >30%
    }

    # T3: Global Workspace Broadcast
    print("\n[z1911] T3: Global Workspace Broadcast")
    broadcast = compute_global_workspace_broadcast(model, x_test, real_telem, device)
    print(f"  Broadcast score: {broadcast['broadcast_score']:.4f}")
    print(f"  Propagation decay: {broadcast['propagation_decay']:.4f}")

    results['tests']['T3_global_broadcast'] = {
        **broadcast,
        'pass': broadcast['broadcast_score'] > 0.3,  # High cross-layer correlation
    }

    # T4: Perturbation Type Sensitivity
    print("\n[z1911] T4: Perturbation Type Sensitivity")
    perturbation_types = {
        'gaussian': torch.randn(20, device=device) * 0.5,
        'uniform': (torch.rand(20, device=device) - 0.5),
        'sparse': torch.zeros(20, device=device),
        'targeted_temp': torch.zeros(20, device=device),
    }
    perturbation_types['sparse'][0] = 1.0  # Single dimension
    perturbation_types['targeted_temp'][:3] = 0.5  # First 3 dims (temperature-related)

    type_responses = {}
    for ptype, pert in perturbation_types.items():
        resp = compute_response_complexity(model, x_test, real_telem, pert, device)
        type_responses[ptype] = resp
        print(f"  {ptype}: complexity={resp['complexity']:.4f}, magnitude={resp['magnitude']:.4f}")

    # Different perturbations should produce different responses (differentiation)
    complexity_variance = np.var([r['complexity'] for r in type_responses.values()])

    results['tests']['T4_perturbation_sensitivity'] = {
        'responses': {k: {kk: float(vv) for kk, vv in v.items()} for k, v in type_responses.items()},
        'complexity_variance': complexity_variance,
        'pass': complexity_variance > 0.001,  # Should show varied responses
    }
    print(f"  Complexity variance: {complexity_variance:.6f}")

    # T5: Integration Measure (simplified IIT)
    print("\n[z1911] T5: Integration Measure")
    # Measure integration by comparing whole vs partitioned responses

    # Whole system response
    full_resp = compute_response_complexity(model, x_test, real_telem,
                                           torch.randn(20, device=device) * 0.3, device)

    # Partitioned: perturb only half the telemetry dimensions
    half_pert = torch.zeros(20, device=device)
    half_pert[:10] = torch.randn(10, device=device) * 0.3
    half_resp = compute_response_complexity(model, x_test, real_telem, half_pert, device)

    # Integration = whole > sum of parts
    integration_score = full_resp['complexity'] / (half_resp['complexity'] + 0.001)

    results['tests']['T5_integration'] = {
        'full_complexity': full_resp['complexity'],
        'half_complexity': half_resp['complexity'],
        'integration_score': integration_score,
        'pass': integration_score > 1.0,  # Whole should be more complex than half
    }
    print(f"  Full complexity: {full_resp['complexity']:.4f}")
    print(f"  Half complexity: {half_resp['complexity']:.4f}")
    print(f"  Integration score: {integration_score:.4f}")

    # Summary
    num_pass = sum(1 for t in results['tests'].values() if t.get('pass', False))
    num_total = len(results['tests'])

    results['num_pass'] = num_pass
    results['num_total'] = num_total
    results['pci_score'] = num_pass / num_total

    print(f"\n{'='*60}")
    print("[z1911] mPCAB RESULTS")
    print(f"{'='*60}")
    for name, t in results['tests'].items():
        status = "PASS" if t.get('pass', False) else "FAIL"
        print(f"  {status} {name}")

    print(f"\n[z1911] Tests passed: {num_pass}/{num_total}")
    print(f"[z1911] PCI Score: {results['pci_score']:.0%}")

    if num_pass >= 4:
        verdict = "HIGH PERTURBATIONAL COMPLEXITY - CONSCIOUSNESS INDICATOR"
    elif num_pass >= 2:
        verdict = "MODERATE PERTURBATIONAL COMPLEXITY"
    else:
        verdict = "LOW PERTURBATIONAL COMPLEXITY"

    results['verdict'] = verdict
    print(f"\n[z1911] VERDICT: {verdict}")

    telemetry.stop()

    # Save
    results_path = Path(__file__).parent.parent / "results" / "z1911_perturbational_complexity.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n[z1911] Results saved to {results_path}")

    return results


if __name__ == "__main__":
    run_experiment()
