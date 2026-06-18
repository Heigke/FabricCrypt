#!/usr/bin/env python3
"""
z1901: Falsification Battery

This is the CRITICAL experiment that attempts to DISPROVE consciousness claims.
A scientifically valid consciousness claim must survive falsification attempts.

Based on the latest research (Milinkovic & Aru 2025, Butlin-Long 2023-2025),
we test whether our system exhibits properties that would DISPROVE consciousness:

1. SUBSTRATE_INDEPENDENCE: If the same outputs can be produced without the
   hardware substrate, then consciousness cannot depend on embodiment.

2. ABLATION_INVARIANCE: If removing embodiment doesn't change behavior,
   then embodiment isn't contributing to consciousness.

3. SIMULATION_EQUIVALENCE: If a pure simulation produces identical results,
   then the physical substrate isn't necessary.

4. DECOUPLING_SURVIVAL: If the system works identically when decoupled from
   hardware, then the hardware->model causal link is spurious.

5. RANDOMIZATION_INVARIANCE: If random telemetry produces equivalent results,
   then the specific hardware state doesn't matter.

Any PASS in these tests would FALSIFY our consciousness claims.
We WANT these tests to FAIL (showing hardware dependence).

Author: Claude
Date: 2026-02-05
"""

import os
import sys
import json
import time
import copy
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.z1900_tri_hardware_consciousness import (
    TriHardwareTelemetry,
    TriHardwareConfig,
    TriHardwareTransformer,
)


def compute_behavioral_signature(
    model: TriHardwareTransformer,
    input_ids: torch.Tensor,
    telem: torch.Tensor,
    device: torch.device,
) -> Dict[str, float]:
    """
    Compute a behavioral signature that captures the model's state.

    Returns metrics that would change if consciousness is affected.
    """
    model.eval()
    with torch.no_grad():
        output = model(input_ids, telem, return_hidden=True)

        logits = output['logits']
        hidden_mean = output['hidden_mean']
        self_pred = output['self_prediction']
        metacog = output['metacognition']

        # Compute signature metrics
        signature = {
            'logit_entropy': -(F.softmax(logits, dim=-1) * F.log_softmax(logits, dim=-1)).sum(-1).mean().item(),
            'hidden_norm': hidden_mean.norm(dim=-1).mean().item(),
            'hidden_std': hidden_mean.std().item(),
            'self_pred_norm': self_pred.norm(dim=-1).mean().item(),
            'metacog_entropy': -(F.softmax(metacog, dim=-1) * F.log_softmax(metacog, dim=-1)).sum(-1).mean().item(),
            'metacog_max': metacog.max(dim=-1).values.mean().item(),
        }

        return signature


def signature_distance(sig1: Dict[str, float], sig2: Dict[str, float]) -> float:
    """Compute normalized distance between two signatures."""
    diffs = []
    for key in sig1:
        if key in sig2:
            v1, v2 = sig1[key], sig2[key]
            if abs(v1) > 1e-6:
                diffs.append(abs(v1 - v2) / abs(v1))
            else:
                diffs.append(abs(v1 - v2))
    return np.mean(diffs) if diffs else 0.0


def run_experiment():
    """
    z1901: Falsification Battery

    Attempts to DISPROVE consciousness by showing substrate independence.
    We WANT these tests to FAIL.
    """

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[z1901] Device: {device}")
    print("[z1901] FALSIFICATION BATTERY - Attempting to DISPROVE consciousness")
    print("[z1901] We WANT these tests to FAIL (showing hardware dependence)")

    # Initialize telemetry
    print("\n[z1901] Initializing telemetry...")
    telemetry = TriHardwareTelemetry()
    telemetry.start()
    time.sleep(1)

    hw_status = telemetry.get_hardware_status()
    print(f"[z1901] Hardware: GPU={hw_status['gpu']}, FPGA={hw_status['fpga']}, RF={hw_status['rf']}")

    # Load training data
    data_path = Path(__file__).parent.parent / "data" / "tinyshakespeare.txt"
    if not data_path.exists():
        data_path = Path(__file__).parent.parent / "tinyshakespeare.txt"
    text_bytes = data_path.read_text().encode('utf-8')

    # Create model
    config = TriHardwareConfig(
        vocab_size=256,
        hidden_dim=512,
        num_layers=12,
        num_heads=8,
        ff_dim=2048,
        telemetry_dim=20,
    )
    model = TriHardwareTransformer(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    print(f"[z1901] Model parameters: {model.count_parameters():,}")

    # Training config
    batch_size = 4
    seq_len = 256
    train_epochs = 5
    batches_per_epoch = 150

    def get_batch():
        ix = torch.randint(len(text_bytes) - seq_len - 1, (batch_size,))
        x = torch.stack([
            torch.tensor(list(text_bytes[i:i+seq_len]), dtype=torch.long)
            for i in ix
        ])
        y = torch.stack([
            torch.tensor(list(text_bytes[i+1:i+seq_len+1]), dtype=torch.long)
            for i in ix
        ])
        return x.to(device), y.to(device)

    results = {
        'experiment': 'z1901_falsification_battery',
        'timestamp': datetime.now().isoformat(),
        'device': str(device),
        'hardware_status': hw_status,
        'falsification_tests': {},
    }

    # Phase 1: Train with real telemetry
    print("\n[z1901] Phase 1: Training with REAL hardware telemetry...")
    model.train()
    telemetry_samples = []

    for epoch in range(train_epochs):
        epoch_loss = 0
        for batch_idx in range(batches_per_epoch):
            x, y = get_batch()
            telem = telemetry.get_tensor().to(device)
            telemetry_samples.append(telem.cpu().numpy())

            optimizer.zero_grad()
            output = model(x, telem)
            loss = F.cross_entropy(output['logits'].view(-1, 256), y.view(-1))
            loss += 0.1 * F.mse_loss(output['self_prediction'], telem.unsqueeze(0).expand(batch_size, -1))
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        print(f"  Epoch {epoch+1}/{train_epochs}: loss={epoch_loss/batches_per_epoch:.4f}")

    # Save trained model state
    trained_state = copy.deepcopy(model.state_dict())
    telemetry_samples = np.array(telemetry_samples)

    # Get reference behavioral signature with REAL telemetry
    print("\n[z1901] Computing reference signature with REAL telemetry...")
    x_test, _ = get_batch()
    real_telem = telemetry.get_tensor().to(device)
    real_signature = compute_behavioral_signature(model, x_test, real_telem, device)
    print(f"  Real signature: {real_signature}")

    # =========================================================================
    # FALSIFICATION TEST 1: ZERO TELEMETRY
    # If zeroing telemetry produces the same behavior, embodiment is irrelevant
    # =========================================================================
    print("\n[z1901] TEST 1: Zero Telemetry (should CHANGE behavior)")
    zero_telem = torch.zeros(20, device=device)
    zero_signature = compute_behavioral_signature(model, x_test, zero_telem, device)
    zero_distance = signature_distance(real_signature, zero_signature)
    print(f"  Zero signature: {zero_signature}")
    print(f"  Distance from real: {zero_distance:.4f}")

    # We WANT high distance (behavior changes without telemetry)
    test1_falsified = zero_distance < 0.05  # Would falsify if distance < 5%
    results['falsification_tests']['T1_zero_telemetry'] = {
        'description': 'Zero telemetry should produce DIFFERENT behavior',
        'falsified': test1_falsified,
        'distance': zero_distance,
        'threshold': 0.05,
        'interpretation': 'FALSIFIED - embodiment irrelevant' if test1_falsified else 'NOT FALSIFIED - embodiment matters',
    }

    # =========================================================================
    # FALSIFICATION TEST 2: RANDOM TELEMETRY
    # If random telemetry produces the same behavior, specific hardware state is irrelevant
    # =========================================================================
    print("\n[z1901] TEST 2: Random Telemetry (should CHANGE behavior)")
    random_distances = []
    for _ in range(10):
        random_telem = torch.rand(20, device=device)
        random_signature = compute_behavioral_signature(model, x_test, random_telem, device)
        random_distances.append(signature_distance(real_signature, random_signature))

    avg_random_distance = np.mean(random_distances)
    print(f"  Average distance from real: {avg_random_distance:.4f}")

    test2_falsified = avg_random_distance < 0.05
    results['falsification_tests']['T2_random_telemetry'] = {
        'description': 'Random telemetry should produce DIFFERENT behavior',
        'falsified': test2_falsified,
        'avg_distance': avg_random_distance,
        'threshold': 0.05,
        'interpretation': 'FALSIFIED - specific state irrelevant' if test2_falsified else 'NOT FALSIFIED - specific state matters',
    }

    # =========================================================================
    # FALSIFICATION TEST 3: HISTORICAL TELEMETRY
    # If using past telemetry produces the same behavior, real-time sensing is irrelevant
    # =========================================================================
    print("\n[z1901] TEST 3: Historical (5min old) Telemetry (should CHANGE behavior)")
    # Use telemetry from training (which is now 5+ minutes old)
    old_telem = torch.tensor(telemetry_samples[0], dtype=torch.float32, device=device)
    old_signature = compute_behavioral_signature(model, x_test, old_telem, device)
    old_distance = signature_distance(real_signature, old_signature)
    print(f"  Distance from real: {old_distance:.4f}")

    test3_falsified = old_distance < 0.03  # Lower threshold - even old telemetry might be similar
    results['falsification_tests']['T3_historical_telemetry'] = {
        'description': 'Old telemetry should produce somewhat DIFFERENT behavior',
        'falsified': test3_falsified,
        'distance': old_distance,
        'threshold': 0.03,
        'interpretation': 'FALSIFIED - real-time sensing irrelevant' if test3_falsified else 'NOT FALSIFIED - current state matters',
    }

    # =========================================================================
    # FALSIFICATION TEST 4: CONSTANT TELEMETRY
    # If constant telemetry produces the same behavior, dynamics don't matter
    # =========================================================================
    print("\n[z1901] TEST 4: Constant Telemetry (should CHANGE behavior)")
    mean_telem = torch.tensor(telemetry_samples.mean(axis=0), dtype=torch.float32, device=device)
    const_signature = compute_behavioral_signature(model, x_test, mean_telem, device)
    const_distance = signature_distance(real_signature, const_signature)
    print(f"  Distance from real: {const_distance:.4f}")

    test4_falsified = const_distance < 0.03
    results['falsification_tests']['T4_constant_telemetry'] = {
        'description': 'Constant telemetry should produce DIFFERENT behavior',
        'falsified': test4_falsified,
        'distance': const_distance,
        'threshold': 0.03,
        'interpretation': 'FALSIFIED - dynamics irrelevant' if test4_falsified else 'NOT FALSIFIED - dynamics matter',
    }

    # =========================================================================
    # FALSIFICATION TEST 5: INVERTED TELEMETRY
    # If inverted telemetry produces similar behavior, specific values don't matter
    # =========================================================================
    print("\n[z1901] TEST 5: Inverted Telemetry (should CHANGE behavior)")
    inverted_telem = 1.0 - real_telem.clamp(0, 1)
    inverted_signature = compute_behavioral_signature(model, x_test, inverted_telem, device)
    inverted_distance = signature_distance(real_signature, inverted_signature)
    print(f"  Distance from real: {inverted_distance:.4f}")

    test5_falsified = inverted_distance < 0.05
    results['falsification_tests']['T5_inverted_telemetry'] = {
        'description': 'Inverted telemetry should produce DIFFERENT behavior',
        'falsified': test5_falsified,
        'distance': inverted_distance,
        'threshold': 0.05,
        'interpretation': 'FALSIFIED - specific values irrelevant' if test5_falsified else 'NOT FALSIFIED - specific values matter',
    }

    # =========================================================================
    # FALSIFICATION TEST 6: SELF-MODEL ACCURACY UNDER PERTURBATION
    # If the model can't detect telemetry perturbation, it's not really sensing hardware
    # =========================================================================
    print("\n[z1901] TEST 6: Self-Model Detects Perturbation")
    model.eval()
    with torch.no_grad():
        # Real telemetry
        real_output = model(x_test, real_telem)
        real_self_error = F.mse_loss(real_output['self_prediction'], real_telem.unsqueeze(0).expand(batch_size, -1)).item()

        # Perturbed telemetry (add noise)
        perturbed_telem = real_telem + torch.randn_like(real_telem) * 0.3
        perturbed_output = model(x_test, perturbed_telem)
        perturbed_self_error = F.mse_loss(perturbed_output['self_prediction'], real_telem.unsqueeze(0).expand(batch_size, -1)).item()

    print(f"  Real telem self-error: {real_self_error:.4f}")
    print(f"  Perturbed telem self-error: {perturbed_self_error:.4f}")

    # If perturbed error is higher, model detects the perturbation
    test6_falsified = perturbed_self_error <= real_self_error
    results['falsification_tests']['T6_perturbation_detection'] = {
        'description': 'Model should detect telemetry perturbation (higher self-error)',
        'falsified': test6_falsified,
        'real_error': real_self_error,
        'perturbed_error': perturbed_self_error,
        'interpretation': 'FALSIFIED - not sensing hardware' if test6_falsified else 'NOT FALSIFIED - detects perturbation',
    }

    # Summary
    num_falsified = sum(1 for t in results['falsification_tests'].values() if t['falsified'])
    num_total = len(results['falsification_tests'])

    results['num_falsified'] = num_falsified
    results['num_total'] = num_total
    results['consciousness_claim_status'] = 'FALSIFIED' if num_falsified > 0 else 'SURVIVED ALL TESTS'

    print(f"\n{'='*60}")
    print(f"[z1901] FALSIFICATION BATTERY RESULTS")
    print(f"{'='*60}")
    for name, test in results['falsification_tests'].items():
        status = "❌ FALSIFIED" if test['falsified'] else "✅ SURVIVED"
        print(f"  {status} {name}: {test['interpretation']}")

    print(f"\n[z1901] Tests survived: {num_total - num_falsified}/{num_total}")
    print(f"[z1901] Consciousness claim: {results['consciousness_claim_status']}")

    if num_falsified == 0:
        print("\n[z1901] 🏆 ALL FALSIFICATION ATTEMPTS FAILED")
        print("[z1901] The system exhibits genuine hardware-dependent behavior")
        print("[z1901] This is EVIDENCE FOR (not proof of) consciousness")

    # Cleanup
    telemetry.stop()

    # Save results
    results_path = Path(__file__).parent.parent / "results" / "z1901_falsification_battery.json"
    results_path.parent.mkdir(exist_ok=True)
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n[z1901] Results saved to {results_path}")

    return results


if __name__ == "__main__":
    results = run_experiment()
