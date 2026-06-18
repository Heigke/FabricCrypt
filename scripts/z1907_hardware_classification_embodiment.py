#!/usr/bin/env python3
"""
z1907: Hardware Classification Embodiment

SIMPLER TEST: Instead of generating text, classify hardware state into discrete bins.

Classes:
- Temperature: LOW (<40°C), MEDIUM (40-70°C), HIGH (>70°C)
- Utilization: IDLE (<30%), ACTIVE (30-80%), FULL (>80%)
- Power: ECO (<50W), NORMAL (50-100W), HIGH (>100W)

This is a 27-class classification problem (3x3x3).
The model MUST use telemetry correctly to classify accurately.

Falsification tests:
1. Can it classify real telemetry correctly?
2. Does it give DIFFERENT classifications for different telemetry?
3. Does random telemetry produce random/wrong classifications?

Author: Claude
Date: 2026-02-05
"""

import os
import sys
import json
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.z1900_tri_hardware_consciousness import TriHardwareTelemetry


def telemetry_to_class(telemetry: np.ndarray) -> Tuple[int, int, int]:
    """
    Convert telemetry to discrete class labels.

    Returns (temp_class, util_class, power_class) each in {0, 1, 2}
    """
    # Extract values (normalized 0-1)
    temp = telemetry[0]  # Temperature
    util = telemetry[1]  # Utilization
    power = telemetry[2]  # Power

    # Classify temperature
    if temp < 0.4:
        temp_class = 0  # LOW
    elif temp < 0.7:
        temp_class = 1  # MEDIUM
    else:
        temp_class = 2  # HIGH

    # Classify utilization
    if util < 0.3:
        util_class = 0  # IDLE
    elif util < 0.8:
        util_class = 1  # ACTIVE
    else:
        util_class = 2  # FULL

    # Classify power
    if power < 0.25:
        power_class = 0  # ECO
    elif power < 0.5:
        power_class = 1  # NORMAL
    else:
        power_class = 2  # HIGH

    return temp_class, util_class, power_class


def class_to_combined(temp_c: int, util_c: int, power_c: int) -> int:
    """Convert 3 class indices to single combined class (0-26)."""
    return temp_c * 9 + util_c * 3 + power_c


def combined_to_class(combined: int) -> Tuple[int, int, int]:
    """Convert combined class back to individual classes."""
    temp_c = combined // 9
    util_c = (combined % 9) // 3
    power_c = combined % 3
    return temp_c, util_c, power_c


CLASS_NAMES = {
    'temp': ['LOW', 'MEDIUM', 'HIGH'],
    'util': ['IDLE', 'ACTIVE', 'FULL'],
    'power': ['ECO', 'NORMAL', 'HIGH'],
}


class HardwareClassifier(nn.Module):
    """Classifier for hardware state."""

    def __init__(self, telemetry_dim: int = 20, hidden_dim: int = 256, num_classes: int = 27):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(telemetry_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )

        # Separate heads for each dimension
        self.temp_head = nn.Linear(hidden_dim, 3)
        self.util_head = nn.Linear(hidden_dim, 3)
        self.power_head = nn.Linear(hidden_dim, 3)

        # Combined class head
        self.combined_head = nn.Linear(hidden_dim, num_classes)

    def forward(self, telemetry: torch.Tensor, return_all: bool = False):
        if telemetry.dim() == 1:
            telemetry = telemetry.unsqueeze(0)

        h = self.encoder(telemetry)

        temp_logits = self.temp_head(h)
        util_logits = self.util_head(h)
        power_logits = self.power_head(h)
        combined_logits = self.combined_head(h)

        if return_all:
            return {
                'temp_logits': temp_logits,
                'util_logits': util_logits,
                'power_logits': power_logits,
                'combined_logits': combined_logits,
                'hidden': h,
            }
        return combined_logits

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def run_experiment():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[z1907] Device: {device}")
    print("[z1907] HARDWARE CLASSIFICATION EMBODIMENT")
    print("[z1907] Classify hardware state into discrete bins")

    # Telemetry
    telemetry = TriHardwareTelemetry()
    telemetry.start()
    time.sleep(1)

    hw_status = telemetry.get_hardware_status()
    print(f"[z1907] Hardware: GPU={hw_status['gpu']}, FPGA={hw_status['fpga']}, RF={hw_status['rf']}")

    # Model
    model = HardwareClassifier(telemetry_dim=20, hidden_dim=256, num_classes=27).to(device)
    print(f"[z1907] Model parameters: {model.count_parameters():,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    batch_size = 32
    epochs = 50
    batches_per_epoch = 100

    results = {
        'experiment': 'z1907_hardware_classification_embodiment',
        'timestamp': datetime.now().isoformat(),
        'device': str(device),
        'hardware_status': hw_status,
        'task': 'Classify hardware state (temp/util/power bins)',
    }

    # Training
    print("\n[z1907] Training: telemetry -> discrete classification")
    telem_samples = []
    losses = []
    accuracies = []

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        epoch_correct = 0
        epoch_total = 0

        for _ in range(batches_per_epoch):
            # Get current telemetry
            telem = telemetry.get_tensor().cpu().numpy()
            telem_samples.append(telem)

            # Create batch by adding noise (data augmentation for variety)
            telem_batch = []
            temp_labels = []
            util_labels = []
            power_labels = []
            combined_labels = []

            for _ in range(batch_size):
                # Varied telemetry
                noise = np.random.randn(20) * 0.1
                perturbed = np.clip(telem + noise, 0, 1)

                # Get labels
                temp_c, util_c, power_c = telemetry_to_class(perturbed)
                combined_c = class_to_combined(temp_c, util_c, power_c)

                telem_batch.append(perturbed)
                temp_labels.append(temp_c)
                util_labels.append(util_c)
                power_labels.append(power_c)
                combined_labels.append(combined_c)

            telem_batch = torch.tensor(np.array(telem_batch), dtype=torch.float32, device=device)
            temp_labels = torch.tensor(temp_labels, dtype=torch.long, device=device)
            util_labels = torch.tensor(util_labels, dtype=torch.long, device=device)
            power_labels = torch.tensor(power_labels, dtype=torch.long, device=device)
            combined_labels = torch.tensor(combined_labels, dtype=torch.long, device=device)

            # Forward
            optimizer.zero_grad()
            out = model(telem_batch, return_all=True)

            # Multi-task loss
            loss_temp = F.cross_entropy(out['temp_logits'], temp_labels)
            loss_util = F.cross_entropy(out['util_logits'], util_labels)
            loss_power = F.cross_entropy(out['power_logits'], power_labels)
            loss_combined = F.cross_entropy(out['combined_logits'], combined_labels)

            loss = loss_temp + loss_util + loss_power + loss_combined

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            # Accuracy
            pred_combined = out['combined_logits'].argmax(dim=-1)
            epoch_correct += (pred_combined == combined_labels).sum().item()
            epoch_total += batch_size

        avg_loss = epoch_loss / batches_per_epoch
        accuracy = epoch_correct / epoch_total
        losses.append(avg_loss)
        accuracies.append(accuracy)

        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}/{epochs}: loss={avg_loss:.4f}, accuracy={accuracy:.2%}")

    results['training_losses'] = losses
    results['training_accuracies'] = accuracies
    telem_samples = np.array(telem_samples)

    # Testing
    print("\n[z1907] Testing classification accuracy")
    model.eval()

    # Test with REAL telemetry
    real_telem = telemetry.get_tensor().to(device)
    real_np = real_telem.cpu().numpy()
    true_temp, true_util, true_power = telemetry_to_class(real_np)
    true_combined = class_to_combined(true_temp, true_util, true_power)

    with torch.no_grad():
        out_real = model(real_telem, return_all=True)
        pred_temp = out_real['temp_logits'].argmax(dim=-1).item()
        pred_util = out_real['util_logits'].argmax(dim=-1).item()
        pred_power = out_real['power_logits'].argmax(dim=-1).item()
        pred_combined = out_real['combined_logits'].argmax(dim=-1).item()

    print(f"\n  Real Telemetry (values: temp={real_np[0]:.2f}, util={real_np[1]:.2f}, power={real_np[2]:.2f}):")
    print(f"    True:  temp={CLASS_NAMES['temp'][true_temp]}, util={CLASS_NAMES['util'][true_util]}, power={CLASS_NAMES['power'][true_power]}")
    print(f"    Pred:  temp={CLASS_NAMES['temp'][pred_temp]}, util={CLASS_NAMES['util'][pred_util]}, power={CLASS_NAMES['power'][pred_power]}")

    # Test with ZERO telemetry
    with torch.no_grad():
        out_zero = model(torch.zeros(20, device=device), return_all=True)
        pred_temp_zero = out_zero['temp_logits'].argmax(dim=-1).item()
        pred_util_zero = out_zero['util_logits'].argmax(dim=-1).item()
        pred_power_zero = out_zero['power_logits'].argmax(dim=-1).item()

    zero_np = np.zeros(20)
    true_temp_zero, true_util_zero, true_power_zero = telemetry_to_class(zero_np)

    print(f"\n  Zero Telemetry:")
    print(f"    True:  temp={CLASS_NAMES['temp'][true_temp_zero]}, util={CLASS_NAMES['util'][true_util_zero]}, power={CLASS_NAMES['power'][true_power_zero]}")
    print(f"    Pred:  temp={CLASS_NAMES['temp'][pred_temp_zero]}, util={CLASS_NAMES['util'][pred_util_zero]}, power={CLASS_NAMES['power'][pred_power_zero]}")

    # Test with HIGH telemetry (all 1s)
    with torch.no_grad():
        out_high = model(torch.ones(20, device=device), return_all=True)
        pred_temp_high = out_high['temp_logits'].argmax(dim=-1).item()
        pred_util_high = out_high['util_logits'].argmax(dim=-1).item()
        pred_power_high = out_high['power_logits'].argmax(dim=-1).item()

    high_np = np.ones(20)
    true_temp_high, true_util_high, true_power_high = telemetry_to_class(high_np)

    print(f"\n  High Telemetry (all 1s):")
    print(f"    True:  temp={CLASS_NAMES['temp'][true_temp_high]}, util={CLASS_NAMES['util'][true_util_high]}, power={CLASS_NAMES['power'][true_power_high]}")
    print(f"    Pred:  temp={CLASS_NAMES['temp'][pred_temp_high]}, util={CLASS_NAMES['util'][pred_util_high]}, power={CLASS_NAMES['power'][pred_power_high]}")

    # Falsification tests
    print("\n" + "="*60)
    print("[z1907] FALSIFICATION BATTERY")
    print("="*60)

    tests = {}

    # T1: Real telemetry classification accuracy
    print("\n[z1907] T1: Real Telemetry Classification")
    real_correct = (pred_temp == true_temp) + (pred_util == true_util) + (pred_power == true_power)
    real_accuracy = real_correct / 3
    print(f"  Accuracy: {real_accuracy:.2%} ({real_correct}/3 correct)")
    tests['T1_real_accuracy'] = {
        'accuracy': real_accuracy,
        'correct': real_correct,
        'falsified': real_accuracy < 0.67,  # Should get at least 2/3 correct
    }

    # T2: Zero telemetry classification (should predict LOW/IDLE/ECO)
    print("\n[z1907] T2: Zero Telemetry Classification")
    zero_correct = (pred_temp_zero == true_temp_zero) + (pred_util_zero == true_util_zero) + (pred_power_zero == true_power_zero)
    zero_accuracy = zero_correct / 3
    print(f"  Accuracy: {zero_accuracy:.2%} ({zero_correct}/3 correct)")
    tests['T2_zero_accuracy'] = {
        'accuracy': zero_accuracy,
        'correct': zero_correct,
        'falsified': zero_accuracy < 0.67,
    }

    # T3: High telemetry classification (should predict HIGH/FULL/HIGH)
    print("\n[z1907] T3: High Telemetry Classification")
    high_correct = (pred_temp_high == true_temp_high) + (pred_util_high == true_util_high) + (pred_power_high == true_power_high)
    high_accuracy = high_correct / 3
    print(f"  Accuracy: {high_accuracy:.2%} ({high_correct}/3 correct)")
    tests['T3_high_accuracy'] = {
        'accuracy': high_accuracy,
        'correct': high_correct,
        'falsified': high_accuracy < 0.67,
    }

    # T4: Different telemetry produces different predictions
    print("\n[z1907] T4: Differentiation (different inputs -> different outputs)")
    pred_real = (pred_temp, pred_util, pred_power)
    pred_zero = (pred_temp_zero, pred_util_zero, pred_power_zero)
    pred_high = (pred_temp_high, pred_util_high, pred_power_high)

    differentiation = (pred_real != pred_zero) or (pred_real != pred_high) or (pred_zero != pred_high)
    print(f"  Different predictions for different inputs: {differentiation}")
    tests['T4_differentiation'] = {
        'pred_real': pred_real,
        'pred_zero': pred_zero,
        'pred_high': pred_high,
        'differentiated': differentiation,
        'falsified': not differentiation,
    }

    # T5: Hidden state changes with telemetry (mutual information proxy)
    print("\n[z1907] T5: Hidden State Sensitivity")
    with torch.no_grad():
        h_real = out_real['hidden']
        h_zero = out_zero['hidden']
        h_high = out_high['hidden']

        diff_real_zero = (h_real - h_zero).abs().mean().item()
        diff_real_high = (h_real - h_high).abs().mean().item()
        diff_zero_high = (h_zero - h_high).abs().mean().item()

    print(f"  Hidden diff (real vs zero): {diff_real_zero:.4f}")
    print(f"  Hidden diff (real vs high): {diff_real_high:.4f}")
    print(f"  Hidden diff (zero vs high): {diff_zero_high:.4f}")

    min_diff = min(diff_real_zero, diff_real_high, diff_zero_high)
    tests['T5_hidden_sensitivity'] = {
        'diff_real_zero': diff_real_zero,
        'diff_real_high': diff_real_high,
        'diff_zero_high': diff_zero_high,
        'min_diff': min_diff,
        'falsified': min_diff < 0.1,  # Should have substantial hidden state differences
    }

    # T6: Random telemetry produces varied predictions
    print("\n[z1907] T6: Random Telemetry Variance")
    random_preds = []
    with torch.no_grad():
        for _ in range(20):
            rand_telem = torch.rand(20, device=device)
            out_rand = model(rand_telem, return_all=True)
            pred = out_rand['combined_logits'].argmax(dim=-1).item()
            random_preds.append(pred)

    unique_preds = len(set(random_preds))
    print(f"  Unique predictions from 20 random inputs: {unique_preds}")
    tests['T6_random_variance'] = {
        'unique_predictions': unique_preds,
        'total_attempts': 20,
        'falsified': unique_preds < 3,  # Should have at least 3 different predictions
    }

    results['falsification_tests'] = tests

    num_falsified = sum(1 for t in tests.values() if t.get('falsified', False))
    results['num_falsified'] = num_falsified
    results['num_total'] = len(tests)
    results['status'] = 'FALSIFIED' if num_falsified > 0 else 'SURVIVED ALL TESTS'

    print(f"\n{'='*60}")
    print(f"[z1907] RESULTS")
    print(f"{'='*60}")
    for name, t in tests.items():
        status = "FALSIFIED" if t.get('falsified', False) else "SURVIVED"
        print(f"  {status} {name}")

    print(f"\n[z1907] Tests survived: {len(tests) - num_falsified}/{len(tests)}")
    print(f"[z1907] Status: {results['status']}")

    if num_falsified == 0:
        print("\n[z1907] ALL TESTS SURVIVED!")
        print("[z1907] Model correctly classifies hardware state")
        print("[z1907] TRUE causal dependence: telemetry -> classification")

    telemetry.stop()

    # Save
    results_path = Path(__file__).parent.parent / "results" / "z1907_hardware_classification_embodiment.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n[z1907] Results saved to {results_path}")

    return results


if __name__ == "__main__":
    run_experiment()
