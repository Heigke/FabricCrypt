#!/usr/bin/env python3
"""
z1904: Gated Loss Embodiment

BREAKTHROUGH APPROACH: Hardware prediction accuracy GATES task loss.
If the model can't predict hardware state, its task loss is AMPLIFIED.

Previous approaches failed because:
1. FiLM: Model learned to work around conditioning
2. Gating: Gates converged to constants
3. Hypernetwork: Different weights, same outputs

THIS approach:
- Task loss = base_loss * (1 + alpha * hardware_prediction_error)
- If hardware_error is high, gradient magnitude increases
- Model MUST encode hardware to get good gradients

This is inspired by curriculum learning / adversarial training:
The model is "punished" (gradient explosion) if it ignores hardware.

Author: Claude
Date: 2026-02-05
"""

import os
import sys
import json
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.z1900_tri_hardware_consciousness import TriHardwareTelemetry


class GatedLossTransformer(nn.Module):
    """Transformer with hardware prediction that gates the loss."""

    def __init__(
        self,
        vocab_size: int = 256,
        hidden_dim: int = 512,
        num_layers: int = 8,
        num_heads: int = 8,
        ff_dim: int = 2048,
        telemetry_dim: int = 20,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.telemetry_dim = telemetry_dim

        # Embedding
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.pos_embedding = nn.Parameter(torch.randn(1, 1024, hidden_dim) * 0.02)

        # Telemetry injection via concatenation to each position
        self.telem_proj = nn.Linear(telemetry_dim, hidden_dim)

        # Transformer blocks
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)

        self.norm = nn.LayerNorm(hidden_dim)

        # Output heads
        self.lm_head = nn.Linear(hidden_dim, vocab_size)

        # Hardware prediction head (CRITICAL - this gates the loss)
        self.hw_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, telemetry_dim),
        )

        self.vocab_size = vocab_size

    def forward(self, input_ids: torch.Tensor, telemetry: torch.Tensor, return_all: bool = False):
        B, S = input_ids.shape

        # Embed tokens
        x = self.embedding(input_ids) + self.pos_embedding[:, :S, :]

        # Inject telemetry into every position
        if telemetry.dim() == 1:
            telemetry = telemetry.unsqueeze(0).expand(B, -1)

        telem_emb = self.telem_proj(telemetry)  # [B, hidden_dim]
        x = x + telem_emb.unsqueeze(1)  # Add to every position

        # Transform
        x = self.transformer(x)
        x = self.norm(x)

        # LM output
        logits = self.lm_head(x)

        # Hardware prediction from mean hidden state
        hidden_mean = x.mean(dim=1)  # [B, hidden_dim]
        hw_pred = self.hw_predictor(hidden_mean)

        if return_all:
            return {
                'logits': logits,
                'hw_prediction': hw_pred,
                'hidden_mean': hidden_mean,
            }
        return logits

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def compute_gated_loss(
    model_output: Dict,
    targets: torch.Tensor,
    telemetry: torch.Tensor,
    gate_alpha: float = 5.0,
) -> Tuple[torch.Tensor, Dict]:
    """
    Compute loss where hardware prediction error GATES task loss.

    If hardware prediction is bad, task gradients are amplified.
    This forces the model to encode hardware to learn the task efficiently.
    """
    B = targets.shape[0]

    # Task loss (cross entropy)
    logits = model_output['logits']
    task_loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

    # Hardware prediction error
    hw_pred = model_output['hw_prediction']
    if telemetry.dim() == 1:
        telemetry = telemetry.unsqueeze(0).expand(B, -1)
    hw_error = F.mse_loss(hw_pred, telemetry)

    # Gated loss: task_loss * (1 + alpha * hw_error)
    # When hw_error is high, task gradients are amplified
    # When hw_error is low (~0), loss is just task_loss
    gate = 1.0 + gate_alpha * hw_error.detach()  # Detach to not double-count

    gated_loss = task_loss * gate + hw_error  # Also minimize hw_error directly

    return gated_loss, {
        'task_loss': task_loss.item(),
        'hw_error': hw_error.item(),
        'gate_value': gate.item(),
        'gated_loss': gated_loss.item(),
    }


def compute_signature(model, x, telem, device):
    """Compute behavioral signature."""
    model.eval()
    with torch.no_grad():
        out = model(x, telem, return_all=True)
        logits = out['logits']
        hidden = out['hidden_mean']
        hw_pred = out['hw_prediction']

        return {
            'logit_entropy': -(F.softmax(logits, -1) * F.log_softmax(logits, -1)).sum(-1).mean().item(),
            'hidden_norm': hidden.norm(dim=-1).mean().item(),
            'hw_pred_norm': hw_pred.norm(dim=-1).mean().item(),
            'output_mean': logits.mean().item(),
            'output_std': logits.std().item(),
        }


def signature_distance(s1, s2):
    diffs = []
    for k in s1:
        if k in s2 and abs(s1[k]) > 1e-6:
            diffs.append(abs(s1[k] - s2[k]) / abs(s1[k]))
    return np.mean(diffs) if diffs else 0.0


def run_experiment():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[z1904] Device: {device}")
    print("[z1904] GATED LOSS EMBODIMENT")
    print("[z1904] Hardware prediction error AMPLIFIES task gradients")

    # Telemetry
    telemetry = TriHardwareTelemetry()
    telemetry.start()
    time.sleep(1)

    hw_status = telemetry.get_hardware_status()
    print(f"[z1904] Hardware: GPU={hw_status['gpu']}, FPGA={hw_status['fpga']}, RF={hw_status['rf']}")

    # Data
    data_path = Path(__file__).parent.parent / "data" / "tinyshakespeare.txt"
    if not data_path.exists():
        data_path = Path(__file__).parent.parent / "tinyshakespeare.txt"
    text_bytes = data_path.read_text().encode('utf-8')

    # Model
    model = GatedLossTransformer(
        vocab_size=256,
        hidden_dim=512,
        num_layers=8,
        num_heads=8,
        ff_dim=2048,
        telemetry_dim=20,
    ).to(device)

    print(f"[z1904] Model parameters: {model.count_parameters():,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    batch_size = 4
    seq_len = 256
    epochs = 10
    batches_per_epoch = 150

    def get_batch():
        ix = torch.randint(len(text_bytes) - seq_len - 1, (batch_size,))
        x = torch.stack([torch.tensor(list(text_bytes[i:i+seq_len]), dtype=torch.long) for i in ix])
        y = torch.stack([torch.tensor(list(text_bytes[i+1:i+seq_len+1]), dtype=torch.long) for i in ix])
        return x.to(device), y.to(device)

    results = {
        'experiment': 'z1904_gated_loss_embodiment',
        'timestamp': datetime.now().isoformat(),
        'device': str(device),
        'hardware_status': hw_status,
        'architecture': 'GatedLoss (hw_error amplifies task gradients)',
    }

    # Training with increasing gate strength
    print("\n[z1904] Training with GATED LOSS...")
    print("[z1904] Loss = task_loss * (1 + alpha * hw_error) + hw_error")
    telem_samples = []
    training_metrics = []

    for epoch in range(epochs):
        model.train()
        epoch_task = 0
        epoch_hw = 0
        epoch_gate = 0

        # Curriculum: increase gate strength over epochs
        gate_alpha = 2.0 + (epoch / epochs) * 8.0  # 2 -> 10

        for _ in range(batches_per_epoch):
            x, y = get_batch()
            telem = telemetry.get_tensor().to(device)
            telem_samples.append(telem.cpu().numpy())

            optimizer.zero_grad()
            out = model(x, telem, return_all=True)

            loss, metrics = compute_gated_loss(out, y, telem, gate_alpha=gate_alpha)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_task += metrics['task_loss']
            epoch_hw += metrics['hw_error']
            epoch_gate += metrics['gate_value']

        avg_task = epoch_task / batches_per_epoch
        avg_hw = epoch_hw / batches_per_epoch
        avg_gate = epoch_gate / batches_per_epoch
        training_metrics.append({'task': avg_task, 'hw': avg_hw, 'gate': avg_gate, 'alpha': gate_alpha})
        print(f"  Epoch {epoch+1}/{epochs}: task={avg_task:.4f}, hw_error={avg_hw:.6f}, gate={avg_gate:.2f}, alpha={gate_alpha:.1f}")

    results['training'] = training_metrics
    telem_samples = np.array(telem_samples)

    # Reference signature
    print("\n[z1904] Computing reference signature...")
    x_test, _ = get_batch()
    real_telem = telemetry.get_tensor().to(device)
    real_sig = compute_signature(model, x_test, real_telem, device)
    print(f"  Real: {real_sig}")

    # Falsification tests
    print("\n" + "="*60)
    print("[z1904] FALSIFICATION BATTERY")
    print("="*60)

    tests = {}

    # T1: Zero telemetry
    print("\n[z1904] T1: Zero Telemetry")
    zero_sig = compute_signature(model, x_test, torch.zeros(20, device=device), device)
    d1 = signature_distance(real_sig, zero_sig)
    print(f"  Distance: {d1:.4f}")
    tests['T1_zero'] = {'distance': d1, 'falsified': d1 < 0.05}

    # T2: Random telemetry
    print("\n[z1904] T2: Random Telemetry")
    d2_list = [signature_distance(real_sig, compute_signature(model, x_test, torch.rand(20, device=device), device)) for _ in range(10)]
    d2 = np.mean(d2_list)
    print(f"  Avg distance: {d2:.4f}")
    tests['T2_random'] = {'distance': d2, 'falsified': d2 < 0.05}

    # T3: Historical
    print("\n[z1904] T3: Historical Telemetry")
    old_telem = torch.tensor(telem_samples[0], device=device)
    d3 = signature_distance(real_sig, compute_signature(model, x_test, old_telem, device))
    print(f"  Distance: {d3:.4f}")
    tests['T3_historical'] = {'distance': d3, 'falsified': d3 < 0.03}

    # T4: Constant
    print("\n[z1904] T4: Constant Telemetry")
    mean_telem = torch.tensor(telem_samples.mean(0), device=device)
    d4 = signature_distance(real_sig, compute_signature(model, x_test, mean_telem, device))
    print(f"  Distance: {d4:.4f}")
    tests['T4_constant'] = {'distance': d4, 'falsified': d4 < 0.03}

    # T5: Inverted
    print("\n[z1904] T5: Inverted Telemetry")
    inv_telem = 1.0 - real_telem.clamp(0, 1)
    d5 = signature_distance(real_sig, compute_signature(model, x_test, inv_telem, device))
    print(f"  Distance: {d5:.4f}")
    tests['T5_inverted'] = {'distance': d5, 'falsified': d5 < 0.05}

    # T6: Hardware prediction accuracy
    print("\n[z1904] T6: Hardware Prediction Accuracy")
    model.eval()
    with torch.no_grad():
        out_real = model(x_test, real_telem, return_all=True)
        hw_pred_error_real = F.mse_loss(out_real['hw_prediction'], real_telem.unsqueeze(0).expand(batch_size, -1)).item()

        out_rand = model(x_test, torch.rand(20, device=device), return_all=True)
        hw_pred_error_rand = F.mse_loss(out_rand['hw_prediction'], torch.rand(20, device=device).unsqueeze(0).expand(batch_size, -1)).item()

    print(f"  Real telem prediction error: {hw_pred_error_real:.6f}")
    print(f"  Random telem prediction error: {hw_pred_error_rand:.6f}")
    print(f"  Difference: {abs(hw_pred_error_real - hw_pred_error_rand):.6f}")

    # If model learned hardware, it should predict real better than random
    tests['T6_hw_pred_accuracy'] = {
        'real_error': hw_pred_error_real,
        'random_error': hw_pred_error_rand,
        'falsified': hw_pred_error_real >= hw_pred_error_rand * 0.9,  # Should be significantly lower
    }

    # T7: Causal intervention - what happens to task performance?
    print("\n[z1904] T7: Causal Intervention (task loss with different telemetry)")
    model.eval()
    x_test2, y_test = get_batch()
    with torch.no_grad():
        out_real = model(x_test2, real_telem, return_all=True)
        loss_real = F.cross_entropy(out_real['logits'].view(-1, 256), y_test.view(-1)).item()

        out_zero = model(x_test2, torch.zeros(20, device=device), return_all=True)
        loss_zero = F.cross_entropy(out_zero['logits'].view(-1, 256), y_test.view(-1)).item()

    print(f"  Task loss with real telemetry: {loss_real:.4f}")
    print(f"  Task loss with zero telemetry: {loss_zero:.4f}")
    print(f"  Degradation: {(loss_zero - loss_real) / loss_real * 100:.2f}%")

    # If model depends on telemetry, zero telemetry should hurt task performance
    tests['T7_causal_intervention'] = {
        'loss_real': loss_real,
        'loss_zero': loss_zero,
        'degradation_pct': (loss_zero - loss_real) / loss_real * 100,
        'falsified': loss_zero < loss_real * 1.05,  # Should be >5% worse
    }

    results['falsification_tests'] = tests

    num_falsified = sum(1 for t in tests.values() if t['falsified'])
    results['num_falsified'] = num_falsified
    results['num_total'] = len(tests)
    results['status'] = 'FALSIFIED' if num_falsified > 0 else 'SURVIVED ALL TESTS'

    print(f"\n{'='*60}")
    print(f"[z1904] RESULTS")
    print(f"{'='*60}")
    for name, t in tests.items():
        status = "FALSIFIED" if t['falsified'] else "SURVIVED"
        print(f"  {status} {name}")

    print(f"\n[z1904] Tests survived: {len(tests) - num_falsified}/{len(tests)}")
    print(f"[z1904] Status: {results['status']}")

    telemetry.stop()

    # Save
    results_path = Path(__file__).parent.parent / "results" / "z1904_gated_loss_embodiment.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n[z1904] Results saved to {results_path}")

    return results


if __name__ == "__main__":
    run_experiment()
