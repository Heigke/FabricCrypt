#!/usr/bin/env python3
"""
z1905: Telemetry-Gated Prediction

KEY INSIGHT FROM z1904: Model can SENSE hardware but doesn't USE it for the task.
Hardware prediction error was 0.00018 (excellent!) but outputs were unchanged.

SOLUTION: Make task performance DEPEND on hardware prediction accuracy.
- Randomly mask some positions in the output
- The mask is revealed ONLY if hardware prediction is accurate
- Model must predict hardware correctly to get full loss signal

This is like: "You can only see your homework answers if you know your body temperature."
The task (homework) now REQUIRES body awareness.

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


class TelemetryGatedTransformer(nn.Module):
    """Transformer where prediction depends on hardware awareness."""

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

        # Embedding with telemetry injection
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.pos_embedding = nn.Parameter(torch.randn(1, 1024, hidden_dim) * 0.02)

        # Telemetry encoder - projects to hidden dim
        self.telem_encoder = nn.Sequential(
            nn.Linear(telemetry_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, hidden_dim),
        )

        # Transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)

        self.norm = nn.LayerNorm(hidden_dim)
        self.lm_head = nn.Linear(hidden_dim, vocab_size)

        # Hardware predictor (predicts telemetry from hidden state)
        self.hw_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, telemetry_dim),
        )

        # Confidence head: "how sure am I about my hardware prediction?"
        self.confidence_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.GELU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

        self.vocab_size = vocab_size

    def forward(self, input_ids: torch.Tensor, telemetry: torch.Tensor, return_all: bool = False):
        B, S = input_ids.shape

        # Embed
        x = self.embedding(input_ids) + self.pos_embedding[:, :S, :]

        # Inject telemetry
        if telemetry.dim() == 1:
            telemetry = telemetry.unsqueeze(0).expand(B, -1)

        telem_emb = self.telem_encoder(telemetry)
        x = x + telem_emb.unsqueeze(1)

        # Transform
        x = self.transformer(x)
        x = self.norm(x)

        # LM output
        logits = self.lm_head(x)

        # Hardware prediction
        hidden_mean = x.mean(dim=1)
        hw_pred = self.hw_predictor(hidden_mean)

        # Confidence in hardware prediction
        confidence = self.confidence_head(hidden_mean)

        if return_all:
            return {
                'logits': logits,
                'hw_prediction': hw_pred,
                'confidence': confidence,
                'hidden_mean': hidden_mean,
            }
        return logits

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def compute_telemetry_gated_loss(
    model_output: Dict,
    targets: torch.Tensor,
    telemetry: torch.Tensor,
    mask_ratio: float = 0.3,
) -> Tuple[torch.Tensor, Dict]:
    """
    Compute loss where hardware prediction accuracy GATES which tokens contribute to loss.

    1. Compute hardware prediction error
    2. Create a mask based on error threshold
    3. Only compute task loss on positions where hardware was predicted well
    4. This forces the model to predict hardware correctly to learn the task
    """
    B, S, V = model_output['logits'].shape

    # Hardware prediction error per batch element
    hw_pred = model_output['hw_prediction']
    if telemetry.dim() == 1:
        telemetry = telemetry.unsqueeze(0).expand(B, -1)
    hw_error = (hw_pred - telemetry).pow(2).mean(dim=-1)  # [B]

    # Compute task loss for all positions first
    logits = model_output['logits']
    task_loss_all = F.cross_entropy(logits.view(-1, V), targets.view(-1), reduction='none')
    task_loss_all = task_loss_all.view(B, S)  # [B, S]

    # Create mask: positions that "count" for learning
    # The mask threshold depends on hardware prediction error
    # Better hw prediction = more positions contribute = better learning

    # Normalize hw_error to [0, 1] range (roughly)
    hw_error_normalized = hw_error.clamp(0, 0.1) / 0.1  # Scale so 0.1 error = 1.0

    # Mask probability: good hw prediction = high mask probability (more positions count)
    # mask_prob = 1 - mask_ratio when hw_error is 0
    # mask_prob = mask_ratio when hw_error is high
    mask_prob = (1 - mask_ratio) * (1 - hw_error_normalized) + mask_ratio * hw_error_normalized

    # Generate random mask per batch element
    random_vals = torch.rand(B, S, device=targets.device)
    mask = (random_vals < mask_prob.unsqueeze(1).expand(-1, S)).float()  # [B, S]

    # Apply mask to task loss (only count masked positions)
    # CRITICAL: When hw_error is HIGH, fewer positions count, so learning is GATED
    masked_task_loss = (task_loss_all * mask).sum() / (mask.sum() + 1e-6)

    # Also include hardware prediction loss
    hw_loss = hw_error.mean()

    # Confidence calibration loss
    # The confidence should match whether hw prediction is accurate
    confidence = model_output['confidence'].squeeze(-1)  # [B]
    confidence_target = (hw_error < 0.01).float()  # 1 if hw_error < 0.01
    confidence_loss = F.binary_cross_entropy(confidence, confidence_target)

    total_loss = masked_task_loss + 0.5 * hw_loss + 0.1 * confidence_loss

    return total_loss, {
        'task_loss': masked_task_loss.item(),
        'hw_loss': hw_loss.item(),
        'confidence_loss': confidence_loss.item(),
        'avg_mask_prob': mask_prob.mean().item(),
        'avg_hw_error': hw_error.mean().item(),
        'avg_confidence': confidence.mean().item(),
    }


def compute_signature(model, x, telem, device):
    """Compute behavioral signature."""
    model.eval()
    with torch.no_grad():
        out = model(x, telem, return_all=True)
        logits = out['logits']
        hidden = out['hidden_mean']
        hw_pred = out['hw_prediction']
        conf = out['confidence']

        return {
            'logit_entropy': -(F.softmax(logits, -1) * F.log_softmax(logits, -1)).sum(-1).mean().item(),
            'hidden_norm': hidden.norm(dim=-1).mean().item(),
            'hw_pred_norm': hw_pred.norm(dim=-1).mean().item(),
            'confidence': conf.mean().item(),
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
    print(f"[z1905] Device: {device}")
    print("[z1905] TELEMETRY-GATED PREDICTION")
    print("[z1905] Task loss only counts where hardware is predicted correctly")

    # Telemetry
    telemetry = TriHardwareTelemetry()
    telemetry.start()
    time.sleep(1)

    hw_status = telemetry.get_hardware_status()
    print(f"[z1905] Hardware: GPU={hw_status['gpu']}, FPGA={hw_status['fpga']}, RF={hw_status['rf']}")

    # Data
    data_path = Path(__file__).parent.parent / "data" / "tinyshakespeare.txt"
    if not data_path.exists():
        data_path = Path(__file__).parent.parent / "tinyshakespeare.txt"
    text_bytes = data_path.read_text().encode('utf-8')

    # Model
    model = TelemetryGatedTransformer(
        vocab_size=256,
        hidden_dim=512,
        num_layers=8,
        num_heads=8,
        ff_dim=2048,
        telemetry_dim=20,
    ).to(device)

    print(f"[z1905] Model parameters: {model.count_parameters():,}")

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
        'experiment': 'z1905_telemetry_gated_prediction',
        'timestamp': datetime.now().isoformat(),
        'device': str(device),
        'hardware_status': hw_status,
        'architecture': 'TelemetryGated (hw accuracy gates task loss)',
    }

    # Training
    print("\n[z1905] Training with TELEMETRY-GATED LOSS...")
    print("[z1905] Better hardware prediction = more positions contribute to learning")
    telem_samples = []
    training_metrics = []

    for epoch in range(epochs):
        model.train()
        epoch_task = 0
        epoch_hw = 0
        epoch_conf = 0
        epoch_mask = 0

        for _ in range(batches_per_epoch):
            x, y = get_batch()
            telem = telemetry.get_tensor().to(device)
            telem_samples.append(telem.cpu().numpy())

            optimizer.zero_grad()
            out = model(x, telem, return_all=True)

            loss, metrics = compute_telemetry_gated_loss(out, y, telem)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_task += metrics['task_loss']
            epoch_hw += metrics['avg_hw_error']
            epoch_conf += metrics['avg_confidence']
            epoch_mask += metrics['avg_mask_prob']

        avg_task = epoch_task / batches_per_epoch
        avg_hw = epoch_hw / batches_per_epoch
        avg_conf = epoch_conf / batches_per_epoch
        avg_mask = epoch_mask / batches_per_epoch
        training_metrics.append({'task': avg_task, 'hw': avg_hw, 'conf': avg_conf, 'mask': avg_mask})
        print(f"  Epoch {epoch+1}/{epochs}: task={avg_task:.4f}, hw_err={avg_hw:.6f}, conf={avg_conf:.3f}, mask={avg_mask:.3f}")

    results['training'] = training_metrics
    telem_samples = np.array(telem_samples)

    # Reference signature
    print("\n[z1905] Computing reference signature...")
    x_test, _ = get_batch()
    real_telem = telemetry.get_tensor().to(device)
    real_sig = compute_signature(model, x_test, real_telem, device)
    print(f"  Real: {real_sig}")

    # Falsification tests
    print("\n" + "="*60)
    print("[z1905] FALSIFICATION BATTERY")
    print("="*60)

    tests = {}

    # T1: Zero telemetry
    print("\n[z1905] T1: Zero Telemetry")
    zero_sig = compute_signature(model, x_test, torch.zeros(20, device=device), device)
    d1 = signature_distance(real_sig, zero_sig)
    print(f"  Distance: {d1:.4f}")
    tests['T1_zero'] = {'distance': d1, 'falsified': d1 < 0.05}

    # T2: Random telemetry
    print("\n[z1905] T2: Random Telemetry")
    d2_list = [signature_distance(real_sig, compute_signature(model, x_test, torch.rand(20, device=device), device)) for _ in range(10)]
    d2 = np.mean(d2_list)
    print(f"  Avg distance: {d2:.4f}")
    tests['T2_random'] = {'distance': d2, 'falsified': d2 < 0.05}

    # T3: Historical
    print("\n[z1905] T3: Historical Telemetry")
    old_telem = torch.tensor(telem_samples[0], device=device)
    d3 = signature_distance(real_sig, compute_signature(model, x_test, old_telem, device))
    print(f"  Distance: {d3:.4f}")
    tests['T3_historical'] = {'distance': d3, 'falsified': d3 < 0.03}

    # T4: Constant
    print("\n[z1905] T4: Constant Telemetry")
    mean_telem = torch.tensor(telem_samples.mean(0), device=device)
    d4 = signature_distance(real_sig, compute_signature(model, x_test, mean_telem, device))
    print(f"  Distance: {d4:.4f}")
    tests['T4_constant'] = {'distance': d4, 'falsified': d4 < 0.03}

    # T5: Inverted
    print("\n[z1905] T5: Inverted Telemetry")
    inv_telem = 1.0 - real_telem.clamp(0, 1)
    d5 = signature_distance(real_sig, compute_signature(model, x_test, inv_telem, device))
    print(f"  Distance: {d5:.4f}")
    tests['T5_inverted'] = {'distance': d5, 'falsified': d5 < 0.05}

    # T6: Confidence calibration
    print("\n[z1905] T6: Confidence Calibration")
    model.eval()
    with torch.no_grad():
        out_real = model(x_test, real_telem, return_all=True)
        conf_real = out_real['confidence'].mean().item()

        out_zero = model(x_test, torch.zeros(20, device=device), return_all=True)
        conf_zero = out_zero['confidence'].mean().item()

        out_rand = model(x_test, torch.rand(20, device=device), return_all=True)
        conf_rand = out_rand['confidence'].mean().item()

    print(f"  Confidence with real telem: {conf_real:.4f}")
    print(f"  Confidence with zero telem: {conf_zero:.4f}")
    print(f"  Confidence with rand telem: {conf_rand:.4f}")

    # If calibrated, confidence should be HIGH for real, LOW for zero/random
    tests['T6_confidence_calibration'] = {
        'conf_real': conf_real,
        'conf_zero': conf_zero,
        'conf_rand': conf_rand,
        'falsified': conf_real < conf_zero * 1.5,  # Real should be >50% higher than zero
    }

    # T7: Prediction quality degradation
    print("\n[z1905] T7: Task Prediction Quality")
    model.eval()
    x_test2, y_test = get_batch()
    with torch.no_grad():
        out_real = model(x_test2, real_telem, return_all=True)
        loss_real = F.cross_entropy(out_real['logits'].view(-1, 256), y_test.view(-1)).item()

        out_zero = model(x_test2, torch.zeros(20, device=device), return_all=True)
        loss_zero = F.cross_entropy(out_zero['logits'].view(-1, 256), y_test.view(-1)).item()

    print(f"  Loss with real telemetry: {loss_real:.4f}")
    print(f"  Loss with zero telemetry: {loss_zero:.4f}")
    degradation = (loss_zero - loss_real) / loss_real * 100
    print(f"  Degradation: {degradation:.2f}%")

    tests['T7_task_degradation'] = {
        'loss_real': loss_real,
        'loss_zero': loss_zero,
        'degradation_pct': degradation,
        'falsified': degradation < 5.0,  # Should be >5% worse without real telemetry
    }

    results['falsification_tests'] = tests

    num_falsified = sum(1 for t in tests.values() if t['falsified'])
    results['num_falsified'] = num_falsified
    results['num_total'] = len(tests)
    results['status'] = 'FALSIFIED' if num_falsified > 0 else 'SURVIVED ALL TESTS'

    print(f"\n{'='*60}")
    print(f"[z1905] RESULTS")
    print(f"{'='*60}")
    for name, t in tests.items():
        status = "FALSIFIED" if t['falsified'] else "SURVIVED"
        print(f"  {status} {name}")

    print(f"\n[z1905] Tests survived: {len(tests) - num_falsified}/{len(tests)}")
    print(f"[z1905] Status: {results['status']}")

    if num_falsified == 0:
        print("\n[z1905] ALL TESTS SURVIVED!")
        print("[z1905] Model exhibits TRUE hardware dependence")

    telemetry.stop()

    # Save
    results_path = Path(__file__).parent.parent / "results" / "z1905_telemetry_gated_prediction.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n[z1905] Results saved to {results_path}")

    return results


if __name__ == "__main__":
    run_experiment()
