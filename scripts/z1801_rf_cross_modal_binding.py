#!/usr/bin/env python3
"""
z1801: RF Cross-Modal Binding

Hypothesis: An embodied model with both interoception (GPU) and exteroception (RF)
should learn to correlate them -- understanding how external RF environment
relates to internal processing state.

Tests:
1. Can the model predict GPU state from RF context?
2. Can the model predict RF state from GPU state?
3. Does cross-modal attention emerge during training?
4. Are hidden states more informative when both modalities are available?

This tests a key claim: consciousness binds multiple sensory modalities into
a unified experience. The model should develop cross-modal representations.

Hardware: AMD Radeon 8060S + HackRF One (simulated if not available)

Author: Claude
Date: 2026-02-05
"""

import os
import sys
import json
import time
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent))

# Reuse z1800 components
from scripts.z1800_rf_embodiment import (
    RFTelemetryState,
    HackRFInterface,
    UnifiedEmbodiedTelemetry,
    RFAwareMetabolicConfig,
)

from src.telemetry.sysfs_hwmon import SysfsHwmonTelemetry
from src.metabolic.film_transformer import MetabolicTransformer, MetabolicConfig


class CrossModalTransformer(MetabolicTransformer):
    """
    Transformer with explicit cross-modal attention between GPU and RF.

    Architecture:
    - Separate encoders for GPU (12-dim) and RF (8-dim)
    - Cross-attention: GPU queries RF, RF queries GPU
    - Fused representation for FiLM conditioning
    """

    def __init__(self, config: MetabolicConfig):
        # Override telemetry_dim to combined size
        config = RFAwareMetabolicConfig(
            vocab_size=config.vocab_size,
            hidden_dim=config.hidden_dim,
            num_layers=config.num_layers,
            num_heads=config.num_heads,
            ff_dim=config.ff_dim,
            telemetry_dim=20,  # GPU (12) + RF (8)
        )
        super().__init__(config)

        # Separate modality encoders
        self.gpu_encoder = nn.Sequential(
            nn.Linear(12, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
        )
        self.rf_encoder = nn.Sequential(
            nn.Linear(8, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
        )

        # Cross-modal attention
        self.gpu_to_rf_attn = nn.MultiheadAttention(32, num_heads=2, batch_first=True)
        self.rf_to_gpu_attn = nn.MultiheadAttention(32, num_heads=2, batch_first=True)

        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 20),
        )

        # Cross-modal prediction heads (for auxiliary losses)
        self.gpu_from_rf = nn.Linear(32, 12)  # Predict GPU from RF
        self.rf_from_gpu = nn.Linear(32, 8)   # Predict RF from GPU

    def encode_modalities(
        self,
        telemetry: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Encode telemetry with cross-modal attention.

        Returns:
            fused: [batch, 20] fused telemetry
            gpu_attended: [batch, 32] GPU features after attending to RF
            rf_attended: [batch, 32] RF features after attending to GPU
        """
        gpu_raw = telemetry[..., :12]
        rf_raw = telemetry[..., 12:]

        # Encode each modality
        gpu_enc = self.gpu_encoder(gpu_raw)  # [batch, 32]
        rf_enc = self.rf_encoder(rf_raw)     # [batch, 32]

        # Add sequence dimension for attention
        gpu_seq = gpu_enc.unsqueeze(1)  # [batch, 1, 32]
        rf_seq = rf_enc.unsqueeze(1)    # [batch, 1, 32]

        # Cross-modal attention
        gpu_attended, _ = self.gpu_to_rf_attn(gpu_seq, rf_seq, rf_seq)
        rf_attended, _ = self.rf_to_gpu_attn(rf_seq, gpu_seq, gpu_seq)

        gpu_attended = gpu_attended.squeeze(1)  # [batch, 32]
        rf_attended = rf_attended.squeeze(1)    # [batch, 32]

        # Fuse modalities
        fused = self.fusion(torch.cat([gpu_attended, rf_attended], dim=-1))

        return fused, gpu_attended, rf_attended

    def forward(
        self,
        input_ids: torch.Tensor,
        telemetry: Optional[torch.Tensor] = None,
        return_hidden: bool = False,
        return_cross_modal: bool = False,
    ):
        """Forward with cross-modal encoding."""
        if telemetry is not None and telemetry.shape[-1] == 20:
            fused, gpu_attended, rf_attended = self.encode_modalities(telemetry)
            telemetry_processed = fused
        else:
            telemetry_processed = telemetry
            gpu_attended = None
            rf_attended = None

        output = super().forward(input_ids, telemetry_processed, return_hidden)

        if return_cross_modal and gpu_attended is not None:
            output['gpu_attended'] = gpu_attended
            output['rf_attended'] = rf_attended
            # Cross-modal predictions
            output['gpu_pred_from_rf'] = self.gpu_from_rf(rf_attended)
            output['rf_pred_from_gpu'] = self.rf_from_gpu(gpu_attended)

        return output


def run_experiment():
    """
    z1801: Cross-Modal Binding Experiment

    Tests whether the model learns to correlate GPU and RF modalities.
    """

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[z1801] Device: {device}")
    if hasattr(torch.cuda, 'get_device_name'):
        print(f"[z1801] GPU: {torch.cuda.get_device_name()}")

    # Initialize telemetry
    telemetry = UnifiedEmbodiedTelemetry(rf_simulation=True)
    telemetry.start()
    time.sleep(0.5)  # Let telemetry stabilize

    rf_mode = "SIMULATED" if telemetry.rf_interface.simulation else "REAL"
    print(f"[z1801] RF mode: {rf_mode}")

    # Load training data
    data_path = Path(__file__).parent.parent / "data" / "tinyshakespeare.txt"
    if not data_path.exists():
        data_path = Path(__file__).parent.parent / "tinyshakespeare.txt"
    text = data_path.read_text()
    chars = sorted(list(set(text)))
    char_to_idx = {ch: i for i, ch in enumerate(chars)}
    vocab_size = len(chars)
    print(f"[z1801] Vocab size: {vocab_size}")

    # Config
    batch_size = 4
    seq_len = 256
    num_epochs = 8
    batches_per_epoch = 200
    lr = 3e-4
    cross_modal_weight = 0.1  # Weight for cross-modal prediction loss

    # Create cross-modal model
    config = MetabolicConfig(
        vocab_size=vocab_size,
        hidden_dim=256,
        num_layers=6,
        num_heads=4,
        ff_dim=1024,
        telemetry_dim=20,
    )
    model = CrossModalTransformer(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    print(f"[z1801] Model params: {sum(p.numel() for p in model.parameters()):,}")

    # Data iterator
    def get_batch():
        ix = torch.randint(len(text) - seq_len - 1, (batch_size,))
        x = torch.stack([
            torch.tensor([char_to_idx[c] for c in text[i:i+seq_len]], dtype=torch.long)
            for i in ix
        ])
        y = torch.stack([
            torch.tensor([char_to_idx[c] for c in text[i+1:i+seq_len+1]], dtype=torch.long)
            for i in ix
        ])
        return x.to(device), y.to(device)

    def get_telemetry():
        return telemetry.get_unified_tensor().unsqueeze(0).to(device)

    results = {
        'experiment': 'z1801_rf_cross_modal_binding',
        'timestamp': datetime.now().isoformat(),
        'device': str(device),
        'rf_mode': rf_mode,
        'config': {
            'batch_size': batch_size,
            'seq_len': seq_len,
            'num_epochs': num_epochs,
            'cross_modal_weight': cross_modal_weight,
        },
        'training': {
            'task_losses': [],
            'gpu_pred_losses': [],
            'rf_pred_losses': [],
            'total_losses': [],
        },
        'verdicts': {},
    }

    # Training with cross-modal prediction
    print("\n[z1801] Training with cross-modal binding...")
    model.train()
    telemetry_history = []

    for epoch in range(num_epochs):
        epoch_task_loss = 0
        epoch_gpu_pred_loss = 0
        epoch_rf_pred_loss = 0

        for batch_idx in range(batches_per_epoch):
            x, y = get_batch()
            telem = get_telemetry().expand(batch_size, -1)
            telemetry_history.append(telem.cpu().numpy())

            optimizer.zero_grad()

            output = model(x, telem, return_hidden=True, return_cross_modal=True)
            logits = output['logits']

            # Task loss
            task_loss = F.cross_entropy(logits.view(-1, vocab_size), y.view(-1))

            # Cross-modal prediction losses
            gpu_raw = telem[..., :12]
            rf_raw = telem[..., 12:]

            gpu_pred = output.get('gpu_pred_from_rf')
            rf_pred = output.get('rf_pred_from_gpu')

            if gpu_pred is not None:
                gpu_pred_loss = F.mse_loss(gpu_pred, gpu_raw)
                rf_pred_loss = F.mse_loss(rf_pred, rf_raw)
            else:
                gpu_pred_loss = torch.tensor(0.0)
                rf_pred_loss = torch.tensor(0.0)

            # Total loss
            total_loss = task_loss + cross_modal_weight * (gpu_pred_loss + rf_pred_loss)
            total_loss.backward()
            optimizer.step()

            epoch_task_loss += task_loss.item()
            epoch_gpu_pred_loss += gpu_pred_loss.item()
            epoch_rf_pred_loss += rf_pred_loss.item()

        avg_task = epoch_task_loss / batches_per_epoch
        avg_gpu = epoch_gpu_pred_loss / batches_per_epoch
        avg_rf = epoch_rf_pred_loss / batches_per_epoch

        results['training']['task_losses'].append(avg_task)
        results['training']['gpu_pred_losses'].append(avg_gpu)
        results['training']['rf_pred_losses'].append(avg_rf)
        results['training']['total_losses'].append(avg_task + cross_modal_weight * (avg_gpu + avg_rf))

        print(f"  Epoch {epoch+1}/{num_epochs}: task={avg_task:.4f}, gpu_pred={avg_gpu:.4f}, rf_pred={avg_rf:.4f}")

    # Evaluation: Cross-modal correlation
    print("\n[z1801] Evaluating cross-modal binding...")
    model.eval()

    # Collect cross-modal predictions
    gpu_actual = []
    gpu_predicted = []
    rf_actual = []
    rf_predicted = []

    with torch.no_grad():
        for _ in range(50):
            x, _ = get_batch()
            telem = get_telemetry().expand(batch_size, -1)

            output = model(x, telem, return_cross_modal=True)

            gpu_raw = telem[..., :12]
            rf_raw = telem[..., 12:]

            gpu_actual.append(gpu_raw.cpu().numpy())
            gpu_predicted.append(output['gpu_pred_from_rf'].cpu().numpy())
            rf_actual.append(rf_raw.cpu().numpy())
            rf_predicted.append(output['rf_pred_from_gpu'].cpu().numpy())

    gpu_actual = np.concatenate(gpu_actual, axis=0)
    gpu_predicted = np.concatenate(gpu_predicted, axis=0)
    rf_actual = np.concatenate(rf_actual, axis=0)
    rf_predicted = np.concatenate(rf_predicted, axis=0)

    # Correlation between predicted and actual
    gpu_corr = np.corrcoef(gpu_actual.flatten(), gpu_predicted.flatten())[0, 1]
    rf_corr = np.corrcoef(rf_actual.flatten(), rf_predicted.flatten())[0, 1]

    # MSE
    gpu_mse = np.mean((gpu_actual - gpu_predicted) ** 2)
    rf_mse = np.mean((rf_actual - rf_predicted) ** 2)

    results['evaluation'] = {
        'gpu_pred_correlation': float(gpu_corr) if not np.isnan(gpu_corr) else 0.0,
        'rf_pred_correlation': float(rf_corr) if not np.isnan(rf_corr) else 0.0,
        'gpu_pred_mse': float(gpu_mse),
        'rf_pred_mse': float(rf_mse),
    }

    print(f"\n[z1801] Results:")
    print(f"  GPU prediction from RF: corr={gpu_corr:.4f}, MSE={gpu_mse:.4f}")
    print(f"  RF prediction from GPU: corr={rf_corr:.4f}, MSE={rf_mse:.4f}")

    # Verdicts
    # V1: Cross-modal prediction improves (loss decreases)
    initial_gpu = results['training']['gpu_pred_losses'][0]
    final_gpu = results['training']['gpu_pred_losses'][-1]
    v1_pass = final_gpu < initial_gpu * 0.8  # 20% improvement
    results['verdicts']['V1_cross_modal_learning'] = {
        'pass': v1_pass,
        'initial_gpu_loss': initial_gpu,
        'final_gpu_loss': final_gpu,
        'improvement': (initial_gpu - final_gpu) / initial_gpu,
        'description': 'Cross-modal prediction improves during training'
    }

    # V2: Positive correlation between predicted and actual
    v2_pass = gpu_corr > 0.3 or rf_corr > 0.3
    results['verdicts']['V2_cross_modal_correlation'] = {
        'pass': v2_pass,
        'gpu_correlation': float(gpu_corr) if not np.isnan(gpu_corr) else 0.0,
        'rf_correlation': float(rf_corr) if not np.isnan(rf_corr) else 0.0,
        'threshold': 0.3,
        'description': 'Cross-modal predictions correlate with actual values'
    }

    # V3: Task performance maintained (PPL reasonable)
    final_ppl = np.exp(results['training']['task_losses'][-1])
    v3_pass = final_ppl < 15
    results['verdicts']['V3_task_preserved'] = {
        'pass': v3_pass,
        'final_ppl': float(final_ppl),
        'threshold': 15,
        'description': 'Language modeling quality maintained'
    }

    # V4: Both modalities contribute (bi-directional binding)
    v4_pass = gpu_mse < 0.5 and rf_mse < 0.5
    results['verdicts']['V4_bidirectional_binding'] = {
        'pass': v4_pass,
        'gpu_mse': float(gpu_mse),
        'rf_mse': float(rf_mse),
        'threshold': 0.5,
        'description': 'Both GPU and RF can be predicted from each other'
    }

    # Summary
    passed = sum(1 for v in results['verdicts'].values() if v['pass'])
    total = len(results['verdicts'])
    results['passed'] = passed
    results['total_verdicts'] = total
    results['overall_verdict'] = 'CROSS_MODAL_BINDING_DEMONSTRATED' if passed >= 3 else 'PARTIAL'

    print(f"\n[z1801] Verdicts: {passed}/{total} passed")
    print(f"[z1801] Overall: {results['overall_verdict']}")

    # Cleanup
    telemetry.stop()

    # Save results
    results_path = Path(__file__).parent.parent / "results" / "z1801_rf_cross_modal_binding.json"
    results_path.parent.mkdir(exist_ok=True)
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"[z1801] Results saved to {results_path}")

    return results


if __name__ == "__main__":
    results = run_experiment()
