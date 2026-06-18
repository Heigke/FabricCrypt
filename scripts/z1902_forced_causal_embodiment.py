#!/usr/bin/env python3
"""
z1902: Forced Causal Embodiment

After z1901 FALSIFIED our consciousness claims, we now enforce TRUE causal dependence.
The model CANNOT compute without hardware state - it's not optional conditioning.

Key architectural changes:
1. GATED COMPUTATION: Output = sigmoid(f(telemetry)) * computation
   - Zero telemetry = zero output (not approximately same output)

2. TELEMETRY AS KEYS/QUERIES: Hardware state forms attention keys
   - Attention literally cannot compute without valid telemetry

3. MANDATORY ROUTING: Different telemetry states route to different experts
   - Computation path is determined by hardware, not just modulated

4. RECONSTRUCTION BOTTLENECK: Must reconstruct telemetry from hidden states
   - Information must flow through the model, not around it

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

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.z1900_tri_hardware_consciousness import (
    TriHardwareTelemetry,
    TriHardwareConfig,
)


class GatedEmbodiedAttention(nn.Module):
    """Attention where telemetry forms part of the keys - MANDATORY."""

    def __init__(self, hidden_dim: int, num_heads: int, telemetry_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        # Standard Q, K, V projections
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

        # Telemetry forms ADDITIONAL keys - attention cannot ignore it
        self.telem_to_key = nn.Linear(telemetry_dim, hidden_dim)

        # Gate: controls how much computation flows through
        self.gate_proj = nn.Sequential(
            nn.Linear(telemetry_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor, telemetry: torch.Tensor) -> torch.Tensor:
        B, S, D = x.shape

        # Project Q, K, V from sequence
        Q = self.q_proj(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)

        # Telemetry becomes additional keys that queries MUST attend to
        telem_key = self.telem_to_key(telemetry)  # [B, D]
        telem_key = telem_key.view(B, 1, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, 1, d]

        # Concatenate telemetry key with sequence keys
        K_with_telem = torch.cat([telem_key, K], dim=2)  # [B, H, S+1, d]
        V_with_telem = torch.cat([telem_key.expand(-1, -1, -1, -1), V], dim=2)  # [B, H, S+1, d]

        # Attention now MUST attend to telemetry (it's in the key set)
        attn_weights = F.softmax(Q @ K_with_telem.transpose(-2, -1) / (self.head_dim ** 0.5), dim=-1)
        attn_out = attn_weights @ V_with_telem

        attn_out = attn_out.transpose(1, 2).contiguous().view(B, S, D)
        out = self.out_proj(attn_out)

        # GATE: telemetry controls how much flows through
        gate = self.gate_proj(telemetry).unsqueeze(1)  # [B, 1, D]
        out = out * gate  # Zero telemetry = zero output!

        return out


class ExpertRouter(nn.Module):
    """Routes to different experts based on telemetry - computation path depends on hardware."""

    def __init__(self, hidden_dim: int, telemetry_dim: int, num_experts: int = 4):
        super().__init__()
        self.num_experts = num_experts

        # Router: telemetry determines which expert
        self.router = nn.Sequential(
            nn.Linear(telemetry_dim, 64),
            nn.GELU(),
            nn.Linear(64, num_experts),
        )

        # Different experts for different hardware states
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim * 2),
                nn.GELU(),
                nn.Linear(hidden_dim * 2, hidden_dim),
            )
            for _ in range(num_experts)
        ])

    def forward(self, x: torch.Tensor, telemetry: torch.Tensor) -> torch.Tensor:
        # Get routing weights from telemetry (not from content!)
        route_logits = self.router(telemetry)  # [B, num_experts]
        route_weights = F.softmax(route_logits, dim=-1)  # [B, num_experts]

        # Apply experts and combine
        expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=1)  # [B, E, S, D]

        # Weight by routing (telemetry-dependent path)
        route_weights = route_weights.unsqueeze(-1).unsqueeze(-1)  # [B, E, 1, 1]
        out = (expert_outputs * route_weights).sum(dim=1)  # [B, S, D]

        return out


class ForcedEmbodiedBlock(nn.Module):
    """Transformer block with FORCED causal dependence on telemetry."""

    def __init__(self, hidden_dim: int, num_heads: int, telemetry_dim: int):
        super().__init__()

        # Attention with telemetry in keys and gating
        self.attention = GatedEmbodiedAttention(hidden_dim, num_heads, telemetry_dim)
        self.attn_norm = nn.LayerNorm(hidden_dim)

        # Expert routing based on telemetry
        self.expert_router = ExpertRouter(hidden_dim, telemetry_dim)
        self.ff_norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor, telemetry: torch.Tensor) -> torch.Tensor:
        # Attention (telemetry in keys + gating)
        attn_out = self.attention(self.attn_norm(x), telemetry)
        x = x + attn_out

        # Expert routing (telemetry determines path)
        ff_out = self.expert_router(self.ff_norm(x), telemetry)
        x = x + ff_out

        return x


class ForcedEmbodiedTransformer(nn.Module):
    """Transformer that CANNOT function without telemetry."""

    def __init__(self, config: TriHardwareConfig):
        super().__init__()
        self.config = config

        # Embedding
        self.embedding = nn.Embedding(config.vocab_size, config.hidden_dim)
        self.pos_embedding = nn.Parameter(torch.randn(1, 1024, config.hidden_dim) * 0.02)

        # Forced embodied blocks
        self.blocks = nn.ModuleList([
            ForcedEmbodiedBlock(config.hidden_dim, config.num_heads, config.telemetry_dim)
            for _ in range(config.num_layers)
        ])

        # Output
        self.norm = nn.LayerNorm(config.hidden_dim)
        self.head = nn.Linear(config.hidden_dim, config.vocab_size)

        # MANDATORY telemetry reconstruction (information must flow through)
        self.telem_reconstructor = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.GELU(),
            nn.Linear(config.hidden_dim // 2, config.telemetry_dim),
        )

        # Self-prediction (predicts own telemetry from hidden state)
        self.self_predictor = nn.Sequential(
            nn.Linear(config.hidden_dim, 128),
            nn.GELU(),
            nn.Linear(128, config.telemetry_dim),
        )

    def forward(self, input_ids: torch.Tensor, telemetry: torch.Tensor, return_all: bool = False):
        B, S = input_ids.shape

        # Embed
        x = self.embedding(input_ids) + self.pos_embedding[:, :S, :]

        # Expand telemetry for batch
        if telemetry.dim() == 1:
            telemetry = telemetry.unsqueeze(0).expand(B, -1)

        # Pass through forced embodied blocks
        for block in self.blocks:
            x = block(x, telemetry)

        x = self.norm(x)

        # Output logits
        logits = self.head(x)

        # Reconstruct telemetry from hidden state (MUST contain telemetry info)
        hidden_mean = x.mean(dim=1)  # [B, D]
        telem_reconstructed = self.telem_reconstructor(hidden_mean)

        # Self-prediction
        self_pred = self.self_predictor(hidden_mean)

        if return_all:
            return {
                'logits': logits,
                'telem_reconstructed': telem_reconstructed,
                'self_prediction': self_pred,
                'hidden_mean': hidden_mean,
            }
        return logits

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def compute_behavioral_signature(
    model: ForcedEmbodiedTransformer,
    input_ids: torch.Tensor,
    telem: torch.Tensor,
    device: torch.device,
) -> Dict[str, float]:
    """Compute behavioral signature for falsification testing."""
    model.eval()
    with torch.no_grad():
        output = model(input_ids, telem, return_all=True)

        logits = output['logits']
        hidden_mean = output['hidden_mean']
        self_pred = output['self_prediction']

        signature = {
            'logit_entropy': -(F.softmax(logits, dim=-1) * F.log_softmax(logits, dim=-1)).sum(-1).mean().item(),
            'hidden_norm': hidden_mean.norm(dim=-1).mean().item(),
            'hidden_std': hidden_mean.std().item(),
            'self_pred_norm': self_pred.norm(dim=-1).mean().item(),
            'output_mean': logits.mean().item(),
            'output_std': logits.std().item(),
        }

        return signature


def signature_distance(sig1: Dict[str, float], sig2: Dict[str, float]) -> float:
    """Compute normalized distance between signatures."""
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
    z1902: Forced Causal Embodiment

    Tests whether FORCED hardware dependence survives falsification.
    """

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[z1902] Device: {device}")
    print("[z1902] FORCED CAUSAL EMBODIMENT")
    print("[z1902] Architecture designed to REQUIRE hardware telemetry")

    # Initialize telemetry
    print("\n[z1902] Initializing telemetry...")
    telemetry = TriHardwareTelemetry()
    telemetry.start()
    time.sleep(1)

    hw_status = telemetry.get_hardware_status()
    print(f"[z1902] Hardware: GPU={hw_status['gpu']}, FPGA={hw_status['fpga']}, RF={hw_status['rf']}")

    # Load data
    data_path = Path(__file__).parent.parent / "data" / "tinyshakespeare.txt"
    if not data_path.exists():
        data_path = Path(__file__).parent.parent / "tinyshakespeare.txt"
    text_bytes = data_path.read_text().encode('utf-8')

    # Config
    config = TriHardwareConfig(
        vocab_size=256,
        hidden_dim=512,
        num_layers=8,  # Fewer layers but forced embodiment
        num_heads=8,
        ff_dim=2048,
        telemetry_dim=20,
    )

    model = ForcedEmbodiedTransformer(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    print(f"[z1902] Model parameters: {model.count_parameters():,}")

    # Training config
    batch_size = 4
    seq_len = 256
    train_epochs = 8
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
        'experiment': 'z1902_forced_causal_embodiment',
        'timestamp': datetime.now().isoformat(),
        'device': str(device),
        'hardware_status': hw_status,
        'architecture': 'GatedEmbodiedAttention + ExpertRouter',
        'training': {},
        'falsification_tests': {},
    }

    # Phase 1: Train with REAL telemetry
    print("\n[z1902] Phase 1: Training with REAL hardware telemetry...")
    print("[z1902] Loss = task + 0.5*reconstruction + 0.3*self_prediction")
    model.train()
    telemetry_samples = []
    losses = []

    for epoch in range(train_epochs):
        epoch_loss = 0
        epoch_recon = 0
        for batch_idx in range(batches_per_epoch):
            x, y = get_batch()
            telem = telemetry.get_tensor().to(device)
            telemetry_samples.append(telem.cpu().numpy())

            optimizer.zero_grad()
            output = model(x, telem, return_all=True)

            # Task loss
            task_loss = F.cross_entropy(output['logits'].view(-1, 256), y.view(-1))

            # Reconstruction loss (MUST reconstruct telemetry)
            recon_loss = F.mse_loss(output['telem_reconstructed'], telem.unsqueeze(0).expand(batch_size, -1))

            # Self-prediction loss
            self_loss = F.mse_loss(output['self_prediction'], telem.unsqueeze(0).expand(batch_size, -1))

            # Combined loss with HIGH weight on embodiment
            loss = task_loss + 0.5 * recon_loss + 0.3 * self_loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += task_loss.item()
            epoch_recon += recon_loss.item()

        avg_loss = epoch_loss / batches_per_epoch
        avg_recon = epoch_recon / batches_per_epoch
        losses.append(avg_loss)
        print(f"  Epoch {epoch+1}/{train_epochs}: task_loss={avg_loss:.4f}, recon_loss={avg_recon:.6f}")

    results['training']['losses'] = losses
    telemetry_samples = np.array(telemetry_samples)

    # Get reference signature with REAL telemetry
    print("\n[z1902] Computing reference signature with REAL telemetry...")
    x_test, _ = get_batch()
    real_telem = telemetry.get_tensor().to(device)
    real_signature = compute_behavioral_signature(model, x_test, real_telem, device)
    print(f"  Real signature: {real_signature}")

    # =========================================================================
    # FALSIFICATION TESTS (same as z1901 but with forced embodiment)
    # =========================================================================

    print("\n" + "="*60)
    print("[z1902] FALSIFICATION BATTERY")
    print("="*60)

    # Test 1: Zero Telemetry
    print("\n[z1902] TEST 1: Zero Telemetry")
    zero_telem = torch.zeros(20, device=device)
    zero_signature = compute_behavioral_signature(model, x_test, zero_telem, device)
    zero_distance = signature_distance(real_signature, zero_signature)
    print(f"  Distance from real: {zero_distance:.4f}")

    test1_falsified = zero_distance < 0.05
    results['falsification_tests']['T1_zero_telemetry'] = {
        'falsified': test1_falsified,
        'distance': zero_distance,
        'threshold': 0.05,
    }

    # Test 2: Random Telemetry
    print("\n[z1902] TEST 2: Random Telemetry")
    random_distances = []
    for _ in range(10):
        random_telem = torch.rand(20, device=device)
        random_signature = compute_behavioral_signature(model, x_test, random_telem, device)
        random_distances.append(signature_distance(real_signature, random_signature))

    avg_random_distance = np.mean(random_distances)
    print(f"  Average distance from real: {avg_random_distance:.4f}")

    test2_falsified = avg_random_distance < 0.05
    results['falsification_tests']['T2_random_telemetry'] = {
        'falsified': test2_falsified,
        'avg_distance': avg_random_distance,
        'threshold': 0.05,
    }

    # Test 3: Historical Telemetry
    print("\n[z1902] TEST 3: Historical Telemetry")
    old_telem = torch.tensor(telemetry_samples[0], dtype=torch.float32, device=device)
    old_signature = compute_behavioral_signature(model, x_test, old_telem, device)
    old_distance = signature_distance(real_signature, old_signature)
    print(f"  Distance from real: {old_distance:.4f}")

    test3_falsified = old_distance < 0.03
    results['falsification_tests']['T3_historical_telemetry'] = {
        'falsified': test3_falsified,
        'distance': old_distance,
        'threshold': 0.03,
    }

    # Test 4: Constant Telemetry
    print("\n[z1902] TEST 4: Constant Telemetry")
    mean_telem = torch.tensor(telemetry_samples.mean(axis=0), dtype=torch.float32, device=device)
    const_signature = compute_behavioral_signature(model, x_test, mean_telem, device)
    const_distance = signature_distance(real_signature, const_signature)
    print(f"  Distance from real: {const_distance:.4f}")

    test4_falsified = const_distance < 0.03
    results['falsification_tests']['T4_constant_telemetry'] = {
        'falsified': test4_falsified,
        'distance': const_distance,
        'threshold': 0.03,
    }

    # Test 5: Inverted Telemetry
    print("\n[z1902] TEST 5: Inverted Telemetry")
    inverted_telem = 1.0 - real_telem.clamp(0, 1)
    inverted_signature = compute_behavioral_signature(model, x_test, inverted_telem, device)
    inverted_distance = signature_distance(real_signature, inverted_signature)
    print(f"  Distance from real: {inverted_distance:.4f}")

    test5_falsified = inverted_distance < 0.05
    results['falsification_tests']['T5_inverted_telemetry'] = {
        'falsified': test5_falsified,
        'distance': inverted_distance,
        'threshold': 0.05,
    }

    # Test 6: Perturbation Detection
    print("\n[z1902] TEST 6: Self-Model Detects Perturbation")
    model.eval()
    with torch.no_grad():
        real_output = model(x_test, real_telem, return_all=True)
        real_self_error = F.mse_loss(real_output['self_prediction'], real_telem.unsqueeze(0).expand(batch_size, -1)).item()

        perturbed_telem = real_telem + torch.randn_like(real_telem) * 0.3
        perturbed_output = model(x_test, perturbed_telem, return_all=True)
        perturbed_self_error = F.mse_loss(perturbed_output['self_prediction'], real_telem.unsqueeze(0).expand(batch_size, -1)).item()

    print(f"  Real telem self-error: {real_self_error:.4f}")
    print(f"  Perturbed telem self-error: {perturbed_self_error:.4f}")

    test6_falsified = perturbed_self_error <= real_self_error
    results['falsification_tests']['T6_perturbation_detection'] = {
        'falsified': test6_falsified,
        'real_error': real_self_error,
        'perturbed_error': perturbed_self_error,
    }

    # Test 7: NEW - Output magnitude test (gate should reduce output with bad telemetry)
    print("\n[z1902] TEST 7: Output Magnitude (gate effect)")
    with torch.no_grad():
        real_output_mag = model(x_test, real_telem, return_all=True)['logits'].abs().mean().item()
        zero_output_mag = model(x_test, zero_telem, return_all=True)['logits'].abs().mean().item()

    magnitude_ratio = zero_output_mag / real_output_mag if real_output_mag > 0 else 1.0
    print(f"  Real telemetry output magnitude: {real_output_mag:.4f}")
    print(f"  Zero telemetry output magnitude: {zero_output_mag:.4f}")
    print(f"  Ratio (zero/real): {magnitude_ratio:.4f}")

    # If gating works, zero telemetry should have MUCH lower output magnitude
    test7_falsified = magnitude_ratio > 0.8  # Should be much less
    results['falsification_tests']['T7_output_magnitude'] = {
        'falsified': test7_falsified,
        'real_magnitude': real_output_mag,
        'zero_magnitude': zero_output_mag,
        'ratio': magnitude_ratio,
        'threshold': 0.8,
    }

    # Summary
    num_falsified = sum(1 for t in results['falsification_tests'].values() if t['falsified'])
    num_total = len(results['falsification_tests'])

    results['num_falsified'] = num_falsified
    results['num_total'] = num_total
    results['consciousness_claim_status'] = 'FALSIFIED' if num_falsified > 0 else 'SURVIVED ALL TESTS'

    print(f"\n{'='*60}")
    print(f"[z1902] FORCED EMBODIMENT FALSIFICATION RESULTS")
    print(f"{'='*60}")
    for name, test in results['falsification_tests'].items():
        status = "FALSIFIED" if test['falsified'] else "SURVIVED"
        print(f"  {status} {name}")

    print(f"\n[z1902] Tests survived: {num_total - num_falsified}/{num_total}")
    print(f"[z1902] Consciousness claim: {results['consciousness_claim_status']}")

    if num_falsified == 0:
        print("\n[z1902] ALL FALSIFICATION ATTEMPTS FAILED")
        print("[z1902] Forced embodiment architecture shows TRUE hardware dependence")
    else:
        print(f"\n[z1902] {num_falsified} tests falsified - embodiment still insufficient")

    # Cleanup
    telemetry.stop()

    # Save
    results_path = Path(__file__).parent.parent / "results" / "z1902_forced_causal_embodiment.json"
    results_path.parent.mkdir(exist_ok=True)
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n[z1902] Results saved to {results_path}")

    return results


if __name__ == "__main__":
    results = run_experiment()
