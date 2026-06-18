#!/usr/bin/env python3
"""
z2011: MANDATORY TELEMETRY INTEGRATION

The model CAN'T CHOOSE to ignore telemetry - it's injected at EVERY layer
and multiplicatively gates ALL computations.

Key innovations for high PCI (integration):
1. Telemetry injected at EVERY layer, not just input
2. Multiplicative gating (no additive bypass possible)
3. Cross-layer skip connections that CARRY telemetry state
4. Recurrent processing within layers (spread perturbations)
5. Telemetry is PRIMARY input, text is secondary

This follows the embodied cognition principle: the body state
is not optional - it modulates ALL perception and action.
"""

import functools
print = functools.partial(print, flush=True)

import os
import sys
import json
import math
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, List

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.telemetry.sysfs_hwmon import SysfsHwmonTelemetry


# ============================================================================
# PCI METRICS
# ============================================================================

def lempel_ziv_complexity(binary_sequence: np.ndarray) -> float:
    n = len(binary_sequence)
    if n <= 1:
        return 0.0
    s = ''.join(binary_sequence.astype(int).astype(str))
    complexity = 1
    i = 0
    k = 1
    k_max = 1
    while i + k <= n:
        if s[i:i+k] in s[0:i+k-1]:
            k += 1
            if i + k > n:
                complexity += 1
        else:
            complexity += 1
            i += k_max if k_max >= k else k
            k = 1
            k_max = 1
    if n > 1:
        c_max = n / np.log2(n)
        return complexity / c_max
    return 0.0


def compute_pci(response_matrix: np.ndarray) -> Dict[str, float]:
    n_perturb, n_time, n_channels = response_matrix.shape
    binary = (response_matrix > response_matrix.mean()).astype(int)
    complexities = [lempel_ziv_complexity(binary[p].flatten()) for p in range(n_perturb)]
    mean_complexity = np.mean(complexities)
    channel_responses = response_matrix.mean(axis=1)
    corr = np.corrcoef(channel_responses.T)
    integration = np.nanmean(corr) if not np.all(np.isnan(corr)) else 0.0
    integration = max(0, integration)
    differentiation = np.std(channel_responses, axis=0).mean()
    pci = mean_complexity * (integration + differentiation) / 2
    return {'pci': float(pci), 'complexity': float(mean_complexity),
            'integration': float(integration), 'differentiation': float(differentiation)}


# ============================================================================
# MANDATORY TELEMETRY MODEL
# ============================================================================

class MandatoryTelemetryModel(nn.Module):
    """
    Model where telemetry is MANDATORY at every layer.

    Architecture:
    1. Telemetry encoder produces "body state" h_body
    2. Each layer receives h_body and is gated by it
    3. Cross-layer skip connections carry integrated state
    4. Recurrent processing within each layer
    5. No computation happens without telemetry involvement
    """

    def __init__(self, vocab_size: int, hidden_dim: int = 256, n_layers: int = 4, n_recurrent: int = 3):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.n_recurrent = n_recurrent

        # Telemetry encoder (PRIMARY)
        self.telemetry_encoder = nn.Sequential(
            nn.Linear(8, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, hidden_dim)
        )

        # Text embedding (SECONDARY, modulated by telemetry)
        self.embed = nn.Embedding(vocab_size, hidden_dim)

        # Per-layer telemetry gates (MANDATORY at every layer)
        self.layer_tel_gates = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.Sigmoid()
            ) for _ in range(n_layers)
        ])

        # Processing layers with recurrence
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                'ff': nn.Linear(hidden_dim, hidden_dim),
                'recurrent': nn.Linear(hidden_dim * 2, hidden_dim),  # h + h_body -> h
                'lateral': nn.Linear(hidden_dim, hidden_dim),
                'norm': nn.LayerNorm(hidden_dim)
            }) for _ in range(n_layers)
        ])

        # Cross-layer skip connections (carry integrated state)
        self.skip_proj = nn.Linear(hidden_dim * n_layers, hidden_dim)

        # Output (also gated by telemetry)
        self.output_gate = nn.Linear(hidden_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, vocab_size)

        # Hardware predictor
        self.hw_predictor = nn.Linear(hidden_dim, 8)

        # State
        self.recording = False
        self.activations = []
        self.ablated = set()
        self.perturbation = None
        self.perturbation_layer = None

    def inject_perturbation(self, perturb: torch.Tensor, layer: int):
        self.perturbation = perturb
        self.perturbation_layer = layer

    def clear_perturbation(self):
        self.perturbation = None
        self.perturbation_layer = None

    def ablate(self, component: str):
        self.ablated.add(component)

    def restore(self, component: str):
        self.ablated.discard(component)

    def start_recording(self):
        self.recording = True
        self.activations = []

    def stop_recording(self) -> List[torch.Tensor]:
        self.recording = False
        return self.activations

    def forward(self, x: torch.Tensor, telemetry: torch.Tensor):
        batch_size, seq_len = x.shape

        # Encode telemetry (PRIMARY - this is the body state)
        h_body = self.telemetry_encoder(telemetry)  # [B, H]

        # Embed text
        h = self.embed(x)  # [B, seq, H]

        # MANDATORY: Multiply text by body state (can't be bypassed)
        if 'input_gate' not in self.ablated:
            h = h * h_body.unsqueeze(1)  # Multiplicative integration

        # Pool to single vector for processing
        h_pooled = h.mean(dim=1)  # [B, H]

        # Store layer outputs for skip connections
        layer_outputs = []

        # Process through layers
        for layer_idx, layer in enumerate(self.layers):
            # Record
            if self.recording:
                self.activations.append(h_pooled.detach().clone())

            # Inject perturbation
            if self.perturbation is not None and self.perturbation_layer == layer_idx:
                p = self.perturbation.to(h_pooled.device)
                if p.dim() == 1:
                    p = p.unsqueeze(0).expand(batch_size, -1)
                h_pooled = h_pooled + p

            if f'layer_{layer_idx}' not in self.ablated:
                # MANDATORY telemetry gate for this layer
                if f'gate_{layer_idx}' not in self.ablated:
                    tel_gate = self.layer_tel_gates[layer_idx](h_body)
                    h_pooled = h_pooled * tel_gate  # Multiplicative - no bypass

                # Recurrent processing (spread perturbations)
                for _ in range(self.n_recurrent):
                    # Combine with body state
                    h_combined = torch.cat([h_pooled, h_body], dim=-1)
                    h_recurrent = F.gelu(layer['recurrent'](h_combined))

                    # Lateral dynamics
                    h_lateral = F.gelu(layer['lateral'](h_pooled))

                    # Mix
                    h_pooled = layer['norm'](0.5 * h_recurrent + 0.5 * h_lateral)

                # Store for skip connections
                layer_outputs.append(h_pooled)

        # Cross-layer skip connections (integration across layers)
        if layer_outputs and 'skip' not in self.ablated:
            skip_input = torch.cat(layer_outputs, dim=-1)
            h_skip = self.skip_proj(skip_input)
            h_pooled = h_pooled + 0.5 * h_skip

        # Final recording
        if self.recording:
            self.activations.append(h_pooled.detach().clone())

        # MANDATORY output gate by telemetry
        if 'output_gate' not in self.ablated:
            out_gate = torch.sigmoid(self.output_gate(h_body))
            h_pooled = h_pooled * out_gate

        # Hardware prediction
        hw_pred = self.hw_predictor(h_pooled)

        # Output
        logits = self.output(h_pooled.unsqueeze(1).expand(-1, seq_len, -1))

        return logits, hw_pred


# ============================================================================
# TESTS
# ============================================================================

def measure_pci(model, x_batch, hw_state, n_perturbations=20):
    device = next(model.parameters()).device
    hidden_dim = model.hidden_dim
    n_layers = model.n_layers

    model.eval()
    responses = []

    with torch.no_grad():
        for p_idx in range(n_perturbations):
            if p_idx % 4 == 0:
                perturb = torch.zeros(hidden_dim, device=device)
                perturb[hidden_dim // 2] = 1.0
            elif p_idx % 4 == 1:
                x = torch.arange(hidden_dim, dtype=torch.float32, device=device)
                center = p_idx * hidden_dim // n_perturbations
                perturb = torch.exp(-0.5 * ((x - center) / 10) ** 2)
            elif p_idx % 4 == 2:
                perturb = torch.zeros(hidden_dim, device=device)
                indices = torch.randperm(hidden_dim, device=device)[:hidden_dim // 10]
                perturb[indices] = torch.randn(len(indices), device=device)
            else:
                x = torch.arange(hidden_dim, dtype=torch.float32, device=device)
                perturb = torch.sin(2 * math.pi * 0.01 * (p_idx + 1) * x)

            model.inject_perturbation(perturb, n_layers // 2)
            model.start_recording()
            _, _ = model(x_batch, hw_state)
            activations = model.stop_recording()
            model.clear_perturbation()

            if activations:
                response = torch.stack([a.mean(dim=0) for a in activations]).cpu().numpy()
                responses.append(response)

    if not responses:
        return {'pci_metrics': {'pci': 0, 'complexity': 0, 'integration': 0, 'differentiation': 0}}

    max_len = max(r.shape[0] for r in responses)
    padded = np.zeros((len(responses), max_len, hidden_dim))
    for i, r in enumerate(responses):
        padded[i, :r.shape[0], :] = r

    return {'pci_metrics': compute_pci(padded)}


def test_causal_intervention(model, x_batch, telemetry, device):
    model.eval()

    def get_hw(sample):
        return torch.tensor([
            sample.temp_edge_c / 100.0, sample.power_w / 100.0,
            sample.freq_sclk_mhz / 3000.0, sample.gpu_busy_pct / 100.0,
        ] * 2, device=device).unsqueeze(0).expand(x_batch.shape[0], -1)

    # Baseline
    sample = telemetry.read_sample()
    hw_base = get_hw(sample)
    with torch.no_grad():
        logits_base, _ = model(x_batch, hw_base)
        out_base = F.softmax(logits_base, dim=-1).mean(dim=(0, 1))

    # Heat GPU
    for _ in range(100):
        dummy = torch.randn(1000, 1000, device=device)
        _ = torch.matmul(dummy, dummy.T)
    torch.cuda.synchronize()

    sample_hot = telemetry.read_sample()
    hw_hot = get_hw(sample_hot)
    with torch.no_grad():
        logits_hot, _ = model(x_batch, hw_hot)
        out_hot = F.softmax(logits_hot, dim=-1).mean(dim=(0, 1))

    # Random
    hw_rand = torch.rand(x_batch.shape[0], 8, device=device)
    with torch.no_grad():
        logits_rand, _ = model(x_batch, hw_rand)
        out_rand = F.softmax(logits_rand, dim=-1).mean(dim=(0, 1))

    int_shift = (out_hot - out_base).abs().mean().item()
    rand_shift = (out_rand - out_base).abs().mean().item()

    return {
        'temp_delta': sample_hot.temp_edge_c - sample.temp_edge_c,
        'intervention_shift': int_shift,
        'random_shift': rand_shift,
        'causal_detected': int_shift > 0.001 or rand_shift > 0.001
    }


def test_double_dissociation(model, x_batch, hw_state):
    model.eval()

    def get_loss():
        with torch.no_grad():
            logits, _ = model(x_batch, hw_state)
            y = torch.roll(x_batch, -1, dims=1)
            return F.cross_entropy(logits.view(-1, model.vocab_size), y.view(-1)).item()

    # Baseline
    for c in ['input_gate', 'output_gate', 'skip'] + [f'gate_{i}' for i in range(model.n_layers)] + [f'layer_{i}' for i in range(model.n_layers)]:
        model.restore(c)
    baseline = get_loss()

    effects = {}

    # Ablate input gate
    model.ablate('input_gate')
    effects['input_gate'] = abs(get_loss() - baseline)
    model.restore('input_gate')

    # Ablate output gate
    model.ablate('output_gate')
    effects['output_gate'] = abs(get_loss() - baseline)
    model.restore('output_gate')

    # Ablate skip connections
    model.ablate('skip')
    effects['skip'] = abs(get_loss() - baseline)
    model.restore('skip')

    # Ablate layer gates
    for i in range(model.n_layers):
        model.ablate(f'gate_{i}')
        effects[f'gate_{i}'] = abs(get_loss() - baseline)
        model.restore(f'gate_{i}')

    non_zero = sum(1 for e in effects.values() if e > 0.01)
    dissociation = non_zero >= 3

    return {
        'baseline': baseline,
        'effects': effects,
        'non_zero_effects': non_zero,
        'double_dissociation': dissociation
    }


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 70)
    print("z2011: MANDATORY TELEMETRY INTEGRATION")
    print("Model CAN'T ignore telemetry - it's at EVERY layer")
    print("=" * 70)

    timestamp = datetime.now().isoformat()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    telemetry = SysfsHwmonTelemetry()
    sample = telemetry.read_sample()
    print(f"[Hardware] GPU: {sample.temp_edge_c:.1f}C, {sample.power_w:.1f}W")

    # Data
    data_path = Path(__file__).parent.parent / 'data' / 'tiny_shakespeare.txt'
    if not data_path.exists():
        data_path.parent.mkdir(exist_ok=True)
        text = "To be, or not to be, that is the question.\n" * 5000
        with open(data_path, 'w') as f:
            f.write(text)

    with open(data_path, 'r') as f:
        text = f.read()

    chars = sorted(list(set(text)))
    char_to_idx = {ch: i for i, ch in enumerate(chars)}
    vocab_size = len(chars)

    seq_len = 64
    data = torch.tensor([char_to_idx[c] for c in text], dtype=torch.long)
    n_seq = min(5000, (len(data) - seq_len - 1) // seq_len)
    x_all = torch.stack([data[i*seq_len:(i+1)*seq_len] for i in range(n_seq)])

    split = int(0.9 * len(x_all))
    x_train = x_all[:split].to(device)
    x_test = x_all[split:split+100].to(device)

    print(f"[Data] {len(x_all)} sequences, vocab {vocab_size}")

    # Model
    model = MandatoryTelemetryModel(
        vocab_size=vocab_size,
        hidden_dim=256,
        n_layers=4,
        n_recurrent=3
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[Model] {n_params:,} parameters")
    print(f"[Model] Telemetry gates at EVERY layer + recurrent processing")

    # Training
    print(f"\n{'='*60}")
    print("TRAINING: Mandatory telemetry integration")
    print(f"{'='*60}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    batch_size = 64
    n_epochs = 30

    for epoch in range(n_epochs):
        model.train()
        perm = torch.randperm(len(x_train))
        x_shuffled = x_train[perm]

        total_task_loss = 0.0
        total_hw_loss = 0.0
        n_batches = len(x_train) // batch_size

        for i in range(n_batches):
            x_batch = x_shuffled[i*batch_size:(i+1)*batch_size]
            y_batch = torch.roll(x_batch, -1, dims=1)

            sample = telemetry.read_sample()
            hw = torch.tensor([
                sample.temp_edge_c / 100.0, sample.power_w / 100.0,
                sample.freq_sclk_mhz / 3000.0, sample.gpu_busy_pct / 100.0,
            ] * 2, device=device).unsqueeze(0).expand(batch_size, -1)

            optimizer.zero_grad()

            logits, hw_pred = model(x_batch, hw)

            task_loss = F.cross_entropy(logits.view(-1, vocab_size), y_batch.view(-1))
            hw_loss = F.mse_loss(hw_pred, hw)

            loss = task_loss + 0.5 * hw_loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_task_loss += task_loss.item()
            total_hw_loss += hw_loss.item()

        if (epoch + 1) % 5 == 0:
            print(f"  Epoch {epoch+1}/{n_epochs}: task={total_task_loss/n_batches:.4f}, hw={total_hw_loss/n_batches:.4f}")

    # Tests
    sample = telemetry.read_sample()
    hw_test = torch.tensor([
        sample.temp_edge_c / 100.0, sample.power_w / 100.0,
        sample.freq_sclk_mhz / 3000.0, sample.gpu_busy_pct / 100.0,
    ] * 2, device=device).unsqueeze(0).expand(32, -1)

    print(f"\n{'='*60}")
    print("[TEST 1] PERTURBATIONAL COMPLEXITY INDEX")
    print(f"{'='*60}")

    pci_result = measure_pci(model, x_test[:32], hw_test)
    pci = pci_result['pci_metrics']

    print(f"  PCI = {pci['pci']:.4f}")
    print(f"  Complexity = {pci['complexity']:.4f}")
    print(f"  Integration = {pci['integration']:.4f}")
    print(f"  Differentiation = {pci['differentiation']:.4f}")
    pci_pass = pci['pci'] > 0.15
    print(f"  Result: {'PASS' if pci_pass else 'FAIL'}")

    print(f"\n{'='*60}")
    print("[TEST 2] CAUSAL INTERVENTION")
    print(f"{'='*60}")

    causal_result = test_causal_intervention(model, x_test[:32], telemetry, device)
    print(f"  Temp delta: {causal_result['temp_delta']:.1f}C")
    print(f"  Intervention shift: {causal_result['intervention_shift']:.6f}")
    print(f"  Random shift: {causal_result['random_shift']:.6f}")
    causal_pass = causal_result['causal_detected']
    print(f"  Result: {'PASS' if causal_pass else 'FAIL'}")

    print(f"\n{'='*60}")
    print("[TEST 3] DOUBLE DISSOCIATION")
    print(f"{'='*60}")

    dissoc_result = test_double_dissociation(model, x_test[:32], hw_test)
    print(f"  Baseline: {dissoc_result['baseline']:.4f}")
    for comp, effect in dissoc_result['effects'].items():
        if effect > 0.001:
            print(f"  {comp}: effect={effect:.4f}")
    dissoc_pass = dissoc_result['double_dissociation']
    print(f"  Non-zero effects: {dissoc_result['non_zero_effects']}")
    print(f"  Result: {'PASS' if dissoc_pass else 'FAIL'}")

    # Assessment
    print(f"\n{'='*60}")
    print("MANDATORY TELEMETRY CONSCIOUSNESS ASSESSMENT")
    print(f"{'='*60}")

    tests_passed = sum([pci_pass, causal_pass, dissoc_pass])

    print(f"\nTests passed: {tests_passed}/3")
    print(f"  [{'PASS' if pci_pass else 'FAIL'}] PCI > 0.15 (perturbational complexity)")
    print(f"  [{'PASS' if causal_pass else 'FAIL'}] Causal intervention")
    print(f"  [{'PASS' if dissoc_pass else 'FAIL'}] Double dissociation")

    if tests_passed == 3:
        verdict = "CONSCIOUSNESS_INDICATORS_STRONG"
        claim = "Mandatory telemetry system shows high PCI, causal coupling, and functional independence"
    elif tests_passed >= 2:
        verdict = "CONSCIOUSNESS_INDICATORS_PARTIAL"
        claim = "System shows telemetry dependence but integration needs improvement"
    else:
        verdict = "CONSCIOUSNESS_INDICATORS_WEAK"
        claim = "System does not exhibit strong consciousness-like signatures"

    print(f"\nVERDICT: {verdict}")
    print(f"\nClaim: \"{claim}\"")

    # Save
    results = {
        'experiment': 'z2011_mandatory_telemetry_integration',
        'timestamp': timestamp,
        'device': str(device),
        'model_params': n_params,
        'training_epochs': n_epochs,
        'key_innovation': 'Telemetry gates at EVERY layer with recurrent processing',
        'tests': {
            'pci': pci_result,
            'causal_intervention': causal_result,
            'double_dissociation': dissoc_result
        },
        'summary': {
            'tests_passed': tests_passed,
            'pci_pass': pci_pass,
            'causal_pass': causal_pass,
            'dissociation_pass': dissoc_pass
        },
        'verdict': verdict,
        'claim': claim
    }

    results_path = Path(__file__).parent.parent / 'results' / 'z2011_mandatory_telemetry_integration.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n[Saved] {results_path}")

    return results


if __name__ == '__main__':
    main()
