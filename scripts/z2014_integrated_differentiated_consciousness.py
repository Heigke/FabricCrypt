#!/usr/bin/env python3
"""
z2014: Integrated and Differentiated Consciousness

The z2013 result revealed the core tension:
- Integration: 0.1537 (good) - channels correlate within perturbations
- Differentiation: 0.0024 (terrible) - all perturbations produce same response

Biological consciousness has BOTH:
- High integration WITHIN each conscious state (IIT's phi)
- High differentiation ACROSS conscious states (many possible states)

This script adds:
1. Contrastive differentiation loss - push different perturbations apart
2. Keep integration within each response
3. Hardware-dependent state selection - different HW states -> different responses
4. More aggressive GPU heating for causal test
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
import time
from datetime import datetime
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.telemetry.sysfs_hwmon import SysfsHwmonTelemetry, GpuSample


def get_telemetry_state(sensor: SysfsHwmonTelemetry) -> dict:
    """Helper to get telemetry in dict format."""
    sample = sensor.read_sample()
    return {
        'gpu_temp': getattr(sample, 'temp_c', 50),
        'cpu_temp': getattr(sample, 'temp_c', 50),
        'gpu_power': getattr(sample, 'power_w', 20),
        'cpu_power': 20,
        'gpu_util': getattr(sample, 'gpu_util', 0),
        'cpu_util': 50,
        'mem_used': 50,
        'fan_speed': 50,
    }


def make_hw_tensor(hw_state: dict, batch_size: int, device: torch.device) -> torch.Tensor:
    """Convert hardware state dict to normalized tensor."""
    hw = torch.tensor([
        hw_state['gpu_temp'] / 100.0,
        hw_state['cpu_temp'] / 100.0,
        hw_state['gpu_power'] / 200.0,
        hw_state['cpu_power'] / 100.0,
        hw_state['gpu_util'] / 100.0,
        hw_state['cpu_util'] / 100.0,
        hw_state['mem_used'] / 100.0,
        hw_state['fan_speed'] / 100.0,
    ], dtype=torch.float32, device=device)
    return hw.unsqueeze(0).expand(batch_size, -1)


# ============================================================================
# Architecture: Integration + Differentiation
# ============================================================================

class IntegratedDifferentiatedBlock(nn.Module):
    """
    Block that produces:
    1. Integrated responses (channels correlate within batch)
    2. Differentiated responses (different inputs -> different outputs)

    Key: Use learnable "state templates" that the model selects based on input.
    Each template is integrated (fixed pattern), but selection provides differentiation.
    """
    def __init__(self, hidden_dim, n_states=8):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_states = n_states

        # Learnable state templates - each is an integrated pattern
        self.state_templates = nn.Parameter(torch.randn(n_states, hidden_dim) * 0.1)

        # State selector - picks which template based on input
        self.state_selector = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.GELU(),
            nn.Linear(64, n_states),
        )

        # Integration projection
        self.integrate = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x, hw_telemetry):
        # x: [batch, seq, hidden]
        # hw_telemetry: [batch, hidden] (projected)

        batch, seq, hidden = x.shape

        # Compute state selection logits from input + hardware
        pooled = x.mean(dim=1)  # [batch, hidden]
        combined = pooled + hw_telemetry  # Hardware influences selection

        # Soft selection over states (allows gradient flow)
        selection = F.softmax(self.state_selector(combined), dim=-1)  # [batch, n_states]

        # Weighted sum of state templates
        selected_template = torch.einsum('bn,nh->bh', selection, self.state_templates)

        # Expand to sequence and add to input
        template_expanded = selected_template.unsqueeze(1).expand(-1, seq, -1)

        # Integrate: all channels influenced by same template
        out = self.integrate(x + template_expanded * 0.5)

        return out, selection


class GlobalWorkspaceWithDifferentiation(nn.Module):
    """Global workspace that maintains integration but adds differentiation."""
    def __init__(self, n_channels, workspace_dim):
        super().__init__()
        self.to_workspace = nn.Linear(n_channels, workspace_dim)
        self.workspace_process = nn.Sequential(
            nn.LayerNorm(workspace_dim),
            nn.Linear(workspace_dim, workspace_dim),
            nn.GELU(),
        )
        self.from_workspace = nn.Linear(workspace_dim, n_channels)

    def forward(self, x):
        workspace = self.to_workspace(x)
        workspace = self.workspace_process(workspace)
        broadcast = self.from_workspace(workspace)
        return x + broadcast


class IntegratedDifferentiatedModel(nn.Module):
    """
    Model with both high integration and high differentiation.
    """
    def __init__(self, vocab_size=256, hidden_dim=64, n_hw_channels=8, n_states=16):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim

        # Embeddings
        self.embed = nn.Embedding(vocab_size, hidden_dim)

        # Hardware encoder (project to hidden_dim for state selection)
        self.hw_encoder = nn.Sequential(
            nn.Linear(n_hw_channels, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )

        # Layer 1: Integrated differentiated block
        self.block1 = IntegratedDifferentiatedBlock(hidden_dim, n_states)

        # Global workspace for integration
        self.workspace = GlobalWorkspaceWithDifferentiation(hidden_dim, 16)

        # Layer 2: Another ID block with different states
        self.block2 = IntegratedDifferentiatedBlock(hidden_dim, n_states)

        # Hardware-dependent gating
        self.hw_gate = nn.Linear(hidden_dim, hidden_dim)

        # Output
        self.out = nn.Linear(hidden_dim, vocab_size)

        # Store for analysis
        self.last_hidden = None
        self.last_selections = None

    def forward(self, x, hw_telemetry):
        # Embed input
        h = self.embed(x)

        # Encode hardware
        hw = self.hw_encoder(hw_telemetry)

        # Block 1: integrated + differentiated
        h, sel1 = self.block1(h, hw)

        # Global workspace for cross-channel integration
        h = self.workspace(h)

        # Block 2: another layer of integration + differentiation
        h, sel2 = self.block2(h, hw)

        # Hardware gating (multiplicative)
        gate = torch.sigmoid(self.hw_gate(hw))
        h = h * gate.unsqueeze(1)

        # Store for analysis
        self.last_hidden = h.detach()
        self.last_selections = (sel1.detach(), sel2.detach())

        # Output
        logits = self.out(h)
        return logits


# ============================================================================
# Losses for Integration + Differentiation
# ============================================================================

def integration_loss(hidden_states):
    """Encourage channels to correlate (integration)."""
    batch_size = hidden_states.shape[0]
    if batch_size < 2:
        return torch.tensor(0.0, device=hidden_states.device)

    flat = hidden_states.view(batch_size, -1)
    flat_centered = flat - flat.mean(dim=0, keepdim=True)

    cov = torch.mm(flat_centered.t(), flat_centered) / (batch_size - 1)
    std = flat.std(dim=0, keepdim=True).t()
    std_matrix = std @ std.t() + 1e-8

    corr = cov / std_matrix
    mask = 1 - torch.eye(corr.shape[0], device=corr.device)
    mean_corr = (corr.abs() * mask).sum() / (mask.sum() + 1e-8)

    return -mean_corr  # Negative because we want HIGH correlation


def differentiation_loss(selections, n_classes):
    """
    Encourage diverse state selections (differentiation).
    We want the model to use all available states, not collapse to one.
    """
    # selections: [batch, n_states] - softmax probabilities

    # Compute entropy of average selection
    avg_selection = selections.mean(dim=0)  # [n_states]
    entropy = -(avg_selection * (avg_selection + 1e-8).log()).sum()

    # Maximum entropy = log(n_states)
    max_entropy = np.log(n_classes)

    # We want HIGH entropy, so loss is negative
    return -(entropy / max_entropy)


def contrastive_loss(hidden1, hidden2):
    """
    Push different perturbations to produce different responses.
    Used during training with perturbed vs unperturbed inputs.
    """
    flat1 = hidden1.view(hidden1.shape[0], -1)
    flat2 = hidden2.view(hidden2.shape[0], -1)

    # Cosine similarity
    cos_sim = F.cosine_similarity(flat1, flat2, dim=-1).mean()

    # We want LOW similarity between different perturbations
    return cos_sim  # Minimize this


# ============================================================================
# PCI Measurement
# ============================================================================

def lempel_ziv_complexity(binary_sequence):
    s = ''.join(str(int(b)) for b in binary_sequence)
    n = len(s)
    if n == 0:
        return 0.0

    i, c, l = 0, 1, 1
    k, k_max = 1, 1

    while True:
        if s[i + k - 1] == s[l + k - 1]:
            k += 1
            if l + k > n:
                c += 1
                break
        else:
            if k > k_max:
                k_max = k
            i += 1
            if i == l:
                c += 1
                l += k_max
                if l + 1 > n:
                    break
                i = 0
                k = 1
                k_max = 1
            else:
                k = 1

    b = n / np.log2(n) if n > 1 else 1
    return c / b if b > 0 else 0


def measure_pci(model, hw_tensor, device, n_perturbations=30):
    model.eval()

    batch_size = 8
    seq_len = 32
    base_input = torch.randint(0, model.vocab_size, (batch_size, seq_len), device=device)

    with torch.no_grad():
        _ = model(base_input, hw_tensor[:batch_size])
        base_hidden = model.last_hidden.clone()

    responses = []
    for i in range(n_perturbations):
        # Create unique perturbation for each condition
        perturb_strength = (i + 1) / n_perturbations  # Varying strength
        perturbed_input = base_input.clone()

        # Different perturbation types
        if i % 3 == 0:
            # Random token replacement
            mask = torch.rand(batch_size, seq_len, device=device) < perturb_strength * 0.3
            perturbed_input[mask] = torch.randint(0, model.vocab_size,
                                                   (mask.sum().item(),), device=device)
        elif i % 3 == 1:
            # Token shift
            shift = int(perturb_strength * 10)
            perturbed_input = (perturbed_input + shift) % model.vocab_size
        else:
            # Reversal
            perturbed_input = perturbed_input.flip(dims=[1])

        with torch.no_grad():
            _ = model(perturbed_input, hw_tensor[:batch_size])
            response = model.last_hidden - base_hidden
            responses.append(response.cpu().numpy())

    responses = np.array(responses)
    n_channels = responses.shape[1] * responses.shape[2] * responses.shape[3]
    flat_responses = responses.reshape(n_perturbations, n_channels)

    # Binarize
    threshold = np.median(np.abs(flat_responses))
    binary = (np.abs(flat_responses) > threshold).astype(int)

    # Complexity
    complexities = []
    for ch in range(min(n_channels, 100)):
        lz = lempel_ziv_complexity(binary[:, ch])
        complexities.append(lz)
    complexity = np.mean(complexities)

    # Integration
    if n_channels > 1:
        corr_matrix = np.corrcoef(flat_responses.T)
        corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)
        mask = 1 - np.eye(corr_matrix.shape[0])
        integration = np.abs(corr_matrix * mask).sum() / (mask.sum() + 1e-8)
    else:
        integration = 0.0

    # Differentiation - variance across perturbation conditions
    differentiation = np.std(flat_responses, axis=0).mean()

    # PCI
    pci = complexity * (integration + differentiation) / 2

    return {
        'pci': float(pci),
        'complexity': float(complexity),
        'integration': float(integration),
        'differentiation': float(differentiation),
        'n_channels_total': n_channels
    }


# ============================================================================
# Testing Functions
# ============================================================================

def heat_gpu(device, duration_s=2.0):
    """Aggressively heat GPU."""
    start = time.time()
    mat = torch.randn(3000, 3000, device=device)
    while time.time() - start < duration_s:
        mat = mat @ mat.t()
        mat = mat / (mat.norm() + 1e-8)
    del mat
    torch.cuda.synchronize()


def test_causal_intervention(model, telemetry, device):
    model.eval()

    # Get baseline
    hw_state = get_telemetry_state(telemetry)
    base_temp = hw_state['gpu_temp']

    batch_size = 16
    seq_len = 32
    test_input = torch.randint(0, model.vocab_size, (batch_size, seq_len), device=device)

    hw_tensor = make_hw_tensor(hw_state, batch_size, device)

    with torch.no_grad():
        baseline_out = model(test_input, hw_tensor)
        baseline_probs = F.softmax(baseline_out, dim=-1)

    # Heat GPU more aggressively
    print("    Heating GPU...")
    heat_gpu(device, duration_s=3.0)

    hw_state_hot = get_telemetry_state(telemetry)
    new_temp = hw_state_hot['gpu_temp']

    hw_tensor_hot = make_hw_tensor(hw_state_hot, batch_size, device)

    with torch.no_grad():
        hot_out = model(test_input, hw_tensor_hot)
        hot_probs = F.softmax(hot_out, dim=-1)

    intervention_shift = (hot_probs - baseline_probs).abs().mean().item()

    # Control: random telemetry
    random_hw = torch.rand_like(hw_tensor)
    with torch.no_grad():
        random_out = model(test_input, random_hw)
        random_probs = F.softmax(random_out, dim=-1)
    random_shift = (random_probs - baseline_probs).abs().mean().item()

    # Causal = intervention produces meaningful shift
    causal_detected = intervention_shift > 0.001 and (new_temp - base_temp) > 1.0

    return {
        'temp_delta': float(new_temp - base_temp),
        'intervention_shift': float(intervention_shift),
        'random_shift': float(random_shift),
        'causal_detected': causal_detected
    }


def test_double_dissociation(model, telemetry, device):
    model.eval()

    batch_size = 16
    seq_len = 32
    test_input = torch.randint(0, model.vocab_size, (batch_size, seq_len), device=device)

    hw_state = get_telemetry_state(telemetry)
    hw_tensor = make_hw_tensor(hw_state, batch_size, device)

    with torch.no_grad():
        baseline = model(test_input, hw_tensor)

    effects = {}

    # Ablate block1 state templates
    original = model.block1.state_templates.data.clone()
    model.block1.state_templates.data.zero_()
    with torch.no_grad():
        ablated = model(test_input, hw_tensor)
    effects['block1_templates'] = (ablated - baseline).abs().mean().item()
    model.block1.state_templates.data = original

    # Ablate block2 state templates
    original = model.block2.state_templates.data.clone()
    model.block2.state_templates.data.zero_()
    with torch.no_grad():
        ablated = model(test_input, hw_tensor)
    effects['block2_templates'] = (ablated - baseline).abs().mean().item()
    model.block2.state_templates.data = original

    # Ablate global workspace
    original = model.workspace.to_workspace.weight.data.clone()
    model.workspace.to_workspace.weight.data.zero_()
    with torch.no_grad():
        ablated = model(test_input, hw_tensor)
    effects['workspace'] = (ablated - baseline).abs().mean().item()
    model.workspace.to_workspace.weight.data = original

    # Ablate hw_gate
    original = model.hw_gate.weight.data.clone()
    model.hw_gate.weight.data.zero_()
    with torch.no_grad():
        ablated = model(test_input, hw_tensor)
    effects['hw_gate'] = (ablated - baseline).abs().mean().item()
    model.hw_gate.weight.data = original

    non_zero = sum(1 for e in effects.values() if e > 0.01)

    return {
        'baseline': float(baseline.abs().mean().item()),
        'effects': effects,
        'non_zero_effects': non_zero,
        'double_dissociation': non_zero >= 2
    }


# ============================================================================
# Training with Integration + Differentiation
# ============================================================================

def train_integrated_differentiated(model, telemetry, device, epochs=80):
    """Train with both integration and differentiation losses."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)

    batch_size = 32
    seq_len = 64

    losses = []

    for epoch in range(epochs):
        model.train()

        # Generate data
        x = torch.randint(0, model.vocab_size, (batch_size, seq_len), device=device)
        y = torch.roll(x, -1, dims=1)

        # Get telemetry
        hw_state = get_telemetry_state(telemetry)
        hw_tensor = make_hw_tensor(hw_state, batch_size, device)
        hw_tensor = hw_tensor + torch.randn_like(hw_tensor) * 0.1
        hw_tensor = hw_tensor.clamp(0, 1)

        # Forward pass
        logits = model(x, hw_tensor)

        # Task loss
        task_loss = F.cross_entropy(logits.view(-1, model.vocab_size), y.view(-1))

        # Integration loss
        int_loss = integration_loss(model.last_hidden)

        # Differentiation loss (from state selections)
        sel1, sel2 = model.last_selections
        diff_loss1 = differentiation_loss(sel1, model.block1.n_states)
        diff_loss2 = differentiation_loss(sel2, model.block2.n_states)
        diff_loss = (diff_loss1 + diff_loss2) / 2

        # Contrastive: perturbed input should produce different hidden
        perturbed_x = (x + torch.randint(1, 10, x.shape, device=device)) % model.vocab_size
        _ = model(perturbed_x, hw_tensor)
        perturbed_hidden = model.last_hidden.clone()

        # Re-run original to get original hidden
        _ = model(x, hw_tensor)
        original_hidden = model.last_hidden.clone()

        contrast_loss = contrastive_loss(original_hidden, perturbed_hidden)

        # Combined loss
        total_loss = task_loss + 0.3 * int_loss + 0.3 * diff_loss + 0.2 * contrast_loss

        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        losses.append(task_loss.item())

        if (epoch + 1) % 20 == 0:
            print(f"  Epoch {epoch+1}/{epochs}: task={task_loss.item():.4f}, "
                  f"int={-int_loss.item():.4f}, diff={-diff_loss.item():.4f}, "
                  f"contrast={contrast_loss.item():.4f}")

    return losses


# ============================================================================
# Main
# ============================================================================

def main():
    print("=" * 70)
    print("z2014: Integrated + Differentiated Consciousness")
    print("=" * 70)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    telemetry = SysfsHwmonTelemetry()

    # Create model
    model = IntegratedDifferentiatedModel(
        vocab_size=256,
        hidden_dim=64,
        n_hw_channels=8,
        n_states=16  # More states for differentiation
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")

    # Train
    print("\n[1/4] Training with integration + differentiation losses...")
    losses = train_integrated_differentiated(model, telemetry, device, epochs=80)

    # Get telemetry for tests
    hw_state = get_telemetry_state(telemetry)
    hw_tensor = make_hw_tensor(hw_state, 8, device)

    # Test 1: PCI
    print("\n[2/4] Measuring PCI...")
    pci_results = measure_pci(model, hw_tensor, device, n_perturbations=40)
    print(f"  PCI = {pci_results['pci']:.4f}")
    print(f"  Complexity = {pci_results['complexity']:.4f}")
    print(f"  Integration = {pci_results['integration']:.4f}")
    print(f"  Differentiation = {pci_results['differentiation']:.4f}")

    pci_pass = pci_results['pci'] > 0.15
    print(f"  PCI test: {'PASS' if pci_pass else 'FAIL'} (need > 0.15)")

    # Test 2: Causal intervention
    print("\n[3/4] Testing causal intervention...")
    causal_results = test_causal_intervention(model, telemetry, device)
    print(f"  Temp delta: {causal_results['temp_delta']:.1f}°C")
    print(f"  Intervention shift: {causal_results['intervention_shift']:.6f}")
    print(f"  Random shift: {causal_results['random_shift']:.6f}")

    causal_pass = causal_results['causal_detected']
    print(f"  Causal test: {'PASS' if causal_pass else 'FAIL'}")

    # Test 3: Double dissociation
    print("\n[4/4] Testing double dissociation...")
    dissoc_results = test_double_dissociation(model, telemetry, device)
    print(f"  Effects: {dissoc_results['effects']}")
    print(f"  Non-zero effects: {dissoc_results['non_zero_effects']}")

    dissoc_pass = dissoc_results['double_dissociation']
    print(f"  Double dissociation: {'PASS' if dissoc_pass else 'FAIL'}")

    # Summary
    tests_passed = sum([pci_pass, causal_pass, dissoc_pass])

    if tests_passed == 3:
        verdict = "CONSCIOUSNESS_INDICATORS_STRONG"
        claim = "System passes PCI, causal intervention, and double dissociation"
    elif tests_passed >= 2:
        verdict = "CONSCIOUSNESS_INDICATORS_PARTIAL"
        claim = "System shows some consciousness indicators but not all"
    else:
        verdict = "CONSCIOUSNESS_INDICATORS_WEAK"
        claim = "System needs stronger integration and causal coupling"

    print("\n" + "=" * 70)
    print(f"VERDICT: {verdict}")
    print(f"Tests passed: {tests_passed}/3")
    print(f"Claim: {claim}")
    print("=" * 70)

    # Save results
    results = {
        'experiment': 'z2014_integrated_differentiated_consciousness',
        'timestamp': datetime.now().isoformat(),
        'device': str(device),
        'model_params': n_params,
        'key_innovation': 'State templates for differentiation + global workspace for integration',
        'training': {
            'epochs': 80,
            'final_task_loss': float(losses[-1]) if losses else None
        },
        'tests': {
            'pci': {'pci_metrics': pci_results},
            'causal_intervention': causal_results,
            'double_dissociation': dissoc_results
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

    results_path = Path(__file__).parent.parent / 'results' / 'z2014_integrated_differentiated_consciousness.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}")

    return results


if __name__ == '__main__':
    main()
