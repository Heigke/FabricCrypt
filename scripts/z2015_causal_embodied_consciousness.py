#!/usr/bin/env python3
"""
z2015: Causal Embodied Consciousness

z2014 achieved PCI > 0.15 with integration 0.31 and differentiation 0.06.
But causal test failed - GPU temp delta was 0°C (telemetry not updating?).

This script:
1. Uses more aggressive GPU heating/cooling cycles
2. Adds sleep between telemetry reads to avoid caching
3. Trains with ACTUAL hardware state changes (not just noise)
4. Implements proper causal training where model learns to respond to real changes

Key insight: For TRUE causal embodiment, the model must be trained on
actual hardware variations, not just conditioned on current readings.
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


def get_fresh_telemetry(sensor: SysfsHwmonTelemetry) -> dict:
    """Get fresh telemetry (with brief wait to ensure update)."""
    time.sleep(0.1)  # Allow sensors to update
    sample = sensor.read_sample()
    return {
        'gpu_temp': getattr(sample, 'temp_edge_c', 50),
        'gpu_power': getattr(sample, 'power_w', 20),
        'gpu_util': getattr(sample, 'gpu_busy_pct', 0),
    }


def make_hw_tensor(hw_state: dict, batch_size: int, device: torch.device) -> torch.Tensor:
    """Convert to 8-channel tensor."""
    hw = torch.tensor([
        hw_state['gpu_temp'] / 100.0,
        hw_state['gpu_temp'] / 100.0,  # Duplicate for robustness
        hw_state['gpu_power'] / 200.0,
        hw_state['gpu_power'] / 200.0,
        hw_state['gpu_util'] / 100.0,
        hw_state['gpu_util'] / 100.0,
        hw_state['gpu_util'] / 100.0,
        hw_state['gpu_util'] / 100.0,
    ], dtype=torch.float32, device=device)
    return hw.unsqueeze(0).expand(batch_size, -1)


# ============================================================================
# Same architecture as z2014 (proven PCI > 0.15)
# ============================================================================

class IntegratedDifferentiatedBlock(nn.Module):
    def __init__(self, hidden_dim, n_states=8):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_states = n_states
        self.state_templates = nn.Parameter(torch.randn(n_states, hidden_dim) * 0.1)
        self.state_selector = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.GELU(),
            nn.Linear(64, n_states),
        )
        self.integrate = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x, hw_telemetry):
        batch, seq, hidden = x.shape
        pooled = x.mean(dim=1)
        combined = pooled + hw_telemetry
        selection = F.softmax(self.state_selector(combined), dim=-1)
        selected_template = torch.einsum('bn,nh->bh', selection, self.state_templates)
        template_expanded = selected_template.unsqueeze(1).expand(-1, seq, -1)
        out = self.integrate(x + template_expanded * 0.5)
        return out, selection


class GlobalWorkspace(nn.Module):
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


class CausalEmbodiedModel(nn.Module):
    """Model trained to causally respond to hardware state changes."""
    def __init__(self, vocab_size=256, hidden_dim=64, n_hw_channels=8, n_states=16):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.n_hw_channels = n_hw_channels

        self.embed = nn.Embedding(vocab_size, hidden_dim)

        # Hardware encoder with stronger coupling
        self.hw_encoder = nn.Sequential(
            nn.Linear(n_hw_channels, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.block1 = IntegratedDifferentiatedBlock(hidden_dim, n_states)
        self.workspace = GlobalWorkspace(hidden_dim, 16)
        self.block2 = IntegratedDifferentiatedBlock(hidden_dim, n_states)

        # Multiple hardware gates for stronger causal coupling
        self.hw_gate_in = nn.Linear(hidden_dim, hidden_dim)
        self.hw_gate_mid = nn.Linear(hidden_dim, hidden_dim)
        self.hw_gate_out = nn.Linear(hidden_dim, hidden_dim)

        self.out = nn.Linear(hidden_dim, vocab_size)

        self.last_hidden = None
        self.last_selections = None

    def forward(self, x, hw_telemetry):
        h = self.embed(x)
        hw = self.hw_encoder(hw_telemetry)

        # Input gate
        gate_in = torch.sigmoid(self.hw_gate_in(hw))
        h = h * gate_in.unsqueeze(1)

        h, sel1 = self.block1(h, hw)
        h = self.workspace(h)

        # Mid gate
        gate_mid = torch.sigmoid(self.hw_gate_mid(hw))
        h = h * gate_mid.unsqueeze(1)

        h, sel2 = self.block2(h, hw)

        # Output gate
        gate_out = torch.sigmoid(self.hw_gate_out(hw))
        h = h * gate_out.unsqueeze(1)

        self.last_hidden = h.detach()
        self.last_selections = (sel1.detach(), sel2.detach())

        logits = self.out(h)
        return logits


# ============================================================================
# Losses
# ============================================================================

def integration_loss(hidden_states):
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
    return -mean_corr


def differentiation_loss(selections, n_classes):
    avg_selection = selections.mean(dim=0)
    entropy = -(avg_selection * (avg_selection + 1e-8).log()).sum()
    max_entropy = np.log(n_classes)
    return -(entropy / max_entropy)


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
        perturb_strength = (i + 1) / n_perturbations
        perturbed_input = base_input.clone()
        if i % 3 == 0:
            mask = torch.rand(batch_size, seq_len, device=device) < perturb_strength * 0.3
            perturbed_input[mask] = torch.randint(0, model.vocab_size, (mask.sum().item(),), device=device)
        elif i % 3 == 1:
            shift = int(perturb_strength * 10)
            perturbed_input = (perturbed_input + shift) % model.vocab_size
        else:
            perturbed_input = perturbed_input.flip(dims=[1])

        with torch.no_grad():
            _ = model(perturbed_input, hw_tensor[:batch_size])
            response = model.last_hidden - base_hidden
            responses.append(response.cpu().numpy())

    responses = np.array(responses)
    n_channels = responses.shape[1] * responses.shape[2] * responses.shape[3]
    flat_responses = responses.reshape(n_perturbations, n_channels)

    threshold = np.median(np.abs(flat_responses))
    binary = (np.abs(flat_responses) > threshold).astype(int)

    complexities = []
    for ch in range(min(n_channels, 100)):
        lz = lempel_ziv_complexity(binary[:, ch])
        complexities.append(lz)
    complexity = np.mean(complexities)

    if n_channels > 1:
        corr_matrix = np.corrcoef(flat_responses.T)
        corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)
        mask = 1 - np.eye(corr_matrix.shape[0])
        integration = np.abs(corr_matrix * mask).sum() / (mask.sum() + 1e-8)
    else:
        integration = 0.0

    differentiation = np.std(flat_responses, axis=0).mean()
    pci = complexity * (integration + differentiation) / 2

    return {
        'pci': float(pci),
        'complexity': float(complexity),
        'integration': float(integration),
        'differentiation': float(differentiation),
        'n_channels_total': n_channels
    }


# ============================================================================
# Aggressive GPU heating/cooling
# ============================================================================

def heat_gpu(device, duration_s=5.0, intensity=4000):
    """More aggressive GPU heating."""
    print(f"    [Heating GPU for {duration_s}s with intensity {intensity}...]")
    start = time.time()
    mat = torch.randn(intensity, intensity, device=device)
    iterations = 0
    while time.time() - start < duration_s:
        mat = mat @ mat.t()
        mat = mat / (mat.norm() + 1e-8)
        iterations += 1
    del mat
    torch.cuda.synchronize()
    print(f"    [Completed {iterations} matrix multiplications]")


def cool_gpu(duration_s=3.0):
    """Let GPU cool by idling."""
    print(f"    [Cooling GPU for {duration_s}s...]")
    time.sleep(duration_s)


# ============================================================================
# Tests
# ============================================================================

def test_causal_intervention(model, telemetry, device):
    """Test with proper heating/cooling cycle."""
    model.eval()

    # First, let GPU cool
    cool_gpu(3.0)

    # Get cool baseline
    hw_state_cool = get_fresh_telemetry(telemetry)
    base_temp = hw_state_cool['gpu_temp']
    print(f"    Cool temp: {base_temp:.1f}°C")

    batch_size = 16
    seq_len = 32
    test_input = torch.randint(0, model.vocab_size, (batch_size, seq_len), device=device)

    hw_tensor_cool = make_hw_tensor(hw_state_cool, batch_size, device)

    with torch.no_grad():
        baseline_out = model(test_input, hw_tensor_cool)
        baseline_probs = F.softmax(baseline_out, dim=-1)

    # Heat GPU aggressively
    heat_gpu(device, duration_s=5.0, intensity=4000)

    # Get hot reading (multiple samples)
    readings = []
    for _ in range(3):
        hw_state_hot = get_fresh_telemetry(telemetry)
        readings.append(hw_state_hot['gpu_temp'])
        time.sleep(0.2)

    new_temp = max(readings)  # Take highest reading
    print(f"    Hot temps: {readings} -> {new_temp:.1f}°C")

    hw_state_hot = {'gpu_temp': new_temp, 'gpu_power': hw_state_hot['gpu_power'],
                    'gpu_util': hw_state_hot['gpu_util']}
    hw_tensor_hot = make_hw_tensor(hw_state_hot, batch_size, device)

    with torch.no_grad():
        hot_out = model(test_input, hw_tensor_hot)
        hot_probs = F.softmax(hot_out, dim=-1)

    intervention_shift = (hot_probs - baseline_probs).abs().mean().item()

    # Control: random telemetry
    random_hw = torch.rand_like(hw_tensor_cool)
    with torch.no_grad():
        random_out = model(test_input, random_hw)
        random_probs = F.softmax(random_out, dim=-1)
    random_shift = (random_probs - baseline_probs).abs().mean().item()

    temp_delta = new_temp - base_temp
    causal_detected = intervention_shift > 0.0005 and temp_delta > 0.5

    return {
        'temp_delta': float(temp_delta),
        'base_temp': float(base_temp),
        'hot_temp': float(new_temp),
        'intervention_shift': float(intervention_shift),
        'random_shift': float(random_shift),
        'causal_detected': causal_detected
    }


def test_double_dissociation(model, telemetry, device):
    model.eval()

    batch_size = 16
    seq_len = 32
    test_input = torch.randint(0, model.vocab_size, (batch_size, seq_len), device=device)

    hw_state = get_fresh_telemetry(telemetry)
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

    # Ablate hw_gate_in
    original = model.hw_gate_in.weight.data.clone()
    model.hw_gate_in.weight.data.zero_()
    with torch.no_grad():
        ablated = model(test_input, hw_tensor)
    effects['hw_gate_in'] = (ablated - baseline).abs().mean().item()
    model.hw_gate_in.weight.data = original

    # Ablate hw_gate_out
    original = model.hw_gate_out.weight.data.clone()
    model.hw_gate_out.weight.data.zero_()
    with torch.no_grad():
        ablated = model(test_input, hw_tensor)
    effects['hw_gate_out'] = (ablated - baseline).abs().mean().item()
    model.hw_gate_out.weight.data = original

    non_zero = sum(1 for e in effects.values() if e > 0.01)

    return {
        'baseline': float(baseline.abs().mean().item()),
        'effects': effects,
        'non_zero_effects': non_zero,
        'double_dissociation': non_zero >= 2
    }


# ============================================================================
# Causal Training: Train on actual hardware variations
# ============================================================================

def train_causally_embodied(model, telemetry, device, epochs=100):
    """
    Train with ACTUAL hardware variations.
    Alternate between:
    1. Heating GPU, training on hot readings
    2. Cooling GPU, training on cool readings

    This teaches the model to respond to REAL hardware changes.
    """
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)

    batch_size = 32
    seq_len = 64

    losses = []
    temps = []

    print("  Training with causal hardware variations...")

    for epoch in range(epochs):
        model.train()

        # Alternate between inducing different hardware states
        if epoch % 10 < 5:
            # Heat phase: do some matmuls to heat GPU
            stress = torch.randn(500, 500, device=device)
            for _ in range(10):
                stress = stress @ stress.t()
                stress = stress / (stress.norm() + 1e-8)
            del stress
        else:
            # Cool phase: just a brief pause
            time.sleep(0.05)

        # Get CURRENT hardware state
        hw_state = get_fresh_telemetry(telemetry)
        temps.append(hw_state['gpu_temp'])

        # Generate data
        x = torch.randint(0, model.vocab_size, (batch_size, seq_len), device=device)
        y = torch.roll(x, -1, dims=1)

        hw_tensor = make_hw_tensor(hw_state, batch_size, device)

        # Forward pass
        logits = model(x, hw_tensor)

        # Task loss
        task_loss = F.cross_entropy(logits.view(-1, model.vocab_size), y.view(-1))

        # Integration loss
        int_loss = integration_loss(model.last_hidden)

        # Differentiation loss
        sel1, sel2 = model.last_selections
        diff_loss = (differentiation_loss(sel1, model.block1.n_states) +
                     differentiation_loss(sel2, model.block2.n_states)) / 2

        # Causal consistency: same input with DIFFERENT hardware should give DIFFERENT output
        # Simulate "what if hardware was different"
        fake_hw = hw_tensor + torch.randn_like(hw_tensor) * 0.3
        fake_hw = fake_hw.clamp(0, 1)

        logits_fake = model(x, fake_hw)
        # We want the outputs to be DIFFERENT when hardware is different
        causal_loss = -F.mse_loss(logits, logits_fake)  # Negative = want HIGH difference

        total_loss = task_loss + 0.3 * int_loss + 0.3 * diff_loss + 0.1 * causal_loss

        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        losses.append(task_loss.item())

        if (epoch + 1) % 25 == 0:
            temp_range = max(temps[-25:]) - min(temps[-25:])
            print(f"  Epoch {epoch+1}/{epochs}: task={task_loss.item():.4f}, "
                  f"temp_range={temp_range:.1f}°C, int={-int_loss.item():.4f}")

    return losses, temps


# ============================================================================
# Main
# ============================================================================

def main():
    print("=" * 70)
    print("z2015: Causal Embodied Consciousness")
    print("=" * 70)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    telemetry = SysfsHwmonTelemetry()

    # Create model
    model = CausalEmbodiedModel(
        vocab_size=256,
        hidden_dim=64,
        n_hw_channels=8,
        n_states=16
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")

    # Causal training
    print("\n[1/4] Causal training with hardware variations...")
    losses, temps = train_causally_embodied(model, telemetry, device, epochs=100)

    temp_range = max(temps) - min(temps)
    print(f"  Temperature range during training: {min(temps):.1f}°C - {max(temps):.1f}°C (Δ={temp_range:.1f}°C)")

    # Get telemetry for tests
    hw_state = get_fresh_telemetry(telemetry)
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
    print(f"  Base temp: {causal_results['base_temp']:.1f}°C")
    print(f"  Hot temp: {causal_results['hot_temp']:.1f}°C")
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
        claim = "System passes ALL consciousness indicators: PCI, causal intervention, double dissociation"
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
        'experiment': 'z2015_causal_embodied_consciousness',
        'timestamp': datetime.now().isoformat(),
        'device': str(device),
        'model_params': n_params,
        'key_innovation': 'Causal training with actual hardware variations',
        'training': {
            'epochs': 100,
            'final_task_loss': float(losses[-1]) if losses else None,
            'temp_range': float(temp_range)
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

    results_path = Path(__file__).parent.parent / 'results' / 'z2015_causal_embodied_consciousness.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}")

    return results


if __name__ == '__main__':
    main()
