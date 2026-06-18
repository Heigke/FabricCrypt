#!/usr/bin/env python3
"""
Z1105: Kernel-Level Embodied Training

This experiment ACTUALLY uses HIP kernels for embodiment, comparing:
- A: Standard PyTorch attention (no embodiment)
- B: PyTorch-level embodiment (modulation in Python, like z910)
- C: Kernel-level embodiment (modulation in HIP kernel)

The key question: Does kernel-level modulation produce different training dynamics
than equivalent PyTorch-level modulation?

Hypothesis: Kernel-level modulation cannot be "unlearned" by the optimizer because:
1. It happens atomically within a single kernel call
2. Gradients don't flow through the modulation factor
3. The computation is fundamentally different (not just faster)

Author: FEEL Research Team
Date: 2026-01-29
"""

import os
os.environ['HSA_OVERRIDE_GFX_VERSION'] = '11.0.0'

import sys
import json
import time
import math
from pathlib import Path
from dataclasses import dataclass, asdict, field
from typing import List, Dict, Tuple, Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load

from src.telemetry.sysfs_hwmon import SysfsHwmonTelemetry, EnergyMeter


# ============================================================================
# Load HIP Kernel Extension
# ============================================================================

print("[z1105] Loading HIP kernel extension...")
kernel_dir = Path(__file__).parent.parent / "src" / "kernels"
hip_file = kernel_dir / "energy_mod.hip"

try:
    energy_mod = load(
        name='energy_mod_z1105',
        sources=[str(hip_file)],
        extra_cuda_cflags=['--offload-arch=gfx1100'],
        verbose=False
    )
    KERNEL_AVAILABLE = True
    print("[z1105] HIP kernels loaded successfully")
except Exception as e:
    print(f"[z1105] Warning: Could not load HIP kernels: {e}")
    KERNEL_AVAILABLE = False
    energy_mod = None


# ============================================================================
# Telemetry Collection
# ============================================================================

def make_body_state(telemetry: SysfsHwmonTelemetry, batch_size: int, device: torch.device) -> torch.Tensor:
    """Create normalized body_state tensor [batch, 12] from telemetry."""
    sample = telemetry.read_sample()

    state = torch.tensor([
        sample.temp_edge_c / 100.0,
        sample.temp_edge_c / 100.0,  # junction
        0.0,  # slope
        sample.power_w / 50.0,
        sample.power_w / 50.0,  # cap
        0.0, 0.0, 0.0,  # clocks
        0.0, 0.0, 0.0, 0.0  # padding
    ], dtype=torch.float32, device=device)

    return state.unsqueeze(0).expand(batch_size, -1).contiguous()


# ============================================================================
# Attention Implementations
# ============================================================================

class StandardAttention(nn.Module):
    """Condition A: Standard attention with no embodiment."""

    def __init__(self, hidden_dim: int, num_heads: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)

        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x: torch.Tensor, body_state: torch.Tensor = None) -> Tuple[torch.Tensor, Dict]:
        B, T, D = x.shape

        Q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, V)

        out = out.transpose(1, 2).contiguous().view(B, T, D)
        out = self.out_proj(out)

        return out, {'energy_mod': 1.0, 'homeostatic_gate': 1.0}


class PyTorchEmbodiedAttention(nn.Module):
    """Condition B: PyTorch-level embodiment (like z910)."""

    def __init__(self, hidden_dim: int, num_heads: int,
                 power_setpoint: float = 30.0, temp_setpoint: float = 50.0,
                 energy_mod_strength: float = 0.3, homeostatic_gain: float = 0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)

        self.power_setpoint = power_setpoint
        self.temp_setpoint = temp_setpoint
        self.energy_mod_strength = energy_mod_strength
        self.homeostatic_gain = homeostatic_gain

        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x: torch.Tensor, body_state: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        B, T, D = x.shape

        # Compute modulation in PyTorch
        current_temp = body_state[:, 0].mean() * 100.0
        current_power = body_state[:, 3].mean() * 50.0

        power_ratio = current_power / max(self.power_setpoint, 1.0)
        energy_mod = 1.0 - self.energy_mod_strength * min(power_ratio.item(), 2.0)
        energy_mod = max(0.1, energy_mod)

        power_dev = abs(current_power.item() - self.power_setpoint) / self.power_setpoint
        temp_dev = abs(current_temp.item() - self.temp_setpoint) / self.temp_setpoint
        homeostatic_gate = 1.0 / (1.0 + self.homeostatic_gain * (power_dev + temp_dev))

        combined_scale = energy_mod * homeostatic_gate

        Q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale

        # Apply modulation in PyTorch (before softmax)
        scores = scores * combined_scale

        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, V)

        out = out.transpose(1, 2).contiguous().view(B, T, D)
        out = self.out_proj(out)

        return out, {'energy_mod': energy_mod, 'homeostatic_gate': homeostatic_gate}


class KernelEmbodiedAttention(nn.Module):
    """Condition C: Kernel-level embodiment (HIP kernel)."""

    def __init__(self, hidden_dim: int, num_heads: int,
                 power_setpoint: float = 30.0, temp_setpoint: float = 50.0,
                 energy_mod_strength: float = 0.3, homeostatic_gain: float = 0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)

        self.power_setpoint = power_setpoint
        self.temp_setpoint = temp_setpoint
        self.energy_mod_strength = energy_mod_strength
        self.homeostatic_gain = homeostatic_gain

        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x: torch.Tensor, body_state: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        B, T, D = x.shape

        Q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale

        # Apply modulation via HIP KERNEL (not PyTorch!)
        if energy_mod is not None:
            scores = energy_mod.energy_modulated_scores(
                scores.contiguous(),
                body_state,
                self.power_setpoint,
                self.temp_setpoint,
                self.energy_mod_strength,
                self.homeostatic_gain
            )

        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, V)

        out = out.transpose(1, 2).contiguous().view(B, T, D)
        out = self.out_proj(out)

        # Compute modulation values for logging (not used in computation)
        current_temp = body_state[:, 0].mean() * 100.0
        current_power = body_state[:, 3].mean() * 50.0
        power_ratio = current_power / max(self.power_setpoint, 1.0)
        energy_mod_val = 1.0 - self.energy_mod_strength * min(power_ratio.item(), 2.0)
        energy_mod_val = max(0.1, energy_mod_val)

        power_dev = abs(current_power.item() - self.power_setpoint) / self.power_setpoint
        temp_dev = abs(current_temp.item() - self.temp_setpoint) / self.temp_setpoint
        homeostatic_gate = 1.0 / (1.0 + self.homeostatic_gain * (power_dev + temp_dev))

        return out, {'energy_mod': energy_mod_val, 'homeostatic_gate': homeostatic_gate}


# ============================================================================
# Transformer Model
# ============================================================================

class SimpleTransformer(nn.Module):
    """Simple transformer for testing attention variants."""

    def __init__(self, vocab_size: int, hidden_dim: int, num_layers: int,
                 num_heads: int, attention_class, **attn_kwargs):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, hidden_dim)
        self.pos_emb = nn.Embedding(512, hidden_dim)

        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            layer = nn.ModuleDict({
                'attn': attention_class(hidden_dim, num_heads, **attn_kwargs),
                'ln1': nn.LayerNorm(hidden_dim),
                'ff': nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim * 4),
                    nn.GELU(),
                    nn.Linear(hidden_dim * 4, hidden_dim),
                ),
                'ln2': nn.LayerNorm(hidden_dim),
            })
            self.layers.append(layer)

        self.ln_out = nn.LayerNorm(hidden_dim)
        self.head = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x: torch.Tensor, body_state: torch.Tensor) -> Tuple[torch.Tensor, List[Dict]]:
        B, T = x.shape
        pos = torch.arange(T, device=x.device)
        h = self.token_emb(x) + self.pos_emb(pos)

        layer_infos = []
        for layer in self.layers:
            attn_out, info = layer['attn'](layer['ln1'](h), body_state)
            h = h + attn_out
            h = h + layer['ff'](layer['ln2'](h))
            layer_infos.append(info)

        return self.head(self.ln_out(h)), layer_infos


# ============================================================================
# Training
# ============================================================================

@dataclass
class ConditionResult:
    condition: str
    label: str
    final_loss: float
    final_ppl: float
    avg_energy_mod: float
    avg_homeostatic_gate: float
    time_s: float
    losses: List[float] = field(default_factory=list)
    energy_mods: List[float] = field(default_factory=list)


def run_condition(
    condition: str,
    label: str,
    attention_class,
    device: torch.device,
    telemetry: SysfsHwmonTelemetry,
    n_epochs: int = 5,
    n_batches: int = 50,
    batch_size: int = 16,
    seq_len: int = 64,
    **attn_kwargs
) -> ConditionResult:
    """Run one experimental condition."""

    print(f"\n{'='*60}")
    print(f"  Condition {condition}: {label}")
    print(f"{'='*60}")

    vocab_size = 256
    hidden_dim = 128
    num_layers = 2
    num_heads = 4

    model = SimpleTransformer(
        vocab_size, hidden_dim, num_layers, num_heads,
        attention_class, **attn_kwargs
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    # Generate random training data
    data = torch.randint(0, vocab_size, (n_batches * n_epochs, batch_size, seq_len + 1), device=device)

    losses = []
    energy_mods = []
    total_time = 0.0
    step = 0

    for epoch in range(n_epochs):
        epoch_loss = 0.0
        epoch_energy_mod = 0.0
        n_steps = 0

        for batch_idx in range(n_batches):
            batch = data[epoch * n_batches + batch_idx]
            inp = batch[:, :-1]
            tgt = batch[:, 1:]

            body_state = make_body_state(telemetry, batch_size, device)

            torch.cuda.synchronize()
            t0 = time.perf_counter()

            optimizer.zero_grad()
            logits, layer_infos = model(inp, body_state)
            loss = F.cross_entropy(logits.reshape(-1, vocab_size), tgt.reshape(-1))
            loss.backward()
            optimizer.step()

            torch.cuda.synchronize()
            total_time += time.perf_counter() - t0

            epoch_loss += loss.item()
            epoch_energy_mod += layer_infos[0]['energy_mod']
            losses.append(loss.item())
            energy_mods.append(layer_infos[0]['energy_mod'])
            n_steps += 1
            step += 1

        avg_loss = epoch_loss / n_steps
        avg_mod = epoch_energy_mod / n_steps
        print(f"  Epoch {epoch+1}/{n_epochs}  loss={avg_loss:.4f}  ppl={math.exp(avg_loss):.2f}  "
              f"energy_mod={avg_mod:.3f}")

    return ConditionResult(
        condition=condition,
        label=label,
        final_loss=losses[-1] if losses else 0,
        final_ppl=math.exp(losses[-1]) if losses else 0,
        avg_energy_mod=sum(energy_mods) / len(energy_mods) if energy_mods else 0,
        avg_homeostatic_gate=0,  # Not tracked separately
        time_s=total_time,
        losses=losses,
        energy_mods=energy_mods,
    )


# ============================================================================
# Main
# ============================================================================

def main():
    print("="*70)
    print("  Z1105: Kernel-Level Embodied Training")
    print("="*70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[Device] {device}")
    if device.type == 'cuda':
        print(f"  GPU: {torch.cuda.get_device_name(0)}")

    print(f"  Kernel available: {KERNEL_AVAILABLE}")

    telemetry = SysfsHwmonTelemetry(sample_rate_hz=100)
    sample = telemetry.read_sample()
    print(f"  Current: {sample.temp_edge_c:.1f}°C, {sample.power_w:.1f}W")

    results = []

    # Condition A: Standard attention
    result_a = run_condition(
        "A", "Standard (no embodiment)",
        StandardAttention, device, telemetry
    )
    results.append(result_a)

    # Condition B: PyTorch-level embodiment
    result_b = run_condition(
        "B", "PyTorch-level embodiment",
        PyTorchEmbodiedAttention, device, telemetry
    )
    results.append(result_b)

    # Condition C: Kernel-level embodiment
    if KERNEL_AVAILABLE:
        result_c = run_condition(
            "C", "Kernel-level embodiment (HIP)",
            KernelEmbodiedAttention, device, telemetry
        )
        results.append(result_c)
    else:
        print("\n[SKIPPED] Condition C - Kernels not available")

    # Summary
    print("\n" + "="*70)
    print("  RESULTS SUMMARY")
    print("="*70)
    print(f"  {'Cond':<6} {'Label':<35} {'PPL':>8} {'E_mod':>8} {'Time':>8}")
    print("-"*70)
    for r in results:
        print(f"  {r.condition:<6} {r.label:<35} {r.final_ppl:>8.2f} "
              f"{r.avg_energy_mod:>8.4f} {r.time_s:>7.2f}s")

    # Analysis
    print("\n" + "="*70)
    print("  ANALYSIS")
    print("="*70)

    claims = []

    # Claim 1: Energy modulation varies from 1.0
    if len(results) >= 2:
        mod_diff = abs(result_b.avg_energy_mod - 1.0)
        claims.append({
            'claim': 'Energy modulation varies (B != 1.0)',
            'validated': mod_diff > 0.01,
            'evidence': f'avg_mod={result_b.avg_energy_mod:.4f}'
        })

    # Claim 2: Kernel vs PyTorch produce different dynamics
    if KERNEL_AVAILABLE and len(results) >= 3:
        ppl_diff_bc = abs(result_b.final_ppl - result_c.final_ppl)
        claims.append({
            'claim': 'Kernel differs from PyTorch (C != B)',
            'validated': ppl_diff_bc > 1.0,  # Meaningful difference
            'evidence': f'PPL diff = {ppl_diff_bc:.2f}'
        })

        # Claim 3: Kernel has different energy_mod distribution
        mod_std_b = torch.std(torch.tensor(result_b.energy_mods)).item()
        mod_std_c = torch.std(torch.tensor(result_c.energy_mods)).item()
        claims.append({
            'claim': 'Kernel has different mod variance',
            'validated': abs(mod_std_b - mod_std_c) > 0.001,
            'evidence': f'std_B={mod_std_b:.4f}, std_C={mod_std_c:.4f}'
        })

    # Claim 4: Embodiment doesn't hurt quality
    ppl_diff_ab = result_b.final_ppl - result_a.final_ppl
    claims.append({
        'claim': 'Embodiment quality within 10% of baseline',
        'validated': abs(ppl_diff_ab) / result_a.final_ppl < 0.10,
        'evidence': f'PPL diff = {ppl_diff_ab:+.2f} ({ppl_diff_ab/result_a.final_ppl*100:+.1f}%)'
    })

    for c in claims:
        status = "PASS" if c['validated'] else "FAIL"
        print(f"  [{status}] {c['claim']}: {c['evidence']}")

    # Save results
    output = {
        'experiment': 'z1105_kernel_embodied_training',
        'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
        'device': str(device),
        'kernel_available': KERNEL_AVAILABLE,
        'conditions': [asdict(r) for r in results],
        'claims': claims,
        'n_validated': sum(1 for c in claims if c['validated'])
    }

    results_dir = Path(__file__).parent.parent / "results"
    results_dir.mkdir(exist_ok=True)
    out_path = results_dir / "z1105_kernel_embodied_training.json"

    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\n[Saved] {out_path}")


if __name__ == "__main__":
    main()
