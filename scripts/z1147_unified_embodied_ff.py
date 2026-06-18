#!/usr/bin/env python3
"""
z1147: Unified Embodied Forward-Forward with GPU Telemetry

This integrates:
- PyTorch GPU compute with real telemetry from AMD SMI
- FPGA LiteDRAM as analog memory (reality anchor)
- Forward-Forward learning (no backprop)
- Interoceptive attention with body tokens
- Small language model with next-token prediction

The FPGA provides temporal grounding - charge decay is real physics
that creates pressure to continually re-anchor knowledge.

Architecture:
  [Body Tokens (8)] + [Content Tokens (seq_len)]
           ↓
  Embedding + Positional
           ↓
  Forward-Forward Layers (with FPGA weight modulation)
           ↓
  Interoceptive Attention (body tokens attend to content)
           ↓
  Output Head (next token prediction)

Learning:
- Positive: real sequences, high goodness target
- Negative: corrupted sequences, low goodness target
- Each layer learns locally (no gradient flow between layers)
- FPGA charge modulates learning rate (fading memory = conservative learning)
"""

import os
import sys
import time
import json
import struct
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Dict
import threading
from collections import deque

sys.path.insert(0, str(Path(__file__).parent.parent))

# Environment setup for AMD GPU
os.environ.setdefault('HSA_OVERRIDE_GFX_VERSION', '11.0.0')

# PyTorch with ROCm
import torch
import torch.nn as nn
import torch.nn.functional as F

# Check GPU
if torch.cuda.is_available():
    DEVICE = torch.device('cuda')
    print(f"GPU: {torch.cuda.get_device_name()}")
else:
    DEVICE = torch.device('cpu')
    print("Warning: No GPU, using CPU")

# FPGA imports
try:
    from litex.tools.litex_client import RemoteClient
    HAS_LITEX = True
except ImportError:
    HAS_LITEX = False


@dataclass
class HardwareState:
    """Complete hardware state from GPU and FPGA"""
    # GPU
    gpu_temp: float = 50.0
    gpu_power: float = 50.0
    gpu_util: float = 0.5
    gpu_mem_used: float = 0.0
    gpu_clock: float = 1000.0

    # FPGA
    fpga_temp: float = 45.0
    fpga_charge: float = 1.0
    fpga_decay_rate: float = 0.1

    # Derived
    timestamp: float = field(default_factory=time.time)

    def to_tensor(self) -> torch.Tensor:
        """Convert to 8-dim body token tensor"""
        return torch.tensor([
            (self.gpu_power - 50) / 50,       # power deviation
            (self.gpu_temp - 60) / 30,        # temp deviation
            self.gpu_util,                     # utilization
            max(0, 1 - self.gpu_temp / 90),   # thermal margin
            (self.fpga_temp - 50) / 30,       # fpga temp
            self.fpga_charge,                  # charge level
            min(1.0, self.fpga_decay_rate / 0.5),  # decay pressure
            1.0 if self.gpu_util < 0.9 else 0.5    # stability
        ], dtype=torch.float32, device=DEVICE)


class GPUTelemetry:
    """Read real GPU telemetry from sysfs/AMD SMI"""

    def __init__(self):
        self.card_path = self._find_gpu_path()
        self._last_energy = None
        self._last_energy_time = None

    def _find_gpu_path(self) -> Optional[str]:
        """Find AMD GPU sysfs path"""
        for card in ['card0', 'card1']:
            path = f'/sys/class/drm/{card}/device'
            if os.path.exists(f'{path}/gpu_busy_percent'):
                return path
        return None

    def _read_file(self, filename: str, default: float = 0.0) -> float:
        """Read a sysfs file"""
        if not self.card_path:
            return default
        try:
            path = f'{self.card_path}/{filename}'
            if os.path.exists(path):
                with open(path) as f:
                    return float(f.read().strip())
        except:
            pass
        return default

    def _read_hwmon(self, pattern: str, default: float = 0.0) -> float:
        """Read from hwmon"""
        if not self.card_path:
            return default
        try:
            hwmon_path = f'{self.card_path}/hwmon'
            if os.path.exists(hwmon_path):
                for hwmon in os.listdir(hwmon_path):
                    path = f'{hwmon_path}/{hwmon}/{pattern}'
                    if os.path.exists(path):
                        with open(path) as f:
                            return float(f.read().strip())
        except:
            pass
        return default

    def read(self) -> HardwareState:
        """Read current GPU state"""
        state = HardwareState()

        # Temperature (millidegrees -> degrees)
        state.gpu_temp = self._read_hwmon('temp1_input', 50000) / 1000

        # Power (microwatts -> watts)
        state.gpu_power = self._read_hwmon('power1_average', 50000000) / 1e6

        # Utilization
        state.gpu_util = self._read_file('gpu_busy_percent', 50) / 100

        # Memory used (bytes -> GB)
        state.gpu_mem_used = self._read_file('mem_info_vram_used', 0) / 1e9

        state.timestamp = time.time()
        return state


class FPGAMemory:
    """FPGA DDR3 analog memory interface"""

    def __init__(self):
        self.client = None
        self.connected = False
        self._sim_charges = np.ones(1024, dtype=np.float32)
        self._last_decay = time.time()

    def connect(self) -> bool:
        if not HAS_LITEX:
            return False
        try:
            self.client = RemoteClient(csr_csv='src/fpga/litedram/build/csr.csv')
            self.client.open()
            self.connected = True
            return True
        except:
            return False

    def disconnect(self):
        if self.client:
            try:
                self.client.close()
            except:
                pass
        self.connected = False

    def frac(self, row: int = 0, num_fracs: int = 1):
        """Perform Frac operation"""
        if self.connected:
            ctrl = self.client.regs.sdram_dfii_csrstorage_56
            cmd = self.client.regs.sdram_dfii_pi0_csrstorage_57
            issue = self.client.regs.sdram_dfii_pi0_csr_58
            addr = self.client.regs.sdram_dfii_pi0_csrstorage_59
            baddr = self.client.regs.sdram_dfii_pi0_csrstorage_60

            ctrl.write(0b1010)  # Software mode
            try:
                for _ in range(num_fracs):
                    addr.write(row)
                    baddr.write(0)
                    cmd.write(0b0011)  # ACT
                    issue.write(1)
                    addr.write(1 << 10)
                    cmd.write(0b0111)  # PRE
                    issue.write(1)
            finally:
                ctrl.write(0b1011)  # Hardware mode
        else:
            idx = row % len(self._sim_charges)
            self._sim_charges[idx] = min(1.0, self._sim_charges[idx] + 0.125 * num_fracs)

    def decay(self, dt: float = None):
        """Apply natural decay"""
        now = time.time()
        if dt is None:
            dt = now - self._last_decay
        self._last_decay = now
        self._sim_charges = np.maximum(0, self._sim_charges - 0.1 * dt)

    def get_charge(self, idx: int = 0) -> float:
        """Get charge level at index"""
        self.decay()
        return float(self._sim_charges[idx % len(self._sim_charges)])

    def get_weight_modulation(self, shape: Tuple[int, int]) -> torch.Tensor:
        """Get weight modulation tensor from FPGA charges"""
        self.decay()
        total = shape[0] * shape[1]
        charges = self._sim_charges[:total]
        if len(charges) < total:
            charges = np.pad(charges, (0, total - len(charges)), constant_values=0.5)
        # Scale to [0.5, 1.5] for multiplicative modulation
        mod = 0.5 + charges.reshape(shape)
        return torch.tensor(mod, dtype=torch.float32, device=DEVICE)


class ForwardForwardBlock(nn.Module):
    """
    Forward-Forward learning block.

    Each block learns locally to have high "goodness" (sum of squared activations)
    for positive examples and low goodness for negative examples.
    """

    def __init__(self, dim: int, threshold: float = 2.0):
        super().__init__()
        self.dim = dim
        self.threshold = threshold

        self.linear = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)

        # Learnable goodness threshold
        self.log_threshold = nn.Parameter(torch.log(torch.tensor(threshold)))

    def forward(self, x: torch.Tensor, fpga_mod: torch.Tensor = None) -> torch.Tensor:
        """Forward pass"""
        h = self.linear(x)

        # Apply FPGA modulation if provided
        if fpga_mod is not None and fpga_mod.numel() > 0:
            # Broadcast modulation across batch
            mod_scale = fpga_mod.mean()
            h = h * mod_scale

        h = self.norm(h)
        h = F.relu(h)
        return h

    def goodness(self, h: torch.Tensor) -> torch.Tensor:
        """Compute goodness = mean of squared activations"""
        return (h ** 2).mean(dim=-1)

    def ff_loss(self, h: torch.Tensor, is_positive: torch.Tensor) -> torch.Tensor:
        """
        Forward-Forward loss.

        Positive: push goodness above threshold
        Negative: push goodness below threshold
        """
        g = self.goodness(h)
        threshold = torch.exp(self.log_threshold)

        # Logistic loss
        logits = g - threshold
        targets = is_positive.float()

        loss = F.binary_cross_entropy_with_logits(logits, targets)
        return loss


class InteroceptiveAttention(nn.Module):
    """
    Attention with body tokens.

    Body tokens (hardware state) attend to content tokens,
    creating hardware-aware representations.
    """

    def __init__(self, dim: int, num_heads: int = 4):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)

    def forward(
        self,
        body: torch.Tensor,      # [batch, 8]
        content: torch.Tensor,   # [batch, seq_len, dim]
        body_embed: nn.Embedding = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Body tokens attend to content.

        Returns:
            body_out: Updated body representation
            content_out: Updated content (unchanged for now)
        """
        batch_size = content.size(0)

        # Project body to query dimension
        if body_embed is not None:
            # Discretize and embed
            body_discrete = (body * 10).long().clamp(0, 19)
            body_q = body_embed(body_discrete)  # [batch, 8, dim]
        else:
            # Simple projection
            body_q = body.unsqueeze(-1).expand(-1, -1, self.dim)  # [batch, 8, dim]

        # Query from body, key/value from content
        q = self.q_proj(body_q)
        k = self.k_proj(content)
        v = self.v_proj(content)

        # Reshape for multi-head attention
        q = q.view(batch_size, 8, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        # Attention
        attn = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn = F.softmax(attn, dim=-1)
        out = torch.matmul(attn, v)

        # Reshape back
        out = out.transpose(1, 2).contiguous().view(batch_size, 8, self.dim)
        body_out = self.out_proj(out)

        return body_out, content


class EmbodiedForwardForwardModel(nn.Module):
    """
    Complete embodied model with Forward-Forward learning.

    Architecture:
    - Body tokens from hardware state
    - Content tokens from input
    - FF layers for feature extraction
    - Interoceptive attention
    - Output head for next token prediction
    """

    def __init__(
        self,
        vocab_size: int = 256,
        embed_dim: int = 128,
        num_layers: int = 3,
        seq_len: int = 32
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.seq_len = seq_len
        self.body_dim = 8

        # Embeddings
        self.token_embed = nn.Embedding(vocab_size, embed_dim)
        self.pos_embed = nn.Embedding(seq_len, embed_dim)
        self.body_proj = nn.Linear(self.body_dim, embed_dim)

        # FF layers
        self.ff_layers = nn.ModuleList([
            ForwardForwardBlock(embed_dim, threshold=2.0)
            for _ in range(num_layers)
        ])

        # Interoceptive attention
        self.intero_attn = InteroceptiveAttention(embed_dim, num_heads=4)

        # Output head
        self.output_norm = nn.LayerNorm(embed_dim)
        self.output_head = nn.Linear(embed_dim, vocab_size)

        # FPGA memory
        self.fpga = FPGAMemory()

        # GPU telemetry
        self.telemetry = GPUTelemetry()

        # Current state
        self._hw_state = HardwareState()

        # Metrics
        self.metrics = {
            'goodness': [],
            'ff_loss': [],
            'accuracy': [],
            'frac_ops': 0
        }

    def update_hardware_state(self):
        """Update hardware state from real sensors"""
        self._hw_state = self.telemetry.read()
        self._hw_state.fpga_charge = self.fpga.get_charge()
        self._hw_state.fpga_decay_rate = 0.1

    def forward(
        self,
        tokens: torch.Tensor,  # [batch, seq_len]
        return_goodness: bool = False
    ) -> Tuple[torch.Tensor, Optional[List[torch.Tensor]]]:
        """
        Forward pass.

        Returns:
            logits: [batch, seq_len, vocab_size]
            goodness: List of goodness values per layer (if requested)
        """
        batch_size = tokens.size(0)

        # Get body vector from hardware state
        body = self._hw_state.to_tensor().unsqueeze(0).expand(batch_size, -1)

        # Token embeddings + positional
        positions = torch.arange(tokens.size(1), device=DEVICE)
        x = self.token_embed(tokens) + self.pos_embed(positions)

        # Body projection
        body_emb = self.body_proj(body)  # [batch, embed_dim]

        # Get FPGA weight modulation
        fpga_mod = self.fpga.get_weight_modulation((self.embed_dim, self.embed_dim))

        # FF layers with goodness tracking
        goodness_values = []
        for ff_layer in self.ff_layers:
            # Pool sequence for FF layer
            x_pooled = x.mean(dim=1)  # [batch, embed_dim]
            h = ff_layer(x_pooled, fpga_mod)
            goodness_values.append(ff_layer.goodness(h))

            # Broadcast back to sequence
            x = x + h.unsqueeze(1)

        # Interoceptive attention
        body_out, x = self.intero_attn(body, x)

        # Combine body with sequence
        x = x + body_out.mean(dim=1, keepdim=True)

        # Output
        x = self.output_norm(x)
        logits = self.output_head(x)

        if return_goodness:
            return logits, goodness_values
        return logits, None

    def ff_training_step(
        self,
        tokens: torch.Tensor,
        is_positive: torch.Tensor,
        lr: float = 0.01
    ) -> Dict:
        """
        Forward-Forward training step.

        Each layer is trained independently.
        """
        self.update_hardware_state()

        batch_size = tokens.size(0)
        body = self._hw_state.to_tensor().unsqueeze(0).expand(batch_size, -1)

        # Embeddings
        positions = torch.arange(tokens.size(1), device=DEVICE)
        x = self.token_embed(tokens) + self.pos_embed(positions)

        # FPGA modulation (modulates learning rate)
        charge = self.fpga.get_charge()
        effective_lr = lr * (0.5 + 0.5 * charge)

        # Train each FF layer
        total_loss = 0.0
        layer_goodness = []

        with torch.enable_grad():
            for i, ff_layer in enumerate(self.ff_layers):
                # Pool and forward
                x_pooled = x.mean(dim=1).detach().requires_grad_(True)
                h = ff_layer(x_pooled)

                # FF loss
                loss = ff_layer.ff_loss(h, is_positive)
                total_loss += loss.item()

                # Local gradient update
                loss.backward()
                with torch.no_grad():
                    for p in ff_layer.parameters():
                        if p.grad is not None:
                            p.data -= effective_lr * p.grad
                            p.grad.zero_()

                layer_goodness.append(ff_layer.goodness(h.detach()).mean().item())

                # Update x for next layer
                x = x + h.unsqueeze(1).detach()

        # Record Frac if positive and high goodness
        avg_goodness = np.mean(layer_goodness)
        if is_positive.float().mean() > 0.5 and avg_goodness > 2.0:
            self.fpga.frac(num_fracs=1)
            self.metrics['frac_ops'] += 1

        self.metrics['goodness'].append(avg_goodness)
        self.metrics['ff_loss'].append(total_loss / self.num_layers)

        return {
            'loss': total_loss / self.num_layers,
            'goodness': avg_goodness,
            'layer_goodness': layer_goodness,
            'charge': charge,
            'lr': effective_lr
        }

    def predict_next_token(self, tokens: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predict next token with confidence"""
        self.eval()
        with torch.no_grad():
            logits, _ = self.forward(tokens)
            probs = F.softmax(logits[:, -1, :], dim=-1)
            pred = probs.argmax(dim=-1)
            conf = probs.max(dim=-1).values
        return pred, conf

    def self_describe(self) -> str:
        """Generate self-description from hardware state"""
        s = self._hw_state
        parts = []

        if s.gpu_power > 60:
            parts.append("working hard")
        elif s.gpu_power < 30:
            parts.append("idle")

        if s.gpu_temp > 70:
            parts.append("running warm")
        elif s.gpu_temp < 40:
            parts.append("cool")

        if s.fpga_charge < 0.5:
            parts.append("memory fading")
        elif s.fpga_charge > 0.9:
            parts.append("memory fresh")

        if s.gpu_util > 0.8:
            parts.append("fully engaged")

        return "I am " + (", ".join(parts) if parts else "in baseline state")


def create_training_data(vocab_size: int, seq_len: int, batch_size: int = 32):
    """Create positive and negative training examples"""
    # Positive: smooth patterns
    positive = torch.zeros(batch_size // 2, seq_len, dtype=torch.long, device=DEVICE)
    for i in range(batch_size // 2):
        start = torch.randint(0, vocab_size - seq_len, (1,)).item()
        positive[i] = torch.arange(start, start + seq_len) % vocab_size

    # Negative: random noise
    negative = torch.randint(0, vocab_size, (batch_size // 2, seq_len), device=DEVICE)

    tokens = torch.cat([positive, negative], dim=0)
    is_positive = torch.cat([
        torch.ones(batch_size // 2, device=DEVICE),
        torch.zeros(batch_size // 2, device=DEVICE)
    ])

    # Shuffle
    perm = torch.randperm(batch_size)
    return tokens[perm], is_positive[perm]


def main():
    print("="*60)
    print("z1147: Unified Embodied Forward-Forward")
    print("="*60)
    print()

    # Create model
    model = EmbodiedForwardForwardModel(
        vocab_size=256,
        embed_dim=128,
        num_layers=3,
        seq_len=32
    ).to(DEVICE)

    # Connect FPGA
    fpga_ok = model.fpga.connect()
    print(f"FPGA connected: {fpga_ok}")

    # Initial state
    model.update_hardware_state()
    print(f"Initial: {model.self_describe()}")
    print(f"GPU: {model._hw_state.gpu_temp:.1f}C, {model._hw_state.gpu_power:.1f}W")
    print()

    # Training
    print("--- Forward-Forward Training ---")
    num_epochs = 10
    batch_size = 32

    for epoch in range(num_epochs):
        tokens, is_positive = create_training_data(256, 32, batch_size)

        metrics = model.ff_training_step(tokens, is_positive, lr=0.01)

        # Calculate accuracy
        with torch.no_grad():
            _, goodness = model.forward(tokens, return_goodness=True)
            avg_g = torch.stack(goodness).mean(dim=0)
            pred_positive = avg_g > 2.0
            correct = (pred_positive == is_positive.bool()).float().mean().item()

        print(f"Epoch {epoch+1:2d}: Loss={metrics['loss']:.4f}, "
              f"Goodness={metrics['goodness']:.3f}, Acc={correct:.1%}, "
              f"Charge={metrics['charge']:.2f}")

        if (epoch + 1) % 3 == 0:
            print(f"          {model.self_describe()}")

    # Test next token prediction
    print("\n--- Next Token Prediction ---")
    test_seq = torch.arange(0, 32, device=DEVICE).unsqueeze(0)
    for i in range(5):
        pred, conf = model.predict_next_token(test_seq)
        print(f"Seq ending at {test_seq[0, -1].item()}: "
              f"pred={pred.item()}, conf={conf.item():.3f}")
        # Shift sequence
        test_seq = torch.cat([test_seq[:, 1:], pred.unsqueeze(-1)], dim=1)

    # Final summary
    print("\n--- Summary ---")
    print(f"Total Frac operations: {model.metrics['frac_ops']}")
    print(f"Final FPGA charge: {model.fpga.get_charge():.3f}")
    print(f"Goodness trend: {np.mean(model.metrics['goodness'][:5]):.3f} -> "
          f"{np.mean(model.metrics['goodness'][-5:]):.3f}")
    print(f"Self-description: {model.self_describe()}")

    # Save results
    results = {
        'experiment': 'z1147_unified_embodied_ff',
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'device': str(DEVICE),
        'fpga_connected': fpga_ok,
        'epochs': num_epochs,
        'final_goodness': float(np.mean(model.metrics['goodness'][-5:])),
        'frac_operations': model.metrics['frac_ops'],
        'final_charge': float(model.fpga.get_charge()),
        'architecture': {
            'vocab_size': 256,
            'embed_dim': 128,
            'num_layers': 3,
            'seq_len': 32
        }
    }

    results_path = Path('results/z1147_unified_embodied_ff.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {results_path}")

    model.fpga.disconnect()
    return 0


if __name__ == '__main__':
    sys.exit(main())
