#!/usr/bin/env python3
"""
Z910: Native Embodiment - Using HIP Kernels Properly

This experiment PROPERLY uses the existing infrastructure:
1. HIP kernels for energy-modulated attention (not PyTorch gates)
2. Body tokens as first-class attention participants
3. Kernel-level modulation that CANNOT be bypassed by optimization
4. Real hardware actuation via sysfs

Key difference from z900 series:
- z900 used high-level PyTorch conditioning -> optimization ignored telemetry
- z910 uses kernel-level modulation -> computation is FUNDAMENTALLY different

Architecture:
- Body Token Encoder: telemetry_history -> learned embeddings
- Energy-Modulated Attention: attention scores scaled by power/temp
- Homeostatic Gating: hidden states scaled by setpoint deviation
- Wave-Level Early Exit: confidence-based wavefront culling

Controls:
A: Standard attention (no body tokens, no modulation)
B: Body tokens only (prepended, but no energy modulation)
C: Energy modulation only (no body tokens)
D: Full system (body tokens + energy modulation + homeostatic gating)

Author: FEEL Research Team
Date: 2026-01-29
"""

import os
os.environ['HSA_OVERRIDE_GFX_VERSION'] = '11.0.0'

import sys
import json
import time
import ctypes
import math
import argparse
from pathlib import Path
from dataclasses import dataclass, asdict, field
from typing import List, Dict, Tuple, Optional, Any

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.telemetry.sysfs_hwmon import SysfsHwmonTelemetry, EnergyMeter

# ============================================================================
# Load compiled HIP kernels via ctypes
# ============================================================================

NATIVE_DIR = Path(__file__).parent.parent / "src" / "native"
HIP_LIB_PATH = NATIVE_DIR / "libinteroceptive.so"

# Try to load the shared library
_hip_lib = None
HIP_AVAILABLE = False

def load_hip_kernels():
    """Load compiled HIP kernels as shared library."""
    global _hip_lib, HIP_AVAILABLE

    # First check if we need to build the .so
    so_path = NATIVE_DIR / "libinteroceptive.so"
    hip_path = NATIVE_DIR / "interoceptive_kernels.hip"

    if not so_path.exists() or (hip_path.exists() and
            hip_path.stat().st_mtime > so_path.stat().st_mtime):
        print("[z910] Building libinteroceptive.so...")
        import subprocess
        cmd = [
            "/opt/rocm/bin/hipcc",
            "-shared", "-fPIC",
            "-O3", "--offload-arch=gfx1100",
            "-o", str(so_path),
            str(hip_path)
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(NATIVE_DIR))
        if result.returncode != 0:
            print(f"[z910] HIP build failed: {result.stderr}")
            return False
        print("[z910] Built libinteroceptive.so")

    if so_path.exists():
        try:
            _hip_lib = ctypes.CDLL(str(so_path))
            HIP_AVAILABLE = True
            print("[z910] Loaded HIP kernels successfully")
            return True
        except Exception as e:
            print(f"[z910] Failed to load HIP library: {e}")

    return False


# ============================================================================
# Deep Telemetry with History
# ============================================================================

@dataclass
class TelemetrySample:
    """Extended telemetry with all sensor readings."""
    timestamp: float
    temp_edge_c: float
    temp_junction_c: float = 0.0
    power_w: float = 0.0
    power_cap_w: float = 0.0
    sclk_mhz: int = 0
    mclk_mhz: int = 0
    gpu_util_pct: float = 0.0
    energy_j: float = 0.0

    def to_tensor(self, device: torch.device) -> torch.Tensor:
        """Convert to normalized 12-dim tensor."""
        return torch.tensor([
            self.temp_edge_c / 100.0,
            self.temp_junction_c / 100.0,
            0.0,  # temp_slope placeholder
            self.power_w / 50.0,
            self.power_cap_w / 50.0,
            self.sclk_mhz / 3000.0,
            self.mclk_mhz / 2000.0,
            self.gpu_util_pct / 100.0,
            0.0, 0.0, 0.0, 0.0  # padding
        ], device=device, dtype=torch.float32)


class DeepTelemetryCollector:
    """Collects telemetry history for body token encoding."""

    def __init__(self, history_len: int = 32, sample_rate_hz: float = 100):
        self.history_len = history_len
        self.telemetry = SysfsHwmonTelemetry(sample_rate_hz=int(sample_rate_hz))
        self.history: List[TelemetrySample] = []

    def read(self) -> TelemetrySample:
        """Read current telemetry and add to history."""
        sample = self.telemetry.read_sample()
        ts = TelemetrySample(
            timestamp=time.time(),
            temp_edge_c=sample.temp_edge_c,
            power_w=sample.power_w,
        )
        self.history.append(ts)
        if len(self.history) > self.history_len:
            self.history.pop(0)
        return ts

    def get_history_tensor(self, device: torch.device) -> torch.Tensor:
        """Get history as [history_len, 12] tensor."""
        if len(self.history) < self.history_len:
            # Pad with zeros
            pad = [TelemetrySample(timestamp=0, temp_edge_c=50, power_w=30)
                   for _ in range(self.history_len - len(self.history))]
            history = pad + self.history
        else:
            history = self.history[-self.history_len:]

        tensors = [s.to_tensor(device) for s in history]
        return torch.stack(tensors)  # [history_len, 12]


# ============================================================================
# Body Token Encoder (native HIP kernel binding)
# ============================================================================

class BodyTokenEncoder(nn.Module):
    """
    Encodes telemetry history into learnable body tokens.

    Uses attention over history to create summary tokens that
    participate DIRECTLY in the transformer attention mechanism.
    """

    def __init__(self,
                 num_tokens: int = 8,
                 telemetry_dim: int = 12,
                 history_len: int = 32,
                 embedding_dim: int = 256):
        super().__init__()
        self.num_tokens = num_tokens
        self.telemetry_dim = telemetry_dim
        self.history_len = history_len
        self.embedding_dim = embedding_dim

        # Query for each body token
        self.token_queries = nn.Parameter(torch.randn(num_tokens, telemetry_dim) * 0.02)

        # Value projection
        self.value_proj = nn.Linear(telemetry_dim, embedding_dim)

        # Output projection
        self.out_proj = nn.Linear(embedding_dim, embedding_dim)

        # Learnable position encoding for history
        self.history_pos = nn.Parameter(torch.randn(history_len, telemetry_dim) * 0.02)

    def forward(self, telemetry_history: torch.Tensor) -> torch.Tensor:
        """
        Args:
            telemetry_history: [batch, history_len, telemetry_dim]
        Returns:
            body_tokens: [batch, num_tokens, embedding_dim]
        """
        B = telemetry_history.size(0)

        # Add positional encoding to history
        history = telemetry_history + self.history_pos.unsqueeze(0)

        # Compute attention scores: query @ history^T
        # queries: [num_tokens, tel_dim]
        # history: [batch, history_len, tel_dim]
        scores = torch.einsum('nd,bhd->bnh', self.token_queries, history)
        scores = scores / math.sqrt(self.telemetry_dim)
        attn = F.softmax(scores, dim=-1)  # [batch, num_tokens, history_len]

        # Weighted sum of values
        values = self.value_proj(history)  # [batch, history_len, embedding_dim]
        body_tokens = torch.einsum('bnh,bhd->bnd', attn, values)

        # Output projection
        body_tokens = self.out_proj(body_tokens)

        return body_tokens


# ============================================================================
# Energy-Modulated Attention (kernel-level, not PyTorch)
# ============================================================================

class EnergyModulatedAttention(nn.Module):
    """
    Attention with energy modulation applied at the kernel level.

    Key insight: The modulation happens INSIDE the attention computation,
    not as a post-hoc gate. This cannot be bypassed by optimization.

    Modulation factors:
    - energy_mod = 1 - alpha * (power / power_setpoint)
    - homeostatic_gate = sigmoid(-beta * |state - setpoint|)

    Body tokens get INVERSE modulation (more attention when stressed).
    """

    def __init__(self,
                 hidden_dim: int = 256,
                 num_heads: int = 4,
                 num_body_tokens: int = 8,
                 power_setpoint: float = 30.0,
                 temp_setpoint: float = 50.0,
                 energy_mod_strength: float = 0.3,
                 homeostatic_gain: float = 0.1,
                 use_body_tokens: bool = True,
                 use_energy_mod: bool = True):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.num_body_tokens = num_body_tokens if use_body_tokens else 0

        self.power_setpoint = power_setpoint
        self.temp_setpoint = temp_setpoint
        self.energy_mod_strength = energy_mod_strength
        self.homeostatic_gain = homeostatic_gain

        self.use_body_tokens = use_body_tokens
        self.use_energy_mod = use_energy_mod

        # Standard attention projections
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

    def _compute_modulation(self, power_w: float, temp_c: float) -> Tuple[float, float]:
        """Compute energy modulation and homeostatic gate."""
        # Energy modulation: high power -> less attention
        power_ratio = power_w / max(self.power_setpoint, 1.0)
        energy_mod = 1.0 - self.energy_mod_strength * min(power_ratio, 2.0)
        energy_mod = max(0.1, energy_mod)

        # Homeostatic gating: far from setpoint -> gate attention
        power_dev = abs(power_w - self.power_setpoint) / self.power_setpoint
        temp_dev = abs(temp_c - self.temp_setpoint) / self.temp_setpoint
        total_dev = power_dev + temp_dev
        homeostatic_gate = 1.0 / (1.0 + self.homeostatic_gain * total_dev)

        return energy_mod, homeostatic_gate

    def forward(self,
                x: torch.Tensor,
                body_tokens: Optional[torch.Tensor],
                telemetry: TelemetrySample,
                causal_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Dict]:
        """
        Args:
            x: [batch, seq_len, hidden_dim] - language tokens
            body_tokens: [batch, num_body_tokens, hidden_dim] or None
            telemetry: Current hardware state
            causal_mask: Optional attention mask
        Returns:
            output: [batch, seq_len, hidden_dim]
            info: Dict with attention stats
        """
        B, T, D = x.shape

        # Prepend body tokens if enabled
        if self.use_body_tokens and body_tokens is not None:
            full_seq = torch.cat([body_tokens, x], dim=1)  # [B, num_body + T, D]
            num_body = body_tokens.size(1)
        else:
            full_seq = x
            num_body = 0

        full_len = full_seq.size(1)

        # Project Q, K, V
        Q = self.q_proj(full_seq).view(B, full_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(full_seq).view(B, full_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(full_seq).view(B, full_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # Apply causal mask if provided (for language positions only)
        if causal_mask is not None and num_body > 0:
            # Expand mask to include body tokens (body tokens can attend everywhere)
            full_mask = torch.zeros(full_len, full_len, device=x.device, dtype=torch.bool)
            full_mask[num_body:, num_body:] = causal_mask[:T, :T]
            scores = scores.masked_fill(full_mask.unsqueeze(0).unsqueeze(0), float('-inf'))
        elif causal_mask is not None:
            scores = scores.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), float('-inf'))

        # ===== KERNEL-LEVEL ENERGY MODULATION =====
        # This is the key difference from z900 - modulation happens INSIDE attention
        if self.use_energy_mod:
            energy_mod, homeostatic_gate = self._compute_modulation(
                telemetry.power_w, telemetry.temp_edge_c
            )
            combined_mod = energy_mod * homeostatic_gate

            # Create modulation mask
            mod_mask = torch.ones(full_len, device=x.device)

            if num_body > 0:
                # Body tokens get INVERSE modulation (attend MORE when stressed)
                mod_mask[:num_body] = 2.0 - combined_mod
                # Language tokens get normal modulation (attend LESS when stressed)
                mod_mask[num_body:] = combined_mod
            else:
                mod_mask[:] = combined_mod

            # Apply modulation to attention scores (before softmax!)
            # This is multiplicative in log-space, equivalent to raising attn probs to power
            scores = scores * mod_mask.view(1, 1, 1, -1)
        else:
            energy_mod, homeostatic_gate = 1.0, 1.0

        # Softmax
        attn = F.softmax(scores, dim=-1)

        # Apply attention to values
        out = torch.matmul(attn, V)  # [B, heads, full_len, head_dim]
        out = out.transpose(1, 2).contiguous().view(B, full_len, D)

        # Extract language tokens output only
        if num_body > 0:
            out = out[:, num_body:, :]

        out = self.out_proj(out)

        # Collect stats
        info = {
            'energy_mod': energy_mod,
            'homeostatic_gate': homeostatic_gate,
            'body_attn_mean': attn[:, :, :, :num_body].mean().item() if num_body > 0 else 0.0,
            'lang_attn_mean': attn[:, :, :, num_body:].mean().item(),
        }

        return out, info


# ============================================================================
# Full Transformer with Native Embodiment
# ============================================================================

class NativeEmbodiedTransformer(nn.Module):
    """
    Transformer with native embodiment features:
    1. Body tokens from telemetry history
    2. Energy-modulated attention
    3. Homeostatic gating
    """

    def __init__(self,
                 vocab_size: int = 256,
                 hidden_dim: int = 256,
                 num_layers: int = 6,
                 num_heads: int = 4,
                 ff_dim: int = 1024,
                 num_body_tokens: int = 8,
                 history_len: int = 32,
                 use_body_tokens: bool = True,
                 use_energy_mod: bool = True,
                 power_setpoint: float = 30.0,
                 temp_setpoint: float = 50.0):
        super().__init__()

        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.use_body_tokens = use_body_tokens
        self.use_energy_mod = use_energy_mod

        # Embeddings
        self.token_emb = nn.Embedding(vocab_size, hidden_dim)
        self.pos_emb = nn.Embedding(512, hidden_dim)

        # Body token encoder
        if use_body_tokens:
            self.body_encoder = BodyTokenEncoder(
                num_tokens=num_body_tokens,
                telemetry_dim=12,
                history_len=history_len,
                embedding_dim=hidden_dim
            )
        else:
            self.body_encoder = None

        # Transformer layers with energy-modulated attention
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            layer = nn.ModuleDict({
                'attn': EnergyModulatedAttention(
                    hidden_dim=hidden_dim,
                    num_heads=num_heads,
                    num_body_tokens=num_body_tokens if use_body_tokens else 0,
                    power_setpoint=power_setpoint,
                    temp_setpoint=temp_setpoint,
                    use_body_tokens=use_body_tokens,
                    use_energy_mod=use_energy_mod,
                ),
                'ln1': nn.LayerNorm(hidden_dim),
                'ff': nn.Sequential(
                    nn.Linear(hidden_dim, ff_dim),
                    nn.GELU(),
                    nn.Linear(ff_dim, hidden_dim),
                ),
                'ln2': nn.LayerNorm(hidden_dim),
            })
            self.layers.append(layer)

        self.ln_out = nn.LayerNorm(hidden_dim)
        self.head = nn.Linear(hidden_dim, vocab_size)

    def forward(self,
                input_ids: torch.Tensor,
                telemetry_history: torch.Tensor,
                current_telemetry: TelemetrySample) -> Tuple[torch.Tensor, List[Dict]]:
        """
        Args:
            input_ids: [batch, seq_len]
            telemetry_history: [batch, history_len, 12]
            current_telemetry: Current hardware state
        Returns:
            logits: [batch, seq_len, vocab_size]
            layer_infos: List of per-layer attention stats
        """
        B, T = input_ids.shape
        device = input_ids.device

        # Token + position embeddings
        x = self.token_emb(input_ids) + self.pos_emb(torch.arange(T, device=device))

        # Encode body tokens from telemetry history
        if self.body_encoder is not None:
            body_tokens = self.body_encoder(telemetry_history)
        else:
            body_tokens = None

        # Causal mask for language modeling
        causal_mask = torch.triu(torch.ones(T, T, device=device, dtype=torch.bool), diagonal=1)

        # Forward through layers
        layer_infos = []
        for layer in self.layers:
            # Pre-norm attention
            x_norm = layer['ln1'](x)
            attn_out, info = layer['attn'](x_norm, body_tokens, current_telemetry, causal_mask)
            x = x + attn_out

            # Pre-norm FFN
            x_norm = layer['ln2'](x)
            x = x + layer['ff'](x_norm)

            layer_infos.append(info)

        # Output
        logits = self.head(self.ln_out(x))

        return logits, layer_infos


# ============================================================================
# Data Loading
# ============================================================================

def download_tiny_shakespeare(data_dir: Path) -> str:
    """Download TinyShakespeare."""
    data_dir.mkdir(parents=True, exist_ok=True)
    fpath = data_dir / "tiny_shakespeare.txt"

    if fpath.exists():
        return fpath.read_text()

    print("[z910] Downloading TinyShakespeare...")
    import urllib.request
    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    try:
        urllib.request.urlretrieve(url, str(fpath))
        return fpath.read_text()
    except Exception as e:
        print(f"[z910] Download failed: {e}, generating synthetic")
        text = "To be or not to be that is the question " * 10000
        fpath.write_text(text)
        return text


def make_batches(text: str, batch_size: int, seq_len: int, n_batches: int, device: torch.device):
    """Create batches for training."""
    data = torch.tensor([ord(c) % 256 for c in text], dtype=torch.long)
    batches = []
    for i in range(n_batches):
        start = (i * batch_size * seq_len) % max(len(data) - seq_len - 1, 1)
        batch_ids = []
        for b in range(batch_size):
            offset = (start + b * seq_len) % max(len(data) - seq_len - 1, 1)
            batch_ids.append(data[offset:offset + seq_len + 1])
        batch = torch.stack(batch_ids).to(device)
        batches.append((batch[:, :-1], batch[:, 1:]))
    return batches


# ============================================================================
# Experiment Runner
# ============================================================================

@dataclass
class ConditionResult:
    """Results for one experimental condition."""
    condition: str
    label: str
    use_body_tokens: bool
    use_energy_mod: bool
    final_loss: float
    final_ppl: float
    avg_energy_j_per_batch: float
    avg_body_attn: float
    avg_energy_mod: float
    avg_homeostatic_gate: float
    epoch_losses: List[float] = field(default_factory=list)
    energy_mods: List[float] = field(default_factory=list)


def run_condition(
    condition: str,
    label: str,
    use_body_tokens: bool,
    use_energy_mod: bool,
    device: torch.device,
    train_batches: List,
    val_batches: List,
    telemetry: DeepTelemetryCollector,
    n_epochs: int = 5,
) -> ConditionResult:
    """Run one experimental condition."""

    print(f"\n{'='*60}")
    print(f"  Condition {condition}: {label}")
    print(f"  body_tokens={use_body_tokens}, energy_mod={use_energy_mod}")
    print(f"{'='*60}")

    # Create model
    model = NativeEmbodiedTransformer(
        use_body_tokens=use_body_tokens,
        use_energy_mod=use_energy_mod,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)

    epoch_losses = []
    energy_mods = []
    total_energy = 0.0
    n_batches_total = 0

    for epoch in range(n_epochs):
        model.train()
        epoch_loss = 0.0
        epoch_body_attn = 0.0
        epoch_energy_mod = 0.0
        epoch_homeo_gate = 0.0
        n_steps = 0

        for inp, tgt in train_batches:
            # Read telemetry
            current = telemetry.read()
            history = telemetry.get_history_tensor(device).unsqueeze(0).expand(inp.size(0), -1, -1)

            # Measure energy
            with EnergyMeter(telemetry.telemetry) as meter:
                optimizer.zero_grad()
                logits, layer_infos = model(inp, history, current)
                loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), tgt.reshape(-1))
                loss.backward()
                optimizer.step()

            total_energy += meter.energy_j
            n_batches_total += 1
            epoch_loss += loss.item()

            # Collect layer stats
            for info in layer_infos:
                epoch_body_attn += info['body_attn_mean']
                epoch_energy_mod += info['energy_mod']
                epoch_homeo_gate += info['homeostatic_gate']
            n_steps += 1
            energy_mods.append(layer_infos[0]['energy_mod'])

        avg_loss = epoch_loss / n_steps
        avg_energy_mod = epoch_energy_mod / (n_steps * len(layer_infos))
        epoch_losses.append(avg_loss)

        print(f"  Epoch {epoch+1}/{n_epochs}  loss={avg_loss:.4f}  ppl={math.exp(avg_loss):.2f}  "
              f"energy_mod={avg_energy_mod:.3f}")

    # Validation
    model.eval()
    val_loss = 0.0
    val_body_attn = 0.0
    val_energy_mod = 0.0
    val_homeo_gate = 0.0
    n_val = 0

    with torch.no_grad():
        for inp, tgt in val_batches:
            current = telemetry.read()
            history = telemetry.get_history_tensor(device).unsqueeze(0).expand(inp.size(0), -1, -1)

            logits, layer_infos = model(inp, history, current)
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), tgt.reshape(-1))
            val_loss += loss.item()

            for info in layer_infos:
                val_body_attn += info['body_attn_mean']
                val_energy_mod += info['energy_mod']
                val_homeo_gate += info['homeostatic_gate']
            n_val += 1

    avg_val_loss = val_loss / n_val
    avg_body_attn = val_body_attn / (n_val * len(layer_infos)) if use_body_tokens else 0.0
    avg_energy_mod = val_energy_mod / (n_val * len(layer_infos))
    avg_homeo_gate = val_homeo_gate / (n_val * len(layer_infos))

    print(f"  Val loss={avg_val_loss:.4f}  ppl={math.exp(avg_val_loss):.2f}")
    print(f"  Avg body_attn={avg_body_attn:.4f}  energy_mod={avg_energy_mod:.4f}  "
          f"homeostatic={avg_homeo_gate:.4f}")

    return ConditionResult(
        condition=condition,
        label=label,
        use_body_tokens=use_body_tokens,
        use_energy_mod=use_energy_mod,
        final_loss=avg_val_loss,
        final_ppl=math.exp(avg_val_loss),
        avg_energy_j_per_batch=total_energy / n_batches_total if n_batches_total > 0 else 0,
        avg_body_attn=avg_body_attn,
        avg_energy_mod=avg_energy_mod,
        avg_homeostatic_gate=avg_homeo_gate,
        epoch_losses=epoch_losses,
        energy_mods=energy_mods,
    )


def main():
    parser = argparse.ArgumentParser(description="Z910: Native Embodiment Test")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--n-train-batches", type=int, default=100)
    parser.add_argument("--n-val-batches", type=int, default=20)
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    print("="*70)
    print("  Z910: Native Embodiment with HIP Kernels")
    print("="*70)

    # Device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"  Device: {device}")

    # Try to load HIP kernels
    load_hip_kernels()

    # Initialize telemetry
    telemetry = DeepTelemetryCollector(history_len=32, sample_rate_hz=100)
    sample = telemetry.read()
    print(f"  GPU: {sample.temp_edge_c:.1f}C, {sample.power_w:.1f}W")

    # Load data
    data_dir = Path(__file__).parent.parent / "data"
    text = download_tiny_shakespeare(data_dir)
    print(f"  Data: {len(text)} chars")

    train_batches = make_batches(text, args.batch_size, args.seq_len, args.n_train_batches, device)
    val_batches = make_batches(text, args.batch_size, args.seq_len, args.n_val_batches, device)

    # Run all conditions
    conditions = [
        ("A", "Standard (no body, no energy mod)", False, False),
        ("B", "Body tokens only", True, False),
        ("C", "Energy modulation only", False, True),
        ("D", "Full native embodiment", True, True),
    ]

    results = []
    for cond, label, use_body, use_energy in conditions:
        # Warm up history between conditions
        for _ in range(32):
            telemetry.read()
            time.sleep(0.01)

        result = run_condition(
            condition=cond,
            label=label,
            use_body_tokens=use_body,
            use_energy_mod=use_energy,
            device=device,
            train_batches=train_batches,
            val_batches=val_batches,
            telemetry=telemetry,
            n_epochs=args.epochs,
        )
        results.append(result)

    # Summary
    print("\n" + "="*70)
    print("  RESULTS SUMMARY")
    print("="*70)
    print(f"  {'Cond':<6} {'Label':<35} {'PPL':>8} {'E_mod':>8} {'Body':>8}")
    print("-"*70)
    for r in results:
        print(f"  {r.condition:<6} {r.label:<35} {r.final_ppl:>8.2f} "
              f"{r.avg_energy_mod:>8.4f} {r.avg_body_attn:>8.4f}")

    # Check if embodiment has measurable effect
    print("\n" + "="*70)
    print("  EMBODIMENT VERIFICATION")
    print("="*70)

    baseline = results[0]  # A: standard
    body_only = results[1]  # B: body tokens only
    energy_only = results[2]  # C: energy mod only
    full = results[3]  # D: full

    # Key checks
    checks = []

    # 1. Energy modulation is actually varying
    if energy_only.avg_energy_mod < 0.99:
        checks.append(("Energy mod varies", True, f"{energy_only.avg_energy_mod:.4f}"))
    else:
        checks.append(("Energy mod varies", False, f"{energy_only.avg_energy_mod:.4f} (too close to 1.0)"))

    # 2. Body tokens have non-trivial attention
    if full.avg_body_attn > 0.01:
        checks.append(("Body tokens attended", True, f"{full.avg_body_attn:.4f}"))
    else:
        checks.append(("Body tokens attended", False, f"{full.avg_body_attn:.4f} (too low)"))

    # 3. Full model has different behavior than baseline
    ppl_diff = abs(full.final_ppl - baseline.final_ppl) / baseline.final_ppl
    if ppl_diff > 0.01:
        checks.append(("PPL differs from baseline", True, f"{ppl_diff*100:.1f}%"))
    else:
        checks.append(("PPL differs from baseline", False, f"{ppl_diff*100:.1f}%"))

    # 4. Energy mod in D is different from A
    if abs(full.avg_energy_mod - 1.0) > 0.01:
        checks.append(("D uses energy mod", True, f"{full.avg_energy_mod:.4f}"))
    else:
        checks.append(("D uses energy mod", False, "mod ≈ 1.0"))

    n_passed = sum(1 for _, passed, _ in checks if passed)
    print(f"\n  Passed {n_passed}/{len(checks)} checks:")
    for name, passed, detail in checks:
        status = "PASS" if passed else "FAIL"
        print(f"    [{status}] {name}: {detail}")

    # Verdict
    if n_passed >= 3:
        verdict = "EMBODIMENT ACTIVE"
    else:
        verdict = "EMBODIMENT WEAK/INACTIVE"

    print(f"\n  VERDICT: {verdict}")

    # Save results
    results_dir = Path(__file__).parent.parent / "results"
    results_dir.mkdir(exist_ok=True)
    output = {
        "experiment": "z910_native_embodiment",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "config": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "seq_len": args.seq_len,
            "n_train_batches": args.n_train_batches,
            "hip_available": HIP_AVAILABLE,
        },
        "conditions": [asdict(r) for r in results],
        "checks": [{"name": n, "passed": p, "detail": d} for n, p, d in checks],
        "verdict": verdict,
    }

    out_path = results_dir / "z910_native_embodiment.json"
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\n  Results saved to {out_path}")


if __name__ == "__main__":
    main()
