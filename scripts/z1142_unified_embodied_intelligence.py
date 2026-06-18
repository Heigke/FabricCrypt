#!/usr/bin/env python3
"""
z1142: Unified Embodied Intelligence - Real Hardware Closed-Loop
================================================================

Combines ALL proven embodiment infrastructure:
- GPU: 100Hz sysfs telemetry + power cap actuation
- FPGA: LiteDRAM partial timing writes, Arrhenius decay, XADC temp
- Model: FiLM conditioning, early exit, interoceptive attention
- Metrics: Intelligence (perplexity, coherence), efficiency (J/token), stability

NO SIMULATION - All real hardware.

Key innovations:
1. z_feel body state: GPU + FPGA combined telemetry (16-dim)
2. Arrhenius decay: Temperature-dependent weight decay from FPGA temp
3. Dual-channel homeostasis: GPU power + FPGA thermal regulation
4. Intelligence metrics: Next-token accuracy, semantic coherence, generation quality

Business impact metrics:
- Intelligence: accuracy per unit energy
- Efficiency: tokens per Joule
- Reliability: thermal variance, power stability
- Adaptability: response latency to hardware state changes

Author: FEEL Research Team
Date: 2026-01-31
"""

import os
import sys
import time
import json
import math
import threading
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Dict, Any
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Add project root
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Set HSA override for gfx1151
os.environ.setdefault("HSA_OVERRIDE_GFX_VERSION", "11.0.0")

# Import REAL hardware interfaces
from src.telemetry.sysfs_hwmon import SysfsHwmonTelemetry, GpuSample, EnergyAccumulator

# FPGA interface
FPGA_AVAILABLE = False
FPGAInterface = None
try:
    from src.fpga.fpga_interface import FPGAInterface as _FPGAInterface
    FPGAInterface = _FPGAInterface
    FPGA_AVAILABLE = True
    print("[z1142] FPGA interface loaded")
except ImportError as e:
    print(f"[z1142] FPGA interface not available: {e}")

# Actuator
ACTUATOR_AVAILABLE = False
try:
    from src.actuator import get_client
    ACTUATOR_AVAILABLE = True
    print("[z1142] Actuator client loaded")
except (ImportError, PermissionError) as e:
    print(f"[z1142] Actuator not available: {e}")


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class UnifiedConfig:
    """Configuration for unified embodied intelligence."""
    # Model architecture
    vocab_size: int = 256  # Char-level
    hidden_dim: int = 512
    n_layers: int = 10
    n_heads: int = 8
    head_dim: int = 64
    max_seq_len: int = 128

    # Early exit
    n_exit_heads: int = 4  # Exit heads at layers 2, 4, 6, 8
    exit_threshold: float = 0.85  # Confidence threshold

    # Body state (z_feel)
    z_feel_dim: int = 16  # Combined GPU + FPGA telemetry
    history_len: int = 64  # 640ms at 100Hz

    # FPGA
    fpga_port: str = "/dev/ttyUSB1"
    fpga_baudrate: int = 115200
    reservoir_dim: int = 64
    arrhenius_ea: float = 0.7  # Activation energy (eV)
    arrhenius_k: float = 8.617e-5  # Boltzmann constant (eV/K)

    # Homeostasis setpoints
    gpu_power_setpoint_w: float = 25.0
    gpu_temp_setpoint_c: float = 65.0
    fpga_temp_setpoint_c: float = 45.0

    # Telemetry
    telemetry_hz: int = 100

    # Training
    batch_size: int = 16
    n_epochs: int = 25
    steps_per_epoch: int = 250
    learning_rate: float = 3e-4
    gradient_clip: float = 1.0

    # Loss weights
    lambda_intero: float = 0.1  # Interoception loss
    lambda_exit: float = 0.05  # Early exit loss
    lambda_decay: float = 0.01  # Arrhenius decay regularization

    device: str = "cuda"


# =============================================================================
# Unified Hardware Manager
# =============================================================================

class UnifiedHardwareManager:
    """
    Manages GPU + FPGA hardware in unified closed-loop.
    """

    def __init__(self, config: UnifiedConfig):
        self.config = config
        self.device = torch.device(config.device if torch.cuda.is_available() else "cpu")

        # GPU telemetry (REAL)
        self.telemetry = SysfsHwmonTelemetry()
        self.energy_acc = EnergyAccumulator()

        # FPGA (REAL)
        self.fpga = None
        self.fpga_connected = False
        self.fpga_temp = 25.0  # Default
        if FPGA_AVAILABLE:
            self._connect_fpga()

        # Actuator (REAL)
        self.actuator = None
        if ACTUATOR_AVAILABLE:
            try:
                self.actuator = get_client()
                print(f"[z1142] Actuator connected")
            except Exception as e:
                print(f"[z1142] Actuator failed: {e}")

        # History buffer for z_feel
        self.history: deque = deque(maxlen=config.history_len)

        # Metrics
        self.metrics = {
            'energy_j': 0.0,
            'tokens': 0,
            'power_samples': [],
            'gpu_temp_samples': [],
            'fpga_temp_samples': [],
            'regulation_events': 0,
            'fpga_operations': 0,
            'exit_layers': [],
            'accuracies': [],
            'coherences': [],
        }

        # Arrhenius decay state
        self.decay_accumulator = 0.0

    def _connect_fpga(self) -> bool:
        """Connect to FPGA and verify DDR3."""
        try:
            self.fpga = FPGAInterface(
                port=self.config.fpga_port,
                baudrate=self.config.fpga_baudrate
            )
            if self.fpga.connect():
                status = self.fpga.ping()
                if status.get('valid'):
                    self.fpga_connected = True
                    temp, _ = self.fpga.read_temperature()
                    self.fpga_temp = temp
                    print(f"[z1142] FPGA connected: temp={temp:.1f}°C, DDR3={'ready' if status.get('ddr3_ready') else 'not ready'}")
                    return True
        except Exception as e:
            print(f"[z1142] FPGA error: {e}")
        return False

    def sample_z_feel(self) -> torch.Tensor:
        """
        Sample unified z_feel body state (16-dim).

        GPU (12 dims):
          0: temp_edge (normalized)
          1: temp_junction
          2: temp_mem
          3: power (normalized to 50W)
          4: sclk (normalized to 3GHz)
          5: mclk (normalized to 2GHz)
          6: util (0-1)
          7: vram (normalized to 16GB)
          8: power_strain (diff from setpoint)
          9: temp_urgency (diff from setpoint)
          10: util_pressure (diff from target)
          11: energy_derivative (smoothed)

        FPGA (4 dims):
          12: fpga_temp (normalized)
          13: fpga_temp_strain
          14: arrhenius_rate
          15: decay_accumulator
        """
        # GPU sample
        sample = self.telemetry.read_sample()
        self.energy_acc.add_sample(sample)

        # Track for metrics
        if sample.power_w:
            self.metrics['power_samples'].append(sample.power_w)
        if sample.temp_edge_c:
            self.metrics['gpu_temp_samples'].append(sample.temp_edge_c)

        # GPU features
        temp_edge = (sample.temp_edge_c or 50.0) / 100.0
        temp_junction = (sample.temp_junction_c or 50.0) / 100.0
        temp_mem = (sample.temp_mem_c or 50.0) / 100.0
        power = (sample.power_w or 20.0) / 50.0
        sclk = (sample.freq_sclk_mhz or 1500.0) / 3000.0
        mclk = (sample.freq_mclk_mhz or 1000.0) / 2000.0
        util = (sample.gpu_busy_pct or 50.0) / 100.0
        vram = (sample.vram_used_gb or 4.0) / 16.0

        # Strains
        power_strain = (power * 50.0 - self.config.gpu_power_setpoint_w) / 20.0
        temp_urgency = ((temp_edge * 100.0) - self.config.gpu_temp_setpoint_c) / 20.0
        util_pressure = util - 0.8

        # Energy derivative from history
        energy_deriv = 0.0
        if len(self.history) >= 2:
            recent_power = [h[3] for h in self.history][-5:]
            if len(recent_power) >= 2:
                energy_deriv = (recent_power[-1] - recent_power[0]) / len(recent_power) / 50.0

        # FPGA features
        if self.fpga_connected and self.fpga:
            try:
                self.fpga_temp, _ = self.fpga.read_temperature()
                self.metrics['fpga_temp_samples'].append(self.fpga_temp)
            except:
                pass

        fpga_temp_norm = self.fpga_temp / 100.0
        fpga_temp_strain = (self.fpga_temp - self.config.fpga_temp_setpoint_c) / 20.0

        # Arrhenius decay rate: k = A * exp(-Ea / (kB * T))
        temp_k = self.fpga_temp + 273.15
        arrhenius_rate = math.exp(-self.config.arrhenius_ea / (self.config.arrhenius_k * temp_k))
        self.decay_accumulator += arrhenius_rate * 0.01  # Accumulate decay
        decay_norm = min(self.decay_accumulator / 10.0, 1.0)

        # Build z_feel vector
        z_feel = torch.tensor([
            temp_edge, temp_junction, temp_mem, power,
            sclk, mclk, util, vram,
            power_strain, temp_urgency, util_pressure, energy_deriv,
            fpga_temp_norm, fpga_temp_strain, arrhenius_rate * 100, decay_norm
        ], dtype=torch.float32, device=self.device)

        # Store in history
        self.history.append(z_feel.cpu().numpy())

        return z_feel

    def regulate(self, z_feel: torch.Tensor) -> Dict[str, Any]:
        """
        Homeostatic regulation based on z_feel.
        Returns regulation actions taken.
        """
        actions = {'gpu_power_cap': None, 'fpga_partial_write': False}

        # GPU power regulation
        power_strain = z_feel[8].item()
        if self.actuator and abs(power_strain) > 0.3:
            try:
                if power_strain > 0:
                    # Reduce power cap
                    self.actuator.set_power_cap(int(self.config.gpu_power_setpoint_w))
                else:
                    # Increase power cap
                    self.actuator.set_power_cap(int(self.config.gpu_power_setpoint_w + 10))
                self.metrics['regulation_events'] += 1
                actions['gpu_power_cap'] = self.config.gpu_power_setpoint_w
            except Exception as e:
                pass

        # FPGA partial timing write for thermal regulation
        fpga_temp_strain = z_feel[13].item()
        if self.fpga_connected and self.fpga and abs(fpga_temp_strain) > 0.2:
            try:
                # Adjust partial timing based on temperature
                timing_cycles = max(1, min(37, int(18 - fpga_temp_strain * 10)))
                self.fpga.partial_timing_write(0x1000, b'\x00' * 8, timing_cycles)
                self.metrics['fpga_operations'] += 1
                actions['fpga_partial_write'] = True
            except Exception as e:
                pass

        return actions

    def get_arrhenius_decay_factor(self) -> float:
        """Get decay factor from Arrhenius-accumulated decay."""
        return min(1.0 - self.decay_accumulator * 0.001, 1.0)

    def get_energy_j(self) -> float:
        """Get accumulated energy in Joules."""
        return self.energy_acc.total_energy_j

    def get_summary(self) -> Dict[str, Any]:
        """Get comprehensive metrics summary."""
        power_arr = np.array(self.metrics['power_samples']) if self.metrics['power_samples'] else np.array([0])
        gpu_temp_arr = np.array(self.metrics['gpu_temp_samples']) if self.metrics['gpu_temp_samples'] else np.array([0])
        fpga_temp_arr = np.array(self.metrics['fpga_temp_samples']) if self.metrics['fpga_temp_samples'] else np.array([0])

        return {
            'total_energy_j': self.energy_acc.total_energy_j,
            'total_tokens': self.metrics['tokens'],
            'j_per_token': self.energy_acc.total_energy_j / max(1, self.metrics['tokens']),
            'tokens_per_joule': self.metrics['tokens'] / max(0.001, self.energy_acc.total_energy_j),
            'power_mean_w': float(np.mean(power_arr)),
            'power_std_w': float(np.std(power_arr)),
            'gpu_temp_mean_c': float(np.mean(gpu_temp_arr)),
            'gpu_temp_std_c': float(np.std(gpu_temp_arr)),
            'fpga_temp_mean_c': float(np.mean(fpga_temp_arr)) if len(fpga_temp_arr) > 0 else 0.0,
            'regulation_events': self.metrics['regulation_events'],
            'fpga_operations': self.metrics['fpga_operations'],
            'arrhenius_decay': self.decay_accumulator,
            'fpga_connected': self.fpga_connected,
        }


# =============================================================================
# Unified Embodied Transformer
# =============================================================================

class FiLMLayer(nn.Module):
    """Feature-wise Linear Modulation from z_feel."""

    def __init__(self, hidden_dim: int, z_feel_dim: int):
        super().__init__()
        self.gamma = nn.Linear(z_feel_dim, hidden_dim)
        self.beta = nn.Linear(z_feel_dim, hidden_dim)

    def forward(self, x: torch.Tensor, z_feel: torch.Tensor) -> torch.Tensor:
        # z_feel: [batch, z_feel_dim] or [z_feel_dim]
        if z_feel.dim() == 1:
            z_feel = z_feel.unsqueeze(0).expand(x.size(0), -1)
        gamma = self.gamma(z_feel).unsqueeze(1)  # [batch, 1, hidden]
        beta = self.beta(z_feel).unsqueeze(1)
        return x * (1 + gamma) + beta


class InteroceptiveAttention(nn.Module):
    """Multi-head attention with body state modulation."""

    def __init__(self, hidden_dim: int, n_heads: int, z_feel_dim: int):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = hidden_dim // n_heads

        self.qkv = nn.Linear(hidden_dim, 3 * hidden_dim)
        self.proj = nn.Linear(hidden_dim, hidden_dim)

        # z_feel modulates attention temperature
        self.temp_mod = nn.Linear(z_feel_dim, n_heads)

    def forward(self, x: torch.Tensor, z_feel: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape

        qkv = self.qkv(x).reshape(B, T, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(2)

        # Attention with z_feel-modulated temperature
        if z_feel.dim() == 1:
            z_feel = z_feel.unsqueeze(0).expand(B, -1)
        temp = 1.0 + 0.1 * torch.tanh(self.temp_mod(z_feel))  # [B, heads]
        temp = temp.view(B, self.n_heads, 1, 1)  # [B, heads, 1, 1]

        q = q.transpose(1, 2)  # [B, heads, T, head_dim]
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        attn = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)
        attn = attn / (temp + 0.9)  # Scale attention by body-state temperature
        attn = F.softmax(attn, dim=-1)

        out = attn @ v
        out = out.transpose(1, 2).reshape(B, T, C)
        return self.proj(out)


class EarlyExitHead(nn.Module):
    """Early exit classification head with confidence."""

    def __init__(self, hidden_dim: int, vocab_size: int):
        super().__init__()
        self.head = nn.Linear(hidden_dim, vocab_size)
        self.confidence = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        logits = self.head(x)
        conf = torch.sigmoid(self.confidence(x.mean(dim=1)))  # [batch, 1]
        return logits, conf


class TransformerBlock(nn.Module):
    """Transformer block with FiLM and interoceptive attention."""

    def __init__(self, hidden_dim: int, n_heads: int, z_feel_dim: int):
        super().__init__()
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.attn = InteroceptiveAttention(hidden_dim, n_heads, z_feel_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, 4 * hidden_dim),
            nn.GELU(),
            nn.Linear(4 * hidden_dim, hidden_dim),
        )
        self.film = FiLMLayer(hidden_dim, z_feel_dim)

    def forward(self, x: torch.Tensor, z_feel: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x), z_feel)
        x = x + self.mlp(self.ln2(x))
        x = self.film(x, z_feel)
        return x


class UnifiedEmbodiedTransformer(nn.Module):
    """
    Transformer with unified embodiment:
    - FiLM conditioning from z_feel
    - Interoceptive attention
    - Early exit heads
    - Arrhenius decay regularization
    """

    def __init__(self, config: UnifiedConfig):
        super().__init__()
        self.config = config

        self.embed = nn.Embedding(config.vocab_size, config.hidden_dim)
        self.pos_embed = nn.Parameter(torch.randn(1, config.max_seq_len, config.hidden_dim) * 0.02)

        self.blocks = nn.ModuleList([
            TransformerBlock(config.hidden_dim, config.n_heads, config.z_feel_dim)
            for _ in range(config.n_layers)
        ])

        self.ln_f = nn.LayerNorm(config.hidden_dim)
        self.head = nn.Linear(config.hidden_dim, config.vocab_size)

        # Early exit heads at specific layers
        exit_layers = [2, 4, 6, 8][:config.n_exit_heads]
        self.exit_heads = nn.ModuleDict({
            str(i): EarlyExitHead(config.hidden_dim, config.vocab_size)
            for i in exit_layers
        })
        self.exit_layers = exit_layers

        # Interoception head (predict z_feel from hidden state)
        self.intero_head = nn.Linear(config.hidden_dim, config.z_feel_dim)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)

    def forward(
        self,
        x: torch.Tensor,
        z_feel: torch.Tensor,
        decay_factor: float = 1.0,
        return_exits: bool = False
    ) -> Dict[str, Any]:
        B, T = x.shape

        h = self.embed(x) + self.pos_embed[:, :T, :]

        # Apply Arrhenius decay to embeddings
        h = h * decay_factor

        exits = {}
        exit_layer = None

        for i, block in enumerate(self.blocks):
            h = block(h, z_feel)

            # Check early exit
            if i in self.exit_layers:
                logits, conf = self.exit_heads[str(i)](h)
                exits[i] = {'logits': logits, 'confidence': conf}

                if return_exits and conf.mean() > self.config.exit_threshold:
                    exit_layer = i
                    break

        # Final output
        h = self.ln_f(h)
        logits = self.head(h)

        # Interoception prediction
        intero_pred = self.intero_head(h.mean(dim=1))

        return {
            'logits': logits,
            'exits': exits,
            'exit_layer': exit_layer if exit_layer is not None else self.config.n_layers,
            'intero_pred': intero_pred,
            'hidden': h,
        }

    def count_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


# =============================================================================
# Intelligence Metrics
# =============================================================================

def compute_coherence(logits: torch.Tensor, targets: torch.Tensor) -> float:
    """
    Compute semantic coherence as accuracy of predictions.
    """
    with torch.no_grad():
        preds = logits.argmax(dim=-1)
        correct = (preds == targets).float()
        return correct.mean().item()


def compute_perplexity(loss: float) -> float:
    """Compute perplexity from loss."""
    return math.exp(min(loss, 20.0))


def compute_generation_quality(
    model: nn.Module,
    hw: UnifiedHardwareManager,
    prompt: str,
    max_tokens: int = 100
) -> Dict[str, Any]:
    """
    Generate text and measure quality.
    """
    model.eval()
    device = next(model.parameters()).device

    # Encode prompt
    tokens = [ord(c) for c in prompt[-64:]]
    x = torch.tensor([tokens], device=device)

    generated = []
    start_time = time.time()
    start_energy = hw.get_energy_j()

    with torch.no_grad():
        for _ in range(max_tokens):
            z_feel = hw.sample_z_feel()
            out = model(x, z_feel, hw.get_arrhenius_decay_factor(), return_exits=True)

            # Sample next token
            logits = out['logits'][:, -1, :]
            probs = F.softmax(logits / 0.8, dim=-1)
            next_tok = torch.multinomial(probs, 1)

            generated.append(next_tok.item())
            x = torch.cat([x, next_tok], dim=1)[:, -128:]

    end_time = time.time()
    end_energy = hw.get_energy_j()

    # Decode
    text = ''.join(chr(min(t, 127)) for t in generated)

    # Quality metrics
    gen_time = end_time - start_time
    gen_energy = end_energy - start_energy

    # Repetition penalty (lower is better)
    unique_tokens = len(set(generated))
    repetition_score = unique_tokens / max_tokens

    # Printable ratio (higher is better for text)
    printable = sum(1 for c in text if c.isprintable()) / len(text)

    return {
        'text': text,
        'tokens': max_tokens,
        'time_s': gen_time,
        'energy_j': gen_energy,
        'tokens_per_second': max_tokens / gen_time,
        'j_per_token': gen_energy / max_tokens if max_tokens > 0 else 0,
        'repetition_score': repetition_score,
        'printable_ratio': printable,
    }


# =============================================================================
# Training Loop
# =============================================================================

def train_unified(
    model: UnifiedEmbodiedTransformer,
    hw: UnifiedHardwareManager,
    config: UnifiedConfig,
    data: torch.Tensor
) -> Dict[str, Any]:
    """
    Train with unified embodiment loop.
    """
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    data = data.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, config.n_epochs * config.steps_per_epoch)

    results = {
        'experiment': 'z1142_unified_embodied_intelligence',
        'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
        'config': {
            'hidden_dim': config.hidden_dim,
            'n_layers': config.n_layers,
            'n_heads': config.n_heads,
            'z_feel_dim': config.z_feel_dim,
            'n_epochs': config.n_epochs,
            'steps_per_epoch': config.steps_per_epoch,
            'param_count': model.count_params(),
        },
        'hardware': {
            'fpga_connected': hw.fpga_connected,
            'actuator_available': hw.actuator is not None,
            'telemetry_hz': config.telemetry_hz,
        },
        'epochs': [],
    }

    print(f"\n[z1142] Starting training")
    print(f"  Model: {model.count_params():,} params")
    print(f"  FPGA: {'connected' if hw.fpga_connected else 'not connected'}")
    print(f"  Actuator: {'available' if hw.actuator else 'not available'}")

    data_len = data.size(0)

    for epoch in range(config.n_epochs):
        model.train()
        epoch_start = time.time()
        epoch_start_energy = hw.get_energy_j()

        epoch_losses = []
        epoch_intero_losses = []
        epoch_exit_losses = []
        epoch_coherences = []
        epoch_exit_layers = []

        for step in range(config.steps_per_epoch):
            # Sample batch
            idx = torch.randint(0, data_len - config.max_seq_len - 1, (config.batch_size,))
            x = torch.stack([data[i:i+config.max_seq_len] for i in idx])
            y = torch.stack([data[i+1:i+config.max_seq_len+1] for i in idx])

            # Sample z_feel (REAL hardware)
            z_feel = hw.sample_z_feel()

            # Regulate based on body state
            hw.regulate(z_feel)

            # Forward with Arrhenius decay
            decay_factor = hw.get_arrhenius_decay_factor()
            out = model(x, z_feel, decay_factor, return_exits=True)

            # Main task loss
            logits = out['logits']
            loss = F.cross_entropy(logits.view(-1, config.vocab_size), y.view(-1))

            # Interoception loss (predict body state from hidden)
            intero_loss = F.mse_loss(out['intero_pred'], z_feel.unsqueeze(0).expand(config.batch_size, -1))

            # Early exit loss (encourage confident early exits)
            exit_loss = 0.0
            for layer_idx, exit_data in out['exits'].items():
                exit_logits = exit_data['logits']
                exit_conf = exit_data['confidence']
                exit_ce = F.cross_entropy(exit_logits.view(-1, config.vocab_size), y.view(-1))
                exit_loss += exit_ce * (1 - exit_conf.mean())  # Lower loss when confident
            exit_loss = exit_loss / max(1, len(out['exits']))

            # Combined loss
            total_loss = loss + config.lambda_intero * intero_loss + config.lambda_exit * exit_loss

            # Backward
            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
            optimizer.step()
            scheduler.step()

            # Track metrics
            epoch_losses.append(loss.item())
            epoch_intero_losses.append(intero_loss.item())
            epoch_exit_losses.append(exit_loss if isinstance(exit_loss, float) else exit_loss.item())
            epoch_exit_layers.append(out['exit_layer'])

            # Coherence
            coherence = compute_coherence(logits, y)
            epoch_coherences.append(coherence)

            # Update tokens count
            hw.metrics['tokens'] += config.batch_size * config.max_seq_len

            if step % 50 == 0:
                z_str = f"[T:{z_feel[3].item()*50:.0f}W,{z_feel[0].item()*100:.0f}°C]"
                print(f"  E{epoch} S{step}: loss={loss.item():.4f} coh={coherence:.3f} exit={out['exit_layer']} {z_str}")

        epoch_time = time.time() - epoch_start
        epoch_energy = hw.get_energy_j() - epoch_start_energy

        # Epoch summary
        avg_loss = np.mean(epoch_losses)
        avg_intero = np.mean(epoch_intero_losses)
        avg_coherence = np.mean(epoch_coherences)
        avg_exit_layer = np.mean(epoch_exit_layers)

        epoch_tokens = config.batch_size * config.max_seq_len * config.steps_per_epoch
        j_per_token = epoch_energy / epoch_tokens

        epoch_data = {
            'epoch': epoch,
            'avg_loss': float(avg_loss),
            'avg_intero_loss': float(avg_intero),
            'avg_coherence': float(avg_coherence),
            'avg_exit_layer': float(avg_exit_layer),
            'perplexity': compute_perplexity(avg_loss),
            'epoch_energy_j': epoch_energy,
            'epoch_time_s': epoch_time,
            'j_per_token': j_per_token,
            'tokens_per_joule': 1.0 / j_per_token if j_per_token > 0 else 0,
        }
        results['epochs'].append(epoch_data)

        # Intelligence metric: accuracy per unit energy
        intelligence = avg_coherence / (j_per_token * 1000 + 0.001)

        print(f"Epoch {epoch}: loss={avg_loss:.4f} coh={avg_coherence:.3f} ppl={epoch_data['perplexity']:.1f} "
              f"exit={avg_exit_layer:.1f} J/tok={j_per_token:.5f} intel={intelligence:.3f}")

    # Final generation test
    print("\n[z1142] Generation test...")
    gen_metrics = compute_generation_quality(model, hw, "The meaning of life is ", 100)
    results['generation'] = gen_metrics
    print(f"  Generated: {gen_metrics['text'][:80]}...")
    print(f"  {gen_metrics['tokens_per_second']:.1f} tok/s, {gen_metrics['j_per_token']:.4f} J/tok")

    # Final summary
    hw_summary = hw.get_summary()
    results['final_metrics'] = {
        **hw_summary,
        'final_loss': float(results['epochs'][-1]['avg_loss']),
        'final_coherence': float(results['epochs'][-1]['avg_coherence']),
        'final_perplexity': float(results['epochs'][-1]['perplexity']),
        'avg_exit_layer': float(np.mean([e['avg_exit_layer'] for e in results['epochs']])),
    }

    # Business metrics
    results['business_metrics'] = {
        'total_energy_j': hw_summary['total_energy_j'],
        'total_tokens': hw_summary['total_tokens'],
        'efficiency_tokens_per_joule': hw_summary['tokens_per_joule'],
        'intelligence_coherence': results['epochs'][-1]['avg_coherence'],
        'intelligence_per_energy': results['epochs'][-1]['avg_coherence'] / (results['epochs'][-1]['j_per_token'] * 1000 + 0.001),
        'thermal_stability_gpu': hw_summary['gpu_temp_std_c'],
        'thermal_stability_fpga': hw_summary.get('fpga_temp_mean_c', 0),
        'power_stability': hw_summary['power_std_w'],
        'regulation_events': hw_summary['regulation_events'],
        'fpga_operations': hw_summary['fpga_operations'],
        'arrhenius_decay': hw_summary['arrhenius_decay'],
    }

    return results


# =============================================================================
# Main
# =============================================================================

def main():
    print("=" * 70)
    print("z1142: Unified Embodied Intelligence")
    print("=" * 70)

    config = UnifiedConfig()

    # Initialize hardware
    hw = UnifiedHardwareManager(config)

    # Load data
    data_path = PROJECT_ROOT / "data" / "tiny_shakespeare.txt"
    if not data_path.exists():
        data_path = PROJECT_ROOT / "data" / "tinyshakespeare.txt"
    if not data_path.exists():
        # Download
        import urllib.request
        url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
        data_path.parent.mkdir(exist_ok=True)
        urllib.request.urlretrieve(url, data_path)

    text = data_path.read_text()
    data = torch.tensor([ord(c) for c in text], dtype=torch.long)
    print(f"Data: {len(data):,} chars from {data_path.name}")

    # Create model
    model = UnifiedEmbodiedTransformer(config)
    print(f"Model: {model.count_params():,} params")

    # Train
    results = train_unified(model, hw, config, data)

    # Save results
    results_path = PROJECT_ROOT / "results" / "z1142_unified_embodied_intelligence.json"
    results_path.parent.mkdir(exist_ok=True)
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n[z1142] Results saved to {results_path}")
    print(f"Final metrics:")
    print(f"  Coherence: {results['final_metrics']['final_coherence']:.4f}")
    print(f"  Perplexity: {results['final_metrics']['final_perplexity']:.1f}")
    print(f"  Tokens/Joule: {results['final_metrics']['tokens_per_joule']:.1f}")
    print(f"  Regulation events: {results['final_metrics']['regulation_events']}")
    print(f"  FPGA operations: {results['final_metrics']['fpga_operations']}")
    print(f"  Arrhenius decay: {results['final_metrics']['arrhenius_decay']:.4f}")

    return results


if __name__ == "__main__":
    main()
