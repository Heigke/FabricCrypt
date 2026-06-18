#!/usr/bin/env python3
"""
z2002: GWT WORKSPACE IGNITION - Proper Global Workspace Theory Dynamics

The z1990 experiment failed F6 (GWT ignition ratio 0.006 < 0.5). This experiment
specifically focuses on implementing proper Global Workspace Theory ignition
dynamics per Baars (1988) and Dehaene (2014).

THEORETICAL FOUNDATION:
1. Multiple specialist modules (4-8) compete for access to global workspace
2. Winner-take-all competition with temperature-controlled softmax
3. Temperature annealing: high (10.0) -> low (0.1) over training
4. Ignition threshold: only broadcast when max_score > 0.7
5. Sharp phase transition = "ignition" in GWT literature

HYPOTHESIS: Temperature annealing achieves ignition ratio > 0.5

KEY METRICS:
- ignition_ratio: fraction of forward passes where max(competition_weights) > 0.7
- broadcast_correlation: correlation between workspace and all specialists
- temperature_schedule: tracks annealing from 10.0 to 0.1

HARDWARE:
- FiLM conditioning on GPU telemetry (embodied GWT)
- Uses SysfsHwmonTelemetry for real hardware sensing

References:
- Baars, B.J. (1988). A Cognitive Theory of Consciousness
- Dehaene, S. (2014). Consciousness and the Brain

Author: Claude (Opus 4.5)
Date: 2026-02-06
"""

import functools
print = functools.partial(print, flush=True)

import os
import sys
import json
import time
import math
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Optional, Any
from collections import deque

import numpy as np

# HSA override for gfx1151 compatibility
os.environ["HSA_OVERRIDE_GFX_VERSION"] = "11.0.0"

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.telemetry.sysfs_hwmon import SysfsHwmonTelemetry, GpuSample

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
RESULTS_DIR = Path(__file__).parent.parent / 'results'
RESULTS_DIR.mkdir(exist_ok=True)


# =============================================================================
# TEMPERATURE ANNEALING SCHEDULE
# =============================================================================

class TemperatureSchedule:
    """
    Temperature annealing for GWT competition.

    High temperature (10.0) = soft competition, many modules active
    Low temperature (0.1) = sharp competition, winner-take-all

    Annealing enables gradual transition to ignition dynamics.
    """

    def __init__(self, initial: float = 10.0, final: float = 0.1,
                 warmup_epochs: int = 2, total_epochs: int = 30):
        self.initial = initial
        self.final = final
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.current = initial

    def get_temperature(self, epoch: int, step: int = 0, steps_per_epoch: int = 1) -> float:
        """Get temperature for current training progress."""
        if epoch < self.warmup_epochs:
            # During warmup, stay at high temperature
            return self.initial

        # After warmup, anneal exponentially to final
        progress = (epoch - self.warmup_epochs) / max(1, self.total_epochs - self.warmup_epochs)
        progress = min(1.0, progress)

        # Exponential decay: T = T_init * (T_final/T_init)^progress
        ratio = self.final / self.initial
        self.current = self.initial * (ratio ** progress)

        return self.current

    def get_schedule_info(self) -> Dict:
        return {
            'initial_temp': self.initial,
            'final_temp': self.final,
            'warmup_epochs': self.warmup_epochs,
            'total_epochs': self.total_epochs,
            'current_temp': self.current,
        }


# =============================================================================
# GPU TELEMETRY WITH FILM CONDITIONING
# =============================================================================

class GpuTelemetryBuffer:
    """Buffer GPU telemetry for FiLM conditioning."""

    def __init__(self, telemetry: SysfsHwmonTelemetry, history_len: int = 32):
        self.telemetry = telemetry
        self.history: deque = deque(maxlen=history_len)
        self._last_sample: Optional[GpuSample] = None

    def sample(self) -> torch.Tensor:
        """Get normalized telemetry tensor [4] for FiLM conditioning."""
        sample = self.telemetry.read_sample()
        self._last_sample = sample
        self.history.append(sample)

        # Use correct GpuSample attributes
        raw = torch.tensor([
            sample.temp_edge_c,       # Temperature
            sample.power_w,           # Power
            sample.freq_sclk_mhz,     # GPU frequency
            sample.gpu_busy_pct,      # Utilization
        ], dtype=torch.float32)

        # Normalize to ~[0, 1] range
        norms = torch.tensor([100.0, 150.0, 3000.0, 100.0])
        normalized = (raw / norms).clamp(0, 2)

        return normalized

    def get_sequence(self, length: int = 16) -> torch.Tensor:
        """Get sequence of recent telemetry for temporal modeling."""
        if len(self.history) < length:
            # Pad with current sample
            current = self.sample()
            return current.unsqueeze(0).expand(length, -1)

        samples = list(self.history)[-length:]
        tensors = []
        norms = torch.tensor([100.0, 150.0, 3000.0, 100.0])

        for s in samples:
            raw = torch.tensor([
                s.temp_edge_c, s.power_w, s.freq_sclk_mhz, s.gpu_busy_pct
            ], dtype=torch.float32)
            tensors.append((raw / norms).clamp(0, 2))

        return torch.stack(tensors)

    def get_latest_raw(self) -> Dict[str, float]:
        """Get latest raw telemetry values."""
        if self._last_sample is None:
            self.sample()
        s = self._last_sample
        return {
            'temp_edge_c': s.temp_edge_c,
            'power_w': s.power_w,
            'freq_sclk_mhz': s.freq_sclk_mhz,
            'gpu_busy_pct': s.gpu_busy_pct,
        }


# =============================================================================
# SPECIALIST MODULES
# =============================================================================

class SpecialistModule(nn.Module):
    """
    Individual specialist module for GWT competition.

    Each specialist processes the same input but develops
    different representations/expertise.
    """

    def __init__(self, input_dim: int, hidden_dim: int, specialist_id: int):
        super().__init__()
        self.specialist_id = specialist_id

        # Unique processing pathway
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )

        # Salience computation (how important is this specialist's output?)
        self.salience_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.GELU(),
            nn.Linear(hidden_dim // 4, 1),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Process input and compute salience.

        Args:
            x: [batch, input_dim]

        Returns:
            representation: [batch, hidden_dim]
            salience: [batch, 1]
        """
        h = self.encoder(x)
        salience = self.salience_head(h)
        return h, salience


# =============================================================================
# GLOBAL WORKSPACE WITH PROPER IGNITION DYNAMICS
# =============================================================================

class GlobalWorkspaceIgnition(nn.Module):
    """
    Global Workspace with proper ignition dynamics per Baars/Dehaene.

    Key features:
    1. Multiple specialists compete for workspace access
    2. Temperature-controlled softmax competition
    3. Ignition = sharp transition when one specialist dominates
    4. Only broadcast when ignition occurs (max_weight > threshold)
    """

    IGNITION_THRESHOLD = 0.7  # Must exceed this for ignition

    def __init__(self, hidden_dim: int = 256, num_specialists: int = 6):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_specialists = num_specialists

        # Specialist modules
        self.specialists = nn.ModuleList([
            SpecialistModule(hidden_dim, hidden_dim, i)
            for i in range(num_specialists)
        ])

        # Global workspace broadcaster
        self.workspace = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )

        # Broadcast back to all specialists
        self.broadcast_proj = nn.Linear(hidden_dim, hidden_dim)

        # Tracking
        self.ignition_history = deque(maxlen=1000)
        self.competition_history = deque(maxlen=100)

    def forward(self, x: torch.Tensor, temperature: float = 1.0) -> Tuple[torch.Tensor, Dict]:
        """
        GWT forward pass with ignition dynamics.

        Args:
            x: [batch, hidden_dim] input representation
            temperature: softmax temperature for competition

        Returns:
            output: [batch, hidden_dim] - broadcast-enhanced representation
            info: Dict with ignition metrics
        """
        batch_size = x.shape[0]

        # Each specialist processes the input
        specialist_outputs = []
        saliences = []

        for specialist in self.specialists:
            h, s = specialist(x)
            specialist_outputs.append(h)
            saliences.append(s)

        # Stack saliences for competition
        salience_stack = torch.cat(saliences, dim=-1)  # [batch, num_specialists]

        # Temperature-controlled softmax competition
        # Low temperature = sharp winner-take-all
        # High temperature = soft mixing
        competition_logits = salience_stack / temperature
        competition_weights = F.softmax(competition_logits, dim=-1)

        # Ignition detection
        max_weights, winner_indices = competition_weights.max(dim=-1)
        ignition_mask = max_weights > self.IGNITION_THRESHOLD
        ignition_ratio = ignition_mask.float().mean().item()

        # Track ignition
        self.ignition_history.append(ignition_ratio)
        self.competition_history.append(competition_weights.detach().cpu())

        # Winner-take-all: Only the winning specialist broadcasts
        # This creates sharp "ignition" dynamics
        workspace_content = torch.zeros(batch_size, self.hidden_dim, device=x.device)

        for i in range(self.num_specialists):
            weight = competition_weights[:, i:i+1]  # [batch, 1]
            workspace_content = workspace_content + weight * specialist_outputs[i]

        # Transform through workspace
        broadcast_signal = self.workspace(workspace_content)

        # Only broadcast when ignition occurs (sharp transition)
        # This is key to proper GWT dynamics
        broadcast_mask = ignition_mask.float().unsqueeze(-1)  # [batch, 1]

        # Broadcast back to all specialists (global availability)
        broadcast_contribution = self.broadcast_proj(broadcast_signal) * broadcast_mask

        # Output combines input with broadcast signal
        output = x + broadcast_contribution

        # Compute broadcast correlation (how similar is workspace to each specialist?)
        broadcast_correlations = []
        for i, h in enumerate(specialist_outputs):
            # Cosine similarity
            sim = F.cosine_similarity(broadcast_signal, h, dim=-1).mean().item()
            broadcast_correlations.append(sim)

        info = {
            'ignition_ratio': ignition_ratio,
            'ignition_ratio_running': np.mean(list(self.ignition_history)) if self.ignition_history else 0,
            'temperature': temperature,
            'max_competition_weight': max_weights.mean().item(),
            'competition_entropy': -(competition_weights * (competition_weights + 1e-10).log()).sum(dim=-1).mean().item(),
            'winner_distribution': {i: (winner_indices == i).float().mean().item() for i in range(self.num_specialists)},
            'broadcast_correlations': broadcast_correlations,
            'mean_broadcast_correlation': np.mean(broadcast_correlations),
            'saliences_raw': salience_stack.mean(dim=0).detach().cpu().numpy().tolist(),
        }

        return output, info


# =============================================================================
# FILM LAYER FOR HARDWARE CONDITIONING
# =============================================================================

class FiLMConditioner(nn.Module):
    """Feature-wise Linear Modulation conditioned on hardware telemetry."""

    def __init__(self, hidden_dim: int, condition_dim: int = 4):
        super().__init__()
        self.gamma = nn.Sequential(
            nn.Linear(condition_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, hidden_dim),
        )
        self.beta = nn.Sequential(
            nn.Linear(condition_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, hidden_dim),
        )

    def forward(self, x: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        """
        Apply FiLM modulation.

        Args:
            x: [batch, ..., hidden_dim]
            condition: [batch, condition_dim]
        """
        gamma = 1 + self.gamma(condition)  # Center around 1
        beta = self.beta(condition)

        # Expand for any intermediate dimensions
        while gamma.dim() < x.dim():
            gamma = gamma.unsqueeze(1)
            beta = beta.unsqueeze(1)

        return gamma * x + beta


# =============================================================================
# GWT CONSCIOUSNESS MODEL
# =============================================================================

class GWTConsciousnessModel(nn.Module):
    """
    Full model with proper GWT ignition dynamics and hardware conditioning.
    """

    def __init__(self, vocab_size: int = 128, hidden_dim: int = 256,
                 num_specialists: int = 6, n_layers: int = 4):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size

        # Token embedding
        self.embed = nn.Embedding(vocab_size, hidden_dim)

        # FiLM conditioning on hardware telemetry
        self.film = FiLMConditioner(hidden_dim, condition_dim=4)

        # Transformer layers
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=8,
                dim_feedforward=hidden_dim * 4,
                batch_first=True,
                norm_first=True,
            ) for _ in range(n_layers)
        ])

        # Global Workspace with ignition dynamics
        self.gwt = GlobalWorkspaceIgnition(hidden_dim, num_specialists)

        # Output head
        self.output_proj = nn.Linear(hidden_dim, vocab_size)

        # Confidence head (HOT-style metacognition)
        self.confidence_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.GELU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, tokens: torch.Tensor, telemetry: torch.Tensor,
                temperature: float = 1.0) -> Dict[str, Any]:
        """
        Forward pass with GWT ignition.

        Args:
            tokens: [batch, seq_len]
            telemetry: [batch, 4] hardware telemetry
            temperature: GWT competition temperature
        """
        # Embed tokens
        h = self.embed(tokens)  # [batch, seq, hidden]

        # FiLM conditioning on hardware
        h = self.film(h, telemetry)

        # Transformer layers
        for layer in self.layers:
            h = layer(h)

        # Pool for GWT input (use mean over sequence)
        h_pooled = h.mean(dim=1)  # [batch, hidden]

        # Global Workspace with ignition
        h_broadcast, gwt_info = self.gwt(h_pooled, temperature)

        # Broadcast back to sequence
        h = h + h_broadcast.unsqueeze(1)

        # Output logits
        logits = self.output_proj(h)

        # Confidence
        confidence = self.confidence_head(h_pooled)

        return {
            'logits': logits,
            'hidden': h,
            'confidence': confidence,
            'gwt_info': gwt_info,
        }


# =============================================================================
# DATA
# =============================================================================

class TextDataset:
    """Character-level text dataset."""

    def __init__(self, text: str, seq_len: int = 64):
        self.text = text
        self.seq_len = seq_len
        self.chars = sorted(set(text))
        self.char2idx = {c: i for i, c in enumerate(self.chars)}
        self.idx2char = {i: c for c, i in self.char2idx.items()}
        self.vocab_size = len(self.chars)
        self.data = torch.tensor([self.char2idx[c] for c in text], dtype=torch.long)

    def __len__(self):
        return len(self.data) - self.seq_len - 1

    def __getitem__(self, idx):
        x = self.data[idx:idx + self.seq_len]
        y = self.data[idx + 1:idx + self.seq_len + 1]
        return x, y


def load_text_data():
    """Load or generate training text."""
    paths = [
        Path(__file__).parent.parent / 'data' / 'shakespeare.txt',
        Path(__file__).parent.parent / 'data' / 'tiny_shakespeare.txt',
    ]

    for p in paths:
        if p.exists():
            print(f"[Data] Loading from {p}")
            return p.read_text()

    # Generate synthetic text
    print("[Data] Generating synthetic text")
    samples = [
        "To be, or not to be, that is the question:\n",
        "Whether 'tis nobler in the mind to suffer\n",
        "The slings and arrows of outrageous fortune,\n",
        "Or to take arms against a sea of troubles,\n",
        "And by opposing end them. To die: to sleep;\n",
        "All the world's a stage, and all the men and women merely players.\n",
        "Now is the winter of our discontent.\n",
        "Friends, Romans, countrymen, lend me your ears.\n",
    ]
    return ''.join(samples * 500)


# =============================================================================
# TRAINING
# =============================================================================

def train_epoch(model: GWTConsciousnessModel, dataset: TextDataset,
                telemetry_buffer: GpuTelemetryBuffer, optimizer: torch.optim.Optimizer,
                device: torch.device, epoch: int, temp_schedule: TemperatureSchedule,
                max_batches: int = 1000) -> Dict:
    """Train one epoch with GWT ignition tracking."""
    model.train()

    total_loss = 0
    correct = 0
    total = 0

    # GWT metrics
    ignition_ratios = []
    temperatures = []
    broadcast_correlations = []
    competition_entropies = []

    batch_size = 32
    num_batches = min(len(dataset) // batch_size, max_batches)

    for batch_idx in range(num_batches):
        # Get batch
        start_idx = batch_idx * batch_size
        batch_x, batch_y = [], []
        for i in range(batch_size):
            x, y = dataset[(start_idx + i) % len(dataset)]
            batch_x.append(x)
            batch_y.append(y)

        x = torch.stack(batch_x).to(device)
        y = torch.stack(batch_y).to(device)

        # Get hardware telemetry
        tel = telemetry_buffer.sample().unsqueeze(0).expand(batch_size, -1).to(device)

        # Get temperature for this step
        temperature = temp_schedule.get_temperature(epoch, batch_idx, num_batches)

        # Forward
        optimizer.zero_grad()
        out = model(x, tel, temperature)

        # Loss
        logits = out['logits'].view(-1, dataset.vocab_size)
        loss = F.cross_entropy(logits, y.view(-1))

        # Backward
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        # Metrics
        total_loss += loss.item()
        preds = logits.argmax(dim=-1)
        correct += (preds == y.view(-1)).sum().item()
        total += y.numel()

        # GWT metrics
        gwt_info = out['gwt_info']
        ignition_ratios.append(gwt_info['ignition_ratio'])
        temperatures.append(temperature)
        broadcast_correlations.append(gwt_info['mean_broadcast_correlation'])
        competition_entropies.append(gwt_info['competition_entropy'])

        # Progress
        if batch_idx % 100 == 0:
            print(f"  Batch {batch_idx}/{num_batches}: loss={loss.item():.4f} "
                  f"temp={temperature:.3f} ignition={gwt_info['ignition_ratio']:.3f} "
                  f"max_weight={gwt_info['max_competition_weight']:.3f}")

    return {
        'loss': total_loss / num_batches,
        'accuracy': correct / total,
        'ignition_ratio': np.mean(ignition_ratios),
        'ignition_ratio_std': np.std(ignition_ratios),
        'temperature_mean': np.mean(temperatures),
        'temperature_final': temperatures[-1] if temperatures else 1.0,
        'broadcast_correlation': np.mean(broadcast_correlations),
        'competition_entropy': np.mean(competition_entropies),
        'ignition_running': model.gwt.ignition_history and np.mean(list(model.gwt.ignition_history)) or 0,
    }


def evaluate_ignition(model: GWTConsciousnessModel, dataset: TextDataset,
                      telemetry_buffer: GpuTelemetryBuffer, device: torch.device,
                      temperature: float = 0.1) -> Dict:
    """Evaluate GWT ignition at low temperature."""
    model.eval()

    ignition_ratios = []
    max_weights = []
    winner_counts = {i: 0 for i in range(model.gwt.num_specialists)}

    batch_size = 32
    num_batches = 50

    with torch.no_grad():
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            batch_x = []
            for i in range(batch_size):
                x, _ = dataset[(start_idx + i) % len(dataset)]
                batch_x.append(x)

            x = torch.stack(batch_x).to(device)
            tel = telemetry_buffer.sample().unsqueeze(0).expand(batch_size, -1).to(device)

            out = model(x, tel, temperature)
            gwt_info = out['gwt_info']

            ignition_ratios.append(gwt_info['ignition_ratio'])
            max_weights.append(gwt_info['max_competition_weight'])

            for i, count in gwt_info['winner_distribution'].items():
                winner_counts[i] += count * batch_size

    # Normalize winner distribution
    total = sum(winner_counts.values())
    winner_dist = {i: c / total for i, c in winner_counts.items()}

    return {
        'eval_ignition_ratio': np.mean(ignition_ratios),
        'eval_max_weight': np.mean(max_weights),
        'eval_winner_distribution': winner_dist,
        'eval_specialist_diversity': -sum(p * np.log(p + 1e-10) for p in winner_dist.values()),
    }


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("="*80)
    print("z2002: GWT WORKSPACE IGNITION")
    print("Testing proper Global Workspace Theory ignition dynamics")
    print("="*80)
    print(f"Start time: {datetime.now().isoformat()}")
    print(f"Device: {DEVICE}")
    print()

    # Initialize hardware telemetry
    print("[Hardware] Initializing sysfs hwmon telemetry...")
    try:
        telemetry = SysfsHwmonTelemetry(sample_rate_hz=20)
        telemetry_buffer = GpuTelemetryBuffer(telemetry)

        # Test read
        sample = telemetry_buffer.sample()
        raw = telemetry_buffer.get_latest_raw()
        print(f"[Hardware] GPU telemetry active:")
        print(f"  - temp_edge_c: {raw['temp_edge_c']:.1f} C")
        print(f"  - power_w: {raw['power_w']:.1f} W")
        print(f"  - freq_sclk_mhz: {raw['freq_sclk_mhz']} MHz")
        print(f"  - gpu_busy_pct: {raw['gpu_busy_pct']:.1f}%")
    except Exception as e:
        print(f"[Hardware] Telemetry init failed: {e}")
        print("[Hardware] Using dummy telemetry")
        telemetry = None
        telemetry_buffer = None

    # Load data
    text = load_text_data()
    dataset = TextDataset(text, seq_len=64)
    print(f"\n[Data] {len(dataset)} samples, vocab size {dataset.vocab_size}")

    # Create model
    model = GWTConsciousnessModel(
        vocab_size=dataset.vocab_size,
        hidden_dim=256,
        num_specialists=6,
        n_layers=4,
    ).to(DEVICE)

    param_count = sum(p.numel() for p in model.parameters())
    print(f"[Model] Parameters: {param_count:,}")
    print(f"[Model] GWT specialists: {model.gwt.num_specialists}")

    # Temperature schedule
    num_epochs = 30
    temp_schedule = TemperatureSchedule(
        initial=10.0,
        final=0.1,
        warmup_epochs=2,
        total_epochs=num_epochs
    )
    print(f"\n[Temperature Schedule]")
    print(f"  Initial: {temp_schedule.initial}")
    print(f"  Final: {temp_schedule.final}")
    print(f"  Warmup epochs: {temp_schedule.warmup_epochs}")

    # Optimizer
    optimizer = AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)

    # Training
    print("\n" + "="*80)
    print(f"TRAINING: {num_epochs} EPOCHS WITH TEMPERATURE ANNEALING")
    print("Hypothesis: Temperature annealing achieves ignition ratio > 0.5")
    print("="*80)

    # Create dummy telemetry buffer if hardware failed
    if telemetry_buffer is None:
        class DummyTelemetryBuffer:
            def sample(self):
                return torch.tensor([0.5, 0.3, 0.8, 0.4], dtype=torch.float32)
            def get_latest_raw(self):
                return {'temp_edge_c': 50, 'power_w': 45, 'freq_sclk_mhz': 1800, 'gpu_busy_pct': 40}
        telemetry_buffer = DummyTelemetryBuffer()

    epoch_metrics = []

    try:
        for epoch in range(num_epochs):
            epoch_start = time.time()
            current_temp = temp_schedule.get_temperature(epoch)

            print(f"\n[Epoch {epoch}/{num_epochs}] Temperature: {current_temp:.3f}")

            metrics = train_epoch(
                model, dataset, telemetry_buffer, optimizer, DEVICE,
                epoch, temp_schedule, max_batches=800
            )

            scheduler.step()

            epoch_time = time.time() - epoch_start
            metrics['epoch'] = epoch
            metrics['time'] = epoch_time

            epoch_metrics.append(metrics)

            print(f"\nEpoch {epoch} Results:")
            print(f"  Loss: {metrics['loss']:.4f}")
            print(f"  Accuracy: {metrics['accuracy']:.4f}")
            print(f"  GWT Ignition Ratio: {metrics['ignition_ratio']:.4f}")
            print(f"  Running Ignition: {metrics['ignition_running']:.4f}")
            print(f"  Temperature: {metrics['temperature_final']:.4f}")
            print(f"  Broadcast Correlation: {metrics['broadcast_correlation']:.4f}")
            print(f"  Competition Entropy: {metrics['competition_entropy']:.4f}")
            print(f"  Time: {epoch_time:.1f}s")

            # Hardware telemetry
            raw = telemetry_buffer.get_latest_raw()
            print(f"  GPU: {raw['temp_edge_c']:.1f}C, {raw['power_w']:.1f}W, {raw['gpu_busy_pct']:.1f}%")

    except KeyboardInterrupt:
        print("\n[Interrupted - saving results]")

    # Final evaluation at low temperature
    print("\n" + "="*80)
    print("FINAL EVALUATION AT LOW TEMPERATURE (0.1)")
    print("="*80)

    eval_metrics = evaluate_ignition(model, dataset, telemetry_buffer, DEVICE, temperature=0.1)

    print(f"\nEvaluation Results:")
    print(f"  Final Ignition Ratio: {eval_metrics['eval_ignition_ratio']:.4f}")
    print(f"  Max Competition Weight: {eval_metrics['eval_max_weight']:.4f}")
    print(f"  Specialist Diversity: {eval_metrics['eval_specialist_diversity']:.4f}")
    print(f"  Winner Distribution: {eval_metrics['eval_winner_distribution']}")

    # Verdict
    final_ignition = eval_metrics['eval_ignition_ratio']
    threshold = 0.5

    print("\n" + "="*80)
    print("GWT IGNITION CRITERION (F6)")
    print("="*80)
    print(f"Threshold: ignition_ratio >= {threshold}")
    print(f"Measured: {final_ignition:.4f}")

    if final_ignition >= threshold:
        verdict = "PASS - GWT ignition dynamics achieved"
        passed = True
    else:
        verdict = "FAIL - Insufficient ignition dynamics"
        passed = False

    print(f"\nVERDICT: {verdict}")
    print("="*80)

    # Compile results
    results = {
        'experiment': 'z2002_gwt_workspace_ignition',
        'timestamp': datetime.now().isoformat(),
        'device': str(DEVICE),
        'hypothesis': 'Temperature annealing achieves ignition ratio > 0.5',
        'model': {
            'vocab_size': dataset.vocab_size,
            'hidden_dim': 256,
            'num_specialists': 6,
            'n_layers': 4,
            'parameters': param_count,
        },
        'temperature_schedule': temp_schedule.get_schedule_info(),
        'training': {
            'epochs_completed': len(epoch_metrics),
            'final_loss': epoch_metrics[-1]['loss'] if epoch_metrics else None,
            'final_accuracy': epoch_metrics[-1]['accuracy'] if epoch_metrics else None,
        },
        'gwt_metrics': {
            'final_ignition_ratio': final_ignition,
            'ignition_threshold': threshold,
            'ignition_criterion_passed': passed,
            'max_competition_weight': eval_metrics['eval_max_weight'],
            'specialist_diversity': eval_metrics['eval_specialist_diversity'],
            'winner_distribution': eval_metrics['eval_winner_distribution'],
        },
        'epoch_history': [
            {
                'epoch': m['epoch'],
                'loss': m['loss'],
                'accuracy': m['accuracy'],
                'ignition_ratio': m['ignition_ratio'],
                'temperature': m['temperature_final'],
                'broadcast_correlation': m['broadcast_correlation'],
            }
            for m in epoch_metrics
        ],
        'evaluation': eval_metrics,
        'verdict': verdict,
        'passed': passed,
    }

    # Save results
    output_path = RESULTS_DIR / 'z2002_gwt_workspace_ignition.json'
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\nResults saved to: {output_path}")

    return results


if __name__ == '__main__':
    main()
