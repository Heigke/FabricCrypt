#!/usr/bin/env python3
"""
z1980: Temporal Coherence FIXED - True Recurrent Body Model

ROOT CAUSE of z1963 failure:
- Model predicted body state CHANGES, not coherent representations
- Each timestep decorrelated from previous
- Temporal binding module lacked proper recurrence

SOLUTION:
1. True Recurrent Body Model (GRU with persistent hidden state)
2. Contrastive Temporal Loss (consecutive similar, distant different)
3. Temporal Smoothness Regularization
4. Teacher Forcing Schedule (phased curriculum)

Target: I4_autocorr > 0.3 for k in [1, 10] at end of training
"""

import os
import sys
import json
import time
import random
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import List, Tuple, Optional, Dict

import numpy as np

# HSA override for gfx1151 compatibility
os.environ["HSA_OVERRIDE_GFX_VERSION"] = "11.0.0"

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.telemetry.sysfs_hwmon import SysfsHwmonTelemetry, GpuSample


@dataclass
class TemporalMetrics:
    """Metrics for temporal coherence tracking."""
    epoch: int
    i4_autocorr_lag1: float
    i4_autocorr_lag5: float
    i4_autocorr_lag10: float
    smoothness_loss: float
    contrastive_loss: float
    reconstruction_loss: float
    total_loss: float
    teacher_forcing_ratio: float
    hidden_state_norm: float
    z_trajectory_variance: float


class TemporalEncoder(nn.Module):
    """Encodes telemetry into latent space."""

    def __init__(self, input_dim: int = 8, hidden_dim: int = 64, latent_dim: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, latent_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TemporalDecoder(nn.Module):
    """Decodes latent back to telemetry for reconstruction."""

    def __init__(self, latent_dim: int = 32, hidden_dim: int = 64, output_dim: int = 8):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class TemporalBodyModel(nn.Module):
    """
    True Recurrent Body Model with GRU.

    Key insight: Maintains persistent hidden state h_t that carries
    temporal coherence across timesteps. The hidden state IS the
    body representation that must stay coherent.
    """

    def __init__(self, latent_dim: int = 32, hidden_dim: int = 64, num_layers: int = 2):
        super().__init__()
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # GRU for temporal dynamics (no dropout - causes MIOpen issues on gfx1151)
        self.gru = nn.GRU(
            input_size=latent_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.0  # Disabled due to MIOpen compilation issues on gfx1151
        )

        # Project hidden state back to latent space
        self.hidden_to_latent = nn.Linear(hidden_dim, latent_dim)

        # Persistent hidden state (will be set per-batch)
        self.register_buffer('h_t', None)

    def init_hidden(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Initialize hidden state."""
        h = torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=device)
        return h

    def step(self, z_t: torch.Tensor, h_prev: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Single timestep update.

        Args:
            z_t: Current latent observation [batch, latent_dim]
            h_prev: Previous hidden state [num_layers, batch, hidden_dim]

        Returns:
            z_next: Predicted next latent [batch, latent_dim]
            h_next: Updated hidden state [num_layers, batch, hidden_dim]
        """
        # GRU expects [batch, seq=1, features]
        out, h_next = self.gru(z_t.unsqueeze(1), h_prev)

        # Project to latent space
        z_next = self.hidden_to_latent(out.squeeze(1))

        return z_next, h_next

    def forward_sequence(self, z_seq: torch.Tensor, h_init: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Process entire sequence.

        Args:
            z_seq: Latent sequence [batch, seq_len, latent_dim]
            h_init: Initial hidden state (optional)

        Returns:
            z_out: Output latent sequence [batch, seq_len, latent_dim]
            h_final: Final hidden state
        """
        batch_size, seq_len, _ = z_seq.shape
        device = z_seq.device

        if h_init is None:
            h_init = self.init_hidden(batch_size, device)

        out, h_final = self.gru(z_seq, h_init)
        z_out = self.hidden_to_latent(out)

        return z_out, h_final


class TemporalCoherenceModel(nn.Module):
    """
    Full temporal coherence model with encoder, body model, and decoder.
    """

    def __init__(self, input_dim: int = 8, latent_dim: int = 32, hidden_dim: int = 64):
        super().__init__()
        self.encoder = TemporalEncoder(input_dim, hidden_dim, latent_dim)
        self.body_model = TemporalBodyModel(latent_dim, hidden_dim)
        self.decoder = TemporalDecoder(latent_dim, hidden_dim, input_dim)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode telemetry to latent."""
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent to telemetry."""
        return self.decoder(z)


def compute_autocorrelation(z_traj: torch.Tensor, max_lag: int = 10) -> Dict[int, float]:
    """
    Compute autocorrelation of latent trajectory.

    Args:
        z_traj: [batch, seq, hidden] or [seq, hidden]
        max_lag: Maximum lag to compute

    Returns:
        Dictionary mapping lag -> autocorrelation
    """
    if z_traj.dim() == 3:
        # Average over batch
        z_traj = z_traj.mean(dim=0)

    # Flatten hidden dimensions
    z_flat = z_traj.reshape(z_traj.size(0), -1)  # [seq, hidden]

    # Normalize
    z_centered = z_flat - z_flat.mean(dim=0, keepdim=True)
    z_norm = z_centered / (z_centered.std(dim=0, keepdim=True) + 1e-8)

    autocorrs = {}
    seq_len = z_norm.size(0)

    for lag in range(1, min(max_lag + 1, seq_len)):
        # Correlation between z[t] and z[t+lag]
        z1 = z_norm[:-lag]  # [seq-lag, hidden]
        z2 = z_norm[lag:]   # [seq-lag, hidden]

        # Mean correlation across dimensions
        corr = (z1 * z2).mean()
        autocorrs[lag] = corr.item()

    return autocorrs


def temporal_contrastive_loss(z_trajectory: torch.Tensor, temperature: float = 0.1) -> torch.Tensor:
    """
    Contrastive loss: consecutive states similar, random states different.

    Args:
        z_trajectory: [batch, seq, hidden]
        temperature: Softmax temperature

    Returns:
        Contrastive loss scalar
    """
    batch_size, seq_len, hidden_dim = z_trajectory.shape

    if seq_len < 3:
        return torch.tensor(0.0, device=z_trajectory.device)

    # Positive pairs: consecutive timesteps
    z_t = z_trajectory[:, :-1]  # [batch, seq-1, hidden]
    z_t1 = z_trajectory[:, 1:]  # [batch, seq-1, hidden]

    # Compute positive similarity (cosine)
    pos_sim = F.cosine_similarity(z_t, z_t1, dim=-1)  # [batch, seq-1]

    # Negative pairs: random timesteps (at least 2 steps apart)
    neg_indices = torch.randint(0, seq_len, (batch_size, seq_len - 1), device=z_trajectory.device)

    # Ensure negatives are at least 2 timesteps away
    base_indices = torch.arange(seq_len - 1, device=z_trajectory.device).unsqueeze(0).expand(batch_size, -1)
    too_close = (neg_indices - base_indices).abs() < 2
    neg_indices = torch.where(too_close, (neg_indices + 3) % seq_len, neg_indices)

    # Gather negative samples
    neg_indices_expanded = neg_indices.unsqueeze(-1).expand(-1, -1, hidden_dim)
    z_neg = z_trajectory.gather(1, neg_indices_expanded)  # [batch, seq-1, hidden]

    neg_sim = F.cosine_similarity(z_t, z_neg, dim=-1)  # [batch, seq-1]

    # InfoNCE-style loss
    logits = torch.stack([pos_sim, neg_sim], dim=-1) / temperature  # [batch, seq-1, 2]
    labels = torch.zeros(batch_size, seq_len - 1, dtype=torch.long, device=z_trajectory.device)

    loss = F.cross_entropy(logits.reshape(-1, 2), labels.reshape(-1))

    return loss


def temporal_coherence_loss(z_trajectory: torch.Tensor, max_lag: int = 15, decay_rate: float = 0.1) -> torch.Tensor:
    """
    Direct temporal coherence loss - enforce exponentially decaying correlation.

    This directly optimizes for the metric we care about: autocorrelation at various lags.
    Target correlation at lag k: exp(-decay_rate * k)

    Args:
        z_trajectory: [batch, seq, hidden]
        max_lag: Maximum lag to enforce correlation
        decay_rate: How fast correlation should decay (smaller = more coherent)

    Returns:
        Loss that penalizes deviation from target correlation structure
    """
    batch_size, seq_len, hidden_dim = z_trajectory.shape
    device = z_trajectory.device

    if seq_len < max_lag + 1:
        max_lag = seq_len - 1

    total_loss = torch.tensor(0.0, device=device)

    # Normalize z for correlation computation
    z_mean = z_trajectory.mean(dim=1, keepdim=True)
    z_centered = z_trajectory - z_mean
    z_std = z_centered.std(dim=1, keepdim=True) + 1e-8
    z_norm = z_centered / z_std  # [batch, seq, hidden]

    for lag in range(1, max_lag + 1):
        # Target correlation decays exponentially
        target_corr = np.exp(-decay_rate * lag)

        # Actual correlation at this lag
        z1 = z_norm[:, :-lag]  # [batch, seq-lag, hidden]
        z2 = z_norm[:, lag:]   # [batch, seq-lag, hidden]

        actual_corr = (z1 * z2).mean(dim=(1, 2))  # [batch]

        # Penalize if correlation is too low (we want it close to target)
        # Use hinge loss: only penalize if actual < target
        corr_deficit = F.relu(target_corr - actual_corr)

        # Weight longer lags more heavily (they're what's failing)
        weight = 1.0 + 0.5 * lag
        total_loss = total_loss + weight * corr_deficit.mean()

    return total_loss / max_lag


def smoothness_loss(z_trajectory: torch.Tensor) -> torch.Tensor:
    """
    Penalize large changes between consecutive timesteps.

    Args:
        z_trajectory: [batch, seq, hidden]

    Returns:
        Smoothness loss scalar
    """
    if z_trajectory.size(1) < 2:
        return torch.tensor(0.0, device=z_trajectory.device)

    # L2 distance between consecutive states
    diff = z_trajectory[:, 1:] - z_trajectory[:, :-1]
    return (diff ** 2).mean()


def collect_telemetry_trajectory(
    telemetry: SysfsHwmonTelemetry,
    duration_s: float = 5.0,
    sample_rate_hz: float = 50.0,
    induce_load: bool = True
) -> np.ndarray:
    """
    Collect a trajectory of GPU telemetry.

    Returns:
        Array of shape [seq_len, 8] with normalized telemetry
    """
    samples = []
    interval = 1.0 / sample_rate_hz

    # Optional: induce some GPU load for variation
    if induce_load and torch.cuda.is_available():
        load_tensor = torch.randn(1000, 1000, device='cuda')

    start_time = time.time()
    while time.time() - start_time < duration_s:
        sample = telemetry.read_sample()

        # Extract 8 features
        features = [
            sample.power_w / 100.0,  # Normalize to ~[0, 1]
            sample.temp_edge_c / 100.0,
            sample.temp_junction_c / 100.0 if sample.temp_junction_c else sample.temp_edge_c / 100.0,
            sample.temp_mem_c / 100.0 if sample.temp_mem_c else sample.temp_edge_c / 100.0,
            sample.gpu_busy_pct / 100.0,
            sample.vram_used_gb / 16.0,  # Assume 16GB max
            sample.freq_sclk_mhz / 3000.0 if sample.freq_sclk_mhz else 0.5,
            sample.freq_mclk_mhz / 2000.0 if sample.freq_mclk_mhz else 0.5,
        ]
        samples.append(features)

        # Vary GPU load
        if induce_load and torch.cuda.is_available() and random.random() < 0.3:
            _ = torch.mm(load_tensor, load_tensor)
            torch.cuda.synchronize()

        time.sleep(interval)

    # Cleanup
    if induce_load and torch.cuda.is_available():
        del load_tensor
        torch.cuda.empty_cache()

    return np.array(samples, dtype=np.float32)


def generate_synthetic_trajectory(
    batch_size: int,
    seq_len: int,
    input_dim: int = 8,
    noise_scale: float = 0.05
) -> torch.Tensor:
    """
    Generate synthetic telemetry-like trajectories for training.

    Creates smooth, temporally coherent trajectories that mimic
    real GPU behavior with gradual changes.
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Base sinusoidal patterns at different frequencies
    t = torch.linspace(0, 4 * np.pi, seq_len, device=device)

    trajectories = []
    for _ in range(batch_size):
        # Random phase and frequency modulation per feature
        phases = torch.rand(input_dim, device=device) * 2 * np.pi
        freqs = 0.5 + torch.rand(input_dim, device=device) * 1.5
        amps = 0.3 + torch.rand(input_dim, device=device) * 0.4
        offsets = 0.3 + torch.rand(input_dim, device=device) * 0.4

        # Generate smooth trajectory
        traj = []
        for i in range(input_dim):
            signal = offsets[i] + amps[i] * torch.sin(freqs[i] * t + phases[i])
            signal = signal + noise_scale * torch.randn_like(signal)
            signal = torch.clamp(signal, 0, 1)
            traj.append(signal)

        traj = torch.stack(traj, dim=-1)  # [seq_len, input_dim]
        trajectories.append(traj)

    return torch.stack(trajectories)  # [batch, seq, input_dim]


def train_epoch(
    model: TemporalCoherenceModel,
    optimizer: torch.optim.Optimizer,
    batch_size: int,
    seq_len: int,
    teacher_forcing_ratio: float,
    smoothness_weight: float,
    contrastive_weight: float,
    device: torch.device,
) -> Dict[str, float]:
    """
    Train one epoch with teacher forcing curriculum.
    """
    model.train()

    # Generate synthetic trajectories
    x_traj = generate_synthetic_trajectory(batch_size, seq_len).to(device)

    # Encode full trajectory (ground truth)
    z_gt = model.encode(x_traj)  # [batch, seq, latent]

    # Forward through body model with teacher forcing
    batch_size, seq_len, latent_dim = z_gt.shape
    h = model.body_model.init_hidden(batch_size, device)

    z_trajectory = []
    z_prev = z_gt[:, 0]  # Start with first encoded state
    z_trajectory.append(z_prev)

    for t in range(1, seq_len):
        # Teacher forcing decision
        if random.random() < teacher_forcing_ratio:
            # Use ground truth
            z_input = z_gt[:, t-1]
        else:
            # Use model's own prediction
            z_input = z_prev

        # Step through body model
        z_pred, h = model.body_model.step(z_input, h)
        z_trajectory.append(z_pred)
        z_prev = z_pred

    z_trajectory = torch.stack(z_trajectory, dim=1)  # [batch, seq, latent]

    # Reconstruction loss
    x_recon = model.decode(z_trajectory)
    recon_loss = F.mse_loss(x_recon, x_traj)

    # Temporal smoothness loss
    smooth_loss = smoothness_loss(z_trajectory)

    # Contrastive temporal loss
    contrast_loss = temporal_contrastive_loss(z_trajectory)

    # NEW: Direct temporal coherence loss (enforces autocorrelation at all lags)
    coherence_loss = temporal_coherence_loss(z_trajectory, max_lag=15, decay_rate=0.05)

    # Total loss - coherence_weight starts high and maintains
    coherence_weight = 0.5
    total_loss = recon_loss + smoothness_weight * smooth_loss + contrastive_weight * contrast_loss + coherence_weight * coherence_loss

    # Optimize
    optimizer.zero_grad()
    total_loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()

    # Compute autocorrelation metrics
    with torch.no_grad():
        autocorrs = compute_autocorrelation(z_trajectory, max_lag=10)

    return {
        'reconstruction_loss': recon_loss.item(),
        'smoothness_loss': smooth_loss.item(),
        'contrastive_loss': contrast_loss.item(),
        'coherence_loss': coherence_loss.item(),
        'total_loss': total_loss.item(),
        'autocorr_lag1': autocorrs.get(1, 0.0),
        'autocorr_lag5': autocorrs.get(5, 0.0),
        'autocorr_lag10': autocorrs.get(10, 0.0),
        'hidden_state_norm': h.norm().item(),
        'z_trajectory_variance': z_trajectory.var().item(),
    }


def evaluate_on_real_telemetry(
    model: TemporalCoherenceModel,
    telemetry: SysfsHwmonTelemetry,
    device: torch.device,
    duration_s: float = 3.0
) -> Dict[str, float]:
    """
    Evaluate temporal coherence on real GPU telemetry.
    """
    model.eval()

    # Collect real trajectory
    real_traj = collect_telemetry_trajectory(telemetry, duration_s=duration_s, induce_load=True)
    x_real = torch.tensor(real_traj, device=device).unsqueeze(0)  # [1, seq, 8]

    with torch.no_grad():
        # Encode
        z_gt = model.encode(x_real)

        # Forward through body model (no teacher forcing)
        batch_size, seq_len, latent_dim = z_gt.shape
        h = model.body_model.init_hidden(batch_size, device)

        z_trajectory = []
        z_prev = z_gt[:, 0]
        z_trajectory.append(z_prev)

        for t in range(1, seq_len):
            z_pred, h = model.body_model.step(z_prev, h)
            z_trajectory.append(z_pred)
            z_prev = z_pred

        z_trajectory = torch.stack(z_trajectory, dim=1)

        # Compute autocorrelations
        autocorrs = compute_autocorrelation(z_trajectory, max_lag=10)

        # Reconstruction error
        x_recon = model.decode(z_trajectory)
        recon_error = F.mse_loss(x_recon, x_real).item()

    return {
        'real_autocorr_lag1': autocorrs.get(1, 0.0),
        'real_autocorr_lag5': autocorrs.get(5, 0.0),
        'real_autocorr_lag10': autocorrs.get(10, 0.0),
        'real_recon_error': recon_error,
        'trajectory_length': seq_len,
    }


def get_teacher_forcing_schedule(epoch: int, total_epochs: int) -> float:
    """
    Curriculum learning schedule for teacher forcing.

    Phase 1 (0-20): High teacher forcing 0.9 -> 0.3
    Phase 2 (21-50): Moderate 0.3 -> 0.1
    Phase 3 (51+): Low 0.1 -> 0.0
    """
    if epoch < 20:
        return 0.9 - (0.6 * epoch / 20)
    elif epoch < 50:
        return 0.3 - (0.2 * (epoch - 20) / 30)
    else:
        return max(0.0, 0.1 - (0.1 * (epoch - 50) / (total_epochs - 50)))


def get_smoothness_weight_schedule(epoch: int, total_epochs: int) -> float:
    """
    Schedule for smoothness regularization.

    Phase 1: High (0.5)
    Phase 2: Moderate (0.2)
    Phase 3: Low (0.05)
    """
    if epoch < 20:
        return 0.5
    elif epoch < 50:
        return 0.2
    else:
        return 0.05


def main():
    print("=" * 70)
    print("z1980: Temporal Coherence FIXED - True Recurrent Body Model")
    print("=" * 70)
    print(f"Timestamp: {datetime.now().isoformat()}")
    print()

    # Configuration
    config = {
        'total_epochs': 100,
        'batch_size': 32,
        'seq_len': 100,
        'input_dim': 8,
        'latent_dim': 32,
        'hidden_dim': 64,
        'learning_rate': 1e-3,
        'contrastive_weight': 0.3,
        'eval_every': 10,
        'seed': 42,
    }

    print("Configuration:")
    for k, v in config.items():
        print(f"  {k}: {v}")
    print()

    # Setup
    random.seed(config['seed'])
    np.random.seed(config['seed'])
    torch.manual_seed(config['seed'])

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"CUDA Version: {torch.version.cuda}")
    print()

    # Initialize telemetry
    try:
        telemetry = SysfsHwmonTelemetry(sample_rate_hz=50.0)
        sample = telemetry.read_sample()
        print(f"Telemetry initialized: {sample.power_w:.1f}W, {sample.temp_edge_c:.1f}C")
        has_telemetry = True
    except Exception as e:
        print(f"Warning: Telemetry unavailable ({e}), using synthetic data only")
        telemetry = None
        has_telemetry = False
    print()

    # Create model
    model = TemporalCoherenceModel(
        input_dim=config['input_dim'],
        latent_dim=config['latent_dim'],
        hidden_dim=config['hidden_dim'],
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")
    print()

    # Optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=config['total_epochs'], eta_min=1e-5)

    # Training history
    history = []

    print("Training with Teacher Forcing Curriculum:")
    print("-" * 70)
    print(f"{'Epoch':>5} {'TF Ratio':>8} {'Smooth W':>8} {'Total':>8} "
          f"{'Recon':>8} {'I4_L1':>8} {'I4_L10':>8}")
    print("-" * 70)

    best_autocorr = -float('inf')
    training_start = time.time()

    for epoch in range(config['total_epochs']):
        # Get curriculum parameters
        tf_ratio = get_teacher_forcing_schedule(epoch, config['total_epochs'])
        smooth_weight = get_smoothness_weight_schedule(epoch, config['total_epochs'])

        # Train epoch
        metrics = train_epoch(
            model=model,
            optimizer=optimizer,
            batch_size=config['batch_size'],
            seq_len=config['seq_len'],
            teacher_forcing_ratio=tf_ratio,
            smoothness_weight=smooth_weight,
            contrastive_weight=config['contrastive_weight'],
            device=device,
        )

        scheduler.step()

        # Record
        epoch_data = TemporalMetrics(
            epoch=epoch,
            i4_autocorr_lag1=metrics['autocorr_lag1'],
            i4_autocorr_lag5=metrics['autocorr_lag5'],
            i4_autocorr_lag10=metrics['autocorr_lag10'],
            smoothness_loss=metrics['smoothness_loss'],
            contrastive_loss=metrics['contrastive_loss'],
            reconstruction_loss=metrics['reconstruction_loss'],
            total_loss=metrics['total_loss'],
            teacher_forcing_ratio=tf_ratio,
            hidden_state_norm=metrics['hidden_state_norm'],
            z_trajectory_variance=metrics['z_trajectory_variance'],
        )
        history.append(asdict(epoch_data))

        # Track best
        if metrics['autocorr_lag10'] > best_autocorr:
            best_autocorr = metrics['autocorr_lag10']

        # Print progress
        if epoch % 5 == 0 or epoch == config['total_epochs'] - 1:
            print(f"{epoch:>5} {tf_ratio:>8.3f} {smooth_weight:>8.3f} "
                  f"{metrics['total_loss']:>8.4f} {metrics['reconstruction_loss']:>8.4f} "
                  f"{metrics['autocorr_lag1']:>8.4f} {metrics['autocorr_lag10']:>8.4f}")

        # Evaluate on real telemetry periodically
        if has_telemetry and (epoch + 1) % config['eval_every'] == 0:
            real_metrics = evaluate_on_real_telemetry(model, telemetry, device)
            history[-1].update(real_metrics)
            print(f"  [Real] L1={real_metrics['real_autocorr_lag1']:.4f}, "
                  f"L10={real_metrics['real_autocorr_lag10']:.4f}")

    training_time = time.time() - training_start
    print("-" * 70)
    print(f"Training complete in {training_time:.1f}s")
    print()

    # Final evaluation
    print("=" * 70)
    print("FINAL EVALUATION")
    print("=" * 70)

    final_metrics = {
        'final_autocorr_lag1': history[-1]['i4_autocorr_lag1'],
        'final_autocorr_lag5': history[-1]['i4_autocorr_lag5'],
        'final_autocorr_lag10': history[-1]['i4_autocorr_lag10'],
        'best_autocorr_lag10': best_autocorr,
        'final_reconstruction_loss': history[-1]['reconstruction_loss'],
    }

    # Extended evaluation on held-out trajectory
    print("\nHeld-out trajectory evaluation (200 timesteps):")
    model.eval()
    with torch.no_grad():
        test_traj = generate_synthetic_trajectory(1, 200).to(device)
        z_gt = model.encode(test_traj)

        h = model.body_model.init_hidden(1, device)
        z_trajectory = [z_gt[:, 0]]
        z_prev = z_gt[:, 0]

        for t in range(1, 200):
            z_pred, h = model.body_model.step(z_prev, h)
            z_trajectory.append(z_pred)
            z_prev = z_pred

        z_trajectory = torch.stack(z_trajectory, dim=1)
        autocorrs = compute_autocorrelation(z_trajectory, max_lag=50)

    print(f"  Lag  1: {autocorrs.get(1, 0):.4f}")
    print(f"  Lag  5: {autocorrs.get(5, 0):.4f}")
    print(f"  Lag 10: {autocorrs.get(10, 0):.4f}")
    print(f"  Lag 20: {autocorrs.get(20, 0):.4f}")
    print(f"  Lag 50: {autocorrs.get(50, 0):.4f}")

    final_metrics['heldout_autocorr_lag1'] = autocorrs.get(1, 0)
    final_metrics['heldout_autocorr_lag5'] = autocorrs.get(5, 0)
    final_metrics['heldout_autocorr_lag10'] = autocorrs.get(10, 0)
    final_metrics['heldout_autocorr_lag20'] = autocorrs.get(20, 0)
    final_metrics['heldout_autocorr_lag50'] = autocorrs.get(50, 0)

    # Real telemetry final test
    if has_telemetry:
        print("\nReal telemetry final test (5s trajectory):")
        real_final = evaluate_on_real_telemetry(model, telemetry, device, duration_s=5.0)
        print(f"  Lag  1: {real_final['real_autocorr_lag1']:.4f}")
        print(f"  Lag  5: {real_final['real_autocorr_lag5']:.4f}")
        print(f"  Lag 10: {real_final['real_autocorr_lag10']:.4f}")
        final_metrics.update({f'real_final_{k}': v for k, v in real_final.items()})

    # Success criteria
    print("\n" + "=" * 70)
    print("SUCCESS CRITERIA CHECK")
    print("=" * 70)

    target_lag1 = 0.3
    target_lag10 = 0.3

    lag1_pass = final_metrics['heldout_autocorr_lag1'] > target_lag1
    lag10_pass = final_metrics['heldout_autocorr_lag10'] > target_lag10

    print(f"Target: I4_autocorr(lag=1) > {target_lag1}")
    print(f"  Result: {final_metrics['heldout_autocorr_lag1']:.4f} - {'PASS' if lag1_pass else 'FAIL'}")
    print(f"Target: I4_autocorr(lag=10) > {target_lag10}")
    print(f"  Result: {final_metrics['heldout_autocorr_lag10']:.4f} - {'PASS' if lag10_pass else 'FAIL'}")

    overall_success = lag1_pass and lag10_pass
    print(f"\nOVERALL: {'SUCCESS - Temporal coherence maintained!' if overall_success else 'NEEDS IMPROVEMENT'}")

    # Save results
    results = {
        'experiment': 'z1980_temporal_coherence_fixed',
        'timestamp': datetime.now().isoformat(),
        'config': config,
        'device': str(device),
        'gpu_name': torch.cuda.get_device_name() if torch.cuda.is_available() else None,
        'training_time_s': training_time,
        'total_params': total_params,
        'history': history,
        'final_metrics': final_metrics,
        'success': overall_success,
        'success_criteria': {
            'autocorr_lag1_target': target_lag1,
            'autocorr_lag10_target': target_lag10,
            'autocorr_lag1_pass': lag1_pass,
            'autocorr_lag10_pass': lag10_pass,
        },
        'solution_summary': {
            'approach': 'True Recurrent Body Model with GRU',
            'key_components': [
                'GRU with persistent hidden state',
                'Contrastive temporal loss',
                'Smoothness regularization',
                'Teacher forcing curriculum',
            ],
            'training_phases': [
                'Phase 1 (0-20): High TF (0.9->0.3), strong smoothness (0.5)',
                'Phase 2 (21-50): Moderate TF (0.3->0.1), moderate smoothness (0.2)',
                'Phase 3 (51-100): Low TF (0.1->0.0), minimal smoothness (0.05)',
            ],
        },
    }

    results_path = Path(__file__).parent.parent / 'results' / 'z1980_temporal_fixed.json'
    results_path.parent.mkdir(parents=True, exist_ok=True)

    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {results_path}")

    return results


if __name__ == "__main__":
    main()
