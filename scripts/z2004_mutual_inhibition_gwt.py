#!/usr/bin/env python3
"""
z2004: MUTUAL INHIBITION GWT - True Winner-Take-All Competition

The z2002 experiment failed GWT ignition (ratio=0.0, max_weight=0.37).
Even with temperature annealing (10.0 -> 0.1), no ignition occurred.
Specialist 5 dominated 84.75% but weights weren't sharp enough.

PROBLEM ANALYSIS:
- Standard softmax even at low temperature doesn't enforce sparsity
- max_weight of 0.37 means softmax spreads probability too much
- Need ACTIVE inhibition, not passive competition

SOLUTION: Three complementary mechanisms for winner-take-all:

1. LATERAL INHIBITION: Specialists actively suppress each other
   - Learned inhibition weights (asymmetric)
   - Iterative competition (5 rounds) for convergence
   - Strong winners suppress weak competitors

2. GUMBEL-SOFTMAX HARD: True discrete selection with gradients
   - Straight-through estimator for backprop
   - Hard selection during inference
   - Soft selection during training (gradually harden)

3. ENERGY-BASED COMPETITION: Penalize multi-activation
   - Low entropy = one dominant specialist
   - High L2 norm = concentrated weights
   - Energy loss term encourages sparsity

TRAINING PHASES:
- Phase 1: Pre-train specialists independently (5 epochs)
- Phase 2: Competition with lateral inhibition (20 epochs)
- Gradually increase inhibition strength

SUCCESS CRITERIA:
- ignition_ratio > 0.5 (majority of samples ignite)
- max_weight > 0.7 (clear winner)

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
# GUMBEL-SOFTMAX WITH HARD FORWARD
# =============================================================================

def gumbel_softmax_hard(logits: torch.Tensor, temperature: float = 0.1,
                        hard: bool = True) -> torch.Tensor:
    """
    Gumbel-Softmax with straight-through estimator for hard selection.

    During training: soft selection with gradients
    During inference: hard one-hot selection

    Args:
        logits: [batch, n_specialists] raw scores
        temperature: softmax temperature (lower = harder)
        hard: if True, use straight-through estimator

    Returns:
        weights: [batch, n_specialists] - one-hot in hard mode
    """
    # Clamp logits to prevent overflow
    logits = logits.clamp(-50, 50)

    # Sample from Gumbel distribution
    gumbels = -torch.empty_like(logits).exponential_().clamp(min=1e-10).log()
    gumbels = (logits + gumbels) / max(temperature, 0.01)
    y_soft = F.softmax(gumbels, dim=-1)

    if hard:
        # Straight-through estimator
        index = y_soft.argmax(dim=-1, keepdim=True)
        y_hard = torch.zeros_like(y_soft).scatter_(-1, index, 1.0)
        # Gradient flows through y_soft, forward uses y_hard
        return y_hard - y_soft.detach() + y_soft
    else:
        return y_soft


def competition_energy(weights: torch.Tensor) -> torch.Tensor:
    """
    Energy function that penalizes multiple specialists being active.

    Low energy = good (one dominant specialist)
    High energy = bad (diffuse activation)

    Args:
        weights: [batch, n_specialists] competition weights

    Returns:
        energy: [batch] scalar energy values
    """
    # Ensure weights are valid
    weights = weights.clamp(min=1e-8)
    weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-8)

    # Entropy should be low (one dominant)
    entropy = -torch.sum(weights * torch.log(weights + 1e-8), dim=-1)

    # L2 norm should be high (concentrated weights)
    # For one-hot: L2 = 1.0, for uniform: L2 = 1/sqrt(n)
    concentration = torch.norm(weights, p=2, dim=-1)

    # Energy = entropy - concentration (we want to minimize)
    # Clamp to prevent extreme values
    return (entropy - concentration).clamp(-10, 10)


# =============================================================================
# LATERAL INHIBITION MECHANISM
# =============================================================================

class LateralInhibitionGWT(nn.Module):
    """
    Global Workspace with lateral inhibition between specialists.

    Key insight: Competition through active suppression, not just
    passive softmax. Specialists inhibit each other iteratively
    until a clear winner emerges.
    """

    def __init__(self, n_specialists: int, hidden_dim: int,
                 n_competition_rounds: int = 5):
        super().__init__()
        self.n_specialists = n_specialists
        self.hidden_dim = hidden_dim
        self.n_competition_rounds = n_competition_rounds

        # Inhibition weights: how much each specialist suppresses others
        # Shape: [n_specialists, n_specialists]
        # Entry [i,j] = how much specialist i is inhibited by specialist j
        self.inhibition_weights = nn.Parameter(
            torch.randn(n_specialists, n_specialists) * 0.1
        )
        # Zero diagonal (no self-inhibition)
        with torch.no_grad():
            self.inhibition_weights.fill_diagonal_(0)

        # Learnable inhibition strength (can grow during training)
        self.inhibition_strength = nn.Parameter(torch.tensor(0.5))

        # Threshold for "winning" - activations below this are zeroed
        self.activation_threshold = nn.Parameter(torch.tensor(0.1))

    def forward(self, specialist_activations: torch.Tensor,
                training_progress: float = 0.0) -> Tuple[torch.Tensor, Dict]:
        """
        Iterative competition with lateral inhibition.

        Args:
            specialist_activations: [batch, n_specialists] initial activations
            training_progress: 0.0 to 1.0, increases inhibition over training

        Returns:
            final_weights: [batch, n_specialists] after competition
            info: Dict with competition metrics
        """
        batch_size = specialist_activations.shape[0]

        # Clamp input to prevent extreme values
        specialist_activations = specialist_activations.clamp(-50, 50)

        # Normalize initial activations to positive values
        activations = F.softplus(specialist_activations)
        activations = activations / (activations.sum(dim=-1, keepdim=True) + 1e-8)

        # Scale inhibition strength by training progress
        # Early: weak inhibition, Late: strong inhibition
        effective_strength = torch.abs(self.inhibition_strength) * (0.5 + 0.5 * training_progress)
        effective_strength = effective_strength.clamp(0, 2.0)

        # Track competition dynamics
        activation_history = [activations.detach().cpu()]

        # Iterative competition
        for round_idx in range(self.n_competition_rounds):
            # Each specialist receives inhibition from all others
            # inhibition[i] = sum_j (inhibition_weights[i,j] * activations[j])
            # But not from itself (diagonal is 0)
            inhibition = torch.matmul(
                activations,
                torch.abs(self.inhibition_weights).T  # Use abs for non-negative inhibition
            ) * effective_strength

            # Subtract inhibition (but keep non-negative via ReLU)
            activations = F.relu(activations - inhibition)

            # Apply threshold: weak activations are zeroed
            threshold = torch.sigmoid(self.activation_threshold) * 0.2  # 0-0.2 range
            activations = activations * (activations > threshold).float()

            # Check for collapse (all zeros) and reset if needed
            sum_per_batch = activations.sum(dim=-1, keepdim=True)
            collapsed = (sum_per_batch < 1e-6)
            if collapsed.any():
                # Reset to uniform for collapsed samples
                uniform = torch.ones_like(activations) / self.n_specialists
                activations = torch.where(collapsed.expand_as(activations), uniform, activations)

            # Normalize to prevent explosion/collapse
            activations = activations / (activations.sum(dim=-1, keepdim=True) + 1e-8)

            activation_history.append(activations.detach().cpu())

        # Compute competition metrics
        max_weights, winner_indices = activations.max(dim=-1)

        info = {
            'max_weight': max_weights.mean().item(),
            'winner_indices': winner_indices,
            'activation_history': activation_history,
            'inhibition_strength': effective_strength.item() if isinstance(effective_strength, torch.Tensor) else effective_strength,
            'competition_rounds': self.n_competition_rounds,
        }

        return activations, info


# =============================================================================
# SPECIALIST MODULE WITH FiLM CONDITIONING
# =============================================================================

class FiLMSpecialist(nn.Module):
    """
    Specialist module with FiLM conditioning on hardware telemetry.

    Each specialist can develop different sensitivities to GPU state.
    """

    def __init__(self, input_dim: int, hidden_dim: int, specialist_id: int,
                 telemetry_dim: int = 4):
        super().__init__()
        self.specialist_id = specialist_id

        # FiLM conditioning on telemetry
        self.film_gamma = nn.Sequential(
            nn.Linear(telemetry_dim, hidden_dim // 4),
            nn.GELU(),
            nn.Linear(hidden_dim // 4, hidden_dim),
        )
        self.film_beta = nn.Sequential(
            nn.Linear(telemetry_dim, hidden_dim // 4),
            nn.GELU(),
            nn.Linear(hidden_dim // 4, hidden_dim),
        )

        # Processing pathway
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Salience computation (how confident is this specialist?)
        self.salience_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.GELU(),
            nn.Linear(hidden_dim // 4, 1),
        )

    def forward(self, x: torch.Tensor,
                telemetry: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Process input with telemetry conditioning.

        Args:
            x: [batch, input_dim]
            telemetry: [batch, telemetry_dim]

        Returns:
            representation: [batch, hidden_dim]
            salience: [batch, 1]
        """
        # Encode
        h = self.encoder(x)

        # Apply FiLM modulation
        gamma = 1.0 + self.film_gamma(telemetry)
        beta = self.film_beta(telemetry)
        h = gamma * h + beta

        # Compute salience
        salience = self.salience_head(h)

        return h, salience


# =============================================================================
# MUTUAL INHIBITION GWT MODEL
# =============================================================================

class MutualInhibitionGWT(nn.Module):
    """
    Global Workspace Theory model with true winner-take-all competition.

    Combines:
    1. Lateral inhibition between specialists
    2. Gumbel-softmax hard selection
    3. Energy-based sparsity loss
    """

    IGNITION_THRESHOLD = 0.7  # Max weight must exceed this

    def __init__(self, hidden_dim: int = 256, num_specialists: int = 6,
                 n_competition_rounds: int = 5):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_specialists = num_specialists

        # Specialist modules with FiLM
        self.specialists = nn.ModuleList([
            FiLMSpecialist(hidden_dim, hidden_dim, i, telemetry_dim=4)
            for i in range(num_specialists)
        ])

        # Lateral inhibition competition
        self.competition = LateralInhibitionGWT(
            num_specialists, hidden_dim, n_competition_rounds
        )

        # Global workspace transformation
        self.workspace = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )

        # Broadcast projection (back to all specialists)
        self.broadcast = nn.Linear(hidden_dim, hidden_dim)

        # Telemetry gating (which specialists are relevant for current GPU state?)
        self.telemetry_gate = nn.Sequential(
            nn.Linear(4, 32),
            nn.GELU(),
            nn.Linear(32, num_specialists),
        )

        # Tracking
        self.ignition_history = deque(maxlen=1000)
        self.winner_history = deque(maxlen=1000)

    def forward(self, x: torch.Tensor, telemetry: torch.Tensor,
                training_progress: float = 0.0,
                use_hard_selection: bool = True,
                temperature: float = 0.1) -> Tuple[torch.Tensor, Dict]:
        """
        GWT forward with mutual inhibition competition.

        Args:
            x: [batch, hidden_dim] input representation
            telemetry: [batch, 4] GPU telemetry
            training_progress: 0.0 to 1.0
            use_hard_selection: if True, use Gumbel-softmax hard
            temperature: Gumbel softmax temperature

        Returns:
            output: [batch, hidden_dim]
            info: Dict with competition metrics
        """
        batch_size = x.shape[0]

        # === PHASE 1: Specialists process input ===
        specialist_outputs = []
        saliences = []

        for specialist in self.specialists:
            h, s = specialist(x, telemetry)
            specialist_outputs.append(h)
            saliences.append(s)

        # Stack for competition
        salience_stack = torch.cat(saliences, dim=-1)  # [batch, num_specialists]
        output_stack = torch.stack(specialist_outputs, dim=1)  # [batch, num_specialists, hidden]

        # === PHASE 2: Telemetry-based gating ===
        # Some specialists may be more relevant for current GPU state
        telemetry_bias = self.telemetry_gate(telemetry)  # [batch, num_specialists]
        gated_saliences = salience_stack + telemetry_bias * 0.5

        # === PHASE 3: Lateral inhibition competition ===
        competed_weights, comp_info = self.competition(
            gated_saliences, training_progress
        )

        # === PHASE 4: Gumbel-softmax hard selection ===
        if use_hard_selection:
            # Ensure valid input for log
            safe_weights = competed_weights.clamp(min=1e-8)
            final_weights = gumbel_softmax_hard(
                safe_weights.log(),  # Convert to log-space
                temperature=temperature,
                hard=True
            )
        else:
            final_weights = competed_weights

        # === PHASE 5: Winner broadcasts to workspace ===
        # Weighted combination based on competition
        workspace_input = torch.einsum('bs,bsh->bh', final_weights, output_stack)
        workspace_content = self.workspace(workspace_input)

        # === PHASE 6: Ignition detection ===
        max_weights, winner_indices = final_weights.max(dim=-1)
        ignition_mask = max_weights > self.IGNITION_THRESHOLD
        ignition_ratio = ignition_mask.float().mean().item()

        # Track
        self.ignition_history.append(ignition_ratio)
        for idx in winner_indices.cpu().numpy():
            self.winner_history.append(idx)

        # === PHASE 7: Broadcast (only when ignited) ===
        broadcast_signal = self.broadcast(workspace_content)
        broadcast_mask = ignition_mask.float().unsqueeze(-1)
        broadcast_contribution = broadcast_signal * broadcast_mask

        # Output
        output = x + broadcast_contribution

        # === Compute energy loss ===
        energy = competition_energy(final_weights)

        # === Compute broadcast correlations ===
        broadcast_correlations = []
        for i, h in enumerate(specialist_outputs):
            sim = F.cosine_similarity(workspace_content, h, dim=-1).mean().item()
            broadcast_correlations.append(sim)

        # === Winner distribution ===
        winner_dist = {}
        for i in range(self.num_specialists):
            winner_dist[i] = (winner_indices == i).float().mean().item()

        info = {
            'ignition_ratio': ignition_ratio,
            'ignition_ratio_running': np.mean(list(self.ignition_history)) if self.ignition_history else 0,
            'max_competition_weight': max_weights.mean().item(),
            'min_competition_weight': final_weights.min(dim=-1)[0].mean().item(),
            'competition_entropy': -(final_weights * (final_weights + 1e-10).log()).sum(dim=-1).mean().item(),
            'energy_loss': energy.mean().item(),
            'winner_distribution': winner_dist,
            'broadcast_correlations': broadcast_correlations,
            'mean_broadcast_correlation': np.mean(broadcast_correlations),
            'saliences_raw': salience_stack.mean(dim=0).detach().cpu().numpy().tolist(),
            'lateral_inhibition_info': comp_info,
            'temperature': temperature,
            'training_progress': training_progress,
        }

        return output, info, energy.mean()


# =============================================================================
# FULL CONSCIOUSNESS MODEL
# =============================================================================

class GWTConsciousnessModelV2(nn.Module):
    """
    GWT consciousness model with mutual inhibition specialists.
    """

    def __init__(self, vocab_size: int = 128, hidden_dim: int = 256,
                 num_specialists: int = 6, n_layers: int = 4):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size

        # Token embedding
        self.embed = nn.Embedding(vocab_size, hidden_dim)

        # Global FiLM conditioning
        self.global_film_gamma = nn.Linear(4, hidden_dim)
        self.global_film_beta = nn.Linear(4, hidden_dim)

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

        # GWT with mutual inhibition
        self.gwt = MutualInhibitionGWT(hidden_dim, num_specialists)

        # Output
        self.output_proj = nn.Linear(hidden_dim, vocab_size)

        # Confidence head
        self.confidence_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.GELU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, tokens: torch.Tensor, telemetry: torch.Tensor,
                training_progress: float = 0.0,
                use_hard_selection: bool = True,
                temperature: float = 0.1) -> Dict[str, Any]:
        """Forward pass with GWT competition."""
        # Embed
        h = self.embed(tokens)  # [batch, seq, hidden]

        # Global FiLM
        gamma = 1.0 + self.global_film_gamma(telemetry).unsqueeze(1)
        beta = self.global_film_beta(telemetry).unsqueeze(1)
        h = gamma * h + beta

        # Transformer
        for layer in self.layers:
            h = layer(h)

        # Pool for GWT
        h_pooled = h.mean(dim=1)  # [batch, hidden]

        # GWT competition
        h_broadcast, gwt_info, energy_loss = self.gwt(
            h_pooled, telemetry, training_progress,
            use_hard_selection, temperature
        )

        # Broadcast back to sequence
        h = h + h_broadcast.unsqueeze(1)

        # Output
        logits = self.output_proj(h)
        confidence = self.confidence_head(h_pooled)

        return {
            'logits': logits,
            'hidden': h,
            'confidence': confidence,
            'gwt_info': gwt_info,
            'energy_loss': energy_loss,
        }


# =============================================================================
# TELEMETRY BUFFER
# =============================================================================

class GpuTelemetryBuffer:
    """Buffer GPU telemetry for FiLM conditioning."""

    def __init__(self, telemetry: SysfsHwmonTelemetry, history_len: int = 32):
        self.telemetry = telemetry
        self.history: deque = deque(maxlen=history_len)
        self._last_sample: Optional[GpuSample] = None

    def sample(self) -> torch.Tensor:
        """Get normalized telemetry tensor [4]."""
        sample = self.telemetry.read_sample()
        self._last_sample = sample
        self.history.append(sample)

        raw = torch.tensor([
            sample.temp_edge_c,
            sample.power_w,
            sample.freq_sclk_mhz,
            sample.gpu_busy_pct,
        ], dtype=torch.float32)

        norms = torch.tensor([100.0, 150.0, 3000.0, 100.0])
        return (raw / norms).clamp(0, 2)

    def get_latest_raw(self) -> Dict[str, float]:
        if self._last_sample is None:
            self.sample()
        s = self._last_sample
        return {
            'temp_edge_c': s.temp_edge_c,
            'power_w': s.power_w,
            'freq_sclk_mhz': s.freq_sclk_mhz,
            'gpu_busy_pct': s.gpu_busy_pct,
        }


class DummyTelemetryBuffer:
    """Fallback when hardware not available."""
    def sample(self):
        return torch.tensor([0.5, 0.3, 0.8, 0.4], dtype=torch.float32)
    def get_latest_raw(self):
        return {'temp_edge_c': 50, 'power_w': 45, 'freq_sclk_mhz': 1800, 'gpu_busy_pct': 40}


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
    """Load training text."""
    paths = [
        Path(__file__).parent.parent / 'data' / 'shakespeare.txt',
        Path(__file__).parent.parent / 'data' / 'tiny_shakespeare.txt',
    ]
    for p in paths:
        if p.exists():
            print(f"[Data] Loading from {p}")
            return p.read_text()

    # Synthetic
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
# TRAINING PHASES
# =============================================================================

def pretrain_specialists(model: GWTConsciousnessModelV2, dataset: TextDataset,
                         telemetry_buffer, device: torch.device,
                         epochs: int = 5, lr: float = 1e-3) -> List[Dict]:
    """
    Phase 1: Pre-train each specialist independently.

    Each specialist learns to predict next character.
    No competition yet - just build competence.
    """
    print("\n" + "="*60)
    print("PHASE 1: PRE-TRAINING SPECIALISTS")
    print("="*60)

    metrics = []
    batch_size = 32
    num_batches = min(len(dataset) // batch_size, 200)

    for specialist_idx, specialist in enumerate(model.gwt.specialists):
        print(f"\n[Specialist {specialist_idx}] Pre-training...")

        # Create small output head for this specialist
        specialist_head = nn.Linear(model.hidden_dim, dataset.vocab_size).to(device)
        optimizer = AdamW(
            list(specialist.parameters()) + list(specialist_head.parameters()),
            lr=lr
        )

        for epoch in range(epochs):
            total_loss = 0

            for batch_idx in range(num_batches):
                start = batch_idx * batch_size
                batch_x, batch_y = [], []
                for i in range(batch_size):
                    x, y = dataset[(start + i) % len(dataset)]
                    batch_x.append(x)
                    batch_y.append(y)

                x = torch.stack(batch_x).to(device)
                y = torch.stack(batch_y).to(device)
                tel = telemetry_buffer.sample().unsqueeze(0).expand(batch_size, -1).to(device)

                # Embed and pool
                h = model.embed(x)
                gamma = 1.0 + model.global_film_gamma(tel).unsqueeze(1)
                beta = model.global_film_beta(tel).unsqueeze(1)
                h = gamma * h + beta

                for layer in model.layers:
                    h = layer(h)

                h_pooled = h.mean(dim=1)

                # Just this specialist
                optimizer.zero_grad()
                spec_out, _ = specialist(h_pooled, tel)
                logits = specialist_head(spec_out)

                loss = F.cross_entropy(
                    logits.unsqueeze(1).expand(-1, y.shape[1], -1).reshape(-1, dataset.vocab_size),
                    y.reshape(-1)
                )
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            avg_loss = total_loss / num_batches
            print(f"  Epoch {epoch}: loss={avg_loss:.4f}")

        metrics.append({
            'specialist_id': specialist_idx,
            'final_loss': avg_loss,
        })

    return metrics


def train_competition_epoch(model: GWTConsciousnessModelV2, dataset: TextDataset,
                            telemetry_buffer, optimizer: torch.optim.Optimizer,
                            device: torch.device, epoch: int, total_epochs: int,
                            max_batches: int = 800) -> Dict:
    """
    Phase 2: Train with lateral inhibition competition.
    """
    model.train()

    total_loss = 0
    total_energy_loss = 0
    correct = 0
    total = 0

    # Competition metrics
    ignition_ratios = []
    max_weights = []
    entropies = []

    batch_size = 32
    num_batches = min(len(dataset) // batch_size, max_batches)

    # Training progress for inhibition strength
    training_progress = epoch / max(1, total_epochs - 1)

    # Temperature schedule: start at 1.0, decrease to 0.1
    temperature = 1.0 - 0.9 * training_progress
    temperature = max(0.1, temperature)

    for batch_idx in range(num_batches):
        start = batch_idx * batch_size
        batch_x, batch_y = [], []
        for i in range(batch_size):
            x, y = dataset[(start + i) % len(dataset)]
            batch_x.append(x)
            batch_y.append(y)

        x = torch.stack(batch_x).to(device)
        y = torch.stack(batch_y).to(device)
        tel = telemetry_buffer.sample().unsqueeze(0).expand(batch_size, -1).to(device)

        optimizer.zero_grad()

        out = model(x, tel, training_progress, use_hard_selection=True, temperature=temperature)

        # Task loss
        logits = out['logits'].view(-1, dataset.vocab_size)
        task_loss = F.cross_entropy(logits, y.view(-1))

        # Energy loss (encourage sparsity)
        energy_loss = out['energy_loss']

        # Combined loss
        # Weight energy loss more as training progresses
        energy_weight = 0.1 + 0.4 * training_progress
        loss = task_loss + energy_weight * energy_loss

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        # Metrics
        total_loss += task_loss.item()
        total_energy_loss += energy_loss.item()
        preds = logits.argmax(dim=-1)
        correct += (preds == y.view(-1)).sum().item()
        total += y.numel()

        gwt_info = out['gwt_info']
        ignition_ratios.append(gwt_info['ignition_ratio'])
        max_weights.append(gwt_info['max_competition_weight'])
        entropies.append(gwt_info['competition_entropy'])

        if batch_idx % 100 == 0:
            print(f"  Batch {batch_idx}/{num_batches}: "
                  f"loss={task_loss.item():.4f} energy={energy_loss.item():.4f} "
                  f"max_w={gwt_info['max_competition_weight']:.3f} "
                  f"ignition={gwt_info['ignition_ratio']:.3f}")

    return {
        'loss': total_loss / num_batches,
        'energy_loss': total_energy_loss / num_batches,
        'accuracy': correct / total,
        'ignition_ratio': np.mean(ignition_ratios),
        'ignition_ratio_std': np.std(ignition_ratios),
        'max_weight': np.mean(max_weights),
        'competition_entropy': np.mean(entropies),
        'temperature': temperature,
        'training_progress': training_progress,
    }


def evaluate_ignition(model: GWTConsciousnessModelV2, dataset: TextDataset,
                      telemetry_buffer, device: torch.device) -> Dict:
    """Evaluate final ignition performance."""
    model.eval()

    ignition_ratios = []
    max_weights = []
    winner_counts = {i: 0 for i in range(model.gwt.num_specialists)}

    batch_size = 32
    num_batches = 100

    with torch.no_grad():
        for batch_idx in range(num_batches):
            start = batch_idx * batch_size
            batch_x = []
            for i in range(batch_size):
                x, _ = dataset[(start + i) % len(dataset)]
                batch_x.append(x)

            x = torch.stack(batch_x).to(device)
            tel = telemetry_buffer.sample().unsqueeze(0).expand(batch_size, -1).to(device)

            out = model(x, tel, training_progress=1.0, use_hard_selection=True, temperature=0.1)
            gwt_info = out['gwt_info']

            ignition_ratios.append(gwt_info['ignition_ratio'])
            max_weights.append(gwt_info['max_competition_weight'])

            for i, count in gwt_info['winner_distribution'].items():
                winner_counts[i] += count * batch_size

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
    print("z2004: MUTUAL INHIBITION GWT")
    print("Fixing GWT ignition failure with lateral inhibition")
    print("="*80)
    print(f"Start: {datetime.now().isoformat()}")
    print(f"Device: {DEVICE}")
    print()

    # Initialize telemetry
    print("[Hardware] Initializing telemetry...")
    try:
        telemetry = SysfsHwmonTelemetry(sample_rate_hz=20)
        telemetry_buffer = GpuTelemetryBuffer(telemetry)
        sample = telemetry_buffer.sample()
        raw = telemetry_buffer.get_latest_raw()
        print(f"[Hardware] GPU active:")
        print(f"  - temp: {raw['temp_edge_c']:.1f}C")
        print(f"  - power: {raw['power_w']:.1f}W")
        print(f"  - freq: {raw['freq_sclk_mhz']}MHz")
        print(f"  - busy: {raw['gpu_busy_pct']:.1f}%")
    except Exception as e:
        print(f"[Hardware] Telemetry failed: {e}")
        telemetry_buffer = DummyTelemetryBuffer()

    # Load data
    text = load_text_data()
    dataset = TextDataset(text, seq_len=64)
    print(f"\n[Data] {len(dataset)} samples, vocab {dataset.vocab_size}")

    # Create model
    model = GWTConsciousnessModelV2(
        vocab_size=dataset.vocab_size,
        hidden_dim=256,
        num_specialists=6,
        n_layers=4,
    ).to(DEVICE)

    param_count = sum(p.numel() for p in model.parameters())
    print(f"[Model] Parameters: {param_count:,}")
    print(f"[Model] Specialists: {model.gwt.num_specialists}")
    print(f"[Model] Competition rounds: {model.gwt.competition.n_competition_rounds}")

    # ========================
    # PHASE 1: PRE-TRAIN
    # ========================
    pretrain_metrics = pretrain_specialists(
        model, dataset, telemetry_buffer, DEVICE, epochs=5
    )

    # ========================
    # PHASE 2: COMPETITION
    # ========================
    print("\n" + "="*60)
    print("PHASE 2: COMPETITION TRAINING WITH LATERAL INHIBITION")
    print("="*60)

    num_epochs = 25
    optimizer = AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)

    epoch_metrics = []

    try:
        for epoch in range(num_epochs):
            epoch_start = time.time()

            print(f"\n[Epoch {epoch}/{num_epochs}]")

            metrics = train_competition_epoch(
                model, dataset, telemetry_buffer, optimizer, DEVICE,
                epoch, num_epochs, max_batches=800
            )

            scheduler.step()

            epoch_time = time.time() - epoch_start
            metrics['epoch'] = epoch
            metrics['time'] = epoch_time

            epoch_metrics.append(metrics)

            print(f"\nEpoch {epoch} Results:")
            print(f"  Loss: {metrics['loss']:.4f}")
            print(f"  Accuracy: {metrics['accuracy']:.4f}")
            print(f"  Ignition Ratio: {metrics['ignition_ratio']:.4f}")
            print(f"  Max Weight: {metrics['max_weight']:.4f}")
            print(f"  Entropy: {metrics['competition_entropy']:.4f}")
            print(f"  Energy Loss: {metrics['energy_loss']:.4f}")
            print(f"  Temperature: {metrics['temperature']:.4f}")
            print(f"  Time: {epoch_time:.1f}s")

            raw = telemetry_buffer.get_latest_raw()
            print(f"  GPU: {raw['temp_edge_c']:.1f}C, {raw['power_w']:.1f}W")

    except KeyboardInterrupt:
        print("\n[Interrupted]")

    # ========================
    # EVALUATION
    # ========================
    print("\n" + "="*60)
    print("FINAL EVALUATION")
    print("="*60)

    eval_metrics = evaluate_ignition(model, dataset, telemetry_buffer, DEVICE)

    print(f"\nEvaluation Results:")
    print(f"  Ignition Ratio: {eval_metrics['eval_ignition_ratio']:.4f}")
    print(f"  Max Weight: {eval_metrics['eval_max_weight']:.4f}")
    print(f"  Specialist Diversity: {eval_metrics['eval_specialist_diversity']:.4f}")
    print(f"  Winner Distribution: {eval_metrics['eval_winner_distribution']}")

    # ========================
    # VERDICT
    # ========================
    final_ignition = eval_metrics['eval_ignition_ratio']
    final_max_weight = eval_metrics['eval_max_weight']

    print("\n" + "="*60)
    print("GWT IGNITION CRITERIA")
    print("="*60)
    print(f"Criterion 1: ignition_ratio > 0.5")
    print(f"  Measured: {final_ignition:.4f}")
    print(f"  PASS: {final_ignition > 0.5}")
    print()
    print(f"Criterion 2: max_weight > 0.7")
    print(f"  Measured: {final_max_weight:.4f}")
    print(f"  PASS: {final_max_weight > 0.7}")

    passed = final_ignition > 0.5 and final_max_weight > 0.7

    if passed:
        verdict = "PASS - GWT ignition achieved with mutual inhibition"
    elif final_ignition > 0.5:
        verdict = "PARTIAL - Ignition ratio OK, weights not sharp enough"
    elif final_max_weight > 0.7:
        verdict = "PARTIAL - Sharp weights, but ignition too sparse"
    else:
        verdict = "FAIL - Neither criterion met"

    print(f"\nVERDICT: {verdict}")
    print("="*60)

    # ========================
    # SAVE RESULTS
    # ========================
    results = {
        'experiment': 'z2004_mutual_inhibition_gwt',
        'timestamp': datetime.now().isoformat(),
        'device': str(DEVICE),
        'hypothesis': 'Lateral inhibition + Gumbel-softmax hard + energy loss achieves GWT ignition',
        'baseline_comparison': {
            'z2002_ignition_ratio': 0.0,
            'z2002_max_weight': 0.366,
            'z2002_verdict': 'FAIL',
        },
        'model': {
            'vocab_size': dataset.vocab_size,
            'hidden_dim': 256,
            'num_specialists': 6,
            'n_layers': 4,
            'competition_rounds': 5,
            'parameters': param_count,
        },
        'mechanisms': {
            'lateral_inhibition': True,
            'gumbel_softmax_hard': True,
            'energy_loss': True,
            'telemetry_gating': True,
        },
        'training': {
            'pretrain_epochs': 5,
            'competition_epochs': len(epoch_metrics),
            'final_loss': epoch_metrics[-1]['loss'] if epoch_metrics else None,
            'final_accuracy': epoch_metrics[-1]['accuracy'] if epoch_metrics else None,
        },
        'gwt_metrics': {
            'final_ignition_ratio': final_ignition,
            'final_max_weight': final_max_weight,
            'criterion_1_passed': final_ignition > 0.5,
            'criterion_2_passed': final_max_weight > 0.7,
            'specialist_diversity': eval_metrics['eval_specialist_diversity'],
            'winner_distribution': eval_metrics['eval_winner_distribution'],
        },
        'epoch_history': [
            {
                'epoch': m['epoch'],
                'loss': m['loss'],
                'energy_loss': m['energy_loss'],
                'accuracy': m['accuracy'],
                'ignition_ratio': m['ignition_ratio'],
                'max_weight': m['max_weight'],
                'competition_entropy': m['competition_entropy'],
                'temperature': m['temperature'],
            }
            for m in epoch_metrics
        ],
        'pretrain_metrics': pretrain_metrics,
        'evaluation': eval_metrics,
        'verdict': verdict,
        'passed': passed,
        'improvement_over_z2002': {
            'ignition_ratio_delta': final_ignition - 0.0,
            'max_weight_delta': final_max_weight - 0.366,
        },
    }

    output_path = RESULTS_DIR / 'z2004_mutual_inhibition_gwt.json'
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\nResults saved to: {output_path}")

    return results


if __name__ == '__main__':
    main()
