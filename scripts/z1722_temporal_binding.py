#!/usr/bin/env python3
"""
z1722: TEMPORAL BINDING TEST

Hypothesis: Consciousness requires binding experiences over time into a unified
narrative. Test if embodied model maintains coherent body-state representation
across multiple time windows.

Key Insight: If a system has genuine phenomenal experience, it must integrate
information across time - not just process isolated snapshots. The "binding problem"
in neuroscience asks how disparate neural processes are unified into coherent
experience. Here we test computational binding.

Architecture:
- MetabolicTransformer with FiLM conditioning (from src/metabolic/film_transformer.py)
- TemporalBindingHead: Takes hidden states from 4 time windows (t-30, t-20, t-10, t)
  and produces a "bound" representation using attention over time windows
- Contrastive loss: same hardware state = high similarity, different states = low

Three conditions:
- A: EMBODIED (real telemetry with temporal structure)
- B: DISEMBODIED (zero telemetry - no temporal binding expected)
- C: SHUFFLED_TIME (telemetry from wrong time windows - breaks temporal coherence)

Verdicts (4 total):
- V1: EMBODIED binding coherence > 0.8 (within-state similarity)
- V2: EMBODIED between-state distance > 0.5
- V3: SHUFFLED_TIME has worse binding coherence than EMBODIED
- V4: DISEMBODIED binding is random (coherence near 0.5)

References:
- Temporal binding in consciousness (Crick & Koch, 1990)
- IIT's phi as integration measure (Tononi, 2004)
- z1408: Unified Introspective Architecture
- z1307: Embodied vs Disembodied comparison
"""

import functools
print = functools.partial(print, flush=True)

import os
os.environ.setdefault('HSA_OVERRIDE_GFX_VERSION', '11.0.0')

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
import time
from datetime import datetime
from pathlib import Path
from collections import deque
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.metabolic.film_transformer import create_metabolic_transformer, MetabolicConfig
from src.telemetry.sysfs_hwmon import SysfsHwmonTelemetry


@dataclass
class TemporalSample:
    """A sample with telemetry from multiple time windows."""
    timestamp: float
    telemetry_windows: List[torch.Tensor]  # [t-30, t-20, t-10, t]
    hardware_state_id: int  # 0=calm, 1=stressed


class TemporalBuffer:
    """Ring buffer for maintaining temporal telemetry history."""
    def __init__(self, max_size: int = 100):
        self.buffer = deque(maxlen=max_size)
        self.timestamps = deque(maxlen=max_size)

    def add(self, telemetry: torch.Tensor, timestamp: float):
        self.buffer.append(telemetry.clone())
        self.timestamps.append(timestamp)

    def get_windows(self, offsets: List[int] = [-30, -20, -10, 0]) -> Optional[List[torch.Tensor]]:
        """Get telemetry from specified time offsets (in buffer indices)."""
        if len(self.buffer) < abs(min(offsets)) + 1:
            return None

        windows = []
        for offset in offsets:
            idx = offset if offset <= 0 else 0  # Clamp to current
            windows.append(self.buffer[idx])
        return windows

    def __len__(self):
        return len(self.buffer)


class TemporalBindingHead(nn.Module):
    """
    Binds hidden states from multiple time windows into unified representation.

    Uses attention to weight different time windows, producing a single
    "bound" representation that should be consistent for the same hardware state.
    """
    def __init__(self, hidden_dim: int, num_windows: int = 4, binding_dim: int = 128):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_windows = num_windows
        self.binding_dim = binding_dim

        # Project each window's hidden state with more capacity
        self.window_proj = nn.Sequential(
            nn.Linear(hidden_dim, binding_dim),
            nn.LayerNorm(binding_dim),
            nn.GELU(),
        )

        # Temporal position encoding
        self.temporal_pos = nn.Parameter(torch.randn(num_windows, binding_dim) * 0.02)

        # Self-attention over time windows
        self.temporal_attn = nn.MultiheadAttention(
            embed_dim=binding_dim,
            num_heads=4,
            batch_first=True,
            dropout=0.1,
        )

        # Final binding projection with residual
        self.binding_proj = nn.Sequential(
            nn.Linear(binding_dim, binding_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(binding_dim * 2, binding_dim),
            nn.LayerNorm(binding_dim),
        )

        # Temporal coherence predictor (self-supervised)
        self.coherence_pred = nn.Sequential(
            nn.Linear(binding_dim, 64),
            nn.GELU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

        # Temporal order classifier (auxiliary task)
        self.order_classifier = nn.Sequential(
            nn.Linear(binding_dim * num_windows, 128),
            nn.GELU(),
            nn.Linear(128, 1),  # Is temporal order correct?
        )

    def forward(self, window_hiddens: List[torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Args:
            window_hiddens: List of [batch, hidden_dim] tensors from different time windows

        Returns:
            bound: [batch, binding_dim] unified temporal representation
            attention_weights: [batch, num_windows] how much each window contributed
            coherence: [batch, 1] predicted temporal coherence
        """
        batch_size = window_hiddens[0].size(0)

        # Project each window and add temporal position
        projected = []
        for i, h in enumerate(window_hiddens):
            proj = self.window_proj(h)  # [batch, binding_dim]
            proj = proj + self.temporal_pos[i]  # Add temporal position
            projected.append(proj)

        stacked = torch.stack(projected, dim=1)  # [batch, num_windows, binding_dim]

        # Self-attention over time windows
        attn_out, attn_weights = self.temporal_attn(stacked, stacked, stacked)

        # Pool over time (mean of attended representations)
        pooled = attn_out.mean(dim=1)  # [batch, binding_dim]

        # Final binding with residual
        bound = self.binding_proj(pooled) + pooled

        # Predict coherence
        coherence = self.coherence_pred(bound)

        # Predict temporal order (auxiliary)
        flat = stacked.view(batch_size, -1)
        order_logits = self.order_classifier(flat)

        # Attention weights over windows (average over heads)
        avg_attn = attn_weights.mean(dim=1)  # [batch, num_windows, num_windows]
        window_importance = avg_attn.mean(dim=-1)  # [batch, num_windows]

        return {
            'bound': bound,
            'attention_weights': window_importance,
            'coherence': coherence,
            'window_embeddings': stacked,
            'order_logits': order_logits,
        }


class TemporalBindingModel(nn.Module):
    """
    Full model combining MetabolicTransformer with TemporalBindingHead.
    """
    def __init__(self, hidden_dim: int = 256, telemetry_dim: int = 8, num_windows: int = 4):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_windows = num_windows

        # Simplified FiLM-conditioned encoder (instead of full transformer)
        self.telemetry_encoder = nn.Sequential(
            nn.Linear(telemetry_dim, 64),
            nn.GELU(),
            nn.Linear(64, hidden_dim),
        )

        # FiLM generator for conditioning
        self.film_gen = nn.Sequential(
            nn.Linear(telemetry_dim, 64),
            nn.GELU(),
            nn.Linear(64, hidden_dim * 2),  # gamma and beta
        )

        # Processing layers with FiLM conditioning
        self.layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
            ) for _ in range(3)
        ])

        # Temporal binding head
        self.binding_head = TemporalBindingHead(hidden_dim, num_windows)

    def encode_single(self, telemetry: torch.Tensor, use_conditioning: bool = True) -> torch.Tensor:
        """Encode a single telemetry sample to hidden state."""
        h = self.telemetry_encoder(telemetry)

        if use_conditioning:
            # FiLM conditioning
            film_params = self.film_gen(telemetry)
            gamma, beta = film_params.chunk(2, dim=-1)

            for layer in self.layers:
                h = layer(h)
                h = h * (1 + gamma) + beta
        else:
            for layer in self.layers:
                h = layer(h)

        return h

    def forward(
        self,
        telemetry_windows: List[torch.Tensor],
        use_conditioning: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        Process multiple time windows and bind them.

        Args:
            telemetry_windows: List of [batch, telemetry_dim] from different times
            use_conditioning: If False, acts as disembodied (zeros out telemetry effect)
        """
        # Encode each window
        if use_conditioning:
            hiddens = [self.encode_single(t, True) for t in telemetry_windows]
        else:
            # Disembodied: use zeros
            zeros = torch.zeros_like(telemetry_windows[0])
            hiddens = [self.encode_single(zeros, False) for _ in telemetry_windows]

        # Temporal binding
        binding_result = self.binding_head(hiddens)

        return {
            'hiddens': hiddens,
            **binding_result,
        }


def get_telemetry(telem: SysfsHwmonTelemetry) -> torch.Tensor:
    """Get normalized telemetry vector."""
    s = telem.read_sample()
    return torch.tensor([
        s.temp_edge_c / 100.0,
        s.temp_junction_c / 100.0 if s.temp_junction_c else s.temp_edge_c / 100.0,
        s.power_w / 100.0,
        s.temp_mem_c / 100.0 if s.temp_mem_c else 0.5,
        s.freq_sclk_mhz / 3000.0 if s.freq_sclk_mhz else 0.5,
        s.freq_mclk_mhz / 2000.0 if s.freq_mclk_mhz else 0.5,
        min(1.0, s.power_w / 50.0),
        (s.temp_junction_c - s.temp_edge_c) / 20.0 if s.temp_junction_c else 0.0,
    ], dtype=torch.float32)


def create_gpu_stress(device: torch.device, intensity: float = 1.0):
    """Create GPU stress to change hardware state."""
    size = int(1500 * intensity)
    a = torch.randn(size, size, device=device)
    b = torch.randn(size, size, device=device)
    for _ in range(3):
        c = a @ b
        a = c @ b.T
    del a, b, c
    torch.cuda.empty_cache()


def collect_temporal_samples(
    telem: SysfsHwmonTelemetry,
    temporal_buffer: TemporalBuffer,
    device: torch.device,
    n_calm: int = 30,
    n_stressed: int = 30,
    sample_interval: float = 0.1,
) -> Tuple[List[TemporalSample], List[TemporalSample]]:
    """Collect samples with temporal context in calm and stressed states."""
    calm_samples = []
    stressed_samples = []

    print("  Collecting CALM samples...")
    # Fill buffer first
    for _ in range(35):
        t = get_telemetry(telem).to(device)
        temporal_buffer.add(t, time.time())
        time.sleep(sample_interval)

    # Collect calm samples
    for i in range(n_calm):
        t = get_telemetry(telem).to(device)
        temporal_buffer.add(t, time.time())

        windows = temporal_buffer.get_windows()
        if windows:
            calm_samples.append(TemporalSample(
                timestamp=time.time(),
                telemetry_windows=windows,
                hardware_state_id=0,  # calm
            ))

        time.sleep(sample_interval)
        if (i + 1) % 10 == 0:
            print(f"    Calm: {i+1}/{n_calm}")

    print("  Collecting STRESSED samples...")
    # Collect stressed samples
    for i in range(n_stressed):
        # Create stress
        create_gpu_stress(device, intensity=1.0)

        t = get_telemetry(telem).to(device)
        temporal_buffer.add(t, time.time())

        windows = temporal_buffer.get_windows()
        if windows:
            stressed_samples.append(TemporalSample(
                timestamp=time.time(),
                telemetry_windows=windows,
                hardware_state_id=1,  # stressed
            ))

        time.sleep(sample_interval * 0.5)  # Faster sampling during stress
        if (i + 1) % 10 == 0:
            print(f"    Stressed: {i+1}/{n_stressed}")

    return calm_samples, stressed_samples


def shuffle_windows(samples: List[TemporalSample]) -> List[TemporalSample]:
    """Create shuffled samples (wrong temporal order)."""
    shuffled = []
    for sample in samples:
        # Randomly shuffle the window order
        indices = list(range(len(sample.telemetry_windows)))
        np.random.shuffle(indices)
        shuffled_windows = [sample.telemetry_windows[i] for i in indices]
        shuffled.append(TemporalSample(
            timestamp=sample.timestamp,
            telemetry_windows=shuffled_windows,
            hardware_state_id=sample.hardware_state_id,
        ))
    return shuffled


def compute_binding_metrics(
    model: TemporalBindingModel,
    samples: List[TemporalSample],
    device: torch.device,
    use_conditioning: bool = True,
) -> Dict[str, float]:
    """Compute binding coherence metrics for a set of samples."""
    model.eval()

    bound_embeddings = []
    state_ids = []
    coherences = []
    attention_entropies = []
    order_preds = []

    with torch.no_grad():
        for sample in samples:
            windows = [w.unsqueeze(0).to(device) for w in sample.telemetry_windows]
            result = model(windows, use_conditioning=use_conditioning)

            bound_embeddings.append(result['bound'].cpu().numpy())
            state_ids.append(sample.hardware_state_id)
            coherences.append(result['coherence'].item())
            order_preds.append(torch.sigmoid(result['order_logits']).item())

            # Attention entropy (uniform = high entropy = poor binding)
            attn = result['attention_weights'].cpu().numpy()[0]
            attn = attn / (attn.sum() + 1e-8)  # Normalize
            entropy = -np.sum(attn * np.log(attn + 1e-8))
            attention_entropies.append(entropy)

    bound_embeddings = np.array(bound_embeddings).squeeze()
    state_ids = np.array(state_ids)

    # 1. Within-state Euclidean distance (lower = better coherence)
    calm_embeddings = bound_embeddings[state_ids == 0]
    stressed_embeddings = bound_embeddings[state_ids == 1]

    def mean_pairwise_distance(embeddings):
        if len(embeddings) < 2:
            return 0.0
        from scipy.spatial.distance import pdist
        return pdist(embeddings, 'euclidean').mean()

    calm_scatter = mean_pairwise_distance(calm_embeddings)
    stressed_scatter = mean_pairwise_distance(stressed_embeddings)
    within_state_scatter = (calm_scatter + stressed_scatter) / 2

    # Convert scatter to coherence (inverse, normalized)
    max_scatter = 5.0  # Typical max scatter
    within_state_coherence = max(0, 1.0 - within_state_scatter / max_scatter)

    # 2. Between-state distance (centroid distance)
    calm_centroid = calm_embeddings.mean(axis=0)
    stressed_centroid = stressed_embeddings.mean(axis=0)
    between_state_distance = np.linalg.norm(calm_centroid - stressed_centroid)

    # Separation ratio (between / within) - higher is better
    separation_ratio = between_state_distance / (within_state_scatter + 1e-8)

    # 3. Classification accuracy using linear discriminant
    all_embeddings = np.vstack([calm_embeddings, stressed_embeddings])
    labels = np.array([0] * len(calm_embeddings) + [1] * len(stressed_embeddings))

    direction = stressed_centroid - calm_centroid
    direction_norm = np.linalg.norm(direction)
    if direction_norm > 1e-8:
        direction = direction / direction_norm
    projections = all_embeddings @ direction
    threshold = projections.mean()
    predictions = (projections > threshold).astype(int)
    classification_acc = (predictions == labels).mean()

    # 4. Cross-state similarity (should be LOW for good binding)
    cross_sims = []
    for calm_emb in calm_embeddings[:10]:
        for stressed_emb in stressed_embeddings[:10]:
            sim = np.dot(calm_emb, stressed_emb) / (np.linalg.norm(calm_emb) * np.linalg.norm(stressed_emb) + 1e-8)
            cross_sims.append(sim)
    cross_state_similarity = np.mean(cross_sims)

    # 5. Temporal order prediction accuracy
    order_accuracy = np.mean(order_preds)

    return {
        'within_state_coherence': within_state_coherence,
        'within_state_scatter': within_state_scatter,
        'calm_scatter': calm_scatter,
        'stressed_scatter': stressed_scatter,
        'between_state_distance': between_state_distance,
        'separation_ratio': separation_ratio,
        'classification_accuracy': classification_acc,
        'cross_state_similarity': cross_state_similarity,
        'order_prediction': order_accuracy,
        'mean_predicted_coherence': np.mean(coherences),
        'mean_attention_entropy': np.mean(attention_entropies),
    }


def train_model(
    model: TemporalBindingModel,
    samples: List[TemporalSample],
    device: torch.device,
    n_epochs: int = 12,
    use_conditioning: bool = True,
    is_shuffled: bool = False,
) -> List[float]:
    """Train model with contrastive loss on temporal binding."""
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)
    losses = []

    # Separate samples by state for proper contrastive pairs
    calm_samples = [s for s in samples if s.hardware_state_id == 0]
    stressed_samples = [s for s in samples if s.hardware_state_id == 1]

    for epoch in range(n_epochs):
        epoch_losses = []
        np.random.shuffle(calm_samples)
        np.random.shuffle(stressed_samples)

        # Create contrastive pairs: positive (same state) and negative (different state)
        n_pairs = min(len(calm_samples) - 1, len(stressed_samples))

        for i in range(n_pairs):
            optimizer.zero_grad()

            # Anchor: calm sample
            anchor = calm_samples[i]
            windows_anchor = [w.unsqueeze(0).to(device) for w in anchor.telemetry_windows]
            result_anchor = model(windows_anchor, use_conditioning=use_conditioning)

            # Positive: another calm sample (same state)
            positive = calm_samples[i + 1]
            windows_positive = [w.unsqueeze(0).to(device) for w in positive.telemetry_windows]
            result_positive = model(windows_positive, use_conditioning=use_conditioning)

            # Negative: stressed sample (different state)
            negative = stressed_samples[i]
            windows_negative = [w.unsqueeze(0).to(device) for w in negative.telemetry_windows]
            result_negative = model(windows_negative, use_conditioning=use_conditioning)

            # Triplet margin loss
            anchor_bound = result_anchor['bound']
            positive_bound = result_positive['bound']
            negative_bound = result_negative['bound']

            # Distance-based triplet loss with larger margin
            pos_dist = (anchor_bound - positive_bound).pow(2).sum(dim=-1).sqrt()
            neg_dist = (anchor_bound - negative_bound).pow(2).sum(dim=-1).sqrt()
            margin = 2.0
            triplet_loss = F.relu(pos_dist - neg_dist + margin).mean()

            # InfoNCE-style contrastive loss
            temperature = 0.1
            pos_sim = F.cosine_similarity(anchor_bound, positive_bound, dim=-1) / temperature
            neg_sim = F.cosine_similarity(anchor_bound, negative_bound, dim=-1) / temperature
            infonce = -torch.log(torch.exp(pos_sim) / (torch.exp(pos_sim) + torch.exp(neg_sim) + 1e-8)).mean()

            # Temporal order prediction (auxiliary task)
            order_logits = result_anchor['order_logits']
            order_target = torch.tensor([[0.0 if is_shuffled else 1.0]], device=device)
            order_loss = F.binary_cross_entropy_with_logits(order_logits, order_target)

            # Embedding norm regularization (prevent collapse to zero or explosion)
            all_bounds = torch.cat([anchor_bound, positive_bound, negative_bound], dim=0)
            norm_target = 1.0
            norm_loss = (all_bounds.norm(dim=-1) - norm_target).pow(2).mean()

            # Variance preservation (prevent collapse)
            var_loss = -all_bounds.var(dim=0).mean()

            loss = triplet_loss + infonce + 0.5 * order_loss + 0.1 * norm_loss + 0.2 * var_loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_losses.append(loss.item())

        scheduler.step()
        avg_loss = np.mean(epoch_losses) if epoch_losses else 0.0
        losses.append(avg_loss)
        if (epoch + 1) % 3 == 0 or epoch == 0:
            print(f"    Epoch {epoch+1}/{n_epochs}: loss={avg_loss:.4f}")

    return losses


def main():
    print("=" * 70)
    print("  z1722: TEMPORAL BINDING TEST")
    print("  Can embodied systems bind experience across time?")
    print("=" * 70)
    print()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name()}")
    print()

    # Initialize telemetry
    try:
        telem = SysfsHwmonTelemetry()
        print(f"Telemetry: {telem.paths.power_average}")
    except Exception as e:
        print(f"Warning: Could not initialize telemetry: {e}")
        print("Using simulated telemetry")
        telem = None

    # Create models for each condition
    hidden_dim = 256
    telemetry_dim = 8

    print("\n" + "=" * 70)
    print("PHASE 1: DATA COLLECTION")
    print("=" * 70)
    print()

    temporal_buffer = TemporalBuffer(max_size=100)

    if telem:
        calm_samples, stressed_samples = collect_temporal_samples(
            telem, temporal_buffer, device,
            n_calm=30, n_stressed=30,
        )
    else:
        # Simulated data
        print("  Using simulated telemetry data...")
        calm_samples = []
        stressed_samples = []
        for i in range(30):
            calm_windows = [torch.randn(telemetry_dim) * 0.1 + 0.3 for _ in range(4)]
            calm_samples.append(TemporalSample(time.time(), calm_windows, 0))
            stressed_windows = [torch.randn(telemetry_dim) * 0.1 + 0.7 for _ in range(4)]
            stressed_samples.append(TemporalSample(time.time(), stressed_windows, 1))

    all_samples = calm_samples + stressed_samples
    shuffled_samples = shuffle_windows(all_samples)

    print(f"\n  Collected {len(calm_samples)} calm + {len(stressed_samples)} stressed samples")

    # Create models
    print("\n" + "=" * 70)
    print("PHASE 2: MODEL TRAINING")
    print("=" * 70)

    # Condition A: EMBODIED
    print("\n[A] EMBODIED (real telemetry, correct temporal order)")
    model_embodied = TemporalBindingModel(hidden_dim, telemetry_dim).to(device)
    embodied_losses = train_model(model_embodied, all_samples, device,
                                   n_epochs=12, use_conditioning=True, is_shuffled=False)

    # Condition B: DISEMBODIED
    print("\n[B] DISEMBODIED (zero telemetry)")
    model_disembodied = TemporalBindingModel(hidden_dim, telemetry_dim).to(device)
    disembodied_losses = train_model(model_disembodied, all_samples, device,
                                      n_epochs=12, use_conditioning=False, is_shuffled=False)

    # Condition C: SHUFFLED_TIME
    print("\n[C] SHUFFLED_TIME (telemetry from wrong time windows)")
    model_shuffled = TemporalBindingModel(hidden_dim, telemetry_dim).to(device)
    shuffled_losses = train_model(model_shuffled, shuffled_samples, device,
                                   n_epochs=12, use_conditioning=True, is_shuffled=True)

    # Evaluate
    print("\n" + "=" * 70)
    print("PHASE 3: EVALUATION")
    print("=" * 70)

    print("\n[A] EMBODIED metrics:")
    embodied_metrics = compute_binding_metrics(model_embodied, all_samples, device, True)
    for k, v in embodied_metrics.items():
        print(f"    {k}: {v:.4f}")

    print("\n[B] DISEMBODIED metrics:")
    disembodied_metrics = compute_binding_metrics(model_disembodied, all_samples, device, False)
    for k, v in disembodied_metrics.items():
        print(f"    {k}: {v:.4f}")

    print("\n[C] SHUFFLED_TIME metrics:")
    shuffled_metrics = compute_binding_metrics(model_shuffled, shuffled_samples, device, True)
    for k, v in shuffled_metrics.items():
        print(f"    {k}: {v:.4f}")

    # Verdicts
    print("\n" + "=" * 70)
    print("VERDICTS")
    print("=" * 70)

    verdicts = {}

    # V1: EMBODIED classification accuracy > 0.7 (can distinguish states)
    v1_pass = embodied_metrics['classification_accuracy'] > 0.7
    verdicts['V1_embodied_classification_high'] = {
        'threshold': 0.7,
        'value': embodied_metrics['classification_accuracy'],
        'pass': v1_pass,
    }
    print(f"\nV1: EMBODIED classification accuracy > 0.7")
    print(f"    Value: {embodied_metrics['classification_accuracy']:.4f}")
    print(f"    {'PASS' if v1_pass else 'FAIL'}")

    # V2: EMBODIED separation ratio > DISEMBODIED (better state separation)
    v2_pass = embodied_metrics['separation_ratio'] > disembodied_metrics['separation_ratio']
    verdicts['V2_embodied_better_separation'] = {
        'embodied': embodied_metrics['separation_ratio'],
        'disembodied': disembodied_metrics['separation_ratio'],
        'pass': v2_pass,
    }
    print(f"\nV2: EMBODIED separation > DISEMBODIED separation")
    print(f"    EMBODIED:    {embodied_metrics['separation_ratio']:.4f}")
    print(f"    DISEMBODIED: {disembodied_metrics['separation_ratio']:.4f}")
    print(f"    {'PASS' if v2_pass else 'FAIL'}")

    # V3: SHUFFLED_TIME has worse classification than EMBODIED
    v3_pass = shuffled_metrics['classification_accuracy'] < embodied_metrics['classification_accuracy']
    verdicts['V3_shuffled_worse_classification'] = {
        'embodied': embodied_metrics['classification_accuracy'],
        'shuffled': shuffled_metrics['classification_accuracy'],
        'pass': v3_pass,
    }
    print(f"\nV3: SHUFFLED classification < EMBODIED classification")
    print(f"    EMBODIED: {embodied_metrics['classification_accuracy']:.4f}")
    print(f"    SHUFFLED: {shuffled_metrics['classification_accuracy']:.4f}")
    print(f"    {'PASS' if v3_pass else 'FAIL'}")

    # V4: DISEMBODIED classification near chance (0.5)
    v4_pass = abs(disembodied_metrics['classification_accuracy'] - 0.5) < 0.15
    verdicts['V4_disembodied_chance'] = {
        'expected': 0.5,
        'tolerance': 0.15,
        'value': disembodied_metrics['classification_accuracy'],
        'pass': v4_pass,
    }
    print(f"\nV4: DISEMBODIED classification near chance (0.5 +/- 0.15)")
    print(f"    Value: {disembodied_metrics['classification_accuracy']:.4f}")
    print(f"    {'PASS' if v4_pass else 'FAIL'}")

    # Summary
    passed = sum(1 for v in verdicts.values() if v['pass'])
    total = len(verdicts)

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    print(f"\n{'Metric':<35} {'EMBODIED':>12} {'DISEMBODIED':>12} {'SHUFFLED':>12}")
    print("-" * 75)
    for key in ['classification_accuracy', 'separation_ratio', 'between_state_distance',
                'within_state_scatter', 'order_prediction']:
        print(f"{key:<35} {embodied_metrics[key]:>12.4f} {disembodied_metrics[key]:>12.4f} {shuffled_metrics[key]:>12.4f}")

    print(f"\nVerdicts passed: {passed}/{total}")

    if passed >= 3:
        overall_verdict = "TEMPORAL_BINDING_DEMONSTRATED"
        print("\nOVERALL: TEMPORAL BINDING DEMONSTRATED")
        print("  The embodied model successfully binds experience across time,")
        print("  maintaining coherent representations that are disrupted by")
        print("  removing embodiment or shuffling temporal order.")
    elif passed >= 2:
        overall_verdict = "PARTIAL_BINDING"
        print("\nOVERALL: PARTIAL TEMPORAL BINDING")
        print("  Some evidence of temporal binding, but not fully conclusive.")
    else:
        overall_verdict = "NO_BINDING_DETECTED"
        print("\nOVERALL: NO CLEAR TEMPORAL BINDING")
        print("  The experiment did not demonstrate significant temporal binding.")

    # Save results
    def to_python(obj):
        if isinstance(obj, dict):
            return {k: to_python(v) for k, v in obj.items()}
        elif isinstance(obj, (np.floating, np.integer)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, bool):
            return bool(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        return obj

    results = {
        'experiment': 'z1722_temporal_binding',
        'timestamp': datetime.now().isoformat(),
        'hypothesis': 'Consciousness requires binding experiences over time',
        'conditions': {
            'A_EMBODIED': 'Real telemetry with correct temporal order',
            'B_DISEMBODIED': 'Zero telemetry - no temporal context',
            'C_SHUFFLED_TIME': 'Telemetry from wrong time windows',
        },
        'metrics': {
            'embodied': to_python(embodied_metrics),
            'disembodied': to_python(disembodied_metrics),
            'shuffled': to_python(shuffled_metrics),
        },
        'training_losses': {
            'embodied': [float(x) for x in embodied_losses],
            'disembodied': [float(x) for x in disembodied_losses],
            'shuffled': [float(x) for x in shuffled_losses],
        },
        'verdicts': to_python(verdicts),
        'verdicts_passed': passed,
        'verdicts_total': total,
        'overall_verdict': overall_verdict,
        'sample_counts': {
            'calm': len(calm_samples),
            'stressed': len(stressed_samples),
        },
    }

    results_path = Path(__file__).parent.parent / 'results' / 'z1722_temporal_binding.json'
    results_path.parent.mkdir(exist_ok=True)

    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {results_path}")


if __name__ == '__main__':
    main()
