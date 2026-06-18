#!/usr/bin/env python3
"""
Z905: GRU-Based Learned Body State (z_feel) Experiment
======================================================

Hypothesis: A learned GRU body-state representation outperforms the hand-coded
BodyStateTracker (26-dim EMA/derivatives) at driving beneficial computation
changes via FiLM conditioning.

Training (2 phases):
  Phase 1 - Self-supervised GRU pre-training: predict next telemetry from
            history. ~200 steps.
  Phase 2 - Joint training: GRU z_feel feeds FiLM into MetabolicTransformer.
            Backprop through both with task + energy loss. 10 epochs.

Conditions:
  A: Hand-coded BodyLatent (12-dim EMA/derivative vector -> learned linear -> FiLM)
  B: Random z_feel (GRU fed random input instead of real telemetry)
  C: Frozen GRU (pre-trained phase 1 only, not jointly optimized in phase 2)
  D: Full GRU (pre-trained + jointly optimized) -- the experimental condition

Metrics:
  - Telemetry prediction accuracy (MSE)
  - J/token with learned vs hand-coded
  - Perplexity at matched energy budgets
  - z_feel latent visualization (PCA saved to results)

Usage:
    HSA_OVERRIDE_GFX_VERSION=11.0.0 python scripts/z905_gru_z_feel.py
    HSA_OVERRIDE_GFX_VERSION=11.0.0 python scripts/z905_gru_z_feel.py --phase1-steps 200 --phase2-epochs 10

Author: FEEL Research Team
Date: 2026-01-28
"""

import os
os.environ['HSA_OVERRIDE_GFX_VERSION'] = '11.0.0'

import sys
import json
import time
import copy
import math
import logging
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict, field
import urllib.request

import numpy as np

# Add project root to path
script_dir = Path(__file__).parent.absolute()
project_root = script_dir.parent
sys.path.insert(0, str(project_root))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW

from src.embodied.gru_body_state import GRUBodyState
from src.metabolic.film_transformer import (
    MetabolicTransformer, MetabolicConfig, FiLMGenerator
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

class CharDataset(Dataset):
    """Character-level dataset for language modeling."""

    def __init__(self, text: str, seq_len: int):
        self.data = torch.tensor([ord(c) % 256 for c in text], dtype=torch.long)
        self.seq_len = seq_len

    def __len__(self):
        return max(1, len(self.data) - self.seq_len - 1)

    def __getitem__(self, idx):
        x = self.data[idx:idx + self.seq_len]
        y = self.data[idx + 1:idx + self.seq_len + 1]
        return x, y


def download_tiny_shakespeare(data_dir: Path) -> str:
    """Download TinyShakespeare dataset, return text content."""
    data_dir.mkdir(parents=True, exist_ok=True)
    data_file = data_dir / "tinyshakespeare.txt"

    if data_file.exists() and data_file.stat().st_size > 10000:
        logger.info(f"Loading cached TinyShakespeare from {data_file}")
        return data_file.read_text(encoding='utf-8')

    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    logger.info(f"Downloading TinyShakespeare from {url} ...")
    try:
        urllib.request.urlretrieve(url, str(data_file))
        text = data_file.read_text(encoding='utf-8')
        logger.info(f"Downloaded {len(text):,} chars")
        return text
    except Exception as e:
        logger.warning(f"Download failed: {e}. Generating synthetic corpus.")
        return _generate_synthetic_corpus()


def _generate_synthetic_corpus(size: int = 500_000) -> str:
    """Fallback synthetic corpus for when download fails."""
    passages = [
        "To be, or not to be, that is the question. ",
        "All that glitters is not gold. ",
        "The course of true love never did run smooth. ",
        "We are such stuff as dreams are made on. ",
        "Brevity is the soul of wit. ",
        "The lady doth protest too much, methinks. ",
        "There are more things in heaven and earth. ",
        "What a piece of work is a man! ",
        "This above all: to thine own self be true. ",
        "If music be the food of love, play on. ",
        "Now is the winter of our discontent. ",
        "Friends, Romans, countrymen, lend me your ears. ",
        "The fault, dear Brutus, is not in our stars, but in ourselves. ",
        "Some are born great, some achieve greatness. ",
        "Love all, trust a few, do wrong to none. ",
        "The better part of valour is discretion. ",
        "Parting is such sweet sorrow. ",
        "Though she be but little, she is fierce. ",
        "Something is rotten in the state of Denmark. ",
        "Out, damned spot! Out, I say! ",
    ]
    corpus = ""
    while len(corpus) < size:
        for p in passages:
            corpus += p
            if len(corpus) >= size:
                break
    return corpus[:size]


# ---------------------------------------------------------------------------
# Telemetry helpers
# ---------------------------------------------------------------------------

def get_telemetry_reader():
    """Try to get real GPU telemetry, else return None."""
    try:
        from src.telemetry.sysfs_hwmon import SysfsHwmonTelemetry
        telem = SysfsHwmonTelemetry(sample_rate_hz=50)
        telem.read_sample()  # test
        logger.info("Real sysfs telemetry available")
        return telem
    except Exception as e:
        logger.warning(f"No real telemetry: {e}")
        return None


def read_telemetry_vector(telem_reader, device: torch.device) -> torch.Tensor:
    """
    Read 8-dim telemetry vector from hardware.
    [power_w, temp_c, util_pct, freq_sclk_mhz, freq_mclk_mhz,
     gpu_busy_pct, vram_used_gb, energy_j_frac]
    """
    if telem_reader is None:
        return _synthetic_telemetry(device)
    try:
        s = telem_reader.read_sample()
        vec = torch.tensor([
            s.power_w,
            s.temp_edge_c,
            float(s.gpu_busy_pct),
            float(s.freq_sclk_mhz),
            float(s.freq_mclk_mhz),
            float(s.gpu_busy_pct),
            s.vram_used_gb,
            s.power_w * 0.01,  # proxy energy_j fraction
        ], dtype=torch.float32, device=device)
        return vec
    except Exception:
        return _synthetic_telemetry(device)


def _synthetic_telemetry(device: torch.device) -> torch.Tensor:
    """Generate noisy synthetic telemetry for testing."""
    base = torch.tensor([35.0, 55.0, 60.0, 1500.0, 900.0, 60.0, 4.0, 0.35],
                        dtype=torch.float32, device=device)
    noise = torch.randn(8, device=device) * torch.tensor(
        [5.0, 3.0, 10.0, 200.0, 100.0, 10.0, 0.5, 0.05],
        dtype=torch.float32, device=device
    )
    return (base + noise).clamp(min=0)


def get_energy_meter(telem_reader):
    """Get an EnergyMeter if real telemetry is available."""
    if telem_reader is None:
        return None
    try:
        from src.telemetry.sysfs_hwmon import EnergyMeter
        return EnergyMeter(telem_reader)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Hand-coded body state (Condition A)
# ---------------------------------------------------------------------------

class HandCodedBodyState(nn.Module):
    """
    Simplified hand-coded body state: EMA + derivatives of power, temp, util.
    Produces a 12-dim vector -> learned linear -> FiLM gamma/beta.
    """

    def __init__(self, body_dim: int = 12, transformer_hidden: int = 256):
        super().__init__()
        self.body_dim = body_dim
        self.film_gamma = nn.Linear(body_dim, transformer_hidden)
        self.film_beta = nn.Linear(body_dim, transformer_hidden)

        # Initialize FiLM to near-identity
        nn.init.zeros_(self.film_gamma.weight)
        nn.init.zeros_(self.film_gamma.bias)
        nn.init.zeros_(self.film_beta.weight)
        nn.init.zeros_(self.film_beta.bias)

        # EMA state (not nn parameters, just buffers)
        self._ema = None
        self._prev = None
        self._alpha = 0.2

    def reset(self):
        self._ema = None
        self._prev = None

    def compute_body_vector(self, power_w: float, temp_c: float, util_pct: float) -> torch.Tensor:
        """
        Compute 12-dim hand-coded body state vector.
        [raw(3), ema(3), derivative(3), normalized(3)]
        """
        raw = [power_w, temp_c, util_pct]

        if self._ema is None:
            self._ema = list(raw)
            self._prev = list(raw)

        # EMA update
        ema = [self._alpha * r + (1 - self._alpha) * e for r, e in zip(raw, self._ema)]
        self._ema = ema

        # Derivatives
        deriv = [r - p for r, p in zip(raw, self._prev)]
        self._prev = list(raw)

        # Normalized
        norm = [power_w / 100.0, temp_c / 100.0, util_pct / 100.0]

        return torch.tensor(raw + ema + deriv + norm, dtype=torch.float32)

    def get_film_params(self, body_vec: torch.Tensor):
        """Get FiLM parameters from hand-coded body vector."""
        if body_vec.dim() == 1:
            body_vec = body_vec.unsqueeze(0)
        return self.film_gamma(body_vec), self.film_beta(body_vec)


# ---------------------------------------------------------------------------
# FiLM-conditioned forward pass wrapper
# ---------------------------------------------------------------------------

def forward_with_film(
    model: MetabolicTransformer,
    input_ids: torch.Tensor,
    gamma: torch.Tensor,
    beta: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    """
    Run MetabolicTransformer forward pass, injecting external FiLM gamma/beta
    into the first layer's LayerNorm conditioning.

    The MetabolicTransformer normally generates its own FiLM params from a
    telemetry vector. Here we bypass that by directly setting telemetry to
    produce our desired gamma/beta. We use a simpler approach: pass a
    synthetic telemetry that after the model's FiLM generators produces
    approximately the desired gamma/beta, OR we monkey-patch the first block.

    For scientific rigor, we use the model's built-in telemetry interface
    when we have a proper telemetry vector, and only use external FiLM
    for the ablation conditions.
    """
    # We use a simple approach: condition_every_layer with the model's own
    # telemetry_dim input. For conditions A/B/C/D, we build a custom
    # telemetry-dim vector that the model's FiLMGenerators will process.
    # However, for proper ablation we need to override the FiLM params directly.
    #
    # The cleanest approach: temporarily replace the film_generators output.
    # We'll use forward hooks.
    batch = input_ids.size(0)
    device = input_ids.device

    # Create a dummy telemetry to enable conditioning path
    dummy_telem = torch.zeros(batch, model.config.telemetry_dim, device=device)

    # Store original film generators' forward methods and override first layer
    original_ln1_forward = model.film_generators[0]['ln1'].forward
    original_ln2_forward = model.film_generators[0]['ln2'].forward

    def patched_ln1_forward(telemetry):
        return gamma, beta

    def patched_ln2_forward(telemetry):
        return gamma, beta

    try:
        model.film_generators[0]['ln1'].forward = patched_ln1_forward
        model.film_generators[0]['ln2'].forward = patched_ln2_forward
        output = model(input_ids, telemetry=dummy_telem)
    finally:
        model.film_generators[0]['ln1'].forward = original_ln1_forward
        model.film_generators[0]['ln2'].forward = original_ln2_forward

    return output


# ---------------------------------------------------------------------------
# Phase 1: Self-supervised GRU pre-training
# ---------------------------------------------------------------------------

def run_phase1(
    gru: GRUBodyState,
    telem_reader,
    device: torch.device,
    num_steps: int = 200,
    lr: float = 1e-3,
) -> Dict:
    """
    Phase 1: Train GRU to predict next telemetry from history.

    Collects telemetry sequences and trains the GRU with MSE loss
    on next-step prediction.

    Returns dict with loss history and final metrics.
    """
    logger.info(f"=== Phase 1: Self-supervised GRU pre-training ({num_steps} steps) ===")
    gru.train()
    optimizer = AdamW(gru.parameters(), lr=lr)

    losses = []
    best_loss = float('inf')

    for step in range(num_steps):
        gru.reset()
        optimizer.zero_grad()

        # Collect a short telemetry sequence (simulate by reading repeatedly)
        seq_len = 16
        telem_seq = []
        for _ in range(seq_len):
            t_vec = read_telemetry_vector(telem_reader, device)
            telem_seq.append(t_vec)
            # Small sleep to get varying readings
            if telem_reader is not None:
                time.sleep(0.005)

        # Stack: [seq_len, 8]
        telem_seq = torch.stack(telem_seq, dim=0)

        # Normalize telemetry for stable training
        telem_mean = telem_seq.mean(dim=0, keepdim=True)
        telem_std = telem_seq.std(dim=0, keepdim=True).clamp(min=1e-6)
        telem_norm = (telem_seq - telem_mean) / telem_std

        # Process sequence, predict next step
        total_loss = torch.tensor(0.0, device=device)
        predictions = 0

        for t in range(seq_len - 1):
            input_t = telem_norm[t].unsqueeze(0)  # [1, 8]
            z_feel = gru.step(input_t)
            pred_next = gru.predict_next_telemetry(z_feel)  # [1, 8]
            target_next = telem_norm[t + 1].unsqueeze(0)  # [1, 8]
            total_loss = total_loss + F.mse_loss(pred_next, target_next)
            predictions += 1

        if predictions > 0:
            avg_loss = total_loss / predictions
            avg_loss.backward()
            torch.nn.utils.clip_grad_norm_(gru.parameters(), 1.0)
            optimizer.step()

            loss_val = avg_loss.item()
            losses.append(loss_val)
            if loss_val < best_loss:
                best_loss = loss_val

            if (step + 1) % 20 == 0 or step == 0:
                logger.info(
                    f"  Phase1 step {step+1:4d}/{num_steps} | "
                    f"loss={loss_val:.6f} | best={best_loss:.6f}"
                )

    logger.info(f"  Phase 1 complete. Final loss={losses[-1]:.6f}, Best={best_loss:.6f}")

    return {
        'losses': losses,
        'final_loss': losses[-1] if losses else 0.0,
        'best_loss': best_loss,
        'num_steps': num_steps,
    }


# ---------------------------------------------------------------------------
# Phase 2: Joint training
# ---------------------------------------------------------------------------

def run_phase2_condition(
    condition_name: str,
    model: MetabolicTransformer,
    gru: Optional[GRUBodyState],
    hand_coded: Optional[HandCodedBodyState],
    train_loader: DataLoader,
    val_loader: DataLoader,
    telem_reader,
    device: torch.device,
    num_epochs: int = 10,
    lr: float = 3e-4,
    use_random_telemetry: bool = False,
    freeze_gru: bool = False,
) -> Dict:
    """
    Phase 2: Joint training for one condition.

    Loss = L_task + 0.1 * L_energy_prediction + 0.05 * L_telemetry_prediction

    Args:
        condition_name: A/B/C/D label
        model: MetabolicTransformer instance
        gru: GRUBodyState (None for condition A)
        hand_coded: HandCodedBodyState (only for condition A)
        train_loader: training data
        val_loader: validation data
        telem_reader: real or None telemetry
        device: torch device
        num_epochs: training epochs
        lr: learning rate
        use_random_telemetry: condition B
        freeze_gru: condition C

    Returns:
        dict with metrics
    """
    logger.info(f"\n=== Phase 2 Condition {condition_name} ({num_epochs} epochs) ===")

    # Build optimizer param groups
    params = list(model.parameters())
    if hand_coded is not None:
        params += list(hand_coded.parameters())
    if gru is not None and not freeze_gru:
        params += list(gru.parameters())

    optimizer = AdamW(params, lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    # Energy meter
    energy_meter = get_energy_meter(telem_reader)

    train_losses = []
    val_losses = []
    val_perplexities = []
    energy_per_epoch = []
    telem_pred_losses = []

    tokens_total = 0
    total_energy_j = 0.0

    for epoch in range(num_epochs):
        model.train()
        if gru is not None:
            if freeze_gru:
                gru.eval()
            else:
                gru.train()
        if hand_coded is not None:
            hand_coded.train()

        epoch_loss = 0.0
        epoch_telem_loss = 0.0
        epoch_tokens = 0
        epoch_energy_j = 0.0
        epoch_start = time.time()

        for batch_idx, (x_batch, y_batch) in enumerate(train_loader):
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            batch_size = x_batch.size(0)

            optimizer.zero_grad()

            # Read telemetry
            if use_random_telemetry:
                raw_telem = torch.rand(8, device=device) * torch.tensor(
                    [100.0, 100.0, 100.0, 2500.0, 1500.0, 100.0, 16.0, 1.0],
                    device=device
                )
            else:
                raw_telem = read_telemetry_vector(telem_reader, device)

            # Compute FiLM params based on condition
            if hand_coded is not None:
                # Condition A: hand-coded body state
                hand_coded.reset()
                body_vec = hand_coded.compute_body_vector(
                    raw_telem[0].item(), raw_telem[1].item(), raw_telem[2].item()
                ).to(device)
                gamma, beta = hand_coded.get_film_params(body_vec)
                gamma = gamma.expand(batch_size, -1)
                beta = beta.expand(batch_size, -1)

                # Use model's own telemetry interface with a telemetry vector
                # derived from hand-coded body state
                telem_for_model = torch.zeros(batch_size, model.config.telemetry_dim, device=device)
                telem_for_model[:, :3] = body_vec[:3].unsqueeze(0).expand(batch_size, -1)
                telem_for_model[:, 3:6] = body_vec[3:6].unsqueeze(0).expand(batch_size, -1)
                telem_for_model[:, 6:9] = body_vec[6:9].unsqueeze(0).expand(batch_size, -1)
                telem_for_model[:, 9:12] = body_vec[9:12].unsqueeze(0).expand(batch_size, -1)

                output = model(x_batch, telemetry=telem_for_model)
                telem_pred_loss = torch.tensor(0.0, device=device)

            elif gru is not None:
                # Conditions B, C, D: GRU-based
                gru.reset()
                telem_input = raw_telem.unsqueeze(0)  # [1, 8]
                z_feel = gru.step(telem_input)  # [1, 32]

                # Predict next telemetry (self-supervised auxiliary)
                pred_next = gru.predict_next_telemetry(z_feel)
                # Read actual next telemetry for target
                if use_random_telemetry:
                    actual_next = torch.rand(1, 8, device=device)
                else:
                    actual_next = read_telemetry_vector(telem_reader, device).unsqueeze(0)
                telem_pred_loss = F.mse_loss(pred_next, actual_next)

                # Get FiLM params from z_feel
                gamma, beta = gru.get_film_params(z_feel)
                gamma = gamma.expand(batch_size, -1)  # [batch, 256]
                beta = beta.expand(batch_size, -1)

                # Build telemetry-dim vector for the model from z_feel projection
                # Project 32-dim z_feel to 12-dim telemetry_dim for model interface
                z_feel_expanded = z_feel.expand(batch_size, -1)
                telem_for_model = z_feel_expanded[:, :model.config.telemetry_dim]

                output = model(x_batch, telemetry=telem_for_model)
            else:
                # Should not happen
                output = model(x_batch)
                telem_pred_loss = torch.tensor(0.0, device=device)

            # Task loss (cross-entropy)
            logits = output['logits']
            task_loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                y_batch.view(-1),
            )

            # Energy prediction loss (auxiliary: predict power from hidden)
            # Use last hidden to predict current power reading
            if 'hidden' not in output:
                output_h = model(x_batch, telemetry=telem_for_model if 'telem_for_model' in dir() else None, return_hidden=True)
                hidden = output_h['hidden'][:, -1, :]  # [batch, hidden_dim]
            else:
                hidden = output['hidden'][:, -1, :]

            # Simple power prediction from hidden state
            power_target = raw_telem[0].unsqueeze(0).expand(batch_size) / 100.0
            power_pred = hidden.mean(dim=-1)  # crude proxy
            energy_pred_loss = F.mse_loss(power_pred, power_target)

            # Combined loss
            loss = task_loss + 0.1 * energy_pred_loss + 0.05 * telem_pred_loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()

            batch_tokens = x_batch.numel()
            epoch_loss += task_loss.item() * batch_tokens
            epoch_telem_loss += telem_pred_loss.item()
            epoch_tokens += batch_tokens

        scheduler.step()

        # Epoch stats
        avg_train_loss = epoch_loss / max(epoch_tokens, 1)
        avg_telem_loss = epoch_telem_loss / max(len(train_loader), 1)
        epoch_time = time.time() - epoch_start
        tokens_total += epoch_tokens

        train_losses.append(avg_train_loss)
        telem_pred_losses.append(avg_telem_loss)

        # Validation
        val_loss, val_ppl, val_tokens = evaluate(
            model, gru, hand_coded, val_loader, telem_reader,
            device, use_random_telemetry
        )
        val_losses.append(val_loss)
        val_perplexities.append(val_ppl)

        # Energy measurement
        epoch_energy = _measure_inference_energy(
            model, gru, hand_coded, val_loader, telem_reader,
            device, use_random_telemetry
        )
        energy_per_epoch.append(epoch_energy)
        total_energy_j += epoch_energy['total_j']

        logger.info(
            f"  [{condition_name}] Epoch {epoch+1:2d}/{num_epochs} | "
            f"train_loss={avg_train_loss:.4f} | val_loss={val_loss:.4f} | "
            f"ppl={val_ppl:.2f} | telem_pred={avg_telem_loss:.6f} | "
            f"J/tok={epoch_energy['j_per_token']:.4f} | "
            f"time={epoch_time:.1f}s"
        )

    # Final J/token
    final_j_per_token = energy_per_epoch[-1]['j_per_token'] if energy_per_epoch else 0.0

    return {
        'condition': condition_name,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'val_perplexities': val_perplexities,
        'energy_per_epoch': energy_per_epoch,
        'telem_pred_losses': telem_pred_losses,
        'final_train_loss': train_losses[-1] if train_losses else 0.0,
        'final_val_loss': val_losses[-1] if val_losses else 0.0,
        'final_perplexity': val_perplexities[-1] if val_perplexities else 0.0,
        'final_j_per_token': final_j_per_token,
        'total_tokens': tokens_total,
        'total_energy_j': total_energy_j,
    }


def evaluate(
    model: MetabolicTransformer,
    gru: Optional[GRUBodyState],
    hand_coded: Optional[HandCodedBodyState],
    val_loader: DataLoader,
    telem_reader,
    device: torch.device,
    use_random_telemetry: bool = False,
) -> Tuple[float, float, int]:
    """Evaluate model, return (avg_loss, perplexity, total_tokens)."""
    model.eval()
    if gru is not None:
        gru.eval()
    if hand_coded is not None:
        hand_coded.eval()

    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        for x_batch, y_batch in val_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            batch_size = x_batch.size(0)

            if use_random_telemetry:
                raw_telem = torch.rand(8, device=device)
            else:
                raw_telem = read_telemetry_vector(telem_reader, device)

            if hand_coded is not None:
                hand_coded.reset()
                body_vec = hand_coded.compute_body_vector(
                    raw_telem[0].item(), raw_telem[1].item(), raw_telem[2].item()
                ).to(device)
                telem_for_model = torch.zeros(batch_size, model.config.telemetry_dim, device=device)
                telem_for_model[:, :3] = body_vec[:3].unsqueeze(0).expand(batch_size, -1)
                telem_for_model[:, 3:6] = body_vec[3:6].unsqueeze(0).expand(batch_size, -1)
                telem_for_model[:, 6:9] = body_vec[6:9].unsqueeze(0).expand(batch_size, -1)
                telem_for_model[:, 9:12] = body_vec[9:12].unsqueeze(0).expand(batch_size, -1)
                output = model(x_batch, telemetry=telem_for_model)
            elif gru is not None:
                gru.reset()
                z_feel = gru.step(raw_telem.unsqueeze(0))
                z_feel_expanded = z_feel.expand(batch_size, -1)
                telem_for_model = z_feel_expanded[:, :model.config.telemetry_dim]
                output = model(x_batch, telemetry=telem_for_model)
            else:
                output = model(x_batch)

            logits = output['logits']
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                y_batch.view(-1),
            )

            batch_tokens = x_batch.numel()
            total_loss += loss.item() * batch_tokens
            total_tokens += batch_tokens

    avg_loss = total_loss / max(total_tokens, 1)
    perplexity = min(math.exp(avg_loss), 1e6)  # cap at 1M
    return avg_loss, perplexity, total_tokens


def _measure_inference_energy(
    model: MetabolicTransformer,
    gru: Optional[GRUBodyState],
    hand_coded: Optional[HandCodedBodyState],
    val_loader: DataLoader,
    telem_reader,
    device: torch.device,
    use_random_telemetry: bool = False,
    max_batches: int = 10,
) -> Dict:
    """Measure energy during inference on validation data."""
    model.eval()
    if gru is not None:
        gru.eval()

    energy_meter = get_energy_meter(telem_reader)

    total_tokens = 0
    start_time = time.time()

    # Try real energy measurement
    if energy_meter is not None:
        try:
            with energy_meter as meter:
                with torch.no_grad():
                    for i, (x_batch, y_batch) in enumerate(val_loader):
                        if i >= max_batches:
                            break
                        x_batch = x_batch.to(device)
                        raw_telem = read_telemetry_vector(telem_reader, device)

                        if hand_coded is not None:
                            hand_coded.reset()
                            bv = hand_coded.compute_body_vector(
                                raw_telem[0].item(), raw_telem[1].item(), raw_telem[2].item()
                            ).to(device)
                            telem_for_model = torch.zeros(x_batch.size(0), model.config.telemetry_dim, device=device)
                            telem_for_model[:, :3] = bv[:3].unsqueeze(0).expand(x_batch.size(0), -1)
                            telem_for_model[:, 3:6] = bv[3:6].unsqueeze(0).expand(x_batch.size(0), -1)
                            telem_for_model[:, 6:9] = bv[6:9].unsqueeze(0).expand(x_batch.size(0), -1)
                            telem_for_model[:, 9:12] = bv[9:12].unsqueeze(0).expand(x_batch.size(0), -1)
                            _ = model(x_batch, telemetry=telem_for_model)
                        elif gru is not None:
                            gru.reset()
                            z_feel = gru.step(raw_telem.unsqueeze(0))
                            z_expanded = z_feel.expand(x_batch.size(0), -1)
                            _ = model(x_batch, telemetry=z_expanded[:, :model.config.telemetry_dim])
                        else:
                            _ = model(x_batch)

                        total_tokens += x_batch.numel()

            duration_s = time.time() - start_time
            j = meter.energy_j
            return {
                'total_j': j,
                'duration_s': duration_s,
                'total_tokens': total_tokens,
                'j_per_token': j / max(total_tokens, 1),
                'mj_per_token': (j * 1000) / max(total_tokens, 1),
                'tokens_per_second': total_tokens / max(duration_s, 1e-6),
                'avg_power_w': meter.avg_power_w,
                'source': 'real',
            }
        except Exception as e:
            logger.debug(f"Energy meter failed: {e}")

    # Fallback: estimate from time and synthetic power
    with torch.no_grad():
        for i, (x_batch, y_batch) in enumerate(val_loader):
            if i >= max_batches:
                break
            x_batch = x_batch.to(device)
            raw_telem = read_telemetry_vector(telem_reader, device)

            if hand_coded is not None:
                hand_coded.reset()
                bv = hand_coded.compute_body_vector(
                    raw_telem[0].item(), raw_telem[1].item(), raw_telem[2].item()
                ).to(device)
                telem_for_model = torch.zeros(x_batch.size(0), model.config.telemetry_dim, device=device)
                telem_for_model[:, :3] = bv[:3].unsqueeze(0).expand(x_batch.size(0), -1)
                telem_for_model[:, 3:6] = bv[3:6].unsqueeze(0).expand(x_batch.size(0), -1)
                telem_for_model[:, 6:9] = bv[6:9].unsqueeze(0).expand(x_batch.size(0), -1)
                telem_for_model[:, 9:12] = bv[9:12].unsqueeze(0).expand(x_batch.size(0), -1)
                _ = model(x_batch, telemetry=telem_for_model)
            elif gru is not None:
                gru.reset()
                z_feel = gru.step(raw_telem.unsqueeze(0))
                z_expanded = z_feel.expand(x_batch.size(0), -1)
                _ = model(x_batch, telemetry=z_expanded[:, :model.config.telemetry_dim])
            else:
                _ = model(x_batch)

            total_tokens += x_batch.numel()

    duration_s = time.time() - start_time
    est_power = raw_telem[0].item() if raw_telem is not None else 35.0
    est_j = est_power * duration_s

    return {
        'total_j': est_j,
        'duration_s': duration_s,
        'total_tokens': total_tokens,
        'j_per_token': est_j / max(total_tokens, 1),
        'mj_per_token': (est_j * 1000) / max(total_tokens, 1),
        'tokens_per_second': total_tokens / max(duration_s, 1e-6),
        'avg_power_w': est_power,
        'source': 'estimated',
    }


# ---------------------------------------------------------------------------
# z_feel latent visualization
# ---------------------------------------------------------------------------

def save_z_feel_visualization(
    gru: GRUBodyState,
    telem_reader,
    device: torch.device,
    save_path: Path,
    num_samples: int = 200,
):
    """
    Collect z_feel vectors and save PCA/t-SNE visualization data.
    Saves raw z_feel vectors + PCA projection as JSON for plotting.
    """
    logger.info("Collecting z_feel vectors for visualization ...")
    gru.eval()
    z_feels = []

    with torch.no_grad():
        gru.reset()
        for i in range(num_samples):
            t_vec = read_telemetry_vector(telem_reader, device)
            z = gru.step(t_vec.unsqueeze(0))
            z_feels.append(z.squeeze(0).cpu().numpy().tolist())
            if telem_reader is not None:
                time.sleep(0.01)

    z_arr = np.array(z_feels)  # [num_samples, 32]

    # PCA (manual, no sklearn dependency)
    z_centered = z_arr - z_arr.mean(axis=0, keepdims=True)
    try:
        cov = np.cov(z_centered.T)
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        # Sort by descending eigenvalue
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        # Project to 2D
        pca_2d = z_centered @ eigenvectors[:, :2]
        explained_var = eigenvalues[:2].sum() / max(eigenvalues.sum(), 1e-10)
    except Exception as e:
        logger.warning(f"PCA failed: {e}")
        pca_2d = z_centered[:, :2]
        explained_var = 0.0

    viz_data = {
        'z_feel_raw': z_feels,
        'pca_2d': pca_2d.tolist(),
        'pca_explained_variance_ratio': float(explained_var),
        'num_samples': num_samples,
        'z_feel_dim': gru.hidden_dim,
    }

    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, 'w') as f:
        json.dump(viz_data, f, indent=2)
    logger.info(f"Saved z_feel visualization to {save_path} (PCA explained var: {explained_var:.3f})")


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Z905: GRU z_feel experiment")
    parser.add_argument('--phase1-steps', type=int, default=200,
                        help='Number of self-supervised pre-training steps')
    parser.add_argument('--phase2-epochs', type=int, default=10,
                        help='Number of joint training epochs')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size for training')
    parser.add_argument('--seq-len', type=int, default=128,
                        help='Sequence length for character LM')
    parser.add_argument('--device', type=str, default='auto',
                        help='Device: auto, cuda, cpu')
    parser.add_argument('--lr', type=float, default=3e-4,
                        help='Learning rate for phase 2')
    parser.add_argument('--hidden-dim', type=int, default=256,
                        help='Transformer hidden dimension')
    parser.add_argument('--num-layers', type=int, default=6,
                        help='Number of transformer layers')
    args = parser.parse_args()

    logger.info("=" * 70)
    logger.info("Z905: GRU-Based Learned Body State (z_feel) Experiment")
    logger.info("=" * 70)
    logger.info(f"Args: {vars(args)}")

    # Device selection
    if args.device == 'auto':
        if torch.cuda.is_available():
            try:
                torch.zeros(1, device='cuda')
                device = torch.device('cuda')
            except Exception:
                device = torch.device('cpu')
        else:
            device = torch.device('cpu')
    else:
        device = torch.device(args.device)
    logger.info(f"Device: {device}")

    if device.type == 'cuda':
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Telemetry
    telem_reader = get_telemetry_reader()

    # Data
    data_dir = project_root / "data" / "shakespeare"
    corpus = download_tiny_shakespeare(data_dir)
    logger.info(f"Corpus: {len(corpus):,} chars")

    split = int(len(corpus) * 0.9)
    train_text = corpus[:split]
    val_text = corpus[split:]

    train_dataset = CharDataset(train_text, args.seq_len)
    val_dataset = CharDataset(val_text, args.seq_len)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              num_workers=0, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                            num_workers=0, drop_last=True)

    logger.info(f"Train: {len(train_dataset):,} samples, Val: {len(val_dataset):,} samples")

    # Model config
    config = MetabolicConfig(
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_heads=4,
        ff_dim=args.hidden_dim * 4,
        telemetry_dim=12,
        max_seq_len=args.seq_len + 16,
    )

    # -----------------------------------------------------------------------
    # Phase 1: Self-supervised GRU pre-training
    # -----------------------------------------------------------------------
    gru_base = GRUBodyState(
        telemetry_dim=8, encoder_dim=16, hidden_dim=32, predict_next=True
    ).to(device)

    phase1_results = run_phase1(
        gru_base, telem_reader, device, num_steps=args.phase1_steps
    )

    # Save phase 1 checkpoint
    ckpt_dir = project_root / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    torch.save(gru_base.state_dict(), ckpt_dir / "z905_gru_phase1.pt")
    logger.info(f"Saved phase 1 checkpoint to {ckpt_dir / 'z905_gru_phase1.pt'}")

    # -----------------------------------------------------------------------
    # Phase 2: Joint training for each condition
    # -----------------------------------------------------------------------
    all_results = {'phase1': phase1_results, 'conditions': {}}

    # Helper to create fresh model
    def make_model():
        m = MetabolicTransformer(config).to(device)
        return m

    # --- Condition A: Hand-coded body state ---
    logger.info("\n" + "=" * 60)
    logger.info("CONDITION A: Hand-coded BodyLatent (12-dim -> linear -> FiLM)")
    logger.info("=" * 60)
    model_a = make_model()
    hand_coded_a = HandCodedBodyState(body_dim=12, transformer_hidden=config.hidden_dim).to(device)
    results_a = run_phase2_condition(
        condition_name="A_HandCoded",
        model=model_a,
        gru=None,
        hand_coded=hand_coded_a,
        train_loader=train_loader,
        val_loader=val_loader,
        telem_reader=telem_reader,
        device=device,
        num_epochs=args.phase2_epochs,
        lr=args.lr,
    )
    all_results['conditions']['A_HandCoded'] = results_a
    torch.save({
        'model': model_a.state_dict(),
        'hand_coded': hand_coded_a.state_dict(),
    }, ckpt_dir / "z905_condition_A.pt")

    # --- Condition B: Random telemetry ---
    logger.info("\n" + "=" * 60)
    logger.info("CONDITION B: Random z_feel (GRU fed random input)")
    logger.info("=" * 60)
    model_b = make_model()
    gru_b = GRUBodyState(telemetry_dim=8, encoder_dim=16, hidden_dim=32, predict_next=True).to(device)
    gru_b.load_state_dict(gru_base.state_dict())  # Start from pre-trained
    results_b = run_phase2_condition(
        condition_name="B_RandomTelemetry",
        model=model_b,
        gru=gru_b,
        hand_coded=None,
        train_loader=train_loader,
        val_loader=val_loader,
        telem_reader=telem_reader,
        device=device,
        num_epochs=args.phase2_epochs,
        lr=args.lr,
        use_random_telemetry=True,
    )
    all_results['conditions']['B_RandomTelemetry'] = results_b
    torch.save({
        'model': model_b.state_dict(),
        'gru': gru_b.state_dict(),
    }, ckpt_dir / "z905_condition_B.pt")

    # --- Condition C: Frozen GRU ---
    logger.info("\n" + "=" * 60)
    logger.info("CONDITION C: Frozen GRU (pre-trained, not jointly optimized)")
    logger.info("=" * 60)
    model_c = make_model()
    gru_c = GRUBodyState(telemetry_dim=8, encoder_dim=16, hidden_dim=32, predict_next=True).to(device)
    gru_c.load_state_dict(gru_base.state_dict())  # Pre-trained weights
    results_c = run_phase2_condition(
        condition_name="C_FrozenGRU",
        model=model_c,
        gru=gru_c,
        hand_coded=None,
        train_loader=train_loader,
        val_loader=val_loader,
        telem_reader=telem_reader,
        device=device,
        num_epochs=args.phase2_epochs,
        lr=args.lr,
        freeze_gru=True,
    )
    all_results['conditions']['C_FrozenGRU'] = results_c
    torch.save({
        'model': model_c.state_dict(),
        'gru': gru_c.state_dict(),
    }, ckpt_dir / "z905_condition_C.pt")

    # --- Condition D: Full GRU (jointly trained) ---
    logger.info("\n" + "=" * 60)
    logger.info("CONDITION D: Full GRU (pre-trained + jointly optimized)")
    logger.info("=" * 60)
    model_d = make_model()
    gru_d = GRUBodyState(telemetry_dim=8, encoder_dim=16, hidden_dim=32, predict_next=True).to(device)
    gru_d.load_state_dict(gru_base.state_dict())  # Start from pre-trained
    results_d = run_phase2_condition(
        condition_name="D_FullGRU",
        model=model_d,
        gru=gru_d,
        hand_coded=None,
        train_loader=train_loader,
        val_loader=val_loader,
        telem_reader=telem_reader,
        device=device,
        num_epochs=args.phase2_epochs,
        lr=args.lr,
    )
    all_results['conditions']['D_FullGRU'] = results_d
    torch.save({
        'model': model_d.state_dict(),
        'gru': gru_d.state_dict(),
    }, ckpt_dir / "z905_condition_D.pt")

    # -----------------------------------------------------------------------
    # z_feel visualization (from condition D)
    # -----------------------------------------------------------------------
    results_dir = project_root / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    save_z_feel_visualization(
        gru_d, telem_reader, device,
        save_path=results_dir / "z905_z_feel_pca.json",
        num_samples=200,
    )

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    logger.info("\n" + "=" * 70)
    logger.info("SUMMARY: Z905 GRU z_feel Experiment")
    logger.info("=" * 70)

    header = f"{'Condition':<25s} {'Val Loss':>10s} {'Perplexity':>12s} {'J/token':>10s} {'Telem MSE':>12s}"
    logger.info(header)
    logger.info("-" * len(header))

    summary_table = []
    for cond_name in ['A_HandCoded', 'B_RandomTelemetry', 'C_FrozenGRU', 'D_FullGRU']:
        r = all_results['conditions'][cond_name]
        row = {
            'condition': cond_name,
            'final_val_loss': r['final_val_loss'],
            'final_perplexity': r['final_perplexity'],
            'final_j_per_token': r['final_j_per_token'],
            'final_telem_pred_loss': r['telem_pred_losses'][-1] if r['telem_pred_losses'] else 0.0,
        }
        summary_table.append(row)
        logger.info(
            f"{cond_name:<25s} "
            f"{r['final_val_loss']:>10.4f} "
            f"{r['final_perplexity']:>12.2f} "
            f"{r['final_j_per_token']:>10.4f} "
            f"{row['final_telem_pred_loss']:>12.6f}"
        )

    # Hypothesis evaluation
    d_ppl = all_results['conditions']['D_FullGRU']['final_perplexity']
    a_ppl = all_results['conditions']['A_HandCoded']['final_perplexity']
    b_ppl = all_results['conditions']['B_RandomTelemetry']['final_perplexity']
    c_ppl = all_results['conditions']['C_FrozenGRU']['final_perplexity']

    d_jpk = all_results['conditions']['D_FullGRU']['final_j_per_token']
    a_jpk = all_results['conditions']['A_HandCoded']['final_j_per_token']

    logger.info("\nHypothesis evaluation:")
    logger.info(f"  D (Full GRU) vs A (Hand-coded): ppl {d_ppl:.2f} vs {a_ppl:.2f}, "
                f"J/tok {d_jpk:.4f} vs {a_jpk:.4f}")
    logger.info(f"  D vs B (Random): ppl {d_ppl:.2f} vs {b_ppl:.2f} "
                f"(real telemetry matters: {'YES' if d_ppl < b_ppl else 'NO'})")
    logger.info(f"  D vs C (Frozen): ppl {d_ppl:.2f} vs {c_ppl:.2f} "
                f"(joint training helps: {'YES' if d_ppl < c_ppl else 'NO'})")

    hypothesis_supported = d_ppl < a_ppl and d_ppl < b_ppl
    logger.info(f"\n  Hypothesis supported: {'YES' if hypothesis_supported else 'NO'}")
    if hypothesis_supported:
        ppl_improvement = (a_ppl - d_ppl) / a_ppl * 100
        logger.info(f"  Perplexity improvement over hand-coded: {ppl_improvement:.1f}%")

    # Save results
    all_results['summary'] = summary_table
    all_results['hypothesis_supported'] = hypothesis_supported
    all_results['device'] = str(device)
    all_results['timestamp'] = datetime.now().isoformat()
    all_results['args'] = vars(args)

    results_path = results_dir / "z905_gru_z_feel.json"
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    logger.info(f"\nResults saved to {results_path}")
    logger.info(f"Checkpoints saved to {ckpt_dir}/z905_*.pt")

    logger.info("\nDone.")


if __name__ == "__main__":
    main()
