#!/usr/bin/env python3
"""
z121 - REAL Training (Fixed)

This fixes the critical training bugs:

1. Phase-1 TRAINS the LM trunk (not freezes it!)
   - Previous bug: set_training_phase("adapter") froze the trunk
   - Fix: Phase-1 trains LM normally + body encoder + reporter head
   - Body encoder learns, but does NOT modulate logits yet

2. Use REAL telemetry during training
   - On GPU machine: read from hwmon
   - Fallback: realistic distributions (not random)

3. KL anchoring for semantic stability
   - Phase-2/3: KL(body_logits || baseline_logits) keeps semantics stable

4. Proper 3-phase training:
   Phase 1: "Read-only body" - LM trains, body encoder trains, no FiLM
   Phase 2: "Gated injection" - FiLM on last 2 layers, gate bias closed, KL anchor
   Phase 3: "LayerDrop actuation" - Train with stochastic depth, distillation

Usage:
    python scripts/z121_real_training.py --phase 1 --epochs 5 --train-samples 5000
    python scripts/z121_real_training.py --phase 2 --epochs 3 --checkpoint results/z121_train/phase1/best.pt
    python scripts/z121_real_training.py --phase 3 --epochs 2 --checkpoint results/z121_train/phase2/best.pt
"""

import os
import sys
import time
import json
import argparse
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Dict, List
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
import numpy as np

from transformers import AutoTokenizer


# =============================================================================
# Clamped Tokenizer
# =============================================================================

class ClampedTokenizer:
    """Tokenizer that clamps IDs to model vocab size."""

    def __init__(self, base_tokenizer, max_vocab_size: int = 32000):
        self.base = base_tokenizer
        self.max_vocab_size = max_vocab_size
        self.pad_token = base_tokenizer.pad_token
        self.eos_token = base_tokenizer.eos_token
        self.eos_token_id = min(base_tokenizer.eos_token_id or 0, max_vocab_size - 1)
        self.pad_token_id = self.eos_token_id
        self.vocab_size = max_vocab_size

    def _clamp(self, ids):
        if isinstance(ids, torch.Tensor):
            return torch.clamp(ids, min=0, max=self.max_vocab_size - 1)
        elif isinstance(ids, list):
            if ids and isinstance(ids[0], list):
                return [[min(max(0, i), self.max_vocab_size - 1) for i in row] for row in ids]
            return [min(max(0, i), self.max_vocab_size - 1) for i in ids]
        return ids

    def __call__(self, text, **kwargs):
        result = self.base(text, **kwargs)
        return {k: (self._clamp(v) if k == "input_ids" else v) for k, v in result.items()}

    def decode(self, ids, **kwargs):
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()
        ids = [min(max(0, i), self.base.vocab_size - 1) for i in ids]
        return self.base.decode(ids, **kwargs)


# =============================================================================
# Real Telemetry Source (for training)
# =============================================================================

class RealTelemetrySource:
    """
    Telemetry source that reads from real hardware when available.

    Falls back to realistic distributions (not pure random!) when
    hardware is not available.
    """

    def __init__(self, card_index: int = None):
        self.hwmon_path = self._find_hwmon(card_index)
        self.has_hardware = self.hwmon_path is not None

        if self.has_hardware:
            print(f"  Real telemetry: {self.hwmon_path}")
        else:
            print("  No hardware telemetry available, using realistic simulation")

        # For realistic fallback: track a "load state" that changes gradually
        self._load_state = 0.5  # 0=idle, 1=full load
        self._last_update = time.time()

    def _find_hwmon(self, card_index: int = None) -> Optional[Path]:
        """Find AMD hwmon path."""
        if card_index is None:
            for i in range(10):
                if Path(f"/sys/class/drm/card{i}/device/gpu_metrics").exists():
                    card_index = i
                    break
            else:
                return None

        base_path = Path(f"/sys/class/drm/card{card_index}/device/hwmon")
        if base_path.exists():
            for d in base_path.iterdir():
                if (d / "power1_average").exists():
                    return d
        return None

    def read_batch(self, batch_size: int, device: str = "cuda") -> torch.Tensor:
        """
        Read telemetry for a batch.

        Returns: (batch_size, 12) tensor of normalized telemetry
        """
        if self.has_hardware:
            return self._read_hardware_batch(batch_size, device)
        else:
            return self._simulate_realistic_batch(batch_size, device)

    def _read_hardware_batch(self, batch_size: int, device: str) -> torch.Tensor:
        """Read from real hardware."""
        telemetry = torch.zeros(batch_size, 12, device=device)

        try:
            # Power (normalized by 50W)
            power_file = self.hwmon_path / "power1_average"
            power_w = int(power_file.read_text().strip()) / 1_000_000
            telemetry[:, 0] = min(1.0, power_w / 50.0)

            # Temperature (normalized by 100C)
            temp_file = self.hwmon_path / "temp1_input"
            if temp_file.exists():
                temp_c = int(temp_file.read_text().strip()) / 1000
                telemetry[:, 1] = min(1.0, temp_c / 100.0)

        except Exception:
            # Fallback if read fails
            return self._simulate_realistic_batch(batch_size, device)

        # Add small per-sample noise for batch diversity
        telemetry[:, 0] += torch.randn(batch_size, device=device) * 0.02
        telemetry[:, 1] += torch.randn(batch_size, device=device) * 0.01
        telemetry = torch.clamp(telemetry, 0, 1)

        return telemetry

    def _simulate_realistic_batch(self, batch_size: int, device: str) -> torch.Tensor:
        """
        Simulate REALISTIC telemetry (not pure random).

        Key insight: GPU load correlates with power/temp, and changes gradually.
        """
        telemetry = torch.zeros(batch_size, 12, device=device)

        # Update load state (gradual change)
        now = time.time()
        dt = now - self._last_update
        self._last_update = now

        # Random walk for load state
        self._load_state += np.random.randn() * 0.1 * dt
        self._load_state = np.clip(self._load_state, 0.1, 0.9)

        # Power correlates with load (15-40W range for typical GPU)
        base_power = 0.3 + 0.5 * self._load_state  # 15-40W normalized
        telemetry[:, 0] = base_power + torch.randn(batch_size, device=device) * 0.05

        # Temp correlates with power (lagged)
        base_temp = 0.35 + 0.35 * self._load_state  # 35-70C normalized
        telemetry[:, 1] = base_temp + torch.randn(batch_size, device=device) * 0.03

        # Util correlates with load
        telemetry[:, 2] = self._load_state + torch.randn(batch_size, device=device) * 0.1

        # Clock (relatively stable)
        telemetry[:, 3] = 0.6 + torch.randn(batch_size, device=device) * 0.05

        telemetry = torch.clamp(telemetry, 0, 1)
        return telemetry


# =============================================================================
# Dataset
# =============================================================================

class TinyStoriesDataset(Dataset):
    """TinyStories dataset."""

    def __init__(self, tokenizer, max_len: int, split: str, max_samples: int):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.samples = []

        print(f"Loading TinyStories ({split})...")
        try:
            from datasets import load_dataset
            ds = load_dataset("roneneldan/TinyStories", split=split, streaming=True)

            for i, item in enumerate(ds):
                if i >= max_samples:
                    break
                text = item["text"].strip()
                if len(text) > 50:
                    self.samples.append(text)

            print(f"  Loaded {len(self.samples)} samples")

        except Exception as e:
            print(f"Could not load TinyStories: {e}")
            for i in range(max_samples):
                self.samples.append(
                    f"Once upon a time, there was a story about things. "
                    f"It was story number {i}. The end was happy."
                )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        text = self.samples[idx]

        encoding = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_len,
            padding="max_length",
        )

        input_ids = encoding["input_ids"].squeeze(0)
        attention_mask = encoding["attention_mask"].squeeze(0)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": input_ids.clone(),
        }


# =============================================================================
# Training Configuration
# =============================================================================

@dataclass
class TrainConfig:
    phase: int = 1
    model_size: str = "30m"
    checkpoint: Optional[str] = None

    # Data
    max_seq_len: int = 256
    train_samples: int = 5000
    val_samples: int = 500

    # Training
    epochs: int = 5
    batch_size: int = 8
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    grad_clip: float = 1.0

    # Phase-specific
    reporter_weight: float = 0.1
    policy_weight: float = 0.05
    invariance_weight: float = 0.5
    distill_weight: float = 0.3
    layerdrop_prob: float = 0.2

    # Output
    output_dir: str = "results/z121_train"
    save_every: int = 500
    eval_every: int = 100


# =============================================================================
# Phase 1: Read-Only Body (LM TRAINS!)
# =============================================================================

class Phase1Trainer:
    """
    Phase 1: Body is "read-only" to language.

    KEY FIX: The LM trunk TRAINS normally!

    What trains:
    - Full LM (embeddings, transformer, lm_head) - UNFROZEN
    - Body encoder
    - Body head (reporter)
    - Policy head

    What does NOT happen yet:
    - FiLM modulation (body doesn't affect logits)
    """

    def __init__(self, model, config: TrainConfig, telemetry: RealTelemetrySource, device: str):
        self.model = model
        self.config = config
        self.telemetry = telemetry
        self.device = device

        # KEY FIX: Set to "full" mode so body encoder runs,
        # but then disable FiLM in all layers
        self.model.base_model.set_training_phase("full")
        self._disable_all_film()

        # Ensure ALL parameters are trainable
        for param in self.model.parameters():
            param.requires_grad = True

        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"  Phase 1 trainable params: {trainable / 1e6:.2f}M")

        self.optimizer = AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

        self.step = 0

    def _disable_all_film(self):
        """Disable FiLM in all layers (body doesn't affect logits yet)."""
        for layer in self.model.base_model.layers:
            if hasattr(layer, "has_film"):
                layer.has_film = False

    def train_step(self, batch: Dict) -> Dict[str, float]:
        """Training step."""
        self.model.train()
        self.optimizer.zero_grad()

        input_ids = batch["input_ids"].to(self.device)
        labels = batch["labels"].to(self.device)

        # Get REAL telemetry (not random!)
        telemetry = self.telemetry.read_batch(input_ids.shape[0], self.device)

        # Forward (FiLM disabled, so body_latent exists but doesn't modulate)
        out = self.model(input_ids, telemetry, return_policy=True)
        logits = out["logits"]

        # LM loss (THE MAIN LOSS)
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        # Use ignore_index for padding
        lm_loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=self.model.base_model.config.vocab_size - 1,
        )

        # Reporter loss (body head predicts telemetry from hidden states)
        reporter_loss = torch.tensor(0.0, device=self.device)
        if "body_pred" in out:
            reporter_loss = F.mse_loss(out["body_pred"], telemetry)

        # Policy loss (learn to predict appropriate action)
        policy_loss = torch.tensor(0.0, device=self.device)
        if "policy_logits" in out:
            # Target: high power/temp -> ECO(0), low -> PERF(2)
            power = telemetry[:, 0]
            targets = torch.ones(input_ids.shape[0], dtype=torch.long, device=self.device)  # Default BALANCED
            targets[power > 0.6] = 0  # ECO
            targets[power < 0.3] = 2  # PERF
            policy_loss = F.cross_entropy(out["policy_logits"], targets)

        # Total loss
        total_loss = (
            lm_loss +
            self.config.reporter_weight * reporter_loss +
            self.config.policy_weight * policy_loss
        )

        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
        self.optimizer.step()

        self.step += 1

        return {
            "total": total_loss.item(),
            "lm": lm_loss.item(),
            "reporter": reporter_loss.item(),
            "policy": policy_loss.item(),
        }


# =============================================================================
# Phase 2: Gated Injection with KL Anchoring
# =============================================================================

class Phase2Trainer:
    """
    Phase 2: Gated injection in last N layers.

    - FiLM active in last 2 layers only
    - KL anchoring: body-conditioned logits stay close to baseline
    """

    def __init__(
        self,
        model,
        baseline_model,  # Frozen copy for KL
        config: TrainConfig,
        telemetry: RealTelemetrySource,
        device: str,
    ):
        self.model = model
        self.baseline = baseline_model
        self.config = config
        self.telemetry = telemetry
        self.device = device

        # Enable FiLM only in last 2 layers
        self.model.base_model.set_training_phase("full")
        self._setup_film_layers()

        # Freeze baseline
        self.baseline.eval()
        for p in self.baseline.parameters():
            p.requires_grad = False

        # Lower LR for fine-tuning
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=config.learning_rate * 0.5,
            weight_decay=config.weight_decay,
        )

        self.step = 0

    def _setup_film_layers(self):
        """Enable FiLM only in last 2 layers."""
        num_layers = len(self.model.base_model.layers)
        film_layers = [num_layers - 2, num_layers - 1]

        for i, layer in enumerate(self.model.base_model.layers):
            if hasattr(layer, "has_film"):
                layer.has_film = (i in film_layers)
                if layer.has_film:
                    # Reset FiLM to near-identity (gate bias closed)
                    if hasattr(layer, "film"):
                        nn.init.zeros_(layer.film.film_proj[0].weight)
                        nn.init.zeros_(layer.film.film_proj[0].bias)

        print(f"  FiLM enabled in layers: {film_layers}")

    def train_step(self, batch: Dict) -> Dict[str, float]:
        """Training step with KL anchoring."""
        self.model.train()
        self.optimizer.zero_grad()

        input_ids = batch["input_ids"].to(self.device)
        labels = batch["labels"].to(self.device)
        telemetry = self.telemetry.read_batch(input_ids.shape[0], self.device)

        # Forward with body (FiLM active in last layers)
        out = self.model(input_ids, telemetry, return_policy=True)
        logits = out["logits"]

        # Baseline forward (no body)
        with torch.no_grad():
            baseline_out = self.baseline.base_model(input_ids)
            baseline_logits = baseline_out["logits"]

        # LM loss
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        lm_loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
        )

        # KL anchoring (semantics lock)
        T = 2.0  # Temperature
        log_probs = F.log_softmax(logits / T, dim=-1)
        baseline_probs = F.softmax(baseline_logits / T, dim=-1)
        kl_loss = F.kl_div(log_probs, baseline_probs, reduction="batchmean") * (T * T)

        # Total
        total_loss = lm_loss + self.config.invariance_weight * kl_loss

        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
        self.optimizer.step()

        self.step += 1

        return {
            "total": total_loss.item(),
            "lm": lm_loss.item(),
            "kl": kl_loss.item(),
        }


# =============================================================================
# Phase 3: LayerDrop with Distillation
# =============================================================================

class Phase3Trainer:
    """
    Phase 3: LayerDrop compute actuation.

    - Stochastic depth during training
    - Distillation: shallow matches full
    """

    def __init__(self, model, config: TrainConfig, telemetry: RealTelemetrySource, device: str):
        self.model = model
        self.config = config
        self.telemetry = telemetry
        self.device = device

        # Enable LayerDrop
        self.model.base_model.set_layerdrop(True, prob=config.layerdrop_prob)
        self.model.base_model.set_training_phase("full")

        # Even lower LR
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=config.learning_rate * 0.3,
            weight_decay=config.weight_decay,
        )

        self.step = 0

    def train_step(self, batch: Dict) -> Dict[str, float]:
        """Training step with LayerDrop + distillation."""
        self.model.train()
        self.optimizer.zero_grad()

        input_ids = batch["input_ids"].to(self.device)
        labels = batch["labels"].to(self.device)
        telemetry = self.telemetry.read_batch(input_ids.shape[0], self.device)

        # Forward WITH LayerDrop
        self.model.base_model.set_layerdrop(True, prob=self.config.layerdrop_prob)
        out_drop = self.model(input_ids, telemetry, return_policy=True)
        logits_drop = out_drop["logits"]

        # Forward WITHOUT LayerDrop (teacher)
        with torch.no_grad():
            self.model.base_model.set_layerdrop(False)
            out_full = self.model(input_ids, telemetry, return_policy=False)
            logits_full = out_full["logits"]
            self.model.base_model.set_layerdrop(True, prob=self.config.layerdrop_prob)

        # LM loss
        shift_logits = logits_drop[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        lm_loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
        )

        # Distillation (dropped matches full)
        T = 2.0
        log_probs = F.log_softmax(logits_drop / T, dim=-1)
        full_probs = F.softmax(logits_full / T, dim=-1)
        distill_loss = F.kl_div(log_probs, full_probs, reduction="batchmean") * (T * T)

        # Total
        total_loss = lm_loss + self.config.distill_weight * distill_loss

        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
        self.optimizer.step()

        self.step += 1

        return {
            "total": total_loss.item(),
            "lm": lm_loss.item(),
            "distill": distill_loss.item(),
        }


# =============================================================================
# Evaluation
# =============================================================================

def evaluate(model, dataloader, telemetry, device) -> float:
    """Evaluate model."""
    model.eval()
    total_loss = 0
    total_samples = 0

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            telem = telemetry.read_batch(input_ids.shape[0], device)

            out = model(input_ids, telem, return_policy=False)
            logits = out["logits"]

            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
            )

            total_loss += loss.item() * input_ids.shape[0]
            total_samples += input_ids.shape[0]

    return total_loss / total_samples if total_samples > 0 else float("inf")


def compute_perplexity(model, dataloader, device) -> float:
    """Compute perplexity."""
    model.eval()
    total_loss = 0
    total_tokens = 0

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)

            if hasattr(model, "base_model"):
                out = model.base_model(input_ids)
            else:
                out = model(input_ids)

            logits = out["logits"]
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = input_ids[..., 1:].contiguous()

            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                reduction="sum",
            )

            total_loss += loss.item()
            total_tokens += shift_labels.numel()

    return np.exp(total_loss / total_tokens) if total_tokens > 0 else float("inf")


# =============================================================================
# Checkpointing
# =============================================================================

def save_checkpoint(model, optimizer, step, epoch, metrics, path):
    """Save checkpoint."""
    torch.save({
        "step": step,
        "epoch": epoch,
        "metrics": metrics,
        "model_state_dict": model.base_model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }, path)
    print(f"  Saved: {path}")


# =============================================================================
# Main Training Loop
# =============================================================================

def train_phase(config: TrainConfig, device: str):
    """Run training for specified phase."""

    print("=" * 60)
    print(f"FEEL Real Training - Phase {config.phase}")
    print("=" * 60)

    os.environ["HSA_OVERRIDE_GFX_VERSION"] = "11.0.0"

    output_dir = Path(config.output_dir) / f"phase{config.phase}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Tokenizer
    print("\n1. Tokenizer...")
    base_tok = AutoTokenizer.from_pretrained("gpt2")
    base_tok.pad_token = base_tok.eos_token
    tokenizer = ClampedTokenizer(base_tok, max_vocab_size=32000)

    # Dataset
    print("\n2. Dataset...")
    train_ds = TinyStoriesDataset(tokenizer, config.max_seq_len, "train", config.train_samples)
    val_ds = TinyStoriesDataset(tokenizer, config.max_seq_len, "validation", config.val_samples)

    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=config.batch_size, num_workers=2)

    # Telemetry source (REAL when available)
    print("\n3. Telemetry...")
    telemetry = RealTelemetrySource()

    # Model
    print(f"\n4. Model ({config.model_size})...")
    from feel_slm.embodied_slm import create_embodied_slm_30m, create_embodied_slm_125m
    from feel_slm.feel_runtime import FEELConfig, FEELModel

    if config.model_size == "30m":
        base_model = create_embodied_slm_30m()
    else:
        base_model = create_embodied_slm_125m()

    if config.checkpoint:
        print(f"   Loading checkpoint: {config.checkpoint}")
        ckpt = torch.load(config.checkpoint, map_location=device)
        base_model.load_state_dict(ckpt["model_state_dict"])

    base_model = base_model.to(device)

    feel_config = FEELConfig(
        layerdrop_eco=0.4,
        layerdrop_balanced=0.0,
        min_layers=4,
    )
    model = FEELModel(base_model, feel_config).to(device)

    print(f"   Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")

    # Trainer
    print(f"\n5. Phase {config.phase} Trainer...")
    if config.phase == 1:
        trainer = Phase1Trainer(model, config, telemetry, device)
    elif config.phase == 2:
        # Need frozen baseline
        baseline_base = create_embodied_slm_30m() if config.model_size == "30m" else create_embodied_slm_125m()
        if config.checkpoint:
            baseline_base.load_state_dict(ckpt["model_state_dict"])
        baseline_base = baseline_base.to(device)
        baseline = FEELModel(baseline_base, feel_config).to(device)
        trainer = Phase2Trainer(model, baseline, config, telemetry, device)
    else:
        trainer = Phase3Trainer(model, config, telemetry, device)

    # Training
    print(f"\n6. Training...")
    print(f"   Epochs: {config.epochs}")
    print(f"   Batch size: {config.batch_size}")
    print(f"   Steps/epoch: {len(train_loader)}")

    best_val_loss = float("inf")
    history = {"train": [], "val": [], "ppl": []}

    for epoch in range(config.epochs):
        print(f"\n--- Epoch {epoch + 1}/{config.epochs} ---")

        model.train()
        epoch_losses = []

        for batch_idx, batch in enumerate(train_loader):
            losses = trainer.train_step(batch)
            epoch_losses.append(losses["total"])

            if (batch_idx + 1) % 50 == 0:
                avg = np.mean(epoch_losses[-50:])
                print(f"  Step {trainer.step}: loss={avg:.4f} (lm={losses['lm']:.4f})")

            if trainer.step % config.save_every == 0:
                save_checkpoint(
                    model, trainer.optimizer, trainer.step, epoch,
                    {"train_loss": np.mean(epoch_losses[-50:])},
                    output_dir / f"step_{trainer.step}.pt"
                )

        # Validation
        val_loss = evaluate(model, val_loader, telemetry, device)
        ppl = compute_perplexity(model, val_loader, device)

        train_loss = np.mean(epoch_losses)
        print(f"  Epoch {epoch + 1}: train={train_loss:.4f} val={val_loss:.4f} ppl={ppl:.1f}")

        history["train"].append(float(train_loss))
        history["val"].append(float(val_loss))
        history["ppl"].append(float(ppl))

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(
                model, trainer.optimizer, trainer.step, epoch,
                {"val_loss": val_loss, "ppl": ppl},
                output_dir / "best.pt"
            )

    # Final save
    save_checkpoint(
        model, trainer.optimizer, trainer.step, config.epochs,
        {"final_val_loss": best_val_loss},
        output_dir / "final.pt"
    )

    with open(output_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    print(f"\n7. Training complete!")
    print(f"   Best val loss: {best_val_loss:.4f}")
    print(f"   Final PPL: {history['ppl'][-1]:.1f}")
    print(f"   Output: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="FEEL Real Training")
    parser.add_argument("--phase", type=int, required=True, choices=[1, 2, 3])
    parser.add_argument("--model-size", type=str, default="30m", choices=["30m", "125m"])
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--train-samples", type=int, default=5000)
    parser.add_argument("--val-samples", type=int, default=500)
    parser.add_argument("--output-dir", type=str, default="results/z121_train")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    config = TrainConfig(
        phase=args.phase,
        model_size=args.model_size,
        checkpoint=args.checkpoint,
        epochs=args.epochs,
        batch_size=args.batch_size,
        train_samples=args.train_samples,
        val_samples=args.val_samples,
        output_dir=args.output_dir,
    )

    train_phase(config, args.device)


if __name__ == "__main__":
    main()
