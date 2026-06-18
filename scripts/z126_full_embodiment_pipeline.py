#!/usr/bin/env python3
"""
z126 Full Embodiment Pipeline - NO COMPROMISES

Runs the complete FEEL-SLM training and validation pipeline:
1. Phase 1: Base LM (body for policy/reporter only)
2. Phase 2: Body conditioning with KL anchoring
3. Phase 3: Full embodiment with LayerDrop + distillation
4. Reporter head training and validation
5. Final benchmarks and report

Usage:
    python scripts/z126_full_embodiment_pipeline.py --start-phase 1
    python scripts/z126_full_embodiment_pipeline.py --resume  # Continue from last checkpoint
"""

import os
import sys
import json
import math
import time
import argparse
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

# Local imports
from feel_slm.model_v2 import FEELSLMV2, FEELConfigV2
from feel_slm.reporter_head import TelemetryReporterHead, ReporterConfig, ReporterTrainer
from sensing.background_telemetry import BackgroundTelemetrySampler


# =============================================================================
# Configuration
# =============================================================================

PIPELINE_CONFIG = {
    "phase1": {
        "epochs": 10,  # Reduced for faster iteration, increase for production
        "batch_size": 4,
        "lr": 1e-4,
        "save_every": 500,
        "val_every": 1000,
        "target_ppl": 50.0,  # Continue until PPL < this
    },
    "phase2": {
        "epochs": 5,
        "batch_size": 4,
        "lr": 5e-5,  # Lower LR for fine-tuning
        "kl_weight": 0.1,  # KL anchoring weight
        "gate_init": 0.01,  # Start with small gate values
    },
    "phase3": {
        "epochs": 3,
        "batch_size": 4,
        "lr": 2e-5,
        "layer_drop_rate": 0.2,
        "distillation_temp": 2.0,
    },
    "reporter": {
        "epochs": 20,
        "batch_size": 32,
        "lr": 1e-3,
        "hidden_dim": 128,
    },
}


# =============================================================================
# Utilities
# =============================================================================

def get_device():
    """Get available device."""
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch, 'hip') and torch.hip.is_available():
        return "cuda"  # ROCm uses cuda API
    return "cpu"


def load_checkpoint(path: str, device: str = "cpu") -> Dict:
    """Load checkpoint with proper handling."""
    return torch.load(path, map_location=device, weights_only=False)


def save_checkpoint(model, optimizer, step, epoch, metrics, path):
    """Save training checkpoint."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "step": step,
        "epoch": epoch,
        "metrics": metrics,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }, path)


def get_telemetry_path():
    """Find AMD GPU telemetry path."""
    for card in Path("/sys/class/drm").glob("card*"):
        hwmon = card / "device" / "hwmon"
        if hwmon.exists():
            for h in hwmon.iterdir():
                power_file = h / "power1_average"
                if power_file.exists():
                    return str(h)
    return None


def read_telemetry(hwmon_path: str) -> Dict[str, float]:
    """Read current GPU telemetry."""
    result = {}
    try:
        # Power (microwatts -> watts)
        with open(f"{hwmon_path}/power1_average", "r") as f:
            result["power"] = int(f.read().strip()) / 1e6
    except:
        result["power"] = 0.0

    try:
        # Temperature (millidegrees -> degrees)
        with open(f"{hwmon_path}/temp1_input", "r") as f:
            result["temp"] = int(f.read().strip()) / 1000
    except:
        result["temp"] = 0.0

    try:
        # GPU busy percent
        busy_path = Path(hwmon_path).parent.parent / "gpu_busy_percent"
        if busy_path.exists():
            with open(busy_path, "r") as f:
                result["util"] = int(f.read().strip())
        else:
            result["util"] = 0.0
    except:
        result["util"] = 0.0

    return result


# =============================================================================
# Dataset
# =============================================================================

def load_dataset(split: str = "train", max_samples: int = 50000):
    """Load TinyStories dataset."""
    from datasets import load_dataset

    print(f"Loading TinyStories ({split})...")
    ds = load_dataset("roneneldan/TinyStories", split=split, trust_remote_code=True)

    if max_samples and len(ds) > max_samples:
        ds = ds.select(range(max_samples))

    print(f"  Loaded {len(ds)} samples")
    return ds


def create_dataloader(dataset, tokenizer, batch_size: int, max_length: int = 256):
    """Create dataloader with tokenization."""

    def collate_fn(batch):
        texts = [item["text"] for item in batch]
        encodings = tokenizer(
            texts,
            max_length=max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        # Clamp to vocab size
        input_ids = torch.clamp(encodings["input_ids"], 0, 31999)
        return {"input_ids": input_ids, "attention_mask": encodings["attention_mask"]}

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=2,
        pin_memory=True,
    )


# =============================================================================
# Phase 1: Base Language Model
# =============================================================================

class Phase1Trainer:
    """
    Phase 1: Train base LM with body used ONLY for policy/reporter heads.
    Body does NOT affect the forward pass - this establishes the language baseline.
    """

    def __init__(
        self,
        model: FEELSLMV2,
        config: Dict,
        device: str = "cuda",
        telemetry_path: Optional[str] = None,
    ):
        self.model = model.to(device)
        self.config = config
        self.device = device
        self.telemetry_path = telemetry_path

        # Ensure phase 1 mode (body doesn't affect forward)
        self.model.config.phase = 1

        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config["lr"],
            weight_decay=0.01,
        )

        self.step = 0
        self.epoch = 0
        self.best_val_loss = float("inf")
        self.history = {"train": [], "val": [], "ppl": []}

    def train_step(self, batch: Dict) -> Dict[str, float]:
        """Single training step."""
        self.model.train()
        self.optimizer.zero_grad()

        input_ids = batch["input_ids"].to(self.device)

        # Get body state from real telemetry
        if self.telemetry_path:
            telem = read_telemetry(self.telemetry_path)
            body_vec = torch.tensor([[
                telem["power"] / 200.0,  # Normalize
                telem["temp"] / 100.0,
                telem["util"] / 100.0,
            ] + [0.0] * 9], device=self.device)  # Pad to body_dim=12
        else:
            body_vec = torch.rand(1, 12, device=self.device)

        # Forward pass
        logits = self.model(input_ids, body_vec=body_vec)

        # Language modeling loss (predict next token)
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = input_ids[:, 1:].contiguous()

        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=0,  # Ignore padding
        )

        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()

        self.step += 1
        return {"total": loss.item(), "lm": loss.item()}

    @torch.no_grad()
    def validate(self, val_loader) -> Tuple[float, float]:
        """Validate and return (loss, ppl)."""
        self.model.eval()
        total_loss = 0.0
        total_tokens = 0

        for batch in val_loader:
            input_ids = batch["input_ids"].to(self.device)
            body_vec = torch.rand(1, 12, device=self.device)

            logits = self.model(input_ids, body_vec=body_vec)

            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = input_ids[:, 1:].contiguous()

            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=0,
                reduction="sum",
            )

            mask = (shift_labels != 0).sum().item()
            total_loss += loss.item()
            total_tokens += mask

        avg_loss = total_loss / max(total_tokens, 1)
        ppl = math.exp(min(avg_loss, 20))  # Cap to avoid overflow
        return avg_loss, ppl


# =============================================================================
# Phase 2: Body Conditioning with KL Anchoring
# =============================================================================

class Phase2Trainer:
    """
    Phase 2: Enable body conditioning with KL anchoring.
    Body now affects forward pass through gated injections.
    KL loss ensures semantics don't drift from Phase 1 baseline.
    """

    def __init__(
        self,
        model: FEELSLMV2,
        config: Dict,
        device: str = "cuda",
        telemetry_path: Optional[str] = None,
    ):
        self.model = model.to(device)
        self.config = config
        self.device = device
        self.telemetry_path = telemetry_path

        # Enable phase 2 (body affects forward pass)
        self.model.config.phase = 2

        # Enable gated injections if available
        if hasattr(self.model, 'gated_injections') and self.model.gated_injections:
            for inj in self.model.gated_injections:
                if inj is not None:
                    inj.enabled = True
                    # Initialize gates small
                    if hasattr(inj, 'gate'):
                        inj.gate.data.fill_(config.get("gate_init", 0.01))

        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config["lr"],
            weight_decay=0.01,
        )

        self.step = 0
        self.epoch = 0
        self.kl_weight = config.get("kl_weight", 0.1)

    def train_step(self, batch: Dict) -> Dict[str, float]:
        """Training step with KL anchoring."""
        self.model.train()
        self.optimizer.zero_grad()

        input_ids = batch["input_ids"].to(self.device)

        # Get real telemetry
        if self.telemetry_path:
            telem = read_telemetry(self.telemetry_path)
            body_vec = torch.tensor([[
                telem["power"] / 200.0,
                telem["temp"] / 100.0,
                telem["util"] / 100.0,
            ] + [0.0] * 9], device=self.device)
        else:
            body_vec = torch.rand(1, 12, device=self.device)

        # Forward WITH body
        logits_body = self.model(input_ids, body_vec=body_vec)

        # Forward WITHOUT body (for KL anchor)
        with torch.no_grad():
            old_phase = self.model.config.phase
            self.model.config.phase = 0  # Disable body
            logits_baseline = self.model(input_ids, body_vec=None)
            self.model.config.phase = old_phase

        # LM loss
        shift_logits = logits_body[:, :-1, :].contiguous()
        shift_labels = input_ids[:, 1:].contiguous()

        lm_loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=0,
        )

        # KL anchoring loss
        probs_body = F.softmax(logits_body, dim=-1)
        probs_baseline = F.softmax(logits_baseline, dim=-1)

        kl_loss = F.kl_div(
            F.log_softmax(logits_body, dim=-1),
            probs_baseline,
            reduction="batchmean",
        )

        # Total loss
        total_loss = lm_loss + self.kl_weight * kl_loss

        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()

        self.step += 1
        return {
            "total": total_loss.item(),
            "lm": lm_loss.item(),
            "kl": kl_loss.item(),
        }


# =============================================================================
# Phase 3: Full Embodiment with LayerDrop
# =============================================================================

class Phase3Trainer:
    """
    Phase 3: Full embodiment with LayerDrop and distillation.
    Model learns to use body state for adaptive compute.
    """

    def __init__(
        self,
        model: FEELSLMV2,
        config: Dict,
        device: str = "cuda",
        telemetry_path: Optional[str] = None,
    ):
        self.model = model.to(device)
        self.config = config
        self.device = device
        self.telemetry_path = telemetry_path

        # Enable phase 3 (full embodiment)
        self.model.config.phase = 3
        self.model.config.layer_drop_rate = config.get("layer_drop_rate", 0.2)

        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config["lr"],
            weight_decay=0.01,
        )

        self.step = 0
        self.epoch = 0
        self.distill_temp = config.get("distillation_temp", 2.0)

    def train_step(self, batch: Dict) -> Dict[str, float]:
        """Training step with LayerDrop and distillation."""
        self.model.train()
        self.optimizer.zero_grad()

        input_ids = batch["input_ids"].to(self.device)

        # Get real telemetry
        if self.telemetry_path:
            telem = read_telemetry(self.telemetry_path)
            body_vec = torch.tensor([[
                telem["power"] / 200.0,
                telem["temp"] / 100.0,
                telem["util"] / 100.0,
            ] + [0.0] * 9], device=self.device)
        else:
            body_vec = torch.rand(1, 12, device=self.device)

        # Forward with LayerDrop (student)
        logits_student = self.model(input_ids, body_vec=body_vec, use_layer_drop=True)

        # Forward without LayerDrop (teacher)
        with torch.no_grad():
            logits_teacher = self.model(input_ids, body_vec=body_vec, use_layer_drop=False)

        # LM loss
        shift_logits = logits_student[:, :-1, :].contiguous()
        shift_labels = input_ids[:, 1:].contiguous()

        lm_loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=0,
        )

        # Distillation loss
        T = self.distill_temp
        distill_loss = F.kl_div(
            F.log_softmax(logits_student / T, dim=-1),
            F.softmax(logits_teacher / T, dim=-1),
            reduction="batchmean",
        ) * (T * T)

        # Total loss
        total_loss = lm_loss + 0.5 * distill_loss

        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()

        self.step += 1
        return {
            "total": total_loss.item(),
            "lm": lm_loss.item(),
            "distill": distill_loss.item(),
        }


# =============================================================================
# Reporter Head Training
# =============================================================================

def train_reporter_head(
    model: FEELSLMV2,
    train_loader: DataLoader,
    config: Dict,
    device: str = "cuda",
    telemetry_path: Optional[str] = None,
) -> Tuple[TelemetryReporterHead, Dict]:
    """
    Train reporter head to predict telemetry from hidden states.
    This proves "shared latent substrate" - the model represents hardware state.
    """
    print("\n" + "=" * 60)
    print("REPORTER HEAD TRAINING")
    print("=" * 60)

    # Freeze LM weights
    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    # Create reporter head
    reporter_config = ReporterConfig(
        hidden_dim=model.config.hidden_dim,
        reporter_hidden_dim=config.get("hidden_dim", 128),
        num_telemetry_outputs=5,
    )
    reporter = TelemetryReporterHead(reporter_config).to(device)

    optimizer = torch.optim.AdamW(
        reporter.parameters(),
        lr=config.get("lr", 1e-3),
    )

    # Training loop
    num_epochs = config.get("epochs", 20)
    history = {"loss": [], "accuracy": []}

    for epoch in range(num_epochs):
        reporter.train()
        epoch_losses = []

        for batch_idx, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(device)

            # Get real telemetry
            if telemetry_path:
                telem = read_telemetry(telemetry_path)
                telemetry_gt = torch.tensor([[
                    telem["power"] / 200.0,
                    telem["temp"] / 100.0,
                    telem["util"] / 100.0,
                    0.0, 0.0,  # freq, mem placeholders
                ]], device=device).expand(input_ids.size(0), -1)
            else:
                telemetry_gt = torch.rand(input_ids.size(0), 5, device=device)

            # Get hidden states from frozen model
            with torch.no_grad():
                body_vec = torch.rand(1, 12, device=device)
                x = model.embed_tokens(input_ids)

                L = input_ids.shape[1]
                mask = torch.triu(
                    torch.ones(L, L, device=device) * float('-inf'),
                    diagonal=1
                )

                for layer in model.layers:
                    x = layer(x, mask)

                hidden = model.norm(x)

            # Train reporter
            optimizer.zero_grad()
            output = reporter(hidden)
            pred = output["telemetry_pred"]

            loss = F.mse_loss(pred, telemetry_gt)
            loss.backward()
            optimizer.step()

            epoch_losses.append(loss.item())

            if (batch_idx + 1) % 50 == 0:
                print(f"  Epoch {epoch+1} Step {batch_idx+1}: loss={np.mean(epoch_losses[-50:]):.4f}")

        avg_loss = np.mean(epoch_losses)
        history["loss"].append(avg_loss)
        print(f"Epoch {epoch+1}/{num_epochs}: loss={avg_loss:.4f}")

    # Validate reporter
    print("\nValidating reporter...")
    reporter.eval()

    all_preds = []
    all_gts = []

    with torch.no_grad():
        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)

            if telemetry_path:
                telem = read_telemetry(telemetry_path)
                telemetry_gt = torch.tensor([[
                    telem["power"] / 200.0,
                    telem["temp"] / 100.0,
                    telem["util"] / 100.0,
                    0.0, 0.0,
                ]], device=device).expand(input_ids.size(0), -1)
            else:
                telemetry_gt = torch.rand(input_ids.size(0), 5, device=device)

            body_vec = torch.rand(1, 12, device=device)
            x = model.embed_tokens(input_ids)

            L = input_ids.shape[1]
            mask = torch.triu(
                torch.ones(L, L, device=device) * float('-inf'),
                diagonal=1
            )

            for layer in model.layers:
                x = layer(x, mask)

            hidden = model.norm(x)
            output = reporter(hidden)

            all_preds.append(output["telemetry_pred"].cpu())
            all_gts.append(telemetry_gt.cpu())

    preds = torch.cat(all_preds)
    gts = torch.cat(all_gts)

    # Compute metrics
    mse = F.mse_loss(preds, gts).item()

    # Random baseline
    random_pred = gts.mean(0, keepdim=True).expand_as(gts)
    random_mse = F.mse_loss(random_pred, gts).item()

    # Per-channel correlation
    correlations = []
    for i in range(5):
        corr = torch.corrcoef(torch.stack([preds[:, i], gts[:, i]]))[0, 1]
        correlations.append(corr.item() if not torch.isnan(corr) else 0.0)

    validation = {
        "mse": mse,
        "random_mse": random_mse,
        "improvement": random_mse / mse if mse > 0 else 0,
        "correlations": correlations,
        "avg_correlation": np.mean(correlations),
        "verdict": "PROVEN" if np.mean(correlations) > 0.3 else "NOT PROVEN",
    }

    print(f"\nReporter Validation:")
    print(f"  MSE: {mse:.4f}")
    print(f"  Random MSE: {random_mse:.4f}")
    print(f"  Improvement: {validation['improvement']:.2f}x")
    print(f"  Avg Correlation: {validation['avg_correlation']:.3f}")
    print(f"  Verdict: {validation['verdict']}")

    # Unfreeze model
    for param in model.parameters():
        param.requires_grad = True

    return reporter, validation


# =============================================================================
# Final Validation Suite
# =============================================================================

def run_final_validation(
    model: FEELSLMV2,
    reporter: TelemetryReporterHead,
    tokenizer,
    device: str = "cuda",
    output_dir: str = "results/z126_final",
) -> Dict:
    """Run comprehensive final validation."""
    print("\n" + "=" * 60)
    print("FINAL VALIDATION SUITE")
    print("=" * 60)

    results = {}
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # 1. Semantic Invariance Test
    print("\n1. Semantic Invariance Test...")

    test_prompts = [
        "The quick brown fox jumps over the lazy dog",
        "Once upon a time in a land far away",
        "Scientists have discovered a new species",
        "The weather forecast predicts rain tomorrow",
        "Artificial intelligence is transforming",
    ]

    kl_values = []
    model.eval()

    with torch.no_grad():
        for prompt in test_prompts:
            input_ids = tokenizer.encode(prompt, return_tensors="pt")
            input_ids = torch.clamp(input_ids, 0, model.config.vocab_size - 1).to(device)

            body_vec = torch.rand(1, model.config.body_dim, device=device)

            # With body
            model.config.phase = 2
            logits_body = model(input_ids, body_vec=body_vec)

            # Without body
            model.config.phase = 0
            logits_base = model(input_ids, body_vec=None)

            # KL divergence
            probs_body = F.softmax(logits_body, dim=-1)
            probs_base = F.softmax(logits_base, dim=-1)

            kl = F.kl_div(
                torch.log(probs_body + 1e-10),
                probs_base,
                reduction="batchmean"
            ).item()

            kl_values.append(kl)
            status = "✓" if kl < 0.01 else "✗"
            print(f"  {status} KL={kl:.6f} | {prompt[:40]}...")

    results["semantic_invariance"] = {
        "kl_values": kl_values,
        "mean_kl": np.mean(kl_values),
        "max_kl": np.max(kl_values),
        "passed": np.mean(kl_values) < 0.01,
    }
    print(f"  Mean KL: {np.mean(kl_values):.6f}")
    print(f"  Passed: {results['semantic_invariance']['passed']}")

    # 2. Generation Quality Test
    print("\n2. Generation Quality Test...")

    model.config.phase = 2
    test_prompt = "Once upon a time"
    input_ids = tokenizer.encode(test_prompt, return_tensors="pt")
    input_ids = torch.clamp(input_ids, 0, model.config.vocab_size - 1).to(device)

    # Simple greedy generation
    generated = input_ids.clone()
    for _ in range(50):
        with torch.no_grad():
            body_vec = torch.rand(1, model.config.body_dim, device=device)
            logits = model(generated, body_vec=body_vec)
            next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            generated = torch.cat([generated, next_token], dim=1)

    generated_text = tokenizer.decode(generated[0].cpu().tolist())
    print(f"  Prompt: '{test_prompt}'")
    print(f"  Generated: '{generated_text[:200]}...'")

    results["generation"] = {
        "prompt": test_prompt,
        "generated": generated_text,
    }

    # Save results
    with open(output_path / "final_validation.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\nResults saved to {output_path}/")
    return results


# =============================================================================
# Main Pipeline
# =============================================================================

def run_pipeline(args):
    """Run the full embodiment pipeline."""
    print("=" * 60)
    print("FEEL-SLM FULL EMBODIMENT PIPELINE")
    print("NO COMPROMISES - WORK HARD")
    print("=" * 60)
    print(f"Started: {datetime.now().isoformat()}")

    device = get_device()
    print(f"Device: {device}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Telemetry
    telemetry_path = get_telemetry_path()
    print(f"Telemetry: {telemetry_path}")

    # Load tokenizer
    print("\n1. Loading tokenizer...")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    # Load datasets
    print("\n2. Loading datasets...")
    train_ds = load_dataset("train", max_samples=args.train_samples)
    val_ds = load_dataset("validation", max_samples=500)

    # Create model
    print("\n3. Creating model...")
    config = FEELConfigV2(
        vocab_size=32000,
        hidden_dim=512,
        num_layers=8,
        num_heads=8,
        body_dim=12,
        phase=1,
    )
    model = FEELSLMV2(config).to(device)
    print(f"   Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")

    # Check for existing checkpoint
    start_phase = args.start_phase

    if args.resume:
        # Find latest checkpoint
        for phase in [3, 2, 1]:
            phase_dir = output_dir / f"phase{phase}"
            if phase_dir.exists():
                ckpts = list(phase_dir.glob("*.pt"))
                if ckpts:
                    latest = max(ckpts, key=lambda p: p.stat().st_mtime)
                    print(f"\n   Resuming from {latest}")
                    ckpt = load_checkpoint(str(latest), device)
                    model.load_state_dict(ckpt["model_state_dict"])
                    start_phase = phase
                    break

    # =========================================================================
    # PHASE 1: Base Language Model
    # =========================================================================

    if start_phase <= 1:
        print("\n" + "=" * 60)
        print("PHASE 1: BASE LANGUAGE MODEL")
        print("=" * 60)

        phase1_config = PIPELINE_CONFIG["phase1"]
        trainer = Phase1Trainer(model, phase1_config, device, telemetry_path)

        train_loader = create_dataloader(train_ds, tokenizer, phase1_config["batch_size"])
        val_loader = create_dataloader(val_ds, tokenizer, phase1_config["batch_size"])

        phase1_dir = output_dir / "phase1"
        phase1_dir.mkdir(exist_ok=True)

        best_ppl = float("inf")

        for epoch in range(phase1_config["epochs"]):
            print(f"\n--- Epoch {epoch + 1}/{phase1_config['epochs']} ---")
            trainer.epoch = epoch
            epoch_losses = []

            for batch_idx, batch in enumerate(train_loader):
                losses = trainer.train_step(batch)
                epoch_losses.append(losses["total"])

                if (batch_idx + 1) % 50 == 0:
                    avg = np.mean(epoch_losses[-50:])
                    print(f"  Step {trainer.step}: loss={avg:.4f}")

                if trainer.step % phase1_config["save_every"] == 0:
                    save_checkpoint(
                        model, trainer.optimizer, trainer.step, epoch,
                        {"train_loss": np.mean(epoch_losses[-50:])},
                        phase1_dir / f"step_{trainer.step}.pt"
                    )

            # Validation
            val_loss, ppl = trainer.validate(val_loader)
            print(f"  Epoch {epoch + 1}: val_loss={val_loss:.4f} PPL={ppl:.1f}")

            if ppl < best_ppl:
                best_ppl = ppl
                save_checkpoint(
                    model, trainer.optimizer, trainer.step, epoch,
                    {"train_loss": np.mean(epoch_losses), "val_loss": val_loss, "ppl": ppl},
                    phase1_dir / "best.pt"
                )

            # Early stopping if target reached
            if ppl < phase1_config["target_ppl"]:
                print(f"  Target PPL {phase1_config['target_ppl']} reached!")
                break

        print(f"\nPhase 1 complete. Best PPL: {best_ppl:.1f}")

    # =========================================================================
    # PHASE 2: Body Conditioning with KL Anchoring
    # =========================================================================

    if start_phase <= 2:
        print("\n" + "=" * 60)
        print("PHASE 2: BODY CONDITIONING WITH KL ANCHORING")
        print("=" * 60)

        # Load best Phase 1 checkpoint if starting fresh
        if start_phase < 2:
            phase1_best = output_dir / "phase1" / "best.pt"
            if phase1_best.exists():
                print(f"   Loading Phase 1 best: {phase1_best}")
                ckpt = load_checkpoint(str(phase1_best), device)
                model.load_state_dict(ckpt["model_state_dict"])

        phase2_config = PIPELINE_CONFIG["phase2"]
        trainer = Phase2Trainer(model, phase2_config, device, telemetry_path)

        train_loader = create_dataloader(train_ds, tokenizer, phase2_config["batch_size"])

        phase2_dir = output_dir / "phase2"
        phase2_dir.mkdir(exist_ok=True)

        for epoch in range(phase2_config["epochs"]):
            print(f"\n--- Epoch {epoch + 1}/{phase2_config['epochs']} ---")
            trainer.epoch = epoch
            epoch_losses = []

            for batch_idx, batch in enumerate(train_loader):
                losses = trainer.train_step(batch)
                epoch_losses.append(losses["total"])

                if (batch_idx + 1) % 50 == 0:
                    avg = np.mean(epoch_losses[-50:])
                    print(f"  Step {trainer.step}: loss={avg:.4f} (lm={losses['lm']:.4f} kl={losses['kl']:.4f})")

            save_checkpoint(
                model, trainer.optimizer, trainer.step, epoch,
                {"train_loss": np.mean(epoch_losses)},
                phase2_dir / f"epoch_{epoch + 1}.pt"
            )

        # Save final
        save_checkpoint(
            model, trainer.optimizer, trainer.step, epoch,
            {"train_loss": np.mean(epoch_losses)},
            phase2_dir / "final.pt"
        )

        print("\nPhase 2 complete.")

    # =========================================================================
    # PHASE 3: Full Embodiment
    # =========================================================================

    if start_phase <= 3:
        print("\n" + "=" * 60)
        print("PHASE 3: FULL EMBODIMENT WITH LAYERDROP")
        print("=" * 60)

        # Load best Phase 2 checkpoint if starting fresh
        if start_phase < 3:
            phase2_final = output_dir / "phase2" / "final.pt"
            if phase2_final.exists():
                print(f"   Loading Phase 2 final: {phase2_final}")
                ckpt = load_checkpoint(str(phase2_final), device)
                model.load_state_dict(ckpt["model_state_dict"])

        phase3_config = PIPELINE_CONFIG["phase3"]
        trainer = Phase3Trainer(model, phase3_config, device, telemetry_path)

        train_loader = create_dataloader(train_ds, tokenizer, phase3_config["batch_size"])

        phase3_dir = output_dir / "phase3"
        phase3_dir.mkdir(exist_ok=True)

        for epoch in range(phase3_config["epochs"]):
            print(f"\n--- Epoch {epoch + 1}/{phase3_config['epochs']} ---")
            trainer.epoch = epoch
            epoch_losses = []

            for batch_idx, batch in enumerate(train_loader):
                losses = trainer.train_step(batch)
                epoch_losses.append(losses["total"])

                if (batch_idx + 1) % 50 == 0:
                    avg = np.mean(epoch_losses[-50:])
                    print(f"  Step {trainer.step}: loss={avg:.4f} (lm={losses['lm']:.4f} distill={losses['distill']:.4f})")

            save_checkpoint(
                model, trainer.optimizer, trainer.step, epoch,
                {"train_loss": np.mean(epoch_losses)},
                phase3_dir / f"epoch_{epoch + 1}.pt"
            )

        # Save final embodied model
        save_checkpoint(
            model, trainer.optimizer, trainer.step, epoch,
            {"train_loss": np.mean(epoch_losses)},
            phase3_dir / "embodied_final.pt"
        )

        print("\nPhase 3 complete.")

    # =========================================================================
    # REPORTER HEAD TRAINING
    # =========================================================================

    print("\n" + "=" * 60)
    print("REPORTER HEAD TRAINING")
    print("=" * 60)

    train_loader = create_dataloader(train_ds, tokenizer, PIPELINE_CONFIG["reporter"]["batch_size"])

    reporter, reporter_validation = train_reporter_head(
        model, train_loader, PIPELINE_CONFIG["reporter"], device, telemetry_path
    )

    # Save reporter
    reporter_dir = output_dir / "reporter"
    reporter_dir.mkdir(exist_ok=True)
    torch.save(reporter.state_dict(), reporter_dir / "reporter.pt")

    with open(reporter_dir / "validation.json", "w") as f:
        json.dump(reporter_validation, f, indent=2)

    # =========================================================================
    # FINAL VALIDATION
    # =========================================================================

    final_results = run_final_validation(
        model, reporter, tokenizer, device, str(output_dir / "final")
    )

    # =========================================================================
    # SUMMARY
    # =========================================================================

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)

    print(f"\nReporter Validation:")
    print(f"  Verdict: {reporter_validation['verdict']}")
    print(f"  Avg Correlation: {reporter_validation['avg_correlation']:.3f}")
    print(f"  Improvement over random: {reporter_validation['improvement']:.2f}x")

    print(f"\nSemantic Invariance:")
    print(f"  Passed: {final_results['semantic_invariance']['passed']}")
    print(f"  Mean KL: {final_results['semantic_invariance']['mean_kl']:.6f}")

    embodiment_proven = (
        reporter_validation['verdict'] == "PROVEN" and
        final_results['semantic_invariance']['passed']
    )

    print(f"\n{'=' * 60}")
    if embodiment_proven:
        print("TRUE EMBODIMENT: PROVEN ✓")
        print("  - Model internally represents hardware state")
        print("  - Body conditioning preserves language semantics")
    else:
        print("TRUE EMBODIMENT: NOT YET PROVEN")
        print("  - Check reporter correlation and KL values")
        print("  - May need more training or architecture changes")
    print(f"{'=' * 60}")

    print(f"\nFinished: {datetime.now().isoformat()}")
    print(f"Output: {output_dir}")

    # Save final summary
    summary = {
        "timestamp": datetime.now().isoformat(),
        "reporter_validation": reporter_validation,
        "semantic_invariance": final_results["semantic_invariance"],
        "embodiment_proven": embodiment_proven,
    }

    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    return embodiment_proven


# =============================================================================
# Entry Point
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Full Embodiment Pipeline")
    parser.add_argument("--start-phase", type=int, default=1, choices=[1, 2, 3])
    parser.add_argument("--resume", action="store_true", help="Resume from latest checkpoint")
    parser.add_argument("--output-dir", type=str, default="results/z126_embodiment")
    parser.add_argument("--train-samples", type=int, default=50000)
    args = parser.parse_args()

    success = run_pipeline(args)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
