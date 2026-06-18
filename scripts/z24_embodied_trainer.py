#!/usr/bin/env python3
"""
FEEL z24: Embodied Training Script
==================================
Complete training pipeline for closed-loop embodied LLM.

Features:
- Multi-objective loss (CE + Metabolic + Strain + Stability + Hardware Target)
- Rich sensor integration (32-dim AMD GPU telemetry)
- Hard-skip blocks with real FLOP savings
- FiLM conditioning for sensor → hidden modulation
- Comprehensive WandB logging (50+ metrics)
- Periodic validation with causal ablations
- Stress curriculum (gradually increasing challenge)

WandB Metrics Logged:
  Training:
    - train/ce_loss, train/total_loss
    - train/metabolic_penalty, train/strain_alignment
    - train/stability_penalty, train/hardware_target_loss
    - train/avg_gate, train/min_gate, train/max_gate
    - train/skip_rate, train/gates_open
    - train/avg_strain, train/total_strain
    - train/film_gamma_mean, train/film_beta_mean
    - train/stress_level, train/lr

  Sensors (per step):
    - sensors/edge_temp, sensors/socket_power
    - sensors/gpu_busy, sensors/vram_pct
    - sensors/stress_composite
    - sensors/thermal_headroom, sensors/power_efficiency

  Validation:
    - val/ce_loss, val/perplexity
    - val/high_stress_words, val/low_stress_words
    - val/word_diff, val/word_ratio
    - val/high_stress_gate, val/low_stress_gate
    - val/gate_sensor_correlation
    - val/strain_behavior_correlation

  Hardware:
    - hardware/real_temp, hardware/real_power
    - hardware/real_utilization
    - hardware/flops_saved_estimate

  Ablation (periodic):
    - ablation/{mode}/word_diff
    - ablation/{mode}/causal_score

Author: FEEL Research Team
Date: 2026-01-13
"""

import os
import sys
import json
import time
import torch
import torch.nn as nn
import numpy as np
import wandb
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
from typing import List, Dict, Optional, Tuple
from torch.utils.data import Dataset, DataLoader
import argparse

os.environ.setdefault("HSA_OVERRIDE_GFX_VERSION", "11.0.0")

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from modeling.z24_sensor_hub import (
    AMDSensorHub, SimulatedSensorHub, create_sensor_hub,
    SENSOR_DIM, SENSOR_NAMES
)
from modeling.z24_embodied_model import (
    EmbodiedDeepSeek, EmbodiedLoss, EmbodiedModelOutput,
    load_embodied_model
)
from modeling.z24_causal_validation import CausalValidator


# =============================================================================
# Dataset
# =============================================================================

class EmbodiedDataset(Dataset):
    """
    Dataset for embodied training.

    Each sample has:
    - prompt: Input text
    - response: Target output
    - stress: Thermal stress level (0-1)
    - mode: "scholar" (verbose) or "survivor" (brief)
    """

    def __init__(
        self,
        data_path: str,
        tokenizer,
        max_length: int = 256,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = []

        with open(data_path) as f:
            for line in f:
                sample = json.loads(line)
                self.samples.append(sample)

        print(f"[EmbodiedDataset] Loaded {len(self.samples)} samples")

        # Compute stress distribution (support both formats)
        def get_stress(s):
            if "stress" in s:
                return s["stress"]
            return 0.7 if s.get("is_stressed", False) else 0.3

        stresses = [get_stress(s) for s in self.samples]
        print(f"  Stress range: [{min(stresses):.2f}, {max(stresses):.2f}]")
        print(f"  Stress mean: {np.mean(stresses):.2f}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # Support both prompt/response and input/output formats
        prompt = sample.get("prompt") or sample.get("input", "")
        response = sample.get("response") or sample.get("output", "")

        # Support stress from either explicit field or is_stressed boolean
        stress = sample.get("stress", None)
        if stress is None:
            is_stressed = sample.get("is_stressed", False)
            stress = 0.7 if is_stressed else 0.3

        mode = sample.get("mode", "neutral")

        full_text = f"{prompt}\n{response}"

        encoding = self.tokenizer(
            full_text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )

        # Target length for strain alignment
        response_tokens = self.tokenizer(
            response,
            truncation=True,
            max_length=self.max_length,
        )
        target_length = len(response_tokens["input_ids"])

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "stress": torch.tensor(stress, dtype=torch.float32),
            "target_length": target_length,
            "mode": mode,
        }


class EmbodiedCollator:
    """Collate batches with metadata."""

    def __call__(self, batch: List[Dict]) -> Dict:
        input_ids = torch.stack([b["input_ids"] for b in batch])
        attention_mask = torch.stack([b["attention_mask"] for b in batch])
        stress = torch.stack([b["stress"] for b in batch])
        target_lengths = [b["target_length"] for b in batch]
        modes = [b["mode"] for b in batch]

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": input_ids.clone(),
            "stress": stress,
            "target_lengths": target_lengths,
            "modes": modes,
        }


# =============================================================================
# Metrics Logger
# =============================================================================

class MetricsLogger:
    """
    Comprehensive metrics logging for WandB.

    Logs 50+ metrics covering:
    - Training losses
    - Gate statistics
    - Strain metrics
    - FiLM metrics
    - Sensor values
    - Hardware telemetry
    - Validation metrics
    """

    def __init__(self, sensor_hub=None):
        self.sensor_hub = sensor_hub
        self.step_count = 0

    def log_training_step(
        self,
        loss_metrics: Dict,
        model_output: EmbodiedModelOutput,
        sensors: torch.Tensor,
        stress_level: float,
        lr: float,
        step: int,
    ):
        """Log comprehensive training metrics."""
        metrics = {}

        # === Loss metrics ===
        metrics["train/ce_loss"] = loss_metrics.get("ce_loss", 0)
        metrics["train/total_loss"] = loss_metrics.get("total_loss", 0)
        metrics["train/metabolic_penalty"] = loss_metrics.get("metabolic_penalty", 0)
        metrics["train/strain_alignment"] = loss_metrics.get("strain_alignment", 0)
        metrics["train/stability_penalty"] = loss_metrics.get("stability_penalty", 0)
        metrics["train/hardware_target_loss"] = loss_metrics.get("hardware_target_loss", 0)

        # === Gate metrics ===
        if model_output.gate_probs:
            metrics["train/avg_gate"] = np.mean(model_output.gate_probs)
            metrics["train/min_gate"] = np.min(model_output.gate_probs)
            metrics["train/max_gate"] = np.max(model_output.gate_probs)
            metrics["train/gate_std"] = np.std(model_output.gate_probs)
        else:
            metrics["train/avg_gate"] = 0.5
            metrics["train/min_gate"] = 0.5
            metrics["train/max_gate"] = 0.5

        metrics["train/skip_rate"] = model_output.total_skips / max(model_output.total_layers, 1)
        metrics["train/gates_open"] = model_output.total_layers - model_output.total_skips

        # === Strain metrics ===
        if model_output.strain_magnitudes:
            metrics["train/avg_strain"] = np.mean(model_output.strain_magnitudes)
            metrics["train/total_strain"] = model_output.total_strain
            metrics["train/max_strain"] = np.max(model_output.strain_magnitudes)
        else:
            metrics["train/avg_strain"] = 0
            metrics["train/total_strain"] = 0

        # === FiLM metrics ===
        if model_output.film_gamma_means:
            metrics["train/film_gamma_mean"] = np.mean(model_output.film_gamma_means)
            metrics["train/film_beta_mean"] = np.mean(model_output.film_beta_means)

        # === Stress/sensor metrics ===
        metrics["train/stress_level"] = stress_level
        metrics["train/lr"] = lr

        # === Individual sensor values ===
        if sensors is not None and len(sensors) >= SENSOR_DIM:
            sensor_np = sensors.cpu().numpy()
            # Log key sensors
            metrics["sensors/edge_temp"] = sensor_np[0]
            metrics["sensors/socket_power"] = sensor_np[4]
            metrics["sensors/power_cap_pct"] = sensor_np[7]
            metrics["sensors/gpu_busy"] = sensor_np[14]
            metrics["sensors/vram_pct"] = sensor_np[17]
            metrics["sensors/stress_composite"] = sensor_np[31]
            metrics["sensors/thermal_headroom"] = sensor_np[30]
            metrics["sensors/power_efficiency"] = sensor_np[29]

        # === Hardware telemetry (from real sensor hub) ===
        if self.sensor_hub is not None:
            try:
                raw = self.sensor_hub.read_raw()
                metrics["hardware/real_temp"] = raw.get("edge_temp", 0)
                metrics["hardware/real_power"] = raw.get("socket_power", 0)
                metrics["hardware/real_utilization"] = raw.get("gpu_busy", 0)
                metrics["hardware/real_vram_pct"] = raw.get("vram_pct", 0)
            except Exception:
                pass

        wandb.log(metrics, step=step)
        self.step_count = step

    def log_validation(
        self,
        val_ce: float,
        high_stress_metrics: Dict,
        low_stress_metrics: Dict,
        step: int,
    ):
        """Log validation metrics."""
        metrics = {}

        metrics["val/ce_loss"] = val_ce
        metrics["val/perplexity"] = np.exp(val_ce)

        # High stress metrics
        metrics["val/high_stress_words"] = high_stress_metrics.get("avg_words", 0)
        metrics["val/high_stress_gate"] = high_stress_metrics.get("avg_gate", 0.5)
        metrics["val/high_stress_strain"] = high_stress_metrics.get("avg_strain", 0)

        # Low stress metrics
        metrics["val/low_stress_words"] = low_stress_metrics.get("avg_words", 0)
        metrics["val/low_stress_gate"] = low_stress_metrics.get("avg_gate", 0.5)
        metrics["val/low_stress_strain"] = low_stress_metrics.get("avg_strain", 0)

        # Differences
        word_diff = low_stress_metrics.get("avg_words", 0) - high_stress_metrics.get("avg_words", 0)
        metrics["val/word_diff"] = word_diff

        if low_stress_metrics.get("avg_words", 0) > 0:
            word_ratio = high_stress_metrics.get("avg_words", 0) / low_stress_metrics.get("avg_words", 1)
        else:
            word_ratio = 1.0
        metrics["val/word_ratio"] = word_ratio

        # Correlations
        metrics["val/gate_sensor_correlation"] = high_stress_metrics.get("gate_correlation", 0)
        metrics["val/strain_behavior_correlation"] = high_stress_metrics.get("strain_correlation", 0)

        wandb.log(metrics, step=step)

    def log_ablation(self, report, step: int):
        """Log ablation test results."""
        metrics = {}

        for mode, scores in report.hot_cold_word_diff.items():
            metrics[f"ablation/{mode}/word_diff"] = scores

        for mode, score in report.causal_scores.items():
            metrics[f"ablation/causal_score/{mode}"] = score

        metrics["ablation/overall_causal_score"] = report.overall_causal_score

        wandb.log(metrics, step=step)

    def log_epoch_summary(
        self,
        epoch: int,
        train_loss: float,
        val_loss: float,
        best_val_loss: float,
        epoch_metrics: Dict,
        step: int,
    ):
        """Log epoch summary."""
        metrics = {
            "epoch/num": epoch,
            "epoch/train_loss": train_loss,
            "epoch/val_loss": val_loss,
            "epoch/best_val_loss": best_val_loss,
        }

        for key, value in epoch_metrics.items():
            metrics[f"epoch/{key}"] = value

        wandb.log(metrics, step=step)


# =============================================================================
# Validation
# =============================================================================

def validate_embodiment(
    model: EmbodiedDeepSeek,
    val_loader: DataLoader,
    loss_fn: EmbodiedLoss,
    n_generation_samples: int = 10,
) -> Tuple[float, Dict, Dict]:
    """
    Validate embodied behavior.

    Returns:
        val_ce: Validation CE loss
        high_stress_metrics: Metrics at high stress
        low_stress_metrics: Metrics at low stress
    """
    model.base_model.eval()

    val_ce = 0.0
    high_stress_gates = []
    low_stress_gates = []
    high_stress_strains = []
    low_stress_strains = []
    high_stress_words = []
    low_stress_words = []

    # CE loss validation
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validation CE"):
            input_ids = batch["input_ids"].to("cuda")
            attention_mask = batch["attention_mask"].to("cuda")
            labels = batch["labels"].to("cuda")
            stress = batch["stress"]

            # Test at different stress levels
            for i, s in enumerate(stress):
                model.set_stress_level(s.item())

                outputs = model(
                    input_ids=input_ids[i:i+1],
                    attention_mask=attention_mask[i:i+1],
                    labels=labels[i:i+1],
                )

                val_ce += outputs.loss.item()

                if outputs.gate_probs:
                    avg_gate = np.mean(outputs.gate_probs)
                    avg_strain = np.mean(outputs.strain_magnitudes) if outputs.strain_magnitudes else 0

                    if s.item() > 0.6:
                        high_stress_gates.append(avg_gate)
                        high_stress_strains.append(avg_strain)
                    elif s.item() < 0.4:
                        low_stress_gates.append(avg_gate)
                        low_stress_strains.append(avg_strain)

    n_samples = len(val_loader.dataset)
    val_ce /= n_samples

    # Generation validation
    test_prompts = [
        "Explain how electricity works.",
        "Describe the water cycle.",
        "What causes earthquakes?",
    ]

    for prompt in test_prompts[:n_generation_samples]:
        # Low stress generation
        model.set_stress_level(0.2)
        response_low, stats_low = model.generate(prompt, max_new_tokens=100)
        response_text_low = response_low[len(prompt):].strip()
        low_stress_words.append(len(response_text_low.split()))

        # High stress generation
        model.set_stress_level(0.8)
        response_high, stats_high = model.generate(prompt, max_new_tokens=100)
        response_text_high = response_high[len(prompt):].strip()
        high_stress_words.append(len(response_text_high.split()))

    high_stress_metrics = {
        "avg_gate": np.mean(high_stress_gates) if high_stress_gates else 0.5,
        "avg_strain": np.mean(high_stress_strains) if high_stress_strains else 0,
        "avg_words": np.mean(high_stress_words) if high_stress_words else 0,
    }

    low_stress_metrics = {
        "avg_gate": np.mean(low_stress_gates) if low_stress_gates else 0.5,
        "avg_strain": np.mean(low_stress_strains) if low_stress_strains else 0,
        "avg_words": np.mean(low_stress_words) if low_stress_words else 0,
    }

    # Compute correlations
    all_gates = high_stress_gates + low_stress_gates
    all_strains = high_stress_strains + low_stress_strains
    all_stress = [0.8] * len(high_stress_gates) + [0.2] * len(low_stress_gates)

    if len(all_gates) > 2:
        gate_corr = np.corrcoef(all_gates, all_stress)[0, 1]
        high_stress_metrics["gate_correlation"] = gate_corr if not np.isnan(gate_corr) else 0
    else:
        high_stress_metrics["gate_correlation"] = 0

    if len(all_strains) > 2 and len(high_stress_words + low_stress_words) > 2:
        # Strain-behavior correlation
        all_words = high_stress_words + low_stress_words
        strain_for_words = [np.mean(high_stress_strains) if high_stress_strains else 0] * len(high_stress_words) + \
                          [np.mean(low_stress_strains) if low_stress_strains else 0] * len(low_stress_words)
        if len(strain_for_words) == len(all_words):
            strain_corr = np.corrcoef(strain_for_words, all_words)[0, 1]
            high_stress_metrics["strain_correlation"] = strain_corr if not np.isnan(strain_corr) else 0
        else:
            high_stress_metrics["strain_correlation"] = 0
    else:
        high_stress_metrics["strain_correlation"] = 0

    return val_ce, high_stress_metrics, low_stress_metrics


# =============================================================================
# Training
# =============================================================================

def train_embodied(
    # Data
    data_path: str = "data/ouroboros/ouroboros_train.jsonl",
    val_path: str = "data/ouroboros/ouroboros_val.jsonl",

    # Model
    base_model_path: str = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
    checkpoint_path: Optional[str] = None,
    gated_layers: List[int] = None,
    use_film: bool = True,
    use_strain: bool = True,

    # Training
    epochs: int = 3,
    batch_size: int = 2,
    gradient_accumulation: int = 8,
    learning_rate: float = 5e-4,
    max_length: int = 256,

    # Loss weights
    ce_weight: float = 1.0,
    metabolic_weight: float = 0.3,
    strain_weight: float = 0.2,
    stability_weight: float = 0.1,
    hardware_target_weight: float = 0.2,

    # Stress curriculum
    stress_curriculum: bool = True,
    initial_stress_range: Tuple[float, float] = (0.2, 0.6),
    final_stress_range: Tuple[float, float] = (0.1, 0.9),

    # Validation
    val_every_steps: int = 100,
    ablation_every_epochs: int = 1,

    # Hardware
    use_real_sensors: bool = True,
    card_id: int = 1,

    # Output
    output_dir: str = "models/embodied_z24",
    run_name: Optional[str] = None,
):
    """
    Train embodied model with comprehensive metrics logging.
    """
    if gated_layers is None:
        gated_layers = [3, 7, 11, 15, 19, 23, 27]

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("FEEL z24: EMBODIED TRAINING")
    print("=" * 70)
    print(f"Data:              {data_path}")
    print(f"Validation:        {val_path}")
    print(f"Output:            {output_dir}")
    print(f"Gated Layers:      {gated_layers}")
    print(f"FiLM: {use_film}, Strain: {use_strain}")
    print(f"Epochs:            {epochs}")
    print(f"Batch (eff):       {batch_size * gradient_accumulation}")
    print(f"Learning Rate:     {learning_rate}")
    print(f"Real Sensors:      {use_real_sensors}")
    print("=" * 70)

    # Initialize WandB
    if run_name is None:
        run_name = f"z24-embodied-{datetime.now().strftime('%Y%m%d_%H%M')}"

    wandb.init(
        project="feel-z24-embodied",
        name=run_name,
        config={
            "base_model": base_model_path,
            "gated_layers": gated_layers,
            "use_film": use_film,
            "use_strain": use_strain,
            "epochs": epochs,
            "batch_size": batch_size * gradient_accumulation,
            "learning_rate": learning_rate,
            "ce_weight": ce_weight,
            "metabolic_weight": metabolic_weight,
            "strain_weight": strain_weight,
            "stability_weight": stability_weight,
            "hardware_target_weight": hardware_target_weight,
            "stress_curriculum": stress_curriculum,
            "use_real_sensors": use_real_sensors,
        }
    )

    # Load model
    print("\n[1/6] Loading EmbodiedDeepSeek...")
    model = load_embodied_model(
        base_model_path=base_model_path,
        checkpoint_path=checkpoint_path,
        gated_layers=gated_layers,
        use_film=use_film,
        use_strain=use_strain,
        use_real_sensors=use_real_sensors,
    )

    # Freeze base model
    print("\n[2/6] Freezing base model...")
    model.freeze_base_model()

    # Create datasets
    print("\n[3/6] Loading datasets...")
    train_dataset = EmbodiedDataset(
        data_path=data_path,
        tokenizer=model.tokenizer,
        max_length=max_length,
    )

    val_dataset = None
    if Path(val_path).exists():
        val_dataset = EmbodiedDataset(
            data_path=val_path,
            tokenizer=model.tokenizer,
            max_length=max_length,
        )

    # Dataloaders
    collator = EmbodiedCollator()
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collator,
    )

    val_loader = None
    if val_dataset:
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collator,
        )

    # Optimizer
    print("\n[4/6] Setting up optimizer...")
    trainable_params = model.get_trainable_parameters()
    optimizer = torch.optim.AdamW(trainable_params, lr=learning_rate)

    # Loss function
    loss_fn = EmbodiedLoss(
        ce_weight=ce_weight,
        metabolic_weight=metabolic_weight,
        strain_weight=strain_weight,
        stability_weight=stability_weight,
        hardware_target_weight=hardware_target_weight,
    )

    # Scheduler
    total_steps = len(train_loader) * epochs // gradient_accumulation
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)

    # Metrics logger
    sensor_hub = model.sensor_hub if hasattr(model, 'sensor_hub') else None
    metrics_logger = MetricsLogger(sensor_hub=sensor_hub)

    # Causal validator (for periodic ablation tests)
    causal_validator = None
    if val_loader:
        causal_validator = CausalValidator(
            model=model,
            n_trials=1,  # Quick validation
            use_real_stress=False,  # Simulated for speed
            output_dir=str(output_path / "ablations"),
        )

    # Training loop
    print("\n[5/6] Starting training...")
    model.base_model.train()
    best_val_loss = float("inf")
    global_step = 0

    for epoch in range(epochs):
        print(f"\n{'='*70}")
        print(f"Epoch {epoch + 1}/{epochs}")
        print(f"{'='*70}")

        epoch_losses = []
        optimizer.zero_grad()

        # Stress curriculum
        if stress_curriculum:
            progress = epoch / max(epochs - 1, 1)
            stress_low = initial_stress_range[0] + (final_stress_range[0] - initial_stress_range[0]) * progress
            stress_high = initial_stress_range[1] + (final_stress_range[1] - initial_stress_range[1]) * progress
            print(f"  Stress range: [{stress_low:.2f}, {stress_high:.2f}]")
        else:
            stress_low, stress_high = 0.1, 0.9

        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}")
        for batch_idx, batch in enumerate(pbar):
            # Move to GPU
            input_ids = batch["input_ids"].to("cuda")
            attention_mask = batch["attention_mask"].to("cuda")
            labels = batch["labels"].to("cuda")
            stress = batch["stress"].to("cuda")
            target_lengths = batch["target_lengths"]

            # Apply stress curriculum
            if stress_curriculum:
                stress = stress * (stress_high - stress_low) + stress_low

            # Set stress level (batch mean)
            avg_stress = stress.mean().item()
            model.set_stress_level(avg_stress)

            # Forward
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )

            ce_loss = outputs.loss
            gate_tensors = outputs.gate_tensors
            strain_magnitudes = outputs.strain_magnitudes

            # Compute loss
            loss_output = loss_fn(
                ce_loss=ce_loss,
                gate_tensors=gate_tensors,
                strain_magnitudes=strain_magnitudes,
                stress_level=avg_stress,
                sensor_history=model._sensor_history,
            )

            total_loss = loss_output.total_loss

            # Backward
            (total_loss / gradient_accumulation).backward()
            epoch_losses.append(loss_output.metrics["total_loss"])

            # Optimizer step
            if (batch_idx + 1) % gradient_accumulation == 0:
                torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                # Log metrics
                sensors = model._get_sensors()
                metrics_logger.log_training_step(
                    loss_metrics=loss_output.metrics,
                    model_output=outputs,
                    sensors=sensors,
                    stress_level=avg_stress,
                    lr=scheduler.get_last_lr()[0],
                    step=global_step,
                )

                # Periodic validation
                if val_loader and global_step % val_every_steps == 0:
                    val_ce, high_metrics, low_metrics = validate_embodiment(
                        model, val_loader, loss_fn
                    )
                    metrics_logger.log_validation(
                        val_ce=val_ce,
                        high_stress_metrics=high_metrics,
                        low_stress_metrics=low_metrics,
                        step=global_step,
                    )
                    model.base_model.train()

            # Update progress bar
            pbar.set_postfix({
                "loss": f"{loss_output.metrics['total_loss']:.3f}",
                "ce": f"{loss_output.metrics['ce_loss']:.3f}",
                "gate": f"{loss_output.metrics['avg_gate']:.2f}",
                "strain": f"{loss_output.metrics.get('avg_strain', 0):.3f}",
                "stress": f"{avg_stress:.2f}",
            })

        # End of epoch validation
        if val_loader:
            print("\nRunning epoch validation...")
            val_ce, high_metrics, low_metrics = validate_embodiment(
                model, val_loader, loss_fn
            )

            print(f"\nEpoch {epoch + 1} Results:")
            print(f"  Val CE Loss:        {val_ce:.4f}")
            print(f"  High Stress Words:  {high_metrics['avg_words']:.1f}")
            print(f"  Low Stress Words:   {low_metrics['avg_words']:.1f}")
            print(f"  Word Diff:          {low_metrics['avg_words'] - high_metrics['avg_words']:.1f}")
            print(f"  High Stress Gate:   {high_metrics['avg_gate']:.3f}")
            print(f"  Low Stress Gate:    {low_metrics['avg_gate']:.3f}")

            metrics_logger.log_validation(
                val_ce=val_ce,
                high_stress_metrics=high_metrics,
                low_stress_metrics=low_metrics,
                step=global_step,
            )

            # Epoch summary
            epoch_loss = np.mean(epoch_losses)
            metrics_logger.log_epoch_summary(
                epoch=epoch + 1,
                train_loss=epoch_loss,
                val_loss=val_ce,
                best_val_loss=best_val_loss,
                epoch_metrics={
                    "word_diff": low_metrics['avg_words'] - high_metrics['avg_words'],
                    "high_stress_gate": high_metrics['avg_gate'],
                    "low_stress_gate": low_metrics['avg_gate'],
                },
                step=global_step,
            )

            # Save if best
            if val_ce < best_val_loss:
                best_val_loss = val_ce
                save_checkpoint(model, output_path / "best", epoch, val_ce, gated_layers)
                print(f"  Saved best model (val_ce={val_ce:.4f})")

            model.base_model.train()

        # Periodic ablation test
        if causal_validator and (epoch + 1) % ablation_every_epochs == 0:
            print("\nRunning ablation tests...")
            try:
                report = causal_validator.run_full_validation()
                causal_validator.print_summary(report)
                metrics_logger.log_ablation(report, global_step)
            except Exception as e:
                print(f"  Ablation test failed: {e}")

    # Final save
    print("\n[6/6] Saving final model...")
    save_checkpoint(model, output_path / "final", epochs, val_ce if val_loader else 0, gated_layers)

    wandb.finish()

    print("\n" + "=" * 70)
    print("EMBODIED TRAINING COMPLETE")
    print("=" * 70)
    print(f"Best Val CE: {best_val_loss:.4f}")
    print(f"Output: {output_dir}")

    return model


def save_checkpoint(
    model: EmbodiedDeepSeek,
    save_path: Path,
    epoch: int,
    val_ce: float,
    gated_layers: List[int],
):
    """Save model checkpoint."""
    save_path.mkdir(parents=True, exist_ok=True)

    # Collect embodied state
    state_dict = {}
    for key, block in model.embodied_blocks.items():
        # Gate weights
        for name, param in block.gate.named_parameters():
            state_dict[f"{key}.gate.{name}"] = param.data.cpu()

        # FiLM weights
        if block.film is not None:
            for name, param in block.film.named_parameters():
                state_dict[f"{key}.film.{name}"] = param.data.cpu()

        # Strain weights
        if block.strain is not None:
            for name, param in block.strain.named_parameters():
                state_dict[f"{key}.strain.{name}"] = param.data.cpu()

    torch.save({
        "embodied": state_dict,
        "gated_layers": gated_layers,
        "epoch": epoch,
        "val_ce": val_ce,
    }, save_path / "embodied.pt")


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FEEL z24: Embodied Training")

    # Data
    parser.add_argument("--data", type=str, default="data/ouroboros/ouroboros_train.jsonl")
    parser.add_argument("--val-data", type=str, default="data/ouroboros/ouroboros_val.jsonl")

    # Model
    parser.add_argument("--base-model", type=str, default="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--no-film", action="store_true", help="Disable FiLM")
    parser.add_argument("--no-strain", action="store_true", help="Disable strain")

    # Training
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--lr", type=float, default=5e-4)

    # Loss weights
    parser.add_argument("--ce-weight", type=float, default=1.0)
    parser.add_argument("--metabolic-weight", type=float, default=0.3)
    parser.add_argument("--strain-weight", type=float, default=0.2)
    parser.add_argument("--stability-weight", type=float, default=0.1)
    parser.add_argument("--hardware-weight", type=float, default=0.2)

    # Hardware
    parser.add_argument("--no-real-sensors", action="store_true", help="Use simulated sensors")
    parser.add_argument("--card-id", type=int, default=1)

    # Output
    parser.add_argument("--output", type=str, default="models/embodied_z24")
    parser.add_argument("--run-name", type=str, default=None)

    args = parser.parse_args()

    train_embodied(
        data_path=args.data,
        val_path=args.val_data,
        base_model_path=args.base_model,
        checkpoint_path=args.checkpoint,
        use_film=not args.no_film,
        use_strain=not args.no_strain,
        epochs=args.epochs,
        batch_size=args.batch_size,
        gradient_accumulation=args.grad_accum,
        learning_rate=args.lr,
        ce_weight=args.ce_weight,
        metabolic_weight=args.metabolic_weight,
        strain_weight=args.strain_weight,
        stability_weight=args.stability_weight,
        hardware_target_weight=args.hardware_weight,
        use_real_sensors=not args.no_real_sensors,
        card_id=args.card_id,
        output_dir=args.output,
        run_name=args.run_name,
    )
