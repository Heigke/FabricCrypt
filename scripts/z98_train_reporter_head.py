#!/usr/bin/env python3
"""
Train ReporterHead on Real FEEL Telemetry

This script trains the ReporterHead auxiliary output on actual telemetry
collected from FEEL inference runs. This creates a grounded, honest
body-state reporter that can verbalize hardware state without injecting
noise into the LLM's main output.

Training data:
    - GPU power/temp/util from actuator daemon
    - Latency metrics from inference runs
    - Profile labels (eco/balanced/performance)
    - State labels derived from thresholds

Author: FEEL Research Team
Date: 2026-01-21
"""

import os
import sys
import json
import time
import argparse
import logging
import random
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, asdict
import urllib.request

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# Add project root
script_dir = Path(__file__).parent.absolute()
project_root = script_dir.parent
sys.path.insert(0, str(project_root))

from src.reporter.reporter_head import (
    ReporterHead, ReporterLoss, BodyReport, BodyProprioception,
    body_latent_to_tensor, proprioception_to_tensor,
    ReporterVerbalizerTemplate
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# Data Collection
# ============================================================================

@dataclass
class TelemetrySample:
    """Single telemetry sample for training."""
    # Hardware telemetry (targets)
    power_watts: float
    temp_c: float
    utilization: float

    # Body latent (inputs)
    strain: float
    urgency: float
    debt: float
    margin: float
    stability: float

    # Proprioception (inputs)
    token_entropy: float = 0.0
    top_logit_margin: float = 0.0
    latency_ms: float = 0.0
    queue_depth: int = 0

    # Labels (derived from telemetry)
    strain_label: int = 1  # 0=low, 1=normal, 2=high, 3=critical
    debt_label: int = 0    # 0=none, 1=accumulating, 2=critical
    margin_label: int = 0  # 0=ample, 1=moderate, 2=limited, 3=critical
    stability_label: int = 0  # 0=stable, 1=transitioning, 2=unstable
    mode_label: int = 1    # 0=EXPLORE, 1=BALANCED, 2=RECOVER, 3=CONSERVE, 4=URGENT
    action_label: int = 0  # 0=maintain, 1=reduce, 2=increase, 3=urgent, 4=reset

    # Profile (for reference)
    profile: str = "balanced"


class ActuatorClient:
    """Actuator client for telemetry collection."""

    def __init__(self, host: str, port: int):
        self.base_url = f"http://{host}:{port}"

    def get_telemetry(self) -> Optional[Dict]:
        try:
            with urllib.request.urlopen(f"{self.base_url}/telemetry", timeout=5) as resp:
                return json.loads(resp.read().decode())
        except:
            return None

    def set_profile(self, profile: str) -> bool:
        try:
            data = json.dumps({'profile': profile}).encode()
            req = urllib.request.Request(
                f"{self.base_url}/profile",
                data=data,
                headers={'Content-Type': 'application/json'},
                method='POST'
            )
            with urllib.request.urlopen(req, timeout=10) as resp:
                result = json.loads(resp.read().decode())
            return result.get('success', False)
        except:
            return False


def derive_labels(sample: TelemetrySample) -> TelemetrySample:
    """Derive classification labels from telemetry values."""

    # Strain label (based on power relative to typical range)
    if sample.power_watts < 100:
        sample.strain_label = 0  # low
    elif sample.power_watts < 200:
        sample.strain_label = 1  # normal
    elif sample.power_watts < 280:
        sample.strain_label = 2  # high
    else:
        sample.strain_label = 3  # critical

    # Debt label (based on sustained high power)
    if sample.strain > 0.7:
        sample.debt_label = 2 if sample.debt > 0.5 else 1
    else:
        sample.debt_label = 0

    # Margin label (based on temperature)
    if sample.temp_c < 60:
        sample.margin_label = 0  # ample
    elif sample.temp_c < 75:
        sample.margin_label = 1  # moderate
    elif sample.temp_c < 85:
        sample.margin_label = 2  # limited
    else:
        sample.margin_label = 3  # critical

    # Stability label (based on variance)
    if sample.stability > 0.8:
        sample.stability_label = 0  # stable
    elif sample.stability > 0.5:
        sample.stability_label = 1  # transitioning
    else:
        sample.stability_label = 2  # unstable

    # Mode label (based on profile and state)
    mode_map = {
        'eco': 3,          # CONSERVE
        'balanced': 1,     # BALANCED
        'performance': 0,  # EXPLORE
    }
    sample.mode_label = mode_map.get(sample.profile, 1)

    # Override based on state
    if sample.strain_label >= 3 or sample.margin_label >= 3:
        sample.mode_label = 4  # URGENT

    # Action label
    if sample.strain_label >= 3:
        sample.action_label = 3  # urgent_reduce
    elif sample.strain_label >= 2:
        sample.action_label = 1  # reduce
    elif sample.strain_label == 0 and sample.margin_label == 0:
        sample.action_label = 2  # increase
    else:
        sample.action_label = 0  # maintain

    return sample


def collect_telemetry_samples(
    actuator: ActuatorClient,
    profiles: List[str],
    samples_per_profile: int = 100,
    sample_interval_s: float = 0.1,
) -> List[TelemetrySample]:
    """Collect telemetry samples for training."""

    samples = []
    power_history = []  # For computing variance

    for profile in profiles:
        logger.info(f"Collecting samples for profile: {profile}")

        if not actuator.set_profile(profile):
            logger.warning(f"Failed to set profile {profile}")
            continue

        time.sleep(2)  # Let profile settle

        for i in range(samples_per_profile):
            telemetry = actuator.get_telemetry()

            if not telemetry:
                continue

            power = telemetry.get('power_watts', 0)
            temp = telemetry.get('temperature_c', 50)
            util = telemetry.get('utilization', 0)

            power_history.append(power)
            if len(power_history) > 20:
                power_history.pop(0)

            # Compute body latent from telemetry
            # (In a real system, these would come from the FEEL controller)
            power_norm = min(1.0, power / 300)  # Normalize to typical max
            temp_norm = min(1.0, (temp - 30) / 70)  # Normalize 30-100C

            strain = power_norm * 0.7 + temp_norm * 0.3
            urgency = max(0, (temp - 70) / 30) if temp > 70 else 0
            margin = 1.0 - temp_norm
            stability = 1.0 - (sum((p - power)**2 for p in power_history) / len(power_history) / 1000) if len(power_history) > 1 else 1.0
            stability = max(0, min(1, stability))
            debt = max(0, strain - 0.5) * 2

            sample = TelemetrySample(
                power_watts=power,
                temp_c=temp,
                utilization=util,
                strain=strain,
                urgency=urgency,
                debt=debt,
                margin=margin,
                stability=stability,
                profile=profile,
            )

            # Derive labels
            sample = derive_labels(sample)

            samples.append(sample)
            time.sleep(sample_interval_s)

            if (i + 1) % 50 == 0:
                logger.info(f"  Collected {i + 1}/{samples_per_profile}")

    return samples


def generate_synthetic_samples(n_samples: int = 1000) -> List[TelemetrySample]:
    """Generate synthetic training samples when actuator not available."""

    samples = []

    profiles = ['eco', 'balanced', 'performance']
    profile_power_ranges = {
        'eco': (80, 150),
        'balanced': (150, 250),
        'performance': (220, 300),
    }

    for _ in range(n_samples):
        profile = random.choice(profiles)
        power_range = profile_power_ranges[profile]

        power = random.uniform(*power_range)
        temp = 40 + power / 10 + random.uniform(-5, 10)
        util = random.uniform(50, 100) if power > 100 else random.uniform(10, 60)

        strain = min(1.0, power / 300)
        margin = max(0, 1.0 - (temp - 30) / 70)
        urgency = max(0, (temp - 70) / 30) if temp > 70 else 0
        stability = random.uniform(0.6, 1.0)
        debt = max(0, strain - 0.5) * 2

        sample = TelemetrySample(
            power_watts=power,
            temp_c=temp,
            utilization=util,
            strain=strain,
            urgency=urgency,
            debt=debt,
            margin=margin,
            stability=stability,
            profile=profile,
        )

        sample = derive_labels(sample)
        samples.append(sample)

    return samples


# ============================================================================
# Dataset
# ============================================================================

class TelemetryDataset(Dataset):
    """Dataset for ReporterHead training."""

    def __init__(self, samples: List[TelemetrySample], hidden_size: int = 896):
        self.samples = samples
        self.hidden_size = hidden_size

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]

        # Simulated hidden state (in real use, this comes from LLM)
        # For training, we use noise + encoded body state
        hidden = torch.randn(self.hidden_size)

        # Encode some body state info into hidden state
        # (In real use, the hidden state naturally contains this info)
        hidden[0] = sample.power_watts / 300
        hidden[1] = sample.temp_c / 100
        hidden[2] = sample.utilization / 100
        hidden[3] = sample.strain
        hidden[4] = sample.margin

        # Body latent
        body_latent = torch.tensor([
            sample.strain,
            sample.urgency,
            sample.debt,
            sample.margin,
            sample.stability,
        ], dtype=torch.float32)

        # Proprioception
        proprioception = torch.tensor([
            sample.token_entropy,
            sample.top_logit_margin,
            sample.latency_ms / 1000,  # Normalize
            sample.queue_depth / 10,   # Normalize
            random.uniform(0, 1),      # Position (random for synthetic)
            random.uniform(0.5, 1),    # Confidence
        ], dtype=torch.float32)

        # Targets
        telemetry_targets = torch.tensor([
            sample.power_watts / 300,  # Normalize
            sample.temp_c / 100,
            sample.utilization / 100,
        ], dtype=torch.float32)

        return {
            'hidden_state': hidden,
            'body_latent': body_latent,
            'proprioception': proprioception,
            'telemetry': telemetry_targets,
            'strain_labels': torch.tensor(sample.strain_label, dtype=torch.long),
            'debt_labels': torch.tensor(sample.debt_label, dtype=torch.long),
            'margin_labels': torch.tensor(sample.margin_label, dtype=torch.long),
            'stability_labels': torch.tensor(sample.stability_label, dtype=torch.long),
            'mode_labels': torch.tensor(sample.mode_label, dtype=torch.long),
            'action_labels': torch.tensor(sample.action_label, dtype=torch.long),
        }


# ============================================================================
# Training
# ============================================================================

def train_reporter_head(
    model: ReporterHead,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = 50,
    lr: float = 1e-3,
    device: str = 'cuda',
) -> Dict[str, List[float]]:
    """Train ReporterHead."""

    model = model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    loss_fn = ReporterLoss()

    history = {'train_loss': [], 'val_loss': [], 'mode_acc': []}

    for epoch in range(epochs):
        # Training
        model.train()
        train_losses = []

        for batch in train_loader:
            # Move to device
            batch = {k: v.to(device) for k, v in batch.items()}

            # Forward
            outputs = model(
                batch['hidden_state'],
                batch['body_latent'],
                batch['proprioception'],
            )

            # Loss
            loss, loss_dict = loss_fn(outputs, batch)

            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())

        # Validation
        model.eval()
        val_losses = []
        mode_correct = 0
        mode_total = 0

        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(device) for k, v in batch.items()}

                outputs = model(
                    batch['hidden_state'],
                    batch['body_latent'],
                    batch['proprioception'],
                )

                loss, _ = loss_fn(outputs, batch)
                val_losses.append(loss.item())

                # Mode accuracy
                mode_pred = outputs['mode_logits'].argmax(dim=-1)
                mode_correct += (mode_pred == batch['mode_labels']).sum().item()
                mode_total += len(mode_pred)

        train_loss = sum(train_losses) / len(train_losses)
        val_loss = sum(val_losses) / len(val_losses)
        mode_acc = mode_correct / mode_total

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['mode_acc'].append(mode_acc)

        scheduler.step()

        if (epoch + 1) % 10 == 0:
            logger.info(
                f"Epoch {epoch+1}/{epochs}: "
                f"train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, "
                f"mode_acc={mode_acc:.3f}"
            )

    return history


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Train ReporterHead')
    parser.add_argument('--actuator-host', default='192.168.0.38')
    parser.add_argument('--actuator-port', type=int, default=9877)
    parser.add_argument('--collect', action='store_true', help='Collect real telemetry')
    parser.add_argument('--samples', type=int, default=500, help='Samples per profile')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--output-dir', default='results/z98_reporter_training')
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Using device: {device}")

    # Collect or generate samples
    if args.collect:
        actuator = ActuatorClient(args.actuator_host, args.actuator_port)
        samples = collect_telemetry_samples(
            actuator,
            profiles=['eco', 'balanced', 'performance'],
            samples_per_profile=args.samples,
        )
        # Reset
        actuator.set_profile('balanced')
    else:
        logger.info("Generating synthetic training data...")
        samples = generate_synthetic_samples(args.samples * 3)

    logger.info(f"Total samples: {len(samples)}")

    # Save samples
    samples_path = output_dir / "training_samples.json"
    with open(samples_path, 'w') as f:
        json.dump([asdict(s) for s in samples], f, indent=2)

    # Split
    random.shuffle(samples)
    split_idx = int(len(samples) * 0.8)
    train_samples = samples[:split_idx]
    val_samples = samples[split_idx:]

    # Create datasets
    train_dataset = TelemetryDataset(train_samples)
    val_dataset = TelemetryDataset(val_samples)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)

    # Create model
    model = ReporterHead(hidden_size=896)
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Train
    logger.info("Starting training...")
    history = train_reporter_head(
        model, train_loader, val_loader,
        epochs=args.epochs, lr=args.lr, device=device
    )

    # Save model
    model_path = output_dir / "reporter_head.pt"
    torch.save({
        'model_state_dict': model.state_dict(),
        'history': history,
        'config': {
            'hidden_size': 896,
            'epochs': args.epochs,
            'samples': len(samples),
        }
    }, model_path)
    logger.info(f"Model saved to: {model_path}")

    # Test prediction
    model.eval()
    test_sample = samples[0]
    test_hidden = torch.randn(1, 896)
    test_body = torch.tensor([[
        test_sample.strain, test_sample.urgency,
        test_sample.debt, test_sample.margin, test_sample.stability
    ]])
    test_proprio = torch.randn(1, 6)

    report = model.predict(
        test_hidden.to(device),
        test_body.to(device),
        test_proprio.to(device),
    )

    logger.info("\nSample prediction:")
    logger.info(ReporterVerbalizerTemplate.verbalize(report, 'full'))

    # Save history
    history_path = output_dir / "training_history.json"
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)

    logger.info(f"\nTraining complete. Results saved to: {output_dir}")


if __name__ == '__main__':
    main()
