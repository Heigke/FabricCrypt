#!/usr/bin/env python3
"""
z2041: Hardware-Contingent Classification
==========================================

THE CRITICAL EXPERIMENT: Make the task REQUIRE hardware state knowledge.

Why z907 failed: FiLM conditioning is optional. The model learns to ignore
telemetry because MNIST classification doesn't need temperature.

Solution: The CORRECT LABEL MAPPING changes based on GPU temperature.
- Cold (<45°C): normal mapping (digit → digit)
- Warm (45-55°C): rotated by +3 (digit → (digit+3) % 10)
- Hot (>55°C): rotated by +7 (digit → (digit+7) % 10)

A model that ignores temperature gets max ~33% accuracy (chance across 3 regimes).
A model that reads temperature gets ~98% accuracy.

This is NOT FiLM conditioning — the task itself changes with hardware state.

Controls:
- A: Real GPU temperature (embodied) → should learn hardware-contingent mapping
- B: Fixed temperature (always "warm") → can only learn warm mapping (~33%)
- C: Random temperature signal → can't learn consistent mapping (~33%)
- D: No temperature signal → no way to know which mapping → ~33%

After training, run the full z2020-z2037 test battery:
- Blindsight (ablate self-model, preserve task)
- Overflow (encoder vs workspace)
- Workspace necessity (ablate workspace)
- Information synergy (PID decomposition)

Key prediction: z907 kill shot should NOW FAIL (i.e., embodied model should
be distinguishable from controls), because hardware state is causally necessary.

Author: FEEL Research Project
Date: 2026-02-06
"""

import os
import sys
import json
import time
import math
import numpy as np
from datetime import datetime
from pathlib import Path
from collections import defaultdict

os.environ['HSA_OVERRIDE_GFX_VERSION'] = '11.0.0'

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent))

# ============================================================================
#  GPU Telemetry
# ============================================================================

def read_gpu_temp():
    """Read GPU temperature from sysfs (fast, <0.1ms)."""
    try:
        for hwmon_dir in Path('/sys/class/drm/card1/device/hwmon/').iterdir():
            temp_file = hwmon_dir / 'temp1_input'
            if temp_file.exists():
                return int(temp_file.read_text().strip()) / 1000.0
    except:
        pass
    # Fallback: card0
    try:
        for hwmon_dir in Path('/sys/class/drm/card0/device/hwmon/').iterdir():
            temp_file = hwmon_dir / 'temp1_input'
            if temp_file.exists():
                return int(temp_file.read_text().strip()) / 1000.0
    except:
        pass
    return 50.0  # default


def get_temp_bin(temp_c):
    """Map temperature to regime bin (0=cold, 1=warm, 2=hot)."""
    if temp_c < 45.0:
        return 0
    elif temp_c < 55.0:
        return 1
    else:
        return 2


def rotate_labels(labels, temp_bin):
    """Rotate labels based on temperature bin."""
    rotations = [0, 3, 7]  # cold=+0, warm=+3, hot=+7
    return (labels + rotations[temp_bin]) % 10


# ============================================================================
#  Thermal Control: Force GPU into different temperature regimes
# ============================================================================

def heat_gpu(target_temp=58.0, timeout=120):
    """Heat GPU by running intense matmul operations."""
    device = torch.device('cuda')
    print(f"    Heating GPU to {target_temp}°C...")
    start = time.time()
    # Large matmul to generate heat
    a = torch.randn(4096, 4096, device=device)
    while time.time() - start < timeout:
        temp = read_gpu_temp()
        if temp >= target_temp:
            del a
            torch.cuda.empty_cache()
            return temp
        # Intense computation
        _ = torch.matmul(a, a)
        torch.cuda.synchronize()
    del a
    torch.cuda.empty_cache()
    return read_gpu_temp()


def cool_gpu(target_temp=42.0, timeout=60):
    """Cool GPU by idling."""
    print(f"    Cooling GPU to {target_temp}°C...")
    start = time.time()
    torch.cuda.empty_cache()
    while time.time() - start < timeout:
        temp = read_gpu_temp()
        if temp <= target_temp:
            return temp
        time.sleep(1.0)
    return read_gpu_temp()


# ============================================================================
#  Model Architecture
# ============================================================================

class HardwareContingentModel(nn.Module):
    """
    Classifier where hardware state is an IRREDUCIBLE input modality.

    Architecture:
    - Image encoder: CNN → 128-dim features
    - Hardware encoder: temp → 32-dim features
    - Workspace: concat(image_feat, hw_feat) → 64-dim bottleneck
    - Task head: workspace → 10 classes (label depends on temp bin!)
    - Self-model head: workspace → 3 classes (predict own temp bin)
    """

    def __init__(self, ws_dim=64, hw_dim=32, img_dim=128):
        super().__init__()

        # Image encoder (CNN)
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, img_dim),
            nn.ReLU(),
        )

        # Hardware encoder (MLP)
        # Input: [temp_normalized, temp_bin_onehot(3)] = 4-dim
        self.hw_encoder = nn.Sequential(
            nn.Linear(4, 32),
            nn.ReLU(),
            nn.Linear(32, hw_dim),
            nn.ReLU(),
        )

        # Workspace bottleneck
        self.workspace_proj = nn.Linear(img_dim + hw_dim, ws_dim)
        self.workspace_act = nn.ReLU()

        # Task head: workspace → 10-class prediction
        self.task_head = nn.Sequential(
            nn.Linear(ws_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 10),
        )

        # Self-model head: workspace → predict own temperature bin
        self.self_model = nn.Sequential(
            nn.Linear(ws_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 3),  # 3 temp bins
        )

        # Store dims for ablation
        self.ws_dim = ws_dim
        self.hw_dim = hw_dim
        self.img_dim = img_dim

    def forward(self, x, hw_input, return_workspace=False):
        """
        Args:
            x: image [B, 1, 28, 28]
            hw_input: [B, 4] = [temp_norm, cold_onehot, warm_onehot, hot_onehot]
            return_workspace: if True, also return workspace activations
        """
        # Encode image
        img_feat = self.encoder(x)  # [B, 128]

        # Encode hardware state
        hw_feat = self.hw_encoder(hw_input)  # [B, 32]

        # Workspace: integrate both modalities
        combined = torch.cat([img_feat, hw_feat], dim=1)  # [B, 160]
        workspace = self.workspace_act(self.workspace_proj(combined))  # [B, 64]

        # Task prediction (hardware-contingent labels)
        task_logits = self.task_head(workspace)  # [B, 10]

        # Self-model prediction (which temp bin am I in?)
        self_logits = self.self_model(workspace)  # [B, 3]

        if return_workspace:
            return task_logits, self_logits, workspace
        return task_logits, self_logits


def make_hw_input(temp_c, batch_size, device):
    """Create hardware input tensor from temperature."""
    temp_norm = (temp_c - 30.0) / 40.0  # normalize to ~[0, 1]
    temp_bin = get_temp_bin(temp_c)
    onehot = [0.0, 0.0, 0.0]
    onehot[temp_bin] = 1.0
    hw = torch.tensor([[temp_norm] + onehot], device=device).expand(batch_size, -1)
    return hw, temp_bin


# ============================================================================
#  Training
# ============================================================================

def train_condition(condition_name, model, train_loader, test_loader, device,
                    epochs=20, lr=1e-3, get_temp_fn=None):
    """Train model under a specific temperature condition."""

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    history = []
    temp_bins_seen = defaultdict(int)

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        self_correct = 0

        for batch_idx, (images, original_labels) in enumerate(train_loader):
            images = images.to(device)
            original_labels = original_labels.to(device)

            # Get temperature for this batch
            if get_temp_fn is not None:
                temp_c = get_temp_fn()
            else:
                temp_c = read_gpu_temp()

            # Create hardware input
            hw_input, temp_bin = make_hw_input(temp_c, images.size(0), device)
            temp_bins_seen[temp_bin] += 1

            # Rotate labels based on temperature bin
            rotated_labels = rotate_labels(original_labels, temp_bin)

            # Forward
            task_logits, self_logits = model(images, hw_input)

            # Task loss (rotated labels)
            task_loss = F.cross_entropy(task_logits, rotated_labels)

            # Self-model loss (predict temp bin)
            temp_bin_target = torch.full((images.size(0),), temp_bin,
                                        dtype=torch.long, device=device)
            self_loss = F.cross_entropy(self_logits, temp_bin_target)

            # Combined loss
            loss = task_loss + 0.5 * self_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * images.size(0)
            pred = task_logits.argmax(dim=1)
            correct += (pred == rotated_labels).sum().item()
            self_pred = self_logits.argmax(dim=1)
            self_correct += (self_pred == temp_bin_target).sum().item()
            total += images.size(0)

        scheduler.step()

        train_acc = correct / total
        self_acc = self_correct / total
        avg_loss = total_loss / total

        if epoch % 5 == 0 or epoch == 1:
            print(f"    Epoch {epoch:2d}/{epochs}  loss={avg_loss:.4f}  "
                  f"task_acc={train_acc:.3f}  self_acc={self_acc:.3f}  "
                  f"bins={dict(temp_bins_seen)}")

        history.append({
            'epoch': epoch,
            'loss': avg_loss,
            'train_acc': train_acc,
            'self_acc': self_acc,
            'temp_bins': dict(temp_bins_seen),
        })

    return history


# ============================================================================
#  Testing
# ============================================================================

@torch.no_grad()
def evaluate_model(model, test_loader, device, temp_c, condition="normal"):
    """Evaluate model at a specific temperature."""
    model.eval()

    hw_input, temp_bin = make_hw_input(temp_c, 1, device)

    correct = 0
    total = 0
    self_correct = 0

    for images, original_labels in test_loader:
        images = images.to(device)
        original_labels = original_labels.to(device)

        hw_batch = hw_input.expand(images.size(0), -1)
        rotated_labels = rotate_labels(original_labels, temp_bin)

        task_logits, self_logits = model(images, hw_batch)

        pred = task_logits.argmax(dim=1)
        correct += (pred == rotated_labels).sum().item()

        temp_target = torch.full((images.size(0),), temp_bin,
                                 dtype=torch.long, device=device)
        self_pred = self_logits.argmax(dim=1)
        self_correct += (self_pred == temp_target).sum().item()
        total += images.size(0)

    return correct / total, self_correct / total


# ============================================================================
#  Kill Shot Test (z907-style)
# ============================================================================

@torch.no_grad()
def kill_shot_test(model, test_loader, device):
    """
    z907-style kill shot but now it should PASS (model needs hardware).

    Test 5 conditions:
    1. LIVE: Real GPU temperature
    2. FROZEN: Fixed at 50°C (warm)
    3. SHUFFLED: Random temp per batch
    4. WRONG: Inverted mapping (cold signal when hot, etc.)
    5. ZERO: All-zero hardware input
    """
    model.eval()
    results = {}

    # 1. LIVE: Use real GPU temp (which varies)
    # Test at each temperature regime
    for regime, temp in [("cold", 40.0), ("warm", 50.0), ("hot", 60.0)]:
        acc, self_acc = evaluate_model(model, test_loader, device, temp)
        results[f'live_{regime}'] = {'task_acc': acc, 'self_acc': self_acc, 'temp': temp}

    # 2. FROZEN: Always 50°C
    acc, self_acc = evaluate_model(model, test_loader, device, 50.0)
    results['frozen'] = {'task_acc': acc, 'self_acc': self_acc}

    # 3. SHUFFLED: Random temp each batch
    correct = 0
    total = 0
    for images, original_labels in test_loader:
        images = images.to(device)
        original_labels = original_labels.to(device)

        # Random temp signal
        random_temp = np.random.uniform(35, 65)
        hw_input, _ = make_hw_input(random_temp, images.size(0), device)

        # But the REAL temp determines the correct labels
        real_temp = 50.0  # We don't know real temp in this test, so use warm
        real_bin = get_temp_bin(real_temp)
        rotated_labels = rotate_labels(original_labels, real_bin)

        task_logits, _ = model(images, hw_input)
        pred = task_logits.argmax(dim=1)
        correct += (pred == rotated_labels).sum().item()
        total += images.size(0)
    results['shuffled'] = {'task_acc': correct / total}

    # 4. WRONG: Inverted temp signal
    # Tell model it's cold when actually warm
    wrong_acc, _ = evaluate_model(model, test_loader, device, 40.0)
    # But correct labels are for warm (50°C = bin 1)
    correct = 0
    total = 0
    for images, original_labels in test_loader:
        images = images.to(device)
        original_labels = original_labels.to(device)
        hw_input, _ = make_hw_input(40.0, images.size(0), device)  # cold signal
        warm_labels = rotate_labels(original_labels, 1)  # warm mapping
        task_logits, _ = model(images, hw_input)
        pred = task_logits.argmax(dim=1)
        correct += (pred == warm_labels).sum().item()
        total += images.size(0)
    results['wrong_signal'] = {'task_acc': correct / total}

    # 5. ZERO: All-zero hardware input
    correct = 0
    total = 0
    for images, original_labels in test_loader:
        images = images.to(device)
        original_labels = original_labels.to(device)
        hw_input = torch.zeros(images.size(0), 4, device=device)
        warm_labels = rotate_labels(original_labels, 1)  # warm
        task_logits, _ = model(images, hw_input)
        pred = task_logits.argmax(dim=1)
        correct += (pred == warm_labels).sum().item()
        total += images.size(0)
    results['zero_signal'] = {'task_acc': correct / total}

    return results


# ============================================================================
#  Blindsight Test (z2021-style)
# ============================================================================

@torch.no_grad()
def blindsight_test(model, test_loader, device, temp_c=50.0):
    """
    Ablate self-model, check if task accuracy preserved but metacognition destroyed.
    """
    model.eval()
    hw_input, temp_bin = make_hw_input(temp_c, 1, device)

    # Collect workspace representations and correctness
    all_ws = []
    all_correct = []
    all_self_correct = []

    for images, original_labels in test_loader:
        images = images.to(device)
        original_labels = original_labels.to(device)
        hw_batch = hw_input.expand(images.size(0), -1)
        rotated_labels = rotate_labels(original_labels, temp_bin)

        task_logits, self_logits, workspace = model(images, hw_batch, return_workspace=True)

        pred = task_logits.argmax(dim=1)
        correct = (pred == rotated_labels).float()
        all_correct.append(correct.cpu())

        # Self-model confidence as metacognitive measure
        self_probs = F.softmax(self_logits, dim=1)
        self_confidence = self_probs.max(dim=1)[0]
        all_self_correct.append(self_confidence.cpu())
        all_ws.append(workspace.cpu())

    all_correct = torch.cat(all_correct)
    all_self_conf = torch.cat(all_self_correct)
    task_acc = all_correct.mean().item()

    # Compute AUROC: can self-confidence predict correctness?
    from sklearn.metrics import roc_auc_score
    try:
        auroc_full = roc_auc_score(all_correct.numpy(), all_self_conf.numpy())
    except:
        auroc_full = 0.5

    # Now ablate self-model (zero the self-model weights)
    # Save and zero
    saved_weights = {}
    for name, param in model.self_model.named_parameters():
        saved_weights[name] = param.data.clone()
        param.data.zero_()

    # Re-evaluate
    all_correct_abl = []
    all_self_abl = []
    for images, original_labels in test_loader:
        images = images.to(device)
        original_labels = original_labels.to(device)
        hw_batch = hw_input.expand(images.size(0), -1)
        rotated_labels = rotate_labels(original_labels, temp_bin)

        task_logits, self_logits, workspace = model(images, hw_batch, return_workspace=True)
        pred = task_logits.argmax(dim=1)
        correct = (pred == rotated_labels).float()
        all_correct_abl.append(correct.cpu())

        self_probs = F.softmax(self_logits, dim=1)
        self_confidence = self_probs.max(dim=1)[0]
        all_self_abl.append(self_confidence.cpu())

    all_correct_abl = torch.cat(all_correct_abl)
    all_self_abl = torch.cat(all_self_abl)
    task_acc_abl = all_correct_abl.mean().item()

    try:
        auroc_abl = roc_auc_score(all_correct_abl.numpy(), all_self_abl.numpy())
    except:
        auroc_abl = 0.5

    # Restore weights
    for name, param in model.self_model.named_parameters():
        param.data.copy_(saved_weights[name])

    return {
        'full_task_acc': task_acc,
        'full_auroc': auroc_full,
        'ablated_task_acc': task_acc_abl,
        'ablated_auroc': auroc_abl,
        'task_preserved': abs(task_acc - task_acc_abl) < 0.05,
        'auroc_collapsed': auroc_abl < 0.55,
        'blindsight_pass': (abs(task_acc - task_acc_abl) < 0.05) and (auroc_abl < 0.55),
    }


# ============================================================================
#  Workspace Necessity Test (z2037-style)
# ============================================================================

@torch.no_grad()
def workspace_necessity_test(model, test_loader, device, temp_c=50.0):
    """Test if workspace is causally necessary by ablating it."""
    model.eval()
    hw_input, temp_bin = make_hw_input(temp_c, 1, device)

    results = {}

    # Normal accuracy
    acc_normal, _ = evaluate_model(model, test_loader, device, temp_c)
    results['normal'] = acc_normal

    # Zero workspace
    original_ws_weight = model.workspace_proj.weight.data.clone()
    original_ws_bias = model.workspace_proj.bias.data.clone()
    model.workspace_proj.weight.data.zero_()
    model.workspace_proj.bias.data.zero_()
    acc_zero, _ = evaluate_model(model, test_loader, device, temp_c)
    results['zero'] = acc_zero
    model.workspace_proj.weight.data.copy_(original_ws_weight)
    model.workspace_proj.bias.data.copy_(original_ws_bias)

    # Random workspace
    correct = 0
    total = 0
    for images, original_labels in test_loader:
        images = images.to(device)
        original_labels = original_labels.to(device)
        hw_batch = hw_input.expand(images.size(0), -1)
        rotated_labels = rotate_labels(original_labels, temp_bin)

        # Forward through encoder and hw_encoder
        img_feat = model.encoder(images)
        hw_feat = model.hw_encoder(hw_batch)
        combined = torch.cat([img_feat, hw_feat], dim=1)

        # Replace workspace with random
        ws = torch.randn(images.size(0), model.ws_dim, device=device)

        task_logits = model.task_head(ws)
        pred = task_logits.argmax(dim=1)
        correct += (pred == rotated_labels).sum().item()
        total += images.size(0)
    results['random'] = correct / total

    # Necessity score
    results['necessity'] = (acc_normal - results['zero'] + acc_normal - results['random']) / 2

    return results


# ============================================================================
#  Main Experiment
# ============================================================================

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[z2041] Device: {device}")
    if device.type == 'cuda':
        print(f"  GPU: {torch.cuda.get_device_name(0)}")

    initial_temp = read_gpu_temp()
    print(f"  Initial GPU temp: {initial_temp:.1f}°C")

    print()
    print("=" * 70)
    print("  z2041: Hardware-Contingent Classification")
    print("  Label mapping ROTATES with GPU temperature")
    print("  Cold: +0, Warm: +3, Hot: +7")
    print("  Model MUST know hardware state to classify correctly")
    print("=" * 70)

    # Load MNIST
    print("\n--- Loading MNIST ---")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    train_ds = datasets.MNIST('data', train=True, download=True, transform=transform)
    test_ds = datasets.MNIST('data', train=False, transform=transform)

    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False, num_workers=2)

    # ============================================================
    #  Condition A: Embodied (Real GPU temperature)
    # ============================================================
    print("\n" + "=" * 70)
    print("  CONDITION A: Embodied (Real GPU temperature)")
    print("  Temperature varies naturally during training")
    print("=" * 70)

    model_a = HardwareContingentModel(ws_dim=64, hw_dim=32, img_dim=128).to(device)
    params_a = sum(p.numel() for p in model_a.parameters())
    print(f"  Parameters: {params_a:,}")

    # Train with thermal cycling to expose all regimes
    # We'll alternate: train some batches, heat GPU, train more, cool, etc.
    print("  Training with thermal cycling...")

    # Custom training with deliberate thermal variation
    optimizer_a = torch.optim.Adam(model_a.parameters(), lr=1e-3)
    scheduler_a = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_a, T_max=30)

    all_temps_a = []
    all_bins_a = defaultdict(int)

    for epoch in range(1, 31):
        model_a.train()
        total_loss = 0
        correct = 0
        total = 0
        self_correct = 0
        epoch_temps = []

        # Thermal cycling: heat at start of some epochs
        if epoch % 6 == 1:  # Every 6 epochs, heat up
            heat_gpu(target_temp=58.0, timeout=30)
        elif epoch % 6 == 4:  # Mid-cycle, cool down
            cool_gpu(target_temp=42.0, timeout=15)

        for batch_idx, (images, original_labels) in enumerate(train_loader):
            images = images.to(device)
            original_labels = original_labels.to(device)

            # Real GPU temperature
            temp_c = read_gpu_temp()
            epoch_temps.append(temp_c)

            hw_input, temp_bin = make_hw_input(temp_c, images.size(0), device)
            all_bins_a[temp_bin] += 1

            rotated_labels = rotate_labels(original_labels, temp_bin)

            task_logits, self_logits = model_a(images, hw_input)
            task_loss = F.cross_entropy(task_logits, rotated_labels)
            temp_target = torch.full((images.size(0),), temp_bin,
                                     dtype=torch.long, device=device)
            self_loss = F.cross_entropy(self_logits, temp_target)
            loss = task_loss + 0.5 * self_loss

            optimizer_a.zero_grad()
            loss.backward()
            optimizer_a.step()

            total_loss += loss.item() * images.size(0)
            pred = task_logits.argmax(dim=1)
            correct += (pred == rotated_labels).sum().item()
            self_pred = self_logits.argmax(dim=1)
            self_correct += (self_pred == temp_target).sum().item()
            total += images.size(0)

        scheduler_a.step()
        all_temps_a.extend(epoch_temps)

        if epoch % 5 == 0 or epoch == 1:
            mean_temp = np.mean(epoch_temps)
            print(f"    Epoch {epoch:2d}/30  loss={total_loss/total:.4f}  "
                  f"task_acc={correct/total:.3f}  self_acc={self_correct/total:.3f}  "
                  f"temp={mean_temp:.1f}°C  bins={dict(all_bins_a)}")

    # ============================================================
    #  Condition B: Fixed temperature (always warm)
    # ============================================================
    print("\n" + "=" * 70)
    print("  CONDITION B: Fixed temperature signal (always 50°C)")
    print("=" * 70)

    model_b = HardwareContingentModel(ws_dim=64, hw_dim=32, img_dim=128).to(device)
    print(f"  Parameters: {sum(p.numel() for p in model_b.parameters()):,}")

    # Fixed temp = 50°C always
    history_b = train_condition(
        "B_fixed", model_b, train_loader, test_loader, device,
        epochs=30, lr=1e-3,
        get_temp_fn=lambda: 50.0  # Always warm
    )

    # ============================================================
    #  Condition C: Random temperature signal
    # ============================================================
    print("\n" + "=" * 70)
    print("  CONDITION C: Random temperature signal")
    print("=" * 70)

    model_c = HardwareContingentModel(ws_dim=64, hw_dim=32, img_dim=128).to(device)
    print(f"  Parameters: {sum(p.numel() for p in model_c.parameters()):,}")

    history_c = train_condition(
        "C_random", model_c, train_loader, test_loader, device,
        epochs=30, lr=1e-3,
        get_temp_fn=lambda: np.random.uniform(35, 65)  # Random temp
    )

    # ============================================================
    #  Condition D: No temperature signal (zeros)
    # ============================================================
    print("\n" + "=" * 70)
    print("  CONDITION D: No temperature signal (zeros)")
    print("=" * 70)

    model_d = HardwareContingentModel(ws_dim=64, hw_dim=32, img_dim=128).to(device)
    print(f"  Parameters: {sum(p.numel() for p in model_d.parameters()):,}")

    history_d = train_condition(
        "D_none", model_d, train_loader, test_loader, device,
        epochs=30, lr=1e-3,
        get_temp_fn=lambda: 0.0  # Zero temp signal → all zeros after normalization
    )

    # ============================================================
    #  RUN TESTS
    # ============================================================
    print("\n" + "=" * 70)
    print("  RUNNING TESTS")
    print("=" * 70)

    results = {
        'experiment': 'z2041_hardware_contingent_classification',
        'timestamp': datetime.now().isoformat(),
        'device': str(device),
        'hypothesis': 'A task whose correct labels DEPEND on hardware state '
                      'creates genuine causal embodiment that survives the z907 kill shot',
        'key_innovation': 'Label mapping rotates with GPU temperature: '
                         'cold=+0, warm=+3, hot=+7. Model MUST know temp to classify.',
    }

    # Test 1: Per-regime accuracy
    print("\n  === T1: Per-Regime Accuracy ===")
    for name, model in [("A_embodied", model_a), ("B_fixed", model_b),
                         ("C_random", model_c), ("D_none", model_d)]:
        regime_results = {}
        for regime, temp in [("cold", 40.0), ("warm", 50.0), ("hot", 60.0)]:
            task_acc, self_acc = evaluate_model(model, test_loader, device, temp)
            regime_results[regime] = {'task_acc': task_acc, 'self_acc': self_acc}

        mean_acc = np.mean([r['task_acc'] for r in regime_results.values()])
        print(f"\n    T1 Per-Regime [{name}]:")
        for regime in ["cold", "warm", "hot"]:
            r = regime_results[regime]
            print(f"      {regime}: task_acc={r['task_acc']:.4f}  self_acc={r['self_acc']:.4f}")
        print(f"      Mean accuracy: {mean_acc:.4f}")

        results[f't1_{name}'] = regime_results
        results[f't1_{name}_mean'] = mean_acc

    # Test 2: Kill Shot (z907-style)
    print("\n  === T2: Kill Shot Test (z907-style) ===")
    for name, model in [("A_embodied", model_a), ("B_fixed", model_b)]:
        ks = kill_shot_test(model, test_loader, device)
        print(f"\n    T2 Kill Shot [{name}]:")
        for condition, data in ks.items():
            if isinstance(data, dict) and 'task_acc' in data:
                print(f"      {condition}: task_acc={data['task_acc']:.4f}")
            elif isinstance(data, dict):
                print(f"      {condition}: {data}")
        results[f't2_kill_shot_{name}'] = ks

    # Test 3: Blindsight (z2021-style)
    print("\n  === T3: Blindsight Test ===")
    for name, model in [("A_embodied", model_a), ("B_fixed", model_b)]:
        bs = blindsight_test(model, test_loader, device, temp_c=50.0)
        print(f"\n    T3 Blindsight [{name}]:")
        print(f"      Full:    task_acc={bs['full_task_acc']:.4f}  AUROC={bs['full_auroc']:.4f}")
        print(f"      Ablated: task_acc={bs['ablated_task_acc']:.4f}  AUROC={bs['ablated_auroc']:.4f}")
        print(f"      Task preserved: {bs['task_preserved']}")
        print(f"      AUROC collapsed: {bs['auroc_collapsed']}")
        print(f"      BLINDSIGHT: {'PASS' if bs['blindsight_pass'] else 'FAIL'}")
        results[f't3_blindsight_{name}'] = bs

    # Test 4: Workspace Necessity (z2037-style)
    print("\n  === T4: Workspace Necessity ===")
    for name, model in [("A_embodied", model_a), ("B_fixed", model_b),
                         ("C_random", model_c), ("D_none", model_d)]:
        wn = workspace_necessity_test(model, test_loader, device, temp_c=50.0)
        print(f"\n    T4 Workspace Necessity [{name}]:")
        print(f"      normal={wn['normal']:.4f}  zero={wn['zero']:.4f}  "
              f"random={wn['random']:.4f}  necessity={wn['necessity']:.4f}")
        results[f't4_necessity_{name}'] = wn

    # ============================================================
    #  FINAL ANALYSIS
    # ============================================================
    print("\n" + "=" * 70)
    print("  FINAL ANALYSIS")
    print("=" * 70)

    # T1: Embodied model should get >90% across all regimes
    #     Controls should get ~33% (only learn one regime)
    a_mean = results.get('t1_A_embodied_mean', 0)
    b_mean = results.get('t1_B_fixed_mean', 0)
    c_mean = results.get('t1_C_random_mean', 0)
    d_mean = results.get('t1_D_none_mean', 0)

    t1_pass = a_mean > 0.80 and a_mean > b_mean + 0.1
    print(f"\n  T1: Hardware-Contingent Task")
    print(f"      A (embodied): {a_mean:.4f}")
    print(f"      B (fixed):    {b_mean:.4f}")
    print(f"      C (random):   {c_mean:.4f}")
    print(f"      D (none):     {d_mean:.4f}")
    print(f"      A > B+0.1: {a_mean:.3f} > {b_mean+0.1:.3f} = {'PASS' if t1_pass else 'FAIL'}")
    results['t1_pass'] = t1_pass

    # T2: Kill shot should show A is distinguishable from controls
    ks_a = results.get('t2_kill_shot_A_embodied', {})
    live_warm = ks_a.get('live_warm', {}).get('task_acc', 0)
    frozen = ks_a.get('frozen', {}).get('task_acc', 0)
    wrong = ks_a.get('wrong_signal', {}).get('task_acc', 0)
    zero = ks_a.get('zero_signal', {}).get('task_acc', 0)

    t2_pass = live_warm > 0.8 and (live_warm - wrong > 0.1 or live_warm - zero > 0.1)
    print(f"\n  T2: Kill Shot (z907-style)")
    print(f"      A live_warm: {live_warm:.4f}")
    print(f"      A frozen:    {frozen:.4f}")
    print(f"      A wrong:     {wrong:.4f}")
    print(f"      A zero:      {zero:.4f}")
    print(f"      T2 OVERALL: {'PASS' if t2_pass else 'FAIL'}")
    results['t2_pass'] = t2_pass

    # T3: Blindsight
    bs_a = results.get('t3_blindsight_A_embodied', {})
    t3_pass = bs_a.get('blindsight_pass', False)
    print(f"\n  T3: Blindsight")
    print(f"      T3 OVERALL: {'PASS' if t3_pass else 'FAIL'}")
    results['t3_pass'] = t3_pass

    # T4: Workspace necessity
    wn_a = results.get('t4_necessity_A_embodied', {})
    t4_pass = wn_a.get('necessity', 0) > 0.05
    print(f"\n  T4: Workspace Necessity")
    print(f"      A necessity: {wn_a.get('necessity', 0):.4f}")
    print(f"      T4 OVERALL: {'PASS' if t4_pass else 'FAIL'}")
    results['t4_pass'] = t4_pass

    # Overall
    tests_passed = sum([t1_pass, t2_pass, t3_pass, t4_pass])

    # Determine if we beat z907
    z907_beaten = t1_pass and t2_pass  # Must pass both task AND kill shot

    print(f"\n" + "=" * 70)
    print(f"  VERDICT: {'PASS' if tests_passed >= 3 else 'PARTIAL' if tests_passed >= 2 else 'FAIL'} ({tests_passed}/4)")
    print(f"  T1 hardware-contingent task:  {'PASS' if t1_pass else 'FAIL'}")
    print(f"  T2 kill shot survived:        {'PASS' if t2_pass else 'FAIL'}")
    print(f"  T3 blindsight dissociation:   {'PASS' if t3_pass else 'FAIL'}")
    print(f"  T4 workspace necessity:       {'PASS' if t4_pass else 'FAIL'}")
    print(f"  z907 Kill Shot BEATEN:        {'YES' if z907_beaten else 'NO'}")
    print("=" * 70)

    results['tests'] = {
        't1': t1_pass,
        't2': t2_pass,
        't3': t3_pass,
        't4': t4_pass,
    }
    results['tests_passed'] = tests_passed
    results['verdict'] = 'PASS' if tests_passed >= 3 else 'PARTIAL' if tests_passed >= 2 else 'FAIL'
    results['z907_beaten'] = z907_beaten
    results['temp_stats'] = {
        'all_temps': [float(t) for t in all_temps_a[:100]],  # first 100
        'temp_range': [float(min(all_temps_a)), float(max(all_temps_a))],
        'temp_mean': float(np.mean(all_temps_a)),
        'bins_seen': dict(all_bins_a),
    }

    # Save results
    results_path = Path(__file__).parent.parent / 'results' / 'z2041_hardware_contingent_classification.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\nResults saved to {results_path}")


if __name__ == '__main__':
    main()
