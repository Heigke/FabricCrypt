#!/usr/bin/env python3
"""
z2028: CIFAR-10 Synthetic Blindsight (Harder Task Validation)

Repeats z2021's strongest result (4/4 PASS) on CIFAR-10 instead of MNIST.
If the blindsight dissociation holds on a harder task, it's not MNIST-specific.

Architecture: ResNet-style encoder (harder task needs deeper network)
Same protocol: train task + self-model, ablate self-model, check dissociation.

NO consciousness losses. Self-model learns to predict own correctness.
"""

import sys
import os
import json
import time
import argparse
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).parent.parent))
try:
    from src.telemetry.sysfs_hwmon import SysfsHwmonTelemetry
    TELEMETRY_AVAILABLE = True
except ImportError:
    TELEMETRY_AVAILABLE = False


class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        h = F.relu(self.bn1(self.conv1(x)))
        h = self.bn2(self.conv2(h))
        return F.relu(h + residual)


class CIFAREncoder(nn.Module):
    def __init__(self, hidden_dim=256):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
        )
        self.block1 = ResBlock(128)
        self.pool1 = nn.Sequential(
            nn.Conv2d(128, 256, 3, stride=2, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
        )
        self.block2 = ResBlock(256)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(256, hidden_dim)

    def forward(self, x):
        h = self.stem(x)
        h = self.block1(h)
        h = self.pool1(h)
        h = self.block2(h)
        h = self.avgpool(h).view(h.size(0), -1)
        return F.relu(self.fc(h))


class TaskHead(nn.Module):
    def __init__(self, hidden_dim=256, num_classes=10):
        super().__init__()
        self.fc = nn.Sequential(nn.Linear(hidden_dim, 128), nn.ReLU(), nn.Linear(128, num_classes))

    def forward(self, h):
        return self.fc(h)


class SelfModelHead(nn.Module):
    def __init__(self, hidden_dim=256):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, h):
        return torch.sigmoid(self.fc(h)).squeeze(-1)


class CIFARBlindsightModel(nn.Module):
    def __init__(self, hidden_dim=256):
        super().__init__()
        self.encoder = CIFAREncoder(hidden_dim)
        self.task_head = TaskHead(hidden_dim)
        self.self_model = SelfModelHead(hidden_dim)

    def forward(self, x):
        h = self.encoder(x)
        return {'logits': self.task_head(h), 'confidence': self.self_model(h), 'hidden': h}


def compute_type2_auroc(confidences, correctness):
    if len(np.unique(correctness)) < 2:
        return 0.5
    try:
        return roc_auc_score(correctness, confidences)
    except ValueError:
        return 0.5


def train_epoch(model, loader, optimizer, device, lambda_self=0.5):
    model.train()
    total_loss, correct, total = 0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        out = model(images)
        loss_task = F.cross_entropy(out['logits'], labels)
        with torch.no_grad():
            preds = out['logits'].argmax(1)
            is_correct = (preds == labels).float()
        loss_self = F.binary_cross_entropy(out['confidence'], is_correct)
        loss = loss_task + lambda_self * loss_self
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * images.size(0)
        correct += (preds == labels).sum().item()
        total += images.size(0)
    return {'loss': total_loss / total, 'acc': correct / total}


@torch.no_grad()
def evaluate_full(model, loader, device):
    model.eval()
    all_conf, all_corr = [], []
    correct, total = 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        out = model(images)
        preds = out['logits'].argmax(1)
        c = (preds == labels)
        all_conf.extend(out['confidence'].cpu().numpy())
        all_corr.extend(c.cpu().numpy().astype(float))
        correct += c.sum().item()
        total += images.size(0)
    conf = np.array(all_conf)
    corr = np.array(all_corr)
    return {
        'task_acc': correct / total,
        'type2_auroc': compute_type2_auroc(conf, corr),
        'mean_conf': float(conf.mean()),
        'conf_correct': float(conf[corr == 1].mean()) if corr.sum() > 0 else 0,
        'conf_wrong': float(conf[corr == 0].mean()) if (1 - corr).sum() > 0 else 0,
    }


def ablate_self_model(model):
    with torch.no_grad():
        for p in model.self_model.parameters():
            p.zero_()


def ablate_encoder(model):
    with torch.no_grad():
        for p in model.encoder.parameters():
            nn.init.normal_(p, 0, 0.01)


def scramble_eval(model, loader, device):
    model.eval()
    all_conf, all_corr = [], []
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            h = model.encoder(images)
            logits = model.task_head(h)
            preds = logits.argmax(1)
            c = (preds == labels)
            scrambled_h = h[torch.randperm(h.size(0))]
            conf = model.self_model(scrambled_h)
            all_conf.extend(conf.cpu().numpy())
            all_corr.extend(c.cpu().numpy().astype(float))
            correct += c.sum().item()
            total += images.size(0)
    conf = np.array(all_conf)
    corr = np.array(all_corr)
    return {'task_acc': correct / total, 'type2_auroc': compute_type2_auroc(conf, corr)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch-size', type=int, default=128)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[z2028] Device: {device}")

    from torchvision import datasets, transforms
    data_dir = Path(__file__).parent.parent / 'data'
    data_dir.mkdir(exist_ok=True)

    # CIFAR-10 with augmentation
    train_tf = transforms.Compose([
        transforms.RandomHorizontalFlip(), transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.262)),
    ])
    test_tf = transforms.Compose([
        transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.262)),
    ])

    print("Loading CIFAR-10...")
    train_data = datasets.CIFAR10(str(data_dir), train=True, download=True, transform=train_tf)
    test_data = datasets.CIFAR10(str(data_dir), train=False, download=True, transform=test_tf)
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_data, batch_size=256, shuffle=False)

    print(f"\n{'='*70}")
    print(f"  z2028: CIFAR-10 Synthetic Blindsight")
    print(f"  Harder task validation of z2021 (MNIST 4/4 PASS)")
    print(f"  If dissociation holds on CIFAR-10, it's not MNIST-specific")
    print(f"  NO consciousness losses — self-model learns task success")
    print(f"{'='*70}")

    model = CIFARBlindsightModel(hidden_dim=256).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {n_params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    for ep in range(1, args.epochs + 1):
        t0 = time.time()
        stats = train_epoch(model, train_loader, optimizer, device)
        scheduler.step()
        elapsed = time.time() - t0
        if ep % 5 == 0 or ep == 1 or ep == args.epochs:
            print(f"  Epoch {ep:2d}/{args.epochs}  loss={stats['loss']:.4f}  acc={stats['acc']:.3f}  ({elapsed:.1f}s)")

    checkpoint = {k: v.clone() for k, v in model.state_dict().items()}

    # Test 1: Full model
    print(f"\n  [1/4] Full model...")
    full = evaluate_full(model, test_loader, device)
    print(f"    Acc={full['task_acc']:.4f}  AUROC={full['type2_auroc']:.4f}")

    # Test 2: Ablate self-model
    print(f"  [2/4] Self-model ablated...")
    model.load_state_dict(checkpoint)
    ablate_self_model(model)
    ablated = evaluate_full(model, test_loader, device)
    print(f"    Acc={ablated['task_acc']:.4f}  AUROC={ablated['type2_auroc']:.4f}")

    # Test 3: Ablate encoder
    print(f"  [3/4] Encoder ablated...")
    model.load_state_dict(checkpoint)
    ablate_encoder(model)
    enc_abl = evaluate_full(model, test_loader, device)
    print(f"    Acc={enc_abl['task_acc']:.4f}  AUROC={enc_abl['type2_auroc']:.4f}")

    # Test 4: Scramble
    print(f"  [4/4] Scrambled...")
    model.load_state_dict(checkpoint)
    scrambled = scramble_eval(model, test_loader, device)
    print(f"    Acc={scrambled['task_acc']:.4f}  AUROC={scrambled['type2_auroc']:.4f}")

    # Analysis
    print(f"\n{'='*70}")
    print(f"  FINAL ANALYSIS: CIFAR-10 Blindsight")
    print(f"{'='*70}")

    acc_preserved = abs(full['task_acc'] - ablated['task_acc']) < 0.02
    auroc_collapsed = ablated['type2_auroc'] < 0.55
    enc_both = enc_abl['task_acc'] < full['task_acc'] - 0.1 and enc_abl['type2_auroc'] < full['type2_auroc'] - 0.1
    scr_collapsed = scrambled['type2_auroc'] < 0.55

    t1 = acc_preserved and auroc_collapsed
    t2 = enc_both
    t3 = scr_collapsed
    t4 = full['type2_auroc'] > 0.6

    print(f"  T1: Blindsight dissociation: {'PASS' if t1 else 'FAIL'} (acc {full['task_acc']:.3f}→{ablated['task_acc']:.3f}, AUROC {full['type2_auroc']:.3f}→{ablated['type2_auroc']:.3f})")
    print(f"  T2: Encoder ablation both:   {'PASS' if t2 else 'FAIL'}")
    print(f"  T3: Scramble collapses:       {'PASS' if t3 else 'FAIL'} (AUROC={scrambled['type2_auroc']:.3f})")
    print(f"  T4: Baseline metacognition:   {'PASS' if t4 else 'FAIL'} (AUROC={full['type2_auroc']:.3f})")

    n_pass = sum([t1, t2, t3, t4])
    verdict = ["NO_DISSOCIATION", "WEAK", "PARTIAL", "MOSTLY_CONFIRMED", "CIFAR_BLINDSIGHT_CONFIRMED"][min(n_pass, 4)]
    print(f"\n  VERDICT: {verdict} ({n_pass}/4)")
    print(f"  Compare z2021 (MNIST): 4/4 PASS")

    output = {
        'experiment': 'z2028_cifar_blindsight',
        'hypothesis': 'Blindsight dissociation holds on harder task (CIFAR-10)',
        'timestamp': datetime.now().isoformat(),
        'device': str(device), 'epochs': args.epochs, 'n_params': n_params,
        'full': full, 'self_model_ablated': ablated,
        'encoder_ablated': enc_abl, 'scrambled': scrambled,
        'tests': {'t1': bool(t1), 't2': bool(t2), 't3': bool(t3), 't4': bool(t4)},
        'tests_passed': n_pass, 'verdict': verdict,
    }

    def json_safe(obj):
        if isinstance(obj, (np.bool_, np.integer)): return int(obj)
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        if isinstance(obj, dict): return {k: json_safe(v) for k, v in obj.items()}
        if isinstance(obj, list): return [json_safe(v) for v in obj]
        return obj

    rp = Path(__file__).parent.parent / 'results' / 'z2028_cifar_blindsight.json'
    with open(rp, 'w') as f:
        json.dump(json_safe(output), f, indent=2)
    print(f"\nResults saved to {rp}")


if __name__ == '__main__':
    main()
