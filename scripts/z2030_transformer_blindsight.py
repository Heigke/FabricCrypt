#!/usr/bin/env python3
"""
z2030: Cross-Architecture Transfer — Transformer Blindsight

Tests whether z2021's blindsight dissociation (4/4 PASS on CNN) also holds
on a transformer architecture. If yes → architecture-independent finding.
If no → architecture-specific artifact.

Architecture: Vision Transformer (ViT-tiny) with patch embedding
Same protocol: task head + self-model, ablate, check dissociation.

NO consciousness losses.
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


class PatchEmbedding(nn.Module):
    def __init__(self, img_size=28, patch_size=7, in_channels=1, embed_dim=128):
        super().__init__()
        self.n_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
        self.pos_embed = nn.Parameter(torch.randn(1, self.n_patches + 1, embed_dim) * 0.02)

    def forward(self, x):
        patches = self.proj(x).flatten(2).transpose(1, 2)  # [B, n_patches, dim]
        cls = self.cls_token.expand(x.size(0), -1, -1)
        tokens = torch.cat([cls, patches], dim=1)
        return tokens + self.pos_embed


class TransformerBlock(nn.Module):
    def __init__(self, dim, n_heads=4, mlp_ratio=2.0, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, n_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)), nn.GELU(),
            nn.Linear(int(dim * mlp_ratio), dim), nn.Dropout(dropout),
        )

    def forward(self, x):
        h = self.norm1(x)
        h, _ = self.attn(h, h, h)
        x = x + h
        x = x + self.mlp(self.norm2(x))
        return x


class ViTBlindsight(nn.Module):
    def __init__(self, img_size=28, patch_size=7, embed_dim=128, depth=4, n_heads=4):
        super().__init__()
        self.patch_embed = PatchEmbedding(img_size, patch_size, 1, embed_dim)
        self.blocks = nn.Sequential(*[TransformerBlock(embed_dim, n_heads) for _ in range(depth)])
        self.norm = nn.LayerNorm(embed_dim)

        # Task head (from CLS token)
        self.task_head = nn.Sequential(nn.Linear(embed_dim, 64), nn.ReLU(), nn.Linear(64, 10))

        # Self-model (from CLS token)
        self.self_model = nn.Sequential(
            nn.Linear(embed_dim, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        tokens = self.patch_embed(x)
        tokens = self.blocks(tokens)
        tokens = self.norm(tokens)
        cls = tokens[:, 0]  # CLS token
        logits = self.task_head(cls)
        confidence = torch.sigmoid(self.self_model(cls)).squeeze(-1)
        return {'logits': logits, 'confidence': confidence, 'hidden': cls}


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
    conf, corr = np.array(all_conf), np.array(all_corr)
    return {
        'task_acc': correct / total,
        'type2_auroc': compute_type2_auroc(conf, corr),
        'mean_conf': float(conf.mean()),
    }


def ablate_self_model(model):
    with torch.no_grad():
        for p in model.self_model.parameters():
            p.zero_()


def ablate_encoder(model):
    with torch.no_grad():
        for p in model.patch_embed.parameters():
            nn.init.normal_(p, 0, 0.01)
        for p in model.blocks.parameters():
            nn.init.normal_(p, 0, 0.01)


def scramble_eval(model, loader, device):
    model.eval()
    all_conf, all_corr = [], []
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            tokens = model.patch_embed(images)
            tokens = model.blocks(tokens)
            tokens = model.norm(tokens)
            cls = tokens[:, 0]
            logits = model.task_head(cls)
            preds = logits.argmax(1)
            c = (preds == labels)
            scrambled = cls[torch.randperm(cls.size(0))]
            conf = torch.sigmoid(model.self_model(scrambled)).squeeze(-1)
            all_conf.extend(conf.cpu().numpy())
            all_corr.extend(c.cpu().numpy().astype(float))
            correct += c.sum().item()
            total += images.size(0)
    conf, corr = np.array(all_conf), np.array(all_corr)
    return {'task_acc': correct / total, 'type2_auroc': compute_type2_auroc(conf, corr)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch-size', type=int, default=128)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[z2030] Device: {device}")

    from torchvision import datasets, transforms
    data_dir = Path(__file__).parent.parent / 'data'
    data_dir.mkdir(exist_ok=True)

    # MNIST with noise for uncertainty
    class NoisyMNIST(torch.utils.data.Dataset):
        def __init__(self, base, noise_frac=0.3, noise_std=1.5):
            self.base = base
            self.noise_frac = noise_frac
            self.noise_std = noise_std
        def __len__(self):
            return len(self.base)
        def __getitem__(self, idx):
            img, label = self.base[idx]
            if idx % int(1.0 / max(self.noise_frac, 0.01)) == 0:
                img = img + torch.randn_like(img) * self.noise_std
                img = img.clamp(0, 1)
            return img, label

    tf = transforms.ToTensor()
    train_data = datasets.MNIST(str(data_dir), train=True, download=True, transform=tf)
    test_base = datasets.MNIST(str(data_dir), train=False, download=True, transform=tf)
    test_data = NoisyMNIST(test_base)

    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_data, batch_size=256, shuffle=False)

    print(f"\n{'='*70}")
    print(f"  z2030: Transformer Blindsight (Cross-Architecture)")
    print(f"  Does z2021's dissociation hold on ViT instead of CNN?")
    print(f"  NO consciousness losses")
    print(f"{'='*70}")

    model = ViTBlindsight(embed_dim=128, depth=4, n_heads=4).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {n_params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    for ep in range(1, args.epochs + 1):
        t0 = time.time()
        stats = train_epoch(model, train_loader, optimizer, device)
        scheduler.step()
        elapsed = time.time() - t0
        if ep % 4 == 0 or ep == 1 or ep == args.epochs:
            print(f"  Epoch {ep:2d}/{args.epochs}  loss={stats['loss']:.4f}  acc={stats['acc']:.3f}  ({elapsed:.1f}s)")

    checkpoint = {k: v.clone() for k, v in model.state_dict().items()}

    print(f"\n  [1/4] Full model...")
    full = evaluate_full(model, test_loader, device)
    print(f"    Acc={full['task_acc']:.4f}  AUROC={full['type2_auroc']:.4f}")

    print(f"  [2/4] Self-model ablated...")
    model.load_state_dict(checkpoint)
    ablate_self_model(model)
    ablated = evaluate_full(model, test_loader, device)
    print(f"    Acc={ablated['task_acc']:.4f}  AUROC={ablated['type2_auroc']:.4f}")

    print(f"  [3/4] Encoder ablated...")
    model.load_state_dict(checkpoint)
    ablate_encoder(model)
    enc_abl = evaluate_full(model, test_loader, device)
    print(f"    Acc={enc_abl['task_acc']:.4f}  AUROC={enc_abl['type2_auroc']:.4f}")

    print(f"  [4/4] Scrambled...")
    model.load_state_dict(checkpoint)
    scrambled = scramble_eval(model, test_loader, device)
    print(f"    Acc={scrambled['task_acc']:.4f}  AUROC={scrambled['type2_auroc']:.4f}")

    # Analysis
    print(f"\n{'='*70}")
    print(f"  FINAL ANALYSIS: Transformer Blindsight")
    print(f"{'='*70}")

    acc_preserved = abs(full['task_acc'] - ablated['task_acc']) < 0.02
    auroc_collapsed = ablated['type2_auroc'] < 0.55
    enc_both = enc_abl['task_acc'] < full['task_acc'] - 0.1 and enc_abl['type2_auroc'] < full['type2_auroc'] - 0.1
    scr_collapsed = scrambled['type2_auroc'] < 0.55

    t1 = acc_preserved and auroc_collapsed
    t2 = enc_both
    t3 = scr_collapsed
    t4 = full['type2_auroc'] > 0.6

    print(f"  T1: Blindsight dissociation: {'PASS' if t1 else 'FAIL'}")
    print(f"  T2: Encoder ablation both:   {'PASS' if t2 else 'FAIL'}")
    print(f"  T3: Scramble collapses:       {'PASS' if t3 else 'FAIL'}")
    print(f"  T4: Baseline metacognition:   {'PASS' if t4 else 'FAIL'}")

    n_pass = sum([t1, t2, t3, t4])
    verdict = {0: "NO_TRANSFER", 1: "WEAK", 2: "PARTIAL", 3: "MOSTLY", 4: "TRANSFORMER_BLINDSIGHT_CONFIRMED"}[n_pass]
    print(f"\n  VERDICT: {verdict} ({n_pass}/4)")
    print(f"  Compare z2021 (CNN): 4/4 PASS")

    output = {
        'experiment': 'z2030_transformer_blindsight',
        'hypothesis': 'Blindsight dissociation transfers from CNN to transformer',
        'timestamp': datetime.now().isoformat(),
        'device': str(device), 'n_params': n_params,
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

    rp = Path(__file__).parent.parent / 'results' / 'z2030_transformer_blindsight.json'
    with open(rp, 'w') as f:
        json.dump(json_safe(output), f, indent=2)
    print(f"\nResults saved to {rp}")


if __name__ == '__main__':
    main()
