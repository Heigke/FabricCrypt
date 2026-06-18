#!/usr/bin/env python3
"""Stage 2: Train a small ResNet on CIFAR-10 with a FIXED permutation layer
baked into the mid-block 512-dim activation. Permutation P is loaded from
results/IDENTITY_BENCHMARK_2026-05-30/vcek/P_<host>.npy and is NOT trainable.

Usage:
  HSA_OVERRIDE_GFX_VERSION=11.0.0 python 02_train.py --seed 0 --epochs 30
"""
import argparse
import json
import os
import socket
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

HOST = socket.gethostname()
ROOT = Path("results/IDENTITY_BENCHMARK_2026-05-30/vcek")
CKPT_DIR = ROOT / "checkpoints"
CKPT_DIR.mkdir(parents=True, exist_ok=True)


def load_perm(host: str) -> torch.Tensor:
    p = np.load(ROOT / f"P_{host}.npy")
    return torch.from_numpy(p).long()


class BasicBlock(nn.Module):
    def __init__(self, c_in, c_out, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(c_in, c_out, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(c_out)
        self.conv2 = nn.Conv2d(c_out, c_out, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(c_out)
        self.short = nn.Identity() if (stride == 1 and c_in == c_out) else nn.Sequential(
            nn.Conv2d(c_in, c_out, 1, stride, bias=False), nn.BatchNorm2d(c_out)
        )

    def forward(self, x):
        h = F.relu(self.bn1(self.conv1(x)))
        h = self.bn2(self.conv2(h))
        return F.relu(h + self.short(x))


class PermResNet(nn.Module):
    """Small ResNet with FIXED permutation on a 512-dim mid representation.

    32x32 input -> stem -> 3 stages -> GAP -> 512-d FC -> Permute(P) -> FC -> 10
    """
    def __init__(self, P: torch.Tensor):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(3, 64, 3, 1, 1, bias=False),
                                  nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.stage1 = nn.Sequential(BasicBlock(64, 64), BasicBlock(64, 64))
        self.stage2 = nn.Sequential(BasicBlock(64, 128, 2), BasicBlock(128, 128))
        self.stage3 = nn.Sequential(BasicBlock(128, 256, 2), BasicBlock(256, 256))
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc_in = nn.Linear(256, 512)
        self.bn_mid = nn.BatchNorm1d(512)
        # P is a non-trainable buffer
        self.register_buffer("P", P.clone())
        self.fc_out = nn.Linear(512, 10)

    def forward(self, x):
        h = self.stem(x)
        h = self.stage3(self.stage2(self.stage1(h)))
        h = self.gap(h).flatten(1)
        h = F.relu(self.bn_mid(self.fc_in(h)))
        h = h.index_select(1, self.P)         # <-- fixed permutation
        return self.fc_out(h)


def get_loaders(bs=128):
    mean = (0.4914, 0.4822, 0.4465); std = (0.2470, 0.2435, 0.2616)
    train_tf = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    test_tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
    root = os.environ.get("CIFAR_ROOT", "data/cifar10")
    tr = datasets.CIFAR10(root, train=True, download=True, transform=train_tf)
    te = datasets.CIFAR10(root, train=False, download=True, transform=test_tf)
    return (DataLoader(tr, bs, shuffle=True, num_workers=2, pin_memory=True),
            DataLoader(te, 512, shuffle=False, num_workers=2, pin_memory=True))


def evaluate(model, loader, device):
    model.eval(); correct = 0; total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            correct += (logits.argmax(1) == y).sum().item(); total += y.numel()
    return correct / total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--bs", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--wd", type=float, default=1e-4)
    args = ap.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[{HOST}] device={device} seed={args.seed} epochs={args.epochs}")

    P = load_perm(HOST)
    model = PermResNet(P).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    tr_loader, te_loader = get_loaders(args.bs)

    log = []
    t0 = time.time()
    for ep in range(args.epochs):
        model.train(); ep_loss = 0.0; n = 0
        for x, y in tr_loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            opt.zero_grad()
            logits = model(x)
            loss = F.cross_entropy(logits, y)
            loss.backward(); opt.step()
            ep_loss += loss.item() * y.numel(); n += y.numel()
        sched.step()
        acc = evaluate(model, te_loader, device)
        rec = {"epoch": ep, "loss": ep_loss / n, "test_acc": acc, "wall": time.time() - t0}
        log.append(rec)
        print(f"[{HOST} s{args.seed}] ep={ep:02d} loss={rec['loss']:.4f} acc={acc:.4f} wall={rec['wall']:.0f}s", flush=True)
        # checkpoint every epoch (overwrite per-seed latest + keep final)
        ckpt = {"state_dict": model.state_dict(), "seed": args.seed, "epoch": ep,
                "host": HOST, "P_first8": P[:8].tolist()}
        torch.save(ckpt, CKPT_DIR / f"W_{HOST}_s{args.seed}_latest.pt")

    torch.save(ckpt, CKPT_DIR / f"W_{HOST}_s{args.seed}_final.pt")
    (CKPT_DIR / f"train_log_{HOST}_s{args.seed}.json").write_text(json.dumps(log, indent=2))
    print(f"[{HOST} s{args.seed}] DONE final_acc={log[-1]['test_acc']:.4f}")


if __name__ == "__main__":
    main()
