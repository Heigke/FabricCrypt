#!/usr/bin/env python3
"""Stage 3+4: Transplant evaluation matrix and gate verdicts.

For each weight checkpoint W (host x seed), evaluate CIFAR-10 test accuracy
under 4 permutation conditions:
  - self:     model uses the permutation that was trained with this W
  - other:    model uses the OTHER host's TPM-derived permutation
  - random:   model uses a fresh random permutation (different RNG seed)
  - identity: model uses identity permutation (P = arange)

Outputs:
  results/IDENTITY_BENCHMARK_2026-05-30/vcek/transplant_<host>.json
"""
import argparse
import json
import socket
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import sys
sys.path.insert(0, str(Path(__file__).parent))
from importlib import import_module
train_mod = import_module("02_train")
PermResNet = train_mod.PermResNet

HOST = socket.gethostname()
ROOT = Path("results/IDENTITY_BENCHMARK_2026-05-30/vcek")
CKPT_DIR = ROOT / "checkpoints"


def get_test_loader():
    mean = (0.4914, 0.4822, 0.4465); std = (0.2470, 0.2435, 0.2616)
    test_tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
    te = datasets.CIFAR10("data/cifar10", train=False, download=True, transform=test_tf)
    return DataLoader(te, 512, shuffle=False, num_workers=2, pin_memory=True)


def load_perms():
    perms = {}
    for h in ("ikaros", "daedalus"):
        p = ROOT / f"P_{h}.npy"
        if p.exists():
            perms[h] = torch.from_numpy(np.load(p)).long()
    rng = np.random.default_rng(12345)
    perms["random"] = torch.from_numpy(rng.permutation(512).astype(np.int64)).long()
    perms["identity"] = torch.arange(512, dtype=torch.long)
    return perms


def eval_with_perm(state_dict, P, loader, device):
    model = PermResNet(P).to(device)
    # Replace the buffer P explicitly (in case load_state_dict keeps trained P)
    model.load_state_dict(state_dict, strict=True)
    model.P.copy_(P.to(device))
    model.eval(); c = t = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            c += (model(x).argmax(1) == y).sum().item(); t += y.numel()
    return c / t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    perms = load_perms()
    print("[eval] perms loaded:", list(perms.keys()), "device=", device)
    loader = get_test_loader()

    results = {"host_evaluator": HOST, "perms_available": list(perms.keys()), "cells": []}

    # Evaluate every checkpoint that lives in CKPT_DIR
    for ckpt_path in sorted(CKPT_DIR.glob("W_*_final.pt")):
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        w_host = ckpt["host"]; w_seed = ckpt["seed"]
        for pname, P in perms.items():
            acc = eval_with_perm(ckpt["state_dict"], P, loader, device)
            rec = {"ckpt": ckpt_path.name, "w_host": w_host, "w_seed": w_seed,
                   "perm": pname, "acc": acc}
            results["cells"].append(rec)
            print(f"[eval] W={w_host}_s{w_seed}  P={pname:8s}  acc={acc:.4f}", flush=True)

    out = ROOT / f"transplant_{HOST}.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"[eval] wrote {out}")


if __name__ == "__main__":
    main()
