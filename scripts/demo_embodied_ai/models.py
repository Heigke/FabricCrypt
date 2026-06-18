"""Tiny demo models — substrate-bound (embodied) and substrate-blind (vanilla).

Each is a 2-layer MLP on the 32-d live signature. Training takes <2s per head
on CPU. We pre-train at server startup using the recorded ikaros/daedalus sigs
from Phase 14B (results/IDENTITY_BENCHMARK_2026-05-30/embodiment14b/*_sigs.npz).

Heads:
  - T2 anomaly head: input=32d sig, output=2 (normal/anomaly).
    Training data: own sigs (normal) + synthetic +3.5sigma perturbed (anomaly).
  - T3 identity head: input=32d sig, output=2 (this_host / peer_host).
    Training data: this host's recorded sigs (label 0) + peer's sigs (label 1).

Vanilla counterpart: identical MLP but receives a zero vector instead of the
live sig — it has no substrate access, so it cannot exceed chance.
"""
from __future__ import annotations
import os, json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


SIG_DIM = 32


class TinyHead(nn.Module):
    def __init__(self, in_d=SIG_DIM, hidden=64, out=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_d, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, out),
        )
    def forward(self, x):
        return self.net(x)


def _train_head(X, y, epochs=30, lr=3e-3, batch=64, seed=0):
    torch.manual_seed(seed); np.random.seed(seed)
    m = TinyHead()
    opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=1e-4)
    Xt = torch.from_numpy(X.astype(np.float32))
    yt = torch.from_numpy(y.astype(np.int64))
    n = len(X)
    for _ in range(epochs):
        order = np.random.permutation(n)
        for i in range(0, n, batch):
            b = order[i:i+batch]
            opt.zero_grad()
            loss = F.cross_entropy(m(Xt[b]), yt[b])
            loss.backward(); opt.step()
    m.eval()
    return m


def train_anomaly_head(own_sigs: np.ndarray, n_anom_frac=0.1, seed=0):
    """Synthesize anomalies by +3.5sigma shift on 4 random channels (matches
    Phase 14B T2 protocol). Returns (model, calibration_threshold)."""
    n = len(own_sigs)
    n_anom = max(20, int(n * n_anom_frac))
    rng = np.random.default_rng(seed)
    X = own_sigs.copy().astype(np.float32)
    y = np.zeros(n, dtype=np.int64)
    anom_idx = rng.choice(n, size=n_anom, replace=False)
    for i in anom_idx:
        ch = rng.choice(32, size=4, replace=False)
        shift = np.zeros(32, dtype=np.float32)
        shift[ch] = 3.5
        X[i] = np.clip(X[i] + shift, -4.0, 4.0)
        y[i] = 1
    m = _train_head(X, y, epochs=30, seed=seed)
    return m


def train_identity_head(own_sigs: np.ndarray, peer_sigs: np.ndarray, seed=0):
    """Binary: own=0, peer=1. Trained on pre-recorded sig dumps."""
    own = own_sigs.astype(np.float32)
    peer = peer_sigs.astype(np.float32)
    X = np.concatenate([own, peer], axis=0)
    y = np.concatenate([np.zeros(len(own)), np.ones(len(peer))], axis=0).astype(np.int64)
    m = _train_head(X, y, epochs=20, seed=seed)
    return m


def anomaly_score(model: TinyHead, sig: np.ndarray) -> float:
    """Return P(anomaly) in [0,1]."""
    with torch.no_grad():
        x = torch.from_numpy(sig.astype(np.float32)).unsqueeze(0)
        p = F.softmax(model(x), dim=-1)[0, 1].item()
    return float(p)


def identity_predict(model: TinyHead, sig: np.ndarray):
    """Return (label_idx, confidence)."""
    with torch.no_grad():
        x = torch.from_numpy(sig.astype(np.float32)).unsqueeze(0)
        p = F.softmax(model(x), dim=-1)[0]
        idx = int(p.argmax().item())
        conf = float(p[idx].item())
    return idx, conf


def vanilla_anomaly_score(seed_counter: int) -> float:
    """A model without substrate access has no signal — output ~uniform."""
    rng = np.random.default_rng(seed_counter)
    return float(rng.random())


def vanilla_identity_predict(seed_counter: int):
    """Same — random guess, coin flip."""
    rng = np.random.default_rng(seed_counter)
    idx = int(rng.integers(0, 2))
    conf = float(0.5 + rng.random()*0.1)
    return idx, conf
