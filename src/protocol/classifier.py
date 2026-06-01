"""TwinMLP classifier + training loops.

Two soft heads built on the [phys ; nonce_emb] = 64-dim input:
  T2 = anomaly head (own normal vs scaled/shifted own)
  T3 = twin head    (own vs peer chip)

In practice the deterministic `verifier.plan_consistency_score` is the
HARD replay gate; T3 is used as a diagnostic and for cross-chip
discrimination only.
"""
from __future__ import annotations
import os
import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .nonce_signature import NonceSig
from .nonce_derivation import fresh_nonce, nonce_embedding

DIM = 64


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)


class TwinMLP(nn.Module):
    def __init__(self, in_d: int = DIM, n_out: int = 2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_d, 96), nn.GELU(),
            nn.Linear(96, 96),    nn.GELU(),
            nn.Linear(96, n_out),
        )

    def forward(self, x):
        return self.net(x)


def collect_paired(sig: NonceSig, n: int, rng: np.random.Generator,
                   every: int = 8, raw: bool = True,
                   thermal_guard=None):
    """Read n (nonce, sig) pairs. If `thermal_guard` is provided it's called
    every `every` reads (use signature.thermal.thermal_guard)."""
    nonces = np.empty((n, 8), dtype=np.uint8)
    sigs = np.empty((n, DIM), dtype=np.float32)
    for i in range(n):
        if thermal_guard is not None and (i % every) == 0:
            thermal_guard()
        nb = fresh_nonce(rng)
        nonces[i] = np.frombuffer(nb, dtype=np.uint8)
        sigs[i] = sig.read(nb, raw=raw)
    return nonces, sigs


def _split_train_test(X, y, frac=0.7):
    perm = np.random.permutation(len(X))
    X, y = X[perm], y[perm]
    split = int(frac * len(X))
    return np.arange(split), np.arange(split, len(X)), X, y


def train_T2_anomaly(host_sigs: np.ndarray, n_seeds: int = 30, epochs: int = 12,
                     device: str = "cpu", verbose: bool = False):
    """Anomaly: 0=normal own sig, 1=perturbed own sig."""
    from sklearn.metrics import roc_auc_score
    N = len(host_sigs)
    rng = np.random.default_rng(7)
    n_anom = N // 5
    idx = rng.choice(N, size=n_anom, replace=False)
    anom = host_sigs[idx].copy()
    phys_std = host_sigs[:, :32].std(axis=0) + 0.1
    anom[:, :32] += (rng.normal(0, 1.0, size=(n_anom, 32)) *
                     (2.0 * phys_std)).astype(np.float32)
    X = np.concatenate([host_sigs, anom], 0).astype(np.float32)
    y = np.concatenate([np.zeros(N), np.ones(n_anom)], 0).astype(np.int64)
    tr, te, X, y = _split_train_test(X, y)
    aurocs = []
    last = None
    for s in range(n_seeds):
        set_seed(s)
        m = TwinMLP().to(device)
        opt = torch.optim.AdamW(m.parameters(), lr=3e-3, weight_decay=1e-4)
        for _ in range(epochs):
            order = np.random.permutation(len(tr))
            for i in range(0, len(order), 64):
                b = tr[order[i:i+64]]
                xb = torch.from_numpy(X[b]).to(device)
                yb = torch.from_numpy(y[b]).to(device)
                loss = F.cross_entropy(m(xb), yb)
                opt.zero_grad(); loss.backward(); opt.step()
        m.eval()
        with torch.no_grad():
            scores = F.softmax(m(torch.from_numpy(X[te]).to(device)),
                               dim=-1)[:, 1].cpu().numpy()
        try:
            a = float(roc_auc_score(y[te], scores))
        except Exception:
            a = 0.5
        aurocs.append(a)
        last = m
        if verbose: print(f"[T2 s={s}] AUROC={a:.3f}")
    return aurocs, last


def train_T3_twin(own_sigs: np.ndarray, peer_sigs: np.ndarray,
                  n_seeds: int = 30, epochs: int = 15,
                  device: str = "cpu", verbose: bool = False):
    """Twin: own (0) vs peer (1)."""
    from sklearn.metrics import roc_auc_score
    if len(peer_sigs) == 0:
        # synthesize a weak negative if no peer available — caller should
        # ideally supply a real peer's paired_sigs.npz
        peer_sigs = own_sigs.copy()
        np.random.shuffle(peer_sigs.T)
        peer_sigs[:, :32] += 1.5
    n = min(len(own_sigs), len(peer_sigs))
    own = own_sigs[:n].astype(np.float32)
    peer = peer_sigs[:n].astype(np.float32)
    X = np.concatenate([own, peer], 0).astype(np.float32)
    y = np.concatenate([np.zeros(n), np.ones(n)], 0).astype(np.int64)
    tr, te, X, y = _split_train_test(X, y)
    aurocs = []
    best_model = None
    best_a = -1.0
    for s in range(n_seeds):
        set_seed(s)
        m = TwinMLP().to(device)
        opt = torch.optim.AdamW(m.parameters(), lr=3e-3, weight_decay=1e-4)
        for _ in range(epochs):
            order = np.random.permutation(len(tr))
            for i in range(0, len(order), 32):
                b = tr[order[i:i+32]]
                xb = torch.from_numpy(X[b]).to(device)
                yb = torch.from_numpy(y[b]).to(device)
                loss = F.cross_entropy(m(xb), yb)
                opt.zero_grad(); loss.backward(); opt.step()
        m.eval()
        with torch.no_grad():
            scores = F.softmax(m(torch.from_numpy(X[te]).to(device)),
                               dim=-1)[:, 1].cpu().numpy()
        try:
            a = float(roc_auc_score(y[te], scores))
        except Exception:
            a = 0.5
        aurocs.append(a)
        if a > best_a:
            best_a = a
            best_model = {k: v.clone() for k, v in m.state_dict().items()}
        if verbose: print(f"[T3 s={s}] AUROC={a:.3f}")
    return aurocs, best_model
