"""E2: Attention-bias via DRAM latency timing.

Hypothesis: per-token DRAM-latency samples (chip memory subsystem state) provide
a structured attention bias that improves a tiny transformer text classifier
relative to random bias of matched magnitude.

Synthetic text-classification: 4 classes, vocabulary 64, sequence length 32.
Variants:
  A vanilla   — standard self-attention
  B embodied  — attention logits += alpha * DRAM_latency_bias (live samples)
  C random    — attention logits += random bias with same magnitude
Pre-reg gate: F1(embodied) > F1(random) by >= 1.5pp (CI lower > 0.015).
"""
from __future__ import annotations
import os, sys, time, json, argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from _common import thermal_guard, save_json, bootstrap_ci, diff_ci, dram_latency_burst

DEV = 'cuda' if torch.cuda.is_available() else 'cpu'


def make_text(n_train=2000, n_test=1000, vocab=64, seq=32, n_class=4, seed=0):
    """Synthetic classification: each class has a 'signature' bag of preferred tokens
    sprinkled in noise."""
    rng = np.random.default_rng(seed)
    n = n_train + n_test
    y = rng.integers(0, n_class, size=n)
    class_tokens = [rng.choice(vocab, size=8, replace=False) for _ in range(n_class)]
    X = rng.integers(0, vocab, size=(n, seq))
    for i in range(n):
        # sprinkle 3-4 class-specific tokens
        idxs = rng.choice(seq, size=4, replace=False)
        X[i, idxs] = rng.choice(class_tokens[y[i]], size=4)
    return (X[:n_train], y[:n_train]), (X[n_train:], y[n_train:])


class TinyAttn(nn.Module):
    """1-block self-attention classifier with optional bias injection on logits."""
    def __init__(self, vocab=64, d=64, n_class=4, seq=32):
        super().__init__()
        self.emb = nn.Embedding(vocab, d)
        self.pos = nn.Parameter(torch.randn(seq, d) * 0.02)
        self.q = nn.Linear(d, d, bias=False)
        self.k = nn.Linear(d, d, bias=False)
        self.v = nn.Linear(d, d, bias=False)
        self.out = nn.Linear(d, n_class)
        self.d = d

    def forward(self, x, bias=None):
        # x: (B, S)
        h = self.emb(x) + self.pos  # (B,S,D)
        q = self.q(h); k = self.k(h); v = self.v(h)
        logits = q @ k.transpose(-1, -2) / (self.d ** 0.5)  # (B,S,S)
        if bias is not None:
            logits = logits + bias  # broadcast (S,) or (S,S)
        att = F.softmax(logits, dim=-1)
        h = att @ v  # (B,S,D)
        pooled = h.mean(dim=1)
        return self.out(pooled)


def chip_attention_bias(seq=32, alpha=1.0):
    """Build per-position bias from DRAM latency samples — one sample per position."""
    n_chunks = (seq + 7) // 8
    parts = []
    for _ in range(n_chunks):
        parts.append(dram_latency_burst(n=8).astype(np.float32))
    arr = np.concatenate(parts)[:seq]
    arr = (arr - arr.mean()) / (arr.std() + 1e-6)
    return torch.from_numpy(arr * alpha).to(DEV)  # shape (seq,)


def random_bias(seq=32, std=1.0):
    return (torch.randn(seq) * std).to(DEV)


def train_one(variant, train_xy, test_xy, seq, alpha, seed,
              epochs=12, bs=64, lr=2e-3):
    torch.manual_seed(seed)
    Xtr, ytr = train_xy
    Xte, yte = test_xy
    Xtr = torch.from_numpy(Xtr).long().to(DEV)
    ytr = torch.from_numpy(ytr).long().to(DEV)
    Xte = torch.from_numpy(Xte).long().to(DEV)
    yte = torch.from_numpy(yte).long().to(DEV)

    model = TinyAttn(vocab=64, d=64, n_class=int(ytr.max().item()) + 1, seq=seq).to(DEV)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    # cache calibration std for random matched-magnitude
    if variant == 'embodied':
        b_samples = [chip_attention_bias(seq, alpha=1.0) for _ in range(8)]
        match_std = float(torch.stack(b_samples).std().item())
    else:
        match_std = 1.0

    n = Xtr.size(0)
    for ep in range(epochs):
        thermal_guard()
        perm = torch.randperm(n, device=DEV)
        model.train()
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            xb = Xtr[idx]; yb = ytr[idx]
            if variant == 'vanilla':
                bias = None
            elif variant == 'embodied':
                bias = chip_attention_bias(seq, alpha=alpha)  # (S,)
            else:  # random
                bias = random_bias(seq, std=alpha * match_std)
            logits = model(xb, bias=bias)
            loss = F.cross_entropy(logits, yb)
            opt.zero_grad(); loss.backward(); opt.step()
    model.eval()
    # multi-eval (average over a few bias samples for the bias variants)
    accs = []
    with torch.no_grad():
        for _ in range(5):
            if variant == 'vanilla':
                bias = None
            elif variant == 'embodied':
                bias = chip_attention_bias(seq, alpha=alpha)
            else:
                bias = random_bias(seq, std=alpha * match_std)
            preds = model(Xte, bias=bias).argmax(1)
            accs.append((preds == yte).float().mean().item())
    return float(np.mean(accs)), match_std


def main(seeds=15):
    print(f"[E2] starting, seeds={seeds}, device={DEV}")
    seq = 32
    alpha = 0.3
    train_xy, test_xy = make_text(n_train=2000, n_test=1000, seq=seq, seed=42)

    results = {'vanilla': [], 'embodied': [], 'random': []}
    for s in range(seeds):
        thermal_guard(verbose=True)
        t0 = time.time()
        for variant in ('vanilla', 'embodied', 'random'):
            acc, _ = train_one(variant, train_xy, test_xy, seq, alpha, seed=s)
            results[variant].append(acc)
        print(f"[E2] seed {s}: van={results['vanilla'][-1]:.4f}  "
              f"emb={results['embodied'][-1]:.4f}  "
              f"rnd={results['random'][-1]:.4f}  "
              f"(elapsed {time.time()-t0:.1f}s)", flush=True)

    summary = {}
    for k, v in results.items():
        m, lo, hi = bootstrap_ci(v, seed=0)
        summary[k] = {'mean': m, 'ci95': [lo, hi], 'values': v}
    dmean, dlo, dhi = diff_ci(results['embodied'], results['random'], seed=0)
    summary['delta_embodied_minus_random'] = {'mean': dmean, 'ci95': [dlo, dhi]}
    summary['gate'] = {
        'criterion': 'embodied - random > 1.5pp (CI lower > 0.015)',
        'delta_mean_pp': dmean * 100,
        'ci95_pp': [dlo * 100, dhi * 100],
        'pass': bool(dlo > 0.015),
    }
    summary['seeds'] = seeds
    save_json('e2_attention_bias.json', summary)
    print(json.dumps(summary['gate'], indent=2))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', type=int, default=15)
    args = ap.parse_args()
    main(args.seeds)
