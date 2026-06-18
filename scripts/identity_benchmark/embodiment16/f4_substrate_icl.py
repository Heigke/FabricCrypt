"""F4: Substrate as in-context example.

Few-shot classification: 5 in-context examples vs 5 examples + chip-state as 6th.

We use a tiny prototype-similarity model (no LLM needed) so we can do 30 seeds
quickly. The "in-context" mechanism is a simple key-value cache: examples are
stored as (embedding, label) pairs, query is classified by softmax over
cosine similarity to keys.

The "substrate" 6th slot has a *learned* projection from the 32-d chip vector
into the same embedding space. Its label is the per-batch majority label
within the support set — i.e. a soft prior derived from the support.

Pre-reg: embodied accuracy > vanilla accuracy by >= 4pp (CI lower > 0.04)
on 1000 test queries averaged over 30 seeds.
"""
from __future__ import annotations
import os, sys, time, json, argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from _common import thermal_guard, save_json, bootstrap_ci, diff_ci, temp_c

sys.path.insert(0, os.path.join(HERE, '..', 'embodiment14b'))
from signature_live import LiveSig

DEV = 'cuda' if torch.cuda.is_available() else 'cpu'


def make_text_like(n_classes=8, n_train=1500, n_test=1000, d=16, seed=0):
    """Each example is a 16-d "text embedding" generated from a per-class prototype + noise.

    This is a stand-in for token embeddings of short sentences. It's honest about
    the abstraction: F4 is testing the MECHANISM of context-augmented prototypes,
    not LLM behaviour specifically.
    """
    rng = np.random.default_rng(seed)
    protos = rng.standard_normal((n_classes, d)).astype(np.float32) * 0.6
    n = n_train + n_test
    y = rng.integers(0, n_classes, size=n)
    X = protos[y] + rng.standard_normal((n, d)).astype(np.float32) * 1.4
    return (X[:n_train], y[:n_train]), (X[n_train:], y[n_train:]), n_classes


class ProtoNet(nn.Module):
    def __init__(self, d=16, hidden=32, sig_dim=32, embodied=False):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(d, hidden), nn.ReLU(), nn.Linear(hidden, hidden))
        self.embodied = embodied
        if embodied:
            self.sig_proj = nn.Sequential(
                nn.Linear(sig_dim, hidden), nn.ReLU(), nn.Linear(hidden, hidden))
            self.sig_weight = nn.Parameter(torch.tensor(0.1))

    def encode(self, x):
        return F.normalize(self.encoder(x), dim=-1)

    def forward(self, support_x, support_y, query_x, n_classes, sig_vec=None):
        # support_x: (k, d), support_y: (k,), query_x: (Q, d)
        s_emb = self.encode(support_x)            # (k, h)
        q_emb = self.encode(query_x)              # (Q, h)
        # per-class prototype
        protos = torch.zeros(n_classes, s_emb.size(1), device=s_emb.device)
        counts = torch.zeros(n_classes, device=s_emb.device)
        for c in range(n_classes):
            m = (support_y == c)
            if m.any():
                protos[c] = s_emb[m].mean(0)
                counts[c] = m.float().sum()
        protos = F.normalize(protos, dim=-1)

        if self.embodied and sig_vec is not None:
            sig_emb = F.normalize(self.sig_proj(sig_vec), dim=-1)  # (h,)
            # add sig as soft prior to each prototype (weighted)
            w = torch.sigmoid(self.sig_weight)
            protos = F.normalize(protos + w * sig_emb.unsqueeze(0), dim=-1)

        # cosine similarity → logits (Q, C)
        logits = q_emb @ protos.T * 10.0
        return logits


def episode_iterator(X, y, n_classes, k_shot=5, q_size=10, n_episodes=50, rng=None):
    rng = rng or np.random.default_rng(0)
    Nt = len(X)
    for _ in range(n_episodes):
        # support: 1 per class (k_shot=5 = one per 5 classes)
        support_idx = []
        for c in range(n_classes):
            cand = np.where(y == c)[0]
            support_idx.append(rng.choice(cand, 1)[0])
        support_idx = np.array(support_idx)
        # query: random q_size
        query_idx = rng.integers(0, Nt, q_size)
        yield support_idx, query_idx


def train_one(variant, train_xy, sig, seed, n_epi=200, n_classes=8):
    torch.manual_seed(seed); np.random.seed(seed)
    Xtr, ytr = train_xy
    Xt = torch.from_numpy(Xtr).to(DEV); yt = torch.from_numpy(ytr).long().to(DEV)
    model = ProtoNet(d=Xtr.shape[1], hidden=32, embodied=(variant == 'embodied')).to(DEV)
    opt = torch.optim.Adam(model.parameters(), lr=2e-3)
    rng = np.random.default_rng(seed)
    model.train()
    step = 0
    for si, qi in episode_iterator(Xtr, ytr, n_classes, k_shot=n_classes,
                                    q_size=20, n_episodes=n_epi, rng=rng):
        step += 1
        if step % 50 == 0:
            thermal_guard()
        sx = Xt[si]; sy = yt[si]
        qx = Xt[qi]; qy = yt[qi]
        sig_vec = None
        if variant == 'embodied':
            sig_vec = torch.from_numpy(np.asarray(sig.read(), dtype=np.float32)).to(DEV)
        logits = model(sx, sy, qx, n_classes, sig_vec)
        loss = F.cross_entropy(logits, qy)
        opt.zero_grad(); loss.backward(); opt.step()
    return model


def eval_one(model, train_xy, test_xy, sig, variant, n_episodes=100, n_classes=8, seed=0):
    Xtr, ytr = train_xy
    Xte, yte = test_xy
    Xt = torch.from_numpy(Xtr).to(DEV); yt = torch.from_numpy(ytr).long().to(DEV)
    Xq = torch.from_numpy(Xte).to(DEV); yq = torch.from_numpy(yte).long().to(DEV)
    rng = np.random.default_rng(1000 + seed)
    correct = 0; total = 0
    model.eval()
    with torch.no_grad():
        for si, qi in episode_iterator(Xte, yte, n_classes, k_shot=n_classes,
                                        q_size=10, n_episodes=n_episodes, rng=rng):
            sx = Xq[si]; sy = yq[si]
            qx = Xq[qi]; qyt = yq[qi]
            sig_vec = None
            if variant == 'embodied':
                sig_vec = torch.from_numpy(np.asarray(sig.read(), dtype=np.float32)).to(DEV)
            logits = model(sx, sy, qx, n_classes, sig_vec)
            pred = logits.argmax(1)
            correct += (pred == qyt).sum().item()
            total += qx.size(0)
    return correct / max(1, total)


def main(seeds=30):
    print(f"[F4] start, seeds={seeds}, device={DEV}, temp={temp_c():.1f}C", flush=True)
    sig = LiveSig()
    train_xy, test_xy, n_cls = make_text_like(n_classes=8, n_train=1500, n_test=1000, d=16, seed=42)
    results = {'vanilla': [], 'embodied': []}
    t_start = time.time()
    for s in range(seeds):
        if s % 3 == 0:
            thermal_guard()
        for v in ('vanilla', 'embodied'):
            m = train_one(v, train_xy, sig, seed=s, n_classes=n_cls)
            acc = eval_one(m, train_xy, test_xy, sig, v, n_episodes=100, n_classes=n_cls, seed=s)
            results[v].append(acc)
        print(f"[F4] s{s}: van={results['vanilla'][-1]:.3f} emb={results['embodied'][-1]:.3f} "
              f"T={temp_c():.1f}C", flush=True)
        if (time.time() - t_start) > 420:
            print(f"[F4] time budget hit", flush=True)
            break

    summary = {}
    for k, v in results.items():
        m, lo, hi = bootstrap_ci(v)
        summary[k] = {'mean': m, 'ci95': [lo, hi], 'values': v}
    dmean, dlo, dhi = diff_ci(results['embodied'], results['vanilla'])
    summary['delta_emb_minus_van'] = {'mean_pp': dmean*100, 'ci95_pp': [dlo*100, dhi*100]}
    gate_pass = bool(dlo > 0.04)
    summary['gate'] = {
        'criterion': 'embodied - vanilla > 4pp (CI lower > 0.04)',
        'delta_mean_pp': dmean*100, 'ci95_pp': [dlo*100, dhi*100],
        'pass': gate_pass,
    }
    summary['n_seeds_run'] = len(results['vanilla'])
    save_json('f4_substrate_icl.json', summary)
    print(json.dumps(summary['gate'], indent=2))
    return summary


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', type=int, default=30)
    args = ap.parse_args()
    main(seeds=args.seeds)
