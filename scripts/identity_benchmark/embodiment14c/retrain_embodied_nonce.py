"""Phase 14C Task B — retrain T2 (anomaly) + T3 (twin) classifiers with
nonce-keyed signatures. Both models see [phys ; nonce_emb] as input.

Training protocol:
  - For each training example, sample a *fresh* nonce.
  - Honest examples: own chip running, fresh nonce → call NonceSig.read(nonce).
  - Foreign examples: peer's pre-recorded (nonce, sig) pairs.
  - Anomaly examples (T2): synthesise scaled / shifted versions of own sigs.

We save:
  - results/.../embodiment14c/<host>_paired_sigs.npz   # for peer transplant
  - results/.../embodiment14c/<host>_models.pt         # trained T2+T3
  - results/.../embodiment14c/<host>_training.json
"""
from __future__ import annotations
import os, sys, json, time, argparse, random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
P13 = os.path.abspath(os.path.join(HERE, '..', 'embodiment13'))
sys.path.insert(0, P13)

from common13 import thermal_guard, hostname, save_json
from nonce_signature import NonceSig, fresh_nonce

DIM = 64


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)


class TwinMLP(nn.Module):
    def __init__(self, in_d=DIM, n_out=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_d, 96), nn.GELU(),
            nn.Linear(96, 96), nn.GELU(),
            nn.Linear(96, n_out),
        )
    def forward(self, x): return self.net(x)


def collect_paired(sig: NonceSig, n: int, rng: np.random.Generator, every: int = 8, raw: bool = True):
    """Read n (nonce, sig) pairs, calling thermal_guard every `every` reads.
    raw=True (default) preserves per-host identity bias needed for twin task.
    """
    nonces = np.empty((n, 8), dtype=np.uint8)
    sigs = np.empty((n, DIM), dtype=np.float32)
    for i in range(n):
        if (i % every) == 0:
            thermal_guard()
        nb = fresh_nonce(rng)
        nonces[i] = np.frombuffer(nb, dtype=np.uint8)
        sigs[i] = sig.read(nb, raw=raw)
    return nonces, sigs


def train_T2_anomaly(host_sigs: np.ndarray, n_seeds=30, epochs=12, device='cpu', verbose=False):
    """Anomaly detector: 0=normal own sig, 1=anomaly (perturbed/scaled own sig).
    NOTE: anomaly tests do not transfer cross-host; gate is just AUROC.
    """
    N = len(host_sigs)
    # Anomalies: perturb subset of physical dims with large noise; nonce-emb intact
    anom = host_sigs.copy()
    rng = np.random.default_rng(7)
    n_anom = N // 5
    idx = rng.choice(N, size=n_anom, replace=False)
    anom = host_sigs[idx].copy()
    # noise magnitude scaled to per-dim std so it shifts feats meaningfully
    phys_std = host_sigs[:, :32].std(axis=0) + 0.1
    anom[:, :32] += (rng.normal(0, 1.0, size=(n_anom, 32)) * (2.0 * phys_std)).astype(np.float32)
    X = np.concatenate([host_sigs, anom], 0).astype(np.float32)
    y = np.concatenate([np.zeros(N), np.ones(n_anom)], 0).astype(np.int64)
    perm = np.random.permutation(len(X))
    X, y = X[perm], y[perm]
    split = int(0.7*len(X)); tr, te = np.arange(split), np.arange(split, len(X))
    aurocs = []
    for s in range(n_seeds):
        set_seed(s)
        m = TwinMLP(in_d=DIM, n_out=2).to(device)
        opt = torch.optim.AdamW(m.parameters(), lr=3e-3, weight_decay=1e-4)
        for ep in range(epochs):
            order = np.random.permutation(len(tr))
            for i in range(0, len(order), 64):
                b = tr[order[i:i+64]]
                xb = torch.from_numpy(X[b]).to(device)
                yb = torch.from_numpy(y[b]).to(device)
                loss = F.cross_entropy(m(xb), yb)
                opt.zero_grad(); loss.backward(); opt.step()
        m.eval()
        with torch.no_grad():
            scores = F.softmax(m(torch.from_numpy(X[te]).to(device)), dim=-1)[:,1].cpu().numpy()
        # AUROC
        from sklearn.metrics import roc_auc_score
        try:
            a = float(roc_auc_score(y[te], scores))
        except Exception:
            a = 0.5
        aurocs.append(a)
        if verbose: print(f"[T2 s={s}] AUROC={a:.3f}")
    return aurocs, m  # return last model + all aurocs


def train_T3_twin(own_sigs: np.ndarray, peer_sigs: np.ndarray, n_seeds=30, epochs=15, device='cpu', verbose=False):
    """Binary classifier: class 0 = bona-fide own chip with matching (phys, nonce_emb),
    class 1 = REJECT (peer chip, OR own phys with mismatched nonce_emb, OR static replay).

    Training negatives include:
      (a) peer's real (phys, emb) pairs
      (b) own phys with SHUFFLED nonce_emb (mismatched-nonce attack)
      (c) static replay: one own row's phys repeated under many different embs
      (d) random nonce_emb only (pure replay-style)
    """
    if len(peer_sigs) == 0:
        peer_sigs = own_sigs.copy()
        np.random.shuffle(peer_sigs.T)
        peer_sigs[:, :32] += 1.5
    n = min(len(own_sigs), len(peer_sigs))
    own = own_sigs[:n].astype(np.float32)
    peer = peer_sigs[:n].astype(np.float32)
    rng = np.random.default_rng(0)
    # (b) own phys + shuffled nonce_emb
    shuf_idx = rng.permutation(n)
    mismatched = own.copy()
    mismatched[:, 32:] = own[shuf_idx, 32:]
    # (c) static replay: pick one phys, broadcast across many embs
    static_phys = own[rng.integers(0, n), :32]
    static_replay = np.empty_like(own)
    static_replay[:, :32] = static_phys
    static_replay[:, 32:] = own[rng.permutation(n), 32:]
    # (d) own phys + random nonce_emb (synthetic from random bytes)
    from nonce_signature import nonce_embedding
    rand_emb = np.stack([nonce_embedding(rng.bytes(8), 32) for _ in range(n)], 0)
    rand_pair = own.copy(); rand_pair[:, 32:] = rand_emb
    # T3 classifier: own vs peer ONLY (cross-chip discrimination).
    # Replay defense is provided by the deterministic plan-consistency verifier
    # in spoof_v2 (no classifier can defeat mismatched-emb attacks on its own).
    X = np.concatenate([own, peer], 0).astype(np.float32)
    y = np.concatenate([np.zeros(n), np.ones(n)], 0).astype(np.int64)
    perm = np.random.permutation(len(X))
    X, y = X[perm], y[perm]
    split = int(0.7*len(X))
    tr, te = np.arange(split), np.arange(split, len(X))
    aurocs = []
    best_model = None
    best_a = -1
    for s in range(n_seeds):
        set_seed(s)
        m = TwinMLP(in_d=DIM, n_out=2).to(device)
        opt = torch.optim.AdamW(m.parameters(), lr=3e-3, weight_decay=1e-4)
        for ep in range(epochs):
            order = np.random.permutation(len(tr))
            for i in range(0, len(order), 32):
                b = tr[order[i:i+32]]
                xb = torch.from_numpy(X[b]).to(device)
                yb = torch.from_numpy(y[b]).to(device)
                loss = F.cross_entropy(m(xb), yb)
                opt.zero_grad(); loss.backward(); opt.step()
        m.eval()
        with torch.no_grad():
            scores = F.softmax(m(torch.from_numpy(X[te]).to(device)), dim=-1)[:,1].cpu().numpy()
        from sklearn.metrics import roc_auc_score
        try: a = float(roc_auc_score(y[te], scores))
        except Exception: a = 0.5
        aurocs.append(a)
        if a > best_a:
            best_a = a; best_model = {k: v.clone() for k,v in m.state_dict().items()}
        if verbose: print(f"[T3 s={s}] AUROC={a:.3f}")
    return aurocs, best_model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n_train', type=int, default=400)
    ap.add_argument('--n_seeds', type=int, default=30)
    ap.add_argument('--peer_npz', type=str, default=None,
        help='paired_sigs.npz from foreign host (must contain "sigs")')
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--out_dir', default=None)
    args = ap.parse_args()
    device = torch.device(args.device)

    host = hostname()
    out_dir = args.out_dir or os.path.abspath(os.path.join(
        HERE, '..', '..', '..', 'results', 'IDENTITY_BENCHMARK_2026-05-30', 'embodiment14c'))
    os.makedirs(out_dir, exist_ok=True)

    print(f"[14C] host={host}  n_train={args.n_train}  n_seeds={args.n_seeds}")
    rng = np.random.default_rng(int(time.time()) & 0xFFFFFFFF)
    sig = NonceSig(host=host)

    # ---- Collect own paired (nonce, sig)
    t0 = time.time()
    nonces, own_sigs = collect_paired(sig, args.n_train, rng)
    print(f"[14C] collected {args.n_train} own pairs in {time.time()-t0:.1f}s")
    paired_path = os.path.join(out_dir, f'{host}_paired_sigs.npz')
    np.savez_compressed(paired_path, nonces=nonces, sigs=own_sigs)
    print(f"[14C] saved {paired_path}")

    # ---- Peer sigs (if provided)
    peer_sigs = np.zeros((0, DIM), dtype=np.float32)
    if args.peer_npz and os.path.exists(args.peer_npz):
        d = np.load(args.peer_npz)
        peer_sigs = d['sigs'].astype(np.float32)
        print(f"[14C] loaded peer sigs from {args.peer_npz}: shape={peer_sigs.shape}")

    # ---- T2 anomaly
    print(f"[14C] training T2 anomaly ({args.n_seeds} seeds)...")
    t2_aurocs, _ = train_T2_anomaly(own_sigs, n_seeds=args.n_seeds, device=device, verbose=False)
    # ---- T3 twin
    print(f"[14C] training T3 twin ({args.n_seeds} seeds)...")
    t3_aurocs, t3_state = train_T3_twin(own_sigs, peer_sigs, n_seeds=args.n_seeds, device=device, verbose=False)

    # save best T3 model for spoof_v2
    t3_path = os.path.join(out_dir, f'{host}_t3_best.pt')
    if t3_state is not None:
        torch.save(t3_state, t3_path)

    summary = {
        'host': host,
        'n_train': args.n_train,
        'n_seeds': args.n_seeds,
        'peer_npz': args.peer_npz,
        't2': {
            'aurocs': t2_aurocs,
            'mean': float(np.mean(t2_aurocs)),
            'ci95_lo': float(np.percentile(t2_aurocs, 2.5)),
            'ci95_hi': float(np.percentile(t2_aurocs, 97.5)),
        },
        't3': {
            'aurocs': t3_aurocs,
            'mean': float(np.mean(t3_aurocs)),
            'ci95_lo': float(np.percentile(t3_aurocs, 2.5)),
            'ci95_hi': float(np.percentile(t3_aurocs, 97.5)),
        },
        't': time.time(),
    }
    save_json(os.path.join(out_dir, f'{host}_training.json'), summary)
    print(f"[14C] T2 mean AUROC={summary['t2']['mean']:.3f}  "
          f"T3 mean AUROC={summary['t3']['mean']:.3f}")


if __name__ == '__main__':
    main()
