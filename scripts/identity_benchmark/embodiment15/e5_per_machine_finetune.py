"""E5: Per-machine emergent fine-tuning via substrate-conditioned vocabulary prior.

Hypothesis: the chip substrate carries a (host-specific) "workload signature"
that, when used as a conditioning vector for a tiny LM's vocabulary prior, lowers
test perplexity on a user-specific corpus, relative to a vanilla LM with no such
conditioning.

Synthetic user-corpus: we simulate the user as preferring 1 of 4 vocabularies
("scientific", "casual", "code", "shopping"). We CHEAT-FAIR: use chip state as
a conditioning vector — we hash it to a stable per-host bias on the vocabulary
prior. The vanilla model has no such bias.

Variants:
  A vanilla   — tiny LM, no conditioning
  B embodied  — same LM + learnable projection from chip signature -> vocab prior
  C random    — same LM + projection from RANDOM vector of same dimension
Pre-reg: embodied PPL < vanilla PPL by >=3% AND embodied PPL < random PPL by >=1%.
"""
from __future__ import annotations
import os, sys, time, json, argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from _common import thermal_guard, save_json, bootstrap_ci, diff_ci

sys.path.insert(0, os.path.join(HERE, '..', 'embodiment14b'))
from signature_live import LiveSig

DEV = 'cuda' if torch.cuda.is_available() else 'cpu'


def make_user_corpus(seed=42, n_seq=4000, seq_len=24, vocab=128, n_persona=4):
    """Each persona has a 'preferred' subset of tokens (size ~24). User picks ONE
    persona; their sequences over-sample those tokens with prob 0.6. Test set
    same persona."""
    rng = np.random.default_rng(seed)
    personas_tokens = [rng.choice(vocab, size=24, replace=False) for _ in range(n_persona)]
    user_persona = 0  # deterministic
    user_tokens = personas_tokens[user_persona]
    X = rng.integers(0, vocab, size=(n_seq, seq_len))
    mask = rng.random((n_seq, seq_len)) < 0.6
    X[mask] = rng.choice(user_tokens, size=mask.sum())
    return X, personas_tokens, user_persona


class TinyLM(nn.Module):
    """LM that predicts next token; optional conditioning bias on output logits."""
    def __init__(self, vocab=128, d=64, seq=24, cond_dim=0):
        super().__init__()
        self.emb = nn.Embedding(vocab, d)
        self.pos = nn.Parameter(torch.randn(seq, d) * 0.02)
        self.layer = nn.TransformerEncoderLayer(
            d_model=d, nhead=4, dim_feedforward=128,
            batch_first=True, dropout=0.0,
            activation='gelu'
        )
        # causal-like by simple subseq mask
        self.head = nn.Linear(d, vocab)
        self.cond_dim = cond_dim
        if cond_dim > 0:
            self.cond_proj = nn.Linear(cond_dim, vocab)

    def forward(self, x, cond=None):
        h = self.emb(x) + self.pos[:x.size(1)]
        # simple causal mask
        S = x.size(1)
        mask = torch.triu(torch.ones(S, S, device=x.device), diagonal=1).bool()
        h = self.layer(h, src_mask=mask)
        logits = self.head(h)
        if cond is not None and self.cond_dim > 0:
            bias = self.cond_proj(cond)  # (vocab,)
            logits = logits + bias[None, None, :]
        return logits


def loss_for(model, X, cond, bs=64):
    """LM cross-entropy on next-token (token t predicts token t+1)."""
    losses = []
    n = X.size(0)
    for i in range(0, n, bs):
        xb = X[i:i + bs]
        logits = model(xb[:, :-1], cond=cond)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                               xb[:, 1:].reshape(-1), reduction='mean')
        losses.append(loss.item() * (xb.size(0)))
    return sum(losses) / n


def train_lm(variant, X_train, X_test, cond_vec_train, cond_vec_test,
             cond_dim, epochs=8, bs=64, lr=3e-3, seed=0):
    torch.manual_seed(seed)
    model = TinyLM(cond_dim=cond_dim).to(DEV)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    n = X_train.size(0)
    for ep in range(epochs):
        thermal_guard()
        perm = torch.randperm(n, device=DEV)
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            xb = X_train[idx]
            logits = model(xb[:, :-1], cond=cond_vec_train)
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                                    xb[:, 1:].reshape(-1))
            opt.zero_grad(); loss.backward(); opt.step()
    # test PPL
    model.eval()
    with torch.no_grad():
        test_ce = loss_for(model, X_test, cond_vec_test, bs=128)
    return float(np.exp(test_ce)), test_ce


def main(seeds=15):
    print(f"[E5] starting, seeds={seeds}")
    sig = LiveSig()

    # Build user corpus
    X, personas, persona = make_user_corpus()
    n = X.shape[0]
    split = int(0.8 * n)
    X_train = torch.from_numpy(X[:split]).long().to(DEV)
    X_test  = torch.from_numpy(X[split:]).long().to(DEV)

    # Embodied conditioning: average of many live reads → stable per-host vector
    # (this is what "the chip carries the user pattern" means)
    print("[E5] sampling chip conditioning vector (96 reads)…")
    emb_reads = []
    for _ in range(96):
        emb_reads.append(sig.read())
        time.sleep(0.005)
    emb_cond_np = np.mean(emb_reads, axis=0).astype(np.float32)
    print(f"[E5] emb_cond norm={np.linalg.norm(emb_cond_np):.3f}")

    cond_dim = 32
    emb_cond = torch.from_numpy(emb_cond_np).to(DEV)

    results = {'vanilla': [], 'embodied': [], 'random': []}
    for s in range(seeds):
        thermal_guard(verbose=True)
        # vanilla
        ppl_v, _ = train_lm('vanilla', X_train, X_test, None, None,
                            cond_dim=0, seed=s)
        # embodied
        ppl_e, _ = train_lm('embodied', X_train, X_test, emb_cond, emb_cond,
                            cond_dim=cond_dim, seed=s)
        # random with matched norm
        rng = np.random.default_rng(1000 + s)
        rcond_np = rng.standard_normal(cond_dim).astype(np.float32)
        rcond_np *= (np.linalg.norm(emb_cond_np) / (np.linalg.norm(rcond_np) + 1e-6))
        rcond = torch.from_numpy(rcond_np).to(DEV)
        ppl_r, _ = train_lm('random', X_train, X_test, rcond, rcond,
                            cond_dim=cond_dim, seed=s)

        results['vanilla'].append(ppl_v)
        results['embodied'].append(ppl_e)
        results['random'].append(ppl_r)
        print(f"[E5] seed {s}: van PPL={ppl_v:.3f}  emb PPL={ppl_e:.3f}  "
              f"rnd PPL={ppl_r:.3f}", flush=True)

    # gate: relative reductions
    rel_emb_vs_van = [(v - e) / v for v, e in zip(results['vanilla'], results['embodied'])]
    rel_emb_vs_rnd = [(r - e) / r for r, e in zip(results['random'], results['embodied'])]
    m_v, lo_v, hi_v = bootstrap_ci(rel_emb_vs_van, seed=0)
    m_r, lo_r, hi_r = bootstrap_ci(rel_emb_vs_rnd, seed=0)
    summary = {
        'ppl': {k: {'mean': float(np.mean(v)), 'values': v} for k, v in results.items()},
        'rel_reduction_emb_vs_vanilla': {'mean_pct': m_v * 100, 'ci95_pct': [lo_v * 100, hi_v * 100]},
        'rel_reduction_emb_vs_random':  {'mean_pct': m_r * 100, 'ci95_pct': [lo_r * 100, hi_r * 100]},
        'emb_cond_norm': float(np.linalg.norm(emb_cond_np)),
        'gate': {
            'criterion': 'emb 3% lower PPL than van AND 1% lower than rnd (CIs exclude 0)',
            'pass': bool(lo_v > 0 and m_v >= 0.03 and lo_r > 0 and m_r >= 0.01),
            'emb_vs_van_pct': m_v * 100,
            'emb_vs_rnd_pct': m_r * 100,
        }
    }
    save_json('e5_per_machine_finetune.json', summary)
    print(json.dumps(summary['gate'], indent=2))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', type=int, default=15)
    main(ap.parse_args().seeds)
