"""E3: Adaptive inference budget via thermal headroom.

Hypothesis: a model that skips layer-4 when APU is hot can deliver MORE
throughput at the same average accuracy, because skipping is automatically
correlated with the periods when sustained throughput matters most.

Setup: pretrained 4-layer tiny transformer on the E2 synthetic task. We
measure inference throughput (queries/sec) at a fixed accuracy threshold.

Note: we make the test workload thermally interesting — repeated long bursts
that actually heat the chip. In an idealized "always cool" benchmark, skipping
just hurts accuracy; the real chip provides a regime where skipping helps.

Variants:
  A vanilla   — always 4 layers
  B embodied  — skip layer-4 if temp_c > THRESHOLD (live read)
  C random    — skip layer-4 with fixed probability matched to embodied skip rate
Pre-reg: embodied throughput / vanilla throughput > 1.15 at acc >= 0.85
         AND embodied acc within 1pp of vanilla.
"""
from __future__ import annotations
import os, sys, time, json, argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from _common import thermal_guard, save_json, temp_c, bootstrap_ci, diff_ci
from e2_attention_bias import make_text

DEV = 'cuda' if torch.cuda.is_available() else 'cpu'


class TinyXFM(nn.Module):
    def __init__(self, vocab=64, d=64, n_class=4, seq=32, n_layer=4):
        super().__init__()
        self.emb = nn.Embedding(vocab, d)
        self.pos = nn.Parameter(torch.randn(seq, d) * 0.02)
        self.layers = nn.ModuleList([nn.TransformerEncoderLayer(
            d_model=d, nhead=4, dim_feedforward=128, batch_first=True,
            dropout=0.0) for _ in range(n_layer)])
        self.out = nn.Linear(d, n_class)

    def forward(self, x, n_active=None):
        h = self.emb(x) + self.pos
        n_active = n_active if n_active is not None else len(self.layers)
        for i, lyr in enumerate(self.layers):
            if i >= n_active:
                break
            h = lyr(h)
        return self.out(h.mean(dim=1))


def train_full(model, train_xy, epochs=8, bs=64, lr=2e-3, seed=0):
    torch.manual_seed(seed)
    Xtr, ytr = train_xy
    Xtr = torch.from_numpy(Xtr).long().to(DEV); ytr = torch.from_numpy(ytr).long().to(DEV)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    n = Xtr.size(0)
    for ep in range(epochs):
        thermal_guard()
        perm = torch.randperm(n, device=DEV)
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            logits = model(Xtr[idx])
            loss = F.cross_entropy(logits, ytr[idx])
            opt.zero_grad(); loss.backward(); opt.step()
    return model


def eval_throughput(model, Xte, yte, policy, n_steps=200, bs=64, threshold=58.0):
    """Run inference for n_steps mini-batches; measure qps + acc + skip rate.
    policy: 'always4', 'embodied', 'random_match'."""
    model.eval()
    n = len(Xte)
    rng = np.random.default_rng(0)

    # first pass for embodied: collect skip rate
    # warm-up: a few inferences at full and reduced depth (untimed)
    for _w in range(5):
        idx = rng.integers(0, n, size=bs)
        xb = torch.from_numpy(Xte[idx]).long().to(DEV)
        with torch.no_grad():
            _ = model(xb, n_active=4)
            _ = model(xb, n_active=3)
    if policy == 'embodied' or policy == 'random_match':
        thermal_guard()
    skip_count = 0
    correct = 0
    total = 0
    t0 = time.time()
    skip_prob = 0.0  # populated by embodied
    for step in range(n_steps):
        idx = rng.integers(0, n, size=bs)
        xb = torch.from_numpy(Xte[idx]).long().to(DEV)
        yb = torch.from_numpy(yte[idx]).long().to(DEV)
        if policy == 'always4':
            n_act = 4
        elif policy == 'embodied':
            t = temp_c()
            n_act = 3 if t > threshold else 4
        else:  # random_match
            n_act = 3 if rng.random() < skip_prob else 4
        with torch.no_grad():
            logits = model(xb, n_active=n_act)
        correct += (logits.argmax(1) == yb).sum().item()
        total += bs
        if n_act < 4:
            skip_count += 1
        if step % 20 == 0:
            thermal_guard()
    elapsed = time.time() - t0
    qps = total / elapsed
    acc = correct / total
    return qps, acc, skip_count / n_steps


def main(seeds=8):
    print(f"[E3] starting, seeds={seeds}")
    train_xy, test_xy = make_text(n_train=2000, n_test=1000, seq=32, seed=42)
    Xte, yte = test_xy

    rows = []
    for s in range(seeds):
        thermal_guard(verbose=True)
        torch.manual_seed(s)
        model = TinyXFM().to(DEV)
        model = train_full(model, train_xy, seed=s)

        # warmup — eliminate one-shot CUDA-init cost
        for _ in range(20):
            xb = torch.from_numpy(Xte[:64]).long().to(DEV)
            with torch.no_grad():
                _ = model(xb, n_active=4)

        # Counterbalance order to avoid systematic warmup bias (van first or emb first)
        van_first = (s % 2 == 0)
        if van_first:
            qps_v, acc_v, _ = eval_throughput(model, Xte, yte, policy='always4', n_steps=150)
            qps_e, acc_e, skip_e = eval_throughput(model, Xte, yte, policy='embodied',
                                                     n_steps=150, threshold=58.0)
        else:
            qps_e, acc_e, skip_e = eval_throughput(model, Xte, yte, policy='embodied',
                                                     n_steps=150, threshold=58.0)
            qps_v, acc_v, _ = eval_throughput(model, Xte, yte, policy='always4', n_steps=150)
        # random_match — uses same skip_prob as embodied
        # rerun eval with skip_prob set
        model.eval()
        n_steps = 150; bs = 64
        rng = np.random.default_rng(s + 1000)
        correct = 0; total = 0; t0 = time.time()
        for step in range(n_steps):
            idx = rng.integers(0, len(Xte), size=bs)
            xb = torch.from_numpy(Xte[idx]).long().to(DEV)
            yb = torch.from_numpy(yte[idx]).long().to(DEV)
            n_act = 3 if rng.random() < skip_e else 4
            with torch.no_grad():
                logits = model(xb, n_active=n_act)
            correct += (logits.argmax(1) == yb).sum().item()
            total += bs
            if step % 20 == 0:
                thermal_guard()
        qps_r = total / (time.time() - t0)
        acc_r = correct / total

        rows.append({'seed': s, 'qps_vanilla': qps_v, 'acc_vanilla': acc_v,
                     'qps_embodied': qps_e, 'acc_embodied': acc_e, 'skip_e': skip_e,
                     'qps_random': qps_r, 'acc_random': acc_r})
        print(f"[E3] seed {s}: van qps={qps_v:.0f} acc={acc_v:.3f} | "
              f"emb qps={qps_e:.0f} acc={acc_e:.3f} skip={skip_e:.2f} | "
              f"rnd qps={qps_r:.0f} acc={acc_r:.3f}", flush=True)

    qv = [r['qps_vanilla'] for r in rows]
    qe = [r['qps_embodied'] for r in rows]
    av = [r['acc_vanilla'] for r in rows]
    ae = [r['acc_embodied'] for r in rows]

    ratios = [e / v for e, v in zip(qe, qv)]
    rmean, rlo, rhi = bootstrap_ci(ratios, seed=0)
    accs_v_mean, _, _ = bootstrap_ci(av)
    accs_e_mean, _, _ = bootstrap_ci(ae)
    summary = {
        'rows': rows,
        'qps_ratio_embodied_over_vanilla': {'mean': rmean, 'ci95': [rlo, rhi]},
        'mean_acc_vanilla': accs_v_mean,
        'mean_acc_embodied': accs_e_mean,
        'gate': {
            'criterion': 'qps_ratio >= 1.15 AND acc(emb) within 1pp of acc(van)',
            'qps_ratio_mean': rmean,
            'qps_ratio_ci95': [rlo, rhi],
            'acc_gap_pp': (accs_v_mean - accs_e_mean) * 100,
            'pass': bool(rlo >= 1.15 and abs(accs_v_mean - accs_e_mean) < 0.01),
        }
    }
    save_json('e3_thermal_inference_budget.json', summary)
    print(json.dumps(summary['gate'], indent=2))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', type=int, default=8)
    main(ap.parse_args().seeds)
