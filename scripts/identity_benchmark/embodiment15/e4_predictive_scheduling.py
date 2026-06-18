"""E4: Predictive self-scheduling — reorder a batch to minimize total latency.

Hypothesis: a tiny "self-latency predictor" trained on chip state + input shape
can reorder a heterogeneous inference batch so that GPU+CPU work overlaps better
than processing in arrival order — yielding lower batch wall-clock latency.

Variants:
  A vanilla   — process 32 inference requests in arrival order
  B embodied  — predict latency for each from (chip-state, input-len); sort
                long-first (FIFO with bin-packing) using PRED that integrates
                chip telemetry
  C oracle    — use true measured per-request latency (upper bound)
  D random    — shuffle randomly (control)
Pre-reg: embodied total latency < vanilla by >= 10% (CI lower >0%).
"""
from __future__ import annotations
import os, sys, time, json, argparse
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from _common import thermal_guard, save_json, bootstrap_ci, diff_ci, temp_c

sys.path.insert(0, os.path.join(HERE, '..', 'embodiment14b'))
from signature_live import LiveSig

DEV = 'cuda' if torch.cuda.is_available() else 'cpu'


class TinyServingModel(nn.Module):
    """Stand-in for an inference model whose runtime grows with seq length."""
    def __init__(self, d=64, vocab=128):
        super().__init__()
        self.emb = nn.Embedding(vocab, d)
        self.layer = nn.TransformerEncoderLayer(
            d_model=d, nhead=4, dim_feedforward=128,
            batch_first=True, dropout=0.0).to(DEV)

    def run(self, seq_len):
        # tiny forward pass at given seq_len
        x = torch.randint(0, 128, (1, seq_len), device=DEV)
        h = self.emb(x)
        h = self.layer(h)
        return h.sum().item()


def make_requests(n=32, min_len=4, max_len=128, seed=0):
    rng = np.random.default_rng(seed)
    return rng.integers(min_len, max_len + 1, size=n).tolist()


def process(model, order):
    """Run requests in given order; return total wall time."""
    t0 = time.perf_counter()
    for s in order:
        model.run(int(s))
    return time.perf_counter() - t0


def measure_oracle_latencies(model, lengths, repeats=2):
    """Per-request measured wall-clock (averaged) — used for the oracle policy."""
    lats = []
    for L in lengths:
        ts = []
        for _ in range(repeats):
            t0 = time.perf_counter()
            model.run(int(L))
            ts.append(time.perf_counter() - t0)
        lats.append(np.median(ts))
    return np.array(lats)


def train_predictor(model, sig, n_train=300, seed=0):
    """Train a tiny MLP to predict per-request latency from (seq_len, chip_state).
    Inputs: [seq_len_norm, sig_32...]  ->  predicted ms.
    """
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    Xs, ys = [], []
    for _ in range(n_train):
        L = int(rng.integers(4, 129))
        sv = sig.read()  # 32-d
        t0 = time.perf_counter()
        model.run(L)
        dt = (time.perf_counter() - t0) * 1000  # ms
        Xs.append(np.concatenate([[L / 128.0], sv]).astype(np.float32))
        ys.append(dt)
        if len(ys) % 50 == 0:
            thermal_guard()
    Xs = torch.from_numpy(np.stack(Xs)).to(DEV)
    ys = torch.from_numpy(np.array(ys, dtype=np.float32)).to(DEV)
    net = nn.Sequential(nn.Linear(33, 32), nn.ReLU(), nn.Linear(32, 1)).to(DEV)
    opt = torch.optim.Adam(net.parameters(), lr=3e-3)
    for ep in range(80):
        pred = net(Xs).squeeze(-1)
        loss = F.smooth_l1_loss(pred, ys)
        opt.zero_grad(); loss.backward(); opt.step()
    return net


def predict_latencies(net, sig, lengths):
    sv = sig.read()
    out = []
    for L in lengths:
        x = torch.from_numpy(np.concatenate([[L / 128.0], sv]).astype(np.float32)).to(DEV)
        with torch.no_grad():
            out.append(net(x).item())
    return np.array(out)


def main(seeds=8):
    print(f"[E4] starting, seeds={seeds}")
    sig = LiveSig()
    model = TinyServingModel().to(DEV)
    # warmup
    for _ in range(20):
        model.run(64)

    # Train one predictor per session — sees enough chip variance
    thermal_guard(verbose=True)
    net = train_predictor(model, sig, n_train=250)

    rows = []
    for s in range(seeds):
        thermal_guard(verbose=True)
        lengths = make_requests(n=32, seed=s)

        # vanilla — arrival order
        t_v = process(model, lengths)

        # oracle — measured per-request latencies sorted descending (longest-first)
        oracle_lat = measure_oracle_latencies(model, lengths, repeats=2)
        order_o = list(np.argsort(-oracle_lat))
        seq_o = [lengths[i] for i in order_o]
        t_o = process(model, seq_o)

        # embodied — predicted latencies sorted descending
        pred_lat = predict_latencies(net, sig, lengths)
        order_e = list(np.argsort(-pred_lat))
        seq_e = [lengths[i] for i in order_e]
        t_e = process(model, seq_e)

        # random
        rng = np.random.default_rng(s + 999)
        order_r = list(rng.permutation(len(lengths)))
        seq_r = [lengths[i] for i in order_r]
        t_r = process(model, seq_r)

        rows.append({'seed': s,
                     'vanilla_s': t_v, 'embodied_s': t_e,
                     'oracle_s': t_o, 'random_s': t_r,
                     'pred_corr_with_oracle': float(np.corrcoef(pred_lat, oracle_lat)[0, 1])})
        print(f"[E4] seed {s}: van={t_v*1e3:.1f}ms  emb={t_e*1e3:.1f}ms  "
              f"orc={t_o*1e3:.1f}ms  rnd={t_r*1e3:.1f}ms  "
              f"corr={rows[-1]['pred_corr_with_oracle']:.3f}", flush=True)

    # Gate on (vanilla - embodied) / vanilla
    rels = [(r['vanilla_s'] - r['embodied_s']) / r['vanilla_s'] for r in rows]
    rmean, rlo, rhi = bootstrap_ci(rels, seed=0)
    summary = {
        'rows': rows,
        'rel_improvement_embodied_over_vanilla': {'mean': rmean, 'ci95': [rlo, rhi]},
        'gate': {
            'criterion': 'embodied reduces latency by >= 10% (CI lower > 0)',
            'rel_improvement_mean_pct': rmean * 100,
            'ci95_pct': [rlo * 100, rhi * 100],
            'pass': bool(rlo > 0 and rmean >= 0.10),
        }
    }
    save_json('e4_predictive_scheduling.json', summary)
    print(json.dumps(summary['gate'], indent=2))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', type=int, default=8)
    main(ap.parse_args().seeds)
