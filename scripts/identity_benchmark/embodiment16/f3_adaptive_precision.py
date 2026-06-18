"""F3: Adaptive precision/depth via thermal state.

Vanilla:  always fp32 inference.
Embodied: read APU temperature each batch; choose dtype based on temp.
    T < 55C  → fp32 (cool, fine to use full precision)
    T >= 55C → fp16 (warm, save energy/latency)
    T >= 60C → int8 quantised (hot, aggressive)

Pre-reg: At iso-accuracy (within 1pp), embodied throughput (qps) > vanilla
by >= 20% averaged over a workload that swings the chip through both regimes.

We run a fixed inference workload of 200 batches with periodic light heating
(matmul filler) between batches to simulate realistic deployment thermals.
"""
from __future__ import annotations
import os, sys, time, json, argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from _common import thermal_guard, save_json, bootstrap_ci, cool_to, temp_c

DEV = 'cuda' if torch.cuda.is_available() else 'cpu'


def make_ds(n=500, seed=0):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 10, size=n)
    centres = rng.uniform(6, 22, size=(10, 2))
    X = np.zeros((n, 28, 28), dtype=np.float32)
    xx, yy = np.meshgrid(np.arange(28), np.arange(28))
    for i in range(n):
        cy, cx = centres[y[i]]
        bump = 1.4 * np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / 22.0)
        bump += rng.normal(0, 0.12, size=(28, 28))
        X[i] = bump
    X = (X - X.mean()) / (X.std() + 1e-6)
    return X, y


class TinyMLP(nn.Module):
    def __init__(self, hidden=256):
        super().__init__()
        self.fc1 = nn.Linear(28*28, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, 10)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


def train(model, X, y, epochs=6, bs=128, lr=2e-3):
    Xt = torch.from_numpy(X).to(DEV); yt = torch.from_numpy(y).long().to(DEV)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    n = Xt.size(0)
    model.train()
    for ep in range(epochs):
        if ep % 4 == 0:
            thermal_guard()
        perm = torch.randperm(n, device=DEV)
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            logits = model(Xt[idx])
            loss = F.cross_entropy(logits, yt[idx])
            opt.zero_grad(); loss.backward(); opt.step()
    return model


def quantize_int8(model):
    """Crude post-training int8 quant — scale fc weights to int8 range, store scale."""
    q = {}
    for name, p in model.named_parameters():
        if 'weight' in name:
            w = p.data
            s = w.abs().max() / 127.0
            qw = (w / (s + 1e-9)).round().clamp(-128, 127).to(torch.int8)
            q[name] = (qw, s)
    return q


def infer_int8(q, x, model_ref):
    """Manual int8 forward — dequantize on the fly. Cheap & honest."""
    # Reconstruct fp32 weights once from int8 storage (sim. real int8 latency win)
    with torch.no_grad():
        fc1_w = q['fc1.weight'][0].float() * q['fc1.weight'][1]
        fc2_w = q['fc2.weight'][0].float() * q['fc2.weight'][1]
        fc3_w = q['fc3.weight'][0].float() * q['fc3.weight'][1]
        h = F.relu(F.linear(x.view(x.size(0), -1), fc1_w, model_ref.fc1.bias))
        h = F.relu(F.linear(h, fc2_w, model_ref.fc2.bias))
        return F.linear(h, fc3_w, model_ref.fc3.bias)


def warm_burst(ms=80):
    """Brief matmul burst to push chip temperature up."""
    A = torch.randn(512, 512, device=DEV)
    B = torch.randn(512, 512, device=DEV)
    t0 = time.time()
    while (time.time() - t0) * 1000 < ms:
        A = A @ B
    if DEV == 'cuda':
        torch.cuda.synchronize()
    return A.sum().item()


def run_workload(mode, model, q_int8, Xte, yte, n_batches=200, bs=64,
                 max_seconds=60):
    """Run inference workload. 'mode' is 'vanilla' or 'embodied'.

    Between batches we occasionally do a warm_burst to drive chip thermals.
    Returns: (qps, accuracy, dtype_histogram, avg_temp).
    """
    rng = np.random.default_rng(0)
    correct = 0; total = 0
    dtype_hist = {'fp32': 0, 'fp16': 0, 'int8': 0}
    temps = []
    t_start = time.time()
    n_done = 0
    model_fp16 = None
    if mode == 'embodied':
        model_fp16 = TinyMLP(hidden=256).to(DEV).half()
        model_fp16.load_state_dict({k: v.half() for k, v in model.state_dict().items()})
        model_fp16.eval()
    model.eval()
    for b in range(n_batches):
        if (time.time() - t_start) > max_seconds:
            break
        # check thermal safety EVERY batch (lightweight)
        t = temp_c()
        if t >= 66:
            # too hot — pause briefly
            time.sleep(2); continue
        temps.append(t)
        idx = rng.integers(0, Xte.size(0), bs)
        xb = Xte[idx]; yb = yte[idx]
        if mode == 'vanilla':
            with torch.no_grad():
                logits = model(xb)
            dtype_hist['fp32'] += 1
        else:
            # embodied: temp-driven dtype
            if t < 55:
                with torch.no_grad():
                    logits = model(xb)
                dtype_hist['fp32'] += 1
            elif t < 60:
                with torch.no_grad():
                    logits = model_fp16(xb.half()).float()
                dtype_hist['fp16'] += 1
            else:
                logits = infer_int8(q_int8, xb, model)
                dtype_hist['int8'] += 1
        pred = logits.argmax(1)
        correct += (pred == yb).sum().item()
        total += bs
        n_done += 1
        # warm filler every 5 batches to push temps
        if b % 5 == 0:
            warm_burst(ms=40)
    elapsed = time.time() - t_start
    qps = total / max(1e-6, elapsed)
    acc = correct / max(1, total)
    return qps, acc, dtype_hist, float(np.mean(temps)) if temps else 0.0, n_done


def main(n_runs=8):
    print(f"[F3] start, n_runs={n_runs}, device={DEV}, temp={temp_c():.1f}C", flush=True)
    X, y = make_ds(n=2000, seed=42)
    Xtr, ytr = X[:1500], y[:1500]
    Xte_np, yte_np = X[1500:], y[1500:]
    torch.manual_seed(0)
    model = TinyMLP(hidden=256).to(DEV)
    print(f"[F3] training model...", flush=True)
    train(model, Xtr, ytr, epochs=4)
    print(f"[F3] quantizing...", flush=True)
    q_int8 = quantize_int8(model)
    print(f"[F3] starting runs, temp={temp_c():.1f}C", flush=True)
    Xte = torch.from_numpy(Xte_np).to(DEV); yte = torch.from_numpy(yte_np).long().to(DEV)

    runs = {'vanilla': [], 'embodied': []}
    t_start = time.time()
    for r in range(n_runs):
        if (time.time() - t_start) > 420:
            print(f"[F3] time budget hit", flush=True)
            break
        # vanilla
        cool_to(target_c=53, max_wait=60)
        thermal_guard()
        qv, av, hv, tv, nv = run_workload('vanilla', model, q_int8, Xte, yte,
                                           n_batches=80, bs=64, max_seconds=18)
        # embodied
        cool_to(target_c=53, max_wait=60)
        thermal_guard()
        qe, ae, he, te, ne = run_workload('embodied', model, q_int8, Xte, yte,
                                           n_batches=80, bs=64, max_seconds=18)
        runs['vanilla'].append({'qps': qv, 'acc': av, 'hist': hv, 'avg_temp': tv, 'n': nv})
        runs['embodied'].append({'qps': qe, 'acc': ae, 'hist': he, 'avg_temp': te, 'n': ne})
        print(f"[F3] run {r}: VAN qps={qv:.1f} acc={av:.3f} T={tv:.1f}  "
              f"EMB qps={qe:.1f} acc={ae:.3f} T={te:.1f} hist={he}", flush=True)

    van_qps = [r['qps'] for r in runs['vanilla']]
    emb_qps = [r['qps'] for r in runs['embodied']]
    van_acc = [r['acc'] for r in runs['vanilla']]
    emb_acc = [r['acc'] for r in runs['embodied']]
    ratios = [e/v for e, v in zip(emb_qps, van_qps)]
    acc_gap = [e - v for e, v in zip(emb_acc, van_acc)]
    rmean, rlo, rhi = bootstrap_ci(ratios)
    amean, alo, ahi = bootstrap_ci(acc_gap)
    # pre-reg: qps ratio >= 1.20 AND |acc_gap| <= 0.01
    iso_acc = abs(amean) <= 0.01
    qps_pass = rlo > 1.20
    summary = {
        'n_runs': len(ratios),
        'qps_ratio': {'mean': rmean, 'ci95': [rlo, rhi]},
        'acc_gap': {'mean': amean, 'ci95': [alo, ahi]},
        'iso_acc_satisfied': bool(iso_acc),
        'qps_gate_passed': bool(qps_pass),
        'all_pass': bool(iso_acc and qps_pass),
        'runs': runs,
        'gate': {
            'criterion': 'qps_ratio CI lower > 1.20 AND |acc_gap| <= 0.01',
            'qps_ratio_mean': rmean,
            'qps_ratio_ci95': [rlo, rhi],
            'acc_gap_mean_pp': amean*100,
            'pass': bool(iso_acc and qps_pass),
        }
    }
    save_json('f3_adaptive_precision.json', summary)
    print(json.dumps(summary['gate'], indent=2))
    return summary


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--runs', type=int, default=8)
    args = ap.parse_args()
    main(n_runs=args.runs)
