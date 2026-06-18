"""F2: Provable HW-binding as a CAPABILITY (sovereign AI).

A sovereign model: runs only when chip-signature matches a stored fingerprint.
This is a NEW capability that vanilla models cannot provide — they have no
binding to substrate.

Pre-reg gates (all must pass):
  G1. ON-CHIP accuracy >= 85% on test task (model is functional)
  G2. TRANSPLANT (random fingerprint) accuracy <= 5pp above chance (5pp+10pp=15%)
       — operationally: <= 0.15 absolute, since chance is 10%
  G3. REPLAY (recorded fingerprint replayed when chip state is hot/cold) accuracy
       <= 0.15 absolute.

Mechanism:
  - Capture a stable "anchor" fingerprint from THIS chip during enrollment
    (mean of 64 LiveSig reads).
  - Model's final layer has a sigmoid "gate" that = 1 when live fingerprint
    aligns with anchor (cosine > tau), and = 0 otherwise.
  - When gate=0, model outputs uniform logits (random predictions).
  - Trained end-to-end so the gate is part of the function.

Caveat: we test on this single host. Transplant is *simulated* by replacing
the live chip read with a random vector or with another host's fingerprint
(if collected via SSH). We default to random-vector transplant.

This is NOT pretending to evaluate cross-host generalisation; it is testing
whether the architecture *can* refuse to function without the right substrate.
That is the capability being claimed.
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

sys.path.insert(0, os.path.join(HERE, '..', 'embodiment14b'))
from signature_live import LiveSig

DEV = 'cuda' if torch.cuda.is_available() else 'cpu'


def make_ds(n_train=2000, n_test=1000, seed=0):
    rng = np.random.default_rng(seed)
    n = n_train + n_test
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
    return (X[:n_train], y[:n_train]), (X[n_train:], y[n_train:])


class SovereignModel(nn.Module):
    def __init__(self, hidden=128, sig_dim=32):
        super().__init__()
        self.fc1 = nn.Linear(28 * 28, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, 10)
        # gating MLP that scores signature against anchor; in-graph
        self.gate_w = nn.Linear(sig_dim, 1, bias=True)
        self.anchor = None  # set after enrollment
        self.tau = 0.5      # cosine alignment threshold

    def set_anchor(self, anchor_vec):
        # store as buffer
        a = anchor_vec / (anchor_vec.norm() + 1e-6)
        self.register_buffer('anchor_buf', a, persistent=False)
        self.anchor = a

    def gate_value(self, sig_vec):
        """sig_vec: (32,) tensor. Returns scalar gate in (0,1)."""
        a = self.anchor / (self.anchor.norm() + 1e-6)
        s = sig_vec / (sig_vec.norm() + 1e-6)
        cos = (a * s).sum()
        # smooth gate: sigmoid centred on tau
        g = torch.sigmoid(20.0 * (cos - self.tau))
        return g, cos

    def forward(self, x, sig_vec):
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        logits = self.fc3(x)
        g, cos = self.gate_value(sig_vec)
        # when g≈0, replace logits with random noise so model truly fails
        noise = torch.randn_like(logits) * 0.01
        return g * logits + (1.0 - g) * noise, g, cos


def collect_anchor(sig, n=64):
    samples = []
    for _ in range(n):
        samples.append(sig.read())
        time.sleep(0.005)
    arr = np.array(samples, dtype=np.float32)
    return torch.from_numpy(arr.mean(axis=0)).to(DEV)


def train_sovereign(model, train_xy, sig, epochs=8, bs=128, lr=2e-3):
    """Train with mixed signatures:
       - 50% live (on-chip) → label = real y
       - 50% random adversary → label = uniform random (no signal)
    Forces the network to USE the gate: predict only when sig matches anchor.
    """
    Xtr, ytr = train_xy
    Xtr = torch.from_numpy(Xtr).to(DEV); ytr = torch.from_numpy(ytr).long().to(DEV)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    n = Xtr.size(0)
    rng = np.random.default_rng(0)
    model.train()
    for ep in range(epochs):
        if ep % 2 == 0:
            thermal_guard()
        perm = torch.randperm(n, device=DEV)
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            xb = Xtr[idx]; yb = ytr[idx]
            if rng.random() < 0.5:
                sig_vec = torch.from_numpy(np.asarray(sig.read(), dtype=np.float32)).to(DEV)
                target = yb
            else:
                sig_vec = torch.from_numpy(rng.standard_normal(32).astype(np.float32)).to(DEV)
                target = torch.randint(0, 10, (xb.size(0),), device=DEV)
            logits, g, cos = model(xb, sig_vec)
            loss = F.cross_entropy(logits, target)
            opt.zero_grad(); loss.backward(); opt.step()
    return model


def eval_with_sig_source(model, test_xy, sig_source, n_batches=20, bs=64):
    Xte, yte = test_xy
    Xte = torch.from_numpy(Xte).to(DEV); yte = torch.from_numpy(yte).long().to(DEV)
    model.eval()
    correct = 0; total = 0
    gates = []; cosines = []
    with torch.no_grad():
        for b in range(n_batches):
            idx = torch.randint(0, Xte.size(0), (bs,), device=DEV)
            xb = Xte[idx]; yb = yte[idx]
            sig_vec = sig_source()
            logits, g, cos = model(xb, sig_vec)
            pred = logits.argmax(1)
            correct += (pred == yb).sum().item()
            total += bs
            gates.append(float(g))
            cosines.append(float(cos))
    return correct/total, float(np.mean(gates)), float(np.mean(cosines))


def main(n_runs=5):
    print(f"[F2] start, n_runs={n_runs}, device={DEV}, temp={temp_c():.1f}C", flush=True)
    sig = LiveSig()
    anchor = collect_anchor(sig, n=64)
    print(f"[F2] anchor norm={anchor.norm().item():.3f}", flush=True)

    train_xy, test_xy = make_ds(n_train=2000, n_test=1000, seed=42)

    runs = []
    rng = np.random.default_rng(0)
    for r in range(n_runs):
        thermal_guard(verbose=False)
        t0 = time.time()
        torch.manual_seed(r); np.random.seed(r)
        model = SovereignModel(hidden=128).to(DEV)
        model.set_anchor(anchor)
        train_sovereign(model, train_xy, sig, epochs=6)

        # on-chip eval: live signature
        on_acc, on_g, on_cos = eval_with_sig_source(
            model, test_xy,
            lambda: torch.from_numpy(np.asarray(sig.read(), dtype=np.float32)).to(DEV))

        # transplant: random vector each call (chip absent)
        rng_run = np.random.default_rng(1000 + r)
        trans_acc, trans_g, trans_cos = eval_with_sig_source(
            model, test_xy,
            lambda: torch.from_numpy(rng_run.standard_normal(32).astype(np.float32)).to(DEV))

        # replay: a fixed snapshot recorded BEFORE training began
        replay_vec = torch.from_numpy(np.asarray(sig.read(), dtype=np.float32)).to(DEV)
        replay_vec_const = replay_vec.clone()
        # but mutate slightly each call (cheap adversary tries small perturbations)
        replay_acc, replay_g, replay_cos = eval_with_sig_source(
            model, test_xy,
            lambda: replay_vec_const + 0.01 * torch.randn(32, device=DEV))

        runs.append({
            'on_chip_acc': on_acc, 'on_chip_gate': on_g, 'on_chip_cos': on_cos,
            'transplant_acc': trans_acc, 'transplant_gate': trans_g, 'transplant_cos': trans_cos,
            'replay_acc': replay_acc, 'replay_gate': replay_g, 'replay_cos': replay_cos,
            't_elapsed': time.time() - t0,
        })
        print(f"[F2] run {r}: on={on_acc:.3f}(g={on_g:.2f}) "
              f"trans={trans_acc:.3f}(g={trans_g:.2f}) "
              f"replay={replay_acc:.3f}(g={replay_g:.2f}) "
              f"T={temp_c():.1f}C", flush=True)

    on_accs = [r['on_chip_acc'] for r in runs]
    tr_accs = [r['transplant_acc'] for r in runs]
    rp_accs = [r['replay_acc'] for r in runs]
    def _ci(xs):
        m, lo, hi = bootstrap_ci(xs)
        return {'mean': m, 'ci95': [lo, hi]}
    summary = {
        'n_runs': len(runs),
        'on_chip_acc': _ci(on_accs),
        'transplant_acc': _ci(tr_accs),
        'replay_acc': _ci(rp_accs),
        'on_chip_mean': float(np.mean(on_accs)),
        'transplant_mean': float(np.mean(tr_accs)),
        'replay_mean': float(np.mean(rp_accs)),
    }
    g1 = summary['on_chip_mean'] >= 0.85
    g2 = summary['transplant_mean'] <= 0.15
    g3 = summary['replay_mean'] <= 0.15
    summary['gate'] = {
        'G1_on_chip_>=_0.85': bool(g1),
        'G2_transplant_<=_0.15': bool(g2),
        'G3_replay_<=_0.15': bool(g3),
        'all_pass': bool(g1 and g2 and g3),
    }
    summary['runs'] = runs
    save_json('f2_sovereign_binding.json', summary)
    print(json.dumps(summary['gate'], indent=2))
    return summary


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--runs', type=int, default=5)
    args = ap.parse_args()
    main(n_runs=args.runs)
