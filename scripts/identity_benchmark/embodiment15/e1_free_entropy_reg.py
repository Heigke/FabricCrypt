"""E1: Free entropy regularization — chip jitter vs synthetic noise.

Hypothesis: chip jitter (TSC variance, RAPL fluctuations) provides BETTER
multiplicative noise regularization than synthetic random with matched variance,
on a label-noise classification task.

Setup: Tiny MLP on synthetic MNIST-like dataset (28x28, 10 classes, 10% label noise).
Variants:
  A vanilla   — dropout p=0.1
  B embodied  — multiplicative noise from live chip jitter samples
  C synthetic — multiplicative gaussian noise with SAME std as embodied
Pre-reg gate: B test_acc > C test_acc by >= 1.5pp (bootstrap 95% CI excludes 0).
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


def make_dataset(n_train=1500, n_test=2000, label_noise=0.30, seed=0):
    """Harder synthetic 28x28 dataset to expose regularization effects.

    - 10 classes, but feature signal is weak (small bumps + much background noise)
    - 30% label noise on training data forces regularization differences
    - SMALL train set (1500) → overfitting is easy → regularizer matters
    """
    rng = np.random.default_rng(seed)
    n = n_train + n_test
    y = rng.integers(0, 10, size=n)
    centres = rng.uniform(6, 22, size=(10, 2))
    X = np.zeros((n, 28, 28), dtype=np.float32)
    xx, yy = np.meshgrid(np.arange(28), np.arange(28))
    for i in range(n):
        cy, cx = centres[y[i]]
        # weak signal
        bump = 0.6 * np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / 30.0)
        # strong background noise
        bump += rng.normal(0, 0.7, size=(28, 28))
        X[i] = bump
    X = (X - X.mean()) / (X.std() + 1e-6)
    y_noisy = y.copy()
    flip_mask = rng.random(n_train) < label_noise
    y_noisy[:n_train][flip_mask] = rng.integers(0, 10, size=flip_mask.sum())
    return (X[:n_train], y_noisy[:n_train]), (X[n_train:], y[n_train:])


class TinyMLP(nn.Module):
    def __init__(self, hidden=128, p_drop=0.1, noise_mode='dropout'):
        super().__init__()
        self.fc1 = nn.Linear(28 * 28, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, 10)
        self.p_drop = p_drop
        self.noise_mode = noise_mode  # 'dropout', 'embodied', 'synthetic'
        self.embodied_std = 0.15      # placeholder; updated from chip stats
        self.synth_std    = 0.15

    def _apply_noise(self, x, embodied_sample=None):
        if not self.training:
            return x
        if self.noise_mode == 'dropout':
            return F.dropout(x, p=self.p_drop, training=True)
        elif self.noise_mode == 'embodied':
            # multiplicative noise from embodied sample, scaled to embodied_std
            n = embodied_sample
            B, D = x.shape
            need = B * D
            tile = n.repeat((need + n.numel() - 1) // n.numel())[:need].view(B, D)
            # rescale to embodied_std (so amplitude matches synthetic comparator)
            tile = tile * (self.embodied_std / (tile.std() + 1e-6))
            return x * (1.0 + tile)
        elif self.noise_mode == 'synthetic':
            return x * (1.0 + torch.randn_like(x) * self.synth_std)
        return x

    def forward(self, x, embodied_sample=None):
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self._apply_noise(x, embodied_sample)
        x = F.relu(self.fc2(x))
        x = self._apply_noise(x, embodied_sample)
        return self.fc3(x)


def chip_noise_vec(sig: LiveSig, k=32, scale=1.0, n_reads=4):
    """Read live signature multiple times → concatenate → torch tensor (k,).

    Multiple reads ensure independent jitter samples per neuron (otherwise
    each batch shares one 32-d vector which is essentially deterministic).
    Cost: ~n_reads * 1ms per call. 4 reads → 4ms (acceptable).
    """
    parts = []
    for _ in range(n_reads):
        parts.append(np.asarray(sig.read(), dtype=np.float32))
    v = np.concatenate(parts)  # 32*n_reads
    # normalise to target std
    v = (v - v.mean()) / (v.std() + 1e-6) * scale
    if v.size < k:
        rep = (k + v.size - 1) // v.size
        v = np.tile(v, rep)
    return torch.from_numpy(v[:k]).to(DEV)


def calibrate_embodied_std(sig, n_reads=64):
    """Estimate std of chip jitter samples → for matched synthetic comparison."""
    samples = []
    for _ in range(n_reads):
        samples.append(sig.read())
        time.sleep(0.005)
    arr = np.array(samples).astype(np.float32)
    # normalise per-sample to unit std (since we do that in chip_noise_vec)
    arr_n = arr / (arr.std(axis=1, keepdims=True) + 1e-6)
    return float(arr_n.std())


def train_one(variant, train_xy, test_xy, sig, embodied_std, seed,
              epochs=20, bs=128, lr=2e-3):
    torch.manual_seed(seed)
    np.random.seed(seed)
    Xtr, ytr = train_xy
    Xte, yte = test_xy
    Xtr = torch.from_numpy(Xtr).to(DEV)
    ytr = torch.from_numpy(ytr).long().to(DEV)
    Xte = torch.from_numpy(Xte).to(DEV)
    yte = torch.from_numpy(yte).long().to(DEV)

    model = TinyMLP(hidden=256, p_drop=0.10, noise_mode=variant).to(DEV)
    model.embodied_std = 0.15
    model.synth_std    = 0.15
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    n = Xtr.size(0)
    model.train()
    for ep in range(epochs):
        thermal_guard()
        perm = torch.randperm(n, device=DEV)
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            xb = Xtr[idx]
            yb = ytr[idx]
            embodied_sample = None
            if variant == 'embodied':
                embodied_sample = chip_noise_vec(sig, k=128, scale=embodied_std)
            logits = model(xb, embodied_sample)
            loss = F.cross_entropy(logits, yb)
            opt.zero_grad(); loss.backward(); opt.step()
    model.eval()
    with torch.no_grad():
        acc = (model(Xte).argmax(1) == yte).float().mean().item()
    return acc


def main(seeds=15):
    print(f"[E1] starting, seeds={seeds}, device={DEV}")
    sig = LiveSig()
    embodied_std = calibrate_embodied_std(sig, n_reads=48)
    print(f"[E1] calibrated embodied std = {embodied_std:.3f}")

    train_xy, test_xy = make_dataset(n_train=4000, n_test=2000, label_noise=0.10, seed=42)

    results = {'vanilla': [], 'embodied': [], 'synthetic': []}
    for s in range(seeds):
        thermal_guard(verbose=True)
        t0 = time.time()
        for variant in ('vanilla', 'embodied', 'synthetic'):
            acc = train_one(variant, train_xy, test_xy, sig, embodied_std, seed=s)
            results[variant].append(acc)
        print(f"[E1] seed {s}: van={results['vanilla'][-1]:.4f}  "
              f"emb={results['embodied'][-1]:.4f}  "
              f"syn={results['synthetic'][-1]:.4f}  "
              f"(elapsed {time.time()-t0:.1f}s)", flush=True)

    # bootstrap CIs
    summary = {}
    for k, v in results.items():
        m, lo, hi = bootstrap_ci(v, seed=0)
        summary[k] = {'mean': m, 'ci95': [lo, hi], 'values': v}
    dmean, dlo, dhi = diff_ci(results['embodied'], results['synthetic'], seed=0)
    summary['delta_embodied_minus_synthetic'] = {'mean': dmean, 'ci95': [dlo, dhi]}
    gate_pass = (dlo > 0.015)
    summary['gate'] = {
        'criterion': 'embodied - synthetic > 1.5pp (CI lower > 0.015)',
        'delta_mean_pp': dmean * 100,
        'ci95_pp': [dlo * 100, dhi * 100],
        'pass': bool(gate_pass),
    }
    summary['embodied_std'] = embodied_std
    summary['seeds'] = seeds
    save_json('e1_free_entropy_reg.json', summary)
    print(json.dumps(summary['gate'], indent=2))
    return summary


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', type=int, default=15)
    args = ap.parse_args()
    main(seeds=args.seeds)
