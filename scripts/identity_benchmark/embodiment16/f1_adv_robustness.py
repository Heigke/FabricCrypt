"""F1: Adversarial robustness gain via chip-noise.

Pre-reg: Embodied (chip-noise) FGSM-eps=0.1 accuracy on MNIST-like task >
synthetic-noise FGSM accuracy by >= 3pp (CI lower bound > 0.03).

Why this might work:
  - chip jitter is autocorrelated (1/f-like), not IID
  - FGSM uses sign(grad); structured noise during training may force the
    model to be smooth in a basis the attacker cannot align with.

Three variants per seed:
  A vanilla       — no noise injection
  B synthetic     — IID Gaussian noise w/ matched std
  C embodied      — chip-jitter samples used as multiplicative noise

We report clean acc and adversarial (FGSM eps=0.1) acc.
Gate is on adversarial acc (C - B).
"""
from __future__ import annotations
import os, sys, time, json, argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from _common import thermal_guard, save_json, bootstrap_ci, diff_ci, cool_to, temp_c

sys.path.insert(0, os.path.join(HERE, '..', 'embodiment14b'))
from signature_live import LiveSig

DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
os.environ.setdefault('HSA_OVERRIDE_GFX_VERSION', '11.0.0')


def make_dataset(n_train=2000, n_test=1000, seed=0):
    """Easy-enough MNIST-like — we want clean acc ~95% so FGSM has somewhere to drop to."""
    rng = np.random.default_rng(seed)
    n = n_train + n_test
    y = rng.integers(0, 10, size=n)
    centres = rng.uniform(6, 22, size=(10, 2))
    X = np.zeros((n, 28, 28), dtype=np.float32)
    xx, yy = np.meshgrid(np.arange(28), np.arange(28))
    for i in range(n):
        cy, cx = centres[y[i]]
        bump = 1.2 * np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / 22.0)
        bump += rng.normal(0, 0.15, size=(28, 28))
        X[i] = bump
    X = (X - X.mean()) / (X.std() + 1e-6)
    return (X[:n_train], y[:n_train]), (X[n_train:], y[n_train:])


class TinyMLP(nn.Module):
    def __init__(self, hidden=128, noise_mode='none'):
        super().__init__()
        self.fc1 = nn.Linear(28 * 28, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, 10)
        self.noise_mode = noise_mode
        self.noise_std = 0.20

    def _inject_noise(self, x, chip_sample=None):
        if not self.training:
            return x
        if self.noise_mode == 'synthetic':
            return x * (1.0 + torch.randn_like(x) * self.noise_std)
        elif self.noise_mode == 'embodied' and chip_sample is not None:
            B, D = x.shape
            need = B * D
            tile = chip_sample.repeat((need + chip_sample.numel() - 1) // chip_sample.numel())[:need].view(B, D)
            tile = tile * (self.noise_std / (tile.std() + 1e-6))
            return x * (1.0 + tile)
        return x

    def forward(self, x, chip_sample=None):
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self._inject_noise(x, chip_sample)
        x = F.relu(self.fc2(x))
        x = self._inject_noise(x, chip_sample)
        return self.fc3(x)


def chip_noise_vec(sig, k=128, n_reads=4):
    parts = [np.asarray(sig.read(), dtype=np.float32) for _ in range(n_reads)]
    v = np.concatenate(parts)
    v = (v - v.mean()) / (v.std() + 1e-6)
    if v.size < k:
        v = np.tile(v, (k + v.size - 1) // v.size)
    return torch.from_numpy(v[:k]).to(DEV)


def fgsm_attack(model, X, y, eps=0.1):
    Xa = X.clone().detach().requires_grad_(True)
    logits = model(Xa)
    loss = F.cross_entropy(logits, y)
    g = torch.autograd.grad(loss, Xa)[0]
    Xadv = (Xa + eps * g.sign()).detach()
    return Xadv


def train_eval(variant, train_xy, test_xy, sig, seed, epochs=12, bs=128, lr=2e-3):
    torch.manual_seed(seed); np.random.seed(seed)
    Xtr, ytr = train_xy
    Xte, yte = test_xy
    Xtr = torch.from_numpy(Xtr).to(DEV); ytr = torch.from_numpy(ytr).long().to(DEV)
    Xte = torch.from_numpy(Xte).to(DEV); yte = torch.from_numpy(yte).long().to(DEV)
    model = TinyMLP(hidden=128, noise_mode=variant).to(DEV)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    n = Xtr.size(0)
    model.train()
    for ep in range(epochs):
        if ep % 3 == 0:
            thermal_guard()
        perm = torch.randperm(n, device=DEV)
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            xb = Xtr[idx]; yb = ytr[idx]
            chip = None
            if variant == 'embodied':
                chip = chip_noise_vec(sig, k=128)
            logits = model(xb, chip)
            loss = F.cross_entropy(logits, yb)
            opt.zero_grad(); loss.backward(); opt.step()
    model.eval()
    with torch.no_grad():
        clean_acc = (model(Xte).argmax(1) == yte).float().mean().item()
    # adversarial — temporarily set train False but require grad on input
    Xadv = fgsm_attack(model, Xte, yte, eps=0.1)
    with torch.no_grad():
        adv_acc = (model(Xadv).argmax(1) == yte).float().mean().item()
    return clean_acc, adv_acc


def main(seeds=30):
    print(f"[F1] start, seeds={seeds}, device={DEV}, temp={temp_c():.1f}C", flush=True)
    sig = LiveSig()
    train_xy, test_xy = make_dataset(n_train=2000, n_test=1000, seed=42)
    results = {'vanilla': {'clean': [], 'adv': []},
               'synthetic': {'clean': [], 'adv': []},
               'embodied': {'clean': [], 'adv': []}}
    t_start = time.time()
    for s in range(seeds):
        thermal_guard(verbose=False)
        t0 = time.time()
        for v in ('vanilla', 'synthetic', 'embodied'):
            c, a = train_eval(v, train_xy, test_xy, sig, seed=s)
            results[v]['clean'].append(c)
            results[v]['adv'].append(a)
        print(f"[F1] seed {s}: van=({results['vanilla']['clean'][-1]:.3f},{results['vanilla']['adv'][-1]:.3f}) "
              f"syn=({results['synthetic']['clean'][-1]:.3f},{results['synthetic']['adv'][-1]:.3f}) "
              f"emb=({results['embodied']['clean'][-1]:.3f},{results['embodied']['adv'][-1]:.3f}) "
              f"[{time.time()-t0:.1f}s T={temp_c():.1f}C]", flush=True)
        if (time.time() - t_start) > 480:
            print(f"[F1] time budget hit at seed {s}", flush=True)
            break

    summary = {}
    for v in results:
        for m in ('clean', 'adv'):
            mean, lo, hi = bootstrap_ci(results[v][m], seed=0)
            summary[f'{v}_{m}'] = {'mean': mean, 'ci95': [lo, hi]}
    # Pre-reg gate: embodied adv - synthetic adv > 3pp
    dmean, dlo, dhi = diff_ci(results['embodied']['adv'], results['synthetic']['adv'], seed=0)
    summary['delta_adv_emb_minus_syn'] = {'mean_pp': dmean*100, 'ci95_pp': [dlo*100, dhi*100]}
    gate_pass = bool(dlo > 0.03)
    summary['gate'] = {
        'criterion': 'embodied_adv - synthetic_adv > 3pp (CI lower > 0.03)',
        'delta_mean_pp': dmean*100,
        'ci95_pp': [dlo*100, dhi*100],
        'pass': gate_pass,
    }
    summary['n_seeds_run'] = len(results['vanilla']['clean'])
    summary['raw'] = results
    save_json('f1_adv_robustness.json', summary)
    print(json.dumps(summary['gate'], indent=2))
    return summary


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', type=int, default=30)
    args = ap.parse_args()
    main(seeds=args.seeds)
