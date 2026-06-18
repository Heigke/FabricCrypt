"""z116 — NARMA-10 with hybrid MLP readout (last cheap Phase B test).

z107 + z114 + z115 + z115b ruled out N-scaling, bias diversity, and
richer per-cell features as ways to close the NARMA-10 absolute-NRMSE
gap (0.95 vs canonical-ESN 0.1-0.3). One remaining cheap parametric
lever: replace the linear ridge readout with a small MLP.

Setup:
  - Same z107 features (N=100 cells, log10|Id|, T=600, ρ=0.9, κ=0.003).
  - 5 seeds.
  - Three readout conditions:
      A: linear ridge baseline (z107 setting).
      B: MLP, 1 hidden layer, 64 units, tanh, weight-decay 1e-3.
      C: MLP, 2 hidden layers, 64-32 units, tanh, weight-decay 1e-3.
  - Same train/test split as z107.

If MLP closes the gap meaningfully, brief gets a substantive Phase B
result. If not, the "architectural" framing is defended on FOUR
falsified hypotheses.

Wall budget ~10 min (cell sim ~9 min unchanged; MLP train ~10 s/seed).
"""
from __future__ import annotations
import json, time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
torch.set_default_dtype(torch.float64)

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results/z116_narma10_mlp_readout"
OUT.mkdir(parents=True, exist_ok=True)

import importlib.util
sp = importlib.util.spec_from_file_location("v1", ROOT / "scripts/z96_narma10_pilot.py")
v1 = importlib.util.module_from_spec(sp); sp.loader.exec_module(v1)

from nsram.bsim4_port.nsram_cell_2T import NSRAMCell2TConfig
from nsram.bsim4_port.bjt import GummelPoonNPN
from nsram.bsim4_port.vectorized import forward_2t_batched


def narma10(u):
    T = len(u); y = np.zeros(T)
    for t in range(10, T):
        y[t] = (0.3*y[t-1] + 0.05*y[t-1]*np.sum(y[t-10:t])
                + 1.5*u[t-10]*u[t-1] + 0.1)
    return y


def make_W(N, rho, rng):
    W = rng.normal(0.0, 1.0, size=(N, N))
    eig = np.linalg.eigvals(W)
    rho_W = float(np.max(np.abs(eig)))
    return torch.tensor(W * (rho / max(rho_W, 1e-9)), dtype=torch.float64)


def collect_features(N, T, rho, kappa, seed):
    rng = np.random.default_rng(seed)
    u = 0.5 * rng.random(T)
    y = narma10(u)
    cfg = NSRAMCell2TConfig(use_iii=True, use_gidl=True, use_bjt=True,
                              newton_max_iters=50)
    M1, M2 = v1.build_calibrated_models()
    bjt = GummelPoonNPN.from_sebas_card(); bjt.Bf = 5e4
    VG1 = torch.tensor(rng.choice([0.2, 0.4, 0.6], size=N), dtype=torch.float64)
    VG2 = torch.tensor(rng.uniform(0.0, 0.5, size=N), dtype=torch.float64)
    W_rec = make_W(N, rho, rng)
    feat_prev = torch.zeros(N, dtype=torch.float64)
    log_Id = np.zeros((N, T))
    for t in range(T):
        Vd_t = torch.tensor([0.5 + 2.0*float(u[t])], dtype=torch.float64)
        recur = (W_rec @ feat_prev) if kappa > 0 else torch.zeros(N, dtype=torch.float64)
        VG2_eff = (VG2 + kappa*recur).clamp(-0.2, 1.0)
        out = forward_2t_batched(cfg, M1, M2, bjt, Vd_t, VG1, VG2_eff,
                                   max_iters=15, tol=1e-9, verbose=False)
        log_Id[:, t] = np.log10(np.maximum(
            out["Id"].abs().squeeze().numpy(), 1e-15))
        feat_prev = torch.tensor(log_Id[:, t], dtype=torch.float64)
    return log_Id, y


def fit_linear_ridge(Xtr, ytr, Xte, yte, ridge=1e-3):
    Xtr1 = np.hstack([np.ones((len(Xtr),1)), Xtr])
    Xte1 = np.hstack([np.ones((len(Xte),1)), Xte])
    W = np.linalg.solve(Xtr1.T @ Xtr1 + ridge * np.eye(Xtr1.shape[1]),
                         Xtr1.T @ ytr)
    pred = Xte1 @ W
    return float(np.sqrt(((pred - yte) ** 2).mean()) / max(yte.std(), 1e-9))


class MLP(nn.Module):
    def __init__(self, n_in, hidden):
        super().__init__()
        layers = []
        prev = n_in
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.Tanh()]
            prev = h
        layers += [nn.Linear(prev, 1)]
        self.net = nn.Sequential(*layers)
    def forward(self, x):
        return self.net(x).squeeze(-1)


def fit_mlp(Xtr, ytr, Xte, yte, hidden, epochs=400, lr=1e-2, wd=1e-3, seed=0):
    torch.manual_seed(seed)
    Xt = torch.tensor(Xtr, dtype=torch.float64)
    yt = torch.tensor(ytr, dtype=torch.float64)
    Xv = torch.tensor(Xte, dtype=torch.float64)
    yv = torch.tensor(yte, dtype=torch.float64)
    model = MLP(Xt.shape[1], hidden)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    for _ in range(epochs):
        opt.zero_grad()
        pred = model(Xt)
        loss = ((pred - yt) ** 2).mean()
        loss.backward()
        opt.step()
    with torch.no_grad():
        pred = model(Xv).numpy()
    return float(np.sqrt(((pred - yte) ** 2).mean()) / max(yte.std(), 1e-9))


def main():
    t0 = time.time()
    print(f"[z116] NARMA-10 MLP-readout test, N=100, κ=0.003, ρ=0.9")
    print(f"  A: linear ridge (z107 baseline)")
    print(f"  B: MLP 1×64 tanh")
    print(f"  C: MLP 2× (64, 32) tanh\n")
    seeds = [42, 43, 44, 45, 46]
    nrmse = {"A": [], "B": [], "C": []}
    for seed in seeds:
        ti = time.time()
        log_Id, y = collect_features(N=100, T=600, rho=0.9,
                                       kappa=0.003, seed=seed)
        warmup = 100; n_train = int(0.6 * (600 - warmup))
        feat = (log_Id - log_Id.mean(axis=1, keepdims=True))
        feat /= (feat.std(axis=1, keepdims=True) + 1e-9)
        X = feat[:, warmup:].T
        yu = y[warmup:]
        Xtr, Xte = X[:n_train], X[n_train:]
        ytr, yte = yu[:n_train], yu[n_train:]

        nA = fit_linear_ridge(Xtr, ytr, Xte, yte)
        nB = fit_mlp(Xtr, ytr, Xte, yte, hidden=[64], seed=seed)
        nC = fit_mlp(Xtr, ytr, Xte, yte, hidden=[64, 32], seed=seed)
        nrmse["A"].append(nA); nrmse["B"].append(nB); nrmse["C"].append(nC)
        print(f"  seed={seed}  A={nA:.4f}  B={nB:.4f}  C={nC:.4f}  "
               f"({time.time()-ti:.1f}s)", flush=True)

    print(f"\n[z116] === Result ===")
    for k in "ABC":
        arr = np.array(nrmse[k])
        print(f"  {k}: NRMSE = {arr.mean():.3f} ± {arr.std(ddof=1):.3f}")
    a = np.array(nrmse["A"]); b = np.array(nrmse["B"]); c = np.array(nrmse["C"])
    for cond, x in [("B vs A", b), ("C vs A", c)]:
        diffs = x - a
        d_mean = diffs.mean()
        d_sem = diffs.std(ddof=1) / np.sqrt(len(seeds))
        t = d_mean / d_sem if d_sem > 1e-9 else float("inf")
        print(f"  paired Δ {cond}: {d_mean:+.4f} ± {d_sem:.4f}  (t = {t:+.2f})")
    print(f"\n[z116] z107 reference (A baseline): NRMSE 0.946 ± 0.018")
    json.dump({"seeds": seeds, "nrmse": nrmse},
               (OUT / "summary.json").open("w"), indent=2)
    print(f"\n[z116] wall: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
