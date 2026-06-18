"""z104 — B.5 third benchmark: temporal-XOR(τ=2) multi-seed.

Same protocol as z102/z103: 5 seeds, N=10 cells, T=300, κ ∈ {0.00, 0.03}.
Task: predict y(t) = u(t) XOR u(t-2) with u ∈ {-1, +1}.

Standard small-reservoir test. The XOR is non-linearly separable in the
input lag space, so a memoryless feature alone gets chance (50%);
recurrence + nonlinear cell response should clear chance.

Metric: classification accuracy (closer to 1.0 = better; chance = 0.5).
"""
from __future__ import annotations
import json, time
from pathlib import Path
import numpy as np
import torch
torch.set_default_dtype(torch.float64)

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results/z104_temporal_xor_multiseed"
OUT.mkdir(parents=True, exist_ok=True)

import importlib.util
sp = importlib.util.spec_from_file_location("v1", ROOT / "scripts/z96_narma10_pilot.py")
v1 = importlib.util.module_from_spec(sp); sp.loader.exec_module(v1)

from nsram.bsim4_port.nsram_cell_2T import NSRAMCell2TConfig
from nsram.bsim4_port.bjt import GummelPoonNPN
from nsram.bsim4_port.vectorized import forward_2t_batched


def run_xor(kappa: float, tau: int = 2, N: int = 10, T: int = 300, seed: int = 42):
    rng = np.random.default_rng(seed)
    u = 2.0 * rng.integers(0, 2, size=T) - 1.0   # ±1 input
    # XOR target: y(t) = u(t) XOR u(t-tau).  In ±1 form: y = -u(t)*u(t-tau)
    y = np.zeros(T)
    for t in range(tau, T):
        y[t] = -u[t] * u[t - tau]   # ±1: identical → -1, different → +1

    cfg = NSRAMCell2TConfig(use_iii=True, use_gidl=True, use_bjt=True, newton_max_iters=50)
    M1, M2 = v1.build_calibrated_models()
    bjt = GummelPoonNPN.from_sebas_card(); bjt.Bf = 5e4

    base_VG1 = torch.tensor(rng.choice([0.2, 0.4, 0.6], size=N), dtype=torch.float64)
    base_VG2 = torch.tensor(rng.uniform(0.0, 0.5, size=N), dtype=torch.float64)
    W_rec = torch.tensor(rng.normal(0.0, 1.0 / np.sqrt(N), size=(N, N)), dtype=torch.float64)

    feat_prev = torch.zeros(N, dtype=torch.float64)
    log_Id_traj = np.zeros((N, T))
    fails = 0
    for t in range(T):
        Vd_t = torch.tensor([1.0 + 0.5 * float(u[t])], dtype=torch.float64)
        recur = (W_rec @ feat_prev) if kappa > 0 else torch.zeros(N, dtype=torch.float64)
        VG2_eff = (base_VG2 + kappa * recur).clamp(-0.2, 1.0)
        try:
            out = forward_2t_batched(cfg, M1, M2, bjt, Vd_t, base_VG1, VG2_eff,
                                       max_iters=15, tol=1e-9, verbose=False)
            log_Id = np.log10(np.maximum(out["Id"].abs().squeeze().numpy(), 1e-15))
        except Exception:
            log_Id = np.zeros(N) - 15
            fails += 1
        log_Id_traj[:, t] = log_Id
        feat_prev = torch.tensor(log_Id, dtype=torch.float64)

    warmup = max(50, tau + 10)
    n_train = int(0.6 * (T - warmup))
    feat_norm = (log_Id_traj - log_Id_traj.mean(axis=1, keepdims=True))
    feat_norm = feat_norm / (feat_norm.std(axis=1, keepdims=True) + 1e-9)
    X = np.hstack([np.ones((T - warmup, 1)), feat_norm[:, warmup:].T])
    y_use = y[warmup:]
    X_tr, X_te = X[:n_train], X[n_train:]
    y_tr, y_te = y_use[:n_train], y_use[n_train:]
    ridge = 1e-3
    XtX = X_tr.T @ X_tr; XtY = X_tr.T @ y_tr
    W = np.linalg.solve(XtX + ridge * np.eye(XtX.shape[0]), XtY)
    pred = X_te @ W
    pred_sign = np.sign(pred)
    pred_sign[pred_sign == 0] = 1
    acc = float((pred_sign == y_te).mean())
    return {"kappa": kappa, "seed": seed, "acc": acc, "fails": fails}


def main():
    t0 = time.time()
    print(f"[z104] starting at {time.strftime('%H:%M:%S')} — temporal-XOR(τ=2) multi-seed")
    kappas = [0.00, 0.03]
    seeds = [42, 43, 44, 45, 46]
    grid = np.zeros((len(kappas), len(seeds)))
    detail = {}
    for j, seed in enumerate(seeds):
        for i, kappa in enumerate(kappas):
            ti = time.time()
            r = run_xor(kappa, seed=seed)
            grid[i, j] = r["acc"]
            detail[f"k{kappa:.2f}_s{seed}"] = r
            print(f"  seed={seed}  κ={kappa:.2f}  acc={r['acc']:.3f}  "
                   f"fails={r['fails']}  ({time.time()-ti:.1f}s)", flush=True)
    means = grid.mean(axis=1)
    stds = grid.std(axis=1, ddof=1)
    sems = stds / np.sqrt(len(seeds))
    print(f"\n[z104] === temporal-XOR(τ=2) accuracy (chance=0.5) ===")
    print(f"  {'κ':>6s}  {'acc mean':>10s}  {'± std':>7s}  {'± SEM':>7s}")
    for i, kappa in enumerate(kappas):
        print(f"  {kappa:>6.2f}  {means[i]:>10.3f}  {stds[i]:>7.3f}  {sems[i]:>7.3f}")
    diffs = grid[1] - grid[0]
    d_mean, d_sem = diffs.mean(), diffs.std(ddof=1) / np.sqrt(len(seeds))
    t_stat = d_mean / d_sem if d_sem > 1e-9 else float("inf")
    print(f"\n[z104] paired Δ acc (κ=0.03 − κ=0): {d_mean:+.3f} ± {d_sem:.3f} "
           f"(t = {t_stat:+.2f})")
    json.dump({"grid": grid.tolist(), "kappas": kappas, "seeds": seeds,
                "means": means.tolist(), "stds": stds.tolist(),
                "paired_diff_mean": float(d_mean), "paired_diff_sem": float(d_sem),
                "paired_t": float(t_stat), "detail": detail},
               (OUT / "summary.json").open("w"), indent=2)
    print(f"\n[z104] wall: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
