"""z115 — NARMA-10 bias-diversity sweep at N=100, κ=0.003, ρ=0.9.

z114 found that doubling N from 100→200 made NARMA-10 NRMSE worse.
Hypothesis: the failure mode is feature collinearity from cells
landing in similar operating points, not reservoir size per se.

This script tests it by widening the per-cell (VG1, VG2) sample
range while holding N=100, T=600, κ=0.003 fixed:

  Condition A (z107 baseline):
    VG1 ∈ {0.2, 0.4, 0.6} (3 discrete),  VG2 ∈ Uniform[0, 0.5]
  Condition B (this script — wider):
    VG1 ∈ Uniform[0.0, 1.0] (continuous),  VG2 ∈ Uniform[-0.3, 0.7]

5 seeds. Wall budget ~9 min.
"""
from __future__ import annotations
import json, time
from pathlib import Path
import numpy as np
import torch
torch.set_default_dtype(torch.float64)

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results/z115_narma10_bias_diversity"
OUT.mkdir(parents=True, exist_ok=True)

import importlib.util
sp = importlib.util.spec_from_file_location("v1", ROOT / "scripts/z96_narma10_pilot.py")
v1 = importlib.util.module_from_spec(sp); sp.loader.exec_module(v1)

from nsram.bsim4_port.nsram_cell_2T import NSRAMCell2TConfig
from nsram.bsim4_port.bjt import GummelPoonNPN
from nsram.bsim4_port.vectorized import forward_2t_batched


def narma10(u: np.ndarray) -> np.ndarray:
    T = len(u); y = np.zeros(T)
    for t in range(10, T):
        y[t] = (0.3 * y[t-1] + 0.05 * y[t-1] * np.sum(y[t-10:t])
                + 1.5 * u[t-10] * u[t-1] + 0.1)
    return y


def make_W_rec(N, rho, rng):
    W = rng.normal(0.0, 1.0, size=(N, N))
    eig = np.linalg.eigvals(W)
    rho_W = float(np.max(np.abs(eig)))
    return torch.tensor(W * (rho / max(rho_W, 1e-9)), dtype=torch.float64)


def run(kappa, condition, seed, N=100, T=600, rho=0.9):
    rng = np.random.default_rng(seed)
    u = 0.5 * rng.random(T)
    y = narma10(u)
    cfg = NSRAMCell2TConfig(use_iii=True, use_gidl=True, use_bjt=True,
                             newton_max_iters=50)
    M1, M2 = v1.build_calibrated_models()
    bjt = GummelPoonNPN.from_sebas_card(); bjt.Bf = 5e4

    if condition == "A":
        # z107 baseline: discrete VG1 trio, narrow VG2
        VG1 = torch.tensor(rng.choice([0.2, 0.4, 0.6], size=N), dtype=torch.float64)
        VG2 = torch.tensor(rng.uniform(0.0, 0.5, size=N), dtype=torch.float64)
    else:
        # B: wider continuous VG1 and VG2
        VG1 = torch.tensor(rng.uniform(0.0, 1.0, size=N), dtype=torch.float64)
        VG2 = torch.tensor(rng.uniform(-0.3, 0.7, size=N), dtype=torch.float64)
    W_rec = make_W_rec(N, rho, rng)

    feat_prev = torch.zeros(N, dtype=torch.float64)
    log_Id = np.zeros((N, T))
    fails = 0
    for t in range(T):
        Vd_t = torch.tensor([0.5 + 2.0 * float(u[t])], dtype=torch.float64)
        recur = (W_rec @ feat_prev) if kappa > 0 else torch.zeros(N, dtype=torch.float64)
        VG2_eff = (VG2 + kappa * recur).clamp(-0.5, 1.2)
        try:
            out = forward_2t_batched(cfg, M1, M2, bjt, Vd_t, VG1, VG2_eff,
                                       max_iters=15, tol=1e-9, verbose=False)
            log_Id[:, t] = np.log10(np.maximum(
                out["Id"].abs().squeeze().numpy(), 1e-15))
        except Exception:
            log_Id[:, t] = -15
            fails += 1
        feat_prev = torch.tensor(log_Id[:, t], dtype=torch.float64)
    warmup = 100; n_train = int(0.6 * (T - warmup))
    feat = (log_Id - log_Id.mean(axis=1, keepdims=True))
    feat /= (feat.std(axis=1, keepdims=True) + 1e-9)
    X = np.hstack([np.ones((T - warmup, 1)), feat[:, warmup:].T])
    yu = y[warmup:]
    Xtr, Xte = X[:n_train], X[n_train:]
    ytr, yte = yu[:n_train], yu[n_train:]
    ridge = 1e-3
    W = np.linalg.solve(Xtr.T @ Xtr + ridge * np.eye(Xtr.shape[1]),
                         Xtr.T @ ytr)
    pred = Xte @ W
    nrmse = float(np.sqrt(((pred - yte) ** 2).mean()) / max(yte.std(), 1e-9))
    feat_cell_std = float(feat[:, warmup:].std(axis=1).mean())
    feat_corr = float(np.median(np.abs(np.corrcoef(feat[:, warmup:]) -
                                          np.eye(N))[np.triu_indices(N, 1)]))
    return {"kappa": kappa, "condition": condition, "seed": seed,
            "nrmse": nrmse, "fails": fails,
            "feat_cell_std_mean": feat_cell_std,
            "feat_pairwise_abs_corr_median": feat_corr}


def main():
    t0 = time.time()
    print(f"[z115] NARMA-10 bias-diversity test at N=100, κ=0.003, ρ=0.9")
    print(f"  A = z107 baseline (VG1 ∈ {{0.2,0.4,0.6}}, VG2 ∈ Uniform[0, 0.5])")
    print(f"  B = wider (VG1 ∈ Uniform[0, 1.0], VG2 ∈ Uniform[-0.3, 0.7])")
    print()
    seeds = [42, 43, 44, 45, 46]
    detail = []
    grid_A = np.zeros(len(seeds)); grid_B = np.zeros(len(seeds))
    for j, seed in enumerate(seeds):
        for cond, store in [("A", grid_A), ("B", grid_B)]:
            ti = time.time()
            r = run(0.003, cond, seed, N=100, T=600, rho=0.9)
            store[j] = r["nrmse"]
            detail.append(r)
            print(f"  seed={seed}  cond={cond}  NRMSE={r['nrmse']:.4f}  "
                   f"feat_std={r['feat_cell_std_mean']:.3f}  "
                   f"|corr|={r['feat_pairwise_abs_corr_median']:.3f}  "
                   f"({time.time()-ti:.1f}s)", flush=True)

    print(f"\n[z115] === Bias-diversity result ===")
    print(f"  A (baseline): NRMSE = {grid_A.mean():.3f} ± {grid_A.std(ddof=1):.3f}")
    print(f"  B (wider):    NRMSE = {grid_B.mean():.3f} ± {grid_B.std(ddof=1):.3f}")
    diffs = grid_B - grid_A
    d_mean = diffs.mean()
    d_sem = diffs.std(ddof=1) / np.sqrt(len(seeds))
    t_stat = d_mean / d_sem if d_sem > 1e-9 else float("inf")
    print(f"  paired Δ (B − A): {d_mean:+.4f} ± {d_sem:.4f} (t = {t_stat:+.2f})")
    if t_stat < -2:
        print(f"  → wider biases HELP (paired-t significant)")
    elif t_stat > 2:
        print(f"  → wider biases HURT")
    else:
        print(f"  → bias diversity neither clearly helps nor hurts")
    print(f"\n  z107 reference (N=100, baseline): NRMSE 0.946")
    json.dump({"seeds": seeds, "A": grid_A.tolist(), "B": grid_B.tolist(),
                "paired_t": float(t_stat), "detail": detail},
               (OUT / "summary.json").open("w"), indent=2)
    print(f"\n[z115] wall: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
