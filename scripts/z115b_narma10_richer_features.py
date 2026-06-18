"""z115b — NARMA-10 with richer per-cell readout features.

z107 baseline (z91 family): readout features = log10|Id|, single value
per cell per timestep (N=100 → 100 features).

z114 (N=200) and z115 (wider biases) both made NRMSE WORSE. The last
cheap test of the "scale doesn't help" finding is whether RICHER per-cell
observables (3 features instead of 1) close the gap to canonical-ESN.

This script:
  N=100, T=600, ρ=0.9, κ=0.003, 5 seeds × 2 conditions.
  A = z107 baseline (single feature: log10|Id|).
  B = three features per cell: [log10|Id|, Vb, Vsint] → 3N total.

If B substantially beats A, we have a brief-grade improvement and a
forward task to extend the protocol. If B ≈ A or B worse, the brief's
"absolute NRMSE high; closing requires Phase B work" framing is fully
defended on three falsified easy hypotheses (N, bias, readout features).
"""
from __future__ import annotations
import json, time
from pathlib import Path
import numpy as np
import torch
torch.set_default_dtype(torch.float64)

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results/z115b_narma10_richer_features"
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
        y[t] = (0.3 * y[t-1] + 0.05 * y[t-1] * np.sum(y[t-10:t])
                + 1.5 * u[t-10] * u[t-1] + 0.1)
    return y


def make_W(N, rho, rng):
    W = rng.normal(0.0, 1.0, size=(N, N))
    eig = np.linalg.eigvals(W)
    rho_W = float(np.max(np.abs(eig)))
    return torch.tensor(W * (rho / max(rho_W, 1e-9)), dtype=torch.float64)


def run(condition, seed, N=100, T=600, rho=0.9, kappa=0.003):
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
    log_Id_t = np.zeros((N, T)); Vb_t = np.zeros((N, T)); Vsint_t = np.zeros((N, T))
    for t in range(T):
        Vd_t = torch.tensor([0.5 + 2.0 * float(u[t])], dtype=torch.float64)
        recur = (W_rec @ feat_prev) if kappa > 0 else torch.zeros(N, dtype=torch.float64)
        VG2_eff = (VG2 + kappa * recur).clamp(-0.2, 1.0)
        out = forward_2t_batched(cfg, M1, M2, bjt, Vd_t, VG1, VG2_eff,
                                   max_iters=15, tol=1e-9, verbose=False)
        log_Id = np.log10(np.maximum(out["Id"].abs().squeeze().numpy(), 1e-15))
        log_Id_t[:, t] = log_Id
        Vb_t[:, t] = out["Vb"].squeeze().numpy()
        Vsint_t[:, t] = out["Vsint"].squeeze().numpy()
        feat_prev = torch.tensor(log_Id, dtype=torch.float64)

    # Build feature matrix per condition
    warmup = 100
    if condition == "A":
        # Single feature per cell: log10|Id|
        feat = log_Id_t
    else:
        # 3 features per cell: log10|Id|, Vb, Vsint
        feat = np.vstack([log_Id_t, Vb_t, Vsint_t])
    feat_n = (feat - feat.mean(axis=1, keepdims=True))
    feat_n = feat_n / (feat_n.std(axis=1, keepdims=True) + 1e-9)
    n_train = int(0.6 * (T - warmup))
    X = np.hstack([np.ones((T - warmup, 1)), feat_n[:, warmup:].T])
    yu = y[warmup:]
    Xtr, Xte = X[:n_train], X[n_train:]
    ytr, yte = yu[:n_train], yu[n_train:]
    ridge = 1e-3
    W = np.linalg.solve(Xtr.T @ Xtr + ridge * np.eye(Xtr.shape[1]), Xtr.T @ ytr)
    pred = Xte @ W
    nrmse = float(np.sqrt(((pred - yte) ** 2).mean()) / max(yte.std(), 1e-9))
    return {"condition": condition, "seed": seed, "nrmse": nrmse,
            "n_features": int(feat.shape[0])}


def main():
    t0 = time.time()
    print(f"[z115b] NARMA-10 richer-features test, N=100, κ=0.003")
    print(f"  A: 1 feat/cell (log10|Id|, 100 features total)")
    print(f"  B: 3 feats/cell (log10|Id|, Vb, Vsint, 300 features total)\n")
    seeds = [42, 43, 44, 45, 46]
    grid_A = np.zeros(len(seeds)); grid_B = np.zeros(len(seeds))
    for j, seed in enumerate(seeds):
        for cond, store in [("A", grid_A), ("B", grid_B)]:
            ti = time.time()
            r = run(cond, seed)
            store[j] = r["nrmse"]
            print(f"  seed={seed}  cond={cond}  NRMSE={r['nrmse']:.4f}  "
                   f"({r['n_features']} features, {time.time()-ti:.1f}s)",
                   flush=True)
    print(f"\n[z115b] === Result ===")
    print(f"  A (1 feat): NRMSE = {grid_A.mean():.3f} ± {grid_A.std(ddof=1):.3f}")
    print(f"  B (3 feats): NRMSE = {grid_B.mean():.3f} ± {grid_B.std(ddof=1):.3f}")
    diffs = grid_B - grid_A
    d_mean = diffs.mean()
    d_sem = diffs.std(ddof=1) / np.sqrt(len(seeds))
    t_stat = d_mean / d_sem if d_sem > 1e-9 else float("inf")
    print(f"  paired Δ (B − A): {d_mean:+.4f} ± {d_sem:.4f} "
           f"(t = {t_stat:+.2f}; negative = improvement)")
    if t_stat < -2:
        print(f"  → richer features HELP")
    elif t_stat > 2:
        print(f"  → richer features HURT")
    else:
        print(f"  → richer features neutral (within noise)")
    json.dump({"seeds": seeds, "A": grid_A.tolist(), "B": grid_B.tolist(),
                "paired_t": float(t_stat)},
               (OUT / "summary.json").open("w"), indent=2)
    print(f"\n[z115b] wall: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
