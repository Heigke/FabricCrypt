"""z121 — MC + XOR at canonical 1/√N W_rec, ER_SPARSE vs RAND_GAUSS.

z120 found that z119b's ER_SPARSE NARMA win at ρ=0.9 vanishes at
canonical 1/√N (no explicit ρ scaling). The C.3 sparse-primary
recommendation rests primarily on MC (+50% at N=100, 200) and
XOR (+29% at N=200) — both also measured at ρ=0.9.

This script tests: do the MC and XOR sparse advantages survive
the W_rec-scaling change?

Sweep:
  topologies: RAND_GAUSS (1/√N), ER_SPARSE (1/√(Np))
  N = 200
  T = 600 (z101/z107 protocol)
  κ = 0.03 (z102 MC optimum, also used in z119)
  seeds = {42, 43, 44, 45, 46}
  ridges per task: GCV-style sweep over {1e-3, 1e-1, 1e+1, 1e+3}
  tasks: MC (15 lags), XOR (τ=2)

Wall budget: 2 topo × 5 seeds × ~50s/sim + readout = ~12 min.
"""
from __future__ import annotations
import json, time
from pathlib import Path
import numpy as np
import torch
torch.set_default_dtype(torch.float64)

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results/z121_mc_xor_canonical_wrec"
OUT.mkdir(parents=True, exist_ok=True)

import importlib.util
sp = importlib.util.spec_from_file_location("v1", ROOT / "scripts/z96_narma10_pilot.py")
v1 = importlib.util.module_from_spec(sp); sp.loader.exec_module(v1)

from nsram.bsim4_port.nsram_cell_2T import NSRAMCell2TConfig
from nsram.bsim4_port.bjt import GummelPoonNPN
from nsram.bsim4_port.vectorized import forward_2t_batched


def build_W_canonical(topology, N, rng):
    if topology == "RAND_GAUSS":
        return rng.normal(0.0, 1.0/np.sqrt(N), size=(N, N))
    elif topology == "ER_SPARSE":
        p = 0.1
        mask = rng.random((N, N)) < p
        return np.where(mask, rng.normal(0.0, 1.0/np.sqrt(N*p), size=(N, N)), 0.0)
    raise ValueError(topology)


def run_cell_sim(topology, N, T, kappa, seed):
    """Binary input drive (z101 protocol for MC + XOR)."""
    rng = np.random.default_rng(seed)
    u_int = rng.integers(0, 2, size=T)
    u = 2.0 * u_int - 1.0
    cfg = NSRAMCell2TConfig(use_iii=True, use_gidl=True, use_bjt=True, newton_max_iters=50)
    M1, M2 = v1.build_calibrated_models()
    bjt = GummelPoonNPN.from_sebas_card(); bjt.Bf = 5e4
    base_VG1 = torch.tensor(rng.choice([0.2, 0.4, 0.6], size=N), dtype=torch.float64)
    base_VG2 = torch.tensor(rng.uniform(0.0, 0.5, size=N), dtype=torch.float64)
    W_rec = torch.tensor(build_W_canonical(topology, N, rng), dtype=torch.float64)
    feat_prev = torch.zeros(N, dtype=torch.float64)
    log_Id = np.zeros((N, T))
    for t in range(T):
        Vd_t = torch.tensor([1.0 + 0.5*float(u[t])], dtype=torch.float64)
        recur = (W_rec @ feat_prev) if kappa > 0 else torch.zeros(N, dtype=torch.float64)
        VG2_eff = (base_VG2 + kappa*recur).clamp(-0.2, 1.0)
        out = forward_2t_batched(cfg, M1, M2, bjt, Vd_t, base_VG1, VG2_eff,
                                   max_iters=15, tol=1e-9, verbose=False)
        log_Id[:, t] = np.log10(np.maximum(out["Id"].abs().squeeze().numpy(), 1e-15))
        feat_prev = torch.tensor(log_Id[:, t], dtype=torch.float64)
    return log_Id, u_int, u


def best_ridge(Xtr, ytr, ridges, mode="r2"):
    """Pick best ridge by held-out 80/20 split inside Xtr; return (best_r, best_W)."""
    n_tr = int(0.8 * len(Xtr))
    Xa, Xb = Xtr[:n_tr], Xtr[n_tr:]
    ya, yb = ytr[:n_tr], ytr[n_tr:]
    best = (None, -np.inf)   # maximize in both modes (r2 and acc)
    for r in ridges:
        XtX = Xa.T @ Xa
        W = np.linalg.solve(XtX + r*np.eye(XtX.shape[0]), Xa.T @ ya)
        p = Xb @ W
        if mode == "r2":
            score = float(np.corrcoef(p, yb)[0,1]**2) if p.std()>1e-12 and yb.std()>1e-12 else 0.0
            if score > best[1]:
                best = (r, score)
        else:  # "acc"
            score = float(((p > 0.5) == (yb > 0.5)).mean())
            if score > best[1]:
                best = (r, score)
    # Refit at best ridge on full Xtr
    r = best[0]
    XtX = Xtr.T @ Xtr
    W = np.linalg.solve(XtX + r*np.eye(XtX.shape[0]), Xtr.T @ ytr)
    return r, W


def eval_MC(log_Id, u, T, ridges):
    warmup = 50
    feat = (log_Id - log_Id.mean(axis=1, keepdims=True))
    feat /= (feat.std(axis=1, keepdims=True) + 1e-9)
    K = 15
    n_train = int(0.6 * (T - warmup - K))
    n_test = T - warmup - n_train - K
    mc_per_k = np.zeros(K)
    best_ridges = []
    for k in range(1, K+1):
        t_idx = np.arange(warmup, warmup + n_train + n_test)
        X = np.hstack([np.ones((len(t_idx),1)), feat[:, t_idx].T])
        y = u[t_idx - k]
        Xtr, Xte = X[:n_train], X[n_train:n_train+n_test]
        ytr, yte = y[:n_train], y[n_train:n_train+n_test]
        r, W = best_ridge(Xtr, ytr, ridges, mode="r2")
        best_ridges.append(r)
        pred = Xte @ W
        if pred.std()>1e-12 and yte.std()>1e-12:
            mc_per_k[k-1] = float(np.corrcoef(pred, yte)[0,1]**2)
    return float(mc_per_k.sum()), best_ridges


def eval_XOR(log_Id, u_bin, tau, T, ridges):
    warmup = 50
    feat = (log_Id - log_Id.mean(axis=1, keepdims=True))
    feat /= (feat.std(axis=1, keepdims=True) + 1e-9)
    target = np.array([u_bin[t-tau] ^ u_bin[t-tau-1] for t in range(T)], dtype=float)
    t_idx = np.arange(max(warmup, tau+1), T)
    X = np.hstack([np.ones((len(t_idx),1)), feat[:, t_idx].T])
    y = target[t_idx]
    n_tr = int(0.6 * len(X))
    Xtr, Xte = X[:n_tr], X[n_tr:]
    ytr, yte = y[:n_tr], y[n_tr:]
    r, W = best_ridge(Xtr, ytr, ridges, mode="acc")
    pred = Xte @ W
    return float(((pred > 0.5) == (yte > 0.5)).mean()), r


def main():
    t0 = time.time()
    topologies = ["RAND_GAUSS", "ER_SPARSE"]
    seeds = [42, 43, 44, 45, 46]
    ridges = [1e-3, 1e-1, 1e+1, 1e+3]
    N = 200; T = 600; kappa = 0.03
    print(f"[z121] MC + XOR at canonical 1/√N W_rec, N={N}")
    print(f"  topologies: {topologies}, seeds: {seeds}, T={T}, κ={kappa}\n")
    print(f"  z119 reference (at ρ=0.9): MC ER_SPARSE=2.79 vs RAND=1.10")
    print(f"                              XOR ER_SPARSE=0.76 vs RAND=0.60\n")

    grid_mc = np.zeros((len(topologies), len(seeds)))
    grid_xor = np.zeros((len(topologies), len(seeds)))
    for i, topo in enumerate(topologies):
        for j, seed in enumerate(seeds):
            ti = time.time()
            log_Id, u_int, u = run_cell_sim(topo, N, T, kappa, seed)
            sim_t = time.time() - ti
            MC, ridges_mc = eval_MC(log_Id, u, T, ridges)
            XOR, r_xor = eval_XOR(log_Id, u_int, tau=2, T=T, ridges=ridges)
            grid_mc[i, j] = MC
            grid_xor[i, j] = XOR
            from collections import Counter
            common_r = Counter(ridges_mc).most_common(1)[0][0]
            print(f"  {topo:11s} s={seed}  sim={sim_t:.0f}s  MC={MC:5.2f}  "
                   f"XOR={XOR:.2f}  (MC mode-λ={common_r:.0e}, XOR λ={r_xor:.0e})",
                   flush=True)

    print(f"\n[z121] === Aggregated ({len(seeds)} seeds, canonical 1/√N) ===")
    print(f"  {'topo':11s} {'MC mean':>9s} {'± std':>7s}  {'XOR mean':>9s} {'± std':>7s}")
    means_mc, stds_mc = grid_mc.mean(axis=1), grid_mc.std(axis=1, ddof=1)
    means_xor, stds_xor = grid_xor.mean(axis=1), grid_xor.std(axis=1, ddof=1)
    for i, topo in enumerate(topologies):
        print(f"  {topo:11s} {means_mc[i]:>9.3f} {stds_mc[i]:>7.3f}  "
               f"{means_xor[i]:>9.3f} {stds_xor[i]:>7.3f}")

    # Paired-t per task
    print(f"\n[z121] === Paired-t (ER_SPARSE − RAND_GAUSS) ===")
    for name, grid in [("MC", grid_mc), ("XOR", grid_xor)]:
        diffs = grid[1] - grid[0]
        d_mean = diffs.mean()
        d_sem = diffs.std(ddof=1) / np.sqrt(len(seeds))
        t_stat = d_mean / d_sem if d_sem > 1e-9 else float("inf")
        print(f"  {name}: Δ = {d_mean:+.3f}  SEM = {d_sem:.3f}  t = {t_stat:+.2f}")

    # Verdict
    print(f"\n[z121] === Verdict (MC, XOR at canonical 1/√N) ===")
    for name, grid, ref_advantage in [("MC", grid_mc, "+1.69 (z119 ρ=0.9)"),
                                          ("XOR", grid_xor, "+0.16 (z119 ρ=0.9)")]:
        diffs = grid[1] - grid[0]
        d_mean = diffs.mean()
        d_sem = diffs.std(ddof=1) / np.sqrt(len(seeds))
        t_stat = d_mean / d_sem if d_sem > 1e-9 else 0.0
        if abs(t_stat) > 2:
            verdict = "REPLICATES" if d_mean > 0 and name == "XOR" else (
                       "REPLICATES" if d_mean > 0 and name == "MC" else "VANISHES")
        else:
            verdict = "VANISHES (n.s.)"
        print(f"  {name} sparse advantage: {verdict}  "
               f"(z121 Δ={d_mean:+.3f}, t={t_stat:+.2f}; ref {ref_advantage})")

    json.dump({"topologies": topologies, "seeds": seeds, "N": N,
                "T": T, "kappa": kappa,
                "MC_grid": grid_mc.tolist(), "XOR_grid": grid_xor.tolist(),
                "MC_means": means_mc.tolist(), "MC_stds": stds_mc.tolist(),
                "XOR_means": means_xor.tolist(), "XOR_stds": stds_xor.tolist()},
               (OUT / "summary.json").open("w"), indent=2)
    print(f"\n[z121] saved {OUT}/summary.json")
    print(f"[z121] wall: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
