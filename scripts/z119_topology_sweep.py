"""z119 — Comprehensive topology × N × task sweep.

Purpose: test whether the C.3 multi-cell-shared-state argument
generalises beyond the random-Gaussian W_rec we have used in
z100-z117. Reviewer will ask: "does mesh, small-world, sparse
behave the same?" We need an answer.

Lessons folded in:
  - z118: ridge MUST be tuned per N (1e-3 at N=10 fails at N=200).
    We sweep ridge per condition and report best-MC.
  - z117/z115/z114: feature collinearity is the dominant failure
    mode at high N; topology that decorrelates features should
    help.
  - z107/z116: NARMA-10 plateaus at NRMSE 0.95; we report
    absolute and treat any topology that breaks ≤ 0.7 as a hit.

Topologies (W_rec construction, then spectral-radius scaled to ρ=0.9):
  RAND_GAUSS   — random N(0, 1/√N)        (current baseline)
  MESH_4N      — 4-neighbor 2D grid       (C.3 tape-out proposal)
  ER_SPARSE    — Erdős-Rényi p=0.1
  WS_SMALLWORLD — Watts-Strogatz k=4, β=0.1
  ALLTOALL     — all-to-all uniform (1/N)

N ∈ {50, 100, 200}.
Tasks: MC, NARMA-10, XOR(τ=2), multi-class waveform.
Seeds: 3 (compromise for wall budget).
Ridge: per-task GCV-style sweep over {1e-3, 1e-1, 1e+1, 1e+3} → report best.

Wall budget estimate: ~30s/seed N=100, ~60s N=200. 5 topo × 3 N × 3
seeds = 45 cell-sims, ~25-40 min. All tasks share the same cell-sim
for a given (topo, N, seed) — they differ only in inputs and
readouts. So actually each cell-sim is run ONCE per task family
(MC and NARMA share Vd-driven sim; XOR uses a different sim;
waveform another). 4 tasks × 45 = 180, but we batch where possible.

Pragmatic choice: run all four tasks against a SHARED cell-sim
where the input is the task's input. That means 4 × 45 = 180
cell-sims. ~80-120 min wall. We accept the budget.
"""
from __future__ import annotations
import json, time
from pathlib import Path
import numpy as np
import torch
torch.set_default_dtype(torch.float64)

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results/z119_topology_sweep"
OUT.mkdir(parents=True, exist_ok=True)

import importlib.util
sp = importlib.util.spec_from_file_location("v1", ROOT / "scripts/z96_narma10_pilot.py")
v1 = importlib.util.module_from_spec(sp); sp.loader.exec_module(v1)

from nsram.bsim4_port.nsram_cell_2T import NSRAMCell2TConfig
from nsram.bsim4_port.bjt import GummelPoonNPN
from nsram.bsim4_port.vectorized import forward_2t_batched


# ---------------------------------------------------------------- topologies

def build_W(topology: str, N: int, rho: float, rng: np.random.Generator) -> np.ndarray:
    if topology == "RAND_GAUSS":
        W = rng.normal(0.0, 1.0/np.sqrt(N), size=(N, N))
    elif topology == "MESH_4N":
        side = int(round(np.sqrt(N)))
        if side * side != N:
            # round N to nearest square; for our chosen N (50,100,200) we use closest grid
            # 50→7×7=49, 100→10×10=100, 200→14×14=196; we mask back to N
            pass
        # Build square mesh with periodic boundary
        W = np.zeros((N, N))
        side = int(np.ceil(np.sqrt(N)))
        coords = [(i, j) for i in range(side) for j in range(side)][:N]
        idx = {c: k for k, c in enumerate(coords)}
        for k, (i, j) in enumerate(coords):
            for di, dj in [(-1,0),(1,0),(0,-1),(0,1)]:
                ni, nj = (i+di) % side, (j+dj) % side
                if (ni, nj) in idx:
                    W[k, idx[(ni, nj)]] = rng.normal(0.0, 1.0)
    elif topology == "ER_SPARSE":
        p = 0.1
        mask = rng.random((N, N)) < p
        W = np.where(mask, rng.normal(0.0, 1.0, size=(N, N)), 0.0)
    elif topology == "WS_SMALLWORLD":
        # Watts-Strogatz with k=4 (2 each side ring) and rewire β=0.1
        k = 4
        beta = 0.1
        W = np.zeros((N, N))
        for i in range(N):
            for off in range(1, k//2 + 1):
                for sign in (-1, +1):
                    j = (i + sign*off) % N
                    if rng.random() < beta:
                        # rewire to random target
                        j = int(rng.integers(0, N))
                        if j == i:
                            continue
                    W[i, j] = rng.normal(0.0, 1.0)
    elif topology == "ALLTOALL":
        W = np.full((N, N), 1.0/N) + 0.01 * rng.normal(0.0, 1.0, size=(N, N))
        np.fill_diagonal(W, 0.0)
    else:
        raise ValueError(topology)
    eig = np.linalg.eigvals(W)
    rho_W = float(np.max(np.abs(eig)))
    return W * (rho / max(rho_W, 1e-9))


# ---------------------------------------------------------------- tasks

def narma10(u):
    T = len(u); y = np.zeros(T)
    for t in range(10, T):
        y[t] = (0.3*y[t-1] + 0.05*y[t-1]*np.sum(y[t-10:t])
                + 1.5*u[t-10]*u[t-1] + 0.1)
    return y


def waveform_inputs(T, n_classes=4, rng=None):
    """Periodic waveforms: sine, square, triangle, sawtooth at random phases."""
    cls = rng.integers(0, n_classes, size=T)
    u = np.zeros(T); freq = 0.1
    phase = 0.0
    for t in range(T):
        if t % 20 == 0:
            phase = rng.uniform(0, 2*np.pi)
        ang = 2*np.pi*freq*t + phase
        c = cls[t]
        if c == 0:
            u[t] = np.sin(ang)
        elif c == 1:
            u[t] = np.sign(np.sin(ang))
        elif c == 2:
            u[t] = 2*np.abs(((ang/np.pi) % 2) - 1) - 1
        else:
            u[t] = ((ang/np.pi) % 2) - 1
    return u, cls


# ---------------------------------------------------------------- cell sim

def run_cell_sim(topology, N, T, kappa, drive_fn, seed):
    """drive_fn(t,u_t) -> Vd_t (scalar). Returns log10|Id| (N,T) array."""
    rng = np.random.default_rng(seed)
    cfg = NSRAMCell2TConfig(use_iii=True, use_gidl=True, use_bjt=True, newton_max_iters=50)
    M1, M2 = v1.build_calibrated_models()
    bjt = GummelPoonNPN.from_sebas_card(); bjt.Bf = 5e4
    base_VG1 = torch.tensor(rng.choice([0.2, 0.4, 0.6], size=N), dtype=torch.float64)
    base_VG2 = torch.tensor(rng.uniform(0.0, 0.5, size=N), dtype=torch.float64)
    W_rec_np = build_W(topology, N, rho=0.9, rng=rng)
    W_rec = torch.tensor(W_rec_np, dtype=torch.float64)
    feat_prev = torch.zeros(N, dtype=torch.float64)
    log_Id = np.zeros((N, T))
    for t in range(T):
        Vd_t = torch.tensor([float(drive_fn(t))], dtype=torch.float64)
        recur = (W_rec @ feat_prev) if kappa > 0 else torch.zeros(N, dtype=torch.float64)
        VG2_eff = (base_VG2 + kappa * recur).clamp(-0.2, 1.0)
        out = forward_2t_batched(cfg, M1, M2, bjt, Vd_t, base_VG1, VG2_eff,
                                   max_iters=15, tol=1e-9, verbose=False)
        log_Id[:, t] = np.log10(np.maximum(out["Id"].abs().squeeze().numpy(), 1e-15))
        feat_prev = torch.tensor(log_Id[:, t], dtype=torch.float64)
    return log_Id


def best_ridge_fit(X, y, ridges=(1e-3, 1e-1, 1e+1, 1e+3)):
    """Return (best_ridge, best_score) using train/val split inside provided X."""
    n = len(X); n_tr = int(0.8 * n)
    Xtr, Xv = X[:n_tr], X[n_tr:]
    ytr, yv = y[:n_tr], y[n_tr:]
    best = (None, -np.inf, None)
    for r in ridges:
        XtX = Xtr.T @ Xtr
        W = np.linalg.solve(XtX + r * np.eye(XtX.shape[0]), Xtr.T @ ytr)
        pred = Xv @ W
        if pred.std() > 1e-12 and yv.std() > 1e-12:
            score = float(np.corrcoef(pred, yv)[0, 1] ** 2)
        else:
            score = 0.0
        if score > best[1]:
            best = (r, score, W)
    return best


# ---------------------------------------------------------------- evaluators

def eval_MC_NARMA(log_Id, u, T):
    """Compute MC and NARMA-10 NRMSE on the same cell-sim."""
    warmup = 50
    feat = (log_Id - log_Id.mean(axis=1, keepdims=True))
    feat /= (feat.std(axis=1, keepdims=True) + 1e-9)
    K = 15
    n_train = int(0.6 * (T - warmup - K))
    n_test = T - warmup - n_train - K
    # MC
    mc_per_k = np.zeros(K)
    for k in range(1, K+1):
        t_idx = np.arange(warmup, warmup + n_train + n_test)
        X = np.hstack([np.ones((len(t_idx),1)), feat[:, t_idx].T])
        y = u[t_idx - k]
        Xtr, Xte = X[:n_train], X[n_train:n_train+n_test]
        ytr, yte = y[:n_train], y[n_train:n_train+n_test]
        # ridge sweep — pick best on small held-out from Xtr
        n_tr2 = int(0.8 * len(Xtr))
        best = (None, -np.inf)
        for r in (1e-3, 1e-1, 1e+1, 1e+3):
            XtX = Xtr[:n_tr2].T @ Xtr[:n_tr2]
            W = np.linalg.solve(XtX + r*np.eye(XtX.shape[0]), Xtr[:n_tr2].T @ ytr[:n_tr2])
            p = Xtr[n_tr2:] @ W
            if p.std()>1e-12 and ytr[n_tr2:].std()>1e-12:
                s = float(np.corrcoef(p, ytr[n_tr2:])[0,1]**2)
            else:
                s = 0.0
            if s > best[1]:
                best = (r, s)
        r = best[0]
        XtX = Xtr.T @ Xtr
        W = np.linalg.solve(XtX + r*np.eye(XtX.shape[0]), Xtr.T @ ytr)
        pred = Xte @ W
        if pred.std()>1e-12 and yte.std()>1e-12:
            mc_per_k[k-1] = float(np.corrcoef(pred, yte)[0,1]**2)
    MC = float(mc_per_k.sum())
    # NARMA-10 on same cell-sim (interpret u as in [0,0.5] via 0.25*(u+1))
    u_narma = 0.25 * (u + 1.0)
    y_narma = narma10(u_narma)
    Xn = np.hstack([np.ones((T-warmup,1)), feat[:, warmup:].T])
    yn = y_narma[warmup:]
    n_tr = int(0.6 * (T-warmup))
    Xntr, Xnte = Xn[:n_tr], Xn[n_tr:]
    yntr, ynte = yn[:n_tr], yn[n_tr:]
    n_tr2 = int(0.8 * len(Xntr))
    best = (1e+1, np.inf)  # safe default if all candidate ridges produce NaN/inf
    for r in (1e-3, 1e-1, 1e+1, 1e+3):
        XtX = Xntr[:n_tr2].T @ Xntr[:n_tr2]
        W = np.linalg.solve(XtX + r*np.eye(XtX.shape[0]), Xntr[:n_tr2].T @ yntr[:n_tr2])
        p = Xntr[n_tr2:] @ W
        nrmse = float(np.sqrt(((p - yntr[n_tr2:])**2).mean()) / max(yntr[n_tr2:].std(),1e-9))
        if np.isfinite(nrmse) and nrmse < best[1]:
            best = (r, nrmse)
    r = best[0]
    XtX = Xntr.T @ Xntr
    W = np.linalg.solve(XtX + r*np.eye(XtX.shape[0]), Xntr.T @ yntr)
    pred = Xnte @ W
    NARMA_NRMSE = float(np.sqrt(((pred - ynte)**2).mean()) / max(ynte.std(),1e-9))
    return MC, NARMA_NRMSE


def eval_XOR(log_Id, u_bin, tau, T):
    """XOR(t) = u_bin(t-tau) XOR u_bin(t-tau-1)."""
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
    n_tr2 = int(0.8 * len(Xtr))
    best = (None, -np.inf, None)
    for r in (1e-3, 1e-1, 1e+1, 1e+3):
        XtX = Xtr[:n_tr2].T @ Xtr[:n_tr2]
        W = np.linalg.solve(XtX + r*np.eye(XtX.shape[0]), Xtr[:n_tr2].T @ ytr[:n_tr2])
        p = Xtr[n_tr2:] @ W
        acc = float(((p > 0.5) == (ytr[n_tr2:] > 0.5)).mean())
        if acc > best[1]:
            best = (r, acc, W)
    r = best[0]
    XtX = Xtr.T @ Xtr
    W = np.linalg.solve(XtX + r*np.eye(XtX.shape[0]), Xtr.T @ ytr)
    pred = Xte @ W
    acc = float(((pred > 0.5) == (yte > 0.5)).mean())
    return acc


def eval_waveform(log_Id, cls, T, n_classes=4):
    warmup = 50
    feat = (log_Id - log_Id.mean(axis=1, keepdims=True))
    feat /= (feat.std(axis=1, keepdims=True) + 1e-9)
    t_idx = np.arange(warmup, T)
    X = np.hstack([np.ones((len(t_idx),1)), feat[:, t_idx].T])
    Y = np.zeros((len(t_idx), n_classes))
    for k, t in enumerate(t_idx):
        Y[k, cls[t]] = 1.0
    n_tr = int(0.6 * len(X))
    Xtr, Xte = X[:n_tr], X[n_tr:]
    Ytr, Yte = Y[:n_tr], Y[n_tr:]
    cls_te = cls[t_idx[n_tr:]]
    n_tr2 = int(0.8 * len(Xtr))
    best = (None, -np.inf)
    for r in (1e-3, 1e-1, 1e+1, 1e+3):
        XtX = Xtr[:n_tr2].T @ Xtr[:n_tr2]
        W = np.linalg.solve(XtX + r*np.eye(XtX.shape[0]), Xtr[:n_tr2].T @ Ytr[:n_tr2])
        p = Xtr[n_tr2:] @ W
        acc = float((p.argmax(axis=1) == cls[t_idx[:n_tr]][n_tr2:]).mean())
        if acc > best[1]:
            best = (r, acc)
    r = best[0]
    XtX = Xtr.T @ Xtr
    W = np.linalg.solve(XtX + r*np.eye(XtX.shape[0]), Xtr.T @ Ytr)
    pred = (Xte @ W).argmax(axis=1)
    return float((pred == cls_te).mean())


# ---------------------------------------------------------------- main

def main():
    t0 = time.time()
    topologies = ["RAND_GAUSS", "MESH_4N", "ER_SPARSE", "WS_SMALLWORLD", "ALLTOALL"]
    Ns = [50, 100, 200]
    seeds = [42, 43, 44]
    T = 300
    kappa = 0.03
    print(f"[z119] Topology × N × task sweep")
    print(f"  topologies: {topologies}")
    print(f"  Ns: {Ns}, seeds: {seeds}, T={T}, κ={kappa}")
    print(f"  Tasks: MC, NARMA-10, XOR(τ=2), waveform-4class")
    print(f"  Per condition: 2 cell-sims (MC/NARMA share Vd-from-binary; XOR shares; waveform separate)")
    print()

    results = {}
    for topo in topologies:
        for N in Ns:
            for seed in seeds:
                ti = time.time()
                rng = np.random.default_rng(seed)
                # MC + NARMA: Vd driven by binary {-1,+1} (z101 protocol)
                u_bin_int = rng.integers(0, 2, size=T)
                u_bin = 2.0 * u_bin_int - 1.0
                drive = lambda t: 1.0 + 0.5 * float(u_bin[t])
                log_Id1 = run_cell_sim(topo, N, T, kappa, drive, seed)
                MC, NARMA_NRMSE = eval_MC_NARMA(log_Id1, u_bin, T)
                XOR_acc = eval_XOR(log_Id1, u_bin_int, tau=2, T=T)

                # Waveform: separate sim with waveform drive
                u_wave, cls = waveform_inputs(T, n_classes=4, rng=np.random.default_rng(seed+1000))
                drive_w = lambda t: 1.0 + 0.5 * float(u_wave[t])
                log_Id2 = run_cell_sim(topo, N, T, kappa, drive_w, seed)
                WAVE_acc = eval_waveform(log_Id2, cls, T, n_classes=4)

                key = f"{topo}_N{N}_s{seed}"
                results[key] = {"topo": topo, "N": N, "seed": seed,
                                  "MC": MC, "NARMA_NRMSE": NARMA_NRMSE,
                                  "XOR_acc": XOR_acc, "WAVE_acc": WAVE_acc,
                                  "wall_s": float(time.time() - ti)}
                print(f"  {topo:14s} N={N:3d} s={seed}  "
                       f"MC={MC:5.2f}  NARMA={NARMA_NRMSE:5.2f}  "
                       f"XOR={XOR_acc:.2f}  WAVE={WAVE_acc:.2f}  "
                       f"({time.time()-ti:.0f}s)", flush=True)

    # Aggregate per (topo, N) over seeds
    print(f"\n[z119] === Aggregated (mean over {len(seeds)} seeds) ===")
    print(f"  {'topo':14s} {'N':>3s}  {'MC':>6s} {'NARMA':>6s} {'XOR':>5s} {'WAVE':>5s}")
    agg = {}
    for topo in topologies:
        for N in Ns:
            keys = [f"{topo}_N{N}_s{s}" for s in seeds]
            mc = np.mean([results[k]["MC"] for k in keys])
            na = np.mean([results[k]["NARMA_NRMSE"] for k in keys])
            xo = np.mean([results[k]["XOR_acc"] for k in keys])
            wa = np.mean([results[k]["WAVE_acc"] for k in keys])
            agg[f"{topo}_N{N}"] = {"MC": float(mc), "NARMA_NRMSE": float(na),
                                     "XOR_acc": float(xo), "WAVE_acc": float(wa)}
            print(f"  {topo:14s} {N:>3d}  {mc:6.2f} {na:6.2f} {xo:5.2f} {wa:5.2f}")

    # Per-task best topology at each N
    print(f"\n[z119] === Best topology per task per N ===")
    for N in Ns:
        rows = {topo: agg[f"{topo}_N{N}"] for topo in topologies}
        best_mc = max(rows, key=lambda t: rows[t]["MC"])
        best_na = min(rows, key=lambda t: rows[t]["NARMA_NRMSE"])
        best_xo = max(rows, key=lambda t: rows[t]["XOR_acc"])
        best_wa = max(rows, key=lambda t: rows[t]["WAVE_acc"])
        print(f"  N={N:3d}: MC→{best_mc} ({rows[best_mc]['MC']:.2f}); "
               f"NARMA→{best_na} ({rows[best_na]['NARMA_NRMSE']:.2f}); "
               f"XOR→{best_xo} ({rows[best_xo]['XOR_acc']:.2f}); "
               f"WAVE→{best_wa} ({rows[best_wa]['WAVE_acc']:.2f})")

    json.dump({"topologies": topologies, "Ns": Ns, "seeds": seeds,
                "T": T, "kappa": kappa, "results": results, "agg": agg},
               (OUT / "summary.json").open("w"), indent=2)
    print(f"\n[z119] saved {OUT}/summary.json")
    print(f"[z119] wall: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
