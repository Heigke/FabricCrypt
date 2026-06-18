"""z105 — B.5 fourth benchmark: Hopfield-style pattern recall.

Same multi-seed protocol (5 seeds, N=10 cells, T=300, κ ∈ {0.00, 0.03}).
Three random ±1 prototype patterns ξ₁, ξ₂, ξ₃ of length N.

Each timestep:
  - Pick μ ∈ {1, 2, 3} uniformly.
  - Corrupt ξ_μ by flipping a fraction p_flip=0.20 of bits.
  - Encode the corrupted pattern into per-cell Vd:
        Vd_i(t) = 1.0 + 0.5 · cue[i, t]   (cue ∈ {-1, +1})
  - Run 1-step DC + recurrence (same as z102).
  - Train a 3-class softmax readout (log-Id features → class μ).

Acceptance: classification accuracy with κ=0.03 > κ=0 (paired t > 2)
and clearly above chance (= 0.333).
"""
from __future__ import annotations
import json, time
from pathlib import Path
import numpy as np
import torch
torch.set_default_dtype(torch.float64)

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results/z105_hopfield_pattern_recall"
OUT.mkdir(parents=True, exist_ok=True)

import importlib.util
sp = importlib.util.spec_from_file_location("v1", ROOT / "scripts/z96_narma10_pilot.py")
v1 = importlib.util.module_from_spec(sp); sp.loader.exec_module(v1)

from nsram.bsim4_port.nsram_cell_2T import NSRAMCell2TConfig
from nsram.bsim4_port.bjt import GummelPoonNPN
from nsram.bsim4_port.vectorized import forward_2t_batched


def run_hopfield(kappa: float, N: int = 10, T: int = 300, M: int = 3,
                  p_flip: float = 0.20, seed: int = 42):
    rng = np.random.default_rng(seed)
    patterns = 2 * rng.integers(0, 2, size=(M, N)) - 1   # M × N, ±1

    # Random sequence of class labels and corrupted cues
    labels = rng.integers(0, M, size=T)
    cues = np.zeros((T, N))
    for t in range(T):
        flips = rng.random(N) < p_flip
        cues[t] = np.where(flips, -patterns[labels[t]], patterns[labels[t]])

    cfg = NSRAMCell2TConfig(use_iii=True, use_gidl=True, use_bjt=True, newton_max_iters=50)
    M1, M2 = v1.build_calibrated_models()
    bjt = GummelPoonNPN.from_sebas_card(); bjt.Bf = 5e4

    base_VG1 = torch.tensor(rng.choice([0.2, 0.4, 0.6], size=N), dtype=torch.float64)
    base_VG2 = torch.tensor(rng.uniform(0.0, 0.5, size=N), dtype=torch.float64)
    W_rec = torch.tensor(rng.normal(0.0, 1.0 / np.sqrt(N), size=(N, N)), dtype=torch.float64)

    # We need per-cell Vd, but forward_2t_batched takes a Vd_seq scalar across cells.
    # Workaround: scale Vd into a single pooled value (cue mean), then encode the
    # bit pattern via VG2 modulation — cue[i] adds δ · cue[i] to VG2[i] each step.
    # Pattern signal lives in the per-cell VG2 bias, which is the substrate's
    # natural input route anyway (matches z102/z104).
    feat_prev = torch.zeros(N, dtype=torch.float64)
    log_Id_traj = np.zeros((N, T))
    fails = 0
    delta = 0.10   # ±0.10 V VG2 swing for ±1 cue
    for t in range(T):
        Vd_t = torch.tensor([1.0], dtype=torch.float64)   # constant Vd
        cue_t = torch.tensor(cues[t], dtype=torch.float64)
        recur = (W_rec @ feat_prev) if kappa > 0 else torch.zeros(N, dtype=torch.float64)
        VG2_eff = (base_VG2 + delta * cue_t + kappa * recur).clamp(-0.2, 1.0)
        try:
            out = forward_2t_batched(cfg, M1, M2, bjt, Vd_t, base_VG1, VG2_eff,
                                       max_iters=15, tol=1e-9, verbose=False)
            log_Id = np.log10(np.maximum(out["Id"].abs().squeeze().numpy(), 1e-15))
        except Exception:
            log_Id = np.zeros(N) - 15
            fails += 1
        log_Id_traj[:, t] = log_Id
        feat_prev = torch.tensor(log_Id, dtype=torch.float64)

    warmup = 50
    n_train = int(0.6 * (T - warmup))
    feat_norm = (log_Id_traj - log_Id_traj.mean(axis=1, keepdims=True))
    feat_norm = feat_norm / (feat_norm.std(axis=1, keepdims=True) + 1e-9)
    X = np.hstack([np.ones((T - warmup, 1)), feat_norm[:, warmup:].T])
    y = labels[warmup:]
    X_tr, X_te = X[:n_train], X[n_train:]
    y_tr, y_te = y[:n_train], y[n_train:]
    # One-vs-rest ridge
    ridge = 1e-3
    Y_tr = np.zeros((len(y_tr), M))
    for c in range(M):
        Y_tr[:, c] = (y_tr == c).astype(float) * 2 - 1
    XtX = X_tr.T @ X_tr; XtY = X_tr.T @ Y_tr
    W = np.linalg.solve(XtX + ridge * np.eye(XtX.shape[0]), XtY)
    pred_scores = X_te @ W
    pred = pred_scores.argmax(axis=1)
    acc = float((pred == y_te).mean())
    return {"kappa": kappa, "seed": seed, "acc": acc, "fails": fails,
            "n_test": int(len(y_te)), "M": M}


def main():
    t0 = time.time()
    print(f"[z105] starting at {time.strftime('%H:%M:%S')} — Hopfield pattern recall")
    kappas = [0.00, 0.03]
    seeds = [42, 43, 44, 45, 46]
    grid = np.zeros((len(kappas), len(seeds)))
    detail = {}
    for j, seed in enumerate(seeds):
        for i, kappa in enumerate(kappas):
            ti = time.time()
            r = run_hopfield(kappa, seed=seed)
            grid[i, j] = r["acc"]
            detail[f"k{kappa:.2f}_s{seed}"] = r
            print(f"  seed={seed}  κ={kappa:.2f}  acc={r['acc']:.3f}  "
                   f"fails={r['fails']}  ({time.time()-ti:.1f}s)", flush=True)
    means = grid.mean(axis=1)
    stds = grid.std(axis=1, ddof=1)
    sems = stds / np.sqrt(len(seeds))
    print(f"\n[z105] === Hopfield pattern recall accuracy (M=3, p_flip=0.20, chance=0.333) ===")
    print(f"  {'κ':>6s}  {'acc mean':>10s}  {'± std':>7s}  {'± SEM':>7s}")
    for i, kappa in enumerate(kappas):
        print(f"  {kappa:>6.2f}  {means[i]:>10.3f}  {stds[i]:>7.3f}  {sems[i]:>7.3f}")
    diffs = grid[1] - grid[0]
    d_mean, d_sem = diffs.mean(), diffs.std(ddof=1) / np.sqrt(len(seeds))
    t_stat = d_mean / d_sem if d_sem > 1e-9 else float("inf")
    print(f"\n[z105] paired Δ acc: {d_mean:+.3f} ± {d_sem:.3f} (t = {t_stat:+.2f})")
    json.dump({"grid": grid.tolist(), "kappas": kappas, "seeds": seeds,
                "means": means.tolist(), "stds": stds.tolist(),
                "paired_diff_mean": float(d_mean), "paired_diff_sem": float(d_sem),
                "paired_t": float(t_stat), "detail": detail},
               (OUT / "summary.json").open("w"), indent=2)
    print(f"\n[z105] wall: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
