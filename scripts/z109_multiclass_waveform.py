"""z109 — B.5 fifth benchmark: multi-class waveform.

Closes the 5/5 grid that the brief promises. Same multi-seed
paired-t protocol as z102/z103/z104/z105/z107.

Task:
  - 4 classes of periodic waveform: sine, square, sawtooth, triangle.
  - Each segment: 25–35 timesteps of one waveform at random phase
    and amplitude.
  - Reservoir features = per-cell log10|Id|.
  - 4-class one-vs-rest ridge readout with sign decision.
  - Metric: per-timestep classification accuracy on test split.
  - Chance = 0.25.

Sweep:
  - κ ∈ {0.00, 0.003}.
  - N = 30 (between Hopfield's z105/z108 sweet spot at 50 and
    z102's MC sweet spot at 10).
  - T = 800.
  - 5 seeds.
"""
from __future__ import annotations
import json, time
from pathlib import Path
import numpy as np
import torch
torch.set_default_dtype(torch.float64)

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results/z109_multiclass_waveform"
OUT.mkdir(parents=True, exist_ok=True)

import importlib.util
sp = importlib.util.spec_from_file_location("v1", ROOT / "scripts/z96_narma10_pilot.py")
v1 = importlib.util.module_from_spec(sp); sp.loader.exec_module(v1)

from nsram.bsim4_port.nsram_cell_2T import NSRAMCell2TConfig
from nsram.bsim4_port.bjt import GummelPoonNPN
from nsram.bsim4_port.vectorized import forward_2t_batched


def waveform(class_id: int, t_local: np.ndarray, phase: float, period: float) -> np.ndarray:
    """Class 0=sine, 1=square, 2=sawtooth, 3=triangle, all in [-1, 1]."""
    x = (2.0 * np.pi / period) * t_local + phase
    if class_id == 0:
        return np.sin(x)
    elif class_id == 1:
        return np.sign(np.sin(x))
    elif class_id == 2:
        return 2.0 * ((x / (2.0 * np.pi)) % 1.0) - 1.0
    else:
        s = (x / (2.0 * np.pi)) % 1.0
        return 4.0 * np.abs(s - 0.5) - 1.0


def build_sequence(T: int, M: int, rng) -> tuple[np.ndarray, np.ndarray]:
    """Concatenate random-class segments to length T. Returns (u, label)."""
    u = np.zeros(T); labels = np.zeros(T, dtype=int)
    t = 0; t_local = 0.0
    while t < T:
        c = int(rng.integers(0, M))
        seg_len = int(rng.integers(25, 36))
        phase = float(rng.uniform(0, 2 * np.pi))
        period = float(rng.uniform(8.0, 14.0))
        for k in range(seg_len):
            if t + k >= T:
                break
            u[t + k] = waveform(c, np.array([t_local + k]), phase, period)[0]
            labels[t + k] = c
        t += seg_len; t_local += seg_len
    return u, labels


def run_waveform(kappa: float, N: int = 30, T: int = 800, M: int = 4, seed: int = 42):
    rng = np.random.default_rng(seed)
    u, labels = build_sequence(T, M, rng)

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

    warmup = 100
    n_train = int(0.6 * (T - warmup))
    feat_norm = (log_Id_traj - log_Id_traj.mean(axis=1, keepdims=True))
    feat_norm = feat_norm / (feat_norm.std(axis=1, keepdims=True) + 1e-9)
    X = np.hstack([np.ones((T - warmup, 1)), feat_norm[:, warmup:].T])
    y = labels[warmup:]
    X_tr, X_te = X[:n_train], X[n_train:]
    y_tr, y_te = y[:n_train], y[n_train:]
    Y_tr = np.zeros((len(y_tr), M))
    for c in range(M):
        Y_tr[:, c] = (y_tr == c).astype(float) * 2 - 1
    ridge = 1e-3
    XtX = X_tr.T @ X_tr; XtY = X_tr.T @ Y_tr
    W = np.linalg.solve(XtX + ridge * np.eye(XtX.shape[0]), XtY)
    pred = (X_te @ W).argmax(axis=1)
    acc = float((pred == y_te).mean())
    return {"kappa": kappa, "seed": seed, "acc": acc, "fails": fails,
            "n_test": int(len(y_te)), "M": M, "N": N}


def main():
    t0 = time.time()
    print(f"[z109] starting at {time.strftime('%H:%M:%S')} — multi-class waveform")
    kappas = [0.00, 0.003]
    seeds = [42, 43, 44, 45, 46]
    grid = np.zeros((len(kappas), len(seeds)))
    detail = {}
    for j, seed in enumerate(seeds):
        for i, kappa in enumerate(kappas):
            ti = time.time()
            r = run_waveform(kappa, seed=seed)
            grid[i, j] = r["acc"]
            detail[f"k{kappa:.3f}_s{seed}"] = r
            print(f"  seed={seed}  κ={kappa:.3f}  acc={r['acc']:.3f}  "
                   f"fails={r['fails']}  ({time.time()-ti:.1f}s)", flush=True)
    means = grid.mean(axis=1)
    stds = grid.std(axis=1, ddof=1)
    sems = stds / np.sqrt(len(seeds))
    print(f"\n[z109] === multi-class waveform (M=4, chance=0.25) ===")
    print(f"  {'κ':>7s}  {'acc mean':>10s}  {'± std':>7s}  {'± SEM':>7s}")
    for i, kappa in enumerate(kappas):
        print(f"  {kappa:>7.3f}  {means[i]:>10.3f}  {stds[i]:>7.3f}  {sems[i]:>7.3f}")
    diffs = grid[1] - grid[0]
    d_mean, d_sem = diffs.mean(), diffs.std(ddof=1) / np.sqrt(len(seeds))
    t_stat = d_mean / d_sem if d_sem > 1e-9 else float("inf")
    print(f"\n[z109] paired Δ acc: {d_mean:+.3f} ± {d_sem:.3f} (t = {t_stat:+.2f})")
    json.dump({"grid": grid.tolist(), "kappas": kappas, "seeds": seeds,
                "means": means.tolist(), "stds": stds.tolist(),
                "paired_diff_mean": float(d_mean), "paired_diff_sem": float(d_sem),
                "paired_t": float(t_stat), "detail": detail,
                "config": {"N": 30, "T": 800, "M": 4}},
               (OUT / "summary.json").open("w"), indent=2)
    print(f"\n[z109] wall: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
