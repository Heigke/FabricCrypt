"""z50_narma_benchmark.py — NARMA-10 time-series prediction benchmark.

Why this experiment:
  z48/z49 showed that local plasticity rules (Hebbian, BCM) HURT
  bistable-cell networks.  But what about classical reservoir computing
  — fixed random W + trained linear readout?  This is the standard
  recipe for memristor arrays in the literature, and it should work
  here too.

Comparison:
  L0 — direct linear regression on input (no reservoir at all)
  L1 — linear-cell reservoir (no bistability) + linear readout
  L2 — bistable cell reservoir + fixed W + linear readout   ← ours
  L3 — bistable cell reservoir + Hebbian-shaped W + linear readout (z48)

Task:
  NARMA-10:  y(t+1) = 0.3·y(t) + 0.05·y(t)·Σ_{k=0..9} y(t-k)
                       + 1.5·u(t-9)·u(t)  +  0.1
  with u(t) ~ U[0, 0.5].  Standard reservoir-computing benchmark.

Metric: NRMSE on held-out test set.
"""
from __future__ import annotations
import json, time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from nsram.cell_fast import CellArray
from nsram.plasticity_net import (NetSim, topo_random)

OUT = Path("/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy/"
           "results/z50_narma_benchmark")
OUT.mkdir(parents=True, exist_ok=True)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[device] {DEVICE}")
N_CELLS = 96
N_SEEDS = 5

CAL = dict(VTH0=0.43, K_back=-0.98, A_iii=4.71, G_bjt=1.00,
            V_bjt_on=0.74, V_latch=0.58, K_leak=0.021)


def narma10(T, seed=0):
    rng = np.random.default_rng(seed)
    u = rng.uniform(0.0, 0.5, T).astype(np.float64)
    y = np.zeros(T)
    for t in range(10, T - 1):
        s = y[t-9:t+1].sum()
        y[t+1] = 0.3 * y[t] + 0.05 * y[t] * s + 1.5 * u[t-9] * u[t] + 0.1
    return u, y


class LinearCellArray:
    def __init__(self, N, alpha=1.5, VG2=None, K_leak=0.05, dt=0.05,
                  device="cpu"):
        self.N = N; self.dt = dt; self.device = device
        self.alpha = (alpha if isinstance(alpha, torch.Tensor)
                       else torch.full((N,), float(alpha), device=device))
        self.VG2 = (VG2.to(device) if VG2 is not None
                     else torch.full((N,), 0.0, device=device))
        self.K_leak = K_leak
        self.Vb = self.VG2.clone()
    def reset(self): self.Vb = self.VG2.clone()
    def step(self, VG1, drive):
        if not isinstance(drive, torch.Tensor):
            drive = torch.full((self.N,), float(drive), device=self.device)
        self.Vb = (1 - self.K_leak * self.dt) * self.Vb + self.alpha * drive * self.dt
        self.Vb = torch.clamp(self.Vb, -2.0, 2.0)
        return self.read(VG1)
    def read(self, VG1): return torch.tanh(self.Vb)


def make_bistable_net(seed=0, spec_rad=0.9, in_scale=0.5, fb=0.1):
    """Tuned for NARMA-10: spec_rad < 1, low input scale, low feedback."""
    VG2 = torch.full((N_CELLS,), -0.20, device=DEVICE)
    cells = CellArray(N_CELLS, alpha=1.5, VG2=VG2, **CAL, device=DEVICE)
    W = topo_random(N_CELLS, p=0.1, seed=seed, device=DEVICE)
    # Rescale to target spectral radius
    eigs = torch.linalg.eigvals(W).abs().max().item()
    if eigs > 1e-9:
        W = W * (spec_rad / eigs)
    g = torch.Generator(device=DEVICE).manual_seed(seed)
    W_in = torch.randn(N_CELLS, 1, generator=g, device=DEVICE) * in_scale
    return NetSim(cells=cells, W=W, W_in=W_in, feedback_gain=fb)


def make_linear_net(seed=0, spec_rad=0.9, in_scale=0.5, fb=0.1):
    VG2 = torch.full((N_CELLS,), -0.20, device=DEVICE)
    cells = LinearCellArray(N_CELLS, alpha=1.5, VG2=VG2, device=DEVICE)
    W = topo_random(N_CELLS, p=0.1, seed=seed, device=DEVICE)
    eigs = torch.linalg.eigvals(W).abs().max().item()
    if eigs > 1e-9:
        W = W * (spec_rad / eigs)
    g = torch.Generator(device=DEVICE).manual_seed(seed)
    W_in = torch.randn(N_CELLS, 1, generator=g, device=DEVICE) * in_scale
    return NetSim(cells=cells, W=W, W_in=W_in, feedback_gain=fb)


def hebbian_pretrain(net, T, U, eta=2e-3):
    VG1 = torch.full((T,), 0.6, device=DEVICE)
    Vb_running = net.cells.Vb.mean().item()
    momentum = 0.95
    for t in range(T):
        net.step(U[t], VG1[t])
        m = float(net.cells.Vb.mean().item())
        Vb_running = momentum * Vb_running + (1 - momentum) * m
        delta = net.cells.Vb - Vb_running
        net.W += eta * torch.outer(delta, delta) - 1e-3 * net.W
        net.W.fill_diagonal_(0)
        rs = net.W.abs().sum(dim=1, keepdim=True).clamp(min=1e-9)
        net.W /= rs


def run_reservoir(net, U, VG1=0.6):
    T = U.shape[0]
    States = torch.zeros(T, net.cells.N, device=DEVICE)
    VG1_seq = torch.full((T,), VG1, device=DEVICE)
    for t in range(T):
        net.step(U[t], VG1_seq[t])
        States[t] = net.cells.Vb
    return States.cpu().numpy()


def train_test_readout(States, target, n_train=1000, washout=200, ridge=1e-3):
    X_train = States[washout:n_train + washout]
    Y_train = target[washout:n_train + washout]
    X_test = States[n_train + washout:]
    Y_test = target[n_train + washout:]
    # Z-score normalize
    mu = X_train.mean(axis=0, keepdims=True)
    sd = X_train.std(axis=0, keepdims=True) + 1e-9
    X_train_n = (X_train - mu) / sd
    X_test_n = (X_test - mu) / sd
    # Ridge
    A = X_train_n.T @ X_train_n + ridge * np.eye(X_train_n.shape[1])
    w = np.linalg.solve(A, X_train_n.T @ Y_train)
    pred = X_test_n @ w
    nrmse = float(np.sqrt(((pred - Y_test) ** 2).mean()) / (Y_test.std() + 1e-9))
    return nrmse, pred, Y_test


def run_baseline_linear(u, y, n_train=1000, washout=200, lookback=10,
                          ridge=1e-3):
    """L0: direct linear regression on input lookback window."""
    T = len(u)
    X = np.stack([np.roll(u, k) for k in range(lookback)], axis=1)
    X[:lookback] = 0.0
    X_train = X[washout:n_train + washout]
    Y_train = y[washout:n_train + washout]
    X_test = X[n_train + washout:]
    Y_test = y[n_train + washout:]
    A = X_train.T @ X_train + ridge * np.eye(lookback)
    w = np.linalg.solve(A, X_train.T @ Y_train)
    pred = X_test @ w
    nrmse = float(np.sqrt(((pred - Y_test) ** 2).mean()) / (Y_test.std() + 1e-9))
    return nrmse, pred, Y_test


def main():
    t0 = time.time()
    T = 2000
    n_train = 1200
    washout = 200

    rs = {"L0_linear_baseline": [], "L1_linear_reservoir": [],
           "L2_bistable_fixed": [], "L3_bistable_hebbian": []}
    ts_panel = {}   # for plotting one example

    for s in range(N_SEEDS):
        u, y = narma10(T, seed=100 + s)
        U = torch.from_numpy(u.astype(np.float32)).unsqueeze(1).to(DEVICE)

        # L0: direct linear baseline
        nrmse0, _, _ = run_baseline_linear(u, y, n_train=n_train,
                                                  washout=washout)
        rs["L0_linear_baseline"].append(nrmse0)

        # L1: linear reservoir
        net = make_linear_net(seed=s)
        States = run_reservoir(net, U)
        nrmse1, pred1, target1 = train_test_readout(States, y,
                                                          n_train=n_train,
                                                          washout=washout)
        rs["L1_linear_reservoir"].append(nrmse1)

        # L2: bistable fixed
        net = make_bistable_net(seed=s)
        States = run_reservoir(net, U)
        nrmse2, pred2, target2 = train_test_readout(States, y,
                                                          n_train=n_train,
                                                          washout=washout)
        rs["L2_bistable_fixed"].append(nrmse2)

        # L3: bistable Hebbian
        net = make_bistable_net(seed=s)
        hebbian_pretrain(net, T // 2, U[:T // 2], eta=2e-3)
        net.cells.reset()
        States = run_reservoir(net, U)
        nrmse3, _, _ = train_test_readout(States, y, n_train=n_train,
                                                washout=washout)
        rs["L3_bistable_hebbian"].append(nrmse3)

        if s == 0:
            ts_panel["target"] = target2.tolist()
            ts_panel["pred_L2"] = pred2.tolist()
            ts_panel["pred_L1"] = pred1.tolist()

        print(f"  seed={s}  L0={nrmse0:.3f}  L1={nrmse1:.3f}  "
              f"L2={nrmse2:.3f}  L3={nrmse3:.3f}", flush=True)

    elapsed = time.time() - t0
    summary = {k: (float(np.mean(v)), float(np.std(v))) for k, v in rs.items()}
    summary["elapsed_s"] = elapsed
    summary["ts_panel"] = ts_panel
    with open(OUT / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\n=== summary (lower NRMSE is better) ===")
    for k, v in rs.items():
        print(f"  {k:28s}  NRMSE = {np.mean(v):.3f} ± {np.std(v):.3f}")
    bistable_gain = (np.mean(rs["L1_linear_reservoir"]) /
                       np.mean(rs["L2_bistable_fixed"]) - 1) * 100
    print(f"\n  bistable advantage over linear reservoir: {bistable_gain:+.0f}%")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    # Bar comparison
    labels = ["L0\nlinear regression", "L1\nlinear reservoir",
                "L2\nbistable + fixed W", "L3\nbistable + Hebbian"]
    means = [np.mean(rs["L0_linear_baseline"]),
              np.mean(rs["L1_linear_reservoir"]),
              np.mean(rs["L2_bistable_fixed"]),
              np.mean(rs["L3_bistable_hebbian"])]
    stds = [np.std(rs["L0_linear_baseline"]),
              np.std(rs["L1_linear_reservoir"]),
              np.std(rs["L2_bistable_fixed"]),
              np.std(rs["L3_bistable_hebbian"])]
    colors = ["#bdc3c7", "#95a5a6", "#3498db", "#e67e22"]
    bars = axes[0].bar(labels, means, yerr=stds, color=colors, capsize=8)
    for b, v in zip(bars, means):
        axes[0].text(b.get_x() + b.get_width()/2, v, f"{v:.3f}",
                       ha="center", va="bottom", fontsize=10, weight="bold")
    axes[0].set_ylabel("NRMSE (lower = better)")
    axes[0].set_title("NARMA-10 prediction — 5 seeds")
    axes[0].grid(alpha=0.3, axis="y")

    # Time-series example
    if ts_panel:
        target = ts_panel["target"]
        pred = ts_panel["pred_L2"]
        axes[1].plot(target[:200], "k-", lw=1.5, label="target NARMA-10")
        axes[1].plot(pred[:200], "b-", lw=1.5, alpha=0.7,
                       label=f"L2 prediction (NRMSE={rs['L2_bistable_fixed'][0]:.3f})")
        axes[1].set_xlabel("step"); axes[1].set_ylabel("y(t)")
        axes[1].set_title("L2 (bistable + fixed W) prediction vs target")
        axes[1].legend(); axes[1].grid(alpha=0.3)

    fig.suptitle(f"z50 — NARMA-10 benchmark  total {elapsed/60:.1f} min, "
                  f"N={N_CELLS} cells, {N_SEEDS} seeds",
                  fontsize=13, weight="bold")
    fig.tight_layout()
    fig.savefig(OUT / "narma.png", dpi=140)
    plt.close(fig)
    print(f"\nWrote {OUT/'narma.png'}")


if __name__ == "__main__":
    main()
