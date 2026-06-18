"""Hyperparameter sweep on z216 NARMA-10 reservoir to find working config.

If we can't get train NRMSE < 0.5 by hyperparams, the surrogate is
fundamentally not suited for ESN-style temporal computation.
"""
from __future__ import annotations
import os
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_k] = "1"
import sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from scripts.z200_topo_rule_sweep import build_topo
from scripts.nsram_surrogate import NSRAMSurrogate

surr = NSRAMSurrogate.build_or_load(grid_size=(20, 20, 25))


def gen(T, seed):
    rng = np.random.default_rng(seed)
    u = rng.uniform(0, 0.5, T)
    y = np.zeros(T)
    for k in range(10, T-1):
        y[k+1] = 0.3*y[k] + 0.05*y[k]*y[k-9:k+1].sum() + 1.5*u[k-9]*u[k] + 0.1
    return u, y


def reservoir_run(u, N, W, base_VG1, base_VG2, sign_mask, W_in, g_in, g_rec, leak):
    T = len(u)
    state = np.zeros((N, T))
    feat = np.zeros(N)
    Vd = np.ones(N)
    for t in range(T):
        VG1 = np.clip(base_VG1 + g_in * W_in * u[t], 0.05, 0.7)
        rec = (W @ feat) * sign_mask
        VG2 = np.clip(base_VG2 + g_rec * rec, 0.0, 0.6)
        log_id = surr.eval(VG1, VG2, Vd)
        feat = (1-leak)*feat + leak*log_id
        state[:, t] = feat
    return state


def evaluate(N, g_in, g_rec, leak, seed=0, T=1500, washout=200, T_train=1000):
    rng = np.random.default_rng(seed)
    base_VG1 = rng.choice([0.2, 0.4, 0.6], N).astype(float)
    base_VG2 = rng.uniform(0.1, 0.4, N).astype(float)
    sign_mask = rng.choice([-1.0, 1.0], N).astype(float)
    W_in = rng.normal(0, 1.0, N)
    W = build_topo("ER_SPARSE", N, rng)
    u, y = gen(T, seed)
    state = reservoir_run(u, N, W, base_VG1, base_VG2, sign_mask, W_in, g_in, g_rec, leak)
    X = state.T
    X = np.hstack([X, np.ones((X.shape[0], 1))])
    Xt = X[washout:T_train]; yt = y[washout:T_train]
    Xv = X[T_train:]; yv = y[T_train:]
    w = np.linalg.solve(Xt.T @ Xt + 1e-4 * np.eye(X.shape[1]), Xt.T @ yt)
    pred_t = Xt @ w; pred_v = Xv @ w
    train_nrmse = np.sqrt(((pred_t - yt)**2).mean()) / yt.std()
    test_nrmse = np.sqrt(((pred_v - yv)**2).mean()) / yv.std()
    return train_nrmse, test_nrmse, state.std(axis=1).mean(), state.std()


print(f"{'N':>4}  {'g_in':>5}  {'g_rec':>6}  {'leak':>5}  {'train':>6}  {'test':>6}  state_std/cell  state_std")
for N in [200, 400]:
    for g_in in [0.5, 1.0, 2.0]:
        for g_rec in [0.3, 0.6, 1.0]:
            for leak in [0.3, 0.6, 1.0]:
                tr, te, sd_per, sd_all = evaluate(N, g_in, g_rec, leak)
                print(f"{N:>4}  {g_in:>5.1f}  {g_rec:>6.1f}  {leak:>5.1f}  "
                      f"{tr:>6.3f}  {te:>6.3f}  {sd_per:>14.4f}  {sd_all:.4f}")
