"""128-neuron tanh ESN with substrate hooks at activation; ridge readout;
NARMA-10 task."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np

from _substrate_hooks import SubstrateSampler


def narma10(T: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    u = 0.5 * rng.uniform(0.0, 1.0, size=T + 10)  # input in [0, 0.5]
    y = np.zeros(T + 10)
    for t in range(10, T + 10):
        y[t] = (0.3 * y[t-1]
                + 0.05 * y[t-1] * np.sum(y[t-10:t])
                + 1.5 * u[t-10] * u[t-1]
                + 0.1)
    return u[10:], y[10:]


@dataclass
class ESNConfig:
    n: int = 128
    spectral_radius: float = 0.9
    input_scale: float = 1.0
    leak: float = 0.3
    seed: int = 0
    substrate_strength: float = 0.05


def build_esn(cfg: ESNConfig):
    rng = np.random.default_rng(cfg.seed)
    W = rng.standard_normal((cfg.n, cfg.n)) * (1.0 / np.sqrt(cfg.n))
    rho = np.max(np.abs(np.linalg.eigvals(W)))
    W *= cfg.spectral_radius / max(rho, 1e-9)
    Win = rng.standard_normal((cfg.n, 1)) * cfg.input_scale
    return W, Win


def run_esn(u: np.ndarray, W, Win, cfg: ESNConfig,
            substrate: SubstrateSampler | None) -> np.ndarray:
    T = len(u)
    n = cfg.n
    x = np.zeros(n)
    X = np.zeros((T, n))
    for t in range(T):
        pre = W @ x + Win[:, 0] * u[t]
        if substrate is not None:
            gain = substrate.rtn_perturbation(n)
            noise = substrate.spatial_noise(n, scale=cfg.substrate_strength)
            pre = pre * gain + noise
        x_new = np.tanh(pre)
        x = (1 - cfg.leak) * x + cfg.leak * x_new
        X[t] = x
    return X


def train_ridge(X_train: np.ndarray, y_train: np.ndarray, alpha: float = 1e-4):
    # X: (T, n) -> add bias col
    n = X_train.shape[1]
    Xb = np.concatenate([X_train, np.ones((X_train.shape[0], 1))], axis=1)
    A = Xb.T @ Xb + alpha * np.eye(n + 1)
    b = Xb.T @ y_train
    return np.linalg.solve(A, b)


def predict(X: np.ndarray, W_out: np.ndarray) -> np.ndarray:
    Xb = np.concatenate([X, np.ones((X.shape[0], 1))], axis=1)
    return Xb @ W_out


def nrmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    err = y_true - y_pred
    return float(np.sqrt(np.mean(err ** 2)) / (np.std(y_true) + 1e-12))


def run_one(substrate: SubstrateSampler | None, seed: int, T_train=2000,
            T_test=500, cfg_kwargs=None) -> dict:
    cfg_kwargs = cfg_kwargs or {}
    cfg = ESNConfig(seed=seed, **cfg_kwargs)
    W, Win = build_esn(cfg)
    u, y = narma10(T_train + T_test, seed=seed * 13 + 7)
    # discard washout
    X = run_esn(u, W, Win, cfg, substrate)
    wash = 100
    X_tr = X[wash:T_train]
    y_tr = y[wash:T_train]
    X_te = X[T_train:]
    y_te = y[T_train:]
    Wout = train_ridge(X_tr, y_tr)
    y_hat = predict(X_te, Wout)
    return {
        "seed": seed,
        "nrmse": nrmse(y_te, y_hat),
        "substrate": substrate.device if substrate else "none",
    }
