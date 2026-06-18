"""Leaky reservoir with 5 substrate-coupling regimes (0=baseline, 5=deepest).

Regime semantics:
 0 BASELINE      : no substrate at all
 1 FEATURE       : substrate vector concatenated to input (route-around case)
 2 INITIAL_STATE : per-CU thermal signature as reservoir IC
 3 LEAK_PER_NEUR : per-neuron leak from per-core latency rank
 4 WEIGHT_MOD    : recurrent weights modulated by cross-core interaction
 5 DYNAMICAL     : substrate stream injected inside tanh on every step
                   PLUS per-neuron leak PLUS IC
                   This is the constitutive condition.

All regimes share W_rec/W_in seeds so we can isolate the regime effect.
"""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass


@dataclass
class ReservoirCfg:
    n_in: int = 1
    n_res: int = 32
    spectral_radius: float = 0.9
    sparsity: float = 0.2
    leak_default: float = 0.3
    input_scale: float = 0.6
    seed: int = 0
    beta_substrate: float = 0.5      # coupling for regime 5
    weight_mod_strength: float = 0.3  # coupling for regime 4


def _make_W(n: int, sr: float, sparsity: float, rng) -> np.ndarray:
    W = rng.standard_normal((n, n))
    mask = rng.random((n, n)) < sparsity
    W = W * mask
    # spectral radius rescale (power iteration; cheap for n=32)
    eigs = np.linalg.eigvals(W)
    rho = float(np.max(np.abs(eigs)))
    if rho > 1e-9:
        W = W * (sr / rho)
    return W


class Reservoir:
    def __init__(self, cfg: ReservoirCfg, regime: int, substrate=None):
        self.cfg = cfg
        self.regime = regime
        self.substrate = substrate
        rng = np.random.default_rng(cfg.seed)
        # input dim: regime 1 widens W_in
        n_in_eff = cfg.n_in + (substrate.n_dim if (regime == 1 and substrate is not None) else 0)
        self.W_in = rng.standard_normal((cfg.n_res, n_in_eff)) * cfg.input_scale
        self.W_rec = _make_W(cfg.n_res, cfg.spectral_radius, cfg.sparsity, rng)

        # apply regime-specific structural modifications
        if regime >= 3 and substrate is not None:
            self.leak = substrate.per_neuron_leak(cfg.n_res)
        else:
            self.leak = np.full(cfg.n_res, cfg.leak_default)

        if regime == 4 and substrate is not None:
            M = substrate.weight_mod(cfg.n_res)
            self.W_rec_eff = self.W_rec * (1.0 + cfg.weight_mod_strength * M)
        else:
            self.W_rec_eff = self.W_rec

        if regime == 5 and substrate is not None:
            # also reuse weight_mod for regime 5 to push silicon deeper
            M = substrate.weight_mod(cfg.n_res)
            self.W_rec_eff = self.W_rec * (1.0 + 0.15 * M)

        if regime >= 2 and substrate is not None:
            self.x0 = substrate.initial_state(cfg.n_res)
        else:
            self.x0 = np.zeros(cfg.n_res)

    def run(self, u: np.ndarray, washout: int = 50) -> np.ndarray:
        """u shape (T, n_in). Returns reservoir state matrix (T-washout, n_res)."""
        T = u.shape[0]
        x = self.x0.copy()
        states = np.empty((T, self.cfg.n_res))
        if self.substrate is not None:
            self.substrate.reset(seed=self.cfg.seed + 17)
        for t in range(T):
            if self.regime == 1 and self.substrate is not None:
                s = self.substrate.step()
                inp = np.concatenate([u[t], s])
                pre = self.W_in @ inp + self.W_rec_eff @ x
            elif self.regime == 5 and self.substrate is not None:
                s = self.substrate.step()
                # constitutive: substrate enters inside tanh, blended with state
                pre = self.W_in @ u[t] + self.W_rec_eff @ (x + self.cfg.beta_substrate * s)
            else:
                pre = self.W_in @ u[t] + self.W_rec_eff @ x
            x_new = (1.0 - self.leak) * x + self.leak * np.tanh(pre)
            x = x_new
            states[t] = x
        return states[washout:]


def ridge_fit(X: np.ndarray, y: np.ndarray, alpha: float = 1e-4):
    XtX = X.T @ X + alpha * np.eye(X.shape[1])
    Xty = X.T @ y
    return np.linalg.solve(XtX, Xty)


def nrmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    err = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    rng = float(y_true.std() + 1e-12)
    return err / rng


def mackey_glass(T: int, tau: int = 5, seed: int = 0, dt: float = 1.0) -> np.ndarray:
    """Mackey-Glass with small tau -> harder, less smooth target. tau=5 is challenging."""
    rng = np.random.default_rng(seed)
    n = T + 200 + tau + 10
    x = np.full(n, 1.2 + 0.01 * rng.standard_normal())
    for t in range(tau, n - 1):
        x[t + 1] = x[t] + dt * (0.2 * x[t - tau] / (1.0 + x[t - tau] ** 10) - 0.1 * x[t])
    # add tiny observation noise
    x = x[200:200 + T] + 1e-3 * rng.standard_normal(T)
    return x.astype(np.float64)
