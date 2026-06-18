"""HNRT — Hardware-Native Reservoir Tuning.

Differentiable NS-RAM reservoir for NARMA-10.

State equation (per cell i, time t):
    s_i(t) = (1-leak_i) * s_i(t-1) + leak_i * Vb_i(t)
where Vb_i(t) = body-state from solve_2t_steady_state(
                  Vd_fixed,
                  VG1_i = VG1_base + W_in_i * u(t) + VG1_bias_i,
                  VG2_i = VG2_base + VG2_bias_i)

Vb is the natural body-state voltage of the 2T NS-RAM cell -- exactly the
"body-state tau" variable the prompt refers to.  We use Vb (not Id) because:
  (a) Vb varies smoothly on order O(1V) across the operating window,
      whereas Id spans 10+ decades and is numerically intractable as
      a reservoir signal.
  (b) The IFT attachment in diff_forward_id makes Vb fully differentiable
      wrt VG1, VG2, Vd  (verified in z474b gradcheck).

Training: ridge readout closed-form; learnable params {VG1_bias_i,
VG2_bias_i, log_leak_i} optimised by Adam through a TRUNCATED gradient
path (k=1 unroll into the cell at each step, state recurrence in
no_grad).  This sidesteps full BPTT yet still reaches the IFT.
"""
from __future__ import annotations
import os, sys
from pathlib import Path

import numpy as np
import torch

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "nsram"))
sys.path.insert(0, str(_ROOT / "scripts" / "GPU_MAX_A_zgx"))

from _common import build_nsram_stack, diff_forward_id  # noqa: E402


# ---------------------------------------------------------------------------
# NARMA-10  (Atiya & Parlos 2000 canonical form)
# ---------------------------------------------------------------------------
def narma10(n_steps: int, seed: int = 0,
            alpha: float = 0.3, beta: float = 0.05,
            gamma: float = 1.5, delta: float = 0.1,
            order: int = 10):
    rng = np.random.default_rng(seed)
    u = rng.uniform(0.0, 0.5, size=n_steps).astype(np.float64)
    y = np.zeros(n_steps, dtype=np.float64)
    for t in range(order, n_steps):
        sum_y = float(np.sum(y[t-order:t]))
        y[t] = (alpha * y[t-1]
                + beta * y[t-1] * sum_y
                + gamma * u[t-1] * u[t-order]
                + delta)
    return u, y


def nrmse(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    err = y_pred - y_true
    var = float(np.var(y_true))
    if var <= 1e-12:
        return float("nan")
    return float(np.sqrt(np.mean(err * err) / var))


# ---------------------------------------------------------------------------
# Hardware-Native Reservoir
# ---------------------------------------------------------------------------
class HNRTReservoir:
    def __init__(self, N: int = 64, *, device: str = "cpu",
                 VG1_base: float = 0.62, VG2_base: float = 0.20,
                 Vd_fixed: float = 1.5,
                 input_scale: float = 0.08,
                 leak_init: float = 0.3,
                 seed: int = 1234):
        self.N = N
        self.device = device
        self.dtype = torch.float64

        self.cfg, self.M1, self.M2, self.bjt = build_nsram_stack(
            use_snapback=False, device=device)

        rng = np.random.default_rng(seed)
        # Random input projection (frozen)
        W_in = rng.uniform(-1.0, 1.0, size=N).astype(np.float64)
        self.W_in = torch.tensor(W_in * input_scale, dtype=self.dtype,
                                 device=device)
        self.VG1_base = float(VG1_base)
        self.VG2_base = float(VG2_base)
        self.Vd_fixed = torch.full((N,), float(Vd_fixed),
                                    dtype=self.dtype, device=device)

        # Learnable parameters -- initialise with random per-cell offsets so
        # the reservoir explores diverse points on the cell I-V surface
        # (homogeneous init collapses all cells to identical state).
        vg1_init = rng.uniform(-0.05, 0.05, size=N).astype(np.float64)
        vg2_init = rng.uniform(-0.10, 0.20, size=N).astype(np.float64)
        self.VG1_bias = torch.tensor(vg1_init, dtype=self.dtype,
                                     device=device, requires_grad=True)
        self.VG2_bias = torch.tensor(vg2_init, dtype=self.dtype,
                                     device=device, requires_grad=True)
        # leak in (0,1) parameterised by sigmoid(raw)
        leak_raw = float(np.log(leak_init / (1.0 - leak_init)))
        self.leak_raw = torch.full((N,), leak_raw, dtype=self.dtype,
                                    device=device, requires_grad=True)

    # ------------------------------------------------------------------
    def parameters(self):
        return [self.VG1_bias, self.VG2_bias, self.leak_raw]

    def leak(self):
        return torch.sigmoid(self.leak_raw)

    # ------------------------------------------------------------------
    def step_cell(self, u_scalar: torch.Tensor):
        """Single-step cell forward → returns Vb (N,) with gradient."""
        VG1 = self.VG1_base + self.W_in * u_scalar + self.VG1_bias
        VG2 = self.VG2_base + self.VG2_bias
        out = diff_forward_id(
            self.cfg, self.M1, self.M2, self.bjt,
            self.Vd_fixed, VG1, VG2,
            max_iters=20, tol=1e-9)
        return out["Vb"], out["converged"]

    # ------------------------------------------------------------------
    def run_sequence_nograd(self, u_seq: np.ndarray) -> np.ndarray:
        """Run reservoir, NO gradients. Returns states (T,N)."""
        T = len(u_seq)
        s = torch.zeros(self.N, dtype=self.dtype, device=self.device)
        leak = self.leak().detach()
        states = np.zeros((T, self.N), dtype=np.float64)
        with torch.no_grad():
            for t in range(T):
                u_t = torch.tensor(float(u_seq[t]), dtype=self.dtype,
                                   device=self.device)
                Vb, _ = self.step_cell(u_t)
                s = (1.0 - leak) * s + leak * Vb
                states[t] = s.cpu().numpy()
        return states

    # ------------------------------------------------------------------
    def grad_loss_single_seq(self, u_seq: np.ndarray, y_seq: np.ndarray,
                             washout: int, ridge: float = 1e-4,
                             trunc_k: int = 1):
        """Compute differentiable readout-NRMSE loss using truncated grad.

        Procedure:
          1. Roll reservoir under no_grad to obtain states S (T,N).
          2. For each step t >= washout, recompute Vb(t) WITH grad,
             and form state-with-grad s_attach(t) = (1-leak)*s_prev_detached
                                                    + leak*Vb(t)
             We treat s_prev as detached -- this is k=1 truncation.
          3. Build X_grad rows from s_attach. Solve ridge regression
             differentiably:  w = (X^T X + λI)^{-1} X^T y.
          4. Loss = MSE(Xw, y) on the same set.
        """
        T = len(u_seq)
        # Phase 1: forward states no_grad
        S = self.run_sequence_nograd(u_seq)
        S_t = torch.tensor(S, dtype=self.dtype, device=self.device)

        # Phase 2: re-evaluate Vb with grad at each t, attach to truncated
        leak = self.leak()
        rows = []
        for t in range(washout, T):
            u_t = torch.tensor(float(u_seq[t]), dtype=self.dtype,
                               device=self.device)
            Vb, _ = self.step_cell(u_t)
            if t == 0:
                s_prev = torch.zeros(self.N, dtype=self.dtype,
                                      device=self.device)
            else:
                s_prev = S_t[t-1]
            s_attach = (1.0 - leak) * s_prev + leak * Vb
            rows.append(s_attach)
        X = torch.stack(rows, dim=0)   # (T-W, N)
        # Add bias column
        ones = torch.ones((X.shape[0], 1), dtype=self.dtype,
                          device=self.device)
        Xb = torch.cat([X, ones], dim=1)
        y = torch.tensor(y_seq[washout:], dtype=self.dtype,
                         device=self.device)
        XtX = Xb.t() @ Xb
        I = torch.eye(Xb.shape[1], dtype=self.dtype, device=self.device)
        w = torch.linalg.solve(XtX + ridge * I, Xb.t() @ y)
        y_pred = Xb @ w
        mse = ((y_pred - y) ** 2).mean()
        var = y.var(unbiased=False)
        loss_nrmse = torch.sqrt(mse / (var + 1e-12))
        return loss_nrmse, w.detach(), float(loss_nrmse.detach().item())

    # ------------------------------------------------------------------
    def fit_readout_nograd(self, u_seq: np.ndarray, y_seq: np.ndarray,
                            washout: int, ridge: float = 1e-4):
        """Closed-form readout fit (no grad)."""
        S = self.run_sequence_nograd(u_seq)
        X = S[washout:]
        ones = np.ones((X.shape[0], 1))
        Xb = np.concatenate([X, ones], axis=1)
        y = y_seq[washout:]
        XtX = Xb.T @ Xb
        w = np.linalg.solve(XtX + ridge * np.eye(Xb.shape[1]),
                            Xb.T @ y)
        return w, Xb @ w

    def predict(self, u_seq: np.ndarray, w: np.ndarray, washout: int):
        S = self.run_sequence_nograd(u_seq)
        X = S[washout:]
        ones = np.ones((X.shape[0], 1))
        Xb = np.concatenate([X, ones], axis=1)
        return Xb @ w


# ---------------------------------------------------------------------------
# Vanilla ESN baseline (Jaeger 2001 standard hyperparams)
# ---------------------------------------------------------------------------
class ESN:
    """Classic leaky ESN.

    Reference: Jaeger (2001) "The echo state approach to analysing and training
    recurrent neural networks", spectral radius rho < 1, sparse W, tanh
    activation, leaky integration.
    """
    def __init__(self, N: int = 128, *, spectral_radius: float = 0.9,
                 leak: float = 0.3, input_scale: float = 0.5,
                 sparsity: float = 0.1, seed: int = 42):
        rng = np.random.default_rng(seed)
        self.N = N
        self.leak = leak
        self.W_in = rng.uniform(-input_scale, input_scale,
                                 size=(N, 2)).astype(np.float64)  # [u,1]
        W = rng.uniform(-1.0, 1.0, size=(N, N)).astype(np.float64)
        mask = rng.uniform(0, 1, size=(N, N)) < sparsity
        W = W * mask
        # Rescale spectral radius
        eigs = np.linalg.eigvals(W)
        rho = float(np.max(np.abs(eigs)))
        if rho > 1e-12:
            W = W * (spectral_radius / rho)
        self.W = W

    def run(self, u_seq):
        T = len(u_seq)
        s = np.zeros(self.N)
        states = np.zeros((T, self.N))
        for t in range(T):
            inp = np.array([u_seq[t], 1.0])
            pre = self.W_in @ inp + self.W @ s
            s = (1.0 - self.leak) * s + self.leak * np.tanh(pre)
            states[t] = s
        return states

    def fit(self, u_seq, y_seq, washout, ridge=1e-4):
        S = self.run(u_seq)
        X = S[washout:]
        ones = np.ones((X.shape[0], 1))
        Xb = np.concatenate([X, ones], axis=1)
        y = y_seq[washout:]
        XtX = Xb.T @ Xb
        w = np.linalg.solve(XtX + ridge * np.eye(Xb.shape[1]), Xb.T @ y)
        return w

    def predict(self, u_seq, w, washout):
        S = self.run(u_seq)
        X = S[washout:]
        ones = np.ones((X.shape[0], 1))
        Xb = np.concatenate([X, ones], axis=1)
        return Xb @ w


# ---------------------------------------------------------------------------
# Vanilla NN regressor (no recurrence; tapped-delay input)
# ---------------------------------------------------------------------------
class TappedNN(torch.nn.Module):
    def __init__(self, taps: int = 20, hidden: int = 64):
        super().__init__()
        self.taps = taps
        self.net = torch.nn.Sequential(
            torch.nn.Linear(taps, hidden),
            torch.nn.Tanh(),
            torch.nn.Linear(hidden, hidden),
            torch.nn.Tanh(),
            torch.nn.Linear(hidden, 1),
        )

    def build_features(self, u_seq):
        T = len(u_seq)
        X = np.zeros((T, self.taps))
        for t in range(T):
            for k in range(self.taps):
                if t - k >= 0:
                    X[t, k] = u_seq[t - k]
        return X

    def fit_predict(self, u_tr, y_tr, u_va, y_va, washout,
                     epochs: int = 200, lr: float = 5e-3):
        Xtr = self.build_features(u_tr)[washout:]
        ytr = y_tr[washout:]
        Xva = self.build_features(u_va)[washout:]
        yva = y_va[washout:]
        Xtr_t = torch.tensor(Xtr, dtype=torch.float32)
        ytr_t = torch.tensor(ytr, dtype=torch.float32).unsqueeze(-1)
        Xva_t = torch.tensor(Xva, dtype=torch.float32)
        opt = torch.optim.Adam(self.parameters(), lr=lr)
        for ep in range(epochs):
            opt.zero_grad()
            p = self(Xtr_t)
            loss = ((p - ytr_t) ** 2).mean()
            loss.backward()
            opt.step()
        with torch.no_grad():
            pred_tr = self(Xtr_t).squeeze().numpy()
            pred_va = self(Xva_t).squeeze().numpy()
        return pred_tr, pred_va, ytr, yva

    def forward(self, x):
        return self.net(x)
