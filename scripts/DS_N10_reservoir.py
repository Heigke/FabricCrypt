"""DS-N10 — NS-RAM physical-dynamics reservoir benchmark.

Goal
----
DS-N5b HDC plateaued. DS-N7 was hash-table-in-disguise. We need a task where
NS-RAM body-state dynamics ARE the computation, not a lookup.

Reservoir computing requires:
  * Rich, nonlinear, fading-memory dynamics in the internal state.
  * Linear readout from internal state → target.

If NS-RAM only encodes its input (digital register style), it cannot beat
random projection. If it has genuine temporal mixing (body charge persists
across timesteps and mixes with input), it should beat random projection on
tasks that require temporal integration.

Three reservoirs, identical input pipeline, identical readout pipeline:

  R1  NS-RAM     — S2b LUT-driven Vb dynamics with sparse VG2 feedback (S3).
  R2  Digital LIF — Leaky integrate-and-fire with recurrent sparse W.
  R3  Random     — Fixed random projection (no dynamics, no recurrence).

Tasks
-----
  N10a  Mackey-Glass (tau=17) 1-step-ahead prediction. Metric: NRMSE.
  N10b  NARMA-10 prediction + memory capacity (lag 1..50 reconstruction).
  N10c  Sine-frequency classification from short snippets.

Pre-registered gates
--------------------
  INFRA       : 10K cells × 1000 steps NS-RAM < 60 s wall.
  HYPOTHESIS  : NS-RAM beats LIF NRMSE by >= 1 pp on Mackey-Glass.
  AMBITIOUS   : NS-RAM beats Random projection on Mackey-Glass.
  KILL-SHOT   : NS-RAM == Random projection → "nothing-special" verdict.

Author: ikaros 2026-05-14.
"""
from __future__ import annotations
import json
import sys
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "results" / "DS_N10_reservoir"
OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(REPO / "scripts"))

from S2b_transient import IiiNetLUT  # noqa: E402


# ─────────────────────────────────────────────────────────────────────────
# TASKS — time series generators
# ─────────────────────────────────────────────────────────────────────────
def mackey_glass(n_steps: int, tau: int = 17, beta: float = 0.2,
                 gamma: float = 0.1, n: float = 10.0,
                 dt: float = 1.0, seed: int = 0) -> np.ndarray:
    """Discrete Mackey-Glass with delay tau. Returns length-n_steps array."""
    rng = np.random.default_rng(seed)
    burnin = 1000
    total = n_steps + burnin
    hist_len = max(tau + 1, 30)
    x = 1.2 + 0.05 * rng.standard_normal(total + hist_len)
    for t in range(hist_len, total + hist_len - 1):
        x[t + 1] = x[t] + dt * (
            beta * x[t - tau] / (1.0 + x[t - tau] ** n) - gamma * x[t]
        )
    return x[hist_len + burnin: hist_len + burnin + n_steps].astype(np.float64)


def narma10(n_steps: int, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Drive u ~ U(0,0.5), output y NARMA-10 nonlinear AR. Returns (u, y)."""
    rng = np.random.default_rng(seed)
    u = rng.uniform(0.0, 0.5, size=n_steps + 50)
    y = np.zeros_like(u)
    for t in range(10, len(u) - 1):
        y[t + 1] = (0.3 * y[t]
                    + 0.05 * y[t] * y[t - 9:t + 1].sum()
                    + 1.5 * u[t - 9] * u[t]
                    + 0.1)
    return u[50:50 + n_steps], y[50:50 + n_steps]


def sine_dataset(n_classes: int = 4, n_per_class: int = 40,
                 snippet_len: int = 200, seed: int = 0
                 ) -> tuple[np.ndarray, np.ndarray]:
    """Frequency classification. Returns (X (n_total, snippet_len), y (n_total,))."""
    rng = np.random.default_rng(seed)
    freqs = np.linspace(0.05, 0.25, n_classes)  # cycles/sample
    X, y = [], []
    for c, f in enumerate(freqs):
        for _ in range(n_per_class):
            phi = rng.uniform(0, 2 * np.pi)
            amp = rng.uniform(0.8, 1.2)
            t = np.arange(snippet_len)
            sig = amp * np.sin(2 * np.pi * f * t + phi)
            sig += 0.05 * rng.standard_normal(snippet_len)
            X.append(sig)
            y.append(c)
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.int64)
    # shuffle
    idx = rng.permutation(len(y))
    return X[idx], y[idx]


# ─────────────────────────────────────────────────────────────────────────
# RESERVOIR — NS-RAM (S2b LUT + sparse spike→VG2 feedback)
# ─────────────────────────────────────────────────────────────────────────
class NSRAMReservoir:
    """LUT-driven Vb dynamics, scalar input projected to Vd, sparse recurrent
    spike → VG2 feedback. Per-cell heterogeneous VG1/VG2/Cb.

    State per cell: Vb (and refractory counter).
    Readout features per output-step: instantaneous Vb of readout_cells.
    """

    def __init__(self, N: int = 10000, n_readout: int = 512,
                 density: float = 0.005, spectral_radius: float = 0.9,
                 dt_s: float = 1e-6, steps_per_input: int = 3,
                 Cb_F: float = 8e-15, V_th: float = 0.60, V_reset: float = 0.30,
                 T_ref_steps: int = 20, Vd_bias: float = 1.5, Vd_gain: float = 1.5,
                 VG1_range: tuple[float, float] = (0.6, 0.78),
                 VG2_base_range: tuple[float, float] = (0.35, 0.55),
                 VG2_fb_gain: float = 0.20,
                 seed: int = 0):
        self.N = N
        self.n_readout = n_readout
        self.dt_s = dt_s
        self.steps_per_input = steps_per_input
        self.Cb_F = Cb_F
        self.V_th = V_th
        self.V_reset = V_reset
        self.T_ref_steps = T_ref_steps
        self.Vd_bias = Vd_bias
        self.Vd_gain = Vd_gain
        self.VG2_fb_gain = VG2_fb_gain

        rng = np.random.default_rng(seed)
        self.rng = rng
        self.lut = IiiNetLUT()

        self.VG1 = rng.uniform(*VG1_range, size=N).astype(np.float64)
        self.VG2_base = rng.uniform(*VG2_base_range, size=N).astype(np.float64)
        # Input projection: each cell gets a per-cell input weight to Vd
        self.W_in = rng.uniform(-1.0, 1.0, size=N).astype(np.float64)

        # Sparse recurrent W (spike feedback into VG2)
        nnz = max(int(density * N * N), N)
        rows = rng.integers(0, N, size=nnz)
        cols = rng.integers(0, N, size=nnz)
        mask = rows != cols
        rows, cols = rows[mask], cols[mask]
        vals = rng.standard_normal(rows.size).astype(np.float64)
        W = sp.csr_matrix((vals, (rows, cols)), shape=(N, N))
        W.sum_duplicates()
        # Approximate spectral radius via power iteration
        v = rng.standard_normal(N); v /= np.linalg.norm(v) + 1e-12
        sr = 1.0
        for _ in range(20):
            v = W @ v
            sr = np.linalg.norm(v) + 1e-12
            v /= sr
        if sr > 0:
            W = W.multiply(spectral_radius / sr).tocsr()
        self.W = W

        # Readout sample
        self.readout_idx = rng.choice(N, size=n_readout, replace=False)

        self.reset()

    def reset(self):
        self.Vb = np.full(self.N, self.V_reset, dtype=np.float64)
        self.refr = np.zeros(self.N, dtype=np.int32)
        self.spike_rate = np.zeros(self.N, dtype=np.float64)

    def run(self, u: np.ndarray) -> np.ndarray:
        """Stream input scalar series u (length T) → return state features (T, n_readout)."""
        T = len(u)
        feats = np.empty((T, self.n_readout), dtype=np.float64)
        inv_Cb = 1.0 / self.Cb_F
        dt = self.dt_s
        Vd_bias = self.Vd_bias
        Vd_gain = self.Vd_gain
        fb_gain = self.VG2_fb_gain
        spike_pulse = np.zeros(self.N, dtype=np.float64)
        for ti in range(T):
            u_t = u[ti]
            # Recurrent feedback: VG2 modulated by recent spike pulses
            VG2_t = self.VG2_base + fb_gain * (self.W @ spike_pulse)
            np.clip(VG2_t, -0.1, 0.6, out=VG2_t)
            spike_pulse[:] = 0.0
            # Input → Vd
            Vd = Vd_bias + Vd_gain * u_t * self.W_in
            np.clip(Vd, 0.25, 3.0, out=Vd)
            # steps_per_input integration steps with held bias
            for _ in range(self.steps_per_input):
                Inet = self.lut(self.VG1, VG2_t, Vd, self.Vb)
                dVb = (Inet * inv_Cb) * dt
                np.clip(dVb, -0.5, 0.5, out=dVb)
                Vb_new = self.Vb + dVb
                np.clip(Vb_new, -0.5, 1.5, out=Vb_new)
                ref_mask = self.refr > 0
                Vb_new[ref_mask] = self.V_reset
                spike_mask = (Vb_new >= self.V_th) & (~ref_mask)
                if spike_mask.any():
                    Vb_new[spike_mask] = self.V_reset
                    self.refr[spike_mask] = self.T_ref_steps
                    spike_pulse[spike_mask] += 1.0
                np.subtract(self.refr, 1, out=self.refr, where=self.refr > 0)
                self.Vb = Vb_new
            feats[ti] = self.Vb[self.readout_idx]
        return feats


# ─────────────────────────────────────────────────────────────────────────
# RESERVOIR — Digital LIF (recurrent spiking, sparse)
# ─────────────────────────────────────────────────────────────────────────
class DigitalLIFReservoir:
    def __init__(self, N: int = 10000, n_readout: int = 512,
                 density: float = 0.005, spectral_radius: float = 0.9,
                 leak: float = 0.9, V_th: float = 1.0, V_reset: float = 0.0,
                 T_ref: int = 2, input_gain: float = 0.5, fb_gain: float = 0.3,
                 steps_per_input: int = 10, seed: int = 0):
        self.N = N
        self.n_readout = n_readout
        self.leak = leak
        self.V_th = V_th
        self.V_reset = V_reset
        self.T_ref = T_ref
        self.input_gain = input_gain
        self.fb_gain = fb_gain
        self.steps_per_input = steps_per_input
        rng = np.random.default_rng(seed)
        self.W_in = rng.uniform(-1.0, 1.0, size=N).astype(np.float64)
        nnz = max(int(density * N * N), N)
        rows = rng.integers(0, N, size=nnz)
        cols = rng.integers(0, N, size=nnz)
        mask = rows != cols
        rows, cols = rows[mask], cols[mask]
        vals = rng.standard_normal(rows.size).astype(np.float64)
        W = sp.csr_matrix((vals, (rows, cols)), shape=(N, N))
        W.sum_duplicates()
        v = rng.standard_normal(N); v /= np.linalg.norm(v) + 1e-12
        sr = 1.0
        for _ in range(20):
            v = W @ v
            sr = np.linalg.norm(v) + 1e-12
            v /= sr
        if sr > 0:
            W = W.multiply(spectral_radius / sr).tocsr()
        self.W = W
        self.readout_idx = rng.choice(N, size=n_readout, replace=False)
        self.reset()

    def reset(self):
        self.V = np.zeros(self.N, dtype=np.float64)
        self.refr = np.zeros(self.N, dtype=np.int32)

    def run(self, u: np.ndarray) -> np.ndarray:
        T = len(u)
        feats = np.empty((T, self.n_readout), dtype=np.float64)
        s_prev = np.zeros(self.N, dtype=np.float64)
        for ti in range(T):
            u_t = u[ti]
            inp = self.input_gain * u_t * self.W_in
            for _ in range(self.steps_per_input):
                rec = self.fb_gain * (self.W @ s_prev)
                Vn = self.leak * self.V + inp + rec
                ref_mask = self.refr > 0
                Vn[ref_mask] = self.V_reset
                spike = (Vn >= self.V_th) & (~ref_mask)
                s_prev = spike.astype(np.float64)
                Vn[spike] = self.V_reset
                self.refr[spike] = self.T_ref
                np.subtract(self.refr, 1, out=self.refr, where=self.refr > 0)
                self.V = Vn
            feats[ti] = self.V[self.readout_idx]
        return feats


# ─────────────────────────────────────────────────────────────────────────
# RESERVOIR — Random projection (NO dynamics, NO recurrence)
# ─────────────────────────────────────────────────────────────────────────
class RandomProjection:
    """Memoryless random projection + tanh. Cannot encode temporal info beyond
    instantaneous input. Kill-shot baseline."""
    def __init__(self, N: int = 10000, n_readout: int = 512, seed: int = 0):
        rng = np.random.default_rng(seed)
        self.N = N
        self.n_readout = n_readout
        self.W_in = rng.standard_normal(N).astype(np.float64)
        self.b = rng.standard_normal(N).astype(np.float64) * 0.5
        self.readout_idx = rng.choice(N, size=n_readout, replace=False)

    def reset(self):
        pass

    def run(self, u: np.ndarray) -> np.ndarray:
        # State = tanh(W_in * u + b), no memory
        u = np.asarray(u, dtype=np.float64).reshape(-1, 1)  # (T, 1)
        H = np.tanh(u * self.W_in[None, :] + self.b[None, :])  # (T, N)
        return H[:, self.readout_idx]


# ─────────────────────────────────────────────────────────────────────────
# READOUT — ridge regression / softmax (numpy)
# ─────────────────────────────────────────────────────────────────────────
def ridge_train(X: np.ndarray, y: np.ndarray, alpha: float = 1e-3) -> np.ndarray:
    """Closed-form ridge. X: (n, d), y: (n,) or (n, k). Returns W: (d+1, k)."""
    n, d = X.shape
    Xb = np.hstack([X, np.ones((n, 1))])
    A = Xb.T @ Xb + alpha * np.eye(d + 1)
    A[-1, -1] = alpha * 1e-3  # do not penalize bias much
    if y.ndim == 1:
        b = Xb.T @ y
    else:
        b = Xb.T @ y
    W = np.linalg.solve(A, b)
    return W


def ridge_predict(X: np.ndarray, W: np.ndarray) -> np.ndarray:
    Xb = np.hstack([X, np.ones((X.shape[0], 1))])
    return Xb @ W


def nrmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    return rmse / (np.std(y_true) + 1e-12)


# ─────────────────────────────────────────────────────────────────────────
# TASK RUNNERS
# ─────────────────────────────────────────────────────────────────────────
def run_mackey_glass(reservoir, seed: int, T_train: int = 1500, T_test: int = 500,
                      washout: int = 200) -> dict:
    x = mackey_glass(T_train + T_test + washout + 1, tau=17, seed=seed)
    # Normalize to [-1, 1] range for input
    x_norm = (x - x.mean()) / (x.std() + 1e-12)
    u = x_norm[:-1]
    y = x_norm[1:]
    reservoir.reset()
    t0 = time.time()
    feats = reservoir.run(u)
    wall = time.time() - t0
    feats = feats[washout:]
    y = y[washout:]
    Xtr, ytr = feats[:T_train], y[:T_train]
    Xte, yte = feats[T_train:T_train + T_test], y[T_train:T_train + T_test]
    W = ridge_train(Xtr, ytr, alpha=1e-3)
    yhat = ridge_predict(Xte, W)
    return {"nrmse": nrmse(yte, yhat), "wall_s": wall,
            "y_true": yte.tolist(), "y_pred": yhat.tolist()}


def run_narma10(reservoir, seed: int, T_train: int = 1500, T_test: int = 500,
                 washout: int = 200) -> dict:
    u, y = narma10(T_train + T_test + washout + 1, seed=seed)
    u_in = (u - u.mean()) / (u.std() + 1e-12)
    reservoir.reset()
    t0 = time.time()
    feats = reservoir.run(u_in)
    wall = time.time() - t0
    feats = feats[washout:]; y = y[washout:]
    Xtr, ytr = feats[:T_train], y[:T_train]
    Xte, yte = feats[T_train:T_train + T_test], y[T_train:T_train + T_test]
    W = ridge_train(Xtr, ytr, alpha=1e-3)
    yhat = ridge_predict(Xte, W)
    return {"nrmse": nrmse(yte, yhat), "wall_s": wall}


def run_memory_capacity(reservoir, seed: int, T_total: int = 2000,
                         washout: int = 200, max_lag: int = 50) -> dict:
    """Standard MC: drive with white noise, predict u(t-k) for k=1..max_lag.
    MC = sum_k r^2(u(t-k), yhat_k).
    """
    rng = np.random.default_rng(seed)
    u = rng.uniform(-1, 1, size=T_total + washout + max_lag).astype(np.float64)
    reservoir.reset()
    feats = reservoir.run(u)
    feats = feats[washout + max_lag:]
    u_use = u[washout + max_lag:]
    n_train = (T_total // 2)
    Xtr = feats[:n_train]; Xte = feats[n_train:]
    mc_total = 0.0
    per_lag = []
    for k in range(1, max_lag + 1):
        ytr = u[washout + max_lag - k: washout + max_lag - k + n_train]
        yte = u[washout + max_lag - k + n_train:
                 washout + max_lag - k + n_train + Xte.shape[0]]
        if len(ytr) != n_train or len(yte) != Xte.shape[0]:
            continue
        W = ridge_train(Xtr, ytr, alpha=1e-3)
        yhat = ridge_predict(Xte, W)
        if np.std(yhat) < 1e-12 or np.std(yte) < 1e-12:
            r2 = 0.0
        else:
            r = np.corrcoef(yte, yhat)[0, 1]
            r2 = float(r * r) if np.isfinite(r) else 0.0
        mc_total += r2
        per_lag.append({"lag": k, "r2": r2})
    return {"MC": mc_total, "per_lag": per_lag}


def run_sine_classification(reservoir, seed: int) -> dict:
    X, y = sine_dataset(n_classes=4, n_per_class=30, snippet_len=150, seed=seed)
    n = len(y); n_train = int(0.7 * n)
    # Get features per snippet: mean of state over snippet
    feats = np.zeros((n, reservoir.n_readout), dtype=np.float64)
    for i in range(n):
        reservoir.reset()
        s = reservoir.run(X[i])
        feats[i] = s.mean(axis=0)
    n_classes = int(y.max() + 1)
    Y_oh = np.zeros((n, n_classes)); Y_oh[np.arange(n), y] = 1.0
    Xtr = feats[:n_train]; Xte = feats[n_train:]
    Ytr = Y_oh[:n_train]; yte = y[n_train:]
    W = ridge_train(Xtr, Ytr, alpha=1e-2)
    Yhat = ridge_predict(Xte, W)
    yhat = Yhat.argmax(axis=1)
    acc = float((yhat == yte).mean())
    return {"acc": acc, "n_train": n_train, "n_test": n - n_train}


# ─────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────
def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=10000)
    ap.add_argument("--n_readout", type=int, default=512)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--infra_only", action="store_true")
    args = ap.parse_args()

    # ── INFRA gate ────────────────────────────────────────────────────
    print(f"[DS-N10] INFRA: N={args.N} cells × 1000 steps NS-RAM", flush=True)
    res_nsram = NSRAMReservoir(N=args.N, n_readout=args.n_readout, seed=0)
    u_probe = np.sin(np.linspace(0, 20 * np.pi, 1000))
    t0 = time.time()
    feats_probe = res_nsram.run(u_probe)
    infra_wall = time.time() - t0
    infra_pass = infra_wall < 60.0
    print(f"  wall = {infra_wall:.2f}s  PASS<60s = {infra_pass}", flush=True)
    print(f"  feature stats: mean={feats_probe.mean():.4f} std={feats_probe.std():.4f}",
          flush=True)
    if args.infra_only:
        return

    # ── Build all three reservoirs (one instance, reuse across tasks/seeds) ──
    def make_reservoirs(seed):
        return {
            "NSRAM":  NSRAMReservoir(N=args.N, n_readout=args.n_readout, seed=seed),
            "LIF":    DigitalLIFReservoir(N=args.N, n_readout=args.n_readout, seed=seed),
            "Random": RandomProjection(N=args.N, n_readout=args.n_readout, seed=seed),
        }

    # ── N10a Mackey-Glass ─────────────────────────────────────────────
    print("\n[DS-N10a] Mackey-Glass τ=17, 1-step prediction", flush=True)
    mg_results = {k: [] for k in ["NSRAM", "LIF", "Random"]}
    mg_first_pred = {}
    for seed in args.seeds:
        revs = make_reservoirs(seed)
        for name, r in revs.items():
            res = run_mackey_glass(r, seed=seed)
            mg_results[name].append(res["nrmse"])
            print(f"  seed={seed} {name}: NRMSE={res['nrmse']:.4f}  wall={res['wall_s']:.1f}s",
                  flush=True)
            if seed == args.seeds[0]:
                mg_first_pred[name] = {"y_true": res["y_true"],
                                        "y_pred": res["y_pred"]}
    mg_summary = {k: {"mean": float(np.mean(v)), "std": float(np.std(v)),
                       "seeds": v} for k, v in mg_results.items()}

    # ── N10b NARMA-10 + memory capacity ───────────────────────────────
    print("\n[DS-N10b] NARMA-10 prediction + Memory Capacity", flush=True)
    narma_results = {k: [] for k in ["NSRAM", "LIF", "Random"]}
    mc_results = {k: [] for k in ["NSRAM", "LIF", "Random"]}
    mc_per_lag_first = {}
    for seed in args.seeds:
        revs = make_reservoirs(seed)
        for name, r in revs.items():
            res_n = run_narma10(r, seed=seed)
            narma_results[name].append(res_n["nrmse"])
            res_m = run_memory_capacity(r, seed=seed)
            mc_results[name].append(res_m["MC"])
            print(f"  seed={seed} {name}: NARMA NRMSE={res_n['nrmse']:.4f}  "
                  f"MC={res_m['MC']:.3f}", flush=True)
            if seed == args.seeds[0]:
                mc_per_lag_first[name] = res_m["per_lag"]
    narma_summary = {k: {"mean": float(np.mean(v)), "std": float(np.std(v)),
                          "seeds": v} for k, v in narma_results.items()}
    mc_summary = {k: {"mean": float(np.mean(v)), "std": float(np.std(v)),
                       "seeds": v} for k, v in mc_results.items()}

    # ── N10c Sine classification ──────────────────────────────────────
    print("\n[DS-N10c] Sine-frequency classification", flush=True)
    sine_results = {k: [] for k in ["NSRAM", "LIF", "Random"]}
    for seed in args.seeds:
        revs = make_reservoirs(seed)
        for name, r in revs.items():
            res = run_sine_classification(r, seed=seed)
            sine_results[name].append(res["acc"])
            print(f"  seed={seed} {name}: acc={res['acc']:.3f}", flush=True)
    sine_summary = {k: {"mean": float(np.mean(v)), "std": float(np.std(v)),
                        "seeds": v} for k, v in sine_results.items()}

    # ── Save ──────────────────────────────────────────────────────────
    payload_mg = {
        "task": "mackey_glass_tau17",
        "n_cells": args.N,
        "seeds": list(args.seeds),
        "results": mg_summary,
        "first_seed_traces": mg_first_pred,
        "infra_wall_s": infra_wall,
        "infra_pass": bool(infra_pass),
    }
    (OUT / "mackey_glass_NRMSE.json").write_text(json.dumps(payload_mg, indent=2))

    payload_narma = {
        "task": "narma10_and_MC",
        "n_cells": args.N,
        "seeds": list(args.seeds),
        "narma_nrmse": narma_summary,
        "memory_capacity": mc_summary,
        "mc_per_lag_first_seed": mc_per_lag_first,
    }
    (OUT / "narma10_MC.json").write_text(json.dumps(payload_narma, indent=2))

    payload_sine = {
        "task": "sine_freq_classification_4class",
        "n_cells": args.N,
        "seeds": list(args.seeds),
        "results": sine_summary,
    }
    (OUT / "sine_classification.json").write_text(json.dumps(payload_sine, indent=2))

    # ── Verdict ───────────────────────────────────────────────────────
    nsram_mg = mg_summary["NSRAM"]["mean"]
    lif_mg = mg_summary["LIF"]["mean"]
    rand_mg = mg_summary["Random"]["mean"]
    hypothesis_pass = (lif_mg - nsram_mg) >= 0.01  # NRMSE: lower better, >=1pp better
    ambitious_pass = (rand_mg - nsram_mg) >= 0.01
    kill_shot = abs(nsram_mg - rand_mg) < 0.005

    summary = {
        "infra_pass": bool(infra_pass),
        "hypothesis_pass_NSRAM_beats_LIF_1pp": bool(hypothesis_pass),
        "ambitious_pass_NSRAM_beats_Random_1pp": bool(ambitious_pass),
        "kill_shot_NSRAM_eq_Random": bool(kill_shot),
        "mackey_glass_nrmse": {
            "NSRAM": nsram_mg, "LIF": lif_mg, "Random": rand_mg,
        },
        "narma_nrmse_mean": {k: v["mean"] for k, v in narma_summary.items()},
        "MC_mean": {k: v["mean"] for k, v in mc_summary.items()},
        "sine_acc_mean": {k: v["mean"] for k, v in sine_summary.items()},
    }

    md_lines = []
    md_lines.append("# DS-N10 Reservoir Computing — NS-RAM vs LIF vs Random\n")
    md_lines.append(f"N = {args.N} cells, readout = {args.n_readout}, seeds = {list(args.seeds)}\n")
    md_lines.append(f"\n**INFRA**: NS-RAM 10K × 1000 steps = {infra_wall:.1f}s  "
                    f"({'PASS' if infra_pass else 'FAIL'})\n")
    md_lines.append("\n## Mackey-Glass τ=17 (1-step ahead, NRMSE, lower better)\n")
    md_lines.append("| Method | mean | std | seeds |\n|---|---|---|---|\n")
    for k in ["NSRAM", "LIF", "Random"]:
        v = mg_summary[k]
        md_lines.append(f"| {k} | {v['mean']:.4f} | {v['std']:.4f} | "
                        f"{[round(s,4) for s in v['seeds']]} |\n")
    md_lines.append("\n## NARMA-10 (NRMSE, lower better)\n")
    md_lines.append("| Method | mean | std |\n|---|---|---|\n")
    for k in ["NSRAM", "LIF", "Random"]:
        v = narma_summary[k]
        md_lines.append(f"| {k} | {v['mean']:.4f} | {v['std']:.4f} |\n")
    md_lines.append("\n## Memory Capacity (sum r² over lags 1..50, higher better)\n")
    md_lines.append("| Method | mean | std |\n|---|---|---|\n")
    for k in ["NSRAM", "LIF", "Random"]:
        v = mc_summary[k]
        md_lines.append(f"| {k} | {v['mean']:.3f} | {v['std']:.3f} |\n")
    md_lines.append("\n## Sine classification (4-class, accuracy, higher better)\n")
    md_lines.append("| Method | mean | std |\n|---|---|---|\n")
    for k in ["NSRAM", "LIF", "Random"]:
        v = sine_summary[k]
        md_lines.append(f"| {k} | {v['mean']:.3f} | {v['std']:.3f} |\n")
    md_lines.append("\n## Verdict\n")
    md_lines.append(f"- HYPOTHESIS (NS-RAM beats LIF by ≥1pp on MG NRMSE): "
                    f"**{'PASS' if hypothesis_pass else 'FAIL'}**\n")
    md_lines.append(f"- AMBITIOUS (NS-RAM beats Random by ≥1pp on MG NRMSE): "
                    f"**{'PASS' if ambitious_pass else 'FAIL'}**\n")
    md_lines.append(f"- KILL-SHOT (NS-RAM ≈ Random within 0.005): "
                    f"**{'TRIGGERED' if kill_shot else 'not triggered'}**\n")
    md_lines.append(f"\n```json\n{json.dumps(summary, indent=2)}\n```\n")
    (OUT / "summary.md").write_text("".join(md_lines))
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))

    # ── Quick plot ────────────────────────────────────────────────────
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 1, figsize=(10, 4))
        for name, col in [("NSRAM", "C0"), ("LIF", "C1"), ("Random", "C2")]:
            yt = np.asarray(mg_first_pred[name]["y_true"])
            yp = np.asarray(mg_first_pred[name]["y_pred"])
            if name == "NSRAM":
                ax.plot(yt[:200], "k-", lw=1.5, label="target")
            ax.plot(yp[:200], color=col, lw=1.0, alpha=0.8,
                    label=f"{name} (NRMSE={mg_summary[name]['mean']:.3f})")
        ax.set_title("DS-N10a Mackey-Glass τ=17 — 1-step prediction (first seed)")
        ax.set_xlabel("step"); ax.set_ylabel("x (normalized)")
        ax.legend(loc="upper right", fontsize=8)
        fig.tight_layout()
        fig.savefig(OUT / "predicted_vs_actual.png", dpi=120)
        plt.close(fig)
    except Exception as e:
        print(f"[plot] skipped: {e}", flush=True)

    print("\n" + "=" * 60)
    print(json.dumps(summary, indent=2))
    print("=" * 60)


if __name__ == "__main__":
    main()
