"""Shared utilities for embodiment2: envelope hashing, reservoir construction,
NARMA-10, Mackey-Glass, ridge regression, multi-axis structure derivation.

Inherits philosophy from embodiment/phase_c_run.py but extends:
  - derive_structure_v2(): 5 structural axes instead of 3 (adds weight scale + leak per neuron)
  - More tasks: NARMA-10, Mackey-Glass-17, MemoryCapacity, sinusoid generation
  - Per-position weight scale derived from envelope
"""
from __future__ import annotations
import hashlib, json
from pathlib import Path
import numpy as np

ROOT = Path("/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy")
PA = ROOT / "results/IDENTITY_BENCHMARK_2026-05-30/embodiment/phase_a"
OUT2 = ROOT / "results/IDENTITY_BENCHMARK_2026-05-30/embodiment2"
ACT_CHOICES = ["tanh", "relu", "sigmoid", "swish", "gelu"]
WASHOUT = 100


def load_vec(name):
    f = PA / f"A1_{name}.json"
    return json.loads(f.read_text())["vec23"]


def env_hash(vec, n_bytes=8192):
    arr = np.asarray(vec, dtype=np.float64)
    rounded = np.round(arr, 1).tobytes()
    return hashlib.shake_256(rounded).digest(n_bytes)


def derive_structure_v2(vec, N):
    """5-axis structure: mask, acts, perm, weight_scale, leak_per_neuron."""
    bits_needed = N * N
    bytes_needed = bits_needed // 8
    n_bytes = bytes_needed * 2 + N * 6  # 2 masks + 1B acts + 1B perm + 4B float scale + 4B float leak
    h = env_hash(vec, n_bytes=max(n_bytes, 8192))
    flat = np.unpackbits(np.frombuffer(h[:bytes_needed], dtype=np.uint8))
    h2 = np.unpackbits(np.frombuffer(h[bytes_needed:2*bytes_needed], dtype=np.uint8))
    mask = (flat & h2).reshape(N, N).astype(bool)
    np.fill_diagonal(mask, False)
    off = 2 * bytes_needed
    act_bytes = np.frombuffer(h[off:off+N], dtype=np.uint8); off += N
    acts = [ACT_CHOICES[b % len(ACT_CHOICES)] for b in act_bytes]
    perm_bytes = np.frombuffer(h[off:off+N], dtype=np.uint8); off += N
    perm = np.argsort(perm_bytes).astype(np.int32)
    # weight scale per neuron (column scale) — extra structural axis D2
    scale_raw = np.frombuffer(h[off:off+2*N], dtype=np.uint16).astype(np.float64) / 65535.0
    off += 2 * N
    weight_scale = 0.5 + 1.0 * scale_raw  # [0.5, 1.5]
    # per-neuron leak — extra structural axis D2
    leak_raw = np.frombuffer(h[off:off+2*N], dtype=np.uint16).astype(np.float64) / 65535.0
    off += 2 * N
    leak = 0.1 + 0.5 * leak_raw  # [0.1, 0.6]
    return {"mask": mask, "acts": acts, "perm": perm,
            "weight_scale": weight_scale, "leak": leak}


def baseline_structure_v2(N, seed=0):
    rng = np.random.default_rng(seed)
    mask = (rng.random((N, N)) < 0.30); np.fill_diagonal(mask, False)
    return {"mask": mask,
            "acts": ["tanh"] * N,
            "perm": np.arange(N, dtype=np.int32),
            "weight_scale": np.ones(N),
            "leak": np.full(N, 0.3)}


def apply_act(name, x):
    if name == "tanh": return np.tanh(x)
    if name == "relu": return np.maximum(0.0, x)
    if name == "sigmoid": return 1.0 / (1.0 + np.exp(-x))
    if name == "swish": return x / (1.0 + np.exp(-x))
    if name == "gelu": return 0.5 * x * (1.0 + np.tanh(np.sqrt(2/np.pi) * (x + 0.044715 * x**3)))
    raise ValueError(name)


def build_reservoir(struct, N, seed=0, spectral_radius=0.95, input_scale=1.0):
    rng = np.random.default_rng(seed)
    W = rng.standard_normal((N, N)) / np.sqrt(N)
    W = W * struct["mask"]
    W = W * struct["weight_scale"][None, :]  # column scale = per-source neuron contribution scale
    try:
        rho = float(np.max(np.abs(np.linalg.eigvals(W))))
    except Exception:
        rho = 1.0
    if rho > 1e-9:
        W *= (spectral_radius / rho)
    Win = rng.standard_normal((N, 1)) * input_scale
    return W, Win


def run_reservoir(u, W, Win, struct, N):
    T = len(u); x = np.zeros(N); X = np.zeros((T, N))
    acts = struct["acts"]; perm = struct["perm"]; leak = struct["leak"]
    # group neurons by activation kind for vectorisation
    kinds = {}
    for i, a in enumerate(acts):
        kinds.setdefault(a, []).append(i)
    kind_idx = {k: np.asarray(v, dtype=np.int32) for k, v in kinds.items()}
    for t in range(T):
        pre = W @ x + Win[:, 0] * u[t]
        post = np.empty(N)
        for k, idx in kind_idx.items():
            post[idx] = apply_act(k, pre[idx])
        x_new = np.empty(N)
        x_new[perm] = (1 - leak[perm]) * x[perm] + leak[perm] * post
        x = x_new
        X[t] = x
    return X


def ridge_fit(X, y, alpha=1e-6):
    Xb = np.concatenate([X, np.ones((X.shape[0], 1))], axis=1)
    A = Xb.T @ Xb + alpha * np.eye(Xb.shape[1])
    return np.linalg.solve(A, Xb.T @ y)


def ridge_predict(X, W):
    Xb = np.concatenate([X, np.ones((X.shape[0], 1))], axis=1)
    return Xb @ W


def nrmse(y, yh):
    return float(np.sqrt(np.mean((y - yh) ** 2)) / (np.std(y) + 1e-12))


# ===== Tasks =====
def narma10(T, seed=0):
    rng = np.random.default_rng(seed)
    u = 0.5 * rng.uniform(0.0, 1.0, size=T + 10)
    y = np.zeros(T + 10)
    for t in range(10, T + 10):
        y[t] = 0.3*y[t-1] + 0.05*y[t-1]*np.sum(y[t-10:t]) + 1.5*u[t-10]*u[t-1] + 0.1
    return u[10:], y[10:]


def mackey_glass(T, tau=17, seed=0, dt=1.0, burn=200):
    rng = np.random.default_rng(seed)
    n_total = T + burn + tau
    x = 1.2 + 0.1 * rng.standard_normal(n_total)
    for t in range(tau, n_total - 1):
        x[t+1] = x[t] + dt * (0.2 * x[t-tau] / (1.0 + x[t-tau]**10) - 0.1 * x[t])
    # input = x[t], target = x[t+5] (5-step ahead prediction)
    x = x[burn:]
    u = x[:T]; y = x[5:T+5]
    return u, y


def memory_capacity_task(T, seed=0, max_lag=30):
    """Input: white noise. Targets: u[t-k] for k=1..max_lag. MC = sum_k r²(y_hat_k, u[t-k])."""
    rng = np.random.default_rng(seed)
    u = rng.uniform(-1, 1, size=T + max_lag)
    targets = np.stack([u[max_lag - k : max_lag - k + T] for k in range(1, max_lag + 1)], axis=1)
    return u[max_lag:], targets


def sinusoid_freq_task(T, seed=0):
    """Input: scalar freq control u[t] in [0,1]. Target: sin(2*pi*freq*t)."""
    rng = np.random.default_rng(seed)
    f_change = rng.uniform(0.01, 0.05, size=T + 1)
    f_change = np.cumsum(rng.standard_normal(T) * 0.001 + 0.02)
    f_change = (f_change - f_change.min()) / (np.ptp(f_change) + 1e-9)
    f_change = 0.02 + 0.08 * f_change
    phase = np.cumsum(2 * np.pi * f_change)
    y = np.sin(phase)
    return f_change, y


# ===== Train / eval =====
def train_eval_task(struct, N, seed, task="narma10", T_tr=2000, T_te=500):
    if task == "narma10":
        u_tr, y_tr = narma10(T_tr, seed=seed*13+7); u_te, y_te = narma10(T_te, seed=seed*13+9991)
        targets_tr = y_tr; targets_te = y_te
    elif task == "mackey17":
        u_tr, y_tr = mackey_glass(T_tr, seed=seed*13+7); u_te, y_te = mackey_glass(T_te, seed=seed*13+9991)
        targets_tr = y_tr; targets_te = y_te
    elif task == "memcap":
        u_tr, targets_tr = memory_capacity_task(T_tr, seed=seed*13+7)
        u_te, targets_te = memory_capacity_task(T_te, seed=seed*13+9991)
    elif task == "sinusoid":
        u_tr, y_tr = sinusoid_freq_task(T_tr, seed=seed*13+7)
        u_te, y_te = sinusoid_freq_task(T_te, seed=seed*13+9991)
        targets_tr = y_tr; targets_te = y_te
    else:
        raise ValueError(task)
    W, Win = build_reservoir(struct, N, seed=seed)
    X_tr = run_reservoir(u_tr, W, Win, struct, N)
    Wout = ridge_fit(X_tr[WASHOUT:], targets_tr[WASHOUT:])
    X_te = run_reservoir(u_te, W, Win, struct, N)
    y_hat = ridge_predict(X_te[WASHOUT:], Wout)
    weights = {"W": W, "Win": Win, "Wout": Wout}
    if task == "memcap":
        # memory capacity score
        score = 0.0
        for k in range(targets_te.shape[1]):
            r = np.corrcoef(y_hat[:, k], targets_te[WASHOUT:, k])[0, 1]
            if np.isfinite(r):
                score += r * r
        return -score, weights  # negative MC so "lower is better" matches NRMSE convention
    return nrmse(targets_te[WASHOUT:], y_hat), weights


def transplant_eval(weights, struct_new, N, seed, task="narma10", T_te=500):
    if task == "narma10":
        u_te, y_te = narma10(T_te, seed=seed*13+9991); targets_te = y_te
    elif task == "mackey17":
        u_te, y_te = mackey_glass(T_te, seed=seed*13+9991); targets_te = y_te
    elif task == "memcap":
        u_te, targets_te = memory_capacity_task(T_te, seed=seed*13+9991)
    elif task == "sinusoid":
        u_te, y_te = sinusoid_freq_task(T_te, seed=seed*13+9991); targets_te = y_te
    W = weights["W"] * struct_new["mask"]
    W = W * struct_new["weight_scale"][None, :]
    Win = weights["Win"]
    X_te = run_reservoir(u_te, W, Win, struct_new, N)
    y_hat = ridge_predict(X_te[WASHOUT:], weights["Wout"])
    if task == "memcap":
        score = 0.0
        for k in range(targets_te.shape[1]):
            r = np.corrcoef(y_hat[:, k], targets_te[WASHOUT:, k])[0, 1]
            if np.isfinite(r): score += r * r
        return -score
    return nrmse(targets_te[WASHOUT:], y_hat)
