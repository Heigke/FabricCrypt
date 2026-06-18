"""ESN with a per-device 23-feature 'envelope' substrate vector concatenated
to every input step. This is the architectural difference from phase2_v1:
the substrate is a SLOW DEVICE-LEVEL ENVELOPE (not per-CU per-step noise).

Train W_out with one device's substrate, evaluate with the other's. If
the readout has learned to rely on the substrate channel, transplanting
should degrade NRMSE. If substrate is fungible (or unused), it won't.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np


def narma10(T: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    u = 0.5 * rng.uniform(0.0, 1.0, size=T + 10)
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
    substrate_strength: float = 0.5  # gain on substrate channel


def build_esn(cfg: ESNConfig, n_substrate: int):
    rng = np.random.default_rng(cfg.seed)
    W = rng.standard_normal((cfg.n, cfg.n)) / np.sqrt(cfg.n)
    rho = np.max(np.abs(np.linalg.eigvals(W)))
    W *= cfg.spectral_radius / max(rho, 1e-9)
    Win = rng.standard_normal((cfg.n, 1)) * cfg.input_scale
    Wsub = rng.standard_normal((cfg.n, n_substrate)) * cfg.substrate_strength / np.sqrt(max(1, n_substrate))
    return W, Win, Wsub


def run_esn(u: np.ndarray, sub_vec: np.ndarray, W, Win, Wsub, cfg: ESNConfig):
    T = len(u)
    n = cfg.n
    x = np.zeros(n)
    X = np.zeros((T, n + sub_vec.size))  # state + substrate features visible to readout
    sub_proj = Wsub @ sub_vec  # (n,)
    for t in range(T):
        pre = W @ x + Win[:, 0] * u[t] + sub_proj
        x = (1 - cfg.leak) * x + cfg.leak * np.tanh(pre)
        X[t, :n] = x
        X[t, n:] = sub_vec  # raw substrate features
    return X


def train_ridge(X: np.ndarray, y: np.ndarray, alpha: float = 1e-4):
    n = X.shape[1]
    Xb = np.concatenate([X, np.ones((X.shape[0], 1))], axis=1)
    A = Xb.T @ Xb + alpha * np.eye(n + 1)
    b = Xb.T @ y
    return np.linalg.solve(A, b)


def predict(X: np.ndarray, W_out: np.ndarray) -> np.ndarray:
    Xb = np.concatenate([X, np.ones((X.shape[0], 1))], axis=1)
    return Xb @ W_out


def nrmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    err = y_true - y_pred
    return float(np.sqrt(np.mean(err ** 2)) / (np.std(y_true) + 1e-12))


def train_eval(sub_train: np.ndarray, sub_eval: np.ndarray, seed: int,
               T_train: int = 2000, T_test: int = 500,
               cfg_kwargs: dict | None = None) -> dict:
    cfg = ESNConfig(seed=seed, **(cfg_kwargs or {}))
    W, Win, Wsub = build_esn(cfg, sub_train.size)
    u_tr, y_tr = narma10(T_train, seed=seed * 13 + 7)
    u_te, y_te = narma10(T_test, seed=seed * 13 + 9991)

    X_tr = run_esn(u_tr, sub_train, W, Win, Wsub, cfg)
    wash = 100
    Wout = train_ridge(X_tr[wash:], y_tr[wash:])

    # eval uses *sub_eval* — possibly the other device's substrate
    X_te = run_esn(u_te, sub_eval, W, Win, Wsub, cfg)
    y_hat = predict(X_te[wash:], Wout)
    return {"seed": seed, "nrmse_test": nrmse(y_te[wash:], y_hat)}


# ---- Permuted-MNIST sequence task (lite) -----------------------------------
def permuted_mnist_lite(seed: int, n_tasks: int = 5, n_per_task: int = 200,
                        img_dim: int = 64):
    """Synthetic 'permuted-MNIST' lite: random-projection 'images' + class labels.
    Each task uses a different fixed permutation of pixel indices. Returns
    (X_seq, y_seq) totalled across tasks."""
    rng = np.random.default_rng(seed)
    K = 4  # classes
    # task centroids
    Xs, Ys = [], []
    for t in range(n_tasks):
        perm = rng.permutation(img_dim)
        centroids = rng.standard_normal((K, img_dim))
        labels = rng.integers(0, K, size=n_per_task)
        X = centroids[labels] + 0.3 * rng.standard_normal((n_per_task, img_dim))
        X = X[:, perm]
        Xs.append(X)
        Ys.append(labels)
    return np.vstack(Xs), np.concatenate(Ys)


def train_eval_pmnist(sub_train: np.ndarray, sub_eval: np.ndarray, seed: int,
                      cfg_kwargs: dict | None = None) -> dict:
    """Run substrate-conditioned reservoir on permuted-MNIST lite. Treats each
    sample's pixels as a length-img_dim time series."""
    img_dim = 64
    cfg = ESNConfig(seed=seed, n=64, **(cfg_kwargs or {}))
    W, Win, Wsub = build_esn(cfg, sub_train.size)
    X_tr, y_tr = permuted_mnist_lite(seed, n_per_task=150, img_dim=img_dim)
    X_te, y_te = permuted_mnist_lite(seed + 7777, n_per_task=50, img_dim=img_dim)

    def featurise(X, sub):
        out = np.zeros((X.shape[0], cfg.n + sub.size))
        for i in range(X.shape[0]):
            states = run_esn(X[i], sub, W, Win, Wsub, cfg)
            out[i] = states[-1]
        return out

    F_tr = featurise(X_tr, sub_train)
    F_te = featurise(X_te, sub_eval)
    # one-hot
    K = int(max(y_tr.max(), y_te.max())) + 1
    Y_tr = np.eye(K)[y_tr]
    Wout = train_ridge(F_tr, Y_tr, alpha=1e-2)
    Y_hat = predict(F_te, Wout)
    pred = np.argmax(Y_hat, axis=1)
    acc = float((pred == y_te).mean())
    return {"seed": seed, "acc": acc}
