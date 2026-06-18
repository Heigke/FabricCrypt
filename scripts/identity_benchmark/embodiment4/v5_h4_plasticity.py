"""V5-H4: Thermal-modulated plasticity for continual learning (5-task Permuted MNIST surrogate).

Hypothesis:
  Plasticity (effective learning rate) gated by live APU thermal headroom:
    cold chip → high LR (acquire new task), hot chip → low LR (preserve old).
  vs. constant LR baseline matched to the mean.

We use a small MLP (so it actually finishes in budget) and a synthetic
"permuted feature" continual learning task with 5 tasks, 1000 train samples
each, evaluated on all-tasks accuracy after sequential training.

The "thermal" signal here is the LIVE APU temp during training: as load
grows the chip heats up and the LR is throttled. Baseline uses a single
LR equal to the time-averaged modulated LR (so total "learning budget"
is matched).

Gate: thermal-modulated continual-learning final mean-task accuracy
≥ 5% absolute higher than constant-LR baseline.
"""
from __future__ import annotations
import json, time, socket
from pathlib import Path
import numpy as np

ROOT = Path("/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy")
HOST = socket.gethostname()
OUT = ROOT / f"results/IDENTITY_BENCHMARK_2026-05-30/embodiment4/v5_h4_{HOST}.json"

N_FEAT = 64
N_HID = 64
N_CLASS = 10
N_TASKS = 5
N_PER_TASK = 1000
N_EPOCHS = 3


def read_apu_temp_c() -> float:
    try:
        return float(open("/sys/class/thermal/thermal_zone0/temp").read().strip()) / 1000.0
    except Exception:
        return 50.0


def make_task(seed: int):
    rng = np.random.default_rng(seed)
    perm = rng.permutation(N_FEAT)
    # cluster centers for 10 classes
    centers = rng.standard_normal((N_CLASS, N_FEAT)) * 1.5
    y = rng.integers(0, N_CLASS, N_PER_TASK)
    X = centers[y] + 0.5 * rng.standard_normal((N_PER_TASK, N_FEAT))
    X = X[:, perm]
    # holdout
    y_te = rng.integers(0, N_CLASS, 300)
    X_te = centers[y_te] + 0.5 * rng.standard_normal((300, N_FEAT))
    X_te = X_te[:, perm]
    return X, y, X_te, y_te


def softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def init_mlp(seed):
    rng = np.random.default_rng(seed)
    W1 = rng.standard_normal((N_FEAT, N_HID)) * np.sqrt(2.0 / N_FEAT)
    b1 = np.zeros(N_HID)
    W2 = rng.standard_normal((N_HID, N_CLASS)) * np.sqrt(2.0 / N_HID)
    b2 = np.zeros(N_CLASS)
    return [W1, b1, W2, b2]


def forward(p, X):
    z1 = X @ p[0] + p[1]
    h = np.maximum(0, z1)
    z2 = h @ p[2] + p[3]
    return softmax(z2), h, z1


def step(p, X, y, lr):
    probs, h, z1 = forward(p, X)
    n = X.shape[0]
    onehot = np.zeros((n, N_CLASS)); onehot[np.arange(n), y] = 1
    d2 = (probs - onehot) / n
    gW2 = h.T @ d2; gb2 = d2.sum(0)
    dh = d2 @ p[2].T
    dh[z1 <= 0] = 0
    gW1 = X.T @ dh; gb1 = dh.sum(0)
    p[0] -= lr * gW1; p[1] -= lr * gb1
    p[2] -= lr * gW2; p[3] -= lr * gb2


def acc(p, X, y):
    probs, _, _ = forward(p, X)
    return float((probs.argmax(axis=1) == y).mean())


def train_continual(p, tasks, lr_fn, log):
    used_lrs = []
    for ti, (Xtr, ytr, _, _) in enumerate(tasks):
        for ep in range(N_EPOCHS):
            idx = np.random.permutation(len(ytr))
            for i in range(0, len(idx), 64):
                b = idx[i:i+64]
                lr = lr_fn()
                used_lrs.append(lr)
                step(p, Xtr[b], ytr[b], lr)
    log["used_lrs_mean"] = float(np.mean(used_lrs))
    log["used_lrs_std"] = float(np.std(used_lrs))
    return p


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    tasks = [make_task(i * 17 + 3) for i in range(N_TASKS)]

    res = {"host": HOST, "n_tasks": N_TASKS, "n_per_task": N_PER_TASK}

    # Run thermal-modulated first to gather LR distribution
    LR_HIGH, LR_LOW = 0.03, 0.003
    def thermal_lr():
        t = read_apu_temp_c()
        # linear interp: 50C → LR_HIGH, 80C → LR_LOW
        alpha = max(0.0, min(1.0, (t - 50.0) / 30.0))
        return LR_HIGH * (1 - alpha) + LR_LOW * alpha
    p_th = init_mlp(seed=42)
    th_log = {}
    train_continual(p_th, tasks, thermal_lr, th_log)
    accs_th = [acc(p_th, X_te, y_te) for (_, _, X_te, y_te) in tasks]
    res["thermal"] = {"per_task_acc": accs_th, "mean_acc": float(np.mean(accs_th)),
                       "lr_log": th_log}
    print(f"[H4] thermal: mean_acc={np.mean(accs_th):.4f} lrs_mean={th_log['used_lrs_mean']:.5f}", flush=True)

    # Constant-LR baseline (match the time-averaged thermal LR)
    matched_lr = th_log["used_lrs_mean"]
    def const_lr():
        return matched_lr
    p_c = init_mlp(seed=42)
    c_log = {}
    train_continual(p_c, tasks, const_lr, c_log)
    accs_c = [acc(p_c, X_te, y_te) for (_, _, X_te, y_te) in tasks]
    res["constant"] = {"per_task_acc": accs_c, "mean_acc": float(np.mean(accs_c)),
                        "lr": matched_lr, "lr_log": c_log}
    print(f"[H4] const lr={matched_lr:.5f}: mean_acc={np.mean(accs_c):.4f}", flush=True)

    # High-LR-only (acquisition focused, catastrophic forgetting expected)
    def high_lr():
        return LR_HIGH
    p_h = init_mlp(seed=42)
    train_continual(p_h, tasks, high_lr, {})
    accs_h = [acc(p_h, X_te, y_te) for (_, _, X_te, y_te) in tasks]
    res["high_lr_ref"] = {"per_task_acc": accs_h, "mean_acc": float(np.mean(accs_h))}

    res["gain_pct_thermal_vs_const"] = 100.0 * (np.mean(accs_th) - np.mean(accs_c)) / max(1e-9, np.mean(accs_c))
    res["abs_gain_thermal_vs_const"] = float(np.mean(accs_th) - np.mean(accs_c))
    res["WIN"] = res["abs_gain_thermal_vs_const"] >= 0.05
    OUT.write_text(json.dumps(res, indent=2, default=str))
    print(f"[H4] gain={res['abs_gain_thermal_vs_const']:+.4f} ({res['gain_pct_thermal_vs_const']:+.1f}%) WIN={res['WIN']}", flush=True)


if __name__ == "__main__":
    main()
