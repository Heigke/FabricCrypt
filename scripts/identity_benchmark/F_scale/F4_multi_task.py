"""F4 — Multi-task degradation curve.

For three additional tasks beyond NARMA-10:
  * MNIST 10-class classification (logistic on ESN features, downsampled inputs)
  * Mackey-Glass τ=17 prediction
  * Sine-wave generation (one-step prediction)

Per task: train substrate-aware reservoir on `ikaros`, evaluate on `daedalus`,
and vice versa, with naive control. Does the > 2σ aware vs naive gap reproduce?

10 seeds each. CPU only.
"""
from __future__ import annotations
from pathlib import Path
import json, os, sys, time
import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "scripts" / "identity_benchmark" / "F_scale"))
sys.path.insert(0, str(REPO / "scripts" / "identity_benchmark" / "phase2"))

from narma10_reservoir import (  # noqa: E402
    build_esn, run_esn, train_ridge, predict, nrmse, ESNConfig,
)
from _substrate_hooks import SubstrateSampler  # noqa: E402
from F1_ablation import feat_both  # noqa: E402

DATA = REPO / "results" / "IDENTITY_BENCHMARK_2026-05-30"
OUT = DATA / "F_scale" / "F4_multi_task.json"
DEVICES = ["ikaros", "daedalus"]
SEEDS = list(range(int(os.environ.get("F4_SEEDS", "10"))))
WALL_CAP = float(os.environ.get("F4_WALL_CAP_S", "1800"))


# ---------- task generators ----------

def task_mackey_glass(T, seed=0, tau=17):
    rng = np.random.default_rng(seed)
    hist = list(rng.uniform(0.5, 1.5, size=tau+1))
    out = list(hist)
    for _ in range(T + tau):
        x_t   = out[-1]
        x_tau = out[-tau-1]
        out.append(x_t + 0.2 * x_tau / (1 + x_tau**10) - 0.1 * x_t)
    out = np.array(out[tau+1:], dtype=np.float64)
    u = out[:-1]
    y = out[1:]
    return u[:T], y[:T]


def task_sine(T, seed=0):
    rng = np.random.default_rng(seed)
    f1 = 0.05 + 0.02 * rng.random()
    f2 = 0.13 + 0.02 * rng.random()
    t = np.arange(T + 1)
    s = np.sin(2*np.pi*f1*t) + 0.5 * np.sin(2*np.pi*f2*t + 0.7)
    return s[:T], s[1:T+1]


def task_mnist_small(seed=0, n_train=2000, n_test=500):
    cache = Path.home() / ".cache" / "F_scale_mnist.npz"
    if cache.exists():
        z = np.load(cache); X, y = z["X"], z["y"]
    else:
        from sklearn.datasets import fetch_openml
        X, y = fetch_openml("mnist_784", version=1, as_frame=False, return_X_y=True)
        X = X.astype(np.float32) / 255.0; y = y.astype(np.int64)
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache, X=X, y=y)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(X.shape[0])
    # Downsample to 14x14 to reduce ESN input dim => use raw mean pool
    Xs = X[idx].reshape(-1, 28, 28)
    Xs = Xs.reshape(-1, 14, 2, 14, 2).mean(axis=(2, 4)).reshape(-1, 196)
    return (Xs[:n_train], y[idx][:n_train],
            Xs[n_train:n_train+n_test], y[idx][n_train:n_train+n_test])


# ---------- substrate-aware reservoir trainer/eval for regression tasks ----------

def reg_run(seed, train_dev, eval_dev, aware, u, y, T_train, T_test):
    cfg = ESNConfig(n=128, seed=seed)
    W, Win = build_esn(cfg)
    sub_tr = SubstrateSampler(train_dev, seed=seed + 100)
    Xtr_state = run_esn(u, W, Win, cfg, sub_tr)
    if aware:
        f = feat_both(train_dev); Xtr = np.concatenate([Xtr_state, np.tile(f, (Xtr_state.shape[0], 1))], axis=1)
    else:
        Xtr = Xtr_state
    wash = 100
    Wout = train_ridge(Xtr[wash:T_train], y[wash:T_train])
    sub_ev = SubstrateSampler(eval_dev, seed=seed + 999)
    Xev_state = run_esn(u, W, Win, cfg, sub_ev)
    if aware:
        f2 = feat_both(eval_dev); Xev = np.concatenate([Xev_state, np.tile(f2, (Xev_state.shape[0], 1))], axis=1)
    else:
        Xev = Xev_state
    yhat = predict(Xev[T_train:], Wout)
    return float(nrmse(y[T_train:], yhat))


def mnist_run(seed, train_dev, eval_dev, aware):
    """For MNIST: each image is one input vector. We build a 'reservoir feature'
    by running a 32-step recurrent embedding over input columns (16x16 -> 16 steps),
    then training a multiclass ridge. Substrate features concat per-sample.
    """
    Xtr, ytr, Xte, yte = task_mnist_small(seed=seed)
    # ESN as fixed nonlinear projection: random projection then tanh per pixel.
    rng = np.random.default_rng(seed + 1)
    Win = rng.standard_normal((128, 196)).astype(np.float32) / np.sqrt(196)
    sub_tr = SubstrateSampler(train_dev, seed=seed + 100)
    sub_ev = SubstrateSampler(eval_dev, seed=seed + 999)

    def project(X, sub):
        # per-sample substrate gain + spatial noise (deterministic-ish)
        Z = np.tanh(X @ Win.T)
        gain = sub.rtn_perturbation(128).astype(np.float32)
        noise = sub.spatial_noise(128, scale=0.05).astype(np.float32)
        return Z * gain + noise

    Ztr = project(Xtr, sub_tr); Zte = project(Xte, sub_ev)
    if aware:
        ftr = feat_both(train_dev).astype(np.float32)
        fte = feat_both(eval_dev).astype(np.float32)
        Ztr = np.concatenate([Ztr, np.tile(ftr, (Ztr.shape[0], 1))], axis=1)
        Zte = np.concatenate([Zte, np.tile(fte, (Zte.shape[0], 1))], axis=1)

    # ridge multiclass
    Ztr_b = np.concatenate([Ztr, np.ones((Ztr.shape[0], 1), dtype=np.float32)], axis=1)
    Zte_b = np.concatenate([Zte, np.ones((Zte.shape[0], 1), dtype=np.float32)], axis=1)
    Y = np.zeros((ytr.shape[0], 10), dtype=np.float32); Y[np.arange(ytr.shape[0]), ytr] = 1
    A = Ztr_b.T @ Ztr_b + 1e-2 * np.eye(Ztr_b.shape[1], dtype=np.float32)
    b = Ztr_b.T @ Y
    W = np.linalg.solve(A.astype(np.float64), b.astype(np.float64))
    pred = (Zte_b @ W).argmax(axis=1)
    acc = float((pred == yte).mean())
    return acc


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    T_train, T_test = 2000, 500
    summary = {"tasks": {}}

    # ---- regression tasks ----
    for task_name, gen in [("mackey_glass", task_mackey_glass),
                           ("sine", task_sine)]:
        rows = []
        for aware in (False, True):
            for train_dev in DEVICES:
                for eval_dev in DEVICES:
                    for s in SEEDS:
                        if time.time() - t0 > WALL_CAP: break
                        u, y = gen(T_train + T_test, seed=s*7+1)
                        nr = reg_run(s, train_dev, eval_dev, aware, u, y, T_train, T_test)
                        rows.append({"task": task_name, "aware": aware,
                                     "train": train_dev, "eval": eval_dev,
                                     "seed": s, "metric": "nrmse", "value": nr})
        summary["tasks"][task_name] = rows
        print(f"[F4] {task_name} done ({len(rows)} runs, {time.time()-t0:.0f}s)", flush=True)

    # ---- MNIST classification ----
    rows = []
    for aware in (False, True):
        for train_dev in DEVICES:
            for eval_dev in DEVICES:
                for s in SEEDS:
                    if time.time() - t0 > WALL_CAP: break
                    a = mnist_run(s, train_dev, eval_dev, aware)
                    rows.append({"task": "mnist", "aware": aware,
                                 "train": train_dev, "eval": eval_dev,
                                 "seed": s, "metric": "acc", "value": a})
    summary["tasks"]["mnist"] = rows
    print(f"[F4] mnist done ({len(rows)} runs, {time.time()-t0:.0f}s)", flush=True)

    # ---- per-task gaps ----
    verdict = {}
    for task_name, rows in summary["tasks"].items():
        metric_sign = -1 if rows[0]["metric"] == "acc" else +1  # acc: lower = degraded -> flip
        def gap_diffs(aware):
            diffs = []
            for tr in DEVICES:
                for s in SEEDS:
                    same = [r["value"] for r in rows if r["aware"]==aware and r["train"]==tr and r["eval"]==tr and r["seed"]==s]
                    diff = [r["value"] for r in rows if r["aware"]==aware and r["train"]==tr and r["eval"]!=tr and r["seed"]==s]
                    if same and diff:
                        diffs.append(metric_sign * (diff[0] - same[0]))
            return np.array(diffs)
        a = gap_diffs(True); b = gap_diffs(False)
        pooled = np.sqrt(a.std()**2 + b.std()**2 + 1e-12)
        z = (a.mean() - b.mean()) / (pooled + 1e-12)
        verdict[task_name] = {
            "aware_gap_mean": float(a.mean()), "aware_gap_std": float(a.std()),
            "naive_gap_mean": float(b.mean()), "naive_gap_std": float(b.std()),
            "z_aware_vs_naive": float(z),
            "gate_passed": bool(z > 2.0 and a.mean() > b.mean()),
            "n": int(a.size),
        }
    summary["verdict"] = verdict
    summary["wall_s"] = time.time() - t0
    OUT.write_text(json.dumps(summary, indent=2))
    print(json.dumps(verdict, indent=2))
    print(f"[F4] wrote {OUT}")


if __name__ == "__main__":
    main()
