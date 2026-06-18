"""F2 — Permuted-MNIST continual learning, substrate-aware vs naive.

Sequence of 5 permutations of MNIST (5000 train / 1000 test each). A SHARED
linear readout is trained sequentially on tasks 1..5; we measure
catastrophic forgetting on task 1 after each subsequent task.

Substrate-aware condition: the input is augmented with the device's substrate
feature vector (`feat_both` from F1). Naive: no augmentation.

Then we transplant: train the FULL sequence on 'ikaros', evaluate task 1 on
'daedalus' substrate features. Does substrate awareness create more forgetting
under transplant? Honest report.

10 seeds. CPU only. ~3 min per seed target.
"""
from __future__ import annotations
from pathlib import Path
import json, os, sys, time
import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "scripts" / "identity_benchmark" / "F_scale"))
sys.path.insert(0, str(REPO / "scripts" / "identity_benchmark" / "phase2"))

from F1_ablation import feat_both  # noqa: E402

DATA = REPO / "results" / "IDENTITY_BENCHMARK_2026-05-30"
OUT = DATA / "F_scale" / "F2_permuted_mnist.json"
DEVICES = ["ikaros", "daedalus"]
N_TASKS = 5
N_TRAIN = 5000
N_TEST = 1000
SEEDS = list(range(int(os.environ.get("F2_SEEDS", "10"))))
WALL_CAP = float(os.environ.get("F2_WALL_CAP_S", "2400"))


def load_mnist():
    from sklearn.datasets import fetch_openml
    cache = Path.home() / ".cache" / "F_scale_mnist.npz"
    if cache.exists():
        z = np.load(cache)
        return z["X"], z["y"]
    X, y = fetch_openml("mnist_784", version=1, as_frame=False, return_X_y=True)
    X = (X.astype(np.float32) / 255.0)
    y = y.astype(np.int64)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, X=X, y=y)
    return X, y


def make_tasks(X, y, seed):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(X.shape[0])
    tr = idx[:N_TRAIN * N_TASKS]; te = idx[N_TRAIN * N_TASKS : N_TRAIN * N_TASKS + N_TEST * N_TASKS]
    tasks = []
    perms = []
    for t in range(N_TASKS):
        perm = rng.permutation(X.shape[1])
        perms.append(perm)
        Xtr = X[tr[t*N_TRAIN:(t+1)*N_TRAIN]][:, perm]
        ytr = y[tr[t*N_TRAIN:(t+1)*N_TRAIN]]
        Xte = X[te[t*N_TEST:(t+1)*N_TEST]][:, perm]
        yte = y[te[t*N_TEST:(t+1)*N_TEST]]
        tasks.append((Xtr, ytr, Xte, yte))
    return tasks


def onehot(y, k=10):
    Y = np.zeros((y.shape[0], k), dtype=np.float32)
    Y[np.arange(y.shape[0]), y] = 1.0
    return Y


class OnlineRidge:
    """Sequential ridge: keep running A=XᵀX+αI and b=XᵀY. Train_all gives
    Wout. To inject 'continual learning + forgetting', we DECAY the running
    accumulators each task (decay=0.3 -> heavy forgetting weight)."""
    def __init__(self, d, k=10, alpha=1e-2, decay=0.5):
        self.A = alpha * np.eye(d, dtype=np.float32)
        self.b = np.zeros((d, k), dtype=np.float32)
        self.alpha = alpha
        self.decay = decay
        self.d = d

    def update(self, X, Y):
        # decay older sufficient stats to simulate plasticity
        self.A = self.decay * self.A + X.T @ X + self.alpha * np.eye(self.d, dtype=np.float32)
        self.b = self.decay * self.b + X.T @ Y

    def solve(self):
        return np.linalg.solve(self.A.astype(np.float64), self.b.astype(np.float64))


def acc(W, X, y):
    s = X @ W
    return float((s.argmax(axis=1) == y).mean())


def run_seed(seed, X, y, substrate_aware, train_dev, eval_dev_for_task1):
    tasks = make_tasks(X, y, seed)
    d = X.shape[1] + (feat_both(train_dev).shape[0] if substrate_aware else 0)
    rid = OnlineRidge(d=d + 1, k=10)  # +1 bias

    def augment(Xb, dev):
        if not substrate_aware:
            Xa = Xb
        else:
            f = feat_both(dev).astype(np.float32)
            Xa = np.concatenate([Xb, np.tile(f, (Xb.shape[0], 1))], axis=1)
        return np.concatenate([Xa, np.ones((Xa.shape[0], 1), dtype=np.float32)], axis=1)

    task1_acc_after = []  # task-1 test acc after each training round
    for t, (Xtr, ytr, Xte, yte) in enumerate(tasks):
        Xtr_aug = augment(Xtr, train_dev)
        rid.update(Xtr_aug, onehot(ytr))
        W = rid.solve()
        # Re-eval task 1 — using train_dev features (own substrate)
        Xt1_te = augment(tasks[0][2], train_dev)
        a1 = acc(W, Xt1_te, tasks[0][3])
        task1_acc_after.append(a1)

    # Final transplant: eval task 1 with WRONG device features
    W_final = rid.solve()
    if substrate_aware:
        Xt1_xplant = augment(tasks[0][2], eval_dev_for_task1)
    else:
        Xt1_xplant = augment(tasks[0][2], train_dev)
    a1_xplant = acc(W_final, Xt1_xplant, tasks[0][3])

    return {
        "seed": seed,
        "aware": substrate_aware,
        "train_dev": train_dev,
        "eval_dev": eval_dev_for_task1,
        "task1_acc_per_round": task1_acc_after,
        "task1_acc_final_own":  task1_acc_after[-1],
        "task1_acc_final_xplant": a1_xplant,
        "forgetting_own":   task1_acc_after[0] - task1_acc_after[-1],
        "transplant_delta": task1_acc_after[-1] - a1_xplant,  # >0 = transplant hurts
    }


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    print("[F2] loading MNIST...", flush=True)
    X, y = load_mnist()
    print(f"[F2] X={X.shape} y={y.shape}", flush=True)
    rows = []
    for aware in (False, True):
        for train_dev in DEVICES:
            other = "daedalus" if train_dev == "ikaros" else "ikaros"
            for s in SEEDS:
                if time.time() - t0 > WALL_CAP:
                    print(f"[F2] WALL CAP hit after {len(rows)} runs", flush=True)
                    break
                r = run_seed(s, X, y, aware, train_dev, other)
                rows.append(r)
                print(f"[F2] aware={aware} train={train_dev} seed={s} "
                      f"task1_own_final={r['task1_acc_final_own']:.3f} "
                      f"xplant={r['task1_acc_final_xplant']:.3f} "
                      f"dt={r['transplant_delta']:+.3f}", flush=True)

    # Summary: transplant_delta for aware vs naive
    def stats(filter_aware):
        d = np.array([r["transplant_delta"] for r in rows if r["aware"] == filter_aware])
        f = np.array([r["forgetting_own"]    for r in rows if r["aware"] == filter_aware])
        return {
            "n": int(d.size),
            "transplant_delta_mean": float(d.mean()) if d.size else None,
            "transplant_delta_std":  float(d.std())  if d.size else None,
            "forgetting_own_mean":   float(f.mean()) if f.size else None,
            "forgetting_own_std":    float(f.std())  if f.size else None,
        }
    summary = {
        "per_run": rows,
        "aware_True":  stats(True),
        "aware_False": stats(False),
        "wall_s": time.time() - t0,
    }
    a, b = summary["aware_True"], summary["aware_False"]
    if a["n"] and b["n"]:
        pooled = np.sqrt(a["transplant_delta_std"]**2 + b["transplant_delta_std"]**2 + 1e-12)
        z = (a["transplant_delta_mean"] - b["transplant_delta_mean"]) / (pooled + 1e-12)
        summary["z_aware_vs_naive_transplant"] = float(z)
        summary["gate_passed"] = bool(z > 2.0 and a["transplant_delta_mean"] > b["transplant_delta_mean"])
    OUT.write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if k != "per_run"}, indent=2))
    print(f"[F2] wrote {OUT}")


if __name__ == "__main__":
    main()
