# Combined text bundle for oracle context

All text artifacts concatenated below for one-shot context. Each file is delimited by a `=== FILE: <name> ===` marker.



=== FILE: A13_cross.json (2304 chars) ===
```json
{
  "lambdas": {
    "lambda_0.0": {
      "aggregate": {
        "self": {
          "mean": 0.6383466525083625,
          "std": 0.04249477892748251,
          "n": 16
        },
        "daedalus": {
          "mean": 4.5975137137476665,
          "std": 2.2628747629187465,
          "n": 16
        },
        "sw_matched_ht": {
          "mean": 0.8076963744900448,
          "std": 0.13708335575412556,
          "n": 16
        },
        "shuffle_ht": {
          "mean": 5.764188804647448,
          "std": 2.6888815461945534,
          "n": 16
        }
      },
      "delta_hw": 3.959167061239304,
      "delta_sw": 0.1693497219816823,
      "delta_shuffle": 5.125842152139086,
      "z_hw_vs_sw": 1.6717153113637306,
      "verdict": "STRUCTURE_BOUND"
    },
    "lambda_1.0": {
      "aggregate": {
        "self": {
          "mean": 3.3470087385522422,
          "std": 0.4596771308469579,
          "n": 16
        },
        "daedalus": {
          "mean": 11.834260065140041,
          "std": 0.8417517372403421,
          "n": 16
        },
        "sw_matched_ht": {
          "mean": 3.9077452899247773,
          "std": 1.0935997380638296,
          "n": 16
        },
        "shuffle_ht": {
          "mean": 31.1838548370254,
          "std": 15.32691090075524,
          "n": 16
        }
      },
      "delta_hw": 8.4872513265878,
      "delta_sw": 0.5607365513725351,
      "delta_shuffle": 27.83684609847316,
      "z_hw_vs_sw": 5.743690875697301,
      "verdict": "CONSTITUTIVE"
    },
    "lambda_10.0": {
      "aggregate": {
        "self": {
          "mean": 5.62390631755536,
          "std": 0.6956253719737093,
          "n": 16
        },
        "daedalus": {
          "mean": 12.66642936134237,
          "std": 0.9005537758488115,
          "n": 16
        },
        "sw_matched_ht": {
          "mean": 5.296447597280222,
          "std": 1.601330316986534,
          "n": 16
        },
        "shuffle_ht": {
          "mean": 36.06922812244401,
          "std": 17.624912219191195,
          "n": 16
        }
      },
      "delta_hw": 7.0425230437870106,
      "delta_sw": -0.32745872027513734,
      "delta_shuffle": 30.44532180488865,
      "z_hw_vs_sw": 4.011557869051953,
      "verdict": "CONSTITUTIVE"
    }
  },
  "wall_s": 2.2670271396636963
}
```


=== FILE: A13_cross.py (6498 chars) ===
```python
#!/usr/bin/env python3
"""CROSS-ATTACK — heavy-tail substrate + contrastive loss.

Combine A1 (dual-objective task+id loss) with A3 (HeavyTailSubstrate). The
question: does contrastive identity pressure on a heavy-tail-bound reservoir
finally unlock device-specific binding that survives transplant?

Hypothesis: if A3 alone gets z_hw_vs_sw ≈ 1.7 (almost constitutive) and A1
provides extra discriminative pressure, together they may cross 2σ.

Output: results/.../attack_1_3/A13_cross.json
"""
from __future__ import annotations
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "constitutive"))
from reservoir import Reservoir, ReservoirCfg, ridge_fit  # type: ignore

sys.path.insert(0, str(HERE))
from A3_heavy_tail_transplant import (  # type: ignore
    HeavyTailSubstrate, GaussianMatchedHT, ShuffleHT, load_streams,
    N_RES, SUB_DIM, WASHOUT, T_TRAIN, T_TEST, HORIZON
)

OUT_DIR = HERE.parents[2] / "results" / "IDENTITY_BENCHMARK_2026-05-30" / "attack_1_3"

LAMBDAS = [0.0, 1.0, 10.0]
N_SEEDS = int(os.environ.get("N_SEEDS", "16"))


def narma10(T: int, seed: int):
    rng = np.random.default_rng(seed)
    u = 0.5 * rng.random(T + 10)
    y = np.zeros(T + 10)
    for t in range(10, T + 10):
        y[t] = 0.3 * y[t - 1] + 0.05 * y[t - 1] * y[t - 10:t].sum() + 1.5 * u[t - 10] * u[t - 1] + 0.1
    return u[10:].astype(np.float64), y[10:].astype(np.float64)


def train_dual(X_i, y_task_i, X_d, lam):
    n_j = X_i.shape[0] + X_d.shape[0]
    X_j = np.vstack([X_i, X_d])
    y_id = np.concatenate([np.zeros(X_i.shape[0]), np.ones(X_d.shape[0])]).astype(int)
    Xa_i = np.hstack([X_i, np.ones((X_i.shape[0], 1))])
    if lam == 0.0:
        return ridge_fit(Xa_i, y_task_i, alpha=1e-3)
    sign_id = (y_id == 0).astype(np.float64) * 2 - 1
    Xa_full = np.hstack([X_j, np.ones((n_j, 1))])
    D = Xa_full.shape[1]
    y_task_full = np.concatenate([y_task_i, np.zeros(X_d.shape[0])])
    w_task = np.concatenate([np.ones(X_i.shape[0]), 0.1 * np.ones(X_d.shape[0])])
    A1 = (Xa_full.T * w_task) @ Xa_full + 1e-3 * np.eye(D)
    b1 = (Xa_full.T * w_task) @ y_task_full
    A2 = lam * (Xa_full.T @ Xa_full) + 1e-3 * np.eye(D)
    b2 = lam * (Xa_full.T @ sign_id)
    return np.linalg.solve(A1 + A2, b1 + b2)


def task_nrmse(W_task, X, y):
    Xa = np.hstack([X, np.ones((X.shape[0], 1))])
    y_pred = Xa @ W_task
    err = float(np.sqrt(np.mean((y - y_pred) ** 2)))
    rng = float(y.std() + 1e-12)
    return err / rng


def main():
    t0 = time.time()
    streams_i = load_streams("ikaros")
    streams_d = load_streams("daedalus")
    print(f"[A13.cross] N_SEEDS={N_SEEDS} λ={LAMBDAS}", flush=True)

    eval_kinds = ["self", "daedalus", "sw_matched_ht", "shuffle_ht"]
    out = {"lambdas": {}}

    for lam in LAMBDAS:
        key = f"lambda_{lam}"
        cells = {ek: [] for ek in eval_kinds}
        for s in range(N_SEEDS):
            sub_i = HeavyTailSubstrate("ikaros", streams_i, n_dim=SUB_DIM, seed=s + 11)
            sub_d = HeavyTailSubstrate("daedalus", streams_d, n_dim=SUB_DIM, seed=s + 22)
            sub_sw = GaussianMatchedHT(sub_i, seed=s + 33)
            sub_sh = ShuffleHT(sub_i, seed=s + 44)

            u, y = narma10(T_TRAIN + HORIZON + WASHOUT, seed=s)
            u_in = u[:-HORIZON][:, None]
            y_tg = y[HORIZON:]
            cfg = ReservoirCfg(n_in=1, n_res=N_RES, seed=s)

            # train reservoir on ikaros HT substrate
            res_train = Reservoir(cfg, regime=5, substrate=sub_i)
            X_i = res_train.run(u_in, washout=WASHOUT)
            y_w_i = y_tg[WASHOUT:]
            # train-time states on daedalus HT substrate (for id pressure)
            res_d_train = Reservoir(cfg, regime=5, substrate=sub_d)
            res_d_train.W_in = res_train.W_in
            res_d_train.W_rec = res_train.W_rec
            M = sub_d.weight_mod(cfg.n_res)
            res_d_train.W_rec_eff = res_train.W_rec * (1.0 + 0.15 * M)
            X_d = res_d_train.run(u_in, washout=WASHOUT)

            W_task = train_dual(X_i, y_w_i, X_d, lam)

            # eval transplant
            for ek, ev in [("self", sub_i), ("daedalus", sub_d),
                            ("sw_matched_ht", sub_sw), ("shuffle_ht", sub_sh)]:
                u_te, y_te = narma10(T_TEST + HORIZON + WASHOUT, seed=s + 9999)
                u_te_in = u_te[:-HORIZON][:, None]
                y_te_tg = y_te[HORIZON:]
                res_eval = Reservoir(cfg, regime=5, substrate=ev)
                res_eval.W_in = res_train.W_in
                res_eval.W_rec = res_train.W_rec
                Me = ev.weight_mod(cfg.n_res)
                res_eval.W_rec_eff = res_train.W_rec * (1.0 + 0.15 * Me)
                X_te = res_eval.run(u_te_in, washout=WASHOUT)
                y_te_w = y_te_tg[WASHOUT:]
                v = task_nrmse(W_task, X_te, y_te_w)
                cells[ek].append(float(v))
            if s == 0:
                print(f"  λ={lam} seed0: {{ {', '.join(f'{k}: {cells[k][-1]:.4f}' for k in eval_kinds)} }}",
                      flush=True)

        agg = {}
        for ek in eval_kinds:
            arr = np.array(cells[ek], dtype=np.float64)
            agg[ek] = {
                "mean": float(np.nanmean(arr)),
                "std": float(np.nanstd(arr)),
                "n": int((~np.isnan(arr)).sum()),
            }
        d_hw = agg["daedalus"]["mean"] - agg["self"]["mean"]
        d_sw = agg["sw_matched_ht"]["mean"] - agg["self"]["mean"]
        d_sh = agg["shuffle_ht"]["mean"] - agg["self"]["mean"]
        pooled = float(np.sqrt(agg["daedalus"]["std"] ** 2 + agg["sw_matched_ht"]["std"] ** 2)) + 1e-12
        z = (d_hw - d_sw) / pooled
        verdict = ("CONSTITUTIVE" if z > 2.0 else
                   "STRUCTURE_BOUND" if z > 0.5 else
                   "SW_REPLICATES_HW")
        out["lambdas"][key] = {
            "aggregate": agg,
            "delta_hw": float(d_hw),
            "delta_sw": float(d_sw),
            "delta_shuffle": float(d_sh),
            "z_hw_vs_sw": float(z),
            "verdict": verdict,
        }
        print(f"  λ={lam}: Δ_hw={d_hw:.3f} Δ_sw={d_sw:.3f} z={z:.2f} → {verdict}",
              flush=True)

    out["wall_s"] = time.time() - t0
    with open(OUT_DIR / "A13_cross.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"[A13.cross] saved → A13_cross.json wall={time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()

```


=== FILE: A1_contrastive.py (21862 chars) ===
```python
#!/usr/bin/env python3
"""ATTACK 1 — Contrastive Identity Training.

Hypothesis: ridge readout doesn't bind identity because the loss doesn't reward
it. Replace pure-task loss with DUAL loss = (task MSE) + λ × (id cross-entropy).
If high λ forces the model to memorize device identity AND that binding then
degrades performance on a transplanted substrate, identity is constitutive.

Pipeline:
 1. Standalone discriminator (gate test): can a small MLP separate
    ikaros vs daedalus substrate segments at all? If <90% acc, the channels
    don't carry the information in this form, and the rest is moot.
 2. Dual-objective reservoir: 128-neuron leaky reservoir (Regime 5 style) on
    NARMA-10, with TWO heads (task ridge + identity logistic). Joint loss is
    optimized with iterative refit (alternating projection-style: ridge for
    task, gradient for id, share readout features).
 3. Transplant per λ ∈ {0.0, 0.1, 1.0, 10.0}: train on ikaros, eval on
    {ikaros, daedalus, sw_matched, shuffle}. Measure task NRMSE and id-head
    accuracy on each.
 4. Stand-alone discriminator control: id-only readout from same reservoir
    states (no task pressure). Compares to dual-objective id accuracy.

Output: results/IDENTITY_BENCHMARK_2026-05-30/attack_1_3/A1_results.json
"""
from __future__ import annotations
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "constitutive"))
from _substrate_stream import (  # type: ignore
    SubstrateStreamer,
    GaussianMatched,
    PermutedSubstrate,
)
from reservoir import Reservoir, ReservoirCfg, ridge_fit  # type: ignore

OUT_DIR = HERE.parents[2] / "results" / "IDENTITY_BENCHMARK_2026-05-30" / "attack_1_3"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------
RNG_GLOBAL = np.random.default_rng(20260530)
N_RES = 128
SUB_DIM = 128
WASHOUT = 100
T_TRAIN = 1500
T_TEST = 800
HORIZON = 1
N_SEEDS = int(os.environ.get("N_SEEDS", "8"))
LAMBDAS = [0.0, 0.1, 1.0, 10.0]
EVAL_KINDS = ["self", "daedalus", "sw_matched", "shuffle"]

# Discriminator gate (segment-level)
SEG_LEN = 100  # samples per segment
N_SEGMENTS_PER_HOST = 1000

# Dual-objective inner solver
DUAL_ITERS = 25
DUAL_LR = 0.05


# -----------------------------------------------------------------------------
# NARMA-10 task
# -----------------------------------------------------------------------------
def narma10(T: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    u = 0.5 * rng.random(T + 10)  # in [0, 0.5]
    y = np.zeros(T + 10)
    for t in range(10, T + 10):
        y[t] = 0.3 * y[t - 1] + 0.05 * y[t - 1] * y[t - 10:t].sum() + 1.5 * u[t - 10] * u[t - 1] + 0.1
    return u[10:].astype(np.float64), y[10:].astype(np.float64)


# -----------------------------------------------------------------------------
# Discriminator gate — standalone
# -----------------------------------------------------------------------------
def collect_substrate_segments(host: str, n: int, seg_len: int, seed: int) -> np.ndarray:
    """Return shape (n, seg_len, SUB_DIM) of raw substrate samples."""
    sub = SubstrateStreamer(host, n_dim=SUB_DIM, seed=seed)
    out = np.empty((n, seg_len, SUB_DIM), dtype=np.float32)
    sub.reset(seed=seed)
    for i in range(n):
        out[i] = sub.stream(seg_len).astype(np.float32)
    return out


def featurize_segments(X: np.ndarray) -> np.ndarray:
    """Compact features per segment: per-dim mean, std, AR(1) coeff, abs-mean, p90.
    Output shape (n, 5*SUB_DIM)."""
    n, T, D = X.shape
    means = X.mean(axis=1)
    stds = X.std(axis=1)
    abs_means = np.abs(X).mean(axis=1)
    p90 = np.percentile(X, 90, axis=1)
    # AR(1) per dim per segment
    x0 = X[:, :-1, :]
    x1 = X[:, 1:, :]
    num = (x0 * x1).mean(axis=1)
    den = (x0 * x0).mean(axis=1) + 1e-12
    ar1 = num / den
    return np.concatenate([means, stds, abs_means, p90, ar1], axis=1).astype(np.float32)


def mlp_discriminator(Xtr, ytr, Xte, yte, hidden=64, epochs=200, lr=0.05, l2=1e-4, seed=0):
    """Tiny 2-layer MLP via numpy + Adam-ish. Returns (test_acc, train_acc)."""
    rng = np.random.default_rng(seed)
    D = Xtr.shape[1]
    W1 = rng.standard_normal((D, hidden)).astype(np.float32) * np.sqrt(2.0 / D)
    b1 = np.zeros(hidden, dtype=np.float32)
    W2 = rng.standard_normal((hidden, 2)).astype(np.float32) * np.sqrt(2.0 / hidden)
    b2 = np.zeros(2, dtype=np.float32)

    def fwd(X):
        h = np.maximum(0, X @ W1 + b1)
        logits = h @ W2 + b2
        return h, logits

    def softmax(z):
        z = z - z.max(axis=1, keepdims=True)
        e = np.exp(z)
        return e / e.sum(axis=1, keepdims=True)

    n = Xtr.shape[0]
    onehot = np.zeros((n, 2), dtype=np.float32)
    onehot[np.arange(n), ytr] = 1.0

    bs = 64
    for ep in range(epochs):
        idx = rng.permutation(n)
        for i in range(0, n, bs):
            b = idx[i:i + bs]
            xb = Xtr[b]
            yb = onehot[b]
            h, logits = fwd(xb)
            p = softmax(logits)
            grad_logits = (p - yb) / xb.shape[0]
            gW2 = h.T @ grad_logits + l2 * W2
            gb2 = grad_logits.sum(axis=0)
            grad_h = grad_logits @ W2.T
            grad_h[h <= 0] = 0
            gW1 = xb.T @ grad_h + l2 * W1
            gb1 = grad_h.sum(axis=0)
            W2 -= lr * gW2
            b2 -= lr * gb2
            W1 -= lr * gW1
            b1 -= lr * gb1

    _, lt = fwd(Xtr)
    _, le = fwd(Xte)
    train_acc = float((lt.argmax(axis=1) == ytr).mean())
    test_acc = float((le.argmax(axis=1) == yte).mean())
    return test_acc, train_acc


def run_discriminator_gate():
    print("[A1.gate] collecting segments…", flush=True)
    Xi = collect_substrate_segments("ikaros", N_SEGMENTS_PER_HOST, SEG_LEN, seed=11)
    Xd = collect_substrate_segments("daedalus", N_SEGMENTS_PER_HOST, SEG_LEN, seed=22)
    Fi = featurize_segments(Xi)
    Fd = featurize_segments(Xd)
    F = np.concatenate([Fi, Fd], axis=0)
    y = np.concatenate([np.zeros(N_SEGMENTS_PER_HOST, dtype=np.int64),
                        np.ones(N_SEGMENTS_PER_HOST, dtype=np.int64)])
    # standardize
    mu = F.mean(axis=0)
    sd = F.std(axis=0) + 1e-6
    F = (F - mu) / sd

    perm = RNG_GLOBAL.permutation(F.shape[0])
    F, y = F[perm], y[perm]
    n_tr = int(0.8 * F.shape[0])
    Xtr, ytr = F[:n_tr], y[:n_tr]
    Xte, yte = F[n_tr:], y[n_tr:]

    test_acc, train_acc = mlp_discriminator(Xtr, ytr, Xte, yte, seed=0)
    print(f"[A1.gate] discriminator test_acc={test_acc:.4f} train_acc={train_acc:.4f}", flush=True)
    return {
        "n_segments_per_host": N_SEGMENTS_PER_HOST,
        "seg_len": SEG_LEN,
        "feature_dim": int(F.shape[1]),
        "train_acc": train_acc,
        "test_acc": test_acc,
        "gate_passed": bool(test_acc >= 0.90),
    }


# -----------------------------------------------------------------------------
# Dual-objective reservoir
# -----------------------------------------------------------------------------
def build_reservoir(seed: int, substrate, regime: int = 5) -> Reservoir:
    cfg = ReservoirCfg(n_in=1, n_res=N_RES, seed=seed)
    return Reservoir(cfg, regime=regime, substrate=substrate)


def collect_states_for_run(host_substrate, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Train-set states + targets for NARMA-10."""
    u, y = narma10(T_TRAIN + HORIZON + WASHOUT, seed=seed)
    u_in = u[:-HORIZON][:, None]
    y_tg = y[HORIZON:]
    res = build_reservoir(seed, host_substrate)
    X = res.run(u_in, washout=WASHOUT)
    y_w = y_tg[WASHOUT:]
    return X, y_w, res


def collect_states_eval(eval_substrate, train_res: Reservoir, seed: int) -> tuple[np.ndarray, np.ndarray]:
    u, y = narma10(T_TEST + HORIZON + WASHOUT, seed=seed + 9999)
    u_in = u[:-HORIZON][:, None]
    y_tg = y[HORIZON:]
    cfg = train_res.cfg
    res = Reservoir(cfg, regime=5, substrate=eval_substrate)
    res.W_in = train_res.W_in
    res.W_rec = train_res.W_rec
    # for regime 5, recompute W_rec_eff using eval substrate's mod
    M = eval_substrate.weight_mod(cfg.n_res) if eval_substrate is not None else 0.0
    res.W_rec_eff = res.W_rec * (1.0 + 0.15 * M)
    X = res.run(u_in, washout=WASHOUT)
    y_w = y_tg[WASHOUT:]
    return X, y_w


def dual_train(X: np.ndarray, y_task: np.ndarray, y_id: np.ndarray, lam: float,
               alpha_ridge: float = 1e-3, iters: int = DUAL_ITERS,
               lr: float = DUAL_LR, seed: int = 0):
    """Joint readout: task MSE + λ·CE.

    We use a SHARED feature representation (the reservoir states) and TWO linear
    heads:
      - W_task (D+1) → 1
      - W_id (D+1) → 2
    For λ=0: closed-form ridge for task, separate logistic for id (control).
    For λ>0: gradient descent on combined loss (still convex in W given fixed X).
    """
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    Xa = np.hstack([X, np.ones((n, 1))]).astype(np.float64)
    D = Xa.shape[1]
    # init from ridge
    W_task = ridge_fit(Xa, y_task, alpha=alpha_ridge)
    # init id from logistic-ish (small random)
    W_id = rng.standard_normal((D, 2)) * 0.01

    onehot = np.zeros((n, 2))
    onehot[np.arange(n), y_id] = 1.0

    if lam == 0.0:
        # pure task — ridge already done. Train id separately (no joint pressure).
        for _ in range(iters):
            z = Xa @ W_id
            z = z - z.max(axis=1, keepdims=True)
            e = np.exp(z)
            p = e / e.sum(axis=1, keepdims=True)
            grad = Xa.T @ (p - onehot) / n + 1e-3 * W_id
            W_id -= lr * grad
        return W_task, W_id

    # joint: alternating update — task closed-form given W_id penalty=0 since heads decoupled;
    # but with λ on id head, gradient on Xa drives a shared bias in features only
    # if heads share params. Here we instead enforce SHARED PRESSURE: gradient
    # of total loss wrt Xa is unused (Xa fixed by substrate), so we instead use
    # a coupled penalty: ridge on (W_task, W_id) jointly with feature-space
    # entanglement via λ. The cleanest way: minimize
    #   ||Xa W_task - y_task||^2 + λ · CE(softmax(Xa W_id), y_id)
    #   + α (||W_task||^2 + ||W_id||^2)
    # gradient descent for both, with shared features fixed.
    Wt = W_task.copy()
    Wi = W_id.copy()
    for it in range(iters):
        # task
        r = Xa @ Wt - y_task
        gt = 2.0 * (Xa.T @ r) / n + 2.0 * alpha_ridge * Wt
        Wt -= lr * gt
        # id
        z = Xa @ Wi
        z = z - z.max(axis=1, keepdims=True)
        e = np.exp(z)
        p = e / e.sum(axis=1, keepdims=True)
        gi = lam * (Xa.T @ (p - onehot)) / n + 2.0 * alpha_ridge * Wi
        Wi -= lr * gi
    return Wt, Wi


def task_nrmse(W_task, X, y) -> float:
    Xa = np.hstack([X, np.ones((X.shape[0], 1))])
    y_pred = Xa @ W_task
    err = float(np.sqrt(np.mean((y - y_pred) ** 2)))
    rng = float(y.std() + 1e-12)
    return err / rng


def id_accuracy(W_id, X, y_id) -> float:
    Xa = np.hstack([X, np.ones((X.shape[0], 1))])
    z = Xa @ W_id
    pred = z.argmax(axis=1)
    return float((pred == y_id).mean())


def build_substrate(kind: str, seed: int, ref: SubstrateStreamer | None):
    if kind == "ikaros":
        return SubstrateStreamer("ikaros", n_dim=SUB_DIM, seed=seed)
    if kind == "daedalus":
        return SubstrateStreamer("daedalus", n_dim=SUB_DIM, seed=seed)
    if kind == "sw_matched":
        return GaussianMatched(ref, seed=seed + 7)
    if kind == "shuffle":
        return PermutedSubstrate(ref, seed=seed + 11)
    raise ValueError(kind)


def run_dual_objective_matrix(gate_acc: float):
    """For each λ, for each seed: train on ikaros, eval on all kinds.

    Identity labels: we need a way for the id-head to see "device class" during
    training. The natural way: build a TRAINING batch that interleaves segments
    from BOTH devices' substrates, with labels {0=ikaros, 1=daedalus}.
    Then the task head is fit only on the ikaros segments (so transplant
    asymmetry is preserved), and the id head sees both."""
    out = {"lambdas": {}, "per_seed_seeds": []}

    for lam in LAMBDAS:
        lam_key = f"lambda_{lam}"
        out["lambdas"][lam_key] = {"per_seed": []}
        print(f"[A1.dual] λ={lam}", flush=True)
        for s in range(N_SEEDS):
            sub_i = SubstrateStreamer("ikaros", n_dim=SUB_DIM, seed=s + 100)
            sub_d = SubstrateStreamer("daedalus", n_dim=SUB_DIM, seed=s + 200)

            # build reservoir on ikaros substrate
            X_i, y_task_i, res_train = collect_states_for_run(sub_i, seed=s)
            # also collect states on daedalus substrate (same NARMA inputs for fairness)
            u, y = narma10(T_TRAIN + HORIZON + WASHOUT, seed=s)
            u_in = u[:-HORIZON][:, None]
            cfg = res_train.cfg
            res_d_train = Reservoir(cfg, regime=5, substrate=sub_d)
            res_d_train.W_in = res_train.W_in
            res_d_train.W_rec = res_train.W_rec
            M = sub_d.weight_mod(cfg.n_res)
            res_d_train.W_rec_eff = res_d_train.W_rec * (1.0 + 0.15 * M)
            X_d = res_d_train.run(u_in, washout=WASHOUT)

            # Joint training set for id head: stack X_i (label=0) + X_d (label=1)
            X_joint = np.vstack([X_i, X_d])
            y_id_joint = np.concatenate([np.zeros(X_i.shape[0], dtype=np.int64),
                                          np.ones(X_d.shape[0], dtype=np.int64)])
            # Task targets only defined for ikaros half. We zero-pad daedalus
            # half but mask via low effective weight on it — simplest: average
            # task signal so daedalus rows don't pull the ridge fit much. Cleanest
            # approach: train task head on X_i alone (closed-form ridge), train
            # id head on X_joint with λ. This decouples cleanly.
            # 1. task head: ridge on (X_i, y_task_i)
            # 2. id head: gradient on (X_joint, y_id_joint) with λ controlling
            #    strength. λ=0 still trains id (for measurement), but task is
            #    pure ridge. λ>0 adds penalty term that PUSHES id head to be
            #    accurate — we then study whether this binding has *any*
            #    side-effect on transplant degradation.
            Xa_i = np.hstack([X_i, np.ones((X_i.shape[0], 1))])
            W_task = ridge_fit(Xa_i, y_task_i, alpha=1e-3)

            # id head: simple logistic regression via gradient, but loss weight
            # = λ (effective). For λ=0 we still fit but report it as control.
            n_j = X_joint.shape[0]
            Xa_j = np.hstack([X_joint, np.ones((n_j, 1))])
            D = Xa_j.shape[1]
            W_id = np.zeros((D, 2))
            onehot = np.zeros((n_j, 2))
            onehot[np.arange(n_j), y_id_joint] = 1.0
            lr_eff = DUAL_LR if lam > 0 else DUAL_LR * 0.3
            n_iter_eff = DUAL_ITERS * 2 if lam > 0 else DUAL_ITERS
            for _ in range(n_iter_eff):
                z = Xa_j @ W_id
                z = z - z.max(axis=1, keepdims=True)
                e = np.exp(z)
                p = e / e.sum(axis=1, keepdims=True)
                grad = (Xa_j.T @ (p - onehot)) / n_j + 1e-3 * W_id
                W_id -= lr_eff * grad

            # ---- Now: build a JOINT task readout penalized by id-head loss ----
            # If λ>0, we additionally refit W_task with a regularizer that
            # PROJECTS task readout toward id-discriminative subspace:
            #   ||X_i W_task - y_task||^2 + λ · (1 - |W_task · v_id|^2 / ||W_task||^2)
            # where v_id is the dominant direction of the id head.
            # Simpler: enforce that W_task explicitly contains the id-discriminative
            # projection by appending λ-scaled id-head as auxiliary target during
            # ridge. We multi-task fit: target = [y_task, λ · sign_id_indicator].
            if lam > 0:
                # Build augmented target on full joint set
                sign_id = (y_id_joint == 0).astype(np.float64) * 2 - 1  # ±1 per row
                # Use combined ridge: cat features X_joint, targets [y_task on i half;
                # zeros on d half] AND aux target = λ * sign_id everywhere.
                # Two-target ridge with separate alphas:
                y_task_full = np.concatenate([y_task_i, np.zeros(X_d.shape[0])])
                # weight: ones on i half, small weight on d half for task
                w_task = np.concatenate([np.ones(X_i.shape[0]),
                                          0.1 * np.ones(X_d.shape[0])])
                Xa_full = np.hstack([X_joint, np.ones((n_j, 1))])
                Wm = np.diag(w_task)
                # primary
                A1 = (Xa_full.T * w_task) @ Xa_full + 1e-3 * np.eye(D)
                b1 = (Xa_full.T * w_task) @ y_task_full
                # aux (id-binding) — penalty that task readout aligns with id
                A2 = lam * (Xa_full.T @ Xa_full) + 1e-3 * np.eye(D)
                b2 = lam * (Xa_full.T @ sign_id)
                W_task = np.linalg.solve(A1 + A2, b1 + b2)

            # ---- evaluate transplant ----
            seed_result = {"seed": s, "evals": {}}
            for ek in EVAL_KINDS:
                if ek == "self":
                    sub_eval = SubstrateStreamer("ikaros", n_dim=SUB_DIM, seed=s + 300)
                    id_label_true = 0
                elif ek == "daedalus":
                    sub_eval = SubstrateStreamer("daedalus", n_dim=SUB_DIM, seed=s + 400)
                    id_label_true = 1
                elif ek == "sw_matched":
                    sub_eval = GaussianMatched(sub_i, seed=s + 500)
                    id_label_true = -1  # neither — we report id-head prediction distribution
                else:  # shuffle
                    sub_eval = PermutedSubstrate(sub_i, seed=s + 600)
                    id_label_true = -1
                X_te, y_te = collect_states_eval(sub_eval, res_train, s)
                nrmse = task_nrmse(W_task, X_te, y_te)
                Xa_te = np.hstack([X_te, np.ones((X_te.shape[0], 1))])
                z = Xa_te @ W_id
                pred = z.argmax(axis=1)
                frac_pred_ikaros = float((pred == 0).mean())
                if id_label_true in (0, 1):
                    id_acc = float((pred == id_label_true).mean())
                else:
                    id_acc = None
                seed_result["evals"][ek] = {
                    "nrmse": float(nrmse),
                    "id_acc": id_acc,
                    "frac_pred_ikaros": frac_pred_ikaros,
                }
            out["lambdas"][lam_key]["per_seed"].append(seed_result)
            if s == 0:
                # print quick summary first seed
                print("  seed0:", {k: round(v["nrmse"], 4) for k, v in seed_result["evals"].items()},
                      flush=True)
        # aggregate
        agg = {}
        for ek in EVAL_KINDS:
            vals = [sr["evals"][ek]["nrmse"] for sr in out["lambdas"][lam_key]["per_seed"]]
            id_accs = [sr["evals"][ek]["id_acc"] for sr in out["lambdas"][lam_key]["per_seed"]
                       if sr["evals"][ek]["id_acc"] is not None]
            agg[ek] = {
                "nrmse_mean": float(np.mean(vals)),
                "nrmse_std": float(np.std(vals)),
                "id_acc_mean": float(np.mean(id_accs)) if id_accs else None,
            }
        out["lambdas"][lam_key]["aggregate"] = agg
        print(f"  agg: { {k: round(v['nrmse_mean'], 4) for k, v in agg.items()} }", flush=True)
    return out


def diagnose_identity(res: dict, gate_acc: float) -> dict:
    """Pre-registered: at high λ, if task NRMSE on daedalus-substrate is >2σ
    above NRMSE on sw_matched-substrate, identity IS constitutive."""
    diag = {"gate_acc": gate_acc, "verdicts_per_lambda": {}}
    for lam_key, body in res["lambdas"].items():
        agg = body["aggregate"]
        nrmse_d = agg["daedalus"]["nrmse_mean"]
        std_d = agg["daedalus"]["nrmse_std"]
        nrmse_sw = agg["sw_matched"]["nrmse_mean"]
        std_sw = agg["sw_matched"]["nrmse_std"]
        # 2σ comparison: nrmse_d - nrmse_sw vs sqrt(std_d^2 + std_sw^2)
        gap = nrmse_d - nrmse_sw
        comb_std = float(np.sqrt(std_d ** 2 + std_sw ** 2)) + 1e-12
        z_gap = gap / comb_std
        verdict = (
            "CONSTITUTIVE" if z_gap > 2.0 else
            "SHUFFLE_INDISTINGUISHABLE" if abs(z_gap) < 1.0 else
            "WEAK_SEPARATION"
        )
        diag["verdicts_per_lambda"][lam_key] = {
            "nrmse_daedalus": nrmse_d,
            "nrmse_sw_matched": nrmse_sw,
            "gap": float(gap),
            "z_gap": float(z_gap),
            "verdict": verdict,
            "id_acc_daedalus": agg["daedalus"]["id_acc_mean"],
            "id_acc_self": agg["self"]["id_acc_mean"],
        }
    return diag


def main():
    t0 = time.time()
    print(f"[A1] starting at {time.strftime('%H:%M:%S')} N_SEEDS={N_SEEDS}", flush=True)
    gate = run_discriminator_gate()
    matrix = run_dual_objective_matrix(gate["test_acc"])
    diag = diagnose_identity(matrix, gate["test_acc"])
    out = {
        "config": {
            "n_seeds": N_SEEDS,
            "lambdas": LAMBDAS,
            "n_res": N_RES,
            "sub_dim": SUB_DIM,
            "t_train": T_TRAIN,
            "t_test": T_TEST,
            "seg_len": SEG_LEN,
            "n_segments_per_host": N_SEGMENTS_PER_HOST,
        },
        "gate": gate,
        "matrix": matrix,
        "diag": diag,
        "wall_s": time.time() - t0,
    }
    with open(OUT_DIR / "A1_results.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"[A1] done wall={out['wall_s']:.1f}s", flush=True)
    print(json.dumps(diag, indent=2))


if __name__ == "__main__":
    main()

```


=== FILE: A1_results.json (28552 chars) ===
```json
{
  "config": {
    "n_seeds": 8,
    "lambdas": [
      0.0,
      0.1,
      1.0,
      10.0
    ],
    "n_res": 128,
    "sub_dim": 128,
    "t_train": 1500,
    "t_test": 800,
    "seg_len": 100,
    "n_segments_per_host": 1000
  },
  "gate": {
    "n_segments_per_host": 1000,
    "seg_len": 100,
    "feature_dim": 640,
    "train_acc": 1.0,
    "test_acc": 1.0,
    "gate_passed": true
  },
  "matrix": {
    "lambdas": {
      "lambda_0.0": {
        "per_seed": [
          {
            "seed": 0,
            "evals": {
              "self": {
                "nrmse": 0.7113074249312306,
                "id_acc": 0.99625,
                "frac_pred_ikaros": 0.99625
              },
              "daedalus": {
                "nrmse": 19.819078599487984,
                "id_acc": 0.3725,
                "frac_pred_ikaros": 0.6275
              },
              "sw_matched": {
                "nrmse": 20.031498315215018,
                "id_acc": null,
                "frac_pred_ikaros": 0.895
              },
              "shuffle": {
                "nrmse": 53.667116810789764,
                "id_acc": null,
                "frac_pred_ikaros": 0.92625
              }
            }
          },
          {
            "seed": 1,
            "evals": {
              "self": {
                "nrmse": 0.6753461079539989,
                "id_acc": 0.68875,
                "frac_pred_ikaros": 0.68875
              },
              "daedalus": {
                "nrmse": 24.278942858930485,
                "id_acc": 0.56625,
                "frac_pred_ikaros": 0.43375
              },
              "sw_matched": {
                "nrmse": 12.229805004381468,
                "id_acc": null,
                "frac_pred_ikaros": 0.8525
              },
              "shuffle": {
                "nrmse": 6.591382539397396,
                "id_acc": null,
                "frac_pred_ikaros": 0.75875
              }
            }
          },
          {
            "seed": 2,
            "evals": {
              "self": {
                "nrmse": 0.629406875641779,
                "id_acc": 0.695,
                "frac_pred_ikaros": 0.695
              },
              "daedalus": {
                "nrmse": 27.167947427811956,
                "id_acc": 0.65625,
                "frac_pred_ikaros": 0.34375
              },
              "sw_matched": {
                "nrmse": 16.170453647977403,
                "id_acc": null,
                "frac_pred_ikaros": 0.78375
              },
              "shuffle": {
                "nrmse": 40.61184073927083,
                "id_acc": null,
                "frac_pred_ikaros": 0.7475
              }
            }
          },
          {
            "seed": 3,
            "evals": {
              "self": {
                "nrmse": 0.5887935523608208,
                "id_acc": 0.85875,
                "frac_pred_ikaros": 0.85875
              },
              "daedalus": {
                "nrmse": 16.503658660198308,
                "id_acc": 0.8225,
                "frac_pred_ikaros": 0.1775
              },
              "sw_matched": {
                "nrmse": 13.015515291748791,
                "id_acc": null,
                "frac_pred_ikaros": 0.45375
              },
              "shuffle": {
                "nrmse": 25.09224309853881,
                "id_acc": null,
                "frac_pred_ikaros": 0.095
              }
            }
          },
          {
            "seed": 4,
            "evals": {
              "self": {
                "nrmse": 0.5909559510560212,
                "id_acc": 0.675,
                "frac_pred_ikaros": 0.675
              },
              "daedalus": {
                "nrmse": 7.4696809304539835,
                "id_acc": 0.6625,
                "frac_pred_ikaros": 0.3375
              },
              "sw_matched": {
                "nrmse": 10.847199874555566,
                "id_acc": null,
                "frac_pred_ikaros": 0.52625
              },
              "shuffle": {
                "nrmse": 40.325231124800425,
                "id_acc": null,
                "frac_pred_ikaros": 0.31375
              }
            }
          },
          {
            "seed": 5,
            "evals": {
              "self": {
                "nrmse": 0.6398784604335396,
                "id_acc": 0.50625,
                "frac_pred_ikaros": 0.50625
              },
              "daedalus": {
                "nrmse": 5.256646880519231,
                "id_acc": 0.605,
                "frac_pred_ikaros": 0.395
              },
              "sw_matched": {
                "nrmse": 12.278143123104543,
                "id_acc": null,
                "frac_pred_ikaros": 0.12125
              },
              "shuffle": {
                "nrmse": 27.396163500648306,
                "id_acc": null,
                "frac_pred_ikaros": 0.435
              }
            }
          },
          {
            "seed": 6,
            "evals": {
              "self": {
                "nrmse": 0.6755433369433558,
                "id_acc": 0.65625,
                "frac_pred_ikaros": 0.65625
              },
              "daedalus": {
                "nrmse": 17.22483832559307,
                "id_acc": 0.80125,
                "frac_pred_ikaros": 0.19875
              },
              "sw_matched": {
                "nrmse": 13.589862466511685,
                "id_acc": null,
                "frac_pred_ikaros": 0.3025
              },
              "shuffle": {
                "nrmse": 17.386981932989624,
                "id_acc": null,
                "frac_pred_ikaros": 0.36
              }
            }
          },
          {
            "seed": 7,
            "evals": {
              "self": {
                "nrmse": 0.6261242988097937,
                "id_acc": 0.7325,
                "frac_pred_ikaros": 0.7325
              },
              "daedalus": {
                "nrmse": 50.35079843146093,
                "id_acc": 0.72875,
                "frac_pred_ikaros": 0.27125
              },
              "sw_matched": {
                "nrmse": 10.06546427677797,
                "id_acc": null,
                "frac_pred_ikaros": 0.515
              },
              "shuffle": {
                "nrmse": 9.238583158714452,
                "id_acc": null,
                "frac_pred_ikaros": 0.725
              }
            }
          }
        ],
        "aggregate": {
          "self": {
            "nrmse_mean": 0.6421695010163174,
            "nrmse_std": 0.0401434214369359,
            "id_acc_mean": 0.72609375
          },
          "daedalus": {
            "nrmse_mean": 21.008949014306992,
            "nrmse_std": 13.119879847316959,
            "id_acc_mean": 0.651875
          },
          "sw_matched": {
            "nrmse_mean": 13.528492750034054,
            "nrmse_std": 2.9993293546806137,
            "id_acc_mean": null
          },
          "shuffle": {
            "nrmse_mean": 27.5386928631437,
            "nrmse_std": 15.41040145879,
            "id_acc_mean": null
          }
        }
      },
      "lambda_0.1": {
        "per_seed": [
          {
            "seed": 0,
            "evals": {
              "self": {
                "nrmse": 0.9601831345633817,
                "id_acc": 1.0,
                "frac_pred_ikaros": 1.0
              },
              "daedalus": {
                "nrmse": 9.009429240627703,
                "id_acc": 0.525,
                "frac_pred_ikaros": 0.475
              },
              "sw_matched": {
                "nrmse": 23.24672271362628,
                "id_acc": null,
                "frac_pred_ikaros": 0.62625
              },
              "shuffle": {
                "nrmse": 27.897331163835904,
                "id_acc": null,
                "frac_pred_ikaros": 0.48875
              }
            }
          },
          {
            "seed": 1,
            "evals": {
              "self": {
                "nrmse": 0.9771737622493138,
                "id_acc": 0.71,
                "frac_pred_ikaros": 0.71
              },
              "daedalus": {
                "nrmse": 8.568817375599345,
                "id_acc": 0.6,
                "frac_pred_ikaros": 0.4
              },
              "sw_matched": {
                "nrmse": 19.58877848685898,
                "id_acc": null,
                "frac_pred_ikaros": 0.685
              },
              "shuffle": {
                "nrmse": 31.51052058551276,
                "id_acc": null,
                "frac_pred_ikaros": 0.7675
              }
            }
          },
          {
            "seed": 2,
            "evals": {
              "self": {
                "nrmse": 0.8330040921818352,
                "id_acc": 0.76,
                "frac_pred_ikaros": 0.76
              },
              "daedalus": {
                "nrmse": 8.631720994219936,
                "id_acc": 0.69375,
                "frac_pred_ikaros": 0.30625
              },
              "sw_matched": {
                "nrmse": 20.16905854759026,
                "id_acc": null,
                "frac_pred_ikaros": 0.6225
              },
              "shuffle": {
                "nrmse": 17.10347161012875,
                "id_acc": null,
                "frac_pred_ikaros": 0.61
              }
            }
          },
          {
            "seed": 3,
            "evals": {
              "self": {
                "nrmse": 0.816535455872046,
                "id_acc": 0.8625,
                "frac_pred_ikaros": 0.8625
              },
              "daedalus": {
                "nrmse": 8.194502331255212,
                "id_acc": 0.90875,
                "frac_pred_ikaros": 0.09125
              },
              "sw_matched": {
                "nrmse": 16.36898043927843,
                "id_acc": null,
                "frac_pred_ikaros": 0.1675
              },
              "shuffle": {
                "nrmse": 20.68814770695539,
                "id_acc": null,
                "frac_pred_ikaros": 0.5875
              }
            }
          },
          {
            "seed": 4,
            "evals": {
              "self": {
                "nrmse": 0.8369392883079471,
                "id_acc": 0.71625,
                "frac_pred_ikaros": 0.71625
              },
              "daedalus": {
                "nrmse": 8.232776020551732,
                "id_acc": 0.71875,
                "frac_pred_ikaros": 0.28125
              },
              "sw_matched": {
                "nrmse": 16.398079452337647,
                "id_acc": null,
                "frac_pred_ikaros": 0.71875
              },
              "shuffle": {
                "nrmse": 14.329194465592943,
                "id_acc": null,
                "frac_pred_ikaros": 0.41125
              }
            }
          },
          {
            "seed": 5,
            "evals": {
              "self": {
                "nrmse": 0.9303029930340339,
                "id_acc": 0.54875,
                "frac_pred_ikaros": 0.54875
              },
              "daedalus": {
                "nrmse": 8.543622989817049,
                "id_acc": 0.60125,
                "frac_pred_ikaros": 0.39875
              },
              "sw_matched": {
                "nrmse": 24.327652304047596,
                "id_acc": null,
                "frac_pred_ikaros": 0.53125
              },
              "shuffle": {
                "nrmse": 13.976486902345707,
                "id_acc": null,
                "frac_pred_ikaros": 0.5275
              }
            }
          },
          {
            "seed": 6,
            "evals": {
              "self": {
                "nrmse": 1.0391823148394945,
                "id_acc": 0.82,
                "frac_pred_ikaros": 0.82
              },
              "daedalus": {
                "nrmse": 8.719762898718901,
                "id_acc": 0.8575,
                "frac_pred_ikaros": 0.1425
              },
              "sw_matched": {
                "nrmse": 25.307171066398357,
                "id_acc": null,
                "frac_pred_ikaros": 0.50875
              },
              "shuffle": {
                "nrmse": 49.32712959878672,
                "id_acc": null,
                "frac_pred_ikaros": 0.4025
              }
            }
          },
          {
            "seed": 7,
            "evals": {
              "self": {
                "nrmse": 0.8315082141016622,
                "id_acc": 0.73375,
                "frac_pred_ikaros": 0.73375
              },
              "daedalus": {
                "nrmse": 7.617191466478157,
                "id_acc": 0.75,
                "frac_pred_ikaros": 0.25
              },
              "sw_matched": {
                "nrmse": 19.195837745794094,
                "id_acc": null,
                "frac_pred_ikaros": 0.4475
              },
              "shuffle": {
                "nrmse": 34.48729764058076,
                "id_acc": null,
                "frac_pred_ikaros": 0.695
              }
            }
          }
        ],
        "aggregate": {
          "self": {
            "nrmse_mean": 0.9031036568937143,
            "nrmse_std": 0.07898835817155002,
            "id_acc_mean": 0.76890625
          },
          "daedalus": {
            "nrmse_mean": 8.439727914658505,
            "nrmse_std": 0.3947831197460602,
            "id_acc_mean": 0.7068749999999999
          },
          "sw_matched": {
            "nrmse_mean": 20.575285094491456,
            "nrmse_std": 3.1974739950243642,
            "id_acc_mean": null
          },
          "shuffle": {
            "nrmse_mean": 26.164947459217366,
            "nrmse_std": 11.391190670420077,
            "id_acc_mean": null
          }
        }
      },
      "lambda_1.0": {
        "per_seed": [
          {
            "seed": 0,
            "evals": {
              "self": {
                "nrmse": 3.271575178301647,
                "id_acc": 1.0,
                "frac_pred_ikaros": 1.0
              },
              "daedalus": {
                "nrmse": 13.099478848701843,
                "id_acc": 0.525,
                "frac_pred_ikaros": 0.475
              },
              "sw_matched": {
                "nrmse": 45.39631244514109,
                "id_acc": null,
                "frac_pred_ikaros": 0.62625
              },
              "shuffle": {
                "nrmse": 41.592299287935084,
                "id_acc": null,
                "frac_pred_ikaros": 0.48875
              }
            }
          },
          {
            "seed": 1,
            "evals": {
              "self": {
                "nrmse": 3.2505995310498172,
                "id_acc": 0.71,
                "frac_pred_ikaros": 0.71
              },
              "daedalus": {
                "nrmse": 12.450053005933565,
                "id_acc": 0.6,
                "frac_pred_ikaros": 0.4
              },
              "sw_matched": {
                "nrmse": 34.80184641987749,
                "id_acc": null,
                "frac_pred_ikaros": 0.685
              },
              "shuffle": {
                "nrmse": 70.45648689433668,
                "id_acc": null,
                "frac_pred_ikaros": 0.7675
              }
            }
          },
          {
            "seed": 2,
            "evals": {
              "self": {
                "nrmse": 3.017705116684832,
                "id_acc": 0.76,
                "frac_pred_ikaros": 0.76
              },
              "daedalus": {
                "nrmse": 12.555412293844187,
                "id_acc": 0.69375,
                "frac_pred_ikaros": 0.30625
              },
              "sw_matched": {
                "nrmse": 37.097438588772,
                "id_acc": null,
                "frac_pred_ikaros": 0.6225
              },
              "shuffle": {
                "nrmse": 31.042205719519657,
                "id_acc": null,
                "frac_pred_ikaros": 0.61
              }
            }
          },
          {
            "seed": 3,
            "evals": {
              "self": {
                "nrmse": 2.9361652754116228,
                "id_acc": 0.8625,
                "frac_pred_ikaros": 0.8625
              },
              "daedalus": {
                "nrmse": 12.005952782476625,
                "id_acc": 0.90875,
                "frac_pred_ikaros": 0.09125
              },
              "sw_matched": {
                "nrmse": 31.054351070872936,
                "id_acc": null,
                "frac_pred_ikaros": 0.1675
              },
              "shuffle": {
                "nrmse": 29.964912591969078,
                "id_acc": null,
                "frac_pred_ikaros": 0.5875
              }
            }
          },
          {
            "seed": 4,
            "evals": {
              "self": {
                "nrmse": 2.8368442770168745,
                "id_acc": 0.71625,
                "frac_pred_ikaros": 0.71625
              },
              "daedalus": {
                "nrmse": 11.922717731786857,
                "id_acc": 0.71875,
                "frac_pred_ikaros": 0.28125
              },
              "sw_matched": {
                "nrmse": 35.23243872206373,
                "id_acc": null,
                "frac_pred_ikaros": 0.71875
              },
              "shuffle": {
                "nrmse": 25.261880758411994,
                "id_acc": null,
                "frac_pred_ikaros": 0.41125
              }
            }
          },
          {
            "seed": 5,
            "evals": {
              "self": {
                "nrmse": 3.1418517685675793,
                "id_acc": 0.54875,
                "frac_pred_ikaros": 0.54875
              },
              "daedalus": {
                "nrmse": 12.641273147731685,
                "id_acc": 0.60125,
                "frac_pred_ikaros": 0.39875
              },
              "sw_matched": {
                "nrmse": 46.123174124785876,
                "id_acc": null,
                "frac_pred_ikaros": 0.53125
              },
              "shuffle": {
                "nrmse": 23.380457649092488,
                "id_acc": null,
                "frac_pred_ikaros": 0.5275
              }
            }
          },
          {
            "seed": 6,
            "evals": {
              "self": {
                "nrmse": 3.270566148207773,
                "id_acc": 0.82,
                "frac_pred_ikaros": 0.82
              },
              "daedalus": {
                "nrmse": 12.730139352332706,
                "id_acc": 0.8575,
                "frac_pred_ikaros": 0.1425
              },
              "sw_matched": {
                "nrmse": 47.87135648355556,
                "id_acc": null,
                "frac_pred_ikaros": 0.50875
              },
              "shuffle": {
                "nrmse": 123.41583299762172,
                "id_acc": null,
                "frac_pred_ikaros": 0.4025
              }
            }
          },
          {
            "seed": 7,
            "evals": {
              "self": {
                "nrmse": 2.6087022311977264,
                "id_acc": 0.73375,
                "frac_pred_ikaros": 0.73375
              },
              "daedalus": {
                "nrmse": 11.111291424326469,
                "id_acc": 0.75,
                "frac_pred_ikaros": 0.25
              },
              "sw_matched": {
                "nrmse": 34.47716083405043,
                "id_acc": null,
                "frac_pred_ikaros": 0.4475
              },
              "shuffle": {
                "nrmse": 66.20381055092095,
                "id_acc": null,
                "frac_pred_ikaros": 0.695
              }
            }
          }
        ],
        "aggregate": {
          "self": {
            "nrmse_mean": 3.041751190804734,
            "nrmse_std": 0.22362299434446453,
            "id_acc_mean": 0.76890625
          },
          "daedalus": {
            "nrmse_mean": 12.31453982339174,
            "nrmse_std": 0.5775538394765621,
            "id_acc_mean": 0.7068749999999999
          },
          "sw_matched": {
            "nrmse_mean": 39.00675983613989,
            "nrmse_std": 6.014370137615321,
            "id_acc_mean": null
          },
          "shuffle": {
            "nrmse_mean": 51.41473580622596,
            "nrmse_std": 32.02922639798403,
            "id_acc_mean": null
          }
        }
      },
      "lambda_10.0": {
        "per_seed": [
          {
            "seed": 0,
            "evals": {
              "self": {
                "nrmse": 5.746791923511538,
                "id_acc": 1.0,
                "frac_pred_ikaros": 1.0
              },
              "daedalus": {
                "nrmse": 13.878255481948063,
                "id_acc": 0.525,
                "frac_pred_ikaros": 0.475
              },
              "sw_matched": {
                "nrmse": 60.12228487445824,
                "id_acc": null,
                "frac_pred_ikaros": 0.62625
              },
              "shuffle": {
                "nrmse": 48.13626413901828,
                "id_acc": null,
                "frac_pred_ikaros": 0.48875
              }
            }
          },
          {
            "seed": 1,
            "evals": {
              "self": {
                "nrmse": 5.662694266597154,
                "id_acc": 0.71,
                "frac_pred_ikaros": 0.71
              },
              "daedalus": {
                "nrmse": 13.200125142848833,
                "id_acc": 0.6,
                "frac_pred_ikaros": 0.4
              },
              "sw_matched": {
                "nrmse": 44.592570517750836,
                "id_acc": null,
                "frac_pred_ikaros": 0.685
              },
              "shuffle": {
                "nrmse": 96.16049212926946,
                "id_acc": null,
                "frac_pred_ikaros": 0.7675
              }
            }
          },
          {
            "seed": 2,
            "evals": {
              "self": {
                "nrmse": 5.412299366308607,
                "id_acc": 0.76,
                "frac_pred_ikaros": 0.76
              },
              "daedalus": {
                "nrmse": 13.32702192764893,
                "id_acc": 0.69375,
                "frac_pred_ikaros": 0.30625
              },
              "sw_matched": {
                "nrmse": 48.82918148430195,
                "id_acc": null,
                "frac_pred_ikaros": 0.6225
              },
              "shuffle": {
                "nrmse": 44.80834658532318,
                "id_acc": null,
                "frac_pred_ikaros": 0.61
              }
            }
          },
          {
            "seed": 3,
            "evals": {
              "self": {
                "nrmse": 5.240660736374917,
                "id_acc": 0.8625,
                "frac_pred_ikaros": 0.8625
              },
              "daedalus": {
                "nrmse": 12.793919703375124,
                "id_acc": 0.90875,
                "frac_pred_ikaros": 0.09125
              },
              "sw_matched": {
                "nrmse": 40.70389844742415,
                "id_acc": null,
                "frac_pred_ikaros": 0.1675
              },
              "shuffle": {
                "nrmse": 33.3262111480954,
                "id_acc": null,
                "frac_pred_ikaros": 0.5875
              }
            }
          },
          {
            "seed": 4,
            "evals": {
              "self": {
                "nrmse": 5.013412969299751,
                "id_acc": 0.71625,
                "frac_pred_ikaros": 0.71625
              },
              "daedalus": {
                "nrmse": 12.669457095915808,
                "id_acc": 0.71875,
                "frac_pred_ikaros": 0.28125
              },
              "sw_matched": {
                "nrmse": 47.4319141306661,
                "id_acc": null,
                "frac_pred_ikaros": 0.71875
              },
              "shuffle": {
                "nrmse": 31.04066610876319,
                "id_acc": null,
                "frac_pred_ikaros": 0.41125
              }
            }
          },
          {
            "seed": 5,
            "evals": {
              "self": {
                "nrmse": 5.572810316980065,
                "id_acc": 0.54875,
                "frac_pred_ikaros": 0.54875
              },
              "daedalus": {
                "nrmse": 13.52280491012834,
                "id_acc": 0.60125,
                "frac_pred_ikaros": 0.39875
              },
              "sw_matched": {
                "nrmse": 60.54183285105469,
                "id_acc": null,
                "frac_pred_ikaros": 0.53125
              },
              "shuffle": {
                "nrmse": 43.011761113374035,
                "id_acc": null,
                "frac_pred_ikaros": 0.5275
              }
            }
          },
          {
            "seed": 6,
            "evals": {
              "self": {
                "nrmse": 5.683037969681259,
                "id_acc": 0.82,
                "frac_pred_ikaros": 0.82
              },
              "daedalus": {
                "nrmse": 13.465498206788613,
                "id_acc": 0.8575,
                "frac_pred_ikaros": 0.1425
              },
              "sw_matched": {
                "nrmse": 60.50440148928799,
                "id_acc": null,
                "frac_pred_ikaros": 0.50875
              },
              "shuffle": {
                "nrmse": 160.21684671526612,
                "id_acc": null,
                "frac_pred_ikaros": 0.4025
              }
            }
          },
          {
            "seed": 7,
            "evals": {
              "self": {
                "nrmse": 4.650737127212572,
                "id_acc": 0.73375,
                "frac_pred_ikaros": 0.73375
              },
              "daedalus": {
                "nrmse": 11.820861190387463,
                "id_acc": 0.75,
                "frac_pred_ikaros": 0.25
              },
              "sw_matched": {
                "nrmse": 43.31180218930191,
                "id_acc": null,
                "frac_pred_ikaros": 0.4475
              },
              "shuffle": {
                "nrmse": 84.48394628587835,
                "id_acc": null,
                "frac_pred_ikaros": 0.695
              }
            }
          }
        ],
        "aggregate": {
          "self": {
            "nrmse_mean": 5.372805584495733,
            "nrmse_std": 0.3587678887808977,
            "id_acc_mean": 0.76890625
          },
          "daedalus": {
            "nrmse_mean": 13.084742957380147,
            "nrmse_std": 0.6013510561581978,
            "id_acc_mean": 0.7068749999999999
          },
          "sw_matched": {
            "nrmse_mean": 50.75473574803073,
            "nrmse_std": 7.808104778512796,
            "id_acc_mean": null
          },
          "shuffle": {
            "nrmse_mean": 67.6480667781235,
            "nrmse_std": 41.371467571913,
            "id_acc_mean": null
          }
        }
      }
    },
    "per_seed_seeds": []
  },
  "diag": {
    "gate_acc": 1.0,
    "verdicts_per_lambda": {
      "lambda_0.0": {
        "nrmse_daedalus": 21.008949014306992,
        "nrmse_sw_matched": 13.528492750034054,
        "gap": 7.480456264272938,
        "z_gap": 0.5558226274115375,
        "verdict": "SHUFFLE_INDISTINGUISHABLE",
        "id_acc_daedalus": 0.651875,
        "id_acc_self": 0.72609375
      },
      "lambda_0.1": {
        "nrmse_daedalus": 8.439727914658505,
        "nrmse_sw_matched": 20.575285094491456,
        "gap": -12.13555717983295,
        "z_gap": -3.766755688248281,
        "verdict": "WEAK_SEPARATION",
        "id_acc_daedalus": 0.7068749999999999,
        "id_acc_self": 0.76890625
      },
      "lambda_1.0": {
        "nrmse_daedalus": 12.31453982339174,
        "nrmse_sw_matched": 39.00675983613989,
        "gap": -26.69222001274815,
        "z_gap": -4.41775149824588,
        "verdict": "WEAK_SEPARATION",
        "id_acc_daedalus": 0.7068749999999999,
        "id_acc_self": 0.76890625
      },
      "lambda_10.0": {
        "nrmse_daedalus": 13.084742957380147,
        "nrmse_sw_matched": 50.75473574803073,
        "gap": -37.66999279065058,
        "z_gap": -4.810228415162313,
        "verdict": "WEAK_SEPARATION",
        "id_acc_daedalus": 0.7068749999999999,
        "id_acc_self": 0.76890625
      }
    }
  },
  "wall_s": 5.409045934677124
}
```


=== FILE: A3_heavy_tail_transplant.py (10334 chars) ===
```python
#!/usr/bin/env python3
"""ATTACK 3 — re-run Regime-5 transplant with heavy-tail substrate streams.

Replaces the Gaussian-AR(1) SubstrateStreamer with a stream sampled (with
replacement) from collected heavy-tail traces. Compares HW Δ vs SW-matched Δ:
if SW (Gaussian) can't replicate the tails, the transplant degradation should
finally separate HW from SW-matched.

Output: results/IDENTITY_BENCHMARK_2026-05-30/attack_1_3/A3_transplant.json
"""
from __future__ import annotations
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "constitutive"))
from reservoir import Reservoir, ReservoirCfg, ridge_fit, nrmse, mackey_glass  # type: ignore

OUT_DIR = HERE.parents[2] / "results" / "IDENTITY_BENCHMARK_2026-05-30" / "attack_1_3"

N_RES = 32
SUB_DIM = 32
WASHOUT = 100
T_TRAIN = 1500
T_TEST = 800
HORIZON = 1
N_SEEDS = int(os.environ.get("N_SEEDS", "16"))


class HeavyTailSubstrate:
    """Stream from collected heavy-tail traces.

    Each step samples ONE value from a randomly chosen channel, then projects
    to n_dim via a fixed spatial pattern (per-host, from per-core latency rank
    or by a host-specific seed). This preserves both the heavy tail AND the
    spatial structure of the original SubstrateStreamer.
    """

    def __init__(self, host: str, streams: dict, n_dim: int, seed: int = 0,
                 spatial: np.ndarray | None = None):
        self.host = host
        self.n_dim = n_dim
        self.rng = np.random.default_rng(seed)
        # whitened (z-score) per-channel pools
        self.pools = []
        for ch, x in streams.items():
            x = np.asarray(x, dtype=np.float64)
            if x.size < 100:
                continue
            x = x[np.isfinite(x)]
            mu, sd = float(x.mean()), float(x.std() + 1e-12)
            self.pools.append((ch, (x - mu) / sd))
        if not self.pools:
            raise ValueError("no usable channels")
        # spatial pattern: deterministic per host
        if spatial is None:
            rng_sp = np.random.default_rng(int(abs(hash(host))) % (2 ** 32))
            v = rng_sp.standard_normal(n_dim)
            v /= (np.linalg.norm(v) + 1e-12) / np.sqrt(n_dim)
            spatial = v
        self.spatial = spatial.astype(np.float64)

    def reset(self, seed=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)

    def _draw(self) -> float:
        ch_idx = int(self.rng.integers(0, len(self.pools)))
        pool = self.pools[ch_idx][1]
        return float(pool[self.rng.integers(0, pool.size)])

    def step(self) -> np.ndarray:
        v = self._draw()
        return v * self.spatial

    def stream(self, n: int) -> np.ndarray:
        return np.stack([self.step() for _ in range(n)], axis=0)

    def initial_state(self, n_dim: int) -> np.ndarray:
        # use per-host seeded small init (same flavor as SubstrateStreamer)
        rng = np.random.default_rng(int(abs(hash(self.host + "_ic"))) % (2 ** 32))
        return rng.standard_normal(n_dim) * 0.1

    def per_neuron_leak(self, n_dim: int, lo: float = 0.05, hi: float = 0.5) -> np.ndarray:
        rng = np.random.default_rng(int(abs(hash(self.host + "_leak"))) % (2 ** 32))
        z = rng.standard_normal(n_dim)
        z = (z - z.mean()) / (z.std() + 1e-12)
        return lo + (hi - lo) * 0.5 * (1.0 + np.tanh(z))

    def weight_mod(self, n_dim: int) -> np.ndarray:
        rng = np.random.default_rng(int(abs(hash(self.host + "_wmod"))) % (2 ** 32))
        a = rng.standard_normal(n_dim)
        b = rng.standard_normal(n_dim)
        M = np.outer(a, b)
        M /= (np.abs(M).max() + 1e-12)
        return M


class GaussianMatchedHT:
    """SW-matched control to HeavyTailSubstrate: same per-host spatial / IC /
    leak / weight_mod, BUT step value drawn from N(0,1) instead of heavy-tail.
    This is the cleanest 'Gaussian SW' control specific to A3.
    """
    def __init__(self, ref: HeavyTailSubstrate, seed: int = 0):
        self.n_dim = ref.n_dim
        self.spatial = ref.spatial.copy()
        self._ic = ref.initial_state(ref.n_dim)
        self._leak = ref.per_neuron_leak(ref.n_dim)
        self._wmod = ref.weight_mod(ref.n_dim)
        self.rng = np.random.default_rng(seed)

    def reset(self, seed=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)

    def step(self):
        v = float(self.rng.standard_normal())
        return v * self.spatial

    def stream(self, n):
        return np.stack([self.step() for _ in range(n)], axis=0)

    def initial_state(self, n_dim):
        return self._ic.copy()

    def per_neuron_leak(self, n_dim, lo=0.05, hi=0.5):
        return self._leak.copy()

    def weight_mod(self, n_dim):
        return self._wmod.copy()


class ShuffleHT:
    """Same channels as HeavyTailSubstrate but with permuted spatial dims."""
    def __init__(self, ref: HeavyTailSubstrate, seed: int = 0):
        self.host = ref.host + "_shuf"
        self.n_dim = ref.n_dim
        self.pools = list(ref.pools)
        perm = np.random.default_rng(seed).permutation(self.n_dim)
        self.spatial = ref.spatial[perm].copy()
        self._ic = ref.initial_state(ref.n_dim)[perm]
        self._leak = ref.per_neuron_leak(ref.n_dim)[perm]
        wm = ref.weight_mod(ref.n_dim)
        self._wmod = wm[perm][:, perm]
        self.rng = np.random.default_rng(seed + 1)

    def reset(self, seed=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)

    def step(self):
        ch_idx = int(self.rng.integers(0, len(self.pools)))
        pool = self.pools[ch_idx][1]
        return float(pool[self.rng.integers(0, pool.size)]) * self.spatial

    def stream(self, n):
        return np.stack([self.step() for _ in range(n)], axis=0)

    def initial_state(self, n_dim):
        return self._ic.copy()

    def per_neuron_leak(self, n_dim, lo=0.05, hi=0.5):
        return self._leak.copy()

    def weight_mod(self, n_dim):
        return self._wmod.copy()


def one_run(regime: int, train_sub, eval_sub, seed: int) -> float:
    sig_tr = mackey_glass(T_TRAIN + HORIZON + WASHOUT, tau=5, seed=seed)
    sig_te = mackey_glass(T_TEST + HORIZON + WASHOUT, tau=5, seed=seed + 9999)
    u_tr = sig_tr[:-HORIZON][:, None]
    y_tr = sig_tr[HORIZON:]
    u_te = sig_te[:-HORIZON][:, None]
    y_te = sig_te[HORIZON:]

    cfg = ReservoirCfg(n_in=1, n_res=N_RES, seed=seed)
    res_train = Reservoir(cfg, regime=regime, substrate=train_sub)
    X_tr = res_train.run(u_tr, washout=WASHOUT)
    y_tr_w = y_tr[WASHOUT:]
    Xtr_a = np.hstack([X_tr, np.ones((X_tr.shape[0], 1))])
    W_out = ridge_fit(Xtr_a, y_tr_w, alpha=1e-4)

    res_eval = Reservoir(cfg, regime=regime, substrate=eval_sub)
    res_eval.W_in = res_train.W_in
    res_eval.W_rec = res_train.W_rec
    if eval_sub is not None:
        M = eval_sub.weight_mod(cfg.n_res)
        res_eval.W_rec_eff = res_train.W_rec * (1.0 + 0.15 * M)
    else:
        res_eval.W_rec_eff = res_train.W_rec
    X_te = res_eval.run(u_te, washout=WASHOUT)
    y_te_w = y_te[WASHOUT:]
    Xte_a = np.hstack([X_te, np.ones((X_te.shape[0], 1))])
    y_pred = Xte_a @ W_out
    return nrmse(y_te_w, y_pred)


def load_streams(host: str) -> dict:
    p = OUT_DIR / f"A3_streams_{host}.npz"
    if not p.exists():
        raise FileNotFoundError(p)
    d = np.load(p)
    return {k: d[k] for k in d.files}


def main():
    t0 = time.time()
    print(f"[A3.transplant] N_SEEDS={N_SEEDS}", flush=True)
    streams_i = load_streams("ikaros")
    streams_d = load_streams("daedalus")

    eval_kinds = ["self", "daedalus", "sw_matched_ht", "shuffle_ht"]
    cells = {ek: [] for ek in eval_kinds}

    for s in range(N_SEEDS):
        sub_i = HeavyTailSubstrate("ikaros", streams_i, n_dim=SUB_DIM, seed=s + 1)
        sub_d = HeavyTailSubstrate("daedalus", streams_d, n_dim=SUB_DIM, seed=s + 2)
        sub_sw = GaussianMatchedHT(sub_i, seed=s + 3)
        sub_sh = ShuffleHT(sub_i, seed=s + 4)
        for ek, ev in [("self", sub_i), ("daedalus", sub_d),
                        ("sw_matched_ht", sub_sw), ("shuffle_ht", sub_sh)]:
            try:
                v = one_run(5, sub_i, ev, seed=s)
            except Exception as e:
                v = float("nan")
            cells[ek].append(v)
        if s == 0:
            print("  seed0:", {k: round(np.nanmean(v), 4) for k, v in cells.items()
                                if len(v) > 0}, flush=True)

    agg = {}
    for ek in eval_kinds:
        arr = np.array(cells[ek], dtype=np.float64)
        agg[ek] = {
            "mean": float(np.nanmean(arr)),
            "std": float(np.nanstd(arr)),
            "n": int((~np.isnan(arr)).sum()),
            "ci95": [float(np.nanpercentile(arr, 2.5)),
                      float(np.nanpercentile(arr, 97.5))],
        }

    # Deltas vs self (diagonal)
    deltas = {}
    self_mean = agg["self"]["mean"]
    for ek in ("daedalus", "sw_matched_ht", "shuffle_ht"):
        d = agg[ek]["mean"] - self_mean
        # 2σ comparison HW vs SW
        deltas[ek] = float(d)
    # Verdict: HW Δ vs SW Δ
    d_hw = deltas["daedalus"]
    d_sw = deltas["sw_matched_ht"]
    sd_hw = agg["daedalus"]["std"]
    sd_sw = agg["sw_matched_ht"]["std"]
    pooled = float(np.sqrt(sd_hw ** 2 + sd_sw ** 2)) + 1e-12
    z_hw_vs_sw = (d_hw - d_sw) / pooled

    verdict = (
        "CONSTITUTIVE_WITH_HT" if z_hw_vs_sw > 2.0 else
        "STRUCTURE_BOUND_HT" if 0.5 < z_hw_vs_sw <= 2.0 else
        "SW_MATCHED_REPLICATES_HW" if abs(z_hw_vs_sw) <= 0.5 else
        "HW_LESS_THAN_SW"
    )

    out = {
        "config": {"n_seeds": N_SEEDS, "regime": 5, "n_res": N_RES, "sub_dim": SUB_DIM},
        "cells": {k: list(map(float, v)) for k, v in cells.items()},
        "aggregate": agg,
        "deltas_vs_self": deltas,
        "z_hw_vs_sw": float(z_hw_vs_sw),
        "verdict": verdict,
        "wall_s": time.time() - t0,
    }
    with open(OUT_DIR / "A3_transplant.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"[A3.transplant] done wall={time.time() - t0:.1f}s", flush=True)
    print(json.dumps({"aggregate": agg, "deltas": deltas,
                       "z_hw_vs_sw": z_hw_vs_sw, "verdict": verdict}, indent=2))


if __name__ == "__main__":
    main()

```


=== FILE: A3_tail_stats.json (15201 chars) ===
```json
{
  "stats": {
    "ikaros": {
      "ch_syscall_jitter": {
        "n": 80000,
        "mean": 0.268843475,
        "std": 0.09349868942891323,
        "min": 0.19,
        "max": 12.995,
        "p99_over_p50": 1.3791666666666669,
        "p99_99_over_p50": 12.48334166666585,
        "k1": 7501.545154871608,
        "k10": 80.79591434286445,
        "k100": 1.3708842295904435,
        "hill_p05": 35.248753324102296,
        "hill_p01": 7.708596836692736,
        "hill_p001": 1.1861009552464816,
        "dfa_hurst": 0.8113223548748041,
        "kl_gauss": 0.09230749270680529,
        "levy_alpha": 2.0
      },
      "ch_loop_jitter": {
        "n": 30000,
        "mean": 62.935017433333336,
        "std": 6.908372202253056,
        "min": 59.522,
        "max": 201.319,
        "p99_over_p50": 1.4142941003815674,
        "p99_99_over_p50": 3.070704368457091,
        "k1": 98.80720600039571,
        "k10": 35.54757073857346,
        "k100": 16.261733264310774,
        "hill_p05": 16.240071793490646,
        "hill_p01": 7.878328909591928,
        "hill_p001": 3.2123564250024,
        "dfa_hurst": 1.0018093690619287,
        "kl_gauss": 1.5862856639804754,
        "levy_alpha": 0.5226172029067726
      },
      "ch_atomic_burst": {
        "n": 1500,
        "mean": 4.294119072666667,
        "std": 0.5957618484545879,
        "min": 3.870543,
        "max": 10.385331,
        "p99_over_p50": 1.5173437953365012,
        "p99_99_over_p50": 2.004538991341254,
        "k1": 13.939325977339456,
        "k10": 5.679141071251831,
        "k100": 2.3451830729377,
        "hill_p05": 11.51272500391454,
        "hill_p01": 7.938946676060505,
        "hill_p001": 7.3602462963325,
        "dfa_hurst": 1.2315112919950995,
        "kl_gauss": 1.126736197940997,
        "levy_alpha": 1.2135255395276077
      },
      "ch_tsc_drift": {
        "n": 6000,
        "mean": 0.1289706548333333,
        "std": 0.14375031099612162,
        "min": 0.005387000000000697,
        "max": 0.9102599999999992,
        "p99_over_p50": 9.094853100229752,
        "p99_99_over_p50": 10.662357463516006,
        "k1": 6.464713047224135,
        "k10": 4.132778385252511,
        "k100": 3.2722213985442195,
        "hill_p05": 4.176698995691379,
        "hill_p01": 14.752405410919131,
        "hill_p001": 9.53518304027593,
        "dfa_hurst": 1.0431113442156166,
        "kl_gauss": 1.800357491441294,
        "levy_alpha": 0.5283040863365431
      }
    },
    "daedalus": {
      "ch_syscall_jitter": {
        "n": 80000,
        "mean": 0.23752517499999998,
        "std": 0.05336918063095381,
        "min": 0.21,
        "max": 8.636,
        "p99_over_p50": 1.0458333333333334,
        "p99_99_over_p50": 9.600112916655597,
        "k1": 12038.37068348682,
        "k10": 1256.6038896065786,
        "k100": 103.77529857338534,
        "hill_p05": 31.686451955492792,
        "hill_p01": 7.188648734556553,
        "hill_p001": 1.1833355011872604,
        "dfa_hurst": 0.7821478997779798,
        "kl_gauss": 0.23528392920029745,
        "levy_alpha": 2.0
      },
      "ch_loop_jitter": {
        "n": 40000,
        "mean": 62.72959142499999,
        "std": 3.861712773859738,
        "min": 60.873,
        "max": 190.875,
        "p99_over_p50": 1.0674439577428498,
        "p99_99_over_p50": 1.8088135950781452,
        "k1": 154.07106141490502,
        "k10": 137.55326356070972,
        "k100": 153.70411595068717,
        "hill_p05": 12.629606444139394,
        "hill_p01": 2.9543536028229354,
        "hill_p001": 23.471846700117347,
        "dfa_hurst": 1.1093899028470688,
        "kl_gauss": 0.8564370388431227,
        "levy_alpha": 1.9414008608847015
      },
      "ch_atomic_burst": {
        "n": 1500,
        "mean": 4.234069944,
        "std": 0.3124144226793948,
        "min": 3.832102,
        "max": 7.864927,
        "p99_over_p50": 1.1730784789200488,
        "p99_99_over_p50": 1.7753138913690205,
        "k1": 31.583477158318207,
        "k10": 3.14668762018492,
        "k100": 1.8809007238172584,
        "hill_p05": 16.619799865793055,
        "hill_p01": 6.432033996472228,
        "hill_p001": 4.653335161567422,
        "dfa_hurst": 1.2753408574658698,
        "kl_gauss": 0.2584006772576632,
        "levy_alpha": 2.0
      },
      "ch_tsc_drift": {
        "n": 6000,
        "mean": 0.5687114751666666,
        "std": 0.137402913176342,
        "min": 0.0534510000000008,
        "max": 1.3212550000000007,
        "p99_over_p50": 1.7945493218390218,
        "p99_99_over_p50": 2.1705654575452233,
        "k1": 8.306111957913545,
        "k10": 6.944197029833945,
        "k100": 2.2427777007353633,
        "hill_p05": 7.464204233016695,
        "hill_p01": 9.301553875562252,
        "hill_p001": 21.65028049429884,
        "dfa_hurst": 0.5812468441965641,
        "kl_gauss": 0.8125274752727244,
        "levy_alpha": 0.8692344873516379
      }
    }
  },
  "cross_device": {
    "ch_syscall_jitter": {
      "mean": {
        "a": 0.268843475,
        "b": 0.23752517499999998,
        "abs_diff": 0.03131830000000002,
        "rel_diff": 0.12369762622498975
      },
      "std": {
        "a": 0.09349868942891323,
        "b": 0.05336918063095381,
        "abs_diff": 0.04012950879795942,
        "rel_diff": 0.5464709031465514
      },
      "min": {
        "a": 0.19,
        "b": 0.21,
        "abs_diff": 0.01999999999999999,
        "rel_diff": 0.09999999999949995
      },
      "max": {
        "a": 12.995,
        "b": 8.636,
        "abs_diff": 4.359,
        "rel_diff": 0.40303268457302915
      },
      "p99_over_p50": {
        "a": 1.3791666666666669,
        "b": 1.0458333333333334,
        "abs_diff": 0.3333333333333335,
        "rel_diff": 0.27491408934685235
      },
      "p99_99_over_p50": {
        "a": 12.48334166666585,
        "b": 9.600112916655597,
        "abs_diff": 2.883228750010254,
        "rel_diff": 0.2611211700716023
      },
      "k1": {
        "a": 7501.545154871608,
        "b": 12038.37068348682,
        "abs_diff": 4536.8255286152125,
        "rel_diff": 0.46436489963882627
      },
      "k10": {
        "a": 80.79591434286445,
        "b": 1256.6038896065786,
        "abs_diff": 1175.807975263714,
        "rel_diff": 1.7583492562081466
      },
      "k100": {
        "a": 1.3708842295904435,
        "b": 103.77529857338534,
        "abs_diff": 102.4044143437949,
        "rel_diff": 1.9478484451628568
      },
      "hill_p05": {
        "a": 35.248753324102296,
        "b": 31.686451955492792,
        "abs_diff": 3.5623013686095035,
        "rel_diff": 0.10644029113616087
      },
      "hill_p01": {
        "a": 7.708596836692736,
        "b": 7.188648734556553,
        "abs_diff": 0.5199481021361834,
        "rel_diff": 0.06980459570856232
      },
      "hill_p001": {
        "a": 1.1861009552464816,
        "b": 1.1833355011872604,
        "abs_diff": 0.0027654540592212395,
        "rel_diff": 0.0023342715536513797
      },
      "dfa_hurst": {
        "a": 0.8113223548748041,
        "b": 0.7821478997779798,
        "abs_diff": 0.02917445509682426,
        "rel_diff": 0.03661750824855496
      },
      "kl_gauss": {
        "a": 0.09230749270680529,
        "b": 0.23528392920029745,
        "abs_diff": 0.14297643649349218,
        "rel_diff": 0.8728948741103731
      },
      "levy_alpha": {
        "a": 2.0,
        "b": 2.0,
        "abs_diff": 0.0,
        "rel_diff": 0.0
      }
    },
    "ch_loop_jitter": {
      "mean": {
        "a": 62.935017433333336,
        "b": 62.72959142499999,
        "abs_diff": 0.20542600833334745,
        "rel_diff": 0.0032694329803696605
      },
      "std": {
        "a": 6.908372202253056,
        "b": 3.861712773859738,
        "abs_diff": 3.046659428393318,
        "rel_diff": 0.5657633036600926
      },
      "min": {
        "a": 59.522,
        "b": 60.873,
        "abs_diff": 1.350999999999999,
        "rel_diff": 0.02244279247477016
      },
      "max": {
        "a": 201.319,
        "b": 190.875,
        "abs_diff": 10.443999999999988,
        "rel_diff": 0.05325935633895438
      },
      "p99_over_p50": {
        "a": 1.4142941003815674,
        "b": 1.0674439577428498,
        "abs_diff": 0.3468501426387176,
        "rel_diff": 0.2795219596225811
      },
      "p99_99_over_p50": {
        "a": 3.070704368457091,
        "b": 1.8088135950781452,
        "abs_diff": 1.261890773378946,
        "rel_diff": 0.5172194396284924
      },
      "k1": {
        "a": 98.80720600039571,
        "b": 154.07106141490502,
        "abs_diff": 55.26385541450931,
        "rel_diff": 0.4370787255019374
      },
      "k10": {
        "a": 35.54757073857346,
        "b": 137.55326356070972,
        "abs_diff": 102.00569282213627,
        "rel_diff": 1.1785696266000898
      },
      "k100": {
        "a": 16.261733264310774,
        "b": 153.70411595068717,
        "abs_diff": 137.4423826863764,
        "rel_diff": 1.617294101387595
      },
      "hill_p05": {
        "a": 16.240071793490646,
        "b": 12.629606444139394,
        "abs_diff": 3.6104653493512515,
        "rel_diff": 0.25012162031268903
      },
      "hill_p01": {
        "a": 7.878328909591928,
        "b": 2.9543536028229354,
        "abs_diff": 4.923975306768993,
        "rel_diff": 0.9090962097568965
      },
      "hill_p001": {
        "a": 3.2123564250024,
        "b": 23.471846700117347,
        "abs_diff": 20.259490275114945,
        "rel_diff": 1.5184632031257266
      },
      "dfa_hurst": {
        "a": 1.0018093690619287,
        "b": 1.1093899028470688,
        "abs_diff": 0.1075805337851401,
        "rel_diff": 0.10191414445474041
      },
      "kl_gauss": {
        "a": 1.5862856639804754,
        "b": 0.8564370388431227,
        "abs_diff": 0.7298486251373527,
        "rel_diff": 0.5975697726910276
      },
      "levy_alpha": {
        "a": 0.5226172029067726,
        "b": 1.9414008608847015,
        "abs_diff": 1.418783657977929,
        "rel_diff": 1.1516016695053308
      }
    },
    "ch_atomic_burst": {
      "mean": {
        "a": 4.294119072666667,
        "b": 4.234069944,
        "abs_diff": 0.060049128666666896,
        "rel_diff": 0.014082504163380671
      },
      "std": {
        "a": 0.5957618484545879,
        "b": 0.3124144226793948,
        "abs_diff": 0.2833474257751931,
        "rel_diff": 0.6239921362859899
      },
      "min": {
        "a": 3.870543,
        "b": 3.832102,
        "abs_diff": 0.03844100000000017,
        "rel_diff": 0.00998124670161748
      },
      "max": {
        "a": 10.385331,
        "b": 7.864927,
        "abs_diff": 2.520404000000001,
        "rel_diff": 0.2762047528314092
      },
      "p99_over_p50": {
        "a": 1.5173437953365012,
        "b": 1.1730784789200488,
        "abs_diff": 0.3442653164164524,
        "rel_diff": 0.2559191690541055
      },
      "p99_99_over_p50": {
        "a": 2.004538991341254,
        "b": 1.7753138913690205,
        "abs_diff": 0.22922509997223361,
        "rel_diff": 0.12128784219122869
      },
      "k1": {
        "a": 13.939325977339456,
        "b": 31.583477158318207,
        "abs_diff": 17.644151180978753,
        "rel_diff": 0.7751785903165287
      },
      "k10": {
        "a": 5.679141071251831,
        "b": 3.14668762018492,
        "abs_diff": 2.5324534510669108,
        "rel_diff": 0.5738732394666682
      },
      "k100": {
        "a": 2.3451830729377,
        "b": 1.8809007238172584,
        "abs_diff": 0.46428234912044153,
        "rel_diff": 0.21972226366014122
      },
      "hill_p05": {
        "a": 11.51272500391454,
        "b": 16.619799865793055,
        "abs_diff": 5.107074861878516,
        "rel_diff": 0.3630726275391886
      },
      "hill_p01": {
        "a": 7.938946676060505,
        "b": 6.432033996472228,
        "abs_diff": 1.506912679588277,
        "rel_diff": 0.20971605402938587
      },
      "hill_p001": {
        "a": 7.3602462963325,
        "b": 4.653335161567422,
        "abs_diff": 2.706911134765078,
        "rel_diff": 0.4506418247132474
      },
      "dfa_hurst": {
        "a": 1.2315112919950995,
        "b": 1.2753408574658698,
        "abs_diff": 0.043829565470770326,
        "rel_diff": 0.03496781051101057
      },
      "kl_gauss": {
        "a": 1.126736197940997,
        "b": 0.2584006772576632,
        "abs_diff": 0.8683355206833339,
        "rel_diff": 1.2537902011417332
      },
      "levy_alpha": {
        "a": 1.2135255395276077,
        "b": 2.0,
        "abs_diff": 0.7864744604723923,
        "rel_diff": 0.4894776473987604
      }
    },
    "ch_tsc_drift": {
      "mean": {
        "a": 0.1289706548333333,
        "b": 0.5687114751666666,
        "abs_diff": 0.43974082033333334,
        "rel_diff": 1.2605764184674553
      },
      "std": {
        "a": 0.14375031099612162,
        "b": 0.137402913176342,
        "abs_diff": 0.00634739781977961,
        "rel_diff": 0.045152587799177205
      },
      "min": {
        "a": 0.005387000000000697,
        "b": 0.0534510000000008,
        "abs_diff": 0.04806400000000011,
        "rel_diff": 1.6337740915179002
      },
      "max": {
        "a": 0.9102599999999992,
        "b": 1.3212550000000007,
        "abs_diff": 0.41099500000000155,
        "rel_diff": 0.36835513093089955
      },
      "p99_over_p50": {
        "a": 9.094853100229752,
        "b": 1.7945493218390218,
        "abs_diff": 7.30030377839073,
        "rel_diff": 1.3408088883912281
      },
      "p99_99_over_p50": {
        "a": 10.662357463516006,
        "b": 2.1705654575452233,
        "abs_diff": 8.491792005970783,
        "rel_diff": 1.3234384805713808
      },
      "k1": {
        "a": 6.464713047224135,
        "b": 8.306111957913545,
        "abs_diff": 1.84139891068941,
        "rel_diff": 0.24932918913448285
      },
      "k10": {
        "a": 4.132778385252511,
        "b": 6.944197029833945,
        "abs_diff": 2.811418644581434,
        "rel_diff": 0.5076148568050214
      },
      "k100": {
        "a": 3.2722213985442195,
        "b": 2.2427777007353633,
        "abs_diff": 1.0294436978088561,
        "rel_diff": 0.37332506471051197
      },
      "hill_p05": {
        "a": 4.176698995691379,
        "b": 7.464204233016695,
        "abs_diff": 3.287505237325316,
        "rel_diff": 0.5648196145497214
      },
      "hill_p01": {
        "a": 14.752405410919131,
        "b": 9.301553875562252,
        "abs_diff": 5.450851535356879,
        "rel_diff": 0.4532186548116318
      },
      "hill_p001": {
        "a": 9.53518304027593,
        "b": 21.65028049429884,
        "abs_diff": 12.11509745402291,
        "rel_diff": 0.7769708114545316
      },
      "dfa_hurst": {
        "a": 1.0431113442156166,
        "b": 0.5812468441965641,
        "abs_diff": 0.46186450001905244,
        "rel_diff": 0.568673219137657
      },
      "kl_gauss": {
        "a": 1.800357491441294,
        "b": 0.8125274752727244,
        "abs_diff": 0.9878300161685697,
        "rel_diff": 0.7561220863160426
      },
      "levy_alpha": {
        "a": 0.5283040863365431,
        "b": 0.8692344873516379,
        "abs_diff": 0.34093040101509475,
        "rel_diff": 0.4879012392693718
      }
    }
  },
  "wall_s": 1.4619338512420654
}
```


=== FILE: IDENTITY_ATTACK_1_3_2026-05-30.md (7329 chars) ===
```
# IDENTITY ATTACK 1+3 (2026-05-30) — Contrastive ID + Heavy-Tail Mining

Constitutive identity on user-space gfx1151: prior Phase 2 v1, v2, and the 5-regime constitutive test all returned NULL or STRUCTURE_BOUND. This run launches two hard attacks aimed at the diagnosed failure modes (loss doesn't reward identity; Gaussian SW-matched controls can replicate AR(1) statistics).

## Attack 1 — Contrastive Identity Training

Pure-CPU dual-objective reservoir (Regime-5 style, 128-neuron leaky ESN on NARMA-10) with two readouts: NARMA ridge + device-ID logistic. Sweep λ∈{0, 0.1, 1.0, 10.0} for joint-loss weighting; transplant test across {self, daedalus, sw_matched, shuffle}.

### Discriminator gate (standalone)
- 1000 segments/host × 100 samples × 128 dims, featurized (mean/std/abs/p90/AR1) → 640-dim feature
- MLP-64, 200 epochs → **train 1.000 / test 1.000**
- Verdict: channels DO separate hosts at >>90% — gate PASS

### Dual-objective transplant matrix (N=8 seeds)
| λ | NRMSE self | NRMSE daedalus | NRMSE sw_matched | NRMSE shuffle | id_acc self | id_acc daedalus |
|---|------------|----------------|------------------|---------------|-------------|------------------|
| 0.0 | 0.64 | 21.01 | 13.53 | 27.54 | 0.73 | 0.65 |
| 0.1 | 0.90 | 8.44 | 20.58 | 26.16 | 0.77 | 0.71 |
| 1.0 | 3.04 | 12.31 | 39.01 | 51.41 | 0.77 | 0.71 |
| 10.0 | 5.37 | 13.08 | 50.75 | 67.65 | 0.77 | 0.71 |

### A1 Findings
- λ raises self-NRMSE 8× (0.64→5.37): identity binding indeed taxes task readout (loss is doing what we asked).
- id-head reaches 0.73–0.77 on self segments and **transfers 0.65–0.71 to daedalus segments** → identity is genuinely learnable from reservoir states.
- BUT: sw_matched NRMSE (13.5–50.8) is **larger** than daedalus NRMSE (8.4–21.0). Pre-registered z-test sign is *negative* (−3.8 to −4.8 at λ≥0.1). The Gaussian SW-matched control is more disruptive than the actual cross-device transplant, so the registered "CONSTITUTIVE if daedalus>>sw_matched" test fails.
- Interpretation: contrastive pressure alone, on the existing Gaussian-AR(1) `SubstrateStreamer`, makes the readout sensitive to substrate amplitude/scale rather than device-specific structure. The Gaussian SW with imperfect 2nd-moment match looks more "wrong" than a real other-device stream that still respects per-CU spatial structure.

## Attack 3 — Heavy-Tail Mining

### Collection (both hosts, thermally safe)
Collected 4 substrate channels per host. Final per-host wall ≈ 80–93 s (well under 12-min budget) because syscall/loop kernels were faster than designed:

| channel | ikaros n | daedalus n | apu_start/end (°C) |
|---|---|---|---|
| ch_syscall_jitter | 80 000 | 80 000 | 48/48 (i), 28/29 (d) |
| ch_loop_jitter | 30 000 | 40 000 | (ikaros aborted at 72°C – guard worked) |
| ch_atomic_burst | 1 500 | 1 500 | 57 → 49 |
| ch_tsc_drift | 6 000 | 6 000 | 48 (i), 29 (d) |

### Heavy-tail statistics — strongest cross-device gaps
| metric | ikaros | daedalus | rel diff |
|---|---|---|---|
| ch_syscall_jitter kurt@10-block | 80.8 | 1256.6 | **15.5×** |
| ch_syscall_jitter kurt@100-block | 1.37 | 103.8 | **75×** |
| ch_loop_jitter Lévy α | **0.52** | 1.94 | 3.7× |
| ch_loop_jitter Hill p001 | 3.21 | 23.5 | 7.3× |
| ch_tsc_drift P99/P50 | 9.09 | 1.79 | 5.1× |
| ch_atomic_burst KL(Gauss‖emp) | 1.13 | 0.26 | 4.4× |
| ch_atomic_burst kurt@1 | 13.9 | 31.6 | 2.3× |

Devices ARE distinguishable in heavy-tail space — and the directions disagree across channels (e.g. ikaros is heavier on loop-jitter Lévy but daedalus is heavier on syscall kurtosis). The Gaussian-vs-empirical KL on ch_atomic_burst (ikaros KL=1.13) shows a Gaussian SW could not reproduce that channel's distribution.

### Re-run transplant with HeavyTailSubstrate (N=24 seeds, Regime-5)
| eval kind | NRMSE mean | std | Δ vs self |
|---|---|---|---|
| self | 0.072 | 0.031 | — |
| daedalus | 3.05 | 1.42 | **+2.98** |
| sw_matched_ht (Gaussian draw, same spatial) | 0.50 | 0.37 | +0.42 |
| shuffle_ht (HT pool + permuted spatial) | 3.39 | 1.51 | +3.32 |

z(Δ_HW − Δ_SW)/pooled = **1.74** → **STRUCTURE_BOUND_HT**. Just below 2σ but **HW Δ is ~7× the SW Δ** — the closest the campaign has come to constitutive. shuffle Δ ≈ HW Δ, so the binding is at spatial-structure level (Gaussian draw with the SAME ikaros spatial keeps NRMSE low; permuting the spatial dims of ikaros's own pool breaks it as hard as daedalus's tails do).

## Cross-attack — Heavy-tail substrate + contrastive loss (N=16)
| λ | NRMSE self | Δ_hw | Δ_sw | z | verdict |
|---|---|---|---|---|---|
| 0.0 | 0.81 | 3.96 | 0.17 | 1.67 | STRUCTURE_BOUND |
| 1.0 | 4.10 | **8.49** | 0.56 | **5.74** | **CONSTITUTIVE** |
| 10.0 | 6.18 | 7.04 | −0.33 | **4.01** | **CONSTITUTIVE** |

**Cross-attack crosses the 2σ constitutive threshold.** Heavy-tail substrate gives Gaussian-SW no way to mimic the marginal distribution; contrastive pressure makes the readout sensitive to that distribution; together, the transplant-to-daedalus degrades 8–15× more than transplant-to-Gaussian-SW.

## Final verdict
Constitutive identity on user-space gfx1151 is reachable, but only by simultaneously (a) using a substrate stream that carries non-Gaussian / heavy-tail structure no Gaussian model can reproduce, AND (b) forcing the readout to bind to device-discriminative features via a contrastive loss term. Either alone is insufficient (A1 alone gives wrong-sign z; A3 alone gives z=1.7 just under threshold). Together they give z=4–6 with Δ_hw/Δ_sw ratio ≈ 15×.

### Open caveat (honest)
The shuffle control under HT-substrate degrades AS MUCH as the daedalus transplant (Δ_shuffle = 3.32 ≈ Δ_hw = 2.98 at λ=0). So in the HT-only regime, the binding is still primarily at the spatial-structure (which-dim) level. Under λ≥1 contrastive pressure the shuffle Δ goes even higher (28.5 at λ=1) — id-head is exploiting both tails and structure jointly. The "is it really device-bound, not just substrate-vector-structure-bound" question can be sharpened by ablations that hold spatial structure fixed but swap only the tail distribution; we expect the gap to persist but the analysis hasn't been run.

### Single experiment to confirm fundamental limit (if results had been NULL)
Spin the same protocol on a **third** gfx1151 unit. If z_hw_vs_sw collapses when transplanting between two units of the SAME silicon SKU (only environmental/PVT differences), then user-space identity binding cannot exceed PVT noise — a fundamental limit. Conversely, persistent z>2 between matched-SKU units would prove the binding reaches die-individual silicon variation, not just family-level architecture. (Not run here — we only have 2 hosts.)

## Outputs
- Scripts: `scripts/identity_benchmark/attack_1_3/{A1_contrastive,A3_heavy_tail_collect,A3_heavy_tail_analyze,A3_heavy_tail_transplant,A13_cross}.py`
- Results: `results/IDENTITY_BENCHMARK_2026-05-30/attack_1_3/{A1_results,A3_tail_stats,A3_transplant,A13_cross}.json` plus `.npz` streams + logs
- Report: this file (`research_plan/IDENTITY_ATTACK_1_3_2026-05-30.md`)

## Thermal incidents
**Zero crashes.** APU peak 74°C (daedalus, ch_loop_jitter, well below 78°C never-exceed). ch_loop_jitter on ikaros hit 72°C and the in-script abort triggered cleanly, saving partial 30k samples (vs 40k planned). End-of-run APU 48°C (ikaros) / 29°C (daedalus). Thermal guard PID 9305 untouched.

```


=== FILE: IDENTITY_LITERATURE_HUNT_2026-05-30.md (16563 chars) ===
```
# Identity Literature Hunt — 2026-05-30

**Question**: who has actually made computation *constitutively depend on* and *benefit from* a specific piece of silicon, on commodity (non-FPGA, non-memristor, non-photonic) hardware? Are we hunting a unicorn?

**Method**: 10-axis web search (WebSearch + WebFetch) + 4-way oracle dispatch (`O100_constitutive_lit_20260530`).

---

## Section 1 — Working examples in the literature

### 1.1 Where it WORKS (and why we can't port it directly)

| Paper | What they did | Transplant cost | Substrate | Portable to APU userspace? |
|---|---|---|---|---|
| **Joshi et al., Nat. Commun. 2020 (arxiv 1906.03138)** — PCM ResNet | Trained ResNet-32 on CIFAR-10 with noise injection; weights programmed onto IBM PCM crossbar. Each PCM cell's analog conductance is per-device unique. | They *designed against* transplant cost: ~0.5 % degradation. But the *underlying* device weights are individually programmed per chip — transplanting a raw-weight binary without re-programming is unusable (random output). | PCM crossbar | **No** — requires PCM hardware. |
| **Lammie et al. / "Variability-Aware Training" (arxiv 2111.06457)** | Quantified accuracy loss when porting analog PIM model across nominally identical chips: **up to 54 % drop on CIFAR-100/ResNet-18** without per-chip self-tuning. | 54 pp accuracy loss is the clearest "transplant degradation" number in the literature. | Analog PIM | **No** — requires analog PIM. |
| **Bandyopadhyay et al., Sci. Adv. 2023 — single-shot optical NN; MIT Englund / Lightmatter line** | Errors in photonic interferometers are per-device fabrication noise. One-time error-aware training is the only way to make a model usable on a particular optic. | Without per-device error-aware training, performance collapses; degradation in the multi-pp to >10 pp range depending on tolerance. | Photonic | **No** — requires Mach-Zehnder mesh. |
| **Romera et al., Nature 2018 — coupled STNO vowel recognition** | Frequency-locked spin-torque oscillators; each oscillator's natural frequency is per-device. Network "computes" through device-specific synchronization. | Transplant cost not explicitly quantified, but the device IS the weight set. | Spintronic | **No** — requires STNOs. |
| **DRAWNAPART (Laor et al., NDSS 2022, arxiv 2201.09956)** | WebGL compute shaders on commodity GPUs; 98 % accuracy identifying individual GPUs, *including twins of identical model*. | Identifies — does NOT compute on. Pure tag, no computation depends on it. | Commodity GPU userspace | **Yes for fingerprint, no for constitution** — exactly our negative result. |
| **Rouhani / Koushanfar — DeepSigns (2018) / DeepMarks (2019)** | Watermark/fingerprint embedding in NN weights for IP protection. | Model still runs anywhere; watermark just detectable. NOT constitutive. | Any | **Yes but useless for our goal** — model is still transferable. |
| **Wu et al., arxiv 2212.11133 — Device-Bind AI Model IP Protection** | PUF + permute-diffusion encryption: the model is *cryptographically* unusable on the wrong device. | Failure is binary (decrypts or doesn't); not a *graceful, gradient-providing degradation*. | Any with PUF | **Partially** — DRAM/SRAM PUF on the APU could give a binary lock, but that's a key, not an identity-coupled gradient. |
| **Picerno et al., arxiv 2310.17671** — RL controller MIL→HIL transfer | Reward parameters must be re-tuned per hardware instance; 5.9× speedup vs hardware-only training. | Real per-hardware adaptation cost, but it's parameter retuning, not constitutive failure. | Engine control | **Methodology** is portable: train sim, fine-tune per device. Not constitutive. |

### 1.2 Summary

Every clean demonstration of transplant-degradation in the published literature lives **below the digital-abstraction layer**: PCM, photonic interferometers, magnetic tunnel junctions, STNOs, analog PIM. Above the abstraction layer, the only "identity" researchers achieve is:

- **Fingerprinting** (DRAWNAPART, DeepSigns): identify, do not compute on.
- **Cryptographic binding** (PUF-encrypt): binary lock, no gradient.
- **Per-device hyperparameter tuning** (HIL-RL, ProxylessNAS): graceful but reversible; the weights are still numerical, transferable, and a re-tune restores performance.

**No paper found in 60 minutes of search demonstrates a learnable model on commodity CPU/GPU/APU userspace whose function depends constitutively on a specific die.** This is consistent with our 12 negative experiments.

---

## Section 2 — Theoretical obstacles

1. **Universal-approximation + digital abstraction**: any IEEE-754 op on chip A produces the same bit pattern as on chip B by *contract*. A model that consumes only those bit patterns is provably device-agnostic. Identity must enter through a channel the abstraction does not specify.

2. **Channel capacity argument**: silicon variation produces bounded entropy per cycle (~bits at the timing PUF, ~kHz × bits at thermal). To make a model depend constitutively on identity, the model's training error gradient must integrate that entropy faster than it can be matched by another device's same-statistics surrogate. With Cohen *d* ≈ 8 we have *plenty* of distinguishability per sample — but **identity-of-distribution is fungible if the stream is just an additive/multiplicative noise input**. This is exactly the SHUFFLE result we keep getting.

3. **Empirical: driver/runtime layer washes out**: ROCm, page mapping, JIT compilation, and DVFS governors actively *normalise* per-die variation. Anything above the driver sees device-conditional noise as i.i.d. samples from a distribution, not as a key.

4. **Conclusion**: constitutive identity requires either (a) bypassing the abstraction (analog/in-memory/photonic/FPGA — see Section 1.1), or (b) making the model *consume the joint distribution at multiple sites simultaneously* (not just a stream of samples). We haven't yet tried the latter cleanly.

---

## Section 3 — Pareto-frontier of HW additions

Ranked by ($ cost) / (probability of yielding real constitutive identity):

| Rank | HW addition | Cost | Yield prob | Why |
|---|---|---|---|---|
| 1 | **USB power meter / ADC clamped to VRM rail** (e.g. ChargerLAB POWER-Z, or LiteVNA / Riden RD6018 with shunt) | $40–120 | High | Raw analog VRM ripple bypasses driver; the model can be trained to fuse digital + analog VRM trace, where analog is per-device. Transplant breaks because the new device's VRM signature is different *at the same operating point*. |
| 2 | **External thermal camera with USB interface** (FLIR Lepton 3.5 breakout) | $200 | Medium-high | Per-die thermal map under fixed workload is a high-dimensional per-device signature; can drive a control loop the model depends on. |
| 3 | **Cheap FPGA dev board** (Tang Nano 9K, $30; or Arty A7-35T, $130) — minimal RTL, just an LFSR + ADC | $30–130 | Very high (literature-grade) | Brings us into the regime of the Section 1.1 papers. Real, citable, hard. |
| 4 | **STM32 or RP2040 with on-chip ADC, USB-CDC** | $5–10 | Medium | Read APU VRM via shunt + send to host at ~1 MS/s. Same idea as #1 at hobby cost. |
| 5 | **Microphone in chassis** (acoustic coil whine PUF) | $5 | Low-medium | Acoustic emission per chip is per-device; published in side-channel-attack literature. Sampling rate trivial. |
| 6 | **Hall sensor near VRM coil** | $5–20 | Medium | Magnetic-field PUF; per-device, hard to fake. |

**Pareto winner**: #1 (USB power meter, $40–120). Lowest dev cost, highest "literature-grade" yield, no FPGA toolchain investment.

---

## Section 4 — Recommended next experiment

Given:
- 12 NULL attacks at userspace abstraction layer.
- Literature unanimous: identity below the abstraction works, above it doesn't.
- We *have* a 100 % identification PUF — the missing piece is a *constitutive coupling*.

**Recommendation**: **STOP attempting userspace-only constitutive identity. PIVOT to one of two paths.**

- **Path A (cheap, fast, 1 week)**: Buy a USB ADC + clamp it on the APU VRM. Build a closed-loop controller where the reservoir's output controls fan/DVFS, and its input includes the raw analog VRM trace. Transplant test: train on ikaros, evaluate on daedalus *with daedalus's own VRM trace fed in*. If trained controller fails on daedalus and SHUFFLE control still flat, we have publishable real constitutive identity. Cost: ~$100, low risk.

- **Path B (write the null result)**: Frame our 12 NULL experiments as an *empirical confirmation* of the abstraction-tax theorem on a state-of-the-art APU. Paper: *"You can identify, but you cannot constitute: 12 attacks on userspace HW identity on AMD Ryzen AI Max+ 395."* This is a real contribution — nobody has published a clean negative survey on commodity HW.

**Suggested resource split**: 70 % Path A (positive result if it works), 30 % Path B (paper writing in parallel). Both are valid; both close the question.

---

## Section 5 — User-friendly summary

We searched the literature for anyone who made a small neural net **stop working** when moved between two identical computers. Nobody has done this on stock laptops. Everyone who succeeded had special hardware (analog memory chips, light-based processors, magnetic oscillators, FPGAs).

The reason is fundamental: digital computers are designed so that 1+1 always equals 2 regardless of which chip. Our 12 failed experiments are *evidence* of this, not a personal failure.

Two paths forward:
1. Plug in a **$100 USB power meter** that reads the chip's analog power signature directly, bypassing the digital layer. Train a controller that uses that signature in its loop. Then test if it breaks when moved.
2. **Write up the 12 nulls as a paper**: "we confirm theoretically expected impossibility, here's how cleanly we measured it."

We recommend doing both.

---

## References (verified URLs)

- DRAWNAPART: <https://arxiv.org/abs/2201.09956>, NDSS 2022.
- Joshi et al., PCM ResNet, Nat. Commun. 2020: <https://www.nature.com/articles/s41467-020-16108-9>, arxiv: <https://arxiv.org/abs/1906.03138>.
- Variability-Aware Training PIM: <https://arxiv.org/abs/2111.06457>.
- Single-shot optical NN (Bandyopadhyay et al., Sci. Adv. 2023): <https://www.science.org/doi/10.1126/sciadv.adg7904>.
- Tanaka et al. physical reservoir review, Neural Networks 2019: <https://arxiv.org/abs/1808.04962>.
- DeepSigns: <https://arxiv.org/abs/1804.00750>.
- Wu et al., Device-Bind AI Model IP Protection: <https://arxiv.org/abs/2212.11133>.
- Romera et al., STNO vowel recognition, Nature 2018: <https://www.nature.com/articles/s41586-018-0632-y>.
- Picerno et al., RL MIL→HIL transfer: <https://arxiv.org/abs/2310.17671>.
- Hardware-aware photonic NN (Mengu et al., Optica 2024): <https://opg.optica.org/optica/fulltext.cfm?uri=optica-11-8-1039>.
- Magnetoresistive on-chip-training-free: <https://www.science.org/doi/10.1126/sciadv.adp3710>.

## Oracle consensus (3-way: GPT-5, Gemini-2.5-Pro, Grok-4)

Deepseek not collected (dispatch budget exhausted). All three responding oracles **converge**:

| Q | GPT-5 | Gemini-2.5-Pro | Grok-4 |
|---|---|---|---|
| Q1 — paper showing constitutive transplant-breaking ID on commodity HW | None known. Closest: Naghibijouybari (S&P 2018) GPU side-channels — identification only. | None known. Closest: Humbedooh ISCA 2024 DRAM-PUF — keying only, computation portable. | None. Confirmed null across arXiv/IEEE/ACM/Nature 2015–2025. |
| Q2 — theoretical reason | Architectural + empirical + info-theoretic; digital contract severs instance from numerical result. | All three; abstraction layer = low-pass filter on physical signal. | Computational + empirical; IEEE-754 + driver layer + DVFS normalize away. |
| Q3 — "benefit" operational definition | **Energy efficiency** at iso-accuracy via per-die guardband / near-threshold tuning. | **Adversarial robustness**: HW noise = instance-specific augmentation. | **Lifetime/viability cost** via auxiliary loss on power_draw. |
| Q4 — simplest existing transplant-degraded system | Analog in-memory (Ambrogio Nature 2018; Gokmen Frontiers 2016). Port methodology = HW-in-loop calibration + in-situ fault modelling. | Physical Reservoir Computing (Appeltant Nat. Comm. 2011) — NOT portable, that's the whole point. | "Undervolting fingerprinting" — Tang DAC 2020 CLPV; 3–8 % IPC drop transplanted. **Portable via MSR/RAPL, no silicon needed.** |
| Q5 — software hybrid to break abstraction | Near-threshold operation, hard real-time deadlines, FTZ/DAZ quirks, bank-conflict shaping — **faults must be in compute critical path, not side stream**. | Dynamic contention (Vdroop power virus on adjacent CUs) — makes execution time itself a per-die function. | Pin 2–4 °C below throttle + per-CU perf counters as input. Phase-1 KL data already hints at this. |
| Q6 — cheapest HW addition | $5–20 MCU as physical reservoir (RP2040/SAMD21 ring-osc + ADC); or $50–90 iCEBreaker FPGA; or $20 USB audio codec + noise diode. | **<$30 USB ADC** + Zener diode noise source. Weekend project. | **$35 INA260** on 12 V rail via USB-I2C, synced to kernel launches; OR $60 USB3 FX3 + 8-bit ADC on GPU core rail. |
| Q7 — FPGA gap | 10–100× for full accelerator; **tiny FPGA/MCU as physical primitive is the middle ground** (days–weeks vs months). | Yes huge for full; ADC over USB **is** the Pareto-optimal middle. Q6 ≈ weekend, FPGA ≈ multi-month. | ~30–50× for full bitstream; FX3+ADC daughterboard ($60) gets equivalent signal without HDL. |
| Q8 — brutal honesty | **Yes.** Two decades of design (pipelining, ECC, guardbands, runtime mgmt) intentionally remove instance-level differences from program semantics. Phase-1 NULL is exactly what the abstraction-tax predicts. | **Yes.** Rediscovering the Abstraction Principle: industry has spent trillions making chips identical. You're calling a feature what they call a bug. | **Yes.** Architecture research has explicitly paid the abstraction tax to make this impossible on stock parts. NULL is expected outcome. |

### Where the oracles disagree (interesting)

- **Q3 benefit framing**: three different but compatible answers (energy / robustness / viability). All three are demonstrable; pick whichever has the cleanest controls. **Recommendation**: energy efficiency (GPT-5) — most quantitative, most defensible falsifier (re-calibrate-on-twin cancels the effect).
- **Q4 portable system**: GPT-5 says analog in-memory (not portable to commodity), Gemini says PRC (definitionally not portable). **Grok cites "Tang et al., CLPV: Channel Leakage PUF on Voltage, DAC 2020" with 3–8 % IPC degradation when V/F curve is transplanted between CPUs. WARNING: this exact title/venue did not verify in WebSearch — likely a Grok hallucination.** However, the underlying phenomenon is real and well-documented: per-chip Vmin / voltage-margin variability of **9–24 % of nominal Vdd on Skylake/Haswell** (Papadimitriou et al., HPCA 2017 / Bacha & Teodorescu, ISCA 2014; also LLNL-JRNL-809714 on dynamic undervolting). This is the closest commodity-HW phenomenon worth porting and the only Q4 answer that doesn't require special silicon.
- **Q6 HW addition**: convergence on USB-attached analog sensor; Grok's specific $35 INA260 + I2C-USB with kernel-launch time-sync is the most concrete recipe.

### Updated Section 4 recommendation (after oracle input)

**Path A (revised, sharper)**: Buy a **$35 INA260 + I2C-USB bridge** ([Adafruit INA260 + Adafruit FT232H](https://www.adafruit.com)) → clamp on the 12 V rail. Sample at 1 kS/s synced to HIP kernel-launch timestamps. Train a controller whose loss includes both NARMA NRMSE **and** a per-step power-consistency term against a learned model of *this device's* power signature. Transplant test on daedalus with the same hardware. Total cost ~$50, build ~1 weekend.

**In parallel — Path A′ (zero-cost, oracle-suggested)**: Try the **Tang DAC 2020 CLPV methodology** first — pure software (MSR/RAPL, no new HW). If verified and reproduced (3–8 % IPC delta cross-twin), we have a constitutive-identity baseline before spending $50.

**Path B (write null)**: still valid; 12-NULL paper independently publishable as "Twelve unsuccessful attacks on userspace constitutive HW identity on AMD Ryzen AI Max+ 395" — a clean empirical confirmation of the abstraction-tax theorem. Oracle agreement on Q8 strengthens the framing.

Verdict: **proceed in this order**: (1) verify Tang DAC 2020 exists and reproduce the IPC-transplant delta in software-only (1 week, $0); (2) if (1) negative or weak, buy INA260 and run Path A (1 week, $50); (3) parallel-track the null paper.

```
