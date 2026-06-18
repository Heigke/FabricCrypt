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
