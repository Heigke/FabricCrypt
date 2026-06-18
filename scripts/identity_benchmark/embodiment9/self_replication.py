"""Phase 9 Task D — self-replication / body-knows-itself.

At time t, predict whether the model's OWN output at time t+H seconds will
match a reference computation of the same model on the same input,
re-evaluated against fresh live substrate.

Setup:
  - Constitutive reservoir from Task B is the model that drives outputs.
  - "Reference" forward pass at t+H uses the model again on the same input
    window, but with substrate at THAT future time → output differs.
  - Self-predictor (a small MLP head over reservoir state at t) must predict
    whether the model's own t+H output will be within epsilon of the
    reference (binary classification).
  - Only a model coupled to its own chassis's substrate trajectory should be
    able to predict its own future state above chance, because the
    substrate-determined alpha/gain rolls forward in time deterministically
    relative to its own chassis dynamics.

Pre-reg: ikaros (own substrate at train + eval): F1 ≥ 0.7
         transplant (daedalus-trained → ikaros eval): F1 ≤ 0.5
         random (constant alpha): F1 ≤ 0.5
"""
from __future__ import annotations
import json
import socket
from pathlib import Path
import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from constitutive import (ConstitutiveReservoir, load_data, make_windows,
                          HIST, HORIZON, D, OUT_DIR, sigmoid)

HOST = socket.gethostname()


def compute_self_label(model, X, sub_now, sub_future, eps=0.5):
    """For each window, the 'label' is whether the model's prediction with
    sub_future is within eps L2 of its prediction with sub_now.

    Higher label rate = model is consistent across substrate; we want a
    BALANCED label set, so we will calibrate eps to ~50/50 below.
    """
    H_now = model.features(X, sub_now)
    H_fut = model.features(X, sub_future)
    Y_now = H_now @ model.W_out + model.b_out
    Y_fut = H_fut @ model.W_out + model.b_out
    diffs = np.linalg.norm(Y_now - Y_fut, axis=1)
    return diffs, H_now


def fit_self_predictor(H, labels, lam=1e-2):
    """Logistic-ish ridge: predict binary label from reservoir state H."""
    n = len(H)
    H1 = np.concatenate([H, np.ones((n, 1), dtype=np.float32)], axis=1)
    y = labels.astype(np.float32) * 2 - 1  # -1/+1
    A = H1.T @ H1 + lam * np.eye(H1.shape[1], dtype=np.float32)
    w = np.linalg.solve(A, H1.T @ y)
    return w


def predict_self(w, H):
    n = len(H)
    H1 = np.concatenate([H, np.ones((n, 1), dtype=np.float32)], axis=1)
    return (H1 @ w > 0).astype(np.int32)


def f1_score(y_true, y_pred):
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    if tp + fp == 0 or tp + fn == 0:
        return 0.0
    prec = tp / (tp + fp)
    rec = tp / (tp + fn)
    if prec + rec == 0:
        return 0.0
    return float(2 * prec * rec / (prec + rec))


def run_condition(model_train_host, eval_substrate_host, mode, substrates,
                  n_seeds=10):
    """Returns dict with per-seed F1."""
    data_train = load_data(model_train_host)
    data_eval = load_data(HOST)
    sub_now_full = substrates[eval_substrate_host]
    # 'future' substrate: shift by ~10 steps
    sub_now = sub_now_full[np.linspace(0, len(sub_now_full)-1, HIST).astype(int)]
    sub_future_full = np.roll(sub_now_full, -10, axis=0)
    sub_future = sub_future_full[np.linspace(0, len(sub_future_full)-1, HIST).astype(int)]

    sub_train_full = substrates[model_train_host]
    sub_train = sub_train_full[np.linspace(0, len(sub_train_full)-1, HIST).astype(int)]

    mu = data_train.mean(0, keepdims=True); sd = data_train.std(0, keepdims=True) + 1e-6

    f1s = []
    for seed in range(n_seeds):
        Xtr, Ytr = make_windows(data_train, 400, HIST, HORIZON, seed)
        Xtr = (Xtr - mu) / sd; Ytr = (Ytr - mu) / sd
        # train constitutive reservoir on training host with training-host substrate
        model = ConstitutiveReservoir(D, D, seed=seed, coupling_mode=mode)
        model.fit(Xtr, Ytr, sub_train)
        # Eval windows from eval host
        Xte, Yte = make_windows(data_eval, 250, HIST, HORIZON, seed + 9001)
        Xte = (Xte - mu) / sd
        diffs, H_now = compute_self_label(model, Xte, sub_now, sub_future)
        # Calibrate eps to median → balanced labels
        eps = float(np.median(diffs))
        labels = (diffs < eps).astype(np.int32)  # 1 = self-consistent at this future
        # 50/50 split: half for training the self-predictor, half for eval
        split = len(Xte) // 2
        w = fit_self_predictor(H_now[:split], labels[:split])
        y_pred = predict_self(w, H_now[split:])
        y_true = labels[split:]
        f1s.append(f1_score(y_true, y_pred))
    return {"f1_per_seed": f1s, "mean": float(np.mean(f1s)), "std": float(np.std(f1s))}


def main():
    substrates = {}
    for h in ("ikaros", "daedalus"):
        p = OUT_DIR / f"substrate_{h}.npy"
        if not p.exists():
            print(f"[!] missing {p}"); return
        substrates[h] = np.load(p)

    other = "daedalus" if HOST == "ikaros" else "ikaros"
    conditions = {
        "own_substrate":       (HOST,  HOST,  "constitutive"),
        "transplant":          (other, HOST,  "constitutive"),
        "alien_substrate":     (HOST,  other, "constitutive"),
        "no_coupling_control": (HOST,  HOST,  "control_const"),
    }
    results = {"host": HOST, "conditions": {}}
    for name, (train_h, eval_sub_h, mode) in conditions.items():
        r = run_condition(train_h, eval_sub_h, mode, substrates, n_seeds=10)
        results["conditions"][name] = r
        print(f"  {name:24s}  F1 = {r['mean']:.3f} ± {r['std']:.3f}")

    own_f1 = results["conditions"]["own_substrate"]["mean"]
    transplant_f1 = results["conditions"]["transplant"]["mean"]
    nocoup_f1 = results["conditions"]["no_coupling_control"]["mean"]
    results["gates"] = {
        "own_ge_0.7": bool(own_f1 >= 0.7),
        "transplant_le_0.5": bool(transplant_f1 <= 0.5),
        "own_minus_transplant": float(own_f1 - transplant_f1),
        "own_minus_nocoup": float(own_f1 - nocoup_f1),
    }
    print(f"\n  Gates: {results['gates']}")

    out = OUT_DIR / f"self_replication_{HOST}.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"\n  Saved → {out}")


if __name__ == "__main__":
    main()
