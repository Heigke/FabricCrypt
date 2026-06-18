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
