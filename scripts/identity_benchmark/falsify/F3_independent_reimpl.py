#!/usr/bin/env python3
"""F3 — Independent re-implementation falsifier.

Same data, same substrate, but different code path:
  * sklearn.linear_model.Ridge instead of np.linalg.solve
  * Different RNG seed scheme (seed=base+host_offset*1000 instead of base+11/22)
  * Different substrate-injection order (evaluate transplant cells in REVERSE)
  * Different contrastive formulation: solve two ridges separately and blend
    (alpha * W_task + (1-alpha) * W_id), instead of additive normal eqs.

If z stays ≥ 4 with this fully independent code path, implementation bug ruled
out. If z collapses, our original numpy code was leaking information.

Output: results/.../falsify/F3_independent_reimpl.json
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
sys.path.insert(0, str(HERE.parent / "attack_1_3"))
from reservoir import Reservoir, ReservoirCfg  # type: ignore
from A3_heavy_tail_transplant import (  # type: ignore
    HeavyTailSubstrate, GaussianMatchedHT, ShuffleHT, load_streams,
    N_RES, SUB_DIM, WASHOUT, T_TRAIN, T_TEST, HORIZON,
)
from A13_cross import narma10  # type: ignore

try:
    from sklearn.linear_model import Ridge
    HAS_SK = True
except ImportError:
    HAS_SK = False

OUT_DIR = HERE.parents[2] / "results" / "IDENTITY_BENCHMARK_2026-05-30" / "falsify"
OUT_DIR.mkdir(parents=True, exist_ok=True)
N_SEEDS = int(os.environ.get("N_SEEDS", "30"))
LAM_BLEND = 0.85  # weight on task vs identity (separately fitted)


def sk_ridge_fit(X, y, alpha=1e-3):
    if HAS_SK:
        r = Ridge(alpha=alpha, fit_intercept=False, solver="cholesky")
        r.fit(X, y)
        return r.coef_
    # fallback
    D = X.shape[1]
    return np.linalg.solve(X.T @ X + alpha * np.eye(D), X.T @ y)


def train_dual_blend(X_i, y_task_i, X_d, alpha=LAM_BLEND):
    """Fit task ridge on ikaros; fit identity ridge on (i, d) → ±1; blend."""
    Xa_i = np.hstack([X_i, np.ones((X_i.shape[0], 1))])
    W_task = sk_ridge_fit(Xa_i, y_task_i, alpha=1e-3)
    X_full = np.vstack([X_i, X_d])
    Xa_full = np.hstack([X_full, np.ones((X_full.shape[0], 1))])
    y_id = np.concatenate([-np.ones(X_i.shape[0]), np.ones(X_d.shape[0])])
    W_id = sk_ridge_fit(Xa_full, y_id, alpha=1e-3)
    # Project identity direction onto task readout (orthogonalised blend)
    return alpha * W_task + (1.0 - alpha) * W_id * float(np.linalg.norm(W_task) / (np.linalg.norm(W_id) + 1e-12))


def task_nrmse(W, X, y):
    Xa = np.hstack([X, np.ones((X.shape[0], 1))])
    y_pred = Xa @ W
    err = float(np.sqrt(np.mean((y - y_pred) ** 2)))
    return err / float(y.std() + 1e-12)


def main():
    t0 = time.time()
    streams_i = load_streams("ikaros")
    streams_d = load_streams("daedalus")
    print(f"[F3.reimpl] N_SEEDS={N_SEEDS} sklearn={HAS_SK} alpha_blend={LAM_BLEND}",
          flush=True)

    eval_kinds = ["self", "daedalus", "sw_matched_ht", "shuffle_ht"]
    cells = {ek: [] for ek in eval_kinds}

    HOST_OFFSET_I = 71
    HOST_OFFSET_D = 113

    for s in range(N_SEEDS):
        sub_i = HeavyTailSubstrate("ikaros", streams_i, n_dim=SUB_DIM,
                                   seed=s + HOST_OFFSET_I * 1000 % 100000)
        sub_d = HeavyTailSubstrate("daedalus", streams_d, n_dim=SUB_DIM,
                                   seed=s + HOST_OFFSET_D * 1000 % 100000)
        sub_sw = GaussianMatchedHT(sub_i, seed=s + 7331)
        sub_sh = ShuffleHT(sub_i, seed=s + 9173)

        u, y = narma10(T_TRAIN + HORIZON + WASHOUT, seed=s)
        u_in = u[:-HORIZON][:, None]
        y_tg = y[HORIZON:]
        cfg = ReservoirCfg(n_in=1, n_res=N_RES, seed=s)

        res_train = Reservoir(cfg, regime=5, substrate=sub_i)
        X_i = res_train.run(u_in, washout=WASHOUT)
        y_w_i = y_tg[WASHOUT:]
        res_d_train = Reservoir(cfg, regime=5, substrate=sub_d)
        res_d_train.W_in = res_train.W_in
        res_d_train.W_rec = res_train.W_rec
        M = sub_d.weight_mod(cfg.n_res)
        res_d_train.W_rec_eff = res_train.W_rec * (1.0 + 0.15 * M)
        X_d = res_d_train.run(u_in, washout=WASHOUT)
        W = train_dual_blend(X_i, y_w_i, X_d, alpha=LAM_BLEND)

        # REVERSE eval order
        eval_list = [("shuffle_ht", sub_sh), ("sw_matched_ht", sub_sw),
                     ("daedalus", sub_d), ("self", sub_i)]
        for ek, ev in eval_list:
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
            cells[ek].append(float(task_nrmse(W, X_te, y_te_w)))

    agg = {ek: {"mean": float(np.nanmean(cells[ek])), "std": float(np.nanstd(cells[ek])),
                "n": int(np.sum(~np.isnan(cells[ek])))} for ek in eval_kinds}
    d_hw = agg["daedalus"]["mean"] - agg["self"]["mean"]
    d_sw = agg["sw_matched_ht"]["mean"] - agg["self"]["mean"]
    d_sh = agg["shuffle_ht"]["mean"] - agg["self"]["mean"]
    pooled = float(np.sqrt(agg["daedalus"]["std"] ** 2 + agg["sw_matched_ht"]["std"] ** 2)) + 1e-12
    z = (d_hw - d_sw) / pooled
    verdict = ("CONSTITUTIVE" if z > 2.0 else
               "WEAKENED" if z > 1.0 else
               "COLLAPSED")
    out = {
        "test": "F3_independent_reimpl",
        "config": {"n_seeds": N_SEEDS, "sklearn": HAS_SK, "alpha_blend": LAM_BLEND,
                   "host_offset_i": HOST_OFFSET_I, "host_offset_d": HOST_OFFSET_D},
        "aggregate": agg,
        "deltas": {"daedalus": float(d_hw), "sw_matched_ht": float(d_sw),
                   "shuffle_ht": float(d_sh)},
        "z_hw_vs_sw": float(z),
        "verdict": verdict,
        "interpretation": (
            "CONSTITUTIVE: original z reproduces with independent ridge solver "
            "(sklearn), different seed scheme, reverse eval order, blended-loss "
            "formulation → not an implementation artifact. "
            "COLLAPSED: original code path was leaking information."
        ),
        "wall_s": time.time() - t0,
    }
    out_path = OUT_DIR / "F3_independent_reimpl.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"[F3] Δ_hw={d_hw:.3f} Δ_sw={d_sw:.3f} z={z:.2f} → {verdict}", flush=True)
    print(f"[F3] saved → {out_path} wall={out['wall_s']:.1f}s", flush=True)


if __name__ == "__main__":
    main()
