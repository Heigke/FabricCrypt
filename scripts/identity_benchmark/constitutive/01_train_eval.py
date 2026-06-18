"""Run the full 5-regime x transplant matrix x controls.

Cells:
  train_host in {ikaros, daedalus}
  eval_host  in {ikaros, daedalus, sw_matched, shuffle, ident_const}
For each regime r in 0..5, run N_SEEDS seeds; the substrate object for
'sw_matched' / 'shuffle' / 'ident_const' is a control variant.

For regime 0 the substrate is unused (baseline floor), so we only report
the diag NRMSE once.

Output: results/IDENTITY_BENCHMARK_2026-05-30/constitutive/regime_{r}_results.json
"""
from __future__ import annotations
import json, time, sys, os
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from _substrate_stream import SubstrateStreamer, GaussianMatched, IdentConstant, PermutedSubstrate
from reservoir import Reservoir, ReservoirCfg, ridge_fit, nrmse, mackey_glass

OUT = HERE.parents[2] / "results" / "IDENTITY_BENCHMARK_2026-05-30" / "constitutive"
OUT.mkdir(parents=True, exist_ok=True)

N_SEEDS = int(os.environ.get("N_SEEDS", "12"))
T_TRAIN = 1500
T_TEST = 800
WASHOUT = 100
N_RES = 32
SUB_DIM = 32  # substrate dimensionality projection (matches n_res for regime 5)
HORIZON = 1   # one-step-ahead prediction


def build_substrate(host_or_kind: str, ref: SubstrateStreamer | None, seed: int):
    if host_or_kind in ("ikaros", "daedalus"):
        return SubstrateStreamer(host_or_kind, n_dim=SUB_DIM, seed=seed)
    if host_or_kind == "sw_matched":
        return GaussianMatched(ref, seed=seed + 7)
    if host_or_kind == "ident_const":
        return IdentConstant(ref)
    raise ValueError(host_or_kind)


def one_run(regime: int, train_sub, eval_sub, seed: int):
    rng = np.random.default_rng(seed + 1000)
    sig_tr = mackey_glass(T_TRAIN + HORIZON + WASHOUT, tau=5, seed=seed)
    sig_te = mackey_glass(T_TEST + HORIZON + WASHOUT, tau=5, seed=seed + 9999)
    u_tr = sig_tr[:-HORIZON][:, None]
    y_tr = sig_tr[HORIZON:]
    u_te = sig_te[:-HORIZON][:, None]
    y_te = sig_te[HORIZON:]

    cfg = ReservoirCfg(n_in=1, n_res=N_RES, seed=seed)

    # TRAIN with train_sub
    res_train = Reservoir(cfg, regime=regime, substrate=train_sub)
    X_tr = res_train.run(u_tr, washout=WASHOUT)
    y_tr_w = y_tr[WASHOUT:]
    X_tr_aug = np.hstack([X_tr, np.ones((X_tr.shape[0], 1))])
    W_out = ridge_fit(X_tr_aug, y_tr_w, alpha=1e-4)

    # EVAL with eval_sub (rebuild reservoir to swap substrate; keep same seeds/W)
    res_eval = Reservoir(cfg, regime=regime, substrate=eval_sub)
    # Ensure SAME learned recurrent weights (the regime-4 weight_mod uses
    # eval substrate by design — that IS the transplant). For regime 4 the
    # transplant naturally flips the modulation. For regimes 2/3/5 the IC /
    # leak / dyn are also taken from eval_sub. This is intentional: identity
    # is encoded by ALL silicon-bound coefficients.
    # However: keep base W_rec / W_in identical to the train run.
    res_eval.W_in = res_train.W_in
    res_eval.W_rec = res_train.W_rec
    # but recompute W_rec_eff for regime 4 with eval substrate's mod
    if regime == 4 and eval_sub is not None:
        M = eval_sub.weight_mod(cfg.n_res)
        res_eval.W_rec_eff = res_train.W_rec * (1.0 + cfg.weight_mod_strength * M)
    elif regime == 5 and eval_sub is not None:
        M = eval_sub.weight_mod(cfg.n_res)
        res_eval.W_rec_eff = res_train.W_rec * (1.0 + 0.15 * M)
    else:
        res_eval.W_rec_eff = res_train.W_rec

    X_te = res_eval.run(u_te, washout=WASHOUT)
    y_te_w = y_te[WASHOUT:]
    X_te_aug = np.hstack([X_te, np.ones((X_te.shape[0], 1))])
    y_pred = X_te_aug @ W_out
    return nrmse(y_te_w, y_pred)


def run_regime(regime: int):
    t0 = time.time()
    results = {"regime": regime, "n_seeds": N_SEEDS, "cells": {}}
    if regime == 0:
        nr = []
        for s in range(N_SEEDS):
            v = one_run(0, None, None, s)
            nr.append(v)
        results["cells"]["baseline"] = nr
        results["wall_s"] = time.time() - t0
        return results

    # transplant matrix
    train_hosts = ["ikaros", "daedalus"]
    eval_kinds = ["ikaros", "daedalus", "sw_matched", "shuffle", "ident_const"]

    for th in train_hosts:
        for ek in eval_kinds:
            cell_key = f"train_{th}__eval_{ek}"
            nrs = []
            for s in range(N_SEEDS):
                # build per-seed
                train_sub = build_substrate(th, None, seed=s + 1)
                if ek == "shuffle":
                    # Real same-device stream with permuted spatial dims:
                    # tests whether spatial *structure* is what carries identity.
                    eval_sub = PermutedSubstrate(train_sub, seed=s + 2)
                elif ek in ("ikaros", "daedalus"):
                    eval_sub = build_substrate(ek, None, seed=s + 3)
                else:
                    eval_sub = build_substrate(ek, train_sub, seed=s + 4)
                try:
                    val = one_run(regime, train_sub, eval_sub, s)
                except Exception as e:
                    val = float("nan")
                nrs.append(val)
            results["cells"][cell_key] = nrs
    results["wall_s"] = time.time() - t0
    return results


def main():
    summary = {}
    t_start = time.time()
    for regime in [0, 1, 2, 3, 4, 5]:
        print(f"[regime {regime}] starting", flush=True)
        res = run_regime(regime)
        with open(OUT / f"regime_{regime}_results.json", "w") as f:
            json.dump(res, f, indent=2)
        # quick mean print
        msg = []
        for k, v in res["cells"].items():
            arr = np.array(v, dtype=np.float64)
            msg.append(f"{k}: {np.nanmean(arr):.4f}±{np.nanstd(arr):.4f}")
        print(f"[regime {regime}] wall={res['wall_s']:.1f}s  " + " | ".join(msg[:6]), flush=True)
        summary[f"regime_{regime}_wall_s"] = res["wall_s"]
    summary["total_wall_s"] = time.time() - t_start
    summary["n_seeds"] = N_SEEDS
    with open(OUT / "_run_meta.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("DONE total wall =", summary["total_wall_s"])


if __name__ == "__main__":
    main()
