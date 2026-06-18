#!/usr/bin/env python3
"""F1 — Tails-only swap falsifier.

Hybrid substrate: ikaros spatial/IC/leak/wmod (host-deterministic structure) BUT
draws values from daedalus's heavy-tail pool.

If z(daedalus-real vs hybrid) drops near 0 (or even flips), the SHUFFLE-evading
binding was actually carried by the *spatial structure* (host-deterministic random
draw), not by the tail-distribution.
If z stays high (hybrid behaves like daedalus, not like ikaros), the tail
distribution is the substrate signal.

30 seeds, λ=1.0 (the regime that gave z=5.74).

Run:
    HSA_OVERRIDE_GFX_VERSION=11.0.0 venv/bin/python scripts/identity_benchmark/falsify/F1_tails_only_swap.py
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
from A13_cross import train_dual, task_nrmse, narma10  # type: ignore

OUT_DIR = HERE.parents[2] / "results" / "IDENTITY_BENCHMARK_2026-05-30" / "falsify"
OUT_DIR.mkdir(parents=True, exist_ok=True)

N_SEEDS = int(os.environ.get("N_SEEDS", "30"))
LAM = 1.0


class TailsSwapSubstrate(HeavyTailSubstrate):
    """ikaros spatial/IC/leak/wmod, daedalus tail pool."""

    def __init__(self, ikaros_streams, daedalus_streams, n_dim, seed):
        # Build ikaros first (sets self.host='ikaros' → spatial/IC/leak/wmod
        # deterministically ikaros-bound)
        super().__init__("ikaros", ikaros_streams, n_dim=n_dim, seed=seed)
        # Replace pools with daedalus
        self.pools = []
        for ch, x in daedalus_streams.items():
            x = np.asarray(x, dtype=np.float64)
            if x.size < 100:
                continue
            x = x[np.isfinite(x)]
            mu, sd = float(x.mean()), float(x.std() + 1e-12)
            self.pools.append((ch, (x - mu) / sd))
        self.host_label = "hybrid_ikspatial_dedtails"


def main():
    t0 = time.time()
    streams_i = load_streams("ikaros")
    streams_d = load_streams("daedalus")
    print(f"[F1.tails_swap] N_SEEDS={N_SEEDS} λ={LAM}", flush=True)

    eval_kinds = ["self", "daedalus", "hybrid", "sw_matched_ht", "shuffle_ht"]
    cells = {ek: [] for ek in eval_kinds}

    for s in range(N_SEEDS):
        sub_i = HeavyTailSubstrate("ikaros", streams_i, n_dim=SUB_DIM, seed=s + 11)
        sub_d = HeavyTailSubstrate("daedalus", streams_d, n_dim=SUB_DIM, seed=s + 22)
        sub_hybrid = TailsSwapSubstrate(streams_i, streams_d, n_dim=SUB_DIM, seed=s + 11)
        sub_sw = GaussianMatchedHT(sub_i, seed=s + 33)
        sub_sh = ShuffleHT(sub_i, seed=s + 44)

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
        W_task = train_dual(X_i, y_w_i, X_d, LAM)

        for ek, ev in [("self", sub_i), ("daedalus", sub_d), ("hybrid", sub_hybrid),
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

    agg = {}
    for ek in eval_kinds:
        arr = np.array(cells[ek], dtype=np.float64)
        agg[ek] = {"mean": float(np.nanmean(arr)), "std": float(np.nanstd(arr)),
                   "n": int((~np.isnan(arr)).sum())}
    d_hw = agg["daedalus"]["mean"] - agg["self"]["mean"]
    d_hybrid = agg["hybrid"]["mean"] - agg["self"]["mean"]
    d_sw = agg["sw_matched_ht"]["mean"] - agg["self"]["mean"]
    d_sh = agg["shuffle_ht"]["mean"] - agg["self"]["mean"]
    pooled_hw_sw = float(np.sqrt(agg["daedalus"]["std"] ** 2 + agg["sw_matched_ht"]["std"] ** 2)) + 1e-12
    pooled_hyb_self = float(np.sqrt(agg["hybrid"]["std"] ** 2 + agg["self"]["std"] ** 2)) + 1e-12
    z_hw_vs_sw = (d_hw - d_sw) / pooled_hw_sw
    # If hybrid behaves like daedalus → tail-bound (binding survives spatial swap)
    # If hybrid behaves like self  → spatial-bound (tail-swap doesn't matter)
    z_hybrid_vs_self = d_hybrid / pooled_hyb_self
    z_hybrid_vs_daedalus = (d_hybrid - d_hw) / pooled_hyb_self

    # Verdict
    if z_hw_vs_sw < 2.0:
        verdict = "BASE_Z_COLLAPSED"
    elif d_hybrid > 0.7 * d_hw:
        verdict = "TAIL_BOUND"  # hybrid behaves like daedalus → tail carries identity
    elif d_hybrid < 0.3 * d_hw:
        verdict = "SPATIAL_BOUND"  # hybrid behaves like ikaros → spatial structure is the actual signal
    else:
        verdict = "MIXED"

    out = {
        "test": "F1_tails_only_swap",
        "config": {"n_seeds": N_SEEDS, "lambda": LAM, "n_res": N_RES, "sub_dim": SUB_DIM},
        "aggregate": agg,
        "deltas": {"daedalus": float(d_hw), "hybrid": float(d_hybrid),
                   "sw_matched_ht": float(d_sw), "shuffle_ht": float(d_sh)},
        "z_hw_vs_sw": float(z_hw_vs_sw),
        "z_hybrid_vs_self": float(z_hybrid_vs_self),
        "z_hybrid_vs_daedalus": float(z_hybrid_vs_daedalus),
        "verdict": verdict,
        "interpretation": (
            "TAIL_BOUND: silicon-bound heavy-tail carries identity (hybrid≈daedalus). "
            "SPATIAL_BOUND: identity is in spatial-structure seed (hybrid≈self); "
            "tail distribution is irrelevant; SHUFFLE escapes by permuting spatial, "
            "but tails-swap reveals the true mechanism."
        ),
        "wall_s": time.time() - t0,
    }
    out_path = OUT_DIR / "F1_tails_only_swap.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"[F1] Δ_hw={d_hw:.3f} Δ_hyb={d_hybrid:.3f} Δ_sw={d_sw:.3f} "
          f"z_hw_vs_sw={z_hw_vs_sw:.2f} z_hyb_vs_self={z_hybrid_vs_self:.2f} "
          f"→ {verdict}", flush=True)
    print(f"[F1] saved → {out_path} wall={out['wall_s']:.1f}s", flush=True)


if __name__ == "__main__":
    main()
