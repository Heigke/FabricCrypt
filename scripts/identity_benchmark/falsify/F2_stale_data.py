#!/usr/bin/env python3
"""F2 — Stale-data ablation falsifier.

Compares z computed on:
  STALE: original A3_streams_{host}.npz (collected earlier — minutes-to-hours old)
  FRESH: re-collected streams right now via A3_heavy_tail_collect.collect_*

If z stays high on STALE → identity is silicon-stable across time.
If z drops on STALE → identity is "this moment's process state" (workload mix).

We re-collect ONLY ikaros (we have access). Daedalus stale is reused as-is for
both conditions.

Output: results/.../falsify/F2_stale_data.json
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
from A3_heavy_tail_collect import (  # type: ignore
    collect_syscall_jitter, collect_loop_jitter, collect_tsc_drift,
    read_apu_temp, wait_cool,
)

OUT_DIR = HERE.parents[2] / "results" / "IDENTITY_BENCHMARK_2026-05-30" / "falsify"
OUT_DIR.mkdir(parents=True, exist_ok=True)

N_SEEDS = int(os.environ.get("N_SEEDS", "30"))
LAM = 1.0


def collect_fresh_ikaros() -> dict:
    print(f"[F2.fresh] apu={read_apu_temp():.1f}", flush=True)
    if read_apu_temp() > 65.0:
        wait_cool(target=55.0, timeout=90.0)
    print("[F2.fresh] ch_syscall_jitter…", flush=True)
    sj = collect_syscall_jitter(n_samples=80000)
    if read_apu_temp() > 65.0:
        wait_cool(target=55.0, timeout=60.0)
    print("[F2.fresh] ch_loop_jitter…", flush=True)
    lj = collect_loop_jitter(n_samples=30000, loop_iters=2000)
    if read_apu_temp() > 65.0:
        wait_cool(target=55.0, timeout=60.0)
    print("[F2.fresh] ch_tsc_drift…", flush=True)
    td = collect_tsc_drift(n_samples=4000, interval_s=0.010)
    return {"ch_syscall_jitter": sj, "ch_loop_jitter": lj, "ch_tsc_drift": td}


def run_pipeline(streams_i, streams_d, label: str) -> dict:
    print(f"[F2.{label}] running pipeline with {label} streams…", flush=True)
    eval_kinds = ["self", "daedalus", "sw_matched_ht", "shuffle_ht"]
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
            cells[ek].append(float(task_nrmse(W_task, X_te, y_te_w)))
    agg = {ek: {"mean": float(np.nanmean(cells[ek])), "std": float(np.nanstd(cells[ek])),
                "n": int(np.sum(~np.isnan(cells[ek])))} for ek in eval_kinds}
    d_hw = agg["daedalus"]["mean"] - agg["self"]["mean"]
    d_sw = agg["sw_matched_ht"]["mean"] - agg["self"]["mean"]
    pooled = float(np.sqrt(agg["daedalus"]["std"] ** 2 + agg["sw_matched_ht"]["std"] ** 2)) + 1e-12
    z = (d_hw - d_sw) / pooled
    return {"aggregate": agg, "delta_hw": float(d_hw), "delta_sw": float(d_sw),
            "z_hw_vs_sw": float(z)}


def main():
    t0 = time.time()
    print(f"[F2.stale_data] N_SEEDS={N_SEEDS}", flush=True)
    streams_d = load_streams("daedalus")
    streams_stale = load_streams("ikaros")
    # Per-stream-file mtime → "age"
    stale_meta_path = (HERE.parents[2] / "results" / "IDENTITY_BENCHMARK_2026-05-30" /
                       "attack_1_3" / "A3_streams_ikaros.npz")
    stale_age_s = time.time() - stale_meta_path.stat().st_mtime
    print(f"[F2.stale] ikaros stream age = {stale_age_s/60:.1f} min", flush=True)

    print("[F2.stale] running pipeline on STALE ikaros streams…", flush=True)
    stale_result = run_pipeline(streams_stale, streams_d, "stale")

    print("[F2.fresh] collecting fresh ikaros streams (~3 min)…", flush=True)
    streams_fresh = collect_fresh_ikaros()
    fresh_result = run_pipeline(streams_fresh, streams_d, "fresh")

    # delta-z
    dz = fresh_result["z_hw_vs_sw"] - stale_result["z_hw_vs_sw"]
    if abs(dz) < 1.0 and stale_result["z_hw_vs_sw"] > 2.0:
        verdict = "SILICON_STABLE"
    elif stale_result["z_hw_vs_sw"] < 1.5 and fresh_result["z_hw_vs_sw"] > 2.0:
        verdict = "MOMENT_BOUND"  # only fresh works → workload state, not silicon
    elif fresh_result["z_hw_vs_sw"] < 1.5 and stale_result["z_hw_vs_sw"] > 2.0:
        verdict = "FRESH_BROKE"  # collection drift on this machine
    elif fresh_result["z_hw_vs_sw"] < 1.5 and stale_result["z_hw_vs_sw"] < 1.5:
        verdict = "BOTH_COLLAPSED"
    else:
        verdict = "PARTIAL"

    out = {
        "test": "F2_stale_data",
        "config": {"n_seeds": N_SEEDS, "lambda": LAM, "stale_age_min": stale_age_s / 60},
        "stale": stale_result,
        "fresh": fresh_result,
        "delta_z_fresh_minus_stale": float(dz),
        "verdict": verdict,
        "interpretation": (
            "SILICON_STABLE: z holds across hours → device-bound. "
            "MOMENT_BOUND: only this-moment state works → workload artifact, "
            "not silicon. FRESH_BROKE: thermal/workload drift on collection. "
            "BOTH_COLLAPSED: base finding non-reproducible today."
        ),
        "wall_s": time.time() - t0,
    }
    out_path = OUT_DIR / "F2_stale_data.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"[F2] stale z={stale_result['z_hw_vs_sw']:.2f} "
          f"fresh z={fresh_result['z_hw_vs_sw']:.2f} Δz={dz:+.2f} → {verdict}",
          flush=True)
    print(f"[F2] saved → {out_path} wall={out['wall_s']:.1f}s", flush=True)


if __name__ == "__main__":
    main()
