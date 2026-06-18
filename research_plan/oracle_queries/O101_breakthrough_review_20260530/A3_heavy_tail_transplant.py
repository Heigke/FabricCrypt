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
