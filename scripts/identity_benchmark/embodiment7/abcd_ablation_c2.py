"""A/B/C/D ablation for C2 (self-anomaly autoencoder).

Same matrix as C1, but eval metric is AUROC on synthetic anomalies.
'Structure' here is the autoencoder W1/W2 init seed (chassi-hash-derived in A,C).
"""
from __future__ import annotations
import json, sys, socket, hashlib, time
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from abcd_ablation import HASHES, hash_to_seed, load_data, bootstrap_diff_ci

HOST = socket.gethostname()
ROOT = Path("/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy")
OUT_DIR = ROOT / "results/IDENTITY_BENCHMARK_2026-05-30/embodiment7"

WIN = 50
D = 5
HID = 16
N_TRAIN = 400
N_TEST_NORMAL = 100
N_TEST_ANOM = 100
N_SEEDS = 30


def inject_power_spike(x, rng, ch=2, mag=4.0):
    x = x.copy(); t = rng.integers(WIN//4, 3*WIN//4); dur = rng.integers(3, 8)
    x[t:t+dur, ch] += mag; return x

def inject_thermal_step(x, rng, ch=0, mag=3.0):
    x = x.copy(); t = rng.integers(WIN//4, WIN//2)
    x[t:, ch] += mag; return x

def inject_latency_burst(x, rng, ch=4, mag=5.0):
    x = x.copy(); t = rng.integers(WIN//4, 3*WIN//4); dur = rng.integers(2, 5)
    x[t:t+dur, ch] += mag; return x

def inject_freq_drop(x, rng, ch=3, mag=-3.0):
    x = x.copy(); t = rng.integers(WIN//4, 2*WIN//3); dur = rng.integers(4, 10)
    x[t:t+dur, ch] += mag; return x

ANOM_FNS = [inject_power_spike, inject_thermal_step, inject_latency_burst, inject_freq_drop]


def slice_windows(data, n, seed=0, hist=WIN):
    rng = np.random.default_rng(seed)
    T = len(data)
    starts = rng.choice(T - hist - 1, n, replace=False)
    return np.stack([data[s:s+hist] for s in starts])


class TinyAE:
    def __init__(self, din, hid, seed):
        rng = np.random.default_rng(seed)
        self.W1 = (rng.standard_normal((din, hid)) * np.sqrt(1.0 / din)).astype(np.float32)
        self.b1 = np.zeros(hid, dtype=np.float32)
        self.W2 = (rng.standard_normal((hid, din)) * np.sqrt(1.0 / hid)).astype(np.float32)
        self.b2 = np.zeros(din, dtype=np.float32)

    def forward(self, x):
        h = np.tanh(x @ self.W1 + self.b1)
        return h @ self.W2 + self.b2, h

    def fit(self, X, epochs=200, lr=0.01, batch=64):
        Xf = X.reshape(len(X), -1).astype(np.float32)
        rng = np.random.default_rng(0)
        for _ in range(epochs):
            idx = rng.permutation(len(Xf))
            for i in range(0, len(Xf), batch):
                xb = Xf[idx[i:i+batch]]
                y, h = self.forward(xb)
                err = y - xb
                gW2 = h.T @ err / len(xb)
                gb2 = err.mean(0)
                gh = err @ self.W2.T * (1 - h * h)
                gW1 = xb.T @ gh / len(xb)
                gb1 = gh.mean(0)
                self.W2 -= lr * gW2; self.b2 -= lr * gb2
                self.W1 -= lr * gW1; self.b1 -= lr * gb1

    def recon_err(self, X):
        Xf = X.reshape(len(X), -1).astype(np.float32)
        y, _ = self.forward(Xf)
        return ((y - Xf) ** 2).mean(axis=1)


def auroc(scores_pos, scores_neg):
    # pos = anomaly (higher score = more anomalous)
    y = np.concatenate([np.ones(len(scores_pos)), np.zeros(len(scores_neg))])
    s = np.concatenate([scores_pos, scores_neg])
    order = np.argsort(-s)
    y = y[order]
    tp = np.cumsum(y); fp = np.cumsum(1 - y)
    tp = np.concatenate([[0], tp]); fp = np.concatenate([[0], fp])
    tp /= tp[-1]; fp /= fp[-1]
    return float(np.trapezoid(tp, fp))


def run_cell_c2(structure_host_or_random: str, data_host: str, eval_host: str,
                n_seeds: int = N_SEEDS) -> dict:
    eval_data = load_data(eval_host)
    train_data = load_data(data_host)
    # Train-data normalization (applied to all)
    mu = train_data.mean(axis=0); sd = train_data.std(axis=0) + 1e-6
    train_n = (train_data - mu) / sd
    eval_n  = (eval_data  - mu) / sd

    aurocs = []
    for seed in range(n_seeds):
        # AE init seed
        if structure_host_or_random == "random":
            struct_seed = seed * 1009 + 7
        else:
            struct_seed = hash_to_seed(HASHES[structure_host_or_random], salt=seed)

        Xtr = slice_windows(train_n, N_TRAIN, seed=seed)
        ae = TinyAE(WIN * D, HID, seed=struct_seed)
        ae.fit(Xtr, epochs=150, lr=0.01, batch=64)

        X_norm = slice_windows(eval_n, N_TEST_NORMAL, seed=seed + 9001)
        X_anom_raw = slice_windows(eval_n, N_TEST_ANOM, seed=seed + 9002)
        rng = np.random.default_rng(seed + 7777)
        X_anom = np.stack([ANOM_FNS[rng.integers(0, 4)](x, rng) for x in X_anom_raw])

        s_norm = ae.recon_err(X_norm)
        s_anom = ae.recon_err(X_anom)
        aurocs.append(auroc(s_anom, s_norm))

    return {
        "structure": structure_host_or_random,
        "data_host": data_host,
        "eval_host": eval_host,
        "auroc_per_seed": aurocs,
        "median": float(np.median(aurocs)),
        "mean":   float(np.mean(aurocs)),
        "std":    float(np.std(aurocs)),
    }


def run_for_eval_host(eval_host: str) -> dict:
    other = "daedalus" if eval_host == "ikaros" else "ikaros"
    print(f"\n=== C2 A/B/C/D eval_host={eval_host} ===")
    cells = {}
    cells["A"] = run_cell_c2(eval_host, eval_host, eval_host)
    print(f"  A: AUROC={cells['A']['median']:.4f}")
    cells["B"] = run_cell_c2("random", eval_host, eval_host)
    print(f"  B: AUROC={cells['B']['median']:.4f}")
    cells["C"] = run_cell_c2(eval_host, other, eval_host)
    print(f"  C: AUROC={cells['C']['median']:.4f}")
    cells["D"] = run_cell_c2("random", other, eval_host)
    print(f"  D: AUROC={cells['D']['median']:.4f}")

    a, b, c, d = (cells[k]["auroc_per_seed"] for k in "ABCD")
    a_m, b_m, c_m, d_m = map(np.mean, (a, b, c, d))
    # For AUROC: higher is better; effect = A - other (in pp)
    ab_diff, ab_lo, ab_hi = bootstrap_diff_ci(a, b)
    ac_diff, ac_lo, ac_hi = bootstrap_diff_ci(a, c)
    cd_diff, cd_lo, cd_hi = bootstrap_diff_ci(c, d)

    structure_effect_pp = (a_m - b_m) * 100  # AUROC is in [0,1] → pp
    data_effect_pp = (a_m - c_m) * 100
    a_max_bcd = max(b_m, c_m, d_m)
    a_sigma_above_max = (a_m - a_max_bcd) / max(np.std(a), 1e-9)

    gates = {
        "embodiment_effect_AB":  {"effect_pp": structure_effect_pp, "PASS": structure_effect_pp >= 5.0,
                                  "ci": [ab_lo*100, ab_hi*100]},
        "data_effect_AC":        {"effect_pp": data_effect_pp, "PASS": data_effect_pp >= 10.0,
                                  "ci": [ac_lo*100, ac_hi*100]},
        "structure_alone_effect_CD": {"effect_pp": (c_m - d_m) * 100,
                                      "PASS": abs((c_m - d_m) * 100) <= 5.0,
                                      "ci": [cd_lo*100, cd_hi*100]},
        "A_strictly_best": {"sigma_above_max": float(a_sigma_above_max),
                            "PASS": a_sigma_above_max >= 1.0},
    }
    return {"eval_host": eval_host, "cells": cells, "gates": gates,
            "summary": {"A": a_m, "B": b_m, "C": c_m, "D": d_m,
                        "structure_pp": structure_effect_pp,
                        "data_pp": data_effect_pp}}


def main():
    t0 = time.time()
    result = {"host_running": HOST, "n_seeds": N_SEEDS, "by_eval_host": {}}
    for eh in ("ikaros", "daedalus"):
        result["by_eval_host"][eh] = run_for_eval_host(eh)
    result["runtime_s"] = time.time() - t0
    out = OUT_DIR / "abcd_ablation_c2.json"
    out.write_text(json.dumps(result, indent=2, default=float))
    print(f"\nsaved: {out}")
    print(f"\n========= C2 SUMMARY =========")
    for eh, r in result["by_eval_host"].items():
        print(f"\n--- eval_host={eh} ---")
        for k, cl in r["cells"].items():
            print(f"  {k}: structure={cl['structure']:<8} data={cl['data_host']:<8} AUROC={cl['median']:.4f} (±{cl['std']:.4f})")
        for gk, gv in r["gates"].items():
            print(f"  GATE {gk}: PASS={gv['PASS']}  {gv.get('effect_pp', gv.get('sigma_above_max'))}")


if __name__ == "__main__":
    main()
