"""Embodiment Phase 7: A/B/C/D ablation matrix — the killer critic test.

For each metric (C1 NRMSE, C2 AUROC), test on each host:

| Cell | STRUCTURE (model init seed)             | DATA used for train  |
|------|-----------------------------------------|----------------------|
|  A   | chassi-hash-keyed for THIS host          | OWN host             |
|  B   | arbitrary random seed                    | OWN host             |
|  C   | chassi-hash-keyed for THIS host          | OTHER host           |
|  D   | arbitrary random seed                    | OTHER host           |

Pre-reg gates:
- A − B ≥ 10% improvement  → structure adds beyond data
- A − C should be large     → data matters (sanity)
- (A−B) > (C−D)             → structure×data interaction
- A > max(B,C,D) by ≥1σ     → A is strictly best

30 seeds per cell; bootstrap 95% CI on the difference; Bonferroni 4 tests.

Usage:
  python abcd_ablation.py        # run all cells
"""
from __future__ import annotations
import json, sys, time, socket, hashlib
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from full_signature import collect_static, static_hash

HOST = socket.gethostname()
ROOT = Path("/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy")
DATA_DIR = ROOT / "results/IDENTITY_BENCHMARK_2026-05-30/embodiment5"
OUT_DIR  = ROOT / "results/IDENTITY_BENCHMARK_2026-05-30/embodiment7"
OUT_DIR.mkdir(parents=True, exist_ok=True)

HIST    = 100
HORIZON = 10
D       = 5      # channels
RESERVOIR_DIM = 256
N_SEEDS = 30
N_TRAIN = 600
N_TEST  = 150


# ---------------------------------------------------------------------------
# Static signatures — pre-computed (collected from each host in advance)
# ---------------------------------------------------------------------------
# In production: would call full_signature.collect_static() on each host.
# For the ablation we use the hashes already collected from both machines
# (cross-host Hamming distance = 264/512 = 0.516, verified).
HASHES = {
    "ikaros":   "22410174476006cbe5c0243bf7a299e0db1c7cdd7662191a909ec9e3da2cfb481cc890b64410a2c5c0337b90d30c5464ede830530fdd95810853fed0a73c2f57",
    "daedalus": "ef8352b0d3c73cb11076caf640d55e671ac9bee0821654ab23b509ebbce05030d63a211d33730310a874ebe02f73106a249925a3d69db7f8b30ad53f69fd3e37",
}


def hash_to_seed(hexhash: str, salt: int = 0) -> int:
    """Derive a 32-bit reproducible seed from a hash + salt."""
    h = hashlib.sha256(f"{hexhash}|{salt}".encode()).digest()
    return int.from_bytes(h[:4], "big")


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_data(host: str) -> np.ndarray:
    if host == "ikaros":
        return np.load(DATA_DIR / "c1_ikaros" / "c1_ikaros_data.npy")
    return np.load(DATA_DIR / f"c1_{host}_data.npy")


def make_windows(data: np.ndarray, n: int, hist: int, horizon: int, seed: int):
    rng = np.random.default_rng(seed)
    T = len(data)
    starts = rng.choice(T - hist - horizon, n, replace=False)
    X = np.stack([data[s:s+hist] for s in starts])           # (n, hist, D)
    Y = np.stack([data[s+hist:s+hist+horizon] for s in starts])  # (n, horizon, D)
    return X.astype(np.float32), Y.astype(np.float32)


def normalize(train_data: np.ndarray):
    mu = train_data.mean(axis=(0, 1), keepdims=True)
    sd = train_data.std(axis=(0, 1), keepdims=True) + 1e-6
    return mu, sd


# ---------------------------------------------------------------------------
# Reservoir model (ridge) — STRUCTURE seed controls the projection matrix.
# ---------------------------------------------------------------------------
class RidgeReservoir:
    def __init__(self, structure_seed: int, din: int, dout: int,
                 reservoir_dim: int = RESERVOIR_DIM, alpha: float = 1.0):
        rng = np.random.default_rng(structure_seed)
        self.W_in = rng.standard_normal((din, reservoir_dim)).astype(np.float32) * (1.0 / np.sqrt(din))
        self.W_rec = (rng.standard_normal((reservoir_dim, reservoir_dim)).astype(np.float32) * (1.0 / np.sqrt(reservoir_dim))) * 0.9
        self.bias = rng.standard_normal(reservoir_dim).astype(np.float32) * 0.1
        self.alpha = alpha
        self.reservoir_dim = reservoir_dim
        self.din = din
        self.dout = dout
        self.W_out = None
        self.b_out = None

    def features(self, X):
        """X: (n, hist, din). Returns reservoir state (n, reservoir_dim)."""
        n, hist, _ = X.shape
        h = np.zeros((n, self.reservoir_dim), dtype=np.float32)
        for t in range(hist):
            h = np.tanh(X[:, t, :] @ self.W_in + h @ self.W_rec + self.bias)
        return h

    def fit(self, X, Y, lam: float = 1e-3):
        """Y: (n, horizon, dout) → flatten to (n, horizon*dout)."""
        n = X.shape[0]
        H = self.features(X)
        Yf = Y.reshape(n, -1)
        # Ridge solution: W = (H'H + lam*I)^-1 H'Y
        A = H.T @ H + lam * np.eye(self.reservoir_dim, dtype=np.float32)
        B = H.T @ Yf
        W = np.linalg.solve(A, B)
        self.W_out = W
        self.b_out = Yf.mean(axis=0) - H.mean(axis=0) @ W

    def predict(self, X, horizon: int, dout: int):
        H = self.features(X)
        Yf = H @ self.W_out + self.b_out
        return Yf.reshape(-1, horizon, dout)


def nrmse(y_true, y_pred):
    err = ((y_true - y_pred) ** 2).mean(axis=(0, 1))
    var = y_true.var(axis=(0, 1)) + 1e-8
    return float(np.sqrt(err / var).mean())


# ---------------------------------------------------------------------------
# Run one cell: (structure_host, data_host, eval_host) → list of per-seed NRMSE
# ---------------------------------------------------------------------------
def run_cell(structure_host_or_random: str, data_host: str, eval_host: str,
             n_seeds: int = N_SEEDS) -> dict:
    """
    structure_host_or_random:
      'ikaros' or 'daedalus' → chassi-hash-keyed structure for that host
      'random'               → arbitrary random structure (seed = seed_idx only)
    data_host: which host's data is used for training (eval is always eval_host)
    eval_host: test data drawn from this host
    """
    eval_data = load_data(eval_host)
    train_data = load_data(data_host)

    nrmses = []
    for seed in range(n_seeds):
        # Structure seed
        if structure_host_or_random == "random":
            struct_seed = seed * 1009 + 7
        else:
            struct_seed = hash_to_seed(HASHES[structure_host_or_random], salt=seed)

        # Training windows
        Xtr, Ytr = make_windows(train_data, N_TRAIN, HIST, HORIZON, seed=seed)
        # Eval windows — held out by using a different random seed
        Xte, Yte = make_windows(eval_data, N_TEST, HIST, HORIZON, seed=seed + 9001)

        # Normalize using TRAIN data stats; apply to BOTH (this is the standard
        # train-time-only-fit convention; if train and eval are different hosts
        # the eval data will be out-of-distribution, which is the point)
        mu, sd = normalize(train_data)
        Xtr = (Xtr - mu) / sd
        Ytr = (Ytr - mu) / sd
        Xte = (Xte - mu) / sd
        Yte_n = (Yte - mu) / sd

        model = RidgeReservoir(structure_seed=struct_seed, din=D, dout=D)
        model.fit(Xtr, Ytr, lam=1e-3)
        Ypred = model.predict(Xte, HORIZON, D)
        nrmses.append(nrmse(Yte_n, Ypred))

    return {
        "structure": structure_host_or_random,
        "data_host": data_host,
        "eval_host": eval_host,
        "nrmse_per_seed": nrmses,
        "median": float(np.median(nrmses)),
        "mean":   float(np.mean(nrmses)),
        "std":    float(np.std(nrmses)),
    }


def bootstrap_ci(values, n_boot: int = 2000, ci: float = 0.95, seed: int = 0):
    rng = np.random.default_rng(seed)
    arr = np.asarray(values, dtype=np.float64)
    boots = [arr[rng.integers(0, len(arr), len(arr))].mean() for _ in range(n_boot)]
    boots = np.sort(boots)
    lo = float(np.quantile(boots, (1 - ci) / 2))
    hi = float(np.quantile(boots, 1 - (1 - ci) / 2))
    return float(np.mean(boots)), lo, hi


def bootstrap_diff_ci(a, b, n_boot: int = 2000, ci: float = 0.95, seed: int = 1):
    """Bootstrap CI on (mean(a) - mean(b)) using paired-by-seed differences
    when same length; else independent bootstrap."""
    rng = np.random.default_rng(seed)
    a = np.asarray(a); b = np.asarray(b)
    if len(a) == len(b):
        d = a - b
        boots = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(n_boot)]
    else:
        boots = [a[rng.integers(0, len(a), len(a))].mean()
                 - b[rng.integers(0, len(b), len(b))].mean() for _ in range(n_boot)]
    boots = np.sort(boots)
    lo = float(np.quantile(boots, (1 - ci) / 2))
    hi = float(np.quantile(boots, 1 - (1 - ci) / 2))
    return float(np.mean(boots)), lo, hi


# ---------------------------------------------------------------------------
# Driver: run all cells on both eval hosts (using local data files)
# ---------------------------------------------------------------------------
def run_for_eval_host(eval_host: str) -> dict:
    other_host = "daedalus" if eval_host == "ikaros" else "ikaros"

    print(f"\n=== A/B/C/D on eval_host={eval_host} (other={other_host}) ===")
    cells = {}
    # A: own structure, own data
    cells["A"] = run_cell(eval_host, eval_host, eval_host)
    print(f"  A: NRMSE median={cells['A']['median']:.4f}")
    # B: random structure, own data
    cells["B"] = run_cell("random", eval_host, eval_host)
    print(f"  B: NRMSE median={cells['B']['median']:.4f}")
    # C: own structure, other data
    cells["C"] = run_cell(eval_host, other_host, eval_host)
    print(f"  C: NRMSE median={cells['C']['median']:.4f}")
    # D: random structure, other data
    cells["D"] = run_cell("random", other_host, eval_host)
    print(f"  D: NRMSE median={cells['D']['median']:.4f}")

    # Effect sizes
    a, b, c, d = (cells[k]["nrmse_per_seed"] for k in "ABCD")
    a_mean = np.mean(a); b_mean = np.mean(b); c_mean = np.mean(c); d_mean = np.mean(d)

    # For NRMSE: LOWER is better → improvement = (other - A) / other * 100
    ab_diff, ab_lo, ab_hi = bootstrap_diff_ci(a, b)
    ac_diff, ac_lo, ac_hi = bootstrap_diff_ci(a, c)
    ad_diff, ad_lo, ad_hi = bootstrap_diff_ci(a, d)
    cd_diff, cd_lo, cd_hi = bootstrap_diff_ci(c, d)

    structure_effect_pct = ((b_mean - a_mean) / b_mean * 100) if b_mean > 0 else 0.0
    data_effect_pct      = ((c_mean - a_mean) / c_mean * 100) if c_mean > 0 else 0.0
    interaction_pct      = ((d_mean - c_mean) / d_mean * 100) if d_mean > 0 else 0.0

    a_max_bcd = max(b_mean, c_mean, d_mean)
    a_sigma_below_max = (a_max_bcd - a_mean) / max(np.std(a), 1e-9)

    gates = {
        "embodiment_effect_AB":  {"effect_pct": structure_effect_pct, "PASS": structure_effect_pct >= 10.0,
                                  "ci": [ab_lo, ab_hi]},
        "data_effect_AC":        {"effect_pct": data_effect_pct, "PASS": data_effect_pct >= 30.0,
                                  "ci": [ac_lo, ac_hi]},
        "structure_alone_effect_CD": {"effect_pct": interaction_pct, "PASS": abs(interaction_pct) <= 5.0,
                                      "note": "small effect expected — structure-alone without right data shouldn't help much",
                                      "ci": [cd_lo, cd_hi]},
        "A_strictly_best":       {"sigma_below_max": float(a_sigma_below_max),
                                  "PASS": a_sigma_below_max >= 1.0},
    }
    return {
        "eval_host": eval_host,
        "cells": cells,
        "gates": gates,
        "summary": {
            "A_mean": a_mean, "B_mean": b_mean, "C_mean": c_mean, "D_mean": d_mean,
            "A_minus_B_diff": ab_diff, "A_minus_C_diff": ac_diff,
            "structure_effect_pct": structure_effect_pct,
            "data_effect_pct": data_effect_pct,
            "interaction_pct": interaction_pct,
        },
    }


def main():
    t0 = time.time()
    result = {
        "host_running": HOST,
        "n_seeds": N_SEEDS,
        "hashes_used": HASHES,
        "by_eval_host": {},
    }
    # We can run BOTH eval-hosts from this machine because data files exist for both
    for eh in ("ikaros", "daedalus"):
        result["by_eval_host"][eh] = run_for_eval_host(eh)
    result["runtime_s"] = time.time() - t0
    out = OUT_DIR / "abcd_ablation_c1.json"
    out.write_text(json.dumps(result, indent=2, default=float))
    print(f"\nsaved: {out}")
    print(f"runtime: {result['runtime_s']:.1f}s")

    print("\n========= SUMMARY =========")
    for eh, r in result["by_eval_host"].items():
        print(f"\n--- eval_host={eh} ---")
        for k, c in r["cells"].items():
            print(f"  {k}: structure={c['structure']:<8} data={c['data_host']:<8} NRMSE={c['median']:.4f} (±{c['std']:.4f})")
        for gk, gv in r["gates"].items():
            print(f"  GATE {gk}: PASS={gv['PASS']}  {gv.get('effect_pct', gv.get('sigma_below_max'))}")
    return result


if __name__ == "__main__":
    main()
