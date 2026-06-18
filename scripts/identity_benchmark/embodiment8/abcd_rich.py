"""Embodiment Phase 8 — A/B/C/D ablation with rich dynamic features.

Reuses the chassi hashes from Phase 7 (ikaros / daedalus full-signature SHA-512
of the static asset bag) and the same A/B/C/D structure-vs-data factorial.

Inputs: results/.../embodiment8/{ikaros,daedalus}_features.npz produced by
dynamic_features.py.  We use the time-series features (~480-channel @10 Hz over
5 min ≈ 3000 samples) as input to:

    C1: self-prediction reservoir ridge (predict horizon=10 steps ahead)
    C2: self-anomaly tiny autoencoder (AUROC under 4 anomaly types)

Cells:
   A: chassi-hash struct for THIS host, OWN data
   B: random struct,                    OWN data
   C: chassi-hash struct for THIS host, OTHER data
   D: random struct,                    OTHER data

Gates (Phase 8 relaxed):
   embodiment effect A−B ≥ 5%
   CI for A−B excludes 0
   data effect A−C should be substantial
   A_strictly_best (≥1σ over max(B,C,D))

30 seeds, bootstrap 2000.
"""
from __future__ import annotations
import argparse, hashlib, json, socket, time
from pathlib import Path
import numpy as np

REPO = Path("/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy")
OUT_DIR = REPO / "results/IDENTITY_BENCHMARK_2026-05-30/embodiment8"

HASHES = {
    "ikaros":   "22410174476006cbe5c0243bf7a299e0db1c7cdd7662191a909ec9e3da2cfb481cc890b64410a2c5c0337b90d30c5464ede830530fdd95810853fed0a73c2f57",
    "daedalus": "ef8352b0d3c73cb11076caf640d55e671ac9bee0821654ab23b509ebbce05030d63a211d33730310a874ebe02f73106a249925a3d69db7f8b30ad53f69fd3e37",
}


def hash_to_seed(h, salt=0):
    return int.from_bytes(hashlib.sha256(f"{h}|{salt}".encode()).digest()[:4], "big")


def load_features(host):
    z = np.load(OUT_DIR / f"{host}_features.npz", allow_pickle=True)
    X = z["features_ts"].astype(np.float32)
    # drop all-zero columns (constant raw channels) — they leak no info but
    # blow up dimensionality
    keep = X.std(axis=0) > 1e-9
    X = X[:, keep]
    names = np.array(z["feature_names_ts"])[keep]
    # Standardise per-column (zscore) so reservoirs don't blow up
    mu = X.mean(axis=0, keepdims=True); sd = X.std(axis=0, keepdims=True) + 1e-6
    X = (X - mu) / sd
    return X.astype(np.float32), names


def make_windows(X, n, hist, horizon, seed):
    rng = np.random.default_rng(seed)
    T, D = X.shape
    if T - hist - horizon < n:
        # Sample with replacement when window pool is small
        starts = rng.integers(0, T - hist - horizon, n)
    else:
        starts = rng.choice(T - hist - horizon, n, replace=False)
    Xw = np.stack([X[s:s+hist] for s in starts])           # (n, hist, D)
    Yw = np.stack([X[s+hist:s+hist+horizon] for s in starts])  # (n, horizon, D)
    return Xw.astype(np.float32), Yw.astype(np.float32)


# ---------------------------------------------------------------------------
# Reservoir model for C1
# ---------------------------------------------------------------------------
RES_DIM = 256


class RidgeReservoir:
    def __init__(self, structure_seed, din, dout, res_dim=RES_DIM):
        rng = np.random.default_rng(structure_seed)
        self.W_in = (rng.standard_normal((din, res_dim)) / np.sqrt(din)).astype(np.float32)
        self.W_rec = (rng.standard_normal((res_dim, res_dim)) / np.sqrt(res_dim) * 0.9).astype(np.float32)
        self.bias = (rng.standard_normal(res_dim) * 0.1).astype(np.float32)
        self.res_dim = res_dim; self.din = din; self.dout = dout
        self.W_out = None; self.b_out = None

    def features(self, X):
        n, hist, _ = X.shape
        h = np.zeros((n, self.res_dim), dtype=np.float32)
        for t in range(hist):
            h = np.tanh(X[:, t, :] @ self.W_in + h @ self.W_rec + self.bias)
        return h

    def fit(self, X, Y, lam=1e-3):
        H = self.features(X)
        Yf = Y.reshape(len(X), -1)
        A = H.T @ H + lam * np.eye(self.res_dim, dtype=np.float32)
        B = H.T @ Yf
        W = np.linalg.solve(A, B)
        self.W_out = W
        self.b_out = Yf.mean(axis=0) - H.mean(axis=0) @ W

    def predict(self, X, horizon, dout):
        H = self.features(X)
        Yf = H @ self.W_out + self.b_out
        return Yf.reshape(-1, horizon, dout)


def nrmse(yt, yp):
    err = ((yt - yp) ** 2).mean(axis=(0, 1))
    var = yt.var(axis=(0, 1)) + 1e-8
    return float(np.sqrt(err / var).mean())


# ---------------------------------------------------------------------------
# Tiny autoencoder for C2
# ---------------------------------------------------------------------------
WIN_C2 = 30
HID_C2 = 32


class TinyAE:
    def __init__(self, din, hid, seed):
        rng = np.random.default_rng(seed)
        self.W1 = (rng.standard_normal((din, hid)) * np.sqrt(1.0/din)).astype(np.float32)
        self.b1 = np.zeros(hid, dtype=np.float32)
        self.W2 = (rng.standard_normal((hid, din)) * np.sqrt(1.0/hid)).astype(np.float32)
        self.b2 = np.zeros(din, dtype=np.float32)

    def forward(self, x):
        h = np.tanh(x @ self.W1 + self.b1)
        return h @ self.W2 + self.b2, h

    def fit(self, X, epochs=120, lr=0.01, batch=64):
        Xf = X.reshape(len(X), -1).astype(np.float32)
        rng = np.random.default_rng(0)
        for _ in range(epochs):
            idx = rng.permutation(len(Xf))
            for i in range(0, len(Xf), batch):
                xb = Xf[idx[i:i+batch]]
                y, h = self.forward(xb)
                err = y - xb
                gW2 = h.T @ err / len(xb); gb2 = err.mean(0)
                gh = err @ self.W2.T * (1 - h*h)
                gW1 = xb.T @ gh / len(xb); gb1 = gh.mean(0)
                self.W2 -= lr*gW2; self.b2 -= lr*gb2
                self.W1 -= lr*gW1; self.b1 -= lr*gb1

    def recon_err(self, X):
        Xf = X.reshape(len(X), -1).astype(np.float32)
        y, _ = self.forward(Xf)
        return ((y - Xf) ** 2).mean(axis=1)


def slice_windows(X, n, win, seed):
    rng = np.random.default_rng(seed)
    T, D = X.shape
    if T - win - 1 < n:
        starts = rng.integers(0, T - win - 1, n)
    else:
        starts = rng.choice(T - win - 1, n, replace=False)
    return np.stack([X[s:s+win] for s in starts])


def inject_spike(x, rng, ch, mag=4.0):
    x = x.copy()
    t = rng.integers(WIN_C2//4, 3*WIN_C2//4)
    dur = rng.integers(2, 5)
    x[t:t+dur, ch] += mag
    return x


def make_anomalies(X_normal, rng, channels_to_perturb):
    Xa = []
    for x in X_normal:
        ch = int(rng.choice(channels_to_perturb))
        mag = float(rng.choice([-4.0, -3.0, 3.0, 4.0]))
        Xa.append(inject_spike(x, rng, ch, mag))
    return np.stack(Xa)


def auroc(scores_neg, scores_pos):
    # higher score = more anomalous
    scores = np.concatenate([scores_neg, scores_pos])
    labels = np.concatenate([np.zeros(len(scores_neg)), np.ones(len(scores_pos))])
    order = np.argsort(-scores)
    labels = labels[order]
    P = labels.sum(); N = len(labels) - P
    if P == 0 or N == 0: return 0.5
    tpr = np.cumsum(labels) / P
    fpr = np.cumsum(1 - labels) / N
    return float(np.trapezoid(tpr, fpr))


# ---------------------------------------------------------------------------
# Bootstrap helpers
# ---------------------------------------------------------------------------
def bootstrap_diff_ci(a, b, n_boot=2000, ci=0.95, seed=1):
    rng = np.random.default_rng(seed)
    a = np.asarray(a); b = np.asarray(b)
    if len(a) == len(b):
        d = a - b
        boots = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(n_boot)])
    else:
        boots = np.array([a[rng.integers(0, len(a), len(a))].mean()
                          - b[rng.integers(0, len(b), len(b))].mean() for _ in range(n_boot)])
    boots.sort()
    return float(boots.mean()), float(np.quantile(boots, (1-ci)/2)), float(np.quantile(boots, 1-(1-ci)/2))


# ---------------------------------------------------------------------------
# C1: self-prediction
# ---------------------------------------------------------------------------
HIST_C1 = 20      # 20*0.1s = 2s history (at 10 Hz output)
HORIZON_C1 = 5    # 0.5s ahead
N_TRAIN_C1 = 300
N_TEST_C1 = 80
N_SEEDS = 30


def run_c1_cell(struct_host_or_random, data_host, eval_host, n_seeds=N_SEEDS):
    eval_X, _ = load_features(eval_host)
    train_X, _ = load_features(data_host)
    # match D between hosts: zero-pad / trim columns
    D = min(eval_X.shape[1], train_X.shape[1])
    eval_X = eval_X[:, :D]; train_X = train_X[:, :D]
    nrmses = []
    for seed in range(n_seeds):
        if struct_host_or_random == "random":
            ss = seed * 1009 + 7
        else:
            ss = hash_to_seed(HASHES[struct_host_or_random], salt=seed)
        Xtr, Ytr = make_windows(train_X, N_TRAIN_C1, HIST_C1, HORIZON_C1, seed=seed)
        Xte, Yte = make_windows(eval_X, N_TEST_C1, HIST_C1, HORIZON_C1, seed=seed + 9001)
        m = RidgeReservoir(ss, D, D)
        m.fit(Xtr, Ytr, lam=1e-2)
        Yp = m.predict(Xte, HORIZON_C1, D)
        nrmses.append(nrmse(Yte, Yp))
    return {
        "structure": struct_host_or_random,
        "data_host": data_host,
        "eval_host": eval_host,
        "nrmse_per_seed": nrmses,
        "median": float(np.median(nrmses)),
        "mean":   float(np.mean(nrmses)),
        "std":    float(np.std(nrmses)),
    }


def c1_run(eval_host):
    other = "daedalus" if eval_host == "ikaros" else "ikaros"
    cells = {
        "A": run_c1_cell(eval_host, eval_host, eval_host),
        "B": run_c1_cell("random",  eval_host, eval_host),
        "C": run_c1_cell(eval_host, other,     eval_host),
        "D": run_c1_cell("random",  other,     eval_host),
    }
    a, b, c, d = (cells[k]["nrmse_per_seed"] for k in "ABCD")
    am, bm, cm, dm = np.mean(a), np.mean(b), np.mean(c), np.mean(d)
    ab_d, ab_lo, ab_hi = bootstrap_diff_ci(a, b)
    ac_d, ac_lo, ac_hi = bootstrap_diff_ci(a, c)
    # NRMSE: lower is better -> structure effect: B - A (positive means A is better)
    structure_pct = 100.0 * (bm - am) / max(bm, 1e-9)
    data_pct      = 100.0 * (cm - am) / max(cm, 1e-9)
    asig = (max(bm, cm, dm) - am) / max(np.std(a), 1e-9)
    gates = {
        "embodiment_AB":      {"pct": structure_pct, "PASS": structure_pct >= 5.0 and ab_lo > 0, "ci_diff": [ab_lo, ab_hi]},
        "data_AC":            {"pct": data_pct,      "PASS": data_pct >= 5.0 and ac_lo > 0,      "ci_diff": [ac_lo, ac_hi]},
        "A_strictly_best":    {"sigma": float(asig), "PASS": asig >= 1.0},
    }
    return {"eval_host": eval_host, "cells": cells, "summary": {
        "A_mean": am, "B_mean": bm, "C_mean": cm, "D_mean": dm,
        "AB_diff": ab_d, "AC_diff": ac_d,
        "structure_pct": structure_pct, "data_pct": data_pct,
    }, "gates": gates}


# ---------------------------------------------------------------------------
# C2: self-anomaly
# ---------------------------------------------------------------------------
N_TRAIN_C2 = 300
N_TEST_C2  = 80


def run_c2_cell(struct_host_or_random, data_host, eval_host, n_seeds=N_SEEDS):
    eval_X, _ = load_features(eval_host)
    train_X, _ = load_features(data_host)
    D = min(eval_X.shape[1], train_X.shape[1])
    eval_X = eval_X[:, :D]; train_X = train_X[:, :D]
    aurocs = []
    # pick the 20 most-variable channels to perturb (more realistic anomalies)
    perturb_chans = np.argsort(-eval_X.std(axis=0))[:20]
    for seed in range(n_seeds):
        if struct_host_or_random == "random":
            ss = seed * 1009 + 7
        else:
            ss = hash_to_seed(HASHES[struct_host_or_random], salt=seed)
        Xtr = slice_windows(train_X, N_TRAIN_C2, WIN_C2, seed=seed)
        Xte_n = slice_windows(eval_X, N_TEST_C2, WIN_C2, seed=seed + 9001)
        rng = np.random.default_rng(seed + 5000)
        Xte_a = make_anomalies(Xte_n, rng, perturb_chans)
        ae = TinyAE(D * WIN_C2, HID_C2, ss)
        ae.fit(Xtr, epochs=80, lr=0.01)
        s_n = ae.recon_err(Xte_n)
        s_a = ae.recon_err(Xte_a)
        aurocs.append(auroc(s_n, s_a))
    return {
        "structure": struct_host_or_random,
        "data_host": data_host,
        "eval_host": eval_host,
        "auroc_per_seed": aurocs,
        "median": float(np.median(aurocs)),
        "mean":   float(np.mean(aurocs)),
        "std":    float(np.std(aurocs)),
    }


def c2_run(eval_host):
    other = "daedalus" if eval_host == "ikaros" else "ikaros"
    cells = {
        "A": run_c2_cell(eval_host, eval_host, eval_host),
        "B": run_c2_cell("random",  eval_host, eval_host),
        "C": run_c2_cell(eval_host, other,     eval_host),
        "D": run_c2_cell("random",  other,     eval_host),
    }
    a, b, c, d = (cells[k]["auroc_per_seed"] for k in "ABCD")
    am, bm, cm, dm = np.mean(a), np.mean(b), np.mean(c), np.mean(d)
    ab_d, ab_lo, ab_hi = bootstrap_diff_ci(a, b)  # AUROC: higher is better
    ac_d, ac_lo, ac_hi = bootstrap_diff_ci(a, c)
    structure_pct = 100.0 * (am - bm) / max(bm, 1e-9)
    data_pct      = 100.0 * (am - cm) / max(cm, 1e-9)
    asig = (am - max(bm, cm, dm)) / max(np.std(a), 1e-9)
    gates = {
        "embodiment_AB":      {"pct": structure_pct, "PASS": structure_pct >= 5.0 and ab_lo > 0, "ci_diff": [ab_lo, ab_hi]},
        "data_AC":            {"pct": data_pct,      "PASS": data_pct >= 5.0 and ac_lo > 0,      "ci_diff": [ac_lo, ac_hi]},
        "A_strictly_best":    {"sigma": float(asig), "PASS": asig >= 1.0},
    }
    return {"eval_host": eval_host, "cells": cells, "summary": {
        "A_mean": am, "B_mean": bm, "C_mean": cm, "D_mean": dm,
        "AB_diff": ab_d, "AC_diff": ac_d,
        "structure_pct": structure_pct, "data_pct": data_pct,
    }, "gates": gates}


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["c1", "c2", "both"], default="both")
    args = ap.parse_args()
    out = {"host_running": socket.gethostname(), "n_seeds": N_SEEDS,
           "hashes_used": HASHES, "by_eval_host": {}}
    t0 = time.time()
    if args.task in ("c1", "both"):
        out["c1"] = {eh: c1_run(eh) for eh in ("ikaros", "daedalus")}
        print("C1 done:", time.time() - t0, "s")
    if args.task in ("c2", "both"):
        out["c2"] = {eh: c2_run(eh) for eh in ("ikaros", "daedalus")}
        print("C2 done:", time.time() - t0, "s")
    out["runtime_s"] = time.time() - t0
    (OUT_DIR / "abcd_rich.json").write_text(json.dumps(out, indent=2, default=float))
    print("saved:", OUT_DIR / "abcd_rich.json", "runtime:", out["runtime_s"])

    # Print summary
    for task in ("c1", "c2"):
        if task not in out: continue
        print(f"\n===== {task.upper()} =====")
        for eh, r in out[task].items():
            s = r["summary"]; g = r["gates"]
            print(f"--- eval_host={eh} ---")
            print(f"  A={s['A_mean']:.4f}  B={s['B_mean']:.4f}  C={s['C_mean']:.4f}  D={s['D_mean']:.4f}")
            print(f"  struct% = {s['structure_pct']:+.2f}%   data% = {s['data_pct']:+.2f}%")
            for k, v in g.items():
                print(f"   {k:18s}  PASS={v.get('PASS', False)}  {v}")


if __name__ == "__main__":
    main()
