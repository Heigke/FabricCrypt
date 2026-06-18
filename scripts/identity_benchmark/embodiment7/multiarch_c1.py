"""Multi-architecture C1 A/B test: does the structure-hash effect (or lack thereof)
hold across model classes?

We test 3 architectures on the C1 self-prediction task:
  - Ridge reservoir (baseline; from abcd_ablation.py)
  - Small MLP (3 layers, ReLU)
  - Small GRU (1 layer)
For each: A (structure=chassi-hash) vs B (structure=random), 15 seeds each.

We do NOT re-test C and D — A/B is the structure-effect-only test.
"""
from __future__ import annotations
import json, sys, socket, time
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from abcd_ablation import (HASHES, hash_to_seed, load_data, make_windows,
                           normalize, nrmse, bootstrap_diff_ci)

HOST = socket.gethostname()
ROOT = Path("/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy")
OUT_DIR = ROOT / "results/IDENTITY_BENCHMARK_2026-05-30/embodiment7"

HIST = 100
HORIZON = 10
D = 5
N_SEEDS = 15
N_TRAIN = 600
N_TEST = 150


class MLP:
    def __init__(self, struct_seed, din, dout, hid=64):
        rng = np.random.default_rng(struct_seed)
        self.W1 = rng.standard_normal((din, hid)).astype(np.float32) * np.sqrt(2.0/din)
        self.b1 = np.zeros(hid, dtype=np.float32)
        self.W2 = rng.standard_normal((hid, hid)).astype(np.float32) * np.sqrt(2.0/hid)
        self.b2 = np.zeros(hid, dtype=np.float32)
        self.W3 = rng.standard_normal((hid, dout)).astype(np.float32) * np.sqrt(2.0/hid)
        self.b3 = np.zeros(dout, dtype=np.float32)

    def fwd(self, X):
        h1 = np.maximum(X @ self.W1 + self.b1, 0)
        h2 = np.maximum(h1 @ self.W2 + self.b2, 0)
        return h2 @ self.W3 + self.b3, (h1, h2)

    def fit(self, X, Y, epochs=120, lr=0.005, batch=64):
        Xf = X.reshape(len(X), -1).astype(np.float32)
        Yf = Y.reshape(len(Y), -1).astype(np.float32)
        # adjust shapes
        if self.W3.shape[1] != Yf.shape[1]:
            rng = np.random.default_rng(0)
            self.W3 = rng.standard_normal((self.W2.shape[1], Yf.shape[1])).astype(np.float32) * np.sqrt(2.0/self.W2.shape[1])
            self.b3 = np.zeros(Yf.shape[1], dtype=np.float32)
        if self.W1.shape[0] != Xf.shape[1]:
            rng = np.random.default_rng(0)
            self.W1 = rng.standard_normal((Xf.shape[1], self.W2.shape[0])).astype(np.float32) * np.sqrt(2.0/Xf.shape[1])
        rng2 = np.random.default_rng(0)
        for _ in range(epochs):
            idx = rng2.permutation(len(Xf))
            for i in range(0, len(Xf), batch):
                xb = Xf[idx[i:i+batch]]; yb = Yf[idx[i:i+batch]]
                y, (h1, h2) = self.fwd(xb)
                err = (y - yb) / len(xb)
                gW3 = h2.T @ err; gb3 = err.sum(0)
                gh2 = err @ self.W3.T * (h2 > 0)
                gW2 = h1.T @ gh2; gb2 = gh2.sum(0)
                gh1 = gh2 @ self.W2.T * (h1 > 0)
                gW1 = xb.T @ gh1; gb1 = gh1.sum(0)
                self.W3 -= lr * gW3; self.b3 -= lr * gb3
                self.W2 -= lr * gW2; self.b2 -= lr * gb2
                self.W1 -= lr * gW1; self.b1 -= lr * gb1

    def predict(self, X):
        Xf = X.reshape(len(X), -1).astype(np.float32)
        y, _ = self.fwd(Xf)
        return y


def run_arch_ab(arch_name: str, eval_host: str, n_seeds: int = N_SEEDS):
    train_data = load_data(eval_host)
    nrmse_A, nrmse_B = [], []
    for seed in range(n_seeds):
        Xtr, Ytr = make_windows(train_data, N_TRAIN, HIST, HORIZON, seed=seed)
        Xte, Yte = make_windows(train_data, N_TEST, HIST, HORIZON, seed=seed+9001)
        mu, sd = normalize(train_data)
        Xtr, Ytr, Xte, Yte_n = (Xtr-mu)/sd, (Ytr-mu)/sd, (Xte-mu)/sd, (Yte-mu)/sd

        # A: chassi-hash
        sA = hash_to_seed(HASHES[eval_host], salt=seed)
        # B: random
        sB = seed * 1009 + 7

        if arch_name == "mlp":
            mA = MLP(sA, HIST*D, HORIZON*D, hid=64); mA.fit(Xtr, Ytr)
            yA = mA.predict(Xte).reshape(-1, HORIZON, D)
            mB = MLP(sB, HIST*D, HORIZON*D, hid=64); mB.fit(Xtr, Ytr)
            yB = mB.predict(Xte).reshape(-1, HORIZON, D)
            nrmse_A.append(nrmse(Yte_n, yA))
            nrmse_B.append(nrmse(Yte_n, yB))

    return {"arch": arch_name, "eval_host": eval_host,
            "A": nrmse_A, "B": nrmse_B,
            "A_med": float(np.median(nrmse_A)), "B_med": float(np.median(nrmse_B)),
            "A_mean": float(np.mean(nrmse_A)), "B_mean": float(np.mean(nrmse_B)),
            "diff_pct": (np.mean(nrmse_B)-np.mean(nrmse_A))/max(np.mean(nrmse_B),1e-9)*100}


def main():
    t0 = time.time()
    out = {"host": HOST, "n_seeds": N_SEEDS, "results": []}
    for arch in ["mlp"]:
        for eh in ["ikaros", "daedalus"]:
            r = run_arch_ab(arch, eh)
            d, lo, hi = bootstrap_diff_ci(r["A"], r["B"])
            r["ab_diff_ci"] = [float(lo), float(hi)]
            r["ab_diff_mean"] = float(d)
            out["results"].append(r)
            print(f"  arch={arch} host={eh} A_med={r['A_med']:.4f} B_med={r['B_med']:.4f} "
                  f"effect_pct={r['diff_pct']:+.2f} CI=[{lo:+.4f},{hi:+.4f}]")
    out["runtime_s"] = time.time() - t0
    p = OUT_DIR / "multiarch_c1.json"
    p.write_text(json.dumps(out, indent=2, default=float))
    print(f"saved: {p}")

if __name__ == "__main__":
    main()
