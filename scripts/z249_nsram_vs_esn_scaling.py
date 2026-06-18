"""z249 — STEP C of NEXT_DIRECTION_PLAN: NS-RAM vs ESN at varied N.

Tests if NS-RAM's relative position vs ESN improves at smaller (or
larger) network sizes. So far 5 head-to-head benchmarks all at N=200
gave 4 ESN wins, 1 tie, 0 NS-RAM wins. STEP C tests if scale matters.

N ∈ {100, 200, 500, 1000} on NARMA-10. 5 seeds per cell.

PRE-REGISTERED gate per N: NS-RAM CI95 upper < ESN CI95 lower (strict
NS-RAM win, no overlap). Any N where NS-RAM strictly wins is a niche.
0/4 PASS = scale-independent ESN dominance, expected pattern.

NO-CHEAT discipline applied.
"""
from __future__ import annotations
import os, sys, json, time
from pathlib import Path
import numpy as np

for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_k] = "1"

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
OUT = ROOT / "results/z249_nsram_vs_esn_scaling"; OUT.mkdir(parents=True, exist_ok=True)
SURR_PATH = ROOT / "results/z220_4d_dense/surrogate_4d_dense.npz"

from scripts.nsram_surrogate_4d import NSRAMSurrogate4D


def gen_narma10(T, seed):
    rng = np.random.default_rng(seed)
    u = rng.uniform(0, 0.5, T)
    y = np.zeros(T)
    for k in range(10, T-1):
        y[k+1] = 0.3*y[k] + 0.05*y[k]*y[k-9:k+1].sum() + 1.5*u[k-9]*u[k] + 0.1
    return u, y


def run_nsram(surr, u, y, N=200, seed=0, leak=0.30, g_VG2=0.20,
                Cb=5e-15, dt=5e-7, g_VG1=0.30,
                washout=300, T_train=1000):
    rng = np.random.default_rng(seed)
    base_VG1 = rng.uniform(0.2, 0.5, N)
    base_VG2 = rng.uniform(0.05, 0.55, N)
    sign_mask = rng.choice([-1.0, 1.0], N).astype(np.float64)
    W_in = rng.normal(0, 1.0, N)
    n_block = min(50, N)
    K = N // n_block
    rng2 = np.random.default_rng(seed)
    Wb = np.zeros((K, n_block, n_block))
    for k in range(K):
        m = (rng2.random((n_block, n_block)) < 0.10).astype(np.float64)
        w = m * rng2.normal(0, 1, (n_block, n_block))
        np.fill_diagonal(w, 0)
        eig = np.abs(np.linalg.eigvals(w)).max()
        if eig > 1e-9: w *= 0.9 / eig
        Wb[k] = w
    T = len(u)
    u_input = (u - 0.25) / 0.25
    Vd_arr = np.ones(N)
    Vb = np.full(N, 0.30)
    feat = np.zeros(N)
    state = np.zeros((T, N))
    for t in range(T):
        VG2 = np.clip(base_VG2 + g_VG2 * W_in * u_input[t], 0.0, 0.55)
        feat_b = feat.reshape(K, n_block)
        rec_b = np.einsum("kij,kj->ki", Wb, feat_b)
        rec = rec_b.reshape(N) * sign_mask
        VG1 = np.clip(base_VG1 + g_VG1 * rec, 0.05, 0.7)
        log_Id, Iii, Ile = surr.eval(VG1, VG2, Vd_arr, Vb)
        Vb = np.clip(Vb + dt * (Iii - Ile) / Cb, 0.0, 0.7)
        feat = (1.0 - leak) * feat + leak * log_Id
        state[t] = feat
    X = np.hstack([state, np.ones((state.shape[0], 1))])
    Xt = X[washout:T_train]; yt = y[washout:T_train]
    Xv = X[T_train:];        yv = y[T_train:]
    XtX = Xt.T @ Xt + 1e-4 * np.eye(X.shape[1])
    w = np.linalg.solve(XtX, Xt.T @ yt)
    return float(np.sqrt(((Xv @ w - yv)**2).mean()) / yv.std())


def run_esn(u, y, N=200, seed=0, leak=0.30, sr=0.9,
              washout=300, T_train=1000):
    rng = np.random.default_rng(seed + 1000)
    W = (rng.random((N, N)) < 0.10) * rng.normal(0, 1, (N, N))
    np.fill_diagonal(W, 0)
    eig = np.abs(np.linalg.eigvals(W)).max()
    if eig > 1e-9: W *= sr / eig
    W_in = rng.normal(0, 1.0, N)
    T = len(u)
    s = np.zeros(N)
    state = np.zeros((T, N))
    for t in range(T):
        s = (1-leak)*s + leak*np.tanh(W @ s + 1.0 * W_in * u[t])
        state[t] = s
    X = np.hstack([state, np.ones((state.shape[0], 1))])
    Xt = X[washout:T_train]; yt = y[washout:T_train]
    Xv = X[T_train:];        yv = y[T_train:]
    XtX = Xt.T @ Xt + 1e-4 * np.eye(X.shape[1])
    w = np.linalg.solve(XtX, Xt.T @ yt)
    return float(np.sqrt(((Xv @ w - yv)**2).mean()) / yv.std())


def boot_ci(arr, n=5000, seed=0):
    rng = np.random.default_rng(seed)
    boots = np.array([arr[rng.integers(0, len(arr), len(arr))].mean()
                        for _ in range(n)])
    return float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))


def main():
    print(f"=== z249 NS-RAM vs ESN scaling (N sweep) ===", flush=True)
    surr = NSRAMSurrogate4D(SURR_PATH)
    Ns = [100, 200, 500, 1000]
    seeds = [0, 1, 2, 3, 4]
    T = 1500
    results = {}
    for N in Ns:
        nsr = []; esn = []
        for s in seeds:
            u, y = gen_narma10(T, s)
            t0 = time.time()
            r_n = run_nsram(surr, u, y, N=N, seed=s)
            r_e = run_esn(u, y, N=N, seed=s)
            wall = time.time() - t0
            nsr.append(r_n); esn.append(r_e)
            print(f"  N={N} seed={s} NS-RAM={r_n:.4f} ESN={r_e:.4f} "
                  f"Δ={r_n-r_e:+.4f} wall={wall:.1f}s", flush=True)
        nsr = np.array(nsr); esn = np.array(esn)
        nsr_ci = boot_ci(nsr)
        esn_ci = boot_ci(esn)
        nsram_wins = bool(nsr_ci[1] < esn_ci[0])
        esn_wins = bool(esn_ci[1] < nsr_ci[0])
        results[str(N)] = {
            "nsram_mean": float(nsr.mean()), "nsram_ci": list(nsr_ci),
            "esn_mean": float(esn.mean()), "esn_ci": list(esn_ci),
            "nsram_strictly_wins": nsram_wins,
            "esn_strictly_wins": esn_wins,
        }
        print(f"  N={N}: NS-RAM {nsr.mean():.4f} CI {nsr_ci}  "
              f"vs ESN {esn.mean():.4f} CI {esn_ci}  "
              f"NS-RAM wins:{nsram_wins} ESN wins:{esn_wins}", flush=True)

    n_nsram = sum(1 for v in results.values() if v["nsram_strictly_wins"])
    summary = {
        "Ns": Ns, "n_seeds": len(seeds), "results": results,
        "n_nsram_strict_wins": n_nsram,
        "stepC_PASS": n_nsram >= 1,
        "interpretation": (
            f"PASS — NS-RAM strictly beats ESN at {n_nsram}/{len(Ns)} N values. "
            f"Replicate at n=30 before brief headline."
            if n_nsram >= 1 else
            f"FAIL — ESN dominates across all tested N. Pattern is scale-"
            f"independent. After STEP A/B/C all FAIL, the NS-RAM vs ESN "
            f"matrix is effectively closed; Mario v4.3 is final."
        ),
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nSTEP C GATE: {'✅ PASS' if summary['stepC_PASS'] else '❌ FAIL'}",
          flush=True)
    print(summary["interpretation"], flush=True)


if __name__ == "__main__":
    main()
