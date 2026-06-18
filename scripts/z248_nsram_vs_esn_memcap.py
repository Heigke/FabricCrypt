"""z248 — STEP B of NEXT_DIRECTION_PLAN: NS-RAM vs ESN on Memory Capacity (Jaeger 2002).

MC sum_k r^2(u[t], y_k[t]) where y_k[t] is a linear readout trained
to recover u[t-k]. MC measures how much information about k past
inputs the reservoir state preserves. Body-RC tau ~1ms (z244b) makes
NS-RAM a candidate at short k where physical memory could help.

Tasks: MC at delays k in {1,2,5,10,20,50,100}.
N=200, 5 seeds, same input projection, identical readouts.

PRE-REGISTERED gate: NS-RAM total MC (sum over k) > ESN total MC
with non-overlapping bootstrap 95% CIs at n=5 seeds.

Per-delay PASS reported separately: any k where NS-RAM strictly beats
ESN with non-overlapping CIs is a candidate niche.
"""
from __future__ import annotations
import os, sys, json, time
from pathlib import Path
import numpy as np

for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_k] = "1"

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
OUT = ROOT / "results/z248_nsram_vs_esn_memcap"; OUT.mkdir(parents=True, exist_ok=True)
SURR_PATH = ROOT / "results/z220_4d_dense/surrogate_4d_dense.npz"

from scripts.nsram_surrogate_4d import NSRAMSurrogate4D


def build_state_nsram(surr, u, N=200, seed=0, leak=0.30, g_VG2=0.20,
                        Cb=5e-15, dt=5e-7, g_VG1=0.30):
    rng = np.random.default_rng(seed)
    base_VG1 = rng.uniform(0.2, 0.5, N)
    base_VG2 = rng.uniform(0.05, 0.55, N)
    sign_mask = rng.choice([-1.0, 1.0], N).astype(np.float64)
    W_in = rng.normal(0, 1.0, N)
    n_block = 50; K = N // n_block
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
    u_input = (u - 0.5)
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
    return state


def build_state_esn(u, N=200, seed=0, leak=0.30, sr=0.9):
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
    return state


def mc_curve(state, u, delays, washout=200, train_ratio=0.7):
    T = state.shape[0]
    X = np.hstack([state, np.ones((T, 1))])
    n_train = int(washout + (T - washout) * train_ratio)
    Xt = X[washout:n_train]; Xv = X[n_train:]
    mcs = []
    for k in delays:
        if k >= n_train - washout:
            mcs.append(0.0); continue
        yt = u[washout-k:n_train-k]
        yv = u[n_train-k:T-k]
        if len(yv) < 50:
            mcs.append(0.0); continue
        Xt_k = Xt[:len(yt)]; Xv_k = Xv[:len(yv)]
        XtX = Xt_k.T @ Xt_k + 1e-4 * np.eye(X.shape[1])
        w = np.linalg.solve(XtX, Xt_k.T @ yt)
        pred = Xv_k @ w
        # r^2 between pred and yv (uncentered correlation squared, standard MC formulation)
        if pred.std() < 1e-12 or yv.std() < 1e-12:
            r2 = 0.0
        else:
            r2 = float(np.corrcoef(pred, yv)[0, 1] ** 2)
        mcs.append(r2)
    return np.array(mcs)


def boot_ci(arr, n=5000, seed=0):
    rng = np.random.default_rng(seed)
    boots = np.array([arr[rng.integers(0, len(arr), len(arr))].mean()
                        for _ in range(n)])
    return float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))


def main():
    print(f"=== z248 NS-RAM vs ESN Memory Capacity (Jaeger) ===", flush=True)
    surr = NSRAMSurrogate4D(SURR_PATH)
    delays = [1, 2, 5, 10, 20, 50, 100]
    seeds = [0, 1, 2, 3, 4]
    T = 2000
    nsram_mc_curves = []
    esn_mc_curves = []
    for s in seeds:
        rng = np.random.default_rng(s + 12345)
        u = rng.uniform(0.0, 1.0, T)
        t0 = time.time()
        sn = build_state_nsram(surr, u, N=200, seed=s)
        se = build_state_esn(u, N=200, seed=s)
        mn = mc_curve(sn, u, delays)
        me = mc_curve(se, u, delays)
        nsram_mc_curves.append(mn); esn_mc_curves.append(me)
        wall = time.time() - t0
        print(f"  seed={s}  NS-RAM MC={mn.sum():.3f}  ESN MC={me.sum():.3f}  "
              f"per-k NS-RAM={[f'{x:.2f}' for x in mn]}  ESN={[f'{x:.2f}' for x in me]}  "
              f"wall={wall:.1f}s", flush=True)

    NSR = np.array(nsram_mc_curves)
    ESN = np.array(esn_mc_curves)
    nsr_tot = NSR.sum(axis=1)
    esn_tot = ESN.sum(axis=1)
    nsr_ci = boot_ci(nsr_tot)
    esn_ci = boot_ci(esn_tot)
    nsram_total_wins = bool(nsr_ci[0] > esn_ci[1])
    esn_total_wins = bool(esn_ci[0] > nsr_ci[1])

    per_delay = {}
    for i, k in enumerate(delays):
        nsr_k = NSR[:, i]
        esn_k = ESN[:, i]
        nsr_kci = boot_ci(nsr_k)
        esn_kci = boot_ci(esn_k)
        per_delay[str(k)] = {
            "nsram_mean": float(nsr_k.mean()), "nsram_ci": list(nsr_kci),
            "esn_mean": float(esn_k.mean()), "esn_ci": list(esn_kci),
            "nsram_strictly_wins_at_k": bool(nsr_kci[0] > esn_kci[1]),
            "esn_strictly_wins_at_k": bool(esn_kci[0] > nsr_kci[1]),
        }
    n_k_nsram_wins = sum(1 for v in per_delay.values() if v["nsram_strictly_wins_at_k"])

    summary = {
        "delays": delays,
        "n_seeds": len(seeds),
        "nsram_total_mc_mean": float(nsr_tot.mean()),
        "nsram_total_mc_ci": list(nsr_ci),
        "esn_total_mc_mean": float(esn_tot.mean()),
        "esn_total_mc_ci": list(esn_ci),
        "nsram_total_wins": nsram_total_wins,
        "esn_total_wins": esn_total_wins,
        "per_delay": per_delay,
        "n_delays_nsram_wins": n_k_nsram_wins,
        "stepB_PASS": nsram_total_wins,
        "interpretation": (
            f"PASS — NS-RAM total MC strictly > ESN total MC ({nsr_tot.mean():.3f} "
            f"vs {esn_tot.mean():.3f}). Body-RC is genuine memory mechanism. "
            f"Replicate at n=30."
            if nsram_total_wins else
            f"FAIL — ESN dominates total MC ({esn_tot.mean():.3f} vs NS-RAM "
            f"{nsr_tot.mean():.3f}). Per-delay candidates: "
            f"{n_k_nsram_wins} of {len(delays)} k-values. "
            f"Continue STEP C if any niche found, else accept silicon-energy-only story."
        ),
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nTotal MC: NS-RAM {nsr_tot.mean():.3f} CI {nsr_ci}  "
          f"vs ESN {esn_tot.mean():.3f} CI {esn_ci}", flush=True)
    print(f"STEP B GATE: {'✅ PASS' if summary['stepB_PASS'] else '❌ FAIL'}", flush=True)
    print(f"Per-delay NS-RAM wins: {n_k_nsram_wins}/{len(delays)}", flush=True)
    print(summary["interpretation"], flush=True)


if __name__ == "__main__":
    main()
