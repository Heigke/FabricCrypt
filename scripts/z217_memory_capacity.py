"""Track T diagnostic — Memory Capacity (MC) of NS-RAM reservoir.

The CLASSICAL reservoir-computing memory test. MC measures how many
past timesteps of input the reservoir can linearly reconstruct.

For each delay k, train a linear readout to predict u(t-k) from
reservoir state at time t. Capacity at delay k = squared Pearson
correlation r²(prediction, target). MC = Σ_k MC_k.

Theoretical max: MC ≤ N (one bit per neuron). Working ESN: 10-20% of N.
Static lookup with trivial recurrence: MC ≪ 1 (essentially no memory).

This is the defining test for whether our surrogate has ANY memory.
If MC < 2: confirmed lookup-table problem; need transient surrogate.
If MC > 10 at N=200: there's real memory; NARMA-10 failure was harness.

Single config, multiple seeds (variance check).
"""
from __future__ import annotations
import os
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_k] = "1"
import sys, json, time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results/z217_memory_capacity"; OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ROOT))

from scripts.z200_topo_rule_sweep import build_topo
from scripts.nsram_surrogate import NSRAMSurrogate

surr = NSRAMSurrogate.build_or_load(grid_size=(20, 20, 25))


def reservoir_run(u, N, W, base_VG1, base_VG2, sign_mask, W_in, g_in, g_rec, leak):
    T = len(u)
    state = np.zeros((N, T))
    feat = np.zeros(N)
    Vd = np.ones(N)
    for t in range(T):
        VG1 = np.clip(base_VG1 + g_in * W_in * u[t], 0.05, 0.7)
        rec = (W @ feat) * sign_mask
        VG2 = np.clip(base_VG2 + g_rec * rec, 0.0, 0.6)
        log_id = surr.eval(VG1, VG2, Vd)
        feat = (1-leak)*feat + leak*log_id
        state[:, t] = feat
    return state


def memory_capacity(N, seed, K_max=30, T=2000, washout=200, T_train=1500,
                     g_in=1.0, g_rec=0.6, leak=0.3, topo="ER_SPARSE"):
    """Train K_max linear readouts to predict u(t-k); return MC_k for each k."""
    rng = np.random.default_rng(seed)
    u = rng.uniform(-1.0, 1.0, T)   # standard MC test uses U[-1, 1]
    base_VG1 = rng.choice([0.2, 0.4, 0.6], N).astype(float)
    base_VG2 = rng.uniform(0.1, 0.4, N).astype(float)
    sign_mask = rng.choice([-1.0, 1.0], N).astype(float)
    W_in = rng.normal(0, 1.0, N)
    W = build_topo(topo, N, rng)

    state = reservoir_run(u, N, W, base_VG1, base_VG2, sign_mask, W_in,
                           g_in, g_rec, leak)
    X = state.T
    X = np.hstack([X, np.ones((X.shape[0], 1))])

    mc_per_k = []
    for k in range(1, K_max + 1):
        # target = u(t - k); valid range: t >= washout AND t-k >= 0
        valid = slice(max(washout, k), T_train)
        Xt = X[valid]
        target = u[valid.start - k : valid.stop - k]
        if len(target) < 50: break
        # ridge
        w = np.linalg.solve(Xt.T @ Xt + 1e-4 * np.eye(X.shape[1]),
                            Xt.T @ target)
        # test
        v_test = slice(T_train, T_train + (T - T_train))
        Xv = X[v_test]
        targ_v = u[v_test.start - k : v_test.stop - k]
        if len(targ_v) < 50: break
        pred = Xv @ w
        if pred.std() < 1e-9 or targ_v.std() < 1e-9:
            r2 = 0.0
        else:
            r = np.corrcoef(pred, targ_v)[0, 1]
            r2 = float(r * r)
        mc_per_k.append(r2)
    return np.array(mc_per_k)


def main():
    print(f"{'config':<22} {'MC_total':>10}  {'MC[1..5]':>20}  {'MC[20..30]':>20}")
    configs = [
        ("ER_SPARSE N=100 leak0.3", 100, "ER_SPARSE", 0.3),
        ("ER_SPARSE N=200 leak0.3", 200, "ER_SPARSE", 0.3),
        ("ER_SPARSE N=200 leak0.6", 200, "ER_SPARSE", 0.6),
        ("ER_SPARSE N=200 leak1.0", 200, "ER_SPARSE", 1.0),
        ("WS_SMALLWORLD N=200 leak0.3", 200, "WS_SMALLWORLD", 0.3),
        ("RAND_GAUSS N=200 leak0.3", 200, "RAND_GAUSS", 0.3),
    ]
    seeds = [0, 1, 2]
    out = []
    for label, N, topo, leak in configs:
        mcs = []
        full = []
        for s in seeds:
            mc = memory_capacity(N=N, seed=s, leak=leak, topo=topo)
            mcs.append(mc.sum())
            full.append(mc)
        full = np.stack(full).mean(axis=0)
        mc_mean = np.mean(mcs)
        mc_std = np.std(mcs)
        early = full[:5].sum()
        late = full[19:30].sum() if len(full) >= 30 else 0.0
        print(f"  {label:<22}  {mc_mean:>5.2f}±{mc_std:.2f}  "
              f"early={early:.2f}  late={late:.2f}")
        out.append({"label": label, "N": N, "topo": topo, "leak": leak,
                    "mc_mean": mc_mean, "mc_std": mc_std,
                    "early_5": float(early), "late_30": float(late),
                    "mc_per_k_avg": full.tolist()})

    print("\nDecision rule:")
    print("  MC < 2  → reservoir has essentially no memory (lookup-table problem)")
    print("  MC ≈ 5-10 at N=200 → modest memory, NARMA-10 likely too hard")
    print("  MC > 20 at N=200 → real memory; NARMA-10 harness was the issue")
    print("  Ideal ESN (theoretical) at N=200: MC ≈ 100-200")

    (OUT / "summary.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
