"""z223 — Tight CI replication + NARMA-N stretch test.

Phase 1: 30 seeds at z222-best (Cb=5fF, dt=500ns, N=200) on NARMA-10.
Goal: tight CI for defensible Mario claim.

Phase 2: same config on NARMA-{5, 10, 20, 30}.
Tests whether 5-step MC translates to N-tap NARMA performance.
Expected: NARMA-5 best, NARMA-10 ~0.62, NARMA-20 worse, NARMA-30 worst.
This characterizes the practical memory horizon.
"""
from __future__ import annotations
import os
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_k] = "1"
import sys, json, time
from pathlib import Path
import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
OUT = ROOT / "results/z223_replication"; OUT.mkdir(parents=True, exist_ok=True)
SURR_PATH = ROOT / "results/z220_4d_dense/surrogate_4d_dense.npz"


def gen_narma(T, seed, K=10):
    """NARMA-K: K-tap nonlinear autoregressive."""
    rng = np.random.default_rng(seed)
    u = rng.uniform(0, 0.5, T)
    y = np.zeros(T)
    for k in range(K, T-1):
        y[k+1] = (0.3 * y[k]
                   + 0.05 * y[k] * y[k-K+1:k+1].sum()
                   + 1.5 * u[k-K+1] * u[k]
                   + 0.1)
    return u, y


def config_key(args):
    K, seed = args
    return f"narma{K}_seed{seed}"


def run_one(args):
    import os as _os
    for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        _os.environ[_k] = "1"
    try:
        from threadpoolctl import threadpool_limits
        threadpool_limits(limits=1)
    except Exception: pass
    import sys as _sys
    _sys.path.insert(0, str(ROOT))
    import numpy as _np
    from scripts.nsram_surrogate_4d import NSRAMSurrogate4D
    from scripts.z221_vg2_input_v3 import reservoir_run_v3
    from scripts.z200_topo_rule_sweep import build_topo

    K, seed = args
    surr = NSRAMSurrogate4D(SURR_PATH)
    N = 200; Cb = 5e-15; dt = 5e-7  # z222-best
    rng = _np.random.default_rng(seed)
    base_VG1 = rng.uniform(0.2, 0.5, N).astype(float)
    base_VG2 = rng.uniform(0.05, 0.55, N).astype(float)
    sign_mask = rng.choice([-1.0, 1.0], N).astype(float)
    W_in = rng.normal(0, 1.0, N)
    W = build_topo("ER_SPARSE", N, rng)
    T_total = 1500; washout = 300; T_train = 1000
    u, y = gen_narma(T_total, seed=seed, K=K)
    u_input = (u - 0.25) / 0.25
    t0 = time.time()
    state = reservoir_run_v3(u_input, N, W, base_VG1, base_VG2, sign_mask,
                              W_in, surr, Cb, dt,
                              g_in_VG2=0.05, g_rec_VG1=0.3, leak=0.30)
    X = state.T
    X = _np.hstack([X, _np.ones((X.shape[0], 1))])
    Xt=X[washout:T_train]; yt=y[washout:T_train]
    Xv=X[T_train:]; yv=y[T_train:]
    w = _np.linalg.solve(Xt.T@Xt + 1e-4*_np.eye(X.shape[1]), Xt.T@yt)
    pred_t = Xt@w; pred_v = Xv@w
    train = float(_np.sqrt(((pred_t-yt)**2).mean()) / yt.std())
    test = float(_np.sqrt(((pred_v-yv)**2).mean()) / yv.std())
    return {"K": K, "seed": seed, "train_nrmse": train, "test_nrmse": test,
             "wall_s": time.time()-t0}


def main():
    from scripts.util_safe_sweep import safe_sweep

    grid = []
    # 30 seeds at NARMA-10
    for seed in range(30):
        grid.append((10, seed))
    # 10 seeds each at NARMA-5/20/30
    for K in [5, 20, 30]:
        for seed in range(10):
            grid.append((K, seed))

    print(f"[z223] {len(grid)} configs")
    results = safe_sweep(
        run_fn=run_one,
        configs=grid,
        out_dir=OUT,
        config_key=config_key,
        max_workers=2,
        thermal_pause_c=75.0,
        thermal_kill_c=88.0,
        per_config_wall_cap_s=60.0,
    )

    print(f"\n=== NARMA-K stretch (best z222 config Cb=5fF dt=500ns) ===")
    print(f"{'NARMA-K':>9}  {'n':>3}  {'NRMSE mean±std':>18}  {'95% CI':>22}")
    for K in [5, 10, 20, 30]:
        xs = [r for r in results if r["K"] == K]
        if not xs: continue
        arr = np.array([r["test_nrmse"] for r in xs])
        n = len(arr)
        ci = stats.t.interval(0.95, n-1, loc=arr.mean(), scale=stats.sem(arr)) if n >= 2 else (np.nan, np.nan)
        print(f"  NARMA-{K:<2}    {n:>3}  {arr.mean():.4f}±{arr.std():.4f}    "
              f"[{ci[0]:.4f}, {ci[1]:.4f}]")

    # Tight 30-seed CI for headline NARMA-10 number
    arr10 = np.array([r["test_nrmse"] for r in results if r["K"]==10])
    if len(arr10) >= 5:
        ci_lower = stats.t.interval(0.95, len(arr10)-1,
                                     loc=arr10.mean(), scale=stats.sem(arr10))
        # bootstrap CI as cross-check
        boot = []
        rng = np.random.default_rng(42)
        for _ in range(2000):
            sample = rng.choice(arr10, size=len(arr10), replace=True)
            boot.append(sample.mean())
        boot = np.array(boot)
        b_ci = (np.percentile(boot, 2.5), np.percentile(boot, 97.5))
        print(f"\n=== HEADLINE NARMA-10 (n={len(arr10)} seeds) ===")
        print(f"  Test NRMSE = {arr10.mean():.4f} ± {arr10.std():.4f}")
        print(f"  t-95% CI:    [{ci_lower[0]:.4f}, {ci_lower[1]:.4f}]")
        print(f"  Bootstrap95: [{b_ci[0]:.4f}, {b_ci[1]:.4f}]")
        print(f"  ESN gate (<0.6) {'PASSED' if ci_lower[1] < 0.6 else 'TOUCHED' if ci_lower[0] < 0.6 else 'NOT MET'}")
        print(f"  Baseline z216:  0.84")
        print(f"  Improvement:    {(0.84 - arr10.mean()):+.4f} ({(arr10.mean()/0.84-1)*100:.1f}%)")

    summary = {"n_results": len(results), "narma10_n": int(len(arr10)),
                "narma10_mean": float(arr10.mean()), "narma10_std": float(arr10.std()),
                "ci_t": list(ci_lower), "ci_boot": list(b_ci)}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
