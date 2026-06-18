"""z222 — gpt-5 O33 Δt-Cb sweep on NARMA-10 + N-scaling check.

Per gpt-5 in O33: "tiny Δt–Cb sweep to hit 10-lag timescales:
Δt ∈ {0.2, 0.5, 1.0} µs × Cb ∈ {2, 5} fF; 10 seeds each."

Also test N=400 at the winning config (grok scalability concern).

Frozen base config (z221): g_VG2=0.05, leak=0.30, base_VG2 ∈ [0.05,0.55],
base_VG1 ∈ [0.2, 0.5], topo=ER_SPARSE.
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
OUT = ROOT / "results/z222_narma_sweep"; OUT.mkdir(parents=True, exist_ok=True)
SURR_PATH = ROOT / "results/z220_4d_dense/surrogate_4d_dense.npz"


def gen_narma10(T, seed):
    rng = np.random.default_rng(seed)
    u = rng.uniform(0, 0.5, T)
    y = np.zeros(T)
    for k in range(10, T-1):
        y[k+1] = 0.3*y[k] + 0.05*y[k]*y[k-9:k+1].sum() + 1.5*u[k-9]*u[k] + 0.1
    return u, y


def config_key(args):
    Cb_fF, dt, N, seed = args
    return f"Cb{Cb_fF:g}_dt{dt:g}_N{N}_s{seed}"


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

    Cb_fF, dt, N, seed = args
    surr = NSRAMSurrogate4D(SURR_PATH)
    rng = _np.random.default_rng(seed)
    base_VG1 = rng.uniform(0.2, 0.5, N).astype(float)
    base_VG2 = rng.uniform(0.05, 0.55, N).astype(float)
    sign_mask = rng.choice([-1.0, 1.0], N).astype(float)
    W_in = rng.normal(0, 1.0, N)
    W = build_topo("ER_SPARSE", N, rng)
    T_total = 1500; washout=300; T_train=1000
    u, y = gen_narma10(T_total, seed=seed)
    u_input = (u - 0.25) / 0.25
    t0 = time.time()
    state = reservoir_run_v3(u_input, N, W, base_VG1, base_VG2, sign_mask,
                              W_in, surr, Cb_fF*1e-15, dt,
                              g_in_VG2=0.05, g_rec_VG1=0.3, leak=0.30)
    X = state.T
    X = _np.hstack([X, _np.ones((X.shape[0], 1))])
    Xt=X[washout:T_train]; yt=y[washout:T_train]
    Xv=X[T_train:]; yv=y[T_train:]
    w = _np.linalg.solve(Xt.T@Xt + 1e-4*_np.eye(X.shape[1]), Xt.T@yt)
    pred_t = Xt@w; pred_v = Xv@w
    train_nrmse = float(_np.sqrt(((pred_t-yt)**2).mean()) / yt.std())
    test_nrmse = float(_np.sqrt(((pred_v-yv)**2).mean()) / yv.std())
    return {
        "Cb_fF": Cb_fF, "dt": dt, "N": N, "seed": seed,
        "train_nrmse": train_nrmse, "test_nrmse": test_nrmse,
        "wall_s": time.time() - t0,
    }


def main():
    from scripts.util_safe_sweep import safe_sweep

    # Phase 1: Δt-Cb sweep at N=200
    grid = []
    for Cb_fF in [2.0, 5.0]:
        for dt in [2e-7, 5e-7, 1e-6]:
            for seed in range(10):
                grid.append((Cb_fF, dt, 200, seed))

    # Phase 2: N-scaling at the z221 winning config
    for N in [100, 400]:
        for seed in range(10):
            grid.append((5.0, 1e-6, N, seed))

    print(f"[z222] {len(grid)} configs (60 Δt-Cb at N=200 + 20 N-scaling)")

    results = safe_sweep(
        run_fn=run_one,
        configs=grid,
        out_dir=OUT,
        config_key=config_key,
        max_workers=2,
        thermal_pause_c=75.0,
        thermal_kill_c=88.0,
        per_config_wall_cap_s=120.0,
    )
    print(f"\n[z222] {len(results)} results")

    # Aggregate Δt-Cb
    print(f"\n=== Δt-Cb sweep at N=200 ===")
    print(f"{'Cb (fF)':>9}  {'dt (s)':>8}  {'NRMSE mean±std':>16}  {'95% CI':>22}")
    for Cb in [2.0, 5.0]:
        for dt in [2e-7, 5e-7, 1e-6]:
            xs = [r for r in results if r["Cb_fF"]==Cb and r["dt"]==dt and r["N"]==200]
            if not xs: continue
            arr = np.array([r["test_nrmse"] for r in xs])
            n = len(arr)
            ci = stats.t.interval(0.95, n-1, loc=arr.mean(), scale=stats.sem(arr)) if n>=2 else (np.nan, np.nan)
            print(f"  {Cb:>7.1f}  {dt:>8.0e}  {arr.mean():.4f}±{arr.std():.4f}  [{ci[0]:.4f}, {ci[1]:.4f}]")

    print(f"\n=== N-scaling at Cb=5fF, dt=1µs ===")
    print(f"{'N':>5}  {'NRMSE mean±std':>16}  {'95% CI':>22}")
    for N in [100, 200, 400]:
        xs = [r for r in results if r["Cb_fF"]==5.0 and r["dt"]==1e-6 and r["N"]==N]
        if not xs: continue
        arr = np.array([r["test_nrmse"] for r in xs])
        n = len(arr)
        ci = stats.t.interval(0.95, n-1, loc=arr.mean(), scale=stats.sem(arr)) if n>=2 else (np.nan, np.nan)
        print(f"  {N:>3}  {arr.mean():.4f}±{arr.std():.4f}  [{ci[0]:.4f}, {ci[1]:.4f}]")

    # Find best config
    by_dtcb = {}
    for r in results:
        if r["N"] != 200: continue
        k = (r["Cb_fF"], r["dt"])
        by_dtcb.setdefault(k, []).append(r["test_nrmse"])
    best_k, best_v = None, 1e9
    for k, vs in by_dtcb.items():
        m = np.mean(vs)
        if m < best_v:
            best_k, best_v = k, m
    print(f"\nBest Δt-Cb at N=200: Cb={best_k[0]}fF dt={best_k[1]:.0e} → NRMSE = {best_v:.4f}")
    print(f"z216 baseline: NRMSE 0.84")
    print(f"z221 frozen (5seed): NRMSE 0.7243")

    summary = {"n_results": len(results),
                "best_at_N200": list(best_k) + [best_v]}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
