"""Track T.1 — NARMA-10 regression on NS-RAM reservoir (pilot).

NARMA-10 is a 10-tap nonlinear AR task widely used for reservoir
benchmarking. It has known headroom (typical ESN NRMSE 0.4-0.6,
no model is at chance/perfect). If an architecture variant gives
a real improvement, it should register here even when MG-vs-sin
saturates.

Task definition:
    u(k) ~ U[0, 0.5]
    y(k+1) = 0.3*y(k) + 0.05*y(k)*sum_{i=0..9}(y(k-i))
              + 1.5*u(k-9)*u(k) + 0.1

Reservoir architecture: N cells with random VG1/VG2, surrogate-driven
dynamics. State = log10|Id| trace. Train ridge readout on N-feature
state at each timestep to predict y(k+1).

5 variants × 5 seeds = 25 configs.

Reports: NRMSE on held-out test split, paired by seed.
"""
from __future__ import annotations
import os
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_k] = "1"
import sys, json, time
from pathlib import Path
import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results/z215_narma10_pilot"; OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ROOT))


VARIANTS = [
    ("baseline",    "ER_SPARSE",    0, 0.0),
    ("inh_r2",      "ER_SPARSE",    2, 0.3),  # known null on MG; sanity
    ("ws_smallworld", "WS_SMALLWORLD", 0, 0.0),
    ("modular",     "MODULAR",      0, 0.0),
    ("rand_gauss",  "RAND_GAUSS",   0, 0.0),
]
SEEDS = list(range(5))
N = 200
T_TOTAL = 1200
T_TRAIN = 800
T_WASHOUT = 100


def gen_narma10(T, seed):
    rng = np.random.default_rng(seed)
    u = rng.uniform(0.0, 0.5, size=T)
    y = np.zeros(T)
    for k in range(10, T-1):
        y[k+1] = (0.3 * y[k]
                  + 0.05 * y[k] * y[k-9:k+1].sum()
                  + 1.5 * u[k-9] * u[k]
                  + 0.1)
    return u, y


def config_key(args):
    name, _, _, _, seed = args
    return f"{name}_seed{seed}"


def run_one(args):
    import os as _os
    for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
               "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        _os.environ[_k] = "1"
    try:
        from threadpoolctl import threadpool_limits
        threadpool_limits(limits=1)
    except Exception:
        pass
    import sys as _sys
    _sys.path.insert(0, str(ROOT))
    import numpy as _np
    from scripts.z200_topo_rule_sweep import build_topo
    from scripts.z210_lateral_inhibition import add_lateral_inhibition
    from scripts.nsram_surrogate import NSRAMSurrogate

    name, topo, inh_r, inh_s, seed = args
    surr = NSRAMSurrogate.build_or_load(grid_size=(20, 20, 25))
    rng = _np.random.default_rng(seed * 13 + hash(name) % 1000)

    base_VG1 = rng.choice([0.2, 0.4, 0.6], size=N).astype(float)
    base_VG2 = rng.uniform(0.0, 0.5, size=N).astype(float)
    sign_mask = rng.choice([-1.0, 1.0], size=N).astype(float)

    W_exc = build_topo(topo, N, rng)
    if inh_r > 0 and inh_s > 0:
        W = add_lateral_inhibition(W_exc, inh_r, inh_s, rng)
    else:
        W = W_exc.copy()

    # Generate task
    u, y = gen_narma10(T_TOTAL, seed=seed)

    # Run reservoir on input u, log10|Id| trace
    t0 = time.time()
    state = _np.zeros((N, T_TOTAL))
    feat_prev = _np.zeros(N)
    for t in range(T_TOTAL):
        # Cell input: shared u(t) plus weighted recurrence
        u_t = float(u[t])
        VG2_t = base_VG2 + 0.3 * u_t  # input encoding via VG2 perturbation
        VG2_t = _np.clip(VG2_t, 0.0, 0.6)
        # Recurrent input
        rec = W @ feat_prev * sign_mask
        VG1_t = _np.clip(base_VG1 + 0.1 * rec, 0.1, 0.7)
        # Surrogate eval at fixed Vd=1.0 across cells
        Vd_arr = _np.ones(N)
        log_id = surr.eval(VG1_t, VG2_t, Vd_arr)
        state[:, t] = log_id
        feat_prev = log_id

    # Train ridge readout on washout-skip + train; test on held-out
    X = state.T  # (T, N)
    target = y.copy()
    X_train = X[T_WASHOUT:T_TRAIN]
    y_train = target[T_WASHOUT:T_TRAIN]
    X_test = X[T_TRAIN:]
    y_test = target[T_TRAIN:]
    # Ridge: w = (X.T X + λI)^-1 X.T y
    lam = 1e-4
    XtX = X_train.T @ X_train + lam * _np.eye(N)
    Xty = X_train.T @ y_train
    w = _np.linalg.solve(XtX, Xty)
    y_pred = X_test @ w
    nrmse = float(_np.sqrt(((y_pred - y_test) ** 2).mean()) / y_test.std())

    return {
        "name": name, "topo": topo, "inh_r": inh_r, "inh_s": inh_s,
        "seed": seed, "N": N, "nrmse": nrmse,
        "wall_s": time.time() - t0,
    }


def main():
    from scripts.util_safe_sweep import safe_sweep
    grid = [(name, topo, inh_r, inh_s, seed)
             for (name, topo, inh_r, inh_s) in VARIANTS
             for seed in SEEDS]
    print(f"[z215] {len(grid)} configs (5 variants × 5 seeds), N={N}")

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
    print(f"\n[z215] {len(results)} results")

    agg = {}
    for r in results:
        agg.setdefault(r["name"], []).append(r)

    print(f"\nNARMA-10 NRMSE (lower is better):")
    print(f"  {'variant':<14} {'n':>3}  {'NRMSE mean±std':>18}  {'95% CI':>20}")
    for name, _, _, _ in VARIANTS:
        xs = agg.get(name, [])
        if not xs: continue
        nrmses = np.array([x["nrmse"] for x in xs])
        n = len(nrmses)
        if n >= 2:
            ci = stats.t.interval(0.95, n-1, loc=nrmses.mean(),
                                  scale=stats.sem(nrmses))
        else:
            ci = (np.nan, np.nan)
        print(f"  {name:<14} {n:>3}  {nrmses.mean():.3f}±{nrmses.std():.3f}  "
              f"[{ci[0]:.3f}, {ci[1]:.3f}]")

    # Paired tests vs baseline
    if "baseline" in agg and len(agg["baseline"]) >= 3:
        print(f"\nPaired t-test vs baseline (NRMSE; lower = better):")
        baseline_by_seed = {x["seed"]: x["nrmse"] for x in agg["baseline"]}
        for name, _, _, _ in VARIANTS:
            if name == "baseline": continue
            xs = agg.get(name, [])
            paired = [(baseline_by_seed.get(x["seed"]), x["nrmse"])
                      for x in xs if x["seed"] in baseline_by_seed]
            paired = [(b, c) for b, c in paired if b is not None]
            if len(paired) < 3: continue
            base_arr = np.array([p[0] for p in paired])
            cand_arr = np.array([p[1] for p in paired])
            d = cand_arr - base_arr   # negative = better
            t, p = stats.ttest_rel(cand_arr, base_arr)
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
            direction = "BETTER" if d.mean() < 0 else "WORSE"
            print(f"  {name:<14} mean Δ = {d.mean():+.3f}  ({direction} by {abs(d.mean()):.3f})  "
                  f"t={t:+.2f}  p={p:.3g}  {sig}  (n={len(paired)})")

    summary = {"VARIANTS": VARIANTS, "SEEDS": SEEDS, "N": N,
                "n_results": len(results)}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
