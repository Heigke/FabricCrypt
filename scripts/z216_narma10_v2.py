"""Track T.1 v2 — NARMA-10 with FIXED reservoir harness.

Diagnosis from z215: NRMSE stuck at ~0.9 (trivial mean predictor).
Root causes:
  1. Same input fed to ALL cells via VG2 — no per-cell input diversity
  2. Recurrent gain (0.1) too weak — recurrent influence dominated by
     base_VG1 randomness
  3. No bias term in ridge readout
  4. Washout=100 may be short for 10-tap dependency
  5. Static surrogate eval has no internal cell dynamics — memory comes
     entirely from W·feat_prev recursion

Fixes in v2:
  1. Per-cell W_in (random gains) so each cell sees a different
     scaling of u(t) (and optionally u(t-9))
  2. Drive input through VG1 directly: VG1 = base + g_in*W_in*u
  3. Recurrence through VG2: VG2 = base + g_rec*(W·feat_prev)
  4. Stronger gains: g_in ≈ 0.5, g_rec ≈ 0.3 (was 0.3, 0.1)
  5. Bias column in ridge readout
  6. Washout=200, train=800, test=400 (was 100/700/400 effective)
  7. Optional leak: feat[t] = (1-α)·feat[t-1] + α·log_id (smoother memory)

If v2 still fails: the issue is fundamental — the surrogate's effective
nonlinearity is too saturating for ESN-style temporal computation.
That would mean NS-RAM dynamics need to be accessed differently
(transient/spike output instead of static |Id|).

5 variants × 5 seeds, N=200.
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
OUT = ROOT / "results/z216_narma10_v2"; OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ROOT))


VARIANTS = [
    ("baseline",      "ER_SPARSE",     0, 0.0),
    ("inh_r2",        "ER_SPARSE",     2, 0.3),
    ("ws_smallworld", "WS_SMALLWORLD", 0, 0.0),
    ("modular",       "MODULAR",       0, 0.0),
    ("rand_gauss",    "RAND_GAUSS",    0, 0.0),
]
SEEDS = list(range(5))
N = 200
T_TOTAL = 1500
T_WASHOUT = 200
T_TRAIN = 1000   # train = [washout, T_TRAIN], test = [T_TRAIN, T_TOTAL]
G_IN = 0.5       # input gain on VG1
G_REC = 0.3      # recurrent gain on VG2
LEAK = 0.5       # 0.0 = no leak (instant); 1.0 = full state replacement


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
    base_VG2 = rng.uniform(0.1, 0.4, size=N).astype(float)
    sign_mask = rng.choice([-1.0, 1.0], size=N).astype(float)
    # FIX 1: per-cell input mask (different gain per cell)
    W_in = rng.normal(0, 1.0, size=N)

    W_exc = build_topo(topo, N, rng)
    if inh_r > 0 and inh_s > 0:
        W = add_lateral_inhibition(W_exc, inh_r, inh_s, rng)
    else:
        W = W_exc.copy()

    u, y = gen_narma10(T_TOTAL, seed=seed)

    t0 = time.time()
    state = _np.zeros((N, T_TOTAL))
    feat_prev = _np.zeros(N)
    Vd_arr = _np.ones(N)
    for t in range(T_TOTAL):
        u_t = float(u[t])
        # FIX 2: drive input via VG1 with per-cell W_in
        VG1_t = _np.clip(base_VG1 + G_IN * W_in * u_t, 0.05, 0.7)
        # FIX 3: recurrence via VG2
        rec = (W @ feat_prev) * sign_mask
        VG2_t = _np.clip(base_VG2 + G_REC * rec, 0.0, 0.6)
        log_id = surr.eval(VG1_t, VG2_t, Vd_arr)
        # FIX 4: optional leak
        feat = (1.0 - LEAK) * feat_prev + LEAK * log_id
        state[:, t] = feat
        feat_prev = feat

    # FIX 5: bias term in readout
    X = state.T  # (T, N)
    X = _np.hstack([X, _np.ones((X.shape[0], 1))])  # add bias column
    target = y.copy()
    X_train = X[T_WASHOUT:T_TRAIN]
    y_train = target[T_WASHOUT:T_TRAIN]
    X_test = X[T_TRAIN:]
    y_test = target[T_TRAIN:]

    # Ridge with bias
    lam = 1e-4
    XtX = X_train.T @ X_train + lam * _np.eye(X.shape[1])
    Xty = X_train.T @ y_train
    w = _np.linalg.solve(XtX, Xty)
    y_pred = X_test @ w
    nrmse = float(_np.sqrt(((y_pred - y_test) ** 2).mean()) / y_test.std())

    # Train NRMSE for sanity (overfit check)
    y_pred_train = X_train @ w
    train_nrmse = float(_np.sqrt(((y_pred_train - y_train) ** 2).mean()) / y_train.std())

    return {
        "name": name, "topo": topo, "inh_r": inh_r, "inh_s": inh_s,
        "seed": seed, "N": N,
        "nrmse": nrmse, "train_nrmse": train_nrmse,
        "wall_s": time.time() - t0,
    }


def main():
    from scripts.util_safe_sweep import safe_sweep
    grid = [(name, topo, inh_r, inh_s, seed)
             for (name, topo, inh_r, inh_s) in VARIANTS
             for seed in SEEDS]
    print(f"[z216] {len(grid)} configs (5 variants × 5 seeds), N={N}")
    print(f"       g_in={G_IN}, g_rec={G_REC}, leak={LEAK}, T={T_TOTAL}")

    results = safe_sweep(
        run_fn=run_one,
        configs=grid,
        out_dir=OUT,
        config_key=config_key,
        max_workers=2,
        thermal_pause_c=75.0,
        thermal_kill_c=88.0,
        per_config_wall_cap_s=180.0,
    )

    agg = {}
    for r in results:
        agg.setdefault(r["name"], []).append(r)
    print(f"\n[z216] {len(results)} results — NARMA-10 NRMSE (lower = better):")
    print(f"  {'variant':<14} {'n':>3}  {'test mean±std':>15}  {'train mean':>11}  {'95% CI':>22}")
    for name, _, _, _ in VARIANTS:
        xs = agg.get(name, [])
        if not xs: continue
        nrmses = np.array([x["nrmse"] for x in xs])
        train_nrmses = np.array([x["train_nrmse"] for x in xs])
        n = len(nrmses)
        if n >= 2:
            ci = stats.t.interval(0.95, n-1, loc=nrmses.mean(), scale=stats.sem(nrmses))
        else:
            ci = (np.nan, np.nan)
        print(f"  {name:<14} {n:>3}  {nrmses.mean():.3f}±{nrmses.std():.3f}    "
              f"{train_nrmses.mean():.3f}    [{ci[0]:.3f}, {ci[1]:.3f}]")

    if "baseline" in agg and len(agg["baseline"]) >= 3:
        print(f"\nPaired t-test vs baseline (NRMSE; lower=better):")
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
            d = cand_arr - base_arr
            t, p = stats.ttest_rel(cand_arr, base_arr)
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
            direction = "BETTER" if d.mean() < 0 else "WORSE"
            print(f"  {name:<14} Δ={d.mean():+.3f}  ({direction})  t={t:+.2f}  p={p:.3g}  {sig}")

    summary = {"VARIANTS": VARIANTS, "SEEDS": SEEDS, "N": N,
                "G_IN": G_IN, "G_REC": G_REC, "LEAK": LEAK,
                "n_results": len(results)}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
