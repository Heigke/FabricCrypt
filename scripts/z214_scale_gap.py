"""Track R.1 — scale-gap test (gemini's O29 killer omission).

Question: does the inhibition "gain" at N=256 (which z213 showed is
non-significant) appear larger at smaller N? If yes, we have evidence
the surrogate is artifact-prone at small N and the original z210/z211
"+28pp" findings were a small-N illusion.

Design:
  variants: baseline (ER_SPARSE+ff), inh_only (ER+r2_s0.3+ff)
  sizes:    N ∈ {64, 128, 256, 512}
  seeds:    5
  = 40 configs. Safe-wrapper ≤2 workers.
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
OUT = ROOT / "results/z214_scale_gap"; OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ROOT))


VARIANTS = [
    ("baseline", "ER_SPARSE", 0, 0.0, "ff"),
    ("inh_only", "ER_SPARSE", 2, 0.3, "ff"),
]
SIZES = [64, 128, 256, 512]
SEEDS = list(range(5))


def config_key(args):
    name, _, _, _, _, N, seed = args
    return f"{name}_N{N}_seed{seed}"


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
    from scripts.z200_topo_rule_sweep import (
        build_topo, gen_signal, run_reservoir_surr,
    )
    from scripts.z210_lateral_inhibition import add_lateral_inhibition
    from scripts.nsram_surrogate import NSRAMSurrogate

    name, topo, inh_r, inh_s, rule, N, seed = args
    surr = NSRAMSurrogate.build_or_load(grid_size=(20, 20, 25))
    rng = _np.random.default_rng(seed * 11 + N + hash(name) % 1000)

    base_VG1 = rng.choice([0.2, 0.4, 0.6], size=N).astype(float)
    base_VG2 = rng.uniform(0.0, 0.5, size=N).astype(float)
    sign_mask = rng.choice([-1.0, 1.0], size=N).astype(float)

    W_exc = build_topo(topo, N, rng)
    if inh_r > 0 and inh_s > 0:
        W = add_lateral_inhibition(W_exc, inh_r, inh_s, rng)
    else:
        W = W_exc.copy()

    EPOCHS = 12
    N_TRAIN = 16
    N_TEST = 24
    T = 60
    LR = 5e-3   # ff

    history = []
    t0 = time.time()
    best_acc = 0.0
    final_acc = 0.0

    for epoch in range(EPOCHS):
        for s in range(N_TRAIN):
            cls = int(rng.integers(0, 2))
            sig = gen_signal(cls, T, seed=epoch*1000+s)
            true_sign = +1.0 if cls == 0 else -1.0
            lid_pos = run_reservoir_surr(surr, N, T, sig, true_sign,
                                          sign_mask, W, base_VG1, base_VG2)
            lid_neg = run_reservoir_surr(surr, N, T, sig, -true_sign,
                                          sign_mask, W, base_VG1, base_VG2)
            a_pos = (lid_pos**2).mean(axis=1)
            a_neg = (lid_neg**2).mean(axis=1)
            err = a_pos - a_neg
            for i in _np.where(err < 0)[0]:
                W[i, :] += LR * (lid_pos[i, -1] * lid_pos[:, -1]
                                  - lid_neg[i, -1] * lid_neg[:, -1])
            _np.fill_diagonal(W, 0)
            if (epoch * N_TRAIN + s) % 16 == 15:
                eig = _np.abs(_np.linalg.eigvals(W)).max()
                if eig > 1e-9:
                    W = W * (0.9 / eig)
        correct = 0
        for s in range(N_TEST):
            cls = int(rng.integers(0, 2))
            sig = gen_signal(cls, T, seed=10000+epoch*100+s)
            scores = []
            for cand in (+1.0, -1.0):
                lid = run_reservoir_surr(surr, N, T, sig, cand,
                                          sign_mask, W, base_VG1, base_VG2)
                scores.append((lid**2).mean())
            pred = +1.0 if scores[0] > scores[1] else -1.0
            true_sign = +1.0 if cls == 0 else -1.0
            if pred == true_sign:
                correct += 1
        acc = correct / N_TEST
        history.append(acc)
        best_acc = max(best_acc, acc)
        final_acc = acc

    return {
        "name": name, "topo": topo, "inh_r": inh_r, "inh_s": inh_s,
        "rule": rule, "N": N, "seed": seed,
        "best_acc": best_acc, "final_acc": final_acc,
        "wall_s": time.time() - t0,
    }


def main():
    from scripts.util_safe_sweep import safe_sweep

    grid = [(name, topo, inh_r, inh_s, rule, N, seed)
             for (name, topo, inh_r, inh_s, rule) in VARIANTS
             for N in SIZES
             for seed in SEEDS]
    print(f"[z214] {len(grid)} configs (2 variants × 4 sizes × 5 seeds)")

    results = safe_sweep(
        run_fn=run_one,
        configs=grid,
        out_dir=OUT,
        config_key=config_key,
        max_workers=2,
        thermal_pause_c=75.0,
        thermal_kill_c=88.0,
        per_config_wall_cap_s=240.0,  # N=512 may be slow
    )
    print(f"\n[z214] {len(results)} results")

    # Aggregate by (name, N)
    agg = {}
    for r in results:
        agg.setdefault((r["name"], r["N"]), []).append(r)

    print(f"\nScale-gap analysis (best test acc, mean ± std):")
    print(f"  {'N':>4}  {'baseline':>14}  {'inh_only':>14}  {'Δ':>10}  {'p (paired)':>12}")
    for N in SIZES:
        bs = agg.get(("baseline", N), [])
        ihs = agg.get(("inh_only", N), [])
        if not bs or not ihs:
            print(f"  {N:>4}  --no data--")
            continue
        b_arr = np.array([x["best_acc"] for x in bs])
        i_arr = np.array([x["best_acc"] for x in ihs])
        # Paired by seed
        bs_by_seed = {x["seed"]: x["best_acc"] for x in bs}
        ihs_by_seed = {x["seed"]: x["best_acc"] for x in ihs}
        common_seeds = set(bs_by_seed) & set(ihs_by_seed)
        if len(common_seeds) >= 3:
            d = np.array([ihs_by_seed[s] - bs_by_seed[s] for s in common_seeds])
            t, p = stats.ttest_rel([ihs_by_seed[s] for s in common_seeds],
                                    [bs_by_seed[s] for s in common_seeds])
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
            print(f"  {N:>4}  {b_arr.mean():.3f}±{b_arr.std():.3f}  "
                  f"{i_arr.mean():.3f}±{i_arr.std():.3f}  {d.mean():+7.3f}  "
                  f"p={p:.3g} {sig}")
        else:
            print(f"  {N:>4}  {b_arr.mean():.3f}±{b_arr.std():.3f}  "
                  f"{i_arr.mean():.3f}±{i_arr.std():.3f}  (insufficient pairs)")

    summary = {"VARIANTS": VARIANTS, "SIZES": SIZES, "n_results": len(results)}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
