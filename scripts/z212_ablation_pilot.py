"""Track V pilot: ablation matrix per O31b grok requirement.

6 variants × 6 seeds = 36 configs at N=256 (small, safe-thermal).
Tests whether the z210/z211 candidate's win comes from:
  - topology (ER_SPARSE vs RAND_GAUSS)
  - inhibition (with vs without)
  - rule (ff vs hebb_ip)
  - or the combination

Uses safe_sweep wrapper (≤2 workers, OMP=1, thermal pause).
"""
from __future__ import annotations
import os
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_k] = "1"
import sys
import json
import time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results/z212_ablation_pilot"; OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ROOT))


# 6-variant ablation matrix per O31b grok
VARIANTS = [
    # (name,           topo,           inh_r, inh_s, rule)
    ("baseline",       "ER_SPARSE",    0,     0.0,   "ff"),
    ("inh_only",       "ER_SPARSE",    2,     0.3,   "ff"),
    ("rule_only",      "ER_SPARSE",    0,     0.0,   "hebb_ip"),
    ("topo_only",      "RAND_GAUSS",   0,     0.0,   "ff"),
    ("inh_plus_rule",  "ER_SPARSE",    2,     0.3,   "hebb_ip"),
    ("candidate",      "ER_SPARSE",    2,     0.3,   "ff"),
]
SEEDS = list(range(6))   # 6 seeds (will extend to 20 once pilot is clean)
N = 256


def config_key(args):
    name, topo, inh_r, inh_s, rule, seed = args
    return f"{name}_seed{seed}"


def run_one(args):
    """Module-level so spawn-context can pickle. Hard env caps INSIDE."""
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

    name, topo, inh_r, inh_s, rule, seed = args
    surr = NSRAMSurrogate.build_or_load(grid_size=(20, 20, 25))
    rng = _np.random.default_rng(seed * 7 + hash(name) % 1000)

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
    LR = {"ff": 5e-3, "hebb_ip": 1e-3, "rhebb": 3e-3}[rule]

    history = []
    t0 = time.time()
    best_acc = 0.0
    final_acc = 0.0

    for epoch in range(EPOCHS):
        for s in range(N_TRAIN):
            cls = int(rng.integers(0, 2))
            sig = gen_signal(cls, T, seed=epoch*1000+s)
            true_sign = +1.0 if cls == 0 else -1.0
            if rule == "ff":
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
            elif rule == "hebb_ip":
                lid = run_reservoir_surr(surr, N, T, sig, true_sign,
                                          sign_mask, W, base_VG1, base_VG2)
                act = lid[:, -1]
                W += LR * _np.outer(act, act)
                W = _np.clip(W, -2.0, 2.0)
            _np.fill_diagonal(W, 0)
            if (epoch * N_TRAIN + s) % 16 == 15:
                eig = _np.abs(_np.linalg.eigvals(W)).max()
                if eig > 1e-9:
                    W = W * (0.9 / eig)
        # Eval
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
        "rule": rule, "seed": seed, "N": N,
        "best_acc": best_acc, "final_acc": final_acc,
        "history": history, "wall_s": time.time() - t0,
    }


def main():
    from scripts.util_safe_sweep import safe_sweep

    grid = [(name, topo, inh_r, inh_s, rule, seed)
             for (name, topo, inh_r, inh_s, rule) in VARIANTS
             for seed in SEEDS]
    print(f"[z212] {len(grid)} configs (6 variants × 6 seeds), N={N}")

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
    print(f"\n[z212] {len(results)} results")

    # Aggregate per variant
    agg = {}
    for r in results:
        agg.setdefault(r["name"], []).append(r)
    print(f"\nAblation results (best test acc, mean ± std over seeds):")
    print(f"  {'variant':<14} {'n':>3}  {'best mean±std':>15}  {'final mean±std':>15}")
    for name, _, _, _, _ in VARIANTS:
        xs = agg.get(name, [])
        if not xs:
            print(f"  {name:<14} {0:>3}  --no data--")
            continue
        bests = np.array([x["best_acc"] for x in xs])
        finals = np.array([x["final_acc"] for x in xs])
        print(f"  {name:<14} {len(xs):>3}  "
              f"{bests.mean():.3f}±{bests.std():.3f}    "
              f"{finals.mean():.3f}±{finals.std():.3f}")

    # Pairwise vs baseline (paired by seed)
    if "baseline" in agg and len(agg["baseline"]) >= 3:
        print(f"\nPaired delta vs baseline (best acc):")
        baseline_by_seed = {x["seed"]: x["best_acc"] for x in agg["baseline"]}
        for name, _, _, _, _ in VARIANTS:
            if name == "baseline": continue
            xs = agg.get(name, [])
            deltas = [x["best_acc"] - baseline_by_seed.get(x["seed"], None)
                      for x in xs if x["seed"] in baseline_by_seed]
            deltas = [d for d in deltas if d is not None]
            if deltas:
                d = np.array(deltas)
                wins = (d > 0).sum()
                print(f"  {name:<14} mean Δ = {d.mean():+.3f}  "
                      f"wins/n = {wins}/{len(d)}")

    summary = {
        "VARIANTS": VARIANTS, "SEEDS": SEEDS, "N": N,
        "n_results": len(results),
        "agg": {k: [r for r in v] for k, v in agg.items()},
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
