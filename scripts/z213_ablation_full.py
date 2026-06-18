"""Track V full: 20-seed ablation for gate-quality CI + paired t-test.

Extends z212 from 6 to 20 seeds. Same 6 variants. Uses the same
safe_sweep wrapper. Per O31 grok: 20 seeds gives >0.9 power at σ=0.1.

Resumes from z212's 36 configs already saved.
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
OUT = ROOT / "results/z213_ablation_full"; OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ROOT))

# Re-use z212's run_one (same 6-variant matrix)
from scripts.z212_ablation_pilot import VARIANTS, run_one, config_key
SEEDS = list(range(20))   # gate-quality 20 seeds


def main():
    from scripts.util_safe_sweep import safe_sweep

    grid = [(name, topo, inh_r, inh_s, rule, seed)
             for (name, topo, inh_r, inh_s, rule) in VARIANTS
             for seed in SEEDS]
    print(f"[z213] {len(grid)} configs (6 variants × 20 seeds), N=256")

    # First: copy any existing z212 pilot results into z213 dir for resume
    pilot_dir = ROOT / "results/z212_ablation_pilot"
    if pilot_dir.exists():
        n_copied = 0
        for fp in pilot_dir.glob("*_seed*.json"):
            target = OUT / fp.name
            if not target.exists():
                target.write_bytes(fp.read_bytes())
                n_copied += 1
        print(f"[z213] copied {n_copied} z212 results as warm start")

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
    print(f"\n[z213] {len(results)} results")

    # Aggregate per variant
    agg = {}
    for r in results:
        agg.setdefault(r["name"], []).append(r)

    print(f"\nFull ablation (20 seeds, best test acc):")
    print(f"  {'variant':<14} {'n':>3}  {'best mean±std':>15}  {'95% CI':>20}")
    for name, _, _, _, _ in VARIANTS:
        xs = agg.get(name, [])
        if not xs: continue
        bests = np.array([x["best_acc"] for x in xs])
        n = len(bests)
        if n >= 2:
            ci = stats.t.interval(0.95, n-1, loc=bests.mean(),
                                  scale=stats.sem(bests))
        else:
            ci = (np.nan, np.nan)
        print(f"  {name:<14} {n:>3}  {bests.mean():.3f}±{bests.std():.3f}    "
              f"[{ci[0]:.3f}, {ci[1]:.3f}]")

    # Paired t-test vs baseline
    if "baseline" in agg and len(agg["baseline"]) >= 5:
        print(f"\nPaired t-test (vs baseline, best_acc):")
        baseline_by_seed = {x["seed"]: x["best_acc"] for x in agg["baseline"]}
        for name, _, _, _, _ in VARIANTS:
            if name == "baseline": continue
            xs = agg.get(name, [])
            paired = [(baseline_by_seed.get(x["seed"]), x["best_acc"])
                      for x in xs if x["seed"] in baseline_by_seed]
            paired = [(b, c) for b, c in paired if b is not None]
            if len(paired) < 5:
                continue
            base_arr = np.array([p[0] for p in paired])
            cand_arr = np.array([p[1] for p in paired])
            d = cand_arr - base_arr
            t, p = stats.ttest_rel(cand_arr, base_arr)
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else " "
            print(f"  {name:<14} mean Δ = {d.mean():+.3f}±{d.std():.3f}  "
                  f"t={t:+.2f}  p={p:.3g}  {sig}  (n={len(paired)})")

    summary = {
        "VARIANTS": VARIANTS, "SEEDS": SEEDS, "n_results": len(results),
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
