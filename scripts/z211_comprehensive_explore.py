"""z211 — Comprehensive architecture sweep: topology × inhibition × rule × seed.

One-shot exhaustive exploration that goes deeper than z200/z210:
  - 6 topologies × 4 inhibition variants × 3 rules × 3 seeds × 2 N
  = 432 configs at base task (MG-vs-sin 2-class classification)
  - Checkpointed: each config saves JSON as it completes
  - Resumable: skip configs whose JSON already exists
  - Thermal-aware: pause if APU > 75°C, resume at < 60°C
  - Per-subprocess thread cap (F.1 pattern)

Output:
  results/z211_comprehensive/<topo>_<inh>_<rule>_N<n>_s<seed>.json
  results/z211_comprehensive/summary.json (when all done)
"""
from __future__ import annotations
import os
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_k, "4")
import json, time, sys, subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results/z211_comprehensive"; OUT.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(ROOT))


# ---------------------------------------------------------------- #
# Sweep grid
# ---------------------------------------------------------------- #
TOPOS = ["ER_SPARSE", "WS_SMALLWORLD", "MODULAR", "SCALE_FREE",
         "HUB_SPOKE", "GRID_2D"]
# Inhibition variants: (radius, strength). r=0 means baseline.
INH_VARIANTS = [
    ("none",       0,   0.0),
    ("r1_s0.1",    1,   0.1),  # I.3 winner from z210
    ("r2_s0.3",    2,   0.3),
    ("r4_s0.3",    4,   0.3),
]
RULES = ["ff", "hebb_ip", "rhebb"]
N_SIZES = [256, 1024]
SEEDS = [0, 1, 2]


def config_key(topo, inh_name, rule, N, seed):
    return f"{topo}_{inh_name}_{rule}_N{N}_s{seed}"


def thermal_ok():
    """Quick non-blocking thermal check."""
    try:
        with open("/sys/class/thermal/thermal_zone0/temp") as f:
            T = int(f.read().strip()) / 1000.0
        return T < 75.0
    except Exception:
        return True  # if probe fails, assume OK (don't deadlock)


def run_config(args):
    """Per-subprocess execution. Self-skips if result exists."""
    # Per-subprocess thread cap
    import os as _os
    for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
               "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        _os.environ[_k] = "4"
    try:
        from threadpoolctl import threadpool_limits
        threadpool_limits(limits=4)
    except Exception:
        pass

    topo, inh_name, inh_r, inh_s, rule, N, seed = args
    key = config_key(topo, inh_name, rule, N, seed)
    fp = OUT / f"{key}.json"
    if fp.exists():
        try:
            return json.loads(fp.read_text())
        except Exception:
            pass  # corrupt; re-run

    sys.path.insert(0, str(ROOT))
    from scripts.z200_topo_rule_sweep import (
        build_topo, gen_signal, run_reservoir_surr,
    )
    from scripts.z210_lateral_inhibition import add_lateral_inhibition
    from scripts.nsram_surrogate import NSRAMSurrogate

    surr = NSRAMSurrogate.build_or_load(grid_size=(20, 20, 25))
    rng = np.random.default_rng(seed * 1000 + hash(topo) % 1000 + hash(rule) % 100)

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
                for i in np.where(err < 0)[0]:
                    W[i, :] += LR * (lid_pos[i, -1] * lid_pos[:, -1]
                                      - lid_neg[i, -1] * lid_neg[:, -1])
            elif rule == "hebb_ip":
                lid = run_reservoir_surr(surr, N, T, sig, true_sign,
                                          sign_mask, W, base_VG1, base_VG2)
                # Hebbian on last-step activity (with intrinsic plasticity to maintain mean)
                act = lid[:, -1]
                W += LR * np.outer(act, act)
                # IP: clip overall scale
                W = np.clip(W, -2.0, 2.0)
            elif rule == "rhebb":
                lid = run_reservoir_surr(surr, N, T, sig, true_sign,
                                          sign_mask, W, base_VG1, base_VG2)
                # Reward-modulated Hebbian
                act_mean = lid.mean(axis=1)
                W += LR * true_sign * np.outer(act_mean, act_mean)
            np.fill_diagonal(W, 0)
            if (epoch * N_TRAIN + s) % 16 == 15:
                eig = np.abs(np.linalg.eigvals(W)).max()
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

    res = {
        "topo": topo, "inh_name": inh_name, "inh_r": inh_r, "inh_s": inh_s,
        "rule": rule, "N": N, "seed": seed,
        "best_acc": best_acc, "final_acc": final_acc,
        "history": history, "wall_s": time.time() - t0,
    }
    fp.write_text(json.dumps(res, indent=2))
    return res


def main():
    grid = []
    for topo in TOPOS:
        for inh_name, inh_r, inh_s in INH_VARIANTS:
            for rule in RULES:
                for N in N_SIZES:
                    for seed in SEEDS:
                        grid.append((topo, inh_name, inh_r, inh_s, rule, N, seed))

    # Skip already-completed
    grid_pending = [g for g in grid
                    if not (OUT / f"{config_key(g[0], g[1], g[4], g[5], g[6])}.json").exists()]
    print(f"[z211] {len(grid)} total configs, {len(grid_pending)} pending "
          f"({len(grid)-len(grid_pending)} resumed)", flush=True)

    if not grid_pending:
        print("[z211] all configs already done; building summary only", flush=True)
    else:
        # 2026-05-07: 12 workers × 4 threads = 48 effective threads on 32-core
        # APU pushed temp to 96°C in 5 min. Reduce to 6 × 4 = 24 effective.
        n_workers = 6
        results = []
        t0 = time.time()
        with ProcessPoolExecutor(max_workers=n_workers) as ex:
            futs = {ex.submit(run_config, a): a for a in grid_pending}
            done_n = 0
            for f in as_completed(futs):
                if not thermal_ok():
                    print("[z211] APU > 75°C; pausing 60s", flush=True)
                    time.sleep(60)
                try:
                    r = f.result()
                    results.append(r)
                    done_n += 1
                    if done_n % 12 == 0 or done_n == len(grid_pending):
                        print(f"[z211] {done_n}/{len(grid_pending)} done; "
                              f"last: {r['topo']}/{r['inh_name']}/{r['rule']}/N{r['N']}/s{r['seed']} "
                              f"best={r['best_acc']:.3f}", flush=True)
                except Exception as e:
                    print(f"[z211] config failed: {e}", flush=True)
        print(f"\n[z211] {done_n} configs in {time.time()-t0:.0f}s", flush=True)

    # Build summary from all per-config JSONs
    all_results = []
    for fp in OUT.glob("*_*_*_N*_s*.json"):
        try:
            all_results.append(json.loads(fp.read_text()))
        except Exception:
            pass
    print(f"\n[z211] aggregate over {len(all_results)} configs:")
    # Aggregate by (topo, inh_name, rule, N) over seeds
    agg = {}
    for r in all_results:
        k = (r["topo"], r["inh_name"], r["rule"], r["N"])
        agg.setdefault(k, []).append(r)
    rows = []
    for (topo, inh, rule, N), xs in agg.items():
        bests = [x["best_acc"] for x in xs]
        finals = [x["final_acc"] for x in xs]
        rows.append({
            "topo": topo, "inh_name": inh, "rule": rule, "N": N, "n_seeds": len(xs),
            "best_mean": float(np.mean(bests)), "best_std": float(np.std(bests)),
            "final_mean": float(np.mean(finals)), "final_std": float(np.std(finals)),
        })
    rows.sort(key=lambda r: -r["best_mean"])
    print(f"\nTop 15 by best_mean (test acc, mean over seeds):")
    print(f"{'topo':<14} {'inh':<10} {'rule':<8} {'N':>4}  {'best±std':>14}  {'final±std':>14}  n")
    for r in rows[:15]:
        print(f"  {r['topo']:<12} {r['inh_name']:<10} {r['rule']:<8} {r['N']:>4}  "
              f"{r['best_mean']:.3f}±{r['best_std']:.3f}    "
              f"{r['final_mean']:.3f}±{r['final_std']:.3f}    {r['n_seeds']}")
    summary = {
        "TOPOS": TOPOS, "INH_VARIANTS": INH_VARIANTS, "RULES": RULES,
        "N_SIZES": N_SIZES, "SEEDS": SEEDS,
        "n_results": len(all_results),
        "agg": rows,
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\n[z211] summary written: {OUT}/summary.json")


if __name__ == "__main__":
    main()
