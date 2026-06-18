"""z210 — I.3 Lateral inhibition sweep on ER_SPARSE + Forward-Forward.

Tests whether adding an inhibitory ring on top of the existing
ER_SPARSE excitatory weights improves classification on the
2-class Mackey-Glass-vs-sin task (z200's setup).

Sweep inhibition radius r ∈ {0, 1, 2, 4, 8} cells; inhibition
strength s ∈ {0.1, 0.3, 0.6} fraction of excitatory mean weight.
3 seeds each = 5 × 3 × 3 = 45 configs at N=1024.

Per-subprocess thread cap to avoid 2026-05-05 thermal-trip pathology.
"""
from __future__ import annotations
import os
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_k, "4")
import json, time, sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results/z210_lateral_inhibition"; OUT.mkdir(parents=True, exist_ok=True)
FIG = ROOT / "figures/z210_lateral_inhibition"; FIG.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(ROOT))
from scripts.z200_topo_rule_sweep import (
    build_topo, run_config as _z200_run_config,
)


def add_lateral_inhibition(W: np.ndarray, radius: int, strength: float,
                            rng: np.random.Generator) -> np.ndarray:
    """Add inhibitory ring connections of given radius and strength.

    For each cell i, set W[i, (i±k) % N] -= strength * mean(|W_excitatory|)
    for k in 1..radius. Existing entries are SUPERIMPOSED (inhibition adds
    on top of any existing excitatory weight).
    """
    if radius <= 0 or strength <= 0:
        return W
    N = W.shape[0]
    pos = W[W > 0]
    base = float(pos.mean()) if pos.size > 0 else 0.5 / np.sqrt(N * 0.10)
    inh = strength * base
    Wnew = W.copy()
    for i in range(N):
        for k in range(1, radius + 1):
            Wnew[i, (i + k) % N] -= inh
            Wnew[i, (i - k) % N] -= inh
    np.fill_diagonal(Wnew, 0)
    # Re-normalize spectral radius to 0.9
    eig = np.abs(np.linalg.eigvals(Wnew)).max()
    if eig > 1e-9:
        Wnew = Wnew * (0.9 / eig)
    return Wnew


def run_config(args):
    """Build ER_SPARSE + lateral inhibition, run z200's reservoir loop."""
    # Per-subprocess thread cap (F.1 safety)
    import os as _os
    for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
               "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        _os.environ[_k] = "4"
    try:
        from threadpoolctl import threadpool_limits
        threadpool_limits(limits=4)
    except Exception:
        pass

    radius, strength, seed = args
    N = 1024
    rule = "ff"

    # Re-import inside subprocess
    sys.path.insert(0, str(ROOT))
    from scripts.z200_topo_rule_sweep import (
        build_topo, gen_signal, run_reservoir_surr,
    )
    from scripts.nsram_surrogate import NSRAMSurrogate

    surr = NSRAMSurrogate.build_or_load(grid_size=(20, 20, 25))
    rng = np.random.default_rng(seed)
    base_VG1 = rng.choice([0.2, 0.4, 0.6], size=N).astype(float)
    base_VG2 = rng.uniform(0.0, 0.5, size=N).astype(float)
    sign_mask = rng.choice([-1.0, 1.0], size=N).astype(float)

    # Excitatory ER_SPARSE first
    W_exc = build_topo("ER_SPARSE", N, rng)
    # Add lateral inhibition
    W = add_lateral_inhibition(W_exc, radius, strength, rng)

    EPOCHS = 12
    N_TRAIN = 16
    N_TEST = 24
    T = 60
    LR = 5e-3   # ff

    history = []
    t0 = time.time()
    best_acc = 0.0
    final_acc = 0.0

    # Use simplified inline FF training (mirrors z200 ff path)
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
            # FF update on W rows where err < 0
            for i in np.where(err < 0)[0]:
                W[i, :] += LR * (lid_pos[i, -1] * lid_pos[:, -1]
                                  - lid_neg[i, -1] * lid_neg[:, -1])
            np.fill_diagonal(W, 0)
            # Renormalize occasionally
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

    wall = time.time() - t0
    return {
        "radius": radius, "strength": strength, "seed": seed,
        "best_acc": best_acc, "final_acc": final_acc,
        "history": history, "wall_s": wall,
    }


def main():
    radii = [0, 1, 2, 4, 8]
    strengths = [0.1, 0.3, 0.6]
    seeds = [0, 1, 2]
    grid = [(r, s, sd) for r in radii for s in strengths for sd in seeds
             if not (r == 0 and s != strengths[0])]  # skip duplicate r=0,s≠s[0]

    print(f"[z210] {len(grid)} configs (radii × strengths × seeds), N=1024",
          flush=True)
    print(f"[z210] estimate ~3-5 min/config × 12 workers ≈ {len(grid) * 4 / 12:.0f} min wall",
          flush=True)
    n_workers = 12

    results = []
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        futs = {ex.submit(run_config, a): a for a in grid}
        for f in as_completed(futs):
            r = f.result()
            results.append(r)
            print(f"[z210] r={r['radius']} s={r['strength']:.1f} sd{r['seed']}: "
                  f"best={r['best_acc']:.3f} final={r['final_acc']:.3f} "
                  f"({r['wall_s']:.0f}s)", flush=True)

    # Aggregate
    agg = {}
    for r in radii:
        for s in (strengths if r > 0 else [strengths[0]]):
            key = f"r{r}_s{s:.1f}"
            bests = [x["best_acc"] for x in results if x["radius"] == r and x["strength"] == s]
            finals = [x["final_acc"] for x in results if x["radius"] == r and x["strength"] == s]
            if bests:
                agg[key] = {
                    "best_mean": float(np.mean(bests)),
                    "best_std": float(np.std(bests)),
                    "final_mean": float(np.mean(finals)),
                    "final_std": float(np.std(finals)),
                    "n_seeds": len(bests),
                }

    summary = {
        "radii": radii, "strengths": strengths, "seeds": seeds,
        "results": results, "agg": agg, "total_wall_s": time.time() - t0,
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))

    print("\n[z210] aggregate (best test acc across 3 seeds):")
    print(f"  {'config':<14} {'best mean±std':>15} {'final mean±std':>15}")
    for k, v in sorted(agg.items()):
        print(f"  {k:<14} {v['best_mean']:.3f}±{v['best_std']:.3f}    "
              f"{v['final_mean']:.3f}±{v['final_std']:.3f}")
    print(f"\nwall: {summary['total_wall_s']:.0f}s")


if __name__ == "__main__":
    main()
