"""z202 — WS_SMALLWORLD + FF scaling at N ∈ {256, 1024, 2048, 4096}.

z200 found WS_SMALLWORLD + Forward-Forward + N=1024 was the only
self-learning rule that reached 1.000 best AND 0.833 final on the
2-class MG vs sin task (others lottery-peaked then collapsed).

This run probes scaling: does it keep improving with N? Three seeds
per size for noise bars. No W_history saved (memory) at N=4096.

Output:
  results/z202_ws_scaling/summary.json
  figures/z202_ws_scaling/scaling.{png,pdf}
"""
from __future__ import annotations
# THREAD CAP — see z200_topo_rule_sweep.py for rationale.
import os
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
            "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_k, "4")
import json, time, sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results/z202_ws_scaling"; OUT.mkdir(parents=True, exist_ok=True)
FIG = ROOT / "figures/z202_ws_scaling"; FIG.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(ROOT))
from scripts.z200_topo_rule_sweep import run_config

SIZES = [256, 1024, 2048, 4096]
SEEDS = [0, 1, 2]
TOPO = "WS_SMALLWORLD"
RULE = "ff"


def run_one(args):
    N, seed = args
    t0 = time.time()
    res = run_config((TOPO, RULE, N, seed))
    res["wall_s"] = time.time() - t0
    return res


def main():
    t0 = time.time()
    grid = [(N, s) for N in SIZES for s in SEEDS]
    print(f"[z202] {len(grid)} configs ({len(SIZES)} sizes × {len(SEEDS)} seeds), "
          f"WS_SMALLWORLD/ff", flush=True)
    n_workers = 6  # heavier per-config (large N), lower parallelism
    results = []
    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        futs = {ex.submit(run_one, a): a for a in grid}
        for f in as_completed(futs):
            r = f.result()
            results.append(r)
            print(f"[z202] N={r['N']:>5} s{r['seed']}: "
                  f"best={r['best_acc']:.3f} final={r['final_acc']:.3f} "
                  f"({r['wall_s']:.0f}s)", flush=True)

    # Aggregate per N
    agg = {}
    for N in SIZES:
        bests = [r["best_acc"] for r in results if r["N"] == N]
        finals = [r["final_acc"] for r in results if r["N"] == N]
        agg[N] = {
            "best_mean": float(np.mean(bests)), "best_std": float(np.std(bests)),
            "final_mean": float(np.mean(finals)),
            "final_std": float(np.std(finals)),
            "raw_best": bests, "raw_final": finals,
        }

    summary = {
        "topo": TOPO, "rule": RULE, "sizes": SIZES, "seeds": SEEDS,
        "results": results, "agg": {str(k): v for k, v in agg.items()},
        "total_wall_s": time.time() - t0,
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))

    print("\n[z202] aggregate:")
    for N in SIZES:
        a = agg[N]
        print(f"  N={N:<5} best={a['best_mean']:.3f}±{a['best_std']:.3f}  "
              f"final={a['final_mean']:.3f}±{a['final_std']:.3f}")

    # Plot scaling
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 5))
    Ns = np.array(SIZES)
    bm  = np.array([agg[N]["best_mean"] for N in SIZES])
    bs  = np.array([agg[N]["best_std"] for N in SIZES])
    fm  = np.array([agg[N]["final_mean"] for N in SIZES])
    fs  = np.array([agg[N]["final_std"] for N in SIZES])
    ax.errorbar(Ns, bm, yerr=bs, marker="o", lw=1.5, color="#27ae60",
                label="best test acc (mean ± std over 3 seeds)", capsize=4)
    ax.errorbar(Ns, fm, yerr=fs, marker="s", lw=1.5, color="#3498db",
                label="final test acc (mean ± std)", capsize=4)
    ax.axhline(0.5, ls=":", color="grey", lw=0.8, label="chance")
    ax.set_xscale("log", base=2)
    ax.set_xticks(SIZES)
    ax.set_xticklabels([str(N) for N in SIZES])
    ax.set_xlabel("Reservoir size N (cells)")
    ax.set_ylabel("Test accuracy (2-class MG vs sin, 24 samples)")
    ax.set_ylim(0.4, 1.05)
    ax.set_title("WS_SMALLWORLD + Forward-Forward scaling on NS-RAM surrogate\n"
                 "(no linear readout, per-cell ±VG2 mask, σ Id² goodness)")
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(FIG / "scaling.png", dpi=150)
    plt.savefig(FIG / "scaling.pdf")
    plt.close()
    print(f"\n[z202] scaling plot saved. wall: {summary['total_wall_s']:.0f}s")


if __name__ == "__main__":
    main()
