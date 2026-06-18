"""z203 — extend WS_SMALLWORLD/ff scaling to N=8192 (single seed).

z202 scan ended at N=4096 (best 0.861±0.098). Push one more octave
to verify the scaling trend (is best plateau or still climbing?).

Single seed (~15-20 min wall) due to per-config compute scaling.
Loads existing z202 summary and appends.

Output:
  results/z203_n8192/summary.json (single config)
  figures/z203_n8192/scaling_extended.{png,pdf} (z202 + this)
"""
from __future__ import annotations
# THREAD CAP — z203 N=8192 single-process numpy caused the 2026-05-05
# 12:54 ACPI thermal-trip crash (APU 101°C). Cap at 8 threads since
# this is a SINGLE process (no ProcessPoolExecutor parallelism).
import os
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
            "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_k, "8")
import json, sys, time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results/z203_n8192"; OUT.mkdir(parents=True, exist_ok=True)
FIG = ROOT / "figures/z203_n8192"; FIG.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(ROOT))
from scripts.z200_topo_rule_sweep import run_config

def main():
    t0 = time.time()
    print(f"[z203] launching WS_SMALLWORLD/ff N=8192 single seed", flush=True)
    res = run_config(("WS_SMALLWORLD", "ff", 8192, 0))
    res["wall_s"] = time.time() - t0
    print(f"[z203] done: best={res['best_acc']:.3f} final={res['final_acc']:.3f} "
          f"({res['wall_s']:.0f}s)", flush=True)
    (OUT / "summary.json").write_text(json.dumps(res, indent=2))

    # Build extended scaling figure
    z202 = json.loads((ROOT / "results/z202_ws_scaling/summary.json").read_text())
    Ns = sorted({int(s) for s in z202["agg"].keys()} | {8192})
    best_means, best_stds = [], []
    final_means, final_stds = [], []
    for N in Ns:
        if N == 8192:
            best_means.append(res["best_acc"]); best_stds.append(0)
            final_means.append(res["final_acc"]); final_stds.append(0)
        else:
            a = z202["agg"][str(N)]
            best_means.append(a["best_mean"]); best_stds.append(a["best_std"])
            final_means.append(a["final_mean"]); final_stds.append(a["final_std"])

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 5))
    Ns_a = np.array(Ns)
    bm = np.array(best_means); bs = np.array(best_stds)
    fm = np.array(final_means); fs = np.array(final_stds)
    ax.errorbar(Ns_a, bm, yerr=bs, marker="o", lw=1.5, color="#27ae60",
                label="best test acc (z202: 3 seeds, z203: 1 seed)", capsize=4)
    ax.errorbar(Ns_a, fm, yerr=fs, marker="s", lw=1.5, color="#3498db",
                label="final test acc", capsize=4)
    ax.axhline(0.5, ls=":", color="grey", lw=0.8, label="chance")
    ax.set_xscale("log", base=2)
    ax.set_xticks(Ns)
    ax.set_xticklabels([str(N) for N in Ns])
    ax.set_xlabel("Reservoir size N (cells)")
    ax.set_ylabel("Test accuracy")
    ax.set_ylim(0.4, 1.05)
    ax.set_title("WS_SMALLWORLD + FF scaling extended to N=8192\n"
                 f"(z203 best={res['best_acc']:.3f} at N=8192)")
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(FIG / "scaling_extended.png", dpi=150)
    plt.savefig(FIG / "scaling_extended.pdf")
    plt.close()
    print(f"[z203] figure saved: {FIG}/scaling_extended.png")


if __name__ == "__main__":
    main()
