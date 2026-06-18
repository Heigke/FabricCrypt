"""F6: 2D Bf × Va (forward Early voltage) parameter sweep — post-O24.

Both gpt-5 and gemini ranked Va as the #1 untested knob. Lateral parasitic
NPN with low-doping base typically has Va ∈ [1, 50]V; default card uses
100V (vertical-NPN value).

Sweep grid (5×5 = 25 fits, parallel via ProcessPoolExecutor):
  Bf ∈ {1e4, 2e4, 3e4, 5e4, 1e5}
  Va ∈ {3, 10, 30, 100, 300}    # broad — confirm 100V is too high

Each fit invokes scripts/z91g_two_model_validation.py with a unique
NSRAM_OUT_SUFFIX and BJT env-vars (NSRAM_BJT_BF, NSRAM_BJT_VA, plus
NSRAM_BJT_IS=1e-9 from the breakthrough optimum).

Runs while v2-demo is busy (which uses 1 of 32 cores). Uses up to 12
parallel workers to stay friendly.

Output:
  results/F6_bf_va_sweep/summary.json
  figures/F6_bf_va_sweep/heatmap.{png,pdf}
"""
from __future__ import annotations
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results/F6_bf_va_sweep"; OUT.mkdir(parents=True, exist_ok=True)
FIG = ROOT / "figures/F6_bf_va_sweep"; FIG.mkdir(parents=True, exist_ok=True)

# Grid
BF_GRID = [1e4, 2e4, 3e4, 5e4, 1e5]
VA_GRID = [3.0, 10.0, 30.0, 100.0, 300.0]
IS_FIXED = 1e-9

N_WORKERS = 12     # leave room for v2 + cron + headroom on 32-core box


def run_one(args):
    bf, va = args
    suffix = f"_F6_bf{bf:.0e}_va{va:g}"
    out_dir = ROOT / f"results/z91g_two_model_validation{suffix}"
    summary_path = out_dir / "summary.json"
    if summary_path.exists():
        try:
            d = json.loads(summary_path.read_text())
            return (bf, va, d.get("median_log_rmse"), d.get("p90_log_rmse"),
                    "cached", 0.0)
        except Exception:
            pass
    env = os.environ.copy()
    env["NSRAM_OUT_SUFFIX"] = suffix
    env["NSRAM_BJT_BF"] = f"{bf}"
    env["NSRAM_BJT_VA"] = f"{va}"
    env["NSRAM_BJT_IS"] = f"{IS_FIXED}"
    # Honest physical mode (η-bounded).
    env["PYTHONUNBUFFERED"] = "1"
    t0 = time.time()
    try:
        cp = subprocess.run(
            [sys.executable, "scripts/z91g_two_model_validation.py"],
            cwd=str(ROOT), env=env, capture_output=True, timeout=900,
            text=True)
    except subprocess.TimeoutExpired:
        return (bf, va, float("nan"), float("nan"), "timeout",
                time.time() - t0)
    wall = time.time() - t0
    if cp.returncode != 0:
        err_tail = (cp.stderr or "")[-300:]
        return (bf, va, float("nan"), float("nan"), f"err:{err_tail!r}",
                wall)
    try:
        d = json.loads(summary_path.read_text())
        return (bf, va, d.get("median_log_rmse"), d.get("p90_log_rmse"),
                "ok", wall)
    except Exception as e:
        return (bf, va, float("nan"), float("nan"), f"parse:{e!s}", wall)


def main():
    t0 = time.time()
    grid = [(bf, va) for bf in BF_GRID for va in VA_GRID]
    print(f"[F6] launching {len(grid)} fits with {N_WORKERS} workers, "
          f"Is fixed at {IS_FIXED:.0e}")
    print(f"[F6] grid: Bf ∈ {BF_GRID} × Va ∈ {VA_GRID}")
    results = []
    with ProcessPoolExecutor(max_workers=N_WORKERS) as ex:
        futures = {ex.submit(run_one, args): args for args in grid}
        for f in as_completed(futures):
            res = f.result()
            results.append(res)
            bf, va, med, p90, status, wall = res
            print(f"[F6] Bf={bf:.0e} Va={va:>5g}: "
                  f"med={med if isinstance(med,float) else 'nan':.3f} "
                  f"p90={p90 if isinstance(p90,float) else 'nan':.3f} "
                  f"({status}, {wall:.0f}s)", flush=True)

    # Aggregate
    M = np.full((len(BF_GRID), len(VA_GRID)), np.nan)
    for bf, va, med, p90, status, _ in results:
        if not isinstance(med, float) or not np.isfinite(med): continue
        i = BF_GRID.index(bf); j = VA_GRID.index(va)
        M[i, j] = med
    summary = {
        "BF_GRID": BF_GRID, "VA_GRID": VA_GRID, "IS_FIXED": IS_FIXED,
        "median_log_rmse": M.tolist(),
        "best_value": float(np.nanmin(M)) if np.isfinite(M).any() else None,
        "best_idx": [int(x) for x in np.unravel_index(np.nanargmin(M), M.shape)] \
            if np.isfinite(M).any() else None,
        "wall_total_s": time.time() - t0,
        "raw": [{"bf": r[0], "va": r[1], "med": r[2], "p90": r[3],
                  "status": r[4], "wall_s": r[5]} for r in results],
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\n[F6] best: med={summary['best_value']:.3f} dec at "
          f"Bf={BF_GRID[summary['best_idx'][0]]:.0e}, "
          f"Va={VA_GRID[summary['best_idx'][1]]:g}")

    # Heatmap
    cmap = LinearSegmentedColormap.from_list(
        "nrmse", [(0.0, "#1a9850"), (0.30, "#a6d96a"),
                  (0.50, "#ffffbf"), (0.70, "#fdae61"),
                  (1.00, "#a50026")])
    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(M, cmap=cmap, vmin=0.5, vmax=1.6, aspect="auto")
    for i in range(len(BF_GRID)):
        for j in range(len(VA_GRID)):
            v = M[i, j]
            if np.isnan(v): continue
            c = "white" if (v < 0.85 or v > 1.4) else "black"
            w = "bold" if v < 0.795 else "normal"
            ax.text(j, i, f"{v:.3f}", ha="center", va="center",
                    color=c, fontsize=10, weight=w)
    if summary["best_idx"]:
        bi, bj = summary["best_idx"]
        ax.plot(bj, bi, "*", color="cyan", markersize=22,
                markeredgecolor="black", markeredgewidth=1.5,
                label=f"best: {summary['best_value']:.3f} dec")
        ax.legend(loc="upper right")
    ax.set_xticks(range(len(VA_GRID)))
    ax.set_xticklabels([f"{v:g}" for v in VA_GRID])
    ax.set_yticks(range(len(BF_GRID)))
    ax.set_yticklabels([f"{b:.0e}" for b in BF_GRID])
    ax.set_xlabel("Va — forward Early voltage (V)")
    ax.set_ylabel("Bf — BJT forward gain")
    ax.set_title("F6: Bf × Va 2D sweep at Is=1×10⁻⁹ — median log-RMSE\n"
                 "(prior best 0.795 at Bf=2e4, Va=100; oracles: lateral NPN Va∈[1,50])")
    plt.colorbar(im, ax=ax, label="log10 RMSE (decades)")
    plt.tight_layout()
    plt.savefig(FIG / "heatmap.png", dpi=150)
    plt.savefig(FIG / "heatmap.pdf")
    plt.close()
    print(f"[F6] heatmap saved to {FIG/'heatmap.png'}")
    print(f"[F6] total wall: {summary['wall_total_s']:.0f}s")


if __name__ == "__main__":
    main()
