"""F6.v3 — extend Bf below 15k and tighten Va around 1V.

F6.v2 best 0.675 dec at Bf=15000, Va=1.0 V — both grid edges. Extend
Bf downward (literature lateral-NPN range starts at 1e3, oracles
suggested 1e4 lower bound is comfortable) and bracket Va=1V tighter.

Grid (5×5 = 25 fits, parallel via z150 wrapper pattern):
  Bf ∈ {8e3, 1.0e4, 1.2e4, 1.4e4, 1.5e4}
  Va ∈ {0.7, 0.85, 1.0, 1.15, 1.3}

Output:
  results/F6v3_bf_va_lower/summary.json
  figures/F6v3_bf_va_lower/heatmap.{png,pdf}
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
OUT = ROOT / "results/F6v3_bf_va_lower"; OUT.mkdir(parents=True, exist_ok=True)
FIG = ROOT / "figures/F6v3_bf_va_lower"; FIG.mkdir(parents=True, exist_ok=True)

BF_GRID = [8e3, 1.0e4, 1.2e4, 1.4e4, 1.5e4]
VA_GRID = [0.7, 0.85, 1.0, 1.15, 1.3]
IS_FIXED = 1e-9
N_WORKERS = 12


def run_one(args):
    bf, va = args
    suffix = f"_F6v3_bf{int(bf)}_va{va:g}"
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
    env["PYTHONUNBUFFERED"] = "1"
    t0 = time.time()
    try:
        cp = subprocess.run(
            [sys.executable, "scripts/z91g_two_model_validation.py"],
            cwd=str(ROOT), env=env, capture_output=True, timeout=900, text=True)
    except subprocess.TimeoutExpired:
        return (bf, va, float("nan"), float("nan"), "timeout", time.time() - t0)
    wall = time.time() - t0
    if cp.returncode != 0:
        return (bf, va, float("nan"), float("nan"),
                f"err:{(cp.stderr or '')[-200:]!r}", wall)
    try:
        d = json.loads(summary_path.read_text())
        return (bf, va, d.get("median_log_rmse"), d.get("p90_log_rmse"),
                "ok", wall)
    except Exception as e:
        return (bf, va, float("nan"), float("nan"), f"parse:{e!s}", wall)


def main():
    t0 = time.time()
    grid = [(bf, va) for bf in BF_GRID for va in VA_GRID]
    print(f"[F6v3] launching {len(grid)} fits / {N_WORKERS} workers", flush=True)
    print(f"[F6v3] Bf ∈ {BF_GRID} × Va ∈ {VA_GRID}", flush=True)
    results = []
    with ProcessPoolExecutor(max_workers=N_WORKERS) as ex:
        futures = {ex.submit(run_one, args): args for args in grid}
        for f in as_completed(futures):
            r = f.result()
            results.append(r)
            bf, va, med, p90, status, wall = r
            mstr = f"{med:.3f}" if isinstance(med,float) and np.isfinite(med) else "nan"
            print(f"[F6v3] Bf={int(bf):>5} Va={va:>4g}: med={mstr} ({status}, {wall:.0f}s)",
                  flush=True)

    M = np.full((len(BF_GRID), len(VA_GRID)), np.nan)
    for bf, va, med, _, _, _ in results:
        if isinstance(med, float) and np.isfinite(med):
            M[BF_GRID.index(bf), VA_GRID.index(va)] = med
    summary = {
        "BF_GRID": BF_GRID, "VA_GRID": VA_GRID, "IS_FIXED": IS_FIXED,
        "median_log_rmse": M.tolist(),
        "best_value": float(np.nanmin(M)) if np.isfinite(M).any() else None,
        "best_idx": [int(x) for x in np.unravel_index(np.nanargmin(M), M.shape)] \
            if np.isfinite(M).any() else None,
        "wall_total_s": time.time() - t0,
        "raw": [{"bf":r[0], "va":r[1], "med":r[2], "p90":r[3],
                  "status":r[4], "wall_s":r[5]} for r in results],
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))

    cmap = LinearSegmentedColormap.from_list("nrmse",
        [(0.0,"#1a9850"),(0.30,"#a6d96a"),(0.50,"#ffffbf"),
         (0.70,"#fdae61"),(1.00,"#a50026")])
    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(M, cmap=cmap, vmin=0.6, vmax=0.85, aspect="auto")
    for i in range(len(BF_GRID)):
        for j in range(len(VA_GRID)):
            v = M[i,j]
            if np.isnan(v): continue
            c = "white" if (v < 0.66 or v > 0.80) else "black"
            w = "bold" if v < 0.675 else "normal"
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
    ax.set_yticklabels([f"{int(b)}" for b in BF_GRID])
    ax.set_xlabel("Va — forward Early voltage (V)")
    ax.set_ylabel("Bf — BJT forward gain")
    ax.set_title("F6.v3: extend Bf↓ + tighten Va around 1V (Is=1e-9)\n"
                 "(F6.v2 best: 0.675 at Bf=15000, Va=1.0 — both grid edges)")
    plt.colorbar(im, ax=ax, label="log10 RMSE (decades)")
    plt.tight_layout()
    plt.savefig(FIG / "heatmap.png", dpi=150)
    plt.savefig(FIG / "heatmap.pdf")
    plt.close()
    print(f"\n[F6v3] best: med={summary['best_value']:.3f} dec at "
          f"Bf={int(BF_GRID[summary['best_idx'][0]])}, "
          f"Va={VA_GRID[summary['best_idx'][1]]:g}")
    print(f"[F6v3] heatmap saved. wall: {summary['wall_total_s']:.0f}s")


if __name__ == "__main__":
    main()
