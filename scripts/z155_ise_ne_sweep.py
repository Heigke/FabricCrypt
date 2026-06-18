"""F6.v6 — ISE × NE sweep at new optimum (Bf=9000, Va=0.55, Is=1e-9).

ISE/NE govern non-ideal B-E diode current at low Vbe. Per O24 oracles
this targets the low-Vbe / weak-inversion corner where VG1=0.40
cluster lives (verified persistent across Bf×Is and Bf×Va sweeps).

Default: ISE=0 (turned off). NE=1.5 (already non-ideal default per
Sebas card). We sweep ISE on, with NE bracketing 1.2-2.0.

Constraint per gpt-5: ISE/Is ≤ ~0.1 to avoid double-counting Is.

Grid (5×5 = 25 fits, parallel):
  ISE ∈ {0, 1e-12, 1e-11, 1e-10, 1e-9}     (0, 0.001×Is, 0.01×Is,
                                              0.1×Is, 1×Is)
  NE  ∈ {1.2, 1.5, 1.7, 2.0, 2.3}

Output:
  results/F6v6_ise_ne/summary.json
  figures/F6v6_ise_ne/heatmap.{png,pdf}
"""
from __future__ import annotations
import json, os, subprocess, sys, time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results/F6v6_ise_ne"; OUT.mkdir(parents=True, exist_ok=True)
FIG = ROOT / "figures/F6v6_ise_ne"; FIG.mkdir(parents=True, exist_ok=True)

ISE_GRID = [0.0, 1e-12, 1e-11, 1e-10, 1e-9]
NE_GRID  = [1.2, 1.5, 1.7, 2.0, 2.3]
BF_FIXED = 9000.0
VA_FIXED = 0.55
IS_FIXED = 1e-9
N_WORKERS = 10


def run_one(args):
    ise, ne = args
    suffix = f"_F6v6_ise{ise:g}_ne{ne:g}".replace("+", "").replace(".", "p")
    out_dir = ROOT / f"results/z91g_two_model_validation{suffix}"
    summary_path = out_dir / "summary.json"
    if summary_path.exists():
        try:
            d = json.loads(summary_path.read_text())
            return (ise, ne, d.get("median_log_rmse"), d.get("p90_log_rmse"),
                    "cached", 0.0)
        except Exception: pass
    env = os.environ.copy()
    env["NSRAM_OUT_SUFFIX"] = suffix
    env["NSRAM_BJT_BF"] = f"{BF_FIXED}"
    env["NSRAM_BJT_VA"] = f"{VA_FIXED}"
    env["NSRAM_BJT_IS"] = f"{IS_FIXED}"
    env["NSRAM_BJT_ISE"] = f"{ise}"
    env["NSRAM_BJT_NE"] = f"{ne}"
    env["PYTHONUNBUFFERED"] = "1"
    t0 = time.time()
    try:
        cp = subprocess.run([sys.executable, "scripts/z91g_two_model_validation.py"],
                             cwd=str(ROOT), env=env, capture_output=True,
                             timeout=900, text=True)
    except subprocess.TimeoutExpired:
        return (ise, ne, float("nan"), float("nan"), "timeout", time.time()-t0)
    wall = time.time() - t0
    if cp.returncode != 0:
        return (ise, ne, float("nan"), float("nan"),
                f"err:{(cp.stderr or '')[-200:]!r}", wall)
    try:
        d = json.loads(summary_path.read_text())
        return (ise, ne, d.get("median_log_rmse"), d.get("p90_log_rmse"),
                "ok", wall)
    except Exception as e:
        return (ise, ne, float("nan"), float("nan"), f"parse:{e!s}", wall)


def main():
    t0 = time.time()
    grid = [(ise, ne) for ise in ISE_GRID for ne in NE_GRID]
    print(f"[F6v6] {len(grid)} fits / {N_WORKERS} workers; "
          f"Bf={BF_FIXED}, Va={VA_FIXED}, Is={IS_FIXED}", flush=True)
    print(f"[F6v6] Ise ∈ {ISE_GRID}", flush=True)
    print(f"[F6v6] Ne  ∈ {NE_GRID}", flush=True)
    results = []
    with ProcessPoolExecutor(max_workers=N_WORKERS) as ex:
        futures = {ex.submit(run_one, args): args for args in grid}
        for f in as_completed(futures):
            r = f.result()
            results.append(r)
            ise, ne, med, _p90, status, wall = r
            ms = f"{med:.3f}" if isinstance(med,float) and np.isfinite(med) else "nan"
            print(f"[F6v6] Ise={ise:>8.0e} Ne={ne:>4g}: med={ms} ({status}, {wall:.0f}s)",
                  flush=True)

    M = np.full((len(ISE_GRID), len(NE_GRID)), np.nan)
    for ise, ne, med, _, _, _ in results:
        if isinstance(med, float) and np.isfinite(med):
            M[ISE_GRID.index(ise), NE_GRID.index(ne)] = med
    summary = {
        "BF_FIXED": BF_FIXED, "VA_FIXED": VA_FIXED, "IS_FIXED": IS_FIXED,
        "ISE_GRID": ISE_GRID, "NE_GRID": NE_GRID,
        "median_log_rmse": M.tolist(),
        "best_value": float(np.nanmin(M)) if np.isfinite(M).any() else None,
        "best_idx": [int(x) for x in np.unravel_index(np.nanargmin(M), M.shape)] \
            if np.isfinite(M).any() else None,
        "wall_total_s": time.time() - t0,
        "raw": [{"ise":r[0], "ne":r[1], "med":r[2], "p90":r[3],
                  "status":r[4], "wall_s":r[5]} for r in results],
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))

    cmap = LinearSegmentedColormap.from_list("nrmse",
        [(0.0,"#1a9850"),(0.30,"#a6d96a"),(0.50,"#ffffbf"),
         (0.70,"#fdae61"),(1.00,"#a50026")])
    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(M, cmap=cmap, vmin=0.6, vmax=0.85, aspect="auto")
    for i in range(len(ISE_GRID)):
        for j in range(len(NE_GRID)):
            v = M[i,j]
            if np.isnan(v): continue
            c = "white" if (v < 0.66 or v > 0.80) else "black"
            w = "bold" if v < 0.657 else "normal"
            ax.text(j, i, f"{v:.3f}", ha="center", va="center",
                    color=c, fontsize=10, weight=w)
    if summary["best_idx"]:
        bi, bj = summary["best_idx"]
        ax.plot(bj, bi, "*", color="cyan", markersize=22,
                markeredgecolor="black", markeredgewidth=1.5,
                label=f"best: {summary['best_value']:.3f} dec")
        ax.legend(loc="upper right")
    ax.set_xticks(range(len(NE_GRID)))
    ax.set_xticklabels([f"{v:g}" for v in NE_GRID])
    ax.set_yticks(range(len(ISE_GRID)))
    ax.set_yticklabels([f"{x:g}" for x in ISE_GRID])
    ax.set_xlabel("Ne — B-E emission coefficient")
    ax.set_ylabel("Ise — B-E leakage saturation (A)")
    ax.set_title(f"F6.v6: ISE × NE at Bf={BF_FIXED:.0e}, Va={VA_FIXED}, "
                 f"Is={IS_FIXED:.0e}\n"
                 f"(F6.v3 best: 0.657 at Ise=0)")
    plt.colorbar(im, ax=ax, label="log10 RMSE (decades)")
    plt.tight_layout()
    plt.savefig(FIG / "heatmap.png", dpi=150)
    plt.savefig(FIG / "heatmap.pdf")
    plt.close()
    if summary["best_idx"]:
        bi, bj = summary["best_idx"]
        print(f"\n[F6v6] best: med={summary['best_value']:.3f} dec at "
              f"Ise={ISE_GRID[bi]:g}, Ne={NE_GRID[bj]:g}")
    print(f"[F6v6] heatmap saved. wall: {summary['wall_total_s']:.0f}s")


if __name__ == "__main__":
    main()
