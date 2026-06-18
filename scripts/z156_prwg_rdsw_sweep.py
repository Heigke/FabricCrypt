"""F6.v7 — M1 PRWG × Rdsw sweep at the new optimum.

Per O24 oracle gpt-5: BSIM4 RDSMOD/PRWG modulates S/D resistance by Vgs
— directly relevant to VG1=0.40 weak-inversion cluster (residual
diagnostic shows all 5 worst rows at VG1=0.40 across all sweeps).

Default M1: PRWG=0 (Vg-dep OFF), Rdsw=100, Rdswmin=35.

Grid (5×5 = 25 fits, parallel):
  PRWG ∈ {0.0, 0.25, 0.5, 0.75, 1.0}     (Vg-dep on/off, smooth)
  Rdsw ∈ {30, 100, 300, 1000, 3000} Ω·µm  (3× nominal sweep)

BJT params held at F6.v3 optimum: Bf=9000, Va=0.55, Is=1e-9.

Output:
  results/F6v7_prwg_rdsw/summary.json
  figures/F6v7_prwg_rdsw/heatmap.{png,pdf}
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
OUT = ROOT / "results/F6v7_prwg_rdsw"; OUT.mkdir(parents=True, exist_ok=True)
FIG = ROOT / "figures/F6v7_prwg_rdsw"; FIG.mkdir(parents=True, exist_ok=True)

PRWG_GRID = [0.0, 0.25, 0.5, 0.75, 1.0]
RDSW_GRID = [30.0, 100.0, 300.0, 1000.0, 3000.0]
BF_FIXED, VA_FIXED, IS_FIXED = 9000.0, 0.55, 1e-9
N_WORKERS = 10


def run_one(args):
    prwg, rdsw = args
    suffix = f"_F6v7_prwg{prwg:g}_rdsw{rdsw:g}".replace(".", "p")
    out_dir = ROOT / f"results/z91g_two_model_validation{suffix}"
    summary_path = out_dir / "summary.json"
    if summary_path.exists():
        try:
            d = json.loads(summary_path.read_text())
            return (prwg, rdsw, d.get("median_log_rmse"), d.get("p90_log_rmse"),
                    "cached", 0.0)
        except Exception: pass
    env = os.environ.copy()
    env.update({
        "NSRAM_OUT_SUFFIX": suffix,
        "NSRAM_BJT_BF": f"{BF_FIXED}",
        "NSRAM_BJT_VA": f"{VA_FIXED}",
        "NSRAM_BJT_IS": f"{IS_FIXED}",
        "NSRAM_M1_PRWG": f"{prwg}",
        "NSRAM_M1_RDSW": f"{rdsw}",
        "PYTHONUNBUFFERED": "1",
    })
    t0 = time.time()
    try:
        cp = subprocess.run([sys.executable, "scripts/z91g_two_model_validation.py"],
                             cwd=str(ROOT), env=env, capture_output=True,
                             timeout=900, text=True)
    except subprocess.TimeoutExpired:
        return (prwg, rdsw, float("nan"), float("nan"), "timeout", time.time()-t0)
    wall = time.time() - t0
    if cp.returncode != 0:
        return (prwg, rdsw, float("nan"), float("nan"),
                f"err:{(cp.stderr or '')[-200:]!r}", wall)
    try:
        d = json.loads(summary_path.read_text())
        return (prwg, rdsw, d.get("median_log_rmse"), d.get("p90_log_rmse"),
                "ok", wall)
    except Exception as e:
        return (prwg, rdsw, float("nan"), float("nan"), f"parse:{e!s}", wall)


def main():
    t0 = time.time()
    grid = [(p, r) for p in PRWG_GRID for r in RDSW_GRID]
    print(f"[F6v7] {len(grid)} fits / {N_WORKERS} workers; "
          f"Bf={BF_FIXED} Va={VA_FIXED} Is={IS_FIXED}", flush=True)
    print(f"[F6v7] PRWG ∈ {PRWG_GRID}", flush=True)
    print(f"[F6v7] Rdsw ∈ {RDSW_GRID}", flush=True)
    results = []
    with ProcessPoolExecutor(max_workers=N_WORKERS) as ex:
        futures = {ex.submit(run_one, args): args for args in grid}
        for f in as_completed(futures):
            r = f.result()
            results.append(r)
            prwg, rdsw, med, _, status, wall = r
            ms = f"{med:.3f}" if isinstance(med,float) and np.isfinite(med) else "nan"
            print(f"[F6v7] PRWG={prwg:>4g} Rdsw={rdsw:>5g}: "
                  f"med={ms} ({status}, {wall:.0f}s)", flush=True)

    M = np.full((len(PRWG_GRID), len(RDSW_GRID)), np.nan)
    for prwg, rdsw, med, _, _, _ in results:
        if isinstance(med, float) and np.isfinite(med):
            M[PRWG_GRID.index(prwg), RDSW_GRID.index(rdsw)] = med
    summary = {
        "BF_FIXED": BF_FIXED, "VA_FIXED": VA_FIXED, "IS_FIXED": IS_FIXED,
        "PRWG_GRID": PRWG_GRID, "RDSW_GRID": RDSW_GRID,
        "median_log_rmse": M.tolist(),
        "best_value": float(np.nanmin(M)) if np.isfinite(M).any() else None,
        "best_idx": [int(x) for x in np.unravel_index(np.nanargmin(M), M.shape)] \
            if np.isfinite(M).any() else None,
        "wall_total_s": time.time() - t0,
        "raw": [{"prwg":r[0], "rdsw":r[1], "med":r[2], "p90":r[3],
                  "status":r[4], "wall_s":r[5]} for r in results],
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))

    cmap = LinearSegmentedColormap.from_list("nrmse",
        [(0.0,"#1a9850"),(0.30,"#a6d96a"),(0.50,"#ffffbf"),
         (0.70,"#fdae61"),(1.00,"#a50026")])
    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(M, cmap=cmap, vmin=0.6, vmax=0.9, aspect="auto")
    for i in range(len(PRWG_GRID)):
        for j in range(len(RDSW_GRID)):
            v = M[i,j]
            if np.isnan(v): continue
            c = "white" if (v < 0.66 or v > 0.84) else "black"
            w = "bold" if v < 0.657 else "normal"
            ax.text(j, i, f"{v:.3f}", ha="center", va="center",
                    color=c, fontsize=10, weight=w)
    if summary["best_idx"]:
        bi, bj = summary["best_idx"]
        ax.plot(bj, bi, "*", color="cyan", markersize=22,
                markeredgecolor="black", markeredgewidth=1.5,
                label=f"best: {summary['best_value']:.3f} dec")
        ax.legend(loc="upper right")
    ax.set_xticks(range(len(RDSW_GRID)))
    ax.set_xticklabels([f"{int(v)}" for v in RDSW_GRID])
    ax.set_yticks(range(len(PRWG_GRID)))
    ax.set_yticklabels([f"{x:g}" for x in PRWG_GRID])
    ax.set_xlabel("Rdsw — S/D resistance (Ω·µm)")
    ax.set_ylabel("PRWG — Vg-dependence coefficient")
    ax.set_title(f"F6.v7: M1 PRWG × Rdsw at Bf={BF_FIXED:.0e}, Va={VA_FIXED}, "
                 f"Is={IS_FIXED:.0e}\n"
                 f"(F6.v3 best: 0.657 at PRWG=0, Rdsw=100 [defaults])")
    plt.colorbar(im, ax=ax, label="log10 RMSE (decades)")
    plt.tight_layout()
    plt.savefig(FIG / "heatmap.png", dpi=150)
    plt.savefig(FIG / "heatmap.pdf")
    plt.close()
    if summary["best_idx"]:
        bi, bj = summary["best_idx"]
        print(f"\n[F6v7] best: med={summary['best_value']:.3f} dec at "
              f"PRWG={PRWG_GRID[bi]:g}, Rdsw={RDSW_GRID[bj]:g}")
    print(f"[F6v7] heatmap saved. wall: {summary['wall_total_s']:.0f}s")


if __name__ == "__main__":
    main()
