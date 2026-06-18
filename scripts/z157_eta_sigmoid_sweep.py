"""F6.v8 — η(Vbe) sigmoid: η_0 × V_turn at fixed k=30, η_final=0.

Per O25 gemini: bias-dependent η_lat. Baseline runs η=0 (constant);
sigmoid form with η_final=0 reproduces it exactly when k is large or
when V_turn is below the operating range. Sweep η_0 in 0..0.8 and
V_turn in 0.3..1.1 to map the landscape.

Output:
  results/F6v8_eta_sigmoid/summary.json
  figures/F6v8_eta_sigmoid/heatmap.{png,pdf}
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
OUT = ROOT / "results/F6v8_eta_sigmoid"; OUT.mkdir(parents=True, exist_ok=True)
FIG = ROOT / "figures/F6v8_eta_sigmoid"; FIG.mkdir(parents=True, exist_ok=True)

ETA0_GRID  = [0.0, 0.1, 0.3, 0.5, 0.8]
VTURN_GRID = [0.3, 0.5, 0.7, 0.9, 1.1]
ETA_FINAL = 0.0; K_FIXED = 30.0
BF, VA, IS = 9000.0, 0.55, 1e-9
N_WORKERS = 10


def run_one(args):
    eta0, vturn = args
    suffix = f"_F6v8_eta{eta0:g}_vt{vturn:g}".replace(".", "p")
    out_dir = ROOT / f"results/z91g_two_model_validation{suffix}"
    summary_path = out_dir / "summary.json"
    if summary_path.exists():
        try:
            d = json.loads(summary_path.read_text())
            return (eta0, vturn, d.get("median_log_rmse"),
                    d.get("p90_log_rmse"), "cached", 0.0)
        except Exception: pass
    env = os.environ.copy()
    env.update({
        "NSRAM_OUT_SUFFIX": suffix,
        "NSRAM_BJT_BF": f"{BF}", "NSRAM_BJT_VA": f"{VA}", "NSRAM_BJT_IS": f"{IS}",
        "NSRAM_ETA_SIGMOID": "1",
        "NSRAM_ETA_0": f"{eta0}",
        "NSRAM_ETA_FINAL": f"{ETA_FINAL}",
        "NSRAM_ETA_K": f"{K_FIXED}",
        "NSRAM_ETA_VTURN": f"{vturn}",
        "PYTHONUNBUFFERED": "1",
    })
    t0 = time.time()
    try:
        cp = subprocess.run([sys.executable, "scripts/z91g_two_model_validation.py"],
                             cwd=str(ROOT), env=env, capture_output=True,
                             timeout=900, text=True)
    except subprocess.TimeoutExpired:
        return (eta0, vturn, float("nan"), float("nan"), "timeout", time.time()-t0)
    wall = time.time() - t0
    if cp.returncode != 0:
        return (eta0, vturn, float("nan"), float("nan"),
                f"err:{(cp.stderr or '')[-200:]!r}", wall)
    try:
        d = json.loads(summary_path.read_text())
        return (eta0, vturn, d.get("median_log_rmse"), d.get("p90_log_rmse"),
                "ok", wall)
    except Exception as e:
        return (eta0, vturn, float("nan"), float("nan"), f"parse:{e!s}", wall)


def main():
    t0 = time.time()
    grid = [(e, v) for e in ETA0_GRID for v in VTURN_GRID]
    print(f"[F6v8] {len(grid)} fits / {N_WORKERS} workers; "
          f"k={K_FIXED}, η_final={ETA_FINAL}, Bf={BF}, Va={VA}, Is={IS}",
          flush=True)
    results = []
    with ProcessPoolExecutor(max_workers=N_WORKERS) as ex:
        futures = {ex.submit(run_one, args): args for args in grid}
        for f in as_completed(futures):
            r = f.result()
            results.append(r)
            e, v, med, _, status, wall = r
            ms = f"{med:.3f}" if isinstance(med, float) and np.isfinite(med) else "nan"
            print(f"[F6v8] η_0={e:>4g} V_turn={v:>4g}: "
                  f"med={ms} ({status}, {wall:.0f}s)", flush=True)

    M = np.full((len(ETA0_GRID), len(VTURN_GRID)), np.nan)
    for e, v, med, _, _, _ in results:
        if isinstance(med, float) and np.isfinite(med):
            M[ETA0_GRID.index(e), VTURN_GRID.index(v)] = med
    summary = {
        "BF": BF, "VA": VA, "IS": IS,
        "ETA_FINAL": ETA_FINAL, "K_FIXED": K_FIXED,
        "ETA0_GRID": ETA0_GRID, "VTURN_GRID": VTURN_GRID,
        "median_log_rmse": M.tolist(),
        "best_value": float(np.nanmin(M)) if np.isfinite(M).any() else None,
        "best_idx": [int(x) for x in np.unravel_index(np.nanargmin(M), M.shape)] \
            if np.isfinite(M).any() else None,
        "wall_total_s": time.time() - t0,
        "raw": [{"eta_0":r[0], "vturn":r[1], "med":r[2], "p90":r[3],
                  "status":r[4], "wall_s":r[5]} for r in results],
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))

    cmap = LinearSegmentedColormap.from_list("nrmse",
        [(0.0,"#1a9850"),(0.30,"#a6d96a"),(0.50,"#ffffbf"),
         (0.70,"#fdae61"),(1.00,"#a50026")])
    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(M, cmap=cmap, vmin=0.6, vmax=1.0, aspect="auto")
    for i in range(len(ETA0_GRID)):
        for j in range(len(VTURN_GRID)):
            v = M[i,j]
            if np.isnan(v): continue
            c = "white" if (v < 0.66 or v > 0.92) else "black"
            w = "bold" if v < 0.654 else "normal"
            ax.text(j, i, f"{v:.3f}", ha="center", va="center",
                    color=c, fontsize=10, weight=w)
    if summary["best_idx"]:
        bi, bj = summary["best_idx"]
        ax.plot(bj, bi, "*", color="cyan", markersize=22,
                markeredgecolor="black", markeredgewidth=1.5,
                label=f"best: {summary['best_value']:.3f} dec")
        ax.legend(loc="upper right")
    ax.set_xticks(range(len(VTURN_GRID)))
    ax.set_xticklabels([f"{v:g}" for v in VTURN_GRID])
    ax.set_yticks(range(len(ETA0_GRID)))
    ax.set_yticklabels([f"{e:g}" for e in ETA0_GRID])
    ax.set_xlabel("V_turn (V) — sigmoid midpoint in Vbe")
    ax.set_ylabel("η_0 — low-Vbe limit (η_final=0)")
    ax.set_title(f"F6.v8: η(Vbe) sigmoid sweep at Bf={BF:.0e}, Va={VA}, "
                 f"k={K_FIXED}\n(F6.v3 baseline 0.654 dec at η=0 constant)")
    plt.colorbar(im, ax=ax, label="log10 RMSE (decades)")
    plt.tight_layout()
    plt.savefig(FIG / "heatmap.png", dpi=150)
    plt.savefig(FIG / "heatmap.pdf")
    plt.close()
    if summary["best_idx"]:
        bi, bj = summary["best_idx"]
        print(f"\n[F6v8] best: med={summary['best_value']:.3f} dec at "
              f"η_0={ETA0_GRID[bi]:g}, V_turn={VTURN_GRID[bj]:g}")
    print(f"[F6v8] heatmap saved. wall: {summary['wall_total_s']:.0f}s")


if __name__ == "__main__":
    main()
