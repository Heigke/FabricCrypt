"""2×2 grid showing the four null sweeps that prove the structural floor.

Visual support for Stage 6 of the brief: each panel shows one of the
post-VAF parameter sweeps that came up null at 0.654 dec.

Output: figures/null_sweeps_quad/null_sweeps.{png,pdf}
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "figures/null_sweeps_quad"; OUT.mkdir(parents=True, exist_ok=True)

cmap = LinearSegmentedColormap.from_list("nrmse",
    [(0.0,"#1a9850"),(0.30,"#a6d96a"),(0.50,"#ffffbf"),
     (0.70,"#fdae61"),(1.00,"#a50026")])

PANELS = [
    {"path": "results/F6v5_ikf_va/summary.json",
     "x_key": "VA_GRID", "y_key": "IKF_GRID",
     "x_lab": "Va (V)", "y_lab": "Ikf (A)",
     "title": "(a) F6.v5 — IKF × Va\n(O24 #2: high-injection knee)",
     "vmin": 0.6, "vmax": 1.6,
     "y_fmt": "{:g}"},
    {"path": "results/F6v6_ise_ne/summary.json",
     "x_key": "NE_GRID", "y_key": "ISE_GRID",
     "x_lab": "Ne", "y_lab": "Ise (A)",
     "title": "(b) F6.v6 — ISE × NE\n(O24 #4: B-E recombination)",
     "vmin": 0.6, "vmax": 1.6,
     "y_fmt": "{:g}"},
    {"path": "results/F6v7_prwg_rdsw/summary.json",
     "x_key": "RDSW_GRID", "y_key": "PRWG_GRID",
     "x_lab": "Rdsw (Ω·µm)", "y_lab": "PRWG",
     "title": "(c) F6.v7 — PRWG × Rdsw\n(O24 #3: S/D Vg-dep S/D resistance)",
     "vmin": 0.65, "vmax": 0.70,
     "y_fmt": "{:g}"},
    {"path": "results/F6v8_eta_sigmoid/summary.json",
     "x_key": "VTURN_GRID", "y_key": "ETA0_GRID",
     "x_lab": "V_turn (V)", "y_lab": "η_0",
     "title": "(d) F6.v8 — η(Vbe) sigmoid\n(O25 #1: bias-dependent collection)",
     "vmin": 0.6, "vmax": 1.0,
     "y_fmt": "{:g}"},
]

fig, axes = plt.subplots(2, 2, figsize=(13, 10))
axes = axes.flatten()

for i, p in enumerate(PANELS):
    d = json.loads(Path(ROOT / p["path"]).read_text())
    M = np.array(d["median_log_rmse"])
    x_grid = d[p["x_key"]]
    y_grid = d[p["y_key"]]
    ax = axes[i]
    im = ax.imshow(M, cmap=cmap, vmin=p["vmin"], vmax=p["vmax"], aspect="auto")
    for ii in range(len(y_grid)):
        for jj in range(len(x_grid)):
            v = M[ii, jj]
            if np.isnan(v): continue
            c = "white" if (v < (p["vmin"]+0.05) or v > (p["vmax"]-0.1)) else "black"
            w = "bold" if v <= 0.66 else "normal"
            ax.text(jj, ii, f"{v:.2f}" if v >= 1.0 else f"{v:.3f}",
                    ha="center", va="center", color=c, fontsize=8, weight=w)
    if "best_idx" in d and d["best_idx"]:
        bi, bj = d["best_idx"]
        ax.plot(bj, bi, "*", color="cyan", markersize=18,
                markeredgecolor="black", markeredgewidth=1.2)
    ax.set_xticks(range(len(x_grid)))
    ax.set_xticklabels([p["y_fmt"].format(v) for v in x_grid],
                        fontsize=8)
    ax.set_yticks(range(len(y_grid)))
    ax.set_yticklabels([p["y_fmt"].format(v) for v in y_grid],
                        fontsize=8)
    ax.set_xlabel(p["x_lab"], fontsize=9)
    ax.set_ylabel(p["y_lab"], fontsize=9)
    ax.set_title(p["title"], fontsize=10, weight="bold")
    plt.colorbar(im, ax=ax, label="median log-RMSE (dec)", fraction=0.045, pad=0.02)

fig.suptitle("Four post-VAF null sweeps — structural floor at 0.654 dec\n"
              "(O24/O25 oracle-ranked candidates each fail to break the floor)",
              fontsize=13, weight="bold")
plt.tight_layout()
plt.savefig(OUT / "null_sweeps.png", dpi=150)
plt.savefig(OUT / "null_sweeps.pdf")
plt.close()
print(f"[fig] saved {OUT}/null_sweeps.{{png,pdf}}")

# Print best per panel for sanity
for p in PANELS:
    d = json.loads(Path(ROOT / p["path"]).read_text())
    print(f"  {p['title'].split(chr(10))[0]}: best={d['best_value']:.3f}")
