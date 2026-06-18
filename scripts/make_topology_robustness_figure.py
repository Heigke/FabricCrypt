"""z142 topology × cross-norm robustness — multi-panel summary.

Reads `results/z142_topology_v2/summary.json` (270 sims = 5 seeds × 6
topologies × 3 N × 3 ρ-norm variants) and renders a single dense
figure showing why ER_SPARSE was the only norm-stable topology.

Layout (4 rows × 6 cols):
  - row 1: MC  (memory capacity)
  - row 2: NARMA-10 NRMSE (lower better)
  - row 3: XOR accuracy
  - row 4: Waveform classification accuracy
  - cols: topologies (RAND_GAUSS, ER_SPARSE, SMALL_WORLD, RING, BIPARTITE, MESH_4N)
  - x-axis per cell: ρ-variant (no_norm, rho_norm, rho_lambda)
  - bars per ρ-variant: median ± IQR over seeds, N as colour

Output: figures/topology_robustness/topology_robustness.{png,pdf}
"""
from __future__ import annotations
import json
from collections import defaultdict
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "results/z142_topology_v2/summary.json"
OUT = ROOT / "figures/topology_robustness"; OUT.mkdir(parents=True, exist_ok=True)

raw = json.loads(DATA.read_text())["results"]

TOPOS = ["RAND_GAUSS", "ER_SPARSE", "WS_SMALLWORLD", "LAYERED", "HUB_SPOKE", "MESH_4N"]
RHOS = ["rho_lambda", "rho_deg_norm", "rho_p95_sv"]
NS = [100, 300, 800]
METRICS = [("MC", "MC", False),
           ("NARMA_NRMSE", "NARMA-10 NRMSE", True),    # lower better
           ("XOR_acc", "XOR accuracy", False),
           ("WAVE_acc", "Wave accuracy", False)]

# Bucket: dict[(topo, rho, N)] = list of metric values across seeds
def bucket(metric):
    out = defaultdict(list)
    for k, r in raw.items():
        if r["topo"] not in TOPOS: continue
        v = r.get(metric)
        if v is None or (isinstance(v, float) and np.isnan(v)): continue
        out[(r["topo"], r["rho_variant"], r["N"])].append(v)
    return out

fig, axes = plt.subplots(len(METRICS), len(TOPOS), figsize=(15, 9),
                          sharex=True, sharey="row")

cmap = plt.cm.viridis
N_colors = {n: cmap(0.15 + 0.35*i) for i, n in enumerate(NS)}

for r, (m_key, m_label, lower_better) in enumerate(METRICS):
    bkt = bucket(m_key)
    # Determine y-limits
    all_vals = [v for vs in bkt.values() for v in vs]
    vmin = max(0, np.min(all_vals)) if not lower_better else np.percentile(all_vals, 5)
    vmax = np.percentile(all_vals, 97) if lower_better else np.max(all_vals)*1.05
    for c, topo in enumerate(TOPOS):
        ax = axes[r, c]
        x_base = np.arange(len(RHOS))
        bar_w = 0.25
        for i_n, N in enumerate(NS):
            xs = x_base + (i_n - 1) * bar_w
            meds, lo, hi = [], [], []
            for rho in RHOS:
                vs = bkt.get((topo, rho, N), [])
                if vs:
                    meds.append(np.median(vs))
                    lo.append(np.percentile(vs, 25))
                    hi.append(np.percentile(vs, 75))
                else:
                    meds.append(np.nan); lo.append(np.nan); hi.append(np.nan)
            meds = np.array(meds); lo = np.array(lo); hi = np.array(hi)
            ax.bar(xs, meds, width=bar_w, color=N_colors[N],
                    yerr=[meds-lo, hi-meds], capsize=2,
                    edgecolor="black", linewidth=0.4,
                    label=f"N={N}" if (r==0 and c==0) else None)
        if r == 0:
            ax.set_title(topo, fontsize=10, weight="bold")
        if c == 0:
            ax.set_ylabel(m_label, fontsize=9)
        if r == len(METRICS)-1:
            ax.set_xticks(x_base)
            ax.set_xticklabels(RHOS, rotation=15, ha="right", fontsize=8)
        ax.set_ylim(vmin, vmax)
        if r == 1:  # NARMA: lower better — flip
            pass
        ax.grid(axis="y", alpha=0.25)

# Legend
handles = [plt.Rectangle((0,0),1,1, color=N_colors[n], label=f"N={n}") for n in NS]
fig.legend(handles=handles, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.0),
            fontsize=10, framealpha=0.9)

fig.suptitle(
    "z142 topology × cross-norm robustness (5 seeds, 4 metrics, 3 sizes) — "
    "ER_SPARSE is the only topology whose ranking holds across all 3 ρ-norm variants",
    fontsize=11, y=1.04)

plt.tight_layout()
plt.savefig(OUT / "topology_robustness.png", dpi=150, bbox_inches="tight")
plt.savefig(OUT / "topology_robustness.pdf", bbox_inches="tight")
plt.close()
print(f"[fig] saved {OUT}/topology_robustness.{{png,pdf}}")

# Print summary for log
print("\n--- ER_SPARSE vs MESH_4N quick stats ---")
for m_key, m_label, _ in METRICS:
    bkt = bucket(m_key)
    print(f"  {m_label}:")
    for topo in ["ER_SPARSE", "MESH_4N", "RAND_GAUSS"]:
        vals_per_rho = []
        for rho in RHOS:
            all_n = []
            for N in NS:
                all_n.extend(bkt.get((topo, rho, N), []))
            if all_n:
                vals_per_rho.append(np.median(all_n))
        if vals_per_rho:
            spread = max(vals_per_rho) - min(vals_per_rho)
            print(f"    {topo:12s}: medians per ρ-variant = "
                   f"{[f'{v:.3f}' for v in vals_per_rho]}  spread={spread:.3f}")
