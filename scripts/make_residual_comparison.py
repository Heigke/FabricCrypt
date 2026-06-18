"""Per-row residual comparison: prior optimum (Bf=2e4, Va=100) vs new
optimum (Bf=9000, Va=0.55). Diagnostic per O24 oracles.

Side-by-side heatmaps + Δ-improvement map. Reveals whether VAF tuning
broke the VG1=0.40 cluster or just shifted it.

Output: figures/per_row_residuals/comparison_v1_vs_v4.{png,pdf}
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

ROOT = Path(__file__).resolve().parent.parent
PRIOR = ROOT / "results/z91g_two_model_validation_stage6_bf2e4/predictions.json"
NEW   = ROOT / "results/z91g_two_model_validation_F6v4_bf9000_va0.55/predictions.json"
OUT = ROOT / "figures/per_row_residuals"; OUT.mkdir(parents=True, exist_ok=True)


def parse(path):
    rows = json.loads(path.read_text())
    rows = [r for r in rows if not r.get("skipped") and "log_rmse" in r]
    VG1s = sorted({float(r["VG1"]) for r in rows})
    VG2s = sorted({float(r["VG2"]) for r in rows})
    M = np.full((len(VG1s), len(VG2s)), np.nan)
    for r in rows:
        i = VG1s.index(float(r["VG1"]))
        j = VG2s.index(float(r["VG2"]))
        M[i, j] = r["log_rmse"]
    return VG1s, VG2s, M, rows


VG1_a, VG2_a, M_prior, rows_prior = parse(PRIOR)
VG1_b, VG2_b, M_new,   rows_new   = parse(NEW)

assert VG1_a == VG1_b and VG2_a == VG2_b, "axis mismatch"
VG1, VG2 = VG1_a, VG2_a

fig = plt.figure(figsize=(15, 5.5))
gs = GridSpec(1, 3, figure=fig, hspace=0.3, wspace=0.30,
               left=0.05, right=0.97, top=0.88, bottom=0.10)

def heat(ax, M, title, vmin=0.4, vmax=2.5, cmap="RdYlGn_r"):
    ext = [min(VG2)-0.05, max(VG2)+0.05, min(VG1)-0.05, max(VG1)+0.05]
    im = ax.imshow(M, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax,
                   origin="lower", extent=ext)
    for i, vg1 in enumerate(VG1):
        for j, vg2 in enumerate(VG2):
            v = M[i, j]
            if np.isnan(v): continue
            c = "white" if (v < 0.85 or v > 1.7) else "black"
            ax.text(vg2, vg1, f"{v:.2f}", ha="center", va="center",
                    fontsize=7.5, color=c,
                    weight="bold" if v > 1.5 else "normal")
    ax.set_xlabel("VG2 (V)"); ax.set_ylabel("VG1 (V)")
    ax.set_title(title, fontsize=10)
    return im

ax1 = fig.add_subplot(gs[0])
im1 = heat(ax1, M_prior, "(A) Prior optimum: Bf=2×10⁴, Va=100\n"
            f"median={np.nanmedian(M_prior):.3f}, "
            f"p90={np.nanpercentile(M_prior, 90):.3f} dec")

ax2 = fig.add_subplot(gs[1])
im2 = heat(ax2, M_new, f"(B) New optimum: Bf=9×10³, Va=0.55\n"
            f"median={np.nanmedian(M_new):.3f}, "
            f"p90={np.nanpercentile(M_new, 90):.3f} dec")

# Δ-map: improvement = prior - new (positive = better fit at new params)
delta = M_prior - M_new
ax3 = fig.add_subplot(gs[2])
ext = [min(VG2)-0.05, max(VG2)+0.05, min(VG1)-0.05, max(VG1)+0.05]
v = np.nanmax(np.abs(delta))
im3 = ax3.imshow(delta, aspect="auto", cmap="RdBu_r",
                 vmin=-v, vmax=+v, origin="lower", extent=ext)
for i, vg1 in enumerate(VG1):
    for j, vg2 in enumerate(VG2):
        d = delta[i, j]
        if np.isnan(d): continue
        ax3.text(vg2, vg1, f"{d:+.2f}", ha="center", va="center",
                 fontsize=7.5,
                 color="white" if abs(d) > v*0.5 else "black",
                 weight="bold" if abs(d) > 0.5 else "normal")
ax3.set_xlabel("VG2 (V)"); ax3.set_ylabel("VG1 (V)")
ax3.set_title(f"(C) Δ improvement (A−B): blue=better at new\n"
               f"mean Δ={np.nanmean(delta):+.3f}, "
               f"max Δ={np.nanmax(delta):+.3f}", fontsize=10)
plt.colorbar(im1, ax=ax1, fraction=0.04, pad=0.02, label="dec")
plt.colorbar(im2, ax=ax2, fraction=0.04, pad=0.02, label="dec")
plt.colorbar(im3, ax=ax3, fraction=0.04, pad=0.02, label="Δ dec (+=better)")

# Find worst rows in each
worst_prior = sorted([r for r in rows_prior], key=lambda r: -r["log_rmse"])[:5]
worst_new   = sorted([r for r in rows_new],   key=lambda r: -r["log_rmse"])[:5]
print("Top-5 worst rows (prior Bf=2e4 Va=100):")
for r in worst_prior:
    print(f"  VG1={r['VG1']:.2f} VG2={r['VG2']:.2f}  log_rmse={r['log_rmse']:.3f}")
print("\nTop-5 worst rows (new Bf=9000 Va=0.55):")
for r in worst_new:
    print(f"  VG1={r['VG1']:.2f} VG2={r['VG2']:.2f}  log_rmse={r['log_rmse']:.3f}")

# Cluster check
import numpy as _np
prior_vg1 = _np.array([r["VG1"] for r in worst_prior])
new_vg1   = _np.array([r["VG1"] for r in worst_new])
print(f"\n[cluster prior]: VG1 mean={prior_vg1.mean():.3f}, "
       f"std={prior_vg1.std():.3f}")
print(f"[cluster new]:   VG1 mean={new_vg1.mean():.3f}, "
       f"std={new_vg1.std():.3f}")

fig.suptitle("Per-row residual diagnostic — VAF tuning impact "
              f"(median {np.nanmedian(M_prior):.3f} → "
              f"{np.nanmedian(M_new):.3f} dec)",
              fontsize=11, weight="bold")

plt.savefig(OUT / "comparison_v1_vs_v4.png", dpi=150, bbox_inches="tight")
plt.savefig(OUT / "comparison_v1_vs_v4.pdf", bbox_inches="tight")
plt.close()
print(f"\n[fig] saved {OUT}/comparison_v1_vs_v4.{{png,pdf}}")
