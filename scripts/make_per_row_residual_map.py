"""Per-row residual map at Bf=2e4, Is=1e-9 (F1.v2 optimum).

Reads `results/z91g_two_model_validation_stage6_bf2e4/predictions.json`
(33 rows × Vd-sweep) and renders the per-row log-RMSE as a (VG1, VG2)
heatmap, plus the top-5 worst rows' Vd-traces overlaid (model vs Sebas).

Diagnostic per O24 oracle audit: if the worst rows cluster in a specific
(VG1, VG2) corner, that signals VAF/IKF (output-conductance corner) or
DIBL (low-VG2/high-Vd corner). Random scatter signals a numerics issue.

Output: figures/per_row_residuals/per_row_residuals.{png,pdf}
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "results/z91g_two_model_validation_F6v4_bf9000_va0.55/predictions.json"
OUT = ROOT / "figures/per_row_residuals_optimum"; OUT.mkdir(parents=True, exist_ok=True)

rows = json.loads(DATA.read_text())
rows = [r for r in rows if not r.get("skipped") and "log_rmse" in r]
print(f"[diag] {len(rows)} usable rows (8 skipped); median log_rmse="
      f"{np.median([r['log_rmse'] for r in rows]):.3f}")

# Build (VG1, VG2) → log_rmse map
VG1s = sorted({float(r["VG1"]) for r in rows})
VG2s = sorted({float(r["VG2"]) for r in rows})
M = np.full((len(VG1s), len(VG2s)), np.nan)
for r in rows:
    if r["skipped"]: continue
    i = VG1s.index(float(r["VG1"]))
    j = VG2s.index(float(r["VG2"]))
    M[i, j] = r["log_rmse"]

# Worst rows for the trace overlay
sorted_rows = sorted([r for r in rows if not r["skipped"]],
                     key=lambda r: -r["log_rmse"])
worst5 = sorted_rows[:5]
best5 = sorted_rows[-5:]

# ─────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(15, 6.5))
gs = fig.add_gridspec(2, 3, width_ratios=[1.3, 1.0, 1.0],
                      height_ratios=[1, 1], hspace=0.35, wspace=0.32,
                      left=0.06, right=0.98, top=0.92, bottom=0.10)

# Panel A: VG1×VG2 heatmap
ax = fig.add_subplot(gs[:, 0])
im = ax.imshow(M, aspect="auto", cmap="RdYlGn_r",
                vmin=0.4, vmax=2.5, origin="lower",
                extent=[min(VG2s)-0.05, max(VG2s)+0.05,
                        min(VG1s)-0.05, max(VG1s)+0.05])
for i, vg1 in enumerate(VG1s):
    for j, vg2 in enumerate(VG2s):
        v = M[i, j]
        if np.isnan(v): continue
        ax.text(vg2, vg1, f"{v:.2f}", ha="center", va="center",
                fontsize=8, color="white" if (v < 0.85 or v > 1.7) else "black",
                weight="bold" if v > 1.5 else "normal")
ax.set_xlabel("VG2 (V)", fontsize=10)
ax.set_ylabel("VG1 (V)", fontsize=10)
ax.set_title("(A) Per-row log-RMSE — F1.v2 fit at Bf=2×10⁴, Is=1×10⁻⁹\n"
              f"median={np.nanmedian(M):.3f} dec, p90={np.nanpercentile(M, 90):.3f} dec",
              fontsize=10)
cbar = plt.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
cbar.set_label("log10 RMSE (decades)", fontsize=9)
cbar.ax.axhline(1.0, color="black", lw=1.0, ls="--")

# Panel B-top: 5 worst rows traces (model vs measured)
ax = fig.add_subplot(gs[0, 1:])
for r in worst5:
    Vd = np.array(r["Vd"])
    Id_meas = np.log10(np.maximum(np.abs(np.array(r["Id_meas"])), 1e-15))
    Id_pred = np.log10(np.maximum(np.abs(np.array(r["Id_pred"])), 1e-15))
    label = f"VG1={r['VG1']:.2f}, VG2={r['VG2']:.2f}  ε={r['log_rmse']:.2f}"
    line, = ax.plot(Vd, Id_meas, "-", lw=1.3, label=label)
    ax.plot(Vd, Id_pred, "--", lw=1.0, color=line.get_color(), alpha=0.7)
ax.set_xlabel("Vd (V)", fontsize=9)
ax.set_ylabel("log₁₀|Id| (A)", fontsize=9)
ax.set_title("(B) Top-5 WORST rows: solid=Sebas, dashed=model",
             fontsize=10, weight="bold")
ax.legend(fontsize=7, loc="lower right", ncol=2)
ax.grid(alpha=0.25)

# Panel B-bot: 5 best rows traces
ax = fig.add_subplot(gs[1, 1:])
for r in best5:
    Vd = np.array(r["Vd"])
    Id_meas = np.log10(np.maximum(np.abs(np.array(r["Id_meas"])), 1e-15))
    Id_pred = np.log10(np.maximum(np.abs(np.array(r["Id_pred"])), 1e-15))
    label = f"VG1={r['VG1']:.2f}, VG2={r['VG2']:.2f}  ε={r['log_rmse']:.2f}"
    line, = ax.plot(Vd, Id_meas, "-", lw=1.3, label=label)
    ax.plot(Vd, Id_pred, "--", lw=1.0, color=line.get_color(), alpha=0.7)
ax.set_xlabel("Vd (V)", fontsize=9)
ax.set_ylabel("log₁₀|Id| (A)", fontsize=9)
ax.set_title("(C) Top-5 BEST rows", fontsize=10, weight="bold")
ax.legend(fontsize=7, loc="lower right", ncol=2)
ax.grid(alpha=0.25)

fig.suptitle(
    "F1.v2 fit residual diagnostic — diagnostic for next 2D sweep target (per O24 oracles)",
    fontsize=11, weight="bold")

plt.savefig(OUT / "per_row_residuals.png", dpi=150, bbox_inches="tight")
plt.savefig(OUT / "per_row_residuals.pdf", bbox_inches="tight")
plt.close()
print(f"[fig] saved {OUT}/per_row_residuals.{{png,pdf}}")

# Worst-row diagnosis
print("\n--- Top-5 worst rows ---")
for r in worst5:
    print(f"  VG1={r['VG1']:.2f}  VG2={r['VG2']:.2f}  log_rmse={r['log_rmse']:.3f}")
print("\n--- Top-5 best rows ---")
for r in best5:
    print(f"  VG1={r['VG1']:.2f}  VG2={r['VG2']:.2f}  log_rmse={r['log_rmse']:.3f}")

# Cluster diagnosis: are worst rows in a specific (VG1, VG2) corner?
worst_vg1 = np.array([r["VG1"] for r in worst5])
worst_vg2 = np.array([r["VG2"] for r in worst5])
print(f"\n[cluster] worst-5 VG1 mean={worst_vg1.mean():.3f}, std={worst_vg1.std():.3f}")
print(f"[cluster] worst-5 VG2 mean={worst_vg2.mean():.3f}, std={worst_vg2.std():.3f}")
all_vg1 = np.array([r["VG1"] for r in rows if not r["skipped"]])
all_vg2 = np.array([r["VG2"] for r in rows if not r["skipped"]])
print(f"[cluster] all-row VG1 mean={all_vg1.mean():.3f}, std={all_vg1.std():.3f}")
print(f"[cluster] all-row VG2 mean={all_vg2.mean():.3f}, std={all_vg2.std():.3f}")
