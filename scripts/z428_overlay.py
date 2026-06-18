"""z428 — Publication overlay plots: Sebas measured IV vs z427 H1+H2 pyport model.

Reuses z427_vsint_fix.py loaders/helpers (build_models, make_cfg, make_bjt,
make_overrides, patch_sd_scaled, forward_2t, load_curves, load_sebas_params,
find_params). Produces:

  results/z427_vsint_fix/overlay_3branches.png         (1x3 combined)
  results/z427_vsint_fix/overlay_VG1_0p2.png           (individual, 200 dpi)
  results/z427_vsint_fix/overlay_VG1_0p4.png
  results/z427_vsint_fix/overlay_VG1_0p6.png
"""
from __future__ import annotations
import math
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

torch.set_default_dtype(torch.float64)

ROOT = Path("/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "nsram"))
sys.path.insert(0, str(ROOT / "scripts"))

import scripts.z427_vsint_fix as z  # noqa: E402

OUT = ROOT / "results" / "z427_vsint_fix"
OUT.mkdir(parents=True, exist_ok=True)

# z427 best-fix flags (cell-wide RMSE 1.733 dec)
COMBINED_FLAGS = {
    "m2_source_Rs": 1.0e6,
    "gidl_route_to_sint": True,
}

# Per-branch RMSE from z427 run.log (COMBINED_H1_H2)
BRANCH_RMSE = {0.2: 2.742, 0.4: 0.560, 0.6: 1.358}
CELL_RMSE = 1.733
BASELINE_RMSE = 3.899

# Representative VG2 picks per VG1 branch
PICKS = [(0.2, 0.30), (0.4, 0.30), (0.6, 0.00)]


def nearest_curve(curves, VG1, VG2):
    """Find curve with this VG1 (exact) and nearest VG2."""
    cands = [c for c in curves if abs(c["VG1"] - VG1) < 1e-3]
    if not cands:
        cands = curves  # fallback
    cands.sort(key=lambda c: abs(c["VG2"] - VG2))
    return cands[0]


def model_predict(model_M1, model_M2, sebas_rows, VG1, VG2, Vd_grid):
    """Build cfg with COMBINED_H1_H2 flags and forward-solve over Vd_grid."""
    cfg, sd_M1, sd_M2 = z.make_cfg(model_M1, model_M2, dict(COMBINED_FLAGS))
    sebas_row = z.find_params(sebas_rows, VG1, VG2)
    if sebas_row is None:
        raise RuntimeError(f"no sebas params for VG1={VG1} VG2={VG2}")
    P_M1, P_M2 = z.make_overrides(sebas_row)
    bjt = z.make_bjt(sebas_row)
    Vd_t = torch.as_tensor(Vd_grid, dtype=torch.float64)
    with torch.no_grad(), z.patch_sd_scaled(sd_M1, P_M1), z.patch_sd_scaled(sd_M2, P_M2):
        out = z.forward_2t(
            cfg, model_M1, bjt,
            Vd_t, torch.tensor(float(VG1)), torch.tensor(float(VG2)),
            model_M1=model_M1, model_M2=model_M2,
            warm_start=True, use_homotopy=True,
        )
    Id = out["Id"].abs().detach().cpu().numpy()
    conv = np.array([bool(x) for x in out["converged"]])
    return Id, conv


def plot_branch(ax, VG1, VG2, curves, model_M1, model_M2, sebas_rows):
    meas = nearest_curve(curves, VG1, VG2)
    actual_VG2 = meas["VG2"]
    Vd_meas = meas["Vd"].numpy()
    Id_meas = meas["Id"].numpy()

    # Dense model grid
    Vd_max = float(max(Vd_meas))
    Vd_grid = np.arange(0.05, Vd_max + 1e-6, 0.05)
    if len(Vd_grid) < 5:
        Vd_grid = np.linspace(0.05, Vd_max, 30)

    print(f"  VG1={VG1} VG2={actual_VG2} (req {VG2}) — Vd∈[{Vd_grid[0]:.2f},{Vd_grid[-1]:.2f}] "
          f"({len(Vd_grid)} pts)")

    try:
        Id_model, conv = model_predict(
            model_M1, model_M2, sebas_rows, VG1, actual_VG2, Vd_grid)
    except Exception as e:
        print(f"    model failed: {e}")
        Id_model = np.full_like(Vd_grid, np.nan)
        conv = np.zeros_like(Vd_grid, dtype=bool)

    rmse_branch = BRANCH_RMSE.get(round(VG1, 1), float("nan"))

    ax.semilogy(Vd_meas, np.abs(Id_meas), "o", color="black",
                markersize=6, label="measured (Sebas 130 nm)",
                alpha=0.75, markeredgewidth=0)
    if np.any(conv):
        ax.semilogy(Vd_grid[conv], np.abs(Id_model[conv]), "-",
                    color="crimson", linewidth=2.2, label="pyport H1+H2 model")
        if np.any(~conv):
            ax.semilogy(Vd_grid[~conv], np.abs(Id_model[~conv]), "--",
                        color="crimson", linewidth=1.2, alpha=0.4,
                        label="model (non-converged)")
    else:
        ax.semilogy(Vd_grid, np.abs(Id_model), "-",
                    color="crimson", linewidth=2.2, label="pyport H1+H2 model")

    ax.set_xlabel(r"$V_D$ [V]", fontsize=11)
    ax.set_ylabel(r"$|I_D|$ [A]", fontsize=11)
    ax.set_title(
        f"VG1 = {VG1:.2f} V, VG2 = {actual_VG2:.2f} V "
        f"(branch RMSE = {rmse_branch:.2f} dec)",
        fontsize=11)
    ax.set_ylim(1e-12, 1e-3)
    ax.set_xlim(0.0, max(2.05, Vd_max + 0.05))
    ax.grid(True, which="both", linestyle=":", alpha=0.5)
    ax.legend(loc="lower right", fontsize=9, framealpha=0.92)
    return actual_VG2


def main():
    print("z428 — building models and loading data...")
    model_M1, model_M2 = z.build_models()
    curves = z.load_curves()
    sebas_rows = z.load_sebas_params()
    print(f"  loaded {len(curves)} curves, {len(sebas_rows)} sebas rows")

    # Combined 1x3 figure
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.0))
    actual_picks = []
    for ax, (VG1, VG2) in zip(axes, PICKS):
        actual_VG2 = plot_branch(ax, VG1, VG2, curves, model_M1, model_M2, sebas_rows)
        actual_picks.append((VG1, actual_VG2))

    fig.suptitle(
        f"z427 H1+H2 (Sint→GND shunt + GIDL→Sint routing): "
        f"cell-wide RMSE = {CELL_RMSE:.3f} dec  (baseline = {BASELINE_RMSE:.3f} dec)",
        fontsize=12, y=1.00)
    plt.tight_layout(rect=(0, 0, 1, 0.96))
    combined_path = OUT / "overlay_3branches.png"
    fig.savefig(combined_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {combined_path}")

    # Individual high-res figures
    written = [combined_path]
    for (VG1, VG2_req), (VG1_a, VG2_a) in zip(PICKS, actual_picks):
        figi, axi = plt.subplots(1, 1, figsize=(6.5, 5.2))
        plot_branch(axi, VG1, VG2_req, curves, model_M1, model_M2, sebas_rows)
        figi.suptitle(
            f"Sebas 130 nm 2T-cell — measured vs pyport H1+H2 model\n"
            f"(cell-wide RMSE {CELL_RMSE:.3f} dec, baseline {BASELINE_RMSE:.3f} dec)",
            fontsize=10.5)
        plt.tight_layout(rect=(0, 0, 1, 0.94))
        tag = f"{VG1:.1f}".replace(".", "p")
        path_i = OUT / f"overlay_VG1_{tag}.png"
        figi.savefig(path_i, dpi=200, bbox_inches="tight")
        plt.close(figi)
        written.append(path_i)
        print(f"wrote {path_i}")

    # Verify
    print("\nVerification:")
    ok = True
    for p in written:
        if not p.exists():
            print(f"  MISSING: {p}")
            ok = False; continue
        sz = p.stat().st_size
        flag = "OK" if sz > 1024 else "TOO SMALL"
        print(f"  {flag}  {p}  ({sz} bytes)")
        if sz <= 1024:
            ok = False
    if not ok:
        sys.exit(2)
    print("\nAll 4 PNGs created and non-empty.")


if __name__ == "__main__":
    main()
