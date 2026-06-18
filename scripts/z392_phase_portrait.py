"""S4-A: 2D phase portrait of (Vsint, Vb) residuals at VG1=0.6, VG2=0.2, Vd=1.5V.

Visualises ALL roots of (R_Sint=0) ∩ (R_B=0) on a 50x50 grid.
Uses R-46 best params (PER_VG1[0.6]) via _z384_shared build_base() + run config.
"""
from __future__ import annotations
import sys, json, time
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _z384_shared import (ROOT, build_base, load_sebas_params,
                          find_or_impute_row, make_overrides,
                          patch_sd_scaled, PER_VG1)

from nsram.bsim4_port.nsram_cell_2T import _residuals

OUT = ROOT / "results/z392_phase_portrait"; OUT.mkdir(parents=True, exist_ok=True)

VG1, VG2, VD = 0.6, 0.2, 1.5
ETAB = 20.0  # match S3-B / z388 config

N = 50
VSINT_RANGE = (0.0, 0.3)
VB_RANGE = (0.0, 0.9)


def main():
    t0 = time.time()
    cfg, M1, M2, bjt = build_base()
    rows = load_sebas_params()
    _, iii, Rs = PER_VG1[VG1]
    cfg.iii_body_gain = iii
    cfg.vnwell_Rs = Rs
    sd_M1 = cfg.size_dep_M1(M1); sd_M2 = cfg.size_dep_M2(M2)

    row = find_or_impute_row(rows, VG1, VG2)
    P_M1, P_M2 = make_overrides(row, etab_override=ETAB)
    # _residuals's _override_sd uses setattr (legacy fitting path), not the
    # sd.scaled dict — so apply via patch_sd_scaled and pass None to _residuals.
    P_M1_residual = None
    P_M2_residual = None

    vs = np.linspace(*VSINT_RANGE, N)
    vb = np.linspace(*VB_RANGE, N)
    VS_grid, VB_grid = np.meshgrid(vs, vb, indexing="xy")

    Vd_t = torch.tensor(VD, dtype=torch.float64)
    VG1_t = torch.tensor(VG1, dtype=torch.float64)
    VG2_t = torch.tensor(VG2, dtype=torch.float64)

    R_S = np.zeros_like(VS_grid); R_B = np.zeros_like(VS_grid)
    Ids_M1 = np.zeros_like(VS_grid)

    with patch_sd_scaled(sd_M1, P_M1), patch_sd_scaled(sd_M2, P_M2), torch.no_grad():
        for i in range(N):
            for j in range(N):
                vs_t = torch.tensor(VS_grid[i, j], dtype=torch.float64)
                vb_t = torch.tensor(VB_grid[i, j], dtype=torch.float64)
                rs_v, rb_v, comp = _residuals(cfg, M1, bjt, Vd_t, VG1_t, VG2_t,
                                              vs_t, vb_t, P_M1_residual, P_M2_residual, model_M2=M2)
                R_S[i, j] = float(rs_v); R_B[i, j] = float(rb_v)
                Ids_M1[i, j] = float(comp["Ids_M1"])

    # Find roots: cells where both |R_S| and |R_B| are local minima of |R_S|^2+|R_B|^2
    mag = np.log10(np.maximum(np.abs(R_S)**2 + np.abs(R_B)**2, 1e-40))
    # Approximate roots: where both contours change sign within neighbours
    sign_S = np.sign(R_S)
    sign_B = np.sign(R_B)
    candidate = np.zeros_like(R_S, dtype=bool)
    for i in range(1, N-1):
        for j in range(1, N-1):
            s_change = (sign_S[i, j] != sign_S[i+1, j]) or (sign_S[i, j] != sign_S[i, j+1])
            b_change = (sign_B[i, j] != sign_B[i+1, j]) or (sign_B[i, j] != sign_B[i, j+1])
            if s_change and b_change:
                candidate[i, j] = True
    cand_pts = np.argwhere(candidate)
    roots = [(float(VS_grid[i, j]), float(VB_grid[i, j]), float(Ids_M1[i, j]))
             for i, j in cand_pts]

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    ax = axes[0]
    im = ax.pcolormesh(VS_grid, VB_grid, mag, cmap="viridis", shading="auto")
    plt.colorbar(im, ax=ax, label="log10(|R_S|² + |R_B|²)")
    ax.contour(VS_grid, VB_grid, R_S, levels=[0], colors="cyan", linewidths=1.5)
    ax.contour(VS_grid, VB_grid, R_B, levels=[0], colors="magenta", linewidths=1.5)
    for vs_r, vb_r, _ in roots:
        ax.plot(vs_r, vb_r, "w*", markersize=12, markeredgecolor="red")
    ax.set_xlabel("Vsint [V]"); ax.set_ylabel("Vb [V]")
    ax.set_title(f"Residual heatmap + zero contours\nVG1={VG1}, VG2={VG2}, Vd={VD}V (etab={ETAB})\ncyan=R_Sint=0, magenta=R_B=0")

    ax = axes[1]
    Ids_clip = np.where(Ids_M1 > 0, np.log10(Ids_M1 + 1e-30), -30)
    im2 = ax.pcolormesh(VS_grid, VB_grid, Ids_clip, cmap="plasma", shading="auto",
                        vmin=-12, vmax=-3)
    plt.colorbar(im2, ax=ax, label="log10(|Ids_M1|) [A]")
    for vs_r, vb_r, ids_r in roots:
        ax.plot(vs_r, vb_r, "w*", markersize=12, markeredgecolor="red")
        ax.annotate(f"{ids_r:.1e}A", (vs_r, vb_r), color="white", fontsize=8)
    ax.set_xlabel("Vsint [V]"); ax.set_ylabel("Vb [V]")
    ax.set_title(f"|Ids_M1| at each (Vsint,Vb)\nstars = candidate roots")

    fig.tight_layout()
    fpath = OUT / "phase_portrait_VG1_0p6.png"
    fig.savefig(fpath, dpi=120); plt.close(fig)

    summary = {
        "VG1": VG1, "VG2": VG2, "Vd": VD, "etab": ETAB,
        "grid": [N, N], "Vsint_range": VSINT_RANGE, "Vb_range": VB_RANGE,
        "n_candidate_roots": len(roots),
        "roots": [{"Vsint": r[0], "Vb": r[1], "Ids_M1": r[2]} for r in roots],
        "elapsed_s": time.time() - t0,
        "plot": str(fpath),
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
