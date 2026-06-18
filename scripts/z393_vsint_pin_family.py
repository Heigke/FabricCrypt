"""S4-B: Pinned-Vsint family sweep at VG1=0.6, VG2=0.2.

For each Vsint_pin ∈ {0.0, 0.05, 0.10, 0.15, 0.20}:
  for each Vd ∈ linspace(0, 2, 41):
     1D Newton on Vb only (Vsint frozen). Record Ids_M1 (which is what
     flows from drain through M1 since Vsint is held fixed).
Compare to measured fold at VG1=0.6 (jump ≈ 2.20 dec).
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
                          patch_sd_scaled, PER_VG1, load_measured)

from nsram.bsim4_port.nsram_cell_2T import _residuals

OUT = ROOT / "results/z393_vsint_pin_family"; OUT.mkdir(parents=True, exist_ok=True)

VG1, VG2 = 0.6, 0.2
ETAB = 20.0
VSINT_PINS = [0.0, 0.05, 0.10, 0.15, 0.20]
VD_GRID = np.linspace(0.0, 2.0, 41)
H_FD = 1e-5
NEWTON_ITERS = 60
TOL = 1e-12


def newton_1d_on_Vb(cfg, M1, M2, bjt, Vd_t, VG1_t, VG2_t, vsint_t, vb_init,
                    P_M1_residual, P_M2_residual):
    """1D Newton on Vb only — Vsint frozen. Uses R_B residual."""
    vb = torch.tensor(vb_init, dtype=torch.float64)
    for _ in range(NEWTON_ITERS):
        with torch.no_grad():
            _, R_B, comp = _residuals(cfg, M1, bjt, Vd_t, VG1_t, VG2_t,
                                      vsint_t, vb, P_M1_residual, P_M2_residual, model_M2=M2)
            _, R_B_p, _ = _residuals(cfg, M1, bjt, Vd_t, VG1_t, VG2_t,
                                     vsint_t, vb + H_FD, P_M1_residual, P_M2_residual, model_M2=M2)
            _, R_B_m, _ = _residuals(cfg, M1, bjt, Vd_t, VG1_t, VG2_t,
                                     vsint_t, vb - H_FD, P_M1_residual, P_M2_residual, model_M2=M2)
            dRb = (R_B_p - R_B_m) / (2 * H_FD)
            if float(R_B.abs()) < TOL:
                break
            if float(dRb.abs()) < 1e-30:
                break
            dvb = -R_B / dRb
            # cap step
            if float(dvb.abs()) > 0.1:
                dvb = dvb * (0.1 / float(dvb.abs()))
            vb = (vb + 0.5 * dvb).clamp(-0.2, 1.5)
    return float(vb), float(R_B), comp


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
    # _residuals's _override_sd is setattr-based; sd.scaled overrides apply
    # via patch_sd_scaled. Don't double-apply.
    P_M1_residual = None
    P_M2_residual = None

    VG1_t = torch.tensor(VG1, dtype=torch.float64)
    VG2_t = torch.tensor(VG2, dtype=torch.float64)

    # Load measured for overlay
    Vd_m, Id_m, _ = load_measured(VG1, VG2)

    family = {}
    with patch_sd_scaled(sd_M1, P_M1), patch_sd_scaled(sd_M2, P_M2):
        for vsint_pin in VSINT_PINS:
            vs_t = torch.tensor(vsint_pin, dtype=torch.float64)
            ids_list = []; vb_list = []
            vb_warm = 0.0
            for vd in VD_GRID:
                Vd_t = torch.tensor(vd, dtype=torch.float64)
                vb_conv, rb_final, comp = newton_1d_on_Vb(
                    cfg, M1, M2, bjt, Vd_t, VG1_t, VG2_t, vs_t, vb_warm,
                    P_M1_residual, P_M2_residual)
                vb_warm = vb_conv  # warm start next
                # Ids from drain = Ids_M1 (which flows D→Sint through M1)
                ids_list.append(float(comp["Ids_M1"]))
                vb_list.append(vb_conv)
            family[vsint_pin] = {"Ids_M1": ids_list, "Vb": vb_list}

    # Compute fold (max d log10 Ids / d Vd) for each pin
    fold = {}
    for vsint_pin, d in family.items():
        ids = np.abs(np.array(d["Ids_M1"]))
        dlog = np.diff(np.log10(np.maximum(ids, 1e-15)))
        Vmid = 0.5 * (VD_GRID[1:] + VD_GRID[:-1])
        valid = Vmid >= 0.5
        if valid.any():
            fold[vsint_pin] = float(dlog[valid].max())
        else:
            fold[vsint_pin] = float("nan")

    # Measured fold for comparison
    dlog_m = np.diff(np.log10(np.maximum(Id_m, 1e-15)))
    Vmid_m = 0.5 * (Vd_m[1:] + Vd_m[:-1])
    meas_fold = float(dlog_m[Vmid_m >= 0.5].max()) if (Vmid_m >= 0.5).any() else float("nan")

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    ax = axes[0]
    for vsint_pin, d in family.items():
        ids = np.abs(np.array(d["Ids_M1"]))
        ax.semilogy(VD_GRID, ids + 1e-15, marker="o", markersize=3,
                    label=f"Vsint_pin={vsint_pin:.2f}V (fold={fold[vsint_pin]:.2f}dec)")
    ax.semilogy(Vd_m, np.abs(Id_m) + 1e-15, "k--", lw=2,
                label=f"meas (fold={meas_fold:.2f}dec)")
    ax.set_xlabel("Vd [V]"); ax.set_ylabel("|Ids_M1| [A]")
    ax.set_title(f"Pinned-Vsint family, VG1={VG1}, VG2={VG2}")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    ax = axes[1]
    for vsint_pin, d in family.items():
        ax.plot(VD_GRID, d["Vb"], marker="o", markersize=3,
                label=f"Vsint_pin={vsint_pin:.2f}V")
    ax.set_xlabel("Vd [V]"); ax.set_ylabel("Vb converged [V]")
    ax.set_title("Vb at converged 1D Newton")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    fig.tight_layout()
    fpath = OUT / "ids_vs_vd_family.png"
    fig.savefig(fpath, dpi=120); plt.close(fig)

    summary = {
        "VG1": VG1, "VG2": VG2, "etab": ETAB,
        "Vsint_pins": VSINT_PINS, "Vd_grid": VD_GRID.tolist(),
        "fold_per_pin_dec": fold, "meas_fold_dec": meas_fold,
        "family": {str(k): v for k, v in family.items()},
        "elapsed_s": time.time() - t0, "plot": str(fpath),
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    # Brief printout
    print(f"meas fold = {meas_fold:.2f} dec")
    for k, v in fold.items():
        print(f"Vsint_pin={k}: fold={v:.3f} dec")
    print(f"elapsed = {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
