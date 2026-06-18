"""S4-E: Physical Vsint→GND shunt Rs at M2.source.

Tests cfg.m2_source_Rs ∈ {1, 5, 10, 100, 1000} Ω at VG1=0.6, VG2=0.2.
For each Rs: full Vd sweep with regular 2D Newton. Check Vsint stays low AND
fold appears.
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

from nsram.bsim4_port.nsram_cell_2T import forward_2t

OUT = ROOT / "results/z394_rs_shunt"; OUT.mkdir(parents=True, exist_ok=True)

VG1, VG2 = 0.6, 0.2
ETAB = 20.0
RS_VALUES = [1.0, 5.0, 10.0, 100.0, 1000.0]


def run_with_Rs(cfg, M1, M2, bjt, rows, Rs_val):
    _, iii, vnwell_Rs = PER_VG1[VG1]
    cfg.iii_body_gain = iii
    cfg.vnwell_Rs = vnwell_Rs
    cfg.m2_source_Rs = float(Rs_val)
    sd_M1 = cfg.size_dep_M1(M1); sd_M2 = cfg.size_dep_M2(M2)
    Vd_m, Id_m, _ = load_measured(VG1, VG2)
    row = find_or_impute_row(rows, VG1, VG2)
    P_M1, P_M2 = make_overrides(row, etab_override=ETAB)
    Vd_t = torch.tensor(Vd_m, dtype=torch.float64)
    with patch_sd_scaled(sd_M1, P_M1), patch_sd_scaled(sd_M2, P_M2):
        out = forward_2t(cfg, model_M1=M1, model_M2=M2, bjt=bjt, Vd_seq=Vd_t,
                         VG1=torch.tensor(VG1, dtype=torch.float64),
                         VG2=torch.tensor(VG2, dtype=torch.float64),
                         warm_start=True)
    Vsint = out["Vsint"].detach().to(torch.float64).cpu().numpy().reshape(-1)
    Vb = out["Vb"].detach().to(torch.float64).cpu().numpy().reshape(-1)
    Id = np.abs(out["Id"].detach().to(torch.float64).cpu().numpy().reshape(-1))
    has_nan = bool(np.any(~np.isfinite(Id)))
    dlog = np.diff(np.log10(np.maximum(Id, 1e-15)))
    Vmid = 0.5 * (Vd_m[1:] + Vd_m[:-1])
    fold = float(dlog[Vmid >= 0.5].max()) if (Vmid >= 0.5).any() else float("nan")
    return {"Vd": Vd_m, "Id": Id, "Id_m": Id_m, "Vsint": Vsint, "Vb": Vb,
            "fold_dec": fold, "has_nan": has_nan}


def main():
    t0 = time.time()
    cfg, M1, M2, bjt = build_base()
    rows = load_sebas_params()
    results = {}
    for Rs in RS_VALUES:
        print(f"Rs={Rs}Ω", flush=True)
        results[Rs] = run_with_Rs(cfg, M1, M2, bjt, rows, Rs)

    Vd_m = results[RS_VALUES[0]]["Vd"]
    Id_m = results[RS_VALUES[0]]["Id_m"]
    dlog_m = np.diff(np.log10(np.maximum(Id_m, 1e-15)))
    Vmid_m = 0.5 * (Vd_m[1:] + Vd_m[:-1])
    meas_fold = float(dlog_m[Vmid_m >= 0.5].max())

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    ax = axes[0]
    for Rs, d in results.items():
        ax.semilogy(d["Vd"], d["Id"] + 1e-15, marker="o", markersize=3,
                    label=f"Rs={Rs:g}Ω (fold={d['fold_dec']:.2f}dec)")
    ax.semilogy(Vd_m, np.abs(Id_m) + 1e-15, "k--", lw=2,
                label=f"meas (fold={meas_fold:.2f}dec)")
    ax.set_xlabel("Vd [V]"); ax.set_ylabel("|Id| [A]")
    ax.set_title(f"Ids(Vd) vs Rs shunt, VG1={VG1}, VG2={VG2}")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    ax = axes[1]
    for Rs, d in results.items():
        ax.plot(d["Vd"], d["Vsint"], marker="o", markersize=3,
                label=f"Rs={Rs:g}Ω")
    ax.set_xlabel("Vd [V]"); ax.set_ylabel("Vsint converged [V]")
    ax.set_title("Vsint (should → 0 for small Rs)")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    ax = axes[2]
    for Rs, d in results.items():
        ax.plot(d["Vd"], d["Vb"], marker="o", markersize=3,
                label=f"Rs={Rs:g}Ω")
    ax.set_xlabel("Vd [V]"); ax.set_ylabel("Vb converged [V]")
    ax.set_title("Vb converged")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    fig.tight_layout()
    fpath = OUT / "ids_vs_rs.png"
    fig.savefig(fpath, dpi=120); plt.close(fig)

    summary = {
        "VG1": VG1, "VG2": VG2, "etab": ETAB, "Rs_values": RS_VALUES,
        "meas_fold_dec": meas_fold,
        "per_Rs": {
            str(Rs): {"fold_dec": d["fold_dec"], "has_nan": d["has_nan"],
                      "Vsint_max": float(np.max(d["Vsint"])),
                      "Vsint_at_Vd_1p5": float(d["Vsint"][np.argmin(np.abs(d["Vd"]-1.5))]),
                      "Vb_at_Vd_1p5": float(d["Vb"][np.argmin(np.abs(d["Vd"]-1.5))]),
                      "Id_at_Vd_1p5": float(d["Id"][np.argmin(np.abs(d["Vd"]-1.5))])}
            for Rs, d in results.items()
        },
        "elapsed_s": time.time() - t0, "plot": str(fpath),
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary["per_Rs"], indent=2))
    print(f"meas fold = {meas_fold:.2f} dec; elapsed={time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
