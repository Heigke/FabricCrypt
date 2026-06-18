"""A1g — multi-root diagnostic for low-VG2 Id under-shoot.

Hypothesis: at low VG2 (VG2~0, VG1=0.4 or 0.6, Vd=1.5), z91g converges
to a low-Vb root (parasitic NPN OFF, Id ~1e-11) while measurement
shows a high-Vb / NPN-firing root (Id ~ 2e-5).

This script:
  1. Loads M1 + M2 cards, applies z91f patch_model_values.
  2. Builds Sebas-CSV per-bias overrides (P_M1, P_M2) + BJT.
  3. At each diagnostic bias (VG1, VG2=0.0, Vd=1.5):
       a. solve with Vb_init=0 (legacy)            -> "low-init"
       b. solve with Vb_init=0.5                  -> "vb05"
       c. solve with Vb_init=0.7                  -> "vb07"
       d. solve with Vb_init=0.9                  -> "high-init"
       e. arclength sweep Vd in [0.05, 2.0]       -> "arclength"
  4. Compare to measurement at Vd~=1.5.
  5. Save trace JSON + arclength PNG.
"""
from __future__ import annotations
import json, math, time, sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

torch.set_default_dtype(torch.float64)

ROOT = Path("/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy")
DATA = ROOT / "data/sebas_2026_04_22"
OUT = ROOT / "research_plan/artifacts"
OUT.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(ROOT))

from nsram.bsim4_port.model_card import BSIM4Model
from nsram.bsim4_port.nsram_cell_2T import (
    NSRAMCell2TConfig, solve_2t_with_homotopy, solve_2t_steady_state,
)
from nsram.bsim4_port.arclength import forward_2t_arclength_grad
from nsram.bsim4_port.temp import compute_size_dep
from nsram.bsim4_port.geometry import Geometry

# Reuse z91f helpers
import importlib.util
_spec = importlib.util.spec_from_file_location(
    "z91f_mod", ROOT / "scripts/z91f_validate_with_sebas_params.py")
z91f = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(z91f)
patch_model_values = z91f.patch_model_values
patch_sd_scaled = z91f.patch_sd_scaled
load_sebas_params = z91f.load_sebas_params
find_params = z91f.find_params
make_overrides = z91f.make_overrides
make_bjt = z91f.make_bjt
M2_STATIC_OVERRIDES = z91f.M2_STATIC_OVERRIDES


def load_meas_at_vd(VG1, VG2, Vd_target=1.5):
    """Read the measured Id at Vd~=Vd_target for the given (VG1, VG2)."""
    sub = DATA / f"2vHCa-2 I-Vs@VG2 VG1={VG1} vnwell=2"
    fname = f"StandardIV_HH_2vHCa-2_VG2={VG2:.2f}_VG=0.6"
    # Find any matching file
    if VG1 == 0.4:
        fname = fname.replace("VG=0.6", "VG=0.4")
    matches = sorted(sub.glob(f"StandardIV*VG2={VG2:.2f}*.csv"))
    if not matches:
        return None, None
    f = matches[0]
    data = np.loadtxt(f, delimiter=",", skiprows=1, usecols=(0, 1))
    half = len(data) // 2
    Vd = data[:half, 0]
    Id = np.abs(data[:half, 1])
    idx = int(np.argmin(np.abs(Vd - Vd_target)))
    return float(Vd[idx]), float(Id[idx])


def m1_m2_overrides_for_z91g(sebas_row):
    """Replicates z91g's override logic: drop k1/k2/etab/beta0 from P_M2
    (they are baked into model_M2 card), keep NFACTOR only."""
    P_M1, P_M2 = make_overrides(sebas_row)
    if P_M2:
        for k in ("k1", "k2", "etab", "beta0"):
            P_M2.pop(k, None)
        if not P_M2:
            P_M2 = None
    return P_M1, P_M2


def run_one_bias(cfg, model_M1, model_M2, sd_M1, sd_M2, bjt,
                 P_M1, P_M2, VG1, VG2, Vd, Vb_init):
    """Run solve_2t_with_homotopy at a single Vd with given Vb_init."""
    Vd_t = torch.tensor([Vd], dtype=torch.float64)
    VG1_t = torch.tensor(VG1, dtype=torch.float64)
    VG2_t = torch.tensor(VG2, dtype=torch.float64)
    Vsint_init = torch.tensor([0.5 * Vd], dtype=torch.float64)
    Vb_init_t = torch.tensor([Vb_init], dtype=torch.float64)
    # NOTE: do NOT pass P_M1/P_M2 to solve_2t_*; z91g relies entirely on
    # patch_sd_scaled (sd.scaled[k]) — _override_sd uses getattr which
    # fails on dict-only entries like etab/alpha0/beta0/nfactor.
    with torch.no_grad(), \
         patch_sd_scaled(sd_M1, P_M1), \
         patch_sd_scaled(sd_M2, P_M2):
        out = solve_2t_with_homotopy(
            cfg, model_M1, bjt,
            Vd=Vd_t, VG1=VG1_t, VG2=VG2_t,
            Vsint_init=Vsint_init, Vb_init=Vb_init_t,
            model_M2=model_M2,
        )
    return {
        "Id": float(out["Id"].abs().item()),
        "Vsint": float(out["Vsint"].item()),
        "Vb": float(out["Vb"].item()),
        "converged": bool(out["converged"].all()),
        "niter": int(out["niter"]) if not isinstance(out["niter"], torch.Tensor)
                 else int(out["niter"].item()),
    }


def run_arclength(cfg, model_M1, model_M2, sd_M1, sd_M2, bjt,
                  P_M1, P_M2, VG1, VG2, Vd_seq, time_budget_s=300):
    """Run arclength sweep with a wall-clock budget."""
    Vd_t = torch.tensor(Vd_seq, dtype=torch.float64)
    VG1_t = torch.tensor(VG1, dtype=torch.float64)
    VG2_t = torch.tensor(VG2, dtype=torch.float64)
    t0 = time.time()
    with torch.no_grad(), \
         patch_sd_scaled(sd_M1, P_M1), \
         patch_sd_scaled(sd_M2, P_M2):
        out = forward_2t_arclength_grad(
            cfg, model_M1=model_M1, model_M2=model_M2, bjt=bjt,
            Vd_seq=Vd_t, VG1=VG1_t, VG2=VG2_t,
        )
    elapsed = time.time() - t0
    return {
        "Vd": Vd_t.numpy().tolist(),
        "Id": out["Id"].abs().numpy().tolist(),
        "Vb": out["Vb"].numpy().tolist(),
        "Vsint": out["Vsint"].numpy().tolist(),
        "converged": out["converged"].numpy().tolist(),
        "n_folds": int(out.get("arclen_n_folds", 0)),
        "n_steps": int(out.get("arclen_n_steps", 0)),
        "elapsed_s": elapsed,
    }


def main():
    print(f"[A1g] start {time.strftime('%H:%M:%S')}", flush=True)

    text_M1 = (DATA / "M1_130DNWFB.txt").read_text()
    model_M1 = BSIM4Model.from_spice(text_M1, model_type="nmos")
    patch_model_values(model_M1, type_n=True)
    text_M2 = (DATA / "M2_130bulkNSRAM.txt").read_text()
    model_M2 = BSIM4Model.from_spice(text_M2, model_type="nmos")
    patch_model_values(model_M2, type_n=True)

    cfg = NSRAMCell2TConfig(use_iii=True, use_gidl=True, use_bjt=True,
                             newton_max_iters=50)
    sd_M1 = compute_size_dep(model_M1, Geometry(L=cfg.Ln, W=cfg.Wn), T_C=cfg.T_C)
    sd_M2 = compute_size_dep(model_M2,
                              Geometry(L=cfg.Ln * cfg.M2_length_factor,
                                       W=cfg.Wn), T_C=cfg.T_C)
    cfg._sd_M1 = sd_M1
    cfg._sd_M2 = sd_M2

    sebas_rows = load_sebas_params()

    biases = [
        {"VG1": 0.6, "VG2": 0.0, "Vd": 1.5},
        {"VG1": 0.4, "VG2": 0.0, "Vd": 1.5},
    ]
    Vd_init_list = [0.0, 0.5, 0.7, 0.9]

    all_results = []
    for b in biases:
        VG1, VG2, Vd = b["VG1"], b["VG2"], b["Vd"]
        print(f"\n[A1g] bias VG1={VG1} VG2={VG2} Vd={Vd}", flush=True)

        sebas_row = find_params(sebas_rows, VG1, VG2)
        if sebas_row is None:
            print("  no Sebas row, skip"); continue
        P_M1, P_M2 = m1_m2_overrides_for_z91g(sebas_row)
        bjt = make_bjt(sebas_row)
        print(f"  Sebas row: K1={sebas_row.get('K1'):.4f} ETAB={sebas_row.get('ETAB'):.4f}"
              f" ALPHA0={sebas_row.get('ALPHA0'):.3e} BETA0={sebas_row.get('BETA0'):.3f}"
              f" NFACTOR={sebas_row.get('NFACTOR'):.3f} mbjt={sebas_row.get('mbjt')}"
              f" IS={sebas_row.get('IS'):.3e} area={sebas_row.get('area'):.3e}",
              flush=True)

        Vd_meas, Id_meas = load_meas_at_vd(VG1, VG2, Vd_target=Vd)
        print(f"  measurement @ Vd={Vd_meas}: Id_meas={Id_meas:.3e}", flush=True)

        # (a-d) sweep Vb_init
        per_init = {}
        for vb0 in Vd_init_list:
            r = run_one_bias(cfg, model_M1, model_M2, sd_M1, sd_M2, bjt,
                             P_M1, P_M2, VG1, VG2, Vd, vb0)
            per_init[f"vb_init={vb0}"] = r
            print(f"  Vb_init={vb0}: Id={r['Id']:.3e} Vb={r['Vb']:.4f}"
                  f" Vsint={r['Vsint']:.4f} conv={r['converged']} niter={r['niter']}",
                  flush=True)

        # (e) arclength sweep Vd 0.05 -> 2.0
        Vd_seq = np.linspace(0.05, 2.0, 40)
        print(f"  arclength sweep Vd[0.05..2.0] N={len(Vd_seq)} ...", flush=True)
        arc = run_arclength(cfg, model_M1, model_M2, sd_M1, sd_M2, bjt,
                            P_M1, P_M2, VG1, VG2, Vd_seq.tolist())
        print(f"  arclength: n_folds={arc['n_folds']} n_steps={arc['n_steps']}"
              f" elapsed={arc['elapsed_s']:.1f}s", flush=True)
        # Pull Id at Vd~=1.5
        idx15 = int(np.argmin(np.abs(np.array(arc["Vd"]) - 1.5)))
        Id_arc = arc["Id"][idx15]
        Vb_arc = arc["Vb"][idx15]
        print(f"  arclength @ Vd={arc['Vd'][idx15]:.3f}: Id={Id_arc:.3e}"
              f" Vb={Vb_arc:.4f} conv={arc['converged'][idx15]}", flush=True)

        all_results.append({
            "VG1": VG1, "VG2": VG2, "Vd": Vd,
            "Id_meas": Id_meas, "Vd_meas": Vd_meas,
            "per_init": per_init,
            "arclength_at_1p5": {"Id": Id_arc, "Vb": Vb_arc,
                                  "converged": arc["converged"][idx15]},
            "arclength_full": arc,
        })

    out_json = OUT / "A1g_multiroot_trace.json"
    out_json.write_text(json.dumps(all_results, indent=2, default=float))
    print(f"\n[A1g] saved {out_json}", flush=True)

    # Plot Id vs Vd from arclength for both biases
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    for ax, res in zip(axes, all_results):
        arc = res["arclength_full"]
        Vd = np.array(arc["Vd"]); Id = np.array(arc["Id"])
        conv = np.array(arc["converged"])
        Id_plot = np.where(conv, Id, np.nan)
        ax.semilogy(Vd, Id_plot, "-o", ms=3, lw=1.0, label="arclength")
        if res["Id_meas"] is not None:
            ax.semilogy([res["Vd_meas"]], [res["Id_meas"]], "r*", ms=14,
                        label=f"meas Vd={res['Vd_meas']:.2f}")
        # Plot per-init Id at Vd=1.5
        markers = {"vb_init=0.0": "v", "vb_init=0.5": "s",
                   "vb_init=0.7": "D", "vb_init=0.9": "^"}
        for k, r in res["per_init"].items():
            ax.semilogy([res["Vd"]], [r["Id"]], markers[k], ms=8, alpha=0.7,
                        label=f"{k} (Vb={r['Vb']:.2f})")
        ax.set_title(f"VG1={res['VG1']} VG2={res['VG2']}")
        ax.set_xlabel("Vd [V]")
        ax.set_ylim(1e-13, 1e-3)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7, loc="lower right")
    axes[0].set_ylabel("|Id| [A]")
    fig.suptitle("A1g multi-root diagnostic: arclength vs Vb_init starts vs measurement",
                 weight="bold")
    fig.tight_layout()
    fig.savefig(OUT / "A1g_multiroot.png", dpi=140)
    plt.close(fig)
    print(f"[A1g] saved {OUT}/A1g_multiroot.png", flush=True)


if __name__ == "__main__":
    main()
