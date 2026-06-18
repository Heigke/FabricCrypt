#!/usr/bin/env python3
"""z94: test pseudo-arclength solver vs the existing Newton+homotopy solver
on the same bias (VG1=0.6, VG2=0.4) where z93 showed bistability artifacts.

If arclength produces a CLEAN snapback knee (no jumping between roots),
we replace forward_curve in the fitting pipeline.
"""
from __future__ import annotations
import sys, time, json
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

from nsram.bsim4_port.nsram_cell_2T import (
    NSRAMCell2TConfig, forward_2t,
)
from nsram.bsim4_port.arclength import solve_2t_arclength, trace_arclength
from nsram.bsim4_port.bjt import GummelPoonNPN
from nsram.bsim4_port.model_card import BSIM4Model
from nsram.bsim4_port.temp import compute_size_dep
from nsram.bsim4_port.geometry import Geometry

DATA_DIR = ROOT / "data/sebas_2026_04_22"
OUT = ROOT / "results/z94_arclength_test"
OUT.mkdir(parents=True, exist_ok=True)

VG1 = 0.6
VG2 = 0.4
Vd_seq = torch.tensor(np.arange(0.05, 1.95 + 1e-9, 0.025), dtype=torch.float64)

# Two parameter sets that gave snapback in z93
VARIANTS = [
    ("alpha0=2e-2 b0=15", dict(alpha0=2e-2, beta0=15, Bf=10000), "#2ca02c"),
    ("alpha0=8e-2 b0=12", dict(alpha0=8e-2, beta0=12, Bf=10000), "#d62728"),
]


def build():
    cfg = NSRAMCell2TConfig(use_iii=True, use_gidl=True, use_bjt=True,
                             newton_max_iters=50)
    text = (DATA_DIR / "PTM130bulkNSRAM.txt").read_text()
    model = BSIM4Model.from_spice(text, model_type="nmos")
    sd_M1 = compute_size_dep(model, Geometry(L=cfg.Ln, W=cfg.Wn), T_C=cfg.T_C)
    sd_M2 = compute_size_dep(model, Geometry(L=cfg.Ln * cfg.M2_length_factor,
                                              W=cfg.Wn), T_C=cfg.T_C)
    cfg._sd_M1 = sd_M1
    cfg._sd_M2 = sd_M2
    return cfg, model


CFG, MODEL = build()


def with_overrides(overrides):
    saved = {"M1": {}, "M2": {}}
    for sd, key in [(CFG._sd_M1, "M1"), (CFG._sd_M2, "M2")]:
        for k, v in overrides.items():
            if k == "Bf":
                continue
            saved[key][k] = sd.scaled.get(k, None)
            sd.scaled[k] = torch.tensor(float(v), dtype=torch.float64)
    bjt = GummelPoonNPN.from_sebas_card()
    if "Bf" in overrides:
        bjt.Bf = float(overrides["Bf"])
    return saved, bjt


def restore(saved):
    for sd, key in [(CFG._sd_M1, "M1"), (CFG._sd_M2, "M2")]:
        for k, v in saved[key].items():
            if v is None:
                sd.scaled.pop(k, None)
            else:
                sd.scaled[k] = v


def main():
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

    summary = []
    for ax_idx, (name, overrides, color) in enumerate(VARIANTS):
        ax = axes[ax_idx]
        ax.set_title(f"{name}", fontsize=11)

        saved, bjt = with_overrides(overrides)
        try:
            # 1. Existing solver (Newton + gmin homotopy)
            t0 = time.time()
            old_out = forward_2t(CFG, MODEL, bjt,
                                  Vd_seq, torch.tensor(VG1), torch.tensor(VG2),
                                  warm_start=True, use_homotopy=True,
                                  dense_vd_in_snapback=True,
                                  snapback_vd_threshold=0.5,
                                  snapback_vd_step=0.01)
            t_old = time.time() - t0
            old_Id = old_out["Id"].abs().detach().numpy()
            old_conv = np.array([bool(c) for c in old_out["converged"]])
            old_Id_plot = np.where(old_conv, old_Id, np.nan)
            ax.semilogy(Vd_seq.numpy(), old_Id_plot, "x-",
                         color="#888888", lw=1.0, ms=4, alpha=0.6,
                         label=f"Newton+homotopy ({int(old_conv.sum())}/{len(old_conv)}, {t_old:.1f}s)")

            # 2. New arclength solver
            t0 = time.time()
            new_out = solve_2t_arclength(CFG, MODEL, bjt, Vd_seq, VG1, VG2)
            t_new = time.time() - t0
            new_Id = new_out["Id"].abs().detach().numpy()
            new_conv = new_out["converged"].numpy()
            new_Id_plot = np.where(new_conv, new_Id, np.nan)
            ax.semilogy(Vd_seq.numpy(), new_Id_plot, "o-",
                         color=color, lw=1.6, ms=3,
                         label=f"arclength ({int(new_conv.sum())}/{len(new_conv)}, "
                               f"{new_out['n_folds']} folds, {t_new:.1f}s)")

            # 3. Trace the actual arclength path (raw, not interpolated)
            path = trace_arclength(CFG, MODEL, bjt, VG1, VG2,
                                    Vd_start=0.05, Vd_max=1.95)
            path_Vd = np.array(path["path_Vd"])
            path_Id = np.abs(np.array(path["path_Id"]))
            ax.semilogy(path_Vd, np.where(path_Id > 0, path_Id, np.nan),
                         ":", color=color, lw=0.8, alpha=0.5,
                         label=f"raw arclength path ({path['n_steps']} pts)")

            print(f"\n=== {name} ===")
            print(f"  Newton+homotopy: {int(old_conv.sum())}/{len(old_conv)} conv, t={t_old:.1f}s")
            print(f"  arclength      : {int(new_conv.sum())}/{len(new_conv)} conv, "
                  f"folds={new_out['n_folds']}, t={t_new:.1f}s")
            print(f"  raw path       : {path['n_steps']} steps")

            summary.append({
                "name": name,
                "newton_conv": int(old_conv.sum()),
                "newton_total": int(len(old_conv)),
                "newton_time": float(t_old),
                "arclen_conv": int(new_conv.sum()),
                "arclen_total": int(len(new_conv)),
                "arclen_time": float(t_new),
                "arclen_folds": int(new_out['n_folds']),
                "arclen_path_steps": int(path['n_steps']),
            })

        finally:
            restore(saved)

        ax.set_xlabel("Vd [V]")
        if ax_idx == 0:
            ax.set_ylabel("|Id| [A]")
        ax.set_ylim(1e-15, 1e-3)
        ax.grid(alpha=0.3)
        ax.legend(loc="upper left", fontsize=8, framealpha=0.9)

    fig.suptitle(
        "z94: pseudo-arclength continuation vs Newton+homotopy at VG1=0.6, VG2=0.4\n"
        "Test: does arclength produce a clean knee where Newton+homotopy hops between roots?",
        fontsize=11, weight="bold",
    )
    fig.tight_layout()
    out_png = OUT / "arclength_vs_newton.png"
    fig.savefig(out_png, dpi=140)
    plt.close(fig)
    print(f"\nSaved {out_png}")

    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
