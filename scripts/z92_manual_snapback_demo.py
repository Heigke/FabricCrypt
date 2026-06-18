#!/usr/bin/env python3
"""z92: manually pick BSIM4 params, show the model CAN produce the snapback knee.

This is a sanity demo — not a fit. Goal: prove the differentiable port can
reproduce the qualitative shape of Sebas's measured curves (off-state floor,
snapback knee, on-state) when we hand-set physically reasonable params.

Sweep 3 VG1 × 3 VG2 = 9 curves, plot Id(Vd) overlay vs measured data.
Uses gmin homotopy + dense Vd in snapback for stable Newton.
"""
from __future__ import annotations
import sys, time
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

from nsram.bsim4_port.nsram_cell_2T import NSRAMCell2TConfig, forward_2t
from nsram.bsim4_port.bjt import GummelPoonNPN
from nsram.bsim4_port.model_card import BSIM4Model
from nsram.bsim4_port.temp import compute_size_dep
from nsram.bsim4_port.geometry import Geometry

# Reuse data loader from z88
from z88_bsim4_port_fit_p7v10_skipnonconv import load_curves

DATA_DIR = ROOT / "data/sebas_2026_04_22"
OUT = ROOT / "results/z92_manual_snapback_demo"
OUT.mkdir(parents=True, exist_ok=True)

# ----- 4 manually-picked parameter SETS to show how knee moves -----
# All start from cleaned Sebas card defaults (alpha0=7.84e-5, beta0=18,
# vth0=0.54). We override to push Iii/Bf into clearly-active regimes.

VARIANTS = {
    "card_baseline": {
        # Cleaned card defaults — what the card actually says.
        "_label": "Sebas card baseline (no override)",
    },
    "weak_iii": {
        "alpha0": 7.84e-4,    # 10× card baseline
        "beta0": 18.0,
        "Bf": 10000.0,
        "_label": "weak Iii (α0×10)",
    },
    "medium_iii": {
        "alpha0": 5e-3,
        "beta0": 18.0,
        "Bf": 10000.0,
        "_label": "medium Iii (α0=5e-3)",
    },
    "strong_iii": {
        "alpha0": 2e-2,
        "beta0": 16.0,        # lower beta0 → earlier snapback onset
        "Bf": 10000.0,
        "_label": "strong Iii (α0=2e-2, β0=16)",
    },
}

# Bias points to plot — pick 3 VG1 groups × 3 VG2 representatives
BIAS_PICKS = [
    (0.2, -0.1),  (0.2, 0.0),  (0.2, 0.10),
    (0.4,  0.0),  (0.4, 0.20), (0.4, 0.30),
    (0.6,  0.0),  (0.6, 0.20), (0.6, 0.40),
]


def build_cfg():
    cfg = NSRAMCell2TConfig(use_iii=True, use_gidl=True, use_bjt=True,
                             newton_max_iters=50)
    text = (DATA_DIR / "PTM130bulkNSRAM.txt").read_text()
    model = BSIM4Model.from_spice(text, model_type="nmos")
    bjt_default = GummelPoonNPN.from_sebas_card()
    sd_M1 = compute_size_dep(model, Geometry(L=cfg.Ln, W=cfg.Wn), T_C=cfg.T_C)
    sd_M2 = compute_size_dep(model, Geometry(L=cfg.Ln * cfg.M2_length_factor, W=cfg.Wn),
                              T_C=cfg.T_C)
    cfg._sd_M1 = sd_M1
    cfg._sd_M2 = sd_M2
    return cfg, model, bjt_default


CFG, MODEL, BJT_BASE = build_cfg()


def patch_overrides(overrides: dict):
    """Override sd.scaled[k] for given params; restore after."""
    saved = {"M1": {}, "M2": {}}
    for sd, key in [(CFG._sd_M1, "M1"), (CFG._sd_M2, "M2")]:
        for k, v in overrides.items():
            if k.startswith("_"):
                continue
            if k == "Bf":
                continue
            saved[key][k] = sd.scaled.get(k, None)
            sd.scaled[k] = torch.tensor(v, dtype=torch.float64)
    bjt = GummelPoonNPN.from_sebas_card()
    if "Bf" in overrides:
        bjt.Bf = overrides["Bf"]
    return saved, bjt


def restore_overrides(saved):
    for sd, key in [(CFG._sd_M1, "M1"), (CFG._sd_M2, "M2")]:
        for k, v in saved[key].items():
            if v is None:
                sd.scaled.pop(k, None)
            else:
                sd.scaled[k] = v


def run_curve(VG1, VG2, Vd_seq, overrides):
    saved, bjt = patch_overrides(overrides)
    try:
        with torch.no_grad():
            out = forward_2t(CFG, MODEL, bjt,
                             Vd_seq, torch.tensor(VG1), torch.tensor(VG2),
                             warm_start=True, use_homotopy=True,
                             dense_vd_in_snapback=True)
        Id = out["Id"].abs().detach().numpy()
        conv = np.array([bool(c) for c in out["converged"]])
        return Id, conv
    finally:
        restore_overrides(saved)


def main():
    curves = load_curves()
    # Index data by (VG1, VG2)
    data_by_bias = {(c["VG1"], c["VG2"]): c for c in curves}

    fig, axes = plt.subplots(3, 3, figsize=(15, 12), sharex=True, sharey=True)
    palette = {
        "card_baseline": ("#888888", "Sebas card (α₀=7.84e-5)"),
        "weak_iii":      ("#1f77b4", "weak (α₀×10)"),
        "medium_iii":    ("#2ca02c", "medium (α₀=5e-3)"),
        "strong_iii":    ("#d62728", "strong (α₀=2e-2, β₀=16)"),
    }

    t0 = time.time()
    for ax, (VG1, VG2) in zip(axes.flat, BIAS_PICKS):
        # Find nearest measured curve
        key = (VG1, VG2)
        meas = data_by_bias.get(key)
        if meas is None:
            best_key = min(data_by_bias.keys(),
                            key=lambda k: abs(k[0] - VG1) + abs(k[1] - VG2))
            meas = data_by_bias[best_key]
        Vd = meas["Vd"]
        ax.semilogy(Vd.numpy(), meas["Id"].abs().numpy(), "ko", ms=4,
                     alpha=0.7, label=f"data VG1={meas['VG1']} VG2={meas['VG2']:+.2f}")

        for vname, overrides in VARIANTS.items():
            color, label = palette[vname]
            try:
                Id, conv = run_curve(VG1, VG2, Vd, overrides)
                # mask non-converged points (keep gaps)
                Id_plot = np.where(conv, np.abs(Id), np.nan)
                ax.semilogy(Vd.numpy(), Id_plot, "-", color=color, lw=1.6,
                             label=f"{label}  ({int(conv.sum())}/{len(conv)})")
            except Exception as e:
                print(f"  [VG1={VG1} VG2={VG2}] {vname} failed: {e}", flush=True)
        ax.set_title(f"VG1={VG1}, VG2={VG2:+.2f}", fontsize=10)
        ax.grid(alpha=0.3)
        ax.set_ylim(1e-14, 1e-3)
        if ax in axes[2]:
            ax.set_xlabel("Vd [V]")
        if ax in axes[:, 0]:
            ax.set_ylabel("|Id| [A]")
        ax.legend(loc="upper left", fontsize=7, framealpha=0.85)
        elapsed = time.time() - t0
        print(f"VG1={VG1} VG2={VG2:+.2f} done ({elapsed:.0f}s)", flush=True)

    fig.suptitle(
        "z92: BSIM4 port — manually-picked Iii strengths sweep snapback knee\n"
        "(no fit: just override α₀, β₀ to show model CAN produce off→knee→on)",
        fontsize=13, weight="bold",
    )
    fig.tight_layout()
    out_path = OUT / "snapback_sweep.png"
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
