#!/usr/bin/env python3
"""z93: focused snapback test — does the model produce a clean knee at some
parameter setting on a bias where M2 is fully on?

Tests at VG1=0.6, VG2=+0.4 (M2 strongly conducting).
Dense Vd sweep 0.05 → 1.95 V at 0.025 V step → 76 points (4× z88's 0.1 V grid).
Tries multiple α₀ and β₀ combinations including some BEYOND current fit bounds.

Diagnostic question:
- Can our verified-correct model produce a 4-6 decade Id jump (snapback) at any
  parameter setting?
- If YES at some setting: model + topology can do snapback; problem is purely
  fitting strategy.
- If NO at any setting we try: there's a structural barrier (M2 series-choke,
  numerical, or physics gap) we need to find.
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
from z88_bsim4_port_fit_p7v10_skipnonconv import load_curves

DATA_DIR = ROOT / "data/sebas_2026_04_22"
OUT = ROOT / "results/z93_focused_snapback_test"
OUT.mkdir(parents=True, exist_ok=True)

# Bias where M2 is strongly on (M2 Vth ≈ 0.18 V, VG2=+0.4 → strongly above-Vth)
VG1 = 0.6
VG2 = 0.4
Vd_seq = torch.tensor(np.arange(0.05, 1.95 + 1e-9, 0.025), dtype=torch.float64)
print(f"Bias: VG1={VG1}, VG2={VG2}; {len(Vd_seq)} Vd points 0.05→1.95 V step 0.025 V")

# Variants — escalating α₀ to see if knee fires
VARIANTS = [
    ("baseline_card",     dict(),                                    "#888888"),
    ("alpha0=1e-3 b0=18", dict(alpha0=1e-3, beta0=18, Bf=10000),     "#1f77b4"),
    ("alpha0=5e-3 b0=18", dict(alpha0=5e-3, beta0=18, Bf=10000),     "#2ca02c"),
    ("alpha0=2e-2 b0=15", dict(alpha0=2e-2, beta0=15, Bf=10000),     "#ff7f0e"),
    ("alpha0=8e-2 b0=12", dict(alpha0=8e-2, beta0=12, Bf=10000),     "#d62728"),
    # ("alpha0=2e-1 b0=10", dict(alpha0=2e-1, beta0=10, Bf=10000), "#9467bd"),
    # ↑ Newton fails to converge in this extreme regime — skipped.
]


def build_cfg():
    cfg = NSRAMCell2TConfig(use_iii=True, use_gidl=True, use_bjt=True,
                             newton_max_iters=80)
    text = (DATA_DIR / "PTM130bulkNSRAM.txt").read_text()
    model = BSIM4Model.from_spice(text, model_type="nmos")
    sd_M1 = compute_size_dep(model, Geometry(L=cfg.Ln, W=cfg.Wn), T_C=cfg.T_C)
    sd_M2 = compute_size_dep(model, Geometry(L=cfg.Ln * cfg.M2_length_factor, W=cfg.Wn),
                              T_C=cfg.T_C)
    cfg._sd_M1 = sd_M1
    cfg._sd_M2 = sd_M2
    return cfg, model


CFG, MODEL = build_cfg()


def run_variant(overrides: dict):
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
    try:
        with torch.no_grad():
            out = forward_2t(CFG, MODEL, bjt,
                             Vd_seq, torch.tensor(VG1), torch.tensor(VG2),
                             warm_start=True, use_homotopy=True,
                             dense_vd_in_snapback=True,
                             snapback_vd_threshold=0.5,
                             snapback_vd_step=0.01)
        return out
    finally:
        for sd, key in [(CFG._sd_M1, "M1"), (CFG._sd_M2, "M2")]:
            for k, v in saved[key].items():
                if v is None:
                    sd.scaled.pop(k, None)
                else:
                    sd.scaled[k] = v


def find_knee(Id_arr, Vd_arr):
    """Largest decade jump over a 0.05 V Vd window. Returns (Vd_at_jump, decades, idx_jump)."""
    log_Id = np.log10(np.abs(Id_arr).clip(1e-15))
    best = (None, 0.0, -1)
    for i in range(len(Vd_arr) - 2):
        for j in range(i + 1, min(i + 6, len(Vd_arr))):  # window up to 0.125 V
            d_decades = log_Id[j] - log_Id[i]
            if d_decades > best[1]:
                best = (float(Vd_arr[j]), float(d_decades), j)
    return best


def main():
    # Load measured data for plotting reference
    curves = load_curves()
    meas = next(c for c in curves if c["VG1"] == VG1 and abs(c["VG2"] - VG2) < 1e-6)

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.semilogy(meas["Vd"].numpy(), meas["Id"].abs().numpy(),
                "ko", ms=6, label=f"Sebas data VG1={VG1} VG2={VG2:+.2f}")

    summary = []
    t0 = time.time()
    for name, overrides, color in VARIANTS:
        out = run_variant(overrides)
        Id = out["Id"].abs().detach().numpy()
        conv = np.array([bool(c) for c in out["converged"]])
        Vd_a = Vd_seq.numpy()
        Id_plot = np.where(conv, Id, np.nan)
        Vd_knee, dec_knee, idx_knee = find_knee(Id, Vd_a)

        ax.semilogy(Vd_a, Id_plot, "-", color=color, lw=1.6,
                    label=f"{name}  conv {int(conv.sum())}/{len(conv)}  "
                          f"max-jump {dec_knee:.1f} dec @ Vd={Vd_knee:.2f}V")

        summary.append({
            "name": name, "overrides": {k: float(v) for k, v in overrides.items()},
            "n_conv": int(conv.sum()), "n_total": len(conv),
            "Id_min": float(Id.min()), "Id_max": float(Id.max()),
            "max_decade_jump": float(dec_knee),
            "Vd_at_max_jump": float(Vd_knee) if Vd_knee else None,
        })
        print(f"  {name:22s}: conv {int(conv.sum())}/{len(conv)}  "
              f"Id [{Id.min():.2e}, {Id.max():.2e}]  jump={dec_knee:.2f} dec",
              flush=True)

    ax.set_xlabel("Vd [V]")
    ax.set_ylabel("|Id| [A]")
    ax.set_ylim(1e-14, 1e-3)
    ax.set_title(
        f"z93: focused snapback test — VG1={VG1}, VG2={VG2}, dense Vd grid (0.025 V step)\n"
        f"Can the model produce a knee at any parameter setting?",
        fontsize=11, weight="bold",
    )
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right", fontsize=8, framealpha=0.9)
    fig.tight_layout()
    out_png = OUT / "snapback_test.png"
    fig.savefig(out_png, dpi=140)
    plt.close(fig)
    print(f"\nSaved {out_png}")
    print(f"Total elapsed: {time.time()-t0:.0f}s")

    import json
    (OUT / "summary.json").write_text(json.dumps({
        "bias": {"VG1": VG1, "VG2": VG2},
        "Vd_seq": Vd_seq.numpy().tolist(),
        "variants": summary,
    }, indent=2))


if __name__ == "__main__":
    main()
