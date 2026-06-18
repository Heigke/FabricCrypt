"""z58a_vg2_regimes.py — sanity check: does VG2 actually shift cell_fast
between distinct dynamical regimes?

z57 found V_latch is INERT in cell_fast at our operating point. Before
running meta-plasticity (z58b) where VG2 is the meta-knob, verify it
actually does what NS-RAM's Nature paper claims:
   VG2 ≈ -0.20  → bistable (latch)
   VG2 ≈  0.00  → soft / synapse-like
   VG2 ≈ +0.40  → integrator / neuron-like

Protocol: 9 isolated cells, each with a different VG2. Drive them with a
square pulse on VG1 (sub-vth → above-vth → off). Watch Vb(t).

If we see distinct regimes → meta-plasticity is meaningful.
If all cells behave the same → cell_fast can't represent the mode-switch
and we need Robert's emulator before doing z58b.
"""
from __future__ import annotations
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from nsram.cell_fast import CellArray

OUT = Path("/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy/"
           "results/z58a_vg2_regimes")
OUT.mkdir(parents=True, exist_ok=True)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CAL = dict(VTH0=0.43, K_back=-0.98, A_iii=4.71, G_bjt=1.00,
           V_bjt_on=0.74, V_latch=0.58, K_leak=0.021)


def main():
    VG2_levels = torch.tensor([-0.40, -0.30, -0.20, -0.10, 0.00,
                                0.10, 0.20, 0.30, 0.40], device=DEVICE)
    N = len(VG2_levels)
    cells = CellArray(N, alpha=1.5, VG2=VG2_levels, device=DEVICE, **CAL)

    # Drive pattern: idle 30, pulse-on 80, idle 200 (to see retention/decay)
    T = 310
    VG1_seq = torch.zeros(T, device=DEVICE)
    VG1_seq[30:110] = 0.7        # write pulse
    drive_seq = torch.zeros(T, device=DEVICE)
    drive_seq[30:110] = 1.0      # drain stimulus

    Vb_hist = np.zeros((T, N))
    Id_hist = np.zeros((T, N))
    for t in range(T):
        Id = cells.step(VG1_seq[t], drive_seq[t])
        Vb_hist[t] = cells.Vb.detach().cpu().numpy()
        Id_hist[t] = Id.detach().cpu().numpy()

    # Quantify regimes
    pulse_end = 109
    relax_start, relax_end = 130, 309
    Vb_at_pulse_end = Vb_hist[pulse_end]
    Vb_relaxed = Vb_hist[relax_end]
    decay_frac = np.where(np.abs(Vb_at_pulse_end) > 1e-3,
                           1.0 - Vb_relaxed / Vb_at_pulse_end,
                           0.0)
    peak_Vb = Vb_hist[30:110].max(axis=0)
    end_Vb = Vb_hist[-1]

    summary = {
        "VG2_levels": VG2_levels.cpu().tolist(),
        "peak_Vb_during_pulse": peak_Vb.tolist(),
        "Vb_at_pulse_end": Vb_at_pulse_end.tolist(),
        "Vb_after_200steps_relax": Vb_relaxed.tolist(),
        "fractional_decay": decay_frac.tolist(),
    }
    with open(OUT / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\n=== VG2 sanity (cell_fast) ===")
    print(f"{'VG2':>7s}  {'peak Vb':>8s}  {'Vb@end':>8s}  {'Vb relaxed':>10s}  {'decay%':>7s}  regime")
    regimes = []
    for i, v in enumerate(VG2_levels.cpu().numpy()):
        d = decay_frac[i] * 100
        if Vb_relaxed[i] > 0.4 and d < 30:
            regime = "BISTABLE (latched)"
        elif Vb_relaxed[i] > 0.05 and d < 70:
            regime = "soft / synapse"
        elif d > 70:
            regime = "INTEGRATOR (decays)"
        else:
            regime = "?"
        regimes.append(regime)
        print(f"{v:+.2f}    {peak_Vb[i]:.3f}    {Vb_at_pulse_end[i]:+.3f}   "
              f"{Vb_relaxed[i]:+.3f}      {d:5.1f}    {regime}")

    # Plot
    fig, axes = plt.subplots(2, 1, figsize=(13, 9), sharex=True)
    cmap = plt.get_cmap("viridis")
    for i, v in enumerate(VG2_levels.cpu().numpy()):
        c = cmap(i / (N - 1))
        axes[0].plot(Vb_hist[:, i], color=c, lw=2,
                     label=f"VG2={v:+.2f}  ({regimes[i].split()[0]})")
    axes[0].axvspan(30, 110, color="orange", alpha=0.15, label="drive pulse")
    axes[0].axhline(CAL["V_latch"], color="red", ls=":", alpha=0.5, label="V_latch")
    axes[0].set_ylabel("body voltage Vb [V]", fontsize=12)
    axes[0].set_title("z58a — VG2 sweep on isolated cells: do we see regimes?",
                      fontsize=13, weight="bold")
    axes[0].legend(loc="center right", fontsize=9, ncol=1); axes[0].grid(alpha=0.3)

    for i in range(N):
        c = cmap(i / (N - 1))
        axes[1].semilogy(np.maximum(Id_hist[:, i], 1e-12), color=c, lw=1.5)
    axes[1].axvspan(30, 110, color="orange", alpha=0.15)
    axes[1].set_xlabel("time step")
    axes[1].set_ylabel("readout current Id [A] (log)", fontsize=12)
    axes[1].grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(OUT / "vg2_sweep.png", dpi=140)
    plt.close(fig)

    # Verdict
    n_bistable = sum("BISTABLE" in r for r in regimes)
    n_integrator = sum("INTEGRATOR" in r for r in regimes)
    n_soft = sum("soft" in r for r in regimes)
    print(f"\nRegime count: bistable={n_bistable} soft={n_soft} integrator={n_integrator}")
    if n_bistable >= 1 and n_integrator >= 1 and n_soft >= 1:
        verdict = "PASS — VG2 spans 3 regimes in cell_fast. Meta-plasticity is meaningful."
    elif n_bistable + n_integrator + n_soft >= 2 and n_bistable + n_integrator >= 1:
        verdict = "PARTIAL — some regime separation; meta-plasticity worth trying."
    else:
        verdict = "FAIL — VG2 is largely inert in cell_fast. Need Robert's emulator first."
    print(f"\nVERDICT: {verdict}")
    print(f"\nWrote {OUT/'vg2_sweep.png'}")
    summary["verdict"] = verdict
    summary["regimes"] = regimes
    with open(OUT / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
