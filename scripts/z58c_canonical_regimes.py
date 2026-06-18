"""z58c_canonical_regimes.py — does the canonical BSIM4 + parasitic-NPN
model exhibit the three regimes that cell_fast missed?

z58a found cell_fast spans only bistable + soft regimes via VG2 (no
integrator).  Canonical has the full physics — let's check whether it
shows the expected regime separation.

Protocol: at each VG2 ∈ {-0.4, -0.2, 0.0, +0.2, +0.4}, do a Vd up-sweep
0→3V then a down-sweep 3→0V using find_vb_continuation. Plot Vb(Vd)
loops.

  Bistable  → wide hysteresis loop (snapback up, slow release down)
  Soft      → narrow loop / partial hysteresis
  Integrator → no hysteresis, smooth monotonic Vb(Vd)
"""
from __future__ import annotations
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from nsram.nsram_canonical import NSRAMParams, find_vb_continuation

OUT = Path("/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy/"
           "results/z58c_canonical_regimes")
OUT.mkdir(parents=True, exist_ok=True)


def sweep(VG1, VG2, p, Vd_max=2.5, n_pts=50):
    """Up-sweep then down-sweep, return (Vd_up, Vb_up, Vd_dn, Vb_dn)."""
    Vd_up = np.linspace(0.05, Vd_max, n_pts)
    Vd_dn = Vd_up[::-1].copy()
    Vb_up = np.zeros(n_pts); Vb_dn = np.zeros(n_pts)

    Vb = float(VG2)
    for k, vd in enumerate(Vd_up):
        Vb = find_vb_continuation(VG1, VG2, float(vd), p, Vb0=Vb)
        Vb_up[k] = Vb
    # continue from latched/last state
    for k, vd in enumerate(Vd_dn):
        Vb = find_vb_continuation(VG1, VG2, float(vd), p, Vb0=Vb)
        Vb_dn[k] = Vb
    return Vd_up, Vb_up, Vd_dn, Vb_dn


def main():
    # Tame the runaway: lower BJT gain + lower Iii so we operate in a
    # regime where VG2 actually selects between behaviors.
    p = NSRAMParams(gamma_VG2=0.3, AGIDL=0.0,
                    ALPHA0_mult=0.3, BJT_BF=200.0, BJT_IKF=1e-3)
    VG1 = 0.6
    VG2_levels = [-0.40, -0.20, 0.00, +0.20, +0.40]

    fig, ax = plt.subplots(1, 1, figsize=(10, 7))
    cmap = plt.get_cmap("viridis")
    summary = {"VG1": VG1, "VG2_levels": VG2_levels, "loops": {}}
    print(f"\n{'VG2':>7s} {'max Vb up':>11s} {'max Vb dn':>11s} {'hysteresis':>11s}  regime")
    for i, VG2 in enumerate(VG2_levels):
        Vd_up, Vb_up, Vd_dn, Vb_dn = sweep(VG1, VG2, p)
        # hysteresis area in Vb-Vd plane
        Vb_at_dn_resampled = np.interp(Vd_up, Vd_dn[::-1], Vb_dn[::-1])
        hyst_area = float(np.trapz(Vb_at_dn_resampled - Vb_up, Vd_up))
        max_up = float(Vb_up.max()); max_dn = float(Vb_dn.max())
        if hyst_area > 0.10:
            regime = "BISTABLE (strong hysteresis)"
        elif hyst_area > 0.02:
            regime = "soft / partial hysteresis"
        else:
            regime = "INTEGRATOR (no hysteresis)"

        summary["loops"][f"{VG2:+.2f}"] = {
            "Vd": Vd_up.tolist(), "Vb_up": Vb_up.tolist(),
            "Vb_dn": Vb_dn.tolist(),
            "hysteresis_area": hyst_area, "regime": regime}
        print(f"{VG2:+5.2f}  {max_up:+10.3f}  {max_dn:+10.3f}  "
              f"{hyst_area:+10.3f}   {regime}")

        c = cmap(i / (len(VG2_levels) - 1))
        ax.plot(Vd_up, Vb_up, "-", color=c, lw=2, label=f"VG2={VG2:+.2f} ({regime.split()[0]})")
        ax.plot(Vd_dn, Vb_dn, "--", color=c, lw=2, alpha=0.6)
        # arrow markers
        ax.annotate("", xy=(Vd_up[20], Vb_up[20]),
                    xytext=(Vd_up[15], Vb_up[15]),
                    arrowprops=dict(arrowstyle="->", color=c, lw=1.5))
        ax.annotate("", xy=(Vd_dn[20], Vb_dn[20]),
                    xytext=(Vd_dn[15], Vb_dn[15]),
                    arrowprops=dict(arrowstyle="->", color=c, lw=1.5))

    ax.set_xlabel("Vd [V]", fontsize=12)
    ax.set_ylabel("body voltage Vb [V]", fontsize=12)
    ax.set_title(f"z58c — Canonical NS-RAM: Vd up/down sweeps at multiple VG2\n"
                 f"VG1={VG1}, gamma_VG2={p.gamma_VG2}, BJT_BF={p.BJT_BF}",
                 fontsize=12, weight="bold")
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "regimes.png", dpi=140)
    plt.close(fig)

    with open(OUT / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    n_bist = sum("BISTABLE" in v["regime"] for v in summary["loops"].values())
    n_int  = sum("INTEGRATOR" in v["regime"] for v in summary["loops"].values())
    n_soft = sum("soft" in v["regime"] for v in summary["loops"].values())
    print(f"\nRegime tally: bistable={n_bist}  soft={n_soft}  integrator={n_int}")
    if n_bist >= 1 and n_int >= 1:
        verdict = "PASS — canonical spans bistable AND integrator regimes via VG2."
    elif n_bist >= 1 and n_soft >= 1:
        verdict = "PARTIAL — bistable + soft visible; integrator regime not reached"
    else:
        verdict = "FAIL — VG2 inert in canonical too."
    print(f"\nVERDICT: {verdict}")
    print(f"\nWrote {OUT/'regimes.png'}")
    summary["verdict"] = verdict
    with open(OUT / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
