"""E5 — BSIM4 junction breakdown (§10.1) vs channel HCI (§6.1) on Chynoweth.

The previous experiment (E2) showed BSIM4 §6.1 channel impact ionization
cannot match Zenodo Chynoweth shape (~1.6 decade mismatch).  That
mismatch may simply be because §6.1 is the WRONG physics for NS-RAM:
Pazos' avalanche-diode subcircuit is Zener-like, which maps onto
BSIM4 §10.1 junction breakdown (f_breakdown multiplier), not §6.1.

This experiment answers: "which BSIM4 path fits silicon-calibrated
Chynoweth better — channel HCI or junction breakdown?"

If junction breakdown wins → Sebastian should probably drive his
new 2T firing model from BSIM4 §10 body-junction avalanche, not §6.1.
"""

from __future__ import annotations
import json, sys, time
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "nsram"))

from nsram.bsim4 import (                                         # noqa: E402
    BSIM4Params, impact_ionization_bsim4, junction_breakdown_current,
)
from nsram.fitting import (                                       # noqa: E402
    fit_bsim4_impact, fit_junction_breakdown,
)
from nsram.physics import avalanche_current                       # noqa: E402

OUT = REPO / "results" / "z2_nsram_bsim4_zenodo"
OUT.mkdir(parents=True, exist_ok=True)

ZENODO = dict(BV0=3.5, k_vg=1.5, Tbv1=-21.3e-6, Ne=1.5, Is=1e-16)


def main():
    print("=" * 70)
    print("E5 — BSIM4 §6.1 channel HCI vs §10.1 junction breakdown")
    print("=" * 70)

    VG1 = np.array([0.6, 0.8, 1.0, 1.2, 1.4])
    Vds_arr = np.linspace(2.0, 4.5, 40)
    SEEDS = 10
    rng = np.random.default_rng(5)

    # Generate silicon-calibrated Chynoweth ground truth
    chan_r2, junc_r2 = [], []
    chan_rms, junc_rms = [], []
    labels = []

    for vg1 in VG1:
        for seed in range(SEEDS):
            y = avalanche_current(Vds_arr, vg1, T=300.0,
                                   I0=ZENODO["Is"], BV0=ZENODO["BV0"],
                                   k_vg=ZENODO["k_vg"], Tbv1=ZENODO["Tbv1"])
            y = np.maximum(y, 1e-14)
            y = np.maximum(y * (1 + 0.05 * rng.standard_normal(len(y))), 1e-14)

            # Channel HCI fit (§6.1) — treat Vds as the sweep axis
            r_chan = fit_bsim4_impact(Vds_arr, y, Vgs=float(vg1), Vbs=0.0,
                                        base=BSIM4Params())
            # Junction breakdown fit (§10.1) — reinterpret Vds as the
            # body-junction reverse bias (drain-body avalanche).
            # The NS-RAM avalanche diode in the Zenodo model sits between
            # drain and body, so -Vds corresponds to Vbd (reverse bias).
            r_junc = fit_junction_breakdown(
                -Vds_arr, y, side="drain",
                base=BSIM4Params(dioMod=2, JSD=1e-18))  # lower Js → better tail

            chan_r2.append(r_chan.get("r_squared", 0.0) if "r_squared" in r_chan else 0.0)
            junc_r2.append(r_junc.get("r_squared", 0.0) if "r_squared" in r_junc else 0.0)

            # compute RMS in decades
            if "params" in r_chan:
                pred_c = np.maximum(np.array([
                    float(impact_ionization_bsim4(vg1, v, 0.0, r_chan["params"]))
                    for v in Vds_arr]), 1e-30)
                chan_rms.append(float(np.sqrt(
                    np.mean((np.log10(pred_c) - np.log10(y)) ** 2))))
            else:
                chan_rms.append(np.nan)

            if "params" in r_junc:
                pred_j = np.abs(junction_breakdown_current(
                    -Vds_arr, r_junc["params"], side="drain"))
                pred_j = np.maximum(pred_j, 1e-30)
                junc_rms.append(float(np.sqrt(
                    np.mean((np.log10(pred_j) - np.log10(y)) ** 2))))
            else:
                junc_rms.append(np.nan)

            labels.append((float(vg1), seed))

    chan_r2 = np.asarray(chan_r2); junc_r2 = np.asarray(junc_r2)
    chan_rms = np.asarray(chan_rms); junc_rms = np.asarray(junc_rms)

    def fmt(a): return f"median={np.nanmedian(a):.3f}, p90={np.nanpercentile(a, 90):.3f}"
    print(f"\n  Channel HCI  (§6.1):  R²  {fmt(chan_r2)}")
    print(f"  Junction BRK (§10.1): R²  {fmt(junc_r2)}")
    print(f"  Channel HCI   RMS (dec):  {fmt(chan_rms)}")
    print(f"  Junction BRK  RMS (dec):  {fmt(junc_rms)}")

    better_junc = (junc_rms < chan_rms).mean()
    print(f"\n  fraction where junction BRK beats channel HCI: {better_junc:.1%}")
    if np.nanmedian(junc_rms) < np.nanmedian(chan_rms):
        delta = np.nanmedian(chan_rms) - np.nanmedian(junc_rms)
        verdict = f"JUNCTION BREAKDOWN wins by {delta:.2f} decades (median RMS)"
    else:
        delta = np.nanmedian(junc_rms) - np.nanmedian(chan_rms)
        verdict = f"CHANNEL HCI wins by {delta:.2f} decades (median RMS)"
    print(f"  VERDICT: {verdict}")

    # Plot
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
        axes[0].boxplot([chan_rms, junc_rms], labels=["Channel HCI\n(§6.1)",
                                                       "Junction BRK\n(§10.1)"])
        axes[0].set_ylabel("RMS error (decades)")
        axes[0].set_title("Fit quality vs Chynoweth (silicon-calibrated)")
        axes[0].grid(alpha=0.3)

        axes[1].scatter(chan_rms, junc_rms, s=20, alpha=0.6)
        lim = max(np.nanmax(chan_rms), np.nanmax(junc_rms)) * 1.05
        axes[1].plot([0, lim], [0, lim], "k--", lw=1, label="y=x")
        axes[1].fill_between([0, lim], [0, lim], 0, alpha=0.1,
                              color="green", label="§10.1 better")
        axes[1].fill_between([0, lim], [0, lim], lim, alpha=0.1,
                              color="red", label="§6.1 better")
        axes[1].set_xlabel("Channel HCI RMS (dec)")
        axes[1].set_ylabel("Junction BRK RMS (dec)")
        axes[1].legend(fontsize=8); axes[1].grid(alpha=0.3)
        axes[1].set_title("Paired per-curve comparison")

        fig.suptitle("E5 — Which BSIM4 path fits NS-RAM silicon better?")
        fig.tight_layout()
        fig.savefig(OUT / "e5_junction_vs_channel.png", dpi=140,
                     bbox_inches="tight")
        plt.close(fig)
        print(f"[plot] {OUT / 'e5_junction_vs_channel.png'}")
    except ImportError:
        pass

    summary = {
        "channel_hci": {
            "r2_median": float(np.nanmedian(chan_r2)),
            "rms_dec_median": float(np.nanmedian(chan_rms)),
            "rms_dec_p90": float(np.nanpercentile(chan_rms, 90)),
        },
        "junction_breakdown": {
            "r2_median": float(np.nanmedian(junc_r2)),
            "rms_dec_median": float(np.nanmedian(junc_rms)),
            "rms_dec_p90": float(np.nanpercentile(junc_rms, 90)),
        },
        "fraction_junction_better": float(better_junc),
        "verdict": verdict,
        "n_curves": len(labels),
    }
    with open(OUT / "e5_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[done] {OUT / 'e5_summary.json'}")


if __name__ == "__main__":
    main()
