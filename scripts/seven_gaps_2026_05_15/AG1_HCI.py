"""Track 6 / AG1: Hot-Carrier-Injection (HCI) Vth drift.

Takeda/Hu-style HCI: dVth_HCI/dt = A_HCI * Ids^n
We integrate over a representative transient where Ids varies between
sub-threshold (10 nA) and saturation (300 uA) to bracket realistic stress.

Sweep stress duration up to 1e4 s; verify total dVth in [5, 20] mV.

Output: results/AG1_HCI/{drift_curve.png, summary.json}
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path(__file__).resolve().parents[2] / "results" / "AG1_HCI"
OUT.mkdir(parents=True, exist_ok=True)

A_HCI = 1e-15      # V * s^-1 * A^-n  (calibrated for n=3)
N_EXP = 3
T_END = 1e4        # 10 ks stress
N_T = 500


def main() -> None:
    # Stress profiles: constant Ids in {10n, 1u, 10u, 100u, 300u}
    cases = {
        "sub_threshold_10nA": 10e-9,
        "weak_1uA": 1e-6,
        "mid_10uA": 10e-6,
        "strong_100uA": 100e-6,
        "saturation_300uA": 300e-6,
    }
    t = np.linspace(0, T_END, N_T)
    fig, ax = plt.subplots(figsize=(7, 5))
    results = {}
    for k, ids in cases.items():
        # dVth/dt constant -> Vth(t) = A_HCI * Ids^n * t
        dVth = A_HCI * (ids ** N_EXP) * t  # V
        ax.plot(t, dVth * 1e3, label=f"{k}: Ids={ids*1e6:.3g} uA, dVth={dVth[-1]*1e3:.3f}mV")
        results[k] = {
            "Ids_A": ids,
            "dVth_at_1e4s_mV": float(dVth[-1] * 1e3),
        }
    # Also dynamic profile: square wave between weak and strong
    duty = 0.3
    period = 1e-3
    ids_dyn = np.where((np.mod(t, period) / period) < duty, 100e-6, 1e-6)
    # cumulative integral
    dVth_dyn = np.cumsum(A_HCI * (ids_dyn ** N_EXP)) * (t[1] - t[0])
    ax.plot(t, dVth_dyn * 1e3, "k--", label=f"dyn 30% duty 100uA: {dVth_dyn[-1]*1e3:.3f}mV")
    results["dynamic_30pct_100uA"] = {"dVth_at_1e4s_mV": float(dVth_dyn[-1] * 1e3)}
    ax.set_xlabel("Stress time [s]")
    ax.set_ylabel("dVth_HCI [mV]")
    ax.set_title("AG1: HCI Vth drift, n=3, A_HCI=1e-15")
    ax.set_xscale("log")
    ax.set_yscale("symlog", linthresh=1e-4)
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "drift_curve.png", dpi=110)
    plt.close(fig)

    # Gate: at least one constant-Ids stress falls in 5-20 mV at 1e4 s
    in_band = [v["dVth_at_1e4s_mV"] for v in results.values() if 5 <= v["dVth_at_1e4s_mV"] <= 20]
    summary = {
        "gate_5_20mV_at_1e4s": len(in_band) > 0,
        "in_band_cases": len(in_band),
        "A_HCI": A_HCI,
        "n_exp": N_EXP,
        "results": results,
        "plot": str(OUT / "drift_curve.png"),
    }
    with open(OUT / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
