"""Track 6 / AG2: NBTI / body-charge decay.

Floating-body Vb decays exponentially:
  Vb(t) = Vb0 * exp(-t / tau_NBTI)
Sweep tau_NBTI in {1h, 1d, 1w}; verify ~50% retention loss at 1 week.

Output: results/AG2_NBTI/{retention_curve.png, summary.json}
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path(__file__).resolve().parents[2] / "results" / "AG2_NBTI"
OUT.mkdir(parents=True, exist_ok=True)

Vb0 = 0.6  # V

TAUS = {"1h": 3600.0, "1d": 86400.0, "1w": 604800.0}


def main() -> None:
    t = np.logspace(0, np.log10(2 * 604800.0), 200)  # 1 s to 2 weeks
    fig, ax = plt.subplots(figsize=(7, 5))
    results = {}
    for k, tau in TAUS.items():
        v = Vb0 * np.exp(-t / tau)
        retention_1w = float(np.exp(-604800.0 / tau))
        results[k] = {
            "tau_s": tau,
            "Vb_at_1h_V": float(Vb0 * np.exp(-3600.0 / tau)),
            "Vb_at_1d_V": float(Vb0 * np.exp(-86400.0 / tau)),
            "Vb_at_1w_V": float(Vb0 * np.exp(-604800.0 / tau)),
            "retention_at_1w": retention_1w,
        }
        ax.plot(t, v, label=f"tau={k}: ret(1w)={retention_1w*100:.1f}%")
    ax.set_xscale("log")
    ax.axhline(Vb0 * 0.5, color="r", linestyle="--", alpha=0.5, label="50% retention")
    ax.axvline(604800.0, color="gray", linestyle=":", alpha=0.5, label="1 week")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Vb(t) [V]")
    ax.set_title("AG2: NBTI body-charge exponential decay")
    ax.legend(fontsize=8)
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "retention_curve.png", dpi=110)
    plt.close(fig)

    # Gate: some tau yields 1-week retention near 50% (in [0.3, 0.7])
    near_half = [k for k, v in results.items() if 0.3 <= v["retention_at_1w"] <= 0.7]
    summary = {
        "gate_1w_retention_near_50pct": len(near_half) > 0,
        "near_half_taus": near_half,
        "results": results,
        "plot": str(OUT / "retention_curve.png"),
    }
    with open(OUT / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
