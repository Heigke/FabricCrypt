"""Track 6 / AG3: Memory retention audit.

Application: DS-N15 Bayesian RNG. The "memory" content is the cell's pink-noise
state vector; if Vb decays per AG2 model, the spectral 1/f character degrades.
We measure how the KL(empirical||target) degrades with aging time at the
medium tau choice (1 day) — the candidate breaking point.

Output: results/AG3_retention/{degradation.png, summary.json}
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
from N1_one_f_noise import gen_1f_noise, K_F_DEFAULT, CWL, FS as F1F  # type: ignore
from N3_bayes_realnoise import mh_gaussian_2d, kl_2d_gaussian_to_standard, N_SAMPLES  # type: ignore

OUT = Path(__file__).resolve().parents[2] / "results" / "AG3_retention"
OUT.mkdir(parents=True, exist_ok=True)


def main() -> None:
    times_s = [0.0, 3600.0, 86400.0, 604800.0]
    labels = ["t=0", "t=1h", "t=24h", "t=168h"]
    # Use tau=1d (medium retention) per AG2
    tau_nbti = 86400.0
    fig, ax = plt.subplots(figsize=(7, 5))
    results = {}
    for tlabel, t in zip(labels, times_s):
        # Vb amplitude scaling factor
        amp_factor = float(np.exp(-t / tau_nbti))
        # Generate 1/f noise scaled by amp_factor
        n_raw = 4 * N_SAMPLES + 10
        v = gen_1f_noise(n_raw, F1F, K_F_DEFAULT, CWL, seed=21) * amp_factor
        # Standardize -> uniform via norm CDF
        from scipy.stats import norm
        v = (v - v.mean()) / (v.std() + 1e-30)
        u = norm.cdf(v).astype(np.float64)[: 4 * N_SAMPLES]
        samples = mh_gaussian_2d(u, N_SAMPLES)
        kl = kl_2d_gaussian_to_standard(samples)
        results[tlabel] = {
            "t_s": t,
            "amp_factor": amp_factor,
            "kl": kl,
            "Vb_eff": 0.6 * amp_factor,
        }
        ax.scatter(t + 1, kl, s=80, label=f"{tlabel} amp={amp_factor:.3f} KL={kl:.4f}")
    ax.set_xscale("log")
    ax.set_xlabel("Aging time [s]")
    ax.set_ylabel("KL(empirical||target)")
    ax.set_title("AG3: DS-N15 Bayes RNG retention vs aging (tau_NBTI=1d)")
    ax.axhline(0.005, color="r", linestyle="--", alpha=0.5, label="N3 gate=0.005")
    ax.legend(fontsize=8)
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "degradation.png", dpi=110)
    plt.close(fig)

    kls = [results[L]["kl"] for L in labels]
    breaking = next((labels[i] for i, k in enumerate(kls) if k > 0.005), None)
    summary = {
        "tau_NBTI_s": tau_nbti,
        "results": results,
        "breaking_point": breaking,
        "kl_trend_monotonic_up": all(kls[i + 1] >= kls[i] - 1e-4 for i in range(len(kls) - 1)),
        "plot": str(OUT / "degradation.png"),
    }
    with open(OUT / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
