"""Track 5 / N3: DS-N15 Bayesian RNG retest with realistic 1/f + RTN noise.

Use 1/f + RTN substrate noise as the random source for a 2D Gaussian posterior
Metropolis-Hastings sampler, instead of pure Gaussian RNG. Compute KL(empirical
|| target) and compare to the published 0.00147 baseline (Voss-McCartney pink).

Note: original DS-N15 already used Voss pink noise; here we add the RTN
component to test mixed 1/f + RTN robustness.

Output: results/N3_bayes_realnoise/{posterior.png, summary.json}
"""
from __future__ import annotations
import json, os, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path(__file__).resolve().parents[2] / "results" / "N3_bayes_realnoise"
OUT.mkdir(parents=True, exist_ok=True)

# Re-use noise generators
sys.path.insert(0, str(Path(__file__).parent))
from N1_one_f_noise import gen_1f_noise, K_F_DEFAULT, CWL, FS as F1F  # type: ignore
from N2_RTN import gen_rtn, DV_RTN  # type: ignore

N_SAMPLES = 10_000


def realnoise_uniforms(n: int, seed: int, K_f_mult: float, tau_rtn_s: float) -> np.ndarray:
    """Combine 1/f and RTN, transform to ~ U(0,1) via Gaussian CDF (after standardization)."""
    one_f = gen_1f_noise(n, F1F, K_F_DEFAULT * K_f_mult, CWL, seed=seed)
    rtn = gen_rtn(n, F1F, tau_rtn_s, DV_RTN, seed=seed + 1)
    raw = one_f + rtn.astype(np.float64)
    # Standardize empirically
    raw = (raw - raw.mean()) / (raw.std() + 1e-30)
    # Approx Gaussian CDF -> uniform
    from scipy.stats import norm
    return norm.cdf(raw).astype(np.float64)


def mh_gaussian_2d(uniforms: np.ndarray, n_samples: int, proposal_sigma: float = 0.6):
    """Metropolis-Hastings on a 2D Gaussian posterior N(mu=[0,0], Sigma=I).

    uniforms: pre-generated uniform stream; pairs (u_prop, u_acc) per step.
    Returns samples of shape (n_samples, 2).
    """
    # Inverse-CDF Gaussian for proposal
    from scipy.stats import norm
    needed = 4 * n_samples
    u = uniforms[:needed]
    # split into proposal-x, proposal-y, accept, _
    u = u.reshape(n_samples, 4)
    z_prop_x = norm.ppf(np.clip(u[:, 0], 1e-9, 1 - 1e-9)) * proposal_sigma
    z_prop_y = norm.ppf(np.clip(u[:, 1], 1e-9, 1 - 1e-9)) * proposal_sigma
    u_accept = u[:, 2]

    samples = np.zeros((n_samples, 2))
    cur = np.array([0.0, 0.0])
    cur_logp = -0.5 * np.dot(cur, cur)
    for i in range(n_samples):
        prop = cur + np.array([z_prop_x[i], z_prop_y[i]])
        prop_logp = -0.5 * np.dot(prop, prop)
        if np.log(u_accept[i] + 1e-30) < prop_logp - cur_logp:
            cur = prop
            cur_logp = prop_logp
        samples[i] = cur
    return samples


def kl_2d_gaussian_to_standard(samples: np.ndarray, bins: int = 40) -> float:
    """KL(empirical || N(0,I)) via histogram + numerical integration."""
    H, xe, ye = np.histogram2d(samples[:, 0], samples[:, 1], bins=bins,
                               range=[[-4, 4], [-4, 4]], density=True)
    dx = xe[1] - xe[0]
    dy = ye[1] - ye[0]
    xc = 0.5 * (xe[:-1] + xe[1:])
    yc = 0.5 * (ye[:-1] + ye[1:])
    Xg, Yg = np.meshgrid(xc, yc, indexing="ij")
    Q = (1.0 / (2 * np.pi)) * np.exp(-0.5 * (Xg ** 2 + Yg ** 2))
    P = H + 1e-12
    Q = Q + 1e-12
    mask = P > 1e-10
    return float(np.sum(P[mask] * np.log(P[mask] / Q[mask]) * dx * dy))


def main() -> None:
    conds = {
        "pure_gauss": ("baseline (PCG64 Gaussian)", None),
        "one_f_only": ("1/f only (Voss-style)", {"K_f_mult": 1.0, "tau_rtn_s": None}),
        "rtn_only":   ("RTN only", {"K_f_mult": 0.0, "tau_rtn_s": 1e-3}),
        "mixed":      ("1/f + RTN", {"K_f_mult": 1.0, "tau_rtn_s": 1e-3}),
    }
    results = {}
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.2))
    for i, (key, (label, cfg)) in enumerate(conds.items()):
        ax = axes[i]
        if cfg is None:
            rng = np.random.default_rng(42)
            u = rng.random(4 * N_SAMPLES)
        else:
            n_raw = 4 * N_SAMPLES + 10
            if cfg["tau_rtn_s"] is None:
                u = realnoise_uniforms(n_raw, seed=7, K_f_mult=cfg["K_f_mult"], tau_rtn_s=1e3)  # tau huge -> ~constant, negligible
            elif cfg["K_f_mult"] == 0:
                # pure RTN -> uniform via standardize+cdf
                rtn = gen_rtn(n_raw, F1F, cfg["tau_rtn_s"], DV_RTN, seed=11)
                raw = (rtn - rtn.mean()) / (rtn.std() + 1e-30)
                # add tiny gauss to avoid 2-point discreteness
                raw = raw + 0.05 * np.random.default_rng(12).standard_normal(n_raw)
                from scipy.stats import norm
                u = norm.cdf(raw)
            else:
                u = realnoise_uniforms(n_raw, seed=13, K_f_mult=cfg["K_f_mult"], tau_rtn_s=cfg["tau_rtn_s"])
        u = np.asarray(u, dtype=np.float64)[: 4 * N_SAMPLES]
        samples = mh_gaussian_2d(u, N_SAMPLES)
        kl = kl_2d_gaussian_to_standard(samples)
        results[key] = {"label": label, "kl": kl, "mean": samples.mean(0).tolist(), "std": samples.std(0).tolist()}
        ax.hexbin(samples[:, 0], samples[:, 1], gridsize=40, extent=[-4, 4, -4, 4], cmap="viridis")
        ax.set_title(f"{label}\nKL={kl:.5f}")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
    fig.suptitle("N3: 2D Gaussian posterior via NS-RAM noise as RNG")
    fig.tight_layout()
    fig.savefig(OUT / "posterior.png", dpi=110)
    plt.close(fig)

    baseline = 0.00147
    # Gate: at least one realistic-noise condition has KL <= 0.005
    realistic_kls = [results["one_f_only"]["kl"], results["rtn_only"]["kl"], results["mixed"]["kl"]]
    gate = any(k <= 0.005 for k in realistic_kls)
    best = min(realistic_kls)
    summary = {
        "gate_kl_le_0.005": gate,
        "baseline_DS_N15_published": baseline,
        "best_realistic_kl": best,
        "improves_baseline": best <= baseline,
        "results": results,
        "plot": str(OUT / "posterior.png"),
    }
    with open(OUT / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
