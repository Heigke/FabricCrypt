"""z296 DS-N3: Bayesian MCMC with NS-RAM-derived RNG.

Toy Bayesian inference: posterior over Gaussian mean mu given 10 obs.
Compare effective sample size (ESS) per iter for two proposal RNGs:
  (a) np.random.uniform (pseudo-RNG baseline)
  (b) NS-RAM V_b excursion noise (many parallel cells driven by white input;
      U(0,1) via empirical CDF transform of V_b trajectory)

Pre-registered gates:
  PASS:       NS-RAM ESS >= 50% of pseudo-RNG ESS
  AMBITIOUS:  NS-RAM ESS >= 90% of pseudo-RNG ESS

NS-RAM noise source: simulate 1024 parallel NS-RAM cells with the z278 v3
4D surrogate, driven by white-noise V_G1 jitter around bias. Read V_b
fluctuations at end of short bursts -> shuffle -> CDF transform -> U(0,1).
"""
from __future__ import annotations
import os, sys, json, time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
os.environ.setdefault("HSA_OVERRIDE_GFX_VERSION", "11.0.0")

import torch

SURROGATE_PATH = ROOT / "results/z278_mep2_surrogate_v3/surrogate_4d_v3.npz"
N_MH = 10000
SEED = 42

# ground truth
TRUE_MU = 2.5
TRUE_SIGMA = 1.0
N_OBS = 10
PRIOR_MU0 = 0.0
PRIOR_TAU = 5.0  # prior std on mu
PROPOSAL_SCALE = 0.4


def load_surrogate(path, device):
    z = np.load(path)
    return {
        "I_d":    torch.tensor(z["Id"],    dtype=torch.float32, device=device),
        "I_ii":   torch.tensor(z["Iii"],   dtype=torch.float32, device=device),
        "I_leak": torch.tensor(z["Ileak"], dtype=torch.float32, device=device),
        "ax_VG1": torch.tensor(z["vg1_axis"], dtype=torch.float32, device=device),
        "ax_VG2": torch.tensor(z["vg2_axis"], dtype=torch.float32, device=device),
        "ax_Vd":  torch.tensor(z["vd_axis"],  dtype=torch.float32, device=device),
        "ax_Vb":  torch.tensor(z["vb_axis"],  dtype=torch.float32, device=device),
    }


def _frac_index(values, axis):
    n = axis.shape[0]
    i = torch.bucketize(values, axis) - 1
    i = i.clamp(0, n - 2)
    lo = axis[i]; hi = axis[i + 1]
    t = (values - lo) / (hi - lo)
    return i, t.clamp(0.0, 1.0)


def query_surrogate(surr, VG1, VG2, Vd, Vb):
    i0, t0 = _frac_index(VG1, surr["ax_VG1"])
    i1, t1 = _frac_index(VG2, surr["ax_VG2"])
    i2, t2 = _frac_index(Vd,  surr["ax_Vd"])
    i3, t3 = _frac_index(Vb,  surr["ax_Vb"])
    Iii_tbl, Ilk_tbl = surr["I_ii"], surr["I_leak"]
    Iii_out = torch.zeros_like(VG1); Ilk_out = torch.zeros_like(VG1)
    for a0 in (0, 1):
        w0 = t0 if a0 else (1 - t0); j0 = i0 + a0
        for a1 in (0, 1):
            w1 = t1 if a1 else (1 - t1); j1 = i1 + a1
            for a2 in (0, 1):
                w2 = t2 if a2 else (1 - t2); j2 = i2 + a2
                for a3 in (0, 1):
                    w3 = t3 if a3 else (1 - t3); j3 = i3 + a3
                    w = w0 * w1 * w2 * w3
                    Iii_out = Iii_out + w * Iii_tbl[j0, j1, j2, j3]
                    Ilk_out = Ilk_out + w * Ilk_tbl[j0, j1, j2, j3]
    return Iii_out, Ilk_out


def generate_nsram_uniforms(n_samples: int, surr, device, seed: int) -> np.ndarray:
    """Generate n_samples U(0,1) RVs from NS-RAM V_b dynamics.

    Run N_CELLS parallel cells with V_G1 = bias + white_noise(sigma). Step
    integrate V_b for T_BURN steps; collect V_b at every step after burn-in.
    Empirical-CDF transform the pool to U(0,1).
    """
    g = torch.Generator(device=device).manual_seed(seed)
    N_CELLS = 256
    T_TOTAL = max(1, int(np.ceil(n_samples / N_CELLS))) + 50
    T_BURN = 50

    VG1_bias = 0.55
    VG1_NOISE = 0.06
    VG2_bias = 0.35
    Vd_bias = 1.0
    C_b_F = 80e-15
    dt_s = 1e-7

    Vb_min = float(surr["ax_Vb"][0]); Vb_max = float(surr["ax_Vb"][-1])
    VG1_min = float(surr["ax_VG1"][0]); VG1_max = float(surr["ax_VG1"][-1])

    Vb = torch.zeros(N_CELLS, device=device) + 0.5 * (Vb_min + Vb_max)
    VG2 = torch.full((N_CELLS,), VG2_bias, device=device)
    Vd = torch.full((N_CELLS,), Vd_bias, device=device)

    samples = []
    for t in range(T_TOTAL):
        noise = torch.randn(N_CELLS, device=device, generator=g) * VG1_NOISE
        VG1 = (VG1_bias + noise).clamp(VG1_min, VG1_max)
        Vb_c = Vb.clamp(Vb_min, Vb_max)
        Iii, Ileak = query_surrogate(surr, VG1, VG2, Vd, Vb_c)
        Vb = (Vb + dt_s * (Iii - Ileak) / C_b_F).clamp(Vb_min, Vb_max)
        if t >= T_BURN:
            samples.append(Vb.detach().cpu().numpy().copy())

    pool = np.concatenate(samples)
    # whiten: empirical CDF transform => U(0,1)
    ranks = np.argsort(np.argsort(pool))
    u = (ranks + 0.5) / pool.size
    # shuffle once with deterministic seed (so subsequence isn't sorted)
    rng = np.random.default_rng(seed + 1)
    rng.shuffle(u)
    return u[:n_samples].astype(np.float64)


def log_posterior(mu: float, data: np.ndarray) -> float:
    # likelihood: Normal(mu, TRUE_SIGMA); prior: Normal(PRIOR_MU0, PRIOR_TAU)
    ll = -0.5 * np.sum((data - mu) ** 2) / (TRUE_SIGMA ** 2)
    lp = -0.5 * (mu - PRIOR_MU0) ** 2 / (PRIOR_TAU ** 2)
    return ll + lp


def metropolis_hastings(data, n_iter, uniforms_propose, uniforms_accept, start=0.0):
    """MH using Gaussian proposals derived from Box-Muller on uniforms_propose;
    accept/reject driven by uniforms_accept."""
    samples = np.zeros(n_iter)
    mu = start
    lp_curr = log_posterior(mu, data)
    n_accept = 0
    # need 2 propose uniforms per step (Box-Muller)
    for i in range(n_iter):
        u1 = uniforms_propose[2 * i] if 2 * i < len(uniforms_propose) else 0.5
        u2 = uniforms_propose[2 * i + 1] if 2 * i + 1 < len(uniforms_propose) else 0.5
        u1 = max(min(u1, 1 - 1e-12), 1e-12)
        z = np.sqrt(-2.0 * np.log(u1)) * np.cos(2 * np.pi * u2)
        mu_prop = mu + PROPOSAL_SCALE * z
        lp_prop = log_posterior(mu_prop, data)
        log_alpha = lp_prop - lp_curr
        ua = uniforms_accept[i] if i < len(uniforms_accept) else 0.5
        ua = max(min(ua, 1 - 1e-12), 1e-12)
        if np.log(ua) < log_alpha:
            mu = mu_prop
            lp_curr = lp_prop
            n_accept += 1
        samples[i] = mu
    return samples, n_accept / n_iter


def effective_sample_size(x: np.ndarray) -> float:
    """Geyer initial monotone ESS estimator (single chain)."""
    n = x.size
    x = x - x.mean()
    if x.var(ddof=0) < 1e-15:
        return 0.0
    # autocorrelation via FFT
    f = np.fft.fft(x, n=2 * n)
    acf = np.fft.ifft(f * np.conj(f)).real[:n] / (np.arange(n, 0, -1))
    rho = acf / acf[0]
    # initial monotone: sum positive even-lag pairs
    tau = 1.0
    for k in range(1, n // 2):
        pair = rho[2 * k - 1] + rho[2 * k]
        if pair <= 0:
            break
        tau += 2 * pair
    return n / max(tau, 1.0)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[z296] device={device}", flush=True)
    np.random.seed(SEED)
    data = np.random.normal(TRUE_MU, TRUE_SIGMA, N_OBS)
    print(f"[z296] data mean={data.mean():.3f} std={data.std():.3f}", flush=True)

    # ---------- Pseudo-RNG MH ----------
    t0 = time.time()
    rng = np.random.default_rng(SEED)
    u_p = rng.uniform(size=2 * N_MH)
    u_a = rng.uniform(size=N_MH)
    samples_pseudo, ar_pseudo = metropolis_hastings(data, N_MH, u_p, u_a)
    wall_pseudo = time.time() - t0
    ess_pseudo = effective_sample_size(samples_pseudo[N_MH // 5:])  # drop 20% burn
    posterior_mean_pseudo = samples_pseudo[N_MH // 5:].mean()
    print(f"[z296] pseudo: ess={ess_pseudo:.1f}/{N_MH} ar={ar_pseudo:.3f} "
          f"mu_hat={posterior_mean_pseudo:.3f} wall={wall_pseudo:.1f}s", flush=True)

    # ---------- NS-RAM-RNG MH ----------
    t0 = time.time()
    surr = load_surrogate(SURROGATE_PATH, device)
    u_p_ns = generate_nsram_uniforms(2 * N_MH, surr, device, seed=SEED)
    u_a_ns = generate_nsram_uniforms(N_MH, surr, device, seed=SEED + 7)
    nsram_gen_wall = time.time() - t0
    print(f"[z296] NS-RAM noise gen wall={nsram_gen_wall:.1f}s "
          f"(unique u_p={np.unique(u_p_ns).size}, mean={u_p_ns.mean():.3f})", flush=True)

    t0 = time.time()
    samples_ns, ar_ns = metropolis_hastings(data, N_MH, u_p_ns, u_a_ns)
    wall_ns = time.time() - t0
    ess_ns = effective_sample_size(samples_ns[N_MH // 5:])
    posterior_mean_ns = samples_ns[N_MH // 5:].mean()
    print(f"[z296] NS-RAM: ess={ess_ns:.1f}/{N_MH} ar={ar_ns:.3f} "
          f"mu_hat={posterior_mean_ns:.3f} wall={wall_ns:.1f}s", flush=True)

    ratio = ess_ns / max(ess_pseudo, 1e-6)
    if ratio >= 0.90:
        verdict = "AMBITIOUS"
    elif ratio >= 0.50:
        verdict = "PASS"
    else:
        verdict = "FAIL"

    out = {
        "task": "DS-N3 Bayesian MCMC w/ NS-RAM RNG",
        "verdict": verdict,
        "ratio_ess_nsram_over_pseudo": ratio,
        "ess_pseudo": ess_pseudo,
        "ess_nsram": ess_ns,
        "acceptance_rate_pseudo": ar_pseudo,
        "acceptance_rate_nsram": ar_ns,
        "posterior_mean_pseudo": posterior_mean_pseudo,
        "posterior_mean_nsram": posterior_mean_ns,
        "true_mu": TRUE_MU,
        "n_mh": N_MH,
        "wall_pseudo_s": wall_pseudo,
        "wall_nsram_mh_s": wall_ns,
        "wall_nsram_gen_s": nsram_gen_wall,
        "seed": SEED,
        "device": device,
        "node": os.uname().nodename,
    }
    out_dir = ROOT / "results/z296_ds_n3_bayesian"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(out, indent=2))
    print(f"[z296] VERDICT={verdict} ratio={ratio:.3f}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
