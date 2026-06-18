"""I3 — Bayesian-RNG retest using actual NS-RAM Vb dynamics (DS-N10 substrate)
as the RNG source, replacing DS-N15's Voss-McCartney synthetic 1/f.

I2-TUNED cells produce zero spikes (rate_ratio=0). Thus we cannot literally
use "I2-tuned substrate spikes". We instead extract the *substrate Vb time
series* from running the BASELINE NSRAM reservoir (which does spike at
~5× rate variance) and compare:
  - numpy_pcg64  (reference)
  - nsram_voss   (DS-N15 original synthetic 1/f)
  - nsram_substrate (NEW: actual NSRAMReservoir Vb trajectory)

10K samples → 2D-Gaussian-posterior MH → KL(empirical||target).

Verify: KL_substrate ≤ 0.005 (DS-N15 published "0.00147" target).

Outputs:
  results/I3_bayes_tuned/summary.json
  results/I3_bayes_tuned/posterior.png
"""
from __future__ import annotations
import json
import sys
import time
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "results" / "I3_bayes_tuned"
OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(REPO / "scripts"))

from DS_N10_reservoir import NSRAMReservoir  # noqa: E402
from DS_N15_nsram_stochastic import NSRAMNoiseEnsemble  # noqa: E402


# ---- Substrate Vb RNG: built from actual NS-RAM reservoir dynamics
class SubstrateRNG:
    name = "nsram_substrate"

    def __init__(self, n_cells=512, n_ticks=2000, seed=0):
        rng = np.random.default_rng(seed)
        # Run NSRAM reservoir with random scalar input to populate Vb trajectory
        r = NSRAMReservoir(N=n_cells, n_readout=n_cells, seed=seed)
        u = rng.standard_normal(n_ticks) * 0.5
        feats = r.run(u)  # (n_ticks, n_cells)
        # Use Vb dynamics as raw analog noise source
        thr = np.median(feats, axis=0, keepdims=True)
        raw = (feats > thr).astype(np.uint8).ravel()
        # von-Neumann debias
        if raw.size & 1:
            raw = raw[:-1]
        a = raw[0::2]; b = raw[1::2]
        keep = a != b
        self._bits = b[keep].astype(np.uint8)
        self._cursor = 0
        print(f"  SubstrateRNG: harvested {self._bits.size} debiased bits "
              f"from {n_ticks} ticks × {n_cells} cells", flush=True)

    def bits(self, n):
        while self._cursor + n > self._bits.size:
            self._bits = np.concatenate([self._bits, self._bits])
        out = self._bits[self._cursor:self._cursor + n]
        self._cursor += n
        return out

    def uniforms(self, n):
        bits = self.bits(n * 32).reshape(n, 32)
        weights = (1 << np.arange(32, dtype=np.uint64))[::-1]
        ints = (bits.astype(np.uint64) * weights).sum(axis=1)
        return (ints.astype(np.float64) + 0.5) / (1 << 32)

    def normals(self, n):
        m = (n + 1) // 2 * 2
        u = self.uniforms(m).clip(1e-12, 1 - 1e-12)
        u1, u2 = u[:m // 2], u[m // 2:]
        r = np.sqrt(-2.0 * np.log(u1))
        z = np.concatenate([r * np.cos(2 * np.pi * u2), r * np.sin(2 * np.pi * u2)])
        return z[:n]


class NumpyRNG:
    name = "numpy_pcg64"
    def __init__(self, seed=0):
        self.r = np.random.default_rng(seed)
    def uniforms(self, n): return self.r.uniform(size=n)
    def normals(self, n): return self.r.standard_normal(size=n)


class VossRNG:
    name = "nsram_voss"
    def __init__(self, seed=0, n_cells=1024):
        self.ens = NSRAMNoiseEnsemble(n_cells=n_cells, seed=seed)
    def uniforms(self, n): return self.ens.uniforms(n)
    def normals(self, n): return self.ens.normals(n)


# 2D Gaussian posterior — Metropolis-Hastings
def mh_2d_gauss(rng, n_iter=10000, target_mean=np.array([1.0, -0.5]),
                target_cov=np.array([[1.0, 0.7], [0.7, 1.5]]), step=0.6):
    inv = np.linalg.inv(target_cov)
    def logp(x):
        d = x - target_mean
        return -0.5 * d @ inv @ d
    samples = np.zeros((n_iter, 2))
    x = target_mean.copy()
    lp = logp(x)
    n_acc = 0
    props_n = rng.normals(n_iter * 2).reshape(n_iter, 2) * step
    accs = rng.uniforms(n_iter)
    for i in range(n_iter):
        cand = x + props_n[i]
        lp_c = logp(cand)
        if np.log(max(accs[i], 1e-300)) < lp_c - lp:
            x = cand; lp = lp_c; n_acc += 1
        samples[i] = x
    return samples, n_acc / n_iter


def kl_emp_target(samples, target_mean, target_cov):
    emp_mean = samples.mean(axis=0)
    emp_cov = np.cov(samples.T) + 1e-6 * np.eye(2)
    inv_t = np.linalg.inv(target_cov)
    d = emp_mean - target_mean
    k = 2
    sign_t, logdet_t = np.linalg.slogdet(target_cov)
    sign_e, logdet_e = np.linalg.slogdet(emp_cov)
    kl = 0.5 * (np.trace(inv_t @ emp_cov) + d @ inv_t @ d - k + logdet_t - logdet_e)
    return float(kl)


SEEDS = [0, 1, 2]
TARGET_MEAN = np.array([1.0, -0.5])
TARGET_COV = np.array([[1.0, 0.7], [0.7, 1.5]])
N_ITER = 10000

results = {}
t0 = time.time()
for name, mk in [("numpy_pcg64", NumpyRNG),
                 ("nsram_voss", VossRNG),
                 ("nsram_substrate", SubstrateRNG)]:
    kls = []; accs = []
    for s in SEEDS:
        print(f"[I3] {name} seed={s} ...", flush=True)
        rng = mk(seed=s)
        samp, acc = mh_2d_gauss(rng, n_iter=N_ITER,
                                 target_mean=TARGET_MEAN, target_cov=TARGET_COV)
        kl = kl_emp_target(samp, TARGET_MEAN, TARGET_COV)
        kls.append(kl); accs.append(acc)
        print(f"  KL={kl:.5f} accept={acc:.2f}", flush=True)
    results[name] = {
        "kl_mean": float(np.mean(kls)), "kl_std": float(np.std(kls)),
        "kl_seeds": kls, "accept_seeds": accs,
    }

elapsed = time.time() - t0

# Plot
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)
for i, (name, _mk) in enumerate([("numpy_pcg64", NumpyRNG),
                                 ("nsram_voss", VossRNG),
                                 ("nsram_substrate", SubstrateRNG)]):
    rng = _mk(seed=0)
    samp, _ = mh_2d_gauss(rng, n_iter=N_ITER,
                          target_mean=TARGET_MEAN, target_cov=TARGET_COV)
    axes[i].hexbin(samp[:, 0], samp[:, 1], gridsize=40, cmap="viridis")
    axes[i].set_title(f"{name}\nKL={results[name]['kl_mean']:.4f}")
    axes[i].set_xlabel("x1"); axes[i].set_ylabel("x2")
fig.suptitle("I3 — 2D-Gaussian posterior, NS-RAM substrate RNG vs numpy/Voss")
fig.savefig(OUT / "posterior.png", dpi=130)
plt.close(fig)

gate_pass = results["nsram_substrate"]["kl_mean"] <= 0.005
summary = {
    "n_iter": N_ITER, "n_seeds": len(SEEDS), "wall_s": elapsed,
    "target_mean": TARGET_MEAN.tolist(), "target_cov": TARGET_COV.tolist(),
    "results": results,
    "gate": {
        "substrate_kl_le_0p005": bool(gate_pass),
        "substrate_kl_mean": results["nsram_substrate"]["kl_mean"],
        "voss_kl_mean": results["nsram_voss"]["kl_mean"],
        "numpy_kl_mean": results["numpy_pcg64"]["kl_mean"],
        "dsn15_published_kl": 0.00147,
    },
    "note": ("I2-tuned cells emit no spikes; substrate RNG built from baseline "
             "NSRAM reservoir Vb trajectory + median-threshold + von-Neumann debias."),
}
(OUT / "summary.json").write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2))
