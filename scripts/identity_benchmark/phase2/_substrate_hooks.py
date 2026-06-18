"""Substrate hooks for Phase 2 transplant matrix.

Load Phase 1b raw_idle.npz signatures for ikaros and daedalus and expose
samplers for the two SURVIVING channels (per Phase 1b verdict):
  - RTN-rate: per-CU bit-flip rate driving sparse multiplicative perturbations
  - spatial-corr: per-CU 80x80 correlation matrix used as a colored-noise
    covariance for additive perturbations

The KILLED channels (1/f knee, stable-bit) are deliberately NOT exposed.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[3]
DATA_DIR = REPO / "results" / "IDENTITY_BENCHMARK_2026-05-30"


class SubstrateSampler:
    def __init__(self, device: str, seed: int = 0):
        self.device = device
        npz_path = DATA_DIR / device / "raw_idle.npz"
        d = np.load(npz_path)
        self.rtn = np.asarray(d["rtn"], dtype=np.float64)            # (80,) flip rate per CU
        self.spatial = np.asarray(d["spatial_corr"], dtype=np.float64)  # (80,80)
        self.n_cu = self.rtn.shape[0]
        self.rng = np.random.default_rng(seed)
        # Pre-compute spatial Cholesky factor for colored sampling
        cov = self._regularize_psd(self.spatial)
        try:
            self.L = np.linalg.cholesky(cov)
        except np.linalg.LinAlgError:
            # Fallback eigen decomposition with floor
            w, V = np.linalg.eigh(cov)
            w = np.clip(w, 1e-6, None)
            self.L = V @ np.diag(np.sqrt(w))

    @staticmethod
    def _regularize_psd(M: np.ndarray, eps: float = 1e-4) -> np.ndarray:
        M = 0.5 * (M + M.T)
        return M + eps * np.eye(M.shape[0])

    def rtn_perturbation(self, n_neurons: int) -> np.ndarray:
        """Sparse multiplicative gain perturbation per neuron, sourced from
        per-CU flip rates. Tile/cycle CU-rates across n_neurons.
        Returns shape (n_neurons,) values in [1-2*rate, 1+2*rate] when active.
        """
        idx = np.arange(n_neurons) % self.n_cu
        rates = self.rtn[idx]                     # per-neuron base rate
        flip = self.rng.random(n_neurons) < rates  # active flip mask
        sign = self.rng.choice([-1.0, 1.0], size=n_neurons)
        return 1.0 + flip * sign * rates          # gain

    def spatial_noise(self, n_neurons: int, scale: float = 0.05) -> np.ndarray:
        """Colored additive noise with spatial correlation matrix from device.
        Tiles 80-CU pattern to fit n_neurons.
        """
        z = self.rng.standard_normal(self.n_cu)
        colored = self.L @ z                       # (80,)
        # tile / truncate
        reps = (n_neurons + self.n_cu - 1) // self.n_cu
        out = np.tile(colored, reps)[:n_neurons]
        return scale * out


def shuffle_sampler(base: SubstrateSampler, seed: int = 0) -> SubstrateSampler:
    """Return a sampler whose RTN/spatial patterns are randomly shuffled —
    destroys identity while preserving marginal statistics."""
    s = SubstrateSampler.__new__(SubstrateSampler)
    s.device = base.device + "-shuffle"
    rng = np.random.default_rng(seed)
    s.rtn = rng.permutation(base.rtn).copy()
    perm = rng.permutation(base.spatial.shape[0])
    s.spatial = base.spatial[perm][:, perm].copy()
    s.n_cu = base.n_cu
    s.rng = np.random.default_rng(seed + 1)
    cov = SubstrateSampler._regularize_psd(s.spatial)
    try:
        s.L = np.linalg.cholesky(cov)
    except np.linalg.LinAlgError:
        w, V = np.linalg.eigh(cov)
        w = np.clip(w, 1e-6, None)
        s.L = V @ np.diag(np.sqrt(w))
    return s


def matched_rng_sampler(base: SubstrateSampler, seed: int = 0) -> SubstrateSampler:
    """SW-matched-RNG control: same per-CU MARGINAL means/std, but iid Gaussian
    spatial structure (no per-device correlation). Tests whether the SHAPE of
    the substrate matters, not just its scale."""
    s = SubstrateSampler.__new__(SubstrateSampler)
    s.device = base.device + "-iid"
    rng = np.random.default_rng(seed)
    s.rtn = rng.uniform(base.rtn.min(), base.rtn.max() + 1e-9, size=base.n_cu)
    s.spatial = np.eye(base.n_cu)  # uncorrelated
    s.n_cu = base.n_cu
    s.rng = np.random.default_rng(seed + 1)
    s.L = np.eye(base.n_cu)
    return s
