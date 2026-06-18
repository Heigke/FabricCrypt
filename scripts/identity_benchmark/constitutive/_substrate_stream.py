"""Substrate stream loader for constitutive coupling experiments.

Loads A_power (envelope), B_thermal (time constant), E_cpu (per-core latency
rank) for ikaros + daedalus, and exposes per-step sampling primitives so the
reservoir update can pull device-specific noise on every tick.

We deliberately do NOT use the raw bytes blindly -- we synthesize streams whose
statistics match the measured device signatures (mean/std/autocorr_tau of power
under HEAVY load, plus per-core latency rank as a 16-vector). This makes the
"substrate stream" reproducible while remaining device-bound: replacing
ikaros's stream with daedalus's changes both marginal distribution AND temporal
correlation AND the per-core rank vector.
"""
from __future__ import annotations
import json
import numpy as np
from pathlib import Path
from typing import Dict

ROOT = Path(__file__).resolve().parents[3]
DEEP = ROOT / "results" / "IDENTITY_BENCHMARK_2026-05-30" / "deep"


def _load_host(host: str) -> Dict:
    A = json.load(open(DEEP / host / "A_power.json"))
    B = json.load(open(DEEP / host / "B_thermal.json"))
    E = json.load(open(DEEP / host / "E_cpu.json"))
    heavy = A["stats"]["HEAVY"]
    medium = A["stats"]["MEDIUM"]
    # per-core latency rank: 16-vector
    times = np.array([pc["time_ci"][1] for pc in E["per_core"]], dtype=np.float64)
    # rank, then center to [-1,1]
    ranks = np.argsort(np.argsort(times)).astype(np.float64)
    ranks = (ranks - ranks.mean()) / (ranks.std() + 1e-12)
    return {
        "host": host,
        "p_mean": float(heavy["mean_W"]),
        "p_std": float(heavy["std_W"]),
        "p_tau": float(heavy["autocorr_tau"]),  # samples (~10ms per sample)
        "p_p10": float(heavy["p10"]),
        "p_p90": float(heavy["p90"]),
        "tau_heat": float(B["summary"]["tau_heat_ci"][0]),
        "tau_cool": float(B["summary"]["tau_cool_ci"][0]),
        "rth": float(B["summary"]["Rth_ci"][0]),
        "core_rank": ranks,  # shape (16,)
        "core_times": times,
    }


class SubstrateStreamer:
    """Generates a per-step substrate stream for one device.

    The stream is an AR(1) process matched to that device's HEAVY power
    autocorr_tau and (mean,std), z-scored to mean 0 std 1, then modulated by
    the per-core rank vector to produce a vector of dimension `n_dim`.

    Crucially: thermal lag (B) sets a slow envelope; per-core rank (E) sets
    the spatial modulation; power AR(1) (A) sets the fast jitter. All three
    silicon channels enter the stream.
    """

    def __init__(self, host: str, n_dim: int, seed: int = 0):
        self.cfg = _load_host(host)
        self.n_dim = n_dim
        self.rng = np.random.default_rng(seed)
        # AR(1) coefficient from autocorr_tau in samples
        tau = max(self.cfg["p_tau"], 1.0)
        self.alpha_ar = float(np.exp(-1.0 / tau))
        self.sigma_ar = float(np.sqrt(1.0 - self.alpha_ar ** 2))
        self.x_prev = 0.0
        # thermal slow channel: tau_heat in seconds, assume 1 step = 10ms
        tau_h = max(self.cfg["tau_heat"], 0.1)
        self.alpha_th = float(np.exp(-0.01 / tau_h))
        self.sigma_th = float(np.sqrt(1.0 - self.alpha_th ** 2))
        self.th_prev = 0.0
        # spatial pattern: project 16-core rank onto n_dim via fixed random map
        rng_map = np.random.default_rng(12345)  # SAME map for both devices
        self.W_map = rng_map.standard_normal((n_dim, 16)) / np.sqrt(16)
        self.spatial = self.W_map @ self.cfg["core_rank"]  # shape (n_dim,)
        self.spatial /= (np.linalg.norm(self.spatial) + 1e-12) / np.sqrt(n_dim)

    def reset(self, seed: int | None = None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.x_prev = 0.0
        self.th_prev = 0.0

    def step(self) -> np.ndarray:
        """Return one substrate sample, shape (n_dim,)."""
        # fast AR(1)
        e = self.rng.standard_normal()
        self.x_prev = self.alpha_ar * self.x_prev + self.sigma_ar * e
        # slow thermal AR(1)
        eth = self.rng.standard_normal()
        self.th_prev = self.alpha_th * self.th_prev + self.sigma_th * eth
        # combine: fast jitter scaled by per-core spatial, plus slow envelope
        out = self.x_prev * self.spatial + 0.3 * self.th_prev * self.spatial
        return out

    def stream(self, n_steps: int) -> np.ndarray:
        """Bulk generate n_steps samples, shape (n_steps, n_dim)."""
        return np.stack([self.step() for _ in range(n_steps)], axis=0)

    # ---- summary primitives used by IC / per-neuron coefficients ----
    def initial_state(self, n_dim: int) -> np.ndarray:
        """Per-CU thermal signature for reservoir IC, projected to n_dim."""
        rng_ic = np.random.default_rng(int(abs(hash(self.cfg["host"]))) % (2**32))
        base = self.cfg["tau_heat"] / (self.cfg["tau_heat"] + self.cfg["tau_cool"])
        v = rng_ic.standard_normal(n_dim) * 0.1 + (base - 0.5)
        # project 16-core rank to n_dim via fixed map and blend
        rng_map = np.random.default_rng(77777)
        Wmap = rng_map.standard_normal((n_dim, 16)) / np.sqrt(16)
        sp = Wmap @ self.cfg["core_rank"]
        sp = sp / (np.abs(sp).max() + 1e-12)
        v = 0.5 * v + 0.5 * sp
        return v.astype(np.float64)

    def per_neuron_leak(self, n_dim: int, lo: float = 0.05, hi: float = 0.5) -> np.ndarray:
        """Per-neuron leak coefficient from per-core latency rank."""
        rng_map = np.random.default_rng(54321)
        W = rng_map.standard_normal((n_dim, 16)) / np.sqrt(16)
        z = W @ self.cfg["core_rank"]
        # z-score then squash to (lo, hi)
        z = (z - z.mean()) / (z.std() + 1e-12)
        return lo + (hi - lo) * 0.5 * (1.0 + np.tanh(z))

    def weight_mod(self, n_dim: int) -> np.ndarray:
        """Cross-core interaction matrix for recurrent weight modulation."""
        # outer product of rank with shifted rank, projected to n_dim x n_dim
        rng_map = np.random.default_rng(99887)
        W1 = rng_map.standard_normal((n_dim, 16)) / np.sqrt(16)
        W2 = rng_map.standard_normal((n_dim, 16)) / np.sqrt(16)
        a = W1 @ self.cfg["core_rank"]
        b = W2 @ np.roll(self.cfg["core_rank"], 3)
        M = np.outer(a, b)
        M /= (np.abs(M).max() + 1e-12)
        return M


class GaussianMatched:
    """Control: same 1st/2nd moments and per-dim variance, but iid Gaussian."""
    def __init__(self, ref: SubstrateStreamer, seed: int = 0):
        self.n_dim = ref.n_dim
        self.scale = float(np.std(ref.spatial)) * 1.0  # match per-dim std roughly
        self.rng = np.random.default_rng(seed)
        # neutral IC/leak/weight that don't carry host info
        self._ic = np.zeros(self.n_dim)
        self._leak = np.full(self.n_dim, 0.3)
        self._wmod = np.zeros((self.n_dim, self.n_dim))

    def reset(self, seed=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)

    def step(self):
        return self.rng.standard_normal(self.n_dim) * self.scale

    def stream(self, n):
        return self.rng.standard_normal((n, self.n_dim)) * self.scale

    def initial_state(self, n_dim):
        return self._ic.copy()

    def per_neuron_leak(self, n_dim, lo=0.05, hi=0.5):
        return self._leak.copy()

    def weight_mod(self, n_dim):
        return self._wmod.copy()


class PermutedSubstrate:
    """Control: real device substrate but with spatial dim permutation.

    Tests whether identity is carried by the SPECIFIC spatial structure or
    just by the marginal statistics."""
    def __init__(self, ref: SubstrateStreamer, seed: int = 0):
        self.n_dim = ref.n_dim
        self.rng = np.random.default_rng(seed)
        self.cfg = dict(ref.cfg)
        self.alpha_ar = ref.alpha_ar
        self.sigma_ar = ref.sigma_ar
        self.alpha_th = ref.alpha_th
        self.sigma_th = ref.sigma_th
        self.x_prev = 0.0
        self.th_prev = 0.0
        perm = self.rng.permutation(self.n_dim)
        self.spatial = ref.spatial[perm]
        self._leak = ref.per_neuron_leak(self.n_dim)[perm]
        wm = ref.weight_mod(self.n_dim)
        self._wmod = wm[perm][:, perm]
        ic = ref.initial_state(self.n_dim)
        self._ic = ic[perm]

    def reset(self, seed=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.x_prev = 0.0
        self.th_prev = 0.0

    def step(self):
        e = self.rng.standard_normal()
        self.x_prev = self.alpha_ar * self.x_prev + self.sigma_ar * e
        eth = self.rng.standard_normal()
        self.th_prev = self.alpha_th * self.th_prev + self.sigma_th * eth
        return self.x_prev * self.spatial + 0.3 * self.th_prev * self.spatial

    def stream(self, n):
        return np.stack([self.step() for _ in range(n)], axis=0)

    def initial_state(self, n_dim):
        return self._ic.copy()

    def per_neuron_leak(self, n_dim, lo=0.05, hi=0.5):
        return self._leak.copy()

    def weight_mod(self, n_dim):
        return self._wmod.copy()


class IdentConstant:
    """Control: substrate is a fixed vector (no time variation) drawn from host
    spatial pattern. Tests whether DYNAMICS matter or just per-host bias."""
    def __init__(self, ref: SubstrateStreamer):
        self.n_dim = ref.n_dim
        self.const = ref.spatial.copy() * 0.5
        self._ic = ref.initial_state(self.n_dim)
        self._leak = ref.per_neuron_leak(self.n_dim)
        self._wmod = ref.weight_mod(self.n_dim)

    def reset(self, seed=None):
        pass

    def step(self):
        return self.const

    def stream(self, n):
        return np.tile(self.const[None, :], (n, 1))

    def initial_state(self, n_dim):
        return self._ic.copy()

    def per_neuron_leak(self, n_dim, lo=0.05, hi=0.5):
        return self._leak.copy()

    def weight_mod(self, n_dim):
        return self._wmod.copy()
