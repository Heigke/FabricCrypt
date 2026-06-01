"""NonceSig — live nonce-keyed signature reader.

Produces a 64-dim float32 = [32 physical features ; 32 nonce embedding].
The classifier in `protocol/verifier.py` (and the trained MLP if loaded)
sees both:
  - HOW the chip responded (32 physical dims, permuted by nonce)
  - WHICH challenge it was responding to (32 nonce-embedding dims)

A wrong-nonce attack therefore looks wrong on TWO axes.
"""
from __future__ import annotations
import os
import sys
import time
import ctypes
import json
import socket
import numpy as np

from .nonce_derivation import derive_plan, nonce_embedding, fresh_nonce

RAPL_PKG  = "/sys/class/powercap/intel-rapl:0/energy_uj"
RAPL_CORE = "/sys/class/powercap/intel-rapl:0:0/energy_uj"
THERMAL_ZONES = [f"/sys/class/thermal/thermal_zone{i}/temp" for i in range(12)]
N_CPU = max(1, os.cpu_count() or 8)
CSTATE_DIRS = [f"/sys/devices/system/cpu/cpu{i}/cpuidle" for i in range(N_CPU)]

_libc = ctypes.CDLL("libc.so.6", use_errno=True)


class _Timespec(ctypes.Structure):
    _fields_ = [("s", ctypes.c_long), ("ns", ctypes.c_long)]


def _read_int(path, default=0):
    try:
        with open(path, "rb") as f:
            return int(f.read())
    except Exception:
        return default


def _available_thermal_zones():
    return [p for p in THERMAL_ZONES if os.path.exists(p)]


def _nanosleep_burst(n, ns):
    ts = _Timespec(0, ns)
    out = np.empty(n, dtype=np.int64)
    perf = time.perf_counter_ns
    for i in range(n):
        t0 = perf()
        _libc.nanosleep(ctypes.byref(ts), None)
        out[i] = perf() - t0
    return out


def _tsc_burst(n):
    out = np.empty(n, dtype=np.int64)
    perf = time.perf_counter_ns
    for i in range(n):
        a = perf()
        _ = (a * 1103515245 + 12345) & 0xFFFFFFFFFFFFFFFF
        b = perf()
        out[i] = b - a
    return out


def _c2c_pingpong(core_a, core_b, n=4):
    """Cheap cross-core latency proxy: switch CPU affinity and measure jitter."""
    out = np.empty(n, dtype=np.int64)
    pid = os.getpid()
    try:
        for i in range(n):
            try:
                os.sched_setaffinity(pid, {core_a % N_CPU})
            except Exception:
                pass
            t0 = time.perf_counter_ns()
            try:
                os.sched_setaffinity(pid, {core_b % N_CPU})
            except Exception:
                pass
            t1 = time.perf_counter_ns()
            out[i] = t1 - t0
    finally:
        try:
            os.sched_setaffinity(pid, set(range(N_CPU)))
        except Exception:
            pass
    return out


class NonceSig:
    DIM_PHYS = 32
    DIM_NONCE = 32
    DIM = 64

    def __init__(self, host: str = None, cal_dir: str = None,
                 calibrate: bool = True):
        self.host = host or socket.gethostname()
        self.cal_dir = cal_dir or os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "_cal")
        os.makedirs(self.cal_dir, exist_ok=True)
        self.cal_path = os.path.join(self.cal_dir, f"cal_{self.host}.json")
        self.zones = _available_thermal_zones()
        self.n_zones = len(self.zones)
        self.n_cpus = N_CPU
        self._last_rapl_pkg  = _read_int(RAPL_PKG)
        self._last_rapl_core = _read_int(RAPL_CORE)
        self._last_t = time.perf_counter_ns()
        self._last_temp = _read_int(THERMAL_ZONES[0])
        self.mu = np.zeros(self.DIM_PHYS, dtype=np.float32)
        self.sigma = np.ones(self.DIM_PHYS, dtype=np.float32)
        self.calibrated = False
        if calibrate:
            self._maybe_calibrate()

    def _raw_read(self, plan) -> np.ndarray:
        out = np.zeros(self.DIM_PHYS, dtype=np.float64)
        # Block A: power & thermal (5)
        now_pkg  = _read_int(RAPL_PKG)
        now_core = _read_int(RAPL_CORE)
        now_t    = time.perf_counter_ns()
        zone_idx0 = plan["zone_subset"][0] if plan["zone_subset"] else 0
        now_temp = _read_int(self.zones[zone_idx0] if self.zones
                             else THERMAL_ZONES[0])
        dt_ns = max(1, now_t - self._last_t)
        pkg_uW  = (now_pkg  - self._last_rapl_pkg)  * 1e9 / dt_ns
        core_uW = (now_core - self._last_rapl_core) * 1e9 / dt_ns
        temp_mC = float(now_temp)
        temp_d  = float(now_temp - self._last_temp)
        out[0] = pkg_uW; out[1] = core_uW; out[2] = temp_mC; out[3] = temp_d
        out[4] = pkg_uW - core_uW
        self._last_rapl_pkg  = now_pkg
        self._last_rapl_core = now_core
        self._last_temp = now_temp
        self._last_t = now_t
        # Block B: extra thermal zones (3)
        for i, zi in enumerate(plan["zone_subset"][:3]):
            out[5 + i] = float(_read_int(self.zones[zi])) if zi < self.n_zones else 0.0
        # Block C: TSC burst (length nonce-dependent), pack 8 + stats
        tsc = _tsc_burst(plan["tsc_count"])
        n_pack = min(8, len(tsc))
        out[8:8 + n_pack] = tsc[:n_pack].astype(np.float64)
        out[16] = float(tsc.mean())
        out[17] = float(tsc.std())
        # Block D: nanosleep burst (4 stats)
        ns = _nanosleep_burst(plan["ns_count"], plan["ns_sleep"])
        out[18] = float(ns.mean())
        out[19] = float(ns.std())
        out[20] = float(ns.min())
        out[21] = float(ns.max())
        # Block E: c-state usage on chosen CPUs (4 dims)
        for i, ci in enumerate(plan["cpu_subset"][:4]):
            p = os.path.join(CSTATE_DIRS[ci % self.n_cpus], "state2", "usage")
            out[22 + i] = float(_read_int(p))
        # Block F: c2c ping-pong, 2 chosen pairs (4 dims: mean,std each)
        for i, (a, b) in enumerate(plan["core_pairs"][:2]):
            p = _c2c_pingpong(a, b, n=3)
            out[26 + i * 2] = float(p.mean())
            out[27 + i * 2] = float(p.std())
        out[30] = float(ns.mean() / (tsc.mean() + 1.0))
        out[31] = float(plan["ns_sleep"])  # nonce-tied — perm'd by plan
        return out

    def _maybe_calibrate(self, n_samples: int = 60):
        if os.path.exists(self.cal_path):
            try:
                d = json.load(open(self.cal_path))
                self.mu    = np.asarray(d["mu"], dtype=np.float32)
                self.sigma = np.asarray(d["sigma"], dtype=np.float32)
                self.calibrated = True
                return
            except Exception:
                pass
        print(f"[nonce_sig] calibrating ({n_samples}) for host={self.host}",
              flush=True)
        rng = np.random.default_rng(1234)
        samples = np.empty((n_samples, self.DIM_PHYS), dtype=np.float64)
        for i in range(n_samples):
            nonce = rng.bytes(8)
            plan = derive_plan(nonce, self.n_cpus, self.n_zones)
            samples[i] = self._raw_read(plan)
            time.sleep(0.005)
        self.mu    = samples.mean(axis=0).astype(np.float32)
        self.sigma = (samples.std(axis=0) + 1e-6).astype(np.float32)
        json.dump(
            {"mu": self.mu.tolist(), "sigma": self.sigma.tolist(),
             "host": self.host, "n_samples": n_samples, "t": time.time()},
            open(self.cal_path, "w"),
        )
        self.calibrated = True

    def read(self, nonce: bytes, raw: bool = False) -> np.ndarray:
        """Return 64-dim float32: [32 phys ; 32 nonce embedding].

        If raw=True, use a global log-scale that PRESERVES per-host bias
        (needed for cross-chip discrimination). If raw=False, z-normalize
        with this host's calibrated mu/sigma (good for keeping classifier
        inputs bounded; erases some cross-host signal).
        """
        if not isinstance(nonce, (bytes, bytearray)):
            raise TypeError("nonce must be bytes")
        plan = derive_plan(nonce, self.n_cpus, self.n_zones)
        rr = self._raw_read(plan).astype(np.float32)
        if raw:
            z = np.sign(rr) * np.log1p(np.abs(rr) * 1e-3)
            z = np.clip(z, -8.0, 8.0).astype(np.float32)
        else:
            z = (rr - self.mu) / self.sigma
            z = np.clip(z, -4.0, 4.0)
        z_perm = z[plan["perm"]]
        emb = nonce_embedding(nonce, self.DIM_NONCE)
        return np.concatenate([z_perm, emb], axis=0).astype(np.float32)


def benchmark(n: int = 1000):
    sig = NonceSig()
    rng = np.random.default_rng(0)
    for _ in range(50):
        sig.read(rng.bytes(8))
    t0 = time.perf_counter()
    for _ in range(n):
        sig.read(rng.bytes(8))
    dt = (time.perf_counter() - t0) / n
    print(f"[nonce_sig bench] {dt*1e6:.1f} us/read  ({n} reads, fresh nonce)")
    return dt


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "bench":
        benchmark()
    else:
        s = NonceSig()
        rng = np.random.default_rng(0)
        for _ in range(3):
            n = rng.bytes(8)
            v = s.read(n)
            print(f"nonce={n.hex()} sig[:6]={v[:6]} "
                  f"norm={float(np.linalg.norm(v)):.2f}")
        benchmark(500)
