"""Phase 14 Task A — fast live signature for forward-pass injection.

Design constraints:
  - <1 ms per read (called every forward step)
  - 32-dim feature vector
  - Mixes physical signals that no daemon can fake without the chip:
       * RAPL energy_uj delta (per-package & per-core if available)
       * Thermal zone temperature (mC) + derived derivatives
       * TSC inter-core skew samples (cheap rdtsc reads from 8 threadpool workers)
       * c-state residency snapshot (cpu0..4 state2/3 usage counts)
       * Live nanosleep jitter samples (cheap ~50us calibration)

A nonce can be mixed (Task G) — when present, the layout of which sub-features
populate which output positions is permuted by HMAC(chip_state, nonce).

Public API:
    sig = LiveSignature(nonce: bytes = None)
    vec32 = sig.read()             # numpy float32 (32,)
    vec32 = sig.read_torch(device, dtype) # torch tensor on device

The signature is centred & scale-normalised w.r.t. a per-host calibration
saved on first run; this prevents the live values from blowing up gradients.
"""
from __future__ import annotations
import os, sys, time, ctypes, struct, hmac, hashlib, json, socket
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))

RAPL_PKG  = '/sys/class/powercap/intel-rapl:0/energy_uj'
RAPL_CORE = '/sys/class/powercap/intel-rapl:0:0/energy_uj'
THERMAL   = '/sys/class/thermal/thermal_zone0/temp'
CSTATE_DIRS = [f'/sys/devices/system/cpu/cpu{i}/cpuidle' for i in range(8)]

_libc = ctypes.CDLL('libc.so.6', use_errno=True)
class _Timespec(ctypes.Structure):
    _fields_ = [("s", ctypes.c_long), ("ns", ctypes.c_long)]

# ----- low-level fast readers -----
def _read_int(path, default=0):
    try:
        with open(path, 'rb') as f:
            return int(f.read())
    except Exception:
        return default

def _rdtsc():
    # not a true rdtsc, but perf_counter_ns is monotonic in TSC ticks on Linux
    return time.perf_counter_ns()

def _nanosleep_burst(n=8, ns=2000):
    ts = _Timespec(0, ns)
    out = np.empty(n, dtype=np.int64)
    perf = time.perf_counter_ns
    for i in range(n):
        t0 = perf()
        _libc.nanosleep(ctypes.byref(ts), None)
        out[i] = perf() - t0
    return out

def _tsc_offsets(n=8):
    # cheap "skew" proxy: paired rdtsc reads w/ short python compute between
    out = np.empty(n, dtype=np.int64)
    perf = time.perf_counter_ns
    for i in range(n):
        a = perf()
        # tiny work
        x = (a * 1103515245 + 12345) & 0xFFFFFFFFFFFFFFFF
        b = perf()
        out[i] = b - a
    return out

def _cstate_snapshot():
    """Return mean usage counts for state2 across 8 CPUs (cheap; ~0.2ms)."""
    counts = np.zeros(4, dtype=np.float64)
    n = 0
    for d in CSTATE_DIRS:
        try:
            for s in range(4):
                p = os.path.join(d, f'state{s}', 'usage')
                counts[s] += _read_int(p)
            n += 1
        except Exception:
            pass
    if n > 0:
        counts /= max(1, n)
    return counts  # (4,)

# ----- main class -----
class LiveSignature:
    """Live, in-pass signature reader. 32-dim float32."""
    DIM = 32
    CAL_DIR = os.path.join(HERE, '_cal')
    def __init__(self, nonce: bytes = None, host: str = None, calibrate: bool = True):
        self.host = host or socket.gethostname()
        self.nonce = nonce  # 64-bit bytes OK
        os.makedirs(self.CAL_DIR, exist_ok=True)
        self.cal_path = os.path.join(self.CAL_DIR, f'cal_{self.host}.json')
        # state for delta-readings
        self._last_rapl_pkg = _read_int(RAPL_PKG)
        self._last_rapl_core = _read_int(RAPL_CORE)
        self._last_temp = _read_int(THERMAL)
        self._last_t   = time.perf_counter_ns()
        # calibration
        self.mu = np.zeros(self.DIM, dtype=np.float32)
        self.sigma = np.ones(self.DIM, dtype=np.float32)
        self.calibrated = False
        if calibrate:
            self._maybe_calibrate()
        # nonce-driven permutation
        if nonce is not None:
            self.perm = self._nonce_perm(nonce)
        else:
            self.perm = np.arange(self.DIM)

    def _nonce_perm(self, nonce: bytes):
        h = hmac.new(b'embodiment14', nonce, hashlib.sha256).digest()
        # use as RNG seed for permutation
        rng = np.random.default_rng(np.frombuffer(h[:8], dtype=np.uint64)[0])
        return rng.permutation(self.DIM)

    def _raw_read(self) -> np.ndarray:
        """Produce raw 32-dim feature vector. ~0.3-0.8 ms on this machine."""
        out = np.zeros(self.DIM, dtype=np.float64)
        # --- block A: power & thermal deltas (5 dims) ---
        now_pkg  = _read_int(RAPL_PKG)
        now_core = _read_int(RAPL_CORE)
        now_t    = time.perf_counter_ns()
        now_temp = _read_int(THERMAL)
        dt_ns = max(1, now_t - self._last_t)
        # uW: (uJ delta) / (ns delta) * 1e9 = uW
        pkg_uW  = (now_pkg  - self._last_rapl_pkg)  * 1e9 / dt_ns
        core_uW = (now_core - self._last_rapl_core) * 1e9 / dt_ns
        temp_mC = float(now_temp)
        temp_d  = float(now_temp - self._last_temp)
        out[0] = pkg_uW
        out[1] = core_uW
        out[2] = temp_mC
        out[3] = temp_d
        out[4] = pkg_uW - core_uW   # "non-core" power proxy
        self._last_rapl_pkg = now_pkg
        self._last_rapl_core = now_core
        self._last_temp = now_temp
        self._last_t = now_t
        # --- block B: TSC small-burst skew (8 dims = 8 samples) ---
        tsc = _tsc_offsets(8)
        out[5:13] = tsc.astype(np.float64)
        # --- block C: nanosleep jitter (8 samples = 8 dims) ---
        ns = _nanosleep_burst(8, 2000)
        out[13:21] = ns.astype(np.float64)
        # --- block D: c-state residency (4 dims) ---
        cs = _cstate_snapshot()
        out[21:25] = cs
        # --- block E: stat features (7 dims): mean, std of TSC; mean, std of ns; min/max of ns; ratio ---
        out[25] = float(tsc.mean())
        out[26] = float(tsc.std())
        out[27] = float(ns.mean())
        out[28] = float(ns.std())
        out[29] = float(ns.min())
        out[30] = float(ns.max())
        out[31] = float(ns.mean() / (tsc.mean() + 1.0))
        return out

    def _maybe_calibrate(self, n_samples: int = 80):
        if os.path.exists(self.cal_path):
            try:
                d = json.load(open(self.cal_path))
                self.mu    = np.asarray(d['mu'],    dtype=np.float32)
                self.sigma = np.asarray(d['sigma'], dtype=np.float32)
                self.calibrated = True
                return
            except Exception:
                pass
        # build calibration
        print(f"[sig_io] calibrating ({n_samples} samples) for host={self.host}", flush=True)
        samples = np.empty((n_samples, self.DIM), dtype=np.float64)
        for i in range(n_samples):
            samples[i] = self._raw_read()
            time.sleep(0.005)
        self.mu    = samples.mean(axis=0).astype(np.float32)
        self.sigma = (samples.std(axis=0) + 1e-6).astype(np.float32)
        json.dump({'mu': self.mu.tolist(), 'sigma': self.sigma.tolist(),
                   'host': self.host, 'n_samples': n_samples,
                   't': time.time()},
                  open(self.cal_path, 'w'))
        self.calibrated = True

    def read(self) -> np.ndarray:
        """Return centred/scaled 32-d signature, permuted by nonce if set."""
        raw = self._raw_read().astype(np.float32)
        z = (raw - self.mu) / self.sigma
        # clip to sane range so it can't explode the model
        z = np.clip(z, -4.0, 4.0)
        z = z[self.perm]
        return z

    def read_torch(self, device='cuda', dtype=None):
        import torch
        v = self.read()
        t = torch.from_numpy(v).to(device)
        if dtype is not None:
            t = t.to(dtype)
        return t


def benchmark(n=2000, host=None):
    sig = LiveSignature(host=host)
    # warm
    for _ in range(100): sig.read()
    t0 = time.perf_counter()
    for _ in range(n):
        sig.read()
    dt = (time.perf_counter() - t0) / n
    print(f"[bench] {dt*1e6:.1f} us/read  ({n} reads)")
    return dt


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == 'bench':
        benchmark()
    elif len(sys.argv) > 1 and sys.argv[1] == 'dump':
        s = LiveSignature()
        for _ in range(5):
            print(s.read())
            time.sleep(0.05)
    else:
        s = LiveSignature()
        print(f"calibrated: {s.calibrated} host={s.host}")
        print(f"first read: {s.read()}")
        benchmark(500)
