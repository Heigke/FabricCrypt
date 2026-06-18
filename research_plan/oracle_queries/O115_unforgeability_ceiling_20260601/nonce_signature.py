"""Phase 14C Task A — nonce-keyed live signature.

Fix for Phase 14B replay attack: in 14B the nonce only permutes OUTPUT positions
of an otherwise-static sampling pattern, so an adversary that records one
ikaros signature can replay it forever (static_replay_p0_rate=1.00 in 14B).

Phase 14C: nonce drives WHAT is sampled, not just where it lands.
  - Which CPUs to read c-state usage from   (subset of 8 picked from nonce)
  - Which thermal zones to read              (subset of available)
  - Number / spacing of nanosleep jitter samples
  - Which TSC-burst indices to keep
  - Output position permutation (kept from 14B)

The model is trained with a *paired* (nonce, sig) input. At inference, an
adversary that does not know the nonce ahead of time cannot pre-record a
signature that matches the chip's response to that specific nonce.

Public API:
    sig = NonceSig(host=...)
    v   = sig.read(nonce=b'\\x01\\x02...')    # 64-dim float32

Output is 64-dim (= 32 physical features + 32-dim nonce embedding) so the
classifier can see both the chip response AND the challenge it was responding
to. A wrong-nonce attack therefore looks wrong on TWO axes (signature stats +
nonce-embedding mismatch with the sample pattern actually used).
"""
from __future__ import annotations
import os, sys, time, ctypes, hmac, hashlib, json, socket
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))

RAPL_PKG  = '/sys/class/powercap/intel-rapl:0/energy_uj'
RAPL_CORE = '/sys/class/powercap/intel-rapl:0:0/energy_uj'
THERMAL_ZONES = [f'/sys/class/thermal/thermal_zone{i}/temp' for i in range(12)]
N_CPU = max(1, os.cpu_count() or 8)
CSTATE_DIRS = [f'/sys/devices/system/cpu/cpu{i}/cpuidle' for i in range(N_CPU)]

_libc = ctypes.CDLL('libc.so.6', use_errno=True)
class _Timespec(ctypes.Structure):
    _fields_ = [("s", ctypes.c_long), ("ns", ctypes.c_long)]


def _read_int(path, default=0):
    try:
        with open(path, 'rb') as f:
            return int(f.read())
    except Exception:
        return default


def _available_thermal_zones():
    out = []
    for p in THERMAL_ZONES:
        if os.path.exists(p):
            out.append(p)
    return out


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
        x = (a * 1103515245 + 12345) & 0xFFFFFFFFFFFFFFFF
        b = perf()
        out[i] = b - a
    return out


def _c2c_pingpong(core_a, core_b, n=4):
    """Cheap cross-core latency proxy: switch CPU affinity & rdtsc.
    True c2c requires shared cache lines + atomics, but the per-thread
    perf_counter latency variance across CPUs already gives signal.
    """
    out = np.empty(n, dtype=np.int64)
    pid = os.getpid()
    try:
        for i in range(n):
            try: os.sched_setaffinity(pid, {core_a % N_CPU})
            except Exception: pass
            t0 = time.perf_counter_ns()
            try: os.sched_setaffinity(pid, {core_b % N_CPU})
            except Exception: pass
            t1 = time.perf_counter_ns()
            out[i] = t1 - t0
    finally:
        try: os.sched_setaffinity(pid, set(range(N_CPU)))
        except Exception: pass
    return out


# -------------------- nonce-derived sampling plan --------------------
def derive_plan(nonce: bytes, n_cpus: int, n_zones: int):
    """HMAC-SHA256(nonce) → deterministic sampling plan.

    Returns a dict:
      cpu_subset:   list[int]   (4 distinct cpu indices)
      zone_subset:  list[int]   (up to 3 distinct thermal-zone indices)
      core_pairs:   list[tuple] (2 pairs for c2c pingpong)
      ns_sleep:     int         (nanosleep target ns, 1000..8000)
      ns_count:     int         (4..10)
      tsc_count:    int         (4..10)
      perm:         np.ndarray  (64,)  output permutation
    """
    h = hmac.new(b'phase14c_nonce_sig', nonce, hashlib.sha256).digest()
    # split into 32 bytes; consume deterministically
    rng = np.random.default_rng(np.frombuffer(h[:8], dtype=np.uint64)[0])
    cpu_subset = list(rng.choice(n_cpus, size=min(4, n_cpus), replace=False))
    if n_zones > 0:
        zone_subset = list(rng.choice(n_zones, size=min(3, n_zones), replace=False))
    else:
        zone_subset = []
    core_pairs = []
    for _ in range(2):
        a, b = rng.choice(n_cpus, size=2, replace=False)
        core_pairs.append((int(a), int(b)))
    ns_sleep = int(1000 + (h[16] | (h[17] << 8)) % 7000)   # 1000..8000 ns
    ns_count = int(4 + h[18] % 7)                          # 4..10
    tsc_count = int(4 + h[19] % 7)                         # 4..10
    perm32 = rng.permutation(32)   # true permutation of phys dims
    return {
        'cpu_subset': [int(x) for x in cpu_subset],
        'zone_subset': [int(x) for x in zone_subset],
        'core_pairs': core_pairs,
        'ns_sleep': ns_sleep,
        'ns_count': ns_count,
        'tsc_count': tsc_count,
        'perm': perm32,
        '_hmac8': h[:8],
    }


def nonce_embedding(nonce: bytes, dim: int = 32) -> np.ndarray:
    """Map nonce to a 32-dim unit-norm vector (so classifier sees the challenge)."""
    out = np.empty(dim, dtype=np.float32)
    block = b''
    i = 0
    while len(block) < dim * 4:
        block += hmac.new(b'phase14c_nonce_embed', nonce + bytes([i]), hashlib.sha256).digest()
        i += 1
    raw = np.frombuffer(block[:dim*4], dtype=np.uint32).astype(np.float64)
    # map to [-1,1]
    v = (raw / 2**32) * 2 - 1
    v = v.astype(np.float32)
    n = float(np.linalg.norm(v)) + 1e-8
    return (v / n).astype(np.float32) * np.sqrt(dim).astype(np.float32) * 0.5


# -------------------- main class --------------------
class NonceSig:
    DIM_PHYS = 32
    DIM_NONCE = 32
    DIM = 64
    CAL_DIR = os.path.join(HERE, '_cal')

    def __init__(self, host: str = None, calibrate: bool = True):
        self.host = host or socket.gethostname()
        os.makedirs(self.CAL_DIR, exist_ok=True)
        self.cal_path = os.path.join(self.CAL_DIR, f'cal_{self.host}.json')
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

    # ---------- raw nonce-keyed read ----------
    def _raw_read(self, plan) -> np.ndarray:
        """Produce 32-dim physical feature vector under plan."""
        out = np.zeros(self.DIM_PHYS, dtype=np.float64)
        # block A: power & thermal (5 dims) — always sampled but quantities depend on plan via zones
        now_pkg  = _read_int(RAPL_PKG)
        now_core = _read_int(RAPL_CORE)
        now_t    = time.perf_counter_ns()
        zone_idx0 = plan['zone_subset'][0] if plan['zone_subset'] else 0
        now_temp = _read_int(self.zones[zone_idx0] if self.zones else THERMAL_ZONES[0])
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
        # block B: extra thermal-zone reads (3 dims) — which zones depends on nonce
        for i, zi in enumerate(plan['zone_subset'][:3]):
            out[5+i] = float(_read_int(self.zones[zi])) if zi < self.n_zones else 0.0
        # block C: TSC burst (length nonce-dependent), pack first 8 + stat
        tsc = _tsc_burst(plan['tsc_count'])
        n_pack = min(8, len(tsc))
        out[8:8+n_pack] = tsc[:n_pack].astype(np.float64)
        out[16] = float(tsc.mean())
        out[17] = float(tsc.std())
        # block D: nanosleep burst at nonce-derived ns (4 stat dims)
        ns = _nanosleep_burst(plan['ns_count'], plan['ns_sleep'])
        out[18] = float(ns.mean())
        out[19] = float(ns.std())
        out[20] = float(ns.min())
        out[21] = float(ns.max())
        # block E: c-state usage on nonce-chosen 4 CPUs (4 dims = mean state2 usage)
        for i, ci in enumerate(plan['cpu_subset'][:4]):
            p = os.path.join(CSTATE_DIRS[ci % self.n_cpus], 'state2', 'usage')
            out[22+i] = float(_read_int(p))
        # block F: c2c pingpong for 2 nonce-chosen core pairs (4 dims = mean,std each)
        for i, (a, b) in enumerate(plan['core_pairs'][:2]):
            p = _c2c_pingpong(a, b, n=3)
            out[26+i*2]   = float(p.mean())
            out[27+i*2]   = float(p.std())
        # final stat: nanosleep/tsc ratio
        out[30] = float(ns.mean() / (tsc.mean() + 1.0))
        out[31] = float(plan['ns_sleep'])  # nonce-tied dimension (rotated by perm)
        return out

    def _maybe_calibrate(self, n_samples: int = 60):
        if os.path.exists(self.cal_path):
            try:
                d = json.load(open(self.cal_path))
                self.mu    = np.asarray(d['mu'], dtype=np.float32)
                self.sigma = np.asarray(d['sigma'], dtype=np.float32)
                self.calibrated = True
                return
            except Exception:
                pass
        print(f"[nonce_sig] calibrating ({n_samples}) for host={self.host}", flush=True)
        # calibrate over a random set of nonces so mu/sigma is plan-agnostic
        rng = np.random.default_rng(1234)
        samples = np.empty((n_samples, self.DIM_PHYS), dtype=np.float64)
        for i in range(n_samples):
            nonce = rng.bytes(8)
            plan = derive_plan(nonce, self.n_cpus, self.n_zones)
            samples[i] = self._raw_read(plan)
            time.sleep(0.005)
        self.mu    = samples.mean(axis=0).astype(np.float32)
        self.sigma = (samples.std(axis=0) + 1e-6).astype(np.float32)
        json.dump({'mu': self.mu.tolist(), 'sigma': self.sigma.tolist(),
                   'host': self.host, 'n_samples': n_samples, 't': time.time()},
                  open(self.cal_path, 'w'))
        self.calibrated = True

    def read(self, nonce: bytes, raw: bool = False) -> np.ndarray:
        """Return 64-dim float32: [32 phys features ; 32 nonce embedding].

        If raw=True: skip per-host calibration (mu/sigma). This preserves
        per-chip bias and is needed for cross-chip twin discrimination — the
        whole point of T3 is that two chips look DIFFERENT in raw phys space.
        Calibration is only useful for keeping a model's input bounded; here
        we use a *shared* log-scale normalization that does NOT erase per-host
        identity.
        """
        if not isinstance(nonce, (bytes, bytearray)):
            raise TypeError("nonce must be bytes")
        plan = derive_plan(nonce, self.n_cpus, self.n_zones)
        rr = self._raw_read(plan).astype(np.float32)
        if raw:
            # global log-scale that preserves cross-host differences
            z = np.sign(rr) * np.log1p(np.abs(rr) * 1e-3)
            z = np.clip(z, -8.0, 8.0).astype(np.float32)
        else:
            z = (rr - self.mu) / self.sigma
            z = np.clip(z, -4.0, 4.0)
        z_perm = z[plan['perm']]  # true 32-element permutation
        emb = nonce_embedding(nonce, self.DIM_NONCE)
        return np.concatenate([z_perm, emb], axis=0).astype(np.float32)

    def read_torch(self, nonce: bytes, device='cpu', dtype=None):
        import torch
        v = self.read(nonce)
        t = torch.from_numpy(v).to(device)
        if dtype is not None:
            t = t.to(dtype)
        return t


def fresh_nonce(rng: np.random.Generator = None) -> bytes:
    rng = rng or np.random.default_rng()
    return rng.bytes(8)


def benchmark(n=1000):
    sig = NonceSig()
    rng = np.random.default_rng(0)
    # warm
    for _ in range(50):
        sig.read(rng.bytes(8))
    t0 = time.perf_counter()
    for _ in range(n):
        sig.read(rng.bytes(8))
    dt = (time.perf_counter() - t0) / n
    print(f"[nonce_sig bench] {dt*1e6:.1f} us/read  ({n} reads, fresh nonce each)")
    return dt


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == 'bench':
        benchmark()
    else:
        s = NonceSig()
        rng = np.random.default_rng(0)
        for _ in range(3):
            n = rng.bytes(8)
            v = s.read(n)
            print(f"nonce={n.hex()} sig[:6]={v[:6]} norm={float(np.linalg.norm(v)):.2f}")
        benchmark(500)
