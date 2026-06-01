"""Phase 14D — Patched FabricCrypt signature (Fixes O115 fatal break).

Tier-1 fixes (versus 14C):

  Fix 1 (real measurement):
    - Drop `out[31] = float(plan['ns_sleep'])` (the input parameter).
    - Replace with REAL tail-latency measurement of an actual nanosleep
      burst at the nonce-derived target ns. The verifier checks the
      *measurement* (with proper tolerance) bounded by the requested
      target, NOT the input parameter itself.

  Fix 3 (keyed plan derivation):
    - derive_plan(nonce, K_chip) — K_chip is a per-die secret. Without
      K_chip the attacker cannot compute plan['ns_sleep'], plan['perm'],
      or any other plan component. K_chip is established at enrollment
      and never sent over the wire.

  Fix 4 (independent SHAKE256 streams per plan component):
    - cpu_subset, zone_subset, core_pairs, ns_sleep, ns_count, tsc_count,
      perm32 each consume bytes from an independent domain-separated
      SHAKE256 stream. Eliminates host-coupled RNG-order bug (O115 S1)
      and the all-dim-fill cross-component leak.

Public API:
    sig = NonceSigV2(host=..., K_chip=...)
    v   = sig.read(nonce=b'\\x01\\x02...')   # 64-dim float32

Output is still 64-dim (32 phys + 32 nonce_emb), so the existing
classifier architecture (TwinMLP) works unchanged.
"""
from __future__ import annotations
import os, sys, time, ctypes, hashlib, hmac, json, socket
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
        x = (a * 1103515245 + 12345) & 0xFFFFFFFFFFFFFFFF
        b = perf()
        out[i] = b - a
    return out


def _c2c_pingpong(core_a, core_b, n=4):
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


# -------------------- KEYED PLAN DERIVATION (Fix 3 + Fix 4) --------------------

def _shake_stream(K_chip: bytes, nonce: bytes, domain: bytes, n_bytes: int) -> bytes:
    """Independent SHAKE256 stream per plan component, keyed by K_chip.

    SHAKE256 is an XOF; we domain-separate via a fixed prefix so each
    component consumes bytes from its own pseudo-random stream. Without
    K_chip an attacker cannot reproduce these bytes.
    """
    h = hashlib.shake_256()
    h.update(b"FabricCrypt-v2-plan|")
    h.update(domain)
    h.update(b"|")
    h.update(K_chip)
    h.update(b"|")
    h.update(nonce)
    return h.digest(n_bytes)


def _stream_int_in_range(stream: bytes, lo: int, hi: int) -> int:
    """Reduce stream bytes to a uniform integer in [lo, hi). Simple modulo
    is fine — bias < 2^-64 for ranges ≤ 10^4."""
    x = int.from_bytes(stream[:8], 'little')
    return lo + x % (hi - lo)


def _stream_choice(stream: bytes, n_items: int, k: int) -> list:
    """Sample k distinct indices in [0, n_items) without replacement, from
    deterministic byte stream."""
    # Use Fisher-Yates style: consume 8 bytes per swap.
    pool = list(range(n_items))
    j = 0
    out = []
    for i in range(min(k, n_items)):
        if j + 8 > len(stream):
            # extend if absurdly short — shouldn't happen given caller allocs enough
            return out
        r = int.from_bytes(stream[j:j+8], 'little') % (n_items - i)
        j += 8
        out.append(pool[r])
        pool[r] = pool[-1 - i]
    return out


def _stream_permutation(stream: bytes, n: int) -> np.ndarray:
    """Deterministic permutation of [0,n) from stream (Fisher-Yates)."""
    arr = list(range(n))
    j = 0
    for i in range(n - 1, 0, -1):
        if j + 8 > len(stream):
            break
        r = int.from_bytes(stream[j:j+8], 'little') % (i + 1)
        j += 8
        arr[i], arr[r] = arr[r], arr[i]
    return np.array(arr, dtype=np.int64)


def derive_plan_keyed(nonce: bytes, K_chip: bytes, n_cpus: int, n_zones: int) -> dict:
    """Keyed plan derivation. Each component uses an INDEPENDENT
    SHAKE256(K_chip || domain || nonce) stream (Fix 4)."""
    s_cpu  = _shake_stream(K_chip, nonce, b"cpu_subset", 64)
    s_zone = _shake_stream(K_chip, nonce, b"zone_subset", 64)
    s_pair = _shake_stream(K_chip, nonce, b"core_pairs", 64)
    s_ns   = _shake_stream(K_chip, nonce, b"ns_sleep", 16)
    s_nc   = _shake_stream(K_chip, nonce, b"ns_count", 8)
    s_tc   = _shake_stream(K_chip, nonce, b"tsc_count", 8)
    s_perm = _shake_stream(K_chip, nonce, b"perm32", 256)

    cpu_subset = _stream_choice(s_cpu, n_cpus, min(4, n_cpus))
    zone_subset = _stream_choice(s_zone, max(n_zones, 1), min(3, n_zones)) if n_zones > 0 else []
    core_pairs = []
    pos = 0
    for _ in range(2):
        # consume 16 bytes per pair (8 for a, 8 for b)
        a = int.from_bytes(s_pair[pos:pos+8], 'little') % n_cpus
        b = int.from_bytes(s_pair[pos+8:pos+16], 'little') % max(n_cpus - 1, 1)
        if b >= a: b += 1
        if b >= n_cpus: b = 0
        core_pairs.append((int(a), int(b)))
        pos += 16
    ns_sleep = _stream_int_in_range(s_ns, 1000, 8001)   # 1000..8000 ns
    ns_count = _stream_int_in_range(s_nc, 4, 11)         # 4..10
    tsc_count = _stream_int_in_range(s_tc, 4, 11)        # 4..10
    perm32 = _stream_permutation(s_perm, 32)

    return {
        'cpu_subset': [int(x) for x in cpu_subset],
        'zone_subset': [int(x) for x in zone_subset],
        'core_pairs': core_pairs,
        'ns_sleep': ns_sleep,
        'ns_count': ns_count,
        'tsc_count': tsc_count,
        'perm': perm32,
    }


def nonce_embedding(nonce: bytes, dim: int = 32) -> np.ndarray:
    """Map nonce to a 32-dim unit-norm vector (so classifier sees the challenge).

    Note: this is intentionally PUBLIC (not keyed) — it is a fingerprint of
    the challenge that the classifier consumes; chip-identity comes from
    the physical block (which IS keyed)."""
    block = hashlib.shake_256(b"FabricCrypt-v2-emb|" + nonce).digest(dim * 4)
    raw = np.frombuffer(block, dtype=np.uint32).astype(np.float64)
    v = (raw / 2**32) * 2 - 1
    v = v.astype(np.float32)
    n = float(np.linalg.norm(v)) + 1e-8
    return (v / n).astype(np.float32) * np.sqrt(dim).astype(np.float32) * 0.5


# -------------------- SIGNATURE CLASS --------------------
class NonceSigV2:
    DIM_PHYS = 32
    DIM_NONCE = 32
    DIM = 64
    CAL_DIR = os.path.join(HERE, '_cal')

    def __init__(self, host: str = None, K_chip: bytes = None, calibrate: bool = True):
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
        # Provisional K_chip: zeros until calibration completes.
        self.K_chip = K_chip if K_chip is not None else b'\x00' * 32
        if calibrate:
            self._maybe_calibrate()

    def _raw_read(self, plan) -> np.ndarray:
        """Produce 32-dim physical feature vector under plan.

        Fix 1: dim 31 is now a REAL measurement (median absolute deviation
        of a SECOND independent nanosleep burst at plan['ns_sleep']),
        bounded by but distinct from the input parameter.
        """
        out = np.zeros(self.DIM_PHYS, dtype=np.float64)
        # block A: power & thermal
        now_pkg  = _read_int(RAPL_PKG)
        now_core = _read_int(RAPL_CORE)
        now_t    = time.perf_counter_ns()
        zone_idx0 = plan['zone_subset'][0] if plan['zone_subset'] else 0
        now_temp = _read_int(self.zones[zone_idx0] if self.zones else THERMAL_ZONES[0])
        dt_ns = max(1, now_t - self._last_t)
        pkg_uW  = (now_pkg  - self._last_rapl_pkg)  * 1e9 / dt_ns
        core_uW = (now_core - self._last_rapl_core) * 1e9 / dt_ns
        temp_mC = float(now_temp); temp_d = float(now_temp - self._last_temp)
        out[0] = pkg_uW; out[1] = core_uW; out[2] = temp_mC; out[3] = temp_d
        out[4] = pkg_uW - core_uW
        self._last_rapl_pkg  = now_pkg
        self._last_rapl_core = now_core
        self._last_temp = now_temp
        self._last_t = now_t
        # block B: extra thermal-zone reads
        for i, zi in enumerate(plan['zone_subset'][:3]):
            out[5+i] = float(_read_int(self.zones[zi])) if zi < self.n_zones else 0.0
        # block C: TSC burst
        tsc = _tsc_burst(plan['tsc_count'])
        n_pack = min(8, len(tsc))
        out[8:8+n_pack] = tsc[:n_pack].astype(np.float64)
        out[16] = float(tsc.mean()); out[17] = float(tsc.std())
        # block D: nanosleep burst at plan['ns_sleep'] (4 stat dims)
        ns = _nanosleep_burst(plan['ns_count'], plan['ns_sleep'])
        out[18] = float(ns.mean()); out[19] = float(ns.std())
        out[20] = float(ns.min());  out[21] = float(ns.max())
        # block E: c-state usage
        for i, ci in enumerate(plan['cpu_subset'][:4]):
            p = os.path.join(CSTATE_DIRS[ci % self.n_cpus], 'state2', 'usage')
            out[22+i] = float(_read_int(p))
        # block F: c2c pingpong
        for i, (a, b) in enumerate(plan['core_pairs'][:2]):
            p = _c2c_pingpong(a, b, n=3)
            out[26+i*2] = float(p.mean()); out[27+i*2] = float(p.std())
        # final stat: nanosleep/tsc ratio
        out[30] = float(ns.mean() / (tsc.mean() + 1.0))

        # === FIX 1: dim 31 is a REAL measurement, not the input parameter ===
        # Second independent burst at the same target, larger N, capture the
        # median-absolute-deviation (a chip-physical-noise signal that the
        # attacker cannot compute from the nonce). The TARGET is plan['ns_sleep']
        # (kept secret via K_chip, so attacker doesn't even know which burst
        # to emulate), but the OBSERVED measurement is the chip's physical
        # response to that target — never the input parameter itself.
        ns2 = _nanosleep_burst(max(plan['ns_count'], 8), plan['ns_sleep'])
        med = float(np.median(ns2))
        mad = float(np.median(np.abs(ns2 - med)))
        out[31] = mad  # chip-physical jitter signature for this plan
        return out

    def _maybe_calibrate(self, n_samples: int = 60):
        from key_derivation import derive_kchip, save_kchip
        if os.path.exists(self.cal_path):
            try:
                d = json.load(open(self.cal_path))
                self.mu    = np.asarray(d['mu'], dtype=np.float32)
                self.sigma = np.asarray(d['sigma'], dtype=np.float32)
                self.calibrated = True
                # rebuild K_chip from cached mu
                self.K_chip = derive_kchip(self.mu, host=self.host)
                save_kchip(self.K_chip, self.host)
                return
            except Exception:
                pass
        print(f"[nonce_sig_v2] calibrating ({n_samples}) for host={self.host}", flush=True)
        # Calibration uses a RANDOMIZED keyed-plan with a temporary K_chip=zeros
        # (we don't yet know K_chip; calibration discovers the fingerprint).
        # mu/sigma are plan-agnostic (averaged over many random plans).
        rng = np.random.default_rng(1234)
        K0 = b'\x00' * 32   # calibration phase uses zero-key
        samples = np.empty((n_samples, self.DIM_PHYS), dtype=np.float64)
        for i in range(n_samples):
            nonce = rng.bytes(8)
            plan = derive_plan_keyed(nonce, K0, self.n_cpus, self.n_zones)
            samples[i] = self._raw_read(plan)
            time.sleep(0.005)
        self.mu    = samples.mean(axis=0).astype(np.float32)
        self.sigma = (samples.std(axis=0) + 1e-6).astype(np.float32)
        json.dump({'mu': self.mu.tolist(), 'sigma': self.sigma.tolist(),
                   'host': self.host, 'n_samples': n_samples, 't': time.time()},
                  open(self.cal_path, 'w'))
        self.calibrated = True
        # Now derive K_chip from the stable fingerprint (fuzzy extractor)
        self.K_chip = derive_kchip(self.mu, host=self.host)
        save_kchip(self.K_chip, self.host)
        print(f"[nonce_sig_v2] K_chip derived (32B) and sealed locally", flush=True)

    def read(self, nonce: bytes, raw: bool = True) -> np.ndarray:
        """64-dim signature under (nonce, K_chip)."""
        if not isinstance(nonce, (bytes, bytearray)):
            raise TypeError("nonce must be bytes")
        plan = derive_plan_keyed(nonce, self.K_chip, self.n_cpus, self.n_zones)
        rr = self._raw_read(plan).astype(np.float32)
        if raw:
            z = np.sign(rr) * np.log1p(np.abs(rr) * 1e-3)
            z = np.clip(z, -8.0, 8.0).astype(np.float32)
        else:
            z = (rr - self.mu) / self.sigma
            z = np.clip(z, -4.0, 4.0)
        z_perm = z[plan['perm']]
        emb = nonce_embedding(nonce, self.DIM_NONCE)
        return np.concatenate([z_perm, emb], axis=0).astype(np.float32)


def fresh_nonce(rng=None) -> bytes:
    """Production NOTE: real verifier should use secrets.token_bytes(16).
    For benchmark parity with 14C we use 8-byte nonces here."""
    if rng is None:
        # Use cryptographically-secure source (closes O115 S2).
        import secrets
        return secrets.token_bytes(8)
    return rng.bytes(8)


if __name__ == '__main__':
    s = NonceSigV2()
    rng = np.random.default_rng(0)
    for _ in range(3):
        n = rng.bytes(8)
        v = s.read(n)
        print(f"nonce={n.hex()} sig[:6]={v[:6]} norm={float(np.linalg.norm(v)):.2f}")
