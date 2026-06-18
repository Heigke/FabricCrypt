"""Phase 21 — shared utilities. Self-contained (no Phase 14 import needed
on remote daedalus). LiveSig replaced by a portable host-signature reader
that produces a 32-d feature vector from sysfs + TSC + thermal."""
from __future__ import annotations
import os, sys, time, json, hashlib, socket, ctypes
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
THERMAL = '/sys/class/thermal/thermal_zone0/temp'
RAPL = '/sys/class/powercap/intel-rapl:0/energy_uj'

# Prefer amdgpu hwmon (real GPU edge temp on daedalus) when available;
# fall back to acpitz. Auto-detected on first call.
_TEMP_CANDIDATES = [
    '/sys/class/hwmon/hwmon7/temp1_input',  # amdgpu on daedalus
    '/sys/class/thermal/thermal_zone0/temp',  # ikaros + fallback
]
_TEMP_PATH = None
def _pick_temp_path():
    global _TEMP_PATH
    if _TEMP_PATH:
        return _TEMP_PATH
    # On daedalus, hwmon7 = amdgpu (correct). On ikaros, thermal_zone0 = APU.
    host = socket.gethostname()
    if host == 'daedalus':
        for p in ['/sys/class/hwmon/hwmon7/temp1_input',
                  '/sys/class/hwmon/hwmon0/temp1_input']:
            if os.path.exists(p):
                # Check it's amdgpu-ish (sane reading)
                try:
                    t = int(open(p).read()) / 1000.0
                    if 20 < t < 75:  # reasonable GPU/CPU temp range
                        _TEMP_PATH = p
                        return p
                except Exception:
                    continue
    _TEMP_PATH = THERMAL
    return _TEMP_PATH


def temp_c():
    try:
        return int(open(_pick_temp_path()).read()) / 1000.0
    except Exception:
        return 0.0


def thermal_guard(abort_c=80, pause_c=72, cool_c=65, wait_max_s=120,
                  verbose=False):
    """Phase 21 — daedalus is cool (~50C under load), ikaros runs hot.
    Default band sized for daedalus; ikaros runs use stricter (65/60/55).
    """
    t = temp_c()
    rec = {'t_start': t, 'action': 'ok'}
    if t >= abort_c:
        rec['action'] = 'abort_cool'
        if verbose:
            print(f"[THERMAL ABORT-COOL] {t:.1f}C >= {abort_c}", flush=True)
        t0 = time.time()
        while temp_c() > cool_c:
            if (time.time() - t0) > wait_max_s:
                rec['t_end'] = temp_c()
                raise SystemExit(f"[THERMAL ABORT] {temp_c():.1f}C")
            time.sleep(5)
        rec['t_end'] = temp_c()
        return rec
    if t >= pause_c:
        rec['action'] = 'pause_cool'
        if verbose:
            print(f"[THERMAL PAUSE] {t:.1f}C >= {pause_c}", flush=True)
        t0 = time.time()
        while temp_c() > cool_c:
            if (time.time() - t0) > wait_max_s:
                break
            time.sleep(3)
        rec['t_end'] = temp_c()
    return rec


def wait_cool(target_c=55, timeout_s=120, verbose=False):
    t0 = time.time()
    while temp_c() > target_c:
        if (time.time() - t0) > timeout_s:
            return False
        time.sleep(3)
    return True


def save_json(path, obj):
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    with open(path, 'w') as f:
        json.dump(obj, f, indent=2, default=str)
    print(f"[save] {path}", flush=True)
    return path


def hostname():
    return socket.gethostname()


# ----- Portable chip signature (works on ikaros & daedalus) -----
_libc = ctypes.CDLL('libc.so.6', use_errno=True)
class _Ts(ctypes.Structure):
    _fields_ = [("s", ctypes.c_long), ("ns", ctypes.c_long)]


def _tsc_burst(n=12):
    out = np.empty(n, dtype=np.int64)
    perf = time.perf_counter_ns
    for i in range(n):
        a = perf()
        x = (a * 1103515245 + 12345) & 0xFFFFFFFFFFFFFFFF
        b = perf()
        out[i] = b - a
    return out


def _nanosleep_jitter(n=6, ns=2000):
    ts = _Ts(0, ns)
    out = np.empty(n, dtype=np.int64)
    perf = time.perf_counter_ns
    for i in range(n):
        t0 = perf()
        _libc.nanosleep(ctypes.byref(ts), None)
        out[i] = perf() - t0
    return out


def _read_int(path, default=0):
    try:
        with open(path, 'rb') as f:
            return int(f.read())
    except Exception:
        return default


class LiveSig:
    """Portable 32-d live chip signature.
    Layout (raw -> permuted by HMAC(nonce)):
      0:   thermal mC (z-scored against running mean)
      1:   thermal derivative
      2:   rapl_uj delta (z-scored)
      3:   wall_time fractional second
      4-15: 12 tsc-burst delta values (log + z-scored)
      16-21: 6 nanosleep jitter values (log + z-scored)
      22-31: cpu hash bits derived from hostname+procfs cpu count
    """
    def __init__(self, nonce=b'phase21'):
        self.nonce = nonce
        self._last_t = temp_c() * 1000.0
        self._last_rapl = _read_int(RAPL)
        self._last_time = time.time()
        # baseline TSC/jitter stats (for z-scoring)
        ts = np.log1p(np.abs(_tsc_burst(60)).astype(np.float64))
        js = np.log1p(np.abs(_nanosleep_jitter(30)).astype(np.float64))
        self._tsc_mu = ts.mean(); self._tsc_sd = ts.std() + 1e-6
        self._jit_mu = js.mean(); self._jit_sd = js.std() + 1e-6
        # host hash bits
        h = hashlib.sha256(socket.gethostname().encode()).digest()
        bits = np.unpackbits(np.frombuffer(h[:8], dtype=np.uint8)).astype(np.float32)
        # 64 bits -> take first 10 as features (scaled to [-1,+1])
        self._host_bits = (bits[:10] * 2 - 1)
        # permutation key from nonce+host
        perm_key = hmac_bytes(self.nonce + h, b'perm32')
        self._perm = np.argsort(np.frombuffer(perm_key, dtype=np.uint8)[:32])

    def read(self):
        # 1: thermal
        t_mc = temp_c() * 1000.0
        dt = t_mc - self._last_t
        self._last_t = t_mc
        # 2: rapl
        r = _read_int(RAPL)
        dr = r - self._last_rapl
        self._last_rapl = r
        # 3: wall frac
        now = time.time()
        frac = now - int(now)
        self._last_time = now
        # 4: TSC
        ts = np.log1p(np.abs(_tsc_burst(12)).astype(np.float64))
        ts_z = (ts - self._tsc_mu) / self._tsc_sd
        # 5: jitter
        js = np.log1p(np.abs(_nanosleep_jitter(6)).astype(np.float64))
        js_z = (js - self._jit_mu) / self._jit_sd

        v = np.zeros(32, dtype=np.float32)
        v[0] = np.tanh(t_mc / 100000.0)
        v[1] = np.tanh(dt / 5000.0)
        v[2] = np.tanh(dr / 1e7)
        v[3] = np.sin(2 * np.pi * frac)
        v[4:16] = np.clip(ts_z, -4, 4).astype(np.float32)
        v[16:22] = np.clip(js_z, -4, 4).astype(np.float32)
        v[22:32] = self._host_bits  # 10 host-bit channels
        # apply permutation derived from nonce+host
        v = v[self._perm]
        return v


def hmac_bytes(key, msg):
    import hmac as _hmac
    return _hmac.new(key, msg, hashlib.sha256).digest()


def sig_to_seed(sig_vec):
    b = np.asarray(sig_vec, dtype=np.float64).tobytes()
    h = hashlib.sha256(b).digest()
    return int.from_bytes(h[:8], 'little')


def bootstrap_ci(values, n_boot=1000, alpha=0.05, seed=0):
    rng = np.random.default_rng(seed)
    arr = np.asarray(values, dtype=np.float64)
    n = len(arr)
    if n < 2:
        m = float(arr.mean()) if n else 0.0
        return m, m, m
    idx = rng.integers(0, n, size=(n_boot, n))
    means = arr[idx].mean(axis=1)
    lo, hi = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(arr.mean()), float(lo), float(hi)


if __name__ == '__main__':
    sig = LiveSig(nonce=b'test')
    t0 = time.perf_counter()
    for _ in range(50):
        v = sig.read()
    dt = (time.perf_counter() - t0) / 50
    print(f"sig dim={v.shape}, {dt*1e6:.0f} us/read, host={hostname()}, T={temp_c():.1f}C")
    print(f"sample: {v[:6]}")
