"""Phase 21B — STRICT thermal common utilities.

Differences vs Phase 21:
  - Default thermal band: 68 abort / 62 pause / 50 cool (much stricter)
  - thermal_guard() exits cleanly (returns abort flag) instead of sleeping forever
  - Smaller, focused sig (only Mechanism 1 — LoRA attention perturbation)
"""
from __future__ import annotations
import os, sys, time, json, hashlib, socket, ctypes
import numpy as np

THERMAL = '/sys/class/thermal/thermal_zone0/temp'
RAPL = '/sys/class/powercap/intel-rapl:0/energy_uj'


def temp_c():
    """Read APU/CPU temperature (mC -> C). Daedalus & ikaros both expose
    thermal_zone0 which is the APU package — what we want for safety."""
    try:
        return int(open(THERMAL).read()) / 1000.0
    except Exception:
        return 0.0


def thermal_guard(abort_c=68, pause_c=62, cool_c=50, wait_max_s=180,
                  verbose=False):
    """STRICT guard. Default = Phase 21B mandate.
      - t >= abort_c: return {'action': 'abort'} — caller MUST checkpoint + exit
      - t >= pause_c: spin-wait until t <= cool_c (with timeout)
    Returns dict with action + temps.
    """
    t = temp_c()
    rec = {'t_start': float(t), 'action': 'ok', 'wait_s': 0.0}
    if t >= abort_c:
        rec['action'] = 'abort'
        if verbose:
            print(f"[THERMAL ABORT] {t:.1f}C >= {abort_c}", flush=True)
        return rec
    if t >= pause_c:
        rec['action'] = 'pause'
        if verbose:
            print(f"[THERMAL PAUSE] {t:.1f}C >= {pause_c} cooling to {cool_c}",
                  flush=True)
        t0 = time.time()
        while temp_c() > cool_c:
            if (time.time() - t0) > wait_max_s:
                rec['action'] = 'abort'  # cooling timeout = abort
                rec['t_end'] = temp_c()
                if verbose:
                    print(f"[THERMAL ABORT] cool timeout, T={temp_c():.1f}C",
                          flush=True)
                return rec
            time.sleep(15)
        rec['t_end'] = temp_c()
        rec['wait_s'] = time.time() - t0
    return rec


def wait_cool(target_c=45, timeout_s=600, verbose=True):
    t0 = time.time()
    while temp_c() > target_c:
        if (time.time() - t0) > timeout_s:
            if verbose:
                print(f"[wait_cool] timeout T={temp_c():.1f}C target={target_c}",
                      flush=True)
            return False
        if verbose and int(time.time() - t0) % 30 == 0:
            print(f"[wait_cool] T={temp_c():.1f}C target={target_c} t={int(time.time()-t0)}s",
                  flush=True)
        time.sleep(15)
    return True


def save_json(path, obj):
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    with open(path, 'w') as f:
        json.dump(obj, f, indent=2, default=str)
    return path


def hostname():
    return socket.gethostname()


# -- LiveSig (32-d) — same shape as Phase 21 but trimmed --
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


def hmac_bytes(key, msg):
    import hmac as _hmac
    return _hmac.new(key, msg, hashlib.sha256).digest()


class LiveSig:
    """32-d per-step chip signature. See Phase 21 for layout details."""
    def __init__(self, nonce=b'phase21b'):
        self.nonce = nonce
        self._last_t = temp_c() * 1000.0
        self._last_rapl = _read_int(RAPL)
        ts = np.log1p(np.abs(_tsc_burst(60)).astype(np.float64))
        js = np.log1p(np.abs(_nanosleep_jitter(30)).astype(np.float64))
        self._tsc_mu = ts.mean(); self._tsc_sd = ts.std() + 1e-6
        self._jit_mu = js.mean(); self._jit_sd = js.std() + 1e-6
        h = hashlib.sha256(socket.gethostname().encode()).digest()
        bits = np.unpackbits(np.frombuffer(h[:8], dtype=np.uint8)).astype(np.float32)
        self._host_bits = (bits[:10] * 2 - 1)
        perm_key = hmac_bytes(self.nonce + h, b'perm32')
        self._perm = np.argsort(np.frombuffer(perm_key, dtype=np.uint8)[:32])

    def read(self):
        t_mc = temp_c() * 1000.0
        dt = t_mc - self._last_t
        self._last_t = t_mc
        r = _read_int(RAPL)
        dr = r - self._last_rapl
        self._last_rapl = r
        frac = time.time() - int(time.time())
        ts = np.log1p(np.abs(_tsc_burst(12)).astype(np.float64))
        ts_z = (ts - self._tsc_mu) / self._tsc_sd
        js = np.log1p(np.abs(_nanosleep_jitter(6)).astype(np.float64))
        js_z = (js - self._jit_mu) / self._jit_sd

        v = np.zeros(32, dtype=np.float32)
        v[0] = np.tanh(t_mc / 100000.0)
        v[1] = np.tanh(dt / 5000.0)
        v[2] = np.tanh(dr / 1e7)
        v[3] = np.sin(2 * np.pi * frac)
        v[4:16] = np.clip(ts_z, -4, 4).astype(np.float32)
        v[16:22] = np.clip(js_z, -4, 4).astype(np.float32)
        v[22:32] = self._host_bits
        v = v[self._perm]
        return v


def sig_to_seed(sig_vec):
    b = np.asarray(sig_vec, dtype=np.float64).tobytes()
    h = hashlib.sha256(b).digest()
    return int.from_bytes(h[:8], 'little')


if __name__ == '__main__':
    sig = LiveSig(nonce=b'test21b')
    for _ in range(5):
        v = sig.read()
    print(f"host={hostname()} T={temp_c():.1f}C dim={v.shape} sample={v[:6]}")
