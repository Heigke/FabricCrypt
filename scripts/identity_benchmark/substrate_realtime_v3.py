"""SubstrateState v3 — 10 channels chosen from deep-analysis findings.

Drops from v1/v2 (post-replay-analysis):
  - C07 xtal Δ (chance-level spoof discrimination, AUC 0.48)
  - C11 drift_abs (96% redundant with C11 drift)

Keeps:
  - C07 xtal value (d=66.7, hard die-bound register)
  - C20 lat_xtal (kurtosis=+55, heavy-tail spoof-resistant)
  - C20 lat_logtail
  - C11 TSC drift (kurtosis=+99)

Adds (from h7 first-pass and O100 oracle synthesis):
  - C09 PM[1]  (d=8.11 in first-pass)
  - C09 PM[3]  (AUC=0.997)
  - C09 PM[5]
  - C05_e0 rate (energy-counter delta — physical work)
  - C06 fast counter rate (sub-µs jitter)
  - C20 lat_energy0 (second independent SMN-latency channel)

PM-table is read every PM_DECIMATE steps and held — keeps 500 Hz core loop.
"""
from __future__ import annotations
import ctypes, mmap, os, struct, threading, time
from collections import deque
from pathlib import Path
from typing import Optional
import numpy as np

MMCFG_BASE     = 0xE0000000
SMN_ADDR_OFF   = 0x60
SMN_DATA_OFF   = 0x64
SMN_XTAL_CNTL  = 0x598C8
SMN_ENERGY0    = 0x5B500
SMN_FAST       = 0x58E00
SMN_BASE_TH    = 0x59800

PM_PATH = "/sys/kernel/ryzen_smu_drv/pm_table"
PM_INDICES = (1, 3, 5)


class _MMCFG:
    def __init__(self) -> None:
        self.fd = os.open("/dev/mem", os.O_RDWR | os.O_SYNC)
        self.mm = mmap.mmap(self.fd, 4096, mmap.MAP_SHARED,
                            mmap.PROT_READ | mmap.PROT_WRITE,
                            offset=MMCFG_BASE)

    def smn_read_timed(self, addr: int):
        t0 = time.perf_counter_ns()
        self.mm.seek(SMN_ADDR_OFF); self.mm.write(struct.pack("<I", addr))
        self.mm.seek(SMN_DATA_OFF); v = struct.unpack("<I", self.mm.read(4))[0]
        t1 = time.perf_counter_ns()
        return v, (t1 - t0)

    def smn_read(self, addr: int) -> int:
        self.mm.seek(SMN_ADDR_OFF); self.mm.write(struct.pack("<I", addr))
        self.mm.seek(SMN_DATA_OFF); return struct.unpack("<I", self.mm.read(4))[0]


_libc = ctypes.CDLL("libc.so.6", use_errno=True)
class _Ts(ctypes.Structure):
    _fields_ = [("tv_sec", ctypes.c_long), ("tv_nsec", ctypes.c_long)]
CLOCK_MONOTONIC_RAW = 4


def _clock_raw_ns() -> int:
    ts = _Ts(); _libc.clock_gettime(CLOCK_MONOTONIC_RAW, ctypes.byref(ts))
    return ts.tv_sec * 1_000_000_000 + ts.tv_nsec


def _drift_sample() -> int:
    a1 = time.monotonic_ns(); b1 = _clock_raw_ns()
    a2 = time.monotonic_ns(); b2 = _clock_raw_ns()
    return (a2 - a1) - (b2 - b1)


def _read_pm() -> Optional[np.ndarray]:
    try:
        with open(PM_PATH, "rb") as f:
            raw = f.read()
        n = len(raw) // 4
        return np.frombuffer(raw[:n * 4], dtype=np.float32)
    except Exception:
        return None


class SubstrateStateV3:
    """10-channel ring buffer at ~500Hz.

    Channels:
       0  C07 xtal value          — die-bound register, d=66
       1  C09 PM[1]                — d=8.11 in first-pass
       2  C20 lat_xtal_ns          — heavy-tail SMN latency
       3  C20 lat_xtal_logtail     — signed log-tail
       4  C11 TSC drift_ns         — clock drift, kurt=99
       5  C05_e0 Δ/Δt              — energy counter rate
       6  C06 fast Δ/Δt            — fast-counter rate (sub-µs jitter)
       7  C09 PM[3]                — AUC=0.997
       8  C09 PM[5]                — AUC=1.000
       9  C20 lat_energy0          — independent SMN-latency channel
    """
    N_CHANNELS = 10
    BUF_LEN    = 4096
    PM_DECIMATE = 25   # read PM table every Nth step (~20Hz @ 500Hz core)

    def __init__(self, hz_target: int = 500) -> None:
        self.dt = 1.0 / hz_target
        self.buf = np.zeros((self.N_CHANNELS, self.BUF_LEN), dtype=np.float32)
        self.idx = 0
        self.total_samples = 0
        self.lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._probe: Optional[_MMCFG] = None
        self._lat_win: deque = deque(maxlen=256)

    def start(self) -> None:
        if self._thread is not None: return
        if os.geteuid() != 0:
            raise PermissionError("SubstrateStateV3 needs root for /dev/mem MMCFG")
        self._probe = _MMCFG()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        time.sleep(0.5)

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None: self._thread.join(timeout=2.0)

    def _run(self) -> None:
        step = 0
        prev_e0 = None; prev_fast = None; prev_t = None
        held_pm = np.zeros(max(PM_INDICES) + 1, dtype=np.float32)
        while not self._stop.is_set():
            t0 = time.perf_counter()
            try:
                xtal, lat_xtal = self._probe.smn_read_timed(SMN_XTAL_CNTL)
                e0, lat_e0 = self._probe.smn_read_timed(SMN_ENERGY0)
                fast = self._probe.smn_read(SMN_FAST)
                drift = _drift_sample()
                self._lat_win.append(lat_xtal)
                med = np.median(self._lat_win)
                log_tail = float(np.sign(lat_xtal - med) * np.log1p(abs(lat_xtal - med)))

                # rates
                now = time.perf_counter_ns()
                if prev_t is not None:
                    dt_ns = max(1, now - prev_t)
                    # wrap-safe Δ on uint32 counters
                    e0_d = ((e0 - prev_e0) & 0xFFFFFFFF)
                    fast_d = ((fast - prev_fast) & 0xFFFFFFFF)
                    e0_rate   = e0_d   * 1e9 / dt_ns
                    fast_rate = fast_d * 1e9 / dt_ns
                else:
                    e0_rate = 0.0; fast_rate = 0.0
                prev_e0 = e0; prev_fast = fast; prev_t = now

                # PM (decimated)
                if step % self.PM_DECIMATE == 0:
                    pm = _read_pm()
                    if pm is not None and pm.size > max(PM_INDICES):
                        held_pm = pm

                with self.lock:
                    i = self.idx
                    self.buf[0, i] = xtal
                    self.buf[1, i] = held_pm[1]
                    self.buf[2, i] = lat_xtal
                    self.buf[3, i] = log_tail
                    self.buf[4, i] = drift
                    self.buf[5, i] = e0_rate
                    self.buf[6, i] = fast_rate
                    self.buf[7, i] = held_pm[3]
                    self.buf[8, i] = held_pm[5]
                    self.buf[9, i] = lat_e0
                    self.idx = (i + 1) % self.BUF_LEN
                    self.total_samples += 1
            except Exception:
                pass
            step += 1
            slack = self.dt - (time.perf_counter() - t0)
            if slack > 0: time.sleep(slack)

    def latest_window(self, length: int = 256) -> np.ndarray:
        with self.lock:
            i = self.idx
            n_have = min(self.total_samples, self.BUF_LEN)
            if n_have == 0:
                return np.zeros((length, self.N_CHANNELS), dtype=np.float32)
            want = min(length, n_have)
            end = i
            start = (i - want) % self.BUF_LEN
            if start < end:
                w = self.buf[:, start:end].T.copy()
            else:
                w = np.concatenate([self.buf[:, start:].T, self.buf[:, :end].T], axis=0).copy()
            if want < length:
                pad = np.zeros((length - want, self.N_CHANNELS), dtype=np.float32)
                w = np.concatenate([pad, w], axis=0)
        return w


# ---------------------------------------------------------------------------
# Higher-moment side-channel — directly hand SE skew, kurt, AC for free
# (matched-spectrum spoof preserves μ,σ,φ but NOT these. Per-channel analysis
#  showed it gives 100% spoof discrimination on its own.)
# ---------------------------------------------------------------------------
def higher_moments(w: np.ndarray) -> np.ndarray:
    """w: (T, C). Returns (C*5,) vector: per channel [skew, kurt, ac1, ac8, log_std]."""
    T, C = w.shape
    out = np.zeros((C, 5), dtype=np.float32)
    mu = w.mean(axis=0); sd = w.std(axis=0) + 1e-9
    z = (w - mu) / sd
    m3 = (z**3).mean(axis=0); m4 = (z**4).mean(axis=0)
    out[:, 0] = m3                           # skew
    out[:, 1] = m4 - 3.0                     # excess kurtosis
    # autocorr at 1, 8
    for k, lag in enumerate((1, 8), start=2):
        if T > lag:
            a = z[:-lag]; b = z[lag:]
            num = (a * b).mean(axis=0)
            out[:, k] = num
    out[:, 4] = np.log1p(sd)                 # log-std
    return out.flatten()


def normalize_window(w: np.ndarray) -> np.ndarray:
    mu = w.mean(axis=0, keepdims=True)
    sd = w.std(axis=0, keepdims=True) + 1e-6
    return ((w - mu) / sd).astype(np.float32)


try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False


if _HAS_TORCH:
    class SubstrateEncoderV3(nn.Module):
        """GRU + higher-moment side-channel.

        Forward: x (B, T, C=10), moments (B, C*5=50)
        Returns: z (B, d_out), next_pred (B, C)
        """
        def __init__(self, n_channels: int = 10, hidden: int = 128,
                     layers: int = 2, d_out: int = 128) -> None:
            super().__init__()
            self.proj_in  = nn.Linear(n_channels, hidden)
            self.gru      = nn.GRU(hidden, hidden, num_layers=layers, batch_first=True)
            self.mom_proj = nn.Linear(n_channels * 5, hidden)
            self.fuse     = nn.Linear(hidden * 2, d_out)
            self.pll_head = nn.Linear(hidden, n_channels)

        def forward(self, x, moments):
            h = F.relu(self.proj_in(x))
            h, _ = self.gru(h)
            h_last = h[:, -1]                        # (B, hidden)
            m = F.relu(self.mom_proj(moments))       # (B, hidden)
            z = self.fuse(torch.cat([h_last, m], dim=-1))
            next_pred = self.pll_head(h_last)
            return z, next_pred
else:
    class SubstrateEncoderV3:
        def __init__(self, *a, **kw): raise RuntimeError("torch missing")


__all__ = ["SubstrateStateV3", "SubstrateEncoderV3", "normalize_window", "higher_moments"]
