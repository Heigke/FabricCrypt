"""Real-time substrate sampler + ring buffer + Substrate Encoder (SE).

Implements the live-feed side of GPT-5's closed-loop architecture:
  - A daemon thread samples C07 (XTAL_CNTL), C11 (TSC drift), C20 (SMN read latency)
    at the highest rate each channel can deliver.
  - A ring buffer holds the last N=4096 samples per channel.
  - SubstrateEncoder consumes the latest window (W=256 timesteps × 6 channels)
    and emits a 64-D embedding z_t that the LM uses to FiLM-modulate every layer.
  - A closed-loop hook lets the LM trigger a HIP-side microkernel between tokens
    to actively change + read substrate state — that's the rooting mechanism,
    not just conditioning.

This module is the *only* place we touch /dev/mem MMCFG at training/inference
time. The rest of the pipeline imports SubstrateState.
"""
from __future__ import annotations

import ctypes
import mmap
import os
import struct
import subprocess
import threading
import time
from collections import deque
from pathlib import Path
from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# Low-level SMN probe (subset of h7_deep_substrate_probe.py — only the
# channels with confirmed identity-bearing potential)
# ---------------------------------------------------------------------------
MMCFG_BASE = 0xE0000000
SMN_ADDR_OFF = 0x60
SMN_DATA_OFF = 0x64
SMN_XTAL_CNTL = 0x598C8


class _MMCFG:
    def __init__(self) -> None:
        self.fd = os.open("/dev/mem", os.O_RDWR | os.O_SYNC)
        self.mm = mmap.mmap(self.fd, 4096, mmap.MAP_SHARED,
                            mmap.PROT_READ | mmap.PROT_WRITE,
                            offset=MMCFG_BASE)

    def smn_read_timed(self, addr: int):
        t0 = time.perf_counter_ns()
        self.mm.seek(SMN_ADDR_OFF)
        self.mm.write(struct.pack("<I", addr))
        self.mm.seek(SMN_DATA_OFF)
        v = struct.unpack("<I", self.mm.read(4))[0]
        t1 = time.perf_counter_ns()
        return v, (t1 - t0)


# ---------------------------------------------------------------------------
# TSC vs CLOCK_MONOTONIC_RAW drift sampler
# ---------------------------------------------------------------------------
_libc = ctypes.CDLL("libc.so.6", use_errno=True)
class _Timespec(ctypes.Structure):
    _fields_ = [("tv_sec", ctypes.c_long), ("tv_nsec", ctypes.c_long)]
CLOCK_MONOTONIC_RAW = 4


def _clock_raw_ns() -> int:
    ts = _Timespec()
    _libc.clock_gettime(CLOCK_MONOTONIC_RAW, ctypes.byref(ts))
    return ts.tv_sec * 1_000_000_000 + ts.tv_nsec


def _drift_sample():
    a1 = time.monotonic_ns()
    b1 = _clock_raw_ns()
    a2 = time.monotonic_ns()
    b2 = _clock_raw_ns()
    return (a2 - a1) - (b2 - b1)   # ns drift across the read pair


# ---------------------------------------------------------------------------
# Live ring buffer
# ---------------------------------------------------------------------------
class SubstrateState:
    """Holds the live ring buffer + provides window snapshots.

    Channels (rows of the buffer):
      0  C07 XTAL_CNTL value
      1  C07 XTAL_CNTL Δ (delta from previous read)
      2  C20 lat_xtal ns (SMN read latency for the XTAL register itself)
      3  C20 lat_xtal log-tail (signed log of (lat - median))
      4  C11 TSC-drift ns
      5  C11 TSC-drift abs
    """
    N_CHANNELS = 6
    BUF_LEN = 4096   # ring length per channel

    def __init__(self, hz_target: int = 500) -> None:
        self.dt = 1.0 / hz_target
        self.buf = np.zeros((self.N_CHANNELS, self.BUF_LEN), dtype=np.float32)
        self.idx = 0
        self.total_samples = 0
        self.lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._probe: Optional[_MMCFG] = None
        self._lat_window: deque = deque(maxlen=256)

    def start(self) -> None:
        if self._thread is not None:
            return
        if os.geteuid() != 0:
            raise PermissionError("SubstrateState needs root for /dev/mem MMCFG")
        self._probe = _MMCFG()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        # let buffer warm up
        time.sleep(0.5)

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    def _run(self) -> None:
        prev_xtal = 0
        while not self._stop.is_set():
            t0 = time.perf_counter()
            try:
                xtal, lat = self._probe.smn_read_timed(SMN_XTAL_CNTL)
                drift = _drift_sample()
                self._lat_window.append(lat)
                med_lat = np.median(self._lat_window) if self._lat_window else lat
                log_tail = float(np.sign(lat - med_lat) * np.log1p(abs(lat - med_lat)))
                with self.lock:
                    i = self.idx
                    self.buf[0, i] = xtal
                    self.buf[1, i] = (xtal - prev_xtal) if self.total_samples else 0.0
                    self.buf[2, i] = lat
                    self.buf[3, i] = log_tail
                    self.buf[4, i] = drift
                    self.buf[5, i] = abs(drift)
                    self.idx = (i + 1) % self.BUF_LEN
                    self.total_samples += 1
                prev_xtal = xtal
            except Exception:
                pass
            slack = self.dt - (time.perf_counter() - t0)
            if slack > 0:
                time.sleep(slack)

    def latest_window(self, length: int = 256) -> np.ndarray:
        """Return last `length` samples per channel as (length, N_CHANNELS) array.

        Older→newer order. If buffer has fewer than `length` samples yet,
        zero-pads at the front.
        """
        with self.lock:
            i = self.idx
            n_have = min(self.total_samples, self.BUF_LEN)
            if n_have == 0:
                return np.zeros((length, self.N_CHANNELS), dtype=np.float32)
            if n_have < length:
                want = n_have
            else:
                want = length
            # contiguous slice ending just before idx
            end = i
            start = (i - want) % self.BUF_LEN
            if start < end:
                w = self.buf[:, start:end].T.copy()
            else:
                w = np.concatenate(
                    [self.buf[:, start:].T, self.buf[:, :end].T], axis=0
                ).copy()
            if want < length:
                pad = np.zeros((length - want, self.N_CHANNELS), dtype=np.float32)
                w = np.concatenate([pad, w], axis=0)
        return w


# ---------------------------------------------------------------------------
# Substrate Encoder (GRU) — emits per-token 64D embedding
# Torch is optional so this module can be imported on hosts without it
# (e.g. for substrate replay recording on a host that lacks pytorch).
# ---------------------------------------------------------------------------
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False


if not _HAS_TORCH:
    class _NoTorch:
        def __init_subclass__(cls, *a, **kw): pass
    nn = type("nnstub", (), {"Module": _NoTorch, "Linear": _NoTorch, "GRU": _NoTorch})()
    class SubstrateEncoder:
        def __init__(self, *a, **kw):
            raise RuntimeError("torch not installed on this host — recording-only mode")
else:
 pass

if _HAS_TORCH:
  class SubstrateEncoder(nn.Module):
    """Causal GRU that maps a (B, T_sub, C) substrate window → (B, D_emb).

    D_emb=64 by default. T_sub=256, C=6. ~30k params.
    """
    def __init__(self, n_channels: int = 6, hidden: int = 64, layers: int = 2,
                 d_out: int = 64) -> None:
        super().__init__()
        self.proj_in = nn.Linear(n_channels, hidden)
        self.gru = nn.GRU(hidden, hidden, num_layers=layers, batch_first=True)
        self.out = nn.Linear(hidden, d_out)
        # PLL head — predict next sample's first channel; aux objective
        self.pll_head = nn.Linear(hidden, n_channels)

    def forward(self, x):
        # x: (B, T_sub, C) — already normalized to roughly unit scale
        h = F.relu(self.proj_in(x))
        h, _ = self.gru(h)
        z = self.out(h[:, -1])              # last hidden → embedding
        next_pred = self.pll_head(h[:, -1])  # predict next frame for aux loss
        return z, next_pred


# ---------------------------------------------------------------------------
# Normalization helper — applied to the raw window before SE
# ---------------------------------------------------------------------------
def normalize_window(w: np.ndarray) -> np.ndarray:
    """Per-channel standardize within window. Robust to constant channels."""
    mu = w.mean(axis=0, keepdims=True)
    sd = w.std(axis=0, keepdims=True) + 1e-6
    return ((w - mu) / sd).astype(np.float32)


__all__ = ["SubstrateState", "SubstrateEncoder", "normalize_window"]
