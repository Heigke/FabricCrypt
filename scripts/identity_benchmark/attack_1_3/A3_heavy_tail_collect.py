#!/usr/bin/env python3
"""ATTACK 3 — Heavy-tail substrate stream collection.

Collects ~10-15 min of multi-channel substrate stream per host. Channels:
  ch_syscall_jitter : rdtsc → getpid → rdtsc, 50k samples (microseconds)
  ch_atomic_burst   : 32-thread CAS contention burst durations
  ch_tsc_drift      : TSC delta at 10ms wall intervals (drift / stability)
  ch_loop_jitter    : 1k-NOP loop wall time, repeated (fast-jitter heavy tail)

Designed to be THERMALLY SAFE: idle DPM, single threaded for the long channel,
abort at 70°C APU. Saves to .npz.

Run as: HSA_OVERRIDE_GFX_VERSION=11.0.0 python A3_heavy_tail_collect.py [--host LABEL] [--minutes M]
"""
from __future__ import annotations
import argparse
import ctypes
import os
import socket
import sys
import threading
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
OUT_DIR = HERE.parents[2] / "results" / "IDENTITY_BENCHMARK_2026-05-30" / "attack_1_3"
OUT_DIR.mkdir(parents=True, exist_ok=True)

THERMAL_ZONE = Path("/sys/class/thermal/thermal_zone0/temp")
THERMAL_ABORT_C = 70.0
THERMAL_PAUSE_C = 65.0


def read_apu_temp() -> float:
    try:
        return int(THERMAL_ZONE.read_text().strip()) / 1000.0
    except Exception:
        return -1.0


def wait_cool(target: float = 58.0, timeout: float = 60.0) -> float:
    t0 = time.time()
    while time.time() - t0 < timeout:
        t = read_apu_temp()
        if t > 0 and t <= target:
            return t
        time.sleep(2.0)
    return read_apu_temp()


# -----------------------------------------------------------------------------
# Channel 1: rdtsc → syscall → rdtsc jitter (very fast, heavy tail expected)
# -----------------------------------------------------------------------------
_LIBC = ctypes.CDLL("libc.so.6", use_errno=True)


def _rdtsc_via_time_ns() -> int:
    # Cross-platform fallback: monotonic_ns is high-resolution & low-overhead
    return time.monotonic_ns()


def collect_syscall_jitter(n_samples: int = 50000) -> np.ndarray:
    """Microseconds elapsed across a getpid() syscall."""
    out = np.empty(n_samples, dtype=np.float64)
    getpid = _LIBC.getpid
    mn = time.monotonic_ns
    for i in range(n_samples):
        a = mn()
        getpid()
        b = mn()
        out[i] = (b - a) / 1000.0  # μs
    return out


# -----------------------------------------------------------------------------
# Channel 2: atomic CAS contention burst durations
# -----------------------------------------------------------------------------
def collect_atomic_contention(n_bursts: int = 2000, n_threads: int = 8,
                              n_iters: int = 5000) -> np.ndarray:
    """Each burst: spawn threads that hammer a shared counter; measure total time.

    Reduced threads (8 vs 200) for thermal safety. Burst-level distribution
    captures contention-driven heavy tail without sustained heavy load.
    """
    out = np.empty(n_bursts, dtype=np.float64)
    state = {"counter": 0}
    lock = threading.Lock()

    def worker():
        for _ in range(n_iters):
            with lock:
                state["counter"] += 1

    for b in range(n_bursts):
        state["counter"] = 0
        threads = [threading.Thread(target=worker) for _ in range(n_threads)]
        t0 = time.monotonic_ns()
        for th in threads:
            th.start()
        for th in threads:
            th.join()
        t1 = time.monotonic_ns()
        out[b] = (t1 - t0) / 1e6  # ms
        if b % 50 == 0:
            T = read_apu_temp()
            if T > THERMAL_ABORT_C:
                print(f"[A3.atomic] ABORT at burst {b} APU={T:.1f}", flush=True)
                return out[:b]
            if T > THERMAL_PAUSE_C:
                wait_cool(target=55.0, timeout=30.0)
    return out


# -----------------------------------------------------------------------------
# Channel 3: TSC drift / wall delta jitter at 10ms intervals
# -----------------------------------------------------------------------------
def collect_tsc_drift(n_samples: int = 6000, interval_s: float = 0.010) -> np.ndarray:
    """Measure actual elapsed time per sleep(10ms). Tail = OS scheduling jitter."""
    out = np.empty(n_samples, dtype=np.float64)
    mn = time.monotonic_ns
    last = mn()
    for i in range(n_samples):
        time.sleep(interval_s)
        now = mn()
        out[i] = (now - last) / 1e6 - interval_s * 1000.0  # deviation in ms
        last = now
        if i % 200 == 0:
            T = read_apu_temp()
            if T > THERMAL_ABORT_C:
                print(f"[A3.tsc] ABORT at sample {i} APU={T:.1f}", flush=True)
                return out[:i]
    return out


# -----------------------------------------------------------------------------
# Channel 4: tight loop wall time jitter (no syscall)
# -----------------------------------------------------------------------------
def collect_loop_jitter(n_samples: int = 50000, loop_iters: int = 2000) -> np.ndarray:
    out = np.empty(n_samples, dtype=np.float64)
    mn = time.monotonic_ns
    # use a precompiled dummy work
    for i in range(n_samples):
        a = mn()
        x = 0
        for j in range(loop_iters):
            x += j ^ (j >> 1)
        b = mn()
        out[i] = (b - a) / 1000.0  # μs
        if i % 5000 == 0:
            T = read_apu_temp()
            if T > THERMAL_ABORT_C:
                print(f"[A3.loop] ABORT at sample {i} APU={T:.1f}", flush=True)
                return out[:i]
    return out


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default=None, help="label override (else hostname)")
    ap.add_argument("--minutes", type=float, default=12.0, help="total budget (informational)")
    ap.add_argument("--out", default=None, help="explicit output path")
    args = ap.parse_args()

    host = args.host or socket.gethostname().split(".")[0]
    out_path = Path(args.out) if args.out else (OUT_DIR / f"A3_streams_{host}.npz")
    print(f"[A3.collect] host={host} budget≈{args.minutes:.1f} min", flush=True)
    print(f"[A3.collect] apu_temp_start={read_apu_temp():.1f}", flush=True)

    if read_apu_temp() > 62.0:
        print("[A3.collect] cooling first…", flush=True)
        wait_cool(target=55.0, timeout=120.0)

    t_global = time.time()
    streams = {}
    meta = {"host": host, "apu_temp": {"start": read_apu_temp()}}

    # ---- ch1 syscall jitter (~1-2 min) ----
    print("[A3.collect] ch_syscall_jitter…", flush=True)
    t0 = time.time()
    streams["ch_syscall_jitter"] = collect_syscall_jitter(n_samples=80000)
    meta["ch_syscall_jitter"] = {"wall_s": time.time() - t0,
                                  "n": int(streams["ch_syscall_jitter"].size),
                                  "apu_after": read_apu_temp()}
    print(f"  done n={streams['ch_syscall_jitter'].size} wall={time.time() - t0:.1f}s "
          f"apu={read_apu_temp():.1f}", flush=True)

    if read_apu_temp() > THERMAL_PAUSE_C:
        wait_cool(target=55.0, timeout=60.0)

    # ---- ch4 loop jitter (~2-3 min) ----
    print("[A3.collect] ch_loop_jitter…", flush=True)
    t0 = time.time()
    streams["ch_loop_jitter"] = collect_loop_jitter(n_samples=40000, loop_iters=2000)
    meta["ch_loop_jitter"] = {"wall_s": time.time() - t0,
                               "n": int(streams["ch_loop_jitter"].size),
                               "apu_after": read_apu_temp()}
    print(f"  done n={streams['ch_loop_jitter'].size} wall={time.time() - t0:.1f}s "
          f"apu={read_apu_temp():.1f}", flush=True)

    if read_apu_temp() > THERMAL_PAUSE_C:
        wait_cool(target=55.0, timeout=60.0)

    # ---- ch2 atomic burst (~3-4 min) ----
    print("[A3.collect] ch_atomic_burst…", flush=True)
    t0 = time.time()
    streams["ch_atomic_burst"] = collect_atomic_contention(n_bursts=1500, n_threads=8, n_iters=4000)
    meta["ch_atomic_burst"] = {"wall_s": time.time() - t0,
                                "n": int(streams["ch_atomic_burst"].size),
                                "apu_after": read_apu_temp()}
    print(f"  done n={streams['ch_atomic_burst'].size} wall={time.time() - t0:.1f}s "
          f"apu={read_apu_temp():.1f}", flush=True)

    if read_apu_temp() > THERMAL_PAUSE_C:
        wait_cool(target=55.0, timeout=60.0)

    # ---- ch3 tsc drift (~1 min — 6000 samples × 10ms) ----
    print("[A3.collect] ch_tsc_drift…", flush=True)
    t0 = time.time()
    streams["ch_tsc_drift"] = collect_tsc_drift(n_samples=6000, interval_s=0.010)
    meta["ch_tsc_drift"] = {"wall_s": time.time() - t0,
                             "n": int(streams["ch_tsc_drift"].size),
                             "apu_after": read_apu_temp()}
    print(f"  done n={streams['ch_tsc_drift'].size} wall={time.time() - t0:.1f}s "
          f"apu={read_apu_temp():.1f}", flush=True)

    meta["apu_temp"]["end"] = read_apu_temp()
    meta["total_wall_s"] = time.time() - t_global

    # save
    np.savez_compressed(out_path, **streams)
    meta_path = out_path.with_suffix(".meta.json")
    import json
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[A3.collect] saved → {out_path} ({out_path.stat().st_size / 1024:.1f} kB)", flush=True)
    print(f"[A3.collect] meta → {meta_path}", flush=True)
    print(f"[A3.collect] total wall={meta['total_wall_s']:.1f}s apu_end={meta['apu_temp']['end']:.1f}",
          flush=True)


if __name__ == "__main__":
    main()
