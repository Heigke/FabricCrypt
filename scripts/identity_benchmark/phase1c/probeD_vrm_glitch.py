#!/usr/bin/env python3
"""probeD_vrm_glitch.py — VRM transient settling fingerprint.

Protocol:
  1. Idle 5 s (let VRM settle).
  2. Trigger a sudden compute burst (matmul on GPU) for 2 s.
  3. During the entire 7 s window, sample gpu_metrics binary blob at 50 Hz
     and read clock/voltage/temp fields.
  4. Repeat N times. Compute per-transient features:
     - overshoot   = max(SCLK or V) - steady-state
     - settle_time = time to reach ±2% of steady state
     - ring_freq   = dominant freq in detrended early window
     - damping     = log-decrement
  5. Feature distribution per device is the fingerprint; compare via
     Mahalanobis distance.

We use torch with HSA_OVERRIDE_GFX_VERSION=11.0.0 for the GPU burst.
"""
from __future__ import annotations
import argparse
import json
import os
import socket
import struct
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from _thermal import (read_temp, wait_cool, inter_burst_sleep, ThermalMonitor,
                      MAX_BURST_S, COOL_TARGET_C, HOSTNAME, ABORT_AT_C)

GPU_METRICS = Path("/sys/class/drm/card0/device/gpu_metrics")

# Default-disabled hosts (heavy GPU burst can spike ikaros APU)
DISABLE_HOSTS_DEFAULT: set = {"ikaros"}  # per restart directive: skip on ikaros
# Hard cap burst length regardless of CLI
HARD_MAX_BURST_S = MAX_BURST_S  # 6 s ceiling


def parse_gpu_metrics(blob: bytes) -> dict:
    """Parse gpu_metrics_v3_0 (RDNA3.5 APU) — best-effort.
    Header: format_revision (u8) at 4, content_revision (u8) at 5.
    We read at common offsets known for v3_0 APU; on this machine the
    interesting numeric fields are sclk @ 0x40-ish range. To stay robust
    we just record raw bytes + a few decoded uint16 candidates."""
    out = {"raw_len": len(blob)}
    if len(blob) < 8:
        return out
    out["fmt_rev"] = blob[2]
    out["ctx_rev"] = blob[3]
    # Heuristic: scan u16 fields 0x40..0xC0 for plausible clock (200-3000 MHz)
    arr = np.frombuffer(blob[:min(len(blob), 256)], dtype=np.uint16)
    out["u16_candidates"] = arr.tolist()
    return out


def read_metrics_once() -> dict:
    try:
        b = GPU_METRICS.read_bytes()
    except Exception as e:
        return {"err": str(e)}
    rec = parse_gpu_metrics(b)
    rec["t"] = time.monotonic()
    try:
        rec["apu_c"] = int(THERMAL_ZONE.read_text().strip()) / 1000.0
    except Exception:
        rec["apu_c"] = -1.0
    return rec


def sample_thread(stop_evt: threading.Event, samples: list, period_s: float):
    next_t = time.monotonic()
    while not stop_evt.is_set():
        rec = read_metrics_once()
        samples.append(rec)
        next_t += period_s
        slp = next_t - time.monotonic()
        if slp > 0:
            time.sleep(slp)
        else:
            next_t = time.monotonic()


def gpu_burst(duration_s: float):
    """Fire a controlled GPU compute burst via torch. Hard-capped at HARD_MAX_BURST_S."""
    duration_s = min(duration_s, HARD_MAX_BURST_S)
    import torch
    dev = torch.device("cuda:0")
    a = torch.randn(2048, 2048, device=dev, dtype=torch.float32)
    b = torch.randn(2048, 2048, device=dev, dtype=torch.float32)
    t0 = time.monotonic()
    while time.monotonic() - t0 < duration_s:
        c = a @ b
        a = c * 0.999 + a * 0.001
        # bail early if temp spikes
        try:
            t = int(Path("/sys/class/thermal/thermal_zone0/temp").read_text()) / 1000.0
            if t >= ABORT_AT_C - 3:
                break
        except Exception:
            pass
    torch.cuda.synchronize()


def one_transient(idle_s: float, burst_s: float, tail_s: float, hz: float) -> list:
    samples: list = []
    stop = threading.Event()
    th = threading.Thread(target=sample_thread, args=(stop, samples, 1.0 / hz), daemon=True)
    th.start()
    time.sleep(idle_s)
    burst_start = time.monotonic()
    gpu_burst(burst_s)
    burst_end = time.monotonic()
    time.sleep(tail_s)
    stop.set()
    th.join(timeout=2.0)
    return [{"t0": burst_start, "t1": burst_end, "samples": samples}]


def feature_extract(transient: dict) -> dict:
    """Extract scalar features from one transient. Returns dict; np.nan if not derivable."""
    samples = transient["samples"]
    t0, t1 = transient["t0"], transient["t1"]
    if not samples:
        return {}
    # Build a time-series matrix for the u16 candidates closest to a plausible clock
    times = np.array([s.get("t", np.nan) for s in samples], dtype=np.float64)
    cand = np.array([s.get("u16_candidates", []) for s in samples], dtype=np.int32)
    if cand.size == 0:
        return {}
    n_fields = cand.shape[1]
    # For each field, score plausibility: variance > 0 AND mean in [200, 3500] MHz-ish
    means = cand.mean(axis=0)
    vars_ = cand.var(axis=0)
    plausible = (means > 200) & (means < 4000) & (vars_ > 0)
    if not plausible.any():
        return {"apu_overshoot": float(np.nan)}
    # Use field with the highest variance among plausible ones as our "clock proxy".
    scores = np.where(plausible, vars_, -1)
    best_field = int(np.argmax(scores))
    sig = cand[:, best_field].astype(np.float64)
    # APU temp transient
    apu = np.array([s.get("apu_c", np.nan) for s in samples], dtype=np.float64)
    # baseline = mean during idle [0, t0)
    idle_mask = times < t0
    burst_mask = (times >= t0) & (times < t1)
    tail_mask = times >= t1
    base = np.nanmean(sig[idle_mask]) if idle_mask.any() else np.nan
    burst_peak = np.nanmax(sig[burst_mask]) if burst_mask.any() else np.nan
    tail_mean = np.nanmean(sig[tail_mask]) if tail_mask.any() else np.nan
    apu_base = np.nanmean(apu[idle_mask]) if idle_mask.any() else np.nan
    apu_peak = np.nanmax(apu[burst_mask | tail_mask]) if (burst_mask | tail_mask).any() else np.nan

    # settle time: first index in tail where sig within 2% of tail_mean
    settle_time = np.nan
    if tail_mask.any():
        tail_t = times[tail_mask]
        tail_s = sig[tail_mask]
        thr = 0.02 * abs(tail_mean) + 1.0
        ok = np.abs(tail_s - tail_mean) < thr
        if ok.any():
            first = int(np.argmax(ok))
            settle_time = float(tail_t[first] - t1)

    # ring freq via FFT over the burst window
    ring_freq = np.nan
    burst_t = times[burst_mask]
    burst_s = sig[burst_mask]
    if burst_s.size >= 16:
        detr = burst_s - np.nanmean(burst_s)
        # rough dt
        dt = float(np.nanmedian(np.diff(burst_t))) if burst_t.size > 1 else 0.02
        if dt > 0 and np.isfinite(dt):
            spec = np.abs(np.fft.rfft(detr - np.nanmean(detr)))
            if spec.size > 1:
                f_axis = np.fft.rfftfreq(detr.size, d=dt)
                # ignore DC
                k = int(np.argmax(spec[1:]) + 1)
                ring_freq = float(f_axis[k])

    return {
        "clock_field": best_field,
        "base":         float(base),
        "burst_peak":   float(burst_peak),
        "tail_mean":    float(tail_mean),
        "overshoot":    float(burst_peak - base) if np.isfinite(burst_peak - base) else np.nan,
        "settle_time":  float(settle_time),
        "ring_freq":    float(ring_freq),
        "apu_base":     float(apu_base),
        "apu_peak":     float(apu_peak),
        "apu_rise":     float(apu_peak - apu_base) if np.isfinite(apu_peak - apu_base) else np.nan,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device-name", required=True)
    ap.add_argument("--reps", type=int, default=30)
    ap.add_argument("--idle-s", type=float, default=2.0)
    ap.add_argument("--burst-s", type=float, default=2.0)
    ap.add_argument("--tail-s", type=float, default=3.0)
    ap.add_argument("--hz", type=float, default=50.0)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    if HOSTNAME in DISABLE_HOSTS_DEFAULT and not args.force:
        msg = {"device": args.device_name, "ts": time.time(),
               "skipped": True, "reason": f"host {HOSTNAME} disabled (thermal safety)"}
        (args.out_dir / "probeD_results.json").write_text(json.dumps(msg, indent=2))
        print(f"[probeD] SKIPPED on {HOSTNAME} (use --force to override)")
        return

    if not GPU_METRICS.exists():
        print(f"[probeD] ERROR: {GPU_METRICS} missing")
        sys.exit(2)

    feats = []
    raw = []
    tm = ThermalMonitor("probeD")
    with tm:
        for r in range(args.reps):
            apu_c = read_temp()
            tm.sample()
            if apu_c > 70.0:
                print(f"[probeD] rep {r}: APU {apu_c}C -> wait_cool")
                wait_cool()
            t = one_transient(args.idle_s, min(args.burst_s, HARD_MAX_BURST_S),
                              args.tail_s, args.hz)
            feat = feature_extract(t[0])
            feats.append(feat)
            inter_burst_sleep()
        # do not save raw samples per rep (huge); keep a single example
            if r == 0:
                raw.append(t[0])
            print(f"[probeD] rep {r}/{args.reps} T={tm.max_c:.1f}C "
                  f"overshoot={feat.get('overshoot')} "
                  f"ring_freq={feat.get('ring_freq')} settle={feat.get('settle_time')}")

    out = {
        "device": args.device_name,
        "ts": time.time(),
        "n_reps": args.reps,
        "params": {"idle_s": args.idle_s, "burst_s": args.burst_s,
                   "tail_s": args.tail_s, "hz": args.hz},
        "thermal": tm.as_dict(),
        "features": feats,
        "example_transient_samples_n": len(raw[0]["samples"]) if raw else 0,
    }
    (args.out_dir / "probeD_results.json").write_text(json.dumps(out, indent=2))
    print(f"[probeD] wrote {args.out_dir/'probeD_results.json'}")


if __name__ == "__main__":
    main()
