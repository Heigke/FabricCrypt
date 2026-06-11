#!/usr/bin/env python3
"""_thermal_clamp.py — thermal regime controller for PUF Phase 1b.

Modes:
    cold: try to set DPM=low (if writable). Idle-wait until temp <= COLD_TARGET.
    idle: DPM=auto (if writable). Wait until temp settles in IDLE_BAND for HOLD_S.
    warm: drive a controlled matmul heater workload until temp in WARM_BAND for HOLD_S.

Reads APU temp from /sys/class/thermal/thermal_zone0/temp (acpitz). On hardware
with amdgpu hwmon also reads /sys/class/hwmon/hwmon*/temp1_input (amdgpu edge)
and returns the *max* of the two so we don't undercount die temperature.

Never exceeds NEVER_EXCEED_C (80C). Aborts if it cannot reach a regime within
TIMEOUT_S — caller should record what was achieved (honest failure).

Usage as library:
    from _thermal_clamp import clamp_regime, read_temp
    achieved = clamp_regime("warm")     # blocks; returns mean temp during hold
"""
from __future__ import annotations
import os
import time
from pathlib import Path
from typing import Optional

THERMAL_ZONE = Path("/sys/class/thermal/thermal_zone0/temp")
DVFS_PATH = Path("/sys/class/drm/card0/device/power_dpm_force_performance_level")

# Try common amdgpu hwmon locations.
def _find_amdgpu_hwmon() -> Optional[Path]:
    for h in Path("/sys/class/hwmon").glob("hwmon*"):
        try:
            if (h / "name").read_text().strip() == "amdgpu":
                t = h / "temp1_input"
                if t.exists():
                    return t
        except Exception:
            pass
    return None

AMDGPU_TEMP = _find_amdgpu_hwmon()

# Targets (C)
COLD_TARGET = 38.0          # ambient permitting; >= this we keep waiting
COLD_TIMEOUT_S = 180.0
IDLE_BAND = (40.0, 50.0)
IDLE_HOLD_S = 30.0
IDLE_TIMEOUT_S = 180.0
WARM_BAND = (62.0, 70.0)
WARM_HOLD_S = 30.0
WARM_TIMEOUT_S = 240.0

NEVER_EXCEED_C = 78.0       # hard cap (NEVER_EXCEED — well below 99C trip)
SAMPLE_S = 1.0


def read_temp() -> float:
    """Return max of acpitz + amdgpu edge in Celsius."""
    vals = []
    try:
        vals.append(int(THERMAL_ZONE.read_text().strip()) / 1000.0)
    except Exception:
        pass
    if AMDGPU_TEMP is not None:
        try:
            vals.append(int(AMDGPU_TEMP.read_text().strip()) / 1000.0)
        except Exception:
            pass
    if not vals:
        return -1.0
    return max(vals)


def set_dvfs(level: str) -> bool:
    """Return True if DPM level was written, False if RO."""
    if not DVFS_PATH.exists():
        return False
    try:
        with open(DVFS_PATH, "w") as f:
            f.write(level)
        return True
    except (PermissionError, OSError):
        return False


def dvfs_writable() -> bool:
    if not DVFS_PATH.exists():
        return False
    return os.access(str(DVFS_PATH), os.W_OK)


def _idle_until(target_c: float, timeout_s: float) -> float:
    t0 = time.time()
    last = read_temp()
    while time.time() - t0 < timeout_s:
        t = read_temp()
        last = t
        if t > 0 and t <= target_c:
            return t
        time.sleep(SAMPLE_S)
    return last


def _wait_in_band(lo: float, hi: float, hold_s: float, timeout_s: float,
                  heater=None) -> tuple:
    """Wait until temp stays in [lo,hi] for hold_s. If heater callable provided,
    invoke it each second (to drive temp up). Returns (mean_temp_during_hold,
    achieved_in_band: bool, samples_list)."""
    t0 = time.time()
    in_band_start = None
    samples_hold = []
    while time.time() - t0 < timeout_s:
        if heater is not None:
            heater()
        t = read_temp()
        if t >= NEVER_EXCEED_C:
            # bail immediately — safety
            return (t, False, samples_hold)
        if lo <= t <= hi:
            if in_band_start is None:
                in_band_start = time.time()
                samples_hold = []
            samples_hold.append(t)
            if time.time() - in_band_start >= hold_s:
                return (float(sum(samples_hold) / len(samples_hold)), True,
                        samples_hold)
        else:
            in_band_start = None
            samples_hold = []
        if heater is None:
            time.sleep(SAMPLE_S)
    # timeout: report whatever current temp; if we have some samples use those
    if samples_hold:
        return (float(sum(samples_hold) / len(samples_hold)), False,
                samples_hold)
    return (read_temp(), False, [])


def _make_heater():
    """Return a closure that runs a small GPU matmul to drive temperature up.
    Falls back to CPU matmul if torch+rocm not available."""
    try:
        import torch
        if torch.cuda.is_available():
            dev = "cuda"
            A = torch.randn(2048, 2048, device=dev, dtype=torch.float32)
            B = torch.randn(2048, 2048, device=dev, dtype=torch.float32)

            def heat():
                for _ in range(4):
                    C = A @ B
                    A.copy_(C * 1e-4 + A * 0.9999)
                torch.cuda.synchronize()
                # short breather so we can sample temp + stay below cap
                time.sleep(0.1)
            return heat
    except Exception:
        pass
    # CPU fallback
    import numpy as np
    A = np.random.randn(1024, 1024).astype(np.float32)
    B = np.random.randn(1024, 1024).astype(np.float32)

    def heat_cpu():
        for _ in range(2):
            C = A @ B
            A[:] = C * 1e-4 + A * 0.9999
        time.sleep(0.1)
    return heat_cpu


def clamp_regime(regime: str) -> dict:
    """Block until regime achieved (or best-effort). Return dict with keys:
        regime, achieved_temp_C, in_band, dvfs_used, notes
    """
    notes = []
    dvfs_used = None
    if regime == "cold":
        if dvfs_writable():
            ok = set_dvfs("low")
            dvfs_used = "low" if ok else None
            notes.append(f"DVFS=low set={ok}")
        else:
            notes.append("DVFS not writable; passive idle-wait only")
        t = _idle_until(COLD_TARGET, COLD_TIMEOUT_S)
        in_band = (t <= COLD_TARGET + 2.0) and t > 0
        if not in_band:
            notes.append(f"could not reach COLD target {COLD_TARGET}C "
                         f"(achieved {t:.1f}C)")
        return {"regime": "cold", "achieved_temp_C": float(t),
                "in_band": bool(in_band), "dvfs_used": dvfs_used,
                "notes": "; ".join(notes)}

    if regime == "idle":
        if dvfs_writable():
            ok = set_dvfs("auto")
            dvfs_used = "auto" if ok else None
            notes.append(f"DVFS=auto set={ok}")
        else:
            notes.append("DVFS not writable; passive only")
        # If currently hot, first cool below idle band hi
        if read_temp() > IDLE_BAND[1] + 2:
            _idle_until(IDLE_BAND[1] - 1, IDLE_TIMEOUT_S)
        mean_t, in_band, samples = _wait_in_band(
            IDLE_BAND[0], IDLE_BAND[1], IDLE_HOLD_S, IDLE_TIMEOUT_S)
        if not in_band:
            notes.append(f"IDLE band {IDLE_BAND} not held; achieved ~{mean_t:.1f}C")
        return {"regime": "idle", "achieved_temp_C": float(mean_t),
                "in_band": bool(in_band), "dvfs_used": dvfs_used,
                "notes": "; ".join(notes)}

    if regime == "warm":
        if dvfs_writable():
            ok = set_dvfs("high")
            dvfs_used = "high" if ok else None
            notes.append(f"DVFS=high set={ok}")
        else:
            notes.append("DVFS not writable; using heater only")
        heater = _make_heater()
        mean_t, in_band, samples = _wait_in_band(
            WARM_BAND[0], WARM_BAND[1], WARM_HOLD_S, WARM_TIMEOUT_S,
            heater=heater)
        if not in_band:
            notes.append(f"WARM band {WARM_BAND} not held; achieved ~{mean_t:.1f}C")
        return {"regime": "warm", "achieved_temp_C": float(mean_t),
                "in_band": bool(in_band), "dvfs_used": dvfs_used,
                "notes": "; ".join(notes)}

    raise ValueError(f"unknown regime {regime}")


if __name__ == "__main__":
    import sys, json
    regime = sys.argv[1] if len(sys.argv) > 1 else "idle"
    print(f"[thermal_clamp] regime={regime} starting at {read_temp():.1f}C")
    print(f"[thermal_clamp] dvfs_writable={dvfs_writable()} amdgpu_temp={AMDGPU_TEMP}")
    res = clamp_regime(regime)
    print(json.dumps(res, indent=2))
