"""Shared substrate telemetry for embodiment5 body-centric experiments.

All measurements are direct sysfs reads (no root needed) on gfx1151.
Channels:
  apu_temp_c  : /sys/class/thermal/thermal_zone0/temp (mC) -> C
  gpu_temp_c  : hwmon7/temp1_input (mC)
  gpu_power_w : hwmon7/power1_average (uW)
  gpu_freq_mhz: hwmon7/freq1_input (Hz)
  gpu_volt_v  : hwmon7/in0_input (mV)
  kern_lat_us : timeit a tiny numpy matmul, returns us

Sampling: sample_substrate() returns a dict of all channels measured in
~1ms (apart from kernel_latency which itself is ~few hundred us).
"""
from __future__ import annotations
import time
import numpy as np
from pathlib import Path

HWMON_GPU = Path("/sys/class/hwmon/hwmon7")
THERMAL_APU = Path("/sys/class/thermal/thermal_zone0/temp")


def _read(p: Path, default: float = 0.0) -> float:
    try:
        return float(p.read_text().strip())
    except Exception:
        return default


def apu_temp_c() -> float:
    return _read(THERMAL_APU) / 1000.0


def gpu_temp_c() -> float:
    return _read(HWMON_GPU / "temp1_input") / 1000.0


def gpu_power_w() -> float:
    return _read(HWMON_GPU / "power1_average") / 1e6


def gpu_freq_mhz() -> float:
    return _read(HWMON_GPU / "freq1_input") / 1e6


def gpu_volt_v() -> float:
    return _read(HWMON_GPU / "in0_input") / 1000.0


# Cached tiny workload for kernel-latency channel
_MAT_A = np.random.default_rng(0).standard_normal((64, 64)).astype(np.float32)
_MAT_B = np.random.default_rng(1).standard_normal((64, 64)).astype(np.float32)


def kern_lat_us(n: int = 5) -> float:
    """Run a tiny numpy matmul n times, return median wallclock in microseconds."""
    ts = []
    for _ in range(n):
        t0 = time.perf_counter()
        _ = _MAT_A @ _MAT_B
        ts.append((time.perf_counter() - t0) * 1e6)
    return float(np.median(ts))


CHANNELS = ["apu_temp_c", "gpu_temp_c", "gpu_power_w",
            "gpu_freq_mhz", "gpu_volt_v", "kern_lat_us"]


def sample_substrate() -> dict:
    return {
        "t": time.time(),
        "apu_temp_c": apu_temp_c(),
        "gpu_temp_c": gpu_temp_c(),
        "gpu_power_w": gpu_power_w(),
        "gpu_freq_mhz": gpu_freq_mhz(),
        "gpu_volt_v": gpu_volt_v(),
        "kern_lat_us": kern_lat_us(),
    }


def collect_window(n: int, dt_s: float = 0.05) -> np.ndarray:
    """Collect n samples spaced dt_s apart. Returns (n, len(CHANNELS)) array."""
    out = np.zeros((n, len(CHANNELS)), dtype=np.float32)
    for i in range(n):
        t0 = time.perf_counter()
        s = sample_substrate()
        for j, c in enumerate(CHANNELS):
            out[i, j] = s[c]
        wait = dt_s - (time.perf_counter() - t0)
        if wait > 0:
            time.sleep(wait)
    return out


def abort_if_hot(thresh_c: float = 75.0) -> bool:
    return apu_temp_c() >= thresh_c


def wait_cool(target_c: float = 60.0, timeout_s: float = 120.0) -> bool:
    t0 = time.time()
    while apu_temp_c() > target_c:
        if time.time() - t0 > timeout_s:
            return False
        time.sleep(2.0)
    return True
