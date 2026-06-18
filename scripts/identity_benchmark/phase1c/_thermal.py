"""Shared thermal-safety helpers for Phase 1c hardened probes."""
from __future__ import annotations
import socket
import time
from pathlib import Path

THERMAL_ZONE = Path("/sys/class/thermal/thermal_zone0/temp")
HOSTNAME = socket.gethostname()

# Hardening rules (per restart directive)
COOL_TARGET_C = 65.0        # wait until <= this between bursts
PAUSE_AT_C    = 72.0        # if hit during burst, pause
ABORT_AT_C    = 78.0        # caller must stop and report
MAX_BURST_S   = 6.0         # cap on any single kernel/probe burst
INTER_BURST_S = 10.0        # mandatory sleep between bursts


def read_temp() -> float:
    try:
        return int(THERMAL_ZONE.read_text().strip()) / 1000.0
    except Exception:
        return -1.0


def wait_cool(target: float = COOL_TARGET_C, timeout: float = 180.0, poll: float = 5.0) -> float:
    """Block until APU temp <= target. Returns final temp."""
    t0 = time.time()
    while time.time() - t0 < timeout:
        t = read_temp()
        if 0 < t <= target:
            return t
        time.sleep(poll)
    return read_temp()


def inter_burst_sleep(extra: float = 0.0):
    time.sleep(INTER_BURST_S + extra)


class ThermalMonitor:
    """Context manager: record start/end/max temp; raise if ABORT crossed."""
    def __init__(self, label: str = ""):
        self.label = label
        self.start_c = -1.0
        self.end_c = -1.0
        self.max_c = -1.0

    def __enter__(self):
        self.start_c = read_temp()
        self.max_c = self.start_c
        self.t0 = time.time()
        return self

    def sample(self):
        t = read_temp()
        if t > self.max_c:
            self.max_c = t
        if t >= ABORT_AT_C:
            raise RuntimeError(f"[{self.label}] thermal abort: {t}C >= {ABORT_AT_C}C")
        return t

    def __exit__(self, *a):
        self.end_c = read_temp()
        self.dur_s = time.time() - self.t0
        if self.end_c > self.max_c:
            self.max_c = self.end_c

    def as_dict(self) -> dict:
        return {"label": self.label, "start_c": self.start_c,
                "end_c": self.end_c, "max_c": self.max_c,
                "duration_s": getattr(self, "dur_s", -1.0)}
