"""Shared thermal-guard for Phase 11B experiments. Strict: pause at 70C, resume <55C."""
from __future__ import annotations
import time

THERMAL_FILE = "/sys/class/thermal/thermal_zone0/temp"
PAUSE_C = 70.0
RESUME_C = 55.0
MAX_WAIT_S = 120


def read_apu_c() -> float:
    try:
        with open(THERMAL_FILE) as f:
            return int(f.read().strip()) / 1000.0
    except Exception:
        return 0.0


def wait_cool(tag: str = "", verbose: bool = True) -> float:
    t = read_apu_c()
    if t < PAUSE_C:
        return t
    t0 = time.time()
    if verbose:
        print(f"[thermal {tag}] APU={t:.1f}C >= {PAUSE_C}, cooling to {RESUME_C}", flush=True)
    while read_apu_c() >= RESUME_C:
        if time.time() - t0 > MAX_WAIT_S:
            if verbose:
                print(f"[thermal {tag}] WARN: still {read_apu_c():.1f}C after {MAX_WAIT_S}s, continuing", flush=True)
            break
        time.sleep(2)
    t = read_apu_c()
    if verbose:
        print(f"[thermal {tag}] resumed at {t:.1f}C ({time.time()-t0:.0f}s wait)", flush=True)
    return t
