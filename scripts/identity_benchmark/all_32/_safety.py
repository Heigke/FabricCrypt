"""Safety + telemetry module for ALL_32 identity campaign.

HARD RULES:
- APU temp ceiling: 72C (any probe must abort if exceeded).
- Pre-flight: refuse to start a probe unless APU <= 50C.
- Max kernel burst: 4.0s wall before mandatory yield.
- Mandatory cooling: 30s between probes; clamp_below(target) helper for tighter loops.
- Two-strike rule: if APU exceeds 72C twice during a campaign run, abort the whole run.
"""
from __future__ import annotations
import json, os, sys, time, signal, subprocess
from pathlib import Path
from typing import Optional, Callable

THERMAL_ZONE = Path("/sys/class/thermal/thermal_zone0/temp")

CEILING_C        = 72.0    # hard ceiling per-probe
PREFLIGHT_MAX_C  = 50.0    # must be below to start a probe
COOLDOWN_TARGET  = 48.0    # target between probes
COOLDOWN_TIMEOUT = 180.0   # max wait for cooldown
KERNEL_BURST_MAX_S = 4.0   # any single kernel/loop slice
INTER_PROBE_S    = 30.0    # mandatory wait

# campaign-level breaker
TWO_STRIKE_FILE = Path(__file__).parent / "state" / "ceiling_strikes.json"


def find_amdgpu_temp() -> Optional[Path]:
    for h in Path("/sys/class/hwmon").glob("hwmon*"):
        try:
            if (h / "name").read_text().strip() == "amdgpu":
                t = h / "temp1_input"
                if t.exists():
                    return t
        except Exception:
            pass
    return None

AMDGPU_TEMP_PATH = find_amdgpu_temp()


def read_apu_c() -> float:
    try:
        return int(THERMAL_ZONE.read_text().strip()) / 1000.0
    except Exception:
        return -1.0

def read_gpu_c() -> float:
    if AMDGPU_TEMP_PATH is None: return -1.0
    try:
        return int(AMDGPU_TEMP_PATH.read_text().strip()) / 1000.0
    except Exception:
        return -1.0

def read_max_c() -> float:
    a, g = read_apu_c(), read_gpu_c()
    return max(a, g)


def preflight(probe_name: str) -> bool:
    """Return True if safe to start. False = caller must skip."""
    t = read_max_c()
    if t < 0 or t > PREFLIGHT_MAX_C:
        print(f"[safety] PREFLIGHT REJECT {probe_name}: temp={t:.1f}C (need <={PREFLIGHT_MAX_C})",
              flush=True)
        return False
    return True


def cool_down(target_c: float = COOLDOWN_TARGET, timeout_s: float = COOLDOWN_TIMEOUT) -> float:
    """Block until max(APU,GPU) <= target_c. Returns final temp."""
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        t = read_max_c()
        if t < 0:
            return t
        if t <= target_c:
            return t
        time.sleep(2.0)
    return read_max_c()


def record_strike() -> int:
    TWO_STRIKE_FILE.parent.mkdir(parents=True, exist_ok=True)
    try:
        d = json.loads(TWO_STRIKE_FILE.read_text())
    except Exception:
        d = {"strikes": 0, "events": []}
    d["strikes"] = int(d.get("strikes", 0)) + 1
    d["events"].append({"t": time.time(), "temp": read_max_c()})
    TWO_STRIKE_FILE.write_text(json.dumps(d, indent=2))
    return d["strikes"]


def get_strikes() -> int:
    try:
        return int(json.loads(TWO_STRIKE_FILE.read_text()).get("strikes", 0))
    except Exception:
        return 0


def reset_strikes() -> None:
    TWO_STRIKE_FILE.parent.mkdir(parents=True, exist_ok=True)
    TWO_STRIKE_FILE.write_text(json.dumps({"strikes": 0, "events": []}))


class CeilingExceeded(RuntimeError):
    pass


def assert_under_ceiling() -> float:
    t = read_max_c()
    if t > CEILING_C:
        s = record_strike()
        raise CeilingExceeded(f"temp={t:.1f}C exceeds ceiling {CEILING_C}C (strike #{s})")
    return t


def run_with_watchdog(target: Callable[[], None], probe_name: str, max_s: float = 180.0) -> dict:
    """Spawn a subprocess that runs `target`. Parent polls temp every 0.5s and SIGKILLs child
    if APU > CEILING_C. Returns dict with status."""
    # We run via fork for simplicity.
    pid = os.fork()
    if pid == 0:
        # child
        try:
            target()
            os._exit(0)
        except Exception as e:
            sys.stderr.write(f"[child] {probe_name} exc: {e}\n")
            os._exit(1)
    # parent
    t0 = time.time()
    killed = False
    peak = read_max_c()
    while True:
        try:
            w_pid, w_status = os.waitpid(pid, os.WNOHANG)
        except ChildProcessError:
            w_pid, w_status = pid, 0
            break
        if w_pid == pid:
            break
        cur = read_max_c()
        if cur > peak: peak = cur
        if cur > CEILING_C:
            print(f"[watchdog] KILL {probe_name} temp={cur:.1f}C", flush=True)
            try: os.kill(pid, signal.SIGKILL)
            except Exception: pass
            killed = True
            record_strike()
            try:
                # restore DPM=auto for safety
                with open("/sys/class/drm/card0/device/power_dpm_force_performance_level", "w") as f:
                    f.write("auto")
            except Exception: pass
            try: os.waitpid(pid, 0)
            except Exception: pass
            break
        if time.time() - t0 > max_s:
            print(f"[watchdog] TIMEOUT {probe_name}", flush=True)
            try: os.kill(pid, signal.SIGKILL)
            except Exception: pass
            try: os.waitpid(pid, 0)
            except Exception: pass
            killed = True
            break
        time.sleep(0.5)
    return {
        "probe": probe_name,
        "killed": killed,
        "wall_s": time.time() - t0,
        "peak_temp_C": peak,
        "final_temp_C": read_max_c(),
    }


if __name__ == "__main__":
    print(json.dumps({
        "apu_c": read_apu_c(),
        "gpu_c": read_gpu_c(),
        "strikes": get_strikes(),
        "amdgpu_temp_path": str(AMDGPU_TEMP_PATH) if AMDGPU_TEMP_PATH else None,
    }, indent=2))
