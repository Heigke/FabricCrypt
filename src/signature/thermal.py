"""Thermal guard for safe capture on small-form-factor APUs.

The default thresholds assume a chassis with limited cooling (e.g.
HP Z2 mini G1a, or any laptop-class form factor). Tune via env vars
or function args if your chassis runs cooler.

  FABRICCRYPT_THERMAL_ABORT_C  (default 68)
  FABRICCRYPT_THERMAL_PAUSE_C  (default 63)
  FABRICCRYPT_THERMAL_COOL_C   (default 50)
"""
import os
import time
import socket
import subprocess
import json

THERMAL_ZONE = "/sys/class/thermal/thermal_zone0/temp"

ABORT_C = float(os.environ.get("FABRICCRYPT_THERMAL_ABORT_C", "68"))
PAUSE_C = float(os.environ.get("FABRICCRYPT_THERMAL_PAUSE_C", "63"))
COOL_C  = float(os.environ.get("FABRICCRYPT_THERMAL_COOL_C", "50"))


def get_apu_temp_c() -> float:
    try:
        return int(open(THERMAL_ZONE).read().strip()) / 1000.0
    except Exception:
        return 0.0


def thermal_guard(abort_c: float = None, pause_c: float = None,
                  cool_c: float = None, verbose: bool = False):
    abort_c = ABORT_C if abort_c is None else abort_c
    pause_c = PAUSE_C if pause_c is None else pause_c
    cool_c  = COOL_C  if cool_c  is None else cool_c
    t = get_apu_temp_c()
    if t >= abort_c:
        raise SystemExit(f"ABORT thermal {t:.1f}C >= {abort_c}C")
    if t >= pause_c:
        print(f"[thermal_guard] PAUSE at {t:.1f}C, cooling to <{cool_c}C", flush=True)
        while t > cool_c:
            time.sleep(5)
            t = get_apu_temp_c()
            print(f"[cool] {t:.1f}C", flush=True)
    elif verbose:
        print(f"[thermal_guard] OK {t:.1f}C", flush=True)


def wait_cool(target_c: float = 50, timeout_s: float = 180) -> float:
    t0 = time.time()
    while True:
        t = get_apu_temp_c()
        if t < target_c:
            return t
        if time.time() - t0 > timeout_s:
            return t
        time.sleep(5)


def hostname() -> str:
    try:
        return open("/etc/hostname").read().strip()
    except Exception:
        return socket.gethostname()


def save_json(path, data):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)


def compile_c(src_path, out_path, extra_flags=None):
    flags = ["-O2", "-march=native", "-pthread"]
    if extra_flags:
        flags += extra_flags
    cmd = ["gcc"] + flags + [src_path, "-o", out_path]
    subprocess.check_call(cmd)
    return out_path
