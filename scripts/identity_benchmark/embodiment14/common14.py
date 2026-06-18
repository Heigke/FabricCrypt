"""Phase 14 common helpers — strict thermal (abort 68C, pause 63C, cool 50C)."""
import time, os, json, socket, subprocess

THERMAL_ZONE = '/sys/class/thermal/thermal_zone0/temp'

def get_apu_temp_c():
    try:
        return int(open(THERMAL_ZONE).read().strip()) / 1000.0
    except Exception:
        return 0.0

def thermal_guard(abort_c=68, pause_c=63, cool_c=50, verbose=False):
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

def wait_cool(target_c=50, timeout_s=180):
    t0 = time.time()
    while True:
        t = get_apu_temp_c()
        if t < target_c:
            return t
        if time.time() - t0 > timeout_s:
            return t
        time.sleep(5)

def save_json(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        json.dump(data, f, indent=2, default=str)

def hostname():
    try:
        return open('/etc/hostname').read().strip()
    except Exception:
        return socket.gethostname()
