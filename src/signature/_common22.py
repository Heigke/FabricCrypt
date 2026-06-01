"""Phase 22 common helpers — light, read-only exotic signals.

Conservative thermal: abort 65, pause 62, cool 55 (matches Phase 20).
"""
import os, sys, time, json, socket, subprocess, glob, hashlib

THERMAL_ZONE = '/sys/class/thermal/thermal_zone0/temp'


def get_apu_temp_c():
    try: return int(open(THERMAL_ZONE).read().strip()) / 1000.0
    except Exception: return 0.0


def thermal_guard(abort_c=65, pause_c=62, cool_c=55):
    t = get_apu_temp_c()
    if t >= abort_c:
        raise SystemExit(f"ABORT thermal {t:.1f}C >= {abort_c}C")
    if t >= pause_c:
        print(f"[thermal_guard] PAUSE {t:.1f}C -> <{cool_c}C", flush=True)
        while get_apu_temp_c() > cool_c:
            time.sleep(5)


def wait_cool(target_c=55, timeout_s=180):
    t0 = time.time()
    while True:
        t = get_apu_temp_c()
        if t < target_c: return t
        if time.time() - t0 > timeout_s: return t
        time.sleep(5)


def save_json(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        json.dump(data, f, indent=2, default=str)


def hostname():
    try: return open('/etc/hostname').read().strip()
    except Exception: return socket.gethostname()


def read_int(p, default=0):
    try: return int(open(p).read().strip())
    except Exception: return default


def read_str(p, default=''):
    try: return open(p).read().strip()
    except Exception: return default


def run_cmd(args, timeout=10):
    """Run command, capture stdout, never raise."""
    try:
        r = subprocess.run(args, capture_output=True, text=True,
                           timeout=timeout)
        return r.stdout or ''
    except Exception as e:
        return ''


def sudo_cmd(args, timeout=10):
    """Try sudo -n; return '' on failure."""
    try:
        r = subprocess.run(['sudo', '-n'] + args, capture_output=True,
                           text=True, timeout=timeout)
        return r.stdout or ''
    except Exception:
        return ''


def hash_bytes(b):
    return hashlib.sha256(b if isinstance(b, (bytes, bytearray))
                          else b.encode()).digest()


def hash_to_floats(h, n=8):
    """Convert leading n bytes of a hash digest into n float features."""
    return [float(h[i]) for i in range(min(n, len(h)))]
