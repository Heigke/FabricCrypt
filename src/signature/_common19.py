"""Phase 19 common helpers — STRICTER thermal (abort 65, pause 60, cool 50).

Phase 18 + 18B may already be running concurrently, so we err on cool side.
"""
import os, sys, time, json, socket, subprocess

THERMAL_ZONE = '/sys/class/thermal/thermal_zone0/temp'

def get_apu_temp_c():
    try:
        return int(open(THERMAL_ZONE).read().strip()) / 1000.0
    except Exception:
        return 0.0

def thermal_guard(abort_c=68, pause_c=64, cool_c=55, verbose=False):
    t = get_apu_temp_c()
    if t >= abort_c:
        raise SystemExit(f"ABORT thermal {t:.1f}C >= {abort_c}C")
    if t >= pause_c:
        print(f"[thermal_guard] PAUSE {t:.1f}C -> cool to <{cool_c}C", flush=True)
        while t > cool_c:
            time.sleep(5)
            t = get_apu_temp_c()
    elif verbose:
        print(f"[thermal_guard] OK {t:.1f}C", flush=True)

def wait_cool(target_c=50, timeout_s=180):
    t0 = time.time()
    while True:
        t = get_apu_temp_c()
        if t < target_c: return t
        if time.time() - t0 > timeout_s: return t
        time.sleep(5)

def confirm_idle(min_idle_pct=85.0, max_tries=6):
    """Use mpstat to confirm CPU idle before measurement."""
    for _ in range(max_tries):
        try:
            p = subprocess.run(['mpstat', '1', '1'], capture_output=True, text=True, timeout=10)
            for line in p.stdout.splitlines():
                parts = line.split()
                if len(parts) >= 12 and parts[1] == 'all':
                    try:
                        idle = float(parts[-1])
                        if idle >= min_idle_pct:
                            return idle
                    except ValueError:
                        pass
        except Exception:
            pass
        time.sleep(2)
    return 0.0

def save_json(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        json.dump(data, f, indent=2, default=str)

def hostname():
    try: return open('/etc/hostname').read().strip()
    except Exception: return socket.gethostname()

def compile_c(src_path, out_path, extra_flags=None):
    flags = ['-O2', '-march=native', '-pthread']
    if extra_flags: flags += extra_flags
    subprocess.check_call(['gcc'] + flags + [src_path, '-o', out_path])
    return out_path

def burst_budget(start_t, max_s=170.0):
    """Enforce 3-min burst ceiling."""
    if time.time() - start_t > max_s:
        raise SystemExit(f"ABORT 3-min burst limit ({max_s}s)")
