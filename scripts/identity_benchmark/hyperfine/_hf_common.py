"""Hyperfine probe shared utilities — fast, low-impact telemetry sampling.

Sensors used (identical on ikaros + daedalus):
  - /sys/class/thermal/thermal_zone0/temp  (acpitz, APU envelope)
  - /sys/class/hwmon/hwmon7/temp1_input    (amdgpu edge)
  - /sys/class/hwmon/hwmon7/power1_input   (SoC power, uW)
  - /sys/class/hwmon/hwmon7/freq1_input    (GPU clock, Hz)

NO GPU compute is launched by these probes — passive observation only
(except P1/P3/P4/P5 which apply CPU-only or DPM perturbations).
"""
import os, time, glob, socket, json, math, struct
import numpy as np

HOST = socket.gethostname()
HW_AMDGPU = None
for d in sorted(glob.glob("/sys/class/hwmon/hwmon*")):
    try:
        if open(d+"/name").read().strip() == "amdgpu":
            HW_AMDGPU = d
            break
    except Exception:
        pass

TZ_ACPI = "/sys/class/thermal/thermal_zone0/temp"
P_FILE  = (HW_AMDGPU+"/power1_input") if HW_AMDGPU else None
T_FILE  = (HW_AMDGPU+"/temp1_input")  if HW_AMDGPU else None
F_FILE  = (HW_AMDGPU+"/freq1_input")  if HW_AMDGPU else None

# Find card for DPM toggle (P1)
CARD = None
for c in ("card1","card0"):
    if os.path.exists(f"/sys/class/drm/{c}/device/power_dpm_force_performance_level"):
        CARD = c; break
DPM = f"/sys/class/drm/{CARD}/device/power_dpm_force_performance_level" if CARD else None


def _ri(path):
    try:
        with open(path,"r") as f: return int(f.read().strip())
    except Exception:
        return -1

def apu_temp_c() -> float:
    return _ri(TZ_ACPI) / 1000.0

def gpu_temp_c() -> float:
    return _ri(T_FILE) / 1000.0 if T_FILE else 0.0

def power_w() -> float:
    return _ri(P_FILE) / 1e6 if P_FILE else 0.0

def gpu_freq_hz() -> int:
    return _ri(F_FILE) if F_FILE else 0


def abort_if_hot(thresh=72.0) -> bool:
    t = apu_temp_c()
    if t > thresh:
        print(f"[ABORT] APU {t:.1f}C > {thresh}", flush=True)
        return True
    return False

def wait_cool(target=60.0, timeout=180):
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout:
        if apu_temp_c() < target: return True
        time.sleep(2)
    return False


def sample_block(duration_s: float, rate_hz: float, want=("p","tg","ta","f")):
    """Tight sampling loop. Returns dict of np.float32 arrays + timestamps (s, monotonic)."""
    n = int(duration_s * rate_hz)
    dt = 1.0 / rate_hz
    out = {k: np.zeros(n, dtype=np.float32) for k in want}
    ts = np.zeros(n, dtype=np.float64)
    t0 = time.monotonic()
    next_t = t0
    for i in range(n):
        next_t += dt
        if "p"  in out: out["p"][i]  = power_w()
        if "tg" in out: out["tg"][i] = gpu_temp_c()
        if "ta" in out: out["ta"][i] = apu_temp_c()
        if "f"  in out: out["f"][i]  = gpu_freq_hz() / 1e6  # MHz
        ts[i] = time.monotonic()
        # busy-wait short, sleep long
        remaining = next_t - time.monotonic()
        if remaining > 1e-3:
            time.sleep(remaining)
    out["ts"] = ts
    return out


def save_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=lambda x: float(x) if isinstance(x,(np.floating,)) else (x.tolist() if isinstance(x,np.ndarray) else str(x)))
    print(f"[OK] {path}", flush=True)


def cohen_d(a, b):
    a = np.asarray(a, dtype=float); b = np.asarray(b, dtype=float)
    if len(a)<2 or len(b)<2: return float("nan")
    s = math.sqrt(((np.var(a, ddof=1)*(len(a)-1)) + (np.var(b, ddof=1)*(len(b)-1)))/(len(a)+len(b)-2))
    if s == 0: return float("inf") if np.mean(a)!=np.mean(b) else 0.0
    return float((np.mean(a)-np.mean(b))/s)


def allan_dev(y, dt, taus):
    """Overlapping Allan deviation of fractional samples y at sample interval dt for averaging times taus (s)."""
    y = np.asarray(y, dtype=float)
    N = len(y)
    out = []
    for tau in taus:
        m = max(1, int(round(tau/dt)))
        if 2*m >= N:
            out.append(float("nan")); continue
        # average into blocks of length m
        ybar = y[:m*(N//m)].reshape(-1, m).mean(axis=1)
        diffs = np.diff(ybar)
        ad = math.sqrt(0.5 * np.mean(diffs**2))
        out.append(ad)
    return np.array(out)
