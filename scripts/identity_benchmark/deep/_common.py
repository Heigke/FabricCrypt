"""Shared utilities for deep silicon fingerprint battery."""
import os, time, glob, socket, json

HOST = socket.gethostname()

def find_card():
    for c in ("card1","card0"):
        p = f"/sys/class/drm/{c}/device/gpu_metrics"
        if os.path.exists(p): return c
    return None

CARD = find_card()
GPU_METRICS = f"/sys/class/drm/{CARD}/device/gpu_metrics" if CARD else None
DPM_FILE = f"/sys/class/drm/{CARD}/device/power_dpm_force_performance_level" if CARD else None

def thermal_zones():
    zones = {}
    for d in sorted(glob.glob("/sys/class/thermal/thermal_zone*")):
        try:
            t = open(d+"/type").read().strip()
            zones[t] = d
        except: pass
    return zones

ZONES = thermal_zones()

def temp_c(zone_type="acpitz"):
    """Returns APU temp in C."""
    z = ZONES.get(zone_type) or list(ZONES.values())[0]
    try:
        return int(open(z+"/temp").read().strip())/1000.0
    except: return 0.0

def _find_amdgpu_hwmon():
    for d in sorted(glob.glob("/sys/class/hwmon/hwmon*")):
        try:
            if open(d+"/name").read().strip()=="amdgpu":
                return d
        except: pass
    return None

AMDGPU_HW = _find_amdgpu_hwmon()

def power_watts():
    """Instantaneous GPU/APU SoC power via amdgpu hwmon power1_input (microW)."""
    if not AMDGPU_HW: return 0.0
    try:
        return int(open(AMDGPU_HW+"/power1_input").read().strip())/1e6
    except:
        try:
            return int(open(AMDGPU_HW+"/power1_average").read().strip())/1e6
        except: return 0.0

def gpu_freq_hz():
    if not AMDGPU_HW: return 0
    try: return int(open(AMDGPU_HW+"/freq1_input").read().strip())
    except: return 0

def gpu_temp_c():
    if not AMDGPU_HW: return 0.0
    try: return int(open(AMDGPU_HW+"/temp1_input").read().strip())/1000.0
    except: return 0.0

# back-compat aliases (we no longer rely on RAPL energy delta)
def rapl_energy_uj():
    """Synthetic 'energy' counter from amdgpu power (uW * us). Monotonic per-process."""
    p_w = power_watts()
    return int(p_w * 1e6 * time.monotonic_ns()/1000)  # not useful as absolute

def rapl_power_watts(dt_s=0.1):
    return power_watts()

def gpu_metrics_raw():
    if not GPU_METRICS: return None
    try: return open(GPU_METRICS,"rb").read()
    except: return None

def wait_cool(thresh=60.0, timeout=120):
    t0=time.time()
    while time.time()-t0 < timeout:
        t = temp_c()
        if t < thresh: return True
        time.sleep(2)
    return False

def abort_if_hot(thresh=72.0):
    t = temp_c()
    if t > thresh:
        print(f"[ABORT] APU temp {t:.1f} > {thresh}", flush=True)
        return True
    return False

def bootstrap_ci(data, fn=None, n=1000, ci=95):
    import numpy as np
    data = np.asarray(data, dtype=float)
    if len(data)==0: return (float("nan"),float("nan"),float("nan"))
    if fn is None: fn = np.mean
    rng = np.random.default_rng(0xABCDEF)
    boots = np.array([fn(rng.choice(data,size=len(data),replace=True)) for _ in range(n)])
    lo, hi = np.percentile(boots,[(100-ci)/2, 100-(100-ci)/2])
    return float(fn(data)), float(lo), float(hi)

def save_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path,"w") as f: json.dump(obj,f,indent=2,default=str)
    print(f"[OK] wrote {path}", flush=True)

def host_label():
    return HOST
