"""Phase 11C — Task C: Bias-grid × time-grid systematic probing.

For 5 CPU bias settings, capture a short trace and compute aggregations at
8 logarithmic windows. Result: per-machine 5x8 signature matrix.

Bias settings (amd-pstate hardware here; falls back to EPP modulation):
  S0 : performance + EPP=power           (low-bias proxy = "powersave")
  S1 : powersave + EPP=balance_performance
  S2 : powersave + EPP=balance_power     (close to "balanced")
  S3 : powersave + scaling_max_freq=1200MHz   (custom 1.2 GHz cap)
  S4 : powersave + scaling_max_freq=800MHz    (custom 800 MHz cap)

Per setting we capture ~8 s at the rates max_sample_rate uses (lower, since we
need 5 settings). Aggregations: mean, std at 8 log windows.

Output: <host>_bias_grid.npz with X (5,8,F), feature_names, settings.
"""
from __future__ import annotations
import argparse, glob, os, socket, subprocess, threading, time
from pathlib import Path
import numpy as np

REPO = Path("/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy")
OUT_DIR = REPO / "results/IDENTITY_BENCHMARK_2026-05-30/embodiment11c"

CAP_SECS = 8
SETTINGS = [
    {"name": "S0_perf_EPP_power",      "gov": "performance", "epp": "power",                 "max_freq": None},
    {"name": "S1_ps_EPP_balperf",      "gov": "powersave",   "epp": "balance_performance",   "max_freq": None},
    {"name": "S2_ps_EPP_balpwr",       "gov": "powersave",   "epp": "balance_power",         "max_freq": None},
    {"name": "S3_ps_cap_1200",         "gov": "powersave",   "epp": "balance_power",         "max_freq": 1200000},
    {"name": "S4_ps_cap_800",          "gov": "powersave",   "epp": "balance_power",         "max_freq": 800000},
]
THERM_BAIL_C = 70.0
LOG_WINDOWS_MS = [1, 5, 25, 100, 500, 2500, 10000, 50000]


def cpu_temp_c():
    try:
        with open("/sys/class/thermal/thermal_zone0/temp") as f:
            return int(f.read().strip()) / 1000.0
    except Exception:
        return 0.0


def wait_cool(target_c=50.0, timeout=180):
    t0 = time.time()
    while cpu_temp_c() > target_c and time.time() - t0 < timeout:
        time.sleep(3)


def _write_all_cpus(rel_path, val, use_sudo=True):
    paths = sorted(glob.glob(f"/sys/devices/system/cpu/cpu*/cpufreq/{rel_path}"))
    if not paths:
        return False
    if use_sudo:
        cmd = ["sudo", "-n", "tee"] + paths
        try:
            subprocess.run(cmd, input=str(val).encode() + b"\n",
                           check=True, stdout=subprocess.DEVNULL,
                           stderr=subprocess.DEVNULL)
            return True
        except subprocess.CalledProcessError:
            return False
    else:
        ok = True
        for p in paths:
            try:
                with open(p, "w") as f:
                    f.write(str(val))
            except Exception:
                ok = False
        return ok


def apply_setting(s):
    okg = _write_all_cpus("scaling_governor", s["gov"])
    if s.get("epp"):
        _write_all_cpus("energy_performance_preference", s["epp"])
    # reset max to hw max first
    try:
        with open("/sys/devices/system/cpu/cpu0/cpufreq/cpuinfo_max_freq") as f:
            hw_max = int(f.read().strip())
        _write_all_cpus("scaling_max_freq", hw_max)
    except Exception:
        pass
    if s.get("max_freq"):
        _write_all_cpus("scaling_max_freq", s["max_freq"])
    time.sleep(0.5)
    return okg


def reset_setting():
    _write_all_cpus("scaling_governor", "powersave")
    _write_all_cpus("energy_performance_preference", "balance_performance")
    try:
        with open("/sys/devices/system/cpu/cpu0/cpufreq/cpuinfo_max_freq") as f:
            hw_max = int(f.read().strip())
        _write_all_cpus("scaling_max_freq", hw_max)
    except Exception:
        pass


# --- light sampler (single thread, RAPL at 200 Hz, hwmon at 50 Hz) ----------

_rapl_path = "/sys/class/powercap/intel-rapl:0/energy_uj"
try:
    _rapl_fd = os.open(_rapl_path, os.O_RDONLY)
except Exception:
    _rapl_fd = None

_gpu_pwr = "/sys/class/hwmon/hwmon7/power1_input"
_gpu_temp = "/sys/class/hwmon/hwmon7/temp1_input"
_zone0 = "/sys/class/thermal/thermal_zone0/temp"
_cpu0_freq = "/sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq"


def _read(path):
    try:
        with open(path) as f:
            return float(f.read().strip())
    except Exception:
        return 0.0


def _read_rapl():
    if _rapl_fd is None:
        return 0.0
    os.lseek(_rapl_fd, 0, 0)
    try:
        return float(os.read(_rapl_fd, 64).strip())
    except Exception:
        return 0.0


def capture_trace(secs, hz=200):
    """Capture ~hz Hz multi-channel trace for `secs` seconds; thermal-bail."""
    ts, rapl, gpu_p, apu_t, gpu_t, cpu_f = [], [], [], [], [], []
    period = 1.0 / hz
    next_t = time.monotonic()
    t_end = next_t + secs
    while time.monotonic() < t_end:
        if cpu_temp_c() > THERM_BAIL_C:
            break
        now = time.monotonic()
        ts.append(now)
        rapl.append(_read_rapl())
        gpu_p.append(_read(_gpu_pwr))
        apu_t.append(_read(_zone0))
        gpu_t.append(_read(_gpu_temp))
        cpu_f.append(_read(_cpu0_freq))
        next_t += period
        sl = next_t - time.monotonic()
        if sl > 0:
            time.sleep(sl)
        else:
            next_t = time.monotonic()
    ts = np.array(ts)
    return {
        "ts": ts,
        "rapl": np.array(rapl),
        "gpu_p": np.array(gpu_p),
        "apu_t": np.array(apu_t),
        "gpu_t": np.array(gpu_t),
        "cpu_f": np.array(cpu_f),
    }


def aggregate_windows(trace):
    """Compute mean,std at log windows for each channel."""
    if trace["ts"].size < 10:
        # zero matrix
        n_ch = 5
        return np.zeros((len(LOG_WINDOWS_MS), n_ch * 2), dtype=np.float32)
    fs = trace["ts"].size / (trace["ts"][-1] - trace["ts"][0])
    # derive power from rapl
    rapl = trace["rapl"]
    pwr = np.diff(rapl) * fs * 1e-6
    pwr = np.concatenate([[pwr[0] if pwr.size else 0], pwr])
    chans = {
        "pwr":  pwr,
        "gpu_p": trace["gpu_p"],
        "apu_t": trace["apu_t"],
        "gpu_t": trace["gpu_t"],
        "cpu_f": trace["cpu_f"],
    }
    rows = []
    for win_ms in LOG_WINDOWS_MS:
        w = max(1, int(win_ms * fs / 1000.0))
        row = []
        for name, x in chans.items():
            if x.size < 2:
                row += [0.0, 0.0]; continue
            # window mean and std
            if w >= x.size:
                row += [float(x.mean()), float(x.std())]
            else:
                # rolling -> aggregate over all windows
                # using stride tricks
                n = x.size - w + 1
                view = np.lib.stride_tricks.sliding_window_view(x, w)
                row += [float(view.mean()), float(view.std(axis=1).mean())]
        rows.append(row)
    return np.array(rows, dtype=np.float32)  # (8, 10)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default=socket.gethostname())
    ap.add_argument("--secs", type=int, default=CAP_SECS)
    args = ap.parse_args()

    grid_mats = []
    setting_names = []
    achieved_temps = []
    try:
        for s in SETTINGS:
            wait_cool(50.0)
            print(f"[bias] {s['name']} (APU={cpu_temp_c():.1f}C)")
            apply_setting(s)
            # warm-up 1 s for governor to take effect
            time.sleep(1.0)
            trace = capture_trace(args.secs, hz=200)
            mat = aggregate_windows(trace)
            grid_mats.append(mat)
            setting_names.append(s["name"])
            achieved_temps.append(cpu_temp_c())
            print(f"  N={trace['ts'].size}, mat shape={mat.shape}, APU_end={cpu_temp_c():.1f}C")
    finally:
        reset_setting()

    X = np.stack(grid_mats, axis=0)  # (5, 8, 10)
    out = OUT_DIR / f"{args.host}_bias_grid.npz"
    np.savez_compressed(out, X=X,
                        setting_names=np.array(setting_names),
                        windows_ms=np.array(LOG_WINDOWS_MS),
                        end_temps=np.array(achieved_temps))
    print(f"[bias] saved {out}, X.shape={X.shape}")


if __name__ == "__main__":
    main()
