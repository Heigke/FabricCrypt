"""Phase 11C — Task A: Maximum sampling depth capture.

Captures multi-channel substrate signals at the highest sustained rates the
kernel permits, for 60 s each (THERMAL <70C; bail + wait_cool if exceeded).

Channels and target rates:
  RAPL (package energy)        -> aim 1 kHz sustained, derive instantaneous W
  per-core util (proc/stat)    -> 10 kHz best-effort (jitter signal)
  GPU power (amdgpu power1)    -> 100 Hz
  GPU temp / freq              -> 100 Hz
  hwmon thermal_zone0..N temp  -> 10 Hz
  NVMe IRQ rate                -> 100 Hz (delta of /proc/interrupts nvme)

Each channel is captured by its own thread with its own monotonic schedule.
Output: results/.../embodiment11c/<host>_max.npz containing one (ts, vals)
per channel as object arrays (different lengths/rates).

Usage:
    sudo HSA_OVERRIDE_GFX_VERSION=11.0.0 \
        venv/bin/python scripts/identity_benchmark/embodiment11c/max_sample_rate.py \
        --host ikaros --secs 60
"""
from __future__ import annotations
import argparse, glob, os, socket, threading, time
from pathlib import Path
import numpy as np

REPO = Path("/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy")
OUT_DIR = REPO / "results/IDENTITY_BENCHMARK_2026-05-30/embodiment11c"
OUT_DIR.mkdir(parents=True, exist_ok=True)

THERM_BAIL_C = 70.0


def cpu_temp_c():
    try:
        with open("/sys/class/thermal/thermal_zone0/temp") as f:
            return int(f.read().strip()) / 1000.0
    except Exception:
        return 0.0


def wait_cool(target_c=55.0, timeout_s=120):
    t0 = time.time()
    while cpu_temp_c() > target_c and time.time() - t0 < timeout_s:
        time.sleep(2)


# --- per-channel samplers --------------------------------------------------

class Sampler(threading.Thread):
    def __init__(self, name, fn, hz, secs, ts_buf, val_buf, stop_evt):
        super().__init__(daemon=True)
        self.name = name
        self.fn = fn
        self.period = 1.0 / hz
        self.secs = secs
        self.ts_buf = ts_buf
        self.val_buf = val_buf
        self.stop_evt = stop_evt

    def run(self):
        t_end = time.monotonic() + self.secs
        next_t = time.monotonic()
        while not self.stop_evt.is_set() and time.monotonic() < t_end:
            v = self.fn()
            self.ts_buf.append(time.monotonic())
            self.val_buf.append(v)
            next_t += self.period
            sleep_s = next_t - time.monotonic()
            if sleep_s > 0:
                # ALWAYS sleep (no spin-wait); accept jitter to protect APU.
                time.sleep(sleep_s)
            else:
                # We're falling behind -> just resync, dropped sample budget
                next_t = time.monotonic()


# --- channel functions -----------------------------------------------------

_rapl_path = "/sys/class/powercap/intel-rapl:0/energy_uj"
try:
    _rapl_fd = os.open(_rapl_path, os.O_RDONLY)
except PermissionError:
    _rapl_fd = None


def read_rapl_uj():
    if _rapl_fd is None:
        return 0.0
    os.lseek(_rapl_fd, 0, 0)
    try:
        return float(os.read(_rapl_fd, 64).strip())
    except Exception:
        return 0.0


_stat_buf = bytearray(65536)
_stat_fd = os.open("/proc/stat", os.O_RDONLY)


def read_cpu_total_busy():
    """Sum across all cores: 1 - idle/total over a snapshot. Cheap proxy."""
    os.lseek(_stat_fd, 0, 0)
    n = os.read(_stat_fd, 65536)
    # only first line "cpu  user nice system idle iowait irq softirq ..."
    line = n.split(b"\n", 1)[0]
    parts = line.split()
    nums = [int(x) for x in parts[1:8]]
    total = sum(nums)
    idle = nums[3]
    return float(total - idle)  # cumulative busy jiffies


_gpu_pwr = "/sys/class/hwmon/hwmon7/power1_input"
_gpu_temp = "/sys/class/hwmon/hwmon7/temp1_input"
_gpu_freq = "/sys/class/hwmon/hwmon7/freq1_input"


def _read_int(path):
    try:
        with open(path) as f:
            return float(f.read().strip())
    except Exception:
        return 0.0


def read_gpu_power_uw():
    return _read_int(_gpu_pwr)


def read_gpu_temp_mc():
    return _read_int(_gpu_temp)


def read_gpu_freq_hz():
    return _read_int(_gpu_freq)


_thermal_paths = sorted(glob.glob("/sys/class/thermal/thermal_zone*/temp"))


def read_thermal_all():
    out = []
    for p in _thermal_paths:
        try:
            with open(p) as f:
                out.append(int(f.read().strip()))
        except Exception:
            out.append(0)
    return out


_irq_fd = os.open("/proc/interrupts", os.O_RDONLY)


def read_nvme_irq_total():
    os.lseek(_irq_fd, 0, 0)
    chunks = []
    while True:
        c = os.read(_irq_fd, 65536)
        if not c:
            break
        chunks.append(c)
    blob = b"".join(chunks).decode("latin1", "replace")
    total = 0
    for line in blob.splitlines():
        if "nvme" in line:
            parts = line.split()
            # parts[1:n_cpu+1] are per-cpu counts
            for tok in parts[1:]:
                if tok.isdigit():
                    total += int(tok)
                else:
                    break
    return float(total)


# --- run --------------------------------------------------------------------

def run(secs, host):
    if cpu_temp_c() > 65:
        print(f"[max] Pre-cool: APU={cpu_temp_c():.1f}C, waiting...")
        wait_cool(55.0)
    print(f"[max] start: APU={cpu_temp_c():.1f}C, secs={secs}")

    channels = {
        "rapl_uj":   (read_rapl_uj,        1000),
        "cpu_busy":  (read_cpu_total_busy,   500),
        "gpu_pwr":   (read_gpu_power_uw,    100),
        "gpu_temp":  (read_gpu_temp_mc,     100),
        "gpu_freq":  (read_gpu_freq_hz,     100),
        "thermal":   (read_thermal_all,      10),
        "nvme_irq":  (read_nvme_irq_total,  100),
    }

    stop = threading.Event()
    bufs = {name: ([], []) for name in channels}
    threads = []
    for name, (fn, hz) in channels.items():
        ts_buf, val_buf = bufs[name]
        t = Sampler(name, fn, hz, secs, ts_buf, val_buf, stop)
        threads.append(t)

    # thermal watchdog
    def watchdog():
        while not stop.is_set():
            if cpu_temp_c() > THERM_BAIL_C:
                print(f"[max] THERMAL BAIL {cpu_temp_c():.1f}C")
                stop.set()
                return
            time.sleep(0.5)
    wd = threading.Thread(target=watchdog, daemon=True); wd.start()

    t0 = time.monotonic()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    stop.set()
    elapsed = time.monotonic() - t0

    # report achieved rates
    achieved = {}
    out_arrays = {}
    for name, (ts, vals) in bufs.items():
        n = len(ts)
        hz = n / elapsed if elapsed else 0
        achieved[name] = hz
        out_arrays[name + "_ts"] = np.asarray(ts, dtype=np.float64)
        # thermal is variable-length list per sample -> 2D array
        if name == "thermal":
            try:
                arr = np.asarray(vals, dtype=np.float32)
            except Exception:
                # ragged -> pad to max length
                m = max((len(v) for v in vals), default=0)
                arr = np.zeros((len(vals), m), dtype=np.float32)
                for i, v in enumerate(vals):
                    arr[i, :len(v)] = v
        else:
            arr = np.asarray(vals, dtype=np.float64)
        out_arrays[name + "_val"] = arr
        print(f"  {name:12s} target={channels[name][1]:>6d} Hz  achieved={hz:8.1f} Hz  N={n}")

    out_arrays["meta_elapsed"] = np.array([elapsed])
    out_arrays["meta_host"] = np.array([host])
    out_arrays["meta_apu_end_c"] = np.array([cpu_temp_c()])
    out_path = OUT_DIR / f"{host}_max.npz"
    np.savez_compressed(out_path, **out_arrays)
    print(f"[max] saved {out_path} (elapsed={elapsed:.2f}s, APU={cpu_temp_c():.1f}C)")
    return achieved


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default=socket.gethostname())
    ap.add_argument("--secs", type=int, default=60)
    args = ap.parse_args()
    run(args.secs, args.host)


if __name__ == "__main__":
    main()
