"""Continuous telemetry logger — captures APU/GPU temperature, CPU load,
memory pressure, GPU utilisation, top processes, dmesg tail. Writes to
results/telemetry/run_<timestamp>.{csv,jsonl} every 5 seconds.

Goal: when the system crashes / reboots, we can replay the last few minutes
to identify thermal trips, OOM, runaway processes, etc.

Run via:
    nohup python scripts/telemetry_logger.py > /dev/null 2>&1 &

Stop via:
    pkill -f telemetry_logger.py
"""
from __future__ import annotations
import json, os, signal, subprocess, sys, time
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results/telemetry"; OUT.mkdir(parents=True, exist_ok=True)

stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
CSV_PATH = OUT / f"run_{stamp}.csv"
JSONL_PATH = OUT / f"run_{stamp}.jsonl"
DMESG_PATH = OUT / f"run_{stamp}_dmesg.log"
LATEST = OUT / "latest.json"

INTERVAL_S = 5
THERMAL_WARN = 85.0
THERMAL_CRIT = 92.0    # below ACPI 99 — force-stop heavy work


def read_thermal():
    """Read all thermal zones; return list of (name, temp_C)."""
    out = []
    for zd in sorted(Path("/sys/class/thermal").glob("thermal_zone*")):
        try:
            t = int((zd / "temp").read_text().strip()) / 1000.0
            n = (zd / "type").read_text().strip()
            out.append((zd.name, n, t))
        except Exception:
            pass
    return out


def read_hwmon():
    """Read all hwmon temp/fan/power inputs."""
    out = []
    for hd in sorted(Path("/sys/class/hwmon").glob("hwmon*")):
        try:
            name = (hd / "name").read_text().strip()
        except Exception:
            name = "?"
        for f in sorted(hd.glob("temp*_input")) + sorted(hd.glob("fan*_input")) \
                + sorted(hd.glob("power*_average")):
            try:
                v = int(f.read_text().strip())
                key = f.name
                if "temp" in key: v_h = v / 1000.0          # mC → C
                elif "power" in key: v_h = v / 1e6           # µW → W
                else: v_h = v                                  # rpm
                out.append((hd.name, name, key, v_h))
            except Exception:
                pass
    return out


def read_loadavg():
    try:
        return [float(x) for x in open("/proc/loadavg").read().split()[:3]]
    except Exception:
        return [None, None, None]


def read_meminfo():
    info = {}
    try:
        for line in open("/proc/meminfo"):
            k, v = line.split(":", 1)
            v = v.strip().split()
            info[k] = int(v[0]) if v else 0
    except Exception:
        pass
    return {
        "MemFree_GB": info.get("MemFree", 0) / 1024**2,
        "MemAvail_GB": info.get("MemAvailable", 0) / 1024**2,
        "SwapUsed_GB": (info.get("SwapTotal", 0) - info.get("SwapFree", 0)) / 1024**2,
    }


def read_gpu():
    try:
        cp = subprocess.run(["rocm-smi", "--showuse", "--json"],
                            capture_output=True, text=True, timeout=2)
        d = json.loads(cp.stdout)
        return d
    except Exception:
        return None


def read_top_procs(n=5):
    try:
        cp = subprocess.run(["ps", "-eo", "pid,pcpu,pmem,comm",
                             "--sort=-pcpu", "--no-headers"],
                            capture_output=True, text=True, timeout=2)
        return cp.stdout.strip().splitlines()[:n]
    except Exception:
        return []


def dmesg_tail(n=20):
    try:
        # journalctl -k = kernel ring; doesn't need sudo
        cp = subprocess.run(["journalctl", "-k", "-n", str(n),
                             "--no-pager", "-o", "short-iso"],
                            capture_output=True, text=True, timeout=3)
        return cp.stdout
    except Exception:
        return ""


def main():
    print(f"[telem] starting; CSV={CSV_PATH.name}", flush=True)
    print(f"[telem] interval={INTERVAL_S}s; thermal warn>{THERMAL_WARN}°C "
          f"crit>{THERMAL_CRIT}°C", flush=True)

    # CSV header
    cols = ["ts", "uptime_s", "load_1m", "load_5m", "mem_avail_GB",
             "swap_GB"]
    # Discover thermal zones once
    tz = read_thermal()
    for zname, ztype, _ in tz:
        cols.append(f"tz_{zname}_{ztype}_C")
    hw = read_hwmon()
    for hd, hn, k, _ in hw:
        cols.append(f"{hd}_{hn}_{k}")
    cols += ["top_proc_1", "top_proc_2", "top_proc_3"]
    with open(CSV_PATH, "w") as f:
        f.write(",".join(cols) + "\n")

    last_dmesg = ""
    iter_idx = 0
    while True:
        ts = datetime.now().isoformat(timespec="seconds")
        try:
            ut = float(open("/proc/uptime").read().split()[0])
        except Exception:
            ut = -1
        load = read_loadavg()
        mem = read_meminfo()
        tz = read_thermal()
        hw = read_hwmon()
        procs = read_top_procs()
        # Build row
        row = [ts, f"{ut:.1f}", f"{load[0]:.2f}", f"{load[1]:.2f}",
                f"{mem['MemAvail_GB']:.2f}", f"{mem['SwapUsed_GB']:.2f}"]
        for _, _, t in tz:
            row.append(f"{t:.1f}")
        for _, _, _, v in hw:
            row.append(f"{v:.2f}" if isinstance(v, float) else str(v))
        for p in procs[:3]:
            row.append(p.replace(",", " "))
        while len(row) < len(cols):
            row.append("")
        with open(CSV_PATH, "a") as f:
            f.write(",".join(row) + "\n")

        # JSONL with full structure for replay
        rec = {
            "ts": ts, "uptime_s": ut,
            "load": load, "mem": mem,
            "thermal_zones": [{"name": n, "type": t, "temp_C": v}
                                for n, t, v in tz],
            "hwmon": [{"hd": hd, "name": hn, "key": k, "val": v}
                       for hd, hn, k, v in hw],
            "top_procs": procs,
        }
        # Highlight max APU/CPU temp
        max_temp = max((v for _, _, v in tz), default=None)
        rec["max_temp_C"] = max_temp
        with open(JSONL_PATH, "a") as f:
            f.write(json.dumps(rec) + "\n")

        # Latest snapshot for tail-followers
        with open(LATEST, "w") as f:
            json.dump(rec, f, indent=2)

        # Print warning if hot
        if max_temp is not None and max_temp >= THERMAL_WARN:
            sev = "CRIT" if max_temp >= THERMAL_CRIT else "WARN"
            print(f"[telem][{sev}] {ts} max_temp={max_temp:.1f}°C "
                  f"load={load[0]:.1f} mem_free={mem['MemAvail_GB']:.1f}GB "
                  f"top={procs[0] if procs else '?'}", flush=True)

        # Dmesg snapshot every 12 iterations (~1 min)
        if iter_idx % 12 == 0:
            d = dmesg_tail(50)
            if d != last_dmesg:
                with open(DMESG_PATH, "a") as f:
                    f.write(f"\n=== {ts} ===\n{d}\n")
                last_dmesg = d
        iter_idx += 1
        time.sleep(INTERVAL_S)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[telem] stopped by signal", flush=True)
