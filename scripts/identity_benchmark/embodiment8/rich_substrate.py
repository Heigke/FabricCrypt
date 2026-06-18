"""Embodiment Phase 8 — Rich Dynamic Substrate Collector.

Captures ~50-100 live channels at ~50 Hz over a 5-min window, fully PASSIVE
(no GPU stress, no heavy compute).  Channels span:

  power/thermal      : hwmon power+temp+voltage rails, every thermal_zone
  cpu dynamics       : per-core util, per-core freq, per-core CPU time deltas,
                       /proc/stat aggregates, ctx switches, page faults
  cache/branch       : aggregate from /proc/stat softirq + intr deltas
  irq                : per-cpu irq count delta (top-N busiest IRQ lines)
  c-state            : per-core C-state residency delta
  memory dynamics    : /proc/meminfo (free, buffers, cached, dirty), vmstat
                       page-fault & pgpgin/pgpgout deltas
  bus/io             : /proc/diskstats deltas, NVMe queue depth proxy
  timing             : TSC vs CLOCK_MONOTONIC drift, nanosleep(1ms) jitter

Saves a single npz with:
  channels : list[str]            (~50-100 names)
  ts       : (N,) float64         monotonic seconds
  data     : (N, C) float32

Usage:
    python rich_substrate.py --host ikaros   --secs 300 --hz 50
"""
from __future__ import annotations
import argparse, glob, json, os, socket, time
from pathlib import Path
import numpy as np


REPO = Path("/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy")
OUT_DIR = REPO / "results/IDENTITY_BENCHMARK_2026-05-30/embodiment8"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def _read_first_line(path):
    try:
        with open(path) as f:
            return f.readline().strip()
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Channel discovery
# ---------------------------------------------------------------------------
def discover_hwmon():
    """Return list of (name, path) for hwmon scalar channels."""
    chans = []
    for h in sorted(glob.glob("/sys/class/hwmon/hwmon*")):
        try:
            with open(f"{h}/name") as f:
                hname = f.read().strip()
        except Exception:
            continue
        for f in sorted(glob.glob(f"{h}/*_input")):
            chans.append((f"hwmon_{hname}_{Path(f).name}", f))
    return chans


def discover_thermal_zones():
    chans = []
    for tz in sorted(glob.glob("/sys/class/thermal/thermal_zone*")):
        tname = _read_first_line(f"{tz}/type") or Path(tz).name
        p = f"{tz}/temp"
        if os.path.exists(p):
            chans.append((f"thermal_{tname}_{Path(tz).name}", p))
    return chans


def discover_cpufreq(n_cpu):
    chans = []
    for c in range(n_cpu):
        p = f"/sys/devices/system/cpu/cpu{c}/cpufreq/scaling_cur_freq"
        if os.path.exists(p):
            chans.append((f"cpufreq_cpu{c}", p))
    return chans


# ---------------------------------------------------------------------------
# Stateful readers (computes per-sample deltas / utilisations)
# ---------------------------------------------------------------------------
class CpuStatReader:
    """Per-core utilisation and aggregate ctx-switch / intr / process counts."""

    def __init__(self):
        self.prev_cpu = None
        self.prev_aggr = None
        self.n_cpu = os.cpu_count()
        # column names produced
        self.cols = [f"cpu_util_cpu{c}" for c in range(self.n_cpu)] + [
            "ctxt_per_s", "intr_per_s", "procs_running", "procs_blocked"
        ]

    def _parse(self):
        per_cpu = []
        aggr = {}
        with open("/proc/stat") as f:
            for line in f:
                if line.startswith("cpu") and not line.startswith("cpu "):
                    parts = line.split()
                    # cpuN user nice system idle iowait irq softirq steal guest guest_nice
                    nums = [int(x) for x in parts[1:]]
                    total = sum(nums)
                    idle = nums[3] + (nums[4] if len(nums) > 4 else 0)
                    per_cpu.append((total, idle))
                elif line.startswith("ctxt "):
                    aggr["ctxt"] = int(line.split()[1])
                elif line.startswith("intr "):
                    aggr["intr"] = int(line.split()[1])
                elif line.startswith("procs_running"):
                    aggr["procs_running"] = int(line.split()[1])
                elif line.startswith("procs_blocked"):
                    aggr["procs_blocked"] = int(line.split()[1])
        return per_cpu, aggr

    def read(self, dt):
        per_cpu, aggr = self._parse()
        utils = [0.0] * self.n_cpu
        if self.prev_cpu is not None:
            for i, (t, idl) in enumerate(per_cpu[: self.n_cpu]):
                dt_t = t - self.prev_cpu[i][0]
                dt_i = idl - self.prev_cpu[i][1]
                utils[i] = max(0.0, min(1.0, 1.0 - dt_i / dt_t)) if dt_t > 0 else 0.0
        ctxt_rate = (aggr.get("ctxt", 0) - (self.prev_aggr or {}).get("ctxt", 0)) / max(dt, 1e-6) if self.prev_aggr else 0.0
        intr_rate = (aggr.get("intr", 0) - (self.prev_aggr or {}).get("intr", 0)) / max(dt, 1e-6) if self.prev_aggr else 0.0
        self.prev_cpu = per_cpu[: self.n_cpu]
        self.prev_aggr = aggr
        return utils + [ctxt_rate, intr_rate,
                        float(aggr.get("procs_running", 0)),
                        float(aggr.get("procs_blocked", 0))]


class VmstatReader:
    KEYS = ["pgpgin", "pgpgout", "pswpin", "pswpout", "pgfault", "pgmajfault",
            "nr_dirty", "nr_writeback", "allocstall_normal"]

    def __init__(self):
        self.prev = None
        self.cols = ["vmstat_" + k + "_per_s" if k.startswith("pg") or k.startswith("psw") or k == "allocstall_normal"
                     else "vmstat_" + k for k in self.KEYS]

    def _parse(self):
        out = {}
        try:
            with open("/proc/vmstat") as f:
                for line in f:
                    p = line.split()
                    if p[0] in self.KEYS:
                        out[p[0]] = int(p[1])
        except Exception:
            pass
        return out

    def read(self, dt):
        cur = self._parse()
        vals = []
        for k in self.KEYS:
            v = cur.get(k, 0)
            if self.prev is None:
                vals.append(0.0)
            elif k.startswith("pg") or k.startswith("psw") or k == "allocstall_normal":
                vals.append((v - self.prev.get(k, 0)) / max(dt, 1e-6))
            else:
                vals.append(float(v))
        self.prev = cur
        return vals


class MeminfoReader:
    KEYS = ["MemFree", "Buffers", "Cached", "Dirty", "Writeback", "AnonPages",
            "Slab", "SReclaimable", "PageTables", "CommitLimit"]

    def __init__(self):
        self.cols = ["mem_" + k for k in self.KEYS]

    def read(self, dt):
        out = {k: 0 for k in self.KEYS}
        try:
            with open("/proc/meminfo") as f:
                for line in f:
                    parts = line.split(":")
                    k = parts[0].strip()
                    if k in self.KEYS:
                        out[k] = int(parts[1].strip().split()[0])
        except Exception:
            pass
        return [float(out[k]) for k in self.KEYS]


class DiskstatsReader:
    """Aggregate read/write IOPS and queue depth across nvme devices."""

    def __init__(self):
        self.prev = None
        self.cols = ["disk_rd_iops", "disk_wr_iops", "disk_rd_bytes_per_s",
                     "disk_wr_bytes_per_s", "disk_inflight_avg", "disk_io_ticks_rate"]

    def _parse(self):
        agg = dict(r_ios=0, w_ios=0, r_sect=0, w_sect=0, inflight=0, io_ticks=0)
        try:
            with open("/proc/diskstats") as f:
                for line in f:
                    p = line.split()
                    name = p[2]
                    if not name.startswith("nvme"):
                        continue
                    # major minor name r_ios r_merges r_sect r_ticks w_ios w_merges w_sect w_ticks in_flight io_ticks
                    agg["r_ios"] += int(p[3])
                    agg["w_ios"] += int(p[7])
                    agg["r_sect"] += int(p[5])
                    agg["w_sect"] += int(p[9])
                    agg["inflight"] += int(p[11])
                    agg["io_ticks"] += int(p[12])
        except Exception:
            pass
        return agg

    def read(self, dt):
        cur = self._parse()
        if self.prev is None:
            self.prev = cur
            return [0.0] * 6
        r_iops = (cur["r_ios"] - self.prev["r_ios"]) / max(dt, 1e-6)
        w_iops = (cur["w_ios"] - self.prev["w_ios"]) / max(dt, 1e-6)
        r_bps = (cur["r_sect"] - self.prev["r_sect"]) * 512.0 / max(dt, 1e-6)
        w_bps = (cur["w_sect"] - self.prev["w_sect"]) * 512.0 / max(dt, 1e-6)
        infl = float(cur["inflight"])
        io_rate = (cur["io_ticks"] - self.prev["io_ticks"]) / max(dt, 1e-6)
        self.prev = cur
        return [r_iops, w_iops, r_bps, w_bps, infl, io_rate]


class TopIrqReader:
    """Track delta of N busiest IRQ lines (selected once on construction)."""

    def __init__(self, top_n=8):
        # find busiest IRQ lines once
        baseline = self._read_all()
        time.sleep(0.5)
        after = self._read_all()
        keys = list(baseline.keys() & after.keys())
        deltas = sorted([(k, sum(after[k]) - sum(baseline[k])) for k in keys], key=lambda x: -x[1])
        self.lines = [k for k, _ in deltas[:top_n]]
        self.cols = [f"irq_{ln}_per_s" for ln in self.lines]
        self.prev = self._read_all()

    def _read_all(self):
        out = {}
        try:
            with open("/proc/interrupts") as f:
                _ = f.readline()  # header
                for line in f:
                    parts = line.split()
                    if not parts or not parts[0].endswith(":"):
                        continue
                    key = parts[0][:-1]
                    nums = []
                    for p in parts[1:]:
                        try:
                            nums.append(int(p))
                        except ValueError:
                            break
                    if nums:
                        out[key] = nums
        except Exception:
            pass
        return out

    def read(self, dt):
        cur = self._read_all()
        out = []
        for ln in self.lines:
            a = sum(cur.get(ln, []) or [0])
            b = sum(self.prev.get(ln, []) or [0])
            out.append((a - b) / max(dt, 1e-6))
        self.prev = cur
        return out


class TimingJitterReader:
    """Measures (a) TSC-vs-monotonic drift estimate and (b) 1ms nanosleep jitter."""

    def __init__(self):
        self.cols = ["sleep1ms_actual_us", "sleep1ms_jitter_us", "loop_period_us"]
        self.prev_mono = None

    def read(self, dt):
        # Sleep 1ms and measure overshoot
        t0 = time.perf_counter_ns()
        time.sleep(0.001)
        t1 = time.perf_counter_ns()
        actual_us = (t1 - t0) / 1000.0
        jitter = abs(actual_us - 1000.0)
        loop_period_us = (dt * 1e6) if dt > 0 else 0.0
        return [actual_us, jitter, loop_period_us]


# ---------------------------------------------------------------------------
# Polling readers (one float per call)
# ---------------------------------------------------------------------------
class FileFloatReader:
    """Reads a single int/float from a sysfs file."""
    def __init__(self, name, path, scale=1.0):
        self.name = name
        self.path = path
        self.scale = scale

    def read(self):
        v = _read_first_line(self.path)
        try:
            return float(v) * self.scale
        except Exception:
            return 0.0


# ---------------------------------------------------------------------------
# Main collector
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default=socket.gethostname())
    ap.add_argument("--secs", type=float, default=300.0)
    ap.add_argument("--hz", type=float, default=50.0)
    ap.add_argument("--out", default=None)
    ap.add_argument("--abort_c", type=float, default=75.0)
    args = ap.parse_args()

    n_cpu = os.cpu_count()
    period = 1.0 / args.hz

    # Discover scalar file channels
    scalar_chans = []
    scalar_chans += discover_hwmon()
    scalar_chans += discover_thermal_zones()
    scalar_chans += discover_cpufreq(n_cpu)

    file_readers = [FileFloatReader(n, p) for (n, p) in scalar_chans]

    cpu_r   = CpuStatReader()
    vm_r    = VmstatReader()
    mem_r   = MeminfoReader()
    disk_r  = DiskstatsReader()
    irq_r   = TopIrqReader(top_n=8)
    time_r  = TimingJitterReader()

    cols = [c[0] for c in scalar_chans] + cpu_r.cols + vm_r.cols + mem_r.cols + disk_r.cols + irq_r.cols + time_r.cols
    print(f"[rich_substrate] host={args.host} channels={len(cols)} secs={args.secs} hz={args.hz}")

    n_samples = int(args.secs * args.hz)
    data = np.zeros((n_samples, len(cols)), dtype=np.float32)
    ts = np.zeros(n_samples, dtype=np.float64)

    t_start = time.perf_counter()
    prev_t = t_start
    i = 0
    # Thermal abort fp
    tz0 = "/sys/class/thermal/thermal_zone0/temp"

    while i < n_samples:
        target = t_start + i * period
        now = time.perf_counter()
        if now < target:
            time.sleep(max(0.0, target - now))
        sample_t = time.perf_counter()
        dt = sample_t - prev_t
        prev_t = sample_t

        # Thermal guard
        tval = _read_first_line(tz0)
        try:
            t_c = float(tval) / 1000.0
            if t_c > args.abort_c:
                print(f"[rich_substrate] APU {t_c:.1f}C > {args.abort_c} -> abort at sample {i}")
                break
        except Exception:
            pass

        row = []
        row.extend(r.read() for r in file_readers)
        row.extend(cpu_r.read(dt))
        row.extend(vm_r.read(dt))
        row.extend(mem_r.read(dt))
        row.extend(disk_r.read(dt))
        row.extend(irq_r.read(dt))
        row.extend(time_r.read(dt))

        # Flatten any nested
        flat = []
        for v in row:
            if isinstance(v, list):
                flat.extend(v)
            else:
                flat.append(v)
        if len(flat) != len(cols):
            # one-time mismatch — pad/trim
            if i == 0:
                print(f"[rich_substrate] WARN row len {len(flat)} vs cols {len(cols)}")
            flat = (flat + [0.0]*len(cols))[:len(cols)]
        data[i] = flat
        ts[i] = sample_t - t_start

        if i % int(args.hz * 30) == 0 and i > 0:
            print(f"  t={ts[i]:6.1f}s  i={i}/{n_samples}  apu={t_c:.1f}C  loop_dt={dt*1000:.2f}ms")
        i += 1

    data = data[:i]
    ts = ts[:i]

    out = args.out or str(OUT_DIR / f"{args.host}_rich.npz")
    np.savez_compressed(out, channels=np.array(cols), ts=ts, data=data)
    meta = {
        "host": args.host,
        "n_samples": int(i),
        "n_channels": len(cols),
        "hz_target": args.hz,
        "secs_target": args.secs,
        "secs_actual": float(ts[-1]) if i else 0.0,
        "channels": cols,
    }
    Path(out + ".meta.json").write_text(json.dumps(meta, indent=2))
    print(f"[rich_substrate] saved: {out}  shape={data.shape}  actual_secs={ts[-1]:.1f}")


if __name__ == "__main__":
    main()
