"""L1-L15 lowest-level silicon-bound identity probes.

Each probe runs READ-ONLY and emits a JSON record:
    {probe, machine, ts, features:{...}, raw_path?}

Designed for thermal-safe operation: APU<68C strict, abort 72C.
"""
from __future__ import annotations
import json, os, time, struct, subprocess, hashlib, math, glob, ctypes
from pathlib import Path
from contextlib import contextmanager

THERM = "/sys/class/thermal/thermal_zone0/temp"
ABORT_C = 72.0
PAUSE_C = 68.0
RESUME_C = 55.0

def temp_c():
    try:
        return int(open(THERM).read().strip()) / 1000.0
    except Exception:
        return -1.0

def wait_cool(max_wait=120):
    t0 = time.time()
    while time.time() - t0 < max_wait:
        t = temp_c()
        if t >= ABORT_C:
            raise RuntimeError(f"ABORT temp={t:.1f}")
        if t <= RESUME_C:
            return
        time.sleep(2)

def thermal_check():
    t = temp_c()
    if t >= ABORT_C:
        raise RuntimeError(f"ABORT temp={t:.1f}")
    if t >= PAUSE_C:
        wait_cool()


# ---------- L1: hwmon ----------
def probe_L1(duration=20.0, hz=10.0):
    """Sample every readable hwmon sensor at 10 Hz for `duration` seconds."""
    sensors = []
    for hw in sorted(glob.glob("/sys/class/hwmon/hwmon*")):
        try:
            name = open(f"{hw}/name").read().strip()
        except Exception:
            name = "?"
        for f in sorted(os.listdir(hw)):
            if not (f.endswith("_input") or f.endswith("_average")):
                continue
            path = f"{hw}/{f}"
            try:
                open(path).read()
            except Exception:
                continue
            sensors.append((f"{name}:{f}", path))
    samples = {k: [] for k, _ in sensors}
    t0 = time.time()
    period = 1.0 / hz
    n = int(duration * hz)
    for i in range(n):
        if i % 50 == 0:
            thermal_check()
        for k, p in sensors:
            try:
                samples[k].append(int(open(p).read().strip()))
            except Exception:
                pass
        time.sleep(max(0, t0 + (i+1)*period - time.time()))
    feat = {}
    for k, vs in samples.items():
        if not vs: continue
        m = sum(vs)/len(vs)
        var = sum((v-m)**2 for v in vs)/len(vs)
        feat[k] = {"mean": m, "std": var**0.5, "n": len(vs),
                   "min": min(vs), "max": max(vs)}
    return {"n_sensors": len(sensors), "duration_s": duration, "features": feat}


# ---------- L2: MSR (read-only) ----------
def probe_L2():
    msr_addrs = {
        "PLATFORM_INFO_0xCE": 0xCE,
        "PERF_STATUS_0x198": 0x198,
        "MPERF_0xE7": 0xE7,
        "APERF_0xE8": 0xE8,
        "PKG_ENERGY_0x611": 0x611,
        "AMD_HWCR_0xC0010015": 0xC0010015,
        "AMD_INT_PENDING_0xC0010073": 0xC0010073,
        "AMD_RAPL_PWR_UNIT_0xC0010299": 0xC0010299,
        "AMD_CORE_ENERGY_0xC001029A": 0xC001029A,
        "AMD_PKG_ENERGY_0xC001029B": 0xC001029B,
        "AMD_PSTATE_LIMIT_0xC0010061": 0xC0010061,
        "AMD_PSTATE_CTL_0xC0010062": 0xC0010062,
        "AMD_PSTATE_STATUS_0xC0010063": 0xC0010063,
        "AMD_HW_PSTATE_0xC0010293": 0xC0010293,
    }
    out = {}
    for cpu in range(min(8, os.cpu_count() or 1)):
        path = f"/dev/cpu/{cpu}/msr"
        try:
            fd = os.open(path, os.O_RDONLY)
        except Exception as e:
            out[f"cpu{cpu}_err"] = str(e); continue
        for name, addr in msr_addrs.items():
            try:
                os.lseek(fd, addr, 0)
                data = os.read(fd, 8)
                v = int.from_bytes(data, "little")
                out[f"cpu{cpu}_{name}"] = v
            except Exception:
                pass
        os.close(fd)
    # Two energy samples 1s apart for RAPL delta (per-chip ADC behavior)
    def read_pkg_energy():
        try:
            fd = os.open("/dev/cpu/0/msr", os.O_RDONLY)
            os.lseek(fd, 0xC001029B, 0)
            v = int.from_bytes(os.read(fd, 8), "little")
            os.close(fd)
            return v
        except Exception:
            return None
    e1 = read_pkg_energy(); time.sleep(1.0); e2 = read_pkg_energy()
    if e1 is not None and e2 is not None:
        out["pkg_energy_delta_1s"] = (e2 - e1) & 0xFFFFFFFF
    return {"features": out}


# ---------- L3: /proc/interrupts ----------
def probe_L3(duration=30.0, hz=1.0):
    def snap():
        try:
            return open("/proc/interrupts").read()
        except Exception:
            return ""
    rows0 = {}
    n_samples = int(duration*hz)
    series = []
    for i in range(n_samples):
        if i % 10 == 0: thermal_check()
        txt = snap()
        d = {}
        for line in txt.splitlines()[1:]:
            parts = line.split()
            if len(parts) < 3: continue
            irq = parts[0].rstrip(":")
            try:
                counts = [int(x) for x in parts[1:1+os.cpu_count()] if x.isdigit()]
            except Exception:
                continue
            d[irq] = counts
        series.append(d)
        time.sleep(1.0/hz)
    feat = {}
    irqs = set()
    for s in series: irqs.update(s.keys())
    for irq in irqs:
        per_cpu_deltas = []
        for cpu_i in range(os.cpu_count() or 1):
            seq = []
            for s in series:
                v = s.get(irq, [])
                if cpu_i < len(v): seq.append(v[cpu_i])
            if len(seq) < 2: continue
            d = seq[-1] - seq[0]
            per_cpu_deltas.append(d)
        if not per_cpu_deltas: continue
        feat[f"irq_{irq}_total_delta"] = sum(per_cpu_deltas)
        # imbalance: max/mean
        m = sum(per_cpu_deltas)/len(per_cpu_deltas)
        if m > 0:
            feat[f"irq_{irq}_imbalance"] = max(per_cpu_deltas)/m
    return {"n_irqs": len(irqs), "features": feat}


# ---------- L4: scheduler jitter ----------
def probe_L4(samples=10000):
    deltas = []
    last = time.clock_gettime_ns(time.CLOCK_MONOTONIC)
    for _ in range(samples):
        now = time.clock_gettime_ns(time.CLOCK_MONOTONIC)
        deltas.append(now - last); last = now
    deltas.sort()
    n = len(deltas)
    mean = sum(deltas)/n
    var = sum((d-mean)**2 for d in deltas)/n
    return {"features": {
        "ns_min": deltas[0], "ns_p50": deltas[n//2], "ns_p99": deltas[int(n*0.99)],
        "ns_max": deltas[-1], "ns_mean": mean, "ns_std": var**0.5,
        "ns_p999": deltas[int(n*0.999)],
    }}


# ---------- L5: cache topology + pointer-chase latency ----------
def probe_L5():
    feat = {}
    # Topology
    for cpu in sorted(glob.glob("/sys/devices/system/cpu/cpu[0-9]*"))[:4]:
        cid = os.path.basename(cpu)
        for cache in sorted(glob.glob(f"{cpu}/cache/index*")):
            ci = os.path.basename(cache)
            try:
                lvl = open(f"{cache}/level").read().strip()
                sz = open(f"{cache}/size").read().strip()
                ws = open(f"{cache}/ways_of_associativity").read().strip()
                feat[f"{cid}_{ci}_L{lvl}_size"] = sz
                feat[f"{cid}_{ci}_L{lvl}_ways"] = int(ws)
            except Exception:
                pass
    # Pointer-chase latency at varying WS — use array shuffle
    import random
    random.seed(0)
    for ws_kb in [32, 256, 2048, 16384]:
        n = ws_kb * 1024 // 8  # 8 bytes per pointer (use index)
        idx = list(range(n)); random.shuffle(idx)
        # build cycle
        chase = [0]*n
        prev = idx[0]
        for k in range(1, n):
            chase[prev] = idx[k]; prev = idx[k]
        chase[prev] = idx[0]
        iters = min(500_000, n*10)
        p = 0
        t0 = time.perf_counter_ns()
        for _ in range(iters):
            p = chase[p]
        t1 = time.perf_counter_ns()
        feat[f"chase_{ws_kb}KB_ns_per_access"] = (t1-t0)/iters
        thermal_check()
    return {"features": feat}


# ---------- L6: NPU /dev/accel/accel0 ----------
def probe_L6():
    feat = {}
    if not os.path.exists("/dev/accel/accel0"):
        return {"features": {"available": False}}
    feat["available"] = True
    # Driver-exposed sysfs
    accel_sys = "/sys/class/accel/accel0"
    if os.path.isdir(accel_sys):
        for root, dirs, files in os.walk(accel_sys):
            for f in files:
                p = os.path.join(root, f)
                try:
                    val = open(p).read().strip()
                    if 0 < len(val) < 256:
                        feat[f"sysfs:{os.path.relpath(p, accel_sys)}"] = val
                except Exception:
                    pass
    # Try uname / driver
    try:
        out = subprocess.check_output(["lsmod"], text=True)
        for line in out.splitlines():
            if "amdxdna" in line or "xdna" in line:
                feat["lsmod"] = line.strip()
    except Exception:
        pass
    # Try opening device read-only (probe permission)
    try:
        fd = os.open("/dev/accel/accel0", os.O_RDONLY)
        feat["openable_ro"] = True
        os.close(fd)
    except Exception as e:
        feat["openable_ro"] = False
        feat["open_err"] = str(e)
    # Look for amdxdna in /sys/bus
    for p in glob.glob("/sys/bus/pci/drivers/amdxdna/*"):
        bn = os.path.basename(p)
        if ":" in bn:
            feat[f"amdxdna_dev"] = bn
    return {"features": feat}


# ---------- L7: DMI / SMBIOS ----------
def probe_L7():
    feat = {}
    # Read DMI sysfs (no sudo needed for some fields)
    base = "/sys/class/dmi/id"
    if os.path.isdir(base):
        for f in os.listdir(base):
            p = os.path.join(base, f)
            try:
                v = open(p).read().strip()
                if v and len(v) < 256:
                    feat[f] = v
            except Exception:
                pass
    # Hash the whole thing for fingerprint
    keys = sorted(k for k in feat if k not in ("modalias",))
    blob = "|".join(f"{k}={feat[k]}" for k in keys)
    feat["_dmi_hash"] = hashlib.sha256(blob.encode()).hexdigest()
    return {"features": feat}


# ---------- L8: ACPI tables ----------
def probe_L8():
    feat = {}
    base = "/sys/firmware/acpi/tables"
    if not os.path.isdir(base):
        return {"features": {"available": False}}
    feat["available"] = True
    total_size = 0
    table_hashes = {}
    for f in sorted(os.listdir(base)):
        p = os.path.join(base, f)
        if not os.path.isfile(p): continue
        try:
            data = open(p, "rb").read()
        except Exception:
            continue
        h = hashlib.sha256(data).hexdigest()[:16]
        table_hashes[f] = {"size": len(data), "sha256_16": h}
        total_size += len(data)
    feat["tables"] = table_hashes
    feat["total_bytes"] = total_size
    feat["n_tables"] = len(table_hashes)
    # Look for thermal trip points
    tz_dir = "/sys/class/thermal"
    if os.path.isdir(tz_dir):
        trips = {}
        for tz in sorted(os.listdir(tz_dir)):
            if not tz.startswith("thermal_zone"): continue
            d = os.path.join(tz_dir, tz)
            try:
                tt = open(f"{d}/type").read().strip()
            except Exception:
                tt = "?"
            tps = []
            for tf in sorted(glob.glob(f"{d}/trip_point_*_temp")):
                try:
                    tps.append(int(open(tf).read().strip()))
                except Exception:
                    pass
            trips[f"{tz}:{tt}"] = tps
        feat["thermal_trips"] = trips
    return {"features": feat}


# ---------- L9: Branch predictor proxy ----------
def probe_L9(iters=500_000):
    """Measure throughput of random-branch loop vs predictable-branch loop.
    Ratio is a proxy for BPU effectiveness."""
    import random
    random.seed(42)
    arr_rand = [random.randint(0,1) for _ in range(iters)]
    arr_pred = [i & 1 for i in range(iters)]
    def run(arr):
        s = 0
        t0 = time.perf_counter_ns()
        for v in arr:
            if v: s += 1
            else: s -= 1
        t1 = time.perf_counter_ns()
        return (t1-t0)/len(arr)
    r1 = run(arr_pred); thermal_check()
    r2 = run(arr_rand); thermal_check()
    r3 = run(arr_pred); thermal_check()
    return {"features": {
        "ns_per_iter_predictable_1": r1,
        "ns_per_iter_random": r2,
        "ns_per_iter_predictable_2": r3,
        "rand_vs_pred_ratio": r2/r1,
    }}


# ---------- L10: memory bandwidth curve ----------
def probe_L10():
    feat = {}
    try:
        import numpy as np
    except Exception:
        return {"features": {"numpy_missing": True}}
    sizes_kb = [4, 16, 64, 256, 1024, 4096, 16384, 65536]
    for kb in sizes_kb:
        n = kb * 1024 // 8  # float64
        a = np.zeros(n, dtype=np.float64); b = np.ones(n, dtype=np.float64)
        # warm
        _ = a + b
        thermal_check()
        # min of 3
        best = 1e18
        for _ in range(3):
            t0 = time.perf_counter_ns()
            c = a + b
            t1 = time.perf_counter_ns()
            best = min(best, t1-t0)
        bw_GB_s = (n*8*3) / best  # read a, read b, write c
        feat[f"bw_{kb}KB_GBps"] = bw_GB_s
    return {"features": feat}


# ---------- L11: TLB miss curve ----------
def probe_L11():
    import random, ctypes
    feat = {}
    PAGE = 4096
    random.seed(7)
    for n_pages in [16, 64, 256, 1024, 4096, 16384, 65536]:
        # Allocate one int per page
        n = n_pages * (PAGE // 8)
        idx = list(range(0, n, PAGE//8))  # one index per page
        random.shuffle(idx)
        # build cycle on a list of len n
        arr = [0]*n
        prev = idx[0]
        for k in range(1, len(idx)):
            arr[prev] = idx[k]; prev = idx[k]
        arr[prev] = idx[0]
        iters = min(200_000, len(idx)*5)
        p = idx[0]
        t0 = time.perf_counter_ns()
        for _ in range(iters):
            p = arr[p]
        t1 = time.perf_counter_ns()
        feat[f"tlb_chase_{n_pages}pg_ns"] = (t1-t0)/iters
        thermal_check()
    return {"features": feat}


# ---------- L12: TPM EK ----------
def probe_L12():
    feat = {}
    if not os.path.exists("/dev/tpm0") and not os.path.exists("/dev/tpmrm0"):
        return {"features": {"available": False}}
    feat["available"] = True
    # Try tpm2_readpublic of EK if tools installed
    for cmd, key in [
        (["tpm2_getcap", "properties-fixed"], "props"),
        (["tpm2_createek", "-c", "/tmp/ek_primary.ctx", "-G", "rsa", "-u", "/tmp/ek.pub"], "createek"),
        (["tpm2_readpublic", "-c", "/tmp/ek_primary.ctx"], "readek"),
    ]:
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
            feat[f"{key}_rc"] = r.returncode
            if r.stdout:
                # Hash the output to avoid leaking key material in JSON
                feat[f"{key}_stdout_sha256"] = hashlib.sha256(r.stdout.encode()).hexdigest()
                feat[f"{key}_stdout_len"] = len(r.stdout)
            if r.stderr and r.returncode != 0:
                feat[f"{key}_err"] = r.stderr[:200]
        except FileNotFoundError:
            feat[f"{key}_err"] = "tool_missing"
            break
        except Exception as e:
            feat[f"{key}_err"] = str(e)[:200]
    # Cleanup
    for p in ["/tmp/ek_primary.ctx", "/tmp/ek.pub"]:
        try: os.remove(p)
        except Exception: pass
    return {"features": feat}


# ---------- L13: power rail ripple ----------
def probe_L13(duration=10.0, hz=200.0):
    """Sample voltage/power sensors at high rate, compute spectrum."""
    try:
        import numpy as np
    except Exception:
        return {"features": {"numpy_missing": True}}
    paths = []
    for hw in sorted(glob.glob("/sys/class/hwmon/hwmon*")):
        try: name = open(f"{hw}/name").read().strip()
        except Exception: name = "?"
        for f in sorted(os.listdir(hw)):
            if f.startswith("in") and f.endswith("_input") or \
               f.startswith("power") and f.endswith("_input"):
                paths.append((f"{name}:{f}", f"{hw}/{f}"))
    if not paths:
        return {"features": {"no_rails": True}}
    n = int(duration * hz)
    period = 1.0/hz
    samples = {k: [] for k,_ in paths}
    t0 = time.time()
    for i in range(n):
        if i % 200 == 0: thermal_check()
        for k, p in paths:
            try: samples[k].append(int(open(p).read().strip()))
            except Exception: pass
        target = t0 + (i+1)*period
        sl = target - time.time()
        if sl > 0: time.sleep(sl)
    feat = {"actual_hz": n/(time.time()-t0)}
    for k, vs in samples.items():
        if len(vs) < 64: continue
        x = np.array(vs, dtype=float)
        x = x - x.mean()
        if x.std() < 1e-9:
            feat[f"{k}_std"] = 0.0
            continue
        sp = np.abs(np.fft.rfft(x))**2
        freqs = np.fft.rfftfreq(len(x), d=1.0/hz)
        # peak frequency
        i_pk = int(np.argmax(sp[1:])) + 1
        feat[f"{k}_std"] = float(x.std())
        feat[f"{k}_peak_hz"] = float(freqs[i_pk])
        feat[f"{k}_peak_pwr"] = float(sp[i_pk])
        feat[f"{k}_lf_pwr"] = float(sp[1:len(sp)//4].sum())
        feat[f"{k}_hf_pwr"] = float(sp[len(sp)//4:].sum())
    return {"features": feat}


# ---------- L14: compiler -march=native fingerprint ----------
def probe_L14():
    feat = {}
    src = """
#include <stdio.h>
int main(int argc, char**argv) {
    volatile double a=1.1, b=2.2, c=3.3;
    for (int i=0;i<argc;i++) c = a*c + b;
    printf("%f\\n", c);
    return 0;
}
"""
    import tempfile
    d = tempfile.mkdtemp(prefix="l14_")
    src_p = os.path.join(d, "ftest.c")
    bin_p = os.path.join(d, "ftest")
    open(src_p,"w").write(src)
    try:
        r = subprocess.run(
            ["gcc","-O3","-march=native","-fno-asynchronous-unwind-tables",
             "-fno-ident","-Wl,--build-id=none",
             src_p,"-o",bin_p],
            capture_output=True, text=True, timeout=30,
        )
        feat["gcc_rc"] = r.returncode
        if r.returncode == 0:
            data = open(bin_p,"rb").read()
            feat["bin_size"] = len(data)
            feat["bin_sha256"] = hashlib.sha256(data).hexdigest()
        # capture which -march was chosen
        r2 = subprocess.run(
            ["gcc","-march=native","-E","-v","-","/dev/null"],
            input="", capture_output=True, text=True, timeout=10,
        )
        for line in r2.stderr.splitlines():
            if "march=" in line or "mtune=" in line:
                feat["march_line"] = line.strip()[:300]; break
        # also gcc --version
        r3 = subprocess.run(["gcc","--version"], capture_output=True, text=True)
        feat["gcc_version"] = r3.stdout.splitlines()[0] if r3.stdout else ""
    except Exception as e:
        feat["err"] = str(e)
    finally:
        import shutil; shutil.rmtree(d, ignore_errors=True)
    return {"features": feat}


# ---------- L15: RDTSC vs CLOCK_MONOTONIC drift ----------
def probe_L15(duration_s=120.0):
    """Sample TSC and CLOCK_MONOTONIC pairs, compute drift + Allan dev."""
    try:
        import numpy as np
    except Exception:
        return {"features": {"numpy_missing": True}}
    # rdtsc via ctypes inline asm not portable; use _PyTime_GetMonotonicClock and
    # CLOCK_MONOTONIC_RAW vs CLOCK_REALTIME as proxy for hardware vs adjusted timing
    pairs = []
    t0 = time.time()
    n = 0
    while time.time() - t0 < duration_s:
        a = time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW)
        b = time.clock_gettime_ns(time.CLOCK_REALTIME)
        pairs.append((a, b))
        n += 1
        time.sleep(0.01)
        if n % 500 == 0: thermal_check()
    a = np.array([p[0] for p in pairs], dtype=np.float64)
    b = np.array([p[1] for p in pairs], dtype=np.float64)
    da = np.diff(a); db = np.diff(b)
    drift = (db - da)
    feat = {
        "n_samples": len(pairs),
        "duration_s": float((a[-1]-a[0])/1e9),
        "drift_mean_ns_per_10ms": float(drift.mean()),
        "drift_std_ns_per_10ms": float(drift.std()),
        "ratio_realtime_per_raw": float((b[-1]-b[0])/(a[-1]-a[0])),
    }
    # Allan deviation at tau=1,10,100 samples
    for tau in [1, 10, 100]:
        if len(drift) > 3*tau:
            chunks = drift[:len(drift)//tau*tau].reshape(-1, tau).mean(axis=1)
            if len(chunks) > 2:
                ad = np.sqrt(0.5*np.mean(np.diff(chunks)**2))
                feat[f"allan_tau{tau}_ns"] = float(ad)
    return {"features": feat}


PROBES = {
    "L1_hwmon": probe_L1,
    "L2_msr": probe_L2,
    "L3_interrupts": probe_L3,
    "L4_sched_jitter": probe_L4,
    "L5_cache_chase": probe_L5,
    "L6_npu": probe_L6,
    "L7_dmi": probe_L7,
    "L8_acpi": probe_L8,
    "L9_bpu": probe_L9,
    "L10_membw": probe_L10,
    "L11_tlb": probe_L11,
    "L12_tpm": probe_L12,
    "L13_rail_ripple": probe_L13,
    "L14_gcc_native": probe_L14,
    "L15_tsc_drift": probe_L15,
}


def run_all(machine, out_dir, only=None):
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    results = {}
    for name, fn in PROBES.items():
        if only and name not in only: continue
        thermal_check()
        t0 = time.time()
        try:
            r = fn()
            r["status"] = "ok"
        except Exception as e:
            r = {"status": "err", "error": str(e)[:500]}
        r["probe"] = name
        r["machine"] = machine
        r["ts"] = time.time()
        r["wall_s"] = time.time() - t0
        r["temp_c_end"] = temp_c()
        results[name] = r
        p = out_dir / f"{name}_{machine}.json"
        json.dump(r, open(p, "w"), indent=2, default=str)
        print(f"[{machine}] {name}: {r['status']} ({r['wall_s']:.1f}s, T={r['temp_c_end']:.1f})", flush=True)
    summary = out_dir / f"_summary_{machine}.json"
    json.dump({k: {"status": v.get("status"), "wall_s": v.get("wall_s")} for k,v in results.items()},
              open(summary, "w"), indent=2)
    return results


if __name__ == "__main__":
    import argparse, socket
    ap = argparse.ArgumentParser()
    ap.add_argument("--machine", default=socket.gethostname())
    ap.add_argument("--out", default="results/IDENTITY_BENCHMARK_2026-05-30/lowest_level")
    ap.add_argument("--only", default=None, help="comma list")
    ap.add_argument("--rep", default="r0", help="rep tag")
    args = ap.parse_args()
    only = args.only.split(",") if args.only else None
    out = Path(args.out) / args.rep
    run_all(args.machine, out, only=only)
