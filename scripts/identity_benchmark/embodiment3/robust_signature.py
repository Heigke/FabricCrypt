"""Robust signature module for embodiment3.

Design principles (fixes from embodiment / embodiment2 post-mortem):

1. ROBUST STATS over N samples (median + IQR), not single-sample raw values.
2. COARSE QUANTIZATION (4 bits/feature) — single-sample noise can't flip a bit.
3. 50+ signals from broad hardware surface (hwmon enum, DMI, /proc, gpu_metrics,
   per-core latency, TSC drift, etc.) → maximum chassi-binding bandwidth.
4. Bins are derived from SAME-MACHINE p5/p95 spread, so per-feature
   quantization automatically respects measurement noise floor.

Public API:
    collect_full_signature(N_samples=100, sample_interval_s=0.5, label='') -> dict
    quantize_robust(signature_dict, n_bits=4) -> dict of bin indices
    signature_hash(quantized_dict) -> 256-bit hex sha256
    bit_distance(qa, qb) -> int (Hamming over bins, each bin = n_bits bits)
"""
from __future__ import annotations
import glob, hashlib, json, os, platform, re, socket, subprocess, sys, time
from pathlib import Path
from typing import Dict, List, Any, Tuple
import numpy as np

HOST = socket.gethostname()

# ---------------------------------------------------------------------------
# Static fingerprints (collected ONCE per chassi — don't vary across reboots)
# ---------------------------------------------------------------------------

def _read(p: str, default: str = "") -> str:
    try:
        return open(p).read().strip()
    except Exception:
        return default

def collect_static() -> Dict[str, Any]:
    """One-shot, ~10ms. DMI + cpuinfo + boot-time hw enum.
    These should be IDENTICAL across reboots on the same chassi."""
    s = {}
    s["dmi_board_name"] = _read("/sys/class/dmi/id/board_name")
    s["dmi_board_serial"] = _read("/sys/class/dmi/id/board_serial")
    s["dmi_product_name"] = _read("/sys/class/dmi/id/product_name")
    s["dmi_product_serial"] = _read("/sys/class/dmi/id/product_serial")
    s["dmi_bios_version"] = _read("/sys/class/dmi/id/bios_version")
    s["dmi_bios_date"] = _read("/sys/class/dmi/id/bios_date")
    s["dmi_chassis_serial"] = _read("/sys/class/dmi/id/chassis_serial")
    s["dmi_sys_vendor"] = _read("/sys/class/dmi/id/sys_vendor")
    # CPU model + count
    try:
        c = open("/proc/cpuinfo").read()
        s["cpu_model"] = (re.search(r"model name\s*:\s*(.*)", c) or ["",""])[1].strip() if re.search(r"model name\s*:\s*(.*)", c) else ""
        s["cpu_count"] = int(len(re.findall(r"^processor\s*:", c, re.M)))
        s["cpu_microcode"] = (re.search(r"microcode\s*:\s*(\S+)", c) or [None, ""])[1] if re.search(r"microcode\s*:\s*(\S+)", c) else ""
        s["cpu_cache_size"] = (re.search(r"cache size\s*:\s*(\S+ \S+)", c) or [None, ""])[1] if re.search(r"cache size\s*:\s*(\S+ \S+)", c) else ""
    except Exception:
        pass
    # Memory size (kB)
    try:
        m = open("/proc/meminfo").read()
        s["mem_total_kB"] = int((re.search(r"MemTotal:\s*(\d+)", m) or [None, 0])[1])
    except Exception:
        pass
    # Kernel + arch
    s["kernel_release"] = platform.release()
    s["arch"] = platform.machine()
    s["hostname"] = socket.gethostname()
    # hwmon NAMES (enum order is hardware-stable per chassi)
    hwmons = {}
    for d in sorted(glob.glob("/sys/class/hwmon/hwmon*")):
        try:
            hwmons[os.path.basename(d)] = open(d + "/name").read().strip()
        except Exception:
            pass
    s["hwmon_enum"] = hwmons
    # PCI device list (vendor:device for top-level devices) — chassi-fixed
    try:
        pci = subprocess.run(["lspci", "-n"], capture_output=True, text=True, timeout=5)
        if pci.returncode == 0:
            ids = sorted([ln.split()[2] for ln in pci.stdout.splitlines() if len(ln.split()) >= 3])
            s["pci_device_ids"] = "|".join(ids)
    except Exception:
        pass
    # GPU device path → marker / SoC info
    try:
        for c in ("card1", "card0"):
            v = "/sys/class/drm/" + c + "/device/vendor"
            if os.path.exists(v):
                s["gpu_card"] = c
                s["gpu_vendor"] = _read(v)
                s["gpu_device"] = _read("/sys/class/drm/" + c + "/device/device")
                s["gpu_revision"] = _read("/sys/class/drm/" + c + "/device/revision")
                break
    except Exception:
        pass
    return s


# ---------------------------------------------------------------------------
# Dynamic signal samplers — each returns a single float (or list of floats)
# ---------------------------------------------------------------------------

def _maybe_float(p: str) -> float:
    try:
        return float(open(p).read().strip())
    except Exception:
        return float("nan")

def sample_hwmon_all() -> Dict[str, float]:
    """Snapshot every hwmon temp/power/in/curr/freq/fan reading."""
    out = {}
    for d in sorted(glob.glob("/sys/class/hwmon/hwmon*")):
        try:
            name = open(d + "/name").read().strip()
        except Exception:
            name = os.path.basename(d)
        for f in sorted(os.listdir(d)):
            if re.match(r"^(temp|power|in|curr|freq|fan)\d+_(input|average|cap)$", f):
                v = _maybe_float(d + "/" + f)
                if np.isfinite(v):
                    out[f"{name}_{f}"] = v
    return out


def sample_thermal_zones() -> Dict[str, float]:
    """All /sys/class/thermal/thermal_zone*/temp values (mC)."""
    out = {}
    for d in sorted(glob.glob("/sys/class/thermal/thermal_zone*")):
        try:
            t = open(d + "/type").read().strip()
            v = _maybe_float(d + "/temp")
            if np.isfinite(v):
                out[f"thz_{t}_{os.path.basename(d)}"] = v
        except Exception:
            pass
    return out


def sample_cpufreq() -> Dict[str, float]:
    """Per-CPU current frequency."""
    out = {}
    for d in sorted(glob.glob("/sys/devices/system/cpu/cpu[0-9]*/cpufreq")):
        cpu = d.split("/cpu")[-1].split("/")[0]
        v = _maybe_float(d + "/scaling_cur_freq")
        if np.isfinite(v):
            out[f"cpu{cpu}_freq"] = v
    return out


def sample_loadavg_meminfo() -> Dict[str, float]:
    out = {}
    try:
        la = open("/proc/loadavg").read().split()
        out["loadavg_1"] = float(la[0])
        out["loadavg_5"] = float(la[1])
        out["loadavg_15"] = float(la[2])
    except Exception:
        pass
    try:
        m = open("/proc/meminfo").read()
        for k in ("MemFree", "MemAvailable", "Buffers", "Cached"):
            mm = re.search(rf"{k}:\s*(\d+)", m)
            if mm:
                out[f"mem_{k}_kB"] = float(mm.group(1))
    except Exception:
        pass
    return out


def sample_irq_counts() -> Dict[str, float]:
    """Per-CPU IRQ counts (top 5 IRQ lines by total)."""
    out = {}
    try:
        lines = open("/proc/interrupts").read().splitlines()
        if not lines:
            return out
        ncpus = len(lines[0].split())
        # Pick first ~20 IRQ lines
        chosen = 0
        for ln in lines[1:]:
            parts = ln.split()
            if len(parts) < ncpus + 1:
                continue
            try:
                irq = parts[0].rstrip(":")
                cnts = [float(x) for x in parts[1:1 + ncpus]]
                out[f"irq_{irq}_total"] = sum(cnts)
                chosen += 1
                if chosen >= 20:
                    break
            except Exception:
                continue
    except Exception:
        pass
    return out


def sample_tsc_drift(window_s: float = 0.05) -> Dict[str, float]:
    """Measure how monotonic_ns drifts vs wall-clock over a short window."""
    t_mono0 = time.monotonic_ns()
    t_wall0 = time.time_ns()
    time.sleep(window_s)
    t_mono1 = time.monotonic_ns()
    t_wall1 = time.time_ns()
    d_mono = t_mono1 - t_mono0
    d_wall = t_wall1 - t_wall0
    return {"tsc_drift_ratio": float(d_mono) / max(1.0, float(d_wall))}


def sample_gpu_metrics() -> Dict[str, float]:
    """Sample a few salient fields from amdgpu gpu_metrics binary blob.
    Best-effort; values that fail are skipped."""
    out = {}
    for c in ("card1", "card0"):
        p = f"/sys/class/drm/{c}/device/gpu_metrics"
        if os.path.exists(p):
            try:
                b = open(p, "rb").read()
                out["gpu_metrics_len"] = float(len(b))
                # First 64 bytes typically contain header + temps + clocks
                # Extract a few u16 / u32 fields at common offsets — these are
                # noisy but quantization will smooth them.
                if len(b) >= 32:
                    # u16s at offsets 8, 10, 12, 14, 16
                    for off in (8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30):
                        v = int.from_bytes(b[off:off + 2], "little")
                        out[f"gm_u16_{off:02d}"] = float(v)
            except Exception:
                pass
            break
    return out


def sample_power_envelope_micro(threads: int, dur_s: float = 0.5) -> Dict[str, float]:
    """Spin briefly, return mean/std of GPU power and APU temp during it."""
    # Use the amdgpu hwmon power1_input directly
    pw_path = None
    for d in glob.glob("/sys/class/hwmon/hwmon*"):
        try:
            if open(d + "/name").read().strip() == "amdgpu":
                pw_path = d + "/power1_input"
                break
        except Exception:
            pass
    tz_path = "/sys/class/thermal/thermal_zone0/temp"

    if threads > 0:
        import threading
        stop = [False]
        def worker():
            a = np.random.randn(256, 256).astype(np.float32)
            b = np.random.randn(256, 256).astype(np.float32)
            while not stop[0]:
                a = a @ b * 1e-3 + 1e-3
        ts = [threading.Thread(target=worker, daemon=True) for _ in range(threads)]
        [t.start() for t in ts]

    pws, tps = [], []
    t_end = time.time() + dur_s
    while time.time() < t_end:
        try:
            if pw_path:
                pws.append(float(open(pw_path).read().strip()) / 1e6)
            tps.append(float(open(tz_path).read().strip()) / 1000.0)
        except Exception:
            pass
        time.sleep(0.02)
    if threads > 0:
        stop[0] = True
        time.sleep(0.03)

    if not pws or not tps:
        return {}
    return {
        f"pw_t{threads}_mean": float(np.mean(pws)),
        f"pw_t{threads}_std": float(np.std(pws)),
        f"tp_t{threads}_mean": float(np.mean(tps)),
        f"tp_t{threads}_std": float(np.std(tps)),
    }


# ---------------------------------------------------------------------------
# Per-core latency rank (heavier — done ONCE per signature, not per sample)
# ---------------------------------------------------------------------------

_CORE_PROBE_SRC = """
import time, numpy as np, sys
N = int(sys.argv[1]); reps = int(sys.argv[2])
A = np.random.RandomState(42).randn(N,N).astype(np.float32)
B = np.random.RandomState(43).randn(N,N).astype(np.float32)
t0=time.time()
for _ in range(reps):
    C = A @ B * 1e-3 + 1e-3
    A = C
print(time.time()-t0)
"""

def measure_per_core_latency(max_cores: int = 8) -> Dict[str, float]:
    """Median time per core for a small matmul (one rep per core)."""
    src = "/tmp/_emb3_core_probe.py"
    Path(src).write_text(_CORE_PROBE_SRC)
    import multiprocessing
    n_cores = min(max_cores, multiprocessing.cpu_count())
    out = {}
    py = sys.executable
    for c in range(n_cores):
        try:
            r = subprocess.run(
                ["taskset", "-c", str(c), py, src, "128", "3"],
                capture_output=True, text=True, timeout=10,
            )
            if r.returncode == 0:
                out[f"core{c}_lat"] = float(r.stdout.strip())
        except Exception:
            pass
    return out


# ---------------------------------------------------------------------------
# Top-level collector
# ---------------------------------------------------------------------------

def collect_one_sample(workload_threads: int = 0, micro_dur_s: float = 0.4) -> Dict[str, float]:
    """One sample = single snapshot of ALL fast signals + a brief power probe."""
    s = {}
    s.update(sample_hwmon_all())
    s.update(sample_thermal_zones())
    s.update(sample_cpufreq())
    s.update(sample_loadavg_meminfo())
    s.update(sample_irq_counts())
    s.update(sample_tsc_drift(0.04))
    s.update(sample_gpu_metrics())
    s.update(sample_power_envelope_micro(workload_threads, micro_dur_s))
    return s


def collect_full_signature(N_samples: int = 100, sample_interval_s: float = 0.5,
                            label: str = "", per_core: bool = True,
                            thermal_guard_c: float = 75.0) -> Dict[str, Any]:
    """Returns dict with:
      static: dict of stable fingerprints
      per_signal: {signal_name: list_of_N_samples}
      per_core: {coreN_lat: float}  (one-shot, ~5s)
      meta: {host, label, started_at, finished_at, n_samples, thermal_max_c}
    Workloads rotated across samples to capture power envelope shape.
    """
    started = time.time()
    static = collect_static()
    per_signal: Dict[str, List[float]] = {}
    thermal_max = 0.0

    # Rotate workload threads so we capture envelope shape (idle/light/medium)
    # but stay thermally safe.
    workload_cycle = [0, 0, 1, 1, 2, 1, 0, 0]  # mostly idle; brief medium

    for i in range(N_samples):
        # Thermal guard: abort/wait if too hot
        try:
            tz = float(open("/sys/class/thermal/thermal_zone0/temp").read().strip()) / 1000.0
        except Exception:
            tz = 0.0
        if tz > thermal_max:
            thermal_max = tz
        if tz > thermal_guard_c:
            # Wait to cool
            wait_start = time.time()
            while time.time() - wait_start < 60.0:
                try:
                    tz = float(open("/sys/class/thermal/thermal_zone0/temp").read().strip()) / 1000.0
                except Exception:
                    tz = 0.0
                if tz < 55.0:
                    break
                time.sleep(2)

        wl = workload_cycle[i % len(workload_cycle)]
        try:
            sample = collect_one_sample(workload_threads=wl, micro_dur_s=0.25)
        except Exception as e:
            print(f"[sig] sample {i} error: {e}", flush=True)
            sample = {}

        for k, v in sample.items():
            per_signal.setdefault(k, []).append(float(v))

        if i % 20 == 0:
            print(f"[sig] sample {i}/{N_samples} apu={tz:.1f}C signals={len(sample)}", flush=True)
        # Sleep between samples (subtract micro_dur)
        time.sleep(max(0.0, sample_interval_s - 0.25))

    per_core_data = {}
    if per_core:
        try:
            per_core_data = measure_per_core_latency()
        except Exception as e:
            print(f"[sig] per-core failed: {e}", flush=True)

    finished = time.time()
    return {
        "static": static,
        "per_signal": per_signal,
        "per_core": per_core_data,
        "meta": {
            "host": HOST,
            "label": label,
            "started_at": started,
            "finished_at": finished,
            "elapsed_s": finished - started,
            "n_samples": N_samples,
            "thermal_max_c": thermal_max,
            "n_signals": len(per_signal),
        }
    }


# ---------------------------------------------------------------------------
# Robust quantization
# ---------------------------------------------------------------------------

# Default per-feature bin spec: each feature's value is mapped to one of 2**n_bits bins
# Bins are anchored at fixed log/linear scales depending on signal type — but
# the COARSE-ENOUGH gate is dominated by the per-feature quantization step.

def _qbin(v: float, lo: float, hi: float, n_bits: int = 4) -> int:
    """Linear bin index in [0, 2**n_bits - 1]."""
    nb = 1 << n_bits
    if not np.isfinite(v):
        return 0
    if hi <= lo:
        return 0
    f = (v - lo) / (hi - lo)
    f = max(0.0, min(0.999999, f))
    return int(f * nb)


def quantize_signal_array(samples: List[float], n_bits: int = 4,
                          anchor_lo: float = None, anchor_hi: float = None) -> Dict[str, int]:
    """Reduce a list of N samples to 3 bin indices: median, p5, p95.
    If anchor_lo/hi not given, use samples' min/max with 5% margin."""
    a = np.asarray([s for s in samples if np.isfinite(s)], dtype=float)
    if len(a) == 0:
        return {"med_bin": 0, "p5_bin": 0, "p95_bin": 0}
    med = float(np.median(a))
    p5 = float(np.percentile(a, 5))
    p95 = float(np.percentile(a, 95))
    if anchor_lo is None:
        lo = float(np.min(a)) - 1e-9
        hi = float(np.max(a)) + 1e-9
    else:
        lo, hi = anchor_lo, anchor_hi
    return {
        "med_bin": _qbin(med, lo, hi, n_bits),
        "p5_bin": _qbin(p5, lo, hi, n_bits),
        "p95_bin": _qbin(p95, lo, hi, n_bits),
    }


# Static feature → string. They contribute via SHA256 of joined string.
STATIC_KEYS = (
    "dmi_board_name", "dmi_board_serial", "dmi_product_name", "dmi_product_serial",
    "dmi_bios_version", "dmi_chassis_serial", "dmi_sys_vendor",
    "cpu_model", "cpu_count", "cpu_microcode",
    "mem_total_kB", "kernel_release", "arch", "hostname",
    "pci_device_ids", "gpu_vendor", "gpu_device", "gpu_revision",
)

# Anchors for shared quantization across machines so cross-machine compare is meaningful.
# Choose ranges wide enough to cover both ikaros & daedalus.
ANCHORS = {
    # Power values typically in [0, 50] W; mC scale temperatures
    # frequency in Hz; mem_kB free is ~1e8-2e8.
}


def quantize_robust(sig: Dict[str, Any], n_bits: int = 4) -> Dict[str, Any]:
    """Reduce raw signature to a coarse quantized representation.

    Returns dict with:
      static_hash: sha256 of joined static keys
      dynamic_bins: {signal_name: {med_bin, p5_bin, p95_bin}}
      per_core_bins: {coreN: bin_index} — rank-based (relative)
      n_features: number of dynamic features included
    """
    out = {"n_bits": n_bits}
    # Static portion → single sha256
    static_str = "|".join(f"{k}={sig['static'].get(k, '')}" for k in STATIC_KEYS)
    out["static_hash"] = hashlib.sha256(static_str.encode()).hexdigest()
    out["static_string_len"] = len(static_str)

    # Dynamic per-signal quantization
    dynamic_bins = {}
    per_signal = sig.get("per_signal", {})
    for k in sorted(per_signal.keys()):
        samples = per_signal[k]
        # Use per-feature anchor = median ± 5*IQR to handle wide cross-machine variation
        a = np.asarray([s for s in samples if np.isfinite(s)], dtype=float)
        if len(a) < 3:
            continue
        med = float(np.median(a))
        iqr = float(np.percentile(a, 75) - np.percentile(a, 25))
        # Use wide anchor — robust to cross-machine but quantizes within-machine cleanly
        spread = max(iqr * 10.0, abs(med) * 0.5, 1.0)
        anchor_lo = med - spread
        anchor_hi = med + spread
        dynamic_bins[k] = quantize_signal_array(samples, n_bits=n_bits,
                                                  anchor_lo=anchor_lo, anchor_hi=anchor_hi)
    out["dynamic_bins"] = dynamic_bins
    out["n_features"] = len(dynamic_bins)

    # Per-core: convert latencies to rank (chassi-invariant under load) → bin
    per_core = sig.get("per_core", {})
    if per_core:
        items = sorted(per_core.items())
        vals = np.array([v for _, v in items])
        ranks = np.argsort(np.argsort(vals))
        out["per_core_bins"] = {k: int(r) for (k, _), r in zip(items, ranks)}
    else:
        out["per_core_bins"] = {}

    return out


def signature_hash(quantized: Dict[str, Any]) -> str:
    """Deterministic 256-bit SHA over (static_hash, sorted dynamic bins, per-core ranks)."""
    h = hashlib.sha256()
    h.update(quantized["static_hash"].encode())
    for k in sorted(quantized["dynamic_bins"].keys()):
        b = quantized["dynamic_bins"][k]
        h.update(f"{k}:{b['med_bin']}:{b['p5_bin']}:{b['p95_bin']};".encode())
    for k in sorted(quantized["per_core_bins"].keys()):
        h.update(f"{k}:{quantized['per_core_bins'][k]};".encode())
    return h.hexdigest()


def quantized_to_bitstring(quantized: Dict[str, Any]) -> str:
    """Concatenate every bin into a deterministic bit string.
    Used for bit-distance Hamming comparisons.
    Static portion is included as the static_hash bits."""
    nb = int(quantized.get("n_bits", 4))
    bits = []
    # Static hash → 256 bits
    sh = bytes.fromhex(quantized["static_hash"])
    for byte in sh:
        bits.append(f"{byte:08b}")
    # Dynamic: 3 bins per feature × nb bits each
    for k in sorted(quantized["dynamic_bins"].keys()):
        b = quantized["dynamic_bins"][k]
        for f in ("med_bin", "p5_bin", "p95_bin"):
            bits.append(f"{b[f]:0{nb}b}")
    # Per-core ranks: assume ≤16 cores, 4 bits each
    for k in sorted(quantized["per_core_bins"].keys()):
        bits.append(f"{quantized['per_core_bins'][k] & 0xF:04b}")
    return "".join(bits)


def bit_distance(qa: Dict[str, Any], qb: Dict[str, Any]) -> Tuple[int, int]:
    """Returns (hamming_distance, total_bits) over the bitstring representation.
    Pads to the shorter of the two if features differ."""
    ba = quantized_to_bitstring(qa)
    bb = quantized_to_bitstring(qb)
    n = min(len(ba), len(bb))
    if n == 0:
        return (0, 0)
    diff = sum(1 for i in range(n) if ba[i] != bb[i])
    return (diff, n)


# Convenience: load/save signature to JSON
def save_signature(sig: Dict[str, Any], path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(sig, indent=2, default=str))


def load_signature(path: str) -> Dict[str, Any]:
    return json.loads(Path(path).read_text())


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=100)
    ap.add_argument("--interval", type=float, default=0.5)
    ap.add_argument("--label", default="default")
    ap.add_argument("--out", required=True)
    ap.add_argument("--no-per-core", action="store_true")
    args = ap.parse_args()
    sig = collect_full_signature(N_samples=args.N, sample_interval_s=args.interval,
                                  label=args.label, per_core=not args.no_per_core)
    save_signature(sig, args.out)
    q = quantize_robust(sig)
    h = signature_hash(q)
    print(f"[sig] {args.label} n_features={q['n_features']} hash={h[:16]}... elapsed={sig['meta']['elapsed_s']:.1f}s")
    print(f"[sig] wrote {args.out}")
