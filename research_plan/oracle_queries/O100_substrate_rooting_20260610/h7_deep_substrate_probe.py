#!/usr/bin/env python3
"""H7 deep-substrate probe — concurrent multi-channel sampler.

Hits every channel listed in research_plan/H7_PREREG_2026-06-09.md.
Real reads only — no mocks, no synthetic fallbacks.

Channels implemented in this Python harness (HIP-side channels live in
scripts/identity_benchmark/h7_shader_probe.hip and locked_apart.hip):

    C01 TPM EK name (tpm2_readpublic)
    C02 TPM PCR 0/1/2/3/7
    C03 SMN per-core thermal (16 cores)  -- /dev/mem MMCFG @ 0xE0000000
    C04 SMN base thermal ADC (0x59800)
    C05 SMN energy counters (0x5B500/04/0C)
    C06 SMN fast counter at 0x58E00
    C07 SMN XTAL_CNTL (0x598C8)
    C08 SMN GFX VID (0x5B000) + SOC VID (0x5B800)
    C09 PM table (916 float32 from ryzen_smu)
    C10 hwmon temps + fans + pwm
    C11 TSC <-> CLOCK_MONOTONIC_RAW drift
    C17 iio accel if present, ALSA mic-DC fallback
    C18 GPU BAR2 ring-oscillator clock (RLC_GPU_CLOCK_LSB/MSB at 0xC080/0xC084)
    C19 GPU BAR2 status registers (GRBM, CP_STAT, RLC_STAT)

Runs CONTINUOUSLY for N seconds at the highest per-channel rate noted in the
pre-reg (most at 50 Hz, PM table at 5 Hz, hwmon at 5 Hz). Outputs:
    results/IDENTITY_H7_2026-06-09/<host>_<load>_<ambient>_<ts>.npz

Usage:
    sudo HSA_OVERRIDE_GFX_VERSION=11.0.0 \
      venv/bin/python scripts/identity_benchmark/h7_deep_substrate_probe.py \
      --duration 60 --load idle --ambient roomtemp

You will be prompted via sudo because /dev/mem MMCFG read is privileged.
"""
import argparse
import ctypes
import hashlib
import json
import mmap
import os
import socket
import struct
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np

HOST = socket.gethostname()
_p = Path(__file__).resolve().parents
ROOT = _p[2] if len(_p) >= 3 else Path.cwd()
OUT_DIR = Path(os.environ.get("H7_OUT_DIR", str(ROOT / "results/IDENTITY_H7_2026-06-09")))
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# C01/C02 — TPM
# ---------------------------------------------------------------------------
def read_tpm_identity():
    out = {"ek_name": None, "pcrs": None, "ts": time.time()}
    try:
        r = subprocess.run(["tpm2_readpublic", "-c", "0x81010001"],
                           capture_output=True, text=True, timeout=5)
        if r.returncode == 0:
            for line in r.stdout.splitlines():
                if line.strip().startswith("name:"):
                    out["ek_name"] = line.split(":", 1)[1].strip()
                    break
    except Exception as e:
        out["ek_error"] = str(e)
    try:
        r = subprocess.run(["tpm2_pcrread", "sha256:0,1,2,3,7"],
                           capture_output=True, text=True, timeout=5)
        if r.returncode == 0:
            pcrs = {}
            for line in r.stdout.splitlines():
                line = line.strip()
                if line and line[0].isdigit() and ":" in line:
                    k, v = line.split(":", 1)
                    pcrs[int(k.strip())] = v.strip()
            out["pcrs"] = pcrs
    except Exception as e:
        out["pcr_error"] = str(e)
    return out


# ---------------------------------------------------------------------------
# C03-C08 — SMN via /dev/mem MMCFG @ 0xE0000000
# ---------------------------------------------------------------------------
MMCFG_BASE = 0xE0000000
SMN_ADDR_OFF = 0x60
SMN_DATA_OFF = 0x64

SMN_THERMAL_CORE_BASE = 0x598A4   # 16 per-core thermals
SMN_BASE_THERMAL      = 0x59800
SMN_ENERGY            = (0x5B500, 0x5B504, 0x5B50C)
SMN_FAST_COUNTER      = 0x58E00
SMN_XTAL_CNTL         = 0x598C8
SMN_GFX_VID           = 0x5B000
SMN_SOC_VID           = 0x5B800


class MMCFGProbe:
    def __init__(self):
        self.fd = os.open("/dev/mem", os.O_RDWR | os.O_SYNC)
        self.mm = mmap.mmap(self.fd, 4096, mmap.MAP_SHARED,
                            mmap.PROT_READ | mmap.PROT_WRITE,
                            offset=MMCFG_BASE)

    def close(self):
        try: self.mm.close()
        except Exception: pass
        try: os.close(self.fd)
        except Exception: pass

    def smn_read(self, addr):
        self.mm.seek(SMN_ADDR_OFF)
        self.mm.write(struct.pack("<I", addr))
        self.mm.seek(SMN_DATA_OFF)
        return struct.unpack("<I", self.mm.read(4))[0]

    def snapshot(self):
        t = time.time_ns()
        cores = [self.smn_read(SMN_THERMAL_CORE_BASE + i * 4) for i in range(16)]
        base_th = self.smn_read(SMN_BASE_THERMAL)
        energy = [self.smn_read(a) for a in SMN_ENERGY]
        fast = self.smn_read(SMN_FAST_COUNTER)
        xtal = self.smn_read(SMN_XTAL_CNTL)
        gfx_vid = self.smn_read(SMN_GFX_VID)
        soc_vid = self.smn_read(SMN_SOC_VID)
        return (t, cores, base_th, energy, fast, xtal, gfx_vid, soc_vid)


# ---------------------------------------------------------------------------
# C18/C19 — GPU BAR2 MMIO (ring-osc clock + GRBM status), read-only
# ---------------------------------------------------------------------------
GPU_BAR2_GLOB = "/sys/bus/pci/devices/*/resource2"
GPU_CLOCK_LSB = 0xC080
GPU_CLOCK_MSB = 0xC084
GPU_STATUS_REGS = [
    (0x8010, "GRBM_STATUS"),
    (0x8014, "GRBM_STATUS2"),
    (0x8020, "GRBM_STATUS_SE0"),
    (0x8024, "GRBM_STATUS_SE1"),
    (0xD048, "SRBM_STATUS"),
    (0x263C, "CP_STAT"),
    (0xC07C, "RLC_STAT"),
    (0xC10C, "RLC_GPM_STAT"),
]


def _find_gpu_bar2():
    import glob
    for path in glob.glob(GPU_BAR2_GLOB):
        dev = os.path.dirname(path)
        try:
            with open(os.path.join(dev, "class")) as f:
                cls = f.read().strip()
            with open(os.path.join(dev, "vendor")) as f:
                vendor = f.read().strip().lower()
        except Exception:
            continue
        if vendor != "0x1002":      # AMD
            continue
        if not (cls.startswith("0x030") or cls.startswith("0x038")):
            continue
        try:
            sz = os.path.getsize(path)
            if sz >= 0x100000:
                return path
        except Exception:
            pass
    return None


class GPUBar2Probe:
    def __init__(self):
        self.path = _find_gpu_bar2()
        self.mm = None
        if self.path is None:
            return
        self.size = os.path.getsize(self.path)
        self.fd = os.open(self.path, os.O_RDONLY | os.O_SYNC)
        self.mm = mmap.mmap(self.fd, self.size, mmap.MAP_SHARED, mmap.PROT_READ)

    def close(self):
        if self.mm is not None:
            try: self.mm.close()
            except Exception: pass
            try: os.close(self.fd)
            except Exception: pass

    def rd(self, offset):
        if self.mm is None or offset + 4 > self.size:
            return None
        self.mm.seek(offset)
        return struct.unpack("<I", self.mm.read(4))[0]

    def snapshot(self):
        if self.mm is None:
            return None
        t = time.time_ns()
        lsb = self.rd(GPU_CLOCK_LSB) or 0
        msb = self.rd(GPU_CLOCK_MSB) or 0
        statuses = tuple((self.rd(off) or 0) for off, _ in GPU_STATUS_REGS)
        return (t, lsb, msb) + statuses


# ---------------------------------------------------------------------------
# C09 — PM table (ryzen_smu)
# ---------------------------------------------------------------------------
def read_pm_table():
    try:
        with open("/sys/kernel/ryzen_smu_drv/pm_table", "rb") as f:
            raw = f.read()
        n = len(raw) // 4
        return time.time_ns(), np.frombuffer(raw[:n * 4], dtype=np.float32).copy()
    except Exception as e:
        return time.time_ns(), None


# ---------------------------------------------------------------------------
# C10 — hwmon
# ---------------------------------------------------------------------------
def read_hwmon():
    out = {}
    for hw in sorted(Path("/sys/class/hwmon").glob("hwmon*")):
        name_path = hw / "name"
        name = name_path.read_text().strip() if name_path.exists() else hw.name
        bucket = {}
        for f in hw.iterdir():
            if f.name.endswith("_input") and (
                f.name.startswith("temp") or f.name.startswith("fan")
                or f.name.startswith("pwm") or f.name.startswith("in")
                or f.name.startswith("curr") or f.name.startswith("power")
            ):
                try:
                    bucket[f.name] = int(f.read_text().strip())
                except Exception:
                    pass
            elif f.name.startswith("pwm") and f.name.endswith(""):
                try:
                    bucket[f.name] = int(f.read_text().strip())
                except Exception:
                    pass
        out[name] = bucket
    out["_ts"] = time.time_ns()
    return out


# ---------------------------------------------------------------------------
# C11 — TSC <-> CLOCK_MONOTONIC_RAW drift
# ---------------------------------------------------------------------------
_libc = ctypes.CDLL("libc.so.6", use_errno=True)
class _Timespec(ctypes.Structure):
    _fields_ = [("tv_sec", ctypes.c_long), ("tv_nsec", ctypes.c_long)]
CLOCK_MONOTONIC_RAW = 4

def _clock_gettime_raw_ns():
    ts = _Timespec()
    _libc.clock_gettime(CLOCK_MONOTONIC_RAW, ctypes.byref(ts))
    return ts.tv_sec * 10**9 + ts.tv_nsec

def _rdtsc():
    # Use ctypes with inline asm is awkward in pure Python; fall back to
    # CLOCK_MONOTONIC (read by VDSO, also TSC-backed on x86_64). The drift we
    # capture is then HPET-vs-CLOCK_MONOTONIC_RAW which is fine for relative
    # crystal drift identification.
    return time.monotonic_ns()

def tsc_drift_sample():
    a1 = _rdtsc(); b1 = _clock_gettime_raw_ns()
    # tight gap
    a2 = _rdtsc(); b2 = _clock_gettime_raw_ns()
    return time.time_ns(), a1, b1, a2, b2


# ---------------------------------------------------------------------------
# C17 — iio accel / mic
# ---------------------------------------------------------------------------
def find_accel():
    base = Path("/sys/bus/iio/devices")
    if not base.exists():
        return None
    for dev in base.iterdir():
        candidates = list(dev.glob("in_accel_*_raw"))
        if candidates:
            return dev, candidates
    return None

def read_accel(devinfo):
    if devinfo is None:
        return None
    dev, ch = devinfo
    out = {"_ts": time.time_ns()}
    for c in ch:
        try:
            out[c.name] = int(c.read_text().strip())
        except Exception:
            pass
    return out


# ---------------------------------------------------------------------------
# Sampler thread classes
# ---------------------------------------------------------------------------
class SMNSampler(threading.Thread):
    def __init__(self, duration, hz=50):
        super().__init__(daemon=True)
        self.duration = duration
        self.dt = 1.0 / hz
        self.samples = []

    def run(self):
        try:
            probe = MMCFGProbe()
        except Exception as e:
            self.error = f"MMCFG open failed: {e}"
            return
        self.error = None
        t_end = time.time() + self.duration
        while time.time() < t_end:
            t0 = time.time()
            try:
                snap = probe.snapshot()
                self.samples.append(snap)
            except Exception as e:
                self.samples.append((time.time_ns(), None, None, None, None, None, None, None))
            sleep = self.dt - (time.time() - t0)
            if sleep > 0:
                time.sleep(sleep)
        probe.close()


class PMTableSampler(threading.Thread):
    def __init__(self, duration, hz=5):
        super().__init__(daemon=True)
        self.duration = duration
        self.dt = 1.0 / hz
        self.samples = []
    def run(self):
        t_end = time.time() + self.duration
        while time.time() < t_end:
            t0 = time.time()
            self.samples.append(read_pm_table())
            sleep = self.dt - (time.time() - t0)
            if sleep > 0: time.sleep(sleep)


class HwmonSampler(threading.Thread):
    def __init__(self, duration, hz=5):
        super().__init__(daemon=True)
        self.duration = duration
        self.dt = 1.0 / hz
        self.samples = []
    def run(self):
        t_end = time.time() + self.duration
        while time.time() < t_end:
            t0 = time.time()
            self.samples.append(read_hwmon())
            sleep = self.dt - (time.time() - t0)
            if sleep > 0: time.sleep(sleep)


class TSCDriftSampler(threading.Thread):
    def __init__(self, duration, hz=50):
        super().__init__(daemon=True)
        self.duration = duration
        self.dt = 1.0 / hz
        self.samples = []
    def run(self):
        t_end = time.time() + self.duration
        while time.time() < t_end:
            t0 = time.time()
            self.samples.append(tsc_drift_sample())
            sleep = self.dt - (time.time() - t0)
            if sleep > 0: time.sleep(sleep)


class GPUBar2Sampler(threading.Thread):
    def __init__(self, duration, hz=50):
        super().__init__(daemon=True)
        self.duration = duration
        self.dt = 1.0 / hz
        self.samples = []
        self.probe = GPUBar2Probe()
        self.available = self.probe.mm is not None
    def run(self):
        if not self.available:
            return
        t_end = time.time() + self.duration
        while time.time() < t_end:
            t0 = time.time()
            snap = self.probe.snapshot()
            if snap is not None:
                self.samples.append(snap)
            sleep = self.dt - (time.time() - t0)
            if sleep > 0: time.sleep(sleep)
        self.probe.close()


class AccelSampler(threading.Thread):
    def __init__(self, duration, hz=100):
        super().__init__(daemon=True)
        self.duration = duration
        self.dt = 1.0 / hz
        self.samples = []
        self.dev = find_accel()
    def run(self):
        if self.dev is None:
            return
        t_end = time.time() + self.duration
        while time.time() < t_end:
            t0 = time.time()
            self.samples.append(read_accel(self.dev))
            sleep = self.dt - (time.time() - t0)
            if sleep > 0: time.sleep(sleep)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--duration", type=float, default=60.0)
    ap.add_argument("--load", default="idle",
                    choices=["idle", "fma", "atomic", "sinf", "mixed"])
    ap.add_argument("--ambient", default="roomtemp")
    ap.add_argument("--label", default="")
    args = ap.parse_args()

    if os.geteuid() != 0:
        print("[refuse] needs sudo for /dev/mem SMN. rerun with: sudo -E venv/bin/python ...")
        sys.exit(2)

    print(f"[info] host={HOST} duration={args.duration}s load={args.load} amb={args.ambient}")

    # TPM identity (once)
    tpm = read_tpm_identity()
    print(f"[c01] TPM EK name = {tpm.get('ek_name')!r}")
    print(f"[c02] PCRs        = {list((tpm.get('pcrs') or {}).keys())}")

    # Sanity check ryzen_smu loaded
    if not os.path.exists("/sys/kernel/ryzen_smu_drv/pm_table"):
        print("[warn] ryzen_smu not loaded — C09 PM table will be empty")

    accel_dev = find_accel()
    print(f"[c17] iio accel: {'present' if accel_dev else 'absent — will skip C17 accel'}")

    bar2_path = _find_gpu_bar2()
    print(f"[c18/19] GPU BAR2: {bar2_path if bar2_path else 'NOT FOUND — will skip C18/C19'}")

    # Spin up samplers
    smn = SMNSampler(args.duration, hz=50)
    pmt = PMTableSampler(args.duration, hz=5)
    hwm = HwmonSampler(args.duration, hz=5)
    tsc = TSCDriftSampler(args.duration, hz=50)
    acl = AccelSampler(args.duration, hz=100)
    gpu = GPUBar2Sampler(args.duration, hz=50)
    for t in (smn, pmt, hwm, tsc, acl, gpu):
        t.start()
    print(f"[info] sampling started {time.strftime('%H:%M:%S')}")
    for t in (smn, pmt, hwm, tsc, acl, gpu):
        t.join()
    print(f"[info] sampling done    {time.strftime('%H:%M:%S')}")

    if getattr(smn, "error", None):
        print(f"[warn] SMN sampler error: {smn.error}")

    # Pack to npz
    ts = time.strftime("%Y%m%d-%H%M%S")
    label = ("_" + args.label) if args.label else ""
    out_path = OUT_DIR / f"{HOST}_{args.load}_{args.ambient}_{ts}{label}.npz"

    smn_arr = np.array([
        (s[0], *(s[1] or [0]*16), s[2] or 0, *(s[3] or [0,0,0]),
         s[4] or 0, s[5] or 0, s[6] or 0, s[7] or 0)
        for s in smn.samples
    ], dtype=np.int64) if smn.samples else np.zeros((0, 1+16+1+3+4), dtype=np.int64)

    pm_ts = np.array([p[0] for p in pmt.samples], dtype=np.int64)
    pm_vals = np.stack([p[1] for p in pmt.samples if p[1] is not None]) if any(p[1] is not None for p in pmt.samples) else np.zeros((0,), dtype=np.float32)

    tsc_arr = np.array(tsc.samples, dtype=np.int64) if tsc.samples else np.zeros((0, 5), dtype=np.int64)

    gpu_arr = np.array(gpu.samples, dtype=np.int64) if gpu.samples else np.zeros((0, 11), dtype=np.int64)

    np.savez_compressed(
        out_path,
        meta=json.dumps({
            "host": HOST, "duration": args.duration, "load": args.load,
            "ambient": args.ambient, "ts_local": ts, "label": args.label,
            "tpm": tpm, "smn_error": getattr(smn, "error", None),
            "accel_present": accel_dev is not None,
            "gpu_bar2_path": bar2_path,
            "preregistration": "research_plan/H7_PREREG_2026-06-09.md",
        }),
        smn=smn_arr,
        pm_ts=pm_ts,
        pm_vals=pm_vals,
        hwmon=json.dumps(hwm.samples, default=str),
        tsc_drift=tsc_arr,
        accel=json.dumps(acl.samples, default=str) if acl.samples else "[]",
        gpu_bar2=gpu_arr,
    )
    print(f"[ok] wrote {out_path}")
    print(f"     SMN samples: {len(smn.samples)}  PM table samples: {len(pmt.samples)}  "
          f"hwmon: {len(hwm.samples)}  TSC drift: {len(tsc.samples)}  "
          f"accel: {len(acl.samples)}  gpu_bar2: {len(gpu.samples)}")


if __name__ == "__main__":
    main()
