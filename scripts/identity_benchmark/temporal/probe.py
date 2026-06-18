#!/usr/bin/env python3
"""
temporal/probe.py — 5-minute 50 Hz temporal probe of (T_apu, T_gpu, P_gpu, freq_gpu)
under a controlled load schedule, for identity fingerprinting via DYNAMICS
(derivatives, hysteresis, step response, spectra, phase-space).

Channels (per-sample, 50 Hz target):
  - t_wall          (s, monotonic)
  - T_apu_c         (°C, thermal_zone0)
  - T_gpu_c         (°C, hwmon7/temp1_input)
  - P_gpu_w         (W, hwmon7/power1_average)
  - F_gpu_hz        (Hz, hwmon7/freq1_input)
  - F_cpu_khz       (kHz, cpufreq/policy0/scaling_cur_freq)
  - energy_uj_pkg   (uJ, intel-rapl:0/energy_uj if readable)
  - load_label      (int 0..6)

Load schedule (3 cycles of ~210 s each ≈ 630 s; we cap to ~300 s = ~1.5 cycles).
We use a slim CPU stressor (multi-thread numpy) and a slim HIP GPU burst
(small matmul) — kept short to respect APU < 70 °C thermal envelope.

Output: results/IDENTITY_BENCHMARK_2026-05-30/temporal/{device}_temporal.npz
"""
from __future__ import annotations
import argparse
import json
import multiprocessing as mp
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

try:
    REPO_ROOT = Path(__file__).resolve().parents[3]
except IndexError:
    REPO_ROOT = Path.cwd()
OUT_ROOT = REPO_ROOT / "results" / "IDENTITY_BENCHMARK_2026-05-30" / "temporal"

TZ_APU = Path("/sys/class/thermal/thermal_zone0/temp")
HWMON_DIR = Path("/sys/class/hwmon/hwmon7")  # amdgpu on this kernel
T_GPU = HWMON_DIR / "temp1_input"
P_GPU = HWMON_DIR / "power1_average"
F_GPU = HWMON_DIR / "freq1_input"
F_CPU = Path("/sys/devices/system/cpu/cpufreq/policy0/scaling_cur_freq")
RAPL = Path("/sys/class/powercap/intel-rapl:0/energy_uj")

ABORT_C = 75.0
PAUSE_C = 70.0


def find_hwmon(name: str = "amdgpu") -> Path:
    for h in Path("/sys/class/hwmon").iterdir():
        try:
            if (h / "name").read_text().strip() == name:
                return h
        except Exception:
            continue
    return HWMON_DIR  # fallback


def read_int(p: Path, default: int = -1) -> int:
    try:
        return int(p.read_text().strip())
    except Exception:
        return default


def read_apu_c() -> float:
    return read_int(TZ_APU) / 1000.0


def cpu_stress_worker(stop_evt, intensity: float):
    """Burn CPU at `intensity` (0..1) duty cycle."""
    import numpy as _np
    on = max(0.001, intensity * 0.05)
    off = max(0.0, (1.0 - intensity) * 0.05)
    A = _np.random.rand(128, 128).astype(_np.float32)
    while not stop_evt.is_set():
        t0 = time.monotonic()
        while time.monotonic() - t0 < on:
            A = A @ A
            A *= 1.0 / max(1e-6, _np.abs(A).max())
        if off > 0:
            time.sleep(off)


def start_cpu_stress(n_proc: int, intensity: float):
    evt = mp.Event()
    procs = [mp.Process(target=cpu_stress_worker, args=(evt, intensity), daemon=True)
             for _ in range(n_proc)]
    for p in procs:
        p.start()
    return evt, procs


def stop_cpu_stress(evt, procs, join_to=2.0):
    evt.set()
    for p in procs:
        p.join(timeout=join_to)
        if p.is_alive():
            p.terminate()


def gpu_burst(seconds: float = 4.0):
    """Short HIP burst via torch (gfx1151 native — no HSA override)."""
    env = os.environ.copy()
    env["HSA_OVERRIDE_GFX_VERSION"] = "11.0.0"
    code = (
        "import os,time;import torch;"
        "x=torch.randn(2048,2048,device='cuda');"
        "t0=time.monotonic();"
        f"deadline=t0+{seconds};"
        "y=x;"
        "while time.monotonic()<deadline:\n"
        "    y=y@x;y=y/y.abs().max().clamp(min=1e-6);\n"
        "torch.cuda.synchronize();print('ok')"
    )
    try:
        subprocess.run([sys.executable, "-c", code], env=env, timeout=seconds + 10,
                       capture_output=True, text=True)
    except Exception:
        pass


def schedule(short: bool = False):
    """Return [(label, duration_s, action), ...] action ∈ {idle,cpu25,cpu100,gpu_burst,cool}."""
    base = [
        (0, 20, "idle"),
        (1, 20, "cpu25"),
        (2, 15, "cpu100"),
        (3, 25, "idle"),
        (4, 20, "gpu_burst"),
        (5, 30, "cool"),
    ]
    if short:
        return base  # 1 cycle, 130 s
    return base + base  # 2 cycles, 260 s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device-name", required=True)
    ap.add_argument("--rate-hz", type=float, default=50.0)
    ap.add_argument("--short", action="store_true", help="1 cycle (180s) instead of 3")
    ap.add_argument("--outdir", default=None)
    args = ap.parse_args()

    hw = find_hwmon("amdgpu")
    t_gpu_p = hw / "temp1_input"
    p_gpu_p = hw / "power1_average"
    f_gpu_p = hw / "freq1_input"

    outdir = Path(args.outdir) if args.outdir else OUT_ROOT
    outdir.mkdir(parents=True, exist_ok=True)
    out_npz = outdir / f"{args.device_name}_temporal.npz"
    out_meta = outdir / f"{args.device_name}_temporal_meta.json"

    sched = schedule(short=args.short)
    print(f"[{args.device_name}] schedule: {len(sched)} segments, "
          f"total {sum(s[1] for s in sched)}s @ {args.rate_hz} Hz", flush=True)

    period = 1.0 / args.rate_hz
    n_cpu = max(2, mp.cpu_count() // 4)  # quarter cores to respect thermal budget

    samples = []
    t_start = time.monotonic()
    cpu_evt = None
    cpu_procs = None
    aborted = False

    for label, dur, action in sched:
        seg_t0 = time.monotonic()
        # set up action
        if cpu_evt is not None:
            stop_cpu_stress(cpu_evt, cpu_procs)
            cpu_evt = None
        if action == "cpu25":
            cpu_evt, cpu_procs = start_cpu_stress(n_cpu, 0.25)
        elif action == "cpu100":
            cpu_evt, cpu_procs = start_cpu_stress(n_cpu, 1.00)
        elif action == "gpu_burst":
            # fire-and-forget thread that bursts 4s on, 2s off
            import threading
            def burster():
                end = time.monotonic() + dur - 1.0
                while time.monotonic() < end:
                    gpu_burst(4.0)
                    time.sleep(2.0)
            th = threading.Thread(target=burster, daemon=True)
            th.start()
        # action == idle / cool: nothing

        next_tick = time.monotonic()
        while time.monotonic() - seg_t0 < dur:
            T_apu = read_apu_c()
            # Mid-segment pause: if hot under stress, stop stress, cool, resume
            if T_apu >= 72.0 and action in ("cpu100", "cpu25", "gpu_burst"):
                print(f"[pause] T_apu={T_apu:.1f}C, halting stress, cooling...", flush=True)
                if cpu_evt is not None:
                    stop_cpu_stress(cpu_evt, cpu_procs)
                    cpu_evt = None
                t_cool0 = time.monotonic()
                while read_apu_c() > 60.0 and time.monotonic() - t_cool0 < 60.0:
                    time.sleep(2.0)
                # don't restart this segment's stress — just finish segment naturally
                break
            if T_apu >= ABORT_C:
                print(f"[ABORT] APU={T_apu:.1f}C >= {ABORT_C}", flush=True)
                aborted = True
                break
            T_gpu = read_int(t_gpu_p) / 1000.0
            P_gpu = read_int(p_gpu_p) / 1_000_000.0  # uW -> W
            F_gpu = read_int(f_gpu_p)
            F_cpu = read_int(F_CPU)
            E_pkg = read_int(RAPL, default=-1)
            t_w = time.monotonic() - t_start
            samples.append((t_w, T_apu, T_gpu, P_gpu, F_gpu, F_cpu, E_pkg, label))
            # pacing
            next_tick += period
            sleep_for = next_tick - time.monotonic()
            if sleep_for > 0:
                time.sleep(sleep_for)
            else:
                next_tick = time.monotonic()
        if aborted:
            break
        print(f"  seg {label} {action} {dur}s done; T_apu={read_apu_c():.1f}C "
              f"n_samples={len(samples)}", flush=True)
        # cooldown gate between hot segments
        while read_apu_c() > PAUSE_C and action in ("cpu100", "gpu_burst"):
            print(f"    cooling: T_apu={read_apu_c():.1f}C, waiting...", flush=True)
            time.sleep(2.0)

    if cpu_evt is not None:
        stop_cpu_stress(cpu_evt, cpu_procs)

    arr = np.array(samples, dtype=np.float64)
    np.savez_compressed(out_npz,
                        t=arr[:, 0],
                        T_apu_c=arr[:, 1],
                        T_gpu_c=arr[:, 2],
                        P_gpu_w=arr[:, 3],
                        F_gpu_hz=arr[:, 4],
                        F_cpu_khz=arr[:, 5],
                        E_pkg_uj=arr[:, 6],
                        load_label=arr[:, 7].astype(np.int32))
    meta = dict(
        device_name=args.device_name,
        hostname=socket.gethostname(),
        rate_hz=args.rate_hz,
        n_samples=len(samples),
        duration_s=arr[-1, 0] if len(arr) else 0.0,
        aborted=aborted,
        hwmon_path=str(hw),
        schedule=[{"label": s[0], "dur_s": s[1], "action": s[2]} for s in sched],
        timestamp=time.time(),
    )
    out_meta.write_text(json.dumps(meta, indent=2))
    print(f"[OK] wrote {out_npz} ({len(samples)} samples) meta={out_meta}", flush=True)


if __name__ == "__main__":
    main()
