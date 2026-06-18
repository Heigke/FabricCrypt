#!/usr/bin/env python3
"""probeC_vth_sweep.py — threshold-crossing voltage sweep PUF.

We don't have direct V_th access. Two channels we DO have:
  (i)  power_dpm_force_performance_level: low/auto/high — coarse DVFS clamp.
  (ii) pp_dpm_sclk: per-DPM-state SCLK levels readable; on this kernel they
       are not user-writable, so we observe natural DVFS scaling.

This probe runs a fixed deterministic compute (probeC_kernel) at each DPM
setting (low, auto, high) and counts per-CU "flip-bits" — bits in the FMA-LSB
output that DIFFER between high and low DPM. Per-CU flip-pattern is a silicon
proxy for ΔVth: CUs whose transistors are slightly slower will produce more
flips when clocked at the same voltage step.

To stay safe (no GPU hang risk) we ONLY use the official DPM levels — no
voltage glitching. The risk is that DPM transitions are too coarse to resolve
per-CU thresholds. We mitigate by averaging many reps per DPM level.

Output: results dict with per-DPM (mean, std) FMA bit-pattern per CU.
"""
from __future__ import annotations
import argparse
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from _thermal import (read_temp, wait_cool, inter_burst_sleep, ThermalMonitor,
                      MAX_BURST_S, COOL_TARGET_C, HOSTNAME)

PROBE_BIN = HERE / "probeA_lds_startup"   # reuse same kernel; FMA-LSB is the bit-stream
DVFS_PATH = Path("/sys/class/drm/card0/device/power_dpm_force_performance_level")

PAUSE_AT_C = 72.0
SAFE_DPM_LEVELS = ["low", "auto", "high"]   # never write anything else
DISABLE_HOSTS_DEFAULT = {"ikaros"}           # high-DPM is what killed ikaros


def set_dpm(level: str) -> bool:
    if level not in SAFE_DPM_LEVELS:
        return False
    try:
        DVFS_PATH.write_text(level)
        return True
    except Exception:
        return False


def parse_probeA_bin(path: Path) -> np.ndarray:
    raw = np.fromfile(path, dtype=np.uint32)
    magic, reps, wpr, _ = raw[:4]
    assert magic == 0x4C445331, f"bad magic {hex(magic)}"
    pl = raw[4:].reshape(int(reps), int(wpr))
    # second half (LDS_WORDS..) is FMA-LSB
    return pl[:, 256:]


def run_at_dpm(level: str, reps: int, out_bin: Path) -> dict:
    set_dpm(level)
    time.sleep(2.0)
    if read_temp() > PAUSE_AT_C:
        wait_cool()
    env = os.environ.copy()
    env.pop("HSA_OVERRIDE_GFX_VERSION", None)
    t0 = time.time()
    r = subprocess.run([str(PROBE_BIN), str(reps), str(out_bin)],
                       env=env, capture_output=True, text=True, timeout=300)
    dur = time.time() - t0
    if r.returncode != 0:
        return {"ok": False, "err": r.stderr, "out": r.stdout}
    fma = parse_probeA_bin(out_bin)
    return {
        "ok": True,
        "level": level,
        "reps": reps,
        "duration_s": dur,
        "fma_mean": fma.mean(axis=0).tolist(),
        "fma_std":  fma.std(axis=0).tolist(),
        "fma_first_rep": fma[0].tolist(),
        "temp_after": read_temp(),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device-name", required=True)
    ap.add_argument("--reps", type=int, default=300)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--force", action="store_true",
                    help="override host disable-list (default off on ikaros)")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    if HOSTNAME in DISABLE_HOSTS_DEFAULT and not args.force:
        msg = {"device": args.device_name, "ts": time.time(),
               "skipped": True, "reason": f"host {HOSTNAME} disabled (thermal safety)"}
        (args.out_dir / "probeC_results.json").write_text(json.dumps(msg, indent=2))
        print(f"[probeC] SKIPPED on {HOSTNAME} (use --force to override)")
        return

    results = {"device": args.device_name, "ts": time.time(), "per_level": {}}
    for lvl in SAFE_DPM_LEVELS:
        binp = args.out_dir / f"probeC_fma_{lvl}.bin"
        print(f"[probeC] {lvl} -> {binp}")
        res = run_at_dpm(lvl, args.reps, binp)
        results["per_level"][lvl] = res
        # cool between regimes
        wait_cool()

    # always end on auto
    set_dpm("auto")

    # compute per-lane flip count between low vs high
    if (results["per_level"]["low"].get("ok") and
        results["per_level"]["high"].get("ok")):
        low_mean  = np.array(results["per_level"]["low"]["fma_mean"])
        high_mean = np.array(results["per_level"]["high"]["fma_mean"])
        # convert mean to bit at MSB of low-16: a "flip" is sign of diff > 1.
        flip = (np.abs(low_mean - high_mean) > 1.0).astype(int)
        results["flip_lanes_low_vs_high"] = flip.tolist()
        results["flip_count_low_vs_high"] = int(flip.sum())

    out_json = args.out_dir / "probeC_results.json"
    out_json.write_text(json.dumps(results, indent=2))
    print(f"[probeC] wrote {out_json}")


if __name__ == "__main__":
    main()
