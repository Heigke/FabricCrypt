#!/usr/bin/env python3
"""F4 — Same-machine reboot test.

Multi-invocation: each call runs ONE measurement cycle (collect fresh streams,
run pipeline, record z), increments state['f4_cycle'], and if more cycles
needed schedules a reboot. After N cycles, computes within-machine variance
of z and compares to between-machine z (5.74).

state['f4_cycle'] schedule:
  cycle 0 (this run): baseline measurement, no reboot yet
  cycle 1..3: after each reboot, measure again
  cycle 4: synthesize variance

If within-machine z stddev > between-machine effect → silicon claim DIES.

Reboot scheduling: F4 writes a sentinel file `state/needs_reboot.flag`. The
caller (resume script or human) acts on it. We do NOT call `sudo reboot`
ourselves — the systemd resume service or an explicit cron will do it.

Output: results/.../falsify/F4_reboot_test.json (accumulates per cycle).
"""
from __future__ import annotations
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path("/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "constitutive"))
sys.path.insert(0, str(HERE.parent / "attack_1_3"))
sys.path.insert(0, str(HERE))

from reservoir import Reservoir, ReservoirCfg  # type: ignore
from A3_heavy_tail_transplant import (  # type: ignore
    HeavyTailSubstrate, GaussianMatchedHT, ShuffleHT, load_streams,
    N_RES, SUB_DIM, WASHOUT, T_TRAIN, T_TEST, HORIZON,
)
from A13_cross import train_dual, task_nrmse, narma10  # type: ignore
from F2_stale_data import collect_fresh_ikaros, run_pipeline  # type: ignore

STATE = ROOT / "state" / "falsify_state.json"
OUT = ROOT / "results" / "IDENTITY_BENCHMARK_2026-05-30" / "falsify" / "F4_reboot_test.json"
REBOOT_FLAG = ROOT / "state" / "needs_reboot.flag"
NEEDED_CYCLES = 4  # cycle 0 + 3 reboots
LAM = 1.0
N_SEEDS = int(os.environ.get("N_SEEDS", "20"))


def load_acc() -> dict:
    if OUT.exists():
        return json.loads(OUT.read_text())
    return {"test": "F4_reboot_test", "cycles": []}


def save_acc(d):
    OUT.write_text(json.dumps(d, indent=2))


def main():
    state = json.loads(STATE.read_text()) if STATE.exists() else {}
    cycle = int(state.get("f4_cycle", 0))
    acc = load_acc()
    print(f"[F4] cycle={cycle}/{NEEDED_CYCLES} apu={open('/sys/class/thermal/thermal_zone0/temp').read().strip()}/1000C",
          flush=True)
    t0 = time.time()
    # equilibrate
    if cycle > 0:
        from A3_heavy_tail_collect import wait_cool  # type: ignore
        wait_cool(target=45.0, timeout=300.0)
    streams_d = load_streams("daedalus")
    streams_fresh = collect_fresh_ikaros()
    result = run_pipeline(streams_fresh, streams_d, f"f4_cycle{cycle}")
    boot_time_s = None
    try:
        with open("/proc/uptime") as f:
            boot_time_s = float(f.read().split()[0])
    except Exception:
        pass
    acc["cycles"].append({
        "cycle": cycle,
        "z_hw_vs_sw": result["z_hw_vs_sw"],
        "delta_hw": result["delta_hw"],
        "delta_sw": result["delta_sw"],
        "uptime_s": boot_time_s,
        "wall_s": time.time() - t0,
        "ts": time.time(),
    })
    save_acc(acc)
    # Decide next action
    if cycle + 1 < NEEDED_CYCLES:
        state["f4_cycle"] = cycle + 1
        STATE.write_text(json.dumps(state, indent=2))
        REBOOT_FLAG.write_text(json.dumps({"requested_at": time.time(),
                                            "next_cycle": cycle + 1}))
        print(f"[F4] cycle {cycle} done, REBOOT requested for cycle {cycle+1}",
              flush=True)
        # Don't crash pipeline — F4 returns success even though more cycles pending
        # The driver / human watches REBOOT_FLAG.
    else:
        # Final cycle: synthesise
        zs = np.array([c["z_hw_vs_sw"] for c in acc["cycles"]])
        within_mean = float(np.mean(zs))
        within_std = float(np.std(zs))
        between_z = 5.74  # the headline finding to beat
        # Verdict: if within-machine z is comparable to between-machine z, silicon
        # claim is dead. We want within << between, ideally within < 2.0 and stable.
        if within_mean < 2.0:
            verdict = "WITHIN_COLLAPSED"  # reboot kills z entirely → moment-bound
        elif within_std > 1.5 and within_mean > 3.0:
            verdict = "WITHIN_UNSTABLE"
        elif within_mean > 4.0 and within_std < 1.0:
            verdict = "SILICON_STABLE_ACROSS_REBOOTS"
        else:
            verdict = "PARTIAL_SURVIVAL"
        acc["synthesis"] = {
            "within_mean_z": within_mean,
            "within_std_z": within_std,
            "between_z": between_z,
            "verdict": verdict,
        }
        save_acc(acc)
        state["f4_cycle"] = NEEDED_CYCLES
        state["needs_reboot"] = False
        if REBOOT_FLAG.exists():
            REBOOT_FLAG.unlink()
        STATE.write_text(json.dumps(state, indent=2))
        print(f"[F4] DONE cycles={NEEDED_CYCLES} within=({within_mean:.2f}±{within_std:.2f}) "
              f"between={between_z:.2f} → {verdict}", flush=True)


if __name__ == "__main__":
    main()
