#!/usr/bin/env python3
"""Falsification pipeline driver.

Reads state/falsify_state.json, runs the next pending test, updates state.
Designed to be re-entrant: if it crashes mid-test, next invocation re-runs
that test. If it succeeds, advances pointer.

Tests in order: F1, F2, F3, [F4 only if F1+F2+F3 all keep z>2.0].

Usage:
    venv/bin/python scripts/identity_benchmark/falsify/run_pipeline.py [--next-only]

  --next-only : run exactly one pending test then exit (used by resume.sh)
  (default)   : keep running tests until none pending OR F4 reached
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

ROOT = Path("/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy")
STATE = ROOT / "state" / "falsify_state.json"
LOGS = ROOT / "logs" / "falsify"
RESULTS = ROOT / "results" / "IDENTITY_BENCHMARK_2026-05-30" / "falsify"
PY = ROOT / "venv" / "bin" / "python"

LOGS.mkdir(parents=True, exist_ok=True)
RESULTS.mkdir(parents=True, exist_ok=True)
STATE.parent.mkdir(parents=True, exist_ok=True)

TESTS = [
    ("F1", "scripts/identity_benchmark/falsify/F1_tails_only_swap.py", "F1_tails_only_swap.json"),
    ("F2", "scripts/identity_benchmark/falsify/F2_stale_data.py",       "F2_stale_data.json"),
    ("F3", "scripts/identity_benchmark/falsify/F3_independent_reimpl.py","F3_independent_reimpl.json"),
    ("F4", "scripts/identity_benchmark/falsify/F4_reboot_test.py",      "F4_reboot_test.json"),
]


def read_apu_temp() -> float:
    try:
        return int(Path("/sys/class/thermal/thermal_zone0/temp").read_text().strip()) / 1000.0
    except Exception:
        return -1.0


def wait_cool(target: float = 55.0, timeout: float = 180.0) -> float:
    t0 = time.time()
    while time.time() - t0 < timeout:
        T = read_apu_temp()
        if T > 0 and T <= target:
            return T
        time.sleep(5.0)
    return read_apu_temp()


def load_state() -> dict:
    if STATE.exists():
        return json.loads(STATE.read_text())
    return {
        "claude_session": "83353dd8-59e1-4c51-a16d-0c4cceb1d1b4",
        "host": socket.gethostname(),
        "started_at": time.time(),
        "completed": [],
        "results": {},
        "next": "F1",
        "needs_reboot": False,
        "f4_cycle": 0,
    }


def save_state(state: dict) -> None:
    STATE.write_text(json.dumps(state, indent=2))


def constitutive_z_from_result(test: str, path: Path) -> float | None:
    try:
        d = json.loads(path.read_text())
    except Exception:
        return None
    if test == "F2":
        return d.get("fresh", {}).get("z_hw_vs_sw")
    return d.get("z_hw_vs_sw")


def should_run_f4(state: dict) -> bool:
    for tname in ("F1", "F2", "F3"):
        rp = RESULTS / dict((n, p) for n, _, p in TESTS)[tname]
        z = constitutive_z_from_result(tname, rp)
        if z is None or z < 2.0:
            return False
    return True


def run_one(test_name: str, script_rel: str, out_name: str, state: dict) -> bool:
    script = ROOT / script_rel
    out_path = RESULTS / out_name
    log_path = LOGS / f"{test_name}_{int(time.time())}.log"
    print(f"[pipeline] starting {test_name}: {script_rel}", flush=True)
    T = read_apu_temp()
    print(f"[pipeline] apu_temp={T:.1f}C", flush=True)
    if T > 70.0:
        print(f"[pipeline] HOT ({T}C), cooling first…", flush=True)
        wait_cool(target=55.0, timeout=240.0)
    env = os.environ.copy()
    env["HSA_OVERRIDE_GFX_VERSION"] = "11.0.0"
    env.setdefault("N_SEEDS", "30")
    t0 = time.time()
    with open(log_path, "w") as lf:
        proc = subprocess.run([str(PY), str(script)], cwd=str(ROOT), env=env,
                              stdout=lf, stderr=subprocess.STDOUT, timeout=3600)
    wall = time.time() - t0
    ok = proc.returncode == 0 and out_path.exists()
    print(f"[pipeline] {test_name} done rc={proc.returncode} wall={wall:.1f}s "
          f"out_exists={out_path.exists()} log={log_path}", flush=True)
    if ok:
        z = constitutive_z_from_result(test_name, out_path)
        state["results"][test_name] = {"z": z, "log": str(log_path),
                                       "result_path": str(out_path)}
        state["completed"].append(test_name)
    return ok


def advance_next(state: dict) -> None:
    completed = set(state["completed"])
    for tname, _, _ in TESTS:
        if tname not in completed:
            if tname == "F4" and not should_run_f4(state):
                state["next"] = None
                return
            state["next"] = tname
            return
    state["next"] = None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--next-only", action="store_true")
    args = ap.parse_args()

    state = load_state()
    save_state(state)
    while True:
        advance_next(state)
        save_state(state)
        nxt = state["next"]
        if nxt is None:
            print("[pipeline] no more tests pending — DONE", flush=True)
            return
        spec = dict((n, (s, o)) for n, s, o in TESTS)
        script_rel, out_name = spec[nxt]
        if nxt == "F4":
            # Special: F4 wants reboot — defer to F4 script which marks state
            state["needs_reboot"] = True
            save_state(state)
        ok = run_one(nxt, script_rel, out_name, state)
        save_state(state)
        if not ok:
            print(f"[pipeline] {nxt} FAILED — leaving in pending state for retry",
                  flush=True)
            sys.exit(1)
        if args.next_only:
            return


if __name__ == "__main__":
    main()
