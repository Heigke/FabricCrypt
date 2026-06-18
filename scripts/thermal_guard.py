"""Thermal-safety wrapper for heavy compute jobs.

Usage from another script:
    from scripts.thermal_guard import wait_if_hot, abort_if_critical
    for step in range(N):
        if step % 50 == 0: wait_if_hot()  # blocks if APU > 85°C until cool
        # ... heavy work ...

Or as a sidecar daemon:
    python scripts/thermal_guard.py --watch  # tails telemetry; SIGSTOP heavy
    procs if APU > 92°C, SIGCONT when < 70°C; logs every event.

Constants tuned for the HP gfx1151 laptop (ACPI trip = 99°C; below that
margin, work pauses).
"""
from __future__ import annotations
import argparse, json, os, signal, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LATEST = ROOT / "results/telemetry/latest.json"
GUARD_LOG = ROOT / "results/telemetry/guard.log"

WARN_C = 75.0      # warn + slow-down threshold
PAUSE_C = 80.0     # pause heavy procs (SIGSTOP) — far below ACPI 99°C trip
RESUME_C = 60.0    # resume threshold (tight hysteresis to avoid thrash)
EMERG_C = 90.0     # log emergency, still don't kill — but flag loudly


def read_max_temp():
    """Return max APU/CPU/edge temp from latest telemetry snapshot, or
    a /sys read fallback if telemetry not running."""
    try:
        if LATEST.exists() and (time.time() - LATEST.stat().st_mtime) < 30:
            d = json.loads(LATEST.read_text())
            return d.get("max_temp_C")
    except Exception:
        pass
    # Fallback: read thermal_zone0 directly
    try:
        return int(open("/sys/class/thermal/thermal_zone0/temp").read()) / 1000.0
    except Exception:
        return None


def log_event(msg):
    GUARD_LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(GUARD_LOG, "a") as f:
        f.write(f"{time.strftime('%Y-%m-%dT%H:%M:%S')} {msg}\n")


def wait_if_hot(threshold=WARN_C, timeout_s=300):
    """Block until APU temp drops below threshold-5°C, capped at timeout_s."""
    t0 = time.time()
    waited = False
    while True:
        T = read_max_temp()
        if T is None or T < threshold:
            if waited:
                log_event(f"resumed at T={T:.1f}°C (waited {time.time()-t0:.0f}s)")
            return T
        if not waited:
            log_event(f"WAIT_IF_HOT triggered at T={T:.1f}°C "
                      f"(threshold {threshold}°C)")
            print(f"[guard] hot ({T:.1f}°C) — waiting for cooldown ...",
                  flush=True)
            waited = True
        if time.time() - t0 > timeout_s:
            log_event(f"WAIT_IF_HOT timeout at T={T:.1f}°C")
            print(f"[guard] timeout — proceeding hot", flush=True)
            return T
        time.sleep(5)


def abort_if_critical(crit_C=PAUSE_C):
    T = read_max_temp()
    if T is not None and T >= crit_C:
        log_event(f"ABORT_IF_CRITICAL T={T:.1f}°C")
        raise SystemExit(f"[guard] critical T={T:.1f}°C — aborting")


# -- Watch mode: SIGSTOP runaway children of caller process group --
def watch(name_filter, interval=10):
    log_event(f"WATCH started filter={name_filter} interval={interval}s")
    paused_pids = set()
    while True:
        T = read_max_temp()
        if T is None:
            time.sleep(interval); continue
        # Find candidate procs
        try:
            import subprocess
            cp = subprocess.run(["pgrep", "-f", name_filter],
                                 capture_output=True, text=True, timeout=3)
            pids = [int(x) for x in cp.stdout.split() if x.isdigit()]
        except Exception:
            pids = []

        if T >= PAUSE_C and pids:
            for p in pids:
                if p in paused_pids: continue
                try:
                    os.kill(p, signal.SIGSTOP)
                    paused_pids.add(p)
                    log_event(f"SIGSTOP pid={p} T={T:.1f}°C")
                except Exception as e:
                    log_event(f"SIGSTOP fail pid={p}: {e}")
        elif T < RESUME_C and paused_pids:
            for p in list(paused_pids):
                try:
                    os.kill(p, signal.SIGCONT)
                    paused_pids.discard(p)
                    log_event(f"SIGCONT pid={p} T={T:.1f}°C")
                except Exception as e:
                    log_event(f"SIGCONT fail pid={p}: {e}")
                    paused_pids.discard(p)

        if T >= EMERG_C:
            log_event(f"EMERGENCY T={T:.1f}°C — paused {len(paused_pids)} procs")

        time.sleep(interval)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--watch", action="store_true",
                    help="run as sidecar, SIGSTOP heavy procs above PAUSE_C")
    ap.add_argument("--filter", default="python.*scripts/(z[0-9]+|demo_local|nsram_)",
                    help="process name regex to watch")
    ap.add_argument("--interval", type=int, default=10)
    ap.add_argument("--probe", action="store_true",
                    help="just print current temp and exit")
    args = ap.parse_args()

    if args.probe:
        T = read_max_temp()
        print(f"max_temp_C = {T}")
    elif args.watch:
        watch(args.filter, args.interval)
    else:
        T = read_max_temp()
        print(f"current max_temp_C = {T}")
        print(f"thresholds: WARN={WARN_C} PAUSE={PAUSE_C} RESUME={RESUME_C} "
              f"EMERG={EMERG_C}")
