#!/usr/bin/env python3
"""ALL-32 identity campaign runner.

Sequentially runs 20 probes (Groups A/B/C/D from prompt). Each probe:
    1. Preflight (APU <= 50C, else SKIP)
    2. Warm regime indicator (we run in current ambient regime; record temp at start)
    3. 30 reps with mandatory cooling between bursts
    4. Save raw + summary JSON per (mechanism, host)

Persists state across restarts via state/all_32_state.json so a reboot or
killed run can resume.

Output: results/IDENTITY_BENCHMARK_2026-05-30/all_32/M{NN}_{host}.json
"""
from __future__ import annotations
import argparse, json, os, platform, struct, subprocess, sys, time
from pathlib import Path
import statistics

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from _safety import (read_apu_c, read_gpu_c, read_max_c, preflight, cool_down,
                     assert_under_ceiling, CeilingExceeded, get_strikes,
                     reset_strikes, INTER_PROBE_S, COOLDOWN_TARGET, CEILING_C)

REPO = HERE.parent.parent.parent          # AMD_gfx1151_energy/
RESULTS = REPO / "results" / "IDENTITY_BENCHMARK_2026-05-30" / "all_32"
LOGS    = REPO / "logs" / "all_32"
STATE   = HERE / "state" / "all_32_state.json"
KERNEL  = HERE / "kernels" / "isa_probes"
RESULTS.mkdir(parents=True, exist_ok=True)
LOGS.mkdir(parents=True, exist_ok=True)
STATE.parent.mkdir(parents=True, exist_ok=True)

HOST = platform.node().split(".")[0]

# Per-mechanism rep budget — kept low to stay <4s per kernel
ISA_REPS = {
    "M2":  4000, "M3": 2000, "M4": 2000, "M5": 2000, "M6": 2000,
    "M7":  2000, "M9": 500,  "M10": 500, "M11": 1000, "M17": 200,
    "M18": 1,    "M19": 1,   "M20": 2000, "M22": 2000, "M23": 2000, "M24": 200,
}

NREPS = 30   # samples per probe per regime

def load_state():
    if STATE.exists():
        return json.loads(STATE.read_text())
    return {"completed": [], "started_t": time.time()}

def save_state(st):
    STATE.write_text(json.dumps(st, indent=2))


def write_result(mech: str, data: dict):
    fp = RESULTS / f"{mech}_{HOST}.json"
    data["host"] = HOST
    data["mech"] = mech
    data["wrote_t"] = time.time()
    fp.write_text(json.dumps(data, indent=2, default=str))
    print(f"[result] wrote {fp}", flush=True)


def run_isa_kernel(mech: str, reps: int, seed: int) -> dict:
    """Returns dict with cyc, wall_us, payload bytes."""
    out = LOGS / f"{mech}_seed{seed}_{int(time.time()*1000)}.bin"
    env = os.environ.copy()
    env["HSA_OVERRIDE_GFX_VERSION"] = "11.0.0"
    t0 = time.time()
    r = subprocess.run([str(KERNEL), mech, str(reps), str(seed), str(out)],
                       env=env, capture_output=True, timeout=10)
    wall = time.time() - t0
    if r.returncode != 0 or not out.exists():
        return {"ok": False, "err": r.stderr.decode("utf8", "replace")[:200],
                "wall_s": wall}
    with open(out, "rb") as f:
        cyc, us, plen = struct.unpack("<QQI", f.read(20))
        payload = f.read(plen)
    out.unlink()        # don't accumulate bins
    return {"ok": True, "cyc": cyc, "wall_us": us, "payload_hex": payload.hex(),
            "wall_s": wall}


def probe_isa(mech: str) -> dict:
    """30 reps of the ISA kernel, with cooling every 5 bursts.

    Preflight gates the probe START only. Per-sample guard uses the hard ceiling
    (72C) — we DO want temps to drift up during the burst series, that's normal.
    """
    reps = ISA_REPS[mech]
    samples = []
    temps = []
    # Probe-start preflight
    if not preflight(mech):
        cool_down(COOLDOWN_TARGET, 120)
        if read_max_c() > 60:  # still too hot to even start
            return {"skip": f"preflight_fail_temp={read_max_c():.1f}", "samples": []}
    for i in range(NREPS):
        try:
            assert_under_ceiling()
        except CeilingExceeded as e:
            return {"aborted": True, "reason": str(e), "samples": samples}
        rec = run_isa_kernel(mech, reps, seed=42 + i)
        rec["temp_C"] = read_max_c()
        rec["i"] = i
        samples.append(rec)
        temps.append(rec["temp_C"])
        if (i + 1) % 5 == 0:
            cool_down(COOLDOWN_TARGET, 60)
    return {"samples": samples, "n_ok": sum(1 for s in samples if s.get("ok")),
            "temps": temps}


# ---- Group B: M15 thermal-induced freq variation ----
def probe_M15_thermal_freq() -> dict:
    """Warm the chip 40->62C via CPU stress only (no GPU MSR risk), sample
    GPU clock + APU temp at 5Hz, look for natural DVFS drops during heating."""
    # gentle CPU heater (numpy only — no torch needed)
    t_start = time.time()
    samples = []
    # use background CPU stress (small numpy matmul loop)
    import numpy as np
    A = np.random.randn(800, 800).astype(np.float32)
    print("[M15] starting gentle warm-up", flush=True)
    while True:
        cur = read_max_c()
        gpu_freq = -1
        try:
            gpu_freq = int(Path("/sys/class/hwmon/hwmon7/freq1_input").read_text()) / 1e6
        except Exception:
            pass
        try:
            power = int(Path("/sys/class/hwmon/hwmon7/power1_average").read_text()) / 1e6
        except Exception:
            power = -1
        samples.append({"t": time.time() - t_start, "temp_C": cur,
                        "gpu_freq_MHz": gpu_freq, "power_W": power})
        if cur > 68:    # safety: lower than ceiling, abort heating
            print(f"[M15] stop at {cur:.1f}C", flush=True)
            break
        if time.time() - t_start > 90:
            print("[M15] timeout", flush=True)
            break
        # one matmul burst (~0.2s)
        _ = A @ A.T
        time.sleep(0.05)
    # cool down
    cool_down(48.0, 120)
    return {"samples": samples, "n": len(samples)}


# ---- Group D actuator probes ----
def probe_M27_dpm_transition() -> dict:
    """Force DPM low->high->low, sample at 10Hz."""
    DPM = Path("/sys/class/drm/card1/device/power_dpm_force_performance_level")
    if not os.access(DPM, os.W_OK):
        return {"skip": "DPM not writable"}
    sequence = ["low", "high", "low", "auto"]
    samples = []
    t0 = time.time()
    try:
        for level in sequence:
            DPM.write_text(level)
            print(f"[M27] DPM={level}", flush=True)
            for _ in range(60):  # 6s at 10Hz per level
                t = read_max_c()
                if t > 68:
                    raise RuntimeError(f"thermal abort at {t:.1f}")
                try: freq = int(Path("/sys/class/hwmon/hwmon7/freq1_input").read_text())/1e6
                except: freq = -1
                try: pw = int(Path("/sys/class/hwmon/hwmon7/power1_average").read_text())/1e6
                except: pw = -1
                samples.append({"t": time.time()-t0, "level": level, "temp_C": t,
                                "gpu_freq_MHz": freq, "power_W": pw})
                time.sleep(0.1)
    finally:
        try: DPM.write_text("auto")
        except: pass
    return {"samples": samples}


def probe_M28_gfxoff_latency() -> dict:
    """Time the propagation of pp_features GFXOFF toggle."""
    pp = Path("/sys/class/drm/card1/device/pp_features")
    if not pp.exists() or not os.access(pp, os.W_OK):
        return {"skip": "pp_features unavailable or not writable"}
    samples = []
    try:
        original = pp.read_text().strip()
        # GFXOFF = bit 4 — toggle entire mask off then on isn't safe;
        # we just record current and refuse if not safe.
        # Honest record: report bitmap only.
        return {"skip": "no-toggle policy", "current": original}
    except Exception as e:
        return {"skip": f"read err {e}"}


def probe_M29_fan_curve() -> dict:
    """Sample power, temp at 1Hz for 60s while ramping load. Without fan RPM,
    record what response is visible at the hwmon level."""
    # gentle ramp
    import numpy as np
    samples = []
    sizes = [400, 600, 800, 1000, 800, 600, 400]
    t0 = time.time()
    for size in sizes:
        A = np.random.randn(size, size).astype(np.float32)
        # ~8s per stage but with thermal break checks
        s_t0 = time.time()
        while time.time() - s_t0 < 8.0:
            t = read_max_c()
            try: pw = int(Path("/sys/class/hwmon/hwmon7/power1_average").read_text())/1e6
            except: pw = -1
            try: freq = int(Path("/sys/class/hwmon/hwmon7/freq1_input").read_text())/1e6
            except: freq = -1
            samples.append({"t": time.time()-t0, "stage": size, "temp_C": t,
                            "power_W": pw, "gpu_freq_MHz": freq})
            if t > 68:
                cool_down(50, 90)
                return {"samples": samples, "early_abort_at_C": t}
            _ = A @ A.T
            time.sleep(0.5)
    cool_down(48, 120)
    return {"samples": samples}


def probe_M31_pll_settling() -> dict:
    """Force DPM step, then sample freq at max rate (~100Hz)."""
    DPM = Path("/sys/class/drm/card1/device/power_dpm_force_performance_level")
    if not os.access(DPM, os.W_OK):
        return {"skip": "DPM not writable"}
    samples = []
    t0 = time.time()
    try:
        DPM.write_text("low")
        time.sleep(0.5)
        DPM.write_text("high")
        for _ in range(500):  # ~5s at 100Hz
            try: freq = int(Path("/sys/class/hwmon/hwmon7/freq1_input").read_text())/1e6
            except: freq = -1
            samples.append({"t": time.time()-t0, "freq_MHz": freq, "temp_C": read_max_c()})
            time.sleep(0.01)
        DPM.write_text("auto")
    finally:
        try: DPM.write_text("auto")
        except: pass
    return {"samples": samples}


# Group C: M23 already handled by ISA kernel (stride-based). M24 same.
# But also do a python-level cache-timing probe as cross-validation.


PROBES_ORDER = [
    # Group A (ISA)
    "M2","M3","M4","M5","M6","M7","M9","M10","M11","M17",
    # Group C cache/mem
    "M18","M19","M20","M22","M23","M24",
    # Group B (safer thermal-induced freq)
    "M15",
    # Group D actuators
    "M27","M28","M29","M31",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", help="comma-separated list of probes to run", default="")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--reset-strikes", action="store_true")
    args = ap.parse_args()

    if args.reset_strikes:
        reset_strikes()
        print("[main] reset strikes", flush=True)

    st = load_state() if args.resume else {"completed": [], "started_t": time.time()}
    only = set(args.only.split(",")) if args.only else None
    todo = [m for m in PROBES_ORDER if (only is None or m in only)
                                       and m not in st["completed"]]

    print(f"[main] HOST={HOST} probes_to_run={todo} resumed={args.resume}",
          flush=True)

    for mech in todo:
        if get_strikes() >= 2:
            print(f"[main] ABORT — strikes={get_strikes()} >= 2", flush=True)
            break
        # universal preflight + 30s wait
        if mech != todo[0]:
            print(f"[main] cooldown {INTER_PROBE_S}s before {mech}", flush=True)
            cool_down(COOLDOWN_TARGET, 90)
            time.sleep(INTER_PROBE_S)
        if not preflight(mech):
            cool_down(COOLDOWN_TARGET, 120)
        print(f"[main] >>> probe {mech} apu={read_apu_c():.1f}C", flush=True)
        t0 = time.time()
        try:
            if mech in ISA_REPS:
                res = probe_isa(mech)
            elif mech == "M15": res = probe_M15_thermal_freq()
            elif mech == "M27": res = probe_M27_dpm_transition()
            elif mech == "M28": res = probe_M28_gfxoff_latency()
            elif mech == "M29": res = probe_M29_fan_curve()
            elif mech == "M31": res = probe_M31_pll_settling()
            else:
                res = {"skip": "no probe defined"}
        except CeilingExceeded as e:
            res = {"aborted": True, "reason": str(e)}
        except Exception as e:
            res = {"error": str(e)[:300]}
        wall = time.time() - t0
        res["wall_s_total"] = wall
        res["start_temp_C"] = read_max_c()
        write_result(mech, res)
        st["completed"].append(mech)
        save_state(st)
        print(f"[main] <<< probe {mech} done in {wall:.1f}s temp_now={read_max_c():.1f}C",
              flush=True)

    print(f"[main] DONE. completed={st['completed']} strikes={get_strikes()}",
          flush=True)


if __name__ == "__main__":
    main()
