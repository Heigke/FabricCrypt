#!/usr/bin/env python3
"""FC-A1 temporal-stability loop for the 466-dim FabricCrypt signature.

Every TICK_S seconds (default 30 min) this captures a full FabricCrypt
signature on the local host and appends one record to

    results/FABRICCRYPT/a1_temporal/{host}_{iso_ts}.json
    results/FABRICCRYPT/a1_temporal/{host}_{iso_ts}.npz   # vec

Thermal-safe: pauses if APU > 65 C (PAUSE_C), resumes < 50 C (RESUME_C),
aborts the current tick at 75 C (ABORT_C). Graceful SIGTERM/SIGINT.

The signature is assembled from the same family-set that defines the v3
466-dim signature in paper_drafts/launch_kit/arxiv_submission/fabriccrypt.tex:

  - Block A: signature_v2 (Phase 13)               -> 290 dims (5 HAL-bypass)
  - Block B: Phase 19 cross-host KS-verified       -> 70  dims  (s4 + s6 + s9)
  - Block C: Phase 22 board-level deterministic    -> up to 98 dims
                                                     (s20..s26)
  - Block D: S27 HPET/RTC stochastic drift         -> 12  dims
  -------------------------------------------------------------
  Total target: 470 dims (the paper's 466 figure rounds to S27=8; we
  keep the *actual* DIM constants the source modules report, so the
  per-tick dimensionality is captured in the JSON header).

The number of available dims may differ on the two chassi (e.g. some
sysfs paths missing). The vector dimensionality is fixed per host on
first tick (signal-by-signal DIM constants), and zero-padded if a
sub-signal fails so that within-host vectors remain alignable.

CLI:
    python3 run_loop.py [--tick-min 30] [--out RESULT_DIR] [--once]

Designed to be launched in tmux/nohup. Logs to stdout (line-buffered).
"""
from __future__ import annotations
import argparse
import json
import os
import signal
import socket
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent.parent       # AMD_gfx1151_energy/
IB = REPO / "scripts" / "identity_benchmark"

# ------------------- thermal --------------------
THERMAL_ZONE = Path("/sys/class/thermal/thermal_zone0/temp")
PAUSE_C = 65.0
RESUME_C = 50.0
ABORT_C = 75.0


def temp_c() -> float:
    try:
        return int(THERMAL_ZONE.read_text().strip()) / 1000.0
    except Exception:
        return -1.0


def thermal_wait_cool(target_c: float = RESUME_C, timeout_s: float = 600.0) -> float:
    """Block until APU <= target_c, or timeout. Return final temp."""
    t0 = time.time()
    while True:
        t = temp_c()
        if t < 0:
            return t
        if t <= target_c:
            return t
        if t >= ABORT_C:
            raise RuntimeError(f"ABORT thermal {t:.1f}C >= {ABORT_C}")
        if time.time() - t0 > timeout_s:
            return t
        time.sleep(5.0)


def thermal_guard():
    t = temp_c()
    if t >= ABORT_C:
        raise RuntimeError(f"ABORT thermal {t:.1f}C")
    if t >= PAUSE_C:
        log(f"[thermal] {t:.1f}C >= {PAUSE_C} -> wait_cool to {RESUME_C}")
        thermal_wait_cool(target_c=RESUME_C, timeout_s=600.0)


# ------------------- logging / state --------------------
_STOP = False


def _sig_handler(signum, frame):
    global _STOP
    _STOP = True
    log(f"[signal] received {signum} -> graceful shutdown after current tick")


def log(msg: str) -> None:
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    print(f"{ts} {msg}", flush=True)


def hostname() -> str:
    try:
        return Path("/etc/hostname").read_text().strip()
    except Exception:
        return socket.gethostname()


# ------------------- capture --------------------
# Block A: re-use embodiment13.signature_v2.extract_one  (290 dims, ~1.3s)
# Block B: Phase 19 light signals (s4, s6, s9) -- s9 is the longest (~10-20s)
# Block C: Phase 22 board-level (s20..s26) -- all sysfs reads, fast
# Block D: s27 short variant (5 s) for the stochastic drift block

def _import_signal(pkg: str, mod_name: str):
    """Import scripts/identity_benchmark/{pkg}/{mod_name}.py."""
    import importlib.util
    p = IB / pkg / f"{mod_name}.py"
    if not p.exists():
        return None
    spec = importlib.util.spec_from_file_location(f"_fc_a1_{pkg}_{mod_name}", p)
    m = importlib.util.module_from_spec(spec)
    # Make sibling imports (common19, common22, common13) work for the module.
    sys.path.insert(0, str(IB / pkg))
    try:
        spec.loader.exec_module(m)
    finally:
        sys.path.pop(0)
    return m


def _safe_capture(fn, *args, **kw):
    """Call fn(); return (feat_vec_or_None, err_str_or_None)."""
    try:
        out = fn(*args, **kw)
        return out, None
    except SystemExit as e:
        return None, f"SystemExit:{e}"
    except Exception as e:
        return None, f"{type(e).__name__}:{e}"


def _build_blockA():
    """Re-create signature_v2.extract_one() inline; return DIM=290 vec.

    We do NOT call signature_v2.main() because that runs `reps` and writes
    its own files. We re-use extract_one() and the compiled C binaries.
    """
    mod = _import_signal("embodiment13", "signature_v2")
    if mod is None:
        return None, 290, "signature_v2 missing"
    em13_dir = IB / "embodiment13"
    tsc_src = em13_dir / "tsc_inter_core.c"
    tsc_bin = em13_dir / "tsc_inter_core"
    cl_src = em13_dir / "cacheline_pingpong.c"
    cl_bin = em13_dir / "cacheline_pingpong"
    if not tsc_bin.exists() and tsc_src.exists():
        try:
            mod.compile_c(str(tsc_src), str(tsc_bin))
        except Exception as e:
            return None, 290, f"compile tsc: {e}"
    if not cl_bin.exists() and cl_src.exists():
        try:
            mod.compile_c(str(cl_src), str(cl_bin))
        except Exception as e:
            return None, 290, f"compile cl: {e}"
    try:
        v = mod.extract_one(str(tsc_bin), str(cl_bin))
        return np.asarray(v, dtype=np.float64), 290, None
    except Exception as e:
        return None, 290, f"extract_one: {e}"


def _call_signal_run(pkg: str, mod_name: str, dim_hint: int):
    """Call run(reps=1, out_dir=tmp) and read back the npz vec[0]."""
    import tempfile
    tmp = tempfile.mkdtemp(prefix="fc_a1_")
    try:
        mod = _import_signal(pkg, mod_name)
        if mod is None:
            return None, dim_hint, "module missing"
        DIM = int(getattr(mod, "DIM", dim_hint))
        try:
            mod.run(reps=1, out_dir=tmp)
        except SystemExit as e:
            return None, DIM, f"SystemExit:{e}"
        except Exception as e:
            return None, DIM, f"run: {type(e).__name__}:{e}"
        host = hostname()
        # find npz produced
        candidates = list(Path(tmp).glob(f"{host}_*.npz"))
        if not candidates:
            candidates = list(Path(tmp).glob("*.npz"))
        if not candidates:
            return None, DIM, "no npz produced"
        npz = np.load(candidates[0])
        if "vec" not in npz.files:
            return None, DIM, f"keys={npz.files}"
        v = npz["vec"]
        if v.ndim == 2 and v.shape[0] >= 1:
            return np.asarray(v[0], dtype=np.float64), DIM, None
        return np.asarray(v.ravel(), dtype=np.float64), DIM, None
    finally:
        try:
            import shutil
            shutil.rmtree(tmp, ignore_errors=True)
        except Exception:
            pass


# Order chosen for thermal smoothness: light sysfs reads first, computational
# bursts last.
SIGNALS_PHASE22 = [
    ("embodiment22", "s20_acpi_pci_topology", 16),
    ("embodiment22", "s21_pcie_link_degradation", 16),
    ("embodiment22", "s22_usb_descriptor", 16),
    ("embodiment22", "s23_dmi_smbios", 16),
    ("embodiment22", "s24_boot_timing", 18),
    ("embodiment22", "s25_ucsi_power", 16),
    ("embodiment22", "s26_umr_safe_reads", 16),
]
SIGNALS_PHASE19 = [
    ("embodiment19", "s4_gpu_clock_jitter", 16),  # ~11 s idle sample
    ("embodiment19", "s6_thermal_spread", 30),    # ~11 s
    ("embodiment19", "s9_jacobian_dynamics", 20), # heavier -- last
]
SIGNAL_S27 = ("embodiment22", "s27_hpet_rtc_drift", 12)


def capture_tick(host: str) -> dict:
    """One full 466-dim capture. Returns {vec, dims, errors}."""
    rec = {
        "host": host,
        "ts_iso": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "ts_unix": time.time(),
        "temp_start_c": temp_c(),
        "blocks": {},
        "errors": {},
        "dim_total": 0,
    }
    parts: list[np.ndarray] = []

    # Block A
    thermal_guard()
    log("[capture] blockA signature_v2 (290)")
    vA, dimA, errA = _build_blockA()
    if vA is None:
        vA = np.zeros(dimA, dtype=np.float64)
        rec["errors"]["blockA"] = errA
    parts.append(vA)
    rec["blocks"]["A_signature_v2"] = {"dim": dimA, "ok": errA is None,
                                       "err": errA}

    # Block B (Phase19)
    for (pkg, name, hint) in SIGNALS_PHASE19:
        thermal_guard()
        log(f"[capture] blockB {name}")
        v, dim, err = _call_signal_run(pkg, name, hint)
        if v is None or v.shape[0] != dim:
            v = np.zeros(dim, dtype=np.float64)
            rec["errors"][name] = err or f"dim mismatch got {None if v is None else v.shape}"
        parts.append(v)
        rec["blocks"][f"B_{name}"] = {"dim": dim, "ok": err is None,
                                      "err": err}

    # Block C (Phase22 board-level)
    for (pkg, name, hint) in SIGNALS_PHASE22:
        thermal_guard()
        log(f"[capture] blockC {name}")
        v, dim, err = _call_signal_run(pkg, name, hint)
        if v is None or v.shape[0] != dim:
            v = np.zeros(dim, dtype=np.float64)
            rec["errors"][name] = err or f"dim mismatch got {None if v is None else v.shape}"
        parts.append(v)
        rec["blocks"][f"C_{name}"] = {"dim": dim, "ok": err is None,
                                      "err": err}

    # Block D (S27)
    thermal_guard()
    pkg, name, hint = SIGNAL_S27
    log(f"[capture] blockD {name}")
    v, dim, err = _call_signal_run(pkg, name, hint)
    if v is None or v.shape[0] != dim:
        v = np.zeros(dim, dtype=np.float64)
        rec["errors"][name] = err or f"dim mismatch got {None if v is None else v.shape}"
    parts.append(v)
    rec["blocks"][f"D_{name}"] = {"dim": dim, "ok": err is None, "err": err}

    vec = np.concatenate(parts, axis=0).astype(np.float64, copy=False)
    rec["dim_total"] = int(vec.shape[0])
    rec["temp_end_c"] = temp_c()
    rec["seconds"] = time.time() - rec["ts_unix"]
    return rec, vec


# ------------------- loop --------------------

def write_tick(out_dir: Path, rec: dict, vec: np.ndarray) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    host = rec["host"]
    stamp = rec["ts_iso"].replace(":", "").replace("-", "")
    base = out_dir / f"{host}_{stamp}"
    np.savez_compressed(str(base) + ".npz", vec=vec,
                        dim=rec["dim_total"], host=host,
                        ts_unix=rec["ts_unix"], ts_iso=rec["ts_iso"])
    with open(str(base) + ".json", "w") as f:
        json.dump(rec, f, indent=2, default=str)
    # also append to an index
    idx = out_dir / f"{host}_index.jsonl"
    with open(idx, "a") as f:
        idx_rec = {"ts_iso": rec["ts_iso"], "ts_unix": rec["ts_unix"],
                   "dim_total": rec["dim_total"],
                   "temp_start_c": rec["temp_start_c"],
                   "temp_end_c": rec["temp_end_c"],
                   "seconds": rec["seconds"],
                   "errors": rec["errors"]}
        f.write(json.dumps(idx_rec, default=str) + "\n")
    return base


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tick-min", type=float, default=30.0,
                    help="Tick period in minutes (default 30)")
    ap.add_argument("--out", type=str,
                    default=str(REPO / "results" / "FABRICCRYPT" / "a1_temporal"))
    ap.add_argument("--once", action="store_true",
                    help="Capture once and exit (smoke test)")
    args = ap.parse_args()

    signal.signal(signal.SIGTERM, _sig_handler)
    signal.signal(signal.SIGINT, _sig_handler)

    out_dir = Path(args.out)
    host = hostname()
    log(f"[fc_a1] start host={host} tick={args.tick_min}min out={out_dir} pid={os.getpid()}")
    log(f"[fc_a1] thermal: PAUSE={PAUSE_C} RESUME={RESUME_C} ABORT={ABORT_C}")

    tick_s = args.tick_min * 60.0
    next_t = time.time()
    while not _STOP:
        try:
            thermal_wait_cool(target_c=RESUME_C, timeout_s=900.0)
            log("[fc_a1] capture begin")
            t0 = time.time()
            rec, vec = capture_tick(host)
            base = write_tick(out_dir, rec, vec)
            log(f"[fc_a1] capture done dim={rec['dim_total']} "
                f"errs={len(rec['errors'])} secs={time.time()-t0:.1f} -> {base.name}")
        except RuntimeError as e:
            log(f"[fc_a1] thermal-abort {e} -- skip tick")
        except Exception as e:
            log(f"[fc_a1] ERROR {type(e).__name__}: {e}")
            log(traceback.format_exc())

        if args.once:
            log("[fc_a1] --once set -> exit")
            return
        next_t += tick_s
        sleep_s = next_t - time.time()
        if sleep_s < 0:
            log(f"[fc_a1] tick overran by {-sleep_s:.0f}s -- realign")
            next_t = time.time() + tick_s
            sleep_s = tick_s
        log(f"[fc_a1] sleep {sleep_s:.0f}s until next tick")
        # interruptible sleep
        end = time.time() + sleep_s
        while time.time() < end and not _STOP:
            time.sleep(min(5.0, end - time.time()))

    log("[fc_a1] stopped cleanly")


if __name__ == "__main__":
    main()
