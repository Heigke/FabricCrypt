"""H7 per-core Fmax / CPPC die-key enrollment — the one die-specific AND reproducible AND on-die channel.

Red-team + O108 verdict: räkna-unikt via PDN u·v is ~a category error at kHz (board/firmware-set). The only
on-die, reproducible, die-specific channel left is per-core speed-bin residual: CPPC capability + achieved Fmax /
delivered-perf under a fixed micro-load per core. This is the SAME physical source as our working identity channel.

Enrolls a per-core vector K times: static CPPC fields + dynamic delivered-perf (from feedback_ctrs delta) and
achieved frequency under a fixed single-core busy-load. Output -> reproducibility (intra cosine across K) now;
die-specificity (vs daedalus) after both enrolled. Thermally safe: ONE core lightly loaded at a time, brief.
Env: K (runs, default 8), LOAD_S (per-core load seconds, default 0.25), RUNTAG. Root not required.
"""
from __future__ import annotations
import os, sys, time, json, socket
from pathlib import Path
import numpy as np

HOST = socket.gethostname()
OUT = Path(__file__).resolve().parents[2] / "results/IDENTITY_H7_2026-06-09"
NCPU = os.cpu_count() or 1
K = int(os.environ.get("K", "8"))
LOAD_S = float(os.environ.get("LOAD_S", "0.25"))
RUNTAG = os.environ.get("RUNTAG", "r1")
ZONE_T = Path("/sys/class/thermal/thermal_zone0/temp")
CPPC = "/sys/devices/system/cpu/cpu{}/acpi_cppc/{}"
FREQ = "/sys/devices/system/cpu/cpu{}/cpufreq/scaling_cur_freq"
CPPC_FIELDS = ["highest_perf", "nominal_perf", "guaranteed_perf", "lowest_nonlinear_perf", "reference_perf"]


def rd(p, default=np.nan):
    try: return float(Path(p).read_text().strip())
    except Exception: return default


def temp_c():
    try: return int(ZONE_T.read_text())/1000.0
    except Exception: return 0.0


def read_fb(core):
    # feedback_ctrs: "ref:<n> del:<n>" — delivered/reference perf counters
    try:
        txt = Path(CPPC.format(core, "feedback_ctrs")).read_text()
        ref = del_ = np.nan
        for tok in txt.replace(":", " ").split():
            pass
        # format is like "ref:12345 del:6789"
        parts = dict(p.split(":") for p in txt.split() if ":" in p)
        ref = float(parts.get("ref", "nan")); del_ = float(parts.get("del", "nan"))
        return ref, del_
    except Exception:
        return np.nan, np.nan


def busy(load_s):
    t0 = time.time(); x = 0.0
    while time.time()-t0 < load_s:
        for _ in range(20000): x = x*1.0000001 + 1.0
    return x


def enroll_core(core):
    cppc = np.array([rd(CPPC.format(core, f)) for f in CPPC_FIELDS])
    try: os.sched_setaffinity(0, {core})
    except Exception: pass
    ref0, del0 = read_fb(core)
    freqs = []
    t0 = time.time()
    while time.time()-t0 < LOAD_S:
        busy(0.02)
        f = rd(FREQ.format(core))
        if not np.isnan(f): freqs.append(f)
    ref1, del1 = read_fb(core)
    delivered = (del1-del0)/((ref1-ref0)+1e-9) if not np.isnan(ref0) else np.nan  # delivered/reference ratio
    fr = np.array(freqs) if freqs else np.array([np.nan])
    return np.concatenate([cppc, [np.nanmean(fr), np.nanmax(fr), np.nanstd(fr), delivered]])


def main():
    try: os.sched_setaffinity(0, set(range(NCPU)))
    except Exception: pass
    print(f"[{HOST}] Fmax enroll NCPU={NCPU} K={K} LOAD_S={LOAD_S} tag={RUNTAG} temp={temp_c():.0f}C", flush=True)
    feat_dim = len(CPPC_FIELDS) + 4
    runs = np.zeros((K, NCPU, feat_dim))
    for k in range(K):
        for c in range(NCPU):
            runs[k, c] = enroll_core(c)
        try: os.sched_setaffinity(0, set(range(NCPU)))
        except Exception: pass
        print(f"  run {k+1}/{K} done temp={temp_c():.0f}C "
              f"|highest_perf spread={np.nanstd(runs[k,:,0]):.1f}|", flush=True)
        time.sleep(1.0)
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT/f"fmax_enroll_{HOST}_{RUNTAG}.npz", runs=runs,
                        fields=np.array(CPPC_FIELDS+["fmean","fmax","fstd","delivered"]))
    # quick intra reproducibility: per-run flattened vector, cosine across the K runs
    flat = runs.reshape(K, -1)
    # robust per-dim z within this die's K runs for a scale-free view
    mu = np.nanmean(flat, 0, keepdims=True); sd = np.nanstd(flat, 0, keepdims=True)+1e-9
    fz = np.nan_to_num((flat-mu)/sd)
    import itertools
    raw = [float(np.nansum(a*b)/((np.linalg.norm(np.nan_to_num(a))*np.linalg.norm(np.nan_to_num(b)))+1e-9))
           for a, b in itertools.combinations(np.nan_to_num(flat), 2)]
    print(f"  RAW intra cosine across {K} runs: mean={np.mean(raw):.4f} min={np.min(raw):.4f} "
          f"(static CPPC is firmware-constant so expect very high)", flush=True)
    # how much of the vector is the static (perfectly reproducible) part vs dynamic
    static_var = np.nanstd(runs[:, :, :len(CPPC_FIELDS)], 0).mean()
    dyn_var = np.nanstd(runs[:, :, len(CPPC_FIELDS):], 0).mean()
    print(f"  static-field run-to-run std={static_var:.3f}  dynamic-field std={dyn_var:.3f}", flush=True)
    print(f">>> saved fmax_enroll_{HOST}_{RUNTAG}.npz", flush=True)


if __name__ == "__main__":
    main()
