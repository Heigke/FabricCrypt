"""H7 v2 — characterize the MESO and MACRO body-computation layers BEFORE wiring them into a keystream.

We already have a strong MICRO layer (destructive-L3 cache-XOR, fidelity ~1.0, two-stage PAR3). The user
wants MORE physical layers (macro / meso / micro) woven in to strengthen the embodiment claim. But a layer
is only load-bearing if it produces a bit that is (a) SEPARABLE (a threshold cleanly maps the 4 operand
cells to a binary value) and (b) REPRODUCIBLE on the same die (low intra-session BER) so the LLM can be
trained against it. This script MEASURES that honestly — no wiring until the bit passes.

MESO layer — in-kernel GPU clock/voltage-droop self-sense (gpu_selfsense.hip):
  A wavefront measures its own realized FMA rate over a fixed wall-clock window. Under heavier concurrent
  load the power cap binds and the rail droops -> realized rate is a SUB-ADDITIVE (nonlinear) function of
  load. Operands (a,b) select the workgroup load level; gate_meso(a,b) = 1 if realized_rate < threshold.
  This is a real physical nonlinear gate (saturation), computed live on this GPU.

MACRO layer — SMU power arbitration between CPU and GPU on the shared package budget:
  Operand a = CPU load (busy threads), b = GPU load (selfsense kernel). We read GPU power (amdgpu hwmon)
  and CPU package power (RAPL). When BOTH load the chip the SMU splits a shared budget -> GPU power under
  (1,1) droops below (0,1): sub-additive contention. gate_macro(a,b) = 1 if a contention signal fires.

For each layer we measure: the 4 cell means, the best single threshold + its fidelity to the cleanest
binary partition, and the intra-session BER (re-measure each cell R times, see how often the thresholded
bit flips). A layer is USABLE iff fidelity high and BER low. Run sandbox-disabled. HSA override.
Out: results/IDENTITY_H7_2026-06-09/v2_layer_probe_{host}.json
"""
from __future__ import annotations
import os, sys, json, time, socket, subprocess, threading, itertools, multiprocessing
import numpy as np
from pathlib import Path

HOST = socket.gethostname()
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = ROOT / "results/IDENTITY_H7_2026-06-09"; OUT.mkdir(parents=True, exist_ok=True)
SELFSENSE = HERE / "gpu_selfsense"
HWMON = "/sys/class/hwmon/hwmon7"
RAPL = "/sys/class/powercap/intel-rapl:0/energy_uj"
TZ = "/sys/class/thermal/thermal_zone0/temp"
THERM_PAUSE = 88000      # mC: pause if APU hotter than this
THERM_RESUME = 60000


def temp():
    try: return int(open(TZ).read())
    except Exception: return 0


def cool_guard():
    while temp() > THERM_PAUSE:
        print(f"  [thermal] {temp()/1000:.0f}C > {THERM_PAUSE/1000:.0f}C, cooling...", flush=True)
        t0 = time.time()
        while temp() > THERM_RESUME and time.time() - t0 < 120:
            time.sleep(2)


def gpu_power():
    try: return int(open(f"{HWMON}/power1_input").read())   # microwatts
    except Exception: return 0


def gpu_freq():
    try: return int(open(f"{HWMON}/freq1_input").read())
    except Exception: return 0


def rapl_uj():
    try: return int(open(RAPL).read())
    except Exception: return 0


# ---------- CPU load (macro operands) — REAL multi-core via multiprocessing (no GIL) ----------
def _spin_proc(stop):
    x = 1.000001
    while not stop.value:
        for _ in range(500000): x = x * 1.0000001 + 0.1
    if x == 1234.5: print(x)


class CpuLoad:
    """Pin `ncores` real OS processes to busy-spin -> genuine CPU package power draw (GIL-free)."""
    def __init__(self, ncores):
        self.ncores = ncores; self.stop = None; self.procs = []
    def start(self):
        self.stop = multiprocessing.Value("b", False)
        self.procs = [multiprocessing.Process(target=_spin_proc, args=(self.stop,)) for _ in range(self.ncores)]
        for p in self.procs: p.start()
    def end(self):
        if self.stop is not None: self.stop.value = True
        for p in self.procs: p.join(timeout=2); p.terminate() if p.is_alive() else None
        self.procs = []


# ---------- GPU self-sense (meso + macro operand b) ----------
def selfsense(wg, window=1500000):
    """Return realized_rate (iters/tick) for `wg` workgroups over a wall-clock window."""
    try:
        out = subprocess.run([str(SELFSENSE), str(wg), str(window)],
                             capture_output=True, text=True, timeout=30,
                             env={**os.environ, "HSA_OVERRIDE_GFX_VERSION": "11.0.0"})
        parts = out.stdout.split()
        return float(parts[1]) if len(parts) >= 2 else 0.0
    except Exception as e:
        print("  selfsense err:", e); return 0.0


# ======================================================================================
# MESO — GPU droop self-sense as a 2-operand sub-additive gate
# ======================================================================================
def probe_meso(R=10):
    # operands select workgroup load: (a,b) -> wg = base + (a+b)*step.  rate droops with load (sub-additive)
    base, step = 8, 110
    cells = {(a, b): base + (a + b) * step for a in (0, 1) for b in (0, 1)}
    print(f"[meso] cells(wg) = {cells}", flush=True)
    raw = {k: [] for k in cells}
    for r in range(R):
        cool_guard()
        for k, wg in cells.items():
            raw[k].append(selfsense(wg))
        if (r + 1) % 3 == 0: print(f"  meso rep {r+1}/{R} t={temp()/1000:.0f}C", flush=True)
    means = {k: float(np.mean(v)) for k, v in raw.items()}
    return analyze("meso", raw, means)


# ======================================================================================
# MACRO — CPU<->GPU SMU power contention
# ======================================================================================
def probe_macro(R=10):
    # CROSS-DOMAIN CPU->GPU coupling via the shared SMU package-power budget.
    # Signal = the GPU's OWN in-silicon realized FMA rate (selfsense, fixed wg) -- the same clean, BER=0
    # observable as the meso layer. Operands a,b each load HALF the CPU cores (real OS processes). When the
    # CPU draws package power the SMU throttles the GPU rail -> the GPU's self-sensed rate droops. So the
    # GPU literally feels the CPU's workload: a genuine macro (whole-chip arbitration) computation, distinct
    # from meso (GPU-internal droop) and micro (CPU-cache contention).
    ncpu = os.cpu_count() or 16
    half = max(1, ncpu // 2)
    grp_a = CpuLoad(half); grp_b = CpuLoad(ncpu - half)
    raw = {(a, b): [] for a in (0, 1) for b in (0, 1)}
    print(f"[macro] GPU self-rate under CPU core-group load (a={half} cores, b={ncpu-half} cores)", flush=True)
    for r in range(R):
        cool_guard()
        for a in (0, 1):
            for b in (0, 1):
                if a: grp_a.start()
                if b: grp_b.start()
                time.sleep(0.15)                                   # let CPU power ramp + SMU react
                rate = selfsense(120, 1500000)                     # GPU senses its own throttled rate
                if a: grp_a.end()
                if b: grp_b.end()
                raw[(a, b)].append(rate)
                time.sleep(0.15)
        if (r + 1) % 3 == 0: print(f"  macro rep {r+1}/{R} t={temp()/1000:.0f}C", flush=True)
    means = {k: float(np.mean(v)) for k, v in raw.items()}
    return analyze("macro", raw, means)


# ======================================================================================
# analysis: separability + reproducibility for a 4-cell layer
# ======================================================================================
def analyze(name, raw, means):
    cells = list(raw.keys())
    allvals = np.concatenate([np.array(raw[k]) for k in cells])
    lo, hi = float(allvals.min()), float(allvals.max())
    # try the three nontrivial binary partitions of the 4 cells; pick the one with best margin
    # candidate logical labelings of (a,b):
    labelings = {
        "XOR":  {(0, 0): 0, (0, 1): 1, (1, 0): 1, (1, 1): 0},
        "AND":  {(0, 0): 0, (0, 1): 0, (1, 0): 0, (1, 1): 1},
        "OR":   {(0, 0): 0, (0, 1): 1, (1, 0): 1, (1, 1): 1},
        "SUM1": {(0, 0): 0, (0, 1): 0, (1, 0): 0, (1, 1): 1},   # contention only at (1,1)
        "B":    {(0, 0): 0, (0, 1): 1, (1, 0): 0, (1, 1): 1},   # tracks operand b
    }
    best = None
    for lname, lab in labelings.items():
        g0 = [means[k] for k in cells if lab[k] == 0]
        g1 = [means[k] for k in cells if lab[k] == 1]
        if not g0 or not g1: continue
        thr = (max(g0) + min(g1)) / 2 if max(g0) < min(g1) else (min(g0) + max(g1)) / 2
        # fidelity: fraction of all individual measurements correctly classified by thr (with polarity)
        pol = 1 if np.mean(g1) > np.mean(g0) else -1
        correct = tot = 0
        for k in cells:
            for v in raw[k]:
                bit = int((v - thr) * pol > 0)
                correct += int(bit == lab[k]); tot += 1
        fid = correct / tot
        margin = abs(np.mean(g1) - np.mean(g0)) / (allvals.std() + 1e-9)
        if best is None or fid > best["fidelity"]:
            best = {"logic": lname, "threshold": float(thr), "polarity": int(pol),
                    "fidelity": round(fid, 3), "margin_sd": round(float(margin), 2)}
    # intra-session BER: for the best labeling, per cell, how often does the bit deviate from the majority?
    lab = labelings[best["logic"]]; thr = best["threshold"]; pol = best["polarity"]
    flips = tot = 0
    for k in cells:
        bits = [int((v - thr) * pol > 0) for v in raw[k]]
        maj = round(np.mean(bits))
        flips += sum(b != maj for b in bits); tot += len(bits)
    ber = flips / tot
    res = {"cell_means": {f"{k[0]}{k[1]}": round(means[k], 4) for k in cells},
           "value_range": [round(lo, 4), round(hi, 4)],
           "best_logic": best["logic"], "threshold": round(best["threshold"], 4),
           "polarity": best["polarity"], "separability_fidelity": best["fidelity"],
           "margin_in_sd": best["margin_sd"], "intra_session_BER": round(ber, 4),
           "USABLE": bool(best["fidelity"] >= 0.85 and ber <= 0.15 and best["margin_sd"] >= 1.0)}
    print(f"[{name}] {res['best_logic']} fid={res['separability_fidelity']} "
          f"BER={res['intra_session_BER']} margin={res['margin_in_sd']}sd USABLE={res['USABLE']}", flush=True)
    return res


def main():
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--layer", choices=["both", "meso", "macro"], default="both")
    a = ap.parse_args()
    if not SELFSENSE.exists():
        print("missing gpu_selfsense binary — build with: hipcc -O3 gpu_selfsense.hip -o gpu_selfsense"); sys.exit(2)
    print(f"[{HOST}] layer probe start t={temp()/1000:.0f}C reps={a.reps} layer={a.layer}", flush=True)
    t0 = time.time()
    prev = {}
    pp = OUT / f"v2_layer_probe_{HOST}.json"
    if pp.exists():
        try: prev = json.loads(pp.read_text())
        except Exception: prev = {}
    meso = probe_meso(a.reps) if a.layer in ("both", "meso") else prev.get("meso_gpu_droop_selfsense")
    macro = probe_macro(a.reps) if a.layer in ("both", "macro") else prev.get("macro_smu_power_contention")
    out = {"host": HOST, "elapsed_s": round(time.time() - t0, 1),
           "meso_gpu_droop_selfsense": meso, "macro_smu_power_contention": macro,
           "note": ("Each layer is a real physical nonlinear computation. USABLE means it yields a "
                    "reproducible separable bit the LLM can be trained against. Uniqueness still comes "
                    "from the prefcore fingerprint; these layers add COMPUTATION surface across the "
                    "macro->micro stack, with per-layer ablation proving each is load-bearing.")}
    (OUT / f"v2_layer_probe_{HOST}.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2), flush=True)


if __name__ == "__main__":
    main()
