"""Fast 23-feature envelope-signature collector.

Layout MATCHES scripts/identity_benchmark/phase2_v2/_substrate_v2.py:
  A_power (8): per-workload (IDLE,LIGHT,MEDIUM,HEAVY) [mean_W, std_W]
  B_thermal (3): tau_heat, tau_cool, R_th_K_per_W
  E_cpu (12): per-core time/freq stats

Designed to run in ~3-5 min so we can repeat for A2/A3/A4 cheaply.

Usage:
  python envelope_fast.py --out PATH [--label TAG] [--quick]
"""
from __future__ import annotations
import argparse, json, os, sys, time, socket, threading, subprocess
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "deep"))
from _common import (power_watts, temp_c, gpu_temp_c, wait_cool, abort_if_hot,
                     host_label)  # type: ignore

WORKLOADS = ["IDLE", "LIGHT", "MEDIUM", "HEAVY"]


def cpu_busy(duration_s, threads=1, intensity="medium", temp_cap=78.0):
    stop = [False]
    def worker():
        size = {"light": 192, "medium": 384, "heavy": 512}[intensity]
        A = np.random.randn(size, size).astype(np.float32)
        B = np.random.randn(size, size).astype(np.float32)
        while not stop[0]:
            A = A @ B * 1e-3 + 1e-3
    ts = [threading.Thread(target=worker, daemon=True) for _ in range(threads)]
    [t.start() for t in ts]
    t_end = time.time() + duration_s
    while time.time() < t_end:
        if temp_c() > temp_cap:
            break
        time.sleep(0.2)
    stop[0] = True
    time.sleep(0.05)


def sample_power(duration_s, hz=50):
    period = 1.0 / hz
    pw, te = [], []
    t_end = time.time() + duration_s
    while time.time() < t_end:
        pw.append(power_watts()); te.append(temp_c())
        time.sleep(period)
    return np.asarray(pw), np.asarray(te)


def measure_A_power(dur_per_wl=6.0, quick=False):
    if quick:
        dur_per_wl = 4.0
    A = {}
    for w in WORKLOADS:
        wait_cool(thresh=55, timeout=30)
        if w == "IDLE":
            time.sleep(0.5)
            pw, _ = sample_power(dur_per_wl)
        elif w == "LIGHT":
            t = threading.Thread(target=cpu_busy, args=(dur_per_wl,), kwargs={"threads": 1, "intensity": "light"}, daemon=True)
            t.start(); pw, _ = sample_power(dur_per_wl); t.join(timeout=1)
        elif w == "MEDIUM":
            t = threading.Thread(target=cpu_busy, args=(dur_per_wl,), kwargs={"threads": 1, "intensity": "medium"}, daemon=True)
            t.start(); pw, _ = sample_power(dur_per_wl); t.join(timeout=1)
        elif w == "HEAVY":
            t = threading.Thread(target=cpu_busy, args=(dur_per_wl,), kwargs={"threads": 2, "intensity": "heavy"}, daemon=True)
            t.start(); pw, _ = sample_power(dur_per_wl); t.join(timeout=1)
        A[w] = {"mean_W": float(np.mean(pw)), "std_W": float(np.std(pw)), "n": int(len(pw))}
    return A


def measure_B_thermal(heat_s=15.0, cool_s=25.0, cycles=2, quick=False):
    if quick:
        heat_s, cool_s, cycles = 10.0, 18.0, 2
    tau_heats, tau_cools, R_ths = [], [], []
    for _ in range(cycles):
        wait_cool(thresh=50, timeout=60)
        # heat phase
        t0 = time.time()
        ts_h, te_h, pw_h = [], [], []
        worker = threading.Thread(target=cpu_busy, args=(heat_s,), kwargs={"threads": 2, "intensity": "heavy"}, daemon=True)
        worker.start()
        while time.time() - t0 < heat_s:
            ts_h.append(time.time() - t0); te_h.append(temp_c()); pw_h.append(power_watts())
            time.sleep(0.5)
        worker.join(timeout=1)
        # cool phase
        t1 = time.time()
        ts_c, te_c = [], []
        while time.time() - t1 < cool_s:
            ts_c.append(time.time() - t1); te_c.append(temp_c())
            time.sleep(0.5)
        # fit
        te_h = np.array(te_h); te_c = np.array(te_c)
        ts_h = np.array(ts_h); ts_c = np.array(ts_c)
        # tau_heat
        try:
            y_inf = te_h[-3:].mean(); y0 = te_h[:3].mean()
            diff = (y_inf - te_h) / max(abs(y_inf - y0), 1e-6)
            mask = diff > 0.05
            if mask.sum() >= 5:
                coef = np.polyfit(ts_h[mask], np.log(diff[mask]), 1)
                tau_h = -1.0 / coef[0] if coef[0] != 0 else float("nan")
                tau_heats.append(float(tau_h))
        except Exception:
            pass
        try:
            y_inf = te_c[-3:].mean(); y0 = te_c[:3].mean()
            diff = (te_c - y_inf) / max(abs(y0 - y_inf), 1e-6)
            mask = diff > 0.05
            if mask.sum() >= 5:
                coef = np.polyfit(ts_c[mask], np.log(diff[mask]), 1)
                tau_c = -1.0 / coef[0] if coef[0] != 0 else float("nan")
                tau_cools.append(float(tau_c))
        except Exception:
            pass
        try:
            dT = te_h[-3:].mean() - te_h[:3].mean()
            dP = np.mean(pw_h[-len(pw_h)//3:]) - np.mean(pw_h[:max(1, len(pw_h)//3)])
            if abs(dP) > 0.5:
                R_ths.append(float(dT / dP))
        except Exception:
            pass
    def mfin(x):
        x = [v for v in x if np.isfinite(v)]
        return float(np.median(x)) if x else 0.0
    return {"tau_heat": mfin(tau_heats), "tau_cool": mfin(tau_cools), "R_th_K_per_W": mfin(R_ths)}


WORK_SRC = """
import time, numpy as np, sys
N = int(sys.argv[1]); reps = int(sys.argv[2])
A = np.random.RandomState(42).randn(N,N).astype(np.float32)
B = np.random.RandomState(43).randn(N,N).astype(np.float32)
t0=time.time()
for _ in range(reps):
    C = A @ B * 1e-3 + 1e-3
    A = C
print(time.time()-t0)
"""


def measure_E_cpu(N=256, reps=8, repeats=2, quick=False):
    if quick:
        N, reps, repeats = 192, 5, 2
    src = "/tmp/_emb_core_probe.py"
    Path(src).write_text(WORK_SRC)
    import multiprocessing
    n_cores = multiprocessing.cpu_count()
    # sample up to 16 cores spaced evenly
    cores = list(range(min(n_cores, 16)))
    per_core_time = []
    per_core_freq = []
    py = sys.executable
    for c in cores:
        times = []
        for _ in range(repeats):
            try:
                r = subprocess.run(
                    ["taskset", "-c", str(c), py, src, str(N), str(reps)],
                    capture_output=True, text=True, timeout=30,
                )
                if r.returncode == 0:
                    times.append(float(r.stdout.strip()))
            except Exception:
                pass
        per_core_time.append(float(np.median(times)) if times else 0.0)
        # freq
        try:
            f = open(f"/sys/devices/system/cpu/cpu{c}/cpufreq/scaling_cur_freq").read().strip()
            per_core_freq.append(float(f))
        except Exception:
            per_core_freq.append(0.0)
    t = np.asarray(per_core_time); f = np.asarray(per_core_freq)
    # rank correlation
    try:
        from scipy.stats import pearsonr
        rc, _ = pearsonr(np.argsort(t).argsort(), np.argsort(f).argsort())
    except Exception:
        rc = float(np.corrcoef(np.argsort(t).argsort(), np.argsort(f).argsort())[0, 1])
    return {
        "per_core_time": per_core_time,
        "per_core_freq": per_core_freq,
        "rank_corr_pearson": float(rc) if np.isfinite(rc) else 0.0,
    }


def vec_from_AB(A: dict, B: dict, E: dict) -> tuple[np.ndarray, list]:
    feats = []
    labels = []
    for w in WORKLOADS:
        feats.append(A[w]["mean_W"]); feats.append(A[w]["std_W"])
        labels.append(f"A_{w}_mean"); labels.append(f"A_{w}_std")
    for k in ("tau_heat", "tau_cool", "R_th_K_per_W"):
        feats.append(B[k]); labels.append(f"B_{k}")
    t = np.asarray(E["per_core_time"], dtype=float)
    f = np.asarray(E["per_core_freq"], dtype=float)
    e_feats = [
        float(np.percentile(t, 25)), float(np.median(t)), float(np.percentile(t, 75)),
        float(np.std(t)), float(np.median(f)), float(np.std(f)),
        float(E["rank_corr_pearson"]),
        float(np.min(t)), float(np.max(t)),
        float(np.min(f)), float(np.max(f)),
        float(np.percentile(t, 75) - np.percentile(t, 25)),
    ]
    feats.extend(e_feats)
    labels.extend(["E_t_p25","E_t_med","E_t_p75","E_t_std","E_f_med","E_f_std",
                   "E_rank_corr","E_t_min","E_t_max","E_f_min","E_f_max","E_t_iqr"])
    return np.asarray(feats, dtype=float), labels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--label", default="default")
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()

    t0 = time.time()
    print(f"[envelope_fast] host={host_label()} label={args.label} apu={temp_c():.1f}C", flush=True)
    if abort_if_hot(78):
        print("[envelope_fast] thermal abort precheck", flush=True)
        sys.exit(2)

    A = measure_A_power(quick=args.quick)
    print(f"[envelope_fast] A done {time.time()-t0:.1f}s", flush=True)
    B = measure_B_thermal(quick=args.quick)
    print(f"[envelope_fast] B done {time.time()-t0:.1f}s", flush=True)
    E = measure_E_cpu(quick=args.quick)
    print(f"[envelope_fast] E done {time.time()-t0:.1f}s", flush=True)

    vec, labels = vec_from_AB(A, B, E)
    out = {
        "host": host_label(),
        "label": args.label,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "uptime_s": float(open("/proc/uptime").read().split()[0]),
        "apu_temp_c_pre": temp_c(),
        "elapsed_s": time.time() - t0,
        "vec23": vec.tolist(),
        "labels": labels,
        "A": A, "B": B, "E": E,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2, default=str))
    print(f"[envelope_fast] wrote {args.out} ({len(vec)} feats, took {time.time()-t0:.1f}s)", flush=True)


if __name__ == "__main__":
    main()
