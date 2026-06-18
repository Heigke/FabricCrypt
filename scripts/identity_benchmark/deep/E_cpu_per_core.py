"""E. CPU per-core silicon fingerprint over N logical cores.
Per core: max_freq from sysfs, completion-time for fixed workload via taskset,
RAPL package idle-vs-active delta.
"""
import argparse, os, sys, time, subprocess, numpy as np, glob
sys.path.insert(0, os.path.dirname(__file__))
from _common import (power_watts, temp_c, wait_cool, abort_if_hot,
                     bootstrap_ci, save_json, host_label)

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

def sysfs(path, default=""):
    try: return open(path).read().strip()
    except: return default

def core_meta(c):
    base=f"/sys/devices/system/cpu/cpu{c}/cpufreq"
    return dict(
        cpu=c,
        scaling_max_freq=sysfs(f"{base}/scaling_max_freq"),
        scaling_min_freq=sysfs(f"{base}/scaling_min_freq"),
        cpuinfo_max_freq=sysfs(f"{base}/cpuinfo_max_freq"),
        amd_pstate_highest_perf=sysfs(f"{base}/amd_pstate_highest_perf"),
        amd_pstate_max_freq=sysfs(f"{base}/amd_pstate_max_freq"),
        amd_pstate_prefcore_ranking=sysfs(f"{base}/amd_pstate_prefcore_ranking"),
    )

def measure_idle_power(secs=1.0):
    samples=[]
    t_end=time.time()+secs
    while time.time()<t_end:
        samples.append(power_watts()); time.sleep(0.05)
    import numpy as _np
    return float(_np.mean(samples)) if samples else 0.0

def time_core(core, N=384, reps=20, repeats=5):
    """Returns list of completion times running fixed work pinned to core."""
    times=[]
    py = sys.executable
    src_path = "/tmp/_core_probe.py"
    with open(src_path,"w") as f: f.write(WORK_SRC)
    for r in range(repeats):
        if abort_if_hot(72): break
        try:
            out = subprocess.check_output(
                ["taskset","-c",str(core), py, src_path, str(N), str(reps)],
                timeout=30, text=True).strip().splitlines()[-1]
            times.append(float(out))
        except Exception as e:
            print(f"[warn] core {core}: {e}", flush=True)
        time.sleep(0.1)
    return times

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--cores", type=int, default=16)  # measure first 16 logical cores
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--N", type=int, default=384)
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--smoke", action="store_true")
    args=ap.parse_args()
    if args.smoke: args.cores, args.repeats = 2, 2

    n_total = len(glob.glob("/sys/devices/system/cpu/cpu[0-9]*"))
    cores = list(range(min(args.cores, n_total)))
    t_start=time.time()
    # idle baseline
    if not wait_cool(thresh=55, timeout=60): print("[warn] precool slow", flush=True)
    idle_w = measure_idle_power(2.0)
    per_core = []
    for c in cores:
        if not wait_cool(thresh=58, timeout=60):
            print(f"[warn] core {c} cool slow", flush=True)
        # active power: measure during a run
        meta = core_meta(c)
        # sample power continuously while workload runs
        import threading as _th
        samples=[]
        stop=[False]
        def sampler():
            while not stop[0]:
                samples.append(power_watts()); time.sleep(0.05)
        _t = _th.Thread(target=sampler, daemon=True); _t.start()
        times = time_core(c, N=args.N, reps=args.reps, repeats=args.repeats)
        stop[0]=True; _t.join(timeout=1)
        active_w = float(np.mean(samples)) if samples else float("nan")
        m, lo, hi = bootstrap_ci(times) if times else (float("nan"),)*3
        per_core.append(dict(
            core=c, meta=meta, times_s=times, time_ci=[m,lo,hi],
            active_W=float(active_w), delta_W=float(active_w-idle_w),
            tmax=float(temp_c())))
        print(f"[core {c}] t={m:.3f}s active={active_w:.1f}W d={active_w-idle_w:+.1f}W", flush=True)
    save_json(args.out, dict(
        host=host_label(), idle_W=idle_w, wall_s=time.time()-t_start,
        per_core=per_core))

if __name__=="__main__": main()
