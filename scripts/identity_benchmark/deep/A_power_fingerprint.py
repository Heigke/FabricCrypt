"""A. Power-draw fingerprint at 4 workloads (IDLE/LIGHT/MEDIUM/HEAVY).
Sample RAPL @ ~50 Hz; per-workload 60 s (=> 3000 samples); 10 reps randomized.
For thermal safety we shorten to per-rep 8 s (400 samples) by default; CLI --full for 60 s.
"""
import argparse, os, sys, time, random, numpy as np, threading
sys.path.insert(0, os.path.dirname(__file__))
from _common import (power_watts, temp_c, gpu_temp_c, wait_cool, abort_if_hot,
                     bootstrap_ci, save_json, host_label)

def cpu_busy(duration_s, threads=1, intensity="medium", temp_cap=88.0):
    """Tunable workload, joinable threads, thermal cap."""
    stop = [False]
    def worker():
        size = {"light":192,"medium":384,"heavy":512}[intensity]
        A = np.random.randn(size,size).astype(np.float32)
        B = np.random.randn(size,size).astype(np.float32)
        while not stop[0]:
            A = A @ B * 1e-3 + 1e-3
    ts = [threading.Thread(target=worker) for _ in range(threads)]
    [t.start() for t in ts]
    t_end = time.time() + duration_s
    while time.time() < t_end:
        from _common import temp_c as _tc
        if _tc() > temp_cap:
            break
        time.sleep(0.2)
    stop[0] = True
    for t in ts: t.join(timeout=2)

def sample_power(duration_s, hz=50):
    """Return list of instantaneous power readings (W) sampled at hz."""
    period = 1.0/hz
    out_p, out_t = [], []
    t_end = time.time() + duration_s
    while time.time() < t_end:
        out_p.append(power_watts())
        out_t.append(temp_c())
        time.sleep(period)
    return out_p, out_t

def autocorr_tau(x):
    x = np.asarray(x,dtype=float)
    if len(x) < 8: return float("nan")
    x = x - x.mean()
    if x.std()==0: return 0.0
    ac = np.correlate(x,x,mode="full")[len(x)-1:]
    ac /= ac[0] if ac[0]!=0 else 1
    # tau = first lag where ac < 1/e
    for i,v in enumerate(ac):
        if v < 1/np.e: return float(i)
    return float(len(x))

def workload(name, duration):
    if name=="IDLE":
        time.sleep(duration)
    elif name=="LIGHT":
        cpu_busy(duration, threads=1, intensity="light")
    elif name=="MEDIUM":
        cpu_busy(duration, threads=1, intensity="medium")
    elif name=="HEAVY":
        cpu_busy(duration, threads=2, intensity="heavy")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--dur", type=float, default=8.0)  # safe default
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke: args.reps, args.dur = 2, 3.0

    wls = ["IDLE","LIGHT","MEDIUM","HEAVY"]
    order = []
    rng = random.Random(0x1151)
    for _ in range(args.reps):
        block = wls[:]; rng.shuffle(block); order.extend(block)

    raw = {w:[] for w in wls}
    temps = {w:[] for w in wls}
    t_start = time.time()
    for i,w in enumerate(order):
        if not wait_cool(thresh=55, timeout=120):
            print(f"[WARN] cooldown timeout before {w} rep at T={temps[w][-1][-1] if temps[w] else '?'}", flush=True)
        if abort_if_hot(70):
            print(f"[ABORT] hot before {w}", flush=True); break
        wt = threading.Thread(target=workload, args=(w, args.dur+0.5))
        wt.start()
        time.sleep(0.3)
        p,t = sample_power(args.dur)
        wt.join(timeout=5)
        raw[w].append(p)
        temps[w].append(t)
        print(f"[{i+1}/{len(order)}] {w} mean={np.mean(p):.2f}W n={len(p)} Tmax={max(t):.1f}", flush=True)

    # stats
    stats = {}
    for w in wls:
        all_p = np.concatenate([np.array(r) for r in raw[w] if len(r)>0]) if raw[w] else np.array([])
        if len(all_p)==0:
            stats[w] = {"n":0}; continue
        means = [float(np.mean(r)) for r in raw[w] if len(r)>0]
        m, lo, hi = bootstrap_ci(means)
        stats[w] = {
            "n_reps": len(raw[w]),
            "n_samples": int(len(all_p)),
            "mean_W": float(np.mean(all_p)),
            "std_W": float(np.std(all_p)),
            "mean_W_ci": [m, lo, hi],
            "p10": float(np.percentile(all_p,10)),
            "p50": float(np.percentile(all_p,50)),
            "p90": float(np.percentile(all_p,90)),
            "autocorr_tau": autocorr_tau(all_p),
            "per_rep_means": means,
            "hist_bins": list(np.histogram(all_p,bins=30)[1]),
            "hist_counts": list(map(int, np.histogram(all_p,bins=30)[0])),
        }
    save_json(args.out, {
        "host": host_label(),
        "wall_s": time.time()-t_start,
        "stats": stats,
        "order": order,
    })

if __name__=="__main__":
    main()
