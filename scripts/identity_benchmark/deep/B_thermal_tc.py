"""B. Thermal time-constant. 10 step cycles of IDLE -> ~25W -> IDLE.
Per-cycle shortened from 5 min to 90s (heat 30s + cool 60s) for budget;
single-exponential fit still well-resolved at 2 Hz sample (180 pts/cycle).
"""
import argparse, os, sys, time, numpy as np, threading
sys.path.insert(0, os.path.dirname(__file__))
from _common import (power_watts, temp_c, wait_cool, abort_if_hot,
                     bootstrap_ci, save_json, host_label)

def cpu_busy(duration_s):
    stop=[False]
    def worker():
        A=np.random.randn(768,768).astype(np.float32)
        B=np.random.randn(768,768).astype(np.float32)
        while not stop[0]:
            A = A @ B * 1e-3 + 1e-3
    t=threading.Thread(target=worker, daemon=True); t.start()
    time.sleep(duration_s); stop[0]=True; time.sleep(0.05)

def sample_temp(duration_s, hz=2):
    period=1.0/hz
    ts, te, pw = [], [], []
    t0=time.time(); t_end=t0+duration_s
    while time.time()<t_end:
        ts.append(time.time()); te.append(temp_c()); pw.append(power_watts())
        time.sleep(period)
    return np.array(ts)-ts[0], np.array(te), np.array(pw)

def fit_exp(t, y, rising=True):
    """y = y_inf + (y0-y_inf)*exp(-t/tau); fit tau and y_inf via linearization."""
    t=np.asarray(t,float); y=np.asarray(y,float)
    if len(t)<5: return float("nan"), float("nan"), float("nan")
    y_inf = y[-3:].mean()
    y0 = y[:3].mean()
    diff = (y - y_inf) / ((y0 - y_inf) if abs(y0-y_inf)>1e-6 else 1e-6)
    mask = diff > 0.05
    if mask.sum()<5: return float("nan"), float(y_inf), float(y0)
    try:
        coef = np.polyfit(t[mask], np.log(diff[mask]), 1)
        tau = -1.0/coef[0] if coef[0]!=0 else float("nan")
    except Exception:
        tau = float("nan")
    return float(tau), float(y_inf), float(y0)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--cycles", type=int, default=10)
    ap.add_argument("--heat_s", type=float, default=30.0)
    ap.add_argument("--cool_s", type=float, default=60.0)
    ap.add_argument("--smoke", action="store_true")
    args=ap.parse_args()
    if args.smoke: args.cycles, args.heat_s, args.cool_s = 2, 8.0, 15.0

    cycles=[]
    t_start=time.time()
    for c in range(args.cycles):
        if not wait_cool(thresh=55, timeout=90):
            print(f"[WARN] cycle {c}: cool timeout", flush=True)
        if abort_if_hot(72):
            print("[ABORT] hot", flush=True); break
        # heat
        wt=threading.Thread(target=cpu_busy, args=(args.heat_s+0.5,), daemon=True); wt.start()
        time.sleep(0.1)
        ts_h, te_h, pw_h = sample_temp(args.heat_s)
        wt.join(timeout=3)
        # cool
        ts_c, te_c, pw_c = sample_temp(args.cool_s)
        tau_h, T_inf, T0 = fit_exp(ts_h, te_h, rising=True)
        # for cooling fit, reverse so it's "decay to baseline"
        tau_c, T_base, T_hot = fit_exp(ts_c, te_c, rising=False)
        P_avg = float(pw_h.mean()) if len(pw_h) else float("nan")
        dT = T_inf - T0
        R_th = dT / P_avg if P_avg>0 else float("nan")  # K/W
        cycles.append(dict(
            tau_heat=tau_h, tau_cool=tau_c, T_inf=T_inf, T0=T0,
            P_avg_W=P_avg, R_th_K_per_W=R_th,
            heat_temps=list(map(float,te_h)), cool_temps=list(map(float,te_c)),
            heat_times=list(map(float,ts_h)), cool_times=list(map(float,ts_c))))
        print(f"[cycle {c+1}/{args.cycles}] tau_h={tau_h:.1f} tau_c={tau_c:.1f} P={P_avg:.1f}W R_th={R_th:.3f}", flush=True)

    # bootstrap
    def safe(arr): return [x for x in arr if x==x and abs(x)<1e4]
    tau_h_v = safe([c["tau_heat"] for c in cycles])
    tau_c_v = safe([c["tau_cool"] for c in cycles])
    rth_v = safe([c["R_th_K_per_W"] for c in cycles])
    summary = dict(
        tau_heat_ci=bootstrap_ci(tau_h_v) if tau_h_v else None,
        tau_cool_ci=bootstrap_ci(tau_c_v) if tau_c_v else None,
        Rth_ci=bootstrap_ci(rth_v) if rth_v else None,
        n_cycles=len(cycles),
    )
    save_json(args.out, dict(host=host_label(), wall_s=time.time()-t_start,
                              cycles=cycles, summary=summary))

if __name__=="__main__": main()
