"""H7 MICRO-level substrate nonlinearity — finer than DVFS. Eric: DVFS is too coarse; create the
interaction smartly with small programs (and writes). Here the interaction is MICROARCHITECTURAL:
two tiny worker processes pinned to the two SMT siblings of ONE physical core share its execution
ports. When both run, they throttle each other -> SUB-ADDITIVE throughput = a real A*B term at
single-core granularity (no coarse clock-domain knob, no dangerous writes — pure affinity + loops).

Inputs A,B in {0,1}: A activates the worker on sibling cpu_a, B on sibling cpu_b (same physical core).
Readout channels (the "body"): per-worker realized THROUGHPUT (iters/window — direct port-contention
signal), the two siblings' scaling_cur_freq, package/gpu power, thermal. Test:
  - per-channel A*B interaction (residual of additive fit vs A*B)
  - LINEAR XOR(A,B) on the readout vs raw-[A,B]-linear (must fail) vs shuffle null.
If linear-on-readout >> raw & shuffle -> the SMT port-contention physics computes XOR at micro scale.

Out: micro_nonlin_{host}.json. No sudo. Thermal-trivial (one physical core).
"""
from __future__ import annotations
import os, sys, time, json, argparse, multiprocessing as mp
import numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import h7_rt_phase0 as P0

HOST = P0.HOST

def worker(cpu, active, counter, stop):
    try: os.sched_setaffinity(0, {cpu})
    except Exception: pass
    x = 1.0103
    while not stop.value:
        if active.value:
            # dependent DIVISION chain -> serializes on the single shared FP divider per physical core,
            # so two SMT siblings contend ~maximally (target: >50% mutual slowdown -> linear-separable XOR)
            for _ in range(1500):
                x = (x + 1.7) / (x * 0.9999 + 0.3)
                x = (x * x + 0.5) / (x + 1.3)
            with counter.get_lock(): counter.value += 1
        else:
            time.sleep(0.0005)

def curfreq(cpu):
    try: return int(Path(f"/sys/devices/system/cpu/cpu{cpu}/cpufreq/scaling_cur_freq").read_text())
    except Exception: return 0

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--win", type=float, default=0.06)
    ap.add_argument("--cpu_a", type=int, default=0)
    ap.add_argument("--cpu_b", type=int, default=1)   # SMT sibling of cpu_a
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    sib = Path(f"/sys/devices/system/cpu/cpu{a.cpu_a}/topology/thread_siblings_list").read_text().strip()
    print(f"[{HOST}] micro-nonlin SMT cpu_a={a.cpu_a} cpu_b={a.cpu_b} (siblings={sib}) steps={a.steps} win={a.win}", flush=True)

    actA = mp.Value('i', 0); actB = mp.Value('i', 0)
    cntA = mp.Value('L', 0); cntB = mp.Value('L', 0); stop = mp.Value('i', 0)
    pA = mp.Process(target=worker, args=(a.cpu_a, actA, cntA, stop), daemon=True)
    pB = mp.Process(target=worker, args=(a.cpu_b, actB, cntB, stop), daemon=True)
    pA.start(); pB.start(); time.sleep(0.3)

    rng = np.random.default_rng(a.seed)
    A = rng.integers(0, 2, a.steps).astype(np.int8); B = rng.integers(0, 2, a.steps).astype(np.int8)
    rows = []
    pwr = lambda: float(P0.read_gpu_power()[0]) if P0.read_gpu_power().size else 0.0
    try:
        for t in range(a.steps):
            with cntA.get_lock(): cntA.value = 0
            with cntB.get_lock(): cntB.value = 0
            actA.value = int(A[t]); actB.value = int(B[t])
            time.sleep(a.win)
            tpA = cntA.value; tpB = cntB.value
            actA.value = 0; actB.value = 0
            rows.append([tpA, tpB, curfreq(a.cpu_a), curfreq(a.cpu_b), pwr(), P0.zone0()])
            if (t+1) % 100 == 0: print(f"  step {t+1}/{a.steps} tpA={tpA} tpB={tpB}", flush=True)
    finally:
        stop.value = 1; time.sleep(0.1)
    X = np.array(rows, float)
    names = ["tp_A", "tp_B", "freq_A", "freq_B", "power", "zone0"]

    # analysis
    Ac = A.astype(float); Bc = B.astype(float)
    DES = np.column_stack([np.ones(len(Ac)), Ac, Bc]); inter = []
    for j in range(X.shape[1]):
        y = X[:, j]
        if y.std() < 1e-9: continue
        beta, *_ = np.linalg.lstsq(DES, y, rcond=None); resid = y - DES@beta
        ab = (Ac*Bc) - (Ac*Bc).mean(); den = resid.std()*ab.std()
        inter.append((names[j], round(float((resid*ab).mean()/den), 3) if den > 1e-12 else 0.0))
    inter.sort(key=lambda kv: -abs(kv[1]))

    def lin(Feat, y, shuffle=False, lam=2.0, folds=5):
        idx = np.arange(len(y)); F = Feat.astype(float)
        sd = F.std(0); F = F[:, sd > 1e-9]
        if F.shape[1] == 0: return np.nan
        F = (F - F.mean(0))/(F.std(0)+1e-9)
        if shuffle: F = F[np.random.default_rng(5).permutation(len(F))]
        F = np.column_stack([F, np.ones(len(F))]); n=len(y); bs=max(1,n//folds); pr=np.full(n,np.nan)
        for k in range(folds):
            te=np.zeros(n,bool); te[k*bs:(k+1)*bs if k<folds-1 else n]=True; tr=~te
            M=F[tr]; W=np.linalg.solve(M.T@M+lam*np.eye(M.shape[1]),M.T@y[tr].astype(float)); pr[te]=F[te]@W
        m=~np.isnan(pr); return float(((pr[m]>0.5).astype(int)==y[m].astype(int)).mean())
    xor = (A.astype(int) ^ B.astype(int)).astype(float); AB = np.column_stack([A, B]).astype(float)
    # sub-additivity sanity: mean throughput-sum per (A,B) cell
    tot = X[:,0] + X[:,1]
    cells = {f"{int(av)}{int(bv)}": round(float(tot[(A==av)&(B==bv)].mean()), 1)
             for av in (0,1) for bv in (0,1)}
    res = {"host": HOST, "steps": a.steps, "channels": names,
           "throughput_sum_by_AB": cells,
           "interaction": inter,
           "XOR": {"readout_linear": round(lin(X, xor), 3),
                   "rawAB_linear": round(lin(AB, xor), 3),
                   "shuffle_null": round(lin(X, xor, shuffle=True), 3),
                   "chance": round(float(max(xor.mean(), 1-xor.mean())), 3)}}
    jp = P0.OUT/f"micro_nonlin_{HOST}.json"; jp.write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2), flush=True)
    x = res["XOR"]
    print(f"\n[{HOST}] MICRO VERDICT (SMT port contention):")
    print(f"  throughput-sum by (A,B): {cells}  (sub-additive if '11' < '10'+'01'-'00')")
    print(f"  interaction: {inter}")
    print(f"  XOR: readout-LIN={x['readout_linear']} raw[A,B]={x['rawAB_linear']} "
          f"shuffle={x['shuffle_null']} chance={x['chance']}")
    print(f"  -> MICRO silicon computes if readout-LIN >> raw & shuffle. saved {jp}")

if __name__ == "__main__":
    main()
