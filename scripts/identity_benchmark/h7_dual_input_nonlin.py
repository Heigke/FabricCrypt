"""H7 dual-input substrate nonlinearity — look DEEPER & BROADER for intrinsic silicon nonlinearity.

Eric: "surely the low-level signals must have nonlinearities." Right — and a SINGLE-knob drive can't
reveal them (one scalar input => all channels collapse to functions of it => collinear by construction).
Intrinsic nonlinearity lives in the INTERACTION of TWO independent loads sharing a budget (package
power cap, thermal envelope, DVFS/boost arbitration, memory bus). When both run, they throttle each
other => SUB-ADDITIVE response => a genuine A*B cross-term in the silicon.

Two independent binary inputs per step:
  A = GPU matmul burst (on/off)      B = CPU all-core matmul burst (on/off, runs in a thread; numpy
  releases the GIL). On an APU these share one power/thermal/boost budget.

Single-step tasks (NO memory needed => immune to thermal-cooling fragmentation):
  - per-channel INTERACTION: fit metric ~ a + b*A + c*B (additive). corr(residual, A*B) = intrinsic
    nonlinear interaction strength of that channel. Broad sweep over ALL ~50 channels.
  - XOR(A,B) / AND / OR via a STRICT LINEAR readout on telemetry. XOR is linearly inseparable in (A,B),
    so linear-on-[A,B] CANNOT do it. If linear-on-TELEMETRY does => the SILICON supplied the A*B
    nonlinearity. Controls: linear-on-[A,B] baseline (XOR must fail), shuffle null.

Out: dual_input_nonlin_{host}.json (+ verdict). Thermal-safe gentle bursts + relaxation gap.
"""
from __future__ import annotations
import os, sys, time, json, argparse, threading
import numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import h7_rt_phase0 as P0
from h7_telemetry_reservoir import snap_vec

HOST = P0.HOST
THERM_ABORT = float(os.environ.get("THERM_ABORT", "78"))
THERM_COOL  = float(os.environ.get("THERM_COOL", "50"))

def make_dual_driver(dev, gpu_n, cpu_n):
    import torch
    if dev == "cuda":
        GA = torch.randn(gpu_n, gpu_n, device="cuda"); GB = torch.randn(gpu_n, gpu_n, device="cuda")
    else:
        GA = GB = None
    CA = np.random.randn(cpu_n, cpu_n); CB = np.random.randn(cpu_n, cpu_n)
    def cpu_work(stop_t):
        x = CA
        while time.monotonic() < stop_t:
            x = np.tanh(x @ CB)          # numpy matmul releases the GIL -> true parallelism w/ GPU loop
    def drive(a, b, secs):
        stop = time.monotonic() + secs
        th = None
        if b:
            th = threading.Thread(target=cpu_work, args=(stop,), daemon=True); th.start()
        if a and dev == "cuda":
            import torch
            nonlocal GA
            while time.monotonic() < stop:
                C = GA @ GB; GA.copy_(torch.tanh(C))
            torch.cuda.synchronize()
        # if a but cpu-only host, just wait (GPU path is the A channel)
        if th is not None: th.join()
        if not a and not b: time.sleep(secs)
    return drive

def record(steps, base_secs, gpu_n, cpu_n, gap, seed):
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    rng = np.random.default_rng(seed)
    A = rng.integers(0, 2, steps).astype(np.int8); B = rng.integers(0, 2, steps).astype(np.int8)
    drive = make_dual_driver(dev, gpu_n, cpu_n)
    samp = P0.Sampler(period=0.0); x0, names = snap_vec(samp); D = x0.size
    X = np.full((steps, D), np.nan); zt = np.zeros(steps)
    print(f"[{HOST}] dual-input dev={dev} steps={steps} base={base_secs}s gap={gap}s gpu_n={gpu_n} "
          f"cpu_n={cpu_n} D={D}", flush=True)
    for t in range(steps):
        if P0.zone0() >= THERM_ABORT: P0.wait_cool(THERM_COOL)
        drive(int(A[t]), int(B[t]), base_secs)
        v, _ = snap_vec(samp)
        if v.size == D: X[t] = v
        time.sleep(gap); zt[t] = P0.zone0()
        if (t+1) % 50 == 0: print(f"  step {t+1}/{steps} zone0={zt[t]:.0f}C", flush=True)
    return dict(host=HOST, dev=dev, A=A, B=B, X=X, zone=zt, names=np.array(names)), names

def lin_acc(Feat, y, lam=5.0, folds=5, shuffle=False):
    idx = np.where(~np.isnan(y) & ~np.isnan(Feat).any(1))[0]
    if idx.size < 40: return np.nan
    F = Feat[idx]; yy = y[idx].astype(float)
    sd = F.std(0); F = F[:, sd > 1e-9]
    if F.shape[1] == 0: return np.nan
    F = (F - F.mean(0)) / (F.std(0) + 1e-9)
    if shuffle:
        rng = np.random.default_rng(3); F = F[rng.permutation(len(F))]
    F = np.column_stack([F, np.ones(len(F))]); n = len(idx); bs = max(1, n//folds); pr = np.full(n, np.nan)
    for k in range(folds):
        te = np.zeros(n, bool); te[k*bs:(k+1)*bs if k < folds-1 else n] = True; tr = ~te
        if tr.sum() < 10 or te.sum() < 2: continue
        M = F[tr]; W = np.linalg.solve(M.T@M + lam*np.eye(M.shape[1]), M.T@yy[tr]); pr[te] = F[te]@W
    m = ~np.isnan(pr)
    return float(((pr[m] > 0.5).astype(int) == yy[m].astype(int)).mean()) if m.sum() >= 5 else np.nan

def analyze(rec):
    X = rec["X"]; A = rec["A"].astype(float); B = rec["B"].astype(float); names = rec["names"]
    valid = ~np.isnan(X).any(1)
    Xc = X[valid]; Ac = A[valid]; Bc = B[valid]
    # per-channel intrinsic interaction: residual of additive fit vs A*B
    DES = np.column_stack([np.ones(len(Ac)), Ac, Bc])
    inter = []
    for j in range(Xc.shape[1]):
        y = Xc[:, j]
        if y.std() < 1e-9: continue
        beta, *_ = np.linalg.lstsq(DES, y, rcond=None)
        resid = y - DES @ beta
        ab = (Ac*Bc) - (Ac*Bc).mean()
        denom = (resid.std() * ab.std())
        c = float((resid*ab).mean()/denom) if denom > 1e-12 else 0.0
        inter.append((names[j], round(c, 3)))
    inter.sort(key=lambda kv: -abs(kv[1]))
    # tasks on (A,B), single-step: telemetry-LINEAR vs raw-[A,B]-linear vs shuffle
    def mk(op):
        if op == "XOR": return (A.astype(int) ^ B.astype(int)).astype(float)
        if op == "AND": return (A.astype(int) & B.astype(int)).astype(float)
        if op == "OR":  return (A.astype(int) | B.astype(int)).astype(float)
    AB = np.column_stack([A, B])
    out = {"n_valid": int(valid.sum()),
           "top_interaction_channels": inter[:10],
           "median_abs_interaction": round(float(np.median([abs(c) for _, c in inter])), 3) if inter else None}
    for op in ["XOR", "AND", "OR"]:
        y = mk(op)
        out[op] = {"telem_linear": round(lin_acc(X, y), 3),
                   "rawAB_linear": round(lin_acc(AB, y), 3),
                   "shuffle_null": round(lin_acc(X, y, shuffle=True), 3),
                   "chance": round(float(max(np.nanmean(y), 1-np.nanmean(y))), 3)}
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--base", type=float, default=0.06)
    ap.add_argument("--gap", type=float, default=0.10)
    ap.add_argument("--gpu_n", type=int, default=768)
    ap.add_argument("--cpu_n", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rec, names = record(a.steps, a.base, a.gpu_n, a.cpu_n, a.gap, a.seed)
    OUT = P0.OUT; p = OUT/f"dual_input_nonlin_{HOST}.npz"
    np.savez_compressed(p.with_name(p.stem+".tmp.npz"), **rec); os.replace(p.with_name(p.stem+".tmp.npz"), p)
    res = {"host": HOST, "dev": rec["dev"], "steps": a.steps, "D": int(rec["X"].shape[1]), **analyze(rec)}
    jp = OUT/f"dual_input_nonlin_{HOST}.json"; jp.write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2), flush=True)
    x = res["XOR"]
    print(f"\n[{HOST}] DUAL-INPUT VERDICT:", flush=True)
    print(f"  median |interaction|={res['median_abs_interaction']}; top: {res['top_interaction_channels'][:4]}")
    print(f"  XOR(A,B): telem-LINEAR={x['telem_linear']} vs raw-[A,B]-linear={x['rawAB_linear']} "
          f"shuffle={x['shuffle_null']} chance={x['chance']}")
    print(f"  -> SILICON computes the gate if telem-LINEAR ≫ raw-[A,B]-linear & ≫ shuffle. saved {jp}")

if __name__ == "__main__":
    main()
