"""H7 LOW-LEVEL substrate nonlinearity — Eric: go below high-level resource contention, into the
firmware/DVFS layer. Same methodology (2 independent inputs -> A*B interaction -> linear XOR) but the
inputs are now clean FIRMWARE knobs: per-core-group FREQUENCY CAPS (amd-pstate scaling_max_freq).
The nonlinearity, if any, is the SMU power-arbitration firmware deciding realized clocks/voltages
under one shared package budget -> "the firmware computes."

Inputs A,B in {f_lo, f_hi}: A caps CPU core-group 0 (cores 0..H-1), B caps group 1 (cores H..N-1).
A constant moderate all-core load makes the caps BIND (idle cores ignore caps). Read ~50-D telemetry
(realized per-core freq, voltage, power, thermal) -> test per-channel A*B interaction + LINEAR XOR(A,B).

SAFETY: writes go through `sudo tee` to standard amd-pstate sysfs (driver-mediated, reversible).
NEVER touches SMU mailbox (C2PMSG) or amdgpu_regs_didt. Original scaling_max_freq is saved and
restored on exit (finally + atexit + SIGTERM). Lowering freq REDUCES heat (thermally safer).

Out: lowlevel_nonlin_{host}.json. Run under a sandbox-disabled shell (needs sudo).
"""
from __future__ import annotations
import os, sys, time, json, argparse, atexit, signal, subprocess, threading
import numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import h7_rt_phase0 as P0
from h7_telemetry_reservoir import snap_vec

HOST = P0.HOST
SUDO_PW = os.environ.get("IKAROS_SUDO_PW", "Ikaros")
NCPU = os.cpu_count() or 16
CPUDIR = "/sys/devices/system/cpu"

def sudo_write(path, value):
    subprocess.run(["sudo", "-S", "tee", path], input=f"{value}\n".encode(),
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                   env={**os.environ, "SUDO_ASKPASS": ""})

def sudo_write_many(paths, value):
    p = subprocess.Popen(["sudo", "-S", "tee", *paths], stdin=subprocess.PIPE,
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    p.communicate(input=f"{value}\n".encode())

def read_int(path):
    try: return int(Path(path).read_text())
    except Exception: return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--settle", type=float, default=0.10)
    ap.add_argument("--gap", type=float, default=0.04)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    # prime sudo (cached ~15 min)
    subprocess.run(["sudo", "-S", "-v"], input=f"{SUDO_PW}\n".encode(),
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    maxf = {c: f"{CPUDIR}/cpu{c}/cpufreq/scaling_max_freq" for c in range(NCPU)}
    orig = {c: read_int(maxf[c]) for c in range(NCPU)}
    f_lo = read_int(f"{CPUDIR}/cpu0/cpufreq/cpuinfo_min_freq") or 599000
    f_hi = orig[0] or 3000000
    H = NCPU // 2
    gA = [maxf[c] for c in range(0, H)]; gB = [maxf[c] for c in range(H, NCPU)]

    def restore():
        for c in range(NCPU):
            if orig[c]: sudo_write(maxf[c], orig[c])
    atexit.register(restore)
    signal.signal(signal.SIGTERM, lambda *_: (restore(), sys.exit(0)))

    # constant moderate all-core load so caps bind
    stop = {"v": False}
    def loadgen():
        X = np.random.randn(512, 512); Y = np.random.randn(512, 512)
        while not stop["v"]:
            X = np.tanh(X @ Y)
    workers = [threading.Thread(target=loadgen, daemon=True) for _ in range(max(2, NCPU//2))]
    for w in workers: w.start()

    rng = np.random.default_rng(a.seed)
    A = rng.integers(0, 2, a.steps).astype(np.int8); B = rng.integers(0, 2, a.steps).astype(np.int8)
    samp = P0.Sampler(period=0.0); x0, names = snap_vec(samp); D = x0.size
    X = np.full((a.steps, D), np.nan); zt = np.zeros(a.steps)
    print(f"[{HOST}] lowlevel-nonlin steps={a.steps} f_lo={f_lo} f_hi={f_hi} H={H} D={D}", flush=True)
    try:
        for t in range(a.steps):
            if P0.zone0() >= 82: P0.wait_cool(55)
            sudo_write_many(gA, f_hi if A[t] else f_lo)
            sudo_write_many(gB, f_hi if B[t] else f_lo)
            time.sleep(a.settle)
            v, _ = snap_vec(samp)
            if v.size == D: X[t] = v
            time.sleep(a.gap); zt[t] = P0.zone0()
            if (t+1) % 50 == 0: print(f"  step {t+1}/{a.steps} zone0={zt[t]:.0f}C", flush=True)
    finally:
        stop["v"] = True; restore()
        print("[restore] scaling_max_freq restored", flush=True)

    # analyze: per-channel A*B interaction + LINEAR XOR(A,B)
    valid = ~np.isnan(X).any(1); Xc = X[valid]; Ac = A[valid].astype(float); Bc = B[valid].astype(float)
    DES = np.column_stack([np.ones(len(Ac)), Ac, Bc]); inter = []
    for j in range(Xc.shape[1]):
        y = Xc[:, j]
        if y.std() < 1e-9: continue
        beta, *_ = np.linalg.lstsq(DES, y, rcond=None); resid = y - DES @ beta
        ab = (Ac*Bc) - (Ac*Bc).mean(); den = resid.std()*ab.std()
        inter.append((str(names[j]), round(float((resid*ab).mean()/den), 3) if den > 1e-12 else 0.0))
    inter.sort(key=lambda kv: -abs(kv[1]))

    def lin_xor(Feat, y, shuffle=False, lam=5.0, folds=5):
        idx = np.where(~np.isnan(y) & ~np.isnan(Feat).any(1))[0]
        if idx.size < 40: return np.nan
        F = Feat[idx]; yy = y[idx].astype(float); sd = F.std(0); F = F[:, sd > 1e-9]
        if F.shape[1] == 0: return np.nan
        F = (F-F.mean(0))/(F.std(0)+1e-9)
        if shuffle: F = F[np.random.default_rng(4).permutation(len(F))]
        F = np.column_stack([F, np.ones(len(F))]); n=len(idx); bs=max(1,n//folds); pr=np.full(n,np.nan)
        for k in range(folds):
            te=np.zeros(n,bool); te[k*bs:(k+1)*bs if k<folds-1 else n]=True; tr=~te
            if tr.sum()<10 or te.sum()<2: continue
            M=F[tr]; W=np.linalg.solve(M.T@M+lam*np.eye(M.shape[1]),M.T@yy[tr]); pr[te]=F[te]@W
        m=~np.isnan(pr); return float(((pr[m]>0.5).astype(int)==yy[m].astype(int)).mean()) if m.sum()>=5 else np.nan
    xor = (A.astype(int) ^ B.astype(int)).astype(float)
    AB = np.column_stack([A, B]).astype(float)
    res = {"host": HOST, "steps": a.steps, "D": int(D), "n_valid": int(valid.sum()),
           "f_lo": f_lo, "f_hi": f_hi,
           "top_interaction_channels": inter[:10],
           "median_abs_interaction": round(float(np.median([abs(c) for _, c in inter])), 3) if inter else None,
           "XOR": {"telem_linear": round(lin_xor(X, xor), 3),
                   "rawAB_linear": round(lin_xor(AB, xor), 3),
                   "shuffle_null": round(lin_xor(X, xor, shuffle=True), 3),
                   "chance": round(float(max(np.nanmean(xor), 1-np.nanmean(xor))), 3)}}
    jp = P0.OUT/f"lowlevel_nonlin_{HOST}.json"; jp.write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2), flush=True)
    x = res["XOR"]
    print(f"\n[{HOST}] LOWLEVEL VERDICT: median|inter|={res['median_abs_interaction']} top={res['top_interaction_channels'][:4]}")
    print(f"  XOR(freq-cap A,B): telem-LIN={x['telem_linear']} raw[A,B]={x['rawAB_linear']} "
          f"shuffle={x['shuffle_null']} chance={x['chance']}")
    print(f"  -> FIRMWARE/DVFS computes the gate if telem-LIN >> raw & shuffle. saved {jp}")

if __name__ == "__main__":
    main()
