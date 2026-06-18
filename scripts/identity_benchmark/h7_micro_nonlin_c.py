"""H7 micro-nonlin with a COMPILED divider stressor (micro_div.c). Two instances pinned to SMT
siblings serialize on the single shared FP divider -> max sub-additive throughput. Tests whether
that microarchitectural contention is severe enough for a LINEAR readout to compute XOR(A,B).
Control/counters via mmap shm. No sudo. Builds micro_div.c with gcc -O2 -march=native on first run.
Out: micro_nonlin_c_{host}.json
"""
from __future__ import annotations
import os, sys, time, json, argparse, struct, mmap, subprocess, tempfile
import numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import h7_rt_phase0 as P0
HOST = P0.HOST
HERE = Path(__file__).resolve().parent

def curfreq(cpu):
    try: return int(Path(f"/sys/devices/system/cpu/cpu{cpu}/cpufreq/scaling_cur_freq").read_text())
    except Exception: return 0

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--win", type=float, default=0.05)
    ap.add_argument("--cpu_a", type=int, default=0)
    ap.add_argument("--cpu_b", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    binp = HERE/"micro_div"; src = HERE/"micro_div.c"
    if not binp.exists() or src.stat().st_mtime > binp.stat().st_mtime:
        r = subprocess.run(["gcc", "-O2", "-march=native", "-o", str(binp), str(src)],
                           capture_output=True, text=True)
        if r.returncode: print("gcc FAILED:\n"+r.stderr, flush=True); sys.exit(1)
        print("[build] micro_div compiled", flush=True)
    # shm file: 64 bytes
    shm = Path(tempfile.gettempdir())/f"h7_microshm_{os.getpid()}"
    shm.write_bytes(b"\x00"*64)
    fd = os.open(str(shm), os.O_RDWR)
    mm = mmap.mmap(fd, 64)
    def set_flag(idx, v): mm.seek(idx*4); mm.write(struct.pack("i", v))
    def get_cnt(idx): mm.seek(8+idx*8); return struct.unpack("Q", mm.read(8))[0]
    def set_cnt(idx, v): mm.seek(8+idx*8); mm.write(struct.pack("Q", v))

    sib = Path(f"/sys/devices/system/cpu/cpu{a.cpu_a}/topology/thread_siblings_list").read_text().strip()
    print(f"[{HOST}] micro-nonlin-C SMT cpu_a={a.cpu_a} cpu_b={a.cpu_b} (siblings={sib}) steps={a.steps}", flush=True)
    pA = subprocess.Popen([str(binp), str(a.cpu_a), "0", str(shm)])
    pB = subprocess.Popen([str(binp), str(a.cpu_b), "1", str(shm)])
    time.sleep(0.3)
    rng = np.random.default_rng(a.seed)
    A = rng.integers(0, 2, a.steps).astype(np.int8); B = rng.integers(0, 2, a.steps).astype(np.int8)
    rows = []
    pwr = lambda: float(P0.read_gpu_power()[0]) if P0.read_gpu_power().size else 0.0
    try:
        for t in range(a.steps):
            set_cnt(0, 0); set_cnt(1, 0)
            set_flag(0, int(A[t])); set_flag(1, int(B[t]))
            time.sleep(a.win)
            tpA = get_cnt(0); tpB = get_cnt(1)
            set_flag(0, 0); set_flag(1, 0)
            rows.append([tpA, tpB, curfreq(a.cpu_a), curfreq(a.cpu_b), pwr(), P0.zone0()])
            if (t+1) % 100 == 0: print(f"  step {t+1}/{a.steps} tpA={tpA} tpB={tpB}", flush=True)
    finally:
        set_flag(0, 2); set_flag(1, 2); time.sleep(0.2)
        pA.terminate(); pB.terminate(); mm.close(); os.close(fd)
        try: shm.unlink()
        except Exception: pass
    X = np.array(rows, float); names = ["tp_A","tp_B","freq_A","freq_B","power","zone0"]
    Ac = A.astype(float); Bc = B.astype(float)
    DES = np.column_stack([np.ones(len(Ac)), Ac, Bc]); inter = []
    for j in range(X.shape[1]):
        y = X[:, j]
        if y.std() < 1e-9: continue
        beta,*_ = np.linalg.lstsq(DES, y, rcond=None); resid = y-DES@beta
        ab=(Ac*Bc)-(Ac*Bc).mean(); den=resid.std()*ab.std()
        inter.append((names[j], round(float((resid*ab).mean()/den),3) if den>1e-12 else 0.0))
    inter.sort(key=lambda kv:-abs(kv[1]))
    def lin(F, y, shuffle=False, lam=2.0, folds=5):
        F=F.astype(float); sd=F.std(0); F=F[:,sd>1e-9]
        if F.shape[1]==0: return np.nan
        F=(F-F.mean(0))/(F.std(0)+1e-9)
        if shuffle: F=F[np.random.default_rng(6).permutation(len(F))]
        F=np.column_stack([F,np.ones(len(F))]); n=len(y); bs=max(1,n//folds); pr=np.full(n,np.nan)
        for k in range(folds):
            te=np.zeros(n,bool); te[k*bs:(k+1)*bs if k<folds-1 else n]=True; tr=~te
            M=F[tr]; W=np.linalg.solve(M.T@M+lam*np.eye(M.shape[1]),M.T@y[tr].astype(float)); pr[te]=F[te]@W
        m=~np.isnan(pr); return float(((pr[m]>0.5).astype(int)==y[m].astype(int)).mean())
    xor=(A.astype(int)^B.astype(int)).astype(float); AB=np.column_stack([A,B]).astype(float)
    tot=X[:,0]+X[:,1]
    cells={f"{av}{bv}": round(float(tot[(A==av)&(B==bv)].mean()),1) for av in (0,1) for bv in (0,1)}
    solo=(cells.get("10",0)+cells.get("01",0))/2 if (cells.get("10") or cells.get("01")) else 0
    res={"host":HOST,"steps":a.steps,"channels":names,"throughput_sum_by_AB":cells,
         "per_thread_slowdown_when_both": round(1-(cells.get("11",0)/2)/solo,3) if solo else None,
         "sum_inverts(11<10)": cells.get("11",9e9) < max(cells.get("10",0),cells.get("01",0)),
         "interaction":inter,
         "XOR":{"readout_linear":round(lin(X,xor),3),"rawAB_linear":round(lin(AB,xor),3),
                "shuffle_null":round(lin(X,xor,shuffle=True),3),"chance":round(float(max(xor.mean(),1-xor.mean())),3)}}
    jp=P0.OUT/f"micro_nonlin_c_{HOST}.json"; jp.write_text(json.dumps(res,indent=2))
    print(json.dumps(res,indent=2),flush=True)
    x=res["XOR"]
    print(f"\n[{HOST}] MICRO-C VERDICT (compiled SMT divider contention):")
    print(f"  tp-sum by(A,B)={cells}  per-thread slowdown when both={res['per_thread_slowdown_when_both']} "
          f"sum_inverts={res['sum_inverts(11<10)']}")
    print(f"  XOR: readout-LIN={x['readout_linear']} raw={x['rawAB_linear']} shuffle={x['shuffle_null']} chance={x['chance']}")
    print(f"  saved {jp}")

if __name__=="__main__":
    main()
