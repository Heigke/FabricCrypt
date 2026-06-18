"""H7 throttling / destructive-interference -> micro linear-XOR. Two memory streamers on cores that
SHARE an L3, working set sized so one fits in L3 but two spill to DRAM -> mutual eviction -> combined
throughput < single (sum inverts) -> linearly-separable XOR. Optional resctrl MBA throttle amplifies.
Builds micro_mem.c with gcc. Run under sandbox-disabled shell (gcc/exec; sudo only if --mba/--cat).
Out: throttle_nonlin_{host}.json
"""
from __future__ import annotations
import os, sys, time, json, argparse, struct, mmap, subprocess, tempfile
import numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import h7_rt_phase0 as P0
HOST = P0.HOST; HERE = Path(__file__).resolve().parent
SUDO_PW = os.environ.get("IKAROS_SUDO_PW", "Ikaros")

def sudo(cmd_list, inp=None):
    return subprocess.run(["sudo","-S",*cmd_list], input=(inp or (SUDO_PW+"\n")).encode(),
                          capture_output=True)
def curfreq(c):
    try: return int(Path(f"/sys/devices/system/cpu/cpu{c}/cpufreq/scaling_cur_freq").read_text())
    except Exception: return 0

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--win", type=float, default=0.05)
    ap.add_argument("--cpu_a", type=int, default=0)
    ap.add_argument("--cpu_b", type=int, default=2)   # diff physical core, same L3 as cpu0
    ap.add_argument("--mb", type=int, default=24)      # MB per worker array
    ap.add_argument("--mba", type=int, default=0)      # resctrl MBA throttle % (0=off)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    binp = HERE/"micro_mem"; src = HERE/"micro_mem.c"
    if not binp.exists() or src.stat().st_mtime > binp.stat().st_mtime:
        r = subprocess.run(["gcc","-O2","-march=native","-o",str(binp),str(src)],capture_output=True,text=True)
        if r.returncode: print("gcc FAILED:\n"+r.stderr,flush=True); sys.exit(1)
        print("[build] micro_mem compiled",flush=True)
    arr_bytes = a.mb*1024*1024
    # optional resctrl MBA throttle
    resctrl_grp = None
    if a.mba > 0:
        sudo(["mount","-t","resctrl","resctrl","/sys/fs/resctrl"])  # ok if already mounted
        resctrl_grp = "/sys/fs/resctrl/h7throt"
        sudo(["mkdir","-p",resctrl_grp])
        # throttle memory bandwidth to a.mba% on both L3 domains
        try:
            schem = Path("/sys/fs/resctrl/schemata").read_text()
            ndom = schem.count("=") if "MB:" in schem else 2
        except Exception: ndom = 2
        mbline = "MB:" + ";".join(f"{d}={a.mba}" for d in range(2))
        sudo(["bash","-c",f"echo '{mbline}' > {resctrl_grp}/schemata"])
        sudo(["bash","-c",f"echo {a.cpu_a},{a.cpu_b} > {resctrl_grp}/cpus_list"])
        print(f"[resctrl] MBA throttled to {a.mba}% on cpus {a.cpu_a},{a.cpu_b}",flush=True)

    shm = Path(tempfile.gettempdir())/f"h7throtshm_{os.getpid()}"; shm.write_bytes(b"\x00"*64)
    fd = os.open(str(shm), os.O_RDWR); mm = mmap.mmap(fd, 64)
    def set_flag(i,v): mm.seek(i*4); mm.write(struct.pack("i",v))
    def set_cnt(i,v): mm.seek(8+i*8); mm.write(struct.pack("Q",v))
    def get_cnt(i): mm.seek(8+i*8); return struct.unpack("Q", mm.read(8))[0]
    l3a = Path(f"/sys/devices/system/cpu/cpu{a.cpu_a}/cache/index3/shared_cpu_list").read_text().strip()
    l3b = Path(f"/sys/devices/system/cpu/cpu{a.cpu_b}/cache/index3/shared_cpu_list").read_text().strip()
    print(f"[{HOST}] throttle-nonlin cpu_a={a.cpu_a}(L3 {l3a}) cpu_b={a.cpu_b}(L3 {l3b}) arr={a.mb}MB "
          f"mba={a.mba}% steps={a.steps}", flush=True)
    pA = subprocess.Popen([str(binp),str(a.cpu_a),"0",str(shm),str(arr_bytes)])
    pB = subprocess.Popen([str(binp),str(a.cpu_b),"1",str(shm),str(arr_bytes)])
    time.sleep(0.5)
    rng = np.random.default_rng(a.seed)
    A = rng.integers(0,2,a.steps).astype(np.int8); B = rng.integers(0,2,a.steps).astype(np.int8)
    rows=[]; pwr=lambda: float(P0.read_gpu_power()[0]) if P0.read_gpu_power().size else 0.0
    try:
        for t in range(a.steps):
            set_cnt(0,0); set_cnt(1,0); set_flag(0,int(A[t])); set_flag(1,int(B[t]))
            time.sleep(a.win)
            tpA=get_cnt(0); tpB=get_cnt(1); set_flag(0,0); set_flag(1,0)
            rows.append([tpA,tpB,curfreq(a.cpu_a),curfreq(a.cpu_b),pwr(),P0.zone0()])
            if (t+1)%100==0: print(f"  step {t+1}/{a.steps} tpA={tpA} tpB={tpB}",flush=True)
    finally:
        set_flag(0,2); set_flag(1,2); time.sleep(0.2); pA.terminate(); pB.terminate()
        mm.close(); os.close(fd)
        try: shm.unlink()
        except Exception: pass
        if resctrl_grp: sudo(["rmdir",resctrl_grp]); print("[resctrl] group removed",flush=True)
    X=np.array(rows,float); names=["tp_A","tp_B","freq_A","freq_B","power","zone0"]
    Ac=A.astype(float); Bc=B.astype(float); DES=np.column_stack([np.ones(len(Ac)),Ac,Bc]); inter=[]
    for j in range(X.shape[1]):
        y=X[:,j]
        if y.std()<1e-9: continue
        beta,*_=np.linalg.lstsq(DES,y,rcond=None); resid=y-DES@beta
        ab=(Ac*Bc)-(Ac*Bc).mean(); den=resid.std()*ab.std()
        inter.append((names[j],round(float((resid*ab).mean()/den),3) if den>1e-12 else 0.0))
    inter.sort(key=lambda kv:-abs(kv[1]))
    def lin(F,y,shuffle=False,lam=2.0,folds=5):
        F=F.astype(float); sd=F.std(0); F=F[:,sd>1e-9]
        if F.shape[1]==0: return np.nan
        F=(F-F.mean(0))/(F.std(0)+1e-9)
        if shuffle: F=F[np.random.default_rng(7).permutation(len(F))]
        F=np.column_stack([F,np.ones(len(F))]); n=len(y); bs=max(1,n//folds); pr=np.full(n,np.nan)
        for k in range(folds):
            te=np.zeros(n,bool); te[k*bs:(k+1)*bs if k<folds-1 else n]=True; tr=~te
            M=F[tr]; W=np.linalg.solve(M.T@M+lam*np.eye(M.shape[1]),M.T@y[tr].astype(float)); pr[te]=F[te]@W
        m=~np.isnan(pr); return float(((pr[m]>0.5).astype(int)==y[m].astype(int)).mean())
    xor=(A.astype(int)^B.astype(int)).astype(float); AB=np.column_stack([A,B]).astype(float)
    tot=X[:,0]+X[:,1]
    cells={f"{av}{bv}":round(float(tot[(A==av)&(B==bv)].mean()),1) for av in(0,1) for bv in(0,1)}
    solo=(cells.get("10",0)+cells.get("01",0))/2 if (cells.get("10") or cells.get("01")) else 0
    res={"host":HOST,"steps":a.steps,"arr_MB":a.mb,"mba_pct":a.mba,"channels":names,
         "throughput_sum_by_AB":cells,
         "per_thread_slowdown_when_both":round(1-(cells.get("11",0)/2)/solo,3) if solo else None,
         "sum_inverts(11<single)": cells.get("11",9e9) < max(cells.get("10",0),cells.get("01",0)),
         "interaction":inter,
         "XOR":{"readout_linear":round(lin(X,xor),3),"rawAB_linear":round(lin(AB,xor),3),
                "shuffle_null":round(lin(X,xor,shuffle=True),3),"chance":round(float(max(xor.mean(),1-xor.mean())),3)}}
    jp=P0.OUT/f"throttle_nonlin_{HOST}.json"; jp.write_text(json.dumps(res,indent=2))
    print(json.dumps(res,indent=2),flush=True)
    x=res["XOR"]
    print(f"\n[{HOST}] THROTTLE VERDICT (L3/bandwidth destructive interference):")
    print(f"  tp-sum by(A,B)={cells} slowdown_both={res['per_thread_slowdown_when_both']} "
          f"sum_inverts={res['sum_inverts(11<single)']}")
    print(f"  XOR: readout-LIN={x['readout_linear']} raw={x['rawAB_linear']} shuffle={x['shuffle_null']} chance={x['chance']}")
    print(f"  saved {jp}")

if __name__=="__main__":
    main()
