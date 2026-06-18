#!/bin/bash
# H7 full campaign: (A) clean system-id retest (matched start-temp) + (B) criticality die-test.
set +e
cd /home/ikaros/Documents/claude_hive/AMD_gfx1151_energy
DPASS=daedalus; DH=daedalus@daedalus.local
RB="results/IDENTITY_H7_2026-06-09"

echo "##### PART A: system-id retest (matched TSTART=48, 5 cycles) #####"
for tag in a1 a2; do
  echo "=== ikaros $tag ==="
  echo $DPASS | sudo -S env RUNTAG=$tag TSTART=48 TMAX=85 CYCLES=5 TON=30 TOFF=38 PIN_MHZ=2800 \
    venv/bin/python scripts/identity_benchmark/h7_system_id.py 2>&1 | grep -E "channels|cycle|saved"
done
sshpass -p $DPASS scp -o StrictHostKeyChecking=no scripts/identity_benchmark/h7_system_id.py $DH:/tmp/ 2>/dev/null
for tag in a1 a2; do
  echo "=== daedalus $tag ==="
  sshpass -p $DPASS ssh -o StrictHostKeyChecking=no $DH \
   "echo $DPASS | sudo -S env H7_OUT=/tmp/h7sid RUNTAG=$tag TSTART=48 TMAX=85 CYCLES=5 TON=30 TOFF=38 PIN_MHZ=2800 /home/daedalus/venvs/torch-rocm/bin/python /tmp/h7_system_id.py 2>&1 | grep -E 'channels|cycle|saved'" 2>/dev/null
  sshpass -p $DPASS scp -o StrictHostKeyChecking=no $DH:/tmp/h7sid/sysid_daedalus_$tag.npz $RB/ 2>/dev/null
done
echo "=== SYSTEM-ID COMPARE (matched start-temp) ==="
RUNTAG=a1 venv/bin/python - <<'PY'
import numpy as np, glob
from pathlib import Path
import sys; sys.path.insert(0,'scripts/identity_benchmark')
import importlib.util
spec=importlib.util.spec_from_file_location("sid","scripts/identity_benchmark/h7_system_id.py")
# reuse compare() over all a1,a2 files
OUT=Path("results/IDENTITY_H7_2026-06-09")
def feats(f):
    d=np.load(f,allow_pickle=True); A=d["data"]; lab=d["labels"]; names=list(d["names"]); fs=float(d["fs"])
    X=A[:,1:]; on=np.where(np.diff(lab.astype(int))==1)[0]; L=int(fs*25)
    common=names
    # step-response SHAPE signature on top-var channels
    var=np.nanvar(X,0); order=[names[i] for i in np.argsort(-var)]
    return A,lab,names,fs,order
def z(v): return (v-np.nanmean(v))/(np.nanstd(v)+1e-9)
def cos(a,b):
    a=np.nan_to_num(a);b=np.nan_to_num(b);return float(a@b/(np.linalg.norm(a)*np.linalg.norm(b)+1e-12))
runs={}
for h in ["ikaros","daedalus"]:
    for t in ["a1","a2"]:
        p=OUT/f"sysid_{h}_{t}.npz"
        if p.exists(): runs[(h,t)]=feats(p)
if len(runs)<4: print("missing runs",list(runs)); raise SystemExit
# common channels
common=set(runs[("ikaros","a1")][2])
for v in runs.values(): common&=set(v[2])
A0=runs[("ikaros","a1")]; var=np.nanvar(A0[0][:,1:],0)
order=[A0[2][i] for i in np.argsort(-var) if A0[2][i] in common][:24]
def sig(v):
    A,lab,names,fs,_=v; X=A[:,1:]; on=np.where(np.diff(lab.astype(int))==1)[0]; L=int(fs*25)
    out=[]
    for c in order:
        j=names.index(c); col=X[:,j]; segs=[col[o:o+L] for o in on if o+L<=len(col) and np.isfinite(col[o:o+L]).all()]
        if not segs: out.append(np.zeros(40)); continue
        m=np.mean(segs,0)-np.mean(segs,0)[0]; idx=np.linspace(0,len(m)-1,40).astype(int); s=m[idx]; s=s/(np.max(np.abs(s))+1e-9); out.append(s)
    return np.concatenate(out)
S={k:sig(v) for k,v in runs.items()}
print(f"INTRA ikaros   = {cos(S[('ikaros','a1')],S[('ikaros','a2')]):+.3f}")
print(f"INTRA daedalus = {cos(S[('daedalus','a1')],S[('daedalus','a2')]):+.3f}")
inter=np.mean([cos(S[('ikaros',a)],S[('daedalus',b)]) for a in['a1','a2'] for b in['a1','a2']])
print(f"INTER          = {inter:+.3f}")
PY

echo ""
echo "##### PART B: criticality die-test (rho=2.0 edge, closed/open/shuffle) #####"
sshpass -p $DPASS scp -o StrictHostKeyChecking=no scripts/identity_benchmark/h7_criticality_loop.py $DH:/tmp/ 2>/dev/null
for mode in closed open shuffle; do
  for tag in r1 r2; do
    echo $DPASS | sudo -S env MODE=$mode RHO=2.0 STEPS=250 DMAX=0.05 GAIN=2.0 RUNTAG=$tag TMAX=85 \
      venv/bin/python scripts/identity_benchmark/h7_criticality_loop.py 2>&1 | grep -E "mode=|no pm"
    sshpass -p $DPASS ssh -o StrictHostKeyChecking=no $DH \
      "echo $DPASS | sudo -S env H7_OUT=/tmp/h7crit MODE=$mode RHO=2.0 STEPS=250 DMAX=0.05 GAIN=2.0 RUNTAG=$tag TMAX=85 /home/daedalus/venvs/torch-rocm/bin/python /tmp/h7_criticality_loop.py 2>&1 | grep -E 'mode=|no pm'" 2>/dev/null
    sshpass -p $DPASS scp -o StrictHostKeyChecking=no $DH:/tmp/h7crit/crit_daedalus_${mode}_${tag}.npz $RB/ 2>/dev/null
  done
done
echo "=== CRITICALITY COMPARE: does closing the loop amplify die-separation? ==="
venv/bin/python - <<'PY'
import numpy as np, glob
from pathlib import Path
OUT=Path("results/IDENTITY_H7_2026-06-09")
def feat(f):
    d=np.load(f); X=d["X"]; L=d["L"]; T=d["T"]
    # trajectory fingerprint: per-dim mean,std,lag1-autocorr of X + load/T dynamics (offset-removed)
    xm=X.mean(0); xs=X.std(0)
    ac=np.array([np.corrcoef(X[:-1,i],X[1:,i])[0,1] if X[:,i].std()>1e-9 else 0 for i in range(X.shape[1])])
    Tz=(T-T.mean())/(T.std()+1e-9)
    return np.concatenate([xm,xs,np.nan_to_num(ac),[L.mean(),L.std()]])
def cos(a,b):
    a=np.nan_to_num(a);b=np.nan_to_num(b);return float(a@b/(np.linalg.norm(a)*np.linalg.norm(b)+1e-12))
for mode in ["closed","open","shuffle"]:
    F={}
    for h in ["ikaros","daedalus"]:
        for t in ["r1","r2"]:
            p=OUT/f"crit_{h}_{mode}_{t}.npz"
            if p.exists(): F[(h,t)]=feat(p)
    if len(F)<4: print(f"  {mode}: missing {set([(h,t) for h in['ikaros','daedalus'] for t in['r1','r2']])-set(F)}"); continue
    ii=cos(F[('ikaros','r1')],F[('ikaros','r2')]); idd=cos(F[('daedalus','r1')],F[('daedalus','r2')])
    inter=np.mean([cos(F[('ikaros',a)],F[('daedalus',b)]) for a in['r1','r2'] for b in['r1','r2']])
    sep=(ii+idd)/2 - inter
    # lyap mean
    ly=np.mean([np.load(OUT/f"crit_{h}_{mode}_{t}.npz")["lyap"] for h in["ikaros","daedalus"] for t in["r1","r2"] if (OUT/f"crit_{h}_{mode}_{t}.npz").exists()])
    print(f"  {mode:8s}: INTRA ik={ii:+.2f} da={idd:+.2f}  INTER={inter:+.2f}  SEPARATION(intra-inter)={sep:+.2f}  lyap={ly:+.3f}")
print("\n  -> if SEPARATION(closed) >> SEPARATION(open): criticality AMPLIFIES die identity (the win).")
print("  -> if closed ~ open: no amplification; if shuffle high-sep: artefact not physical loop.")
PY
echo "=== restore boost ==="; echo $DPASS | sudo -S tee /sys/devices/system/cpu/cpufreq/boost <<<1 >/dev/null 2>&1
echo "##### CAMPAIGN DONE temp=$(($(cat /sys/class/thermal/thermal_zone0/temp)/1000))C #####"
