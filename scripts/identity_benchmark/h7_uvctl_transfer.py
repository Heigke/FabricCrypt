"""H7 controlled-u·v transfer verdict — intra (same die, SHARED baseline) vs inter, the decisive read.

With the shared fixed baseline + temp-lock + DVFS-pin + N-epoch averaging + bin-aware cores, the feature should
finally reproduce. Compares the rich feature [fast,slow,diff]×[Auv,Au2v,Auv2]×ch across runs. Honest verdict.
"""
from __future__ import annotations
import glob, itertools, json
from pathlib import Path
import numpy as np
OUT = Path(__file__).resolve().parents[2] / "results/IDENTITY_H7_2026-06-09"

def load(p):
    d = np.load(p)
    return np.concatenate([d["fast"].ravel(), d["slow"].ravel(), d["diff"].ravel()])

def cos(a,b):
    a=a-a.mean(); b=b-b.mean()
    return float(a@b/((np.linalg.norm(a)*np.linalg.norm(b))+1e-12))

def main():
    runs={}
    for p in sorted(glob.glob(str(OUT/"uvctl_*_r*.npz"))):
        name=Path(p).stem.replace("uvctl_","")
        runs[name]=load(p)
    print("runs:", list(runs))
    if len(runs)<2: print("need >=2 runs"); return
    die={}
    for k in runs: die.setdefault(k.split("_")[0],[]).append(k)
    print("dies:", {d:v for d,v in die.items()})
    intra=[]; intra_lab=[]
    for d,ks in die.items():
        for a,b in itertools.combinations(ks,2): intra.append(cos(runs[a],runs[b])); intra_lab.append(f"{a}~{b}")
    inter=[]; inter_lab=[]
    dl=list(die)
    for i in range(len(dl)):
        for j in range(i+1,len(dl)):
            for a in die[dl[i]]:
                for b in die[dl[j]]: inter.append(cos(runs[a],runs[b])); inter_lab.append(f"{a}~{b}")
    out={"intra":[round(x,3) for x in intra],"intra_lab":intra_lab,
         "inter":[round(x,3) for x in inter],"inter_lab":inter_lab}
    if intra:
        out["mean_intra"]=float(np.mean(intra)); out["min_intra"]=float(min(intra))
    if inter:
        out["mean_inter"]=float(np.mean(inter)); out["max_inter"]=float(max(inter))
    if intra and inter:
        out["gap"]=out["mean_intra"]-out["mean_inter"]
        out["clean_sep"]=bool(min(intra)>max(inter))
        repro = out["mean_intra"]>0.7
        out["VERDICT"]=("REAL die-specific reproducible u·v — DREAM ACHIEVED (controlled)" if (repro and out["clean_sep"]) else
                        ("reproducible now (intra>0.7) but not die-separating" if repro else
                         "STILL not reproducible (intra<=0.7) even fully controlled — death EARNED"))
    (OUT/"uvctl_transfer.json").write_text(json.dumps(out,indent=2))
    print(json.dumps(out,indent=2))

if __name__=="__main__":
    main()
