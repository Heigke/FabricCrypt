"""H7 fresh-vs-stale steering — does an OLD (copied) fingerprint drive DIFFERENT LLM behavior
than the LIVE one? And can we DIMENSION how fast a stale copy 'breaks' the persona?

Built on REAL measured time series (signal_drift_{host}.npz), not simulation. We split the
fingerprint into two blocks, guided by the drift measurement:
  IDENTITY (stable, per-die, copyable-but-die-specific):
     - CPPC fused binning (cosΔ=0)            weight wC
     - Vcore DIRECTION unit(raw) (cosΔ~1e-3)  weight wV
  FRESHNESS (drifts second-to-second -> a replay goes stale):
     - z-scored Vcore jitter (cosΔ up to ~1)  weight wF   <- the knob we sweep
     - thermal jitter (if >1 zone)            folded into freshness

fp(t) = concat( identity(stable) , freshness(t) ).  We steer a frozen GPT-2 with v=alpha*unit(P@fp).
  fp_fresh = fp(last sample) ; fp_stale = fp(first sample)   [the two are ~150 s apart, REAL]
We measure behavioral divergence (first-diverging token, fraction of differing tokens) between
fresh and stale generations, and sweep wF to show the dimensioning trade-off:
  wF=0  -> identity only -> fresh==stale (control: a copy is NOT detectable)
  wF>0  -> stale drifts away from fresh -> persona changes (copy detectable / 'breaks')

HONEST LIMIT: this makes a STALE/REPLAYED snapshot diverge, and a FOREIGN die diverge (different
identity block). It does NOT stop a root attacker on THIS machine reading the live values fresh.
That residual is covered by the TPM-sealed-secret factor (h7_unfakeable_steer.py), not by drift.

Usage: python h7_fresh_steer.py --host ikaros --wf 0,0.25,0.5,1.0
Out: results/IDENTITY_H7_2026-06-09/fresh_steer_{host}.json
"""
from __future__ import annotations
import argparse, json, socket
from pathlib import Path
import numpy as np, torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

OUT=Path(__file__).resolve().parents[2]/"results/IDENTITY_H7_2026-06-09"
DEV="cuda" if torch.cuda.is_available() else "cpu"
PROMPTS=["When I wake, I","My purpose is to","The data arrives and I"]

def z(v):
    v=np.asarray(v,float); s=v.std(); return (v-v.mean())/(s+1e-9) if s>1e-12 else v*0.0
def ue(v,w):
    v=np.asarray(v,float); n=np.linalg.norm(v); return (v/n)*np.sqrt(w) if n>1e-12 else v
def unit(v):
    v=np.asarray(v,float); n=np.linalg.norm(v); return v/n if n>1e-12 else v

def build_fp(npz, t, wC, wV, wF, wT):
    """fp at sample index t. identity blocks use t but are ~constant; freshness uses t."""
    parts=[]
    if "cppc" in npz:      parts.append(ue(z(npz["cppc"][t]), wC))       # stable identity
    if "vcore_raw" in npz: parts.append(ue(unit(npz["vcore_raw"][t]),wV))# stable identity (direction)
    if "vcore_z" in npz:   parts.append(ue(npz["vcore_z"][t], wF))       # FRESHNESS (drifts)
    if wT>0 and "thermal" in npz and npz["thermal"].shape[1]>1:
        parts.append(ue(z(npz["thermal"][t]), wT))                       # FRESHNESS (thermal)
    return np.concatenate(parts).astype(np.float32)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--host",default=socket.gethostname())
    ap.add_argument("--wf",default="0,0.25,0.5,1.0",help="freshness weights to sweep")
    ap.add_argument("--wc",type=float,default=0.6); ap.add_argument("--wv",type=float,default=0.4)
    ap.add_argument("--wt",type=float,default=0.0); ap.add_argument("--alpha",type=float,default=6.0)
    a=ap.parse_args()
    npz=dict(np.load(OUT/f"signal_drift_{a.host}.npz"))
    nT=min(v.shape[0] for v in npz.values())
    t0, tN = 0, nT-1
    print(f"[{a.host}] {nT} samples; comparing t0 vs t{tN} (~{(tN)*3}s apart, REAL measured)")

    tok=GPT2TokenizerFast.from_pretrained("gpt2"); lm=GPT2LMHeadModel.from_pretrained("gpt2").to(DEV).eval()
    for p in lm.parameters(): p.requires_grad=False
    d=lm.config.n_embd

    steer={"v":None}
    def hook(m,i,o):
        if steer["v"] is None: return o
        return (o[0]+steer["v"],)+o[1:] if isinstance(o,tuple) else o+steer["v"]
    hooks=[blk.register_forward_hook(hook) for blk in lm.transformer.h]

    @torch.no_grad()
    def gen(P,fp,prompt,n=40):
        f=torch.tensor(fp,device=DEV); pv=P@f
        steer["v"]=a.alpha*pv/(pv.norm()+1e-9)
        x=tok(prompt,return_tensors="pt").input_ids.to(DEV)
        for _ in range(n): x=torch.cat([x,torch.argmax(lm(x).logits[:,-1,:],-1,keepdim=True)],1)
        steer["v"]=None
        ids=x[0].tolist()[ tok(prompt,return_tensors="pt").input_ids.shape[1]: ]
        return tok.decode(x[0],skip_special_tokens=True), ids

    def divergence(ids_a, ids_b):
        L=min(len(ids_a),len(ids_b)); fd=L
        for i in range(L):
            if ids_a[i]!=ids_b[i]: fd=i; break
        diff=sum(1 for i in range(L) if ids_a[i]!=ids_b[i])
        return {"first_diverge":fd,"frac_diff":round(diff/max(L,1),3),"len":L}

    res={"host":a.host,"alpha":a.alpha,"samples":nT,"seconds_apart":(tN)*3,
         "weights":{"wc":a.wc,"wv":a.wv,"wt":a.wt},"sweep":{}}
    wfs=[float(s) for s in a.wf.split(",")]
    for wF in wfs:
        # build P sized to this fp dim (fixed seed -> deterministic transducer)
        fp_fresh=build_fp(npz,tN,a.wc,a.wv,wF,a.wt)
        fp_stale=build_fp(npz,t0,a.wc,a.wv,wF,a.wt)
        g=torch.Generator().manual_seed(0); P=torch.randn(d,len(fp_fresh),generator=g).to(DEV); P=P/P.norm(dim=0,keepdim=True)
        vec_cos=float(1-np.dot(unit(P.cpu().numpy()@fp_fresh),unit(P.cpu().numpy()@fp_stale)))
        per={}
        difs=[]
        for pr in PROMPTS:
            tf,idf=gen(P,fp_fresh,pr); ts,ids=gen(P,fp_stale,pr)
            dv=divergence(idf,ids); difs.append(dv["frac_diff"])
            per[pr]={"fresh":tf,"stale":ts,"div":dv}
        res["sweep"][f"wF={wF}"]={"steer_vec_cosdist_fresh_vs_stale":round(vec_cos,4),
                                  "mean_frac_diff_tokens":round(float(np.mean(difs)),3),
                                  "fp_dim":len(fp_fresh),"per_prompt":per}
        print(f"  wF={wF:<5} vecΔ(fresh,stale)={vec_cos:.3f}  mean token-diff={np.mean(difs):.3f}  (dim {len(fp_fresh)})")
    for h in hooks: h.remove()
    (OUT/f"fresh_steer_{a.host}.json").write_text(json.dumps(res,indent=2))
    print(f"saved fresh_steer_{a.host}.json")
    print("\nINTERPRET: wF=0 -> fresh==stale (a copy is undetectable by drift).")
    print("           wF>0 -> stale snapshot DIVERGES from live -> persona 'breaks' as signal ages.")

if __name__=="__main__": main()
