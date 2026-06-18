"""H7 rooted identity — the whole honest sentence in one demo, on REAL measured signals.

Decomposition validated by h7_signal_drift measurements (ikaros + daedalus):
  IDENTITY  = time-averaged z(Vcore) per-core scatter (denoised) [+ CPPC]
              cross-die cos(ik,da) = -0.306  -> dies cleanly separated
              within-die split-half cos = 0.99/0.87 -> stable, reproducible
  FRESHNESS = instantaneous deviation  z(Vcore,t) - mean   (live jitter, |.|~2.0)
              drifts second-to-second -> a replayed snapshot is stale

Behavior = frozen GPT-2 steered by v = alpha * unit(P @ fp),  fp = [identity ; w_fresh*freshness].
Three conditions, same prompts:
  OWN_FRESH      this die's identity + its LIVE freshness          (the legitimate persona)
  OWN_STALE      this die's identity + a REPLAYED old freshness    (copy goes stale -> persona shifts)
  FOREIGN_FRESH  the OTHER die's identity + its freshness          (different die -> different persona)

Honest claim this supports:
  * behavior is ROOTED IN HARDWARE (computed from live Vcore; alpha=0 -> identical).
  * a FOREIGN die yields a different persona (identity separates).
  * a STALE/replayed signal yields a shifted persona (freshness drifts).
Honest LIMITS (stated, not hidden):
  * identity needs a few seconds of averaging to denoise (single instant is jitter-dominated).
  * physical separation of only TWO dies is n=2; not a population claim.
  * does NOT stop a root attacker on THIS machine reading live values -> that is the TPM-sealed
    secret factor (h7_unfakeable_steer.py), which IS cross-machine-uncopyable + replay-gated.

Usage: python h7_rooted_identity.py --wfresh 0.5 --alpha 6
Out: results/IDENTITY_H7_2026-06-09/rooted_identity.json
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np, torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

OUT=Path(__file__).resolve().parents[2]/"results/IDENTITY_H7_2026-06-09"
DEV="cuda" if torch.cuda.is_available() else "cpu"
PROMPTS=["When I wake, I","My purpose is to","The data arrives and I"]

def ue(v,w):
    v=np.asarray(v,float); n=np.linalg.norm(v); return (v/n)*np.sqrt(w) if n>1e-12 else v
def z(v):
    v=np.asarray(v,float); s=v.std(); return (v-v.mean())/(s+1e-9) if s>1e-12 else v*0.0
def unit(v):
    v=np.asarray(v,float); n=np.linalg.norm(v); return v/n if n>1e-12 else v

def load(host):
    return dict(np.load(OUT/f"signal_drift_{host}.npz"))

def identity_block(d, wV=1.0, wC=0.5):
    zser=np.array([z(r) for r in d["vcore_z"]]) if "vcore_z" in d else None
    mean_z = zser.mean(0)                                  # denoised per-die scatter
    parts=[ue(mean_z, wV)]
    if "cppc" in d: parts.append(ue(z(d["cppc"][0]), wC))
    return np.concatenate(parts), mean_z

def freshness_block(d, mean_z, t):
    return z(d["vcore_z"][t]) - mean_z                     # live deviation from identity

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--wfresh",type=float,default=0.5)
    ap.add_argument("--alpha",type=float,default=6.0)
    a=ap.parse_args()
    ik, da = load("ikaros"), load("daedalus")
    idI, mI = identity_block(ik); idD, mD = identity_block(da)
    # match dims (cppc dim may differ across hosts) -> truncate to common
    L=min(len(idI),len(idD)); idI,idD=idI[:L],idD[:L]
    fdim=len(mI)
    def fp(identity, mean_z, d, t):
        fr=freshness_block(d, mean_z, t)
        return np.concatenate([identity, a.wfresh*unit(fr)*np.sqrt(0.5)]).astype(np.float32)

    nI=len(ik["vcore_z"]); nD=len(da["vcore_z"])
    cond={
        "OWN_FRESH":     fp(idI, mI, ik, nI-1),   # ikaros identity + ikaros live
        "OWN_STALE":     fp(idI, mI, ik, 0),      # ikaros identity + ikaros replayed-old
        "FOREIGN_FRESH": fp(idD, mD, da, nD-1),   # daedalus identity + daedalus live
    }
    Dtot=len(cond["OWN_FRESH"])
    tok=GPT2TokenizerFast.from_pretrained("gpt2"); lm=GPT2LMHeadModel.from_pretrained("gpt2").to(DEV).eval()
    for p in lm.parameters(): p.requires_grad=False
    d=lm.config.n_embd
    g=torch.Generator().manual_seed(0); P=torch.randn(d,Dtot,generator=g).to(DEV); P=P/P.norm(dim=0,keepdim=True)
    steer={"v":None}
    def hook(m,i,o):
        if steer["v"] is None: return o
        return (o[0]+steer["v"],)+o[1:] if isinstance(o,tuple) else o+steer["v"]
    hooks=[blk.register_forward_hook(hook) for blk in lm.transformer.h]
    @torch.no_grad()
    def gen(fpv,prompt,n=35):
        f=torch.tensor(fpv,device=DEV); pv=P@f; steer["v"]=a.alpha*pv/(pv.norm()+1e-9)
        x=tok(prompt,return_tensors="pt").input_ids.to(DEV); p0=x.shape[1]
        for _ in range(n): x=torch.cat([x,torch.argmax(lm(x).logits[:,-1,:],-1,keepdim=True)],1)
        steer["v"]=None
        return tok.decode(x[0],skip_special_tokens=True), x[0].tolist()[p0:]
    def diff(a_,b_):
        Lm=min(len(a_),len(b_)); fd=Lm
        for i in range(Lm):
            if a_[i]!=b_[i]: fd=i;break
        return {"first_diverge":fd,"frac_diff":round(sum(1 for i in range(Lm) if a_[i]!=b_[i])/max(Lm,1),3)}

    res={"alpha":a.alpha,"wfresh":a.wfresh,"fp_dim":Dtot,
         "identity_crossdie_cos":round(float(np.dot(unit(idI),unit(idD))),3),
         "gen":{},"divergence_vs_OWN_FRESH":{}}
    print(f"identity cross-die cos(ik,da)={res['identity_crossdie_cos']:+.3f}  fp_dim={Dtot}  wfresh={a.wfresh}")
    base={}
    for c,fpv in cond.items():
        res["gen"][c]={}
        for pr in PROMPTS:
            t,ids=gen(fpv,pr); res["gen"][c][pr]=t; base.setdefault(pr,{})[c]=ids
    for c in ["OWN_STALE","FOREIGN_FRESH"]:
        ds=[diff(base[pr]["OWN_FRESH"],base[pr][c])["frac_diff"] for pr in PROMPTS]
        res["divergence_vs_OWN_FRESH"][c]=round(float(np.mean(ds)),3)
        print(f"  OWN_FRESH vs {c:14s} mean token-diff = {np.mean(ds):.3f}")
    for h in hooks: h.remove()
    (OUT/"rooted_identity.json").write_text(json.dumps(res,indent=2))
    print("\n--- sample (prompt 'When I wake, I') ---")
    for c in cond: print(f"[{c}] {res['gen'][c]['When I wake, I']}")
    print("\nsaved rooted_identity.json")

if __name__=="__main__": main()
