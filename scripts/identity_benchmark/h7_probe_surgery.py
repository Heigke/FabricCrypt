"""H7 probe surgery — prove the HW signal changes the model's COMPUTATION, not just that
greedy decoding amplified a microscopic flip into a different continuation (the "generative fluke").

Everything here is TEACHER-FORCED (one forward over a fixed corpus, no sampling, no autoregressive
cascade), so divergence cannot come from the butterfly effect. We measure, per condition (steering
fp injected at every block), against baseline (alpha=0):

 P1 DETERMINISM   rerun own twice -> max|logit diff| must be 0 (effect is reproducible, not random).
 P2 DISTRIBUTION  mean KL(P_C || P_0) over all positions + argmax-flip rate at the DISTRIBUTION level
                  (the honest, cascade-free divergence) -- contrast with the inflated free-run token-diff.
 P3 DOSE-RESPONSE KL(own||baseline) vs alpha sweep -> must be monotone & smooth (structured, not chaos).
 P4 STRUCTURE     logit-shift vector D[pos]=logit_C[pos]-logit_0[pos] (len=vocab). If the signal does
                  STRUCTURED global work, D[pos_i].D[pos_j] correlate HIGH across unrelated positions
                  (same tokens pushed everywhere). A fluke -> ~0. And own-vs-foreign shift directions
                  must DIFFER (low cos) -> the two dies steer the vocabulary differently, consistently.
 P5 NULL          K random fingerprints: KL(rand||baseline) distribution -> own/foreign effect sits in
                  the structured regime, and the own/foreign DIRECTION pair is reproducible vs random pairs.
 + logit-lens     top promoted/suppressed tokens for own vs foreign (what the signal actually does).

Out: results/IDENTITY_H7_2026-06-09/probe_surgery.json
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np, torch, torch.nn.functional as F
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

OUT=Path(__file__).resolve().parents[2]/"results/IDENTITY_H7_2026-06-09"
DEV="cuda" if torch.cuda.is_available() else "cpu"
CORPUS=("The system reads its sensors and reports the state. A model loads its weights and computes "
"the next value. When the load rises the temperature climbs and the schedule adjusts in response. "
"Every measurement is recorded and compared against the previous one. The result is returned to the "
"caller and the loop continues. In the evening the wind moved across the field and the light fell "
"slowly behind the hills, and the river carried the sound of the town away into the dark water.")

def z(v):
    v=np.asarray(v,float); s=v.std(); return (v-v.mean())/(s+1e-9) if s>1e-12 else v*0.0
def ue(v,w):
    v=np.asarray(v,float); n=np.linalg.norm(v); return (v/n)*np.sqrt(w) if n>1e-12 else v
def unit(v):
    v=np.asarray(v,float); n=np.linalg.norm(v); return v/n if n>1e-12 else v

def load(h): return dict(np.load(OUT/f"signal_drift_{h}.npz"))
def identity_block(d,wV=1.0,wC=0.5):
    mean_z=np.array([z(r) for r in d["vcore_z"]]).mean(0)
    parts=[ue(mean_z,wV)]
    if "cppc" in d: parts.append(ue(z(d["cppc"][0]),wC))
    return np.concatenate(parts), mean_z
def fresh(d,mz,t): return z(d["vcore_z"][t])-mz

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--alpha",type=float,default=6.0)
    ap.add_argument("--wfresh",type=float,default=0.5); a=ap.parse_args()
    ik,da=load("ikaros"),load("daedalus")
    idI,mI=identity_block(ik); idD,mD=identity_block(da)
    L=min(len(idI),len(idD)); idI,idD=idI[:L],idD[:L]
    def fp(idblk,mz,d,t): return np.concatenate([idblk,a.wfresh*unit(fresh(d,mz,t))*np.sqrt(0.5)]).astype(np.float32)
    FPS={
        "own_fresh":     fp(idI,mI,ik,len(ik["vcore_z"])-1),
        "own_stale":     fp(idI,mI,ik,0),
        "foreign_fresh": fp(idD,mD,da,len(da["vcore_z"])-1),
    }
    D=len(FPS["own_fresh"])
    tok=GPT2TokenizerFast.from_pretrained("gpt2"); lm=GPT2LMHeadModel.from_pretrained("gpt2").to(DEV).eval()
    for p in lm.parameters(): p.requires_grad=False
    nd=lm.config.n_embd
    g=torch.Generator().manual_seed(0); P=torch.randn(nd,D,generator=g).to(DEV); P=P/P.norm(dim=0,keepdim=True)
    def vec(fpv,alpha):
        f=torch.tensor(fpv,device=DEV); pv=P@f; return alpha*pv/(pv.norm()+1e-9)
    steer={"v":None}
    def hook(m,i,o):
        if steer["v"] is None: return o
        return (o[0]+steer["v"],)+o[1:] if isinstance(o,tuple) else o+steer["v"]
    hooks=[blk.register_forward_hook(hook) for blk in lm.transformer.h]
    ids=tok(CORPUS,return_tensors="pt").input_ids.to(DEV)
    @torch.no_grad()
    def logits(v):
        steer["v"]=v; out=lm(ids).logits[0]; steer["v"]=None; return out   # [seq, vocab]
    base=logits(None)
    P0=F.log_softmax(base,-1)
    def klvs0(v):
        lg=logits(v); lp=F.log_softmax(lg,-1)
        kl=(lp.exp()*(lp-P0)).sum(-1)                # KL(P_C||P_0) per position
        flip=(lg.argmax(-1)!=base.argmax(-1)).float().mean().item()
        return lg, kl.mean().item(), flip
    res={"alpha":a.alpha,"wfresh":a.wfresh,"fp_dim":D,"seq_len":int(ids.shape[1]),"probes":{}}

    # P1 determinism
    v_own=vec(FPS["own_fresh"],a.alpha)
    l1,_,_=klvs0(v_own); l2,_,_=klvs0(v_own)
    res["probes"]["P1_determinism_max_abs_logit_diff_rerun"]=float((l1-l2).abs().max())

    # P2 distribution-level divergence per condition (cascade-free) + logit-shift cache
    shift={}; condkl={}
    for name,fpv in FPS.items():
        lg,kl,flip=klvs0(vec(fpv,a.alpha))
        condkl[name]={"meanKL_vs_baseline":round(kl,4),"argmax_flip_rate_distrib":round(flip,3)}
        shift[name]=(lg-base)                          # [seq,vocab] logit-shift
    # pairwise distribution distance own vs others
    def klpair(va,vb):
        la=F.log_softmax(logits(va),-1); lb=F.log_softmax(logits(vb),-1)
        return float((la.exp()*(la-lb)).sum(-1).mean())
    condkl["pair_own_vs_foreign_meanKL"]=round(klpair(vec(FPS["own_fresh"],a.alpha),vec(FPS["foreign_fresh"],a.alpha)),4)
    condkl["pair_own_vs_stale_meanKL"]=round(klpair(vec(FPS["own_fresh"],a.alpha),vec(FPS["own_stale"],a.alpha)),4)
    res["probes"]["P2_distribution"]=condkl

    # P3 dose-response
    dose={}
    for al in [0.0,1.0,2.0,4.0,6.0,8.0,10.0]:
        _,kl,flip=klvs0(vec(FPS["own_fresh"],al) if al>0 else None)
        dose[f"alpha={al}"]={"meanKL":round(kl,4),"flip":round(flip,3)}
    res["probes"]["P3_dose_response"]=dose
    kls=[dose[f"alpha={al}"]["meanKL"] for al in [0.0,1.0,2.0,4.0,6.0,8.0,10.0]]
    res["probes"]["P3_monotone_increasing"]=all(kls[i]<=kls[i+1]+1e-6 for i in range(len(kls)-1))

    # P4 structure: per-position logit-shift consistency (own), and own-vs-foreign direction
    def offdiag_mean_cos(Dmat):                         # Dmat [seq,vocab]
        N=F.normalize(Dmat,dim=-1); G=N@N.T; n=G.shape[0]
        return float((G.sum()-G.diag().sum())/(n*(n-1)))
    own_consistency=offdiag_mean_cos(shift["own_fresh"])
    fgn_consistency=offdiag_mean_cos(shift["foreign_fresh"])
    # mean own-direction vs mean foreign-direction (averaged over positions)
    own_mean=F.normalize(shift["own_fresh"].mean(0),dim=-1); fgn_mean=F.normalize(shift["foreign_fresh"].mean(0),dim=-1)
    res["probes"]["P4_structure"]={
        "own_perposition_shift_consistency_cos":round(own_consistency,3),
        "foreign_perposition_shift_consistency_cos":round(fgn_consistency,3),
        "own_vs_foreign_shift_direction_cos":round(float(own_mean@fgn_mean),3),
    }

    # P5 null: random fingerprints
    rkl=[]; rng=np.random.default_rng(0)
    for k in range(8):
        rf=rng.standard_normal(D).astype(np.float32)
        _,kl,_=klvs0(vec(rf,a.alpha)); rkl.append(kl)
    res["probes"]["P5_null_random_fp_meanKL"]={"mean":round(float(np.mean(rkl)),4),"std":round(float(np.std(rkl)),4),
        "own_meanKL":condkl["own_fresh"]["meanKL_vs_baseline"]}

    # logit-lens: top promoted / suppressed tokens (mean shift over positions)
    def toptokens(name,k=8):
        s=shift[name].mean(0)
        up=torch.topk(s,k).indices.tolist(); dn=torch.topk(-s,k).indices.tolist()
        return {"promotes":[tok.decode([i]).strip() for i in up],"suppresses":[tok.decode([i]).strip() for i in dn]}
    res["probes"]["logit_lens"]={n:toptokens(n) for n in ["own_fresh","foreign_fresh"]}

    for h in hooks: h.remove()
    (OUT/"probe_surgery.json").write_text(json.dumps(res,indent=2))
    p=res["probes"]
    print(f"P1 determinism (rerun max|Δlogit|) = {p['P1_determinism_max_abs_logit_diff_rerun']:.2e}  (0 = perfectly reproducible)")
    print(f"P2 distribution-level (teacher-forced, NO cascade):")
    for n in FPS: print(f"   {n:14s} meanKL_vs_base={p['P2_distribution'][n]['meanKL_vs_baseline']:.3f}  argmax-flip={p['P2_distribution'][n]['argmax_flip_rate_distrib']:.3f}")
    print(f"   own vs foreign meanKL={p['P2_distribution']['pair_own_vs_foreign_meanKL']:.3f}  own vs stale meanKL={p['P2_distribution']['pair_own_vs_stale_meanKL']:.3f}")
    print(f"P3 dose-response monotone={p['P3_monotone_increasing']}  KL@α: "+", ".join(f"{al}:{dose[f'alpha={al}']['meanKL']:.2f}" for al in [0.0,2.0,6.0,10.0]))
    print(f"P4 structure: own per-pos shift consistency={p['P4_structure']['own_perposition_shift_consistency_cos']:.3f} "
          f"(high=structured global push); own-vs-foreign direction cos={p['P4_structure']['own_vs_foreign_shift_direction_cos']:.3f} (low=dies steer differently)")
    print(f"P5 null random-fp meanKL={p['P5_null_random_fp_meanKL']['mean']:.3f}±{p['P5_null_random_fp_meanKL']['std']:.3f} vs own {p['P5_null_random_fp_meanKL']['own_meanKL']:.3f}")
    print(f"logit-lens own promotes: {p['logit_lens']['own_fresh']['promotes'][:6]}")
    print(f"logit-lens foreign promotes: {p['logit_lens']['foreign_fresh']['promotes'][:6]}")
    print("saved probe_surgery.json")

if __name__=="__main__": main()
