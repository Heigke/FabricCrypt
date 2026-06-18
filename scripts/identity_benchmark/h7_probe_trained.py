"""H7 probe of the TRAINED rooted model — the honest gate. Loads rooted_train_{die}.pt and tests on
HELD-OUT text (NOT the training corpus) so we measure generalisation, not memorisation.

Decisive contrast vs the untrained surgery (h7_probe_surgery: own KL 0.684 == random KL 0.684 -> NULL,
i.e. real fp indistinguishable from random):
  TRAINED model should give PPL(own) << PPL(random)  on held-out text -> real fp carries STRUCTURE that
  a random vector cannot fake. Plus determinism (rerun identical) and qualitative generations.

Usage: python h7_probe_trained.py --die daedalus
Out: results/IDENTITY_H7_2026-06-09/probe_trained_{die}.json
"""
from __future__ import annotations
import argparse, json, math, sys, socket
from pathlib import Path
import numpy as np, torch, torch.nn.functional as F
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
sys.path.insert(0,str(Path(__file__).parent))
from h7_rooted_train import SubEnc

OUT=Path(__file__).resolve().parents[2]/"results/IDENTITY_H7_2026-06-09"
DEV="cuda" if torch.cuda.is_available() else "cpu"
# HELD-OUT text — different topic/style from the training corpus (no overlap)
HELDOUT=("""A ship leaves the harbour before sunrise, its sails grey against the paling sky. The captain
checks the chart and the compass, then orders a new heading toward the northern islands. Below deck the
cook lights the stove and the smell of porridge drifts up the narrow stair. Birds follow in the wake,
diving for the scraps the crew throws over the rail. By noon the wind freshens and the hull leans hard,
spray crossing the deck with every wave. In the laboratory a student repeats the experiment for the third
time, adjusting the temperature and recording the current at each step. The instrument drifts a little, so
she calibrates it again and notes the correction in her book. Results that cannot be reproduced are not yet
knowledge, her supervisor likes to say. She plots the points and a clear line appears, and for a moment the
long week feels worth it.""").replace("\n"," ")
PROMPTS=["When I wake, I","The result of the experiment","She looked at the sea and"]

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--die",required=True); ap.add_argument("--ckpt",default=""); a=ap.parse_args()
    host=socket.gethostname()
    ckpath=Path(a.ckpt) if a.ckpt else OUT/f"rooted_train_{a.die}.pt"
    ck=torch.load(ckpath,map_location=DEV,weights_only=False)
    fpd=ck["fp_dim"]; own=torch.tensor(ck["own"],device=DEV); fgn=torch.tensor(ck["foreign"],device=DEV)
    tok=GPT2TokenizerFast.from_pretrained("gpt2"); lm=GPT2LMHeadModel.from_pretrained("gpt2").to(DEV).eval()
    for p in lm.parameters(): p.requires_grad=False
    d=lm.config.n_embd; nlayer=lm.config.n_layer
    enc=SubEnc(fpd,d,nlayer).to(DEV); enc.load_state_dict(ck["enc"]); enc.eval()
    steer={"vecs":None}
    def mk(li):
        def hook(m,i,o):
            if steer["vecs"] is None: return o
            v=steer["vecs"][li]; return (o[0]+v,)+o[1:] if isinstance(o,tuple) else o+v
        return hook
    hooks=[blk.register_forward_hook(mk(li)) for li,blk in enumerate(lm.transformer.h)]
    wte=lm.transformer.wte
    @torch.no_grad()
    def logits(ids_bt,fp):
        if fp is None:
            steer["vecs"]=None; return lm(ids_bt).logits
        g,b,s=enc(fp); emb=wte(ids_bt)*torch.exp(torch.tanh(g)*math.log(3.0))[None,None,:]*0+wte(ids_bt)  # placeholder
        # recompute FiLM exactly as training
        gamma,beta,steerv=enc(fp); emb=wte(ids_bt)*gamma[None,None,:]+beta[None,None,:]
        steer["vecs"]=[enc.gate[li]*steerv for li in range(nlayer)]
        out=lm(inputs_embeds=emb).logits; steer["vecs"]=None; return out
    ids=tok(HELDOUT,return_tensors="pt").input_ids.to(DEV)
    @torch.no_grad()
    def ppl_heldout(fp):
        x=ids[:,:-1]; y=ids[:,1:]; lg=logits(x,fp)
        return math.exp(min(F.cross_entropy(lg.reshape(-1,lg.shape[-1]),y.reshape(-1)).item(),20))
    rnd=[ppl_heldout(torch.randn(fpd,device=DEV)) for _ in range(8)]
    tbl={"own":ppl_heldout(own),"zero":ppl_heldout(torch.zeros(fpd,device=DEV)),"foreign":ppl_heldout(fgn),
         "random_mean":float(np.mean(rnd)),"random_std":float(np.std(rnd)),"random_min":float(np.min(rnd)),
         "random_median":float(np.median(rnd)),"plain":ppl_heldout(None)}
    # determinism
    l1=logits(ids[:,:-1],own); l2=logits(ids[:,:-1],own); det=float((l1-l2).abs().max())
    # qualitative generations
    @torch.no_grad()
    def gen(fp,prompt,n=35):
        x=tok(prompt,return_tensors="pt").input_ids.to(DEV)
        for _ in range(n): x=torch.cat([x,torch.argmax(logits(x,fp)[:,-1,:],-1,keepdim=True)],1)
        return tok.decode(x[0],skip_special_tokens=True)
    gens={p:{"own":gen(own,p),"random":gen(torch.randn(fpd,device=DEV),p),"foreign":gen(fgn,p)} for p in PROMPTS}
    gate={"own_below_random": tbl["own"]<tbl["random_mean"]-tbl["random_std"],
          "own_below_zero": tbl["own"]<tbl["zero"], "own_below_foreign": tbl["own"]<tbl["foreign"],
          "deterministic": det==0.0,
          "random_breaks_ratio": round(tbl["random_mean"]/max(tbl["own"],1e-6),2)}
    res={"die":a.die,"host":host,"heldout":True,"ppl_heldout":tbl,"determinism_max_abs":det,
         "null_gate":gate,"generations":gens}
    for h in hooks: h.remove()
    (OUT/f"probe_trained_{a.die}.json").write_text(json.dumps(res,indent=2))
    print(f"[{a.die}] HELD-OUT PPL  own={tbl['own']:.2f}  zero={tbl['zero']:.2f}  foreign={tbl['foreign']:.2f}  "
          f"random={tbl['random_mean']:.2f}±{tbl['random_std']:.2f}  plain={tbl['plain']:.2f}")
    print(f"[{a.die}] determinism max|Δ|={det:.1e}  random breaks own by {gate['random_breaks_ratio']}x")
    print(f"[{a.die}] NULL GATE on held-out: own<random={gate['own_below_random']} own<zero={gate['own_below_zero']} own<foreign={gate['own_below_foreign']} det={gate['deterministic']}")
    print(f"  [own]    {gens[PROMPTS[0]]['own']}")
    print(f"  [random] {gens[PROMPTS[0]]['random']}")
    print(f"saved probe_trained_{a.die}.json")

if __name__=="__main__": main()
