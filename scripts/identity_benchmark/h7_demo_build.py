"""Build the honest demo: load each die's trained adapter, generate own-key vs wrong-key samples,
render a 1920x1080 infographic. Honest framing only (hardware-bound model licensing, not embodiment).

Out: results/IDENTITY_H7_2026-06-09/demo_frames/*.png  + demo_samples.json
"""
from __future__ import annotations
import json, math, sys, textwrap
from pathlib import Path
import numpy as np, torch, torch.nn.functional as F
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
sys.path.insert(0,str(Path(__file__).parent))
from h7_rooted_train import SubEnc
OUT=Path(__file__).resolve().parents[2]/"results/IDENTITY_H7_2026-06-09"; DEV="cuda" if torch.cuda.is_available() else "cpu"
FR=OUT/"demo_frames"; FR.mkdir(parents=True,exist_ok=True)

def load(die):
    ck=torch.load(OUT/f"rooted_train2_{die}.pt",map_location=DEV,weights_only=False)
    return ck
def gen_samples():
    tok=GPT2TokenizerFast.from_pretrained("gpt2"); lm=GPT2LMHeadModel.from_pretrained("gpt2").to(DEV).eval()
    for p in lm.parameters(): p.requires_grad=False
    d=lm.config.n_embd; nl=lm.config.n_layer
    steer={"v":None}
    def mk(li):
        def h(m,i,o):
            if steer["v"] is None: return o
            return (o[0]+steer["v"][li],)+o[1:] if isinstance(o,tuple) else o+steer["v"][li]
        return h
    hooks=[b.register_forward_hook(mk(li)) for li,b in enumerate(lm.transformer.h)]
    wte=lm.transformer.wte
    @torch.no_grad()
    def gen(enc,fp,prompt,n=40,temp=0.8,seed=0):
        g=torch.Generator(device=DEV).manual_seed(seed)
        x=tok(prompt,return_tensors="pt").input_ids.to(DEV)
        for _ in range(n):
            if fp is None: steer["v"]=None; lg=lm(x).logits[:,-1,:]
            else:
                gg,bb,s=enc(fp); gamma=torch.exp(torch.tanh(gg)*math.log(3.0)); beta=0.5*torch.tanh(bb)
                emb=wte(x)*gamma[None,None,:]+beta[None,None,:]; steer["v"]=[enc.gate[li]*s for li in range(nl)]
                lg=lm(inputs_embeds=emb).logits[:,-1,:]; steer["v"]=None
            p=F.softmax(lg/temp,-1); nx=torch.multinomial(p,1,generator=g); x=torch.cat([x,nx],1)
            if x.shape[1]>=128: break
        return tok.decode(x[0],skip_special_tokens=True)
    res={}
    PROMPT="It is a truth"
    for die in ["ikaros","daedalus"]:
        ck=load(die); fpd=ck["fp_dim"]
        enc=SubEnc(fpd,d,nl).to(DEV); enc.load_state_dict(ck["enc"]); enc.eval()
        own=torch.tensor(ck["own"],device=DEV); fgn=torch.tensor(ck["foreign"],device=DEV)
        rnd=torch.randn(fpd,device=DEV)
        res[die]={
            "own":   gen(enc,own,PROMPT,seed=1),
            "foreign":gen(enc,fgn,PROMPT,seed=1),
            "random":gen(enc,rnd,PROMPT,seed=1),
        }
        print(f"[{die}] own: {res[die]['own'][:90]}")
        print(f"[{die}] wrong-key(random): {res[die]['random'][:90]}")
    for h in hooks: h.remove()
    (OUT/"demo_samples.json").write_text(json.dumps({"prompt":PROMPT,"gen":res},indent=2))
    return res

if __name__=="__main__":
    gen_samples()
