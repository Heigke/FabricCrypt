"""H7 rooted-LM on the PER-CORE VOLTAGE FINGERPRINT — the first conditioning signal that actually
carries die identity (config-immune per-core Vcore scatter, idx756-771, INTER 0.74 << INTRA 0.98).

Unlike rooted_lm v1/v2 (6-ch time series, no die info -> wrong-die ceiling), here the model is FiLM-
conditioned on the static 16-D per-core fingerprint of THIS die. Hard-negative margin training forces
the LM to depend on its own die's key: it is trained to be GOOD with its own fingerprint and WORSE with
the other die's fingerprint. A model trained on ikaros, run with daedalus's fingerprint, should degrade.

Falsification eval: PPL(own) vs PPL(wrong-die) vs PPL(zero) vs PPL(shuffled). Margin in nats = rooting.
Thermal-guarded GPU training (pause if zone0 > THERM_HI).

Usage: python h7_rooted_percore_lm.py train --steps 1500
       python h7_rooted_percore_lm.py eval
Env/flags: HOST auto. fingerprints loaded from results dir: fingerprint_{ikaros,daedalus}.npy
"""
from __future__ import annotations
import argparse, math, os, socket, time
from pathlib import Path
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F

HOST=socket.gethostname()
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/"results/IDENTITY_H7_2026-06-09"; OUT.mkdir(parents=True,exist_ok=True)
DEV="cuda" if torch.cuda.is_available() else "cpu"
THERM_HI=float(os.environ.get("THERM_HI","78")); THERM_LO=float(os.environ.get("THERM_LO","60"))
def zone0():
    try: return int(Path("/sys/class/thermal/thermal_zone0/temp").read_text())/1000.0
    except: return 0.0
def thermal_guard():
    if zone0()>THERM_HI:
        t0=time.time()
        while zone0()>THERM_LO and time.time()-t0<120: time.sleep(1.0)

# ---- fingerprints (the die keys) ----
def load_fp(host):
    p=OUT/f"fingerprint_{host}.npy"
    if p.exists(): return np.load(p).astype(np.float32)
    # fallback for dry-run: deterministic per-host pseudo-fingerprint
    rng=np.random.default_rng(abs(hash(host))%(2**32)); v=rng.standard_normal(16).astype(np.float32)
    return (v-v.mean())/(v.std()+1e-9)
OWN=("daedalus" if "daedalus" in HOST else "ikaros")
OTHER=("ikaros" if OWN=="daedalus" else "daedalus")

# ---- structured byte corpus (deterministic, identical on both machines; learnable so PPL is meaningful) ----
def make_stream(n=400_000, seed=0):
    rng=np.random.default_rng(seed)
    # 1st-order Markov over a 64-symbol alphabet -> structured, learnable, sub-uniform entropy
    K=64; T=rng.dirichlet(np.ones(K)*0.3, size=K).astype(np.float64)
    cdf=np.cumsum(T,1)                                  # (K,K)
    s=np.zeros(n,dtype=np.int64); c=0; u=rng.random(n)
    for i in range(n):
        c=int(np.searchsorted(cdf[c],u[i])); c=min(c,K-1); s[i]=c
    return s
class Corpus:
    def __init__(self,ctx,seed=0):
        self.ctx=ctx; self.s=make_stream(seed=seed); self.n=len(self.s)
    def batch(self,bs,dev):
        ix=np.random.randint(0,self.n-self.ctx-1,size=bs)
        x=np.stack([self.s[i:i+self.ctx] for i in ix]); y=np.stack([self.s[i+1:i+self.ctx+1] for i in ix])
        return torch.from_numpy(x).to(dev), torch.from_numpy(y).to(dev)

VOCAB=64
class FiLMBlock(nn.Module):
    def __init__(self,d,heads,d_sub):
        super().__init__(); self.ln1=nn.LayerNorm(d); self.attn=nn.MultiheadAttention(d,heads,batch_first=True)
        self.ln2=nn.LayerNorm(d); self.ff=nn.Sequential(nn.Linear(d,4*d),nn.GELU(),nn.Linear(4*d,d))
        self.film=nn.Linear(d_sub,2*d)
    def forward(self,x,mask,z):
        g,b=self.film(z).chunk(2,-1)
        g=torch.exp(0.5*torch.tanh(g))          # exp-FiLM: wrong z breaks the scale ladder
        x=x*g.unsqueeze(1)+ (0.5*torch.tanh(b)).unsqueeze(1)
        h=self.ln1(x); a,_=self.attn(h,h,h,attn_mask=mask,need_weights=False); x=x+a
        return x+self.ff(self.ln2(x))
class RootedLM(nn.Module):
    def __init__(self,d=192,n_layers=4,heads=4,d_sub=64,ctx=96):
        super().__init__(); self.ctx=ctx
        self.tok=nn.Embedding(VOCAB,d); self.pos=nn.Embedding(ctx,d)
        self.fp=nn.Sequential(nn.Linear(16,64),nn.GELU(),nn.Linear(64,d_sub))
        self.emb_film=nn.Linear(d_sub,2*d)
        self.blocks=nn.ModuleList([FiLMBlock(d,heads,d_sub) for _ in range(n_layers)])
        self.head=nn.Linear(d,VOCAB)
        self.register_buffer("mask",torch.triu(torch.full((ctx,ctx),float("-inf")),1))
    def forward(self,ids,fp16):
        B,T=ids.shape; z=self.fp(fp16)
        h=self.tok(ids)+self.pos(torch.arange(T,device=ids.device)[None])
        g,b=self.emb_film(z).chunk(2,-1)         # modulate base representation
        h=h*torch.exp(0.5*torch.tanh(g)).unsqueeze(1)+(0.5*torch.tanh(b)).unsqueeze(1)
        m=self.mask[:T,:T]
        for blk in self.blocks: h=blk(h,m,z)
        return self.head(h)

def ce(model,x,y,fp):
    return F.cross_entropy(model(x,fp).reshape(-1,VOCAB), y.reshape(-1))

def train(steps,bs=48,ctx=96,lr=3e-4,margin=1.5,lam=1.0,seed=0):
    torch.manual_seed(seed)
    own=torch.tensor(load_fp(OWN),device=DEV); other=torch.tensor(load_fp(OTHER),device=DEV)
    print(f"[{HOST}] OWN={OWN} fp={np.round(own.cpu().numpy(),2)}",flush=True)
    print(f"[{HOST}] OTHER={OTHER} fp_corr_to_own={float(F.cosine_similarity(own[None],other[None])):+.3f}",flush=True)
    corp=Corpus(ctx,seed=0); model=RootedLM(ctx=ctx).to(DEV)
    opt=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=0.01)
    n=sum(p.numel() for p in model.parameters()); print(f"params={n/1e6:.2f}M dev={DEV}",flush=True)
    t0=time.time()
    for step in range(steps):
        if step%50==0: thermal_guard()
        x,y=corp.batch(bs,DEV)
        ownb=own[None].expand(bs,-1)
        real=ce(model,x,y,ownb)                          # good ONLY with exact own fp
        # MULTI-NEGATIVE: must be worse on wrong-die AND shuffled-own AND zero AND random.
        # This forces dependence on the EXACT per-core pattern, not binary die-vs-die gating.
        negs=[other,                                       # other die
              own[torch.randperm(16,device=DEV)],          # shuffled own (kills the shuffle wall)
              torch.zeros(16,device=DEV),                  # zero
              torch.randn(16,device=DEV)]                  # random
        idents=[]
        for ng in negs:
            sp=ce(model,x,y,ng[None].expand(bs,-1))
            idents.append(F.softplus(margin-(sp-real)))
        ident=torch.stack(idents).mean()
        loss=real+lam*ident
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
        if step%100==0 or step==steps-1:
            with torch.no_grad():
                shuf=own[torch.randperm(16,device=DEV)]
                sps=ce(model,x,y,shuf[None].expand(bs,-1)).item()
            print(f"  step {step:4d} real_ppl={math.exp(min(real.item(),20)):.3f} "
                  f"shuf_margin={sps-real.item():+.3f}nat T={zone0():.0f}C t={time.time()-t0:.0f}s",flush=True)
    ck=OUT/f"percore_lm_{OWN}.pt"; torch.save({"model":model.state_dict(),"own":OWN},ck)
    print(f"[{HOST}] saved {ck.name}",flush=True)
    evaluate(model,corp,own,other)

@torch.no_grad()
def evaluate(model,corp,own,other,n=60,bs=48):
    model.eval(); rng=np.random.default_rng(7)
    def ppl(fpvec):
        tot=0
        for _ in range(n):
            x,y=corp.batch(bs,DEV); tot+=ce(model,x,y,fpvec[None].expand(bs,-1)).item()
        return math.exp(tot/n)
    zero=torch.zeros(16,device=DEV)
    shuf=own[torch.randperm(16,device=DEV)]
    p_own=ppl(own); p_oth=ppl(other); p_zero=ppl(zero); p_shuf=ppl(shuf)
    print(f"\n[{HOST}] FALSIFICATION (PPL, lower=better):")
    print(f"   own-die fp   = {p_own:.3f}")
    print(f"   wrong-die fp = {p_oth:.3f}   margin={math.log(p_oth/p_own):+.3f} nat ({p_oth/p_own:.2f}x)")
    print(f"   zero fp      = {p_zero:.3f}   margin={math.log(p_zero/p_own):+.3f} nat")
    print(f"   shuffled fp  = {p_shuf:.3f}   margin={math.log(p_shuf/p_own):+.3f} nat")
    print(f"   -> wrong-die margin >0 and large = LM is ROOTED in its own die's per-core silicon signature.")
    model.train()

def eval_only():
    own=torch.tensor(load_fp(OWN),device=DEV); other=torch.tensor(load_fp(OTHER),device=DEV)
    corp=Corpus(96,seed=0); model=RootedLM(ctx=96).to(DEV)
    ck=OUT/f"percore_lm_{OWN}.pt"
    if ck.exists(): model.load_state_dict(torch.load(ck,map_location=DEV)["model"]); print(f"loaded {ck.name}")
    evaluate(model,corp,own,other)

if __name__=="__main__":
    ap=argparse.ArgumentParser(); ap.add_argument("cmd",choices=["train","eval"]); ap.add_argument("--steps",type=int,default=1500)
    a=ap.parse_args()
    (train(a.steps) if a.cmd=="train" else eval_only())
