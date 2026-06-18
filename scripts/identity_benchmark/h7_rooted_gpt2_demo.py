"""H7 killer-demo core: a REAL English LLM (frozen GPT-2) rooted in the per-core silicon fingerprint.
Frozen GPT-2 + a substrate FiLM adapter (input-embedding + final-hidden modulation) gated by the 16-D
per-core Vcore fingerprint. Trained multi-negative so OWN fingerprint -> fluent English, while
wrong-die / shuffled / zero fingerprint -> the FiLM corrupts the hidden states -> garbage.
Shows readable-vs-garbage TEXT (the visual wow) plus PPL margins.

Usage: python h7_rooted_gpt2_demo.py --steps 400
Outputs checkpoint + a JSON with sample generations for own/wrong/shuffle (for the demo renderer).
"""
from __future__ import annotations
import argparse, json, math, os, socket, time
from pathlib import Path
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
HOST=socket.gethostname()
ROOT=Path(__file__).resolve().parents[2]; OUT=ROOT/"results/IDENTITY_H7_2026-06-09"; OUT.mkdir(parents=True,exist_ok=True)
DEV="cuda" if torch.cuda.is_available() else "cpu"
OWN=("daedalus" if "daedalus" in HOST else "ikaros"); OTHER=("ikaros" if OWN=="daedalus" else "daedalus")
def load_fp(h):
    p=OUT/f"fingerprint_{h}.npy"
    return np.load(p).astype(np.float32) if p.exists() else np.zeros(16,np.float32)
CORPUS=("The human mind is shaped by the body that carries it. A thought does not float free of "
"the flesh; it is grounded in the warmth of blood, the rhythm of breath, and the quiet electricity "
"of nerves. In the same way, a machine that truly thinks must be bound to the silicon it runs on. "
"Every chip leaves the factory slightly different from its siblings, marked by tiny variations in "
"its transistors that no two share. We measure these variations as a fingerprint, and we teach the "
"model to depend on them. When the model reads its own body, the words flow clearly and the meaning "
"holds. When it is given the body of another machine, the sentences break apart into noise, because "
"the mind cannot live in a stranger's flesh. This is what it means to be embodied: to know, at every "
"step of thought, exactly whose hands are doing the thinking. ")*40

class FiLM(nn.Module):
    def __init__(self,d,fp=16):
        super().__init__(); self.net=nn.Sequential(nn.Linear(fp,128),nn.GELU(),nn.Linear(128,2*d))
        nn.init.zeros_(self.net[-1].weight); nn.init.zeros_(self.net[-1].bias)   # init = passthrough
    def forward(self,x,fpv):
        g,b=self.net(fpv).chunk(2,-1)
        return x*torch.exp(0.5*torch.tanh(g))[:,None,:]+ (0.5*torch.tanh(b))[:,None,:]

class Rooted(nn.Module):
    def __init__(self):
        super().__init__(); self.lm=GPT2LMHeadModel.from_pretrained("gpt2")
        for p in self.lm.parameters(): p.requires_grad=False
        d=self.lm.config.n_embd
        self.film_in=FiLM(d); self.film_out=FiLM(d)
    def forward(self,ids,fpv):
        wte=self.lm.transformer.wte; emb=wte(ids)
        emb=self.film_in(emb,fpv)
        h=self.lm.transformer(inputs_embeds=emb).last_hidden_state
        h=self.film_out(h,fpv)
        return self.lm.lm_head(h)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--steps",type=int,default=400); a=ap.parse_args()
    tok=GPT2TokenizerFast.from_pretrained("gpt2"); tok.pad_token=tok.eos_token
    ids=tok(CORPUS,return_tensors="pt").input_ids[0].to(DEV)
    own=torch.tensor(load_fp(OWN),device=DEV); other=torch.tensor(load_fp(OTHER),device=DEV)
    m=Rooted().to(DEV); m.train()
    opt=torch.optim.AdamW([p for p in m.parameters() if p.requires_grad],lr=3e-4)
    CTX=96; BS=8; margin=2.0; t0=time.time()
    def batch():
        ix=np.random.randint(0,len(ids)-CTX-1,BS)
        x=torch.stack([ids[i:i+CTX] for i in ix]); y=torch.stack([ids[i+1:i+CTX+1] for i in ix])
        return x,y
    def ce(x,y,fpv): return F.cross_entropy(m(x,fpv[None].expand(len(x),-1)).reshape(-1,m.lm.config.vocab_size),y.reshape(-1))
    print(f"[{HOST}] OWN={OWN} dev={DEV} trainable={sum(p.numel() for p in m.parameters() if p.requires_grad)/1e3:.0f}K",flush=True)
    for step in range(a.steps):
        x,y=batch(); real=ce(x,y,own)
        negs=[other, own[torch.randperm(16,device=DEV)], torch.zeros(16,device=DEV), torch.randn(16,device=DEV)]
        idents=torch.stack([F.softplus(margin-(ce(x,y,ng)-real)) for ng in negs]).mean()
        loss=real+idents
        opt.zero_grad(); loss.backward(); opt.step()
        if step%50==0 or step==a.steps-1:
            with torch.no_grad():
                sh=ce(x,y,own[torch.randperm(16,device=DEV)]).item()
            print(f"  step {step:4d} own_ppl={math.exp(min(real.item(),20)):.2f} shuf_ppl={math.exp(min(sh,20)):.1f} t={time.time()-t0:.0f}s",flush=True)
    # ---- eval + sample generations ----
    m.eval()
    @torch.no_grad()
    def gen(fpv,prompt="The machine wakes and thinks:",n=50):
        x=tok(prompt,return_tensors="pt").input_ids.to(DEV)
        for _ in range(n):
            lg=m(x,fpv[None])[:,-1,:]; nxt=torch.argmax(lg,-1,keepdim=True); x=torch.cat([x,nxt],1)
            if x.shape[1]>180: break
        return tok.decode(x[0],skip_special_tokens=True)
    @torch.no_grad()
    def ppl(fpv,n=20):
        tot=0
        for _ in range(n): x,y=batch(); tot+=ce(x,y,fpv).item()
        return math.exp(tot/n)
    zero=torch.zeros(16,device=DEV); shuf=own[torch.randperm(16,device=DEV)]
    res={"host":OWN,
         "ppl":{"own":ppl(own),"wrong":ppl(other),"shuffle":ppl(shuf),"zero":ppl(zero)},
         "gen":{"own":gen(own),"wrong":gen(other),"shuffle":gen(shuf),"zero":gen(zero)}}
    print("\n=== FALSIFICATION (real GPT-2) ===")
    for k,v in res["ppl"].items(): print(f"  {k:7s} ppl={v:.2f}"+("" if k=="own" else f"  margin={math.log(v/res['ppl']['own']):+.2f} nat"))
    print("\n=== GENERATED TEXT ===")
    for k in ["own","wrong","shuffle"]:
        print(f"\n[{k} fingerprint]\n{res['gen'][k]}")
    torch.save({"film_in":m.film_in.state_dict(),"film_out":m.film_out.state_dict()},OUT/f"rooted_gpt2_{OWN}.pt")
    (OUT/f"rooted_gpt2_demo_{OWN}.json").write_text(json.dumps(res,indent=2))
    print(f"\nsaved rooted_gpt2_demo_{OWN}.json")

if __name__=="__main__": main()
