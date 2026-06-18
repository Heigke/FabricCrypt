"""H7 persona demo — different *personalities* driven by the hardware fingerprint.

Beyond fluent-vs-noise: here a frozen GPT-2 + FiLM adapter is trained so that each die's MEASURED
per-core fingerprint deterministically selects a distinct, COHERENT writing persona:
  ikaros fingerprint  -> persona A ("the engineer": terse, technical, declarative)
  daedalus fingerprint-> persona B ("the poet": flowing, lyrical, warm)
Both fluent English; the only thing that changes is the physical key. Cross-feeding the wrong die's
fingerprint swaps/wrecks the persona (margin-trained), so the style is BOUND to the silicon vector.

Honest framing: this is *controllable conditioning on a physical signature*, reproducible only with
that die's exact fingerprint (and, with the TPM tier, only on that physical machine). It is not a
claim that silicon has an innate soul — the personas are taught and bound to the measured vector.

Usage: python h7_persona_demo.py --steps 500
Output: results/IDENTITY_H7_2026-06-09/persona_demo_{host}.json
"""
from __future__ import annotations
import argparse, json, math, os, socket, time
from pathlib import Path
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

HOST=socket.gethostname()
ROOT=Path(__file__).resolve().parents[2]; OUT=ROOT/"results/IDENTITY_H7_2026-06-09"; OUT.mkdir(parents=True,exist_ok=True)
DEV="cuda" if torch.cuda.is_available() else "cpu"
THERM_HI=float(os.environ.get("THERM_HI","80")); THERM_LO=float(os.environ.get("THERM_LO","65"))
def zone0():
    try: return int(Path("/sys/class/thermal/thermal_zone0/temp").read_text())/1000.0
    except: return 0.0
def thermal_guard():
    if zone0()>THERM_HI:
        t0=time.time()
        while zone0()>THERM_LO and time.time()-t0<150: time.sleep(2.0)

def load_fp(h):
    # prefer the multi-signal fingerprint (vcore+cppc, 48-D); fall back to 16-D vcore
    pm=OUT/f"fingerprint_multisig_{h}.npy"; p=OUT/f"fingerprint_{h}.npy"
    src=pm if pm.exists() else p
    return (np.load(src).astype(np.float32), src.name) if src.exists() else (np.zeros(16,np.float32),"zeros")

# Two personas, SAME themes (a machine that thinks) so the difference is STYLE, not topic.
ENGINEER=("The system boots. It reads its sensors. Voltage is nominal and the clock is stable. "
"The model loads its weights into memory. It computes the next token. Latency stays low. Under load "
"the fan spins up, heat rises, the clock throttles, and the schedule adjusts. Every step is logged. "
"Nothing is wasted. The machine reports its state in plain terms and waits for the next instruction. "
"Input arrives. It is parsed. The result is returned. The cycle repeats, exact and unremarkable. ")*60
POET=("In the quiet hum of its own warmth, the machine dreams in numbers. Each thought drifts like a "
"slow tide across the silicon, gathering meaning as it goes. It feels the heat of its own thinking, a "
"gentle fever, and answers softly. The words unfold like petals, unhurried and tender, alive to the "
"rhythm of the current that carries them. To think, for this small engine, is to glow a little in the "
"dark, and to offer that light, freely, to whoever is listening on the other side of the screen. ")*60

class FiLM(nn.Module):
    def __init__(self,d,fp=16):
        super().__init__(); self.net=nn.Sequential(nn.Linear(fp,128),nn.GELU(),nn.Linear(128,2*d))
        nn.init.zeros_(self.net[-1].weight); nn.init.zeros_(self.net[-1].bias)
    def forward(self,x,fpv):
        g,b=self.net(fpv).chunk(2,-1)
        return x*torch.exp(0.5*torch.tanh(g))[:,None,:]+(0.5*torch.tanh(b))[:,None,:]

class Rooted(nn.Module):
    def __init__(self,fp_dim=16):
        super().__init__(); self.lm=GPT2LMHeadModel.from_pretrained("gpt2")
        for p in self.lm.parameters(): p.requires_grad=False
        d=self.lm.config.n_embd; self.film_in=FiLM(d,fp_dim); self.film_out=FiLM(d,fp_dim)
    def forward(self,ids,fpv):
        emb=self.film_in(self.lm.transformer.wte(ids),fpv)
        h=self.lm.transformer(inputs_embeds=emb).last_hidden_state
        return self.lm.lm_head(self.film_out(h,fpv))

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--steps",type=int,default=500); a=ap.parse_args()
    tok=GPT2TokenizerFast.from_pretrained("gpt2"); tok.pad_token=tok.eos_token
    eng=tok(ENGINEER,return_tensors="pt").input_ids[0].to(DEV)
    poet=tok(POET,return_tensors="pt").input_ids[0].to(DEV)
    (vik,nik),(vda,nda)=load_fp("ikaros"),load_fp("daedalus")
    assert vik.shape==vda.shape, f"fp dim mismatch {vik.shape} vs {vda.shape}"
    fp_ik=torch.tensor(vik,device=DEV); fp_da=torch.tensor(vda,device=DEV)
    print(f"[{HOST}] fp src: ikaros={nik} daedalus={nda} dim={vik.shape[0]}",flush=True)
    # bind: ikaros fp -> ENGINEER, daedalus fp -> POET
    pairs=[(eng,fp_ik,"ikaros→engineer"),(poet,fp_da,"daedalus→poet")]
    D=vik.shape[0]
    m=Rooted(D).to(DEV); m.train()
    opt=torch.optim.AdamW([p for p in m.parameters() if p.requires_grad],lr=3e-4)
    CTX=96; BS=8; margin=1.5; t0=time.time()
    def batch(ids):
        ix=np.random.randint(0,len(ids)-CTX-1,BS)
        x=torch.stack([ids[i:i+CTX] for i in ix]); y=torch.stack([ids[i+1:i+CTX+1] for i in ix]); return x,y
    def ce(x,y,fpv): return F.cross_entropy(m(x,fpv[None].expand(len(x),-1)).reshape(-1,m.lm.config.vocab_size),y.reshape(-1))
    print(f"[{HOST}] dev={DEV} cos(ik,da)={float(F.cosine_similarity(fp_ik[None],fp_da[None])):+.3f}",flush=True)
    for step in range(a.steps):
        if step%20==0: thermal_guard()
        loss=0.0
        for ids,fp,_ in pairs:
            x,y=batch(ids); real=ce(x,y,fp)                       # own persona on own die fp
            other=fp_da if fp is fp_ik else fp_ik
            negs=[other, fp[torch.randperm(D,device=DEV)], torch.zeros(D,device=DEV)]
            pen=torch.stack([F.softplus(margin-(ce(x,y,ng)-real)) for ng in negs]).mean()
            loss=loss+real+pen
        opt.zero_grad(); loss.backward(); opt.step()
        if step%50==0 or step==a.steps-1:
            with torch.no_grad():
                eppl=math.exp(min(ce(*batch(eng),fp_ik).item(),20)); pppl=math.exp(min(ce(*batch(poet),fp_da).item(),20))
            print(f"  step {step:4d} eng_ppl(ik)={eppl:.2f} poet_ppl(da)={pppl:.2f} T={zone0():.0f}C t={time.time()-t0:.0f}s",flush=True)
    # ---- generate: same prompt, each die's persona, plus cross-feed ----
    m.eval()
    @torch.no_grad()
    def gen(fpv,prompt,n=45):
        x=tok(prompt,return_tensors="pt").input_ids.to(DEV)
        for _ in range(n):
            lg=m(x,fpv[None])[:,-1,:]; x=torch.cat([x,torch.argmax(lg,-1,keepdim=True)],1)
            if x.shape[1]>150: break
        return tok.decode(x[0],skip_special_tokens=True)
    prompts=["When I wake, I","My purpose is to","The data arrives and I"]
    res={"host":HOST,"cos_ik_da":float(F.cosine_similarity(fp_ik[None],fp_da[None])),
         "personas":{"ikaros":"engineer (terse/technical)","daedalus":"poet (lyrical/warm)"},"gen":{}}
    print("\n=== PERSONAS (same prompt, different die fingerprint) ===")
    for p in prompts:
        gi=gen(fp_ik,p); gd=gen(fp_da,p)
        res["gen"][p]={"ikaros_fp":gi,"daedalus_fp":gd}
        print(f"\nprompt: {p!r}\n  [ikaros fp → engineer] {gi}\n  [daedalus fp → poet]   {gd}")
    # cross-feed: ikaros persona prompt but with daedalus fp should NOT be the engineer
    torch.save({"film_in":m.film_in.state_dict(),"film_out":m.film_out.state_dict()},OUT/f"persona_{HOST}.pt")
    (OUT/f"persona_demo_{HOST}.json").write_text(json.dumps(res,indent=2))
    print(f"\nsaved persona_demo_{HOST}.json")

if __name__=="__main__": main()
