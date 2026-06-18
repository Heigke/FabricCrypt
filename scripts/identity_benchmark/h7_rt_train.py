"""H7 real-time-ONLY rooted model — the embodiment layer driven purely by LIVE hardware signals.
NO fused constant (CPPC dropped), NO TPM key in the conditioning. Fingerprint built only from channels
that the inventory proved genuinely move: per-core Vcore + per-core clock (scaling_cur_freq) + power.

identity = time-averaged z(channel) per block (denoised live signal) ; freshness = instantaneous deviation.
Frozen GPT-2 + tiny adapter; loss = CE_own + base-anchor + margins over {zero,random,foreign}. If this
binds behaviour to the body, then the LIVE signals alone carry the embodiment (TPM is then only the lock).

Usage (run on the die itself): THERM_HI=.. python h7_rt_train.py --die ikaros --textfile /tmp/pg1342.txt
Out: rt_train_{die}.pt + rt_train_{die}.json
"""
from __future__ import annotations
import argparse,json,math,os,socket,sys,time
from pathlib import Path
import numpy as np, torch, torch.nn.functional as F
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
sys.path.insert(0,str(Path(__file__).parent))
from h7_rooted_train import SubEnc, thermal_guard, zone0, OUT, DEV
LIVE=["vcore","cur_freq","power"]   # channels proven live on BOTH dies (consistent dim)

def z(v): v=np.asarray(v,float); s=v.std(); return (v-v.mean())/(s+1e-9) if s>1e-12 else v*0.0
def ue(v,w): v=np.asarray(v,float); n=np.linalg.norm(v); return (v/n)*np.sqrt(w) if n>1e-12 else v
def unit(v): v=np.asarray(v,float); n=np.linalg.norm(v); return v/n if n>1e-12 else v
def build_fp(npz):
    idp=[]; devp=[]
    for ch,w in [("vcore",1.0),("cur_freq",1.0),("power",0.5)]:
        if ch not in npz: continue
        M=np.array([z(r) for r in npz[ch]]); mz=M.mean(0); idp.append(ue(mz,w)); devp.append(M[-1]-mz)
    ident=np.concatenate(idp); fresh=0.5*unit(np.concatenate(devp))*np.sqrt(0.5)
    return np.concatenate([ident,fresh]).astype(np.float32)
def load_pair(die):
    other="daedalus" if die=="ikaros" else "ikaros"
    do=dict(np.load(OUT/f"rt_signals_{die}.npz")); df=dict(np.load(OUT/f"rt_signals_{other}.npz"))
    fo=build_fp(do); ff=build_fp(df); L=min(len(fo),len(ff)); return fo[:L],ff[:L],L

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--die",required=True,choices=["ikaros","daedalus"])
    ap.add_argument("--steps",type=int,default=450); ap.add_argument("--bs",type=int,default=2); ap.add_argument("--ctx",type=int,default=64)
    ap.add_argument("--margin",type=float,default=0.5); ap.add_argument("--lam",type=float,default=2.0); ap.add_argument("--delta",type=float,default=0.15)
    ap.add_argument("--textfile",default="/tmp/pg1342.txt"); ap.add_argument("--maxtok",type=int,default=16000); a=ap.parse_args()
    own,foreign,fpd=load_pair(a.die); host=socket.gethostname()
    tok=GPT2TokenizerFast.from_pretrained("gpt2"); tok.pad_token=tok.eos_token
    lm=GPT2LMHeadModel.from_pretrained("gpt2").to(DEV).eval()
    for p in lm.parameters(): p.requires_grad=False
    d=lm.config.n_embd; nl=lm.config.n_layer
    enc=SubEnc(fpd,d,nl).to(DEV).train()
    own_t=torch.tensor(own,device=DEV); fgn_t=torch.tensor(foreign,device=DEV)
    raw=Path(a.textfile).read_text(errors="ignore")
    if "*** START" in raw: raw=raw.split("*** START",1)[1].split("***",1)[-1]
    if "*** END" in raw: raw=raw.split("*** END",1)[0]
    ids=tok(" ".join(raw.split()),return_tensors="pt").input_ids[0][:a.maxtok].to(DEV); N=ids.shape[0]
    split=int(N*0.85); tr=ids[:split]; va=ids[split:]
    print(f"[{host}] RT-only die={a.die} fp={fpd} (live-only: Vcore+clock+power) train={split} val={N-split}",flush=True)
    steer={"vecs":None}
    def mk(li):
        def h(m,i,o):
            if steer["vecs"] is None: return o
            v=steer["vecs"][li]; return (o[0]+v,)+o[1:] if isinstance(o,tuple) else o+v
        return h
    hooks=[b.register_forward_hook(mk(li)) for li,b in enumerate(lm.transformer.h)]; wte=lm.transformer.wte
    def ce(b,fp):
        x=b[:,:-1]; y=b[:,1:]
        if fp is None: steer["vecs"]=None; lg=lm(x).logits
        else:
            g,bb,s=enc(fp); gamma=torch.exp(torch.tanh(g)*math.log(3.0)); beta=0.5*torch.tanh(bb)
            emb=wte(x)*gamma[None,None,:]+beta[None,None,:]; steer["vecs"]=[enc.gate[li]*s for li in range(nl)]
            lg=lm(inputs_embeds=emb).logits; steer["vecs"]=None
        return F.cross_entropy(lg.reshape(-1,lg.shape[-1]),y.reshape(-1))
    def batch(src):
        ix=np.random.randint(0,len(src)-a.ctx-1,a.bs); return torch.stack([src[i:i+a.ctx+1] for i in ix])
    opt=torch.optim.AdamW(enc.parameters(),lr=5e-4); COOL=float(os.environ.get("STEP_SLEEP","0.6")); t0=time.time()
    for step in range(a.steps):
        thermal_guard(); b=batch(tr)
        with torch.no_grad(): cp=ce(b,None)
        co=ce(b,own_t); anchor=a.lam*F.relu(co-cp-a.delta)
        negs=[torch.zeros(fpd,device=DEV),torch.randn(fpd,device=DEV),fgn_t]
        pen=sum(F.softplus(a.margin-(ce(b,nf)-co)) for nf in negs)
        loss=co+anchor+pen; opt.zero_grad(); loss.backward(); opt.step(); time.sleep(COOL)
        if step>0 and step%100==0: torch.save({"enc":enc.state_dict(),"fp_dim":fpd,"own":own,"foreign":foreign},OUT/f"rt_train_{a.die}.pt")
        if step%50==0 or step==a.steps-1:
            with torch.no_grad(): vo=math.exp(min(ce(batch(va),own_t).item(),20)); vp=math.exp(min(ce(batch(va),None).item(),20))
            print(f"  step {step:4d} VAL own={vo:.2f} plain={vp:.2f} T={zone0():.0f}C t={time.time()-t0:.0f}s",flush=True)
    enc.eval()
    @torch.no_grad()
    def ppl(fp,reps=12): return math.exp(min(float(np.mean([ce(batch(va),fp).item() for _ in range(reps)])),20))
    rv=[ppl(torch.randn(fpd,device=DEV)) for _ in range(6)]
    val={"own":ppl(own_t),"zero":ppl(torch.zeros(fpd,device=DEV)),"foreign":ppl(fgn_t),"random_median":float(np.median(rv)),"plain":ppl(None)}
    gate={"own_below_all_random":val["own"]<min(rv),"own_below_zero":val["own"]<val["zero"],"own_below_foreign":val["own"]<val["foreign"],"own_near_plain":val["own"]<=val["plain"]*1.5}
    torch.save({"enc":enc.state_dict(),"fp_dim":fpd,"own":own,"foreign":foreign},OUT/f"rt_train_{a.die}.pt")
    (OUT/f"rt_train_{a.die}.json").write_text(json.dumps({"die":a.die,"channels":LIVE,"fp_dim":fpd,"val_ppl":val,"null_gate":gate},indent=2))
    for h in hooks: h.remove()
    print(f"\n[{a.die}] RT-ONLY VAL own={val['own']:.2f} plain={val['plain']:.2f} zero={val['zero']:.2f} foreign={val['foreign']:.2f} random~{val['random_median']:.0f}")
    print(f"[{a.die}] GATE own<allrand={gate['own_below_all_random']} own<zero={gate['own_below_zero']} own<foreign={gate['own_below_foreign']} own≈plain={gate['own_near_plain']}")

if __name__=="__main__": main()
