"""H7 rooted training v2 — keep the (proven) fingerprint DEPENDENCE but recover GENERALISING fluency.

v1 finding: own<<random/foreign/zero by 100-200x (dependence real, deterministic) BUT own held-out
PPL 636 >> plain 60 -> the heavy margin + tiny corpus overfit and degraded base fluency.

v2 fixes:
  1. BIGGER, more varied corpus (multi-paragraph), 85/15 train/val split (val never trained) so we
     monitor GENERALISATION directly during training.
  2. BASE-QUALITY ANCHOR: loss += LAMBDA*relu(ce_own - ce_plain - DELTA) -> own must stay within DELTA
     nats of plain GPT-2 (keeps text good), while margins still push negatives away.
  3. Gentler margin (0.5) so separation does not require nuking quality.
Reuses SubEnc / load_fp_pair from h7_rooted_train. Thermal-guarded (same env).
Out: rooted_train2_{die}.pt + rooted_train2_{die}.json (reports TRAIN and HELD-OUT ppl tables).
"""
from __future__ import annotations
import argparse, json, math, os, socket, sys, time
from pathlib import Path
import numpy as np, torch, torch.nn.functional as F
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
sys.path.insert(0,str(Path(__file__).parent))
from h7_rooted_train import SubEnc, load_fp_pair, thermal_guard, zone0, OUT, DEV

CORPUS=("""The system reads its sensors and reports the state to the operator. A model loads its weights
into memory and computes the next value in the sequence. When the load rises the temperature climbs and
the scheduler lowers the clock until the heat settles. Every measurement is recorded, compared with the
previous reading, and written to the log for later study. Engineers trace the slow paths and keep the
common case fast while the rare case stays correct. A careful design fails loudly rather than quietly.
In the morning the market opens along the river and the farmers set out their grain and apples. The baker
carries warm bread to the corner stall and the smell drifts over the square. Children chase each other
between the carts while the older people sit in the shade and talk about the weather and the harvest. A
traveller follows the road north, counting the miles by the old stone markers her grandmother once named.
The ship leaves the harbour before sunrise, its grey sails dim against the paling sky. The captain checks
the chart and the compass and calls a new heading toward the northern islands. Below deck the cook lights
the stove and a thin smoke rises through the planks. By noon the wind freshens, the hull leans, and spray
crosses the rail with every wave. History is written slowly, one careful record at a time, and the past is
known only as well as its evidence allows. A claim earns trust after it survives honest attempts to prove
it wrong, not before. The student repeats the experiment a third time, adjusting the temperature and noting
the current at each step; the instrument drifts, so she calibrates it again and writes the correction down.
Numbers that cannot be reproduced are not yet knowledge. Far to the south the desert holds its heat long
after dark, and the stars come out hard and bright above the cooling sand. A scholar copies an old letter by
lamplight, weighing each word, aware that a single mistake will travel down the centuries. In the workshop the
smith works the iron while it glows, and the hammer falls in an even rhythm that the apprentices learn to keep.
Good work, like good thinking, is mostly patience and attention repeated until it becomes a habit.""").replace("\n"," ")

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--die",required=True,choices=["ikaros","daedalus"])
    ap.add_argument("--steps",type=int,default=400); ap.add_argument("--bs",type=int,default=2)
    ap.add_argument("--ctx",type=int,default=64); ap.add_argument("--margin",type=float,default=0.5)
    ap.add_argument("--lam",type=float,default=2.0); ap.add_argument("--delta",type=float,default=0.15)
    ap.add_argument("--lr",type=float,default=5e-4)
    ap.add_argument("--textfile",default="",help="path to a large plain-text corpus (else built-in)")
    ap.add_argument("--maxtok",type=int,default=16000); a=ap.parse_args()
    own,foreign,fpd=load_fp_pair(a.die); host=socket.gethostname()
    tok=GPT2TokenizerFast.from_pretrained("gpt2"); tok.pad_token=tok.eos_token
    lm=GPT2LMHeadModel.from_pretrained("gpt2").to(DEV).eval()
    for p in lm.parameters(): p.requires_grad=False
    d=lm.config.n_embd; nlayer=lm.config.n_layer
    enc=SubEnc(fpd,d,nlayer).to(DEV).train()
    own_t=torch.tensor(own,device=DEV); fgn_t=torch.tensor(foreign,device=DEV)
    if a.textfile and Path(a.textfile).exists():
        raw=Path(a.textfile).read_text(errors="ignore")
        # strip Project Gutenberg header/footer if present
        if "*** START" in raw: raw=raw.split("*** START",1)[1].split("***",1)[-1]
        if "*** END" in raw: raw=raw.split("*** END",1)[0]
        raw=" ".join(raw.split())
        ids=tok(raw,return_tensors="pt").input_ids[0][:a.maxtok].to(DEV)
    else:
        ids=tok(CORPUS,return_tensors="pt").input_ids[0].to(DEV)
    N=ids.shape[0]
    split=int(N*0.85); tr=ids[:split]; va=ids[split:]
    print(f"[{host}] v2 die={a.die} fp={fpd} corpus={N}tok train={split} val={N-split} steps={a.steps}",flush=True)

    steer={"vecs":None}
    def mk(li):
        def hook(m,i,o):
            if steer["vecs"] is None: return o
            v=steer["vecs"][li]; return (o[0]+v,)+o[1:] if isinstance(o,tuple) else o+v
        return hook
    hooks=[blk.register_forward_hook(mk(li)) for li,blk in enumerate(lm.transformer.h)]
    wte=lm.transformer.wte
    def ce(b,fp):
        x=b[:,:-1]; y=b[:,1:]
        if fp is None:
            steer["vecs"]=None; lg=lm(x).logits
        else:
            g,bb,s=enc(fp); gamma=torch.exp(torch.tanh(g)*math.log(3.0)); beta=0.5*torch.tanh(bb)
            emb=wte(x)*gamma[None,None,:]+beta[None,None,:]
            steer["vecs"]=[enc.gate[li]*s for li in range(nlayer)]
            lg=lm(inputs_embeds=emb).logits; steer["vecs"]=None
        return F.cross_entropy(lg.reshape(-1,lg.shape[-1]),y.reshape(-1))
    def batch(src):
        ix=np.random.randint(0,len(src)-a.ctx-1,a.bs)
        return torch.stack([src[i:i+a.ctx+1] for i in ix])

    opt=torch.optim.AdamW(enc.parameters(),lr=a.lr); COOL=float(os.environ.get("STEP_SLEEP","0.6")); t0=time.time()
    for step in range(a.steps):
        thermal_guard()
        b=batch(tr)
        with torch.no_grad(): ce_plain=ce(b,None)            # frozen baseline quality
        ce_own=ce(b,own_t)
        anchor=a.lam*F.relu(ce_own-ce_plain-a.delta)          # keep own near plain quality
        negs={"zero":torch.zeros(fpd,device=DEV),"random":torch.randn(fpd,device=DEV),"foreign":fgn_t}
        pen=sum(F.softplus(a.margin-(ce(b,nf)-ce_own)) for nf in negs.values())
        loss=ce_own+anchor+pen
        opt.zero_grad(); loss.backward(); opt.step(); time.sleep(COOL)
        if step>0 and step%100==0:
            torch.save({"enc":enc.state_dict(),"fp_dim":fpd,"own":own,"foreign":foreign},OUT/f"rooted_train2_{a.die}.pt")
        if step%50==0 or step==a.steps-1:
            with torch.no_grad(): vown=math.exp(min(ce(batch(va),own_t).item(),20)); vpl=math.exp(min(ce(batch(va),None).item(),20))
            print(f"  step {step:4d} train_ppl_own={math.exp(min(ce_own.item(),20)):.2f} VAL own={vown:.2f} plain={vpl:.2f} anchor={float(anchor):.2f} T={zone0():.0f}C t={time.time()-t0:.0f}s",flush=True)
    enc.eval()
    @torch.no_grad()
    def ppl(src,fp,reps=12): return math.exp(min(float(np.mean([ce(batch(src),fp).item() for _ in range(reps)])),20))
    rv=[ppl(va,torch.randn(fpd,device=DEV)) for _ in range(6)]
    val={"own":ppl(va,own_t),"zero":ppl(va,torch.zeros(fpd,device=DEV)),"foreign":ppl(va,fgn_t),
         "random_median":float(np.median(rv)),"plain":ppl(va,None)}
    gate={"own_below_all_random": val["own"]<min(rv), "own_below_zero":val["own"]<val["zero"],
          "own_below_foreign":val["own"]<val["foreign"], "own_near_plain": val["own"]<=val["plain"]*1.5,
          "random_breaks_x": round(val["random_median"]/max(val["own"],1e-6),1)}
    res={"die":a.die,"host":host,"fp_dim":fpd,"steps":a.steps,"val_ppl":val,"null_gate":gate}
    torch.save({"enc":enc.state_dict(),"fp_dim":fpd,"own":own,"foreign":foreign},OUT/f"rooted_train2_{a.die}.pt")
    (OUT/f"rooted_train2_{a.die}.json").write_text(json.dumps(res,indent=2))
    for h in hooks: h.remove()
    print(f"\n[{a.die}] VAL own={val['own']:.2f} plain={val['plain']:.2f} zero={val['zero']:.2f} foreign={val['foreign']:.2f} random~{val['random_median']:.0f}")
    print(f"[{a.die}] GATE own<allrandom={gate['own_below_all_random']} own<zero={gate['own_below_zero']} own<foreign={gate['own_below_foreign']} own≈plain(≤1.5x)={gate['own_near_plain']} random_breaks={gate['random_breaks_x']}x")
    print(f"saved rooted_train2_{a.die}.json")

if __name__=="__main__": main()
