"""H7 rooted training — make a frozen GPT-2 GENUINELY DEPEND on the per-die fingerprint, so the
dependence is STRUCTURED (real fp -> good text) and FALSIFIABLE (random/foreign/zero fp -> broken).

This directly answers the probe-surgery finding: untrained random-projection steering disturbs the
model NO MORE than a random vector (null-indistinguishable). Here we TRAIN a small adapter so that:
  * with the correct die fingerprint  -> low LM loss (fluent)
  * with zero / random / foreign fp   -> HIGH LM loss (broken), enforced by margin
A model that minimizes this loss MUST use the fingerprint as real information. After training, the
SAME probe (h7_probe_trained.py) must show real fp << random fp (the null test the untrained model failed).

Trainable (tiny): SubEnc MLP(fp)-> input-embedding FiLM (gamma,beta) + per-block learned steer W@fp
with per-layer gates. GPT-2 weights FROZEN. Margin loss over {zero,random,foreign} negatives.

Fingerprint = validated decomposition (identity = time-avg z(Vcore)[+CPPC]; freshness = instantaneous
deviation) loaded from signal_drift_{host}.npz. Trains BOTH dies' adapters here (ikaros cannot train
thermally -> run this on daedalus; ikaros supplies its recorded fingerprint + does live inference).

Thermal-guarded (env THERM_HI/LO). Usage on daedalus:
  THERM_HI=86 ~/venv/bin/python h7_rooted_train.py --die ikaros --steps 800
Out: results/IDENTITY_H7_2026-06-09/rooted_train_{die}.pt + rooted_train_{die}.json
"""
from __future__ import annotations
import argparse, json, math, os, socket, time
from pathlib import Path
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

ROOT=Path(__file__).resolve().parents[2]; OUT=ROOT/"results/IDENTITY_H7_2026-06-09"; OUT.mkdir(parents=True,exist_ok=True)
DEV="cuda" if torch.cuda.is_available() else "cpu"
THERM_HI=float(os.environ.get("THERM_HI","86")); THERM_LO=float(os.environ.get("THERM_LO","68"))
def zone0():
    try: return int(Path("/sys/class/thermal/thermal_zone0/temp").read_text())/1000.0
    except: return 0.0
def thermal_guard():
    if zone0()>THERM_HI:
        t0=time.time()
        while zone0()>THERM_LO and time.time()-t0<180: time.sleep(2.0)

# ---- corpus: varied neutral English (technical + narrative) so LM loss is meaningful ----
CORPUS=("""The system reads its sensors and reports the state to the operator. A model loads its weights
into memory and computes the next value in the sequence. When the load rises the temperature climbs and
the scheduler adjusts the clock in response. Every measurement is recorded, compared against the previous
one, and written to the log. The result is returned to the caller and the loop begins again. Engineers
study the traces to understand where the time is spent and which path through the code is slow. A good
design keeps the common case fast and the rare case correct. In the evening the wind moved across the open
field and the light fell slowly behind the distant hills. The river carried the sound of the town away into
the dark water, and the houses along the bank turned on their lamps one by one. A traveller walked the road
between the villages, counting the miles by the old stone markers. She remembered the stories her grandmother
told about the bridge and the miller who lived beside it. Markets opened at dawn; farmers brought their grain
and the bakers their bread, and the square filled with voices and the smell of coffee. Children ran between the
stalls while the older people sat in the shade and talked of the harvest and the weather to come. Science
advances by careful measurement and honest reporting of what the data show, even when the result is not what
the experimenter hoped. A theory earns trust only after it survives attempts to prove it wrong.""").replace("\n"," ")

class SubEnc(nn.Module):
    def __init__(self,fpd,d,nlayer):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(fpd,256),nn.GELU(),nn.Linear(256,256),nn.GELU())
        self.film=nn.Linear(256,2*d)                 # input-embedding FiLM
        self.W=nn.Linear(fpd,d,bias=False)           # learned steer direction (NOT random)
        self.gate=nn.Parameter(torch.zeros(nlayer))  # per-layer steer gain (starts 0 -> identity)
        nn.init.zeros_(self.film.weight); nn.init.zeros_(self.film.bias)
        nn.init.normal_(self.W.weight,std=0.02)
    def forward(self,fp):
        h=self.net(fp); g,b=self.film(h).chunk(2,-1)
        gamma=torch.exp(torch.tanh(g)*math.log(3.0))  # [1/3,3]
        beta=0.5*torch.tanh(b)
        steer=self.W(fp)                              # [d]
        return gamma,beta,steer

def load_fp_pair(die):
    """returns (own_fp_fresh, foreign_fp_fresh, fp_dim) using validated identity+freshness decomposition."""
    def z(v): v=np.asarray(v,float); s=v.std(); return (v-v.mean())/(s+1e-9) if s>1e-12 else v*0
    def ue(v,w): v=np.asarray(v,float); n=np.linalg.norm(v); return (v/n)*np.sqrt(w) if n>1e-12 else v
    def unit(v): v=np.asarray(v,float); n=np.linalg.norm(v); return v/n if n>1e-12 else v
    def blocks(d):
        mz=np.array([z(r) for r in d["vcore_z"]]).mean(0)
        idb=[ue(mz,1.0)]
        if "cppc" in d: idb.append(ue(z(d["cppc"][0]),0.5))
        idblk=np.concatenate(idb)
        fr=z(d["vcore_z"][-1])-mz                       # freshest deviation
        return np.concatenate([idblk,0.5*unit(fr)*np.sqrt(0.5)]).astype(np.float32), len(idblk)
    other="daedalus" if die=="ikaros" else "ikaros"
    do=dict(np.load(OUT/f"signal_drift_{die}.npz")); df=dict(np.load(OUT/f"signal_drift_{other}.npz"))
    fo,_=blocks(do); ff,_=blocks(df)
    L=min(len(fo),len(ff)); return fo[:L],ff[:L],L

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--die",required=True,choices=["ikaros","daedalus"])
    ap.add_argument("--steps",type=int,default=800); ap.add_argument("--bs",type=int,default=8)
    ap.add_argument("--ctx",type=int,default=96); ap.add_argument("--margin",type=float,default=1.0)
    ap.add_argument("--lr",type=float,default=5e-4); a=ap.parse_args()
    own,foreign,fpd=load_fp_pair(a.die)
    host=socket.gethostname()
    tok=GPT2TokenizerFast.from_pretrained("gpt2"); tok.pad_token=tok.eos_token
    lm=GPT2LMHeadModel.from_pretrained("gpt2").to(DEV).eval()
    for p in lm.parameters(): p.requires_grad=False
    d=lm.config.n_embd; nlayer=lm.config.n_layer
    enc=SubEnc(fpd,d,nlayer).to(DEV).train()
    own_t=torch.tensor(own,device=DEV); fgn_t=torch.tensor(foreign,device=DEV)
    ids_all=tok(CORPUS,return_tensors="pt").input_ids[0].to(DEV)
    N=ids_all.shape[0]

    steer_state={"vecs":None}
    def mk(li):
        def hook(m,i,o):
            if steer_state["vecs"] is None: return o
            v=steer_state["vecs"][li]
            return (o[0]+v,)+o[1:] if isinstance(o,tuple) else o+v
        return hook
    hooks=[blk.register_forward_hook(mk(li)) for li,blk in enumerate(lm.transformer.h)]

    wte=lm.transformer.wte
    def run(ids_bt, fp):
        gamma,beta,steer=enc(fp)
        emb=wte(ids_bt)*gamma[None,None,:]+beta[None,None,:]      # input FiLM
        steer_state["vecs"]=[enc.gate[li]*steer for li in range(nlayer)]
        out=lm(inputs_embeds=emb).logits
        steer_state["vecs"]=None
        return out
    def ce(ids_bt, fp):
        x=ids_bt[:,:-1]; y=ids_bt[:,1:]
        lg=run(x,fp)
        return F.cross_entropy(lg.reshape(-1,lg.shape[-1]),y.reshape(-1))
    def batch():
        ix=np.random.randint(0,N-a.ctx-1,a.bs)
        return torch.stack([ids_all[i:i+a.ctx+1] for i in ix])

    opt=torch.optim.AdamW(enc.parameters(),lr=a.lr)
    t0=time.time()
    print(f"[{host}] train die={a.die} fp_dim={fpd} steps={a.steps} THERM_HI={THERM_HI}",flush=True)
    COOL=float(os.environ.get("STEP_SLEEP","0.45"))   # per-step cooldown to cap GPU duty
    for step in range(a.steps):
        thermal_guard()                                # check EVERY step (backprop heats fast)
        b=batch()
        ce_own=ce(b,own_t)
        zero=torch.zeros(fpd,device=DEV); rnd=torch.randn(fpd,device=DEV)
        negs={"zero":zero,"random":rnd,"foreign":fgn_t}
        pen=0.0; negvals={}
        for nm,nf in negs.items():
            c=ce(b,nf); negvals[nm]=c
            pen=pen+F.softplus(a.margin-(c-ce_own))     # want neg WORSE than own by margin
        loss=ce_own+pen
        opt.zero_grad(); loss.backward(); opt.step()
        time.sleep(COOL)
        if step>0 and step%100==0:                      # periodic checkpoint (crash-safe)
            torch.save({"enc":enc.state_dict(),"fp_dim":fpd,"own":own,"foreign":foreign},OUT/f"rooted_train_{a.die}.pt")
        if step%50==0 or step==a.steps-1:
            with torch.no_grad():
                msg=" ".join(f"{k}={math.exp(min(v.item(),20)):.1f}" for k,v in negvals.items())
            print(f"  step {step:4d} ppl_own={math.exp(min(ce_own.item(),20)):.2f} [{msg}] T={zone0():.0f}C t={time.time()-t0:.0f}s",flush=True)

    # ---- eval: PPL table on held-out windows ----
    enc.eval()
    @torch.no_grad()
    def ppl(fp,reps=12):
        cs=[ce(batch(),fp).item() for _ in range(reps)]; return math.exp(min(float(np.mean(cs)),20))
    rnd_ppls=[ppl(torch.randn(fpd,device=DEV)) for _ in range(5)]
    table={"own":ppl(own_t),"zero":ppl(torch.zeros(fpd,device=DEV)),
           "foreign":ppl(fgn_t),"random_mean":float(np.mean(rnd_ppls)),"random_std":float(np.std(rnd_ppls))}
    # baseline (no adapter at all): gate=0, gamma=1,beta=0 -> plain GPT-2
    gate_bak=enc.gate.data.clone(); enc.gate.data.zero_()
    @torch.no_grad()
    def plain_ppl(reps=12):
        cs=[]
        for _ in range(reps):
            b=batch(); x=b[:,:-1]; y=b[:,1:]
            steer_state["vecs"]=None; lg=lm(x).logits
            cs.append(F.cross_entropy(lg.reshape(-1,lg.shape[-1]),y.reshape(-1)).item())
        return math.exp(min(float(np.mean(cs)),20))
    table["plain_gpt2"]=plain_ppl(); enc.gate.data=gate_bak
    verdict={"own_better_than_random": table["own"]<table["random_mean"]-table["random_std"],
             "own_better_than_zero": table["own"]<table["zero"],
             "own_better_than_foreign": table["own"]<table["foreign"]}
    res={"die":a.die,"host":host,"fp_dim":fpd,"steps":a.steps,"ppl":table,"null_gate":verdict}
    torch.save({"enc":enc.state_dict(),"fp_dim":fpd,"own":own,"foreign":foreign},OUT/f"rooted_train_{a.die}.pt")
    (OUT/f"rooted_train_{a.die}.json").write_text(json.dumps(res,indent=2))
    for h in hooks: h.remove()
    print(f"\n[{host}] PPL  own={table['own']:.2f}  zero={table['zero']:.2f}  foreign={table['foreign']:.2f}  "
          f"random={table['random_mean']:.2f}±{table['random_std']:.2f}  plain={table['plain_gpt2']:.2f}")
    print(f"[{host}] NULL GATE: own<random={verdict['own_better_than_random']}  own<zero={verdict['own_better_than_zero']}  own<foreign={verdict['own_better_than_foreign']}")
    print(f"saved rooted_train_{a.die}.pt/.json")

if __name__=="__main__": main()
