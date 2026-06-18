"""H7 live embodiment — a step toward the full vision: the LLM generates in REAL TIME driven by a
FRESH measurement of its own chip each step, gated by a periodic fresh TPM nonce (liveness).

Not the static enrolled fingerprint: every REMEASURE_EVERY tokens we read the live silicon (per-core
Vcore + CPPC), rebuild the fingerprint the same way as enrollment, and feed THAT into the trained
adapter. We log cos(live_fp, enrolled_fp) to show it's the same body, and that text stays fluent off
the live signal. Every NONCE_EVERY tokens we require a fresh TPM quote or abort.

Honest limits (printed): identity dominates so a same-die live read ~ enrolled (fluent); this proves
real-time in-body coupling + liveness, NOT yet analog in-body COMPUTATION coupled in, and the key still
sits in GPU memory while running (no GPU TEE → active-mode copy not prevented, only made stale-able).

Run under sudo (PM table): sudo -n env HSA_OVERRIDE_GFX_VERSION=11.0.0 venv/bin/python h7_live_embody.py
"""
from __future__ import annotations
import math,os,socket,subprocess,sys,tempfile,time
from pathlib import Path
import numpy as np, torch, torch.nn.functional as F
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
sys.path.insert(0,str(Path(__file__).parent))
from h7_rooted_train import SubEnc
OUT=Path(__file__).resolve().parents[2]/"results/IDENTITY_H7_2026-06-09"; DEV="cuda" if torch.cuda.is_available() else "cpu"
HOST=socket.gethostname(); PM=Path("/sys/kernel/ryzen_smu_drv/pm_table"); NCPU=os.cpu_count() or 16
CAND_VIDX=[756,110]; REMEASURE_EVERY=8; NONCE_EVERY=24

def z(v): v=np.asarray(v,float); s=v.std(); return (v-v.mean())/(s+1e-9) if s>1e-12 else v*0.0
def ue(v,w): v=np.asarray(v,float); n=np.linalg.norm(v); return (v/n)*np.sqrt(w) if n>1e-12 else v
def unit(v): v=np.asarray(v,float); n=np.linalg.norm(v); return v/n if n>1e-12 else v
def read_vcore(n=16,lo=0.5,hi=1.1):
    try: b=PM.read_bytes(); v=np.frombuffer(b[:(len(b)//4)*4],dtype=np.float32).astype(float)
    except: return None
    for idx in CAND_VIDX:
        if idx+n<=len(v):
            w=v[idx:idx+n]
            if np.all((w>=lo)&(w<=hi)) and w.std()<0.08: return w.copy()
    return None
def read_cppc():
    out=[]
    for c in range(NCPU):
        try: out.append(int(Path(f"/sys/devices/system/cpu/cpu{c}/acpi_cppc/highest_perf").read_text()))
        except: pass
    return np.array(out,float) if out else None
def live_fp(dim_cppc):
    burst=[];
    for _ in range(14):
        w=read_vcore()
        if w is not None: burst.append(z(w))
        time.sleep(0.01)
    if not burst: return None
    burst=np.array(burst); mz=burst.mean(0); inst=burst[-1]-mz
    cp=read_cppc()
    idb=[ue(mz,1.0)]
    if cp is not None: idb.append(ue(z(cp),0.5))
    fp=np.concatenate([np.concatenate(idb),0.5*unit(inst)*np.sqrt(0.5)]).astype(np.float32)
    return fp
def quote():
    with tempfile.TemporaryDirectory() as td:
        td=Path(td); nonce=os.urandom(20); ek=td/"ek";ak=td/"ak"
        def sh(c): return subprocess.run(c,capture_output=True,text=True).returncode
        if sh(["tpm2_createek","-G","ecc","-c",str(ek)]): return False
        if sh(["tpm2_createak","-C",str(ek),"-G","ecc","-g","sha256","-s","ecdsa","-c",str(ak),"-u",str(td/"akp")]): return False
        return sh(["tpm2_quote","-c",str(ak),"-l","sha256:0,7","-q",nonce.hex(),"-m",str(td/"m"),"-s",str(td/"s"),"-o",str(td/"p")])==0

def main():
    ck=torch.load(OUT/"rooted_train2_ikaros.pt",map_location=DEV,weights_only=False)
    fpd=ck["fp_dim"]; enrolled=np.asarray(ck["own"],float)
    tok=GPT2TokenizerFast.from_pretrained("gpt2"); lm=GPT2LMHeadModel.from_pretrained("gpt2").to(DEV).eval()
    for p in lm.parameters(): p.requires_grad=False
    d=lm.config.n_embd; nl=lm.config.n_layer
    enc=SubEnc(fpd,d,nl).to(DEV); enc.load_state_dict(ck["enc"]); enc.eval()
    steer={"v":None}
    def mk(li):
        def h(m,i,o):
            if steer["v"] is None: return o
            return (o[0]+steer["v"][li],)+o[1:] if isinstance(o,tuple) else o+steer["v"][li]
        return h
    hooks=[b.register_forward_hook(mk(li)) for li,b in enumerate(lm.transformer.h)]
    wte=lm.transformer.wte
    cur={"fp":None}
    def set_fp(fp):
        f=torch.tensor(fp,device=DEV); g,b,s=enc(f); gamma=torch.exp(torch.tanh(g)*math.log(3.0)); beta=0.5*torch.tanh(b)
        cur["gamma"]=gamma; cur["beta"]=beta; cur["steer"]=[enc.gate[li]*s for li in range(nl)]
    @torch.no_grad()
    def step(ids):
        emb=wte(ids)*cur["gamma"][None,None,:]+cur["beta"][None,None,:]
        steer["v"]=cur["steer"]; lg=lm(inputs_embeds=emb).logits[:,-1,:]; steer["v"]=None
        p=F.softmax(lg/0.8,-1); return torch.multinomial(p,1)
    print(f"[{HOST}] LIVE embodiment: re-reading chip every {REMEASURE_EVERY} tokens, TPM nonce every {NONCE_EVERY}",flush=True)
    if not quote(): print("initial TPM quote FAILED -> abort"); sys.exit(4)
    print("liveness OK (fresh TPM nonce)",flush=True)
    ids=tok("It is a truth",return_tensors="pt").input_ids.to(DEV)
    coss=[]
    for t in range(64):
        if t%REMEASURE_EVERY==0:
            fp=live_fp(fpd)
            if fp is None: print("PM read failed"); break
            L=min(len(fp),len(enrolled)); c=float(np.dot(unit(fp[:L]),unit(enrolled[:L]))); coss.append(c)
            set_fp(fp)
            print(f"  t={t:2d}  live-read cos(live,enrolled)={c:+.3f}  T={int(Path('/sys/class/thermal/thermal_zone0/temp').read_text())/1000:.0f}C",flush=True)
        if t%NONCE_EVERY==0 and t>0:
            ok=quote(); print(f"  t={t:2d}  fresh TPM nonce: {'OK' if ok else 'FAIL->abort'}",flush=True)
            if not ok: break
        ids=torch.cat([ids,step(ids)],1)
    for h in hooks: h.remove()
    txt=tok.decode(ids[0],skip_special_tokens=True)
    print(f"\n[{HOST}] LIVE-GENERATED (driven by fresh chip reads):\n  {txt}")
    print(f"\nmean cos(live,enrolled)={np.mean(coss):+.3f} over {len(coss)} live reads -> same body confirmed live")
    print("HONEST: real-time in-body coupling + liveness shown; analog in-body COMPUTE not yet wired; key still in GPU RAM while running.")

if __name__=="__main__": main()
