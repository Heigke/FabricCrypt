"""H7 emergent persona — the hardware's own signals drive the voice, with NO chosen persona
and NO training. We do NOT label "engineer"/"poet" or train toward any target style.

Mechanism (activation steering / representation engineering):
  * Frozen GPT-2 (no weights changed, no FiLM trained).
  * The die's measured 48-D fingerprint vector (per-core Vcore + CPPC) is projected through a
    FIXED, deterministic transducer P (seed-fixed, identical on every machine — it is the
    "wiring", not a per-die choice) into the model's hidden space: v = alpha * unit(P @ fp).
  * v is ADDED to the residual stream at every block. The frozen model then "thinks with" the
    chip's own numbers mixed in. Whatever coherent-but-distinct voice emerges IS that chip's.

We choose exactly ONE knob: alpha (injection strength). Too low -> no effect; too high -> noise.
We do NOT choose the content. Different die -> different fp -> different emergent voice,
reproducible only with that die's exact measured numbers.

Inference-only (forward passes) -> light -> safe to run on each machine on its own body.
Usage: python h7_emergent_persona.py --alphas 0,4,8,12 [--fp ikaros|daedalus|self]
"""
from __future__ import annotations
import argparse, json, socket
from pathlib import Path
import numpy as np, torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

HOST=socket.gethostname()
OUT=Path(__file__).resolve().parents[2]/"results/IDENTITY_H7_2026-06-09"
DEV="cuda" if torch.cuda.is_available() else "cpu"

def load_fp(h):
    pm=OUT/f"fingerprint_multisig_{h}.npy"; p=OUT/f"fingerprint_{h}.npy"
    s=pm if pm.exists() else p
    return np.load(s).astype(np.float32)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--alphas",default="0,4,8,12")
    ap.add_argument("--fp",default="both",help="ikaros|daedalus|both|self")
    ap.add_argument("--prompt",default="When I wake, I")
    ap.add_argument("--prompts",default="",help="semicolon-separated; overrides --prompt")
    a=ap.parse_args()
    PROMPTS=[p.strip() for p in a.prompts.split(";") if p.strip()] or [a.prompt]
    tok=GPT2TokenizerFast.from_pretrained("gpt2"); lm=GPT2LMHeadModel.from_pretrained("gpt2").to(DEV).eval()
    for p in lm.parameters(): p.requires_grad=False
    d=lm.config.n_embd
    # FIXED transducer P (same on every machine): deterministic, NOT trained, NOT per-die.
    g=torch.Generator().manual_seed(0); P=torch.randn(d,48,generator=g).to(DEV)
    P=P/P.norm(dim=0,keepdim=True)

    dies = (["ikaros","daedalus"] if a.fp=="both" else
            [HOST if "ikaros" in HOST else "daedalus"] if a.fp=="self" else [a.fp])
    fps={h:torch.tensor(load_fp(h),device=DEV) for h in dies}

    steer={"vec":None}
    hooks=[]
    def mk():
        def hook(mod,inp,out):
            if steer["vec"] is None: return out
            if isinstance(out,tuple): return (out[0]+steer["vec"],)+out[1:]
            return out+steer["vec"]
        return hook
    for blk in lm.transformer.h: hooks.append(blk.register_forward_hook(mk()))

    @torch.no_grad()
    def gen(fp,alpha,prompt,n=40):
        steer["vec"]= None if alpha==0 else (alpha*(P@fp)/(P@fp).norm())
        x=tok(prompt,return_tensors="pt").input_ids.to(DEV)
        for _ in range(n):
            lg=lm(x).logits[:,-1,:]; x=torch.cat([x,torch.argmax(lg,-1,keepdim=True)],1)
        steer["vec"]=None
        return tok.decode(x[0],skip_special_tokens=True)

    alphas=[float(s) for s in a.alphas.split(",")]
    res={"host":HOST,"prompts":PROMPTS,"transducer":"fixed seed=0, identical per machine",
         "computed_on":HOST,"gen":{}}
    print(f"[{HOST}] emergent persona — NO training, NO chosen persona. computed_on={HOST}\n")
    for h,fp in fps.items():
        res["gen"][h]={}
        print(f"### die={h}  (fp dim={fp.shape[0]}, computed on {HOST})")
        for pr in PROMPTS:
            res["gen"][h][pr]={}
            for al in alphas:
                t=gen(fp,al,pr); res["gen"][h][pr][f"alpha={al}"]=t
                print(f"  [{pr!r} α={al}] {t}")
        print()
    for hk in hooks: hk.remove()
    (OUT/f"emergent_persona_{HOST}.json").write_text(json.dumps(res,indent=2))
    print(f"saved emergent_persona_{HOST}.json")

if __name__=="__main__": main()
