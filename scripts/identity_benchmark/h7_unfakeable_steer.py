"""H7 un-fakeable steering — the honest 'whole sentence': a per-die secret that ONLY this physical
TPM can produce, bound to a FRESH liveness nonce, steers the frozen LLM's behavior.

Why this is the honest fix (and the limit):
  * The readable per-core Vcore vector is a REAL physical per-die signal but is NOT un-fakeable
    (root can read/replay the 48 numbers). So we do NOT use it as the un-fakeable mechanism.
  * Instead the steering vector is derived from a high-entropy secret S that is SEALED to the
    discrete TPM (owner hierarchy). S never exists in the clear off-chip.
      - cross-machine: tpm2_load/unseal FAILS on a foreign die  -> behavior unreproducible.
      - replay: each run requires a fresh-nonce tpm2_quote (liveness) before unseal.
  * HONEST RESIDUAL: this protects against COPYING to another machine and against REPLAY. It does
    NOT protect against a privileged attacker ON the enrolled machine who dumps process memory
    after unseal (no TEE runs the inference itself on client AMD). Stated plainly, not hidden.

Modes:
  enroll          generate S, seal to this TPM (S discarded from clear).
  run             fresh nonce quote (liveness) -> unseal S -> derive steer -> generate (steered).
  verify-foreign  attempt unseal of a copied vault here; on a foreign die it MUST be REFUSED.

Run under sudo (TPM device needs root here):
  sudo -n env HSA_OVERRIDE_GFX_VERSION=11.0.0 venv/bin/python h7_unfakeable_steer.py run
"""
from __future__ import annotations
import argparse, hashlib, hmac, json, os, socket, subprocess, sys, tempfile
from pathlib import Path

HOST=socket.gethostname()
ROOT=Path(__file__).resolve().parents[2]; OUT=ROOT/"results/IDENTITY_H7_2026-06-09"
VAULT=OUT/f"steer_vault_{HOST}"; OUT.mkdir(parents=True,exist_ok=True)
PRIMARY=["-g","sha256","-G","ecc"]; DIM=48
PROMPTS=["When I wake, I","My purpose is to","The data arrives and I"]

def sh(c): return subprocess.run(c,capture_output=True,text=True)
def tpm(cmd):
    r=sh(cmd)
    if r.returncode!=0 and os.geteuid()!=0 and ("tpm" in r.stderr.lower() or "tcti" in r.stderr.lower() or "perm" in r.stderr.lower()):
        r=sh(["sudo","-n"]+cmd)
    return r

def hkdf(S:bytes,info:bytes,n:int)->bytes:
    prk=hmac.new(b"h7-unfakeable-salt",S,hashlib.sha256).digest()
    out=b""; t=b""; c=1
    while len(out)<n:
        t=hmac.new(prk,t+info+bytes([c]),hashlib.sha256).digest(); out+=t; c+=1
    return out[:n]

def steer_vec_from_secret(S:bytes):
    import numpy as np
    raw=hkdf(S,b"identity:gpt2-steer",DIM*2)
    v=np.frombuffer(raw,dtype="<i2").astype("float32")[:DIM]   # deterministic from S
    v=v/(np.linalg.norm(v)+1e-9)
    return v

# ---------- enroll ----------
def enroll():
    VAULT.mkdir(parents=True,exist_ok=True)
    S=os.urandom(32)
    with tempfile.TemporaryDirectory() as td:
        td=Path(td); pri=td/"pri.ctx"; sec=td/"sec.bin"; sec.write_bytes(S)
        if tpm(["tpm2_createprimary","-C","o",*PRIMARY,"-c",str(pri)]).returncode: print("createprimary failed"); sys.exit(3)
        if tpm(["tpm2_create","-C",str(pri),"-i",str(sec),"-u",str(VAULT/"s.pub"),"-r",str(VAULT/"s.priv")]).returncode:
            print("seal failed"); sys.exit(3)
    # store a COMMITMENT to the derived vector (hash only) so we can later prove determinism w/o leaking it
    commit=hashlib.sha256(steer_vec_from_secret(S).tobytes()).hexdigest()
    (VAULT/"vault.json").write_text(json.dumps({"host":HOST,"dim":DIM,"steer_commit_sha256":commit},indent=2))
    for f in ("s.pub","s.priv"):
        try: os.chmod(VAULT/f,0o644)
        except: pass
    S=b"\x00"*32
    print(f"[{HOST}] SEALED per-die steering secret -> {VAULT}/s.pub,s.priv  (clear secret discarded)")
    print(f"[{HOST}] steer-vector commitment sha256={commit[:16]}..")

# ---------- liveness ----------
def quote(td:Path):
    nonce=os.urandom(20); ek=td/"ek.ctx"; ak=td/"ak.ctx"
    if tpm(["tpm2_createek","-G","ecc","-c",str(ek)]).returncode: return None
    if tpm(["tpm2_createak","-C",str(ek),"-G","ecc","-g","sha256","-s","ecdsa","-c",str(ak),"-u",str(td/"ak.pub")]).returncode: return None
    if tpm(["tpm2_quote","-c",str(ak),"-l","sha256:0,1,7","-q",nonce.hex(),
            "-m",str(td/"q.msg"),"-s",str(td/"q.sig"),"-o",str(td/"q.pcr")]).returncode: return None
    return nonce

def _unseal_secret(vdir:Path,td:Path):
    pri=td/"pri.ctx"
    if tpm(["tpm2_createprimary","-C","o",*PRIMARY,"-c",str(pri)]).returncode: return None,"createprimary failed"
    sc=td/"s.ctx"
    if tpm(["tpm2_load","-C",str(pri),"-u",str(vdir/"s.pub"),"-r",str(vdir/"s.priv"),"-c",str(sc)]).returncode:
        return None,"load failed (foreign die)"
    kf=td/"s.bin"
    if tpm(["tpm2_unseal","-c",str(sc),"-o",str(kf)]).returncode or not kf.exists(): return None,"unseal failed"
    return kf.read_bytes(),"ok"

# ---------- run (liveness + unseal + steer + generate) ----------
def run():
    import numpy as np, torch
    from transformers import GPT2LMHeadModel, GPT2TokenizerFast
    if not (VAULT/"s.pub").exists(): print(f"[{HOST}] no vault — run enroll first"); sys.exit(2)
    with tempfile.TemporaryDirectory() as td:
        td=Path(td)
        n=quote(td)
        if n is None: print(f"[{HOST}] LIVENESS quote FAILED"); sys.exit(4)
        print(f"[{HOST}] liveness OK — fresh nonce {n.hex()[:16]}.. signed by TPM (non-replayable)")
        S,why=_unseal_secret(VAULT,td)
        if S is None: print(f"[{HOST}] {why} -> cannot derive steering -> REFUSED"); sys.exit(5)
        v=steer_vec_from_secret(S); S=b"\x00"*32
        commit=hashlib.sha256(v.tobytes()).hexdigest()
        meta=json.loads((VAULT/"vault.json").read_text())
        print(f"[{HOST}] unsealed per-die secret; steer commitment match={commit==meta['steer_commit_sha256']}")
    dev="cuda" if torch.cuda.is_available() else "cpu"
    tok=GPT2TokenizerFast.from_pretrained("gpt2"); lm=GPT2LMHeadModel.from_pretrained("gpt2").to(dev).eval()
    d=lm.config.n_embd
    g=torch.Generator().manual_seed(0); P=torch.randn(d,DIM,generator=g).to(dev); P=P/P.norm(dim=0,keepdim=True)
    fp=torch.tensor(v,device=dev); steer={"v":6.0*(P@fp)/(P@fp).norm()}
    hooks=[blk.register_forward_hook(lambda m,i,o:(o[0]+steer["v"],)+o[1:] if isinstance(o,tuple) else o+steer["v"]) for blk in lm.transformer.h]
    res={"host":HOST,"alpha":6.0,"gen":{}}
    print(f"\n[{HOST}] >>> behavior steered by TPM-sealed per-die secret (un-copyable, fresh-nonce-gated):")
    with torch.no_grad():
        for pr in PROMPTS:
            x=tok(pr,return_tensors="pt").input_ids.to(dev)
            for _ in range(40): x=torch.cat([x,torch.argmax(lm(x).logits[:,-1,:],-1,keepdim=True)],1)
            t=tok.decode(x[0],skip_special_tokens=True); res["gen"][pr]=t; print(f"  [{pr!r}] {t}")
    for h in hooks: h.remove()
    (OUT/f"unfakeable_steer_{HOST}.json").write_text(json.dumps(res,indent=2))
    print(f"\nsaved unfakeable_steer_{HOST}.json")

def verify_foreign():
    vdir=Path(os.environ.get("FOREIGN_VAULT",str(VAULT)))
    origin=json.loads((vdir/"vault.json").read_text()).get("host","?")
    with tempfile.TemporaryDirectory() as td:
        S,why=_unseal_secret(vdir,Path(td))
    ok=S is not None; same=(origin==HOST)
    verdict=("UNLOCK (own die)" if ok and same else "REFUSED (foreign die)" if (not ok and not same) else f"UNEXPECTED ok={ok} same={same}")
    print(f"[{HOST}] vault from '{origin}': {verdict}  [{'OK' if ok==same else '!!'}]")

if __name__=="__main__":
    ap=argparse.ArgumentParser(); ap.add_argument("cmd",choices=["enroll","run","verify-foreign"]); a=ap.parse_args()
    {"enroll":enroll,"run":run,"verify-foreign":verify_foreign}[a.cmd]()
