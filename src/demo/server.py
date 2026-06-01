"""FabricCrypt demo server.

POST /challenge   -> issues a fresh nonce
POST /sign        -> chip signs the nonce (this host's NonceSig.read())
POST /verify      -> deterministic plan-consistency check (+ optional classifier)

Default port 8770. Static page at / shows a sign-and-verify button.

Run:
    python -m src.demo.server --host 127.0.0.1 --port 8770

Optional: --t3_pt data/<host>_t3_best.pt to enable the classifier head.
"""
from __future__ import annotations
import os
import sys
import time
import argparse
import secrets
import numpy as np

try:
    from fastapi import FastAPI, HTTPException
    from fastapi.staticfiles import StaticFiles
    from fastapi.responses import FileResponse
    import uvicorn
except ImportError:
    print("Demo requires fastapi + uvicorn. pip install -r requirements.txt",
          file=sys.stderr)
    raise

# Allow running as `python src/demo/server.py` or as a module.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from src.protocol.nonce_signature import NonceSig
from src.protocol.nonce_derivation import nonce_embedding, fresh_nonce
from src.protocol.verifier import plan_consistency_score
from src.demo.models import (ChallengeRequest, SignResponse,
                              VerifyRequest, VerifyResponse)


HERE = os.path.dirname(os.path.abspath(__file__))
STATIC = os.path.join(HERE, "static")

app = FastAPI(title="FabricCrypt demo")

_state = {"sig": None, "model": None, "device": "cpu"}


@app.get("/")
def index():
    return FileResponse(os.path.join(STATIC, "index.html"))


@app.post("/challenge")
def challenge(req: ChallengeRequest):
    nonce = bytes.fromhex(req.nonce_hex) if req.nonce_hex else secrets.token_bytes(8)
    return {"nonce_hex": nonce.hex()}


@app.post("/sign", response_model=SignResponse)
def sign(req: ChallengeRequest):
    if _state["sig"] is None:
        raise HTTPException(503, "NonceSig not initialised")
    nonce = bytes.fromhex(req.nonce_hex) if req.nonce_hex else secrets.token_bytes(8)
    t0 = time.perf_counter()
    v = _state["sig"].read(nonce, raw=True)
    dt = (time.perf_counter() - t0) * 1000
    return SignResponse(
        nonce_hex=nonce.hex(),
        sig=v.tolist(),
        host=_state["sig"].host,
        elapsed_ms=dt,
    )


@app.post("/verify", response_model=VerifyResponse)
def verify(req: VerifyRequest):
    if _state["sig"] is None:
        raise HTTPException(503, "NonceSig not initialised")
    nonce = bytes.fromhex(req.nonce_hex)
    v = np.asarray(req.sig, dtype=np.float32)
    if v.shape != (64,):
        raise HTTPException(400, f"sig must be 64-dim, got {v.shape}")
    t0 = time.perf_counter()
    ps = plan_consistency_score(v[:32], nonce,
                                 _state["sig"].n_cpus, _state["sig"].n_zones)
    cls_p0 = None
    if _state["model"] is not None:
        import torch, torch.nn.functional as F
        with torch.no_grad():
            logits = _state["model"](torch.from_numpy(v[None, :]))
            cls_p0 = float(F.softmax(logits, dim=-1)[0, 0].item())
    dt = (time.perf_counter() - t0) * 1000
    return VerifyResponse(
        accept=bool(ps > 0.5),
        plan_score=float(ps),
        plan_thresh=0.5,
        classifier_p0=cls_p0,
        elapsed_ms=dt,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8770)
    ap.add_argument("--t3_pt", default=None,
                    help="optional trained T3 classifier state_dict")
    args = ap.parse_args()

    print("[demo] initialising NonceSig (this may calibrate on first run)...")
    _state["sig"] = NonceSig(calibrate=True)
    if args.t3_pt and os.path.exists(args.t3_pt):
        import torch
        from src.protocol.classifier import TwinMLP, DIM
        m = TwinMLP(in_d=DIM, n_out=2)
        m.load_state_dict(torch.load(args.t3_pt, map_location="cpu"))
        m.eval()
        _state["model"] = m
        print(f"[demo] loaded classifier {args.t3_pt}")

    app.mount("/static", StaticFiles(directory=STATIC), name="static")
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
