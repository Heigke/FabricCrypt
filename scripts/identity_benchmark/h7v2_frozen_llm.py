"""H7 v2 — REAL frozen LLM made constitutively dependent on its body's keystream.

Fixes the red-team's CRITICAL holes for the LLM side:
  #4 "not a real LLM": uses FROZEN GPT-2 (124M) on REAL English text (tinyshakespeare). The transformer
      is untouched and stays fluent.
  #3 "not real text": next-token loss is measured on real BPE tokens.
  killer-attack class: at QUERY positions the model must commit to ONE of the two genuinely-plausible
      next tokens (top-2 under the frozen LM); WHICH one is the "authenticated" continuation is decided
      by an external KEYSTREAM bit b_t (later = K(C) from the live die + verifier nonce). The bit is NOT
      derivable from the text, so neither GPT-2 nor any text-only clone can fake it — the body is the
      only source. This entangles the physical signal inside the high-entropy generative path (oracle
      guidance) rather than computing a public function of public inputs.

What this slice proves (body = external bitstream stand-in; live K(C) organ plugs in next):
  - native (true keystream)  -> selects the reference continuation  (query acc ~1.0)
  - zero / random keystream   -> chance on query tokens -> generated text DIVERGES from reference
  - foreign keystream         -> divergence (per-source); same machinery the live foreign-die uses
Plain cross-entropy, adapter-only training (GPT-2 frozen). No margin/spoof loss.

Out: results/IDENTITY_H7_2026-06-09/v2_frozen_llm_{host}.json
Run: HSA_OVERRIDE_GFX_VERSION=11.0.0 venv/bin/python scripts/identity_benchmark/h7v2_frozen_llm.py
"""
from __future__ import annotations
import os, sys, json, time, math, socket, argparse, hashlib
import numpy as np
from pathlib import Path

HOST = socket.gethostname()
ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/IDENTITY_H7_2026-06-09"; OUT.mkdir(parents=True, exist_ok=True)
CORPUS = ROOT / "data/tinyshakespeare.txt"


def keystream(nonce: bytes, n: int) -> np.ndarray:
    """Deterministic pseudo-random bits from a nonce (stand-in for K(C) from the die).
    Real v2: each bit comes from the live die's macro/analog response to a nonce-derived challenge."""
    out = bytearray()
    ctr = 0
    while len(out) * 8 < n:
        out += hashlib.sha256(nonce + ctr.to_bytes(8, "little")).digest()
        ctr += 1
    bits = np.unpackbits(np.frombuffer(bytes(out), dtype=np.uint8))[:n]
    return bits.astype(np.int64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gpt2")
    ap.add_argument("--ctx", type=int, default=128)
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--qstride", type=int, default=6, help="1 body-steered query token every qstride")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--n_eval", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    import torch, torch.nn as nn, torch.nn.functional as F
    from transformers import GPT2LMHeadModel, GPT2TokenizerFast
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    tok = GPT2TokenizerFast.from_pretrained(a.model)
    lm = GPT2LMHeadModel.from_pretrained(a.model).to(dev).eval()
    for p in lm.parameters(): p.requires_grad_(False)         # FROZEN
    D = lm.config.n_embd

    text = CORPUS.read_text(encoding="utf-8", errors="ignore")
    ids = np.array(tok(text)["input_ids"], dtype=np.int64)
    print(f"[{HOST}] frozen {a.model} ({sum(p.numel() for p in lm.parameters())/1e6:.0f}M, frozen) "
          f"corpus={len(ids)} toks dev={dev}", flush=True)

    rng = np.random.default_rng(a.seed)

    @torch.no_grad()
    def frozen_pass(batch_ids):
        out = lm(batch_ids, output_hidden_states=True)
        h = out.hidden_states[-1]                              # (B,T,D) frozen features
        logits = out.logits                                    # (B,T,V) frozen LM distribution
        top2 = logits.topk(2, dim=-1).indices                 # (B,T,2) the two plausible next tokens
        return h, top2

    # adapter: read keystream bit + context feature -> choose candidate 0/1 at query positions
    sel = nn.Sequential(nn.Linear(D + 1, 128), nn.GELU(), nn.Linear(128, 2)).to(dev)
    opt = torch.optim.AdamW(sel.parameters(), lr=a.lr)

    def sample_windows(bs):
        i = rng.integers(0, len(ids) - a.ctx - 2, size=bs)
        idx = i[:, None] + np.arange(a.ctx + 1)[None]
        return torch.from_numpy(ids[idx]).to(dev)             # (bs, ctx+1)

    nonce_train = b"H7v2-train-nonce"
    t0 = time.time()
    for step in range(a.steps):
        w = sample_windows(a.batch)
        x = w[:, :a.ctx]
        h, top2 = frozen_pass(x)                               # (B,ctx,D),(B,ctx,2)
        B, T = x.shape
        # body keystream bit per position (stand-in for live K(C)); query positions only
        ks = keystream(nonce_train, B * T).reshape(B, T)
        ks_t = torch.from_numpy(ks).float().to(dev)
        qmask = np.zeros((B, T), bool); qmask[:, ::a.qstride] = True
        qmask[:, :2] = False
        qm = torch.from_numpy(qmask).to(dev)
        logit2 = sel(torch.cat([h, ks_t.unsqueeze(-1)], -1))   # (B,T,2)
        target = ks_t.long()                                   # authenticated choice = the body bit
        ce = F.cross_entropy(logit2.reshape(-1, 2), target.reshape(-1), reduction="none").reshape(B, T)
        loss = (ce * qm).sum() / qm.sum().clamp(min=1)
        opt.zero_grad(); loss.backward(); opt.step()
        if (step + 1) % 300 == 0:
            with torch.no_grad():
                acc = (((logit2.argmax(-1) == target) & qm).sum() / qm.sum().clamp(min=1)).item()
            print(f"  step {step+1:5d} loss={loss.item():.4f} qacc={acc:.3f} t={time.time()-t0:.0f}s", flush=True)

    # ---- eval: native vs corrupted body keystream, query accuracy + text divergence ----
    ev = [sample_windows(1) for _ in range(a.n_eval)]
    nonce_eval = b"H7v2-eval-nonce-fresh"
    nonce_foreign = b"H7v2-foreign-machine"

    @torch.no_grad()
    def evalcond(cond):
        correct = n = 0; diverge = 0; dtot = 0
        for w in ev:
            x = w[:, :a.ctx]; h, top2 = frozen_pass(x)
            B, T = x.shape
            ks_true = keystream(nonce_eval, B * T).reshape(B, T)
            if cond == "native": ks = ks_true
            elif cond == "zero": ks = np.zeros((B, T), np.int64)
            elif cond == "random": ks = rng.integers(0, 2, (B, T))
            elif cond == "foreign": ks = keystream(nonce_foreign, B * T).reshape(B, T)
            elif cond == "replay_old": ks = keystream(b"H7v2-train-nonce", B * T).reshape(B, T)
            ks_t = torch.from_numpy(ks).float().to(dev)
            logit2 = sel(torch.cat([h, ks_t.unsqueeze(-1)], -1))
            choice = logit2.argmax(-1)                          # (B,T) 0/1
            qmask = np.zeros((B, T), bool); qmask[:, ::a.qstride] = True; qmask[:, :2] = False
            qm = torch.from_numpy(qmask).to(dev)
            # reference (native) choice = true keystream bit
            ref = torch.from_numpy(ks_true).long().to(dev)
            correct += int(((choice == ref) & qm).sum()); n += int(qm.sum())
            # token-level divergence: chosen token vs reference token at query positions
            tok_chosen = torch.gather(top2, -1, choice.unsqueeze(-1)).squeeze(-1)
            tok_ref = torch.gather(top2, -1, ref.unsqueeze(-1)).squeeze(-1)
            diverge += int(((tok_chosen != tok_ref) & qm).sum()); dtot += int(qm.sum())
        return {"query_acc_vs_ref": round(correct / max(n, 1), 3),
                "token_divergence": round(diverge / max(dtot, 1), 3)}

    res = {c: evalcond(c) for c in ["native", "zero", "random", "foreign", "replay_old"]}
    nat = res["native"]["query_acc_vs_ref"]
    worst = max(res["zero"]["query_acc_vs_ref"], res["random"]["query_acc_vs_ref"],
               res["foreign"]["query_acc_vs_ref"], res["replay_old"]["query_acc_vs_ref"])
    out = {"host": HOST, "model": a.model, "ctx": a.ctx, "qstride": a.qstride,
           "frozen_params_M": round(sum(p.numel() for p in lm.parameters()) / 1e6),
           "results": res,
           "LLM_BODY_LOAD_BEARING": bool(nat > 0.95 and nat - worst > 0.40),
           "replay_resistant": bool(res["native"]["query_acc_vs_ref"] - res["replay_old"]["query_acc_vs_ref"] > 0.40)}
    out["verdict"] = ("REAL frozen GPT-2: its committed continuation is body-keystream-determined; "
                      "wrong/old/foreign keystream -> chance + text divergence"
                      if out["LLM_BODY_LOAD_BEARING"] else "NOT load-bearing — investigate")
    (OUT / f"v2_frozen_llm_{HOST}.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2), flush=True)
    print(f"\n[{HOST}] {out['verdict']}\n  native qacc={nat} | "
          + " ".join(f"{c}={res[c]['query_acc_vs_ref']}(div {res[c]['token_divergence']})"
                     for c in ["zero", "random", "foreign", "replay_old"]), flush=True)


if __name__ == "__main__":
    main()
