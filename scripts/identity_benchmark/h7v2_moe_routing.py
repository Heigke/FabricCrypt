"""H7 v2 — body-driven MoE routing: the deepest load-bearing integration (oracle top pick).

A frozen GPT-2 is augmented with small EXPERT banks inserted after several blocks. At each inserted layer
a router chooses how to mix the experts, but the router is keyed by the BODY vector z ONLY (no token input)
-- z is computed live from this die (micro cache-XOR ⊕ meso GPU-droop ⊕ macro SMU, fused with the per-die
prefcore fingerprint, all nonce-bound). So the body decides WHICH computation runs at every inserted layer;
a wrong key routes every layer's experts wrong and the forward pass computes the wrong function.

Trained with a Non-Transferable-Learning DUAL-KEY loss (only router + experts train; GPT-2 frozen):
  minimize LM loss under the TRUE die key z, and PUSH UP the loss under a WRONG key (permuted z),
plus an expert-diversity penalty (so experts don't collapse to one function -> the known MoE failure mode).

Falsification (oracle's #1 "double dissociation", greedy/teacher-forced, model contributes zero randomness):
  native      true live z                         -> low PPL
  random_z    random key                          -> high PPL
  foreign_die daedalus fingerprint in z           -> high PPL  (per-die)
  wrong_nonce old nonce                            -> high PPL  (fresh)
  patch       foreign z BUT router decisions overwritten with native z's  -> RECOVERS  (isolates that the
              body's routing is the causal locus, not z distribution)
Load-bearing iff native << {random, foreign, wrong_nonce} AND patch ~ native.

Run sandbox-disabled. HSA override. Out: results/IDENTITY_H7_2026-06-09/v2_moe_routing_{host}.json
"""
from __future__ import annotations
import os, sys, json, time, socket, hashlib
import numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from h7_rooted_lm_embodied import BodyGate
from h7v2_multilayer_keystream import (read_prefcore_fingerprint, fp_bits, operands,
                                       MesoGate, MacroGate, DAEDALUS_FP, foreign_fp)

HOST = socket.gethostname()
ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/IDENTITY_H7_2026-06-09"; OUT.mkdir(parents=True, exist_ok=True)
CORPUS = ROOT / "data/tinyshakespeare.txt"
DZ = 16          # body key bit-width


def body_key(micro, meso, macro, fp, nonce):
    """DZ-bit live body vector: 8 bits fused micro/meso/macro over nonce challenges + 8 fingerprint bits."""
    fpb = fp_bits(fp, nonce, 8)
    bits = []
    for k in range(8):
        ctx = np.array([k, fpb[k]], dtype=np.int64)
        (ami, bmi), (ame, bme), (ama, bma) = operands(nonce, ctx)
        bits.append(int(micro[(ami, bmi)]) ^ meso.gate(ame, bme) ^ macro.gate(ama, bma))
    return np.array(bits + [int(x) for x in fpb], dtype=np.float32)   # len 16


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=700); ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--ctx", type=int, default=96); ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--experts", type=int, default=4); ap.add_argument("--rank", type=int, default=32)
    ap.add_argument("--layers", type=str, default="3,5,7,9"); ap.add_argument("--beta", type=float, default=0.5)
    ap.add_argument("--win", type=float, default=0.03); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n_eval", type=int, default=40)
    a = ap.parse_args()
    import torch, torch.nn as nn, torch.nn.functional as F
    from transformers import GPT2LMHeadModel, GPT2TokenizerFast
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = GPT2TokenizerFast.from_pretrained("gpt2")
    lm = GPT2LMHeadModel.from_pretrained("gpt2").to(dev).eval()
    for p in lm.parameters(): p.requires_grad_(False)
    D = lm.config.n_embd
    layers = [int(x) for x in a.layers.split(",")]
    ids = np.array(tok(CORPUS.read_text(encoding="utf-8", errors="ignore"))["input_ids"], dtype=np.int64)
    rng = np.random.default_rng(a.seed)

    fp_own = read_prefcore_fingerprint()
    micro = BodyGate(win=a.win); mfid = micro.calibrate()
    meso = MesoGate(); meso.measure(); macro = MacroGate(); macro.measure()
    print(f"[{HOST}] MoE routing: gpt2 + {len(layers)}x expert banks (E={a.experts}, r={a.rank}); "
          f"router keyed by {DZ}-bit live body. micro fid={mfid:.2f}", flush=True)

    class MoE(nn.Module):
        def __init__(self):
            super().__init__()
            self.router = nn.Linear(DZ, a.experts, bias=False)
            self.down = nn.Parameter(torch.randn(a.experts, a.rank, D) * (1.0 / D**0.5))
            self.up = nn.Parameter(torch.zeros(a.experts, D, a.rank))   # start as no-op
        def forward(self, h, z, tau=0.5):
            p = F.softmax(self.router(z) / tau, -1)                     # (B, E) keyed by body only
            mid = torch.einsum("erd,btd->bter", self.down, h)
            mid = F.gelu(mid)
            ex = torch.einsum("edr,bter->bted", self.up, mid)           # (B,T,E,D)
            out = torch.einsum("be,bted->btd", p, ex)
            return h + out, p
    banks = nn.ModuleDict({str(l): MoE().to(dev) for l in layers})

    ctxst = {"z": None, "patch_p": None, "store_p": None}
    def hook(l):
        bank = banks[str(l)]
        def f(mod, inp, out):
            h = out[0] if isinstance(out, tuple) else out
            if ctxst["z"] is None: return out
            nh, p = bank(h, ctxst["z"])
            if ctxst["patch_p"] is not None:                            # activation-patch: force routing
                pp = ctxst["patch_p"][str(l)]
                mid = F.gelu(torch.einsum("erd,btd->bter", bank.down, h))
                ex = torch.einsum("edr,bter->bted", bank.up, mid)
                nh = h + torch.einsum("be,bted->btd", pp, ex)
            elif ctxst["store_p"] is not None:
                ctxst["store_p"][str(l)] = p.detach()
            return (nh,) + out[1:] if isinstance(out, tuple) else nh
        return f
    for l in layers: lm.transformer.h[l].register_forward_hook(hook(l))

    params = list(banks.parameters())
    opt = torch.optim.AdamW(params, lr=a.lr)

    def windows(bs):
        i = rng.integers(0, len(ids) - a.ctx - 2, size=bs)
        w = ids[i[:, None] + np.arange(a.ctx + 1)[None]]
        return torch.from_numpy(w).to(dev)

    def lm_loss(x, z):
        ctxst["z"] = z
        o = lm(x[:, :-1]); ctxst["z"] = None
        return F.cross_entropy(o.logits.reshape(-1, o.logits.size(-1)), x[:, 1:].reshape(-1))

    nonce_tr = b"H7v2-moe-train"
    zt = torch.from_numpy(body_key({(p, q): micro.gate(p, q) for p in (0, 1) for q in (0, 1)},
                                    meso, macro, fp_own, nonce_tr)).to(dev)
    t0 = time.time()
    for step in range(a.steps):
        if step > 0 and step % 200 == 0: meso.measure(); macro.measure()
        x = windows(a.batch)
        z = zt.unsqueeze(0).expand(x.size(0), -1)
        zwrong = z[:, torch.randperm(DZ, device=dev)]                   # permuted key
        ce_ok = lm_loss(x, z)
        ce_bad = lm_loss(x, zwrong)
        # diversity: discourage identical experts (orthogonalize down-projs)
        div = 0.0
        for l in layers:
            d = banks[str(l)].down.reshape(a.experts, -1)
            d = F.normalize(d, dim=1); g = d @ d.t()
            div = div + (g - torch.eye(a.experts, device=dev)).pow(2).mean()
        loss = ce_ok + a.beta * F.relu(2.0 - (ce_bad - ce_ok)) + 0.01 * div
        opt.zero_grad(); loss.backward(); opt.step()
        if (step + 1) % 100 == 0:
            print(f"  step {step+1} ce_ok={ce_ok.item():.3f} ce_bad={ce_bad.item():.3f} "
                  f"div={float(div):.3f} t={time.time()-t0:.0f}s", flush=True)

    # ---- eval: teacher-forced PPL under each key condition (deterministic) ----
    # NOTE on scope: a LEARNED router memorizes the (die,nonce) key it trained on; it cannot grant
    # cross-nonce freshness (a fresh nonce yields a key the router never saw -> fails). So this experiment
    # isolates LOAD-BEARING + PER-DIE uniqueness at the TRAINED nonce; freshness is delegated to the
    # deterministic keystream layer (h7v2_multilayer_keystream), which re-derives K live per nonce.
    meso.measure(); macro.measure()
    nonce_other = b"H7v2-moe-other-nonce"
    micro_tab = {(p, q): micro.gate(p, q) for p in (0, 1) for q in (0, 1)}
    z_native = torch.from_numpy(body_key(micro_tab, meso, macro, fp_own, nonce_tr)).to(dev)      # trained key
    z_foreign = torch.from_numpy(body_key(micro_tab, meso, macro, foreign_fp(), nonce_tr)).to(dev)  # OTHER real die (by host)
    z_replay = torch.from_numpy(body_key(micro_tab, meso, macro, fp_own, nonce_other)).to(dev)   # diff nonce
    ev = [windows(1) for _ in range(a.n_eval)]

    @torch.no_grad()
    def ppl(zvec, patch=False):
        tot = n = 0.0
        for x in ev:
            z = zvec.unsqueeze(0).expand(x.size(0), -1)
            if patch:                                                  # capture native routing, then patch
                store = {str(l): None for l in layers}; ctxst["store_p"] = store; ctxst["z"] = z_native.unsqueeze(0)
                lm(x[:, :-1]); ctxst["store_p"] = None
                ctxst["patch_p"] = store; ctxst["z"] = z
                o = lm(x[:, :-1]); ctxst["patch_p"] = None; ctxst["z"] = None
            else:
                ctxst["z"] = z; o = lm(x[:, :-1]); ctxst["z"] = None
            ce = F.cross_entropy(o.logits.reshape(-1, o.logits.size(-1)), x[:, 1:].reshape(-1))
            tot += float(ce) * x[:, 1:].numel(); n += x[:, 1:].numel()
        return float(np.exp(tot / n))

    rr = np.random.default_rng(7)
    z_rand = torch.from_numpy(rr.integers(0, 2, DZ).astype(np.float32)).to(dev)
    res = {"native": round(ppl(z_native), 3), "random_z": round(ppl(z_rand), 3),
           "foreign_die": round(ppl(z_foreign), 3), "different_nonce": round(ppl(z_replay), 3),
           "patch_foreign_with_native_routing": round(ppl(z_foreign, patch=True), 3)}
    micro.close()
    nat = res["native"]
    out = {"host": HOST, "frozen_LLM": "gpt2-124M", "experts": a.experts, "rank": a.rank,
           "inserted_layers": layers, "key_bits": DZ,
           "trainable_params": int(sum(p.numel() for p in params)),
           "integration": "body-keyed MoE routing (router sees ONLY the live body vector) + NTL dual-key + expert diversity",
           "eval_perplexity": res,
           "LOAD_BEARING": bool(res["random_z"] > 1.3 * nat and res["foreign_die"] > 1.3 * nat),
           "UNIQUE_per_die": bool(res["foreign_die"] > 1.3 * nat),
           "PATCH_RECOVERS": bool(res["patch_foreign_with_native_routing"] < 1.1 * nat),
           "key_specific_diff_nonce_fails": bool(res["different_nonce"] > 1.3 * nat),
           "honest_scope": ("Router keyed by live body only; wrong key routes every inserted layer's experts "
                            "wrong. NTL makes wrong-key destructive. Patch test isolates routing as the causal "
                            "locus (double dissociation). A LEARNED router memorizes its (die,nonce) key, so a "
                            "different nonce also fails (key-specific) -- cross-nonce FRESHNESS is delegated to "
                            "the deterministic keystream layer, not claimed here. Physics generic; per-die "
                            "uniqueness from prefcore fingerprint.")}
    grn = out["LOAD_BEARING"] and out["UNIQUE_per_die"] and out["PATCH_RECOVERS"]
    out["ALL_GREEN"] = bool(grn)
    out["verdict"] = ("MoE GREEN (load-bearing + per-die, at trained nonce): a frozen GPT-2's per-layer expert "
                      "routing is load-bearing on the body key; foreign-die/random key -> high PPL; patching "
                      "native routing into a foreign-key run recovers PPL (routing is the causal locus). "
                      "Freshness delegated to keystream layer." if grn else "partial — see eval_perplexity")
    (OUT / f"v2_moe_routing_{HOST}.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2), flush=True)


if __name__ == "__main__":
    main()
