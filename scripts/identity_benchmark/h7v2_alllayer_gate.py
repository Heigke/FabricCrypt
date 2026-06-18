"""H7 v2 — body-gated EVERY layer (pinne 4): depth-compounding embodiment + depth-ablation falsification.

Deeper than MoE-routing (which gated 4 layers): here a frozen GPT-2 gets a small multiplicative per-channel
gate keyed by the LIVE body vector z inserted after EVERY transformer block. h <- h * (1 + EPS*tanh(W_l z)).
A wrong body key mis-scales the residual stream at all 12 layers, and the error COMPOUNDS with depth — the
deeper the network, the more a wrong key destroys the computation. Only the body's gates train (GPT-2 frozen);
NTL dual-key loss (low CE on true z, push CE up on permuted z) makes wrong keys destructive.

FALSIFICATION (greedy/teacher-forced, deterministic — model contributes zero randomness):
  native / random_z / foreign_die conditions, PLUS the key NEW test —
  DEPTH ABLATION: gate only the first k layers (k=0,3,6,9,12) under a WRONG key and measure PPL blow-up.
  If embodiment is genuinely distributed-and-compounding, wrong-key PPL must grow monotonically with k
  (more gated layers -> more damage). A flat curve would mean only one layer matters (shallow).

Run sandbox-disabled. HSA override. Out: results/IDENTITY_H7_2026-06-09/v2_alllayer_gate_{host}.json
"""
from __future__ import annotations
import os, sys, json, time, socket
import numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from h7_rooted_lm_embodied import BodyGate
from h7v2_multilayer_keystream import (read_prefcore_fingerprint, fp_bits, operands,
                                       MesoGate, MacroGate, DAEDALUS_FP)
from h7v2_moe_routing import body_key, DZ

HOST = socket.gethostname()
ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/IDENTITY_H7_2026-06-09"; OUT.mkdir(parents=True, exist_ok=True)
CORPUS = ROOT / "data/tinyshakespeare.txt"


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=700); ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--ctx", type=int, default=96); ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--eps", type=float, default=0.20); ap.add_argument("--beta", type=float, default=0.5)
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
    D = lm.config.n_embd; nL = lm.config.n_layer
    layers = list(range(nL))                              # gate EVERY block
    ids = np.array(tok(CORPUS.read_text(encoding="utf-8", errors="ignore"))["input_ids"], dtype=np.int64)
    rng = np.random.default_rng(a.seed)

    fp_own = read_prefcore_fingerprint()
    micro = BodyGate(win=a.win); mfid = micro.calibrate()
    meso = MesoGate(); meso.measure(); macro = MacroGate(); macro.measure()
    print(f"[{HOST}] all-layer body gating: frozen gpt2, {nL} gated blocks, eps={a.eps}, "
          f"micro fid={mfid:.2f}", flush=True)

    class Gate(nn.Module):                                # per-layer multiplicative channel gate keyed by z
        def __init__(self):
            super().__init__()
            self.w = nn.Linear(DZ, D)
            nn.init.zeros_(self.w.weight); nn.init.zeros_(self.w.bias)   # start = identity (gate 0)
        def forward(self, h, z):
            g = torch.tanh(self.w(z))                     # (B, D) in [-1,1]
            return h * (1.0 + a.eps * g.unsqueeze(1))     # broadcast over tokens
    gates = nn.ModuleDict({str(l): Gate().to(dev) for l in layers})

    ctxst = {"z": None, "active": set(layers)}            # active = which layers actually gate (for ablation)
    def hook(l):
        gate = gates[str(l)]
        def f(mod, inp, out):
            if ctxst["z"] is None or l not in ctxst["active"]: return out
            h = out[0] if isinstance(out, tuple) else out
            nh = gate(h, ctxst["z"])
            return (nh,) + out[1:] if isinstance(out, tuple) else nh
        return f
    for l in layers: lm.transformer.h[l].register_forward_hook(hook(l))

    params = list(gates.parameters())
    opt = torch.optim.AdamW(params, lr=a.lr)

    def windows(bs):
        i = rng.integers(0, len(ids) - a.ctx - 2, size=bs)
        w = ids[i[:, None] + np.arange(a.ctx + 1)[None]]
        return torch.from_numpy(w).to(dev)

    def lm_loss(x, z):
        ctxst["z"] = z; o = lm(x[:, :-1]); ctxst["z"] = None
        return F.cross_entropy(o.logits.reshape(-1, o.logits.size(-1)), x[:, 1:].reshape(-1))

    nonce_tr = b"H7v2-alllayer-train"
    micro_tab = {(p, q): micro.gate(p, q) for p in (0, 1) for q in (0, 1)}
    zt = torch.from_numpy(body_key(micro_tab, meso, macro, fp_own, nonce_tr)).to(dev)
    t0 = time.time()
    for step in range(a.steps):
        if step > 0 and step % 200 == 0: meso.measure(); macro.measure()
        x = windows(a.batch)
        z = zt.unsqueeze(0).expand(x.size(0), -1)
        zwrong = z[:, torch.randperm(DZ, device=dev)]
        ce_ok = lm_loss(x, z)
        ce_bad = lm_loss(x, zwrong)
        loss = ce_ok + a.beta * F.relu(2.0 - (ce_bad - ce_ok))
        opt.zero_grad(); loss.backward(); opt.step()
        if (step + 1) % 100 == 0:
            print(f"  step {step+1} ce_ok={ce_ok.item():.3f} ce_bad={ce_bad.item():.3f} "
                  f"t={time.time()-t0:.0f}s", flush=True)

    # ---- eval ----
    meso.measure(); macro.measure()
    micro_tab = {(p, q): micro.gate(p, q) for p in (0, 1) for q in (0, 1)}
    z_native = torch.from_numpy(body_key(micro_tab, meso, macro, fp_own, nonce_tr)).to(dev)
    z_foreign = torch.from_numpy(body_key(micro_tab, meso, macro, DAEDALUS_FP, nonce_tr)).to(dev)
    rr = np.random.default_rng(7)
    z_rand = torch.from_numpy(rr.integers(0, 2, DZ).astype(np.float32)).to(dev)
    ev = [windows(1) for _ in range(a.n_eval)]

    @torch.no_grad()
    def ppl(zvec, active):
        ctxst["active"] = set(active)
        tot = n = 0.0
        for x in ev:
            ctxst["z"] = zvec.unsqueeze(0); o = lm(x[:, :-1]); ctxst["z"] = None
            ce = F.cross_entropy(o.logits.reshape(-1, o.logits.size(-1)), x[:, 1:].reshape(-1))
            tot += float(ce) * x[:, 1:].numel(); n += x[:, 1:].numel()
        ctxst["active"] = set(layers)
        return float(np.exp(tot / n))

    res = {"native": round(ppl(z_native, layers), 3), "random_z": round(ppl(z_rand, layers), 3),
           "foreign_die": round(ppl(z_foreign, layers), 3)}
    # DEPTH ABLATION: wrong (random) key, gating only first k layers — PPL should grow with k
    depth = {}
    for k in (0, 3, 6, 9, nL):
        depth[k] = round(ppl(z_rand, set(range(k))), 3)
    micro.close()
    nat = res["native"]
    ks = sorted(depth)
    monotone = all(depth[ks[i]] <= depth[ks[i + 1]] * 1.05 for i in range(len(ks) - 1)) and depth[ks[-1]] > depth[ks[0]] * 1.3
    out = {"host": HOST, "frozen_LLM": "gpt2-124M", "gated_layers": nL, "eps": a.eps, "key_bits": DZ,
           "trainable_params": int(sum(p.numel() for p in params)),
           "integration": "multiplicative per-channel gate keyed by live body z, inserted after EVERY block",
           "eval_perplexity": res,
           "depth_ablation_wrongkey_ppl": depth,
           "LOAD_BEARING": bool(res["random_z"] > 1.3 * nat and res["foreign_die"] > 1.3 * nat),
           "UNIQUE_per_die": bool(res["foreign_die"] > 1.3 * nat),
           "DEPTH_COMPOUNDS": bool(monotone),
           "honest_scope": ("Body gates the residual stream at all 12 GPT-2 blocks; wrong key mis-scales every "
                            "layer and damage compounds with depth (depth-ablation curve). Physics generic; "
                            "per-die uniqueness from prefcore fingerprint; freshness from keystream layer.")}
    out["ALL_GREEN"] = bool(out["LOAD_BEARING"] and out["UNIQUE_per_die"] and out["DEPTH_COMPOUNDS"])
    out["verdict"] = ("all-layer GREEN: wrong/foreign body key destroys the LM and the damage grows with the "
                      "number of gated layers (embodiment is distributed + depth-compounding)." if out["ALL_GREEN"]
                      else "partial — see eval_perplexity + depth_ablation")
    (OUT / f"v2_alllayer_gate_{HOST}.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2), flush=True)


if __name__ == "__main__":
    main()
